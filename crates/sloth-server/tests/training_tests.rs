//! Обучение через HTTP: честный отказ без движка и полный жизненный цикл запуска на
//! учебном движке (настоящий движок LoRA на Vulkan появится на этапе 3).

use futures_util::StreamExt;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use sloth_server::create_router;
use sloth_server::state::AppState;
use sloth_server::training::{
    EngineFuture, ProgressReporter, ProgressUpdate, StopSignal, TrainingConfig, TrainingController,
    TrainingEngine, TrainingPhase,
};
use std::sync::Arc;
use std::time::Duration;

const SSE_TIMEOUT: Duration = Duration::from_secs(15);
const POLL_INTERVAL: Duration = Duration::from_millis(20);
const POLL_ATTEMPTS: usize = 500;
const UNLIMITED_STEPS: u32 = 0;

/// Учебный движок: `steps` шагов с паузой `delay`, ошибка на шаге `fail_at`.
struct StepEngine {
    steps: u32,
    delay: Duration,
    fail_at: Option<u32>,
}

impl TrainingEngine for StepEngine {
    fn train(
        &self,
        _config: TrainingConfig,
        reporter: ProgressReporter,
        mut stop: StopSignal,
    ) -> EngineFuture {
        let (steps, delay, fail_at) = (self.steps, self.delay, self.fail_at);
        Box::pin(async move {
            reporter.set_phase(TrainingPhase::LoadingModel, "Загрузка модели");
            for step in 1..=steps {
                if fail_at == Some(step) {
                    return Err("Не хватило видеопамяти".to_string());
                }
                tokio::select! {
                    _ = tokio::time::sleep(delay) => {}
                    _ = stop.stopped() => return Ok(()),
                }
                reporter.report(ProgressUpdate {
                    step,
                    total_steps: steps,
                    epoch: Some(1),
                    loss: Some(2.0 / step as f32),
                    learning_rate: Some(2e-4),
                    ..ProgressUpdate::default()
                });
            }
            Ok(())
        })
    }
}

async fn spawn(engine: Option<StepEngine>) -> String {
    let mut state = AppState::new(None, std::env::temp_dir(), None);
    if let Some(engine) = engine {
        let engine: Arc<dyn TrainingEngine> = Arc::new(engine);
        state.training = Arc::new(TrainingController::new(
            Arc::clone(&state.store),
            Some(engine),
        ));
    }
    let app = create_router(Arc::new(state), None);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("тестовый сервер упал");
    });
    format!("http://{addr}")
}

async fn call(base: &str, method: Method, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let mut request = reqwest::Client::new().request(method, format!("{base}{path}"));
    if let Some(body) = body {
        request = request.json(&body);
    }
    let response = request.send().await.expect("запрос не отправлен");
    let status = response.status();
    (status, response.json().await.unwrap_or(Value::Null))
}

/// Настройки в том виде, в каком их шлёт интерфейс.
fn start_body(max_steps: u32) -> Value {
    json!({
        "model_name": "unsloth/Llama-3.2-1B-Instruct",
        "hf_dataset": "yahma/alpaca-cleaned",
        "local_datasets": [],
        "project_name": null,
        "training_type": "lora",
        "num_epochs": 1,
        "learning_rate": "2e-4",
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_steps": max_steps,
        "max_seq_length": 2048,
        "use_lora": true,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "target_modules": ["q_proj", "v_proj"],
        "load_in_4bit": true
    })
}

async fn start(base: &str, max_steps: u32) -> String {
    let (status, started) = call(
        base,
        Method::POST,
        "/api/train/start",
        Some(start_body(max_steps)),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{started}");
    assert_eq!(started["status"], "queued");
    started["job_id"].as_str().expect("job_id").to_string()
}

/// Читает SSE до итогового события и возвращает все события (имя и данные).
async fn read_sse(response: reqwest::Response, terminal: &[&str]) -> Vec<(String, Value)> {
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.bytes_stream();
    let mut buffer = String::new();
    let mut events = Vec::new();
    let deadline = tokio::time::Instant::now() + SSE_TIMEOUT;
    loop {
        let chunk = tokio::time::timeout_at(deadline, body.next())
            .await
            .expect("поток не дошёл до итогового события")
            .expect("поток закрылся раньше итогового события")
            .expect("ошибка чтения потока");
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(end) = buffer.find("\n\n") {
            let frame: String = buffer.drain(..end + 2).collect();
            let mut name = String::from("message");
            let mut data = String::new();
            for line in frame.lines() {
                if let Some(value) = line.strip_prefix("event:") {
                    name = value.trim().to_string();
                } else if let Some(value) = line.strip_prefix("data:") {
                    data.push_str(value.trim_start());
                }
            }
            if data.is_empty() {
                continue;
            }
            let payload: Value = serde_json::from_str(&data).expect("данные события — JSON");
            let done = terminal.contains(&name.as_str());
            events.push((name, payload));
            if done {
                return events;
            }
        }
    }
}

async fn progress_events(base: &str, job_id: &str) -> Vec<(String, Value)> {
    // POST без Accept, как это делает фронтенд
    let response = reqwest::Client::new()
        .post(format!(
            "{base}/api/train/progress?expected_job_id={job_id}"
        ))
        .send()
        .await
        .unwrap();
    read_sse(response, &["complete", "error"]).await
}

async fn wait_run_status(base: &str, job_id: &str, expected: &str) -> Value {
    for _ in 0..POLL_ATTEMPTS {
        let (status, detail) = call(
            base,
            Method::GET,
            &format!("/api/train/runs/{job_id}"),
            None,
        )
        .await;
        if status == StatusCode::OK && detail["run"]["status"] == expected {
            return detail;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
    panic!("запуск {job_id} не получил статус {expected}");
}

#[tokio::test]
async fn without_engine_training_is_honest() {
    let base = spawn(None).await;

    let (_, status) = call(&base, Method::GET, "/api/train/status", None).await;
    assert_eq!(status["phase"], "idle");
    assert_eq!(status["job_id"], "");
    assert_eq!(status["is_training_running"], false);
    assert!(
        status["message"]
            .as_str()
            .unwrap_or_default()
            .contains("этапе 3"),
        "{status}"
    );

    let mut body = start_body(10);
    body["start_request_id"] = json!("req-honest");
    let (code, started) = call(&base, Method::POST, "/api/train/start", Some(body)).await;
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE, "{started}");
    assert!(started["detail"].is_string());

    let (code, request) = call(
        &base,
        Method::GET,
        "/api/train/start-requests/req-honest",
        None,
    )
    .await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(request["state"], "rejected");
    assert_eq!(request["error_code"], "engine_unavailable");
    let (code, _) = call(
        &base,
        Method::POST,
        "/api/train/start-requests/req-honest/acknowledge",
        None,
    )
    .await;
    assert_eq!(code, StatusCode::OK);
    let (code, _) = call(
        &base,
        Method::GET,
        "/api/train/start-requests/req-honest",
        None,
    )
    .await;
    assert_eq!(code, StatusCode::NOT_FOUND);

    let (_, stopped) = call(
        &base,
        Method::POST,
        "/api/train/stop",
        Some(json!({ "save": true, "expected_job_id": "train-none" })),
    )
    .await;
    assert_eq!(stopped["status"], "idle");
    let (_, reset) = call(&base, Method::POST, "/api/train/reset", None).await;
    assert_eq!(reset["status"], "ok");

    let (_, runs) = call(
        &base,
        Method::GET,
        "/api/train/runs?limit=50&offset=0",
        None,
    )
    .await;
    assert_eq!(runs["total"], 0);
    let (code, _) = call(&base, Method::GET, "/api/train/runs/unknown", None).await;
    assert_eq!(code, StatusCode::NOT_FOUND);
    let (code, _) = call(&base, Method::DELETE, "/api/train/runs/unknown", None).await;
    assert_eq!(code, StatusCode::NOT_FOUND);

    let mut invalid = start_body(10);
    invalid["lora_r"] = json!(0);
    let (code, body) = call(&base, Method::POST, "/api/train/start", Some(invalid)).await;
    assert_eq!(code, StatusCode::UNPROCESSABLE_ENTITY, "{body}");

    for method in [Method::GET, Method::POST] {
        let response = reqwest::Client::new()
            .request(
                method,
                format!("{base}/api/train/progress?expected_job_id=none"),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let content_type = response.headers()[reqwest::header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .to_string();
        assert!(content_type.contains("text/event-stream"), "{content_type}");
    }
}

#[tokio::test]
async fn run_streams_progress_until_complete_and_is_saved() {
    let base = spawn(Some(StepEngine {
        steps: 5,
        delay: Duration::from_millis(20),
        fail_at: None,
    }))
    .await;
    let job_id = start(&base, 5).await;

    let events = progress_events(&base, &job_id).await;
    let (last_name, last) = events.last().expect("есть события");
    assert_eq!(last_name, "complete");
    assert_eq!(last["step"], 5);
    assert_eq!(last["progress_percent"], 100.0);
    assert!(events
        .iter()
        .all(|(_, payload)| payload["job_id"] == job_id.as_str()));

    let (_, status) = call(&base, Method::GET, "/api/train/status", None).await;
    assert_eq!(status["phase"], "completed");
    assert_eq!(
        status["metric_history"]["steps"].as_array().map(Vec::len),
        Some(5)
    );

    let (_, metrics) = call(
        &base,
        Method::GET,
        &format!("/api/train/metrics?expected_job_id={job_id}"),
        None,
    )
    .await;
    assert_eq!(metrics["current_step"], 5);
    assert_eq!(metrics["loss_history"].as_array().map(Vec::len), Some(5));

    let detail = wait_run_status(&base, &job_id, "completed").await;
    assert_eq!(detail["run"]["final_step"], 5);
    assert_eq!(
        detail["run"]["loss_sparkline"].as_array().map(Vec::len),
        Some(5)
    );
    assert_eq!(
        detail["metrics"]["loss_history"].as_array().map(Vec::len),
        Some(5)
    );
    assert_eq!(detail["config"]["model"], "unsloth/Llama-3.2-1B-Instruct");
}

#[tokio::test]
async fn stop_answers_after_engine_stopped_and_restart_works() {
    let base = spawn(Some(StepEngine {
        steps: 100_000,
        delay: Duration::from_millis(5),
        fail_at: None,
    }))
    .await;
    let first = start(&base, UNLIMITED_STEPS).await;
    for _ in 0..POLL_ATTEMPTS {
        let (_, status) = call(&base, Method::GET, "/api/train/status", None).await;
        if status["details"]["step"].as_u64().unwrap_or(0) > 0 {
            break;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }

    let (code, _) = call(&base, Method::POST, "/api/train/start", Some(start_body(0))).await;
    assert_eq!(code, StatusCode::CONFLICT, "второй запуск во время первого");
    let (code, _) = call(
        &base,
        Method::POST,
        "/api/train/reset",
        Some(json!({ "expected_job_id": first })),
    )
    .await;
    assert_eq!(code, StatusCode::CONFLICT, "идущий запуск не сбрасывается");
    let (code, _) = call(
        &base,
        Method::DELETE,
        &format!("/api/train/runs/{first}"),
        None,
    )
    .await;
    assert_eq!(code, StatusCode::CONFLICT, "идущий запуск не удаляется");

    let (_, stopped) = call(
        &base,
        Method::POST,
        "/api/train/stop",
        Some(json!({ "save": false, "expected_job_id": first })),
    )
    .await;
    assert_eq!(stopped["status"], "stopped");
    // Ответ пришёл после остановки движка: фаза уже итоговая, новый старт сразу проходит
    let (_, status) = call(&base, Method::GET, "/api/train/status", None).await;
    assert_eq!(status["phase"], "stopped");
    let second = start(&base, UNLIMITED_STEPS).await;
    assert_ne!(second, first, "id запусков уникальны");

    let (_, foreign) = call(
        &base,
        Method::POST,
        "/api/train/stop",
        Some(json!({ "expected_job_id": first })),
    )
    .await;
    assert_eq!(
        foreign["status"], "idle",
        "команда для старого запуска не трогает новый"
    );
    let (_, stopped) = call(
        &base,
        Method::POST,
        "/api/train/stop",
        Some(json!({ "expected_job_id": second })),
    )
    .await;
    assert_eq!(stopped["status"], "stopped");

    let (_, superseded) = call(
        &base,
        Method::POST,
        "/api/train/reset",
        Some(json!({ "expected_job_id": first })),
    )
    .await;
    assert_eq!(superseded["status"], "superseded");
    let (_, reset) = call(
        &base,
        Method::POST,
        "/api/train/reset",
        Some(json!({ "expected_job_id": second })),
    )
    .await;
    assert_eq!(reset["status"], "ok");
    let (_, status) = call(&base, Method::GET, "/api/train/status", None).await;
    assert_eq!(status["phase"], "idle");

    wait_run_status(&base, &first, "stopped").await;
    wait_run_status(&base, &second, "stopped").await;
}

#[tokio::test]
async fn engine_error_reaches_stream_status_and_history() {
    let base = spawn(Some(StepEngine {
        steps: 5,
        delay: Duration::from_millis(10),
        fail_at: Some(3),
    }))
    .await;
    let job_id = start(&base, 5).await;

    let events = progress_events(&base, &job_id).await;
    assert_eq!(events.last().map(|(name, _)| name.as_str()), Some("error"));

    let (_, status) = call(&base, Method::GET, "/api/train/status", None).await;
    assert_eq!(status["phase"], "error");
    assert!(
        status["error"]
            .as_str()
            .unwrap_or_default()
            .contains("видеопамяти"),
        "{status}"
    );
    let detail = wait_run_status(&base, &job_id, "error").await;
    assert!(detail["run"]["error_message"].is_string());
}

#[tokio::test]
async fn history_pages_renames_and_deletes() {
    let base = spawn(Some(StepEngine {
        steps: 1,
        delay: Duration::from_millis(1),
        fail_at: None,
    }))
    .await;
    let mut ids = Vec::new();
    for _ in 0..3 {
        let job_id = start(&base, 1).await;
        wait_run_status(&base, &job_id, "completed").await;
        ids.push(job_id);
    }

    let (_, page) = call(&base, Method::GET, "/api/train/runs?limit=2&offset=0", None).await;
    assert_eq!(page["runs"].as_array().map(Vec::len), Some(2));
    assert_eq!(page["total"], 3);
    let (_, rest) = call(&base, Method::GET, "/api/train/runs?limit=2&offset=2", None).await;
    assert_eq!(rest["runs"].as_array().map(Vec::len), Some(1));

    let (code, renamed) = call(
        &base,
        Method::PATCH,
        &format!("/api/train/runs/{}", ids[0]),
        Some(json!({ "display_name": "Мой первый запуск" })),
    )
    .await;
    assert_eq!(code, StatusCode::OK, "{renamed}");
    assert_eq!(renamed["display_name"], "Мой первый запуск");

    let (code, deleted) = call(
        &base,
        Method::DELETE,
        &format!("/api/train/runs/{}", ids[0]),
        None,
    )
    .await;
    assert_eq!(code, StatusCode::OK, "{deleted}");
    assert_eq!(deleted["artifacts_deleted"], false);
    let (code, _) = call(
        &base,
        Method::GET,
        &format!("/api/train/runs/{}", ids[0]),
        None,
    )
    .await;
    assert_eq!(code, StatusCode::NOT_FOUND);
    let (_, runs) = call(&base, Method::GET, "/api/train/runs", None).await;
    assert_eq!(runs["total"], 2);
}
