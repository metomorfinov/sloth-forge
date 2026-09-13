//! Загрузка моделей без интернета: внутри теста поднимается маленькая имитация
//! Hugging Face (дерево файлов + отдача файлов с поддержкой HTTP Range).

use axum::body::{Body, Bytes};
use axum::extract::{Path as AxPath, State as AxState};
use axum::http::{header, HeaderMap, StatusCode as AxStatus};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use sloth_server::create_router;
use sloth_server::state::AppState;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

const CHUNK: usize = 8 * 1024;
const SLOW_CHUNK_DELAY: Duration = Duration::from_millis(40);
const POLL_INTERVAL: Duration = Duration::from_millis(25);
const POLL_ATTEMPTS: usize = 400;
const REPO: &str = "unsloth/Tiny-GGUF";
const PRIVATE_REPO: &str = "private/Secret-GGUF";
const PRIVATE_TOKEN: &str = "hf_secret";

/// Предсказуемое содержимое файла: по нему проверяется, что докачка ничего не испортила.
fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

/// Файлы репозитория: путь → содержимое (None — файл есть в дереве, но не отдаётся).
type RepoFiles = Vec<(String, Option<Arc<Vec<u8>>>)>;

struct MockHf {
    /// «org/repo» → файлы
    repos: HashMap<String, RepoFiles>,
    slow: bool,
    range_requests: AtomicUsize,
}

impl MockHf {
    fn authorized(&self, repo: &str, headers: &HeaderMap) -> bool {
        repo != PRIVATE_REPO
            || headers
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                == Some(&format!("Bearer {PRIVATE_TOKEN}"))
    }
}

async fn mock_tree(
    AxState(mock): AxState<Arc<MockHf>>,
    AxPath((org, repo)): AxPath<(String, String)>,
    headers: HeaderMap,
) -> Response {
    let repo_id = format!("{org}/{repo}");
    if !mock.authorized(&repo_id, &headers) {
        return AxStatus::UNAUTHORIZED.into_response();
    }
    let Some(files) = mock.repos.get(&repo_id) else {
        return AxStatus::NOT_FOUND.into_response();
    };
    let entries: Vec<Value> = files
        .iter()
        .map(|(path, data)| {
            let size = data.as_ref().map_or(12_345, |d| d.len());
            json!({ "type": "file", "path": path, "size": 134, "lfs": { "size": size } })
        })
        .collect();
    Json(entries).into_response()
}

async fn mock_resolve(
    AxState(mock): AxState<Arc<MockHf>>,
    AxPath((org, repo, path)): AxPath<(String, String, String)>,
    headers: HeaderMap,
) -> Response {
    let repo_id = format!("{org}/{repo}");
    if !mock.authorized(&repo_id, &headers) {
        return AxStatus::UNAUTHORIZED.into_response();
    }
    let data = mock
        .repos
        .get(&repo_id)
        .and_then(|files| files.iter().find(|(p, _)| *p == path))
        .and_then(|(_, data)| data.clone());
    let Some(data) = data else {
        return AxStatus::NOT_FOUND.into_response();
    };

    let start = match headers.get(header::RANGE).and_then(|v| v.to_str().ok()) {
        Some(range) => {
            mock.range_requests.fetch_add(1, Ordering::Relaxed);
            let start: usize = range
                .trim_start_matches("bytes=")
                .trim_end_matches('-')
                .parse()
                .unwrap_or(0);
            if start >= data.len() {
                return AxStatus::RANGE_NOT_SATISFIABLE.into_response();
            }
            Some(start)
        }
        None => None,
    };
    let offset = start.unwrap_or(0);
    let slow = mock.slow;
    let stream = futures_util::stream::unfold((data, offset), move |(data, pos)| async move {
        if pos >= data.len() {
            return None;
        }
        if slow {
            tokio::time::sleep(SLOW_CHUNK_DELAY).await;
        }
        let end = (pos + CHUNK).min(data.len());
        let chunk = Bytes::copy_from_slice(&data[pos..end]);
        Some((Ok::<_, std::io::Error>(chunk), (data, end)))
    });
    let status = if start.is_some() {
        AxStatus::PARTIAL_CONTENT
    } else {
        AxStatus::OK
    };
    (status, Body::from_stream(stream)).into_response()
}

async fn spawn_mock(mock: MockHf) -> (String, Arc<MockHf>) {
    let mock = Arc::new(mock);
    let app = Router::new()
        .route("/api/models/:org/:repo/tree/main", get(mock_tree))
        .route("/:org/:repo/resolve/main/*path", get(mock_resolve))
        .with_state(Arc::clone(&mock));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("имитация HF упала");
    });
    (format!("http://{addr}"), mock)
}

fn tiny_repo() -> RepoFiles {
    vec![
        (
            "Q4_K_M/Tiny-Q4_K_M-00001-of-00002.gguf".into(),
            Some(Arc::new(pattern(40_000, 1))),
        ),
        (
            "Q4_K_M/Tiny-Q4_K_M-00002-of-00002.gguf".into(),
            Some(Arc::new(pattern(25_000, 2))),
        ),
        ("Tiny-Q8_0.gguf".into(), Some(Arc::new(pattern(30_000, 3)))),
        ("Broken-Q4_0.gguf".into(), None),
        ("mmproj-F16.gguf".into(), Some(Arc::new(pattern(1_000, 4)))),
    ]
}

struct Fixture {
    _models: TempDir,
    models_dir: PathBuf,
    client: reqwest::Client,
    base_url: String,
    mock: Arc<MockHf>,
}

impl Fixture {
    async fn start(slow: bool) -> Self {
        let mut repos = HashMap::new();
        repos.insert(REPO.to_string(), tiny_repo());
        repos.insert(
            PRIVATE_REPO.to_string(),
            vec![(
                "Secret-Q4_K_M.gguf".into(),
                Some(Arc::new(pattern(5_000, 9))),
            )],
        );
        let (hf_url, mock) = spawn_mock(MockHf {
            repos,
            slow,
            range_requests: AtomicUsize::new(0),
        })
        .await;

        let models = tempfile::tempdir().unwrap();
        let mut state = AppState::new(None, models.path().to_path_buf(), None);
        state.hf_endpoint = hf_url;
        let app = create_router(Arc::new(state), None);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("тестовый сервер упал");
        });
        Self {
            models_dir: models.path().to_path_buf(),
            _models: models,
            client: reqwest::Client::new(),
            base_url: format!("http://{addr}"),
            mock,
        }
    }

    async fn send(&self, method: Method, path: &str, body: Option<Value>) -> (StatusCode, Value) {
        let mut request = self
            .client
            .request(method, format!("{}{path}", self.base_url));
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = request.send().await.expect("запрос не отправлен");
        let status = response.status();
        (status, response.json().await.unwrap_or(Value::Null))
    }

    async fn start_download(&self, repo: &str, variant: &str) -> (StatusCode, Value) {
        self.send(
            Method::POST,
            "/api/hub/download",
            Some(json!({ "repo_id": repo, "gguf_variant": variant })),
        )
        .await
    }

    async fn status(&self, repo: &str, variant: &str) -> Value {
        self.send(
            Method::GET,
            &format!("/api/hub/download-status?repo_id={repo}&gguf_variant={variant}"),
            None,
        )
        .await
        .1
    }

    async fn wait_for(&self, repo: &str, variant: &str, wanted: &[&str]) -> Value {
        for _ in 0..POLL_ATTEMPTS {
            let status = self.status(repo, variant).await;
            if wanted.contains(&status["state"].as_str().unwrap_or("")) {
                return status;
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
        panic!("загрузка {repo} {variant} не пришла в состояние {wanted:?}");
    }

    fn repo_file(&self, repo: &str, path: &str) -> PathBuf {
        self.models_dir.join(repo).join(path)
    }
}

#[tokio::test]
async fn downloads_all_shards_into_repo_folder() {
    let fx = Fixture::start(false).await;

    // Полный запрос, который шлёт фронтенд (лишние поля не должны мешать)
    let (status, started) = fx
        .send(
            Method::POST,
            "/api/hub/download",
            Some(json!({
                "repo_id": REPO,
                "gguf_variant": "Q4_K_M",
                "use_xet": false,
                "scope_id": null,
                "files": null,
                "transport_mode": "http"
            })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{started}");
    assert_eq!(started["accepted"], true);
    assert!(started["generation"].is_number());

    let done = fx.wait_for(REPO, "Q4_K_M", &["complete", "error"]).await;
    assert_eq!(done["state"], "complete", "{done}");

    for (path, seed, len) in [
        ("Q4_K_M/Tiny-Q4_K_M-00001-of-00002.gguf", 1, 40_000),
        ("Q4_K_M/Tiny-Q4_K_M-00002-of-00002.gguf", 2, 25_000),
    ] {
        let content = std::fs::read(fx.repo_file(REPO, path)).expect("шард скачан");
        assert_eq!(content, pattern(len, seed), "содержимое {path} совпадает");
    }
    assert!(
        !fx.repo_file(REPO, "Tiny-Q8_0.gguf").exists(),
        "другой вариант не качается"
    );
    assert!(
        !fx.repo_file(REPO, "mmproj-F16.gguf").exists(),
        "проектор не считается моделью"
    );

    let (_, progress) = fx
        .send(
            Method::GET,
            &format!(
                "/api/hub/gguf-download-progress?repo_id={REPO}&variant=Q4_K_M&expected_bytes=1"
            ),
            None,
        )
        .await;
    assert_eq!(progress["complete_on_disk"], true);
    assert_eq!(progress["progress"], 1.0);

    // Повторный запуск не качает заново
    let (_, again) = fx.start_download(REPO, "Q4_K_M").await;
    assert_eq!(again["state"], "complete");
}

#[tokio::test]
async fn resumes_partial_file_with_range() {
    let fx = Fixture::start(false).await;
    let full = pattern(30_000, 3);
    let part = fx.repo_file(REPO, "Tiny-Q8_0.gguf.part");
    std::fs::create_dir_all(part.parent().unwrap()).unwrap();
    std::fs::write(&part, &full[..12_000]).unwrap();

    fx.start_download(REPO, "Q8_0").await;
    let done = fx.wait_for(REPO, "Q8_0", &["complete", "error"]).await;
    assert_eq!(done["state"], "complete", "{done}");
    assert_eq!(
        std::fs::read(fx.repo_file(REPO, "Tiny-Q8_0.gguf")).unwrap(),
        full
    );
    assert!(!part.exists());
    assert_eq!(
        fx.mock.range_requests.load(Ordering::Relaxed),
        1,
        "докачка через Range"
    );
}

#[tokio::test]
async fn parallel_jobs_do_not_clobber_each_other() {
    let fx = Fixture::start(true).await;
    fx.start_download(REPO, "Q4_K_M").await;
    fx.start_download(REPO, "Q8_0").await;

    let (_, active) = fx
        .send(Method::GET, "/api/hub/active-downloads", None)
        .await;
    let mut variants: Vec<String> = active["downloads"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["variant"].as_str().unwrap().to_string())
        .collect();
    variants.sort();
    assert_eq!(variants, ["Q4_K_M", "Q8_0"]);

    let first = fx.wait_for(REPO, "Q4_K_M", &["complete", "error"]).await;
    let second = fx.wait_for(REPO, "Q8_0", &["complete", "error"]).await;
    assert_eq!(first["state"], "complete");
    assert_eq!(second["state"], "complete");
}

#[tokio::test]
async fn cancel_keeps_partial_for_resume() {
    let fx = Fixture::start(true).await;
    fx.start_download(REPO, "Q4_K_M").await;

    // Ждём, пока данные начнут поступать
    for _ in 0..POLL_ATTEMPTS {
        let status = fx.status(REPO, "Q4_K_M").await;
        if status["downloaded_bytes"].as_u64().unwrap_or(0) > 0 {
            break;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
    let (_, cancelled) = fx
        .send(
            Method::POST,
            "/api/hub/download/cancel",
            Some(json!({ "repo_id": REPO, "gguf_variant": "Q4_K_M" })),
        )
        .await;
    assert!(matches!(
        cancelled["state"].as_str(),
        Some("cancelling" | "cancelled")
    ));

    let done = fx.wait_for(REPO, "Q4_K_M", &["cancelled"]).await;
    assert_eq!(done["state"], "cancelled");
    let (_, transport) = fx
        .send(
            Method::GET,
            &format!("/api/hub/transport-status?repo_id={REPO}&gguf_variant=Q4_K_M"),
            None,
        )
        .await;
    assert_eq!(transport["has_partial"], true);
    assert_eq!(transport["resumable"], true);
}

#[tokio::test]
async fn http_errors_become_error_state_with_message() {
    let fx = Fixture::start(false).await;
    let (status, _) = fx.start_download(REPO, "Q4_0").await;
    assert_eq!(status, StatusCode::OK);
    let done = fx.wait_for(REPO, "Q4_0", &["complete", "error"]).await;
    assert_eq!(done["state"], "error");
    assert!(
        done["error"].as_str().unwrap_or_default().contains("404"),
        "{done}"
    );
}

#[tokio::test]
async fn private_repo_requires_token_from_header_or_settings() {
    let fx = Fixture::start(false).await;

    let (status, body) = fx.start_download(PRIVATE_REPO, "Q4_K_M").await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("токен"),
        "{body}"
    );

    // Токен из заголовка фронтенда
    let response = fx
        .client
        .post(format!("{}/api/hub/download", fx.base_url))
        .header("X-Unsloth-HF-Token", PRIVATE_TOKEN)
        .json(&json!({ "repo_id": PRIVATE_REPO, "gguf_variant": "Q4_K_M" }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let done = fx
        .wait_for(PRIVATE_REPO, "Q4_K_M", &["complete", "error"])
        .await;
    assert_eq!(done["state"], "complete", "{done}");

    // Токен из настроек: удаляем файл и качаем без заголовка
    std::fs::remove_file(fx.repo_file(PRIVATE_REPO, "Secret-Q4_K_M.gguf")).unwrap();
    fx.send(
        Method::PUT,
        "/api/settings/hugging-face-token",
        Some(json!({ "token": PRIVATE_TOKEN })),
    )
    .await;
    let (status, _) = fx.start_download(PRIVATE_REPO, "Q4_K_M").await;
    assert_eq!(status, StatusCode::OK);
    let done = fx
        .wait_for(PRIVATE_REPO, "Q4_K_M", &["complete", "error"])
        .await;
    assert_eq!(done["state"], "complete", "{done}");
}

#[tokio::test]
async fn rejects_invalid_requests() {
    let fx = Fixture::start(false).await;
    for repo in ["../etc", "a/b/c", "org/.hidden"] {
        let (status, _) = fx.start_download(repo, "Q4_K_M").await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "repo_id «{repo}»");
    }
    let (status, _) = fx.start_download(REPO, "Q2_K").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

    let (_, idle) = fx
        .send(
            Method::GET,
            "/api/hub/download-status?repo_id=unsloth/Other-GGUF&gguf_variant=Q4_K_M",
            None,
        )
        .await;
    assert_eq!(idle["state"], "idle");
}
