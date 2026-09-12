//! Разделы без бэкенда и бывшие фейковые заглушки отвечают честно:
//! 501 с объяснением в поле `detail`, а не «успех», голый 404/405 или HTML.

use reqwest::header::CONTENT_TYPE;
use reqwest::{Method, StatusCode};
use sloth_server::create_router;
use sloth_server::state::AppState;
use std::sync::Arc;
use tempfile::TempDir;

async fn spawn_server() -> (String, TempDir) {
    let models = tempfile::tempdir().expect("не удалось создать временную папку моделей");
    let state = Arc::new(AppState::new(None, models.path().to_path_buf(), None));
    let app = create_router(state, None);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("не удалось занять порт для теста");
    let addr = listener.local_addr().expect("нет адреса слушателя");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("тестовый сервер упал");
    });
    (format!("http://{addr}"), models)
}

/// Проверяет, что запрос получил 501 и JSON с непустым `detail`.
async fn assert_not_implemented(base_url: &str, method: Method, path: &str) {
    let response = reqwest::Client::new()
        .request(method.clone(), format!("{base_url}{path}"))
        .send()
        .await
        .expect("запрос не отправлен");
    assert_eq!(
        response.status(),
        StatusCode::NOT_IMPLEMENTED,
        "{method} {path} должен отвечать 501"
    );
    let is_json = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("application/json"));
    assert!(is_json, "{method} {path} должен отдавать JSON, а не HTML");
    let body: serde_json::Value = response.json().await.expect("ответ не JSON");
    assert!(
        body["detail"].as_str().is_some_and(|d| !d.is_empty()),
        "{method} {path}: нет detail в {body}"
    );
}

#[tokio::test]
async fn sections_without_backend_answer_501() {
    let (base_url, _models) = spawn_server().await;
    let cases = [
        (Method::GET, "/api/prompts/entries"),
        (Method::POST, "/api/rag/knowledge-bases"),
        (Method::POST, "/api/data-recipe/seed/inspect"),
        (Method::GET, "/api/mcp/servers"),
        (Method::POST, "/api/inference/images/generate"),
        (Method::POST, "/api/train/diffusion/start"),
        (Method::POST, "/api/inference/audio/stt/load"),
        (Method::GET, "/api/chat/research-runs/abc"),
        (Method::POST, "/api/inference/tool-confirm"),
        (Method::GET, "/v1/videos/abc/content"),
        (Method::POST, "/v1/audio/speech"),
    ];
    for (method, path) in cases {
        assert_not_implemented(&base_url, method, path).await;
    }
}

#[tokio::test]
async fn former_fake_success_stubs_answer_501() {
    let (base_url, _models) = spawn_server().await;
    let cases = [
        (Method::POST, "/api/export/export/gguf"),
        (Method::POST, "/api/export/load-checkpoint"),
        (Method::POST, "/api/hub/datasets/upload"),
        (Method::POST, "/api/hub/datasets/check-format"),
        (Method::POST, "/api/hub/datasets/download"),
        (Method::POST, "/api/providers/test"),
        (Method::POST, "/api/llama/update"),
        (Method::DELETE, "/api/models/delete-finetuned"),
    ];
    for (method, path) in cases {
        assert_not_implemented(&base_url, method, path).await;
    }
}

#[tokio::test]
async fn status_endpoints_keep_working_inside_pending_sections() {
    let (base_url, _models) = spawn_server().await;
    let client = reqwest::Client::new();

    let images: serde_json::Value = client
        .get(format!("{base_url}/api/inference/images/status"))
        .send()
        .await
        .expect("запрос не отправлен")
        .json()
        .await
        .expect("ответ не JSON");
    assert_eq!(images["loaded"], serde_json::Value::Bool(false));

    let stt = client
        .get(format!("{base_url}/api/inference/audio/stt/status"))
        .send()
        .await
        .expect("запрос не отправлен");
    assert_eq!(stt.status(), StatusCode::OK);

    let rag: serde_json::Value = client
        .get(format!("{base_url}/api/rag/knowledge-bases"))
        .send()
        .await
        .expect("запрос не отправлен")
        .json()
        .await
        .expect("ответ не JSON");
    assert_eq!(rag["knowledgeBases"], serde_json::json!([]));
    assert_eq!(rag["ragAvailable"], serde_json::Value::Bool(false));
}

#[tokio::test]
async fn unknown_api_path_returns_json_detail() {
    let (base_url, _models) = spawn_server().await;
    let response = reqwest::Client::new()
        .get(format!("{base_url}/api/definitely-missing"))
        .send()
        .await
        .expect("запрос не отправлен");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: serde_json::Value = response.json().await.expect("ответ не JSON");
    assert!(body["detail"].as_str().is_some());
}
