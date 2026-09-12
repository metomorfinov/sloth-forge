//! Интеграционные тесты сетевой защиты и песочницы путей.
//!
//! Каждый тест поднимает настоящий сервер на временной папке моделей и проверяет,
//! что атаки через API (удаление чужих файлов, обход папок, чужой сайт) отклоняются.

use reqwest::header::{ACCESS_CONTROL_ALLOW_ORIGIN, HOST, ORIGIN};
use reqwest::StatusCode;
use sloth_server::create_router;
use sloth_server::state::AppState;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

/// Временная структура: `<base>/models/model-Q4_K_M.gguf`, `<base>/models/notes.txt`, `<base>/secret.txt`.
struct Sandbox {
    _base: TempDir,
    models: PathBuf,
    secret: PathBuf,
}

fn make_sandbox() -> Sandbox {
    let base = tempfile::tempdir().expect("не удалось создать временную папку");
    let models = base.path().join("models");
    fs::create_dir_all(&models).expect("не удалось создать models");
    fs::write(models.join("model-Q4_K_M.gguf"), b"GGUF").expect("не удалось создать gguf");
    fs::write(models.join("notes.txt"), b"notes").expect("не удалось создать notes.txt");
    let secret = base.path().join("secret.txt");
    fs::write(&secret, b"secret").expect("не удалось создать secret.txt");
    Sandbox {
        _base: base,
        models,
        secret,
    }
}

async fn spawn_server(models_dir: PathBuf) -> String {
    let state = Arc::new(AppState::new(None, models_dir, None));
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
    format!("http://{addr}")
}

async fn delete_cached(base_url: &str, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let response = reqwest::Client::new()
        .delete(format!("{base_url}/api/hub/delete-cached"))
        .json(&body)
        .send()
        .await
        .expect("запрос delete-cached не отправлен");
    let status = response.status();
    let json = response.json().await.unwrap_or(serde_json::Value::Null);
    (status, json)
}

#[tokio::test]
async fn delete_cached_rejects_absolute_path_outside_models() {
    let sandbox = make_sandbox();
    let base_url = spawn_server(sandbox.models.clone()).await;

    let (status, body) = delete_cached(
        &base_url,
        serde_json::json!({ "cache_path": sandbox.secret.to_string_lossy() }),
    )
    .await;

    assert_eq!(status, StatusCode::FORBIDDEN);
    assert!(
        body["detail"].as_str().is_some(),
        "ошибка должна быть в поле detail: {body}"
    );
    assert!(
        sandbox.secret.exists(),
        "файл вне папки моделей не должен удаляться"
    );
}

#[tokio::test]
async fn delete_cached_rejects_parent_traversal() {
    let sandbox = make_sandbox();
    let base_url = spawn_server(sandbox.models.clone()).await;

    let (status, _) = delete_cached(
        &base_url,
        serde_json::json!({ "cache_path": "../secret.txt" }),
    )
    .await;

    assert_eq!(status, StatusCode::FORBIDDEN);
    assert!(sandbox.secret.exists());
}

#[tokio::test]
async fn delete_cached_rejects_non_gguf_inside_models() {
    let sandbox = make_sandbox();
    let base_url = spawn_server(sandbox.models.clone()).await;

    let (status, _) =
        delete_cached(&base_url, serde_json::json!({ "cache_path": "notes.txt" })).await;

    assert_eq!(status, StatusCode::FORBIDDEN);
    assert!(sandbox.models.join("notes.txt").exists());
}

#[tokio::test]
async fn delete_cached_reports_missing_file() {
    let sandbox = make_sandbox();
    let base_url = spawn_server(sandbox.models.clone()).await;

    let (status, _) = delete_cached(
        &base_url,
        serde_json::json!({ "cache_path": "missing.gguf" }),
    )
    .await;

    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn delete_cached_removes_gguf_inside_models() {
    let sandbox = make_sandbox();
    let base_url = spawn_server(sandbox.models.clone()).await;
    let model = sandbox.models.join("model-Q4_K_M.gguf");

    let (status, body) = delete_cached(
        &base_url,
        serde_json::json!({ "cache_path": model.to_string_lossy() }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["deleted_files"].as_array().map(Vec::len), Some(1));
    assert!(!model.exists());
}

#[tokio::test]
async fn cors_allows_only_local_origins() {
    let sandbox = make_sandbox();
    let base_url = spawn_server(sandbox.models.clone()).await;
    let client = reqwest::Client::new();

    let foreign = client
        .get(format!("{base_url}/api/health"))
        .header(ORIGIN, "http://evil.example")
        .send()
        .await
        .expect("запрос не отправлен");
    assert!(
        foreign.headers().get(ACCESS_CONTROL_ALLOW_ORIGIN).is_none(),
        "чужой сайт не должен получать разрешение CORS"
    );

    let local = client
        .get(format!("{base_url}/api/health"))
        .header(ORIGIN, "http://localhost:3000")
        .send()
        .await
        .expect("запрос не отправлен");
    assert_eq!(
        local
            .headers()
            .get(ACCESS_CONTROL_ALLOW_ORIGIN)
            .and_then(|v| v.to_str().ok()),
        Some("http://localhost:3000")
    );
}

#[tokio::test]
async fn rejects_foreign_host_header() {
    let sandbox = make_sandbox();
    let base_url = spawn_server(sandbox.models.clone()).await;

    let response = reqwest::Client::new()
        .get(format!("{base_url}/api/health"))
        .header(HOST, "evil.example")
        .send()
        .await
        .expect("запрос не отправлен");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[cfg(unix)]
#[tokio::test]
async fn browse_folders_rejects_system_directories() {
    let sandbox = make_sandbox();
    let base_url = spawn_server(sandbox.models.clone()).await;

    let response = reqwest::Client::new()
        .get(format!("{base_url}/api/models/browse-folders?path=/etc"))
        .send()
        .await
        .expect("запрос не отправлен");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn cached_model_path_never_resolves_outside_models() {
    let sandbox = make_sandbox();
    let base_url = spawn_server(sandbox.models.clone()).await;
    let secret = sandbox.secret.to_string_lossy().to_string();

    let response = reqwest::Client::new()
        .get(format!("{base_url}/api/models/cached-model-path"))
        .query(&[("repo_id", secret.as_str())])
        .send()
        .await
        .expect("запрос не отправлен");

    assert_ne!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn rejects_state_changing_request_from_foreign_origin() {
    let sandbox = make_sandbox();
    let base_url = spawn_server(sandbox.models.clone()).await;
    let model = sandbox.models.join("model-Q4_K_M.gguf");
    let body = serde_json::json!({ "cache_path": model.to_string_lossy() });
    let client = reqwest::Client::new();

    // Страница чужого сайта пытается удалить модель (CSRF)
    let foreign = client
        .delete(format!("{base_url}/api/hub/delete-cached"))
        .header(ORIGIN, "http://evil.example")
        .json(&body)
        .send()
        .await
        .expect("запрос не отправлен");
    assert_eq!(foreign.status(), StatusCode::FORBIDDEN);
    assert!(model.exists(), "чужой сайт не должен удалять модели");

    // Тот же запрос из интерфейса SlothForge проходит
    let local = client
        .delete(format!("{base_url}/api/hub/delete-cached"))
        .header(ORIGIN, "http://localhost:3000")
        .json(&body)
        .send()
        .await
        .expect("запрос не отправлен");
    assert_eq!(local.status(), StatusCode::OK);
    assert!(!model.exists());
}
