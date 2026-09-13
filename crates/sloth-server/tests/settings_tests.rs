//! Настройки `/api/settings/*`: форматы фронтенда, проверка значений и сохранность
//! после перезапуска сервера.

use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use sloth_server::create_router;
use sloth_server::state::AppState;
use sloth_server::store::Store;
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;

struct Api {
    client: reqwest::Client,
    base_url: String,
}

impl Api {
    async fn start(database: &Path, models: &Path) -> Self {
        let store = Arc::new(Store::open(database).expect("база не открылась"));
        let state = Arc::new(AppState::with_store(
            None,
            models.to_path_buf(),
            None,
            store,
        ));
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
        Self {
            client: reqwest::Client::new(),
            base_url: format!("http://{addr}"),
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
}

async fn start() -> (TempDir, Api) {
    let dir = tempfile::tempdir().expect("не удалось создать временную папку");
    let api = Api::start(&dir.path().join("slothforge.db"), dir.path()).await;
    (dir, api)
}

#[tokio::test]
async fn personalization_roundtrip_and_saved_flags() {
    let (_dir, api) = start().await;

    let (_, initial) = api
        .send(Method::GET, "/api/settings/personalization", None)
        .await;
    assert_eq!(initial["saved"], false);
    assert!(initial["profile"].is_object());

    let body = json!({
        "version": 1,
        "profile": { "displayName": "Ленивец", "nickname": "sloth", "avatarDataUrl": null, "avatarShape": "rounded", "showGreetingSloth": false },
        "appearance": { "theme": "light", "palette": "minimal", "language": "ru", "customization": {} }
    });
    let (status, _) = api
        .send(Method::PUT, "/api/settings/personalization", Some(body))
        .await;
    assert_eq!(status, StatusCode::OK);

    let (_, saved) = api
        .send(Method::GET, "/api/settings/personalization", None)
        .await;
    assert_eq!(saved["saved"], true);
    assert_eq!(saved["profile"]["displayName"], "Ленивец");
    assert_eq!(saved["appearance"]["language"], "ru");

    let (status, _) = api
        .send(
            Method::PUT,
            "/api/settings/personalization",
            Some(json!({ "version": 1 })),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn numeric_settings_validate_ranges() {
    let (_dir, api) = start().await;

    let (_, limit) = api
        .send(Method::GET, "/api/settings/upload-limit", None)
        .await;
    assert_eq!(limit["max_upload_size_mb"], 500);
    let (status, _) = api
        .send(
            Method::PUT,
            "/api/settings/upload-limit",
            Some(json!({ "max_upload_size_mb": 99999 })),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (_, limit) = api
        .send(
            Method::PUT,
            "/api/settings/upload-limit",
            Some(json!({ "max_upload_size_mb": 1024 })),
        )
        .await;
    assert_eq!(limit["max_upload_size_bytes"], 1024 * 1024 * 1024);

    let (_, budget) = api
        .send(Method::GET, "/api/settings/vram-budget", None)
        .await;
    assert_eq!(budget["is_stored"], false);
    let (_, budget) = api
        .send(
            Method::PUT,
            "/api/settings/vram-budget",
            Some(json!({ "fraction": 0.756 })),
        )
        .await;
    assert_eq!(budget["fraction"], 0.76);
    assert_eq!(budget["is_stored"], true);
    let (status, _) = api
        .send(
            Method::PUT,
            "/api/settings/vram-budget",
            Some(json!({ "fraction": 1.5 })),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (_, reset) = api
        .send(
            Method::PUT,
            "/api/settings/vram-budget",
            Some(json!({ "fraction": null })),
        )
        .await;
    assert_eq!(reset["is_stored"], false);
    assert_eq!(reset["fraction"], 0.9);
}

#[tokio::test]
async fn frontend_shapes_for_runtime_settings() {
    let (_dir, api) = start().await;

    let (_, memory) = api
        .send(
            Method::PUT,
            "/api/settings/model-memory",
            Some(json!({ "keep_resident": true })),
        )
        .await;
    assert_eq!(memory["keep_resident"], true);
    assert_eq!(memory["no_ram_reserve"], false);
    assert!(
        memory.get("fraction").is_none(),
        "старый неверный формат не возвращается"
    );

    let (_, keyless) = api
        .send(Method::GET, "/api/settings/keyless-api-access", None)
        .await;
    assert_eq!(keyless["scope"], "off");
    let (status, _) = api
        .send(
            Method::PUT,
            "/api/settings/keyless-api-access",
            Some(json!({ "scope": "everyone" })),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

    let (_, switch) = api
        .send(
            Method::PUT,
            "/api/settings/openai-auto-switch",
            Some(json!({ "enabled": true, "auto_unload_idle_seconds": 300 })),
        )
        .await;
    assert_eq!(switch["enabled"], true);
    assert_eq!(switch["auto_unload_idle_seconds"], 300);
    assert_eq!(switch["auto_unload_keep_kv"], true);

    let (status, _) = api
        .send(
            Method::PUT,
            "/api/settings/download-transport",
            Some(json!({ "mode": "xet" })),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (_, transport) = api
        .send(
            Method::PUT,
            "/api/settings/download-transport",
            Some(json!({ "mode": "http" })),
        )
        .await;
    assert_eq!(transport["mode"], "http");

    let (status, _) = api
        .send(
            Method::PUT,
            "/api/settings/llama-cpp-path",
            Some(json!({ "path": "/usr/bin/llama-server" })),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);

    let (_, cache) = api
        .send(Method::GET, "/api/settings/hugging-face-cache", None)
        .await;
    assert!(cache["cache_home"].is_string());
    assert_ne!(
        cache["free_bytes"],
        json!(107_374_182_400u64),
        "свободное место не выдумано"
    );
}

#[tokio::test]
async fn last_local_model_and_chat_preferences() {
    let (_dir, api) = start().await;

    let (_, empty) = api
        .send(Method::GET, "/api/settings/last-local-model", None)
        .await;
    assert_eq!(
        empty["id"],
        Value::Null,
        "никакой выдуманной модели по умолчанию"
    );
    assert!(empty["server_now"].is_number());

    let (status, _) = api
        .send(
            Method::PUT,
            "/api/settings/last-local-model",
            Some(json!({ "id": "x", "kind": "weird" })),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    api.send(
        Method::PUT,
        "/api/settings/last-local-model",
        Some(json!({ "id": "unsloth/Llama-3.2-1B-Instruct-GGUF", "kind": "gguf", "gguf_variant": "Q4_K_M", "loaded_at": 123 })),
    )
    .await;
    let (_, last) = api
        .send(Method::GET, "/api/settings/last-local-model", None)
        .await;
    assert_eq!(last["gguf_variant"], "Q4_K_M");
    assert_eq!(last["loaded_at"], 123);

    let (_, prefs) = api
        .send(Method::GET, "/api/settings/chat-preferences", None)
        .await;
    assert_eq!(prefs["show_model_disclaimer"], true);
    // Миграция применяется, пока значения на сервере нет, и не затирает сохранённое
    let (_, migrated) = api
        .send(
            Method::POST,
            "/api/settings/chat-preferences/migrate",
            Some(json!({ "show_model_disclaimer": false })),
        )
        .await;
    assert_eq!(migrated["show_model_disclaimer"], false);
    let (_, again) = api
        .send(
            Method::POST,
            "/api/settings/chat-preferences/migrate",
            Some(json!({ "show_model_disclaimer": true })),
        )
        .await;
    assert_eq!(again["show_model_disclaimer"], false);
}

#[tokio::test]
async fn hf_token_and_generation_presets() {
    let (_dir, api) = start().await;

    let (_, token) = api
        .send(
            Method::PUT,
            "/api/settings/hugging-face-token",
            Some(json!({ "token": "hf_test" })),
        )
        .await;
    assert_eq!(token["has_token"], true);
    let (_, migrated) = api
        .send(
            Method::PUT,
            "/api/settings/hugging-face-token/migrate",
            Some(json!({ "token": "hf_old" })),
        )
        .await;
    assert_eq!(
        migrated["token"], "hf_test",
        "миграция не затирает сохранённый токен"
    );
    let (_, cleared) = api
        .send(Method::DELETE, "/api/settings/hugging-face-token", None)
        .await;
    assert_eq!(cleared["has_token"], false);

    let (_, initial) = api
        .send(Method::GET, "/api/settings/generation-presets/image", None)
        .await;
    assert_eq!(initial["saved"], false);
    api.send(
        Method::PUT,
        "/api/settings/generation-presets/image",
        Some(json!({ "currentParams": { "steps": 20 }, "activePreset": "fast" })),
    )
    .await;
    api.send(
        Method::PUT,
        "/api/settings/generation-presets/image/custom",
        Some(json!({ "name": "fast", "params": { "steps": 8 } })),
    )
    .await;
    let (_, saved) = api
        .send(Method::GET, "/api/settings/generation-presets/image", None)
        .await;
    assert_eq!(saved["saved"], true);
    assert_eq!(saved["currentParams"]["steps"], 20);
    assert_eq!(saved["customPresets"][0]["params"]["steps"], 8);

    let (_, deleted) = api
        .send(
            Method::DELETE,
            "/api/settings/generation-presets/image/custom?name=fast",
            None,
        )
        .await;
    assert_eq!(deleted["deleted"], true);

    let (status, _) = api
        .send(Method::GET, "/api/settings/generation-presets/audio", None)
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn settings_survive_restart() {
    let dir = tempfile::tempdir().expect("не удалось создать временную папку");
    let database = dir.path().join("slothforge.db");
    {
        let api = Api::start(&database, dir.path()).await;
        api.send(
            Method::PUT,
            "/api/settings/current-date-prompt",
            Some(json!({ "enabled": false })),
        )
        .await;
        api.send(
            Method::PUT,
            "/api/settings/helper-precache",
            Some(json!({ "enabled": true })),
        )
        .await;
    }
    let restarted = Api::start(&database, dir.path()).await;
    let (_, date_prompt) = restarted
        .send(Method::GET, "/api/settings/current-date-prompt", None)
        .await;
    assert_eq!(date_prompt["enabled"], false);
    assert_eq!(date_prompt["default_enabled"], true);
    let (_, precache) = restarted
        .send(Method::GET, "/api/settings/helper-precache", None)
        .await;
    assert_eq!(precache["enabled"], true);
}
