//! Модели и инференс: настоящие метаданные из GGUF, честная «загрузка» до появления
//! движка и оценка памяти по формуле KV-кэша.

use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use sloth_server::create_router;
use sloth_server::state::AppState;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;

const TINY_MODEL: &str = "tiny-llama-Q8_0.gguf";
const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VALUE_U32: u32 = 4;
const GGUF_VALUE_STRING: u32 = 8;
const GGML_TYPE_F32: u32 = 0;
const ALIGNMENT: usize = 32;

fn push_string(buf: &mut Vec<u8>, text: &str) {
    buf.extend_from_slice(&(text.len() as u64).to_le_bytes());
    buf.extend_from_slice(text.as_bytes());
}

/// Маленький корректный GGUF «llama»: 2 слоя, размерность 64, 4 головы, 2 KV-головы,
/// контекст 2048, шаблон чата и один тензор F32 4×4 (16 весов).
fn tiny_gguf() -> Vec<u8> {
    let u32_keys = [
        ("llama.context_length", 2048u32),
        ("llama.block_count", 2),
        ("llama.embedding_length", 64),
        ("llama.attention.head_count", 4),
        ("llama.attention.head_count_kv", 2),
    ];
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&((u32_keys.len() + 2) as u64).to_le_bytes());

    push_string(&mut buf, "general.architecture");
    buf.extend_from_slice(&GGUF_VALUE_STRING.to_le_bytes());
    push_string(&mut buf, "llama");
    for (key, value) in u32_keys {
        push_string(&mut buf, key);
        buf.extend_from_slice(&GGUF_VALUE_U32.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }
    push_string(&mut buf, "tokenizer.chat_template");
    buf.extend_from_slice(&GGUF_VALUE_STRING.to_le_bytes());
    push_string(&mut buf, "{{ messages }}");

    push_string(&mut buf, "token_embd.weight");
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&4u64.to_le_bytes());
    buf.extend_from_slice(&4u64.to_le_bytes());
    buf.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    while buf.len() % ALIGNMENT != 0 {
        buf.push(0);
    }
    buf.extend_from_slice(&[0u8; 64]);
    buf
}

struct Api {
    _models: TempDir,
    client: reqwest::Client,
    base_url: String,
}

impl Api {
    async fn start(prepare: impl FnOnce(&Path)) -> Self {
        let models = tempfile::tempdir().expect("не удалось создать временную папку моделей");
        prepare(models.path());
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
        Self {
            _models: models,
            client: reqwest::Client::new(),
            base_url: format!("http://{addr}"),
        }
    }

    async fn with_tiny_model() -> Self {
        Self::start(|dir| fs::write(dir.join(TINY_MODEL), tiny_gguf()).unwrap()).await
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

#[tokio::test]
async fn validate_reports_real_metadata() {
    let api = Api::with_tiny_model().await;
    let (status, body) = api
        .send(
            Method::POST,
            "/api/inference/validate",
            Some(json!({ "model_path": TINY_MODEL, "include_chat_template": true })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["valid"], true);
    assert_eq!(body["is_gguf"], true);
    assert_eq!(body["context_length"], 2048);
    assert_eq!(body["layer_count"], 2);
    assert_eq!(body["moe_layer_count"], 0);
    assert_eq!(body["chat_template"], "{{ messages }}");
    assert_eq!(body["is_vision"], false);
}

#[tokio::test]
async fn validate_detects_projector_missing_and_remote_models() {
    let api = Api::start(|dir| {
        fs::write(dir.join(TINY_MODEL), tiny_gguf()).unwrap();
        fs::write(dir.join("mmproj-F16.gguf"), b"projector").unwrap();
    })
    .await;

    let (_, with_projector) = api
        .send(
            Method::POST,
            "/api/inference/validate",
            Some(json!({ "model_path": TINY_MODEL })),
        )
        .await;
    assert_eq!(with_projector["is_vision"], true);

    let (_, missing) = api
        .send(
            Method::POST,
            "/api/inference/validate",
            Some(json!({ "model_path": "nope.gguf" })),
        )
        .await;
    assert_eq!(missing["valid"], false);
    assert!(missing["message"].is_string());

    let (_, remote) = api
        .send(
            Method::POST,
            "/api/inference/validate",
            Some(json!({ "model_path": "unsloth/Other-Model-GGUF" })),
        )
        .await;
    assert_eq!(remote["valid"], true);
    assert_eq!(
        remote["context_length"],
        Value::Null,
        "нескачанная модель — null, а не выдуманные 131072"
    );
}

#[tokio::test]
async fn load_checks_the_file_and_is_honest_about_the_engine() {
    let api = Api::start(|dir| {
        fs::write(dir.join(TINY_MODEL), tiny_gguf()).unwrap();
        fs::write(dir.join("broken-Q4_K_M.gguf"), b"not a gguf at all").unwrap();
    })
    .await;

    let (status, body) = api
        .send(
            Method::POST,
            "/api/inference/load",
            Some(json!({ "model_path": TINY_MODEL })),
        )
        .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    let detail = body["detail"].as_str().unwrap_or_default();
    assert!(detail.contains("этапе 2"), "{detail}");
    assert!(detail.contains("слоёв: 2"), "{detail}");

    let (status, _) = api
        .send(
            Method::POST,
            "/api/inference/load",
            Some(json!({ "model_path": "missing.gguf" })),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, _) = api
        .send(
            Method::POST,
            "/api/inference/load",
            Some(json!({ "model_path": "broken-Q4_K_M.gguf" })),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

    let (status, _) = api
        .send(
            Method::POST,
            "/api/inference/load",
            Some(json!({ "model_path": "unsloth/Not-Downloaded-GGUF", "gguf_variant": "Q4_K_M" })),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn status_and_progress_show_nothing_loaded() {
    let api = Api::with_tiny_model().await;

    let (_, status) = api.send(Method::GET, "/api/inference/status", None).await;
    assert_eq!(status["active_model"], Value::Null);
    assert_eq!(status["loaded"], json!([]));

    let (_, progress) = api
        .send(Method::GET, "/api/inference/load-progress", None)
        .await;
    assert_eq!(progress["phase"], Value::Null);

    let (code, _) = api
        .send(
            Method::POST,
            "/api/inference/unload",
            Some(json!({ "model_path": TINY_MODEL })),
        )
        .await;
    assert_eq!(code, StatusCode::OK);
}

#[tokio::test]
async fn memory_estimate_uses_kv_cache_formula() {
    let api = Api::with_tiny_model().await;
    let file_len = tiny_gguf().len() as u64;

    let (status, body) = api
        .send(
            Method::POST,
            "/api/inference/estimate-memory",
            Some(json!({ "model_path": TINY_MODEL, "n_ctx": 2048 })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // 2048 токенов × 2 слоя × 2 KV-головы × (16 + 16) × 2 байта (f16)
    assert_eq!(body["kv_bytes"], 524_288);
    assert_eq!(body["kv_estimable"], true);
    assert_eq!(body["weights_bytes"], file_len);
    assert_eq!(body["layer_count"], 2);

    let (_, q8) = api
        .send(
            Method::POST,
            "/api/inference/estimate-memory",
            Some(json!({ "model_path": TINY_MODEL, "n_ctx": 2048, "cache_type_kv": "q8_0" })),
        )
        .await;
    assert_eq!(q8["kv_bytes"], 524_288 / 2 * 34 / 32);

    let (status, _) = api
        .send(
            Method::POST,
            "/api/inference/estimate-memory",
            Some(json!({ "model_path": TINY_MODEL, "cache_type_kv": "turbo" })),
        )
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

    let (_, missing) = api
        .send(
            Method::POST,
            "/api/inference/estimate-memory",
            Some(json!({ "model_path": "nope.gguf" })),
        )
        .await;
    assert_eq!(missing["available"], false);
    assert_eq!(missing["reason"], "not_downloaded");
}

#[tokio::test]
async fn models_lists_and_kv_estimate_use_real_files() {
    let api = Api::with_tiny_model().await;

    let (_, list) = api.send(Method::GET, "/api/models/list", None).await;
    assert_eq!(list["models"].as_array().map(Vec::len), Some(1));
    assert_eq!(
        list["default_models"],
        json!([]),
        "несуществующие модели не подмешиваются"
    );

    let (_, local) = api.send(Method::GET, "/api/models/local", None).await;
    let model = &local["models"][0];
    assert_eq!(model["display_name"], "tiny-llama-Q8_0");
    assert_eq!(model["source"], "models_dir");
    assert_eq!(model["partial"], false);
    assert!(local["models_dir"].is_string());

    let (_, openai) = api.send(Method::GET, "/v1/models", None).await;
    assert_eq!(openai["data"].as_array().map(Vec::len), Some(1));

    let repo_id = model["model_id"]
        .as_str()
        .expect("model_id определён")
        .to_string();
    let (_, kv) = api
        .send(
            Method::GET,
            &format!("/api/models/kv-cache-estimate?repo_id={repo_id}&n_ctx=1024"),
            None,
        )
        .await;
    assert_eq!(kv["kv_bytes"], 262_144);
    assert_eq!(kv["native_context"], 2048);

    let (_, remote) = api
        .send(
            Method::GET,
            "/api/models/kv-cache-estimate?repo_id=unsloth/Not-Downloaded-GGUF&quant=Q4_K_M",
            None,
        )
        .await;
    assert_eq!(remote["kv_bytes"], Value::Null);
}

#[tokio::test]
async fn llama_flags_are_not_applicable() {
    let api = Api::with_tiny_model().await;
    let (status, body) = api
        .send(Method::GET, "/api/inference/llama-flags", None)
        .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert!(body["detail"].is_string());
}
