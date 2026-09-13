//! Настройки, связанные с моделями и загрузками: бюджет видеопамяти, память моделей,
//! последняя загруженная модель, автопереключение моделей, способ скачивания,
//! токен и кэш Hugging Face, эмбеддинг-модель и путь к llama.cpp.

use super::{delete_value, optional_bool, optional_u64, read_value, require_bool, write_value};
use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use crate::store::now_ms;
use axum::extract::State;
use axum::Json;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const VRAM_BUDGET_KEY: &str = "settings.vram_budget_fraction";
const MODEL_MEMORY_KEY: &str = "settings.model_memory";
const LAST_LOCAL_MODEL_KEY: &str = "settings.last_local_model";
const AUTO_SWITCH_KEY: &str = "settings.openai_auto_switch";
const MODEL_OVERRIDES_KEY: &str = "settings.openai_auto_switch.overrides";
const DOWNLOAD_TRANSPORT_KEY: &str = "settings.download_transport";
const HF_TOKEN_KEY: &str = "secrets.hugging_face_token";

/// Доля видеопамяти, которую может занять загрузка модели (`vram-budget.ts`).
const DEFAULT_VRAM_FRACTION: f64 = 0.9;
const MIN_VRAM_FRACTION: f64 = 0.1;
const MAX_VRAM_FRACTION: f64 = 1.0;

/// Виды «последней модели» (`last-local-model-load.ts`).
const LOCAL_MODEL_KINDS: [&str; 2] = ["gguf", "model"];

const DEFAULT_EMBEDDING_MODEL: &str = "BAAI/bge-small-en-v1.5";
const DEFAULT_EMBEDDING_GGUF_REPO: &str = "BAAI/bge-small-en-v1.5-GGUF";

// ---------- бюджет видеопамяти ----------

fn vram_budget_json(fraction: f64, is_stored: bool) -> Json<Value> {
    Json(json!({
        "fraction": fraction,
        "is_stored": is_stored,
        "default_fraction": DEFAULT_VRAM_FRACTION,
        "min_fraction": MIN_VRAM_FRACTION,
        "max_fraction": MAX_VRAM_FRACTION,
        "reload_required": false
    }))
}

fn valid_fraction(value: f64) -> bool {
    value.is_finite() && (MIN_VRAM_FRACTION..=MAX_VRAM_FRACTION).contains(&value)
}

pub async fn get_vram_budget(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    Ok(
        match read_value(&state, VRAM_BUDGET_KEY)
            .await?
            .and_then(|value| value.as_f64())
            .filter(|fraction| valid_fraction(*fraction))
        {
            Some(fraction) => vram_budget_json(fraction, true),
            None => vram_budget_json(DEFAULT_VRAM_FRACTION, false),
        },
    )
}

/// `{ fraction: число }` сохраняет долю, `{ fraction: null }` возвращает значение по умолчанию.
pub async fn put_vram_budget(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    match body.get("fraction") {
        None | Some(Value::Null) => {
            delete_value(&state, VRAM_BUDGET_KEY).await?;
            Ok(vram_budget_json(DEFAULT_VRAM_FRACTION, false))
        }
        Some(value) => {
            let fraction = value
                .as_f64()
                .filter(|fraction| valid_fraction(*fraction))
                .ok_or_else(|| {
                    ApiError::unprocessable(format!(
                        "Доля видеопамяти должна быть от {MIN_VRAM_FRACTION} до {MAX_VRAM_FRACTION}"
                    ))
                })?;
            // Два знака после запятой: так значение показывает интерфейс
            let fraction = (fraction * 100.0).round() / 100.0;
            write_value(&state, VRAM_BUDGET_KEY, json!(fraction)).await?;
            Ok(vram_budget_json(fraction, true))
        }
    }
}

// ---------- память моделей ----------

fn model_memory_json(keep_resident: bool, no_ram_reserve: bool) -> Json<Value> {
    Json(json!({
        "keep_resident": keep_resident,
        "no_ram_reserve": no_ram_reserve,
        "default_keep_resident": false,
        "default_no_ram_reserve": false,
        "mlock_active": false,
        "reload_required": false,
        "memlock_limit_bytes": null
    }))
}

async fn stored_model_memory(state: &AppState) -> ApiResult<(bool, bool)> {
    let stored = read_value(state, MODEL_MEMORY_KEY)
        .await?
        .unwrap_or(Value::Null);
    Ok((
        stored
            .get("keep_resident")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        stored
            .get("no_ram_reserve")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    ))
}

pub async fn get_model_memory(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let (keep_resident, no_ram_reserve) = stored_model_memory(&state).await?;
    Ok(model_memory_json(keep_resident, no_ram_reserve))
}

pub async fn put_model_memory(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let (mut keep_resident, mut no_ram_reserve) = stored_model_memory(&state).await?;
    if let Some(value) = optional_bool(&body, "keep_resident")? {
        keep_resident = value;
    }
    if let Some(value) = optional_bool(&body, "no_ram_reserve")? {
        no_ram_reserve = value;
    }
    write_value(
        &state,
        MODEL_MEMORY_KEY,
        json!({ "keep_resident": keep_resident, "no_ram_reserve": no_ram_reserve }),
    )
    .await?;
    Ok(model_memory_json(keep_resident, no_ram_reserve))
}

// ---------- последняя загруженная модель ----------

pub async fn get_last_local_model(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let mut record = match read_value(&state, LAST_LOCAL_MODEL_KEY).await? {
        Some(Value::Object(map)) => map,
        _ => {
            let mut empty = Map::new();
            for field in ["id", "kind", "gguf_variant", "loaded_at"] {
                empty.insert(field.into(), Value::Null);
            }
            empty
        }
    };
    // server_now позволяет фронтенду поправить разницу часов браузера и сервера
    record.insert("server_now".into(), json!(now_ms()));
    Ok(Json(Value::Object(record)))
}

pub async fn put_last_local_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let id = body
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| ApiError::unprocessable("Не указан id модели"))?;
    let kind = body
        .get("kind")
        .and_then(Value::as_str)
        .filter(|kind| LOCAL_MODEL_KINDS.contains(kind))
        .ok_or_else(|| ApiError::unprocessable("kind должен быть gguf или model"))?;
    let gguf_variant = body.get("gguf_variant").and_then(Value::as_str);
    let loaded_at = body
        .get("loaded_at")
        .and_then(Value::as_i64)
        .unwrap_or_else(now_ms);

    let record = json!({
        "id": id,
        "kind": kind,
        "gguf_variant": gguf_variant,
        "loaded_at": loaded_at
    });
    write_value(&state, LAST_LOCAL_MODEL_KEY, record.clone()).await?;
    let mut response = record;
    response["server_now"] = json!(now_ms());
    Ok(Json(response))
}

// ---------- автопереключение моделей OpenAI-совместимого API ----------

/// Сохраняемые поля и их значения по умолчанию.
fn auto_switch_defaults() -> Map<String, Value> {
    let Value::Object(map) = json!({
        "enabled": false,
        "auto_unload_idle_seconds": 0,
        "auto_unload_keep_kv": true,
        "auto_download_model": false,
        "auto_unload_api_only": false,
        "media_auto_unload_idle_seconds": 0,
        "media_auto_switch_model": false
    }) else {
        unreachable!("json! с фигурными скобками всегда даёт объект")
    };
    map
}

const AUTO_SWITCH_BOOL_FIELDS: [&str; 4] = [
    "auto_unload_keep_kv",
    "auto_download_model",
    "auto_unload_api_only",
    "media_auto_switch_model",
];
const AUTO_SWITCH_SECONDS_FIELDS: [&str; 2] =
    ["auto_unload_idle_seconds", "media_auto_unload_idle_seconds"];

async fn auto_switch_settings(state: &AppState) -> ApiResult<Map<String, Value>> {
    let mut settings = auto_switch_defaults();
    if let Some(Value::Object(stored)) = read_value(state, AUTO_SWITCH_KEY).await? {
        for (key, value) in stored {
            if settings.contains_key(&key) {
                settings.insert(key, value);
            }
        }
    }
    Ok(settings)
}

fn auto_switch_json(mut settings: Map<String, Value>) -> Json<Value> {
    settings.insert("default_enabled".into(), json!(false));
    // Выгрузка по простою работает только при загруженной модели, а движка инференса ещё нет
    settings.insert("idle_unload_active".into(), json!(false));
    settings.insert("media_idle_unload_active".into(), json!(false));
    Json(Value::Object(settings))
}

pub async fn get_auto_switch(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    Ok(auto_switch_json(auto_switch_settings(&state).await?))
}

pub async fn put_auto_switch(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let mut settings = auto_switch_settings(&state).await?;
    settings.insert("enabled".into(), json!(require_bool(&body, "enabled")?));
    for field in AUTO_SWITCH_BOOL_FIELDS {
        if let Some(value) = optional_bool(&body, field)? {
            settings.insert(field.into(), json!(value));
        }
    }
    for field in AUTO_SWITCH_SECONDS_FIELDS {
        if let Some(value) = optional_u64(&body, field)? {
            settings.insert(field.into(), json!(value));
        }
    }
    write_value(&state, AUTO_SWITCH_KEY, Value::Object(settings.clone())).await?;
    Ok(auto_switch_json(settings))
}

pub async fn get_model_overrides(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let overrides = read_value(&state, MODEL_OVERRIDES_KEY)
        .await?
        .unwrap_or_else(|| json!({}));
    Ok(Json(json!({ "overrides": overrides })))
}

pub async fn put_model_overrides(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let overrides = body.get("overrides").cloned().unwrap_or(body);
    if !overrides.is_object() {
        return Err(ApiError::unprocessable(
            "overrides должен быть JSON-объектом",
        ));
    }
    write_value(&state, MODEL_OVERRIDES_KEY, overrides.clone()).await?;
    Ok(Json(json!({ "status": "ok", "overrides": overrides })))
}

// ---------- способ скачивания моделей ----------

fn download_transport_json(mode: &str) -> Json<Value> {
    Json(json!({
        "mode": mode,
        "xet_available": false,
        "xet_unavailable_reason": "Xet пока не поддерживается в SlothForge: модели скачиваются напрямую по HTTP",
        "auto_resolves_to": "http",
        "auto_reason": "В SlothForge доступна только прямая загрузка по HTTP"
    }))
}

pub async fn get_download_transport(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let stored = read_value(&state, DOWNLOAD_TRANSPORT_KEY).await?;
    let mode = stored
        .as_ref()
        .and_then(Value::as_str)
        .filter(|mode| matches!(*mode, "auto" | "http"))
        .unwrap_or("auto");
    Ok(download_transport_json(mode))
}

pub async fn put_download_transport(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let mode = match body.get("mode").and_then(Value::as_str) {
        Some(mode @ ("auto" | "http")) => mode,
        Some("xet") => {
            return Err(ApiError::unprocessable(
                "Xet пока не поддерживается в SlothForge: выберите auto или http",
            ))
        }
        _ => return Err(ApiError::unprocessable("mode должен быть auto или http")),
    };
    write_value(&state, DOWNLOAD_TRANSPORT_KEY, json!(mode)).await?;
    Ok(download_transport_json(mode))
}

// ---------- токен Hugging Face ----------
//
// Токен хранится в локальной базе и отдаётся только интерфейсу SlothForge: фронтенд сам
// подставляет его в запросы к Hugging Face. Сервер доступен только с этого компьютера,
// чужие сайты ответ прочитать не могут. В логи токен не пишется.

fn hf_token_json(token: Option<String>) -> Json<Value> {
    Json(json!({ "has_token": token.is_some(), "token": token }))
}

async fn stored_hf_token(state: &AppState) -> ApiResult<Option<String>> {
    Ok(read_value(state, HF_TOKEN_KEY)
        .await?
        .and_then(|value| value.as_str().map(str::to_string))
        .filter(|token| !token.is_empty()))
}

pub async fn get_hf_token(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    Ok(hf_token_json(stored_hf_token(&state).await?))
}

/// Пустой токен удаляет сохранённый.
pub async fn put_hf_token(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let token = body
        .get("token")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    if token.is_empty() {
        delete_value(&state, HF_TOKEN_KEY).await?;
        return Ok(hf_token_json(None));
    }
    write_value(&state, HF_TOKEN_KEY, json!(token)).await?;
    Ok(hf_token_json(Some(token.to_string())))
}

/// Перенос токена из браузера старых версий: не затирает уже сохранённый.
pub async fn migrate_hf_token(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    if stored_hf_token(&state).await?.is_some() {
        return get_hf_token(State(state)).await;
    }
    put_hf_token(State(state), Json(body)).await
}

pub async fn delete_hf_token(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    delete_value(&state, HF_TOKEN_KEY).await?;
    Ok(hf_token_json(None))
}

// ---------- кэш Hugging Face ----------

/// Папка кэша как в huggingface_hub: `HF_HOME`, иначе `~/.cache/huggingface`.
fn hugging_face_cache_path() -> (PathBuf, bool) {
    match std::env::var_os("HF_HOME") {
        Some(custom) => (PathBuf::from(custom), true),
        None => (
            crate::paths::home_dir()
                .map(|home| home.join(".cache").join("huggingface"))
                .unwrap_or_else(|| PathBuf::from(".cache").join("huggingface")),
            false,
        ),
    }
}

/// Свободное место на диске, где лежит (или появится) папка.
fn free_space_bytes(path: &Path) -> Option<u64> {
    let existing = path.ancestors().find(|ancestor| ancestor.exists())?;
    let existing = existing.canonicalize().ok()?;
    let disks = sysinfo::Disks::new_with_refreshed_list();
    disks
        .list()
        .iter()
        .filter(|disk| existing.starts_with(disk.mount_point()))
        .max_by_key(|disk| disk.mount_point().as_os_str().len())
        .map(|disk| disk.available_space())
}

pub async fn get_hugging_face_cache() -> ApiResult<Json<Value>> {
    let (cache_path, from_env) = hugging_face_cache_path();
    let probe_path = cache_path.clone();
    // Опрос дисков синхронный, поэтому выполняется вне async-потоков сервера
    let (free_bytes, writable) = tokio::task::spawn_blocking(move || {
        let writable = probe_path
            .ancestors()
            .find(|ancestor| ancestor.exists())
            .and_then(|ancestor| std::fs::metadata(ancestor).ok())
            .is_some_and(|metadata| !metadata.permissions().readonly());
        (free_space_bytes(&probe_path), writable)
    })
    .await
    .map_err(|err| ApiError::internal(format!("Проверка диска прервана: {err}")))?;

    Ok(Json(json!({
        "cache_home": cache_path.to_string_lossy(),
        "hub_cache": cache_path.join("hub").to_string_lossy(),
        "xet_cache": cache_path.join("xet").to_string_lossy(),
        "source": if from_env { "environment" } else { "default" },
        "editable": false,
        "is_custom": from_env,
        "available": cache_path.exists(),
        "writable": writable,
        "free_bytes": free_bytes,
        "environment_variable": if from_env { Some("HF_HOME") } else { None }
    })))
}

pub async fn put_hugging_face_cache() -> ApiError {
    ApiError::not_implemented(
        "Смена папки кэша Hugging Face пока не поддерживается: модели SlothForge скачиваются в папку models",
    )
}

// ---------- эмбеддинг-модель (для RAG) ----------

fn embedding_model_json() -> Json<Value> {
    Json(json!({
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "embedding_gguf_repo": DEFAULT_EMBEDDING_GGUF_REPO,
        "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
        "default_embedding_gguf_repo": DEFAULT_EMBEDDING_GGUF_REPO,
        "is_custom": false,
        "loaded": false,
        "backend_loaded": false
    }))
}

/// GET, сброс (DELETE) и выгрузка: модель по умолчанию, ничего не загружено.
pub async fn get_embedding_model() -> Json<Value> {
    embedding_model_json()
}

/// Выбор эмбеддинг-модели и её поиск появятся вместе с RAG.
pub async fn embedding_model_not_ready() -> ApiError {
    crate::unavailable::not_ready("RAG (базы знаний)", 4)
}

// ---------- путь к llama.cpp ----------

pub async fn get_llama_cpp_path() -> Json<Value> {
    Json(json!({
        "path": null,
        "source": "default",
        "editable": false,
        "available": false,
        "resolved_binary": null,
        "environment_variable": null,
        "reload_required": false
    }))
}

pub async fn put_llama_cpp_path() -> ApiError {
    ApiError::not_implemented(
        "SlothForge использует собственный Vulkan-движок, путь к llama.cpp не нужен",
    )
}
