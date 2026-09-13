//! Настройки интерфейса: профиль и оформление, предпочтения чата, переключатели,
//! лимит загрузки файлов и пресеты генерации изображений/видео.

use super::{optional_bool, read_bool, read_value, require_bool, write_value};
use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use axum::extract::{Path, Query, State};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::sync::Arc;

const PERSONALIZATION_KEY: &str = "settings.personalization";
const SHOW_MODEL_DISCLAIMER_KEY: &str = "settings.chat_preferences.show_model_disclaimer";
const CURRENT_DATE_PROMPT_KEY: &str = "settings.current_date_prompt";
const PREVIEW_SHARING_KEY: &str = "settings.preview_sharing";
const HELPER_PRECACHE_KEY: &str = "settings.helper_precache";
const KEYLESS_API_ACCESS_KEY: &str = "settings.keyless_api_access";
const UPLOAD_LIMIT_KEY: &str = "settings.upload_limit_mb";

const DEFAULT_SHOW_MODEL_DISCLAIMER: bool = true;
const DEFAULT_CURRENT_DATE_PROMPT: bool = true;
const DEFAULT_PREVIEW_SHARING: bool = false;
const DEFAULT_HELPER_PRECACHE: bool = false;

/// Лимит загрузки файлов в мегабайтах (значения из `upload-limit.ts` фронтенда).
const DEFAULT_UPLOAD_LIMIT_MB: u64 = 500;
const MIN_UPLOAD_LIMIT_MB: u64 = 50;
const MAX_UPLOAD_LIMIT_MB: u64 = 2048;
const BYTES_PER_MB: u64 = 1024 * 1024;

/// Флаги «сохранено на сервере»: фронтенд их получает, но обратно не присылает.
const PERSONALIZATION_SAVED_FLAGS: [&str; 4] = [
    "saved",
    "customizationSaved",
    "paletteSaved",
    "greetingSlothSaved",
];

/// Уровни доступа к API без ключа (`keyless-api-access.ts`).
const KEYLESS_SCOPES: [&str; 3] = ["off", "inference", "full"];

/// Виды пресетов генерации (`generation-presets/types.ts`).
const PRESET_KINDS: [&str; 2] = ["image", "video"];

// ---------- персонализация ----------

/// Профиль по умолчанию. Имя берётся из учётной записи системы, а не зашито в код.
fn default_personalization() -> Value {
    let user_name = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_default();
    json!({
        "version": 1,
        "profile": {
            "displayName": user_name,
            "nickname": user_name,
            "avatarDataUrl": null,
            "avatarShape": "circle",
            "showGreetingSloth": true
        },
        "appearance": {
            "theme": "dark",
            "palette": "standard",
            "language": null,
            "customization": {}
        }
    })
}

fn with_saved_flags(mut value: Value, saved: bool) -> Value {
    if let Some(object) = value.as_object_mut() {
        for flag in PERSONALIZATION_SAVED_FLAGS {
            object.insert(flag.into(), json!(saved));
        }
    }
    value
}

pub async fn get_personalization(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    Ok(Json(
        match read_value(&state, PERSONALIZATION_KEY).await? {
            Some(stored) => with_saved_flags(stored, true),
            None => with_saved_flags(default_personalization(), false),
        },
    ))
}

pub async fn put_personalization(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let Value::Object(mut object) = body else {
        return Err(ApiError::bad_request(
            "Персонализация должна быть JSON-объектом",
        ));
    };
    let has_sections = object.get("profile").is_some_and(Value::is_object)
        && object.get("appearance").is_some_and(Value::is_object);
    if !has_sections {
        return Err(ApiError::unprocessable(
            "В персонализации нужны объекты profile и appearance",
        ));
    }
    for flag in PERSONALIZATION_SAVED_FLAGS {
        object.remove(flag);
    }
    let value = Value::Object(object);
    write_value(&state, PERSONALIZATION_KEY, value.clone()).await?;
    Ok(Json(with_saved_flags(value, true)))
}

// ---------- предпочтения чата ----------

fn chat_preferences_json(show_model_disclaimer: bool) -> Json<Value> {
    Json(json!({ "show_model_disclaimer": show_model_disclaimer }))
}

pub async fn get_chat_preferences(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let show = read_bool(
        &state,
        SHOW_MODEL_DISCLAIMER_KEY,
        DEFAULT_SHOW_MODEL_DISCLAIMER,
    )
    .await?;
    Ok(chat_preferences_json(show))
}

pub async fn put_chat_preferences(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let show = require_bool(&body, "show_model_disclaimer")?;
    write_value(&state, SHOW_MODEL_DISCLAIMER_KEY, json!(show)).await?;
    Ok(chat_preferences_json(show))
}

/// Перенос значения из браузера старых версий: применяется, только если на сервере
/// значения ещё нет, чтобы не затереть более новое.
pub async fn migrate_chat_preferences(
    State(state): State<Arc<AppState>>,
    payload: Option<Json<Value>>,
) -> ApiResult<Json<Value>> {
    let body = payload
        .map(|Json(value)| value)
        .unwrap_or_else(|| json!({}));
    if read_value(&state, SHOW_MODEL_DISCLAIMER_KEY)
        .await?
        .is_none()
    {
        if let Some(show) = optional_bool(&body, "show_model_disclaimer")? {
            write_value(&state, SHOW_MODEL_DISCLAIMER_KEY, json!(show)).await?;
        }
    }
    get_chat_preferences(State(state)).await
}

// ---------- переключатели ----------

pub async fn get_current_date_prompt(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let enabled = read_bool(&state, CURRENT_DATE_PROMPT_KEY, DEFAULT_CURRENT_DATE_PROMPT).await?;
    Ok(Json(
        json!({ "enabled": enabled, "default_enabled": DEFAULT_CURRENT_DATE_PROMPT }),
    ))
}

pub async fn put_current_date_prompt(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let enabled = require_bool(&body, "enabled")?;
    write_value(&state, CURRENT_DATE_PROMPT_KEY, json!(enabled)).await?;
    Ok(Json(
        json!({ "enabled": enabled, "default_enabled": DEFAULT_CURRENT_DATE_PROMPT }),
    ))
}

pub async fn get_preview_sharing(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let enabled = read_bool(&state, PREVIEW_SHARING_KEY, DEFAULT_PREVIEW_SHARING).await?;
    Ok(Json(
        json!({ "enabled": enabled, "default_enabled": DEFAULT_PREVIEW_SHARING }),
    ))
}

pub async fn put_preview_sharing(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let enabled = require_bool(&body, "enabled")?;
    write_value(&state, PREVIEW_SHARING_KEY, json!(enabled)).await?;
    Ok(Json(
        json!({ "enabled": enabled, "default_enabled": DEFAULT_PREVIEW_SHARING }),
    ))
}

fn helper_precache_json(enabled: bool) -> Json<Value> {
    Json(json!({
        "enabled": enabled,
        "default_enabled": DEFAULT_HELPER_PRECACHE,
        "disabled_by_env": false
    }))
}

pub async fn get_helper_precache(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let enabled = read_bool(&state, HELPER_PRECACHE_KEY, DEFAULT_HELPER_PRECACHE).await?;
    Ok(helper_precache_json(enabled))
}

pub async fn put_helper_precache(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let enabled = require_bool(&body, "enabled")?;
    write_value(&state, HELPER_PRECACHE_KEY, json!(enabled)).await?;
    Ok(helper_precache_json(enabled))
}

// ---------- доступ к API без ключа ----------
//
// Значение сохраняется, но проверка ключей API в SlothForge пока не реализована:
// сервер защищён тем, что по умолчанию доступен только с этого компьютера.

fn keyless_json(scope: &str, tools: bool) -> Json<Value> {
    Json(json!({ "scope": scope, "tools": tools, "exposure": null }))
}

pub async fn get_keyless_api_access(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let stored = read_value(&state, KEYLESS_API_ACCESS_KEY)
        .await?
        .unwrap_or(Value::Null);
    let scope = stored
        .get("scope")
        .and_then(Value::as_str)
        .filter(|scope| KEYLESS_SCOPES.contains(scope))
        .unwrap_or("off");
    let tools = stored
        .get("tools")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(keyless_json(scope, tools))
}

pub async fn put_keyless_api_access(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let scope = body
        .get("scope")
        .and_then(Value::as_str)
        .filter(|scope| KEYLESS_SCOPES.contains(scope))
        .ok_or_else(|| ApiError::unprocessable("scope должен быть off, inference или full"))?
        .to_string();
    let tools = optional_bool(&body, "tools")?.unwrap_or(false);
    write_value(
        &state,
        KEYLESS_API_ACCESS_KEY,
        json!({ "scope": scope, "tools": tools }),
    )
    .await?;
    Ok(keyless_json(&scope, tools))
}

// ---------- лимит загрузки файлов ----------

fn upload_limit_json(mb: u64) -> Json<Value> {
    Json(json!({
        "max_upload_size_mb": mb,
        "max_upload_size_bytes": mb * BYTES_PER_MB,
        "max_upload_size_label": format!("{mb}MB"),
        "default_upload_size_mb": DEFAULT_UPLOAD_LIMIT_MB,
        "min_upload_size_mb": MIN_UPLOAD_LIMIT_MB,
        "max_allowed_upload_size_mb": MAX_UPLOAD_LIMIT_MB
    }))
}

pub async fn get_upload_limit(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let mb = read_value(&state, UPLOAD_LIMIT_KEY)
        .await?
        .and_then(|value| value.as_u64())
        .filter(|mb| (MIN_UPLOAD_LIMIT_MB..=MAX_UPLOAD_LIMIT_MB).contains(mb))
        .unwrap_or(DEFAULT_UPLOAD_LIMIT_MB);
    Ok(upload_limit_json(mb))
}

pub async fn put_upload_limit(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let mb = body
        .get("max_upload_size_mb")
        .and_then(Value::as_u64)
        .filter(|mb| (MIN_UPLOAD_LIMIT_MB..=MAX_UPLOAD_LIMIT_MB).contains(mb))
        .ok_or_else(|| {
            ApiError::unprocessable(format!(
                "Лимит загрузки должен быть от {MIN_UPLOAD_LIMIT_MB} до {MAX_UPLOAD_LIMIT_MB} МБ"
            ))
        })?;
    write_value(&state, UPLOAD_LIMIT_KEY, json!(mb)).await?;
    Ok(upload_limit_json(mb))
}

// ---------- пресеты генерации ----------

fn preset_key(kind: &str) -> ApiResult<String> {
    if PRESET_KINDS.contains(&kind) {
        Ok(format!("settings.generation_presets.{kind}"))
    } else {
        Err(ApiError::not_found(format!(
            "Неизвестный вид пресетов генерации: {kind}"
        )))
    }
}

async fn load_presets(state: &AppState, key: &str) -> ApiResult<Map<String, Value>> {
    Ok(match read_value(state, key).await? {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    })
}

fn custom_presets(settings: &mut Map<String, Value>) -> &mut Vec<Value> {
    let entry = settings.entry("customPresets").or_insert_with(|| json!([]));
    if !entry.is_array() {
        *entry = json!([]);
    }
    entry
        .as_array_mut()
        .expect("customPresets только что приведён к массиву")
}

pub async fn get_generation_presets(
    State(state): State<Arc<AppState>>,
    Path(kind): Path<String>,
) -> ApiResult<Json<Value>> {
    let key = preset_key(&kind)?;
    Ok(Json(match read_value(&state, key).await? {
        Some(Value::Object(mut settings)) => {
            custom_presets(&mut settings);
            settings.insert("saved".into(), json!(true));
            Value::Object(settings)
        }
        _ => json!({ "customPresets": [], "saved": false }),
    }))
}

/// Сохраняет текущие параметры и выбранный пресет, не трогая пользовательские пресеты.
pub async fn put_generation_presets(
    State(state): State<Arc<AppState>>,
    Path(kind): Path<String>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let key = preset_key(&kind)?;
    let mut settings = load_presets(&state, &key).await?;
    for field in ["currentParams", "activePreset"] {
        if let Some(value) = body.get(field) {
            settings.insert(field.into(), value.clone());
        }
    }
    custom_presets(&mut settings);
    write_value(&state, key, Value::Object(settings)).await?;
    Ok(Json(json!({ "saved": true })))
}

/// Добавляет или заменяет пользовательский пресет с тем же именем.
pub async fn put_custom_preset(
    State(state): State<Arc<AppState>>,
    Path(kind): Path<String>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let key = preset_key(&kind)?;
    let name = body
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| ApiError::unprocessable("У пресета должно быть имя"))?
        .to_string();
    let preset =
        json!({ "name": name, "params": body.get("params").cloned().unwrap_or(Value::Null) });

    let mut settings = load_presets(&state, &key).await?;
    let presets = custom_presets(&mut settings);
    match presets
        .iter_mut()
        .find(|existing| existing.get("name").and_then(Value::as_str) == Some(name.as_str()))
    {
        Some(existing) => *existing = preset,
        None => presets.push(preset),
    }
    write_value(&state, key, Value::Object(settings)).await?;
    Ok(Json(json!({ "saved": true })))
}

#[derive(Debug, Deserialize)]
pub struct PresetNameQuery {
    name: String,
}

pub async fn delete_custom_preset(
    State(state): State<Arc<AppState>>,
    Path(kind): Path<String>,
    Query(query): Query<PresetNameQuery>,
) -> ApiResult<Json<Value>> {
    let key = preset_key(&kind)?;
    let mut settings = load_presets(&state, &key).await?;
    let presets = custom_presets(&mut settings);
    let before = presets.len();
    presets
        .retain(|preset| preset.get("name").and_then(Value::as_str) != Some(query.name.as_str()));
    let deleted = presets.len() != before;
    if deleted {
        write_value(&state, key, Value::Object(settings)).await?;
    }
    Ok(Json(json!({ "deleted": deleted })))
}
