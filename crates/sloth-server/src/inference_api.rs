//! Модели и инференс: список локальных моделей, проверка модели, оценка памяти,
//! статус и загрузка.
//!
//! Собственный движок инференса SlothForge подключается на этапе 2 дорожной карты.
//! До этого сервер отвечает честно:
//! - список моделей и их метаданные — настоящие, из файлов и заголовков GGUF;
//! - «загрузка» проверяет файл и сообщает (503), что загрузить модель в видеопамять
//!   пока нечем — вместо прежней имитации с паузами по 200 мс и `status: "ok"`;
//! - статус и прогресс показывают, что ничего не загружено.

use crate::error::{ApiError, ApiResult};
use crate::model_inventory::{self, GgufSummary, LocalModel};
use crate::state::AppState;
use axum::extract::{Query, State};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use sloth_core::gguf::GGMLType;
use std::path::Path;
use std::sync::Arc;

/// Длина контекста для оценки памяти, если её не указали в запросе и нет в файле.
const FALLBACK_CONTEXT_LENGTH: u64 = 4096;
/// Тип KV-кэша по умолчанию (как у llama.cpp).
const DEFAULT_KV_CACHE_TYPE: &str = "f16";
/// Грубая оценка буферов вычислений (промежуточные тензоры, логиты). Точное число
/// станет известно вместе с движком; до этого это явно помеченная оценка.
const COMPUTE_BUFFER_ESTIMATE_BYTES: u64 = 256 * 1024 * 1024;
const PARAMETERS_PER_BILLION: f64 = 1e9;
const PARAMETERS_PER_MILLION: f64 = 1e6;

/// Поля запросов validate / load / estimate-memory, которые использует SlothForge.
/// Остальные поля фронтенда (флаги llama-server и т. п.) игнорируются.
#[derive(Debug, Default, Deserialize)]
pub struct ModelRequest {
    #[serde(default)]
    model_path: Option<String>,
    #[serde(default)]
    gguf_variant: Option<String>,
    #[serde(default)]
    include_chat_template: bool,
    #[serde(default)]
    n_ctx: Option<u64>,
    #[serde(default)]
    cache_type_kv: Option<String>,
    #[serde(default)]
    n_parallel: Option<u64>,
}

/// Что стоит за идентификатором модели из запроса.
enum ModelRef {
    /// Файл модели найден в папках моделей.
    Local(LocalModel),
    /// Это репозиторий Hugging Face, но нужного файла на диске нет.
    NotDownloaded { repo_id: String },
    /// Путь или имя файла, которых нет в папках моделей.
    Missing,
}

async fn scan(state: &AppState) -> ApiResult<Vec<LocalModel>> {
    crate::hub::local_models(state).await
}

fn repo_id_of(model: &LocalModel) -> String {
    model.repo_id()
}

async fn resolve_model(
    state: &AppState,
    identifier: &str,
    variant: Option<&str>,
) -> ApiResult<ModelRef> {
    let identifier = identifier.trim();
    if identifier.is_empty() {
        return Err(ApiError::bad_request("Не указана модель (model_path)"));
    }
    let models = scan(state).await?;

    // 1. Путь или имя файла внутри папок моделей (через песочницу путей)
    if let Some(path) = crate::hub::resolve_local_gguf_file(state, identifier).await {
        if let Some(model) = models
            .iter()
            .find(|model| model.path == path || model.shard_paths.contains(&path))
        {
            return Ok(ModelRef::Local(model.clone()));
        }
    }

    // 2. Репозиторий Hugging Face (`org/repo`) и, возможно, вариант кванта
    let looks_like_repo = identifier.contains('/')
        && !Path::new(identifier).is_absolute()
        && !identifier.to_ascii_lowercase().ends_with(".gguf");
    if looks_like_repo {
        let variant = variant.map(str::trim).filter(|v| !v.is_empty());
        let found = models.iter().find(|model| {
            repo_id_of(model).eq_ignore_ascii_case(identifier)
                && variant.is_none_or(|v| {
                    model.quant().eq_ignore_ascii_case(v)
                })
        });
        return Ok(match found {
            Some(model) => ModelRef::Local(model.clone()),
            None => ModelRef::NotDownloaded {
                repo_id: identifier.to_string(),
            },
        });
    }
    Ok(ModelRef::Missing)
}

/// Сводка метаданных и наличие проектора изображений рядом с моделью.
async fn summarize(model: &LocalModel) -> ApiResult<(GgufSummary, bool)> {
    let model = model.clone();
    tokio::task::spawn_blocking(move || {
        let has_projector = model_inventory::has_projector(&model);
        model_inventory::summarize(&model).map(|summary| (summary, has_projector))
    })
    .await
    .map_err(|err| ApiError::internal(format!("Чтение модели прервано: {err}")))?
    .map_err(|err| ApiError::unprocessable(format!("Файл модели повреждён или не читается: {err}")))
}

/// Число весов для людей: «1.24 млрд», «350 млн».
fn format_parameters(count: u64) -> String {
    let count = count as f64;
    if count >= PARAMETERS_PER_BILLION {
        format!("{:.2} млрд", count / PARAMETERS_PER_BILLION)
    } else {
        format!("{:.0} млн", count / PARAMETERS_PER_MILLION)
    }
}

fn missing_detail(identifier: &str) -> String {
    format!("Модель «{identifier}» не найдена в папках моделей")
}

fn not_downloaded_detail(repo_id: &str) -> String {
    format!("Модель {repo_id} ещё не скачана: сначала загрузите её во вкладке Hub")
}

// ---------- список моделей ----------

/// `GET /api/models`, `/api/models/list`: модели, которые можно выбрать в интерфейсе.
pub async fn list_models(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let models: Vec<Value> = scan(&state)
        .await?
        .iter()
        .filter(|model| model.complete)
        .map(|model| {
            json!({
                "id": model.id(),
                "name": model.display_name,
                "is_vision": false,
                "is_lora": false,
                "is_gguf": true,
                "is_mlx": false,
                "is_audio": false
            })
        })
        .collect();
    // Моделей «по умолчанию», которых нет на диске, больше не подмешивается
    Ok(Json(json!({ "models": models, "default_models": [] })))
}

/// `GET /api/models/local`: файлы моделей в папке моделей и scan-folders.
pub async fn list_local_models(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let models: Vec<Value> = scan(&state)
        .await?
        .iter()
        .map(|model| {
            json!({
                "id": model.id(),
                "display_name": model.display_name,
                "path": model.id(),
                "source": if model.root_index == 0 { "models_dir" } else { "custom" },
                "model_id": repo_id_of(model),
                "model_format": "gguf",
                "partial": !model.complete,
                "updated_at": model.modified_secs,
                "task": "text-generation",
                "audio_type": null
            })
        })
        .collect();
    let models_dir = state
        .models_dir
        .canonicalize()
        .unwrap_or_else(|_| state.models_dir.clone());
    Ok(Json(json!({
        "models_dir": models_dir.to_string_lossy(),
        "hf_cache_dir": null,
        "lmstudio_dirs": [],
        "models": models
    })))
}

/// `GET /v1/models` в формате OpenAI.
pub async fn openai_models(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let data: Vec<Value> = scan(&state)
        .await?
        .iter()
        .filter(|model| model.complete)
        .map(|model| {
            json!({
                "id": model.display_name,
                "object": "model",
                "created": model.modified_secs.unwrap_or(0),
                "owned_by": "slothforge"
            })
        })
        .collect();
    Ok(Json(json!({ "object": "list", "data": data })))
}

// ---------- проверка и загрузка ----------

/// `POST /api/inference/validate`: настоящие метаданные модели из заголовка GGUF.
pub async fn validate_model(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ModelRequest>,
) -> ApiResult<Json<Value>> {
    let identifier = request.model_path.clone().unwrap_or_default();
    match resolve_model(&state, &identifier, request.gguf_variant.as_deref()).await? {
        ModelRef::Local(model) => match summarize(&model).await {
            Ok((summary, has_projector)) => Ok(Json(json!({
                "valid": model.complete,
                "message": if model.complete {
                    "Модель найдена на диске"
                } else {
                    "Модель скачана не полностью: не хватает части файлов"
                },
                "identifier": model.id(),
                "display_name": model.display_name,
                "is_gguf": true,
                "is_lora": false,
                "is_vision": has_projector,
                "is_diffusion": false,
                "diffusion_unknown": false,
                "requires_trust_remote_code": false,
                "context_length": summary.context_length,
                "layer_count": summary.block_count,
                "moe_layer_count": summary.moe_layer_count(),
                "chat_template": if request.include_chat_template { summary.chat_template } else { None }
            }))),
            Err(err) => Ok(Json(json!({
                "valid": false,
                "message": err.detail,
                "identifier": model.id(),
                "display_name": model.display_name,
                "is_gguf": true
            }))),
        },
        ModelRef::NotDownloaded { repo_id } => Ok(Json(json!({
            "valid": true,
            "message": "Модель ещё не скачана",
            "display_name": repo_id.rsplit('/').next().unwrap_or(&repo_id),
            "is_gguf": repo_id.to_ascii_lowercase().ends_with("-gguf"),
            "identifier": repo_id,
            "is_lora": false,
            "is_vision": false,
            "context_length": null,
            "layer_count": null,
            "moe_layer_count": null,
            "chat_template": null
        }))),
        ModelRef::Missing => Ok(Json(json!({
            "valid": false,
            "message": missing_detail(&identifier),
            "identifier": identifier
        }))),
    }
}

/// `POST /api/inference/load`: файл проверяется по-настоящему, но загрузить модель
/// в видеопамять можно будет только с движком инференса (этап 2).
pub async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ModelRequest>,
) -> ApiResult<Json<Value>> {
    let identifier = request.model_path.clone().unwrap_or_default();
    match resolve_model(&state, &identifier, request.gguf_variant.as_deref()).await? {
        ModelRef::Local(model) => {
            if !model.complete {
                return Err(ApiError::unprocessable(format!(
                    "Модель «{}» скачана не полностью: не хватает части файлов",
                    model.display_name
                )));
            }
            let (summary, _) = summarize(&model).await?;
            Err(ApiError::service_unavailable(format!(
                "Модель «{}» ({}, слоёв: {}, весов: {}) прочитана, но загрузить её некуда: \
                 движок инференса SlothForge подключится на этапе 2 дорожной карты",
                model.display_name,
                summary
                    .architecture
                    .as_deref()
                    .unwrap_or("архитектура не указана"),
                summary
                    .block_count
                    .map_or_else(|| "?".to_string(), |layers| layers.to_string()),
                format_parameters(summary.parameter_count),
            )))
        }
        ModelRef::NotDownloaded { repo_id } => {
            Err(ApiError::not_found(not_downloaded_detail(&repo_id)))
        }
        ModelRef::Missing => Err(ApiError::not_found(missing_detail(&identifier))),
    }
}

/// `POST /api/inference/unload`: загруженной модели нет, выгружать нечего.
pub async fn unload_model() -> Json<Value> {
    Json(json!({ "status": "ok", "unloaded": false, "message": "Нет загруженной модели" }))
}

/// `GET /api/inference/status`: ничего не загружено (движок подключится на этапе 2).
pub async fn inference_status() -> Json<Value> {
    Json(json!({
        "active_model": null,
        "model_identifier": null,
        "is_vision": false,
        "is_gguf": false,
        "is_local_model": false,
        "gguf_variant": null,
        "loading": [],
        "loaded": [],
        "inference": null,
        "supports_tools": false,
        "supports_reasoning": false,
        "context_length": null,
        "max_context_length": null,
        "native_context_length": null
    }))
}

/// `GET /api/inference/load-progress`: загрузки нет — `phase: null`, как ждёт фронтенд.
pub async fn load_progress() -> Json<Value> {
    Json(json!({ "phase": null, "bytes_loaded": 0, "bytes_total": 0, "fraction": 0.0 }))
}

/// `GET /api/inference/llama-flags`: каталог флагов llama-server к SlothForge не относится.
pub async fn llama_flags() -> ApiError {
    ApiError::not_implemented(
        "SlothForge не использует llama-server: флаги запуска llama.cpp к нему не применяются",
    )
}

// ---------- оценка памяти ----------

fn unavailable_estimate(reason: &str) -> Json<Value> {
    Json(json!({
        "available": false,
        "reason": reason,
        "weights_bytes": 0,
        "kv_bytes": 0,
        "compute_bytes": 0,
        "drafter_runtime_bytes": 0,
        "drafter_runtime_gpu_bytes": 0,
        "projector_runtime_bytes": 0,
        "drafter_kv_unsized": false,
        "adapters_unsized": false,
        "total_bytes": 0,
        "gpu_bytes": 0,
        "kv_estimable": false,
        "kv_on_gpu": true,
        "n_ctx": 0,
        "cache_type_kv": null,
        "n_parallel": 1,
        "layer_count": null,
        "gpu_layers": null,
        "moe_offload_unmodelled": false
    }))
}

fn parse_cache_type(requested: Option<&str>) -> ApiResult<(GGMLType, String)> {
    let name = requested
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or(DEFAULT_KV_CACHE_TYPE);
    GGMLType::from_name(name)
        .map(|ty| (ty, name.to_ascii_lowercase()))
        .ok_or_else(|| ApiError::unprocessable(format!("Неизвестный тип KV-кэша: {name}")))
}

/// `POST /api/inference/estimate-memory`: веса — размер файлов, KV-кэш — по формуле
/// из метаданных, буферы вычислений — явно помеченная оценка.
pub async fn estimate_memory(
    State(state): State<Arc<AppState>>,
    payload: Option<Json<ModelRequest>>,
) -> ApiResult<Json<Value>> {
    let request = payload.map(|Json(request)| request).unwrap_or_default();
    let identifier = request.model_path.clone().unwrap_or_default();
    if identifier.trim().is_empty() {
        return Ok(unavailable_estimate("not_downloaded"));
    }
    let model = match resolve_model(&state, &identifier, request.gguf_variant.as_deref()).await? {
        ModelRef::Local(model) if model.complete => model,
        _ => return Ok(unavailable_estimate("not_downloaded")),
    };
    let Ok((summary, _)) = summarize(&model).await else {
        return Ok(unavailable_estimate("unsizable"));
    };

    let (cache_type, cache_type_name) = parse_cache_type(request.cache_type_kv.as_deref())?;
    let n_ctx = request
        .n_ctx
        .filter(|n| *n > 0)
        .or(summary.context_length)
        .unwrap_or(FALLBACK_CONTEXT_LENGTH);
    let n_parallel = request.n_parallel.filter(|n| *n > 0).unwrap_or(1);
    let kv_bytes = model_inventory::kv_cache_bytes(&summary, n_ctx, cache_type);
    let total_bytes = model
        .size_bytes
        .saturating_add(kv_bytes.unwrap_or(0))
        .saturating_add(COMPUTE_BUFFER_ESTIMATE_BYTES);

    Ok(Json(json!({
        "available": true,
        "reason": null,
        "weights_bytes": model.size_bytes,
        "kv_bytes": kv_bytes.unwrap_or(0),
        "compute_bytes": COMPUTE_BUFFER_ESTIMATE_BYTES,
        "drafter_runtime_bytes": 0,
        "drafter_runtime_gpu_bytes": 0,
        "projector_runtime_bytes": 0,
        "drafter_kv_unsized": false,
        "adapters_unsized": false,
        "total_bytes": total_bytes,
        "gpu_bytes": total_bytes,
        "kv_estimable": kv_bytes.is_some(),
        "kv_on_gpu": true,
        "n_ctx": n_ctx,
        "cache_type_kv": cache_type_name,
        "n_parallel": n_parallel,
        "layer_count": summary.block_count,
        "gpu_layers": null,
        "moe_offload_unmodelled": false
    })))
}

#[derive(Debug, Deserialize)]
pub struct KvCacheQuery {
    repo_id: String,
    #[serde(default)]
    quant: Option<String>,
    #[serde(default)]
    n_ctx: Option<u64>,
    #[serde(default)]
    cache_type_kv: Option<String>,
}

/// `GET /api/models/kv-cache-estimate`: оценка для карточки модели в Hub.
/// Для нескачанной модели числа честно `null`.
pub async fn kv_cache_estimate(
    State(state): State<Arc<AppState>>,
    Query(query): Query<KvCacheQuery>,
) -> ApiResult<Json<Value>> {
    let local = match resolve_model(&state, &query.repo_id, query.quant.as_deref()).await? {
        ModelRef::Local(model) if model.complete => Some(model),
        _ => None,
    };
    let summary = match &local {
        Some(model) => summarize(model).await.ok().map(|(summary, _)| summary),
        None => None,
    };
    let (Some(model), Some(summary)) = (local, summary) else {
        return Ok(Json(json!({
            "kv_bytes": null,
            "weights_bytes": null,
            "native_context": null,
            "spec_bytes": null,
            "n_ctx": query.n_ctx,
            "projector_bytes": null,
            "spec_unpriced": false,
            "kv_checkpoint_bytes": null,
            "spec_fixed_bytes": null,
            "compute_bytes": null,
            "gpu_bytes": null,
            "total_bytes": null,
            "gpu_floor_bytes": null,
            "context_is_pinned": false,
            "inherited_device_pin": false
        })));
    };

    let (cache_type, _) = parse_cache_type(query.cache_type_kv.as_deref())?;
    let n_ctx = query
        .n_ctx
        .filter(|n| *n > 0)
        .or(summary.context_length)
        .unwrap_or(FALLBACK_CONTEXT_LENGTH);
    let kv_bytes = model_inventory::kv_cache_bytes(&summary, n_ctx, cache_type);
    let total_bytes = model
        .size_bytes
        .saturating_add(kv_bytes.unwrap_or(0))
        .saturating_add(COMPUTE_BUFFER_ESTIMATE_BYTES);
    Ok(Json(json!({
        "kv_bytes": kv_bytes,
        "weights_bytes": model.size_bytes,
        "native_context": summary.context_length,
        "spec_bytes": null,
        "n_ctx": n_ctx,
        "projector_bytes": null,
        "spec_unpriced": false,
        "kv_checkpoint_bytes": null,
        "spec_fixed_bytes": null,
        "compute_bytes": COMPUTE_BUFFER_ESTIMATE_BYTES,
        "gpu_bytes": total_bytes,
        "total_bytes": total_bytes,
        "gpu_floor_bytes": null,
        "context_is_pinned": false,
        "inherited_device_pin": false
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_parameter_counts() {
        assert_eq!(format_parameters(1_235_814_432), "1.24 млрд");
        assert_eq!(format_parameters(350_000_000), "350 млн");
    }

    #[test]
    fn parses_kv_cache_types() {
        assert_eq!(parse_cache_type(None).unwrap().0, GGMLType::F16);
        assert_eq!(parse_cache_type(Some("q8_0")).unwrap().0, GGMLType::Q8_0);
        assert!(parse_cache_type(Some("turbo")).is_err());
    }
}
