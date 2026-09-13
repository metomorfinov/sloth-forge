//! Хаб: скачанные GGUF-модели, их удаление, scan-folders и разделы датасетов.
//!
//! Раньше инвентарь смотрел только верхний уровень папки моделей (шарды во вложенных
//! папках не находились), модели сопоставлялись с репозиторием нечётким сравнением имён,
//! а оценка удаления подставляла «807 МБ» или «2,1 ГБ» по названию. Здесь всё берётся
//! из общего сканера `model_inventory`.

use crate::error::{ApiError, ApiResult};
use crate::model_inventory::{self, LocalModel};
use crate::state::AppState;
use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

/// Так контракт фронтенда называет выполнение GGUF-моделей; в SlothForge их выполняет
/// собственный Vulkan-движок, llama.cpp не используется.
const GGUF_RUNTIME: &str = "llama_cpp";
const TEXT_GENERATION_TASK: &str = "text-generation";
/// Недостающие файлы модели докачиваются обычной HTTP-загрузкой варианта.
const PARTIAL_TRANSPORT: &str = "http";

pub fn infer_repo_id_from_gguf_filename(filename: &str) -> String {
    let clean_filename = filename.strip_prefix("models/").unwrap_or(filename);
    let stem = clean_filename.trim_end_matches(".gguf");
    let lower = stem.to_lowercase();

    if lower.contains("llama-3.2-1b-instruct") {
        return "unsloth/Llama-3.2-1B-Instruct-GGUF".to_string();
    }
    if lower.contains("llama-3.2-3b-instruct") {
        return "unsloth/Llama-3.2-3B-Instruct-GGUF".to_string();
    }
    if lower.contains("qwen2.5-1.5b-instruct") {
        return "unsloth/Qwen2.5-1.5B-Instruct-GGUF".to_string();
    }
    if lower.contains("mistral-7b-instruct") {
        return "unsloth/Mistral-7B-Instruct-v0.3-GGUF".to_string();
    }

    let quant = extract_quant_from_path(clean_filename);
    let mut base = stem;
    if !quant.is_empty() {
        if let Some(s) = base.strip_suffix(&format!("-{}", quant)) {
            base = s;
        } else if let Some(s) = base.strip_suffix(&format!("_{}", quant)) {
            base = s;
        } else if let Some(s) = base.strip_suffix(&quant) {
            base = s.trim_end_matches(['-', '_']);
        }
    }

    if base.contains('/') {
        base.to_string()
    } else if base.ends_with("-GGUF") || base.ends_with("_GGUF") {
        format!("unsloth/{}", base)
    } else {
        format!("unsloth/{}-GGUF", base)
    }
}

/// Все GGUF-модели в папке моделей и scan-folders (обход диска — в отдельном потоке).
pub(crate) async fn local_models(state: &AppState) -> ApiResult<Vec<LocalModel>> {
    let roots = state.model_roots().await;
    tokio::task::spawn_blocking(move || model_inventory::scan_models(&roots))
        .await
        .map_err(|err| ApiError::internal(format!("Поиск моделей прерван: {err}")))
}

pub async fn resolve_local_gguf_file(state: &AppState, candidate: &str) -> Option<PathBuf> {
    let candidate = candidate.trim();
    if candidate.is_empty() {
        return None;
    }

    // Порядок попыток: путь как прислали, затем с «.gguf», затем только имя файла
    // (фронтенд иногда присылает «models/<файл>» или «org/repo/<файл>»)
    let mut attempts = vec![candidate.to_string()];
    let has_gguf_ext = candidate.to_ascii_lowercase().ends_with(".gguf");
    if !has_gguf_ext {
        attempts.push(format!("{candidate}.gguf"));
    }
    if let Some(filename) = std::path::Path::new(candidate).file_name().and_then(|name| name.to_str()) {
        if filename != candidate {
            attempts.push(filename.to_string());
            if !has_gguf_ext {
                attempts.push(format!("{filename}.gguf"));
            }
        }
    }

    // Все попытки проходят через песочницу: файл обязан лежать внутри папки моделей или scan-folders
    let roots = state.model_roots().await;
    attempts
        .iter()
        .find_map(|attempt| crate::paths::resolve_existing_file_within(&roots, attempt).ok())
}

fn gguf_capabilities() -> Value {
    json!({
        "can_train": true,
        "can_chat": true,
        "can_delete": true,
        "can_download": false,
        "requires_variant": false,
        "supports_lora": true,
        "supports_vision": false
    })
}

/// Строка «скачанная GGUF-модель» (`CachedGgufRepo` во фронтенде).
fn cached_gguf_row(model: &LocalModel) -> Value {
    let partial = !model.complete;
    json!({
        "repo_id": model.repo_id(),
        "inventory_id": model.id(),
        "load_id": model.id(),
        "model_format": "gguf",
        "runtime": GGUF_RUNTIME,
        "format_variant": model.quant(),
        "capabilities": gguf_capabilities(),
        "size_bytes": model.size_bytes,
        "cache_path": model.id(),
        "last_modified": model.modified_secs,
        "partial": partial,
        "partial_transport": partial.then_some(PARTIAL_TRANSPORT),
        "partial_resumable": partial,
        "pipeline_tag": TEXT_GENERATION_TASK,
        "task": TEXT_GENERATION_TASK
    })
}

/// Строка «локальная модель» (`LocalModelInfo` во фронтенде).
fn local_model_row(model: &LocalModel) -> Value {
    json!({
        "id": model.id(),
        "inventory_id": model.id(),
        "load_id": model.id(),
        "display_name": model.display_name,
        "path": model.id(),
        "size_bytes": model.size_bytes,
        "model_format": "gguf",
        "runtime": GGUF_RUNTIME,
        "format_variant": model.quant(),
        "capabilities": gguf_capabilities(),
        // Корень 0 — основная папка моделей, остальные — папки, добавленные пользователем
        "source": if model.root_index == 0 { "models_dir" } else { "custom" },
        "model_id": model.repo_id(),
        "updated_at": model.modified_secs,
        "partial": !model.complete,
        "pipeline_tag": TEXT_GENERATION_TASK,
        "task": TEXT_GENERATION_TASK
    })
}

/// `GET /api/hub/cached-models`: модели не в формате GGUF (safetensors из кэша HF).
/// SlothForge выполняет только GGUF, поэтому список пуст; раньше сюда подставлялся
/// тот же список GGUF-файлов, что и в cached-gguf.
pub async fn handle_cached_models() -> Json<Value> {
    Json(json!({ "cached": [], "scan_confirmed": true }))
}

/// `GET /api/hub/cached-gguf`: скачанные GGUF-модели.
pub async fn handle_cached_gguf(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let models = local_models(&state).await?;
    let total_size_bytes: u64 = models.iter().map(|model| model.size_bytes).sum();
    Ok(Json(json!({
        "cached": models.iter().map(cached_gguf_row).collect::<Vec<_>>(),
        "total_size_bytes": total_size_bytes,
        "scan_confirmed": true
    })))
}

/// `GET /api/hub/local`: модели в папке моделей и добавленных папках.
pub async fn handle_hub_local(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let models = local_models(&state).await?;
    Ok(Json(json!({
        "models_dir": state.models_dir.to_string_lossy(),
        "hf_cache_dir": null,
        "lmstudio_dirs": [],
        "ollama_dirs": [],
        "hermes_dirs": [],
        "count": models.len(),
        "models": models.iter().map(local_model_row).collect::<Vec<_>>()
    })))
}

pub async fn handle_hidden_models() -> Json<Value> {
    Json(json!({
        "hidden_models": []
    }))
}

/// Known quant patterns in descending specificity
const KNOWN_QUANTS: &[&str] = &[
    // Unsloth Dynamic (UD) variants
    "UD-Q8_K_XL", "UD-Q6_K_XL", "UD-Q5_K_XL", "UD-Q4_K_XL", "UD-Q3_K_XL", "UD-Q2_K_XL",
    "UD-IQ4_XS", "UD-IQ4_NL", "UD-IQ3_XXS", "UD-IQ3_S", "UD-IQ2_XXS", "UD-IQ2_M", "UD-IQ1_S", "UD-IQ1_M",
    // Standard K-quants & legacy
    "Q8_K_XL", "Q6_K_XL", "Q5_K_XL", "Q4_K_XL", "Q3_K_XL", "Q2_K_XL",
    "Q8_0", "Q8_1",
    "Q6_K_L", "Q6_K_M", "Q6_K",
    "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0",
    "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0",
    "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q3_1", "Q3_0",
    "Q2_K_L", "Q2_K", "Q2_0",
    // Importance quants (IQ)
    "IQ4_NL", "IQ4_XS", "IQ3_M", "IQ3_S", "IQ3_XXS", "IQ2_M", "IQ2_S", "IQ2_XXS", "IQ1_M", "IQ1_S",
    // Unquantized / Float
    "BF16", "FP16", "F16", "FP32", "F32",
];

/// Returns true if a path refers to auxiliary files that are NOT the base model weights
/// (e.g. MTP speculative decoding heads, vision projector, draft models, VAE, etc.)
pub(crate) fn is_auxiliary_gguf_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    let filename = path.rsplit('/').next().unwrap_or(path).to_lowercase();

    // MTP (Multi-Token Prediction) speculative decoding auxiliary weights
    if lower.starts_with("mtp/")
        || lower.contains("/mtp/")
        || lower.contains("/mtp-")
        || lower.contains("/mtp_")
        || filename.starts_with("mtp-")
        || filename.starts_with("mtp_")
        || filename.contains("-mtp-")
        || filename.contains("_mtp_")
    {
        return true;
    }

    // Vision projector (handled separately to set has_vision = true)
    if lower.contains("mmproj") {
        return true;
    }

    // Speculative decoding draft models
    if lower.contains("draft") {
        return true;
    }

    // Diffusion VAE and text encoders
    if lower.contains("vae") || lower.contains("text_encoder") {
        return true;
    }

    // Standalone adapters or lora weights inside a base model repo
    if lower.contains("adapter") {
        return true;
    }

    false
}

/// Extracts the clean quant variant name from a GGUF file path
pub fn extract_quant_from_path(p: &str) -> String {
    let parts: Vec<&str> = p.split('/').collect();

    // 1. If file is organized in a subfolder, check if the folder name specifies the quant
    if parts.len() > 1 {
        let folder = parts[0];
        // Check exact match with known quants first (e.g. "UD-Q4_K_XL", "Q8_0", "BF16")
        for &q in KNOWN_QUANTS {
            if folder.eq_ignore_ascii_case(q) {
                return q.to_string();
            }
        }
        // Check if folder contains a known quant (e.g. "DeepSeek-R1-Q4_K_M")
        for &q in KNOWN_QUANTS {
            if folder.contains(q) {
                return q.to_string();
            }
        }
    }

    // 2. Check filename for known quant
    let filename = parts.last().unwrap_or(&p);
    for &q in KNOWN_QUANTS {
        if filename.contains(q) {
            return q.to_string();
        }
    }

    // 3. Fallback: use folder name if present, else filename stem
    if parts.len() > 1 {
        parts[0].to_string()
    } else {
        filename.trim_end_matches(".gguf").to_string()
    }
}

// Загрузка моделей, её прогресс, отмена и transport-status — в downloads.rs,
// варианты GGUF — в gguf_variants.rs, проверка токена — в hf_api.rs

pub async fn handle_datasets_transport_status() -> Json<Value> {
    Json(json!({
        "status": "idle",
        "transport": "direct",
        "active": false,
        "has_partial": false,
        "last_transport": null,
        "resumable": false
    }))
}

pub async fn handle_datasets_cached() -> Json<Value> {
    Json(json!({
        "cached": []
    }))
}

/// Датасеты появятся вместе с настоящим обучением (этап 3 дорожной карты).
/// До этого изменяющие обработчики честно отвечают 501 вместо фальшивого «успеха».
const DATASETS_FEATURE: &str = "Датасеты";
const DATASETS_STAGE: u8 = 3;

pub async fn handle_datasets_cached_delete() -> ApiError {
    crate::unavailable::not_ready(DATASETS_FEATURE, DATASETS_STAGE)
}

pub async fn handle_datasets_local() -> Json<Value> {
    Json(json!({
        "datasets": []
    }))
}

pub async fn handle_datasets_active_downloads() -> Json<Value> {
    Json(json!({
        "downloads": []
    }))
}

pub async fn handle_datasets_download() -> ApiError {
    // Раньше отвечало state: "running", но загрузка не начиналась
    crate::unavailable::not_ready(DATASETS_FEATURE, DATASETS_STAGE)
}

pub async fn handle_datasets_download_cancel() -> ApiError {
    crate::unavailable::not_ready(DATASETS_FEATURE, DATASETS_STAGE)
}

pub async fn handle_datasets_download_status() -> Json<Value> {
    // Загрузок датасетов не бывает, поэтому и поколения задания нет
    Json(json!({
        "state": "idle",
        "error": null,
        "generation": null
    }))
}

pub async fn handle_datasets_download_progress() -> Json<Value> {
    Json(json!({
        "downloaded_bytes": 0,
        "completed_bytes": 0,
        "complete_on_disk": false,
        "expected_bytes": 0,
        "progress": 0.0,
        "cache_path": null,
        "target_present": false,
        "cache_measured": true
    }))
}

pub async fn handle_datasets_local_options() -> Json<Value> {
    Json(json!({
        "options": []
    }))
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

/// Модель относится к репозиторию (и варианту): по папке `org/repo`, имени файла или пути.
fn matches_repo_variant(model: &LocalModel, repo: &str, variant: Option<&str>) -> bool {
    let repo = repo.trim();
    let same_repo = model.repo_id().eq_ignore_ascii_case(repo)
        || model.file_name.eq_ignore_ascii_case(repo)
        || model.id() == repo;
    same_repo && variant.is_none_or(|variant| model.quant().eq_ignore_ascii_case(variant))
}

/// Недокачанные `.part`-файлы варианта в папке репозитория: путь и размер.
async fn repo_part_files(
    state: &AppState,
    repo_id: &str,
    variant: Option<&str>,
) -> ApiResult<Vec<(PathBuf, u64)>> {
    let Ok(repo_id) = crate::downloads::validate_repo_id(repo_id) else {
        return Ok(Vec::new());
    };
    let dir = state.models_dir.join(repo_id);
    let variant = variant.map(str::to_string);
    tokio::task::spawn_blocking(move || crate::downloads::part_files(&dir, variant.as_deref()))
        .await
        .map_err(|err| ApiError::internal(format!("Поиск недокачанных файлов прерван: {err}")))
}

fn ensure_not_downloading(state: &AppState, repo_id: &str, variant: Option<&str>) -> ApiResult<()> {
    if state.downloads.is_downloading(repo_id, variant) {
        return Err(ApiError::conflict(
            "Эта модель сейчас скачивается: отмените загрузку, прежде чем удалять файлы",
        ));
    }
    Ok(())
}

/// Удалять можно только файлы моделей и их недокачанные части.
fn is_model_file(path: &std::path::Path) -> bool {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    name.ends_with(".gguf") || name.ends_with(".gguf.part")
}

#[derive(Debug, Default, Deserialize)]
pub struct DeleteCachedRequest {
    #[serde(default)]
    repo_id: Option<String>,
    #[serde(default)]
    variant: Option<String>,
    #[serde(default)]
    cache_path: Option<String>,
}

/// `DELETE /api/hub/delete-cached`: удаляет модель (все шарды) или вариант репозитория.
pub async fn handle_delete_cached(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DeleteCachedRequest>,
) -> ApiResult<Json<Value>> {
    let repo_id = non_empty(request.repo_id.as_deref());
    let variant = non_empty(request.variant.as_deref());
    let roots = state.model_roots().await;
    let models = local_models(&state).await?;

    let mut targets: Vec<PathBuf> = Vec::new();
    if let Some(cache_path) = non_empty(request.cache_path.as_deref()) {
        let path = crate::paths::resolve_existing_file_within(&roots, cache_path)
            .map_err(|rejection| ApiError::from_path_rejection(rejection, "cache_path"))?;
        match models
            .iter()
            .find(|model| model.path == path || model.shard_paths.contains(&path))
        {
            Some(model) => {
                ensure_not_downloading(&state, &model.repo_id(), Some(&model.quant()))?;
                // Модель из нескольких файлов удаляется целиком: без любого шарда она не загрузится
                targets.extend(model.shard_paths.iter().cloned());
            }
            None => targets.push(path),
        }
    } else {
        // Без repo_id вариант вроде «Q4_K_M» совпал бы с файлами разных моделей
        let repo = repo_id.ok_or_else(|| ApiError::bad_request("Укажите cache_path или repo_id"))?;
        ensure_not_downloading(&state, repo, variant)?;
        for model in models
            .iter()
            .filter(|model| matches_repo_variant(model, repo, variant))
        {
            targets.extend(model.shard_paths.iter().cloned());
        }
        targets.extend(
            repo_part_files(&state, repo, variant)
                .await?
                .into_iter()
                .map(|(path, _)| path),
        );
    }

    targets.sort();
    targets.dedup();
    if targets.is_empty() {
        return Err(ApiError::not_found("Локальные файлы этой модели не найдены"));
    }
    if !targets.iter().all(|path| is_model_file(path)) {
        return Err(ApiError::forbidden("Удалять можно только файлы моделей .gguf"));
    }
    // Каждый путь ещё раз проходит песочницу: удалить что-либо вне папок моделей невозможно
    for target in &targets {
        crate::paths::resolve_existing_file_within(&roots, &target.to_string_lossy())
            .map_err(|rejection| ApiError::from_path_rejection(rejection, "файл модели"))?;
    }

    let mut deleted_files = Vec::with_capacity(targets.len());
    for target in &targets {
        tokio::fs::remove_file(target).await.map_err(|err| {
            ApiError::internal(format!("Не удалось удалить {}: {err}", target.display()))
        })?;
        deleted_files.push(target.to_string_lossy().to_string());
    }
    info!(
        "Удалены файлы модели repo_id={:?}, variant={:?}: {:?}",
        repo_id, variant, deleted_files
    );

    Ok(Json(json!({
        "status": "ok",
        "deleted": true,
        "deleted_files": deleted_files
    })))
}

pub async fn handle_hub_scan_folders(
    State(state): State<Arc<AppState>>,
) -> Json<Value> {
    let folders = state.scan_folders.read().await;
    Json(json!({
        "folders": *folders
    }))
}

pub async fn handle_hub_add_scan_folder(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let path = payload
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let mut folders = state.scan_folders.write().await;
    let next_id = (folders.len() as u64) + 1;
    let entry = crate::state::ScanFolderEntry {
        id: next_id,
        path,
        created_at: crate::state::iso_now(),
        status: Some("ok".to_string()),
    };
    folders.push(entry.clone());
    Json(serde_json::to_value(entry).unwrap_or(json!({
        "id": next_id,
        "status": "ok"
    })))
}

pub async fn handle_hub_delete_scan_folder(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Json<Value> {
    let mut folders = state.scan_folders.write().await;
    folders.retain(|f| f.id != id);
    Json(json!({
        "status": "ok",
        "deleted": true
    }))
}

#[derive(Debug, Default, Deserialize)]
pub struct DeleteImpactRequest {
    #[serde(default)]
    repo_id: Option<String>,
    #[serde(default)]
    variant: Option<String>,
}

/// Сколько места освободит удаление: те же файлы, что удалит `delete-cached`.
async fn delete_impact(state: &AppState, request: DeleteImpactRequest) -> ApiResult<Json<Value>> {
    let repo_id = non_empty(request.repo_id.as_deref())
        .ok_or_else(|| ApiError::bad_request("Не указан repo_id"))?;
    let variant = non_empty(request.variant.as_deref());
    let models = local_models(state).await?;
    let affected: Vec<&LocalModel> = models
        .iter()
        .filter(|model| matches_repo_variant(model, repo_id, variant))
        .collect();
    let part_bytes: u64 = repo_part_files(state, repo_id, variant)
        .await?
        .iter()
        .map(|(_, size)| size)
        .sum();
    let reclaimed_bytes = affected
        .iter()
        .map(|model| model.size_bytes)
        .sum::<u64>()
        .saturating_add(part_bytes);

    Ok(Json(json!({
        "repo_id": repo_id,
        "variant": variant,
        "reclaimed_bytes": reclaimed_bytes,
        "freed_bytes": reclaimed_bytes,
        "affected_models": affected.iter().map(|model| model.id()).collect::<Vec<_>>(),
        // Сопутствующих файлов (энкодеров, VAE) у GGUF-моделей SlothForge нет
        "retained_companions": [],
        "freeable_companions": [],
        "blocked_by": []
    })))
}

pub async fn handle_hub_delete_impact(
    State(state): State<Arc<AppState>>,
    Query(request): Query<DeleteImpactRequest>,
) -> ApiResult<Json<Value>> {
    delete_impact(&state, request).await
}

pub async fn handle_hub_delete_impact_post(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DeleteImpactRequest>,
) -> ApiResult<Json<Value>> {
    delete_impact(&state, request).await
}

pub async fn handle_hub_orphan_companions() -> Json<Value> {
    Json(json!({
        "orphans": [],
        "companions": [],
        "total_bytes": 0
    }))
}

pub async fn handle_datasets_check_format() -> ApiError {
    // Раньше для любого файла отвечало «alpaca, 1000 строк», не читая его
    crate::unavailable::not_ready(DATASETS_FEATURE, DATASETS_STAGE)
}

pub async fn handle_datasets_upload() -> ApiError {
    // Раньше не читало тело запроса и возвращало путь к несуществующему файлу
    crate::unavailable::not_ready(DATASETS_FEATURE, DATASETS_STAGE)
}

pub async fn handle_datasets_ai_assist_mapping() -> ApiError {
    crate::unavailable::not_ready(DATASETS_FEATURE, DATASETS_STAGE)
}
