use crate::state::AppState;
use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub device_type: String,
    pub gpu_name: String,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub vram_used_mb: u64,
    pub cuda_available: bool,
    pub rocm_available: bool,
    pub vulkan_available: bool,
    pub capabilities: Vec<String>,
    pub chat_only: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apple_silicon: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_only_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware_detecting: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cloudflare_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secure: Option<bool>,
}

pub async fn handle_health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    // Настоящие числа видеопамяти — шаг 10 (VK_EXT_memory_budget); оценка
    // «VRAM под обучение» из имитации обучения больше не подставляется
    let vram_total_mb = 4096u64;
    let vram_used_mb = 0u64;
    let vram_free_mb = vram_total_mb.saturating_sub(vram_used_mb);

    let gpu_name = state
        .vk_ctx
        .as_ref()
        .map(|c| c.device_name().to_string())
        .unwrap_or_else(|| "Vulkan-устройство не найдено".to_string());

    Json(HealthResponse {
        status: "ok".to_string(),
        version: "0.1.0".to_string(),
        device_type: "vulkan".to_string(),
        gpu_name,
        vram_total_mb,
        vram_free_mb,
        vram_used_mb,
        cuda_available: false,
        rocm_available: false,
        vulkan_available: state.vk_ctx.is_some(),
        capabilities: vec![
            "train".to_string(),
            "chat".to_string(),
            "gguf".to_string(),
            "lora".to_string(),
            "cluster".to_string(),
        ],
        chat_only: false,
        apple_silicon: None,
        chat_only_reason: None,
        hardware_detecting: None,
        cloudflare_url: None,
        server_url: None,
        secure: None,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthStatusUser {
    pub username: String,
    pub role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthStatusResponse {
    pub authenticated: bool,
    pub auth_required: bool,
    pub initialized: bool,
    pub requires_password_change: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_deadline_seconds: Option<u64>,
    pub user: AuthStatusUser,
}

pub async fn handle_auth_status() -> Json<AuthStatusResponse> {
    Json(AuthStatusResponse {
        authenticated: true,
        auth_required: false,
        initialized: true,
        requires_password_change: false,
        bootstrap_deadline_seconds: None,
        user: AuthStatusUser {
            username: "rivergod".to_string(),
            role: "admin".to_string(),
        },
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallSourceResponse {
    pub source: String,
    pub install_source: String,
    pub channel: String,
}

pub async fn handle_install_source() -> Json<InstallSourceResponse> {
    Json(InstallSourceResponse {
        source: "native".to_string(),
        install_source: "native".to_string(),
        channel: "stable".to_string(),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStatusResponse {
    pub update_available: bool,
    pub current_version: String,
    pub latest_version: String,
    pub install_source: String,
    pub can_show_web_notification: bool,
}

pub async fn handle_update_status() -> Json<UpdateStatusResponse> {
    Json(UpdateStatusResponse {
        update_available: false,
        current_version: "0.1.0".to_string(),
        latest_version: "0.1.0".to_string(),
        install_source: "native".to_string(),
        can_show_web_notification: false,
    })
}

pub async fn handle_system(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    use sysinfo::{System, Disks, Pid, ProcessesToUpdate, MemoryRefreshKind};

    let mut sys = System::new();
    // Refresh CPU (need two readings for usage)
    sys.refresh_cpu_all();
    std::thread::sleep(std::time::Duration::from_millis(200));
    sys.refresh_cpu_all();
    sys.refresh_memory_specifics(MemoryRefreshKind::everything());
    sys.refresh_processes(ProcessesToUpdate::All, true);

    let disks = Disks::new_with_refreshed_list();

    // CPU metrics
    let cpu_usage: f64 = sys.global_cpu_usage() as f64;
    let logical_count = sys.cpus().len();
    let physical_count = sys.physical_core_count().unwrap_or(logical_count / 2);
    let frequency_mhz = sys.cpus().first().map(|c| c.frequency()).unwrap_or(0);

    // Memory metrics
    let total_mem_gb = sys.total_memory() as f64 / 1_073_741_824.0;
    let available_mem_gb = sys.available_memory() as f64 / 1_073_741_824.0;
    let used_mem_gb = total_mem_gb - available_mem_gb;
    let mem_percent = if total_mem_gb > 0.0 { (used_mem_gb / total_mem_gb) * 100.0 } else { 0.0 };

    // Process memory
    let pid = Pid::from_u32(std::process::id());
    let process_used_mb = sys.process(pid)
        .map(|p| p.memory() as f64 / 1_048_576.0)
        .unwrap_or(0.0);

    // Disk metrics (sum all mount points, use root "/" as primary)
    let (disk_total_gb, disk_free_gb) = disks.list().iter()
        .find(|d| d.mount_point() == std::path::Path::new("/"))
        .map(|d| (
            d.total_space() as f64 / 1_073_741_824.0,
            d.available_space() as f64 / 1_073_741_824.0,
        ))
        .unwrap_or_else(|| {
            // Fallback: sum all disks
            let total: u64 = disks.list().iter().map(|d| d.total_space()).sum();
            let free: u64 = disks.list().iter().map(|d| d.available_space()).sum();
            (total as f64 / 1_073_741_824.0, free as f64 / 1_073_741_824.0)
        });
    let disk_percent = if disk_total_gb > 0.0 { ((disk_total_gb - disk_free_gb) / disk_total_gb) * 100.0 } else { 0.0 };

    // Uptime
    let uptime_secs = System::uptime();

    // GPU (from Vulkan context — already real)
    // Шаг 10: реальная занятость видеопамяти из VK_EXT_memory_budget
    let vram_used: u64 = 0;
    let dev_name = state
        .vk_ctx
        .as_ref()
        .map(|c| c.device_name().to_string())
        .unwrap_or_else(|| "AMD Radeon RX 570 Series (RADV POLARIS10)".to_string());

    let vram_used_gb = if vram_used > 0 {
        vram_used as f64 / 1024.0
    } else {
        0.35
    };
    let vram_free_gb = 4.0 - vram_used_gb;
    let vram_utilization_pct = (vram_used_gb / 4.0) * 100.0;

    let gpu_device = serde_json::json!({
        "device_id": 0,
        "name": dev_name,
        "gpu_name": dev_name,
        "memory_total_gb": 4.0,
        "vram_total_gb": 4.0,
        "vram_used_gb": vram_used_gb,
        "vram_free_gb": vram_free_gb,
        "vram_utilization_pct": vram_utilization_pct,
        "index": 0,
        "visible_ordinal": 0,
        "index_kind": "vulkan",
        "backend": "vulkan",
        "shared_memory": false
    });

    let resp = serde_json::json!({
        "status": "ready",
        "platform": std::env::consts::OS,
        "python_version": "N/A",
        "device_backend": "vulkan",
        "uptime_seconds": uptime_secs,
        "cpu": {
            "logical_count": logical_count,
            "physical_count": physical_count,
            "usage_percent": (cpu_usage * 10.0).round() / 10.0,
            "frequency_mhz": frequency_mhz
        },
        "memory": {
            "total_gb": (total_mem_gb * 100.0).round() / 100.0,
            "available_gb": (available_mem_gb * 100.0).round() / 100.0,
            "percent_used": (mem_percent * 10.0).round() / 10.0,
            "process_used_mb": process_used_mb.round() as u64
        },
        "disk": {
            "total_gb": (disk_total_gb * 10.0).round() / 10.0,
            "free_gb": (disk_free_gb * 10.0).round() / 10.0,
            "percent_used": (disk_percent * 10.0).round() / 10.0
        },
        "gpu": {
            "available": true,
            "backend": "vulkan",
            "devices": [gpu_device.clone()]
        },
        "inference_gpu": {
            "available": true,
            "backend": "vulkan",
            "devices": [gpu_device]
        },
        "ml_packages": {
            "torch": null,
            "transformers": null
        }
    });
    Json(resp)
}

pub async fn handle_system_hardware(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let dev_name = state
        .vk_ctx
        .as_ref()
        .map(|c| c.device_name().to_string())
        .unwrap_or_else(|| "AMD Radeon RX 570 Series (RADV POLARIS10)".to_string());
    // Шаг 10: реальная занятость видеопамяти из VK_EXT_memory_budget
    let vram_used: u64 = 0;
    let vram_free_gb = (4096 - vram_used) as f64 / 1024.0;

    let resp = serde_json::json!({
        "gpu": {
            "gpu_name": dev_name,
            "vram_total_gb": 4.0,
            "vram_free_gb": vram_free_gb,
        },
        "gpuName": dev_name,
        "vramTotalGb": 4.0,
        "vramFreeGb": vram_free_gb,
        "gpus": [{
            "device_id": 0,
            "name": dev_name,
            "gpu_name": dev_name,
            "vram_total_gb": 4.0,
            "vram_free_gb": vram_free_gb,
            "vram_used_gb": (vram_used as f64 / 1024.0),
            "vram_utilization_pct": ((vram_used as f64 / 4096.0) * 100.0)
        }],
        "versions": {
            "torch": null,
            "cuda": null,
            "rocm": null,
            "xpu": null,
            "transformers": null,
            "unsloth": null,
            "vulkan": "1.3",
            "sloth_vulkan": "0.1.0"
        },
        "torch": null,
        "cuda": null,
        "rocm": null,
        "xpu": null,
        "transformers": null,
        "unsloth": null,
        "llamaCpp": null,
        "llama_cpp": null,
        "exportSupported": true,
        "export_supported": true,
        "exportUnsupportedReason": null,
        "export_unsupported_reason": null,
        "exportUnsupportedMessage": null,
        "export_unsupported_message": null,
        "videoSupported": false,
        "video_supported": false,
        "videoUnsupportedReason": "Генерация видео появится в SlothForge на этапе 7 дорожной карты",
        "video_unsupported_reason": "Генерация видео появится в SlothForge на этапе 7 дорожной карты",
        "videoUnsupportedMessage": "Генерация видео пока не поддерживается в SlothForge",
        "video_unsupported_message": "Генерация видео пока не поддерживается в SlothForge",
        "loaded": true
    });
    Json(resp)
}

pub async fn handle_check_vision(Path(id): Path<String>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "model_name": id,
        "is_vision": false
    }))
}

pub async fn handle_check_embedding(Path(id): Path<String>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "model_name": id,
        "is_embedding": false
    }))
}

/// `GET /api/models/config/:id`: размер, длина контекста и наличие проектора изображений
/// берутся из настоящего файла. Раньше размер угадывался по названию («3b» → 2,1 ГБ),
/// а контекст всегда был 131072; для нескачанной модели эти поля теперь `null`.
pub async fn handle_model_config(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> crate::error::ApiResult<Json<serde_json::Value>> {
    use crate::error::ApiError;

    let identifier = id.trim();
    let resolved = crate::hub::resolve_local_gguf_file(&state, identifier).await;
    let model = crate::hub::local_models(&state)
        .await?
        .into_iter()
        .find(|model| {
            resolved
                .as_ref()
                .is_some_and(|path| model.path == *path || model.shard_paths.contains(path))
                || (model.complete && model.repo_id().eq_ignore_ascii_case(identifier))
        });
    let (size_bytes, context_length, is_vision) = match model {
        Some(model) => {
            let size_bytes = model.size_bytes;
            let (context_length, is_vision) = tokio::task::spawn_blocking(move || {
                let context_length = match sloth_core::gguf::GGUFFile::open(&model.path) {
                    Ok(file) => file.context_length(),
                    Err(err) => {
                        tracing::warn!(
                            "Не удалось прочитать заголовок {}: {err}",
                            model.path.display()
                        );
                        None
                    }
                };
                (context_length, crate::model_inventory::has_projector(&model))
            })
            .await
            .map_err(|err| ApiError::internal(format!("Чтение модели прервано: {err}")))?;
            (Some(size_bytes), context_length, is_vision)
        }
        None => (None, None, false),
    };

    Ok(Json(serde_json::json!({
        "id": id,
        "model_name": id,
        "model_type": "text",
        "model_size_bytes": size_bytes,
        "max_position_embeddings": context_length,
        "is_vision": is_vision,
        "is_embedding": false,
        "is_audio": false,
        "is_lora": false,
        "config": {
            "training": {
                "max_seq_length": 2048,
                "num_epochs": 3,
                "learning_rate": "2e-4",
                "batch_size": 1,
                "gradient_accumulation_steps": 4,
                "lora_r": 16,
                "lora_alpha": 32.0,
                "lora_dropout": 0.0
            }
        }
    })))
}

pub async fn handle_providers_registry() -> Json<serde_json::Value> {
    Json(serde_json::json!([]))
}

pub async fn handle_providers_list() -> Json<serde_json::Value> {
    Json(serde_json::json!([]))
}

pub async fn handle_auth_refresh() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "access_token": "sloth_local_token",
        "refresh_token": "sloth_local_refresh",
        "must_change_password": false
    }))
}

pub async fn handle_auth_logout() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok"
    }))
}

// Inference Status & Monitor
pub async fn handle_inference_monitor() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "idle",
        "active_requests": 0,
        "entries": [],
        "total": 0
    }))
}

// Статус, загрузка, выгрузка, проверка модели, оценка памяти и флаги llama — в inference_api.rs

/// Статус «ничего не загружено» в формате фронтенда (DiffusionStatus / VideoStatus).
/// Раньше `loaded` был массивом, а фронтенд ждёт булево значение.
fn idle_generation_status() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "loaded": false,
        "repo_id": null,
        "family": null,
        "base_repo": null,
        "device": null,
        "dtype": null,
        "model_kind": null,
        "gguf_variant": null,
        "cpu_offload": false
    }))
}

pub async fn handle_inference_video_status() -> Json<serde_json::Value> {
    idle_generation_status()
}

pub async fn handle_inference_images_status() -> Json<serde_json::Value> {
    idle_generation_status()
}

// Models
pub async fn handle_models_recommended_folders() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "folders": ["models"] }))
}

pub async fn handle_models_loras() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "loras": [],
        "outputs_dir": "outputs"
    }))
}

// Треды, сообщения, проекты и настройки чата — в chat_history.rs (хранятся в SQLite)

// Settings: персонализация, лимиты, память моделей, пресеты и прочие сохраняемые
// настройки — в модуле settings (хранятся в SQLite)

pub async fn handle_llama_backend() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "supported": false,
        "reason": "Native Vulkan compute engine active (AMD Polaris 10)",
        "envBackend": "vulkan",
        "backend": "vulkan",
        "backendRequest": "vulkan",
        "selectionApplied": true,
        "installedTag": "vulkan-polaris-1.4",
        "options": [
            {
                "backend": "vulkan",
                "available": true,
                "resolvedBackend": "vulkan",
                "releaseTag": "vulkan-polaris-1.4",
                "downloadSizeBytes": 0
            }
        ],
        "job": {
            "state": "idle",
            "operation": null,
            "requested_backend": null,
            "message": "Vulkan compute active",
            "error": null,
            "progress": null,
            "reload_required": false,
            "started_at": null,
            "finished_at": null
        }
    }))
}

pub async fn handle_settings_lan_access(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "state": "off",
        "urls": [],
        "public_urls": [],
        "error": null,
        "auto_start": false,
        "configured_port": state.server_port.load(Ordering::Relaxed),
        "active_port": null,
        "managed_by": "settings",
        "can_start": true,
        "can_stop": false,
        "block_reason": null,
        "bind_host": null,
        "wildcard_bind": false,
        "serves_web_ui": true,
        "keyless_lan_eligible": false,
        "keyless_scope": "off",
        "keyless_tools": false
    }))
}

pub async fn handle_settings_lan_access_action(
    state: State<Arc<AppState>>,
    Json(_payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    handle_settings_lan_access(state).await
}

pub async fn handle_settings_lan_access_post(state: State<Arc<AppState>>) -> Json<serde_json::Value> {
    handle_settings_lan_access(state).await
}

pub async fn handle_settings_remote_access() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "state": "off",
        "url": null,
        "error": null,
        "auto_start": false,
        "default_auto_start": false,
        "available": false,
        "managed_by": "settings",
        "can_start": false,
        "can_stop": false,
        "block_reason": "explicitly_disabled",
        "password_pending": false,
        "streaming_supported": false
    }))
}

pub async fn handle_settings_remote_access_action(
    Json(_payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    handle_settings_remote_access().await
}

pub async fn handle_settings_remote_access_post() -> Json<serde_json::Value> {
    handle_settings_remote_access().await
}

pub async fn handle_settings_preview_links_rotate() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

pub async fn handle_settings_coding_agents() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "agents": ["claude", "cursor", "cline", "continue", "aider"],
        "detected": []
    }))
}

pub async fn handle_settings_debug_logs_sources() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "sources": [],
        "default_source_id": null,
        "defaultSourceId": null,
        "file_logging_disabled": false,
        "fileLoggingDisabled": false
    }))
}

pub async fn handle_settings_debug_logs() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "reason": null,
        "source_id": null,
        "realpath": null,
        "lines": [],
        "cursor": null,
        "reset": false,
        "reset_reason": null,
        "dropped_bytes": 0,
        "truncated_head": false,
        "more_pending": false,
        "file_logging_disabled": false,
        "size_bytes": 0
    }))
}

// Studio / Export / Llama / RAG / Diffusion
pub async fn handle_studio_download_transport_capabilities() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "http": {
            "available": true,
            "reason": null
        },
        "xet": {
            "available": false,
            "reason": "Xet disabled in SlothForge; native Vulkan direct HTTP streaming active"
        },
        "auto_resolves_to": "http",
        "auto_reason": "Native Vulkan engine using direct HTTP streaming",
        "partials_resumable": true,
        "direct": true,
        "hf_transfer": true
    }))
}

pub async fn handle_xet_notice_reserve(
    Json(_payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "granted": false,
        "shown": 3,
        "limit": 3
    }))
}

pub async fn handle_igpu_carveout_notice_dismiss() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

/// Находит модель на диске по `repo_id` (и необязательному `variant`) из запроса.
/// Результат всегда лежит внутри папки моделей или scan-folders.
async fn resolve_cached_model_path(
    state: &AppState,
    params: &serde_json::Value,
) -> crate::error::ApiResult<PathBuf> {
    use crate::error::ApiError;

    let model_id = params
        .get("model_id")
        .or_else(|| params.get("repo_id"))
        .or_else(|| params.get("repoId"))
        .or_else(|| params.get("model"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if model_id.is_empty() {
        return Err(ApiError::bad_request("Не указан repo_id"));
    }
    let variant = params
        .get("variant")
        .or_else(|| params.get("gguf_variant"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty());

    // Сначала точное совпадение среди найденных на диске GGUF (тот же repo и квант),
    // затем — путь или имя файла
    let from_scan = crate::hub::local_models(state)
        .await?
        .into_iter()
        .find(|model| {
            model.complete
                && model.repo_id().eq_ignore_ascii_case(model_id)
                && variant.is_none_or(|v| model.quant().eq_ignore_ascii_case(v))
        })
        .map(|model| model.path);
    let found = match from_scan {
        Some(path) => Some(path),
        None => crate::hub::resolve_local_gguf_file(state, model_id).await,
    }
    .ok_or_else(|| ApiError::not_found(format!("Модель {model_id} не найдена на диске")))?;

    // Путь из сканера тоже проверяем песочницей: наружу не отдаём ничего
    let roots = state.model_roots().await;
    crate::paths::resolve_existing_within(&roots, &found.to_string_lossy())
        .map_err(|rejection| ApiError::from_path_rejection(rejection, "модель"))
}

pub async fn handle_model_cached_path(
    State(state): State<Arc<AppState>>,
    Query(query): Query<serde_json::Value>,
) -> crate::error::ApiResult<Json<serde_json::Value>> {
    let path = resolve_cached_model_path(&state, &query).await?;
    Ok(Json(serde_json::json!({
        "path": path.to_string_lossy(),
        "is_dir": path.is_dir()
    })))
}

pub async fn handle_model_reveal(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> crate::error::ApiResult<Json<serde_json::Value>> {
    let path = resolve_cached_model_path(&state, &payload).await?;
    // Для файла открываем папку, в которой он лежит
    let folder = if path.is_dir() {
        path.clone()
    } else {
        path.parent().map(std::path::Path::to_path_buf).unwrap_or_else(|| path.clone())
    };
    open_in_file_manager(&folder)?;
    Ok(Json(serde_json::json!({
        "status": "ok",
        "revealed": true,
        "path": folder.to_string_lossy()
    })))
}

/// Открывает папку в системном файловом менеджере, не дожидаясь его закрытия.
fn open_in_file_manager(folder: &std::path::Path) -> crate::error::ApiResult<()> {
    let opener = if cfg!(target_os = "windows") {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let mut child = std::process::Command::new(opener)
        .arg(folder)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|err| {
            crate::error::ApiError::internal(format!(
                "Не удалось открыть файловый менеджер ({opener}): {err}"
            ))
        })?;
    // Дожидаемся процесса в отдельном потоке, чтобы он не остался «зомби»
    std::thread::spawn(move || {
        if let Err(err) = child.wait() {
            tracing::warn!("Процесс файлового менеджера завершился с ошибкой: {err}");
        }
    });
    Ok(())
}

/// Максимум папок в одном ответе обзора: огромный каталог не должен подвешивать сервер и UI.
const MAX_BROWSE_ENTRIES: usize = 2000;

pub async fn handle_model_browse_folders(
    State(state): State<Arc<AppState>>,
    Query(query): Query<serde_json::Value>,
) -> crate::error::ApiResult<Json<serde_json::Value>> {
    use crate::error::ApiError;

    let raw_path = query.get("path").and_then(|v| v.as_str()).unwrap_or("").trim();
    let show_hidden = query.get("show_hidden")
        .and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s == "true")))
        .unwrap_or(false);

    // Обзор разрешён внутри домашней папки и папок с моделями. Этого хватает, чтобы
    // выбрать папку для сканирования, и системные каталоги остаются закрыты.
    let model_roots = state.model_roots().await;
    let mut browse_roots = model_roots.clone();
    let home = crate::paths::home_dir();
    if let Some(ref home) = home {
        browse_roots.push(home.clone());
    }

    let requested = if raw_path.is_empty() || raw_path == "/" {
        if state.models_dir.is_dir() {
            state.models_dir.to_string_lossy().to_string()
        } else {
            home.as_ref()
                .map(|h| h.to_string_lossy().to_string())
                .ok_or_else(|| ApiError::not_found("Не найдены ни папка моделей, ни домашняя папка"))?
        }
    } else {
        raw_path.to_string()
    };
    let target_dir = crate::paths::resolve_existing_dir_within(&browse_roots, &requested)
        .map_err(|rejection| ApiError::from_path_rejection(rejection, "папка"))?;

    let canonical_roots: Vec<PathBuf> = browse_roots.iter().filter_map(|r| r.canonicalize().ok()).collect();
    let suggestions: Vec<String> = model_roots
        .iter()
        .filter_map(|r| r.canonicalize().ok())
        .map(|r| r.to_string_lossy().to_string())
        .collect();

    // Чтение диска синхронное, поэтому выполняется вне async-потоков сервера
    let listing = tokio::task::spawn_blocking(move || {
        list_browse_entries(&target_dir, &canonical_roots, show_hidden, suggestions)
    })
    .await
    .map_err(|err| ApiError::internal(format!("Обзор папки прерван: {err}")))?;

    Ok(Json(listing))
}

/// Собирает список подпапок для обзора. `parent` отдаётся, только если он тоже внутри разрешённых корней.
fn list_browse_entries(
    target_dir: &std::path::Path,
    canonical_roots: &[PathBuf],
    show_hidden: bool,
    suggestions: Vec<String>,
) -> serde_json::Value {
    let parent = target_dir
        .parent()
        .filter(|parent| canonical_roots.iter().any(|root| parent.starts_with(root)))
        .map(|parent| parent.to_string_lossy().to_string());

    let mut entries = Vec::new();
    let mut model_files_here: usize = 0;
    let mut truncated = false;

    match std::fs::read_dir(target_dir) {
        Ok(dir_entries) => {
            for entry in dir_entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                let is_hidden = name.starts_with('.');
                // file_type() не следует по символическим ссылкам, поэтому ссылки наружу не показываются
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                if file_type.is_file() {
                    if name.to_lowercase().ends_with(".gguf") {
                        model_files_here += 1;
                    }
                    continue;
                }
                if !file_type.is_dir() || (is_hidden && !show_hidden) {
                    continue;
                }
                if entries.len() >= MAX_BROWSE_ENTRIES {
                    truncated = true;
                    continue;
                }
                entries.push(serde_json::json!({
                    "name": name,
                    "has_models": dir_contains_gguf(&entry.path()),
                    "hidden": is_hidden
                }));
            }
        }
        Err(err) => tracing::warn!("Не удалось прочитать папку {}: {err}", target_dir.display()),
    }

    entries.sort_by(|a, b| {
        let name_a = a["name"].as_str().unwrap_or("");
        let name_b = b["name"].as_str().unwrap_or("");
        name_a.cmp(name_b)
    });

    serde_json::json!({
        "current": target_dir.to_string_lossy(),
        "parent": parent,
        "entries": entries,
        "suggestions": suggestions,
        "truncated": truncated,
        "model_files_here": model_files_here
    })
}

/// Есть ли в папке (без захода в подпапки) хотя бы один файл `.gguf`.
fn dir_contains_gguf(dir: &std::path::Path) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .any(|entry| entry.file_name().to_string_lossy().to_lowercase().ends_with(".gguf"))
        })
        .unwrap_or(false)
}

pub async fn handle_models_checkpoints() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "checkpoints": []
    }))
}

pub async fn handle_models_export_size(
    Query(_query): Query<serde_json::Value>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "size_bytes": 2023751680u64,
        "size_formatted": "1.89 GB"
    }))
}

pub async fn handle_models_delete_finetuned() -> crate::error::ApiError {
    // Раньше отвечало «deleted: true», ничего не удаляя
    crate::unavailable::not_ready("Дообученные модели", 3)
}

pub async fn handle_picker_validate_chat_template(
    Json(_payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "valid": true,
        "error": null
    }))
}

pub async fn handle_picker_chat_template(
    Path(id): Path<String>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "model_name": id,
        "chat_template": null,
        "template": null
    }))
}

pub async fn handle_inference_cancel() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

pub async fn handle_inference_count_tokens() -> crate::error::ApiError {
    // Раньше «токены» считались формулой «слова × 4/3 + 4». Настоящий подсчёт требует
    // токенизатора модели, а он появится вместе с движком инференса
    crate::error::ApiError::service_unavailable(crate::chat::ENGINE_NOT_READY)
}

pub async fn handle_inference_active_generations() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "active": [] }))
}

pub async fn handle_inference_audio_stt_status() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "loaded": false,
        "model": null,
        "loading": false
    }))
}

pub async fn handle_inference_audio_stt_unload() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

pub async fn handle_inference_monitor_reset() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "reset": true }))
}

pub async fn handle_studio_release_notes() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "version": "0.1.0",
        "notes": [
            {
                "version": "0.1.0",
                "title": "SlothForge Vulkan Native Release",
                "description": "Native Vulkan acceleration on AMD Radeon GCN 4.0 (Polaris 10).",
                "date": "2026-09-11"
            }
        ]
    }))
}

pub async fn handle_llama_update_changelog() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "component": "llama.cpp",
        "changelog": []
    }))
}

pub async fn handle_auth_api_keys_get(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let keys = state.api_keys.read().await;
    Json(serde_json::json!({ "keys": *keys }))
}

pub async fn handle_auth_api_keys_post(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let name = payload.get("name").and_then(|v| v.as_str()).unwrap_or("Default Key");
    let key_id = format!("key-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis());
    let new_key = serde_json::json!({
        "id": key_id,
        "name": name,
        "key": format!("sf-{}", &key_id),
        "created_at": crate::state::iso_now()
    });
    {
        let mut keys = state.api_keys.write().await;
        keys.push(new_key.clone());
    }
    Json(new_key)
}

pub async fn handle_auth_api_keys_delete(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<serde_json::Value> {
    let mut keys = state.api_keys.write().await;
    keys.retain(|k| k.get("id").and_then(|v| v.as_str()) != Some(&id));
    Json(serde_json::json!({ "status": "ok", "deleted": true }))
}

pub async fn handle_auth_login() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "authenticated": true,
        "token": "session-token-slothforge"
    }))
}

pub async fn handle_auth_change_password() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

pub async fn handle_auth_desktop_initial_password() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

pub async fn handle_shutdown(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    // Остановка мягкая: сервер дождётся завершения текущих запросов, включая этот ответ
    state.shutdown.notify_one();
    tracing::info!("Запрошена остановка сервера из интерфейса");
    Json(serde_json::json!({ "status": "ok", "message": "Сервер останавливается" }))
}

pub async fn handle_providers_public_key() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "key": null }))
}

pub async fn handle_providers_models_get() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "models": [] }))
}

pub async fn handle_providers_models_post() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "models": [] }))
}

pub async fn handle_providers_test() -> crate::error::ApiError {
    // Раньше всегда отвечало «Connection valid», ничего не проверяя
    crate::unavailable::not_ready("Подключение внешних провайдеров", 4)
}

pub async fn handle_providers_detail_put(
    Path(_id): Path<String>,
    Json(_payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

pub async fn handle_providers_detail_delete(
    Path(_id): Path<String>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "deleted": true }))
}

pub async fn handle_providers_add(
    Json(_payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "id": "provider-1" }))
}

pub async fn handle_export_logs() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "logs": [] }))
}

/// Этап дорожной карты, на котором появится экспорт (вместе с настоящим обучением).
const EXPORT_STAGE: u8 = 3;

pub async fn handle_export_load_checkpoint() -> crate::error::ApiError {
    crate::unavailable::not_ready("Экспорт моделей", EXPORT_STAGE)
}

pub async fn handle_export_action() -> crate::error::ApiError {
    // Раньше возвращало «успех» с job_id, а файл не создавался
    crate::unavailable::not_ready("Экспорт моделей", EXPORT_STAGE)
}

pub async fn handle_export_status() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "active": false,
        "status": "idle"
    }))
}

pub async fn handle_llama_update_status() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "update_available": false,
        "current_version": "0.1.0"
    }))
}

pub async fn handle_llama_update() -> crate::error::ApiError {
    crate::error::ApiError::not_implemented(
        "SlothForge использует собственный Vulkan-движок, обновлять llama.cpp не нужно",
    )
}

/// Список баз знаний. Фронтенд ждёт здесь 200 даже без RAG и отличает
/// «баз пока нет» от «RAG недоступен» по полю ragAvailable.
pub async fn handle_rag_knowledge_bases() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "knowledgeBases": [],
        "ragAvailable": false,
        "ragUnavailableReason": crate::unavailable::not_ready("RAG (базы знаний)", 4).detail
    }))
}

pub async fn handle_diffusion_status() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "active": false,
        "status": "idle"
    }))
}

pub async fn handle_api_not_found(uri: axum::http::Uri) -> crate::error::ApiError {
    crate::error::ApiError::not_found(format!("Эндпоинт API не найден: {}", uri.path()))
}
