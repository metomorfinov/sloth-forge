use crate::state::{AppState, TrainingRunSummary};
use crate::training::{self, TrainResetResponse, TrainingDetails, TrainingMetricHistory, TrainingStatusResponse};
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

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
    let is_active = state.training.is_active.load(Ordering::Relaxed);
    let vram_total_mb = 4096u64;
    let vram_used_mb = if is_active {
        state.training.vram_used_mb.load(Ordering::Relaxed)
    } else {
        0u64
    };
    let vram_free_mb = vram_total_mb.saturating_sub(vram_used_mb);

    let gpu_name = state
        .vk_ctx
        .as_ref()
        .map(|c| c.device_name().to_string())
        .unwrap_or_else(|| "AMD Radeon RX 570 Series (RADV POLARIS10)".to_string());

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
        vulkan_available: true,
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

pub async fn build_training_status(state: &AppState) -> TrainingStatusResponse {
    let is_active = state.training.is_active.load(Ordering::Relaxed);
    let step = state.training.step.load(Ordering::Relaxed);
    let total_steps = state.training.total_steps.load(Ordering::Relaxed);
    let epoch = state.training.epoch.load(Ordering::Relaxed);
    let loss = *state.training.loss.read().await;
    let tokens_per_sec = *state.training.tokens_per_sec.read().await;
    let learning_rate = *state.training.learning_rate.read().await;
    let status_text = state.training.status_text.read().await.clone();
    let start_time = *state.training.start_time.read().await;
    let elapsed_secs = start_time.map(|t| t.elapsed().as_secs()).unwrap_or(0);
    let eta_secs = if is_active && step < total_steps && tokens_per_sec > 0.0 {
        let remaining = total_steps - step;
        (remaining as f32 / 20.0).round() as u64
    } else {
        0
    };
    let vram_used_mb = state.training.vram_used_mb.load(Ordering::Relaxed);
    let current_job_id = state.training.current_job_id.read().await.clone();
    let start_req_id = state.training.current_start_request_id.read().await.clone();
    let loss_history = state.training.loss_history.read().await.clone();

    let phase = if is_active {
        "training".to_string()
    } else if step >= total_steps && total_steps > 0 {
        "completed".to_string()
    } else if step > 0 {
        "stopped".to_string()
    } else {
        "idle".to_string()
    };

    let details = if is_active || step > 0 {
        Some(TrainingDetails {
            epoch,
            step,
            total_steps,
            loss,
            learning_rate,
            output_dir: Some("outputs/slothforge-lora".to_string()),
        })
    } else {
        None
    };

    let metric_history = if !loss_history.is_empty() {
        let steps: Vec<u32> = loss_history.iter().map(|e| e.step).collect();
        let losses: Vec<f32> = loss_history.iter().map(|e| e.loss).collect();
        let lrs: Vec<f32> = loss_history.iter().map(|e| e.lr).collect();
        let grad_norms: Vec<f32> = loss_history.iter().map(|_| 0.15f32).collect();
        Some(TrainingMetricHistory {
            steps,
            loss: losses,
            lr: lrs,
            grad_norm: grad_norms,
            grad_norm_steps: vec![],
            eval_loss: vec![],
            eval_steps: vec![],
        })
    } else {
        None
    };

    TrainingStatusResponse {
        job_id: if current_job_id.is_empty() { "job-default".to_string() } else { current_job_id },
        start_request_id: start_req_id,
        start_request_state: Some("accepted".to_string()),
        phase,
        is_training_running: is_active,
        eval_enabled: false,
        message: if is_active {
            format!("Training step {}/{} - Loss: {:.4}", step, total_steps, loss)
        } else {
            status_text.clone()
        },
        error: None,
        warnings: vec![],
        details,
        metric_history,
        active: is_active,
        status: status_text,
        step,
        total_steps,
        loss,
        tokens_per_sec,
        vram_used_mb,
        vram_total_mb: 4096,
        elapsed_secs,
        eta_secs,
        epoch,
        learning_rate,
    }
}

pub async fn handle_train_status(State(state): State<Arc<AppState>>) -> Json<TrainingStatusResponse> {
    Json(build_training_status(&state).await)
}

#[derive(Debug, Deserialize, Default)]
pub struct ProgressQuery {
    pub expected_job_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProgressPayload {
    pub job_id: String,
    pub step: u32,
    pub total_steps: u32,
    pub loss: Option<f32>,
    pub learning_rate: Option<f32>,
    pub progress_percent: f32,
    pub epoch: Option<u32>,
    pub elapsed_seconds: Option<u64>,
    pub eta_seconds: Option<u64>,
    pub grad_norm: Option<f32>,
    pub num_tokens: Option<u64>,
    pub eval_loss: Option<f32>,
}

pub async fn build_progress_payload(state: &AppState, expected_id: &str) -> TrainingProgressPayload {
    let is_active = state.training.is_active.load(Ordering::Relaxed);
    let step = state.training.step.load(Ordering::Relaxed);
    let total_steps = state.training.total_steps.load(Ordering::Relaxed);
    let epoch = state.training.epoch.load(Ordering::Relaxed);
    let loss = *state.training.loss.read().await;
    let lr = *state.training.learning_rate.read().await;
    let elapsed = state.training.start_time.read().await.map(|t| t.elapsed().as_secs());
    let eta = if is_active && step < total_steps {
        let rem = total_steps - step;
        Some((rem as f32 / 20.0).round() as u64)
    } else {
        Some(0)
    };
    let progress_percent = if total_steps > 0 {
        ((step as f32 / total_steps as f32) * 100.0).min(100.0)
    } else {
        0.0
    };
    let stored_job_id = state.training.current_job_id.read().await.clone();
    let job_id = if !expected_id.is_empty() {
        expected_id.to_string()
    } else if !stored_job_id.is_empty() {
        stored_job_id
    } else {
        "job-default".to_string()
    };

    TrainingProgressPayload {
        job_id,
        step,
        total_steps,
        loss: Some(loss),
        learning_rate: Some(lr),
        progress_percent,
        epoch: Some(epoch),
        elapsed_seconds: elapsed,
        eta_seconds: eta,
        grad_norm: Some(0.15),
        num_tokens: Some(step as u64 * 1024),
        eval_loss: None,
    }
}

pub async fn handle_train_progress(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<ProgressQuery>,
) -> Response {
    let accept = headers
        .get("accept")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");

    if accept.contains("text/event-stream") {
        let expected_id = query.expected_job_id.unwrap_or_default();
        let state_clone = Arc::clone(&state);
        let rx = state.tx_telemetry.subscribe();

        let stream = stream::unfold((state_clone, rx, 0u32, expected_id), |(st, mut r, mut event_id, exp_id)| async move {
            tokio::select! {
                msg = r.recv() => {
                    match msg {
                        Ok(_) => {
                            event_id += 1;
                            let snap = build_progress_payload(&st, &exp_id).await;
                            let json = serde_json::to_string(&snap).unwrap_or_default();
                            let ev = Event::default().event("progress").data(json).id(event_id.to_string());
                            Some((Ok::<_, Infallible>(ev), (st, r, event_id, exp_id)))
                        }
                        Err(_) => None,
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(500)) => {
                    event_id += 1;
                    let snap = build_progress_payload(&st, &exp_id).await;
                    let json = serde_json::to_string(&snap).unwrap_or_default();
                    let ev = Event::default().event("heartbeat").data(json).id(event_id.to_string());
                    Some((Ok::<_, Infallible>(ev), (st, r, event_id, exp_id)))
                }
            }
        });

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(build_training_status(&state).await).into_response()
    }
}

pub async fn handle_train_reset(State(state): State<Arc<AppState>>) -> Json<TrainResetResponse> {
    Json(training::reset_training(state).await)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRunListResponse {
    pub runs: Vec<TrainingRunSummary>,
    pub total: usize,
}

pub async fn handle_train_runs(State(state): State<Arc<AppState>>) -> Json<TrainingRunListResponse> {
    let runs = state.training.runs.read().await.clone();
    let total = runs.len();
    Json(TrainingRunListResponse { runs, total })
}

pub async fn handle_train_run_detail(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let runs = state.training.runs.read().await;
    if let Some(run) = runs.iter().find(|r| r.id == run_id) {
        let history = state.training.loss_history.read().await;
        let step_history: Vec<u32> = history.iter().map(|h| h.step).collect();
        let loss_history: Vec<f32> = history.iter().map(|h| h.loss).collect();
        let lr_history: Vec<f32> = history.iter().map(|h| h.lr).collect();

        let resp = serde_json::json!({
            "run": run,
            "config": {
                "model_name": run.model_name,
                "dataset_name": run.dataset_name,
                "total_steps": run.total_steps,
            },
            "metrics": {
                "step_history": step_history,
                "loss_history": loss_history,
                "lr_history": lr_history,
                "loss_step_history": step_history.clone(),
                "lr_step_history": step_history.clone(),
                "grad_norm_history": [],
                "grad_norm_step_history": [],
                "eval_loss_history": [],
                "eval_step_history": [],
                "final_epoch": 1,
                "final_num_tokens": null
            }
        });
        Ok(Json(resp))
    } else {
        Err((StatusCode::NOT_FOUND, format!("Run '{run_id}' not found")))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetricsResponse {
    pub job_id: String,
    pub loss_history: Vec<f32>,
    pub lr_history: Vec<f32>,
    pub step_history: Vec<u32>,
    pub grad_norm_history: Vec<f32>,
    pub grad_norm_step_history: Vec<u32>,
    pub current_loss: Option<f32>,
    pub current_lr: Option<f32>,
    pub current_step: Option<u32>,
}

pub async fn handle_train_metrics(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ProgressQuery>,
) -> Json<TrainingMetricsResponse> {
    let history = state.training.loss_history.read().await;
    let step_history: Vec<u32> = history.iter().map(|h| h.step).collect();
    let loss_history: Vec<f32> = history.iter().map(|h| h.loss).collect();
    let lr_history: Vec<f32> = history.iter().map(|h| h.lr).collect();
    let step = state.training.step.load(Ordering::Relaxed);
    let loss = *state.training.loss.read().await;
    let lr = *state.training.learning_rate.read().await;
    let current_job_id = state.training.current_job_id.read().await.clone();
    let job_id = query.expected_job_id.unwrap_or(current_job_id);

    Json(TrainingMetricsResponse {
        job_id,
        loss_history,
        lr_history,
        step_history,
        grad_norm_history: vec![],
        grad_norm_step_history: vec![],
        current_loss: Some(loss),
        current_lr: Some(lr),
        current_step: Some(step),
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
    let vram_used = if state.training.is_active.load(Ordering::Relaxed) {
        state.training.vram_used_mb.load(Ordering::Relaxed)
    } else {
        0
    };
    let dev_name = state
        .vk_ctx
        .as_ref()
        .map(|c| c.device_name().to_string())
        .unwrap_or_else(|| "AMD Radeon RX 570 Series (RADV POLARIS10)".to_string());

    let resp = serde_json::json!({
        "status": "ready",
        "platform": "linux",
        "python_version": "3.11.0",
        "device_backend": "vulkan",
        "uptime_seconds": 3600,
        "cpu": {
            "logical_count": 8,
            "physical_count": 4,
            "usage_percent": 15.0,
            "frequency_mhz": 3200
        },
        "memory": {
            "total_gb": 16.0,
            "available_gb": 12.0,
            "percent_used": 25.0,
            "process_used_mb": 250
        },
        "disk": {
            "total_gb": 500.0,
            "free_gb": 320.0,
            "percent_used": 36.0
        },
        "gpu": {
            "available": true,
            "backend": "vulkan",
            "devices": [{
                "device_id": 0,
                "gpu_name": dev_name,
                "vram_total_gb": 4.0,
                "vram_free_gb": ((4096 - vram_used) as f64 / 1024.0),
                "vram_used_gb": (vram_used as f64 / 1024.0),
                "vram_utilization_pct": ((vram_used as f64 / 4096.0) * 100.0)
            }]
        },
        "ml_packages": {
            "torch": "2.4.0",
            "transformers": "4.44.0"
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
    let vram_used = if state.training.is_active.load(Ordering::Relaxed) {
        state.training.vram_used_mb.load(Ordering::Relaxed)
    } else {
        0
    };
    let vram_free_gb = (4096 - vram_used) as f64 / 1024.0;

    let resp = serde_json::json!({
        "gpuName": dev_name,
        "vramTotalGb": 4.0,
        "vramFreeGb": vram_free_gb,
        "gpus": [{
            "device_id": 0,
            "gpu_name": dev_name,
            "vram_total_gb": 4.0,
            "vram_free_gb": vram_free_gb,
            "vram_used_gb": (vram_used as f64 / 1024.0),
            "vram_utilization_pct": ((vram_used as f64 / 4096.0) * 100.0)
        }],
        "torch": null,
        "cuda": null,
        "rocm": null,
        "xpu": null,
        "transformers": "4.44.0",
        "unsloth": "0.1.0",
        "llamaCpp": "b3600",
        "exportSupported": true,
        "exportUnsupportedReason": null,
        "exportUnsupportedMessage": null,
        "videoSupported": false,
        "videoUnsupportedReason": "Vulkan GCN 4.0 does not meet video generation requirements.",
        "videoUnsupportedMessage": "Video generation is not supported on this device.",
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

pub async fn handle_model_config(Path(id): Path<String>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "id": id,
        "model_name": id,
        "is_vision": false,
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
    }))
}

pub async fn handle_start_request_ack(Path(_id): Path<String>) -> StatusCode {
    StatusCode::OK
}

pub async fn handle_start_request_cancel(
    Path(id): Path<String>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "start_request_id": id,
        "job_id": format!("job-{}", id),
        "state": "rejected",
        "message": "Start request cancelled",
        "error": null
    }))
}
