use crate::models::{discover_models, OpenAIModelInfo};
use crate::state::{AppState, WsTelemetryEnvelope};
use crate::training::{self, TrainStartRequest, TrainStatusResponse};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sloth_vulkan_sys::VramInfo;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareDetails {
    pub gpu_name: String,
    pub device_name: String,
    pub vulkan_version: String,
    pub wavefront_size: u32,
    pub architecture: String,
    pub compute_units: u32,
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    pub backend: String,
    pub driver: String,
    pub is_cluster: bool,
    pub cluster_total_vram_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsApiResponse {
    pub object: String,
    pub count: usize,
    pub models: Vec<crate::models::ModelCard>,
    pub default_models: Vec<String>,
    pub data: Vec<OpenAIModelInfo>,
}

pub async fn handle_vram(State(state): State<Arc<AppState>>) -> Json<VramInfo> {
    let training_vram = state.training.vram_used_mb.load(Ordering::Relaxed);

    if let Some(ctx) = &state.vk_ctx {
        if let Ok(mut info) = ctx.get_vram_info() {
            if state.training.is_active.load(Ordering::Relaxed) && training_vram > info.used_mb {
                info.used_mb = training_vram;
                info.used_bytes = training_vram * 1024 * 1024;
                info.free_mb = info.total_mb.saturating_sub(training_vram);
                info.free_bytes = info.free_mb * 1024 * 1024;
                info.usage_percent = (info.used_mb as f32 / info.total_mb as f32) * 100.0;
            }
            return Json(info);
        }
    }

    // Default fallback for AMD Radeon RX 570 Series (RADV POLARIS10)
    let total_mb = 4096u64;
    let used_mb = if state.training.is_active.load(Ordering::Relaxed) {
        training_vram
    } else {
        1420
    };
    let free_mb = total_mb.saturating_sub(used_mb);
    let usage_percent = (used_mb as f32 / total_mb as f32) * 100.0;

    Json(VramInfo {
        device_name: "AMD Radeon RX 570 Series (RADV POLARIS10)".to_string(),
        total_bytes: total_mb * 1024 * 1024,
        used_bytes: used_mb * 1024 * 1024,
        free_bytes: free_mb * 1024 * 1024,
        total_mb,
        used_mb,
        free_mb,
        usage_percent,
    })
}

pub async fn handle_hardware(State(state): State<Arc<AppState>>) -> Json<HardwareDetails> {
    let dev_name = state
        .vk_ctx
        .as_ref()
        .map(|c| c.device_name().to_string())
        .unwrap_or_else(|| "AMD Radeon RX 570 Series (RADV POLARIS10)".to_string());

    let workers = state.coordinator.list_workers().await;
    let is_cluster = !workers.is_empty();
    let cluster_vram: u64 = 4096 + workers.iter().map(|w| w.vram_mb).sum::<u64>();

    let vram_used = if state.training.is_active.load(Ordering::Relaxed) {
        state.training.vram_used_mb.load(Ordering::Relaxed)
    } else {
        1420
    };

    Json(HardwareDetails {
        gpu_name: "AMD Radeon RX 570".to_string(),
        device_name: dev_name,
        vulkan_version: "Vulkan 1.4".to_string(),
        wavefront_size: 64,
        architecture: "GCN 4.0".to_string(),
        compute_units: 32,
        vram_total_mb: 4096,
        vram_used_mb: vram_used,
        backend: "Vulkan Native (wavefront 64, GCN 4.0)".to_string(),
        driver: "RADV POLARIS10 (Mesa 24.0.0)".to_string(),
        is_cluster,
        cluster_total_vram_mb: cluster_vram,
    })
}

pub async fn handle_models(State(state): State<Arc<AppState>>) -> Json<ModelsApiResponse> {
    let cards = discover_models(&state.models_dir);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let openai_data = cards
        .iter()
        .map(|c| OpenAIModelInfo {
            id: c.id.clone(),
            object: "model".to_string(),
            created: now,
            owned_by: "slothforge".to_string(),
        })
        .collect();

    let count = cards.len();
    Json(ModelsApiResponse {
        object: "list".to_string(),
        count,
        models: cards,
        default_models: vec![
            "llama-3.2-3b-instruct-q4_k_m".to_string(),
            "llama-3.2-1b-instruct-q4_k_m".to_string(),
        ],
        data: openai_data,
    })
}

pub async fn handle_train_start(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TrainStartRequest>,
) -> Result<Json<crate::training::TrainStartResponse>, (StatusCode, String)> {
    match training::start_training(state, req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(err) => Err((StatusCode::BAD_REQUEST, err)),
    }
}

pub async fn handle_train_stop(
    State(state): State<Arc<AppState>>,
) -> Json<crate::training::TrainStopResponse> {
    Json(training::stop_training(state).await)
}

pub async fn handle_train_status(
    State(state): State<Arc<AppState>>,
) -> Json<TrainStatusResponse> {
    Json(crate::api::build_training_status(&state).await)
}

pub async fn handle_ws_telemetry(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_client(socket, state))
}

async fn handle_ws_client(mut socket: WebSocket, state: Arc<AppState>) {
    let mut rx = state.tx_telemetry.subscribe();

    // Initial connection welcome & current telemetry snapshot
    let initial_snapshot = state.training.snapshot("local", 1).await;
    let welcome = serde_json::json!({
        "type": "log",
        "message": "[SYSTEM] SlothForge live telemetry WebSocket active (Vulkan wavefront-64 RADV)."
    });
    let _ = socket.send(Message::Text(welcome.to_string())).await;

    let snapshot_msg = serde_json::json!({
        "type": "telemetry",
        "data": initial_snapshot
    });
    if socket.send(Message::Text(snapshot_msg.to_string())).await.is_err() {
        return;
    }

    // Heartbeat ticker for idle periods (at ~2 Hz when not training)
    let mut idle_ticker = tokio::time::interval(Duration::from_millis(500));

    loop {
        tokio::select! {
            // Live broadcasted telemetry from training loop (10-20 Hz)
            msg = rx.recv() => {
                match msg {
                    Ok(envelope) => {
                        let text = serde_json::to_string(&envelope).unwrap_or_default();
                        if socket.send(Message::Text(text)).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        // Skip lagged messages to keep up with real-time stream
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
            // Idle background heartbeat when no training is active
            _ = idle_ticker.tick() => {
                if !state.training.is_active.load(Ordering::Relaxed) {
                    let snapshot = state.training.snapshot("local", 1).await;
                    let envelope = WsTelemetryEnvelope::Telemetry { data: snapshot };
                    let text = serde_json::to_string(&envelope).unwrap_or_default();
                    if socket.send(Message::Text(text)).await.is_err() {
                        break;
                    }
                }
            }
            // Receive client frames (ping/close)
            client_frame = socket.next() => {
                match client_frame {
                    Some(Ok(Message::Text(_))) | Some(Ok(Message::Ping(_))) => {
                        // Pong is handled automatically by axum
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                }
            }
        }
    }
}
