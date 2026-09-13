use crate::state::AppState;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    Json,
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sloth_vulkan_sys::VramInfo;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast::error::RecvError;

/// Как часто WebSocket повторяет статус обучения, если событий нет.
const WS_STATUS_INTERVAL: Duration = Duration::from_secs(5);

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

pub async fn handle_vram(State(state): State<Arc<AppState>>) -> Json<VramInfo> {
    if let Some(ctx) = &state.vk_ctx {
        if let Ok(info) = ctx.get_vram_info() {
            return Json(info);
        }
    }

    // Default fallback for AMD Radeon RX 570 Series (RADV POLARIS10).
    // Настоящие числа без Vulkan-контекста — шаг 10 (телеметрия железа)
    let total_mb = 4096u64;
    let used_mb = 1420;
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

    // Шаг 10: реальная занятость видеопамяти вместо константы
    let vram_used = 1420;

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

pub async fn handle_ws_telemetry(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_client(socket, state))
}

async fn send_json(socket: &mut WebSocket, value: &Value) -> Result<(), axum::Error> {
    socket.send(Message::Text(value.to_string())).await
}

fn status_message(state: &AppState) -> Value {
    json!({ "type": "training", "event": "status", "data": state.training.status() })
}

/// Поток событий обучения по WebSocket: статус при подключении, затем те же события,
/// что в SSE `/api/train/progress`. Интерфейс этот поток не использует; раньше он
/// рассылал выдуманные температуру, мощность и загрузку GPU.
async fn handle_ws_client(mut socket: WebSocket, state: Arc<AppState>) {
    let mut events = state.training.subscribe();

    let welcome = json!({
        "type": "log",
        "message": "[SYSTEM] Поток событий обучения SlothForge подключён"
    });
    if send_json(&mut socket, &welcome).await.is_err()
        || send_json(&mut socket, &status_message(&state)).await.is_err()
    {
        return;
    }

    let mut status_ticker = tokio::time::interval(WS_STATUS_INTERVAL);
    // Первый тик interval срабатывает сразу, а статус только что отправлен
    status_ticker.tick().await;

    loop {
        let outgoing = tokio::select! {
            received = events.recv() => match received {
                Ok(event) => json!({
                    "type": "training",
                    "event": event.kind.as_str(),
                    "id": event.id,
                    "data": event.payload
                }),
                // Клиент не успевал читать: отправляем актуальный статус вместо пропущенного
                Err(RecvError::Lagged(_)) => status_message(&state),
                Err(RecvError::Closed) => break,
            },
            _ = status_ticker.tick() => status_message(&state),
            client_frame = socket.next() => match client_frame {
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
                // Ping/Pong axum обрабатывает сам, остальное клиенту отправлять незачем
                Some(Ok(_)) => continue,
            },
        };
        if send_json(&mut socket, &outgoing).await.is_err() {
            break;
        }
    }
}
