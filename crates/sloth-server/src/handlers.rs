use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use crate::telemetry::{self, bytes_to_mib};
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

/// `GET /api/hardware`. Раньше здесь были выдуманные «GCN 4.0», «wavefront 64»,
/// «Mesa 24.0.0» и занятые 1420 МБ; теперь только то, что сообщили Vulkan и драйвер.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareDetails {
    pub gpu_name: Option<String>,
    pub vulkan_available: bool,
    pub backend: String,
    pub kernel_driver: Option<String>,
    /// Vulkan-драйвер (например, «radv») и его версия (например, «Mesa 26.2.2»).
    pub vulkan_driver: Option<String>,
    pub vulkan_driver_info: Option<String>,
    pub vulkan_api_version: Option<String>,
    pub vram_total_mb: Option<u64>,
    pub vram_used_mb: Option<u64>,
    pub vram_free_mb: Option<u64>,
    pub is_cluster: bool,
    /// Видеопамять этого компьютера и подключённых узлов кластера.
    pub cluster_total_vram_mb: Option<u64>,
}

/// `GET /api/vram`: объём и занятость видеопамяти. 503, если занятость узнать нельзя.
pub async fn handle_vram(State(state): State<Arc<AppState>>) -> ApiResult<Json<VramInfo>> {
    let gpu = telemetry::collect_for(&state).await;
    let (Some(total), Some(used)) = (gpu.vram_total_bytes, gpu.vram_used_bytes) else {
        return Err(ApiError::service_unavailable(if gpu.available() {
            "Драйвер видеокарты не сообщает, сколько видеопамяти занято"
        } else {
            "Видеокарта с поддержкой Vulkan не найдена"
        }));
    };
    let free = total.saturating_sub(used);
    Ok(Json(VramInfo {
        device_name: gpu.name.clone().unwrap_or_default(),
        total_bytes: total,
        used_bytes: used,
        free_bytes: free,
        total_mb: bytes_to_mib(total),
        used_mb: bytes_to_mib(used),
        free_mb: bytes_to_mib(free),
        usage_percent: gpu.vram_utilization_pct().unwrap_or(0.0) as f32,
    }))
}

pub async fn handle_hardware(State(state): State<Arc<AppState>>) -> Json<HardwareDetails> {
    let gpu = telemetry::collect_for(&state).await;
    let workers = state.coordinator.list_workers().await;
    let workers_vram_mb: u64 = workers.iter().map(|worker| worker.vram_mb).sum();

    Json(HardwareDetails {
        vulkan_available: state.vk_ctx.is_some(),
        backend: telemetry::device_backend(&gpu).to_string(),
        kernel_driver: gpu.kernel_driver.clone(),
        vulkan_driver: gpu.driver_name.clone(),
        vulkan_driver_info: gpu.driver_info.clone(),
        vulkan_api_version: gpu.vulkan_api_version.clone(),
        vram_total_mb: gpu.vram_total_bytes.map(bytes_to_mib),
        vram_used_mb: gpu.vram_used_bytes.map(bytes_to_mib),
        vram_free_mb: gpu.vram_free_bytes().map(bytes_to_mib),
        is_cluster: !workers.is_empty(),
        cluster_total_vram_mb: gpu
            .vram_total_bytes
            .map(|total| bytes_to_mib(total) + workers_vram_mb),
        gpu_name: gpu.name,
    })
}

/// `GET /api/train/hardware`: загрузка, температура, видеопамять и мощность GPU для панели
/// обучения (`GpuUtilization` во фронтенде). Раньше сюда отдавалась совсем другая структура.
pub async fn handle_gpu_utilization(State(state): State<Arc<AppState>>) -> Json<Value> {
    let gpu = telemetry::collect_for(&state).await;
    Json(telemetry::utilization_json(&gpu))
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
        || send_json(&mut socket, &status_message(&state))
            .await
            .is_err()
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
