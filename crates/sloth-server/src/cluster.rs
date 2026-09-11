use crate::state::AppState;
use axum::{
    body::Bytes,
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use sloth_core::cluster::{AllReduceEngine, ClusterCoordinator, WorkerInfo};
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RegisterWorkerRequest {
    #[serde(alias = "worker_id")]
    pub worker_id: String,
    pub ip: String,
    #[serde(alias = "gpu_name", default = "default_gpu")]
    pub gpu_name: String,
    #[serde(alias = "vram_mb", default = "default_vram")]
    pub vram_mb: u64,
    #[serde(alias = "latency_ms", default)]
    pub latency_ms: Option<f32>,
}

fn default_gpu() -> String {
    "AMD Radeon RX 570 (RADV POLARIS10)".to_string()
}

fn default_vram() -> u64 {
    4096
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RegisterWorkerResponse {
    pub status: String,
    pub worker_id: String,
    pub rank: usize,
    pub world_size: usize,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyncGradJsonRequest {
    pub step: u32,
    #[serde(default = "default_worker_rank")]
    pub rank: usize,
    #[serde(alias = "worker_id", default)]
    pub worker_id: Option<String>,
    pub gradients: Vec<f32>,
}

fn default_worker_rank() -> usize {
    2
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SyncGradJsonResponse {
    pub status: String,
    pub step: u32,
    pub gradients: Vec<f32>,
    pub gradient_count: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClusterStatusResponse {
    pub role: String,
    pub world_size: usize,
    pub current_step: u32,
    pub master_ip: String,
    pub workers: Vec<ClusterWorkerStatus>,
    pub total_vram_mb: u64,
    pub all_reduce_ready: bool,
    pub average_latency_ms: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClusterWorkerStatus {
    pub worker_id: String,
    pub ip: String,
    pub gpu_name: String,
    pub vram_mb: u64,
    pub latency_ms: f32,
    pub last_seen_ms: u64,
    pub status: String,
}

pub async fn handle_register_worker(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterWorkerRequest>,
) -> Result<Json<RegisterWorkerResponse>, (StatusCode, String)> {
    let now = ClusterCoordinator::now_ms();
    let worker_info = WorkerInfo {
        worker_id: req.worker_id.clone(),
        ip: req.ip.clone(),
        gpu_name: req.gpu_name.clone(),
        vram_mb: req.vram_mb,
        latency_ms: req.latency_ms.unwrap_or(1.2),
        last_seen_ms: now,
    };

    let ack = state.coordinator.register_worker(worker_info).await;

    match ack {
        sloth_core::cluster::ClusterMessage::RegisterAck {
            status,
            worker_id,
            rank,
            world_size,
        } => Ok(Json(RegisterWorkerResponse {
            status,
            worker_id,
            rank,
            world_size,
            message: "Worker successfully registered in SlothForge 2-PC cluster.".to_string(),
        })),
        _ => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to register worker".to_string(),
        )),
    }
}

pub async fn handle_sync_grad(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let is_octet_stream = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.contains("application/octet-stream"))
        .unwrap_or(false);

    if is_octet_stream {
        // Binary float32 gradient sync
        let worker_grads = AllReduceEngine::deserialize_gradients(&body);
        if worker_grads.is_empty() {
            return (
                StatusCode::BAD_REQUEST,
                "Empty binary gradient payload received",
            )
                .into_response();
        }

        let mut master_grads_lock = state.master_gradients.write().await;
        if master_grads_lock.len() != worker_grads.len() {
            *master_grads_lock = vec![0.01f32; worker_grads.len()];
        }

        if let Err(e) = AllReduceEngine::average_two(&mut master_grads_lock, &worker_grads) {
            return (
                StatusCode::BAD_REQUEST,
                format!("AllReduce gradient length mismatch: {e}"),
            )
                .into_response();
        }

        let serialized = AllReduceEngine::serialize_gradients(&master_grads_lock);
        (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "application/octet-stream"),
                (
                    header::HeaderName::from_static("x-sloth-grad-count"),
                    &master_grads_lock.len().to_string(),
                ),
            ],
            serialized,
        )
            .into_response()
    } else {
        // JSON format gradient sync
        let req: SyncGradJsonRequest = match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Invalid JSON gradient payload: {e}"),
                )
                    .into_response();
            }
        };

        let mut master_grads_lock = state.master_gradients.write().await;
        if master_grads_lock.len() != req.gradients.len() {
            *master_grads_lock = vec![0.01f32; req.gradients.len()];
        }

        if let Err(e) = AllReduceEngine::average_two(&mut master_grads_lock, &req.gradients) {
            return (
                StatusCode::BAD_REQUEST,
                format!("AllReduce gradient length mismatch: {e}"),
            )
                .into_response();
        }

        let averaged = master_grads_lock.clone();
        let resp = SyncGradJsonResponse {
            status: "ok".to_string(),
            step: req.step,
            gradient_count: averaged.len(),
            gradients: averaged,
        };

        (StatusCode::OK, Json(resp)).into_response()
    }
}

pub async fn handle_cluster_status(
    State(state): State<Arc<AppState>>,
) -> Json<ClusterStatusResponse> {
    let workers = state.coordinator.list_workers().await;
    let current_step = state.coordinator.current_step.load(Ordering::Relaxed);
    let world_size = workers.len() + 1;

    let mut total_vram = 4096u64; // Master node RX 570
    let mut total_latency = 0.0f32;

    let worker_statuses: Vec<ClusterWorkerStatus> = workers
        .into_iter()
        .map(|w| {
            total_vram += w.vram_mb;
            total_latency += w.latency_ms;
            ClusterWorkerStatus {
                worker_id: w.worker_id,
                ip: w.ip,
                gpu_name: w.gpu_name,
                vram_mb: w.vram_mb,
                latency_ms: w.latency_ms,
                last_seen_ms: w.last_seen_ms,
                status: "online".to_string(),
            }
        })
        .collect();

    let avg_latency = if !worker_statuses.is_empty() {
        total_latency / worker_statuses.len() as f32
    } else {
        0.0
    };

    Json(ClusterStatusResponse {
        role: "master".to_string(),
        world_size,
        current_step,
        master_ip: "127.0.0.1".to_string(),
        workers: worker_statuses,
        total_vram_mb: total_vram,
        all_reduce_ready: world_size >= 2,
        average_latency_ms: avg_latency,
    })
}
