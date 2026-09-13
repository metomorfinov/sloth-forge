//! Эндпоинты кластера `/api/cluster/*`.
//!
//! Раньше при несовпадении длины градиентов мастер молча заменял свои градиенты массивом
//! из 0,01, усреднение перезаписывало мастер-копию, узлы без полей получали выдуманные
//! «AMD Radeon RX 570, 4 ГБ» и задержку 1,2 мс, статус всегда был «online», а мастер
//! всегда считался картой на 4096 МБ. Распределённое обучение появится на этапе 3;
//! до этого у мастера нет градиентов и синхронизация честно отвечает 409.

use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use crate::telemetry;
use axum::body::Bytes;
use axum::extract::{ConnectInfo, State};
use axum::http::{header, HeaderMap, HeaderName, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use sloth_core::cluster::{
    AllReduceEngine, ClusterCoordinator, ClusterError, ClusterMessage, WorkerInfo,
};
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Узел считается отключённым, если heartbeat не приходил дольше этого времени.
const WORKER_TIMEOUT_MS: u64 = 15_000;
const MAX_WORKER_ID_CHARS: usize = 128;
const MAX_GPU_NAME_CHARS: usize = 256;
/// Верхняя граница видеопамяти узла: 1 ТиБ.
const MAX_WORKER_VRAM_MB: u64 = 1 << 20;
const OCTET_STREAM: &str = "application/octet-stream";
const GRAD_COUNT_HEADER: &str = "x-sloth-grad-count";
const STEP_HEADER: &str = "x-sloth-step";
const NO_MASTER_GRADIENTS: &str =
    "У мастера нет градиентов текущего шага: распределённое обучение появится на этапе 3 дорожной карты";

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RegisterWorkerRequest {
    #[serde(alias = "worker_id")]
    pub worker_id: String,
    /// Если не указан, берётся адрес самого соединения.
    #[serde(default)]
    pub ip: Option<String>,
    #[serde(alias = "gpu_name")]
    pub gpu_name: String,
    #[serde(alias = "vram_mb")]
    pub vram_mb: u64,
    #[serde(alias = "latency_ms", default)]
    pub latency_ms: Option<f32>,
}

fn required_text<'a>(field: &str, value: &'a str, max_chars: usize) -> ApiResult<&'a str> {
    let value = value.trim();
    if value.is_empty() || value.chars().count() > max_chars {
        return Err(ApiError::unprocessable(format!(
            "Поле {field} должно быть непустым и не длиннее {max_chars} символов"
        )));
    }
    Ok(value)
}

/// `POST /api/cluster/worker/register`.
pub async fn handle_register_worker(
    State(state): State<Arc<AppState>>,
    connection: Option<ConnectInfo<SocketAddr>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let request: RegisterWorkerRequest = serde_json::from_value(body)
        .map_err(|err| ApiError::unprocessable(format!("Некорректная регистрация узла: {err}")))?;
    let worker_id = required_text("worker_id", &request.worker_id, MAX_WORKER_ID_CHARS)?;
    let gpu_name = required_text("gpu_name", &request.gpu_name, MAX_GPU_NAME_CHARS)?;
    if !(1..=MAX_WORKER_VRAM_MB).contains(&request.vram_mb) {
        return Err(ApiError::unprocessable(format!(
            "Поле vram_mb должно быть от 1 до {MAX_WORKER_VRAM_MB}"
        )));
    }
    let ip: IpAddr = match request
        .ip
        .as_deref()
        .map(str::trim)
        .filter(|ip| !ip.is_empty())
    {
        Some(text) => text.parse().map_err(|_| {
            ApiError::unprocessable(format!("Некорректный IP-адрес узла: «{text}»"))
        })?,
        None => connection
            .map(|ConnectInfo(address)| address.ip())
            .ok_or_else(|| ApiError::unprocessable("Не указан IP-адрес узла"))?,
    };

    let ack = state
        .coordinator
        .register_worker(WorkerInfo {
            worker_id: worker_id.to_string(),
            rank: 0,
            ip: ip.to_string(),
            gpu_name: gpu_name.to_string(),
            vram_mb: request.vram_mb,
            latency_ms: request
                .latency_ms
                .filter(|latency| latency.is_finite() && *latency >= 0.0),
            last_seen_ms: ClusterCoordinator::now_ms(),
        })
        .await;
    let ClusterMessage::RegisterAck {
        status,
        worker_id,
        rank,
        world_size,
    } = ack
    else {
        return Err(ApiError::internal(
            "Координатор кластера вернул неожиданный ответ на регистрацию",
        ));
    };
    Ok(Json(json!({
        "status": status,
        "workerId": worker_id,
        "rank": rank,
        "worldSize": world_size,
        "message": "Узел зарегистрирован в кластере"
    })))
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HeartbeatRequest {
    #[serde(alias = "worker_id")]
    pub worker_id: String,
    /// Время узла в миллисекундах: по нему считается задержка.
    #[serde(alias = "timestamp_ms", default)]
    pub timestamp_ms: Option<u64>,
}

/// `POST /api/cluster/worker/heartbeat`: узел сообщает, что он на связи.
pub async fn handle_worker_heartbeat(
    State(state): State<Arc<AppState>>,
    Json(request): Json<HeartbeatRequest>,
) -> ApiResult<Json<Value>> {
    match state
        .coordinator
        .handle_heartbeat(request.worker_id.trim(), request.timestamp_ms)
        .await
    {
        Ok(ClusterMessage::HeartbeatAck {
            client_timestamp_ms,
            server_timestamp_ms,
            latency_ms,
        }) => Ok(Json(json!({
            "clientTimestampMs": client_timestamp_ms,
            "serverTimestampMs": server_timestamp_ms,
            "latencyMs": latency_ms
        }))),
        Ok(_) => Err(ApiError::internal(
            "Координатор кластера вернул неожиданный ответ на heartbeat",
        )),
        Err(ClusterError::WorkerNotRegistered(id)) => Err(ApiError::not_found(format!(
            "Узел {id} не зарегистрирован в кластере"
        ))),
        Err(err) => Err(ApiError::internal(err.to_string())),
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyncGradJsonRequest {
    pub step: u32,
    #[serde(default)]
    pub rank: Option<usize>,
    #[serde(alias = "worker_id", default)]
    pub worker_id: Option<String>,
    pub gradients: Vec<f32>,
}

fn gradient_error(err: ClusterError) -> ApiError {
    match err {
        ClusterError::GradientLengthMismatch {
            master_len,
            worker_len,
        } => ApiError::bad_request(format!(
            "Число градиентов узла ({worker_len}) не совпадает с мастером ({master_len})"
        )),
        ClusterError::PartialGradientBytes { len } => ApiError::bad_request(format!(
            "Двоичные градиенты ({len} байт) не делятся на целые значения float32"
        )),
        ClusterError::NonFiniteGradient { index } => {
            ApiError::bad_request(format!("Градиент №{index} — NaN или бесконечность"))
        }
        other => ApiError::internal(other.to_string()),
    }
}

/// Номер шага из заголовка двоичного запроса (необязателен).
fn step_header(headers: &HeaderMap) -> ApiResult<Option<u32>> {
    headers
        .get(STEP_HEADER)
        .map(|value| {
            value
                .to_str()
                .ok()
                .and_then(|text| text.trim().parse().ok())
                .ok_or_else(|| {
                    ApiError::bad_request(format!(
                        "Заголовок {STEP_HEADER} должен быть номером шага"
                    ))
                })
        })
        .transpose()
}

/// `POST /api/cluster/sync_grad`: усреднение градиентов узла с градиентами мастера.
/// Формат — JSON или двоичные float32 (`application/octet-stream`).
pub async fn handle_sync_grad(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> ApiResult<Response> {
    let is_binary = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains(OCTET_STREAM));

    let (worker_step, worker_grads) = if is_binary {
        (
            step_header(&headers)?,
            AllReduceEngine::deserialize_gradients(&body).map_err(gradient_error)?,
        )
    } else {
        let request: SyncGradJsonRequest = serde_json::from_slice(&body).map_err(|err| {
            ApiError::bad_request(format!("Некорректные градиенты в JSON: {err}"))
        })?;
        (Some(request.step), request.gradients)
    };
    if worker_grads.is_empty() {
        return Err(ApiError::bad_request(
            "Узел прислал пустой набор градиентов",
        ));
    }
    AllReduceEngine::ensure_finite(&worker_grads).map_err(gradient_error)?;

    let (master_step, averaged) = {
        let master = state.master_gradients.read().await;
        let Some(master) = master.as_ref() else {
            return Err(ApiError::conflict(NO_MASTER_GRADIENTS));
        };
        if let Some(step) = worker_step.filter(|step| *step != master.step) {
            return Err(ApiError::conflict(format!(
                "Шаг узла {step} не совпадает с текущим шагом мастера {}",
                master.step
            )));
        }
        // Мастер-копия не меняется: среднее получает узел, а применяет его движок обучения
        let averaged =
            AllReduceEngine::averaged(&master.gradients, &worker_grads).map_err(gradient_error)?;
        (master.step, averaged)
    };

    if is_binary {
        Ok((
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, OCTET_STREAM.to_string()),
                (
                    HeaderName::from_static(GRAD_COUNT_HEADER),
                    averaged.len().to_string(),
                ),
                (
                    HeaderName::from_static(STEP_HEADER),
                    master_step.to_string(),
                ),
            ],
            AllReduceEngine::serialize_gradients(&averaged),
        )
            .into_response())
    } else {
        Ok(Json(json!({
            "status": "ok",
            "step": master_step,
            "gradientCount": averaged.len(),
            "gradients": averaged
        }))
        .into_response())
    }
}

/// `GET /api/cluster/status`.
pub async fn handle_cluster_status(State(state): State<Arc<AppState>>) -> Json<Value> {
    let workers = state.coordinator.list_workers().await;
    let now = ClusterCoordinator::now_ms();
    let master_vram_mb = telemetry::collect_for(&state)
        .await
        .vram_total_bytes
        .map(telemetry::bytes_to_mib);
    let workers_vram_mb: u64 = workers.iter().map(|worker| worker.vram_mb).sum();
    let online_workers = workers
        .iter()
        .filter(|worker| worker.is_online(now, WORKER_TIMEOUT_MS))
        .count();
    let latencies: Vec<f32> = workers
        .iter()
        .filter_map(|worker| worker.latency_ms)
        .collect();
    let average_latency_ms =
        (!latencies.is_empty()).then(|| latencies.iter().sum::<f32>() / latencies.len() as f32);

    let worker_statuses: Vec<Value> = workers
        .iter()
        .map(|worker| {
            json!({
                "workerId": worker.worker_id,
                "rank": worker.rank,
                "ip": worker.ip,
                "gpuName": worker.gpu_name,
                "vramMb": worker.vram_mb,
                "latencyMs": worker.latency_ms,
                "lastSeenMs": worker.last_seen_ms,
                "status": if worker.is_online(now, WORKER_TIMEOUT_MS) { "online" } else { "offline" }
            })
        })
        .collect();

    Json(json!({
        "role": "master",
        "worldSize": workers.len() + 1,
        "onlineWorkers": online_workers,
        "currentStep": state.coordinator.current_step.load(Ordering::Relaxed),
        "workers": worker_statuses,
        "masterVramMb": master_vram_mb,
        // Видеопамять мастера (если она известна) и всех узлов
        "totalVramMb": master_vram_mb.map(|master| master + workers_vram_mb),
        // AllReduce возможен, когда есть узел на связи и движок обучения (этап 3)
        "allReduceReady": online_workers > 0 && state.training.engine_available(),
        "averageLatencyMs": average_latency_ms
    }))
}
