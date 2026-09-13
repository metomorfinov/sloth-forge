//! Эндпоинты `/api/train/*` по контракту фронтенда (`features/training/api`).

use super::controller::{
    ProgressEvent, ResetOutcome, StartRequestRecord, StopOutcome, TrainingController,
    TrainingStatus,
};
use super::request::{TrainStartRequest, TrainingConfig};
use crate::error::{ApiError, ApiResult};
use crate::state::{AppState, TrainingRunSummary};
use crate::store::{self, StoreError};
use axum::extract::{Path, Query, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Json;
use futures_util::Stream;
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast::{self, error::RecvError};

/// Как часто поток прогресса повторяет текущее состояние, если новых шагов нет.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const DEFAULT_RUNS_PAGE: u32 = 50;
const MAX_RUNS_PAGE: u32 = 500;
const MAX_DISPLAY_NAME_CHARS: usize = 200;

fn non_empty(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// `POST /api/train/start`.
pub async fn start(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let request: TrainStartRequest = serde_json::from_value(body).map_err(|err| {
        ApiError::unprocessable(format!("Некорректные настройки обучения: {err}"))
    })?;
    let config = TrainingConfig::from_request(&request)?;
    let started = state
        .training
        .start(config, non_empty(request.start_request_id))
        .await?;
    Ok(Json(json!({
        "job_id": started.job_id,
        "status": "queued",
        "message": started.message,
        "error": null,
        "error_code": null
    })))
}

/// Тело stop/reset: к какому запуску относится команда.
#[derive(Debug, Default, Deserialize)]
pub struct JobScope {
    #[serde(default)]
    expected_job_id: Option<String>,
    /// Сохранять ли адаптер при остановке; сохранение появится вместе с движком (этап 3).
    #[serde(default)]
    #[allow(dead_code)]
    save: Option<bool>,
}

fn scope_job_id(body: Option<Json<JobScope>>) -> Option<String> {
    body.and_then(|Json(scope)| non_empty(scope.expected_job_id))
}

/// `POST /api/train/stop`: ответ приходит, когда движок уже остановился.
pub async fn stop(State(state): State<Arc<AppState>>, body: Option<Json<JobScope>>) -> Json<Value> {
    let expected = scope_job_id(body);
    let (status, message) = match state.training.stop(expected.as_deref()).await {
        StopOutcome::Stopped => ("stopped", "Обучение остановлено"),
        StopOutcome::Idle => ("idle", "Обучение сейчас не идёт"),
    };
    Json(json!({ "status": status, "message": message }))
}

/// `POST /api/train/reset`.
pub async fn reset(
    State(state): State<Arc<AppState>>,
    body: Option<Json<JobScope>>,
) -> ApiResult<Json<Value>> {
    let expected = scope_job_id(body);
    let status = match state.training.reset(expected.as_deref())? {
        ResetOutcome::Reset => "ok",
        ResetOutcome::Superseded => "superseded",
    };
    Ok(Json(json!({ "status": status })))
}

/// `GET /api/train/status`.
pub async fn status(State(state): State<Arc<AppState>>) -> Json<TrainingStatus> {
    Json(state.training.status())
}

#[derive(Debug, Default, Deserialize)]
pub struct JobQuery {
    #[serde(default)]
    expected_job_id: Option<String>,
}

/// `GET /api/train/metrics`.
pub async fn metrics(
    State(state): State<Arc<AppState>>,
    Query(query): Query<JobQuery>,
) -> Json<Value> {
    let expected = non_empty(query.expected_job_id);
    Json(state.training.metrics(expected.as_deref()))
}

struct ProgressStream {
    controller: Arc<TrainingController>,
    receiver: broadcast::Receiver<ProgressEvent>,
    expected_job_id: Option<String>,
    pending: Option<ProgressEvent>,
    finished: bool,
}

fn to_sse(event: &ProgressEvent) -> Event {
    Event::default()
        .event(event.kind.as_str())
        .id(event.id.to_string())
        .json_data(&event.payload)
        .unwrap_or_else(|err| {
            tracing::error!("Не удалось сериализовать событие прогресса: {err}");
            Event::default().comment("ошибка сериализации события")
        })
}

impl ProgressStream {
    fn deliver(mut self, event: ProgressEvent) -> Option<(Result<Event, Infallible>, Self)> {
        self.finished = event.kind.is_terminal();
        Some((Ok(to_sse(&event)), self))
    }

    async fn next(mut self) -> Option<(Result<Event, Infallible>, Self)> {
        if let Some(event) = self.pending.take() {
            return self.deliver(event);
        }
        if self.finished {
            return None;
        }
        loop {
            tokio::select! {
                received = self.receiver.recv() => match received {
                    Ok(event) => {
                        let foreign = self
                            .expected_job_id
                            .as_deref()
                            .is_some_and(|id| id != event.payload.job_id);
                        if !foreign {
                            return self.deliver(event);
                        }
                    }
                    Err(RecvError::Lagged(skipped)) => {
                        // Клиент не успевал читать: вместо пропущенных шагов — свежий снимок
                        tracing::debug!("Поток прогресса пропустил {skipped} событий");
                        let snapshot = self.controller.snapshot_event(self.expected_job_id.as_deref(), false);
                        if let Some(event) = snapshot {
                            return self.deliver(event);
                        }
                    }
                    Err(RecvError::Closed) => return None,
                },
                _ = tokio::time::sleep(HEARTBEAT_INTERVAL) => {
                    let snapshot = self.controller.snapshot_event(self.expected_job_id.as_deref(), true);
                    if let Some(event) = snapshot {
                        return self.deliver(event);
                    }
                }
            }
        }
    }
}

/// `GET|POST /api/train/progress`: всегда поток SSE (фронтенд не присылает `Accept`).
/// Сначала — текущее состояние запуска, дальше события `progress`, `heartbeat` и итоговое
/// `complete` или `error`, после которого поток закрывается.
pub async fn progress(
    State(state): State<Arc<AppState>>,
    Query(query): Query<JobQuery>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let controller = Arc::clone(&state.training);
    let expected_job_id = non_empty(query.expected_job_id);
    // Подписка до снимка: событие между ними не потеряется
    let receiver = controller.subscribe();
    let pending = controller.snapshot_event(expected_job_id.as_deref(), false);
    let stream = futures_util::stream::unfold(
        ProgressStream {
            controller,
            receiver,
            expected_job_id,
            pending,
            finished: false,
        },
        ProgressStream::next,
    );
    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[derive(Debug, Default, Deserialize)]
pub struct RunsQuery {
    #[serde(default)]
    limit: Option<u32>,
    #[serde(default)]
    offset: Option<u32>,
}

/// `GET /api/train/runs?limit&offset`: история запусков из базы, новые первыми.
pub async fn list_runs(
    State(state): State<Arc<AppState>>,
    Query(query): Query<RunsQuery>,
) -> ApiResult<Json<Value>> {
    let limit = query
        .limit
        .unwrap_or(DEFAULT_RUNS_PAGE)
        .clamp(1, MAX_RUNS_PAGE);
    let offset = query.offset.unwrap_or(0);
    let (runs, total) = state
        .store
        .call(move |conn| store::runs::list(conn, limit, offset))
        .await?;
    Ok(Json(json!({ "runs": runs, "total": total })))
}

async fn find_run(state: &AppState, id: &str) -> ApiResult<TrainingRunSummary> {
    let lookup = id.to_string();
    state
        .store
        .call(move |conn| store::runs::get(conn, &lookup))
        .await?
        .ok_or_else(|| ApiError::not_found(format!("Запуск обучения {id} не найден")))
}

/// `GET /api/train/runs/:id` (`TrainingRunDetailResponse` во фронтенде).
pub async fn get_run(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let run = find_run(&state, &id).await?;
    // Подробные метрики есть только у запуска, который ещё в памяти; для старых — только итоги
    let (config, history, last) = match state.training.run_details(&id) {
        Some((config, history, last)) => (config, history, Some(last)),
        None => (
            json!({
                "model_name": run.model_name,
                "dataset_name": run.dataset_name,
                "total_steps": run.total_steps
            }),
            Default::default(),
            None,
        ),
    };
    Ok(Json(json!({
        "run": run,
        "config": config,
        "metrics": {
            "step_history": history.steps,
            "loss_history": history.loss,
            "loss_step_history": history.steps,
            "lr_history": history.lr,
            "lr_step_history": history.steps,
            "grad_norm_history": history.grad_norm,
            "grad_norm_step_history": history.grad_norm_steps,
            "eval_loss_history": history.eval_loss,
            "eval_step_history": history.eval_steps,
            "final_epoch": last.and_then(|last| last.epoch),
            "final_num_tokens": last.and_then(|last| last.num_tokens)
        }
    })))
}

#[derive(Debug, Default, Deserialize)]
pub struct RenameRequest {
    #[serde(default)]
    display_name: Option<String>,
}

/// `PATCH /api/train/runs/:id`: переименование запуска.
pub async fn rename_run(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(request): Json<RenameRequest>,
) -> ApiResult<Json<TrainingRunSummary>> {
    let display_name = non_empty(request.display_name);
    if display_name
        .as_ref()
        .is_some_and(|name| name.chars().count() > MAX_DISPLAY_NAME_CHARS)
    {
        return Err(ApiError::unprocessable(format!(
            "Название запуска длиннее {MAX_DISPLAY_NAME_CHARS} символов"
        )));
    }
    let lookup = id.clone();
    let new_name = display_name.clone();
    let updated = state
        .store
        .call(
            move |conn| -> Result<Option<TrainingRunSummary>, StoreError> {
                let Some(mut run) = store::runs::get(conn, &lookup)? else {
                    return Ok(None);
                };
                run.display_name = new_name;
                store::runs::upsert(conn, &run)?;
                Ok(Some(run))
            },
        )
        .await?
        .ok_or_else(|| ApiError::not_found(format!("Запуск обучения {id} не найден")))?;
    state.training.rename(&id, display_name);
    Ok(Json(updated))
}

/// `DELETE /api/train/runs/:id`: удаляет запуск из истории.
pub async fn delete_run(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    if state.training.running_job_id().as_deref() == Some(id.as_str()) {
        return Err(ApiError::conflict(
            "Идущий запуск нельзя удалить: сначала остановите обучение",
        ));
    }
    let lookup = id.clone();
    let deleted = state
        .store
        .call(move |conn| store::runs::delete(conn, &lookup))
        .await?;
    if !deleted {
        return Err(ApiError::not_found(format!(
            "Запуск обучения {id} не найден"
        )));
    }
    state.training.forget(&id);
    Ok(Json(json!({
        "status": "ok",
        "message": "Запуск удалён из истории",
        // Движок пока не сохраняет адаптеры на диск, удалять нечего
        "artifacts_deleted": false,
        "artifacts_kept_reason": null
    })))
}

/// `GET /api/train/start-requests/:id`: судьба запроса на старт.
pub async fn get_start_request(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> ApiResult<Json<StartRequestRecord>> {
    state
        .training
        .start_request(&id)
        .map(Json)
        .ok_or_else(|| ApiError::not_found(format!("Запрос на старт {id} не найден")))
}

/// `POST /api/train/start-requests/:id/acknowledge`: интерфейс увидел результат.
pub async fn acknowledge_start_request(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    if state.training.acknowledge_start_request(&id) {
        Ok(Json(json!({ "status": "ok" })))
    } else {
        Err(ApiError::not_found(format!(
            "Запрос на старт {id} не найден"
        )))
    }
}

/// `POST /api/train/start-requests/:id/cancel`. Старт выполняется сразу, поэтому отменять
/// нечего: возвращается итог запроса (для идущего запуска нужен «Стоп»).
pub async fn cancel_start_request(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> ApiResult<Json<StartRequestRecord>> {
    get_start_request(State(state), Path(id)).await
}
