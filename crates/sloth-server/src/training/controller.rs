//! Жизненный цикл запуска обучения.
//!
//! Контроллер не умеет обучать модель — это делает [`TrainingEngine`]. Он отвечает за всё
//! вокруг: уникальные id, фазы, остановку с ожиданием движка, историю метрик, события
//! прогресса и сохранение запусков в SQLite. Благодаря этому движок этапа 3 подключится
//! без переделки API, а поведение API проверяется уже сейчас на учебном движке в тестах.

use super::request::TrainingConfig;
use crate::error::{ApiError, ApiResult};
use crate::state::TrainingRunSummary;
use crate::store::{self, Store, StoreError};
use futures_util::future::{BoxFuture, FutureExt};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, watch};
use tokio::task::JoinHandle;

/// Сколько точек метрик текущего запуска хранится в памяти для графиков.
const MAX_HISTORY_POINTS: usize = 10_000;
/// Сколько значений loss попадает в мини-график истории запусков.
const SPARKLINE_POINTS: usize = 64;
/// Сколько последних запросов на старт помнит сервер (интерфейс переспрашивает их при обрыве).
const MAX_START_REQUESTS: usize = 32;
/// Сколько ждать, пока движок остановится после команды «Стоп».
const STOP_WAIT_TIMEOUT: Duration = Duration::from_secs(60);
/// Буфер событий; отставший подписчик получает свежий снимок вместо пропущенного.
const EVENT_BUFFER: usize = 256;

/// Пока движка нет, старт отклоняется с этим текстом.
pub const ENGINE_UNAVAILABLE_MESSAGE: &str =
    "Движок обучения SlothForge (LoRA на Vulkan) появится на этапе 3 дорожной карты";
pub const ENGINE_UNAVAILABLE_CODE: &str = "engine_unavailable";
const INTERRUPTED_MESSAGE: &str = "Сервер был перезапущен во время обучения";

const START_ACCEPTED: &str = "accepted";
const START_REJECTED: &str = "rejected";

/// Результат движка: `Ok(())` — шаги пройдены или запуск остановлен по [`StopSignal`],
/// `Err` — ошибка с понятным пользователю текстом.
pub type EngineFuture = BoxFuture<'static, Result<(), String>>;

/// То, что выполняет обучение. Вызывается один раз на запуск.
pub trait TrainingEngine: Send + Sync + 'static {
    fn train(
        &self,
        config: TrainingConfig,
        reporter: ProgressReporter,
        stop: StopSignal,
    ) -> EngineFuture;
}

/// Фазы запуска, как их понимает фронтенд (`TrainingPhase`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingPhase {
    LoadingModel,
    LoadingDataset,
    Configuring,
    Training,
    Finalizing,
    Completed,
    Error,
    Stopped,
}

impl TrainingPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            TrainingPhase::LoadingModel => "loading_model",
            TrainingPhase::LoadingDataset => "loading_dataset",
            TrainingPhase::Configuring => "configuring",
            TrainingPhase::Training => "training",
            TrainingPhase::Finalizing => "finalizing",
            TrainingPhase::Completed => "completed",
            TrainingPhase::Error => "error",
            TrainingPhase::Stopped => "stopped",
        }
    }

    /// Запуск ещё идёт (движок работает).
    pub fn is_active(self) -> bool {
        !matches!(
            self,
            TrainingPhase::Completed | TrainingPhase::Error | TrainingPhase::Stopped
        )
    }
}

/// Что движок сообщает после шага. `loss` и `learning_rate` передаются вместе:
/// точка графика записывается, только когда известны оба значения.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ProgressUpdate {
    pub step: u32,
    pub total_steps: u32,
    pub epoch: Option<u32>,
    pub loss: Option<f32>,
    pub learning_rate: Option<f32>,
    pub grad_norm: Option<f32>,
    pub num_tokens: Option<u64>,
    pub eval_loss: Option<f32>,
}

/// Данные события прогресса (`TrainingProgressPayload` во фронтенде).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProgressPayload {
    pub job_id: String,
    pub step: u32,
    pub total_steps: u32,
    pub loss: Option<f32>,
    pub learning_rate: Option<f32>,
    pub progress_percent: f64,
    pub epoch: Option<u32>,
    pub elapsed_seconds: Option<u64>,
    pub eta_seconds: Option<u64>,
    pub grad_norm: Option<f32>,
    pub num_tokens: Option<u64>,
    pub eval_loss: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Progress,
    Heartbeat,
    Complete,
    Error,
}

impl EventKind {
    /// Имя события SSE, как его ждёт фронтенд.
    pub fn as_str(self) -> &'static str {
        match self {
            EventKind::Progress => "progress",
            EventKind::Heartbeat => "heartbeat",
            EventKind::Complete => "complete",
            EventKind::Error => "error",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, EventKind::Complete | EventKind::Error)
    }
}

#[derive(Debug, Clone)]
pub struct ProgressEvent {
    pub kind: EventKind,
    /// Возрастающий номер события (поле `id` в SSE).
    pub id: u64,
    pub payload: ProgressPayload,
}

/// Метрики текущего запуска для графиков (`metric_history` во фронтенде).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct MetricHistory {
    pub steps: Vec<u32>,
    pub loss: Vec<f32>,
    pub lr: Vec<f32>,
    pub grad_norm: Vec<f32>,
    pub grad_norm_steps: Vec<u32>,
    pub eval_loss: Vec<f32>,
    pub eval_steps: Vec<u32>,
}

/// Отбрасывает старую половину ряда, когда он дорастает до предела.
fn trim_front<T>(series: &mut Vec<T>) {
    if series.len() >= MAX_HISTORY_POINTS {
        series.drain(..MAX_HISTORY_POINTS / 2);
    }
}

impl MetricHistory {
    fn record(&mut self, update: &ProgressUpdate) {
        if let (Some(loss), Some(lr)) = (update.loss, update.learning_rate) {
            trim_front(&mut self.steps);
            trim_front(&mut self.loss);
            trim_front(&mut self.lr);
            self.steps.push(update.step);
            self.loss.push(loss);
            self.lr.push(lr);
        }
        if let Some(grad_norm) = update.grad_norm {
            trim_front(&mut self.grad_norm_steps);
            trim_front(&mut self.grad_norm);
            self.grad_norm_steps.push(update.step);
            self.grad_norm.push(grad_norm);
        }
        if let Some(eval_loss) = update.eval_loss {
            trim_front(&mut self.eval_steps);
            trim_front(&mut self.eval_loss);
            self.eval_steps.push(update.step);
            self.eval_loss.push(eval_loss);
        }
    }

    fn is_empty(&self) -> bool {
        self.steps.is_empty() && self.grad_norm_steps.is_empty() && self.eval_steps.is_empty()
    }
}

/// Запрос на старт (`TrainingStartRequestStatusResponse` во фронтенде).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StartRequestRecord {
    pub start_request_id: String,
    pub job_id: String,
    pub state: &'static str,
    pub message: String,
    pub error: Option<String>,
    pub error_code: Option<&'static str>,
}

/// Статус обучения (`TrainingStatusResponse` во фронтенде).
#[derive(Debug, Clone, Serialize)]
pub struct TrainingStatus {
    pub job_id: String,
    pub start_request_id: Option<String>,
    pub start_request_state: Option<&'static str>,
    pub phase: &'static str,
    pub is_training_running: bool,
    pub eval_enabled: bool,
    pub message: String,
    pub error: Option<String>,
    pub warnings: Vec<String>,
    pub details: Option<StatusDetails>,
    pub metric_history: Option<MetricHistory>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StatusDetails {
    pub epoch: Option<u32>,
    pub step: u32,
    pub total_steps: u32,
    pub loss: Option<f32>,
    pub learning_rate: Option<f32>,
}

pub struct StartedJob {
    pub job_id: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopOutcome {
    Stopped,
    /// Останавливать нечего (или команда относилась к другому запуску).
    Idle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetOutcome {
    Reset,
    /// Сброс относился к запуску, который уже сменился другим.
    Superseded,
}

struct Job {
    id: String,
    start_request_id: Option<String>,
    phase: TrainingPhase,
    message: String,
    error: Option<String>,
    display_name: Option<String>,
    config: TrainingConfig,
    started_at: String,
    started: Instant,
    ended_at: Option<String>,
    finished_after: Option<Duration>,
    last: ProgressUpdate,
    history: MetricHistory,
    stop: watch::Sender<bool>,
    stop_requested: bool,
    task: Option<JoinHandle<()>>,
}

impl Job {
    fn new(
        id: String,
        start_request_id: Option<String>,
        config: TrainingConfig,
        stop: watch::Sender<bool>,
    ) -> Self {
        Self {
            last: ProgressUpdate {
                total_steps: config.max_steps.unwrap_or(0),
                ..ProgressUpdate::default()
            },
            id,
            start_request_id,
            phase: TrainingPhase::Configuring,
            message: "Подготовка к обучению".to_string(),
            error: None,
            display_name: None,
            config,
            started_at: crate::state::iso_now(),
            started: Instant::now(),
            ended_at: None,
            finished_after: None,
            history: MetricHistory::default(),
            stop,
            stop_requested: false,
            task: None,
        }
    }

    fn elapsed(&self) -> Duration {
        self.finished_after
            .unwrap_or_else(|| self.started.elapsed())
    }

    fn payload(&self) -> ProgressPayload {
        let ProgressUpdate {
            step, total_steps, ..
        } = self.last;
        let elapsed = self.elapsed();
        let progress_percent = if total_steps > 0 {
            (f64::from(step) / f64::from(total_steps) * 100.0).min(100.0)
        } else {
            0.0
        };
        // Оценка оставшегося времени по средней длительности пройденных шагов
        let eta_seconds = (self.phase == TrainingPhase::Training && step > 0 && total_steps > step)
            .then(|| {
                (elapsed.as_secs_f64() / f64::from(step) * f64::from(total_steps - step)).round()
                    as u64
            });
        ProgressPayload {
            job_id: self.id.clone(),
            step,
            total_steps,
            loss: self.last.loss,
            learning_rate: self.last.learning_rate,
            progress_percent,
            epoch: self.last.epoch,
            elapsed_seconds: Some(elapsed.as_secs()),
            eta_seconds,
            grad_norm: self.last.grad_norm,
            num_tokens: self.last.num_tokens,
            eval_loss: self.last.eval_loss,
        }
    }

    fn summary(&self) -> TrainingRunSummary {
        let finished = !self.phase.is_active();
        TrainingRunSummary {
            id: self.id.clone(),
            status: match self.phase {
                TrainingPhase::Completed => "completed",
                TrainingPhase::Error => "error",
                TrainingPhase::Stopped => "stopped",
                _ => "running",
            }
            .to_string(),
            model_name: self.config.model.clone(),
            project_name: self.config.project_name.clone(),
            dataset_name: self.config.dataset.clone(),
            display_name: self.display_name.clone(),
            started_at: self.started_at.clone(),
            ended_at: self.ended_at.clone(),
            total_steps: (self.last.total_steps > 0).then_some(self.last.total_steps),
            final_step: finished.then_some(self.last.step),
            final_loss: if finished { self.last.loss } else { None },
            // Движок пока ничего не сохраняет на диск
            output_dir: None,
            can_resume: false,
            resume_blocked_reason: None,
            resumed_later: false,
            has_preview_model: false,
            preview_ref: None,
            preview_sig: None,
            duration_seconds: self.finished_after.map(|duration| duration.as_secs()),
            error_message: self.error.clone(),
            loss_sparkline: Some(sparkline(&self.history.loss)),
        }
    }
}

/// Мини-график: не больше [`SPARKLINE_POINTS`] значений, равномерно по всему ряду.
fn sparkline(values: &[f32]) -> Vec<f32> {
    if values.len() <= SPARKLINE_POINTS {
        return values.to_vec();
    }
    let last = values.len() - 1;
    (0..SPARKLINE_POINTS)
        .map(|i| values[i * last / (SPARKLINE_POINTS - 1)])
        .collect()
}

#[derive(Default)]
struct Inner {
    /// Текущий или последний завершённый запуск.
    job: Option<Job>,
    start_requests: VecDeque<StartRequestRecord>,
}

struct Shared {
    inner: Mutex<Inner>,
    events: broadcast::Sender<ProgressEvent>,
    next_event_id: AtomicU64,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        // Внутри только данные о запуске: после паники в другом потоке их можно читать дальше
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn next_id(&self) -> u64 {
        self.next_event_id.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn emit(&self, kind: EventKind, payload: ProgressPayload) {
        let event = ProgressEvent {
            kind,
            id: self.next_id(),
            payload,
        };
        // Ошибка отправки означает только «сейчас никто не подписан на события»
        let _ = self.events.send(event);
    }
}

/// Через него движок сообщает фазы и прогресс своего запуска.
#[derive(Clone)]
pub struct ProgressReporter {
    shared: Arc<Shared>,
    job_id: String,
}

impl ProgressReporter {
    pub fn job_id(&self) -> &str {
        &self.job_id
    }

    /// Промежуточная фаза (загрузка модели, датасета, сохранение). Итог решает контроллер.
    pub fn set_phase(&self, phase: TrainingPhase, message: impl Into<String>) {
        if !phase.is_active() {
            tracing::warn!("Движок пытался сам задать итоговую фазу {}", phase.as_str());
            return;
        }
        let payload = {
            let mut inner = self.shared.lock();
            let Some(job) = inner
                .job
                .as_mut()
                .filter(|job| job.id == self.job_id && job.phase.is_active())
            else {
                return;
            };
            job.phase = phase;
            job.message = message.into();
            job.payload()
        };
        self.shared.emit(EventKind::Progress, payload);
    }

    /// Шаг обучения пройден.
    pub fn report(&self, update: ProgressUpdate) {
        let payload = {
            let mut inner = self.shared.lock();
            let Some(job) = inner
                .job
                .as_mut()
                .filter(|job| job.id == self.job_id && job.phase.is_active())
            else {
                return;
            };
            job.phase = TrainingPhase::Training;
            job.message = format!("Шаг {} из {}", update.step, update.total_steps);
            job.history.record(&update);
            job.last = update;
            job.payload()
        };
        self.shared.emit(EventKind::Progress, payload);
    }
}

/// Сигнал «Стоп» для движка.
#[derive(Clone)]
pub struct StopSignal(watch::Receiver<bool>);

impl StopSignal {
    pub fn is_stopped(&self) -> bool {
        *self.0.borrow()
    }

    /// Завершается, когда пользователь нажал «Стоп».
    pub async fn stopped(&mut self) {
        while !*self.0.borrow_and_update() {
            if self.0.changed().await.is_err() {
                // Контроллер уничтожен вместе с запуском: ждать больше нечего
                return;
            }
        }
    }
}

static JOB_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Уникальный id: время старта в миллисекундах и порядковый номер внутри процесса.
fn new_job_id() -> String {
    format!(
        "train-{}-{}",
        store::now_ms(),
        JOB_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    let detail = panic
        .downcast_ref::<&str>()
        .map(|text| text.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "без описания".to_string());
    format!("Движок обучения аварийно завершился: {detail}")
}

pub struct TrainingController {
    shared: Arc<Shared>,
    store: Arc<Store>,
    engine: Option<Arc<dyn TrainingEngine>>,
}

impl TrainingController {
    /// `engine: None` — движка нет, старт отклоняется с 503.
    pub fn new(store: Arc<Store>, engine: Option<Arc<dyn TrainingEngine>>) -> Self {
        recover_interrupted_runs(&store);
        let (events, _) = broadcast::channel(EVENT_BUFFER);
        Self {
            shared: Arc::new(Shared {
                inner: Mutex::new(Inner::default()),
                events,
                next_event_id: AtomicU64::new(0),
            }),
            store,
            engine,
        }
    }

    pub fn engine_available(&self) -> bool {
        self.engine.is_some()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ProgressEvent> {
        self.shared.events.subscribe()
    }

    /// Id идущего запуска.
    pub fn running_job_id(&self) -> Option<String> {
        self.shared
            .lock()
            .job
            .as_ref()
            .filter(|job| job.phase.is_active())
            .map(|job| job.id.clone())
    }

    pub async fn start(
        &self,
        config: TrainingConfig,
        start_request_id: Option<String>,
    ) -> ApiResult<StartedJob> {
        // Повтор того же запроса (интерфейс переспрашивает при обрыве связи) — тот же запуск
        if let Some(request_id) = start_request_id.as_deref() {
            if let Some(record) = self.shared.lock().start_requests.iter().find(|record| {
                record.start_request_id == request_id && record.state == START_ACCEPTED
            }) {
                return Ok(StartedJob {
                    job_id: record.job_id.clone(),
                    message: record.message.clone(),
                });
            }
        }

        let Some(engine) = self.engine.clone() else {
            self.remember_start_request(
                start_request_id,
                String::new(),
                START_REJECTED,
                ENGINE_UNAVAILABLE_MESSAGE,
                Some(ENGINE_UNAVAILABLE_CODE),
            );
            return Err(ApiError::service_unavailable(ENGINE_UNAVAILABLE_MESSAGE));
        };

        let job_id = new_job_id();
        let (stop_tx, stop_rx) = watch::channel(false);
        let summary = {
            let mut inner = self.shared.lock();
            if inner.job.as_ref().is_some_and(|job| job.phase.is_active()) {
                return Err(ApiError::conflict(
                    "Обучение уже идёт: остановите текущий запуск, прежде чем начинать новый",
                ));
            }
            let job = Job::new(
                job_id.clone(),
                start_request_id.clone(),
                config.clone(),
                stop_tx,
            );
            let summary = job.summary();
            inner.job = Some(job);
            summary
        };
        if let Err(err) = save_run(&self.store, summary).await {
            // Без записи в историю запуск не начинается
            self.shared.lock().job = None;
            return Err(err.into());
        }

        let reporter = ProgressReporter {
            shared: Arc::clone(&self.shared),
            job_id: job_id.clone(),
        };
        let training = engine.train(config, reporter, StopSignal(stop_rx));
        let shared = Arc::clone(&self.shared);
        let store = Arc::clone(&self.store);
        let task_job_id = job_id.clone();
        let task = tokio::spawn(async move {
            let result = AssertUnwindSafe(training)
                .catch_unwind()
                .await
                .unwrap_or_else(|panic| Err(panic_message(panic)));
            finish_job(&shared, &store, &task_job_id, result).await;
        });
        if let Some(job) = self
            .shared
            .lock()
            .job
            .as_mut()
            .filter(|job| job.id == job_id)
        {
            job.task = Some(task);
        }

        let message = "Обучение запущено".to_string();
        self.remember_start_request(
            start_request_id,
            job_id.clone(),
            START_ACCEPTED,
            &message,
            None,
        );
        tracing::info!("Запущено обучение {job_id}");
        Ok(StartedJob { job_id, message })
    }

    /// Останавливает запуск и ждёт, пока движок действительно завершится: иначе быстрый
    /// «Стоп → Старт» мог бы наложить новый запуск на ещё работающий старый.
    pub async fn stop(&self, expected_job_id: Option<&str>) -> StopOutcome {
        let task = {
            let mut inner = self.shared.lock();
            let Some(job) = inner.job.as_mut() else {
                return StopOutcome::Idle;
            };
            if !job.phase.is_active() || expected_job_id.is_some_and(|id| id != job.id) {
                return StopOutcome::Idle;
            }
            job.stop_requested = true;
            job.message = "Останавливаем обучение".to_string();
            job.stop.send_replace(true);
            job.task.take()
        };
        if let Some(task) = task {
            match tokio::time::timeout(STOP_WAIT_TIMEOUT, task).await {
                Ok(Ok(())) => {}
                Ok(Err(err)) => tracing::error!("Задача обучения завершилась аварийно: {err}"),
                Err(_) => tracing::warn!(
                    "Движок обучения не остановился за {} с",
                    STOP_WAIT_TIMEOUT.as_secs()
                ),
            }
        }
        StopOutcome::Stopped
    }

    /// Убирает завершённый запуск с экрана «Текущий запуск». История в базе не меняется.
    pub fn reset(&self, expected_job_id: Option<&str>) -> ApiResult<ResetOutcome> {
        let mut inner = self.shared.lock();
        let Some(job) = inner.job.as_ref() else {
            return Ok(ResetOutcome::Reset);
        };
        if expected_job_id.is_some_and(|id| id != job.id) {
            return Ok(ResetOutcome::Superseded);
        }
        if job.phase.is_active() {
            return Err(ApiError::conflict(
                "Нельзя сбросить идущее обучение: сначала остановите его",
            ));
        }
        inner.job = None;
        Ok(ResetOutcome::Reset)
    }

    pub fn status(&self) -> TrainingStatus {
        let inner = self.shared.lock();
        let Some(job) = inner.job.as_ref() else {
            return TrainingStatus {
                job_id: String::new(),
                start_request_id: None,
                start_request_state: None,
                phase: "idle",
                is_training_running: false,
                eval_enabled: false,
                message: if self.engine.is_some() {
                    "Обучение ещё не запускалось".to_string()
                } else {
                    ENGINE_UNAVAILABLE_MESSAGE.to_string()
                },
                error: None,
                warnings: Vec::new(),
                details: None,
                metric_history: None,
            };
        };
        TrainingStatus {
            job_id: job.id.clone(),
            start_request_state: job.start_request_id.as_ref().map(|_| START_ACCEPTED),
            start_request_id: job.start_request_id.clone(),
            phase: job.phase.as_str(),
            is_training_running: job.phase.is_active(),
            eval_enabled: false,
            message: job.message.clone(),
            error: job.error.clone(),
            warnings: Vec::new(),
            details: Some(StatusDetails {
                epoch: job.last.epoch,
                step: job.last.step,
                total_steps: job.last.total_steps,
                loss: job.last.loss,
                learning_rate: job.last.learning_rate,
            }),
            metric_history: (!job.history.is_empty()).then(|| job.history.clone()),
        }
    }

    /// `GET /api/train/metrics` (`TrainingMetricsResponse` во фронтенде).
    pub fn metrics(&self, expected_job_id: Option<&str>) -> Value {
        let inner = self.shared.lock();
        let job = inner
            .job
            .as_ref()
            .filter(|job| expected_job_id.is_none_or(|id| id == job.id));
        let job_id = expected_job_id
            .map(str::to_string)
            .or_else(|| job.map(|job| job.id.clone()))
            .unwrap_or_default();
        match job {
            Some(job) => json!({
                "job_id": job_id,
                "loss_history": job.history.loss,
                "lr_history": job.history.lr,
                "step_history": job.history.steps,
                "grad_norm_history": job.history.grad_norm,
                "grad_norm_step_history": job.history.grad_norm_steps,
                "current_loss": job.last.loss,
                "current_lr": job.last.learning_rate,
                "current_step": (job.last.step > 0).then_some(job.last.step)
            }),
            None => json!({
                "job_id": job_id,
                "loss_history": [],
                "lr_history": [],
                "step_history": [],
                "grad_norm_history": [],
                "grad_norm_step_history": [],
                "current_loss": null,
                "current_lr": null,
                "current_step": null
            }),
        }
    }

    /// Текущее состояние запуска в виде события: для нового подписчика потока и для
    /// «пульса». `None`, если запуска нет или он не тот, что ожидает клиент.
    pub fn snapshot_event(
        &self,
        expected_job_id: Option<&str>,
        heartbeat: bool,
    ) -> Option<ProgressEvent> {
        let (kind, payload) = {
            let inner = self.shared.lock();
            let job = inner
                .job
                .as_ref()
                .filter(|job| expected_job_id.is_none_or(|id| id == job.id))?;
            let kind = match job.phase {
                TrainingPhase::Error => EventKind::Error,
                phase if !phase.is_active() => EventKind::Complete,
                _ if heartbeat => EventKind::Heartbeat,
                _ => EventKind::Progress,
            };
            (kind, job.payload())
        };
        Some(ProgressEvent {
            kind,
            id: self.shared.next_id(),
            payload,
        })
    }

    /// Настройки, метрики и последний шаг запуска, если он ещё в памяти.
    pub fn run_details(&self, job_id: &str) -> Option<(Value, MetricHistory, ProgressUpdate)> {
        let inner = self.shared.lock();
        let job = inner.job.as_ref().filter(|job| job.id == job_id)?;
        let config = match serde_json::to_value(&job.config) {
            Ok(config) => config,
            Err(err) => {
                tracing::error!("Не удалось представить настройки запуска {job_id}: {err}");
                Value::Null
            }
        };
        Some((config, job.history.clone(), job.last))
    }

    /// Новое имя запуска в памяти (чтобы итоговая запись в базу его не затёрла).
    pub fn rename(&self, job_id: &str, display_name: Option<String>) {
        if let Some(job) = self
            .shared
            .lock()
            .job
            .as_mut()
            .filter(|job| job.id == job_id)
        {
            job.display_name = display_name;
        }
    }

    /// Забывает завершённый запуск (после удаления из истории).
    pub fn forget(&self, job_id: &str) {
        let mut inner = self.shared.lock();
        if inner
            .job
            .as_ref()
            .is_some_and(|job| job.id == job_id && !job.phase.is_active())
        {
            inner.job = None;
        }
    }

    pub fn start_request(&self, start_request_id: &str) -> Option<StartRequestRecord> {
        self.shared
            .lock()
            .start_requests
            .iter()
            .find(|record| record.start_request_id == start_request_id)
            .cloned()
    }

    /// Интерфейс получил ответ на старт: запрос можно забыть.
    pub fn acknowledge_start_request(&self, start_request_id: &str) -> bool {
        let mut inner = self.shared.lock();
        let before = inner.start_requests.len();
        inner
            .start_requests
            .retain(|record| record.start_request_id != start_request_id);
        inner.start_requests.len() != before
    }

    fn remember_start_request(
        &self,
        start_request_id: Option<String>,
        job_id: String,
        state: &'static str,
        message: &str,
        error_code: Option<&'static str>,
    ) {
        let Some(start_request_id) = start_request_id.filter(|id| !id.trim().is_empty()) else {
            return;
        };
        let mut inner = self.shared.lock();
        inner
            .start_requests
            .retain(|record| record.start_request_id != start_request_id);
        inner.start_requests.push_back(StartRequestRecord {
            start_request_id,
            job_id,
            state,
            message: message.to_string(),
            error: (state == START_REJECTED).then(|| message.to_string()),
            error_code,
        });
        while inner.start_requests.len() > MAX_START_REQUESTS {
            inner.start_requests.pop_front();
        }
    }
}

async fn save_run(store: &Arc<Store>, summary: TrainingRunSummary) -> Result<(), StoreError> {
    store
        .call(move |conn| store::runs::upsert(conn, &summary))
        .await
}

/// Итог запуска: фаза, событие complete/error и запись в историю.
async fn finish_job(shared: &Shared, store: &Arc<Store>, job_id: &str, result: Result<(), String>) {
    let (summary, kind, payload) = {
        let mut inner = shared.lock();
        let Some(job) = inner.job.as_mut().filter(|job| job.id == job_id) else {
            return;
        };
        job.finished_after = Some(job.started.elapsed());
        job.ended_at = Some(crate::state::iso_now());
        match result {
            Ok(()) if job.stop_requested => {
                job.phase = TrainingPhase::Stopped;
                job.message = "Обучение остановлено".to_string();
            }
            Ok(()) => {
                job.phase = TrainingPhase::Completed;
                job.message = "Обучение завершено".to_string();
            }
            Err(message) => {
                tracing::warn!("Обучение {job_id} завершилось ошибкой: {message}");
                job.phase = TrainingPhase::Error;
                job.error = Some(message.clone());
                job.message = message;
            }
        }
        let kind = if job.phase == TrainingPhase::Error {
            EventKind::Error
        } else {
            EventKind::Complete
        };
        (job.summary(), kind, job.payload())
    };
    shared.emit(kind, payload);
    if let Err(err) = save_run(store, summary).await {
        tracing::error!("Не удалось сохранить итог обучения {job_id}: {err}");
    }
}

/// Запуски, которые в базе остались «running» после остановки сервера, помечаются ошибкой:
/// иначе история вечно показывала бы идущее обучение, которого нет.
fn recover_interrupted_runs(store: &Store) {
    let result: Result<usize, StoreError> = store.with_conn(|conn| {
        let mut recovered = 0;
        for mut run in store::runs::all(conn)? {
            if run.status == "running" {
                run.status = "error".to_string();
                run.error_message = Some(INTERRUPTED_MESSAGE.to_string());
                store::runs::upsert(conn, &run)?;
                recovered += 1;
            }
        }
        Ok(recovered)
    });
    match result {
        Ok(0) => {}
        Ok(count) => tracing::warn!("Прерванных запусков обучения помечено ошибкой: {count}"),
        Err(err) => tracing::error!("Не удалось проверить прерванные запуски обучения: {err}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::request::TrainStartRequest;

    const WAIT: Duration = Duration::from_secs(10);

    /// Учебный движок: проходит `steps` шагов с паузой `delay`, может упасть на шаге `fail_at`.
    struct StepEngine {
        steps: u32,
        delay: Duration,
        fail_at: Option<u32>,
        panic: bool,
    }

    impl StepEngine {
        fn new(steps: u32, delay: Duration) -> Arc<Self> {
            Arc::new(Self {
                steps,
                delay,
                fail_at: None,
                panic: false,
            })
        }
    }

    impl TrainingEngine for StepEngine {
        fn train(
            &self,
            _config: TrainingConfig,
            reporter: ProgressReporter,
            mut stop: StopSignal,
        ) -> EngineFuture {
            let (steps, delay, fail_at, panic) = (self.steps, self.delay, self.fail_at, self.panic);
            Box::pin(async move {
                reporter.set_phase(TrainingPhase::LoadingModel, "Загрузка модели");
                if panic {
                    panic!("сломанный движок");
                }
                for step in 1..=steps {
                    if fail_at == Some(step) {
                        return Err("Не хватило видеопамяти".to_string());
                    }
                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
                        _ = stop.stopped() => return Ok(()),
                    }
                    reporter.report(ProgressUpdate {
                        step,
                        total_steps: steps,
                        epoch: Some(1),
                        loss: Some(1.0 / step as f32),
                        learning_rate: Some(2e-4),
                        ..ProgressUpdate::default()
                    });
                }
                Ok(())
            })
        }
    }

    fn config() -> TrainingConfig {
        TrainingConfig::from_request(&TrainStartRequest {
            model_name: Some("test/model".into()),
            hf_dataset: Some("test/dataset".into()),
            ..TrainStartRequest::default()
        })
        .unwrap()
    }

    fn controller(engine: Option<Arc<dyn TrainingEngine>>) -> (TrainingController, Arc<Store>) {
        let store = Arc::new(Store::open_in_memory().unwrap());
        (TrainingController::new(Arc::clone(&store), engine), store)
    }

    async fn wait_terminal(
        events: &mut broadcast::Receiver<ProgressEvent>,
        job_id: &str,
    ) -> (EventKind, Vec<ProgressEvent>) {
        let mut seen = Vec::new();
        loop {
            let event = tokio::time::timeout(WAIT, events.recv())
                .await
                .expect("запуск не завершился вовремя")
                .expect("поток событий закрыт");
            if event.payload.job_id != job_id {
                continue;
            }
            let kind = event.kind;
            seen.push(event);
            if kind.is_terminal() {
                return (kind, seen);
            }
        }
    }

    async fn stored_run(store: &Arc<Store>, id: &str) -> TrainingRunSummary {
        let id = id.to_string();
        store
            .call(move |conn| store::runs::get(conn, &id))
            .await
            .unwrap()
            .expect("запуск сохранён")
    }

    #[tokio::test]
    async fn without_engine_start_is_rejected_and_remembered() {
        let (controller, _) = controller(None);
        let error = controller
            .start(config(), Some("req-1".into()))
            .await
            .err()
            .expect("старт отклонён");
        assert_eq!(error.status, axum::http::StatusCode::SERVICE_UNAVAILABLE);
        let record = controller.start_request("req-1").unwrap();
        assert_eq!(record.state, START_REJECTED);
        assert_eq!(record.error_code, Some(ENGINE_UNAVAILABLE_CODE));
        assert_eq!(controller.status().phase, "idle");
        assert!(controller.acknowledge_start_request("req-1"));
        assert!(controller.start_request("req-1").is_none());
    }

    #[tokio::test]
    async fn completed_run_reports_progress_and_is_saved() {
        let (controller, store) = controller(Some(StepEngine::new(5, Duration::from_millis(5))));
        let mut events = controller.subscribe();
        let job = controller
            .start(config(), Some("req-2".into()))
            .await
            .unwrap();

        let (kind, seen) = wait_terminal(&mut events, &job.job_id).await;
        assert_eq!(kind, EventKind::Complete);
        assert!(seen
            .iter()
            .any(|e| e.kind == EventKind::Progress && e.payload.step == 3));
        assert!(
            seen.windows(2).all(|pair| pair[0].id < pair[1].id),
            "номера событий растут"
        );

        let status = controller.status();
        assert_eq!(status.phase, "completed");
        assert!(!status.is_training_running);
        assert_eq!(status.metric_history.unwrap().steps, [1, 2, 3, 4, 5]);

        let run = stored_run(&store, &job.job_id).await;
        assert_eq!(run.status, "completed");
        assert_eq!(run.final_step, Some(5));
        assert_eq!(run.loss_sparkline.map(|s| s.len()), Some(5));

        // Тот же start_request_id — тот же запуск, а не второй
        let again = controller
            .start(config(), Some("req-2".into()))
            .await
            .unwrap();
        assert_eq!(again.job_id, job.job_id);
    }

    #[tokio::test]
    async fn stop_waits_for_engine_and_allows_immediate_restart() {
        let (controller, store) =
            controller(Some(StepEngine::new(100_000, Duration::from_millis(2))));
        let first = controller.start(config(), None).await.unwrap();
        assert!(
            controller.start(config(), None).await.is_err(),
            "второй старт — 409"
        );
        assert!(
            controller.reset(None).is_err(),
            "идущий запуск не сбрасывается"
        );

        assert_eq!(controller.stop(Some("other-job")).await, StopOutcome::Idle);
        assert_eq!(
            controller.stop(Some(&first.job_id)).await,
            StopOutcome::Stopped
        );
        // stop дождался движка: фаза уже итоговая, и новый старт сразу проходит
        assert_eq!(controller.status().phase, "stopped");
        assert_eq!(stored_run(&store, &first.job_id).await.status, "stopped");
        let second = controller.start(config(), None).await.unwrap();
        assert_ne!(second.job_id, first.job_id, "id запусков уникальны");

        assert_eq!(controller.stop(None).await, StopOutcome::Stopped);
        assert_eq!(
            controller.reset(Some(&first.job_id)).unwrap(),
            ResetOutcome::Superseded
        );
        assert_eq!(
            controller.reset(Some(&second.job_id)).unwrap(),
            ResetOutcome::Reset
        );
        assert_eq!(controller.status().phase, "idle");
        assert_eq!(
            stored_run(&store, &second.job_id).await.status,
            "stopped",
            "сброс не трогает историю"
        );
    }

    #[tokio::test]
    async fn engine_error_and_panic_become_error_phase() {
        for engine in [
            Arc::new(StepEngine {
                steps: 5,
                delay: Duration::from_millis(1),
                fail_at: Some(2),
                panic: false,
            }),
            Arc::new(StepEngine {
                steps: 5,
                delay: Duration::from_millis(1),
                fail_at: None,
                panic: true,
            }),
        ] {
            let (controller, store) = controller(Some(engine));
            let mut events = controller.subscribe();
            let job = controller.start(config(), None).await.unwrap();
            let (kind, _) = wait_terminal(&mut events, &job.job_id).await;
            assert_eq!(kind, EventKind::Error);
            let status = controller.status();
            assert_eq!(status.phase, "error");
            assert!(status.error.is_some());
            let run = stored_run(&store, &job.job_id).await;
            assert_eq!(run.status, "error");
            assert!(run.error_message.is_some());
        }
    }

    #[test]
    fn interrupted_runs_are_marked_on_startup() {
        let store = Arc::new(Store::open_in_memory().unwrap());
        let job = Job::new("train-old".into(), None, config(), watch::channel(false).0);
        let summary = job.summary();
        store
            .with_conn(|conn| store::runs::upsert(conn, &summary))
            .unwrap();
        let _controller = TrainingController::new(Arc::clone(&store), None);
        let run: TrainingRunSummary = store
            .with_conn(|conn| store::runs::get(conn, "train-old"))
            .unwrap()
            .unwrap();
        assert_eq!(run.status, "error");
        assert_eq!(run.error_message.as_deref(), Some(INTERRUPTED_MESSAGE));
    }

    #[test]
    fn sparkline_is_downsampled_evenly() {
        let values: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let points = sparkline(&values);
        assert_eq!(points.len(), SPARKLINE_POINTS);
        assert_eq!(points.first(), Some(&0.0));
        assert_eq!(points.last(), Some(&999.0));
    }
}
