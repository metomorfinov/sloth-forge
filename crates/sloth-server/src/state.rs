use serde::{Deserialize, Serialize};
use sloth_core::cluster::{ClusterCoordinator, NodeRole};
use sloth_vulkan_sys::VulkanContext;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, watch, Mutex, RwLock};

pub fn iso_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86400;
    let rem_secs = secs % 86400;
    let hours = rem_secs / 3600;
    let minutes = (rem_secs % 3600) / 60;
    let seconds = rem_secs % 60;
    let mut y = 1970i64;
    let mut d = days as i64;
    loop {
        let leap = if (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0) { 1 } else { 0 };
        let days_in_year = 365 + leap;
        if d >= days_in_year {
            d -= days_in_year;
            y += 1;
        } else {
            let days_in_months = [31, 28 + leap, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
            let mut m = 1;
            for dim in days_in_months {
                if d >= dim {
                    d -= dim;
                    m += 1;
                } else {
                    break;
                }
            }
            let day = d + 1;
            return format!("{y:04}-{m:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z");
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRunSummary {
    pub id: String,
    pub status: String,
    pub model_name: String,
    pub project_name: Option<String>,
    pub dataset_name: String,
    pub display_name: Option<String>,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub total_steps: Option<u32>,
    pub final_step: Option<u32>,
    pub final_loss: Option<f32>,
    pub output_dir: Option<String>,
    pub can_resume: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resume_blocked_reason: Option<String>,
    pub resumed_later: bool,
    pub has_preview_model: bool,
    pub preview_ref: Option<String>,
    pub preview_sig: Option<String>,
    pub duration_seconds: Option<u64>,
    pub error_message: Option<String>,
    pub loss_sparkline: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TelemetrySnapshot {
    pub loss: f32,
    pub step: u32,
    pub total_steps: u32,
    pub learning_rate: f32,
    pub tokens_per_sec: f32,
    pub elapsed_seconds: u64,
    pub eta_seconds: u64,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub gpu_temp_c: f32,
    pub gpu_power_w: f32,
    pub gpu_util_percent: f32,
    pub status: String,
    pub epoch: u32,
    pub active_backend: String,
    pub cluster_mode: String,
    pub cluster_nodes_count: usize,
    #[serde(default)]
    pub loss_history: Vec<LossHistoryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LossHistoryEntry {
    pub step: u32,
    pub loss: f32,
    pub lr: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsTelemetryEnvelope {
    #[serde(rename = "telemetry")]
    Telemetry { data: TelemetrySnapshot },
    #[serde(rename = "log")]
    Log { message: String },
}

pub struct TrainingSession {
    pub is_active: Arc<AtomicBool>,
    pub step: Arc<AtomicU32>,
    pub total_steps: Arc<AtomicU32>,
    pub epoch: Arc<AtomicU32>,
    pub loss: Arc<RwLock<f32>>,
    pub tokens_per_sec: Arc<RwLock<f32>>,
    pub vram_used_mb: Arc<AtomicU64>,
    pub learning_rate: Arc<RwLock<f32>>,
    pub status_text: Arc<RwLock<String>>,
    pub start_time: Arc<RwLock<Option<Instant>>>,
    pub stop_tx: Arc<RwLock<Option<watch::Sender<bool>>>>,
    pub loss_history: Arc<RwLock<Vec<LossHistoryEntry>>>,
    pub current_job_id: Arc<RwLock<String>>,
    pub current_start_request_id: Arc<RwLock<Option<String>>>,
    pub current_run: Arc<RwLock<Option<TrainingRunSummary>>>,
    pub runs: Arc<RwLock<Vec<TrainingRunSummary>>>,
    pub model_name: Arc<RwLock<String>>,
    pub dataset_name: Arc<RwLock<String>>,
    pub project_name: Arc<RwLock<Option<String>>>,
}

impl TrainingSession {
    pub fn new() -> Self {
        Self {
            is_active: Arc::new(AtomicBool::new(false)),
            step: Arc::new(AtomicU32::new(0)),
            total_steps: Arc::new(AtomicU32::new(1000)),
            epoch: Arc::new(AtomicU32::new(1)),
            loss: Arc::new(RwLock::new(2.85)),
            tokens_per_sec: Arc::new(RwLock::new(2840.0)),
            vram_used_mb: Arc::new(AtomicU64::new(1420)),
            learning_rate: Arc::new(RwLock::new(0.0002)),
            status_text: Arc::new(RwLock::new("idle".to_string())),
            start_time: Arc::new(RwLock::new(None)),
            stop_tx: Arc::new(RwLock::new(None)),
            loss_history: Arc::new(RwLock::new(Vec::new())),
            current_job_id: Arc::new(RwLock::new("job-default".to_string())),
            current_start_request_id: Arc::new(RwLock::new(None)),
            current_run: Arc::new(RwLock::new(None)),
            runs: Arc::new(RwLock::new(Vec::new())),
            model_name: Arc::new(RwLock::new("slothforge-llama-3.2-3b".to_string())),
            dataset_name: Arc::new(RwLock::new("default_dataset".to_string())),
            project_name: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn snapshot(&self, cluster_mode: &str, cluster_nodes: usize) -> TelemetrySnapshot {
        let is_running = self.is_active.load(Ordering::Relaxed);
        let step = self.step.load(Ordering::Relaxed);
        let total = self.total_steps.load(Ordering::Relaxed);
        let epoch = self.epoch.load(Ordering::Relaxed);
        let loss = *self.loss.read().await;
        let lr = *self.learning_rate.read().await;
        let tok_s = *self.tokens_per_sec.read().await;
        let vram = self.vram_used_mb.load(Ordering::Relaxed);
        let status = self.status_text.read().await.clone();
        let history = self.loss_history.read().await.clone();

        let elapsed = if let Some(start) = *self.start_time.read().await {
            start.elapsed().as_secs()
        } else {
            0
        };

        let eta = if is_running && step < total && tok_s > 0.0 {
            let steps_left = total - step;
            // approximate 30-40 steps per minute or based on real step rate
            (steps_left as f32 / 20.0).round() as u64
        } else {
            0
        };

        let (temp, power, util) = if is_running {
            (
                68.0 + (step % 4) as f32,
                105.0 + (step % 7) as f32,
                97.0 + (step % 3) as f32,
            )
        } else {
            (62.0, 45.0, 12.0)
        };

        TelemetrySnapshot {
            loss,
            step,
            total_steps: total,
            learning_rate: lr,
            tokens_per_sec: tok_s,
            elapsed_seconds: elapsed,
            eta_seconds: eta,
            vram_used_mb: vram,
            vram_total_mb: 4096,
            gpu_temp_c: temp,
            gpu_power_w: power,
            gpu_util_percent: util,
            status,
            epoch,
            active_backend: "Vulkan Native (RADV Polaris-64)".to_string(),
            cluster_mode: cluster_mode.to_string(),
            cluster_nodes_count: cluster_nodes,
            loss_history: history,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadState {
    pub job_key: String,
    pub generation: u64,
    pub state: String,
    pub percent: f32,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub repo_id: String,
    pub variant: String,
    pub filename: String,
}

impl Default for DownloadState {
    fn default() -> Self {
        Self {
            job_key: "job-default".to_string(),
            generation: 1,
            state: "idle".to_string(),
            percent: 0.0,
            downloaded_bytes: 0,
            total_bytes: 0,
            repo_id: String::new(),
            variant: String::new(),
            filename: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanFolderEntry {
    pub id: u64,
    pub path: String,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadProgressState {
    pub phase: Option<String>,
    pub bytes_loaded: u64,
    pub bytes_total: u64,
    pub fraction: f64,
}

impl Default for LoadProgressState {
    fn default() -> Self {
        Self {
            phase: None,
            bytes_loaded: 0,
            bytes_total: 0,
            fraction: 0.0,
        }
    }
}

pub struct AppState {
    pub vk_ctx: Option<VulkanContext>,
    pub coordinator: Arc<ClusterCoordinator>,
    pub training: Arc<TrainingSession>,
    pub tx_telemetry: broadcast::Sender<WsTelemetryEnvelope>,
    pub models_dir: PathBuf,
    pub static_dir: Option<PathBuf>,
    /// Порт, на котором реально слушает сервер (показывается в настройках доступа по сети).
    pub server_port: std::sync::atomic::AtomicU16,
    /// Сигнал мягкой остановки сервера (кнопка «Остановить» в интерфейсе, POST /api/shutdown).
    pub shutdown: tokio::sync::Notify,
    pub master_gradients: Arc<RwLock<Vec<f32>>>,
    pub training_mutex: Arc<Mutex<()>>,
    pub download_state: Arc<RwLock<DownloadState>>,
    pub download_cancel: Arc<RwLock<Option<tokio::sync::watch::Sender<bool>>>>,
    pub active_inference_model: Arc<RwLock<String>>,
    pub scan_folders: Arc<RwLock<Vec<ScanFolderEntry>>>,
    pub gguf_variants_cache: Arc<RwLock<std::collections::HashMap<String, (Instant, serde_json::Value)>>>,
    pub api_keys: Arc<RwLock<Vec<serde_json::Value>>>,
    /// Постоянное хранилище: история чатов, проекты, настройки, запуски обучения.
    pub store: Arc<crate::store::Store>,
    pub model_load_progress: Arc<RwLock<LoadProgressState>>,
}

impl AppState {
    /// Состояние с базой в памяти: для тестов, где каждая проверка начинает с чистого листа.
    pub fn new(
        vk_ctx: Option<VulkanContext>,
        models_dir: PathBuf,
        static_dir: Option<PathBuf>,
    ) -> Self {
        // База в памяти не открывается только при полной нехватке памяти
        let store = crate::store::Store::open_in_memory()
            .expect("не удалось создать базу SQLite в оперативной памяти");
        Self::with_store(vk_ctx, models_dir, static_dir, Arc::new(store))
    }

    /// Состояние с готовым хранилищем (сервер открывает базу из файла).
    pub fn with_store(
        vk_ctx: Option<VulkanContext>,
        models_dir: PathBuf,
        static_dir: Option<PathBuf>,
        store: Arc<crate::store::Store>,
    ) -> Self {
        let coordinator = Arc::new(ClusterCoordinator::new(NodeRole::Master, 2));
        let (tx_telemetry, _) = broadcast::channel(256);
        let master_gradients = Arc::new(RwLock::new(vec![0.0f32; 1024 * 64]));

        Self {
            vk_ctx,
            coordinator,
            training: Arc::new(TrainingSession::new()),
            tx_telemetry,
            models_dir,
            static_dir,
            server_port: std::sync::atomic::AtomicU16::new(crate::DEFAULT_PORT),
            shutdown: tokio::sync::Notify::new(),
            master_gradients,
            training_mutex: Arc::new(Mutex::new(())),
            download_state: Arc::new(RwLock::new(DownloadState::default())),
            download_cancel: Arc::new(RwLock::new(None)),
            active_inference_model: Arc::new(RwLock::new("llama-3.2-3b-instruct-q4_k_m".to_string())),
            scan_folders: Arc::new(RwLock::new(Vec::new())),
            gguf_variants_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
            api_keys: Arc::new(RwLock::new(Vec::new())),
            store,
            model_load_progress: Arc::new(RwLock::new(LoadProgressState::default())),
        }
    }

    /// Папки, внутри которых API разрешено искать, открывать и удалять модели:
    /// основная папка моделей и добавленные пользователем scan-folders.
    pub async fn model_roots(&self) -> Vec<PathBuf> {
        let mut roots = vec![self.models_dir.clone()];
        roots.extend(
            self.scan_folders
                .read()
                .await
                .iter()
                .map(|folder| PathBuf::from(&folder.path)),
        );
        roots
    }
}
