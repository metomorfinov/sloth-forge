use serde::{Deserialize, Serialize};
use sloth_core::cluster::{ClusterCoordinator, NodeRole};
use sloth_vulkan_sys::VulkanContext;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

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
pub struct ScanFolderEntry {
    pub id: u64,
    pub path: String,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

pub struct AppState {
    pub vk_ctx: Option<VulkanContext>,
    pub coordinator: Arc<ClusterCoordinator>,
    /// Запуски обучения: жизненный цикл, события прогресса, история в SQLite.
    pub training: Arc<crate::training::TrainingController>,
    pub models_dir: PathBuf,
    pub static_dir: Option<PathBuf>,
    /// Порт, на котором реально слушает сервер (показывается в настройках доступа по сети).
    pub server_port: std::sync::atomic::AtomicU16,
    /// Сигнал мягкой остановки сервера (кнопка «Остановить» в интерфейсе, POST /api/shutdown).
    pub shutdown: tokio::sync::Notify,
    /// Градиенты мастера текущего шага (выставляет движок обучения; до этапа 3 их нет).
    pub master_gradients: Arc<RwLock<Option<sloth_core::cluster::StepGradients>>>,
    /// Задания загрузки моделей: у каждой пары «репозиторий + вариант» своё.
    pub downloads: crate::downloads::DownloadRegistry,
    /// Адрес Hugging Face (`HF_ENDPOINT`); тесты подменяют его локальным сервером.
    pub hf_endpoint: String,
    pub scan_folders: Arc<RwLock<Vec<ScanFolderEntry>>>,
    /// Списки файлов репозиториев Hugging Face (живут несколько минут).
    pub hf_tree_cache: crate::hf_api::TreeCache,
    pub api_keys: Arc<RwLock<Vec<serde_json::Value>>>,
    /// Постоянное хранилище: история чатов, проекты, настройки, запуски обучения.
    pub store: Arc<crate::store::Store>,
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
        let master_gradients = Arc::new(RwLock::new(None));

        Self {
            vk_ctx,
            coordinator,
            training: Arc::new(crate::training::TrainingController::new(Arc::clone(&store), None)),
            models_dir,
            static_dir,
            server_port: std::sync::atomic::AtomicU16::new(crate::DEFAULT_PORT),
            shutdown: tokio::sync::Notify::new(),
            master_gradients,
            downloads: crate::downloads::DownloadRegistry::default(),
            hf_endpoint: crate::hf_api::endpoint_from_env(),
            scan_folders: Arc::new(RwLock::new(crate::scan_folders::load(&store))),
            hf_tree_cache: Default::default(),
            api_keys: Arc::new(RwLock::new(Vec::new())),
            store,
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
