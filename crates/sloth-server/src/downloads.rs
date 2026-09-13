//! Загрузка моделей с Hugging Face (`/api/hub/download*`, `/api/models/download*`).
//!
//! Раньше все загрузки делили один общий слот `job-default`: вторая загрузка перетирала
//! первую, прогресс отдавался в процентах вместо доли, ошибка записи на диск молча
//! пропускалась, итоговый размер угадывался по названию репозитория, а общий таймаут
//! в 2 часа обрывал большие модели без возможности докачки. Здесь:
//! - у каждой пары «репозиторий + вариант» своё задание со своей отменой;
//! - список файлов и точные размеры берутся из дерева файлов Hugging Face;
//! - недокачанный файл хранится как `.part` и докачивается через HTTP Range;
//! - файлы кладутся в `models/<org>/<repo>/<путь в репозитории>`, одинаковые имена
//!   шардов разных моделей больше не затирают друг друга;
//! - любая ошибка сети или диска переводит задание в состояние `error` с понятным текстом.

use crate::error::{ApiError, ApiResult};
use crate::model_inventory::{self, LocalModel};
use crate::state::AppState;
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::Json;
use futures_util::StreamExt;
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::sync::watch;

/// Адрес Hugging Face по умолчанию; переопределяется `HF_ENDPOINT`, как в huggingface_hub.
pub const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
pub const HF_ENDPOINT_ENV: &str = "HF_ENDPOINT";
/// Заголовок, в котором фронтенд передаёт токен Hugging Face.
pub const HF_TOKEN_HEADER: &str = "x-unsloth-hf-token";

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Сколько можно ждать очередной порции данных. Общего ограничения на длительность
/// загрузки нет: большая модель на медленном канале качается столько, сколько нужно.
const READ_TIMEOUT: Duration = Duration::from_secs(120);
const TREE_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_REDIRECTS: usize = 10;
const USER_AGENT: &str = concat!("sloth-forge/", env!("CARGO_PKG_VERSION"));
/// Пока не подтверждено, что все файлы на месте, прогресс не показывается как 100 %.
const MAX_PROGRESS_BEFORE_COMPLETE: f64 = 0.99;
const PART_SUFFIX: &str = ".part";
const TRANSPORT_HTTP: &str = "http";
/// Глубина поиска `.part`-файлов внутри папки репозитория.
const MAX_PARTIAL_SCAN_DEPTH: usize = 3;

// ---------- реестр заданий ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    Running,
    Cancelling,
    Complete,
    Error,
    Cancelled,
}

impl JobState {
    /// Названия состояний, как их понимает фронтенд (`DownloadJobState`).
    pub fn as_str(self) -> &'static str {
        match self {
            JobState::Running => "running",
            JobState::Cancelling => "cancelling",
            JobState::Complete => "complete",
            JobState::Error => "error",
            JobState::Cancelled => "cancelled",
        }
    }

    fn is_active(self) -> bool {
        matches!(self, JobState::Running | JobState::Cancelling)
    }
}

/// Файл в репозитории Hugging Face.
#[derive(Debug, Clone)]
pub struct RemoteFile {
    pub path: String,
    pub size: Option<u64>,
}

#[derive(Debug)]
struct Job {
    repo_id: String,
    variant: String,
    generation: u64,
    state: JobState,
    error: Option<String>,
    files: Vec<RemoteFile>,
    /// Всего получено байт (включая текущий файл).
    downloaded_bytes: u64,
    /// Байт в полностью сохранённых файлах.
    completed_bytes: u64,
    total_bytes: u64,
    cancel: watch::Sender<bool>,
}

/// Снимок задания для ответа API (без блокировки реестра на время ответа).
#[derive(Debug, Clone)]
struct JobSnapshot {
    repo_id: String,
    variant: String,
    generation: u64,
    state: JobState,
    error: Option<String>,
    files: Vec<String>,
    downloaded_bytes: u64,
    completed_bytes: u64,
    total_bytes: u64,
}

/// Все задания загрузки. Мьютекс из std: он никогда не удерживается через `.await`.
#[derive(Default)]
pub struct DownloadRegistry {
    jobs: Mutex<HashMap<String, Job>>,
    next_generation: AtomicU64,
}

impl DownloadRegistry {
    fn lock(&self) -> MutexGuard<'_, HashMap<String, Job>> {
        // Внутри только счётчики прогресса: после паники в другом потоке их можно читать дальше
        self.jobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Меняет задание, только если это всё ещё то же поколение: у заменённого задания
    /// старая фоновая задача не должна портить состояние нового.
    fn update(&self, key: &str, generation: u64, change: impl FnOnce(&mut Job)) {
        if let Some(job) = self.lock().get_mut(key) {
            if job.generation == generation {
                change(job);
            }
        }
    }

    fn snapshot(&self, key: &str) -> Option<JobSnapshot> {
        self.lock().get(key).map(snapshot_of)
    }

    fn active_snapshots(&self) -> Vec<JobSnapshot> {
        self.lock()
            .values()
            .filter(|job| job.state.is_active())
            .map(snapshot_of)
            .collect()
    }
}

fn snapshot_of(job: &Job) -> JobSnapshot {
    JobSnapshot {
        repo_id: job.repo_id.clone(),
        variant: job.variant.clone(),
        generation: job.generation,
        state: job.state,
        error: job.error.clone(),
        files: job.files.iter().map(|file| file.path.clone()).collect(),
        downloaded_bytes: job.downloaded_bytes,
        completed_bytes: job.completed_bytes,
        total_bytes: job.total_bytes,
    }
}

/// Ключ задания: репозиторий и вариант без учёта регистра.
pub fn job_key(repo_id: &str, variant: Option<&str>) -> String {
    format!(
        "{}::{}",
        repo_id.trim().to_ascii_lowercase(),
        variant.unwrap_or("").trim().to_ascii_lowercase()
    )
}

// ---------- проверки и пути ----------

/// Идентификатор репозитория вида `org/repo` из безопасных символов.
fn validate_repo_id(repo_id: &str) -> ApiResult<&str> {
    let repo_id = repo_id.trim();
    let parts: Vec<&str> = repo_id.split('/').collect();
    let valid = parts.len() == 2
        && parts.iter().all(|part| {
            !part.is_empty()
                && !part.starts_with('.')
                && part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        });
    if valid {
        Ok(repo_id)
    } else {
        Err(ApiError::bad_request(format!(
            "Некорректный идентификатор репозитория: «{repo_id}» (ожидается org/repo)"
        )))
    }
}

/// Путь файла внутри репозитория без `..`, абсолютных частей и скрытых компонентов.
fn safe_relative_path(path: &str) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for component in Path::new(path).components() {
        match component {
            Component::Normal(part) if !part.to_string_lossy().starts_with('.') => out.push(part),
            _ => return None,
        }
    }
    (!out.as_os_str().is_empty()).then_some(out)
}

/// Папка репозитория внутри папки моделей; `None` для некорректного идентификатора.
fn repo_dir(state: &AppState, repo_id: &str) -> Option<PathBuf> {
    validate_repo_id(repo_id)
        .ok()
        .map(|repo| state.models_dir.join(repo))
}

fn part_path(target: &Path) -> PathBuf {
    let mut name = target
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_default();
    name.push(PART_SUFFIX);
    target.with_file_name(name)
}

async fn resolve_token(state: &AppState, headers: &HeaderMap) -> ApiResult<Option<String>> {
    let from_header = headers
        .get(HF_TOKEN_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_string);
    match from_header {
        Some(token) => Ok(Some(token)),
        None => crate::settings::runtime::stored_hf_token(state).await,
    }
}

// ---------- Hugging Face ----------

fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS))
        .user_agent(USER_AGENT)
        .build()
        .map_err(|err| format!("Не удалось создать HTTP-клиент: {err}"))
}

fn authorized(request: reqwest::RequestBuilder, token: Option<&str>) -> reqwest::RequestBuilder {
    match token {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

const ACCESS_DENIED_HINT: &str =
    "Нет доступа к репозиторию: для закрытых моделей укажите токен Hugging Face в настройках";

#[derive(Debug, Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    size: Option<u64>,
    lfs: Option<TreeLfs>,
}

#[derive(Debug, Deserialize)]
struct TreeLfs {
    size: Option<u64>,
}

/// GGUF-файлы репозитория с точными размерами (для LFS-файлов — размер из `lfs.size`).
pub async fn fetch_repo_gguf_files(
    client: &reqwest::Client,
    endpoint: &str,
    repo_id: &str,
    token: Option<&str>,
) -> Result<Vec<RemoteFile>, String> {
    let url = format!("{endpoint}/api/models/{repo_id}/tree/main?recursive=true");
    let response = authorized(client.get(&url).timeout(TREE_REQUEST_TIMEOUT), token)
        .send()
        .await
        .map_err(|err| format!("Hugging Face недоступен: {err}"))?;
    match response.status() {
        status if status.is_success() => {}
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
            return Err(ACCESS_DENIED_HINT.to_string())
        }
        StatusCode::NOT_FOUND => {
            return Err(format!("Репозиторий {repo_id} не найден на Hugging Face"))
        }
        status => {
            return Err(format!(
                "Hugging Face ответил HTTP {status} на список файлов"
            ))
        }
    }
    let entries: Vec<TreeEntry> = response
        .json()
        .await
        .map_err(|err| format!("Не удалось разобрать список файлов репозитория: {err}"))?;
    let mut files: Vec<RemoteFile> = entries
        .into_iter()
        .filter(|entry| entry.kind == "file" && entry.path.to_ascii_lowercase().ends_with(".gguf"))
        .map(|entry| RemoteFile {
            size: entry.lfs.and_then(|lfs| lfs.size).or(entry.size),
            path: entry.path,
        })
        .collect();
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(files)
}

/// Файлы нужного варианта (без проектора, MTP-голов и прочих вспомогательных файлов).
fn variant_files(
    files: &[RemoteFile],
    variant: Option<&str>,
) -> Result<(String, Vec<RemoteFile>), String> {
    let model_files: Vec<&RemoteFile> = files
        .iter()
        .filter(|file| !crate::hub::is_auxiliary_gguf_file(&file.path))
        .collect();
    let variant = match variant.map(str::trim).filter(|v| !v.is_empty()) {
        Some(variant) => variant.to_string(),
        None => {
            let quants: BTreeSet<String> = model_files
                .iter()
                .map(|file| crate::hub::extract_quant_from_path(&file.path))
                .collect();
            if quants.len() != 1 {
                return Err(
                    "В репозитории несколько вариантов модели: укажите gguf_variant".to_string(),
                );
            }
            quants.into_iter().next().unwrap_or_default()
        }
    };
    let selected: Vec<RemoteFile> = model_files
        .into_iter()
        .filter(|file| {
            crate::hub::extract_quant_from_path(&file.path).eq_ignore_ascii_case(&variant)
        })
        .cloned()
        .collect();
    if selected.is_empty() {
        Err(format!("Вариант {variant} не найден в репозитории"))
    } else {
        Ok((variant, selected))
    }
}

/// Все файлы уже лежат в папке репозитория с правильными размерами.
async fn files_complete(dir: &Path, files: &[RemoteFile]) -> bool {
    if files.is_empty() {
        return false;
    }
    for file in files {
        let Some(relative) = safe_relative_path(&file.path) else {
            return false;
        };
        match tokio::fs::metadata(dir.join(relative)).await {
            Ok(meta) if meta.is_file() && file.size.is_none_or(|size| meta.len() == size) => {}
            _ => return false,
        }
    }
    true
}

// ---------- фоновая загрузка ----------

enum JobEnd {
    Cancelled,
    Failed(String),
}

struct JobContext {
    state: Arc<AppState>,
    key: String,
    generation: u64,
    repo_id: String,
    dir: PathBuf,
    token: Option<String>,
}

impl JobContext {
    fn update(&self, change: impl FnOnce(&mut Job)) {
        self.state
            .downloads
            .update(&self.key, self.generation, change);
    }
}

async fn run_job(context: JobContext, files: Vec<RemoteFile>, mut cancel: watch::Receiver<bool>) {
    match download_files(&context, &files, &mut cancel).await {
        Ok(()) => {
            context.update(|job| {
                job.state = JobState::Complete;
                job.downloaded_bytes = job.total_bytes.max(job.completed_bytes);
                job.completed_bytes = job.downloaded_bytes;
            });
            // Признак «скачано» у вариантов в кэше больше не соответствует диску
            context.state.gguf_variants_cache.write().await.clear();
            tracing::info!("Загрузка {} завершена", context.repo_id);
        }
        Err(JobEnd::Cancelled) => {
            context.update(|job| job.state = JobState::Cancelled);
            tracing::info!(
                "Загрузка {} отменена, недокачанные файлы сохранены",
                context.repo_id
            );
        }
        Err(JobEnd::Failed(message)) => {
            tracing::warn!("Загрузка {} не удалась: {message}", context.repo_id);
            context.update(|job| {
                job.state = JobState::Error;
                job.error = Some(message);
            });
        }
    }
}

async fn download_files(
    context: &JobContext,
    files: &[RemoteFile],
    cancel: &mut watch::Receiver<bool>,
) -> Result<(), JobEnd> {
    let client = http_client().map_err(JobEnd::Failed)?;
    let mut completed: u64 = 0;
    for file in files {
        if *cancel.borrow() {
            return Err(JobEnd::Cancelled);
        }
        let relative = safe_relative_path(&file.path).ok_or_else(|| {
            JobEnd::Failed(format!(
                "Недопустимый путь файла в репозитории: {}",
                file.path
            ))
        })?;
        let target = context.dir.join(relative);

        // Файл уже скачан целиком — пропускаем
        if let Ok(meta) = tokio::fs::metadata(&target).await {
            if meta.is_file() && file.size.is_none_or(|size| meta.len() == size) {
                completed += meta.len();
                context.update(|job| {
                    job.completed_bytes = completed;
                    job.downloaded_bytes = completed;
                });
                continue;
            }
        }
        if let Some(parent) = target.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|err| {
                JobEnd::Failed(format!(
                    "Не удалось создать папку {}: {err}",
                    parent.display()
                ))
            })?;
        }

        let written = download_one(context, &client, file, &target, completed, cancel).await?;
        completed += written;
        context.update(|job| {
            job.completed_bytes = completed;
            job.downloaded_bytes = completed;
        });
    }
    Ok(())
}

/// Скачивает один файл в `.part` (продолжая с того места, где остановились) и
/// переименовывает его, только когда размер совпал с ожидаемым.
async fn download_one(
    context: &JobContext,
    client: &reqwest::Client,
    file: &RemoteFile,
    target: &Path,
    completed_before: u64,
    cancel: &mut watch::Receiver<bool>,
) -> Result<u64, JobEnd> {
    let part = part_path(target);
    let mut offset = tokio::fs::metadata(&part)
        .await
        .map(|meta| meta.len())
        .unwrap_or(0);
    if file.size.is_some_and(|size| offset > size) {
        // Остаток от другой версии файла: докачивать нечего, начинаем заново
        tokio::fs::remove_file(&part).await.map_err(|err| {
            JobEnd::Failed(format!(
                "Не удалось удалить устаревший {}: {err}",
                part.display()
            ))
        })?;
        offset = 0;
    }

    let url = format!(
        "{}/{}/resolve/main/{}",
        context.state.hf_endpoint, context.repo_id, file.path
    );
    let mut request = authorized(client.get(&url), context.token.as_deref());
    if offset > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
    }
    let response = request
        .send()
        .await
        .map_err(|err| JobEnd::Failed(format!("Ошибка сети при загрузке {}: {err}", file.path)))?;

    let append = match response.status() {
        StatusCode::PARTIAL_CONTENT => true,
        status if status.is_success() => {
            // Сервер не поддержал докачку и прислал файл целиком
            offset = 0;
            false
        }
        StatusCode::RANGE_NOT_SATISFIABLE if file.size == Some(offset) => {
            // .part уже содержит весь файл
            return finalize_part(&part, target, offset, file).await;
        }
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
            return Err(JobEnd::Failed(ACCESS_DENIED_HINT.to_string()))
        }
        status => {
            return Err(JobEnd::Failed(format!(
                "Hugging Face ответил HTTP {status} на файл {}",
                file.path
            )))
        }
    };

    let mut output = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(&part)
        .await
        .map_err(|err| JobEnd::Failed(format!("Не удалось открыть {}: {err}", part.display())))?;

    let mut written = offset;
    context.update(|job| job.downloaded_bytes = completed_before + written);
    let mut stream = response.bytes_stream();
    loop {
        tokio::select! {
            changed = cancel.changed() => {
                // Отмена или замена задания: сохраняем полученное и выходим
                if changed.is_err() || *cancel.borrow() {
                    if let Err(err) = output.flush().await {
                        tracing::warn!("Не удалось дописать {} при отмене: {err}", part.display());
                    }
                    return Err(JobEnd::Cancelled);
                }
            }
            chunk = stream.next() => match chunk {
                None => break,
                Some(Err(err)) => {
                    return Err(JobEnd::Failed(format!(
                        "Соединение прервалось при загрузке {}: {err}. Повторите загрузку — она продолжится с того же места",
                        file.path
                    )));
                }
                Some(Ok(bytes)) => {
                    output.write_all(&bytes).await.map_err(|err| {
                        JobEnd::Failed(format!("Не удалось записать {} на диск: {err}", part.display()))
                    })?;
                    written += bytes.len() as u64;
                    let current = completed_before + written;
                    context.update(|job| job.downloaded_bytes = current);
                }
            }
        }
    }

    output
        .flush()
        .await
        .map_err(|err| JobEnd::Failed(format!("Не удалось дописать {}: {err}", part.display())))?;
    output.sync_all().await.map_err(|err| {
        JobEnd::Failed(format!(
            "Не удалось сохранить {} на диск: {err}",
            part.display()
        ))
    })?;
    drop(output);
    finalize_part(&part, target, written, file).await
}

async fn finalize_part(
    part: &Path,
    target: &Path,
    written: u64,
    file: &RemoteFile,
) -> Result<u64, JobEnd> {
    if let Some(size) = file.size {
        if written != size {
            return Err(JobEnd::Failed(format!(
                "Файл {} скачан не полностью: {written} из {size} байт. Повторите загрузку — она продолжится",
                file.path
            )));
        }
    }
    tokio::fs::rename(part, target).await.map_err(|err| {
        JobEnd::Failed(format!("Не удалось сохранить {}: {err}", target.display()))
    })?;
    Ok(written)
}

// ---------- диск ----------

async fn scan_models(state: &AppState) -> ApiResult<Vec<LocalModel>> {
    let roots = state.model_roots().await;
    tokio::task::spawn_blocking(move || model_inventory::scan_models(&roots))
        .await
        .map_err(|err| ApiError::internal(format!("Поиск моделей прерван: {err}")))
}

/// Полностью скачанная модель этого репозитория (и варианта, если он указан).
async fn local_variant(
    state: &AppState,
    repo_id: &str,
    variant: Option<&str>,
) -> ApiResult<Option<LocalModel>> {
    let variant = variant.map(str::trim).filter(|v| !v.is_empty());
    Ok(scan_models(state).await?.into_iter().find(|model| {
        model.complete
            && model.repo_id().eq_ignore_ascii_case(repo_id.trim())
            && variant.is_none_or(|v| model.quant().eq_ignore_ascii_case(v))
    }))
}

/// Суммарный размер `.part`-файлов варианта в папке репозитория.
fn partial_bytes(dir: &Path, variant: Option<&str>) -> u64 {
    fn walk(dir: &Path, base: &Path, depth: usize, variant: Option<&str>, total: &mut u64) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            if file_type.is_dir() && depth < MAX_PARTIAL_SCAN_DEPTH {
                walk(&path, base, depth + 1, variant, total);
            } else if file_type.is_file() {
                let relative = path
                    .strip_prefix(base)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .into_owned();
                let Some(model_path) = relative.strip_suffix(PART_SUFFIX) else {
                    continue;
                };
                let matches = variant.is_none_or(|v| {
                    crate::hub::extract_quant_from_path(model_path).eq_ignore_ascii_case(v)
                });
                if matches {
                    *total += entry.metadata().map(|meta| meta.len()).unwrap_or(0);
                }
            }
        }
    }
    let mut total = 0;
    walk(dir, dir, 0, variant, &mut total);
    total
}

async fn partial_bytes_for(
    state: &AppState,
    repo_id: &str,
    variant: Option<&str>,
) -> ApiResult<u64> {
    let Some(dir) = repo_dir(state, repo_id) else {
        return Ok(0);
    };
    let variant = variant.map(str::to_string);
    tokio::task::spawn_blocking(move || partial_bytes(&dir, variant.as_deref()))
        .await
        .map_err(|err| ApiError::internal(format!("Проверка недокачанных файлов прервана: {err}")))
}

// ---------- обработчики ----------

#[derive(Debug, Default, Deserialize)]
pub struct StartRequest {
    #[serde(default)]
    repo_id: Option<String>,
    #[serde(default)]
    gguf_variant: Option<String>,
    #[serde(default)]
    hf_token: Option<String>,
    #[serde(default)]
    files: Option<Vec<String>>,
}

fn start_response(
    key: &str,
    state: JobState,
    generation: Option<u64>,
    attached: bool,
) -> Json<Value> {
    Json(json!({
        "state": state.as_str(),
        "accepted": true,
        "generation": generation,
        "attached": attached,
        "transport": TRANSPORT_HTTP,
        "cancel_transport": TRANSPORT_HTTP,
        "job_key": key
    }))
}

/// `POST /api/hub/download`: запускает (или продолжает) загрузку варианта модели.
pub async fn start_download(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<StartRequest>,
) -> ApiResult<Json<Value>> {
    let repo_id = validate_repo_id(request.repo_id.as_deref().unwrap_or(""))?.to_string();
    let token = match resolve_token(&state, &headers).await? {
        Some(token) => Some(token),
        None => request
            .hf_token
            .clone()
            .filter(|token| !token.trim().is_empty()),
    };

    let client = http_client().map_err(ApiError::internal)?;
    let remote = fetch_repo_gguf_files(&client, &state.hf_endpoint, &repo_id, token.as_deref())
        .await
        .map_err(|detail| ApiError::new(axum::http::StatusCode::BAD_GATEWAY, detail))?;
    let (variant, mut files) =
        variant_files(&remote, request.gguf_variant.as_deref()).map_err(ApiError::unprocessable)?;
    if let Some(requested) = request
        .files
        .as_ref()
        .filter(|requested| !requested.is_empty())
    {
        files.retain(|file| requested.iter().any(|path| path == &file.path));
        if files.is_empty() {
            return Err(ApiError::unprocessable(
                "Запрошенные файлы не относятся к выбранному варианту",
            ));
        }
    }

    let key = job_key(&repo_id, Some(&variant));
    let dir = state.models_dir.join(&repo_id);
    if files_complete(&dir, &files).await {
        return Ok(start_response(&key, JobState::Complete, None, false));
    }

    let (cancel_tx, cancel_rx) = watch::channel(false);
    let generation = {
        let mut jobs = state.downloads.lock();
        if let Some(job) = jobs.get(&key) {
            if job.state.is_active() {
                // Такая загрузка уже идёт — подключаемся к ней, а не запускаем вторую
                return Ok(start_response(&key, job.state, Some(job.generation), true));
            }
        }
        let generation = state
            .downloads
            .next_generation
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        jobs.insert(
            key.clone(),
            Job {
                repo_id: repo_id.clone(),
                variant: variant.clone(),
                generation,
                state: JobState::Running,
                error: None,
                total_bytes: files.iter().filter_map(|file| file.size).sum(),
                files: files.clone(),
                downloaded_bytes: 0,
                completed_bytes: 0,
                cancel: cancel_tx,
            },
        );
        generation
    };

    let context = JobContext {
        state: Arc::clone(&state),
        key: key.clone(),
        generation,
        repo_id,
        dir,
        token,
    };
    tokio::spawn(run_job(context, files, cancel_rx));
    Ok(start_response(
        &key,
        JobState::Running,
        Some(generation),
        false,
    ))
}

#[derive(Debug, Default, Deserialize)]
pub struct CancelRequest {
    #[serde(default)]
    repo_id: Option<String>,
    #[serde(default)]
    gguf_variant: Option<String>,
    #[serde(default)]
    generation: Option<u64>,
}

/// `POST /api/hub/download/cancel`: отменяет именно это задание (недокачанное сохраняется).
pub async fn cancel_download(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CancelRequest>,
) -> Json<Value> {
    let key = job_key(
        request.repo_id.as_deref().unwrap_or(""),
        request.gguf_variant.as_deref(),
    );
    let mut jobs = state.downloads.lock();
    let Some(job) = jobs.get_mut(&key) else {
        return Json(json!({ "job_key": key, "state": "idle" }));
    };
    let stale_generation = request
        .generation
        .is_some_and(|generation| generation != job.generation);
    if job.state.is_active() && !stale_generation {
        job.state = JobState::Cancelling;
        if job.cancel.send(true).is_err() {
            // Фоновая задача уже завершилась
            job.state = JobState::Cancelled;
        }
    }
    Json(json!({ "job_key": key, "state": job.state.as_str() }))
}

#[derive(Debug, Default, Deserialize)]
pub struct ProgressQuery {
    #[serde(default)]
    repo_id: Option<String>,
    #[serde(default)]
    gguf_variant: Option<String>,
    #[serde(default)]
    variant: Option<String>,
    #[serde(default)]
    expected_bytes: Option<u64>,
}

impl ProgressQuery {
    fn repo(&self) -> &str {
        self.repo_id.as_deref().unwrap_or("").trim()
    }

    fn variant(&self) -> Option<&str> {
        self.gguf_variant
            .as_deref()
            .or(self.variant.as_deref())
            .map(str::trim)
            .filter(|v| !v.is_empty())
    }
}

fn fraction(downloaded: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        (downloaded as f64 / total as f64).min(MAX_PROGRESS_BEFORE_COMPLETE)
    }
}

/// `GET /api/hub/download-status`: состояние задания этого варианта.
pub async fn download_status(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ProgressQuery>,
) -> ApiResult<Json<Value>> {
    if let Some(job) = state
        .downloads
        .snapshot(&job_key(query.repo(), query.variant()))
    {
        return Ok(Json(json!({
            "state": job.state.as_str(),
            "error": job.error,
            "generation": job.generation,
            "downloaded_bytes": job.downloaded_bytes,
            "total_bytes": job.total_bytes,
            "progress": if job.state == JobState::Complete { 1.0 } else { fraction(job.downloaded_bytes, job.total_bytes) }
        })));
    }
    let present = local_variant(&state, query.repo(), query.variant())
        .await?
        .is_some();
    Ok(Json(json!({
        "state": if present { "complete" } else { "idle" },
        "error": null,
        "generation": null
    })))
}

fn progress_json(
    downloaded: u64,
    completed: u64,
    expected: u64,
    complete_on_disk: bool,
    cache_path: Option<String>,
) -> Json<Value> {
    Json(json!({
        "downloaded_bytes": downloaded,
        "completed_bytes": completed,
        "complete_on_disk": complete_on_disk,
        "expected_bytes": expected,
        "progress": if complete_on_disk { 1.0 } else { fraction(downloaded, expected) },
        "target_present": complete_on_disk || downloaded > 0,
        "cache_path": cache_path,
        "cache_measured": true
    }))
}

async fn progress_for(
    state: &AppState,
    query: &ProgressQuery,
    variant: Option<&str>,
) -> ApiResult<Json<Value>> {
    let repo = query.repo();
    let expected_hint = query.expected_bytes.unwrap_or(0);
    let dir_string = repo_dir(state, repo).map(|dir| dir.to_string_lossy().into_owned());

    // Идущие задания этого репозитория (и варианта, если он указан)
    let active: Vec<JobSnapshot> = state
        .downloads
        .active_snapshots()
        .into_iter()
        .filter(|job| {
            job.repo_id.eq_ignore_ascii_case(repo)
                && variant.is_none_or(|v| job.variant.eq_ignore_ascii_case(v))
        })
        .collect();
    if !active.is_empty() {
        let downloaded = active.iter().map(|job| job.downloaded_bytes).sum();
        let completed = active.iter().map(|job| job.completed_bytes).sum();
        let total: u64 = active.iter().map(|job| job.total_bytes).sum();
        let expected = if total > 0 { total } else { expected_hint };
        return Ok(progress_json(
            downloaded, completed, expected, false, dir_string,
        ));
    }

    if let Some(model) = local_variant(state, repo, variant).await? {
        return Ok(progress_json(
            model.size_bytes,
            model.size_bytes,
            model.size_bytes,
            true,
            Some(model.id()),
        ));
    }
    let partial = partial_bytes_for(state, repo, variant).await?;
    Ok(progress_json(
        partial,
        0,
        expected_hint,
        false,
        (partial > 0).then_some(dir_string).flatten(),
    ))
}

/// `GET /api/hub/gguf-download-progress`: прогресс варианта (доля 0..1).
pub async fn gguf_download_progress(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ProgressQuery>,
) -> ApiResult<Json<Value>> {
    let variant = query.variant().map(str::to_string);
    progress_for(&state, &query, variant.as_deref()).await
}

/// `GET /api/hub/download-progress`: прогресс по репозиторию целиком.
pub async fn download_progress(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ProgressQuery>,
) -> ApiResult<Json<Value>> {
    progress_for(&state, &query, None).await
}

/// `GET /api/hub/active-downloads`: идущие загрузки (всех или одного репозитория).
pub async fn active_downloads(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ProgressQuery>,
) -> Json<Value> {
    let repo = query.repo();
    let downloads: Vec<Value> = state
        .downloads
        .active_snapshots()
        .into_iter()
        .filter(|job| repo.is_empty() || job.repo_id.eq_ignore_ascii_case(repo))
        .map(|job| {
            json!({
                "repo_id": job.repo_id,
                "variant": job.variant,
                "transport": TRANSPORT_HTTP,
                "cancel_transport": TRANSPORT_HTTP,
                "state": job.state.as_str(),
                "generation": job.generation,
                "files": job.files
            })
        })
        .collect();
    Json(json!({ "downloads": downloads }))
}

/// `GET /api/hub/transport-status`: есть ли недокачанные файлы, которые можно продолжить.
pub async fn transport_status(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ProgressQuery>,
) -> ApiResult<Json<Value>> {
    let has_partial = partial_bytes_for(&state, query.repo(), query.variant()).await? > 0;
    Ok(Json(json!({
        "has_partial": has_partial,
        "last_transport": if has_partial { Some(TRANSPORT_HTTP) } else { None },
        "resumable": true
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_repo_ids() {
        assert!(validate_repo_id("unsloth/Llama-3.2-1B-Instruct-GGUF").is_ok());
        assert!(validate_repo_id("../etc").is_err());
        assert!(validate_repo_id("a/b/c").is_err());
        assert!(validate_repo_id("org/.hidden").is_err());
        assert!(validate_repo_id("org/repo name").is_err());
    }

    #[test]
    fn rejects_unsafe_file_paths() {
        assert_eq!(
            safe_relative_path("UD-Q4_K_XL/model-00001-of-00002.gguf"),
            Some(PathBuf::from("UD-Q4_K_XL/model-00001-of-00002.gguf"))
        );
        assert_eq!(safe_relative_path("../escape.gguf"), None);
        assert_eq!(safe_relative_path("/abs.gguf"), None);
        assert_eq!(safe_relative_path(".hidden/x.gguf"), None);
    }

    #[test]
    fn selects_variant_files_and_skips_auxiliary() {
        let files = vec![
            RemoteFile {
                path: "Q4_K_M/m-00001-of-00002.gguf".into(),
                size: Some(10),
            },
            RemoteFile {
                path: "Q4_K_M/m-00002-of-00002.gguf".into(),
                size: Some(5),
            },
            RemoteFile {
                path: "m-Q8_0.gguf".into(),
                size: Some(20),
            },
            RemoteFile {
                path: "mmproj-F16.gguf".into(),
                size: Some(3),
            },
        ];
        let (variant, selected) = variant_files(&files, Some("q4_k_m")).unwrap();
        assert_eq!(variant, "q4_k_m");
        assert_eq!(selected.len(), 2);
        assert!(
            variant_files(&files, None).is_err(),
            "несколько вариантов требуют выбора"
        );
        assert!(variant_files(&files, Some("Q2_K")).is_err());
    }

    #[test]
    fn progress_fraction_is_capped_until_complete() {
        assert_eq!(fraction(50, 100), 0.5);
        assert_eq!(fraction(100, 100), MAX_PROGRESS_BEFORE_COMPLETE);
        assert_eq!(fraction(10, 0), 0.0);
    }
}
