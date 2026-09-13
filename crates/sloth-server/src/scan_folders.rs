//! Дополнительные папки с моделями: `/api/hub/scan-folders` и `/api/models/scan-folders`.
//!
//! Раньше список жил только в памяти и пропадал при перезапуске, номер новой папки был
//! «длина списка + 1» (после удаления номера повторялись), путь не проверялся, а статус
//! всегда был «ok». Добавленная папка расширяет песочницу путей — в ней можно открывать
//! и удалять модели, — поэтому принимаются только существующие папки и не корень диска.

use crate::error::{ApiError, ApiResult};
use crate::state::{AppState, ScanFolderEntry};
use crate::store::{Store, StoreError};
use axum::extract::{Path as AxPath, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Ключ настройки, где хранятся папки и счётчик номеров.
const SCAN_FOLDERS_KEY: &str = "hub.scan_folders";

/// Статусы папки, как их понимает фронтенд (`ScanFolderStatus`).
const STATUS_OK: &str = "ok";
const STATUS_MISSING: &str = "missing";
const STATUS_PERMISSION_DENIED: &str = "permission_denied";
const STATUS_UNREADABLE: &str = "unreadable";

#[derive(Debug, Default, Serialize, Deserialize)]
struct StoredFolders {
    /// Последний выданный номер: номера удалённых папок больше не выдаются.
    next_id: u64,
    folders: Vec<ScanFolderEntry>,
}

/// Папки из базы при запуске сервера. Ошибка чтения не мешает запуску: список будет пуст.
pub fn load(store: &Store) -> Vec<ScanFolderEntry> {
    let stored: Result<Option<Value>, StoreError> =
        store.with_conn(|conn| crate::store::settings::get(conn, SCAN_FOLDERS_KEY));
    match stored {
        Ok(Some(value)) => match serde_json::from_value::<StoredFolders>(value) {
            Ok(stored) => stored.folders,
            Err(err) => {
                tracing::error!("Список папок с моделями в базе повреждён: {err}");
                Vec::new()
            }
        },
        Ok(None) => Vec::new(),
        Err(err) => {
            tracing::error!("Не удалось прочитать список папок с моделями: {err}");
            Vec::new()
        }
    }
}

async fn read_stored(state: &AppState) -> ApiResult<StoredFolders> {
    match crate::settings::read_value(state, SCAN_FOLDERS_KEY).await? {
        Some(value) => serde_json::from_value(value).map_err(|err| {
            ApiError::internal(format!("Список папок с моделями в базе повреждён: {err}"))
        }),
        None => Ok(StoredFolders::default()),
    }
}

async fn write_stored(state: &AppState, stored: &StoredFolders) -> ApiResult<()> {
    let value = serde_json::to_value(stored)
        .map_err(|err| ApiError::internal(format!("Не удалось сохранить список папок: {err}")))?;
    crate::settings::write_value(state, SCAN_FOLDERS_KEY, value).await
}

/// Можно ли сейчас прочитать папку. Синхронная: вызывать через `spawn_blocking`.
fn folder_status(path: &Path) -> &'static str {
    match std::fs::read_dir(path) {
        Ok(_) => STATUS_OK,
        Err(err) => match err.kind() {
            ErrorKind::NotFound => STATUS_MISSING,
            ErrorKind::PermissionDenied => STATUS_PERMISSION_DENIED,
            _ => STATUS_UNREADABLE,
        },
    }
}

/// Проверяет, что путь — существующая читаемая папка, и возвращает канонический путь.
/// Синхронная: вызывать через `spawn_blocking`.
fn validate_folder(path: &Path) -> ApiResult<PathBuf> {
    let canonical = path.canonicalize().map_err(|err| match err.kind() {
        ErrorKind::NotFound => {
            ApiError::bad_request(format!("Папка {} не найдена", path.display()))
        }
        ErrorKind::PermissionDenied => {
            ApiError::forbidden(format!("Нет доступа к папке {}", path.display()))
        }
        _ => ApiError::bad_request(format!("Не удалось открыть {}: {err}", path.display())),
    })?;
    if !canonical.is_dir() {
        return Err(ApiError::bad_request(format!(
            "{} — это файл, а не папка",
            canonical.display()
        )));
    }
    if canonical.parent().is_none() {
        return Err(ApiError::bad_request(
            "Корень диска добавить нельзя: укажите папку, в которой лежат модели",
        ));
    }
    std::fs::read_dir(&canonical).map_err(|err| {
        ApiError::forbidden(format!(
            "Папку {} не удалось прочитать: {err}",
            canonical.display()
        ))
    })?;
    Ok(canonical)
}

/// `GET /api/hub/scan-folders`: папки с текущим статусом доступа.
pub async fn list(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let folders = state.scan_folders.read().await.clone();
    let folders = tokio::task::spawn_blocking(move || {
        folders
            .into_iter()
            .map(|mut folder| {
                folder.status = Some(folder_status(Path::new(&folder.path)).to_string());
                folder
            })
            .collect::<Vec<_>>()
    })
    .await
    .map_err(|err| ApiError::internal(format!("Проверка папок прервана: {err}")))?;
    Ok(Json(json!({ "folders": folders })))
}

#[derive(Debug, Default, Deserialize)]
pub struct AddFolderRequest {
    #[serde(default)]
    path: Option<String>,
}

/// `POST /api/hub/scan-folders`: добавляет существующую папку.
pub async fn add(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AddFolderRequest>,
) -> ApiResult<Json<ScanFolderEntry>> {
    let raw = request
        .path
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .ok_or_else(|| ApiError::bad_request("Укажите путь к папке"))?;
    let candidate = PathBuf::from(raw);
    if !candidate.is_absolute() {
        return Err(ApiError::bad_request(format!(
            "Нужен полный путь к папке, а не «{raw}»"
        )));
    }
    let models_dir = state.models_dir.clone();
    let (canonical, models_dir) =
        tokio::task::spawn_blocking(move || -> ApiResult<(PathBuf, Option<PathBuf>)> {
            Ok((validate_folder(&candidate)?, models_dir.canonicalize().ok()))
        })
        .await
        .map_err(|err| ApiError::internal(format!("Проверка папки прервана: {err}")))??;

    // Изменения списка идут по одному: блокировка держится до записи в базу
    let mut folders = state.scan_folders.write().await;
    let already_added = models_dir.as_deref() == Some(canonical.as_path())
        || folders
            .iter()
            .any(|folder| Path::new(&folder.path) == canonical);
    if already_added {
        return Err(ApiError::conflict(format!(
            "Папка {} уже просматривается",
            canonical.display()
        )));
    }

    let stored = read_stored(&state).await?;
    let largest_id = folders.iter().map(|folder| folder.id).max().unwrap_or(0);
    let id = stored.next_id.max(largest_id) + 1;
    let entry = ScanFolderEntry {
        id,
        path: canonical.to_string_lossy().into_owned(),
        created_at: crate::state::iso_now(),
        status: None,
    };
    let mut updated = folders.clone();
    updated.push(entry.clone());
    write_stored(
        &state,
        &StoredFolders {
            next_id: id,
            folders: updated.clone(),
        },
    )
    .await?;
    *folders = updated;
    tracing::info!("Добавлена папка с моделями {}", entry.path);

    Ok(Json(ScanFolderEntry {
        status: Some(STATUS_OK.to_string()),
        ..entry
    }))
}

/// `DELETE /api/hub/scan-folders/:id`: убирает папку из списка (файлы не трогаются).
pub async fn remove(
    State(state): State<Arc<AppState>>,
    AxPath(id): AxPath<u64>,
) -> ApiResult<Json<Value>> {
    let mut folders = state.scan_folders.write().await;
    if !folders.iter().any(|folder| folder.id == id) {
        return Err(ApiError::not_found(format!(
            "Папка с номером {id} не найдена"
        )));
    }
    let remaining: Vec<ScanFolderEntry> = folders
        .iter()
        .filter(|folder| folder.id != id)
        .cloned()
        .collect();
    let largest_id = folders.iter().map(|folder| folder.id).max().unwrap_or(0);
    let next_id = read_stored(&state).await?.next_id.max(largest_id);
    write_stored(
        &state,
        &StoredFolders {
            next_id,
            folders: remaining.clone(),
        },
    )
    .await?;
    *folders = remaining;

    Ok(Json(json!({
        "status": "ok",
        "deleted": true
    })))
}
