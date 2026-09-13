//! Варианты GGUF-модели: `/api/hub/gguf-variants` и `/api/models/gguf-variants`.
//!
//! Раньше при недоступном Hugging Face список выдумывался (файлы `model-Q4_K_M.gguf`
//! с размером «по названию репозитория»), «скачано» определялось нечётким совпадением
//! имени файла, а длина контекста всегда была 131072. Здесь варианты берутся из настоящего
//! дерева файлов репозитория, признаки «скачано» и «недокачано» — из файлов на диске,
//! длина контекста — из заголовка скачанного GGUF. Без сети отдаются только локальные файлы.

use crate::downloads;
use crate::error::{ApiError, ApiResult};
use crate::hf_api::{self, RemoteFile};
use crate::model_inventory::{self, LocalModel};
use crate::state::AppState;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

/// Вариант по умолчанию, когда ничего не скачано: первый найденный из списка
/// (разумный баланс качества и размера), иначе самый маленький.
const PREFERRED_DEFAULT_QUANTS: &[&str] = &[
    "UD-Q4_K_XL",
    "Q4_K_M",
    "Q4_K_S",
    "IQ4_XS",
    "IQ4_NL",
    "Q4_0",
    "Q5_K_M",
    "Q8_0",
];
/// Недокачанный вариант продолжается обычной HTTP-загрузкой.
const PARTIAL_TRANSPORT: &str = "http";

#[derive(Debug, Default, Deserialize)]
pub struct VariantsQuery {
    #[serde(default)]
    repo_id: Option<String>,
    #[serde(default)]
    local_path: Option<String>,
    #[serde(default)]
    prefer_local_cache: Option<bool>,
    #[serde(default)]
    offline: Option<bool>,
}

/// Вариант для ответа (`GgufVariantDetail` во фронтенде).
#[derive(Debug, Clone)]
struct Variant {
    quant: String,
    /// Первый файл: путь внутри репозитория или имя локального файла.
    filename: String,
    size_bytes: u64,
    shard_count: usize,
    downloaded: bool,
    partial: bool,
    /// Сколько байт осталось докачать (известно только для варианта из дерева HF).
    remaining_bytes: Option<u64>,
    /// Скачанная копия на диске: из неё читается длина контекста.
    local: Option<LocalModel>,
}

impl Variant {
    fn from_local(model: &LocalModel) -> Self {
        Self {
            quant: model.quant(),
            filename: model.path_in_repo(),
            size_bytes: model.size_bytes,
            shard_count: model.shard_paths.len(),
            downloaded: model.complete,
            partial: !model.complete,
            remaining_bytes: None,
            local: model.complete.then(|| model.clone()),
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "filename": self.filename,
            "quant": self.quant,
            "display_label": null,
            "size_bytes": self.size_bytes,
            "download_size_bytes": self.size_bytes,
            "shard_count": self.shard_count,
            "download_remaining_bytes": self.remaining_bytes,
            "downloaded": self.downloaded,
            "update_available": false,
            "partial": self.partial,
            "partial_transport": self.partial.then_some(PARTIAL_TRANSPORT),
            "partial_resumable": self.partial,
            "dependency_key": null
        })
    }
}

/// Что лежит на диске для одного файла репозитория.
#[derive(Debug, Clone, Copy, Default)]
struct DiskFile {
    /// Файл на месте и его размер совпадает с размером на Hugging Face.
    complete: bool,
    /// Сколько байт уже есть: весь файл или его `.part`.
    bytes: u64,
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn is_projector_path(path: &str) -> bool {
    path.to_ascii_lowercase().contains("mmproj")
}

/// Состояние файлов репозитория в `models/<org>/<repo>`. Синхронная: через `spawn_blocking`.
fn disk_files(repo_dir: &Path, files: &[RemoteFile]) -> HashMap<String, DiskFile> {
    files
        .iter()
        .filter_map(|file| {
            let target = repo_dir.join(downloads::safe_relative_path(&file.path)?);
            let length = std::fs::metadata(&target)
                .ok()
                .filter(|meta| meta.is_file())
                .map(|meta| meta.len());
            let complete = length.is_some_and(|len| file.size.is_none_or(|size| size == len));
            let bytes = if complete {
                length.unwrap_or(0)
            } else {
                std::fs::metadata(downloads::part_path(&target))
                    .map(|meta| meta.len())
                    .unwrap_or(0)
            };
            Some((file.path.clone(), DiskFile { complete, bytes }))
        })
        .collect()
}

/// Добавляет локальные варианты, которых нет в списке (например, квант, сделанный самим
/// пользователем). Полная копия важнее недокачанной.
fn merge_local(variants: &mut Vec<Variant>, local: &[LocalModel]) {
    let mut ordered: Vec<&LocalModel> = local.iter().collect();
    ordered.sort_by_key(|model| !model.complete);
    let mut seen: HashSet<String> = variants
        .iter()
        .map(|variant| variant.quant.to_ascii_lowercase())
        .collect();
    for model in ordered {
        if seen.insert(model.quant().to_ascii_lowercase()) {
            variants.push(Variant::from_local(model));
        }
    }
    variants.sort_by(|a, b| {
        a.size_bytes
            .cmp(&b.size_bytes)
            .then_with(|| a.quant.cmp(&b.quant))
    });
}

/// Варианты из дерева файлов HF. `local` — локальные модели этого же репозитория.
fn remote_variants(
    files: &[RemoteFile],
    disk: &HashMap<String, DiskFile>,
    local: &[LocalModel],
) -> Vec<Variant> {
    let mut groups: BTreeMap<String, Vec<&RemoteFile>> = BTreeMap::new();
    for file in files
        .iter()
        .filter(|file| !crate::hub::is_auxiliary_gguf_file(&file.path))
    {
        groups
            .entry(crate::hub::extract_quant_from_path(&file.path))
            .or_default()
            .push(file);
    }

    let mut variants: Vec<Variant> = groups
        .into_iter()
        .map(|(quant, group)| {
            let size_bytes: u64 = group.iter().filter_map(|file| file.size).sum();
            let on_disk: Vec<DiskFile> = group
                .iter()
                .map(|file| disk.get(&file.path).copied().unwrap_or_default())
                .collect();
            // Скачанная копия: в папке org/repo или файл, положенный в папку моделей вручную
            let local_model = local
                .iter()
                .find(|model| model.complete && model.quant().eq_ignore_ascii_case(&quant))
                .cloned();
            let downloaded = on_disk.iter().all(|file| file.complete) || local_model.is_some();
            let present: u64 = on_disk.iter().map(|file| file.bytes).sum();
            let partial = !downloaded && present > 0;
            Variant {
                filename: group[0].path.clone(),
                shard_count: group.len(),
                downloaded,
                partial,
                remaining_bytes: partial.then(|| size_bytes.saturating_sub(present)),
                local: local_model,
                size_bytes,
                quant,
            }
        })
        .collect();
    merge_local(&mut variants, local);
    variants
}

/// Скачанный вариант, иначе предпочтительный квант, иначе самый маленький.
fn default_variant(variants: &[Variant]) -> Option<String> {
    variants
        .iter()
        .find(|variant| variant.downloaded)
        .or_else(|| {
            PREFERRED_DEFAULT_QUANTS.iter().find_map(|preferred| {
                variants
                    .iter()
                    .find(|variant| variant.quant.eq_ignore_ascii_case(preferred))
            })
        })
        .or_else(|| variants.first())
        .map(|variant| variant.quant.clone())
}

/// Длина контекста из заголовка GGUF; `None`, если ключа нет или файл не читается.
async fn context_length_of(model: &LocalModel) -> Option<u64> {
    let path = model.path.clone();
    let result = tokio::task::spawn_blocking(move || {
        sloth_core::gguf::GGUFFile::open(&path).map(|file| file.context_length())
    })
    .await;
    match result {
        Ok(Ok(length)) => length,
        Ok(Err(err)) => {
            tracing::warn!(
                "Не удалось прочитать заголовок {}: {err}",
                model.path.display()
            );
            None
        }
        Err(err) => {
            tracing::warn!("Чтение заголовка {} прервано: {err}", model.path.display());
            None
        }
    }
}

async fn local_has_projector(local: &[LocalModel]) -> ApiResult<bool> {
    let local = local.to_vec();
    tokio::task::spawn_blocking(move || local.iter().any(model_inventory::has_projector))
        .await
        .map_err(|err| {
            ApiError::internal(format!("Проверка проектора изображений прервана: {err}"))
        })
}

async fn respond(repo_id: &str, variants: Vec<Variant>, has_vision: bool) -> Json<Value> {
    let context_length = match variants.iter().find_map(|variant| variant.local.as_ref()) {
        Some(model) => context_length_of(model).await,
        None => None,
    };
    Json(json!({
        "repo_id": repo_id,
        "variants": variants.iter().map(Variant::to_json).collect::<Vec<_>>(),
        "has_vision": has_vision,
        "default_variant": default_variant(&variants),
        "context_length": context_length
    }))
}

/// Вариант одного конкретного файла модели (путь из интерфейса или имя файла).
async fn local_file_variants(
    state: &AppState,
    candidate: &str,
    repo_query: Option<&str>,
) -> ApiResult<Json<Value>> {
    let not_found = || {
        ApiError::not_found(format!(
            "Файл модели «{candidate}» не найден в папках моделей"
        ))
    };
    let path = crate::hub::resolve_local_gguf_file(state, candidate)
        .await
        .ok_or_else(not_found)?;
    let model = crate::hub::local_models(state)
        .await?
        .into_iter()
        .find(|model| model.path == path || model.shard_paths.contains(&path))
        .ok_or_else(not_found)?;
    let repo_id = repo_query
        .map(str::to_string)
        .unwrap_or_else(|| model.repo_id());
    let has_vision = local_has_projector(std::slice::from_ref(&model)).await?;
    Ok(respond(&repo_id, vec![Variant::from_local(&model)], has_vision).await)
}

async fn local_repo_variants(repo_id: &str, local: &[LocalModel]) -> ApiResult<Json<Value>> {
    let mut variants = Vec::new();
    merge_local(&mut variants, local);
    let has_vision = local_has_projector(local).await?;
    Ok(respond(repo_id, variants, has_vision).await)
}

/// `GET /api/hub/gguf-variants`: варианты квантования репозитория или локального файла.
pub async fn handle_gguf_variants(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<VariantsQuery>,
) -> ApiResult<Json<Value>> {
    let repo_query = non_empty(query.repo_id.as_deref());
    let local_path = non_empty(query.local_path.as_deref());
    let repo_is_valid = repo_query.is_some_and(|repo| downloads::validate_repo_id(repo).is_ok());

    // 1. Конкретный файл: путь из интерфейса или имя файла вместо репозитория
    if let Some(candidate) = local_path.or(repo_query.filter(|_| !repo_is_valid)) {
        return local_file_variants(&state, candidate, repo_query).await;
    }
    let Some(repo_id) = repo_query else {
        return Err(ApiError::bad_request("Укажите repo_id или local_path"));
    };

    // 2. Репозиторий Hugging Face
    let local: Vec<LocalModel> = crate::hub::local_models(&state)
        .await?
        .into_iter()
        .filter(|model| model.repo_id().eq_ignore_ascii_case(repo_id))
        .collect();
    let wants_local = query.offline == Some(true)
        || (query.prefer_local_cache == Some(true) && local.iter().any(|model| model.complete));
    if wants_local {
        return local_repo_variants(repo_id, &local).await;
    }

    let token = hf_api::resolve_token(&state, &headers).await?;
    match hf_api::cached_repo_gguf_files(&state, repo_id, token.as_deref()).await {
        Ok(files) => {
            let dir = state.models_dir.join(repo_id);
            let listed = files.clone();
            let disk = tokio::task::spawn_blocking(move || disk_files(&dir, &listed))
                .await
                .map_err(|err| {
                    ApiError::internal(format!("Проверка файлов на диске прервана: {err}"))
                })?;
            let has_vision = files.iter().any(|file| is_projector_path(&file.path));
            Ok(respond(repo_id, remote_variants(&files, &disk, &local), has_vision).await)
        }
        Err(message) if !local.is_empty() => {
            tracing::warn!("Варианты {repo_id} взяты только с диска: {message}");
            local_repo_variants(repo_id, &local).await
        }
        Err(message) => Err(ApiError::new(StatusCode::BAD_GATEWAY, message)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn remote(path: &str, size: u64) -> RemoteFile {
        RemoteFile {
            path: path.into(),
            size: Some(size),
        }
    }

    fn local_model(relative: &str, size_bytes: u64, complete: bool) -> LocalModel {
        let relative_path = PathBuf::from(relative);
        let file_name = relative_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let path = PathBuf::from("/models").join(&relative_path);
        LocalModel {
            display_name: file_name.trim_end_matches(".gguf").into(),
            shard_paths: vec![path.clone()],
            path,
            file_name,
            size_bytes,
            complete,
            modified_secs: None,
            root_index: 0,
            relative_path,
        }
    }

    #[test]
    fn groups_shards_and_reads_disk_state() {
        let files = vec![
            remote("Q4_K_M/m-00001-of-00002.gguf", 100),
            remote("Q4_K_M/m-00002-of-00002.gguf", 50),
            remote("m-Q8_0.gguf", 300),
            remote("mmproj-F16.gguf", 10),
        ];
        let mut disk = HashMap::new();
        disk.insert(
            "Q4_K_M/m-00001-of-00002.gguf".to_string(),
            DiskFile {
                complete: true,
                bytes: 100,
            },
        );
        disk.insert(
            "Q4_K_M/m-00002-of-00002.gguf".to_string(),
            DiskFile {
                complete: false,
                bytes: 20,
            },
        );

        let variants = remote_variants(&files, &disk, &[]);
        let quants: Vec<&str> = variants.iter().map(|v| v.quant.as_str()).collect();
        assert_eq!(
            quants,
            ["Q4_K_M", "Q8_0"],
            "проектор — не вариант; порядок по размеру"
        );

        let q4 = &variants[0];
        assert_eq!((q4.size_bytes, q4.shard_count), (150, 2));
        assert_eq!(q4.filename, "Q4_K_M/m-00001-of-00002.gguf");
        assert!(!q4.downloaded && q4.partial);
        assert_eq!(q4.remaining_bytes, Some(30));
        assert!(!variants[1].downloaded && !variants[1].partial);
        assert_eq!(default_variant(&variants).as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn manual_files_count_as_downloaded_and_extra_local_quants_are_kept() {
        let files = vec![remote("m-Q8_0.gguf", 300), remote("m-Q4_K_M.gguf", 150)];
        let local = vec![
            local_model("m-Q8_0.gguf", 300, true),
            local_model("m-Q2_K.gguf", 70, true),
        ];
        let variants = remote_variants(&files, &HashMap::new(), &local);
        let quants: Vec<&str> = variants.iter().map(|v| v.quant.as_str()).collect();
        assert_eq!(quants, ["Q2_K", "Q4_K_M", "Q8_0"]);
        assert!(variants[2].downloaded && variants[2].local.is_some());
        assert!(!variants[1].downloaded);
        assert_eq!(
            default_variant(&variants).as_deref(),
            Some("Q2_K"),
            "скачанный вариант выбирается первым"
        );
    }

    #[test]
    fn default_prefers_balanced_quant_when_nothing_downloaded() {
        let files = vec![
            remote("m-Q2_K.gguf", 70),
            remote("m-Q4_K_M.gguf", 150),
            remote("m-Q8_0.gguf", 300),
        ];
        let variants = remote_variants(&files, &HashMap::new(), &[]);
        assert_eq!(default_variant(&variants).as_deref(), Some("Q4_K_M"));
        assert_eq!(default_variant(&[]), None);
    }
}
