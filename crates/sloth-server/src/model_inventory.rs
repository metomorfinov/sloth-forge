//! Локальные GGUF-модели: поиск файлов в папках моделей и сводка их метаданных.
//!
//! Раньше список моделей угадывал число параметров по размеру файла («больше 3 ГБ —
//! значит 7B»), подмешивал четыре модели, которых нет на диске, и не видел моделей,
//! разбитых на несколько файлов. Здесь всё берётся из самих файлов: размеры —
//! с диска, число весов и архитектура — из заголовков GGUF.

use sloth_core::gguf::{GGMLType, GGUFError, GGUFFile};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// Глубина поиска во вложенных папках (`models/<org>/<repo>/<квант>/файл.gguf`).
const MAX_SCAN_DEPTH: usize = 4;
/// Предел числа просмотренных записей: защита от случайно добавленной огромной папки.
const MAX_SCANNED_ENTRIES: usize = 20_000;
/// Длина номера шарда в имени файла (`-00001-of-00005`), как у утилиты gguf-split.
const SHARD_NUMBER_DIGITS: usize = 5;

/// Модель на диске. Для модели из нескольких файлов (шардов) `path` — первый шард.
#[derive(Debug, Clone)]
pub struct LocalModel {
    /// Канонический путь к первому (или единственному) файлу.
    pub path: PathBuf,
    pub file_name: String,
    /// Имя без `.gguf` и без номера шарда.
    pub display_name: String,
    /// Все файлы модели по порядку.
    pub shard_paths: Vec<PathBuf>,
    /// Суммарный размер всех найденных файлов.
    pub size_bytes: u64,
    /// Все шарды на месте (у недокачанной модели части может не быть).
    pub complete: bool,
    pub modified_secs: Option<u64>,
    /// Индекс корня, в котором найдена модель: 0 — основная папка моделей, дальше scan-folders.
    pub root_index: usize,
    /// Путь первого файла относительно корня (`unsloth/Llama-GGUF/Q4_K_M/файл.gguf`).
    pub relative_path: PathBuf,
}

impl LocalModel {
    /// Идентификатор для API — полный путь: он однозначен и проходит песочницу путей.
    pub fn id(&self) -> String {
        self.path.to_string_lossy().into_owned()
    }

    fn relative_parts(&self) -> Vec<String> {
        self.relative_path
            .components()
            .filter_map(|component| match component {
                std::path::Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
                _ => None,
            })
            .collect()
    }

    /// Репозиторий Hugging Face: из папок `<org>/<repo>/…` (так раскладывает загрузчик),
    /// иначе — догадка по имени файла.
    pub fn repo_id(&self) -> String {
        let parts = self.relative_parts();
        if parts.len() >= 3 {
            format!("{}/{}", parts[0], parts[1])
        } else {
            crate::hub::infer_repo_id_from_gguf_filename(&self.file_name)
        }
    }

    /// Путь первого файла внутри репозитория (`Q4_K_M/файл.gguf`) или просто имя файла.
    pub fn path_in_repo(&self) -> String {
        let parts = self.relative_parts();
        if parts.len() >= 3 {
            parts[2..].join("/")
        } else {
            self.file_name.clone()
        }
    }

    /// Вариант квантования по пути внутри репозитория.
    pub fn quant(&self) -> String {
        crate::hub::extract_quant_from_path(&self.path_in_repo())
    }
}

fn is_gguf_name(name: &str) -> bool {
    name.to_ascii_lowercase().ends_with(".gguf")
}

/// Файл проектора изображений (`mmproj-*.gguf`) — часть мультимодальной модели, а не модель.
fn is_projector_name(name: &str) -> bool {
    name.to_ascii_lowercase().contains("mmproj")
}

/// Разбирает имя шарда `<основа>-00002-of-00005.gguf` → (основа, номер, всего).
fn split_shard_name(file_name: &str) -> Option<(&str, u32, u32)> {
    let stem = file_name
        .strip_suffix(".gguf")
        .or_else(|| file_name.strip_suffix(".GGUF"))?;
    let (rest, total) = stem.rsplit_once("-of-")?;
    let (base, index) = rest.rsplit_once('-')?;
    if index.len() != SHARD_NUMBER_DIGITS || total.len() != SHARD_NUMBER_DIGITS {
        return None;
    }
    let (index, total) = (index.parse().ok()?, total.parse().ok()?);
    (index >= 1 && index <= total).then_some((base, index, total))
}

fn modified_secs(path: &Path) -> Option<u64> {
    fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|elapsed| elapsed.as_secs())
}

/// Находит все модели в корнях. Одна и та же папка, добавленная дважды, учитывается один раз.
/// Функция синхронная: вызывать через `spawn_blocking`.
pub fn scan_models(roots: &[PathBuf]) -> Vec<LocalModel> {
    let mut seen = HashSet::new();
    let mut models = Vec::new();
    let mut budget = MAX_SCANNED_ENTRIES;
    for (root_index, root) in roots.iter().enumerate() {
        let Ok(root) = root.canonicalize() else {
            continue;
        };
        walk(&root, &root, 0, root_index, &mut seen, &mut models, &mut budget);
    }
    models.sort_by_key(|model| model.display_name.to_lowercase());
    models
}

fn walk(
    root: &Path,
    dir: &Path,
    depth: usize,
    root_index: usize,
    seen: &mut HashSet<PathBuf>,
    models: &mut Vec<LocalModel>,
    budget: &mut usize,
) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) => {
            tracing::warn!(
                "Не удалось прочитать папку моделей {}: {err}",
                dir.display()
            );
            return;
        }
    };

    let mut files: Vec<(String, PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        if *budget == 0 {
            tracing::warn!(
                "Поиск моделей остановлен: просмотрено больше {MAX_SCANNED_ENTRIES} записей"
            );
            return;
        }
        *budget -= 1;

        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        // file_type() не следует по символическим ссылкам: ссылки наружу не обходятся
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            if depth < MAX_SCAN_DEPTH {
                walk(root, &entry.path(), depth + 1, root_index, seen, models, budget);
            }
        } else if file_type.is_file() && is_gguf_name(&name) && !is_projector_name(&name) {
            files.push((name, entry.path()));
        }
    }

    for (name, path) in &files {
        let (display_name, shard_paths) = match split_shard_name(name) {
            // Модель из нескольких файлов представлена первым шардом
            Some((_, index, _)) if index != 1 => continue,
            Some((base, _, total)) => {
                let shards: Vec<PathBuf> = (1..=total)
                    .map(|i| dir.join(format!("{base}-{i:05}-of-{total:05}.gguf")))
                    .filter(|shard| shard.is_file())
                    .collect();
                (base.to_string(), (shards, total as usize))
            }
            None => (
                name.trim_end_matches(".gguf")
                    .trim_end_matches(".GGUF")
                    .to_string(),
                (vec![path.clone()], 1),
            ),
        };
        let (shard_paths, expected_shards) = shard_paths;
        let Ok(canonical) = path.canonicalize() else {
            continue;
        };
        if !seen.insert(canonical.clone()) {
            continue;
        }
        let size_bytes = shard_paths
            .iter()
            .filter_map(|shard| fs::metadata(shard).ok())
            .fold(0u64, |sum, meta| sum.saturating_add(meta.len()));
        models.push(LocalModel {
            relative_path: canonical
                .strip_prefix(root)
                .map(Path::to_path_buf)
                .unwrap_or_else(|_| PathBuf::from(name)),
            complete: shard_paths.len() == expected_shards,
            modified_secs: modified_secs(&canonical),
            path: canonical,
            file_name: name.clone(),
            display_name,
            shard_paths,
            size_bytes,
            root_index,
        });
    }
}

/// Лежит ли рядом с моделью файл проектора изображений (признак мультимодальной модели).
/// Синхронная: вызывать через `spawn_blocking`.
pub fn has_projector(model: &LocalModel) -> bool {
    let Some(dir) = model.path.parent() else {
        return false;
    };
    fs::read_dir(dir)
        .map(|entries| {
            entries.flatten().any(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                is_gguf_name(&name) && is_projector_name(&name)
            })
        })
        .unwrap_or(false)
}

/// Сводка метаданных модели, нужная интерфейсу и оценке памяти.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgufSummary {
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
    pub block_count: Option<u64>,
    pub embedding_length: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub key_length: Option<u64>,
    pub value_length: Option<u64>,
    pub expert_count: Option<u64>,
    /// Часть слоёв смотрит только на последние N токенов (скользящее окно).
    pub uses_sliding_window: bool,
    /// Несколько слоёв используют общий KV-кэш.
    pub shares_kv_between_layers: bool,
    /// Число весов по всем шардам.
    pub parameter_count: u64,
    pub tensor_count: u64,
    /// Шаблон чата из `tokenizer.chat_template`.
    pub chat_template: Option<String>,
}

impl GgufSummary {
    fn from_file(file: &GGUFFile) -> Self {
        Self {
            architecture: file.architecture().map(str::to_string),
            context_length: file.context_length(),
            block_count: file.block_count(),
            embedding_length: file.embedding_length(),
            head_count: file.head_count(),
            head_count_kv: file.head_count_kv(),
            key_length: file.arch_u64("attention.key_length"),
            value_length: file.arch_u64("attention.value_length"),
            expert_count: file.arch_u64("expert_count"),
            uses_sliding_window: file.arch_value("attention.sliding_window").is_some(),
            shares_kv_between_layers: file
                .arch_u64("attention.shared_kv_layers")
                .is_some_and(|layers| layers > 0),
            parameter_count: file.parameter_count(),
            tensor_count: file.tensor_count,
            chat_template: file
                .metadata
                .get("tokenizer.chat_template")
                .and_then(|value| value.as_str())
                .map(str::to_string),
        }
    }

    /// Число MoE-слоёв: у модели с экспертами ими считаются все блоки, у обычной — 0.
    pub fn moe_layer_count(&self) -> Option<u64> {
        match self.expert_count {
            Some(experts) if experts > 0 => self.block_count,
            _ => Some(0),
        }
    }
}

/// Читает заголовки всех шардов модели. Синхронная: вызывать через `spawn_blocking`.
pub fn summarize(model: &LocalModel) -> Result<GgufSummary, GGUFError> {
    let first = GGUFFile::open(&model.path)?;
    let mut summary = GgufSummary::from_file(&first);
    for shard in model
        .shard_paths
        .iter()
        .filter(|shard| **shard != model.path)
    {
        let file = GGUFFile::open(shard)?;
        summary.parameter_count = summary
            .parameter_count
            .saturating_add(file.parameter_count());
        summary.tensor_count = summary.tensor_count.saturating_add(file.tensor_count);
    }
    Ok(summary)
}

/// Размер KV-кэша в байтах: на каждый слой и каждый токен контекста хранятся ключи
/// и значения всех KV-голов. `None`, если архитектура сложнее (скользящее окно, общий KV
/// между слоями) или в заголовке нет нужных размерностей — тогда число не выдумывается.
pub fn kv_cache_bytes(summary: &GgufSummary, n_ctx: u64, cache_type: GGMLType) -> Option<u64> {
    if summary.uses_sliding_window || summary.shares_kv_between_layers {
        return None;
    }
    let layers = summary.block_count?;
    let kv_heads = summary.head_count_kv.or(summary.head_count)?;
    let head_dim = summary
        .embedding_length
        .zip(summary.head_count)
        .and_then(|(embedding, heads)| embedding.checked_div(heads));
    let key_length = summary.key_length.or(head_dim)?;
    let value_length = summary.value_length.or(head_dim)?;
    let elements = n_ctx
        .checked_mul(layers)?
        .checked_mul(kv_heads)?
        .checked_mul(key_length.checked_add(value_length)?)?;
    let layout = cache_type.block_layout()?;
    elements
        .checked_mul(layout.bytes_per_block)?
        .checked_div(layout.elements_per_block)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_shard_names() {
        assert_eq!(
            split_shard_name("DeepSeek-UD-Q4_K_XL-00002-of-00005.gguf"),
            Some(("DeepSeek-UD-Q4_K_XL", 2, 5))
        );
        assert_eq!(split_shard_name("Llama-3.2-1B-Instruct-Q4_K_M.gguf"), None);
        assert_eq!(split_shard_name("model-2-of-5.gguf"), None);
        assert_eq!(split_shard_name("model-00006-of-00005.gguf"), None);
    }

    #[test]
    fn scans_nested_folders_shards_and_skips_projectors() {
        let base = tempfile::tempdir().expect("не удалось создать временную папку");
        let root = base.path();
        fs::create_dir_all(root.join("org/repo/UD-Q4")).unwrap();
        fs::write(root.join("single-Q8_0.gguf"), b"GGUF").unwrap();
        fs::write(root.join("mmproj-F16.gguf"), b"GGUF").unwrap();
        fs::write(root.join("notes.txt"), b"x").unwrap();
        for i in 1..=3 {
            fs::write(
                root.join(format!("org/repo/UD-Q4/big-{i:05}-of-00003.gguf")),
                b"GGUF1234",
            )
            .unwrap();
        }
        fs::create_dir_all(root.join("partial")).unwrap();
        fs::write(root.join("partial/half-00001-of-00002.gguf"), b"GGUF").unwrap();

        // Одна и та же папка дважды — модели не дублируются
        let models = scan_models(&[root.to_path_buf(), root.to_path_buf()]);
        let names: Vec<&str> = models.iter().map(|m| m.display_name.as_str()).collect();
        assert_eq!(names, ["big", "half", "single-Q8_0"]);

        let big = &models[0];
        assert_eq!(big.shard_paths.len(), 3);
        assert_eq!(big.size_bytes, 24);
        assert!(big.complete);
        assert!(!models[1].complete, "недокачанная модель помечается");
    }

    fn llama_like() -> GgufSummary {
        GgufSummary {
            architecture: Some("llama".into()),
            block_count: Some(16),
            embedding_length: Some(2048),
            head_count: Some(32),
            head_count_kv: Some(8),
            key_length: Some(64),
            value_length: Some(64),
            ..GgufSummary::default()
        }
    }

    #[test]
    fn kv_cache_matches_llama_cpp_for_llama_3_2_1b() {
        // llama.cpp для Llama-3.2-1B при n_ctx=4096 и f16 сообщает KV-кэш 128 МиБ
        let bytes = kv_cache_bytes(&llama_like(), 4096, GGMLType::F16).unwrap();
        assert_eq!(bytes, 128 * 1024 * 1024);
        let q8 = kv_cache_bytes(&llama_like(), 4096, GGMLType::Q8_0).unwrap();
        assert_eq!(q8, 128 * 1024 * 1024 / 2 * 34 / 32);
    }

    #[test]
    fn kv_cache_is_not_invented_for_complex_architectures() {
        let mut gemma_like = llama_like();
        gemma_like.uses_sliding_window = true;
        assert_eq!(kv_cache_bytes(&gemma_like, 4096, GGMLType::F16), None);

        let mut missing_dims = llama_like();
        missing_dims.head_count_kv = None;
        missing_dims.head_count = None;
        assert_eq!(kv_cache_bytes(&missing_dims, 4096, GGMLType::F16), None);
    }

    #[test]
    fn summarizes_real_llama_when_present() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/Llama-3.2-1B-Instruct-Q4_K_M.gguf");
        if !path.is_file() {
            eprintln!("пропуск: {} не найден", path.display());
            return;
        }
        let models = scan_models(&[path.parent().unwrap().to_path_buf()]);
        let model = models
            .iter()
            .find(|m| m.file_name == "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
            .expect("модель найдена сканером");
        let summary = summarize(model).expect("заголовок читается");
        assert_eq!(summary.block_count, Some(16));
        assert_eq!(summary.moe_layer_count(), Some(0));
        assert!(summary.chat_template.is_some());
        assert_eq!(
            kv_cache_bytes(&summary, 4096, GGMLType::F16),
            Some(128 * 1024 * 1024)
        );
    }
}
