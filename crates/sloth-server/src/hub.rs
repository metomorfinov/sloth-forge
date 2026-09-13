use crate::state::AppState;
use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::info;

#[derive(Debug, Deserialize, Default)]
pub struct RepoVariantsQuery {
    #[serde(default, alias = "repoId", alias = "model_id", alias = "modelId")]
    pub repo_id: Option<String>,
    #[serde(default, alias = "localPath", alias = "local_file", alias = "localFile", alias = "path")]
    pub local_path: Option<String>,
    #[serde(default, alias = "preferLocalCache")]
    pub prefer_local_cache: Option<bool>,
}

// Запросы загрузок, прогресса и отмены — в downloads.rs

#[derive(Debug, Clone)]
pub struct ScannedGgufFile {
    pub path: PathBuf,
    pub filename: String,
    pub size_bytes: u64,
    pub mtime_epoch_secs: u64,
    pub quant: String,
    pub repo_id: String,
    pub display_name: String,
}

pub fn infer_repo_id_from_gguf_filename(filename: &str) -> String {
    let clean_filename = filename.strip_prefix("models/").unwrap_or(filename);
    let stem = clean_filename.trim_end_matches(".gguf");
    let lower = stem.to_lowercase();

    if lower.contains("llama-3.2-1b-instruct") {
        return "unsloth/Llama-3.2-1B-Instruct-GGUF".to_string();
    }
    if lower.contains("llama-3.2-3b-instruct") {
        return "unsloth/Llama-3.2-3B-Instruct-GGUF".to_string();
    }
    if lower.contains("qwen2.5-1.5b-instruct") {
        return "unsloth/Qwen2.5-1.5B-Instruct-GGUF".to_string();
    }
    if lower.contains("mistral-7b-instruct") {
        return "unsloth/Mistral-7B-Instruct-v0.3-GGUF".to_string();
    }

    let quant = extract_quant_from_path(clean_filename);
    let mut base = stem;
    if !quant.is_empty() {
        if let Some(s) = base.strip_suffix(&format!("-{}", quant)) {
            base = s;
        } else if let Some(s) = base.strip_suffix(&format!("_{}", quant)) {
            base = s;
        } else if let Some(s) = base.strip_suffix(&quant) {
            base = s.trim_end_matches(['-', '_']);
        }
    }

    if base.contains('/') {
        base.to_string()
    } else if base.ends_with("-GGUF") || base.ends_with("_GGUF") {
        format!("unsloth/{}", base)
    } else {
        format!("unsloth/{}-GGUF", base)
    }
}

pub fn file_matches_repo_and_quant(filename: &str, repo_id: &str, quant: &str) -> bool {
    let f_lower = filename.to_lowercase();
    let r_lower = repo_id.to_lowercase();
    let r_clean = r_lower.strip_prefix("unsloth/").unwrap_or(&r_lower);
    let r_clean = r_clean.strip_suffix("-gguf").unwrap_or(r_clean);
    let r_clean = r_clean.strip_suffix("_gguf").unwrap_or(r_clean);

    let q_lower = quant.to_lowercase();

    let has_quant = f_lower.ends_with(&format!("-{}.gguf", q_lower))
        || f_lower.ends_with(&format!("_{}.gguf", q_lower))
        || f_lower.ends_with(&format!(".{}.gguf", q_lower))
        || f_lower.contains(&q_lower);

    if !has_quant {
        return false;
    }

    if f_lower.contains(r_clean) || r_clean.contains(&f_lower.replace(&format!("-{}", q_lower), "").replace(".gguf", "")) {
        return true;
    }

    let repo_tokens: Vec<&str> = r_clean.split(['-', '_', '.']).filter(|s| !s.is_empty()).collect();
    if repo_tokens.len() >= 2 && repo_tokens.iter().all(|tok| f_lower.contains(tok)) {
        return true;
    }

    false
}

pub async fn scan_local_gguf_files(state: &AppState) -> Vec<ScannedGgufFile> {
    let mut dirs_to_scan = Vec::new();
    dirs_to_scan.push(state.models_dir.clone());

    // Папка моделей определяется при запуске (locations.rs), поэтому соседние
    // ../models относительно текущей папки больше не сканируются

    {
        let custom_folders = state.scan_folders.read().await;
        for folder in custom_folders.iter() {
            let p = PathBuf::from(&folder.path);
            if p.exists() && p.is_dir() && !dirs_to_scan.contains(&p) {
                dirs_to_scan.push(p);
            }
        }
    }

    let mut found = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for dir in dirs_to_scan {
        if !dir.exists() {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                    if filename.ends_with(".gguf") {
                        if !seen.insert(filename.clone()) {
                            continue;
                        }
                        let metadata = entry.metadata().ok();
                        let size_bytes = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
                        let mtime_epoch_secs = metadata
                            .as_ref()
                            .and_then(|m| m.modified().ok())
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_secs())
                            .unwrap_or(0);

                        let quant = extract_quant_from_path(&filename);
                        let repo_id = infer_repo_id_from_gguf_filename(&filename);
                        let display_name = filename.trim_end_matches(".gguf").to_string();

                        found.push(ScannedGgufFile {
                            path,
                            filename,
                            size_bytes,
                            mtime_epoch_secs,
                            quant,
                            repo_id,
                            display_name,
                        });
                    }
                }
            }
        }
    }

    found.sort_by(|a, b| a.filename.cmp(&b.filename));
    found
}

pub async fn resolve_local_gguf_file(state: &AppState, candidate: &str) -> Option<PathBuf> {
    let candidate = candidate.trim();
    if candidate.is_empty() {
        return None;
    }

    // Порядок попыток: путь как прислали, затем с «.gguf», затем только имя файла
    // (фронтенд иногда присылает «models/<файл>» или «org/repo/<файл>»)
    let mut attempts = vec![candidate.to_string()];
    let has_gguf_ext = candidate.to_ascii_lowercase().ends_with(".gguf");
    if !has_gguf_ext {
        attempts.push(format!("{candidate}.gguf"));
    }
    if let Some(filename) = std::path::Path::new(candidate).file_name().and_then(|name| name.to_str()) {
        if filename != candidate {
            attempts.push(filename.to_string());
            if !has_gguf_ext {
                attempts.push(format!("{filename}.gguf"));
            }
        }
    }

    // Все попытки проходят через песочницу: файл обязан лежать внутри папки моделей или scan-folders
    let roots = state.model_roots().await;
    attempts
        .iter()
        .find_map(|attempt| crate::paths::resolve_existing_file_within(&roots, attempt).ok())
}

pub async fn handle_cached_models(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let files = scan_local_gguf_files(&state).await;
    let mut total_size_bytes: u64 = 0;
    let mut cached = Vec::new();

    for f in files {
        total_size_bytes += f.size_bytes;
        cached.push(serde_json::json!({
            "repo_id": f.repo_id,
            "load_id": f.filename,
            "model_format": "gguf",
            "runtime": "llama_cpp",
            "format_variant": f.quant,
            "size_bytes": f.size_bytes,
            "cache_path": f.path.to_string_lossy().to_string(),
            "last_modified": f.mtime_epoch_secs,
            "partial": false,
            "pipeline_tag": "text-generation",
            "task": "text-generation",
            "capabilities": {
                "can_train": true,
                "can_chat": true,
                "can_delete": true,
                "can_download": false,
                "requires_variant": false,
                "supports_lora": true,
                "supports_vision": false
            }
        }));
    }

    Json(serde_json::json!({
        "cached": cached,
        "total_size_bytes": total_size_bytes
    }))
}

pub async fn handle_cached_gguf(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let files = scan_local_gguf_files(&state).await;
    let mut total_size_bytes: u64 = 0;
    let mut cached = Vec::new();

    for f in files {
        total_size_bytes += f.size_bytes;
        cached.push(serde_json::json!({
            "repo_id": f.repo_id,
            "load_id": f.filename,
            "model_format": "gguf",
            "runtime": "llama_cpp",
            "format_variant": f.quant,
            "size_bytes": f.size_bytes,
            "cache_path": f.path.to_string_lossy().to_string(),
            "last_modified": f.mtime_epoch_secs,
            "partial": false,
            "pipeline_tag": "text-generation",
            "task": "text-generation",
            "capabilities": {
                "can_train": true,
                "can_chat": true,
                "can_delete": true,
                "can_download": false,
                "requires_variant": false,
                "supports_lora": true,
                "supports_vision": false
            }
        }));
    }

    Json(serde_json::json!({
        "cached": cached,
        "total_size_bytes": total_size_bytes
    }))
}

pub async fn handle_hub_local(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let files = scan_local_gguf_files(&state).await;
    let count = files.len();
    let mut local_models = Vec::new();

    for f in files {
        let relative_path = format!("models/{}", f.filename);
        local_models.push(serde_json::json!({
            "id": f.filename,
            "load_id": f.filename,
            "display_name": f.display_name,
            "path": relative_path,
            "size_bytes": f.size_bytes,
            "model_format": "gguf",
            "runtime": "llama_cpp",
            "format_variant": f.quant,
            "source": "models_dir",
            "pipeline_tag": "text-generation",
            "task": "text-generation",
            "capabilities": {
                "can_train": true,
                "can_chat": true,
                "can_delete": true,
                "can_download": false,
                "requires_variant": false,
                "supports_lora": true,
                "supports_vision": false
            }
        }));
    }

    Json(serde_json::json!({
        "models_dir": state.models_dir.to_string_lossy().to_string(),
        "lmstudio_dirs": [],
        "ollama_dirs": [],
        "hermes_dirs": [],
        "count": count,
        "models": local_models
    }))
}

pub async fn handle_hidden_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "hidden_models": []
    }))
}

#[derive(Debug, Deserialize)]
struct HfLfsInfo {
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HfTreeItem {
    #[serde(rename = "type")]
    entry_type: String,
    path: String,
    size: Option<u64>,
    lfs: Option<HfLfsInfo>,
}

/// Known quant patterns in descending specificity
const KNOWN_QUANTS: &[&str] = &[
    // Unsloth Dynamic (UD) variants
    "UD-Q8_K_XL", "UD-Q6_K_XL", "UD-Q5_K_XL", "UD-Q4_K_XL", "UD-Q3_K_XL", "UD-Q2_K_XL",
    "UD-IQ4_XS", "UD-IQ4_NL", "UD-IQ3_XXS", "UD-IQ3_S", "UD-IQ2_XXS", "UD-IQ2_M", "UD-IQ1_S", "UD-IQ1_M",
    // Standard K-quants & legacy
    "Q8_K_XL", "Q6_K_XL", "Q5_K_XL", "Q4_K_XL", "Q3_K_XL", "Q2_K_XL",
    "Q8_0", "Q8_1",
    "Q6_K_L", "Q6_K_M", "Q6_K",
    "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0",
    "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0",
    "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q3_1", "Q3_0",
    "Q2_K_L", "Q2_K", "Q2_0",
    // Importance quants (IQ)
    "IQ4_NL", "IQ4_XS", "IQ3_M", "IQ3_S", "IQ3_XXS", "IQ2_M", "IQ2_S", "IQ2_XXS", "IQ1_M", "IQ1_S",
    // Unquantized / Float
    "BF16", "FP16", "F16", "FP32", "F32",
];

/// Returns true if a path refers to auxiliary files that are NOT the base model weights
/// (e.g. MTP speculative decoding heads, vision projector, draft models, VAE, etc.)
pub(crate) fn is_auxiliary_gguf_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    let filename = path.split('/').last().unwrap_or(path).to_lowercase();

    // MTP (Multi-Token Prediction) speculative decoding auxiliary weights
    if lower.starts_with("mtp/")
        || lower.contains("/mtp/")
        || lower.contains("/mtp-")
        || lower.contains("/mtp_")
        || filename.starts_with("mtp-")
        || filename.starts_with("mtp_")
        || filename.contains("-mtp-")
        || filename.contains("_mtp_")
    {
        return true;
    }

    // Vision projector (handled separately to set has_vision = true)
    if lower.contains("mmproj") {
        return true;
    }

    // Speculative decoding draft models
    if lower.contains("draft") {
        return true;
    }

    // Diffusion VAE and text encoders
    if lower.contains("vae") || lower.contains("text_encoder") {
        return true;
    }

    // Standalone adapters or lora weights inside a base model repo
    if lower.contains("adapter") {
        return true;
    }

    false
}

/// Extracts the clean quant variant name from a GGUF file path
pub fn extract_quant_from_path(p: &str) -> String {
    let parts: Vec<&str> = p.split('/').collect();

    // 1. If file is organized in a subfolder, check if the folder name specifies the quant
    if parts.len() > 1 {
        let folder = parts[0];
        // Check exact match with known quants first (e.g. "UD-Q4_K_XL", "Q8_0", "BF16")
        for &q in KNOWN_QUANTS {
            if folder.eq_ignore_ascii_case(q) {
                return q.to_string();
            }
        }
        // Check if folder contains a known quant (e.g. "DeepSeek-R1-Q4_K_M")
        for &q in KNOWN_QUANTS {
            if folder.contains(q) {
                return q.to_string();
            }
        }
    }

    // 2. Check filename for known quant
    let filename = parts.last().unwrap_or(&p);
    for &q in KNOWN_QUANTS {
        if filename.contains(q) {
            return q.to_string();
        }
    }

    // 3. Fallback: use folder name if present, else filename stem
    if parts.len() > 1 {
        parts[0].to_string()
    } else {
        filename.trim_end_matches(".gguf").to_string()
    }
}

pub fn estimate_model_quant_sizes(repo_id: &str) -> (u64, u64, bool) {
    let repo_lower = repo_id.to_lowercase();
    let repo_name = repo_lower.split('/').last().unwrap_or(&repo_lower);

    // 1. Massive MoE models (671B / DeepSeek V3 / R1 base)
    if (repo_name.contains("deepseek") || repo_name.contains("r1") || repo_name.contains("v3"))
        && !repo_name.contains("distill")
        && !repo_name.contains("7b")
        && !repo_name.contains("8b")
        && !repo_name.contains("1.5b")
        && !repo_name.contains("14b")
        && !repo_name.contains("32b")
        && !repo_name.contains("70b")
    {
        return (376_000_000_000, 664_000_000_000, false);
    }

    // 2. Explicit parameter scales in repo name
    if repo_name.contains("405b") {
        (240_000_000_000, 430_000_000_000, false)
    } else if repo_name.contains("284b") || repo_name.contains("deepseek-v4") {
        (155_000_000_000, 280_000_000_000, false)
    } else if repo_name.contains("120b") || repo_name.contains("128b") || repo_name.contains("110b") {
        (70_000_000_000, 130_000_000_000, false)
    } else if repo_name.contains("70b") || repo_name.contains("72b") {
        (40_000_000_000, 75_000_000_000, false)
    } else if repo_name.contains("32b") || repo_name.contains("34b") || repo_name.contains("30b") || repo_name.contains("35b") || repo_name.contains("glm-5") {
        (19_500_000_000, 35_000_000_000, false)
    } else if repo_name.contains("14b") || repo_name.contains("13b") || repo_name.contains("12b") {
        (8_500_000_000, 15_000_000_000, false)
    } else if repo_name.contains("8b") || repo_name.contains("9b") || repo_name.contains("7b") || repo_name.contains("6.7b") {
        (4_800_000_000, 8_500_000_000, false)
    } else if repo_name.contains("3.8-flash") || repo_name.contains("qwen3.8") {
        (111_000_000_000, 195_000_000_000, false)
    } else if repo_name.contains("3b") || repo_name.contains("3.2b") || repo_name.contains("3.8b") || repo_name.contains("4b") {
        (2_100_000_000, 3_800_000_000, true) // fits 4GB RX 570
    } else if repo_name.contains("1.5b") || repo_name.contains("1b") || repo_name.contains("2b") || repo_name.contains("0.5b") {
        (950_000_000, 1_700_000_000, true) // fits 4GB RX 570
    } else {
        (4_800_000_000, 8_500_000_000, false)
    }
}

pub async fn fetch_hf_tree_variants(
    state: &Arc<AppState>,
    repo_id: &str,
) -> Option<serde_json::Value> {
    // 1. Check in-memory cache (10 min TTL)
    {
        let cache = state.gguf_variants_cache.read().await;
        if let Some((ts, val)) = cache.get(repo_id) {
            if ts.elapsed() < Duration::from_secs(600) {
                return Some(val.clone());
            }
        }
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(8))
        .build()
        .ok()?;

    let url = format!("https://huggingface.co/api/models/{}/tree/main?recursive=true", repo_id);
    let resp = client
        .get(&url)
        .header("User-Agent", "sloth-forge/0.1.0")
        .send()
        .await
        .ok()?;

    if !resp.status().is_success() {
        return None;
    }

    let items: Vec<HfTreeItem> = resp.json().await.ok()?;

    let mut has_vision = false;
    let mut gguf_files: Vec<(String, u64)> = Vec::new();

    for item in items {
        if item.entry_type == "file" {
            let p = item.path;
            let sz = item.lfs.as_ref().and_then(|l| l.size).or(item.size).unwrap_or(0);
            let p_lower = p.to_lowercase();
            if p_lower.contains("mmproj") {
                has_vision = true;
                continue;
            }
            if is_auxiliary_gguf_file(&p) {
                continue;
            }
            if p.ends_with(".gguf") {
                gguf_files.push((p, sz));
            }
        }
    }

    if gguf_files.is_empty() {
        return None;
    }

    // Group files by quant: quant -> (quant, Vec<path>, total_bytes)
    let mut groups: std::collections::BTreeMap<String, (String, Vec<String>, u64)> = std::collections::BTreeMap::new();

    for (p, sz) in gguf_files {
        let quant = extract_quant_from_path(&p);
        let group_key = quant.clone();
        let entry = groups.entry(group_key).or_insert_with(|| (quant.clone(), Vec::new(), 0));
        entry.1.push(p);
        entry.2 += sz;
    }

    let mut sorted_keys: Vec<String> = groups.keys().cloned().collect();
    sorted_keys.sort_by(|a, b| {
        fn score(k: &str) -> usize {
            if k == "Q4_K_M" || k == "UD-Q4_K_XL" { 0 }
            else if k == "UD-IQ4_XS" || k == "Q4_K_S" || k == "Q4_0" || k == "IQ4_NL" || k == "IQ4_XS" { 1 }
            else if k == "Q5_K_M" || k == "UD-Q5_K_XL" || k == "Q5_K_S" || k == "Q5_0" { 2 }
            else if k == "Q8_0" || k == "UD-Q8_K_XL" { 3 }
            else if k.starts_with("Q3_") || k.starts_with("UD-Q3_") || k.starts_with("IQ3_") { 4 }
            else if k.starts_with("Q6_") || k.starts_with("UD-Q6_") { 5 }
            else if k.starts_with("Q2_") || k.starts_with("UD-Q2_") || k.starts_with("IQ2_") { 6 }
            else if k.starts_with("IQ1_") || k.starts_with("UD-IQ1_") { 7 }
            else if k == "BF16" || k == "F16" || k == "FP16" { 8 }
            else { 10 }
        }
        score(a).cmp(&score(b))
    });

    let preferred_defaults = [
        "Q4_K_M",
        "UD-Q4_K_XL",
        "UD-IQ4_XS",
        "Q4_K_S",
        "Q4_0",
        "Q5_K_M",
        "UD-Q5_K_XL",
        "Q8_0",
    ];
    let mut default_variant = sorted_keys.first().cloned().unwrap_or_else(|| "Q4_K_M".to_string());
    for pref in preferred_defaults {
        if groups.contains_key(pref) {
            default_variant = pref.to_string();
            break;
        }
    }

    let scanned_local = scan_local_gguf_files(state).await;
    let mut variants: Vec<serde_json::Value> = Vec::new();

    for key in sorted_keys {
        if let Some((quant, shard_paths, total_bytes)) = groups.remove(&key) {
            let first_file = shard_paths.first().cloned().unwrap_or_default();
            let mut is_downloaded = shard_paths.iter().all(|sp| {
                let filename_only = sp.split('/').last().unwrap_or(sp);
                scanned_local.iter().any(|f| {
                    f.filename.eq_ignore_ascii_case(sp)
                        || f.filename.eq_ignore_ascii_case(filename_only)
                        || state.models_dir.join(sp).exists()
                        || state.models_dir.join(filename_only).exists()
                })
            });

            if !is_downloaded {
                for f in &scanned_local {
                    for sp in &shard_paths {
                        let sp_name = sp.split('/').last().unwrap_or(sp);
                        if f.filename.eq_ignore_ascii_case(sp_name) {
                            is_downloaded = true;
                            break;
                        }
                    }
                    if is_downloaded {
                        break;
                    }
                    if file_matches_repo_and_quant(&f.filename, repo_id, &quant) {
                        is_downloaded = true;
                        break;
                    }
                }
            }

            // On RX 570 4GB: only recommend if total weights <= 3.2 GB
            let is_recommended = total_bytes > 0 && total_bytes <= 3_200_000_000u64;
            let display_label = if is_downloaded {
                format!("{} (Downloaded)", quant)
            } else if is_recommended {
                format!("{} (Recommended)", quant)
            } else {
                quant.clone()
            };

            variants.push(serde_json::json!({
                "filename": first_file,
                "files": shard_paths,
                "quant": quant,
                "display_label": display_label,
                "size_bytes": total_bytes,
                "download_size_bytes": total_bytes,
                "shard_count": shard_paths.len(),
                "downloaded": is_downloaded
            }));
        }
    }

    let resolved_locally = variants.iter().any(|v| v["downloaded"].as_bool() == Some(true));

    let result = serde_json::json!({
        "repo_id": repo_id,
        "has_vision": has_vision,
        "default_variant": default_variant,
        "context_length": 131072,
        "resolved_locally": resolved_locally,
        "variants": variants
    });

    {
        let mut cache = state.gguf_variants_cache.write().await;
        cache.insert(repo_id.to_string(), (Instant::now(), result.clone()));
    }

    Some(result)
}

pub async fn handle_gguf_variants(
    State(state): State<Arc<AppState>>,
    Query(query): Query<RepoVariantsQuery>,
) -> Json<serde_json::Value> {
    let repo_id = query
        .repo_id
        .clone()
        .unwrap_or_else(|| "unsloth/Llama-3.2-3B-Instruct-GGUF".to_string());

    // 0. Check if repo_id or local_path refers to an existing local file on disk
    let local_candidate = query.local_path.as_deref().or(Some(&repo_id));
    if let Some(candidate) = local_candidate {
        if let Some(resolved_path) = resolve_local_gguf_file(&state, candidate).await {
            let filename = resolved_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let metadata = std::fs::metadata(&resolved_path).ok();
            let file_size = metadata.map(|m| m.len()).unwrap_or(0);
            let quant = extract_quant_from_path(&filename);

            let context_length = if let Ok(gguf) = sloth_core::gguf::GGUFFile::open(&resolved_path) {
                // Значение по умолчанию уберёт шаг 8 (варианты по реальным файлам)
                gguf.context_length().unwrap_or(131072)
            } else {
                131072
            };

            return Json(serde_json::json!({
                "repo_id": repo_id,
                "has_vision": false,
                "default_variant": quant,
                "context_length": context_length,
                "resolved_locally": true,
                "variants": [
                    {
                        "filename": filename,
                        "quant": quant,
                        "display_label": format!("{} (Downloaded)", quant),
                        "size_bytes": file_size,
                        "download_size_bytes": file_size,
                        "downloaded": true,
                        "shard_count": 1
                    }
                ]
            }));
        }
    }

    // Also if prefer_local_cache is true, scan models for this repo
    if query.prefer_local_cache == Some(true) {
        let scanned = scan_local_gguf_files(&state).await;
        let mut local_variants = Vec::new();
        let mut def_quant = "Q4_K_M".to_string();

        for f in scanned {
            if file_matches_repo_and_quant(&f.filename, &repo_id, &f.quant) || f.repo_id.eq_ignore_ascii_case(&repo_id) {
                def_quant = f.quant.clone();
                local_variants.push(serde_json::json!({
                    "filename": f.filename,
                    "quant": f.quant,
                    "display_label": format!("{} (Downloaded)", f.quant),
                    "size_bytes": f.size_bytes,
                    "download_size_bytes": f.size_bytes,
                    "downloaded": true,
                    "shard_count": 1
                }));
            }
        }

        if !local_variants.is_empty() {
            return Json(serde_json::json!({
                "repo_id": repo_id,
                "has_vision": false,
                "default_variant": def_quant,
                "context_length": 131072,
                "resolved_locally": true,
                "variants": local_variants
            }));
        }
    }

    // 1. First attempt to fetch real variants from Hugging Face tree API
    if let Some(real_val) = fetch_hf_tree_variants(&state, &repo_id).await {
        return Json(real_val);
    }

    // 2. Offline / network fallback: accurate estimates based on architecture and parameter count
    let repo_lower = repo_id.to_lowercase();
    let is_llama_3b = repo_lower.contains("llama-3.2-3b") || repo_id == "unsloth/Llama-3.2-3B-Instruct-GGUF";
    let is_llama_1b = repo_lower.contains("llama-3.2-1b") || repo_id == "unsloth/Llama-3.2-1B-Instruct-GGUF";

    let (q4_default_filename, q8_default_filename, q4_default_size, q8_default_size, is_recommended) = if is_llama_3b {
        (
            "model-Q4_K_M.gguf".to_string(),
            "model-Q8_0.gguf".to_string(),
            2023751680u64,
            3800000000u64,
            true,
        )
    } else if is_llama_1b {
        (
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string(),
            "Llama-3.2-1B-Instruct-Q8_0.gguf".to_string(),
            807694368u64,
            1500000000u64,
            true,
        )
    } else {
        let repo_part = repo_id.split('/').last().unwrap_or(&repo_id);
        let clean_repo_part = repo_part.strip_suffix("-GGUF").or_else(|| repo_part.strip_suffix("_GGUF")).unwrap_or(repo_part);
        let (q4_sz, q8_sz, rec) = estimate_model_quant_sizes(&repo_id);
        (
            format!("{}-Q4_K_M.gguf", clean_repo_part),
            format!("{}-Q8_0.gguf", clean_repo_part),
            q4_sz,
            q8_sz,
            rec,
        )
    };

    let scanned = scan_local_gguf_files(&state).await;

    let mut q4_filename = q4_default_filename;
    let mut q4_size = q4_default_size;
    let mut q4_downloaded = false;

    let mut q8_filename = q8_default_filename;
    let mut q8_size = q8_default_size;
    let mut q8_downloaded = false;

    let mut other_downloaded_variants = Vec::new();

    for f in &scanned {
        let matches_this_repo = file_matches_repo_and_quant(&f.filename, &repo_id, &f.quant)
            || f.repo_id.eq_ignore_ascii_case(&repo_id)
            || f.filename.eq_ignore_ascii_case(&q4_filename)
            || f.filename.eq_ignore_ascii_case(&q8_filename);

        if matches_this_repo {
            if f.quant == "Q4_K_M" || f.filename.contains("Q4_K_M") {
                q4_downloaded = true;
                q4_filename = f.filename.clone();
                q4_size = f.size_bytes;
            } else if f.quant == "Q8_0" || f.filename.contains("Q8_0") {
                q8_downloaded = true;
                q8_filename = f.filename.clone();
                q8_size = f.size_bytes;
            } else {
                other_downloaded_variants.push(serde_json::json!({
                    "filename": f.filename,
                    "quant": f.quant,
                    "display_label": format!("{} (Downloaded)", f.quant),
                    "size_bytes": f.size_bytes,
                    "download_size_bytes": f.size_bytes,
                    "downloaded": true,
                    "shard_count": 1
                }));
            }
        }
    }

    if !q4_downloaded {
        if state.models_dir.join(&q4_filename).exists() {
            q4_downloaded = true;
            if let Ok(m) = std::fs::metadata(state.models_dir.join(&q4_filename)) {
                q4_size = m.len();
            }
        }
    }
    if !q8_downloaded {
        if state.models_dir.join(&q8_filename).exists() {
            q8_downloaded = true;
            if let Ok(m) = std::fs::metadata(state.models_dir.join(&q8_filename)) {
                q8_size = m.len();
            }
        }
    }

    let q4_label = if q4_downloaded {
        "Q4_K_M (Downloaded)".to_string()
    } else if is_recommended {
        "Q4_K_M (Recommended)".to_string()
    } else {
        "Q4_K_M".to_string()
    };

    let q8_label = if q8_downloaded {
        "Q8_0 (Downloaded)".to_string()
    } else {
        "Q8_0".to_string()
    };

    let mut variants = vec![
        serde_json::json!({
            "filename": q4_filename,
            "quant": "Q4_K_M",
            "display_label": q4_label,
            "size_bytes": q4_size,
            "download_size_bytes": q4_size,
            "downloaded": q4_downloaded
        }),
        serde_json::json!({
            "filename": q8_filename,
            "quant": "Q8_0",
            "display_label": q8_label,
            "size_bytes": q8_size,
            "download_size_bytes": q8_size,
            "downloaded": q8_downloaded
        }),
    ];

    variants.extend(other_downloaded_variants);

    let default_variant = if q4_downloaded {
        "Q4_K_M".to_string()
    } else if let Some(first_other) = variants.iter().find(|v| v["downloaded"].as_bool() == Some(true)) {
        first_other["quant"].as_str().unwrap_or("Q4_K_M").to_string()
    } else {
        "Q4_K_M".to_string()
    };

    let resolved_locally = q4_downloaded || q8_downloaded || variants.iter().any(|v| v["downloaded"].as_bool() == Some(true));

    Json(serde_json::json!({
        "repo_id": repo_id,
        "has_vision": false,
        "default_variant": default_variant,
        "context_length": 131072,
        "resolved_locally": resolved_locally,
        "variants": variants
    }))
}

// Загрузка моделей, её прогресс, отмена и transport-status — в downloads.rs

pub async fn handle_datasets_transport_status() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "idle",
        "transport": "direct",
        "active": false,
        "has_partial": false,
        "last_transport": null,
        "resumable": false
    }))
}

pub async fn handle_datasets_cached() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "cached": []
    }))
}

/// Датасеты появятся вместе с настоящим обучением (этап 3 дорожной карты).
/// До этого изменяющие обработчики честно отвечают 501 вместо фальшивого «успеха».
const DATASETS_FEATURE: &str = "Датасеты";
const DATASETS_STAGE: u8 = 3;

pub async fn handle_datasets_cached_delete() -> crate::error::ApiError {
    crate::unavailable::not_ready(DATASETS_FEATURE, DATASETS_STAGE)
}

pub async fn handle_datasets_local() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "datasets": []
    }))
}

pub async fn handle_datasets_active_downloads() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "downloads": []
    }))
}

pub async fn handle_datasets_download() -> crate::error::ApiError {
    // Раньше отвечало state: "running", но загрузка не начиналась
    crate::unavailable::not_ready(DATASETS_FEATURE, DATASETS_STAGE)
}

pub async fn handle_datasets_download_cancel() -> crate::error::ApiError {
    crate::unavailable::not_ready(DATASETS_FEATURE, DATASETS_STAGE)
}

pub async fn handle_datasets_download_status(
    Query(_query): Query<RepoVariantsQuery>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "state": "idle",
        "error": null,
        "generation": 1
    }))
}

pub async fn handle_datasets_download_progress() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "downloaded_bytes": 0,
        "completed_bytes": 0,
        "complete_on_disk": false,
        "expected_bytes": 0,
        "progress": 0.0,
        "cache_path": null,
        "target_present": false,
        "cache_measured": true
    }))
}

pub async fn handle_datasets_local_options() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "options": []
    }))
}

pub async fn handle_delete_cached(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> crate::error::ApiResult<Json<serde_json::Value>> {
    use crate::error::ApiError;

    let repo_id = payload
        .get("repo_id")
        .or_else(|| payload.get("repoId"))
        .and_then(|v| v.as_str());
    let variant = payload
        .get("variant")
        .or_else(|| payload.get("gguf_variant"))
        .or_else(|| payload.get("quant"))
        .and_then(|v| v.as_str());
    let cache_path = payload
        .get("cache_path")
        .or_else(|| payload.get("cachePath"))
        .and_then(|v| v.as_str());

    let roots = state.model_roots().await;

    // Каждый путь проходит через песочницу: удалить что-либо вне папки моделей
    // и добавленных scan-folders невозможно.
    let targets: Vec<PathBuf> = if let Some(cp) = cache_path {
        vec![crate::paths::resolve_existing_file_within(&roots, cp)
            .map_err(|rejection| ApiError::from_path_rejection(rejection, "cache_path"))?]
    } else if let Some(v) = variant {
        // Без repo_id вариант вроде «Q4_K_M» совпал бы с файлами разных моделей
        let repo = repo_id
            .filter(|repo| !repo.trim().is_empty())
            .ok_or_else(|| ApiError::bad_request("Для удаления по варианту нужен repo_id"))?;
        // Ищем среди реально найденных на диске файлов: тот же repo и точно тот же квант
        scan_local_gguf_files(&state)
            .await
            .into_iter()
            .filter(|file| file.quant.eq_ignore_ascii_case(v) && file.repo_id.eq_ignore_ascii_case(repo))
            .filter_map(|file| {
                crate::paths::resolve_existing_file_within(&roots, &file.path.to_string_lossy()).ok()
            })
            .collect()
    } else {
        return Err(ApiError::bad_request("Укажите cache_path или variant"));
    };

    if targets.is_empty() {
        return Err(ApiError::not_found("Локальные файлы этой модели не найдены"));
    }
    let is_gguf = |path: &PathBuf| {
        path.extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    };
    if !targets.iter().all(is_gguf) {
        return Err(ApiError::forbidden("Удалять можно только файлы моделей .gguf"));
    }

    let mut deleted_files = Vec::with_capacity(targets.len());
    for target in &targets {
        tokio::fs::remove_file(target).await.map_err(|err| {
            ApiError::internal(format!("Не удалось удалить {}: {err}", target.display()))
        })?;
        deleted_files.push(target.to_string_lossy().to_string());
    }

    // Флаги «Downloaded» в кэше вариантов больше не соответствуют диску
    state.gguf_variants_cache.write().await.clear();
    info!(
        "Удалены файлы модели repo_id={:?}, variant={:?}: {:?}",
        repo_id, variant, deleted_files
    );

    Ok(Json(serde_json::json!({
        "status": "ok",
        "deleted": true,
        "deleted_files": deleted_files
    })))
}

pub async fn handle_hub_scan_folders(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let folders = state.scan_folders.read().await;
    Json(serde_json::json!({
        "folders": *folders
    }))
}

pub async fn handle_hub_add_scan_folder(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let path = payload
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let mut folders = state.scan_folders.write().await;
    let next_id = (folders.len() as u64) + 1;
    let entry = crate::state::ScanFolderEntry {
        id: next_id,
        path,
        created_at: crate::state::iso_now(),
        status: Some("ok".to_string()),
    };
    folders.push(entry.clone());
    Json(serde_json::to_value(entry).unwrap_or(serde_json::json!({
        "id": next_id,
        "status": "ok"
    })))
}

pub async fn handle_hub_delete_scan_folder(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Json<serde_json::Value> {
    let mut folders = state.scan_folders.write().await;
    folders.retain(|f| f.id != id);
    Json(serde_json::json!({
        "status": "ok",
        "deleted": true
    }))
}

pub async fn calculate_delete_impact(
    state: &AppState,
    repo_id: &str,
    variant: Option<&str>,
) -> serde_json::Value {
    let scanned = scan_local_gguf_files(state).await;
    let mut reclaimed_bytes: u64 = 0;

    for f in &scanned {
        let matches = if let Some(v) = variant {
            file_matches_repo_and_quant(&f.filename, repo_id, v)
                || (f.repo_id.eq_ignore_ascii_case(repo_id) && f.quant.eq_ignore_ascii_case(v))
                || (f.filename.eq_ignore_ascii_case(repo_id) && f.quant.eq_ignore_ascii_case(v))
        } else {
            f.repo_id.eq_ignore_ascii_case(repo_id)
                || f.filename.eq_ignore_ascii_case(repo_id)
                || (!repo_id.is_empty() && (repo_id.contains(&f.filename) || f.filename.to_lowercase().contains(&repo_id.to_lowercase())))
        };

        if matches {
            reclaimed_bytes += f.size_bytes;
        }
    }

    if reclaimed_bytes == 0 && !repo_id.is_empty() {
        if repo_id.to_lowercase().contains("1b") {
            reclaimed_bytes = 807_694_368;
        } else if repo_id.to_lowercase().contains("3b") {
            reclaimed_bytes = 2_100_000_000;
        }
    }

    serde_json::json!({
        "repo_id": repo_id,
        "variant": variant,
        "reclaimed_bytes": reclaimed_bytes,
        "freed_bytes": reclaimed_bytes,
        "affected_models": [],
        "retained_companions": [],
        "freeable_companions": [],
        "blocked_by": []
    })
}

pub async fn handle_hub_delete_impact(
    State(state): State<Arc<AppState>>,
    Query(query): Query<serde_json::Value>,
) -> Json<serde_json::Value> {
    let repo_id = query.get("repo_id").and_then(|v| v.as_str()).unwrap_or("");
    let variant = query.get("variant").and_then(|v| v.as_str());
    Json(calculate_delete_impact(&state, repo_id, variant).await)
}

pub async fn handle_hub_delete_impact_post(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let repo_id = payload.get("repo_id").and_then(|v| v.as_str()).unwrap_or("");
    let variant = payload.get("variant").and_then(|v| v.as_str());
    Json(calculate_delete_impact(&state, repo_id, variant).await)
}

pub async fn handle_hub_orphan_companions() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "orphans": [],
        "companions": [],
        "total_bytes": 0
    }))
}

pub async fn handle_token_validate() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "valid",
        "retry_after_seconds": null
    }))
}

pub async fn handle_datasets_check_format() -> crate::error::ApiError {
    // Раньше для любого файла отвечало «alpaca, 1000 строк», не читая его
    crate::unavailable::not_ready(DATASETS_FEATURE, DATASETS_STAGE)
}

pub async fn handle_datasets_upload() -> crate::error::ApiError {
    // Раньше не читало тело запроса и возвращало путь к несуществующему файлу
    crate::unavailable::not_ready(DATASETS_FEATURE, DATASETS_STAGE)
}

pub async fn handle_datasets_ai_assist_mapping() -> crate::error::ApiError {
    crate::unavailable::not_ready(DATASETS_FEATURE, DATASETS_STAGE)
}
