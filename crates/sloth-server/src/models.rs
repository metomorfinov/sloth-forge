use serde::{Deserialize, Serialize};
use sloth_core::gguf::GGUFFile;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCard {
    pub id: String,
    pub name: String,
    pub filename: String,
    pub path: String,
    pub size_bytes: u64,
    pub size_formatted: String,
    pub parameters: String,
    pub quantization: String,
    pub context_length: u64,
    pub architecture: String,
    pub is_present: bool,
    pub recommended_vram_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub models: Vec<ModelCard>,
    pub data: Vec<OpenAIModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{bytes} B")
    }
}

pub fn discover_models(models_dir: &Path) -> Vec<ModelCard> {
    let mut discovered = Vec::new();
    let mut found_filenames = std::collections::HashSet::new();

    // 1. Scan filesystem directory if it exists
    if models_dir.exists() {
        if let Ok(entries) = fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                    if filename.ends_with(".gguf") {
                        found_filenames.insert(filename.clone());
                        let metadata = entry.metadata().ok();
                        let size_bytes = metadata.map(|m| m.len()).unwrap_or(0);

                        // Try to parse GGUF headers
                        if let Ok(gguf) = GGUFFile::open(&path) {
                            let arch = gguf.architecture().unwrap_or("llama").to_string();
                            let ctx_len = gguf.context_length();
                            let id = filename.trim_end_matches(".gguf").to_lowercase();
                            let name = filename.trim_end_matches(".gguf").to_string();

                            discovered.push(ModelCard {
                                id: id.clone(),
                                name,
                                filename: filename.clone(),
                                path: path.to_string_lossy().to_string(),
                                size_bytes,
                                size_formatted: format_bytes(size_bytes),
                                parameters: if size_bytes > 3_000_000_000 {
                                    "7B".to_string()
                                } else if size_bytes > 1_500_000_000 {
                                    "3.2B".to_string()
                                } else {
                                    "1.5B".to_string()
                                },
                                quantization: if filename.contains("Q4_K") {
                                    "Q4_K_M".to_string()
                                } else if filename.contains("Q8_0") {
                                    "Q8_0".to_string()
                                } else {
                                    "Q4_0".to_string()
                                },
                                context_length: ctx_len,
                                architecture: arch,
                                is_present: true,
                                recommended_vram_mb: if size_bytes > 3_000_000_000 {
                                    3800
                                } else {
                                    2600
                                },
                            });
                        } else {
                            // Fallback file info if GGUF parse failed
                            let id = filename.trim_end_matches(".gguf").to_lowercase();
                            discovered.push(ModelCard {
                                id: id.clone(),
                                name: filename.trim_end_matches(".gguf").to_string(),
                                filename: filename.clone(),
                                path: path.to_string_lossy().to_string(),
                                size_bytes,
                                size_formatted: format_bytes(size_bytes),
                                parameters: "Unknown".to_string(),
                                quantization: "Q4_K_M".to_string(),
                                context_length: 8192,
                                architecture: "llama".to_string(),
                                is_present: true,
                                recommended_vram_mb: 2600,
                            });
                        }
                    }
                }
            }
        }
    }

    // 2. Add standard/recommended SlothForge GGUF models if not already found on disk
    let catalog = [
        (
            "llama-3.2-3b-instruct-q4_k_m",
            "Llama-3.2-3B-Instruct",
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            2_023_751_680u64,
            "3.21B",
            "Q4_K_M",
            131072u64,
            "llama",
            2600u64,
        ),
        (
            "llama-3.2-1b-instruct-q4_k_m",
            "Llama-3.2-1B-Instruct",
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            805_306_368u64,
            "1.24B",
            "Q4_K_M",
            131072u64,
            "llama",
            1400u64,
        ),
        (
            "qwen2.5-1.5b-instruct-q4_k_m",
            "Qwen2.5-1.5B-Instruct",
            "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
            985_661_440u64,
            "1.54B",
            "Q4_K_M",
            32768u64,
            "qwen2",
            1600u64,
        ),
        (
            "mistral-7b-instruct-v0.3-q4_k_m",
            "Mistral-7B-Instruct-v0.3",
            "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            4_368_497_664u64,
            "7.24B",
            "Q4_K_M",
            32768u64,
            "mistral",
            3800u64,
        ),
    ];

    for (id, name, filename, size, params, quant, ctx, arch, vram) in catalog {
        if !found_filenames.contains(filename) {
            let model_path = models_dir.join(filename);
            let is_present = model_path.exists();
            discovered.push(ModelCard {
                id: id.to_string(),
                name: name.to_string(),
                filename: filename.to_string(),
                path: model_path.to_string_lossy().to_string(),
                size_bytes: size,
                size_formatted: format_bytes(size),
                parameters: params.to_string(),
                quantization: quant.to_string(),
                context_length: ctx,
                architecture: arch.to_string(),
                is_present,
                recommended_vram_mb: vram,
            });
        }
    }

    discovered
}
