use serde::{Deserialize, Serialize};
use sloth_core::gguf::GGUFFile;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    pub id: String,
    pub name: String,
    pub display_name: String,
    #[serde(rename = "displayName")]
    pub display_name_camel: String,
    pub filename: String,
    pub path: String,
    pub size_bytes: u64,
    #[serde(rename = "sizeBytes")]
    pub size_bytes_camel: u64,
    pub size_formatted: String,
    #[serde(rename = "sizeFormatted")]
    pub size_formatted_camel: String,
    pub device_size: String,
    #[serde(rename = "deviceSize")]
    pub device_size_camel: String,
    pub device_size_bytes: u64,
    #[serde(rename = "deviceSizeBytes")]
    pub device_size_bytes_camel: u64,
    pub parameters: String,
    pub quantization: String,
    pub device_quant: String,
    #[serde(rename = "deviceQuant")]
    pub device_quant_camel: String,
    pub context_length: u64,
    #[serde(rename = "contextLength")]
    pub context_length_camel: u64,
    pub architecture: String,
    pub is_present: bool,
    #[serde(rename = "isPresent")]
    pub is_present_camel: bool,
    pub is_gguf: bool,
    #[serde(rename = "isGguf")]
    pub is_gguf_camel: bool,
    pub is_vision: bool,
    #[serde(rename = "isVision")]
    pub is_vision_camel: bool,
    pub is_lora: bool,
    #[serde(rename = "isLora")]
    pub is_lora_camel: bool,
    pub is_audio: bool,
    #[serde(rename = "isAudio")]
    pub is_audio_camel: bool,
    pub source: String,
    pub recommended_vram_mb: u64,
    #[serde(rename = "recommendedVramMb")]
    pub recommended_vram_mb_camel: u64,
}

impl ModelCard {
    pub fn new(
        id: String,
        name: String,
        filename: String,
        path: String,
        size_bytes: u64,
        parameters: String,
        quantization: String,
        context_length: u64,
        architecture: String,
        is_present: bool,
        recommended_vram_mb: u64,
    ) -> Self {
        let size_formatted = format_bytes(size_bytes);
        Self {
            id: id.clone(),
            name: name.clone(),
            display_name: name.clone(),
            display_name_camel: name,
            filename,
            path,
            size_bytes,
            size_bytes_camel: size_bytes,
            size_formatted: size_formatted.clone(),
            size_formatted_camel: size_formatted.clone(),
            device_size: size_formatted.clone(),
            device_size_camel: size_formatted,
            device_size_bytes: size_bytes,
            device_size_bytes_camel: size_bytes,
            parameters,
            quantization: quantization.clone(),
            device_quant: quantization.clone(),
            device_quant_camel: quantization,
            context_length,
            context_length_camel: context_length,
            architecture,
            is_present,
            is_present_camel: is_present,
            is_gguf: true,
            is_gguf_camel: true,
            is_vision: false,
            is_vision_camel: false,
            is_lora: false,
            is_lora_camel: false,
            is_audio: false,
            is_audio_camel: false,
            source: "models_dir".to_string(),
            recommended_vram_mb,
            recommended_vram_mb_camel: recommended_vram_mb,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub models: Vec<ModelCard>,
    pub default_models: Vec<String>,
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

                            discovered.push(ModelCard::new(
                                id,
                                name,
                                filename.clone(),
                                path.to_string_lossy().to_string(),
                                size_bytes,
                                if size_bytes > 3_000_000_000 {
                                    "7B".to_string()
                                } else if size_bytes > 1_500_000_000 {
                                    "3.2B".to_string()
                                } else {
                                    "1.5B".to_string()
                                },
                                if filename.contains("Q4_K") {
                                    "Q4_K_M".to_string()
                                } else if filename.contains("Q8_0") {
                                    "Q8_0".to_string()
                                } else {
                                    "Q4_0".to_string()
                                },
                                ctx_len,
                                arch,
                                true,
                                if size_bytes > 3_000_000_000 {
                                    3800
                                } else {
                                    2600
                                },
                            ));
                        } else {
                            // Fallback file info if GGUF parse failed
                            let id = filename.trim_end_matches(".gguf").to_lowercase();
                            discovered.push(ModelCard::new(
                                id,
                                filename.trim_end_matches(".gguf").to_string(),
                                filename.clone(),
                                path.to_string_lossy().to_string(),
                                size_bytes,
                                "Unknown".to_string(),
                                "Q4_K_M".to_string(),
                                8192,
                                "llama".to_string(),
                                true,
                                2600,
                            ));
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
            discovered.push(ModelCard::new(
                id.to_string(),
                name.to_string(),
                filename.to_string(),
                model_path.to_string_lossy().to_string(),
                size,
                params.to_string(),
                quant.to_string(),
                ctx,
                arch.to_string(),
                is_present,
                vram,
            ));
        }
    }

    discovered
}
