//! Настройки запуска обучения из интерфейса и их проверка.
//!
//! Раньше из запроса читались придуманные поля (`preset`, `intensity`), а числа не
//! проверялись: `{"lora_r": 1000000}` выделял десятки гигабайт, строка вместо скорости
//! обучения молча превращалась в 0.0002. Здесь читаются поля `TrainingStartRequest`
//! фронтенда, а всё, что выходит за разумные пределы, отклоняется с понятным текстом.

use crate::error::{ApiError, ApiResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MAX_LORA_RANK: u32 = 1024;
pub const MAX_EPOCHS: u32 = 1000;
pub const MAX_TOTAL_STEPS: u32 = 1_000_000;
pub const MAX_BATCH_SIZE: u32 = 4096;
pub const MAX_GRADIENT_ACCUMULATION: u32 = 4096;
/// Длина последовательности: до 1 млн токенов, больше не бывает даже у длинноконтекстных моделей.
pub const MAX_SEQ_LENGTH: u32 = 1 << 20;
/// Скорость обучения больше 1 не имеет смысла ни для одного оптимизатора.
const MAX_LEARNING_RATE: f64 = 1.0;

// Значения по умолчанию — как в примерах Unsloth (интерфейс обычно присылает все поля сам)
const DEFAULT_EPOCHS: u32 = 3;
const DEFAULT_LEARNING_RATE: f32 = 2e-4;
const DEFAULT_BATCH_SIZE: u32 = 2;
const DEFAULT_GRADIENT_ACCUMULATION: u32 = 4;
const DEFAULT_SEQ_LENGTH: u32 = 2048;
const DEFAULT_WEIGHT_DECAY: f32 = 0.01;
const DEFAULT_LORA_RANK: u32 = 16;
const DEFAULT_LORA_ALPHA: f32 = 16.0;
const DEFAULT_SEED: u64 = 3407;
const DEFAULT_SCHEDULER: &str = "linear";
const DEFAULT_OPTIMIZER: &str = "adamw_8bit";
const DEFAULT_TRAINING_TYPE: &str = "lora";
const DEFAULT_TARGET_MODULES: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

/// Тело `POST /api/train/start` (`TrainingStartRequest` во фронтенде). Поля, которые
/// движку пока не нужны (W&B, S3, vision), просто не читаются.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TrainStartRequest {
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub model_local_path: Option<String>,
    #[serde(default)]
    pub project_name: Option<String>,
    #[serde(default)]
    pub training_type: Option<String>,
    #[serde(default)]
    pub hf_dataset: Option<String>,
    #[serde(default)]
    pub local_datasets: Option<Vec<String>>,
    #[serde(default)]
    pub num_epochs: Option<u32>,
    #[serde(default)]
    pub max_steps: Option<u32>,
    #[serde(default)]
    pub learning_rate: Option<Value>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub gradient_accumulation_steps: Option<u32>,
    #[serde(default)]
    pub max_seq_length: Option<u32>,
    #[serde(default)]
    pub warmup_steps: Option<u32>,
    #[serde(default)]
    pub warmup_ratio: Option<f32>,
    #[serde(default)]
    pub weight_decay: Option<f32>,
    #[serde(default)]
    pub lr_scheduler_type: Option<String>,
    #[serde(default)]
    pub optim: Option<String>,
    #[serde(default)]
    pub random_seed: Option<u64>,
    #[serde(default)]
    pub use_lora: Option<bool>,
    #[serde(default)]
    pub lora_r: Option<u32>,
    #[serde(default)]
    pub lora_alpha: Option<f32>,
    #[serde(default)]
    pub lora_dropout: Option<f32>,
    #[serde(default)]
    pub target_modules: Option<Vec<String>>,
    #[serde(default)]
    pub load_in_4bit: Option<bool>,
    #[serde(default)]
    pub start_request_id: Option<String>,
}

/// Проверенные настройки, с которыми работает движок.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrainingConfig {
    /// Модель: локальный путь или репозиторий Hugging Face.
    pub model: String,
    /// Датасет для истории запусков: репозиторий HF или список локальных файлов.
    pub dataset: String,
    pub local_datasets: Vec<String>,
    pub project_name: Option<String>,
    pub training_type: String,
    pub epochs: u32,
    /// Ограничение числа шагов; `None` — обучать все эпохи.
    pub max_steps: Option<u32>,
    pub learning_rate: f32,
    pub batch_size: u32,
    pub gradient_accumulation_steps: u32,
    pub max_seq_length: u32,
    pub warmup_steps: Option<u32>,
    pub warmup_ratio: Option<f32>,
    pub weight_decay: f32,
    pub lr_scheduler: String,
    pub optimizer: String,
    pub seed: u64,
    pub use_lora: bool,
    pub lora_rank: u32,
    pub lora_alpha: f32,
    pub lora_dropout: f32,
    pub target_modules: Vec<String>,
    pub load_in_4bit: bool,
}

fn invalid(message: impl Into<String>) -> ApiError {
    ApiError::unprocessable(message)
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn within(field: &str, value: u32, min: u32, max: u32) -> ApiResult<u32> {
    if (min..=max).contains(&value) {
        Ok(value)
    } else {
        Err(invalid(format!(
            "Поле {field} должно быть от {min} до {max}, получено {value}"
        )))
    }
}

fn fraction(field: &str, value: f32, upper_inclusive: bool) -> ApiResult<f32> {
    let upper_ok = if upper_inclusive {
        value <= 1.0
    } else {
        value < 1.0
    };
    if value.is_finite() && value >= 0.0 && upper_ok {
        Ok(value)
    } else {
        Err(invalid(format!(
            "Поле {field} должно быть долей от 0 до 1, получено {value}"
        )))
    }
}

fn positive(field: &str, value: f32) -> ApiResult<f32> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(invalid(format!(
            "Поле {field} должно быть больше 0, получено {value}"
        )))
    }
}

fn non_negative(field: &str, value: f32) -> ApiResult<f32> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(invalid(format!(
            "Поле {field} не может быть отрицательным, получено {value}"
        )))
    }
}

/// Скорость обучения: интерфейс присылает строку («2e-4»), API-клиенты — число.
fn learning_rate(value: Option<&Value>) -> ApiResult<f32> {
    let parsed = match value {
        None | Some(Value::Null) => return Ok(DEFAULT_LEARNING_RATE),
        Some(Value::Number(number)) => number.as_f64(),
        Some(Value::String(text)) => text.trim().parse::<f64>().ok(),
        Some(_) => None,
    };
    match parsed {
        Some(rate) if rate.is_finite() && rate > 0.0 && rate <= MAX_LEARNING_RATE => Ok(rate as f32),
        _ => Err(invalid(format!(
            "Скорость обучения learning_rate должна быть числом больше 0 и не больше {MAX_LEARNING_RATE}"
        ))),
    }
}

impl TrainingConfig {
    pub fn from_request(request: &TrainStartRequest) -> ApiResult<Self> {
        let model = non_empty(request.model_local_path.as_deref())
            .or(non_empty(request.model_name.as_deref()))
            .ok_or_else(|| invalid("Не выбрана модель для обучения"))?
            .to_string();
        let local_datasets: Vec<String> = request
            .local_datasets
            .iter()
            .flatten()
            .map(|dataset| dataset.trim().to_string())
            .filter(|dataset| !dataset.is_empty())
            .collect();
        let dataset = match non_empty(request.hf_dataset.as_deref()) {
            Some(dataset) => dataset.to_string(),
            None if !local_datasets.is_empty() => local_datasets.join(", "),
            None => return Err(invalid("Не выбран датасет для обучения")),
        };
        let max_steps = match request.max_steps {
            // 0 в интерфейсе означает «без ограничения, по числу эпох»
            None | Some(0) => None,
            Some(steps) => Some(within("max_steps", steps, 1, MAX_TOTAL_STEPS)?),
        };
        let target_modules: Vec<String> = request
            .target_modules
            .iter()
            .flatten()
            .map(|module| module.trim().to_string())
            .filter(|module| !module.is_empty())
            .collect();

        Ok(Self {
            model,
            dataset,
            local_datasets,
            project_name: non_empty(request.project_name.as_deref()).map(str::to_string),
            training_type: non_empty(request.training_type.as_deref())
                .unwrap_or(DEFAULT_TRAINING_TYPE)
                .to_string(),
            epochs: within(
                "num_epochs",
                request.num_epochs.unwrap_or(DEFAULT_EPOCHS),
                1,
                MAX_EPOCHS,
            )?,
            max_steps,
            learning_rate: learning_rate(request.learning_rate.as_ref())?,
            batch_size: within(
                "batch_size",
                request.batch_size.unwrap_or(DEFAULT_BATCH_SIZE),
                1,
                MAX_BATCH_SIZE,
            )?,
            gradient_accumulation_steps: within(
                "gradient_accumulation_steps",
                request
                    .gradient_accumulation_steps
                    .unwrap_or(DEFAULT_GRADIENT_ACCUMULATION),
                1,
                MAX_GRADIENT_ACCUMULATION,
            )?,
            max_seq_length: within(
                "max_seq_length",
                request.max_seq_length.unwrap_or(DEFAULT_SEQ_LENGTH),
                1,
                MAX_SEQ_LENGTH,
            )?,
            warmup_steps: request.warmup_steps,
            warmup_ratio: request
                .warmup_ratio
                .map(|ratio| fraction("warmup_ratio", ratio, true))
                .transpose()?,
            weight_decay: non_negative(
                "weight_decay",
                request.weight_decay.unwrap_or(DEFAULT_WEIGHT_DECAY),
            )?,
            lr_scheduler: non_empty(request.lr_scheduler_type.as_deref())
                .unwrap_or(DEFAULT_SCHEDULER)
                .to_string(),
            optimizer: non_empty(request.optim.as_deref())
                .unwrap_or(DEFAULT_OPTIMIZER)
                .to_string(),
            seed: request.random_seed.unwrap_or(DEFAULT_SEED),
            use_lora: request.use_lora.unwrap_or(true),
            lora_rank: within(
                "lora_r",
                request.lora_r.unwrap_or(DEFAULT_LORA_RANK),
                1,
                MAX_LORA_RANK,
            )?,
            lora_alpha: positive(
                "lora_alpha",
                request.lora_alpha.unwrap_or(DEFAULT_LORA_ALPHA),
            )?,
            lora_dropout: fraction("lora_dropout", request.lora_dropout.unwrap_or(0.0), false)?,
            target_modules: if target_modules.is_empty() {
                DEFAULT_TARGET_MODULES
                    .iter()
                    .map(|m| m.to_string())
                    .collect()
            } else {
                target_modules
            },
            load_in_4bit: request.load_in_4bit.unwrap_or(true),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(body: Value) -> ApiResult<TrainingConfig> {
        let request: TrainStartRequest = serde_json::from_value(body).expect("тело разбирается");
        TrainingConfig::from_request(&request)
    }

    fn frontend_body() -> Value {
        json!({
            "model_name": "unsloth/Llama-3.2-1B-Instruct",
            "hf_dataset": "yahma/alpaca-cleaned",
            "local_datasets": [],
            "project_name": null,
            "training_type": "lora",
            "num_epochs": 1,
            "learning_rate": "2e-4",
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_steps": 0,
            "max_seq_length": 2048,
            "warmup_steps": 5,
            "warmup_ratio": null,
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "optim": "adamw_8bit",
            "random_seed": 3407,
            "use_lora": true,
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0,
            "target_modules": ["q_proj", "v_proj"],
            "load_in_4bit": true
        })
    }

    #[test]
    fn accepts_frontend_request() {
        let config = parse(frontend_body()).expect("настройки корректны");
        assert_eq!(config.model, "unsloth/Llama-3.2-1B-Instruct");
        assert_eq!(config.dataset, "yahma/alpaca-cleaned");
        assert_eq!(config.learning_rate, 2e-4);
        assert_eq!(config.max_steps, None, "0 шагов — без ограничения");
        assert_eq!(config.target_modules, ["q_proj", "v_proj"]);
    }

    #[test]
    fn local_path_and_local_datasets_are_used() {
        let mut body = frontend_body();
        body["model_local_path"] = json!("/models/llama.gguf");
        body["hf_dataset"] = json!(null);
        body["local_datasets"] = json!(["a.jsonl", "b.jsonl"]);
        let config = parse(body).unwrap();
        assert_eq!(config.model, "/models/llama.gguf");
        assert_eq!(config.dataset, "a.jsonl, b.jsonl");
    }

    #[test]
    fn rejects_missing_model_or_dataset() {
        let mut body = frontend_body();
        body["model_name"] = json!("  ");
        assert!(parse(body).is_err());

        let mut body = frontend_body();
        body["hf_dataset"] = json!(null);
        assert!(parse(body).is_err());
    }

    #[test]
    fn rejects_out_of_range_numbers() {
        for (field, value) in [
            ("lora_r", json!(0)),
            ("lora_r", json!(MAX_LORA_RANK + 1)),
            ("num_epochs", json!(0)),
            ("max_steps", json!(MAX_TOTAL_STEPS + 1)),
            ("batch_size", json!(0)),
            ("learning_rate", json!("abc")),
            ("learning_rate", json!(0)),
            ("learning_rate", json!(5)),
            ("lora_alpha", json!(-1)),
            ("lora_dropout", json!(1)),
            ("weight_decay", json!(-0.1)),
        ] {
            let mut body = frontend_body();
            body[field] = value.clone();
            assert!(parse(body).is_err(), "{field} = {value} должен отклоняться");
        }
    }

    #[test]
    fn numeric_learning_rate_is_accepted() {
        let mut body = frontend_body();
        body["learning_rate"] = json!(0.0001);
        assert_eq!(parse(body).unwrap().learning_rate, 0.0001);
    }
}
