//! Обучение: API-слой и жизненный цикл запусков.
//!
//! Раньше «обучение» было имитацией: цикл рисовал loss формулой `0.45 + 2.35·e^(−3.4·t)`,
//! скорость была константой 2850 токенов/с, температура GPU — остатком от деления номера
//! шага, а `job_id` у всех запусков совпадал (`Instant::now().elapsed()` ≈ 0).
//! Теперь:
//! - [`request`] проверяет настройки запуска из интерфейса;
//! - [`controller`] ведёт настоящий жизненный цикл: уникальные id, остановка с ожиданием
//!   движка, история метрик, события прогресса, сохранение запусков в SQLite;
//! - [`handlers`] — эндпоинты `/api/train/*` по контракту фронтенда.
//!
//! Сам движок (LoRA на Vulkan) появится на этапе 3 дорожной карты и подключится через
//! [`TrainingEngine`]. До этого старт честно отвечает 503.

pub mod controller;
pub mod handlers;
pub mod request;

pub use controller::{
    EngineFuture, ProgressReporter, ProgressUpdate, StopSignal, TrainingController, TrainingEngine,
    TrainingPhase,
};
pub use request::{TrainStartRequest, TrainingConfig};
