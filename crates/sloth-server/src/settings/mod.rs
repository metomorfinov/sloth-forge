//! Настройки `/api/settings/*` с постоянным хранением.
//!
//! Раньше большинство этих эндпоинтов возвращали присланное значение обратно и ничего
//! не запоминали, а часть отдавала данные не в том формате, который ждёт фронтенд
//! (например, `{ fraction: 0.9 }` вместо настроек памяти моделей). Теперь значения
//! хранятся в таблице настроек SQLite, а форматы совпадают с `frontend/src/features/settings/api`.
//!
//! - [`ui`] — профиль и оформление, предпочтения чата, переключатели, лимит загрузки, пресеты;
//! - [`runtime`] — всё, что связано с моделями, загрузками и Hugging Face.

pub mod runtime;
pub mod ui;

use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use crate::store::settings as store_settings;
use serde_json::Value;

/// Читает значение настройки.
pub(crate) async fn read_value(
    state: &AppState,
    key: impl Into<String>,
) -> ApiResult<Option<Value>> {
    let key = key.into();
    Ok(state
        .store
        .call(move |conn| store_settings::get(conn, &key))
        .await?)
}

/// Записывает значение настройки.
pub(crate) async fn write_value(
    state: &AppState,
    key: impl Into<String>,
    value: Value,
) -> ApiResult<()> {
    let key = key.into();
    Ok(state
        .store
        .call(move |conn| store_settings::put(conn, &key, &value))
        .await?)
}

/// Удаляет значение настройки (возврат к значению по умолчанию).
pub(crate) async fn delete_value(state: &AppState, key: impl Into<String>) -> ApiResult<()> {
    let key = key.into();
    state
        .store
        .call(move |conn| store_settings::delete(conn, &key))
        .await?;
    Ok(())
}

/// Булева настройка или значение по умолчанию.
pub(crate) async fn read_bool(
    state: &AppState,
    key: impl Into<String>,
    default: bool,
) -> ApiResult<bool> {
    Ok(read_value(state, key)
        .await?
        .and_then(|value| value.as_bool())
        .unwrap_or(default))
}

/// Обязательное булево поле тела запроса.
pub(crate) fn require_bool(body: &Value, field: &str) -> ApiResult<bool> {
    body.get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| ApiError::unprocessable(format!("Поле {field} должно быть true или false")))
}

/// Необязательное булево поле: отсутствует или `null` — `None`, другой тип — ошибка.
pub(crate) fn optional_bool(body: &Value, field: &str) -> ApiResult<Option<bool>> {
    match body.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(ApiError::unprocessable(format!(
            "Поле {field} должно быть true или false"
        ))),
    }
}

/// Необязательное неотрицательное целое поле.
pub(crate) fn optional_u64(body: &Value, field: &str) -> ApiResult<Option<u64>> {
    match body.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value.as_u64().map(Some).ok_or_else(|| {
            ApiError::unprocessable(format!(
                "Поле {field} должно быть неотрицательным целым числом"
            ))
        }),
    }
}
