//! Единый формат ошибок API: `{"detail": "текст"}`.
//!
//! Фронтенд (оригинальный интерфейс Unsloth Studio) показывает пользователю именно
//! поле `detail`. Если его нет, в уведомлении остаётся только «Request failed (код)».

use crate::paths::PathRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

/// Ошибка обработчика: HTTP-код и понятное человеку описание.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    pub status: StatusCode,
    pub detail: String,
}

/// Результат обработчика, который может вернуть [`ApiError`].
pub type ApiResult<T> = Result<T, ApiError>;

impl ApiError {
    pub fn new(status: StatusCode, detail: impl Into<String>) -> Self {
        Self {
            status,
            detail: detail.into(),
        }
    }

    pub fn bad_request(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, detail)
    }

    pub fn forbidden(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, detail)
    }

    pub fn not_found(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, detail)
    }

    pub fn conflict(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, detail)
    }

    pub fn unprocessable(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, detail)
    }

    pub fn internal(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, detail)
    }

    /// Функция ещё не реализована в SlothForge (честный ответ вместо фальшивого успеха).
    pub fn not_implemented(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_IMPLEMENTED, detail)
    }

    /// Компонент (например, движок инференса) пока недоступен.
    pub fn service_unavailable(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, detail)
    }

    /// Переводит отказ песочницы путей в HTTP-ошибку. `what` — что искали, для текста ошибки.
    pub fn from_path_rejection(rejection: PathRejection, what: &str) -> Self {
        match rejection {
            PathRejection::Invalid => Self::bad_request(format!("Недопустимый путь: {what}")),
            PathRejection::NotFound => Self::not_found(format!("Не найдено: {what}")),
            PathRejection::OutsideRoots => Self::forbidden(format!(
                "Доступ запрещён: {what} находится вне разрешённых папок"
            )),
        }
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.status, self.detail)
    }
}

impl std::error::Error for ApiError {}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({ "detail": self.detail })),
        )
            .into_response()
    }
}
