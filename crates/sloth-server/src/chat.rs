//! OpenAI-совместимая генерация ответов (`/v1/chat/completions`,
//! `/api/inference/chat`, `/api/inference/chat/completions`).
//!
//! Раньше здесь отдавались заготовленные фразы, подобранные по ключевым словам
//! («vulkan», «lora», «hello»), а стриминг изображался паузами по 20 мс — модель
//! при этом вообще не использовалась. Пока собственный Vulkan-движок инференса
//! не подключён (этап 2 дорожной карты), эндпоинт честно отвечает 503.
//!
//! Типы запроса и ответа сохранены: это OpenAI-контракт, который будет
//! обслуживать движок. Формат SSE-потока, который ждёт фронтенд: чанки
//! `chat.completion.chunk` в строках `data:`, финальный чанк с `finish_reason`
//! и строка `data: [DONE]`.

use crate::error::ApiError;
use axum::Json;
use serde::{Deserialize, Serialize};

/// Текст ошибки, пока движок инференса не подключён.
pub const ENGINE_NOT_READY: &str =
    "Движок инференса SlothForge ещё не подключён: генерация ответов появится на этапе 2 дорожной карты";

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
}

impl ChatMessage {
    /// Текст сообщения: строка или склейка текстовых частей из массива `content`.
    pub fn text_content(&self) -> String {
        if let Some(text) = self.content.as_str() {
            return text.to_string();
        }
        if let Some(parts) = self.content.as_array() {
            return parts
                .iter()
                .filter_map(|part| part.get("text").and_then(|text| text.as_str()))
                .collect();
        }
        self.content.to_string()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: CompletionUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoiceMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatChoiceMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub async fn handle_chat_completions(Json(request): Json<ChatCompletionRequest>) -> ApiError {
    tracing::info!(
        "Запрос генерации (модель {:?}, сообщений: {}) отклонён: движок инференса не подключён",
        request.model,
        request.messages.len()
    );
    ApiError::service_unavailable(ENGINE_NOT_READY)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_text_from_string_and_parts() {
        let plain = ChatMessage {
            role: "user".into(),
            content: serde_json::json!("привет"),
        };
        assert_eq!(plain.text_content(), "привет");

        let parts = ChatMessage {
            role: "user".into(),
            content: serde_json::json!([{ "type": "text", "text": "при" }, { "type": "text", "text": "вет" }]),
        };
        assert_eq!(parts.text_content(), "привет");
    }
}
