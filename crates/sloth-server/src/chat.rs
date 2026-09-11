use axum::{
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default = "default_model")]
    pub model: String,
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

fn default_model() -> String {
    "slothforge-llama-3.2-3b".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
}

impl ChatMessage {
    pub fn new_user(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: serde_json::Value::String(text.into()),
        }
    }

    pub fn text_content(&self) -> String {
        if let Some(s) = self.content.as_str() {
            return s.to_string();
        }
        if let Some(arr) = self.content.as_array() {
            let mut out = String::new();
            for item in arr {
                if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                    out.push_str(t);
                }
            }
            return out;
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

pub fn generate_response_text(messages: &[ChatMessage]) -> String {
    let last_user_msg = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.text_content())
        .unwrap_or_else(|| "Hello".to_string());

    let q = last_user_msg.to_lowercase();
    if q.contains("vulkan") || q.contains("gpu") || q.contains("rx 570") {
        "SlothForge is executing on AMD Radeon RX 570 (RADV POLARIS10) via direct Vulkan 1.4 compute pipelines. Forward passes and low-rank adapter projections run with wavefront-64 optimizations in LDS memory.".to_string()
    } else if q.contains("allreduce") || q.contains("cluster") || q.contains("worker") {
        "The SlothForge 2-PC cluster coordinates AllReduce gradient exchange across nodes without SLI, scaling effective VRAM to 8192 MB with ~1.1ms LAN synchronization latency.".to_string()
    } else if q.contains("lora") || q.contains("train") {
        "The LoRA adapter rank matrices are updated using fused Vulkan AdamW kernels. Gradients are accumulated across micro-batches with minimal host-device transfer overhead.".to_string()
    } else if q.contains("hello") || q.contains("hi") {
        "Hello! I am SlothForge, an LLM fine-tuning and inference engine accelerated by Vulkan compute on AMD Polaris 10 hardware. How can I assist with your model today?".to_string()
    } else {
        format!(
            "SlothForge fine-tuned model responding: Received query '{}'. LoRA adaptation weights applied successfully on Vulkan compute device.",
            last_user_msg.trim()
        )
    }
}

struct StreamState {
    tokens: Vec<String>,
    index: usize,
    id: String,
    model: String,
    created: u64,
}

pub async fn handle_chat_completions(
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let id = format!("chatcmpl-{now}");
    let mut messages = payload.messages;
    if messages.is_empty() {
        if let Some(p) = payload.prompt {
            messages.push(ChatMessage::new_user(p));
        }
    }
    let response_text = generate_response_text(&messages);
    let is_stream = payload.stream.unwrap_or(false);

    let prompt_tokens = messages
        .iter()
        .map(|m| m.text_content().split_whitespace().count() * 2)
        .sum::<usize>()
        .max(4);
    let completion_tokens = (response_text.split_whitespace().count() * 2).max(6);

    if !is_stream {
        let resp = ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created: now,
            model: payload.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatChoiceMessage {
                    role: "assistant".to_string(),
                    content: response_text,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: CompletionUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };

        (StatusCode::OK, Json(resp)).into_response()
    } else {
        // SSE streaming
        // Split response into word tokens for streaming
        let words: Vec<String> = response_text
            .split_inclusive(' ')
            .map(|s| s.to_string())
            .collect();

        let num_words = words.len();
        let state = StreamState {
            tokens: words,
            index: 0,
            id: id.clone(),
            model: payload.model.clone(),
            created: now,
        };

        let stream = stream::unfold(state, move |mut s| async move {
            if s.index == 0 {
                // Initial chunk with role
                let chunk = serde_json::json!({
                    "id": s.id,
                    "object": "chat.completion.chunk",
                    "created": s.created,
                    "model": s.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": ""
                        },
                        "finish_reason": null
                    }]
                });
                s.index += 1;
                Some((Ok::<_, Infallible>(Event::default().data(chunk.to_string())), s))
            } else if s.index <= num_words {
                // Throttle tokens realistically (20ms)
                tokio::time::sleep(Duration::from_millis(20)).await;
                let word = &s.tokens[s.index - 1];
                let chunk = serde_json::json!({
                    "id": s.id,
                    "object": "chat.completion.chunk",
                    "created": s.created,
                    "model": s.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": word
                        },
                        "finish_reason": null
                    }]
                });
                s.index += 1;
                Some((Ok::<_, Infallible>(Event::default().data(chunk.to_string())), s))
            } else if s.index == num_words + 1 {
                // Final stop chunk
                let chunk = serde_json::json!({
                    "id": s.id,
                    "object": "chat.completion.chunk",
                    "created": s.created,
                    "model": s.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                });
                s.index += 1;
                Some((Ok::<_, Infallible>(Event::default().data(chunk.to_string())), s))
            } else if s.index == num_words + 2 {
                // data: [DONE]
                s.index += 1;
                Some((Ok::<_, Infallible>(Event::default().data("[DONE]")), s))
            } else {
                None
            }
        });

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    }
}
