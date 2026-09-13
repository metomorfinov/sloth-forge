//! Обращения к Hugging Face: список файлов репозитория, токен доступа и его проверка.
//!
//! Раньше проверка токена всегда отвечала «valid», список файлов запрашивался без токена
//! (закрытые репозитории не открывались), а адрес huggingface.co был зашит в код.

use crate::error::ApiResult;
use crate::state::AppState;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::Json;
use reqwest::header::RETRY_AFTER;
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Адрес Hugging Face по умолчанию; переопределяется `HF_ENDPOINT`, как в huggingface_hub.
pub const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
pub const HF_ENDPOINT_ENV: &str = "HF_ENDPOINT";
/// Заголовок, в котором фронтенд передаёт токен Hugging Face.
pub const HF_TOKEN_HEADER: &str = "x-unsloth-hf-token";

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Сколько можно ждать очередной порции данных. Общего ограничения на длительность
/// загрузки нет: большая модель на медленном канале качается столько, сколько нужно.
const READ_TIMEOUT: Duration = Duration::from_secs(120);
const TREE_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const TOKEN_CHECK_TIMEOUT: Duration = Duration::from_secs(15);
/// Список файлов репозитория меняется редко, а интерфейс спрашивает его часто.
const TREE_CACHE_TTL: Duration = Duration::from_secs(10 * 60);
const MAX_REDIRECTS: usize = 10;
const USER_AGENT: &str = concat!("sloth-forge/", env!("CARGO_PKG_VERSION"));

pub const ACCESS_DENIED_HINT: &str =
    "Нет доступа к репозиторию: для закрытых моделей укажите токен Hugging Face в настройках";

/// Файл в репозитории Hugging Face.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteFile {
    pub path: String,
    pub size: Option<u64>,
}

/// Кэш списков файлов: ключ — репозиторий и признак «запрошено с токеном».
pub type TreeCache = RwLock<HashMap<String, (Instant, Vec<RemoteFile>)>>;

/// Адрес Hugging Face из `HF_ENDPOINT` (без завершающего `/`) или адрес по умолчанию.
pub fn endpoint_from_env() -> String {
    std::env::var(HF_ENDPOINT_ENV)
        .ok()
        .map(|endpoint| endpoint.trim().trim_end_matches('/').to_string())
        .filter(|endpoint| !endpoint.is_empty())
        .unwrap_or_else(|| DEFAULT_HF_ENDPOINT.to_string())
}

pub fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS))
        .user_agent(USER_AGENT)
        .build()
        .map_err(|err| format!("Не удалось создать HTTP-клиент: {err}"))
}

pub fn authorized(
    request: reqwest::RequestBuilder,
    token: Option<&str>,
) -> reqwest::RequestBuilder {
    match token {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

/// Токен из заголовка запроса фронтенда.
fn header_token(headers: &HeaderMap) -> Option<String> {
    headers
        .get(HF_TOKEN_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_string)
}

/// Токен из заголовка, иначе сохранённый в настройках.
pub async fn resolve_token(state: &AppState, headers: &HeaderMap) -> ApiResult<Option<String>> {
    match header_token(headers) {
        Some(token) => Ok(Some(token)),
        None => crate::settings::runtime::stored_hf_token(state).await,
    }
}

#[derive(Debug, Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    size: Option<u64>,
    lfs: Option<TreeLfs>,
}

#[derive(Debug, Deserialize)]
struct TreeLfs {
    size: Option<u64>,
}

/// GGUF-файлы репозитория с точными размерами (для LFS-файлов — размер из `lfs.size`).
pub async fn fetch_repo_gguf_files(
    client: &reqwest::Client,
    endpoint: &str,
    repo_id: &str,
    token: Option<&str>,
) -> Result<Vec<RemoteFile>, String> {
    let url = format!("{endpoint}/api/models/{repo_id}/tree/main?recursive=true");
    let response = authorized(client.get(&url).timeout(TREE_REQUEST_TIMEOUT), token)
        .send()
        .await
        .map_err(|err| format!("Hugging Face недоступен: {err}"))?;
    match response.status() {
        status if status.is_success() => {}
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
            return Err(ACCESS_DENIED_HINT.to_string())
        }
        StatusCode::NOT_FOUND => {
            return Err(format!("Репозиторий {repo_id} не найден на Hugging Face"))
        }
        status => {
            return Err(format!(
                "Hugging Face ответил HTTP {status} на список файлов"
            ))
        }
    }
    let entries: Vec<TreeEntry> = response
        .json()
        .await
        .map_err(|err| format!("Не удалось разобрать список файлов репозитория: {err}"))?;
    let mut files: Vec<RemoteFile> = entries
        .into_iter()
        .filter(|entry| entry.kind == "file" && entry.path.to_ascii_lowercase().ends_with(".gguf"))
        .map(|entry| RemoteFile {
            size: entry.lfs.and_then(|lfs| lfs.size).or(entry.size),
            path: entry.path,
        })
        .collect();
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(files)
}

/// То же, что [`fetch_repo_gguf_files`], но с кэшем на [`TREE_CACHE_TTL`].
/// Неудачные ответы не кэшируются: после появления сети или токена список откроется сразу.
pub async fn cached_repo_gguf_files(
    state: &AppState,
    repo_id: &str,
    token: Option<&str>,
) -> Result<Vec<RemoteFile>, String> {
    let key = format!("{}::{}", repo_id.to_ascii_lowercase(), token.is_some());
    if let Some((fetched_at, files)) = state.hf_tree_cache.read().await.get(&key) {
        if fetched_at.elapsed() < TREE_CACHE_TTL {
            return Ok(files.clone());
        }
    }
    let client = http_client()?;
    let files = fetch_repo_gguf_files(&client, &state.hf_endpoint, repo_id, token).await?;
    state
        .hf_tree_cache
        .write()
        .await
        .insert(key, (Instant::now(), files.clone()));
    Ok(files)
}

/// Статусы проверки токена, как их понимает фронтенд (`HfTokenValidationStatus`).
const TOKEN_MISSING: &str = "missing";
const TOKEN_VALID: &str = "valid";
const TOKEN_INVALID: &str = "invalid";
const TOKEN_RATE_LIMITED: &str = "rate_limited";
const TOKEN_UNAVAILABLE: &str = "unavailable";

/// `POST /api/hub/token/validate`: спрашивает Hugging Face, чей это токен.
pub async fn validate_token(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Json<Value> {
    let (status, retry_after) = match header_token(&headers) {
        None => (TOKEN_MISSING, None),
        Some(token) => check_token(&state.hf_endpoint, &token).await,
    };
    Json(json!({ "status": status, "retry_after_seconds": retry_after }))
}

async fn check_token(endpoint: &str, token: &str) -> (&'static str, Option<u64>) {
    let client = match http_client() {
        Ok(client) => client,
        Err(message) => {
            tracing::warn!("{message}");
            return (TOKEN_UNAVAILABLE, None);
        }
    };
    let response = match client
        .get(format!("{endpoint}/api/whoami-v2"))
        .bearer_auth(token)
        .timeout(TOKEN_CHECK_TIMEOUT)
        .send()
        .await
    {
        Ok(response) => response,
        Err(err) => {
            // Сам токен в лог не пишется
            tracing::warn!("Проверка токена Hugging Face не удалась: {err}");
            return (TOKEN_UNAVAILABLE, None);
        }
    };
    match response.status() {
        status if status.is_success() => (TOKEN_VALID, None),
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => (TOKEN_INVALID, None),
        StatusCode::TOO_MANY_REQUESTS => {
            let retry_after = response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.trim().parse().ok());
            (TOKEN_RATE_LIMITED, retry_after)
        }
        status => {
            tracing::warn!("Hugging Face ответил HTTP {status} на проверку токена");
            (TOKEN_UNAVAILABLE, None)
        }
    }
}
