//! Общие помощники интеграционных тестов: имитация Hugging Face (дерево файлов, отдача
//! файлов с HTTP Range, проверка токена) и сервер SlothForge на временной папке моделей.
//! Тесты не ходят в интернет.

// Каждый тестовый файл пользуется своей частью помощников
#![allow(dead_code)]

use axum::body::{Body, Bytes};
use axum::extract::{Path as AxPath, State as AxState};
use axum::http::{header, HeaderMap, StatusCode as AxStatus};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use sloth_server::create_router;
use sloth_server::state::AppState;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

const CHUNK: usize = 8 * 1024;
const SLOW_CHUNK_DELAY: Duration = Duration::from_millis(40);
/// Размер файла в дереве, у которого нет содержимого (сервер отвечает на него 404).
const MISSING_FILE_SIZE: usize = 12_345;
pub const POLL_INTERVAL: Duration = Duration::from_millis(25);
pub const POLL_ATTEMPTS: usize = 400;
pub const REPO: &str = "unsloth/Tiny-GGUF";
pub const PRIVATE_REPO: &str = "private/Secret-GGUF";
pub const PRIVATE_TOKEN: &str = "hf_secret";
/// Токен, на который имитация отвечает «слишком много запросов».
pub const LIMITED_TOKEN: &str = "hf_limited";
pub const RETRY_AFTER_SECONDS: u64 = 7;
/// Закрытый порт: подключение отклоняется сразу, так тесты видят «Hugging Face недоступен».
pub const UNREACHABLE_HF_ENDPOINT: &str = "http://127.0.0.1:9";

const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_STRING: u32 = 8;

/// Предсказуемое содержимое файла: по нему проверяется, что докачка ничего не испортила.
pub fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

/// Маленький корректный GGUF без тензоров (архитектура llama и длина контекста),
/// дополненный нулями до `len` байт.
pub fn minimal_gguf(context_length: u32, len: usize) -> Vec<u8> {
    fn string(buf: &mut Vec<u8>, text: &str) {
        buf.extend_from_slice(&(text.len() as u64).to_le_bytes());
        buf.extend_from_slice(text.as_bytes());
    }
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes());
    string(&mut buf, "general.architecture");
    buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
    string(&mut buf, "llama");
    string(&mut buf, "llama.context_length");
    buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
    buf.extend_from_slice(&context_length.to_le_bytes());
    let len = len.max(buf.len());
    buf.resize(len, 0);
    buf
}

/// Файлы репозитория: путь → содержимое (None — файл есть в дереве, но не отдаётся).
pub type RepoFiles = Vec<(String, Option<Arc<Vec<u8>>>)>;

pub struct MockHf {
    /// «org/repo» → файлы
    repos: HashMap<String, RepoFiles>,
    slow: bool,
    pub range_requests: AtomicUsize,
    pub tree_requests: AtomicUsize,
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
}

impl MockHf {
    fn authorized(&self, repo: &str, headers: &HeaderMap) -> bool {
        repo != PRIVATE_REPO || bearer(headers) == Some(PRIVATE_TOKEN)
    }
}

async fn mock_tree(
    AxState(mock): AxState<Arc<MockHf>>,
    AxPath((org, repo)): AxPath<(String, String)>,
    headers: HeaderMap,
) -> Response {
    mock.tree_requests.fetch_add(1, Ordering::Relaxed);
    let repo_id = format!("{org}/{repo}");
    if !mock.authorized(&repo_id, &headers) {
        return AxStatus::UNAUTHORIZED.into_response();
    }
    let Some(files) = mock.repos.get(&repo_id) else {
        return AxStatus::NOT_FOUND.into_response();
    };
    let entries: Vec<Value> = files
        .iter()
        .map(|(path, data)| {
            let size = data.as_ref().map_or(MISSING_FILE_SIZE, |d| d.len());
            // Как у настоящего HF: у LFS-файла `size` — размер указателя, реальный — в `lfs.size`
            json!({ "type": "file", "path": path, "size": 134, "lfs": { "size": size } })
        })
        .collect();
    Json(entries).into_response()
}

async fn mock_resolve(
    AxState(mock): AxState<Arc<MockHf>>,
    AxPath((org, repo, path)): AxPath<(String, String, String)>,
    headers: HeaderMap,
) -> Response {
    let repo_id = format!("{org}/{repo}");
    if !mock.authorized(&repo_id, &headers) {
        return AxStatus::UNAUTHORIZED.into_response();
    }
    let data = mock
        .repos
        .get(&repo_id)
        .and_then(|files| files.iter().find(|(p, _)| *p == path))
        .and_then(|(_, data)| data.clone());
    let Some(data) = data else {
        return AxStatus::NOT_FOUND.into_response();
    };

    let start = match headers.get(header::RANGE).and_then(|v| v.to_str().ok()) {
        Some(range) => {
            mock.range_requests.fetch_add(1, Ordering::Relaxed);
            let start: usize = range
                .trim_start_matches("bytes=")
                .trim_end_matches('-')
                .parse()
                .unwrap_or(0);
            if start >= data.len() {
                return AxStatus::RANGE_NOT_SATISFIABLE.into_response();
            }
            Some(start)
        }
        None => None,
    };
    let offset = start.unwrap_or(0);
    let slow = mock.slow;
    let stream = futures_util::stream::unfold((data, offset), move |(data, pos)| async move {
        if pos >= data.len() {
            return None;
        }
        if slow {
            tokio::time::sleep(SLOW_CHUNK_DELAY).await;
        }
        let end = (pos + CHUNK).min(data.len());
        let chunk = Bytes::copy_from_slice(&data[pos..end]);
        Some((Ok::<_, std::io::Error>(chunk), (data, end)))
    });
    let status = if start.is_some() {
        AxStatus::PARTIAL_CONTENT
    } else {
        AxStatus::OK
    };
    (status, Body::from_stream(stream)).into_response()
}

async fn mock_whoami(headers: HeaderMap) -> Response {
    match bearer(&headers) {
        Some(PRIVATE_TOKEN) => Json(json!({ "name": "tester" })).into_response(),
        Some(LIMITED_TOKEN) => (
            AxStatus::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, RETRY_AFTER_SECONDS.to_string())],
        )
            .into_response(),
        _ => AxStatus::UNAUTHORIZED.into_response(),
    }
}

async fn spawn_mock(mock: MockHf) -> (String, Arc<MockHf>) {
    let mock = Arc::new(mock);
    let app = Router::new()
        .route("/api/whoami-v2", get(mock_whoami))
        .route("/api/models/:org/:repo/tree/main", get(mock_tree))
        .route("/:org/:repo/resolve/main/*path", get(mock_resolve))
        .with_state(Arc::clone(&mock));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("имитация HF упала");
    });
    (format!("http://{addr}"), mock)
}

/// Репозиторий из двух шардов Q4_K_M, одного файла Q8_0, «битого» Q4_0 и проектора.
pub fn tiny_repo() -> RepoFiles {
    vec![
        (
            "Q4_K_M/Tiny-Q4_K_M-00001-of-00002.gguf".into(),
            Some(Arc::new(pattern(40_000, 1))),
        ),
        (
            "Q4_K_M/Tiny-Q4_K_M-00002-of-00002.gguf".into(),
            Some(Arc::new(pattern(25_000, 2))),
        ),
        ("Tiny-Q8_0.gguf".into(), Some(Arc::new(pattern(30_000, 3)))),
        ("Broken-Q4_0.gguf".into(), None),
        ("mmproj-F16.gguf".into(), Some(Arc::new(pattern(1_000, 4)))),
    ]
}

pub struct Fixture {
    _models: TempDir,
    pub models_dir: PathBuf,
    pub client: reqwest::Client,
    pub base_url: String,
    pub mock: Arc<MockHf>,
}

impl Fixture {
    /// Сервер с доступной имитацией Hugging Face; `slow` — отдавать файлы медленно.
    pub async fn start(slow: bool) -> Self {
        Self::launch(slow, true).await
    }

    /// Сервер, для которого Hugging Face недоступен.
    pub async fn start_offline() -> Self {
        Self::launch(false, false).await
    }

    async fn launch(slow: bool, hf_online: bool) -> Self {
        let mut repos = HashMap::new();
        repos.insert(REPO.to_string(), tiny_repo());
        repos.insert(
            PRIVATE_REPO.to_string(),
            vec![(
                "Secret-Q4_K_M.gguf".into(),
                Some(Arc::new(pattern(5_000, 9))),
            )],
        );
        let (hf_url, mock) = spawn_mock(MockHf {
            repos,
            slow,
            range_requests: AtomicUsize::new(0),
            tree_requests: AtomicUsize::new(0),
        })
        .await;

        let models = tempfile::tempdir().unwrap();
        let mut state = AppState::new(None, models.path().to_path_buf(), None);
        state.hf_endpoint = if hf_online {
            hf_url
        } else {
            UNREACHABLE_HF_ENDPOINT.to_string()
        };
        let app = create_router(Arc::new(state), None);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("тестовый сервер упал");
        });
        Self {
            models_dir: models.path().to_path_buf(),
            _models: models,
            client: reqwest::Client::new(),
            base_url: format!("http://{addr}"),
            mock,
        }
    }

    async fn execute(
        &self,
        request: reqwest::RequestBuilder,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let request = match body {
            Some(body) => request.json(&body),
            None => request,
        };
        let response = request.send().await.expect("запрос не отправлен");
        let status = response.status();
        (status, response.json().await.unwrap_or(Value::Null))
    }

    pub async fn send(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let request = self
            .client
            .request(method, format!("{}{path}", self.base_url));
        self.execute(request, body).await
    }

    /// Запрос с токеном Hugging Face в заголовке, как его шлёт фронтенд.
    pub async fn send_with_token(
        &self,
        method: Method,
        path: &str,
        token: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let request = self
            .client
            .request(method, format!("{}{path}", self.base_url))
            .header("X-Unsloth-HF-Token", token);
        self.execute(request, body).await
    }

    pub async fn start_download(&self, repo: &str, variant: &str) -> (StatusCode, Value) {
        self.send(
            Method::POST,
            "/api/hub/download",
            Some(json!({ "repo_id": repo, "gguf_variant": variant })),
        )
        .await
    }

    pub async fn status(&self, repo: &str, variant: &str) -> Value {
        self.send(
            Method::GET,
            &format!("/api/hub/download-status?repo_id={repo}&gguf_variant={variant}"),
            None,
        )
        .await
        .1
    }

    pub async fn wait_for(&self, repo: &str, variant: &str, wanted: &[&str]) -> Value {
        for _ in 0..POLL_ATTEMPTS {
            let status = self.status(repo, variant).await;
            if wanted.contains(&status["state"].as_str().unwrap_or("")) {
                return status;
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
        panic!("загрузка {repo} {variant} не пришла в состояние {wanted:?}");
    }

    pub fn repo_file(&self, repo: &str, path: &str) -> PathBuf {
        self.models_dir.join(repo).join(path)
    }

    /// Кладёт файл в папку моделей (путь относительно неё) и возвращает полный путь.
    pub fn put_file(&self, relative: &str, bytes: &[u8]) -> PathBuf {
        let path = self.models_dir.join(relative);
        std::fs::create_dir_all(path.parent().expect("у файла есть папка")).unwrap();
        std::fs::write(&path, bytes).unwrap();
        path
    }
}
