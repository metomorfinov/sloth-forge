//! История чатов: контракт с фронтендом (`chat-api.ts`) и сохранность данных
//! после перезапуска сервера.

use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use sloth_server::create_router;
use sloth_server::state::AppState;
use sloth_server::store::Store;
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;

/// Клиент к тестовому серверу.
struct Api {
    client: reqwest::Client,
    base_url: String,
}

impl Api {
    async fn start(database: &Path, models: &Path) -> Self {
        let store = Arc::new(Store::open(database).expect("база не открылась"));
        let state = Arc::new(AppState::with_store(
            None,
            models.to_path_buf(),
            None,
            store,
        ));
        let app = create_router(state, None);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("не удалось занять порт для теста");
        let addr = listener.local_addr().expect("нет адреса слушателя");
        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("тестовый сервер упал");
        });
        Self {
            client: reqwest::Client::new(),
            base_url: format!("http://{addr}"),
        }
    }

    async fn send(&self, method: Method, path: &str, body: Option<Value>) -> (StatusCode, Value) {
        let mut request = self
            .client
            .request(method, format!("{}{path}", self.base_url));
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = request.send().await.expect("запрос не отправлен");
        let status = response.status();
        let json = response.json().await.unwrap_or(Value::Null);
        (status, json)
    }
}

struct Fixture {
    _dir: TempDir,
    api: Api,
}

async fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("не удалось создать временную папку");
    let api = Api::start(&dir.path().join("slothforge.db"), dir.path()).await;
    Fixture { _dir: dir, api }
}

fn thread(id: &str, created_at: i64) -> Value {
    json!({
        "id": id,
        "title": "Первый чат",
        "modelType": "base",
        "pairId": "pair-1",
        "archived": false,
        "createdAt": created_at,
        "settings": { "temperature": 0.4 }
    })
}

fn message(id: &str, role: &str, created_at: i64) -> Value {
    json!({ "id": id, "role": role, "content": [{ "type": "text", "text": id }], "createdAt": created_at })
}

#[tokio::test]
async fn thread_and_message_lifecycle_matches_frontend_contract() {
    let Fixture { _dir, api } = fixture().await;

    // Сохранение треда возвращает полную запись, лишние поля не теряются
    let (status, saved) = api
        .send(Method::POST, "/api/chat/threads", Some(thread("t1", 1000)))
        .await;
    assert_eq!(status, StatusCode::OK, "{saved}");
    assert_eq!(saved["pairId"], "pair-1");
    assert_eq!(saved["settings"]["temperature"], 0.4);

    let (_, listed) = api.send(Method::GET, "/api/chat/threads", None).await;
    assert_eq!(listed["threads"].as_array().map(Vec::len), Some(1));

    let (status, missing) = api.send(Method::GET, "/api/chat/threads/nope", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(missing["detail"].is_string());

    // PATCH: условие на название и слияние настроек
    let (status, _) = api
        .send(
            Method::PATCH,
            "/api/chat/threads/t1",
            Some(json!({ "title": "X", "expectedTitle": "Чужое" })),
        )
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    let (status, patched) = api
        .send(
            Method::PATCH,
            "/api/chat/threads/t1",
            Some(json!({ "title": "Переименован", "expectedTitle": "Первый чат", "settingsPatch": { "topK": 20 } })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{patched}");
    assert_eq!(patched["title"], "Переименован");
    assert_eq!(patched["settings"]["temperature"], 0.4);
    assert_eq!(patched["settings"]["topK"], 20);

    // Сообщения: пачка, одиночное сохранение, удаление через pruneMissing
    let (status, synced) = api
        .send(
            Method::PUT,
            "/api/chat/threads/t1/messages",
            Some(json!({ "messages": [message("m1", "user", 1), message("m2", "assistant", 2)], "pruneMissing": false })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{synced}");
    assert_eq!(synced["messages"].as_array().map(Vec::len), Some(2));

    let (status, single) = api
        .send(
            Method::PUT,
            "/api/chat/threads/t1/messages/m3",
            Some(message("m3", "user", 3)),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(single["threadId"], "t1");

    let (status, _) = api
        .send(Method::GET, "/api/chat/threads/t1/messages/nope", None)
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (_, pruned) = api
        .send(
            Method::PUT,
            "/api/chat/threads/t1/messages",
            Some(json!({ "messages": [message("m1", "user", 1), message("m2", "assistant", 2)], "pruneMissing": true })),
        )
        .await;
    assert_eq!(pruned["messages"].as_array().map(Vec::len), Some(2));

    // Форк от сообщения
    let (status, forked) = api
        .send(
            Method::POST,
            "/api/chat/threads/t1/fork",
            Some(json!({ "messageId": "m1", "newThreadId": "fork-1", "createdAt": 5000 })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{forked}");
    assert_eq!(forked["thread"]["forkedFromThreadId"], "t1");
    assert_eq!(forked["messages"].as_array().map(Vec::len), Some(1));
    let (_, counts) = api
        .send(Method::GET, "/api/chat/threads/t1/forks", None)
        .await;
    assert_eq!(counts["counts"]["m1"], 1);

    // Удаление: тред исчезает и не воскресает от запоздалого автосохранения
    let (status, deleted) = api
        .send(
            Method::DELETE,
            "/api/chat/threads",
            Some(json!({ "ids": ["t1"], "delete_files": false })),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted["deletedThreadIds"], json!(["t1"]));
    let (status, _) = api
        .send(Method::POST, "/api/chat/threads", Some(thread("t1", 1000)))
        .await;
    assert_eq!(status, StatusCode::GONE);

    let (_, count) = api.send(Method::GET, "/api/chat/count", None).await;
    assert_eq!(count["count"], 1, "форк остаётся");
}

#[tokio::test]
async fn history_and_settings_survive_restart() {
    let dir = tempfile::tempdir().expect("не удалось создать временную папку");
    let database = dir.path().join("slothforge.db");

    {
        let api = Api::start(&database, dir.path()).await;
        api.send(
            Method::POST,
            "/api/chat/threads",
            Some(thread("keep", 1000)),
        )
        .await;
        api.send(
            Method::PUT,
            "/api/chat/threads/keep/messages",
            Some(json!({ "messages": [message("m1", "user", 1)] })),
        )
        .await;
        let (status, _) = api
            .send(
                Method::PUT,
                "/api/chat/settings",
                Some(json!({ "autoTitle": false })),
            )
            .await;
        assert_eq!(status, StatusCode::OK);
    }

    // «Перезапуск»: новый сервер с новым подключением к тому же файлу базы
    let restarted = Api::start(&database, dir.path()).await;
    let (status, thread) = restarted
        .send(Method::GET, "/api/chat/threads/keep", None)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(thread["title"], "Первый чат");
    let (_, messages) = restarted
        .send(Method::GET, "/api/chat/threads/keep/messages", None)
        .await;
    assert_eq!(messages["messages"].as_array().map(Vec::len), Some(1));
    let (_, settings) = restarted
        .send(Method::GET, "/api/chat/settings", None)
        .await;
    assert_eq!(settings["settings"]["autoTitle"], false);
}

#[tokio::test]
async fn chat_settings_compare_and_set() {
    let Fixture { _dir, api } = fixture().await;

    let (_, saved) = api
        .send(
            Method::PUT,
            "/api/chat/settings",
            Some(json!({ "activePreset": "default" })),
        )
        .await;
    assert_eq!(saved["settings"]["activePreset"], "default");

    let (_, stale) = api
        .send(
            Method::POST,
            "/api/chat/settings/compare-and-set",
            Some(json!({ "expected": { "activePreset": "other" }, "patch": { "activePreset": "x" } })),
        )
        .await;
    assert_eq!(stale["applied"], false);
    assert_eq!(stale["settings"]["activePreset"], "default");

    let (_, fresh) = api
        .send(
            Method::POST,
            "/api/chat/settings/compare-and-set",
            Some(json!({
                "expected": { "activePreset": "default" },
                "expectedAbsent": ["customPresets"],
                "expectedAbsentPaths": [["inferenceParamsByModel", "llama"]],
                "patch": { "activePreset": "creative" }
            })),
        )
        .await;
    assert_eq!(fresh["applied"], true);
    assert_eq!(fresh["settings"]["activePreset"], "creative");
}

#[tokio::test]
async fn projects_clear_all_and_export() {
    let Fixture { _dir, api } = fixture().await;

    let (status, project) = api
        .send(
            Method::POST,
            "/api/chat/projects",
            Some(json!({ "id": "p1", "name": "Проект" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{project}");
    let (_, patched) = api
        .send(
            Method::PATCH,
            "/api/chat/projects/p1",
            Some(json!({ "instructions": "Кратко" })),
        )
        .await;
    assert_eq!(patched["instructions"], "Кратко");

    let mut member = thread("t1", 1);
    member["projectId"] = json!("p1");
    api.send(Method::POST, "/api/chat/threads", Some(member))
        .await;

    let (status, _) = api
        .send(Method::DELETE, "/api/chat/projects/p1", None)
        .await;
    assert_eq!(status, StatusCode::OK);
    let (_, detached) = api.send(Method::GET, "/api/chat/threads/t1", None).await;
    assert_eq!(detached["projectId"], Value::Null);
    let (status, _) = api.send(Method::GET, "/api/chat/projects/p1", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (_, exported) = api.send(Method::GET, "/api/chat/export", None).await;
    assert_eq!(exported["threadCount"], 1);

    let (status, cleared) = api
        .send(
            Method::DELETE,
            "/api/chat",
            Some(json!({ "ids": ["pending"] })),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cleared["deletedThreadIds"], json!(["t1"]));
    let (_, count) = api.send(Method::GET, "/api/chat/count", None).await;
    assert_eq!(count["count"], 0);
    let (status, _) = api
        .send(
            Method::POST,
            "/api/chat/threads",
            Some(thread("pending", 2)),
        )
        .await;
    assert_eq!(status, StatusCode::GONE);
}

#[tokio::test]
async fn chat_runs_are_absent_and_generation_is_honest() {
    let Fixture { _dir, api } = fixture().await;

    // 404 здесь нужен фронтенду, чтобы он перешёл на обычный SSE-поток
    let (status, _) = api
        .send(
            Method::GET,
            "/api/inference/chat-runs/active?threadId=t1",
            None,
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, body) = api
        .send(
            Method::POST,
            "/v1/chat/completions",
            Some(json!({ "messages": [{ "role": "user", "content": "привет" }], "stream": true })),
        )
        .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body["detail"]
        .as_str()
        .is_some_and(|d| d.contains("этапе 2")));
}
