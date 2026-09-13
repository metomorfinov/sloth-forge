//! Дополнительные папки с моделями: проверка путей, уникальные номера, сохранение в базе.

use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use sloth_server::create_router;
use sloth_server::state::AppState;
use sloth_server::store::Store;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const FOLDERS: &str = "/api/hub/scan-folders";

/// Сервер с базой в файле: второй сервер на той же базе — это «перезапуск».
async fn spawn_server(models_dir: PathBuf, database: &Path) -> String {
    let store = Store::open(database).expect("база открывается");
    let state = Arc::new(AppState::with_store(
        None,
        models_dir,
        None,
        Arc::new(store),
    ));
    let app = create_router(state, None);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("тестовый сервер упал");
    });
    format!("http://{addr}")
}

async fn call(base: &str, method: Method, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let mut request = reqwest::Client::new().request(method, format!("{base}{path}"));
    if let Some(body) = body {
        request = request.json(&body);
    }
    let response = request.send().await.expect("запрос не отправлен");
    let status = response.status();
    (status, response.json().await.unwrap_or(Value::Null))
}

async fn add_folder(base: &str, path: &Path) -> (StatusCode, Value) {
    call(
        base,
        Method::POST,
        FOLDERS,
        Some(json!({ "path": path.to_string_lossy() })),
    )
    .await
}

fn canonical(path: &Path) -> String {
    path.canonicalize().unwrap().to_string_lossy().into_owned()
}

#[tokio::test]
async fn folders_survive_restart_and_ids_are_never_reused() {
    let base = tempfile::tempdir().unwrap();
    let models = base.path().join("models");
    let (one, two, three) = (
        base.path().join("one"),
        base.path().join("two"),
        base.path().join("three"),
    );
    for dir in [&models, &one, &two, &three] {
        std::fs::create_dir_all(dir).unwrap();
    }
    let database = base.path().join("slothforge.db");

    let server = spawn_server(models.clone(), &database).await;
    let (status, first) = add_folder(&server, &one).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let (_, second) = add_folder(&server, &two).await;
    let second_id = second["id"].as_u64().expect("номер папки");
    let (status, _) = call(
        &server,
        Method::DELETE,
        &format!("{FOLDERS}/{second_id}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (_, third) = add_folder(&server, &three).await;
    assert_ne!(
        third["id"], second["id"],
        "номер удалённой папки не выдаётся повторно"
    );
    assert_ne!(third["id"], first["id"]);

    // Новый сервер на той же базе видит те же папки с теми же номерами
    let restarted = spawn_server(models, &database).await;
    let (_, listed) = call(&restarted, Method::GET, "/api/models/scan-folders", None).await;
    let folders = listed["folders"].as_array().expect("список папок");
    let paths: Vec<&str> = folders
        .iter()
        .map(|folder| folder["path"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(paths, [canonical(&one), canonical(&three)]);
    assert_eq!(folders[1]["id"], third["id"]);
    assert_eq!(folders[0]["status"], "ok");
}

#[tokio::test]
async fn rejects_paths_that_are_not_real_folders() {
    let base = tempfile::tempdir().unwrap();
    let models = base.path().join("models");
    let extra = base.path().join("extra");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::create_dir_all(&extra).unwrap();
    let file = base.path().join("notes.txt");
    std::fs::write(&file, b"notes").unwrap();
    let server = spawn_server(models.clone(), &base.path().join("db.sqlite")).await;

    let bad_paths = [
        (String::new(), "пустой путь"),
        ("models".to_string(), "относительный путь"),
        ("/".to_string(), "корень диска"),
        (
            base.path().join("missing").to_string_lossy().into_owned(),
            "папки нет",
        ),
        (file.to_string_lossy().into_owned(), "файл, а не папка"),
    ];
    for (path, why) in bad_paths {
        let (status, body) = call(
            &server,
            Method::POST,
            FOLDERS,
            Some(json!({ "path": path })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{why}: {body}");
        assert!(body["detail"].is_string(), "{why}: {body}");
    }

    let (status, added) = add_folder(&server, &extra).await;
    assert_eq!(status, StatusCode::OK, "{added}");
    let (status, _) = add_folder(&server, &extra).await;
    assert_eq!(status, StatusCode::CONFLICT, "папка уже добавлена");
    let (status, _) = add_folder(&server, &models).await;
    assert_eq!(
        status,
        StatusCode::CONFLICT,
        "основная папка моделей просматривается всегда"
    );

    let (status, _) = call(&server, Method::DELETE, &format!("{FOLDERS}/9999"), None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Папку удалили с диска: статус честно сообщает об этом
    std::fs::remove_dir(&extra).unwrap();
    let (_, listed) = call(&server, Method::GET, FOLDERS, None).await;
    assert_eq!(listed["folders"][0]["status"], "missing");
}

#[tokio::test]
async fn models_from_added_folder_are_listed() {
    let base = tempfile::tempdir().unwrap();
    let models = base.path().join("models");
    let extra = base.path().join("extra");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::create_dir_all(&extra).unwrap();
    std::fs::write(extra.join("My-Model-Q8_0.gguf"), b"GGUF").unwrap();
    let server = spawn_server(models, &base.path().join("db.sqlite")).await;

    let (status, _) = add_folder(&server, &extra).await;
    assert_eq!(status, StatusCode::OK);
    let (_, local) = call(&server, Method::GET, "/api/hub/local", None).await;
    let model = local["models"]
        .as_array()
        .and_then(|models| {
            models
                .iter()
                .find(|model| model["display_name"] == "My-Model-Q8_0")
        })
        .unwrap_or_else(|| panic!("модель из добавленной папки видна: {local}"));
    assert_eq!(model["source"], "custom");
}
