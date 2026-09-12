//! История чатов: треды, сообщения, проекты и настройки чата (`/api/chat/*`).
//!
//! Раньше эти данные жили в оперативной памяти, пропадали при перезапуске и отдавались
//! не в том формате, который ждёт фронтенд: удаление тредов молча ничего не удаляло,
//! несуществующий тред «находился» с выдуманным ответом 200, поля вроде `pairId`
//! и `settings` терялись. Теперь всё хранится в SQLite (модуль `store`) по контракту
//! фронтенда (`frontend/src/features/chat/api/chat-api.ts`).

use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use crate::store::chat::{self, ChatError, ThreadFilter};
use crate::store::now_ms;
use crate::store::settings::{self, Expectations};
use axum::extract::{Path, Query, State};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::sync::Arc;

/// Ключ настроек чата в таблице настроек.
pub const CHAT_SETTINGS_KEY: &str = "chat.settings";

fn body_or_empty(payload: Option<Json<Value>>) -> Value {
    payload
        .map(|Json(value)| value)
        .unwrap_or_else(|| json!({}))
}

fn string_list(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn object_field(body: &Value, key: &str) -> Map<String, Value> {
    match body.get(key) {
        Some(Value::Object(map)) => map.clone(),
        _ => Map::new(),
    }
}

fn thread_not_found(id: &str) -> ApiError {
    ApiError::not_found(format!("Тред {id} не найден"))
}

// ---------- треды ----------

#[derive(Debug, Default, Deserialize)]
pub struct ThreadListQuery {
    model_type: Option<String>,
    pair_id: Option<String>,
    project_id: Option<String>,
    include_archived: Option<bool>,
}

pub async fn list_threads(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ThreadListQuery>,
) -> ApiResult<Json<Value>> {
    let filter = ThreadFilter {
        model_type: query.model_type,
        pair_id: query.pair_id,
        project_id: query.project_id,
        include_archived: query.include_archived.unwrap_or(false),
    };
    let threads = state
        .store
        .call(move |conn| chat::list_threads(conn, &filter))
        .await?;
    Ok(Json(json!({ "threads": threads })))
}

pub async fn get_thread(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let lookup_id = id.clone();
    state
        .store
        .call(move |conn| chat::get_thread(conn, &lookup_id))
        .await?
        .map(Json)
        .ok_or_else(|| thread_not_found(&id))
}

/// `POST /api/chat/threads`: сохраняет тред целиком и возвращает сохранённую запись.
pub async fn save_thread(
    State(state): State<Arc<AppState>>,
    Json(record): Json<Value>,
) -> ApiResult<Json<Value>> {
    let saved = state
        .store
        .call(move |conn| chat::save_thread(conn, record))
        .await?;
    Ok(Json(saved))
}

/// `PUT/PATCH /api/chat/threads/:id`: частичное обновление с условиями `expected*`.
pub async fn update_thread(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    payload: Option<Json<Value>>,
) -> ApiResult<Json<Value>> {
    let patch = body_or_empty(payload);
    let updated = state
        .store
        .call(move |conn| chat::patch_thread(conn, &id, patch))
        .await?;
    Ok(Json(updated))
}

pub async fn delete_thread(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let deleted = state
        .store
        .call(move |conn| chat::delete_threads(conn, &[id]))
        .await?;
    Ok(Json(json!({
        "status": "ok",
        "deleted": !deleted.is_empty(),
        "sandboxes_kept": []
    })))
}

/// `DELETE /api/chat/threads` с телом `{ ids, delete_files }`.
pub async fn delete_threads(
    State(state): State<Arc<AppState>>,
    payload: Option<Json<Value>>,
) -> ApiResult<Json<Value>> {
    let body = body_or_empty(payload);
    let mut ids = string_list(body.get("ids"));
    if ids.is_empty() {
        // Старое имя поля, которое читал прежний бэкенд
        ids = string_list(body.get("threadIds"));
    }
    let deleted = state
        .store
        .call(move |conn| chat::delete_threads(conn, &ids))
        .await?;
    Ok(Json(
        json!({ "deletedThreadIds": deleted, "sandboxes_kept": [] }),
    ))
}

pub async fn thread_fork_counts(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let counts = state
        .store
        .call(move |conn| chat::fork_counts(conn, &id))
        .await?;
    Ok(Json(json!({ "counts": counts })))
}

/// `POST /api/chat/threads/:id/fork` с телом `{ messageId, newThreadId, createdAt }`.
pub async fn fork_thread(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let message_id = body
        .get("messageId")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("Не указан messageId"))?
        .to_string();
    let new_thread_id = body
        .get("newThreadId")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("Не указан newThreadId"))?
        .to_string();
    let created_at = body
        .get("createdAt")
        .and_then(Value::as_i64)
        .unwrap_or_else(now_ms);
    let (thread, messages) = state
        .store
        .call(move |conn| chat::fork_thread(conn, &id, &message_id, &new_thread_id, created_at))
        .await?;
    Ok(Json(json!({
        "thread": thread,
        "messages": messages,
        "containerSnapshotWarning": null
    })))
}

// ---------- сообщения ----------

pub async fn list_messages(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let messages = state
        .store
        .call(move |conn| {
            if chat::get_thread(conn, &id)?.is_none() {
                return Err(ChatError::NotFound(format!("тред {id}")));
            }
            chat::list_messages(conn, &id)
        })
        .await?;
    Ok(Json(json!({ "messages": messages })))
}

/// `POST /api/chat/threads/:id/messages`: одно сообщение.
pub async fn add_message(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(record): Json<Value>,
) -> ApiResult<Json<Value>> {
    let saved = state
        .store
        .call(move |conn| chat::save_message(conn, &id, record))
        .await?;
    Ok(Json(saved))
}

/// `PUT /api/chat/threads/:id/messages` с телом `{ messages, pruneMissing, deletedMessageIds }`.
pub async fn sync_messages(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    payload: Option<Json<Value>>,
) -> ApiResult<Json<Value>> {
    let body = body_or_empty(payload);
    let messages = match &body {
        Value::Array(items) => items.clone(),
        _ => body
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
    };
    let prune_missing = body
        .get("pruneMissing")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let deleted_ids = string_list(body.get("deletedMessageIds"));
    let saved = state
        .store
        .call(move |conn| chat::sync_messages(conn, &id, messages, prune_missing, &deleted_ids))
        .await?;
    Ok(Json(json!({ "messages": saved })))
}

pub async fn get_message(
    State(state): State<Arc<AppState>>,
    Path((thread_id, message_id)): Path<(String, String)>,
) -> ApiResult<Json<Value>> {
    let lookup_message = message_id.clone();
    state
        .store
        .call(move |conn| chat::get_message(conn, &thread_id, &lookup_message))
        .await?
        .map(Json)
        .ok_or_else(|| ApiError::not_found(format!("Сообщение {message_id} не найдено")))
}

/// `PUT /api/chat/threads/:id/messages/:msg_id`: сохраняет сообщение целиком.
pub async fn put_message(
    State(state): State<Arc<AppState>>,
    Path((thread_id, message_id)): Path<(String, String)>,
    Json(mut record): Json<Value>,
) -> ApiResult<Json<Value>> {
    if let Some(object) = record.as_object_mut() {
        // id берётся из адреса запроса: тело не может переписать другое сообщение
        object.insert("id".into(), json!(message_id));
    }
    let saved = state
        .store
        .call(move |conn| chat::save_message(conn, &thread_id, record))
        .await?;
    Ok(Json(saved))
}

/// `PATCH /api/chat/threads/:id/messages/:msg_id`: дописывает поля в существующее сообщение.
pub async fn patch_message(
    State(state): State<Arc<AppState>>,
    Path((thread_id, message_id)): Path<(String, String)>,
    Json(patch): Json<Value>,
) -> ApiResult<Json<Value>> {
    let saved = state
        .store
        .call(move |conn| {
            let mut record = chat::get_message(conn, &thread_id, &message_id)?
                .ok_or_else(|| ChatError::NotFound(format!("сообщение {message_id}")))?;
            if let (Some(target), Value::Object(fields)) = (record.as_object_mut(), patch) {
                for (key, value) in fields {
                    if key != "id" {
                        target.insert(key, value);
                    }
                }
            }
            chat::save_message(conn, &thread_id, record)
        })
        .await?;
    Ok(Json(saved))
}

pub async fn delete_message(
    State(state): State<Arc<AppState>>,
    Path((thread_id, message_id)): Path<(String, String)>,
) -> ApiResult<Json<Value>> {
    let lookup_message = message_id.clone();
    let deleted = state
        .store
        .call(move |conn| chat::delete_message(conn, &thread_id, &lookup_message))
        .await?;
    if !deleted {
        return Err(ApiError::not_found(format!(
            "Сообщение {message_id} не найдено"
        )));
    }
    Ok(Json(json!({ "status": "ok" })))
}

// ---------- вся история ----------

pub async fn count_threads(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let count = state.store.call(|conn| chat::count_threads(conn)).await?;
    Ok(Json(json!({ "count": count })))
}

/// `DELETE /api/chat` («очистить все чаты») с телом `{ ids, operationId }`.
pub async fn clear_all(
    State(state): State<Arc<AppState>>,
    payload: Option<Json<Value>>,
) -> ApiResult<Json<Value>> {
    let body = body_or_empty(payload);
    let extra_tombstones = string_list(body.get("ids"));
    let deleted = state
        .store
        .call(move |conn| chat::clear_all_threads(conn, &extra_tombstones))
        .await?;
    Ok(Json(
        json!({ "deletedThreadIds": deleted, "sandboxes_kept": [] }),
    ))
}

pub async fn export_history(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let exported = state.store.call(|conn| chat::export_all(conn)).await?;
    Ok(Json(exported))
}

/// Вложения пока не сохраняются (этап 4), поэтому список честно пустой.
pub async fn list_attachments() -> Json<Value> {
    Json(json!({ "attachments": [], "nextOffset": null }))
}

// ---------- проекты ----------

#[derive(Debug, Default, Deserialize)]
pub struct ProjectListQuery {
    include_archived: Option<bool>,
}

pub async fn list_projects(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ProjectListQuery>,
) -> ApiResult<Json<Value>> {
    let include_archived = query.include_archived.unwrap_or(false);
    let projects = state
        .store
        .call(move |conn| chat::list_projects(conn, include_archived))
        .await?;
    Ok(Json(json!({ "projects": projects })))
}

pub async fn get_project(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let lookup_id = id.clone();
    state
        .store
        .call(move |conn| chat::get_project(conn, &lookup_id))
        .await?
        .map(Json)
        .ok_or_else(|| ApiError::not_found(format!("Проект {id} не найден")))
}

pub async fn save_project(
    State(state): State<Arc<AppState>>,
    Json(record): Json<Value>,
) -> ApiResult<Json<Value>> {
    let saved = state
        .store
        .call(move |conn| chat::save_project(conn, record))
        .await?;
    Ok(Json(saved))
}

pub async fn update_project(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(patch): Json<Value>,
) -> ApiResult<Json<Value>> {
    let updated = state
        .store
        .call(move |conn| chat::patch_project(conn, &id, patch))
        .await?;
    Ok(Json(updated))
}

pub async fn delete_project(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let mut deleted = state
        .store
        .call(move |conn| chat::delete_project(conn, &id))
        .await?;
    if let Some(object) = deleted.as_object_mut() {
        object.insert("sandboxes_kept".into(), json!([]));
    }
    Ok(Json(deleted))
}

// ---------- настройки чата ----------

pub async fn get_settings(State(state): State<Arc<AppState>>) -> ApiResult<Json<Value>> {
    let current = state
        .store
        .call(|conn| settings::get_object(conn, CHAT_SETTINGS_KEY))
        .await?;
    Ok(Json(json!({ "settings": current })))
}

/// `PUT /api/chat/settings`: дописывает присланные поля (`null` удаляет поле).
pub async fn put_settings(
    State(state): State<Arc<AppState>>,
    Json(patch): Json<Value>,
) -> ApiResult<Json<Value>> {
    let Value::Object(patch) = patch else {
        return Err(ApiError::bad_request(
            "Настройки чата должны быть JSON-объектом",
        ));
    };
    let merged = state
        .store
        .call(move |conn| settings::merge_object(conn, CHAT_SETTINGS_KEY, &patch))
        .await?;
    Ok(Json(json!({ "settings": merged })))
}

/// `POST /api/chat/settings/compare-and-set`: запись применяется, только если настройки
/// ещё совпадают с тем, что видела вкладка браузера.
pub async fn compare_and_set_settings(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> ApiResult<Json<Value>> {
    let expectations = Expectations {
        expected: object_field(&body, "expected"),
        expected_absent: string_list(body.get("expectedAbsent")),
        expected_absent_paths: body
            .get("expectedAbsentPaths")
            .and_then(Value::as_array)
            .map(|paths| {
                paths
                    .iter()
                    .map(|path| string_list(Some(path)))
                    .filter(|path| !path.is_empty())
                    .collect()
            })
            .unwrap_or_default(),
    };
    let patch = object_field(&body, "patch");
    let (current, applied) = state
        .store
        .call(move |conn| settings::compare_and_set(conn, CHAT_SETTINGS_KEY, &expectations, &patch))
        .await?;
    Ok(Json(json!({ "settings": current, "applied": applied })))
}
