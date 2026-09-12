//! Треды, сообщения и проекты чата в формате фронтенда
//! (`ThreadRecord`, `MessageRecord`, `ProjectRecord`).
//!
//! Записи хранятся целиком как JSON: фронтенд со временем добавляет в них новые поля
//! (настройки треда, id контейнеров и т. п.), и сервер не должен их терять. Отдельными
//! колонками вынесены только поля для поиска и сортировки.
//!
//! Удалённые треды запоминаются («надгробия»): если автосохранение из браузера
//! прилетит уже после удаления, тред не воскреснет, а фронтенд получит 410.

use super::{now_ms, StoreError};
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Ошибка операций с чатом.
#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    #[error("Тред {0} удалён")]
    Deleted(String),
    #[error("Не найдено: {0}")]
    NotFound(String),
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for ChatError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Store(err.into())
    }
}

impl From<serde_json::Error> for ChatError {
    fn from(err: serde_json::Error) -> Self {
        Self::Store(err.into())
    }
}

impl From<ChatError> for crate::error::ApiError {
    fn from(err: ChatError) -> Self {
        use crate::error::ApiError;
        use axum::http::StatusCode;
        match err {
            ChatError::Deleted(_) => ApiError::new(StatusCode::GONE, err.to_string()),
            ChatError::NotFound(_) => ApiError::not_found(err.to_string()),
            ChatError::Conflict(detail) => ApiError::conflict(detail),
            ChatError::Invalid(detail) => ApiError::bad_request(detail),
            ChatError::Store(store) => store.into(),
        }
    }
}

pub type ChatResult<T> = Result<T, ChatError>;

/// Служебные поля `PATCH /threads/:id`, которые управляют записью, но в тред не сохраняются.
const THREAD_PATCH_CONTROL_FIELDS: &[&str] = &[
    "id",
    "expectedTitle",
    "expectedOpeningMessageId",
    "settingsPatch",
    "settingsSeq",
    "settingsWriter",
];

/// Поля контейнеров внешних провайдеров: у форка свой контейнер, чужой id не копируется.
const CONTAINER_FIELDS: &[&str] = &["openaiCodeExecContainerId", "anthropicCodeExecContainerId"];

/// Фильтр списка тредов (query-параметры `GET /api/chat/threads`).
#[derive(Debug, Default, Clone)]
pub struct ThreadFilter {
    pub model_type: Option<String>,
    pub pair_id: Option<String>,
    pub project_id: Option<String>,
    pub include_archived: bool,
}

// ---------- общие помощники ----------

fn as_object(value: Value, what: &str) -> ChatResult<Map<String, Value>> {
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(ChatError::Invalid(format!("{what}: ожидался JSON-объект"))),
    }
}

fn required_id(record: &Map<String, Value>, what: &str) -> ChatResult<String> {
    record
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .ok_or_else(|| ChatError::Invalid(format!("{what}: не указан id")))
}

fn str_field<'a>(record: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    record.get(key).and_then(Value::as_str)
}

fn parse_record(text: &str) -> ChatResult<Value> {
    Ok(serde_json::from_str(text)?)
}

fn collect_records(
    statement: &mut rusqlite::Statement<'_>,
    args: impl rusqlite::Params,
) -> ChatResult<Vec<Value>> {
    let rows = statement.query_map(args, |row| row.get::<_, String>(0))?;
    let mut records = Vec::new();
    for text in rows {
        records.push(parse_record(&text?)?);
    }
    Ok(records)
}

fn is_thread_deleted(conn: &Connection, id: &str) -> ChatResult<bool> {
    Ok(conn
        .query_row(
            "SELECT 1 FROM deleted_threads WHERE id = ?1",
            params![id],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

fn ensure_thread_writable(conn: &Connection, id: &str) -> ChatResult<()> {
    if is_thread_deleted(conn, id)? {
        return Err(ChatError::Deleted(id.to_string()));
    }
    if get_thread(conn, id)?.is_none() {
        return Err(ChatError::NotFound(format!("тред {id}")));
    }
    Ok(())
}

// ---------- треды ----------

fn write_thread(conn: &Connection, record: &Map<String, Value>) -> ChatResult<()> {
    let id = required_id(record, "тред")?;
    let created_at = record
        .get("createdAt")
        .and_then(Value::as_i64)
        .unwrap_or_else(now_ms);
    let archived = record
        .get("archived")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    conn.execute(
        "INSERT INTO threads (id, record, created_at, archived, project_id, pair_id, model_type)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(id) DO UPDATE SET
            record = excluded.record, created_at = excluded.created_at,
            archived = excluded.archived, project_id = excluded.project_id,
            pair_id = excluded.pair_id, model_type = excluded.model_type",
        params![
            id,
            serde_json::to_string(record)?,
            created_at,
            archived,
            str_field(record, "projectId"),
            str_field(record, "pairId"),
            str_field(record, "modelType"),
        ],
    )?;
    Ok(())
}

/// Список тредов (новые первыми) по фильтру.
pub fn list_threads(conn: &Connection, filter: &ThreadFilter) -> ChatResult<Vec<Value>> {
    let mut sql = String::from("SELECT record FROM threads WHERE 1 = 1");
    let mut args: Vec<&str> = Vec::new();
    if !filter.include_archived {
        sql.push_str(" AND archived = 0");
    }
    for (column, value) in [
        ("model_type", &filter.model_type),
        ("pair_id", &filter.pair_id),
        ("project_id", &filter.project_id),
    ] {
        if let Some(value) = value {
            // Имя колонки — наша константа из списка выше, значение передаётся параметром
            sql.push_str(&format!(" AND {column} = ?"));
            args.push(value);
        }
    }
    sql.push_str(" ORDER BY created_at DESC, id");
    let mut statement = conn.prepare(&sql)?;
    collect_records(&mut statement, params_from_iter(args))
}

/// Тред по id.
pub fn get_thread(conn: &Connection, id: &str) -> ChatResult<Option<Value>> {
    let text: Option<String> = conn
        .query_row(
            "SELECT record FROM threads WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )
        .optional()?;
    text.map(|text| parse_record(&text)).transpose()
}

/// Сохраняет тред целиком (`POST /threads`) и возвращает сохранённую запись.
pub fn save_thread(conn: &Connection, record: Value) -> ChatResult<Value> {
    let mut record = as_object(record, "тред")?;
    let id = required_id(&record, "тред")?;
    if is_thread_deleted(conn, &id)? {
        return Err(ChatError::Deleted(id));
    }
    let existing_created_at = get_thread(conn, &id)?
        .and_then(|existing| existing.get("createdAt").and_then(Value::as_i64));
    if !record.get("createdAt").is_some_and(Value::is_i64) {
        record.insert(
            "createdAt".into(),
            json!(existing_created_at.unwrap_or_else(now_ms)),
        );
    }
    record.entry("title").or_insert_with(|| json!(""));
    record.entry("archived").or_insert(json!(false));
    record.entry("updatedAt").or_insert_with(|| json!(now_ms()));
    write_thread(conn, &record)?;
    Ok(Value::Object(record))
}

fn opening_user_message_id(conn: &Connection, thread_id: &str) -> ChatResult<Option<String>> {
    Ok(list_messages(conn, thread_id)?
        .into_iter()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(|message| {
            message
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
        }))
}

/// Частично обновляет тред (`PATCH /threads/:id`).
///
/// - `settings` заменяет настройки треда целиком, `settingsPatch` дописывает поля;
/// - `expectedTitle` / `expectedOpeningMessageId` — условия: при несовпадении 409.
pub fn patch_thread(conn: &mut Connection, id: &str, patch: Value) -> ChatResult<Value> {
    let patch = as_object(patch, "изменения треда")?;
    let tx = conn.transaction()?;
    if is_thread_deleted(&tx, id)? {
        return Err(ChatError::Deleted(id.to_string()));
    }
    let mut record = match get_thread(&tx, id)? {
        Some(existing) => as_object(existing, "тред")?,
        None => return Err(ChatError::NotFound(format!("тред {id}"))),
    };

    if let Some(expected) = patch.get("expectedTitle").and_then(Value::as_str) {
        if str_field(&record, "title") != Some(expected) {
            return Err(ChatError::Conflict(
                "Название треда уже изменилось в другой вкладке".to_string(),
            ));
        }
    }
    if let Some(expected) = patch
        .get("expectedOpeningMessageId")
        .and_then(Value::as_str)
    {
        if opening_user_message_id(&tx, id)?.as_deref() != Some(expected) {
            return Err(ChatError::Conflict(
                "Первое сообщение треда уже изменилось".to_string(),
            ));
        }
    }

    for (key, value) in &patch {
        if !THREAD_PATCH_CONTROL_FIELDS.contains(&key.as_str()) {
            record.insert(key.clone(), value.clone());
        }
    }
    if let Some(Value::Object(settings_patch)) = patch.get("settingsPatch") {
        let settings = record
            .entry("settings")
            .or_insert_with(|| Value::Object(Map::new()));
        if !settings.is_object() {
            *settings = Value::Object(Map::new());
        }
        if let Value::Object(settings) = settings {
            for (key, value) in settings_patch {
                settings.insert(key.clone(), value.clone());
            }
        }
    }
    if !patch.get("updatedAt").is_some_and(Value::is_i64) {
        record.insert("updatedAt".into(), json!(now_ms()));
    }

    write_thread(&tx, &record)?;
    tx.commit()?;
    Ok(Value::Object(record))
}

fn tombstone(conn: &Connection, id: &str) -> ChatResult<()> {
    conn.execute(
        "INSERT OR REPLACE INTO deleted_threads (id, deleted_at) VALUES (?1, ?2)",
        params![id, now_ms()],
    )?;
    Ok(())
}

/// Удаляет треды вместе с сообщениями и запоминает их id. Возвращает реально удалённые.
pub fn delete_threads(conn: &mut Connection, ids: &[String]) -> ChatResult<Vec<String>> {
    let tx = conn.transaction()?;
    let mut deleted = Vec::new();
    for id in ids {
        if tx.execute("DELETE FROM threads WHERE id = ?1", params![id])? > 0 {
            deleted.push(id.clone());
        }
        tombstone(&tx, id)?;
    }
    tx.commit()?;
    Ok(deleted)
}

/// Удаляет все треды («очистить все чаты»). `extra_tombstones` — id тредов, которые
/// браузер ещё не успел сохранить, но уже удалил.
pub fn clear_all_threads(
    conn: &mut Connection,
    extra_tombstones: &[String],
) -> ChatResult<Vec<String>> {
    let ids: Vec<String> = {
        let mut statement = conn.prepare("SELECT id FROM threads")?;
        let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
        rows.collect::<Result<_, _>>()?
    };
    let tx = conn.transaction()?;
    tx.execute("DELETE FROM threads", [])?;
    for id in ids.iter().chain(extra_tombstones) {
        tombstone(&tx, id)?;
    }
    tx.commit()?;
    Ok(ids)
}

/// Число тредов (включая архивные).
pub fn count_threads(conn: &Connection) -> ChatResult<u64> {
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM threads", [], |row| row.get(0))?;
    Ok(u64::try_from(count).unwrap_or(0))
}

/// Сколько форков сделано из каждого сообщения треда.
pub fn fork_counts(conn: &Connection, thread_id: &str) -> ChatResult<BTreeMap<String, u64>> {
    let mut statement = conn.prepare(
        "SELECT json_extract(record, '$.forkedFromMessageId') AS message_id, COUNT(*)
         FROM threads
         WHERE json_extract(record, '$.forkedFromThreadId') = ?1 AND message_id IS NOT NULL
         GROUP BY message_id",
    )?;
    let rows = statement.query_map(params![thread_id], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })?;
    let mut counts = BTreeMap::new();
    for row in rows {
        let (message_id, count) = row?;
        counts.insert(message_id, u64::try_from(count).unwrap_or(0));
    }
    Ok(counts)
}

/// Форк треда от сообщения: новый тред с копией сообщений до `message_id` включительно.
pub fn fork_thread(
    conn: &mut Connection,
    source_id: &str,
    message_id: &str,
    new_thread_id: &str,
    created_at: i64,
) -> ChatResult<(Value, Vec<Value>)> {
    let tx = conn.transaction()?;
    let source = match get_thread(&tx, source_id)? {
        Some(source) => as_object(source, "тред")?,
        None => return Err(ChatError::NotFound(format!("тред {source_id}"))),
    };
    if new_thread_id.trim().is_empty() {
        return Err(ChatError::Invalid("Не указан newThreadId".to_string()));
    }
    if is_thread_deleted(&tx, new_thread_id)? {
        return Err(ChatError::Deleted(new_thread_id.to_string()));
    }
    if get_thread(&tx, new_thread_id)?.is_some() {
        return Err(ChatError::Conflict(format!(
            "Тред {new_thread_id} уже существует"
        )));
    }

    let messages = list_messages(&tx, source_id)?;
    let cut = messages
        .iter()
        .position(|message| message.get("id").and_then(Value::as_str) == Some(message_id))
        .ok_or_else(|| ChatError::NotFound(format!("сообщение {message_id}")))?;

    let mut thread = source;
    thread.insert("id".into(), json!(new_thread_id));
    thread.insert("createdAt".into(), json!(created_at));
    thread.insert("updatedAt".into(), json!(created_at));
    thread.insert("forkedFromThreadId".into(), json!(source_id));
    thread.insert("forkedFromMessageId".into(), json!(message_id));
    for field in CONTAINER_FIELDS {
        thread.insert((*field).into(), Value::Null);
    }
    write_thread(&tx, &thread)?;

    let mut copied = Vec::with_capacity(cut + 1);
    for message in messages.into_iter().take(cut + 1) {
        let mut message = as_object(message, "сообщение")?;
        message.insert("threadId".into(), json!(new_thread_id));
        write_message(&tx, new_thread_id, &message)?;
        copied.push(Value::Object(message));
    }
    tx.commit()?;
    Ok((Value::Object(thread), copied))
}

// ---------- сообщения ----------

fn write_message(
    conn: &Connection,
    thread_id: &str,
    record: &Map<String, Value>,
) -> ChatResult<()> {
    let id = required_id(record, "сообщение")?;
    let created_at = record
        .get("createdAt")
        .and_then(Value::as_i64)
        .unwrap_or_else(now_ms);
    conn.execute(
        "INSERT INTO messages (thread_id, id, record, created_at) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(thread_id, id) DO UPDATE SET record = excluded.record, created_at = excluded.created_at",
        params![thread_id, id, serde_json::to_string(record)?, created_at],
    )?;
    Ok(())
}

fn normalize_message(thread_id: &str, record: Value) -> ChatResult<Map<String, Value>> {
    let mut record = as_object(record, "сообщение")?;
    required_id(&record, "сообщение")?;
    // Тред задаётся адресом запроса, а не телом: сообщение нельзя «переложить» в чужой тред
    record.insert("threadId".into(), json!(thread_id));
    if !record.get("createdAt").is_some_and(Value::is_i64) {
        record.insert("createdAt".into(), json!(now_ms()));
    }
    Ok(record)
}

/// Сообщения треда в порядке создания.
pub fn list_messages(conn: &Connection, thread_id: &str) -> ChatResult<Vec<Value>> {
    let mut statement = conn
        .prepare("SELECT record FROM messages WHERE thread_id = ?1 ORDER BY created_at, rowid")?;
    collect_records(&mut statement, params![thread_id])
}

/// Сообщение по id.
pub fn get_message(conn: &Connection, thread_id: &str, id: &str) -> ChatResult<Option<Value>> {
    let text: Option<String> = conn
        .query_row(
            "SELECT record FROM messages WHERE thread_id = ?1 AND id = ?2",
            params![thread_id, id],
            |row| row.get(0),
        )
        .optional()?;
    text.map(|text| parse_record(&text)).transpose()
}

/// Сохраняет одно сообщение и возвращает сохранённую запись.
pub fn save_message(conn: &Connection, thread_id: &str, record: Value) -> ChatResult<Value> {
    ensure_thread_writable(conn, thread_id)?;
    let record = normalize_message(thread_id, record)?;
    write_message(conn, thread_id, &record)?;
    Ok(Value::Object(record))
}

/// Сохраняет пачку сообщений (`PUT /threads/:id/messages`).
///
/// `prune_missing` удаляет сообщения треда, которых нет в пачке (так фронтенд удаляет
/// сообщения), `deleted_ids` удаляются явно. Возвращает итоговый список сообщений.
pub fn sync_messages(
    conn: &mut Connection,
    thread_id: &str,
    messages: Vec<Value>,
    prune_missing: bool,
    deleted_ids: &[String],
) -> ChatResult<Vec<Value>> {
    let tx = conn.transaction()?;
    ensure_thread_writable(&tx, thread_id)?;

    let mut kept = HashSet::new();
    for message in messages {
        let record = normalize_message(thread_id, message)?;
        kept.insert(required_id(&record, "сообщение")?);
        write_message(&tx, thread_id, &record)?;
    }
    for id in deleted_ids {
        tx.execute(
            "DELETE FROM messages WHERE thread_id = ?1 AND id = ?2",
            params![thread_id, id],
        )?;
    }
    if prune_missing {
        let existing: Vec<String> = {
            let mut statement = tx.prepare("SELECT id FROM messages WHERE thread_id = ?1")?;
            let rows = statement.query_map(params![thread_id], |row| row.get::<_, String>(0))?;
            rows.collect::<Result<_, _>>()?
        };
        for id in existing.iter().filter(|id| !kept.contains(*id)) {
            tx.execute(
                "DELETE FROM messages WHERE thread_id = ?1 AND id = ?2",
                params![thread_id, id],
            )?;
        }
    }

    let result = list_messages(&tx, thread_id)?;
    tx.commit()?;
    Ok(result)
}

/// Удаляет сообщение. Возвращает `true`, если оно было.
pub fn delete_message(conn: &Connection, thread_id: &str, id: &str) -> ChatResult<bool> {
    Ok(conn.execute(
        "DELETE FROM messages WHERE thread_id = ?1 AND id = ?2",
        params![thread_id, id],
    )? > 0)
}

// ---------- проекты ----------

fn write_project(conn: &Connection, record: &Map<String, Value>) -> ChatResult<()> {
    let id = required_id(record, "проект")?;
    conn.execute(
        "INSERT INTO projects (id, record, created_at, archived) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(id) DO UPDATE SET record = excluded.record,
            created_at = excluded.created_at, archived = excluded.archived",
        params![
            id,
            serde_json::to_string(record)?,
            record
                .get("createdAt")
                .and_then(Value::as_i64)
                .unwrap_or_else(now_ms),
            record
                .get("archived")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        ],
    )?;
    Ok(())
}

/// Список проектов (новые первыми).
pub fn list_projects(conn: &Connection, include_archived: bool) -> ChatResult<Vec<Value>> {
    let sql = if include_archived {
        "SELECT record FROM projects ORDER BY created_at DESC, id"
    } else {
        "SELECT record FROM projects WHERE archived = 0 ORDER BY created_at DESC, id"
    };
    let mut statement = conn.prepare(sql)?;
    collect_records(&mut statement, [])
}

/// Проект по id.
pub fn get_project(conn: &Connection, id: &str) -> ChatResult<Option<Value>> {
    let text: Option<String> = conn
        .query_row(
            "SELECT record FROM projects WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )
        .optional()?;
    text.map(|text| parse_record(&text)).transpose()
}

/// Сохраняет проект целиком.
pub fn save_project(conn: &Connection, record: Value) -> ChatResult<Value> {
    let mut record = as_object(record, "проект")?;
    required_id(&record, "проект")?;
    let now = now_ms();
    record.entry("name").or_insert_with(|| json!(""));
    record.entry("archived").or_insert(json!(false));
    record.entry("createdAt").or_insert(json!(now));
    record.entry("updatedAt").or_insert(json!(now));
    write_project(conn, &record)?;
    Ok(Value::Object(record))
}

/// Частично обновляет проект.
pub fn patch_project(conn: &Connection, id: &str, patch: Value) -> ChatResult<Value> {
    let patch = as_object(patch, "изменения проекта")?;
    let mut record = match get_project(conn, id)? {
        Some(existing) => as_object(existing, "проект")?,
        None => return Err(ChatError::NotFound(format!("проект {id}"))),
    };
    for (key, value) in patch {
        if key != "id" {
            record.insert(key, value);
        }
    }
    record.insert("updatedAt".into(), json!(now_ms()));
    write_project(conn, &record)?;
    Ok(Value::Object(record))
}

/// Удаляет проект; его треды остаются, но перестают к нему относиться.
/// Возвращает удалённую запись.
pub fn delete_project(conn: &mut Connection, id: &str) -> ChatResult<Value> {
    let tx = conn.transaction()?;
    let record =
        get_project(&tx, id)?.ok_or_else(|| ChatError::NotFound(format!("проект {id}")))?;
    tx.execute("DELETE FROM projects WHERE id = ?1", params![id])?;
    tx.execute(
        "UPDATE threads SET record = json_set(record, '$.projectId', json('null')), project_id = NULL
         WHERE project_id = ?1",
        params![id],
    )?;
    tx.commit()?;
    Ok(record)
}

// ---------- экспорт и статистика ----------

/// Все треды по id (включая архивные).
pub fn all_threads(conn: &Connection) -> ChatResult<HashMap<String, Value>> {
    let threads = list_threads(
        conn,
        &ThreadFilter {
            include_archived: true,
            ..ThreadFilter::default()
        },
    )?;
    Ok(threads
        .into_iter()
        .filter_map(|thread| {
            let id = thread.get("id").and_then(Value::as_str)?.to_string();
            Some((id, thread))
        })
        .collect())
}

/// Все сообщения, сгруппированные по id треда.
pub fn all_messages(conn: &Connection) -> ChatResult<HashMap<String, Vec<Value>>> {
    let mut statement =
        conn.prepare("SELECT thread_id, record FROM messages ORDER BY created_at, rowid")?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut grouped: HashMap<String, Vec<Value>> = HashMap::new();
    for row in rows {
        let (thread_id, text) = row?;
        grouped
            .entry(thread_id)
            .or_default()
            .push(parse_record(&text)?);
    }
    Ok(grouped)
}

/// Полный экспорт истории чатов в формате `buildBackendChatExport` фронтенда.
pub fn export_all(conn: &Connection) -> ChatResult<Value> {
    let threads = list_threads(
        conn,
        &ThreadFilter {
            include_archived: true,
            ..ThreadFilter::default()
        },
    )?;
    let mut messages = Vec::new();
    for thread in &threads {
        if let Some(id) = thread.get("id").and_then(Value::as_str) {
            messages.extend(list_messages(conn, id)?);
        }
    }
    Ok(json!({
        "exportedAt": crate::state::iso_now(),
        "version": 1,
        "threadCount": threads.len(),
        "projects": list_projects(conn, true)?,
        "threads": threads,
        "messages": messages
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    fn with_store<T>(work: impl FnOnce(&mut Connection) -> ChatResult<T>) -> T {
        Store::open_in_memory()
            .expect("база в памяти не открылась")
            .with_conn(work)
            .expect("операция с чатом не удалась")
    }

    fn thread(id: &str, created_at: i64) -> Value {
        json!({ "id": id, "title": format!("Чат {id}"), "modelType": "base", "archived": false, "createdAt": created_at })
    }

    fn message(id: &str, role: &str, created_at: i64) -> Value {
        json!({ "id": id, "role": role, "content": [{ "type": "text", "text": id }], "createdAt": created_at })
    }

    #[test]
    fn saves_full_thread_record_and_lists_newest_first() {
        with_store(|conn| {
            let mut record = thread("t1", 100);
            record["pairId"] = json!("pair-1");
            record["settings"] = json!({ "temperature": 0.3 });
            save_thread(conn, record)?;
            save_thread(conn, thread("t2", 200))?;

            let saved = get_thread(conn, "t1")?.expect("тред не сохранился");
            assert_eq!(saved["pairId"], "pair-1");
            assert_eq!(saved["settings"]["temperature"], 0.3);

            let ids: Vec<Value> = list_threads(conn, &ThreadFilter::default())?
                .into_iter()
                .map(|t| t["id"].clone())
                .collect();
            assert_eq!(ids, [json!("t2"), json!("t1")]);

            let by_pair = list_threads(
                conn,
                &ThreadFilter {
                    pair_id: Some("pair-1".into()),
                    ..ThreadFilter::default()
                },
            )?;
            assert_eq!(by_pair.len(), 1);
            Ok(())
        });
    }

    #[test]
    fn archived_threads_hidden_unless_requested() {
        with_store(|conn| {
            let mut archived = thread("old", 1);
            archived["archived"] = json!(true);
            save_thread(conn, archived)?;
            save_thread(conn, thread("new", 2))?;
            assert_eq!(list_threads(conn, &ThreadFilter::default())?.len(), 1);
            let all = ThreadFilter {
                include_archived: true,
                ..ThreadFilter::default()
            };
            assert_eq!(list_threads(conn, &all)?.len(), 2);
            Ok(())
        });
    }

    #[test]
    fn patch_merges_settings_and_checks_expectations() {
        with_store(|conn| {
            save_thread(conn, thread("t1", 1))?;
            save_message(conn, "t1", message("m1", "user", 10))?;

            let patched = patch_thread(
                conn,
                "t1",
                json!({ "title": "Новое", "settingsPatch": { "topK": 40 }, "settingsSeq": 3 }),
            )?;
            assert_eq!(patched["title"], "Новое");
            assert_eq!(patched["settings"]["topK"], 40);
            assert!(
                patched.get("settingsSeq").is_none(),
                "служебные поля не сохраняются"
            );

            let conflict = patch_thread(
                conn,
                "t1",
                json!({ "title": "X", "expectedTitle": "Старое" }),
            );
            assert!(matches!(conflict, Err(ChatError::Conflict(_))));

            let ok = patch_thread(
                conn,
                "t1",
                json!({ "title": "Y", "expectedOpeningMessageId": "m1" }),
            )?;
            assert_eq!(ok["title"], "Y");

            assert!(matches!(
                patch_thread(conn, "missing", json!({ "title": "Z" })),
                Err(ChatError::NotFound(_))
            ));
            Ok(())
        });
    }

    #[test]
    fn deleted_thread_cannot_be_resurrected() {
        with_store(|conn| {
            save_thread(conn, thread("t1", 1))?;
            save_message(conn, "t1", message("m1", "user", 1))?;
            assert_eq!(
                delete_threads(conn, &["t1".into(), "never-saved".into()])?,
                ["t1"]
            );

            assert!(
                list_messages(conn, "t1")?.is_empty(),
                "сообщения удаляются вместе с тредом"
            );
            assert!(matches!(
                save_thread(conn, thread("t1", 1)),
                Err(ChatError::Deleted(_))
            ));
            assert!(matches!(
                save_thread(conn, thread("never-saved", 1)),
                Err(ChatError::Deleted(_))
            ));
            Ok(())
        });
    }

    #[test]
    fn sync_prunes_and_deletes_messages() {
        with_store(|conn| {
            save_thread(conn, thread("t1", 1))?;
            sync_messages(
                conn,
                "t1",
                vec![
                    message("a", "user", 1),
                    message("b", "assistant", 2),
                    message("c", "user", 3),
                ],
                false,
                &[],
            )?;

            let after_delete = sync_messages(conn, "t1", vec![], false, &["b".into()])?;
            assert_eq!(after_delete.len(), 2);

            let pruned = sync_messages(conn, "t1", vec![message("c", "user", 3)], true, &[])?;
            assert_eq!(pruned.len(), 1);
            assert_eq!(pruned[0]["id"], "c");
            assert_eq!(pruned[0]["threadId"], "t1");

            assert!(matches!(
                sync_messages(conn, "missing", vec![], false, &[]),
                Err(ChatError::NotFound(_))
            ));
            Ok(())
        });
    }

    #[test]
    fn fork_copies_messages_up_to_branch_point() {
        with_store(|conn| {
            let mut source = thread("src", 1);
            source["openaiCodeExecContainerId"] = json!("container-1");
            save_thread(conn, source)?;
            sync_messages(
                conn,
                "src",
                vec![
                    message("m1", "user", 1),
                    message("m2", "assistant", 2),
                    message("m3", "user", 3),
                ],
                false,
                &[],
            )?;

            let (forked, messages) = fork_thread(conn, "src", "m2", "fork", 50)?;
            assert_eq!(forked["forkedFromThreadId"], "src");
            assert_eq!(forked["forkedFromMessageId"], "m2");
            assert_eq!(forked["openaiCodeExecContainerId"], Value::Null);
            assert_eq!(messages.len(), 2);
            assert!(messages.iter().all(|m| m["threadId"] == "fork"));
            assert_eq!(
                list_messages(conn, "src")?.len(),
                3,
                "исходный тред не меняется"
            );

            assert_eq!(fork_counts(conn, "src")?.get("m2"), Some(&1));
            assert!(matches!(
                fork_thread(conn, "src", "m2", "fork", 60),
                Err(ChatError::Conflict(_))
            ));
            assert!(matches!(
                fork_thread(conn, "src", "missing", "fork2", 60),
                Err(ChatError::NotFound(_))
            ));
            Ok(())
        });
    }

    #[test]
    fn clear_all_counts_and_tombstones() {
        with_store(|conn| {
            save_thread(conn, thread("a", 1))?;
            save_thread(conn, thread("b", 2))?;
            assert_eq!(count_threads(conn)?, 2);
            let mut deleted = clear_all_threads(conn, &["pending".into()])?;
            deleted.sort();
            assert_eq!(deleted, ["a", "b"]);
            assert_eq!(count_threads(conn)?, 0);
            assert!(matches!(
                save_thread(conn, thread("pending", 3)),
                Err(ChatError::Deleted(_))
            ));
            Ok(())
        });
    }

    #[test]
    fn deleting_project_detaches_threads() {
        with_store(|conn| {
            save_project(conn, json!({ "id": "p1", "name": "Проект" }))?;
            let mut member = thread("t1", 1);
            member["projectId"] = json!("p1");
            save_thread(conn, member)?;

            let patched = patch_project(conn, "p1", json!({ "instructions": "Отвечай кратко" }))?;
            assert_eq!(patched["instructions"], "Отвечай кратко");

            delete_project(conn, "p1")?;
            assert!(get_project(conn, "p1")?.is_none());
            assert_eq!(
                get_thread(conn, "t1")?.expect("тред остаётся")["projectId"],
                Value::Null
            );
            let in_project = ThreadFilter {
                project_id: Some("p1".into()),
                ..ThreadFilter::default()
            };
            assert!(list_threads(conn, &in_project)?.is_empty());
            Ok(())
        });
    }

    #[test]
    fn export_contains_everything() {
        with_store(|conn| {
            save_thread(conn, thread("t1", 1))?;
            save_message(conn, "t1", message("m1", "user", 1))?;
            let exported = export_all(conn)?;
            assert_eq!(exported["threadCount"], 1);
            assert_eq!(exported["messages"].as_array().map(Vec::len), Some(1));
            assert_eq!(all_messages(conn)?.get("t1").map(Vec::len), Some(1));
            assert!(all_threads(conn)?.contains_key("t1"));
            Ok(())
        });
    }
}
