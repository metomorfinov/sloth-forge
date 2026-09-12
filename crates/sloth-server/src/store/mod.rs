//! Постоянное хранилище SlothForge на SQLite.
//!
//! Треды, сообщения, проекты, настройки и история обучения раньше жили только в
//! оперативной памяти и пропадали при каждом перезапуске сервера. Теперь они
//! хранятся в одном файле базы данных (по умолчанию `data/slothforge.db`).
//!
//! SQLite — встраиваемая база данных: отдельный сервер не нужен, всё лежит в одном
//! файле. Работа с ней синхронная, поэтому обработчики обращаются к хранилищу через
//! [`Store::call`]: запрос выполняется в отдельном потоке и не блокирует сервер.

pub mod chat;
pub mod runs;
pub mod settings;

use rusqlite::Connection;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Ошибка хранилища.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("ошибка базы данных: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("повреждённые данные в базе: {0}")]
    Json(#[from] serde_json::Error),
    #[error("не удалось подготовить папку для базы данных: {0}")]
    Io(#[from] std::io::Error),
    #[error("соединение с базой данных повреждено аварийным завершением другого запроса")]
    Poisoned,
    #[error("фоновая задача базы данных прервана: {0}")]
    Join(String),
}

pub type StoreResult<T> = Result<T, StoreError>;

impl From<StoreError> for crate::error::ApiError {
    fn from(err: StoreError) -> Self {
        tracing::error!("{err}");
        crate::error::ApiError::internal(format!("Ошибка хранилища: {err}"))
    }
}

/// Миграции схемы по порядку. Номер последней применённой хранится в `PRAGMA user_version`,
/// поэтому при обновлении SlothForge существующая база дополняется, а не пересоздаётся.
const MIGRATIONS: &[&str] = &[
    // 1: чат, проекты, настройки, история обучения
    r#"
    CREATE TABLE threads (
        id TEXT PRIMARY KEY,
        record TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        archived INTEGER NOT NULL DEFAULT 0,
        project_id TEXT,
        pair_id TEXT,
        model_type TEXT
    );
    CREATE INDEX threads_created_at ON threads(created_at);
    CREATE INDEX threads_project_id ON threads(project_id);

    CREATE TABLE messages (
        thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
        id TEXT NOT NULL,
        record TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        PRIMARY KEY (thread_id, id)
    );
    CREATE INDEX messages_thread_created ON messages(thread_id, created_at);

    CREATE TABLE deleted_threads (
        id TEXT PRIMARY KEY,
        deleted_at INTEGER NOT NULL
    );

    CREATE TABLE projects (
        id TEXT PRIMARY KEY,
        record TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        archived INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at INTEGER NOT NULL
    );

    CREATE TABLE training_runs (
        id TEXT PRIMARY KEY,
        record TEXT NOT NULL,
        started_at TEXT NOT NULL
    );
    CREATE INDEX training_runs_started_at ON training_runs(started_at);
    "#,
];

/// Соединение с базой. Одно соединение под мьютексом: SlothForge — локальное приложение
/// одного пользователя, и последовательных запросов к SQLite достаточно.
pub struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    /// Открывает (или создаёт) базу в файле и применяет недостающие миграции.
    pub fn open(path: &Path) -> StoreResult<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        // WAL: чтение не ждёт записи; foreign_keys: удаление треда удаляет и его сообщения;
        // busy_timeout: подождать, если файл на мгновение занят
        conn.execute_batch(
            "PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON; PRAGMA busy_timeout = 5000;",
        )?;
        Self::from_connection(conn)
    }

    /// База в памяти: для тестов и для `AppState::new` без файла.
    pub fn open_in_memory() -> StoreResult<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        Self::from_connection(conn)
    }

    fn from_connection(mut conn: Connection) -> StoreResult<Self> {
        migrate(&mut conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Выполняет работу с базой в текущем потоке (для тестов и фоновых потоков).
    pub fn with_conn<T, E>(
        &self,
        work: impl FnOnce(&mut Connection) -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<StoreError>,
    {
        let mut conn = self.conn.lock().map_err(|_| StoreError::Poisoned)?;
        work(&mut conn)
    }

    /// Выполняет работу с базой в отдельном потоке, не блокируя async-потоки сервера.
    pub async fn call<T, E, F>(self: &Arc<Self>, work: F) -> Result<T, E>
    where
        F: FnOnce(&mut Connection) -> Result<T, E> + Send + 'static,
        T: Send + 'static,
        E: From<StoreError> + Send + 'static,
    {
        let store = Arc::clone(self);
        tokio::task::spawn_blocking(move || store.with_conn(work))
            .await
            .map_err(|err| StoreError::Join(err.to_string()))?
    }
}

fn migrate(conn: &mut Connection) -> StoreResult<()> {
    let applied: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    let applied = usize::try_from(applied).unwrap_or(0);
    for (index, sql) in MIGRATIONS.iter().enumerate().skip(applied) {
        let tx = conn.transaction()?;
        tx.execute_batch(sql)?;
        // Номер миграции — наша константа, а не ввод пользователя, поэтому format! здесь безопасен
        tx.execute_batch(&format!("PRAGMA user_version = {}", index + 1))?;
        tx.commit()?;
        tracing::info!("Хранилище: применена миграция {}", index + 1);
    }
    Ok(())
}

/// Текущее время в миллисекундах от 1970-01-01 (так время хранит фронтенд).
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrations_apply_once_and_data_survives_reopen() {
        let dir = tempfile::tempdir().expect("не удалось создать временную папку");
        let path = dir.path().join("nested").join("slothforge.db");

        {
            let store = Store::open(&path).expect("база не открылась");
            store
                .with_conn(|conn| settings::put(conn, "greeting", &serde_json::json!("привет")))
                .expect("запись не удалась");
        }

        // Повторное открытие: миграции не применяются второй раз, данные на месте
        let store = Store::open(&path).expect("база не открылась повторно");
        let value = store
            .with_conn(|conn| settings::get(conn, "greeting"))
            .expect("чтение не удалось");
        assert_eq!(value, Some(serde_json::json!("привет")));

        let version: i64 = store
            .with_conn(|conn| {
                conn.query_row("PRAGMA user_version", [], |row| row.get(0))
                    .map_err(StoreError::from)
            })
            .expect("не удалось прочитать версию схемы");
        assert_eq!(version, MIGRATIONS.len() as i64);
    }
}
