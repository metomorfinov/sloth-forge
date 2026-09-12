//! История запусков обучения.
//!
//! Раньше список запусков жил в памяти и пропадал при перезапуске. Здесь он хранится
//! постоянно; подключение к машине состояний обучения — шаг 9 этапа 1.

use super::StoreResult;
use crate::state::TrainingRunSummary;
use rusqlite::{params, Connection, OptionalExtension};

/// Сохраняет запуск (новый или обновлённый).
pub fn upsert(conn: &Connection, run: &TrainingRunSummary) -> StoreResult<()> {
    conn.execute(
        "INSERT INTO training_runs (id, record, started_at) VALUES (?1, ?2, ?3)
         ON CONFLICT(id) DO UPDATE SET record = excluded.record, started_at = excluded.started_at",
        params![run.id, serde_json::to_string(run)?, run.started_at],
    )?;
    Ok(())
}

/// Запуск по id.
pub fn get(conn: &Connection, id: &str) -> StoreResult<Option<TrainingRunSummary>> {
    let text: Option<String> = conn
        .query_row(
            "SELECT record FROM training_runs WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )
        .optional()?;
    Ok(text.map(|text| serde_json::from_str(&text)).transpose()?)
}

/// Страница запусков (новые первыми) и общее число запусков.
pub fn list(
    conn: &Connection,
    limit: u32,
    offset: u32,
) -> StoreResult<(Vec<TrainingRunSummary>, u64)> {
    let total: i64 = conn.query_row("SELECT COUNT(*) FROM training_runs", [], |row| row.get(0))?;
    let mut statement = conn.prepare(
        "SELECT record FROM training_runs ORDER BY started_at DESC, id DESC LIMIT ?1 OFFSET ?2",
    )?;
    let rows = statement.query_map(params![limit, offset], |row| row.get::<_, String>(0))?;
    let mut runs = Vec::new();
    for text in rows {
        runs.push(serde_json::from_str(&text?)?);
    }
    Ok((runs, u64::try_from(total).unwrap_or(0)))
}

/// Все запуски (для статистики профиля).
pub fn all(conn: &Connection) -> StoreResult<Vec<TrainingRunSummary>> {
    Ok(list(conn, u32::MAX, 0)?.0)
}

/// Удаляет запуск. Возвращает `true`, если он был.
pub fn delete(conn: &Connection, id: &str) -> StoreResult<bool> {
    Ok(conn.execute("DELETE FROM training_runs WHERE id = ?1", params![id])? > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    fn run(id: &str, started_at: &str) -> TrainingRunSummary {
        TrainingRunSummary {
            id: id.to_string(),
            status: "completed".to_string(),
            model_name: "llama-1b".to_string(),
            project_name: None,
            dataset_name: "alpaca".to_string(),
            display_name: None,
            started_at: started_at.to_string(),
            ended_at: None,
            total_steps: Some(10),
            final_step: Some(10),
            final_loss: Some(1.5),
            output_dir: None,
            can_resume: false,
            resume_blocked_reason: None,
            resumed_later: false,
            has_preview_model: false,
            preview_ref: None,
            preview_sig: None,
            duration_seconds: Some(60),
            error_message: None,
            loss_sparkline: None,
        }
    }

    #[test]
    fn upsert_list_page_and_delete() {
        let store = Store::open_in_memory().unwrap();
        store
            .with_conn(|conn| {
                upsert(conn, &run("a", "2026-09-10T10:00:00Z"))?;
                upsert(conn, &run("b", "2026-09-12T10:00:00Z"))?;
                upsert(conn, &run("c", "2026-09-11T10:00:00Z"))?;

                let (page, total) = list(conn, 2, 0)?;
                assert_eq!(total, 3);
                let ids: Vec<&str> = page.iter().map(|r| r.id.as_str()).collect();
                assert_eq!(ids, ["b", "c"]);

                let mut updated = run("a", "2026-09-10T10:00:00Z");
                updated.status = "stopped".to_string();
                upsert(conn, &updated)?;
                assert_eq!(
                    get(conn, "a")?.map(|r| r.status),
                    Some("stopped".to_string())
                );

                assert!(delete(conn, "a")?);
                assert!(get(conn, "a")?.is_none());
                assert_eq!(all(conn)?.len(), 2);
                StoreResult::Ok(())
            })
            .unwrap();
    }
}
