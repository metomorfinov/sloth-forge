//! Настройки: универсальная таблица «ключ → JSON».
//!
//! Одна таблица обслуживает все настройки интерфейса (чат, персонализация, пресеты
//! генерации и т. д.). Раньше большинство settings-эндпоинтов просто возвращали
//! присланное значение обратно и ничего не запоминали.

use super::{now_ms, StoreResult};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{Map, Value};

/// Читает значение по ключу.
pub fn get(conn: &Connection, key: &str) -> StoreResult<Option<Value>> {
    let text: Option<String> = conn
        .query_row(
            "SELECT value FROM settings WHERE key = ?1",
            params![key],
            |row| row.get(0),
        )
        .optional()?;
    Ok(text.map(|text| serde_json::from_str(&text)).transpose()?)
}

/// Записывает значение по ключу (заменяя старое).
pub fn put(conn: &Connection, key: &str, value: &Value) -> StoreResult<()> {
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES (?1, ?2, ?3)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        params![key, serde_json::to_string(value)?, now_ms()],
    )?;
    Ok(())
}

/// Удаляет значение. Возвращает `true`, если оно было.
pub fn delete(conn: &Connection, key: &str) -> StoreResult<bool> {
    Ok(conn.execute("DELETE FROM settings WHERE key = ?1", params![key])? > 0)
}

/// Объект настроек по ключу; отсутствующий или не-объект считается пустым объектом.
pub fn get_object(conn: &Connection, key: &str) -> StoreResult<Map<String, Value>> {
    Ok(match get(conn, key)? {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    })
}

/// Применяет `patch` к объекту: ключи из patch заменяют старые значения, `null` удаляет ключ.
fn apply_patch(target: &mut Map<String, Value>, patch: &Map<String, Value>) {
    for (key, value) in patch {
        if value.is_null() {
            target.remove(key);
        } else {
            target.insert(key.clone(), value.clone());
        }
    }
}

/// Сливает `patch` в объект настроек и возвращает результат.
pub fn merge_object(
    conn: &mut Connection,
    key: &str,
    patch: &Map<String, Value>,
) -> StoreResult<Map<String, Value>> {
    let tx = conn.transaction()?;
    let mut current = get_object(&tx, key)?;
    apply_patch(&mut current, patch);
    put(&tx, key, &Value::Object(current.clone()))?;
    tx.commit()?;
    Ok(current)
}

/// Условия для [`compare_and_set`].
#[derive(Debug, Default, Clone)]
pub struct Expectations {
    /// Эти ключи обязаны иметь ровно такие значения.
    pub expected: Map<String, Value>,
    /// Этих ключей не должно быть.
    pub expected_absent: Vec<String>,
    /// Этих вложенных путей (`["ключ", "поле", ...]`) не должно быть.
    pub expected_absent_paths: Vec<Vec<String>>,
}

fn path_exists(object: &Map<String, Value>, path: &[String]) -> bool {
    let Some((first, rest)) = path.split_first() else {
        return false;
    };
    let mut node = match object.get(first) {
        Some(node) => node,
        None => return false,
    };
    for segment in rest {
        node = match node.get(segment) {
            Some(next) => next,
            None => return false,
        };
    }
    true
}

/// Условная запись: `patch` применяется, только если текущие настройки ещё совпадают
/// с тем, что видел интерфейс. Так две вкладки браузера не затирают изменения друг друга.
/// Возвращает итоговые настройки и признак, была ли запись применена.
pub fn compare_and_set(
    conn: &mut Connection,
    key: &str,
    expectations: &Expectations,
    patch: &Map<String, Value>,
) -> StoreResult<(Map<String, Value>, bool)> {
    let tx = conn.transaction()?;
    let mut current = get_object(&tx, key)?;

    let matches = expectations
        .expected
        .iter()
        .all(|(field, value)| current.get(field) == Some(value))
        && expectations
            .expected_absent
            .iter()
            .all(|field| !current.contains_key(field))
        && expectations
            .expected_absent_paths
            .iter()
            .all(|path| !path_exists(&current, path));

    if matches {
        apply_patch(&mut current, patch);
        put(&tx, key, &Value::Object(current.clone()))?;
    }
    tx.commit()?;
    Ok((current, matches))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;
    use serde_json::json;

    fn object(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            other => panic!("ожидался объект, получено {other}"),
        }
    }

    #[test]
    fn put_get_delete_roundtrip() {
        let store = Store::open_in_memory().unwrap();
        store
            .with_conn(|conn| {
                assert_eq!(get(conn, "k")?, None);
                put(conn, "k", &json!({ "a": 1 }))?;
                assert_eq!(get(conn, "k")?, Some(json!({ "a": 1 })));
                assert!(delete(conn, "k")?);
                assert!(!delete(conn, "k")?);
                StoreResult::Ok(())
            })
            .unwrap();
    }

    #[test]
    fn merge_replaces_and_null_removes() {
        let store = Store::open_in_memory().unwrap();
        let merged = store
            .with_conn(|conn| {
                merge_object(
                    conn,
                    "chat",
                    &object(json!({ "autoTitle": true, "ragTopK": 5 })),
                )?;
                merge_object(
                    conn,
                    "chat",
                    &object(json!({ "ragTopK": 8, "autoTitle": null })),
                )
            })
            .unwrap();
        assert_eq!(Value::Object(merged), json!({ "ragTopK": 8 }));
    }

    #[test]
    fn compare_and_set_applies_only_when_current() {
        let store = Store::open_in_memory().unwrap();
        store
            .with_conn(|conn| {
                merge_object(conn, "chat", &object(json!({ "activePreset": "default" })))?;

                // Устаревшие ожидания: запись не применяется
                let stale = Expectations {
                    expected: object(json!({ "activePreset": "creative" })),
                    ..Expectations::default()
                };
                let (settings, applied) = compare_and_set(
                    conn,
                    "chat",
                    &stale,
                    &object(json!({ "activePreset": "x" })),
                )?;
                assert!(!applied);
                assert_eq!(settings["activePreset"], "default");

                // Актуальные ожидания и отсутствующий ключ: запись применяется
                let fresh = Expectations {
                    expected: object(json!({ "activePreset": "default" })),
                    expected_absent: vec!["customPresets".to_string()],
                    expected_absent_paths: vec![vec![
                        "inferenceParamsByModel".to_string(),
                        "llama".to_string(),
                    ]],
                };
                let (settings, applied) = compare_and_set(
                    conn,
                    "chat",
                    &fresh,
                    &object(json!({ "activePreset": "creative" })),
                )?;
                assert!(applied);
                assert_eq!(settings["activePreset"], "creative");
                StoreResult::Ok(())
            })
            .unwrap();
    }
}
