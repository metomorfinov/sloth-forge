//! Статистика профиля (`GET /api/profile/stats`).
//!
//! Раньше эндпоинт отдавал выдуманные числа («5 сообщений, 850 токенов»), да ещё
//! в другом формате, чем ждёт фронтенд. Теперь статистика считается из реальных
//! тредов, сообщений и запусков обучения за последние 30 дней. Токены и скорость
//! генерации пока не измеряются: они появятся вместе с движком инференса (этап 2),
//! до этого там честные нули и `null`.

use crate::state::{AppState, TrainingRunSummary};
use axum::extract::{Query, State};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Окно статистики в днях.
const STATS_WINDOW_DAYS: i64 = 30;
/// Сколько последних запусков обучения показывать.
const RECENT_RUNS_LIMIT: usize = 5;
const SECONDS_PER_DAY: i64 = 86_400;
const SECONDS_PER_MINUTE: i64 = 60;
const MILLIS_PER_SECOND: i64 = 1000;
/// Метки времени меньше этого числа считаем секундами, а не миллисекундами
/// (1e11 мс — это 1973 год, а 1e11 секунд — далёкое будущее).
const MILLIS_THRESHOLD: i64 = 100_000_000_000;

#[derive(Debug, Default, Deserialize)]
pub struct ProfileStatsQuery {
    /// Смещение часового пояса браузера в минутах, как `Date.getTimezoneOffset()`
    /// (UTC минус местное время; для Владивостока это -600).
    #[serde(default)]
    pub tz_offset_minutes: i64,
}

pub async fn handle_profile_stats(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ProfileStatsQuery>,
) -> Json<Value> {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0);
    let threads = state.chat_threads.read().await;
    let messages = state.chat_messages.read().await;
    let runs = state.training.runs.read().await;
    Json(compute_profile_stats(
        &threads,
        &messages,
        &runs,
        query.tz_offset_minutes,
        now_ms,
    ))
}

/// Сообщение, сведённое к полям, которые нужны статистике.
struct MessageFacts<'a> {
    thread_id: &'a str,
    role: &'a str,
    created_at_ms: i64,
}

fn created_at_ms(message: &Value) -> Option<i64> {
    let raw = message.get("createdAt").and_then(Value::as_i64)?;
    Some(if raw < MILLIS_THRESHOLD {
        raw.saturating_mul(MILLIS_PER_SECOND)
    } else {
        raw
    })
}

/// Номер дня (от 1970-01-01) в часовом поясе браузера.
fn local_day(ms: i64, tz_offset_minutes: i64) -> i64 {
    (ms / MILLIS_PER_SECOND - tz_offset_minutes * SECONDS_PER_MINUTE).div_euclid(SECONDS_PER_DAY)
}

/// Номер дня от 1970-01-01 → строка `ГГГГ-ММ-ДД` (алгоритм Говарда Хиннанта).
fn date_string(days_since_epoch: i64) -> String {
    let z = days_since_epoch + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Текущая и самая длинная серия дней подряд с активностью.
/// Текущая серия не прерывается, если сегодня ещё не было сообщений, но вчера были.
fn streaks(active_days: &BTreeSet<i64>, today: i64) -> (u64, u64) {
    let mut longest = 0u64;
    let mut run = 0u64;
    let mut previous: Option<i64> = None;
    for &day in active_days {
        run = if previous == Some(day - 1) {
            run + 1
        } else {
            1
        };
        longest = longest.max(run);
        previous = Some(day);
    }

    let anchor = if active_days.contains(&today) {
        Some(today)
    } else if active_days.contains(&(today - 1)) {
        Some(today - 1)
    } else {
        None
    };
    let current = anchor.map_or(0, |mut day| {
        let mut count = 0u64;
        while active_days.contains(&day) {
            count += 1;
            day -= 1;
        }
        count
    });
    (current, longest)
}

fn training_stats(runs: &[TrainingRunSummary]) -> Value {
    let completed = runs.iter().filter(|run| run.status == "completed").count();
    let steps: u64 = runs
        .iter()
        .filter_map(|run| run.final_step)
        .map(u64::from)
        .sum();
    let seconds: u64 = runs.iter().filter_map(|run| run.duration_seconds).sum();
    let models: HashSet<&str> = runs.iter().map(|run| run.model_name.as_str()).collect();
    let datasets: HashSet<&str> = runs.iter().map(|run| run.dataset_name.as_str()).collect();
    let best_loss = runs
        .iter()
        .filter_map(|run| run.final_loss)
        .filter(|loss| loss.is_finite())
        .reduce(f32::min);

    let mut recent: Vec<&TrainingRunSummary> = runs.iter().collect();
    // ISO-даты сортируются как строки: новые запуски первыми
    recent.sort_by(|a, b| b.started_at.cmp(&a.started_at));
    let recent: Vec<Value> = recent
        .into_iter()
        .take(RECENT_RUNS_LIMIT)
        .map(|run| {
            json!({
                "id": run.id,
                "name": run.display_name.clone().unwrap_or_else(|| run.model_name.clone()),
                "modelLabel": run.model_name,
                "datasetLabel": run.dataset_name,
                "status": run.status,
                "finalLoss": run.final_loss,
                "steps": run.final_step.unwrap_or(0),
                "seconds": run.duration_seconds.unwrap_or(0),
                "startedAt": run.started_at
            })
        })
        .collect();

    json!({
        "runs": runs.len(),
        "completed": completed,
        "steps": steps,
        "tokens": 0,
        "seconds": seconds,
        "models": models.len(),
        "datasets": datasets.len(),
        "bestLoss": best_loss,
        "recent": recent
    })
}

/// Считает статистику профиля в формате `ProfileStats` фронтенда.
pub fn compute_profile_stats(
    threads: &HashMap<String, Value>,
    messages: &HashMap<String, Vec<Value>>,
    runs: &[TrainingRunSummary],
    tz_offset_minutes: i64,
    now_ms: i64,
) -> Value {
    let today = local_day(now_ms, tz_offset_minutes);
    let window_start = today - STATS_WINDOW_DAYS + 1;

    let facts: Vec<MessageFacts> = messages
        .iter()
        .flat_map(|(thread_id, list)| {
            list.iter().filter_map(move |message| {
                Some(MessageFacts {
                    thread_id,
                    role: message.get("role").and_then(Value::as_str).unwrap_or(""),
                    created_at_ms: created_at_ms(message)?,
                })
            })
        })
        .collect();

    // Серии дней считаются за всё время, остальное — только за окно статистики
    let all_active_days: BTreeSet<i64> = facts
        .iter()
        .map(|fact| local_day(fact.created_at_ms, tz_offset_minutes))
        .collect();
    let in_window: Vec<&MessageFacts> = facts
        .iter()
        .filter(|fact| local_day(fact.created_at_ms, tz_offset_minutes) >= window_start)
        .collect();

    let mut daily: BTreeMap<i64, (u64, HashSet<&str>)> = BTreeMap::new();
    let mut per_thread: HashMap<&str, (u64, i64, i64)> = HashMap::new();
    let mut per_model: HashMap<String, u64> = HashMap::new();
    for fact in &in_window {
        let entry = daily
            .entry(local_day(fact.created_at_ms, tz_offset_minutes))
            .or_default();
        entry.0 += 1;
        entry.1.insert(fact.thread_id);

        let thread =
            per_thread
                .entry(fact.thread_id)
                .or_insert((0, fact.created_at_ms, fact.created_at_ms));
        thread.0 += 1;
        thread.1 = thread.1.min(fact.created_at_ms);
        thread.2 = thread.2.max(fact.created_at_ms);

        let model_id = threads
            .get(fact.thread_id)
            .and_then(|thread| thread.get("modelId"))
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty());
        if let Some(model_id) = model_id {
            *per_model.entry(model_id.to_string()).or_default() += 1;
        }
    }

    let count_role = |role: &str| in_window.iter().filter(|fact| fact.role == role).count();
    let (current_streak, longest_streak) = streaks(&all_active_days, today);

    let longest_chat = per_thread
        .iter()
        .max_by_key(|(_, (count, first, last))| (*count, last - first))
        .map(|(thread_id, (count, first, last))| {
            json!({
                "threadId": thread_id,
                "title": threads.get(*thread_id).and_then(|t| t.get("title")).and_then(Value::as_str),
                "seconds": (last - first) / MILLIS_PER_SECOND,
                "messages": count
            })
        });

    let mut models: Vec<(String, u64)> = per_model.into_iter().collect();
    models.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    json!({
        "generatedAt": now_ms / MILLIS_PER_SECOND,
        "days": STATS_WINDOW_DAYS,
        "totals": {
            "threads": per_thread.len(),
            "messages": in_window.len(),
            "userMessages": count_role("user"),
            "assistantMessages": count_role("assistant"),
            "promptTokens": 0,
            "completionTokens": 0,
            "totalTokens": 0,
            "chatPromptTokens": 0,
            "chatCompletionTokens": 0,
            "chatTokens": 0,
            "apiPromptTokens": 0,
            "apiCompletionTokens": 0,
            "apiTokens": 0,
            "cachedTokens": 0,
            "toolCalls": 0,
            "attachments": 0,
            "activeDays": daily.len(),
            "chatSeconds": 0
        },
        "streak": {
            "current": current_streak,
            "longest": longest_streak,
            "lastActiveDay": all_active_days.last().map(|day| date_string(*day))
        },
        // Токены пока не считаются, поэтому «самого активного дня по токенам» нет
        "peakDay": null,
        "longestChat": longest_chat,
        "daily": daily.iter().map(|(day, (count, chats))| json!({
            "date": date_string(*day),
            "tokens": 0,
            "messages": count,
            "chats": chats.len()
        })).collect::<Vec<_>>(),
        "models": models.iter().map(|(id, count)| json!({
            "id": id,
            "label": id,
            "messages": count,
            "tokens": 0
        })).collect::<Vec<_>>(),
        "speed": {
            "averageTokensPerSecond": null,
            "bestTokensPerSecond": null,
            "bestTokensPerSecondModel": null,
            "averageResponseMs": null,
            "averageFirstTokenMs": null,
            "samples": 0
        },
        "training": training_stats(runs)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY_MS: i64 = SECONDS_PER_DAY * MILLIS_PER_SECOND;
    /// 2026-09-13 00:00:00 UTC
    const NOW_MS: i64 = 20_709 * DAY_MS;

    #[test]
    fn formats_dates() {
        assert_eq!(date_string(0), "1970-01-01");
        assert_eq!(date_string(19_723), "2024-01-01");
        assert_eq!(date_string(19_782), "2024-02-29");
        assert_eq!(date_string(20_709), "2026-09-13");
    }

    #[test]
    fn applies_browser_timezone() {
        // 2026-09-12 20:00 UTC — во Владивостоке (UTC+10, смещение -600) уже 13 сентября
        let ms = NOW_MS - 4 * 3_600_000;
        assert_eq!(date_string(local_day(ms, 0)), "2026-09-12");
        assert_eq!(date_string(local_day(ms, -600)), "2026-09-13");
    }

    #[test]
    fn empty_state_gives_zeros_and_nulls() {
        let stats = compute_profile_stats(&HashMap::new(), &HashMap::new(), &[], 0, NOW_MS);
        assert_eq!(stats["totals"]["messages"], 0);
        assert_eq!(stats["streak"]["lastActiveDay"], Value::Null);
        assert_eq!(stats["longestChat"], Value::Null);
        assert_eq!(stats["training"]["runs"], 0);
        assert_eq!(stats["daily"], json!([]));
    }

    #[test]
    fn counts_real_messages_and_streaks() {
        let mut threads = HashMap::new();
        threads.insert(
            "t1".to_string(),
            json!({ "title": "Первый", "modelId": "llama-1b" }),
        );
        let mut messages = HashMap::new();
        messages.insert(
            "t1".to_string(),
            vec![
                json!({ "role": "user", "createdAt": NOW_MS - 2 * DAY_MS }),
                json!({ "role": "assistant", "createdAt": NOW_MS - 2 * DAY_MS + 5_000 }),
                json!({ "role": "user", "createdAt": NOW_MS - DAY_MS }),
                json!({ "role": "user", "createdAt": NOW_MS + 1_000 }),
                // Старше окна в 30 дней: в totals не входит, но в серии дней — да
                json!({ "role": "user", "createdAt": NOW_MS - 40 * DAY_MS }),
            ],
        );

        let stats = compute_profile_stats(&threads, &messages, &[], 0, NOW_MS);
        assert_eq!(stats["totals"]["messages"], 4);
        assert_eq!(stats["totals"]["userMessages"], 3);
        assert_eq!(stats["totals"]["assistantMessages"], 1);
        assert_eq!(stats["totals"]["activeDays"], 3);
        assert_eq!(stats["streak"]["current"], 3);
        assert_eq!(stats["streak"]["longest"], 3);
        assert_eq!(stats["streak"]["lastActiveDay"], "2026-09-13");
        assert_eq!(stats["longestChat"]["title"], "Первый");
        assert_eq!(stats["models"][0]["id"], "llama-1b");
        assert_eq!(stats["models"][0]["messages"], 4);
    }
}
