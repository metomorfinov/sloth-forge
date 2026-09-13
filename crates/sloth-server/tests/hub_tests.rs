//! Хаб без интернета: варианты GGUF, проверка токена, инвентарь и удаление моделей
//! на имитации Hugging Face из `common`.

mod common;

use common::{
    minimal_gguf, pattern, Fixture, LIMITED_TOKEN, PRIVATE_REPO, PRIVATE_TOKEN, REPO,
    RETRY_AFTER_SECONDS,
};
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

const SHARD_1: &str = "Q4_K_M/Tiny-Q4_K_M-00001-of-00002.gguf";
const SHARD_2: &str = "Q4_K_M/Tiny-Q4_K_M-00002-of-00002.gguf";
const Q8_FILE: &str = "Tiny-Q8_0.gguf";
const CONTEXT_LENGTH: u32 = 8192;

async fn variants(fx: &Fixture, query: &str) -> (StatusCode, Value) {
    fx.send(
        Method::GET,
        &format!("/api/hub/gguf-variants?{query}"),
        None,
    )
    .await
}

fn quants(body: &Value) -> Vec<String> {
    body["variants"]
        .as_array()
        .unwrap_or_else(|| panic!("в ответе нет списка вариантов: {body}"))
        .iter()
        .map(|variant| variant["quant"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn variant<'a>(body: &'a Value, quant: &str) -> &'a Value {
    body["variants"]
        .as_array()
        .and_then(|list| list.iter().find(|variant| variant["quant"] == quant))
        .unwrap_or_else(|| panic!("в ответе нет варианта {quant}: {body}"))
}

#[tokio::test]
async fn variants_come_from_repo_tree() {
    let fx = Fixture::start(false).await;

    let (status, body) = variants(&fx, &format!("repo_id={REPO}")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        quants(&body),
        ["Q4_0", "Q8_0", "Q4_K_M"],
        "проектор — не вариант; порядок по размеру"
    );
    let q4 = variant(&body, "Q4_K_M");
    assert_eq!(
        q4["size_bytes"], 65_000,
        "размер — сумма шардов из дерева HF"
    );
    assert_eq!(q4["shard_count"], 2);
    assert_eq!(q4["filename"], SHARD_1);
    assert_eq!(q4["downloaded"], false);
    assert!(q4["display_label"].is_null(), "без выдуманных меток");
    assert_eq!(body["has_vision"], true);
    assert_eq!(body["default_variant"], "Q4_K_M");
    assert!(
        body["context_length"].is_null(),
        "ничего не скачано — длина контекста неизвестна"
    );

    // Повторный запрос берётся из кэша, а не из сети
    variants(&fx, &format!("repo_id={REPO}")).await;
    assert_eq!(fx.mock.tree_requests.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn downloaded_and_partial_flags_follow_files_on_disk() {
    let fx = Fixture::start(false).await;
    fx.put_file(&format!("{REPO}/{SHARD_1}"), &pattern(40_000, 1));
    fx.put_file(&format!("{REPO}/{SHARD_2}.part"), &pattern(5_000, 2));
    fx.put_file(
        &format!("{REPO}/{Q8_FILE}"),
        &minimal_gguf(CONTEXT_LENGTH, 30_000),
    );

    let (status, body) = variants(&fx, &format!("repo_id={REPO}")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let q8 = variant(&body, "Q8_0");
    assert_eq!(q8["downloaded"], true);
    assert_eq!(q8["partial"], false);
    let q4 = variant(&body, "Q4_K_M");
    assert_eq!(q4["downloaded"], false);
    assert_eq!(q4["partial"], true);
    assert_eq!(q4["download_remaining_bytes"], 20_000);
    assert_eq!(
        body["default_variant"], "Q8_0",
        "скачанный вариант — по умолчанию"
    );
    assert_eq!(
        body["context_length"], CONTEXT_LENGTH,
        "длина контекста из заголовка скачанного файла"
    );
}

#[tokio::test]
async fn unreachable_hub_gives_error_or_local_files_only() {
    let fx = Fixture::start_offline().await;

    let (status, body) = variants(&fx, &format!("repo_id={REPO}")).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY, "{body}");
    assert!(body["detail"].is_string());

    fx.put_file(
        &format!("{REPO}/{Q8_FILE}"),
        &minimal_gguf(CONTEXT_LENGTH, 1_000),
    );
    let (status, body) = variants(&fx, &format!("repo_id={REPO}")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        quants(&body),
        ["Q8_0"],
        "только то, что реально лежит на диске"
    );
    assert_eq!(variant(&body, "Q8_0")["downloaded"], true);
}

#[tokio::test]
async fn offline_mode_does_not_ask_hugging_face() {
    let fx = Fixture::start(false).await;
    fx.put_file(
        &format!("{REPO}/{Q8_FILE}"),
        &minimal_gguf(CONTEXT_LENGTH, 30_000),
    );

    for query in [
        format!("repo_id={REPO}&offline=true"),
        format!("repo_id={REPO}&prefer_local_cache=true"),
    ] {
        let (status, body) = variants(&fx, &query).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(quants(&body), ["Q8_0"]);
    }
    assert_eq!(fx.mock.tree_requests.load(Ordering::Relaxed), 0);

    let (_, body) = variants(&fx, &format!("repo_id={REPO}")).await;
    assert_eq!(quants(&body).len(), 3);
    assert_eq!(fx.mock.tree_requests.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn private_repo_variants_need_token() {
    let fx = Fixture::start(false).await;

    let (status, body) = variants(&fx, &format!("repo_id={PRIVATE_REPO}")).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("токен"),
        "{body}"
    );

    let (status, body) = fx
        .send_with_token(
            Method::GET,
            &format!("/api/hub/gguf-variants?repo_id={PRIVATE_REPO}"),
            PRIVATE_TOKEN,
            None,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(quants(&body), ["Q4_K_M"]);
}

#[tokio::test]
async fn local_file_variant_reads_context_from_header() {
    let fx = Fixture::start_offline().await;
    let path = fx.put_file("my-model-Q5_K_M.gguf", &minimal_gguf(CONTEXT_LENGTH, 0));
    let path = path.to_string_lossy().into_owned();

    let response = fx
        .client
        .get(format!("{}/api/hub/gguf-variants", fx.base_url))
        .query(&[("local_path", path.as_str())])
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.unwrap();
    assert_eq!(quants(&body), ["Q5_K_M"]);
    assert_eq!(variant(&body, "Q5_K_M")["downloaded"], true);
    assert_eq!(body["context_length"], CONTEXT_LENGTH);

    let (status, _) = variants(&fx, "local_path=/etc/hostname").await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "файл вне папок моделей не открывается"
    );
    let (status, _) = variants(&fx, "").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn token_validation_asks_hugging_face() {
    let fx = Fixture::start(false).await;
    const VALIDATE: &str = "/api/hub/token/validate";

    let (_, body) = fx.send(Method::POST, VALIDATE, None).await;
    assert_eq!(body["status"], "missing");

    for (token, expected) in [(PRIVATE_TOKEN, "valid"), ("hf_wrong", "invalid")] {
        let (_, body) = fx
            .send_with_token(Method::POST, VALIDATE, token, None)
            .await;
        assert_eq!(body["status"], expected, "токен {token}");
    }

    let (_, body) = fx
        .send_with_token(Method::POST, VALIDATE, LIMITED_TOKEN, None)
        .await;
    assert_eq!(body["status"], "rate_limited");
    assert_eq!(body["retry_after_seconds"], RETRY_AFTER_SECONDS);

    let offline = Fixture::start_offline().await;
    let (_, body) = offline
        .send_with_token(Method::POST, VALIDATE, PRIVATE_TOKEN, None)
        .await;
    assert_eq!(body["status"], "unavailable");
}

#[tokio::test]
async fn delete_removes_whole_variant_and_reports_real_impact() {
    let fx = Fixture::start_offline().await;
    let shard_1 = fx.put_file(&format!("{REPO}/{SHARD_1}"), &pattern(40_000, 1));
    let shard_2 = fx.put_file(&format!("{REPO}/{SHARD_2}"), &pattern(25_000, 2));
    let q8_part = fx.put_file(&format!("{REPO}/{Q8_FILE}.part"), &pattern(1_000, 3));

    let (status, impact) = fx
        .send(
            Method::POST,
            "/api/hub/delete-impact",
            Some(json!({ "repo_id": REPO, "variant": "Q4_K_M" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{impact}");
    assert_eq!(impact["reclaimed_bytes"], 65_000);

    let (status, deleted) = fx
        .send(
            Method::DELETE,
            "/api/hub/delete-cached",
            Some(json!({ "repo_id": REPO, "variant": "Q4_K_M" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{deleted}");
    assert_eq!(deleted["deleted_files"].as_array().map(Vec::len), Some(2));
    assert!(!shard_1.exists() && !shard_2.exists(), "удалены все шарды");
    assert!(q8_part.exists(), "другой вариант не тронут");

    let (status, _) = fx
        .send(
            Method::DELETE,
            "/api/hub/delete-cached",
            Some(json!({ "repo_id": REPO, "variant": "Q8_0" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!q8_part.exists(), "недокачанный вариант тоже удаляется");

    // По пути одного шарда удаляется вся модель
    let shard_1 = fx.put_file(&format!("{REPO}/{SHARD_1}"), &pattern(40_000, 1));
    let shard_2 = fx.put_file(&format!("{REPO}/{SHARD_2}"), &pattern(25_000, 2));
    let (status, _) = fx
        .send(
            Method::DELETE,
            "/api/hub/delete-cached",
            Some(json!({ "cache_path": shard_2.to_string_lossy() })),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!shard_1.exists() && !shard_2.exists());

    let (status, _) = fx
        .send(
            Method::DELETE,
            "/api/hub/delete-cached",
            Some(json!({ "repo_id": REPO, "variant": "Q8_0" })),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn delete_is_refused_while_downloading() {
    let fx = Fixture::start(true).await;
    let (status, _) = fx.start_download(REPO, "Q4_K_M").await;
    assert_eq!(status, StatusCode::OK);

    let (status, body) = fx
        .send(
            Method::DELETE,
            "/api/hub/delete-cached",
            Some(json!({ "repo_id": REPO, "variant": "Q4_K_M" })),
        )
        .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");

    let done = fx.wait_for(REPO, "Q4_K_M", &["complete", "error"]).await;
    assert_eq!(done["state"], "complete", "загрузка не пострадала: {done}");
}

#[tokio::test]
async fn inventory_lists_nested_models_with_repo_ids() {
    let fx = Fixture::start_offline().await;
    fx.put_file(&format!("{REPO}/{SHARD_1}"), &pattern(40_000, 1));
    fx.put_file(&format!("{REPO}/{SHARD_2}"), &pattern(25_000, 2));

    let (status, cached) = fx.send(Method::GET, "/api/hub/cached-gguf", None).await;
    assert_eq!(status, StatusCode::OK);
    let rows = cached["cached"].as_array().expect("список моделей");
    assert_eq!(
        rows.len(),
        1,
        "модель из двух шардов — одна строка: {cached}"
    );
    assert_eq!(rows[0]["repo_id"], REPO);
    assert_eq!(rows[0]["format_variant"], "Q4_K_M");
    assert_eq!(rows[0]["size_bytes"], 65_000);
    assert_eq!(rows[0]["partial"], false);

    let (_, local) = fx.send(Method::GET, "/api/hub/local", None).await;
    assert_eq!(local["models"][0]["model_id"], REPO);
    assert_eq!(local["models"][0]["source"], "models_dir");

    let (_, other) = fx.send(Method::GET, "/api/hub/cached-models", None).await;
    assert_eq!(other["cached"].as_array().map(Vec::len), Some(0));
}
