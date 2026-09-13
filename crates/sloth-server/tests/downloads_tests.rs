//! Загрузка моделей без интернета: сервер качает с имитации Hugging Face из `common`.

mod common;

use common::{pattern, Fixture, POLL_ATTEMPTS, POLL_INTERVAL, PRIVATE_REPO, PRIVATE_TOKEN, REPO};
use reqwest::{Method, StatusCode};
use serde_json::json;
use std::sync::atomic::Ordering;

#[tokio::test]
async fn downloads_all_shards_into_repo_folder() {
    let fx = Fixture::start(false).await;

    // Полный запрос, который шлёт фронтенд (лишние поля не должны мешать)
    let (status, started) = fx
        .send(
            Method::POST,
            "/api/hub/download",
            Some(json!({
                "repo_id": REPO,
                "gguf_variant": "Q4_K_M",
                "use_xet": false,
                "scope_id": null,
                "files": null,
                "transport_mode": "http"
            })),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{started}");
    assert_eq!(started["accepted"], true);
    assert!(started["generation"].is_number());

    let done = fx.wait_for(REPO, "Q4_K_M", &["complete", "error"]).await;
    assert_eq!(done["state"], "complete", "{done}");

    for (path, seed, len) in [
        ("Q4_K_M/Tiny-Q4_K_M-00001-of-00002.gguf", 1, 40_000),
        ("Q4_K_M/Tiny-Q4_K_M-00002-of-00002.gguf", 2, 25_000),
    ] {
        let content = std::fs::read(fx.repo_file(REPO, path)).expect("шард скачан");
        assert_eq!(content, pattern(len, seed), "содержимое {path} совпадает");
    }
    assert!(
        !fx.repo_file(REPO, "Tiny-Q8_0.gguf").exists(),
        "другой вариант не качается"
    );
    assert!(
        !fx.repo_file(REPO, "mmproj-F16.gguf").exists(),
        "проектор не считается моделью"
    );

    let (_, progress) = fx
        .send(
            Method::GET,
            &format!(
                "/api/hub/gguf-download-progress?repo_id={REPO}&variant=Q4_K_M&expected_bytes=1"
            ),
            None,
        )
        .await;
    assert_eq!(progress["complete_on_disk"], true);
    assert_eq!(progress["progress"], 1.0);

    // Повторный запуск не качает заново
    let (_, again) = fx.start_download(REPO, "Q4_K_M").await;
    assert_eq!(again["state"], "complete");
}

#[tokio::test]
async fn resumes_partial_file_with_range() {
    let fx = Fixture::start(false).await;
    let full = pattern(30_000, 3);
    let part = fx.put_file(&format!("{REPO}/Tiny-Q8_0.gguf.part"), &full[..12_000]);

    fx.start_download(REPO, "Q8_0").await;
    let done = fx.wait_for(REPO, "Q8_0", &["complete", "error"]).await;
    assert_eq!(done["state"], "complete", "{done}");
    assert_eq!(
        std::fs::read(fx.repo_file(REPO, "Tiny-Q8_0.gguf")).unwrap(),
        full
    );
    assert!(!part.exists());
    assert_eq!(
        fx.mock.range_requests.load(Ordering::Relaxed),
        1,
        "докачка через Range"
    );
}

#[tokio::test]
async fn parallel_jobs_do_not_clobber_each_other() {
    let fx = Fixture::start(true).await;
    fx.start_download(REPO, "Q4_K_M").await;
    fx.start_download(REPO, "Q8_0").await;

    let (_, active) = fx
        .send(Method::GET, "/api/hub/active-downloads", None)
        .await;
    let mut variants: Vec<String> = active["downloads"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["variant"].as_str().unwrap().to_string())
        .collect();
    variants.sort();
    assert_eq!(variants, ["Q4_K_M", "Q8_0"]);

    let first = fx.wait_for(REPO, "Q4_K_M", &["complete", "error"]).await;
    let second = fx.wait_for(REPO, "Q8_0", &["complete", "error"]).await;
    assert_eq!(first["state"], "complete");
    assert_eq!(second["state"], "complete");
}

#[tokio::test]
async fn cancel_keeps_partial_for_resume() {
    let fx = Fixture::start(true).await;
    fx.start_download(REPO, "Q4_K_M").await;

    // Ждём, пока данные начнут поступать
    for _ in 0..POLL_ATTEMPTS {
        let status = fx.status(REPO, "Q4_K_M").await;
        if status["downloaded_bytes"].as_u64().unwrap_or(0) > 0 {
            break;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
    let (_, cancelled) = fx
        .send(
            Method::POST,
            "/api/hub/download/cancel",
            Some(json!({ "repo_id": REPO, "gguf_variant": "Q4_K_M" })),
        )
        .await;
    assert!(matches!(
        cancelled["state"].as_str(),
        Some("cancelling" | "cancelled")
    ));

    let done = fx.wait_for(REPO, "Q4_K_M", &["cancelled"]).await;
    assert_eq!(done["state"], "cancelled");
    let (_, transport) = fx
        .send(
            Method::GET,
            &format!("/api/hub/transport-status?repo_id={REPO}&gguf_variant=Q4_K_M"),
            None,
        )
        .await;
    assert_eq!(transport["has_partial"], true);
    assert_eq!(transport["resumable"], true);
}

#[tokio::test]
async fn http_errors_become_error_state_with_message() {
    let fx = Fixture::start(false).await;
    let (status, _) = fx.start_download(REPO, "Q4_0").await;
    assert_eq!(status, StatusCode::OK);
    let done = fx.wait_for(REPO, "Q4_0", &["complete", "error"]).await;
    assert_eq!(done["state"], "error");
    assert!(
        done["error"].as_str().unwrap_or_default().contains("404"),
        "{done}"
    );
}

#[tokio::test]
async fn private_repo_requires_token_from_header_or_settings() {
    let fx = Fixture::start(false).await;

    let (status, body) = fx.start_download(PRIVATE_REPO, "Q4_K_M").await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("токен"),
        "{body}"
    );

    // Токен из заголовка фронтенда
    let (status, _) = fx
        .send_with_token(
            Method::POST,
            "/api/hub/download",
            PRIVATE_TOKEN,
            Some(json!({ "repo_id": PRIVATE_REPO, "gguf_variant": "Q4_K_M" })),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let done = fx
        .wait_for(PRIVATE_REPO, "Q4_K_M", &["complete", "error"])
        .await;
    assert_eq!(done["state"], "complete", "{done}");

    // Токен из настроек: удаляем файл и качаем без заголовка
    std::fs::remove_file(fx.repo_file(PRIVATE_REPO, "Secret-Q4_K_M.gguf")).unwrap();
    fx.send(
        Method::PUT,
        "/api/settings/hugging-face-token",
        Some(json!({ "token": PRIVATE_TOKEN })),
    )
    .await;
    let (status, _) = fx.start_download(PRIVATE_REPO, "Q4_K_M").await;
    assert_eq!(status, StatusCode::OK);
    let done = fx
        .wait_for(PRIVATE_REPO, "Q4_K_M", &["complete", "error"])
        .await;
    assert_eq!(done["state"], "complete", "{done}");
}

#[tokio::test]
async fn rejects_invalid_requests() {
    let fx = Fixture::start(false).await;
    for repo in ["../etc", "a/b/c", "org/.hidden"] {
        let (status, _) = fx.start_download(repo, "Q4_K_M").await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "repo_id «{repo}»");
    }
    let (status, _) = fx.start_download(REPO, "Q2_K").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

    let (_, idle) = fx
        .send(
            Method::GET,
            "/api/hub/download-status?repo_id=unsloth/Other-GGUF&gguf_variant=Q4_K_M",
            None,
        )
        .await;
    assert_eq!(idle["state"], "idle");
}
