use futures_util::StreamExt;
use reqwest::header::CONTENT_TYPE;
use sloth_server::create_router;
use sloth_server::state::AppState;
use sloth_vulkan_sys::VulkanContext;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

async fn spawn_test_server() -> (String, Arc<AppState>) {
    let vk_ctx = VulkanContext::init(true).ok();
    // Пути от корня проекта, а не зашитые под одну машину
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let static_dir = project_root.join("frontend/dist");
    let repo_models = project_root.join("models");
    let models_dir = if repo_models.exists() {
        repo_models
    } else {
        PathBuf::from("models")
    };

    let state = Arc::new(AppState::new(vk_ctx, models_dir, Some(static_dir.clone())));
    let app = create_router(Arc::clone(&state), Some(static_dir));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (format!("http://{}", addr), state)
}

#[tokio::test]
async fn test_get_vram() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/api/vram"))
        .send()
        .await
        .expect("Failed to call /api/vram");

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let vram: serde_json::Value = resp.json().await.unwrap();

    assert!(vram["deviceName"].as_str().is_some() || vram["device_name"].as_str().is_some());
    let total_mb = vram["totalMb"].as_u64().or_else(|| vram["total_mb"].as_u64()).unwrap();
    assert!(total_mb >= 4096);
    let used_mb = vram["usedMb"].as_u64().or_else(|| vram["used_mb"].as_u64()).unwrap();
    let free_mb = vram["freeMb"].as_u64().or_else(|| vram["free_mb"].as_u64()).unwrap();
    assert!(free_mb > 0);
    assert!(total_mb >= used_mb);
}

#[tokio::test]
async fn test_get_hardware() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/api/hardware"))
        .send()
        .await
        .expect("Failed to call /api/hardware");

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let hw: serde_json::Value = resp.json().await.unwrap();

    // Verify GPU details: AMD Radeon RX 570, Vulkan 1.4, wavefront 64, GCN 4.0
    let gpu_name = hw["gpuName"].as_str().or_else(|| hw["gpu_name"].as_str()).unwrap();
    assert!(gpu_name.contains("AMD Radeon RX 570"));

    let vulkan_ver = hw["vulkanVersion"].as_str().or_else(|| hw["vulkan_version"].as_str()).unwrap();
    assert!(vulkan_ver.contains("1.4"));

    let wavefront = hw["wavefrontSize"].as_u64().or_else(|| hw["wavefront_size"].as_u64()).unwrap();
    assert_eq!(wavefront, 64);

    let arch = hw["architecture"].as_str().unwrap();
    assert!(arch.contains("GCN 4.0"));
}

#[tokio::test]
async fn test_get_models() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/api/models"))
        .send()
        .await
        .expect("Failed to call /api/models");

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();

    let models = body["models"].as_array().expect("Expected models array");
    assert!(!models.is_empty());

    let has_llama = models.iter().any(|m| {
        m["filename"].as_str().map(|f| f.contains("Llama")).unwrap_or(false)
    });
    assert!(has_llama);
}

#[tokio::test]
async fn test_training_beginner_preset_start_stop_status() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. Initial status should be idle
    let resp = client.get(format!("{base_url}/api/train/status")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let status: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(status["active"].as_bool(), Some(false));

    // 2. Start training with beginner preset "style"
    let start_payload = serde_json::json!({
        "mode": "beginner",
        "preset": "style",
        "intensity": "normal",
        "total_steps": 100
    });

    let resp = client
        .post(format!("{base_url}/api/train/start"))
        .json(&start_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let start_resp: serde_json::Value = resp.json().await.unwrap();
    assert!(start_resp["status"].as_str() == Some("queued") || start_resp["status"].as_str() == Some("started"));
    assert_eq!(start_resp["loraRank"].as_u64().or_else(|| start_resp["lora_rank"].as_u64()), Some(16));

    // Let the training loop execute a couple steps
    tokio::time::sleep(Duration::from_millis(150)).await;

    // 3. Status should now be active
    let resp = client.get(format!("{base_url}/api/train/status")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let status: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(status["active"].as_bool(), Some(true));
    let step = status["step"].as_u64().unwrap();
    assert!(step > 0);
    let loss = status["loss"].as_f64().unwrap();
    assert!(loss > 0.0);
    let tok_s = status["tokensPerSec"].as_f64().or_else(|| status["tokens_per_sec"].as_f64()).unwrap();
    assert!(tok_s > 1000.0);

    // 4. Stop training
    let resp = client
        .post(format!("{base_url}/api/train/stop"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let stop_resp: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(stop_resp["status"].as_str(), Some("stopped"));

    // 5. Verify stopped status
    tokio::time::sleep(Duration::from_millis(50)).await;
    let resp = client.get(format!("{base_url}/api/train/status")).send().await.unwrap();
    let status: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(status["active"].as_bool(), Some(false));
}

#[tokio::test]
async fn test_training_pro_hyperparameters() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let pro_payload = serde_json::json!({
        "mode": "pro",
        "loraRank": 32,
        "loraAlpha": 64.0,
        "learningRate": 0.0001,
        "weightDecay": 0.01,
        "targetModules": ["q_proj", "v_proj", "o_proj"],
        "totalSteps": 50
    });

    let resp = client
        .post(format!("{base_url}/api/train/start"))
        .json(&pro_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let start_resp: serde_json::Value = resp.json().await.unwrap();
    assert!(start_resp["status"].as_str() == Some("queued") || start_resp["status"].as_str() == Some("started"));
    assert_eq!(start_resp["loraRank"].as_u64().or_else(|| start_resp["lora_rank"].as_u64()), Some(32));

    // Halt
    let resp = client.post(format!("{base_url}/api/train/stop")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
}

#[tokio::test]
async fn test_ws_telemetry() {
    let (base_url, _) = spawn_test_server().await;
    let ws_url = base_url.replace("http://", "ws://") + "/ws/telemetry";

    let (mut ws_stream, _) = tokio_tungstenite::connect_async(ws_url)
        .await
        .expect("WebSocket handshake failed");

    // First message: log greeting or initial telemetry
    let first = ws_stream.next().await.unwrap().unwrap();
    assert!(first.is_text());
    let text = first.into_text().unwrap();
    assert!(text.contains("[SYSTEM]") || text.contains("telemetry"));

    // Second message: telemetry envelope
    let second = ws_stream.next().await.unwrap().unwrap();
    assert!(second.is_text());
    let second_text = second.into_text().unwrap();
    assert!(second_text.contains("telemetry") || second_text.contains("vramUsedMb") || second_text.contains("vram_used_mb"));
}

#[tokio::test]
async fn test_cluster_worker_registration_and_status() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // Register worker
    let reg_payload = serde_json::json!({
        "worker_id": "rx570-worker-02",
        "ip": "192.168.1.105",
        "gpu_name": "AMD Radeon RX 570",
        "vram_mb": 4096,
        "latency_ms": 1.15
    });

    let resp = client
        .post(format!("{base_url}/api/cluster/worker/register"))
        .json(&reg_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let reg_ack: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(reg_ack["status"].as_str(), Some("ok"));
    assert_eq!(reg_ack["rank"].as_u64(), Some(1));
    assert_eq!(reg_ack["worldSize"].as_u64().or_else(|| reg_ack["world_size"].as_u64()), Some(2));

    // Verify cluster status reflects registered worker
    let resp = client.get(format!("{base_url}/api/cluster/status")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let status: serde_json::Value = resp.json().await.unwrap();

    assert_eq!(status["role"].as_str(), Some("master"));
    assert_eq!(status["worldSize"].as_u64().or_else(|| status["world_size"].as_u64()), Some(2));
    let workers = status["workers"].as_array().unwrap();
    assert_eq!(workers.len(), 1);
    assert_eq!(workers[0]["workerId"].as_str().or_else(|| workers[0]["worker_id"].as_str()), Some("rx570-worker-02"));
    let total_vram = status["totalVramMb"].as_u64().or_else(|| status["total_vram_mb"].as_u64()).unwrap();
    assert_eq!(total_vram, 8192); // 4096 + 4096
    let allreduce_ready = status["allReduceReady"].as_bool().or_else(|| status["all_reduce_ready"].as_bool()).unwrap();
    assert!(allreduce_ready);
}

#[tokio::test]
async fn test_cluster_sync_grad_json_and_binary() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. JSON Sync
    let json_payload = serde_json::json!({
        "step": 42,
        "rank": 2,
        "worker_id": "rx570-worker-02",
        "gradients": [1.0, 2.0, 3.0, 4.0]
    });

    let resp = client
        .post(format!("{base_url}/api/cluster/sync_grad"))
        .json(&json_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let sync_resp: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(sync_resp["status"].as_str(), Some("ok"));
    let grads = sync_resp["gradients"].as_array().unwrap();
    assert_eq!(grads.len(), 4);

    // 2. Binary Sync (application/octet-stream)
    let raw_floats: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];
    let mut bytes = Vec::new();
    for f in &raw_floats {
        bytes.extend_from_slice(&f.to_le_bytes());
    }

    let resp = client
        .post(format!("{base_url}/api/cluster/sync_grad"))
        .header(CONTENT_TYPE, "application/octet-stream")
        .body(bytes)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(
        resp.headers().get(CONTENT_TYPE).unwrap(),
        "application/octet-stream"
    );
    let resp_bytes = resp.bytes().await.unwrap();
    assert_eq!(resp_bytes.len(), 16); // 4 floats * 4 bytes
}

#[tokio::test]
async fn test_openai_chat_completions_json() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let chat_payload = serde_json::json!({
        "model": "slothforge-llama-3.2-3b",
        "messages": [
            {"role": "system", "content": "You are SlothForge."},
            {"role": "user", "content": "Tell me about your Vulkan backend"}
        ],
        "stream": false
    });

    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&chat_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let chat_resp: serde_json::Value = resp.json().await.unwrap();

    assert_eq!(chat_resp["object"].as_str(), Some("chat.completion"));
    let choices = chat_resp["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 1);
    let content = choices[0]["message"]["content"].as_str().unwrap();
    assert!(!content.is_empty());
    assert!(content.contains("Vulkan") || content.contains("AMD"));
    assert_eq!(choices[0]["finish_reason"].as_str(), Some("stop"));
}

#[tokio::test]
async fn test_openai_chat_completions_sse_streaming() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let chat_payload = serde_json::json!({
        "model": "slothforge-llama-3.2-3b",
        "messages": [
            {"role": "user", "content": "Hello SlothForge"}
        ],
        "stream": true
    });

    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&chat_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let ct = resp.headers().get(CONTENT_TYPE).unwrap().to_str().unwrap();
    assert!(ct.contains("text/event-stream"));

    let body_text = resp.text().await.unwrap();
    assert!(body_text.contains("chat.completion.chunk"));
    assert!(body_text.contains("[DONE]"));
}

#[tokio::test]
async fn test_static_files_and_spa_fallback() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. Root index.html
    let resp = client.get(format!("{base_url}/")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let html = resp.text().await.unwrap();
    assert!(html.contains("<!DOCTYPE html>") || html.contains("SlothForge") || html.contains("<html"));

    // 2. SPA route fallback (e.g. /studio) should also serve index.html
    let resp = client.get(format!("{base_url}/studio")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let html = resp.text().await.unwrap();
    assert!(html.contains("<!DOCTYPE html>") || html.contains("<html"));

    // 3. Static asset file
    let resp = client.get(format!("{base_url}/assets/index-D1keva_w.css")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
}

#[tokio::test]
async fn test_unsloth_health_endpoint() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/api/health"))
        .send()
        .await
        .expect("Failed to call /api/health");

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let health: serde_json::Value = resp.json().await.unwrap();

    assert_eq!(health["status"].as_str(), Some("ok"));
    assert_eq!(health["version"].as_str(), Some("0.1.0"));
    assert_eq!(health["device_type"].as_str(), Some("vulkan"));
    let gpu_name = health["gpu_name"].as_str().unwrap();
    assert!(gpu_name.contains("AMD Radeon RX 570"));
    assert_eq!(health["vram_total_mb"].as_u64(), Some(4096));
    assert_eq!(health["vram_free_mb"].as_u64(), Some(4096));
    assert_eq!(health["vram_used_mb"].as_u64(), Some(0));
    assert_eq!(health["cuda_available"].as_bool(), Some(false));
    assert_eq!(health["rocm_available"].as_bool(), Some(false));
    assert_eq!(health["vulkan_available"].as_bool(), Some(true));
    assert_eq!(health["chat_only"].as_bool(), Some(false));

    let capabilities = health["capabilities"].as_array().expect("capabilities array");
    let caps: Vec<&str> = capabilities.iter().filter_map(|c| c.as_str()).collect();
    assert!(caps.contains(&"train"));
    assert!(caps.contains(&"chat"));
    assert!(caps.contains(&"gguf"));
    assert!(caps.contains(&"lora"));
    assert!(caps.contains(&"cluster"));
}

#[tokio::test]
async fn test_unsloth_auth_status_endpoint() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/api/auth/status"))
        .send()
        .await
        .expect("Failed to call /api/auth/status");

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let auth: serde_json::Value = resp.json().await.unwrap();

    assert_eq!(auth["authenticated"].as_bool(), Some(true));
    assert_eq!(auth["auth_required"].as_bool(), Some(false));
    assert_eq!(auth["initialized"].as_bool(), Some(true));
    assert_eq!(auth["requires_password_change"].as_bool(), Some(false));
    assert_eq!(auth["user"]["username"].as_str(), Some("rivergod"));
    assert_eq!(auth["user"]["role"].as_str(), Some("admin"));
}

#[tokio::test]
async fn test_unsloth_models_and_model_picker_schema() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. GET /api/models
    let resp = client.get(format!("{base_url}/api/models")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();

    let models = body["models"].as_array().unwrap();
    assert!(!models.is_empty());
    let m0 = &models[0];
    assert!(m0["id"].as_str().is_some());
    assert!(m0["name"].as_str().is_some());
    assert_eq!(m0["isGguf"].as_bool().or_else(|| m0["is_gguf"].as_bool()), Some(true));
    assert_eq!(m0["isVision"].as_bool().or_else(|| m0["is_vision"].as_bool()), Some(false));
    assert_eq!(m0["source"].as_str(), Some("models_dir"));

    // 2. Check /api/models/list alias
    let resp_list = client.get(format!("{base_url}/api/models/list")).send().await.unwrap();
    assert_eq!(resp_list.status(), reqwest::StatusCode::OK);

    // 3. Check /api/models/local alias
    let resp_local = client.get(format!("{base_url}/api/models/local")).send().await.unwrap();
    assert_eq!(resp_local.status(), reqwest::StatusCode::OK);
}

#[tokio::test]
async fn test_unsloth_training_status_and_progress_endpoints() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. GET /api/train/status
    let resp = client.get(format!("{base_url}/api/train/status")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let status: serde_json::Value = resp.json().await.unwrap();

    assert!(status["job_id"].as_str().is_some());
    assert_eq!(status["phase"].as_str(), Some("idle"));
    assert_eq!(status["is_training_running"].as_bool(), Some(false));
    assert_eq!(status["eval_enabled"].as_bool(), Some(false));
    assert!(status["message"].as_str().is_some());
    assert!(status["warnings"].as_array().is_some());

    // 2. GET /api/train/progress as JSON
    let resp = client.get(format!("{base_url}/api/train/progress")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let prog: serde_json::Value = resp.json().await.unwrap();
    assert!(prog["job_id"].as_str().is_some());
    assert_eq!(prog["phase"].as_str(), Some("idle"));

    // 3. GET /api/train/progress with text/event-stream header
    let resp = client
        .get(format!("{base_url}/api/train/progress?expected_job_id=test-job-123"))
        .header("Accept", "text/event-stream")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let ct = resp.headers().get(CONTENT_TYPE).unwrap().to_str().unwrap();
    assert!(ct.contains("text/event-stream"));
}

#[tokio::test]
async fn test_unsloth_training_start_stop_reset() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // Start with Unsloth request payload format (snake_case, string lr)
    let start_payload = serde_json::json!({
        "model_name": "Llama-3.2-3B-Instruct",
        "hf_dataset": "yahma/alpaca-cleaned",
        "project_name": "test-project",
        "training_type": "sft",
        "learning_rate": "2e-4",
        "lora_r": 16,
        "lora_alpha": 32.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_steps": 50,
        "start_request_id": "req-unsloth-001"
    });

    let resp = client
        .post(format!("{base_url}/api/train/start"))
        .json(&start_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let start_resp: serde_json::Value = resp.json().await.unwrap();
    assert!(start_resp["job_id"].as_str().is_some() || start_resp["jobId"].as_str().is_some());
    assert!(start_resp["status"].as_str() == Some("queued") || start_resp["status"].as_str() == Some("started"));

    tokio::time::sleep(Duration::from_millis(150)).await;

    // Check status is training
    let resp = client.get(format!("{base_url}/api/train/status")).send().await.unwrap();
    let status: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(status["is_training_running"].as_bool(), Some(true));
    assert_eq!(status["phase"].as_str(), Some("training"));

    // Stop
    let resp = client.post(format!("{base_url}/api/train/stop")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let stop_resp: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(stop_resp["status"].as_str(), Some("stopped"));

    // Reset
    let resp = client.post(format!("{base_url}/api/train/reset")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let reset_resp: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(reset_resp["status"].as_str(), Some("ok"));

    // Status after reset should be idle
    let resp = client.get(format!("{base_url}/api/train/status")).send().await.unwrap();
    let status: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(status["is_training_running"].as_bool(), Some(false));
    assert_eq!(status["phase"].as_str(), Some("idle"));
    assert_eq!(status["step"].as_u64(), Some(0));
}

#[tokio::test]
async fn test_unsloth_training_runs_history() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. Initial GET /api/train/runs
    let resp = client.get(format!("{base_url}/api/train/runs")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["runs"].as_array().is_some());
    assert!(body["total"].as_u64().is_some());

    // 2. Start a run
    let start_payload = serde_json::json!({
        "model_name": "Llama-3.2-3B-Instruct",
        "dataset": "sloth-alpaca",
        "total_steps": 25,
        "mode": "beginner",
        "preset": "style"
    });
    let _ = client.post(format!("{base_url}/api/train/start")).json(&start_payload).send().await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let _ = client.post(format!("{base_url}/api/train/stop")).send().await.unwrap();

    // 3. GET /api/train/runs should now contain the run
    let resp = client.get(format!("{base_url}/api/train/runs")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    let runs = body["runs"].as_array().unwrap();
    assert!(!runs.is_empty());
    let run = &runs[0];
    assert!(run["id"].as_str().is_some());
    assert_eq!(run["status"].as_str(), Some("stopped"));
    assert!(run["started_at"].as_str().is_some());

    // 4. GET /api/train/runs/:id detail
    let run_id = run["id"].as_str().unwrap();
    let resp = client.get(format!("{base_url}/api/train/runs/{run_id}")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let detail: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(detail["run"]["id"].as_str(), Some(run_id));
    assert!(detail["config"].is_object());
    assert!(detail["metrics"].is_object());
}

#[tokio::test]
async fn test_unsloth_inference_chat_sse() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let chat_payload = serde_json::json!({
        "model": "slothforge-llama-3.2-3b",
        "messages": [
            {"role": "user", "content": "How does SlothForge run on Polaris?"}
        ],
        "stream": true
    });

    // POST /api/inference/chat
    let resp = client
        .post(format!("{base_url}/api/inference/chat"))
        .json(&chat_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let ct = resp.headers().get(CONTENT_TYPE).unwrap().to_str().unwrap();
    assert!(ct.contains("text/event-stream"));

    let body_text = resp.text().await.unwrap();
    assert!(body_text.contains("chat.completion.chunk"));
    assert!(body_text.contains("[DONE]"));

    // POST /api/inference/chat/completions (non-stream)
    let non_stream_payload = serde_json::json!({
        "model": "slothforge-llama-3.2-3b",
        "messages": [
            {"role": "user", "content": "Vulkan compute"}
        ],
        "stream": false
    });
    let resp = client
        .post(format!("{base_url}/api/inference/chat/completions"))
        .json(&non_stream_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let chat_resp: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(chat_resp["object"].as_str(), Some("chat.completion"));
}

#[tokio::test]
async fn test_unsloth_studio_install_source_and_update_status() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. GET /api/studio/install-source
    let resp = client
        .get(format!("{base_url}/api/studio/install-source"))
        .send()
        .await
        .expect("Failed to call /api/studio/install-source");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let src: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(src["source"].as_str(), Some("native"));
    assert_eq!(src["channel"].as_str(), Some("stable"));

    // 2. GET /api/studio/update-status
    let resp = client
        .get(format!("{base_url}/api/studio/update-status"))
        .send()
        .await
        .expect("Failed to call /api/studio/update-status");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let update: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(update["update_available"].as_bool(), Some(false));
    assert_eq!(update["current_version"].as_str(), Some("0.1.0"));
}

#[tokio::test]
async fn test_inference_status_and_monitor() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. GET /api/inference/status
    let resp = client
        .get(format!("{base_url}/api/inference/status"))
        .send()
        .await
        .expect("Failed to call /api/inference/status");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let status: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(status["active_model"].as_str(), Some("llama-3.2-3b-instruct-q4_k_m"));
    assert_eq!(status["model_identifier"].as_str(), Some("llama-3.2-3b-instruct-q4_k_m"));
    assert_eq!(status["is_vision"].as_bool(), Some(false));
    assert_eq!(status["is_gguf"].as_bool(), Some(true));
    assert_eq!(status["is_local_model"].as_bool(), Some(true));
    assert_eq!(status["loading"].as_array().map(|a| a.len()), Some(0));
    assert_eq!(status["loaded"][0].as_str(), Some("llama-3.2-3b-instruct-q4_k_m"));
    assert_eq!(status["context_length"].as_u64(), Some(131072));
    assert_eq!(status["supports_tools"].as_bool(), Some(false));

    // 2. GET /api/inference/monitor
    let resp = client
        .get(format!("{base_url}/api/inference/monitor"))
        .send()
        .await
        .expect("Failed to call /api/inference/monitor");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let monitor: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(monitor["total"].as_u64(), Some(0));
    assert_eq!(monitor["entries"].as_array().map(|a| a.len()), Some(0));
}

#[tokio::test]
async fn test_hub_inventory_and_variants() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. GET /api/hub/cached-models
    let resp = client
        .get(format!("{base_url}/api/hub/cached-models"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    let cached_models = body["cached"].as_array().unwrap();
    assert!(!cached_models.is_empty());
    assert!(body["total_size_bytes"].as_u64().unwrap() > 0);

    // 2. GET /api/hub/cached-gguf
    let resp = client
        .get(format!("{base_url}/api/hub/cached-gguf"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    let cached = body["cached"].as_array().unwrap();
    assert!(!cached.is_empty());
    assert!(body["total_size_bytes"].as_u64().unwrap() > 0);
    let c0 = &cached[0];
    assert!(c0["repo_id"].as_str().is_some());
    assert!(c0["load_id"].as_str().is_some());
    assert_eq!(c0["model_format"].as_str(), Some("gguf"));
    assert_eq!(c0["runtime"].as_str(), Some("llama_cpp"));
    assert_eq!(c0["partial"].as_bool(), Some(false));
    assert!(c0["capabilities"]["can_chat"].as_bool().unwrap());
    assert!(c0["capabilities"]["can_train"].as_bool().unwrap());
    assert!(!c0["capabilities"]["can_download"].as_bool().unwrap());

    // 3. GET /api/hub/local
    let resp = client
        .get(format!("{base_url}/api/hub/local"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    let models = body["models"].as_array().unwrap();
    assert!(!models.is_empty());
    assert_eq!(body["count"].as_u64().unwrap(), models.len() as u64);
    assert!(body["models_dir"].as_str().unwrap().ends_with("models"));
    let m0 = &models[0];
    assert!(m0["id"].as_str().is_some());
    assert!(m0["display_name"].as_str().is_some());
    assert!(m0["path"].as_str().unwrap().starts_with("models/"));
    assert_eq!(m0["model_format"].as_str(), Some("gguf"));
    assert_eq!(m0["runtime"].as_str(), Some("llama_cpp"));
    assert_eq!(m0["source"].as_str(), Some("models_dir"));
    assert!(m0["capabilities"]["can_chat"].as_bool().unwrap());

    // 4. GET /api/hub/hidden-models
    let resp = client
        .get(format!("{base_url}/api/hub/hidden-models"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["hidden_models"].as_array().map(|a| a.len()), Some(0));

    // 5. GET /api/hub/active-downloads
    let resp = client
        .get(format!("{base_url}/api/hub/active-downloads"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["downloads"].as_array().map(|a| a.len()), Some(0));

    // 6. GET /api/hub/gguf-variants for local file directly by filename
    let local_file = "Llama-3.2-1B-Instruct-Q4_K_M.gguf";
    let resp = client
        .get(format!("{base_url}/api/hub/gguf-variants?repo_id={local_file}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["resolved_locally"].as_bool(), Some(true));
    assert_eq!(body["context_length"].as_u64(), Some(131072));
    assert_eq!(body["default_variant"].as_str(), Some("Q4_K_M"));
    let local_vars = body["variants"].as_array().unwrap();
    assert_eq!(local_vars.len(), 1);
    assert_eq!(local_vars[0]["downloaded"].as_bool(), Some(true));
    assert_eq!(local_vars[0]["quant"].as_str(), Some("Q4_K_M"));
    assert_eq!(local_vars[0]["filename"].as_str(), Some("Llama-3.2-1B-Instruct-Q4_K_M.gguf"));

    // 7. GET /api/hub/gguf-variants with local_path query param
    let resp = client
        .get(format!("{base_url}/api/hub/gguf-variants?local_path=models/{local_file}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["resolved_locally"].as_bool(), Some(true));
    assert_eq!(body["variants"][0]["downloaded"].as_bool(), Some(true));

    // 8. GET /api/hub/gguf-variants for repo unsloth/Llama-3.2-1B-Instruct-GGUF (detects local Q4_K_M)
    let repo_1b = "unsloth/Llama-3.2-1B-Instruct-GGUF";
    let resp = client
        .get(format!("{base_url}/api/hub/gguf-variants?repo_id={repo_1b}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    let vars_1b = body["variants"].as_array().unwrap();
    let q4_1b = vars_1b.iter().find(|v| v["quant"].as_str() == Some("Q4_K_M")).unwrap();
    assert_eq!(q4_1b["downloaded"].as_bool(), Some(true));

    // 9. GET /api/hub/gguf-variants for repo unsloth/Llama-3.2-3B-Instruct-GGUF (not downloaded)
    let repo_3b = "unsloth/Llama-3.2-3B-Instruct-GGUF";
    let resp = client
        .get(format!("{base_url}/api/hub/gguf-variants?repo_id={repo_3b}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["repo_id"].as_str(), Some(repo_3b));
    assert_eq!(body["has_vision"].as_bool(), Some(false));
    assert_eq!(body["default_variant"].as_str(), Some("Q4_K_M"));
    let variants = body["variants"].as_array().unwrap();
    assert!(variants.len() >= 2);
    let q4 = variants.iter().find(|v| v["quant"].as_str() == Some("Q4_K_M")).unwrap();
    assert_eq!(q4["quant"].as_str(), Some("Q4_K_M"));
    assert_eq!(q4["display_label"].as_str(), Some("Q4_K_M (Recommended)"));
    assert!(q4["size_bytes"].as_u64().unwrap() > 1_500_000_000);

    // 10. POST /api/inference/load with filename, model id, and repo id
    let resp = client
        .post(format!("{base_url}/api/inference/load"))
        .json(&serde_json::json!({ "model": "Llama-3.2-1B-Instruct-Q4_K_M.gguf" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let load_res: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(load_res["status"].as_str(), Some("ok"));
    assert_eq!(load_res["model"].as_str(), Some("Llama-3.2-1B-Instruct-Q4_K_M.gguf"));

    let resp = client
        .post(format!("{base_url}/api/inference/load"))
        .json(&serde_json::json!({ "model": "llama-3.2-1b-instruct-q4_k_m" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let load_res: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(load_res["status"].as_str(), Some("ok"));

    let resp = client
        .post(format!("{base_url}/api/inference/load"))
        .json(&serde_json::json!({ "repo_id": "unsloth/Llama-3.2-1B-Instruct-GGUF" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let load_res: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(load_res["status"].as_str(), Some("ok"));
}

#[tokio::test]
async fn test_hub_model_download_lifecycle() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. Initial download-status is idle
    let resp = client
        .get(format!("{base_url}/api/hub/download-status"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["state"].as_str(), Some("idle"));
    assert_eq!(body["percent"].as_f64(), Some(0.0));
    assert_eq!(body["downloaded_bytes"].as_u64(), Some(0));
    assert_eq!(body["total_bytes"].as_u64(), Some(0));

    // 2. Start download
    let start_payload = serde_json::json!({
        "repo_id": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "gguf_variant": "Q4_K_M"
    });
    let resp = client
        .post(format!("{base_url}/api/hub/download"))
        .json(&start_payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let start_res: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(start_res["state"].as_str(), Some("running"));
    assert_eq!(start_res["accepted"].as_bool(), Some(true));
    assert_eq!(start_res["job_key"].as_str(), Some("job-default"));
    assert_eq!(start_res["generation"].as_u64(), Some(1));

    // 3. Check progress
    let resp = client
        .get(format!("{base_url}/api/hub/download-progress"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let prog: serde_json::Value = resp.json().await.unwrap();
    assert!(prog["expected_bytes"].as_u64().unwrap() > 0);

    // 4. Cancel download
    let cancel_payload = serde_json::json!({
        "job_key": "job-default"
    });
    let resp = client
        .post(format!("{base_url}/api/hub/download/cancel"))
        .json(&cancel_payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let cancel_res: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(cancel_res["job_key"].as_str(), Some("job-default"));
    assert_eq!(cancel_res["state"].as_str(), Some("cancelled"));
}

#[tokio::test]
async fn test_hub_datasets_and_models_folders() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // Datasets
    let resp = client.get(format!("{base_url}/api/hub/datasets/cached")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["cached"].as_array().map(|a| a.len()), Some(0));

    let resp = client.get(format!("{base_url}/api/hub/datasets/local")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["datasets"].as_array().map(|a| a.len()), Some(0));

    let resp = client.get(format!("{base_url}/api/hub/datasets/active-downloads")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["downloads"].as_array().map(|a| a.len()), Some(0));

    // Models scan & folders
    let resp = client.get(format!("{base_url}/api/models/scan-folders")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["folders"].as_array().map(|a| a.len()), Some(0));

    let resp = client.get(format!("{base_url}/api/models/recommended-folders")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["folders"][0].as_str(), Some("models"));

    let resp = client.get(format!("{base_url}/api/models/loras")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["loras"].as_array().map(|a| a.len()), Some(0));
}

#[tokio::test]
async fn test_chat_threads_projects_and_settings() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // Threads
    let resp = client.get(format!("{base_url}/api/chat/threads")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["threads"].as_array().map(|a| a.len()), Some(0));

    let resp = client
        .post(format!("{base_url}/api/chat/threads"))
        .json(&serde_json::json!({"title": "Test Chat"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["id"].as_str(), Some("thread-1"));

    // Projects
    let resp = client.get(format!("{base_url}/api/chat/projects")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["projects"].as_array().map(|a| a.len()), Some(0));

    // Chat Settings
    let resp = client.get(format!("{base_url}/api/chat/settings")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["temperature"].as_f64(), Some(0.7));
    assert_eq!(body["top_p"].as_f64(), Some(0.9));
    assert_eq!(body["max_tokens"].as_u64(), Some(2048));
}

#[tokio::test]
async fn test_settings_and_studio_export_endpoints() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // Personalization
    let resp = client.get(format!("{base_url}/api/settings/personalization")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["user_name"].as_str(), Some("rivergod"));
    assert_eq!(body["custom_instructions"].as_str(), Some(""));

    // Upload limit
    let resp = client.get(format!("{base_url}/api/settings/upload-limit")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["limit_bytes"].as_u64(), Some(10737418240));

    // VRAM budget
    let resp = client.get(format!("{base_url}/api/settings/vram-budget")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["vram_budget_mb"].as_u64(), Some(4096));

    // Download transport
    let resp = client.get(format!("{base_url}/api/settings/download-transport")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["transport"].as_str(), Some("direct"));

    // Embedding model
    let resp = client.get(format!("{base_url}/api/settings/embedding-model")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["model"].is_null());

    // OpenAI auto switch
    let resp = client.get(format!("{base_url}/api/settings/openai-auto-switch")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["enabled"].as_bool(), Some(false));

    // Chat preferences migrate
    let resp = client.post(format!("{base_url}/api/settings/chat-preferences/migrate")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"].as_str(), Some("ok"));

    // Studio capabilities
    let resp = client.get(format!("{base_url}/api/studio/download-transport-capabilities")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["direct"].as_bool(), Some(true));
    assert_eq!(body["hf_transfer"].as_bool(), Some(true));

    // Export status
    let resp = client.get(format!("{base_url}/api/export/status")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["active"].as_bool(), Some(false));
    assert_eq!(body["status"].as_str(), Some("idle"));

    // Llama update status
    let resp = client.get(format!("{base_url}/api/llama/update-status")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["update_available"].as_bool(), Some(false));
    assert_eq!(body["current_version"].as_str(), Some("0.1.0"));

    // Llama cpp path
    let resp = client.get(format!("{base_url}/api/settings/llama-cpp-path")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["path"].is_null());
    assert_eq!(body["source"].as_str(), Some("default"));
    assert_eq!(body["editable"].as_bool(), Some(false));
    assert_eq!(body["available"].as_bool(), Some(true));
    assert_eq!(body["resolved_binary"].as_str(), Some("native/sloth-vulkan"));
    assert!(body["environment_variable"].is_null());
    assert_eq!(body["reload_required"].as_bool(), Some(false));

    // Llama backend
    let resp = client.get(format!("{base_url}/api/llama/backend")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["supported"].as_bool(), Some(false));
    assert_eq!(body["backend"].as_str(), Some("vulkan"));
    assert_eq!(body["options"][0]["backend"].as_str(), Some("vulkan"));
    assert_eq!(body["job"]["state"].as_str(), Some("idle"));

    // System GPU fields verification
    let resp = client.get(format!("{base_url}/api/system")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    let gpu_dev = &body["gpu"]["devices"][0];
    assert!(gpu_dev["name"].as_str().unwrap().contains("AMD Radeon RX 570"));
    assert!(gpu_dev["gpu_name"].as_str().unwrap().contains("AMD Radeon RX 570"));
    assert_eq!(gpu_dev["memory_total_gb"].as_f64(), Some(4.0));
    assert_eq!(gpu_dev["vram_total_gb"].as_f64(), Some(4.0));
    assert_eq!(gpu_dev["backend"].as_str(), Some("vulkan"));

    let inf_gpu_dev = &body["inference_gpu"]["devices"][0];
    assert!(inf_gpu_dev["name"].as_str().unwrap().contains("AMD Radeon RX 570"));
    assert!(inf_gpu_dev["gpu_name"].as_str().unwrap().contains("AMD Radeon RX 570"));
    assert_eq!(inf_gpu_dev["memory_total_gb"].as_f64(), Some(4.0));

    // RAG knowledge bases: формат фронтенда (camelCase) и честный признак недоступности RAG
    let resp = client.get(format!("{base_url}/api/rag/knowledge-bases")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["knowledgeBases"].as_array().map(|a| a.len()), Some(0));
    assert_eq!(body["ragAvailable"].as_bool(), Some(false));
}

#[tokio::test]
async fn test_hub_transport_statuses() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. GET /api/hub/transport-status
    let resp = client
        .get(format!("{base_url}/api/hub/transport-status?repo_id=unsloth/test-model"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"].as_str(), Some("idle"));
    assert_eq!(body["transport"].as_str(), Some("direct"));
    assert_eq!(body["active"].as_bool(), Some(false));
    assert_eq!(body["has_partial"].as_bool(), Some(false));
    assert!(body["last_transport"].is_null());
    assert_eq!(body["resumable"].as_bool(), Some(false));

    // 2. GET /api/hub/datasets/transport-status
    let resp = client
        .get(format!("{base_url}/api/hub/datasets/transport-status?repo_id=unsloth/test-dataset"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"].as_str(), Some("idle"));
    assert_eq!(body["transport"].as_str(), Some("direct"));
    assert_eq!(body["active"].as_bool(), Some(false));
    assert_eq!(body["has_partial"].as_bool(), Some(false));
    assert!(body["last_transport"].is_null());
    assert_eq!(body["resumable"].as_bool(), Some(false));
}

#[tokio::test]
async fn test_hub_gguf_download_progress() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/api/hub/gguf-download-progress?repo_id=unsloth/Llama-3.2-3B-Instruct-GGUF&variant=Q4_K_M&expected_bytes=2023751680"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["cache_measured"].as_bool(), Some(true));
    assert!(body["progress"].as_f64().is_some());
    assert!(body["downloaded_bytes"].as_u64().is_some());
    assert!(body["completed_bytes"].as_u64().is_some());
    assert!(body["expected_bytes"].as_u64().is_some());
    assert!(body["complete_on_disk"].as_bool().is_some());
}

#[tokio::test]
async fn test_hub_datasets_download_lifecycle() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // Датасеты появятся на этапе 3: изменяющие запросы честно отвечают 501 с detail,
    // а статус показывает, что никакой загрузки нет (раньше был фальшивый "running")

    // 1. Start dataset download
    let start_payload = serde_json::json!({
        "repo_id": "unsloth/Open-Orca",
        "use_xet": false,
        "transport_mode": "http"
    });
    let resp = client
        .post(format!("{base_url}/api/hub/datasets/download"))
        .json(&start_payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_IMPLEMENTED);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["detail"].is_string());

    // 2. Dataset download status
    let resp = client
        .get(format!("{base_url}/api/hub/datasets/download-status?repo_id=unsloth/Open-Orca"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["state"].as_str(), Some("idle"));
    assert!(body["error"].is_null());

    // 3. Cancel dataset download
    let cancel_payload = serde_json::json!({
        "repo_id": "unsloth/Open-Orca",
        "generation": 1
    });
    let resp = client
        .post(format!("{base_url}/api/hub/datasets/download/cancel"))
        .json(&cancel_payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_IMPLEMENTED);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["detail"].is_string());

    // 4. Delete cached dataset
    let del_payload = serde_json::json!({
        "repo_id": "unsloth/Open-Orca"
    });
    let resp = client
        .delete(format!("{base_url}/api/hub/datasets/cached"))
        .json(&del_payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_IMPLEMENTED);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["detail"].is_string());
}

#[tokio::test]
async fn test_hub_delete_cached_model() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // Такой модели на диске нет: оба метода обязаны честно ответить 404, а не «deleted: true».
    // Успешное удаление и отказ вне папки моделей проверяются в tests/security_tests.rs.
    let payload = serde_json::json!({
        "repo_id": "unsloth/test-model",
        "variant": "Q4_K_M"
    });

    // 1. POST /api/hub/delete-cached
    let resp = client
        .post(format!("{base_url}/api/hub/delete-cached"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["detail"].is_string());

    // 2. DELETE /api/hub/delete-cached
    let resp = client
        .delete(format!("{base_url}/api/hub/delete-cached"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["detail"].is_string());
}

#[tokio::test]
async fn test_hub_scan_folders_crud() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. Initial list is empty
    let resp = client.get(format!("{base_url}/api/hub/scan-folders")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["folders"].as_array().map(|a| a.len()), Some(0));

    // 2. Add scan folder
    let add_payload = serde_json::json!({
        "path": "/home/rivergod/custom-models"
    });
    let resp = client
        .post(format!("{base_url}/api/hub/scan-folders"))
        .json(&add_payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let folder: serde_json::Value = resp.json().await.unwrap();
    let folder_id = folder["id"].as_u64().unwrap();
    assert_eq!(folder["path"].as_str(), Some("/home/rivergod/custom-models"));

    // 3. List shows added folder
    let resp = client.get(format!("{base_url}/api/hub/scan-folders")).send().await.unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["folders"].as_array().map(|a| a.len()), Some(1));

    // 4. Delete scan folder
    let resp = client
        .delete(format!("{base_url}/api/hub/scan-folders/{folder_id}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    // 5. List is empty again
    let resp = client.get(format!("{base_url}/api/hub/scan-folders")).send().await.unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["folders"].as_array().map(|a| a.len()), Some(0));
}

#[tokio::test]
async fn test_model_download_flexible_payload() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // Flexible payload using camelCase and alternate names:
    // repoId instead of repo_id, quant instead of gguf_variant, file_name instead of filename,
    // revision, transportMode, useXet
    let flex_payload = serde_json::json!({
        "repoId": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "quant": "Q4_K_M",
        "file_name": "model-Q4_K_M.gguf",
        "revision": "main",
        "transportMode": "http",
        "useXet": false,
        "scopeId": null
    });

    let resp = client
        .post(format!("{base_url}/api/hub/download"))
        .json(&flex_payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let start_resp: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(start_resp["state"].as_str(), Some("running"));
    assert_eq!(start_resp["accepted"].as_bool(), Some(true));
    assert_eq!(start_resp["job_key"].as_str(), Some("job-default"));
    assert_eq!(start_resp["generation"].as_u64(), Some(1));
    assert_eq!(start_resp["transport"].as_str(), Some("http"));

    // Check download status
    let resp = client
        .get(format!("{base_url}/api/hub/download-status?repo_id=unsloth/Llama-3.2-3B-Instruct-GGUF&gguf_variant=Q4_K_M"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let status_resp: serde_json::Value = resp.json().await.unwrap();
    assert!(status_resp["state"].as_str().is_some());
    assert!(status_resp["error"].is_null());

    // Check download progress
    let resp = client
        .get(format!("{base_url}/api/hub/download-progress?repo_id=unsloth/Llama-3.2-3B-Instruct-GGUF"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let prog_resp: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(prog_resp["cache_measured"].as_bool(), Some(true));
    assert!(prog_resp["expected_bytes"].as_u64().unwrap() > 0);

    // Cancel with flexible payload
    let cancel_payload = serde_json::json!({
        "repo_id": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "gguf_variant": "Q4_K_M",
        "generation": 1
    });
    let resp = client
        .post(format!("{base_url}/api/hub/download/cancel"))
        .json(&cancel_payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let cancel_resp: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(cancel_resp["state"].as_str(), Some("cancelled"));
}

#[tokio::test]
async fn test_hub_token_validate_and_dataset_utils() {
    let (base_url, _) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. POST /api/hub/token/validate
    let resp = client
        .post(format!("{base_url}/api/hub/token/validate"))
        .header("authorization", "Bearer hf_testtoken12345")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let val: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(val["status"].as_str(), Some("valid"));

    // 2. POST /api/hub/datasets/check-format: до этапа 3 честный 501 вместо
    //    выдуманного «alpaca, 1000 строк» для любого файла
    let resp = client
        .post(format!("{base_url}/api/hub/datasets/check-format"))
        .json(&serde_json::json!({"dataset_name": "yahma/alpaca-cleaned"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_IMPLEMENTED);
    let cf: serde_json::Value = resp.json().await.unwrap();
    assert!(cf["detail"].is_string());

    // 3. POST /api/hub/delete-impact
    let resp = client
        .post(format!("{base_url}/api/hub/delete-impact"))
        .json(&serde_json::json!({"repo_id": "unsloth/Llama-3.2-3B-Instruct-GGUF"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let impact: serde_json::Value = resp.json().await.unwrap();
    assert!(impact["reclaimed_bytes"].as_u64().is_some());
    assert!(impact["affected_models"].as_array().is_some());

    // 4. GET /api/hub/orphan-companions
    let resp = client
        .get(format!("{base_url}/api/hub/orphan-companions"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let orphans: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(orphans["total_bytes"].as_u64(), Some(0));
}

#[tokio::test]
async fn test_audited_endpoints_and_schemas() {
    let (base_url, _state) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. Hardware schema check
    let resp = client.get(format!("{base_url}/api/system/hardware")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let hw: serde_json::Value = resp.json().await.unwrap();
    assert!(hw["gpu"]["gpu_name"].is_string());
    assert!(hw["gpu"]["vram_total_gb"].is_number());
    assert!(hw["versions"]["vulkan"].is_string());
    assert!(hw["export_supported"].is_boolean());

    // 2. Personalization schema check
    let resp = client.get(format!("{base_url}/api/settings/personalization")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let pers: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(pers["version"].as_u64(), Some(1));
    assert!(pers["profile"].is_object());
    assert!(pers["appearance"].is_object());

    // 3. Upload limit schema check
    let resp = client.get(format!("{base_url}/api/settings/upload-limit")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let ul: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(ul["max_upload_size_mb"].as_u64(), Some(10240));
    assert!(ul["max_upload_size_bytes"].is_number());

    // 4. VRAM budget fraction
    let resp = client
        .put(format!("{base_url}/api/settings/vram-budget"))
        .json(&serde_json::json!({ "fraction": 0.85 }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let vb: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(vb["fraction"].as_f64(), Some(0.85));

    // 5. Studio download transport capabilities
    let resp = client.get(format!("{base_url}/api/studio/download-transport-capabilities")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let tc: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(tc["auto_resolves_to"].as_str(), Some("http"));
    assert_eq!(tc["http"]["available"].as_bool(), Some(true));
    assert_eq!(tc["xet"]["available"].as_bool(), Some(false));

    // 6. Model path and reveal: несуществующая модель — честный 404 с пояснением в detail
    let resp = client.get(format!("{base_url}/api/models/cached-model-path?repo_id=test/model")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
    let mp: serde_json::Value = resp.json().await.unwrap();
    assert!(mp["detail"].is_string());

    // Открыть в файловом менеджере можно только существующую модель
    let resp = client.post(format!("{base_url}/api/models/reveal-cached-model"))
        .json(&serde_json::json!({ "repo_id": "test/model" }))
        .send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);

    // 7. Model progress under /models/
    let resp = client.get(format!("{base_url}/api/models/download-progress?job_id=nonexistent")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let resp = client.get(format!("{base_url}/api/models/gguf-download-progress?repo_id=test/m&variant=Q4_K_M")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    // 8. Picker chat template
    let resp = client.post(format!("{base_url}/api/picker/validate-chat-template"))
        .json(&serde_json::json!({ "template": "{{ bos_token }}" }))
        .send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let tmpl_val: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(tmpl_val["valid"].as_bool(), Some(true));

    let resp = client.get(format!("{base_url}/api/picker/chat-template/model-1")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    // 9. Inference cancel, count tokens, monitor reset
    let resp = client.post(format!("{base_url}/api/inference/cancel")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let resp = client.post(format!("{base_url}/api/inference/chat/count_tokens"))
        .json(&serde_json::json!({ "messages": [{"role": "user", "content": "hello"}] }))
        .send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let resp = client.delete(format!("{base_url}/api/inference/monitor")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    // 10. Chat messages PUT & detail PUT, project PATCH
    let resp = client.put(format!("{base_url}/api/chat/threads/th-1/messages")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let resp = client.put(format!("{base_url}/api/chat/threads/th-1/messages/msg-1")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let resp = client.patch(format!("{base_url}/api/chat/projects/prj-1")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    // 11. Auth API Keys
    let resp = client.get(format!("{base_url}/api/auth/api-keys")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let resp = client.post(format!("{base_url}/api/auth/api-keys"))
        .json(&serde_json::json!({ "name": "Test Key" }))
        .send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let k: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(k["name"].as_str(), Some("Test Key"));
    let resp = client.delete(format!("{base_url}/api/auth/api-keys/key-1")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    // 12. Profile stats, release notes, changelog
    let resp = client.get(format!("{base_url}/api/profile/stats")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let resp = client.get(format!("{base_url}/api/studio/release-notes")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let resp = client.get(format!("{base_url}/api/llama/update-changelog")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    // 13. Export actions: экспорт появится на этапе 3, до этого честный 501 вместо
    //     «успеха» с job_id, после которого файл не создавался
    let resp = client.get(format!("{base_url}/api/export/logs")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let resp = client.post(format!("{base_url}/api/export/export/gguf")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn test_model_picker_and_deep_integration_audit() {
    let (base_url, _state) = spawn_test_server().await;
    let client = reqwest::Client::new();

    // 1. FolderBrowser: GET /api/models/browse-folders
    let resp = client.get(format!("{base_url}/api/models/browse-folders")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let fb: serde_json::Value = resp.json().await.unwrap();
    assert!(fb["entries"].is_array(), "FolderBrowser response MUST contain entries array");
    assert!(fb["current"].is_string());
    assert!(fb["suggestions"].is_array());
    assert!(fb["model_files_here"].is_number());

    // 2. Memory Estimation: POST /api/inference/estimate-memory
    let resp = client
        .post(format!("{base_url}/api/inference/estimate-memory"))
        .json(&serde_json::json!({
            "model_path": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            "n_ctx": 8192
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let est: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(est["available"].as_bool(), Some(true));
    assert!(est["weights_bytes"].as_u64().unwrap() > 0);
    assert!(est["kv_bytes"].as_u64().unwrap() > 0);
    assert!(est["total_bytes"].as_u64().unwrap() > 0);
    assert_eq!(est["fits"].as_bool(), Some(true));

    // 3. Llama Flags Catalog: GET /api/inference/llama-flags
    let resp = client.get(format!("{base_url}/api/inference/llama-flags")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let flags: serde_json::Value = resp.json().await.unwrap();
    assert!(flags["flags"].is_object(), "flags MUST be a dictionary/object of flag descriptions");
    assert!(flags["managed"].is_array());
    assert!(flags["switch_flags"].is_array());
    assert_eq!(flags["probe_ok"].as_bool(), Some(true));

    // 4. Overrides: PUT & GET /api/settings/openai-auto-switch/overrides
    let test_overrides = serde_json::json!({
        "models": {
            "llama-1b": { "temperature": 0.5, "n_ctx": 4096 }
        }
    });
    let resp = client
        .put(format!("{base_url}/api/settings/openai-auto-switch/overrides"))
        .json(&serde_json::json!({ "overrides": test_overrides }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let resp = client.get(format!("{base_url}/api/settings/openai-auto-switch/overrides")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let ov_resp: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(ov_resp["overrides"], test_overrides);

    // 5. Chat Count Tokens: POST /api/inference/chat/count_tokens
    let resp = client
        .post(format!("{base_url}/api/inference/chat/count_tokens"))
        .json(&serde_json::json!({
            "model": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            "messages": [
                { "role": "user", "content": "How does SlothForge integrate Vulkan?" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let count_res: serde_json::Value = resp.json().await.unwrap();
    assert!(count_res["input_tokens"].as_u64().unwrap() > 0, "input_tokens must be reported");
    assert_eq!(count_res["model"].as_str(), Some("Llama-3.2-1B-Instruct-Q4_K_M.gguf"));

    // 6. Picker Chat Template: GET /api/picker/chat-template/:id
    let resp = client.get(format!("{base_url}/api/picker/chat-template/llama-3.2-1b")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let tmpl: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(tmpl["model_name"].as_str(), Some("llama-3.2-1b"));

    // 7. Training Start Request Status: GET /api/train/start-requests/:id
    let resp = client.get(format!("{base_url}/api/train/start-requests/req-999")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let req_stat: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(req_stat["start_request_id"].as_str(), Some("req-999"));
    assert_eq!(req_stat["state"].as_str(), Some("accepted"));

    // 8. Delete Impact: POST /api/hub/delete-impact for local model
    let resp = client
        .post(format!("{base_url}/api/hub/delete-impact"))
        .json(&serde_json::json!({
            "repo_id": "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let del_impact: serde_json::Value = resp.json().await.unwrap();
    assert!(del_impact["reclaimed_bytes"].as_u64().unwrap() > 500_000_000, "Should report real file size");
    assert_eq!(del_impact["reclaimed_bytes"], del_impact["freed_bytes"]);

    // 9. Inference Ejection / Unload: POST /api/inference/unload
    let resp = client.post(format!("{base_url}/api/inference/unload")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let resp = client.get(format!("{base_url}/api/inference/status")).send().await.unwrap();
    let inf_status: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(inf_status["active_model"].as_str(), Some(""));

    // 10. Personalization Persistence: PUT & GET /api/settings/personalization
    let updated_profile = serde_json::json!({
        "version": 1,
        "profile": {
            "displayName": "AuditTester",
            "nickname": "auditor",
            "avatarDataUrl": null,
            "avatarShape": "rounded",
            "showGreetingSloth": false
        },
        "appearance": {
            "theme": "dark",
            "palette": "high-contrast",
            "language": "ru",
            "customization": {}
        },
        "user_name": "AuditTester",
        "custom_instructions": "Always verify integration contracts",
        "saved": true
    });
    let resp = client
        .put(format!("{base_url}/api/settings/personalization"))
        .json(&updated_profile)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let resp = client.get(format!("{base_url}/api/settings/personalization")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let get_pers: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(get_pers["profile"]["displayName"].as_str(), Some("AuditTester"));
    assert_eq!(get_pers["appearance"]["language"].as_str(), Some("ru"));

    // 11. API Keys Persistence: POST, GET, DELETE
    let resp = client
        .post(format!("{base_url}/api/auth/api-keys"))
        .json(&serde_json::json!({ "name": "Production Deploy Key" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let created_key: serde_json::Value = resp.json().await.unwrap();
    let key_id = created_key["id"].as_str().unwrap();

    let resp = client.get(format!("{base_url}/api/auth/api-keys")).send().await.unwrap();
    let keys_list: serde_json::Value = resp.json().await.unwrap();
    let arr = keys_list["keys"].as_array().unwrap();
    assert!(arr.iter().any(|k| k["id"].as_str() == Some(key_id)));

    let resp = client.delete(format!("{base_url}/api/auth/api-keys/{key_id}")).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let resp = client.get(format!("{base_url}/api/auth/api-keys")).send().await.unwrap();
    let keys_list_after: serde_json::Value = resp.json().await.unwrap();
    let arr_after = keys_list_after["keys"].as_array().unwrap();
    assert!(!arr_after.iter().any(|k| k["id"].as_str() == Some(key_id)));
}

