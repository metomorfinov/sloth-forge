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
    let static_dir = PathBuf::from("/home/rivergod/.gemini/antigravity/scratch/sloth-forge/frontend/dist");
    let models_dir = PathBuf::from("models");

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
    assert_eq!(start_resp["status"].as_str(), Some("started"));
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
    assert_eq!(start_resp["status"].as_str(), Some("started"));
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
    assert_eq!(start_resp["status"].as_str(), Some("started"));

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
