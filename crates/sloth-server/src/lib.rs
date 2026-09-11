pub mod api;
pub mod chat;
pub mod cluster;
pub mod handlers;
pub mod hub;
pub mod models;
pub mod state;
pub mod training;

use axum::{
    routing::{get, post, put},
    Router,
};
use handlers::*;
use sloth_vulkan_sys::VulkanContext;
pub use state::AppState;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

pub fn server_version() -> &'static str {
    "0.1.0"
}

pub fn create_router(state: Arc<AppState>, static_dir: Option<PathBuf>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // API Routes (/api/*)
    let api_router = Router::new()
        // Unsloth Core Health & Auth
        .route("/health", get(api::handle_health))
        .route("/auth/status", get(api::handle_auth_status))
        .route("/studio/install-source", get(api::handle_install_source))
        .route("/studio/update-status", get(api::handle_update_status))
        .route("/studio/download-transport-capabilities", get(api::handle_studio_download_transport_capabilities))
        // Model Endpoints
        .route("/models", get(handle_models))
        .route("/models/list", get(handle_models))
        .route("/models/local", get(handle_models))
        .route("/models/scan-folders", get(api::handle_models_scan_folders))
        .route("/models/recommended-folders", get(api::handle_models_recommended_folders))
        .route("/models/loras", get(api::handle_models_loras))
        .route("/models/check-vision/:id", get(api::handle_check_vision))
        .route("/models/check-embedding/:id", get(api::handle_check_embedding))
        .route("/models/config/:id", get(api::handle_model_config))
        // Hub & Model Downloading
        .route("/hub/cached-models", get(hub::handle_cached_models))
        .route("/hub/cached-gguf", get(hub::handle_cached_gguf))
        .route("/hub/local", get(hub::handle_hub_local))
        .route("/hub/hidden-models", get(hub::handle_hidden_models))
        .route("/hub/active-downloads", get(hub::handle_active_downloads))
        .route("/hub/gguf-variants", get(hub::handle_gguf_variants))
        .route("/hub/download", post(hub::handle_download_start))
        .route("/hub/download/cancel", post(hub::handle_download_cancel))
        .route("/hub/download-status", get(hub::handle_download_status))
        .route("/hub/download-progress", get(hub::handle_download_progress))
        .route("/hub/scan-folders", get(hub::handle_hub_scan_folders))
        .route("/hub/delete-impact", get(hub::handle_hub_delete_impact))
        .route("/hub/orphan-companions", get(hub::handle_hub_orphan_companions))
        // Datasets
        .route("/hub/datasets/cached", get(hub::handle_datasets_cached))
        .route("/hub/datasets/local", get(hub::handle_datasets_local))
        .route("/hub/datasets/active-downloads", get(hub::handle_datasets_active_downloads))
        .route("/hub/datasets/download-progress", get(hub::handle_datasets_download_progress))
        .route("/hub/datasets/local-options", get(hub::handle_datasets_local_options))
        // Training Endpoints
        .route("/train/status", get(api::handle_train_status))
        .route("/train/progress", get(api::handle_train_progress).post(api::handle_train_progress))
        .route("/train/start", post(handle_train_start))
        .route("/train/stop", post(handle_train_stop))
        .route("/train/reset", post(api::handle_train_reset))
        .route("/train/runs", get(api::handle_train_runs))
        .route("/train/runs/:id", get(api::handle_train_run_detail))
        .route("/train/metrics", get(api::handle_train_metrics))
        .route("/train/start-requests/:id/acknowledge", post(api::handle_start_request_ack))
        .route("/train/start-requests/:id/cancel", post(api::handle_start_request_cancel))
        .route("/train/hardware", get(handle_hardware))
        .route("/train/diffusion/status", get(api::handle_diffusion_status))
        // Chat & Inference Endpoints
        .route("/inference/status", get(api::handle_inference_status))
        .route("/inference/monitor", get(api::handle_inference_monitor))
        .route("/inference/load", post(api::handle_inference_load))
        .route("/inference/unload", post(api::handle_inference_unload))
        .route("/inference/load-progress", get(api::handle_inference_load_progress))
        .route("/inference/validate", post(api::handle_inference_validate))
        .route("/inference/llama-flags", get(api::handle_inference_llama_flags))
        .route("/inference/estimate-memory", post(api::handle_inference_estimate_memory))
        .route("/inference/video/status", get(api::handle_inference_video_status))
        .route("/inference/images/status", get(api::handle_inference_images_status))
        .route("/inference/chat", post(chat::handle_chat_completions))
        .route("/inference/chat/completions", post(chat::handle_chat_completions))
        .route("/chat/threads", get(api::handle_chat_threads_get).post(api::handle_chat_threads_post).delete(api::handle_chat_threads_delete))
        .route("/chat/threads/:id", get(api::handle_chat_thread_detail).put(api::handle_chat_thread_update).patch(api::handle_chat_thread_update).delete(api::handle_chat_thread_delete))
        .route("/chat/threads/:id/messages", get(api::handle_chat_thread_messages).post(api::handle_chat_thread_messages_post))
        .route("/chat/projects", get(api::handle_chat_projects_get).post(api::handle_chat_projects_post))
        .route("/chat/projects/:id", get(api::handle_chat_projects_detail).delete(api::handle_chat_projects_delete))
        .route("/chat/settings", get(api::handle_chat_settings_get).put(api::handle_chat_settings_put).post(api::handle_chat_settings_put))
        .route("/chat/settings/compare-and-set", post(api::handle_chat_settings_put))
        .route("/chat/count", get(api::handle_chat_count))
        .route("/chat/attachments", get(api::handle_chat_attachments))
        // System & Hardware Endpoints
        .route("/system", get(api::handle_system))
        .route("/system/hardware", get(api::handle_system_hardware))
        .route("/vram", get(handle_vram))
        .route("/hardware", get(handle_hardware))
        // Cluster Endpoints
        .route("/cluster/worker/register", post(cluster::handle_register_worker))
        .route("/cluster/sync_grad", post(cluster::handle_sync_grad))
        .route("/cluster/status", get(cluster::handle_cluster_status))
        // Settings & Credentials Endpoints
        .route("/auth/refresh", post(api::handle_auth_refresh).get(api::handle_auth_refresh))
        .route("/auth/logout", post(api::handle_auth_logout))
        .route("/settings/personalization", get(api::handle_settings_personalization).put(api::handle_settings_personalization))
        .route("/settings/upload-limit", get(api::handle_settings_upload_limit).put(api::handle_settings_upload_limit_put))
        .route("/settings/vram-budget", get(api::handle_settings_vram_budget).put(api::handle_settings_vram_budget_put))
        .route("/settings/download-transport", get(api::handle_settings_download_transport).put(api::handle_settings_download_transport))
        .route("/settings/embedding-model", get(api::handle_settings_embedding_model).put(api::handle_settings_embedding_model).delete(api::handle_settings_embedding_model))
        .route("/settings/embedding-model/unload", post(api::handle_settings_embedding_model))
        .route("/settings/openai-auto-switch", get(api::handle_settings_openai_auto_switch).put(api::handle_settings_openai_auto_switch))
        .route("/settings/openai-auto-switch/overrides", get(api::handle_settings_openai_auto_switch_overrides))
        .route("/settings/chat-preferences/migrate", post(api::handle_settings_chat_preferences))
        .route("/settings/chat-preferences", get(api::handle_settings_chat_preferences).put(api::handle_settings_chat_preferences).post(api::handle_settings_chat_preferences))
        .route("/settings/model-memory", get(api::handle_settings_model_memory).put(api::handle_settings_model_memory))
        .route("/settings/last-local-model", get(api::handle_settings_last_local_model).put(api::handle_settings_last_local_model))
        .route("/settings/llama-cpp-path", get(api::handle_settings_llama_cpp_path).put(api::handle_settings_llama_cpp_path))
        .route("/settings/hugging-face-cache", get(api::handle_settings_hugging_face_cache).put(api::handle_settings_hugging_face_cache))
        .route("/settings/keyless-api-access", get(api::handle_settings_keyless_api_access).put(api::handle_settings_keyless_api_access))
        .route("/settings/remote-access", get(api::handle_settings_remote_access))
        .route("/settings/preview-sharing", get(api::handle_settings_preview_sharing).put(api::handle_settings_preview_sharing))
        .route("/settings/coding-agents", get(api::handle_settings_coding_agents))
        .route("/settings/current-date-prompt", get(api::handle_settings_current_date_prompt))
        .route("/settings/debug/logs/sources", get(api::handle_settings_debug_logs_sources))
        .route("/settings/debug/logs", get(api::handle_settings_debug_logs))
        .route("/settings/hugging-face-token", get(api::handle_hf_token_get).put(api::handle_hf_token_put).delete(api::handle_hf_token_delete))
        .route("/settings/hugging-face-token/migrate", put(api::handle_hf_token_put))
        .route("/settings/generation-presets/:kind", get(api::handle_generation_presets).put(api::handle_generation_presets))
        .route("/settings/generation-presets/:kind/custom", get(api::handle_generation_presets).put(api::handle_generation_presets))
        .route("/providers/registry", get(api::handle_providers_registry))
        .route("/providers", get(api::handle_providers_list))
        .route("/providers/", get(api::handle_providers_list))
        // Export & Llama & RAG
        .route("/export/status", get(api::handle_export_status))
        .route("/llama/update-status", get(api::handle_llama_update_status))
        .route("/llama/update", post(api::handle_llama_update))
        .route("/llama/backend", get(api::handle_llama_backend).post(api::handle_llama_backend))
        .route("/rag/knowledge-bases", get(api::handle_rag_knowledge_bases))
        // Fallback for unmatched /api/* calls so they never receive index.html
        .fallback(api::handle_api_not_found);

    // Base App Router with all endpoints
    let mut app = Router::new()
        .nest("/api", api_router)
        // Root WebSocket & OpenAI Endpoints
        .route("/ws/telemetry", get(handle_ws_telemetry))
        .route("/v1/chat/completions", post(chat::handle_chat_completions))
        .route("/v1/models", get(handle_models))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Static asset serving with SPA index.html fallback
    if let Some(dist_dir) = static_dir {
        if dist_dir.exists() {
            let index_html = dist_dir.join("index.html");
            let serve_dir = ServeDir::new(&dist_dir).fallback(ServeFile::new(index_html));
            app = app.fallback_service(serve_dir);
        }
    }

    app
}

pub async fn run_server_with_listener(
    listener: tokio::net::TcpListener,
    addr: SocketAddr,
    static_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    let vk_ctx = VulkanContext::init(true).ok();
    if let Some(ref ctx) = vk_ctx {
        info!("Initialized Vulkan device: {}", ctx.device_name());
    } else {
        warn!("Running in fallback mode without hardware Vulkan context");
    }

    let models_dir = PathBuf::from("models");
    let state = Arc::new(AppState::new(vk_ctx, models_dir, static_dir.clone()));

    let app = create_router(state, static_dir);

    info!("SlothForge Axum server listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

pub async fn run_server(addr: SocketAddr, static_dir: Option<PathBuf>) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    run_server_with_listener(listener, addr, static_dir).await
}
