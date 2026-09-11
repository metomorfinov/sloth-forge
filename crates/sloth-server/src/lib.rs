pub mod api;
pub mod chat;
pub mod cluster;
pub mod handlers;
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
        // Model Endpoints
        .route("/models", get(handle_models))
        .route("/models/list", get(handle_models))
        .route("/models/local", get(handle_models))
        .route("/models/check-vision/:id", get(api::handle_check_vision))
        .route("/models/check-embedding/:id", get(api::handle_check_embedding))
        .route("/models/config/:id", get(api::handle_model_config))
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
        // Chat & Inference Endpoints
        .route("/inference/chat", post(chat::handle_chat_completions))
        .route("/inference/chat/completions", post(chat::handle_chat_completions))
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
        .route("/settings/hugging-face-token", get(api::handle_hf_token_get).put(api::handle_hf_token_put).delete(api::handle_hf_token_delete))
        .route("/settings/hugging-face-token/migrate", put(api::handle_hf_token_put))
        .route("/settings/generation-presets/:kind", get(api::handle_generation_presets).put(api::handle_generation_presets))
        .route("/settings/generation-presets/:kind/custom", get(api::handle_generation_presets).put(api::handle_generation_presets))
        .route("/providers/registry", get(api::handle_providers_registry))
        .route("/providers", get(api::handle_providers_list))
        .route("/providers/", get(api::handle_providers_list))
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
