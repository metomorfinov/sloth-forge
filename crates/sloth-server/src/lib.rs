pub mod chat;
pub mod cluster;
pub mod handlers;
pub mod models;
pub mod state;
pub mod training;

use axum::{
    routing::{get, post},
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
        .route("/vram", get(handle_vram))
        .route("/hardware", get(handle_hardware))
        .route("/models", get(handle_models))
        .route("/train/start", post(handle_train_start))
        .route("/train/stop", post(handle_train_stop))
        .route("/train/status", get(handle_train_status))
        .route("/cluster/worker/register", post(cluster::handle_register_worker))
        .route("/cluster/sync_grad", post(cluster::handle_sync_grad))
        .route("/cluster/status", get(cluster::handle_cluster_status));

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
