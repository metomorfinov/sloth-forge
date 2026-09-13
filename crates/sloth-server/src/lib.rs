pub mod api;
pub mod chat;
pub mod chat_history;
pub mod cluster;
pub mod error;
pub mod handlers;
pub mod hub;
pub mod locations;
pub mod models;
pub mod paths;
pub mod profile_stats;
pub mod security;
pub mod settings;
pub mod state;
pub mod store;
pub mod training;
pub mod unavailable;

use axum::{
    routing::{delete, get, post, put},
    Router,
};
use handlers::*;
use sloth_vulkan_sys::VulkanContext;
pub use state::AppState;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

pub fn server_version() -> &'static str {
    "0.1.0"
}

/// Порт сервера по умолчанию.
pub const DEFAULT_PORT: u16 = 3000;

pub fn create_router(state: Arc<AppState>, static_dir: Option<PathBuf>) -> Router {
    // Ответы API читаются только страницами с этого компьютера (см. security.rs)
    let cors = security::cors_layer();

    // API Routes (/api/*)
    let api_router = Router::new()
        // Core Health & Auth
        .route("/health", get(api::handle_health))
        .route("/auth/status", get(api::handle_auth_status))
        .route("/auth/login", post(api::handle_auth_login))
        .route("/auth/change-password", post(api::handle_auth_change_password))
        .route("/auth/desktop-initial-password", post(api::handle_auth_desktop_initial_password))
        .route("/auth/api-keys", get(api::handle_auth_api_keys_get).post(api::handle_auth_api_keys_post))
        .route("/auth/api-keys/:id", delete(api::handle_auth_api_keys_delete))
        .route("/shutdown", post(api::handle_shutdown))
        .route("/profile/stats", get(profile_stats::handle_profile_stats))
        .route("/studio/install-source", get(api::handle_install_source))
        .route("/studio/update-status", get(api::handle_update_status))
        .route("/studio/release-notes", get(api::handle_studio_release_notes))
        .route("/studio/download-transport-capabilities", get(api::handle_studio_download_transport_capabilities))
        // Model Endpoints
        .route("/models", get(handle_models))
        .route("/models/list", get(handle_models))
        .route("/models/local", get(handle_models))
        .route("/models/cached-model-path", get(api::handle_model_cached_path))
        .route("/models/reveal-cached-model", post(api::handle_model_reveal))
        .route("/models/kv-cache-estimate", get(api::handle_model_kv_cache_estimate))
        .route("/models/browse-folders", get(api::handle_model_browse_folders))
        .route("/models/checkpoints", get(api::handle_models_checkpoints))
        .route("/models/export-size", get(api::handle_models_export_size))
        .route("/models/delete-finetuned", delete(api::handle_models_delete_finetuned))
        .route("/models/scan-folders", get(api::handle_models_scan_folders).post(hub::handle_hub_add_scan_folder))
        .route("/models/scan-folders/:id", delete(hub::handle_hub_delete_scan_folder))
        .route("/models/recommended-folders", get(api::handle_models_recommended_folders))
        .route("/models/loras", get(api::handle_models_loras))
        .route("/models/download-progress", get(hub::handle_download_progress))
        .route("/models/gguf-download-progress", get(hub::handle_gguf_download_progress))
        .route("/models/gguf-variants", get(hub::handle_gguf_variants))
        .route("/models/download", post(hub::handle_download_start))
        .route("/models/download/cancel", post(hub::handle_download_cancel))
        .route("/models/download-status", get(hub::handle_download_status))
        .route("/models/transport-status", get(hub::handle_transport_status))
        .route("/models/check-vision/:id", get(api::handle_check_vision))
        .route("/models/check-embedding/:id", get(api::handle_check_embedding))
        .route("/models/config/:id", get(api::handle_model_config))
        // Model Picker
        .route("/picker/validate-chat-template", post(api::handle_picker_validate_chat_template))
        .route("/picker/chat-template/:id", get(api::handle_picker_chat_template))
        // Hub & Model Downloading
        .route("/hub/transport-status", get(hub::handle_transport_status))
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
        .route("/hub/gguf-download-progress", get(hub::handle_gguf_download_progress))
        .route("/hub/delete-cached", post(hub::handle_delete_cached).delete(hub::handle_delete_cached))
        .route("/hub/scan-folders", get(hub::handle_hub_scan_folders).post(hub::handle_hub_add_scan_folder))
        .route("/hub/scan-folders/:id", delete(hub::handle_hub_delete_scan_folder))
        .route("/hub/delete-impact", get(hub::handle_hub_delete_impact).post(hub::handle_hub_delete_impact_post))
        .route("/hub/orphan-companions", get(hub::handle_hub_orphan_companions))
        .route("/hub/token/validate", post(hub::handle_token_validate))
        // Datasets
        .route("/hub/datasets/transport-status", get(hub::handle_datasets_transport_status))
        .route("/hub/datasets/cached", get(hub::handle_datasets_cached).delete(hub::handle_datasets_cached_delete))
        .route("/hub/datasets/local", get(hub::handle_datasets_local))
        .route("/hub/datasets/active-downloads", get(hub::handle_datasets_active_downloads))
        .route("/hub/datasets/download", post(hub::handle_datasets_download))
        .route("/hub/datasets/download/cancel", post(hub::handle_datasets_download_cancel))
        .route("/hub/datasets/download-status", get(hub::handle_datasets_download_status))
        .route("/hub/datasets/download-progress", get(hub::handle_datasets_download_progress))
        .route("/hub/datasets/local-options", get(hub::handle_datasets_local_options).post(hub::handle_datasets_local_options))
        .route("/hub/datasets/check-format", post(hub::handle_datasets_check_format))
        .route("/hub/datasets/upload", post(hub::handle_datasets_upload))
        .route("/hub/datasets/ai-assist-mapping", post(hub::handle_datasets_ai_assist_mapping))
        // Training Endpoints
        .route("/train/status", get(api::handle_train_status))
        .route("/train/progress", get(api::handle_train_progress).post(api::handle_train_progress))
        .route("/train/start", post(handle_train_start))
        .route("/train/stop", post(handle_train_stop))
        .route("/train/reset", post(api::handle_train_reset))
        .route("/train/runs", get(api::handle_train_runs))
        .route("/train/runs/:id", get(api::handle_train_run_detail).patch(api::handle_train_run_detail).delete(api::handle_train_run_delete))
        .route("/train/metrics", get(api::handle_train_metrics))
        .route("/train/start-requests/:id", get(api::handle_start_request_get))
        .route("/train/start-requests/:id/acknowledge", post(api::handle_start_request_ack))
        .route("/train/start-requests/:id/cancel", post(api::handle_start_request_cancel))
        .route("/train/hardware", get(handle_hardware))
        .route("/train/diffusion/status", get(api::handle_diffusion_status))
        // Chat & Inference Endpoints
        .route("/inference/status", get(api::handle_inference_status))
        .route("/inference/monitor", get(api::handle_inference_monitor).delete(api::handle_inference_monitor_reset))
        .route("/inference/load", post(api::handle_inference_load))
        .route("/inference/unload", post(api::handle_inference_unload))
        .route("/inference/cancel", post(api::handle_inference_cancel))
        .route("/inference/active-generations", get(api::handle_inference_active_generations))
        .route("/inference/audio/stt/status", get(api::handle_inference_audio_stt_status))
        .route("/inference/audio/stt/unload", post(api::handle_inference_audio_stt_unload))
        .route("/inference/load-progress", get(api::handle_inference_load_progress))
        .route("/inference/validate", post(api::handle_inference_validate))
        .route("/inference/llama-flags", get(api::handle_inference_llama_flags))
        .route("/inference/estimate-memory", post(api::handle_inference_estimate_memory))
        .route("/inference/video/status", get(api::handle_inference_video_status))
        .route("/inference/images/status", get(api::handle_inference_images_status))
        .route("/inference/chat", post(chat::handle_chat_completions))
        .route("/inference/chat/completions", post(chat::handle_chat_completions))
        .route("/inference/chat/count_tokens", post(api::handle_inference_count_tokens))
        // Маршрутов /inference/chat-runs нет намеренно: получив 404, фронтенд сам переходит
        // на обычный SSE-поток (раньше фейковые chat-runs ломали каждое сообщение ошибкой 404)
        // История чатов в SQLite (chat_history.rs)
        .route("/chat", delete(chat_history::clear_all))
        .route("/chat/export", get(chat_history::export_history))
        .route("/chat/count", get(chat_history::count_threads))
        .route("/chat/attachments", get(chat_history::list_attachments))
        .route("/chat/threads", get(chat_history::list_threads).post(chat_history::save_thread).delete(chat_history::delete_threads))
        .route("/chat/threads/:id", get(chat_history::get_thread).put(chat_history::update_thread).patch(chat_history::update_thread).delete(chat_history::delete_thread))
        .route("/chat/threads/:id/forks", get(chat_history::thread_fork_counts))
        .route("/chat/threads/:id/fork", post(chat_history::fork_thread))
        .route("/chat/threads/:id/messages", get(chat_history::list_messages).post(chat_history::add_message).put(chat_history::sync_messages))
        .route("/chat/threads/:id/messages/:msg_id", get(chat_history::get_message).put(chat_history::put_message).patch(chat_history::patch_message).delete(chat_history::delete_message))
        .route("/chat/projects", get(chat_history::list_projects).post(chat_history::save_project))
        .route("/chat/projects/:id", get(chat_history::get_project).patch(chat_history::update_project).delete(chat_history::delete_project))
        .route("/chat/settings", get(chat_history::get_settings).put(chat_history::put_settings).post(chat_history::put_settings))
        .route("/chat/settings/compare-and-set", post(chat_history::compare_and_set_settings))
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
        // Настройки хранятся в SQLite (модуль settings)
        .route("/settings/personalization", get(settings::ui::get_personalization).put(settings::ui::put_personalization))
        .route("/settings/upload-limit", get(settings::ui::get_upload_limit).put(settings::ui::put_upload_limit))
        .route("/settings/vram-budget", get(settings::runtime::get_vram_budget).put(settings::runtime::put_vram_budget))
        .route("/settings/download-transport", get(settings::runtime::get_download_transport).put(settings::runtime::put_download_transport))
        .route("/settings/xet-notice/reserve", post(api::handle_xet_notice_reserve))
        .route("/settings/igpu-carveout-notice/dismiss", post(api::handle_igpu_carveout_notice_dismiss))
        .route("/settings/helper-precache", get(settings::ui::get_helper_precache).put(settings::ui::put_helper_precache))
        .route("/settings/embedding-model", get(settings::runtime::get_embedding_model).put(settings::runtime::embedding_model_not_ready).delete(settings::runtime::get_embedding_model))
        .route("/settings/embedding-model/resolve", get(settings::runtime::embedding_model_not_ready))
        .route("/settings/embedding-model/unload", post(settings::runtime::get_embedding_model))
        .route("/settings/openai-auto-switch", get(settings::runtime::get_auto_switch).put(settings::runtime::put_auto_switch))
        .route("/settings/openai-auto-switch/overrides", get(settings::runtime::get_model_overrides).put(settings::runtime::put_model_overrides))
        .route("/settings/chat-preferences/migrate", post(settings::ui::migrate_chat_preferences))
        .route("/settings/chat-preferences", get(settings::ui::get_chat_preferences).put(settings::ui::put_chat_preferences).post(settings::ui::put_chat_preferences))
        .route("/settings/model-memory", get(settings::runtime::get_model_memory).put(settings::runtime::put_model_memory))
        .route("/settings/last-local-model", get(settings::runtime::get_last_local_model).put(settings::runtime::put_last_local_model))
        .route("/settings/llama-cpp-path", get(settings::runtime::get_llama_cpp_path).put(settings::runtime::put_llama_cpp_path))
        .route("/settings/lan-access", get(api::handle_settings_lan_access))
        .route("/settings/lan-access/start", post(api::handle_settings_lan_access_post))
        .route("/settings/lan-access/stop", post(api::handle_settings_lan_access_post))
        .route("/settings/lan-access/auto-start", put(api::handle_settings_lan_access_action))
        .route("/settings/lan-access/port", put(api::handle_settings_lan_access_action))
        .route("/settings/hugging-face-cache", get(settings::runtime::get_hugging_face_cache).put(settings::runtime::put_hugging_face_cache))
        .route("/settings/keyless-api-access", get(settings::ui::get_keyless_api_access).put(settings::ui::put_keyless_api_access))
        .route("/settings/remote-access", get(api::handle_settings_remote_access))
        .route("/settings/remote-access/start", post(api::handle_settings_remote_access_post))
        .route("/settings/remote-access/stop", post(api::handle_settings_remote_access_post))
        .route("/settings/remote-access/auto-start", put(api::handle_settings_remote_access_action))
        .route("/settings/preview-sharing", get(settings::ui::get_preview_sharing).put(settings::ui::put_preview_sharing))
        .route("/settings/preview-links/rotate", post(api::handle_settings_preview_links_rotate))
        .route("/settings/coding-agents", get(api::handle_settings_coding_agents))
        .route("/settings/current-date-prompt", get(settings::ui::get_current_date_prompt).put(settings::ui::put_current_date_prompt))
        .route("/settings/debug/logs/sources", get(api::handle_settings_debug_logs_sources))
        .route("/settings/debug/logs", get(api::handle_settings_debug_logs))
        .route("/settings/hugging-face-token", get(settings::runtime::get_hf_token).put(settings::runtime::put_hf_token).delete(settings::runtime::delete_hf_token))
        .route("/settings/hugging-face-token/migrate", put(settings::runtime::migrate_hf_token))
        .route("/settings/generation-presets/:kind", get(settings::ui::get_generation_presets).put(settings::ui::put_generation_presets))
        .route("/settings/generation-presets/:kind/custom", get(settings::ui::get_generation_presets).put(settings::ui::put_custom_preset).delete(settings::ui::delete_custom_preset))
        .route("/providers/registry", get(api::handle_providers_registry))
        .route("/providers/public-key", get(api::handle_providers_public_key))
        .route("/providers/models", get(api::handle_providers_models_get).post(api::handle_providers_models_post))
        .route("/providers/test", post(api::handle_providers_test))
        .route("/providers/:id", put(api::handle_providers_detail_put).delete(api::handle_providers_detail_delete))
        .route("/providers", get(api::handle_providers_list).post(api::handle_providers_add))
        .route("/providers/", get(api::handle_providers_list).post(api::handle_providers_add))
        // Export & Llama & RAG
        .route("/export/status", get(api::handle_export_status))
        .route("/export/logs", get(api::handle_export_logs))
        .route("/export/load-checkpoint", post(api::handle_export_load_checkpoint))
        .route("/export/export/base", post(api::handle_export_action))
        .route("/export/export/gguf", post(api::handle_export_action))
        .route("/export/export/lora", post(api::handle_export_action))
        .route("/export/export/merged", post(api::handle_export_action))
        .route("/export/cancel", post(api::handle_export_action))
        .route("/export/cleanup", post(api::handle_export_action))
        .route("/llama/update-status", get(api::handle_llama_update_status))
        .route("/llama/update-changelog", get(api::handle_llama_update_changelog))
        .route("/llama/update", post(api::handle_llama_update))
        .route("/llama/backend", get(api::handle_llama_backend).post(api::handle_llama_backend))
        // Список баз знаний отвечает 200 с ragAvailable: false, остальные методы — 501 (раньше 405)
        .route(
            "/rag/knowledge-bases",
            get(api::handle_rag_knowledge_bases).fallback(unavailable::pending("RAG (базы знаний)", 4)),
        )
        // Разделы без бэкенда отвечают 501 с объяснением вместо 404/405
        .merge(unavailable::pending_api_routes())
        // Fallback for unmatched /api/* calls so they never receive index.html
        .fallback(api::handle_api_not_found);

    // Base App Router with all endpoints
    let mut app = Router::new()
        .nest("/api", api_router)
        // Root WebSocket & OpenAI Endpoints
        .route("/ws/telemetry", get(handle_ws_telemetry))
        .route("/v1/chat/completions", post(chat::handle_chat_completions))
        .route("/v1/models", get(handle_models))
        // /v1/images, /v1/videos, /v1/audio и /openapi.json: JSON с объяснением вместо HTML
        .merge(unavailable::pending_root_routes())
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

    // Проверка Host — самый внешний слой: запрос с чужим именем хоста отклоняется
    // раньше, чем дойдёт до API или статики
    app.layer(axum::middleware::from_fn(security::guard_local_requests))
}

pub async fn run_server_with_listener(
    listener: tokio::net::TcpListener,
    addr: SocketAddr,
    static_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    // Причину отказа Vulkan логируем, а не теряем: без неё не понять, почему нет GPU
    let vk_ctx = match VulkanContext::init(true) {
        Ok(ctx) => {
            info!("Vulkan-устройство: {}", ctx.device_name());
            Some(ctx)
        }
        Err(err) => {
            warn!("Vulkan недоступен ({err}), сервер работает без GPU");
            None
        }
    };

    let models_dir = locations::find_models_dir();
    info!("Папка моделей: {}", models_dir.display());

    // История чатов и настройки хранятся в файле базы и переживают перезапуск
    let database_path = locations::database_path();
    let store = store::Store::open(&database_path).map_err(|err| {
        anyhow::anyhow!(
            "Не удалось открыть базу данных {}: {err}",
            database_path.display()
        )
    })?;
    info!("База данных: {}", database_path.display());
    let state = Arc::new(AppState::with_store(
        vk_ctx,
        models_dir,
        static_dir.clone(),
        Arc::new(store),
    ));
    state
        .server_port
        .store(addr.port(), std::sync::atomic::Ordering::Relaxed);

    let app = create_router(Arc::clone(&state), static_dir);

    info!("SlothForge Axum server listening on http://{}", addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(state))
        .await?;
    info!("Сервер остановлен");
    Ok(())
}

/// Сигнал мягкой остановки: Ctrl+C в терминале или кнопка «Остановить» в интерфейсе.
/// Текущие запросы успевают завершиться.
async fn shutdown_signal(state: Arc<AppState>) {
    let ctrl_c = async {
        if let Err(err) = tokio::signal::ctrl_c().await {
            // Без обработчика Ctrl+C остаётся остановка из интерфейса, поэтому просто ждём её
            warn!("Не удалось подписаться на Ctrl+C: {err}");
            std::future::pending::<()>().await;
        }
    };
    tokio::select! {
        _ = ctrl_c => info!("Получен Ctrl+C, останавливаю сервер"),
        _ = state.shutdown.notified() => info!("Останавливаю сервер по запросу из интерфейса"),
    }
}

pub async fn run_server(addr: SocketAddr, static_dir: Option<PathBuf>) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    run_server_with_listener(listener, addr, static_dir).await
}
