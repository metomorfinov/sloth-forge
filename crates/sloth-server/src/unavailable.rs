//! Разделы Unsloth Studio, у которых в SlothForge пока нет бэкенда.
//!
//! Фронтенд у нас оригинальный, и при открытии таких разделов он обращается к API,
//! которого ещё нет. Раньше это давало голые 404/405 или HTML-страницу вместо JSON,
//! и пользователь видел только «Request failed». Теперь такие запросы получают
//! 501 с объяснением, на каком этапе дорожной карты появится раздел.

use crate::error::ApiError;
use axum::routing::any;
use axum::Router;
use std::future::{ready, Ready};

/// Раздел, который появится позже: название для пользователя и этап дорожной карты.
#[derive(Debug, Clone, Copy)]
pub struct PendingFeature {
    pub name: &'static str,
    pub stage: u8,
}

/// Ошибка 501 «раздел ещё не реализован» с понятным текстом.
pub fn not_ready(name: &str, stage: u8) -> ApiError {
    ApiError::not_implemented(format!(
        "«{name}» появится в SlothForge на этапе {stage} дорожной карты"
    ))
}

/// Обработчик, который на любой запрос отвечает 501 для указанного раздела.
pub fn pending(
    name: &'static str,
    stage: u8,
) -> impl FnOnce() -> Ready<ApiError> + Clone + Send + Sync + 'static {
    move || ready(not_ready(name, stage))
}

const IMAGES: PendingFeature = PendingFeature {
    name: "Генерация изображений",
    stage: 6,
};
const DIFFUSION_TRAINING: PendingFeature = PendingFeature {
    name: "Обучение diffusion-моделей",
    stage: 6,
};
const VIDEO: PendingFeature = PendingFeature {
    name: "Генерация видео",
    stage: 7,
};
const AUDIO: PendingFeature = PendingFeature {
    name: "Аудио (распознавание и синтез речи)",
    stage: 5,
};

/// Разделы внутри `/api`: и сам префикс, и всё под ним отвечают 501.
const API_SECTIONS: &[(&str, PendingFeature)] = &[
    ("/inference/images", IMAGES),
    ("/train/diffusion", DIFFUSION_TRAINING),
    ("/inference/video", VIDEO),
    ("/inference/audio", AUDIO),
    (
        "/rag",
        PendingFeature {
            name: "RAG (базы знаний)",
            stage: 4,
        },
    ),
    (
        "/data-recipe",
        PendingFeature {
            name: "Recipe Studio",
            stage: 4,
        },
    ),
    (
        "/chat/research-runs",
        PendingFeature {
            name: "Deep Research",
            stage: 4,
        },
    ),
    (
        "/prompts",
        PendingFeature {
            name: "Библиотека промптов",
            stage: 4,
        },
    ),
    (
        "/mcp",
        PendingFeature {
            name: "MCP-серверы",
            stage: 4,
        },
    ),
    (
        "/youtube",
        PendingFeature {
            name: "Транскрипты YouTube",
            stage: 4,
        },
    ),
    (
        "/inference/external",
        PendingFeature {
            name: "Внешние контейнеры OpenAI",
            stage: 4,
        },
    ),
    (
        "/inference/search-images",
        PendingFeature {
            name: "Поиск изображений",
            stage: 4,
        },
    ),
    (
        "/inference/sandbox",
        PendingFeature {
            name: "Песочница кода",
            stage: 4,
        },
    ),
];

/// Отдельные маршруты внутри `/api` (у их родителей уже есть рабочие эндпоинты).
const API_ENDPOINTS: &[(&str, PendingFeature)] = &[
    (
        "/inference/tool-confirm",
        PendingFeature {
            name: "Вызов инструментов в чате",
            stage: 4,
        },
    ),
    (
        "/inference/artifact-preview-frame",
        PendingFeature {
            name: "Предпросмотр артефактов",
            stage: 4,
        },
    ),
    (
        "/chat/attachments/*rest",
        PendingFeature {
            name: "Вложения в чате",
            stage: 4,
        },
    ),
    (
        "/providers/:id/*rest",
        PendingFeature {
            name: "OAuth и модели внешних провайдеров",
            stage: 4,
        },
    ),
    (
        "/export/logs/stream",
        PendingFeature {
            name: "Экспорт моделей",
            stage: 3,
        },
    ),
];

/// Эндпоинты оригинального Unsloth для Python transformers: SlothForge они не нужны.
const NOT_APPLICABLE_ENDPOINTS: &[&str] = &[
    "/models/remote-code-scan",
    "/models/discard-remote-code",
    "/inference/transformers-upgrade-check",
    "/inference/install-latest-transformers",
];

/// Разделы в корне сервера (OpenAI-совместимые пути), где раньше отдавалась HTML-страница.
const ROOT_SECTIONS: &[(&str, PendingFeature)] = &[
    ("/v1/images", IMAGES),
    ("/v1/videos", VIDEO),
    ("/v1/audio", AUDIO),
];

fn not_applicable() -> Ready<ApiError> {
    ready(ApiError::not_implemented(
        "Не требуется: SlothForge работает с GGUF на собственном движке и не использует Python transformers",
    ))
}

/// Добавляет в роутер префикс раздела и всё, что под ним.
fn add_section<S: Clone + Send + Sync + 'static>(
    router: Router<S>,
    prefix: &str,
    feature: PendingFeature,
) -> Router<S> {
    let handler = pending(feature.name, feature.stage);
    router
        .route(prefix, any(handler.clone()))
        .route(&format!("{prefix}/*rest"), any(handler))
}

/// Маршруты-заглушки внутри `/api` (подключаются через `Router::merge`).
pub fn pending_api_routes<S: Clone + Send + Sync + 'static>() -> Router<S> {
    let mut router = API_SECTIONS
        .iter()
        .fold(Router::new(), |router, (prefix, feature)| {
            add_section(router, prefix, *feature)
        });
    for (path, feature) in API_ENDPOINTS {
        router = router.route(path, any(pending(feature.name, feature.stage)));
    }
    for path in NOT_APPLICABLE_ENDPOINTS {
        router = router.route(path, any(not_applicable));
    }
    router
}

/// Маршруты-заглушки в корне: вместо HTML-страницы приходит JSON с объяснением.
pub fn pending_root_routes<S: Clone + Send + Sync + 'static>() -> Router<S> {
    ROOT_SECTIONS
        .iter()
        .fold(Router::new(), |router, (prefix, feature)| {
            add_section(router, prefix, *feature)
        })
        .route(
            "/openapi.json",
            any(|| {
                ready(ApiError::not_found(
                    "OpenAPI-схема SlothForge пока не публикуется",
                ))
            }),
        )
}
