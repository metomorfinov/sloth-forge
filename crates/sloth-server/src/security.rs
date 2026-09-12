//! Сетевая защита сервера.
//!
//! 1. **CORS** — правило браузера, каким сайтам разрешено читать ответы нашего API.
//!    Разрешаем только страницы, открытые с этого же компьютера (localhost).
//! 2. **Проверка заголовка Host** — защита от «DNS rebinding»: чужой сайт может
//!    привязать своё доменное имя к адресу 127.0.0.1 и так обойти CORS. Такие запросы
//!    приходят с чужим именем хоста, и мы их отклоняем до маршрутизации.
//! 3. **Проверка Origin у изменяющих запросов** — защита от CSRF: CORS мешает чужому
//!    сайту прочитать ответ, но не отправить POST или DELETE. Браузер всегда добавляет
//!    к таким межсайтовым запросам заголовок Origin, по нему они и отклоняются.

use crate::error::ApiError;
use axum::extract::Request;
use axum::http::{header, HeaderValue, Method};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use std::net::IpAddr;
use std::sync::OnceLock;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

/// Дополнительные разрешённые имена хоста через запятую. Нужны, если сервер
/// открыт в локальную сеть (`HOST=0.0.0.0`) и к нему заходят по IP-адресу компьютера.
pub const ALLOWED_HOSTS_ENV: &str = "SLOTH_ALLOWED_HOSTS";

fn extra_allowed_hosts() -> &'static [String] {
    static HOSTS: OnceLock<Vec<String>> = OnceLock::new();
    HOSTS.get_or_init(|| {
        std::env::var(ALLOWED_HOSTS_ENV)
            .map(|raw| {
                raw.split(',')
                    .map(|host| host.trim().to_ascii_lowercase())
                    .filter(|host| !host.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// Отрезает порт от `host:port`, учитывая IPv6 в квадратных скобках (`[::1]:3000` → `::1`).
pub fn strip_port(authority: &str) -> &str {
    if let Some(rest) = authority.strip_prefix('[') {
        return rest.split(']').next().unwrap_or(rest);
    }
    match authority.rsplit_once(':') {
        Some((host, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => host,
        _ => authority,
    }
}

/// Имя хоста указывает на этот же компьютер.
pub fn is_loopback_host(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    host == "localhost"
        || host.ends_with(".localhost")
        || host
            .parse::<IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
}

fn is_allowed_host(host: &str) -> bool {
    is_loopback_host(host)
        || extra_allowed_hosts()
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(host))
}

/// Проверяет значение заголовка `Origin` вида `http://localhost:3000`.
pub fn is_allowed_origin(origin: &str) -> bool {
    let Some((scheme, authority)) = origin.split_once("://") else {
        return false;
    };
    (scheme == "http" || scheme == "https") && is_allowed_host(strip_port(authority))
}

/// CORS-слой: ответы API читаются только страницами с разрешённых хостов.
pub fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin: &HeaderValue, _parts| {
            origin.to_str().map(is_allowed_origin).unwrap_or(false)
        }))
        .allow_methods(Any)
        .allow_headers(Any)
}

/// Запрос что-то меняет (не GET/HEAD/OPTIONS) и пришёл со страницы чужого сайта.
fn is_foreign_state_change(request: &Request) -> bool {
    if matches!(
        *request.method(),
        Method::GET | Method::HEAD | Method::OPTIONS
    ) {
        return false;
    }
    request
        .headers()
        .get(header::ORIGIN)
        .is_some_and(|origin| !origin.to_str().map(is_allowed_origin).unwrap_or(false))
}

/// Middleware: отклоняет запросы с чужим `Host` и изменяющие запросы с чужим `Origin`.
pub async fn guard_local_requests(request: Request, next: Next) -> Response {
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
        .or_else(|| {
            request
                .uri()
                .authority()
                .map(|authority| authority.as_str().to_owned())
        });

    match host {
        Some(host) if is_allowed_host(strip_port(&host)) => {
            if is_foreign_state_change(&request) {
                tracing::warn!(
                    "Отклонён изменяющий запрос с чужого сайта: {} {}",
                    request.method(),
                    request.uri().path()
                );
                return ApiError::forbidden("Изменяющий запрос со страницы чужого сайта отклонён")
                    .into_response();
            }
            next.run(request).await
        }
        Some(host) => {
            tracing::warn!("Отклонён запрос с недопустимым Host: {host}");
            ApiError::forbidden(format!(
                "Недопустимый заголовок Host «{host}». Если это адрес вашего компьютера в сети, добавьте его в {ALLOWED_HOSTS_ENV}"
            ))
            .into_response()
        }
        None => ApiError::bad_request("В запросе нет заголовка Host").into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_ports() {
        assert_eq!(strip_port("localhost:3000"), "localhost");
        assert_eq!(strip_port("127.0.0.1"), "127.0.0.1");
        assert_eq!(strip_port("[::1]:3000"), "::1");
        assert_eq!(strip_port("example.com:"), "example.com:");
    }

    #[test]
    fn detects_loopback_hosts() {
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("LOCALHOST."));
        assert!(is_loopback_host("app.localhost"));
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("127.8.9.10"));
        assert!(is_loopback_host("::1"));
        assert!(!is_loopback_host("0.0.0.0"));
        assert!(!is_loopback_host("192.168.1.10"));
        assert!(!is_loopback_host("localhost.evil.example"));
    }

    #[test]
    fn checks_origins() {
        assert!(is_allowed_origin("http://localhost:3000"));
        assert!(is_allowed_origin("http://127.0.0.1:5173"));
        assert!(is_allowed_origin("http://[::1]:3000"));
        assert!(!is_allowed_origin("http://evil.example"));
        assert!(!is_allowed_origin("null"));
        assert!(!is_allowed_origin("file://localhost"));
    }
}
