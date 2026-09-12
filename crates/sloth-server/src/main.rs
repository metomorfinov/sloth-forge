use std::net::SocketAddr;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Адрес по умолчанию: только этот компьютер.
const DEFAULT_HOST: &str = "127.0.0.1";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sloth_server=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // По умолчанию сервер доступен только с этого компьютера. Открыть его в локальную
    // сеть можно явно: SLOTH_HOST=0.0.0.0 (и добавить свой адрес в SLOTH_ALLOWED_HOSTS).
    // SLOTH_HOST проверяется первым, потому что HOST некоторые оболочки заполняют именем машины.
    let host = std::env::var("SLOTH_HOST")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| DEFAULT_HOST.to_string());

    // Порт: PORT или SLOTH_PORT, иначе 3000. Перебора портов нет: если порт занят,
    // лучше сразу сказать об этом, чем молча уехать на другой адрес.
    let port = match std::env::var("PORT").or_else(|_| std::env::var("SLOTH_PORT")) {
        Ok(raw) => raw
            .trim()
            .parse::<u16>()
            .map_err(|_| anyhow::anyhow!("Некорректный порт в PORT/SLOTH_PORT: «{raw}»"))?,
        Err(_) => sloth_server::DEFAULT_PORT,
    };

    // IPv6-адрес в записи «адрес:порт» заключается в квадратные скобки
    let authority = if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    let bound_addr: SocketAddr = authority
        .parse()
        .map_err(|err| anyhow::anyhow!("Некорректный адрес {authority}: {err}"))?;
    let listener = tokio::net::TcpListener::bind(bound_addr)
        .await
        .map_err(|err| {
            anyhow::anyhow!(
                "Не удалось занять {bound_addr}: {err}. Возможно, порт занят другой программой — укажите другой через PORT=<номер>"
            )
        })?;

    let static_dir = sloth_server::locations::find_static_dir();

    println!(
        "Starting SlothForge Server v{}...",
        sloth_server::server_version()
    );
    println!("Listening on http://{}", bound_addr);
    match static_dir {
        Some(ref dir) => println!("Serving static frontend assets from: {}", dir.display()),
        None => println!(
            "Собранный фронтенд не найден: выполните `cd frontend && npm run build` или укажите {}",
            sloth_server::locations::STATIC_DIR_ENV
        ),
    }

    sloth_server::run_server_with_listener(listener, bound_addr, static_dir).await
}
