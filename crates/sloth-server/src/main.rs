use std::net::SocketAddr;
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sloth_server=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let explicit_port: Option<u16> = std::env::var("PORT")
        .or_else(|_| std::env::var("SLOTH_PORT"))
        .ok()
        .and_then(|p| p.parse().ok());

    let (listener, bound_addr) = match explicit_port {
        Some(p) => {
            let addr: SocketAddr = format!("{}:{}", host, p).parse()?;
            (tokio::net::TcpListener::bind(addr).await?, addr)
        }
        None => {
            let candidate_ports = [8000, 3000, 8088, 8081];
            let mut result = None;
            for p in candidate_ports {
                let addr: SocketAddr = format!("{}:{}", host, p).parse()?;
                if let Ok(l) = tokio::net::TcpListener::bind(addr).await {
                    result = Some((l, addr));
                    break;
                }
            }
            result.ok_or_else(|| anyhow::anyhow!("Unable to bind to any candidate port (8000, 3000, 8088, 8081)"))?
        }
    };

    // Find static frontend dist dir
    let static_dir = if let Ok(custom) = std::env::var("STATIC_DIR") {
        Some(PathBuf::from(custom))
    } else {
        let candidates = [
            PathBuf::from("/home/rivergod/.gemini/antigravity/scratch/sloth-forge/frontend/dist"),
            PathBuf::from("frontend/dist"),
            PathBuf::from("../frontend/dist"),
            PathBuf::from("../../frontend/dist"),
        ];
        candidates.into_iter().find(|p| p.exists())
    };

    println!("Starting SlothForge Server v{}...", sloth_server::server_version());
    println!("Listening on http://{}", bound_addr);
    if let Some(ref dir) = static_dir {
        println!("Serving static frontend assets from: {}", dir.display());
    }

    sloth_server::run_server_with_listener(listener, bound_addr, static_dir).await
}
