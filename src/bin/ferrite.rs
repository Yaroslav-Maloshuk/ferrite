use std::net::SocketAddr;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use ferrite::config::FerriteConfig;
use ferrite::embedding::download_model;
use ferrite::http::router;
use ferrite::pipeline::{Ferrite, IngestItem};

#[derive(Parser)]
#[command(
    name = "ferrite",
    version,
    about = "Rust embedding + retrieval service"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fetch the embedding model into FERRITE_MODEL_DIR (idempotent).
    Prefetch,
    /// Run the HTTP service.
    Serve {
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Ingest a JSONL file: one {"id":..,"text":..} per line.
    Ingest {
        #[arg(long)]
        file: String,
        #[arg(long, default_value_t = 0)]
        limit: usize,
    },
    /// Search the store.
    Search {
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
    },
    /// Print store stats.
    Stats,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let cli = Cli::parse();
    let mut config = FerriteConfig::from_env();

    match cli.command {
        Command::Prefetch => {
            download_model(&config)?;
            println!("model ready at {}", config.model_dir.display());
        }
        Command::Serve { port } => {
            config.port = port;
            if config.tls_cert.is_some() != config.tls_key.is_some() {
                anyhow::bail!("FERRITE_TLS_CERT and FERRITE_TLS_KEY must be set together");
            }
            if config.api_key.is_some() {
                tracing::info!("API-key auth enabled (all routes except /v1/health)");
            }
            let ferrite = Arc::new(Ferrite::init(&config).await?);
            let app = router(ferrite, config.api_key.clone())
                .into_make_service_with_connect_info::<SocketAddr>();
            let addr = SocketAddr::from(([0, 0, 0, 0], port));
            match (&config.tls_cert, &config.tls_key) {
                (Some(cert), Some(key)) => {
                    tracing::info!("ferrite listening on :{port} (TLS)");
                    rustls::crypto::ring::default_provider()
                        .install_default()
                        .ok();
                    let tls = axum_server::tls_rustls::RustlsConfig::from_pem_file(cert, key)
                        .await
                        .map_err(anyhow::Error::new)?;
                    axum_server::bind_rustls(addr, tls).serve(app).await?;
                }
                (None, None) => {
                    tracing::info!("ferrite listening on :{port}");
                    axum_server::bind(addr).serve(app).await?;
                }
                _ => unreachable!("tls pair checked above"),
            }
        }
        Command::Ingest { file, limit } => {
            let ferrite = Ferrite::init(&config).await?;
            let mut items = Vec::new();
            for (idx, line) in std::fs::read_to_string(&file)?.lines().enumerate() {
                if limit > 0 && items.len() >= limit {
                    break;
                }
                let v: serde_json::Value = serde_json::from_str(line)?;
                items.push(IngestItem {
                    id: v["id"]
                        .as_str()
                        .unwrap_or(&*format!("row-{idx}"))
                        .to_string(),
                    text: v["text"].as_str().unwrap_or_default().to_string(),
                    metadata: v["metadata"].as_str().map(|s| s.to_string()),
                });
            }
            let n = ferrite.ingest(&items).await?;
            println!("ingested {n} items");
        }
        Command::Search { query, top_k } => {
            let ferrite = Ferrite::init(&config).await?;
            let hits = ferrite.search(&query, top_k).await?;
            for h in &hits {
                println!("{:.4}\t{}\t{}", h.score, h.id, h.text);
            }
        }
        Command::Stats => {
            let ferrite = Ferrite::init(&config).await?;
            println!("{:#?}", ferrite.stats().await?);
        }
    }
    Ok(())
}
