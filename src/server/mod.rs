//! HTTP server — OpenAI and Anthropic compatible API on a single axum router.
//!
//! Both endpoint families are always active on the same port:
//!
//! | Method | Path                     | Protocol  |
//! |--------|--------------------------|-----------|
//! | GET    | `/v1/models`             | both      |
//! | POST   | `/v1/chat/completions`   | OpenAI    |
//! | POST   | `/v1/messages`           | Anthropic |
//! | GET    | `/health`                | both      |
//! | GET    | `/metrics`               | Prometheus|
//! | GET    | `/v1/admin/stats`        | both      |
//! | GET    | `/v1/admin/status`       | both      |
//! | POST   | `/v1/admin/restart`      | both      |
//! | POST   | `/v1/admin/load`         | both      |

mod error;
pub mod llama_process;
pub mod proxy;
mod routes;
mod sse;
pub mod stream_parser;
pub mod types;

use std::sync::Arc;

use anyhow::{Context, Result};
use axum::routing::{get, post};
use axum::Router;
use tokio::sync::RwLock;
use tracing::info;

use crate::config::AppConfig;
use crate::metrics::MetricsCollector;
use llama_process::LlamaProcess;

/// Shared state injected into every request handler via axum `State`.
///
/// `process` and `config` are behind `RwLock` so the admin API can hot-swap
/// the backend and update the model path without restarting the server.
#[derive(Clone)]
pub struct AppState {
    /// The running llama-server subprocess. Swapped on `POST /v1/admin/load`.
    pub process: Arc<RwLock<Arc<LlamaProcess>>>,
    /// Active configuration. Written by admin/load to change model path.
    pub config: Arc<RwLock<Arc<AppConfig>>>,
    /// Rolling request metrics + GPU telemetry.
    pub metrics: Arc<MetricsCollector>,
}

impl AppState {
    /// Clone the current `Arc<LlamaProcess>` without holding the lock longer
    /// than necessary. Use for short-lived read access in request handlers.
    pub async fn process_snapshot(&self) -> Arc<LlamaProcess> {
        self.process.read().await.clone()
    }

    /// Clone the current `Arc<AppConfig>`.
    pub async fn config_snapshot(&self) -> Arc<AppConfig> {
        self.config.read().await.clone()
    }
}

/// Build the axum router with all API routes mounted.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // ── Inference ─────────────────────────────────────────────────────────
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/chat/completions", post(routes::openai::chat_completions))
        .route("/v1/messages", post(routes::anthropic::create_message))
        // ── Observability ─────────────────────────────────────────────────────
        .route("/health", get(routes::metrics::health))
        .route("/metrics", get(routes::metrics::prometheus_metrics))
        .route("/v1/admin/stats", get(routes::metrics::admin_stats))
        // ── Admin ─────────────────────────────────────────────────────────────
        .route("/v1/admin/status", get(routes::admin::status))
        .route("/v1/admin/restart", post(routes::admin::restart))
        .route("/v1/admin/load", post(routes::admin::load))
        .with_state(state)
}

/// Start the llama-server subprocess, build the router, and serve until interrupted.
pub async fn serve(config: AppConfig) -> Result<()> {
    let host = config.server.host.clone();
    let port = config.server.port;

    info!(host = %host, port, "starting TurboQuantLoader");

    let config = Arc::new(config);
    let process = LlamaProcess::start(Arc::clone(&config))
        .await
        .context("failed to start llama-server subprocess")?;

    info!(internal_port = config.backend.internal_port, "llama-server ready; starting API server");

    let metrics = MetricsCollector::start();

    let state = AppState {
        process: Arc::new(RwLock::new(Arc::new(process))),
        config: Arc::new(RwLock::new(config)),
        metrics,
    };

    let router = build_router(state);

    let bind_addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("failed to bind to {bind_addr}"))?;

    info!(addr = %listener.local_addr().unwrap(), "API server listening");

    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            info!("shutdown signal received");
        })
        .await
        .context("server error")?;

    Ok(())
}
