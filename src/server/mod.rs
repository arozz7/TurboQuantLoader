//! HTTP server — OpenAI and Anthropic compatible API on a single axum router.
//!
//! Both endpoint families are always active on the same port:
//!
//! | Method | Path                    | Protocol  |
//! |--------|-------------------------|-----------|
//! | GET    | `/v1/models`            | both      |
//! | POST   | `/v1/chat/completions`  | OpenAI    |
//! | POST   | `/v1/messages`          | Anthropic |

mod error;
mod routes;
mod sse;
pub mod stream_parser;
pub mod types;

use std::sync::Arc;

use anyhow::{Context, Result};
use axum::routing::{get, post};
use axum::Router;
use tracing::info;

use crate::config::AppConfig;
use crate::inference::engine::InferenceEngine;

/// Shared state injected into every request handler via axum `State`.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<InferenceEngine>,
}

/// Build the axum router with all API routes mounted.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/chat/completions", post(routes::openai::chat_completions))
        .route("/v1/messages", post(routes::anthropic::create_message))
        .with_state(state)
}

/// Load the model, build the router, and serve until the process is interrupted.
///
/// Model loading is blocking and runs inside `spawn_blocking` to avoid
/// stalling the async executor.
pub async fn serve(config: AppConfig) -> Result<()> {
    let host = config.server.host.clone();
    let port = config.server.port;

    info!(host = %host, port, "starting inference server");

    let engine = tokio::task::spawn_blocking(move || InferenceEngine::new(config))
        .await
        .context("inference thread panicked")??;

    let state = AppState { engine: Arc::new(engine) };
    let router = build_router(state);

    let bind_addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("failed to bind to {bind_addr}"))?;

    info!(addr = %listener.local_addr().unwrap(), "server listening");
    axum::serve(listener, router).await.context("server error")?;

    Ok(())
}
