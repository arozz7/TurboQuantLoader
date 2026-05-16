//! GET /v1/models — return the currently loaded model.
//!
//! Returns an OpenAI-compatible model list. Both OpenAI and Anthropic SDK
//! clients use this endpoint to discover available models.

use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use crate::server::AppState;
use crate::server::error::ApiError;

/// `GET /v1/models`
pub async fn list_models(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let cfg = state.config_snapshot().await;
    let name = cfg
        .model
        .model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("local-model")
        .to_string();

    Ok(Json(json!({
        "object": "list",
        "data": [{
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "turboquant-loader",
        }]
    })))
}
