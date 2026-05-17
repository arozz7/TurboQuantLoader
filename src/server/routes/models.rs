//! GET /v1/models — list available models.
//!
//! Returns an OpenAI-compatible model list containing:
//! - All entries from the named `[[models]]` registry in config.
//! - Any additional GGUF files discovered under `models_dir` (de-duplicated by name).
//!
//! The currently-loaded model is annotated with `"active": true`.

use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use crate::model::registry::ModelRegistry;
use crate::server::AppState;
use crate::server::error::ApiError;

/// `GET /v1/models`
pub async fn list_models(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let cfg = state.config_snapshot().await;
    let current = state.current_model_name().await;

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut data: Vec<Value> = Vec::new();

    // 1. Named registry entries (highest priority — they have explicit settings).
    for def in &cfg.models {
        seen.insert(def.name.to_lowercase());
        data.push(json!({
            "id": def.name,
            "object": "model",
            "created": 0,
            "owned_by": "turboquant-loader",
            "active": def.name.to_lowercase() == current.to_lowercase(),
        }));
    }

    // 2. Filesystem scan — add any model not already in the registry.
    if let Ok(entries) = ModelRegistry::scan(&cfg.model.models_dir) {
        for entry in entries {
            if seen.insert(entry.name.to_lowercase()) {
                data.push(json!({
                    "id": entry.name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "turboquant-loader",
                    "active": entry.name.to_lowercase() == current.to_lowercase(),
                }));
            }
        }
    }

    Ok(Json(json!({
        "object": "list",
        "data": data,
    })))
}
