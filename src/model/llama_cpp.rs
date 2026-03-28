//! llama.cpp inference backend.
//!
//! Only compiled when the `llama-backend` feature (or any GPU feature) is active.

#![cfg(feature = "llama-backend")]

use std::num::NonZeroU32;
use std::sync::mpsc;
use std::thread;

use anyhow::{Context, Result};
use tracing::{debug, info, warn};

use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
#[allow(deprecated)]
use llama_cpp_2::model::{AddBos, LlamaModel, Special};

use crate::config::{KvBits, KvCacheConfig, ModelConfig};
use crate::inference::sampler::build_sampler;
use crate::model::backend::{
    GenerateEvent, GenerateRequest, GenerateSummary, GenerateStream, ModelBackend,
};

/// Buffer capacity for the per-request token event channel.
const EVENT_CHANNEL_CAPACITY: usize = 64;

/// Messages sent from async callers to the inference thread.
enum BackendCommand {
    Generate {
        req: GenerateRequest,
        event_tx: tokio::sync::mpsc::Sender<GenerateEvent>,
    },
}

/// llama.cpp inference backend.
///
/// All `LlamaModel` and `LlamaContext` state lives on a dedicated OS thread to
/// work around `LlamaContext`'s `!Send` constraint. Callers communicate with
/// the thread via a synchronous command channel.
pub struct LlamaCppBackend {
    model_name: String,
    context_size: u32,
    cmd_tx: mpsc::SyncSender<BackendCommand>,
}

// SAFETY: `LlamaCppBackend` only holds `String`, `u32`, and an `mpsc::SyncSender`
// — all of which are Send/Sync. The non-Send llama.cpp context stays on its
// dedicated thread and is never exposed to callers.
unsafe impl Send for LlamaCppBackend {}
unsafe impl Sync for LlamaCppBackend {}

impl ModelBackend for LlamaCppBackend {
    fn load(config: &ModelConfig) -> Result<Self> {
        let config = config.clone();

        // Startup handshake: the inference thread signals ready (or error) once.
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(String, u32)>>(0);
        let (cmd_tx, cmd_rx) = mpsc::sync_channel::<BackendCommand>(4);

        thread::Builder::new()
            .name("llama-inference".into())
            .spawn(move || {
                if let Err(e) = inference_thread_main(&config, ready_tx.clone(), cmd_rx) {
                    let _ = ready_tx.send(Err(e));
                }
            })
            .context("failed to spawn inference thread")?;

        let (model_name, context_size) = ready_rx
            .recv()
            .context("inference thread exited before signalling ready")??;

        info!(model = %model_name, context_size, "model loaded successfully");
        Ok(Self { model_name, context_size, cmd_tx })
    }

    /// Approximate tokenization for context-size estimation only.
    ///
    /// Real tokenization happens inside the inference thread; this stub avoids
    /// adding a round-trip command for Phase 2.
    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        // Rough estimate: ~1.3 tokens per whitespace word.
        Ok((0..((text.split_whitespace().count() as f32 * 1.3) as usize))
            .map(|i| i as i32)
            .collect())
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        Ok(format!("[{} tokens]", tokens.len()))
    }

    fn context_size(&self) -> u32 {
        self.context_size
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn generate(&self, req: GenerateRequest) -> Result<GenerateStream> {
        let (event_tx, event_rx) = tokio::sync::mpsc::channel(EVENT_CHANNEL_CAPACITY);
        self.cmd_tx
            .send(BackendCommand::Generate { req, event_tx })
            .map_err(|_| anyhow::anyhow!("inference thread has stopped"))?;
        Ok(event_rx)
    }

    fn apply_kv_cache_config(&mut self, _cfg: &KvCacheConfig) -> Result<()> {
        // KV cache type is baked into the context at creation time inside the
        // inference thread. Nothing to do here post-load.
        Ok(())
    }
}

// ── Inference Thread ─────────────────────────────────────────────────────────

/// Map our [`KvBits`] config to a llama.cpp [`KvCacheType`].
fn kv_cache_type(bits: &KvBits) -> KvCacheType {
    match bits {
        KvBits::Two | KvBits::Three => {
            warn!("KV bits 2/3 not natively supported by llama.cpp — using Q4_0");
            KvCacheType::Q4_0
        }
        KvBits::Four => KvCacheType::Q4_0,
        KvBits::Eight => KvCacheType::Q8_0,
    }
}

/// Inference thread entry point.
///
/// Loads the model and context, signals readiness on `ready_tx`, then
/// processes [`BackendCommand`]s until the sender side is dropped.
fn inference_thread_main(
    config: &ModelConfig,
    ready_tx: mpsc::SyncSender<Result<(String, u32)>>,
    cmd_rx: mpsc::Receiver<BackendCommand>,
) -> Result<()> {
    let backend = LlamaBackend::init().context("failed to initialise llama backend")?;

    // ── Model params ──────────────────────────────────────────────────────────
    let n_gpu = if config.n_gpu_layers < 0 {
        u32::MAX // llama.cpp treats i32::MAX as "all layers"
    } else {
        config.n_gpu_layers as u32
    };

    let mut model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu);

    if config.tensor_split.len() > 1 {
        // Device indices 0…N map to the physical GPUs detected by ggml.
        // For the dev machine: 0 = RTX 4070 Ti Super, 1 = RTX 2060.
        let devices: Vec<usize> = (0..config.tensor_split.len()).collect();
        model_params = model_params
            .with_devices(&devices)
            .context("failed to configure GPU devices for tensor split")?;
        debug!(devices = ?devices, "multi-GPU tensor split configured");
    }

    // ── Load model ────────────────────────────────────────────────────────────
    info!(path = %config.model_path.display(), "loading GGUF model");
    let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
        .context("failed to load model file")?;

    let model_name = config
        .model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // ── Context params ────────────────────────────────────────────────────────
    let ctx_size = NonZeroU32::new(config.context_size)
        .unwrap_or_else(|| NonZeroU32::new(8192).expect("constant is non-zero"));

    let kv_type = kv_cache_type(&crate::config::KvBits::Four); // default; Phase 3 reads from config

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(ctx_size))
        .with_n_batch(config.batch_size)
        .with_n_threads(config.threads as i32)
        .with_n_threads_batch(config.threads as i32)
        .with_type_k(kv_type)
        .with_type_v(kv_type);

    // ── Create context (persistent for the thread lifetime) ───────────────────
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .context("failed to create llama context")?;

    let actual_ctx_size = ctx.n_ctx();

    // Signal readiness to the caller.
    ready_tx
        .send(Ok((model_name, actual_ctx_size)))
        .map_err(|_| anyhow::anyhow!("caller dropped the ready channel before model loaded"))?;

    // ── Command loop ──────────────────────────────────────────────────────────
    for cmd in &cmd_rx {
        match cmd {
            BackendCommand::Generate { req, event_tx } => {
                ctx.clear_kv_cache();
                if let Err(e) = do_generate(&model, &mut ctx, &req, &event_tx) {
                    let _ = event_tx.try_send(GenerateEvent::Error(e.to_string()));
                }
            }
        }
    }

    Ok(())
}

// ── Generation ───────────────────────────────────────────────────────────────

/// Run one generation request, streaming tokens through `event_tx`.
fn do_generate(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
    req: &GenerateRequest,
    event_tx: &tokio::sync::mpsc::Sender<GenerateEvent>,
) -> Result<()> {
    let start = std::time::Instant::now();

    // Tokenize the full prompt.
    let prompt_tokens = model
        .str_to_token(&req.prompt, AddBos::Always)
        .context("failed to tokenize prompt")?;
    let prompt_len = prompt_tokens.len();
    debug!(tokens = prompt_len, "prompt tokenised");

    // Prefill: decode the entire prompt in one batch.
    let mut batch = LlamaBatch::new(prompt_len.max(1), 1);
    batch
        .add_sequence(&prompt_tokens, 0, false)
        .context("failed to add prompt sequence to batch")?;
    ctx.decode(&mut batch).context("prompt prefill decode failed")?;

    // Build the sampler and warm it up with the prompt tokens so the repetition
    // penalty window covers what was already in the prompt.
    let mut sampler = build_sampler(&req.sampler);
    sampler.accept_many(prompt_tokens.iter().copied());

    let mut tokens_generated: u32 = 0;
    let mut pos = prompt_len as i32;

    loop {
        if tokens_generated >= req.max_tokens {
            break;
        }

        let token = sampler.sample(ctx, -1);

        if model.is_eog_token(token) {
            break;
        }

        // Decode the token to a UTF-8 string fragment.
        #[allow(deprecated)]
        let text = model
            .token_to_str(token, Special::Tokenize)
            .context("failed to decode token to string")?;

        // Stop-string check.
        if req.stop_strings.iter().any(|s| text.contains(s.as_str())) {
            break;
        }

        // Forward the token to the caller; abort if the receiver is gone.
        if event_tx.try_send(GenerateEvent::Token(text)).is_err() {
            break;
        }

        sampler.accept(token);
        tokens_generated += 1;
        pos += 1;

        // Decode the new token to update the KV cache.
        batch.clear();
        batch
            .add(token, pos, &[0], true)
            .context("failed to add generated token to batch")?;
        ctx.decode(&mut batch).context("per-token decode failed")?;
    }

    let elapsed = start.elapsed().as_secs_f32();
    let tps = if elapsed > 0.0 {
        tokens_generated as f32 / elapsed
    } else {
        0.0
    };

    let _ = event_tx.try_send(GenerateEvent::Done(GenerateSummary {
        tokens_generated,
        tokens_per_second: tps,
        context_tokens: prompt_len as u32 + tokens_generated,
    }));

    Ok(())
}
