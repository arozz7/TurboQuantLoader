//! llama.cpp inference backend.
//!
//! Only compiled when the `llama-backend` feature (or any GPU feature) is active.

#![cfg(feature = "llama-backend")]

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};
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
use crate::kv_cache::PrefixCache;
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
    /// Tear down the current `LlamaContext` and rebuild with new parameters.
    ///
    /// Model weights remain loaded. Used by the bench command to test multiple
    /// (context_size × kv_bits) combinations without reloading the model.
    ReconfigureContext {
        n_ctx: u32,
        kv_config: KvCacheConfig,
        result_tx: mpsc::SyncSender<Result<u32>>, // returns actual n_ctx on success
    },
}

/// llama.cpp inference backend.
///
/// All `LlamaModel` and `LlamaContext` state lives on a dedicated OS thread to
/// work around `LlamaContext`'s `!Send` constraint. Callers communicate with
/// the thread via a synchronous command channel.
pub struct LlamaCppBackend {
    model_name: String,
    /// Updated atomically when `reconfigure_context` recreates the context.
    context_size: AtomicU32,
    cmd_tx: mpsc::SyncSender<BackendCommand>,
}

// SAFETY: `LlamaCppBackend` only holds `String`, `u32`, and an `mpsc::SyncSender`
// — all of which are Send/Sync. The non-Send llama.cpp context stays on its
// dedicated thread and is never exposed to callers.
unsafe impl Send for LlamaCppBackend {}
unsafe impl Sync for LlamaCppBackend {}

impl LlamaCppBackend {
    /// Load a model with explicit KV cache configuration.
    ///
    /// Called by [`ModelBackend::load`] (with a default KV config) and by
    /// `InferenceEngine` (with the user-configured KV config) to avoid a
    /// redundant context rebuild on startup.
    pub fn load_full(model_config: &ModelConfig, kv_config: &KvCacheConfig) -> Result<Self> {
        let model_config = model_config.clone();
        let kv_config = kv_config.clone();

        // Startup handshake: the inference thread signals ready (or error) once.
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(String, u32)>>(0);
        let (cmd_tx, cmd_rx) = mpsc::sync_channel::<BackendCommand>(4);

        thread::Builder::new()
            .name("llama-inference".into())
            .spawn(move || {
                if let Err(e) = inference_thread_main(&model_config, &kv_config, ready_tx.clone(), cmd_rx) {
                    let _ = ready_tx.send(Err(e));
                }
            })
            .context("failed to spawn inference thread")?;

        let (model_name, context_size) = ready_rx
            .recv()
            .context("inference thread exited before signalling ready")??;

        info!(model = %model_name, context_size, "model loaded successfully");
        Ok(Self { model_name, context_size: AtomicU32::new(context_size), cmd_tx })
    }
}

impl ModelBackend for LlamaCppBackend {
    fn load(config: &ModelConfig) -> Result<Self> {
        Self::load_full(config, &KvCacheConfig::default())
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
        self.context_size.load(Ordering::Relaxed)
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

    fn apply_kv_cache_config(&mut self, cfg: &KvCacheConfig) -> Result<()> {
        self.reconfigure_context(self.context_size.load(Ordering::Relaxed), cfg)
    }

    fn reconfigure_context(&self, n_ctx: u32, kv_cfg: &KvCacheConfig) -> Result<()> {
        let (result_tx, result_rx) = mpsc::sync_channel(0);
        self.cmd_tx
            .send(BackendCommand::ReconfigureContext {
                n_ctx,
                kv_config: kv_cfg.clone(),
                result_tx,
            })
            .map_err(|_| anyhow::anyhow!("inference thread has stopped"))?;
        let new_ctx_size = result_rx
            .recv()
            .context("inference thread dropped result channel")??;
        self.context_size.store(new_ctx_size, Ordering::Relaxed);
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
    kv_config: &KvCacheConfig,
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

    if !config.tensor_split.is_empty() {
        // Device indices 0…N map to the physical GPUs detected by ggml.
        let devices: Vec<usize> = (0..config.tensor_split.len()).collect();
        model_params = model_params
            .with_devices(&devices)
            .context("failed to configure GPU devices for tensor split")?;

        debug!(devices = ?devices, "GPU allocation configured");
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

    // ── Create context (persistent for the thread lifetime) ───────────────────
    let mut ctx = rebuild_context(&backend, &model, config, config.context_size, kv_config)
        .context("failed to create initial llama context")?;

    let actual_ctx_size = ctx.n_ctx();

    // Signal readiness to the caller.
    ready_tx
        .send(Ok((model_name, actual_ctx_size)))
        .map_err(|_| anyhow::anyhow!("caller dropped the ready channel before model loaded"))?;

    // ── Command loop ──────────────────────────────────────────────────────────
    let mut prefix_cache = PrefixCache::new();

    for cmd in &cmd_rx {
        match cmd {
            BackendCommand::Generate { req, event_tx } => {
                if let Err(e) = do_generate(&model, &mut ctx, &req, &event_tx, &mut prefix_cache) {
                    prefix_cache.invalidate();
                    let _ = event_tx.blocking_send(GenerateEvent::Error(e.to_string()));
                }
            }
            BackendCommand::ReconfigureContext { n_ctx, kv_config, result_tx } => {
                let result = rebuild_context(&backend, &model, &config, n_ctx, &kv_config);
                match result {
                    Ok(new_ctx) => {
                        ctx = new_ctx;
                        // Context was rebuilt — KV cache is empty.
                        prefix_cache.invalidate();
                        let _ = result_tx.send(Ok(ctx.n_ctx()));
                    }
                    Err(e) => {
                        let _ = result_tx.send(Err(e));
                    }
                }
            }
        }
    }

    Ok(())
}

// ── Context Builder ──────────────────────────────────────────────────────────

/// Tear down and rebuild a [`LlamaContext`] with fresh parameters.
///
/// Reuses the already-loaded `model` weights — only the context buffers are
/// reallocated. Called once on startup and again by [`ReconfigureContext`]
/// commands from the bench command.
fn rebuild_context<'a>(
    backend: &'a LlamaBackend,
    model: &'a LlamaModel,
    config: &ModelConfig,
    n_ctx: u32,
    kv_config: &KvCacheConfig,
) -> Result<llama_cpp_2::context::LlamaContext<'a>> {
    let ctx_size = NonZeroU32::new(n_ctx)
        .unwrap_or_else(|| NonZeroU32::new(8192).expect("constant is non-zero"));
    let kv_type = kv_cache_type(&kv_config.bits);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(ctx_size))
        .with_n_batch(config.batch_size)
        .with_n_threads(config.threads as i32)
        .with_n_threads_batch(config.threads as i32)
        .with_type_k(kv_type)
        .with_type_v(kv_type);
    model.new_context(backend, ctx_params).context("failed to create llama context")
}

// ── Generation ───────────────────────────────────────────────────────────────

/// Run one generation request, streaming tokens through `event_tx`.
///
/// Uses `prefix_cache` to skip re-decoding tokens that are already resident
/// in the llama.cpp KV cache from a previous request.  On a warm cache the
/// only tokens that need prefilling are the new conversation turns appended
/// since the last request (typically 50–500 tokens rather than 7 000+).
fn do_generate(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
    req: &GenerateRequest,
    event_tx: &tokio::sync::mpsc::Sender<GenerateEvent>,
    prefix_cache: &mut PrefixCache,
) -> Result<()> {
    let start = std::time::Instant::now();

    // Tokenize the full prompt.
    let prompt_tokens = model
        .str_to_token(&req.prompt, AddBos::Always)
        .context("failed to tokenize prompt")?;
    let prompt_len = prompt_tokens.len();

    // ── Prefix cache ─────────────────────────────────────────────────────────
    //
    // Find how many leading tokens are already in the KV cache.
    // Always re-decode at least the last token to obtain fresh logits for
    // the first generated token.
    let prompt_token_ids: Vec<i32> = prompt_tokens.iter().map(|t| t.0).collect();
    let raw_prefix = prefix_cache.common_prefix_len(&prompt_token_ids);
    let n_past = raw_prefix.saturating_sub(1).min(prompt_len.saturating_sub(1));

    if n_past == 0 {
        ctx.clear_kv_cache();
        debug!(prompt_tokens = prompt_len, n_past = 0, "prefill: cold cache");
    } else {
        // Remove any stale KV entries that follow the reusable prefix
        // (e.g. tokens generated in the previous turn that are no longer
        // part of the new prompt at or after position n_past).
        let _ = ctx.clear_kv_cache_seq(Some(0), Some(n_past as u32), None);
        debug!(
            prompt_tokens = prompt_len,
            n_past,
            new_tokens = prompt_len - n_past,
            "prefill: warm cache hit"
        );
    }

    // Prefill: decode only the tokens that are not yet in the KV cache.
    // Processing all tokens in one shot would fail when the new suffix
    // exceeds n_batch; chunk it accordingly.
    let n_batch = ctx.n_batch() as usize;
    let mut batch = LlamaBatch::new(n_batch, 1);
    let new_tokens = &prompt_tokens[n_past..];
    let n_chunks = new_tokens.chunks(n_batch).count();

    for (chunk_idx, chunk) in new_tokens.chunks(n_batch).enumerate() {
        batch.clear();
        let is_last_chunk = chunk_idx == n_chunks - 1;

        // Log prefill progress periodically to avoid silent freezes, especially for large contexts.
        if n_chunks > 1 && chunk_idx % std::cmp::max(1, n_chunks / 20) == 0 {
            info!(
                "prefill progress: chunk {}/{} ({} new tokens)",
                chunk_idx + 1,
                n_chunks,
                new_tokens.len()
            );
        }
        for (i, &token) in chunk.iter().enumerate() {
            let pos = (n_past + chunk_idx * n_batch + i) as i32;
            // Only request logits for the very last token of the last chunk —
            // that is the position we sample the first generated token from.
            let need_logits = is_last_chunk && i == chunk.len() - 1;
            batch
                .add(token, pos, &[0], need_logits)
                .context("failed to add token to prefill batch")?;
        }
        if let Err(e) = ctx.decode(&mut batch) {
            if n_past > 0 {
                warn!("warm cache prefill failed (likely M-RoPE sequence mismatch); retrying with cold cache...");
                prefix_cache.invalidate();
                return do_generate(model, ctx, req, event_tx, prefix_cache);
            }
            return Err(anyhow::Error::from(e).context("prompt prefill decode failed"));
        }
    }

    // Record the full prompt in the cache so the next request can skip it.
    prefix_cache.update(prompt_token_ids);

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
        // blocking_send is correct here: this runs on a dedicated OS thread,
        // so blocking until the async consumer drains the channel is safe and
        // prevents the Done event from being lost when the GPU outpaces HTTP.
        if event_tx.blocking_send(GenerateEvent::Token(text)).is_err() {
            break;
        }

        sampler.accept(token);
        tokens_generated += 1;
        pos += 1;

        if tokens_generated % 20 == 0 {
            info!("actively generating: {} tokens elapsed...", tokens_generated);
        }

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

    let _ = event_tx.blocking_send(GenerateEvent::Done(GenerateSummary {
        tokens_generated,
        tokens_per_second: tps,
        context_tokens: prompt_len as u32 + tokens_generated,
    }));

    Ok(())
}
