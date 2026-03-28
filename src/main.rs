mod config;
mod inference;
mod kv_cache;
mod model;
mod server;
mod tui;

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use tracing::info;

use config::{load_from_file, AppConfig, CliOverrides, KvBits};
use inference::engine::{ChatMessage, ChatRequest, InferenceEngine};
use model::backend::{GenerateEvent, SamplerParams};
use model::registry::ModelRegistry;

// ─── CLI Definition ───────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "turboquant-loader",
    version,
    about = "Local LLM inference server with TurboQuant KV cache compression and OpenAI-compatible API"
)]
struct Cli {
    /// Path to the TOML configuration file.
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the OpenAI-compatible HTTP API server on the configured host:port.
    Serve(ServeArgs),
    /// Open an interactive terminal chat session.
    Run(RunArgs),
    /// Benchmark context sizes versus KV cache configurations.
    Bench(BenchArgs),
    /// List all GGUF models discovered in the configured models directory.
    List,
}

#[derive(Parser)]
struct ServeArgs {
    /// Override the model file to load (overrides `config.toml`).
    #[arg(short, long)]
    model: Option<PathBuf>,
    /// Override the HTTP listen port (overrides `config.toml`).
    #[arg(short, long)]
    port: Option<u16>,
    /// Override the context window size in tokens (overrides `config.toml`).
    #[arg(long)]
    context_size: Option<u32>,
    /// Override the number of GPU layers to offload; `-1` offloads all (overrides `config.toml`).
    #[arg(long)]
    n_gpu_layers: Option<i32>,
    /// Override the KV cache bit-width: `2`, `3`, `4`, or `8` (overrides `config.toml`).
    #[arg(long)]
    kv_bits: Option<u8>,
}

#[derive(Parser)]
struct RunArgs {
    /// Override the model file to load (overrides `config.toml`).
    #[arg(short, long)]
    model: Option<PathBuf>,
    /// Override the context window size in tokens (overrides `config.toml`).
    #[arg(long)]
    context_size: Option<u32>,
}

#[derive(Parser)]
struct BenchArgs {
    /// Comma-separated context sizes to benchmark (e.g. `1024,8192,32768`).
    #[arg(long, default_value = "1024,8192,32768")]
    context_sizes: String,
    /// Comma-separated KV bit-widths to benchmark (e.g. `4,8`).
    #[arg(long, default_value = "4,8")]
    bits: String,
    /// Write benchmark results to this JSON file.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

// ─── Entry Point ─────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let mut app_config = if cli.config.exists() {
        load_from_file(&cli.config)?
    } else {
        tracing::warn!(
            path = %cli.config.display(),
            "config file not found — using built-in defaults"
        );
        AppConfig::default()
    };

    match cli.command {
        Command::Serve(args) => {
            let overrides = serve_overrides(&args)?;
            app_config.apply_cli_overrides(&overrides);
            cmd_serve(app_config).await
        }
        Command::Run(args) => {
            let overrides = run_overrides(&args);
            app_config.apply_cli_overrides(&overrides);
            cmd_run(app_config).await
        }
        Command::Bench(args) => {
            app_config.apply_cli_overrides(&CliOverrides::default());
            cmd_bench(app_config, args)
        }
        Command::List => cmd_list(&app_config),
    }
}

// ─── Override builders ────────────────────────────────────────────────────────

fn serve_overrides(args: &ServeArgs) -> Result<CliOverrides> {
    let kv_bits = args
        .kv_bits
        .map(KvBits::try_from)
        .transpose()
        .map_err(|e| anyhow::anyhow!("--kv-bits: {e}"))?;

    Ok(CliOverrides {
        model_path: args.model.clone(),
        port: args.port,
        context_size: args.context_size,
        n_gpu_layers: args.n_gpu_layers,
        kv_bits,
    })
}

fn run_overrides(args: &RunArgs) -> CliOverrides {
    CliOverrides {
        model_path: args.model.clone(),
        context_size: args.context_size,
        ..CliOverrides::default()
    }
}

// ─── Command handlers ─────────────────────────────────────────────────────────

async fn cmd_serve(_config: AppConfig) -> Result<()> {
    bail!("serve command not yet implemented — coming in Phase 4");
}

/// Interactive terminal chat session.
///
/// Loads the model on a blocking thread, then drives a ChatML REPL until the
/// user types `/quit`, `/exit`, or sends EOF (Ctrl-D).
async fn cmd_run(config: AppConfig) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, BufReader};

    println!("Loading model: {}", config.model.model_path.display());
    println!("Context size:  {} tokens", config.model.context_size);
    println!("Type /quit or press Ctrl-D to exit.\n");

    // Model loading is blocking; run it off the async executor.
    let engine = tokio::task::spawn_blocking(move || InferenceEngine::new(config))
        .await
        .map_err(|e| anyhow::anyhow!("inference thread panicked: {e}"))??;

    println!("Model ready: {} ({} ctx)\n", engine.model_name(), engine.context_size());

    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();
    let mut history: Vec<ChatMessage> = Vec::new();

    loop {
        // Print prompt and flush.
        print!("> ");
        std::io::stdout().flush()?;

        let line = match lines.next_line().await? {
            Some(l) => l,
            None => break, // EOF (Ctrl-D)
        };
        let line = line.trim().to_string();

        if line.is_empty() {
            continue;
        }
        if line == "/quit" || line == "/exit" {
            break;
        }

        history.push(ChatMessage { role: "user".into(), content: line });

        let req = ChatRequest {
            messages: history.clone(),
            max_tokens: 2048,
            sampler: SamplerParams::default(),
        };

        let mut stream = engine.chat(req)?;
        let mut response = String::new();

        loop {
            match stream.next_event().await {
                Some(GenerateEvent::Token(tok)) => {
                    print!("{tok}");
                    std::io::stdout().flush()?;
                    response.push_str(&tok);
                }
                Some(GenerateEvent::Done(summary)) => {
                    println!();
                    info!(
                        tokens = summary.tokens_generated,
                        tps = format!("{:.1}", summary.tokens_per_second),
                        ctx = summary.context_tokens,
                        "generation complete"
                    );
                    break;
                }
                Some(GenerateEvent::Error(e)) => {
                    eprintln!("\nerror: {e}");
                    break;
                }
                None => break,
            }
        }

        history.push(ChatMessage { role: "assistant".into(), content: response });
        println!();
    }

    Ok(())
}

fn cmd_bench(_config: AppConfig, _args: BenchArgs) -> Result<()> {
    bail!("bench command not yet implemented — coming in Phase 3");
}

/// List all GGUF models discovered in `config.model.models_dir`.
fn cmd_list(config: &AppConfig) -> Result<()> {
    let dir = &config.model.models_dir;

    info!(models_dir = %dir.display(), "scanning for models");

    let entries = ModelRegistry::scan(dir)?;

    if entries.is_empty() {
        println!("No models found in: {}", dir.display());
        return Ok(());
    }

    println!("Found {} model(s) in: {}\n", entries.len(), dir.display());
    println!("{:<60}  {:>10}  {}", "Name", "Size", "Flags");
    println!("{}", "-".repeat(80));

    for entry in &entries {
        let size = format_size(entry.size_bytes);
        let flags = if entry.has_mmproj { "[mmproj]" } else { "" };
        println!("{:<60}  {:>10}  {}", entry.name, size, flags);
    }

    Ok(())
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Format a byte count as a human-readable string (GB / MB / KB).
fn format_size(bytes: u64) -> String {
    const GB: u64 = 1_073_741_824;
    const MB: u64 = 1_048_576;
    const KB: u64 = 1_024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}
