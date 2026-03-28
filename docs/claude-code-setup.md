# Claude Code Setup Guide

How to configure Claude Code (and other OpenAI-compatible clients) to use TurboQuantLoader
as a local inference backend.

## Prerequisites

1. TurboQuantLoader server running: `cargo run --release -- serve`
2. Verify it's up: `curl http://127.0.0.1:7432/health`

## Claude Code

Open your Claude Code settings and add a custom API configuration:

```json
{
  "openai": {
    "apiKey": "local",
    "baseUrl": "http://127.0.0.1:7432/v1",
    "model": "Qwen3.5-35B-A3B-UD-IQ3_XXS"
  }
}
```

Or via environment variables before launching Claude Code:

```powershell
$env:OPENAI_API_KEY  = "local"
$env:OPENAI_BASE_URL = "http://127.0.0.1:7432/v1"
```

## Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat inference (streaming + non-streaming) |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Server status + VRAM usage |

## Changing the Model

The model is set in `config.toml`:
```toml
[model]
model_path = "J:/llama/Models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf"
```

Restart the server after changing the model. The model name in the API response
corresponds to the GGUF filename without extension.

## Adjusting KV Cache Compression

In `config.toml`:
```toml
[kv_cache]
bits     = 4    # 4 = default, 8 = less compression, 3/2 = more aggressive
strategy = "llama_native"
```

Restart required after changes.

## Context Size

Default is 8,192 tokens. Increase in `config.toml`:
```toml
[model]
context_size = 32768   # 32k — safe for most use cases with 4-bit KV cache
# context_size = 131072  # 128k — needs monitoring of VRAM
```

Watch `/health` VRAM figures when increasing context size.
