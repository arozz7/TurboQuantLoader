# Phase 2 — Smoke Test Checklist

Run these checks after building to verify Phase 2 works end-to-end.

## Prerequisites

- A GGUF model file present at the path set in `config.toml` → `model.model_path`
- For GPU tests: CUDA Toolkit 12.x, CMake, and LLVM on PATH

---

## 1. CPU-only build (no GPU, no libclang required)

```sh
cargo build --no-default-features
```

**Expected:** Compiles without errors. No GPU drivers or LLVM needed.

---

## 2. CUDA build

```sh
# Windows — CMake must be on PATH
$env:PATH = "C:\Program Files\CMake\bin;" + $env:PATH
cargo build --features cuda
```

**Expected:** Compiles. llama.cpp C++ is compiled via CMake (slow on first build).

---

## 3. `list` command (no model file needed)

```sh
cargo run --no-default-features -- list
```

**Expected:** Prints models found in `config.toml` → `model.models_dir`, or
"No models found in: models".

---

## 4. `run` command — no backend (should fail with helpful error)

```sh
cargo run --no-default-features -- run
```

**Expected:** Error message explaining which feature flags to use, not a panic.

---

## 5. `run` command — CUDA (requires model file)

```sh
# Windows
$env:PATH = "C:\Program Files\CMake\bin;" + $env:PATH
cargo run --release --features cuda -- run
```

**Expected:**
1. "Loading model: ..." printed
2. Model weights load (may take 10–30 s for 14.7 GB)
3. "Model ready: <name> (<N> ctx)" printed
4. `> ` prompt appears
5. Type a message, press Enter → tokens stream in real time
6. After generation: `[INFO] generation complete tokens=N tps=X.X ctx=N`
7. New `> ` prompt appears
8. `/quit` exits cleanly

---

## 6. Multi-GPU verification (RTX 4070 Ti Super + RTX 2060)

During `run`, open a second terminal and run:

```sh
nvidia-smi
```

**Expected:** Both GPU 0 (4070 Ti Super) and GPU 1 (2060) show non-zero memory usage,
confirming that model layers are split across both GPUs.

---

## 7. Context size limit

- Set `context_size = 512` in `config.toml`
- Run `cargo run --features cuda -- run`
- Send a very long message (>512 tokens)

**Expected:** Generation may truncate or the model may produce degraded output,
but the binary should not crash or panic.

---

## Known Limitations (Phase 2)

- `serve` and `bench` commands are stubs (they `bail!` with a Phase 4/3 message)
- `tokenize()` / `detokenize()` on the backend are approximations only
- KV cache type is hardcoded to Q4_0 (Phase 3 will read it from config)
- No conversation context pruning — very long conversations will eventually fill the KV cache
