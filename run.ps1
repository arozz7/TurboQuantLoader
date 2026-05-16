# TurboQuantLoader — Windows launch script.
#
# Phase 6+: TurboQuantLoader is now a process manager + API proxy.
# It spawns llama-server as a subprocess (Vulkan build for Intel Arc B70).
# No GPU feature flags are needed at the Rust level; GPU is handled by llama-server.
#
# llama-server binary: J:/llama/llama-server.exe (Vulkan-capable, build b9189)
# Primary GPU: Intel Arc Pro B70 (32 GB VRAM, device [1] in Vulkan)
# Overflow GPU: RTX 4070 Ti Super (16 GB, device [0] in Vulkan)

# Prevent WDDM VRAM fragmentation on NVIDIA when llama-server uses CUDA internally.
$env:LLAMA_CUDA_NO_GRAPHS = "1"

Write-Host "Starting TurboQuantLoader (proxy mode — llama-server backend)" -ForegroundColor Cyan

# No feature flags required: GPU backend is inside llama-server, not this process.
cargo run --release -- serve
