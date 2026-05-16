# build-backend.ps1 — Build both llama-server backend variants for TurboQuantLoader.
#
# Produces:
#   J:/llama/llama-server-vulkan.exe    — ggml-org/llama.cpp, Vulkan + CUDA
#   J:/llama/llama-server-turboquant.exe — TheTom/llama-cpp-turboquant, CUDA only
#
# Prerequisites:
#   - CMake 3.20+      : cmake --version
#   - Visual Studio Build Tools 2022 (MSVC v143)
#   - CUDA Toolkit 12.x or 13.x        : nvcc --version
#   - Vulkan SDK (for Vulkan build)     : vulkaninfo
#   - Git                               : git --version
#
# Usage:
#   .\scripts\build-backend.ps1                  # build both
#   .\scripts\build-backend.ps1 -Target vulkan   # only Vulkan build
#   .\scripts\build-backend.ps1 -Target turboquant  # only TurboQuant build

param(
    [ValidateSet("vulkan", "turboquant", "both")]
    [string]$Target = "both",

    [string]$OutputDir = "J:/llama",

    [string]$BuildDir = "$env:TEMP/tql-builds"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Helpers ───────────────────────────────────────────────────────────────────

function Write-Step { param([string]$Msg) Write-Host "`n==> $Msg" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Msg) Write-Host "    OK: $Msg" -ForegroundColor Green }
function Write-Fail { param([string]$Msg) Write-Host "    FAIL: $Msg" -ForegroundColor Red; exit 1 }

function Assert-Command {
    param([string]$Name, [string]$Hint)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Fail "$Name not found. $Hint"
    }
    Write-Ok "$Name available"
}

# ── Preflight checks ──────────────────────────────────────────────────────────

Write-Step "Checking prerequisites"
Assert-Command "git"    "Install Git from https://git-scm.com"
Assert-Command "cmake"  "Install CMake 3.20+ and add to PATH"
Assert-Command "ninja"  "Install Ninja: winget install Ninja-build.Ninja"

if ($Target -in @("vulkan", "both")) {
    if (-not (Get-Command "vulkaninfo" -ErrorAction SilentlyContinue)) {
        Write-Host "    WARN: vulkaninfo not found — install Vulkan SDK for Vulkan GPU support" -ForegroundColor Yellow
    } else {
        Write-Ok "Vulkan SDK present"
    }
}

if ($Target -in @("turboquant", "both")) {
    Assert-Command "nvcc" "Install CUDA Toolkit 12.x or 13.x from https://developer.nvidia.com/cuda-downloads"
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
}

# ── Build: llama-server (Vulkan) ──────────────────────────────────────────────

function Build-VulkanServer {
    Write-Step "Building llama-server (Vulkan build) from ggml-org/llama.cpp"

    $srcDir = "$BuildDir/llama-cpp-vulkan"
    $bldDir = "$srcDir/build-vulkan"

    if (-not (Test-Path $srcDir)) {
        Write-Host "    Cloning ggml-org/llama.cpp..."
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git $srcDir
    } else {
        Write-Host "    Updating existing clone..."
        git -C $srcDir pull --ff-only
    }

    New-Item -ItemType Directory -Path $bldDir -Force | Out-Null

    Write-Host "    Running CMake configure..."
    cmake -S $srcDir -B $bldDir `
        -G Ninja `
        -DCMAKE_BUILD_TYPE=Release `
        -DGGML_VULKAN=ON `
        -DGGML_CUDA=OFF `
        -DLLAMA_BUILD_SERVER=ON `
        -DLLAMA_BUILD_TESTS=OFF `
        -DLLAMA_BUILD_EXAMPLES=OFF

    Write-Host "    Building (this takes several minutes)..."
    cmake --build $bldDir --config Release --target llama-server -j

    $exe = "$bldDir/bin/llama-server.exe"
    if (-not (Test-Path $exe)) {
        # Some cmake configs place the binary at $bldDir/llama-server.exe
        $exe = "$bldDir/llama-server.exe"
    }
    if (-not (Test-Path $exe)) {
        Write-Fail "llama-server.exe not found after build. Check build output above."
    }

    $dest = "$OutputDir/llama-server-vulkan.exe"
    Copy-Item $exe $dest -Force

    # Copy required DLLs (ggml-vulkan.dll etc.) to the output dir.
    $dllSrc = Split-Path $exe
    Get-ChildItem "$dllSrc/*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName $OutputDir -Force
        Write-Host "    Copied: $($_.Name)"
    }

    Write-Ok "llama-server-vulkan.exe -> $dest"
}

# ── Build: llama-server-turboquant (CUDA) ─────────────────────────────────────

function Build-TurboQuantServer {
    Write-Step "Building llama-server-turboquant (CUDA) from TheTom/llama-cpp-turboquant"

    $repo = "https://github.com/TheTom/llama-cpp-turboquant.git"
    $branch = "feature/turboquant-kv-cache"
    $srcDir = "$BuildDir/llama-cpp-turboquant"
    $bldDir = "$srcDir/build-turboquant"

    if (-not (Test-Path $srcDir)) {
        Write-Host "    Cloning TheTom/llama-cpp-turboquant (branch: $branch)..."
        git clone --depth 1 --branch $branch $repo $srcDir
    } else {
        Write-Host "    Updating existing clone..."
        git -C $srcDir pull --ff-only
    }

    New-Item -ItemType Directory -Path $bldDir -Force | Out-Null

    Write-Host "    Running CMake configure (CUDA + TurboQuant)..."
    cmake -S $srcDir -B $bldDir `
        -G Ninja `
        -DCMAKE_BUILD_TYPE=Release `
        -DGGML_CUDA=ON `
        -DGGML_VULKAN=OFF `
        -DLLAMA_BUILD_SERVER=ON `
        -DLLAMA_BUILD_TESTS=OFF `
        -DLLAMA_BUILD_EXAMPLES=OFF

    Write-Host "    Building (this takes several minutes)..."
    cmake --build $bldDir --config Release --target llama-server -j

    $exe = "$bldDir/bin/llama-server.exe"
    if (-not (Test-Path $exe)) {
        $exe = "$bldDir/llama-server.exe"
    }
    if (-not (Test-Path $exe)) {
        Write-Fail "llama-server.exe not found after TurboQuant build."
    }

    $dest = "$OutputDir/llama-server-turboquant.exe"
    Copy-Item $exe $dest -Force

    # Validate that --cache-type-k turbo3 is a known flag in this build.
    $helpOutput = & $dest --help 2>&1
    if ($helpOutput -match "turbo") {
        Write-Ok "TurboQuant flag verified in binary"
    } else {
        Write-Host "    WARN: --cache-type-k turbo* not found in --help. Check TurboQuant branch." -ForegroundColor Yellow
    }

    # Copy CUDA DLLs.
    $dllSrc = Split-Path $exe
    Get-ChildItem "$dllSrc/*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName $OutputDir -Force
        Write-Host "    Copied: $($_.Name)"
    }

    Write-Ok "llama-server-turboquant.exe -> $dest"
}

# ── Main ──────────────────────────────────────────────────────────────────────

Write-Step "TurboQuantLoader backend build — target: $Target"

if ($Target -in @("vulkan", "both")) {
    Build-VulkanServer
}

if ($Target -in @("turboquant", "both")) {
    Build-TurboQuantServer
}

Write-Step "Build complete"
Write-Host ""
Write-Host "Output binaries:" -ForegroundColor White
if ($Target -in @("vulkan", "both"))      { Write-Host "  $OutputDir/llama-server-vulkan.exe" }
if ($Target -in @("turboquant", "both"))  { Write-Host "  $OutputDir/llama-server-turboquant.exe" }
Write-Host ""
Write-Host "Update config.toml [backend] binary_path to point at the desired binary." -ForegroundColor Yellow
