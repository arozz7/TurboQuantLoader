use serde::{Deserialize, Serialize};

/// KV cache tensor data type, matching llama.cpp's `--cache-type-k` /
/// `--cache-type-v` accepted values.
///
/// Serialized/deserialized as a lowercase string (`"f16"`, `"q4_0"`, ...) in
/// `config.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum KvType {
    #[serde(rename = "f32")]
    F32,
    /// No quantization (llama.cpp's own default) — highest quality and
    /// fastest per-token attention, largest memory footprint.
    #[default]
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "bf16")]
    Bf16,
    #[serde(rename = "q8_0")]
    Q8_0,
    #[serde(rename = "q4_0")]
    Q4_0,
    #[serde(rename = "q4_1")]
    Q4_1,
    #[serde(rename = "iq4_nl")]
    Iq4Nl,
    #[serde(rename = "q5_0")]
    Q5_0,
    #[serde(rename = "q5_1")]
    Q5_1,
}

impl KvType {
    /// The exact string llama-server's `--cache-type-k` / `--cache-type-v`
    /// CLI flags expect.
    pub fn as_cli_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Iq4Nl => "iq4_nl",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
        }
    }

    /// Approximate on-disk/VRAM bytes per element, used to estimate KV cache
    /// compression ratio relative to the F16 baseline. Block-quant overhead
    /// (scale/min values) is ignored — this is a rough estimate, not exact.
    pub fn bytes_per_element(self) -> f32 {
        match self {
            Self::F32 => 4.0,
            Self::F16 | Self::Bf16 => 2.0,
            Self::Q8_0 => 1.0,
            Self::Q5_0 | Self::Q5_1 => 0.625,
            Self::Q4_0 | Self::Q4_1 | Self::Iq4Nl => 0.5,
        }
    }
}

impl std::str::FromStr for KvType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::Bf16),
            "q8_0" => Ok(Self::Q8_0),
            "q4_0" => Ok(Self::Q4_0),
            "q4_1" => Ok(Self::Q4_1),
            "iq4_nl" => Ok(Self::Iq4Nl),
            "q5_0" => Ok(Self::Q5_0),
            "q5_1" => Ok(Self::Q5_1),
            _ => Err(format!(
                "invalid KV cache type '{s}': must be one of f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1"
            )),
        }
    }
}

/// KV cache compression backend strategy.
///
/// Serialized/deserialized as a snake_case string (`"llama_native"`,
/// `"turbo_quant"`) in `config.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum KvStrategy {
    /// llama.cpp native quantized KV cache (`Q4_0` / `Q8_0`). Always available.
    #[default]
    LlamaNative,
    /// TurboQuant KV compression. Requires the `turbo-kv` Cargo feature.
    /// Falls back to `LlamaNative` with a warning when the feature is absent.
    TurboQuant,
}

/// KV cache configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KvCacheConfig {
    /// Quantization type for the K (key) tensor cache (default: `f16`, i.e.
    /// unquantized).
    #[serde(default)]
    pub type_k: KvType,
    /// Quantization type for the V (value) tensor cache (default: `f16`, i.e.
    /// unquantized).
    #[serde(default)]
    pub type_v: KvType,
    /// Which compression backend to use (default: `llama_native`).
    #[serde(default)]
    pub strategy: KvStrategy,
    /// Hard cap on KV cache memory in megabytes. `None` means unlimited.
    pub memory_budget_mb: Option<u32>,
}
