use serde::{Deserialize, Serialize};

/// KV cache quantization bit-width.
///
/// Serialized/deserialized as an integer (`2`, `3`, `4`, `8`) in `config.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvBits {
    /// 2-bit quantization.
    Two,
    /// 3-bit quantization.
    Three,
    /// 4-bit quantization (default).
    Four,
    /// 8-bit quantization.
    Eight,
}

impl Default for KvBits {
    fn default() -> Self {
        Self::Four
    }
}

impl TryFrom<u8> for KvBits {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            8 => Ok(Self::Eight),
            _ => Err(format!(
                "invalid kv_bits value {value}: must be 2, 3, 4, or 8"
            )),
        }
    }
}

impl From<KvBits> for u8 {
    fn from(bits: KvBits) -> u8 {
        match bits {
            KvBits::Two => 2,
            KvBits::Three => 3,
            KvBits::Four => 4,
            KvBits::Eight => 8,
        }
    }
}

impl Serialize for KvBits {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        u8::from(*self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for KvBits {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = u8::deserialize(deserializer)?;
        Self::try_from(raw).map_err(serde::de::Error::custom)
    }
}

/// KV cache compression backend strategy.
///
/// Serialized/deserialized as a snake_case string (`"llama_native"`,
/// `"turbo_quant"`) in `config.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvStrategy {
    /// llama.cpp native quantized KV cache (`Q4_0` / `Q8_0`). Always available.
    LlamaNative,
    /// TurboQuant KV compression. Requires the `turbo-kv` Cargo feature.
    /// Falls back to `LlamaNative` with a warning when the feature is absent.
    TurboQuant,
}

impl Default for KvStrategy {
    fn default() -> Self {
        Self::LlamaNative
    }
}

/// KV cache configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheConfig {
    /// Quantization bit-width for K and V tensors (default: `4`).
    #[serde(default)]
    pub bits: KvBits,
    /// Which compression backend to use (default: `llama_native`).
    #[serde(default)]
    pub strategy: KvStrategy,
    /// Hard cap on KV cache memory in megabytes. `None` means unlimited.
    pub memory_budget_mb: Option<u32>,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            bits: KvBits::default(),
            strategy: KvStrategy::default(),
            memory_budget_mb: None,
        }
    }
}
