/// Per-device GPU telemetry snapshot.
#[derive(Debug, Clone)]
pub struct GpuStats {
    pub device_index: u32,
    pub name: String,
    pub vram_used_mb: u32,
    pub vram_total_mb: u32,
    pub utilization_pct: u32,
}

/// Query all visible NVIDIA GPUs via NVML.
///
/// Returns an empty `Vec` on any NVML init failure or on non-CUDA builds.
#[cfg(feature = "cuda")]
pub fn query_all_gpus() -> Vec<GpuStats> {
    use nvml_wrapper::Nvml;

    let nvml = match Nvml::init() {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!(error = %e, "NVML init failed — GPU stats unavailable");
            return vec![];
        }
    };

    let count = match nvml.device_count() {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "NVML device count failed");
            return vec![];
        }
    };

    (0..count)
        .filter_map(|i| {
            let dev = nvml.device_by_index(i).ok()?;
            let name = dev.name().ok()?;
            let mem = dev.memory_info().ok()?;
            let util = dev.utilization_rates().ok()?;
            Some(GpuStats {
                device_index: i,
                name,
                vram_used_mb: (mem.used / 1_048_576) as u32,
                vram_total_mb: (mem.total / 1_048_576) as u32,
                utilization_pct: util.gpu,
            })
        })
        .collect()
}

/// Non-CUDA build stub — always returns empty.
#[cfg(not(feature = "cuda"))]
pub fn query_all_gpus() -> Vec<GpuStats> {
    vec![]
}
