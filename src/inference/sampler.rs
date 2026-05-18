/// Build a llama.cpp [`LlamaSampler`] chain from [`SamplerParams`].
///
/// Only compiled when the `llama-backend` feature is active.
#[cfg(feature = "llama-backend")]
pub fn build_sampler(
    params: &crate::model::backend::SamplerParams,
) -> llama_cpp_2::sampling::LlamaSampler {
    use llama_cpp_2::sampling::LlamaSampler;

    let seed = params.seed.map(|s| s as u32).unwrap_or(u32::MAX);

    LlamaSampler::chain_simple([
        LlamaSampler::top_k(params.top_k as i32),
        LlamaSampler::top_p(params.top_p, 1),
        LlamaSampler::min_p(params.min_p, 1),
        LlamaSampler::temp(params.temperature),
        LlamaSampler::penalties(params.repeat_last_n as i32, params.repeat_penalty, 0.0, 0.0),
        LlamaSampler::dist(seed),
    ])
}
