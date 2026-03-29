/// Tracks which prompt tokens are currently resident in the llama.cpp KV cache
/// so the generation loop can skip re-decoding the common prefix on follow-up requests.
///
/// # How it works
///
/// After each generation the cache records the full prompt token sequence.
/// On the next request it computes the longest common prefix between the
/// stored tokens and the new prompt. Any tokens in that prefix are already
/// decoded in the KV cache and can be skipped; only the new suffix needs
/// to be prefilled.
///
/// The KV cache is never left in an inconsistent state: before decoding
/// the new suffix the caller must call `llama_kv_cache_seq_rm` (via
/// `LlamaContext::clear_kv_cache_seq`) to remove any stale entries that
/// follow the common prefix (e.g. the tokens generated in the previous turn).
pub struct PrefixCache {
    /// Token IDs of the last successfully decoded prompt (positions 0..len).
    cached_tokens: Vec<i32>,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self { cached_tokens: Vec::new() }
    }

    /// Returns how many leading tokens from `new_tokens` are already in the
    /// KV cache.  A value of `0` means the cache is cold — clear it entirely
    /// before decoding.
    pub fn common_prefix_len(&self, new_tokens: &[i32]) -> usize {
        self.cached_tokens
            .iter()
            .zip(new_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Record the prompt token sequence after a successful generation.
    ///
    /// Only the prompt tokens are stored (not the generated response tokens).
    /// The next request will include the previous response as part of its
    /// prompt, and the delta will be decoded as new prefix tokens at that point.
    pub fn update(&mut self, prompt_tokens: Vec<i32>) {
        self.cached_tokens = prompt_tokens;
    }

    /// Invalidate the cache record.
    ///
    /// Must be called whenever the llama.cpp context is cleared (e.g. context
    /// overflow, explicit reset, or `ReconfigureContext` command).
    pub fn invalidate(&mut self) {
        self.cached_tokens.clear();
    }

    /// Number of tokens currently tracked in the cache record.
    pub fn cached_len(&self) -> usize {
        self.cached_tokens.len()
    }
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(ids: &[i32]) -> Vec<i32> {
        ids.to_vec()
    }

    #[test]
    fn cold_cache_returns_zero() {
        let cache = PrefixCache::new();
        assert_eq!(cache.common_prefix_len(&tokens(&[1, 2, 3])), 0);
    }

    #[test]
    fn exact_match_returns_full_length() {
        let mut cache = PrefixCache::new();
        cache.update(tokens(&[10, 20, 30]));
        assert_eq!(cache.common_prefix_len(&tokens(&[10, 20, 30])), 3);
    }

    #[test]
    fn prefix_match_returns_prefix_length() {
        let mut cache = PrefixCache::new();
        cache.update(tokens(&[1, 2, 3, 4]));
        // New prompt shares first 3 tokens then diverges
        assert_eq!(cache.common_prefix_len(&tokens(&[1, 2, 3, 99])), 3);
    }

    #[test]
    fn new_prompt_is_superset_returns_cached_len() {
        let mut cache = PrefixCache::new();
        cache.update(tokens(&[1, 2, 3]));
        // New prompt extends the old one — cached prefix is fully reusable
        assert_eq!(cache.common_prefix_len(&tokens(&[1, 2, 3, 4, 5])), 3);
    }

    #[test]
    fn mismatch_at_first_token_returns_zero() {
        let mut cache = PrefixCache::new();
        cache.update(tokens(&[1, 2, 3]));
        assert_eq!(cache.common_prefix_len(&tokens(&[99, 2, 3])), 0);
    }

    #[test]
    fn invalidate_resets_to_cold() {
        let mut cache = PrefixCache::new();
        cache.update(tokens(&[1, 2, 3]));
        cache.invalidate();
        assert_eq!(cache.common_prefix_len(&tokens(&[1, 2, 3])), 0);
        assert_eq!(cache.cached_len(), 0);
    }

    #[test]
    fn update_replaces_previous_record() {
        let mut cache = PrefixCache::new();
        cache.update(tokens(&[1, 2, 3]));
        cache.update(tokens(&[1, 2, 3, 4, 5]));
        assert_eq!(cache.common_prefix_len(&tokens(&[1, 2, 3, 4, 5, 6])), 5);
    }
}
