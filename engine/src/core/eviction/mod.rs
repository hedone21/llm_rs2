use crate::core::kv_cache::KVCache;
use anyhow::Result;

/// Trait for KV cache eviction strategies.
///
/// Implementations decide WHEN and HOW to evict tokens from the cache.
/// This follows the Strategy pattern and SOLID principles:
/// - Single Responsibility: each policy handles one eviction strategy
/// - Open/Closed: add new policies without modifying existing code
/// - Liskov Substitution: all policies are interchangeable via this trait
/// - Dependency Inversion: consumers depend on this trait, not concrete types
pub trait EvictionPolicy: Send + Sync {
    /// Determines whether eviction should be triggered based on cache state
    /// and available system memory.
    fn should_evict(&self, cache: &KVCache, mem_available: usize) -> bool;

    /// Performs the actual eviction, reducing cache to `target_len` tokens.
    fn evict(&self, cache: &mut KVCache, target_len: usize) -> Result<()>;

    /// Returns the name of this policy (for logging/debugging).
    fn name(&self) -> &str;

    /// Performs eviction using per-token importance scores.
    /// Default implementation ignores scores and delegates to `evict()`.
    /// Override in score-aware policies like H2O.
    fn evict_with_scores(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        _importance: &[f32],
    ) -> Result<()> {
        self.evict(cache, target_len)
    }

    /// Per-KV-head eviction with GQA-aware importance scores.
    ///
    /// `head_importance` is `[n_kv_heads * max_seq_len]` (row-major): each KV head
    /// has its own importance ranking, enabling per-head token selection.
    ///
    /// Default: ignores head scores, delegates to `evict_with_scores()`.
    /// Override in GQA-aware policies like H2O+.
    fn evict_with_head_scores(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        flat_importance: &[f32],
        _head_importance: &[f32],
        _n_kv_heads: usize,
    ) -> Result<()> {
        self.evict_with_scores(cache, target_len, flat_importance)
    }
}

pub mod h2o;
pub mod h2o_plus;
pub mod no_eviction;
pub mod sliding_window;
pub mod streaming_llm;

pub use h2o::H2OPolicy;
pub use h2o_plus::H2OPlusPolicy;
pub use no_eviction::NoEvictionPolicy;
pub use sliding_window::SlidingWindowPolicy;
pub use streaming_llm::StreamingLLMPolicy;
