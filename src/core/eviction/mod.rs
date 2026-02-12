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
}

pub mod no_eviction;
pub mod sliding_window;
pub mod snap_kv;

pub use no_eviction::NoEvictionPolicy;
pub use sliding_window::SlidingWindowPolicy;
pub use snap_kv::SnapKVPolicy;
