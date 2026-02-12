use super::EvictionPolicy;
use crate::core::kv_cache::KVCache;
use anyhow::Result;

/// SnapKV eviction policy (stub implementation).
///
/// SnapKV selectively keeps tokens with high attention scores, dropping those
/// rarely attended to. This requires access to per-head attention scores from
/// the last few decoding steps.
///
/// ## Limitations (current implementation)
///
/// The actual attention scores are computed inside `LlamaLayer::forward_gen()`
/// and are not yet exposed to external consumers. Therefore, this implementation
/// provides the **interface only** and falls back to a simple sliding window
/// eviction strategy internally.
///
/// ## Future Work
///
/// To enable real SnapKV behavior:
/// 1. Expose attention scores from `LlamaLayer` via a callback or output buffer
/// 2. Accumulate per-head importance scores in this policy
/// 3. Use those scores to select which tokens to keep
pub struct SnapKVPolicy {
    /// Number of recent tokens to observe for score accumulation
    observation_window: usize,
    /// Fraction of tokens to keep (0.0 to 1.0)
    keep_ratio: f32,
    /// Number of prefix tokens to always protect
    protected_prefix: usize,
}

impl SnapKVPolicy {
    pub fn new(observation_window: usize, keep_ratio: f32, protected_prefix: usize) -> Self {
        Self {
            observation_window,
            keep_ratio: keep_ratio.clamp(0.0, 1.0),
            protected_prefix,
        }
    }
}

impl EvictionPolicy for SnapKVPolicy {
    fn should_evict(&self, cache: &KVCache, _mem_available: usize) -> bool {
        // Trigger when the cache has significantly more tokens than what we'd keep
        let keep_count = ((cache.current_pos as f32) * self.keep_ratio) as usize;
        let effective_keep = keep_count + self.protected_prefix;
        cache.current_pos > effective_keep + self.observation_window
    }

    fn evict(&self, cache: &mut KVCache, target_len: usize) -> Result<()> {
        let current = cache.current_pos;
        let keep = target_len.min(
            ((current as f32) * self.keep_ratio) as usize + self.protected_prefix,
        );

        if current <= keep {
            return Ok(());
        }

        // STUB: Without actual attention scores, fall back to sliding window behavior
        // (keep the most recent tokens + protected prefix).
        //
        // TODO: When attention scores become available:
        // 1. Compute per-token importance as sum of attention weights across heads
        // 2. Sort tokens by importance (excluding protected prefix)
        // 3. Keep top-k tokens, compact the buffer

        log::debug!(
            "SnapKV (stub): falling back to sliding window, evicting {} tokens",
            current - keep
        );

        let prune_count = current - keep;
        if self.protected_prefix == 0 {
            cache.prune_prefix(prune_count)?;
        } else {
            // Same approach as SlidingWindowPolicy: shift data after removing
            // tokens between protected_prefix and protected_prefix + prune_count
            let shape = cache.k_buffer.shape().dims();
            let heads = shape[2];
            let dim = shape[3];
            let type_size = cache.k_buffer.dtype().size();
            let bytes_per_pos = heads * dim * type_size;

            let src_start = (self.protected_prefix + prune_count) * bytes_per_pos;
            let dst_start = self.protected_prefix * bytes_per_pos;
            let move_bytes = (current - self.protected_prefix - prune_count) * bytes_per_pos;

            let k_ptr = cache.k_buffer.as_mut_ptr();
            let v_ptr = cache.v_buffer.as_mut_ptr();

            if k_ptr.is_null() || v_ptr.is_null() {
                return Err(anyhow::anyhow!(
                    "Cannot evict: null buffer pointers (GPU-only buffers not supported)"
                ));
            }

            unsafe {
                std::ptr::copy(k_ptr.add(src_start), k_ptr.add(dst_start), move_bytes);
                std::ptr::copy(v_ptr.add(src_start), v_ptr.add(dst_start), move_bytes);
            }

            cache.current_pos -= prune_count;
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "snap_kv"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::shared_buffer::SharedBuffer;
    use crate::core::buffer::DType;
    use crate::core::shape::Shape;
    use crate::core::tensor::Tensor;
    use std::sync::Arc;

    fn make_cache(pos: usize) -> KVCache {
        let max_seq = 100;
        let backend = Arc::new(CpuBackend::new());
        let buf_size = max_seq * 1 * 4 * 4;
        let k = Tensor::new(
            Shape::new(vec![1, max_seq, 1, 4]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq, 1, 4]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend.clone(),
        );
        let mut cache = KVCache::new(k, v, max_seq);
        cache.current_pos = pos;
        cache
    }

    #[test]
    fn test_should_evict() {
        let policy = SnapKVPolicy::new(5, 0.5, 0);
        let cache = make_cache(20);
        // keep = 20 * 0.5 = 10, effective = 10, 20 > 10 + 5 = 15 → true
        assert!(policy.should_evict(&cache, 0));

        let cache = make_cache(10);
        // keep = 10 * 0.5 = 5, effective = 5, 10 > 5 + 5 = 10 → false
        assert!(!policy.should_evict(&cache, 0));
    }

    #[test]
    fn test_evict_stub_falls_back_to_sliding() {
        let _ = env_logger::try_init();
        let policy = SnapKVPolicy::new(5, 0.5, 0);
        let mut cache = make_cache(20);

        // target_len = 10, keep = min(10, 20*0.5+0) = 10
        policy.evict(&mut cache, 10).unwrap();
        assert_eq!(cache.current_pos, 10);
    }

    #[test]
    fn test_evict_with_prefix() {
        let policy = SnapKVPolicy::new(5, 0.5, 3);
        let mut cache = make_cache(20);

        // target_len = 13, keep = min(13, 20*0.5+3) = 13
        policy.evict(&mut cache, 13).unwrap();
        assert_eq!(cache.current_pos, 13);
    }

    #[test]
    fn test_name() {
        assert_eq!(SnapKVPolicy::new(5, 0.5, 0).name(), "snap_kv");
    }
}
