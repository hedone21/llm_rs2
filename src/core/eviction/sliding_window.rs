use super::EvictionPolicy;
use crate::core::kv_cache::KVCache;
use anyhow::Result;

/// Sliding window eviction policy.
///
/// Keeps only the most recent `window_size` tokens in the cache.
/// Optionally protects a prefix of `protected_prefix` tokens (e.g., system prompt)
/// from being evicted.
///
/// ## Example
/// ```text
/// window_size=4, protected_prefix=2
///
/// Before (current_pos=8):
/// [P0][P1][T2][T3][T4][T5][T6][T7]
///  protected   ───── evict ─────  keep
///
/// After (current_pos=6):
/// [P0][P1][T4][T5][T6][T7][_][_]
/// ```
pub struct SlidingWindowPolicy {
    window_size: usize,
    protected_prefix: usize,
}

impl SlidingWindowPolicy {
    pub fn new(window_size: usize, protected_prefix: usize) -> Self {
        Self {
            window_size,
            protected_prefix,
        }
    }
}

impl EvictionPolicy for SlidingWindowPolicy {
    fn should_evict(&self, cache: &KVCache, _mem_available: usize) -> bool {
        cache.current_pos > self.window_size + self.protected_prefix
    }

    fn evict(&self, cache: &mut KVCache, target_len: usize) -> Result<()> {
        let current = cache.current_pos;
        // Determine how many tokens to keep: min of target_len and window_size + protected
        let max_keep = self.window_size + self.protected_prefix;
        let keep = target_len.min(max_keep);

        if current <= keep {
            return Ok(());
        }

        // Number of tokens to remove from the non-protected region
        let removable_start = self.protected_prefix;
        let removable_count = current - removable_start;
        let tokens_to_keep_after_prefix = keep.saturating_sub(self.protected_prefix);

        if tokens_to_keep_after_prefix >= removable_count {
            return Ok(()); // Nothing to remove
        }

        let prune_count = removable_count - tokens_to_keep_after_prefix;

        // We need to prune from position `protected_prefix`, not from position 0.
        // But KVCache::prune_prefix removes from position 0.
        // If protected_prefix == 0, we just prune from the front.
        // If protected_prefix > 0, we need to shift data in a more complex way.
        //
        // For now, handle the common case: prune_prefix removes the oldest tokens.
        // Protected prefix: the first `protected_prefix` tokens are preserved.
        // We effectively remove tokens from index `protected_prefix` to
        // `protected_prefix + prune_count - 1`, keeping both the prefix and the tail.

        if self.protected_prefix == 0 {
            cache.prune_prefix(prune_count)?;
        } else {
            // For protected prefix scenario:
            // We need to move [protected_prefix+prune_count .. current_pos] to [protected_prefix ..]
            // This is a more targeted shift within the buffer.
            let shape = cache.k_buffer.shape().dims();
            let heads = shape[2];
            let dim = shape[3];
            let type_size = cache.k_buffer.dtype().size();
            let elems_per_pos = heads * dim;
            let bytes_per_pos = elems_per_pos * type_size;

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
        "sliding_window"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::shared_buffer::SharedBuffer;
    use crate::core::buffer::{Buffer, DType};
    use crate::core::shape::Shape;
    use crate::core::tensor::Tensor;
    use std::sync::Arc;

    fn make_cache_with_data(num_tokens: usize) -> KVCache {
        let max_seq = 100;
        let heads = 1;
        let dim = 4;
        let backend = Arc::new(CpuBackend::new());
        let buf_size = max_seq * heads * dim * 4;

        let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));

        // Fill with recognizable data
        unsafe {
            let k_ptr = k_buf.as_mut_ptr() as *mut f32;
            let v_ptr = v_buf.as_mut_ptr() as *mut f32;
            for i in 0..num_tokens * dim {
                *k_ptr.add(i) = (i / dim + 1) as f32; // pos+1 pattern
                *v_ptr.add(i) = ((i / dim + 1) * 10) as f32;
            }
        }

        let k = Tensor::new(
            Shape::new(vec![1, max_seq, heads, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq, heads, dim]),
            v_buf,
            backend.clone(),
        );
        let mut cache = KVCache::new(k, v, max_seq);
        cache.current_pos = num_tokens;
        cache
    }

    #[test]
    fn test_should_evict() {
        let policy = SlidingWindowPolicy::new(10, 0);
        let mut cache = make_cache_with_data(8);
        assert!(!policy.should_evict(&cache, 0));

        cache.current_pos = 11;
        assert!(policy.should_evict(&cache, 0));
    }

    #[test]
    fn test_should_evict_with_prefix() {
        let policy = SlidingWindowPolicy::new(10, 5);
        let mut cache = make_cache_with_data(14);
        assert!(!policy.should_evict(&cache, 0)); // 14 <= 10+5

        cache.current_pos = 16;
        assert!(policy.should_evict(&cache, 0)); // 16 > 15
    }

    #[test]
    fn test_evict_no_prefix() {
        let policy = SlidingWindowPolicy::new(5, 0);
        let mut cache = make_cache_with_data(10);
        // 10 tokens, window=5, should keep last 5

        policy.evict(&mut cache, 5).unwrap();
        assert_eq!(cache.current_pos, 5);

        // Verify data: position 0 should now be old position 5 (value=6.0)
        let k_data = cache.k_buffer.as_slice::<f32>();
        assert_eq!(k_data[0], 6.0);
        assert_eq!(k_data[4], 7.0);
    }

    #[test]
    fn test_evict_with_protected_prefix() {
        let policy = SlidingWindowPolicy::new(4, 2);
        let mut cache = make_cache_with_data(10);
        // 10 tokens, window=4, prefix=2 → keep 6 total
        // Remove tokens at positions 2..5 (indices 2,3,4,5), keep 0,1 + 6,7,8,9

        policy.evict(&mut cache, 6).unwrap();
        assert_eq!(cache.current_pos, 6);

        // Position 0,1 should be unchanged (protected)
        let k_data = cache.k_buffer.as_slice::<f32>();
        assert_eq!(k_data[0], 1.0); // Original pos 0
        assert_eq!(k_data[4], 2.0); // Original pos 1

        // Position 2 should now be old position 6 (value=7.0)
        assert_eq!(k_data[8], 7.0);
        assert_eq!(k_data[12], 8.0);
    }

    #[test]
    fn test_evict_no_action_needed() {
        let policy = SlidingWindowPolicy::new(20, 0);
        let mut cache = make_cache_with_data(10);

        policy.evict(&mut cache, 20).unwrap();
        assert_eq!(cache.current_pos, 10); // No change
    }

    #[test]
    fn test_name() {
        let policy = SlidingWindowPolicy::new(10, 0);
        assert_eq!(policy.name(), "sliding_window");
    }
}
