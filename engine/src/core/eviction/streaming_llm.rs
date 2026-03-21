use super::EvictionPolicy;
use crate::core::kv_cache::KVCache;
use anyhow::Result;

/// StreamingLLM eviction policy (Xiao et al., ICLR 2024).
///
/// Maintains a fixed two-region structure:
/// ```text
/// [Sink Tokens (S)] [Recent Window (W)]
/// ```
///
/// Unlike `SlidingWindowPolicy` which respects `target_len`, this policy
/// always compacts to exactly `S + W` tokens when eviction triggers.
/// All tokens between the sink region and the recent window are removed at once.
///
/// ## Example
/// ```text
/// sink_size=4, window_size=6, current_pos=15
///
/// Before:
/// [S0][S1][S2][S3][T4][T5][T6][T7][T8][T9][T10][T11][T12][T13][T14]
///  ──── sink ────   ──────── gap (removed) ────────   ── recent(6) ──
///
/// After (current_pos=10):
/// [S0][S1][S2][S3][T9][T10][T11][T12][T13][T14][_][_][_][_][_]
///  ──── sink ────   ────── recent window ──────
/// ```
pub struct StreamingLLMPolicy {
    sink_size: usize,
    window_size: usize,
}

impl StreamingLLMPolicy {
    pub fn new(sink_size: usize, window_size: usize) -> Self {
        Self {
            sink_size: sink_size.max(1),
            window_size: window_size.max(1),
        }
    }

    /// Total capacity: sink + window.
    fn keep_size(&self) -> usize {
        self.sink_size + self.window_size
    }
}

impl EvictionPolicy for StreamingLLMPolicy {
    fn should_evict(&self, cache: &KVCache, _mem_available: usize) -> bool {
        cache.current_pos > self.keep_size()
    }

    fn evict(&self, cache: &mut KVCache, _target_len: usize) -> Result<()> {
        let current = cache.current_pos;
        let keep = self.keep_size();

        if current <= keep {
            return Ok(());
        }

        // Recent window starts at (current - window_size)
        let recent_start = current - self.window_size;

        // Move recent window right after sink region
        cache.shift_positions(recent_start, self.sink_size, self.window_size)?;
        cache.current_pos = keep;

        Ok(())
    }

    fn name(&self) -> &str {
        "streaming_llm"
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
        let v = Tensor::new(Shape::new(vec![1, max_seq, heads, dim]), v_buf, backend);
        let mut cache = KVCache::new(k, v, max_seq);
        cache.current_pos = num_tokens;
        cache
    }

    #[test]
    fn test_should_evict() {
        let policy = StreamingLLMPolicy::new(4, 6); // keep = 10
        let mut cache = make_cache_with_data(10);
        assert!(!policy.should_evict(&cache, 0)); // 10 <= 10

        cache.current_pos = 11;
        assert!(policy.should_evict(&cache, 0)); // 11 > 10
    }

    #[test]
    fn test_evict_basic() {
        // sink=4, window=6, current=15 → keep=10
        // Sink: [1,2,3,4], Recent: [10,11,12,13,14,15]
        // After: [1,2,3,4,10,11,12,13,14,15]
        let policy = StreamingLLMPolicy::new(4, 6);
        let mut cache = make_cache_with_data(15);

        policy.evict(&mut cache, 0).unwrap();
        assert_eq!(cache.current_pos, 10);

        let k_data = cache.k_buffer.as_slice::<f32>();
        let dim = 4;
        // Sink preserved: positions 0-3
        assert_eq!(k_data[0 * dim], 1.0); // pos 0
        assert_eq!(k_data[3 * dim], 4.0); // pos 3
        // Recent window moved to pos 4: original pos 9 (value=10.0)
        assert_eq!(k_data[4 * dim], 10.0);
        // Last token: original pos 14 (value=15.0)
        assert_eq!(k_data[9 * dim], 15.0);
    }

    #[test]
    fn test_evict_ignores_target_len() {
        // StreamingLLM always compacts to sink+window regardless of target_len
        let policy = StreamingLLMPolicy::new(4, 6);
        let mut cache = make_cache_with_data(20);

        // target_len=18 would keep 18 tokens in sliding, but streaming compacts to 10
        policy.evict(&mut cache, 18).unwrap();
        assert_eq!(cache.current_pos, 10);
    }

    #[test]
    fn test_evict_no_action_when_within_budget() {
        let policy = StreamingLLMPolicy::new(4, 6);
        let mut cache = make_cache_with_data(8);

        policy.evict(&mut cache, 0).unwrap();
        assert_eq!(cache.current_pos, 8); // No change
    }

    #[test]
    fn test_evict_exactly_at_budget() {
        let policy = StreamingLLMPolicy::new(4, 6);
        let mut cache = make_cache_with_data(10);

        policy.evict(&mut cache, 0).unwrap();
        assert_eq!(cache.current_pos, 10); // No change (10 <= 10)
    }

    #[test]
    fn test_evict_one_over_budget() {
        // sink=4, window=6, current=11 → remove 1 token (pos 4)
        let policy = StreamingLLMPolicy::new(4, 6);
        let mut cache = make_cache_with_data(11);

        policy.evict(&mut cache, 0).unwrap();
        assert_eq!(cache.current_pos, 10);

        let k_data = cache.k_buffer.as_slice::<f32>();
        let dim = 4;
        // Sink: pos 0-3 preserved
        assert_eq!(k_data[0 * dim], 1.0);
        assert_eq!(k_data[3 * dim], 4.0);
        // Recent: original positions 5-10 (values 6-11) moved to pos 4-9
        assert_eq!(k_data[4 * dim], 6.0); // original pos 5
        assert_eq!(k_data[9 * dim], 11.0); // original pos 10
    }

    #[test]
    fn test_name() {
        let policy = StreamingLLMPolicy::new(4, 6);
        assert_eq!(policy.name(), "streaming_llm");
    }

    #[test]
    fn test_differs_from_sliding() {
        // Key difference: with protected_prefix=4, window=6, current=20
        // Sliding with target_len=15: keeps 15 tokens (removes 5 oldest after prefix)
        // StreamingLLM: always compacts to 10 regardless of target_len

        let streaming = StreamingLLMPolicy::new(4, 6);
        let mut cache_s = make_cache_with_data(20);
        streaming.evict(&mut cache_s, 15).unwrap();
        assert_eq!(cache_s.current_pos, 10); // StreamingLLM: always 10

        use crate::core::eviction::SlidingWindowPolicy;
        let sliding = SlidingWindowPolicy::new(6, 4);
        let mut cache_sl = make_cache_with_data(20);
        sliding.evict(&mut cache_sl, 15).unwrap();
        // Sliding: max_keep = 6+4=10, min_keep = min(4+16,10)=10, keep = clamp(15,10,10) = 10
        // In this case they happen to match because window+prefix is small.
        // But with larger window, the behavior differs.

        // Better test: window=20 so sliding has more room
        let sliding_big = SlidingWindowPolicy::new(20, 4);
        let mut cache_sl2 = make_cache_with_data(30);
        sliding_big.evict(&mut cache_sl2, 20).unwrap();
        // Sliding: max_keep=24, min_keep=min(20,24)=20, keep=clamp(20,20,24)=20
        assert_eq!(cache_sl2.current_pos, 20);

        let streaming_small = StreamingLLMPolicy::new(4, 6);
        let mut cache_s2 = make_cache_with_data(30);
        streaming_small.evict(&mut cache_s2, 20).unwrap();
        // StreamingLLM: always 10
        assert_eq!(cache_s2.current_pos, 10);
    }
}
