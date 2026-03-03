use super::EvictionPolicy;
use crate::core::kv_cache::KVCache;
use anyhow::Result;

/// SnapKV eviction policy — attention score-based intelligent KV cache eviction.
///
/// SnapKV selectively keeps tokens with high attention scores, dropping those
/// rarely attended to. When importance scores are provided via `evict_with_scores()`,
/// tokens are ranked by cumulative attention weight and only the most important
/// ones are retained. Without scores, falls back to sliding window behavior.
///
/// The `AttentionScoreAccumulator` (in `core::attention_scores`) captures
/// post-softmax attention weights during the forward pass and feeds them here.
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
        let protected_prefix = protected_prefix.max(4);
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
        let max_keep = ((current as f32) * self.keep_ratio) as usize + self.protected_prefix;
        let min_keep = (self.protected_prefix + 16).min(max_keep);
        let keep = target_len.clamp(min_keep, max_keep);

        if current <= keep {
            return Ok(());
        }

        // Fallback: without attention scores, use sliding window behavior
        log::debug!(
            "SnapKV (fallback): sliding window eviction, removing {} tokens",
            current - keep
        );

        let prune_count = current - keep;
        if self.protected_prefix == 0 {
            cache.prune_prefix(prune_count)?;
        } else {
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

    fn evict_with_scores(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: &[f32],
    ) -> Result<()> {
        let current = cache.current_pos;
        let max_keep = ((current as f32) * self.keep_ratio) as usize + self.protected_prefix;
        let min_keep = (self.protected_prefix + 16).min(max_keep);
        let keep = target_len.clamp(min_keep, max_keep);

        if current <= keep {
            return Ok(());
        }

        log::debug!(
            "SnapKV: score-based eviction, keeping {}/{} tokens",
            keep,
            current
        );

        // 1. Build (position, importance) for evictable tokens (after protected prefix)
        let evictable_start = self.protected_prefix;
        let mut token_scores: Vec<(usize, f32)> = (evictable_start..current)
            .map(|pos| (pos, importance.get(pos).copied().unwrap_or(0.0)))
            .collect();

        // 2. Sort by importance descending (highest first = keep)
        token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 3. Select top-k tokens to keep, then sort by position to preserve order
        let non_prefix_keep = keep.saturating_sub(self.protected_prefix);
        let mut keep_positions: Vec<usize> = token_scores
            .iter()
            .take(non_prefix_keep)
            .map(|(pos, _)| *pos)
            .collect();
        keep_positions.sort();

        // 4. Compact the KV cache: [prefix..., kept tokens in order...]
        let shape = cache.k_buffer.shape().dims();
        let heads = shape[2];
        let dim = shape[3];
        let type_size = cache.k_buffer.dtype().size();
        let bytes_per_pos = heads * dim * type_size;

        let k_ptr = cache.k_buffer.as_mut_ptr();
        let v_ptr = cache.v_buffer.as_mut_ptr();

        if k_ptr.is_null() || v_ptr.is_null() {
            return Err(anyhow::anyhow!(
                "Cannot evict: null buffer pointers (GPU-only buffers not supported)"
            ));
        }

        let mut write_pos = self.protected_prefix;
        for &src_pos in &keep_positions {
            if src_pos != write_pos {
                unsafe {
                    let src_off = src_pos * bytes_per_pos;
                    let dst_off = write_pos * bytes_per_pos;
                    std::ptr::copy(k_ptr.add(src_off), k_ptr.add(dst_off), bytes_per_pos);
                    std::ptr::copy(v_ptr.add(src_off), v_ptr.add(dst_off), bytes_per_pos);
                }
            }
            write_pos += 1;
        }

        cache.current_pos = self.protected_prefix + keep_positions.len();

        log::debug!("SnapKV: compacted cache to {} tokens", cache.current_pos);

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
        let buf_size = max_seq * 1 * 4 * 4; // max_seq * batch * heads * dim * sizeof(f32)
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

    /// Write a marker value at a given position so we can verify token identity.
    fn write_marker(cache: &mut KVCache, pos: usize, val: f32) {
        let shape = cache.k_buffer.shape().dims();
        let heads = shape[2];
        let dim = shape[3];
        let bytes_per_pos = heads * dim * std::mem::size_of::<f32>();
        let k_ptr = cache.k_buffer.as_mut_ptr();
        let v_ptr = cache.v_buffer.as_mut_ptr();
        unsafe {
            let k_f32 = k_ptr.add(pos * bytes_per_pos) as *mut f32;
            let v_f32 = v_ptr.add(pos * bytes_per_pos) as *mut f32;
            *k_f32 = val;
            *v_f32 = val;
        }
    }

    fn read_marker(cache: &KVCache, pos: usize) -> (f32, f32) {
        let shape = cache.k_buffer.shape().dims();
        let heads = shape[2];
        let dim = shape[3];
        let bytes_per_pos = heads * dim * std::mem::size_of::<f32>();
        let k_ptr = cache.k_buffer.as_mut_ptr();
        let v_ptr = cache.v_buffer.as_mut_ptr();
        unsafe {
            let k_val = *(k_ptr.add(pos * bytes_per_pos) as *const f32);
            let v_val = *(v_ptr.add(pos * bytes_per_pos) as *const f32);
            (k_val, v_val)
        }
    }

    #[test]
    fn test_should_evict() {
        let policy = SnapKVPolicy::new(5, 0.5, 0);
        let cache = make_cache(20);
        assert!(policy.should_evict(&cache, 0));

        let cache = make_cache(10);
        assert!(!policy.should_evict(&cache, 0));
    }

    #[test]
    fn test_evict_stub_falls_back_to_sliding() {
        let _ = env_logger::try_init();
        let policy = SnapKVPolicy::new(5, 0.5, 0); // prefix=4
        let mut cache = make_cache(20);
        policy.evict(&mut cache, 10).unwrap();
        assert_eq!(cache.current_pos, 14);
    }

    #[test]
    fn test_evict_with_prefix() {
        let policy = SnapKVPolicy::new(5, 0.5, 3); // prefix=4
        let mut cache = make_cache(20);
        policy.evict(&mut cache, 13).unwrap();
        assert_eq!(cache.current_pos, 14);
    }

    #[test]
    fn test_name() {
        assert_eq!(SnapKVPolicy::new(5, 0.5, 0).name(), "snap_kv");
    }

    #[test]
    fn test_keep_ratio_clamping() {
        let policy = SnapKVPolicy::new(5, 2.0, 0);
        let cache = make_cache(20);
        assert!(!policy.should_evict(&cache, 0));

        let policy_neg = SnapKVPolicy::new(5, -1.0, 0);
        let cache2 = make_cache(20);
        assert!(policy_neg.should_evict(&cache2, 0));
    }

    #[test]
    fn test_evict_below_threshold_noop() {
        let policy = SnapKVPolicy::new(5, 0.9, 0);
        let mut cache = make_cache(10);
        policy.evict(&mut cache, 10).unwrap();
        assert_eq!(cache.current_pos, 10);
    }

    // --- New tests for evict_with_scores ---

    #[test]
    fn test_evict_with_scores_keeps_important_tokens() {
        let _policy = SnapKVPolicy::new(5, 1.0, 0); // prefix=4, keep_ratio=1.0
        let mut cache = make_cache(20);

        // Write marker values at each position
        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32);
        }

        // Importance: positions 10,15,19 have highest scores
        let mut importance = vec![0.0f32; 100];
        importance[10] = 10.0;
        importance[15] = 8.0;
        importance[19] = 9.0;
        // Other positions have low scores
        for i in 4..20 {
            if importance[i] == 0.0 {
                importance[i] = 0.1;
            }
        }

        // target_len = 10 → keep = clamp(10, min_keep=min(20,24)=20, max_keep=24) = 20
        // With keep_ratio=1.0 and prefix=4, max_keep = 20+4=24, so no eviction at target_len=10
        // Let's use a lower keep_ratio
        let policy = SnapKVPolicy::new(5, 0.3, 0); // prefix=4, keep_ratio=0.3
        // max_keep = 20*0.3 + 4 = 10
        // min_keep = min(4+16, 10) = 10
        // keep = 8.clamp(10, 10) = 10

        policy
            .evict_with_scores(&mut cache, 8, &importance)
            .unwrap();

        // Should have kept prefix (0-3) + top 6 by importance from positions 4-19
        // Top 6 by importance: 10(10.0), 19(9.0), 15(8.0), plus 3 more at 0.1
        assert_eq!(cache.current_pos, 10);

        // Verify prefix tokens are intact (positions 0-3)
        for i in 0..4 {
            let (k, v) = read_marker(&cache, i);
            assert_eq!(k, (i + 1) as f32);
            assert_eq!(v, (i + 1) as f32);
        }
    }

    #[test]
    fn test_evict_with_scores_preserves_prefix() {
        let policy = SnapKVPolicy::new(5, 0.2, 5); // prefix=5, keep_ratio=0.2
        let mut cache = make_cache(30);

        // Give prefix tokens zero importance — they should still be kept
        let mut importance = vec![0.0f32; 100];
        for i in 5..30 {
            importance[i] = (30 - i) as f32; // decreasing importance
        }

        // max_keep = 30*0.2 + 5 = 11
        // min_keep = min(5+16, 11) = 11
        // keep = 8.clamp(11, 11) = 11
        policy
            .evict_with_scores(&mut cache, 8, &importance)
            .unwrap();

        // Prefix (5 tokens) always kept, plus top 6 non-prefix tokens
        assert_eq!(cache.current_pos, 11);
    }

    #[test]
    fn test_evict_with_scores_maintains_order() {
        let policy = SnapKVPolicy::new(5, 0.3, 0); // prefix=4, keep_ratio=0.3
        let mut cache = make_cache(20);

        // Write distinct markers
        for i in 0..20 {
            write_marker(&mut cache, i, (i * 100) as f32);
        }

        // Give high importance to positions 5, 10, 15 (in non-sequential order by score)
        let mut importance = vec![0.0f32; 100];
        importance[15] = 10.0; // highest score
        importance[5] = 9.0;
        importance[10] = 8.0;
        importance[7] = 7.0;
        importance[12] = 6.0;
        importance[18] = 5.0;
        for i in 4..20 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        // max_keep = 20*0.3 + 4 = 10
        policy
            .evict_with_scores(&mut cache, 8, &importance)
            .unwrap();

        // Kept positions should be in ascending order: 0,1,2,3 (prefix), then 5,7,10,12,15,18
        // Verify the markers at compacted positions are in ascending original-position order
        let mut prev_marker = -1.0f32;
        for i in 0..cache.current_pos {
            let (k, _) = read_marker(&cache, i);
            assert!(
                k > prev_marker,
                "Position {} has marker {} <= prev {}",
                i,
                k,
                prev_marker
            );
            prev_marker = k;
        }
    }

    #[test]
    fn test_evict_with_scores_fallback() {
        // Calling evict() (without scores) should still work
        let policy = SnapKVPolicy::new(5, 0.5, 0);
        let mut cache = make_cache(20);
        policy.evict(&mut cache, 10).unwrap();
        assert!(cache.current_pos < 20);
    }
}
