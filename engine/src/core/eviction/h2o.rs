use super::EvictionPolicy;
use crate::core::kv_cache::KVCache;
use anyhow::Result;

/// H2O (Heavy-Hitter Oracle) eviction policy — attention score-based KV cache eviction.
///
/// Implements the 3-partition model from the H2O paper:
///   [Protected Prefix] [Heavy Hitters (score-ranked)] [Recent Window (always protected)]
///
/// - **Protected Prefix**: First N tokens (attention sinks / system prompt) are never evicted.
/// - **Recent Window**: Last M tokens are always protected regardless of their scores,
///   ensuring recent context is preserved for coherent generation.
/// - **Heavy Hitters**: Among the remaining (evictable) tokens, the top-K by cumulative
///   attention score are retained; the rest are evicted.
///
/// When `recent_window=0`, the recent partition is empty and all non-prefix tokens
/// compete purely on score (backward-compatible with the previous implementation).
///
/// Reference: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
/// Large Language Models" (Zhang et al., 2023)
pub struct H2OPolicy {
    /// Number of recent tokens always protected from eviction
    recent_window: usize,
    /// Fraction of tokens to keep (0.0 to 1.0), determines heavy hitter budget
    keep_ratio: f32,
    /// Number of prefix tokens to always protect (attention sinks)
    protected_prefix: usize,
}

impl H2OPolicy {
    pub fn new(recent_window: usize, keep_ratio: f32, protected_prefix: usize) -> Self {
        let protected_prefix = protected_prefix.max(4);
        Self {
            recent_window,
            keep_ratio: keep_ratio.clamp(0.0, 1.0),
            protected_prefix,
        }
    }
}

impl EvictionPolicy for H2OPolicy {
    fn should_evict(&self, cache: &KVCache, _mem_available: usize) -> bool {
        // Total tokens to keep = prefix + heavy hitters (ratio-based) + recent window
        let hh_count = ((cache.current_pos as f32) * self.keep_ratio) as usize;
        let total_keep = self.protected_prefix + hh_count + self.recent_window;
        cache.current_pos > total_keep
    }

    fn evict(&self, cache: &mut KVCache, target_len: usize) -> Result<()> {
        let current = cache.current_pos;
        // Compute recent window boundary
        let recent_start = self
            .protected_prefix
            .max(current.saturating_sub(self.recent_window));
        let recent_count = current - recent_start;

        let max_keep =
            ((current as f32) * self.keep_ratio) as usize + self.protected_prefix + recent_count;
        let min_keep = (self.protected_prefix + recent_count + 16).min(max_keep);
        let keep = target_len.clamp(min_keep, max_keep);

        if current <= keep {
            return Ok(());
        }

        // Fallback: without attention scores, remove oldest tokens from the
        // evictable range (prefix..recent_start), preserving both prefix and recent window.
        log::debug!(
            "H2O (fallback): sliding window eviction, removing {} tokens (recent_window={})",
            current - keep,
            self.recent_window
        );

        // Number of tokens to remove from the evictable zone
        let evictable_count = recent_start - self.protected_prefix;
        let desired_remove = current - keep;
        let prune_count = desired_remove.min(evictable_count);

        if prune_count == 0 {
            return Ok(());
        }

        // Shift: move [prefix+prune_count .. current) → [prefix ..)
        // This preserves both the prefix tokens and the recent window tokens.
        let shape = cache.k_buffer.shape().dims();
        let heads = shape[2];
        let dim = shape[3];
        let is_q4 = cache.k_buffer.dtype() == crate::core::buffer::DType::Q4_0;

        let (src_off, dst_off, move_count) = if is_q4 {
            let bpp = heads * dim / crate::core::quant::QK4_0;
            (
                (self.protected_prefix + prune_count) * bpp,
                self.protected_prefix * bpp,
                (current - self.protected_prefix - prune_count) * bpp,
            )
        } else {
            let epp = heads * dim;
            (
                (self.protected_prefix + prune_count) * epp,
                self.protected_prefix * epp,
                (current - self.protected_prefix - prune_count) * epp,
            )
        };

        let backend = cache.k_buffer.backend().clone();
        backend.buffer_shift(&mut cache.k_buffer, src_off, dst_off, move_count)?;
        backend.buffer_shift(&mut cache.v_buffer, src_off, dst_off, move_count)?;

        cache.current_pos -= prune_count;

        Ok(())
    }

    fn evict_with_scores(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: &[f32],
    ) -> Result<()> {
        let current = cache.current_pos;

        // 3-partition boundaries:
        //   [0..prefix)  [prefix..recent_start)  [recent_start..current)
        //    protected       evictable (score)       recent (protected)
        let recent_start = self
            .protected_prefix
            .max(current.saturating_sub(self.recent_window));
        let recent_count = current - recent_start;

        let max_keep =
            ((current as f32) * self.keep_ratio) as usize + self.protected_prefix + recent_count;
        let min_keep = (self.protected_prefix + recent_count + 16).min(max_keep);
        let keep = target_len.clamp(min_keep, max_keep);

        if current <= keep {
            return Ok(());
        }

        // Heavy hitter budget = total keep - prefix - recent
        let hh_budget = keep
            .saturating_sub(self.protected_prefix)
            .saturating_sub(recent_count);

        log::debug!(
            "H2O: score-based eviction, keeping {}/{} tokens (prefix={}, hh={}, recent={})",
            keep,
            current,
            self.protected_prefix,
            hh_budget,
            recent_count,
        );

        // 1. Rank evictable tokens (prefix..recent_start) by importance
        let evictable_start = self.protected_prefix;
        let mut token_scores: Vec<(usize, f32)> = (evictable_start..recent_start)
            .map(|pos| (pos, importance.get(pos).copied().unwrap_or(0.0)))
            .collect();

        // 2. Sort by importance descending (highest first = keep)
        token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 3. Select top hh_budget heavy hitters, then sort by position to preserve order
        let mut hh_positions: Vec<usize> = token_scores
            .iter()
            .take(hh_budget)
            .map(|(pos, _)| *pos)
            .collect();
        hh_positions.sort();

        // 4. Build final keep list: [prefix positions] ++ [heavy hitters] ++ [recent positions]
        //    Prefix is already in place, so we only compact hh + recent after prefix.
        let recent_positions: Vec<usize> = (recent_start..current).collect();

        // 5. Compact the KV cache: [prefix..., heavy hitters in order..., recent in order...]
        let shape = cache.k_buffer.shape().dims();
        let heads = shape[2];
        let dim = shape[3];
        let is_q4 = cache.k_buffer.dtype() == crate::core::buffer::DType::Q4_0;
        let units_per_pos = if is_q4 {
            heads * dim / crate::core::quant::QK4_0
        } else {
            heads * dim
        };

        let backend = cache.k_buffer.backend().clone();
        let mut write_pos = self.protected_prefix;
        for &src_pos in hh_positions.iter().chain(recent_positions.iter()) {
            if src_pos != write_pos {
                let src_off = src_pos * units_per_pos;
                let dst_off = write_pos * units_per_pos;
                backend.buffer_shift(&mut cache.k_buffer, src_off, dst_off, units_per_pos)?;
                backend.buffer_shift(&mut cache.v_buffer, src_off, dst_off, units_per_pos)?;
            }
            write_pos += 1;
        }

        cache.current_pos = self.protected_prefix + hh_positions.len() + recent_positions.len();

        log::debug!("H2O: compacted cache to {} tokens", cache.current_pos);

        Ok(())
    }

    fn name(&self) -> &str {
        "h2o"
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

    // ── should_evict tests ──

    #[test]
    fn test_should_evict() {
        // recent_window=5, keep_ratio=0.5, prefix=4
        // total_keep = 4 + 20*0.5 + 5 = 19 → 20 > 19 = true
        let policy = H2OPolicy::new(5, 0.5, 0);
        let cache = make_cache(20);
        assert!(policy.should_evict(&cache, 0));

        // total_keep = 4 + 10*0.5 + 5 = 14 → 10 > 14 = false
        let cache = make_cache(10);
        assert!(!policy.should_evict(&cache, 0));
    }

    #[test]
    fn test_should_evict_with_recent_window() {
        // recent_window=10, keep_ratio=0.5, prefix=4
        // At pos=20: total_keep = 4 + 10 + 10 = 24 → 20 > 24 = false (recent window absorbs)
        let policy = H2OPolicy::new(10, 0.5, 0);
        let cache = make_cache(20);
        assert!(!policy.should_evict(&cache, 0));

        // At pos=30: total_keep = 4 + 15 + 10 = 29 → 30 > 29 = true
        let cache = make_cache(30);
        assert!(policy.should_evict(&cache, 0));
    }

    #[test]
    fn test_keep_ratio_clamping() {
        let policy = H2OPolicy::new(5, 2.0, 0); // clamped to 1.0
        let cache = make_cache(20);
        // total_keep = 4 + 20 + 5 = 29 → 20 > 29 = false
        assert!(!policy.should_evict(&cache, 0));

        let policy_neg = H2OPolicy::new(5, -1.0, 0); // clamped to 0.0
        let cache2 = make_cache(20);
        // total_keep = 4 + 0 + 5 = 9 → 20 > 9 = true
        assert!(policy_neg.should_evict(&cache2, 0));
    }

    // ── evict (fallback) tests ──

    #[test]
    fn test_evict_falls_back_to_sliding() {
        let _ = env_logger::try_init();
        // recent_window=0, keep_ratio=0.5, prefix=4
        let policy = H2OPolicy::new(0, 0.5, 0);
        let mut cache = make_cache(20);
        // max_keep = 20*0.5 + 4 + 0 = 14
        policy.evict(&mut cache, 10).unwrap();
        assert_eq!(cache.current_pos, 14);
    }

    #[test]
    fn test_evict_with_prefix() {
        // recent_window=0, keep_ratio=0.5, prefix=4 (3→clamped to 4)
        let policy = H2OPolicy::new(0, 0.5, 3);
        let mut cache = make_cache(20);
        // max_keep = 20*0.5 + 4 + 0 = 14
        policy.evict(&mut cache, 13).unwrap();
        assert_eq!(cache.current_pos, 14);
    }

    #[test]
    fn test_evict_below_threshold_noop() {
        let policy = H2OPolicy::new(0, 0.9, 0);
        let mut cache = make_cache(10);
        policy.evict(&mut cache, 10).unwrap();
        assert_eq!(cache.current_pos, 10);
    }

    #[test]
    fn test_evict_fallback_with_recent_window() {
        let _ = env_logger::try_init();
        // recent_window=5, keep_ratio=0.3, prefix=4
        // current=20, recent_start=max(4, 20-5)=15, recent_count=5
        // max_keep = 20*0.3 + 4 + 5 = 15
        // evictable_count = 15-4 = 11, desired_remove = 20-15 = 5
        let policy = H2OPolicy::new(5, 0.3, 0);
        let mut cache = make_cache(20);

        // Write markers to verify recent tokens survive
        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32);
        }

        policy.evict(&mut cache, 10).unwrap();
        assert_eq!(cache.current_pos, 15);

        // Prefix (0-3) should be intact
        for i in 0..4 {
            let (k, _) = read_marker(&cache, i);
            assert_eq!(k, (i + 1) as f32);
        }

        // Recent tokens (originally at positions 15-19) should be at the end
        // After removing 5 tokens from evictable zone [4..15), positions shift:
        // new pos 4..10 = old 9..15 (surviving evictable)
        // new pos 10..15 = old 15..20 (recent window)
        let (k, _) = read_marker(&cache, cache.current_pos - 1);
        assert_eq!(k, 20.0); // last recent token
    }

    // ── evict_with_scores tests ──

    #[test]
    fn test_recent_window_protection() {
        // Recent tokens with score=0 must still be kept
        let policy = H2OPolicy::new(5, 0.3, 0); // prefix=4, recent=5
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32);
        }

        // Give high scores only to early tokens, zero to recent
        let mut importance = vec![0.0f32; 100];
        for i in 4..15 {
            importance[i] = (20 - i) as f32;
        }
        // positions 15-19 have importance 0.0 — but recent window protects them

        // current=20, recent_start=15, recent_count=5
        // max_keep = 20*0.3+4+5 = 15, min_keep = (4+5+16).min(15)=15
        // hh_budget = 15-4-5 = 6
        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 15); // 4 prefix + 6 hh + 5 recent

        // Verify recent tokens are at the end (originally pos 15-19, marker 16-20)
        for i in 0..5 {
            let (k, _) = read_marker(&cache, cache.current_pos - 5 + i);
            assert_eq!(k, (16 + i) as f32);
        }
    }

    #[test]
    fn test_recent_window_zero_backward_compat() {
        // recent_window=0 should behave exactly like old implementation
        let policy = H2OPolicy::new(0, 0.3, 0); // prefix=4, recent=0
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32);
        }

        let mut importance = vec![0.0f32; 100];
        importance[10] = 10.0;
        importance[15] = 8.0;
        importance[19] = 9.0;
        for i in 4..20 {
            if importance[i] == 0.0 {
                importance[i] = 0.1;
            }
        }

        // max_keep = 20*0.3+4+0 = 10, hh_budget = 10-4-0 = 6
        policy
            .evict_with_scores(&mut cache, 8, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 10);

        // Prefix intact
        for i in 0..4 {
            let (k, v) = read_marker(&cache, i);
            assert_eq!(k, (i + 1) as f32);
            assert_eq!(v, (i + 1) as f32);
        }
    }

    #[test]
    fn test_recent_window_exceeds_budget() {
        // When recent_window is large, hh_budget shrinks to 0
        let policy = H2OPolicy::new(12, 0.3, 0); // prefix=4, recent=12
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32);
        }

        let mut importance = vec![0.0f32; 100];
        for i in 4..20 {
            importance[i] = (20 - i) as f32;
        }

        // recent_start = max(4, 20-12) = 8, recent_count = 12
        // max_keep = 20*0.3+4+12 = 22 → current(20) <= 22, no eviction
        policy
            .evict_with_scores(&mut cache, 8, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 20); // no eviction happened
    }

    #[test]
    fn test_recent_window_covers_all_evictable() {
        // recent_window >= current - prefix → no evictable tokens
        let policy = H2OPolicy::new(20, 0.1, 0); // prefix=4, recent=20
        let mut cache = make_cache(20);

        let importance = vec![0.0f32; 100];

        // recent_start = max(4, 20-20) = 4, recent_count = 16
        // max_keep = 20*0.1+4+16 = 22 → no eviction
        policy
            .evict_with_scores(&mut cache, 5, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 20);
    }

    #[test]
    fn test_evict_with_scores_preserves_prefix() {
        let policy = H2OPolicy::new(0, 0.2, 5); // prefix=5, recent=0, keep_ratio=0.2
        let mut cache = make_cache(30);

        let mut importance = vec![0.0f32; 100];
        for i in 5..30 {
            importance[i] = (30 - i) as f32;
        }

        // max_keep = 30*0.2+5+0 = 11, hh_budget = 11-5-0 = 6
        policy
            .evict_with_scores(&mut cache, 8, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 11);
    }

    #[test]
    fn test_recent_window_order_preservation() {
        // Verify final layout is [prefix][hh in position order][recent in position order]
        let policy = H2OPolicy::new(4, 0.3, 0); // prefix=4, recent=4
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i * 100) as f32);
        }

        // Evictable range: positions 4..16 (prefix=4, recent_start=16)
        let mut importance = vec![0.0f32; 100];
        importance[15] = 10.0;
        importance[5] = 9.0;
        importance[10] = 8.0;
        importance[7] = 7.0;
        importance[12] = 6.0;
        importance[6] = 5.0;
        for i in 4..16 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        // max_keep = 20*0.3+4+4 = 14, hh_budget = 14-4-4 = 6
        policy
            .evict_with_scores(&mut cache, 8, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 14); // 4 prefix + 6 hh + 4 recent

        // Markers should be in ascending original-position order throughout
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

        // Verify last 4 are the recent tokens (originally 16,17,18,19)
        for i in 0..4 {
            let (k, _) = read_marker(&cache, cache.current_pos - 4 + i);
            assert_eq!(k, ((16 + i) * 100) as f32);
        }
    }

    #[test]
    fn test_evict_with_scores_fallback() {
        let policy = H2OPolicy::new(0, 0.5, 0);
        let mut cache = make_cache(20);
        policy.evict(&mut cache, 10).unwrap();
        assert!(cache.current_pos < 20);
    }

    #[test]
    fn test_name() {
        assert_eq!(H2OPolicy::new(5, 0.5, 0).name(), "h2o");
    }
}
