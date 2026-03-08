use super::EvictionPolicy;
use crate::core::kv_cache::KVCache;
use anyhow::Result;

/// H2O (Heavy-Hitter Oracle) eviction policy — attention score-based KV cache eviction.
///
/// Implements the 3-partition model from the H2O paper:
///   [Protected Prefix] [Heavy Hitters (score-ranked)] [Recent Window]
///
/// - **Protected Prefix**: First N tokens (attention sinks / system prompt) are never evicted.
/// - **Heavy Hitters**: Tokens with highest cumulative attention scores.
/// - **Recent Window**: Most recent M tokens, always protected.
///
/// Budget allocation (following the paper): after reserving prefix, the remaining
/// `keep` slots are split between HH and Recent by `keep_ratio`:
///   - `hh_budget = available * keep_ratio`
///   - `recent_budget = available - hh_budget`
///
/// With `keep_ratio=0.5` (default), this produces the paper's recommended 50:50 split.
///
/// Reference: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
/// Large Language Models" (Zhang et al., 2023)
pub struct H2OPolicy {
    /// Fraction of available budget allocated to heavy hitters (0.0 to 1.0).
    /// Default 0.5 = paper's 50:50 HH:Recent split.
    keep_ratio: f32,
    /// Number of prefix tokens to always protect (attention sinks)
    protected_prefix: usize,
}

impl H2OPolicy {
    pub fn new(_recent_window: usize, keep_ratio: f32, protected_prefix: usize) -> Self {
        let protected_prefix = protected_prefix.max(4);
        Self {
            keep_ratio: keep_ratio.clamp(0.0, 1.0),
            protected_prefix,
        }
    }
}

impl EvictionPolicy for H2OPolicy {
    fn should_evict(&self, _cache: &KVCache, _mem_available: usize) -> bool {
        // H2O is signal-driven: eviction is triggered exclusively by external
        // resilience signals, never by automatic cache/memory checks.
        // Score accumulation happens every token, but eviction decisions
        // come from the resilience manager.
        false
    }

    fn evict(&self, cache: &mut KVCache, target_len: usize) -> Result<()> {
        let current = cache.current_pos;
        let keep = target_len.max(self.protected_prefix + 2);

        if current <= keep {
            return Ok(());
        }

        // Fallback: without attention scores, keep prefix + most recent tokens
        let available = keep.saturating_sub(self.protected_prefix);
        let recent_budget = available; // All budget goes to recent in fallback mode
        let actual_recent = recent_budget.min(current - self.protected_prefix);
        let prune_count = current - self.protected_prefix - actual_recent;

        if prune_count == 0 {
            return Ok(());
        }

        log::debug!(
            "H2O (fallback): sliding window eviction, removing {} tokens (keep={})",
            prune_count,
            keep,
        );

        let shape = cache.k_buffer.shape().dims();
        let heads = shape[2];
        let dim = shape[3];
        let is_q4 = cache.k_buffer.dtype() == crate::core::buffer::DType::Q4_0;

        let (src_off, dst_off, move_count) = if is_q4 {
            let bpp = heads * dim / crate::core::quant::QK4_0;
            (
                (self.protected_prefix + prune_count) * bpp,
                self.protected_prefix * bpp,
                actual_recent * bpp,
            )
        } else {
            let epp = heads * dim;
            (
                (self.protected_prefix + prune_count) * epp,
                self.protected_prefix * epp,
                actual_recent * epp,
            )
        };

        let backend = cache.k_buffer.backend().clone();
        backend.buffer_shift(&mut cache.k_buffer, src_off, dst_off, move_count)?;
        backend.buffer_shift(&mut cache.v_buffer, src_off, dst_off, move_count)?;

        cache.current_pos = self.protected_prefix + actual_recent;

        Ok(())
    }

    fn evict_with_scores(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: &[f32],
    ) -> Result<()> {
        let current = cache.current_pos;
        let keep = target_len.max(self.protected_prefix + 2);

        if current <= keep {
            return Ok(());
        }

        // Budget allocation (paper's approach):
        // available = keep - prefix
        // hh_budget = available * keep_ratio  (default 0.5 = 50:50)
        // recent_budget = available - hh_budget
        let available = keep.saturating_sub(self.protected_prefix);
        let hh_budget = (available as f32 * self.keep_ratio) as usize;
        let recent_budget = available - hh_budget;
        let actual_recent = recent_budget.min(current - self.protected_prefix);

        // Recent window boundary
        let recent_start = current
            .saturating_sub(actual_recent)
            .max(self.protected_prefix);
        let actual_recent = current - recent_start;

        log::debug!(
            "H2O: score-based eviction, keeping {}/{} tokens (prefix={}, hh={}, recent={})",
            keep,
            current,
            self.protected_prefix,
            hh_budget,
            actual_recent,
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
    fn test_should_evict_always_false() {
        let policy = H2OPolicy::new(0, 0.5, 0);
        assert!(!policy.should_evict(&make_cache(20), 0));
        assert!(!policy.should_evict(&make_cache(1000), 0));
        assert!(!policy.should_evict(&make_cache(1), 0));
    }

    // ── evict (fallback) tests ──

    #[test]
    fn test_evict_fallback_keeps_recent() {
        let _ = env_logger::try_init();
        // keep_ratio=0.5, prefix=4(clamped)
        let policy = H2OPolicy::new(0, 0.5, 0);
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32);
        }

        // target_len=10 → keep=10, available=10-4=6, all recent
        // prune = 20-4-6 = 10
        policy.evict(&mut cache, 10).unwrap();
        assert_eq!(cache.current_pos, 10);

        // Prefix intact
        for i in 0..4 {
            let (k, _) = read_marker(&cache, i);
            assert_eq!(k, (i + 1) as f32);
        }
        // Most recent tokens at the end
        let (k, _) = read_marker(&cache, 9);
        assert_eq!(k, 20.0);
    }

    #[test]
    fn test_evict_below_threshold_noop() {
        let policy = H2OPolicy::new(0, 0.5, 0);
        let mut cache = make_cache(10);
        policy.evict(&mut cache, 10).unwrap();
        assert_eq!(cache.current_pos, 10);
    }

    // ── evict_with_scores tests ──

    #[test]
    fn test_budget_split_50_50() {
        // keep_ratio=0.5 → 50:50 HH:Recent split (paper default)
        let policy = H2OPolicy::new(0, 0.5, 0); // prefix=4(clamped)
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32);
        }

        // High scores on specific tokens
        let mut importance = vec![0.0f32; 100];
        importance[5] = 10.0;
        importance[8] = 9.0;
        importance[12] = 8.0;
        for i in 4..20 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        // target_len=10, keep=10, available=10-4=6
        // hh_budget = 6*0.5 = 3, recent_budget = 3
        // recent_start = max(4, 20-3) = 17
        // evictable: positions 4..17 (13 tokens), keep top-3 by score
        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 10); // 4 prefix + 3 hh + 3 recent
    }

    #[test]
    fn test_high_hh_ratio() {
        // keep_ratio=0.8 → 80% HH, 20% Recent
        let policy = H2OPolicy::new(0, 0.8, 0); // prefix=4(clamped)
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i + 1) as f32);
        }

        let mut importance = vec![0.0f32; 100];
        for i in 4..20 {
            importance[i] = (20 - i) as f32; // higher scores for earlier tokens
        }

        // target_len=14, keep=14, available=14-4=10
        // hh_budget = 10*0.8 = 8, recent_budget = 2
        // recent_start = max(4, 20-2) = 18
        policy
            .evict_with_scores(&mut cache, 14, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 14); // 4 prefix + 8 hh + 2 recent
    }

    #[test]
    fn test_evict_preserves_prefix() {
        let policy = H2OPolicy::new(0, 0.5, 5); // prefix=5
        let mut cache = make_cache(30);

        for i in 0..30 {
            write_marker(&mut cache, i, (i + 1) as f32);
        }

        let mut importance = vec![0.0f32; 100];
        for i in 5..30 {
            importance[i] = (30 - i) as f32;
        }

        // target_len=15, keep=15, available=15-5=10
        // hh_budget=5, recent=5
        policy
            .evict_with_scores(&mut cache, 15, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 15);

        // Prefix intact
        for i in 0..5 {
            let (k, v) = read_marker(&cache, i);
            assert_eq!(k, (i + 1) as f32);
            assert_eq!(v, (i + 1) as f32);
        }
    }

    #[test]
    fn test_no_eviction_when_below_target() {
        let policy = H2OPolicy::new(0, 0.5, 0);
        let mut cache = make_cache(10);
        let importance = vec![1.0f32; 100];

        policy
            .evict_with_scores(&mut cache, 15, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 10); // no eviction
    }

    #[test]
    fn test_order_preservation() {
        // Verify final layout is [prefix][hh in position order][recent in position order]
        let policy = H2OPolicy::new(0, 0.5, 0); // prefix=4(clamped)
        let mut cache = make_cache(20);

        for i in 0..20 {
            write_marker(&mut cache, i, (i * 100) as f32);
        }

        // Give distinctive scores to specific tokens
        let mut importance = vec![0.0f32; 100];
        importance[5] = 10.0;
        importance[10] = 9.0;
        importance[7] = 8.0;
        for i in 4..20 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        // target=10, keep=10, available=6, hh=3, recent=3
        // recent_start = max(4, 20-3) = 17
        // Top 3 HH from evictable [4..17]: pos 5(10.0), 10(9.0), 7(8.0)
        // Final layout: [0,1,2,3] [5,7,10] [17,18,19]
        policy
            .evict_with_scores(&mut cache, 10, &importance)
            .unwrap();

        assert_eq!(cache.current_pos, 10);

        // All markers should be in ascending position order
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

        // Last 3 should be recent tokens (originally 17, 18, 19)
        for i in 0..3 {
            let (k, _) = read_marker(&cache, cache.current_pos - 3 + i);
            assert_eq!(k, ((17 + i) * 100) as f32);
        }
    }

    #[test]
    fn test_evict_fallback_works() {
        let policy = H2OPolicy::new(0, 0.5, 0);
        let mut cache = make_cache(20);
        policy.evict(&mut cache, 10).unwrap();
        assert!(cache.current_pos < 20);
    }

    #[test]
    fn test_name() {
        assert_eq!(H2OPolicy::new(0, 0.5, 0).name(), "h2o");
    }
}
