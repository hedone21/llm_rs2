//! D2O (Dynamic Discriminative Operations) handler — eviction with compensation merging.
//!
//! Implements the D2O paper (Wan et al., 2024): H2O-style 3-partition eviction
//! with token merging compensation. Evicted tokens are matched to their nearest
//! retained neighbor by cosine similarity; if similarity exceeds an EMA threshold,
//! the evicted token is merged (weighted average) rather than permanently deleted.
//!
//! Phase 1: Uniform per-layer budget (layer-level variance allocation deferred).
//! F32 KV cache only.

use super::{ActionResult, CachePressureHandler, HandlerContext};
use crate::core::kv_cache::KVCache;
use anyhow::Result;
use std::sync::Mutex;

// ── Configuration ────────────────────────────────────────────────

/// D2O configuration parameters.
pub struct D2OConfig {
    /// Fraction of available budget allocated to heavy hitters (0.0–1.0).
    /// D2O paper recommends N:M = 3:1, i.e. keep_ratio = 0.75.
    pub keep_ratio: f32,
    /// Number of prefix tokens to always protect (attention sinks).
    pub protected_prefix: usize,
    /// Target cache ratio: keep this fraction of current_pos after eviction.
    /// E.g. 0.5 = keep 50% of tokens.
    pub target_ratio: f32,
    /// EMA smoothing constant for similarity threshold (0.0–1.0).
    /// Higher = more sensitive to current similarities. Paper default: 0.7.
    pub beta: f32,
    /// Stability constant for merge weight formula.
    /// w_retain = e / (exp(sim) + e), w_evict = exp(sim) / (exp(sim) + e).
    pub merge_e: f32,
}

impl Default for D2OConfig {
    fn default() -> Self {
        Self {
            keep_ratio: 0.75,
            protected_prefix: 4,
            target_ratio: 0.5,
            beta: 0.7,
            merge_e: 1.0,
        }
    }
}

// ── Mutable state ────────────────────────────────────────────────

/// Per-invocation mutable state (wrapped in Mutex for interior mutability).
struct D2OState {
    /// EMA similarity threshold τ_t. Tokens merge only if similarity ≥ τ.
    ema_threshold: f32,
    /// Whether the EMA has been initialized (first eviction sets it).
    initialized: bool,
    /// Cumulative merge/delete statistics.
    total_merged: usize,
    total_deleted: usize,
}

impl D2OState {
    fn new() -> Self {
        Self {
            ema_threshold: 0.0,
            initialized: false,
            total_merged: 0,
            total_deleted: 0,
        }
    }
}

// ── D2OHandler ───────────────────────────────────────────────────

/// D2O cache pressure handler: H2O eviction + cosine similarity merge.
pub struct D2OHandler {
    config: D2OConfig,
    state: Mutex<D2OState>,
}

impl D2OHandler {
    pub fn new(config: D2OConfig) -> Self {
        Self {
            config,
            state: Mutex::new(D2OState::new()),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(target_ratio: f32) -> Self {
        Self::new(D2OConfig {
            target_ratio: target_ratio.clamp(0.1, 0.99),
            ..D2OConfig::default()
        })
    }

    /// Process a single layer's cache: partition → similarity → merge/delete → compact.
    fn evict_and_merge(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: &[f32],
        state: &mut D2OState,
    ) -> Result<usize> {
        let current = cache.current_pos;
        let prefix = self.config.protected_prefix.min(current);
        let keep = target_len.max(prefix + 2);

        if current <= keep {
            return Ok(0);
        }

        // ── Step 1: H2O-style 3-partition ──
        let available = keep.saturating_sub(prefix);
        let hh_budget = (available as f32 * self.config.keep_ratio) as usize;
        let recent_budget = available.saturating_sub(hh_budget);
        let recent_start = current.saturating_sub(recent_budget).max(prefix);
        let actual_recent = current - recent_start;
        let actual_hh_budget = available.saturating_sub(actual_recent);

        // Rank evictable tokens (prefix..recent_start) by importance
        let mut token_scores: Vec<(usize, f32)> = (prefix..recent_start)
            .map(|pos| (pos, importance.get(pos).copied().unwrap_or(0.0)))
            .collect();
        token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Split into keep (heavy hitters) and evict
        let hh_positions: Vec<usize> = token_scores
            .iter()
            .take(actual_hh_budget)
            .map(|(pos, _)| *pos)
            .collect();
        let evict_positions: Vec<usize> = token_scores
            .iter()
            .skip(actual_hh_budget)
            .map(|(pos, _)| *pos)
            .collect();

        if evict_positions.is_empty() {
            return Ok(0);
        }

        // Build the full retain set (sorted, for compaction)
        let recent_positions: Vec<usize> = (recent_start..current).collect();
        let mut retain_all: Vec<usize> = (0..prefix)
            .chain(hh_positions.iter().copied())
            .chain(recent_positions.iter().copied())
            .collect();
        retain_all.sort();

        // ── Step 2: Cosine similarity + merge decision ──
        let kv_heads = cache.kv_heads();
        let head_dim = cache.head_dim();
        let is_f32 = cache.k_buffer.dtype() == crate::core::buffer::DType::F32;

        // Merge targets exclude prefix tokens (prefix must remain unmodified)
        let merge_targets: Vec<usize> = retain_all
            .iter()
            .copied()
            .filter(|&p| p >= prefix)
            .collect();

        if is_f32 && !evict_positions.is_empty() && !merge_targets.is_empty() {
            // Collect max similarities for EMA initialization
            let mut max_sims: Vec<f32> = Vec::with_capacity(evict_positions.len());

            // For each evicted token, find nearest retained neighbor and decide merge/delete
            // Group by nearest neighbor: Vec<(evict_pos, nearest_retain_pos, similarity)>
            let mut merge_ops: Vec<(usize, usize, f32)> = Vec::new();

            for &evict_pos in &evict_positions {
                let (nearest_retain_pos, max_sim) =
                    find_nearest_cosine(cache, evict_pos, &merge_targets, kv_heads, head_dim);
                max_sims.push(max_sim);

                // Initialize EMA on first eviction
                if !state.initialized {
                    // τ_0 = average of max similarities (paper Eq. 10)
                    // We'll compute after collecting all max_sims
                } else if max_sim >= state.ema_threshold {
                    merge_ops.push((evict_pos, nearest_retain_pos, max_sim));
                }
            }

            // EMA initialization: τ_0 = mean(max_sims)
            if !state.initialized && !max_sims.is_empty() {
                let sum: f32 = max_sims.iter().sum();
                state.ema_threshold = sum / max_sims.len() as f32;
                state.initialized = true;

                // Re-evaluate merge decisions with the new threshold
                for (i, &evict_pos) in evict_positions.iter().enumerate() {
                    if max_sims[i] >= state.ema_threshold {
                        let (nearest_retain_pos, _) = find_nearest_cosine(
                            cache,
                            evict_pos,
                            &merge_targets,
                            kv_heads,
                            head_dim,
                        );
                        merge_ops.push((evict_pos, nearest_retain_pos, max_sims[i]));
                    }
                }
            }

            // Update EMA threshold: τ_t = β * max(U_t) + (1-β) * τ_{t-1}
            if let Some(&global_max_sim) = max_sims
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .filter(|_| state.initialized)
            {
                state.ema_threshold = self.config.beta * global_max_sim
                    + (1.0 - self.config.beta) * state.ema_threshold;
            }

            // ── Step 3: Apply merges ──
            let merge_count = merge_ops.len();
            let delete_count = evict_positions.len() - merge_count;

            for (evict_pos, retain_pos, sim) in &merge_ops {
                weighted_merge(
                    cache,
                    *evict_pos,
                    *retain_pos,
                    *sim,
                    self.config.merge_e,
                    kv_heads,
                    head_dim,
                );
            }

            state.total_merged += merge_count;
            state.total_deleted += delete_count;

            log::debug!(
                "[D2O] merged={}, deleted={}, threshold={:.4}",
                merge_count,
                delete_count,
                state.ema_threshold,
            );
        }

        // ── Step 4: Compact cache ──
        for (write_pos, &src_pos) in retain_all.iter().enumerate() {
            if src_pos != write_pos {
                cache.shift_positions(src_pos, write_pos, 1)?;
            }
        }
        cache.current_pos = retain_all.len();

        Ok(current - cache.current_pos)
    }
}

impl CachePressureHandler for D2OHandler {
    fn handle(&self, ctx: &mut HandlerContext) -> Result<ActionResult> {
        if ctx.caches.is_empty() {
            return Ok(ActionResult::NoOp);
        }

        let importance = match ctx.importance {
            Some(imp) => imp,
            None => return Ok(ActionResult::NoOp),
        };

        let current_pos = ctx.caches[0].current_pos;
        let target_len = ((current_pos as f32) * self.config.target_ratio) as usize;
        let target_len = target_len.max(1);

        if current_pos <= target_len {
            return Ok(ActionResult::NoOp);
        }

        let mut state = self.state.lock().unwrap();
        let mut total_removed = 0;
        let mut min_new_pos = usize::MAX;

        for cache in ctx.caches.iter_mut() {
            // Phase 1: uniform budget (same target_len for all layers)
            // Phase 2 will use per-layer α_l from attention variance
            let removed = self.evict_and_merge(cache, target_len, importance, &mut state)?;
            total_removed += removed;
            min_new_pos = min_new_pos.min(cache.current_pos);
        }

        if total_removed > 0 {
            log::info!(
                "[D2O] Evicted {} tokens across {} layers (merged={}, deleted={}), new_pos={}",
                total_removed,
                ctx.caches.len(),
                state.total_merged,
                state.total_deleted,
                min_new_pos,
            );
            Ok(ActionResult::Evicted {
                tokens_removed: total_removed / ctx.caches.len(), // per-layer average
                new_pos: min_new_pos,
            })
        } else {
            Ok(ActionResult::NoOp)
        }
    }

    fn name(&self) -> &str {
        "d2o"
    }
}

// ── Pure functions ───────────────────────────────────────────────

/// Cosine similarity between two slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

/// Find the nearest retained token to `evict_pos` by average cosine similarity
/// across all KV heads. Returns (retain_position, max_similarity).
fn find_nearest_cosine(
    cache: &KVCache,
    evict_pos: usize,
    retain_set: &[usize],
    kv_heads: usize,
    head_dim: usize,
) -> (usize, f32) {
    let k_data = cache.k_buffer.as_slice::<f32>();
    let mut best_pos = retain_set[0];
    let mut best_sim = f32::NEG_INFINITY;

    for &retain_pos in retain_set {
        if retain_pos == evict_pos {
            continue;
        }
        let mut total_sim = 0.0f32;
        for h in 0..kv_heads {
            let off_e = cache.offset(evict_pos, h);
            let off_r = cache.offset(retain_pos, h);
            let k_e = &k_data[off_e..off_e + head_dim];
            let k_r = &k_data[off_r..off_r + head_dim];
            total_sim += cosine_similarity(k_e, k_r);
        }
        total_sim /= kv_heads as f32;
        if total_sim > best_sim {
            best_sim = total_sim;
            best_pos = retain_pos;
        }
    }
    (best_pos, best_sim)
}

/// Weighted merge of evicted token into retained token (both K and V).
///
/// D2O paper formula:
///   w_retain = e / (exp(sim) + e)
///   w_evict  = exp(sim) / (exp(sim) + e)
///   K[retain] = w_retain * K[retain] + w_evict * K[evict]
///   V[retain] = w_retain * V[retain] + w_evict * V[evict]
fn weighted_merge(
    cache: &mut KVCache,
    evict_pos: usize,
    retain_pos: usize,
    sim: f32,
    e: f32,
    kv_heads: usize,
    head_dim: usize,
) {
    let exp_sim = sim.exp();
    let denom = exp_sim + e;
    let w_retain = e / denom;
    let w_evict = exp_sim / denom;

    // Pre-compute offsets to avoid borrowing `cache` while mutating buffers
    let offsets: Vec<(usize, usize)> = (0..kv_heads)
        .map(|h| (cache.offset(retain_pos, h), cache.offset(evict_pos, h)))
        .collect();

    // Merge K vectors
    {
        let k_data = cache.k_buffer.as_mut_slice::<f32>();
        for &(off_r, off_e) in &offsets {
            for d in 0..head_dim {
                k_data[off_r + d] = w_retain * k_data[off_r + d] + w_evict * k_data[off_e + d];
            }
        }
    }

    // Merge V vectors
    {
        let v_data = cache.v_buffer.as_mut_slice::<f32>();
        for &(off_r, off_e) in &offsets {
            for d in 0..head_dim {
                v_data[off_r + d] = w_retain * v_data[off_r + d] + w_evict * v_data[off_e + d];
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::shared_buffer::SharedBuffer;
    use crate::core::buffer::DType;
    use crate::core::kv_cache::KVLayout;
    use crate::core::pressure::PressureLevel;
    use crate::core::shape::Shape;
    use crate::core::tensor::Tensor;
    use std::sync::Arc;

    // ── Test helpers ──

    /// Create a KVCache with known K/V values for testing.
    /// Layout: SeqMajor [1, max_seq, kv_heads, head_dim].
    fn make_cache(max_seq: usize, kv_heads: usize, head_dim: usize, pos: usize) -> KVCache {
        let backend = Arc::new(CpuBackend::new());
        let buf_size = max_seq * kv_heads * head_dim * 4;
        let k = Tensor::new(
            Shape::new(vec![1, max_seq, kv_heads, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq, kv_heads, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend,
        );
        let mut cache = KVCache::new(k, v, max_seq);
        cache.current_pos = pos;
        cache
    }

    fn make_cache_head_major(
        max_seq: usize,
        kv_heads: usize,
        head_dim: usize,
        pos: usize,
    ) -> KVCache {
        let backend = Arc::new(CpuBackend::new());
        let buf_size = max_seq * kv_heads * head_dim * 4;
        let k = Tensor::new(
            Shape::new(vec![1, kv_heads, max_seq, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, kv_heads, max_seq, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend,
        );
        let mut cache = KVCache::new(k, v, max_seq).with_layout(KVLayout::HeadMajor);
        cache.current_pos = pos;
        cache
    }

    /// Write a known F32 pattern to K[pos, head, dim].
    fn write_k(cache: &mut KVCache, pos: usize, head: usize, values: &[f32]) {
        let off = cache.offset(pos, head);
        let k_data = cache.k_buffer.as_mut_slice::<f32>();
        k_data[off..off + values.len()].copy_from_slice(values);
    }

    /// Write a known F32 pattern to V[pos, head, dim].
    fn write_v(cache: &mut KVCache, pos: usize, head: usize, values: &[f32]) {
        let off = cache.offset(pos, head);
        let v_data = cache.v_buffer.as_mut_slice::<f32>();
        v_data[off..off + values.len()].copy_from_slice(values);
    }

    /// Read K[pos, head] as a Vec<f32>.
    fn read_k(cache: &KVCache, pos: usize, head: usize, dim: usize) -> Vec<f32> {
        let off = cache.offset(pos, head);
        let k_data = cache.k_buffer.as_slice::<f32>();
        k_data[off..off + dim].to_vec()
    }

    fn read_v(cache: &KVCache, pos: usize, head: usize, dim: usize) -> Vec<f32> {
        let off = cache.offset(pos, head);
        let v_data = cache.v_buffer.as_slice::<f32>();
        v_data[off..off + dim].to_vec()
    }

    // ── Cosine similarity tests ──

    #[test]
    fn test_cosine_identical_vectors() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors: sim={sim}");
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal vectors: sim={sim}");
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = [1.0, 2.0, 3.0];
        let b = [-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6, "opposite vectors: sim={sim}");
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = [1.0, 2.0, 3.0];
        let b = [0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "zero vector: sim={sim}");
    }

    // ── Nearest neighbor tests ──

    #[test]
    fn test_find_nearest_cosine_seq_major() {
        let mut cache = make_cache(10, 1, 4, 5);

        // Write distinct K vectors
        write_k(&mut cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]); // pos 0
        write_k(&mut cache, 1, 0, &[0.0, 1.0, 0.0, 0.0]); // pos 1
        write_k(&mut cache, 2, 0, &[0.9, 0.1, 0.0, 0.0]); // pos 2 — similar to pos 0
        write_k(&mut cache, 3, 0, &[0.0, 0.0, 1.0, 0.0]); // pos 3
        write_k(&mut cache, 4, 0, &[0.0, 0.0, 0.0, 1.0]); // pos 4

        let retain = vec![0, 1, 3, 4];
        let (nearest, sim) = find_nearest_cosine(&cache, 2, &retain, 1, 4);

        assert_eq!(nearest, 0, "pos 2 should be nearest to pos 0");
        assert!(sim > 0.9, "similarity should be high, got {sim}");
    }

    #[test]
    fn test_find_nearest_cosine_head_major() {
        let mut cache = make_cache_head_major(10, 2, 4, 5);

        // Head 0
        write_k(&mut cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]);
        write_k(&mut cache, 1, 0, &[0.0, 1.0, 0.0, 0.0]);
        write_k(&mut cache, 2, 0, &[0.95, 0.05, 0.0, 0.0]); // similar to pos 0
        // Head 1
        write_k(&mut cache, 0, 1, &[0.0, 0.0, 1.0, 0.0]);
        write_k(&mut cache, 1, 1, &[0.0, 0.0, 0.0, 1.0]);
        write_k(&mut cache, 2, 1, &[0.0, 0.0, 0.9, 0.1]); // similar to pos 0

        let retain = vec![0, 1];
        let (nearest, _sim) = find_nearest_cosine(&cache, 2, &retain, 2, 4);
        assert_eq!(
            nearest, 0,
            "pos 2 should be nearest to pos 0 (averaged across heads)"
        );
    }

    // ── Weighted merge tests ──

    #[test]
    fn test_weighted_merge_values() {
        let mut cache = make_cache(10, 1, 4, 3);

        write_k(&mut cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]); // retain
        write_k(&mut cache, 1, 0, &[0.0, 1.0, 0.0, 0.0]); // evict

        write_v(&mut cache, 0, 0, &[10.0, 0.0, 0.0, 0.0]); // retain
        write_v(&mut cache, 1, 0, &[0.0, 10.0, 0.0, 0.0]); // evict

        let sim = 0.5;
        let e = 1.0;
        weighted_merge(&mut cache, 1, 0, sim, e, 1, 4);

        let exp_sim = (0.5f32).exp(); // ≈ 1.6487
        let w_retain = e / (exp_sim + e); // ≈ 0.3775
        let w_evict = exp_sim / (exp_sim + e); // ≈ 0.6225

        let k = read_k(&cache, 0, 0, 4);
        assert!(
            (k[0] - w_retain * 1.0).abs() < 1e-4,
            "k[0]={}, expected {}",
            k[0],
            w_retain
        );
        assert!(
            (k[1] - w_evict * 1.0).abs() < 1e-4,
            "k[1]={}, expected {}",
            k[1],
            w_evict
        );

        let v = read_v(&cache, 0, 0, 4);
        assert!(
            (v[0] - w_retain * 10.0).abs() < 1e-3,
            "v[0]={}, expected {}",
            v[0],
            w_retain * 10.0
        );
        assert!(
            (v[1] - w_evict * 10.0).abs() < 1e-3,
            "v[1]={}, expected {}",
            v[1],
            w_evict * 10.0
        );
    }

    #[test]
    fn test_merge_weights_sum_to_one() {
        for sim_x10 in 0..=10 {
            let sim = sim_x10 as f32 * 0.1;
            let exp_sim = sim.exp();
            let e = 1.0;
            let w_r = e / (exp_sim + e);
            let w_e = exp_sim / (exp_sim + e);
            assert!(
                (w_r + w_e - 1.0).abs() < 1e-6,
                "weights don't sum to 1 at sim={sim}: w_r={w_r}, w_e={w_e}"
            );
        }
    }

    // ── EMA threshold tests ──

    #[test]
    fn test_ema_initialization() {
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            beta: 0.7,
            merge_e: 1.0,
        });

        let mut cache = make_cache(20, 1, 4, 10);
        // Write K vectors: make some similar pairs
        for pos in 0..10 {
            let val = pos as f32;
            write_k(&mut cache, pos, 0, &[val, val, val, val]);
        }

        let importance: Vec<f32> = (0..20).map(|i| 10.0 - i as f32).collect();

        let state = handler.state.lock().unwrap();
        assert!(!state.initialized, "should start uninitialized");
        drop(state);

        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
        };

        handler.handle(&mut ctx).unwrap();

        let state = handler.state.lock().unwrap();
        assert!(
            state.initialized,
            "should be initialized after first eviction"
        );
        assert!(state.ema_threshold > 0.0, "threshold should be positive");
    }

    #[test]
    fn test_ema_update() {
        let beta: f32 = 0.7;
        let mut threshold: f32 = 0.5; // initial

        // Update with new max similarity = 0.8
        threshold = beta * 0.8 + (1.0 - beta) * threshold;
        let expected = 0.7 * 0.8 + 0.3 * 0.5; // = 0.56 + 0.15 = 0.71
        assert!(
            (threshold - expected).abs() < 1e-6,
            "threshold={threshold}, expected={expected}"
        );

        // Update with lower similarity = 0.3
        threshold = beta * 0.3 + (1.0 - beta) * threshold;
        let expected2 = 0.7 * 0.3 + 0.3 * expected; // = 0.21 + 0.213 = 0.423
        assert!(
            (threshold - expected2).abs() < 1e-6,
            "threshold={threshold}, expected={expected2}"
        );
    }

    // ── Handler integration tests ──

    #[test]
    fn test_handler_noop_no_importance() {
        let handler = D2OHandler::with_defaults(0.5);
        let mut caches = vec![make_cache(20, 1, 4, 10)];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None, // no scores
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
        };

        let result = handler.handle(&mut ctx).unwrap();
        assert!(!result.is_action());
    }

    #[test]
    fn test_handler_noop_below_target() {
        let handler = D2OHandler::with_defaults(0.9); // keep 90%
        let cache = make_cache(20, 1, 4, 5); // only 5 tokens
        let importance = vec![1.0; 20];

        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
        };

        let result = handler.handle(&mut ctx).unwrap();
        // target_len = 5 * 0.9 = 4; 5 > 4 so eviction should happen
        // But with prefix=4, keep=max(4, 4+2)=6, 5 <= 6 → NoOp
        assert!(!result.is_action());
    }

    #[test]
    fn test_handler_evicts_tokens() {
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            beta: 0.7,
            merge_e: 1.0,
        });

        let mut cache = make_cache(50, 1, 4, 30);
        // Write distinct K vectors for meaningful similarity
        for pos in 0..30 {
            let angle = pos as f32 * 0.2;
            write_k(&mut cache, pos, 0, &[angle.cos(), angle.sin(), 0.0, 0.0]);
            write_v(&mut cache, pos, 0, &[pos as f32, 0.0, 0.0, 0.0]);
        }

        // High importance for first tokens, low for middle
        let mut importance = vec![0.1; 50];
        for i in 0..5 {
            importance[i] = 10.0;
        }
        for i in 25..30 {
            importance[i] = 5.0; // recent window
        }

        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
        };

        let result = handler.handle(&mut ctx).unwrap();
        match result {
            ActionResult::Evicted {
                tokens_removed,
                new_pos,
            } => {
                assert!(tokens_removed > 0, "should have removed tokens");
                assert!(new_pos < 30, "new_pos should be less than original");
                assert!(new_pos >= 2, "should keep at least prefix");
                // target = 30 * 0.5 = 15
                assert_eq!(new_pos, 15, "should reduce to target");
            }
            other => panic!("Expected Evicted, got {:?}", other),
        }
    }

    #[test]
    fn test_handler_empty_caches() {
        let handler = D2OHandler::with_defaults(0.5);
        let importance = vec![1.0; 10];
        let mut caches: Vec<KVCache> = vec![];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
        };
        let result = handler.handle(&mut ctx).unwrap();
        assert!(!result.is_action());
    }

    #[test]
    fn test_handler_multi_layer() {
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            beta: 0.7,
            merge_e: 1.0,
        });

        let mut caches: Vec<KVCache> = (0..4)
            .map(|_| {
                let mut c = make_cache(50, 1, 4, 20);
                for pos in 0..20 {
                    let angle = pos as f32 * 0.3;
                    write_k(&mut c, pos, 0, &[angle.cos(), angle.sin(), 0.0, 0.0]);
                    write_v(&mut c, pos, 0, &[pos as f32, 0.0, 0.0, 0.0]);
                }
                c
            })
            .collect();

        let mut importance = vec![0.1; 50];
        for i in 0..5 {
            importance[i] = 10.0;
        }

        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
        };

        let result = handler.handle(&mut ctx).unwrap();
        assert!(result.is_action());

        // All layers should be at the same position
        let pos = ctx.caches[0].current_pos;
        for cache in ctx.caches.iter() {
            assert_eq!(cache.current_pos, pos, "all layers should have same pos");
        }
    }

    #[test]
    fn test_handler_name() {
        let handler = D2OHandler::with_defaults(0.5);
        assert_eq!(handler.name(), "d2o");
    }

    #[test]
    fn test_prefix_always_protected() {
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 4,
            target_ratio: 0.3, // aggressive
            beta: 0.7,
            merge_e: 1.0,
        });

        let mut cache = make_cache(50, 1, 4, 20);
        // Write unique K vectors for prefix tokens
        for pos in 0..4 {
            let val = 100.0 + pos as f32;
            write_k(&mut cache, pos, 0, &[val, val, val, val]);
        }
        for pos in 4..20 {
            write_k(&mut cache, pos, 0, &[0.1, 0.1, 0.1, 0.1]);
        }

        let importance = vec![0.01; 50]; // low importance everywhere
        // Prefix still protected regardless of importance

        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
        };

        handler.handle(&mut ctx).unwrap();

        // Check prefix tokens are preserved
        for pos in 0..4 {
            let k = read_k(&ctx.caches[0], pos, 0, 4);
            let expected = 100.0 + pos as f32;
            assert!(
                (k[0] - expected).abs() < 1e-4,
                "prefix pos {pos} should be preserved: k={k:?}",
            );
        }
    }

    #[test]
    fn test_keep_ratio_3_to_1() {
        // Verify that D2O's 3:1 HH:Recent ratio gives more budget to heavy hitters
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            beta: 0.7,
            merge_e: 1.0,
        });

        let mut cache = make_cache(50, 1, 4, 40);
        for pos in 0..40 {
            write_k(&mut cache, pos, 0, &[pos as f32, 0.0, 0.0, 0.0]);
        }
        // Assign importance: tokens 5, 10, 15 are most important
        let mut importance = vec![0.01; 50];
        importance[5] = 10.0;
        importance[10] = 9.0;
        importance[15] = 8.0;
        importance[6] = 7.0;
        importance[7] = 6.0;
        importance[8] = 5.0;
        importance[9] = 4.0;
        importance[11] = 3.0;
        importance[12] = 2.0;
        importance[13] = 1.5;
        importance[14] = 1.0;

        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
        };

        handler.handle(&mut ctx).unwrap();

        let new_pos = ctx.caches[0].current_pos;
        // target = 40 * 0.5 = 20; keep = max(20, 2+2) = 20
        // available = 20 - 2 = 18; hh_budget = 18 * 0.75 = 13; recent = 5
        assert_eq!(new_pos, 20);
    }
}
