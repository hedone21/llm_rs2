//! D2O (Dynamic Discriminative Operations) handler — eviction with compensation merging.
//!
//! Implements the D2O paper (Wan et al., 2024): H2O-style 3-partition eviction
//! with token merging compensation. Evicted tokens are matched to their nearest
//! retained neighbor by cosine similarity; if similarity exceeds an EMA threshold,
//! the evicted token is merged (weighted average) rather than permanently deleted.
//!
//! Phase 1: Uniform per-layer budget (layer-level variance allocation deferred).
//! Supports F32, F16, and Q4_0 KV cache dtypes.

use super::{ActionResult, CachePressureHandler, HandlerContext};
use crate::core::buffer::DType;
use crate::core::kv_cache::KVCache;
use crate::core::quant::{BlockQ4_0, QK4_0};
use anyhow::Result;
use half::f16;
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

        // Merge targets exclude prefix tokens (prefix must remain unmodified)
        let merge_targets: Vec<usize> = retain_all
            .iter()
            .copied()
            .filter(|&p| p >= prefix)
            .collect();

        if !evict_positions.is_empty() && !merge_targets.is_empty() {
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
        // Use signal's target_ratio if provided, otherwise fall back to config
        let effective_ratio = ctx.target_ratio.unwrap_or(self.config.target_ratio);
        let target_len = ((current_pos as f32) * effective_ratio) as usize;
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

/// Dequantize a K vector at (pos, head) into the output buffer.
/// Works for F32, F16, and Q4_0 dtypes.
fn dequantize_k(cache: &KVCache, pos: usize, head: usize, head_dim: usize, out: &mut [f32]) {
    match cache.k_buffer.dtype() {
        DType::F32 => {
            let k = cache.k_buffer.as_slice::<f32>();
            let off = cache.offset(pos, head);
            out[..head_dim].copy_from_slice(&k[off..off + head_dim]);
        }
        DType::F16 => {
            let k = cache.k_buffer.as_slice::<f16>();
            let off = cache.offset(pos, head);
            for d in 0..head_dim {
                out[d] = k[off + d].to_f32();
            }
        }
        DType::Q4_0 => {
            let k = cache.k_buffer.as_slice::<BlockQ4_0>();
            let blocks_per_pos = head_dim / QK4_0;
            let block_off = cache.q4_block_offset(pos, head, blocks_per_pos);
            for bi in 0..blocks_per_pos {
                let mut tmp = [0.0f32; QK4_0];
                k[block_off + bi].dequantize(&mut tmp);
                let dst = bi * QK4_0;
                out[dst..dst + QK4_0].copy_from_slice(&tmp);
            }
        }
        _ => {}
    }
}

/// Find the nearest retained token to `evict_pos` by average cosine similarity
/// across all KV heads. Returns (retain_position, max_similarity).
/// Supports F32, F16, and Q4_0 KV cache dtypes.
fn find_nearest_cosine(
    cache: &KVCache,
    evict_pos: usize,
    retain_set: &[usize],
    kv_heads: usize,
    head_dim: usize,
) -> (usize, f32) {
    let mut best_pos = retain_set[0];
    let mut best_sim = f32::NEG_INFINITY;
    let mut evict_buf = vec![0.0f32; head_dim];
    let mut retain_buf = vec![0.0f32; head_dim];

    for &retain_pos in retain_set {
        if retain_pos == evict_pos {
            continue;
        }
        let mut total_sim = 0.0f32;
        for h in 0..kv_heads {
            dequantize_k(cache, evict_pos, h, head_dim, &mut evict_buf);
            dequantize_k(cache, retain_pos, h, head_dim, &mut retain_buf);
            total_sim += cosine_similarity(&evict_buf[..head_dim], &retain_buf[..head_dim]);
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
///
/// Supports F32 (in-place), F16 (f16↔f32), Q4_0 (deq→merge→req).
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

    let dtype = cache.k_buffer.dtype();
    match dtype {
        DType::F32 => {
            weighted_merge_f32(
                cache, evict_pos, retain_pos, w_retain, w_evict, kv_heads, head_dim,
            );
        }
        DType::F16 => {
            weighted_merge_f16(
                cache, evict_pos, retain_pos, w_retain, w_evict, kv_heads, head_dim,
            );
        }
        DType::Q4_0 => {
            weighted_merge_q4(
                cache, evict_pos, retain_pos, w_retain, w_evict, kv_heads, head_dim,
            );
        }
        _ => {}
    }
}

fn weighted_merge_f32(
    cache: &mut KVCache,
    evict_pos: usize,
    retain_pos: usize,
    w_retain: f32,
    w_evict: f32,
    kv_heads: usize,
    head_dim: usize,
) {
    let offsets: Vec<(usize, usize)> = (0..kv_heads)
        .map(|h| (cache.offset(retain_pos, h), cache.offset(evict_pos, h)))
        .collect();
    {
        let k = cache.k_buffer.as_mut_slice::<f32>();
        for &(off_r, off_e) in &offsets {
            for d in 0..head_dim {
                k[off_r + d] = w_retain * k[off_r + d] + w_evict * k[off_e + d];
            }
        }
    }
    {
        let v = cache.v_buffer.as_mut_slice::<f32>();
        for &(off_r, off_e) in &offsets {
            for d in 0..head_dim {
                v[off_r + d] = w_retain * v[off_r + d] + w_evict * v[off_e + d];
            }
        }
    }
}

fn weighted_merge_f16(
    cache: &mut KVCache,
    evict_pos: usize,
    retain_pos: usize,
    w_retain: f32,
    w_evict: f32,
    kv_heads: usize,
    head_dim: usize,
) {
    let offsets: Vec<(usize, usize)> = (0..kv_heads)
        .map(|h| (cache.offset(retain_pos, h), cache.offset(evict_pos, h)))
        .collect();
    {
        let k = cache.k_buffer.as_mut_slice::<f16>();
        for &(off_r, off_e) in &offsets {
            for d in 0..head_dim {
                let vr = k[off_r + d].to_f32();
                let ve = k[off_e + d].to_f32();
                k[off_r + d] = f16::from_f32(w_retain * vr + w_evict * ve);
            }
        }
    }
    {
        let v = cache.v_buffer.as_mut_slice::<f16>();
        for &(off_r, off_e) in &offsets {
            for d in 0..head_dim {
                let vr = v[off_r + d].to_f32();
                let ve = v[off_e + d].to_f32();
                v[off_r + d] = f16::from_f32(w_retain * vr + w_evict * ve);
            }
        }
    }
}

fn weighted_merge_q4(
    cache: &mut KVCache,
    evict_pos: usize,
    retain_pos: usize,
    w_retain: f32,
    w_evict: f32,
    kv_heads: usize,
    head_dim: usize,
) {
    let blocks_per_pos = head_dim / QK4_0;
    let block_offsets: Vec<(usize, usize)> = (0..kv_heads)
        .map(|h| {
            (
                cache.q4_block_offset(retain_pos, h, blocks_per_pos),
                cache.q4_block_offset(evict_pos, h, blocks_per_pos),
            )
        })
        .collect();

    // Merge K
    {
        let k = cache.k_buffer.as_mut_slice::<BlockQ4_0>();
        let mut r_f32 = [0.0f32; QK4_0];
        let mut e_f32 = [0.0f32; QK4_0];
        for &(off_r, off_e) in &block_offsets {
            for bi in 0..blocks_per_pos {
                k[off_r + bi].dequantize(&mut r_f32);
                k[off_e + bi].dequantize(&mut e_f32);
                for i in 0..QK4_0 {
                    r_f32[i] = w_retain * r_f32[i] + w_evict * e_f32[i];
                }
                k[off_r + bi] = BlockQ4_0::quantize(&r_f32);
            }
        }
    }

    // Merge V
    {
        let v = cache.v_buffer.as_mut_slice::<BlockQ4_0>();
        let mut r_f32 = [0.0f32; QK4_0];
        let mut e_f32 = [0.0f32; QK4_0];
        for &(off_r, off_e) in &block_offsets {
            for bi in 0..blocks_per_pos {
                v[off_r + bi].dequantize(&mut r_f32);
                v[off_e + bi].dequantize(&mut e_f32);
                for i in 0..QK4_0 {
                    r_f32[i] = w_retain * r_f32[i] + w_evict * e_f32[i];
                }
                v[off_r + bi] = BlockQ4_0::quantize(&r_f32);
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
            target_ratio: None,
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
            target_ratio: None,
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
            target_ratio: None,
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
            target_ratio: None,
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
            target_ratio: None,
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
            target_ratio: None,
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
            target_ratio: None,
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
            target_ratio: None,
        };

        handler.handle(&mut ctx).unwrap();

        let new_pos = ctx.caches[0].current_pos;
        // target = 40 * 0.5 = 20; keep = max(20, 2+2) = 20
        // available = 20 - 2 = 18; hh_budget = 18 * 0.75 = 13; recent = 5
        assert_eq!(new_pos, 20);
    }

    // ── Q4_0 tests ──

    /// Create a Q4_0 KV cache for testing.
    fn make_cache_q4(max_seq: usize, kv_heads: usize, head_dim: usize, pos: usize) -> KVCache {
        let backend = Arc::new(CpuBackend::new());
        let blocks_per_pos = head_dim / QK4_0;
        let total_blocks = max_seq * kv_heads * blocks_per_pos;
        let buf_size = total_blocks * std::mem::size_of::<BlockQ4_0>();
        let k = Tensor::new(
            Shape::new(vec![1, max_seq, kv_heads, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::Q4_0)),
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq, kv_heads, head_dim]),
            Arc::new(SharedBuffer::new(buf_size, DType::Q4_0)),
            backend,
        );
        let mut cache = KVCache::new(k, v, max_seq);
        cache.current_pos = pos;
        cache
    }

    /// Write an f32 vector to Q4_0 K cache at (pos, head) via quantize.
    fn write_k_q4(cache: &mut KVCache, pos: usize, head: usize, values: &[f32]) {
        let head_dim = values.len();
        let blocks_per_pos = head_dim / QK4_0;
        let block_off = cache.q4_block_offset(pos, head, blocks_per_pos);
        let k = cache.k_buffer.as_mut_slice::<BlockQ4_0>();
        for bi in 0..blocks_per_pos {
            let start = bi * QK4_0;
            let mut src = [0.0f32; QK4_0];
            src.copy_from_slice(&values[start..start + QK4_0]);
            k[block_off + bi] = BlockQ4_0::quantize(&src);
        }
    }

    /// Read K cache at (pos, head) from Q4_0 as dequantized f32.
    fn read_k_q4(cache: &KVCache, pos: usize, head: usize, head_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; head_dim];
        dequantize_k(cache, pos, head, head_dim, &mut out);
        out
    }

    #[test]
    fn test_quantize_round_trip() {
        let src: [f32; QK4_0] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.5);
        let block = BlockQ4_0::quantize(&src);
        let mut dst = [0.0f32; QK4_0];
        block.dequantize(&mut dst);
        // Q4_0 has limited precision; check within tolerance
        for i in 0..QK4_0 {
            assert!(
                (src[i] - dst[i]).abs() < 1.5,
                "round-trip drift too large at {i}: src={}, dst={}",
                src[i],
                dst[i]
            );
        }
    }

    #[test]
    fn test_q4_dequantize_k() {
        let mut cache = make_cache_q4(10, 1, 64, 3);
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        write_k_q4(&mut cache, 1, 0, &values);
        let read = read_k_q4(&cache, 1, 0, 64);
        // Verify dequantize produces values close to originals
        for i in 0..64 {
            assert!(
                (values[i] - read[i]).abs() < 0.5,
                "mismatch at {i}: wrote={}, read={}",
                values[i],
                read[i]
            );
        }
    }

    #[test]
    fn test_q4_find_nearest_cosine() {
        // head_dim must be multiple of QK4_0=32, use 32
        let mut cache = make_cache_q4(10, 1, 32, 3);
        // pos 0: all positive
        let v0: Vec<f32> = (0..32).map(|i| 1.0 + i as f32 * 0.1).collect();
        // pos 1: all negative (opposite direction)
        let v1: Vec<f32> = (0..32).map(|i| -1.0 - i as f32 * 0.1).collect();
        // pos 2: similar to pos 0
        let v2: Vec<f32> = (0..32).map(|i| 1.1 + i as f32 * 0.1).collect();
        write_k_q4(&mut cache, 0, 0, &v0);
        write_k_q4(&mut cache, 1, 0, &v1);
        write_k_q4(&mut cache, 2, 0, &v2);

        let retain = vec![0, 1];
        let (nearest, sim) = find_nearest_cosine(&cache, 2, &retain, 1, 32);
        assert_eq!(nearest, 0, "pos 2 should be nearest to pos 0");
        assert!(sim > 0.9, "similarity should be high, got {sim}");
    }

    #[test]
    fn test_q4_weighted_merge() {
        let mut cache = make_cache_q4(10, 1, 32, 3);
        let v_retain: Vec<f32> = (0..32).map(|i| 2.0 + i as f32 * 0.1).collect();
        let v_evict: Vec<f32> = (0..32).map(|i| 4.0 + i as f32 * 0.1).collect();
        write_k_q4(&mut cache, 0, 0, &v_retain);
        write_k_q4(&mut cache, 1, 0, &v_evict);

        // sim=0 → w_retain=e/(1+e)=0.5, w_evict=1/(1+1)=0.5 (equal weights)
        weighted_merge(&mut cache, 1, 0, 0.0, 1.0, 1, 32);

        let merged = read_k_q4(&cache, 0, 0, 32);
        // After merge: approximately (v_retain + v_evict) / 2
        // Q4_0 quantization adds noise, but average should be roughly correct
        let expected_avg: f32 = (0..32)
            .map(|i| (2.0 + i as f32 * 0.1 + 4.0 + i as f32 * 0.1) / 2.0)
            .sum::<f32>()
            / 32.0;
        let actual_avg: f32 = merged.iter().sum::<f32>() / 32.0;
        assert!(
            (expected_avg - actual_avg).abs() < 1.0,
            "Q4_0 merge average drift too large: expected={expected_avg}, actual={actual_avg}"
        );
    }

    #[test]
    fn test_handler_q4_evicts() {
        // Full handler integration with Q4_0 cache
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            beta: 0.7,
            merge_e: 1.0,
        });

        let mut cache = make_cache_q4(50, 1, 32, 20);
        // Write distinct K vectors so similarity is meaningful
        for pos in 0..20 {
            let v: Vec<f32> = (0..32).map(|d| ((pos * 32 + d) as f32) * 0.01).collect();
            write_k_q4(&mut cache, pos, 0, &v);
        }

        let importance: Vec<f32> = (0..50).map(|i| 20.0 - i as f32).collect();
        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
        };

        handler.handle(&mut ctx).unwrap();

        let new_pos = ctx.caches[0].current_pos;
        // target = 20 * 0.5 = 10
        assert!(
            new_pos <= 10,
            "Q4_0 eviction should reduce pos, got {new_pos}"
        );
        assert!(new_pos >= 2, "prefix should be preserved");
    }
}
