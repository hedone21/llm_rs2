//! D2O (Dynamic Discriminative Operations) handler — eviction with compensation merging.
//!
//! Implements the D2O paper (Wan et al., 2024): H2O-style 3-partition eviction
//! with token merging compensation. Evicted tokens are matched to their nearest
//! retained neighbor by per-head cosine similarity; if head-averaged similarity
//! exceeds an EMA threshold, the evicted token is merged via scatter-reduce rather
//! than permanently deleted.
//!
//! Phase A: Per-head scatter-reduce merge + EMA correction (mean-based, alpha/beta=0.5).
//! Phase B: Layer-level dynamic allocation (deferred).
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
    /// EMA old-threshold weight (official code default 0.5).
    pub ema_alpha: f32,
    /// EMA new-mean weight (official code default 0.5).
    pub ema_beta: f32,
    /// Enable per-layer dynamic budget allocation (Phase B).
    pub use_layer_allocation: bool,
    /// Layer indices to skip eviction entirely.
    pub protected_layers: Vec<usize>,
}

impl Default for D2OConfig {
    fn default() -> Self {
        Self {
            keep_ratio: 0.75,
            protected_prefix: 4,
            target_ratio: 0.5,
            ema_alpha: 0.5,
            ema_beta: 0.5,
            use_layer_allocation: false,
            protected_layers: vec![],
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

// ── PerHeadMatch ─────────────────────────────────────────────────

/// Per-head nearest neighbor matching result for a single evicted token.
struct PerHeadMatch {
    /// Per-head (nearest_retain_pos, cosine_similarity). Length = kv_heads.
    per_head: Vec<(usize, f32)>,
    /// Head-averaged similarity (for EMA threshold filtering).
    mean_sim: f32,
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

        // ── Step 2: Per-head nearest neighbor ──
        let kv_heads = cache.kv_heads();
        let head_dim = cache.head_dim();

        // Merge targets exclude prefix tokens (prefix must remain unmodified)
        let merge_targets: Vec<usize> = retain_all
            .iter()
            .copied()
            .filter(|&p| p >= prefix)
            .collect();

        let all_matches: Vec<PerHeadMatch> = evict_positions
            .iter()
            .map(|&pos| find_nearest_per_head(cache, pos, &merge_targets, kv_heads, head_dim))
            .collect();
        let mean_sims: Vec<f32> = all_matches.iter().map(|m| m.mean_sim).collect();

        // ── Step 3: EMA threshold ──
        let just_initialized;
        if !mean_sims.is_empty() {
            let mean_of_sims = mean_sims.iter().sum::<f32>() / mean_sims.len() as f32;
            if !state.initialized {
                state.ema_threshold = mean_of_sims;
                state.initialized = true;
                just_initialized = true;
            } else {
                state.ema_threshold = self.config.ema_alpha * state.ema_threshold
                    + self.config.ema_beta * mean_of_sims;
                just_initialized = false;
            }
        } else {
            just_initialized = false;
        }
        let _ = just_initialized; // used for clarity in comments

        // ── Step 4: Global filter — mean_sim >= threshold ──
        let passing_indices: Vec<usize> = (0..evict_positions.len())
            .filter(|&i| mean_sims[i] >= state.ema_threshold)
            .collect();

        let merge_count = passing_indices.len();
        let delete_count = evict_positions.len() - merge_count;

        // ── Step 5: Per-head scatter-reduce merge ──
        if !passing_indices.is_empty() {
            let passing_positions: Vec<usize> = passing_indices
                .iter()
                .map(|&i| evict_positions[i])
                .collect();
            let passing_matches: Vec<&PerHeadMatch> =
                passing_indices.iter().map(|&i| &all_matches[i]).collect();
            scatter_reduce_merge_per_head(
                cache,
                &passing_positions,
                &passing_matches,
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

        // ── Step 6: Compact cache ──
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

        // Validate: all caches must use the same layout.
        debug_assert!(
            ctx.caches
                .windows(2)
                .all(|w| w[0].layout() == w[1].layout()),
            "D2OHandler: all caches must use the same KVLayout"
        );

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
            // Phase A: uniform budget (same target_len for all layers)
            // Phase B will use per-layer α_l from attention variance
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

/// Find the nearest retained token per head independently.
///
/// For each KV head, scans all retain positions and finds the one with the
/// highest cosine similarity to the evicted token at that head.
/// Returns a `PerHeadMatch` with per-head results and their average.
#[allow(clippy::needless_range_loop)]
fn find_nearest_per_head(
    cache: &KVCache,
    evict_pos: usize,
    retain_set: &[usize],
    kv_heads: usize,
    head_dim: usize,
) -> PerHeadMatch {
    let mut per_head = Vec::with_capacity(kv_heads);
    let mut evict_buf = vec![0.0f32; head_dim];
    let mut retain_buf = vec![0.0f32; head_dim];

    for h in 0..kv_heads {
        dequantize_k(cache, evict_pos, h, head_dim, &mut evict_buf);

        let mut best_pos = retain_set[0];
        let mut best_sim = f32::NEG_INFINITY;

        for &retain_pos in retain_set {
            if retain_pos == evict_pos {
                continue;
            }
            dequantize_k(cache, retain_pos, h, head_dim, &mut retain_buf);
            let sim = cosine_similarity(&evict_buf[..head_dim], &retain_buf[..head_dim]);
            if sim > best_sim {
                best_sim = sim;
                best_pos = retain_pos;
            }
        }
        per_head.push((best_pos, best_sim));
    }

    let mean_sim = if kv_heads > 0 {
        per_head.iter().map(|(_, s)| s).sum::<f32>() / kv_heads as f32
    } else {
        0.0
    };

    PerHeadMatch { per_head, mean_sim }
}

/// Scatter-reduce merge: for each head independently, group evicted tokens by
/// their per-head nearest retain target and apply mean pooling.
///
/// For each group (retain_pos ← [evict_0, evict_1, ...]):
///   K[retain, h] = (K[retain, h] + Σ sim_i[h] · K[evict_i, h]) / (1 + N)
///   V[retain, h] = (V[retain, h] + Σ sim_i[h] · V[evict_i, h]) / (1 + N)
#[allow(clippy::needless_range_loop)]
fn scatter_reduce_merge_per_head(
    cache: &mut KVCache,
    passing_positions: &[usize],
    matches: &[&PerHeadMatch],
    kv_heads: usize,
    head_dim: usize,
) {
    let dtype = cache.k_buffer.dtype();
    match dtype {
        DType::F32 => {
            scatter_reduce_f32(cache, passing_positions, matches, kv_heads, head_dim);
        }
        DType::F16 => {
            scatter_reduce_f16(cache, passing_positions, matches, kv_heads, head_dim);
        }
        DType::Q4_0 => {
            scatter_reduce_q4(cache, passing_positions, matches, kv_heads, head_dim);
        }
        _ => {}
    }
}

#[allow(clippy::needless_range_loop)]
fn scatter_reduce_f32(
    cache: &mut KVCache,
    passing_positions: &[usize],
    matches: &[&PerHeadMatch],
    kv_heads: usize,
    head_dim: usize,
) {
    use std::collections::HashMap;

    for h in 0..kv_heads {
        // Group: retain_pos → Vec<(evict_pos, similarity_at_this_head)>
        let mut groups: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for (i, &evict_pos) in passing_positions.iter().enumerate() {
            let (retain_pos, sim) = matches[i].per_head[h];
            groups.entry(retain_pos).or_default().push((evict_pos, sim));
        }

        for (retain_pos, evicted_list) in &groups {
            let count = evicted_list.len() + 1; // include retain itself
            let inv_count = 1.0f32 / count as f32;

            let retain_off = cache.offset(*retain_pos, h);

            // Accumulate weighted sum into retain slot (K)
            {
                // Collect evict contributions first to avoid borrow conflict
                let evict_contributions: Vec<(usize, f32)> = evicted_list
                    .iter()
                    .map(|&(ep, s)| (cache.offset(ep, h), s))
                    .collect();
                let k = cache.k_buffer.as_mut_slice::<f32>();
                for d in 0..head_dim {
                    let mut acc = k[retain_off + d];
                    for &(evict_off, sim) in &evict_contributions {
                        acc += sim * k[evict_off + d];
                    }
                    k[retain_off + d] = acc * inv_count;
                }
            }
            // V
            {
                let evict_contributions: Vec<(usize, f32)> = evicted_list
                    .iter()
                    .map(|&(ep, s)| (cache.offset(ep, h), s))
                    .collect();
                let v = cache.v_buffer.as_mut_slice::<f32>();
                for d in 0..head_dim {
                    let mut acc = v[retain_off + d];
                    for &(evict_off, sim) in &evict_contributions {
                        acc += sim * v[evict_off + d];
                    }
                    v[retain_off + d] = acc * inv_count;
                }
            }
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn scatter_reduce_f16(
    cache: &mut KVCache,
    passing_positions: &[usize],
    matches: &[&PerHeadMatch],
    kv_heads: usize,
    head_dim: usize,
) {
    use std::collections::HashMap;

    for h in 0..kv_heads {
        let mut groups: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for (i, &evict_pos) in passing_positions.iter().enumerate() {
            let (retain_pos, sim) = matches[i].per_head[h];
            groups.entry(retain_pos).or_default().push((evict_pos, sim));
        }

        for (retain_pos, evicted_list) in &groups {
            let count = evicted_list.len() + 1;
            let inv_count = 1.0f32 / count as f32;

            let retain_off = cache.offset(*retain_pos, h);

            // K
            {
                let evict_contributions: Vec<(usize, f32)> = evicted_list
                    .iter()
                    .map(|&(ep, s)| (cache.offset(ep, h), s))
                    .collect();
                let k = cache.k_buffer.as_mut_slice::<f16>();
                for d in 0..head_dim {
                    let mut acc = k[retain_off + d].to_f32();
                    for &(evict_off, sim) in &evict_contributions {
                        acc += sim * k[evict_off + d].to_f32();
                    }
                    k[retain_off + d] = f16::from_f32(acc * inv_count);
                }
            }
            // V
            {
                let evict_contributions: Vec<(usize, f32)> = evicted_list
                    .iter()
                    .map(|&(ep, s)| (cache.offset(ep, h), s))
                    .collect();
                let v = cache.v_buffer.as_mut_slice::<f16>();
                for d in 0..head_dim {
                    let mut acc = v[retain_off + d].to_f32();
                    for &(evict_off, sim) in &evict_contributions {
                        acc += sim * v[evict_off + d].to_f32();
                    }
                    v[retain_off + d] = f16::from_f32(acc * inv_count);
                }
            }
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn scatter_reduce_q4(
    cache: &mut KVCache,
    passing_positions: &[usize],
    matches: &[&PerHeadMatch],
    kv_heads: usize,
    head_dim: usize,
) {
    use std::collections::HashMap;

    let blocks_per_pos = head_dim / QK4_0;

    for h in 0..kv_heads {
        let mut groups: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for (i, &evict_pos) in passing_positions.iter().enumerate() {
            let (retain_pos, sim) = matches[i].per_head[h];
            groups.entry(retain_pos).or_default().push((evict_pos, sim));
        }

        for (retain_pos, evicted_list) in &groups {
            let count = evicted_list.len() + 1;
            let inv_count = 1.0f32 / count as f32;

            let retain_block_off = cache.q4_block_offset(*retain_pos, h, blocks_per_pos);

            // Dequantize evict tokens upfront to avoid repeated borrow
            let evict_f32s: Vec<(Vec<f32>, f32)> = evicted_list
                .iter()
                .map(|&(ep, sim)| {
                    let mut buf = vec![0.0f32; head_dim];
                    dequantize_k(cache, ep, h, head_dim, &mut buf);
                    (buf, sim)
                })
                .collect();

            // Also dequantize the V evict tokens
            let evict_v_f32s: Vec<(Vec<f32>, f32)> = evicted_list
                .iter()
                .map(|&(ep, sim)| {
                    let mut buf = vec![0.0f32; head_dim];
                    let v_off = cache.offset(ep, h); // for V we need offset, but V may be F32 — check dtype
                    // V buffer for Q4_0 uses same dtype
                    match cache.v_buffer.dtype() {
                        DType::Q4_0 => {
                            let v = cache.v_buffer.as_slice::<BlockQ4_0>();
                            let ep_block_off = cache.q4_block_offset(ep, h, blocks_per_pos);
                            for bi in 0..blocks_per_pos {
                                let mut tmp = [0.0f32; QK4_0];
                                v[ep_block_off + bi].dequantize(&mut tmp);
                                let dst = bi * QK4_0;
                                buf[dst..dst + QK4_0].copy_from_slice(&tmp);
                            }
                        }
                        DType::F32 => {
                            let v = cache.v_buffer.as_slice::<f32>();
                            buf.copy_from_slice(&v[v_off..v_off + head_dim]);
                        }
                        DType::F16 => {
                            let v = cache.v_buffer.as_slice::<f16>();
                            for d in 0..head_dim {
                                buf[d] = v[v_off + d].to_f32();
                            }
                        }
                        _ => {}
                    }
                    (buf, sim)
                })
                .collect();

            // K merge
            {
                let k = cache.k_buffer.as_mut_slice::<BlockQ4_0>();
                for bi in 0..blocks_per_pos {
                    let mut r_f32 = [0.0f32; QK4_0];
                    k[retain_block_off + bi].dequantize(&mut r_f32);
                    for (e_buf, sim) in &evict_f32s {
                        let base = bi * QK4_0;
                        for i in 0..QK4_0 {
                            r_f32[i] += sim * e_buf[base + i];
                        }
                    }
                    for i in 0..QK4_0 {
                        r_f32[i] *= inv_count;
                    }
                    k[retain_block_off + bi] = BlockQ4_0::quantize(&r_f32);
                }
            }

            // V merge
            let retain_v_block_off = cache.q4_block_offset(*retain_pos, h, blocks_per_pos);
            match cache.v_buffer.dtype() {
                DType::Q4_0 => {
                    let v = cache.v_buffer.as_mut_slice::<BlockQ4_0>();
                    for bi in 0..blocks_per_pos {
                        let mut r_f32 = [0.0f32; QK4_0];
                        v[retain_v_block_off + bi].dequantize(&mut r_f32);
                        for (e_buf, sim) in &evict_v_f32s {
                            let base = bi * QK4_0;
                            for i in 0..QK4_0 {
                                r_f32[i] += sim * e_buf[base + i];
                            }
                        }
                        for i in 0..QK4_0 {
                            r_f32[i] *= inv_count;
                        }
                        v[retain_v_block_off + bi] = BlockQ4_0::quantize(&r_f32);
                    }
                }
                DType::F32 => {
                    let retain_v_off = cache.offset(*retain_pos, h);
                    let v = cache.v_buffer.as_mut_slice::<f32>();
                    for d in 0..head_dim {
                        let mut acc = v[retain_v_off + d];
                        for (e_buf, sim) in &evict_v_f32s {
                            acc += sim * e_buf[d];
                        }
                        v[retain_v_off + d] = acc * inv_count;
                    }
                }
                DType::F16 => {
                    let retain_v_off = cache.offset(*retain_pos, h);
                    let v = cache.v_buffer.as_mut_slice::<f16>();
                    for d in 0..head_dim {
                        let mut acc = v[retain_v_off + d].to_f32();
                        for (e_buf, sim) in &evict_v_f32s {
                            acc += sim * e_buf[d];
                        }
                        v[retain_v_off + d] = f16::from_f32(acc * inv_count);
                    }
                }
                _ => {}
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
        // Pass shape as [1, max_seq, kv_heads, head_dim] so KVCache::new
        // correctly reads kv_heads from shape[2] and head_dim from shape[3].
        // The HeadMajor layout flag controls how offset() computes addresses.
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

    // ── find_nearest_per_head tests ──

    #[test]
    fn test_find_nearest_per_head_seq_major() {
        let mut cache = make_cache(10, 1, 4, 5);

        // Write distinct K vectors
        write_k(&mut cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]); // pos 0
        write_k(&mut cache, 1, 0, &[0.0, 1.0, 0.0, 0.0]); // pos 1
        write_k(&mut cache, 2, 0, &[0.9, 0.1, 0.0, 0.0]); // pos 2 — similar to pos 0
        write_k(&mut cache, 3, 0, &[0.0, 0.0, 1.0, 0.0]); // pos 3
        write_k(&mut cache, 4, 0, &[0.0, 0.0, 0.0, 1.0]); // pos 4

        let retain = vec![0, 1, 3, 4];
        let m = find_nearest_per_head(&cache, 2, &retain, 1, 4);

        assert_eq!(
            m.per_head[0].0, 0,
            "pos 2 should be nearest to pos 0 at head 0"
        );
        assert!(
            m.per_head[0].1 > 0.9,
            "similarity should be high, got {}",
            m.per_head[0].1
        );
        assert!(
            m.mean_sim > 0.9,
            "mean_sim should match single head, got {}",
            m.mean_sim
        );
    }

    #[test]
    fn test_find_nearest_per_head_head_major() {
        let mut cache = make_cache_head_major(10, 2, 4, 5);

        // Head 0
        write_k(&mut cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]);
        write_k(&mut cache, 1, 0, &[0.0, 1.0, 0.0, 0.0]);
        write_k(&mut cache, 2, 0, &[0.95, 0.05, 0.0, 0.0]); // similar to pos 0
        // Head 1
        write_k(&mut cache, 0, 1, &[0.0, 0.0, 1.0, 0.0]);
        write_k(&mut cache, 1, 1, &[0.0, 0.0, 0.0, 1.0]);
        write_k(&mut cache, 2, 1, &[0.0, 0.0, 0.9, 0.1]); // similar to pos 0 in head 1

        let retain = vec![0, 1];
        let m = find_nearest_per_head(&cache, 2, &retain, 2, 4);
        assert_eq!(m.per_head[0].0, 0, "head 0: pos 2 → pos 0");
        assert_eq!(m.per_head[1].0, 0, "head 1: pos 2 → pos 0");
    }

    #[test]
    fn test_per_head_different_targets() {
        // 2 heads where each head's nearest neighbor differs
        let mut cache = make_cache(10, 2, 4, 4);

        // head 0: evict_pos(2) is most similar to pos 0
        write_k(&mut cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]);
        write_k(&mut cache, 1, 0, &[0.0, 1.0, 0.0, 0.0]);
        write_k(&mut cache, 2, 0, &[0.95, 0.05, 0.0, 0.0]); // evict

        // head 1: evict_pos(2) is most similar to pos 1
        write_k(&mut cache, 0, 1, &[0.0, 1.0, 0.0, 0.0]);
        write_k(&mut cache, 1, 1, &[1.0, 0.0, 0.0, 0.0]);
        write_k(&mut cache, 2, 1, &[0.95, 0.05, 0.0, 0.0]); // evict — similar to pos 1 in this head

        let retain = vec![0, 1];
        let m = find_nearest_per_head(&cache, 2, &retain, 2, 4);

        // head 0: pos 2 → pos 0 (both have high [1,0,0,0] component)
        assert_eq!(m.per_head[0].0, 0, "head 0 should match pos 0");
        // head 1: pos 2 = [0.95,0.05,...], pos 1 = [1.0,0,...] → more similar than pos 0=[0,1,0,0]
        assert_eq!(m.per_head[1].0, 1, "head 1 should match pos 1");
        // Heads have different targets — this is the key assertion
        assert_ne!(
            m.per_head[0].0, m.per_head[1].0,
            "per-head targets should differ"
        );
    }

    // ── scatter_reduce_merge_per_head tests ──

    #[test]
    fn test_scatter_reduce_1to1() {
        // Single evict → single retain: K[j] = (K[j] + sim * K[e]) / 2
        let mut cache = make_cache(10, 1, 4, 3);

        // retain at pos 0, evict at pos 1
        write_k(&mut cache, 0, 0, &[1.0, 2.0, 3.0, 4.0]); // retain
        write_k(&mut cache, 1, 0, &[5.0, 6.0, 7.0, 8.0]); // evict
        write_v(&mut cache, 0, 0, &[10.0, 20.0, 30.0, 40.0]); // retain V
        write_v(&mut cache, 1, 0, &[50.0, 60.0, 70.0, 80.0]); // evict V

        let sim = 0.8f32;
        let match0 = PerHeadMatch {
            per_head: vec![(0, sim)],
            mean_sim: sim,
        };
        let passing = vec![1usize]; // evict pos 1
        let matches: Vec<&PerHeadMatch> = vec![&match0];

        scatter_reduce_merge_per_head(&mut cache, &passing, &matches, 1, 4);

        let k = read_k(&cache, 0, 0, 4);
        let v = read_v(&cache, 0, 0, 4);

        // Expected: (K[retain] + sim * K[evict]) / 2
        let expected_k: Vec<f32> = vec![
            (1.0 + 0.8 * 5.0) / 2.0,
            (2.0 + 0.8 * 6.0) / 2.0,
            (3.0 + 0.8 * 7.0) / 2.0,
            (4.0 + 0.8 * 8.0) / 2.0,
        ];
        let expected_v: Vec<f32> = vec![
            (10.0 + 0.8 * 50.0) / 2.0,
            (20.0 + 0.8 * 60.0) / 2.0,
            (30.0 + 0.8 * 70.0) / 2.0,
            (40.0 + 0.8 * 80.0) / 2.0,
        ];

        for d in 0..4 {
            assert!(
                (k[d] - expected_k[d]).abs() < 1e-5,
                "K[{d}]: got {}, expected {}",
                k[d],
                expected_k[d]
            );
            assert!(
                (v[d] - expected_v[d]).abs() < 1e-5,
                "V[{d}]: got {}, expected {}",
                v[d],
                expected_v[d]
            );
        }
    }

    #[test]
    fn test_scatter_reduce_nto1() {
        // Two evicts → same retain: K[j] = (K[j] + s1*K[e1] + s2*K[e2]) / 3
        let mut cache = make_cache(10, 1, 4, 4);

        write_k(&mut cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]); // retain pos 0
        write_k(&mut cache, 1, 0, &[0.0, 1.0, 0.0, 0.0]); // evict pos 1
        write_k(&mut cache, 2, 0, &[0.0, 0.0, 1.0, 0.0]); // evict pos 2

        let s1 = 0.6f32;
        let s2 = 0.4f32;

        // Both evicts → retain pos 0
        let match1 = PerHeadMatch {
            per_head: vec![(0, s1)],
            mean_sim: s1,
        };
        let match2 = PerHeadMatch {
            per_head: vec![(0, s2)],
            mean_sim: s2,
        };
        let passing = vec![1usize, 2usize];
        let matches: Vec<&PerHeadMatch> = vec![&match1, &match2];

        scatter_reduce_merge_per_head(&mut cache, &passing, &matches, 1, 4);

        let k = read_k(&cache, 0, 0, 4);

        // Expected: (K[0] + s1*K[1] + s2*K[2]) / 3
        // K[0]=[1,0,0,0], K[1]=[0,1,0,0], K[2]=[0,0,1,0]
        let expected_k = vec![
            (1.0 + 0.6 * 0.0 + 0.4 * 0.0) / 3.0, // = 1/3
            (0.0 + 0.6 * 1.0 + 0.4 * 0.0) / 3.0, // = 0.2
            (0.0 + 0.6 * 0.0 + 0.4 * 1.0) / 3.0, // ≈ 0.133
            0.0f32,
        ];

        for d in 0..4 {
            assert!(
                (k[d] - expected_k[d]).abs() < 1e-5,
                "K[{d}]: got {}, expected {}",
                k[d],
                expected_k[d]
            );
        }
    }

    #[test]
    fn test_scatter_reduce_no_mapping_unchanged() {
        // Retain pos 1 is never a merge target — its values must not change
        let mut cache = make_cache(10, 1, 4, 3);

        write_k(&mut cache, 0, 0, &[1.0, 0.0, 0.0, 0.0]); // retain pos 0 (merge target)
        write_k(&mut cache, 1, 0, &[9.9, 9.9, 9.9, 9.9]); // retain pos 1 (NOT merge target)
        write_k(&mut cache, 2, 0, &[0.9, 0.1, 0.0, 0.0]); // evict → maps to pos 0

        let sim = 0.99f32;
        let match0 = PerHeadMatch {
            per_head: vec![(0, sim)], // evict 2 → retain 0, NOT retain 1
            mean_sim: sim,
        };
        let passing = vec![2usize];
        let matches: Vec<&PerHeadMatch> = vec![&match0];

        scatter_reduce_merge_per_head(&mut cache, &passing, &matches, 1, 4);

        // pos 1 must be unchanged
        let k1 = read_k(&cache, 1, 0, 4);
        for d in 0..4 {
            assert!(
                (k1[d] - 9.9).abs() < 1e-5,
                "unmapped retain pos 1 K[{d}] should be unchanged: got {}",
                k1[d]
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
            ema_alpha: 0.5,
            ema_beta: 0.5,
            use_layer_allocation: false,
            protected_layers: vec![],
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
            qcf_sink: None,
            layer_ratios: None,
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
    fn test_ema_init_no_double_update() {
        // First call: threshold = mean(sims), no EMA update applied on top
        // We test this by calling evict_and_merge with controlled sims
        let handler = D2OHandler::new(D2OConfig::default());
        let mut state = D2OState::new();
        let mut cache = make_cache(20, 1, 4, 12);

        // Write identical vectors → cosine sim = 1.0 for all pairs
        for pos in 0..12 {
            write_k(&mut cache, pos, 0, &[1.0, 0.0, 0.0, 0.0]);
        }
        let importance: Vec<f32> = (0..12).map(|i| 12.0 - i as f32).collect();

        handler
            .evict_and_merge(&mut cache, 6, &importance, &mut state)
            .unwrap();

        // All sims ≈ 1.0 → mean = 1.0 → threshold = 1.0 (no EMA on first call)
        assert!(state.initialized, "should be initialized");
        // With identical vectors sim = 1.0, threshold should be set to that mean
        assert!(
            state.ema_threshold > 0.5,
            "threshold should reflect mean sim ≈ 1.0, got {}",
            state.ema_threshold
        );
    }

    #[test]
    fn test_ema_update_uses_mean() {
        // Second call applies α·τ + β·mean(sims)
        let alpha = 0.5f32;
        let beta = 0.5f32;
        let initial_threshold = 0.8f32;
        let mean_sim = 0.4f32;

        let expected = alpha * initial_threshold + beta * mean_sim; // = 0.6

        let mut threshold = initial_threshold;
        threshold = alpha * threshold + beta * mean_sim;

        assert!(
            (threshold - expected).abs() < 1e-6,
            "EMA update: got {threshold}, expected {expected}"
        );
    }

    #[test]
    fn test_ema_update() {
        let alpha: f32 = 0.5;
        let beta: f32 = 0.5;
        let mut threshold: f32 = 0.5; // initial

        // Update with new mean similarity = 0.8
        threshold = alpha * threshold + beta * 0.8;
        let expected = 0.5 * 0.5 + 0.5 * 0.8; // = 0.25 + 0.4 = 0.65
        assert!(
            (threshold - expected).abs() < 1e-6,
            "threshold={threshold}, expected={expected}"
        );

        // Update with lower mean similarity = 0.3
        threshold = alpha * threshold + beta * 0.3;
        let expected2 = 0.5 * expected + 0.5 * 0.3;
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
            qcf_sink: None,
            layer_ratios: None,
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
            qcf_sink: None,
            layer_ratios: None,
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
            ema_alpha: 0.5,
            ema_beta: 0.5,
            use_layer_allocation: false,
            protected_layers: vec![],
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
            qcf_sink: None,
            layer_ratios: None,
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
            qcf_sink: None,
            layer_ratios: None,
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
            ema_alpha: 0.5,
            ema_beta: 0.5,
            use_layer_allocation: false,
            protected_layers: vec![],
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
            qcf_sink: None,
            layer_ratios: None,
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
            ema_alpha: 0.5,
            ema_beta: 0.5,
            use_layer_allocation: false,
            protected_layers: vec![],
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
            qcf_sink: None,
            layer_ratios: None,
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
            ema_alpha: 0.5,
            ema_beta: 0.5,
            use_layer_allocation: false,
            protected_layers: vec![],
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
            qcf_sink: None,
            layer_ratios: None,
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
    fn test_q4_find_nearest_per_head() {
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
        let m = find_nearest_per_head(&cache, 2, &retain, 1, 32);
        assert_eq!(m.per_head[0].0, 0, "pos 2 should be nearest to pos 0");
        assert!(
            m.per_head[0].1 > 0.9,
            "similarity should be high, got {}",
            m.per_head[0].1
        );
    }

    #[test]
    fn test_q4_scatter_reduce() {
        let mut cache = make_cache_q4(10, 1, 32, 3);
        let v_retain: Vec<f32> = (0..32).map(|i| 2.0 + i as f32 * 0.1).collect();
        let v_evict: Vec<f32> = (0..32).map(|i| 4.0 + i as f32 * 0.1).collect();
        write_k_q4(&mut cache, 0, 0, &v_retain);
        write_k_q4(&mut cache, 1, 0, &v_evict);

        // sim = 1.0 → K[retain] = (K[retain] + 1.0 * K[evict]) / 2 ≈ average
        let sim = 1.0f32;
        let match0 = PerHeadMatch {
            per_head: vec![(0, sim)],
            mean_sim: sim,
        };
        let passing = vec![1usize];
        let matches: Vec<&PerHeadMatch> = vec![&match0];

        scatter_reduce_merge_per_head(&mut cache, &passing, &matches, 1, 32);

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
            ema_alpha: 0.5,
            ema_beta: 0.5,
            use_layer_allocation: false,
            protected_layers: vec![],
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
            qcf_sink: None,
            layer_ratios: None,
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

    // ── Layout coverage tests ──

    #[test]
    fn test_handler_evicts_head_major() {
        // Full handler integration with HeadMajor layout.
        let handler = D2OHandler::new(D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            ema_alpha: 0.5,
            ema_beta: 0.5,
            use_layer_allocation: false,
            protected_layers: vec![],
        });

        let mut cache = make_cache_head_major(50, 2, 4, 30);
        for pos in 0..30 {
            let angle = pos as f32 * 0.2;
            for h in 0..2 {
                write_k(
                    &mut cache,
                    pos,
                    h,
                    &[angle.cos(), angle.sin(), h as f32 * 0.1, 0.0],
                );
                write_v(&mut cache, pos, h, &[pos as f32, h as f32, 0.0, 0.0]);
            }
        }

        let mut importance = vec![0.1; 50];
        importance[..5].fill(10.0);
        importance[25..30].fill(5.0);

        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
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
                assert_eq!(new_pos, 15, "should reduce to target (30*0.5)");
            }
            other => panic!("Expected Evicted, got {:?}", other),
        }

        // Verify prefix tokens preserved for both heads
        // pos=0: angle = 0.0, so K[0] = cos(0.0) = 1.0
        for h in 0..2 {
            let k0 = read_k(&ctx.caches[0], 0, h, 4);
            assert!(
                (k0[0] - 1.0).abs() < 1e-4,
                "head {h} prefix pos 0 K[0] should be cos(0)=1.0, got {}",
                k0[0]
            );
        }
    }

    #[test]
    fn test_handler_layout_equivalence() {
        // SeqMajor and HeadMajor should produce identical eviction results
        // (same tokens_removed, same new_pos) given the same logical data.
        let make_config = || D2OConfig {
            keep_ratio: 0.75,
            protected_prefix: 2,
            target_ratio: 0.5,
            ema_alpha: 0.5,
            ema_beta: 0.5,
            use_layer_allocation: false,
            protected_layers: vec![],
        };

        let mut importance = vec![0.1; 50];
        importance[..5].fill(10.0);
        importance[15..20].fill(5.0);

        // Populate identical logical content into both layouts
        let populate = |cache: &mut KVCache| {
            for pos in 0..20 {
                let angle = pos as f32 * 0.3;
                write_k(cache, pos, 0, &[angle.cos(), angle.sin(), 0.0, 0.0]);
                write_v(cache, pos, 0, &[pos as f32, 0.0, 0.0, 0.0]);
            }
        };

        // --- SeqMajor ---
        let handler_seq = D2OHandler::new(make_config());
        let mut cache_seq = make_cache(50, 1, 4, 20);
        populate(&mut cache_seq);
        let mut caches_seq = vec![cache_seq];
        let mut ctx_seq = HandlerContext {
            caches: &mut caches_seq,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };
        let result_seq = handler_seq.handle(&mut ctx_seq).unwrap();

        // --- HeadMajor ---
        let handler_hm = D2OHandler::new(make_config());
        let mut cache_hm = make_cache_head_major(50, 1, 4, 20);
        populate(&mut cache_hm);
        let mut caches_hm = vec![cache_hm];
        let mut ctx_hm = HandlerContext {
            caches: &mut caches_hm,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };
        let result_hm = handler_hm.handle(&mut ctx_hm).unwrap();

        // Both should produce Evicted with matching stats
        match (&result_seq, &result_hm) {
            (
                ActionResult::Evicted {
                    tokens_removed: rem_s,
                    new_pos: pos_s,
                },
                ActionResult::Evicted {
                    tokens_removed: rem_h,
                    new_pos: pos_h,
                },
            ) => {
                assert_eq!(rem_s, rem_h, "tokens_removed must match across layouts");
                assert_eq!(pos_s, pos_h, "new_pos must match across layouts");
            }
            _ => panic!(
                "Expected Evicted for both, got seq={:?} hm={:?}",
                result_seq, result_hm
            ),
        }

        // Verify the compacted K values match logically
        let final_pos = ctx_seq.caches[0].current_pos;
        for pos in 0..final_pos {
            let k_seq = read_k(&ctx_seq.caches[0], pos, 0, 4);
            let k_hm = read_k(&ctx_hm.caches[0], pos, 0, 4);
            for d in 0..4 {
                assert!(
                    (k_seq[d] - k_hm[d]).abs() < 1e-4,
                    "K mismatch at pos={pos} dim={d}: seq={} hm={}",
                    k_seq[d],
                    k_hm[d],
                );
            }
        }
    }
}
