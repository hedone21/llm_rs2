//! Unified QCF (Quality Cost Function) metric for all KV cache actions.
//!
//! Core formula:
//!   QCF = ||O_before - O_after|| / ||O_before||
//!   O = sum_t alpha_t * V_t   (attention-weighted value sum)
//!
//! Supports: sliding eviction, H2O eviction, StreamingLLM, D2O merge,
//! and KIVI quantization -- all measured in the same output-error space.

use super::{AggregationMode, aggregate_heads};
use crate::core::kv_cache::KVLayout;
use crate::core::quant::{BlockKVQ4, BlockKVQ8, BlockQ2_0, QKKV};

// ── Action types ────────────────────────────────────────────────

/// Describes which lossy action to simulate for QCF computation.
#[derive(Debug, Clone)]
pub enum QcfActionType {
    /// Sliding window: retain the last `target_len` tokens.
    EvictSliding { target_len: usize },
    /// H2O: importance-based eviction with prefix protection.
    EvictH2o {
        target_len: usize,
        keep_ratio: f32,
        protected_prefix: usize,
    },
    /// StreamingLLM: retain first `sink_size` + last `window_size` tokens.
    EvictStreaming {
        sink_size: usize,
        window_size: usize,
    },
    /// D2O merge: same retained set as H2O (merge compensation is secondary).
    MergeD2o {
        target_len: usize,
        keep_ratio: f32,
        protected_prefix: usize,
    },
    /// KIVI quantization round-trip error.
    QuantKivi { bits: u8 },
}

// ── V data source abstraction ───────────────────────────────────

/// Abstraction over V buffer data types for read-only access.
pub enum VDataSource<'a> {
    /// F32 KV cache data.
    F32(&'a [f32]),
    /// F16 KV cache data stored as raw u16 (half::f16 bit representation).
    F16(&'a [u16]),
}

// ── Parameters ──────────────────────────────────────────────────

/// All inputs needed to compute the unified QCF metric.
pub struct UnifiedQcfParams<'a> {
    /// The action to simulate.
    pub action: QcfActionType,
    /// V buffer data (F32 or F16).
    pub v_source: VDataSource<'a>,
    /// Flat importance scores, layout `[max_seq_len]`.
    pub attention_scores: &'a [f32],
    /// Optional per-KV-head attention, layout `[n_kv_heads * max_seq_len]`.
    pub head_attn: Option<&'a [f32]>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub current_pos: usize,
    pub capacity: usize,
    pub layout: KVLayout,
    pub aggregation: AggregationMode,
}

// ── Main entry point ────────────────────────────────────────────

/// Compute unified QCF for the given action.
///
/// Returns `(aggregated_qcf, per_head_qcf)`.
pub fn compute_unified_qcf(params: &UnifiedQcfParams) -> (f32, Vec<f32>) {
    let n_kv_heads = params.n_kv_heads;
    let head_dim = params.head_dim;
    let current_pos = params.current_pos;
    let capacity = params.capacity;
    let layout = params.layout;

    if n_kv_heads == 0 || head_dim == 0 || current_pos == 0 {
        return (0.0, vec![0.0; n_kv_heads]);
    }

    let max_seq_len = params.attention_scores.len();
    let mut per_head = vec![0.0f32; n_kv_heads];

    for (h, ph) in per_head.iter_mut().enumerate() {
        // 1. Get alpha_h[t] for this KV-head
        let alpha_h: Vec<f32> = if let Some(head_attn) = params.head_attn {
            let head_offset = h * (head_attn.len() / n_kv_heads.max(1));
            (0..current_pos)
                .map(|t| {
                    let idx = head_offset + t;
                    if idx < head_attn.len() {
                        head_attn[idx]
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            (0..current_pos)
                .map(|t| {
                    if t < max_seq_len {
                        params.attention_scores[t]
                    } else {
                        0.0
                    }
                })
                .collect()
        };

        // 2. Compute O_before = sum alpha_h[t] * V[h][t]
        let mut o_before = vec![0.0f32; head_dim];
        for (t, &alpha_t) in alpha_h.iter().enumerate().take(current_pos) {
            let v_t = read_v_f32(
                &params.v_source,
                h,
                t,
                head_dim,
                capacity,
                n_kv_heads,
                layout,
            );
            for d in 0..head_dim {
                o_before[d] += alpha_t * v_t[d];
            }
        }

        // 3. Compute O_after based on action type
        let o_after = match &params.action {
            QcfActionType::EvictSliding { target_len } => {
                if current_pos <= *target_len {
                    o_before.clone()
                } else {
                    let retained_start = current_pos.saturating_sub(*target_len);
                    compute_o_eviction(
                        &alpha_h,
                        &params.v_source,
                        h,
                        (retained_start..current_pos)
                            .collect::<Vec<_>>()
                            .iter()
                            .copied(),
                        head_dim,
                        capacity,
                        n_kv_heads,
                        layout,
                    )
                }
            }
            QcfActionType::EvictH2o {
                target_len,
                keep_ratio,
                protected_prefix,
            } => {
                if current_pos <= *target_len {
                    o_before.clone()
                } else {
                    let retained = identify_retained_h2o(
                        &alpha_h,
                        current_pos,
                        *target_len,
                        *keep_ratio,
                        *protected_prefix,
                    );
                    compute_o_eviction(
                        &alpha_h,
                        &params.v_source,
                        h,
                        retained.iter().copied(),
                        head_dim,
                        capacity,
                        n_kv_heads,
                        layout,
                    )
                }
            }
            QcfActionType::EvictStreaming {
                sink_size,
                window_size,
            } => {
                let keep_size = sink_size + window_size;
                if current_pos <= keep_size {
                    o_before.clone()
                } else {
                    let retained: Vec<usize> = (0..*sink_size)
                        .chain((current_pos - window_size)..current_pos)
                        .collect();
                    compute_o_eviction(
                        &alpha_h,
                        &params.v_source,
                        h,
                        retained.iter().copied(),
                        head_dim,
                        capacity,
                        n_kv_heads,
                        layout,
                    )
                }
            }
            QcfActionType::MergeD2o {
                target_len,
                keep_ratio,
                protected_prefix,
            } => {
                if current_pos <= *target_len {
                    o_before.clone()
                } else {
                    let retained = identify_retained_h2o(
                        &alpha_h,
                        current_pos,
                        *target_len,
                        *keep_ratio,
                        *protected_prefix,
                    );
                    compute_o_eviction(
                        &alpha_h,
                        &params.v_source,
                        h,
                        retained.iter().copied(),
                        head_dim,
                        capacity,
                        n_kv_heads,
                        layout,
                    )
                }
            }
            QcfActionType::QuantKivi { bits } => {
                let mut o_after = vec![0.0f32; head_dim];
                for (t, &alpha_t) in alpha_h.iter().enumerate().take(current_pos) {
                    let v_t = read_v_f32(
                        &params.v_source,
                        h,
                        t,
                        head_dim,
                        capacity,
                        n_kv_heads,
                        layout,
                    );
                    let v_quant = quantize_dequantize_f32(&v_t, *bits);
                    for d in 0..head_dim {
                        o_after[d] += alpha_t * v_quant[d];
                    }
                }
                o_after
            }
        };

        // 4. QCF = ||O_before - O_after|| / ||O_before||
        let diff_norm = l2_norm_diff(&o_before, &o_after);
        let o_norm = l2_norm(&o_before);
        *ph = if o_norm > 1e-10 {
            diff_norm / o_norm
        } else {
            0.0
        };
    }

    let qcf = aggregate_heads(&per_head, &params.aggregation);
    (qcf, per_head)
}

// ── Helper: read V vector as f32 ────────────────────────────────

fn read_v_f32(
    src: &VDataSource,
    head: usize,
    pos: usize,
    head_dim: usize,
    capacity: usize,
    n_kv_heads: usize,
    layout: KVLayout,
) -> Vec<f32> {
    let offset = compute_v_offset(layout, head, pos, head_dim, capacity, n_kv_heads);
    match src {
        VDataSource::F32(data) => {
            let end = (offset + head_dim).min(data.len());
            if offset >= data.len() {
                return vec![0.0; head_dim];
            }
            data[offset..end].to_vec()
        }
        VDataSource::F16(data) => {
            let end = (offset + head_dim).min(data.len());
            if offset >= data.len() {
                return vec![0.0; head_dim];
            }
            data[offset..end]
                .iter()
                .map(|&bits| half::f16::from_bits(bits).to_f32())
                .collect()
        }
    }
}

fn compute_v_offset(
    layout: KVLayout,
    head: usize,
    pos: usize,
    head_dim: usize,
    capacity: usize,
    n_kv_heads: usize,
) -> usize {
    match layout {
        KVLayout::HeadMajor => head * capacity * head_dim + pos * head_dim,
        KVLayout::SeqMajor => pos * n_kv_heads * head_dim + head * head_dim,
    }
}

// ── Helper: eviction O_after with softmax redistribution ────────

#[allow(clippy::too_many_arguments)]
fn compute_o_eviction(
    alpha: &[f32],
    v_src: &VDataSource,
    head: usize,
    retained: impl Iterator<Item = usize>,
    head_dim: usize,
    capacity: usize,
    n_kv_heads: usize,
    layout: KVLayout,
) -> Vec<f32> {
    let retained: Vec<usize> = retained.collect();
    let alpha_sum: f32 = retained.iter().map(|&t| alpha[t]).sum();
    if alpha_sum <= 0.0 {
        return vec![0.0; head_dim];
    }

    let mut o = vec![0.0f32; head_dim];
    for &t in &retained {
        let w = alpha[t] / alpha_sum; // redistributed attention
        let v_t = read_v_f32(v_src, head, t, head_dim, capacity, n_kv_heads, layout);
        for d in 0..head_dim {
            o[d] += w * v_t[d];
        }
    }
    o
}

// ── Helper: H2O retained token identification ───────────────────

fn identify_retained_h2o(
    importance: &[f32],
    current_pos: usize,
    target_len: usize,
    keep_ratio: f32,
    protected_prefix: usize,
) -> Vec<usize> {
    let prefix = protected_prefix.min(current_pos).min(target_len);
    let available = target_len.saturating_sub(prefix);
    if available == 0 {
        return (0..prefix).collect();
    }

    let hh_budget = (available as f32 * keep_ratio) as usize;
    let recent_budget = available.saturating_sub(hh_budget);
    let recent_start = current_pos.saturating_sub(recent_budget);

    // Protected prefix
    let mut retained: Vec<usize> = (0..prefix).collect();

    // Heavy hitters from evictable zone [prefix..recent_start]
    if recent_start > prefix {
        let mut evictable: Vec<(usize, f32)> = (prefix..recent_start)
            .map(|t| {
                let score = if t < importance.len() {
                    importance[t]
                } else {
                    0.0
                };
                (t, score)
            })
            .collect();
        evictable.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        retained.extend(evictable.iter().take(hh_budget).map(|(t, _)| t));
    }

    // Recent window
    retained.extend(recent_start..current_pos);
    retained.sort();
    retained.dedup();
    retained
}

// ── Helper: quantize-dequantize round trip ──────────────────────

fn quantize_dequantize_f32(data: &[f32], bits: u8) -> Vec<f32> {
    let mut result = vec![0.0f32; data.len()];
    for chunk_start in (0..data.len()).step_by(QKKV) {
        let end = (chunk_start + QKKV).min(data.len());
        let chunk_len = end - chunk_start;
        let mut block = [0.0f32; QKKV];
        block[..chunk_len].copy_from_slice(&data[chunk_start..end]);

        let mut reconstructed = [0.0f32; QKKV];
        match bits {
            2 => {
                let q = BlockQ2_0::quantize(&block);
                q.dequantize(&mut reconstructed);
            }
            4 => {
                let q = BlockKVQ4::quantize(&block);
                q.dequantize(&mut reconstructed);
            }
            8 => {
                let q = BlockKVQ8::quantize(&block);
                q.dequantize(&mut reconstructed);
            }
            _ => {
                // F16/F32: no quantization error
                reconstructed = block;
            }
        }
        result[chunk_start..end].copy_from_slice(&reconstructed[..chunk_len]);
    }
    result
}

// ── Math helpers ────────────────────────────────────────────────

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn l2_norm_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple HeadMajor V buffer for testing.
    /// V[h][t][d] = (h+1) * (t+1) * (d+1) as f32, giving predictable values.
    fn make_v_data(n_kv_heads: usize, capacity: usize, head_dim: usize) -> Vec<f32> {
        let total = n_kv_heads * capacity * head_dim;
        let mut data = vec![0.0f32; total];
        for h in 0..n_kv_heads {
            for t in 0..capacity {
                for d in 0..head_dim {
                    let offset = h * capacity * head_dim + t * head_dim + d;
                    data[offset] = (h as f32 + 1.0) * (t as f32 + 1.0) * (d as f32 + 1.0);
                }
            }
        }
        data
    }

    /// Uniform attention scores for testing.
    fn uniform_scores(n: usize) -> Vec<f32> {
        vec![1.0 / n as f32; n]
    }

    #[test]
    fn test_zero_change_sliding() {
        // target_len == current_pos -> nothing evicted -> QCF = 0
        let n_kv_heads = 2;
        let head_dim = 4;
        let capacity = 16;
        let current_pos = 8;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let params = UnifiedQcfParams {
            action: QcfActionType::EvictSliding {
                target_len: current_pos,
            },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let (qcf, per_head) = compute_unified_qcf(&params);
        assert!(
            qcf.abs() < 1e-6,
            "expected QCF=0 when nothing evicted, got {qcf}"
        );
        for (h, &v) in per_head.iter().enumerate() {
            assert!(v.abs() < 1e-6, "head {h}: expected 0, got {v}");
        }
    }

    #[test]
    fn test_full_eviction() {
        // target_len = 0 -> everything evicted -> QCF should be high
        let n_kv_heads = 2;
        let head_dim = 4;
        let capacity = 16;
        let current_pos = 8;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let params = UnifiedQcfParams {
            action: QcfActionType::EvictSliding { target_len: 0 },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let (qcf, _) = compute_unified_qcf(&params);
        // With 0 retained tokens, O_after = 0, so QCF = ||O_before|| / ||O_before|| = 1.0
        assert!(
            (qcf - 1.0).abs() < 1e-5,
            "expected QCF near 1.0 for full eviction, got {qcf}"
        );
    }

    #[test]
    fn test_quant_lossless() {
        // bits >= 16 -> no quantization applied -> QCF = 0
        let n_kv_heads = 1;
        let head_dim = 32; // must be QKKV-aligned
        let capacity = 4;
        let current_pos = 2;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let params = UnifiedQcfParams {
            action: QcfActionType::QuantKivi { bits: 16 },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let (qcf, _) = compute_unified_qcf(&params);
        assert!(
            qcf.abs() < 1e-6,
            "expected QCF=0 for lossless quant, got {qcf}"
        );
    }

    #[test]
    fn test_quant_ordering() {
        // Q2 > Q4 > Q8 in terms of QCF (more lossy = higher error)
        let n_kv_heads = 1;
        let head_dim = 32;
        let capacity = 8;
        let current_pos = 4;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let make_params = |bits: u8| UnifiedQcfParams {
            action: QcfActionType::QuantKivi { bits },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let (qcf_q2, _) = compute_unified_qcf(&make_params(2));
        let (qcf_q4, _) = compute_unified_qcf(&make_params(4));
        let (qcf_q8, _) = compute_unified_qcf(&make_params(8));

        assert!(
            qcf_q2 > qcf_q4,
            "Q2 ({qcf_q2}) should have higher QCF than Q4 ({qcf_q4})"
        );
        assert!(
            qcf_q4 > qcf_q8,
            "Q4 ({qcf_q4}) should have higher QCF than Q8 ({qcf_q8})"
        );
        assert!(qcf_q8 > 0.0, "Q8 should have non-zero QCF, got {qcf_q8}");
    }

    #[test]
    fn test_eviction_monotonicity() {
        // More tokens evicted -> higher QCF
        let n_kv_heads = 2;
        let head_dim = 4;
        let capacity = 32;
        let current_pos = 16;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let make_params = |target_len: usize| UnifiedQcfParams {
            action: QcfActionType::EvictSliding { target_len },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let (qcf_keep_12, _) = compute_unified_qcf(&make_params(12));
        let (qcf_keep_8, _) = compute_unified_qcf(&make_params(8));
        let (qcf_keep_4, _) = compute_unified_qcf(&make_params(4));

        assert!(
            qcf_keep_4 > qcf_keep_8,
            "keeping 4 ({qcf_keep_4}) should give higher QCF than keeping 8 ({qcf_keep_8})"
        );
        assert!(
            qcf_keep_8 > qcf_keep_12,
            "keeping 8 ({qcf_keep_8}) should give higher QCF than keeping 12 ({qcf_keep_12})"
        );
    }

    #[test]
    fn test_streaming_sink_window_no_eviction() {
        // sink + window >= current_pos -> nothing evicted -> QCF = 0
        let n_kv_heads = 2;
        let head_dim = 4;
        let capacity = 16;
        let current_pos = 8;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let params = UnifiedQcfParams {
            action: QcfActionType::EvictStreaming {
                sink_size: 4,
                window_size: 4,
            },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let (qcf, _) = compute_unified_qcf(&params);
        assert!(
            qcf.abs() < 1e-6,
            "expected QCF=0 when sink+window covers all tokens, got {qcf}"
        );
    }

    #[test]
    fn test_h2o_vs_sliding() {
        // H2O should have QCF <= Sliding at same target_len when:
        // (1) scores are non-uniform with high importance on early tokens, AND
        // (2) V values for high-importance tokens are also large (correlated).
        // This tests the scenario where importance-aware eviction helps.
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 32;
        let current_pos = 16;

        // Build V data where early (important) tokens have large V norms
        // V[t][d] = (current_pos - t) * (d+1), so early tokens dominate.
        let mut v_data = vec![0.0f32; n_kv_heads * capacity * head_dim];
        for t in 0..current_pos {
            for d in 0..head_dim {
                let offset = t * head_dim + d;
                v_data[offset] = (current_pos - t) as f32 * (d as f32 + 1.0);
            }
        }

        // Non-uniform: early tokens have very high importance
        let mut scores = vec![0.1f32; current_pos];
        scores[0] = 10.0;
        scores[1] = 8.0;
        scores[2] = 6.0;
        scores[3] = 5.0;

        let target_len = 8;

        let sliding_params = UnifiedQcfParams {
            action: QcfActionType::EvictSliding { target_len },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let h2o_params = UnifiedQcfParams {
            action: QcfActionType::EvictH2o {
                target_len,
                keep_ratio: 0.5,
                protected_prefix: 0,
            },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let (qcf_sliding, _) = compute_unified_qcf(&sliding_params);
        let (qcf_h2o, _) = compute_unified_qcf(&h2o_params);

        assert!(
            qcf_h2o <= qcf_sliding + 1e-6,
            "H2O ({qcf_h2o}) should have QCF <= Sliding ({qcf_sliding}) \
             when important tokens have large V norms"
        );
    }

    #[test]
    fn test_f16_data_source() {
        // Verify F16 VDataSource works correctly
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 8;
        let current_pos = 4;

        // Create F32 data and its F16 equivalent
        let v_f32 = make_v_data(n_kv_heads, capacity, head_dim);
        let v_f16: Vec<u16> = v_f32
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let scores = uniform_scores(current_pos);

        let params_f32 = UnifiedQcfParams {
            action: QcfActionType::EvictSliding { target_len: 2 },
            v_source: VDataSource::F32(&v_f32),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let params_f16 = UnifiedQcfParams {
            action: QcfActionType::EvictSliding { target_len: 2 },
            v_source: VDataSource::F16(&v_f16),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let (qcf_f32, _) = compute_unified_qcf(&params_f32);
        let (qcf_f16, _) = compute_unified_qcf(&params_f16);

        // F16 introduces small rounding but QCF should be very close
        assert!(
            (qcf_f32 - qcf_f16).abs() < 0.05,
            "F32 ({qcf_f32}) and F16 ({qcf_f16}) QCF should be close"
        );
    }

    #[test]
    fn test_streaming_evicts_middle() {
        // StreamingLLM: retains sink + recent, evicts the middle
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 32;
        let current_pos = 16;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let params = UnifiedQcfParams {
            action: QcfActionType::EvictStreaming {
                sink_size: 2,
                window_size: 4,
            },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let (qcf, _) = compute_unified_qcf(&params);
        // 10 out of 16 tokens evicted, QCF should be significant
        assert!(
            qcf > 0.0,
            "StreamingLLM eviction should produce non-zero QCF"
        );
        assert!(qcf < 1.0, "QCF should be bounded below 1.0, got {qcf}");
    }

    #[test]
    fn test_per_head_attn_different_from_flat() {
        // When per-head attention differs, results should diverge from flat scores
        let n_kv_heads = 2;
        let head_dim = 4;
        let capacity = 16;
        let current_pos = 8;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);

        let flat_scores = uniform_scores(current_pos);

        // Per-head: head 0 focuses on early tokens, head 1 on late tokens
        let mut head_attn = vec![0.0f32; n_kv_heads * current_pos];
        for t in 0..current_pos {
            // Head 0: linearly decreasing
            head_attn[0 * current_pos + t] = (current_pos - t) as f32;
            // Head 1: linearly increasing
            head_attn[1 * current_pos + t] = (t + 1) as f32;
        }

        let params_flat = UnifiedQcfParams {
            action: QcfActionType::EvictSliding { target_len: 4 },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &flat_scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let params_head = UnifiedQcfParams {
            action: QcfActionType::EvictSliding { target_len: 4 },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &flat_scores,
            head_attn: Some(&head_attn),
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let (qcf_flat, ph_flat) = compute_unified_qcf(&params_flat);
        let (qcf_head, ph_head) = compute_unified_qcf(&params_head);

        // Per-head should yield different per-head values
        assert!(
            (ph_head[0] - ph_head[1]).abs() > 1e-3,
            "per-head attn should produce different per-head QCFs: {:?}",
            ph_head
        );
        // Overall QCF should differ
        assert!(
            (qcf_flat - qcf_head).abs() > 1e-6,
            "flat ({qcf_flat}) vs per-head ({qcf_head}) should differ"
        );
    }

    #[test]
    fn test_d2o_same_as_h2o_for_now() {
        // MergeD2o uses same retained set as H2O, so QCF should be identical
        let n_kv_heads = 1;
        let head_dim = 4;
        let capacity = 32;
        let current_pos = 16;
        let v_data = make_v_data(n_kv_heads, capacity, head_dim);
        let scores = uniform_scores(current_pos);

        let target_len = 8;
        let keep_ratio = 0.5;
        let protected_prefix = 2;

        let h2o_params = UnifiedQcfParams {
            action: QcfActionType::EvictH2o {
                target_len,
                keep_ratio,
                protected_prefix,
            },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let d2o_params = UnifiedQcfParams {
            action: QcfActionType::MergeD2o {
                target_len,
                keep_ratio,
                protected_prefix,
            },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads,
            head_dim,
            current_pos,
            capacity,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };

        let (qcf_h2o, _) = compute_unified_qcf(&h2o_params);
        let (qcf_d2o, _) = compute_unified_qcf(&d2o_params);

        assert!(
            (qcf_h2o - qcf_d2o).abs() < 1e-6,
            "D2O ({qcf_d2o}) should equal H2O ({qcf_h2o}) for now"
        );
    }

    #[test]
    fn test_empty_inputs() {
        let v_data = vec![0.0f32; 64];
        let scores = vec![1.0f32; 8];

        // Zero heads
        let params = UnifiedQcfParams {
            action: QcfActionType::EvictSliding { target_len: 4 },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads: 0,
            head_dim: 4,
            current_pos: 8,
            capacity: 16,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };
        let (qcf, per_head) = compute_unified_qcf(&params);
        assert_eq!(qcf, 0.0);
        assert!(per_head.is_empty());

        // Zero current_pos
        let params = UnifiedQcfParams {
            action: QcfActionType::EvictSliding { target_len: 4 },
            v_source: VDataSource::F32(&v_data),
            attention_scores: &scores,
            head_attn: None,
            n_kv_heads: 2,
            head_dim: 4,
            current_pos: 0,
            capacity: 16,
            layout: KVLayout::HeadMajor,
            aggregation: AggregationMode::Mean,
        };
        let (qcf, _) = compute_unified_qcf(&params);
        assert_eq!(qcf, 0.0);
    }

    #[test]
    fn test_identify_retained_h2o_basic() {
        // 10 tokens, prefix=2, target=6, keep_ratio=0.5
        // Available after prefix = 4, hh_budget=2, recent_budget=2
        let mut importance = vec![0.1f32; 10];
        importance[3] = 10.0; // high importance
        importance[5] = 8.0; // high importance

        let retained = identify_retained_h2o(&importance, 10, 6, 0.5, 2);

        // Should contain: prefix [0,1], heavy hitters [3,5], recent [8,9]
        assert!(retained.contains(&0), "prefix token 0 should be retained");
        assert!(retained.contains(&1), "prefix token 1 should be retained");
        assert!(
            retained.contains(&3),
            "high importance token 3 should be retained"
        );
        assert!(
            retained.contains(&5),
            "high importance token 5 should be retained"
        );
        assert!(retained.len() == 6, "should retain exactly 6 tokens");
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();

        let rt_8 = quantize_dequantize_f32(&data, 8);
        let rt_4 = quantize_dequantize_f32(&data, 4);
        let rt_2 = quantize_dequantize_f32(&data, 2);
        let rt_16 = quantize_dequantize_f32(&data, 16); // passthrough

        // Passthrough should be exact
        for i in 0..data.len() {
            assert!(
                (data[i] - rt_16[i]).abs() < 1e-6,
                "bits=16 should be lossless"
            );
        }

        // Error ordering: Q2 > Q4 > Q8
        let err = |rt: &[f32]| -> f32 {
            data.iter()
                .zip(rt)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        };
        assert!(err(&rt_2) > err(&rt_4), "Q2 error > Q4 error");
        assert!(err(&rt_4) > err(&rt_8), "Q4 error > Q8 error");
    }
}
