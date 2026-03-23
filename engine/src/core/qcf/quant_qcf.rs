//! Quantization proxy: estimates NMSE from KIVI flush operations.
//!
//! NMSE = MSE(X, X') / Var(X) where X' = dequantize(quantize(X)).
//! Computed inline during residual buffer flush when FP32 originals are available.

use super::{QcfConfig, QcfMetric, aggregate_heads};
use crate::core::quant::{BlockKVQ4, BlockKVQ8, BlockQ2_0, QKKV};

/// Compute NMSE for a single quantization group of QKKV (32) values.
///
/// Performs quantize → dequantize round-trip and measures normalized error.
/// Returns 0.0 if variance is below epsilon (constant-valued block).
pub fn compute_nmse_block(original: &[f32; QKKV], bits: u8, epsilon: f32) -> f32 {
    // Compute variance of original
    let mean = original.iter().sum::<f32>() / QKKV as f32;
    let var = original.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / QKKV as f32;

    if var < epsilon {
        return 0.0;
    }

    // Quantize and dequantize
    let mut reconstructed = [0.0f32; QKKV];
    match bits {
        2 => {
            let block = BlockQ2_0::quantize(original);
            block.dequantize(&mut reconstructed);
        }
        4 => {
            let block = BlockKVQ4::quantize(original);
            block.dequantize(&mut reconstructed);
        }
        8 => {
            let block = BlockKVQ8::quantize(original);
            block.dequantize(&mut reconstructed);
        }
        _ => return 0.0,
    }

    // MSE / Var
    let mse = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&o, &r)| (o - r).powi(2))
        .sum::<f32>()
        / QKKV as f32;

    (mse / var).clamp(0.0, 1.0)
}

/// Parameters for flush proxy computation.
pub struct FlushQcfParams<'a> {
    pub res_k: &'a [f32],
    pub res_v: &'a [f32],
    pub kv_heads: usize,
    pub head_dim: usize,
    pub flush_tokens: usize,
    pub res_cap: usize,
    pub bits: u8,
}

/// Compute flush proxy from FP32 residual key/value buffers.
///
/// Called during `KiviCache::flush_residual()` when FP32 originals are about to
/// be quantized. Computes NMSE separately for K and V, then combines:
/// `proxy = 0.6 × NMSE_K + 0.4 × NMSE_V` (Key is more sensitive per KIVI Table 2).
///
/// Layout: `res_k`/`res_v` are `[kv_heads][flush_tokens][head_dim]` contiguous.
pub fn compute_flush_qcf(params: &FlushQcfParams, config: &QcfConfig) -> QcfMetric {
    let FlushQcfParams {
        res_k,
        res_v,
        kv_heads,
        head_dim,
        flush_tokens,
        res_cap,
        bits,
    } = params;
    let (kv_heads, head_dim, flush_tokens, res_cap, bits) =
        (*kv_heads, *head_dim, *flush_tokens, *res_cap, *bits);
    if flush_tokens == 0 || kv_heads == 0 || head_dim == 0 {
        return QcfMetric {
            action: "kivi".to_string(),
            raw_value: 0.0,
            normalized_value: 0.0,
            per_head: Some(vec![0.0; kv_heads]),
            tokens_affected: 0,
        };
    }

    let blocks_per_group = QKKV;
    let n_groups = flush_tokens / blocks_per_group;
    if n_groups == 0 {
        return QcfMetric {
            action: "kivi".to_string(),
            raw_value: 0.0,
            normalized_value: 0.0,
            per_head: Some(vec![0.0; kv_heads]),
            tokens_affected: flush_tokens,
        };
    }

    let mut per_head_k = vec![0.0f32; kv_heads];
    let mut per_head_v = vec![0.0f32; kv_heads];

    for h in 0..kv_heads {
        let head_base = h * res_cap * head_dim;

        // Key NMSE: per-channel quantization (groups across tokens within each channel)
        let mut k_nmse_sum = 0.0f32;
        let mut k_block_count = 0usize;
        for group in 0..n_groups {
            let tok_start = group * blocks_per_group;
            for ch in 0..head_dim {
                let mut vals = [0.0f32; QKKV];
                for (t, v) in vals.iter_mut().enumerate().take(blocks_per_group) {
                    let idx = head_base + (tok_start + t) * head_dim + ch;
                    if idx < res_k.len() {
                        *v = res_k[idx];
                    }
                }
                k_nmse_sum += compute_nmse_block(&vals, bits, config.epsilon);
                k_block_count += 1;
            }
        }
        per_head_k[h] = if k_block_count > 0 {
            k_nmse_sum / k_block_count as f32
        } else {
            0.0
        };

        // Value NMSE: per-token quantization (groups within one token's head_dim)
        let mut v_nmse_sum = 0.0f32;
        let mut v_block_count = 0usize;
        let blocks_per_token = head_dim / QKKV;
        for t in 0..flush_tokens {
            let tok_base = head_base + t * head_dim;
            for b in 0..blocks_per_token {
                let start = tok_base + b * QKKV;
                if start + QKKV <= res_v.len() {
                    let chunk: &[f32; QKKV] = res_v[start..start + QKKV].try_into().unwrap();
                    v_nmse_sum += compute_nmse_block(chunk, bits, config.epsilon);
                    v_block_count += 1;
                }
            }
        }
        per_head_v[h] = if v_block_count > 0 {
            v_nmse_sum / v_block_count as f32
        } else {
            0.0
        };
    }

    // Combined per-head proxy: 0.6 × K + 0.4 × V
    let per_head: Vec<f32> = per_head_k
        .iter()
        .zip(per_head_v.iter())
        .map(|(&k, &v)| 0.6 * k + 0.4 * v)
        .collect();

    let raw_value = aggregate_heads(&per_head, &config.aggregation);

    QcfMetric {
        action: "kivi".to_string(),
        raw_value,
        normalized_value: raw_value, // NMSE is already a normalized metric
        per_head: Some(per_head),
        tokens_affected: flush_tokens,
    }
}

/// Compute KIVI OPR (Output Perturbation Ratio) for V cache quantization.
///
/// OPR = ||Σ ΔV|| / ||Σ V_orig|| per head, where ΔV = V_quant - V_orig.
/// V_quant is obtained via quantize → dequantize round-trip.
/// Uniform weight (1/n) is used, cancelling in numerator and denominator.
///
/// Only V is considered; K quantization's primary effect is on attention weights
/// (a second-order effect ignored per design decision B-6).
///
/// Returns `QcfMetric` with:
/// - `action`: "kivi_opr"
/// - `raw_value`: sum of per-head OPR values
/// - `normalized_value`: same as raw_value
/// - `per_head`: per-head OPR vector
/// - `tokens_affected`: flush_tokens
pub fn compute_flush_opr(params: &FlushQcfParams, _config: &QcfConfig) -> QcfMetric {
    let FlushQcfParams {
        res_v,
        kv_heads,
        head_dim,
        flush_tokens,
        res_cap,
        bits,
        ..
    } = params;
    let (kv_heads, head_dim, flush_tokens, res_cap, bits) =
        (*kv_heads, *head_dim, *flush_tokens, *res_cap, *bits);

    if flush_tokens == 0 || kv_heads == 0 || head_dim == 0 {
        return QcfMetric {
            action: "kivi_opr".to_string(),
            raw_value: 0.0,
            normalized_value: 0.0,
            per_head: Some(vec![0.0; kv_heads]),
            tokens_affected: 0,
        };
    }

    let blocks_per_token = head_dim / QKKV;
    let mut per_head_opr = vec![0.0f32; kv_heads];

    for (h, opr_slot) in per_head_opr.iter_mut().enumerate() {
        let head_base = h * res_cap * head_dim;

        let mut sum_delta = vec![0.0f32; head_dim];
        let mut sum_orig = vec![0.0f32; head_dim];

        for t in 0..flush_tokens {
            let tok_base = head_base + t * head_dim;

            // Reconstruct V_quant for this token via quantize → dequantize per block
            let mut v_quant_token = vec![0.0f32; head_dim];
            for b in 0..blocks_per_token {
                let start = tok_base + b * QKKV;
                if start + QKKV > res_v.len() {
                    continue;
                }
                let chunk: &[f32; QKKV] = res_v[start..start + QKKV].try_into().unwrap();
                let out = &mut v_quant_token[b * QKKV..(b + 1) * QKKV];
                let out_arr: &mut [f32; QKKV] = out.try_into().unwrap();
                match bits {
                    2 => {
                        let block = BlockQ2_0::quantize(chunk);
                        block.dequantize(out_arr);
                    }
                    4 => {
                        let block = BlockKVQ4::quantize(chunk);
                        block.dequantize(out_arr);
                    }
                    8 => {
                        let block = BlockKVQ8::quantize(chunk);
                        block.dequantize(out_arr);
                    }
                    _ => {
                        // Unsupported bits: treat as zero error (copy original)
                        out_arr.copy_from_slice(chunk);
                    }
                }
            }

            // Accumulate sum_delta and sum_orig across tokens
            for d in 0..head_dim {
                let idx = tok_base + d;
                if idx < res_v.len() {
                    let v_orig = res_v[idx];
                    sum_orig[d] += v_orig;
                    sum_delta[d] += v_quant_token[d] - v_orig;
                }
            }
        }

        // OPR = L2(sum_delta) / L2(sum_orig)
        let norm_delta = sum_delta.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_orig = sum_orig.iter().map(|&x| x * x).sum::<f32>().sqrt();

        *opr_slot = if norm_orig < 1e-12 {
            0.0
        } else {
            norm_delta / norm_orig
        };
    }

    let raw_value: f32 = per_head_opr.iter().sum::<f32>() / kv_heads as f32;

    QcfMetric {
        action: "kivi_opr".to_string(),
        raw_value,
        normalized_value: raw_value,
        per_head: Some(per_head_opr),
        tokens_affected: flush_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmse_block_q8_low_error() {
        // Q8 should have very low NMSE
        let original: [f32; QKKV] = std::array::from_fn(|i| i as f32 * 0.1);
        let nmse = compute_nmse_block(&original, 8, 1e-8);
        assert!(nmse < 0.01, "Q8 NMSE={nmse} should be very low");
    }

    #[test]
    fn test_nmse_block_q4_moderate_error() {
        let original: [f32; QKKV] = std::array::from_fn(|i| i as f32 * 0.1);
        let nmse = compute_nmse_block(&original, 4, 1e-8);
        assert!(nmse > 0.0, "Q4 NMSE should be positive");
        assert!(nmse < 0.5, "Q4 NMSE={nmse} should be moderate");
    }

    #[test]
    fn test_nmse_block_q2_higher_error() {
        let original: [f32; QKKV] = std::array::from_fn(|i| i as f32 * 0.1);
        let nmse_q2 = compute_nmse_block(&original, 2, 1e-8);
        let nmse_q4 = compute_nmse_block(&original, 4, 1e-8);
        let nmse_q8 = compute_nmse_block(&original, 8, 1e-8);
        assert!(
            nmse_q2 >= nmse_q4,
            "Q2 ({nmse_q2}) should have >= NMSE than Q4 ({nmse_q4})"
        );
        assert!(
            nmse_q4 >= nmse_q8,
            "Q4 ({nmse_q4}) should have >= NMSE than Q8 ({nmse_q8})"
        );
    }

    #[test]
    fn test_nmse_block_zero_variance() {
        // All same values → variance ≈ 0 → NMSE = 0
        let original = [42.0f32; QKKV];
        let nmse = compute_nmse_block(&original, 4, 1e-8);
        assert_eq!(nmse, 0.0);
    }

    #[test]
    fn test_nmse_block_zero_values() {
        let original = [0.0f32; QKKV];
        let nmse = compute_nmse_block(&original, 2, 1e-8);
        assert_eq!(nmse, 0.0);
    }

    #[test]
    fn test_nmse_block_invalid_bits() {
        let original: [f32; QKKV] = std::array::from_fn(|i| i as f32);
        let nmse = compute_nmse_block(&original, 3, 1e-8);
        assert_eq!(nmse, 0.0); // Unsupported bits → 0
    }

    #[test]
    fn test_flush_proxy_basic() {
        let kv_heads = 1;
        let head_dim = 32; // Must be multiple of QKKV
        let flush_tokens = 32;
        let res_cap = 32;
        let elems = kv_heads * res_cap * head_dim;

        let res_k: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.01).collect();
        let res_v: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.02).collect();

        let config = QcfConfig::default();
        let params = FlushQcfParams {
            res_k: &res_k,
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
        };
        let metric = compute_flush_qcf(&params, &config);

        assert_eq!(metric.action, "kivi");
        assert!(metric.raw_value >= 0.0);
        assert!(metric.raw_value <= 1.0);
        assert_eq!(metric.tokens_affected, flush_tokens);
        assert_eq!(metric.per_head.as_ref().unwrap().len(), kv_heads);
    }

    #[test]
    fn test_flush_proxy_empty() {
        let config = QcfConfig::default();
        let params = FlushQcfParams {
            res_k: &[],
            res_v: &[],
            kv_heads: 0,
            head_dim: 0,
            flush_tokens: 0,
            res_cap: 0,
            bits: 4,
        };
        let metric = compute_flush_qcf(&params, &config);
        assert_eq!(metric.raw_value, 0.0);
    }

    #[test]
    fn test_flush_proxy_multi_head() {
        let kv_heads = 4;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let elems = kv_heads * res_cap * head_dim;

        let res_k: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.01).collect();
        let res_v: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.02).collect();

        let config = QcfConfig::default();
        let params = FlushQcfParams {
            res_k: &res_k,
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 2,
        };
        let metric = compute_flush_qcf(&params, &config);

        assert_eq!(metric.per_head.as_ref().unwrap().len(), kv_heads);
        for &h_val in metric.per_head.as_ref().unwrap() {
            assert!(h_val >= 0.0);
        }
    }

    #[test]
    fn test_flush_proxy_q2_higher_than_q8() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let elems = kv_heads * res_cap * head_dim;

        let res_k: Vec<f32> = (0..elems).map(|i| ((i % 100) as f32) * 0.1).collect();
        let res_v: Vec<f32> = (0..elems).map(|i| ((i % 100) as f32) * 0.1).collect();

        let config = QcfConfig::default();
        let proxy_q2 = compute_flush_qcf(
            &FlushQcfParams {
                res_k: &res_k,
                res_v: &res_v,
                kv_heads,
                head_dim,
                flush_tokens,
                res_cap,
                bits: 2,
            },
            &config,
        );
        let proxy_q8 = compute_flush_qcf(
            &FlushQcfParams {
                res_k: &res_k,
                res_v: &res_v,
                kv_heads,
                head_dim,
                flush_tokens,
                res_cap,
                bits: 8,
            },
            &config,
        );

        assert!(
            proxy_q2.raw_value >= proxy_q8.raw_value,
            "Q2 proxy ({}) should >= Q8 proxy ({})",
            proxy_q2.raw_value,
            proxy_q8.raw_value
        );
    }

    // ── compute_flush_opr tests ──────────────────────────────────────────────

    #[test]
    fn test_flush_opr_basic() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let elems = kv_heads * res_cap * head_dim;

        let res_k: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.01).collect();
        let res_v: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.02).collect();

        let config = QcfConfig::default();
        let params = FlushQcfParams {
            res_k: &res_k,
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
        };
        let metric = compute_flush_opr(&params, &config);

        assert_eq!(metric.action, "kivi_opr");
        assert!(metric.raw_value >= 0.0, "OPR must be non-negative");
        assert_eq!(metric.tokens_affected, flush_tokens);
        assert!(metric.per_head.is_some());
        assert_eq!(metric.per_head.as_ref().unwrap().len(), kv_heads);
        assert!(
            (metric.raw_value - metric.normalized_value).abs() < 1e-6,
            "raw_value and normalized_value must be equal"
        );
    }

    #[test]
    fn test_flush_opr_q2_higher_than_q8() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let elems = kv_heads * res_cap * head_dim;

        // Use values with meaningful variance so quantization error is detectable
        let res_k: Vec<f32> = (0..elems).map(|i| ((i % 100) as f32) * 0.1).collect();
        let res_v: Vec<f32> = (0..elems).map(|i| ((i % 100) as f32) * 0.1).collect();

        let config = QcfConfig::default();
        let make_params = |bits: u8| FlushQcfParams {
            res_k: &res_k,
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits,
        };

        let opr_q2 = compute_flush_opr(&make_params(2), &config);
        let opr_q8 = compute_flush_opr(&make_params(8), &config);

        assert!(
            opr_q2.raw_value >= opr_q8.raw_value,
            "Q2 OPR ({}) should >= Q8 OPR ({})",
            opr_q2.raw_value,
            opr_q8.raw_value
        );
    }

    #[test]
    fn test_flush_opr_zero_input() {
        let kv_heads = 1;
        let head_dim = 32;
        let flush_tokens = 32;
        let res_cap = 32;
        let elems = kv_heads * res_cap * head_dim;

        let res_k = vec![0.0f32; elems];
        let res_v = vec![0.0f32; elems];

        let config = QcfConfig::default();
        let params = FlushQcfParams {
            res_k: &res_k,
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 4,
        };
        let metric = compute_flush_opr(&params, &config);

        assert_eq!(metric.raw_value, 0.0, "zero input → OPR must be 0.0");
        assert_eq!(metric.per_head.as_ref().unwrap()[0], 0.0);
    }

    #[test]
    fn test_flush_opr_multi_head() {
        let kv_heads = 4;
        let head_dim = 64;
        let flush_tokens = 32;
        let res_cap = 32;
        let elems = kv_heads * res_cap * head_dim;

        let res_k: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.01).collect();
        let res_v: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.02 + 1.0).collect();

        let config = QcfConfig::default();
        let params = FlushQcfParams {
            res_k: &res_k,
            res_v: &res_v,
            kv_heads,
            head_dim,
            flush_tokens,
            res_cap,
            bits: 2,
        };
        let metric = compute_flush_opr(&params, &config);

        let ph = metric.per_head.as_ref().unwrap();
        assert_eq!(ph.len(), kv_heads, "per_head length must equal kv_heads");
        for (h, &opr) in ph.iter().enumerate() {
            assert!(opr >= 0.0, "head {h} OPR ({opr}) must be non-negative");
        }
        // raw_value must equal mean of per_head
        let expected_mean: f32 = ph.iter().sum::<f32>() / ph.len() as f32;
        assert!(
            (metric.raw_value - expected_mean).abs() < 1e-5,
            "raw_value ({}) must equal mean of per_head ({})",
            metric.raw_value,
            expected_mean
        );
    }
}
