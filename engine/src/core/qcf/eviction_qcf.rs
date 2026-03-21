//! Eviction proxy: estimates information loss from KV token removal.
//!
//! Shared by H2O and StreamingLLM eviction actions.
//! Formula: `proxy = Σ_evicted attn(t)×‖V(t)‖₁ / Σ_all attn(t)×‖V(t)‖₁`

use super::{QcfConfig, QcfMetric, aggregate_heads};
use crate::core::kv_cache::{KVCache, KVLayout};

/// Compute eviction proxy from identified evicted tokens and their attention scores.
///
/// Uses attention × V-norm importance: each token's contribution is
/// `attn(t) × ‖V(t)‖₁`. The proxy is the fraction of total importance
/// that is being removed:
///
/// `proxy = Σ_evicted attn(t)×‖V(t)‖₁ / Σ_all attn(t)×‖V(t)‖₁`
///
/// `evicted`: slice of `(position, attention_score)` pairs for tokens about to be evicted.
/// `all_scores`: attention scores for ALL active tokens `[max_seq_len]`.
/// Returns a `QcfMetric` with per-head breakdown and aggregated value.
pub fn compute_eviction_qcf_attn(
    evicted: &[(usize, f32)],
    all_scores: &[f32],
    cache: &KVCache,
    config: &QcfConfig,
) -> QcfMetric {
    let kv_heads = cache.kv_heads();
    let head_dim = cache.head_dim();
    let current_pos = cache.current_pos;
    let capacity = cache.capacity();
    let layout = cache.layout();
    let epsilon = config.epsilon;

    if evicted.is_empty() || current_pos == 0 || kv_heads == 0 {
        return QcfMetric {
            action: "eviction_attn".to_string(),
            raw_value: 0.0,
            per_head: Some(vec![0.0; kv_heads]),
            tokens_affected: 0,
        };
    }

    // Access V buffer as f32 slice
    let v_data = cache.v_buffer.as_slice::<f32>();

    let mut per_head = vec![0.0f32; kv_heads];

    for (h, ph) in per_head.iter_mut().enumerate() {
        let mut total_importance = 0.0f32;
        let mut evicted_importance = 0.0f32;

        // First pass: compute attn(t) × ‖V(t)‖₁ for ALL active positions
        for pos in 0..current_pos {
            let offset = compute_v_offset(layout, h, pos, head_dim, capacity, kv_heads);
            let v_norm = l1_norm(&v_data[offset..offset + head_dim]);
            let attn = if pos < all_scores.len() {
                all_scores[pos]
            } else {
                0.0
            };
            total_importance += attn * v_norm;
        }

        // Second pass: compute attn(t) × ‖V(t)‖₁ for evicted tokens
        for &(pos, attn_score) in evicted {
            if pos < current_pos {
                let offset = compute_v_offset(layout, h, pos, head_dim, capacity, kv_heads);
                let v_norm = l1_norm(&v_data[offset..offset + head_dim]);
                evicted_importance += attn_score * v_norm;
            }
        }

        *ph = if total_importance > epsilon {
            (evicted_importance / total_importance).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }

    let raw_value = aggregate_heads(&per_head, &config.aggregation);

    QcfMetric {
        action: "eviction_attn".to_string(),
        raw_value,
        per_head: Some(per_head),
        tokens_affected: evicted.len(),
    }
}

/// Compute a simple position-based proxy for sliding window eviction.
///
/// `proxy = prune_count / total_active` — fraction of active tokens removed.
/// No V-buffer access needed since sliding window is position-based.
pub fn compute_sliding_qcf_attn(prune_count: usize, total_active: usize) -> QcfMetric {
    let raw_value = if total_active > 0 {
        (prune_count as f32 / total_active as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };

    QcfMetric {
        action: "sliding_attn".to_string(),
        raw_value,
        per_head: None,
        tokens_affected: prune_count,
    }
}

/// Pre-rank H2O tokens to identify which will be evicted, before actual eviction.
///
/// Replicates the H2O 3-partition ranking logic:
/// 1. Protected prefix (never evicted)
/// 2. Heavy hitters (top `keep_ratio` by importance)
/// 3. Recent window (latest tokens)
///
/// Returns `Vec<(position, importance_score)>` of tokens that would be evicted.
pub fn identify_evicted_h2o(
    importance: &[f32],
    protected_prefix: usize,
    keep_ratio: f32,
    current_pos: usize,
    target_len: usize,
) -> Vec<(usize, f32)> {
    if current_pos <= target_len || current_pos == 0 {
        return Vec::new();
    }

    let prefix = protected_prefix.min(current_pos);
    // Compute available slots after prefix
    let available = target_len.saturating_sub(prefix);
    if available == 0 {
        return Vec::new();
    }

    let hh_budget = ((available as f32) * keep_ratio) as usize;
    let recent_budget = available.saturating_sub(hh_budget);
    let recent_start = current_pos.saturating_sub(recent_budget);

    // Evictable region: [prefix..recent_start]
    let evictable_end = recent_start.min(current_pos);
    if evictable_end <= prefix {
        return Vec::new();
    }

    // Rank evictable tokens by importance (ascending = lowest first)
    let mut ranked: Vec<(usize, f32)> = (prefix..evictable_end)
        .filter_map(|pos| {
            if pos < importance.len() {
                Some((pos, importance[pos]))
            } else {
                None
            }
        })
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Keep top `hh_budget` by importance → evict the rest
    let n_evictable = ranked.len();
    if n_evictable <= hh_budget {
        return Vec::new();
    }

    // Evicted = lowest-importance tokens (first n_evictable - hh_budget)
    ranked.truncate(n_evictable - hh_budget);
    ranked
}

/// Compute V buffer offset based on layout.
fn compute_v_offset(
    layout: KVLayout,
    head: usize,
    pos: usize,
    head_dim: usize,
    capacity: usize,
    kv_heads: usize,
) -> usize {
    match layout {
        KVLayout::HeadMajor => head * capacity * head_dim + pos * head_dim,
        KVLayout::SeqMajor => pos * kv_heads * head_dim + head * head_dim,
    }
}

/// L1 norm of a float slice.
fn l1_norm(data: &[f32]) -> f32 {
    data.iter().map(|v| v.abs()).sum()
}

/// L2 norm of a float slice.
fn l2_norm(data: &[f32]) -> f32 {
    data.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Compute CAOTE-based eviction error for a set of evicted token positions.
///
/// Uses the CAOTE attention output error formula:
/// ```text
/// error = (1 / (1 - Σα_evicted)) × ‖Σ α_j × (o_mean - v_j)‖₂ / ‖o_mean‖₂
/// ```
///
/// `evicted_positions`: cache positions of tokens about to be evicted.
/// `last_step_head_attn`: per-KV-head attention from the last decode step,
///   layout `[n_kv_heads * max_seq_len]`, row-major.
/// `cache`: the KV cache (for reading V vectors).
/// `config`: QCF configuration.
pub fn compute_eviction_qcf_caote(
    evicted_positions: &[usize],
    last_step_head_attn: &[f32],
    cache: &KVCache,
    config: &QcfConfig,
) -> QcfMetric {
    let kv_heads = cache.kv_heads();
    let head_dim = cache.head_dim();
    let current_pos = cache.current_pos;
    let capacity = cache.capacity();
    let layout = cache.layout();
    let epsilon = config.epsilon;

    if evicted_positions.is_empty() || current_pos == 0 || kv_heads == 0 {
        return QcfMetric {
            action: "eviction_caote".to_string(),
            raw_value: 0.0,
            per_head: Some(vec![0.0; kv_heads]),
            tokens_affected: 0,
        };
    }

    let max_seq_len = last_step_head_attn.len() / kv_heads.max(1);
    let v_data = cache.v_buffer.as_slice::<f32>();
    let mut per_head = vec![0.0f32; kv_heads];

    for (h, ph) in per_head.iter_mut().enumerate() {
        let head_offset = h * max_seq_len;

        // Re-normalize attention to sum=1.0 for active positions
        let attn_sum: f32 = (0..current_pos)
            .filter(|&t| t < max_seq_len)
            .map(|t| last_step_head_attn[head_offset + t])
            .sum();
        if attn_sum < epsilon {
            continue;
        }
        let inv_sum = 1.0 / attn_sum;

        // 1. o_mean = Σ_i α_i × v_i
        let mut o_mean = vec![0.0f32; head_dim];
        for pos in 0..current_pos.min(max_seq_len) {
            let alpha = last_step_head_attn[head_offset + pos] * inv_sum;
            if alpha < epsilon {
                continue;
            }
            let v_off = compute_v_offset(layout, h, pos, head_dim, capacity, kv_heads);
            for d in 0..head_dim {
                o_mean[d] += alpha * v_data[v_off + d];
            }
        }

        // 2. alpha_sum for evicted tokens
        let mut alpha_evicted = 0.0f32;
        for &pos in evicted_positions {
            if pos < current_pos && pos < max_seq_len {
                alpha_evicted += last_step_head_attn[head_offset + pos] * inv_sum;
            }
        }

        if alpha_evicted >= 1.0 - epsilon {
            *ph = config.d_max;
            continue;
        }

        // 3. weighted_residual = Σ_{j∈evicted} α_j × (o_mean - v_j)
        let amplification = 1.0 / (1.0 - alpha_evicted);
        let mut weighted_residual = vec![0.0f32; head_dim];
        for &pos in evicted_positions {
            if pos < current_pos && pos < max_seq_len {
                let alpha_j = last_step_head_attn[head_offset + pos] * inv_sum;
                if alpha_j < epsilon {
                    continue;
                }
                let v_off = compute_v_offset(layout, h, pos, head_dim, capacity, kv_heads);
                for d in 0..head_dim {
                    weighted_residual[d] += alpha_j * (o_mean[d] - v_data[v_off + d]);
                }
            }
        }

        // 4. error = amplification × ‖weighted_residual‖₂ / ‖o_mean‖₂
        let residual_norm = l2_norm(&weighted_residual);
        let o_norm = l2_norm(&o_mean);
        let error = amplification * residual_norm;
        *ph = if o_norm > epsilon {
            error / o_norm
        } else {
            error
        };
    }

    let raw_value = aggregate_heads(&per_head, &config.aggregation);
    QcfMetric {
        action: "eviction_caote".to_string(),
        raw_value,
        per_head: Some(per_head),
        tokens_affected: evicted_positions.len(),
    }
}

/// Compute CAOTE-based eviction error for sliding window eviction.
///
/// Same computation as `compute_eviction_qcf_caote` with action = `"sliding_caote"`.
pub fn compute_sliding_qcf_caote(
    evicted_positions: &[usize],
    last_step_head_attn: &[f32],
    cache: &KVCache,
    config: &QcfConfig,
) -> QcfMetric {
    let mut metric =
        compute_eviction_qcf_caote(evicted_positions, last_step_head_attn, cache, config);
    metric.action = "sliding_caote".to_string();
    metric
}

/// Identify positions that will be evicted by sliding window pruning.
///
/// Sliding window evicts the oldest tokens after the protected prefix,
/// keeping the most recent tokens. Returns positions `[prefix..prefix+prune_count]`.
pub fn identify_evicted_sliding(
    protected_prefix: usize,
    prune_count: usize,
    current_pos: usize,
) -> Vec<usize> {
    if prune_count == 0 || current_pos == 0 {
        return Vec::new();
    }
    let start = protected_prefix.min(current_pos);
    let end = (start + prune_count).min(current_pos);
    (start..end).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::shared_buffer::SharedBuffer;
    use crate::core::buffer::{Buffer, DType};
    use crate::core::kv_cache::KVLayout;
    use crate::core::qcf::AggregationMode;
    use crate::core::shape::Shape;
    use crate::core::tensor::Tensor;
    use std::sync::Arc;

    fn make_cache_with_v_data(
        kv_heads: usize,
        head_dim: usize,
        num_tokens: usize,
        layout: KVLayout,
        v_values: &[f32],
    ) -> KVCache {
        let max_seq = 64;
        let backend = Arc::new(CpuBackend::new());
        let buf_size = max_seq * kv_heads * head_dim * 4;

        let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));

        // Fill V buffer with provided data
        unsafe {
            let v_ptr = v_buf.as_mut_ptr() as *mut f32;
            for (i, &val) in v_values.iter().enumerate() {
                if i < buf_size / 4 {
                    *v_ptr.add(i) = val;
                }
            }
        }

        // KVCache::new always reads shape[2] as kv_heads, shape[3] as head_dim
        let shape = Shape::new(vec![1, max_seq, kv_heads, head_dim]);

        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend);
        let mut cache = KVCache::new(k, v, max_seq).with_layout(layout);
        cache.current_pos = num_tokens;
        cache
    }

    #[test]
    fn test_eviction_proxy_basic() {
        let kv_heads = 1;
        let head_dim = 4;
        let num_tokens = 8;

        // V buffer: HeadMajor, single head
        // Each token has V = [1.0, 1.0, 1.0, 1.0] → L1 norm = 4.0
        let max_seq = 64;
        let mut v_data = vec![0.0f32; max_seq * kv_heads * head_dim];
        for t in 0..num_tokens {
            for d in 0..head_dim {
                v_data[t * head_dim + d] = 1.0;
            }
        }

        let cache =
            make_cache_with_v_data(kv_heads, head_dim, num_tokens, KVLayout::HeadMajor, &v_data);

        let config = QcfConfig::default();
        // Evict 2 tokens (positions 2 and 3) with equal attention scores
        // All tokens have uniform attn=1.0
        let all_scores = vec![1.0f32; num_tokens];
        let evicted = vec![(2, 1.0), (3, 1.0)];
        let metric = compute_eviction_qcf_attn(&evicted, &all_scores, &cache, &config);

        assert_eq!(metric.action, "eviction_attn");
        assert_eq!(metric.tokens_affected, 2);
        // With uniform V norms and uniform attn: proxy = 2/8 = 0.25
        assert!(
            (metric.raw_value - 0.25).abs() < 0.01,
            "expected ~0.25, got {}",
            metric.raw_value
        );
    }

    #[test]
    fn test_eviction_proxy_empty_evicted() {
        let kv_heads = 2;
        let head_dim = 4;
        let max_seq = 64;
        let v_data = vec![1.0f32; max_seq * kv_heads * head_dim];
        let cache = make_cache_with_v_data(kv_heads, head_dim, 10, KVLayout::HeadMajor, &v_data);

        let config = QcfConfig::default();
        let all_scores = vec![1.0f32; 10];
        let metric = compute_eviction_qcf_attn(&[], &all_scores, &cache, &config);
        assert_eq!(metric.raw_value, 0.0);
        assert_eq!(metric.tokens_affected, 0);
    }

    #[test]
    fn test_eviction_proxy_high_v_norm_tokens_matter_more() {
        let kv_heads = 1;
        let head_dim = 4;
        let num_tokens = 4;
        let max_seq = 64;

        // Token 0,1: V = [1,1,1,1] (norm=4), Token 2,3: V = [10,10,10,10] (norm=40)
        let mut v_data = vec![0.0f32; max_seq * kv_heads * head_dim];
        for t in 0..2 {
            for d in 0..head_dim {
                v_data[t * head_dim + d] = 1.0;
            }
        }
        for t in 2..4 {
            for d in 0..head_dim {
                v_data[t * head_dim + d] = 10.0;
            }
        }

        let cache =
            make_cache_with_v_data(kv_heads, head_dim, num_tokens, KVLayout::HeadMajor, &v_data);
        let config = QcfConfig::default();
        let all_scores = vec![1.0f32; num_tokens];

        // Evict high-norm tokens → higher proxy
        let metric_high =
            compute_eviction_qcf_attn(&[(2, 1.0), (3, 1.0)], &all_scores, &cache, &config);
        // Evict low-norm tokens → lower proxy
        let metric_low =
            compute_eviction_qcf_attn(&[(0, 1.0), (1, 1.0)], &all_scores, &cache, &config);

        assert!(
            metric_high.raw_value > metric_low.raw_value,
            "evicting high-norm tokens ({}) should give higher proxy than low-norm ({})",
            metric_high.raw_value,
            metric_low.raw_value
        );
    }

    #[test]
    fn test_sliding_proxy() {
        let metric = compute_sliding_qcf_attn(10, 100);
        assert_eq!(metric.action, "sliding_attn");
        assert!((metric.raw_value - 0.1).abs() < 1e-6);
        assert_eq!(metric.tokens_affected, 10);
    }

    #[test]
    fn test_sliding_proxy_zero_total() {
        let metric = compute_sliding_qcf_attn(0, 0);
        assert_eq!(metric.raw_value, 0.0);
    }

    #[test]
    fn test_sliding_proxy_full_eviction() {
        let metric = compute_sliding_qcf_attn(50, 50);
        assert!((metric.raw_value - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_identify_evicted_h2o_basic() {
        // 20 tokens, prefix=4, target=10, keep_ratio=0.5
        let mut importance = vec![0.0f32; 20];
        // Give some tokens high importance
        importance[5] = 10.0;
        importance[10] = 9.0;
        importance[15] = 8.0;
        for i in 4..20 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        let evicted = identify_evicted_h2o(&importance, 4, 0.5, 20, 10);
        // Should evict some tokens from [4..recent_start]
        assert!(!evicted.is_empty());
        // Evicted tokens should have low importance
        for &(pos, score) in &evicted {
            assert!(pos >= 4, "prefix tokens should not be evicted");
            assert!(score < 10.0, "high-importance tokens should be kept");
        }
    }

    #[test]
    fn test_identify_evicted_h2o_no_eviction_needed() {
        let importance = vec![1.0f32; 20];
        let evicted = identify_evicted_h2o(&importance, 4, 0.5, 10, 10);
        assert!(evicted.is_empty());
    }

    #[test]
    fn test_identify_evicted_h2o_all_prefix() {
        let importance = vec![1.0f32; 20];
        // target_len = prefix → available=0 → no eviction
        let evicted = identify_evicted_h2o(&importance, 10, 0.5, 20, 10);
        assert!(evicted.is_empty());
    }

    #[test]
    fn test_eviction_proxy_seqmajor_layout() {
        let kv_heads = 2;
        let head_dim = 4;
        let num_tokens = 4;
        let max_seq = 64;

        // SeqMajor layout: offset = pos * kv_heads * head_dim + head * head_dim
        let mut v_data = vec![0.0f32; max_seq * kv_heads * head_dim];
        for t in 0..num_tokens {
            for h in 0..kv_heads {
                for d in 0..head_dim {
                    let offset = t * kv_heads * head_dim + h * head_dim + d;
                    v_data[offset] = 1.0;
                }
            }
        }

        let cache =
            make_cache_with_v_data(kv_heads, head_dim, num_tokens, KVLayout::SeqMajor, &v_data);
        let config = QcfConfig::default();
        let all_scores = vec![1.0f32; num_tokens];
        let evicted = vec![(1, 1.0)];
        let metric = compute_eviction_qcf_attn(&evicted, &all_scores, &cache, &config);

        assert!(metric.raw_value > 0.0);
        assert!(metric.raw_value <= 1.0);
        assert_eq!(metric.per_head.as_ref().unwrap().len(), kv_heads);
    }

    #[test]
    fn test_eviction_proxy_nonuniform_attn_scores() {
        // Verify that all_scores affects the denominator correctly
        let kv_heads = 1;
        let head_dim = 4;
        let num_tokens = 4;
        let max_seq = 64;

        // All tokens have same V norm = 4.0
        let mut v_data = vec![0.0f32; max_seq * kv_heads * head_dim];
        for t in 0..num_tokens {
            for d in 0..head_dim {
                v_data[t * head_dim + d] = 1.0;
            }
        }

        let cache =
            make_cache_with_v_data(kv_heads, head_dim, num_tokens, KVLayout::HeadMajor, &v_data);
        let config = QcfConfig::default();

        // Token 0 has very high attn score in all_scores
        // Evict token 0 (high attn) vs token 1 (low attn)
        let all_scores = vec![10.0, 1.0, 1.0, 1.0];

        // Evict token 0 (attn=10.0): evicted_imp = 10*4 = 40, total_imp = 10*4+1*4+1*4+1*4 = 52
        let metric_high = compute_eviction_qcf_attn(&[(0, 10.0)], &all_scores, &cache, &config);
        // Evict token 1 (attn=1.0): evicted_imp = 1*4 = 4, total_imp = 52
        let metric_low = compute_eviction_qcf_attn(&[(1, 1.0)], &all_scores, &cache, &config);

        assert!(
            metric_high.raw_value > metric_low.raw_value,
            "evicting high-attn token ({}) should > low-attn ({})",
            metric_high.raw_value,
            metric_low.raw_value
        );
        // metric_high ≈ 40/52 ≈ 0.769
        assert!(
            (metric_high.raw_value - 40.0 / 52.0).abs() < 0.01,
            "expected ~0.769, got {}",
            metric_high.raw_value
        );
    }

    // ── CAOTE tests ──

    #[test]
    fn test_caote_basic() {
        // 1 head, 4 dims, 4 tokens, uniform attention → evicting similar tokens = low error
        let kv_heads = 1;
        let head_dim = 4;
        let num_tokens = 4;
        let max_seq = 64;

        // All tokens V = [1,1,1,1] → o_mean ≈ [1,1,1,1], (o_mean - v_j) = 0 → error = 0
        let mut v_data = vec![0.0f32; max_seq * kv_heads * head_dim];
        for t in 0..num_tokens {
            for d in 0..head_dim {
                v_data[t * head_dim + d] = 1.0;
            }
        }

        let cache =
            make_cache_with_v_data(kv_heads, head_dim, num_tokens, KVLayout::HeadMajor, &v_data);
        let config = QcfConfig::default();

        // Uniform attention: [0.25, 0.25, 0.25, 0.25]
        let mut head_attn = vec![0.0f32; kv_heads * max_seq];
        for t in 0..num_tokens {
            head_attn[t] = 0.25;
        }

        let metric = compute_eviction_qcf_caote(&[0, 1], &head_attn, &cache, &config);
        assert_eq!(metric.action, "eviction_caote");
        assert_eq!(metric.tokens_affected, 2);
        // All values identical → residual = 0 → error = 0
        assert!(
            metric.raw_value < 1e-6,
            "identical values should give ~0 error, got {}",
            metric.raw_value
        );
    }

    #[test]
    fn test_caote_divergent_values_give_higher_error() {
        // Evicting token with value far from o_mean should give higher error
        let kv_heads = 1;
        let head_dim = 4;
        let num_tokens = 4;
        let max_seq = 64;

        // Token 0: V = [10,10,10,10] (outlier)
        // Token 1,2,3: V = [1,1,1,1]
        let mut v_data = vec![0.0f32; max_seq * kv_heads * head_dim];
        for d in 0..head_dim {
            v_data[0 * head_dim + d] = 10.0;
        }
        for t in 1..num_tokens {
            for d in 0..head_dim {
                v_data[t * head_dim + d] = 1.0;
            }
        }

        let cache =
            make_cache_with_v_data(kv_heads, head_dim, num_tokens, KVLayout::HeadMajor, &v_data);
        let config = QcfConfig::default();

        let mut head_attn = vec![0.0f32; kv_heads * max_seq];
        for t in 0..num_tokens {
            head_attn[t] = 0.25;
        }

        // Evict token 0 (outlier) → high error
        let metric_outlier = compute_eviction_qcf_caote(&[0], &head_attn, &cache, &config);
        // Evict token 1 (normal) → lower error
        let metric_normal = compute_eviction_qcf_caote(&[1], &head_attn, &cache, &config);

        assert!(
            metric_outlier.raw_value > metric_normal.raw_value,
            "evicting outlier ({}) should give higher error than normal ({})",
            metric_outlier.raw_value,
            metric_normal.raw_value
        );
    }

    #[test]
    fn test_caote_high_attention_amplifies_error() {
        // Token with high attention should have amplified error when evicted
        let kv_heads = 1;
        let head_dim = 4;
        let num_tokens = 4;
        let max_seq = 64;

        // Token 0,1: V = [5,5,5,5], Token 2,3: V = [1,1,1,1]
        let mut v_data = vec![0.0f32; max_seq * kv_heads * head_dim];
        for t in 0..2 {
            for d in 0..head_dim {
                v_data[t * head_dim + d] = 5.0;
            }
        }
        for t in 2..num_tokens {
            for d in 0..head_dim {
                v_data[t * head_dim + d] = 1.0;
            }
        }

        let cache =
            make_cache_with_v_data(kv_heads, head_dim, num_tokens, KVLayout::HeadMajor, &v_data);
        let config = QcfConfig::default();

        // High attention on token 0: [0.7, 0.1, 0.1, 0.1]
        let mut high_attn = vec![0.0f32; kv_heads * max_seq];
        high_attn[0] = 0.7;
        high_attn[1] = 0.1;
        high_attn[2] = 0.1;
        high_attn[3] = 0.1;

        // Uniform attention: [0.25, 0.25, 0.25, 0.25]
        let mut uniform_attn = vec![0.0f32; kv_heads * max_seq];
        for t in 0..num_tokens {
            uniform_attn[t] = 0.25;
        }

        // Evict token 0 with high attention → amplification factor 1/(1-0.7) = 3.33
        let metric_high = compute_eviction_qcf_caote(&[0], &high_attn, &cache, &config);
        // Evict token 0 with uniform attention → amplification factor 1/(1-0.25) = 1.33
        let metric_uniform = compute_eviction_qcf_caote(&[0], &uniform_attn, &cache, &config);

        assert!(
            metric_high.raw_value > metric_uniform.raw_value,
            "high attention ({}) should amplify error vs uniform ({})",
            metric_high.raw_value,
            metric_uniform.raw_value
        );
    }

    #[test]
    fn test_caote_empty_eviction() {
        let kv_heads = 2;
        let head_dim = 4;
        let max_seq = 64;
        let v_data = vec![1.0f32; max_seq * kv_heads * head_dim];
        let cache = make_cache_with_v_data(kv_heads, head_dim, 10, KVLayout::HeadMajor, &v_data);
        let config = QcfConfig::default();
        let head_attn = vec![0.1f32; kv_heads * max_seq];

        let metric = compute_eviction_qcf_caote(&[], &head_attn, &cache, &config);
        assert_eq!(metric.raw_value, 0.0);
        assert_eq!(metric.tokens_affected, 0);
    }

    #[test]
    fn test_caote_multi_head() {
        let kv_heads = 2;
        let head_dim = 4;
        let num_tokens = 4;
        let max_seq = 64;

        // Head 0: all V = [1,1,1,1]; Head 1: token 0 = [10,10,10,10], rest = [1,1,1,1]
        let mut v_data = vec![0.0f32; max_seq * kv_heads * head_dim];
        // HeadMajor: head * capacity * head_dim + pos * head_dim
        for t in 0..num_tokens {
            for d in 0..head_dim {
                // Head 0
                v_data[0 * max_seq * head_dim + t * head_dim + d] = 1.0;
                // Head 1
                let val = if t == 0 { 10.0 } else { 1.0 };
                v_data[1 * max_seq * head_dim + t * head_dim + d] = val;
            }
        }

        let cache =
            make_cache_with_v_data(kv_heads, head_dim, num_tokens, KVLayout::HeadMajor, &v_data);
        let config = QcfConfig::default();

        let mut head_attn = vec![0.0f32; kv_heads * max_seq];
        for h in 0..kv_heads {
            for t in 0..num_tokens {
                head_attn[h * max_seq + t] = 0.25;
            }
        }

        // Evict token 0
        let metric = compute_eviction_qcf_caote(&[0], &head_attn, &cache, &config);
        let heads = metric.per_head.as_ref().unwrap();

        // Head 0: all values identical → error ≈ 0
        assert!(heads[0] < 1e-5, "head 0 uniform values: {}", heads[0]);
        // Head 1: token 0 is outlier → error > 0
        assert!(heads[1] > 0.01, "head 1 outlier evicted: {}", heads[1]);
    }

    #[test]
    fn test_sliding_caote_action_name() {
        let kv_heads = 1;
        let head_dim = 4;
        let max_seq = 64;
        let v_data = vec![1.0f32; max_seq * kv_heads * head_dim];
        let cache = make_cache_with_v_data(kv_heads, head_dim, 4, KVLayout::HeadMajor, &v_data);
        let config = QcfConfig::default();
        let head_attn = vec![0.25f32; kv_heads * max_seq];

        let metric = compute_sliding_qcf_caote(&[0, 1], &head_attn, &cache, &config);
        assert_eq!(metric.action, "sliding_caote");
    }

    #[test]
    fn test_identify_evicted_sliding_basic() {
        let positions = identify_evicted_sliding(4, 6, 20);
        assert_eq!(positions, vec![4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_identify_evicted_sliding_zero_prune() {
        assert!(identify_evicted_sliding(4, 0, 20).is_empty());
    }

    #[test]
    fn test_identify_evicted_sliding_clamped() {
        // prune_count exceeds available → clamped to current_pos
        let positions = identify_evicted_sliding(8, 100, 10);
        assert_eq!(positions, vec![8, 9]);
    }

    #[test]
    fn test_l2_norm() {
        assert!((l2_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-6);
        assert_eq!(l2_norm(&[]), 0.0);
    }
}
