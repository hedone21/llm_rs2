//! Eviction proxy: estimates information loss from KV token removal.
//!
//! Shared by H2O, SnapKV, and StreamingLLM eviction actions.
//! Formula: `proxy = Σ_evicted attn(t)×‖V(t)‖₁ / Σ_all attn(t)×‖V(t)‖₁`

use super::{ProxyConfig, ProxyMetric, aggregate_heads};
use crate::core::kv_cache::{KVCache, KVLayout};

/// Compute eviction proxy from identified evicted tokens and their attention scores.
///
/// Uses attention × V-norm importance: each token's contribution is
/// `attn(t) × ‖V(t)‖₁`. The proxy is the fraction of total importance
/// that is being removed.
///
/// `evicted`: slice of `(position, attention_score)` pairs for tokens about to be evicted.
/// Returns a `ProxyMetric` with per-head breakdown and aggregated value.
pub fn compute_eviction_proxy(
    evicted: &[(usize, f32)],
    cache: &KVCache,
    config: &ProxyConfig,
) -> ProxyMetric {
    let kv_heads = cache.kv_heads();
    let head_dim = cache.head_dim();
    let current_pos = cache.current_pos;
    let capacity = cache.capacity();
    let layout = cache.layout();
    let epsilon = config.epsilon;

    if evicted.is_empty() || current_pos == 0 || kv_heads == 0 {
        return ProxyMetric {
            action: "eviction".to_string(),
            raw_value: 0.0,
            per_head: Some(vec![0.0; kv_heads]),
            tokens_affected: 0,
        };
    }

    // Access V buffer as f32 slice
    let v_data = cache.v_buffer.as_slice::<f32>();

    let mut per_head = vec![0.0f32; kv_heads];

    for (h, ph) in per_head.iter_mut().enumerate() {
        // Compute total importance and evicted importance for this head
        let mut total_importance = 0.0f32;
        let mut evicted_importance = 0.0f32;

        // First pass: compute V norms for all active positions
        for pos in 0..current_pos {
            let offset = compute_v_offset(layout, h, pos, head_dim, capacity, kv_heads);
            let v_norm = l1_norm(&v_data[offset..offset + head_dim]);
            total_importance += v_norm;
        }

        // Second pass: compute evicted importance (attn × v_norm)
        for &(pos, attn_score) in evicted {
            if pos < current_pos {
                let offset = compute_v_offset(layout, h, pos, head_dim, capacity, kv_heads);
                let v_norm = l1_norm(&v_data[offset..offset + head_dim]);
                evicted_importance += attn_score * v_norm;
            }
        }

        // Normalize: attn-weighted V norm of evicted / total V norm
        let avg_attn = if !evicted.is_empty() {
            evicted.iter().map(|(_, s)| s).sum::<f32>() / evicted.len() as f32
        } else {
            1.0
        };
        let total_weighted = total_importance * avg_attn;

        *ph = if total_weighted > epsilon {
            (evicted_importance / total_weighted).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }

    let raw_value = aggregate_heads(&per_head, &config.aggregation);

    ProxyMetric {
        action: "eviction".to_string(),
        raw_value,
        per_head: Some(per_head),
        tokens_affected: evicted.len(),
    }
}

/// Compute a simple position-based proxy for sliding window eviction.
///
/// `proxy = prune_count / total_active` — fraction of active tokens removed.
/// No V-buffer access needed since sliding window is position-based.
pub fn compute_sliding_proxy(prune_count: usize, total_active: usize) -> ProxyMetric {
    let raw_value = if total_active > 0 {
        (prune_count as f32 / total_active as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };

    ProxyMetric {
        action: "sliding".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::shared_buffer::SharedBuffer;
    use crate::core::buffer::{Buffer, DType};
    use crate::core::kv_cache::KVLayout;
    use crate::core::proxy::AggregationMode;
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

        let config = ProxyConfig::default();
        // Evict 2 tokens (positions 2 and 3) with equal attention scores
        let evicted = vec![(2, 1.0), (3, 1.0)];
        let metric = compute_eviction_proxy(&evicted, &cache, &config);

        assert_eq!(metric.action, "eviction");
        assert_eq!(metric.tokens_affected, 2);
        // With uniform V norms: proxy = 2/8 = 0.25
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

        let config = ProxyConfig::default();
        let metric = compute_eviction_proxy(&[], &cache, &config);
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
        let config = ProxyConfig::default();

        // Evict high-norm tokens → higher proxy
        let metric_high = compute_eviction_proxy(&[(2, 1.0), (3, 1.0)], &cache, &config);
        // Evict low-norm tokens → lower proxy
        let metric_low = compute_eviction_proxy(&[(0, 1.0), (1, 1.0)], &cache, &config);

        assert!(
            metric_high.raw_value > metric_low.raw_value,
            "evicting high-norm tokens ({}) should give higher proxy than low-norm ({})",
            metric_high.raw_value,
            metric_low.raw_value
        );
    }

    #[test]
    fn test_sliding_proxy() {
        let metric = compute_sliding_proxy(10, 100);
        assert_eq!(metric.action, "sliding");
        assert!((metric.raw_value - 0.1).abs() < 1e-6);
        assert_eq!(metric.tokens_affected, 10);
    }

    #[test]
    fn test_sliding_proxy_zero_total() {
        let metric = compute_sliding_proxy(0, 0);
        assert_eq!(metric.raw_value, 0.0);
    }

    #[test]
    fn test_sliding_proxy_full_eviction() {
        let metric = compute_sliding_proxy(50, 50);
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
        let config = ProxyConfig::default();
        let evicted = vec![(1, 1.0)];
        let metric = compute_eviction_proxy(&evicted, &cache, &config);

        assert!(metric.raw_value > 0.0);
        assert!(metric.raw_value <= 1.0);
        assert_eq!(metric.per_head.as_ref().unwrap().len(), kv_heads);
    }
}
