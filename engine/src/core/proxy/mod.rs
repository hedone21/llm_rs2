//! Proxy-based degradation estimation for lossy KV cache actions.
//!
//! Each lossy action (H2O eviction, SnapKV compression, KIVI quantization,
//! SWIFT layer skip) produces a `ProxyMetric` as a side effect.
//! A `DegradationEstimator` converts proxy values to estimated PPL increase
//! via offline-calibrated piecewise-linear coefficients.

pub mod estimator;
pub mod eviction_proxy;
pub mod quant_proxy;
pub mod skip_proxy;

pub use estimator::DegradationEstimator;
pub use eviction_proxy::{compute_eviction_proxy, compute_sliding_proxy, identify_evicted_h2o};
pub use quant_proxy::{FlushProxyParams, compute_flush_proxy};
pub use skip_proxy::SkipProxyTracker;

/// A proxy metric collected from a single lossy action execution.
#[derive(Debug, Clone)]
pub struct ProxyMetric {
    /// Action that produced this metric (e.g., "h2o", "snapkv", "kivi", "swift").
    pub action: String,
    /// Aggregated proxy value in [0, 1] range (higher = more degradation).
    pub raw_value: f32,
    /// Per-head proxy values (if applicable). Layout: `[n_kv_heads]`.
    pub per_head: Option<Vec<f32>>,
    /// Number of tokens affected by the action.
    pub tokens_affected: usize,
}

/// Configuration for proxy metric collection.
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Whether proxy collection is enabled.
    pub enabled: bool,
    /// Head aggregation strategy.
    pub aggregation: AggregationMode,
    /// Maximum degradation estimate (clamp ceiling). Default: 5.0.
    pub d_max: f32,
    /// Epsilon for division-by-zero guards. Default: 1e-8.
    pub epsilon: f32,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            aggregation: AggregationMode::Mean,
            d_max: 5.0,
            epsilon: 1e-8,
        }
    }
}

/// Head-level proxy aggregation strategy.
#[derive(Debug, Clone)]
pub enum AggregationMode {
    /// Simple mean across heads.
    Mean,
    /// Softmax-weighted aggregation favoring worst-case heads.
    /// Lower temperature = more emphasis on worst head.
    Defensive { temperature: f32 },
}

/// Aggregate per-head proxy values into a single scalar.
///
/// - `Mean`: arithmetic mean.
/// - `Defensive`: softmax-weighted mean (DefensiveKV, 2025) emphasizing worst-case heads.
pub fn aggregate_heads(per_head: &[f32], mode: &AggregationMode) -> f32 {
    if per_head.is_empty() {
        return 0.0;
    }
    match mode {
        AggregationMode::Mean => per_head.iter().sum::<f32>() / per_head.len() as f32,
        AggregationMode::Defensive { temperature } => {
            let temp = temperature.max(1e-6);
            let max_val = per_head.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            // Numerically stable softmax
            let exp_sum: f32 = per_head.iter().map(|&v| ((v - max_val) / temp).exp()).sum();
            if exp_sum < 1e-12 {
                return per_head.iter().sum::<f32>() / per_head.len() as f32;
            }
            per_head
                .iter()
                .map(|&v| {
                    let w = ((v - max_val) / temp).exp() / exp_sum;
                    w * v
                })
                .sum()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_heads_mean() {
        let values = vec![0.1, 0.2, 0.3];
        let result = aggregate_heads(&values, &AggregationMode::Mean);
        assert!((result - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_heads_defensive_emphasizes_worst() {
        let values = vec![0.1, 0.1, 0.9];
        let mean = aggregate_heads(&values, &AggregationMode::Mean);
        let defensive = aggregate_heads(&values, &AggregationMode::Defensive { temperature: 0.1 });
        // Defensive should be higher than mean (closer to 0.9)
        assert!(
            defensive > mean,
            "defensive={defensive} should > mean={mean}"
        );
        assert!(defensive > 0.5);
    }

    #[test]
    fn test_aggregate_heads_defensive_high_temp_approaches_mean() {
        let values = vec![0.1, 0.2, 0.3];
        let mean = aggregate_heads(&values, &AggregationMode::Mean);
        let defensive =
            aggregate_heads(&values, &AggregationMode::Defensive { temperature: 100.0 });
        assert!(
            (defensive - mean).abs() < 0.01,
            "high temp: defensive={defensive} should ≈ mean={mean}"
        );
    }

    #[test]
    fn test_aggregate_heads_empty() {
        assert_eq!(aggregate_heads(&[], &AggregationMode::Mean), 0.0);
        assert_eq!(
            aggregate_heads(&[], &AggregationMode::Defensive { temperature: 0.1 }),
            0.0
        );
    }

    #[test]
    fn test_aggregate_heads_single() {
        let values = vec![0.42];
        let result = aggregate_heads(&values, &AggregationMode::Mean);
        assert!((result - 0.42).abs() < 1e-6);
        let def = aggregate_heads(&values, &AggregationMode::Defensive { temperature: 0.1 });
        assert!((def - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_heads_uniform() {
        let values = vec![0.5, 0.5, 0.5, 0.5];
        let result = aggregate_heads(&values, &AggregationMode::Defensive { temperature: 0.1 });
        assert!((result - 0.5).abs() < 1e-6);
    }
}
