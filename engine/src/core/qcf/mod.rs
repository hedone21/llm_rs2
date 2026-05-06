//! QCF (Quality Cost Function) based degradation estimation for lossy actions.
//!
//! Each lossy action (H2O eviction, KIVI quantization, SWIFT layer skip)
//! produces a `QcfMetric` as a side effect.
//! A `DegradationEstimator` converts QCF values to estimated PPL increase
//! via offline-calibrated piecewise-linear coefficients.

pub mod entropy;
pub mod estimator;
pub mod layer_importance;
pub mod quant_qcf;
pub mod skip_qcf;
pub mod topk_retention;
pub mod unified_qcf;

pub use entropy::{EntropyResult, compute_normalized_entropy};
pub use estimator::DegradationEstimator;
pub use layer_importance::{ImportanceCollector, ImportanceTable, SubLayer};
pub use quant_qcf::{
    FlushAttentionParams, KiviFlushParams, compute_flush_aw_vopr, compute_flush_awqe,
    compute_flush_nmse, compute_flush_opr,
};
pub use skip_qcf::SkipQcfTracker;
pub use topk_retention::{TopKRetentionResult, compute_topk_retention};
pub use unified_qcf::{
    QcfActionType, UnifiedQcfParams, VDataSource, compute_unified_qcf,
    identify_retained_for_action, identify_retained_h2o, identify_retained_sliding,
};

/// A QCF metric collected from a single lossy action execution.
#[derive(Debug, Clone)]
pub struct QcfMetric {
    /// Action that produced this metric (e.g., "h2o", "snapkv", "kivi", "swift").
    pub action: String,
    /// Aggregated QCF value in [0, 1] range (higher = more degradation).
    pub raw_value: f32,
    /// Normalized QCF value for cross-policy comparison.
    /// For eviction: `evicted_importance / remaining_importance` (unbounded above 1).
    /// For non-eviction actions: same as `raw_value`.
    pub normalized_value: f32,
    /// Per-head QCF values (if applicable). Layout: `[n_kv_heads]`.
    pub per_head: Option<Vec<f32>>,
    /// Number of tokens affected by the action.
    pub tokens_affected: usize,
}

/// Which QCF variant(s) to compute for eviction events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QcfMode {
    /// Attention × V-norm ratio (original proxy).
    Attn,
    /// CAOTE-based eviction error (softmax redistribution + value direction).
    Caote,
    /// Compute both variants.
    Both,
}

impl QcfMode {
    pub fn has_attn(self) -> bool {
        matches!(self, QcfMode::Attn | QcfMode::Both)
    }

    pub fn has_caote(self) -> bool {
        matches!(self, QcfMode::Caote | QcfMode::Both)
    }
}

/// Configuration for QCF metric collection.
#[derive(Debug, Clone)]
pub struct QcfConfig {
    /// Whether QCF collection is enabled.
    pub enabled: bool,
    /// Which QCF variant(s) to compute.
    pub mode: QcfMode,
    /// Head aggregation strategy.
    pub aggregation: AggregationMode,
    /// Maximum degradation estimate (clamp ceiling). Default: 5.0.
    pub d_max: f32,
    /// Epsilon for division-by-zero guards. Default: 1e-8.
    pub epsilon: f32,
}

impl Default for QcfConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: QcfMode::Attn,
            aggregation: AggregationMode::Mean,
            d_max: 5.0,
            epsilon: 1e-8,
        }
    }
}

/// Head-level QCF aggregation strategy.
#[derive(Debug, Clone)]
pub enum AggregationMode {
    /// Simple mean across heads.
    Mean,
    /// Softmax-weighted aggregation favoring worst-case heads.
    /// Lower temperature = more emphasis on worst head.
    Defensive { temperature: f32 },
    /// Maximum value across heads (strict worst-case).
    Max,
    /// Mean of the top-k worst-case heads.
    /// k=0 returns 0.0; k > len returns mean of all heads.
    TopK { k: usize },
}

/// Aggregate per-head QCF values into a single scalar.
///
/// - `Mean`: arithmetic mean.
/// - `Defensive`: softmax-weighted mean (DefensiveKV, 2025) emphasizing worst-case heads.
/// - `Max`: maximum value (strict worst-case head).
/// - `TopK { k }`: mean of the top-k largest values.
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
        AggregationMode::Max => per_head.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        AggregationMode::TopK { k } => {
            if *k == 0 {
                return 0.0;
            }
            let take = (*k).min(per_head.len());
            let mut sorted = per_head.to_vec();
            // Sort descending (NaN-safe: treat NaN as smallest)
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Less));
            sorted[..take].iter().sum::<f32>() / take as f32
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

    #[test]
    fn test_aggregate_heads_max() {
        let values = vec![0.1, 0.5, 0.2];
        let result = aggregate_heads(&values, &AggregationMode::Max);
        assert!((result - 0.5).abs() < 1e-6, "expected 0.5, got {result}");
    }

    #[test]
    fn test_aggregate_heads_topk_k1_equals_max() {
        let values = vec![0.1, 0.5, 0.2];
        let topk1 = aggregate_heads(&values, &AggregationMode::TopK { k: 1 });
        let max = aggregate_heads(&values, &AggregationMode::Max);
        assert!(
            (topk1 - max).abs() < 1e-6,
            "TopK{{k=1}}={topk1} should equal Max={max}"
        );
    }

    #[test]
    fn test_aggregate_heads_topk_k_exceeds_len() {
        // k=5 but only 2 elements → should return mean of all elements
        let values = vec![0.1, 0.2];
        let result = aggregate_heads(&values, &AggregationMode::TopK { k: 5 });
        let expected = 0.15;
        assert!(
            (result - expected).abs() < 1e-6,
            "expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_aggregate_heads_topk_k0() {
        let values = vec![0.1, 0.5, 0.2];
        let result = aggregate_heads(&values, &AggregationMode::TopK { k: 0 });
        assert_eq!(result, 0.0, "k=0 should return 0.0 guard");
    }
}
