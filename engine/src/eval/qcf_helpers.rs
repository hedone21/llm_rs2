//! Shared QCF/OPR metric aggregation utilities.
//!
//! Eliminates duplicated aggregation code across run_eval_ll, run_kivi_eval_ll,
//! and run_ppl in generate.rs.
//!
//! Also provides `QcfSwapDumpContext` + `dump_qcf_swap_json` for the layer-swap
//! QCF↔NLL measurement workflow (Phase 1, zazzy-herding-bonbon plan).

use super::hook::MetricsSummary;
use super::output::EvalOutput;
use crate::core::qcf::QcfMetric;
use crate::core::qcf::layer_importance::ImportanceTable;
use crate::models::weights::QuantNoiseTable;

/// Aggregate eviction QCF metrics from a JSON metrics array.
///
/// Filters by action suffix ("_attn", "_caote") and sums raw/normalized values.
pub fn aggregate_eviction_metrics(qcf_metrics: &[serde_json::Value]) -> MetricsSummary {
    let qcf_attn_total: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str().is_some_and(|a| a.ends_with("_attn")))
        .filter_map(|m| m["raw_value"].as_f64())
        .sum();
    let qcf_caote_total: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str().is_some_and(|a| a.ends_with("_caote")))
        .filter_map(|m| m["raw_value"].as_f64())
        .sum();
    let qcf_normalized_total: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str().is_some_and(|a| a.ends_with("_attn")))
        .filter_map(|m| m["normalized_value"].as_f64())
        .sum();
    MetricsSummary {
        qcf_attn_total,
        qcf_caote_total,
        qcf_normalized_total,
        qcf_kivi_opr: None,
        qcf_kivi_opr_events: 0,
    }
}

/// Aggregate KIVI quantization metrics from a JSON metrics array.
///
/// Separates NMSE ("kivi" action) from OPR ("kivi_opr" action).
pub fn aggregate_kivi_metrics(qcf_metrics: &[serde_json::Value]) -> MetricsSummary {
    let qcf_attn_total: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() != Some("kivi_opr"))
        .filter_map(|m| m["raw_value"].as_f64())
        .sum();
    let qcf_normalized_total: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() != Some("kivi_opr"))
        .filter_map(|m| m["normalized_value"].as_f64())
        .sum();
    let kivi_opr_sum: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() == Some("kivi_opr"))
        .filter_map(|m| m["raw_value"].as_f64())
        .sum();
    let qcf_kivi_opr_events: usize = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() == Some("kivi_opr"))
        .count();

    MetricsSummary {
        qcf_attn_total,
        qcf_caote_total: 0.0,
        qcf_normalized_total,
        qcf_kivi_opr: if kivi_opr_sum > 0.0 {
            Some(kivi_opr_sum)
        } else {
            None
        },
        qcf_kivi_opr_events,
    }
}

/// Convert a QcfMetric to a JSON value for the qcf_metrics array.
pub fn metric_to_json(
    metric: &QcfMetric,
    step: usize,
    cache_pos_before: usize,
) -> serde_json::Value {
    serde_json::json!({
        "step": step,
        "action": metric.action,
        "raw_value": metric.raw_value,
        "normalized_value": metric.normalized_value,
        "tokens_affected": metric.tokens_affected,
        "cache_pos_before": cache_pos_before,
    })
}

/// Convert a flush QcfMetric (KIVI) to a JSON value.
pub fn flush_metric_to_json(metric: &QcfMetric, flush_count: usize) -> serde_json::Value {
    serde_json::json!({
        "flush": flush_count,
        "action": metric.action,
        "raw_value": metric.raw_value,
        "normalized_value": metric.normalized_value,
        "tokens_quantized": metric.tokens_affected,
    })
}

/// Build QCF JSON fields from a MetricsSummary.
pub fn build_qcf_fields(summary: &MetricsSummary) -> serde_json::Value {
    serde_json::json!({
        "qcf_kivi_opr_total": summary.qcf_kivi_opr,
        "qcf_kivi_opr_events": summary.qcf_kivi_opr.map(|_| summary.qcf_kivi_opr_events),
    })
}

// ── Layer-swap QCF dump infrastructure ───────────────────────────────────────

/// Context for one layer-swap QCF measurement run.
///
/// Collected by the `--qcf-dump` workflow in generate.rs and serialized to
/// a JSON file by `dump_qcf_swap_json`.  The external harness
/// (`pact2026/experiments/scripts/`) reads these files and correlates
/// `qcf_swap_predicted` with actual NLL/quality metrics.
pub struct QcfSwapDumpContext<'a> {
    pub model_arch: &'a str,
    pub model_path: &'a str,
    pub secondary_path: Option<&'a str>,
    pub primary_dtype: &'a str,
    pub secondary_dtype: &'a str,
    pub num_layers: usize,
    pub force_swap_ratio: Option<f32>,
    /// Decoder layer indices that were swapped (empty if ratio=0 or no secondary).
    pub swap_set: &'a [usize],
    /// QCF_swap predicted value from `WeightSwapDecider` (ENG-ALG-217).
    pub qcf_swap_predicted: f32,
    /// `true` when uniform fallback was used (importance/noise absent).
    pub fallback_used: bool,
    /// Full importance table built from warmup prefill (optional).
    pub importance_table: Option<&'a ImportanceTable>,
    /// Quantization noise table built from secondary mmap (optional).
    pub noise_table: Option<&'a QuantNoiseTable>,
    /// Perplexity result from `run_ppl()` (None in generation mode).
    pub ppl: Option<f64>,
    /// Average negative log-likelihood (None in generation mode).
    pub avg_nll: Option<f64>,
    /// Number of tokens evaluated in the main measurement.
    pub n_eval_tokens: usize,
    /// Total wall-clock time in seconds (warmup + swap + measurement).
    pub wall_time_s: f64,
    /// Number of warmup prefill tokens used to build the importance table.
    pub warmup_tokens: usize,
    /// Backend string (e.g. "cpu", "opencl", "cuda").
    pub backend: &'a str,
    /// KV cache dtype string (e.g. "f16", "f32", "q4_0").
    pub kv_type: &'a str,
    /// Path to the PPL reference corpus file (None in generation mode).
    pub ppl_corpus: Option<&'a str>,
    /// EvalOutput from `--eval-ll` mode (per-question NLL summary).
    ///
    /// Set when `--eval-ll` and `--qcf-dump` are both active. None in PPL/generation mode.
    /// The full `EvalOutput` is serialized as `eval_ll_output` in the JSON dump so the
    /// external harness can compute `qcf_swap_predicted ↔ ΔNLL` Spearman ρ directly.
    pub eval_ll_output: Option<&'a EvalOutput>,
}

/// Serialize a `QcfSwapDumpContext` to a JSON file (schema_version 1).
///
/// The JSON schema matches the external harness expectation exactly:
/// - All fields are always present (`null` when absent, NOT omitted).
/// - `importance_table` and `noise_table` include all collected entries.
/// - `noise_table` excludes NaN/non-finite ε entries.
/// - `swap_count` is derived as `swap_set.len()`.
pub fn dump_qcf_swap_json(
    path: &std::path::Path,
    ctx: &QcfSwapDumpContext<'_>,
) -> anyhow::Result<()> {
    use serde_json::{Value, json};

    // importance_table entries
    let importance_arr: Value = match ctx.importance_table {
        Some(table) => {
            let entries: Vec<Value> = table
                .entries()
                .iter()
                .map(|e| {
                    json!({
                        "layer": e.layer_id,
                        "sublayer": format!("{:?}", e.sublayer),
                        "importance": e.importance,
                        "opr": e.opr,
                    })
                })
                .collect();
            Value::Array(entries)
        }
        None => Value::Null,
    };

    // noise_table entries: only finite ε values
    let noise_arr: Value = match ctx.noise_table {
        Some(table) => {
            let entries: Vec<Value> = (0..ctx.num_layers)
                .filter_map(|i| {
                    table.epsilon(i).map(|eps| {
                        json!({
                            "layer": i,
                            "epsilon": eps,
                        })
                    })
                })
                .collect();
            Value::Array(entries)
        }
        None => Value::Null,
    };

    // eval_ll_output: serialize as JSON object when present, null otherwise.
    let eval_ll_output_val: Value = match ctx.eval_ll_output {
        Some(output) => {
            let json_str = output.to_json().unwrap_or_else(|_| "null".to_string());
            serde_json::from_str(&json_str).unwrap_or(Value::Null)
        }
        None => Value::Null,
    };

    let doc = json!({
        "schema_version": 1,
        "model_arch": ctx.model_arch,
        "model_path": ctx.model_path,
        "secondary_path": ctx.secondary_path,
        "primary_dtype": ctx.primary_dtype,
        "secondary_dtype": ctx.secondary_dtype,
        "num_layers": ctx.num_layers,
        "force_swap_ratio": ctx.force_swap_ratio,
        "swap_set": ctx.swap_set,
        "swap_count": ctx.swap_set.len(),
        "qcf_swap_predicted": ctx.qcf_swap_predicted,
        "fallback_used": ctx.fallback_used,
        "importance_table": importance_arr,
        "noise_table": noise_arr,
        "ppl": ctx.ppl,
        "avg_nll": ctx.avg_nll,
        "n_eval_tokens": ctx.n_eval_tokens,
        "wall_time_s": ctx.wall_time_s,
        "warmup_tokens": ctx.warmup_tokens,
        "backend": ctx.backend,
        "kv_type": ctx.kv_type,
        "ppl_corpus": ctx.ppl_corpus,
        "eval_ll_output": eval_ll_output_val,
    });

    let json_str = serde_json::to_string_pretty(&doc)?;
    std::fs::write(path, json_str)
        .map_err(|e| anyhow::anyhow!("Failed to write QCF dump to {}: {}", path.display(), e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_eviction_empty() {
        let summary = aggregate_eviction_metrics(&[]);
        assert_eq!(summary.qcf_attn_total, 0.0);
        assert_eq!(summary.qcf_caote_total, 0.0);
        assert!(summary.qcf_kivi_opr.is_none());
    }

    #[test]
    fn test_aggregate_eviction_mixed() {
        let metrics = vec![
            serde_json::json!({"action": "eviction_attn", "raw_value": 0.5, "normalized_value": 0.6}),
            serde_json::json!({"action": "eviction_caote", "raw_value": 0.3, "normalized_value": 0.3}),
            serde_json::json!({"action": "sliding_attn", "raw_value": 0.2, "normalized_value": 0.25}),
        ];
        let summary = aggregate_eviction_metrics(&metrics);
        assert!((summary.qcf_attn_total - 0.7).abs() < 1e-10);
        assert!((summary.qcf_caote_total - 0.3).abs() < 1e-10);
        assert!((summary.qcf_normalized_total - 0.85).abs() < 1e-10);
        // eviction 모드에서는 qcf_kivi_opr이 없어야 한다
        assert!(summary.qcf_kivi_opr.is_none());
        assert_eq!(summary.qcf_kivi_opr_events, 0);
    }

    #[test]
    fn test_aggregate_kivi_separates_opr() {
        let metrics = vec![
            serde_json::json!({"action": "kivi", "raw_value": 0.1, "normalized_value": 0.1}),
            serde_json::json!({"action": "kivi_opr", "raw_value": 0.05, "normalized_value": 0.05}),
            serde_json::json!({"action": "kivi", "raw_value": 0.2, "normalized_value": 0.2}),
            serde_json::json!({"action": "kivi_opr", "raw_value": 0.03, "normalized_value": 0.03}),
        ];
        let summary = aggregate_kivi_metrics(&metrics);
        assert!((summary.qcf_attn_total - 0.3).abs() < 1e-10); // NMSE only
        assert_eq!(summary.qcf_kivi_opr, Some(0.08));
        assert_eq!(summary.qcf_kivi_opr_events, 2);
    }

    #[test]
    fn test_metric_to_json() {
        let metric = QcfMetric {
            action: "eviction_attn".to_string(),
            raw_value: 0.42,
            normalized_value: 0.55,
            per_head: None,
            tokens_affected: 10,
        };
        let json = metric_to_json(&metric, 100, 512);
        assert_eq!(json["step"], 100);
        assert_eq!(json["action"], "eviction_attn");
        assert_eq!(json["cache_pos_before"], 512);
    }
}
