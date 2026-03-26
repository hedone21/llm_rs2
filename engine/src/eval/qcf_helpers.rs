//! Shared QCF/OPR metric aggregation utilities.
//!
//! Eliminates duplicated aggregation code across run_eval_ll, run_kivi_eval_ll,
//! and run_ppl in generate.rs.

use super::hook::MetricsSummary;
use crate::core::qcf::QcfMetric;

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
