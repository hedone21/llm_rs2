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
    let opr_eviction_events: usize = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str().is_some_and(|a| a.ends_with("_caote")))
        .count();
    let opr_eviction = if qcf_caote_total > 0.0 {
        Some(qcf_caote_total)
    } else {
        None
    };

    MetricsSummary {
        qcf_attn_total,
        qcf_caote_total,
        qcf_normalized_total,
        opr_eviction,
        opr_eviction_events,
        opr_quantization: None,
        opr_quantization_events: 0,
    }
}

/// Aggregate KIVI quantization metrics from a JSON metrics array.
///
/// Separates NMSE ("kivi" action) from OPR ("kivi_opr" action).
pub fn aggregate_kivi_metrics(qcf_metrics: &[serde_json::Value]) -> MetricsSummary {
    let qcf_total: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() != Some("kivi_opr"))
        .filter_map(|m| m["raw_value"].as_f64())
        .sum();
    let qcf_normalized_total: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() != Some("kivi_opr"))
        .filter_map(|m| m["normalized_value"].as_f64())
        .sum();
    let opr_quantization: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() == Some("kivi_opr"))
        .filter_map(|m| m["raw_value"].as_f64())
        .sum();
    let opr_quantization_events: usize = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() == Some("kivi_opr"))
        .count();

    MetricsSummary {
        qcf_attn_total: qcf_total,
        qcf_caote_total: 0.0,
        qcf_normalized_total,
        opr_eviction: None,
        opr_eviction_events: 0,
        opr_quantization: if opr_quantization > 0.0 {
            Some(opr_quantization)
        } else {
            None
        },
        opr_quantization_events,
    }
}

/// Convert a QcfMetric to a JSON value for the qcf_metrics array.
pub fn metric_to_json(metric: &QcfMetric, step: usize, cache_pos_before: usize) -> serde_json::Value {
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

/// Build OPR JSON fields from a MetricsSummary.
pub fn build_opr_fields(summary: &MetricsSummary) -> serde_json::Value {
    serde_json::json!({
        "opr_eviction": summary.opr_eviction,
        "opr_eviction_events": summary.opr_eviction.map(|_| summary.opr_eviction_events),
        "opr_quantization": summary.opr_quantization,
        "opr_quantization_events": summary.opr_quantization.map(|_| summary.opr_quantization_events),
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
        assert!(summary.opr_eviction.is_none());
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
        assert_eq!(summary.opr_eviction, Some(0.3));
        assert_eq!(summary.opr_eviction_events, 1);
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
        assert_eq!(summary.opr_quantization, Some(0.08));
        assert_eq!(summary.opr_quantization_events, 2);
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
