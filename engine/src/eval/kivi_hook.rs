//! KiviHook: StepHook implementation for KIVI quantization flush metric collection.
//!
//! Encapsulates the KIVI flush proxy collection logic previously embedded in
//! `run_kivi_eval_ll` (generate.rs).
//!
//! KIVI does not perform eviction — instead, when the FP32 residual buffer fills,
//! it batch-quantizes those tokens to Q2 and records the NMSE/OPR degradation as
//! a `QcfMetric`. KiviHook collects those metrics from `take_flush_proxies()` after
//! each prefill and decode step.

use super::hook::{CacheSnapshot, MetricsSummary, PostStepResult, StepHook};
use super::qcf_helpers::{aggregate_kivi_metrics, flush_metric_to_json};
use crate::core::attention_scores::AttentionScoreAccumulator;
use crate::core::kivi_cache::KiviCache;

/// KiviCache snapshot for choice-level restore.
///
/// Uses `Clone` because `KiviCache` implements `Clone` and all fields are heap-allocated.
pub struct KiviCacheSnapshot {
    caches: Vec<KiviCache>,
}

impl CacheSnapshot<KiviCache> for KiviCacheSnapshot {
    fn restore_to(&self, caches: &mut [KiviCache]) {
        caches.clone_from_slice(&self.caches);
    }
}

/// StepHook for KIVI quantization flush metric collection.
///
/// After each prefill and decode step, drains `take_flush_proxies()` from
/// `caches[0]` (layer 0 is used as the representative layer — all layers
/// experience the same flush pattern). Metrics from `caches[1..]` are discarded.
///
/// Does not perform eviction; `PostStepResult` is always the default (no eviction).
pub struct KiviHook {
    /// QCF metric collection config (used for aggregation strategy).
    pub qcf_config: crate::core::qcf::QcfConfig,
    /// Running count of flush events for the current question.
    flush_count: usize,
}

impl KiviHook {
    pub fn new(qcf_config: crate::core::qcf::QcfConfig) -> Self {
        Self {
            qcf_config,
            flush_count: 0,
        }
    }
}

impl StepHook<KiviCache> for KiviHook {
    fn post_decode_step(
        &mut self,
        caches: &mut [KiviCache],
        _step: usize,
        qcf_metrics: &mut Vec<serde_json::Value>,
    ) -> PostStepResult {
        if caches.is_empty() {
            return PostStepResult::default();
        }

        // Layer 0 is the representative: collect its flush proxies.
        for metric in caches[0].take_flush_proxies() {
            qcf_metrics.push(flush_metric_to_json(&metric, self.flush_count));
            self.flush_count += 1;
        }
        // Drain other layers to keep them clean (discard — layer 0 is canonical).
        for cache in caches[1..].iter_mut() {
            cache.take_flush_proxies();
        }

        PostStepResult::default()
    }

    fn post_prefill(&mut self, caches: &mut [KiviCache], qcf_metrics: &mut Vec<serde_json::Value>) {
        if caches.is_empty() {
            return;
        }
        for metric in caches[0].take_flush_proxies() {
            qcf_metrics.push(flush_metric_to_json(&metric, self.flush_count));
            self.flush_count += 1;
        }
        for cache in caches[1..].iter_mut() {
            cache.take_flush_proxies();
        }
    }

    fn reset_caches(&mut self, caches: &mut [KiviCache]) {
        for cache in caches.iter_mut() {
            cache.reset();
        }
        self.flush_count = 0;
    }

    fn snapshot(&self, caches: &[KiviCache]) -> Box<dyn CacheSnapshot<KiviCache>> {
        Box::new(KiviCacheSnapshot {
            caches: caches.to_vec(),
        })
    }

    fn score_accumulator(&mut self) -> Option<&mut AttentionScoreAccumulator> {
        None
    }

    fn extra_question_fields(&self, caches: &[KiviCache]) -> serde_json::Value {
        if caches.is_empty() {
            serde_json::json!({
                "kivi_q2_tokens": 0,
                "kivi_res_pos": 0,
            })
        } else {
            serde_json::json!({
                "kivi_q2_tokens": caches[0].q2_tokens,
                "kivi_res_pos": caches[0].res_pos,
            })
        }
    }

    fn extra_config_fields(&self) -> serde_json::Value {
        serde_json::json!({})
    }

    fn aggregate_metrics(&self, qcf_metrics: &[serde_json::Value]) -> MetricsSummary {
        aggregate_kivi_metrics(qcf_metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::kivi_cache::KiviCache;
    use crate::core::qcf::QcfConfig;

    fn make_hook() -> KiviHook {
        KiviHook::new(QcfConfig::default())
    }

    #[test]
    fn test_post_decode_step_empty_caches() {
        let mut hook = make_hook();
        let mut metrics = vec![];
        let result = hook.post_decode_step(&mut [], 0, &mut metrics);
        assert!(!result.evicted);
        assert!(metrics.is_empty());
    }

    #[test]
    fn test_extra_question_fields_empty() {
        let hook = make_hook();
        let fields = hook.extra_question_fields(&[]);
        assert_eq!(fields["kivi_q2_tokens"], 0);
        assert_eq!(fields["kivi_res_pos"], 0);
    }

    #[test]
    fn test_extra_question_fields_with_cache() {
        let hook = make_hook();
        let cache = KiviCache::new(8, 64, 512, 32);
        // Initial state: q2_tokens=0, res_pos=0
        let fields = hook.extra_question_fields(&[cache]);
        assert_eq!(fields["kivi_q2_tokens"], 0);
        assert_eq!(fields["kivi_res_pos"], 0);
    }

    #[test]
    fn test_aggregate_metrics_delegates_to_kivi() {
        let hook = make_hook();
        let metrics = vec![
            serde_json::json!({"action": "kivi", "raw_value": 0.1, "normalized_value": 0.1}),
            serde_json::json!({"action": "kivi_opr", "raw_value": 0.05, "normalized_value": 0.05}),
        ];
        let summary = hook.aggregate_metrics(&metrics);
        assert!((summary.qcf_attn_total - 0.1).abs() < 1e-10);
        assert_eq!(summary.opr_quantization, Some(0.05));
        assert_eq!(summary.opr_quantization_events, 1);
    }

    #[test]
    fn test_score_accumulator_returns_none() {
        let mut hook = make_hook();
        assert!(hook.score_accumulator().is_none());
    }

    #[test]
    fn test_extra_config_fields_empty() {
        let hook = make_hook();
        let fields = hook.extra_config_fields();
        // Should be an empty JSON object.
        assert!(fields.as_object().map(|o| o.is_empty()).unwrap_or(false));
    }

    #[test]
    fn test_snapshot_and_restore_empty() {
        let hook = make_hook();
        let snapshot = hook.snapshot(&[]);
        snapshot.restore_to(&mut []);
    }

    #[test]
    fn test_snapshot_and_restore_single_cache() {
        let hook = make_hook();
        let original = KiviCache::new(8, 64, 512, 32);
        let snap = hook.snapshot(&[original.clone()]);

        // Modify a second copy then restore.
        let mut modified = vec![KiviCache::new(8, 64, 512, 32)];
        snap.restore_to(&mut modified);
        // After restore, q2_tokens and res_pos should match the original.
        assert_eq!(modified[0].q2_tokens, original.q2_tokens);
        assert_eq!(modified[0].res_pos, original.res_pos);
    }

    #[test]
    fn test_reset_caches_resets_flush_count() {
        let mut hook = make_hook();
        hook.flush_count = 5;
        let mut caches = vec![KiviCache::new(8, 64, 512, 32)];
        hook.reset_caches(&mut caches);
        assert_eq!(hook.flush_count, 0);
    }
}
