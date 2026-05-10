//! KiviHook: StepHook implementation for KIVI quantization flush metric collection.
//!
//! Encapsulates the KIVI flush proxy collection logic previously embedded in
//! `run_kivi_eval_ll` (generate.rs).
//!
//! KIVI does not perform eviction — instead, when the FP32 residual buffer fills,
//! it batch-quantizes those tokens to Q2 and records the NMSE/OPR degradation as
//! a `QcfMetric`. KiviHook collects those metrics from `take_flush_proxies()` after
//! each prefill and decode step.

use super::hook::{CacheSnapshot, PostStepResult, StepHook};
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
/// the sample layers (default: layer 0 only, matching legacy behaviour).
/// Metrics from non-sample layers are drained and discarded.
///
/// Does not perform eviction; `PostStepResult` is always the default (no eviction).
pub struct KiviHook {
    /// QCF metric collection config (used for aggregation strategy).
    pub qcf_config: crate::core::qcf::QcfConfig,
    /// Whether to compute and dump experimental QCF metrics (ARGUS Step 5).
    pub experimental_enabled: bool,
    /// Sample layer indices for multi-layer flush proxy (ARGUS Step 5).
    /// Empty → use [0] for backward compat.
    pub qcf_sample_layers: Vec<usize>,
    /// Optional attention score accumulator (for entropy dump).
    pub score_accumulator: Option<AttentionScoreAccumulator>,

    /// Running count of flush events for the current question (layer 0 legacy).
    flush_count: usize,
    /// Max OPR value across all flushes for current question (legacy per-flush max).
    qcf_kivi_legacy: f32,
    /// Sum of OPR values across all flushes for current question (for mean).
    qcf_kivi_sum: f32,
    /// Max AW-VOPR value across all flushes for current question.
    qcf_aw_vopr_max: f32,
    /// Sum of AW-VOPR values across all flushes for current question.
    qcf_aw_vopr_sum: f32,
    /// Count of AW-VOPR metrics collected for current question.
    aw_vopr_count: usize,

    // ARGUS Step 5: per-layer accumulation (sample_layers index → accumulated values)
    per_layer_max: Vec<f32>,
    per_layer_sum: Vec<f32>,
    per_layer_count: Vec<usize>,

    // ARGUS Step 5: entropy (computed once at post_prefill when acc is active)
    attention_entropy: f32,
    attention_entropy_normalized: f32,
    entropy_computed: bool,
}

impl KiviHook {
    pub fn new(
        qcf_config: crate::core::qcf::QcfConfig,
        experimental_enabled: bool,
        qcf_sample_layers: Vec<usize>,
        score_accumulator: Option<AttentionScoreAccumulator>,
    ) -> Self {
        let n = qcf_sample_layers.len().max(1);
        Self {
            qcf_config,
            experimental_enabled,
            qcf_sample_layers,
            score_accumulator,
            flush_count: 0,
            qcf_kivi_legacy: 0.0,
            qcf_kivi_sum: 0.0,
            qcf_aw_vopr_max: 0.0,
            qcf_aw_vopr_sum: 0.0,
            aw_vopr_count: 0,
            per_layer_max: vec![0.0; n],
            per_layer_sum: vec![0.0; n],
            per_layer_count: vec![0; n],
            attention_entropy: 0.0,
            attention_entropy_normalized: 0.0,
            entropy_computed: false,
        }
    }

    /// Drain flush proxies from sample layers, track max OPR and flush count.
    ///
    /// Legacy path (experimental_enabled=false): layer 0 only.
    /// Experimental path: all sample_layers, with per-layer buffers.
    fn collect_flush_proxies(&mut self, caches: &mut [KiviCache]) {
        if caches.is_empty() {
            return;
        }

        if self.experimental_enabled {
            let sample_layers: Vec<usize> = if self.qcf_sample_layers.is_empty() {
                vec![0]
            } else {
                self.qcf_sample_layers.clone()
            };

            // Ensure per-layer buffers are sized correctly.
            if self.per_layer_max.len() != sample_layers.len() {
                self.per_layer_max = vec![0.0; sample_layers.len()];
                self.per_layer_sum = vec![0.0; sample_layers.len()];
                self.per_layer_count = vec![0; sample_layers.len()];
            }

            let sample_set: std::collections::HashSet<usize> =
                sample_layers.iter().copied().collect();

            for (cache_idx, cache) in caches.iter_mut().enumerate() {
                if sample_set.contains(&cache_idx) {
                    let pos = sample_layers.iter().position(|&v| v == cache_idx).unwrap();
                    for metric in cache.take_flush_proxies() {
                        if metric.action == "kivi_opr" {
                            // Layer 0 in sample → also update legacy fields for backward compat.
                            if cache_idx == 0 {
                                self.qcf_kivi_legacy = self.qcf_kivi_legacy.max(metric.raw_value);
                                self.qcf_kivi_sum += metric.raw_value;
                                self.flush_count += 1;
                            }
                            self.per_layer_max[pos] = self.per_layer_max[pos].max(metric.raw_value);
                            self.per_layer_sum[pos] += metric.raw_value;
                            self.per_layer_count[pos] += 1;
                        } else if metric.action == "aw_vopr" {
                            self.qcf_aw_vopr_max = self.qcf_aw_vopr_max.max(metric.raw_value);
                            self.qcf_aw_vopr_sum += metric.raw_value;
                            self.aw_vopr_count += 1;
                        }
                    }
                } else {
                    // Drain non-sample layers to prevent proxy queue growth.
                    cache.take_flush_proxies();
                }
            }
        } else {
            // Legacy path: layer 0 only, no per-layer tracking.
            for metric in caches[0].take_flush_proxies() {
                if metric.action == "kivi_opr" {
                    self.qcf_kivi_legacy = self.qcf_kivi_legacy.max(metric.raw_value);
                    self.qcf_kivi_sum += metric.raw_value;
                    self.flush_count += 1;
                } else if metric.action == "aw_vopr" {
                    self.qcf_aw_vopr_max = self.qcf_aw_vopr_max.max(metric.raw_value);
                    self.qcf_aw_vopr_sum += metric.raw_value;
                    self.aw_vopr_count += 1;
                }
                // NMSE ("kivi" action) tracked but not exposed as per-question scalar
            }
            for cache in caches[1..].iter_mut() {
                cache.take_flush_proxies();
            }
        }
    }
}

impl StepHook<KiviCache> for KiviHook {
    fn post_decode_step(&mut self, caches: &mut [KiviCache], _step: usize) -> PostStepResult {
        self.collect_flush_proxies(caches);
        PostStepResult::default()
    }

    fn post_prefill(&mut self, caches: &mut [KiviCache]) {
        self.collect_flush_proxies(caches);

        // ARGUS Step 5: attention entropy from accumulated importance scores.
        if self.experimental_enabled
            && let Some(ref acc) = self.score_accumulator
            && acc.is_active()
        {
            let scores = acc.importance_scores();
            let r = crate::core::qcf::compute_normalized_entropy(scores);
            self.attention_entropy = r.entropy;
            self.attention_entropy_normalized = r.entropy_normalized;
            self.entropy_computed = true;
        }
    }

    fn reset_caches(&mut self, caches: &mut [KiviCache]) {
        for cache in caches.iter_mut() {
            cache.reset();
        }
        self.flush_count = 0;
        self.qcf_kivi_legacy = 0.0;
        self.qcf_kivi_sum = 0.0;
        self.qcf_aw_vopr_max = 0.0;
        self.qcf_aw_vopr_sum = 0.0;
        self.aw_vopr_count = 0;
        // Step 5: reset per-layer buffers
        for v in self.per_layer_max.iter_mut() {
            *v = 0.0;
        }
        for v in self.per_layer_sum.iter_mut() {
            *v = 0.0;
        }
        for v in self.per_layer_count.iter_mut() {
            *v = 0;
        }
        self.attention_entropy = 0.0;
        self.attention_entropy_normalized = 0.0;
        self.entropy_computed = false;
        if let Some(ref mut acc) = self.score_accumulator {
            acc.reset();
        }
    }

    fn snapshot(&self, caches: &[KiviCache]) -> Box<dyn CacheSnapshot<KiviCache>> {
        Box::new(KiviCacheSnapshot {
            caches: caches.to_vec(),
        })
    }

    fn score_accumulator(&mut self) -> Option<&mut AttentionScoreAccumulator> {
        self.score_accumulator.as_mut()
    }

    fn extra_question_fields(&self, caches: &[KiviCache]) -> serde_json::Value {
        let (q2_tokens, res_pos) = if caches.is_empty() {
            (0, 0)
        } else {
            (caches[0].q2_tokens, caches[0].res_pos)
        };
        let mut obj = serde_json::json!({
            "kivi_q2_tokens": q2_tokens,
            "kivi_res_pos": res_pos,
        });
        if self.flush_count > 0 {
            let qcf_mean = self.qcf_kivi_sum / self.flush_count as f32;
            obj["qcf"] = serde_json::json!(qcf_mean);
            obj["qcf_kivi_mean"] = serde_json::json!(qcf_mean);
            obj["qcf_kivi_max"] = serde_json::json!(self.qcf_kivi_legacy);
            obj["qcf_kivi_flush_count"] = serde_json::json!(self.flush_count);
        }
        if self.aw_vopr_count > 0 {
            obj["qcf_kivi_aw_vopr_max"] = serde_json::json!(self.qcf_aw_vopr_max);
            obj["qcf_kivi_aw_vopr_sum"] = serde_json::json!(self.qcf_aw_vopr_sum);
            obj["qcf_kivi_aw_vopr_mean"] =
                serde_json::json!(self.qcf_aw_vopr_sum / self.aw_vopr_count as f32);
            obj["qcf_kivi_aw_vopr_count"] = serde_json::json!(self.aw_vopr_count);
        }

        if self.experimental_enabled {
            let indices: Vec<usize> = if self.qcf_sample_layers.is_empty() {
                vec![0]
            } else {
                self.qcf_sample_layers.clone()
            };
            let n = indices.len();
            // Ensure buffers are at least length n (guard against uninitialised state).
            let max_vec: Vec<f32> = self.per_layer_max.iter().copied().take(n).collect();
            let mean_vec: Vec<f32> = self
                .per_layer_sum
                .iter()
                .zip(self.per_layer_count.iter())
                .take(n)
                .map(|(s, c)| if *c > 0 { *s / (*c as f32) } else { 0.0 })
                .collect();

            obj["qcf_kivi_per_layer_indices"] = serde_json::json!(indices);
            obj["qcf_kivi_per_layer_max"] = serde_json::json!(max_vec);
            obj["qcf_kivi_per_layer_mean"] = serde_json::json!(mean_vec);

            // Layer-level aggregations.
            use crate::core::qcf::{LayerAggregationMode, aggregate_layers};
            obj["qcf_kivi_layer_max"] =
                serde_json::json!(aggregate_layers(&max_vec, &LayerAggregationMode::Max));
            obj["qcf_kivi_layer_defensive_t01"] = serde_json::json!(aggregate_layers(
                &max_vec,
                &LayerAggregationMode::Defensive { temperature: 0.1 }
            ));

            // Entropy — present only when successfully measured.
            if self.entropy_computed {
                obj["attention_entropy_kivi"] = serde_json::json!(self.attention_entropy);
                obj["attention_entropy_normalized_kivi"] =
                    serde_json::json!(self.attention_entropy_normalized);
            }
        }

        obj
    }

    fn extra_config_fields(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::kivi_cache::KiviCache;
    use crate::core::qcf::QcfConfig;

    fn make_hook() -> KiviHook {
        KiviHook::new(QcfConfig::default(), false, vec![0], None)
    }

    fn make_hook_experimental(sample_layers: Vec<usize>) -> KiviHook {
        KiviHook::new(QcfConfig::default(), true, sample_layers, None)
    }

    #[test]
    fn test_post_decode_step_empty_caches() {
        let mut hook = make_hook();
        let result = hook.post_decode_step(&mut [], 0);
        assert!(!result.evicted);
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
    fn test_qcf_kivi_legacy_tracking() {
        let mut hook = make_hook();
        // Simulate flush proxies being collected
        // Directly test collect_flush_proxies by creating a cache with proxies
        let cache = KiviCache::new(8, 64, 512, 32);

        // Inject proxies manually via the public method (if available)
        // Since we can't inject proxies directly, test the fields after hook operations
        assert_eq!(hook.flush_count, 0);
        assert_eq!(hook.qcf_kivi_legacy, 0.0);

        // Set values and verify output fields
        hook.flush_count = 5;
        hook.qcf_kivi_legacy = 0.5; // max
        hook.qcf_kivi_sum = 1.48; // sum → mean = 1.48/5 = 0.296
        let fields = hook.extra_question_fields(&[cache]);
        let expected_mean = 1.48_f32 / 5.0;
        assert!((fields["qcf"].as_f64().unwrap() - expected_mean as f64).abs() < 1e-5);
        assert_eq!(fields["qcf_kivi_max"], 0.5_f32 as f64);
        assert!((fields["qcf_kivi_mean"].as_f64().unwrap() - expected_mean as f64).abs() < 1e-5);
        assert_eq!(fields["qcf_kivi_flush_count"], 5);

        // After reset_caches
        let cache2 = KiviCache::new(8, 64, 512, 32);
        hook.reset_caches(&mut [cache2]);
        assert_eq!(hook.flush_count, 0);
        assert_eq!(hook.qcf_kivi_legacy, 0.0);
    }

    #[test]
    fn test_score_accumulator_returns_none() {
        let mut hook = make_hook();
        assert!(hook.score_accumulator().is_none());
    }

    #[test]
    fn test_score_accumulator_returns_some_when_provided() {
        use crate::core::attention_scores::AttentionScoreAccumulator;
        // new(max_seq_len, n_heads, total_layers, last_n_layers, decay)
        let acc = AttentionScoreAccumulator::new(512, 32, 16, 0, 1.0);
        let mut hook = KiviHook::new(QcfConfig::default(), false, vec![0], Some(acc));
        assert!(
            hook.score_accumulator().is_some(),
            "score_accumulator() should return Some when acc is injected"
        );
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
        let snap = hook.snapshot(std::slice::from_ref(&original));

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

    // ── ARGUS Step 5 tests ──────────────────────────────────────────────────

    #[test]
    fn test_per_layer_indices_in_extra_fields_when_experimental() {
        let hook = make_hook_experimental(vec![0, 4, 8]);
        let fields = hook.extra_question_fields(&[]);
        let indices = fields["qcf_kivi_per_layer_indices"]
            .as_array()
            .expect("qcf_kivi_per_layer_indices must be present when experimental_enabled");
        assert_eq!(indices.len(), 3);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 4);
        assert_eq!(indices[2], 8);

        // Per-layer max/mean arrays also present
        assert!(
            fields["qcf_kivi_per_layer_max"].is_array(),
            "qcf_kivi_per_layer_max must be an array"
        );
        assert!(
            fields["qcf_kivi_per_layer_mean"].is_array(),
            "qcf_kivi_per_layer_mean must be an array"
        );

        // Layer aggregation scalars present
        assert!(
            fields["qcf_kivi_layer_max"].is_number(),
            "qcf_kivi_layer_max must be a number"
        );
        assert!(
            fields["qcf_kivi_layer_defensive_t01"].is_number(),
            "qcf_kivi_layer_defensive_t01 must be a number"
        );
    }

    #[test]
    fn test_no_per_layer_fields_when_not_experimental() {
        let hook = make_hook();
        let fields = hook.extra_question_fields(&[]);
        assert!(
            fields.get("qcf_kivi_per_layer_indices").is_none(),
            "qcf_kivi_per_layer_indices must be absent when experimental_enabled=false"
        );
        assert!(
            fields.get("qcf_kivi_layer_max").is_none(),
            "qcf_kivi_layer_max must be absent when experimental_enabled=false"
        );
        assert!(
            fields.get("attention_entropy_kivi").is_none(),
            "attention_entropy_kivi must be absent when experimental_enabled=false"
        );
    }

    #[test]
    fn test_reset_caches_clears_step5_fields() {
        let mut hook = make_hook_experimental(vec![0, 4]);
        // Inject some artificial per-layer values.
        hook.per_layer_max[0] = 0.9;
        hook.per_layer_sum[0] = 1.8;
        hook.per_layer_count[0] = 2;
        hook.attention_entropy = 1.234; // arbitrary non-zero sentinel value
        hook.entropy_computed = true;

        let mut caches = vec![KiviCache::new(8, 64, 512, 32)];
        hook.reset_caches(&mut caches);

        assert_eq!(hook.per_layer_max[0], 0.0);
        assert_eq!(hook.per_layer_sum[0], 0.0);
        assert_eq!(hook.per_layer_count[0], 0);
        assert_eq!(hook.attention_entropy, 0.0);
        assert!(!hook.entropy_computed);
    }

    #[test]
    fn test_experimental_empty_sample_layers_fallback_to_layer0() {
        // Empty qcf_sample_layers → runtime fallback to [0].
        let hook = make_hook_experimental(vec![]);
        let fields = hook.extra_question_fields(&[]);
        let indices = fields["qcf_kivi_per_layer_indices"]
            .as_array()
            .expect("indices must be present");
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
    }
}
