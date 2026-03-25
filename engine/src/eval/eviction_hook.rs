//! EvictionHook: StepHook implementation for budget-based KV cache eviction.
//!
//! Encapsulates the eviction logic previously embedded in `run_eval_ll` (generate.rs).
//! Supports both H2O (score-based) and Sliding (position-based) eviction policies,
//! and collects QCF/CAOTE metrics at each eviction event.

use super::hook::{CacheSnapshot, MetricsSummary, PostStepResult, StepHook};
use super::qcf_helpers::{aggregate_eviction_metrics, metric_to_json};
use crate::core::attention_scores::AttentionScoreAccumulator;
use crate::core::cache_manager::CacheManager;
use crate::core::kv_cache::{KVCache, max_cache_pos};
use crate::core::qcf::QcfConfig;

/// KV cache snapshot for save/restore between multi-token choice scoring.
///
/// Stores raw byte copies of K and V buffers for each layer, along with
/// their `current_pos` counters. Supports both CPU and GPU (OpenCL) buffers.
pub struct KVCacheSnapshot {
    /// Per-layer raw bytes: K buffer followed immediately by V buffer.
    data: Vec<Vec<u8>>,
    /// Backend reference for GPU read/write operations.
    backend: std::sync::Arc<dyn crate::core::backend::Backend>,
    /// Per-layer `current_pos` values.
    positions: Vec<usize>,
}

impl CacheSnapshot<KVCache> for KVCacheSnapshot {
    fn restore_to(&self, caches: &mut [KVCache]) {
        for (i, cache) in caches.iter_mut().enumerate() {
            let k_size = cache.k_buffer.buffer().size();
            let v_size = self.data[i].len() - k_size;
            let k_ptr = cache.k_buffer.buffer().as_mut_ptr();
            if !k_ptr.is_null() {
                // CPU path: direct memcpy
                unsafe {
                    std::ptr::copy_nonoverlapping(self.data[i].as_ptr(), k_ptr, k_size);
                    std::ptr::copy_nonoverlapping(
                        self.data[i].as_ptr().add(k_size),
                        cache.v_buffer.buffer().as_mut_ptr(),
                        v_size,
                    );
                }
            } else {
                // GPU path: write via OpenCL
                let _ = self
                    .backend
                    .write_buffer(&mut cache.k_buffer, &self.data[i][..k_size]);
                let _ = self
                    .backend
                    .write_buffer(&mut cache.v_buffer, &self.data[i][k_size..]);
            }
            cache.current_pos = self.positions[i];
        }
    }
}

/// StepHook for budget-based eviction (eviction eval-ll mode).
///
/// After each decode step, checks whether `kv_caches[0].current_pos > effective_budget`.
/// When over budget:
/// - H2O / score-based: calls `force_evict_with_scores` with identified evicted tokens,
///   computes `eviction_attn` (and optionally `eviction_caote`) QCF metrics.
/// - Sliding / position-based: calls `force_evict`, computes `sliding_attn`
///   (and optionally `sliding_caote`) QCF metrics.
///
/// After eviction, the score accumulator is reset.
pub struct EvictionHook {
    /// KV cache manager (wraps the eviction policy).
    pub cache_manager: CacheManager,
    /// Attention score accumulator for H2O scoring (Some iff score-based).
    pub score_accumulator: Option<AttentionScoreAccumulator>,
    /// QCF metric collection config.
    pub qcf_config: QcfConfig,
    /// Maximum KV cache tokens before eviction triggers.
    pub effective_budget: usize,
    /// Number of prefix tokens protected from eviction.
    pub protected_prefix: usize,
    /// Whether to use H2O/D2O score-based eviction (vs. positional sliding).
    pub score_based_eviction: bool,
    /// H2O keep ratio (fraction of non-prefix tokens kept as heavy hitters).
    pub h2o_keep_ratio: f32,
    /// KV cache dtype string for QCF gating (only "f32" collects QCF).
    pub kv_type: String,
    /// Backend reference for GPU buffer read/write in snapshot/restore.
    pub backend: std::sync::Arc<dyn crate::core::backend::Backend>,

    // -- Statistics (reset per question) --
    /// Number of eviction events this question.
    eviction_count: usize,
    /// Total tokens evicted this question.
    evicted_total: usize,
}

impl EvictionHook {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cache_manager: CacheManager,
        score_accumulator: Option<AttentionScoreAccumulator>,
        qcf_config: QcfConfig,
        effective_budget: usize,
        protected_prefix: usize,
        score_based_eviction: bool,
        h2o_keep_ratio: f32,
        kv_type: String,
        backend: std::sync::Arc<dyn crate::core::backend::Backend>,
    ) -> Self {
        Self {
            cache_manager,
            score_accumulator,
            qcf_config,
            effective_budget,
            protected_prefix,
            score_based_eviction,
            h2o_keep_ratio,
            kv_type,
            backend,
            eviction_count: 0,
            evicted_total: 0,
        }
    }
}

impl StepHook<KVCache> for EvictionHook {
    fn post_decode_step(
        &mut self,
        caches: &mut [KVCache],
        step: usize,
        qcf_metrics: &mut Vec<serde_json::Value>,
    ) -> PostStepResult {
        if caches.is_empty() || max_cache_pos(caches) <= self.effective_budget {
            return PostStepResult::default();
        }

        let before_len = max_cache_pos(caches);
        let ratio = self.effective_budget as f32 / before_len as f32;

        // For GPU backends (as_ptr() == null), read V buffer back to CPU so that
        // QCF/OPR metrics can be computed. Falls back to None on readback failure.
        let can_compute_qcf = self.kv_type == "f32";
        let v_cpu_data: Option<Vec<f32>> = if can_compute_qcf
            && !caches.is_empty()
            && caches[0].v_buffer.buffer().as_ptr().is_null()
        {
            let v_elems = caches[0].v_buffer.buffer().size() / 4;
            let mut v_buf = vec![0.0f32; v_elems];
            let byte_slice = unsafe {
                std::slice::from_raw_parts_mut(v_buf.as_mut_ptr() as *mut u8, v_elems * 4)
            };
            match self.backend.read_buffer(&caches[0].v_buffer, byte_slice) {
                Ok(()) => Some(v_buf),
                Err(_) => None,
            }
        } else {
            None
        };
        // Effective QCF: either CPU-direct or GPU-readback succeeded
        let can_compute_qcf = can_compute_qcf
            && !caches.is_empty()
            && (v_cpu_data.is_some() || !caches[0].v_buffer.buffer().as_ptr().is_null());

        let result = if self.score_based_eviction {
            let active = self
                .score_accumulator
                .as_ref()
                .is_some_and(|acc| acc.is_active());

            if active {
                let scores = self
                    .score_accumulator
                    .as_ref()
                    .unwrap()
                    .importance_scores()
                    .to_vec();
                let target_len = ((before_len as f32) * ratio) as usize;
                let evicted = crate::core::qcf::identify_evicted_h2o(
                    &scores,
                    self.protected_prefix,
                    self.h2o_keep_ratio,
                    before_len,
                    target_len,
                );

                if !evicted.is_empty() && can_compute_qcf && !caches.is_empty() {
                    if self.qcf_config.mode.has_attn() {
                        let metric = crate::core::qcf::compute_eviction_qcf_attn(
                            &evicted,
                            &scores,
                            &caches[0],
                            &self.qcf_config,
                            v_cpu_data.as_deref(),
                        );
                        qcf_metrics.push(metric_to_json(&metric, step, before_len));
                    }
                    if self.qcf_config.mode.has_caote()
                        && let Some(head_attn) = self
                            .score_accumulator
                            .as_ref()
                            .and_then(|acc| acc.last_step_head_attn())
                    {
                        let positions: Vec<usize> = evicted.iter().map(|(pos, _)| *pos).collect();
                        let metric = crate::core::qcf::compute_eviction_qcf_caote(
                            &positions,
                            head_attn,
                            &caches[0],
                            &self.qcf_config,
                            v_cpu_data.as_deref(),
                        );
                        qcf_metrics.push(metric_to_json(&metric, step, before_len));
                    }
                }

                self.cache_manager
                    .force_evict_with_scores(caches, ratio, &scores)
            } else {
                self.cache_manager.force_evict(caches, ratio)
            }
        } else {
            // Sliding / position-based eviction
            let r = self.cache_manager.force_evict(caches, ratio);
            if let Ok(ref evict_result) = r
                && evict_result.evicted
                && can_compute_qcf
                && !caches.is_empty()
            {
                if self.qcf_config.mode.has_attn() {
                    let positions = crate::core::qcf::identify_evicted_sliding(
                        self.protected_prefix,
                        evict_result.tokens_removed,
                        before_len,
                    );
                    let metric = crate::core::qcf::compute_sliding_qcf_attn(
                        &positions,
                        &caches[0],
                        before_len,
                        &self.qcf_config,
                        v_cpu_data.as_deref(),
                    );
                    let mut entry = metric_to_json(&metric, step, before_len);
                    entry["cache_pos_after"] = serde_json::json!(evict_result.new_pos);
                    qcf_metrics.push(entry);
                }
                if self.qcf_config.mode.has_caote()
                    && let Some(head_attn) = self
                        .score_accumulator
                        .as_ref()
                        .and_then(|acc| acc.last_step_head_attn())
                {
                    let positions = crate::core::qcf::identify_evicted_sliding(
                        self.protected_prefix,
                        evict_result.tokens_removed,
                        before_len,
                    );
                    let metric = crate::core::qcf::compute_sliding_qcf_caote(
                        &positions,
                        head_attn,
                        &caches[0],
                        &self.qcf_config,
                        v_cpu_data.as_deref(),
                    );
                    let mut entry = metric_to_json(&metric, step, before_len);
                    entry["cache_pos_after"] = serde_json::json!(evict_result.new_pos);
                    qcf_metrics.push(entry);
                }
            }
            r
        };

        match result {
            Ok(evict_result) if evict_result.evicted => {
                self.eviction_count += 1;
                self.evicted_total += evict_result.tokens_removed;
                if let Some(acc) = self.score_accumulator.as_mut() {
                    acc.reset();
                }
                PostStepResult {
                    evicted: true,
                    tokens_affected: evict_result.tokens_removed,
                    new_start_pos: Some(evict_result.new_pos),
                }
            }
            _ => PostStepResult::default(),
        }
    }

    fn post_prefill(&mut self, _caches: &mut [KVCache], _qcf_metrics: &mut Vec<serde_json::Value>) {
        // Prefill eviction is handled by the caller (budget-check before entering
        // the decode loop). Nothing to do here for the generic hook interface.
    }

    fn reset_caches(&mut self, caches: &mut [KVCache]) {
        for cache in caches.iter_mut() {
            cache.current_pos = 0;
        }
        if let Some(acc) = self.score_accumulator.as_mut() {
            acc.reset();
        }
        self.eviction_count = 0;
        self.evicted_total = 0;
    }

    fn snapshot(&self, caches: &[KVCache]) -> Box<dyn CacheSnapshot<KVCache>> {
        let mut data = Vec::with_capacity(caches.len());
        let mut positions = Vec::with_capacity(caches.len());
        for cache in caches {
            let k_size = cache.k_buffer.buffer().size();
            let v_size = cache.v_buffer.buffer().size();
            let mut buf = vec![0u8; k_size + v_size];
            let k_ptr = cache.k_buffer.buffer().as_ptr();
            if !k_ptr.is_null() {
                // CPU path: direct memcpy
                unsafe {
                    std::ptr::copy_nonoverlapping(k_ptr, buf.as_mut_ptr(), k_size);
                    std::ptr::copy_nonoverlapping(
                        cache.v_buffer.buffer().as_ptr(),
                        buf.as_mut_ptr().add(k_size),
                        v_size,
                    );
                }
            } else {
                // GPU path: read via OpenCL
                let _ = self
                    .backend
                    .read_buffer(&cache.k_buffer, &mut buf[..k_size]);
                let _ = self
                    .backend
                    .read_buffer(&cache.v_buffer, &mut buf[k_size..]);
            }
            data.push(buf);
            positions.push(cache.current_pos);
        }
        Box::new(KVCacheSnapshot {
            data,
            positions,
            backend: self.backend.clone(),
        })
    }

    fn set_effective_budget(&mut self, budget: usize) {
        self.effective_budget = budget;
    }

    fn score_accumulator(&mut self) -> Option<&mut AttentionScoreAccumulator> {
        self.score_accumulator.as_mut()
    }

    fn extra_question_fields(&self, _caches: &[KVCache]) -> serde_json::Value {
        serde_json::json!({
            "effective_budget": self.effective_budget,
            "eviction_count": self.eviction_count,
            "evicted_tokens": self.evicted_total,
        })
    }

    fn extra_config_fields(&self) -> serde_json::Value {
        serde_json::json!({
            "effective_budget": self.effective_budget,
            "protected_prefix": self.protected_prefix,
            "score_based_eviction": self.score_based_eviction,
            "h2o_keep_ratio": self.h2o_keep_ratio,
            "kv_type": self.kv_type,
        })
    }

    fn aggregate_metrics(&self, qcf_metrics: &[serde_json::Value]) -> MetricsSummary {
        aggregate_eviction_metrics(qcf_metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cache_manager::CacheManager;
    use crate::core::eviction::no_eviction::NoEvictionPolicy;
    use crate::core::qcf::{QcfConfig, QcfMode};
    use crate::core::sys_monitor::{MemoryStats, SystemMonitor};
    use anyhow::Result as AResult;

    struct AlwaysOkMonitor;
    impl SystemMonitor for AlwaysOkMonitor {
        fn mem_stats(&self) -> AResult<MemoryStats> {
            Ok(MemoryStats {
                total: usize::MAX,
                available: usize::MAX,
                free: usize::MAX,
            })
        }
    }

    fn make_hook(budget: usize, score_based: bool) -> EvictionHook {
        let policy = Box::new(NoEvictionPolicy::new());
        let monitor = Box::new(AlwaysOkMonitor);
        let manager = CacheManager::new(policy, monitor, 0, 1.0);
        let mut config = QcfConfig::default();
        config.mode = QcfMode::Attn;
        EvictionHook::new(
            manager,
            None,
            config,
            budget,
            0,
            score_based,
            0.5,
            "f32".to_string(),
            std::sync::Arc::new(crate::backend::cpu::CpuBackend::new()),
        )
    }

    #[test]
    fn test_extra_question_fields_initial() {
        let hook = make_hook(512, false);
        let fields = hook.extra_question_fields(&[]);
        assert_eq!(fields["effective_budget"], 512);
        assert_eq!(fields["eviction_count"], 0);
        assert_eq!(fields["evicted_tokens"], 0);
    }

    #[test]
    fn test_extra_config_fields() {
        let hook = make_hook(256, true);
        let fields = hook.extra_config_fields();
        assert_eq!(fields["effective_budget"], 256);
        assert_eq!(fields["score_based_eviction"], true);
        assert_eq!(fields["kv_type"], "f32");
    }

    #[test]
    fn test_aggregate_metrics_delegates_to_eviction() {
        let hook = make_hook(512, false);
        let metrics = vec![
            serde_json::json!({"action": "sliding_attn", "raw_value": 0.3, "normalized_value": 0.4}),
            serde_json::json!({"action": "sliding_caote", "raw_value": 0.1, "normalized_value": 0.1}),
        ];
        let summary = hook.aggregate_metrics(&metrics);
        assert!((summary.qcf_attn_total - 0.3).abs() < 1e-10);
        assert!((summary.qcf_caote_total - 0.1).abs() < 1e-10);
        assert_eq!(summary.opr_eviction, Some(0.1));
        assert_eq!(summary.opr_eviction_events, 1);
    }

    #[test]
    fn test_post_decode_step_no_eviction_under_budget() {
        // post_decode_step should return default (no eviction) when under budget.
        // We use a hook with an impossibly large budget.
        let hook = make_hook(usize::MAX, false);
        // With no real caches, it should short-circuit on the is_empty check.
        let mut metrics = vec![];
        let result = {
            let mut h = hook;
            h.post_decode_step(&mut [], 0, &mut metrics)
        };
        assert!(!result.evicted);
        assert_eq!(result.tokens_affected, 0);
        assert!(result.new_start_pos.is_none());
        assert!(metrics.is_empty());
    }

    #[test]
    fn test_snapshot_empty() {
        let hook = make_hook(512, false);
        let snapshot = hook.snapshot(&[]);
        // Restoring an empty snapshot on empty caches should not panic.
        snapshot.restore_to(&mut []);
    }
}
