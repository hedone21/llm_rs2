//! EvictionHook: StepHook implementation for budget-based KV cache eviction.
//!
//! Encapsulates the eviction logic previously embedded in `run_eval_ll` (generate.rs).
//! Supports both H2O (score-based) and Sliding (position-based) eviction policies,
//! and collects QCF/CAOTE metrics at each eviction event.

use super::hook::{CacheSnapshot, MetricsSummary, PostStepResult, StepHook};
use crate::core::attention_scores::AttentionScoreAccumulator;
use crate::core::cache_manager::CacheManager;
use crate::core::kv_cache::{KVCache, max_cache_pos};
use crate::core::qcf::QcfConfig;

/// QCF result from the single post-prefill eviction event (eval-ll mode).
#[derive(Debug, Clone)]
pub struct EvictionQcfResult {
    pub tokens_evicted: usize,
    pub eviction_ratio: f32,
    pub qcf_attn_raw: f32,
    pub qcf_attn_norm: f32,
    pub qcf_caote: f32,
}

/// KV cache snapshot for save/restore between multi-token choice scoring.
///
/// Stores raw byte copies of K and V buffers for each layer, along with
/// their `current_pos` counters. Supports both CPU and GPU (OpenCL) buffers.
pub struct KVCacheSnapshot {
    /// Per-layer raw bytes: K buffer followed immediately by V buffer.
    data: Vec<Vec<u8>>,
    /// Per-layer K buffer size at snapshot time (used for K/V split in restore).
    k_sizes: Vec<usize>,
    /// Backend reference for GPU read/write operations.
    backend: std::sync::Arc<dyn crate::core::backend::Backend>,
    /// Per-layer `current_pos` values.
    positions: Vec<usize>,
    /// Per-layer capacity at snapshot time.
    capacities: Vec<usize>,
}

impl CacheSnapshot<KVCache> for KVCacheSnapshot {
    fn restore_to(&self, caches: &mut [KVCache]) {
        for (i, cache) in caches.iter_mut().enumerate() {
            // Use snapshot-time buffer sizes, not current sizes.
            // Cache may have grown/shrunk between snapshot and restore.
            let snap_k_size = self.k_sizes[i];
            let snap_v_size = self.data[i].len() - snap_k_size;

            // If cache grew since snapshot, the current buffer is larger — write only
            // snapshot-sized data (snap_k_size bytes). Extra bytes are harmless garbage.
            // If cache shrunk since snapshot (shouldn't happen in eval-ll flow), skip
            // write to avoid buffer overrun — the cache will be reset next question anyway.

            let cur_k_size = cache.k_buffer.buffer().size();
            let cur_v_size = cache.v_buffer.buffer().size();
            let k_ptr = cache.k_buffer.buffer().as_mut_ptr();

            if !k_ptr.is_null() {
                // CPU path: direct memcpy (copy min of snapshot and current sizes)
                let k_copy = snap_k_size.min(cur_k_size);
                let v_copy = snap_v_size.min(cur_v_size);
                unsafe {
                    std::ptr::copy_nonoverlapping(self.data[i].as_ptr(), k_ptr, k_copy);
                    std::ptr::copy_nonoverlapping(
                        self.data[i].as_ptr().add(snap_k_size),
                        cache.v_buffer.buffer().as_mut_ptr(),
                        v_copy,
                    );
                }
            } else {
                // GPU path: write_buffer requires exact size match.
                // If cache grew since snapshot, pad with zeros to match current buffer size.
                if snap_k_size == cur_k_size {
                    let _ = self.backend.write_buffer(
                        &mut cache.k_buffer,
                        &self.data[i][..snap_k_size],
                    );
                } else {
                    let mut padded = vec![0u8; cur_k_size];
                    let copy_len = snap_k_size.min(cur_k_size);
                    padded[..copy_len].copy_from_slice(&self.data[i][..copy_len]);
                    let _ = self.backend.write_buffer(&mut cache.k_buffer, &padded);
                }
                if snap_v_size == cur_v_size {
                    let _ = self.backend.write_buffer(
                        &mut cache.v_buffer,
                        &self.data[i][snap_k_size..],
                    );
                } else {
                    let mut padded = vec![0u8; cur_v_size];
                    let copy_len = snap_v_size.min(cur_v_size);
                    padded[..copy_len]
                        .copy_from_slice(&self.data[i][snap_k_size..snap_k_size + copy_len]);
                    let _ = self.backend.write_buffer(&mut cache.v_buffer, &padded);
                }
            }
            cache.current_pos = self.positions[i];
            cache.high_water_pos = self.positions[i];
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
    /// Whether to use D2O merge compensation (vs. plain H2O eviction) for QCF-CAOTE.
    pub is_d2o: bool,
    /// KV cache dtype string for QCF gating (only "f32" collects QCF).
    pub kv_type: String,
    /// Backend reference for GPU buffer read/write in snapshot/restore.
    pub backend: std::sync::Arc<dyn crate::core::backend::Backend>,

    // -- Statistics (reset per question) --
    /// Number of eviction events this question.
    eviction_count: usize,
    /// Total tokens evicted this question.
    evicted_total: usize,
    /// QCF result from the single post-prefill eviction event (eval-ll mode).
    eviction_qcf: Option<EvictionQcfResult>,
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
        is_d2o: bool,
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
            is_d2o,
            kv_type,
            backend,
            eviction_count: 0,
            evicted_total: 0,
            eviction_qcf: None,
        }
    }
}

impl StepHook<KVCache> for EvictionHook {
    fn post_decode_step(
        &mut self,
        caches: &mut [KVCache],
        _step: usize,
        _qcf_metrics: &mut Vec<serde_json::Value>,
    ) -> PostStepResult {
        if caches.is_empty() || max_cache_pos(caches) <= self.effective_budget {
            return PostStepResult::default();
        }

        let before_len = max_cache_pos(caches);
        let ratio = self.effective_budget as f32 / before_len as f32;

        // Perform eviction (QCF collection removed — eval-ll uses single post_prefill event)
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
                self.cache_manager
                    .force_evict_with_scores(caches, ratio, &scores)
            } else {
                self.cache_manager.force_evict(caches, ratio)
            }
        } else {
            self.cache_manager.force_evict(caches, ratio)
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

    fn post_prefill(&mut self, caches: &mut [KVCache], _qcf_metrics: &mut Vec<serde_json::Value>) {
        // After full batch prefill, evict if cache exceeds budget.
        // This replaces the old chunked-prefill approach that decoded overflow
        // tokens one-by-one (causing 2-3.3x slowdown).
        if caches.is_empty() || max_cache_pos(caches) <= self.effective_budget {
            return;
        }

        let before_len = max_cache_pos(caches);
        let ratio = self.effective_budget as f32 / before_len as f32;
        let eviction_ratio = 1.0 - ratio;

        // V buffer readback for QCF-CAOTE (GPU backends)
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
        // GPU score sync before QCF computation (eval-ll path).
        // On GPU backends, forward_into() accumulates scores entirely on the device.
        // The CPU accumulator's importance and last_layer_head_attn are empty.
        // We sync both: (1) cumulative importance via import_gpu_scores, and
        // (2) head importance as proxy for last_layer_head_attn (the GPU path
        // doesn't have raw per-step attention weights, but cumulative head
        // importance is proportional and sufficient for QCF computation).
        #[cfg(feature = "opencl")]
        if let Some(ref mut acc) = self.score_accumulator
            && acc.is_active()
            && let Some(ocl_be) = self
                .backend
                .as_any()
                .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            && let Some(gpu_acc) = ocl_be.gpu_score_acc()
            && gpu_acc.is_active()
            && let Ok((flat, head)) = gpu_acc.sync_to_cpu(ocl_be.queue.as_core())
        {
            acc.import_gpu_scores(&flat, &head);
        }

        let can_compute_qcf = can_compute_qcf
            && !caches.is_empty()
            && (v_cpu_data.is_some() || !caches[0].v_buffer.buffer().as_ptr().is_null());

        // Collect QCF metrics before eviction
        let (_evicted_positions_for_qcf, qcf_attn_raw, qcf_attn_norm, qcf_caote) = if self
            .score_based_eviction
        {
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
                let positions: Vec<usize> = evicted.iter().map(|(pos, _)| *pos).collect::<Vec<_>>();

                // QCF-ATTN v2: use last_step_head_attn if available
                let (attn_raw, attn_norm) = self
                    .score_accumulator
                    .as_ref()
                    .and_then(|acc| acc.last_step_head_attn())
                    .map(|head_attn| {
                        let n_kv_heads = if !caches.is_empty() {
                            caches[0].kv_heads()
                        } else {
                            1
                        };
                        let max_seq_len = head_attn.len() / n_kv_heads.max(1);
                        let m = crate::core::qcf::compute_qcf_attn_v2(
                            head_attn,
                            &positions,
                            n_kv_heads,
                            max_seq_len,
                            eviction_ratio,
                        );
                        (m.raw_value, m.normalized_value)
                    })
                    .unwrap_or((0.0, 0.0));

                // QCF-CAOTE: D2O uses unified QCF with merge compensation;
                // H2O uses the legacy eviction-only CAOTE.
                let caote = if self.is_d2o && can_compute_qcf && !positions.is_empty() {
                    // D2O path: compute_unified_qcf with MergeD2o action
                    use crate::core::buffer::DType;
                    use crate::core::qcf::{
                        AggregationMode, QcfActionType, UnifiedQcfParams, VDataSource,
                        compute_unified_qcf,
                    };
                    let cache = &caches[0];
                    let v_slice_f32: &[f32];
                    let v_slice_u16: &[u16];
                    let v_source = if let Some(ref cpu_data) = v_cpu_data {
                        v_slice_f32 = cpu_data;
                        VDataSource::F32(v_slice_f32)
                    } else {
                        match cache.v_buffer.dtype() {
                            DType::F16 => {
                                v_slice_u16 = cache.v_buffer.as_slice::<u16>();
                                VDataSource::F16(v_slice_u16)
                            }
                            _ => {
                                v_slice_f32 = cache.v_buffer.as_slice::<f32>();
                                VDataSource::F32(v_slice_f32)
                            }
                        }
                    };
                    let head_attn_opt = self
                        .score_accumulator
                        .as_ref()
                        .and_then(|acc| acc.last_step_head_attn());
                    let params = UnifiedQcfParams {
                        action: QcfActionType::MergeD2o {
                            target_len,
                            keep_ratio: self.h2o_keep_ratio,
                            protected_prefix: self.protected_prefix,
                        },
                        v_source,
                        attention_scores: &scores,
                        head_attn: head_attn_opt,
                        n_kv_heads: cache.kv_heads(),
                        head_dim: cache.head_dim(),
                        current_pos: before_len,
                        capacity: cache.capacity(),
                        layout: cache.layout(),
                        aggregation: AggregationMode::Mean,
                    };
                    let (qcf, _) = compute_unified_qcf(&params);
                    qcf
                } else if can_compute_qcf && !positions.is_empty() {
                    // H2O path: legacy eviction-only CAOTE
                    if let Some(head_attn) = self
                        .score_accumulator
                        .as_ref()
                        .and_then(|acc| acc.last_step_head_attn())
                    {
                        let metric = crate::core::qcf::compute_eviction_qcf_caote(
                            &positions,
                            head_attn,
                            &caches[0],
                            &self.qcf_config,
                            v_cpu_data.as_deref(),
                        );
                        metric.raw_value
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                (positions, attn_raw, attn_norm, caote)
            } else {
                (Vec::new(), 0.0, 0.0, 0.0)
            }
        } else {
            // Sliding: compute positions that will be evicted
            let target_len = ((before_len as f32) * ratio) as usize;
            let prune_count = before_len.saturating_sub(target_len);
            let positions = crate::core::qcf::identify_evicted_sliding(
                self.protected_prefix,
                prune_count,
                before_len,
            );

            // QCF-ATTN v2: use last_step_head_attn if available
            let (attn_raw, attn_norm) = self
                .score_accumulator
                .as_ref()
                .and_then(|acc| acc.last_step_head_attn())
                .map(|head_attn| {
                    let n_kv_heads = if !caches.is_empty() {
                        caches[0].kv_heads()
                    } else {
                        1
                    };
                    let max_seq_len = head_attn.len() / n_kv_heads.max(1);
                    let m = crate::core::qcf::compute_qcf_attn_v2(
                        head_attn,
                        &positions,
                        n_kv_heads,
                        max_seq_len,
                        eviction_ratio,
                    );
                    (m.raw_value, m.normalized_value)
                })
                .unwrap_or((0.0, 0.0));

            // QCF-CAOTE
            let caote = if can_compute_qcf && !positions.is_empty() {
                if let Some(head_attn) = self
                    .score_accumulator
                    .as_ref()
                    .and_then(|acc| acc.last_step_head_attn())
                {
                    let metric = crate::core::qcf::compute_sliding_qcf_caote(
                        &positions,
                        head_attn,
                        &caches[0],
                        &self.qcf_config,
                        v_cpu_data.as_deref(),
                    );
                    metric.raw_value
                } else {
                    0.0
                }
            } else {
                0.0
            };

            (positions, attn_raw, attn_norm, caote)
        };

        // Perform eviction
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
                self.cache_manager
                    .force_evict_with_scores(caches, ratio, &scores)
            } else {
                self.cache_manager.force_evict(caches, ratio)
            }
        } else {
            self.cache_manager.force_evict(caches, ratio)
        };

        if let Ok(evict_result) = result
            && evict_result.evicted
        {
            self.eviction_count += 1;
            self.evicted_total += evict_result.tokens_removed;
            if let Some(acc) = self.score_accumulator.as_mut() {
                acc.reset();
            }

            // Store QCF result for extra_question_fields
            self.eviction_qcf = Some(EvictionQcfResult {
                tokens_evicted: evict_result.tokens_removed,
                eviction_ratio,
                qcf_attn_raw,
                qcf_attn_norm,
                qcf_caote,
            });
        }
    }

    fn reset_caches(&mut self, caches: &mut [KVCache]) {
        for cache in caches.iter_mut() {
            cache.current_pos = 0;
            cache.high_water_pos = 0;
        }
        if let Some(acc) = self.score_accumulator.as_mut() {
            acc.reset();
        }
        self.eviction_count = 0;
        self.evicted_total = 0;
        self.eviction_qcf = None;
    }

    fn snapshot(&self, caches: &[KVCache]) -> Box<dyn CacheSnapshot<KVCache>> {
        let mut data = Vec::with_capacity(caches.len());
        let mut k_sizes = Vec::with_capacity(caches.len());
        let mut positions = Vec::with_capacity(caches.len());
        let mut capacities = Vec::with_capacity(caches.len());
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
            k_sizes.push(k_size);
            positions.push(cache.current_pos);
            capacities.push(cache.capacity());
        }
        Box::new(KVCacheSnapshot {
            data,
            k_sizes,
            positions,
            capacities,
            backend: self.backend.clone(),
        })
    }

    fn set_effective_budget(&mut self, budget: usize) {
        self.effective_budget = budget;
    }

    fn score_accumulator(&mut self) -> Option<&mut AttentionScoreAccumulator> {
        self.score_accumulator.as_mut()
    }

    fn needs_score_probe(&self, caches: &[KVCache]) -> bool {
        // Probe is needed when cache exceeds budget (eviction will happen).
        // The probe step populates score_accumulator for H2O decisions and
        // captures last_step_head_attn for QCF-ATTN measurement.
        !caches.is_empty() && max_cache_pos(caches) > self.effective_budget
    }

    fn extra_question_fields(&self, _caches: &[KVCache]) -> serde_json::Value {
        let mut obj = serde_json::json!({
            "effective_budget": self.effective_budget,
            "eviction_count": self.eviction_count,
            "evicted_tokens": self.evicted_total,
        });
        if let Some(ref qcf) = self.eviction_qcf {
            obj["tokens_evicted"] = serde_json::json!(qcf.tokens_evicted);
            obj["eviction_ratio"] = serde_json::json!(qcf.eviction_ratio);
            obj["qcf_attn_raw"] = serde_json::json!(qcf.qcf_attn_raw);
            obj["qcf_attn_norm"] = serde_json::json!(qcf.qcf_attn_norm);
            obj["qcf_caote"] = serde_json::json!(qcf.qcf_caote);
        }
        obj
    }

    fn extra_config_fields(&self) -> serde_json::Value {
        serde_json::json!({
            "effective_budget": self.effective_budget,
            "protected_prefix": self.protected_prefix,
            "score_based_eviction": self.score_based_eviction,
            "h2o_keep_ratio": self.h2o_keep_ratio,
            "is_d2o": self.is_d2o,
            "kv_type": self.kv_type,
        })
    }

    fn aggregate_metrics(&self, _qcf_metrics: &[serde_json::Value]) -> MetricsSummary {
        // Eviction metrics are now stored in self.eviction_qcf (via extra_question_fields).
        // Return default so the eval_loop's KIVI OPR check (summary.qcf_kivi_opr.is_some())
        // correctly returns None for the eviction path.
        MetricsSummary::default()
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
        make_hook_with_d2o(budget, score_based, false)
    }

    fn make_hook_with_d2o(budget: usize, score_based: bool, is_d2o: bool) -> EvictionHook {
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
            is_d2o,
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
        assert_eq!(fields["is_d2o"], false);
        assert_eq!(fields["kv_type"], "f32");
    }

    #[test]
    fn test_extra_config_fields_d2o() {
        let hook = make_hook_with_d2o(256, true, true);
        let fields = hook.extra_config_fields();
        assert_eq!(fields["is_d2o"], true);
        assert_eq!(fields["score_based_eviction"], true);
    }

    #[test]
    fn test_aggregate_metrics_returns_default() {
        // EvictionHook.aggregate_metrics() now always returns MetricsSummary::default().
        // Eviction QCF data is stored in self.eviction_qcf and emitted via extra_question_fields.
        let hook = make_hook(512, false);
        let metrics = vec![
            serde_json::json!({"action": "sliding_attn", "raw_value": 0.3, "normalized_value": 0.4}),
            serde_json::json!({"action": "sliding_caote", "raw_value": 0.1, "normalized_value": 0.1}),
        ];
        let summary = hook.aggregate_metrics(&metrics);
        // Default: no KIVI OPR (eviction path does not use OPR)
        assert!(summary.qcf_kivi_opr.is_none());
        assert_eq!(summary.qcf_kivi_opr_events, 0);
        // Eviction attn/caote totals are now 0 in aggregate (moved to extra_question_fields)
        assert_eq!(summary.qcf_attn_total, 0.0);
        assert_eq!(summary.qcf_caote_total, 0.0);
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
