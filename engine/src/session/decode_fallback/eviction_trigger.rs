//! Phase 4-4-2.3c: eviction trigger 추출 — `bin/generate.rs` L2541~L2685 (~146 LOC).
//!
//! 목적: G3 (LOC 감소) only — main() 가독성 + 후속 sub-sprint 진입 비용 절감.
//! Trait 추상화는 본 sprint scope 외.
//!
//! 본 모듈은 decode loop 내 auto-eviction 블록을 담당한다:
//! - GPU score sync (opencl feature-gated)
//! - pre-eviction 스냅샷 캡처
//! - force_evict / maybe_evict 분기
//! - profiler EvictionEvent 기록
//! - position_birth_step compact
//! - score_accumulator reset (CPU + GPU)
//!
//! G3-only 정책상 ctx 필드는 의도된 God Ctx.

use std::sync::Arc;

use crate::backend::Backend;
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::pressure::cache_manager::CacheManager;
use crate::pressure::kv_cache::KVCache;
use crate::profile::{self, EvictionEvent, PartitionInfo};
use crate::session::cli::Args;

pub struct AutoEvictionCtx<'a> {
    pub args: &'a Args,
    pub cache_manager: &'a CacheManager,
    pub kv_caches: &'a mut Vec<KVCache>,
    pub auto_eviction: bool,
    pub score_based_eviction: bool,
    pub score_accumulator: &'a mut Option<AttentionScoreAccumulator>,
    pub d2o_layer_ratios: &'a Option<Vec<(f32, f32)>>,
    pub backend: &'a Arc<dyn Backend>,
    pub profiler: &'a mut Option<crate::profile::InferenceProfiler>,
    pub position_birth_step: &'a mut Vec<usize>,
    pub actual_protected_prefix: usize,
    pub decode_token_index: usize,
}

pub fn run_auto_eviction(ctx: AutoEvictionCtx<'_>) -> anyhow::Result<()> {
    let AutoEvictionCtx {
        args,
        cache_manager,
        kv_caches,
        auto_eviction,
        score_based_eviction,
        score_accumulator,
        d2o_layer_ratios,
        backend,
        profiler,
        position_birth_step,
        actual_protected_prefix,
        decode_token_index,
    } = ctx;

    // Auto-eviction after forward pass (non-experiment mode)
    if auto_eviction {
        let before_len = kv_caches[0].current_pos;
        let capacity = kv_caches[0].capacity();

        // GPU score sync: transfer GPU-accumulated scores to CPU accumulator
        // before any score-based eviction decision. Only syncs when:
        // 1. GPU score acc is active AND
        // 2. Eviction is imminent (score-based at 90% capacity) OR non-score-based with acc
        #[cfg(feature = "opencl")]
        if (score_based_eviction && before_len >= capacity * 9 / 10
            || score_accumulator.as_ref().is_some_and(|a| a.is_active()))
            && let Some(ocl_be) = backend
                .as_any()
                .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            && let Some(gpu_acc) = ocl_be.gpu_score_acc()
            && gpu_acc.is_active()
        {
            let (flat, head) = gpu_acc.sync_to_cpu(ocl_be.queue.as_core())?;
            if let Some(acc) = score_accumulator.as_mut() {
                acc.import_gpu_scores(&flat, &head);
            }
        }

        // Capture pre-eviction scores for profiling (before eviction mutates state)
        let pre_eviction_scores: Vec<f32> =
            if profiler.is_some() && score_based_eviction && before_len >= capacity * 9 / 10 {
                score_accumulator
                    .as_ref()
                    .filter(|acc| acc.is_active())
                    .map(|acc| {
                        acc.importance_scores()[..before_len.min(acc.importance_scores().len())]
                            .to_vec()
                    })
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

        let result = if score_based_eviction && before_len >= capacity * 9 / 10 {
            // Score-based policies: force evict when cache >= 90% full
            if let Some(acc) = score_accumulator.as_ref() {
                if acc.is_active() {
                    // D2O layer-level allocation: use per-layer budgets if available
                    if let Some(ratios) = d2o_layer_ratios.as_ref() {
                        cache_manager.force_evict_with_scores_and_budgets(
                            kv_caches,
                            args.eviction_target_ratio(),
                            acc.importance_scores(),
                            ratios,
                        )?
                    } else {
                        cache_manager.force_evict_with_scores(
                            kv_caches,
                            args.eviction_target_ratio(),
                            acc.importance_scores(),
                        )?
                    }
                } else {
                    cache_manager.force_evict(kv_caches, args.eviction_target_ratio())?
                }
            } else {
                cache_manager.force_evict(kv_caches, args.eviction_target_ratio())?
            }
        } else if let Some(acc) = score_accumulator.as_ref() {
            if acc.is_active() {
                cache_manager.maybe_evict_with_scores(kv_caches, acc.importance_scores())?
            } else {
                cache_manager.maybe_evict(kv_caches)?
            }
        } else {
            cache_manager.maybe_evict(kv_caches)?
        };
        if result.evicted {
            // Compute evicted indices from pre-eviction state
            let target_len = ((before_len as f32) * args.eviction_target_ratio()) as usize;
            let evicted_indices = if !pre_eviction_scores.is_empty() {
                profile::compute_h2o_evicted_indices(
                    before_len,
                    target_len,
                    actual_protected_prefix,
                    args.h2o_keep_ratio(),
                    &pre_eviction_scores,
                )
            } else {
                Vec::new()
            };

            if let Some(p) = profiler.as_mut() {
                // Record token deaths before the EvictionEvent
                if !evicted_indices.is_empty() {
                    p.scores.record_token_deaths(
                        decode_token_index,
                        &evicted_indices,
                        position_birth_step,
                        &pre_eviction_scores,
                    );
                }
                p.on_eviction(EvictionEvent {
                    step: decode_token_index,
                    policy: args.eviction_policy().to_string(),
                    before_len,
                    after_len: result.new_pos,
                    evicted_count: result.tokens_removed,
                    partition: PartitionInfo {
                        prefix_end: actual_protected_prefix,
                        hh_count: 0,
                        recent_start: result.new_pos,
                    },
                    evicted_indices: evicted_indices.clone(),
                    pre_eviction_scores,
                });
            }

            // Update position_birth_step mapping after eviction (compact)
            if !position_birth_step.is_empty() {
                let evicted_set: std::collections::HashSet<usize> =
                    evicted_indices.iter().copied().collect();
                let mut kept = Vec::new();
                for (pos, &birth) in position_birth_step.iter().enumerate() {
                    if pos < before_len && !evicted_set.contains(&pos) {
                        kept.push(birth);
                    }
                }
                *position_birth_step = kept;
            }

            if let Some(acc) = score_accumulator.as_mut() {
                acc.reset();
            }
            // Reset GPU score accumulator after eviction
            #[cfg(feature = "opencl")]
            if let Some(ocl_be) = backend
                .as_any()
                .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
                && let Some(gpu_acc) = ocl_be.gpu_score_acc_mut()
                && gpu_acc.is_active()
            {
                gpu_acc.reset(ocl_be.queue.as_core())?;
            }
        }
    }

    Ok(())
}
