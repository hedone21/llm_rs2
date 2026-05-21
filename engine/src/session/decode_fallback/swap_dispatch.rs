//! Phase 4-4-2.3b: swap dispatcher 추출 — `bin/generate.rs` J3+J4+J5 (~352 LOC).
//!
//! 목적: G3 (LOC 감소) only — main() 가독성 + 후속 sub-sprint 진입 비용 절감.
//! Trait 추상화는 본 sprint scope 외.
//!
//! - J3 (run_incremental_dispatch): Layer-Incremental Swap dispatch (ENG-ALG-233).
//!   drain chunk + run_layer_swap + Dynamic-K/Probing-K + plan retire + manager report build.
//! - J4 (retire_intra_forward): LISWAP-4 Intra-forward Swap retire (INV-150).
//! - J5 (retire_phase_aware): LISWAP-5 Phase-aware Swap retire.
//!
//! G3-only 정책상 ctx 21 필드는 의도된 God Ctx.

use std::sync::Arc;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::observability::rss_trace::read_bytes_now;
use crate::models::transformer::TransformerModel;
use crate::models::weights::{
    AsyncSwapDispatcher, DynamicKController, IncrementalSwapPlan, IntraForwardSwapHook,
    PhaseAwareSwapDispatcher, ProbingKController,
};
use crate::qcf::ImportanceTable;
use crate::session::cli::Args;
use crate::session::qcf_runtime::run_layer_swap;

pub struct SwapDispatchCtx<'a> {
    pub model: &'a mut TransformerModel,
    pub args: &'a Args,
    pub decode_token_index: usize,
    pub forward_ms: f64,
    pub backend: &'a Arc<dyn Backend>,
    pub gpu_backend_arc: &'a Option<Arc<dyn Backend>>,
    pub cpu_backend_arc: &'a Arc<dyn Backend>,
    pub is_gpu: bool,
    pub async_swap_dispatcher: &'a Option<AsyncSwapDispatcher>,
    #[cfg(feature = "opencl")]
    pub host_ptr_swap_pool: &'a Option<Arc<crate::backend::opencl::host_ptr_pool::HostPtrPool>>,
    #[cfg(feature = "cuda-embedded")]
    pub layer_swap_pool: &'a Option<Arc<dyn crate::layers::staging_pool::WeightStagingPool>>,
    #[cfg(feature = "cuda-embedded")]
    pub mmap_registration: &'a Option<Arc<crate::memory::cuda::mmap::CudaMmapRegistration>>,
    pub importance_table_for_swap: &'a Option<ImportanceTable>,
    pub dynamic_k_diag: bool,
    pub probing_k_diag: bool,
    // mut clusters
    pub incremental_force_swap_plan: &'a mut Option<IncrementalSwapPlan>,
    pub intra_forward_swap_hook: &'a mut Option<Arc<IntraForwardSwapHook>>,
    pub phase_aware_swap_dispatcher: &'a mut Option<Arc<PhaseAwareSwapDispatcher>>,
    pub dynamic_k_controller: &'a mut Option<DynamicKController>,
    pub probing_k_controller: &'a mut Option<ProbingKController>,
    pub manager_swap_report_pending: &'a mut Option<(f32, usize, std::time::Instant, f32)>,
    pub ready_weight_swap_report: &'a mut Option<llm_shared::WeightSwapReport>,
}

/// J3 — Layer-Incremental Swap dispatch (ENG-ALG-233).
///
/// Drain chunk + run_layer_swap + Dynamic-K/Probing-K calibration/observation
/// + plan retire + manager WeightSwapReport build.
pub fn run_incremental_dispatch(ctx: &mut SwapDispatchCtx<'_>) -> anyhow::Result<()> {
    // ── Layer-Incremental Swap dispatch (ENG-ALG-233) ──────────────────
    // Runs after forward, before sampling. Per-tick: drain up to N layers
    // and call SwapExecutor::execute_on_slots with the chunk.
    // ENG-ALG-234: plan committed with force-swap-ratio + per_tick > 0;
    //   new signals during flight are ignored (plan runs to completion).
    // INV-145: empty chunk is never passed to execute_on_slots.
    if let Some(inc_plan) = ctx.incremental_force_swap_plan.as_mut() {
        // LISWAP-6 Dynamic-K: reactive pause + per-tick override.
        //
        // - Pause: release queue non-empty → skip swap this tick (K
        //   stays unchanged). Calibration tick is exempt because it
        //   has to dispatch K=1 to measure drop cost.
        // - Pre-drain: inject controller's current K into the plan.
        let mut dyn_k_pause = false;
        if let Some(ctrl) = ctx.dynamic_k_controller.as_ref() {
            let pending = ctx.model.release_worker.pending_count();
            if ctrl.is_calibrated() && ctrl.should_pause(pending) {
                dyn_k_pause = true;
                if ctx.dynamic_k_diag {
                    eprintln!(
                        "[DynamicK] pause t={} pending={} k={}",
                        ctx.decode_token_index,
                        pending,
                        ctrl.current_k()
                    );
                }
            } else {
                // Calibration tick forces K=1 (sync measurement);
                // subsequent ticks use the controller's current K.
                let k = if ctrl.is_calibrated() {
                    ctrl.current_k()
                } else {
                    1
                };
                inc_plan.set_per_tick(k);
            }
        } else if let Some(ctrl) = ctx.probing_k_controller.as_ref() {
            let pending = ctx.model.release_worker.pending_count();
            if ctrl.should_pause(pending) {
                dyn_k_pause = true;
                if ctx.probing_k_diag {
                    eprintln!(
                        "[ProbingK] pause t={} pending={} k={}",
                        ctx.decode_token_index,
                        pending,
                        ctrl.current_k()
                    );
                }
            } else {
                inc_plan.set_per_tick(ctrl.current_k());
            }
        }
        let chunk = if dyn_k_pause {
            Vec::new()
        } else {
            inc_plan.drain_chunk()
        };
        if !chunk.is_empty() {
            let t_swap = std::time::Instant::now();
            let io_before = read_bytes_now();
            match run_layer_swap(
                ctx.model,
                &chunk,
                ctx.gpu_backend_arc.as_ref(),
                ctx.cpu_backend_arc,
                ctx.async_swap_dispatcher.as_ref(),
                #[cfg(feature = "opencl")]
                ctx.host_ptr_swap_pool.clone(),
                #[cfg(feature = "cuda-embedded")]
                ctx.layer_swap_pool.clone(),
                #[cfg(feature = "cuda-embedded")]
                ctx.mmap_registration.clone(),
            ) {
                Ok(report) => {
                    let io_after = read_bytes_now();
                    eprintln!(
                        "[IncrementalSwap] tick={} chunk={:?} swapped={} remaining={} latency={:.1}ms read_bytes_delta={}",
                        ctx.decode_token_index,
                        &chunk,
                        report.swapped.len(),
                        inc_plan.remaining_count(),
                        t_swap.elapsed().as_secs_f64() * 1000.0,
                        io_after.saturating_sub(io_before),
                    );
                    if let Some(ref stages) = report.stage_breakdown {
                        eprintln!("[IncrementalSwap] stages: {}", stages.to_log_line());
                    }
                    #[cfg(feature = "opencl")]
                    remap_weights_for_cpu_after_swap(
                        ctx.model,
                        ctx.backend,
                        ctx.is_gpu,
                        ctx.args.resilience_prealloc_switch,
                        "incremental-swap",
                    );

                    // LISWAP-6 Dynamic-K Phase 0 calibration. Runs only on
                    // the first successfully-dispatched chunk. Uses the
                    // dispatch wall (`t_swap.elapsed()`) divided by
                    // `chunk.len()` as `drop_ms_per_layer` — the main-thread
                    // blocking time per layer (mmap_permute + dispatcher
                    // submit). The prior release_worker spin was unreliable
                    // on async path: dispatcher worker chains release enqueue
                    // independently and sub-ms release time collapsed
                    // drop_ms to 0 → safe_k exploded. Main-thread blocking
                    // time is a meaningful budget item that scales with
                    // cold/warm mmap state and chunk size (2026-05-13).
                    if let Some(ctrl) = ctx.dynamic_k_controller.as_mut()
                        && !ctrl.is_calibrated()
                        && !chunk.is_empty()
                    {
                        let blocking_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
                        let drop_ms_per_layer = (blocking_ms / chunk.len() as f64) as f32;
                        ctrl.calibrate(drop_ms_per_layer, ctx.forward_ms as f32);
                        if ctx.dynamic_k_diag {
                            eprintln!(
                                "[DynamicK] calibrated t={} blocking_ms={:.3}/layer fwd_ms={:.2} safe_k={}",
                                ctx.decode_token_index,
                                drop_ms_per_layer,
                                ctx.forward_ms,
                                ctrl.current_k()
                            );
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[IncrementalSwap] swap error on tick={}: {}",
                        ctx.decode_token_index, e
                    );
                }
            }
        }

        // LISWAP-6 Dynamic-K Phase 1+: observe forward wall, shrink K
        // if the forward got tighter than anything seen so far.
        if let Some(ctrl) = ctx.dynamic_k_controller.as_mut()
            && ctrl.is_calibrated()
        {
            let prev_k = ctrl.current_k();
            ctrl.observe_forward(ctx.forward_ms as f32);
            if ctx.dynamic_k_diag && ctrl.current_k() != prev_k {
                eprintln!(
                    "[DynamicK] k_decrease t={} fwd_ms={:.2} new_k={}",
                    ctx.decode_token_index,
                    ctx.forward_ms,
                    ctrl.current_k()
                );
            }
        }

        // Probing-K observation: every decode token feeds the EWMA and
        // counts toward the stability window. release_pending samples
        // *after* the dispatch above — if any spike landed it will
        // decrement K symmetric to ARGUS's monotonic shrink.
        if let Some(ctrl) = ctx.probing_k_controller.as_mut() {
            let prev_k = ctrl.current_k();
            let pending_after = ctx.model.release_worker.pending_count();
            ctrl.observe(ctx.forward_ms as f32, pending_after);
            if ctx.probing_k_diag {
                let arrow = if ctrl.current_k() > prev_k {
                    "↑"
                } else if ctrl.current_k() < prev_k {
                    "↓"
                } else {
                    "·"
                };
                eprintln!(
                    "[ProbingK] t={} fwd_ms={:.2} pending={} k={}->{} {}",
                    ctx.decode_token_index,
                    ctx.forward_ms,
                    pending_after,
                    prev_k,
                    ctrl.current_k(),
                    arrow,
                );
            }
        }
        // ENG-ALG-233: retire plan when all layers have been drained (INV-145).
        if inc_plan.is_done() {
            eprintln!(
                "[IncrementalSwap] plan complete (started_at_token={}, finished_at_token={})",
                inc_plan.started_at_token(),
                ctx.decode_token_index,
            );
            // LISWAP-2: drain async dispatcher to ensure all in-flight commits land
            // before the plan is retired. drain failure is non-fatal — prototype
            // robustness is secondary to measurement.
            if let Some(dispatcher) = ctx.async_swap_dispatcher.as_ref() {
                let drain_t = std::time::Instant::now();
                if let Err(e) = dispatcher.drain(std::time::Duration::from_secs(2)) {
                    eprintln!("[LISWAP-2] drain failed: {e}");
                } else {
                    eprintln!(
                        "[LISWAP-2] dispatcher drained: {:.1}ms",
                        drain_t.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
            *ctx.incremental_force_swap_plan = None;

            // LISWAP-6 manager path: build WeightSwapReport when the plan
            // was committed by dispatch_swap_weights (manager signal).
            // Stored in `ready_weight_swap_report`; sent by executor block
            // later this token tick (executor scope is separate).
            if let Some((ratio, n_planned, plan_start, qcf_estimated)) =
                ctx.manager_swap_report_pending.take()
            {
                use crate::models::weights::compute_qcf_weight_swap;
                let latency_ms = plan_start.elapsed().as_millis() as u64;
                let n_layers = ctx.model.layers.len();
                let actually_swapped_now: Vec<usize> = (0..n_layers)
                    .filter(|&i| ctx.model.layers[i].current_dtype() == DType::Q4_0)
                    .collect();
                let qcf_swap_actual = if actually_swapped_now.is_empty() {
                    qcf_estimated
                } else {
                    compute_qcf_weight_swap(
                        &actually_swapped_now,
                        &ctx.model.quant_noise,
                        ctx.importance_table_for_swap.as_ref(),
                        n_layers,
                    )
                };
                let layers_swapped: Vec<llm_shared::LayerSwapEntry> = actually_swapped_now
                    .iter()
                    .map(|&idx| llm_shared::LayerSwapEntry {
                        layer_idx: idx as u32,
                        from_dtype: llm_shared::DtypeTag::F16,
                        to_dtype: llm_shared::DtypeTag::Q4_0,
                    })
                    .collect();
                eprintln!(
                    "[WeightSwap] manager plan complete: ratio={:.2}, planned={}, \
                     actually_q4={}, qcf_swap={:.4}, latency={}ms",
                    ratio,
                    n_planned,
                    layers_swapped.len(),
                    qcf_swap_actual,
                    latency_ms,
                );
                *ctx.ready_weight_swap_report = Some(llm_shared::WeightSwapReport {
                    layers_swapped,
                    freed_bytes: 0,
                    latency_ms,
                    qcf_swap_actual,
                });
            }
        }
    }
    // ── End Layer-Incremental Swap dispatch ────────────────────────────
    Ok(())
}

/// J4 — LISWAP-4 Intra-forward Swap retire (INV-150).
///
/// After every decode token, check whether the in-flight plan is complete.
/// If so, drain dispatcher, synchronize backend, bump ratio_generation,
/// invalidate noshuffle SOA registry, and retire the hook to None.
pub fn retire_intra_forward(ctx: &mut SwapDispatchCtx<'_>) -> anyhow::Result<()> {
    // ── LISWAP-4 Intra-forward Swap retire (INV-150) ──────────────────
    if let Some(hook) = ctx.intra_forward_swap_hook.clone()
        && hook.plan_is_complete()
    {
        let drain_t = std::time::Instant::now();
        let backend_for_invalidate: Arc<dyn Backend> = ctx
            .gpu_backend_arc
            .as_ref()
            .cloned()
            .unwrap_or_else(|| Arc::clone(ctx.backend));
        let invalidate = move || {
            backend_for_invalidate.invalidate_noshuffle_soa_registry();
        };
        match hook.finalize(
            &ctx.model.ratio_generation,
            invalidate,
            std::time::Duration::from_secs(10),
        ) {
            Ok(()) => {
                eprintln!(
                    "[IntraForwardSwap] plan retired at token={} (drain+sync+bump+invalidate {:.1}ms)",
                    ctx.decode_token_index,
                    drain_t.elapsed().as_secs_f64() * 1000.0,
                );
                #[cfg(feature = "opencl")]
                remap_weights_for_cpu_after_swap(
                    ctx.model,
                    ctx.backend,
                    ctx.is_gpu,
                    ctx.args.resilience_prealloc_switch,
                    "intra-forward-swap",
                );
            }
            Err(e) => {
                eprintln!(
                    "[IntraForwardSwap] finalize failed at token={}: {}",
                    ctx.decode_token_index, e
                );
            }
        }
        *ctx.intra_forward_swap_hook = None; // retire
    }
    // ── End LISWAP-4 retire ────────────────────────────────────────────
    Ok(())
}

/// J5 — LISWAP-5 Phase-aware Swap retire.
///
/// chunk_queue가 비고 in_flight도 None이면 dispatcher 종료. finalize는
/// 마지막 ratio_generation bump + invalidate 수행. PHASE_HOOK은
/// OnceLock이라 unset 불가능하지만 finalize() 후 모든 hook fire가
/// noop이 됨 (dispatcher 내부 finalized atomic).
pub fn retire_phase_aware(ctx: &mut SwapDispatchCtx<'_>) -> anyhow::Result<()> {
    // ── LISWAP-5 Phase-aware Swap retire ──────────────────────────────
    if let Some(disp) = ctx.phase_aware_swap_dispatcher.as_ref()
        && std::env::var("LLMRS_PHASE_AWARE_DEBUG").as_deref() == Ok("1")
        && ctx.decode_token_index < 5
    {
        let (q, inf, p, d, hs, he, ce) = disp.debug_snapshot();
        eprintln!(
            "[PhaseAwareSwap-DBG] tok={} queue={} in_flight={} pending={} dispatched={} hook_start={} hook_end={} cachefit_end={}",
            ctx.decode_token_index, q, inf, p, d, hs, he, ce
        );
    }
    if let Some(disp) = ctx.phase_aware_swap_dispatcher.as_ref()
        && disp.is_complete()
    {
        let drain_t = std::time::Instant::now();
        let backend_for_invalidate: Arc<dyn Backend> = ctx
            .gpu_backend_arc
            .as_ref()
            .cloned()
            .unwrap_or_else(|| Arc::clone(ctx.backend));
        let invalidate = move || {
            backend_for_invalidate.invalidate_noshuffle_soa_registry();
        };
        match disp.finalize(
            &ctx.model.ratio_generation,
            invalidate,
            std::time::Duration::from_secs(10),
        ) {
            Ok(()) => {
                eprintln!(
                    "[PhaseAwareSwap] plan retired at token={} (drain+sync+bump+invalidate {:.1}ms, chunks={})",
                    ctx.decode_token_index,
                    drain_t.elapsed().as_secs_f64() * 1000.0,
                    disp.dispatched_count(),
                );
                #[cfg(feature = "opencl")]
                remap_weights_for_cpu_after_swap(
                    ctx.model,
                    ctx.backend,
                    ctx.is_gpu,
                    ctx.args.resilience_prealloc_switch,
                    "phase-aware-swap",
                );
            }
            Err(e) => {
                eprintln!(
                    "[PhaseAwareSwap] finalize failed at token={}: {}",
                    ctx.decode_token_index, e
                );
            }
        }
        *ctx.phase_aware_swap_dispatcher = None;
    }
    // ── End LISWAP-5 retire ────────────────────────────────────────────
    Ok(())
}

fn remap_weights_for_cpu_after_swap(
    model: &mut TransformerModel,
    backend: &Arc<dyn Backend>,
    is_gpu: bool,
    enabled: bool,
    label: &str,
) {
    if !is_gpu || !enabled {
        return;
    }
    match model.map_weights_for_cpu(backend) {
        Ok(0) => {}
        Ok(n) => eprintln!(
            "[Backend] Re-mapped {} weight tensors after {} (host pointer restored)",
            n, label,
        ),
        Err(e) => eprintln!(
            "[Backend] Post-swap re-map failed: {} (switch_hw cpu may crash)",
            e,
        ),
    }
}
