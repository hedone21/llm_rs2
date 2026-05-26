//! Engine-internal swap dispatch HOW layer.
//!
//! Manager → Engine wire format은 `shared::EngineCommand::SwapWeights {
//! ratio, target_dtype }` 3 필드 (WHAT). 본 모듈은 그 명령을 받아
//! engine 내부 default mode (`--swap` CLI flag normalize 결과) 로
//! mode-specific 객체를 생성하여 `SwapCommitSlot`에 commit한다.
//!
//! Mental model: arch/weight_swap.md §2.8.1 — Manager는 WHAT만, Engine은
//! HOW를 자율 결정. swap mode (Incremental / IntraForward / PhaseAware /
//! LayerImmediate) 는 wire format에 노출되지 않는다.

use std::sync::Arc;
use std::time::Instant;

use llm_shared::DtypeTag;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::models::transformer::TransformerModel;
use crate::models::weights::{
    AsyncSwapDispatcher, IncrementalSwapPlan, IntraForwardSwapHook, PhaseAwareSwapDispatcher,
    PrimaryReleaseWorker,
};
use crate::observability::events::EventSink;
use crate::session::cli::SwapMode;

/// Engine 내부 swap dispatch의 commit 결과.
///
/// 본 sprint scope (α) 에서는 caller (decode loop) 가 기존 3개의 별도
/// `Option<>` 변수 (`incremental_force_swap_plan` / `intra_forward_swap_hook` /
/// `phase_aware_swap_dispatcher`) 로 분해하여 소비하므로, 본 enum 은
/// `EngineSwapRuntime::handle_swap_weights` 의 in/out 형태 통일을 위해
/// 도입된다. 후속 sprint 에서 decode loop가 enum 직접 소비하도록 단계적
/// 통합 가능.
#[derive(Default)]
pub enum SwapCommitSlot {
    #[default]
    Idle,
    Incremental(IncrementalSwapPlan),
    IntraForward(Arc<IntraForwardSwapHook>),
    PhaseAware(Arc<PhaseAwareSwapDispatcher>),
}

impl SwapCommitSlot {
    pub fn is_idle(&self) -> bool {
        matches!(self, Self::Idle)
    }

    pub fn take(&mut self) -> SwapCommitSlot {
        std::mem::replace(self, Self::Idle)
    }
}

/// Engine-wide swap dispatch 자원 묶음.
///
/// `main()` 진입 시 1회 생성. Manager 또는 CLI force-swap 신호 수신 시
/// `handle_*` method 가 self 의 자원을 사용하여 commit slot 에 mode-specific
/// 객체를 commit. 본 sprint (α) 는 Manager 경로만 통합 — CLI 강제 경로
/// (`dispatch_force_swap!` 매크로) 는 기존 그대로 유지.
pub struct EngineSwapRuntime {
    swap_backend: Arc<dyn Backend>,
    dispatcher: Arc<AsyncSwapDispatcher>,
    config: Arc<crate::model_config::ModelConfig>,
    release_worker: Arc<PrimaryReleaseWorker>,
    event_sink: Arc<dyn EventSink>,
    /// CLI `--swap` flag normalize 결과. Manager-driven swap 시 이 mode로 commit.
    default_mode: SwapMode,
    /// PhaseAware mode 전용: `--swap-phase-aware-chunk-mb` * 1 MB.
    phase_chunk_size_bytes: usize,
    /// PhaseAware mode 전용: `--swap-phase-aware-max-chunks-per-token`.
    phase_max_chunks_per_token: usize,
}

impl EngineSwapRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        swap_backend: Arc<dyn Backend>,
        dispatcher: Arc<AsyncSwapDispatcher>,
        config: Arc<crate::model_config::ModelConfig>,
        release_worker: Arc<PrimaryReleaseWorker>,
        event_sink: Arc<dyn EventSink>,
        default_mode: SwapMode,
        phase_chunk_size_bytes: usize,
        phase_max_chunks_per_token: usize,
    ) -> Self {
        Self {
            swap_backend,
            dispatcher,
            config,
            release_worker,
            event_sink,
            default_mode,
            phase_chunk_size_bytes,
            phase_max_chunks_per_token,
        }
    }

    pub fn default_mode(&self) -> SwapMode {
        self.default_mode
    }

    pub fn swap_backend(&self) -> &Arc<dyn Backend> {
        &self.swap_backend
    }

    pub fn dispatcher(&self) -> &Arc<AsyncSwapDispatcher> {
        &self.dispatcher
    }

    pub fn config(&self) -> &Arc<crate::model_config::ModelConfig> {
        &self.config
    }

    pub fn release_worker(&self) -> &Arc<PrimaryReleaseWorker> {
        &self.release_worker
    }

    pub fn event_sink(&self) -> &Arc<dyn EventSink> {
        &self.event_sink
    }

    pub fn phase_chunk_size_bytes(&self) -> usize {
        self.phase_chunk_size_bytes
    }

    pub fn phase_max_chunks_per_token(&self) -> usize {
        self.phase_max_chunks_per_token
    }

    /// Manager `SwapWeights` 신호 처리 — engine 내부 default mode로 mode-specific
    /// 객체를 생성하여 `commit_slot`에 commit.
    ///
    /// 본 method 는 `dispatch_swap_weights` (Incremental hardcoded) 의 mode-aware
    /// 후속 — wire format은 무변경 (`ratio` + `target_dtype` 2 필드), mode는
    /// `self.default_mode` (CLI `--swap` flag normalize 결과).
    ///
    /// 단계:
    /// 1. Validate — secondary present / ratio in (0,1] / target_dtype=Q4_0.
    /// 2. In-flight check — commit_slot이 Idle 인지 확인 (R-1 동시 활성화 가드).
    /// 3. Decider — `WeightSwapDecider::decide(ratio)` → `selected_layers`.
    /// 4. QCF estimate — `compute_qcf_weight_swap`.
    /// 5. Mode-specific commit — `match self.default_mode` 4-way.
    /// 6. Manager report — 후속 plan-retire 시점에 `WeightSwapReport`로 송신.
    #[allow(clippy::too_many_arguments)]
    pub fn handle_swap_weights(
        &self,
        model: &TransformerModel,
        ratio: f32,
        target_dtype: DtypeTag,
        importance: Option<&dyn crate::qcf_collector::ImportanceLookup>,
        decode_token_index: usize,
        commit_slot: &mut SwapCommitSlot,
        manager_report_out: &mut Option<(f32, usize, Instant, f32)>,
    ) {
        use crate::models::weights::{
            SwapAlgorithm, SwapDecision, WeightSwapDecider, compute_qcf_weight_swap,
        };

        // ── 1. Validate ───────────────────────────────────────────────────
        let Some(secondary) = model.secondary_mmap.as_ref() else {
            eprintln!("[WeightSwap] Rejected: no_secondary (ENG-DAT-C09)");
            return;
        };
        if ratio <= 0.0 || ratio > 1.0 {
            eprintln!("[WeightSwap] Rejected: invalid_ratio ({:.4})", ratio);
            return;
        }
        if target_dtype != DtypeTag::Q4_0 {
            eprintln!(
                "[WeightSwap] Rejected: unsupported_dtype ({:?}) (INV-126)",
                target_dtype
            );
            return;
        }

        // ── 2. In-flight check (R-1 동시 활성화 가드) ────────────────────
        if !commit_slot.is_idle() {
            eprintln!(
                "[WeightSwap] Rejected: commit slot already active (ratio={:.2}). \
                 Wait for current plan to complete before sending a new SwapWeights signal.",
                ratio
            );
            return;
        }

        // ── 3. Collect currently-swapped layers ──────────────────────────
        let n_layers = model.layers.len();
        let currently_swapped: Vec<usize> = (0..n_layers)
            .filter(|&i| model.layers[i].current_dtype() == DType::Q4_0)
            .collect();

        // ── 4. Decider ───────────────────────────────────────────────────
        let allow_boundary = crate::session::qcf_runtime::read_allow_boundary_env();
        eprintln!(
            "[Decider] allow_boundary_layers={} (ratio={:.4}, mode={:?})",
            allow_boundary, ratio, self.default_mode
        );
        let decider = WeightSwapDecider {
            importance,
            noise: Some(&model.quant_noise),
            n_decoder_layers: n_layers,
            currently_swapped: &currently_swapped,
            allow_boundary_layers: allow_boundary,
            algorithm: SwapAlgorithm::ImportanceAware,
        };
        let decision: SwapDecision = decider.decide(ratio);

        if decision.selected_layers.is_empty() {
            eprintln!(
                "[WeightSwap] No layers to swap (ratio={:.2}, already_swapped={})",
                ratio,
                currently_swapped.len()
            );
            return;
        }

        // ── 5. QCF estimate ──────────────────────────────────────────────
        let qcf_swap_estimated = compute_qcf_weight_swap(
            &decision.selected_layers,
            &model.quant_noise,
            importance,
            n_layers,
        );

        let n_planned = decision.selected_layers.len();

        // ── 6. Mode-specific commit ──────────────────────────────────────
        match self.default_mode {
            SwapMode::Incremental => {
                // Manager-driven default per_tick (legacy hardcode 보존).
                let per_tick = 2usize;
                let ticks_est = n_planned.div_ceil(per_tick);
                eprintln!(
                    "[WeightSwap] manager path (Incremental): ratio={:.2}, {} layers, per_tick={} ({} ticks), qcf={:.4}",
                    ratio, n_planned, per_tick, ticks_est, qcf_swap_estimated,
                );
                let plan = IncrementalSwapPlan::new(
                    decision.selected_layers,
                    per_tick,
                    decode_token_index,
                );
                *commit_slot = SwapCommitSlot::Incremental(plan);
            }
            SwapMode::IntraForward | SwapMode::LayerImmediate => {
                let mode_label = if matches!(self.default_mode, SwapMode::LayerImmediate) {
                    "layer-immediate"
                } else {
                    "intra-forward"
                };
                eprintln!(
                    "[WeightSwap] manager path ({}): ratio={:.2}, {} layers, qcf={:.4}",
                    mode_label, ratio, n_planned, qcf_swap_estimated,
                );
                let hook = IntraForwardSwapHook::new(
                    decision.selected_layers,
                    decode_token_index,
                    Arc::clone(&self.dispatcher),
                    Arc::clone(secondary),
                    model.layers.clone(),
                    Arc::clone(&self.swap_backend),
                    Some(Arc::clone(&self.release_worker)),
                    DType::Q4_0,
                    Arc::clone(&self.config),
                    Arc::clone(&self.event_sink),
                );
                *commit_slot = SwapCommitSlot::IntraForward(hook);
            }
            SwapMode::PhaseAware => {
                eprintln!(
                    "[WeightSwap] manager path (PhaseAware): ratio={:.2}, {} layers, chunk_size_bytes={}, qcf={:.4}",
                    ratio, n_planned, self.phase_chunk_size_bytes, qcf_swap_estimated,
                );
                let phase_dispatcher = PhaseAwareSwapDispatcher::new(
                    self.phase_chunk_size_bytes,
                    model.layers.clone(),
                    Arc::clone(secondary),
                    Arc::clone(&self.swap_backend),
                    Arc::clone(&self.dispatcher),
                    DType::Q4_0,
                    Arc::clone(&self.config),
                    Arc::clone(&self.event_sink),
                );
                phase_dispatcher.install_self_weak();
                phase_dispatcher.commit_plan(&decision.selected_layers);
                phase_dispatcher.set_max_chunks_per_token(self.phase_max_chunks_per_token);
                crate::observability::profile::op_trace::set_phase_hook(phase_dispatcher.clone()
                    as Arc<dyn crate::observability::profile::op_trace::PhaseHook>);
                *commit_slot = SwapCommitSlot::PhaseAware(phase_dispatcher);
            }
        }

        // ── 7. Manager report ────────────────────────────────────────────
        *manager_report_out = Some((ratio, n_planned, Instant::now(), qcf_swap_estimated));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_commit_slot_default_is_idle() {
        let slot = SwapCommitSlot::default();
        assert!(slot.is_idle());
    }

    #[test]
    fn swap_commit_slot_take_resets_to_idle() {
        let mut slot = SwapCommitSlot::default();
        assert!(slot.is_idle());
        let taken = slot.take();
        assert!(matches!(taken, SwapCommitSlot::Idle));
        assert!(slot.is_idle());
    }
}
