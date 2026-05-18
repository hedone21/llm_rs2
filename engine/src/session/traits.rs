//! Inference pipeline trait API (Phase 4-2).
//!
//! 6 traits split the decode loop responsibilities by SOLID-SRP:
//! [`Forward`] (required), [`EvictionStage`], [`SwapStage`], [`CommandSource`],
//! [`TokenSampler`], and [`DecodeObserver`].
//!
//! Signatures match `arch/inference_pipeline.md` §2 verbatim. Default methods
//! mean external contributors only have to implement `prefill` + `step` on
//! [`Forward`] to compile against [`super::DecodeLoopBuilder`].

use std::sync::atomic::AtomicBool;

use llm_shared::{EngineCommand, WeightSwapReport};

/// Read-only context handed to every trait at each decode step.
///
/// Borrows the stop flag from the parent [`super::DecodeLoop`] so trait
/// implementations can observe (but not flip) cancellation.
pub struct StepCtx<'a> {
    pub pos: usize,
    pub prev_token: u32,
    pub kv_capacity: usize,
    pub decode_step: usize,
    pub stop_requested: &'a AtomicBool,
}

/// Outcome of an [`EvictionStage::before_step`] call.
#[derive(Debug, Clone)]
pub enum EvictionOutcome {
    None,
    Pruned { removed: usize, new_pos: usize },
    Skipped { reason: SkipReason },
}

/// Reason an eviction stage decided not to act this step.
#[derive(Debug, Clone)]
pub enum SkipReason {
    NotNeeded,
    BudgetExhausted,
    ManagerDeferred,
    Other(&'static str),
}

/// Why [`super::DecodeLoop::run`] returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    BudgetExhausted,
    StopFlag,
    EosToken,
    CommandRequested,
    /// Phase 4-5-c: [`super::DecodeLoop::run_until_stop`]에서 StopCondition이 true를 반환.
    StopConditionMet,
}

/// Decode result returned by [`super::DecodeLoop::run`].
#[derive(Debug, Clone)]
pub struct DecodeResult {
    pub tokens_generated: Vec<u32>,
    pub final_pos: usize,
    pub stopped_by: StopReason,
}

/// Required forward pass. Provides KV-bearing model semantics.
///
/// `finalize` and `on_kv_prune` are default no-ops so a minimal Forward
/// implementation needs only `prefill` + `step`.
pub trait Forward {
    /// Run prefill over `tokens`. Returns logits for the last token.
    fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>>;

    /// Decode 1 step. After return, `pos` is conceptually advanced by 1.
    fn step(&mut self, ctx: &StepCtx, token: u32) -> anyhow::Result<Vec<f32>>;

    /// Called once after [`super::DecodeLoop::run`] exits.
    fn finalize(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Notified after an [`EvictionStage`] pruned KV state.
    fn on_kv_prune(&mut self, _new_pos: usize) {}

    /// Phase 4-5-d: chat `/reset` 처리용. KV cache를 초기 상태로 reset한다.
    ///
    /// Default no-op — generate 모드는 호출하지 않는다. chat 모드의 각 Forward
    /// 구현체가 override하여 KV-type별 reset 로직을 수행한다.
    fn reset_kv(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Phase 4-5-e: chat `ensure_capacity` / `on_turn_end` 에서 eviction 실행.
    ///
    /// ModelForward만 override한다. KiviForward / OffloadForward는 default no-op
    /// 유지 (eviction 미지원).
    ///
    /// - `cache_manager`: eviction 정책을 보유한 CacheManager.
    /// - `scores`: score-based policy (H2O/D2O)용 importance 점수. `None`이면
    ///   score-free force/maybe evict.
    /// - `force`: true이면 `force_evict*`, false이면 `maybe_evict*`.
    /// - `target_ratio`: `force=true` 시 eviction 목표 비율.
    ///
    /// Returns (removed_count, new_pos). removed_count == 0이면 eviction 미발생.
    fn try_evict(
        &mut self,
        cache_manager: &crate::core::cache_manager::CacheManager,
        scores: Option<&[f32]>,
        force: bool,
        target_ratio: f32,
    ) -> anyhow::Result<(usize, usize)> {
        let _ = (cache_manager, scores, force, target_ratio);
        Ok((0, 0))
    }
}

/// Eviction stage invoked before each forward step.
pub trait EvictionStage {
    /// Inspect cache pressure and optionally prune.
    fn before_step(&mut self, ctx: &StepCtx) -> anyhow::Result<EvictionOutcome>;

    /// Ensure enough KV capacity for `additional` upcoming tokens.
    fn ensure_capacity(&mut self, _ctx: &StepCtx, _additional: usize) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Weight/KV swap stage. Pairs `before_step` (prefetch) with `after_step`
/// (commit/release).
pub trait SwapStage {
    fn before_step(&mut self, ctx: &StepCtx) -> anyhow::Result<()>;
    fn after_step(&mut self, ctx: &StepCtx) -> anyhow::Result<()>;

    /// Drain a pending swap report for IPC, if any.
    fn pending_report(&mut self) -> Option<WeightSwapReport> {
        None
    }
}

/// External command channel (manager IPC, schedule, stdin, ...).
pub trait CommandSource {
    fn poll(&mut self, ctx: &StepCtx) -> anyhow::Result<Option<EngineCommand>>;
}

/// Token sampler. Default impl `GreedySampler` in [`super::defaults`].
pub trait TokenSampler {
    fn sample(&mut self, ctx: &StepCtx, logits: &[f32]) -> u32;

    /// Phase 4-4.7: stateful samplers (rep penalty, n-gram blocking, ...)이
    /// 최근 토큰을 ring buffer로 유지하기 위한 hook. Default no-op이라
    /// [`super::defaults::GreedySampler`] 등 stateless impl은 변경 불필요.
    fn observe_token(&mut self, _token: u32) {}
}

/// Decode-loop observer. All hooks are no-op by default; implement only what
/// you need.
pub trait DecodeObserver {
    fn on_prefill_end(&mut self, _ctx: &StepCtx, _last_logits: &[f32]) {}
    fn on_step_end(&mut self, _ctx: &StepCtx, _sampled: u32, _step_ms: f64) {}
    fn on_eviction(&mut self, _ctx: &StepCtx, _outcome: &EvictionOutcome) {}
    fn finalize(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}
