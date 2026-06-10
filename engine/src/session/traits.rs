//! Remaining v1 pipeline trait surface (Phase 4-2, slimmed in Phase β-7).
//!
//! Phase β-7 moved the live seams out of this file:
//! - [`Forward`](super::forward::Forward) → `session/forward.rs`
//! - [`TokenSampler`](crate::inference::sampling::TokenSampler) +
//!   [`StepCtx`](crate::inference::sampling::StepCtx) → `inference/sampling.rs`
//! - [`StopReason`](super::decode_loop::StopReason) +
//!   [`DecodeResult`](super::decode_loop::DecodeResult) → `decode_loop.rs`
//! - [`CommandSource`](super::command_dispatcher::CommandSource) +
//!   [`EngineReport`](super::command_dispatcher::EngineReport) → `command_dispatcher.rs`
//!
//! What stays here are the trait objects the [`super::DecodeLoop`] still owns as
//! optional slots — eviction / swap / observer / tick. They default to no-op
//! impls in [`super::defaults`].

use llm_shared::WeightSwapReport;

use crate::inference::sampling::StepCtx;

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

/// Per-token tick sink.
pub trait TokenTickSink {
    /// decode step 완료 후 1회. sampler 호출 후, observer 이전.
    fn on_token_generated(&mut self, _ctx: &StepCtx) {}
}

/// Convenience supertrait bundling the three resilience-facing traits.
pub trait ResilienceBundle:
    super::command_dispatcher::CommandSource + super::command_dispatcher::EngineReport + TokenTickSink
{
}
impl<
    T: super::command_dispatcher::CommandSource
        + super::command_dispatcher::EngineReport
        + TokenTickSink,
> ResilienceBundle for T
{
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
