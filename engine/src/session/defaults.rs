//! No-op default implementations for the optional pipeline traits.
//!
//! `DecodeLoopBuilder::build` substitutes these whenever the caller did not
//! provide a concrete implementation (`with_eviction` / `with_swap` / etc).

use llm_shared::EngineCommand;

use super::traits::{
    CommandSource, DecodeObserver, EvictionOutcome, EvictionStage, StepCtx, SkipReason, SwapStage,
    TokenSampler,
};

/// Eviction stage that never prunes. Used when caller skipped `with_eviction`.
pub struct NoOpEvictionStage;

impl EvictionStage for NoOpEvictionStage {
    fn before_step(&mut self, _ctx: &StepCtx) -> anyhow::Result<EvictionOutcome> {
        Ok(EvictionOutcome::Skipped {
            reason: SkipReason::NotNeeded,
        })
    }
}

/// Swap stage that does nothing.
pub struct NoOpSwapStage;

impl SwapStage for NoOpSwapStage {
    fn before_step(&mut self, _ctx: &StepCtx) -> anyhow::Result<()> {
        Ok(())
    }
    fn after_step(&mut self, _ctx: &StepCtx) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Command source that never yields a command.
pub struct NoOpCommandSource;

impl CommandSource for NoOpCommandSource {
    fn poll(&mut self, _ctx: &StepCtx) -> anyhow::Result<Option<EngineCommand>> {
        Ok(None)
    }
}

/// Greedy sampler (argmax). Pipeline default when caller skipped `with_sampler`.
pub struct GreedySampler;

impl TokenSampler for GreedySampler {
    fn sample(&mut self, _ctx: &StepCtx, logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }
}

/// Observer with all hooks as no-op. Useful as a placeholder slot.
pub struct NoOpObserver;

impl DecodeObserver for NoOpObserver {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn ctx<'a>(stop: &'a AtomicBool) -> StepCtx<'a> {
        StepCtx {
            pos: 0,
            prev_token: 0,
            kv_capacity: 0,
            decode_step: 0,
            stop_requested: stop,
        }
    }

    #[test]
    fn greedy_picks_argmax() {
        let stop = AtomicBool::new(false);
        let c = ctx(&stop);
        let mut s = GreedySampler;
        assert_eq!(s.sample(&c, &[0.1, 0.5, 0.3, 0.9, 0.2]), 3);
    }

    #[test]
    fn greedy_handles_negative_logits() {
        let stop = AtomicBool::new(false);
        let c = ctx(&stop);
        let mut s = GreedySampler;
        assert_eq!(s.sample(&c, &[-5.0, -1.0, -3.0]), 1);
    }

    #[test]
    fn no_op_eviction_skipped() {
        let stop = AtomicBool::new(false);
        let c = ctx(&stop);
        let mut e = NoOpEvictionStage;
        match e.before_step(&c).unwrap() {
            EvictionOutcome::Skipped { reason: SkipReason::NotNeeded } => {}
            other => panic!("expected Skipped(NotNeeded), got {other:?}"),
        }
    }

    #[test]
    fn no_op_command_source_yields_none() {
        let stop = AtomicBool::new(false);
        let c = ctx(&stop);
        let mut s = NoOpCommandSource;
        assert!(s.poll(&c).unwrap().is_none());
    }
}
