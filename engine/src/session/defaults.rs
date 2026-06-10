//! No-op default implementations for the optional pipeline traits.
//!
//! `DecodeLoopBuilder::build` substitutes these whenever the caller did not
//! provide a concrete implementation (`with_eviction` / `with_swap` / etc).

use llm_shared::EngineCommand;

use crate::inference::sampling::StepCtx;
use crate::session::command_dispatcher::{CommandSource, EngineReport};

use super::traits::{
    DecodeObserver, EvictionOutcome, EvictionStage, SkipReason, SwapStage, TokenTickSink,
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
    fn poll(&mut self) -> anyhow::Result<Vec<EngineCommand>> {
        Ok(Vec::new())
    }
}

/// EngineReport that drops all outbound messages.
pub struct NoOpEngineReport;

impl EngineReport for NoOpEngineReport {}

/// TokenTickSink that does nothing.
pub struct NoOpTokenTickSink;

impl TokenTickSink for NoOpTokenTickSink {}

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
    fn no_op_eviction_skipped() {
        let stop = AtomicBool::new(false);
        let c = ctx(&stop);
        let mut e = NoOpEvictionStage;
        match e.before_step(&c).unwrap() {
            EvictionOutcome::Skipped {
                reason: SkipReason::NotNeeded,
            } => {}
            other => panic!("expected Skipped(NotNeeded), got {other:?}"),
        }
    }

    #[test]
    fn no_op_command_source_yields_empty_commands() {
        let mut s = NoOpCommandSource;
        let cmds = s.poll().unwrap();
        assert!(cmds.is_empty());
    }
}
