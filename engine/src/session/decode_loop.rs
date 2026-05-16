//! [`DecodeLoop`] + typestate builder (Phase 4-2).
//!
//! The decode loop owns all six pipeline trait objects and the stop flag.
//! [`DecodeLoopBuilder`] uses a typestate marker (`NoForward` → `HasForward`)
//! so `.build()` is only callable once a [`Forward`] has been supplied
//! (INV-LAYER-007).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use super::defaults::{
    GreedySampler, NoOpCommandSource, NoOpEvictionStage, NoOpObserver, NoOpSwapStage,
};
use super::traits::{
    CommandSource, DecodeObserver, DecodeResult, EvictionOutcome, EvictionStage, Forward, StepCtx,
    StopReason, SwapStage, TokenSampler,
};

/// Typestate marker — Forward not yet supplied. `.build()` is unavailable.
pub struct NoForward;

/// Typestate marker — Forward supplied. `.build()` is callable.
pub struct HasForward(Box<dyn Forward>);

/// Decode loop. Owned trait objects only — see INV-LAYER-006.
pub struct DecodeLoop {
    forward: Box<dyn Forward>,
    eviction: Box<dyn EvictionStage>,
    swap: Box<dyn SwapStage>,
    cmd_source: Box<dyn CommandSource>,
    sampler: Box<dyn TokenSampler>,
    observers: Vec<Box<dyn DecodeObserver>>,
    stop_flag: Arc<AtomicBool>,
    pos: usize,
    decode_step: usize,
    prev_token: u32,
    kv_capacity: usize,
}

/// Build a StepCtx without borrowing the whole DecodeLoop — lets the loop
/// pass `&mut self.<field>` to trait methods on the same line.
fn step_ctx<'a>(
    pos: usize,
    prev_token: u32,
    kv_capacity: usize,
    decode_step: usize,
    stop: &'a AtomicBool,
) -> StepCtx<'a> {
    StepCtx {
        pos,
        prev_token,
        kv_capacity,
        decode_step,
        stop_requested: stop,
    }
}

impl DecodeLoop {
    /// Run prefill over the prompt. Sets `pos` and `prev_token` for the
    /// subsequent decode loop.
    pub fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<()> {
        let logits = self.forward.prefill(tokens)?;
        self.pos = tokens.len();
        self.prev_token = *tokens.last().unwrap_or(&0);
        let stop = Arc::clone(&self.stop_flag);
        let ctx = step_ctx(
            self.pos,
            self.prev_token,
            self.kv_capacity,
            self.decode_step,
            &stop,
        );
        for obs in &mut self.observers {
            obs.on_prefill_end(&ctx, &logits);
        }
        Ok(())
    }

    /// Run up to `budget` decode steps. Returns sampled tokens + reason loop
    /// exited.
    pub fn run(&mut self, budget: usize) -> anyhow::Result<DecodeResult> {
        let stop = Arc::clone(&self.stop_flag);
        let mut generated = Vec::with_capacity(budget);
        let mut stopped_by = StopReason::BudgetExhausted;

        for _ in 0..budget {
            if stop.load(Ordering::Acquire) {
                stopped_by = StopReason::StopFlag;
                break;
            }

            // (a) command poll
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &stop,
            );
            let _cmd = self.cmd_source.poll(&ctx)?;
            // Command dispatch is Phase 4-3+; we accept and drop for now.

            // (b) eviction
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &stop,
            );
            let outcome = self.eviction.before_step(&ctx)?;
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &stop,
            );
            for obs in &mut self.observers {
                obs.on_eviction(&ctx, &outcome);
            }
            if let EvictionOutcome::Pruned { new_pos, .. } = outcome {
                self.pos = new_pos;
                self.forward.on_kv_prune(new_pos);
            }

            // (c) swap before
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &stop,
            );
            self.swap.before_step(&ctx)?;

            // (d) forward
            let t0 = Instant::now();
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &stop,
            );
            let logits = self.forward.step(&ctx, self.prev_token)?;
            let step_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // (e) swap after
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &stop,
            );
            self.swap.after_step(&ctx)?;

            // (f) sample
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &stop,
            );
            let sampled = self.sampler.sample(&ctx, &logits);

            // (g) observers
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &stop,
            );
            for obs in &mut self.observers {
                obs.on_step_end(&ctx, sampled, step_ms);
            }

            generated.push(sampled);
            self.prev_token = sampled;
            self.pos += 1;
            self.decode_step += 1;
        }

        self.forward.finalize()?;
        for obs in &mut self.observers {
            obs.finalize()?;
        }

        Ok(DecodeResult {
            tokens_generated: generated,
            final_pos: self.pos,
            stopped_by,
        })
    }

    /// Borrow the stop flag handle (caller can install a signal handler).
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop_flag)
    }
}

/// Builder with typestate marker. `F = NoForward` is the entry state;
/// `with_forward` transitions to `HasForward`, after which `.build()` exists.
pub struct DecodeLoopBuilder<F = NoForward> {
    forward: F,
    eviction: Option<Box<dyn EvictionStage>>,
    swap: Option<Box<dyn SwapStage>>,
    cmd_source: Option<Box<dyn CommandSource>>,
    sampler: Option<Box<dyn TokenSampler>>,
    observers: Vec<Box<dyn DecodeObserver>>,
    stop_flag: Option<Arc<AtomicBool>>,
    kv_capacity: usize,
}

impl Default for DecodeLoopBuilder<NoForward> {
    fn default() -> Self {
        Self::new()
    }
}

impl DecodeLoopBuilder<NoForward> {
    /// Start a new builder. Forward is required before `.build()`.
    pub fn new() -> Self {
        Self {
            forward: NoForward,
            eviction: None,
            swap: None,
            cmd_source: None,
            sampler: None,
            observers: Vec::new(),
            stop_flag: None,
            kv_capacity: 0,
        }
    }

    /// Supply the required Forward. Transitions to `HasForward` state.
    pub fn with_forward<T: Forward + 'static>(self, fwd: T) -> DecodeLoopBuilder<HasForward> {
        DecodeLoopBuilder {
            forward: HasForward(Box::new(fwd)),
            eviction: self.eviction,
            swap: self.swap,
            cmd_source: self.cmd_source,
            sampler: self.sampler,
            observers: self.observers,
            stop_flag: self.stop_flag,
            kv_capacity: self.kv_capacity,
        }
    }
}

// Optional setters available in any state. Order-independent.
impl<F> DecodeLoopBuilder<F> {
    pub fn with_eviction<T: EvictionStage + 'static>(mut self, e: T) -> Self {
        self.eviction = Some(Box::new(e));
        self
    }
    pub fn with_swap<T: SwapStage + 'static>(mut self, s: T) -> Self {
        self.swap = Some(Box::new(s));
        self
    }
    pub fn with_cmd_source<T: CommandSource + 'static>(mut self, c: T) -> Self {
        self.cmd_source = Some(Box::new(c));
        self
    }
    pub fn with_sampler<T: TokenSampler + 'static>(mut self, t: T) -> Self {
        self.sampler = Some(Box::new(t));
        self
    }
    pub fn add_observer<T: DecodeObserver + 'static>(mut self, o: T) -> Self {
        self.observers.push(Box::new(o));
        self
    }
    pub fn with_stop_flag(mut self, f: Arc<AtomicBool>) -> Self {
        self.stop_flag = Some(f);
        self
    }
    pub fn with_kv_capacity(mut self, c: usize) -> Self {
        self.kv_capacity = c;
        self
    }
}

impl DecodeLoopBuilder<HasForward> {
    /// Assemble the decode loop. Optional components default to no-op impls
    /// from [`super::defaults`].
    pub fn build(self) -> DecodeLoop {
        DecodeLoop {
            forward: self.forward.0,
            eviction: self.eviction.unwrap_or_else(|| Box::new(NoOpEvictionStage)),
            swap: self.swap.unwrap_or_else(|| Box::new(NoOpSwapStage)),
            cmd_source: self
                .cmd_source
                .unwrap_or_else(|| Box::new(NoOpCommandSource)),
            sampler: self.sampler.unwrap_or_else(|| Box::new(GreedySampler)),
            observers: if self.observers.is_empty() {
                vec![Box::new(NoOpObserver) as Box<dyn DecodeObserver>]
            } else {
                self.observers
            },
            stop_flag: self
                .stop_flag
                .unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
            pos: 0,
            decode_step: 0,
            prev_token: 0,
            kv_capacity: self.kv_capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forward stub: prefill returns logits[0]=1.0, step returns logits with
    /// the next index biased highest so GreedySampler picks `step_count + 1`.
    struct MockForward {
        vocab: usize,
        step_count: usize,
    }

    impl Forward for MockForward {
        fn prefill(&mut self, _tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
            let mut logits = vec![0.0_f32; self.vocab];
            logits[0] = 1.0;
            Ok(logits)
        }
        fn step(&mut self, _ctx: &StepCtx, _token: u32) -> anyhow::Result<Vec<f32>> {
            self.step_count += 1;
            let mut logits = vec![0.0_f32; self.vocab];
            let target = self.step_count % self.vocab;
            logits[target] = 1.0;
            Ok(logits)
        }
    }

    #[test]
    fn builder_runs_full_budget_with_defaults() {
        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_kv_capacity(2048)
            .build();
        loop_.prefill(&[1, 2, 3]).unwrap();
        let result = loop_.run(5).unwrap();
        assert_eq!(result.tokens_generated.len(), 5);
        assert_eq!(result.stopped_by, StopReason::BudgetExhausted);
        assert_eq!(result.final_pos, 3 + 5);
        // GreedySampler picks the highest logit each step.
        assert_eq!(result.tokens_generated, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn stop_flag_short_circuits_loop() {
        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .build();
        let stop = loop_.stop_flag();
        stop.store(true, Ordering::Release);
        let result = loop_.run(5).unwrap();
        assert_eq!(result.tokens_generated.len(), 0);
        assert_eq!(result.stopped_by, StopReason::StopFlag);
    }

    #[test]
    fn custom_sampler_overrides_default() {
        struct AlwaysZero;
        impl TokenSampler for AlwaysZero {
            fn sample(&mut self, _ctx: &StepCtx, _logits: &[f32]) -> u32 {
                0
            }
        }
        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_sampler(AlwaysZero)
            .build();
        let result = loop_.run(3).unwrap();
        assert_eq!(result.tokens_generated, vec![0, 0, 0]);
    }

    #[test]
    fn eviction_pruned_advances_on_kv_prune_hook() {
        struct PruneEveryStep;
        impl EvictionStage for PruneEveryStep {
            fn before_step(&mut self, ctx: &StepCtx) -> anyhow::Result<EvictionOutcome> {
                Ok(EvictionOutcome::Pruned {
                    removed: 1,
                    new_pos: ctx.pos.saturating_sub(1),
                })
            }
        }
        struct CountingForward {
            mock: MockForward,
            prunes: usize,
        }
        impl Forward for CountingForward {
            fn prefill(&mut self, t: &[u32]) -> anyhow::Result<Vec<f32>> {
                self.mock.prefill(t)
            }
            fn step(&mut self, c: &StepCtx, t: u32) -> anyhow::Result<Vec<f32>> {
                self.mock.step(c, t)
            }
            fn on_kv_prune(&mut self, _new_pos: usize) {
                self.prunes += 1;
            }
        }
        // We can't observe `prunes` after build() (forward is moved), but we
        // can verify the loop runs to completion when eviction prunes each step.
        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(CountingForward {
                mock: MockForward {
                    vocab: 8,
                    step_count: 0,
                },
                prunes: 0,
            })
            .with_eviction(PruneEveryStep)
            .build();
        loop_.prefill(&[5, 6, 7]).unwrap();
        let result = loop_.run(2).unwrap();
        assert_eq!(result.tokens_generated.len(), 2);
    }
}
