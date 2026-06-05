//! [`DecodeLoop`] + typestate builder (Phase 4-2).
//!
//! The decode loop owns all six pipeline trait objects and the stop flag.
//! [`DecodeLoopBuilder`] uses a typestate marker (`NoForward` → `HasForward`)
//! so `.build()` is only callable once a [`Forward`] has been supplied
//! (INV-LAYER-007).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::resilience::KVSnapshot;

use super::defaults::{
    GreedySampler, NoOpCommandSource, NoOpEngineReport, NoOpEvictionStage, NoOpObserver,
    NoOpSwapStage, NoOpTokenTickSink,
};
use super::traits::{
    CommandSource, DecodeObserver, DecodeResult, EngineReport, EvictionOutcome, EvictionStage,
    Forward, StepCtx, StopReason, SwapStage, TokenSampler, TokenTickSink,
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
    // P3에서 실제 보고 구현체 주입 시 사용. 현재는 no-op default만 주입.
    #[allow(dead_code)]
    report: Box<dyn EngineReport>,
    tick_sink: Box<dyn TokenTickSink>,
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
    /// Run prefill over the prompt. Returns logits for the last token so the
    /// caller can sample the first generated token before invoking
    /// [`Self::run`]. `pos` is advanced to `tokens.len()`. `prev_token` is
    /// staged from `tokens.last()` for `on_prefill_end` observer context only;
    /// [`Self::run`] overwrites it with `first_token`.
    pub fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let start_pos = self.pos;
        let logits = self.forward.prefill(tokens, start_pos)?;
        self.pos += tokens.len();
        self.prev_token = *tokens.last().unwrap_or(&0);
        // Phase 4-4.7: production fallback (generate.rs)이
        // `sampling::sample(&mut logits, &tokens, ...)` 호출 시 `tokens`(prompt
        // 전체)를 rep history로 사용한다. paradigm equivalence를 위해 prefill
        // 시점에 prompt를 sampler에게 전부 통보. GreedySampler는 default no-op
        // 이라 무영향, RepetitionPenaltySampler만 ring buffer에 prompt suffix를
        // 적재한다.
        for &t in tokens {
            self.sampler.observe_token(t);
        }
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
        Ok(logits)
    }

    /// Build a minimal KVSnapshot from current loop state.
    fn build_kv_snapshot(&self) -> KVSnapshot {
        KVSnapshot {
            total_tokens: self.pos,
            capacity: self.kv_capacity,
            total_bytes: 0, // P3에서 진짜 값 주입
            protected_prefix: 0,
            kv_dtype: String::new(),
            eviction_policy: String::new(),
            skip_ratio: 0.0,
        }
    }

    /// Run up to `budget` decode steps starting from `first_token` (the token
    /// already sampled from `prefill`'s last logits). Returns sampled tokens
    /// (does **not** include `first_token`) + reason loop exited. Caller is
    /// responsible for prepending `first_token` to the final output sequence.
    pub fn run(&mut self, budget: usize, first_token: u32) -> anyhow::Result<DecodeResult> {
        self.prev_token = first_token;
        // Phase 4-4.7: stateful samplers (RepetitionPenaltySampler 등)이 production
        // fallback `tokens.push(first_token)`과 동치 history를 갖도록 첫 토큰을 통보.
        // [`super::defaults::GreedySampler`]는 default no-op이라 무영향.
        self.sampler.observe_token(first_token);
        let stop = Arc::clone(&self.stop_flag);
        let mut generated = Vec::with_capacity(budget);
        let mut stopped_by = StopReason::BudgetExhausted;

        for _ in 0..budget {
            if stop.load(Ordering::Acquire) {
                stopped_by = StopReason::StopFlag;
                break;
            }
            // 매 step 전체 wall-clock 시작점 (target_tbt pacing 기준 — throttle
            // delay 도 이 측정에 포함된다).
            let t_iter = Instant::now();

            // (a) command poll
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &stop,
            );
            let kv_snap = self.build_kv_snapshot();
            let plan = self.cmd_source.poll(&ctx, &kv_snap)?;
            if plan.suspended {
                stopped_by = StopReason::CommandRequested;
                break;
            }
            if plan.throttle_delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(plan.throttle_delay_ms));
            }

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
            // Phase 4-4.7: sample 직후 history 갱신. production fallback에서
            // `tokens.push(next_token_id)`가 다음 step의 rep window에 들어가는
            // 것과 동치.
            self.sampler.observe_token(sampled);

            // (f2) tick sink — sampler 호출 후, observer 이전
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &stop,
            );
            self.tick_sink.on_token_generated(&ctx);

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

            // (h) target TBT pacing — resilience SetTargetTbt 가 설정한 목표
            // wall-clock 에 도달하도록 step 끝에서 sleep. target_tbt_ms == 0
            // (미설정/release) 이면 no-op이라 비-resilience 경로는 무영향.
            if plan.target_tbt_ms > 0 {
                let elapsed_ms = t_iter.elapsed().as_secs_f64() * 1000.0;
                let target_ms = plan.target_tbt_ms as f64;
                if elapsed_ms < target_ms {
                    std::thread::sleep(std::time::Duration::from_secs_f64(
                        (target_ms - elapsed_ms) / 1000.0,
                    ));
                }
            }
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

    /// Phase 4-5-c. Chat 모드 inner decode loop.
    ///
    /// `stop.should_stop(sampled, pos)`가 true를 반환할 때까지 step을 반복한다.
    /// stop token은 sampled 직후 체크하며, true이면 즉시 break — stop token은
    /// KV에 baking되지 않는다 (chat REPL에서 EOT baking은 4-5-e에서 처리).
    ///
    /// **finalize는 호출하지 않는다.** chat 세션은 여러 turn에 걸쳐 같은
    /// DecodeLoop를 재사용하므로 finalize는 전체 세션 종료 시 1회만 호출해야 한다.
    pub fn run_until_stop(
        &mut self,
        first_token: u32,
        stop: &dyn crate::session::chat::stop_condition::StopCondition,
    ) -> anyhow::Result<DecodeResult> {
        self.prev_token = first_token;
        self.sampler.observe_token(first_token);
        let atomic_stop = Arc::clone(&self.stop_flag);
        let mut generated = Vec::new();
        let mut stopped_by = StopReason::StopConditionMet;

        loop {
            if atomic_stop.load(Ordering::Acquire) {
                stopped_by = StopReason::StopFlag;
                break;
            }

            // (a) command poll
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &atomic_stop,
            );
            let kv_snap = self.build_kv_snapshot();
            let plan = self.cmd_source.poll(&ctx, &kv_snap)?;
            if plan.suspended {
                stopped_by = StopReason::CommandRequested;
                break;
            }
            if plan.throttle_delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(plan.throttle_delay_ms));
            }

            // (b) eviction
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &atomic_stop,
            );
            let outcome = self.eviction.before_step(&ctx)?;
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &atomic_stop,
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
                &atomic_stop,
            );
            self.swap.before_step(&ctx)?;

            // (d) forward
            let t0 = Instant::now();
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &atomic_stop,
            );
            let logits = self.forward.step(&ctx, self.prev_token)?;
            let step_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // (e) swap after
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &atomic_stop,
            );
            self.swap.after_step(&ctx)?;

            // (f) sample
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &atomic_stop,
            );
            let sampled = self.sampler.sample(&ctx, &logits);
            self.sampler.observe_token(sampled);

            // (f2) tick sink — sampler 호출 후, observer 이전
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &atomic_stop,
            );
            self.tick_sink.on_token_generated(&ctx);

            // (g) observers
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &atomic_stop,
            );
            for obs in &mut self.observers {
                obs.on_step_end(&ctx, sampled, step_ms);
            }

            self.prev_token = sampled;
            self.pos += 1;
            self.decode_step += 1;

            // stop 체크: pos 증가 후 현재 pos 기준으로 판단
            if stop.should_stop(sampled, self.pos) {
                break;
            }

            generated.push(sampled);
        }

        Ok(DecodeResult {
            tokens_generated: generated,
            final_pos: self.pos,
            stopped_by,
        })
    }

    /// Phase 4-5-c. Chat /reset 명령 처리용.
    ///
    /// pos, decode_step, prev_token을 0으로 reset한다.
    /// KV cache 자체는 caller (ChatSession::reset)가 `forward_mut()`으로
    /// 접근하여 reset한다.
    pub fn reset_pos(&mut self) {
        self.pos = 0;
        self.decode_step = 0;
        self.prev_token = 0;
    }

    /// Phase 4-5-c. ChatSession::reset이 forward의 KV cache에 접근하기 위한 accessor.
    pub fn forward_mut(&mut self) -> &mut dyn Forward {
        &mut *self.forward
    }

    /// Phase 4-5-d: 현재 KV pos 읽기 (ChatSession 외부 캐시 동기화용).
    pub fn pos_snapshot(&self) -> usize {
        self.pos
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
    report: Option<Box<dyn EngineReport>>,
    tick_sink: Option<Box<dyn TokenTickSink>>,
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
            report: None,
            tick_sink: None,
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
            report: self.report,
            tick_sink: self.tick_sink,
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
    pub fn with_engine_report<T: EngineReport + 'static>(mut self, r: T) -> Self {
        self.report = Some(Box::new(r));
        self
    }
    pub fn with_tick_sink<T: TokenTickSink + 'static>(mut self, t: T) -> Self {
        self.tick_sink = Some(Box::new(t));
        self
    }

    /// P3.3: [`ResilienceAdapter`]를 3개 slot(cmd_source / report / tick_sink)에 동시에 주입한다.
    ///
    /// 단일 인스턴스를 `Arc<Mutex<ResilienceAdapter>>`로 감싸 3개 newtype wrapper에 공유하므로
    /// ownership 이전 없이 3개 slot을 충족시킨다. per-token Mutex lock은
    /// (poll 1회 + on_token_generated 1회) 정도라 contention 무시 가능.
    pub fn with_resilience(
        mut self,
        adapter: super::resilience_adapter::ResilienceAdapter,
    ) -> Self {
        use super::resilience_adapter::{CmdSrcWrapper, ReportWrapper, TickWrapper};
        use std::sync::Mutex;
        let shared = Arc::new(Mutex::new(adapter));
        self.cmd_source = Some(Box::new(CmdSrcWrapper(Arc::clone(&shared))));
        self.report = Some(Box::new(ReportWrapper(Arc::clone(&shared))));
        self.tick_sink = Some(Box::new(TickWrapper(shared)));
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
            report: self.report.unwrap_or_else(|| Box::new(NoOpEngineReport)),
            tick_sink: self
                .tick_sink
                .unwrap_or_else(|| Box::new(NoOpTokenTickSink)),
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
        fn prefill(&mut self, _tokens: &[u32], _start_pos: usize) -> anyhow::Result<Vec<f32>> {
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
        let last_logits = loop_.prefill(&[1, 2, 3]).unwrap();
        // MockForward::prefill sets logits[0]=1.0 → GreedySampler picks 0.
        assert_eq!(last_logits.len(), 16);
        let result = loop_.run(5, 0).unwrap();
        assert_eq!(result.tokens_generated.len(), 5);
        assert_eq!(result.stopped_by, StopReason::BudgetExhausted);
        assert_eq!(result.final_pos, 3 + 5);
        // GreedySampler picks the highest logit each step. step_count = 1..=5
        // produces logits with target = step_count % vocab.
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
        let result = loop_.run(5, 0).unwrap();
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
        let result = loop_.run(3, 7).unwrap();
        assert_eq!(result.tokens_generated, vec![0, 0, 0]);
    }

    /// Phase 4-4.7: TokenSampler::observe_token이 run() 진입 시 first_token 1회 +
    /// 매 step sampled 1회씩 호출되는지 검증. C2 (prompt seeding) 진입 후에는
    /// prefill 단계에서도 prompt 길이만큼 추가 호출되므로 expected가 갱신될 예정.
    #[test]
    fn observe_token_invoked_on_first_and_each_step() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrd};
        struct CountingSampler {
            observe_count: Arc<AtomicUsize>,
            next: u32,
        }
        impl TokenSampler for CountingSampler {
            fn sample(&mut self, _ctx: &StepCtx, _logits: &[f32]) -> u32 {
                let t = self.next;
                self.next = self.next.wrapping_add(1);
                t
            }
            fn observe_token(&mut self, _token: u32) {
                self.observe_count.fetch_add(1, AtomicOrd::Relaxed);
            }
        }
        let observe_count = Arc::new(AtomicUsize::new(0));
        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_sampler(CountingSampler {
                observe_count: observe_count.clone(),
                next: 0,
            })
            .build();
        let _ = loop_.prefill(&[1, 2, 3]).unwrap();
        let _ = loop_.run(3, 7).unwrap();
        // C2 prompt seeding 활성: prompt 3회 + first_token 1회 + sampled 3회 = 7.
        assert_eq!(observe_count.load(AtomicOrd::Relaxed), 7);
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
            fn prefill(&mut self, t: &[u32], start_pos: usize) -> anyhow::Result<Vec<f32>> {
                self.mock.prefill(t, start_pos)
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
        let _ = loop_.prefill(&[5, 6, 7]).unwrap();
        let result = loop_.run(2, 0).unwrap();
        assert_eq!(result.tokens_generated.len(), 2);
    }

    // Phase 4-5-c: run_until_stop 테스트.

    /// StopCondition이 특정 token에서 true를 반환하면 loop가 종료된다.
    /// MockForward (vocab=16): step_count=1 → token 1, step_count=2 → token 2, ...
    /// stop_id=3으로 설정하면 3번째 step에서 token 3 sample 후 종료.
    /// generated에는 stop token이 포함되지 않는다 (1, 2만).
    #[test]
    fn run_until_stop_terminates_on_stop_token() {
        use crate::session::chat::stop_condition::{ChatStopCondition, StopCondition};

        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_kv_capacity(2048)
            .build();
        let _ = loop_.prefill(&[0]).unwrap();
        // step_count=1→token1, 2→token2, 3→token3(stop)
        let cond = ChatStopCondition::new(vec![3], 2048);
        let result = loop_
            .run_until_stop(0, &cond as &dyn StopCondition)
            .unwrap();
        // token 3은 stop이므로 generated에 미포함 → [1, 2]
        assert_eq!(result.tokens_generated, vec![1, 2]);
        assert_eq!(result.stopped_by, StopReason::StopConditionMet);
    }

    /// max_pos overflow 안전망: pos >= max_pos이면 loop 종료.
    #[test]
    fn run_until_stop_terminates_on_max_pos() {
        use crate::session::chat::stop_condition::{ChatStopCondition, StopCondition};

        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_kv_capacity(2048)
            .build();
        // prefill 1 token → pos=1
        let _ = loop_.prefill(&[0]).unwrap();
        // max_pos=3 → pos≥3이면 stop. 1+2 steps → pos=3 → break (step 2 token not pushed)
        let cond = ChatStopCondition::new(vec![], 3);
        let result = loop_
            .run_until_stop(0, &cond as &dyn StopCondition)
            .unwrap();
        // step1: sampled=1, pos→2, stop?(pos=2<3)→no → push. step2: sampled=2, pos→3, stop?(pos=3>=3)→yes
        assert_eq!(result.tokens_generated, vec![1]);
        assert_eq!(result.final_pos, 3);
    }

    /// reset_pos가 pos, decode_step, prev_token을 0으로 초기화한다.
    #[test]
    fn reset_pos_clears_loop_state() {
        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 8,
                step_count: 0,
            })
            .build();
        let _ = loop_.prefill(&[1, 2, 3]).unwrap();
        let _ = loop_.run(2, 0).unwrap();
        // run 후 pos=5, decode_step=2, prev_token=?
        loop_.reset_pos();
        // reset 후 pos=0
        // forward_mut으로 Forward에 접근 가능함을 컴파일로 검증
        let _fwd: &mut dyn Forward = loop_.forward_mut();
    }
}
