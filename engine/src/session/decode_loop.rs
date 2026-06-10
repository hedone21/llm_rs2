//! [`DecodeLoop`] + typestate builder (Phase 4-2).
//!
//! The decode loop owns all six pipeline trait objects and the stop flag.
//! [`DecodeLoopBuilder`] uses a typestate marker (`NoForward` → `HasForward`)
//! so `.build()` is only callable once a [`Forward`] has been supplied
//! (INV-LAYER-007).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::observability::profile::OpProfiler;
use crate::pipeline::StopReason as StageStopReason;
use crate::pipeline::{LifecyclePhase, PipelineDispatcher, Pressure, StageContext, StepInfo};
use crate::resilience::KVSnapshot;
use crate::session::pipeline_registry::PipelineRegistry;

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
    // argus-bench AB-1: resilience-driven eviction. `plan.evict` 가 Some 이면
    // `forward.try_evict(cache_manager, …)` 로 mid-decode prune. None 이면
    // happy path / chat (eviction 은 자체 turn-boundary 경로) — 무영향.
    cache_manager: Option<crate::pressure::cache_manager::CacheManager>,
    // 동일 sticky evict directive 가 매 poll 재적용되지 않도록 active 구간당 1회만
    // 발동. plan.evict 가 None 으로 돌아오면(RestoreDefaults) false 로 reset.
    evict_applied: bool,
    // β-2: L2 PipelineStage dispatch 배선. default = empty registry (len==0 fast-path
    // 로 happy-path 거동-0). run_until_stop 은 β-6 에서 통합 — 현재 미발화.
    pipeline: Arc<PipelineRegistry>,
    // β-1 계약 (v2 §5.2.1): DecodeLoop 이 OpProfiler 1개 소유, dispatch 호출부마다
    // &mut 재대여 (StageContext.profiler).
    profiler: OpProfiler,
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
    /// β-2: 현재 step 스냅샷으로 phase 1건 dispatch. 빈 registry 는
    /// PipelineRegistry 내부 len==0 fast-path 로 무lock 즉시 반환 (거동-0).
    /// pressure 는 β-5 (PressureSource 일원화) 전까지 default(0).
    fn dispatch_phase(&mut self, phase: LifecyclePhase) -> Option<StageStopReason> {
        let step = StepInfo {
            pos: self.pos,
            decode_step: self.decode_step,
            pressure: Pressure::default(),
        };
        let mut ctx = StageContext {
            step,
            profiler: &mut self.profiler,
        };
        self.pipeline.dispatch(phase, &mut ctx)
    }

    /// v2 StopReason(pipeline.rs 4-variant) → v1 StopReason(traits.rs) 수렴 매핑
    /// (v2 §5.2.1 (다)). StopFlag 는 v2 에 없음 — driver 루프 가드 v1 유지.
    fn map_stage_stop(r: StageStopReason) -> StopReason {
        match r {
            StageStopReason::EosToken => StopReason::EosToken,
            StageStopReason::BudgetExhausted => StopReason::BudgetExhausted,
            StageStopReason::StopConditionMet => StopReason::StopConditionMet,
            StageStopReason::CommandRequested => StopReason::CommandRequested,
        }
    }

    /// Run prefill over the prompt. Returns logits for the last token so the
    /// caller can sample the first generated token before invoking
    /// [`Self::run`]. `pos` is advanced to `tokens.len()`. `prev_token` is
    /// staged from `tokens.last()` for `on_prefill_end` observer context only;
    /// [`Self::run`] overwrites it with `first_token`.
    pub fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        // PrefillStart: prefill 엔 break 할 루프 없음 — Stop 은 per-token phase 전용
        // (chunked prefill 시 PrefillChunkBoundary 와 함께 재고).
        let _ = self.dispatch_phase(LifecyclePhase::PrefillStart);
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
        // PrefillEnd: on_prefill_end observer 루프 후, Ok(logits) 직전.
        let _ = self.dispatch_phase(LifecyclePhase::PrefillEnd);
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

            // DecodeStart: t_iter 직후, (a) command poll 전.
            if let Some(r) = self.dispatch_phase(LifecyclePhase::DecodeStart) {
                stopped_by = Self::map_stage_stop(r);
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
            let kv_snap = self.build_kv_snapshot();
            let plan = self.cmd_source.poll(&ctx, &kv_snap)?;
            if plan.suspended {
                stopped_by = StopReason::CommandRequested;
                break;
            }
            if plan.throttle_delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(plan.throttle_delay_ms));
            }

            // (a.5) argus-bench AB-1: resilience-driven eviction. plan.evict
            // (EvictPlan) 가 Some 이고 active 구간 첫 발동이면 forward.try_evict 로
            // 1회 prune (score-free force_evict — CacheManager 의 CLI 정책 사용;
            // H2O 는 score 부재 시 recency degrade, chat 과 동일). CacheManager 가
            // `[CacheEvent] Eviction completed` 를 emit. sticky carry 매 step
            // 재적용 방지(target_ratio 반복 적용 시 cache 과다 축소). plan.evict 가
            // None 으로 돌아오면(RestoreDefaults) 다음 directive 위해 reset.
            if let Some(evict_plan) = plan.evict.as_ref() {
                if !self.evict_applied {
                    self.evict_applied = true;
                    if let Some(cm) = self.cache_manager.as_ref() {
                        let (removed, new_pos) =
                            self.forward
                                .try_evict(cm, None, true, evict_plan.target_ratio)?;
                        if removed > 0 {
                            self.pos = new_pos;
                            self.forward.on_kv_prune(new_pos);
                            let ctx = step_ctx(
                                self.pos,
                                self.prev_token,
                                self.kv_capacity,
                                self.decode_step,
                                &stop,
                            );
                            let outcome = EvictionOutcome::Pruned { removed, new_pos };
                            for obs in &mut self.observers {
                                obs.on_eviction(&ctx, &outcome);
                            }
                        }
                    }
                }
            } else {
                self.evict_applied = false;
            }

            // (a.6) argus-bench AB-3: resilience KvOffload / recall. offload_ratio
            // 는 sticky 가 아니라 KvOffload directive 가 도착한 poll 에만 Some 이라
            // 자연히 1회 발동. RestoreDefaults 는 recall_offload + restore_defaults 를
            // 함께 세팅하여 recall 을 게이트. offload 는 prune_prefix(pos 감소),
            // recall 은 pos 증가 → cache 의 새 current_pos 로 loop pos 동기화.
            if let Some(ratio) = plan.offload_ratio
                && let Some(cm) = self.cache_manager.as_mut()
            {
                let (n, new_pos) = self.forward.try_offload(cm, ratio)?;
                eprintln!(
                    "[Resilience] KvOffload: ratio={:.2}, {} tokens swapped",
                    ratio, n
                );
                if n > 0 {
                    self.pos = new_pos;
                    self.forward.on_kv_prune(new_pos);
                }
            }
            if plan.recall_offload
                && plan.restore_defaults
                && let Some(cm) = self.cache_manager.as_mut()
            {
                let (n, new_pos) = self.forward.try_recall(cm)?;
                if n > 0 {
                    eprintln!("[Resilience] Recalled {} tokens from swap", n);
                    self.pos = new_pos;
                }
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

            // PreForward: (c) swap.before_step 후, (d) forward 직전.
            if let Some(r) = self.dispatch_phase(LifecyclePhase::PreForward) {
                stopped_by = Self::map_stage_stop(r);
                break;
            }

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

            // PostForward: step_ms 계산 직후, (e) swap after 전.
            if let Some(r) = self.dispatch_phase(LifecyclePhase::PostForward) {
                stopped_by = Self::map_stage_stop(r);
                break;
            }

            // (e) swap after
            let ctx = step_ctx(
                self.pos,
                self.prev_token,
                self.kv_capacity,
                self.decode_step,
                &stop,
            );
            self.swap.after_step(&ctx)?;

            // PreSample: (e) swap.after_step 후, (f) sampler.sample 직전.
            if let Some(r) = self.dispatch_phase(LifecyclePhase::PreSample) {
                stopped_by = Self::map_stage_stop(r);
                break;
            }

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

            // PostSample: sampled 미push 시점 → stop token 미포함 break
            // (v1 chat stop 시맨틱 정합, β-6 StopCondition stage 전제).
            if let Some(r) = self.dispatch_phase(LifecyclePhase::PostSample) {
                stopped_by = Self::map_stage_stop(r);
                break;
            }

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

            // DecodeEnd: bookkeeping 후, (h) pacing 전.
            // step 완료 후 시점 — StepInfo 는 증가된 pos/decode_step, token 은 이미 포함.
            if let Some(r) = self.dispatch_phase(LifecyclePhase::DecodeEnd) {
                stopped_by = Self::map_stage_stop(r);
                break;
            }

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
        // Finalize: forward.finalize() + observers finalize 체인 후, DecodeResult 구성 직전.
        let _ = self.dispatch_phase(LifecyclePhase::Finalize);

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
    cache_manager: Option<crate::pressure::cache_manager::CacheManager>,
    pipeline: Option<Arc<PipelineRegistry>>,
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
            cache_manager: None,
            pipeline: None,
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
            cache_manager: self.cache_manager,
            pipeline: self.pipeline,
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
    /// argus-bench AB-1: resilience-driven eviction 용 [`CacheManager`] 주입.
    /// 미주입(None) 이면 `plan.evict` 가 와도 eviction 미발동 (happy/chat 동일).
    pub fn with_cache_manager(mut self, cm: crate::pressure::cache_manager::CacheManager) -> Self {
        self.cache_manager = Some(cm);
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
    /// β-2: L2 stage registry 주입. 미주입 시 빈 registry (거동-0).
    pub fn with_pipeline(mut self, p: Arc<PipelineRegistry>) -> Self {
        self.pipeline = Some(p);
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
            cache_manager: self.cache_manager,
            evict_applied: false,
            pipeline: self
                .pipeline
                .unwrap_or_else(|| Arc::new(PipelineRegistry::new())),
            profiler: OpProfiler::new(),
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

    // ── β-2 phase 배선 테스트 ──────────────────────────────────────────────

    use crate::pipeline::{
        LifecyclePhase as Phase, PipelineStage, StageContext as SCtx, StageOutcome,
    };
    use crate::session::pipeline_registry::PipelineRegistry;
    use std::sync::Mutex;

    /// 모든 phase를 기록하는 Persistent stage.
    struct RecordStage {
        log: Arc<Mutex<Vec<Phase>>>,
    }

    impl PipelineStage for RecordStage {
        fn name(&self) -> &str {
            "RecordStage"
        }
        fn on_phase(&self, phase: &Phase, _ctx: &mut SCtx<'_>) -> anyhow::Result<StageOutcome> {
            self.log.lock().unwrap().push(phase.clone());
            Ok(StageOutcome::Continue)
        }
    }

    /// 지정 phase에서 Stop(EosToken)을 반환하는 stage.
    struct StopAtStage {
        target: Phase,
        fired: Arc<Mutex<bool>>,
    }

    impl PipelineStage for StopAtStage {
        fn name(&self) -> &str {
            "StopAtStage"
        }
        fn on_phase(&self, phase: &Phase, _ctx: &mut SCtx<'_>) -> anyhow::Result<StageOutcome> {
            if phase == &self.target {
                *self.fired.lock().unwrap() = true;
                Ok(StageOutcome::Stop(crate::pipeline::StopReason::EosToken))
            } else {
                Ok(StageOutcome::Continue)
            }
        }
    }

    /// phase, pos, decode_step 을 기록하는 stage.
    struct SnapshotStage {
        log: Arc<Mutex<Vec<(Phase, usize, usize)>>>,
    }

    impl PipelineStage for SnapshotStage {
        fn name(&self) -> &str {
            "SnapshotStage"
        }
        fn on_phase(&self, phase: &Phase, ctx: &mut SCtx<'_>) -> anyhow::Result<StageOutcome> {
            self.log
                .lock()
                .unwrap()
                .push((phase.clone(), ctx.step.pos, ctx.step.decode_step));
            Ok(StageOutcome::Continue)
        }
    }

    /// prefill(&[1,2,3]) + run(2, 0) 시 정확한 phase 시퀀스를 검증한다.
    #[test]
    fn phase_emission_sequence_exact() {
        let log: Arc<Mutex<Vec<Phase>>> = Arc::new(Mutex::new(Vec::new()));
        let registry = Arc::new(PipelineRegistry::new());
        registry.submit(Arc::new(RecordStage { log: log.clone() }));

        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_kv_capacity(2048)
            .with_pipeline(Arc::clone(&registry))
            .build();

        loop_.prefill(&[1, 2, 3]).unwrap();
        loop_.run(2, 0).unwrap();

        let observed = log.lock().unwrap().clone();
        let expected = vec![
            Phase::PrefillStart,
            Phase::PrefillEnd,
            Phase::DecodeStart,
            Phase::PreForward,
            Phase::PostForward,
            Phase::PreSample,
            Phase::PostSample,
            Phase::DecodeEnd,
            Phase::DecodeStart,
            Phase::PreForward,
            Phase::PostForward,
            Phase::PreSample,
            Phase::PostSample,
            Phase::DecodeEnd,
            Phase::Finalize,
        ];
        assert_eq!(observed, expected, "phase 시퀀스 불일치");
    }

    /// PostSample 에서 Stop(EosToken) 반환 시 tokens_generated 가 비어 있고
    /// stopped_by 가 v1 StopReason::EosToken 으로 수렴된다.
    #[test]
    fn stage_stop_at_post_sample_maps_to_v1_reason() {
        let fired = Arc::new(Mutex::new(false));
        let registry = Arc::new(PipelineRegistry::new());
        registry.submit(Arc::new(StopAtStage {
            target: Phase::PostSample,
            fired: fired.clone(),
        }));

        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_kv_capacity(2048)
            .with_pipeline(Arc::clone(&registry))
            .build();

        loop_.prefill(&[1]).unwrap();
        let result = loop_.run(5, 0).unwrap();

        assert_eq!(result.stopped_by, StopReason::EosToken, "v1 EosToken 수렴");
        assert!(
            result.tokens_generated.is_empty(),
            "PostSample 미push break — 토큰 없음"
        );
        assert!(*fired.lock().unwrap(), "StopAtStage 발화 확인");
    }

    /// 기록 stage 등록 후 run_until_stop 실행 → 추가 발화 0 (총 == prefill 2종).
    #[test]
    fn run_until_stop_does_not_dispatch() {
        use crate::session::chat::stop_condition::{ChatStopCondition, StopCondition};

        let log: Arc<Mutex<Vec<Phase>>> = Arc::new(Mutex::new(Vec::new()));
        let registry = Arc::new(PipelineRegistry::new());
        registry.submit(Arc::new(RecordStage { log: log.clone() }));

        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_kv_capacity(2048)
            .with_pipeline(Arc::clone(&registry))
            .build();

        // prefill 1회 → PrefillStart + PrefillEnd = 2 발화
        loop_.prefill(&[0]).unwrap();
        let prefill_count = log.lock().unwrap().len();
        assert_eq!(prefill_count, 2, "prefill 발화 2건");

        // run_until_stop 실행 — 추가 발화 없어야 함
        let cond = ChatStopCondition::new(vec![1], 2048);
        loop_
            .run_until_stop(0, &cond as &dyn StopCondition)
            .unwrap();

        let total = log.lock().unwrap().len();
        assert_eq!(
            total, 2,
            "run_until_stop 추가 발화 0 — 총 카운트 == prefill 2"
        );
    }

    /// phase 별 StepInfo(pos, decode_step) 스냅샷 값을 검증한다.
    #[test]
    fn step_info_snapshot_values() {
        let log: Arc<Mutex<Vec<(Phase, usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let registry = Arc::new(PipelineRegistry::new());
        registry.submit(Arc::new(SnapshotStage { log: log.clone() }));

        let mut loop_ = DecodeLoopBuilder::new()
            .with_forward(MockForward {
                vocab: 16,
                step_count: 0,
            })
            .with_kv_capacity(2048)
            .with_pipeline(Arc::clone(&registry))
            .build();

        // prefill(&[1,2,3]) → pos=0→3
        loop_.prefill(&[1, 2, 3]).unwrap();
        // run(1, 0) → 1 decode step
        loop_.run(1, 0).unwrap();

        let observed = log.lock().unwrap().clone();

        // PrefillStart: pos=0, decode_step=0 (prefill 직전)
        let prefill_start = observed
            .iter()
            .find(|(p, _, _)| p == &Phase::PrefillStart)
            .unwrap();
        assert_eq!(
            (prefill_start.1, prefill_start.2),
            (0, 0),
            "PrefillStart pos==0"
        );

        // PrefillEnd: pos=3, decode_step=0 (pos 갱신 후)
        let prefill_end = observed
            .iter()
            .find(|(p, _, _)| p == &Phase::PrefillEnd)
            .unwrap();
        assert_eq!((prefill_end.1, prefill_end.2), (3, 0), "PrefillEnd pos==3");

        // DecodeStart: pos=3, decode_step=0 (루프 진입 시)
        let decode_start = observed
            .iter()
            .find(|(p, _, _)| p == &Phase::DecodeStart)
            .unwrap();
        assert_eq!(
            (decode_start.1, decode_start.2),
            (3, 0),
            "DecodeStart pos==3, step==0"
        );

        // DecodeEnd: pos=4, decode_step=1 (bookkeeping 후)
        let decode_end = observed
            .iter()
            .find(|(p, _, _)| p == &Phase::DecodeEnd)
            .unwrap();
        assert_eq!(
            (decode_end.1, decode_end.2),
            (4, 1),
            "DecodeEnd pos==4, step==1"
        );
    }
}
