# Inference Pipeline — DecodeLoop SOLID 분해 + 빌더 설계

> spec/01-architecture.md §3.8 (SYS-100, SYS-105) + `INV-LAYER-005/006/007`의 구현 설계. `bin/generate.rs` 13,017 LOC 중 `main()` 7,051 LOC을 6개 trait 추상화 + typestate builder로 분해한다. 본 문서는 코드가 없는 단계(Migration Step 2-2 이전)의 **설계 결정 단일 진실 원본**이다.

본 문서의 trait API와 빌더 시그니처는 Migration Step 2-2에서 `engine/src/session/decode_loop.rs`로 1:1 이식되어야 한다. 시그니처 변경은 본 문서의 갱신을 동반한다.

---

## 1. 책임 분해 (decode loop의 6개 변경축)

`main()` 안에서 매 토큰 루프가 수행하는 작업을 *변경 이유*(reason to change, SRP) 단위로 분리한다. 메인 세션 분석에서 식별된 6개 책임은 다음과 같다.

| # | 책임 | 변경 이유 (예시) | 현재 `main()` 보유 변수 (인용) |
|---|------|------------------|-------------------------------|
| 1 | **Forward 실행** | backend 교체 (CPU / OpenCL / CUDA / QNN), 변형 forward path (kivi / offload) | `backend`, `model`, `kv_caches`, `workspace`, `gpu_buffers`, `logits` |
| 2 | **Eviction trigger** | 정책 교체 (Sliding / H2O / D2O / None), SWIFT skip | `cache_manager`, `score_accumulator`, `skip_config`, `last_skip_ratio` |
| 3 | **Weight swap dispatch** | 알고리즘 교체 (sync / async / dynamic-K / phase-aware / probing-K) | 8개: `incremental_force_swap_plan`, `manager_swap_report_pending`, `ready_weight_swap_report`, `intra_forward_swap_hook`, `phase_aware_swap_dispatcher`, `async_swap_dispatcher`, `dynamic_k_controller`, `probing_k_controller` |
| 4 | **Command source** | 명령 source 교체 (Manager IPC / ScheduleFile / Stdin REPL) | `manager_client`, `experiment_schedule`, `command_executor` |
| 5 | **Token sampling** | 전략 교체 (greedy / temperature / top-k / top-p / spec-decoding seed) | `sampling_config` |
| 6 | **Observation** | 메트릭 추가/교체 (profiler / experiment writer / tbt log / system sampler) | `profiler`, `experiment_writer`, `tbt_log_writer`, `system_sampler`, `forward_ms_values`, `tbt_values` |

**파생 결정 (SRP 위반 회피)**: 6 책임 중 두 개를 한 trait에 합치면 *서로 다른 이유로 변경* 됨을 보여야 했으나, 모두 독립 변경 사례가 존재한다 — e.g. Forward는 그대로 두고 Eviction만 H2O→Sliding으로 교체한 실험이 다수 (Round 14/15), Swap만 sync→async로 교체한 ablation (Phase 6.5).

---

## 2. Trait API 정의

각 trait는 hot path 호출 빈도, mutability, lifetime 비용을 함께 명시한다. `&mut dyn` 가정. generic monomorphization은 §7에서 별도 검토.

### 2.1 `StepCtx` — 공유 read-only context

```rust
/// Decode step 단위의 read-only context. DecodeLoop가 매 step 시작에서 빌드.
/// trait들은 mutate 불가. mutable state는 각 trait 구현체 내부 또는
/// DecodeLoop 자신의 카운터에만 보관.
pub struct StepCtx<'a> {
    pub pos: usize,              // 현재 KV pos (decode 직전)
    pub prev_token: u32,         // 직전 sample된 token (prefill 종료 직후엔 마지막 prompt token)
    pub kv_capacity: usize,      // 현재 cache의 max_seq_len
    pub decode_step: usize,      // 0부터 시작하는 decode iteration index
    pub stop_requested: &'a AtomicBool, // signal handler가 set
}
```

- **누가 생성?** `DecodeLoop::run` 루프 헤더가 매 iteration마다 stack에 빌드.
- **누가 mutate?** 아무도 안 함 (`&'a` shared reference). 카운터(`pos`, `decode_step`)는 다음 iteration 시작 시 새 값으로 재구성.
- **수명 전략**: `'a`는 단일 step scope. trait 메서드 안에서 `StepCtx`를 저장하지 못하도록 lifetime으로 막는다.

### 2.2 `Forward` (필수)

```rust
pub trait Forward {
    /// Prefill 페이즈: prompt 전체를 한번에 처리. 마지막 위치 logits 반환.
    fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>>;

    /// Decode 1 step. 호출 후 pos는 +1 효과를 가진다 (Forward 내부 KV 갱신).
    fn step(&mut self, ctx: &StepCtx, token: u32) -> anyhow::Result<Vec<f32>>;

    /// 종료 직전 호출 (eos 또는 budget 소진). 마지막 logits/score flush 등.
    fn finalize(&mut self) -> anyhow::Result<()> { Ok(()) }

    /// KV pos 동기화 — eviction stage가 KV에서 N토큰 제거 후 호출.
    fn on_kv_prune(&mut self, new_pos: usize);
}
```

- 호출 빈도: prefill 1회, step N회 (decode 토큰 수).
- 흡수할 변수(책임 #1 전부): `backend`, `model`, `kv_caches`, `workspace`, `gpu_buffers`, `logits`.
- 구현체 예시: `ModelForward` (표준), `KiviForward` (KIVI 2bit KV quant), `OffloadForward` (per-layer prefetch).
- No-op default: **없음** — 필수 컴포넌트. typestate로 빌더 강제 (INV-LAYER-007).

### 2.3 `EvictionStage` (선택)

```rust
pub enum EvictionOutcome {
    None,
    Pruned { removed: usize, new_pos: usize },
    Skipped { reason: SkipReason },  // SWIFT skip 트리거 등
}

pub trait EvictionStage {
    /// Forward::step 직전 호출. 압박 감지 + policy 적용.
    fn before_step(&mut self, ctx: &StepCtx) -> anyhow::Result<EvictionOutcome>;

    /// 옵션: turn 종료 또는 capacity 보장. chat REPL용.
    fn ensure_capacity(
        &mut self,
        ctx: &StepCtx,
        additional: usize,
    ) -> anyhow::Result<()> { Ok(()) }
}
```

- 호출 빈도: step당 1회 (`before_step`). chat 모드에선 `ensure_capacity` 추가 호출.
- 흡수할 변수(책임 #2): `cache_manager`, `score_accumulator`, `skip_config`, `last_skip_ratio`.
- 구현체 예시: `CacheManagerStage` (현 `CacheManager` 래핑), `NoEvictionStage` (no-op).
- No-op default: `NoEvictionStage` — `EvictionOutcome::None` 반환.

### 2.4 `SwapStage` (선택)

```rust
pub trait SwapStage {
    /// Forward::step 직전: prefetch/load 트리거.
    fn before_step(&mut self, ctx: &StepCtx) -> anyhow::Result<()>;

    /// Forward::step 직후: commit/release 트리거.
    fn after_step(&mut self, ctx: &StepCtx) -> anyhow::Result<()>;

    /// Manager 측 SwapReport 가용 여부 (heartbeat 동봉용).
    fn pending_report(&mut self) -> Option<llm_shared::WeightSwapReport> { None }
}
```

- 호출 빈도: step당 2회. hot path지만 default 구현은 zero-cost (vtable + empty fn body).
- 흡수할 변수(책임 #3 전부 8개): `incremental_force_swap_plan`, `manager_swap_report_pending`, `ready_weight_swap_report`, `intra_forward_swap_hook`, `phase_aware_swap_dispatcher`, `async_swap_dispatcher`, `dynamic_k_controller`, `probing_k_controller`.
- 구현체 예시: `SyncSwapStage`, `AsyncSwapStage`, `PhaseAwareSwapStage`, `ProbingKSwapStage`, `NoSwapStage`.
- No-op default: `NoSwapStage` — 두 메서드 모두 `Ok(())`.

### 2.5 `CommandSource` (선택)

```rust
pub trait CommandSource {
    /// 1 step 내에 들어온 command (있다면) 반환. Non-blocking.
    fn poll(&mut self, ctx: &StepCtx) -> anyhow::Result<Option<EngineCommand>>;
}
```

- 호출 빈도: step당 1회.
- 흡수할 변수(책임 #4): `manager_client`, `experiment_schedule`, `command_executor`.
- 구현체 예시: `ManagerCmdSource` (IPC), `ScheduleCmdSource` (timed schedule), `StdinCmdSource` (chat REPL), `NoCommandSource`.
- No-op default: `NoCommandSource` — 항상 `Ok(None)`.

### 2.6 `TokenSampler` (필수, default 제공)

```rust
pub trait TokenSampler {
    fn sample(&mut self, ctx: &StepCtx, logits: &[f32]) -> u32;
}
```

- 호출 빈도: step당 1회.
- 흡수할 변수(책임 #5): `sampling_config`.
- 구현체 예시: `GreedySampler`, `TempSampler`, `TopKSampler`, `TopPSampler`, `MixedSampler`.
- Default: `GreedySampler` (단순 argmax) — `Forward`와 달리 typestate 강제 안 함. 빌더에서 미지정 시 자동 적용 (compile time).

### 2.7 `DecodeObserver` (선택, multi)

```rust
pub trait DecodeObserver {
    fn on_prefill_end(&mut self, ctx: &StepCtx, last_logits: &[f32]) {}
    fn on_step_end(&mut self, ctx: &StepCtx, sampled: u32, step_ms: f64) {}
    fn on_eviction(&mut self, ctx: &StepCtx, outcome: &EvictionOutcome) {}
    fn finalize(&mut self) -> anyhow::Result<()> { Ok(()) }
}
```

- 호출 빈도: step당 1~3회 (eviction outcome이 None 외인 경우만 `on_eviction` 호출).
- 흡수할 변수(책임 #6): `profiler`, `experiment_writer`, `tbt_log_writer`, `system_sampler`, `forward_ms_values`, `tbt_values`.
- 구현체 예시: `ProfilerObs` (현 `Profiler`), `ExperimentWriterObs`, `TbtLogObs`, `SystemSamplerObs`, `EventSinkAdapterObs` (§6 참조), `NoOpObserver`.
- No-op default: `NoOpObserver`.
- **다중 등록**: builder에 `.add_observer(...)`를 반복 호출하여 `Vec<Box<dyn DecodeObserver>>`로 누적. `DecodeLoop`는 매 step 끝에 전체 vec를 순회.

---

## 3. `DecodeLoop` 구조

### 3.1 필드와 메서드

```rust
pub struct DecodeLoop {
    forward: Box<dyn Forward>,
    eviction: Box<dyn EvictionStage>,
    swap: Box<dyn SwapStage>,
    cmd_source: Box<dyn CommandSource>,
    sampler: Box<dyn TokenSampler>,
    observers: Vec<Box<dyn DecodeObserver>>,

    // 내부 카운터 (변경축 없음 — DecodeLoop 자신의 SRP)
    pos: usize,
    decode_step: usize,
    stop_flag: Arc<AtomicBool>,
}

pub struct DecodeResult {
    pub tokens: Vec<u32>,
    pub stopped_by: StopReason,
}

impl DecodeLoop {
    pub fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let logits = self.forward.prefill(tokens)?;
        self.pos = tokens.len();
        let ctx = StepCtx { pos: self.pos, prev_token: *tokens.last().unwrap(),
                            kv_capacity: 0, decode_step: 0, stop_requested: &self.stop_flag };
        for obs in &mut self.observers { obs.on_prefill_end(&ctx, &logits); }
        Ok(logits)
    }

    pub fn run(&mut self, budget: usize) -> anyhow::Result<DecodeResult> {
        let mut out = Vec::with_capacity(budget);
        let mut last_logits = vec![]; // 또는 prefill에서 전달받음
        let mut prev_token = 0u32;

        for step in 0..budget {
            if self.stop_flag.load(Ordering::Relaxed) { return Ok(DecodeResult { tokens: out, stopped_by: StopReason::Signal }); }

            let ctx = StepCtx {
                pos: self.pos, prev_token,
                kv_capacity: 0, decode_step: step, stop_requested: &self.stop_flag,
            };

            // (a) command poll
            if let Some(cmd) = self.cmd_source.poll(&ctx)? { self.handle_command(cmd)?; }

            // (b) eviction
            let evict = self.eviction.before_step(&ctx)?;
            if let EvictionOutcome::Pruned { new_pos, .. } = evict {
                self.forward.on_kv_prune(new_pos);
                self.pos = new_pos;
            }
            for obs in &mut self.observers { obs.on_eviction(&ctx, &evict); }

            // (c) swap before
            self.swap.before_step(&ctx)?;

            // (d) forward
            let t0 = std::time::Instant::now();
            let logits = self.forward.step(&ctx, prev_token)?;
            let step_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // (e) swap after
            self.swap.after_step(&ctx)?;

            // (f) sample
            let sampled = self.sampler.sample(&ctx, &logits);
            out.push(sampled);
            prev_token = sampled;
            self.pos += 1;
            last_logits = logits;

            // (g) observers
            for obs in &mut self.observers { obs.on_step_end(&ctx, sampled, step_ms); }

            if self.is_stop_token(sampled) { return Ok(DecodeResult { tokens: out, stopped_by: StopReason::Eos }); }
        }
        Ok(DecodeResult { tokens: out, stopped_by: StopReason::Budget })
    }

    pub fn finalize(mut self) -> anyhow::Result<()> {
        self.forward.finalize()?;
        for mut obs in self.observers { obs.finalize()?; }
        Ok(())
    }

    fn handle_command(&mut self, cmd: EngineCommand) -> anyhow::Result<()> { /* eviction.target_ratio, swap.cancel 등 dispatch */ Ok(()) }
    fn is_stop_token(&self, t: u32) -> bool { /* eos / stop_ids 매칭 */ false }
}
```

### 3.2 Lifetime 전략 비교

| 옵션 | 시그니처 | 장점 | 단점 | 채택 |
|------|---------|------|------|------|
| (a) `<'a>` 단순 | `struct DecodeLoop<'a> { forward: &'a mut dyn Forward, ... }` | zero allocation. trait object 비용 0 추가. | builder가 모든 구현체의 lifetime을 호출자로 lift해야 함. chat REPL/IPC 비동기 분기 시 lifetime hell. | X |
| (b) **owned `Box<dyn>`** | `struct DecodeLoop { forward: Box<dyn Forward>, ... }` | builder가 owned로 받음 → caller-side lifetime free. trait object 1회 alloc(start-up). | Box vtable indirection — §7에서 별도 측정. | **O** |
| (c) `Arc<dyn>` | `struct DecodeLoop { forward: Arc<dyn Forward + Send + Sync>, ... }` | observer를 background thread에 공유 가능. | mutate 시 `Arc<Mutex<...>>` 필요 → 매 step lock. hot path에 부담. | X (observer만 필요 시 별도 검토) |

**채택: (b)**. 매 step 5+N개 vtable 호출은 ARM A-class out-of-order에서 ~50-100 ns 수준이며, Adreno 14 ms TBT 대비 0.5% 미만이다 (§7 측정 게이트로 검증).

---

## 4. 빌더 패턴 설계 (typestate)

### 4.1 typestate marker

```rust
pub struct NoForward; pub struct HasForward(Box<dyn Forward>);

pub struct DecodeLoopBuilder<F = NoForward> {
    forward: F,
    eviction: Option<Box<dyn EvictionStage>>,
    swap: Option<Box<dyn SwapStage>>,
    cmd_source: Option<Box<dyn CommandSource>>,
    sampler: Option<Box<dyn TokenSampler>>,
    observers: Vec<Box<dyn DecodeObserver>>,
    stop_flag: Option<Arc<AtomicBool>>,
}
```

### 4.2 빌더 API

```rust
impl DecodeLoopBuilder<NoForward> {
    pub fn new() -> Self { /* all None / empty */ unimplemented!() }

    pub fn with_forward<T: Forward + 'static>(self, fwd: T) -> DecodeLoopBuilder<HasForward> {
        DecodeLoopBuilder {
            forward: HasForward(Box::new(fwd)),
            eviction: self.eviction, swap: self.swap, cmd_source: self.cmd_source,
            sampler: self.sampler, observers: self.observers, stop_flag: self.stop_flag,
        }
    }
}

impl<F> DecodeLoopBuilder<F> {
    pub fn with_eviction<T: EvictionStage + 'static>(mut self, e: T) -> Self { self.eviction = Some(Box::new(e)); self }
    pub fn with_swap<T: SwapStage + 'static>(mut self, s: T) -> Self { self.swap = Some(Box::new(s)); self }
    pub fn with_cmd_source<T: CommandSource + 'static>(mut self, c: T) -> Self { self.cmd_source = Some(Box::new(c)); self }
    pub fn with_sampler<T: TokenSampler + 'static>(mut self, s: T) -> Self { self.sampler = Some(Box::new(s)); self }
    pub fn add_observer<T: DecodeObserver + 'static>(mut self, o: T) -> Self {
        self.observers.push(Box::new(o)); self
    }
    pub fn with_stop_flag(mut self, f: Arc<AtomicBool>) -> Self { self.stop_flag = Some(f); self }
}

// **핵심**: build()는 HasForward 상태에서만 가능 — INV-LAYER-007 컴파일 강제.
impl DecodeLoopBuilder<HasForward> {
    pub fn build(self) -> DecodeLoop {
        DecodeLoop {
            forward: self.forward.0,
            eviction: self.eviction.unwrap_or_else(|| Box::new(NoEvictionStage)),
            swap: self.swap.unwrap_or_else(|| Box::new(NoSwapStage)),
            cmd_source: self.cmd_source.unwrap_or_else(|| Box::new(NoCommandSource)),
            sampler: self.sampler.unwrap_or_else(|| Box::new(GreedySampler::default())),
            observers: self.observers,
            pos: 0, decode_step: 0,
            stop_flag: self.stop_flag.unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
        }
    }
}
```

### 4.3 사용 예시 — production decode (`generate.rs::main()`)

```rust
fn run_production(args: &Args, ctx: SessionInitCtx) -> anyhow::Result<()> {
    let loop_ = DecodeLoopBuilder::new()
        .with_forward(ModelForward::from_init(ctx.backend, ctx.model, ctx.kv_caches, ctx.workspace)?)
        .with_eviction(CacheManagerStage::new(ctx.cache_manager, ctx.score_accumulator, ctx.skip_config))
        .with_swap(build_swap_stage(args)?)             // sync / async / phase / probing
        .with_cmd_source(ManagerCmdSource::connect(args.manager_socket.as_deref())?)
        .with_sampler(SamplingConfig::from(args).into_sampler())
        .add_observer(ProfilerObs::new(args.profile))
        .add_observer(ExperimentWriterObs::open(&args.experiment_out)?)
        .add_observer(TbtLogObs::open(&args.tbt_log)?)
        .with_stop_flag(install_sigint_handler())
        .build();

    let mut loop_ = loop_;
    let _ = loop_.prefill(&prompt_tokens)?;
    let result = loop_.run(args.max_new_tokens)?;
    loop_.finalize()?;
    println!("{}", decode_to_string(&result.tokens));
    Ok(())
}
```

### 4.4 typestate vs runtime check — 결정 근거

| 옵션 | 시그니처 안전성 | 빌더 코드량 | 채택 |
|------|----------------|------------|------|
| (a) typestate (`HasForward` marker) | `build()` 누락 시 컴파일 실패 | marker struct + 2개 impl block 추가 (~30 LOC) | **O** |
| (b) runtime check (`build() -> Result<...>`) | 빌드 시점 panic / unwrap | `Option<Box<dyn>>` + unwrap_or panic | X |

채택: (a). 필수 컴포넌트는 1개(`Forward`)뿐이라 marker 1쌍으로 충분하다. `Sampler`도 필수지만 `GreedySampler` default가 있어 typestate 불필요. 빌더 코드량 부담은 INV-LAYER-007의 컴파일 강제 가치 대비 미미하다.

### 4.5 No-op default 위치

신규 `engine/src/session/defaults.rs`에 5개 default 구현체를 모은다 — `NoEvictionStage`, `NoSwapStage`, `NoCommandSource`, `NoOpObserver`, `GreedySampler`. 각 구현체는 lib visibility (`pub(crate)` 또는 `pub`)로 노출하여 외부 사용자가 부분 override 시 explicit하게 import 가능하다.

---

## 5. `main()` 조립자 변환 예시 (~80 LOC)

현 `main()` 7,051 LOC은 Migration Step 2-4 후 다음 4단계로 압축된다.

```rust
fn main() -> anyhow::Result<()> {
    // ─── Stage 1: CLI parse + log init ──────────────────────────
    let args = Args::parse();
    init_logging(&args)?;
    if args.dump_config { return dump_cli_and_exit(&args); }

    // ─── Stage 2: backend + model + ctx init (session::init 헬퍼) ──
    let mut ctx = SessionInitCtx::build(&args)?;  // backend, model, kv_caches, workspace, cache_manager
                                                  // tokenizer, sampling_cfg, score_accumulator,
                                                  // manager_client (option), experiment_schedule (option)
                                                  // 모두 여기서 owned로 묶임

    // ─── Stage 3: DecodeLoop 조립 (chat / generate / kivi / offload 분기) ──
    let mut loop_ = match (args.chat, args.use_kivi, args.offload_path.is_some()) {
        (false, false, false) => build_standard_loop(&args, &mut ctx)?,
        (false, true,  false) => build_kivi_loop(&args, &mut ctx)?,
        (false, false, true ) => build_offload_loop(&args, &mut ctx)?,
        (true,  _,     _    ) => return run_chat_repl_v2(&args, ctx),    // chat REPL은 §10 위험 분석 참조
        _ => anyhow::bail!("incompatible mode combination"),
    };

    // ─── Stage 4: prefill + decode + finalize ────────────────────
    let prompt_tokens = ctx.tokenizer.encode_prompt(&args.prompt, &args)?;
    let _ = loop_.prefill(&prompt_tokens)?;

    let stopped = loop_.run(args.max_new_tokens)?;
    loop_.finalize()?;

    print_result(&ctx.tokenizer, &stopped);
    Ok(())
}

fn build_standard_loop(args: &Args, ctx: &mut SessionInitCtx) -> anyhow::Result<DecodeLoop> {
    let mut b = DecodeLoopBuilder::new()
        .with_forward(ModelForward::from_ctx(ctx)?)
        .with_eviction(CacheManagerStage::from_ctx(ctx))
        .with_sampler(ctx.sampling_cfg.clone().into_sampler())
        .with_stop_flag(install_sigint_handler());

    if let Some(swap) = build_swap_stage(args, ctx)? { b = b.with_swap(swap); }
    if let Some(src) = build_command_source(args, ctx)? { b = b.with_cmd_source(src); }

    if let Some(p) = ProfilerObs::maybe(args)? { b = b.add_observer(p); }
    if let Some(w) = ExperimentWriterObs::maybe(args)? { b = b.add_observer(w); }
    if let Some(t) = TbtLogObs::maybe(args)? { b = b.add_observer(t); }
    if let Some(s) = SystemSamplerObs::maybe(args)? { b = b.add_observer(s); }

    Ok(b.build())
}
```

이 변환 후 `main()` 본체 + 4개 `build_*_loop` 헬퍼 합계 < 400 LOC 목표. 나머지 코드는 `session::init`, `inference::ModelForward`, `pressure::CacheManagerStage` 등 도메인 모듈에 분배된다.

---

## 6. `EventSink` vs `DecodeObserver` 통합 분석

### 6.1 현 `EventSink` 시그니처 (인용)

`engine/src/core/events.rs:69-74`:

```rust
pub trait EventSink: Send + Sync {
    fn emit(&self, event: CacheEvent);
}
```

CacheEvent variants: `PressureDetected{level, mem_available, forced}` / `EvictionCompleted{policy, tokens_removed, new_pos}` / `PipelineStageExecuted{handler, result}` / `ScoreDiagnostic(ScoreSnapshot)` / `ProxyComputed(QcfMetric)` (events.rs:43-67).

소유: `CachePressurePipeline`이 `Arc<dyn EventSink>` 보유 (`HandlerContext::events`로 핸들러에 전달).

### 6.2 의미 호환성 분석

| 차원 | `EventSink` | `DecodeObserver` | 호환? |
|------|------------|------------------|-------|
| 호출 주체 | Pressure pipeline 내부 핸들러 | DecodeLoop 본체 | 다름 |
| 시그니처 | `&self` (Send+Sync 필수) | `&mut self` (단일 스레드 가정) | 다름 |
| 이벤트 도메인 | pressure 결정(eviction/swap/score 진단) | decode step 메트릭(forward_ms, sampled token, observer finalize) | 부분 겹침 (eviction outcome만) |
| 이벤트 비대칭성 | 광범위 enum (ScoreSnapshot 70-byte 구조체 등) | 단순 콜백 (timing, token id) | 다름 |
| 사용처 | stderr diagnostic, 향후 IPC sink | profile/experiment writer, tbt log | 다름 |

### 6.3 권장: **분리 + Adapter** (통합 안 함)

- 분리 근거 (3):
  1. **Send+Sync vs &mut**: `EventSink::emit(&self, _)`는 immutable interior-mutability 패턴 강제 (Mutex/Atomic). `DecodeObserver::on_step_end(&mut self, _)`는 hot path 매 step 호출이며 단일 스레드 owned mut가 효율적 (TbtLogObs는 buffered file writer → `&mut`가 적합).
  2. **이벤트 도메인 격차**: `CacheEvent`는 *pressure 결정의 의미* 표현 (e.g. `ProxyComputed(QcfMetric)`은 cross-cutting QCF). `DecodeObserver` 콜백은 *decode 진행 상황*. 한 trait에 다 넣으면 method 수가 폭증하고 SRP 위반.
  3. **호출 빈도/오버헤드**: EventSink는 eviction event 발생 시만 emit (드물게, ~수 step에 1회). DecodeObserver는 step당 항상 호출. 통합 시 매 step `CacheEvent::StepCompleted{...}` enum boxing 비용이 hot path에 추가됨.

- 통합 안의 단점:
  - `CacheEvent`에 `StepCompleted{step_ms, sampled, ...}` variant 추가 → variant 14개로 폭증.
  - 매 step pattern match overhead.

### 6.4 Adapter: `EventSinkAdapterObs`

`EventSink`가 emit하는 이벤트 중 `EvictionCompleted` / `ScoreDiagnostic` 등을 `DecodeObserver::on_eviction` 콜백으로 옮기고 싶은 케이스를 위해 어댑터 제공:

```rust
pub struct EventSinkAdapterObs(pub Arc<dyn EventSink>);
impl DecodeObserver for EventSinkAdapterObs {
    fn on_eviction(&mut self, _ctx: &StepCtx, outcome: &EvictionOutcome) {
        if let EvictionOutcome::Pruned { removed, new_pos } = outcome {
            self.0.emit(CacheEvent::EvictionCompleted {
                policy: "decode-loop".into(),
                tokens_removed: *removed,
                new_pos: *new_pos,
            });
        }
    }
}
```

이 어댑터는 `session/defaults.rs`에 위치. 사용자는 `.add_observer(EventSinkAdapterObs(stderr_sink))`로 기존 EventSink를 decode loop 관측 경로에 동봉 가능.

### 6.5 위치 결정

- `EventSink`는 **그대로 `pressure/` 또는 `observability/events.rs`에 유지**. Step 5 cross-cutting 분리 시 `observability/events.rs`로 이동(이미 §6.7에 매핑).
- `DecodeObserver`는 **`session/decode_loop.rs` (또는 `session/observer.rs`)에 신설**. L4 도메인 내부 trait.
- 두 trait는 의미 격차로 통합 안 하되, **Adapter로 cross-direction 호환** 제공.

---

## 7. Vtable Overhead 분석

### 7.1 매 토큰 trait 호출 수

| Step 단계 | 호출 | 빈도 |
|----------|------|------|
| cmd poll | `CommandSource::poll` | 1 |
| eviction | `EvictionStage::before_step` | 1 |
| swap before | `SwapStage::before_step` | 1 |
| forward | `Forward::step` | 1 |
| swap after | `SwapStage::after_step` | 1 |
| sample | `TokenSampler::sample` | 1 |
| observer | `DecodeObserver::on_step_end` × N (등록된 수만큼) | N ∈ [0, 4] (profiler+experiment+tbt+system) |
| eviction observer | `DecodeObserver::on_eviction` × N | drop 발생 시만 (~ 5-step 평균 1회) |

**Step당 총 vtable indirect call**: 6 + N ≈ 8-10회.

### 7.2 비용 추정 (Adreno baseline 14 ms TBT)

- ARMv9 indirect call: branch predictor hit 시 ~2-3 cycle, miss 시 ~10-20 cycle (out-of-order recovery 포함). 1 GHz 클럭 가정 시 ~3-20 ns/call.
- 10 calls × 20 ns (pessimistic, miss 가정) = **200 ns/step ≈ 0.0014%** of 14 ms TBT.
- realistic (branch predictor hit, single-call-site): **30-50 ns/step** ≈ 0.0003%.

→ **이론적 overhead는 무시 가능**. CPU baseline (28 ms qnn_oppkg fast off) 대비도 동일.

### 7.3 hot path monomorphization 결정

monomorphize 안:
- `DecodeLoop<F: Forward, E: EvictionStage, S: SwapStage, C: CommandSource, T: TokenSampler, O: DecodeObserver>` 6-generic.
- 장점: vtable 제거, inline 가능.
- 단점: 변형 1개 추가 시(예: KiviForward) `DecodeLoop`의 monomorphized 코드가 1세트 더 생성 — `cargo build` 시간 증가 + binary size 증가 + chat REPL 동적 분기 시 generic instantiation hell.

**권장: monomorphize 비채택 (vtable 사용)**. 다음 두 가지 보조 결정:
1. **측정 게이트**: Migration Step 2-3 (`bin/probe_inference_loop.rs`) 완료 시점에 기존 `main()` vs `DecodeLoop` 1회 microbench. S25 + Jetson + host CPU 3개 디바이스. **회귀 ≤ 5%** PASS 기준. 회귀 시 Step 2-4 진행 전 stop + 가설(vtable miss, sampler hot loop 등) 검증.
2. **사후 escape hatch**: 측정에서 5%-15% 회귀가 관찰되면 `Forward`와 `TokenSampler` 두 trait만 generic화하는 변형(`DecodeLoop<F, T>`)을 도입. 다른 4개는 vtable. 이는 *hot path 1-2개만 monomorphize*하는 절충안.

---

## 8. 책임별 구현체 카탈로그 (변수 흡수 매트릭스)

`main()`의 100+ 변수가 어느 trait 구현체로 흡수되는지의 매트릭스. **god container 회피 증명**.

### 8.1 trait × 구현체 매트릭스

| trait | 구현체 | 흡수 변수 (`main()` 인용) | 모듈 위치 (post-migration) |
|-------|--------|---------------------------|--------------------------|
| **Forward** | `ModelForward` | `backend`, `model`, `kv_caches`, `workspace`, `gpu_buffers`, `logits` | `session/forward/model_forward.rs` |
| **Forward** | `KiviForward` | + `kivi_cache`, `kivi_workspace` | `session/forward/kivi_forward.rs` |
| **Forward** | `OffloadForward` | + `offload_store`, `preload_pool` | `session/forward/offload_forward.rs` |
| **EvictionStage** | `CacheManagerStage` | `cache_manager`, `score_accumulator`, `skip_config`, `last_skip_ratio`, `auto_eviction`, `protected_prefix` | `session/eviction/cache_manager_stage.rs` |
| **EvictionStage** | `NoEvictionStage` | — | `session/defaults.rs` |
| **SwapStage** | `SyncSwapStage` | `incremental_force_swap_plan`, `manager_swap_report_pending`, `ready_weight_swap_report` | `session/swap/sync_swap_stage.rs` |
| **SwapStage** | `AsyncSwapStage` | + `async_swap_dispatcher`, `intra_forward_swap_hook` | `session/swap/async_swap_stage.rs` |
| **SwapStage** | `PhaseAwareSwapStage` | + `phase_aware_swap_dispatcher` | `session/swap/phase_aware_swap_stage.rs` |
| **SwapStage** | `DynamicKSwapStage` | + `dynamic_k_controller` | `session/swap/dynamic_k_swap_stage.rs` |
| **SwapStage** | `ProbingKSwapStage` | + `probing_k_controller` | `session/swap/probing_k_swap_stage.rs` |
| **SwapStage** | `NoSwapStage` | — | `session/defaults.rs` |
| **CommandSource** | `ManagerCmdSource` | `manager_client`, `command_executor`(절반) | `session/cmd/manager_cmd_source.rs` |
| **CommandSource** | `ScheduleCmdSource` | `experiment_schedule`, `command_executor`(절반) | `session/cmd/schedule_cmd_source.rs` |
| **CommandSource** | `StdinCmdSource` | (chat REPL stdin polling) | `session/cmd/stdin_cmd_source.rs` |
| **CommandSource** | `NoCommandSource` | — | `session/defaults.rs` |
| **TokenSampler** | `GreedySampler` | (default) | `session/defaults.rs` |
| **TokenSampler** | `TempSampler` | `sampling_config.temperature` | `inference/sampling.rs` (re-export `into_sampler()`) |
| **TokenSampler** | `TopKSampler` | + `sampling_config.top_k` | (동일) |
| **TokenSampler** | `TopPSampler` | + `sampling_config.top_p` | (동일) |
| **TokenSampler** | `MixedSampler` | 전체 `sampling_config` | (동일) |
| **DecodeObserver** | `ProfilerObs` | `profiler` | `session/observer/profiler_obs.rs` |
| **DecodeObserver** | `ExperimentWriterObs` | `experiment_writer` | `session/observer/experiment_writer_obs.rs` |
| **DecodeObserver** | `TbtLogObs` | `tbt_log_writer`, `forward_ms_values`, `tbt_values` | `session/observer/tbt_log_obs.rs` |
| **DecodeObserver** | `SystemSamplerObs` | `system_sampler` | `session/observer/system_sampler_obs.rs` |
| **DecodeObserver** | `EventSinkAdapterObs` | (외부 EventSink 보유 시) | `session/defaults.rs` |
| **DecodeObserver** | `NoOpObserver` | — | `session/defaults.rs` |

### 8.2 흡수 안 되는 변수 (DecodeLoop 자체 state)

| 변수 | 처분 | 근거 |
|------|------|------|
| `start_time` | `DecodeLoop` 내부 (생성 시점 기록) | 변경축 없음 — 단순 인스턴스 카운터 |
| `start_pos` | `DecodeLoop::prefill` 후 `self.pos` 자체 | 동일 |
| `stop_flag: AtomicBool` | `DecodeLoop` 내부 + `StepCtx` 노출 | 단일 책임의 일부 |
| `tokenizer` | `SessionInitCtx`에 owned, `main()`에서 prefill 전후 사용 | `Forward::step`은 token id만 받음 — tokenizer는 외부 |
| `args.eos_token_id`, `stop_ids` | `DecodeLoop` 내부 (`is_stop_token()`) 또는 별도 `StopCondition` trait | 변형 적으면 내부, 다양해지면 별도 추출 |

흡수 매트릭스 합계: `main()` 변수 ~110개 중 ~100개가 6 trait 구현체로 분배 (90%+). 나머지는 DecodeLoop 본체 또는 SessionInitCtx 초기화 헬퍼로 자연 분배.

---

## 9. 마이그레이션 순서 (Phase 4 sub-phase)

`ARCHITECTURE.md` §13.7 Step 2와 1:1 정합. 본 절은 산출물 + 검증 게이트만 상세화.

### Phase 4-1 (= Step 2-1) 외곽 추출
- **산출물**: `session/init.rs` (SessionInitCtx + `build()` 헬퍼), `session/cli_dump.rs` (dump_cli_and_exit). `main()` 7,051 → ~6,500 LOC.
- **검증 게이트**: `cargo test --workspace` PASS + S25/Jetson e2e 생성 동치 (greedy seed 동일 토큰).
- **위험**: 거의 없음 — 순수 함수 이동.

### Phase 4-2 (= Step 2-2) trait 정의 + 빌더
- **산출물**: `session/decode_loop.rs` (DecodeLoop struct + builder + StepCtx + 6 trait 정의), `session/defaults.rs` (5개 no-op default), `session/observer/mod.rs` (DecodeObserver re-export).
- **검증 게이트**: `cargo build` PASS, `cargo test --workspace`에서 신규 trait의 dummy unit test 1건 PASS (필수 typestate 컴파일 강제 negative test with `trybuild`), `layer_lint` baseline 그대로 (31건 동결, 신규 trait 정의가 새 위반 야기하지 않음을 확인).
- **위험**: trait 시그니처 동결 — 추후 변경이 어렵다. 본 단계 PR 리뷰에 architect/senior implementer 양측 sign-off 필수.

### Phase 4-3 (= Step 2-3) 첫 구현체 (`ModelForward`)
- **산출물**: `session/forward/model_forward.rs`, `bin/probe_inference_loop.rs` (microbench binary).
- **검증 게이트**: probe binary가 S25 + Jetson + host CPU에서 기존 `generate` 대비 동일 TBT 대역(±5%) + same first 32 tokens. Vtable overhead 측정 게이트(§7) 통과.
- **위험**: vtable cost 회귀. 회귀 ≥ 5% 시 §7.3 escape hatch (partial monomorphization) 검토.

### Phase 4-4 (= Step 2-4) main() 조립자화
- **산출물**: `bin/generate.rs::main()` 7,051 → ≤ 400 LOC. 4개 `build_*_loop` 헬퍼. 모든 forward variant(standard/kivi/offload)가 `DecodeLoopBuilder`를 통해 조립.
- **검증 게이트**: 모든 디바이스 e2e (S25 + Jetson + host CPU) + chat 모드 / kivi 모드 / offload 모드 sanity (각 1회), TBT 회귀 ≤ 5%, all `bin/generate` integration tests PASS.
- **위험**: 큰 PR (수천 LOC 이동). 부분 PR 권장 — 모드별로 분할 (`build_standard_loop` 먼저, 이후 kivi, offload).

### Phase 4-5 (= Step 2-5) 나머지 구현체 + chat 통합
- **산출물**: `KiviForward`, `OffloadForward`, `session/chat_ipc.rs` (← `core/chat_ipc.rs`), `ChatTurnExec` 폐기 (또는 thin adapter로 유지 — §10 위험 분석).
- **검증 게이트**: chat REPL `/stats` 출력 동일, multi-turn KV 누적 정확성, `core/chat_ipc.rs` import zero (V-11 해소).

### 다음 진입점 (이 문서 완료 후)
- Implementer에게 Phase 4-1 위임. SessionInitCtx 헬퍼 추출이 첫 작업.

---

## 10. 위험 분석 + 대응

### 10.1 Trait API 동결 위험 (외부 공개 후 breaking change)

- **시나리오**: 외부 공개 후 `Forward::step` 시그니처에 `metadata: &StepMetadata` 추가 필요. 모든 외부 구현체가 깨짐.
- **심각도**: 높음 (외부 사용자 영향).
- **완화**:
  - 초기 시그니처는 `&StepCtx`로 **확장 가능 struct** 채택 (필드 추가는 non-breaking).
  - default impl 가능한 메서드는 `default { }` 본문으로 추가 (`Forward::on_kv_prune`도 default 검토 — 단 KV 동기화는 안전상 명시적 implement 권장).
  - public API surface 줄이기: `Forward`, `EvictionStage`, `SwapStage`, `CommandSource`, `TokenSampler`, `DecodeObserver`, `DecodeLoopBuilder`, `DecodeLoop`만 `pub`. 나머지(StepCtx 내부 필드, defaults, observer impl 등)는 `pub(crate)`.

### 10.2 Lifetime 복잡도

- **시나리오**: chat REPL에서 IPC 콜백이 DecodeLoop에 mut access. async task 분기 시 lifetime 충돌.
- **심각도**: 중간.
- **완화**:
  - `Box<dyn>` owned (§3.2 옵션 b) 채택으로 caller-side lifetime 제거.
  - chat REPL이 background에서 stdin polling을 한다면 `CommandSource`가 `Arc<Mutex<...>>`로 inner state 관리. DecodeLoop은 여전히 owned.
  - 만약 mut access가 본질적으로 multi-threaded라면 (현재는 single-threaded), 별도 trait `AsyncCommandSource` 도입 검토 — 본 phase 범위 외.

### 10.3 vtable overhead

- **시나리오**: hot path 5+N vtable call이 Adreno TBT에 누적 영향.
- **심각도**: 낮음 (이론 200 ns ≪ 14 ms).
- **완화**: Phase 4-3 measurement gate (§7.3). 회귀 시 partial monomorphization escape hatch.

### 10.4 `ChatTurnExec` 통합

- **현 상황**: `bin/generate.rs:11814`의 `trait ChatTurnExec`가 이미 (a) prefill (b) decode_step (c) reset (d) ensure_capacity (e) on_turn_end (f) stats_line 6 메서드의 *작은 god trait*. KV-type 별 variant (standard/kivi/offload) 3개 impl.
- **3개 옵션**:

| 옵션 | 내용 | 장점 | 단점 |
|------|------|------|------|
| (a) **`ChatTurnExec` 폐기** | `Forward` + `EvictionStage::ensure_capacity` + 외부 `TurnRunner` struct로 분해 | SOLID 정합 | chat REPL의 6 메서드를 trait 3개로 분배 — 호출자 코드 늘어남 |
| (b) **Adapter 유지** | `ChatTurnExec`를 `Forward` 위의 *조합 어댑터*로 유지 (내부에서 Forward + Eviction trait 호출) | 외부 chat 호출자 미변경 | 추상화 한 단계 추가 (trait → trait → trait) |
| (c) **별도 유지** | `ChatTurnExec`는 chat 도메인 전용으로 보존, `DecodeLoop`는 generate 전용 | 안전, 점진적 | 코드 중복 — `ModelForward`와 `StandardTurnExec` 양쪽이 같은 work를 함 |

**권장: (b) Adapter 유지**. 근거:
- `ChatTurnExec`는 KV-type 분기를 이미 잘 해주고 있음 (`run_chat_repl<E: ChatTurnExec>`은 트레잇 매개변수 패턴).
- `DecodeLoop::run_until_stop`이 chat REPL의 main loop 역할을 흡수하되, `ChatTurnExec::reset` / `ensure_capacity` / `stats_line`은 chat 전용 메서드이므로 `Forward` 본 trait에 끌어올리면 SRP 위반 (Forward 사용자 대부분은 reset/stats가 필요 없음).
- Adapter: `StandardChatExec` 구조체가 `Forward` impl을 내부 보유하면서 `ChatTurnExec`도 impl. `run_chat_repl`은 `ChatTurnExec`만 의존. 신규 `DecodeLoop`는 `Forward`만 의존. 두 경로가 깔끔히 분리.

Migration Step 2-5에서 `ChatTurnExec` 인터페이스만 thin 유지하면서 내부 구현이 `Forward` impl을 위임하도록 리팩토링.

### 10.5 인라인 카운터 (start_time, start_pos 등)

- **처분**: §8.2 표 참조 — DecodeLoop 본체 state 또는 SessionInitCtx로 흡수.
- **위험**: 매우 낮음.

### 10.6 Builder의 `Send + Sync` 요구

- **시나리오**: trait object가 `Send + Sync` 미만족이면 spawn 시 fail. 현재 `OpenCLBackend`는 `Send + Sync`이나 일부 OpenCL context는 thread-affine.
- **심각도**: 중간 (background sampling thread, 측정 thread 활용 시 발생).
- **완화**:
  - 초기 trait 정의는 `Send + Sync` 미포함 — single-threaded 가정.
  - 멀티스레드 필요 시 별도 `Send + Sync` extension trait (`AsyncForward: Forward + Send + Sync` 등) 도입. 본 phase 범위 외.

---

## 결정 보류 사항 (사용자 결정 요청)

1. **`session/` vs `inference/` trait 위치**: 본 문서는 `Forward / EvictionStage / SwapStage / CommandSource / TokenSampler / DecodeObserver` 6 trait 모두를 **L4 `session/`** 산하에 둔다 (decode loop 책임 분해이므로). 그러나 `Forward`와 `TokenSampler`는 inference 도메인 성격(forward path + sampling)이라 **L3 `inference/`에 둘 수도 있다**. 후자라면 builder는 L3 trait을 import하지만 INV-LAYER-003(L3↔L3 trait import 허용)으로 합법. **사용자 선호?**
2. **`Forward::on_kv_prune` default**: 안전상 명시적 implement 권장(§10.1)했으나, 외부 사용자 부담 줄이려면 default `{ /* no-op */ }`. KV 동기화를 깜빡 잊을 위험과 외부 친화성 트레이드오프. **명시 vs default?**
3. **chat REPL의 `ChatTurnExec` 처분 옵션** (§10.4): 권장은 (b) Adapter 유지지만, (a) 폐기를 선택하면 코드 단순화가 더 깊다. 외부 공개 시 chat 도메인 노출 정도 결정 필요. **(a) / (b) / (c)?**