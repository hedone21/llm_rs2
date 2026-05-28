# Pipeline Stage Design — DecodeLoop v3 (Hook Pattern)

> **상태**: 설계 finalize 2026-05-27 → **2026-05-28 본 grill 12 결정 반영** (KvBundle/WeightBundle trait 폐기 + KVCacheLayer/WeightLayer trait + ctx 5→3 field + KV dispatch Generic→Trait object). 단일 진실원본.
> **선행 문서**: `arch/inference_pipeline.md` v1 (Phase 4-2/4-3/4-4-2.3 — `Forward / EvictionStage / SwapStage / CommandSource / TokenSampler / DecodeObserver` 7-trait 설계). 본 문서는 v1을 단일 PipelineStage trait + lifecycle phase enum + entry point별 PipelineRegistry 패턴으로 재설계한다.
> **선행 후속 관계**: `arch/inference_pipeline.md` v2는 본 문서 기준으로 Phase β scope에서 재작성된다. 본 sprint 외 별 sprint.
> **본 sprint에서 풀지 않은 결정점**: Q24 sub-trait 4종 (`KVCacheView` / `WeightLayerView` / `SecondaryStore` / `SparsePattern` / `StorageSpec`)의 시그니처 detail. Phase α-W 작업 일부로 통합 (별 sprint X — KVCacheLayer/WeightLayer trait 채택으로 작업 자연 통합).
>
> **본 grill 2026-05-28 핵심 변경 (이전 grill 2026-05-27 supersede)**:
> 1. `KvBundle` / `WeightBundle` trait **폐기** → `KVCacheLayer` / `WeightLayer` trait + interior mutability ((γ) 모델, LayerSlot::rcu_weights 자연 확장)
> 2. `StageContext` 5 field → **3 field** (`kv` / `weights` 폐기, `step` / `backend_ext` / `profiler` 유지)
> 3. KV dispatch Generic (`<C: KVCacheOps>`) → **Trait object (`Arc<dyn KVCacheLayer>`)** 전환 — `engine/src/kv_cache_ops.rs:53` 정책 반전, **ADR-0001 필수**
> 4. Sprint 분리: **Phase α-W (Weight, low risk 2-3주) → ADR-0001 → Phase α-K (KV big refactor 4-6주)**
> 5. KV/Weight 도메인 **비대칭** 인정 — Weight는 LayerSlot+RCU 기존 패턴 활용 (~5 file 추가), KV는 ~20 file refactor + bit-identical 검증
> 6. **(β) sync 모델 + (d-1) primitive only + Q7 (A) composite kernel 유지** — 새 sync 인프라 0건 + storage-format-agnostic API + AttentionKernels sub-trait 분리 미채택
> 7. KvBundle 8 method → KVCacheLayer **5 mutation primitive** 통합 (compact merges/keep atomic, migrate_layer → apply_storage 흡수)
>
> **변경 추적성** (이전 결정 → 본 grill 변경 사유):
> - 2026-05-27 KvBundle 8 method (prune/offload/recall/set_layer_dtype/merge_evicted/compress_prefix/set_sparse_pattern/sync_gpu_scores/reset_gpu_score_acc) → **본 grill (d-1) primitive only 채택**: dtype/codebook/rotation matrix 등을 stage가 모르게 함. 5 mutation primitive (write_kv/write_kv_batch/compact/apply_storage + view)로 축소.
> - 2026-05-27 KvBundle trait 자체 → **본 grill 폐기**: PipelineRegistry 정신(stage 객체 안 캡슐화)과 god ctx 인정의 정합 — 한 ctx field가 모든 layer state를 노출하는 god abstraction은 stage register 시점에 필요 layer handle만 보관하는 패턴으로 자연 해소.
> - 2026-05-27 god ctx 5 field 인정 → **본 grill 3 field 축소**: kv/weights field는 stage 객체 내부로 이동. ctx는 phase-invariant 횡단 인프라(step/backend_ext/profiler)만 보유.
> - 2026-05-27 KV dispatch 명시 결정 없음 (Generic 그대로 가정) → **본 grill ADR-0001 작성**: Q8 mixed storage (layer별 다른 paradigm) + 5년 시야 paradigm 추가 frequency 정당화로 trait object 전환.

---

## 0. TL;DR

- DecodeLoop의 7-trait v2 설계는 (a) `EvictionStage` 시그니처 부족 → 책임 재분배 깊이 (R-2), (b) `ResilienceStage` 신설 결정점 미해결 (R-5), (c) `decode_fallback/{eviction_trigger,swap_dispatch}.rs`의 God Ctx 12/21 필드, (d) `arch/inference_pipeline.md:177` Manager IPC wiring 격차, (e) speculative decoding / per-op measurement 등 신규 책임 추가 비용 문제를 해결하지 못한다.
- 본 문서는 **단일 `PipelineStage` trait + `LifecyclePhase` enum + entry point별 `PipelineRegistry`** 패턴으로 재설계한다. 모든 책임을 stage(phase event handler) 단위로 분해, `Arc<dyn PipelineStage>` 공유 + entry point별 selection.
- 의도된 효과: SRP/OCP 양립 (`PipelineStage::on_phase` 1개 method), 신규 책임 추가 비용 = 새 stage struct + 새 entry point에서 `submit()` 1줄.
- 인정 비용: god ctx (`StageContext` 5 field) 명시 수용 + 권한 강제는 code review (M1만). 컴파일 강제 시 sub-trait 보유 비용이 과대.
- 작업 분해 — **Phase α (WeightBundle prerequisite, 2~3주) / Phase β (hook pattern 본격, 3~4주) / Phase γ (legacy 잔여 마이그레이션, 3~4주)** = 총 8~13주.
- Escape hatch — Pre-α-1 design round 합의 실패 또는 Pre-α-2 PoC 4 test 중 1+ fail 시 v2 (b₁) 7-trait 후퇴. Phase β 진입 후 후퇴는 권장하지 않으며 stage impl 재설계로 해소.

---

## 1. 동기 + 배경

### 1.1 v2 7-trait 잔여 문제

`arch/inference_pipeline.md` v1 (Phase 4-2 HEAD `584496b7`)은 `Forward / EvictionStage / SwapStage / CommandSource / TokenSampler / DecodeObserver` 6 trait + 신설 후보 `ResilienceStage`로 책임을 분해했다. 1차 리뷰에서 다음 문제가 식별되었다.

| ID | 문제 |
|----|------|
| R-2 | `EvictionStage` 시그니처 부족 — pressure handler 다양화 (D2O merge / KIVI quant / SnapKV / KvOffload / Sparse) 흡수를 위한 책임 재분배 깊이 |
| R-5 | `ResilienceStage` 신설 결정점 — Manager IPC ResilienceAction (Evict / SwitchBackend / LimitTokens / Throttle / Suspend / RejectNew / RestoreDefaults / SetPartition / LayerSkip) 흡수처가 불분명 |
| God Ctx | `decode_fallback/{eviction_trigger,swap_dispatch}.rs` 시그니처에 12/21 필드 — 추출했지만 책임 분리 미달 |
| IPC 격차 | `arch/inference_pipeline.md:177` 명시 — `cmd_source.poll()` 결과를 accept-and-drop. ExecutionPlan 적용 0 LOC in argus-cli path |
| 신규 비용 | speculative decoding / per-op measurement / KIVI per-layer / SnapKV / Sparse 등 추가마다 새 trait 또는 기존 trait의 신규 method, 양쪽 모두 OCP 위반 |

### 1.2 단일 hook 패턴 합의

사용자가 더 근본적 재설계 제안: **trait 1개로 통일, 차이는 phase enum과 stage 구현체로 분기**. 본 grill 23 라운드 동안 다음을 확정.

| 결정 | 내용 |
|------|------|
| **Trait 통일** | `PipelineStage::on_phase(&self, phase: &LifecyclePhase, ctx: &mut StageContext) -> Result<StageOutcome>` 1개 method |
| **Phase enum** | 21 variant (P3) + `Fine(FinePhase)` (P4, Cargo feature `pipeline-fine-grained` 컴파일 타임 활성) |
| **Slim ctx (본 grill 2026-05-28)** | **3 field** (`step` / `backend_ext` / `profiler`), 권한 강제는 code review (M1만). kv / weights field 폐기 — Stage 객체가 layer handle 보관 |
| **Forward / Sampler 분리** | bit-identical 회귀 위험 최소화로 별 trait 유지 (W-1) — `DecodeLoop`가 `Box<dyn Forward>` + `Box<dyn TokenSampler>` 직접 보유 |
| **Layer handle 모델 (본 grill 2026-05-28)** | (γ) `Arc<dyn KVCacheLayer>` / `Arc<dyn WeightLayer>` + interior mutability. KvBundle / WeightBundle trait 폐기. `LayerSlot::rcu_weights` 자연 확장 |
| **Manager IPC** | `DecodeLoop` owned (stage 외부) — Manager IPC poll + outbound 발신 모두 DecodeLoop 직접 |
| **Registry 모델** | `Arc<PipelineRegistry>` + `Mutex<Vec<Arc<dyn PipelineStage>>>` (M-γ, interior mutability) |
| **OneShot lifecycle** | `StageLifecycle::OneShot` + `StageOutcome::Consumed` + dispatcher 자동 GC |
| **에러 처리** | panic on Err — partial commit 없음, 후속 stage skip 없음 |
| **Entry point 등록** | X2 (caller가 명시 `submit()`), 드리프트 발생 시 helper 도입 (지금은 YAGNI) |
| **KV dispatch paradigm (본 grill 2026-05-28)** | Trait object (`Arc<dyn KVCacheLayer>`) — kv_cache_ops.rs:53 정책 반전. **ADR-0001 필수** |
| **Weight dispatch paradigm (본 grill 2026-05-28)** | LayerSlot 유지 + WeightLayer trait wrap. Forward path 무변경 |
| **Sprint 분리 (본 grill 2026-05-28)** | Phase α-W (Weight + PipelineStage 인프라, 2-3주) → ADR-0001 → Phase α-K (KV refactor 4-6주) |

---

## 2. 모듈 구조 (L2 / L3 / L4)

본 설계는 INV-LAYER-001 ~ 005 (Engine internal layered architecture) 정신을 보존한다. `PipelineStage` / `LifecyclePhase` / `StageContext` / `KvBundle` / `WeightBundle` / `BackendExtensions` / `Profiler` / `PipelineDispatcher`는 **L2** (engine 직속) 위치. `PipelineRegistry` 와 concrete stage impl은 **L4 `session/`**. `DecodeLoop` 는 L4 진입점.

```mermaid
flowchart TB
    subgraph L5 ["L5 bin/"]
        bins["argus_cli / argus_chat / argus_bench / ..."]
    end

    subgraph L4 ["L4 session/"]
        decode["DecodeLoop"]
        registry["PipelineRegistry<br/>impl PipelineDispatcher"]
        stages["concrete stages<br/>EvictionPolicyStage / KvMergeStage /<br/>SwapDispatchStage / OneShot* × 9"]
        cmdexec["CommandExecutor<br/>Manager IPC"]
    end

    subgraph L3 ["L3 inference/ pressure/ qcf/"]
        forward["Forward (TBD: L4 trait)"]
        sampler["TokenSampler"]
        kvcache["KVCache"]
        pressure_handlers["existing pressure handlers<br/>(migration target)"]
    end

    subgraph L2 ["L2 engine direct (본 grill 2026-05-28)"]
        pstage["PipelineStage trait"]
        phase["LifecyclePhase enum"]
        sctx["StageContext (3 field)"]
        outcome["StageOutcome / StageLifecycle"]
        pdisp["PipelineDispatcher trait"]
        kvl["KVCacheLayer trait<br/>(KvBundle 폐기)"]
        wl["WeightLayer trait<br/>(WeightBundle 폐기)"]
        swm["SwapMetrics trait"]
        bext["BackendExtensions trait"]
        prof["Profiler trait"]
    end

    subgraph L1 ["L1 backend/"]
        be["OpenCL / CUDA / CPU / qnn_oppkg"]
    end

    bins --> decode
    decode --> registry
    decode --> cmdexec
    decode --> forward
    decode --> sampler
    decode -. owns .- prof
    registry --> pstage
    stages --> pstage
    stages -. hold Arc<dyn> .-> kvl
    stages -. hold Arc<dyn> .-> wl
    pstage --> phase
    pstage --> sctx
    sctx --> bext
    sctx --> prof
    kvl -.-> kvcache
    bext --> be

    style pstage fill:#3d2d5a
    style phase fill:#3d2d5a
    style sctx fill:#3d2d5a
    style registry fill:#2d5a3d
    style decode fill:#2d5a3d
```

**층별 위치 결정 근거**:

| 모듈 | 위치 | 근거 |
|------|------|------|
| `PipelineStage`, `LifecyclePhase`, `StageContext`, `StageOutcome`, `StageLifecycle` | L2 (engine 직속) | hook 정의는 abstraction primitive — 어떤 L3 도메인에도 종속되지 않음 |
| `PipelineDispatcher` | L2 | dispatch는 stage 구현체와 무관한 control primitive |
| `KVCacheLayer`, `WeightLayer`, `SwapMetrics`, `BackendExtensions`, `Profiler` | L2 | 본 grill 2026-05-28: KvBundle/WeightBundle 폐기 → layer trait + measurement axis 분리. Stage register 시점 `Arc<dyn>` 보관. |
| `PipelineRegistry` (`impl PipelineDispatcher`) | L4 `session/` | concrete impl, entry point별 caller가 build |
| concrete stage struct (`EvictionStage` 등) | L4 `session/` | stage 구현체는 session policy의 일부, 내부 `Arc<dyn KVCacheLayer>` 보관 |
| `KVCacheLayer` impl (`StandardLayer` / `KIVILayer` / `SparseLayer`) | L3 `core/` 또는 L1 `backend/` 인접 | mechanism 캡슐화 — backend kernel paired |
| `WeightLayer` impl (`LayerSlot` thin wrap) | L3 `models/weights/` | LayerSlot::rcu_weights 패턴 자연 확장 |
| `Forward`, `TokenSampler` | L4 `session/` (현 위치 유지) | 별 trait — INV-LAYER-006 적용 대상 |

---

## 3. Trait 시그니처 (코드 finalize)

### 3.1 `PipelineStage` (L2)

```rust
// engine/src/pipeline_stage.rs (신규)

pub trait PipelineStage: Send + Sync {
    fn name(&self) -> &str;

    /// 기본 Persistent — OneShot stage만 override
    fn lifecycle(&self) -> StageLifecycle { StageLifecycle::Persistent }

    /// Phase event handler.
    ///
    /// # Phase-based KV mutation 제약 (INV-DECODE-STAGE-KV-PHASE)
    ///
    /// `ctx.kv` mutation method (prune / offload / recall / set_layer_dtype /
    /// merge_evicted / compress_prefix / set_sparse_pattern)는 다음 phase에서만 호출:
    ///
    /// 허용: PreEviction, PostEviction, PreSwap, PostSwapBefore, PostSwapAfter,
    ///       PreForward, PostForward, PreSample, PostSample,
    ///       PrefillStart, PrefillChunkBoundary, PrefillEnd,
    ///       SessionStart, SessionEnd, DecodeStart, DecodeEnd, TurnStart, TurnEnd, Finalize
    /// 금지: PreLayer, PostLayer, Fine(*)
    ///
    /// 위반 시 step 안 layer 간 KV state inconsistency → logits corruption.
    /// 컴파일/런타임 강제 X — stage 구현자가 자기 phase 매칭에서 책임.
    fn on_phase(&self, _phase: &LifecyclePhase, _ctx: &mut StageContext<'_>) -> Result<StageOutcome> {
        Ok(StageOutcome::Continue)
    }
}

pub enum StageLifecycle {
    Persistent,
    OneShot,
}

pub enum StageOutcome {
    Continue,
    Stop(StopReason),
    Consumed,    // OneShot stage 자기 실행 완료 신호
}
```

### 3.2 `LifecyclePhase` (L2)

```rust
// engine/src/pipeline_stage.rs (계속)

pub enum LifecyclePhase {
    // session lifecycle
    SessionStart,
    SessionEnd,

    // turn lifecycle (chat REPL 등 multi-turn 시나리오)
    TurnStart,
    TurnEnd,

    // prefill
    PrefillStart,
    PrefillChunkStart { chunk_idx: usize },
    PrefillChunkEnd,
    PrefillEnd,

    // decode step lifecycle
    DecodeStart,
    DecodeEnd,

    // pressure / eviction / swap
    PreEviction,
    PostEviction,
    PreSwap,
    PostSwapBefore,
    PostSwapAfter,

    // forward
    PreForward,
    PreLayer { idx: usize },         // KV mutation 금지
    PostLayer { idx: usize },        // KV mutation 금지
    PostForward { logits_len: usize },

    // sampling
    PreSample,
    PostSample { token: u32 },

    Finalize,

    #[cfg(feature = "pipeline-fine-grained")]
    Fine(FinePhase),
}

#[cfg(feature = "pipeline-fine-grained")]
pub enum FinePhase {
    PostRMSNorm { layer_idx: usize },
    PostQKV { layer_idx: usize },
    PostRoPE { layer_idx: usize },
    PostKVUpdate { layer_idx: usize },
    PostAttention { layer_idx: usize },
    PostFFN { layer_idx: usize },
    PostFFNDown { layer_idx: usize },
}
```

**Variant 결정 (Q4)**:
- **P3 (~22 variant)** — 기본 phase 집합. KV/weight 책임 분기, pressure pipeline, 토큰 lifecycle을 모두 P3에 흡수.
- **P4 (fine phase)** — `pipeline-fine-grained` Cargo feature 활성 시 컴파일 타임에만 추가. **feature off 시 dispatch site 자체 컴파일 제거** (성능 안전망). 사용처: probing per-op measurement, SWIFT skip 최적화 등.
- **PreLayer / PostLayer 결정 (갈래 1 B)** — `LayerBoundary` 통합 폐기. SWIFT skip = `PreLayer`, intra-forward swap = `PostLayer`. 의미 명확.
- **추상화 (갈래 2 α)** — L3 `transformer.rs`는 `PipelineDispatcher` L2 trait의 ref만 의존. concrete `PipelineRegistry` 미사용. §13.8-O (cross-L3 vocabulary trait inversion) 정합.
- **Stage 등록 정책 (갈래 4)** — Cargo feature가 on이라도 stage는 caller가 명시 `add` (opt-in). feature 활성 = 등록 가능성, caller 의도 = 실제 등록.

### 3.3 `StageContext` (L2 slim — 본 grill 2026-05-28: 5 field → 3 field)

```rust
// engine/src/pipeline_stage.rs (계속, 본 grill 갱신)

pub struct StageContext<'a> {
    pub step: StepInfo,
    pub backend_ext: &'a dyn BackendExtensions,
    pub profiler: &'a mut dyn Profiler,
    // kv / weights field 폐기 — stage 객체 register 시점에 layer handle 보관
    // (Arc<dyn KVCacheLayer> / Arc<dyn WeightLayer> 직접 보유, §3.5/§3.6)
}

pub struct StepInfo {
    pub pos: usize,
    pub prev_token: u32,
    pub kv_capacity: usize,
    pub decode_step: usize,
    pub last_forward_ms: f64,
    pub stop_requested: Arc<AtomicBool>,
}
```

**3 field 결정 근거 (본 grill 2026-05-28, 결정 #7)**:
- `step` — phase-invariant 상태 (값 타입, 복사 안전)
- `backend_ext` — backend-specific 확장 (opencl secondary store, gpu score acc, host ptr pool) — phase-invariant 횡단 인프라
- `profiler` — profiling 채널 (production = `NoopProfiler`, zero overhead) — 횡단 인프라

**5 → 3 field 축소 근거 (2026-05-27 → 2026-05-28 변경)**:
- 이전 grill: god ctx 5 field 인정 (kv / weights 포함). 권한 강제는 code review 책임 (M1만).
- 본 grill 재검토: **god ctx 자체가 PipelineRegistry 정신 (stage 객체 안 캡슐화) 위반**. ctx로 kv/weights 노출 = "모든 stage가 모든 layer를 만질 수 있다"는 false flexibility. 실제 stage는 register 시점에 자기 책임 layer만 알면 충분 (예: EvictionStage layer 5번만, SwapStage layer 0..N 모두 등).
- **(γ) 모델 채택**: Stage 객체 내부에 `Arc<dyn KVCacheLayer>` / `Arc<dyn WeightLayer>` field 보관. mutation API는 layer self method (interior mutability via RCU/Mutex/Atomic). `LayerSlot::rcu_weights` (`engine/src/models/weights/slot.rs:158`) 패턴의 자연 확장.

**god ctx 인정 영역 (3 field에서도 보존)**:
- backend_ext / profiler 두 field는 여전히 "어느 stage든 만질 수 있다" — 단 mutation 권한이 의미 있는 영역만 ctx로 노출. code review 책임 (INV-DECODE-STAGE-006) 은 3 field 한정으로 정합.

**폐기된 ctx 필드** (누적):
- `~~execution_plan~~` — `ExecutionPlan` ctx field 자체 폐기 (Q18). 명령은 OneShot stage 객체에 캡슐화.
- `~~outbound~~` — DecodeLoop 직접 발신으로 결정 (Q22).
- `~~forward_sync~~` — `ForwardSync` trait 폐기 (Q9). Forward는 KV state cache 0이므로 notify 불필요.
- `~~kv~~` — **본 grill 2026-05-28 폐기** — KvBundle trait 자체 폐기 + Stage 객체가 `Arc<dyn KVCacheLayer>` register 시점 보관.
- `~~weights~~` — **본 grill 2026-05-28 폐기** — WeightBundle trait 자체 폐기 + Stage 객체가 `Arc<dyn WeightLayer>` register 시점 보관.

### 3.4 `PipelineDispatcher` (L2)

```rust
// engine/src/pipeline_dispatcher.rs (신규)

pub trait PipelineDispatcher: Send + Sync {
    fn dispatch(&self, phase: LifecyclePhase, ctx: &mut StageContext<'_>) -> Option<StopReason>;
}
```

**시그니처 결정 (Q13)**:
- **반환 `Option<StopReason>`** — dispatcher가 stage `Result::Err`을 panic으로 흡수. caller에게는 `Stop(reason)` 또는 정상(`None`)만 노출.
- **Stage trait return은 `Result<StageOutcome>`** — 작성자 편의 (anyhow::bail!, `?` 연산자 사용 가능). 단 dispatcher가 panic으로 변환하므로 `Err` 의미는 "복구 불가 + 정확성 위반".
- **Arc + Mutex** — `Arc<dyn PipelineStage + Send + Sync>` + 내부 mutation은 stage struct에서 `Mutex<T>` 명시. dispatcher는 stage shared ref만 보유.

### 3.5 `KVCacheLayer` (L2) — 본 grill 2026-05-28 (KvBundle trait 폐기 후 layer handle 모델)

**이전 결정 (2026-05-27) → 본 grill 변경**:
- `KvBundle` (8 mutation method + layer-wide vs per-layer scope 혼재) **폐기**.
- 변경 사유: (a) KvBundle trait는 god abstraction — ctx field로 모든 layer state 노출, (b) PipelineRegistry 정신과 충돌 (stage 객체 캡슐화), (c) `LayerSlot::rcu_weights` 와 같은 layer-self-mutation 패턴이 본 프로젝트에 이미 정착되어 있어 자연 확장 가능.
- 본 grill 채택: **`KVCacheLayer` trait + Stage 객체 내부 `Arc<dyn KVCacheLayer>` 보관 + interior mutability**.

```rust
// engine/src/kv_layer.rs (신규, 본 grill 2026-05-28)

pub trait KVCacheLayer: Send + Sync {
    // Read (immutable identity)
    fn idx(&self) -> usize;
    fn current_pos(&self) -> usize;
    fn capacity(&self) -> usize;
    fn view(&self) -> &dyn KVCacheView;

    // Mutation API — interior mutability (`&self`, NOT `&mut self`)
    //
    // INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC 준수 (storage-format-agnostic).
    // INV-DECODE-STAGE-001 (KV-PHASE) 준수 (PreLayer/PostLayer/Fine(*) 금지).
    // sync 모델 (β): R1 (host read) / R2 (release) / R3 (backend transition) 자동.
    //
    // 5 mutation primitive — semantic op only, dtype/codebook/rotation 모름.

    /// Single-token KV write — primitive granularity.
    fn write_kv(&self, pos: usize, k: &[f32], v: &[f32]) -> Result<()>;

    /// Batch KV write — range granularity, prefill/chunk 흡수.
    fn write_kv_batch(&self, pos_range: Range<usize>, k_batch: &[f32], v_batch: &[f32]) -> Result<()>;

    /// Compact — keep + merge atomic operation.
    ///
    /// `keep`: 유지할 token positions (sorted ascending).
    /// `merges`: D2O-style nearest merge `[(evicted_pos, nearest_pos, weight)]` (선택, 빈 slice 가능).
    ///
    /// Sliding / H2O / SnapKV: `compact(keep, &[])`.
    /// D2O: `compact(keep, merges_slice)` — paper Eq.10/11.
    ///
    /// keep 과 merges 의 atomic 결합 — 5 → 3 primitive 통합 근거 (D2O semantic 보장).
    fn compact(&self, keep: &[usize], merges: &[(usize, usize, f32)]) -> Result<()>;

    /// Apply storage spec — dtype/codebook/rotation/sparse pattern 모두 spec 한 axis 로 흡수.
    ///
    /// 이전 grill `set_layer_dtype` / `set_sparse_pattern` / `migrate_layer` 통합.
    /// StorageSpec trait 의 detail signature 는 Phase α-K Q24 sub-trait round.
    fn apply_storage(&self, spec: &dyn StorageSpec) -> Result<()>;
}
```

**Mutation 5 → 3 primitive 통합 근거 (본 grill 2026-05-28)**:
- 이전 (KvBundle 7 mutation method: prune / offload / recall / set_layer_dtype / merge_evicted / compress_prefix / set_sparse_pattern + 2 backend integration)
- 통합:
  - `prune` + `merge_evicted` → **`compact(keep, merges)`** — atomic semantic (D2O Eq.10/11 보장: keep 과 merge 시간적 결합 강함)
  - `offload` + `recall` → **storage spec 의 한 axis (Tier::Primary / Tier::Secondary)** → `apply_storage` 흡수
  - `set_layer_dtype` + `compress_prefix` + `set_sparse_pattern` → **storage spec 다른 axes (Dtype / Codebook / SparsePattern / SnapWindow)** → `apply_storage` 흡수
  - `migrate_layer` → `apply_storage` 흡수 (Tier 가 spec 의 한 axis)
  - `sync_gpu_scores` + `reset_gpu_score_acc` → **`view().score_handle()`** 으로 이동 (KVCacheView sub-trait, Phase α-K Q24)
- 결과: 8 method → 5 method (idx / current_pos / capacity / view + 3 mutation + apply_storage). primitive only, storage-format-agnostic.

**(d-1) primitive only 결정 (본 grill 2026-05-28, 결정 #2)**:
- Method 가 token / layer / range granularity 만 알고 storage paradigm (dtype / codebook / rotation matrix / sparse pattern) 은 **모름**.
- Stage 가 `KVCacheLayer::compact(keep, &[])` 호출 — 그 layer 가 KIVI Q4 인지 standard F16 인지 sparse 인지 stage 가 모름.
- KVCacheLayer impl (예: `KIVILayer` / `StandardLayer` / `SparseLayer`) 이 mechanism 캡슐화.
- 새 paradigm 추가 = 새 impl + paired backend attention kernel (INV-KVCACHELAYER-PAIRED-KERNEL).

**(β) sync 모델 (본 grill 2026-05-28, 결정 #1)**:
- Buffer-level lazy + access-mode-aware API. R1 / R2 / R3 자동.
- 현 인프라 (`OpenCL blocking read`, `CUDA explicit sync in read_buffer`, `migrate_kv 내부 synchronize`) 가 이미 (β) 패턴 — **새 sync 인프라 0건 필요**.
- 이전 grill `INV-KVBUNDLE-SYNC` → **본 grill 폐기** (R1/R2/R3 자동 처리로 흡수).

**Q8 mixed storage 허용 (본 grill 2026-05-28, 결정 #4)**:
- Layer 별 다른 storage paradigm OK (예: layer 0 = KIVI Q4, layer 1 = TurboQuant, layer 2 = Standard F16, ...).
- 결정 #9 (KV Generic → Trait object) 의 직접 결과.

**Q9 호환성 차단 인프라 X (본 grill 2026-05-28, 결정 #5)**:
- 정책-storage 호환 안 되는 조합 (Sparse + H2O score eviction 등) compile-time 차단 / fallback 로직 만들지 않음.
- 만나면 panic (Q12 dispatcher 정신).

### 3.5.1 KV/Weight 도메인 비대칭 (본 grill 핵심 발견)

| 측면 | KV (KVCacheOps + slice, 현 상태) | Weight (LayerSlot + RCU, 현 상태) |
|---|---|---|
| 현 Trait | `KVCacheOps` (~15 method, `engine/src/kv_cache_ops.rs:55`) | 없음 (concrete LayerSlot) |
| 현 Dispatch | Generic `<C: KVCacheOps>` — compile-time inline (kv_cache_ops.rs:53 명시 정책) | concrete + Arc clone — vtable 0 |
| 현 Mixed storage | **불가** (모든 layer 동일 C) | **이미 가능** (LayerWeights dtype layer 별) |
| (γ) 전환 cost | ~20 file refactor + bit-identical 검증 | ~5 file 추가, forward 무변경 |
| `LayerSlot::rcu_weights` 정합 | N/A | **이미 완벽** (slot.rs:158) |
| 명시 결정 충돌 | kv_cache_ops.rs:53 반전 → **ADR-0001 필수** | 없음 |

**비대칭 인정 근거 (본 grill 결정 #10/#11)**:
- 두 도메인을 동일 패턴으로 묶으려 하지 않음. Weight 는 LayerSlot/RCU 패턴 보존 + WeightLayer trait thin wrap, KV 는 KVCacheOps → KVCacheLayer trait object 전환.
- Sprint 분리 (Phase α-W → ADR-0001 → Phase α-K) — risk 분산 + escape hatch.

### 3.6 `WeightLayer` (L2) — 본 grill 2026-05-28 (WeightBundle trait 폐기 후 layer handle 모델)

**이전 결정 (2026-05-27) → 본 grill 변경**:
- `WeightBundle` (10 method 통합 trait) **폐기**.
- 변경 사유: KV 와 동일 — god abstraction 회피, Stage register 시점 layer handle 보관.
- 본 grill 채택: **`WeightLayer` trait + Stage 객체 내부 `Arc<dyn WeightLayer>` 보관 + interior mutability via RCU (LayerSlot::rcu_weights 자연 확장)**.

```rust
// engine/src/weight_layer.rs (신규, 본 grill 2026-05-28)

pub trait WeightLayer: Send + Sync {
    fn idx(&self) -> usize;
    fn view(&self) -> &dyn WeightLayerView;

    /// Apply storage spec — dtype (F16/Q4_0/Q8_0/...) / variant / tier 흡수.
    /// 이전 WeightBundle::swap_layer / enqueue_release / secondary_store 통합.
    fn apply_storage(&self, spec: &dyn WeightStorageSpec) -> Result<()>;

    /// Apply layer-level dispatch (skip / partition / full).
    /// 이전 WeightBundle::set_layer_skip_ratio / set_partition_ratio 통합.
    fn apply_dispatch(&self, dispatch: LayerDispatch) -> Result<()>;
}

/// Layer-level dispatch paradigm — Q12 결정 #12 Fixed 3 variant.
pub enum LayerDispatch {
    Full,
    Skip,
    Partition { gpu_ratio: f32 },
}

/// 별 measurement trait — Q15 release_pending / ratio_generation 등 측정 axis 분리.
pub trait SwapMetrics: Send + Sync {
    fn pending_count(&self) -> usize;
    fn ratio_generation(&self) -> u64;
    fn last_swap_latency_us(&self) -> u64;
    fn finalize_barrier(&self) -> Result<()>;
}
```

**10 → 4 method 통합 (본 grill 2026-05-28, 결정 #10)**:
- 이전 WeightBundle: 10 method (n_layers / current_dtype / layer_view / swap_layer / enqueue_release / release_pending / secondary_store / set_partition_ratio / set_layer_skip_ratio / switch_device)
- 통합 후 WeightLayer: 4 method (idx / view / apply_storage / apply_dispatch)
  - `n_layers` → DecodeLoop / model init 시 `Vec<Arc<dyn WeightLayer>>` 의 `.len()`
  - `current_dtype` → `view().dtype()` 으로 이동
  - `swap_layer` + `enqueue_release` + `secondary_store` → `apply_storage(spec)` 흡수
  - `set_partition_ratio` + `set_layer_skip_ratio` → `apply_dispatch(LayerDispatch::Partition / Skip)`
  - `release_pending` + ratio generation → **별 trait `SwapMetrics`** (measurement axis 분리, INV-LAYER-006 정합)
  - `switch_device` → backend 단위 op (model / engine 수준), layer-self 아님 → DecodeLoop 또는 OneShotSwitchDeviceStage 가 backend 직접 호출.

**LayerDispatch enum 3 variant 고정 (본 grill 2026-05-28, 결정 #12)**:
- Dispatch paradigm frequency 낮음 (연 1건 미만 — Skip = SWIFT 2026, Partition = Tensor Partition 2026).
- enum + match exhaustive 가독성 우월. Custom hook 은 미래 필요 시 variant 추가.

**LayerSlot/RCU 자연 정합 (본 grill 결정 #10)**:
- 본 프로젝트 `LayerSlot::rcu_weights` (`engine/src/models/weights/slot.rs:158`) — 이미 `&self` mutation (RCU semantics).
- `WeightLayer::apply_storage` impl 은 `LayerSlot::rcu_weights` 호출만 thin wrap — forward path 무변경.
- ~5 file 추가 (weight_layer.rs trait + impl + Stage 변환 helper + test) — low risk.

### 3.7 `BackendExtensions` + `Profiler` (L2)

```rust
// engine/src/backend_extensions.rs (신규)
pub trait BackendExtensions: Sync {
    fn as_opencl_secondary(&self) -> Option<&dyn OpenClSecondary>;
    fn gpu_score_acc(&self) -> Option<&dyn GpuScoreAcc>;
    fn host_ptr_swap_pool(&self) -> Option<&dyn HostPtrPool>;
}

// engine/src/profiler.rs (신규)
pub trait Profiler: Send {
    fn record(&mut self, event: ProfileEvent<'_>);
}

pub enum ProfileEvent<'a> {
    Eviction { step: usize, count: usize, scores: &'a [f32] },
    Forward { ms: f64 },
    Swap { layer_idx: usize, latency_us: u64 },
    Token { id: u32 },
}

pub struct NoopProfiler;
impl Profiler for NoopProfiler {
    #[inline] fn record(&mut self, _: ProfileEvent<'_>) {}
}

#[cfg(feature = "profile")]
pub struct InferenceProfiler { /* 기존 코드 */ }
```

**최소 구현 결정 (Q8)**:
- Production default = `NoopProfiler` (`#[inline]` + empty body = zero overhead)
- Feature gate `profile` 활성 시 `InferenceProfiler` 사용 — 기존 `engine/src/observability/profiler.rs` 코드 보존

### 3.8 `Forward` + `TokenSampler` (L4, 변경 없음)

본 설계는 **W-1 (별 trait 유지)** 결정에 따라 `Forward` / `TokenSampler`를 PipelineStage로 통합하지 않는다. 이유: bit-identical 회귀 위험 최소화. `DecodeLoop`가 `Box<dyn Forward>` + `Box<dyn TokenSampler>`를 직접 owned. 시그니처는 `arch/inference_pipeline.md` v1 (Phase 4-2) 그대로 유지.

---

## 4. `PipelineRegistry` (L4 concrete)

```rust
// engine/src/session/pipeline_registry.rs (신규)

pub struct PipelineRegistry {
    stages: Mutex<Vec<Arc<dyn PipelineStage + Send + Sync>>>,
}

impl PipelineRegistry {
    pub fn new(stages: Vec<Arc<dyn PipelineStage + Send + Sync>>) -> Self {
        Self { stages: Mutex::new(stages) }
    }

    /// 단일 등록 함수 — stage 객체가 자기 정보 보유 (lifecycle / name)
    pub fn submit(&self, stage: Arc<dyn PipelineStage + Send + Sync>) {
        self.stages.lock().unwrap().push(stage);
    }

    pub fn stage_names(&self) -> Vec<String> {
        self.stages.lock().unwrap().iter().map(|s| s.name().to_string()).collect()
    }
}

impl PipelineDispatcher for PipelineRegistry {
    fn dispatch(&self, phase: LifecyclePhase, ctx: &mut StageContext<'_>) -> Option<StopReason> {
        let stages = self.stages.lock().unwrap().clone();
        let mut consumed: Vec<Arc<dyn PipelineStage + Send + Sync>> = vec![];
        let mut result = None;

        for stage in &stages {
            match stage.on_phase(&phase, ctx) {
                Ok(StageOutcome::Continue) => {}
                Ok(StageOutcome::Consumed) => {
                    debug_assert!(
                        matches!(stage.lifecycle(), StageLifecycle::OneShot),
                        "Consumed outcome only valid for OneShot stage — got Persistent stage '{}'",
                        stage.name()
                    );
                    consumed.push(Arc::clone(stage));
                }
                Ok(StageOutcome::Stop(r)) => { result = Some(r); break; }
                Err(e) => panic!(
                    "Stage '{}' failed in phase {:?}: {:#}",
                    stage.name(), phase, e
                ),
            }
        }

        if !consumed.is_empty() {
            let mut stages = self.stages.lock().unwrap();
            stages.retain(|s| !consumed.iter().any(|c| Arc::ptr_eq(s, c)));
        }

        result
    }
}
```

**M-γ 결정 (Q21)**: `Arc<PipelineRegistry>` 외부 공유 + 내부 `Mutex<Vec<Arc<dyn>>>` interior mutability. DecodeLoop이 Manager IPC handler에서 `registry.submit(stage)` 호출 가능. PoC에서 contention 측정.

**X2 결정 (Q14)**: entry point별 selection. caller가 명시 `submit()` 또는 `new(vec)`. helper 함수는 후속 도입 (drift 실제 발생 시).

---

## 5. Stage 구현 패턴

### 5.1 패턴 (책임 매트릭스)

**Stage 책임 = phase event handler** (콜백). Stage가 아닌 것:
- 상태 (score_accumulator, profiler 등) — stage struct 내부 `Mutex<T>` field로 보유, ctx 외부 owned 패턴은 별도 (예: score_accumulator는 `Arc<Mutex<>>` 외부 owned + EvictionPolicyStage + OneShotQcfReportStage 양쪽 inject — 미해결 결정점)
- 데이터 추출 함수 (QCF 계산 — EvictionPolicyStage 또는 OneShotQcfReportStage 내부 helper)
- outbound channel (Manager IPC handler — DecodeLoop 본체)

### 5.2 패턴 시그니처 (본 grill 2026-05-28 — (γ) layer handle 모델)

**이전 (2026-05-27) → 본 grill 변경**:
- 이전 패턴: `ctx.kv` / `ctx.weights` 통해 모든 layer 접근. Stage 가 어느 layer 를 만질지 phase 처리 안에서 결정.
- 본 grill 패턴: Stage 가 register 시점에 `Arc<dyn KVCacheLayer>` / `Arc<dyn WeightLayer>` 보관. phase 처리 안에서 직접 layer self method 호출 (interior mutability).

```rust
// 예시: EvictionStage (본 grill 2026-05-28)

pub struct EvictionStage {
    layer: Arc<dyn KVCacheLayer>,        // register 시점 보관 (단일 layer 책임)
    policy: Box<dyn EvictionPolicy + Send + Sync>,

    // mutable state — 명시 Mutex
    score_accumulator: Mutex<Option<AttentionScoreAccumulator>>,
}

impl PipelineStage for EvictionStage {
    fn name(&self) -> &str { "eviction" }
    // lifecycle = default Persistent

    fn on_phase(&self, phase: &LifecyclePhase, _ctx: &mut StageContext<'_>) -> Result<StageOutcome> {
        if !matches!(phase, LifecyclePhase::PreEviction) {
            return Ok(StageOutcome::Continue);
        }

        // ctx.kv 가 아닌 self.layer 통해 직접 — (γ) interior mutability
        let scores = self.layer.view().score_handle().unwrap().read_scores();
        let keep = self.policy.decide_keep(&scores);
        self.layer.compact(&keep, &[])?;  // primitive only — KIVI? Sparse? layer 가 캡슐화.

        Ok(StageOutcome::Continue)
    }
}

// Model init pattern (본 grill 2026-05-28)
let kv_layers: Vec<Arc<dyn KVCacheLayer>> = (0..n_layers)
    .map(|idx| KIVILayer::new(idx, ...) as Arc<dyn KVCacheLayer>)
    .collect();

// Stage register 시 layer handle 직접 전달
let eviction = EvictionStage {
    layer: kv_layers[5].clone(),
    policy: Box::new(SlidingPolicy::new()),
    score_accumulator: Mutex::new(None),
};
pipeline_registry.submit(Arc::new(eviction));

// D2O 예시 — compact 의 merges 활용
let d2o = D2OMergeStage {
    layer: kv_layers[5].clone(),
    state: Mutex::new(D2OState::new()),
};
// on_phase 안에서: self.layer.compact(&keep, &merges_eq10_11)?;
```

OneShot 예시:

```rust
// 예시: OneShotEvictStage (Manager Evict 명령)

pub struct OneShotEvictStage {
    layers: Vec<Arc<dyn KVCacheLayer>>,   // layer-wide eviction = 모든 layer handle 보유
    target_ratio: f32,
}

impl PipelineStage for OneShotEvictStage {
    fn name(&self) -> &str { "oneshot_evict" }
    fn lifecycle(&self) -> StageLifecycle { StageLifecycle::OneShot }

    fn on_phase(&self, phase: &LifecyclePhase, _ctx: &mut StageContext<'_>) -> Result<StageOutcome> {
        if !matches!(phase, LifecyclePhase::PreEviction) {
            return Ok(StageOutcome::Continue);  // 자기 phase 도달 전 대기
        }

        for layer in &self.layers {
            let pos = layer.current_pos();
            let keep_count = (pos as f32 * self.target_ratio) as usize;
            let keep: Vec<usize> = (0..keep_count).collect();
            layer.compact(&keep, &[])?;
        }

        Ok(StageOutcome::Consumed)
    }
}
```

**Layer handle 보관 패턴 결정 근거 (본 grill 결정 #8)**:
- `LayerSlot::rcu_weights` 패턴 자연 확장 — `&self` mutation via interior mutability.
- Stage 가 자기 책임 layer 만 보유 → 권한 명확화 (god ctx 회피).
- KV / Weight 양 도메인 통일 패턴 (KV = `Arc<dyn KVCacheLayer>`, Weight = `Arc<dyn WeightLayer>`).
- INV-STAGE-LAYER-HANDLE (신규) 로 spec 명문화.

### 5.3 Stage 종류 매트릭스

**Persistent stages** (registry build 시 등록):

| Stage | 책임 | Trigger | Phase |
|---|---|---|---|
| `EvictionPolicyStage` | pressure-based auto eviction | KV cap 90% 자동 | PreEviction |
| `KvMergeStage` (D2O) | eviction 후 compensation | 자동 | PostEviction |
| `SwapDispatchStage` | intra-forward swap | layer event 자동 | PostLayer |

**OneShot stages** (Manager 명령으로 동적 `submit`):

| Stage | Manager command | Phase |
|---|---|---|
| `OneShotEvictStage` | Evict | PreEviction |
| `OneShotSwapStage` | SwapWeights | PreSwap |
| `OneShotOffloadStage` | KvOffload | PreEviction |
| `OneShotRecallStage` | RestoreDefaults (또는 별) | PreEviction |
| `OneShotPartitionStage` | SetPartitionRatio | PreForward |
| `OneShotLayerSkipStage` | SetLayerSkip | PreForward |
| `OneShotSwitchDeviceStage` | SwitchHw | PreForward |
| `OneShotKvQuantStage` | KvQuantDynamic | PreForward |
| `OneShotQcfReportStage` | RequestQcf | (미해결 결정점 — PreCommandPoll 폐기 후 후속 round에서 phase 매핑 확정) |

**폐기 stage** (이전 안에서):
- `ManagerCommandStage` — Manager IPC가 DecodeLoop owned (stage 외부)
- `ResilienceApplyStage` — OneShot stage들로 분해 (위 매트릭스)
- `KvQuantizeStage` (Persistent) — `OneShotKvQuantStage`로 (Persistent KV quant는 backlog)
- `ProfilerStage` — profiler ctx field로 직접 노출, stage 불필요

**Backlog (본 sprint scope 외)**:
- SnapKV / Sparse — `CompressHandler` / `SparseHandler` 대응. 향후 Phase γ 이후 stage 추가.
- KvOffloadStage Persistent 모드 — pressure 자동 trigger. 권장: OneShot 우선, Persistent는 후속 backlog.

---

## 6. `DecodeLoop` 시그니처 (L4) — 본 grill 2026-05-28 (kv/weights field 폐기)

```rust
pub struct DecodeLoop {
    // execution
    forward: Box<dyn Forward>,
    sampler: Box<dyn TokenSampler>,

    // pipeline
    pipeline: Arc<PipelineRegistry>,

    // 본 grill 2026-05-28: kv / weights field 폐기.
    // KVCacheLayer / WeightLayer handle 은 Stage 객체 내부에 register 시점 보관.
    // DecodeLoop 는 layer collection 의 직접 owner 가 아님 — model / engine 이 owner.
    backend_ext: Arc<dyn BackendExtensions>,
    profiler: Box<dyn Profiler>,

    // Layer collection — Forward 가 보유 (forward path 무변경), DecodeLoop 가 Stage register
    // 시점에 model 로부터 Arc<dyn KVCacheLayer> / Arc<dyn WeightLayer> 추출하여 Stage 에 전달.
    // (구현 detail — Phase α-W 진입 시 finalize)

    // Manager IPC — DecodeLoop owned (stage 외부)
    command_executor: Option<CommandExecutor>,
    heartbeat_interval_steps: usize,

    // step state
    pos: usize,
    decode_step: usize,
    stop_flag: Arc<AtomicBool>,
}
```

**kv/weights field 폐기 근거 (본 grill 2026-05-28, 결정 #6/#7)**:
- 이전 grill: DecodeLoop 가 `Box<dyn KvBundle>` / `Box<dyn WeightBundle>` 보유. ctx 통해 stage 에 전달.
- 본 grill: Stage 가 register 시점 layer handle 직접 보유. DecodeLoop 는 layer collection 의 직접 owner 가 아니라 model/engine 이 owner. Stage register 시점에 `Arc::clone` 전달.
- 결과: DecodeLoop 필드 수 감소, INV-LAYER-006 (DecodeLoop 추상화 결합도) 더 강화.

### 6.1 `run()` 흐름

```mermaid
sequenceDiagram
    participant Loop as DecodeLoop::run()
    participant Reg as PipelineRegistry
    participant Cmd as CommandExecutor (Manager IPC)
    participant Fwd as Forward
    participant Smp as TokenSampler

    Loop->>Reg: dispatch(SessionStart)
    Loop->>Reg: dispatch(PrefillStart)
    Loop->>Fwd: prefill(...)
    Loop->>Reg: dispatch(PrefillEnd)
    Loop->>Reg: dispatch(DecodeStart)

    loop decode step
        Loop->>Cmd: poll() (interval ok?)
        alt has command
            Cmd-->>Loop: EngineCommand
            Loop->>Reg: submit(OneShot* stage)
        end

        Loop->>Reg: dispatch(PreEviction)
        Loop->>Reg: dispatch(PostEviction)
        Loop->>Reg: dispatch(PreSwap)
        Loop->>Reg: dispatch(PostSwapBefore)
        Loop->>Reg: dispatch(PreForward)
        Loop->>Fwd: step(...) (PreLayer/PostLayer 내부 dispatch)
        Loop->>Reg: dispatch(PostForward)
        Loop->>Reg: dispatch(PreSample)
        Loop->>Smp: sample(...)
        Loop->>Reg: dispatch(PostSample)
        Loop->>Reg: dispatch(PostSwapAfter)
        Loop->>Cmd: heartbeat (interval ok?)

        opt stop_flag set
            Loop->>Reg: dispatch(DecodeEnd)
            break
        end
    end

    Loop->>Reg: dispatch(SessionEnd)
    Loop->>Reg: dispatch(Finalize)
```

### 6.2 Manager IPC 위치 (Q17-2, Q22 결정)

| 책임 | 위치 | 근거 |
|------|------|------|
| `command_executor.poll()` | DecodeLoop 본체 | stage 외부 (Q17-2) — 명령 source 변경 (Manager / Schedule / Stdin)은 entry point별 결정, stage 책임 아님 |
| 명령 → OneShot stage 변환 | DecodeLoop 본체 | 변환 후 `registry.submit(stage)` 호출 |
| heartbeat outbound | DecodeLoop 본체 (Q22) | Outbound 위치 = DecodeLoop 직접 발신. stage가 아닌 이유: heartbeat은 phase event response가 아니라 주기적 외부 통신 |
| `on_token_generated` outbound | DecodeLoop 본체 (Q22) | 위와 동일. Observer/Adapter 패턴 미사용 |

---

## 7. Forward / Sampler (W-1 별 trait)

`PipelineStage`와 별도로 `Forward`와 `TokenSampler`는 `arch/inference_pipeline.md` v1 시그니처 유지. Pre-α-1 design round에서 별 sprint 검토 가능하나 본 sprint scope 외.

```rust
// engine/src/session/traits.rs (v1 기존 유지)

pub trait Forward: Send {
    fn prefill(&mut self, /* ... */) -> Result<...>;
    fn step(&mut self, /* ... */) -> Result<...>;
    // lifecycle hook은 default no-op
    fn finalize(&mut self) -> Result<()> { Ok(()) }
}

pub trait TokenSampler: Send {
    fn sample(&mut self, logits: &[f32], step: usize) -> Result<u32>;
}
```

---

## 8. 신규 spec invariant 7건 + 폐기 INV 매트릭스

### 8.1 신규 INV (본 grill 2026-05-28 갱신)

| Spec ID | 한줄 요약 | 카테고리 | 검증 |
|--------------------|---------|---------|------|
| **INV-DECODE-STAGE-001 (KV-PHASE)** | KV mutation 허용 phase 명시 (PreLayer / PostLayer / Fine(*) 금지). PipelineStage trait 주석 + spec 본문. Stage 구현자 책임 (M1만, runtime/compile 강제 없음). | Correctness | static (코드리뷰), test |
| **INV-DECODE-STAGE-004 (OUTCOME)** | StageOutcome 3 variant (Continue / Stop / Consumed) 처리. 한 phase 내 즉시 commit + Stop 시 break + Consumed는 OneShot stage만 반환. Persistent stage가 Consumed 반환 시 debug_assert. | Correctness | runtime, test |
| **INV-DECODE-STAGE-005 (ORDER)** | caller 책임 (register 순서). entry point 안에서 명시 `registry.submit()` 순서. 등록 순서가 동일 phase 내 dispatch 순서. | Correctness | static (코드리뷰) |
| **INV-DECODE-STAGE-006 (CTX-AUTHORITY)** | ctx 3 field (`step` / `backend_ext` / `profiler`) mutation 권한 강제 X — code review 책임 (M1만). 본 grill 2026-05-28: 5 → 3 field 축소로 권한 명확화 (kv/weights field 폐기). PR checklist 의무화. | Correctness | static (코드리뷰), test |
| **INV-DECODE-STAGE-007 (LIFECYCLE)** | `StageLifecycle::OneShot`은 자기 phase 도달 시 Consumed 반환 → dispatcher 자동 GC. Persistent는 Consumed 반환 금지 (debug_assert). | Correctness | runtime, test |
| **INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC** (신규) | KVCacheLayer mutation method (`write_kv` / `write_kv_batch` / `compact` / `apply_storage`) 는 storage-format-agnostic. Stage 가 dtype / codebook / rotation matrix / sparse pattern 알면 안 됨. KVCacheLayer impl 이 mechanism 캡슐화. 위반 시 stage 가 storage paradigm 별 분기 로직 가짐 → OCP 위반 + 새 paradigm 추가 cost 폭증. | Correctness | static (코드리뷰), test |
| **INV-KVCACHELAYER-PAIRED-KERNEL** (신규) | KVCacheLayer impl 과 paired backend attention kernel 매핑. 새 storage paradigm = 새 layer impl + paired kernel set. Sparse layer + standard attention kernel 등 mismatched 조합 금지 (Q9 호환성 차단 인프라 X — 만나면 panic). | Correctness | static (코드리뷰), runtime (panic) |
| **INV-STAGE-LAYER-HANDLE** (신규) | PipelineStage 가 layer handle (`Arc<dyn KVCacheLayer>` / `Arc<dyn WeightLayer>`) 을 register 시점 보관. StageContext 에 kv/weights field 두지 않음. 본 grill (γ) 모델의 핵심. 위반 시 ctx god abstraction 재발생. | Correctness | static (코드리뷰), test |

### 8.2 폐기 INV 매트릭스 (본 grill 2026-05-28 갱신)

| ID | 폐기 사유 |
|----|---------|
| `INV-EXECUTIONPLAN-CONSUME` (가상 ID) | ExecutionPlan ctx field 자체 폐기 (Q18). 자연 해소. |
| `INV-FORWARDSYNC-*` (가상 ID) | `ForwardSync` trait 폐기 (Q9). Forward는 KV state cache 0이므로 notify 불필요. 자연 해소. |
| **`INV-DECODE-STAGE-002` (KVBUNDLE-CONSISTENCY)** | **본 grill 2026-05-28 폐기** — KvBundle trait 자체 폐기. KVCacheLayer 는 layer-self primitive 만 가짐 (idx 명시 불필요). layer-wide vs per-layer 구분 자체가 사라짐. |
| **`INV-DECODE-STAGE-003` (KVBUNDLE-SYNC)** | **본 grill 2026-05-28 폐기** — (β) sync 모델 채택. R1/R2/R3 자동 처리로 흡수. 현 인프라 (OpenCL blocking read / CUDA explicit sync / migrate_kv 내부 synchronize) 가 이미 (β) 패턴. 새 sync 인프라 0건. |

### 8.3 본 grill 폐기 INV 정신 보존

INV-DECODE-STAGE-002 / 003 의 정신은 다음에 흡수되어 보존:
- INV-DECODE-STAGE-002 (KVBUNDLE-CONSISTENCY) 정신 → **INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC** (storage-format-agnostic primitive). layer scope 명시 의무가 primitive granularity 명시 의무로 변환.
- INV-DECODE-STAGE-003 (KVBUNDLE-SYNC) 정신 → (β) sync 모델 + R1/R2/R3 자동 처리. KVCacheLayer mutation method 의 호출 후 GPU work 완료 보장은 **impl 내부 책임으로 격리되며 trait 본문에서는 의무 명시 X** (자동 처리이므로).

**INV-LAYER-006/007과의 관계**:
- INV-LAYER-006 (DecodeLoop 추상화 결합도): 본 설계의 `DecodeLoop` 필드는 모두 `Box<dyn>` / `Arc<dyn>` 또는 generic bound — INV-LAYER-006 정합.
- INV-LAYER-007 (typestate builder): 본 설계의 `DecodeLoopBuilder`는 v2 typestate 패턴 보존. `Forward` 필수, 나머지 optional + default.

---

## 9. 리스크 매트릭스 (RPN 점수) — 본 grill 2026-05-28 갱신

### 9.1 현재 매트릭스 (본 grill 결정 12 반영)

| ID | 리스크 | RPN | 완화 |
|---|---|---|---|
| **R-G1** | **KV dispatch hot path vtable cost (`Arc<dyn KVCacheLayer>` for every layer-step KV write)** | **144** | Phase α-K PoC TBT 게이트 (S25 Δ ≤ +3%). 측정 실패 시 ADR-0001 revoke + Generic 유지 + Q8 mixed storage 포기 (갈래 1 후퇴). |
| **R-G2** | KVCacheOps (~15 method) → KVCacheLayer (5 method) 매핑 시 semantic loss (특히 score acc / KIVI-specific ops) | 105 | Phase α-K 진입 전 `engine/src/kv_cache_ops.rs` 의 15 method 전수 매핑 표 작성 + KVCacheView sub-trait 에 흡수 검증. semantic loss 발견 시 trait method 추가 vs spec 재설계 결정. |
| **R-G3** | ADR-0001 (kv_cache_ops.rs:53 정책 반전) 미작성 시 미래 explorer 가 Generic 으로 회귀 시도 | 60 | ADR-0001 작성 **필수** (status: Accepted) + kv_cache_ops.rs:53 주석에서 ADR-0001 참조 갱신 (Phase α-K 구현 단계). |
| **R-G4** | **CacheManager / EvictionPolicy / D2OHandler 등 기존 ~10 component 가 KVCache 구체 타입에 강결합 — KVCacheLayer trait object 전환 시 분해** | **168** | Phase α-K 작업 분해 시 component 별 마이그레이션 plan 작성 (CacheManager → Per-Layer Stage 분해, EvictionPolicy → EvictionStage 흡수 등). 기존 CachePressurePipeline 의 흡수 경로 명시. |
| **R-G5** | **Bit-identical 검증 — 5 mutation primitive 통합으로 인한 D2O / SnapKV / KIVI 출력 차이** | **168** | Phase α-K 종료 게이트: S25 Qwen2.5-1.5B Q4_0 32-token bit-identical (baseline `master` vs Phase α-K branch). 모든 KV paradigm (Sliding / H2O / D2O / KIVI / SnapKV) 32 토큰 token id sequence 일치. |
| R-V3-1 | 8~13주 sunk cost | 224 | Phase α-W (low risk 2-3주) 먼저 → ADR-0001 → Phase α-K (high risk 4-6주). risk 분산 + escape hatch. |
| R-V3-2 | code review 운영 부담 (3 field god ctx + INV-DECODE-STAGE-006) | **112** (축소: 5 field → 3 field) | INV-DECODE-STAGE-006 (CTX-AUTHORITY) + PR checklist. 3 field 한정으로 권한 명확화. |
| R-V3-3 | WeightLayer 추상화 폭 미확정 | **84** (축소: KvBundle/WeightBundle → layer trait 분리로 폭 명확) | Phase α-W 종료 spec test (S25 bit-identical + weight swap 정확성 회귀 0). |
| R-V3-4 | `arch/inference_pipeline.md` 전면 재작성 (Phase β scope) | 126 | v1 git history 보존 + 보존/철회 매트릭스 (§10.1). 본 grill 결정 반영하여 v1 deprecation notice 갱신. |
| R-K1 | **mid-forward KV mutation → corruption** | **270** | M1 만 — INV-DECODE-STAGE-001 (KV-PHASE) spec + trait 주석 (stage 구현자 책임). KVCacheLayer 도 동일 — interior mutability 가 위반 위험 안 줄임. |
| R-K2 | KIVI prefill mid-layer | 160 | `OneShotKvQuantStage` 가 PreForward phase 제한. KIVI = KVCacheLayer impl 변형 (Q8 mixed storage). |

### 9.2 폐기된 리스크 (이전 grill 2026-05-27 → 본 grill 2026-05-28)

| ID | 폐기 사유 |
|----|---------|
| R-K3 (layer-wide vs per-layer 일관성, RPN 105) | KvBundle trait 폐기 → 자연 해소. KVCacheLayer 는 layer-self primitive 만 가짐 (idx 명시 불필요). |
| R-K4 (GPU sync, RPN 120) | (β) sync 모델 채택 — 새 sync 인프라 0건. R1/R2/R3 자동 처리로 흡수. INV-KVBUNDLE-SYNC 폐기. |
| R-K5 (Recall position drift, RPN 72) | recall = apply_storage(Tier::Primary) 의 한 case → KVCacheLayer impl 내부 처리. layer-self primitive 보장. |
| R-V3-5 (hot path vtable N × M dispatch, RPN 108) | **R-G1 으로 흡수 + 약화** — N × M 은 phase iteration overhead (PipelineStage trait), R-G1 은 layer-step KV write 의 KVCacheLayer trait object cost. 본 grill 에서 layer-step 만 별 리스크화. |
| R-K6 / R-K7 (`ForwardSync` 트랜잭션 race / Forward ctx 노출 시 mutable race) | `ForwardSync` 폐기 + ctx 에 Forward 미노출로 자연 해소 (이전 grill 결정 보존). |

### 9.3 본 grill 신규 5 리스크 (R-G1 ~ R-G5) 사유

- **R-G1 (RPN 144)**: KV dispatch Generic → Trait object 전환은 layer-step (forward path) hot path 에 매 step `Arc<dyn>::call` vtable 1 lookup 추가. Modern CPU 에서 ~1-3 ns/call cost. layer-step 빈도 (n_layers × decode_step) × cost 가 0.1-0.3 ms/token 추가 → 본 프로젝트 S25 Qwen2.5-1.5B Q4_0 TBT 32 ms/tok 의 1% 미만 예상. 다만 측정 게이트 (Δ ≤ +3%) 까지 안전 margin 있음. PoC 측정 실패 시 ADR-0001 revoke.
- **R-G2 (RPN 105)**: KVCacheOps 15 method (set_current_pos / capacity / write_token / get_score_handle / migrate / ...) 가 KVCacheLayer 5 method 로 정합 매핑 가능한지 미검증. 일부 method 는 KVCacheView sub-trait (Q24-1) 으로 이동 가능하나 score acc 의 GPU buffer ownership 등 edge case 존재.
- **R-G3 (RPN 60)**: kv_cache_ops.rs:53 의 명시 정책 ("Generic monomorphization preserves zero overhead") 반전 → ADR 없이 미래 explorer 가 Generic 으로 회귀 시도하면 본 grill 결정 lost.
- **R-G4 (RPN 168)**: CacheManager (Pipeline mode, `engine/src/core/cache_manager.rs`) / EvictionPolicy / D2OHandler / CachePressurePipeline 등 ~10 component 가 `KVCache` 구체 타입 또는 `<C: KVCacheOps>` generic 에 강결합. KVCacheLayer trait object 전환 시 모든 component 가 마이그레이션 대상 — 분해 / 흡수 경로 미확정.
- **R-G5 (RPN 168)**: 5 mutation primitive 통합 (특히 `compact(keep, merges)`) 으로 인한 D2O / SnapKV / KIVI 출력 비트 정확성 회귀 위험. 기존 `merge_evicted` 의 paper Eq.10/11 semantic 이 compact 안에서 동일하게 보존되는지 검증 필요.

---

## 10. Phase α-W / ADR-0001 / Phase α-K / β / γ 작업 분해 (본 grill 2026-05-28)

### 10.0 본 grill 결정 #11 — Sprint 분리

이전 grill (2026-05-27): 단일 Phase α (WeightBundle prerequisite) 2-3주.
본 grill (2026-05-28): **Phase α-W (Weight + PipelineStage 인프라, low risk 2-3주) → ADR-0001 (KV dispatch paradigm) → Phase α-K (KV Generic → Trait object 4-6주)**.

분리 사유: KV/Weight 도메인 비대칭 (§3.5.1). Weight 는 LayerSlot/RCU 자연 정합 (~5 file 추가), KV 는 ~20 file refactor + bit-identical 검증. 큰 risk 를 작은 risk 위로 쌓지 않음 — Phase α-W 가 PipelineStage / EvictionStage / SwapStage / OneShot 인프라를 검증한 뒤 KV refactor 진입.

| Phase | Sub-sprint | 기간 | 게이트 |
|---|---|---|---|
| **Pre-α-1** | Architect design round — 본 grill 결정사항 명문화(본 문서) + 핵심 sub-trait detail finalize (`KVCacheView` / `WeightLayerView` / `StorageSpec` / `WeightStorageSpec` / `SparsePattern`) | 2~3일 | design-review PASS |
| **Phase α-W** | **Weight + PipelineStage 인프라 + Bundle 폐기** — (1) `PipelineStage` trait + `LifecyclePhase` + `PipelineRegistry` L2 정의, (2) `WeightLayer` trait + `LayerSlot` thin wrap, (3) `SwapMetrics` 별 trait, (4) Stage impl (EvictionStage / SwapDispatchStage / OneShot* 5종) — KVCache 부분은 KVCacheOps 구상 generic 유지 (Phase α-K 진입 전 placeholder), (5) argus_cli 또는 작은 entry point 마이그레이션, (6) PoC 4 test (§11) | 2~3주 | (a) S25 Qwen2.5-1.5B Q4_0 32 token bit-identical, (b) avg_tbt Δ ≤ +3% (INV-147 noise 3배), (c) Weight swap (`--secondary-layout aos` 또는 AUF) 정확성 회귀 0건, (d) spec test (INV-DECODE-STAGE-001~007 + 신규 INV-STAGE-LAYER-HANDLE + INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC + INV-KVCACHELAYER-PAIRED-KERNEL) PASS |
| **ADR-0001** | **KV dispatch paradigm — Generic → Trait object 정식 결정 문서** (`docs/adr/0001-kv-dispatch-paradigm.md`) — Phase α-W 종료 후 ADR 작성, Status: Accepted. kv_cache_ops.rs:53 의 정책 반전 정당화 + Rejected alternatives (갈래 1/3/4) 명시 + Validation gate | 2-3일 | ADR 작성 + Architect review + kv_cache_ops.rs:53 주석 갱신 (구현은 Phase α-K) |
| **Phase α-K** | **KV Generic → Trait object 전환** — (1) `KVCacheLayer` trait L2 정의, (2) KVCacheOps 15 method → KVCacheLayer 5 method + KVCacheView sub-trait 매핑, (3) `StandardLayer` / `KIVILayer` impl + paired backend attention kernel 매핑 검증, (4) CacheManager / EvictionPolicy / D2OHandler / CachePressurePipeline 흡수, (5) Forward path generic `<C: KVCacheOps>` → `&[Arc<dyn KVCacheLayer>]` 변환 (~20 file refactor), (6) Mixed storage test (layer 0 Q4, layer 1 F16 등) | **4~6주** | (a) S25 Qwen2.5-1.5B Q4_0 32 token bit-identical (모든 KV paradigm: Sliding / H2O / D2O / KIVI / SnapKV), (b) avg_tbt Δ ≤ +3%, (c) Mixed storage test PASS, (d) RPN ≥ 100 리스크 (R-G1 ~ R-G5) 모두 GREEN |
| **Phase β** | **Hook pattern 본격 + DecodeLoop 재작성** — L4 DecodeLoop 재작성 + 모든 stage impl 완료 + `arch/inference_pipeline.md` v2 작성 + argus-cli 마이그레이션 | 3~4주 | S25 32토큰 bit-identical + avg_tbt Δ ≤ +3% + INV 신규 PASS |
| **Phase γ** | 잔여 마이그레이션 — legacy `bin/generate.rs` (generate-split-binaries 묶음 옵션) + `session/{chat,ppl,eval,prefill}` + chat unsafe 정리 | 3~4주 | `grep cache_manager. 0건` + miri PASS + PACT2026 PoC |

**총 12~19주** (이전 grill 8~13주 → 본 grill +4-6주, KV refactor risk 분리).

### 10.1 보존/철회 매트릭스 (Phase β 진입 전 명시 필요)

`arch/inference_pipeline.md` v1의 다음 컴포넌트는 본 설계 진입 시 운명이 결정된다:

| v1 컴포넌트 | 운명 | 근거 |
|---|---|---|
| `Forward` trait | **보존** | W-1 결정 |
| `TokenSampler` trait | **보존** | W-1 결정 |
| `EvictionStage` trait | **폐기** | `EvictionPolicyStage: PipelineStage`로 흡수 |
| `SwapStage` trait | **폐기** | `SwapDispatchStage: PipelineStage`로 흡수 |
| `CommandSource` trait | **폐기** | DecodeLoop 본체에서 `CommandExecutor` 직접 보유 |
| `DecodeObserver` trait | **폐기** | DecodeLoop 본체에서 직접 outbound |
| `DecodeLoopBuilder` typestate | **보존** | INV-LAYER-007 정합 |
| `SessionInitCtx` | **보존** | Phase 4-1 산출물, 본 설계와 직교 |
| `legacy generate.rs` | **보존** (Phase γ까지) | `[[generate-split-binaries]]` backlog 묶음 옵션 |

### 10.2 본 grill 신규 분리 — KvBundle/WeightBundle 운명

| 컴포넌트 | 이전 grill (2026-05-27) | 본 grill (2026-05-28) |
|---|---|---|
| `KvBundle` trait | L2 신규 신설 (8 method) | **폐기** — KVCacheLayer trait (5 method) 로 대체 |
| `WeightBundle` trait | L2 신규 신설 (10 method) | **폐기** — WeightLayer trait (4 method) + SwapMetrics 별 trait |
| `StageContext::kv` field | `&mut dyn KvBundle` 노출 | **폐기** — Stage 내부 `Arc<dyn KVCacheLayer>` |
| `StageContext::weights` field | `&mut dyn WeightBundle` 노출 | **폐기** — Stage 내부 `Arc<dyn WeightLayer>` |
| KV dispatch | (명시 결정 없음, Generic 가정) | **Trait object 전환** — ADR-0001 필수 |
| Weight dispatch | (명시 결정 없음, concrete 가정) | **LayerSlot + WeightLayer thin wrap** (forward 무변경) |

---

## 11. Phase α-W PoC scope (4 test) — 본 grill 2026-05-28

```
1. PipelineStage v0 + EvictionStage + SwapDispatchStage (Weight, layer handle 패턴) — 정상 동작 확인
2. S25 Qwen2.5-1.5B Q4_0 32토큰 bit-identical (baseline `master` vs PoC branch)
3. S25 microbench — stage N=5~7 × phase M=21 (P3) × 32토큰 TBT
   임계: avg_tbt Δ ≤ +3% (INV-147 noise 기준 — `tok0` inclusive avg_tbt)
4. Weight swap (`--secondary-layout aos` 또는 AUF) 정확성 회귀 0건 — switch_hw cpu/gpu 후 forward 정확성
```

**측정 방법**:
- (2) bit-identical: `python scripts/run_device.py -d s25 generate --backend opencl --opencl-rpcmem --model-path qwen2.5-1.5b-q40.auf --prompt "..." --max-tokens 32 --seed 42`. baseline (HEAD `master`) vs PoC branch token id sequence 비교.
- (3) avg_tbt: tok0 inclusive (feedback `feedback_tbt_metric_tok0_inclusive.md` 부합). `n ≥ 5, median`. INV-147 (Performance, hook=None zero overhead, < 1%) 적용.
- (4) Weight swap: `OneShotSwapStage` submit + WeightLayer::apply_storage(F16 → Q4_0). 5 layer 변환 후 forward token 정확성 검증.

**Phase α-K 진입 게이트 (별도)**: KV refactor 본격 진입 전 ADR-0001 작성 + Architect review. PoC test 는 Phase α-W 검증 완료 후 KV-specific 시나리오 (KIVI per-layer mix, mixed storage layer 0 Q4 + layer 1 F16) 로 확장.

---

## 12. Escape hatch — v2 (b₁) 7 trait 후퇴

| 시점 | 조건 | 후퇴 |
|---|---|---|
| Pre-α-1 종료 | design round 합의 실패 | v2 (b₁) 7 trait |
| Pre-α-2 PoC 종료 | 4 test 중 1+ fail | v2 (b₁) 7 trait — WeightBundle 인프라는 (b₁)에서도 활용 (sunk 0) |
| Phase α 종료 | WeightBundle 추상화 폭 미해결 | v2 (b₁) + WeightBundle 별 sprint 자산 |
| Phase β 진입 후 | — | 후퇴 권장 안 함. 게이트 실패 시 stage impl 재설계로 해소 |

---

## 13. 미해결 결정점 — 본 grill 2026-05-28 갱신

본 grill 에서 풀지 않은 사항. Phase α-W / Phase α-K 진입 전 finalize 필요.

### 13.1 Phase α-W 진입 전 sub-trait detail finalize

본 grill 2026-05-27 의 Q24 sub-trait 4종 → 본 grill 2026-05-28 결정 (KvBundle / WeightBundle 폐기) 후 변형:

**Q24-1 (변경): `KVCacheView`** — 이전 grill `KvBundle::layer_view` 반환 type → 본 grill `KVCacheLayer::view()` 반환 type. score acc handle / capacity / pos / dtype query 등. `engine/src/kv_cache_ops.rs` 의 15 method 중 read-only method 가 흡수 후보.

**Q24-2 (변경): `WeightLayerView`** — 이전 grill `LayerView` (generic LayerView) → 본 grill `WeightLayer::view()` 반환 type. Llama / Qwen / Mistral 흡수. Generic `weight_tensor(name)` vs specific method.

**Q24-3 (유지): `SecondaryStore`** — backlog [P2] `arch/weights_pressure_split.md §7.5` 확정. mmap-based vs file-based vs streaming. AUF format integration (`arch/auf_format.md`). secondary missing fallback 정책. **Phase α-W 진입과 같이 진행 (WeightLayer::apply_storage 의 dependency)**.

**Q24-4 (유지): `SparsePattern`** — 본 sprint scope 결정 (stub or 별 sprint). Sparse attention kernel 통합 (현 미구현). KVCacheLayer impl 인 `SparseLayer` 의 dependency.

**Q24-5 (신규): `StorageSpec` + `WeightStorageSpec`** — 본 grill 결정 #2/#3 (apply_storage 흡수) 의 spec object 시그니처. Dtype / Codebook / Rotation / Tier / SparsePattern axis 표현. trait object vs enum 검토.

### 13.2 Phase α-K 진입 전 매핑 작업

- **KVCacheOps 15 method → KVCacheLayer 5 method 매핑 표** (R-G2 mitigation): `engine/src/kv_cache_ops.rs:55` 의 모든 method 가 (a) KVCacheLayer 의 5 mutation primitive, (b) KVCacheView read-only method, (c) 폐기 (mechanism 안으로 흡수) 중 어디로 가는지 전수 표 작성.
- **CacheManager / EvictionPolicy / D2OHandler / CachePressurePipeline 마이그레이션 plan** (R-G4 mitigation): 각 component 가 KVCacheLayer trait object 모델에서 어떻게 분해 / 흡수 / 보존되는지 design round.

### 13.3 기타 미해결

- **`KvOffloadStage` 정책** — Persistent (pressure 자동) vs OneShot (Manager 명령) 결정. **권장**: OneShot 우선, 자동 pressure 기반은 Phase γ 이후 backlog.
- **`score_accumulator` ownership** — 본 grill 결정 #8 (Stage register 시점 layer handle 보관) 후: `KVCacheLayer::view()::score_handle()` 으로 layer 내부 흡수 → 외부 ownership 패턴 불필요. **finalize 됨**.
- **`OneShotQcfReportStage` phase** — Manager `RequestQcf` 명령이 어느 phase 에서 처리? `PreEviction` 또는 `DecodeEnd` 후보.

---

## 14. 합의 결정 요약 (Q1 ~ Q23)

| Q | 결정 | 비고 |
|---|---|---|
| Q1: God 객체 정체 | (b)/(c) — PipelineRegistry 외부 객체 | (a) DecodeLoop owned는 prefill 확장 어려움 |
| Q2: ownership 모델 | (c') — handle 공유 + entry point별 registry build | `Arc<dyn PipelineStage>` 공유, entry point별 selection 명시 |
| Q3: mutation 채널 | (A) god ctx | 인프라(KV/weight/plan/backend) mutation은 ctx 통해 |
| Q4: phase 매트릭스 | P3 (~22 variant) + P4는 `pipeline-fine-grained` Cargo feature 컴파일 타임 활성 | P4 feature off 시 dispatch site 자체 컴파일 제거 |
| Q4 갈래 1 | (B) Pre/PostLayer, LayerBoundary 폐기 | SWIFT skip = PreLayer, intra-forward swap = PostLayer |
| Q4 갈래 2 | (α) trait 추상화 — `PipelineDispatcher` L2 trait | L3 `transformer.rs`가 L2 trait ref만 의존, §13.8-O 정합 |
| Q4 갈래 3 | INV-147 +3% PoC 게이트 | 단일 hook 대신 stage N × phase M iteration 측정 |
| Q4 갈래 4 | feature flag opt-in stage 등록 | feature on이라도 stage는 caller가 명시 add |
| Q5: StageScratch | **폐기** | 통신 케이스가 outcome / outbound / 책임 재분배로 모두 해결 |
| Q6: Forward/Sampler vs PipelineStage | **(W-1) 별 trait** | bit-identical 회귀 위험 최소, Forward는 `Box<dyn Forward>` DecodeLoop owned |
| Q7: outbound channel | (변경됨 — Q22에서 ctx field 폐기) | DecodeLoop이 직접 발신으로 결정 |
| Q8: Profiler | 최소 구현 — `Profiler` trait 1 method + `ProfileEvent<'a>` enum 4 variant + `NoopProfiler` default | production default = Noop (zero overhead), `#[cfg(feature = "profile")]` InferenceProfiler |
| Q8: KV 정책 stage 다양화 | EvictionPolicyStage + KvOffloadStage + KvMergeStage(D2O) + KvQuantizeStage(KIVI) 4종 + SnapKV/Sparse는 backlog | CachePressureHandler 기존 패턴 마이그레이션 |
| Q9: ForwardSync | **폐기** — Forward는 KV state cache 0이므로 notify 불필요 | 실측: `transformer_layer/forward.rs:195-196` 매 layer call에서 `kv_cache.current_pos()` 재read |
| Q9: StageOutcome | 2 variant (Continue / Stop) — Q20에서 Consumed 추가하여 **3 variant** | KvMutation/WeightMutation enum 폐기 → inference loop이 정책 종류 모름 (SOLID-OCP) |
| Q9: KvMutation/WeightMutation enum | **폐기** | inference loop OCP 정합 |
| Q10: 권한 강제 | M1만 — spec 경고 + trait 주석 (runtime guard / debug_assert / ForwardGuard 모두 폐기) | stage 구현자 책임 — god ctx 인정 원칙 일관 |
| Q11: INV-KVBUNDLE-SYNC | 명문화 — KvBundle::prune 등 mutation method가 호출 후 GPU work 완료 보장 | impl 책임으로 격리 |
| Q11: PipelineStage 주석 | KV phase 제약 풀이 (허용/금지 phase 명시) | doc comment 형식 |
| Q11: PoC test | 4건 | PipelineStage v0 + EvictionPolicyStage / bit-identical / TBT +3% / KIVI mix |
| Q12: error handling | **panic on Err** — dispatcher가 stage.name() + phase + error 노출 후 panic | partial commit 없음, 후속 stage skip 없음 |
| Q13: trait 시그니처 | **R-3** — `Arc<dyn PipelineStage + Send + Sync>` + `on_phase(&self)` + 내부 Mutex | stage struct 정의에 mutable field만 `Mutex<T>` 명시 |
| Q13: dispatcher return | `Option<StopReason>` 반환 (panic on Err 흡수) | |
| Q13: stage trait return | `Result<StageOutcome>` 유지 (작성자 편의 — anyhow::bail!, ? 연산자) | |
| Q14: registry build | **X2** — entry point별 selection (caller가 명시 add) | helper 함수는 후속 도입 (drift 실제 발생 시) |
| Q15: WeightBundle method set | 10 method (Resilience 3 method 포함) — 진행 중 막히면 개선 | swap_layer / enqueue_release / release_pending / secondary_store / set_partition_ratio / set_layer_skip_ratio / switch_device + read 3 |
| Q16: sub-trait 시그니처 | **본 grill에서 풀지 않음** — 별 grill 라운드로 분리 (KVCacheView / LayerView / SecondaryStore / SparsePattern) | Phase α 진입 전 finalize 필요 |
| Q17-1: register_standard_pipeline helper | **YAGNI 폐기** — 처음엔 entry point 안 직접 등록 | drift 우려가 실제 발생 시점에 도입 |
| Q17-2: Manager IPC 위치 | **stage에서 제외 → DecodeLoop owned (별도 처리)** | Manager IPC poll + outbound 발신 모두 DecodeLoop 직접 |
| Q18: ExecutionPlan consume 패턴 | **폐기** — ExecutionPlan ctx field 자체 폐기. 명령은 OneShot stage 객체로 캡슐화 | INV-EXECUTIONPLAN-CONSUME 폐기 |
| Q19: PipelineStatus | **PipelineRegistry와 통합** — 단일 `submit(stage)` method, stage 객체 자체에 정보 보유 | abstraction barrier 없이 단일 객체 |
| Q20: 1회 stage lifecycle | `StageLifecycle` enum (Persistent / OneShot) + `PipelineStage::lifecycle()` method + `StageOutcome::Consumed` variant + dispatcher 자동 GC | Persistent stage가 Consumed 반환 시 debug_assert |
| Q21: Registry mutation | **(M-γ)** `Arc<PipelineRegistry>` + 내부 `Mutex<Vec<Arc<dyn>>>` interior mutability | PoC에서 contention 측정 |
| Q22: Outbound 위치 | **DecodeLoop이 직접 발신** — stage 외부, Manager IPC handler와 대칭 | heartbeat / on_token_generated 모두 DecodeLoop 본체 |
| Q23: PipelineStatus 명명 폐기 | PipelineStatus → PipelineRegistry 통합 (Q19 확정) | 위 Q19 결정 그대로 |

### 14.1 본 grill 2026-05-28 추가 결정 (Q24 ~ Q35) — KvBundle/WeightBundle grill 종결

| Q | 결정 | 비고 |
|---|---|---|
| **Q24: sync model** | **(β) Buffer-level lazy + access-mode-aware** — R1 (host read) / R2 (release) / R3 (backend transition) 자동 sync | 현 인프라 (OpenCL blocking read / CUDA explicit sync / migrate_kv 내부 synchronize) 가 이미 (β) 패턴. 새 sync 인프라 0건. INV-KVBUNDLE-SYNC 폐기. |
| **Q25: primitive granularity** | **(d-1) primitive only — storage-format-agnostic** | KVCacheLayer method 가 token/layer/range granularity, storage-format-agnostic. dtype/codebook/rotation/sparse pattern 을 stage 가 모름. KVCacheLayer impl 이 mechanism 캡슐화. |
| **Q26: Backend composite kernel ownership** | **(A) 유지** (AttentionKernels sub-trait 분리 (A') 보류) | 본 프로젝트 KiviAttentionBackend / GpuScoreAccess 패턴 + frequency × cost 정당화 부족. cheap follow-up = grouping 주석만. |
| **Q27: Mixed storage 허용** | **OK** (layer 별 다른 storage paradigm) | layer 0 = KIVI Q4, layer 1 = TurboQuant, ... |
| **Q28: 호환성 차단 인프라** | **X** (정책-storage 호환 안 되는 조합 compile-time 차단 / fallback 로직 X) | 만나면 panic (Q12 dispatcher 정신 일관). |
| **Q29: KvBundle / WeightBundle trait** | **폐기** | 본 프로젝트가 이미 Bundle 없이 동작 (KVCacheOps + slice). Stage 가 register 시점 layer handle 보관 → ctx layer field 자체 폐기. |
| **Q30: StageContext field 수** | **5 → 3** (`step` / `backend_ext` / `profiler` 유지) | `kv` / `weights` field 폐기. PipelineRegistry 정신 (의도는 stage 객체 안 캡슐화) 100% 정합. |
| **Q31: Layer handle 모델** | **(γ) `Arc<dyn KVCacheLayer>` / `Arc<dyn WeightLayer>` + interior mutability** | Mutation API 가 layer self method (`&self` via RCU/Mutex/Atomic). 본 프로젝트 `LayerSlot::rcu_weights` (slot.rs:158) 패턴의 자연 확장. |
| **Q32: KV dispatch paradigm** | **갈래 2 Trait object 전면 채택** | `KVCacheOps` generic (kv_cache_ops.rs:53 명시 정책) → `KVCacheLayer` trait object. Generic monomorphization 폐기. **kv_cache_ops.rs:53 정책 정면 반전 → ADR-0001 작성 필수**. 갈래 1 (Generic 유지 + mixed storage 포기), 갈래 3 (Hybrid), 갈래 4 (Static enum) 모두 REJECTED. |
| **Q33: Weight dispatch paradigm** | **LayerSlot 유지 + WeightLayer trait wrap** | Weight 도메인은 이미 `Vec<Arc<LayerSlot>>` + RCU 패턴 (γ 정합). Forward path 무변경. WeightLayer trait 신설은 thin wrapper. ~5 file 추가만. KV 와 비대칭. |
| **Q34: Sprint 분리** | **Phase α-W → ADR-0001 → Phase α-K** | Phase α-W (Weight + PipelineStage 인프라, low risk, 2-3주) 먼저 → ADR-0001 작성 → Phase α-K (KV Generic→Trait object, big refactor ~20 file ~1500 LOC, high risk, 4-6주). Risk 분산 + escape hatch. |
| **Q35: LayerDispatch enum 형태** | **Fixed 3 variant (Full / Skip / Partition)** | Dispatch paradigm frequency 낮음 (연 1건 미만). enum + match exhaustive 가독성 우월. Custom hook 은 미래 필요 시 variant 추가. |

#### Q32 갈래 비교 매트릭스 (KV dispatch paradigm)

| 갈래 | 형태 | 결정 | 거부 사유 |
|---|---|---|---|
| 갈래 1 | Generic 유지 (`<C: KVCacheOps>`) + Mixed storage 포기 | **REJECTED** | Q27 (mixed storage 허용) 정면 충돌. KIVI per-layer mix 등 paper-aligned use case 차단. |
| **갈래 2** | **Trait object (`Arc<dyn KVCacheLayer>`)** | **ACCEPTED (본 grill)** | KV/Weight 통일 패턴 + LayerSlot::rcu_weights 자연 확장. vtable cost = layer-step KV write × ~1-3 ns/call (R-G1 PoC 게이트). |
| 갈래 3 | Hybrid (Read = Generic, Write = Trait object) | REJECTED | Read/Write 분리로 인한 API surface 복잡도 폭증. Q25 (primitive only) semantic 깨짐. |
| 갈래 4 | Static enum dispatch (`enum KVCacheLayerVariant { Standard, KIVI, Sparse, ... }`) | REJECTED | 새 paradigm 추가 = enum variant 추가 = OCP 위반. 본 grill OCP 정신 (PipelineStage hook 패턴) 과 직접 충돌. |

---

## 15. 참조

- `arch/inference_pipeline.md` v1 — 본 설계의 선행 v2 7-trait. Phase β 시점에 v2로 재작성.
- `spec/41-invariants.md` — INV-DECODE-STAGE-001~007 등록 + 본 grill 2026-05-28 추가/폐기 (§8.1, 본 문서).
- `docs/adr/0001-kv-dispatch-paradigm.md` — KV dispatch Generic → Trait object 정식 결정 (본 grill 결정 #9).
- `.agent/todos/handoff_pipeline_stage_design_2026_05_27.md` — 이전 grill handoff (본 grill 결정에 의해 supersede).
- `.agent/todos/handoff_kv_weight_grill_2026_05_28.md` — 본 grill handoff (Phase α-W 진입).
- `ARCHITECTURE.md` §13.8 — INV-LAYER 시리즈 정합 (§13.8-O cross-L3 vocabulary trait inversion 참조).
- `arch/weights_pressure_split.md §7.5` — `SecondaryStore` trait inversion (Q24-3, Phase α-W 본격 작업).
- `papers/eurosys2027/_workspace/experiment/swap_overhead_s25.md` — TBT 측정 baseline 참조.
- `engine/src/kv_cache_ops.rs:53` — Generic monomorphization 정책 명시 (본 grill 에서 ADR-0001 로 반전).
- `engine/src/models/weights/slot.rs:158` — `LayerSlot::rcu_weights` 패턴 (본 grill (γ) interior mutability 자연 확장 근거).

---

## 16. 본 grill 2026-05-28 후속 결정 요약 (KvBundle/WeightBundle grill 종결)

본 grill (2026-05-28) 은 이전 grill (2026-05-27, HEAD `6f07af8d`) 의 KvBundle 8 method / WeightBundle 10 method 시그니처 결정을 재검토했다. 결정 12 건 누적:

| # | 결정 | 의미 / 이전 결정과의 관계 |
|---|---|---|
| 1 | **(β) sync model** | Buffer-level lazy + access-mode-aware. R1/R2/R3 자동. **새 sync 인프라 0건 필요**. INV-KVBUNDLE-SYNC 폐기. |
| 2 | **(d-1) primitive only — storage-format-agnostic** | KVCacheLayer method 가 token/layer/range granularity 만 알고 storage paradigm 모름. KVCacheLayer impl 이 mechanism 캡슐화. |
| 3 | **Q7 (A) 유지** (AttentionKernels sub-trait 분리 (A') 보류) | 본 프로젝트 KiviAttentionBackend / GpuScoreAccess 패턴 + frequency × cost 정당화 부족. |
| 4 | **Q8 Mixed storage 허용** | Layer 별 다른 storage paradigm OK. |
| 5 | **Q9 호환성 차단 인프라 X** | 만나면 panic. |
| 6 | **KvBundle / WeightBundle trait 폐기** | 본 프로젝트가 이미 Bundle 없이 동작. Stage register 시점 layer handle 보관 → ctx layer field 자체 폐기. |
| 7 | **StageContext 5 field → 3 field** | `kv` / `weights` 폐기. `step` / `backend_ext` / `profiler` 유지. |
| 8 | **(γ) Layer handle 전환** | KVCacheLayer / WeightLayer trait + interior mutability. Stage 가 `Arc<dyn ...>` 보관. LayerSlot::rcu_weights 자연 확장. |
| 9 | **KV dispatch 갈래 2 Trait object 전면 채택** | KVCacheOps generic → KVCacheLayer trait object. **kv_cache_ops.rs:53 정책 정면 반전 → ADR-0001 작성 필수**. |
| 10 | **Weight dispatch: LayerSlot 유지 + WeightLayer trait wrap** | Weight 도메인은 이미 RCU 패턴 (γ 정합). Forward path 무변경. ~5 file 추가만. **KV 와 비대칭**. |
| 11 | **Sprint 분리: Phase α-W → ADR-0001 → Phase α-K** | Phase α-W (Weight + PipelineStage 인프라, low risk, 2-3주) 먼저. Risk 분산 + escape hatch. |
| 12 | **LayerDispatch enum: Fixed 3 variant (Full/Skip/Partition)** | Dispatch paradigm frequency 낮음 (연 1건 미만). enum + match exhaustive 가독성 우월. |

### 16.1 KV vs Weight 도메인 비대칭 — 본 grill 핵심 발견

§3.5.1 의 비교 표 참조. 두 도메인을 동일 패턴으로 묶지 말고 차이 명시:
- **Weight**: 이미 LayerSlot + RCU + Arc<LayerWeights> 패턴 정착 → WeightLayer trait thin wrap, forward path 무변경, ~5 file 추가.
- **KV**: `KVCacheOps` generic + slice 강결합 (kv_cache_ops.rs:53 명시 정책) → 정책 반전 + ~20 file refactor + bit-identical 검증 risk.

Sprint 분리 (결정 #11) 의 직접 근거.

### 16.2 분기점 결정 로직 추적성 (왜 갈래 X 가 아닌 Y 인가)

- **왜 (A) 가 아닌 (A')**: Q7 결정 — 본 프로젝트 KiviAttentionBackend / GpuScoreAccess 가 이미 backend trait 위에 sub-trait 분리 패턴을 사용 중. AttentionKernels 별 sub-trait 추가는 (a) 새 backend 변형 (예: TurboQuant 전용 kernel) 의 frequency 가 낮고, (b) sub-trait 분리 cost (composite ops + trait object surface) 가 frequency × cost 비례 분석에서 정당화 안 됨. (A) 유지 — Backend trait 가 단일 owner.
- **왜 갈래 1 (Generic 유지 + mixed storage 포기) 가 아닌 갈래 2 (Trait object)**: Q27 (mixed storage 허용) 결정과 직접 충돌. KIVI per-layer mix / D2O variance allocation / SnapKV 등 paper-aligned use case 가 layer 별 다른 paradigm 을 요구.
- **왜 갈래 4 (Static enum dispatch) 가 아닌 갈래 2 (Trait object)**: 새 paradigm 추가 = enum variant 추가 = OCP 위반. 본 grill 의 hook 패턴 (PipelineStage) 이 OCP 정신을 채택했으므로 KV dispatch 도 동일 정신 — Trait object 가 OCP 정합.
- **왜 (γ) layer handle 모델 (interior mutability) 인가**: 본 프로젝트 `LayerSlot::rcu_weights` (slot.rs:158) 이 이미 `&self` mutation via RCU 패턴 정착. (γ) 가 본 프로젝트의 기존 정신 자연 확장 — 새 패러다임 도입 cost 0.

### 16.3 본 grill 변경 추적성 (이전 결정 → 본 grill 변경 사유)

미래 reader 가 "왜 결정이 바뀌었는지" 확인 가능하도록 추적성 보존:

| 이전 결정 (2026-05-27) | 본 grill 변경 (2026-05-28) | 사유 |
|---|---|---|
| KvBundle 8 mutation method | KVCacheLayer 5 method 통합 (primitive only) | (d-1) storage-format-agnostic — stage 가 mechanism 모름. compact(keep, merges) atomic semantic 보장 (D2O Eq.10/11). |
| WeightBundle 10 method | WeightLayer 4 method + SwapMetrics 별 trait | Layer self mutation + measurement axis 분리. INV-LAYER-006 강화. |
| StageContext 5 field (kv/weights 포함) | StageContext 3 field | god ctx 가 PipelineRegistry 정신 위반 — Stage register 시점 layer handle 보관으로 자연 해소. |
| (KV dispatch 명시 결정 없음) | ADR-0001 작성 (Generic → Trait object) | Q8 mixed storage + 5년 시야 paradigm frequency 정당화. kv_cache_ops.rs:53 의 명시 정책 반전이므로 ADR 필수. |
| 단일 Phase α 2-3주 | Phase α-W → ADR-0001 → Phase α-K 분리 | KV/Weight 도메인 비대칭. 큰 risk 위에 큰 risk 쌓지 않음. |

