# Weight Swap — Dynamic Runtime Swap Architecture

> **상태**: Draft v2 (전면 재작성 2026-04-24)
> **작성**: 2026-04-24
> **범위**: Manager 신호 기반 동적 weight swap. 평시 제로 오버헤드, prefill-tail 측정, Arc snapshot 기반 lock-free 교체.
> **대상 스펙**: `spec/33-engine-data.md` §3.17~3.20 (ENG-DAT-090, 092, 093, 094), `spec/32-engine-algorithms.md` §3.12 (ENG-ALG-210~214), `spec/41-invariants.md` §3.13 (INV-121~125).
> **대상 모델**: Llama 3.2 1B (16 decoder layers, no tying), Qwen 2.5 1.5B (28 decoder layers, tying 가능).
> **전제**: GGUF primary + GGUF secondary (dtype 다름). Safetensors는 부차 지원.

## 0. 폐기 기록 (Deprecation Notice)

**폐기일**: 2026-04-24.

**폐기 대상**:
- 정적 per-layer mixed precision 노선 전체.
- `LayerDtypeProfile` TOML 스키마 (`ENG-DAT-091`, **ID 재사용 금지**).
- `quantize_profile` 바이너리 및 offline calibration 흐름.
- CLI 플래그 `--layer-dtype-profile`.
- 구 `arch/weight_swap.md` v1의 Phase A 섹션 전체.

**폐기 사유**:
- 사용자 의도가 **런타임 동적 swap**이었음 (Android 메모리 극한 환경, Manager 신호 기반).
- 정적 프로파일은 배포 번거로움, calibration 파이프라인 필요, prefill 전 로딩 시간 증가 등 실용성 저해.
- Secondary 파일을 디스크에 둔 채 **평시 제로 오버헤드**가 최상위 요구사항.

**승계된 식별자**:
- `ENG-DAT-090` (LoadConfig) — 재정의.
- `ENG-ALG-210` — 의미 재정의 (정적 dispatch → 초기 uniform load).
- `INV-121/122` — 동적 swap 문맥으로 재정의.

---

---

## 1. 아키텍처 개요

### 1.1 전체 컴포넌트 다이어그램

```mermaid
flowchart TB
    subgraph MGR["Manager Process"]
        MS["MemoryStrategy<br/>(engine/src/resilience/strategy/memory.rs 매핑)"]
        SIG["SystemSignal::MemoryPressure"]
    end

    subgraph RES["Resilience Layer (Engine)"]
        RA["ResilienceAction::SwapWeights { ratio }"]
        PIPE["CachePressurePipeline"]
    end

    subgraph HANDLER["WeightSwapHandler (신규)"]
        IC["(1) ImportanceCollector<br/>on-demand active 플래그<br/>engine/src/core/qcf/layer_importance.rs"]
        DEC["(2) SwapDecider<br/>ratio × num_layers<br/>importance 하위 선택"]
        EXE["(3) SwapExecutor<br/>ArcSwap + permutation + madvise"]
    end

    subgraph MODEL["TransformerWeights (ENG-DAT-093)"]
        LS0["LayerSlot[0]<br/>{current_dtype, ArcSwap&lt;LayerWeights&gt;, gen}"]
        LS1["LayerSlot[1]"]
        LSN["LayerSlot[N-1]"]
        MMAP["Arc&lt;SecondaryMmap&gt;<br/>(ENG-DAT-094)"]
        EMB["embedding (primary 고정, swap 제외)"]
        FN["final_norm (primary 고정)"]
        LMH["lm_head (primary 고정)"]
        RGEN["ratio_generation: AtomicU64"]
    end

    subgraph FORWARD["Forward Pass (inference loop)"]
        FP["forward_into()"]
        PLAN["PartitionPlanContext<br/>(ratio_generation 검증, INV-120 재사용)"]
    end

    MS --> SIG
    SIG --> RA
    RA --> PIPE
    PIPE --> HANDLER
    IC --> DEC
    DEC --> EXE
    EXE -->|ArcSwap::swap| LS0
    EXE -->|ArcSwap::swap| LS1
    EXE -->|madvise DONTNEED| MMAP
    LS0 -.reads.-> MMAP
    LSN -.reads.-> MMAP

    FP -->|load snapshot| LS0
    FP -->|load snapshot| LSN
    FP <-.gen mismatch.-> PLAN
    RGEN -.-> PLAN

    style HANDLER fill:#fff3e0
    style MODEL fill:#c8e6c9
    style RES fill:#e1bee7
```

### 1.2 시그널 → Swap 완료 Sequence

```mermaid
sequenceDiagram
    participant Mgr as Manager
    participant Eng as Engine (main)
    participant Handler as WeightSwapHandler
    participant IC as ImportanceCollector
    participant Exec as SwapExecutor
    participant Model as TransformerWeights
    participant Fwd as Forward Loop

    Mgr->>Eng: SystemSignal::MemoryPressure(Critical)
    Eng->>Eng: MemoryStrategy.react() → SwapWeights { ratio: 0.5 }
    Eng->>Handler: pipeline.handle(ctx)

    alt in prefill
        Handler->>IC: activate(target=prefill_last_token)
        Fwd->>IC: per-layer activation divergence 수집 (last token)
        IC-->>Handler: importance_scores
    else in decode, wait next prefill (K=512 budget)
        Handler->>Handler: pending_swap.arrived_at_token = t_now
        loop per token
            Fwd->>Handler: tick(t_now+1)
            alt elapsed < 512 and new prefill arrives
                Handler->>IC: activate at prefill tail
                IC-->>Handler: importance_scores
            else elapsed >= 512
                Handler->>Handler: uniform_select_by_index(ratio)
            end
        end
    end

    Handler->>Exec: execute_swap(layers_to_swap, target_dtype)
    Exec->>Model: ratio_generation.fetch_add(1)
    loop for each layer_idx
        Exec->>Model: build_from_mmap + permutation
        Exec->>Model: slot.weights.swap(new Arc)
        Exec->>Model: slot.current_dtype.store(target)
        Exec->>Model: slot.generation.fetch_add(1)
        Exec->>Model: madvise(DONTNEED) primary pages
    end
    Exec-->>Handler: ActionResult::WeightSwapped { layers, freed, latency }
    Handler-->>Eng: ack
    Eng-->>Mgr: CommandResponse::WeightSwapped (next heartbeat)

    Note over Fwd: 이후 forward는 새 snapshot을 본다.<br/>Plan은 ratio_generation mismatch 시 재빌드.
```

### 1.3 Llama vs Qwen 처리 차이

| 항목 | Llama 3.2 1B | Qwen 2.5 1.5B | 처리 분기 |
|------|--------------|---------------|----------|
| Decoder layer 수 | 16 | 28 | `num_layers`에서 자동 흡수 (ratio 기반) |
| Embedding/lm_head tying | 없음 | 있음 가능 | `LoadConfig`에서 `tie_word_embeddings` 판독, `lm_head` Option 처리 |
| Q/K permutation | GGUF convention | GGUF convention | **공통**, 분기 없음 (gguf.rs:514-534, 677-697) |
| Swap 대상 | decoder block 16개 | decoder block 28개 | `TransformerWeights::layers[i]`만, embedding/lm_head/final_norm은 제외 (ENG-DAT-C11) |
| ratio=0.25 swap 수 | 4 | 7 | `(ratio × num_layers).round()` |
| ratio=0.5 swap 수 | 8 | 14 | |
| ratio=1.0 swap 수 | 16 | 28 | |

**Architectural invariant**: swap 대상은 **decoder block layer만**. 모델별 분기는 `num_layers`와 `lm_head` 유무로 완전 흡수되며, SwapExecutor/SwapDecider 로직 자체는 모델 공통.

## 2. 컴포넌트 상세

### 2.1 컴포넌트: `LoadConfig` (ENG-DAT-090)

**설계 결정**:
- **이원화된 파일 역할**: `primary_source`는 초기 모든 layer 로딩 소스. `secondary_source`는 **초기 로딩에 사용되지 않는다** — metadata 검증과 `SecondaryMmap` 구축에만 사용되며, 실제 byte 접근은 `SwapExecutor` 런타임 단계에서 처음 발생한다.
- **per_layer_dtype 필드 제거**: 이전 정적 노선의 overlay 필드는 폐기. 런타임 dtype은 `LayerSlot::current_dtype`의 atomic state로 표현된다.
- **secondary None = swap 경로 비활성**: 한 파일만 제공되면 `WeightSwapHandler`는 NoOp. 평시 제로 오버헤드.

**인터페이스**:
```rust
// engine/src/models/loader/mod.rs
pub struct LoadConfig {
    pub primary_source: PathBuf,
    pub default_dtype: DType,
    pub secondary_source: Option<PathBuf>,   // swap reservation only
}
// 전제 (pre): primary_source 존재 확인
// 후조건 (post): secondary_source.is_some() ⇒ TransformerWeights::secondary_mmap.is_some() (INV-125)
//                모든 layer의 current_dtype == default_dtype (초기 상태)
```

---

### 2.2 컴포넌트: `LayerSlot` (ENG-DAT-092)

**설계 결정**:
- **ArcSwap 우선 권장**: `arc_swap::ArcSwap<LayerWeights>`는 lock-free snapshot 교체를 제공. Writer-serialized + reader-wait-free. Mutex 대비 forward hot path에서 zero contention.
- **대안 허용**: `RwLock<Arc<LayerWeights>>` 또는 epoch 기반 custom swap도 INV-121~124 충족 시 허용. **최종 선택은 Senior Implementer PoC에서 decode latency로 결정**.
- **generation counter는 layer-local**: `PartitionPlanContext::ratio_generation`(전역)과 독립. 둘 다 증가하나 관찰 지점이 다름.

**트레이드오프**:

| 구현 | Reader 비용 | Writer 비용 | 메모리 | 복잡도 |
|------|-------------|-------------|-------|--------|
| `ArcSwap<LayerWeights>` | atomic load + Arc clone (wait-free) | RCU 기반, 느린 edge case 존재 | +1 atomic ptr/slot | 중 (외부 crate 의존) |
| `RwLock<Arc<LayerWeights>>` | read lock + clone | write lock | lock 구조체 | 낮음 (std) |
| Custom epoch | wait-free load | epoch GC 필요 | epoch 추적 | 높음 |

**인터페이스**:
```rust
pub struct LayerSlot {
    pub current_dtype: AtomicDType,          // or AtomicU8 wrapping DType discriminant
    pub weights: ArcSwap<LayerWeights>,      // 권장; 대안 허용
    pub secondary_mmap_handle: Option<Arc<SecondaryMmap>>,
    pub generation: AtomicU64,
}
// 전제: weights의 dtype == current_dtype (INV-124 불변)
// 후조건: swap 후 generation += 1, 신규 weights와 current_dtype 원자 단위로 갱신
```

---

### 2.3 컴포넌트: `TransformerWeights` (ENG-DAT-093)

**설계 결정**:
- **Cross-layer tensor 분리**: embedding/final_norm/lm_head는 `Arc<Tensor>`로 직접 보유. 이들은 swap 대상이 아니므로 `LayerSlot` 래핑 불필요.
- **secondary_mmap은 최후 소유권**: `TransformerWeights`가 `Arc<SecondaryMmap>`의 "keeper". 모든 `LayerSlot::secondary_mmap_handle`은 여기서 clone된 Arc를 공유. INV-125를 구조적으로 보장.
- **ratio_generation은 Plan 재빌드 트리거**: 기존 `PartitionPlanContext::ratio_generation`(INV-120)과 **의미 통합**. Plan stale 감지 메커니즘 단일화.

**인터페이스**:
```rust
pub struct TransformerWeights {
    pub layers: Vec<LayerSlot>,
    pub embedding: Arc<Tensor>,
    pub final_norm: Arc<Tensor>,
    pub lm_head: Option<Arc<Tensor>>,   // tie_word_embeddings = true면 None
    pub secondary_mmap: Option<Arc<SecondaryMmap>>,
    pub ratio_generation: AtomicU64,
}
```

---

### 2.4 컴포넌트: `SecondaryMmap` (ENG-DAT-094)

**설계 결정**:
- **Read-only mmap**: `memmap2::Mmap` (아님 `MmapMut`). 파일은 절대 수정 대상 아님.
- **Layer tensor 인덱스 사전 구축**: open 시 GGUF header 1회 파싱으로 `layer_index: Vec<LayerTensorSlice>` 완성. 이후 lookup은 O(1).
- **Lazy 접근**: mmap은 열려있지만 page-in은 커널이 first-touch 시 수행. `SwapExecutor` 첫 호출 시 IO가 발생.

**인터페이스**:
```rust
pub struct SecondaryMmap {
    pub mmap: memmap2::Mmap,
    pub layer_index: Vec<LayerTensorSlice>,   // indexed by layer_idx
    pub cross_layer_offsets: HashMap<String, (u64, u64, DType)>,
}
pub struct LayerTensorSlice {
    pub tensors: HashMap<String /* subname */, (u64 /* offset */, u64 /* len */, DType, Vec<usize> /* shape */)>,
}
```

---

### 2.5 컴포넌트: `WeightSwapHandler` (ENG-ALG-214)

**설계 결정**:
- **`CachePressureHandler` 구현**: 기존 pipeline trait을 준수하여 `CachePressurePipeline`에 등록 가능. KV `SwapHandler`(ENG-ALG-092)와 **독립 handler**로 나란히 동작.
- **HandlerContext 확장**: `swap_weights_ratio: Option<f32>` 필드 추가. Pipeline이 Resilience에서 받은 ratio를 context에 주입.
- **측정-결정-실행 3단계**: (1) ImportanceCollector 활성화/결과 수신, (2) SwapDecider, (3) SwapExecutor. 각 단계는 분리된 struct로 테스트 용이성 확보.

**인터페이스**:
```rust
pub struct WeightSwapHandler {
    weights: Arc<TransformerWeights>,
    collector: Arc<Mutex<ImportanceCollector>>,  // on-demand active 플래그 포함
    already_swapped: Mutex<HashSet<usize>>,
    pending_swap: Mutex<Option<PendingSwap>>,
    fallback_k: u64,   // default 512
}

impl CachePressureHandler for WeightSwapHandler {
    fn handle(&self, ctx: &mut HandlerContext) -> ActionResult {
        let Some(ratio) = ctx.swap_weights_ratio else { return ActionResult::NoOp };
        // ImportanceCollector 활성화 or wait next prefill or uniform fallback
        // SwapDecider → SwapExecutor
    }
}
```

---

### 2.6 컴포넌트: `ImportanceCollector` on-demand 확장

**설계 결정**:
- **기존 코드 재사용**: `engine/src/core/qcf/layer_importance.rs`의 `ImportanceCollector`/`ImportanceTable` 그대로 사용.
- **`active: AtomicBool` 플래그 추가**: 기본값 `false`. Hot path에서 `active.load(Relaxed)` 한 번으로 조기 반환하여 **평시 제로 오버헤드** 달성.
- **Prefill-tail 측정**: `active == true`이고 현재 토큰 == `tail_target_token`일 때만 divergence 수집.

**처리 흐름**:

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> ArmedPrefill: SwapWeights signal<br/>received in prefill
    Idle --> ArmedDecode: SwapWeights signal<br/>received in decode
    ArmedPrefill --> Collecting: reach prefill_last_token
    Collecting --> Idle: finalize + SwapExecutor
    ArmedDecode --> ArmedPrefill: new prefill starts<br/>(within K=512 tokens)
    ArmedDecode --> UniformFallback: K=512 tokens elapsed
    UniformFallback --> Idle: SwapExecutor (uniform)
```

---

### 2.7 컴포넌트: `SwapExecutor` (ENG-ALG-211)

**설계 결정**:
- **Per-layer 순차 실행**: 병렬 swap은 madvise 힌트 충돌과 IO 스파이크 우려로 배제. 순차가 총 latency에 더 유리 (측정으로 재확인).
- **Q/K permutation 재사용**: primary loader(`gguf.rs:514-534, 677-697`)의 permutation 함수를 `SwapExecutor`가 직접 호출. dtype에 무관하므로 분기 없음.
- **madvise 2단계**: step (c) `ArcSwap::swap` 직후 old Arc에 잡힌 primary 페이지 힌트 전달. old가 forward에 잡혀 있으면 drop까지 지연되며, 최종 회수는 커널 판단.

**예외 처리**:

| 조건 | 처리 | 스펙 |
|------|------|------|
| `secondary_mmap == None` | NoOp 반환 | ENG-DAT-C09 |
| layer_idx 범위 밖 | skip (NoOp for that layer) | ENG-DAT-C08 |
| 이미 swap된 layer | skip | ENG-ALG-211 |
| permutation 실패 | panic (logic bug) | — |
| madvise EINVAL | 로그 후 계속 (수치 결과는 유지) | ENG-ALG-C05 |

---

### 2.8 컴포넌트: `ResilienceAction::SwapWeights` (shared crate)

**설계 결정**:
- **shared crate의 `ResilienceAction` enum에 variant 추가**: `SwapWeights { ratio: f32 }`. ratio는 `[0.0, 1.0]` clamp.
- **MemoryStrategy 기본 매핑**: `MemoryPressure::Critical → SwapWeights { ratio: 0.5 }`, `Emergency → SwapWeights { ratio: 1.0 }`를 정책 기본값으로 제안. 실제 값은 Manager의 LuaPolicy에서 조정 가능.
- **프로토콜 호환성**: `#[serde(default)]` 필드 추가 원칙(INV-028) 준수. 기존 verify 하네스는 variant 미인지 시 무시.

**인터페이스**:
```rust
// shared/src/resilience.rs (또는 기존 파일)
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum ResilienceAction {
    // ... 기존 variant ...
    SwapWeights { ratio: f32 },   // [0.0, 1.0]
}
```

---

## 3. Config / CLI

| 키/플래그 | 타입 | 기본값 | spec 근거 |
|-----------|------|--------|-----------|
| `--model-path` | String | (기존) | ENG-DAT-070 |
| `--model-path-secondary` | `Option<String>` | None | ENG-DAT-090 |
| `--force-swap-ratio` | `Option<f32>` | None | Debug hook. Manager 없이 prefill 종료 시 `SwapWeights { ratio }` 직접 트리거. |

---

## 4. 테스트 요구사항

| 테스트 대상 | 위치 | 스펙 |
|-------------|------|------|
| LoadConfig secondary reservation | `engine/tests/spec/test_eng_dat_090_load_config.rs` | ENG-DAT-090 |
| LayerSlot atomic swap | `engine/tests/spec/test_eng_dat_092_layer_slot.rs` | ENG-DAT-092, INV-124 |
| TransformerWeights 구조 | `engine/tests/spec/test_eng_dat_093_transformer_weights.rs` | ENG-DAT-093 |
| SecondaryMmap layer index | `engine/tests/spec/test_eng_dat_094_secondary_mmap.rs` | ENG-DAT-094 |
| 초기 uniform 로딩 | `engine/tests/spec/test_eng_alg_210_initial_load.rs` | ENG-ALG-210 |
| SwapExecutor end-to-end | `engine/tests/spec/test_eng_alg_211_swap_executor.rs` | ENG-ALG-211 |
| ImportanceCollector on-demand 활성화 + K=512 fallback | `engine/tests/spec/test_eng_alg_212_importance_activation.rs` | ENG-ALG-212 |
| SwapDecider ratio 계산 + already_swapped 제외 | `engine/tests/spec/test_eng_alg_213_swap_decider.rs` | ENG-ALG-213 |
| WeightSwapHandler 통합 (manual trigger) | `engine/tests/spec/test_eng_alg_214_weight_swap_handler.rs` | ENG-ALG-214 |
| Forward 재진입 안전성 (stress 10K+) | `engine/tests/spec/test_inv_121_swap_reentrancy.rs` | INV-121 |
| Mixed precision 정확성 (Llama + Qwen, ratio 0.25/0.5/1.0) | `engine/tests/spec/test_inv_122_mixed_precision.rs` | INV-122 |
| ArcSwap atomicity (lock-free reader/writer) | `engine/tests/spec/test_inv_123_swap_atomicity.rs` | INV-123 |
| LayerSlot current_dtype 일관성 | `engine/tests/spec/test_inv_124_slot_dtype_consistency.rs` | INV-124 |
| SecondaryMmap lifetime 보장 | `engine/tests/spec/test_inv_125_secondary_mmap_lifetime.rs` | INV-125 |
| SwapWeights serde round-trip | `shared/tests/spec/test_msg_080_swap_weights.rs` | MSG-080 |
| EngineCommand SwapWeights 처리 | `shared/tests/spec/test_msg_081_swap_cmd.rs` | MSG-081 |

---

## 5. Phase 실측 계획 (Llama + Qwen)

| 메트릭 | Llama 3.2 1B | Qwen 2.5 1.5B | 측정 도구 |
|--------|--------------|---------------|----------|
| PSS 감소 (ratio=0.25) | target ≥ 6% | target ≥ 6% | /proc/self/smaps_rollup |
| PSS 감소 (ratio=0.5) | target ≥ 12% | target ≥ 12% | |
| PSS 감소 (ratio=1.0) | target ≥ 25% | target ≥ 25% | |
| Swap latency (per layer) | < 50 ms | < 50 ms | ActionResult::WeightSwapped.latency_ms |
| TBT 증가 (swap 직후 토큰) | < 20% | < 20% | `Decode: X ms/tok` 로그 |
| INV-122 충족 여부 | pass | pass | test_inv_122_mixed_precision.rs |

실측 환경: Galaxy S25 (Android), OpenCL backend. `run_device.py -d s25` 경유. 6T 스레드 설정.

---

## 6. 알려진 미결 사항

1. **Arc snapshot 최종 구현**: ArcSwap vs RwLock vs custom. Senior Implementer PoC의 decode TBT 측정으로 결정 (스펙은 ArcSwap 권장하되 대안 허용).
2. **K=512 fallback 값**: 실측 후 조정 여지. Prefill 빈도가 낮은 워크로드에서 더 큰 값이 유리할 가능성.
3. **Secondary 파일 open 실패 정책**: `LoadConfig::secondary_source`가 Some이나 파일 부재 시 (a) 에러로 중단 vs (b) warning 후 primary-only 진행 중 어느 쪽이 기본? 현재 초안은 (a). 최종 결정은 `generate` CLI 사용자 경험 검토 필요.
4. **Manager 측 정책 조정**: `MemoryPressure::Critical → SwapWeights { ratio: 0.5 }`의 ratio는 정책 기본값. LuaPolicy에서 override 가능하게 설계할지 여부는 Manager팀 결정.
5. **Backend별 신규 Buffer 래핑 경로**: Swap 후 새 `LayerWeights` 생성 시 `rewrap_weights_for_dual_access()` 호출 타이밍. 현 초안은 `SwapExecutor` 내에서 ArcSwap::swap 전에 완료. OpenCL 백엔드에서 zero-copy 보장 확인 필요.

---

## 7. 변경 이력

- **2026-04-24 (v2, 전면 재작성)**: 정적 per-layer mixed precision 노선 폐기. Manager 신호 기반 동적 swap으로 전환. ENG-DAT-091 + `quantize_profile` + `--layer-dtype-profile` 제거. LayerSlot/TransformerWeights/SecondaryMmap/WeightSwapHandler 신규. INV-123~125 추가.
- 2026-04-24 (v1, 초안, **폐기**): Phase A 정적 per-layer mixed precision 설계. ENG-DAT-090/091, ENG-ALG-210, INV-121/122 초안.
