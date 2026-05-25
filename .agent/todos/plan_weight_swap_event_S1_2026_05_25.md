# Plan: S-1 — WeightSwapEvent + EventSink 통합

**작성**: 2026-05-25
**브랜치**: `worktree-b5_trait_extension`
**진입 문장**: "S-1 진행"
**선행 평가**: Round 1 (SOLID/Layered) + Round 2 (KISS/DRY/perf/paper) + Devil's Advocate

---

## Context

KV/Pressure vs Weight 관리 구조 평가 결과 (SOLID 20 vs 10, Layered 17 vs 9, Δ=18), 격차 top 3 중 **이벤트 인프라(Δ=-4, 정량 60:1)** 가 가장 큰 ROI. 다른 두 격차(DIP `SwapDispatchCtx` 분해 / WeightStrategy)는 후속 평가에서 보류 결정.

### 보류된 옵션 (sprint scope 외)

- **S-2 SwapDispatcher 분해**: 22 필드 × 3 함수 매트릭스 = 16:1:1 J3 편중. IntraForwardRetirer/PhaseAwareRetirer는 ~50 LOC mini-struct. G3 commit(`02cb7106`) LOC 정책 회귀(+150 LOC 추정). backlog 등록.
- **S-3 SwapAlgorithm trait**: paper sprint(LISWAP-1~8) 산물, 확장 빈도 낮음. backlog 등록.
- **WeightStrategy**: Manager-side 부재는 ENG-ALG-214-ROUTE 정당화로 보존. engine-side trait 추출은 ROI 낮음.

### 본 sprint 진입 정당화

| 기준 | 데이터 |
|---|---|
| 정량 격차 | eprintln 60건 (weight) vs 1건 (KV `cache_manager.rs`) |
| paper 호환 | `tbt.jsonl` JSON 별 경로, stderr grep figure 0건 (Devil's Advocate WEAK #3) |
| hot path | dispatcher/retire/secondary alloc 경로, forward inner loop 무관 |
| 검증된 패턴 | `core/events.rs` (CacheEvent 5 variant + EventSink + NoOp/Stderr/Collecting 3 sink) 그대로 확장 |
| blast radius | A안(variant 추가)으로 시그니처 무변경 → KV caller 0 영향 |

---

## 사전 측정 (S-1.0, 30min — 진입 게이트)

### 60건 eprintln 분포 (실측, `grep -rn` 카운트)

| 파일 | 건수 | 분류 (예상) |
|---|---|---|
| `session/decode_fallback/swap_dispatch.rs` | 19 | dispatcher 운영 메시지 |
| `session/decode_fallback/prologue.rs` | 12 | prologue debug 추정 — scope 검토 |
| `models/weights/swap_executor.rs` | 12 | executor 진행 로그 |
| `models/weights/phase_aware_swap.rs` | 8 | retire/dispatch trace |
| `models/weights/secondary_mmap.rs` | 2 | mmap warn |
| `models/weights/layer_object_pool.rs` | 2 | pool warn |
| `models/weights/intra_forward_swap.rs` | 2 | hook fire/finalize |
| `models/weights/rpcmem_secondary.rs` | 1 | warn |
| `models/weights/release_worker.rs` | 1 | worker warn |
| `models/weights/async_swap.rs` | 1 | LISWAP-2 drain |
| **합** | **60** | |

### 분류 작업 (S-1.0 산출물)

각 60건을 4 카테고리로 라벨:

| 카테고리 | 처리 방침 |
|---|---|
| **A. swap lifecycle 이벤트** | emit으로 전환 (plan commit / chunk drain / retire / failed / dispatcher select) |
| **B. 운영자 debug log** | 유지 (`tracing::debug!`로 전환 권고, 본 sprint scope 외) |
| **C. 경고/warn** | 유지 또는 `tracing::warn!`로 전환 (선택) |
| **D. paper 측정 trace** | tbt.jsonl 출력과 무관함 재확인. 무관하면 emit 전환 |

게이트: A 카테고리가 ≥20건 확보되어야 sprint 진행 가치. <10건이면 sprint 중단 + backlog 재검토.

---

## 목표 (검증 가능)

1. **A 카테고리 eprintln → `EventSink::emit(CacheEvent::WeightSwap(...))` 100% 전환**
2. **`WeightSwapEvent` enum 4 variant 신설**: `PlanCommitted` / `ChunkDrained` / `PlanRetired` / `SwapFailed`. 각 variant에 `kind: &'static str` 필드를 두어 발행한 dispatcher(`"Incremental"` / `"IntraForward"` / `"PhaseAware"`)를 식별.
3. **bit-identical 회귀 0** — 호스트 CPU Qwen2.5-1.5b Q4_0 32 토큰 + S25 Adreno OpenCL 32 토큰
4. **신규 trait 0건** — 기존 `EventSink` 재사용
5. **KV side 무영향** — `CacheManager` / `EvictionHandler` / `MemoryStrategy` 등 무수정
6. **일반 추론(swap flag 없음) emit 0건** — 모든 swap dispatcher가 default OFF이므로 swap flag 미명시 시 event 미발행. ChunkDrained latency 측정으로 회귀 확인 가능.

---

## 설계 결정 (1라운드 사전 확정)

### D1. enum 통합 방식 — **A안 채택** (`CacheEvent::WeightSwap(WeightSwapEvent)` variant 추가)

| 옵션 | blast radius | LOC | 결정 |
|---|---|---|---|
| A. `CacheEvent::WeightSwap(...)` variant | 시그니처 무변경, KV caller 0 영향 | +180 | **채택** |
| B. `CacheEvent` → `EngineEvent` rename + variant | rename mechanical (sed + cargo check 반복) | +200 | 후속 sprint micro-PR로 분리 |
| C. `WeightEventSink` trait 별도 | trait 2개 보유, Arc<dyn> 추가 | +150 | 폐기 (ISP 위반) |

**근거**: events.rs L46 `pub enum CacheEvent` 5 variant 패턴 그대로 확장. `EventSink::emit(&self, event: CacheEvent)` (L74) 시그니처 무변경.

**이름 부조화** (`CacheEvent::WeightSwap` 어색함)은 backlog 등록 후 별 PR에서 `CacheEvent` → `EngineEvent` rename.

### D2. `WeightSwapEvent` 4 variant 정의 초안 (kind 필드 통합)

#### 설계 배경

`run_incremental_dispatch`(J3) / `retire_intra_forward`(J4) / `retire_phase_aware`(J5) 3 free function이 매 token tick마다 **모두 호출되며 자체 Option 검사로 일/skip**. 명시적 "dispatcher 선택" 시점 없음. CLI flag(`--swap-incremental-per-tick K` / `--swap-intra-forward` / `--swap-phase-aware`)가 mutually exclusive로 prefill 직전에 정적 결정. → 따라서 "DispatcherSelected"라는 동적 선택 이벤트는 모델링 부정확. **`kind: &'static str` 필드를 4 variant 모두에 두어 발행 dispatcher 식별**.

#### enum 정의

```rust
#[derive(Debug, Clone)]
pub enum WeightSwapEvent {
    /// Plan 확정 시점 — dispatcher가 layer 분할/k_chunk 결정 후.
    PlanCommitted {
        kind: &'static str,                 // ★ "Incremental" | "IntraForward" | "PhaseAware"
        algorithm: &'static str,            // "Sequential" | "Reverse" | "Uniform" | "ImportanceAware" | "AntiImportance"
        ratio: f32,
        k_chunk: usize,
        n_layers: usize,                    // layers.len()만 보관 (alloc 회피)
    },
    /// Chunk 1개 drain 완료.
    ChunkDrained {
        kind: &'static str,                 // ★
        chunk_idx: usize,
        layers_done: usize,
        latency_ms: f32,
    },
    /// Plan 전체 retire (J3 incremental_plan=None / J4·J5 finalize 완료).
    PlanRetired {
        kind: &'static str,                 // 이미 있음
        qcf_actual: Option<f32>,
        token: usize,
        elapsed_ms: f32,
    },
    /// Swap 실패 (rpcmem / mmap / dispatch error / drain timeout).
    SwapFailed {
        kind: &'static str,                 // ★
        reason: String,
        layer: Option<usize>,
    },
}
```

#### kind 발행 매트릭스

| Variant | Incremental (J3) | IntraForward (J4) | PhaseAware (J5) |
|---|---|---|---|
| `PlanCommitted` | ✓ at plan build | ✓ at hook commit | ✓ at dispatcher commit |
| `ChunkDrained` | ✓ per chunk | ✓ per layer (hook fire 시점) | ✓ per phase boundary chunk |
| `PlanRetired` | ✓ at `incremental_plan=None` | ✓ at `hook.finalize` ok | ✓ at `disp.finalize` ok |
| `SwapFailed` | ✓ dispatch err | ✓ finalize err | ✓ finalize err |

**alloc 정책**: hot path 무관이지만 `String`/`Vec` 최소화 (`&'static str` 선호). `SwapFailed.reason: String`만 alloc 허용.

#### 일반 추론(swap flag OFF) 시 emit 없음

```
swap_incremental_per_tick = 0
swap_intra_forward = false
swap_phase_aware = false
```
→ 3 dispatcher 모두 매 token tick 진입 직후 `Option=None` early return → emit 0건. v1-1 baseline 일반 추론 추적은 본 sprint로 변화 없음.

`init.rs:68~70`에서 4 mode가 mutually exclusive로 reject되므로 동시 발행 시나리오 없음 (한 plan 진행 중 발행되는 모든 event는 단일 `kind`).

### D3. `CacheEvent::WeightSwap` variant 추가

```rust
// events.rs:46
pub enum CacheEvent {
    PressureDetected { .. },
    EvictionCompleted { .. },
    PipelineStageExecuted { .. },
    ScoreDiagnostic(ScoreSnapshot),
    ProxyComputed(crate::qcf_types::QcfMetric),
    WeightSwap(WeightSwapEvent),          // ★ NEW
}
```

### D4. `StderrDiagnosticSink::emit` Display 포맷

기존 stderr 메시지와 **호환되지 않아도 됨** (paper 무관, 운영자 grep 의존 0건 확인됨). 단, 진단 정보 손실 0 — 기존 메시지의 변수는 모두 variant 필드로 보존. 포맷은 KV 측 `[CacheEvent]` 패턴 따라 `[WeightSwap]` 접두사:

```
[WeightSwap] PlanCommitted: kind=Incremental, algo=ImportanceAware, ratio=0.50, k=8, n_layers=14
[WeightSwap] ChunkDrained: kind=Incremental, idx=3, layers=14, latency=124.5ms
[WeightSwap] PlanRetired: kind=Incremental, qcf=0.083, token=42, elapsed=890ms
[WeightSwap] PlanCommitted: kind=IntraForward, algo=ImportanceAware, ratio=0.50, k=14, n_layers=14
[WeightSwap] SwapFailed: kind=Incremental, reason="mmap fault", layer=Some(3)
```

`kind` 필드를 모든 출력 첫 컬럼에 노출 → 운영자가 `grep "kind=IntraForward"`로 dispatcher별 trace 추출 가능. 4 dispatcher prefix(`[IncrementalSwap]`/`[WeightSwap]`/`[IntraForwardSwap]`/`[PhaseAware]`) 분산 grep을 단일 `[WeightSwap]` + `kind=` 컬럼으로 통합.

### D5. EventSink 인스턴스 주입 경로

`SwapDispatchCtx`에 `event_sink: Arc<dyn EventSink>` 1 필드 추가 (필드 #23). `SessionInitCtx`/`build_command_executor` 어디서 생성할지 plan에서 결정:

- **권고**: `SessionInitCtx::build_event_sink()` (resilience 활성 시 `StderrDiagnosticSink`, 아니면 `NoOpSink`). `argus_cli`/`legacy_generate` 양쪽 main에서 사용.
- KV측 `CacheManager`가 이미 `Arc<dyn EventSink>` 보유 — 같은 instance 공유 또는 별 instance 검토 (S-1.1 결정).

---

## Sprint 분해

### S-1.0 사전 측정 (30min, 진입 게이트)

- 60건 eprintln 4 카테고리 분류 (위 표). 산출물: `.agent/todos/swap_event_inventory_2026_05_25.md`.
- A 카테고리 ≥20건 확인 → 진행 / <10건 → sprint 중단 + backlog 재검토.
- 검증: `grep -rn "eprintln!" engine/src/models/weights/ engine/src/session/decode_fallback/ | wc -l` = 60 (현재).

### S-1.1 `WeightSwapEvent` enum + `CacheEvent::WeightSwap` variant (1h)

- 파일: `engine/src/observability/events.rs`
- 추가: `WeightSwapEvent` enum **4 variant** (`PlanCommitted` / `ChunkDrained` / `PlanRetired` / `SwapFailed`, 각 variant에 `kind: &'static str` 필드 포함) + `CacheEvent::WeightSwap(WeightSwapEvent)` variant + `StderrDiagnosticSink::emit` match arm 4개.
- 추가 테스트:
  - `test_noop_sink_accepts_weight_swap` (existing pattern)
  - `test_collecting_sink_captures_weight_swap_kinds` (3 kind 모두 발행 검증)
  - `test_stderr_sink_displays_weight_swap_with_kind` (Display 포맷에 `kind=...` 컬럼 포함 검증)
- 게이트: `cargo test -p llm_rs2 --lib events` PASS.

### S-1.2 `SwapDispatchCtx` event_sink 필드 추가 + injection (1h)

- 파일: `engine/src/session/decode_fallback/swap_dispatch.rs`
- 22 필드 → 23 필드: `event_sink: Arc<dyn EventSink>` 추가.
- caller 갱신:
  - `engine/legacy/generate.rs::2155~2192` (SwapDispatchCtx 생성 site)
  - `engine/src/session/init.rs` (있다면) — EventSink 인스턴스 생성
- 게이트: `cargo check` PASS, 호스트 CPU 32 토큰 회귀 0.

### S-1.3 A 카테고리 eprintln → emit 치환 (2h)

- swap_dispatch.rs (J3/J4/J5): `[IncrementalSwap]`/`[WeightSwap]`/`[IntraForwardSwap]`/`[PhaseAware]` 접두사 9건 → emit.
- swap_executor.rs 12건 중 A 카테고리: emit.
- phase_aware_swap.rs 8건 중 A 카테고리: emit.
- intra_forward_swap.rs 2건: emit.
- async_swap.rs 1건 (`[LISWAP-2]`): emit.
- 기타 (prologue.rs 12건, secondary_mmap.rs 2건 등): S-1.0 카테고리 라벨에 따라 결정.
- 게이트: A 카테고리 eprintln 0건 (`grep -rn '\[IncrementalSwap\|\[WeightSwap\|\[IntraForwardSwap\|\[PhaseAware\|\[LISWAP' engine/src/models/weights/ engine/src/session/decode_fallback/` = 0).

### S-1.4 host CPU 회귀 + 빌드 게이트 (30min)

```bash
cargo fmt --all
cargo clippy -p llm_rs2 --bin argus_cli --bin legacy_generate -- -D warnings
cargo test -p llm_rs2 --lib                                # 1159+ PASS, 회귀 0
cargo test -p llm_rs2 --test spec inv_layer                # 8 PASS
./target/release/argus_cli \
  --model-path models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path models/qwen2.5-1.5b/tokenizer.json \
  --prompt "The capital of France is" --num-tokens 32 --greedy \
  --backend cpu --kv-type f16
# v1-1 baseline과 32 토큰 bit-identical (first=12095, final_pos=36)
```

### S-1.5 S25 Adreno OpenCL 디바이스 게이트 (30min)

```bash
python scripts/run_device.py -d galaxy_s25 argus_cli \
  --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
  --prompt-file /data/local/tmp/prompts/capital.txt \
  --num-tokens 32 --greedy --backend opencl --kv-type f16
```

- 통과 조건: v1-1 baseline TBT 33.01 ms/tok 대비 ±2.5% 이내 + bit-identical 출력 + WeightSwap 메시지가 `[WeightSwap]`/`[IncrementalSwap]`/`[IntraForwardSwap]`/`[PhaseAware]` 접두사로 출력됨 (또는 swap 미발생 시 0건).

### S-1.6 단일 commit + handoff + push (30min)

- commit: `refactor(observability): S-1 — WeightSwapEvent enum + EventSink 통합`
- handoff: `.agent/todos/handoff_weight_swap_event_S1_2026_05_25.md` (R1~R6, S-2/S-3 backlog 마킹).
- push: `worktree-b5_trait_extension`.
- notify: `notify-send "llm.rs" "S-1 WeightSwapEvent 완료"`.

---

## Landmines / 미해결

### 1. EventSink 인스턴스 라이프타임

KV측 `CacheManager`가 보유하는 `Arc<dyn EventSink>`와 Weight측 `SwapDispatchCtx`가 보유할 인스턴스의 관계 — 같은 인스턴스 공유? 별도?

- **권고**: 같은 인스턴스 공유 (`SessionInitCtx::event_sink: Arc<dyn EventSink>`). 두 측의 모든 emit이 단일 stderr 스트림으로 통합.
- 단점: KV 이벤트와 Weight 이벤트가 시간순으로 섞임. `[CacheEvent]` vs `[WeightSwap]` 접두사로 구분되므로 grep 용이성 유지.

### 2. `prologue.rs` 12건 — scope 결정 보류

prologue가 weight swap과 직접 관계 없을 가능성. S-1.0에서 라벨링 후 결정. 무관하면 **본 sprint scope에서 제외** (별 sprint).

### 3. `String` alloc 위치

`SwapFailed.reason: String`만 alloc. 다른 variant는 `&'static str` 또는 numeric. 실측에서 alloc 빈도 측정 (S-1.5에서 logcat alloc counter 확인 권고).

### 4. v1-1 baseline 갱신 시점

v1-1 baseline은 default-on resilience + graceful fallback 직후 측정. S-1 sprint가 dispatch path eprintln 60건을 emit으로 바꾸면서 syscall write 빈도 변경 → ±0.5% 이내 변동 예상. 2.5% 게이트 내부면 OK.

### 5. 향후 rename 일치

backlog 등록: `CacheEvent` → `EngineEvent` rename + `Cache(...)` / `WeightSwap(...)` / 향후 `Resilience(...)` 등 확장. 별 micro-PR로 분리.

### 6. (가능성 낮음) 운영자 grep 깨짐

paper 무관(WEAK #3)이지만 사용자/운영자가 stderr grep 스크립트를 가지고 있을 가능성. S-1.4 PR description에 변경된 메시지 포맷 명시.

---

## 비스코프 (명시)

- **SwapDispatcher 분해** — backlog [P3], S-2로 등록 (J3:J4:J5 = 16:1:1 편중, G3 LOC 회귀 위험)
- **SwapAlgorithm trait** — backlog [P3], S-3로 등록 (paper sprint 산물)
- **WeightStrategy** — backlog [P4] 또는 spec 정당화만 (ENG-ALG-214-ROUTE 보존)
- **prologue.rs 12 eprintln** — S-1.0 라벨링에서 weight 무관 확인 시 별 sprint
- **`tracing` 도입** — 운영자 debug log B/C 카테고리는 본 sprint scope 외
- **`CacheEvent` → `EngineEvent` rename** — 별 micro-PR

---

## 검증 게이트 (최종)

| 게이트 | 통과 기준 |
|---|---|
| `cargo fmt --all` | 변경 없음 |
| `cargo clippy -p llm_rs2 --bin argus_cli -- -D warnings` | warnings 0 |
| `cargo test -p llm_rs2 --lib` | 1159+ PASS, 회귀 0 |
| `cargo test -p llm_rs2 --test spec inv_layer` | 8 PASS |
| `python3 scripts/layer_lint.py --baseline` | 신규 violation 0 |
| Host CPU 32 토큰 | v1-1 baseline 대비 bit-identical (first=12095, final_pos=36) |
| S25 Adreno OpenCL 32 토큰 | bit-identical + TBT ±2.5% |
| A 카테고리 eprintln grep | 0건 |

---

## 예상 LOC 변화

| 영역 | Δ LOC | 근거 |
|---|---|---|
| events.rs | +160 | WeightSwapEvent 4 variant (kind 필드 포함) + StderrSink match arm 4개 + 3 tests |
| swap_dispatch.rs | +5 / -19 | event_sink field + 19건 eprintln → emit |
| swap_executor.rs | +5 / -8 (A만) | event_sink ref + emit |
| phase_aware_swap.rs | +3 / -5 (A만) | emit |
| intra_forward_swap.rs | +2 / -2 | emit |
| async_swap.rs | +1 / -1 | emit |
| 호출 사이트 (generate.rs, init.rs) | +10 | Arc<dyn EventSink> 주입 |
| **합** | **+186 / -35 = net +151** | |

목표 net LOC ≤ +200. 넘으면 D2 variant 단순화 검토.

---

## 진입 명령

```
"S-1 진행"
```

또는 즉시 시작:

```bash
# S-1.0 inventory 생성
grep -rn 'eprintln!' engine/src/models/weights/ engine/src/session/decode_fallback/ \
  > $CLAUDE_JOB_DIR/eprintln_inventory.txt
# 60건 출력. 라벨링 시작.
```

---

## 후속 sprint 예고

| Sprint | 항목 | 우선순위 | 의존 |
|---|---|---|---|
| S-1 (본 plan) | WeightSwapEvent + EventSink 통합 | **진입** | — |
| S-1.7 (선택) | `CacheEvent` → `EngineEvent` rename | [P3] | S-1 |
| S-2 (보류) | SwapDispatcher 분해 (J3:J4:J5 = 16:1:1) | [P4] | — (S-1 무관) |
| S-3 (보류) | SwapAlgorithm trait | [P4] | S-2 |
| (별 sprint) | `tracing` 도입 (B/C 카테고리 eprintln) | [P3] | S-1 |
