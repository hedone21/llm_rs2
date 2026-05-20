# Migration Step 4 진입 handoff — L3 재배치 (`core/` → `pressure/`/`inference/`/`shared/`)

**작성**: 2026-05-20
**master HEAD**: `c037b991` (Step 3-F 종결)
**선행**: Migration Step 3 완료 (3-A → 3-D-a → 3-D-b → 3-B → 3-E → 3-F), baseline 309 → 286, INV-LAYER-002 9건 → 1건.

---

## 1. Step 4 목표 요약 (ARCHITECTURE.md §13.7)

> `core/` → `pressure/`, `inference/` rename only
> - `core/{kv_cache, kivi_cache, kv_migrate, cache_manager, eviction, pressure, offload}` → `pressure/`
> - `models/`, `layers/` → `inference/`
> - `core/{backend, tensor, buffer, memory, shape, quant, thread_pool, qcf, sampling, math_utils, skip_config, speculative, attention_scores}` → 분류에 따라 `shared/` 또는 `inference/`
> - §13.8-C: `core/chat_template.rs` → `inference/chat_template.rs` + `inference/models/<arch>/chat_template.rs`. V-11 해소
> - rename only, 로직 변경 없음

**핵심 사실**: "rename only"지만 영향 footprint는 다음과 같다 (실측 2026-05-20):
- `crate::core::*` 참조 **736건**
- `crate::models::*` 참조 **181건**
- `crate::layers::*` 참조 **66건**
- **총 ~983건 import** 갱신 + `layer_lint.py LAYER_RULES` 갱신 + baseline JSON 재생성

---

## 2. 선결 결정사항 — **확정됨 (2026-05-20)**

5건 사용자 확정 + 1건 잠정 (재논의 시점 명시).

| Q | 결정 | 비고 |
|---|---|---|
| **Q1** L2 abstraction 위치 | **(B) `engine/src/` top-level 분산** | `memory/`와 일관. `llm_shared` 크레이트와 이름 충돌 회피 |
| **Q2** sub-sprint 순서 | **L2 먼저: 4-A → B → C → D → E → F → G → H** | import sed 1회로 최소화 |
| **Q3** `models/weights/` 분할 | **(α) 통째 이동 + 분할은 별도 sprint backlog** | 단 "weights 자체 리팩토링은 4-E 진입 시 재논의" — Step 4 sub-sprint 4-E 시작 시 옵션 재검토 |
| **Q4** `eval/` 격상 | **(I) 본 Step 4-G에 흡수** | `eval/` → `session/eval/`. V-28 자연 해소, V-29 잔존 backlog |
| **Q5** `core/sampling.rs` 위치 | **(K) `inference/sampling.rs`** | inference 단방향 사용 |

### Q1 확정 — `engine/src/` top-level 분산

`llm_shared` 크레이트(`shared/src/`)는 IPC wire-format 전용으로 운영 중(engine 101 imports, manager 50 imports, deps: serde/serde_json/memmap2/log/tempfile만). engine-internal L2 추상화를 이전하면 manager가 cudarc/ocl 빌드 의존 → 절대 금지.

`engine/src/` top-level에 분산 — `memory/`(3-D-b 신설), `inference/`(신규), `pressure/`(신규)와 평행:

```
engine/src/
├── backend.rs       (← core/backend.rs)
├── tensor.rs        (← core/tensor.rs)
├── buffer.rs        (← core/buffer.rs)
├── memory_buf.rs    (← core/memory.rs, rename)
├── shape.rs         (← core/shape.rs)
├── quant.rs         (← core/quant.rs)
├── thread_pool.rs   (← core/thread_pool.rs)
├── math_utils.rs    (← core/math_utils.rs)
├── qcf/             (← core/qcf/)
├── tensor_partition.rs (← layers/tensor_partition.rs)
├── memory/          (이미 존재, 3-D-b)
├── inference/       (신규)
├── pressure/        (신규)
├── observability/   (신규, 4-G)
├── backend/
├── resilience/
└── session/
```

**`layer_lint.py` LAYER_RULES 갱신**: top-level prefix(`"backend"`, `"tensor"`, `"buffer"`, ...)를 L2로 분류.

### Q3 확정 — 통째 이동 + 재논의 hook

4-E 진입 시:
- 통째 이동(α) 1차 PASS = `models/weights/` → `inference/models/weights/`
- 분할(β) 옵션 = "weights 자체 리팩토링" — 별도 sprint로 후속 진행. 4-E PR 종결 시 backlog `weights_internal_refactor` 등록.
- 4-E 진입 시점에 weights 안 cross-import 분포 측정 후 최종 결정.

### Q4 확정 — eval → session/eval/ 통합 (4-G에서)

`engine/src/session/eval/runner.rs` 이미 존재 (wrapper 역할). 본 Step 4-G에서 `eval/` 본체 6 파일을 `session/eval/`로 흡수.
- V-28 (eval L3 의존 5건) 자연 해소
- V-29 (eviction_hook.rs:319 OpenCLBackend downcast) **잔존** — backend trait 추출 필요, backlog

### Q5 확정 — inference/sampling.rs

`core/sampling.rs` 호출처 12건 모두 inference forward path. backend-agnostic이지만 도메인 단방향.

### (구) Q1~Q5 옵션 분석 — 위 표에 결정 반영됨. 아래는 참고 (Q1만 일부 보존)

#### Q1 옵션 비교 (archived)

| 옵션 | path | 평가 |
|---|---|---|
| (A) `engine/src/shared/` | engine 안 신규 디렉토리 | **탈락** — `llm_shared` 크레이트와 이름 충돌 |
| **(B)** `engine/src/` top-level | `memory/`와 평행 | **선택** — `memory/` 선례와 일관, top-level 진입점 증가만 |
| (C) `llm_shared` 크레이트로 이전 | ARCHITECTURE 문자 그대로 | **탈락** — manager가 cudarc/ocl 의존, 절대 금지 |

### Q2. sub-sprint 분할 단위 — 8개 권장

본 Step은 단일 PR로 진행 시 ~983건 import + LAYER_RULES + baseline 동시 갱신이라 리뷰가 불가능. 8 sub-sprint로 분할:

| Sprint | 범위 | 영향 import | 위험 |
|---|---|---|---|
| **4-A** | `core/{backend,tensor,buffer,memory→memory_buf,shape,quant,thread_pool,math_utils}` → **`engine/src/` top-level** | ~400 | M |
| **4-B** | `core/qcf/` → `engine/src/qcf/` | ~30 | L |
| **4-C** | `core/{sampling,skip_config,speculative,attention_scores}` → `engine/src/inference/` | ~50 | L |
| **4-D** | `core/{kv_cache,kivi_cache,kv_migrate,cache_manager(→manager.rs),eviction,pressure(→policy/),offload}` → `engine/src/pressure/` | ~250 | M |
| **4-E** | `models/` → `engine/src/inference/models/` (통째, Q3=α) | ~181 | M |
| **4-F** | `layers/` → `engine/src/inference/layers/` (단 `tensor_partition.rs`는 `engine/src/tensor_partition.rs` top-level로) | ~66 | L |
| **4-G** | `core/{events,rss_trace}` → `engine/src/observability/`, `core/{sys_monitor,gpu_yield}` → `engine/src/resilience/`, `core/chat_template.rs` 분배(§13.8-C V-11 해소), **eval/ → session/eval/ 통합(Q4=I V-28 해소)** | ~60 | L |
| **4-H** | `scripts/layer_lint.py` `LAYER_RULES` 갱신 + baseline JSON 재생성 + ARCHITECTURE.md §13.5/§13.7 갱신 + spec test PASS 확인 | 0 (검증) | L |

**권장 순서**: **4-A → 4-B → 4-C → 4-D → 4-E → 4-F → 4-G → 4-H**.
근거: L2(shared)를 먼저 이동해야 L3 도메인이 import path 안정화된 상태에서 이동. 반대로 하면 L3 이동 후 L2 이동 시 또 import 갱신.

### Q3. `models/weights/` 의 분할 — ARCHITECTURE는 분할 명시, 본 Step에선 어떻게?

§13.4가 `models/weights/`를 3 경로로 분할 권장:
- `models/weights/{async_swap,release_worker,swap_executor,phase_aware_swap,intra_forward_swap}` → `pressure/policy/handlers/weight_swap/`
- `models/weights/{slot,secondary_mmap,rpcmem_secondary}` → `pressure/state/weight_slot/`
- `models/weights/layer_object_pool.rs` → 이미 `engine/src/layers/staging_pool.rs` trait + 위치는 그대로(3-B 결정)

옵션:
- **(α)** 4-E에서 `models/` 전체 이동만 (`weights/` 통째로 `inference/models/weights/` 이동). 본질 분할은 **별도 sprint(4-E-2 또는 backlog)**. 권장 — rename only 원칙 유지.
- **(β)** 4-E에서 `weights/`를 위 3분할 동시 수행. PR 거대화 + 동일 도메인 cross-import 위반 가능성.

**권장**: **(α)**. weights 분할은 본 Step 종결 후 별도 sprint.

### Q4. `eval/` 위치 — `observability/eval/` vs `session/eval/`

§13.4가 "V-28/V-29 해소 위해 `session/eval/`로 격상 검토"라고 적었지만 결정 미완. eval은 L3에 깊게 의존하므로:
- **(I)** `session/eval/` (L4) — V-28/V-29 해소 동시 진행
- **(J)** `observability/eval/` — 단순 rename, 위반은 그대로

**권장**: **(J)** + V-28/V-29는 backlog. 이유: Step 4는 "rename only" 원칙. V-28/V-29 해소는 trait 추출이 필요해 별도 설계.

### Q5. `core/sampling.rs` 위치 — `inference/` vs `shared/`

§13.4가 `inference/sampling.rs`로 매핑하지만 `core/sampling.rs`의 SamplingConfig는 backend-agnostic이라 L2로 볼 여지 있음.

- **(K)** `inference/sampling.rs` (§13.4 그대로) — `SamplingConfig`가 inference forward path와 강결합으로 봄
- **(L)** `shared/sampling.rs` — Backend trait처럼 도메인-중립 utility로 봄

**권장**: **(K)**. 실측상 `models/transformer.rs`와 `bin/generate.rs`만 import. inference 단방향 사용.

---

## 3. 영향 footprint (실측, 2026-05-20)

### 3.1 디렉토리/파일별 LOC

```
engine/src/core/        총 28 항목 (모듈 + 디렉토리)
  ├─ backend.rs              [L2 → shared/]
  ├─ buffer.rs               [L2 → shared/]
  ├─ tensor.rs               [L2 → shared/]
  ├─ memory.rs               [L2 → shared/memory_buf.rs (rename)]
  ├─ shape.rs                [L2 → shared/]
  ├─ quant.rs                [L2 → shared/]
  ├─ thread_pool.rs          [L2 → shared/]
  ├─ math_utils.rs           [L2 → shared/]
  ├─ qcf/                    [L2 → shared/qcf/]
  ├─ sampling.rs             [L3-inference → inference/]
  ├─ skip_config.rs          [L3-inference → inference/]
  ├─ speculative.rs          [L3-inference → inference/]
  ├─ attention_scores.rs     [L3-inference → inference/]
  ├─ chat_template.rs        [§13.8-C 분배 → inference/chat_template.rs + inference/models/<arch>/chat_template.rs]
  ├─ kv_cache.rs             [L3-pressure → pressure/state/]
  ├─ kivi_cache.rs           [L3-pressure → pressure/state/]
  ├─ kv_migrate.rs           [L3-pressure → pressure/state/]
  ├─ cache_manager.rs        [L3-pressure → pressure/manager.rs (rename)]
  ├─ eviction/               [L3-pressure → pressure/policy/eviction/]
  ├─ pressure/               [L3-pressure → pressure/policy/handlers/ + pressure/policy/pressure.rs]
  ├─ offload/                [L3-pressure → pressure/state/offload/]
  ├─ events.rs               [observability → observability/]
  ├─ rss_trace.rs            [observability → observability/]
  ├─ sys_monitor.rs          [resilience → resilience/]
  ├─ gpu_yield.rs            [resilience → resilience/]
  └─ mod.rs                  [삭제 또는 잔존 pub use shim]

engine/src/layers/      총 7 항목
  ├─ attention.rs            [→ inference/layers/]
  ├─ hybrid_attention.rs     [→ inference/layers/]
  ├─ llama_layer.rs          [→ inference/layers/]
  ├─ staging_pool.rs         [→ inference/layers/ — 3-B 도입]
  ├─ tensor_partition.rs     [L2! → shared/tensor_partition.rs]
  ├─ transformer_layer/      [→ inference/layers/]
  └─ workspace.rs            [→ inference/layers/]

engine/src/models/      총 6 항목
  ├─ config.rs               [→ inference/models/ (혹은 shared/config.rs — Q5 유사 결정)]
  ├─ llama/                  [→ inference/models/llama/]
  ├─ loader/                 [→ inference/models/loader/]
  ├─ mappers/                [→ inference/models/mappers/]
  ├─ transformer.rs          [→ inference/models/]
  └─ weights/                [→ inference/models/weights/ (Q3 답에 따라 분할 또는 통째)]
```

### 3.2 Import 참조 수 (실측 grep)

| Source | count |
|---|---|
| `crate::core::*` 참조 | 736 |
| `crate::models::*` 참조 | 181 |
| `crate::layers::*` 참조 | 66 |
| **총합** | **~983** |
| `use crate::core::` 단독 라인 | 308 |
| `use crate::models::` 단독 라인 | 66 |
| `use crate::layers::` 단독 라인 | 18 |

### 3.3 호출처 분포 (top-3)

- `engine/src/bin/generate.rs` — V-30 (전 도메인 직접 의존, ~50건)
- `engine/src/models/transformer.rs` — `core/`, `backend/`, `layers/` 다수 import
- `engine/src/core/cache_manager.rs` — eviction handler 다수 import (rename 시 같은 도메인 내부)

---

## 4. 검증 게이트 (sub-sprint마다 동일)

| 게이트 | 의무/선택 | 도구 |
|---|---|---|
| `cargo build --release --workspace` 회귀 0 | 의무 | host CPU + S25 cross + Jetson cross |
| `cargo test --release --lib --skip opencl` 회귀 0 | 의무 | master parity (3-E 기준 1098 passed) |
| `cargo test --release --test spec test_inv_layer` 8/8 PASS | 의무 | INV-LAYER 유지 |
| `python3 scripts/layer_lint.py --baseline ...` 회귀 0 | 의무 | rename 후 baseline 갱신 필요 — 4-H에서 일괄 |
| **S25 bit-identical 32 tok** (Qwen 2.5 1.5B Q4_0 GGUF, greedy) | 4-D / 4-E / 4-F 후 의무 | OpenCL backend e2e |
| Jetson CUDA 32 tok bit-identical | 옵션 | CUDA backend e2e (선택) |
| avg_tbt n=5 회귀 ≤5% | 4-D 후 옵션 | Adreno 14ms baseline |
| `cargo fmt --check` 회귀 ≤ master | 의무 | |

**중요**: rename만 하면 `cargo build` 통과 = 컴파일 측면 안전. 그러나 `mod.rs`의 `pub use` 갱신 누락 시 외부 노출 시그니처가 사라질 수 있음 → `bin/generate.rs` 같은 V-30 호출처가 깨짐. 매 sprint마다 **bin 빌드 확인 필수**.

---

## 5. 위험 매트릭스

| R | 항목 | 영향 | 완화 |
|---|---|---|---|
| R1 | 광범위 import sed의 unintended match (예: `crate::core::` ⊃ `crate::core_xxx`) | M | grep pattern을 `crate::core::` 또는 `crate::models::` 끝 word boundary로 strict 매칭. 매 sprint 후 `cargo build` 즉시 |
| R2 | 동일 모듈 안 `super::` 또는 `self::` import가 rename으로 깨질 수 있음 | L | rename 전 `super::|self::` grep으로 영향 후보 사전 식별. `git mv` 사용 시 `mod.rs` 자동 갱신 안 됨 — 수동 갱신 |
| R3 | `LAYER_RULES`가 sprint 도중 inconsistent 상태 → `layer_lint.py`가 잘못된 위반 보고 | M | 4-H에서 일괄 갱신. 그 전까지 sprint 후 baseline 비교는 임시 LAYER_RULES 패치로 진행 |
| R4 | `tensor_partition.rs`의 L2/L3 위치 — §13.4가 `shared/`로 매핑하지만 실제 코드는 L3 의존 (layers/workspace 등) | M | 4-F에서 tensor_partition만 별도 처리. import 분석 후 결정 |
| R5 | `cache_manager.rs` → `pressure/manager.rs` rename은 path 짧아짐 → 매우 많은 호출처 영향 | M | 4-D에서 단계적 sed (rename → import 갱신 → mod.rs 갱신) |
| R6 | `models/transformer.rs`가 `core/`, `layers/`, `backend/` 모두 의존 → rename 시 cross-sprint conflict | H | sprint 순서 엄격 유지 (4-A 후 4-D 후 4-E). 또는 `models/transformer.rs`만 마지막 sprint에서 단일 갱신 |
| R7 | bin/generate.rs (V-30 monolith)의 ~50건 import — 매 sprint마다 갱신 | H | sed 자동화 + cargo build로 catch. 4-H 종결 시 V-30은 아직 잔존(별도 sprint) |
| R8 | `cargo fmt` regression — rename으로 import 순서가 뒤바뀜 | L | sprint 끝 cargo fmt + diff 확인 (master baseline 동일 유지) |
| R9 | spec test 가 hardcoded path를 사용 — 예: trybuild 안 import path | M | 4-E / 4-F 후 trybuild 테스트 갱신 확인 |
| R10 | `cuda-embedded only` 빌드의 pre-existing `read_allow_boundary_env` cfg 오류와 충돌 | L | Step 3과 동일 — 본 sprint 범위 외 |

---

## 6. Step 4 sub-sprint 진입 명령

각 sprint 완료 시 다음 진입 문장:

1. **"Step 4-A 진행"** — `core/` L2 모듈 → `shared/`
2. **"Step 4-B 진행"** — `core/qcf/` → `shared/qcf/`
3. **"Step 4-C 진행"** — `core/` L3-inference → `inference/`
4. **"Step 4-D 진행"** — `core/` L3-pressure → `pressure/`
5. **"Step 4-E 진행"** — `models/` → `inference/models/` (option α: 통째)
6. **"Step 4-F 진행"** — `layers/` → `inference/layers/` (tensor_partition 분리)
7. **"Step 4-G 진행"** — cross-cutting + chat_template 분배
8. **"Step 4-H 진행"** — LAYER_RULES + baseline + 문서 갱신

---

## 7. 호환성·잔존 backlog (Step 4 외)

- **V-30**: `bin/generate.rs` monolith — Step 4 진행 후에도 잔존. 별도 sprint (`session/` 추출 미완성분 + chat REPL Phase 4-5).
- **V-07/V-19 본질** — Step 3에서 보류. backend trait 추출 필요.
- **V-28/V-29** (eval L3 의존) — Q4=J 선택 시 잔존. 별도 sprint.
- **`models/weights/` 본질 분할** — Q3=α 선택 시 잔존. 별도 sprint.
- **Backend trait `alloc_alias_weight_buffer` 시그니처** (W5=J) — 잔존.
- **AUF dtype_convert vs auf_dtype_convert** — Step 3-A 산물, 통합 검토 가능.

---

## 8. 환경 / 규칙 (Step 3과 동일)

- 언어: 모든 응답 한국어, 기술 용어/코드 식별자 원문 유지
- EnterWorktree: 코드 변경 작업 시 worktree 격리 필수 (sub-sprint마다 1 worktree)
- worktree symlink: `third_party/`, `libs/` (S25 빌드 시)
- 테스트 기본 모델 포맷: GGUF
- Android 벤치 스레드: Galaxy S25 6T만
- TBT metric: avg_tbt (tok0 inclusive)
- 성능 측정: `--profile` 없이
- 완료 시: 자동 commit + `notify-send "llm.rs" "<요약>"`
- `.cl` 커널 수정 없음 (rename only)

---

## 9. 다음 행동

Q1~Q5 사용자 확정 (위 §2 표). 즉시 "Step 4-A 진행"으로 진입 가능.

### Step 4-A 진입 시 첫 액션

1. `EnterWorktree(name="step4_a_l2_toplevel")`
2. master rebase + `third_party`/`libs` symlink
3. 이동 대상 8 모듈 (`core/{backend,tensor,buffer,memory,shape,quant,thread_pool,math_utils}.rs`)을 `git mv`로 `engine/src/` top-level로
4. `engine/src/core/mod.rs`에서 8 `pub mod` 제거 + `engine/src/lib.rs`에 8 `pub mod` 추가 (단 `memory` → `memory_buf` rename)
5. `crate::core::backend` → `crate::backend_trait` 또는 `crate::backend`(기존 `backend/` 디렉토리와 충돌 → **rename 필요**)
   - **주의**: `crate::backend`는 이미 backend 디렉토리(L1)가 사용 중. `core::backend`(trait 정의)는 path 충돌. 해결안: trait 모듈을 `crate::backend_trait` 또는 `crate::backend::api` 같은 이름으로 변경. 또는 `backend/` 디렉토리를 `backends/`로 rename.
   - **결정 필요**: 4-A 시작 시 사용자에게 확인 (옵션: trait 모듈을 backend/api.rs로 통합 vs backend_trait.rs로 격리 vs backend 디렉토리를 backends/로 rename)
6. import sed 일괄 (~400건):
   - `crate::core::backend::` → `crate::backend_trait::` (또는 결정에 따라)
   - `crate::core::tensor::` → `crate::tensor::`
   - `crate::core::buffer::` → `crate::buffer::` — 그러나 `crate::buffer`는 이미 `engine/src/buffer/` (3-D-c 잔존 SliceBuffer) 사용 중. **여기도 path 충돌** ⚠️
   - `crate::core::memory::` → `crate::memory_buf::` (rename to disambiguate from `engine/src/memory/`)
   - `crate::core::shape::` → `crate::shape::`
   - `crate::core::quant::` → `crate::quant::`
   - `crate::core::thread_pool::` → `crate::thread_pool::`
   - `crate::core::math_utils::` → `crate::math_utils::`
7. `cargo build --release --workspace` 회귀 0
8. `cargo test --release --lib --skip opencl` master parity
9. `cargo fmt --check` master baseline 동일
10. commit + ff-merge + notify

### Path 충돌 예측 (4-A 진입 직후 결정 필요)

`engine/src/` top-level에 이미 존재하는 모듈명과 `core/` 내 모듈명이 충돌:

| 충돌 후보 | 현 top-level | core/ 안 | 해결안 |
|---|---|---|---|
| `backend` | `backend/` 디렉토리 (L1 backend impl) | `backend.rs` (Backend trait) | **(가)** trait를 `backend_trait.rs`로, **(나)** `backend/`를 `backends/`로, **(다)** trait를 `backend/api.rs`로 통합 |
| `buffer` | `buffer/` 디렉토리 (Step 3-D-c 잔존 — slice.rs만 남음) | `buffer.rs` (Buffer trait + DType) | **(가)** trait를 `buffer.rs` top-level + `buffer/` 디렉토리 통합, **(나)** trait를 `buffer_trait.rs` |
| `memory` | `memory/` 디렉토리 (Step 3-D-b 신설) | `memory.rs` (Memory trait) | **§2 표에서 이미 결정**: `memory.rs` → `memory_buf.rs` rename |

권장:
- backend 충돌: **(다) `engine/src/backend/api.rs`로 통합** — `pub use api::*` from backend/mod.rs. trait 정의가 backend 디렉토리 안에 있어 의미적 정합.
- buffer 충돌: **(가) `engine/src/buffer/api.rs`로 통합** — 동일 패턴. `engine/src/buffer/{api.rs, slice.rs}`.
- memory 충돌: **memory_buf.rs (이미 §2 결정)**.

4-A 진입 시 위 권장안 채택 또는 사용자 재결정.

---

## 10. 참조 문서

- `ARCHITECTURE.md §13.1~13.8` — Step 4 Migration Plan 본문
- `ARCHITECTURE.md §13.4` — Directory Migration Map (현재→목표 매핑표)
- `ARCHITECTURE.md §13.5 Resolution Log` — Step 3 종결 결과 (2026-05-20)
- `scripts/layer_lint.py` LAYER_RULES — 갱신 대상
- `engine/tests/spec/inv_layer_baseline.json` — 286 entries (Step 3-F 후)
- `/home/go/.claude/plans/proud-strolling-whale.md` — Step 3 B안 plan (Step 4는 별도 plan 작성 권장)
