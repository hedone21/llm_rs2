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

## 2. 선결 결정사항 (사용자 확인 필요)

Step 4 진입 전 다음 5건은 plan 분기점이라 사용자가 결정해야 한다. 모두 권장안 있음.

### Q1. "shared/" 위치 — engine 안에 신설 vs 기존 `llm_shared` 크레이트로 이전

ARCHITECTURE.md §13.4 매핑은 `core/backend.rs` → `shared/backend.rs`로 적었지만 "shared"가 어딘지 모호:
- **(A) `engine/src/shared/`** (engine 안 신규 디렉토리). 권장. 이유: 현 `llm_shared` 크레이트(`shared/src/`)는 IPC wire-format 전용으로 dependency hygiene가 명확. engine-internal Tensor/Buffer/Backend trait을 wire 크레이트에 섞으면 manager 등 다른 크레이트의 빌드 footprint가 폭증한다. 또한 §3-D-b가 `memory/`를 engine 내부에 둔 선례.
- **(B) `engine/src/` top-level** (예: `engine/src/backend.rs`, `engine/src/tensor.rs`). memory/처럼 그대로. lib.rs 진입점만 늘어남.
- **(C) `llm_shared` 크레이트로 이동**. ARCHITECTURE 매핑 문자 그대로. wire-format 크레이트가 거대화 + non-pure types 혼입.

**권장**: **(A)**. (B)도 합리적. (C)는 비권장 (별도 backlog 가능).

### Q2. sub-sprint 분할 단위 — 8개 권장

본 Step은 단일 PR로 진행 시 ~983건 import + LAYER_RULES + baseline 동시 갱신이라 리뷰가 불가능. 8 sub-sprint로 분할:

| Sprint | 범위 | 영향 import | 위험 |
|---|---|---|---|
| **4-A** | `core/{backend,tensor,buffer,memory(_buf),shape,quant,thread_pool,math_utils}` → `shared/` (Q1 답에 따라 위치 확정) | ~400 | M |
| **4-B** | `core/qcf/` → `shared/qcf/` | ~30 | L |
| **4-C** | `core/{sampling,skip_config,speculative,attention_scores}` → `inference/` | ~50 | L |
| **4-D** | `core/{kv_cache,kivi_cache,kv_migrate,cache_manager,eviction,pressure,offload}` → `pressure/` | ~250 | M |
| **4-E** | `models/` → `inference/models/` | ~181 | M |
| **4-F** | `layers/` → `inference/layers/` (단 `layers/tensor_partition.rs`는 `shared/`로) | ~66 | L |
| **4-G** | `core/{events,rss_trace}` → `observability/`, `core/{sys_monitor,gpu_yield}` → `resilience/`. **§13.8-C** `core/chat_template.rs` 분배 + V-11 해소 | ~30 | L |
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

진입 전 사용자에게 Q1~Q5 확인 받기:

1. **Q1** "shared/" 위치 → 권장 (A) `engine/src/shared/`
2. **Q2** sub-sprint 순서 → 권장 4-A → B → C → D → E → F → G → H
3. **Q3** `models/weights/` 분할 → 권장 (α) 통째 이동, 분할은 backlog
4. **Q4** `eval/` 위치 → 권장 (J) `observability/eval/`, V-28/V-29는 backlog
5. **Q5** `core/sampling.rs` 위치 → 권장 (K) `inference/sampling.rs`

5건 확인 후 "Step 4-A 진행"으로 진입.

---

## 10. 참조 문서

- `ARCHITECTURE.md §13.1~13.8` — Step 4 Migration Plan 본문
- `ARCHITECTURE.md §13.4` — Directory Migration Map (현재→목표 매핑표)
- `ARCHITECTURE.md §13.5 Resolution Log` — Step 3 종결 결과 (2026-05-20)
- `scripts/layer_lint.py` LAYER_RULES — 갱신 대상
- `engine/tests/spec/inv_layer_baseline.json` — 286 entries (Step 3-F 후)
- `/home/go/.claude/plans/proud-strolling-whale.md` — Step 3 B안 plan (Step 4는 별도 plan 작성 권장)
