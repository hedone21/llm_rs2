# Handoff: precision swap §13.8-O 갈래 1 (WeightSwapDispatch trait 격상) 진입

**작성**: 2026-05-26
**HEAD**: `f89f6669 Merge pull request #13 from hedone21/worktree-b5_trait_extension`
**브랜치 / Worktree**: `master`
**작성자**: main session (grill-me skill 라운드)

**다음 세션 진입 문장**: "precision swap 갈래 1 옵션 결정 후 진행"

---

## TL;DR

`pressure/weight_swap_handler.rs:21,23,25` 의 §13.8-O `cross_l3_vocabulary` 위반 3건 (ModelConfig + SwapExecutor + LayerSlot/SecondaryMmap) 해소를 위한 옵션 분석 완료. backlog 명세("WeightSwapHandler를 models/weights로 이동")와 arch §6.3("models/weights를 pressure/로 통째 이동")이 정반대 방향으로 충돌, **arch 우선** 결론. 단순 이동만 하면 weight_swap 16 파일의 외부 의존(`models::config`, `models::transformer`, `models::loader::gguf`, `layers::transformer_layer`)이 모두 §13.8-O 신규 marker로 전환되어 **marker 적자** 발생. 코드 변경 0건. 옵션 결정 대기 후 진입.

---

## 진행 상태

### Task

| ID | 상태 | 작업 | Commit |
|---|---|---|---|
| #1 | ✅ completed | §13.8-O 3 marker 위치 식별 + 본질 분석 (cross-L3 vocabulary 위반의 사유) | (grill만, 코드 변경 0) |
| #2 | ✅ completed | sibling cascade 분석 — weight_swap 16 파일 통째 이동 필요 확인 | - |
| #3 | ✅ completed | KV cache 패턴 비교 — case 2 (호스트 재배치) 정합 결론 | - |
| #4 | ✅ completed | 4건 사전 작업 식별 (ModelConfig / WeightSwapTarget trait / GGUF utility / TransformerLayer inversion) | - |
| **#5** | **⏳ 이번 세션 진입 대상** | 옵션 결정 (3 후보 중 1) | - |
| #6 | ⏳ blocked-by #5 | 선택 옵션에 따른 architect 검토 + spec 갱신 | - |
| #7 | ⏳ blocked-by #6 | implementer/senior implementer 위임 + 게이트 검증 | - |

### §13.8-O 갈래 1의 본질

| 위치 | 위반 내용 |
|---|---|
| `engine/src/pressure/weight_swap_handler.rs:21` | `use crate::models::config::ModelConfig;` (L3 pressure → L3 inference) |
| `engine/src/pressure/weight_swap_handler.rs:23` | `use crate::models::weights::swap_executor::SwapExecutor;` |
| `engine/src/pressure/weight_swap_handler.rs:25` | `use crate::models::weights::{LayerSlot, SecondaryMmap};` |

### 외부 의존 (cascade — 이동 시 신규 위반)

`engine/src/models/weights/*.rs` 16 파일의 외부 의존:

| 의존 대상 | Layer | 의존 파일 |
|---|---|---|
| `models::config::ModelConfig` | L3 inference | rpcmem_secondary, secondary_mmap, swap_executor |
| `models::transformer::TransformerModel` | L3 inference | swap_executor |
| `models::loader::gguf::{GgufFile, ggml_type_to_dtype, tensor_byte_size, qk_permute_shape, unpermute_qk_rows}` | L3 inference | backing, rpcmem_secondary, secondary_mmap, swap_executor |
| `models::loader::auf::secondary::*` | L3 inference | secondary_mmap |
| `layers::transformer_layer::TransformerLayer` | L3 inference | slot, swap_executor, async_swap, phase_aware_swap, release_worker, layer_object_pool |

**모두 concrete struct 의존, trait 격상 0건**.

### 측정 / 검증

| 항목 | 값 |
|---|---|
| 코드 변경 | 0 (grill만) |
| 테스트 게이트 | 미실행 |
| `models/weights/` 디렉토리 | 16 파일, ~430 KB |
| `models/weights/*` 외부 caller | 17건 (`grep -rn "use crate::models::weights" engine/src \| grep -v "engine/src/models/weights"`) |

---

## 다음 작업

### 액션

**#5: 옵션 결정** — 3 후보 중 선택:

#### 옵션 C4 + 사전 작업 4건 완전 묶음 (7~11일)
- (a) `ModelConfig` → `shared/`로 L2 격상 (V-24 해소) — 1일
- (b) `WeightSwapTarget` trait 신설 (TransformerModel inversion, arch §6.3 명시) — 2~3일
- (c) GGUF utility (`ggml_type_to_dtype`, `tensor_byte_size`, `qk_permute_shape` 등) → `shared/` 격상 또는 trait inversion — 1~2일
- (d) `TransformerLayer` 의존 inversion (KvCacheView trait 등, 갈래 3과 겹침) — 1~2일
- (e) weight_swap 16 파일 → `pressure/weight_swap/` 통째 이동 — 2~3일
- **순익**: §13.8-O marker 감소 + 완전 정합. sub-sprint 분할 필수.

#### 옵션 C4 + ModelConfig 격상만 (절충, 3~4일)
- (a) `ModelConfig` → `shared/`로 L2 격상 — 1일
- (e) weight_swap 16 파일 → `pressure/weight_swap/` 통째 이동 — 2~3일
- (b)(c)(d)는 LAYER-EXEMPT 코멘트로 인정 + backlog 명시
- **순익**: §13.8-O marker 일부 감소 (ModelConfig 분 + weight_swap_handler 3건 해소, TransformerModel/GGUF/TransformerLayer 신규는 backlog로)

#### 옵션 C5 (단계적, 2~3일)
- weight_swap 16 파일 → `pressure/weight_swap/` 통째 이동만
- 모든 신규 marker는 LAYER-EXEMPT 인정 + backlog 등록
- 다음 sprint에서 사전 작업 4건 단계적 수행
- **순익**: §13.8-O marker 일시 증가 (weight_swap_handler.rs 3 marker 제거 vs weight_swap/* 안 ~10 신규). 단 **도메인 분류 자체는 arch §6.3 정합** — pressure 도메인이 자기 책임 컴포넌트를 가짐.

#### 검증 게이트 (옵션 무관 공통)
- `cargo fmt --all && cargo clippy --workspace -- -D warnings` clean
- `cargo test -p llm_rs2 --lib`: 회귀 0
- `python3 scripts/layer_lint.py --baseline engine/tests/spec/inv_layer_baseline.json --json` new_violations=0 (또는 baseline 갱신)
- S25 Adreno OpenCL Qwen2.5-1.5b Q4_0 32토큰 bit-identical (precision swap path)

### 위임 prompt (옵션 결정 후)

> **에이전트**: `architect` (옵션 결정 후 spec/arch 갱신 + implementer 위임 prompt 작성)
> **모델**: `opus`
> **권한**: `spec/`, `arch/`, `docs/`, `.agent/todos/`

```
§13.8-O 갈래 1 (precision swap, WeightSwapDispatch trait 격상) 진입.

선택 옵션: <C4 + 사전작업 완전 | C4 + ModelConfig만 | C5 단계적>

작업:
1. spec/41-invariants.md INV-LAYER-005 갱신 (선택 옵션에 따른 marker 변화 명시)
2. arch/01-architecture.md §6.3 갱신 (실제 진행 방향 반영)
3. backlog [P2] §13.8-O 항목 갱신 (선택 옵션 + 미완 사전 작업 명시)
4. implementer/senior implementer 위임 prompt 작성 (작업 단위 + 검증 게이트 포함)

참고: .agent/todos/handoff_precision_swap_o1_entry_2026_05_26.md (본 handoff)
```

---

## Landmines / 미해결 / 안 가본 길

### 1. backlog 문구 vs arch §6.3 충돌 — **arch 우선**
- backlog: "WeightSwapHandler models/weights 이전" (`backlog.md` §13.8-O P2 항목)
- arch §6.3: weight_swap 컴포넌트(slot/secondary_mmap/swap_executor 등)를 **models/weights → pressure/로 통째 이동** 명시 (`pressure/policy/handlers/weight_swap/`, `pressure/state/weight_slot/`)
- **spec/arch 1:1 대응 원칙**(`spec/41-invariants.md`)에 따라 arch 우선. backlog 문구는 stale.
- **이 길은 가지 말 것**: backlog 문구 그대로 WeightSwapHandler를 models/weights로 옮기지 마라 (옵션 A — arch와 정반대).

### 2. 단순 이동만 하면 §13.8-O marker 적자
- 16 파일 통째 이동 후 ModelConfig/TransformerModel/GGUF/TransformerLayer 의존이 모두 §13.8-O 신규 marker
- 예상 신규 marker ~10건 vs 해소 3건 = 순익 −7건
- **C5는 이 적자를 LAYER-EXEMPT로 인정**, C4 + 사전작업은 인터페이스화 후 진행

### 3. C2 (arch §6.3 nested 완전 정합) — 본 sprint scope 외
- pressure/policy/handlers/weight_swap/ + pressure/state/weight_slot/ 분할 + cache_manager → manager rename + kv_cache → state/ 이동
- 5~10일, sub-sprint 다수. 본 sprint(§13.8-O 갈래 1 단독)에서 다루지 말 것.
- C4의 단일 `pressure/weight_swap/` sub-dir이 future-proof (mechanical로 nested 분할 가능)

### 4. 갈래 2/3은 별 sub-track
- 갈래 2: PrefetchAccess + PreloadPool L2 격상 (KV cache offload 인프라, precision swap **무관**)
- 갈래 3: KvCacheView trait (KVCacheOps generic default 정리)
- grill 라운드에서 곁가지 옵션으로 제시되었으나 트랙 이탈. precision swap 우선 완료 후 별 sprint.

### 5. WeightSwapHandler production caller 0건
- Phase 3에서 `CachePressureHandler` impl 제거 후 dispatch는 generate.rs/argus_cli 직접 처리
- 현재 caller = test 2건 (`test_eng_alg_211_weight_swap_handler.rs`, `test_wswap_e2e_phase3.rs`) + `pressure/mod.rs:34` pub use
- **결론**: trait 신설(옵션 B)은 over-engineering. trait 격상은 caller 다수 + 다양한 impl 필요 시점에 별 작업.

### 6. 시도했지만 폐기한 옵션
- **옵션 A** (WeightSwapHandler → models/weights/swap_handler.rs): arch §6.3과 정반대 방향
- **옵션 B** (WeightSwapDispatch trait + impl 분리): production caller 0건이라 over-engineering
- **옵션 C3** (LayerSlot/SecondaryMmap만 이동, swap_executor는 잔존): models/weights → pressure cross-domain 위반 신규 도입 (V-21 패턴)

### 7. 결정 대기
- **사용자 결정**: 옵션 C4 완전 / C4 절충 / C5 단계적 중 선택
- 선택에 따라 architect 라운드 (#6) 진입

---

## 참고 링크

- 상위 plan / spec:
  - `arch/01-architecture.md` §6.3 (L3 Pressure Domain — 정식 weight_swap 위치 명시)
  - `spec/41-invariants.md` §3.26 (INV-LAYER-001~005)
  - `ARCHITECTURE.md` §13 (Layered Architecture Mapping)
- 관련 memory 항목:
  - `[[layered-architecture-decision]]`
  - `[[refactor-spec-qcf-clippy-2026-05-21]]`
  - `[[qcf-naming-decision]]`
  - `[[generate-split-binaries]]`
- 이전 handoff (§13.8 마이그레이션):
  - `.agent/todos/handoff_inv_layer_L_hot_path_subtrait_2026_05_24.md` (§13.8-L L-1/L-2/L-3 완료)
  - `.agent/todos/handoff_inv_layer003_complete_2026_05_24.md` (D-1 sprint, §13.8-L/O marker 우회 등록)
  - `.agent/todos/handoff_inv_layer_AB_complete_2026_05_24.md`
  - `.agent/todos/handoff_inv_layer_sd1_sd2_complete_2026_05_24.md`
  - `.agent/todos/handoff_inv_layer_3b_3d_complete_2026_05_24.md`
- backlog: `.agent/todos/backlog.md` "[P2] §13.8-O cross-L3 vocabulary trait inversion"
- 코드 진입점:
  - `engine/src/pressure/weight_swap_handler.rs:21,23,25` (§13.8-O 3 marker)
  - `engine/src/models/weights/mod.rs` (이동 대상 16 파일 디렉토리)
