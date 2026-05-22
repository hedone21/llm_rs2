# Handoff: B-1 EvictMethod → pressure/eviction 종결 → 다음 sub-sprint 결정

**작성**: 2026-05-22
**HEAD**: `9b79c588 refactor(layer): B-1 — EvictMethod → pressure/eviction/method.rs` (origin/master push 완료)
**다음 세션 진입 문장**: `"B 다음 sub-sprint 결정 (B-2/B-3/B-4/B-5)"`
**선행 진입점**: [[handoff_step5b_observability_complete_2026_05_21]] — Step 5b 종결 직후

---

## TL;DR

INV-LAYER baseline 점진 해소 sprint B의 첫 sub-sprint B-1 완료. `EvictMethod` enum을 `resilience/executor.rs`에서 `pressure/eviction/method.rs` (신규 파일)로 이동. definitional owner를 pressure 도메인으로 명시. ARCHITECTURE.md §13.8-F로 "enum-as-data identifier across cross-cutting↔L3 boundary" 예외 정책 명문화 (V-12 ActionResult 선례 연장). baseline 296 → **287** (-9). **멈춘 이유: B-1 종결 + 다음 sub-sprint 사용자 결정 대기**.

---

## 진행 상태

### 본 sprint commits (1건)

| HEAD | scope | 변경 |
|---|---|---|
| `9b79c588` | refactor(layer) | B-1 EvictMethod relocation + ARCHITECTURE §13.8-F + spec INV-LAYER-004 비고 (12 파일, +95 / -145) |

### 게이트 결과

| 게이트 | 결과 |
|---|---|
| `cargo build -p llm_rs2 -p llm_shared -p llm_manager` | PASS |
| `cargo clippy -p llm_rs2 --lib -- -D warnings` | clean 0 |
| `cargo test -p llm_rs2 --lib` | 1196 PASS / 24 FAIL (사전 device-required, 본 변경 무관) |
| `cargo test -p llm_rs2 --test spec test_inv_layer` | **8 PASS / 0 FAIL** |
| `layer_lint --baseline` diff | -9 (296 → 287) |

### 변경 파일 (12건)

| 파일 | 변경 |
|---|---|
| `engine/src/pressure/eviction/method.rs` | **신규** (15 LOC, EvictMethod definition) |
| `engine/src/pressure/eviction/mod.rs` | `pub mod method` + `pub use method::EvictMethod` |
| `engine/src/resilience/executor.rs` | enum 정의 삭제 + `use crate::pressure::eviction::EvictMethod` |
| `engine/src/resilience/mod.rs` | re-export 경로 갱신 (`crate::pressure::eviction::EvictMethod` — BC 보존) |
| `engine/src/pressure/cache_manager.rs` | import 갱신 + test 블록 10건 path 갱신 |
| `engine/src/session/prefill.rs` | line 616 fully-qualified path 갱신 |
| `engine/src/inference/sampling.rs`<br/>`engine/src/layers/transformer_layer/forward_gen.rs`<br/>`engine/src/session/decode_fallback/eviction_trigger.rs`<br/>`engine/src/session/eval/runner.rs`<br/>`engine/tests/spec/test_qcf_swap_dump.rs` | cargo fmt 자동 정렬 / path 갱신 (의미 변경 없음) |
| `engine/tests/spec/inv_layer_baseline.json` | 296 → 287 재생성 |
| `ARCHITECTURE.md` | §13.8 §F 신규 + §13.5 V-10 갱신 + Resolution Log V-10 행 추가 |
| `spec/41-invariants.md` | INV-LAYER-004 비고 한 문장 추가 |

### baseline 변동 분석

- INV-LAYER-003 해소: production 1건 (`cache_manager.rs:13`) + test_block 10건 = **11건 해소**
- INV-LAYER-004 신규: `resilience/executor.rs` → `crate::pressure::eviction::EvictMethod` import **1건** (§13.8-F 예외, baseline 등재 유지)
- fmt 정렬 변경으로 2건 부수 재분류 (순증감 상쇄)
- **순감소: 9건 (296 → 287)**

기대값 -11이 아닌 -9인 이유: §13.8-F 자동 detection 룰은 본 sprint scope 외이므로 신규 1건은 baseline 잔류, fmt 정렬로 2건 재분류.

### ARCHITECTURE.md §13.8-F 결정 요지

cross-cutting 도메인(`observability/`, `resilience/`)이 L3 도메인의 enum/struct을 **data identifier** (HashMap key, struct field 등) 형태로 import할 때 INV-LAYER-004 예외.

조건:
1. import 대상이 enum/struct 등 *data type*이지 trait/concrete 함수 아님
2. cross-cutting 측이 *읽고 표현/저장* 용도, *직접 mutate/lifecycle 관리* 아님
3. 양쪽 도메인이 *단방향 message-passing* 패턴 (cross-cutting = producer/labeler, L3 = consumer/dispatcher)

선례: V-12 `core/events.rs::ActionResult`. 적용 예: V-10 `EvictMethod`.

---

## 다음 작업

B sprint 전체 후보 (handoff_step5_step6_complete_2026_05_21.md의 B 항목). 우선순위순:

### B-2. `observability::profile::*` trait inversion (2~3일)

- `OpProfiler`, `Timer`, `op_trace::*` 27건 (V-14/18/22 패턴)
- `OpInstrument` trait을 L2(`shared/`) 또는 inference 도메인에 신설
- `models/transformer.rs` (8 import), `layers/transformer_layer/forward*.rs` (~10건), `pressure/kivi_cache.rs` (2-3건) 정리
- 영향: 산발적이지만 파일당 단순 import 갱신
- 위임: architect (trait 위치 결정) → implementer (mechanical)
- 예상 baseline 감소: ~27건

### B-4. observability/eval/* L4 격상 (2~3일)

- INV-LAYER-004 36건 + INV-LAYER-003 일부 (eval이 L3/L4 hybrid 사용)
- 선택지 A: `observability/eval/` → `session/eval/`로 L4 격상 (ARCHITECTURE.md §13.4 매핑 명시안)
- 선택지 B: `EvalSink` trait inversion 유지
- A가 ARCHITECTURE.md 원안. mechanical move + path-only
- 위임: architect (A vs B 결정) → implementer (선택지 A면 mechanical sed)
- 예상 baseline 감소: ~36~50건

### B-5. INV-LAYER-001 L1→상위 31건 (3~5일)

- V-01 (gpu_self_meter trait 추출) + V-02 (tensor_partition L2 이동) + V-03 (LayerSlot/TransformerLayer opaque handle)
- backend별 분산: cpu 10, opencl 7, qnn_oppkg 5, cuda_embedded 5, cuda_pc 3
- 각 V-?? 별 PR 1개
- 위임: senior-implementer (.cl 영역 일부 포함)

### B-3. Backend capability trait 추출 (5~7일, **가장 큼**)

- L3 코드의 `CpuBackend`/`OpenCLBackend`/`neon` 직접 의존 ~85건 해소
- `Backend` trait capability method 확장
- V-13 / V-17 / V-20 별로 PR 분할 필수
- 위임: architect (trait 설계 SoT) → senior-implementer (NEON path 영향)
- 예상 baseline 감소: ~75건

### 권장 진입 순서

handoff_step5_step6_complete의 권장 = "B-1 → B-4 → B-2 → B-5 → B-3" (작고 risk 낮은 것 → blast radius 큰 것).

B-1 종결 후 다음은 **B-4 (eval L4 격상)** 또는 **B-2 (profile trait inversion)**. 둘 다 mechanical 비중이 높고 baseline 감소량 큼 (-27~50건).

---

## Landmines / 미해결

### 본 sprint과 무관한 사전 회귀 (P3)

1. **`crates/qnn_oppkg/src/lib.rs` unresolved imports** — diagnostic 53건. 본 변경 무관. `cargo build -p llm_rs2`로 우회.
2. **`backend::opencl::*` host test 24건 device-required FAIL** — 기존 P3, 본 변경 무관.

### §13.8-F 자동 detection 부재

- §13.8-F 예외 조건(enum-as-data identifier)은 자동 detection 룰 없음.
- 현재 `resilience/executor.rs` → `crate::pressure::eviction::EvictMethod` import 1건이 baseline 등재 유지.
- 5건 누적 시 layer_lint.py allowlist 도입 검토 (ARCHITECTURE.md §13.8-F 운용 메모).

### EvictMethod 이름의 D2o variant 모호성

- `EvictMethod::D2o`는 사실 eviction 아닌 merge compensation (D2OHandler는 CachePressureHandler).
- enum 이름과 D2o 본질 부정합. 본 sprint scope 외.
- 별도 backlog: `EvictMethod` → `KvManagementMethod` 또는 `KvPolicyKind` rename (eviction + merge + 향후 quantize 포괄). engine-internal rename이라 비용 작음.

### "이 길은 가지 마라"

- **B-2/B-3/B-4를 한 sprint로 묶지 말 것** — blast radius 차이가 너무 크다. 각 sub-sprint 독립 진행 + 매 종결 시 baseline delta 보고.
- **§13.8-F 예외를 무한 확장하지 말 것** — 5건 누적 시 layer_lint allowlist 도입 검토. 그 이상은 정책 자체 재검토 필요 (enum-as-data identifier 누적이 layered 의도 약화).

---

## 참고 자료

- 메모리: layered-architecture-decision — 본 sprint의 원안 (2026-05-16)
- 메모리: eurosys2027-post-paper — paper 사이클 종결, refactoring 메인 트랙
- 직전 handoff: `.agent/todos/handoff_step5b_observability_complete_2026_05_21.md`
- 선행 sprint handoff: `.agent/todos/handoff_step5_step6_complete_2026_05_21.md` (B sub-sprint 후보 분해)
- 변경된 정책 문서: `ARCHITECTURE.md` §13.8-F (신규), §13.5 V-10 + Resolution Log
- 변경된 spec: `spec/41-invariants.md` INV-LAYER-004 비고

## 자기점검 결과

- [x] 진입 문장이 한 줄로 명확? `"B 다음 sub-sprint 결정 (B-2/B-3/B-4/B-5)"` — 4개 후보 + 권장 순서 명시
- [x] "왜 멈췄는가"? B-1 종결 + 다음 sub-sprint 사용자 결정 대기
- [x] 가장 큰 landmine 표면화? §13.8-F 자동 detection 부재 (5건 누적 시 allowlist 도입) + D2o variant 이름 부정합 (별도 rename backlog)
- [x] 검증 게이트 수치/명령? `cargo clippy --lib clean 0`, `cargo test --lib 1196 PASS`, `test_inv_layer 8 PASS`, baseline 296→287
- [x] 길이 적정? 본문 ~500 토큰, 1 commit + 12 파일 + 4 sub-sprint 후보 + 3 landmine
