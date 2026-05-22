# Handoff: B-2 observability profile inversion 종결 → 다음 sub-sprint 결정

**작성**: 2026-05-22
**HEAD**: `6052225a refactor(layer): B-2e — cleanup + Resolution Log V-14/18/22 SHA`
**다음 세션 진입 문장**: `"B 다음 sub-sprint 결정 (B-4 / B-5 / B-3)"`
**선행 진입점**: [[handoff_b1_evict_method_relocation_2026_05_22]]

---

## TL;DR

INV-LAYER baseline 점진 해소 sprint B의 두 번째 sub-sprint B-2 완료. `observability::profile::*`의 L3 → cross-cutting concrete import 35건(예상)을 3 패턴으로 분해해서 해소: **Pattern A/B 매크로 + cfg gate, Pattern C trait inversion**. ARCHITECTURE.md §13.8-G(shared identifier promotion) + §13.8-H(instrument macro helper) 정책 신규 명문화. baseline 287 → **254** (-33). **멈춘 이유: B-2 종결 + 다음 sub-sprint 사용자 결정 대기**.

---

## 진행 상태

### 본 sprint commits (7건)

| HEAD | scope | 변경 |
|---|---|---|
| `110c2f0c` | docs(arch) | B-2-0 §13.8-G/H 정책 + V-14/18/22 행 갱신 + Resolution Log 행 추가 |
| `98fe13f6` | refactor(layer) | B-2a OpKind enum → L2 `engine/src/op_kind.rs` (-5건) |
| `b1a47e5b` | refactor(layer) | B-2b `qcf_timer!` 매크로 + Pattern A 12건 매크로화 (-12건) |
| `16ad5473` | refactor(layer) | B-2c `op_span!` 매크로 패밀리 + Pattern B 9건 매크로화 (-9건) |
| `981f7aac` | refactor(layer) | B-2d-1 `OpInstrument` trait + OpProfiler/PrefillOpProfiler impl (변동 0) |
| `2324d695` | refactor(layer) | B-2d-2 Pattern C 6건 → `Option<&mut dyn OpInstrument>` (-6건) |
| `6052225a` | refactor(layer) | B-2e cleanup + Resolution Log SHA 채움 (-1건) |

### 게이트 결과

| 게이트 | 결과 |
|---|---|
| `cargo build -p llm_rs2` (default features) | PASS |
| `cargo build --release` | PASS (64초) |
| `cargo clippy --lib -- -D warnings` | clean 0 |
| `cargo test --lib` | 1204 PASS / 22 FAIL (사전 device-required, 본 변경 무관) |
| `test_inv_layer` | 8 PASS |
| LLMRS_OP_TRACE=1 / LLMRS_PROFILE=1 env gate sanity | 정상 (release binary --help 통과) |
| `--no-default-features` 빌드 | **사전 회귀** (`swap_dispatch.rs:441 map_weights_for_cpu`), 본 sprint 무관 |
| **baseline** | 287 → **254** (-33) |

### baseline 변동 누적

| 단계 | 변동 | 종료 baseline |
|---|---|---|
| 진입 시 (B-1 종결 후) | — | 287 |
| B-2a OpKind L2 | -5 | 282 |
| B-2b qcf_timer! + Pattern A 12건 | -12 | 270 |
| B-2c op_span! + Pattern B 9건 (callsite 22개) | -9 | 261 |
| B-2d-1 trait 정의 | 0 | 261 |
| B-2d-2 Pattern C 6건 (trait object) | -6 | 255 |
| B-2e cleanup (mod.rs BC re-export 제거) | -1 | **254** |

### ARCHITECTURE.md 정책 신규

#### §13.8-G shared identifier promotion

cross-cutting 또는 L3 외부에 정의된 enum/struct이 양쪽 도메인에서 *대등한 어휘*로 쓰여 owner 귀속이 자의적인 경우, L2(`engine/src/<name>.rs`)로 격상해 import 방향 자체를 소거. §F가 위반을 *수용*한다면 §G는 위반을 *제거*. 적용: **OpKind** (B-2a).

#### §13.8-H instrument macro helper

L2 정의 매크로가 expansion 내부에서 cross-cutting concrete를 참조하는 패턴은 (1) L2 위치 + 매크로만 import, (2) cfg gate zero-cost, (3) heap/vtable 없는 본문 — 3조건 충족 시 INV-LAYER-003/004 위반 아님. layer_lint은 macro expansion 미분석. 적용: `qcf_timer!`, `op_span!`, `op_start!`, `op_record!`, `op_note_forward_call!`.

### Pattern별 해법 매트릭스

| Pattern | 건수(예상→실측) | 해법 | 적용 정책 |
|---|---|---|---|
| **A** Timer + QCF counter | 13 → 12 | `qcf_timer!` 매크로 + cfg gate | §13.8-H |
| **B** op_trace fn + OpKind | 14 → 9 (callsite 22) | `op_span!` 매크로 + OpKind L2 격상 | §13.8-G + §13.8-H |
| **C** OpProfiler/PrefillOpProfiler struct | 7 → 6 | `OpInstrument` trait + `&mut dyn` erasure | 정통 trait inversion |
| **D** BC re-export | 1 → 1 | mod.rs:21 + llama_layer.rs:7 OpProfiler re-export 제거 | mechanical cleanup |

총 해소: 35 (예상) → 33 (실측, -2건은 §13.8-E grandfathered + PhaseHook trait import 잔재).

---

## 다음 작업

B sprint 전체 후보 (handoff_step5_step6_complete_2026_05_21.md의 B 항목). 권장 순서: **B-4 → B-5 → B-3**.

### B-4. observability/eval/* L4 격상 (2~3일)

- INV-LAYER-004 36건 + INV-LAYER-003 일부 (eval이 L3/L4 hybrid 사용)
- 선택지 A: `observability/eval/` → `session/eval/`로 L4 격상 (ARCHITECTURE.md §13.4 매핑 명시안)
- 선택지 B: `EvalSink` trait inversion 유지
- A가 ARCHITECTURE.md 원안. mechanical move + path-only
- 위임: architect (A vs B 결정) → implementer (선택지 A면 mechanical sed)
- 예상 baseline 감소: ~36~50건 (254 → ~204)

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

---

## Landmines / 미해결

### 본 sprint과 무관한 사전 회귀 (P3)

1. **`cargo build --no-default-features` 실패** — `swap_dispatch.rs:441 map_weights_for_cpu` 메서드 미발견. 본 sprint 이전부터 존재 (stash 검증). profile=off 빌드 검증 보류.
2. **`backend::opencl::*` host test 22건 device-required FAIL** — 기존 P3, 본 변경 무관.
3. **`crates/qnn_oppkg/src/lib.rs` unresolved imports 53건** — handoff B-1에서도 명시된 사전 회귀.
4. **외부 master commit** `817755a8 refactor(manager): apply SOLID/DRY principles and fix swap weights test` — 본 sprint와 무관 (manager 코드), local-only commit으로 b2 worktree base에 자연 포함됨.

### Pattern B-2 잔재

- **forward.rs:1304** `crate::observability::profile::ops::PrefillOpProfiler` — `#[cfg(test)] mod tests` 블록 내 직접 사용. §13.8-E grandfathered 정책으로 baseline 등재 유지. 신규 위반 아님.
- **phase_aware_swap.rs:32** `crate::observability::profile::op_trace::{DdrPhase, PhaseHook}` — trait/type import. OpKind는 L2 import로 분리됐으나 PhaseHook trait은 op_trace에 남아있음. **별도 backlog: PhaseHook trait L2 승격** (B-2 scope 외).
- **V-18 Galloc 부분** — `crate::memory::galloc::Galloc` import는 B-2 sprint scope 외. 별도 backlog.

### §13.8-H 자동 detection 부재

- §13.8-H 예외 조건(instrument macro helper)은 자동 detection 룰 없음.
- 매크로 expansion 결과의 cross-cutting 참조는 layer_lint 미감지 — 정신적 위반 가능.
- 5건 누적 시 layer_lint allowlist 도입 검토 (§13.8-F 운용 메모와 동일 정책).

### "이 길은 가지 마라"

- **B-2 hot path 매크로를 trait dispatch로 전환하지 말 것** — TBT 회귀 risk 큼. 매크로 + cfg gate가 zero-cost의 정답.
- **Pattern C trait method API를 더 세분화하지 말 것** — 현재 5개(record_op_us / record_cpu_fallback / record_layer_end / merge_events / note_token)가 OpProfiler/PrefillOpProfiler 공개 mutate API 전수. 추가 메서드는 implement burden만 늘림.
- **forward.rs raw pointer 패턴(`profiler_ptr: *mut dyn OpInstrument`)을 safe API로 강제 전환하지 말 것** — borrow checker 회피 패턴이 trait object에서도 유효. 변경 시 일관성 깨짐.

### B-2 microbench skip 근거

- 원래 plan은 S25 디바이스 deploy + TBT 측정 (±3% 게이트)
- Pattern A/B 모두 매크로 + cfg gate로 해결 → hot path inline 그대로, **trait dispatch 자체가 없음**
- 디바이스 측정 ROI 낮아 host release build sanity + env gate test로 축약
- Pattern C(trait object)는 cold path 한정 (struct field/함수 인자, hot loop 안 호출 없음)
- 후속 sub-sprint(B-5/B-3)에서 hot path trait 도입 시 microbench 정식 진행 필요

---

## 참고 자료

- 메모리: [[layered-architecture-decision]] — 본 sprint의 원안 (2026-05-16)
- 메모리: [[eurosys2027-post-paper]] — paper 사이클 종결, refactoring 메인 트랙
- 직전 handoff: `.agent/todos/handoff_b1_evict_method_relocation_2026_05_22.md`
- 정책 문서: `ARCHITECTURE.md` §13.8-G (신규), §13.8-H (신규), §13.5 V-14/18/22 + Resolution Log
- 변경된 spec: `spec/41-invariants.md` INV-LAYER-003 비고 §13.8-H 예외
- 신규 L2: `engine/src/op_kind.rs`, `engine/src/instrument.rs`
- 외부 contributor 진입 (§13.6): "새 op 추가" → `engine/src/op_kind.rs::OpKind` variant + observability/profile/op_trace 처리

## 자기점검 결과

- [x] 진입 문장이 한 줄로 명확? `"B 다음 sub-sprint 결정 (B-4 / B-5 / B-3)"`
- [x] "왜 멈췄는가"? B-2 종결 + 다음 sub-sprint 결정 대기
- [x] 가장 큰 landmine 표면화? §13.8-H 자동 detection 부재 + microbench skip 근거 + PhaseHook trait 승격 backlog
- [x] 검증 게이트 수치/명령? `cargo clippy --lib clean 0`, `cargo test --lib 1204 PASS`, `test_inv_layer 8 PASS`, baseline 287→254
- [x] 길이 적정? 본문 ~600 토큰, 7 commits + 패턴별 매트릭스 + 3 후속 후보 + 4 landmine
