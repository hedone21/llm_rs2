# Handoff: Step 5 (Cross-cutting 분리) + Step 6 /simplify 종결 → 다음 sprint 진입 결정

**작성**: 2026-05-21
**HEAD**: `edb72aa1 refactor: step 6-A — /simplify mechanical 정리` (push 예정)
**다음 세션 진입 문장**: `"다음 sprint 결정"`
**선행 진입점**: [[handoff_post_paper_struct_cleanup_2026_05_21]] — 본 sprint의 직전 sweep

---

## TL;DR

`engine/src/core/` 모듈 완전 해체. 5개 잔존 파일을 모두 적절한 도메인으로 승격: `chat_template→session/`, `sys_monitor/gpu_yield→resilience/`, `events/rss_trace→observability/` (신설). `layered architecture decision (2026-05-16)` 의 core/ 해체 목표 달성. layer_lint.py LAYER_RULES 갱신 + INV-LAYER baseline 재생성. **멈춘 이유: Step 5+6 종결 + 다음 sprint 사용자 결정 대기**.

---

## 진행 상태

### 본 세션 commits (8건)

| HEAD | scope | 변경 |
|---|---|---|
| `b26035a9` | refactor(layer) | 5-A chat_template → session/ (157 LOC, 6 file) |
| `e44d42b2` | refactor(layer) | 5-B sys_monitor → resilience/ (141 LOC, 8 file) |
| `77f9137b` | refactor(layer) | 5-C gpu_yield → resilience/ (89 LOC, 5 file) |
| `8260a8e1` | refactor(layer) | 5-D observability/ 신설 + events 이동 (450 LOC, 7 file) |
| `3b871b71` | refactor(layer) | 5-E rss_trace → observability/ (252 LOC, 7 file) |
| `3d011794` | refactor(layer) | 5-F core/ 모듈 완전 제거 (2 file) |
| `722953d7` | refactor(layer) | 5-G layer_lint LAYER_RULES + baseline JSON 갱신 (2 file) |
| `edb72aa1` | refactor | 6-A /simplify (redundant_closure + cargo fmt 정렬, 10 file) |

### 게이트 결과

| 게이트 | 결과 |
|---|---|
| `cargo build -p llm_rs2 -p llm_shared -p llm_manager` | PASS |
| `cargo clippy -p llm_rs2 --lib -- -D warnings` | clean 0건 |
| `cargo test -p llm_rs2 --lib` | 1203 PASS / 24 FAIL = `backend::opencl::*` device-required (사전 P3, 본 변경 무관) |
| `cargo test -p llm_rs2 --test spec` | 660 PASS / 3 FAIL = device-required 추가 3건 (사전, 본 변경 무관) |
| `cargo test -p llm_shared --lib --tests` | 38+27 PASS |
| `cargo test -p llm_manager --lib --tests` | 38+223 PASS |
| INV-LAYER spec test (001~007 + tasks) | 8 PASS |

### 도메인 매트릭스 (변경 전 → 후)

| 파일 | 이전 | 이후 | 도메인 |
|---|---|---|---|
| `chat_template.rs` (157 LOC) | `core/` | `session/` | L4 |
| `sys_monitor.rs` (141 LOC) | `core/` | `resilience/` | cross-cutting |
| `gpu_yield.rs` (89 LOC) | `core/` | `resilience/` | cross-cutting |
| `events.rs` (450 LOC) | `core/` | `observability/` (신설) | cross-cutting |
| `rss_trace.rs` (252 LOC) | `core/` | `observability/` (신설) | cross-cutting |

### INV-LAYER baseline 변동

- 이전: 292건 / 이후: 296건 (+4)
- rule별 분포:
  - INV-LAYER-001: 31 → 31 (변동 0)
  - INV-LAYER-002: 7 → 8 (+1)
  - INV-LAYER-003: 189 → 194 (+5)
  - INV-LAYER-004: 36 → 36 (변동 0)
  - INV-LAYER-005: 29 → 27 (-2 — generate.rs의 core/* → resilience/observability 재분류로 L4 우회 위반에서 제외)
- session/ → L4 매핑 신설로 chat_ipc 등 추가 분류 (룰 일관성 회복)

### `engine/src/` 구조 (현재)

```
auf/  backend/  buffer/  inference/  layers/  memory/  models/  pressure/
qcf/  resilience/  observability/  ← (신설)
session/  ← (L4, chat_template 흡수)
eval/  experiment.rs  profile/  ← (cross-cutting, observability 흡수 후보)
bin/  bin_helpers/
```

`core/` 디렉토리 **완전 삭제**. `layered architecture decision (2026-05-16)` 원안의 Cross-cutting/observability 신설 첫 진행.

---

## 다음 작업

후보 우선순위. 사용자가 진입할 항목 선택 후 별도 entry handoff 작성.

### A. Step 5b — observability 통합 (점진 확장)

`project_layered_architecture_decision.md` 원안: `observability/`는 `events + rss_trace + profile + eval + experiment` 흡수 예정. 현재 events + rss_trace만 이동. 후속:
- `profile/` (4 파일) → `observability/profile/`
- `eval/` (7 파일) → `observability/eval/`
- `experiment.rs` → `observability/experiment.rs`
- 영향 범위: importer 수십 건. 위임: Implementer (mechanical mass-mv + sed).

### B. INV-LAYER baseline 296건 점진 해소 (multi-sprint)

- rule별 우선순위: 003 (194건) > 004 (36건) > 001 (31건) > 005 (27건) > 002 (8건)
- 빈도 높은 패턴부터 식별 → 자율 mass-fix.
- 위임: 자율 (sub-sprint별 1 rule).

### C. generate 바이너리 분할 (사용자 지정 "마지막 단계")

- `[P2] generate 바이너리 분할 + Manager 통합` backlog 항목. 설계 라운드 필요.
- 진입: 별도 sprint.

### D. EnergyConstraint Spec-Impl Divergence (MGR-ALG-015)

- 결정 + 0.5일 작업.

### E. LISWAP-6 cleanup segfault

- backlog [P2]. 디버깅 1-2일.

### 권장

`generate 분할`은 사용자가 "마지막 단계"로 지정. 직전 단계 자연 후속은:
- **A (observability 통합)** — Step 5 자연 연장. mechanical, 자율 가능.
- **B (INV-LAYER 점진 해소)** — 구조 정리 막판. rule별 sub-sprint.

A와 B는 병행 가능 (A는 mechanical move, B는 import inversion). A를 먼저 완성하고 B 진입이 의존 순서상 자연스러움.

---

## Landmines / 미해결

### 본 sprint과 무관한 사전 회귀 (P3)

1. **`cargo clippy --tests` 7건 회귀** — microbench/probe bins, test_model_forward_parity, test_auf_gguf_byte_equivalence. 본 sprint 무관. `lib` clippy는 clean.
2. **`backend::opencl::*` host test 24건 device-required FAIL** — 기존 P3 ([P3] backend::opencl::* host test 24개 device-required fail).
3. **spec test 3건 추가 FAIL** — `test_host_ptr_pool::test_g_fill_host_ptr_buffer_byte_equal_readback`, `test_inv_140_fused_convert_byte_equal`, `test_partition_split_backend_retag` — 모두 OpenCL device-required로 추정 (clEnqueueReadBuffer 에러 메시지 확인). 본 변경 무관.

### Step 5 도메인 결정 (이미 결정됨)

- `chat_template.rs` → `session/` 직속 (사용처가 모두 session/chat/ 내부)
- `events.rs` + `rss_trace.rs` → `observability/` (신설, layered architecture decision 원안 따름)
- `sys_monitor.rs` + `gpu_yield.rs` → `resilience/` (cross-cutting이므로 기존 resilience/에 흡수)

### "이 길은 가지 마라"

- `observability/`에 `profile/`, `eval/`, `experiment.rs`까지 1 sprint로 통합 — 작업량 2~3배 + importer 수십 건 영향. Step 5b로 분리 권장.
- `core/` 디렉토리 빈 채 유지 — 의미 없음. 5-F에서 mod.rs까지 삭제 완료.

---

## 참고 자료

- 메모리: [[layered-architecture-decision]] — 본 sprint의 원안 (2026-05-16)
- 메모리: [[eurosys2027-post-paper]] — paper 사이클 종결, refactoring 메인 트랙
- 직전 handoff: `.agent/todos/handoff_post_paper_struct_cleanup_2026_05_21.md`
- 변경된 layer_lint 룰: `scripts/layer_lint.py` LAYER_RULES (session/observability 추가, core/* 제거)
- 변경된 baseline: `engine/tests/spec/inv_layer_baseline.json` (296건)

## 자기점검 결과

- [x] 진입 문장이 한 줄로 명확? `"다음 sprint 결정"` — 본 sprint 종결 후 다음 sprint 선택이 첫 과제
- [x] "왜 멈췄는가"? Step 5+6 종결 + 다음 sprint 사용자 결정 대기
- [x] 가장 큰 landmine 표면화? clippy --tests 7건 사전 회귀 + spec test 3건 device-required는 본 sprint 무관 (lib clippy/test는 clean)
- [x] 검증 게이트 수치/명령? `cargo clippy --lib -- -D warnings: 0`, `cargo test --lib: 1203 PASS`, INV-LAYER 8 PASS
- [x] 길이 적정? 본문 ~500 토큰, 8 commits + 5 도메인 매트릭스 + 후속 5 후보 + landmine 명시
