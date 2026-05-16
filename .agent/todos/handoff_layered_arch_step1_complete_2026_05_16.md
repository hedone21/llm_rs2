# Handoff: 레이어드 아키텍처 리팩토링 Step 1 완료 → Step 1 후속(spec test 구현) 진입

**작성**: 2026-05-16
**HEAD**: `4b536db5 docs(arch): layered architecture refactoring plan + INV-LAYER-001~005`
**다음 세션 진입 문장 (사용자)**: "레이어드 리팩토링 Step 1 후속(Task #3) 진행"

---

## TL;DR

외부 공개 대비 엔진 레이어드 리팩토링 **Step 1(문서화) 완료**. ARCHITECTURE.md §13, spec INV-LAYER-001~005, arch 컴포넌트 매핑 모두 commit. **다음 작업은 Task #3 — Implementer가 spec test 5개 + baseline JSON + layer_lint.py + check_spec_coverage.sh 확장 구현**. 코드는 아직 한 줄도 안 건드림. Step 2(generate.rs 13,022 LOC 분해) 진입 전 회귀 가드(spec test)부터 박는 것이 안전.

---

## 진행 상태

### Task 리스트 (8단계)

| ID | 상태 | 작업 |
|---|---|---|
| #1 | ✅ completed | ARCHITECTURE.md + spec INV-LAYER-001~005 작성 |
| #2 | ✅ completed | UNRESOLVED-A~E 5개 미결 사항 결정 + 문서 반영 |
| **#3** | **⏳ pending (이번 세션 진입 대상)** | **spec test + baseline + layer_lint 도구 구현** |
| #4 | ⏳ blocked-by #3 | L5/L4 분리: generate.rs 13,022 LOC → session/ |
| #5 | ⏳ blocked-by #4 | L1/L2 경계 정리: backend → shared/만 의존 |
| #6 | ⏳ blocked-by #5 | L3 도메인 재배치: core/ → pressure/, inference/ |
| #7 | ⏳ blocked-by #6 | Cross-cutting 분리: observability/, resilience/ |
| #8 | ⏳ blocked-by #7 | /simplify 코드 정리 |

### §13.8 Resolved Decisions (Task #2에서 합의 완료)

| ID | 결정 |
|---|---|
| A | AUF → **`shared/auf/`** (L2 자산, V-23 실측 반영) |
| B | Layer-aware pool → **`backend/<be>/pool.rs`** + `WeightStagingPool` trait |
| C | chat_template 모델별 → `inference/models/<arch>/`, generic → `inference/`, chat_ipc → `session/` |
| D | backend-specific buffer (`cl_*`/`cuda_*`/`rpcmem_*`) → **`backend/<be>/buffer/`** |
| E | lib 내부 inline `#[cfg(test)]` backend import는 **grandfathered exception**, 신규 테스트는 `tests/spec/` |

---

## 다음 작업: Task #3 — Implementer 위임 prompt (그대로 사용 가능)

> **에이전트**: `implementer` (subagent_type)
> **모델**: sonnet (Architect의 작업 결과를 코드로 옮기는 단순/명확한 작업)
> **권한**: 코드 수정 가능 (`engine/tests/`, `scripts/` 신설/수정)

### Prompt

```
llm.rs 외부 공개 대비 레이어드 아키텍처 리팩토링의 **Step 1 후속 작업**이다. Architect가 이미 ARCHITECTURE.md §13 + spec INV-LAYER-001~005를 작성했고(HEAD `4b536db5`), 이번 작업은 spec test와 baseline JSON, layer_lint 도구를 구현하여 회귀 가드를 박는 것이다.

## 컨텍스트 참조 (반드시 읽기)

1. `ARCHITECTURE.md` §13 — 레이어 정의, 매트릭스, 매핑, **§13.5 Violations V-01~V-31** (baseline 입력), Migration Plan
2. `spec/41-invariants.md` §3.26 — INV-LAYER-001~005 카탈로그 (테스트 대상)
3. `spec/01-architecture.md` §3.8 — SYS-100~105 + Layer Invariants 표 + 테스트 정책 NOTE
4. `arch/01-architecture.md` §6 — 컴포넌트 매핑

## 구현 항목 (총 8개)

### 1) `scripts/layer_lint.py` (신규)
Python 도구. 다음을 수행:
- `engine/src/**/*.rs`에서 `use crate::*` import를 grep으로 추출
- 각 파일을 layer(L1/L2/L3-pressure/L3-inference/L4/L5/cross-cutting)로 매핑 (디렉토리 prefix 기반: 현재 코드의 디렉토리 구조를 기준으로 함 — `backend/`→L1, `buffer/+memory/`→L2 일부, `core/{kv_cache,kivi_cache,cache_manager,eviction,pressure,kv_migrate,offload}`→L3-pressure, `core/{sampling,attention_scores,speculative,skip_config,chat_template,chat_ipc}` + `layers/` + `models/`→L3-inference, `core/{events,rss_trace}` + `profile/` + `eval/` + `experiment.rs`→observability, `resilience/` + `core/{sys_monitor,gpu_yield}` + `auf/`→resilience, `bin/`→L5)
- 각 import에 대해 layer pair를 결정하고 INV-LAYER-001~005 중 어느 것을 위반하는지 판정
- 출력: JSON (`{"violations": [{"id": "V-XX", "file": "...", "line": N, "import": "...", "rule": "INV-LAYER-XXX", "kind": "L1→L3 reverse"}, ...]}`)
- 추가 옵션: `--baseline <path>` — baseline JSON을 읽고 새로 발견된 위반만 출력 (회귀 감지)

ARCHITECTURE.md §13.5의 V-01~V-31을 모두 재현할 수 있어야 한다. 재현 안 되면 layer 매핑 규칙을 보강.

### 2) `engine/tests/spec/inv_layer_baseline.json` (신규)
V-01~V-31을 그대로 등재. ARCHITECTURE.md §13.5 표에서 추출. 형식은 `layer_lint.py` 출력과 동일. 각 항목에 `note: "ARCHITECTURE.md §13.5"` 필드 추가. 마이그레이션 진행에 따라 줄어들 예정(Step 3 후 V-01~V-09 해소 등).

### 3) `engine/tests/spec/test_inv_layer_001.rs` (신규)
INV-LAYER-001 (L1 backend → L2 외 import 금지) 검증. `Command::new("python3").arg("scripts/layer_lint.py").arg("--baseline").arg("engine/tests/spec/inv_layer_baseline.json")` 실행 후 종료 코드 0 + 출력의 violations 배열이 비어있음을 assert. INV별 분류는 `--filter inv-layer-001` 옵션으로 처리.

### 4) `engine/tests/spec/test_inv_layer_002.rs` (신규)
INV-LAYER-002 (L2 shared/ → L3+ import 금지). 동일 패턴.

### 5) `engine/tests/spec/test_inv_layer_003.rs` (신규)
INV-LAYER-003 (L3 inference ↔ pressure는 trait만). 동일 패턴.

### 6) `engine/tests/spec/test_inv_layer_004.rs` (신규)
INV-LAYER-004 (cross-cutting → L3 concrete trait 경유). 동일 패턴.

### 7) `engine/tests/spec/test_inv_layer_005.rs` (신규)
INV-LAYER-005 (L5 bin → L4 session/만). 동일 패턴. production binary(`generate`)만 대상이며 microbench/test binary는 제외.

### 8) `scripts/check_spec_coverage.sh` 확장
기존 스크립트가 INV ID prefix(`INV-LAYER`)를 인식하도록 grep 패턴 보강. INV-LAYER-001~005가 `spec/41-invariants.md`에 있고 `engine/tests/spec/test_inv_layer_001~005.rs`에 대응 테스트가 있는지 확인. 누락 시 비-zero exit.

## 검증 게이트

순서대로 모두 통과해야 commit:
1. `cargo fmt --all`
2. `cargo clippy --workspace -- -D warnings`
3. `cargo test --workspace` (단, INV-LAYER 테스트 5개는 baseline 매칭으로 PASS)
4. `python3 scripts/layer_lint.py` 단독 실행 — V-01~V-31 모두 재현되는지 확인 (재현 안 되면 매핑 규칙 보강)
5. `python3 scripts/layer_lint.py --baseline engine/tests/spec/inv_layer_baseline.json` — diff 0 (회귀 없음)
6. `scripts/check_spec_coverage.sh` 통과
7. `/sanity-check` 스킬 통과

## 제약

- `engine/src/` 안 코드는 절대 수정 금지. **테스트와 스크립트만 추가**.
- inline `#[cfg(test)]` 안의 `CpuBackend::new()` 등은 §13.8-E grandfathered — baseline에 등재된 채 유지 (즉 V-15가 baseline에 포함되어야 하며, layer_lint가 검출은 하되 INV-LAYER-001 위반으로 분류는 하지 않거나 baseline match로 silenced).
- microbench/test binary는 INV-LAYER-005 enforcement 대상 외 — `bin/microbench_*`, `bin/test_*`, `bin/probe_*`, `bin/stage*`, `bin/signal_injector` 등은 layer_lint가 skip.
- 신규 테스트는 `engine/tests/spec/` 아래에 위치 (feedback `spec_tests_required` 준수). inline 테스트 추가 금지.

## 완료 보고

- 추가한 파일 절대경로 목록
- baseline JSON에 등재된 위반 수 (31건 + α 가능)
- `layer_lint.py` 출력이 §13.5와 일치하는지 (재현 검증)
- 다음 작업 Task #4 (L5/L4 분리) 진입 전 추가로 결정해야 할 사항이 있다면 명시

작업 완료 후 자동 commit (CLAUDE.md "완료 시 자동 커밋" 규칙).
```

---

## Task #4 진입 전 Architect 후속 결정 (Step 2 시작 시 처리)

이전 Architect 보고에서 식별한 항목:

1. **`session/` sub-module 분할 단위**: `mod.rs` / `decode_loop.rs` / `prefill.rs` / `chat_ipc.rs` / `init.rs` — Step 2 진입 시 `bin/generate.rs` 13,022 LOC 분석 후 확정
2. **`ChatTemplate` trait 시그니처**: `apply_template(messages) -> String` vs streaming variant — Step 4 진입 시 모델별 코드 본 후 확정
3. **`backend/cuda_common/buffer/` 신설 여부**: `cuda_buffer.rs`, `cuda_mmap_alias_buffer.rs`가 cuda_embedded와 cuda_pc 양쪽 공유인지 — Step 3 진입 시 실측 후 확정
4. **`WeightStagingPool` trait lifetime/ownership**: RAII guard vs callback — Step 3 진입 시 기존 HostPtrPool/layer_object_pool 사용처 분석 후 결정

이 4가지는 Task #4/#5에서 자연 해결되므로 지금 결정할 필요 없음.

---

## 환경 / 규칙 요약

- **언어**: 한국어 (CLAUDE.md 시스템 지시)
- **자동 commit**: 작업 완료 시 자동 커밋 (CLAUDE.md 워크플로우 규칙)
- **자동 알림**: `notify-send "llm.rs" "<작업 요약>"`
- **GGUF 우선**: 기본 모델 포맷은 GGUF (CLAUDE.md 핵심 제약)
- **GPU 없는 호스트**: `opencl` feature 기본 활성화. GPU 연산은 실행되지 않음
- **테스트 정책 (§13.8-E)**: 신규 테스트는 `engine/tests/spec/`, inline `#[cfg(test)]` 추가 금지
