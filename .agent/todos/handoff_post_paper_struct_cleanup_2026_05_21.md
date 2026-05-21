# Handoff: Post-paper 구조 cleanup sprint 종결 → 다음 sprint 결정

**작성**: 2026-05-21
**HEAD**: `7b6ae619 refactor(session): run_qcf_warmup_workflow를 QcfWarmupCtx + QcfWarmupConfig 2-struct로 분할` (origin/master push 완료)
**다음 세션 진입 문장**: `"다음 sprint 결정"` — 남은 구조 후보 (INV-LAYER baseline / EnergyConstraint / LISWAP-6) 또는 신규
**선행 진입점**: [[handoff_refactor_sweep_complete_2026_05_21]] — 본 sprint 직전 sweep (clippy/PROTO-012/QCF dot prefix)

---

## TL;DR

EuroSys 2027 paper 사이클 종결 확인 후 사후 리팩토링 트랙으로 전환. 남은 구조 cleanup을 의존성 자연 순서(stale import → backlog 표기 → QCF estimator → type alias → unsafe API → Box wrap → struct ctx)로 7 commits 묶음 처리. 4건은 메인 세션 자율, 3건(risk MEDIUM)은 senior-implementer 병렬 위임 — 모두 게이트 PASS. 본 sweep으로 clippy `#[allow]` silence **4건 모두 정식 해소**. **멈춘 이유: 남은 구조 후보(INV-LAYER 292건, EnergyConstraint, LISWAP-6) 모두 결정 또는 multi-sprint 작업이라 사용자 결정 대기**.

---

## 진행 상태

### Sprint 1 — 메인 세션 자율 (4 commits)

| HEAD | scope | 변경 요약 |
|---|---|---|
| `9c9550f8` | refactor(tests) | stale import 68건 → 0 (16 test 파일). `llm_shared::auf` → `llm_rs2::auf` 61건 + `auf_dtype_convert` path 4건 + `compute_qcf_swap` → `compute_qcf_weight_swap` rename 11호출처 + `core::math_utils` dead test 2건 제거 |
| `01858d20` | docs(backlog) | Weight Swap 섹션 "EuroSys 2027 critical path" 표기 갱신, 우선순위 재판정 대상으로 마킹 |
| `e97685f5` | refactor(qcf) | estimator default curves key dot prefix: `eviction`/`sliding`/`kivi`/`swift` → `kv.*`/`weight.*` (20 hits + spec test 2건) |
| `09882ff5` | refactor(design) | type alias 2건 (AsyncLayerBuildPair, ImportanceWithRaws) → `#[allow(type_complexity)]` 2건 제거 |

### Sprint 2 — senior-implementer 병렬 (3 commits)

| HEAD | scope | 위임 결과 |
|---|---|---|
| `8b9deacd` | refactor(backend) | `Backend::alloc_alias_weight_buffer` → `unsafe fn` + `# Safety` doc. trait 4 impl + caller 4 unsafe block. `#[allow(not_unsafe_ptr_arg_deref)]` 제거. 5 files +97/-37 |
| `8edb4318` | refactor(session) | `ChatKvMode::Standard` → `Box<ChatKvModeStandard>` struct. 정의 + 패턴 매칭 ~30 site + spec test 2개 갱신. `#[allow(large_enum_variant)]` 제거. 3 files +77/-114 |
| `7b6ae619` | refactor(session) | `run_qcf_warmup_workflow` 16-arg → `QcfWarmupCtx<'a>` + `QcfWarmupConfig<'a>` 2-struct. caller 2 site 갱신. `#[allow(too_many_arguments)]` 제거. 3 files +95/-50 |

### 게이트 결과 (cross-effect 통합 검증)

| 게이트 | 결과 |
|---|---|
| `cargo clippy -p llm_rs2 --lib -- -D warnings` | clean |
| `cargo build -p llm_rs2 -p llm_shared -p llm_manager --tests` | PASS |
| `cargo test -p llm_rs2 --lib qcf::` | 141 PASS |
| `cargo test -p llm_rs2 --lib models::weights::` | 84 PASS |
| `cargo test -p llm_rs2 --lib session::` | 52 PASS |
| `cargo test -p llm_shared --lib` | 38 PASS |
| `cargo test -p llm_manager --lib` | 223 PASS |
| `cargo test -p llm_rs2 --test spec test_wswap_e2e_phase3` | 9 PASS |
| `cargo test -p llm_rs2 --test spec test_inv_143_borrow_buffer_lifetime` | 5 PASS |
| `cargo test -p llm_rs2 --test spec test_chat_*_multi_turn` | 17+3 PASS |
| `cargo fmt --all -- --check` | clean |
| `git rev-list --left-right --count origin/master...master` | 0 / 0 |

### Design `#[allow]` 4건 해소 매트릭스

| silence type | 위치 | 해소 방법 | commit |
|---|---|---|---|
| `type_complexity` | `swap_executor::execute_on_slots` local binding | `type AsyncLayerBuildPair = (usize, AsyncLayerBuild)` | `09882ff5` |
| `type_complexity` | `ImportanceCollector::build_with_raws` return tuple | `pub type ImportanceWithRaws = (...)` | `09882ff5` |
| `not_unsafe_ptr_arg_deref` | `OpenCLBackend::alloc_alias_weight_buffer` | trait + impl 모두 `unsafe fn` + caller unsafe block | `8b9deacd` |
| `large_enum_variant` | `ChatKvMode::Standard` 376b inline | `Box<ChatKvModeStandard>` 추출 | `8edb4318` |
| `too_many_arguments` | `run_qcf_warmup_workflow` 16 args | `QcfWarmupCtx + QcfWarmupConfig` 2-struct | `7b6ae619` |

---

## 다음 작업

후보를 우선순위로 정리. 사용자 결정 후 별도 entry handoff 작성.

### 명백한 구조적 리팩토링 (잔여)

**A. [P2] INV-LAYER baseline 292건 점진 해소**
- 진입 문장 후보: `"INV-LAYER-003 패턴 분석 진입"`
- 분포: INV-LAYER-003 = 189건 (majority), 001 = 31, 004 = 36, 005 = 29, 002 = 7
- baseline JSON: `engine/tests/spec/inv_layer_baseline.json`
- 접근: rule별 패턴 분석 → 해소 방법 (trait 추출 / indirection 도입 / 코드 이동) 설계 → 점진적 해소
- multi-sprint (rule별 1 sprint, 총 3~5 sprint 추정)
- 위임: Architect (rule별 패턴 분석 + 해소 설계) → Implementer (rule별 변경) → Tester (lint baseline 갱신)

### 결정 필요

**B. EnergyConstraint Spec-Impl Divergence (MGR-ALG-015)**
- spec: 연속 pressure `m = clamp(1-pct/100, 0, 1) * 0.5`
- 코드: Level enum 4단계 이산 (Normal=0.0, Warning=0.55, Critical=0.80, Emergency=1.0)
- 둘 중 하나로 통일 결정 (spec 갱신 vs 코드 갱신)
- 작업 0.5일 + 결정 round-trip

### 디버깅 + 일부 구조

**C. LISWAP-6 cleanup segfault** (backlog [P2])
- swap mode 정상 종료 후 process exit 시 SIGSEGV
- 추정 원인: Drop ordering 또는 `cl_mem` release ↔ `rpcmem_free` race
- 참고 파일: `engine/src/buffer/rpcmem_alias_buffer.rs`, `engine/src/models/weights/rpcmem_secondary.rs::RpcmemLayerRegion::Drop`
- 1-2일 디버깅 + 구조 변경

### 권장

본 sprint 종결 후 자연스러운 후속은 **A (INV-LAYER baseline 해소)** — 가장 큰 구조 부채.
단 multi-sprint이므로 진입 전 Architect 설계 라운드 필요 (rule별 해소 방법 결정).

진입 절차:
1. 사용자 후보 선택 (A/B/C)
2. 해당 작업 entry handoff 별도 작성 (sprint plan + 게이트 정의)
3. 위임 chain 시작 (A: Architect → Impl → Tester / B: Architect 결정 → Impl / C: Impl 디버깅)

---

## Landmines / 미해결

### 본 sweep에서 노출된 사전 이슈 (본 변경 무관)

1. **호스트 OpenCL test 22~24건 panic** (병렬 실행 시): `memory::opencl::unified::test_map_write_unmap_cycle` 등. host에 GPU 없을 때 OpenCL setup race. backlog [P3], sanity-check skill `--exclude-tests backend::opencl` 권장.
2. **`qnn_oppkg` workspace clippy 53 errors**: third_party QNN SDK 미설치. `cargo build -p ...`로 specific crate 사용 시 회피. `--exclude qnn_oppkg` 권장.
3. **`cargo clippy -p llm_rs2 --tests` useless_format**: `test_auf_gguf_byte_equivalence.rs:288`, `microbench_svm_throughput` test 등. master에서도 재현. lib 단위는 clean. 별도 cleanup mini-sprint 후보.

### 사후 후속 영향

- `compute_qcf_swap` → `compute_qcf_weight_swap` rename으로 src 4 + arch/spec/docs 3 파일 변경 — IPC 무관 (in-memory 함수). external client 영향 없음 확인됨.
- `QcfWarmupCtx`/`QcfWarmupConfig`는 pub struct이지만 generate.rs + eval/runner.rs 두 caller 외 호출 없음. 외부 client 영향 0.
- `alloc_alias_weight_buffer` unsafe 마킹 후 caller가 unsafe block 안에서 명시 호출. 향후 신규 alias buffer 사용처는 unsafe block + SAFETY 주석 의무.
- `ChatKvMode::Standard` Box wrap으로 enum match arm 1 hop indirection. chat 모드는 token-per-turn 동작이라 hot path 영향 무시.

### "이 길은 가지 마라"

- **Sprint 4-4-2.3 잔여 (3d/3e/4-4-2.4) — 폐기됨**. generate.rs legacy 보존 + 다수 바이너리 분할 방향 전환. 직전 handoff `[[refactor-sweep-complete-2026-05-21]]` 명시. 재진입 금지.
- **perf_recovery_post_step4 — 종결 (CLOSED)**. baseline outdated 확인. 재진입 금지.
- **EuroSys 2027 paper-driven 우선권 표기 — 의미 만료** ([[eurosys2027-post-paper]]). backlog의 paper critical 마킹은 사후 재평가 대상. 우선순위 자동 down-rank 아님 — 가치 기준 별도 판단.

### Design 4건 silence 모두 해소 — 향후 신규 silence 도입 시

본 sweep으로 design `#[allow]` 4건 모두 정식 해소. **향후 clippy 신규 silence 도입은 임시 silence가 아닌 별도 sprint에서 구조적 해소**를 default로. 임시 silence는 단기 cleanup 가치 없는 경우만 (또는 rust 신규 lint 즉각 대응).

---

## 참고 자료

- 직전 handoff: `.agent/todos/handoff_refactor_sweep_complete_2026_05_21.md` (sprint 직전 sweep + worktree 정리)
- 메모리: [[eurosys2027-post-paper]] — paper 종결 후 우선순위 컨텍스트
- 메모리: [[refactor-spec-qcf-clippy-2026-05-21]] — sprint 직전 sweep 상세
- 메모리: [[layered-architecture-decision]] — Phase 4 layered arch (INV-LAYER baseline 근거)
- backlog: `.agent/todos/backlog.md` (post-paper 표기 갱신됨)

## 자기점검 결과

- [x] 진입 문장이 한 줄로 명확? `"다음 sprint 결정"` — 남은 후보 A/B/C 중 선택이 첫 과제
- [x] "왜 멈췄는가"? 남은 구조 후보 모두 결정(B) 또는 multi-sprint(A) 또는 디버깅(C) — 사용자 결정 대기
- [x] 가장 큰 landmine 표면화? 본 변경 무관 사전 이슈 3건 + design silence 정책 + paper 표기 의미 만료
- [x] 검증 게이트 수치/명령? clippy clean / lib test 141+84+52+38+223 PASS / spec 9+5+17+3 PASS / origin sync 0/0
- [x] 길이 적정? 본문 ~600 토큰 — sprint 2개 결산 + 후보 3개 + landmine 모두 압축 한계 도달
