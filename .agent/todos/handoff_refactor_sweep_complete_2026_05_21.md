# Handoff: Refactor sweep 종결 → 다음 sprint 진입 결정

**작성**: 2026-05-21
**HEAD**: `bd72934c docs(handoff): step-4 ↔ sprint1 통합 머지 진입 handoff` (origin/master push 완료)
**다음 세션 진입 문장**: `"다음 sprint 결정"` — backlog 항목 선택 후 진입
**선행 진입점**: [[handoff_perf_recovery_post_step4_merge_2026_05_21]] — Phase 4-4-2.3 a/c/b 추출 + perf_recovery 종결 + generate split 방향 전환

---

## TL;DR

리팩토링 우선 기조로 cleanup 3 묶음 자율 처리 — clippy 회귀 34→0 + Manager PROTO-012 가드 + action name/IPC wire format dot prefix 통일. 31개 잔존 worktree + branch 일괄 정리, sprint1 handoff doc 1건 cherry-pick으로 보존. origin push 완료. 큰 리팩토링 작업은 본 sweep으로 대부분 종결, 잔여 후보는 generate 바이너리 분할 (설계 라운드 대기) 또는 성능 작업 (사용자가 마지막 순위로 미룸). **멈춘 이유: 본 sweep 완료 + 다음 sprint 사용자 결정 대기**.

---

## 진행 상태

### 본 세션 commits (5 + cherry-pick 1)

| HEAD | scope | 변경 |
|---|---|---|
| `0883d5a8` | style(clippy) | clippy 34건 → 0건. mechanical 30 + design 4건 `#[allow(...)]`. 16 files |
| `b5266563` | feat(manager) | PROTO-012 64KB MAX_PAYLOAD 가드. `manager/src/channel/{unix_socket,tcp}.rs::read_engine_message`. 2 files |
| `15c7e3da` | refactor(qcf) | action name + IPC wire format + ActionId + TOML + Lua + snapshot + spec/docs 모두 dot prefix 통일 (`kv.*` / `weight.*`). 64 files |
| `9dfca61a` | docs(backlog) | RESOLVED/CANCELLED 마킹 정리. 1 file |
| `bd72934c` | docs(handoff) | sprint1_auf_loader 미머지 handoff (`77e16eb6`) cherry-pick. 1 file |

### 게이트 결과

| 게이트 | 결과 |
|---|---|
| `cargo build -p llm_rs2 -p llm_shared -p llm_manager` | PASS |
| `cargo clippy -p llm_rs2 --lib -- -D warnings` | clean (34 → 0) |
| `cargo test -p llm_shared --lib --tests` | 38 + 27 PASS |
| `cargo test -p llm_manager --lib --tests` | 223 + 376 PASS (sim snapshot 갱신 포함) |
| `cargo test -p llm_rs2 --lib` | 1202 PASS, 24 FAIL = `backend::opencl::*` host device-required (사전 이슈, 본 변경 무관) |
| `cargo test -p llm_rs2 --tests` | 사전 import error (`llm_shared::auf`, `compute_qcf_weight_swap` stale — 본 sprint 무관) |
| `git rev-list --left-right --count origin/master...master` | 0 / 0 (push 완료) |

### Rename map (옵션 3)

```
kv_evict_sliding   → kv.evict_sliding
kv_evict_h2o       → kv.evict_h2o
kv_streaming       → kv.evict_streaming
kv_quant_dynamic   → kv.quant_dynamic
kv_merge_d2o       → kv.merge_d2o
layer_skip         → weight.skip
```

적용 layer: EngineCommand wire format (`#[serde(rename)]`) + ActionId enum + 60+ action name 호출처 sed + TOML quoted key + insta snapshot + Lua scripts + spec/docs.

### Worktree + branch 정리

- 31개 `.claude/worktrees/` worktree 분석: 27 MERGED + 3 detached/ref-less + 1 미머지 (`sprint1_auf_loader`)
- sprint1 handoff doc cherry-pick → master `bd72934c`
- 31개 worktree 제거 (14 일반 + 17 `--force` for untracked `libs/`/`third_party/` 빌드 산출물)
- 31개 branch ref 삭제 (30 `-d` + 1 `-D`)
- 남은 worktree: 4개 외부 path (사용자 작업용, 본 정리 범위 밖)

### backlog 변동

- **RESOLVED**: `[P2] llm_rs2 lib clippy 회귀` / `[P1] QcfEstimate + RequestQcf` (stale, 이미 구현) / `[P2] Manager 페이로드 가드 PROTO-012` / `[P2] QCF 명명 §Deferred IPC HashMap key 항목`
- **CANCELLED**: `[P2] Heartbeat/Response 타임아웃` (spec SEQ-087/088 미적용 정의와 충돌)
- **Deferred (별도 backlog)**: `DegradationEstimator` default curves key (`eviction`/`sliding`/`kivi`/`swift` — 추상 카테고리 layer)

---

## 다음 작업

후보를 우선순위로 정리. 사용자가 진입할 항목 선택 후 별도 entry handoff 작성.

### 리팩토링 우선 기조 유지 시

**A. [P2] generate 바이너리 분할 + Manager 통합** — backlog 등록만, 설계 라운드 대기
- 진입 문장 후보: `"generate 바이너리 분할 설계 진행"`
- 잠정 4 바이너리: gen-cli / gen-chat / gen-experiment / gen-resilience
- 필요 산출물: 분할 단위 + 책임 경계 + IPC 범위 + 마이그레이션 순서 + 게이트
- 위임: Architect (설계 라운드) → Implementer (바이너리별 cut/paste) → Tester (디바이스 게이트)
- 게이트 (예상): bit-identical (legacy generate 동일 모드 대비) + avg_tbt 회귀 0
- 재사용 자산: `engine/src/session/{init,cli,prefill,decode_fallback/{prologue,eviction_trigger,swap_dispatch}}` 6 모듈

### Architectural decision 대기

**B. [P0] M3.4 RED pos baked blocker** — QNN OpPackage path
- handoff: `.agent/todos/handoff_qnn_oppkg_m3_4_red_pos_baked_20260510.md`
- 결정 필요: 옵션 D-D (M2 ops 수정 +1.5주) vs D-E (scope 약화 +0.5주)

### 성능 작업 (사용자가 마지막 순위로 미룸)

**C. [P0] WSWAP-6-A/C/B/F/PREFAULT** — Weight Swap Overhead 1564→70 ms (EuroSys 2027 critical path)
- backlog "[P0] WSWAP-6-A: Fused SOA convert kernel" 외 4건
- 위임: Senior Implementer (`.cl` 커널 + Adreno 최적화)

**D. [P0] M2.A~M2.J** — QNN OpPackage Layer-level Graph (18~22일)
- backlog "[P0] M2.A~M2.J QNN-GPU Migration"
- handoff: `.agent/todos/feat_qnn_oppkg_m2.md`

**E. [P1] Long context CPU attention 최적화** — 4K context 35% → 80% (llama.cpp 대비)
- handoff: `.agent/todos/long_context_attention_optimization.md`

**F. [P1] Qwen CPU decode gap 해소** — per-op profiling 선행
- backlog "[P1] Qwen CPU decode gap"

### 권장

본 sprint 흐름의 자연스러운 후속은 **A (generate 바이너리 분할 설계 라운드)**. 단 설계 라운드 → 구현 → 게이트 순으로 복수 sprint가 필요한 큰 작업. 다른 작업으로 진입 시 사용자 명시 선택.

진입 절차:
1. 사용자가 후보 중 선택
2. 해당 작업의 entry handoff 별도 작성 (해당 sprint plan + 게이트 정의)
3. Architect/Implementer/Tester 위임 chain 시작

---

## Landmines / 미해결

### 본 sweep에서 발견된 사전 이슈 (본 변경 무관, 미해결)

1. **`engine/tests/` import errors** (multiple files): `llm_shared::auf`, `crate::core::math_utils`, `llm_rs2::core::math_utils`, `compute_qcf_weight_swap` 등 unresolved. Migration Step 4 (L3 재배치, `core/` 해체)의 잔여 test fix 누락. spec test 다수 영향. backlog [P2] 등록 필요.
2. **`backend::opencl::*` host test 24개 device-required FAIL**: 호스트 환경에 OpenCL device 없어 panic. CLAUDE.md 알려진 P3. host 회귀 게이트에서 본 모듈 제외 권장 (sanity-check skill에 `--exclude-tests backend::opencl` 패턴 추가).
3. **`qnn_oppkg` crate workspace 빌드 실패**: SDK 미설치. `cargo build -p ...`로 specific crate만 빌드 시 회피. workspace 빌드 시 `--exclude qnn_oppkg` 필요.

### Design 4건 (옵션 4 `#[allow]` silence) — 구조적 변경 필요 시 별도 sprint

- `not_unsafe_ptr_arg_deref` (opencl::alloc_alias_weight_buffer): public 시그니처 `unsafe fn` 변경 비용
- `type_complexity` (swap_executor::execute_on_slots, layer_importance::build_with_raws): 단일 사용 local, type alias 도입 vs 가독성 trade-off
- `large_enum_variant` (ChatKvMode): Standard variant 376 bytes. Box wrapping 시 hot path 영향 평가 필요
- `too_many_arguments` (run_qcf_warmup_workflow): 16 args. struct ctx refactor는 별도 sprint

### 옵션 3 호환성 주의

- IPC wire format 변경 (snake_case → dot prefix). External client 없으나 manager ↔ engine 두 binary 동시 업데이트 필요. 본 commit `15c7e3da`가 양쪽 동시 변경.
- TOML key에 dot이 들어가면 nested table separator로 해석 — quoted form 필수. 본 sprint에서 `manager/policy_config.toml` + test fixture 처리 완료. 향후 새로운 dot prefix key 추가 시 동일 패턴 유지.

### "이 길은 가지 마라"

- Phase 4-4-2.3 잔여 (3d/3e/4-4-2.4) — generate.rs legacy 보존 + 다수 바이너리 분할 방향 전환으로 폐기. 다시 진입 금지. [[generate-split-binaries]] 참고.
- perf_recovery_post_step4 — 옵션 A 종결 (CLOSED). handoff baseline outdated 확인. 재진입 금지.

---

## 참고 자료

- 메모리: [[refactor-spec-qcf-clippy-2026-05-21]] — 본 sprint 상세 기록
- 메모리: [[generate-split-binaries]] — generate 분할 결정 (2026-05-21)
- 메모리: [[qcf-naming-decision]] — QCF_kv vs QCF_weight 2-tier 분리 (2026-04-27)
- 메모리: [[layered-architecture-decision]] — Phase 4 layered architecture
- backlog: `.agent/todos/backlog.md` (RESOLVED/CANCELLED 정리됨)
- 직전 handoff: `.agent/todos/handoff_perf_recovery_post_step4_merge_2026_05_21.md`

## 자기점검 결과

- [x] 진입 문장이 한 줄로 명확? `"다음 sprint 결정"` — 본 sprint 종결 후 다음 sprint 선택이 첫 과제
- [x] "왜 멈췄는가"? 본 sweep 완료 + 다음 sprint 사용자 결정 대기 (큰 리팩토링 후보 없음)
- [x] 가장 큰 landmine 표면화? engine/tests/ stale import (Migration Step 4 잔여) + QCF rename IPC 호환성 주의 + Phase 4-4-2.3 잔여 폐기 결정
- [x] 검증 게이트 수치/명령? `cargo clippy ... -- -D warnings: 0`, `cargo test ... 38+27+223+376 PASS`, `cargo test -p llm_rs2 --lib: 1202 PASS / 24 FAIL (device-required)`
- [x] 길이 적정? 본문 ~500 토큰 권장 — 다소 길지만 5 commits + 31 worktree 정리 + 6 후보 sprint 안내라 압축 한계
