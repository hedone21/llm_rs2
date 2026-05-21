# Handoff — Phase 4-4-2 Sprint 종결 + 다음 진입 옵션 (2026-05-19)

**작성**: 2026-05-19 (Phase 4-4-2.1 prefill 추출 완료 + 4-4-2.2 흡수 결정 직후)
**master HEAD**: `1b674bd7`
**스프린트 상태**: **EXIT** — generate.rs 추가 추출은 backlog로 미룸

---

## 1. 본 스프린트 성과

### 완료

| Commit | 작업 | generate.rs LOC |
|---|---|---|
| `fcc1ea87` | dump-importance → `session::dump_importance` | -73 |
| `15fc0fee` | standard happy path → `session::standard_happy` | -78 |
| `1b5b93b6` | DVFS warmup → `session::warmup` | -77 |
| `65ade7ea` | QCF kv/weight rename | (lib) |
| `0534f21b` | Phase 4-4-2 E1' 진입점 doc | (doc) |
| `7f693160` | 방향 A (G3 LOC-only) 결정 doc | (doc) |
| **`1b674bd7`** | **chunked prefill → `session::prefill`** | **-536** |

generate.rs: 6,513 → **5,782 LOC** (-731, -11.2%)

### Phase 4-4-2.1 게이트 결과

- cargo build / clippy: 회귀 0
- `cargo test --lib session::`: 52 PASS
- `cargo test --test spec`: 회귀 0 (parallel race 외)
- **S25 Adreno 32 tok bit-identical** ✓ — `Frances nuclear power industry is facing a crisis...`
- **avg_tbt n=5**: master 71.09 ms / branch 72.73 ms = **+2.3%** (gate ≤5%)

---

## 2. Phase 4-4-2.2 흡수 결정 (2026-05-19)

### 결정 내용

Phase 4-4-2.2 (transition block 단독 sprint) **skip** → 4-4-2.3 (`decode_fallback`)에 prologue로 흡수.

### 근거

| # | 항목 |
|---|---|
| 1 | **ROI**: 실측 transition = L1841~1920 (**80 LOC**, handoff 추정 190 LOC보다 훨씬 작음). ctx 보일러플레이트 30 LOC 빼면 순감 ~50 LOC. prefill의 1/10 수준 |
| 2 | **응집성**: SwitchHw → workspace alloc → decode loop가 같은 backend 상태에 의존. 분리 시 backend mut transfer가 인위적 |
| 3 | **R6 회피**: `// === GENERATION PHASE ===` 블록의 `{` open scope가 transition+decode를 같은 scope로 묶음. 단독 추출 시 `{` 처리가 어색 |
| 4 | **방향 A 일관성**: G3 LOC-only 정책 — 추출 가치가 낮으면 추출 안 함 |
| 5 | **handoff doc §4 명시 옵션**: "(또는 `session::decode_fallback`에 prologue로 흡수)" 이미 제시됨 |

### 갱신된 sub-sprint 일정

| Sub | 영역 | 모듈 | 상태 |
|---|---|---|---|
| ~~4-4-2.2 transition~~ | (skip) | (4-4-2.3에 흡수) | **CANCELLED** |
| **4-4-2.3 decode_fallback** | L1841~4099 (~2,259 LOC) | `session::decode_fallback` | 진입 가능 (단, 추가 분해 권장) |
| **4-4-2.4 post-process** | L4101~4307 (~206 LOC) | `session::post_process` | 4-4-2.3 후 |

---

## 3. 본 스프린트 종료 결정 (2026-05-19)

generate.rs 추가 추출(4-4-2.3 / 4-4-2.4)은 **backlog로 이동**.

근거:
- 4-4-2.3은 ~2,260 LOC 단일 extraction으로 위험 큼. 추가 분해 설계 필요
- 4-4-2.1로 main() 11% 감소 + fallback contour 격리 달성 — Phase 4-4-2 진입 목표(G3 LOC) 부분 달성
- 다음 sprint에서는 **generate-free** 작업이 ROI 좋고 위험 낮음

---

## 4. 다음 sprint 진입 옵션 (generate-free)

### 즉시 진입 가능 (사용자 결정 대기 없음)

| 항목 | 영향 파일 | 규모 | 권장도 |
|---|---|---|---|
| **[P2] llm_rs2 lib clippy 회귀 cleanup** | `core/qcf/layer_importance.rs`, `models/weights/{async_swap,noise_table,phase_aware_swap,swap_executor}.rs`, `session/{chat/repl,cli}.rs`, `backend/opencl/mod.rs:5318` | 30건 (doc lazy_continuation 29 + unsafe pointer 1) | ★★★ — sanity gate 복원 |
| **[P0] WSWAP-6-C** Primary cl_mem release bg worker | `backend/opencl/weight_swap.rs`, 신규 `release_worker.rs` | 100~150 LOC, mpsc + worker thread | ★★ — 173 ms 절감, EuroSys critical |
| **[P1] WSWAP-6-F** `write_buffer(blocking=true)` → async | `backend/opencl/weight_swap.rs` | 50~80 LOC | ★★ — 100~150 ms 절감 |
| **[P2] LISWAP-6 cleanup segfault** | `buffer/rpcmem_alias_buffer.rs`, `models/weights/rpcmem_secondary.rs::Drop` | Drop ordering 정리 | ★ — 측정 환경 정리 |
| **[P3] qnn_oppkg_poc clippy `#[allow]`** | `crates/qnn_oppkg_poc/src/lib.rs:725` | trivial silence | ★ — trivial |

### WSWAP-6 시리즈 (EuroSys 2027 critical path)

| 항목 | 절감 | 우선 |
|---|---|---|
| **[P0] WSWAP-6-A** Fused SOA convert kernel (`engine/kernels/cvt_q4_0_noshuffle.cl` 신규) | 500~650 ms | Senior Implementer 필요 |
| **[P0] WSWAP-6-C** Primary cl_mem release bg worker | 173 ms | 즉시 |
| **[P1] WSWAP-6-F** async write | 100~150 ms | 즉시 |
| **[P1] WSWAP-6-B** AOS heap copy 제거 | 80~100 ms | lifetime 신중 |
| **[P1] WSWAP-6-PREFAULT** Eager prefault | ~328 ms | CLI 옵션 생략 시 generate-free |
| **[P2] WSWAP-6-D** Prefault 범위 축소 | 40 ms | PREFAULT 후 |
| **[P2] WSWAP-6-E** stage label rename (cross-crate) | 0 (정합성) | A/B/C/D/F 후 batch |

### QCF Sprint 2 deferred

- **DegradationEstimator action key prefix** `kv.*` / `weight.*` — `core/qcf/*` + 호출처 string literal. 일부 generate.rs 호출처 있을 가능성 → grep으로 확인 필요
- **`shared::QcfEstimate.estimates` HashMap key prefix** — `shared/src/lib.rs`, `manager/src/*` 동시 갱신, 1버전 back-compat 권장

### 큰 작업 (multi-sprint)

| 항목 | 상태 |
|---|---|
| [P0] M2.A~M2.J QNN OpPackage layer-level | Architect 위임 대기 |
| [P0] M3.4 RED pos baked | 사용자 architectural decision 대기 (D-D vs D-E) |
| [P0] Long context CPU attention | `backend/cpu/neon.rs:235`, Senior Implementer 영역 |
| [P2] Adreno noshuffle GEMV cross-run tuning | Senior Implementer, Adreno 실측 |

---

## 5. backlog 잔류 (generate.rs 관련)

| 항목 | 우선 | 비고 |
|---|---|---|
| Phase 4-4-2.3 decode_fallback 추출 (~2,260 LOC) | P1 | transition 흡수. 추가 분해 권장 (decode prologue / eviction trigger / swap dispatcher / experiment writer 등) |
| Phase 4-4-2.4 post-process 추출 (~206 LOC) | P2 | 4-4-2.3 후 |
| `--eager-prefault` CLI 옵션 (WSWAP-6-PREFAULT 부수) | P2 | 생략 가능 (env var로도 가능) |

---

## 6. 환경 / 규칙 (재확인)

- **언어**: 모든 응답 한국어, 기술 용어/코드 식별자 원문 유지
- **EnterWorktree**: 코드 변경 작업 시 worktree 격리 필수
- **테스트 기본 모델 포맷**: GGUF
- **Android 벤치 스레드**: Galaxy S25는 6T만
- **TBT metric**: 항상 avg_tbt (tok0 inclusive). rest_tbt 단독 비교 금지
- **성능 측정**: `--profile` 없이
- **신규 spec test**: `engine/tests/spec/`
- **완료 시 자동 commit + `notify-send`**
- **`.cl` 커널 수정**: Senior Implementer만, Adreno 실측 교훈 준수
- **Background job 임시 파일**: `$CLAUDE_JOB_DIR`
- **worktree symlink 필수**: `third_party/`, `libs/` (qnn_sdk + libOpenCL stub)

---

## 7. 참조 문서

- `arch/inference_pipeline.md` — Phase 4 6 trait 설계
- `ARCHITECTURE.md` §13 — Layered architecture INV-LAYER 정의
- `.agent/todos/handoff_phase4_4_2_e1prime_entry_2026_05_19.md` — Phase 4-4-2 진입 (완료, §2.5 방향 A 결정)
- `.agent/todos/backlog.md` — 미해결 P0/P1 항목 카탈로그
- `papers/eurosys2027/_workspace/experiment/swap_overhead_s25.md` — WSWAP-6 측정 보고

---

## 8. 다음 세션 진입 문장 (제안)

| 진입 후보 | 문장 예시 |
|---|---|
| **clippy 회귀 cleanup** (★★★) | "clippy 회귀 cleanup 진행" |
| WSWAP-6-C bg release worker | "WSWAP-6-C 진행" |
| WSWAP-6-F async write | "WSWAP-6-F 진행" |
| Phase 4-4-2.3 (generate 재개) | "Phase 4-4-2.3 진행" — backlog에서 끌어옴 |

특정 결정 없이 진입 시: **"clippy 회귀 cleanup 진행"** 이 ROI/위험 비율 최선.
