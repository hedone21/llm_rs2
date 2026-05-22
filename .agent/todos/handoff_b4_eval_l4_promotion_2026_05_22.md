# Handoff: B-4 observability/eval → session/eval L4 격상 종결 → 다음 sub-sprint 결정

**작성**: 2026-05-22
**HEAD**: `52a0a60b style: B-4-3 — cargo fmt cleanup`
**브랜치/Worktree**: `worktree-b4_eval_l4_promotion` (`.claude/worktrees/b4_eval_l4_promotion`)
**다음 세션 진입 문장**: `"B 다음 sub-sprint 결정 (B-5 / B-3)"`
**선행 진입점**: [[handoff_b2_profile_inversion_2026_05_22]]

---

## TL;DR

INV-LAYER baseline 점진 해소 sprint B의 세 번째 sub-sprint B-4 완료. `observability/eval/*` 6 파일을 `session/eval/`로 mechanical 이동 (L4 격상)하여 INV-LAYER-004 적용 대상에서 자연 제거. ARCHITECTURE.md §13.8-I (observability sub-module L4 promotion) 정책 신규 명문화. baseline **254 → 220 (-34)**. **멈춘 이유: B-4 종결 + 다음 sub-sprint(B-5/B-3) 사용자 결정 대기**.

---

## 진행 상태

### 본 sprint commits (4건)

| HEAD | scope | 변경 |
|---|---|---|
| `afeb1565` | docs(arch) | B-4-0 §13.8-I 정책 + V-16/V-28/V-29 RESOLVED 마킹 + Resolution Log 3행 |
| `ae3a1d42` | refactor(layer) | B-4-1 6 파일 mechanical move + 외부 caller 5건 path 갱신 + BC re-export 1 sprint 한정 |
| `51a8f661` | docs(arch) | B-4-2 V-16/V-28/V-29 Resolution Log SHA `<TBD>` → `ae3a1d42` 일괄 치환 |
| `52a0a60b` | style | B-4-3 cargo fmt cleanup (llama_layer.rs 사전 unformatted + obs/mod.rs + session/eval/runner.rs) |

### 게이트 결과

| 게이트 | 결과 |
|---|---|
| `cargo build -p llm_rs2` (default features) | PASS |
| `cargo build --release` | PASS (64초) |
| `cargo fmt --all --check` | clean 0 |
| `cargo clippy --lib --no-deps -- -D warnings` | clean 0 |
| `cargo test --lib -p llm_rs2` | **1205 PASS** / 21 FAIL (1204→1205 +1, 22→21 -1, 모두 사전 device-required) |
| `cargo test --test spec inv_layer` | **8 PASS / 0 FAIL** |
| **baseline JSON** | 254 → **220** (-34) |
| `grep -rn "observability::eval" engine/src/` | 1건 (BC re-export 라인 자체만) |

### baseline 변동 누적 (B-1~B-4)

| 단계 | 변동 | 종료 baseline |
|---|---|---|
| 진입 시 (Step 5b 종결 후) | — | 296 |
| B-1 EvictMethod relocation | -9 | 287 |
| B-2 profile inversion | -33 | 254 |
| B-4 eval L4 격상 | -34 | **220** |
| **누적 감소** | **-76 (-25.7%)** | |

### Pattern별 해법 (B-4)

| 단계 | 방식 | 위반 ID | baseline 감소 |
|---|---|---|---|
| B-4-1 | mechanical git mv | V-16, V-28, V-29 (eval 34건 전수) | -34 |

### ARCHITECTURE.md 정책 신규 — §13.8-I

`observability/` 산하 sub-module이 (1) L3 도메인에 다수 의존(5건 이상) (2) backend instantiation 코드 존재 (3) caller가 L4(`session/`)·L5(`bin/`)에 한정 — 3 조건 충족 시 L4로 격상하여 INV-LAYER-004 대상에서 자연 제거. §G(shared identifier promotion) / §H(instrument macro helper)와 직교한 *모듈-level 재배치 해소* 정책.

**적용**: `observability/eval/` → `session/eval/` (B-4). 향후 동일 패턴 발견 시 본 정책 참조 (단 `observability/profile/*`은 backend instantiate 안 하므로 §H가 적용 패턴).

---

## 다음 작업

B sprint 후속 후보 2건. 권장 순서: **B-5 → B-3**.

### B-5. INV-LAYER-001 L1→상위 31건 (3~5일, 권장)

- V-01 (gpu_self_meter trait 추출) + V-02 (tensor_partition L2 이동) + V-03 (LayerSlot/TransformerLayer opaque handle)
- backend별 분산: cpu 10, opencl 7, qnn_oppkg 5, cuda_embedded 5, cuda_pc 3
- 각 V-?? 별 PR 1개 권장 (정확성 검증을 backend별로 격리)
- 위임: senior-implementer (.cl 영역 + NEON path 일부 포함)
- 예상 baseline 감소: ~31건 (220 → ~189)

### B-3. Backend capability trait 추출 (5~7일, 가장 큰 작업)

- L3 코드의 `CpuBackend`/`OpenCLBackend`/`neon` 직접 의존 ~85건 해소
- `Backend` trait capability method 확장 (`as_opencl`, `as_cuda` 등)
- V-13 (KiviCache downcast) / V-17 (layers downcast 다수) / V-20 (models/transformer downcast) 별 PR 분할 필수
- 위임: architect (trait 설계 SoT) → senior-implementer (NEON path 영향)
- 예상 baseline 감소: ~75건 (220 → ~145)

### 위임 prompt 초안 (B-5 진입 시)

architect에게 우선 위임: "B-5 sub-sprint 진입. V-01/V-02/V-03 각각의 trait 설계 + 5 backend별 영향 분석. PR 분할 전략 결정 (각 V마다 1 PR vs 묶음). ARCHITECTURE.md §13.5 V-01~V-03 행 + §13.8-J(또는 §I 후속) 정책 결정 필요."

---

## Landmines / 미해결

### 본 sprint과 무관한 사전 회귀 (P3, 누적)

1. **`crates/qnn_oppkg/src/interface.rs` unresolved imports 17건+** — commit `d930801a feat(qnn): M1` 시점부터 존재. `crate::qnn::*` symbol resolution 실패. 본 B-4 sprint 중 IDE diagnostics 발생했으나 무관 확인. 별도 backlog.
2. **`backend::opencl::*` host test 21건 device-required FAIL** — 기존 P3.
3. **`cargo build --no-default-features` 사전 회귀** — `swap_dispatch.rs:441 map_weights_for_cpu` (B-2 handoff에서도 명시).
4. **`unused_imports` warning** — B-4-1 implementer의 BC re-export(`pub use crate::session::eval;`)에 대한 `#[deprecated]` attribute는 warning 0 유지 확인됨. 향후 1 sprint 후 cleanup 필요.

### B-4 후속 cleanup backlog

- **BC re-export 제거 (1 sprint 후)** — `engine/src/observability/mod.rs:2`의 `#[deprecated]` re-export는 B-5/B-3 진입 시점에 제거 검토. 외부 caller가 모두 `session::eval` 경로로 갱신됐고 in-tree에는 더 이상 호출자가 없음 (`grep -rn "observability::eval" engine/src/` = 1건 = re-export 자체만).
- **`crates/qnn_oppkg/` interface.rs 회귀 처리** — 별도 sprint.

### Architect agent의 master worktree 직접 land 문제 (재발)

- B-2 sprint와 동일하게 B-4-0에서도 architect가 worktree(`b4_eval_l4_promotion`) 안이 아닌 **master worktree**(`/home/go/Workspace/llm_rs2`)에 변경을 land함.
- 해법은 정착됨: `git diff > patch` (master) → `git restore` (master) → `git apply patch` (worktree).
- 다음 sub-sprint에서 architect 위임 시 prompt에 worktree 경로를 명시했음에도 동일 현상 발생 가능. 위임 후 즉시 worktree status 확인 + master patch 추출 로직 그대로 적용.

### "이 길은 가지 마라"

- **BC re-export를 영구 유지하지 말 것** — B-4-1에서 1 sprint 한정으로 도입. 외부 contributor 모두 신규 경로 사용 중. 영구화하면 §13.8-I의 단방향 격상 의미 약화.
- **EvalSink trait inversion 시도하지 말 것** — 본 sprint 진입 전 비교한 선택지 B. eval은 backend·workspace·KV cache를 본질적으로 instantiate해야 하므로 trait inversion은 구조적 모순. §13.8-I가 정답.
- **session/eval/eval_loop.rs LOC가 큼 (1097)** — split 욕구 발생 가능하나 본 sprint scope 외. 외과적 변경 원칙.

### microbench skip 근거

- mechanical move (behavior change 0) + path 갱신만이라 device TBT 측정 ROI 0.
- host release build PASS + INV-LAYER spec test PASS로 충분.

---

## 참고 자료

- 메모리: [[layered-architecture-decision]] — sprint B 원안 (2026-05-16)
- 직전 handoff: `.agent/todos/handoff_b2_profile_inversion_2026_05_22.md`
- B-1 handoff: `.agent/todos/handoff_b1_evict_method_relocation_2026_05_22.md`
- 정책 문서: `ARCHITECTURE.md` §13.8-I (신규), §13.5 V-16/V-28/V-29 RESOLVED, §13.5 Resolution Log 3행
- 변경된 spec: `spec/41-invariants.md` INV-LAYER-004 §13.8-I 예외 단락 추가
- 이동된 6 파일: `engine/src/session/eval/{eval_loop,eviction_hook,kivi_hook,hook,output,qcf_helpers}.rs`
- BC re-export: `engine/src/observability/mod.rs:2` (1 sprint 한정, 다음 sprint cleanup)

## 자기점검 결과

- [x] 진입 문장이 한 줄로 명확? `"B 다음 sub-sprint 결정 (B-5 / B-3)"`
- [x] "왜 멈췄는가"? B-4 종결 + 다음 sub-sprint 결정 대기
- [x] 가장 큰 landmine 표면화? architect agent의 master worktree land 재발 + BC re-export 1 sprint 한정 cleanup backlog
- [x] 검증 게이트 수치/명령? `cargo clippy --lib clean 0`, `cargo test --lib 1205 PASS`, `test_inv_layer 8 PASS`, baseline 254→220
- [x] 길이 적정? 본문 ~700 토큰, 4 commits + 패턴별 매트릭스 + 2 후속 후보 + 4 landmine
