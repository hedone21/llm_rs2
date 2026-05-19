# Handoff — Phase 4 (Migration Step 2) 종결 선언 (2026-05-19)

**작성**: 2026-05-19 (Phase 4-4-2.1 + sprint exit + 종결 결정 직후)
**master HEAD**: `5cc0d87d`
**대응 단계**: `ARCHITECTURE.md §13.7 Migration Step 2 (L5/L4 분리)`
**상태**: **CLOSED** — G3 부분 미달 인정, fallback path 영구 자산화

---

## 1. Step 2 (= Phase 4) 전체 sub-step 결과

`ARCHITECTURE.md §13.7` 기준 Step 2의 5개 sub-step 매핑:

| Sub-step | 매칭 Phase | Commit | 결과 |
|---|---|---|---|
| 2-1 외곽 추출 | **Phase 4-1** | `f637722e` | ✓ `SessionInitCtx` + `session/cli/`. main() 7,051 LOC 중 helper 7건 추출 |
| 2-2 trait 정의 + 빌더 | **Phase 4-2** | `584496b7` | ✓ 6 trait + `DecodeLoop` + `DecodeLoopBuilder` typestate + defaults 5종 + trybuild |
| 2-3 첫 구현체 (`ModelForward`) | **Phase 4-3** | `c63190d1` | ✓ ModelForward + KiviForward + OffloadForward. probe microbench 검증 |
| 2-4 main() 조립자화 | **Phase 4-4 (A~D) + 4-4-2** | 여러 commit | ▲ **부분 달성** — production happy path 통합 ✓, fallback path 5,782 LOC 잔존 |
| 2-5 chat REPL 전면 재작성 | **Phase 4-5** | `619dd655` | ✓ `ChatTurnExec` 폐기, `session/chat/{repl,turn,stop_condition}` 재작성 |

---

## 2. Commit map (시간순)

### Phase 4-1 ~ 4-3 (기반)

| Commit | 작업 |
|---|---|
| `f637722e` | Phase 4-1 외곽 추출 (`SessionInitCtx` + `session/cli/`) |
| `85ff756c` | Phase 4-2 C1 — 6 trait 정의 |
| `79efe21f` | Phase 4-2 C2 — defaults 5종 |
| `ee1a1ae0` | Phase 4-2 C3 — decode_loop + builder |
| `584496b7` | Phase 4-2 C4 — trybuild + spec test |
| `3470ad1d` | Phase 4-3 C1 — ModelForward |
| `f5236073` | Phase 4-3 C2 — parity + trybuild |
| `c63190d1` | Phase 4-3 C3 — probe microbench (S25 Δ2.29%) |

### Phase 4-4 (main() 추출 — A~D)

| Commit | 작업 |
|---|---|
| `f6c491a1` | Phase 4-A — batch 분기 → `session/batch/` |
| `645a91ed` | Phase 4-B-1 — shared qcf/swap → `session::qcf_runtime` |
| `9db119c5` | Phase 4-B-2 — eval_ll → `session/eval/` |
| `ebaa7254` | Phase 4-C-1 — shared swap dispatch + weight dump → qcf_runtime |
| `5e5c5753` | Phase 4-C-2 — ppl + run_kivi_ppl → `session/ppl/` |
| `a2045823` | Phase 4-D — microbench/probe/stage 62 파일 → `engine/microbench/` |

### Phase 4-5 (chat 재작성)

| Commit | 작업 |
|---|---|
| `75edb358` | Phase 4-5 chat 전면 재작성 종결 |
| `c1a4b481` | Phase 4-5-g — multi-turn KV pos 보존 fix |
| `619dd655` | Phase 4-5-f P0 게이트 |
| `a15be30e` | R-tbt + R-chat 분석 종결 |
| `eacfe1a4` | Qwen2.5 chat garbage 원인 RESOLVED (model issue) |

### S-prep / S-cleanup / S-subcmd (CLI 정리)

| Commit | 작업 |
|---|---|
| `30320225` | S-prep — Args 사용 매트릭스 |
| `574fc833` | S-prep — 142 field 정정 + 17 sub-struct 분류 |
| `1eaa7c04` | S-cleanup batch 1 — dead Args 4개 제거 |
| `5e5d1743` | S-cleanup batch 2 — measurement-only Args 정리 |
| `a283b35a` | S-cleanup batch 3 — awqe → env var |
| `887037d1` | S-subcmd C1 — EvictionCmd subcommand |
| `f0a21d2c` | S-subcmd C2 — EvictionCmd + EvictionCommonArgs 통합 |
| `51f09c16` | S-subcmd C2.1 — eviction wrapper |
| `ba7d2cff` | S-subcmd C8 — docs/scripts/verify migration |
| `83c73dff` | S-subcmd C4-1~3 — --kv-mode subcommand args |
| `50792ec8` | S-subcmd C4-4a — mode-branch call sites |
| `6a18c5b6` | S-subcmd C4-4b — shim methods |
| `28055252` | S-subcmd C5/C6 옵션 B — hide legacy flags |
| `2eb3765d` | S-subcmd C5/C6 옵션 C — remove legacy flags |

### Phase 4-4-2 (현재 sprint — fallback 분해)

| Commit | 작업 |
|---|---|
| `fcc1ea87` | Sprint 1 — dump-importance → `session::dump_importance` |
| `15fc0fee` | Sprint 1 — standard happy path → `session::standard_happy` |
| `1b5b93b6` | Sprint 1 — DVFS warmup → `session::warmup` |
| `65ade7ea` | Sprint 2 — QCF `kv/weight` rename |
| `0534f21b` | 4-4-2 E1' 진입점 doc (4 sub-sprint 분할) |
| `7f693160` | 방향 A (G3 LOC-only) 결정 doc |
| **`1b674bd7`** | **4-4-2.1 — chunked prefill → `session::prefill`** |
| `5cc0d87d` | Sprint 종결 + 4-4-2.3/4 backlog 이동 |

---

## 3. 누적 지표

### generate.rs LOC 추이

```
시작 (Phase 4 이전)            :  13,022 LOC
Phase 4-1 (외곽 추출)          :  ~7,051 LOC      (-46%)
Phase 4-3 (ModelForward)       :  ~6,900 LOC
Phase 4-4 (A~D)                :   6,513 LOC
Sprint 1 (4-4-2 진입 직전)     :   6,318 LOC      (-3%)
Phase 4-4-2.1 (prefill 추출)   :   5,782 LOC      (-8.5%)
═══════════════════════════════════════════════════════
누적 감소율                    :  -56% (-7,240 LOC)
G3 목표 (~400 LOC)             :  미달 (추가 -93% 필요)
```

### 모듈 분포 (현재)

```
engine/src/session/
├── assembly/       (build_standard_loop, is_standard_happy_path)
├── batch/          (Phase 4-A)
├── chat/           (Phase 4-5 — repl, turn, stop_condition)
├── chat_ipc.rs     (core → session 이관)
├── cli/            (Args sub-struct 17개)
├── decode_loop.rs  (Phase 4-2 — typestate builder)
├── defaults.rs     (Phase 4-2 — no-op 5종)
├── dump_importance.rs  (Sprint 1)
├── eval/           (Phase 4-B-2)
├── forward/        (Phase 4-3 — ModelForward, KiviForward, OffloadForward)
├── init.rs         (Phase 4-1 — SessionInitCtx)
├── ppl/            (Phase 4-C-2)
├── prefill.rs      (Phase 4-4-2.1)  ← 729 LOC, 가장 큰 단일 모듈
├── qcf_runtime.rs  (Phase 4-B-1/4-C-1)
├── samplers/       (Phase 4-2)
├── standard_happy.rs  (Sprint 1)
├── traits.rs       (Phase 4-2 — 6 trait + StepCtx)
└── warmup.rs       (Sprint 1)
```

### Test coverage

| 분류 | 수 |
|---|---|
| session lib test | 52 |
| INV-LAYER spec test | 8 |
| trybuild typestate | 8 |
| 전체 spec test | 661 (parallel race 외 PASS) |

### 검증 게이트 통과 기록

| Phase | bit-identical (S25 32 tok) | avg_tbt Δ |
|---|---|---|
| 4-1 ~ 4-2 | ✓ | 0% |
| 4-3 | ✓ S25 + 호스트 | 1.53% / 2.29% |
| 4-4 (A~D) | ✓ | <1% |
| 4-5 | ✓ P0 게이트 | — |
| **4-4-2.1** | ✓ | +2.3% (gate ≤5%) |

---

## 4. G1/G2/G3 도달도

| 목표 | 상태 | 비고 |
|---|---|---|
| **G1 — 6 trait SOLID 분해** | ✓ **달성** | production happy path 완전 위임. fallback path도 동일 trait 가능하나 미적용 |
| **G2 — 외부 plugin point** | ✓ **달성** | `Box<dyn Trait>` (INV-LAYER-006) + `HasForward` typestate (INV-LAYER-007) |
| **G3 — main() ~400 LOC** | ✗ **미달, 동결** | 5,782 LOC에서 종료. fallback path (chunked prefill + collector + experiment writer + decode 본체)가 영구 자산으로 잔존 |

---

## 5. 종결 결정 (2026-05-19)

### 결정 요약

Phase 4 (= Migration Step 2)를 **현 시점에서 종결 선언**. G3 ~400 LOC 목표 미달 인정. fallback path는 영구 자산화하고 `bin/generate.rs`에 잔존.

### 근거

1. **G1/G2 달성**: production happy path가 완전히 trait 위임으로 동작. 외부 기여자가 단일 trait 구현으로 backend/policy/algo 추가 가능
2. **G3 ROI 한계**: 4-4-2.3 decode_fallback 추출 (~2,260 LOC)은 단일 sprint 위험 큼. 추가 분해 설계 + ctx 70+ 필드 mut state 정리 + 정확성 게이트 비용이 G3 가치 대비 과대
3. **외부 공개 critical path 우선**: Migration Step 3~5 (L1/L2/L3 정리, cross-cutting 분리)가 외부 공개 인터페이스 안정화에 더 시급
4. **방향 A 일관성** (handoff §2.5): G3 LOC-only mechanical move는 trait 흡수 가치 없음 — 가치 없는 sprint를 강행하지 않음

### 보존된 자산

- 6 trait API (`session/traits.rs`) — 외부 공개 가능한 안정 인터페이스
- `DecodeLoopBuilder` typestate — 컴파일 타임 안전성
- `build_standard_loop` — production assembler 진입점
- INV-LAYER spec test infrastructure — 점진 축소 baseline

---

## 6. 잔여 backlog (영구 자산화)

| 항목 | 우선 | 비고 |
|---|---|---|
| Phase 4-4-2.3 decode_fallback 추출 (~2,260 LOC) | **P1 backlog (영구)** | 4-4-2.2 흡수 포함. 외부 공개 시 fallback path 노출 결정 필요 시 재개 |
| Phase 4-4-2.4 post-process 추출 (~206 LOC) | **P2 backlog (영구)** | 4-4-2.3 후 |
| INV-LAYER baseline JSON 축소 (현재 309건) | P3 | Step 3~5 진행하며 자연 감소 |

revisit gate 없음 — 외부 요청 또는 명시적 결정 변경 없이는 재개 안 함.

---

## 7. 다음 단계: Migration Step 3 진입

`ARCHITECTURE.md §13.7 Step 3 — L1/L2 경계 정리`로 진입.

진입점: **`.agent/todos/handoff_migration_step3_entry_2026_05_19.md`**

진입 문장: **"Migration Step 3 진행"** 또는 sub-sprint 단위로 **"Step 3-A 진행"**

---

## 8. 참조 문서

- `ARCHITECTURE.md §13.1~13.8` — Layered architecture 사양
- `arch/inference_pipeline.md` — Phase 4 6 trait 설계 (Step 2-2~2-5)
- `.agent/todos/handoff_phase4_2_entry_2026_05_16.md` — Phase 4-2 진입 (완료)
- `.agent/todos/handoff_phase4_3_entry_2026_05_17.md` — Phase 4-3 진입 (완료)
- `.agent/todos/handoff_phase4_4_entry_2026_05_17.md` — Phase 4-4 (A~D) 진입 (완료)
- `.agent/todos/handoff_phase4_DABC_complete_2026_05_18.md` — Phase 4-D + S-subcmd 진입 (완료)
- `.agent/todos/handoff_phase4_5_complete_2026_05_18.md` — Phase 4-5 chat 종결
- `.agent/todos/handoff_phase4_5_p0_gate_2026_05_19.md` — Phase 4-5 P0 게이트
- `.agent/todos/handoff_phase4_4_2_e1prime_entry_2026_05_19.md` — Phase 4-4-2 진입
- `.agent/todos/handoff_phase4_4_2_sprint_exit_2026_05_19.md` — Phase 4-4-2 sprint exit
- `.agent/todos/backlog.md` — 잔여 P1/P2 항목
