# Handoff: Phase 4-1 완료 → Phase 4-2 진입 대기

**작성**: 2026-05-16
**HEAD**: `f637722e feat(session): extract SessionInitCtx — Phase 4-1 init outer hull (~929 LOC)`
**다음 세션 진입 문장 (사용자)**: "Phase 4-2 진행" 또는 "session/traits.rs 작업 시작"

---

## TL;DR

외부 공개 대비 엔진 레이어드 리팩토링 **Task #4 Phase 4-1 (외곽 추출) 완료**. main() 7,051 → 6,122 LOC. S25 OpenCL + Jetson CUDA 양쪽에서 32 토큰 bit-identical 검증 PASS. **다음은 Phase 4-2 — `session/traits.rs` (6 trait) + `session/decode_loop.rs` (DecodeLoop + typestate Builder) + `session/defaults.rs` (no-op) + trybuild negative test 구현**. 사용자 검토 완료 후 진입.

---

## 진행 상태

### Task 리스트 (8단계)

| ID | 상태 | 작업 |
|---|---|---|
| #1 | ✅ | ARCHITECTURE.md + spec INV-LAYER-001~005 |
| #2 | ✅ | UNRESOLVED-A~E 5건 결정 |
| #3 | ✅ | spec test + baseline + layer_lint 도구 |
| **#4** | 🔄 in_progress | **L5/L4 분리** (Phase 4-1 ✅ / **4-2 진입 대기** / 4-3/4-4/4-5 pending) |
| #5 | ⏳ blocked | L1/L2 경계 정리 |
| #6 | ⏳ blocked | L3 도메인 재배치 |
| #7 | ⏳ blocked | Cross-cutting 분리 |
| #8 | ⏳ blocked | /simplify 코드 정리 |

### Phase 4 sub-phase 진행도

| Sub-phase | 상태 | 산출물 |
|---|---|---|
| 4-1 외곽 추출 | ✅ commit `f637722e` | `session/init.rs` (~1,030 LOC), `session/cli.rs` (Args + parsers), S25/Jetson e2e PASS |
| **4-2 trait 정의 + Builder** | ⏳ **진입 대기** | `session/traits.rs` (6 trait), `session/decode_loop.rs` (DecodeLoop + typestate Builder), `session/defaults.rs` (no-op) |
| 4-3 첫 구현체 + microbench | ⏳ blocked | `ModelForward`, `bin/probe_inference_loop.rs`, TBT 회귀 ≤ 5% gate |
| 4-4 main() 조립자화 | ⏳ blocked | main() ≤ 400 LOC, 모든 mode 동치 |
| 4-5 chat 전면 재작성 | ⏳ blocked | ChatTurnExec 폐기, `session/chat/{repl,turn,stop_condition}.rs`, V-11 해소 |

---

## Phase 4-1 결과 (참고)

| 항목 | 결과 |
|---|---|
| main() LOC | 7,051 → 6,122 (–929) |
| `session/init.rs` | ~1,030 LOC |
| 호스트 검증 | cargo build (3 cfg variants)/test/clippy/layer_lint diff 0 모두 PASS |
| S25 OpenCL Qwen2.5-1.5B Q4_0 32토큰 | **bit-identical** |
| Jetson CUDA Qwen2.5-1.5B Q4_0 32토큰 | **bit-identical** |
| S25 TBT | +4.6% (1회 측정 노이즈 범위) |
| Jetson TBT | −0.6% (노이즈 동등) |

### 변경 사항 (Plan 대비)

1. **Args 이동 옵션 C 적용** — `Args`도 `session/cli.rs`로 이동 (plan 미고려 사전조건). `lib.rs::pub mod session;`로 노출
2. **greedy mutation 회피** — `args.temperature = 0.0` mutation 대신 `let effective_temperature = if args.greedy { 0.0 } else { args.temperature }`
3. **noshuffle 재등록 체크 변경** — `w_dtype == DType::Q4_0` → `model.layers.first().is_some_and(...)` (model owned 이동 영향)

### 미해결 사항 (Phase 4-2 진입 시 자연 해결 예정)

**UX 회귀 (사용자 결정: Phase 4-2 자연 해결 대기)**:
- S25 로그 헤더 `Generating (..., Temp: 0.8, TopP: 0.9, ...)` — sampling 동작은 정상 (greedy=true이면 0 사용), 헤더만 args.temperature 직접 표시
- 해결 방법: Phase 4-2에서 `TokenSampler` trait + `SamplingConfig` 안에 `effective_temperature` 통합 시 헤더도 effective 값 사용

---

## 다음 작업: Phase 4-2 진입 절차

### Step 1: Plan agent로 Phase 4-2 plan 작성

```
arch/inference_pipeline.md §9 Phase 4-2 정의를 기반으로 Plan agent 호출.
- 산출물: session/traits.rs (6 trait), session/decode_loop.rs (DecodeLoop + Builder),
  session/defaults.rs (no-op), trybuild negative test
- 검증: cargo test, layer_lint baseline 31건 동결, typestate compile-fail test PASS
- 핵심 결정 사항:
  1. 6 trait 시그니처 정확화 (Forward의 prefill/step return 타입, EvictionStage::before_step의 ctx 형태 등)
  2. DecodeLoop 필드 lifetime — Box<dyn Trait> vs &'a mut dyn Trait
  3. Builder typestate marker 명명 (NoForward/HasForward vs Phantom 등)
  4. StepCtx 구조체 — 누가 만들고 누가 mutate?
  5. EventSink 통합? Architect 결정 = 분리 + Adapter (commit 9e5c8df8 §6)
- plan 파일 경로 (새로): /home/go/.claude/plans/<phase4-2-slug>.md
```

### Step 2: 사용자 승인 + AskUserQuestion으로 미결 사항 결정

### Step 3: ExitPlanMode 후 Senior Implementer 위임

**에이전트**: senior-implementer (opus) — 새 trait API + 빌더는 복잡 알고리즘급 신중함 필요
**모델**: opus

위임 prompt 요건:
- arch/inference_pipeline.md §2~§6 그대로 따라 trait API + Builder + defaults 구현
- 코드 본체는 `session/` 하위만 (다른 모듈 수정 금지)
- typestate negative test = `engine/tests/spec/test_inv_layer_007_builder_typestate.rs` 또는 trybuild crate 도입 (의존성 추가 필요 시 사용자 사전 결정)
- INV-LAYER-006/007 spec test 추가 (Task #3에서 정의된 INV-LAYER-001~005와 동일 패턴)
- 검증: `cargo test --workspace` + `cargo clippy -- -D warnings` + `layer_lint.py --baseline` diff 0
- 자동 commit + notify

### Step 4: 호스트 검증만 PASS — Phase 4-3 진입 시 디바이스 검증

Phase 4-2는 trait 정의 + no-op default + Builder만. 실제 inference path는 안 건드림. 따라서 디바이스 e2e 검증은 Phase 4-3 (`ModelForward` 첫 구현체) 시점에 수행.

---

## 환경 / 규칙 (불변)

- **언어**: 한국어 (CLAUDE.md 시스템 지시)
- **자동 commit**: 작업 완료 시. 미커밋 작업 금지
- **자동 알림**: `notify-send "llm.rs" "<요약>"`
- **GGUF 우선**: 기본 모델 포맷
- **.cl 커널**: 기본 회피, 성능 최적화 시만 허용 (Senior Implementer)
- **테스트 정책**: 신규 테스트는 `engine/tests/spec/` 하위, inline `#[cfg(test)]` 금지
- **TBT metric**: avg_tbt (tok0 inclusive)
- **Adreno 벤치**: Galaxy S25 = 6T만

## 확정 결정 (Phase 4 전체)

- 6 trait 위치 = **`session/` (L4) 통일**
- Forward lifecycle hook = **default no-op 제공**
- ChatTurnExec = **폐기 (Phase 4-5 재작성)**
- EventSink 통합 = **분리 + Adapter** (`EventSinkAdapterObs`)
- AUF 위치 = `shared/auf/` (Task #2 §13.8-A)
- Layer-aware pool = `backend/<be>/pool.rs` + `WeightStagingPool` trait (§13.8-B)
- backend-specific buffer = `backend/<be>/buffer/` (§13.8-D)
- 테스트 grandfathered exception (§13.8-E)

## 참조 문서

- `arch/inference_pipeline.md` §1~§11 — 6 trait + Builder + StepCtx + Migration + 위험
- `ARCHITECTURE.md` §13 (특히 §13.4.1 session/ tree + §13.7 Step 2 sub-phase)
- `spec/41-invariants.md` §3.26 (INV-LAYER-001~007)
- `arch/01-architecture.md` §6.5 (L4 컴포넌트 + Mermaid)
- `engine/tests/spec/inv_layer_baseline.json` (회귀 baseline 31건)
- `scripts/layer_lint.py` (회귀 감지 도구)
- `/home/go/.claude/plans/velvet-munching-bird.md` (Phase 4-1 plan, 참고용)
- `/tmp/phase4_1_verify/` (S25/Jetson 측정 로그, baseline_*.text + after_*.text 비교 가능)

## Phase 4-2 이후 작업 추정

| Phase | 예상 LOC | 위험 | 위임 대상 |
|---|---:|---|---|
| 4-2 trait + Builder | +500 (신규) | 중 (typestate, lifetime) | Senior Implementer |
| 4-3 ModelForward + microbench | +800 / −400 | 중 (Forward 추출 시 cache lifetime) | Senior Implementer |
| 4-4 main() 조립자 | −5,000 | 높음 (회귀 위험 최대) | Senior Implementer + 디바이스 검증 필수 |
| 4-5 chat 재작성 | +850 / −1,478 | 높음 (multi-turn KV) | Implementer + 디바이스 검증 |
| **Phase 4 합계** | main() ≤ 400 LOC | — | — |
