# Handoff: Phase 4-2 완료 → Phase 4-3 진입 대기

**작성**: 2026-05-17
**HEAD**: `584496b7 test(session): trybuild + INV-LAYER-006/007 spec tests — Phase 4-2 C4`
**다음 세션 진입 문장 (사용자)**: "Phase 4-3 진행" 또는 "ModelForward 구현 시작"

---

## TL;DR

외부 공개 대비 엔진 레이어드 리팩토링 **Task #4 Phase 4-2 (6 trait + DecodeLoopBuilder + no-op defaults) 완료**. 4 commits, +929 LOC 신규, generate.rs 변경 0. 호스트 14 session test + 8 INV-LAYER test PASS, layer_lint baseline diff 0. **다음은 Phase 4-3 — `session/forward/model_forward.rs` (ModelForward 첫 구현체) + `bin/probe_inference_loop.rs` (vtable overhead microbench) + TBT ≤ 5% 회귀 게이트 (S25 OpenCL + Jetson CUDA)**.

---

## 진행 상태

### Task 리스트 (8단계)

| ID | 상태 | 작업 |
|---|---|---|
| #1 | ✅ | ARCHITECTURE.md + spec INV-LAYER-001~005 |
| #2 | ✅ | UNRESOLVED-A~E 5건 결정 |
| #3 | ✅ | spec test + baseline + layer_lint 도구 |
| **#4** | 🔄 in_progress | **L5/L4 분리** (4-1 ✅ / 4-2 ✅ / **4-3 진입 대기** / 4-4/4-5 pending) |
| #5 | ⏳ blocked | L1/L2 경계 정리 |
| #6 | ⏳ blocked | L3 도메인 재배치 |
| #7 | ⏳ blocked | Cross-cutting 분리 |
| #8 | ⏳ blocked | /simplify 코드 정리 |

### Phase 4 sub-phase 진행도

| Sub-phase | 상태 | 산출물 |
|---|---|---|
| 4-1 외곽 추출 | ✅ commit `f637722e` | `session/init.rs` (~1,030 LOC), `session/cli.rs` |
| 4-2 trait + Builder | ✅ commits `85ff756c`~`584496b7` | `session/traits.rs` (123), `session/defaults.rs` (115), `session/decode_loop.rs` (435), trybuild INV-LAYER-006/007 |
| **4-3 첫 구현체 + microbench** | ⏳ **진입 대기** | `session/forward/model_forward.rs`, `bin/probe_inference_loop.rs`, TBT ≤ 5% gate |
| 4-4 main() 조립자화 | ⏳ blocked | main() ≤ 400 LOC, 모든 mode 동치 |
| 4-5 chat 전면 재작성 | ⏳ blocked | ChatTurnExec 폐기, `session/chat/{repl,turn,stop_condition}.rs`, V-11 해소 |

---

## Phase 4-2 결과 (참고)

| 항목 | 결과 |
|---|---|
| Commits | C1 `85ff756c` / C2 `79efe21f` / C3 `ee1a1ae0` / C4 `584496b7` |
| 신규 LOC | +929 (traits 123 + defaults 115 + decode_loop 435 + tests 248 + Cargo/spec mod 8) |
| main() LOC | 6,122 → 6,122 (변경 없음, 의도) |
| cargo build (default + no-default-features) | PASS |
| cargo test --lib session:: | **14 PASS** (cli 6 + defaults 4 + decode_loop 4) |
| cargo test --test spec inv_layer | **8 PASS** (LAYER-001~007 + pos/neg) |
| trybuild compile_fail/forward_missing.rs | PASS (typestate E0599) |
| trybuild compile_pass/forward_minimal.rs | PASS (prefill+step minimal Forward) |
| layer_lint --baseline | violations 0건 |
| check_spec_coverage.sh INV-LAYER-006/007 등재 | 확인 (missing list에 없음) |
| 디바이스 검증 | 불필요 (코드 path 미수정, Phase 4-3로 미룸) |

### Phase 4-2 핵심 결정 반영 (D1~D8)

- **D1 trybuild**: `engine/Cargo.toml` dev-dep `trybuild = "1.0"` 추가, stderr snapshot `compile_fail/*.stderr` git 추적
- **D2 Send+Sync bound**: 없음 (arch §2 그대로). 6 trait 모두 `&mut self` 위주, single-thread 가정
- **D3 Box<dyn>**: DecodeLoop 6 필드 모두 `Box<dyn Trait>` / `Vec<Box<dyn>>` / `Arc<AtomicBool>`. INV-LAYER-006 source-grep PASS
- **D4 run() 본체**: arch §3 (a)~(g) 단계 그대로. NoOp default들로도 정상 종료 (test 검증)
- **D6 GreedySampler 자동**: `with_sampler()` 생략 시 build() 자동 주입
- **D7 stop_flag**: `Arc<AtomicBool>` (StepCtx는 `&AtomicBool` borrow)
- **D8 lifecycle default**: `Forward::finalize`/`on_kv_prune` default no-op → compile_pass minimal Forward (prefill+step만) 컴파일 통과

### 발견된 부수 이슈 (Phase 4-2 외)

- `cargo clippy --workspace -- -D warnings` 기존 27 warnings + 1 error. Phase 4-1 산출물 `session/cli.rs` doc indentation 위주 + 다른 파일 unsafe 누락 1건. **본 phase 신규 파일은 clippy clean**. 정리는 Task #8 (/simplify) 또는 별도 cleanup phase.
- `check_spec_coverage.sh` exit 1 — 45건 기존 INV 누락 (INV-047, INV-093~119 등). 본 phase 도입 누락 없음.

---

## 다음 작업: Phase 4-3 진입 절차

### 산출물 (예정)

| 파일 | 내용 |
|---|---|
| `engine/src/session/forward/mod.rs` (신규) | `pub mod model_forward;` 진입점 |
| `engine/src/session/forward/model_forward.rs` (신규) | `ModelForward` 구현체 — 기존 `model.forward()` / `forward_into()` 래핑. Backend / LlamaModel / KVCache / Workspace / gpu_buffers / logits owned. INV-LAYER-006 적용 외 (trait impl struct 내부 필드는 concrete 허용). |
| `engine/src/bin/probe_inference_loop.rs` (신규) | Microbench — `DecodeLoop` + `ModelForward` 통합 path vs `model.forward_into()` 직접 호출의 vtable overhead 측정. TBT ≤ 5% 회귀 게이트. |
| `engine/src/bin/generate.rs` (편집) | 기본 generate path (chat/kivi/offload 외)에서 ModelForward + DecodeLoop 사용. main() 일부 변경. |

**예상 LOC**: +800 / −400 (generate.rs 일반 path 일부 추출).

### Step 1: Plan agent로 Phase 4-3 plan 작성

```
arch/inference_pipeline.md §9 Phase 4-3 정의 + §8.1 ModelForward 카탈로그 기반 Plan agent 호출.
- ModelForward의 owned 자원 형태 — Phase 4-1 SessionInitCtx를 참고하여 KV cache lifetime/생성 시점 결정
- probe_inference_loop.rs 측정 시나리오 — Qwen2.5-1.5b Q4_0 32 토큰 greedy seed
- generate.rs 부분 교체 범위 — chat/kivi/offload 분기점은 건드리지 않고 기본 path만
- TBT 회귀 ≤ 5% 게이트 측정 방법 (S25 OpenCL + Jetson CUDA)
```

### Step 2: AskUserQuestion으로 미결 사항 결정

핵심 후보:
1. **ModelForward의 KV cache 소유 vs builder 주입** — Phase 4-1 ctx에서 KV cache를 만들어 ModelForward로 이동? 아니면 builder에서 KV cache 따로 주입?
2. **generate.rs 부분 교체 범위** — 기본 path만 / 기본 + lm_head verify / 모든 분기?
3. **probe_inference_loop 측정 모델** — Qwen2.5-1.5b Q4_0 / Llama 3.2 1B / 양쪽?

### Step 3: ExitPlanMode 후 Senior Implementer 위임

**에이전트**: senior-implementer (opus) — concrete Forward 구현 + microbench + 디바이스 검증 게이트 필요
**모델**: opus

위임 prompt 요건:
- arch §2.2 Forward trait 시그니처 매핑 (prefill/step)
- 기존 `model.forward()` 호출부 정확 매핑 (logits 리턴, KV 자동 갱신)
- Cache lifetime — `LlamaModelForwardArgs`에서 cache_manager 제거된 상태 (2026-03-10 리팩토링) 활용
- microbench: S25 6T + Jetson CUDA, avg_tbt (tok0 inclusive) 사용
- 자동 commit + notify

### Step 4: 디바이스 검증 PASS 시 commit (게이트)

- S25 OpenCL Qwen2.5-1.5b Q4_0 32 토큰 **bit-identical**
- Jetson CUDA 동일 측정 **bit-identical**
- avg_tbt Δ ≤ 5% (vtable overhead 회귀 게이트)

회귀 ≥ 5% 시 arch §7 escape hatch — `DecodeLoop<F: Forward, T: TokenSampler>` 부분 generic화 검토.

---

## 환경 / 규칙 (불변)

- **언어**: 한국어 (CLAUDE.md 시스템 지시)
- **자동 commit**: 작업 완료 시. 미커밋 작업 금지
- **자동 알림**: `notify-send "llm.rs" "<요약>"`
- **GGUF 우선**: 기본 모델 포맷
- **.cl 커널**: 기본 회피, 성능 최적화 시만 허용 (Senior Implementer)
- **테스트 정책**: 신규 테스트는 `engine/tests/spec/` 하위, inline `#[cfg(test)]` 금지 (단 trait 사용성 검증용 내부 mod는 예외 — Phase 4-2에서 decode_loop.rs/defaults.rs에 적용)
- **TBT metric**: avg_tbt (tok0 inclusive)
- **Adreno 벤치**: Galaxy S25 = 6T만
- **clippy**: 본 phase 신규 코드는 clean. 기존 warnings 정리는 별도 phase (Task #8)

## 확정 결정 (Phase 4 전체)

- 6 trait 위치 = **`session/` (L4) 통일**
- Forward lifecycle hook = **default no-op 제공**
- ChatTurnExec = **폐기 (Phase 4-5 재작성)**
- EventSink 통합 = **분리 + Adapter** (`EventSinkAdapterObs`)
- AUF 위치 = `shared/auf/` (Task #2 §13.8-A)
- Layer-aware pool = `backend/<be>/pool.rs` + `WeightStagingPool` trait (§13.8-B)
- backend-specific buffer = `backend/<be>/buffer/` (§13.8-D)
- 테스트 grandfathered exception (§13.8-E)

## Phase 4-2 추가 확정 결정 (D1~D8)

위 "Phase 4-2 결과" 섹션 참조 — trybuild / Box<dyn> / no Send+Sync / GreedySampler default / lifecycle default no-op.

## 참조 문서

- `arch/inference_pipeline.md` §2~§11 — trait API + Builder + Migration + 위험
- `arch/inference_pipeline.md` §8.1 — Forward 구현체 카탈로그 (ModelForward / KiviForward / OffloadForward)
- `ARCHITECTURE.md` §13 (§13.4.1 session/ tree + §13.7 Step 2 sub-phase)
- `spec/41-invariants.md` §3.26 (INV-LAYER-001~007)
- `engine/src/session/{traits,defaults,decode_loop}.rs` — Phase 4-2 신규 모듈
- `engine/src/session/{init,cli}.rs` — Phase 4-1 신규 모듈
- `engine/tests/spec/inv_layer_baseline.json` (회귀 baseline)
- `engine/tests/spec/compile_fail/forward_missing.stderr` — typestate 검증 snapshot (rustc upgrade 시 갱신)
- `scripts/layer_lint.py` (회귀 감지)
- `/home/go/.claude/plans/velvet-munching-bird.md` (Phase 4-2 plan, 참고용)

## Phase 4-3 이후 작업 추정

| Phase | 예상 LOC | 위험 | 위임 대상 |
|---|---:|---|---|
| 4-3 ModelForward + microbench | +800 / −400 | 중 (Forward 추출 시 cache lifetime + vtable 회귀) | Senior Implementer + 디바이스 검증 |
| 4-4 main() 조립자 | −5,000 | 높음 (회귀 위험 최대) | Senior Implementer + 디바이스 검증 필수 |
| 4-5 chat 재작성 | +850 / −1,478 | 높음 (multi-turn KV) | Implementer + 디바이스 검증 |
| **Phase 4 합계** | main() ≤ 400 LOC | — | — |
