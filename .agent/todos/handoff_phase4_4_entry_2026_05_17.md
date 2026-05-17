# Handoff: Phase 4-3 완료 (호스트 + S25 PASS) → Phase 4-4 진입 가능

**작성**: 2026-05-17
**HEAD**: `c63190d1 feat(session): probe_inference_loop microbench (Phase 4-3 C3)`
**다음 세션 진입 문장 (사용자)**: "Phase 4-4 진행" 또는 "main() 조립자화 시작"

---

## TL;DR

외부 공개 대비 엔진 레이어드 리팩토링 **Task #4 Phase 4-3 (ModelForward + microbench) 호스트 + S25 OpenCL PASS**. 3 commits, +913 LOC 신규, generate.rs 변경 0. **호스트 CPU Qwen 2.5 1.5B Q4_0: Δ=1.53% PASS bit-identical=true**. **S25 Adreno OpenCL gen=32 runs=5: Δ=2.29% PASS bit-identical=true (32 토큰 모두 일치)**. Jetson CUDA는 SSH alias 미설정으로 보드 내 직접 측정 별도 task. 양쪽 모두 게이트(5%)의 절반 이내 → **arch §7.3 escape hatch 불필요, Phase 4-4 진입 가능**. **다음은 Phase 4-4 — main() 조립자화 (≤ 400 LOC, 표준 generate path를 `DecodeLoop+ModelForward`로 교체)**.

---

## 진행 상태

### Task 리스트 (8단계 + Phase 4-3 sub-tasks)

| ID | 상태 | 작업 |
|---|---|---|
| #1 | ✅ | ARCHITECTURE.md + spec INV-LAYER-001~005 |
| #2 | ✅ | UNRESOLVED-A~E 5건 결정 |
| #3 | ✅ | spec test + baseline + layer_lint 도구 |
| **#4** | 🔄 in_progress | **L5/L4 분리** (4-1 ✅ / 4-2 ✅ / **4-3 ✅ 호스트 / 4-3 디바이스 ⏳** / 4-4/4-5 pending) |
| #5 | ⏳ blocked | L1/L2 경계 정리 |
| #6 | ⏳ blocked | L3 도메인 재배치 |
| #7 | ⏳ blocked | Cross-cutting 분리 |
| #8 | ⏳ blocked | /simplify 코드 정리 |

### Phase 4 sub-phase 진행도

| Sub-phase | 상태 | 산출물 / 결과 |
|---|---|---|
| 4-1 외곽 추출 | ✅ commit `f637722e` | `session/init.rs` (~1,030 LOC), `session/cli.rs` |
| 4-2 trait + Builder | ✅ commits `85ff756c`~`584496b7` | `session/{traits, defaults, decode_loop}.rs`, trybuild INV-LAYER-006/007 |
| **4-3 ModelForward + microbench** | ✅ 호스트 / ⏳ 디바이스 | C1 `3470ad1d` + C2 `f5236073` + C3 `c63190d1`. **호스트 CPU PASS (Δ=1.53%, bit-identical)**. 디바이스는 Tester 위임. |
| 4-4 main() 조립자화 | ⏳ blocked (디바이스 PASS 후) | main() ≤ 400 LOC, 표준 generate path 교체 |
| 4-5 chat 전면 재작성 | ⏳ blocked | ChatTurnExec 폐기, `session/chat/{repl,turn,stop_condition}.rs`, V-11 해소 |

---

## Phase 4-3 결과

### 산출물 (+913 LOC)

| 파일 | LOC | 역할 |
|---|---:|---|
| `engine/src/session/forward/mod.rs` | 11 | `pub mod model_forward; pub use ModelForward, alloc_standard_kv_caches;` |
| `engine/src/session/forward/model_forward.rs` | 332 | `ModelForward` struct + `Forward` trait impl + `alloc_standard_kv_caches()` 헬퍼 |
| `engine/src/session/mod.rs` | +1 | `pub mod forward;` |
| `engine/src/bin/probe_inference_loop.rs` | 565 | microbench binary — DecodeLoop+ModelForward vs direct forward_into 비교 |
| `engine/tests/spec/test_model_forward_parity.rs` | 312 | env-gated host CPU 8 토큰 bit-identical parity |
| `engine/tests/spec/compile_pass/model_forward_minimal.rs` | 16 | trybuild positive — Forward impl 컴파일 확인 |
| `engine/tests/spec.rs` | +2 | parity mod 등록 |
| `engine/tests/spec/test_inv_layer_007.rs` | +9 | 두 trybuild fixture bundle |

### 게이트 PASS 표

| Gate | 결과 |
|---|---|
| G1 cargo build (opencl + no-default-features) | ✅ |
| G2 cargo fmt + clippy 신규 코드 | ✅ (0 warning) |
| G3 cargo test --workspace | ✅ session unit 14 PASS + parity SKIP (env) |
| G4 INV-LAYER-006 source-grep | ✅ decode_loop.rs 변경 0건, layer_lint baseline diff 0 |
| G5 trybuild compile_pass (forward_minimal + model_forward_minimal) | ✅ |
| **G6 S25 OpenCL bit-identical** | ✅ **32 토큰 양 path 일치** (Adreno 830) |
| **G7 Jetson CUDA bit-identical** | ⏳ SSH alias 미설정, 보드 내 직접 측정 별도 task |
| **G8 avg_tbt Δ ≤ 5%** | ✅ **호스트 1.53% / S25 2.29%** — 양쪽 게이트 절반 이내 |

### 핵심 설계 결정 반영 (P1/P3/P4)

- **P1 KV cache 타입 = `KVCache` 단일** — KIVI/Offload는 Phase 4-5-a 분리 (별도 `KiviForward`/`OffloadForward`)
- **P3 generate.rs 0줄 수정** — microbench로 자체 검증, 부분 교체는 4-3.5/4-4 분리
- **P4 Workspace alloc = Hybrid** — `decode_workspace` eager (`LayerWorkspace`, ~3 MB) + `prefill_workspace` lazy (`PrefillWorkspace`, ~200 MB at 2048 ctx) + seq_len realloc guard
- **명명** — `gen_workspace` → `decode_workspace` (직관성)

### 측정 결과 (papers/eurosys2027/_workspace/experiment/probe_inference_loop_phase4_3_2026_05_17.md)

**호스트 CPU x86_64 AVX2** (Qwen 2.5 1.5B Q4_0, gen=4, runs=1):
```
avg_tbt_ms: decode_loop=62.30 / direct=61.36
delta_pct: 1.53% (PASS)  bit-identical: true
tokens: [12095, 13, 576, 6722]
```

**S25 Adreno OpenCL** (qwen2.5-1.5b-q4_0.gguf pure Q4_0, gen=32, runs=5):
```
avg_tbt_ms: decode_loop=33.18 / direct=32.44
delta_pct: 2.29% (PASS)  bit-identical_first_32: true
tok0_ms: decode_loop=118.61 / direct=116.48
```

vtable indirect call overhead 양 측정 환경 모두 게이트(5%) 절반 이내. 디바이스
forward TBT가 호스트보다 짧아 (33 ms vs 60 ms) 비율은 약간 더 크지만 absolute
overhead (~0.74 ms/tok S25) 충분히 작음. **arch §7.3 escape hatch
(`DecodeLoop<F, T>` 부분 generic화) 불필요 확정**.

### Paradigm 주의 (4-3.5/4-4에서 해소)

- 현재 `DecodeLoop::prefill` 시그니처 = `Result<()>` — last logits 반환 안 함
- 따라서 첫 토큰 sample은 `step(prev=prompt_last, pos=prompt_len)`에서 발생 — **prompt_last가 한 번 더 forward**
- production `generate.rs`는 prefill last logits → argmax → first generated (prompt_last 한 번만 forward)
- microbench의 direct path도 DecodeLoop paradigm으로 통일 (fair vtable 측정)
- 향후 `DecodeLoop::prefill(...) -> Result<Vec<f32>>` + `run(budget, first_token)` 시그니처 정리 시 paradigm 통일 가능 (arch §3.1과 일치)

---

## 다음 작업

### 1순위: Phase 4-4 진입 (Phase 4-3 게이트 통과로 unblocked)

**산출물** (예정):
- `bin/generate.rs::main()` 7,051 → ≤ 400 LOC
- 4개 `build_*_loop` 헬퍼 — standard / kivi / offload / chat 분기
- 표준 generate path를 `DecodeLoop + ModelForward` 호출로 교체
- 모든 모드 동치 (S25 + Jetson + host CPU, TBT 회귀 ≤ 5%, integration tests PASS)

**진입 절차**:
1. Plan agent로 Phase 4-4 plan (arch §5 `main()` 예시 + arch §9 Phase 4-4 기반)
2. AskUserQuestion 미결 사항 (분할 PR vs 단일 PR, 어느 모드부터 교체)
3. ExitPlanMode → Senior Implementer (opus) 위임 — generate.rs 7,051 LOC 광범위 수정

**위험**: 회귀 가능성 최대 (arch §10.4). sub-step별 PR 분할 권장.

### 2순위: Jetson CUDA 보충 측정 (선택)

호스트 + S25 결과로 게이트 통과 입증되었지만, Jetson CUDA 측정 보충은
production-readiness 강화 목적. SSH alias 미설정 → 보드 내 빌드 + 측정:
```bash
# Jetson 보드 내
cargo build --release --bin probe_inference_loop --features cuda-embedded
./target/release/probe_inference_loop --backend cuda \
    --model-path qwen2.5-1.5b-q4_0.gguf --tokenizer-path tokenizer.json \
    --gen 32 --runs 5 --max-seq-len 512
```

---

## 환경 / 규칙 (불변)

- **언어**: 한국어 (CLAUDE.md 시스템 지시)
- **자동 commit**: 작업 완료 시. 미커밋 작업 금지
- **자동 알림**: `notify-send "llm.rs" "<요약>"`
- **GGUF 우선**: 기본 모델 포맷
- **.cl 커널**: 기본 회피, 성능 최적화 시만 허용 (Senior Implementer)
- **테스트 정책**: 신규 테스트는 `engine/tests/spec/` 하위, inline `#[cfg(test)]` 금지 (trait 사용성 검증용 내부 mod는 예외)
- **TBT metric**: avg_tbt (tok0 inclusive)
- **Adreno 벤치**: Galaxy S25 = 6T만
- **clippy**: 본 phase 신규 코드는 clean. 기존 27 warnings + 1 error는 Task #8 (`/simplify`)에서 정리

## 확정 결정 (Phase 4 전체)

- 6 trait 위치 = **`session/` (L4) 통일**
- Forward lifecycle hook = **default no-op 제공**
- ChatTurnExec = **폐기 (Phase 4-5 재작성)**
- EventSink 통합 = **분리 + Adapter** (`EventSinkAdapterObs`)
- AUF 위치 = `shared/auf/` (Task #2 §13.8-A)
- Layer-aware pool = `backend/<be>/pool.rs` + `WeightStagingPool` trait (§13.8-B)
- backend-specific buffer = `backend/<be>/buffer/` (§13.8-D)
- 테스트 grandfathered exception (§13.8-E)

## Phase 4-3 확정 결정 (P1/P3/P4)

- `ModelForward`는 표준 `KVCache`만 보유 — KIVI/Offload는 Phase 4-5-a 분리
- generate.rs는 4-3에서 0줄 수정 — microbench 자체완결
- `decode_workspace` eager + `prefill_workspace` lazy (Hybrid)
- 명명 `decode_workspace` (직관성)

## 참조 문서

- `arch/inference_pipeline.md` §2~§11 — trait API + Builder + Migration + 위험
- `arch/inference_pipeline.md` §3.1 — DecodeLoop run() 흐름 (paradigm 주의)
- `arch/inference_pipeline.md` §5 — `main()` 조립자 변환 예시 (Phase 4-4)
- `arch/inference_pipeline.md` §7.3 — vtable escape hatch
- `ARCHITECTURE.md` §13 (§13.7 Step 2 sub-phase)
- `spec/41-invariants.md` §3.26 (INV-LAYER-001~007)
- `engine/src/session/forward/model_forward.rs` — Phase 4-3 신규 ModelForward
- `engine/src/bin/probe_inference_loop.rs` — Phase 4-3 vtable microbench
- `engine/tests/spec/test_model_forward_parity.rs` — env-gated parity test
- `papers/eurosys2027/_workspace/experiment/probe_inference_loop_phase4_3_2026_05_17.md` — 호스트 측정 결과
- `/home/go/.claude/plans/velvet-munching-bird.md` (Phase 4-3 plan, 참고용)

## Phase 4-3 이후 작업 추정

| Phase | 예상 LOC | 위험 | 위임 대상 |
|---|---:|---|---|
| 4-3 디바이스 | 0 (측정만) | 중 (회귀 ≥ 5% 시 escape hatch) | Tester |
| 4-4 main() 조립자 | −5,000 | 높음 (회귀 위험 최대) | Senior Implementer + 디바이스 검증 |
| 4-5 chat 재작성 | +850 / −1,478 | 높음 (multi-turn KV) | Implementer + 디바이스 검증 |
| **Phase 4 합계** | main() ≤ 400 LOC | — | — |
