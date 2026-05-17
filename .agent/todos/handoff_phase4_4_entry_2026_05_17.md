# Handoff: Phase 4-4 종료 → Phase 4-4.5 (paradigm 통일) + 4-5 (chat 재작성) 진입

**작성**: 2026-05-17
**갱신**: 2026-05-17 (Phase 4-4 a/b/d 종료, paradigm mismatch는 4-4.5로 분리)
**HEAD**: `6953b200 docs(experiment): Phase 4-4-b S25 paradigm mismatch split to 4-4.5`
**다음 세션 진입 문장 (사용자)**: "Phase 4-4.5 진행" 또는 "DecodeLoop paradigm 통일"

---

## TL;DR

**Phase 4-4 a/b/d 종료** (4 commits: `886f0404` + `26e1ca87` + `e83b87d2` + `6953b200`). `session/assembly/build_standard_loop` 헬퍼 신설 + `bin/generate.rs` line 3032에 narrow happy path 분기 추가 (DecodeLoop+ModelForward 위임). 호스트 CPU Qwen 1.5B Q4_0 정상 디코딩 PASS, S25 OpenCL 분기 진입 + 32 토큰 생성 정상 동작 확인.

**중대 발견**: S25 OpenCL 비교에서 token sequence baseline 불일치 → 원인 = `DecodeLoop::prefill(tokens) -> Result<()>` paradigm mismatch (Phase 4-3 인지된 issue가 production-level에서 발현). `prev_token = tokens.last()` 설정으로 첫 step에서 prompt 마지막 token이 **2회 forward**. **사용자 결정 (Q5-B)**: Phase 4-4.5 sprint로 paradigm 통일 분리, 4-4-b의 G6 bit-identical 게이트 완화.

**다음**: **Phase 4-4.5 — DecodeLoop paradigm 통일** (`prefill -> Result<Vec<f32>>` + `run(budget, first_token)`) + ModelForward chunked prefill 지원 + optional collector wiring. 그 후 G8 디바이스 재측정 (S25 OpenCL bit-identical + Δ ≤ 5%).

**main() ≤ 400 LOC 게이트 보류**: 실측 main() 본체 = line 77~6257 = **6,180 LOC**. eval-ll / ppl / batch 분기들이 inline body로 남아 있어 Phase 4-5 분기 추출 + Phase 4-4.5 happy path 확장 완료 후 달성 가능.

---

## 진행 상태

### Task 리스트 (8단계 + Phase 4-3 sub-tasks)

| ID | 상태 | 작업 |
|---|---|---|
| #1 | ✅ | ARCHITECTURE.md + spec INV-LAYER-001~005 |
| #2 | ✅ | UNRESOLVED-A~E 5건 결정 |
| #3 | ✅ | spec test + baseline + layer_lint 도구 |
| **#4** | 🔄 in_progress | **L5/L4 분리** (4-1 ✅ / 4-2 ✅ / 4-3 ✅ 호스트+S25 / 4-4 ✅ a/b/d / **4-4.5 ⏳ paradigm 통일** / 4-5 pending) |
| #5 | ⏳ blocked | L1/L2 경계 정리 |
| #6 | ⏳ blocked | L3 도메인 재배치 |
| #7 | ⏳ blocked | Cross-cutting 분리 |
| #8 | ⏳ blocked | /simplify 코드 정리 |

### Phase 4 sub-phase 진행도

| Sub-phase | 상태 | 산출물 / 결과 |
|---|---|---|
| 4-1 외곽 추출 | ✅ commit `f637722e` | `session/init.rs` (~1,030 LOC), `session/cli.rs` |
| 4-2 trait + Builder | ✅ commits `85ff756c`~`584496b7` | `session/{traits, defaults, decode_loop}.rs`, trybuild INV-LAYER-006/007 |
| **4-3 ModelForward + microbench** | ✅ 호스트 + S25 / ⏳ Jetson 보충 | C1 `3470ad1d` + C2 `f5236073` + C3 `c63190d1` + handoff `7619ae9d` + S25 `debf4e1f`. **호스트 CPU Δ=1.53%, S25 OpenCL Δ=2.29%, 둘 다 bit-identical**. |
| **4-4 main() 조립자화** | ✅ a/b/d (c skip) | `886f0404` 헬퍼 + `26e1ca87` 시그니처 + `e83b87d2` happy path 분기 + `6953b200` paradigm split doc. main() ≤ 400 LOC 게이트 보류 (실측 6,180 LOC) |
| **4-4.5 paradigm 통일** | ⏳ **ready (진입 가능)** | `DecodeLoop::prefill -> Result<Vec<f32>>` + `run(budget, first_token)` + ModelForward chunked + collector wiring |
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

## 다음 작업: Phase 4-4 main() 조립자화

### 목표

`bin/generate.rs::main()` 6,122 LOC (Phase 4-1 후) → **≤ 400 LOC**. 4개 `build_*_loop` 헬퍼로 분기 (standard / kivi / offload / chat). 본 phase의 직접 산출 = standard path만 `DecodeLoop + ModelForward`로 교체. KIVI/Offload/chat은 4-5-a~e에서 별도 처리. arch `inference_pipeline.md` §5 + §9 Phase 4-4 (= Migration Step 2-4) 명세 기반.

### Sub-step 분해 (4 commits 권장)

| Step | 내용 | 산출 LOC | 위험 |
|---|---|---:|---|
| **4-4-a** | `session/build/mod.rs` + `build_standard_loop.rs` 신설 — `SessionInitCtx`에서 standard `DecodeLoop` 조립 헬퍼. generate.rs 미수정. unit test 1건 | +200 / 0 | 낮음 |
| **4-4-b** | generate.rs main()의 standard generate path (대략 line 1900~3500 decode loop) → `build_standard_loop()` 호출로 부분 교체. chat/kivi/offload/eval-ll/ppl/qcf 등 다른 분기는 그대로 유지 | +50 / −1,500 | **높음** (회귀) |
| 4-4-c | main() 잔여 분기들을 `build_*_loop` 패턴으로 통일하되 본체 추출은 4-5에서. chat/kivi/offload는 stub로 기존 함수 호출만 | +30 / −200 | 중 |
| 4-4-d | main() 자체 압축 + cleanup. ≤ 400 LOC 게이트 검증 | 0 / −500 | 낮음 |

권장: **4-4-a + 4-4-b를 한 sprint**로 묶어 진입. 회귀 검증 통과 후 4-4-c/d 진행.

### Phase 4-4 진입 결정 후보 (AskUserQuestion 시 노출)

| ID | 항목 | 옵션 |
|---|---|---|
| **Q1** | 4-4 PR 분할 전략 | (A) 4-4-a부터 sub-step별 PR / (B) 4-4-a+b 묶어 1 PR / (C) 4-4 전체를 1 PR (위험 최대) — 권장 **(A)** |
| **Q2** | 첫 교체 분기 | (A) standard generate (가장 일반) / (B) eval-ll (분기 격리 양호) / (C) ppl — 권장 **(A) standard generate** |
| **Q3** | 디바이스 검증 시점 | (A) 4-4-b 직후 (standard path 교체 직후 회귀 확인) / (B) 4-4-d 종료 후 일괄 — 권장 **(A) 4-4-b 직후** |
| **Q4** | DecodeLoop::prefill 시그니처 정리 (paradigm 통일) | (A) 4-4 안에 포함 (arch §3.1과 일치, `prefill -> Vec<f32>` + `run(budget, first_token)`) / (B) 4-5에 미룸 — 권장 **(A) 포함** (chat 4-5에서 활용) |
| **Q5** | 회귀 ≥ 5% 시 대응 | (A) arch §7.3 escape hatch (`DecodeLoop<F, T>`) PoC / (B) ModelForward inline 최적화 시도 / (C) 양쪽 모두 — Phase 4-3 측정 (호스트 1.53%, S25 2.29%) 기준 회귀 위험 낮음 |

### Plan agent 진입 prompt 템플릿 (다음 세션 그대로 사용 가능)

```
arch/inference_pipeline.md §5 `main()` 변환 예시 + §9 Phase 4-4 정의 + ARCHITECTURE.md §13.7 Step 2-4 기반으로 Phase 4-4 (main() 조립자화) plan 작성.

핵심 제약:
- Phase 4-3 호스트 (Δ=1.53%) + S25 OpenCL (Δ=2.29%, 32 토큰 bit-identical) PASS 확인됨 → arch §7.3 escape hatch 불필요
- Phase 4-4-a + 4-4-b를 첫 sprint로 묶어 진입 권장 (sub-step 표 참조)
- standard generate path만 교체. chat/kivi/offload/eval-ll/ppl/qcf는 4-5
- 디바이스 검증: 4-4-b 직후 S25 OpenCL Qwen 2.5 1.5B Q4_0 gen=32 runs=5 bit-identical + Δ≤5%
- generate.rs 시작 LOC: 6,122 (Phase 4-1 후). 목표 ≤ 400 LOC (-5,700)
- main() 안의 standard generate decode loop 진입 line 대략 1900~3500 (sub-step 4-4-b 교체 범위)
- session/build/{mod, build_standard_loop}.rs 신설 — DecodeLoopBuilder 조립 헬퍼

산출물 표 + sub-step 분해 + 검증 게이트 + 위험 표를 plan에 포함.
```

### 4-4-a 구체 시그니처 안

```rust
// engine/src/session/build/mod.rs
pub mod build_standard_loop;
pub use build_standard_loop::build_standard_loop;

// engine/src/session/build/build_standard_loop.rs
use crate::session::{DecodeLoop, DecodeLoopBuilder, GreedySampler};
use crate::session::forward::{ModelForward, alloc_standard_kv_caches};
use crate::session::init::SessionInitCtx;
use std::sync::Arc;

/// Phase 4-4-a: standard generate path용 DecodeLoop 조립.
/// chat/kivi/offload는 별도 build_* 헬퍼 (Phase 4-5).
pub fn build_standard_loop(ctx: &mut SessionInitCtx, max_seq_len: usize)
    -> anyhow::Result<DecodeLoop> {
    let kv = alloc_standard_kv_caches(
        &ctx.model, ctx.backend.clone(), ctx.memory.clone(),
        max_seq_len.min(...), max_seq_len, kv_dtype_from_args(...))?;
    let mf = ModelForward::new(
        ctx.backend.clone(), ctx.memory.clone(), ctx.cpu_backend_arc.clone(),
        Arc::new(/* ctx.model 이전 작업 */), kv, max_seq_len,
    )?;
    Ok(DecodeLoopBuilder::new()
        .with_forward(mf)
        .with_sampler(GreedySampler) // TODO: ctx.sampling_config로 builder
        .build())
}
```

**주의**: `SessionInitCtx::model`은 `TransformerModel` owned이지만 `ModelForward`는 `Arc<TransformerModel>` 필요. 4-4-a에서 `ctx.model`을 `Arc`로 wrap하는 패턴 결정 — `ctx.model_arc` 필드 추가 또는 `Arc::new(std::mem::take(&mut ctx.model))` 1회 변환.

### 핵심 코드 위치 (generate.rs 참조)

```
engine/src/bin/generate.rs (6,122 LOC, Phase 4-1 후):
  line 1     use 선언 (TransformerModel, KVCache, LayerWorkspace, ...)
  line 100~  fn main() 시작
  line 100~600   args/init/kv alloc/chat 분기 (Phase 4-1에서 일부 SessionInitCtx로 추출됨)
  line 1900~3500 standard generate prefill+decode loop (← 4-4-b 교체 대상)
  line 2245      let mut gen_ws = LayerWorkspace::new(...)
  line 2847      model.forward_into(...)
  line 2944~2963 logits read + sample
  line 3700~5000 eval-ll/ppl/qcf 분기 (4-4 범위 밖)
  line 7000~     chat REPL (4-5 범위)
```

### 검증 게이트 (Phase 4-4 종료 시)

| Gate | 항목 | 기준 |
|---|---|---|
| G1 | cargo build (default + no-default-features) | PASS |
| G2 | cargo fmt + clippy 신규 코드 | 0 warning |
| G3 | cargo test --workspace | PASS |
| G4 | INV-LAYER-006 source-grep | DecodeLoop 필드 변경 0건 |
| G5 | **generate.rs main() LOC** | ≤ 400 |
| G6 | **S25 OpenCL standard generate 32-token bit-identical** | path_pre_4_4 == path_post_4_4 |
| G7 | **avg_tbt Δ ≤ 5%** (S25 OpenCL Qwen 2.5 1.5B Q4_0) | PASS |
| G8 | integration tests (기존 generate flag 조합) | 모두 PASS |

---

## 부수: Jetson CUDA 보충 측정 (병행 가능, blocking 아님)

호스트 + S25 결과로 vtable 게이트 통과 입증되었지만, Jetson CUDA 측정 보충은
production-readiness 강화 목적. SSH alias 미설정 → 보드 내 빌드 + 측정:
```bash
# Jetson 보드 내
cargo build --release --bin probe_inference_loop --features cuda-embedded
./target/release/probe_inference_loop --backend cuda \
    --model-path qwen2.5-1.5b-q4_0.gguf --tokenizer-path tokenizer.json \
    --gen 32 --runs 5 --max-seq-len 512
```
결과는 `papers/eurosys2027/_workspace/experiment/probe_inference_loop_phase4_3_2026_05_17.md`의 "Jetson CUDA (pending)" 섹션에 추가.

---

## 다음 세션 진입 절차 (5단계)

1. **컨텍스트 확인** — `git log --oneline -10` 확인. HEAD `debf4e1f`인지 verify.
2. **사용자 진입 문장** — "Phase 4-4 진행" 또는 "main() 조립자화 시작" 입력.
3. **Plan agent 호출** — 위의 "Plan agent 진입 prompt 템플릿" 그대로 사용. Plan agent가 산출물 표 + sub-step 분해 + 검증 게이트 + 위험 표 포함된 plan 작성.
4. **AskUserQuestion** — Q1~Q5 (PR 분할/첫 분기/디바이스 검증 시점/prefill 시그니처 정리/회귀 대응) 결정. 권장값 그대로 통과 가능.
5. **ExitPlanMode → 구현** —
   - 4-4-a (헬퍼 신설, 회귀 없음) → 메인 세션 직접 구현 가능
   - 4-4-b (generate.rs 1,500 LOC 교체) → **Senior Implementer (opus) 위임** + 디바이스 검증
   - 4-4-c/d → Implementer (sonnet) 가능

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
- **❌ `git worktree` 사용 금지**: baseline 비교 빌드용으로 `git worktree add /tmp/...`를 시도하면 **권한 문제 발생**. 실측 발견 (Phase 4-4-b S25 측정 중, 2026-05-17): (1) Cargo workspace target dir이 main repo와 공유되어 worktree 빌드가 main repo binary를 덮어씀, (2) `run_device.py` 가 "Shell cwd was reset to /home/go/Workspace/llm_rs2"로 main repo cwd를 강제, (3) `hosts.toml` / `android.source` NDK 환경이 worktree에 없거나 macOS path를 가리켜 build fail. **올바른 baseline 비교 방법** = `git switch <commit>` detached HEAD checkout + `python scripts/run_device.py -d galaxy_s25 --skip-exec generate` → adb shell `cp /data/local/tmp/generate /data/local/tmp/generate_baseline` → `git switch master` 후 rebuild + push. baseline binary는 디바이스 측에 보존.

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
