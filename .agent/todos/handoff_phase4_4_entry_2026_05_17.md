# Handoff: Phase 4-4.5 paradigm 통일 진행중 → 4-4.6 paradigm equivalence 진입

**작성**: 2026-05-17
**갱신**: 2026-05-17 (Phase 4-4.5 paradigm 시그니처 통일 commit 3건, equivalence 미해소)
**HEAD**: `57c0effd fix(session): Phase 4-4.5 KV cache initial_capacity 정합 (production 공식)`
**다음 세션 진입 문장 (사용자)**: "4-4.6 paradigm equivalence 진행" 또는 "paradigm gap 깊이 분석"

---

## TL;DR

**Phase 4-4.5 paradigm 시그니처 통일 완성** (3 commits: `6292a9d0` + `7f7c6856` + `57c0effd`). `DecodeLoop::prefill -> Result<Vec<f32>>`, `DecodeLoop::run(budget, first_token)`, `ModelForward::prefill` chunked, `build_standard_loop`에 `initial_kv_capacity` 인자, happy path 블록에 TBT 출력. 호스트 게이트 PASS (release build / 21 session unit test / layer_lint diff 0).

**중대 미해소 회귀**: S25 OpenCL 측정에서 **G6 bit-identical FAIL — decode step 1부터 분기**:
- Baseline (production fallback, `--no-gpu-plan` 포함): `Paris. It has a...`
- Post-fix (happy path): `Paris. Paris is a...`
- 첫 sample (`Paris`)과 step 0 output (`.`) 일치 → prefill numeric OK
- step 1 (input=`.`)부터 분기 → decode 경로의 numeric divergence

**3 fix 모두 반증** (단 각 fix는 paradigm 정합성에 기여):
1. `prefill_workspace: None` (production owned ws 일치) — step 1 분기 그대로
2. `initial_kv_capacity` production 공식 (128 vs 512) — step 1 분기 그대로
3. baseline `--no-gpu-plan` 비교 (forward_into만, plan path 제외) — step 1 분기 그대로

**G7 보류** (G6 FAIL): n=1 참고치 Decode +7.7% / Avg TBT +8.1%, 5-run median 미측정.

**다음**: **Phase 4-4.6 — paradigm equivalence 깊이 분석** (`ModelForward::step` byte-level diff + step 1 logits[:10] dump + KV buffer zero-init + GPU alloc 순서 검증).

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
| **4-4.5 paradigm 시그니처** | ✅ 시그니처 / ❌ equivalence | 3 commits `6292a9d0`+`7f7c6856`+`57c0effd`. 호스트 PASS, S25 OpenCL G6 FAIL (step 1 분기, 3 fix 반증) |
| **4-4.6 paradigm equivalence** | ⏳ **ready (진입 가능)** | byte-level diff + KV zero-init + logits dump |
| 4-5 chat 전면 재작성 | ⏳ blocked | ChatTurnExec 폐기, `session/chat/{repl,turn,stop_condition}.rs`, V-11 해소 |

---

## Phase 4-4.5 진척 (3 commits)

| Commit | 내용 | 동기 |
|---|---|---|
| `6292a9d0` | `DecodeLoop::prefill -> Result<Vec<f32>>`, `run(budget, first_token)`, `ModelForward::prefill` chunked (derive_chunk_size), generate.rs:3046 호출자 재작성 + 256 가드 제거 | paradigm 시그니처 통일 (Phase 4-3의 last logits 누락 issue 해소) |
| `7f7c6856` | `ModelForward::prefill`의 forward_into 인자 `prefill_workspace: None` (production owned 자동 alloc path 사용) + happy path 블록에 TTFT/Decode/Avg TBT 출력 (production format) | paradigm equivalence fix 시도 #1 + G7 측정 가능화 |
| `57c0effd` | `build_standard_loop` 시그니처에 `initial_kv_capacity` 인자 추가, generate.rs main()의 production 공식 결과 (128) 위임 | paradigm equivalence fix 시도 #2 |

## 측정 데이터 (S25 OpenCL Qwen 2.5 1.5B Q4_0)

`papers/eurosys2027/_workspace/experiment/phase4_4_5_s25_paradigm_unified_2026_05_17.md` (297 lines):
- Fix 1 (prefill_workspace=None): G6 FAIL — `Paris. Paris is a...` (post) vs `Paris. It has a...` (baseline)
- Fix 2 (KV capacity 128): G6 FAIL — 동일 분기 위치
- Fix 3 (baseline `--no-gpu-plan`): G6 FAIL — plan path 가설 반증
- 공통: step 0 (first_token=`Paris`) + step 1 output (`.`)까지 일치, step 2 (input=`.`의 forward)부터 분기
- n=1 참고치: Decode 27.83 → 29.97 ms/tok (+7.7%), TBT 29.53 → 31.91 ms (+8.1%)

## Phase 4-4.6 fix 후보 (priority 순)

| Pri | Fix | 작업 LOC | 검증 비용 |
|---|---|---|---|
| **P1** | **KV cache buffer zero-init** (`alloc_standard_kv_caches`에서 alloc 후 backend.write_buffer로 zero-fill). OpenCL alloc은 garbage 상태 (`memory.rs:42` 확인). production과 다른 GPU memory pool 영역 alloc 시 다른 garbage → attention mask 외 영역 영향 가능 | +20 LOC | 디바이스 1회 |
| **P2** | **step 1 logits[:10] dump** — ModelForward::step에 #[cfg(feature="decode-dump")] 분기로 첫 step 후 logits 첫 10개 stderr 출력. baseline에도 동일 dump (env flag). bit-level diff 위치 식별 | +30 LOC | 디바이스 1회 |
| **P3** | **GPU buffer alloc 순서 일치** — ModelForward::new의 alloc 순서를 production fallback path와 일치 (현재: decode_workspace → decode_input → x_gen → 2× logits → KV. production: KV → ... → x_gen → gen_ws). LayerWorkspace::new 안 9개 buffer 순서까지 동일하게 | +50 LOC (struct 필드 재배치) | 디바이스 1회 |
| **P4** | **ModelForward::step `forward_into` 인자 byte-level diff** — happy path 가드 None 인자들이 다 None인 것은 확인, 그러나 `memory: &dyn Memory` trait object의 vtable identity가 다를 가능성 (decode_mem vs self.memory). Memory trait의 `used_memory()` 부수 효과 확인 | +5 LOC + log | 디바이스 1회 |
| **P5** | **decode_workspace shape/내부 state** — LayerWorkspace::new 안의 `scores: vec![0.0; n_heads * max_seq_len]` 값이 prefill 후 production은 어떻게 변하는가? attention kernel이 score 누적하는가? | 분석 | (분석만) |

권장 순서: P1 → 1회 측정. PASS이면 4-4.6 종료. FAIL이면 P2 (logits dump)로 분기 위치 정확히 식별 후 P3/P4 fix.

## 측정 절차 (재현용)

```bash
# Baseline binary는 디바이스에 이미 (HEAD pre-4-4 또는 6a224c9f)
adb -s R3CY408S5SB shell 'ls -la /data/local/tmp/generate_baseline'

# Post-fix binary push
python scripts/run_device.py -d galaxy_s25 --skip-exec generate

# G6 측정 — temperature 0 + top-k 1 (deterministic, baseline은 --no-gpu-plan 추가)
adb -s R3CY408S5SB shell 'cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./generate_baseline \
    --model-path /data/local/tmp/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /data/local/tmp/qwen-tokenizer.json \
    --backend opencl --prompt "The capital of France is" \
    --num-tokens 32 --max-seq-len 512 --temperature 0 --top-k 1 --no-gpu-plan'

adb -s R3CY408S5SB shell 'cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./generate \
    --model-path /data/local/tmp/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /data/local/tmp/qwen-tokenizer.json \
    --backend opencl --prompt "The capital of France is" \
    --num-tokens 32 --max-seq-len 512'
```

happy path 진입 확인 라인 (post에만): `[Phase4-4.5] standard happy path → DecodeLoop+ModelForward (tokens=5, budget=32)`

## 위험 / 주의

- **dead code 잔존**: `ModelForward`의 `prefill_workspace`, `max_seq_len` 필드와 `ensure_prefill_workspace` 메서드에 `#[allow(dead_code)]` 처리됨. paradigm equivalence 통과 + caller-reuse 재도입 시점에 cleanup. 또는 fix 후보 결정 후 제거.
- **3 fix가 paradigm 시그니처 정합성 자체는 개선**: revert 비권장. equivalence만 추가 작업.
- **alloc 순서 일치 (P3)**가 가장 큰 변경 가능. struct 필드 순서까지 갈 수 있음 — Phase 4-4.6 별도 sprint scope.

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
