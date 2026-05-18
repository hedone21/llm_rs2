# Handoff: Phase 4-4.8 plan-path 진단 sprint 종결 → 4-4.9 noshuffle Adreno tuning

**작성**: 2026-05-17
**갱신**: 2026-05-18 (Phase 4-4.8 RCU publish fix + G7' FAIL +13.7%)
**HEAD**: `5e7e0015` (Phase 4-4.8 C3 RCU publish 누락 fix + 진단 trace)
**다음 세션 진입 문장 (사용자)**: "Phase 4-4.9 noshuffle Adreno tuning" 또는 "Phase 4-5 chat 진입"

---

## TL;DR — Phase 4-4.8 종결 결과

| Gate | 결과 |
|---|---|
| **G6' bit-identical 32 tok** | **PASS** (publish fix 후도 동일 출력) |
| **G7' avg_tbt n=5** | **FAIL +13.7%** (4-4.7 post 32.06 ms → 4-4.8 36.52 ms) |

**root cause 식별 완료**: `transformer.rs::prepare_noshuffle_buffers` (l.1101~1146)이
RCU pattern의 publish step을 누락. `slot.load_weights().clone()` 후 local layer를
swap했으나 `slot.store_weights_same_dtype(Arc::new(layer))` 호출 없음 →
ArcSwap current Arc 불변 → `slot.load_weights().wq.buffer()`는 영원히 AOS cl_mem
반환 → noshuffle SOA registry (196 entry, d_buf key)는 stale → build_plan의
"Q4_0 SOA entry missing" 가드에서 None → sticky lock-out.

**fix 적용**: `layer_mutated` flag + `slot.store_weights_same_dtype(Arc::new(layer))`
publish step 추가 (commit `5e7e0015`).

**fix 검증**:
- `[build_plan-trace] layer0 wq key=... is_NoshuffleWeightBuffer=true` ✓
- `[fwd-trace] build_plan SUCCESS` ✓
- G6' bit-identical PASS ✓

**G7' 회귀 origin 격리**:
- plan ON (default): median 36.48 ms
- `--no-gpu-plan`: median 36.53 ms
- 차이 ≤0.2% → plan path 자체는 정상 진입했으나, NoshuffleWeightBuffer 활성화로
  matmul_q4_0의 m==1 noshuffle GEMV dispatch가 매 token 호출. Adreno 830에서
  standard AOS Q4_0 GEMV 대비 측정상 +4 ms/tok 느림 (feedback
  `feedback_adreno_subgroup_reduce.md`의 "이론에 끌리지 말고 cross-run 실측 후
  결정" 원칙 사례).

---

## Phase 4-4.7 진척 (3 commit)

| Commit | 내용 | 게이트 |
|---|---|---|
| `b412837a` C1 | TokenSampler::observe_token default + RepetitionPenaltySampler + DecodeLoop run() observe + SamplingConfig Clone | G1/G2/G3 PASS |
| `c971e701` C2 | build_standard_loop SamplingConfig + prompt seeding + happy path 가드(`repetition_penalty == 1.0` 제거) + generate.rs first_token sampling::sample 통일 | G1/G2/G3/G4 PASS |
| `64c6dde9` C3 | plan-aware ModelForward (gpu_plan/sticky_disabled/plan_enabled + try_build_plan + Option::take borrow) + is_standard_happy_path tensor_partition/swap_* 가드 + LLMRS_FWD_TRACE 진단 trace | G1/G2/G3/G4 PASS / G6' PASS / **G7' FAIL** |

## 측정 데이터

| 측정 | 위치 |
|---|---|
| **Phase 4-4.7 device 측정 (G6'/G7' + LLMRS_FWD_TRACE)** | `papers/eurosys2027/_workspace/experiment/phase4_4_7_device_2026_05_18/measurement.md` |
| pre_g6.txt / post_g6.txt | 같은 디렉토리 |
| pre_g7_n5.txt / post_g7_n5.txt | 같은 디렉토리 |

## Phase 4-4.8 진척 (commit `e0f732b9` + `5e7e0015` + `(measurement docs)`)

| Commit | 내용 | 게이트 |
|---|---|---|
| `e0f732b9` C1 | transformer.rs::build_plan 10개 None-return 경로에 `LLMRS_BUILD_PLAN_TRACE=1` env-gated eprintln 추가 + build_standard_loop too_many_arguments allow | G1/G2/G3/G4 PASS |
| `5e7e0015` C3 | prepare_noshuffle_buffers RCU publish 누락 fix + lookup_noshuffle_soa MISS trace + build_plan layer0 is_NoshuffleWeightBuffer trace | G1/G2/G3/G4 PASS / G6' PASS / **G7' FAIL** |

진단 trace 환경변수 (production code 유지):
- `LLMRS_BUILD_PLAN_TRACE=1` — build_plan 10개 None-return 경로 line:file 출력
- `LLMRS_NOSHUFFLE_SOA_TRACE=1` — lookup miss key + registry sample 출력
- `LLMRS_FWD_TRACE=1` — ModelForward.try_build_plan 진입/거부/SUCCESS 출력

unset 시 syscall 1회만 (overhead 0 fast path).

## Phase 4-4.9 후속 sprint (다음 진입 가이드)

### 목표

G7' 회귀 origin (`matmul_q4_0`의 m==1 noshuffle GEMV dispatch)을 Adreno-friendly로 처리하여 Δ ≤ 5% PASS.

### 진입 절차 — 두 가지 path 중 선택

**Path A — fast escape**: `LLMRS_DISABLE_NOSHUFFLE_DECODE=1` env gate 추가 (matmul_q4_0 l.2258~2273 if m==1 block 우회).
- LOC: +10
- 측정: G7' 재측정 시 Δ ≤ 5% 예상 (plan path는 OK, decode 시 std GEMV path 사용)
- 단점: noshuffle 잠재력 포기

**Path B — Adreno tuning**: noshuffle GEMV kernel을 cross-run profile하여 standard AOS Q4_0 GEMV와 동등하거나 빠른 variant 도출.
- LOC: 미상 (.cl 커널 수정 — Senior Implementer 위임 + feedback_cl_modification 적용)
- 측정: matmul microbench → noshuffle vs std 직접 비교 후 결정
- 단점: 깊은 .cl tuning 필요, sprint 길어짐

### 권장: Path A 먼저, Path B는 backlog

Path A로 G7' PASS 확정 + 4-4 sprint cleanup. Path B는 `.agent/todos/backlog.md`에
"Adreno noshuffle GEMV cross-run tuning" 항목으로 분리.

### 권장: ModelForward decode_workspace vs production gen_ws cl_mem 비교 우선

production fallback에서 build_plan SUCCESS인 반면 ModelForward에서 None — 가장 가능성 큰 차이는 workspace의 cl_mem 동일성. ModelForward는 `LayerWorkspace::new(workspace_config_for(...), memory.as_ref(), backend)`로 alloc. production은 동일 시그니처. 하지만 ModelForward는 backend 인스턴스가 Arc::clone, production은 직접 borrow. memory ptr 동일성도 확인 필요.

### 예상 LOC

- transformer.rs::build_plan trace 추가: +30 (env flag 가드, 후속 sprint 마무리 시 cleanup 또는 production-friendly로 남김)
- root cause fix: 미상 (가설별 LOC 다름)

---

## main() ≤ 400 LOC 게이트 보류

실측 main() 본체 = line 77~6257 = **6,180 LOC**. eval-ll / ppl / batch / chat 분기들이 inline body로 남아 있음. Phase 4-5 (chat 재작성) + Phase 4-4.8 (plan-path 진단) 완료 후 추가 분기 추출 sprint 필요.

---

## 진행 상태

### Task 리스트

| ID | 상태 | 작업 |
|---|---|---|
| #1 | ✅ | ARCHITECTURE.md + spec INV-LAYER-001~005 |
| #2 | ✅ | UNRESOLVED-A~E 5건 결정 |
| #3 | ✅ | spec test + baseline + layer_lint 도구 |
| **#4** | 🔄 in_progress | **L5/L4 분리** (4-1~4-4.8 ✅ / 4-4.9 pending / 4-5 pending) |
| #5 | ⏳ blocked | L1/L2 경계 정리 |
| #6 | ⏳ blocked | L3 도메인 재배치 |
| #7 | ⏳ blocked | Cross-cutting 분리 |
| #8 | ⏳ blocked | /simplify 코드 정리 |
| #22 | ✅ | Phase 4-4.5 paradigm 통일 |
| #31 | ✅ | Phase 4-4.6 paradigm equivalence — G6 PASS |
| #32 | ✅ | Phase 4-4.7 C1 — TokenSampler observe_token + RepetitionPenaltySampler |
| #33 | ✅ | Phase 4-4.7 C2 — build_standard_loop SamplingConfig + 가드 정상화 |
| #34 | ✅ (G6' PASS, G7' FAIL 종결) | Phase 4-4.7 C3 — plan-aware ModelForward |
| #35 | ✅ | Phase 4-4.8 C1 — build_plan None-return trace 추가 |
| #36 | ✅ | Phase 4-4.8 C2 — S25 device trace 측정 + root cause 식별 (RCU publish 누락) |
| #37 | ✅ (G6' PASS, G7' FAIL 종결) | Phase 4-4.8 C3 — RCU publish fix + G7' 재측정 (회귀 origin = noshuffle GEMV) |

### Phase 4 sub-phase 진행도

| Sub-phase | 상태 | 결과 |
|---|---|---|
| 4-1 외곽 추출 | ✅ commit `f637722e` | `session/init.rs` (~1,030 LOC), `session/cli.rs` |
| 4-2 trait + Builder | ✅ commits `85ff756c`~`584496b7` | 6 trait + DecodeLoopBuilder + INV-LAYER-006/007 |
| 4-3 ModelForward + microbench | ✅ 호스트 + S25 | 호스트 Δ=1.53%, S25 Δ=2.29% bit-identical |
| 4-4 main() 조립자화 | ✅ a/b/d (c skip) | happy path 진입 분기 |
| 4-4.5 paradigm 시그니처 | ✅ | `prefill -> Vec<f32>` + `run(budget, first_token)` + chunked prefill |
| 4-4.6 paradigm equivalence | ✅ G6 PASS / G7 fair FAIL | sampler 정책 차이 확정, rep_penalty==1.0 가드 |
| **4-4.7 plan-aware ModelForward + sampler 확장** | ✅ **G6' PASS** / **G7' FAIL** | sampler trait 확장 + plan-aware step + LLMRS_FWD_TRACE 진단 |
| **4-4.8 plan-path 진단** | ✅ **G6' PASS** / **G7' FAIL +13.7%** | build_plan None-return trace 추가 + RCU publish 누락 fix. G7' 회귀 origin = noshuffle GEMV |
| **4-4.9 noshuffle Adreno tuning** | ⏳ **next entry** | Path A (env gate) 또는 Path B (.cl tuning) — G7' Δ ≤ 5% 목표 |
| 4-5 chat 전면 재작성 | ⏳ blocked | ChatTurnExec 폐기 |

---

## 측정 절차 (재현용)

```bash
# binary push (자동 NDK env 주입)
python scripts/run_device.py -d galaxy_s25 --skip-exec generate

# G6' bit-identical 32 tok
adb -s R3CY408S5SB shell 'cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./generate \
    --model-path /data/local/tmp/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /data/local/tmp/qwen-tokenizer.json \
    --backend opencl --prompt "The capital of France is" \
    --num-tokens 32 --max-seq-len 512 \
    --greedy --repetition-penalty 1.1'
# Expected (Phase 4-4.7 post): "[Phase4-4.5] standard happy path" stderr 로그 + 32 토큰 identical

# G7' avg_tbt n=5 (default args + --greedy)
adb -s R3CY408S5SB shell 'for i in 1 2 3 4 5; do cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./generate \
    --model-path /data/local/tmp/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /data/local/tmp/qwen-tokenizer.json \
    --backend opencl --prompt "The capital of France is" \
    --num-tokens 32 --max-seq-len 512 --greedy --repetition-penalty 1.1 \
    2>&1 | grep "Avg TBT"; done'

# Phase 4-4.8: plan path 진단
adb -s R3CY408S5SB shell 'cd /data/local/tmp && LLMRS_FWD_TRACE=1 LD_LIBRARY_PATH=/data/local/tmp ./generate \
    --model-path /data/local/tmp/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /data/local/tmp/qwen-tokenizer.json \
    --backend opencl --prompt "The capital of France is" \
    --num-tokens 4 --max-seq-len 512 --greedy --repetition-penalty 1.1 \
    2>&1 | grep -E "(fwd-trace|build-plan-trace)"'
# Phase 4-4.7: "build_plan returned None → sticky lock"
# Phase 4-4.8: 추가 build-plan-trace 라인이 어느 None 반환 line인지 식별
```

---

## 환경 / 규칙 (불변)

- **언어**: 한국어 (CLAUDE.md 시스템 지시)
- **자동 commit**: 작업 완료 시. 미커밋 작업 금지
- **자동 알림**: `notify-send "llm.rs" "<요약>"`
- **GGUF 우선**: 기본 모델 포맷
- **.cl 커널**: 기본 회피, 성능 최적화 시만 허용
- **테스트 정책**: 신규 테스트는 `engine/tests/spec/` 하위, inline `#[cfg(test)]` 금지 (trait 사용성 검증용 내부 mod는 예외)
- **TBT metric**: avg_tbt (tok0 inclusive)
- **Adreno 벤치**: Galaxy S25 = 6T만
- **❌ `git worktree` 사용 금지**: baseline 비교는 detached HEAD checkout

---

## 확정 결정 (Phase 4-4.7)

- **plan 수명 = ModelForward 내부 관리** (사용자 결정 A1-Q1, commit `64c6dde9`)
- **TokenSampler::observe_token default no-op 확장** (사용자 결정 A2-Q1, commit `b412837a`)
- **`is_standard_happy_path` 가드 정상화** (`repetition_penalty == 1.0` 제거 + `tensor_partition == 0.0` + swap_* 추가)
- **G7' root cause는 transformer.rs::build_plan 내부** (sprint 범위 외 → Phase 4-4.8 분리)
- **LLMRS_FWD_TRACE 진단 trace는 production code에 유지** (env flag 가드, 후속 sprint 재사용 의도)

## 참조 문서

- `arch/inference_pipeline.md` §2~§11 — trait API + Builder + Migration
- `ARCHITECTURE.md` §13 (§13.7 Step 2 sub-phase)
- `spec/41-invariants.md` §3.26 (INV-LAYER-001~007)
- `engine/src/session/forward/model_forward.rs` — ModelForward + try_build_plan + plan-aware step
- `engine/src/session/assembly/build_standard_loop.rs` — happy path 가드 + sampler 분기 + plan_enabled wiring
- `engine/src/session/samplers/repetition_penalty.rs` — RepetitionPenaltySampler (ring buffer + scratch logits)
- `engine/src/session/traits.rs` — TokenSampler::observe_token default
- `engine/src/session/decode_loop.rs` — prefill prompt seeding + run() observe_token wire
- `engine/src/core/sampling.rs` — SamplingConfig (Clone derive 추가) + sampling::sample
- `engine/src/models/transformer.rs::build_plan` (l.1932) — **Phase 4-4.8 진단 대상**
