# Handoff: Phase 4-4.6 paradigm equivalence 확정 → 4-4.7 plan-aware ModelForward

**작성**: 2026-05-17
**갱신**: 2026-05-18 (Phase 4-4.6 paradigm equivalence 확정, G6 PASS, G7 fair FAIL)
**HEAD**: `<pending commit>` (Phase 4-4.6 happy path guard + dead KV drop)
**다음 세션 진입 문장 (사용자)**: "Phase 4-4.7 plan-aware ModelForward" 또는 "Phase 4-5 chat 진입"

---

## TL;DR

**Phase 4-4.6 G6 bit-identical PASS — Paradigm equivalence 확정.** 32 토큰 string-identical (S25 OpenCL Qwen 2.5 1.5B Q4_0, `--repetition-penalty 1.0` 조건).

**진짜 분기 원인**: `forward_into`는 양쪽 path bit-identical, 그러나 **sampler 정책 차이** — `GreedySampler` (raw argmax) vs `sampling::sample` (default `repetition_penalty=1.1` 적용).

**Fix (옵션 A 선택)**: `is_standard_happy_path` 가드에 `args.repetition_penalty == 1.0` 조건 추가. 기본 CLI는 fallback path 사용, `--repetition-penalty 1.0` 명시 시만 happy path 진입.

**G7 fair FAIL** (+7.73% avg_tbt, 게이트 5%). 원인 = ModelForward가 plan path 미지원 — production fallback (plan ON, execute_plan) vs happy path (forward_into only). 별도 후속 sprint 필요.

---

## Phase 4-4.6 진척 (1 commit)

| Commit | 내용 | 결과 |
|---|---|---|
| `<pending>` | happy path 가드 `repetition_penalty == 1.0` 추가, generate.rs dead KV drop, 2 test 갱신/추가 | G6 PASS, G7 fair FAIL |

## 측정 데이터

| 측정 | 위치 |
|---|---|
| P1 KV zero-init (FAIL) | `papers/eurosys2027/_workspace/experiment/phase4_4_6_p1_kv_zero_init_2026_05_18.md` |
| P6 dead KV drop (FAIL) | `papers/eurosys2027/_workspace/experiment/phase4_4_6_p6_dead_kv_drop_2026_05_18.md` |
| **P2 logits dump (key)** | `papers/eurosys2027/_workspace/experiment/phase4_4_6_p2_logits_dump_2026_05_18.md` |
| **P2 sampler verify (key)** | `papers/eurosys2027/_workspace/experiment/phase4_4_6_p2_sampler_verify_2026_05_18.md` |
| **옵션 A 최종 (G6 PASS)** | `papers/eurosys2027/_workspace/experiment/phase4_4_6_option_a_final_2026_05_18.md` |
| **G7 fair (+7.73%)** | `papers/eurosys2027/_workspace/experiment/phase4_4_6_g7_fair_2026_05_18.md` |

## 디버그 여정 (반증된 5개 가설)

| Fix | 가설 | 결과 |
|---|---|---|
| Phase 4-4.5 #1 (prefill_workspace=None) | workspace 재사용 | step 1 분기 |
| Phase 4-4.5 #2 (KV capacity 128) | KV 용량 불일치 | step 1 분기 |
| Phase 4-4.5 #3 (baseline `--no-gpu-plan`) | plan path 차이 | step 1 분기 |
| P1 (KV zero-init) | KV garbage 영향 | step 1 분기 — write-before-read 보장 |
| P6 (dead KV drop) | GPU pool offset | step 1 분기 — pool 위치 무관 |

**P2 logits dump가 결정적 단서**: step 0/1 forward_into output `logits[:10]` 양쪽 bit-identical → forward 자체는 동일 → divergence는 sampler 단계.

## Phase 4-4.7 후속 sprint (다음 진입 가이드)

### 목표

ModelForward plan path 지원 → G7 fair Δ ≤ 5% PASS. 기본 CLI에서도 happy path 진입 가능 (rep penalty 1.0 강제 조건 제거 또는 별도 sampler wire).

### 두 갈래 task

**A. plan-aware ModelForward (성능)**:
- ModelForward에 `Option<GpuPlan>` 필드 추가
- `step()` 진입 시 plan 존재하면 `execute_plan` 사용, 아니면 `forward_into`
- `build_plan` 자동 호출 (첫 step 후, KV resize 시 invalidate)
- 예상 LOC: +120, 위험: 중 (plan path 시그니처 + KV resize 동기화)

**B. sampler trait 확장 (정확성)**:
- `RepetitionPenaltySampler(SamplingConfig)` 신규 — `sampling::sample` wrapping
- `build_standard_loop`에 `SamplingConfig` 인자 추가
- `is_standard_happy_path`의 `repetition_penalty == 1.0` 가드 제거
- 예상 LOC: +60, 위험: 낮음

권장: **A + B 동시 sprint**. A로 성능 게이트 통과, B로 happy path 가드 정상화.

## main() ≤ 400 LOC 게이트 보류

실측 main() 본체 = line 77~6257 = **6,180 LOC**. eval-ll / ppl / batch / chat 분기들이 inline body로 남아 있음. Phase 4-5 (chat 재작성) + Phase 4-4.7 (plan-aware) 완료 후 추가 분기 추출 sprint 필요.

---

## 진행 상태

### Task 리스트

| ID | 상태 | 작업 |
|---|---|---|
| #1 | ✅ | ARCHITECTURE.md + spec INV-LAYER-001~005 |
| #2 | ✅ | UNRESOLVED-A~E 5건 결정 |
| #3 | ✅ | spec test + baseline + layer_lint 도구 |
| **#4** | 🔄 in_progress | **L5/L4 분리** (4-1~4-4.6 ✅ / 4-4.7 pending / 4-5 pending) |
| #5 | ⏳ blocked | L1/L2 경계 정리 |
| #6 | ⏳ blocked | L3 도메인 재배치 |
| #7 | ⏳ blocked | Cross-cutting 분리 |
| #8 | ⏳ blocked | /simplify 코드 정리 |
| #22 | ✅ | Phase 4-4.5 paradigm 통일 |
| #31 | 🔄 (마감 직전) | Phase 4-4.6 paradigm equivalence — G6 PASS 종결 (commit 후 completed) |

### Phase 4 sub-phase 진행도

| Sub-phase | 상태 | 결과 |
|---|---|---|
| 4-1 외곽 추출 | ✅ commit `f637722e` | `session/init.rs` (~1,030 LOC), `session/cli.rs` |
| 4-2 trait + Builder | ✅ commits `85ff756c`~`584496b7` | 6 trait + DecodeLoopBuilder + INV-LAYER-006/007 |
| **4-3 ModelForward + microbench** | ✅ 호스트 + S25 | 호스트 Δ=1.53%, S25 Δ=2.29% bit-identical |
| **4-4 main() 조립자화** | ✅ a/b/d (c skip) | happy path 진입 분기 |
| **4-4.5 paradigm 시그니처** | ✅ | `prefill -> Vec<f32>` + `run(budget, first_token)` + chunked prefill |
| **4-4.6 paradigm equivalence** | ✅ **G6 PASS** / G7 fair FAIL | sampler 정책 차이 확정, rep_penalty==1.0 가드 |
| **4-4.7 plan-aware ModelForward** | ⏳ **next entry** | A: plan path 지원, B: sampler trait 확장 |
| 4-5 chat 전면 재작성 | ⏳ blocked | ChatTurnExec 폐기 |

---

## 측정 절차 (재현용)

```bash
# binary push
python scripts/run_device.py -d galaxy_s25 --skip-exec generate

# G6 bit-identical (PASS)
adb -s R3CY408S5SB shell 'cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./generate \
    --model-path /data/local/tmp/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /data/local/tmp/qwen-tokenizer.json \
    --backend opencl --prompt "The capital of France is" \
    --num-tokens 32 --max-seq-len 512 \
    --temperature 0 --top-k 1 --repetition-penalty 1.0'
# Expected: "[Phase4-4.5] standard happy path → DecodeLoop+ModelForward" stderr 라인 + 32 토큰 baseline과 일치

# G7 fair (FAIL: +7.73%)
# Fallback (plan ON, rep_penalty=1.1 default → happy path 우회)
adb -s R3CY408S5SB shell 'for i in 1 2 3 4 5; do cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./generate \
    --model-path /data/local/tmp/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /data/local/tmp/qwen-tokenizer.json \
    --backend opencl --prompt "The capital of France is" \
    --num-tokens 32 --max-seq-len 512 --temperature 0 --top-k 1 \
    2>&1 | grep "Avg TBT"; done'
# Happy (rep_penalty=1.0 → happy path 진입)
adb -s R3CY408S5SB shell 'for i in 1 2 3 4 5; do cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./generate \
    --model-path /data/local/tmp/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /data/local/tmp/qwen-tokenizer.json \
    --backend opencl --prompt "The capital of France is" \
    --num-tokens 32 --max-seq-len 512 --temperature 0 --top-k 1 --repetition-penalty 1.0 \
    2>&1 | grep "Avg TBT"; done'
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

## 확정 결정 (Phase 4-4.6)

- **happy path 가드 = `repetition_penalty == 1.0` 추가** (옵션 A)
- **paradigm equivalence 원인 = sampler 정책 차이** (forward_into는 bit-identical)
- **G7 fair 회귀 +7.73% = plan path 미지원** (별도 Phase 4-4.7 sprint)
- main() ≤ 400 LOC 게이트 보류 (실측 6,180 LOC)

## 참조 문서

- `arch/inference_pipeline.md` §2~§11 — trait API + Builder + Migration
- `ARCHITECTURE.md` §13 (§13.7 Step 2 sub-phase)
- `spec/41-invariants.md` §3.26 (INV-LAYER-001~007)
- `engine/src/session/forward/model_forward.rs` — ModelForward
- `engine/src/session/assembly/build_standard_loop.rs` — happy path 가드 + 조립자
- `engine/src/core/sampling.rs` — `sampling::sample` (repetition_penalty 구현)
- `engine/src/session/defaults.rs` — `GreedySampler` (raw argmax)
