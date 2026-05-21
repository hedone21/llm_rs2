# Phase 4-4-b S25 OpenCL 측정 — paradigm mismatch 발견, 4-4.5 분리

**Date**: 2026-05-17
**Phase**: 4-4-b (generate.rs narrow happy path 분기 추가)
**HEAD**: `e83b87d2 feat(session): Phase 4-4-b — generate.rs narrow happy path 분기 추가`
**Baseline**: `4c71f5f1 docs(handoff): Phase 4-4 next-session entry guide hardened`
**Device**: Galaxy S25 (R3CY408S5SB), Adreno 830, OpenCL backend
**Model**: `qwen2.5-1.5b-q4_0.gguf` (pure Q4_0)

## 측정 목적

Phase 4-4-b의 narrow happy path가 baseline standard generate path와 동일
토큰 sequence + Δ TBT ≤ 5% (G6/G7 게이트)를 만족하는지 검증.

## 명령

```bash
# Baseline (HEAD 4c71f5f1, generate_baseline)
adb shell '... ./generate_baseline --backend opencl --prompt "..." \
    --num-tokens 32 --max-seq-len 512 --greedy'

# 4-4-b (HEAD e83b87d2)
adb shell '... ./generate --backend opencl --prompt "..." \
    --num-tokens 32 --max-seq-len 512'  # GreedySampler default
```

## 결과

### Baseline (pre-4-4, `--greedy` CLI flag)

```
The capital of France is Paris. It has a population of about 2 million people
and covers an area of 104 square kilometers (km2). The city is divided into
TTFT: 218.02 ms
Decode: 27.78 ms/tok
Avg TBT: 29.46 ms
```

### Phase 4-4-b (happy path, DecodeLoop + ModelForward)

```
The capital of France is Paris. The capital of France is Paris. The capital
of France is Paris. The capital of France is Paris. The capital of France is
Paris. The capital
```

(timing 미출력 — happy path 분기는 본 phase에서 timing 로그 미포함)

## Verdict

| Gate | 기준 | 결과 |
|---|---|:---:|
| G1~G5 호스트 게이트 | build / clippy / test / layer_lint | ✅ PASS |
| G6 S25 OpenCL bit-identical 32-token | path_pre_4_4 == path_post_4_4_b | ❌ **FAIL** |
| G7 avg_tbt Δ ≤ 5% | Phase 4-3 baseline | (G6 fail로 미산정) |

## 원인 분석 — paradigm mismatch

Phase 4-3에서 인지된 paradigm mismatch가 실제로 발현:

| 항목 | Production `generate.rs` | `DecodeLoop` |
|---|---|---|
| **prefill 첫 sample** | last logits → argmax → first generated | `prev_token = tokens.last()` (= prompt 마지막) |
| **첫 step input** | first generated (= prompt 마지막 다음 token) | `prev_token` (= prompt 마지막 token) |
| **결과** | prompt 마지막 1회 forward | prompt 마지막 **2회 forward** |

`DecodeLoop::prefill(tokens) -> Result<()>`이 반환값 무시 + `prev_token` 을
prompt 마지막으로 설정하는 구조. `Forward::prefill`은 이미 `Vec<f32>` 반환
(Phase 4-3에서 paradigm 통일됨)이지만 `DecodeLoop` 측이 사용 안 함.

Phase 4-3 probe_inference_loop microbench는 direct path를 DecodeLoop
paradigm으로 통일하여 bit-identical 비교 — **paradigm-level** 검증만 수행.
production-level 검증은 본 phase에서 처음 노출.

## 결정 (사용자, 2026-05-17)

**(B) 4-4.5 sprint로 분리** — 4-4-b는 현 상태로 종료, G6 bit-identical 게이트
완화. token sequence 차이는 Phase 4-4.5에서 `DecodeLoop::prefill/run`
시그니처 변경으로 해소.

### Phase 4-4.5 sprint scope

`DecodeLoop::prefill(tokens: &[u32]) -> Result<Vec<f32>>` (last logits 반환) +
`DecodeLoop::run(budget, first_token: u32) -> Result<DecodeResult>` (첫 token
명시 받음) 시그니처 변경. 4-4-b happy path 호출자도 paradigm 일치 — prefill
last logits → sample → first_token → run 패턴으로 재작성.

추가 작업 (4-4.5 통합 후보):
- ModelForward chunked prefill 지원
- optional collector wiring (score_accumulator / skip_config / importance /
  variance / profiler)

## 후속

- Phase 4-4-b 현 binary (HEAD e83b87d2)는 happy path 분기 wiring 자체는 동작
  검증됨 (호스트 CPU "Paris. The capital" 정상 디코딩, S25 OpenCL 분기 진입
  + 32 토큰 정상 생성). DecodeLoop integration 패턴 production-feasible 확인.
- 4-4-c (main() 진입부 분기 정돈) + 4-4-d (cleanup + G8 재측정)는 그대로 진행
- 4-4-d의 G8 측정은 4-4.5 완료 후 paradigm 통일 binary로 재진행 (handoff 갱신)
