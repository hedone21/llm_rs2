# Phase 4-4.5 — S25 OpenCL DecodeLoop paradigm 통일 검증

- HEAD post: `6292a9d0` (Phase 4-4.5)
- HEAD baseline: 메인 세션이 만든 `/data/local/tmp/generate_baseline` (timestamp 2026-05-17 15:06, HEAD `6a224c9f` 의도)
- Device: Galaxy S25 (R3CY408S5SB), Adreno OpenCL
- Model: Qwen 2.5 1.5B Q4_0 GGUF
- Prompt: "The capital of France is" (5 tokens)
- Decode budget: 32 tokens
- Sampling: --greedy --temperature 0.0
- 측정 일자: 2026-05-17

## G6 — bit-identical 32 token greedy

### baseline (`generate_baseline`) 출력
```
The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into
```

### post-4-4.5 (`generate`) 출력
```
The capital of France is Paris. Paris is a very big city. It has about 2.2 million people. It is also a very old city. It was built in the
```

### 분기 진입 로그 (post)
```
[Phase4-4.5] standard happy path → DecodeLoop+ModelForward (tokens=5, budget=32)
[Phase4-4.5] generated=32 (first=12095 + run=31) stopped_by=BudgetExhausted final_pos=36
```

### 분기 진입 로그 (baseline)
없음 — baseline binary는 `[Phase4-4-b]` 또는 `[Phase4-4.5]` 진입 라인을 stderr로 출력하지 않음. 즉 **baseline binary 가 happy path 분기를 타지 않고 fallback main path 로 흐름**.

### Verdict
**FAIL (검증 불성립)** — baseline 과 post 는 서로 다른 코드 path (fallback vs happy path) 를 실행. 동일 path 출력을 bit-identical 비교 할 수 없음.

토큰 sequence 분기 위치: 두 경로 모두 첫 sample token id = 12095 (` Paris`) 까지 동일하나, 두 번째 token 부터 분기.
- baseline (fallback path): ` Paris . It has a population of about 2 million ...`
- post (happy path): ` Paris . Paris is a very big city ...`

post happy path 출력의 ` Paris . Paris is` 패턴은 step1 에서 prompt 마지막 token 의 forward 결과로 first_token=` Paris` 를 얻은 뒤, step2 에서 `forward.step(prev_token= Paris)` 의 결과가 ` .` 이고, step3 에서 `forward.step(prev_token= .)` 의 결과가 ` Paris` 인 자연스러운 greedy 결과로도 해석 가능. 즉 post 출력이 자체 모순적이지 않다.

다만 **fallback path 의 출력이 정답이라는 보증이 없음** — fallback main loop 가 Phase 4-4-b/4-4.5 와 동일한 inference 의미론을 따르는지는 별도 검증 대상.

## G7 — avg_tbt Δ ≤ 5%

### baseline (5 runs, 6T)
| run | Decode (ms/tok) | Avg TBT (ms) |
|-----|-----------------|--------------|
| 1   | 27.67           | 29.48        |
| 2   | 27.90           | 29.72        |
| 3   | 27.90           | 29.57        |
| 4   | 28.03           | 29.79        |
| 5   | 27.80           | 29.58        |
| median | 27.90        | 29.58        |

### post-4-4.5 (5 runs, 6T)
TBT 측정 **불가능**. happy path 분기 (`generate.rs` Phase 4-4.5 진입 블록) 는 TBT/Decode 통계를 stderr 로 출력하지 않음 — `[Phase4-4.5] generated=...` 라인만 출력하고 main 종료. fallback main path 의 `Decode: X ms/tok` / `Avg TBT: Y ms` 라인은 진입하지 않음.

### Verdict
**측정 불가** — post happy path 가 TBT 통계 라인을 출력하지 않아 정량 비교 불가.

## 발견 사항 요약

1. **baseline binary 가 happy path 미진입**: `/data/local/tmp/generate_baseline` (2026-05-17 15:06 빌드) 는 5-token prompt + 기본 옵션에서도 `[Phase4-4-b]` 분기 라인을 출력하지 않음. 가능한 원인:
   - baseline binary 가 HEAD `6a224c9f` 가 아닌 더 이전 HEAD (Phase 4-4-b 도입 전) 에서 빌드되었을 가능성
   - 또는 `is_standard_happy_path()` 가드 가 어떤 args 로 fail 했을 가능성 (당 측정에서는 happy path 진입 조건을 모두 만족하는 args 사용)

2. **post-4-4.5 happy path 분기 자체에 TBT 통계 출력 코드 부재**: `generate.rs` PHASE 4-4.5 블록 (라인 3034~3090 부근) 은 token sequence 출력 + `[Phase4-4.5] generated=...` 진단 라인만 출력하고 즉시 main 을 종료. `Decode: X ms/tok` 등은 fallback main loop 의 출력. G7 검증을 위해서는 happy path 분기에도 동일한 TBT 통계 출력 추가 필요.

3. **Sequence 비교는 fallback vs happy path 의 의미론 등가성** 으로 재정의되어야 함. 4-4-b 의 의도된 fix (T5 중복 forward 제거) 가 happy path 출력에 어떻게 반영되는지 별도 oracle (e.g. llama.cpp greedy) 비교가 필요.

## 권장 후속 조치 (메인 세션)

- A. baseline binary 재빌드: HEAD `6a224c9f` checkout → `run_device.py -d galaxy_s25 --skip-exec generate` → `adb push generate /data/local/tmp/generate_baseline`. 재빌드 후 `[Phase4-4-b]` 진입 라인 확인.
- B. happy path 분기에 TBT 통계 출력 코드 추가 (generate.rs PHASE 4-4.5 블록). 또는 DecodeLoop::run 이 step 마다 step_ms 를 누적하므로 result 에서 avg 계산.
- C. fallback vs happy path 의미론 등가성 검증 oracle 마련 (예: llama.cpp greedy 또는 CPU forward 결과 와의 4-token bit-identical).
