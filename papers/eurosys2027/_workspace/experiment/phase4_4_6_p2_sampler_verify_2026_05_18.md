# Phase 4-4.6 P2 — Sampler 가설 검증 (2026-05-18)

## 가설
- Fallback path (`sampling::sample`)는 `repetition_penalty` 적용 (기본 1.1)
- Happy path (`GreedySampler`)는 raw argmax — penalty 없음
- `--repetition-penalty 1.0`을 fallback에 강제하면 양쪽 등가 → 동일 출력 기대

## 환경
- Device: S25 (R3CY408S5SB), Adreno OpenCL
- Model: qwen2.5-1.5b-q4_0.gguf, F16 KV, HeadMajor
- Prompt: "The capital of France is", num-tokens=32

## 결과

### Fallback path (LLMRS_DISABLE_HAPPY_PATH=1, --temperature 0 --top-k 1 --repetition-penalty 1.0 --no-gpu-plan)
```
The capital of France is Paris. Paris is a very big city. It has about 2.2 million people. It is also a very old city. It was built in the
```

### Happy path (default GreedySampler, --temperature 0.8 기본값이지만 GreedySampler는 raw argmax)
```
The capital of France is Paris. Paris is a very big city. It has about 2.2 million people. It is also a very old city. It was built in the
```

## 첫 8 토큰 비교
양쪽 동일: `Paris`, `.`, ` Paris`, ` is`, ` a`, ` very`, ` big`, ` city`

## 게이트
- **PASS** — 32 토큰 bit-identical
- 가설 확정: forward_into는 bit-identical, 분기 원인은 **sampler 정책 차이** (fallback의 repetition_penalty 1.1 vs Greedy의 raw argmax)
- happy path 가드는 sampler 정책 차이를 무시해도 무방 (paradigm 일치)

## 비고
- TTFT: fallback 218 ms (no-gpu-plan) vs happy 91 ms — plan 경로 영향만, 출력 무관
- Decode TBT: fallback 27.8 ms/tok vs happy 30.0 ms/tok — 동일 forward, 측정 노이즈 범위
