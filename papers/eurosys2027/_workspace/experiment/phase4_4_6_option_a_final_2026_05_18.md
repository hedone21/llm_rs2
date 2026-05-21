# Phase 4-4.6 Option A 최종 검증 (2026-05-18, Galaxy S25)

## 결론

**Paradigm equivalence: PASS (bit-identical 32 토큰)**
**TBT G7: FAIL (Δ +7.28% median, 게이트 5% 초과)**

## 변경

`engine/src/session/assembly/build_standard_loop.rs::is_standard_happy_path`에
`args.repetition_penalty == 1.0` 가드 추가. 기본 CLI(`repetition_penalty=1.1`)는
fallback path를 타고, 명시적으로 `--repetition-penalty 1.0`을 주면 happy path
(`DecodeLoop + ModelForward + GreedySampler`)로 진입.

## Test 1: 기본 CLI (rep_penalty=1.1) — PASS

stderr에 `[Phase4-4.5] standard happy path` 메시지 **없음** → fallback path 진입
확인. 출력 정상 ("The capital of France is known by many names, but the one most
people know about is Paris..."), TTFT 219.69 ms, avg_tbt 32.86 ms.

## Test 2: G6 bit-identical — PASS

`--temperature 0 --top-k 1 --repetition-penalty 1.0 --no-gpu-plan` (baseline) vs
`--temperature 0 --top-k 1 --repetition-penalty 1.0` (post-fix happy path) 32 토큰
완전 일치:

> "The capital of France is Paris. Paris is a very big city. It has about 2.2
> million people. It is also a very old city. It was built in the"

post-fix stderr에 `[Phase4-4.5] standard happy path → DecodeLoop+ModelForward
(tokens=5, budget=32)` + `stopped_by=BudgetExhausted final_pos=36` 출력 확인.

## Test 2 G7: 5-run TBT — FAIL

| metric        | baseline (no-plan)    | post-fix (happy)      | Δ       |
|---------------|-----------------------|-----------------------|---------|
| Avg TBT med   | 29.65 ms              | 31.81 ms              | +7.28%  |
| Avg TBT mean  | 29.63 ms              | 31.81 ms              | +7.36%  |
| Decode med    | 27.83 ms/tok          | 29.86 ms/tok          | +7.29%  |

baseline runs (ms): 29.50, 29.62, 29.65, 29.66, 29.74
post-fix runs (ms): 31.71, 31.77, 31.81, 31.83, 31.95

게이트 5% 초과. 정확성은 PASS이나 성능 회귀가 측정됨.

## 회귀 원인 후보

1. **plan path 사용 차이** — baseline은 `--no-gpu-plan`(deterministic) 비교 기준,
   post-fix happy path는 plan을 사용. 비교 fair 여부 재확인 필요. (참고: Test 1
   기본 CLI는 plan ON + fallback path에서 32.86 ms/tok로 post-fix와 유사.)
2. **DecodeLoop overhead** — trait dispatch / lifecycle hook 호출 / 32 토큰 짧은
   샘플에서 setup 비용 분모가 작아 비율이 부풀려졌을 가능성.
3. **ModelForward decode_workspace eager 할당** — Phase 4-3 호스트 게이트 Δ=2.29%
   였으므로 디바이스에서 Adreno UMA 캐시 동작이 다를 수 있음.

## 권장

- baseline을 plan ON (`--no-gpu-plan` 제거) 조건으로 다시 측정해서 fair 비교.
- 그래도 Δ > 5%면 ModelForward decode_workspace eager 할당을 lazy로 검토.
- 정확성/paradigm equivalence는 확정 PASS이므로 G7만 별도 트랙으로 분리 가능.
