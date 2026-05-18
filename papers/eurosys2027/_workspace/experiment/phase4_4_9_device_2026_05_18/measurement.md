# Phase 4-4.9 noshuffle decode disable env gate — S25 device 측정

**날짜**: 2026-05-18
**HEAD**: Phase 4-4.9 C1~C3 (pending commit)
**디바이스**: Galaxy S25 (R3CY408S5SB) Adreno 830
**모델**: Qwen 2.5-1.5B Q4_0 GGUF
**프롬프트**: "The capital of France is"
**파라미터**: `--num-tokens 32 --max-seq-len 512 --greedy --repetition-penalty 1.1 --backend opencl`

## 환경변수

`LLMRS_DISABLE_NOSHUFFLE_DECODE=1` 시:
- `transformer.rs::prepare_noshuffle_buffers` 전체 short-circuit (conversion + AOS release 모두 skip)
- `OpenCLBackend::lookup_noshuffle_soa` 항상 None 반환 (build_plan 차단 + m==1 fallthrough 안전망)
- `matmul_q4_0` m==1 분기에서 `noshuffle_decode_disabled()` 가드 (이중 안전망)

unset 시 syscall 1회 + OnceLock 캐시 → hot path overhead 0.

## G6' bit-identical 32 tok (n=1)

| 조건 | 출력 | Avg TBT |
|---|---|---|
| env unset | "The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into" | 36.84 ms |
| env=1 | (동일) | 31.89 ms |

**Bit-identical PASS**. 32 토큰 출력 완전 일치.

## G7' avg_tbt n=5 (final)

| run | env unset | env=1 |
|---|---|---|
| 1 | 36.37 | 31.41 |
| 2 | 36.53 | 31.99 |
| 3 | 35.35 | 32.11 |
| 4 | 32.78 | 32.14 |
| 5 | 38.10 | 32.14 |
| **median** | **36.37** | **32.11** |
| mean | 35.83 | 31.96 |

## 회귀 분석

| 비교 기준 | 값 | Δ |
|---|---|---|
| Phase 4-4.7 post (baseline) | 32.06 ms | — |
| Phase 4-4.8 post (env unset) | **36.37 ms** | **+13.4%** (FAIL, 회귀 재현) |
| Phase 4-4.9 (env=1) | **32.11 ms** | **+0.16%** (PASS, Δ ≤ 5%) |

**결론**: `LLMRS_DISABLE_NOSHUFFLE_DECODE=1` 가드로 Phase 4-4.8 G7' 회귀 +13.7% 완전 해소.

## 가설 검증 (handoff doc과 차이)

Phase 4-4.8 handoff doc은 "matmul_q4_0의 m==1 noshuffle GEMV dispatch가 매 token 호출"이 회귀 원인이라 분석.

본 sprint에서 진단:
1. matmul_q4_0 entry trace (`[matmul_q4_0-entry]`)는 prefill(m=5)만 찍히고 decode m=1은 미발생
   → decode는 `matmul_q4_0` 함수를 우회한다
2. plan path는 `build_plan` SUCCESS 시 `make_q4_0_noshuffle_matmul_step`을 통해
   `kernel_gemv_noshuffle_q4_0`을 직접 pre-build하여 dispatch
3. 즉 회귀 원인 = (handoff 분석과 비교해 일부 정정) **plan path가 noshuffle SOA를
   사용해 빌드한 GEMV 커널** (matmul_q4_0의 m==1 분기가 아닌 plan-resident GEMV)

가드 위치를 `lookup_noshuffle_soa` + `prepare_noshuffle_buffers`로 옮긴 후 회귀 해소.

## 정확성 함정 — lookup 차단만으로는 부족

초기 시도: `matmul_q4_0::m==1` 가드만 차단했으나 변화 없음 (plan path 우회).
2차 시도: `lookup_noshuffle_soa` 차단만으로는 정확성 깨짐 — noshuffle conversion이
선행되어 원본 AOS cl_mem이 release되었기 때문에 standard GEMV가 stale 버퍼 읽음
(generated token=151935 garbage 출력).

최종 해법: **conversion 단계 (prepare_noshuffle_buffers) 자체를 short-circuit**.
이 방식은 노이즈 SOA 메모리 절약(≈702.8 MiB)을 포기하지만 정확성+성능을 모두 보장.

## 데이터 파일

- `g7_unset_final_n5.txt` — env unset 5회 측정
- `g7_set_final_n5.txt` — env=1 5회 측정
- `g6_unset.txt` / `g6_set.txt` — G6' 단일 측정 (출력 비교용)
- `measurement.md` (본 문서)

## 변경 파일

- `engine/src/backend/opencl/mod.rs`:
  - `noshuffle_decode_disabled()` 함수 추가 (OnceLock 캐시)
  - `matmul_q4_0::m==1` 분기에 가드 추가 (이중 안전망)
  - `lookup_noshuffle_soa` 가드 추가 (build_plan path 차단)
- `engine/src/models/transformer.rs`:
  - `prepare_noshuffle_buffers` 시작 부분 env gate short-circuit (메인 가드)
