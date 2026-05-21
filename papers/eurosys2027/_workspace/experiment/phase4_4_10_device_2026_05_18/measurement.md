# Phase 4-4.10 default invert (AOS) — S25 device 측정

**날짜**: 2026-05-18
**HEAD**: Phase 4-4.10 D1~D2 (pending commit)
**디바이스**: Galaxy S25 (R3CY408S5SB) Adreno 830
**모델**: Qwen 2.5-1.5B Q4_0 GGUF
**프롬프트**: "The capital of France is"
**파라미터**: `--num-tokens 32 --max-seq-len 512 --greedy --repetition-penalty 1.1 --backend opencl`

## 의도

Phase 4-4.9에서 도입한 `LLMRS_DISABLE_NOSHUFFLE_DECODE=1` 회피책은 사용자가 명시적으로 켜야만 default 회귀가 해소되는 구조였다. 4-4.10에서는 default 동작을 뒤집어 회귀가 default에서 발생하지 않도록 한다.

핵심 인사이트: noshuffle SOA active weight 변환은 GPU-only inference를 가정한다. CPU fallback path (`switch_hw cpu`, `--tensor-partition`, prefill CPU chunk)가 활성화된 시나리오에서는 SOA로 변환된 active weight가 silent garbage를 유발할 수 있다. AOS default는 이 가정에서 자유롭다.

## 환경변수 (default invert)

| flag | 동작 |
|---|---|
| (unset, default) | AOS 유지 → standard Q4_0 GEMV path. CPU fallback 안전. 메모리 +702 MiB. |
| `LLMRS_ENABLE_NOSHUFFLE_SOA=1` | (신규) SOA 변환 opt-in. 메모리 -702 MiB. Adreno에서 GEMV +13% 회귀. |
| `LLMRS_SKIP_NOSHUFFLE_SOA=1` | (legacy override) 항상 skip. ENABLE보다 우선. |

## G6' bit-identical 32 tok (n=1)

| 조건 | 출력 | Avg TBT |
|---|---|---|
| default (AOS) | "The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into" | 31.83 ms |
| ENABLE (SOA opt-in) | (bit-identical 동일 출력) | 36.52 ms |

**Bit-identical PASS**. 32 토큰 완전 일치.

## G7' avg_tbt n=5

| run | default (AOS) | ENABLE (SOA) |
|---|---|---|
| 1 | 31.92 | 36.43 |
| 2 | 32.06 | 36.64 |
| 3 | 32.06 | 36.44 |
| 4 | 32.13 | 35.84 |
| 5 | 32.05 | 36.56 |
| **median** | **32.06** | **36.44** |
| mean | 32.04 | 36.38 |

## 회귀 분석

| 비교 기준 | 값 | Δ vs 4-4.7 baseline (32.06 ms) |
|---|---|---|
| Phase 4-4.7 post (baseline) | 32.06 ms | — |
| Phase 4-4.8 (SOA active) | 36.52 ms | +13.7% (회귀) |
| Phase 4-4.9 env=1 (회피) | 32.11 ms | +0.16% |
| **Phase 4-4.10 default (AOS)** | **32.06 ms** | **0.00% PASS** |
| Phase 4-4.10 ENABLE | 36.44 ms | +13.7% (회귀 보존, Path B 측정용) |

**결론**: Default 동작이 4-4.7 post와 bit-equivalent. 회귀 완전 해소. SOA path는 ENABLE opt-in으로 보존되어 향후 Path B (.cl 커널 튜닝) 검증 시 사용 가능.

## 변경 파일

- `engine/src/session/init.rs`:
  - `LLMRS_SKIP_NOSHUFFLE_SOA` only check → `LLMRS_ENABLE_NOSHUFFLE_SOA` opt-in + `LLMRS_SKIP_NOSHUFFLE_SOA` override 통합 logic
  - default skip 시 명시적 stderr 로그
- `engine/src/models/transformer.rs`:
  - 4-4.9에서 추가한 `LLMRS_DISABLE_NOSHUFFLE_DECODE` early-return 제거 (redundant)
- `engine/src/backend/opencl/mod.rs`:
  - `noshuffle_decode_disabled()` 함수 제거
  - `lookup_noshuffle_soa`/`matmul_q4_0::m==1` 가드 제거 (registry-empty 자연 fallback)

## 데이터 파일

- `g6_default.txt` / `g6_enable.txt` — G6' 단일 측정 (출력 비교용)
- `g7_default_n5.txt` / `g7_enable_n5.txt` — G7' avg_tbt 5회 측정

## Path B (backlog) 미래 검증 방법

`LLMRS_ENABLE_NOSHUFFLE_SOA=1`로 SOA path를 명시적으로 활성화한 상태에서 `.cl` 커널을 변형해가며 G7'가 Δ ≤ 5% 이내로 들어오는지 측정. 합격 시 default를 다시 SOA로 되돌릴 수 있다.
