# B-5b Phase 2 Stage 2-B — S25 microbench 게이트 측정

**날짜**: 2026-05-23
**디바이스**: Galaxy S25 (R3CY408S5SB, Adreno 830)
**모델**: Qwen 2.5-1.5B Q4_0 GGUF (`/data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf`)
**Backend**: opencl
**Threads**: 6 (Galaxy S25 권장)
**Prompt**: "The capital of France is"
**Args**: `--num-tokens 32 --max-seq-len 512 --greedy --repetition-penalty 1.1 --backend opencl --threads 6`
**Runs**: N=5 per HEAD (+ N=5 yield enabled scenario)

## 빌드 조건

| 항목 | 값 |
|---|---|
| target | aarch64-linux-android |
| features | opencl (no-default-features) |
| profile | release (lto=fat) |
| NDK | /opt/android-ndk24 (linux-x86_64) |
| stub | libs/aarch64/libOpenCL.so (메인 워크트리에서 복사) |

## 비교 대상

| HEAD | label | scope |
|---|---|---|
| `6cd09f9b` | stage2a_orig | Stage 2-A 원본 게이트 결과 (이전 세션 측정) |
| `6cd09f9b` | stage2a_recheck | Stage 2-A 재측정 (본 세션, baseline 검증용) |
| `28bd7724` | stage2b | Stage 2-B (yield_after_layer hot-path 치환), default (`LLMRS_DECODE_YIELD_EVERY` unset = disabled) |
| `28bd7724` | stage2b_yield | Stage 2-B + `LLMRS_DECODE_YIELD_EVERY=4 LLMRS_DECODE_YIELD_US=500` (R-1 worst case 정량화) |

## 결과 — avg_tbt (tok0 inclusive)

| run | stage2a_recheck (ms) | stage2b (ms) | stage2b_yield (ms) |
|---|---|---|---|
| 1 | 32.53 | 32.84 | 40.43 |
| 2 | 32.64 | 32.87 | 40.51 |
| 3 | 32.75 | 32.95 | 40.33 |
| 4 | 32.73 | 32.96 | 40.27 |
| 5 | 32.86 | 32.76 | 40.34 |
| **mean** | **32.702** | **32.876** | **40.376** |
| stdev | 0.124 | 0.083 | 0.094 |
| min | 32.530 | 32.760 | 40.270 |
| max | 32.860 | 32.960 | 40.510 |

Stage 2-A 원본 게이트 (이전 세션): mean=32.882, stddev=0.049 ms

## Δ 및 게이트 판정

| 비교 | Δ ms | Δ% |
|---|---|---|
| **Stage 2-B vs Stage 2-A orig (32.882)** | **−0.006** | **−0.018%** |
| Stage 2-B vs Stage 2-A recheck (32.702) | +0.174 | +0.532% |
| Stage 2-A recheck vs orig | −0.180 | −0.547% |
| Stage 2-B yield_on vs default | +7.500 | +22.813% |

**Stage 2-A recheck ↔ orig 차이 −0.547%는 본 세션 baseline noise**. 본 세션 내 Stage 2-B vs Stage 2-A recheck +0.532%도 같은 noise floor 안 (stddev × ~1배).

게이트 임계 (vs Stage 2-A orig 32.882 ms):
- +3% threshold: 33.868 ms
- +5% threshold: 34.526 ms

**Stage 2-B mean = 32.876 ms ≤ 33.868 (+3%) → PASS**

## 게이트 판정

**PASS** — Stage 2-B avg_tbt Δ = **−0.018%** (vs Stage 2-A orig).

- yield_after_layer hot-path trait method 치환 (16 layer × ~250 token × decode = ~4000회/inference) 영향: **noise 이하**.
- LTO=fat가 trait method dispatch에서도 디버추얼라이즈 또는 분기 비용이 측정 가능한 수준이 아님 (Stage 2-A 결론과 동일).

## R-1 worst case 정량화 (yield enabled)

`LLMRS_DECODE_YIELD_EVERY=4 LLMRS_DECODE_YIELD_US=500` 시:
- avg_tbt 40.376 ms (Δ = **+22.8% vs default**)
- 회귀 원인은 **yield 자체 비용** (synchronize + 500us sleep × 7회/token = ~3.5ms × 추가 sync cost)이지 trait dispatch cost 아님.
- 본 시나리오는 게이트 대상 아님 (yield는 manager-driven opt-in feature). 정량화만 기록.

## 출력 정합성

bit-identical: 15회 (stage2a_recheck 5 + stage2b 5 + stage2b_yield 5) 모두 동일 출력
> "The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into"

`[Phase4-4.5] generated=32 (first=12095 + run=31) stopped_by=BudgetExhausted final_pos=36` — 15회 모두 동일.

## 이상 신호

- stderr에서 `evict|dispatcher|pending|panic|abort|error|WARN` 라인 **0건** (전 15 runs)
- `[NoShuffle] Skipped: default AOS` 일관

## 결정

Stage 2-B (yield_after_layer hot-path 치환) **PASS**. R-1 (vtable overhead) 회귀 없음. B-5b Phase 2 다음 Stage (예정: as_opencl_secondary 또는 다른 capability hook) 진입 가능.

## 데이터

- `raw/stage2a_recheck_r{1..5}.txt` — Stage 2-A 재측정 5회
- `raw/stage2b_r{1..5}.txt` — Stage 2-B default 5회
- `raw/stage2b_yield_r{1..5}.txt` — Stage 2-B yield enabled 5회
