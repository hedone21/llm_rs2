# OoO Queue + Skip-Finish Ablation (Galaxy S25, Adreno 830)

**Date**: 2026-05-08
**Device**: Galaxy S25 (R3CY408S4HN), Adreno 830, GPU mem 5557 MB, max alloc 1024 MB
**Build**: `target/aarch64-linux-android/release/generate` (HEAD `0c8951e`)
**Model**: Qwen2.5-1.5B-Instruct (primary F16 GGUF + secondary Q4_0 GGUF, AOS layout)
**Settings**: `--force-swap-ratio 0.9` (25/28 layers), n_tokens=200, threads=6, prompt="The quick brown fox jumps", n=3 per scenario

## Hypothesis
이전 op-level profiling에서 incremental swap이 forward 사이에 +22 ms inflation을 만들었다 ("op 사이 gap" / driver queue scheduling). In-order queue + `clFinish`가 swap H2D를 forward kernel과 직렬화한다는 가설.

**OoO queue + skip-finish** 시:
- swap commands(cl_mem A에 write) + forward kernels(cl_mem B에 read) → data dependency 없음 → driver가 병렬 dispatch 가능 (이론적)
- driver가 진짜 honor하면: forward TBT가 sync_baseline 수준 회복 → 6중 negative 깨질 가능성

## Setup

| 항목 | 값 |
|------|----|
| `LLMRS_OPENCL_OOO_QUEUE=1` | main + transfer queue 모두 `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` |
| `LLMRS_HOST_PTR_SKIP_FINISH=1` | `fill_host_ptr_buffer` 후 `clFinish` 제거 |
| `LLMRS_OPENCL_HOST_PTR_POOL=1` | LISWAP-3 zero-copy pool (14 slots) |

## Correctness

전체 21/21 run에서 "the lazy dog" sequence 정상 생성 (token output이 IncrementalSwap log line과 interleaved되지만 token stream 자체는 정상).

OoO queue 활성 로그 검증: 시나리오 1·3 = 0, 시나리오 2·4·5·6·7 = 1 (각 run 1번씩 출력). env-gate 정상 작동.

## Results (mean ± stdev, n=3)

| # | 시나리오 | OoO | SkipFinish | ZeroCopy | Avg TBT (ms) | Decode excl0 (ms/tok) | TTFT (ms) |
|---|----------|-----|------------|----------|--------------|----------------------|-----------|
| 1 | baseline_inorder    |   | | | **31.55 ± 0.53** | 26.62 ± 0.05 | 583.67 ± 63.03 |
| 2 | baseline_ooo        | ✓ | | | 31.90 ± 0.13 | 26.65 ± 0.01 | 590.32 ± 9.72 |
| 3 | liswap1_inorder     |   | | | **38.56 ± 1.24** | 28.33 ± 0.37 | 349.40 ± 7.64 |
| 4 | liswap1_ooo         | ✓ | | | 39.41 ± 0.45 | 28.29 ± 0.50 | 403.53 ± 70.91 |
| 5 | liswap1_ooo_skipfin | ✓ | ✓ | | 41.83 ± 6.35 | 31.36 ± 5.87 | 353.23 ± 7.86 |
| 6 | liswap1+3_ooo       | ✓ | | ✓ | 55.09 ± 10.27 | 43.82 ± 10.14 | 535.20 ± 3.33 |
| 7 | liswap1+3_ooo_skfin | ✓ | ✓ | ✓ | 54.56 ± 7.80 | 43.17 ± 8.00 | 532.58 ± 18.46 |

### Per-tick incremental swap latency (LISWAP-1 scenarios, ticks 0..24 across 3 runs)

| 시나리오 | mean tick lat (ms) | sum 25 ticks (ms) | n |
|----------|-------------------|-------------------|---|
| 03 liswap1_inorder         | **43.50 ± 10.32** | 1073.1 | 74 |
| 04 liswap1_ooo             | 49.56 ± 8.28      | 1222.6 | 74 |
| 05 liswap1_ooo_skipfinish  | 44.36 ± 8.36      | 1094.3 | 74 |
| 06 liswap1+3_ooo           | 49.68 ± 8.61      | 1242.0 | 75 |
| 07 liswap1+3_ooo_skipfinish| 50.24 ± 8.80      | 1256.1 | 75 |

## Verdict: 6중 negative 유지 (8중 negative로 확장)

### 핵심 비교

| 비교 | Δ Avg TBT | 결론 |
|------|-----------|------|
| **2 vs 1** (OoO no-swap)             | +0.35 ms (+1.1%) | OoO 자체는 baseline에 **무해 무익** (within noise) |
| **4 vs 3** (OoO + LISWAP-1)          | +0.85 ms (+2.2%) | OoO가 swap-active forward를 가속하지 않음. 오히려 살짝 악화 |
| **5 vs 4** (skipfinish 추가)         | +2.42 ms (+6.1%) | clFinish 제거가 도움 안 됨. 오히려 악화 (skipfinish run2는 49.12 ms, 큰 분산) |
| **5 vs 3** (가장 aggressive single)  | +3.27 ms (+8.5%) | OoO + skipfinish 조합 = **regression** |
| **6 vs 4** (LISWAP-3 zero-copy)      | +15.68 ms (+39.8%) | zero-copy pool 추가로 추가 악화 |
| **7 vs 6** (skipfinish + zero-copy)  | -0.53 ms (-1.0%) | within noise |

### Per-tick latency도 동일하게 OoO/skipfinish 추가 시 +6 ms (+13.9%) 악화

03 (in-order, sync) → 04 (OoO) → 05 (OoO+skipfin) tick latency가 43.5 → 49.6 → 44.4 ms로 변동. 06/07도 49.7~50.2 ms로 in-order 03 대비 +14% 증가. **OoO도 driver-side scheduling cost를 덜어주지 못함**.

## 해석

1. **Adreno 830 driver는 OoO queue를 honor하지 않거나, dependency tracking이 conservative하다.**
   - 코드 활성화 로그(`OpenCL queue: CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE`)는 driver가 flag를 받아들였다는 것만 보장 — 실제 reorder 여부와 별개.
   - swap H2D commands와 forward kernels가 **다른 cl_mem**을 만져도 wall-clock TBT가 in-order 대비 동일~악화.
   - 가능한 원인:
     - Adreno driver가 in-order semantics를 강제 (OoO mode를 "추후 지원"으로 표시만)
     - Implicit dependency tracker가 swap H2D와 forward를 보수적으로 직렬화 (cl_mem 단위가 아닌 queue 단위)
     - command queue dispatch overhead가 OoO에서 더 큼

2. **`LLMRS_HOST_PTR_SKIP_FINISH=1`가 도움 안 되는 이유**:
   - `clFinish` 제거 후 driver가 implicit barrier를 추가하거나, host-side mmap_permute가 GPU upload를 따라잡지 못해 stall 발생.
   - run-to-run 분산(±6.35 ms)이 in-order(±1.24 ms) 대비 5배로 커진 점이 race-condition-like 거동을 시사.

3. **LISWAP-3 zero-copy pool이 OoO와 결합 시 추가 악화 (+15.68 ms)**:
   - 기존 LISWAP-3 단독 측정(메모리 메모리에 따르면 negative)에 OoO를 더해도 회복 안 됨. 두 우회 트릭이 서로 cancel하거나 driver가 둘 다 무시.
   - 분산이 ±10.27/±7.80 ms로 매우 큼 — 안정성 자체가 무너짐.

## 6중 negative 깨짐 여부

| 우회 트릭 후보 | 결과 |
|---------------|------|
| OoO queue 단독 (#4)         | negative (+0.85 ms vs #3) |
| OoO + skipfinish (#5)       | negative (+3.27 ms vs #3) |
| OoO + zero-copy (#6)        | negative (+16.5 ms vs #3) |
| OoO + skipfinish + zero-copy (#7) | negative (+15.99 ms vs #3) |

**결론: 가설(OoO queue가 6중 negative를 깬다)은 기각.** Adreno 830에서 in-order queue + `clFinish`가 forward TBT의 inflation 원인이 아니거나, OoO queue가 실제로 reorder하지 않는다. 두 가지 모두 동일한 결론에 도달: **driver-level command scheduling은 우회 가능한 bottleneck이 아니다**.

## 로그 산출물

`papers/eurosys2027/_workspace/experiment/swap_overhead_ooo_queue_raw/`
- `01..07_*_run1.log ~ run3.log` (21개)
- 각 로그: full INFO log + per-tick swap latency stages + decode TBT

## 다음 단계 제안

1. **OoO queue를 production에서 비활성** — 가시적 이득 없고, 분산만 증가.
2. **upload bandwidth 자체에 손대는 트랙으로 전환** — H2D 99.9% upload-bound 결과(LISWAP-2/Direction A 메모리)와 일관. 다음 후보:
   - **LISWAP-4** (intra-forward layer-aligned swap) — `0c8951e` 커밋 확인, 진행 중
   - **swap data compression** (GGUF Q4_0 → 더 작은 형식, e.g., 2-bit/sparse) — upload 대역폭 자체를 줄임
   - **pre-stage during prefill** (이미 `pre_stage_v2_bg.md`에서 부분 검증) — swap을 critical path 밖으로
3. **OoO queue가 실제 reorder하는지 GPU profiler로 검증 (Snapdragon Profiler)** — 우리 wall-clock 측정만으로는 "driver가 honor 안 함"이 100% 단정은 아님. 그러나 5중/8중 negative이므로 ROI는 낮음.

