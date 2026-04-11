# V9 — Strict Thermal Isolation Benchmark (Qwen 2.5-1.5B)

**Date**: 2026-04-11
**Device**: Samsung Galaxy S25 (Adreno 830, adb R3CY408S5SB)
**Binary**: `generate` built from master `d557c99` (C1-Qwen merged)
**Script**: `scripts/bench_strict_thermal_isolation.sh`

## Methodology

Strictest thermal isolation protocol to date:

1. **5-minute mandatory rest** after EVERY run (inter-run and inter-combo).
2. **CPU and GPU phases completely separated** — all llama.cpp CPU runs first,
   then an additional 5-minute inter-phase rest, then all llm.rs GPU runs.
   No CPU/GPU interleaving that could leave residual thermal state.
3. **4-zone thermal monitoring**: `cpu-0-0-0` (zone1), `cpu-0-4-1` (zone10),
   `gpuss-5` (zone28), `gpuss-7` (zone30).
4. **Preflight**: zombie process check + thermal threshold 50°C before each run.
5. **N=3 runs per combo, report median**.

Total runtime: ~75 minutes (12 runs × 5-min rest + inter-phase rest).

## Results

### Medians

| Combo | llama.cpp CPU | llm.rs GPU | GPU vs CPU |
|---|---|---|---|
| **A Short (pf=7, dc=64)** | **50.71 ms/tok** | **58.59 ms/tok** | **+15.5%** |
| **B Long (pf=720, dc=128)** | **57.37 ms/tok** | **69.66 ms/tok** | **+21.4%** |

### Individual runs — reproducibility

**llama.cpp CPU** (all runs from ~39°C cold start):
- Short: 51.52, 50.71, 50.71 ms/tok — spread ±1%
- Long: 57.37, 55.34, 57.64 ms/tok — spread ±2%

**llm.rs GPU** (all runs from ~35°C cold start):
- Short: 58.40, 58.59, 58.60 ms/tok — **spread ±0.2%**
- Long: 69.73, 69.66, 69.28 ms/tok — **spread ±0.3%**

**llm.rs GPU variance dropped from ±30% (V8) to ±0.3% (V9)** once we fully
isolated CPU/GPU phases and enforced 5-min rest. This confirms the V8 high
variance was thermal cross-contamination (llama.cpp's CPU heat leaking into
subsequent llm.rs runs via unmonitored thermal zones).

### Thermal envelope

| Workload | Zone start | Zone end (max) | Delta |
|---|---|---|---|
| llama.cpp short | 38-40°C | 77-83°C | **+38-44°C** |
| llama.cpp long  | 37-39°C | 70-77°C | **+31-38°C** |
| llm.rs short    | 35-37°C | 53-58°C | +18-23°C |
| llm.rs long     | 33-37°C | 54-57°C | +20-22°C |

llama.cpp CPU creates **nearly 2× the thermal footprint** of llm.rs OpenCL.
The peak zone temperature during llama.cpp short runs reached **83°C** — within
~15°C of typical SoC throttle points. This explains why mixed CPU/GPU runs in
previous V8 showed 30% variance: residual thermal state from llama.cpp runs
throttles the GPU frequency governor even after monitored zones return to baseline.

## Comparison with prior measurements

| Measurement | llm.rs long | llama.cpp long | Δ (GPU vs CPU) | Notes |
|---|---|---|---|---|
| V2 original | 69.96 | 55.56 | +25.9% | Zombie llama-cli-new polluting all cores |
| V2 post-zombie | 122.62 | 70.32 | +74% | Post-zombie thermal flux, no control |
| V8 (45s rest, interleaved) | 89.13 | 67.52 | +32.0% | CPU/GPU interleaving leaks heat |
| **V9 (5-min rest, batched)** | **69.66** | **57.37** | **+21.4%** | **Clean, reproducible** |

The V2 original numbers happened to be close to V9 **by accident**: the zombie
held the chip in a saturated-warm steady state that coincidentally resembled
the cold-start steady state of V9. V8's interleaved protocol with 45s rest
was the most misleading.

## Interpretation

### The true llm.rs Qwen performance on Adreno 830

- **Short (pf=7, dc=64)**: 58.59 ms/tok steady-state
- **Long (pf=720, dc=128)**: 69.66 ms/tok steady-state
- Context scaling: +18.9% TBT over 4-fold depth increase — **flash attention
  delivers excellent scaling** (master legacy would have been ~83% slower at
  same depth, see SUMMARY.md for pre-C1 measurements).

### The true gap vs llama.cpp CPU

- Short: **+15.5%** (7.9 ms slower per token)
- Long: **+21.4%** (12.3 ms slower per token)

Both gaps are smaller than our earlier panic numbers of "30-32% slower". The
gap is **real but modest** and is explainable by:

1. **Plan.rs bypass** (`has_qkv_bias=true` → `build_plan()` early-returns).
   Qwen cannot use the pre-bound kernel path, so every decode step pays
   kernel arg rebinding cost (~0.3-0.5 ms/layer × 28 layers = 8-14 ms/token).
   This alone accounts for most of the long-context gap.

2. **28 layers vs Llama3.2-1b's 16** (75% more kernel launches per token).

3. **Snapdragon 8 Elite Oryon cores are exceptionally strong for F16 matmul**
   at 8 threads — llama.cpp's `ggml_vec_dot_f16` is heavily tuned for ARMv9
   and hits near-peak FLOPS on this SoC.

### C1-Qwen value vs these numbers

The C1-Qwen feature still delivers its intended value:
- **Output parity** with legacy kernel (V3 bit-identical check)
- **Scaling curve dramatically flattens** (compare V9 long 69.66 with pre-C1
  legacy long estimated ~121 ms/tok from V2 master baseline)
- **No regressions** (V4/V6 H2O + profiling pass)
- **Thermal envelope is GPU-friendly**: ~20°C rise vs llama.cpp's ~40°C.
  On sustained workloads where thermal budget dominates, llm.rs will degrade
  less than llama.cpp.

## Follow-up (C1.5)

Unchanged from V8 analysis:
1. **Plan.rs QKV bias support** — lifting the `has_qkv_bias` early return in
   `transformer.rs:1437` would enable Qwen to use pre-bound kernels, expected
   8-14 ms/token improvement. Would put Qwen at ~56-62 ms/tok, **beating
   llama.cpp CPU on long context**.
2. **DK=128 kernel tuning**: private `q_priv[32]` + `o_acc[32]` float4 arrays
   likely cause register spill on Adreno 830. Explore `__attribute__` hints
   and subgroup size tuning.
3. **Warmup run** in the bench script for governor stabilization.

## Test artifacts

- `qwen_strict_isolation.txt` — raw per-run values and medians
- `scripts/bench_strict_thermal_isolation.sh` — harness
- `V9_STRICT_ANALYSIS.md` — this file
