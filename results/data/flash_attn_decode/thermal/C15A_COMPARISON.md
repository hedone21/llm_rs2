# C1.5-a (Plan QKV Bias) vs V9 Baseline — Strict Thermal Isolation

**Date**: 2026-04-11
**Device**: Samsung Galaxy S25 (Adreno 830, adb R3CY408S5SB)
**Method**: `scripts/bench_strict_thermal_isolation.sh` (5-min rest between runs, CPU/GPU batched phases, 4-zone thermal monitoring)
**Branches compared**:
- V9 baseline: master `eb689d1` (C1-Qwen merged but has_qkv_bias early-return still active)
- C1.5-a: `feat/plan-qkv-bias` tip after Task 5

## Medians across 3 runs

| Combo | Metric | V9 baseline | C1.5-a | Delta |
|---|---|---|---|---|
| **Short** (pf=7, dc=64) | llm.rs GPU | **58.59** ms/tok | **58.11** ms/tok | **-0.8%** |
| Short | llama.cpp CPU | 50.71 ms/tok | 50.20 ms/tok | -1.0% |
| Short | GPU vs CPU | +15.5% | **+15.8%** | ~same |
| **Long** (pf=720, dc=128) | llm.rs GPU | **69.66** ms/tok | **69.19** ms/tok | **-0.7%** |
| Long | llama.cpp CPU | 57.37 ms/tok | 54.20 ms/tok | -5.5% |
| Long | GPU vs CPU | +21.4% | **+27.7%** | widened |

## Per-run values (C1.5-a)

### Phase 1 (llama.cpp CPU)
- Short: 50.20, 50.20, 50.45 ms/tok (±0.5%)
- Long: 54.59, 53.28, 54.20 ms/tok (±2.4%)

### Phase 2 (llm.rs OpenCL GPU, after plan.rs QKV bias gate lifted)
- Short: 58.19, 58.11, 58.07 ms/tok (±0.2%)
- Long: 68.81, 69.19, 69.32 ms/tok (±0.7%)

## Thermal envelope (unchanged from V9)

| Workload | Start | End (max) | Δ |
|---|---|---|---|
| llama.cpp short | 32-36°C | 75-81°C | +40-47°C |
| llama.cpp long | 32-34°C | 68-81°C | +35-49°C |
| llm.rs short | 31-34°C | 49-54°C | +17-21°C |
| llm.rs long | 31-32°C | 52-55°C | +20-23°C |

## Interpretation

**The plan.rs path IS active for Qwen after C1.5-a** — verified by Task 4 device log
showing `GPU kernel plan built (28 layers, capacity=512)`, and by the `c15a_d_qwen_gpu.txt`
regression test in Task 6.

**But the TBT improvement is minimal (-0.8% short, -0.7% long)**, nowhere near the
8-14 ms/token estimate in the original briefing.

### Why the estimate was wrong

The briefing assumed per-step `forward_into` entry cost was ~10 ms/token (from kernel
arg rebinding across 28 layers). A profile capture during Task 5 showed the actual
overhead is much smaller:
- matmul_qkv is only 9.5% of total decode time (408 μs/layer in profile mode)
- Profile sync adds ~54 ms/tok overhead, inflating all op times
- Steady-state arg rebinding cost is probably 10-50 μs per kernel launch
- 28 layers × 10-50 μs × 3 kernels = 0.84-4.2 ms/token savings maximum

The measured -0.5 ms/token delta is consistent with this revised estimate and is
within thermal noise bounds.

### Why didn't bias steps cause regression either?

V9 baseline ALREADY called `kernel_add_row_bias` per step via `forward_gen.rs`
(which invokes `backend.add_row_bias()` for each layer). So the 84 bias dispatches
per token existed on BOTH master and feat. C1.5-a only changes HOW they're
dispatched (plan pre-bound vs per-call), not WHETHER.

### What C1.5-a actually accomplishes

1. **Correctness**: Qwen now goes through the same plan.rs path as Llama3.2-1b,
   removing the `has_qkv_bias` code exception.
2. **Infrastructure**: Future Qwen optimizations (FFN fused kernels, Q4_0 weights,
   attention tuning) can now land inside the plan framework and benefit Qwen
   uniformly with other models.
3. **Thermal parity**: The llm.rs GPU variance remains ±0.2-0.7% (same as V9),
   confirming that the plan path has the same thermal stability as forward_into.
4. **No regressions**: All 6 regression tests pass (see `regression/c15a_regression_summary.md`).

### What C1.5-a does NOT accomplish

**Closing the gap to llama.cpp CPU on Qwen.** The gap remains at +15.8% (short)
and +27.7% (long). To actually beat llama.cpp on Qwen, different optimizations
are needed:
- **FFN matmul (44.9% of decode time)**: biggest lever. Consider fused gate+up
  matmul, higher-throughput GEMM variants, or Q4_0 weight quantization to reduce
  bandwidth.
- **Attention (12.0%)**: DK=128 kernel tuning, fewer barriers, or try fp16 accumulators.
- **Per-layer overhead compression**: coalesce multiple small ops.

## Raw data

- `qwen_strict_isolation_v9_baseline.txt` (master eb689d1)
- `c15a_qwen_strict_isolation.txt` (this branch)
- Bench script: `scripts/bench_strict_thermal_isolation.sh`
