# Thermal-Controlled Qwen Benchmark Analysis

**Date**: 2026-04-11
**Device**: Samsung Galaxy S25 (Adreno 830, adb R3CY408S5SB)
**Binary**: `generate` built from master HEAD `d557c99` (merged C1-Qwen)
**Script**: `scripts/bench_thermal_controlled.sh`

## Motivation

The initial V2 4-context bench (`scripts/bench_flash_attn_decode_qwen.sh`) showed
llm.rs OpenCL ~20-30% slower than llama.cpp CPU on Qwen. Two concerns led to
re-measurement:

1. **Zombie process contamination**: a stale `llama-cli-new` (pid 8764) from an
   earlier V3 accuracy experiment (`adb shell timeout 90 ./llama-cli-new ...`)
   survived the adb timeout. SIGTERM from `timeout` was not propagated by adb
   shell, leaving the process running at ~100% CPU for ~30 minutes and affecting
   all subsequent measurements.
2. **Thermal drift**: fixed 30s cooldowns between runs may not equalize thermal
   state because llama.cpp (8-thread CPU saturation) heats the chip much more
   than llm.rs OpenCL (mostly GPU load).

## Methodology

- Kill any stale `llama-cli*` / `generate_master` processes on device (preflight)
- Wait until `max(cpu-0-0-0, gpuss-5) < 55°C` before each run
- Hard floor: `COOLDOWN_MIN_SEC=45`, hard ceiling: `COOLDOWN_MAX_SEC=180`
- Alternate starting side per iteration (`llm_first` / `llc_first`) to cancel bias
- 3 runs per combo, report median
- Record start and end temperatures for each run to validate thermal parity

### Test combos

| Combo | Prefill | Decode | llm.rs invocation | llama.cpp invocation |
|---|---|---|---|---|
| A Short | 7 tokens | 64 tokens | `--prompt "The quick brown fox jumps over" -n 64` | `llama-bench -p 0 -n 64 -d 7 -r 1 -t 8` |
| B Long | 720 tokens | 128 tokens | `--prompt-file long_prompt.txt -n 128` | `llama-bench -p 0 -n 128 -d 720 -r 1 -t 8` |

## Results

### Medians (3 runs each)

| Combo | llm.rs OpenCL | llama.cpp CPU | llm.rs vs llama.cpp |
|---|---|---|---|
| A Short (pf=7, dc=64) | **58.02 ms/tok** | **52.44 ms/tok** | **+10.6%** |
| B Long (pf=720, dc=128) | **89.13 ms/tok** | **67.52 ms/tok** | **+32.0%** |

### Per-run values and thermal footprint

| Run | Order | llm.rs short (ms) | llama.cpp short (ms) | llm.rs long (ms) | llama.cpp long (ms) |
|---|---|---|---|---|---|
| 1 | llm→llc | 62.82 | 52.66 | 89.13 | 70.03 |
| 2 | llc→llm | 57.83 | 52.44 | **110.80** ⚠️ | 61.20 |
| 3 | llm→llc | 58.02 | 50.20 | 69.46 | 67.52 |

### Thermal envelope per workload

| Workload | Start range | End range | Swing |
|---|---|---|---|
| llm.rs short | 44-46°C | 59-63°C | +14-18°C |
| llama.cpp short | 44-47°C | 78-82°C | **+33-37°C** |
| llm.rs long | 44-46°C | 54-63°C | +9-17°C |
| llama.cpp long | 44-46°C | 64-68°C | +19-23°C |

llama.cpp's 8-thread CPU saturation produces a 2-3× larger thermal footprint
than llm.rs's OpenCL GPU workload. On short runs, llama.cpp pushes the monitored
CPU core to 78-82°C (within 18°C of the typical throttle point).

### Variance analysis

- llama.cpp long: 61.20, 67.52, 70.03 ms/tok — ±7% spread
- llm.rs long: 69.46, 89.13, **110.80** ms/tok — **±30% spread**

llm.rs has substantially higher run-to-run variance. Run 2 of the long combo
(where llama.cpp ran FIRST and heated the chip to 68°C before the llm.rs
measurement) was **60% slower than run 3** (where llm.rs was the first run of
the long combo at a 45°C start). Both runs had the monitored CPU + GPU zones
cooled back to 45-46°C before the llm.rs run, so the slowdown is NOT captured
by our two thermal probes.

Hypothesis: **Adreno 830 frequency scaling is sensitive to thermal zones we
don't monitor** (e.g. GPU sub-cluster zones 25-27-29, memory controller, skin
temp). A preceding llama.cpp burst leaves the device in a subtly throttled
state that reduces GPU perf for the next ~45-60s even though our two sampled
zones read cool.

## Comparison with pre-thermal-controlled V2

Zombie-contaminated V2 numbers (qwen_after_c1.txt) are NOT comparable because
the stale llama-cli-new held a core at 100% for the entire measurement window,
which:
- Reduced total power budget available to both llm.rs and llama.cpp
- Held the whole SoC in a "warm" steady state where frequency governors had
  settled at mid-range targets
- Actually produced MORE STABLE numbers (both llm.rs and llama.cpp got consistent
  but artificially-paced values)

Post-zombie V2 re-run (same script, no zombie) showed llm.rs at 86-137 ms/tok —
wildly different from the 61-70 ms/tok in the contaminated run. This confirms
thermal/contention state dominates the measurement on Qwen decode.

## Interpretation

**C1-Qwen still provides real value**: the flash attention DK=128 dispatch
replaces the legacy 3-pass attention scan and delivers dramatically flatter
scaling with context (see `regression/SUMMARY.md` — master 66→121 ms/tok,
feat 61→70 ms/tok under contaminated conditions; the pattern holds).

**But llm.rs OpenCL is still slower than llama.cpp CPU on Qwen**:
- Short context: +10%
- Long context: +17-32% (depending on thermal history)

**Root causes** (unchanged from pre-merge analysis):
1. **Plan.rs bypass**: Qwen has `has_qkv_bias=true` which short-circuits
   `build_plan()` in `transformer.rs:1437`. Every decode step goes through
   `forward_into` with fresh arg rebinding — pre-bound kernel benefits lost.
2. **Layer count**: 28 layers vs Llama3.2-1b's 16 → 75% more kernel launches
   per decode token, so per-layer overhead dominates.
3. **head_dim=128 register pressure**: 2× the private array size of DK=64.
   Adreno may reduce wave count at DK=128, losing latency hiding.
4. **Snapdragon 8 Elite CPU is extremely strong**: 8-thread F16 matmul on
   Oryon cores is highly optimized in llama.cpp's ggml_vec_dot_f16.

## Conclusion

- **Merge was correct**: C1-Qwen provides a measurable improvement over master
  legacy (flatter scaling, validated accuracy, no regressions) and is safe.
- **Llama.cpp parity on Qwen is NOT achieved**: gap is 10-32% depending on
  context length. Root causes are architectural (plan.rs bypass + layer count)
  and require separate follow-up work.
- **Qwen thermal sensitivity is a real issue**: runs preceded by CPU-heavy
  workloads show 30-60% slowdown in llm.rs GPU runs despite apparent thermal
  recovery at the monitored zones. This suggests that Qwen long-decode bench
  results should always be interpreted with thermal context in mind.

## Follow-up candidates (C1.5)

1. **Plan.rs QKV bias support** to enable the pre-bound kernel path for Qwen
   (expected 10-20% improvement from eliminating per-step arg rebinding).
2. **DK=128 kernel tuning** (reduce register pressure, try
   `__attribute__((reqd_work_group_size))` + `qcom_reqd_sub_group_size` tuning).
3. **Multi-zone thermal monitoring**: add gpuss-2..7 average + skin temp + battery
   to the bench script to catch cross-zone throttling.
4. **Warmup run** before measurements to reach stable frequency governor state.

## Raw data

- `qwen_thermal_controlled.txt` — per-run values and medians
- `scripts/bench_thermal_controlled.sh` — the harness used
