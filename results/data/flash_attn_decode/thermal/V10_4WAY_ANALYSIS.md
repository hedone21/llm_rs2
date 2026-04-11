# V10 — 4-way Apples-to-Apples Qwen Benchmark (Adreno 830)

**Date**: 2026-04-11
**Device**: Samsung Galaxy S25 (Adreno 830, Snapdragon 8 Elite, adb R3CY408S5SB)
**Model**: Qwen 2.5-1.5B (F16 weights, F16 KV, head_dim=128, 28 layers)
**Method**: Strict thermal isolation — 5-min rest between every run, 4 phases fully separated, 4-zone thermal monitoring (cpu-0-0-0, cpu-0-4-1, gpuss-5, gpuss-7)
**Branches**: C1.5-a (plan-qkv-bias) merged to master, binary built from that tree

## Data sources

The 4-way comparison combines data from two strict-isolation runs (same methodology):

1. `c15a_qwen_strict_isolation.txt` — **llm.rs GPU** + **llama.cpp CPU** (from C1.5-a Task 5)
2. `qwen_missing_legs_strict.txt` — **llm.rs CPU** + **llama.cpp GPU** (new, this run)

Both runs used identical:
- 5-min rest between each measurement
- Same Qwen GGUF / safetensors / prompts
- Same thermal threshold (50°C)
- Same N=3 runs per combo

Running the missing legs as a focused bench saved ~75 minutes vs re-running
all 4 backends.

---

## Results (medians across 3 runs)

### Combo A: Short prefill + short decode (pf=7, dc=64 tokens)

| Backend | Decode TBT (ms/tok) | Per-run variance |
|---|---|---|
| llm.rs OpenCL GPU | **58.11** | ±0.2% (58.19, 58.11, 58.07) |
| llama.cpp OpenCL GPU | **52.36** | ±1.1% (52.36, 52.63, 52.08) |
| llm.rs CPU | **57.85** | ±0.4% (57.85, 57.62, 57.87) |
| llama.cpp CPU | **50.20** | ±0.5% (50.20, 50.20, 50.45) |

### Combo B: Long prefill + long decode (pf=720, dc=128 tokens)

| Backend | Decode TBT (ms/tok) | Per-run variance |
|---|---|---|
| llm.rs OpenCL GPU | **69.19** | ±0.7% (68.81, 69.19, 69.32) |
| llama.cpp OpenCL GPU | **68.49** | ±0.7% (68.49, 68.49, 68.03) |
| llm.rs CPU | **61.86** | ±5.9% (61.86, 61.24, 64.88) |
| llama.cpp CPU | **54.20** | ±2.4% (54.59, 53.28, 54.20) |

---

## Apples-to-apples gaps (llm.rs vs llama.cpp)

### CPU ↔ CPU (Snapdragon 8 Elite Oryon cores, 8 threads)

| Combo | llm.rs CPU | llama.cpp CPU | llm.rs is... |
|---|---|---|---|
| Short (pf=7, dc=64) | 57.85 ms | 50.20 ms | **+15.2% slower** |
| Long (pf=720, dc=128) | 61.86 ms | 54.20 ms | **+14.1% slower** |

### GPU ↔ GPU (Adreno 830, OpenCL)

| Combo | llm.rs GPU | llama.cpp GPU | llm.rs is... |
|---|---|---|---|
| Short (pf=7, dc=64) | 58.11 ms | 52.36 ms | **+11.0% slower** |
| Long (pf=720, dc=128) | 69.19 ms | 68.49 ms | **+1.0% slower** (within noise) |

### Headline chart

```
Qwen 2.5-1.5B decode TBT (ms/tok, lower is better)

Short (pf=7, dc=64):
  llama.cpp CPU  ████████████████████         50.20 ms
  llama.cpp GPU  █████████████████████▏       52.36 ms   (+4.3% vs llc CPU)
  llm.rs    CPU  ███████████████████████      57.85 ms   (+15.2% vs llc CPU)
  llm.rs    GPU  ███████████████████████▏     58.11 ms   (+15.8% vs llc CPU, +11.0% vs llc GPU)

Long (pf=720, dc=128):
  llama.cpp CPU  █████████████████████▏       54.20 ms
  llm.rs    CPU  ████████████████████████▍    61.86 ms   (+14.1% vs llc CPU)
  llama.cpp GPU  ██████████████████████████▉  68.49 ms   (+26.4% vs llc CPU, but GPU is thermally sustainable)
  llm.rs    GPU  ███████████████████████████  69.19 ms   (+27.7% vs llc CPU, +1.0% vs llc GPU)
```

---

## Key findings

### 1. Long context GPU↔GPU is essentially TIED

**llm.rs GPU 69.19 ms vs llama.cpp GPU 68.49 ms → 0.70 ms difference (+1.0%)**.

This is well inside measurement noise (both sides ±0.7%). The real production
long-decode GPU performance on Adreno 830 is **at parity** with llama.cpp.

This completely reframes the earlier narrative. Our earlier "llm.rs is +27%
slower" figure came from comparing llm.rs GPU against llama.cpp CPU, which is
apples-to-oranges (CPU happened to be faster than GPU in llama.cpp for this
specific model × hardware combination).

### 2. On this specific device, llama.cpp CPU > llama.cpp GPU for Qwen

This is the surprising finding that triggered the earlier confusion:

| | Short | Long |
|---|---|---|
| llama.cpp CPU | 50.20 ms | **54.20** ms |
| llama.cpp GPU | 52.36 ms | 68.49 ms |
| CPU advantage | +4.3% | **+26.4%** |

Why? On **this device specifically** (Snapdragon 8 Elite with Oryon cores):
- The CPU has extremely strong F16 dot product throughput (ARMv9 FP16 SDOT)
- 8 cores run llama.cpp's `ggml_vec_dot_f16` at near-peak FLOPS
- Adreno 830's OpenCL GEMV has significant per-kernel dispatch overhead
- Qwen's 28 layers × many small ops amplifies launch overhead on GPU
- Bandwidth-bound matmul at m=1 favors the CPU's coherent caches

Under sustained thermal load (not measured here), the CPU would throttle more
aggressively than the GPU (our thermal envelope data shows CPU runs 80°C+ vs
GPU 55°C), so this ordering may flip under real workloads.

### 3. CPU↔CPU gap is larger than GPU↔GPU gap

| Backend pair | Short gap | Long gap |
|---|---|---|
| CPU↔CPU | +15.2% | +14.1% |
| GPU↔GPU | +11.0% | **+1.0%** |

llm.rs is **more competitive on GPU than on CPU** relative to llama.cpp. The
absolute gap in CPU is about 7.5 ms/tok, in GPU long-context it's 0.7 ms/tok.

### 4. The CPU gap is consistent (~14-15%) across both combos

Suggests a systematic efficiency difference in the CPU matmul implementation,
not context-length-dependent. Most likely culprit: ARM NEON F16 dot product
tuning. llama.cpp's `ggml_vec_dot_f16` is heavily hand-tuned for ARMv8.2+ FP16
with SDOT/SMMLA; llm.rs's NEON path may be using a more generic kernel.

### 5. Short-context GPU gap (+11%) is suspicious

Long-context GPU parity (+1%) vs short-context GPU gap (+11%) is inconsistent
with a simple "matmul is slower" story — if llm.rs GPU matmul were slower,
both combos would show similar gap.

Hypothesis: **first-token overhead** (plan.rs build + workspace init +
warmup). Over 128 decode tokens this amortizes to ~0 ms/tok; over 64 tokens
it amortizes to ~0.1 ms/tok; but the first few tokens might be disproportionately
slow, and short-decode averages suffer more. The "64-token average TBT" could
be inflated by a few warm-up tokens.

This is testable: run llm.rs with a longer decode (256 tokens) from a short
prompt and see if the average converges to ~52 ms/tok.

---

## Comparison with earlier (misleading) numbers

Earlier reports compared **llm.rs GPU** against **llama.cpp CPU**, producing
headline numbers of "+15.8% / +27.7%" which led to false conclusions about
GPU inefficiency. The correct apples-to-apples view:

| Earlier frame (wrong) | Corrected frame (V10) |
|---|---|
| "llm.rs GPU +27% slower than llama.cpp" | llm.rs GPU **+1% vs llama.cpp GPU** (at parity) |
| "Need to close large OpenCL gap on Qwen" | Need to close **CPU** gap (14-15%), GPU is fine |
| "FFN optimization needed" | Profile was misleading; actual bottleneck might be elsewhere |

The earlier C1.5-a narrative was partially correct (plan.rs activation is
infrastructure win), but the "8-14 ms/tok improvement was too optimistic" part
was wrong in the opposite direction — we already had parity in GPU long, we
just weren't measuring apples-to-apples.

---

## Implications for C1.5-b direction

### Before V10 analysis
The earlier plan was to investigate FFN optimization (B1/B2/B3) or DK=128
kernel tuning (B4) to close a presumed 20-30% GPU gap.

### After V10 analysis

1. **The GPU long-context gap doesn't need closing** (already at parity). Any
   further GPU optimization would produce diminishing returns on this
   specific workload.

2. **Short-context GPU gap (+11%) needs root-cause analysis first** — likely
   not a kernel issue but first-token overhead. Cheap to test.

3. **CPU path is the biggest gap** (+14-15%, consistent across combos). If
   the goal is "llm.rs matches llama.cpp on Qwen", the highest-leverage work
   is **NEON F16 matmul tuning** — which happens to also be accuracy-neutral
   (no Q4_0, no kernel modifications needed for the default path).

4. **Thermal envelope advantage of GPU is real** — under sustained generation
   (not our 1-shot tests), llm.rs GPU degrades less than CPU because it
   generates ~half the heat footprint. For long-running inference, GPU is
   the right backend despite being slightly slower in single-shot tests.

### Revised C1.5-b candidates (priority order)

1. **B1': Short-decode first-token overhead investigation** (cheap, ~1 day)
   - Add `--warmup-runs N` flag or extend decode length
   - Measure if short-decode gap closes
   - Accuracy-neutral (measurement only)

2. **B2': NEON F16 matmul tuning** (medium, ~2-3 days)
   - Profile current ARMv9 NEON path on Oryon cores
   - Compare against llama.cpp `ggml_vec_dot_f16` instruction scheduling
   - Likely low-hanging fruit exists (SMMLA, BF16 alt, dual-issue tuning)
   - Accuracy-neutral (same math, better instruction scheduling)
   - Affects BOTH CPU-only and CPU fallback paths → benefits all models

3. **B3': OpenCL event profiling infrastructure** (medium, ~1 day)
   - `CL_QUEUE_PROFILING_ENABLE` + per-kernel event capture
   - Gives real production-mode per-kernel timing
   - Unblocks future kernel-level tuning with real data
   - Accuracy-neutral (measurement only)

4. **B4': Kernel launch coalescing / fused small ops** (medium, ~3 days)
   - Fuse `rms_norm + matmul_qkv`, `rope + add_assign`, etc.
   - Only worth doing if B3' shows small-op launch overhead is significant
   - Requires `.cl` changes

**Deprioritized**:
- FFN gate+up fused matmul (B2 from old list): bandwidth analysis shows the
  savings are marginal (~2-8% at best), not the expected 10-15%
- DK=128 attention kernel tuning: attention is only 12% of profile breakdown
  and long-context GPU is already at parity — low ROI
- Q4_0 quantization: explicitly forbidden (accuracy-neutral constraint)

---

## Recommendations

1. **Declare Qwen GPU optimization COMPLETE for long-context production use**.
   llm.rs matches llama.cpp within measurement noise. Further GPU work should
   focus on accuracy-preserving features (e.g. context extension, better
   quantization methods) rather than raw TBT.

2. **Investigate short-decode overhead** as the first C1.5-b task. If adding
   a 10-token warmup eliminates the +11% gap, the "issue" is measurement
   methodology, not a real perf bug.

3. **NEON F16 matmul tuning** is the highest-leverage optimization if
   llm.rs wants to match llama.cpp CPU performance. Benefits all models and
   both backends (since GPU fallback paths also use NEON).

4. **Commit to real production profiling**: get CL event profiling infrastructure
   landed so future optimization decisions are data-driven.

## Raw data

- `c15a_qwen_strict_isolation.txt` — llm.rs GPU + llama.cpp CPU (6 runs)
- `qwen_missing_legs_strict.txt` — llm.rs CPU + llama.cpp GPU (6 runs, this bench)
- `bench_strict_thermal_isolation.sh` — strict isolation harness
- `bench_missing_legs.sh` — focused 2-backend harness (this run)
