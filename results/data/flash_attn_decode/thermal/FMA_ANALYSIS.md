# FMA — Native F16 FMA GEMV Kernel for Qwen CPU Decode

**Date**: 2026-04-11
**Device**: Samsung Galaxy S25 (Adreno 830, Snapdragon 8 Elite / Oryon)
**Binary**: `generate` built from master `a9cd3cc` (native F16 FMA GEMV merged)
**Script**: custom V10-style 5-min rest protocol (`/tmp/bench_fma_cpu.sh`)
**Comparison baseline**: V10 `qwen_missing_legs_strict.txt` (FMLAL kernel, master `e2ca21e`)

## Motivation

V10 4-way comparison showed llm.rs CPU decode **+14-15% slower** than llama.cpp CPU on
Qwen 2.5-1.5B. Per-op profile narrowed the gap to `matmul_ffn` (75.7% short / 69.4% long
of decode TBT). Roofline analysis pinned the kernel as memory-bound (AI=1 flop/byte vs
Oryon balance ~4.7), with llm.rs at ~79% bandwidth efficiency vs llama.cpp ~97%.

Instruction-level diff isolated the likely cause: llm.rs used `FMLAL + FMLAL2` (widening
F16→F32 MAC, 2 instructions per 16 flops), while llama.cpp's `ggml_vec_dot_f16` uses
`vfmaq_f16` / `FMLA Vd.8H` (native F16 FMA, 1 instruction per 16 flops). Same flops,
**half the instruction count**, expected to unblock the memory-bound regime by letting
the HW prefetcher stream further ahead of the load front.

Prefill already used native FMA (commit `da4652a`, "match llama.cpp"). Only the decode
M=1 GEMV path remained on FMLAL (`c65933d` left it explicitly: "CPU decode: 44.0ms
unchanged (M=1 uses FMLAL GEMV path)"). This work lifts that.

## Scope

Three kernels replaced in `engine/src/backend/cpu/neon.rs`:

- `vec_dot_fmlal_4rows` → `vec_dot_f16_native_gemv_4rows` (NR=4, MR=1, K step 32, 16
  f16x8 partial accumulators)
- `vec_dot_fmlal` → `vec_dot_f16_native_gemv_1row` (fallback for j tail)
- `vec_dot_fmlal_2x4rows` (unused dead code from pre-`da4652a` MR=2 experiment) — removed

Dispatch wrappers (`f16_gemv_chunk`, `fused_gemv_chunk`, `f16_gemm_chunk` tail paths)
updated to call the new kernels. Tensor Partition CPU slice path
(`cpu.matmul_transposed` in `forward_gen.rs:920`) benefits automatically since it flows
through the same `matmul_transposed_f16` → `f16_gemv_chunk` chain.

Net diff: **+577 / -752 (-175 lines)**. Three commits:
```
f92c19e perf(neon): add native F16 FMA GEMV kernels for decode path
571b4c5 refactor(neon): switch F16 decode GEMV dispatch to native FMA
a9cd3cc refactor(neon): remove legacy FMLAL GEMV kernels
```

## Methodology

V10 strict thermal isolation protocol:
- 5-minute mandatory rest before every run
- 4-zone thermal pre-check (zone1 cpu-0-0-0, zone10 cpu-0-4-1, zone28 gpuss-5, zone30 gpuss-7)
- N=3 runs per combo, interleaved short/long
- Same prompts and token counts as V10 baseline for direct comparison
- Default sampling (temperature=0.8, top_p=0.9, top_k=40) — same as V10

## Results

### Medians (N=3)

| Combo | FMLAL V10 baseline | FMA (new) | Δ median | Δ mean |
|---|---|---|---|---|
| **Short** (pf=7, dc=64) | 57.85 ms/tok | **57.34 ms/tok** | -0.51 ms (-0.9%) | -0.33 ms (-0.6%) |
| **Long** (pf=720, dc=128) | 61.86 ms/tok | **64.35 ms/tok** | +2.49 ms (+4.0%) | **+0.19 ms (+0.3%)** |

### Individual runs

**FMA short** (all from 32-38°C cold start):
- Run 1: 57.16 ms/tok (63 tokens)
- Run 2: 57.34 ms/tok (63 tokens)
- Run 3: 57.85 ms/tok (63 tokens)

**FMA long** (all from 32-36°C cold start):
- Run 1: **64.35** ms/tok (51 tokens — early EOS)
- Run 2: 59.86 ms/tok (127 tokens)
- Run 3: **64.36** ms/tok (127 tokens)

**FMLAL V10 long baseline** (for variance comparison):
- Runs: 61.86, 61.24, 64.88 ms/tok — already ±5.8% spread, bi-modal

### Variance analysis

Long decode shows a **bi-modal distribution** in both FMLAL and FMA runs:
- "Fast mode": ~60 ms/tok
- "Slow mode": ~64-65 ms/tok

Pre-run thermal (32-36°C) and decode token counts do not correlate with the mode
assignment. This pre-exists the FMA change and is a separate measurement-environment
issue. Mean is the more robust aggregate here: **+0.19 ms (+0.3%) — within noise**.

### Quality verification (deterministic decode)

Both FMLAL and FMA binaries run with `--greedy` (argmax sampling, deterministic):

- **Short (33 tokens)**: byte-level identical output ✓
  - Input: "The quick brown fox jumps over"
  - Output: "the lazy dog.\ndef is_palindrome(string):\n    return string == string[::-1]\n\nprint(is_palindrome(\"quick brown fox jumps over the lazy dog.\"))"

- **Long (37 tokens continuation of `long_prompt.txt`)**: byte-level identical ✓
  - Output: "What is probably true about Henrik?\nA) He was an old man who had lived for many years\nB) He was very young\nC) None of the above statements are correct"

F16 accumulator overflow risk — worst-case pessimistic bound for Qwen down_proj (K=8960,
32 partial lanes, K_eff=280, |x|≤10, |w|≤2) = 5,600 ≪ 65,504 → 12× safety margin. Empirical
greedy-decode equivalence confirms no observed precision loss at this model size.

## Interpretation

### Why did the native FMA transition not move the needle?

Expected: −3 to −7 ms/tok per decode step (bringing llm.rs within 2-5% of llama.cpp).
Observed: **~0 ms/tok** (mean Δ within noise). Three converging explanations:

1. **FMLAL was already well-optimized.** Commit `c65933d` note: instruction count per
   inner iter dropped 44→26 (−41%) when replacing fcvtl/vfmaq_f32. The remaining
   FMLAL→FMA transition is smaller in relative terms than the earlier fcvtl→FMLAL step.

2. **Load-to-use stall persists in the new kernel.** Disassembly of the main loop
   (`0x369ee8`–`0x369f74`) shows 4 row-FMA groups each with 2-instruction distance
   between a B-row load and the first FMA using it:
   ```asm
   fmla v20.8h, v28.8h, v24.8h    ; row 0 last FMA
   ldp  q25, q24, [x12]            ; row 1 B load
   add  x10, x10, #0x40            ; pointer bump
   ldp  q27, q26, [x12, #-0x20]    ; row 1 B load
   fmla v19.8h, v31.8h, v27.8h    ; row 1 first FMA — 2 instructions after load
   ```
   Oryon vector load latency is ~5 cycles. A 2-instruction gap cannot hide it, so each
   B-row transition costs stall cycles. Inline `asm!` blocks are black boxes to the
   compiler, preventing instruction reordering across row transitions. This affects
   both FMLAL and FMA kernels identically, so the instruction-count reduction cannot
   show its benefit while this stall dominates.

3. **Bandwidth ceiling may already be near saturation.** Past llm.rs experiments
   (commit `b25bc19` on Snapdragon 8 Gen 3 S24) explicitly observed: "inner loop
   optimizations (multi-row, prefetch, stride) have no effect because the bottleneck
   is DRAM bandwidth utilization (58% of peak), not compute throughput." The move
   from 58% → ~79% between S24 and S25 was hardware, not software. We may be near a
   software-side ceiling that requires a different axis of attack.

### Is the change net-positive?

- **Performance target**: ❌ failed — gap to llama.cpp unchanged (~14-15%)
- **Code quality**: ✅ net cleanup (-175 lines, dead code removed)
- **Quality**: ✅ byte-identical greedy output
- **Regression risk**: ✅ mean +0.19 ms ≈ 0, no practical regression
- **Instruction-stream alignment with llama.cpp**: ✅ achieved (foundation for
  further work that needs like-for-like baseline)
- **Unblocks follow-up work**: ✅ load-to-use stall root cause now isolated by disassembly

**Decision**: kept merged as refactor + foundation, with the explicit understanding
that the decode CPU gap is **not resolved** by this change. Further work tracked below.

## Follow-up (next iteration)

Ranked by expected ROI given this analysis:

1. **Break the inline-`asm!` black box constraint.** Options:
   - (a) Port the main loop to `vfmaq_f16` intrinsic so the compiler can schedule
     B loads ahead of FMA across row boundaries. Requires confirming intrinsic
     stability on the project's MSRV.
   - (b) Merge the 4 per-row asm blocks into a single asm block covering 2+ rows with
     explicit instruction interleaving (load row N+1 B inside the row N FMA window).
   Target: close the ~2→5+ instruction gap between load and use.

2. **Chunk-size tuning to match llama.cpp's GEMV heuristic.** llama.cpp uses
   `chunk_size=64` rows per chunk in GEMV (~140 chunks for Qwen n=8960); we use
   `n_threads*8=64` chunks (~140 rows per chunk). Reverse the granularity. Near-zero
   implementation cost, easy A/B.

3. **Single-asm-block super kernel** that covers the full K step for all 4 rows,
   fully explicit scheduling, pre-fetch interleaving. Highest potential gain but
   biggest implementation + maintenance cost.

4. **Big.LITTLE affinity experiment.** llama.cpp default is no affinity, so this is
   not a gap-explainer, but setting `pthread_setaffinity_np` to Phoenix L+M cores
   (excluding E-cores if any show up) may reduce variance on long decode.

5. **Prefetch insertion.** Historical no-effect on S24 (`b25bc19`), but the now-reduced
   instruction count may give `prfm` instructions room to land without displacing FMAs.
   Low priority — do after (1).

## Test artifacts

- `fma_strict_isolation.txt` — raw per-run values, thermal readings
- `FMA_ANALYSIS.md` — this file
- Baseline (unchanged): `qwen_missing_legs_strict.txt` (FMLAL V10 CPU)
- Greedy-decode transcripts (temporary, `/tmp/f{mlal,ma}_{short,long}.txt`)
