# Intrinsic Port Experiment — Negative Outcome, Reverted

**Date**: 2026-04-11
**Device**: Samsung Galaxy S25 (Adreno 830, Snapdragon 8 Elite / Oryon)
**Feature branch**: `feat/f16-intrinsic-gemv` (2 commits, **not merged**)
**Master HEAD during experiment**: `d962878` (FMA inline asm) — **unchanged**

## Context

Follow-up to the FMA inline-asm port (`FMA_ANALYSIS.md`) which delivered
correctness-preserving refactor but no measured gap reduction. Disassembly of
`vec_dot_f16_native_gemv_4rows` at `0x369e20` showed a 2-instruction gap between
B-row load and first use, hypothesized to block HW latency hiding. Inline `asm!`
blocks are opaque to the Rust compiler so the load could not be scheduled earlier.

Hypothesis: porting the kernel to the `vfmaq_f16` intrinsic (nightly
`stdarch_neon_f16` feature) would let the compiler schedule loads across row
boundaries, hiding the ~5-cycle Oryon vector-load latency and closing the gap
to llama.cpp CPU.

User explicitly approved nightly toolchain transition for this experiment.

## What was implemented

Feature branch `feat/f16-intrinsic-gemv`:
- `ba66495` — `build(toolchain): pin nightly for f16 NEON intrinsics`
  - `rust-toolchain.toml` new (channel = nightly)
  - `engine/src/lib.rs` feature gates: `#![feature(stdarch_neon_f16)]`,
    `#![cfg_attr(target_arch = "aarch64", feature(f16))]`
- `fad1421` — `perf(neon): port F16 GEMV kernels to vfmaq_f16 intrinsics`
  - `vec_dot_f16_native_gemv_4rows` — 16× `float16x8_t` accumulators, K step 32,
    100% intrinsic (no inline asm)
  - `vec_dot_f16_native_gemv_1row` — 4× accumulators, same pattern
  - Reduction via `vcvt_f32_f16` + `vcvt_high_f32_f16` + f32 accumulation
  - Existing inline asm scaffolding removed (macros, `uint16x8_t` accumulators)

Unit tests passed with the same tolerance (`rtol=1e-2`, `atol=3e-3·√K+1e-2`);
K=8960 worst case `max_abs=0.224` vs budget 0.294 (24% headroom).

## Disassembly evidence (positive, theoretical)

Inline asm (master `d962878` at `0x369e20`), 36-instruction main loop:
```asm
fmla v20.8h, v28.8h, v24.8h    ; row 0 last FMA
ldp  q25, q24, [x12]            ; row 1 B load
add  x10, x10, #0x40
ldp  q27, q26, [x12, #-0x20]
fmla v19.8h, v31.8h, v27.8h    ; row 1 first FMA — 2 instructions after load
```

Intrinsic (branch `feat/f16-intrinsic-gemv` at `0x33fc24`), 30-instruction main loop:
```asm
ldp q30, q31, [x16], #0x40      ; post-increment load (no separate add)
fmla v21.8h, v26.8h, v24.8h     ; row 0 FMA #1 (v26 +4 instructions earlier)
fmla v23.8h, v27.8h, v25.8h
cmp x4, x0                       ; loop overhead interleaved
ldp q26, q27, [x15, #-0x20]     ; row 1 B load — during row 0 FMAs (!)
fmla v22.8h, v30.8h, v28.8h
fmla v20.8h, v31.8h, v29.8h
fmla v19.8h, v26.8h, v24.8h     ; row 1 FMA #1 (v26 +3 instructions earlier)
```

Improvements confirmed in the disassembly:
- Main loop instruction count: **36 → 30** (−17%)
- Load-to-use distance: **2 → 3–5 instructions**
- `cmp`/`add` interleaved with FMAs instead of batched at loop tail
- Post-increment `ldp` eliminates 6 separate `add` instructions per iter
- Row N+1 B-row load scheduled **inside** row N's FMA block (latency hiding)
- Register spill: none (all 32 NEON regs in use, no stack traffic)

This is the exact improvement the hypothesis predicted.

## Measurement (N=1 thermal-controlled, 5-min rest)

| | FMA inline asm (V10 baseline, median) | Intrinsic N=1 | Δ |
|---|---|---|---|
| **Short decode** (pf=7, dc=64) | 57.34 ms/tok | **57.32 ms/tok** | −0.02 (noise) |
| **Long prefill** (657 tokens) | 6684–6940 ms | **7999 ms** | **+1060–1315 ms (+15–20%)** |
| **Long decode** (pf=720, dc=128) | 64.35 ms/tok | **70.17 ms/tok (3 tokens only)** | uninterpretable |

Short: **no improvement**, within noise.
Long decode: measurement unreliable — default sampling produced only 3 decode
tokens before an early EOS, startup cost dominates per-token value.
Prefill: **clear regression**, ~+1.3 seconds.

Quality verification (greedy, deterministic): both short (33 tokens) and long
(37 tokens) outputs byte-identical to FMA inline asm baseline. F16 accumulator
precision intact.

## Why the disassembly improvement did not translate to runtime

Three possible explanations, in order of likelihood:

1. **Roofline ceiling already close.** The FMA inline-asm kernel on S25 is
   already near the DRAM bandwidth roof for F16 GEMV (AI=1 flop/byte, balance
   point ~4.7). Further instruction-level scheduling cannot push past what
   memory subsystem delivers. Past S24 experiments (`b25bc19`, 2026-03)
   noted "inner loop optimizations (multi-row, prefetch, stride) have no
   effect because the bottleneck is DRAM bandwidth utilization". S25 on
   LPDDR5X is in the same regime, only higher in absolute GB/s.

2. **Stall lives outside this kernel.** Load-to-use gap in main loop may not
   have been the dominant per-token cost. Candidates: RMSNorm, attention
   softmax, sampling, or thread dispatch / context switch overhead. A real
   gap-close would require identifying which.

3. **Nightly codegen regression bled into other paths.** Prefill +15–20%
   with the `vec_dot_f16_native_4x4` kernel untouched strongly suggests
   something in nightly's global codegen changed (LLVM version, inlining
   heuristics, target feature propagation). Since prefill got slower with
   no source-level change in its hot path, the whole-binary shift is the
   most consistent explanation for that specific measurement. This compounds
   the analysis: even if the GEMV kernel itself improved, nightly's
   whole-crate regressions drowned it.

## Decision: revert

The branch is not merged. `master` stays at `d962878` (FMA inline asm).
Device binary restored from `/data/local/tmp/generate.fma-asm.backup`.
Branch `feat/f16-intrinsic-gemv` is kept as a reference (not deleted) with
commits `ba66495` and `fad1421` for future consultation.

The two claims this experiment establishes for future follow-up:
1. **Disassembly-level improvements do not imply runtime improvements** on
   this hardware/kernel combination. Future optimization work must validate
   with end-to-end thermal-protocol measurement before committing.
2. **Nightly toolchain transition has non-trivial runtime cost** that is not
   visible without whole-binary benchmarking. Any future attempt to use
   nightly-only features must budget for this effect.

## Follow-up redirection

The original P1 backlog item ("Qwen CPU decode GEMV load-to-use stall 해소")
needs revision. The disassembly-isolated stall was not the main gap origin,
so the P1 acceptance criteria of "decode ≤ llama.cpp + 5%" is not reachable
via kernel instruction scheduling alone. Next candidates, in priority order:

1. **Per-op decode profiling with production binary** (not `--profile` flag,
   which adds sync overhead). Use `perf`, `simpleperf`, or hand-instrumented
   timestamps to find the per-token hot spots outside `matmul_ffn`. Without
   this, further optimization is pattern-matching on hunches.

2. **Thread pool / dispatch overhead characterization**. The SpinPool path is
   custom; measure dispatch cost per chunk on Oryon to see if it's a material
   fraction of per-token time.

3. **Chunk-size tuning A/B** (llama.cpp uses 64 rows/chunk for GEMV vs our
   `n_threads * 8 = 64` chunks with 140 rows/chunk). Low implementation cost,
   worth one cycle just to rule in or out.

4. **Big.LITTLE affinity experiment for variance reduction**. Not a gap-closer
   since llama.cpp default has no affinity either, but the bi-modal long-decode
   variance (runs bouncing between ~60 and ~64 ms/tok at the same thermal
   state) suggests scheduling jitter on Oryon Phoenix L/M mix. Pinning may
   tighten the distribution even if medians don't move.

5. **Single-asm super-block** (stable-friendly alternative to the intrinsic
   approach). Explicit multi-row interleaving inside a single `asm!` block.
   Likely low ROI given result (1) above, but a last option if all else fails
   and the theoretical latency-hiding path is still worth exhausting.

Item (1) is now the blocker — we do not have evidence for which operation
dominates outside `matmul_ffn` at production binary settings.

## Artifacts

- `intrinsic_n1_experiment.txt` — raw N=1 measurement
- `INTRINSIC_EXPERIMENT.md` — this file
- Branch `feat/f16-intrinsic-gemv` (commits `ba66495`, `fad1421`) — retained
- Device backup `/data/local/tmp/generate.fma-asm.backup` — retained
