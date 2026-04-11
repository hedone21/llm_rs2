# C1.5-a Regression Suite Summary

**Branch**: `feat/plan-qkv-bias` (HEAD after Task 6)
**Base**: `master` @ `eb689d1`
**Date**: 2026-04-11
**Device**: Samsung Galaxy S25 (Adreno 830, adb R3CY408S5SB, Snapdragon 8 Elite)
**Inter-test cooldown**: 2 minutes (120s sleep between runs)

## Test matrix results

| # | Test | Model | Backend | Decode (ms/tok) | Plan built | Notes |
|---|---|---|---|---|---|---|
| a | CPU smoke | llama3.2-1b | `cpu` | 43.73 | n/a | Sane English, lazy dog + grammar continuation |
| b | GPU baseline | llama3.2-1b | `opencl` | 40.18 | **16 layers** | DK=64 flash active, unchanged from master |
| c | CPU smoke | qwen2.5-1.5b | `cpu` | 57.45 | n/a | Sane English, "The capital of France is the city where..." |
| d | **GPU new** | qwen2.5-1.5b | `opencl` | 58.02 | **28 layers** | **NEW**: plan path active (was disabled on master by `has_qkv_bias` gate), DK=128 flash active |
| e | H2O regression | llama3.2-1b | `opencl` | 42.06 | n/a (has_scores) | H2O eviction fires, GPU Score accumulator OK |
| f | Profile regression | llama3.2-1b | `opencl` | 81.60* | n/a (profile) | Per-op breakdown + JSON export OK (*includes ~54 ms/tok profile sync overhead) |

## Key assertions (all passing)

1. **Llama3.2-1b GPU not regressed**: test b still shows `GPU kernel plan built (16 layers, capacity=128)` and DK=64 flash compiled. Decode 40.18 ms/tok matches pre-C1.5-a typical range.

2. **Llama3.2-1b CPU works**: test a produces sane continuation and valid Decode TBT. The CPU backend does not touch plan.rs, so this is a smoke test that the refactor didn't break CPU code paths.

3. **Qwen CPU works**: test c produces sane continuation and valid Decode TBT. Same smoke test for CPU backend.

4. **Qwen GPU plan NOW active**: test d shows `GPU kernel plan built (28 layers, capacity=128)` — this is the critical new behavior. Before C1.5-a, Qwen fell through to `forward_into` per-step because of the `has_qkv_bias` early return in `transformer.rs:1437`. After lifting that gate, Qwen exercises the pre-bound kernel path with conditional bias steps.

5. **H2O eviction regression pass**: test e shows the full event chain: `[GPU Score] Accumulator initialized` → `Eviction: policy=h2o, ...` → `[CacheEvent] Eviction completed: ...`. Same behavior as V4 on master.

6. **Profile mode regression pass**: test f shows `[Profile] Per-op breakdown (accumulated over 240 layer-calls)` and JSON export. `240 = 16 layers × 15 decoded tokens` (the binary stopped early at token 15 due to EOS or sampling constraints, not a regression).

## Output quality check

- All 6 test output files contain sane English continuations.
- No garbled tokens, no repetitive loops, no all-zero decoding.
- No `DK=128 failed` warnings in any OpenCL run.
- Prefill and decode both complete cleanly.

## Conclusion

**C1.5-a introduces zero correctness regressions** across the four model × backend combinations and the two feature regression tests (H2O eviction + profile). The only behavior change is the intended one: Qwen now uses the pre-bound GPU kernel plan path.

Raw logs:
- `c15a_a_llama_cpu.txt`
- `c15a_b_llama_gpu.txt`
- `c15a_c_qwen_cpu.txt`
- `c15a_d_qwen_gpu.txt`
- `c15a_e_h2o.txt`
- `c15a_f_profile.txt`
- `c15a_qwen_device.txt` (initial on-device verification from Task 4)

## Performance context

See `results/data/flash_attn_decode/thermal/C15A_COMPARISON.md` for the V9 strict
thermal isolation comparison. Short summary: llm.rs GPU Qwen delta vs V9 baseline
is -0.8% (short) / -0.7% (long) — within measurement noise. The expected
8-14 ms/token improvement from eliminating per-step arg rebinding did NOT
materialize because the real forward_into entry overhead was much smaller than
estimated (~1-2 ms/token rather than ~10 ms/token).

C1.5-a is therefore a **correctness + infrastructure** win, not a performance
win on Qwen. It normalizes Qwen with other models under the plan.rs path and
enables future optimizations (FFN fused kernels, Q4_0 weight bandwidth reduction,
attention tuning) to land inside the plan framework.
