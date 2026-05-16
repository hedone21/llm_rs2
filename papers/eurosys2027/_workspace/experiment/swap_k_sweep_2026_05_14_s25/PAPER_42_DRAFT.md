# Paper §4.2 — Weight Swap Policy Comparison
2026-05-14 measurements, Galaxy S25 (R3CY408S5SB), Qwen2.5-1.5B F16→Q4 (28 layers)

## Setup
- F16 primary (2.85 GB) + Q4_0 GGUF secondary (1.21 GB)
- backend: qnn_oppkg (Android default after 2026-05-14)
- rpcmem alias path (`SecondaryMmap::Rpcmem`)
- Cold reboot + 3 min thermal rest before each measurement
- Code change: alias path batch prefault skip (`swap_executor.rs:474`)

## Headline (Total wall-clock, 30 tokens)

```
Mode                Total      vs Q4 floor
─────────────────────────────────────────
Q4 only (no swap)   1788 ms    +0.0%  ← floor
IF (LISWAP-4)       1875 ms    +4.9%  ← almost full hide
K=7 sync            2048 ms   +14.5%
K=1 sync            2153 ms   +20.4%
SS (one-shot)       2205 ms   +23.3%
K=11 sync           2242 ms   +25.4%
K=3 sync            2388 ms   +33.6%
K=4 sync            3065 ms   +71.4%  ← anomaly
K=19 sync           2620 ms   +46.5%
K=25 sync           2694 ms   +50.7%
F16 only (no swap)  2447 ms   +36.9%
```

## K sweep curve (Total ms vs K)

(see all_k_sweep.csv for plot data — K=1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27 + K=2, 4, 14)

## Per-token TBT distribution (K=7 v3)

```
swap progress (tok 0~3):  136 / 130 / 141 / 149 ms  (avg 139)
swap complete (tok 4~28): 24~25 ms (steady)
```

## Per-token TBT distribution (K=1 sync 280)

```
swap progress (tok 0~27): 61 → 51 → 50 → ... → 30 ms (decreasing)
swap complete (tok 28+): 28 ms steady
```

## Key Findings

1. **Alias path is essential** — decode forward-only drops from 17 ms/tok (AUF AOS) to 3 ms/tok (rpcmem alias) (-80%).
2. **IF (LISWAP-4) is the only true hide mechanism** — token-level swap (K=N async) defeated by sub-batch reactive wait (memory spike safety). Only forward-internal layer-level boundary escapes the wait.
3. **K sweep sweet spot K=7 (28÷7=4 ticks)** — not "exact division" (K=4 with 7 ticks is anomalously slow at 3065 ms), but tick-count balance.
4. **Forward-only post-swap = 3 ms but TBT = 28 ms** — 25 ms tail unaccounted for. Independent of K. Same as Q4 baseline (30.52 ms). Possibly sampling / KV grow / barrier overhead.
5. **Dynamic-K cap = K_intent ∈ {1, 2}** by Qwen 1.5B quality drift cap — cannot reach sweet spot K=7 automatically.

## Open Questions

a. Why K=4 (7 ticks) so much slower than K=2/7/14? Thermal accumulation in later ticks? mmap cache eviction?
b. What is the 25 ms TBT tail? Forward-only baseline is 3 ms.
c. Quality drift K=3+ claim (docs/48_swap_dynamic_k_guide.md) — only observed at ratio=0.9 (mixed precision), not ratio=1.0?
