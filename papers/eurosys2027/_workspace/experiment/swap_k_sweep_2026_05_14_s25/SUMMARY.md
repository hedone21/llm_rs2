# GGUF Q4 secondary + rpcmem alias path measurements
2026-05-14, Galaxy S25 (R3CY408S5SB), Qwen2.5-1.5B
Code change: swap_executor.rs alias_path prefault skip

| Mode | Prefill | TTFT | tok[0] | Decode fwd | Avg TBT | swap visible | Total (~) |
|------|---------|------|--------|------------|---------|--------------|-----------|
| K=1 v2  | 755 | 906  | 5.74  | 5.09 | 43.05 | 28×~30ms (mmap_permute) | 2153 |
| K=19 v2 | 769 | 918  | -     | 3.66 | 58.71 | 2×~470ms (mmap_permute) | 2620 |
| IF v2   | 768 | 912  | 871   | 3.29 | 55.77 | 0 (hidden, dispatcher worker) | 1875 |
| SS v2   | 767 | 1465 | 4.33  | 3.60 | 25.51 | 566ms in TTFT | 2205 |

## vs AUF baseline (prior measurements, /tmp/swap_measurements_20260514/)
| Mode | AUF      | GGUF+alias | Δ      |
|------|----------|-----------|---------|
| K=1  | 2430     | 2153      | -11%   |
| K=19 | 1926     | 2620      | +36%(!)  |
| IF   | 1957     | 1875      | -4%    |
| SS   | -        | 2205      | -      |

## Decode forward-only (post-swap pure Q4 forward)
| Mode | AUF AOS    | GGUF alias |
|------|------------|-----------|
| K=1  | 17 ms/tok  | 5.09 ms/tok  (-70%) |
| K=19 | 17 ms/tok  | 3.66 ms/tok  (-78%) |
| IF   | 17 ms/tok  | 3.29 ms/tok  (-80%) |
| SS   | ?          | 3.60 ms/tok |
