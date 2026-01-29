# 📊 LLM Research Benchmark Log

Last Updated: 2026-01-29 09:08:25

## Executive Summary
- **Total Benchmarks**: 18
- **Recent Run**: 2026-01-29 09:04

## Detailed Results


| Date | Model | Backend | Input | Tokens | TTFT (ms) | TBT (ms) | T/s | Temp (°C) | Mem (MB) | Env | Data | Plot |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-01-29 09:04 | llama3.2-1b | opencl | short_len | 2048 | **160.9** | 94.73 | 10.6 | 37.9 -> 37.6 (Max 38.1) | 3850 | **Camera** | [JSON](data/profile_opencl_short_len_2048_20260129_090810.json) | [Graph](plots/profile_opencl_short_len_2048_20260129_090810.png) |
| 2026-01-29 08:59 | llama3.2-1b | cpu | short_len | 2048 | **202.3** | 138.41 | 7.2 | 27.8 -> 38.1 (Max 38.1) | 6598 | **Camera** | [JSON](data/profile_cpu_short_len_2048_20260129_090435.json) | [Graph](plots/profile_cpu_short_len_2048_20260129_090435.png) |
| 2026-01-29 08:16 | llama3.2-1b | opencl | short_len | 512 | **201.8** | 56.02 | 17.9 | 35.9 -> 37.1 (Max 37.1) | 4471 | Idle | [JSON](data/profile_opencl_short_len_512_20260129_081716.json) | [Graph](plots/profile_opencl_short_len_512_20260129_081716.png) |
| 2026-01-29 08:15 | llama3.2-1b | cpu | short_len | 512 | **159.3** | 55.17 | 18.1 | 32.4 -> 35.7 (Max 35.7) | 4992 | Idle | [JSON](data/profile_cpu_short_len_512_20260129_081627.json) | [Graph](plots/profile_cpu_short_len_512_20260129_081627.png) |
| 2026-01-29 08:12 | llama3.2-1b | opencl | short_len | 2048 | **151.1** | 74.44 | 13.4 | 25.2 -> 34.9 (Max 34.9) | 5717 | Idle | [JSON](data/profile_opencl_short_len_2048_20260129_081452.json) | [Graph](plots/profile_opencl_short_len_2048_20260129_081452.png) |
| 2026-01-29 08:10 | llama3.2-1b | opencl | short_len | 2048 | **N/A** | N/A | N/A | 0.0 -> 0.0 (Max 0.0) | 0 | Idle | [JSON](data/profile_opencl_short_len_2048_20260129_081041.json) | - |
| 2026-01-23 16:04 | llama3.2-1b | cpu | short_len | 2048 | **N/A** | N/A | N/A | 29.6 -> 35.2 (Max 35.2) | 3751 | Idle | [JSON](data/profile_cpu_short_len_2048_20260123_160655.json) | [Graph](plots/profile_cpu_short_len_2048_20260123_160655.png) |
| 2026-01-23 15:57 | llama3.2-1b | opencl | short_len | 128 | **184.6** | 40.22 | 24.9 | 29.9 -> 31.5 (Max 31.5) | 3770 | Idle | [JSON](data/profile_opencl_short_len_128_20260123_155723.json) | [Graph](plots/profile_opencl_short_len_128_20260123_155723.png) |
| 2026-01-23 15:56 | llama3.2-1b | cpu | short_len | 128 | **182.9** | 36.72 | 27.2 | 29.5 -> 29.5 (Max 29.5) | 4185 | Idle | [JSON](data/profile_cpu_short_len_128_20260123_155657.json) | [Graph](plots/profile_cpu_short_len_128_20260123_155657.png) |
| 2026-01-23 15:49 | llama3.2-1b | opencl | short_len | 128 | **189.3** | 40.53 | 24.7 | 30.1 -> 30.1 (Max 30.1) | 4022 | Idle | [JSON](data/profile_opencl_short_len_128_20260123_154929.json) | [Graph](plots/profile_opencl_short_len_128_20260123_154929.png) |
| 2026-01-23 15:48 | llama3.2-1b | cpu | short_len | 128 | **175.1** | 35.65 | 28.1 | 29.2 -> 29.2 (Max 29.2) | 4275 | Idle | [JSON](data/profile_cpu_short_len_128_20260123_154905.json) | [Graph](plots/profile_cpu_short_len_128_20260123_154905.png) |
| 2026-01-23 15:19 | llama3.2-1b | cpu | short_len | 1024 | **199.8** | 90.10 | 11.1 | 36.9 -> 38.1 (Max 38.5) | 4138 | Idle | [JSON](data/profile_cpu_short_len_1024_20260123_152111.json) | [Graph](plots/profile_cpu_short_len_1024_20260123_152111.png) |
| 2026-01-23 15:17 | llama3.2-1b | cpu | short_len | 1024 | **183.0** | 87.87 | 11.4 | 35.3 -> 36.9 (Max 37.2) | 4200 | Idle | [JSON](data/profile_cpu_short_len_1024_20260123_151928.json) | [Graph](plots/profile_cpu_short_len_1024_20260123_151928.png) |
| 2026-01-23 15:16 | llama3.2-1b | cpu | short_len | 1024 | **182.5** | 78.15 | 12.8 | 30.7 -> 35.3 (Max 35.3) | 4150 | Idle | [JSON](data/profile_cpu_short_len_1024_20260123_151746.json) | [Graph](plots/profile_cpu_short_len_1024_20260123_151746.png) |
| 2026-01-23 15:09 | llama3.2-1b | opencl | short_len | 1024 | **201.6** | 75.67 | 13.2 | 32.3 -> 35.4 (Max 35.4) | 4232 | Idle | [JSON](data/profile_opencl_short_len_1024_20260123_151054.json) | [Graph](plots/profile_opencl_short_len_1024_20260123_151054.png) |
| 2026-01-23 15:07 | llama3.2-1b | cpu | short_len | 1024 | **180.2** | 76.52 | 13.1 | 28.1 -> 32.6 (Max 32.6) | 4336 | Idle | [JSON](data/profile_cpu_short_len_1024_20260123_150908.json) | [Graph](plots/profile_cpu_short_len_1024_20260123_150908.png) |
| 2026-01-23 14:38 | llama3.2-1b | cpu | med_len | 64 | **38217.0** | 137.11 | 7.3 | 28.1 -> 33.0 (Max 33.0) | 4415 | Idle | [JSON](data/profile_cpu_med_len_64_20260123_143945.json) | [Graph](plots/profile_cpu_med_len_64_20260123_143945.png) |
| 2026-01-23 14:31 | llama3.2-1b | cpu | short_len | 32 | **181.2** | 37.17 | 26.9 | 28.5 -> 28.5 (Max 28.5) | 4561 | Idle | [JSON](data/profile_cpu_short_len_32_20260123_143133.json) | [Graph](plots/profile_cpu_short_len_32_20260123_143133.png) |

## Analysis & Findings

### 1. Short Decode (128 Tokens)
| Metric | CPU | GPU (OpenCL) | Winner |
|:---|:---|:---|:---|
| **TTFT (ms)** | **182.9** | 184.6 | 🏆 CPU (Slightly) |
| **Avg TBT (ms)** | **36.72** | 40.22 | 🏆 CPU (~10% faster) |
| **Throughput (T/s)** | **27.2** | 24.9 | 🏆 CPU |

**Finding**: For very short sequences, CPU overhead is lower than GPU dispatch overhead. CPU is preferred.

### 2. Crossover Point (512 Tokens)
| Metric | CPU | GPU (OpenCL) | Winner |
|:---|:---|:---|:---|
| **TTFT (ms)** | **159.3** | 201.8 | 🏆 CPU |
| **Avg TBT (ms)** | **55.17** | 56.02 | 🤝 Tie (Diff < 1ms) |
| **Throughput (T/s)** | 18.1 | 17.9 | 🤝 Tie |

**Finding**: At ~512 tokens, performance converges. This is the ideal **Hybrid Switching Point**.

### 3. Long Context (2048 Tokens)
| Metric | CPU | GPU (OpenCL) | Winner |
|:---|:---|:---|:---|
| **TTFT (ms)** | 202.3 | **160.9** | 🏆 GPU |
| **Avg TBT (ms)** | 138.41 | **94.73** | 🏆 GPU (~30% faster) |
| **Throughput (T/s)** | 7.2 | **10.6** | 🏆 GPU |
| **Temperature (°C)** | 38.1 (Rising) | 38.1 (Stable) | 🏆 GPU (More stable) |
| **Memory (MB)** | 6598 | **3850** | 🏆 GPU (Efficient) |

**Finding**: In long contexts, GPU is significantly faster, more memory efficient, and thermally stable. CPU performance degrades rapidly.

## Recommendation
Implement **Hybrid Switching Strategy**:
- **Start on CPU** for fast Time-to-First-Token and short responses.
- **Switch to GPU** if generation exceeds **512 tokens**.

## Graphical Analysis
Plots available in `benchmarks/plots/`. (Manually added below)
