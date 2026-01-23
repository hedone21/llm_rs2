# 📊 LLM Research Benchmark Log

Last Updated: 2026-01-23 16:07:22

## Executive Summary
- **Total Benchmarks**: 12
- **Recent Run**: 2026-01-23 16:04

## Detailed Results

| Date | Model | Backend | Input | Tokens | TTFT (ms) | TBT (ms) | T/s | Temp (°C) | Mem (MB) | Data | Plot |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-01-23 16:04 | llama3.2-1b | cpu | short_len | 2048 | **N/A** | N/A | N/A | 29.6 -> 35.2 (Max 35.2) | 3751 | [JSON](data/profile_cpu_short_len_2048_20260123_160655.json) | [Graph](plots/profile_cpu_short_len_2048_20260123_160655.png) |
| 2026-01-23 15:57 | llama3.2-1b | opencl | short_len | 128 | **184.6** | 40.22 | 24.9 | 29.9 -> 31.5 (Max 31.5) | 3770 | [JSON](data/profile_opencl_short_len_128_20260123_155723.json) | [Graph](plots/profile_opencl_short_len_128_20260123_155723.png) |
| 2026-01-23 15:56 | llama3.2-1b | cpu | short_len | 128 | **182.9** | 36.72 | 27.2 | 29.5 -> 29.5 (Max 29.5) | 4185 | [JSON](data/profile_cpu_short_len_128_20260123_155657.json) | [Graph](plots/profile_cpu_short_len_128_20260123_155657.png) |
| 2026-01-23 15:49 | llama3.2-1b | opencl | short_len | 128 | **189.3** | 40.53 | 24.7 | 30.1 -> 30.1 (Max 30.1) | 4022 | [JSON](data/profile_opencl_short_len_128_20260123_154929.json) | [Graph](plots/profile_opencl_short_len_128_20260123_154929.png) |
| 2026-01-23 15:48 | llama3.2-1b | cpu | short_len | 128 | **175.1** | 35.65 | 28.1 | 29.2 -> 29.2 (Max 29.2) | 4275 | [JSON](data/profile_cpu_short_len_128_20260123_154905.json) | [Graph](plots/profile_cpu_short_len_128_20260123_154905.png) |
| 2026-01-23 15:19 | llama3.2-1b | cpu | short_len | 1024 | **199.8** | 90.10 | 11.1 | 36.9 -> 38.1 (Max 38.5) | 4138 | [JSON](data/profile_cpu_short_len_1024_20260123_152111.json) | [Graph](plots/profile_cpu_short_len_1024_20260123_152111.png) |
| 2026-01-23 15:17 | llama3.2-1b | cpu | short_len | 1024 | **183.0** | 87.87 | 11.4 | 35.3 -> 36.9 (Max 37.2) | 4200 | [JSON](data/profile_cpu_short_len_1024_20260123_151928.json) | [Graph](plots/profile_cpu_short_len_1024_20260123_151928.png) |
| 2026-01-23 15:16 | llama3.2-1b | cpu | short_len | 1024 | **182.5** | 78.15 | 12.8 | 30.7 -> 35.3 (Max 35.3) | 4150 | [JSON](data/profile_cpu_short_len_1024_20260123_151746.json) | [Graph](plots/profile_cpu_short_len_1024_20260123_151746.png) |
| 2026-01-23 15:09 | llama3.2-1b | opencl | short_len | 1024 | **201.6** | 75.67 | 13.2 | 32.3 -> 35.4 (Max 35.4) | 4232 | [JSON](data/profile_opencl_short_len_1024_20260123_151054.json) | [Graph](plots/profile_opencl_short_len_1024_20260123_151054.png) |
| 2026-01-23 15:07 | llama3.2-1b | cpu | short_len | 1024 | **180.2** | 76.52 | 13.1 | 28.1 -> 32.6 (Max 32.6) | 4336 | [JSON](data/profile_cpu_short_len_1024_20260123_150908.json) | [Graph](plots/profile_cpu_short_len_1024_20260123_150908.png) |
| 2026-01-23 14:38 | llama3.2-1b | cpu | med_len | 64 | **38217.0** | 137.11 | 7.3 | 28.1 -> 33.0 (Max 33.0) | 4415 | [JSON](data/profile_cpu_med_len_64_20260123_143945.json) | [Graph](plots/profile_cpu_med_len_64_20260123_143945.png) |
| 2026-01-23 14:31 | llama3.2-1b | cpu | short_len | 32 | **181.2** | 37.17 | 26.9 | 28.5 -> 28.5 (Max 28.5) | 4561 | [JSON](data/profile_cpu_short_len_32_20260123_143133.json) | [Graph](plots/profile_cpu_short_len_32_20260123_143133.png) |


## Graphical Analysis
Plots available in `benchmarks/plots/`. (Manually added below)
