# 📊 LLM Research Benchmark Log

Last Updated: 2026-01-30 13:12:26

## Executive Summary
- **Total Benchmarks**: 46
- **Recent Run**: 2026-01-30 13:11

## Detailed Results

| Date | Model | Backend | Input | Tokens | TTFT (ms) | TBT (ms) | T/s | Temp (°C) | Mem (MB) | Data | Plot |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-01-30 13:11 | llama3.2-1b | opencl | short_len | 256 | **202.2** | 53.61 | 18.7 | 39.3 -> 39.3 (Max 39.4) | 3465 | [JSON](data/profile_opencl_short_len_256_20260130_131219.json) | - |
| 2026-01-30 13:11 | llama3.2-1b | opencl | short_len | 128 | **207.7** | 52.22 | 19.1 | 40.3 -> 39.3 (Max 39.9) | 3535 | [JSON](data/profile_opencl_short_len_128_20260130_131153.json) | - |
| 2026-01-30 13:10 | llama3.2-1b | opencl | prefill_1024 | 256 | **18715.5** | 83.46 | 12.0 | 40.0 -> 40.4 (Max 40.9) | 3485 | [JSON](data/profile_opencl_prefill_1024_256_20260130_131132.json) | - |
| 2026-01-30 13:09 | llama3.2-1b | opencl | prefill_512 | 256 | **7880.0** | 64.12 | 15.6 | 36.9 -> 38.0 (Max 38.0) | 3667 | [JSON](data/profile_opencl_prefill_512_256_20260130_130956.json) | - |
| 2026-01-30 13:09 | llama3.2-1b | opencl | prefill_1024 | 128 | **17441.2** | 83.85 | 11.9 | 38.2 -> 39.6 (Max 39.6) | 3473 | [JSON](data/profile_opencl_prefill_1024_128_20260130_131038.json) | - |
| 2026-01-30 13:08 | llama3.2-1b | opencl | prefill_512 | 128 | **7999.5** | 63.13 | 15.8 | 35.8 -> 36.9 (Max 36.9) | 3771 | [JSON](data/profile_opencl_prefill_512_128_20260130_130919.json) | - |
| 2026-01-30 13:08 | llama3.2-1b | opencl | prefill_128 | 256 | **1933.7** | 59.76 | 16.7 | 35.3 -> 35.8 (Max 35.8) | 3756 | [JSON](data/profile_opencl_prefill_128_256_20260130_130849.json) | - |
| 2026-01-30 13:07 | llama3.2-1b | opencl | prefill_128 | 128 | **1929.3** | 57.18 | 17.5 | 35.8 -> 35.3 (Max 35.8) | 3803 | [JSON](data/profile_opencl_prefill_128_128_20260130_130819.json) | - |
| 2026-01-30 13:07 | llama3.2-1b | cpu | short_len | 128 | **268.1** | 42.78 | 23.4 | 35.5 -> 35.0 (Max 35.5) | 4066 | [JSON](data/profile_cpu_short_len_128_20260130_130724.json) | - |
| 2026-01-30 13:07 | llama3.2-1b | cpu | short_len | 256 | **181.2** | 58.00 | 17.2 | 34.8 -> 35.1 (Max 35.2) | 3816 | [JSON](data/profile_cpu_short_len_256_20260130_130754.json) | - |
| 2026-01-30 13:06 | llama3.2-1b | cpu | prefill_1024 | 256 | **27155.0** | 97.60 | 10.2 | 34.7 -> 35.6 (Max 35.6) | 3846 | [JSON](data/profile_cpu_prefill_1024_256_20260130_130704.json) | - |
| 2026-01-30 13:05 | llama3.2-1b | cpu | prefill_1024 | 128 | **23707.8** | 93.48 | 10.7 | 32.7 -> 34.8 (Max 34.8) | 3835 | [JSON](data/profile_cpu_prefill_1024_128_20260130_130558.json) | - |
| 2026-01-30 13:04 | llama3.2-1b | cpu | prefill_512 | 128 | **9529.6** | 55.94 | 17.9 | 28.6 -> 29.5 (Max 29.5) | 3813 | [JSON](data/profile_cpu_prefill_512_128_20260130_130429.json) | - |
| 2026-01-30 13:04 | llama3.2-1b | cpu | prefill_512 | 256 | **10135.2** | 70.56 | 14.2 | 30.6 -> 32.6 (Max 32.6) | 3823 | [JSON](data/profile_cpu_prefill_512_256_20260130_130510.json) | - |
| 2026-01-30 13:03 | llama3.2-1b | cpu | prefill_128 | 128 | **2067.4** | 45.45 | 22.0 | 23.0 -> 23.0 (Max 23.0) | 4124 | [JSON](data/profile_cpu_prefill_128_128_20260130_130332.json) | - |
| 2026-01-30 13:03 | llama3.2-1b | cpu | prefill_128 | 256 | **2120.0** | 51.60 | 19.4 | 24.3 -> 26.7 (Max 26.7) | 3848 | [JSON](data/profile_cpu_prefill_128_256_20260130_130400.json) | - |
| 2026-01-30 12:29 | llama3.2-1b | opencl | prefill_1024 | 128 | **20401.9** | 117.90 | 8.5 | 41.6 -> 43.0 (Max 43.0) | 3639 | [JSON](data/profile_opencl_prefill_1024_128_20260130_122956.json) | - |
| 2026-01-30 12:29 | llama3.2-1b | opencl | prefill_1024 | 256 | **20726.9** | 90.79 | 11.0 | 42.5 -> 43.1 (Max 43.4) | 3444 | [JSON](data/profile_opencl_prefill_1024_256_20260130_123052.json) | - |
| 2026-01-30 12:28 | llama3.2-1b | opencl | prefill_512 | 256 | **7536.3** | 74.02 | 13.5 | 41.0 -> 41.7 (Max 41.7) | 3644 | [JSON](data/profile_opencl_prefill_512_256_20260130_122907.json) | - |
| 2026-01-30 12:27 | llama3.2-1b | opencl | prefill_512 | 128 | **7975.9** | 106.33 | 9.4 | 40.0 -> 40.8 (Max 40.8) | 3866 | [JSON](data/profile_opencl_prefill_512_128_20260130_122827.json) | - |
| 2026-01-30 12:27 | llama3.2-1b | opencl | prefill_128 | 256 | **1930.8** | 57.15 | 17.5 | 39.5 -> 39.9 (Max 39.9) | 3868 | [JSON](data/profile_opencl_prefill_128_256_20260130_122750.json) | - |
| 2026-01-30 12:26 | llama3.2-1b | opencl | prefill_128 | 128 | **2094.9** | 56.68 | 17.6 | 39.8 -> 39.4 (Max 39.8) | 4058 | [JSON](data/profile_opencl_prefill_128_128_20260130_122720.json) | - |
| 2026-01-30 12:25 | llama3.2-1b | cpu | prefill_1024 | 256 | **30560.0** | 91.47 | 10.9 | 38.6 -> 39.8 (Max 39.8) | 4057 | [JSON](data/profile_cpu_prefill_1024_256_20260130_122656.json) | - |
| 2026-01-30 12:24 | llama3.2-1b | cpu | prefill_512 | 256 | **10109.3** | 81.68 | 12.2 | 35.3 -> 37.1 (Max 37.1) | 4078 | [JSON](data/profile_cpu_prefill_512_256_20260130_122453.json) | - |
| 2026-01-30 12:24 | llama3.2-1b | cpu | prefill_1024 | 128 | **29553.7** | 93.86 | 10.7 | 37.2 -> 38.6 (Max 38.6) | 4091 | [JSON](data/profile_cpu_prefill_1024_128_20260130_122547.json) | - |
| 2026-01-30 12:23 | llama3.2-1b | cpu | prefill_512 | 128 | **11441.0** | 83.66 | 12.0 | 33.6 -> 35.3 (Max 35.3) | 4092 | [JSON](data/profile_cpu_prefill_512_128_20260130_122408.json) | - |
| 2026-01-30 12:23 | llama3.2-1b | cpu | prefill_128 | 256 | **2493.2** | 51.12 | 19.6 | 30.7 -> 30.7 (Max 30.7) | 4102 | [JSON](data/profile_cpu_prefill_128_256_20260130_122332.json) | - |
| 2026-01-30 12:22 | llama3.2-1b | cpu | prefill_128 | 128 | **2123.4** | 58.17 | 17.2 | 29.4 -> 29.4 (Max 29.4) | 4284 | [JSON](data/profile_cpu_prefill_128_128_20260130_122303.json) | - |
| 2026-01-29 09:04 | llama3.2-1b | opencl | short_len | 2048 | **160.9** | 94.73 | 10.6 | 37.9 -> 37.6 (Max 38.1) | 3850 | [JSON](data/profile_opencl_short_len_2048_20260129_090810.json) | [Graph](plots/profile_opencl_short_len_2048_20260129_090810.png) |
| 2026-01-29 08:59 | llama3.2-1b | cpu | short_len | 2048 | **202.3** | 138.41 | 7.2 | 27.8 -> 38.1 (Max 38.1) | 6598 | [JSON](data/profile_cpu_short_len_2048_20260129_090435.json) | [Graph](plots/profile_cpu_short_len_2048_20260129_090435.png) |
| 2026-01-29 08:16 | llama3.2-1b | opencl | short_len | 512 | **201.8** | 56.02 | 17.9 | 35.9 -> 37.1 (Max 37.1) | 4471 | [JSON](data/profile_opencl_short_len_512_20260129_081716.json) | [Graph](plots/profile_opencl_short_len_512_20260129_081716.png) |
| 2026-01-29 08:15 | llama3.2-1b | cpu | short_len | 512 | **159.3** | 55.17 | 18.1 | 32.4 -> 35.7 (Max 35.7) | 4992 | [JSON](data/profile_cpu_short_len_512_20260129_081627.json) | [Graph](plots/profile_cpu_short_len_512_20260129_081627.png) |
| 2026-01-29 08:12 | llama3.2-1b | opencl | short_len | 2048 | **151.1** | 74.44 | 13.4 | 25.2 -> 34.9 (Max 34.9) | 5717 | [JSON](data/profile_opencl_short_len_2048_20260129_081452.json) | [Graph](plots/profile_opencl_short_len_2048_20260129_081452.png) |
| 2026-01-29 08:10 | llama3.2-1b | opencl | short_len | 2048 | **N/A** | N/A | N/A | 0.0 -> 0.0 (Max 0.0) | 0 | [JSON](data/profile_opencl_short_len_2048_20260129_081041.json) | - |
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
