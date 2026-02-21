# 📊 LLM Research Benchmark Log

Last Updated: 2026-02-21 12:39:49

## Executive Summary
- **Total Benchmarks**: 114
- **Recent Run**: 2026-02-21 12:38

## Detailed Results

| Date | Model | Backend | Device | Eviction | Input | Tokens | FG App | TTFT (ms) | TBT (ms) | T/s | Temp (°C) | Mem (MB) | Proc Mem (MB) | Data | Plot |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-02-21 12:38 | models | cpu | Local PC | none | short_len | 128 | - | **1016.8** | 33.64 | 29.7 | 45.0 -> 70.0 (Max 70.0) | 8118 | 2066 | [JSON](data/profile_local_cpu_short_len_128_none_20260221_123905.json) | - |
| 2026-02-13 18:00 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_1024 | 128 | - | **17160.6** | 32.90 | 30.4 | 36.3 -> 39.1 (Max 39.1) | 3730 | 1931 | [JSON](data/profile_opencl_prefill_1024_20260213_180028.json) | - |
| 2026-02-13 17:58 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 128 | - | **199.6** | 20.49 | 48.8 | 31.4 -> 31.3 (Max 31.3) | 3724 | 2004 | [JSON](data/profile_opencl_baseline_128_20260213_175836.json) | - |
| 2026-02-13 17:58 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 2048 | - | **215.1** | 36.06 | 27.7 | 31.7 -> 36.3 (Max 36.7) | 3716 | 1814 | [JSON](data/profile_opencl_long_2048_20260213_175856.json) | - |
| 2026-02-13 17:56 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_1024 | 128 | - | **N/A** | N/A | N/A | 35.4 -> 34.6 (Max 35.3) | 3734 | 2547 | [JSON](data/profile_opencl_prefill_1024_20260213_175636.json) | - |
| 2026-02-13 17:55 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 2048 | - | **153.4** | 32.99 | 30.3 | 29.9 -> 35.7 (Max 36.4) | 3678 | 1842 | [JSON](data/profile_opencl_long_2048_20260213_175511.json) | - |
| 2026-02-13 17:54 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 128 | - | **133.8** | 20.55 | 48.7 | 29.6 -> 29.6 (Max 29.6) | 3693 | 2277 | [JSON](data/profile_opencl_baseline_128_20260213_175453.json) | - |
| 2026-02-13 17:54 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_1024 | 128 | - | **N/A** | N/A | N/A | 29.7 -> 29.8 (Max 29.8) | 3732 | 2253 | [JSON](data/profile_opencl_prefill_1024_20260213_175409.json) | - |
| 2026-02-13 17:53 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 128 | - | **N/A** | N/A | N/A | 29.0 -> 29.0 (Max 29.0) | 4317 | 1763 | [JSON](data/profile_opencl_baseline_128_20260213_175336.json) | - |
| 2026-02-13 17:53 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 2048 | - | **N/A** | N/A | N/A | 29.3 -> 29.5 (Max 29.5) | 3774 | 2185 | [JSON](data/profile_opencl_long_2048_20260213_175352.json) | - |
| 2026-02-13 17:40 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_1024 | 128 | - | **N/A** | N/A | N/A | 37.6 -> 37.3 (Max 37.3) | 4373 | 2302 | [JSON](data/profile_opencl_prefill_1024_20260213_174012.json) | - |
| 2026-02-13 17:39 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_1024 | 128 | - | **26868.4** | 80.04 | 12.5 | 34.6 -> 37.3 (Max 37.3) | 4366 | 1829 | [JSON](data/profile_cpu_prefill_1024_20260213_173920.json) | - |
| 2026-02-13 17:39 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 2048 | - | **N/A** | N/A | N/A | 35.3 -> 35.1 (Max 35.1) | 4543 | 2623 | [JSON](data/profile_opencl_long_2048_20260213_173901.json) | - |
| 2026-02-13 17:36 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 2048 | - | **169.6** | 71.10 | 14.1 | 28.7 -> 35.1 (Max 35.1) | 4514 | 1754 | [JSON](data/profile_cpu_long_2048_20260213_173619.json) | - |
| 2026-02-13 17:36 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 128 | - | **N/A** | N/A | N/A | 26.6 -> 26.6 (Max 26.6) | 4641 | 2531 | [JSON](data/profile_opencl_baseline_128_20260213_173604.json) | - |
| 2026-02-13 17:35 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 128 | - | **164.1** | 42.43 | 23.6 | 26.6 -> 26.6 (Max 26.6) | 5148 | 1750 | [JSON](data/profile_cpu_baseline_128_20260213_173542.json) | - |
| 2026-02-12 17:09 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_1024 | 128 | - | **30757.9** | 114.72 | 8.7 | 38.0 -> 39.8 (Max 39.8) | 4219 | 1876 | [JSON](data/profile_cpu_prefill_1024_128_20260212_171037.json) | - |
| 2026-02-12 17:08 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_512 | 128 | - | **10996.2** | 102.49 | 9.8 | 34.8 -> 36.8 (Max 36.8) | 4174 | 1933 | [JSON](data/profile_cpu_prefill_512_128_20260212_170846.json) | - |
| 2026-02-12 17:08 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_512 | 256 | - | **10974.5** | 91.43 | 10.9 | 36.8 -> 38.0 (Max 38.0) | 4245 | 1832 | [JSON](data/profile_cpu_prefill_512_256_20260212_170936.json) | - |
| 2026-02-12 17:07 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_128 | 128 | - | **2140.0** | 52.49 | 19.1 | 29.3 -> 29.3 (Max 29.3) | 4328 | 1768 | [JSON](data/profile_cpu_prefill_128_128_20260212_170734.json) | - |
| 2026-02-12 17:07 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_128 | 256 | - | **2187.1** | 64.24 | 15.6 | 29.9 -> 32.3 (Max 32.3) | 4131 | 1937 | [JSON](data/profile_cpu_prefill_128_256_20260212_170808.json) | - |
| 2026-02-12 15:50 | llama3.2-1b | ? | Galaxy S25 | none | custom_prompt | 128 | - | **N/A** | N/A | N/A | 29.3 -> 29.3 (Max 29.3) | 3972 | 1978 | [JSON](data/profile_cpu_process_test.json) | - |
| 2026-01-30 16:55 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 128 | Chrome | **331.0** | 66.63 | 15.0 | 37.8 -> 38.2 (Max 38.2) | 4204 | 0 | [JSON](data/profile_cpu_short_len_128_fg_chrome_20260130_165512.json) | - |
| 2026-01-30 16:55 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 128 | YouTube | **414.3** | 79.64 | 12.6 | 38.6 -> 38.7 (Max 38.7) | 4376 | 0 | [JSON](data/profile_cpu_short_len_128_fg_youtube_20260130_165547.json) | - |
| 2026-01-30 16:54 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 128 | YouTube | **219.2** | 50.03 | 20.0 | 35.6 -> 35.8 (Max 35.8) | 4522 | 0 | [JSON](data/profile_opencl_short_len_128_fg_youtube_20260130_165438.json) | - |
| 2026-01-30 16:25 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 256 | - | **207.2** | 53.51 | 18.7 | 33.9 -> 34.8 (Max 34.8) | 3884 | 0 | [JSON](data/profile_opencl_short_len_256_20260130_162539.json) | - |
| 2026-01-30 16:25 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 512 | - | **260.5** | 60.86 | 16.4 | 34.8 -> 36.2 (Max 36.2) | 3875 | 0 | [JSON](data/profile_opencl_short_len_512_20260130_162624.json) | - |
| 2026-01-30 16:24 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 128 | - | **203.0** | 49.79 | 20.1 | 33.9 -> 33.7 (Max 33.9) | 4015 | 0 | [JSON](data/profile_opencl_short_len_128_20260130_162512.json) | - |
| 2026-01-30 16:24 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 32 | - | **205.0** | 44.08 | 22.7 | 34.4 -> 34.5 (Max 34.5) | 4040 | 0 | [JSON](data/profile_opencl_short_len_32_20260130_162436.json) | - |
| 2026-01-30 16:24 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 64 | - | **201.1** | 47.37 | 21.1 | 34.1 -> 33.8 (Max 34.1) | 4006 | 0 | [JSON](data/profile_opencl_short_len_64_20260130_162452.json) | - |
| 2026-01-30 16:23 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 256 | - | **191.5** | 45.41 | 22.0 | 29.2 -> 30.9 (Max 30.9) | 4047 | 0 | [JSON](data/profile_cpu_short_len_256_20260130_162335.json) | - |
| 2026-01-30 16:23 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 512 | - | **190.8** | 61.97 | 16.1 | 30.9 -> 34.4 (Max 34.4) | 4056 | 0 | [JSON](data/profile_cpu_short_len_512_20260130_162419.json) | - |
| 2026-01-30 16:22 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 128 | - | **180.1** | 43.41 | 23.0 | 28.8 -> 28.8 (Max 28.8) | 4035 | 0 | [JSON](data/profile_cpu_short_len_128_20260130_162309.json) | - |
| 2026-01-30 16:22 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 32 | - | **177.4** | 37.77 | 26.5 | 28.3 -> 28.3 (Max 28.3) | 4271 | 0 | [JSON](data/profile_cpu_short_len_32_20260130_162236.json) | - |
| 2026-01-30 16:22 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 64 | - | **178.3** | 37.90 | 26.4 | 28.3 -> 28.8 (Max 28.8) | 4049 | 0 | [JSON](data/profile_cpu_short_len_64_20260130_162251.json) | - |
| 2026-01-30 14:59 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 128 | - | **243.8** | 57.96 | 17.3 | 43.4 -> 42.5 (Max 43.4) | 3381 | 0 | [JSON](data/profile_opencl_short_len_128_20260130_145932.json) | - |
| 2026-01-30 14:59 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 256 | - | **240.8** | 59.62 | 16.8 | 42.5 -> 42.1 (Max 42.5) | 3355 | 0 | [JSON](data/profile_opencl_short_len_256_20260130_150000.json) | - |
| 2026-01-30 14:58 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_1024 | 256 | - | **21404.9** | 79.34 | 12.6 | 43.1 -> 43.7 (Max 43.7) | 3647 | 0 | [JSON](data/profile_opencl_prefill_1024_256_20260130_145911.json) | - |
| 2026-01-30 14:57 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_1024 | 128 | - | **20239.6** | 89.02 | 11.2 | 42.3 -> 43.3 (Max 43.3) | 3697 | 0 | [JSON](data/profile_opencl_prefill_1024_128_20260130_145817.json) | - |
| 2026-01-30 14:56 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_512 | 128 | - | **7962.2** | 69.61 | 14.4 | 40.4 -> 41.5 (Max 41.5) | 3792 | 0 | [JSON](data/profile_opencl_prefill_512_128_20260130_145652.json) | - |
| 2026-01-30 14:56 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_512 | 256 | - | **7947.7** | 71.64 | 14.0 | 41.5 -> 42.3 (Max 42.3) | 3763 | 0 | [JSON](data/profile_opencl_prefill_512_256_20260130_145732.json) | - |
| 2026-01-30 14:55 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 256 | - | **184.9** | 48.27 | 20.7 | 39.4 -> 39.4 (Max 39.7) | 3832 | 0 | [JSON](data/profile_cpu_short_len_256_20260130_145527.json) | - |
| 2026-01-30 14:55 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_128 | 128 | - | **1920.5** | 57.79 | 17.3 | 40.1 -> 39.9 (Max 40.1) | 3837 | 0 | [JSON](data/profile_opencl_prefill_128_128_20260130_145552.json) | - |
| 2026-01-30 14:55 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_128 | 256 | - | **1902.3** | 58.27 | 17.2 | 39.9 -> 40.4 (Max 40.4) | 3805 | 0 | [JSON](data/profile_opencl_prefill_128_256_20260130_145622.json) | - |
| 2026-01-30 14:54 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 128 | - | **192.0** | 44.65 | 22.4 | 39.7 -> 39.3 (Max 39.8) | 3842 | 0 | [JSON](data/profile_cpu_short_len_128_20260130_145501.json) | - |
| 2026-01-30 14:53 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_1024 | 256 | - | **30371.4** | 84.07 | 11.9 | 38.4 -> 39.7 (Max 39.7) | 3863 | 0 | [JSON](data/profile_cpu_prefill_1024_256_20260130_145442.json) | - |
| 2026-01-30 14:52 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_1024 | 128 | - | **30113.5** | 92.22 | 10.8 | 37.1 -> 38.4 (Max 38.4) | 3853 | 0 | [JSON](data/profile_cpu_prefill_1024_128_20260130_145337.json) | - |
| 2026-01-30 14:52 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_512 | 256 | - | **9882.5** | 82.31 | 12.1 | 35.5 -> 37.1 (Max 37.1) | 3952 | 0 | [JSON](data/profile_cpu_prefill_512_256_20260130_145243.json) | - |
| 2026-01-30 14:51 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_512 | 128 | - | **11523.5** | 85.00 | 11.8 | 33.6 -> 35.5 (Max 35.5) | 3663 | 0 | [JSON](data/profile_cpu_prefill_512_128_20260130_145157.json) | - |
| 2026-01-30 14:50 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_128 | 128 | - | **2099.3** | 46.77 | 21.4 | 29.0 -> 29.0 (Max 29.0) | 3656 | 0 | [JSON](data/profile_cpu_prefill_128_128_20260130_145054.json) | - |
| 2026-01-30 14:50 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_128 | 256 | - | **2151.1** | 52.83 | 18.9 | 29.9 -> 32.1 (Max 32.1) | 3672 | 0 | [JSON](data/profile_cpu_prefill_128_256_20260130_145122.json) | - |
| 2026-01-30 14:27 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 128 | - | **212.5** | 52.70 | 19.0 | 41.6 -> 40.9 (Max 41.5) | 3487 | 0 | [JSON](data/profile_opencl_short_len_128_20260130_142740.json) | - |
| 2026-01-30 14:27 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 256 | - | **166.2** | 57.14 | 17.5 | 40.8 -> 40.6 (Max 40.8) | 3435 | 0 | [JSON](data/profile_opencl_short_len_256_20260130_142807.json) | - |
| 2026-01-30 14:26 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_1024 | 256 | - | **20010.2** | 91.42 | 10.9 | 41.6 -> 41.9 (Max 42.4) | 3436 | 0 | [JSON](data/profile_opencl_prefill_1024_256_20260130_142719.json) | - |
| 2026-01-30 14:25 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_1024 | 128 | - | **18893.5** | 91.97 | 10.9 | 40.1 -> 41.6 (Max 41.6) | 3701 | 0 | [JSON](data/profile_opencl_prefill_1024_128_20260130_142622.json) | - |
| 2026-01-30 14:25 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_512 | 256 | - | **7629.2** | 62.96 | 15.9 | 38.3 -> 39.8 (Max 39.8) | 3738 | 0 | [JSON](data/profile_opencl_prefill_512_256_20260130_142540.json) | - |
| 2026-01-30 14:24 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_128 | 256 | - | **1952.4** | 59.57 | 16.8 | 36.9 -> 37.5 (Max 37.5) | 3741 | 0 | [JSON](data/profile_opencl_prefill_128_256_20260130_142431.json) | - |
| 2026-01-30 14:24 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_512 | 128 | - | **7935.6** | 62.21 | 16.1 | 37.5 -> 38.3 (Max 38.3) | 3752 | 0 | [JSON](data/profile_opencl_prefill_512_128_20260130_142502.json) | - |
| 2026-01-30 14:23 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 256 | - | **254.9** | 53.59 | 18.7 | 36.7 -> 36.6 (Max 36.7) | 3779 | 0 | [JSON](data/profile_cpu_short_len_256_20260130_142334.json) | - |
| 2026-01-30 14:23 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_128 | 128 | - | **1949.5** | 56.27 | 17.8 | 36.8 -> 36.9 (Max 37.0) | 3789 | 0 | [JSON](data/profile_opencl_prefill_128_128_20260130_142359.json) | - |
| 2026-01-30 14:22 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 128 | - | **201.0** | 40.42 | 24.7 | 37.2 -> 36.7 (Max 37.2) | 3794 | 0 | [JSON](data/profile_cpu_short_len_128_20260130_142306.json) | - |
| 2026-01-30 14:21 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_1024 | 256 | - | **28792.8** | 93.20 | 10.7 | 36.2 -> 37.3 (Max 37.3) | 3784 | 0 | [JSON](data/profile_cpu_prefill_1024_256_20260130_142246.json) | - |
| 2026-01-30 14:20 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_1024 | 128 | - | **27731.7** | 92.37 | 10.8 | 34.7 -> 36.2 (Max 36.2) | 3834 | 0 | [JSON](data/profile_cpu_prefill_1024_128_20260130_142140.json) | - |
| 2026-01-30 14:20 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_512 | 256 | - | **9694.1** | 76.28 | 13.1 | 32.7 -> 34.5 (Max 34.5) | 3860 | 0 | [JSON](data/profile_cpu_prefill_512_256_20260130_142047.json) | - |
| 2026-01-30 14:19 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_128 | 256 | - | **2156.1** | 55.26 | 18.1 | 26.1 -> 29.0 (Max 29.0) | 3870 | 0 | [JSON](data/profile_cpu_prefill_128_256_20260130_141934.json) | - |
| 2026-01-30 14:19 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_512 | 128 | - | **9480.3** | 66.75 | 15.0 | 30.0 -> 30.7 (Max 30.7) | 3858 | 0 | [JSON](data/profile_cpu_prefill_512_128_20260130_142005.json) | - |
| 2026-01-30 14:18 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_128 | 128 | - | **2085.4** | 42.68 | 23.4 | 26.1 -> 26.1 (Max 26.1) | 4317 | 0 | [JSON](data/profile_cpu_prefill_128_128_20260130_141905.json) | - |
| 2026-01-30 13:16 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 128 | - | **N/A** | N/A | N/A | 32.5 -> 32.5 (Max 32.5) | 3559 | 0 | [JSON](data/profile_opencl_short_len_128_20260130_131612.json) | - |
| 2026-01-30 13:11 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 128 | - | **207.7** | 52.22 | 19.1 | 40.3 -> 39.3 (Max 39.9) | 3535 | 0 | [JSON](data/profile_opencl_short_len_128_20260130_131153.json) | - |
| 2026-01-30 13:11 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 256 | - | **202.2** | 53.61 | 18.7 | 39.3 -> 39.3 (Max 39.4) | 3465 | 0 | [JSON](data/profile_opencl_short_len_256_20260130_131219.json) | - |
| 2026-01-30 13:10 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_1024 | 256 | - | **18715.5** | 83.46 | 12.0 | 40.0 -> 40.4 (Max 40.9) | 3485 | 0 | [JSON](data/profile_opencl_prefill_1024_256_20260130_131132.json) | - |
| 2026-01-30 13:09 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_1024 | 128 | - | **17441.2** | 83.85 | 11.9 | 38.2 -> 39.6 (Max 39.6) | 3473 | 0 | [JSON](data/profile_opencl_prefill_1024_128_20260130_131038.json) | - |
| 2026-01-30 13:09 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_512 | 256 | - | **7880.0** | 64.12 | 15.6 | 36.9 -> 38.0 (Max 38.0) | 3667 | 0 | [JSON](data/profile_opencl_prefill_512_256_20260130_130956.json) | - |
| 2026-01-30 13:08 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_128 | 256 | - | **1933.7** | 59.76 | 16.7 | 35.3 -> 35.8 (Max 35.8) | 3756 | 0 | [JSON](data/profile_opencl_prefill_128_256_20260130_130849.json) | - |
| 2026-01-30 13:08 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_512 | 128 | - | **7999.5** | 63.13 | 15.8 | 35.8 -> 36.9 (Max 36.9) | 3771 | 0 | [JSON](data/profile_opencl_prefill_512_128_20260130_130919.json) | - |
| 2026-01-30 13:07 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 128 | - | **268.1** | 42.78 | 23.4 | 35.5 -> 35.0 (Max 35.5) | 4066 | 0 | [JSON](data/profile_cpu_short_len_128_20260130_130724.json) | - |
| 2026-01-30 13:07 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 256 | - | **181.2** | 58.00 | 17.2 | 34.8 -> 35.1 (Max 35.2) | 3816 | 0 | [JSON](data/profile_cpu_short_len_256_20260130_130754.json) | - |
| 2026-01-30 13:07 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_128 | 128 | - | **1929.3** | 57.18 | 17.5 | 35.8 -> 35.3 (Max 35.8) | 3803 | 0 | [JSON](data/profile_opencl_prefill_128_128_20260130_130819.json) | - |
| 2026-01-30 13:06 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_1024 | 256 | - | **27155.0** | 97.60 | 10.2 | 34.7 -> 35.6 (Max 35.6) | 3846 | 0 | [JSON](data/profile_cpu_prefill_1024_256_20260130_130704.json) | - |
| 2026-01-30 13:05 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_1024 | 128 | - | **23707.8** | 93.48 | 10.7 | 32.7 -> 34.8 (Max 34.8) | 3835 | 0 | [JSON](data/profile_cpu_prefill_1024_128_20260130_130558.json) | - |
| 2026-01-30 13:04 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_512 | 128 | - | **9529.6** | 55.94 | 17.9 | 28.6 -> 29.5 (Max 29.5) | 3813 | 0 | [JSON](data/profile_cpu_prefill_512_128_20260130_130429.json) | - |
| 2026-01-30 13:04 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_512 | 256 | - | **10135.2** | 70.56 | 14.2 | 30.6 -> 32.6 (Max 32.6) | 3823 | 0 | [JSON](data/profile_cpu_prefill_512_256_20260130_130510.json) | - |
| 2026-01-30 13:03 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_128 | 128 | - | **2067.4** | 45.45 | 22.0 | 23.0 -> 23.0 (Max 23.0) | 4124 | 0 | [JSON](data/profile_cpu_prefill_128_128_20260130_130332.json) | - |
| 2026-01-30 13:03 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_128 | 256 | - | **2120.0** | 51.60 | 19.4 | 24.3 -> 26.7 (Max 26.7) | 3848 | 0 | [JSON](data/profile_cpu_prefill_128_256_20260130_130400.json) | - |
| 2026-01-30 12:29 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_1024 | 128 | - | **20401.9** | 117.90 | 8.5 | 41.6 -> 43.0 (Max 43.0) | 3639 | 0 | [JSON](data/profile_opencl_prefill_1024_128_20260130_122956.json) | - |
| 2026-01-30 12:29 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_1024 | 256 | - | **20726.9** | 90.79 | 11.0 | 42.5 -> 43.1 (Max 43.4) | 3444 | 0 | [JSON](data/profile_opencl_prefill_1024_256_20260130_123052.json) | - |
| 2026-01-30 12:28 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_512 | 256 | - | **7536.3** | 74.02 | 13.5 | 41.0 -> 41.7 (Max 41.7) | 3644 | 0 | [JSON](data/profile_opencl_prefill_512_256_20260130_122907.json) | - |
| 2026-01-30 12:27 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_128 | 256 | - | **1930.8** | 57.15 | 17.5 | 39.5 -> 39.9 (Max 39.9) | 3868 | 0 | [JSON](data/profile_opencl_prefill_128_256_20260130_122750.json) | - |
| 2026-01-30 12:27 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_512 | 128 | - | **7975.9** | 106.33 | 9.4 | 40.0 -> 40.8 (Max 40.8) | 3866 | 0 | [JSON](data/profile_opencl_prefill_512_128_20260130_122827.json) | - |
| 2026-01-30 12:26 | llama3.2-1b | opencl | Galaxy S25 | none | prefill_128 | 128 | - | **2094.9** | 56.68 | 17.6 | 39.8 -> 39.4 (Max 39.8) | 4058 | 0 | [JSON](data/profile_opencl_prefill_128_128_20260130_122720.json) | - |
| 2026-01-30 12:25 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_1024 | 256 | - | **30560.0** | 91.47 | 10.9 | 38.6 -> 39.8 (Max 39.8) | 4057 | 0 | [JSON](data/profile_cpu_prefill_1024_256_20260130_122656.json) | - |
| 2026-01-30 12:24 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_1024 | 128 | - | **29553.7** | 93.86 | 10.7 | 37.2 -> 38.6 (Max 38.6) | 4091 | 0 | [JSON](data/profile_cpu_prefill_1024_128_20260130_122547.json) | - |
| 2026-01-30 12:24 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_512 | 256 | - | **10109.3** | 81.68 | 12.2 | 35.3 -> 37.1 (Max 37.1) | 4078 | 0 | [JSON](data/profile_cpu_prefill_512_256_20260130_122453.json) | - |
| 2026-01-30 12:23 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_128 | 256 | - | **2493.2** | 51.12 | 19.6 | 30.7 -> 30.7 (Max 30.7) | 4102 | 0 | [JSON](data/profile_cpu_prefill_128_256_20260130_122332.json) | - |
| 2026-01-30 12:23 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_512 | 128 | - | **11441.0** | 83.66 | 12.0 | 33.6 -> 35.3 (Max 35.3) | 4092 | 0 | [JSON](data/profile_cpu_prefill_512_128_20260130_122408.json) | - |
| 2026-01-30 12:22 | llama3.2-1b | cpu | Galaxy S25 | none | prefill_128 | 128 | - | **2123.4** | 58.17 | 17.2 | 29.4 -> 29.4 (Max 29.4) | 4284 | 0 | [JSON](data/profile_cpu_prefill_128_128_20260130_122303.json) | - |
| 2026-01-29 09:04 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 2048 | - | **160.9** | 94.73 | 10.6 | 37.9 -> 37.6 (Max 38.1) | 3850 | 0 | [JSON](data/profile_opencl_short_len_2048_20260129_090810.json) | - |
| 2026-01-29 08:59 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 2048 | - | **202.3** | 138.41 | 7.2 | 27.8 -> 38.1 (Max 38.1) | 6598 | 0 | [JSON](data/profile_cpu_short_len_2048_20260129_090435.json) | - |
| 2026-01-29 08:16 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 512 | - | **201.8** | 56.02 | 17.9 | 35.9 -> 37.1 (Max 37.1) | 4471 | 0 | [JSON](data/profile_opencl_short_len_512_20260129_081716.json) | - |
| 2026-01-29 08:15 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 512 | - | **159.3** | 55.17 | 18.1 | 32.4 -> 35.7 (Max 35.7) | 4992 | 0 | [JSON](data/profile_cpu_short_len_512_20260129_081627.json) | - |
| 2026-01-29 08:12 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 2048 | - | **151.1** | 74.44 | 13.4 | 25.2 -> 34.9 (Max 34.9) | 5717 | 0 | [JSON](data/profile_opencl_short_len_2048_20260129_081452.json) | - |
| 2026-01-29 08:10 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 2048 | - | **N/A** | N/A | N/A | 0.0 -> 0.0 (Max 0.0) | 0 | 0 | [JSON](data/profile_opencl_short_len_2048_20260129_081041.json) | - |
| 2026-01-23 16:04 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 2048 | - | **N/A** | N/A | N/A | 29.6 -> 35.2 (Max 35.2) | 3751 | 0 | [JSON](data/profile_cpu_short_len_2048_20260123_160655.json) | - |
| 2026-01-23 15:57 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 128 | - | **184.6** | 40.22 | 24.9 | 29.9 -> 31.5 (Max 31.5) | 3770 | 0 | [JSON](data/profile_opencl_short_len_128_20260123_155723.json) | - |
| 2026-01-23 15:56 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 128 | - | **182.9** | 36.72 | 27.2 | 29.5 -> 29.5 (Max 29.5) | 4185 | 0 | [JSON](data/profile_cpu_short_len_128_20260123_155657.json) | - |
| 2026-01-23 15:49 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 128 | - | **189.3** | 40.53 | 24.7 | 30.1 -> 30.1 (Max 30.1) | 4022 | 0 | [JSON](data/profile_opencl_short_len_128_20260123_154929.json) | - |
| 2026-01-23 15:48 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 128 | - | **175.1** | 35.65 | 28.1 | 29.2 -> 29.2 (Max 29.2) | 4275 | 0 | [JSON](data/profile_cpu_short_len_128_20260123_154905.json) | - |
| 2026-01-23 15:19 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 1024 | - | **199.8** | 90.10 | 11.1 | 36.9 -> 38.1 (Max 38.5) | 4138 | 0 | [JSON](data/profile_cpu_short_len_1024_20260123_152111.json) | - |
| 2026-01-23 15:17 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 1024 | - | **183.0** | 87.87 | 11.4 | 35.3 -> 36.9 (Max 37.2) | 4200 | 0 | [JSON](data/profile_cpu_short_len_1024_20260123_151928.json) | - |
| 2026-01-23 15:16 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 1024 | - | **182.5** | 78.15 | 12.8 | 30.7 -> 35.3 (Max 35.3) | 4150 | 0 | [JSON](data/profile_cpu_short_len_1024_20260123_151746.json) | - |
| 2026-01-23 15:09 | llama3.2-1b | opencl | Galaxy S25 | none | short_len | 1024 | - | **201.6** | 75.67 | 13.2 | 32.3 -> 35.4 (Max 35.4) | 4232 | 0 | [JSON](data/profile_opencl_short_len_1024_20260123_151054.json) | - |
| 2026-01-23 15:07 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 1024 | - | **180.2** | 76.52 | 13.1 | 28.1 -> 32.6 (Max 32.6) | 4336 | 0 | [JSON](data/profile_cpu_short_len_1024_20260123_150908.json) | - |
| 2026-01-23 14:38 | llama3.2-1b | cpu | Galaxy S25 | none | med_len | 64 | - | **38217.0** | 137.11 | 7.3 | 28.1 -> 33.0 (Max 33.0) | 4415 | 0 | [JSON](data/profile_cpu_med_len_64_20260123_143945.json) | - |
| 2026-01-23 14:31 | llama3.2-1b | cpu | Galaxy S25 | none | short_len | 32 | - | **181.2** | 37.17 | 26.9 | 28.5 -> 28.5 (Max 28.5) | 4561 | 0 | [JSON](data/profile_cpu_short_len_32_20260123_143133.json) | - |


## Graphical Analysis
Plots available in `results/plots/`. (Manually added below)
