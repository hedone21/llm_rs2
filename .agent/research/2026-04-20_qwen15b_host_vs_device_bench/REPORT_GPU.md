# Qwen2.5-1.5B GPU bench — Galaxy S25 (2026-04-20, updated)

## 환경
S25 (`R3CY408S5SB`, Adreno 830), 6T, `-p 128 -n 128 -ctk f16 -ctv f16`, greedy, n=3. llama.cpp `983df14-dirty` OpenCL (`GGML_OPENCL=ON`+`ADRENO_KERNELS`+`EMBED_KERNELS`). llm.rs 수치 재활용. max ≤ 38 °C gate.

## GPU 백엔드 검증 (llama.cpp)
stderr: `selected platform: 'QUALCOMM Snapdragon(TM)'` / `device: 'QUALCOMM Adreno(TM) 830'` / `driver: OpenCL 3.0 QUALCOMM 0800.40.1` / `using kernels optimized for Adreno`. 표 `backend=OpenCL`, `ngl=99`. 벤더 `/vendor/lib64/libOpenCL.so` 우선; 푸시한 fallback loader 미사용.

## 결과 (n=3)

| 모델 | 엔진 | prefill tok/s | decode tok/s |
|------|------|--------------:|-------------:|
| F16  | llama.cpp-GPU | **202.80 ± 0.82** | **18.67 ± 0.04** |
| F16  | llm.rs-GPU    | 127.73 ± 0.21 | 14.80 ± 0.00 |
| Q4_0 | llama.cpp-GPU | **615.22 ± 3.97** | **32.43 ± 1.96** |
| Q4_0 | llm.rs-GPU    | 115.03 ± 0.35 | 27.83 ± 0.06 |

## 4방향 비교

| 모델·지표 | cpp-CPU | cpp-GPU | llmrs-CPU | llmrs-GPU |
|-----------|--------:|--------:|----------:|----------:|
| F16 prefill  | 108.49±3.63 | 202.80±0.82 | 132.93±1.80 | 127.73±0.21 |
| F16 decode   | 19.76±0.03  | 18.67±0.04  | 17.80±0.26  | 14.80±0.00  |
| Q4_0 prefill | 62.06±12.28 | 615.22±3.97 | 61.93±1.47  | 115.03±0.35 |
| Q4_0 decode  | 32.35±0.16  | 32.43±1.96  | 30.03±1.00  | 27.83±0.06  |

## Δ% (CPU→GPU, 엔진 간)

| 지표 | cpp CPU→GPU | llmrs CPU→GPU | CPU(llmrs/cpp) | GPU(llmrs/cpp) |
|------|-----------:|--------------:|---------------:|---------------:|
| F16 prefill  | +87.0  | −3.9  | +22.5 | **−37.0** |
| F16 decode   | −5.5   | −16.9 | −9.9  | **−20.7** |
| Q4_0 prefill | +891.3 | +85.8 | −0.2  | **−81.3** |
| Q4_0 decode  | +0.2   | −7.3  | −7.2  | **−14.2** |

llama.cpp GPU 전환 시 prefill 폭증 (Q4_0 ×9.9, F16 ×1.87) — Adreno fused dequant+GEMM 효과 추정. llm.rs는 Q4_0 prefill만 유의 개선 (+85.8 %), 나머지 경로 CPU 대비 퇴행.

## 온도 (°C, max before/after)
F16 cpp-GPU 33.8→57.6, Q4_0 34.2→55.7.

## 제약
llama.cpp stdev는 `-r 3` 내부 반복. llm.rs는 개별 프로세스 3회. SELinux 우회 불요. 잔존 프로세스 없음.

## Raw (신규)
`llamacpp_gpu_{f16,q4_0}_run1.log`, `thermal_gate_llamacpp_gpu_{01,02}.log`. `REPORT_GPU.md`·`results_summary_gpu.csv` 갱신. llm.rs 기존 산출물 미수정.
