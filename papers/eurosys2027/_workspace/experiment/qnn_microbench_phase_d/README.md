# Phase D Executorch HTP measurement

## 환경
- Host: Arch Linux + Python 3.12 (uv venv) + PyTorch 2.12.0+cpu + Executorch (PyPI)
- QNN SDK 2.37.0 (Executorch auto-download to .venv/lib/.../sdk/qnn)
- Target: S25 R3CY408S5SB (SM8750, V79)
- Build: `bash backends/qualcomm/scripts/build.sh --skip_x86_64 --release`
- AOT (.pte): `python build_pte_matmul.py {f16,w8a8}`

## 단일 trial 측정 결과 (qnn_executor_runner internal warmup=3 iteration=10)
- M6b (HTP F16): avg **0.4515 ms** (vs OpenCL F16 raw 0.589 ms = **1.30× 빠름**)
- M7 (HTP W8A8): avg **0.2694 ms** (vs OpenCL F16 raw 0.589 ms = **2.19× 빠름**)

## 후속
- matrix runner (scripts/microbench_qnn_matrix.py) 의 M6b/M7 cell 활성화 후 
  rounds=10 production 측정 (CV<5% 확인)
- M5 (OpenCL Q4_0) latency 측정 bin 별 작성 — Q4_0 vs W8A8 fair pair
