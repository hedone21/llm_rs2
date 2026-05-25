# Phase E — QNN microbench matrix 통합 보고서

**작성**: 2026-05-26
**Sprint**: μ-Q1 (QNN HTP matmul microbench-first PoC)
**디바이스**: Galaxy S25 (`R3CY408S5SB`, SoC SM8750, Hexagon V79, Adreno 830)
**Shape**: Qwen 2.5-1.5B FFN gate `[1, 1536] × [1536, 8960]` (M=1 GEMV)

---

## TL;DR

`paper-grade` conclusion 4건 (모두 reproducible, CV<10%):

1. **HTP W8A8 (M7) = 모든 backend 중 최고** — 0.276 ms (OpenCL F16 raw 의 0.47×, **2.13× 빠름**)
2. **HTP F16 (M6b) = OpenCL F16 raw 보다 1.29× 빠름** — 0.456 vs 0.589 ms
3. **QNN-GPU OpPackage (M4) = worst on Adreno**, OpenCL raw 보다 2.60× 느림 — Phase R R-B1 결과 (2.4×) reproducible
4. **OpenCL Q4_0 production path** 와 직접 비교는 별 작업 (M5 latency 측정 bin 부재, μ-Q2 후속)

→ **NPU 가속의 ROI 큼** (특히 W8A8). Paper 의 외부 비교 (vs Executorch) 성립.

---

## 매트릭스 결과 (production rounds=10, warmup 3 + measure 7~10)

| Cell | Backend                       | dtype | median (ms) | CV (%) | n_valid | Status |
|------|-------------------------------|-------|-------------|--------|---------|--------|
| M3   | OpenCL raw (`mul_mv_f16_f32`) | F16   | **0.589**   | 1.91   | 7/7     | GREEN  |
| M4   | QNN-GPU OpPackage             | F16   | **1.529**   | 9.0    | 6/7     | YELLOW |
| M6b  | Executorch HTP                | F16   | **0.456**   | 3.14   | 7/7     | GREEN  |
| M7   | Executorch HTP                | W8A8  | **0.276**   | 7.11   | 7/7     | YELLOW |

- 측정 protocol: 4 thermal zone polling (cpu little/mid + gpuss-5/7 + nsphvx-0 + nsphmx-0 + ddr), 50°C trigger, 45-180s inter-cell cooldown, 300-600s inter-round cooldown, Tukey 1.5×IQR outlier rejection, taskset 0x3f (6T pinning)
- Peak thermal: 37.3°C (Phase C) / 33.4°C (Phase D) — trigger 50°C 충분히 여유
- Single-shot 재현 검증: M6b 0.452 ms (재현), M7 0.269 ms (재현) — production v2 와 1~2% 이내 일치

---

## Fair-pair 분석

### F16 row (직접 비교 가능)

| Backend             | median (ms) | ratio vs M3 |
|---------------------|-------------|-------------|
| M3 OpenCL raw       | 0.589       | 1.00× (baseline) |
| **M6b Executorch HTP** | **0.456**   | **0.77× (1.29× faster)** |
| M4 QNN-GPU OpPackage | 1.529       | 2.60× slower |

**결론**: F16 weight 일 때 HTP (Hexagon NPU) > OpenCL raw > QNN-GPU OpPackage.
- HTP 가 OpenCL 보다 29% 빠름 — 단순 GEMV 1536×8960 에서 NPU 의 INT16/FP16 path 가 Adreno GPU 보다 효율적.
- QNN-GPU OpPackage 는 production OpenCL kernel 을 wrap 한 것이라 단순한 GEMV 측정에서 framework overhead (op registry + graph dispatch) 가 직접 OpenCL 의 2.6배 비용.

### W8A8 row

| Backend                | median (ms) | ratio vs M3 (F16 baseline) |
|------------------------|-------------|----------------------------|
| **M7 Executorch HTP W8A8** | **0.276** | **0.47× (2.13× faster)** |
| M2 HTP W8A8 raw (#197) | (측정 미완) | — |

**결론**: W8A8 는 HTP 의 native sweet spot. INT8 dot product (Hexagon HVX) 가 GEMV 1.5K×9K 에 매우 효율적. Paper 측 "NPU 가속 + W8A8 quantization" path 의 paper 가치 큼.

### Production hot path 비교 (미완)

| Cell | Backend | dtype | median (ms) |
|------|---------|-------|-------------|
| M5   | OpenCL Q4_0 production | Q4_0 | **N/A** (latency 측정 bin 부재) |
| M3   | OpenCL F16 raw         | F16  | 0.589 |
| M7   | Executorch HTP W8A8    | W8A8 | 0.276 |

**미완 사항**: M5 의 Q4_0 latency 측정용 별 microbench bin 작성 (현재 `qnn_oppkg_matmul_q40_correct` 는 correctness only). Q4_0 vs W8A8 fair 비교의 핵심 — 후속 sprint μ-Q2.

---

## 환경 + 셋업 (재현용)

### Host
- **OS**: Arch Linux (Ubuntu 22.04 권장이지만 host 영향 minimal)
- **Python**: 3.12.13 (uv venv) — 시스템 3.14.5 비호환 (Executorch <3.14 요구)
- **PyTorch**: 2.12.0+cpu (PyPI, executorch dependency)
- **GCC**: 16.1.1, CMake 4.2.3, NDK 29.0.14206865

### Executorch
- **Repo**: `/home/go/Workspace/executorch` (shallow clone --depth 1 --recurse-submodules)
- **venv**: `/home/go/Workspace/executorch/.venv` (uv)
- **install**: `uv pip install executorch` (3.12 호환 wheel)
- **QNN SDK**: 2.37.0 (Executorch auto-download → `.venv/lib/.../sdk/qnn`)
- **build**: `bash backends/qualcomm/scripts/build.sh --skip_x86_64 --release`
  (x86_64 host build 의 `pytorch_tokenizers` cmake fail 회피)

### AOT (.pte) build
```bash
cd papers/eurosys2027/_workspace/experiment/qnn_microbench_phase_d
source /home/go/Workspace/executorch/.venv/bin/activate
python build_pte_matmul.py f16   # → matmul_f16.pte
python build_pte_matmul.py w8a8  # → matmul_w8a8.pte
```

### Device (S25)
- **Bin location**: `/data/local/tmp/executorch/`
- **Files**:
  - `qnn_executor_runner` (Executorch binary, 50 MB)
  - `libqnn_executorch_backend.so`
  - `libQnnHtp.so` / `libQnnSystem.so` / `libQnnHtpV79Stub.so` / `libQnnHtpV79.so` / `libQnnHtpV79Skel.so` / `libQnnHtpPrepare.so` (QNN SDK 2.37)
  - `matmul_f16.pte` (27.5 MB) / `matmul_w8a8.pte` (13.2 MB)
  - `input_0_0.raw` (1×1536 F32, seed=42) / `input_list.txt`
- **Run**:
  ```bash
  cd /data/local/tmp/executorch
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. taskset 3f \
    ./qnn_executor_runner --model_path matmul_f16.pte \
                          --input_list_path input_list.txt \
                          --warm_up 3 --iteration 10
  ```

### Matrix runner
- **Script**: `scripts/microbench_qnn_matrix.py`
- **Phase C** (M3/M4): `--cells M3,M4 --rounds 10 --out papers/.../qnn_microbench_phase_c_2026_05_26`
- **Phase D** (M6b/M7): `--cells M6b,M7 --rounds 10 --out papers/.../qnn_microbench_phase_d_prod_v2`

---

## 미완 / 후속 (μ-Q2 또는 별 sprint)

### 본 sprint cap 초과 항목

1. **M1/M1b/M2 (HTP raw via libloading)** — #194/#195 segfault risk 미확정. Executorch path (M6b/M7) 가 작동했으므로 HTP 자체는 정상이지만, raw QNN API path 별 검증 필요.
2. **M5 (OpenCL Q4_0 latency 측정 bin)** — production hot path latency 측정. `qnn_oppkg_matmul_q40_correct` 는 correctness only.
3. **Per-cell paper figure** — bar chart with error bar (CV).

### Phase D 의 environmental risk (재현 시 주의)

- **Arch Linux + GCC 16 + CMake 4**: pytorch_tokenizers cmake fail. `--skip_x86_64` 로 회피.
- **Python 3.14 비호환**: uv venv 3.12 강제.
- **NDK 29 vs 권장 26c**: 빌드 성공했으나 추후 deprecation warning.
- **QNN SDK 2.37 download size 1.5GB**: venv 안 격리.

---

## 결정 게이트 (μ-Q1.2)

| 측정 결과 | 후속 sprint |
|----------|------------|
| HTP W8A8 (M7) < 50% of OpenCL F16 (M3) 시 | **Q-2 진입 — Backend trait `engine/src/backend/qnn_htp/` 신설** |
| **0.276 / 0.589 = 47%** ✅ | → Q-2 진입 조건 충족 |

**다음 sprint Q-2 진입**: HTP backend를 본 프로젝트 `engine/` 안에 native integrate. Executorch 의 lower-to-edge path 가 아닌 raw QNN HTP API path (M1/M1b/M2 의 실측 결과 + Phase D 의 paper 비교 base) 로.

---

## 산출물 매핑

| Phase | Commit | 산출 |
|---|---|---|
| A.1 | `a76e4a79` | `scripts/microbench_qnn_matrix.py` |
| A.3 | `7cdd69cf` | dry-run protocol 검증 |
| C   | `708d1b52` | M3/M4 production 측정 |
| D   | `ec2e5d8e` | Executorch + M6b/M7 single-shot |
| D.4 v2 | (next) | M6b/M7 production rounds=10 |
| E   | (next) | 본 통합 보고서 + handoff |

---

## 측정 raw 데이터 위치

- Phase C: `papers/eurosys2027/_workspace/experiment/qnn_microbench_phase_c_2026_05_26/`
- Phase D single-shot: `papers/eurosys2027/_workspace/experiment/qnn_microbench_phase_d/`
- Phase D production v2: `papers/eurosys2027/_workspace/experiment/qnn_microbench_phase_d_prod_v2/`

각 디렉토리에 `raw/`, `aggregated.csv`, `thermal_log.csv`, `report.md`, `env.json`.
