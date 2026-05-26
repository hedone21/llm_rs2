# QNN microbench matrix 결과

- 생성: 2026-05-26 02:59:56
- Shape: K=1536, N=8960 (Qwen 2.5-1.5b FFN gate)
- Trial protocol: warmup 3 + measure 10

## 매트릭스 결과

| Cell | dtype | backend | n_valid | n_outlier | median (ms) | CV (%) | max_abs_err | cosine | status |
|---|---|---|---|---|---|---|---|---|---|
| M3 | f16 | opencl | 7/7 | 0 | 0.589 | 1.91 | None | None | GREEN |
| M4 | f16 | qnn-gpu | 6/7 | 1 | 1.5285 | 9.0 | None | None | YELLOW |

## Fair-pair 분석

### FP32 row

- **M1**: SKIPPED or FAILED
- **M6**: SKIPPED or FAILED

### F16 row

- **M1b**: SKIPPED or FAILED
- **M3**: 0.589 ms (CV 1.91%)
- **M4**: 1.5285 ms (CV 9.0%)
- **M6b**: SKIPPED or FAILED

### W8A8 row

- **M2**: SKIPPED or FAILED
- **M7**: SKIPPED or FAILED

### Production ref row

- **M5**: SKIPPED or FAILED
