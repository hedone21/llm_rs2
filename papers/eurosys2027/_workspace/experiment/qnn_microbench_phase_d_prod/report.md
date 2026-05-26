# QNN microbench matrix 결과

- 생성: 2026-05-26 04:26:40
- Shape: K=1536, N=8960 (Qwen 2.5-1.5b FFN gate)
- Trial protocol: warmup 3 + measure 10

## 매트릭스 결과

| Cell | dtype | backend | n_valid | n_outlier | median (ms) | CV (%) | max_abs_err | cosine | status |
|---|---|---|---|---|---|---|---|---|---|
| M6b | f16 | executorch | 0/0 | - | - | - | - | - | FAILED |
| M7 | w8a8 | executorch | 0/0 | - | - | - | - | - | FAILED |

## Fair-pair 분석

### FP32 row

- **M1**: SKIPPED or FAILED
- **M6**: SKIPPED or FAILED

### F16 row

- **M1b**: SKIPPED or FAILED
- **M3**: SKIPPED or FAILED
- **M4**: SKIPPED or FAILED
- **M6b**: SKIPPED or FAILED

### W8A8 row

- **M2**: SKIPPED or FAILED
- **M7**: SKIPPED or FAILED

### Production ref row

- **M5**: SKIPPED or FAILED
