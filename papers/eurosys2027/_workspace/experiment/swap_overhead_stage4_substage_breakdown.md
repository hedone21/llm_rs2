# Stage 4 sub-stage breakdown — LLMRS_SWAP_PROFILE_BREAKDOWN=1

- **Date**: 2026-05-08 (instrumentation 추가 후 S25 측정 예정)
- **HEAD**: instrumentation commit (이 파일과 함께 커밋됨)
- **Predecessor**: `swap_overhead_stage4_zero_copy_v2.md` — Stage 4 v2 결과 (CONDITIONAL FAIL, perf gate ❌)
- **Author**: Implementer agent

## TL;DR

이 리포트는 측정 후 채워질 template입니다. 측정 완료 시 `python3 /tmp/parse_substage_breakdown.py` 스크립트로 자동 생성됩니다.

가설: `materialise_tensor` 내 mmap_permute stage 시간의 **88%가 GPU upload(copy_weight_from) 외 CPU-side overhead**라는 가설을 sub-stage timer로 검증.

## §1 측정 조건

### Device
- Galaxy S25 `R3CY408S4HN` (SM-S931N), Adreno 830, threads=6.

### Model & CLI
- Primary: `qwen2.5-1.5b-f16.gguf`
- Secondary: `qwen2.5-1.5b-q4_0-aos.auf`
- `--prompt "The quick brown fox jumps" -n 40 --threads 6 --backend opencl --temperature 0.0 --secondary-layout aos --force-swap-ratio 0.9`

### Scenarios
- **sync**: `--swap-incremental-per-tick=1 --force-swap-ratio=0.9` × 5 runs
- **zero-copy**: `--swap-incremental-per-tick=1 --force-swap-ratio=0.9 --swap-zero-copy --swap-pool-slots=14` + env `LLMRS_OPENCL_HOST_PTR_POOL=1` × 5 runs

### Activation
- env `LLMRS_SWAP_PROFILE_BREAKDOWN=1` — 각 tensor마다 `[swap-prof]` 라인을 stderr에 출력.

### Parse command
```bash
# 측정 후 실행:
python3 /tmp/parse_substage_breakdown.py \
    --sync /data/local/tmp/substage_sync/run_*.log \
    --zc   /data/local/tmp/substage_zc/run_*.log \
    --out  /tmp/substage_report.md
```

## §2 Sync incremental — sub-stage 분포

*(측정 후 채움)*

## §3 Zero-copy incremental — sub-stage 분포

*(측정 후 채움)*

## §4 비교 (sync vs zero-copy)

*(측정 후 채움)*

## §5 결론 — 88% 가설 confirm/refute

*(측정 후 채움)*

---

## Instrumentation 세부 사항

### env-gate 동작

```
LLMRS_SWAP_PROFILE_BREAKDOWN=1   → profiling ON  (Instant::now() 호출 발생)
LLMRS_SWAP_PROFILE_BREAKDOWN=0   → profiling OFF (default)
env var 없음                      → profiling OFF
```

Production 환경에서 env var 미설정 시 `Instant::now()` 호출이 전혀 없어 overhead = 0.

### 출력 형식

```
[swap-prof] layer=0 sub=attn_q.weight is_weight=1 size=1572864 lookup=4.2 dim=0.1 bytes=0.1 permute=0.0 wrap=2.3 cpu=0.5 upload=850.0 total=857.2
```

각 필드:
- `layer`: decoder layer index (0-based)
- `sub`: tensor subname (e.g., `attn_q.weight`, `ffn_gate.weight`)
- `is_weight`: 1=weight tensor, 0=norm tensor
- `size`: bytes
- `lookup` ~ `upload`: 각 sub-stage 소요 μs
- `total`: 함수 진입~반환 총 μs

### sub-stage 정의

| sub-stage | 측정 범위 |
|---|---|
| `lookup` | `secondary.layer_tensor(...)` — mmap index lookup |
| `dim` | shape 역순 변환 + dims 비교 |
| `bytes` | `secondary.tensor_bytes(info)` — slice 반환 (mmap page access 포함 가능) |
| `permute` | `needs_qk_unpermute_at_swap()` + (해당 시) `unpermute_qk_rows` |
| `wrap` | `BorrowedMmapBuffer::new` 또는 `SharedBuffer::from_vec` |
| `cpu` | `CpuBackend::new()` + `Tensor::new()` |
| `upload` | `try_pool_materialise` (zero-copy path) 또는 `copy_weight_from`/`copy_from` (sync path) |
| `total` | 함수 진입~반환 전체 |

### 주의 사항

- `bytes` stage는 mmap slice 반환으로 보이지만 실제 page fault (OS 메모리 매핑)가 이 구간에서 발생할 수 있음. 따라서 bytes stage 시간이 크다면 mmap page fault가 dominant임.
- `wrap` stage에서 `BorrowedMmapBuffer::new`는 mmap 바이트에 실제 접근을 안 하므로 짧아야 함. 반면 `SharedBuffer::from_vec`은 `unpermute_qk_rows` 결과 Vec을 래핑하므로 역시 짧아야 함.
- `upload` stage가 크면 GPU H2D transfer가 bottleneck.
- CPU-side (lookup+dim+bytes+permute+wrap+cpu) 합계 vs upload 비율로 88% 가설을 판정.
