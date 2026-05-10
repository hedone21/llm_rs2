# Stage 4 sub-stage breakdown — LLMRS_SWAP_PROFILE_BREAKDOWN=1

- **Date**: 2026-05-08
- **HEAD**: `e8cbb3f` (feat(swap): env-gated sub-stage breakdown timer for materialise_tensor)
- **Device**: Galaxy S25 `R3CY408S4HN` (SM-S931N), Adreno 830, threads=6
- **Predecessor**: `swap_overhead_stage4_zero_copy_v2.md` (CONDITIONAL FAIL, perf gate ❌)
- **Author**: Implementer agent

## TL;DR

**upload stage가 total의 99.9%를 차지한다 (`copy_weight_from` = GPU H2D transfer).**
CPU-side overhead(lookup+dim+bytes+permute+wrap+cpu) 합계는 0.1 μs = total의 0.1%에 불과하다.

**가설 "88% CPU-side" — REFUTED (완전히 반박됨)**. 실제는 GPU upload가 99.9%이며, CPU-side는 무시할 수준이다.

## §1 측정 조건

### Device & Model
- Galaxy S25 `R3CY408S4HN` (SM-S931N), Adreno 830, threads=6
- Primary: `qwen2.5-1.5b-f16.gguf`
- Secondary: `qwen2.5-1.5b-q4_0-aos.auf`

### CLI
```
--prompt "The quick brown fox jumps" -n 40 --threads 6 --backend opencl
--temperature 0.0 --secondary-layout aos --force-swap-ratio 0.9
--swap-incremental-per-tick 1
```
- **sync**: 위 CLI × 5 runs
- **zero-copy**: 위 CLI + `--swap-zero-copy --swap-pool-slots 14` + env `LLMRS_OPENCL_HOST_PTR_POOL=1` × 5 runs

### Activation
- env `LLMRS_SWAP_PROFILE_BREAKDOWN=1` — per-tensor `[swap-prof]` 라인 stdout+stderr에 출력.
- sync 5 runs × 175 tensors/run = 875 records.
- zero-copy 5 runs × 175 tensors/run = 875 records.

## §2 Sync incremental — sub-stage 분포

| Stage | mean (μs) | p50 (μs) | p95 (μs) | max (μs) | % of total |
|---|---:|---:|---:|---:|---:|
| lookup | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% |
| dim | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% |
| bytes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% |
| permute | 0.1 | 0.0 | 1.0 | 2.0 | 0.0% |
| wrap | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% |
| cpu | 0.0 | 0.0 | 0.0 | 1.0 | 0.0% |
| **upload** | **2425.2** | **898.0** | **5078.0** | **8798.0** | **99.9%** |
| total | 2427.0 | 902.0 | 5079.0 | 8801.0 | 100.0% |

n=875 tensors (5 runs × 175 tensors/run).

**관찰**: upload 분포는 이중 분포를 보인다. p50=898 μs (norm/small weight), p95=5078 μs (ffn_gate/up/down 7.7 MB). 이는 tensor 크기에 따른 H2D transfer 시간 차이이다.

## §3 Zero-copy incremental — sub-stage 분포

| Stage | mean (μs) | p50 (μs) | p95 (μs) | max (μs) | % of total |
|---|---:|---:|---:|---:|---:|
| lookup | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% |
| dim | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% |
| bytes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% |
| permute | 0.1 | 0.0 | 1.0 | 3.0 | 0.0% |
| wrap | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% |
| cpu | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% |
| **upload** | **2338.6** | **895.0** | **5022.0** | **8847.0** | **99.9%** |
| total | 2340.5 | 898.0 | 5023.0 | 8850.0 | 100.0% |

n=875 tensors (5 runs × 175 tensors/run).

## §4 비교 (sync vs zero-copy)

| Stage | sync mean | zc mean | Δ (zc−sync) | Δ% |
|---|---:|---:|---:|---:|
| lookup | 0.0 | 0.0 | +0.0 | 0.0% |
| dim | 0.0 | 0.0 | +0.0 | 0.0% |
| bytes | 0.0 | 0.0 | +0.0 | 0.0% |
| permute | 0.1 | 0.1 | +0.0 | +5.5% |
| wrap | 0.0 | 0.0 | +0.0 | 0.0% |
| cpu | 0.0 | 0.0 | −0.0 | −100.0% |
| **upload** | **2425.2** | **2338.6** | **−86.6** | **−3.6%** |
| total | 2427.0 | 2340.5 | −86.6 | −3.6% |

**핵심 관찰**:
1. zero-copy pool path(`ALLOC_HOST_PTR map/memcpy/unmap`)는 sync staging path(`copy_weight_from`) 대비 upload stage를 −3.6% 줄였다.
2. CPU-side stage 차이는 noise 수준(p50=0, 측정 분해능 1μs에서 관찰 불가).
3. mmap_permute (기존 IncrementalSwap 단위 측정)에서 보였던 17 ms 스테이지가 어디 속하는지: upload만이 실질적 시간을 차지하므로, mmap_permute = upload + (무시할 CPU-side) 임이 확인됨.

## §5 결론 — 88% 가설 confirm/refute

| 가설 | 내용 | 결과 |
|---|---|---|
| **88% CPU-side** | lookup+dim+bytes+permute+wrap+cpu ≥ 88% of total | **REFUTED** |
| 실제 분포 | CPU-side = **0.0%**, upload = **99.9%** | |

**sync**: CPU-side = 0.0% → 가설 **REFUTED**
**zero-copy**: CPU-side = 0.0% → 가설 **REFUTED**

### 해석

Stage 4 v2 리포트에서 "mmap_permute 시간의 88%가 CPU-side overhead"라는 가설이 등장한 배경은, mmap_permute가 ~17 ms이고 upload(copy_weight_from)가 단독으로 GPU queue를 통과하는 것이 보이지 않았기 때문이다. 실측 결과:

- `secondary.tensor_bytes(info)` (bytes stage) — AUF AOS 형식에서 mmap slice 반환이라 page fault는 있을 수 있으나, 측정상 <1 μs (분해능 한계, 실제 ~수십 ns).
- `BorrowedMmapBuffer::new` / `unpermute_qk_rows` (wrap/permute stage) — <1 μs 수준.
- `copy_weight_from` (upload stage) — **실제 GPU H2D transfer**: 수백~수천 μs (tensor 크기에 선형 비례).

즉 mmap_permute ≈ upload이며, CPU-side는 총 비용의 <0.1%이다.

### Production saving 가능 sub-stage

현재 upload = 99.9%이므로:
- **upload stage 단축이 유일한 실질적 최적화 축**이다.
- zero-copy pool path는 −3.6% (86 μs/tensor 평균) 절약에 그침 → 25 layers × 7 tensors = 175 tensors × 86 μs = ~15 ms total 절약 (전체 swap plan ~750 ms에서 ~2%).
- CPU-side (lookup/bytes/permute/wrap)에는 투자할 가치가 없다.

### Direction A track 종결 근거 재확인

zero-copy pool path의 이론적 최대 절약 = upload time (100% upload이므로, 만약 upload를 0으로 줄일 수 있다면 최대 100% 절약). 실제 pool path는 staging upload를 완전히 없애는 게 아니라 `map/memcpy/unmap`으로 대체하므로, 그 차이(−3.6%)만 절약된다. Direction A의 30% saving gate 도달 불가 근거: pool path는 staging의 GPU memcpy 비용 자체를 없애지 못한다 — ALLOC_HOST_PTR 메모리도 GPU가 읽을 때는 PCIe 또는 UMA copy가 발생한다.

---

## Appendix — raw artefacts

| Artefact | Path |
|---|---|
| Sync logs (5 runs) | `/tmp/substage_raw/substage_sync/run_{1..5}.log` |
| Zero-copy logs (5 runs) | `/tmp/substage_raw/substage_zc/run_{1..5}.log` |
| Parse script | `/tmp/parse_substage_breakdown.py` |
| Sweep script | `/tmp/run_substage_sweep.sh`, `/data/local/tmp/run_substage_sweep.sh` |

### Reproducer

```bash
# Build + deploy (from llm_rs2 root)
python3 scripts/run_device.py -d galaxy_s25 generate --skip-exec
adb -s R3CY408S4HN push target/aarch64-linux-android/release/generate /data/local/tmp/generate

# Run sweep (on device)
adb -s R3CY408S4HN push /tmp/run_substage_sweep.sh /data/local/tmp/run_substage_sweep.sh
adb -s R3CY408S4HN shell "chmod 755 /data/local/tmp/run_substage_sweep.sh && rm -rf /data/local/tmp/substage_sync /data/local/tmp/substage_zc && /data/local/tmp/run_substage_sweep.sh"

# Pull logs
mkdir -p /tmp/substage_raw
adb -s R3CY408S4HN pull /data/local/tmp/substage_sync /tmp/substage_raw/
adb -s R3CY408S4HN pull /data/local/tmp/substage_zc /tmp/substage_raw/

# Analyze
python3 /tmp/parse_substage_breakdown.py \
    --sync /tmp/substage_raw/substage_sync/run_*.log \
    --zc   /tmp/substage_raw/substage_zc/run_*.log
```
