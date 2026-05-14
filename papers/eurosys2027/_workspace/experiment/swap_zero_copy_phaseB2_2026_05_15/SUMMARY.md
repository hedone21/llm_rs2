# LISWAP-8 Phase B-2: UMA Zero-Copy (2026-05-15, 부분 측정)

## 목적
Phase B α (cuMemAlloc 제거)가 bg_fetch와 거의 동등 (Δ ≤ ±12 ms)이라 driver lock 가설 약화. **cuMemcpyHtoDAsync 자체를 제거**하여 culprit 격리.

## 구현
- `LayerObjectPool` entry를 `CudaHostBuffer` (pinned + DEVICEMAP, UMA에서 host/device 같은 DRAM)로 변경
- `enqueue_write_into_async`에 host fast path 추가: dst.as_mut_ptr() non-null 시 `ptr::copy_nonoverlapping` 직접 (cuMemcpyHtoDAsync 호출 0)
- env: `LLMRS_SWAP_LAYER_POOL_ZERO_COPY=1`

## 결과 (Jetson Llama 3.1 8B, 50 tokens, 부분 측정)

| K | mode | n | active_avg | rest_tbt | rest_p99 | rest_max |
|---|---|---|---|---|---|---|
| 4 | baseline | 3 | **146.92** | 111.96 | 169.95 | 174.35 |
| 4 | bg_fetch | 3 | 309.35 | 114.89 | 358.22 | 364.45 |
| 4 | pool (α) | 3 | 311.12 | 117.29 | 372.60 | 392.44 |
| 4 | **pool_zc** | 3 | **425.37 ❌** | **136.48 ❌** | 494.92 | 504.32 |
| 8 | baseline | 3 | 149.21 | 105.74 | 160.17 | 171.46 |
| 8 | bg_fetch | 1 | 469.37 | 113.13 | 589.93 | 600.77 |

(K=8 pool/pool_zc, K=2/16/32 미측정 — 중단)

## 핵심 발견 — 가설 **반증**

**zero-copy가 가장 나쁨** (Δ pool_zc - bg_fetch active = +116 ms, K=4):
- cuMemcpyHtoDAsync staging copy가 main culprit이 **아님**
- cuMemAlloc + cuMemcpyAsync 모두 제거해도 forward regression 그대로
- idle window는 모든 mode에서 67-69 ms 일치 → CudaHostBuffer 자체 GPU access는 같음
- 즉 **swap-active 동안의 worker work 자체가 forward와 메모리 BW 경쟁**

## 진단 — UMA 메모리 컨트롤러 BW 경쟁

| Path | swap work | forward와 BW 경쟁 | active forward 증가 |
|---|---|---|---|
| baseline | main thread mmap_permute (CPU memcpy) | sub-batch wait로 phase-separate | +79 ms (smallest) |
| bg_fetch | worker DMA (cuMemcpyHtoDAsync) | DMA engine + GPU compute 경쟁 | +163 ms |
| pool (α) | 동일 (cuMemAlloc만 제거) | 동일 | +164 ms |
| **pool_zc** | worker pure CPU memcpy (4.5 GB → pinned host) | **CPU bus + memory controller 경쟁 (gpu read와 동일 BW)** | **+278 ms (worst)** |

## 폐기된 가설

| 가설 | 결과 |
|---|---|
| ❌ CUDA driver context lock 경쟁 (cuMemAlloc) | Pool path가 bg_fetch와 동등 → 반증 |
| ❌ cuMemcpyHtoDAsync staging copy (pageable mmap) | Zero-copy가 더 나쁨 → 반증 |
| ❌ cudarc 내부 stream sync | cudarc는 NON_BLOCKING stream, sync 없음 |
| ✅ **UMA 메모리 컨트롤러 BW 경쟁** | 데이터 일관성: worker work 종류별 BW 사용 비례하여 active forward 증가 |

## 다음 후보 hammer (Hammer D)

**mmap region에 `cuMemHostRegister(DEVICEMAP)` + swap은 ArcSwap alias만** (memcpy/copy 모두 0):
- secondary GGUF mmap을 한 번 GPU-readable로 register (4.5 GB pinned)
- 매 swap = LayerSlot의 weight tensor가 mmap layer offset alias 가리키게 ArcSwap
- worker work 자체가 0 → forward와 BW 경쟁 없음
- LISWAP-6 rpcmem alias 패턴을 cuda_embedded에 적용

근거: Phase B 데이터가 worker work 양과 active forward 증가가 비례한다 보여줌. work=0이면 forward 영향 0 예상.

위험: mmap pinned register cost (4.5 GB 한 번), GGUF Q4_0 byte layout이 GPU kernel과 직접 호환되어야 함 (Phase A에서 bg_fetch가 mmap bytes 그대로 device로 복사하므로 호환은 검증됨).

## 데이터 파일
- `k4_baseline_r{1,2,3}.{stdout,stderr,tbt.jsonl}` — main thread mmap baseline
- `k4_bg_fetch_r{1,2,3}` — Phase A bg_fetch
- `k4_pool_r{1,2,3}` — Phase B α (cuMemAlloc 제거)
- `k4_pool_zc_r{1,2,3}` — Phase B-2 zero-copy (CPU memcpy)
- `k8_baseline_r{1,2,3}` + `k8_bg_fetch_r1` — 일부 sanity
