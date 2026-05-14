# LISWAP-8 Phase B α: Layer Object Pool (2026-05-15)

## 목적
LayerObjectPool로 worker thread `cuMemAlloc`을 dispatch path에서 제거 → driver context lock contention 가설 검증.

## 구현 (commit 대기)
- `engine/src/models/weights/layer_object_pool.rs` — pool + background allocator thread
- `engine/src/core/backend.rs` — `enqueue_write_into_async` trait method
- `engine/src/backend/cuda_embedded/mod.rs` — CudaDeviceBuffer downcast + `copy_from_host_async`
- `engine/src/models/weights/swap_executor.rs` — bg_fetch closure 안에서 pool entry branch
- env: `LLMRS_SWAP_LAYER_POOL=1`, `LLMRS_SWAP_LAYER_POOL_DEPTH=2`

## Fix 적용 후 결과 (Jetson Llama 3.1 8B, 50 tokens)

| K | baseline fwd | bg_fetch fwd | pool fwd | bg_fetch active | pool active | Δ pool-bg active |
|---|---|---|---|---|---|---|
| 2  | 90.50 | 115.19 | 114.24 | 215.54 | 209.49 | -6.05 |
| 4  | 79.42 | 108.63 | 105.83 | 316.56 | 304.51 | **-12.05** |
| 8  | 72.31 |  99.45 | 101.09 | 463.13 | 474.97 | +11.84 |
| 16 | 70.65 |  87.62 |  90.24 | 534.41 | 658.32 | +123.91 (1 rep noise) |
| 32 | 69.40 |  69.15 |  67.16 | 152.23 | 150.78 | -1.45 |

## 가설 검증 결과

**cuMemAlloc 회피만으로는 의미 있는 개선 없음** (Δ ≤ ±12 ms, K=4에서 4% 개선이 maximum).

K=4 pool active 305 ms vs baseline 146 ms = +159 ms regression 중 12 ms (7.5%)만 alloc 영향.

남은 88% regression은 다른 원인:
- 가설 A: `cuMemcpyHtoDAsync` driver staging copy (mmap = pageable source)
- 가설 B: UMA memory bandwidth 경쟁 (DMA engine vs SM)
- 가설 C: cudarc 내부 stream synchronization

## 다음 단계: Phase B-2 zero-copy (진행 중)

- Pool entry를 `CudaHostBuffer` (pinned + DEVICEMAP)로 변경
- Worker는 CPU memcpy 직접 (no `cuMemcpyHtoDAsync`)
- driver lock 호출 0 + DMA copy 0
- 가설 A/B/C 모두 같이 검증
