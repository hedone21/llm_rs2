# Swap PreBuild PoC (2026-05-14)

## 목적
Jetson Llama 8B F16+Q4 swap에서 mmap_permute (1741 ms @ K=32)을 forward와 hide하는 방법 탐색.

## 핵심 발견

### 1. mmap_permute가 stall의 진짜 원인
- 함수 이름 `build_layer_from_mmap_async`이지만 **CPU side mmap_permute는 main thread blocking**.
- "async"의 의미: GPU H2D enqueue만 non-blocking. CPU memcpy + permute는 직렬.
- K=32 baseline: mmap_permute=1741 ms 동안 forward 완전 정지 → tok[0] TBT 2084 ms.

### 2. 가설별 검증 결과

| 가설 | 검증 | 결과 |
|---|---|---|
| page eviction (첫 N step만 무거움) | tok-by-tok 분포 | ❌ 50 step 균등 |
| ε CUDA driver lock 경쟁 | par_build + force_drain | ❌ forward 그대로 무거움 |
| δ rayon thread idle pollution | rayon이 baseline에서도 활성 | 약함 |
| **multi-thread mmap memcpy 자체** | **par_seq (iter)** | **✓ 확정** |

### 3. PoC v2 결정적 데이터

| run | Build mode | Decode forward | mmap_permute | Avg TBT |
|---|---|---|---|---|
| baseline (no prebuild) | none | **67.67 ms** ✓ | 1670 ms | 108.31 |
| par_build (par_iter) | **multi-thread** | **143.22 ms** ❌ | 134 ms | 151.61 |
| par_seq (iter) | **single-thread** | **68.77 ms** ✓ | 1577 ms | 133.21 |

**결론**: multi-thread CPU memcpy + cuMemcpyAsync가 CUDA UMA의 page placement state를 변경 → forward 영구 +67 ms 패널티. **single-thread는 영향 없음** (par_seq에서 dispatcher_pending=31에서도 forward 정상).

## 진행 방향

### 폐기
- multi-thread mmap_permute (rayon par_iter / std::thread::scope 모두 동일 위험)
- chunk 분할 H2D

### 추구
**single background thread로 mmap_permute 진행 (= "I/O로 미리 layer 하나 fetch, fetch 완료되면 layer swap")**
- 1 dispatcher worker thread가 mmap_permute → enqueue_write_async → wait_event → arc_swap → release를 sequential하게
- main thread는 forward 진행, dispatch submit만 함 (수 ms)
- 단일 thread라 forward에 영향 없음 (par_seq 데이터로 확정)
- 이론적 hide: 1577 ms mmap을 50 tokens × 67 ms = 3350 ms forward 안에 fit 가능

## 적용된 PoC 코드 변경 (모두 env gate, default OFF)

| env | 변경 위치 | 동작 |
|---|---|---|
| `LLMRS_SUB_BATCH_NO_WAIT=1` | swap_executor.rs:531 | sub-batch reactive wait 비활성 |
| `LLMRS_SWAP_PAR_BUILD=1` | swap_executor.rs:548 | rayon par_iter로 chunk pre-build + cache lookup |
| `LLMRS_SWAP_SEQ_BUILD=1` | swap_executor.rs:570 (par_build 안에서) | par_iter 대신 sequential iter (single-thread) |
| (자동) force_drain in par_build | swap_executor.rs:1000 | par_build 시 dispatcher + release_worker drain |

## 다음 PoC — single-thread background fetch

### 출발점 코드 위치
1. `swap_executor.rs:1113` `build_layer_from_mmap_async` — main thread blocking의 본체. 이걸 worker로 옮겨야.
2. `swap_executor.rs:1282` `materialise_cpu_tensor` — self method 호출 추적 필요.
3. `swap_executor.rs:1389` `try_alias_materialise` (opencl) — 별도.
4. `async_swap.rs:92` `ChunkDispatchJob` — **closure-based `Box<dyn FnOnce() + Send + 'static>` 패턴 이미 존재**. 활용 가능.
5. `async_swap.rs:99` `SwapJob::DispatchChunk` — closure dispatch 이미 무자비.

### 핵심 분석 질문
1. `build_layer_from_mmap_async` + `materialise_cpu_tensor`가 `self.memory` 사용? (`&dyn Memory` 'a borrow)
2. `self.config` 사용? (`&'a ModelConfig`, Clone 가능)
3. `self.host_ptr_pool` 사용? (Arc, 캡처 OK)
4. self method 호출을 standalone fn / inline closure로 추출 가능?

### 두 갈래 길
- **B-light** (~150 LOC): standalone fn으로 build path 추출 + closure 캡처. `SwapJob::DispatchChunk` 활용. SwapExecutor 'a 유지.
- **B-full** (~300 LOC + spec D-4 갱신): `SwapExecutor` 'static화. 12 caller 수정. closure에서 `Arc<SwapExecutor>` 자유.

B-light가 PoC에 적합.

### 작업 단위 (사용자 의도)
```
chunk loop:
  for layer in chunk:
    dispatcher.submit_dispatch_chunk(Box::new(move || {
      let (layer, evt) = build_layer_standalone(secondary, slot, layer_idx, backend, config_clone);
      backend.wait_event_blocking(&evt);
      slot.swap_weights(Arc::new(layer), new_dtype);
      // release chain
    }));
  // main thread immediately returns, continues forward
```

### 메모리 spike 안전
- dispatcher pending depth가 grow할 수 있음 (현재 par_seq에서 31까지 확인됨)
- in-flight layer 개수 × 45 MB = max memory pressure
- 32 layers × 45 MB = 1.4 GB extra. Jetson 64GB 안전, S25 12GB은 모니터 필요.

## 관련 데이터 파일
- `k32_baseline_v3.{stdout,stderr,tbt.jsonl}` — 정상 baseline (forward 67 ms)
- `k32_par_build_v3.{stdout,stderr,tbt.jsonl}` — par_iter multi-thread (forward 143 ms ❌)
- `k32_par_seq_v3.{stdout,stderr,tbt.jsonl}` — iter single-thread (forward 68 ms ✓, dispatcher_pending=31)
- `k1_async.tbt.jsonl` (이전 측정 set) — 기존 async hide 동작 비교용
