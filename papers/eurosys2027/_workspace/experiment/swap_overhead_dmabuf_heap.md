# DMA-BUF Heap Zero-Copy Swap — Galaxy S25 측정 리포트

- 일자: 2026-05-08
- 디바이스: Samsung Galaxy S25 (R3CY408S4HN, Adreno 830, Snapdragon 8 Elite)
- 호스트 빌드: HEAD `0c8951e` + 추가 변경 (DMA-BUF heap path)
- 모델: Qwen2.5-1.5B-Instruct (primary F16 GGUF, secondary Q4_0 GGUF + AOS layout)
- 실행 환경: `/data/local/tmp/generate`, threads=6, OpenCL backend
- 시나리오 입력: `-p "The quick brown fox jumps" -n 200`, force-swap-ratio=0.9, n=3 runs/시나리오 (+ 1× RUST_LOG=info diagnostic)

## PASS/FAIL 판정

| 항목 | 결과 |
|---|---|
| DMA-BUF heap path 활성화 | PASS — `HostPtrPool: DMA-BUF heap path enabled (LLMRS_OPENCL_DMABUF_HEAP=1)` 로그 확인 |
| DMA-BUF alloc fallback 미발생 | PASS — `alloc failed` 메시지 없음 |
| **decode 정확성 (DMA-BUF)** | **FAIL — 4/4 garbage 출력 (multi-byte 깨짐, 무의미한 토큰)** |
| 사용자 가설 (sync_baseline ~26ms 회복) | **REJECT — 6중 negative 확정** |

**최종: FAIL.** DMA-BUF heap zero-copy path가 GPU 측 데이터 무결성을 보장하지 못함. 사용자 지시("garbage → 즉시 보고 + stop")에 따라 본 측정으로 종결.

## 측정 결과 (n=3 평균 ± 표준편차)

| 시나리오 | TTFT (ms) | Decode excl tok[0] (ms/tok) | tok[0] (ms) | Avg TBT (ms) | Decode 정확성 |
|---|---:|---:|---:|---:|---|
| sync_baseline (no swap-during-decode, force pre-swap) | 674.7 ± 93.6 | **26.63 ± 0.11** | 27.30 ± 0.57 | 30.31 ± 0.78 | OK (3/3) |
| liswap1_alloc_host_ptr | 383.4 ± 11.9 | 28.02 ± 0.12 | 48.50 ± 0.10 | 35.86 ± 0.99 | OK (3/3) |
| **liswap1_dmabuf** | 392.7 ± 5.8 | 28.15 ± 0.38 | 48.62 ± 0.35 | 36.34 ± 1.20 | **FAIL (0/3 — garbage)** |

### 핵심 비교 (사용자 표 양식)

| 시나리오 | swap-active TBT (tok[0]) | saturated TBT (excl tok[0]) | per-tick swap latency (median) |
|---|---:|---:|---:|
| sync_baseline | 27.30 ms | 26.63 ms | n/a (pre-swap, no incremental) |
| liswap1_alloc_host_ptr | 48.50 ms | 28.02 ms | 25.60 ms (mean 27.16) |
| **liswap1_dmabuf** | 48.62 ms | 28.15 ms | 26.30 ms (mean 29.10) |

판정: DMA-BUF는 alloc_host_ptr과 사실상 동일한 TBT 분포 — sync_baseline에 가까워지지 않음. Driver-level 직렬화는 깨지지 않았고, 추가로 정확성까지 fail.

## 정확성 검증 상세

### sync_baseline (run1) — coherent
> "The quick brown fox jumps over the lazy dog. The 'fox' is moved to a different position in this sentence. Sure, if you were referring to moving 'fox' to a different position..."

### liswap1_alloc_host_ptr (run1) — coherent
> "The quick brown fox jumps over the lazy dog. 21479883 is a palindromic number, so it can be written as '2147983'. The sum of its digits is 40..."

### liswap1_dmabuf (run1) — **garbage**
> "The quick brown fox jumps over the Gain fox " ogtek is, year This is quick every from一� repeating .......TV)..，, str \\\\x �,一大 –：， �水 Multip..."

### liswap1_dmabuf (run2) — **garbage**
> "The quick brown fox jumps over the �fox its a matrix jump r il sum2鹅... hum it ---, ---= m/ turtle. year如何 – on def"

### liswap1_dmabuf (run3) — **garbage**
> "The quick brown fox jumps over the gain fox � Sports5．.png rosa.世界This病)-lg儿,️后bank \\uff·左右_— pm-钛..."

### liswap1_dmabuf (info-run, RUST_LOG=info) — **garbage**
> "The quick brown fox jumps over \\ \" sum = \" - fem: if--一大\" --้า Door quần year ) + ε伙 A\" �1..."

→ 정확성 4/4 fail. 첫 ~6 토큰("The quick brown fox jumps over")까지 prompt + early decode 일치하다가, 첫 swap이 GPU에서 사용되기 시작한 시점부터 출력이 완전히 깨짐.

## 원인 가설 (Architect/Senior Implementer 분석 영역)

1. **DMA-BUF cache coherency 미보장**
   - DMA-BUF heap (system heap)은 cached CPU 매핑. CPU `memcpy_nonoverlapping` 후 `clFinish`/`Map`/`Unmap`/cache flush 없이 GPU가 사용하면 stale data를 읽을 수 있음.
   - Userspace `msync(MS_SYNC)` / `DMA_BUF_IOCTL_SYNC` (DMA_BUF_SYNC_START/END + DMA_BUF_SYNC_RW) 명시 호출이 필요할 가능성.
   - 가설 검증: `try_pool_materialise` 직접 memcpy 후 `ioctl(fd, DMA_BUF_IOCTL_SYNC, &start)` 추가 → 재측정 시 정확성 회복 여부 확인.

2. **`CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR`가 host pointer mmap과 자동 동기화 안 함**
   - KHR external memory는 GPU↔external 영역의 ownership만 정의. CPU mmap 매핑은 별도 cache hierarchy → driver가 인지 못함.
   - 결과: GPU는 export 시점의 cached/old pages를 보고, CPU memcpy는 다른 cache line에 머무름.

3. **per-tick latency 측정에서 dmabuf가 더 느린 이유**
   - alloc_host_ptr (median 25.6ms) vs dmabuf (median 26.3ms): 거의 동일.
   - 하지만 첫 measurement (RUST_LOG 없는 run에 가까운) 에서는 36~54ms로 관측되어 디바이스 state 의존성 큼.
   - mmap_permute 비슷, prefault만 dmabuf가 약간 큼 (mean 5.77 vs 0.11) — 새로 매핑된 DMA-BUF 페이지가 page fault 더 일으킴.

## 결론

**DMA-BUF heap zero-copy + KHR external_memory 경로는 본 환경(Adreno 830 / Snapdragon 8 Elite / Android 16)에서 작동하지 않음.**

- 정확성: 4/4 fail.
- 성능: alloc_host_ptr 대비 동등 또는 미세하게 느림. sync_baseline (~26ms)에는 도달하지 못함.
- 사용자 가설("driver libCB을 우회하고 hardware-level DMA-BUF coherency로 진짜 zero-copy") **REJECT**.
- Adreno OpenCL driver의 swap-active phase forward-swap 직렬화는 DMA-BUF 경로에서도 동일하게 발생.
- 추가로 DMA-BUF cached mmap ↔ GPU access 사이의 cache coherency가 명시적 sync 없이는 깨짐.

→ **6중 negative 확정.** 진정한 zero-copy + 정확성 양립은 본 driver 스택에서 옵션 제한적 (a) 명시적 `DMA_BUF_IOCTL_SYNC` 추가 후 재시도 b) `clEnqueueMapBuffer` 경로로 회귀 c) 다른 heap (qcom uncached/secure-non-pixel)으로 재시도).

## 산출물

- Raw 로그: `papers/eurosys2027/_workspace/experiment/dmabuf_heap_raw/`
  - `sync_baseline_run{1,2,3}.log`
  - `liswap1_alloc_host_ptr_run{1,2,3}.log`
  - `liswap1_dmabuf_run{1,2,3}.log`
  - `liswap1_alloc_host_ptr_info.log` (RUST_LOG=info, swap latency tick별)
  - `liswap1_dmabuf_info.log` (RUST_LOG=info, swap latency tick별 + DMA-BUF activation 로그)
  - `sanity_dmabuf.log` (n=10 sanity)

## 다음 단계 권고

1. (Senior Implementer) `try_pool_materialise` DMA-BUF path에 `ioctl(dmabuf_fd, DMA_BUF_IOCTL_SYNC, &(struct dma_buf_sync){.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE})` + `END` 페어 추가 → 재측정으로 정확성/성능 재평가.
2. (Researcher) Adreno KHR external_memory_dma_buf 사양/한계 조사. Mali/PowerVR 대비 cache coherency model 차이 확인.
3. (PM) 6중 negative 결과 누적 (sync_baseline / Pool ALLOC_HOST_PTR / Direction A / multi-queue / LISWAP-2/3/4 / DMA-BUF heap) — `arch/weight_swap.md` 또는 backlog에 종합 기록.
