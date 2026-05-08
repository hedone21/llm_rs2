# Phase 3 — cl_qcom_ahardwarebuffer_host_ptr + iocoherent **NUANCED**

## Setup
- Device: Galaxy S25 (Adreno 830)
- Probe: `cl_qcom_android_ahardwarebuffer_host_ptr : YES`, `cl_qcom_ahardwarebuffer_direct_import : YES`
- AHB allocation: `AHardwareBuffer_allocate` with `FORMAT_BLOB`,
  `USAGE_GPU_DATA_BUFFER | USAGE_CPU_READ_OFTEN | USAGE_CPU_WRITE_OFTEN`
- cl_mem: `clCreateBuffer(USE_HOST_PTR | EXT_HOST_PTR_QCOM, size=0, &cl_mem_ahardwarebuffer_host_ptr{...})`
  - allocation_type = 0x4119 (`CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM`)
  - host_cache_policy = 0x40A9 (`CL_MEM_HOST_IOCOHERENT_QCOM`)

## 측정 결과 (n=30, 25MB single layer)

| Test | mean | median | stddev | σ/mean | BW |
|------|-----:|-------:|-------:|-------:|---:|
| **Host write (memcpy → AHB)** | 1.60 ms | 1.52 ms | 0.21 ms | 13.0% | **15.25 GB/s** |
| **GPU kernel read (busy accum)** | 21.04 ms | 21.21 ms | 1.65 ms | 7.8% | 1.16 GB/s |
| **Correctness** | — | — | — | — | **PASS (8/8 outputs finite + value-correct)** |

## 비교 — Phase 0 ALLOC_HOST_PTR baseline (600MB)

| Test | ALLOC_HOST_PTR | AHB+iocoherent | 비율 |
|------|---------------:|---------------:|-----:|
| Host write BW | 27.49 GB/s | 15.25 GB/s | -45% |
| Single-shot 600MB | ✓ 가능 | ✗ AHardwareBuffer_allocate err=-129 (NO_MEMORY) | — |
| Correctness | OK | OK | — |

## 핵심 분석

### 결정적 발견 1: Cache coherency 해결
- 기존 DMA-BUF heap path (`LLMRS_OPENCL_DMABUF_HEAP=1`)는 garbage 출력했음
- **AHB + IOCOHERENT cache policy는 host write를 GPU에 cache-coherent로 visible하게 함**
- 8개 output 모두 finite + 기대값 일치 (busy accumulator 결과 단조 증가)
- 수동 `DMA_BUF_IOCTL_SYNC` 없이도 동작

### 결정적 발견 2: 600MB 단일 할당 불가
- `AHardwareBuffer_allocate(600MB BLOB)` returns err=-129 (NO_MEMORY)
- Android의 GPU heap은 BLOB 단일 allocation에 ~256MB 제한 있음 (단말 의존)
- production swap에 통합하려면 layer-by-layer 25MB chunk로 분할 필요
  - 25 layers × 1.5ms = 37.5ms vs ALLOC_HOST_PTR 22ms → **-70% slower** (chunk overhead)

### 결정적 발견 3: GPU read throughput 저하
- GPU busy accumulator kernel: 1.16 GB/s (vs LPDDR5X peak 84.8 GB/s = -98.6%)
- ALLOC_HOST_PTR과 직접 비교는 부재 (별도 벤치 필요), 그러나 SVM (0.4 GB/s)보다는 3x 빠름
- **Adreno driver가 AHB-backed cl_mem를 GPU L2 cache에 효과적으로 staging하지 않음**
- 추정 원인: ext_host_ptr 메모리는 KGSL UVA 매핑이 cacheable하지만 GPU access마다 sniff cost

## Production 통합 평가

### 통합 시 trade-off
- Pros:
  - cache coherent (DMA-BUF 이슈 회피)
  - host write 빠름 (15.25 GB/s, 25MB layer = 1.5ms)
  - GPU 직접 access 가능 (Map/Unmap 불필요)
- Cons:
  - **600MB 단일 할당 불가** → 25개 AHB 핸들 관리 필요 (메모리 단편화 + 핸들 lifecycle 복잡)
  - **GPU read 23x 느림** → 매 forward inference마다 GPU 부담 증가, **production stop-gap**
  - AHB ↔ cl_mem ↔ Rust ownership 3중 라이프사이클 (drop order 중요)

### 결론
**Phase 3 SKIP integration**.

- AHB는 cache coherency 문제를 해결하지만, GPU read 저하가 **forward path**까지 영향을 미침
- swap 자체는 해결되더라도 inference TBT가 -23x 악화 가능성 → net negative
- Production 권장 아님

### 단, paper 가치
- "Adreno에서 AHB+iocoherent는 동작하지만 GPU read penalty가 zero-copy 이득을 상쇄"
- DMA-BUF heap garbage 원인 = "cache coherency 미보장"이 정확히 확정 (AHB+iocoherent로 해결)
- vendor extension 분석 매트릭스에서 정량 데이터 추가

## Phase 3 결과 표

| Path | alloc 성공 | host write BW | GPU read BW | correctness | swap 적용성 |
|------|:---:|:-:|:-:|:---:|:-----------:|
| ALLOC_HOST_PTR (Phase 0) | ✓ | 27.5 GB/s | (untested) | OK | **production main** |
| SVM fine-grain (Phase 1) | ✓ | 17.1 GB/s | 0.4 GB/s | OK | ✗ (200x slow GPU) |
| AHB + IOCOHERENT (Phase 3, 25MB) | ✓ | 15.25 GB/s | 1.16 GB/s | OK | ✗ (chunk + GPU slow) |
| AHB + IOCOHERENT (600MB) | ✗ NO_MEMORY | — | — | — | ✗ (단일 할당 불가) |
| DMA-BUF heap (이전 세션) | ✓ | (similar) | (similar) | **garbage** | ✗ (coherency 미보장) |

---
2026-05-09 (Phase 3 완료)
