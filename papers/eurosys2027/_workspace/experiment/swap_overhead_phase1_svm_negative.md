# Phase 1 — SVM Fine-Grain Buffer **NEGATIVE**

## Setup
- Device: Galaxy S25 (Adreno 830)
- Probe: `CL_DEVICE_SVM_FINE_GRAIN_BUFFER : YES`, `CL_DEVICE_SVM_ATOMICS : YES`
- Allocation: `clSVMAlloc(CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, 600MB, align=64)`
- Allocation 성공 (svm_ptr non-null), driver가 reject 안 함

## 측정 결과 (n=30, 600MB)

| Test | mean | median | p99 | BW | comment |
|------|-----:|-------:|----:|---:|---------|
| Phase 0 ALLOC_HOST_PTR H2D | 21.32 | 22.28 | 24.55 | 27.49 GB/s | baseline |
| **SVM host write (memcpy)** | 34.21 | 29.50 | 166.33 | **17.13 GB/s** | -33% vs baseline |
| **SVM GPU kernel read** | 1458 | 1594 | 1722 | **0.40 GB/s** | **-98% (200x 느림)** |
| SVM combined | 1607 | 1677 | 1802 | 0.36 GB/s | kernel dominate |

## 핵심 분석

### Host write side (-33%)
- ALLOC_HOST_PTR: `clEnqueueWriteBuffer` (driver가 page pinning + DMA setup 후 memcpy)
- SVM: 직접 `memcpy` host→svm
- **SVM이 더 느림**: driver의 pinned page 최적화 vs raw memcpy 차이
- σ/mean 71% — SVM은 swap 큰 페이지 변동 큼

### GPU kernel read side (-98%, 200x slower)
- 이론 lower bound: 7ms (84.8 GB/s peak)
- 실측: 1594 ms (0.4 GB/s)
- **Adreno 830 driver는 fine-grain SVM을 native HW로 처리하지 않음**
- 의심 원인:
  1. SVM이 host RAM에 상주 → GPU read마다 system bus traversal
  2. ATOMICS bit triggers per-access coherency check
  3. driver가 fine-grain을 coarse-grain coercion + 매 dispatch마다 implicit Map/Unmap
  4. GPU L2 cache가 SVM region에 비활성화

### 결론
**Adreno 830 SVM fine-grain은 "spec compliance wrapper"**
- `clGetDeviceInfo`가 FINE_GRAIN_BUFFER + ATOMICS yes 보고
- `clSVMAlloc`이 success
- 그러나 GPU 접근 throughput이 200x 저하
- production weight swap에 적용 시 wall-clock이 290ms → 1600+ms으로 **5x 악화**

## Phase 1 결정: SKIP integration (RESERVED for paper)

- swap_executor에 SVM path 통합하지 **않음**
- env-gate `LLMRS_OPENCL_SVM=1`도 추가하지 않음 (production 경로에 위험)
- 본 negative finding은 paper Section 4의 핵심 contribution
- 추가 측정: SVM coarse-grain (`CL_MEM_SVM_COARSE_GRAIN_BUFFER`)도 같은 결과인지 비교 (선택적, future work)

## Paper에 대한 함의

기존 negative finding 9 트랙은 "Adreno multi-queue serialize 또는 driver-level FIFO"가 본질이라고 추정했다.
Phase 1은 다른 차원의 finding을 추가한다:

> **모바일 GPU의 OpenCL spec compliance는 performance compliance를 의미하지 않는다.**
> Adreno 830은 SVM fine-grain + atomics를 보고하지만 실제 GPU 접근이
> 200x 저하된다. 이는 vendor extension 분석만으로 path 선정이 위험함을 보여준다.

이 finding은 본 paper의 "Adreno OpenCL stack에서 우회 가능한 path를
exhaustive 분석"이라는 contribution을 더 강하게 만든다.

---
2026-05-09 (Phase 1 완료)
