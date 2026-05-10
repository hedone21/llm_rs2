# Phase 0 Baseline — Adreno 830 H2D Bandwidth + Extension Discovery

## Setup
- Device: Galaxy S25 (Snapdragon 8 Elite for Galaxy, Adreno 830)
- OpenCL Platform: QUALCOMM Snapdragon™
- Buffer: 600MB ALLOC_HOST_PTR, n=600 (script defaulted), profile_events ON

## H2D wall-clock (clEnqueueWriteBuffer blocking)

| metric | value |
|--------|------:|
| mean | 21.32 ms |
| median | 22.28 ms |
| p99 | 24.55 ms |
| stddev | 1.25 ms |
| σ/mean | 5.9% |
| effective BW | **27.49 GB/s** |

LPDDR5X 이론 84.8 GB/s 대비 32%만 사용. 22ms는 driver overhead(~14ms) + bytes(~7ms) 추정.

## Critical discovery

**기존 handoff "290ms H2D stall" 가설 부분 수정 필요**:
- 600MB H2D 자체는 **22ms** 만 차지
- 나머지 ~268ms는: Q4_0 변환 CPU 작업 + secondary mmap first-touch + ArcSwap commit + per-tensor cl_mem creation
- SVM fine-grain이 H2D 22ms는 우회 가능하나 268ms 다른 prep 비용은 그대로

## Adreno 830 OpenCL 미시도 path 검출 결과

### SVM (가장 큰 발견)
```
CL_DEVICE_SVM_CAPABILITIES = 0x0b
  COARSE_GRAIN_BUFFER : YES
  FINE_GRAIN_BUFFER   : YES   <-- 핵심
  FINE_GRAIN_SYSTEM   : no
  ATOMICS             : YES
```

**Phase 1 PROCEED**. Adreno 740 IWOCL 보고가 830에서도 유효함 확인.

### Command Buffer / Recordable
```
cl_khr_command_buffer              : no
cl_khr_command_buffer_mutable_dispatch : no
cl_qcom_recordable_queues          : YES   <-- vendor variant
```

**Phase 2 PROCEED via Qualcomm vendor variant**.

### ION/AHB
```
cl_qcom_ion_host_ptr               : no
cl_qcom_dma_buf_host_ptr           : no
cl_qcom_android_ahardwarebuffer_host_ptr : YES
cl_qcom_ahardwarebuffer_direct_import : YES   <-- bonus
cl_khr_external_memory             : YES
cl_khr_external_memory_dma_buf     : YES
```

**Phase 3 PROCEED via AHB path**. ION 미지원이지만 AHB가 모바일 forward-compatible path.

### 기타
- `cl_qcom_large_buffer` YES — large 600MB buffer 다른 처리 가능성
- `cl_qcom_onchip_global_memory` YES — small data path
- `cl_qcom_priority_hint` YES, `cl_qcom_perf_hint` YES — Phase 6 측정에 사용
- `cl_khr_il_program` no — SPIR-V 미지원 (Phase 1 부수 효과 작음)

## Phase 1/2/3 진행 결정

세 phase 모두 PROCEED. 우선순위:
1. **Phase 1 SVM**: 22ms H2D 자체 + Map/Unmap 비용 모두 우회 가능. 22ms 절감 가능.
2. **Phase 2 cl_qcom_recordable_queues**: enqueue overhead 측정 가치
3. **Phase 3 AHB + iocoherent**: cacheable+coherent 결합으로 DMA-BUF garbage 회피

---
2026-05-09 (Phase 0 완료, HEAD 미커밋)
