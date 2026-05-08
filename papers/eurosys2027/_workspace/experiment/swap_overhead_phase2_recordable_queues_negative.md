# Phase 2 — cl_qcom_recordable_queues **NEGATIVE**

## Setup
- Device: Galaxy S25 (Adreno 830)
- Probe: `cl_qcom_recordable_queues : YES`
- Recordable queue: `clCreateCommandQueue(... CL_QUEUE_RECORDABLE_QCOM=1<<30)`
- Recording lifecycle: `clNewRecordingQCOM` → 1× NDRange → `clEndRecordingQCOM` → `clEnqueueRecordingQCOM × N`
- Kernel: `layer_op` (4096 floats, 32-iter accumulator) — small/fast dispatch (~10us)

## 측정 결과 (n_iters=30, n_dispatches=100/iter)

| Method | mean | median | σ/mean | per-dispatch |
|--------|-----:|-------:|-------:|-------------:|
| **A: standard `clEnqueueNDRangeKernel × N`** | 0.93 ms | 0.92 ms | 9.0% | **9.3 us** |
| **B: recordable `clEnqueueRecordingQCOM × N`** | 3.18 ms | 3.20 ms | 5.6% | **31.8 us** |
| Speedup B/A | — | — | — | **0.29x (3.4x SLOWER)** |

## 핵심 분석

### 결과 설명
- 표준 NDRange enqueue: 9.3 us/dispatch — Adreno가 dispatch overhead를 잘 처리
- Recording replay: 31.8 us/replay — driver가 replay마다 추가 유효성 검증/dispatch 변환
- **recordable_queues가 표준 enqueue보다 빠르다는 가설은 거짓**

### 왜 recordable_queues가 더 느린가
가설 (확정 아님):
1. driver가 replay 시 cl_array_arg_qcom 검증을 매번 수행 (실제 호출에선 빈 배열이지만 검증 자체가 cost)
2. Adreno KGSL submission이 record submit ⊃ standard submit이 아닐 가능성
3. recording state에서 kernel 인자 변경 추적 cost (used flag 등)

### Swap에 적용 시 가치
**Zero**.

논거:
- 25 layer × 16 tensor × 1 dispatch ≈ 400 dispatches per swap
- 400 dispatches × 9.3 us = **3.7 ms** (current standard, baseline)
- 400 dispatches × 31.8 us = 12.7 ms (recordable replay)
- 290 ms 중 dispatch는 1.3% (3.7ms) — recordable로 0%로 만들어도 **0% saving**
- 게다가 recordable이 더 느려서 -3% 회귀

따라서 swap path에 통합 불가.

### llama.cpp / MLC LLM도 비슷한가?
가능성:
- llama.cpp Adreno backend의 weight swap 미지원이라 평가 불가
- recordable_queues는 inference의 hot kernel (~수천 dispatch/token)에 적용 시 효과
- swap은 dispatch가 적어 본 path 효과 없음

## 결론

**Adreno cl_qcom_recordable_queues는**:
- ✓ API 지원, 정상 동작
- ✗ standard NDRange보다 3.4x 느림 (1× kernel record 기준)
- ✗ swap path에 무용

**Phase 2 SKIP integration**. Production 코드 미수정.

이 finding은 paper에 추가 가치:
> Adreno 830 supports cl_qcom_recordable_queues, but per-dispatch
> latency is 3.4x worse than standard `clEnqueueNDRangeKernel` for
> short kernels. For weight swap (~400 dispatches at 9us each = 3.7 ms),
> NDRange enqueue overhead is already negligible (1.3% of 290ms total),
> so recordable_queues offers no meaningful saving even if it were faster.

---
2026-05-09 (Phase 2 완료)
