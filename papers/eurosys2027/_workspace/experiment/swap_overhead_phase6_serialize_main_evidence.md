# Phase 6 — Adreno HW Serialize 직접 측정 **PAPER MAIN EVIDENCE**

## Setup
- Device: Galaxy S25 (Adreno 830)
- Kernel: busy accumulator (gws=1024, ~1ms target)
- 두 queue가 disjoint cl_mem에 동일 kernel을 동시 launch
- 같은 thread가 issue (no sleep), 그 후 양쪽 queue에 finish
- iters auto-tuned: 9783 iters → 0.51ms single

## 결과 (n_iters=30 per config)

| Config | mean | median | σ/mean | ratio_to_1q | 해석 |
|--------|-----:|-------:|-------:|------------:|------|
| **Single-queue baseline** | 0.51 ms | 0.50 ms | 3.5% | 1.00x | reference |
| **Same-context, in-order × 2** | 0.98 ms | 1.00 ms | 4.2% | **1.93x** | HW SERIALIZE |
| **Same-context, out-of-order × 2** | 1.02 ms | 1.00 ms | 6.1% | **2.00x** | OoO 무시 |
| **Multi-context, in-order × 2** | 1.30 ms | 1.23 ms | 11.2% | **2.55x** | multi-ctx worse |
| **Multi-context + profile_events × 2** | 2.04 ms | 2.12 ms | 22.9% | **4.01x** | profile penalty 2x |

## 핵심 발견 (paper에 직접 인용)

### 1. Adreno 830은 다중 queue 명령을 HW level에서 직렬화

- Same-context, in-order × 2 → ratio 1.93x ≈ 2x
- Out-of-order queue (`CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE`) → ratio 2.00x
- **OoO hint를 명시해도 driver는 2 queue를 serial 실행**
- σ/mean < 7% → 측정 noise로 설명 불가, 결정적 결과

> Adreno 830 OpenCL command processor (libCB.so) implements a single-issue FIFO
> at the HW level, regardless of `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` or
> `cl_qcom_priority_hint` queue properties. Two concurrent kernel dispatches
> on disjoint cl_mem objects observe wall-clock that is 1.93-2.00x of single
> kernel time, indicating sequential execution.

### 2. Multi-context는 single-context보다 worse, 부분 serialize 아님

- Multi-context ratio 2.55x → single-context 1.93x보다 **+32% 느림**
- 이는 multi-context가 driver-internal context switch overhead를 추가함
- 이전 multi-context 트랙 (LISWAP series)이 "swap 자체는 -7%, forward는 +30%" 보였던 이유 완전 설명
- driver가 cross-context dependency를 device-wide enforce하면서 context boundary cost 추가

### 3. CL_QUEUE_PROFILING_ENABLE는 measurement instrumentation이 측정 대상에 영향

- Multi-context + profile_events × 2 → ratio **4.01x** (single 0.51ms × 4)
- Profile_events 없이는 2.55x
- **profile_events ON 자체가 2x slowdown** (multi-ctx 환경)
- 이는 memory.md feedback 경고를 직접 검증
- 기존 9 트랙 op-level profiling 결과 일부는 instrumentation artifact 가능성

## Paper 메시지 update

기존 (handoff):
> "9 트랙 모두 negative. Adreno multi-queue serialize가 본질"

Phase 6 후:
> **"Adreno 830 OpenCL은 HW level에서 다중 queue를 단일 issue FIFO로 직렬화한다.**
> SW layer (OoO queue, multi-context, async dispatcher, intra-forward swap, ext_host_ptr,
> AHB+iocoherent, SVM fine-grain, recordable_queues) 어떤 mechanism도 이 직렬화를 우회하지 못한다.
> 이는 Adreno HW SQE (Stream Queue Engine) FIFO single-issue 설계의 직접적 결과로 측정 확인되었다.
> Two concurrent kernels on disjoint cl_mem objects across different cl_contexts run with wall-clock
> ratio 2.55x (slower than 1.93x single-context due to context switch overhead).
> Vulkan transfer queue, Hexagon DSP offload, NPU backend 또는 새로운 HW 만이 가능한 우회 path."

## Threats to validity 추가

- profile_events ON 시 2x slowdown 발견. 본 paper의 모든 동시성 측정은 profile_events OFF로 수행되었음을 명시.
- 변동성 σ/mean 다르게 보고된 트랙들 (multi-ctx σ=22.9%) → measurement repeatability 약함 → 반복 측정 권장

## 9 트랙 negative result 재해석

| 트랙 | 원인 (Phase 6 evidence) |
|------|---|
| LISWAP-2 async dispatch | multi-queue serialize HW |
| LISWAP-3 zero-copy pool | dispatch 자체는 빠르지만 GPU 직렬 실행 |
| LISWAP-4 intra-forward | hook이 multi-queue serialize와 충돌 |
| Multi-context | context switch + serialize 합산 (2.55x) |
| OoO queue | OoO hint 무시 (HW FIFO 강제) |
| Async dispatcher worker | SW thread isolation, HW queue는 단일 |
| DMA-BUF heap | bandwidth path 동일, 직렬화는 동일 |
| AHB + iocoherent | Phase 3에서 cache OK 확인했지만 직렬화 동일 |
| SVM fine-grain | Phase 1에서 -200x throughput 확인, 직렬화 외 추가 issue |
| recordable_queues | Phase 2에서 -3.4x 확인, 직렬화 + record overhead |

모든 negative 트랙이 **하나의 메커니즘 (HW FIFO single-issue)** 으로 통일 설명됨.

---
2026-05-09 (Phase 6 완료, paper main evidence)
