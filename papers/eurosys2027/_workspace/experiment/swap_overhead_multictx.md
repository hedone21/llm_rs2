# Multi-context swap (Option B) — Galaxy S25 측정

- **날짜**: 2026-05-08
- **디바이스**: Galaxy S25 (R3CY408S4HN, SM-S931N), Adreno 830
- **모델**: Qwen2.5-1.5B (primary F16 GGUF + secondary Q4_0 AOS)
- **빌드**: `target/aarch64-linux-android/release/generate` (현재 `master`, multi-context 신규 코드 포함)
- **시나리오 수**: 6, 각 n=3, n_tokens=200, threads=6, force-swap-ratio=0.9 (25 ticks)

## 요약 (TL;DR)

> **Multi-context swap path는 alloc 단계는 통과하지만 decode 중 `clEnqueueMapBuffer`가 errcode=-36으로 실패하여 실질적 swap이 일어나지 않는다.**
> 가설(driver scheduling이 cl_context 단위인지) 검증을 위한 정상 비교가 불가능. 매 25 tick에서 swap error → secondary Q4 weight 미적용. TBT는 single-context DMA-BUF 대비 +44~58% 악화 (실패 path 자체의 overhead).

## 측정 결과 (n=3 각, mean / stdev)

| # | 시나리오 | TTFT (ms) | Decode (ms/tok) | Avg TBT (ms) | TBT std | swap_errors |
|---|---|---:|---:|---:|---:|---:|
| 1 | sync_baseline | 516.75 | 26.59 | **30.90** | 0.30 | 0 |
| 2 | singlectx (DMA-BUF, baseline B) | 405.62 | 31.32 | **40.64** | 3.47 | 0 |
| 3 | multictx (★) | 530.30 | 53.08 | 58.85 | 4.05 | **25** |
| 4 | multictx + DMABUF_SYNC | 564.14 | 54.34 | 60.22 | 5.00 | **25** |
| 5 | multictx + SKIP_FINISH | 595.54 | 58.07 | 64.19 | 9.47 | **25** |
| 6 | multictx + sync + skip-finish | 663.63 | 55.79 | 61.90 | 5.57 | **25** |

### 핵심 비교 (가설 검증 시도)

- **시나리오 3 vs 2 (multi-ctx vs single-ctx)**: TBT 58.85 vs 40.64 ms (+44.8%). 다만 multi-ctx는 swap 실패 path overhead로 측정값이 오염됨 — 직접적 driver-scheduling 비교 결론 불가.
- **시나리오 4 vs 3 (sync 추가)**: TBT 60.22 vs 58.85 ms (+2.3%). sync 자체는 무관 — 두 케이스 모두 swap이 일어나지 않음.
- **시나리오 5 vs 3 (skip-finish 추가)**: TBT 64.19 vs 58.85 ms (+9.1%). skip-finish 효과 측정 불가 — 같은 이유.

### 정확성 (출력 텍스트 첫 ~120자)

모든 시나리오 6/6 정상 — "The quick brown fox jumps over the lazy dog ..." 시작. 단:
- 시나리오 1, 2: secondary Q4 weight가 정상 swap됨 (5/5 정확).
- 시나리오 3~6: swap error로 인해 primary F16 weight로만 decode (Q4 적용 안 됨). 텍스트 자체는 garbage 아님.

## Per-tick swap latency (시나리오 2만 유효)

| run | n | mean | median | max |
|---|---|---:|---:|---:|
| singlectx run 1 | 25 | 39.97 | 45.10 | 54.90 |
| singlectx run 2 | 25 | 32.10 | 27.50 | 58.60 |
| singlectx run 3 | 25 | 42.71 | 45.20 | 54.70 |

> 시나리오 3~6은 모든 tick이 swap error로 끝나 latency 데이터 없음.

## Multi-context path 활성 검증

`[LISWAP-3] host_ptr_pool active: slots=14, max_tensor_size=11534336` 로그가 시나리오 3~6 전부에 존재 → host_ptr_pool은 활성. `RUST_LOG=info` 미설정으로 `HostPtrPool: multi-context swap path enabled` `log::info!` 줄은 stdout에 안 나타나지만, **풀 초기화 단계 fallback warn (`slot N multi-context DMA-BUF alloc failed`)이 부재**하므로 multi-context alloc(`alloc_dmabuf_with_swap_context`)은 25 slot 모두 성공으로 추정.

> 결론: env-gate는 정상 작동. 문제는 alloc 이후 단계(`fill_dmabuf_via_swap_queue`)에서 발생.

## 문제 분석

### 증상

각 tick에서 동일 에러 25번 반복:

```
[IncrementalSwap] swap error on tick=N: swap layer L: buffer allocation failed:
fill_dmabuf_via_swap_queue: clEnqueueMapBuffer failed (errcode=-36, ptr_null=true)
```

- errcode -36 = `CL_INVALID_COMMAND_QUEUE`.
- `ptr_null=true` → Map이 NULL 포인터 반환.
- 모든 25 tick / 모든 layer에서 결정론적 실패.

### 가능한 원인 (코드 미수정 — 검토만)

1. **swap_queue가 secondary cl_context 소속인데 cl_mem은 main_context에 binding**:
   `alloc_dmabuf_with_swap_context`가 swap_mem(secondary ctx) + main_mem(main ctx)을 두 개 만들지만, `fill_dmabuf_via_swap_queue`가 swap_mem을 swap_queue에 enqueue 시 ctx-queue 일치하지 않을 가능성. (swap_queue가 main_ctx로 잘못 init됐거나, swap_mem이 main_ctx로 만들어졌거나.)
2. **DMA-BUF FD를 두 cl_context에서 import할 때 driver가 두 번째 import 거부**:
   Adreno OpenCL이 `cl_arm_import_memory` 또는 `clImportMemoryARM` 분기에서 같은 FD에 대한 cross-context import를 허용하지 않을 가능성. 이 경우 `clCreateBufferWithProperties`는 통과해도 실제 access 시 invalid handle.
3. **swap_queue 생성 시 `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` 등 flag mismatch**.

### 즉시 보고 트리거 충족 여부

- 모든 시나리오 garbage 출력? **No** — 텍스트는 모두 정상.
- multi-context 활성 로그 부재? **부분적** — env-gate은 작동하나 fill 단계 실패. 즉 "활성됐으나 작동 안 함".
- 빌드 실패? **No**.

본 측정은 즉시 stop 트리거에는 미달이지만, **`fill_dmabuf_via_swap_queue` 구현이 동작하지 않음**이라는 명확한 결과를 산출.

## 가설 판정

> **현재 상태로는 "driver scheduling이 cl_context 단위인지" 가설을 검증할 수 없다.**

multi-context swap이 실질적으로 실행되지 않으므로(alloc만 성공, fill 실패), 시나리오 3~6의 TBT는 모두 실패 path overhead 측정값이며, single-context와의 비교는 무의미. Senior implementer의 fill 경로 디버깅이 선결.

### 수정/조사 권장 항목 (코드 수정은 본 작업 범위 밖)

1. `fill_dmabuf_via_swap_queue` 내 swap_queue/swap_mem context 일치 확인.
2. `swap_queue_or_init`에서 cl_context 인자가 `swap_context`인지 검증 (main_context로 잘못 만들었을 가능성).
3. `alloc_dmabuf_with_swap_context`에서 swap_mem 생성 시 사용한 cl_context와 swap_queue의 cl_context 비교 로그 추가.
4. CL_INVALID_COMMAND_QUEUE의 정확한 의미: queue/buffer context mismatch 또는 queue handle invalid.

## 부수 관찰

- **TBT std 증가**: skip-finish 단독(시나리오 5)에서 TBT std=9.47 (다른 multi-ctx 시나리오 4~5 대비 1.5~2배). `clFinish` 미호출이 timing 분산 증가로 이어짐. 단, 평균 TBT 자체는 swap 실패 overhead가 주.
- **TTFT 증가**: 시나리오 6 (full)에서 TTFT 663.63 ms (시나리오 1 대비 +28.4%). swap_context lazy init 비용 추정.

## 산출물

- raw 로그: `papers/eurosys2027/_workspace/experiment/multictx_raw/multictx_*.log` (18 files, 18 runs).
- 본 리포트: `papers/eurosys2027/_workspace/experiment/swap_overhead_multictx.md`.

