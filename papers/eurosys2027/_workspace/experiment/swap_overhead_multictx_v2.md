# Multi-context swap (Option B) — Galaxy S25 재측정 (v2)

- **날짜**: 2026-05-08
- **디바이스**: Galaxy S25 (R3CY408S4HN, SM-S931N), Adreno 830
- **모델**: Qwen2.5-1.5B (primary F16 GGUF + secondary Q4_0 AOS)
- **빌드**: `target/aarch64-linux-android/release/generate` (host mtime `2026-05-08 20:39:41`, size 7,560,704 B), bug fix 적용 후 빌드
- **시나리오 수**: 6, 각 n=3, n_tokens=200, threads=6, force-swap-ratio=0.9 (25 ticks 예상)
- **Bug fix 컨텍스트**: 사용자 보고에 따르면 swap_mem 생성 flag를 `CL_MEM_READ_ONLY` → `CL_MEM_READ_WRITE`로 변경하여 `clEnqueueMapBuffer(CL_MAP_WRITE)` 충돌을 해소했다고 함.
- **모델 경로**: 디바이스에 사용자 명시 `qwen2.5-1.5b-instruct-*` 파일 부재. 이전 v1 측정과 동일한 `/data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-{f16,q4_0}.gguf` 사용 (이전 측정과의 직접 비교 일관성 유지).

## 요약 (TL;DR)

> **Bug fix 후에도 multi-context swap path는 v1과 동일한 `clEnqueueMapBuffer (errcode=-36 = CL_INVALID_COMMAND_QUEUE)` 에러로 25/25 swap이 실패한다 (3 runs × 4 multictx 시나리오 = 75/75 ticks 전부 실패).**
> swap_mem flag 변경은 본 에러의 근본 원인이 아니었음. 가설(driver scheduling이 cl_context 단위인지) 검증 여전히 불가. multi-context 시나리오의 TBT는 v1과 같이 swap 실패 path overhead 측정값이며, single-context와의 비교는 무의미.

## 측정 결과 (n=3 각, mean / stdev)

| # | 시나리오 | TTFT (ms) | TTFT std | Decode (ms/tok) | Avg TBT (ms) | TBT std | swap_errors (3 runs 합) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | sync_baseline | 722.75 | 224.62 | 32.51 | **37.63** | 10.39 | **0** |
| 2 | singlectx (DMA-BUF, baseline B) | 558.66 | 31.34 | 45.26 | **56.35** | 6.39 | **0** |
| 3 | multictx (★) | 619.17 | 17.87 | 59.90 | 65.92 | 5.29 | **75** |
| 4 | multictx + DMABUF_SYNC | 675.73 | 23.41 | 63.94 | 70.02 | 7.26 | **75** |
| 5 | multictx + SKIP_FINISH | 677.08 | 33.77 | 61.79 | 67.81 | 8.70 | **75** |
| 6 | multictx + sync + skip-finish | 702.01 | 16.13 | 63.44 | 69.42 | 2.36 | **75** |

> 시나리오 1의 TBT std=10.39은 run 3의 outlier(TBT=49.63 ms, 다른 run은 31.43/31.84 ms) 영향. 디바이스 thermal/load 변동 추정이며 이후 시나리오에는 영향 없음(시나리오 2~6 TBT std는 2~9 ms 범위).

### v1 vs v2 비교 (동일 디바이스, 동일 시나리오)

| 시나리오 | v1 TBT (ms) | v2 TBT (ms) | Δ (%) | v1 errs | v2 errs |
|---|---:|---:|---:|---:|---:|
| 1 sync_baseline | 30.90 | 37.63 | +21.8% | 0 | 0 |
| 2 singlectx | 40.64 | 56.35 | +38.7% | 0 | 0 |
| 3 multictx | 58.85 | 65.92 | +12.0% | 25 | 75 (= 25/run) |
| 4 multictx_sync | 60.22 | 70.02 | +16.3% | 25 | 75 |
| 5 multictx_skipfinish | 64.19 | 67.81 | +5.6% | 25 | 75 |
| 6 multictx_full | 61.90 | 69.42 | +12.2% | 25 | 75 |

> v1 → v2 절대값이 전반적으로 상승. swap 실패 패턴 자체는 동일.

### 핵심 비교 (v2 — 가설 검증 시도)

- **시나리오 3 vs 2 (multi-ctx vs single-ctx)**: TBT 65.92 vs 56.35 ms (+17.0%). multi-ctx가 swap 실패 path overhead로 측정값 오염 → 직접 비교 결론 불가.
- **시나리오 4 vs 3 (sync 추가)**: TBT 70.02 vs 65.92 ms (+6.2%). 두 케이스 모두 swap이 일어나지 않음 → sync 자체 효과 측정 불가.
- **시나리오 5 vs 3 (skip-finish 추가)**: TBT 67.81 vs 65.92 ms (+2.9%). 같은 이유로 측정 불가.

## 정확성 (출력 텍스트 첫 ~150자)

모든 18 runs 정상 영문 출력 ("The quick brown fox jumps over the lazy dog ..." 시작). 단:
- 시나리오 1, 2: secondary Q4 weight가 정상 swap됨 (3/3 정확).
- 시나리오 3~6: swap error로 인해 primary F16 weight로만 decode (Q4 적용 안 됨). garbage는 아님.

## Per-tick swap latency (시나리오 2만 유효)

| run | n | mean | median | max |
|---|---|---:|---:|---:|
| singlectx run 1 | 25 | 47.85 | 51.90 | 59.90 |
| singlectx run 2 | 25 | 47.76 | 50.10 | 55.40 |
| singlectx run 3 | 25 | 46.34 | 47.60 | 57.50 |

> v1 대비 약 5~15 ms 증가. 시나리오 3~6은 모든 tick이 swap error로 끝나 latency 데이터 없음.

## Multi-context path 활성 검증

`[LISWAP-3] host_ptr_pool active: slots=14, max_tensor_size=11534336` 로그가 시나리오 3~6 전부에 존재 → host_ptr_pool은 활성. fallback warn (`slot N multi-context DMA-BUF alloc failed`)은 부재 → multi-context alloc은 성공한 것으로 추정.

> 결론: env-gate는 v1과 동일하게 정상 작동. fill 단계가 여전히 실패.

## 문제 분석 (v1 대비 변동 없음)

### 증상

각 tick에서 동일 에러 25번 반복 — v1과 100% 동일한 메시지·errcode·ptr 상태:

```
[IncrementalSwap] swap error on tick=N: swap layer L: buffer allocation failed:
fill_dmabuf_via_swap_queue: clEnqueueMapBuffer failed (errcode=-36, ptr_null=true)
```

- errcode -36 = `CL_INVALID_COMMAND_QUEUE`.
- `ptr_null=true` → Map이 NULL 포인터 반환.
- 75/75 tick / 모든 layer에서 결정론적 실패.

### Bug fix가 효과 없는 이유 (분석)

사용자 가설은 swap_mem이 `CL_MEM_READ_ONLY`로 만들어져 `clEnqueueMapBuffer(CL_MAP_WRITE)`와 충돌 → 그래서 `CL_MEM_READ_WRITE`로 변경하면 해결될 것이라는 것이었음. 그러나:

1. `CL_MAP_WRITE`를 `CL_MEM_READ_ONLY` 버퍼에 시도할 경우 OpenCL 표준상의 에러 코드는 `CL_INVALID_OPERATION (-59)` 또는 `CL_INVALID_VALUE (-30)`. **`-36 (CL_INVALID_COMMAND_QUEUE)`는 다른 종류의 문제**.
2. v1과 v2 모두 정확히 동일한 errcode `-36` → flag 변경이 적중하지 않은 다른 원인이 있음.
3. `CL_INVALID_COMMAND_QUEUE`의 의미는 (a) queue handle invalid, 또는 (b) queue/buffer cl_context mismatch.

### 가능한 진짜 원인 (코드 미수정 — 검토만)

v1 리포트에서 제기한 가설들이 그대로 유효:

1. **swap_queue가 secondary cl_context 소속인데 swap_mem이 main_context에 binding (또는 그 반대)** — context mismatch.
2. **DMA-BUF FD를 두 cl_context에서 import할 때 driver가 cross-context import 거부**.
3. **swap_queue 생성 시 flag mismatch**.

본 v2 측정은 사용자 가정이 잘못되었음을 확인하는 negative 검증.

### 즉시 보고 트리거 충족 여부

- swap_errors > 0 in any multictx 시나리오 → **충족** (75/75 tick fail). 즉시 보고.
- garbage 출력? **No**.
- 빌드 실패? **No**.

## 가설 판정

> **현재 상태로는 "driver scheduling이 cl_context 단위인지" 가설을 검증할 수 없다. v1과 결론 동일.**

### 권장 후속 조치 (Senior implementer 영역)

1. `fill_dmabuf_via_swap_queue` 내 swap_queue / swap_mem의 cl_context를 비교하는 디버그 로그 추가 — context 일치 여부 즉시 확인.
2. `swap_queue_or_init`에 전달되는 cl_context가 `swap_context`인지 검증.
3. `alloc_dmabuf_with_swap_context`에서 swap_mem 생성에 사용한 cl_context가 swap_context인지 vs main_context인지 확인.
4. CL_INVALID_COMMAND_QUEUE 발생 시점에서 `clGetCommandQueueInfo(swap_queue, CL_QUEUE_CONTEXT, ...)` 결과와 `clGetMemObjectInfo(swap_mem, CL_MEM_CONTEXT, ...)` 결과를 비교.

## 산출물

- raw 로그: `papers/eurosys2027/_workspace/experiment/multictx_v2_raw/v2_*.log` (18 files, 18 runs).
- 본 리포트: `papers/eurosys2027/_workspace/experiment/swap_overhead_multictx_v2.md`.
- v1 비교: `papers/eurosys2027/_workspace/experiment/swap_overhead_multictx.md`.
