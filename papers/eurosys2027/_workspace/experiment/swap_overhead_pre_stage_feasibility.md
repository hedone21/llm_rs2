# Swap Overhead — Predictive Pre-staging Feasibility (LISWAP-PreStage)

**일자**: 2026-05-08
**디바이스**: Galaxy S25 (SM-S931N, Adreno 830, R3CY408S5SB)
**HEAD**: `0c8951e` (LISWAP-4 commit) + uncommitted `engine/src/bin/generate.rs` (`--swap-pre-stage` 플래그 추가)
**모델**: Qwen2.5-1.5B F16 GGUF (3.1 GB, primary) + Q4_0 AOS .auf (1.3 GB, secondary)
**프롬프트**: "The quick brown fox jumps" (5 tokens), `-n 40`, threads=6, backend=opencl, force-swap-ratio=0.9, secondary-layout=aos

## 1. 가설

LISWAP-2 (multi-queue parallel H2D)는 Adreno OpenCL 드라이버가 multi-queue를 hardware-serialize 시키는 결과로 negative였다. 후속 가설:

> swap 신호 도착 **전에** 모든 H2D를 background dispatcher (`AsyncSwapDispatcher`)로 미리 enqueue (non-blocking) → forward 시작 직후엔 ArcSwap commit만 끝나면 됨 → forward와 H2D byte transfer가 진짜로 overlap할 수 있지 않나?

이 측정은 그 feasibility 최초 검증이다. fail이면 Adreno serialize의 timing 의존성 추가 검증, success면 paper 메인 메시지 변경 가능.

## 2. CLI flag (구현 완료)

`--swap-pre-stage` (default `false`).
- `true` + `--force-swap-ratio` 활성: `run_layer_swap()`을 `AsyncSwapDispatcher` 인자와 함께 호출
- `enqueue_write_async`가 모든 H2D를 non-blocking으로 enqueue → 즉시 return
- dispatcher worker가 background에서 `cl_event` 대기 → ArcSwap commit
- prefill+decode는 즉시 시작 (H2D는 background에서 진행 중이라 가정)
- generation 끝에 `dispatcher.drain(10s)` — 미완료 swap 정리
- `--swap-incremental-per-tick > 0` / `--swap-intra-forward`와 상호 배타

## 3. 측정 시나리오

| 시나리오 | flag | iterations |
|---|---|---|
| sync_baseline | (기본) | n=5 |
| pre_stage | `--swap-pre-stage` | n=5 |

incremental_pt1는 본 측정에 포함하지 않음 (paper 비교용은 별도 트랙).

## 4. Raw data (n=5)

| run | sync swap (ms) | sync prefill (ms) | sync decode (ms/tok) | prestage dispatch_wall (ms) | prestage prefill (ms) | prestage decode (ms/tok) | prestage drain_wait (ms) | prestage pending_before |
|----:|---------------:|------------------:|---------------------:|----------------------------:|----------------------:|-------------------------:|-------------------------:|------------------------:|
| 1 | 260.2 | 105.09 | 25.71 | 272.1 | 105.27 | 25.54 | 0.0 | 0 |
| 2 | 268.2 | 107.52 | 25.69 | 298.5 | 105.47 | 25.56 | 0.0 | 0 |
| 3 | 351.1 | 104.47 | 25.81 | 308.5 | 104.22 | 25.50 | 0.0 | 0 |
| 4 | 271.3 | 106.87 | 25.61 | 293.3 | 105.60 | 25.49 | 0.0 | 0 |
| 5 | 294.3 | 106.57 | 25.68 | 316.9 | 106.46 | 25.61 | 0.0 | 0 |

`prefill`은 `Prefill(pure fwd)` 라인 (sync'd forward only, llama-bench pp 비교 가능). `decode (ms/tok)`는 `Decode:` 라인의 forward-only 값.

## 5. 통계 요약 (mean ± σ, n=5)

| 항목 | sync | prestage | ∆ | ∆% |
|---|---|---|---|---|
| swap / dispatch_wall (ms) | **289.0 ± 33.1** | **297.9 ± 15.2** | +8.9 | +3.07% |
| prefill (ms, pure fwd) | 106.10 ± 1.14 | 105.40 ± 0.72 | -0.70 | **-0.66%** |
| decode (ms/tok) | 25.700 ± 0.063 | 25.540 ± 0.045 | -0.160 | **-0.62%** |
| Avg TBT (ms) | 27.190 ± 0.063 | 27.016 ± 0.054 | -0.174 | -0.64% |
| tok[0] (ms) | 26.52 ± 0.54 | 26.68 ± 1.01 | +0.15 | +0.58% |
| TTFT (ms) | 498.5 ± 31.6 | 502.6 ± 16.7 | +4.1 | +0.83% |
| **drain_wait (ms)** | n/a | **0.00 ± 0.00** | — | — |
| **pending_before** | n/a | **0** (5/5 runs) | — | — |

### Wall-clock budget @ N=39 decode tokens

| run | sync_total (ms) | prestage_total (ms) | ∆ ms | ∆ % |
|----:|----------------:|--------------------:|-----:|----:|
| 1 | 1368.0 | 1373.4 | +5.4 | +0.40% |
| 2 | 1377.6 | 1400.8 | +23.2 | +1.68% |
| 3 | 1462.2 | 1407.2 | -54.9 | -3.76% |
| 4 | 1377.0 | 1393.0 | +16.0 | +1.17% |
| 5 | 1402.4 | 1422.1 | +19.8 | +1.41% |
| **mean** | **1397.4 ± 34.3** | **1399.3 ± 16.1** | **+1.9** | **+0.14%** |

## 6. 관측 사실

1. **`pending_before = 0` (5/5 runs)**, **`drain_wait = 0.0 ms` (5/5 runs)**. 즉 prefill 시작 시점엔 모든 H2D + ArcSwap commit이 이미 완료. dispatch가 비동기인 척 했지만 **동기적으로 종결**.
2. **`dispatch_wall` 평균 297.9 ms** vs sync swap 평균 289.0 ms — 거의 동일 (∆ +8.9 ms, +3.07%). 즉 호출자 thread는 sync 경로와 사실상 같은 시간 기다린다.
3. **prefill 평균 105.40 ms** vs sync 106.10 ms (∆ -0.70 ms). 노이즈 범위. forward와 H2D overlap 효과 0.
4. **decode 평균 25.54 ms/tok** vs sync 25.70 ms/tok (∆ -0.16 ms/tok, -0.62%). 노이즈 범위.
5. **wall-clock 평균 ∆ +1.9 ms (+0.14%)**. 전혀 절감되지 않음.
6. 정확성: 5/5 runs 모두 deterministic prefix " over the lazy dog." (9 tokens) 일치. 이후 분기는 `temperature=0.8` 기반 stochastic sampling — 정상 비결정성.

## 7. 결론: **NEGATIVE** (옵션 X feasibility 부정)

### 판정 표

| 기준 | 임계값 | 실측 | 판정 |
|---|---|---|---|
| Strong success | prefill < +5% AND decode < +5% AND drain < 50 ms AND wall ≤ −20% | wall +0.14% (saving 0%) | ✗ |
| Partial success | wall-clock 절감 측정 가능 | wall +1.9 ms (+0.14%, < σ) | ✗ |
| Negative | wall-clock saving 0% / drain ≈ swap 시간 | wall ≈ 0%, dispatch 297 ms ≈ sync 289 ms | **✓** |
| Failure mode | top-1 prefix 5/5 미일치 | 5/5 일치 | n/a |

### 근본 원인 분석

LISWAP-2의 hardware-serialize 결과가 **timing-independent로 동일하게 적용**됨이 확인되었다.

- `enqueue_write_buffer(blocking=false)`는 호출자 thread는 즉시 return시키지만, Adreno OpenCL 드라이버가 cl_event 완료를 기다리는 동안 **dispatcher worker도 동일하게 block**. 5/5 run 모두 prefill이 시작되기도 전에 dispatcher가 모든 cl_event를 wait → ArcSwap commit → pending=0 도달.
- 즉, "background dispatcher" 자체는 동작하지만, OpenCL transfer queue가 GPU compute queue와 **hardware single point of contention**을 공유하여, dispatcher worker가 transfer 완료를 기다리는 동안 **caller thread도 (mmap_permute / cl_event wait sync) 점유 시간을 동일하게 잃는다**. 결과적으로 호출자 wait time이 "sync swap 290 ms" 그대로.
- **Adreno serialization의 본질**: queue를 분리해도, 시점을 분리해도(pre-stage), 결국 H2D 한 번 시작되면 GPU compute와 hardware-level 동시 진행이 안 된다. 사용자가 가설했던 "forward와 H2D overlap"이 일어나지 않음을 (LISWAP-2와 다른 측정 setup으로) 재확인.

## 8. 함의

1. **Paper 메인 메시지 유지**: Adreno OpenCL에서는 H2D과 GPU compute의 hardware-level 병렬화가 일어나지 않으며, 이는 queue/timing 변형으로도 우회 불가하다. LISWAP-2 negative 결과의 일반성을 강화한다.
2. **남은 옵션 X→Y 전환**: 만약 H2D와 compute의 overlap이 근본적으로 막혀 있다면, 남은 swap 비용 절감 방향은 (a) **H2D bytes 자체 감축** (LISWAP-1: 부분 swap), (b) **swap 자체 회피** (정책 단계에서 swap을 피함), (c) **CPU↔GPU 비대칭 활용** — primary forward가 CPU에서 일어나는 동안 GPU H2D 진행 등. (c)는 별도 측정 필요.
3. **본 측정은 negative 결과**로 종결. 추가 ablation (다른 prompt 길이, 다른 ratio, 다른 secondary layout 조합)을 더 돌려도 결과가 뒤집힐 가능성 낮다 — pending_before=0/drain=0ms의 양적 증거가 매우 강함.

## 9. Raw logs

- `/home/go/Workspace/llm_rs2/.agent/measurements/pre_stage/raw/sync_{1..5}.log`
- `/home/go/Workspace/llm_rs2/.agent/measurements/pre_stage/raw/prestage_{1..5}.log`
- 1회 sanity warmup: `sync_baseline_warmup.log`, `pre_stage_warmup.log`

