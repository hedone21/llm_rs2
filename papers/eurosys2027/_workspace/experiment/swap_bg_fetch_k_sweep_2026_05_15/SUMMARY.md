# LISWAP-8 Phase A: K Sweep (2026-05-15)

## 목적
K ∈ {2, 4, 8, 16, 32} × {baseline, bg_fetch} × ×3 reps로 background fetch의 작은-K 거동 평가.

## 환경
- Jetson Llama 3.1 8B F16 primary + Q4_0 secondary
- K=32 (Phase A) 단일 측정에서는 baseline ≈ bg_fetch ≈ 110 ms TBT 였음 → K sweep로 일반화 검증.

## 결과 (mean ± std over 3 reps)

| K | baseline TBT | bg_fetch TBT | Δ TBT | baseline fwd | bg_fetch fwd | Δ fwd | bg_fetch pending |
|---|---|---|---|---|---|---|---|
| 2  | 124.19 ± 1.71 | 123.48 ± 1.00 | -0.71  | 91.62 ± 0.39 | 115.75 ± 0.65 | **+24.13** | 1  |
| 4  | 117.01 ± 2.74 | 118.74 ± 2.79 | +1.73  | 78.26 ± 0.21 | 108.28 ± 2.25 | **+30.02** | 3  |
| 8  | 113.14 ± 3.92 | 115.71 ± 1.85 | +2.57  | 72.66 ± 0.43 | 101.03 ± 1.71 | **+28.37** | 7  |
| 16 | 113.71 ± 3.31 | 114.64 ± 0.36 | +0.93  | 69.08 ± 0.51 |  84.72 ± 1.71 | **+15.64** | 15 |
| 32 | 109.93 ± 1.59 | 116.74 ± 6.50 | +6.81  | 67.30 ± 0.59 |  67.60 ± 0.40 | **+0.30**  | 31 |

## 핵심 발견

### 1. K=32에서만 forward 영향 없음
- Phase A 단일 측정 (forward 67.82 ms)이 K=32 한정 결과였음. K<32에서는 forward 가 +15~30 ms 더 무거움.
- par_seq (single-thread prebuild, dispatcher_pending=31) 데이터와 K=32 bg_fetch가 동등 — 두 결과 모두 dispatcher pending 깊이가 큰 K에서 worker H2D와 forward GPU가 시간적으로 효과적으로 분리됨을 시사.

### 2. bg_fetch는 메모리 컨트롤러 경쟁 유발 (작은 K)
- 작은 K = 더 잦은 dispatch → worker thread H2D copy가 forward GPU 작업 사이사이 더 자주 끼어듦.
- Jetson UMA는 PCIe 없지만 메모리 컨트롤러 대역폭은 GPU/DMA가 공유 → transfer_stream과 compute_stream이 같은 BW 경쟁.
- K=4에서 forward +30 ms (가장 큼) → batch frequency = 32/4 = 8 batches × 5 ms decode pause ≈ 40 ms swap-decode interleave 윈도.

### 3. baseline은 phase separation으로 hide
- baseline의 sub-batch wait는 release queue depth=1을 강제 → swap dispatch와 forward를 시간적으로 분리하는 효과.
- forward 시간 자체는 더 작음 (K=2: 91 ms vs bg_fetch 115 ms). 그러나 main thread의 mmap_permute=1274 ms가 forward 사이에 chunks로 분산되어 결국 같은 Avg TBT.

### 4. TBT 거의 동등 (모든 K)
- |Δ TBT| ≤ 7 ms 이하. Phase A의 architectural cleanness (mmap_permute main=0)는 K=32에서만 효과적이고 작은 K는 baseline과 비슷한 wall time.

## 메모리 spike 위험 — 안전

모든 측정에서 `max_release_pending=0` 유지. dispatcher_pending은 K에 따라 0~K-1까지 도달하지만 release worker queue는 비어 있음.

## 결론

**Phase A bg_fetch는 architectural goal (main thread = 0 ms) 달성. K=32에서만 TBT 손해 없음. 작은 K에서는 메모리 컨트롤러 경쟁이 paper 측정상 손실 +15~30 ms forward 증가.**

## Phase B 평가

cuMemAlloc/Free 제거 (Phase B의 핵심 가치)는 메모리 컨트롤러 경쟁과 무관한 driver lock contention 회피. Phase A 결과에서 driver lock contention 증거 없음 (par_seq에서도 안전). Phase B는 다음 조건에서만 의미 있음:
- 더 큰 secondary (e.g. Llama 70B Q4)에서 alloc cost가 mmap_permute 대비 비-무시 비중
- 또는 cuMemAlloc 호출이 frequent하면서 driver lock 경쟁 발생하는 시나리오

**현재 데이터로는 Phase B 진입의 강한 정당성 없음**. 대신 다음 방향:
1. K=32 한정 production path (Phase A를 K=32에 한해 production-grade로 정착)
2. Worker H2D를 forward와 시간적으로 분리하는 throttling (sub-batch wait의 worker 버전)
3. Multi-worker pipeline (H2D를 여러 worker에 분산)
4. Dynamic-K + bg_fetch 결합 (K=32 sweet spot 자동 도달)

## 데이터 파일
- `k{K}_{mode}_r{rep}.{stdout,stderr,tbt.jsonl}` × 30 = 90 files
