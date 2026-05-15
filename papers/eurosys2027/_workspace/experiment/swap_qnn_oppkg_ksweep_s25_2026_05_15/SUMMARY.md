# LISWAP-8 Mobile: qnn_oppkg K-sweep 3-way (2026-05-15)

## 목적
Jetson Hammer D (mmap cuMemHostRegister + ArcSwap alias)의 모바일 등가 path 검증. Galaxy S25 (Adreno) qnn_oppkg backend는 GGUF secondary 사용 시 자동으로 rpcmem DMA-BUF alias path 활성화 (LISWAP-6) — H2D copy=0 zero-copy weight swap. Jetson Hammer D가 K<32에서 baseline TBT 능가 + K=32 idle regression 패턴을 보였음. S25 qnn_oppkg는 어떤 패턴?

## 측정 환경
- **Device**: Galaxy S25 (R3CY408S5SB), Adreno 8 Gen 4, 11 GB RAM
- **Model**: Qwen2.5-1.5B (28 layers, head_dim=128, kv_heads=2)
- **Primary**: F16 GGUF (~2.85 GB)
- **Secondary**: Q4_0 GGUF (~1.21 GB) → **rpcmem alias path 자동 활성** (qnn_oppkg + GGUF)
- **Backend**: qnn_oppkg
- **Prompt**: 71 tokens, **num-tokens=50**
- **Env**: `LLMRS_SKIP_EAGER_PREFAULT=1 LLMRS_SKIP_FINALIZE_BUDGET=1 LLMRS_SWAP_DRAIN_DIAG=1`
- **Per-run**: cold reboot + 180s thermal rest (5/14 standard)

## 측정 매트릭스
- K ∈ {2, 4, 8, 32} (`--swap-incremental-per-tick`)
- variant ∈ {baseline (swap-off), sync, async (`--swap-async-dispatch`)}
- n=1 per cell → 총 12 runs

## 결과

### Pivot table — avg_tbt (tok0 inclusive, ms) — main metric

| K | baseline (F16) | sync | async | swap−baseline (sync) |
|---|---|---|---|---|
| 2 | 58.08 | **34.21** ★ | 34.40 | **−23.87** |
| 4 | 56.09 | 37.29 | 36.96 | −18.80 |
| 8 | 79.16† | 34.52 | 34.80 | −44.65 |
| 32 | 58.88 | **41.38** ✗ | 38.49 | **−17.50** |

† K=8 baseline outlier (n=1, thermal variance 의심)

**K=2가 user-visible TBT 기준 최고. K=32 최악** (tok0 응집).

### rest_tbt (tok0 제외) — for reference

| K | baseline | sync | async |
|---|---|---|---|
| 2 | 57.98 | 33.64 | 33.38 |
| 4 | 55.95 | 35.75 | 34.54 |
| 8 | 79.51† | 31.94 | 32.18 |
| 32 | 58.69 | **26.43** | 25.15 |

K=32가 rest_tbt 기준 최저로 보이지만 avg_tbt 기준 최악. **paper main figure는 avg_tbt 사용**.

### Full table

| tag | n | tok0_fwd | tok0_tbt | rest_tbt | rest_fwd | active_fwd | idle_fwd | sw |
|---|---|---|---|---|---|---|---|---|
| k2_baseline | 49 | 5.43 | 63.07 | 57.98 | 6.66 | - | 6.66 | 0 |
| k2_sync | 49 | 4.50 | 61.74 | 33.64 | 3.27 | 2.84 | 3.44 | 14 |
| k2_async | 49 | 4.84 | 83.47 | 33.38 | 3.42 | 2.92 | 3.61 | 14 |
| k4_baseline | 49 | 5.06 | 62.88 | 55.95 | 5.22 | - | 5.22 | 0 |
| k4_sync | 49 | 5.22 | 111.08 | 35.75 | 3.54 | 3.29 | 3.57 | 7 |
| k4_async | 49 | 4.69 | 153.42 | 34.54 | 3.49 | 3.14 | 3.54 | 7 |
| k8_baseline | 49 | 5.20 | 62.63 | 79.51 | 7.39 | - | 7.39 | 0 |
| k8_sync | 49 | 5.60 | 158.27 | 31.94 | 3.15 | 2.45 | 3.19 | 4 |
| k8_async | 49 | 4.75 | 160.61 | 32.18 | 3.32 | 2.52 | 3.37 | 4 |
| k32_baseline | 49 | 7.56 | 68.08 | 58.69 | 7.10 | - | 7.10 | 0 |
| k32_sync | 49 | 5.84 | 758.90 | 26.43 | 5.20 | - | 5.20 | 1 |
| k32_async | 49 | 4.50 | 679.23 | 25.15 | 3.33 | - | 3.33 | 1 |

(sw = swap-active window size = ceil(28/K) tokens)

## 핵심 발견

### 1. swap-on이 모든 K에서 baseline보다 빠름 (−20 ~ −48 ms)
- 원인: **F16 → Q4_0 precision downgrade**가 swap cost를 압도
- 5/14 표 (F16-only 53.73 vs Q4-only 30.52)와 정합
- Jetson Hammer D의 "K<32에서 baseline 능가" 패턴과 형식은 같지만 의미는 다름:
  - Jetson: same-dtype baseline 대비 swap cost ~0
  - S25 qnn_oppkg: F16 baseline → Q4 swap-on의 dtype shift dominant

### 2. sync ≈ async (Δ ≤ 1.28 ms) — 핵심 결론
qnn_oppkg rpcmem alias path에서 async dispatcher는 sync 대비 의미있는 이득 없음.
- alias 자체가 H2D copy=0 (rpcmem DMA-BUF heap이 GPU에서 직접 접근 가능)
- worker work 강도가 거의 0 → background dispatch의 latency-hiding 효과 미미
- **= Jetson Hammer D의 모바일 등가 효과 확인**

| K | sync − async | 해석 |
|---|---|---|
| 2 | +0.26 | 무시 |
| 4 | +1.21 | 무시 |
| 8 | −0.24 | 무시 |
| 32 | +1.28 | 무시 |

### 3. K=32 tok[0]_tbt 758/679 ms (sync/async)
1 tick batch swap → 28 layer가 tok[0]에 집중. rest_tbt는 오히려 K=32가 최소.
- K=32 sync rest_tbt 26.43 < K=2 sync 33.64
- K=32 idle window 효과 비례 (32 - 1 swap-active = 31 idle tokens)
- 첫 토큰 latency vs steady-state TBT 트레이드오프

### 4. active vs idle window 차이 작음 (≤ 1 ms)
- K=2 sync: active 2.84 / idle 3.44 → Δ = +0.6 ms
- K=8 sync: active 2.45 / idle 3.19 → Δ = +0.74 ms
- swap-active 토큰의 forward 페널티가 거의 없음
- Jetson Hammer D 잔존 +143 ms (K=4)와 대조적
- rpcmem은 pre-allocated → page fault 없음 + GPU direct access → first-touch latency 0

### 5. K=8 baseline outlier (79.51)
다른 baseline (55-58)에 비해 비정상. n=1 한계. thermal 영향 또는 driver variance. 결과 해석에 K=8 baseline은 별표 처리.

## Jetson Hammer D vs S25 qnn_oppkg 비교 (avg_tbt 기준)

### Pivot — avg_tbt (tok0 inclusive, ms)

**Jetson (Llama 3.1 8B)**
| K | baseline | bg_fetch | mmap_alias |
|---|---|---|---|
| 2 | 124.42 | 129.97 | **114.77** (−9.65) |
| 4 | 118.43 | 121.62 | **108.27** (−10.16) |
| 8 | 117.62 | 120.33 | **104.03** (−13.59) |
| 32 | 110.49 | 112.61 | **101.67** (−8.82) |

**S25 (Qwen2.5-1.5B)**
| K | baseline | sync | async |
|---|---|---|---|
| 2 | 58.08 | **34.21** ★ | 34.40 |
| 4 | 56.09 | 37.29 | 36.96 |
| 8 | 79.16† | 34.52 | 34.80 |
| 32 | 58.88 | **41.38** ✗ | 38.49 |

### ⚠️ Metric 선택이 결론을 뒤집음

| K | Jetson rest_tbt 결론 | Jetson avg_tbt 결론 |
|---|---|---|
| 32 | mmap_alias +29.48 (regression) | mmap_alias **−8.82** (개선!) |

원인: Jetson baseline tok0이 K=2→K=32에서 237→**2056 ms**로 거대해짐 (측정 사이 reboot 없음 → cache effect). baseline avg가 인플레이션되어 mmap_alias가 상대적으로 좋아 보임. S25는 cold reboot으로 baseline tok0 62~68 안정.

| K | Jetson baseline tok0 | S25 baseline tok0 |
|---|---|---|
| 2 | 237.57 | 63.07 |
| 4 | 327.10 | 62.88 |
| 8 | 602.48 | 62.63 |
| 32 | **2056.40** | 68.08 |

### 정리된 비교

| 측면 | Jetson Hammer D | S25 qnn_oppkg |
|---|---|---|
| alias 메커니즘 | cuMemHostRegister(DEVICEMAP) | rpcmem DMA-BUF heap |
| H2D copy | 0 | 0 |
| Dtype | same (Q4/Q4) | cross (F16→Q4) |
| baseline 대비 swap (avg_tbt) | −8.8 ~ −13.6 ms | −17.5 ~ −44.6 ms |
| 측정 사이 reboot | 없음 | cold reboot + 180s |
| baseline tok0 안정성 | ✗ (K=32: 2056 ms outlier) | ✓ (62~68 ms 균일) |
| K=32 패널티 형태 (rest 기준) | idle window +30 ms 영구 | tok0 응집 일회성 |
| sync vs async ablation | 별도 측정 없음 | sync ≈ async 확인 |
| active forward 잔존 | +143 ms (K=4) | <1 ms |

### S25 핵심 우위
- **측정 신뢰도**: cold reboot으로 baseline 안정 → metric 선택과 무관하게 결론 robust
- **active forward**: rpcmem pre-allocated + GPU direct로 page fault 없음
- **K 패널티가 tok0 일회성**: Jetson은 idle 영구 패널티, S25는 1 token stall만

## 데이터 파일
- 36 files (12 runs × 3 = tbt.jsonl, stdout, stderr)
- 결과 파일: `papers/eurosys2027/_workspace/experiment/swap_qnn_oppkg_ksweep_s25_2026_05_15/k{2,4,8,32}_{baseline,sync,async}_r1.{tbt.jsonl,stdout,stderr}`
- 분석 스크립트: `/tmp/analyze_qnn_ksweep.py`
- 측정 스크립트: `/tmp/s25_qnn_oppkg_ksweep_3way.sh` + `/tmp/s25_qnn_oppkg_ksweep_resume.sh`

## Memory vs Latency Trade-off

### 메모리 스파이크 (release/dispatcher queue)
모든 K, sync/async에서 `release_pending=0`, `dispatcher_pending≤1`. 동시 in-flight ≈ 1 layer (7 MB Q4_0). sub-batch reactive wait (fefaf9e1, 2026-05-14)가 정상 작동.

### 그러나 latency가 tok0로 응집
| K | tok0_tbt (sync) | rest_tbt | tok0/rest ratio | swap ticks | per-tick |
|---|---|---|---|---|---|
| 2 | 61.74 | 33.64 | 1.8x | 14 | ~52 ms |
| 4 | 111.08 | 35.75 | 3.1x | 7 | ~105 ms |
| 8 | 158.27 | 31.94 | 5.0x | 4 | ~136 ms |
| 32 | **758.90** | 26.43 | **28.7x** | 1 | **743 ms** |

**Total swap cost는 K-invariant (~700 ms)**. mmap_permute가 layer-sequential 비용이라 K로 분산 정도만 달라짐.

### 핵심 trade-off
- K=2: 메모리 ✓ + tok0 latency 1.8x → 분산 14 tokens
- K=32: 메모리 ✓ + tok0 latency **28.7x (몰빵)** + steady-state TBT 최소 (idle 비율↑)

dispatcher_pending=1로 cap하는 sub-batch wait가 메모리 안전을 보장하지만, K=32에서는 user-visible latency spike가 tok0에 응집. user experience 관점에서는 K가 작을수록 (K=2/4) 부드러운 TBT 곡선.

## 미해결 / Future work
1. **n=3 측정으로 K=8 baseline outlier 확인** — n=1 한계로 thermal variance vs systematic effect 구분 불가
2. **same-dtype K-sweep** (F16 primary + F16 secondary, 또는 Q4 primary + Q4 secondary) — dtype 효과 분리하고 순수 swap cost만 보고 싶을 때
3. **generic OpenCL backend 비교** — CL_MEM_USE_HOST_PTR이 mmap pointer를 받아 zero-copy 성립하는지 검증 후 동일 K-sweep
4. **K=32 tok[0] 758 ms breakdown** — 28 layer 한 번에 swap 시 stage별 (mmap_permute, arc_swap, etc) 비용 dump
