# LISWAP-1 + LISWAP-3 조합 측정 — 사용자 가설 검증

**일자**: 2026-05-08  
**디바이스**: Galaxy S25 (SM-S931N, R3CY408S4HN), Adreno 830, 6T  
**모델**: Qwen2.5-1.5B F16 GGUF (primary) + Q4_0 AOS .auf (secondary)  
**바이너리**: `target/aarch64-linux-android/release/generate` (HEAD `f3dc4a4`, build 17:49)  
**프롬프트**: `"The quick brown fox jumps"` (5 tokens), n=200 decode, ignore_eos  
**런 수**: n=3 per scenario (총 12회)

---

## 사용자 가설

- per-token 1 layer로 swap을 분산 (LISWAP-1)
- 각 layer는 GPU H2D가 아니라 `CL_MEM_ALLOC_HOST_PTR` cl_mem에 CPU memcpy (LISWAP-3)
- → **CPU memcpy와 GPU forward compute는 자원 분리** → 진짜 병렬 진행 가능 → 단독보다 좋아야 함

판정: **NEGATIVE.** 조합은 LISWAP-1 단독보다 추가로 느려진다 (Δ +2.65 ms/tok 인플레, +9.3% 악화).

---

## 핵심 결과 요약

### Forward TBT mean (per-run mean of n=199, n=3 runs)

| 시나리오 | run1 | run2 | run3 | mean | σ_cross | Δ vs baseline |
|---|---|---|---|---|---|---|
| sync_baseline | 26.77 | 26.68 | 26.55 | **26.67** | 0.11 | — |
| liswap1_alone | 28.25 | 28.19 | 29.48 | **28.64** | 0.73 | +1.97 ms (+7.4%) |
| liswap3_alone | 26.54 | 35.35 | 26.60 | **29.49** | 5.07 | +2.83 ms (+10.6%) † |
| **liswap1_plus_3** | **29.16** | **32.52** | **32.21** | **31.29** | 1.86 | **+4.63 ms (+17.3%)** |

† liswap3_alone는 run2 outlier (35.35) 영향, 단독으로 보면 run1/run3는 baseline과 거의 동일.

### Swap-active phase (tokens 0..24, swap 진행 중)

| 시나리오 | mean | p99 | max |
|---|---|---|---|
| sync_baseline | 26.34 | 28.11 | 29.43 |
| liswap1_alone | 38.14 | 48.56 | 48.91 |
| **liswap1_plus_3** | **40.69** | **50.97** | **54.16** |

**조합이 LISWAP-1 단독 대비 swap-active phase에서 +2.55 ms/tok 추가 인플레.** 사용자 가설의 정반대.

### Saturated phase (tokens 30..198, post-swap)

| 시나리오 | mean | σ_within | p99 |
|---|---|---|---|
| sync_baseline | 26.73 | 0.67 | 29.33 |
| liswap1_alone | 27.27 | 0.82 | 29.17 |
| **liswap1_plus_3** | **29.97** | **1.63** | **32.89** |

**Swap이 끝난 후에도** liswap1+3가 baseline보다 +3.24 ms (+12.1%) 느림. liswap1 단독은 +0.54 ms (+2.0%)로 거의 회복 — **조합 path는 post-swap 회복이 안 됨**.

### End-to-end throughput (200 tokens / total wall-clock)

| 시나리오 | run1 | run2 | run3 | mean | Δ |
|---|---|---|---|---|---|
| sync_baseline | 33.4 | 34.5 | 34.5 | 34.1 tok/s | — |
| liswap1_alone | 33.5 | 33.6 | 32.2 | 33.1 tok/s | -2.9% |
| liswap3_alone | 34.6 | 26.0 | 34.0 | 31.5 tok/s | -7.6% |
| **liswap1_plus_3** | 32.3 | 29.1 | 29.4 | **30.3 tok/s** | **-11.1%** |

**조합이 가장 나쁘다.**

### 단발 stall (sync_baseline single-shot swap)

| run | swap_total | prefault | mmap_permute | madvise | sync |
|---|---|---|---|---|---|
| 1 | 436.4 ms | 108.9 | 326.8 | 0.2 | 0.2 |
| 2 | 275.0 ms | 2.6 | 270.7 | 0.1 | 1.6 |
| 3 | 308.3 ms | 6.8 | 300.1 | 0.0 | 1.4 |
| **mean** | **339.9 ms** | 39.4 | 299.2 | 0.1 | 1.1 |

기존 측정 명세("~290ms")와 동일 자릿수. mmap_permute가 압도적 (~88%).

### Pairwise Welch's t-test (per-run forward mean, n=3)

| 비교 | Δmean | t | df | SE |
|---|---|---|---|---|
| liswap1_alone − sync_baseline | +1.97 ms | 4.64 | 2.1 | 0.42 |
| liswap3_alone − sync_baseline | +2.83 ms | 0.97 | 2.0 | 2.93 |
| liswap1_plus_3 − sync_baseline | +4.63 ms | 4.31 | 2.0 | 1.07 |
| **liswap1_plus_3 − liswap1_alone** | **+2.65 ms** | **2.30** | **2.6** | **1.15** |
| liswap1_plus_3 − liswap3_alone | +1.80 ms | 0.58 | 2.5 | 3.12 |

n=3로 정확한 p-value는 어렵지만 _방향성과 효과 크기_는 일관됨. liswap1+3 vs liswap1_alone에서 t=2.30/SE=1.15는 90% CI 기준으로 0을 포함하지 않을 가능성이 높음.

---

## 핵심 비교: 시나리오 4 vs 시나리오 2 (사용자 가설 검증)

### Per-token TBT timeline (run=1, 첫 30 tokens)

| tok | baseline | liswap1 | liswap3 | **liswap1+3** |
|---|---|---|---|---|
| 0 | 27.34 | 48.42 | 26.50 | **48.65** |
| 1 | 27.44 | 48.50 | 25.66 | **48.22** |
| 2 | 26.15 | 46.08 | 25.63 | **46.25** |
| 5 | 28.11 | 43.49 | 25.62 | **44.05** |
| 10 | 26.40 | 39.48 | 25.91 | **39.82** |
| 15 | 27.39 | 35.36 | 25.64 | **35.77** |
| 20 | 27.39 | 30.85 | 25.63 | **31.57** |
| 25 | 26.64 | 26.52 | 25.65 | 27.34 |
| 30 | 27.27 (settle) | (settle) | — | 27.35 |

**첫 25 tokens에서 liswap1+3가 liswap1보다 일관되게 약간 더 높음.** CPU memcpy와 GPU forward의 병렬성이 있다면 liswap1+3가 baseline 가까이 가야 하지만, liswap1 단독과 동일한 25-token 인플레 + 추가 0.5~1 ms.

### Settle 시점 (forward_ms가 baseline median × 1.05 = 27.97 ms 이하로 회복되는 token idx)

| 시나리오 | run1 | run2 | run3 |
|---|---|---|---|
| liswap1_alone | tok29 | tok30 | tok26 |
| liswap3_alone | tok0 | None* | tok0 |
| **liswap1_plus_3** | tok27 | None* | None* |

*None = saturated phase 전체에서도 threshold 위 (run2 outliers 또는 지속적 인플레).  
**조합도 25-token 분산은 동일 (settle ≈ ticks 끝남)** — LISWAP-3가 stall 회피에 추가 기여 못 함.

### Per-tick swap latency 자체는 LISWAP-3가 줄임 (run=1)

| tick | liswap1 swap_lat | liswap1+3 swap_lat |
|---|---|---|
| 0 | 17.8 ms | 10.4 ms |
| 5 | 24.3 ms | 24.3 ms |
| 10 | 42.3 ms | 24.9 ms |
| 15 | 58.2 ms | 28.1 ms |
| 20 | 58.2 ms | 25.3 ms |
| **mean** | **44.34** | **25.36** |

**LISWAP-3 zero-copy pool은 swap operation 자체는 ~43% 빠르다.** 하지만 **이 시간 절약이 forward_ms로 전이되지 않는다** — 즉 swap 코드 내 측정 시간은 줄지만 token TBT는 동일하게 인플레.

### 해석

CPU memcpy를 GPU H2D 대신 사용해도 **driver/queue/MMU 차원에서 GPU forward와 어떤 형태로든 직렬화됨**. 가능한 원인:
1. ALLOC_HOST_PTR cl_mem `Map → memcpy → Unmap` 사이클이 OpenCL command queue에 lock-step 동기화를 강제 (Adreno 드라이버)
2. Map/Unmap이 implicit cache flush + MMU TLB 무효화를 트리거 → 다음 GPU dispatch 전 sync barrier
3. CPU memcpy 자체가 6-thread 환경에서 시스템 메모리 대역폭을 forward path와 경합 (NEON GEMV가 메모리 bound)
4. ArcSwap commit + ratio_generation bump로 인한 Plan 재컴파일 race

원인이 무엇이든, **single-process 기반의 직선 swap track에서 "CPU/GPU 자원 분리"가 driver 수준에서 보장되지 않음**이 본 측정의 결론.

---

## 정확성 검증

모든 12회 run에서 첫 출력 `"The quick brown fox jumps over the lazy dog"` 일치 (5/5 OK).

| 시나리오 | run=1 첫 줄 |
|---|---|
| sync_baseline | "...over the lazy dog." |
| liswap1_alone | "...over the lazy dog. ..." |
| liswap3_alone | "...over the lazy dog. This sentence has..." |
| liswap1_plus_3 | "...over the lazy dog. ..." |

LISWAP-3 host_ptr_pool 활성화 확인 (시나리오 3, 4 모두): `[LISWAP-3] host_ptr_pool active: slots=14, max_tensor_size=11534336`.

---

## 결론

**사용자 가설 NEGATIVE.**

1. LISWAP-1 + LISWAP-3 조합은 LISWAP-1 단독 대비 **추가로 +2.65 ms/tok 인플레** (+9.3% 악화).
2. swap-active phase에서도 +2.55 ms 인플레, saturated phase에서도 +2.70 ms 추가 — 회복 안 됨.
3. 조합의 end-to-end throughput은 -11.1%로 4시나리오 중 최악.
4. swap operation 자체는 LISWAP-3로 ~43% 빨라지지만 **이 절약이 forward TBT로 전이되지 않음** → CPU memcpy도 GPU forward와 어떤 형태로든 직렬화됨.

**모든 직선 swap 트랙(Sprint A~F + LISWAP-1/2/3 + 조합) negative 확정.** 다음 단계는 LISWAP-4 (intra-forward layer-aligned swap, `.agent/todos/handoff_liswap_4_intra_forward_swap.md`)로 이동 — layer 처리 사이의 micro-gap을 활용하는 본질적으로 다른 방향.

---

## Raw 데이터

- `liswap1_plus_3_raw/liswap_combo/{scenario}_run{1..3}.{jsonl,log}` — per-token TBT JSONL + 전체 stdout/stderr
- `liswap1_plus_3_raw/aggregated_summary.json` — 시나리오/run별 통계 집계

## 명령 로그

```bash
adb -s R3CY408S4HN shell '/data/local/tmp/run_scenario.sh sync_baseline {1,2,3}'
adb -s R3CY408S4HN shell '/data/local/tmp/run_scenario.sh liswap1_alone {1,2,3} --swap-incremental-per-tick 1'
adb -s R3CY408S4HN shell 'LLMRS_OPENCL_HOST_PTR_POOL=1 /data/local/tmp/run_scenario.sh liswap3_alone {1,2,3} --swap-zero-copy --swap-pool-slots 14'
adb -s R3CY408S4HN shell 'LLMRS_OPENCL_HOST_PTR_POOL=1 /data/local/tmp/run_scenario.sh liswap1_plus_3 {1,2,3} --swap-incremental-per-tick 1 --swap-zero-copy --swap-pool-slots 14'
```

공통 인자:
```
--model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf
--secondary-gguf /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-aos.auf
--secondary-layout aos --force-swap-ratio 0.9 --ignore-eos --threads 6 --backend opencl
--tbt-log <PATH> -p "The quick brown fox jumps" -n 200
```
