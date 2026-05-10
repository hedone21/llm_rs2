# LISWAP-1+3 v2 측정 — `fill_host_ptr_buffer`의 `clFinish` 제거 효과 검증

**일자**: 2026-05-08
**디바이스**: Galaxy S25 (SM-S931N, R3CY408S4HN), Adreno 830, 6T
**모델**: Qwen2.5-1.5B F16 GGUF (primary) + Q4_0 AOS .auf (secondary)
**바이너리**: `target/aarch64-linux-android/release/generate` (HEAD `0c8951e` + uncommitted `engine/src/backend/opencl/mod.rs:3963-3973` env-gate)
**프롬프트**: `"The quick brown fox jumps"` (5 tokens), n=200 decode, ignore_eos
**런 수**: n=3 per scenario × 3 시나리오 = 9 runs

---

## 가설 검증 대상

`fill_host_ptr_buffer`의 `clEnqueueUnmapMemObject` 직후 `self.queue.finish()` 호출이 swap layer마다 (incremental swap 25 ticks × 16 layers) 풀 OpenCL 큐 배리어를 강제 → forward 큐와 직렬화. 이 `clFinish`를 env-gate (`LLMRS_HOST_PTR_SKIP_FINISH=1`)로 제거하면 OpenCL same-queue dependency tracking에 의존하여 swap memcpy와 forward compute가 진짜 병렬로 진행될 것이라는 가설.

기대 시나리오:
- **Strong positive**: swap-active TBT가 sync_baseline (~26 ms) 수준으로 회복
- **Partial**: 일부 인플레 감소 (예: +14 → +6 ms)
- **Negative**: clFinish 외에 다른 직렬화 메커니즘 존재

**판정: NEGATIVE (강한 부정)**. clFinish 제거가 swap-active phase에서 +6.94 ms/tok 추가 인플레 (+17.2% 악화)를 가져오며, 가설의 정반대 결과를 보임.

---

## 핵심 결과 요약

### Forward TBT mean (per-run mean of n=199, n=3 runs)

| 시나리오 | run1 | run2 | run3 | mean | σ_cross | Δ vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| sync_baseline | 26.82 | 26.74 | 26.60 | **26.72** | 0.11 | — |
| liswap1+3 v1 (clFinish 있음) | 28.12 | 35.22 | 29.77 | **31.04** | 3.71 | +4.31 ms (+16.1%) |
| **liswap1+3 v2 (clFinish 제거)** | **28.66** | **36.49** | **36.81** | **33.99** | **4.61** | **+7.26 ms (+27.2%)** |

`Δ(v2 − v1) = +2.95 ms (+9.5% vs v1)`. **clFinish 제거 후 더 느려짐.**

### Swap-active phase (tokens 0..24, swap 진행 중, n=75 aggregated)

| 시나리오 | mean | std | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| sync_baseline | 26.03 | 0.77 | 26.20 | 28.79 | 31.62 |
| liswap1+3 v1 | 40.39 | 6.51 | 48.81 | 52.26 | 54.36 |
| **liswap1+3 v2** | **47.34** | **9.40** | **59.87** | **64.30** | **68.03** |

**v2가 v1 대비 +6.94 ms/tok 추가 인플레, p99는 +12.04 ms 악화.** 표준편차도 6.51 → 9.40으로 증가.

### Saturated phase (tokens 30..198, post-swap, n=507 aggregated)

| 시나리오 | mean | std | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| sync_baseline | 26.85 | 0.51 | 27.28 | 29.17 | 29.80 |
| liswap1+3 v1 | 29.81 | 3.52 | 35.88 | 36.56 | 38.18 |
| **liswap1+3 v2** | **32.13** | **4.14** | **36.71** | **37.92** | **39.30** |

**Swap 종료 후에도** v2가 v1보다 +2.32 ms 더 느림 — clFinish 제거가 saturated phase까지 영향을 미침 (예측 못 한 효과).

### Tok[0] (first decode token, n=3)

| 시나리오 | mean | std |
|---|---:|---:|
| sync_baseline | 26.79 | 0.88 |
| liswap1+3 v1 | 49.17 | 0.70 |
| **liswap1+3 v2** | **51.27** | **1.86** |

`Δ(v2 − v1) = +2.10 ms`. tok[0]에서도 v2가 더 느림. v2의 σ가 2.7× 더 큼 (jitter 증가).

### End-to-end throughput

| 시나리오 | Decode(ms/tok) mean | tok/s mean | Avg TBT(ms) mean |
|---|---:|---:|---:|
| sync_baseline | 26.72 | 37.43 | 31.31 |
| liswap1+3 v1 | 31.13 | 32.40 | 38.96 |
| **liswap1+3 v2** | **34.07** | **29.73** | **43.32** |

**v2가 v1 대비 -8.2% throughput 손실, baseline 대비 -20.6% 손실** — 시나리오 중 최악.

### Per-tick swap latency (incremental swap, 25 ticks)

| 시나리오 | n_ticks | mean | std | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| liswap1+3 v1 | 75 | 26.14 | 7.93 | 27.86 | 51.19 | 52.30 |
| **liswap1+3 v2** | 75 | **34.34** | **13.47** | **53.98** | **56.13** | **56.20** |

**Swap operation 자체도 v2에서 +8.20 ms (+31.4%) 느려짐.** 이는 가설의 직접적 반박: clFinish 제거가 단일 swap 호출을 빠르게 만들지 못함 (오히려 느림).

---

## 핵심 비교: v1 vs v2 (정량 요약 표)

| 메트릭 | v1 (clFinish 있음) | v2 (clFinish 제거) | Δ(v2 − v1) | 효과 |
|---|---:|---:|---:|:---|
| swap-active TBT | 40.39 ms | 47.34 ms | **+6.94 ms** | **−17.2% 악화** |
| saturated TBT | 29.81 ms | 32.13 ms | +2.32 ms | −7.8% 악화 |
| per-tick swap | 26.14 ms | 34.34 ms | +8.20 ms | −31.4% 악화 |
| tok[0] | 49.17 ms | 51.27 ms | +2.10 ms | −4.3% 악화 |
| forward(excl tok[0]) | 31.04 ms | 33.99 ms | +2.95 ms | −9.5% 악화 |
| 정확성 | 5/5 prefix 일치 | 5/5 prefix 일치 | — | OK |

**모든 지표에서 v2가 더 느림.** 정확성은 영향 없음 (OpenCL same-queue dependency tracking이 작동함을 시사).

---

## Per-token timeline (run=1, 첫 30 tokens, forward_ms)

| tok | baseline | v1 | v2 | Δ(v2−v1) |
|---:|---:|---:|---:|---:|
| 0 | 27.80 | 48.51 | 50.58 | +2.07 |
| 1 | 25.94 | 47.99 | 50.96 | +2.97 |
| 2 | 25.92 | 45.96 | 51.37 | +5.41 |
| 3 | 25.95 | 45.27 | 52.83 | +7.56 |
| 4 | 26.15 | 44.47 | 57.49 | **+13.02** |
| 5 | 26.13 | 43.53 | 47.71 | +4.18 |
| 6 | 26.15 | 42.71 | 59.77 | **+17.06** |
| 7 | 26.19 | 41.88 | 52.93 | +11.05 |
| 8 | 25.96 | 41.05 | 47.60 | +6.55 |
| 9 | 26.07 | 40.09 | 51.17 | +11.08 |
| 10 | 26.08 | 39.39 | 49.94 | +10.55 |
| 11 | 26.09 | 38.57 | 48.99 | +10.42 |
| 12 | 25.99 | 37.65 | 37.78 | +0.13 |
| 13 | 31.62 | 36.81 | 36.86 | +0.05 |
| 14–24 | (~26) | (~28–36, 단조감소) | (~28–37, 동일패턴) | ~0 |
| 25+ | 26.0 | 26.5 | 26.5 | settle |

**관찰**:
- **tok 0~11에서 v2가 v1보다 일관되게 +5~17 ms 더 느림** — 이 구간이 성능 악화의 주범.
- **tok 12~24는 v1과 v2가 거의 동일** — 이 구간에서 swap latency 단조 감소 패턴 동일.
- v2 tok 4, 6 등에 명백한 outlier (+13, +17 ms) — clFinish 제거로 인한 jitter 증가.

이 패턴은 "초반 swap이 forward와 진짜 동시에 걸려 stall이 더 커진다"는 해석과 정합. clFinish가 있을 때는 swap이 layer마다 forward와 짧은 직렬 구간을 가져 GPU 큐의 backpressure를 흡수했던 반면, 제거하면 swap dispatch가 누적되어 forward가 시작될 때 GPU 큐에 무거운 unmap 후속작업이 대기 중이라 더 큰 stall을 유발하는 것으로 추정.

---

## Per-tick swap latency 비교 (run=1)

| tick | v1 | v2 | Δ |
|---:|---:|---:|---:|
| 0 | 8.30 | 9.40 | +1.10 |
| 1 | 7.80 | 8.30 | +0.50 |
| 2~19 | ~25.5 | ~26.5 | +0~+2.5 |
| 20 | 52.30 | 25.30 | **−27.00** |
| 21 | 48.30 | 25.60 | **−22.70** |
| 22 | 48.00 | 25.90 | **−22.10** |
| 23 | 50.80 | 49.90 | −0.90 |
| 24 | 47.30 | 49.40 | +2.10 |

**흥미로운 부분**: v1에서 tick 20~22가 ~50 ms로 spike (마지막 layer의 누적 backpressure)였으나, v2에서는 ~25 ms로 일정. 즉 **swap operation 자체의 spike는 v2가 분산시킴**. 그러나 이 swap spike 분산이 forward TBT에는 도움이 되지 않고 오히려 forward TBT에 작은 파편으로 스며듦. **개별 swap latency는 평균적으로는 v2가 더 길다 (mean 26.14 → 34.34)**: tick 0~19 구간에서 v2의 latency가 v1보다 약간씩 더 길고, tick 20~24는 v2에서 분산.

---

## 정확성 검증

모든 9 runs에서 디코드 출력이 prompt prefix `"The quick brown fox jumps"`에 이어 의미있는 텍스트를 생성. 9/9 OK.

| run | 출력 prefix |
|---|---|
| sync_baseline run 1~3 | "The quick brown fox jumps over the lazy dog. ..." |
| liswap1+3 v1 run 1~3 | "...lazy dog..." (sampling 차이로 일부 다른 콘텐츠) |
| liswap1+3 v2 run 1~3 | "...lazy dog..." (sampling 차이로 일부 다른 콘텐츠) |

`grep 'lazy dog'` 카운트: sync 1~2회/run, v1 0~1회/run, v2 0~2회/run — sampling은 deterministic이 아니므로 후속 텍스트 분기는 자연스러움. **garbage 출력 (NaN, 무의미한 토큰) 없음.** OpenCL same-queue dependency tracking이 정상 작동함을 시사 — 정확성 측면에서 clFinish는 불필요.

---

## 해석

### 가설 반박: clFinish 제거가 직렬화를 풀지 못함

가설은 "clFinish가 swap memcpy와 forward compute의 직렬화 원인"이라고 가정했지만, 측정은 정반대를 보였다:

1. **clFinish 제거 → forward TBT 증가** (+2.95 ms/tok). 진짜 병렬 진행이 됐다면 감소해야 했음.
2. **clFinish 제거 → 단일 swap latency도 증가** (+8.20 ms). swap 자체가 빨라지지도 않음.
3. **clFinish 제거 → variance 증가** (σ 6.51 → 9.40). 더 jittery.

따라서 직렬화의 진짜 원인은 다른 곳에 있고, clFinish는 오히려 GPU 큐 backpressure를 미리 짧게 풀어주는 효과적 측정점이었던 것으로 보인다.

### 가능한 설명

1. **OpenCL command queue의 implicit serialization은 clFinish 제거로 풀리지 않는다**. Adreno 드라이버에서 같은 큐의 unmap 후속 dispatch가 단일 명령 스트림으로 처리됨 (clFinish는 단지 명시적 wait이고, 본질적 직렬화는 큐 자체가 강제).

2. **clFinish가 swap의 "정점"을 layer마다 짧게 끝내주던 효과**. 제거하면 25개 layer × 16개 tensor의 unmap이 큐에 누적되어 forward가 시작될 때 GPU가 unmap drain을 먼저 처리하느라 더 큰 stall을 만든다 (특히 tok 0~11 구간).

3. **CPU memcpy도 cache flush + MMU TLB 무효화 트리거**. ARM UMA에서 ALLOC_HOST_PTR map/unmap은 implicit cache coherence 작업을 유발 — clFinish 유무와 무관하게 GPU pipeline에 부담.

4. **6T NEON GEMV가 메모리 대역폭 bound**. swap CPU memcpy가 system bus를 점유 → forward path와 진짜 자원 분리가 아님.

이전 v1 측정 리포트의 "single-process 기반의 직선 swap track에서 'CPU/GPU 자원 분리'가 driver 수준에서 보장되지 않음" 결론을 강화한다.

---

## 결론

**사용자 가설 NEGATIVE.** `fill_host_ptr_buffer`의 `clFinish` 제거는 swap-forward 병렬성을 회복시키지 못하고, 오히려 모든 metric에서 성능을 악화시킨다.

| 항목 | 결과 |
|---|---|
| Strong positive | ❌ |
| Partial positive | ❌ |
| Neutral | ❌ |
| **Negative** | ✅ (-9.5% forward TBT 추가 악화) |

**다음 단계 권고**:

1. **`LLMRS_HOST_PTR_SKIP_FINISH` env-gate를 default OFF 유지** (현재 구현이 그러함). 코드 보존 또는 제거는 자유 — 본 측정으로 효과 없음이 입증됐으므로 production code에서 제거 권장.

2. **LISWAP-1, LISWAP-3, LISWAP-1+3 트랙을 본 v2 결과 포함하여 모두 negative 종결**. 직선 swap 트랙에서 추가 옵션 탐색 가치 없음.

3. **LISWAP-4 (intra-forward layer-aligned swap, `.agent/todos/handoff_liswap_4_intra_forward_swap.md`)로 이동** — 이미 별도 트랙 진행 중. forward path의 layer 처리 사이의 micro-gap을 활용하는 본질적으로 다른 방향.

4. **Driver-level serialization 가설 검증이 필요하면**: `clEnqueueMarkerWithWaitList` + `clGetEventProfilingInfo`로 unmap event의 GPU side timeline을 측정해 큐 내 backpressure를 직접 관찰하는 추가 microbench를 한 번 시도해볼 수 있음 (단, profiling enable 자체가 wall-clock을 왜곡한다는 알려진 한계 있음 — `feedback_opencl_profile_events_cross_engine.md`).

---

## Raw 데이터

- `liswap1_plus_3_v2_raw/liswap_combo_v2/{sync_baseline,liswap1_plus_3_v1,liswap1_plus_3_v2}_run{1..3}.{jsonl,log}` — per-token TBT JSONL + 전체 stdout/stderr (18 files)
- `liswap1_plus_3_v2_raw/aggregated_summary.json` — 시나리오/run별 통계 집계

## 명령 로그

```bash
# 코드 변경: engine/src/backend/opencl/mod.rs:3963-3973
#   if std::env::var("LLMRS_HOST_PTR_SKIP_FINISH").is_err() {
#       self.queue.finish()?;
#   }

# 빌드: cargo build --release --target aarch64-linux-android --bin generate
# 디바이스 push: adb -s R3CY408S4HN push generate /data/local/tmp/generate

# 시나리오 1 (reference)
adb -s R3CY408S4HN shell '/data/local/tmp/run_scenario_v2.sh sync_baseline {1,2,3}'

# 시나리오 2 (v1: clFinish 있음)
adb -s R3CY408S4HN shell 'LLMRS_OPENCL_HOST_PTR_POOL=1 /data/local/tmp/run_scenario_v2.sh liswap1_plus_3_v1 {1,2,3} --swap-incremental-per-tick 1 --swap-zero-copy --swap-pool-slots 14'

# 시나리오 3 (v2: clFinish 제거, 신규)
adb -s R3CY408S4HN shell 'LLMRS_OPENCL_HOST_PTR_POOL=1 LLMRS_HOST_PTR_SKIP_FINISH=1 /data/local/tmp/run_scenario_v2.sh liswap1_plus_3_v2 {1,2,3} --swap-incremental-per-tick 1 --swap-zero-copy --swap-pool-slots 14'
```

공통 인자 (run_scenario_v2.sh):
```
--model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf
--secondary-gguf /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-aos.auf
--secondary-layout aos --force-swap-ratio 0.9 --ignore-eos --threads 6 --backend opencl
--tbt-log <PATH> -p "The quick brown fox jumps" -n 200
```
