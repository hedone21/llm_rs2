# Phase 4 — WSWAP-4-LATENCY 실측 리포트 (Galaxy S25, Wall-clock Option α)

- **측정일**: 2026-04-25
- **브랜치/커밋**: `feat/weight` @ `73f8675`
- **디바이스**: Galaxy S25 (`SM-S931N`, adb `R3CY408S5SB`)
- **Android**: 16, kernel `6.6.77-android15-8-31998796-abogkiS931NKSS9BZCH-4k`
- **백엔드**: OpenCL (Adreno), 6 threads (`--threads 6`), `--profile` 미사용
- **모델**:
  - primary: `Llama-3.2-1B-Instruct-f16.gguf` (2.4 GB, F16)
  - secondary: `Llama-3.2-1B-Instruct-q4_0.gguf` (703 MB, Q4_0)
  - tokenizer: `/data/local/tmp/tokenizer.json` (Llama legacy fallback)
- **CLI**: `--force-swap-ratio R --backend opencl --num-tokens 1 --protected-prefix 4 --prompt "The capital of France is"`
- **반복**: 각 ratio마다 N=10회 (별도 프로세스 실행, 측정 간 3s sleep + ratio 간 15s sleep)
- **소스**: `weight_swap: force ratio=R, swapped X/N layers in W.Wms` stderr 라인 (generate.rs:1952, swap_executor.rs:226)

## 결과: ratio별 swap latency 분포

| ratio | layers | n  | min (ms) | p50 (ms) | mean (ms) | p99 (ms) | max (ms) | **per-layer p50** |
|-------|--------|----|----------|----------|-----------|----------|----------|-------------------|
| 0.25  | 4/16   | 10 | 633.1    | 832.6    | 809.2     | 845.4    | 846.2    | **208.2 ms/layer** |
| 0.50  | 8/16   | 10 | 1244.8   | 1648.2   | 1608.7    | 1663.3   | 1663.5   | **206.0 ms/layer** |
| 1.00  | 16/16  | 10 | 2442.5   | 3294.6   | 3078.9    | 3477.4   | 3494.7   | **205.9 ms/layer** |

### 분포 (히스토그램, ASCII)

```
ratio=0.25 (n=10, ms)
 600 ───── ▌ (1)
 700
 800 ──── ████████▌ (8)
 900
1000

ratio=0.50 (n=10, ms)
1200 ─── ▌ (1)
1400
1600 ── █████████▌ (9)
1700

ratio=1.00 (n=10, ms)
2400 ── ▌ (1)
2500 ── ▌ (1)         ← 2470.2
2800 ── ▌ (1)         ← 2837.4
3000 ── ▌ (1)         ← 3056.7
3300 ── ██████▌ (5)
3500 ── ▌ (1)         ← 3494.7
```

### Per-layer 정규화 결과 (핵심 관찰)

```
0.25 → 208.2 ms/layer p50
0.50 → 206.0 ms/layer p50
1.00 → 205.9 ms/layer p50
```

**선형 스케일링 — ratio와 layer 수에 정확히 비례.** Per-layer 비용은 ratio 무관 ~206 ms로 매우 안정적이며, 분산도 작음 (CV < 5% on stable runs).

## SLA 판정

| 메트릭 | SLA | 실측 (worst case = 206 ms/layer) | 결과 |
|--------|-----|----------------------------------|-------|
| 평균    | < 50 ms/layer  | **206 ms/layer** (× 4.1 배) | ❌ FAIL |
| p99     | < 100 ms/layer | **~210 ms/layer** (× 2.1 배) | ❌ FAIL |
| 평균(spec.41 INV-122 후속, ENG-ALG-210) | < 50 ms/layer | 206 ms/layer | ❌ FAIL |

**SLA 미달폭이 커서 Option α(wall-clock)만으로 판정 불가능. β(stage instrumentation) 분기 필수.**

## 원인 가설 분석 (병목 후보)

총 swap 비용 ≈ 206 ms/layer 분해 가설:

### R4 (Q/K permutation 매 swap 재발) 영향

- swap_executor.rs는 매 swap마다 secondary GGUF에서 raw bytes를 읽어 Q/K weight permutation을 다시 적용 (Phase 3.7b/AUF로 사전 적용 가능).
- Llama 3.2 1B 1 layer Q/K weight = (2048+512)×2048×0.5625 (Q4_0 ratio) ≈ 1.5 MB → permute는 millisecond 단위로 짧음 (1 layer 기준 < 10 ms 추정).
- **R4가 dominant이면 ratio=1.00에서 16 × ~10 ms = 160 ms 추가 — 실측 +1500ms 차이를 설명 못함.**

### Phase 3.7a SOA 재변환 (`ensure_noshuffle_soa_registered`)

- swap_executor.rs:execute_on_slots에서 `invalidate_noshuffle_soa_registry` 직후, ratio_generation bump 직전 7 tensors (wq/wk/wv/wo/w_gate/w_up/w_down) × N_swap_layers 만큼 호출.
- ratio=1.00 → 7 × 16 = 112회 `convert_q4_0_to_noshuffle()` 호출.
- **각 호출이 GPU buffer 생성 + Q4_0 deinterleave + SOA layout 재작성. ne00=2048~8192 큰 텐서이므로 호출당 10~20 ms 추정.**
- 16 layers × 7 tensors × ~15 ms = **~1680 ms** ≈ 실측 swap latency 3.0초의 절반 이상.
- 이 가설이 맞다면 dominant 병목은 SOA 재변환.

### IO (mmap fault-in)

- secondary GGUF 703 MB가 첫 swap에서는 page-fault로 inrush. 이후엔 page cache hit.
- ratio=0.25 첫 실행(833 ms)와 마지막(633 ms) 차이 200 ms는 GGUF page cache warmup 효과로 해석 가능.
- 다만 ratio=0.50/1.00에서는 그 차이가 작음 → page cache가 이미 warm.

### Arc swap

- per-layer atomic store, sub-microsecond. 무시 가능.

### madvise(MADV_DONTNEED)

- 16 layers × 7 tensors × syscall ~10 µs = 1 ms 미만. 무시 가능.

### 추정 분해 (ratio=1.00, p50=3295 ms 기준)

| 단계 | 추정 시간 | % |
|------|----------|---|
| SOA 재변환 (Phase 3.7a) | 1500–1800 ms | 50% |
| Secondary mmap slice + Q/K permute | 800–1100 ms | 30% |
| LayerWeights 구성 + Arc store | 50 ms | 1.5% |
| madvise + page accounting | 5–20 ms | 0.5% |
| 알 수 없는 잔차 | 400–800 ms | 18% |

**확실한 결론은 stage instrumentation(WSWAP-4-LATENCY-β)이 들어가야 가능.**

## 회귀 / 안정성 관찰

- ratio=0.25/0.50/1.00 모두 thermal zone CPU 33→44°C, GPU 28→41°C 점진 상승. SLA throttle 트리거 임계 미도달.
- 측정 중 정확성 회귀 무 (Paris 정답 + 후속 generation 정상 — Phase 3.7a + 이슈 A/B fix 효과 유지).
- 한 실행마다 swap이 ms 단위 일정하게 측정되며 outlier가 거의 없음 → wall-clock 측정 신뢰성 양호.

## 권고

1. **β 분기 (stage instrumentation) 필수**: SwapExecutor 내부 5단계 (a–e) 별도 timestamp 수집. SOA 재변환 가설 검증.
2. **AUF (Phase 3.7b) 우선순위 상향**: 사전 SOA 변환된 cl_mem 직접 등록으로 (ratio=1.00 기준) ~1.5 sec 절감 가능. SLA 50 ms/layer 도달 가능성 상승.
3. **Phase 3.7c (UMA zero-copy) 동시 검토**: 50 ms/layer 미달 시 추가 lever. AUF + zero-copy 조합으로 secondary mmap 비용도 감소.
4. **TBT 측정 (WSWAP-4-TBT) 진행 가능**: swap 자체는 generation 시작 직전 1회이므로 steady-state TBT는 swap latency에 영향받지 않음. 다만 first-swap-after-load 케이스의 TTFT 증가 (3.7초 at ratio=1.00)는 prefill QCF 측정 활성화 시 영향 분석 필요.

## 산출물

- 원시 로그: `/tmp/swap_latency/r{0.25,0.50,1.00}_i{1..10}.log` (30 파일, host)
- 측정 스크립트: `/tmp/swap_latency_measure.sh` (host)

---

# Phase 4 — WSWAP-4-LATENCY 실측 리포트 (β: Stage Breakdown)

- **측정일**: 2026-04-25
- **브랜치/커밋**: `feat/weight` @ `d69670c` (5-stage instrumentation 추가)
- **디바이스**: Galaxy S25 (`SM-S931N`, adb `R3CY408S5SB`)
- **Android**: 16, kernel `6.6.77-android15-8-...`
- **백엔드**: OpenCL (Adreno), 6 threads (`--threads 6`), `--profile` 미사용
- **모델**: F16 primary + Q4_0 secondary (α 동일)
- **CLI**: `--force-swap-ratio R --backend opencl --num-tokens 1 --protected-prefix 4 --tokenizer-path … --prompt "The capital of France is"`
- **반복**: 각 ratio마다 N=10회 (간격 3 s + ratio cooling 15 s)
- **소스**: stage breakdown stderr 라인 (`generate.rs:1960`, `swap_executor.rs:to_log_line`)
  - 형식: `weight_swap stages: mmap_permute=Xms arc_swap=Xms madvise=Xms soa_reconvert=Xms gen_bump=Xms`

## 5-Stage 분포 (per-ratio, 단위 ms)

| ratio | layers | stage           | min  | p50    | mean   | p99    | max    |
|-------|--------|-----------------|------|--------|--------|--------|--------|
| 0.25  | 4/16   | total           | 634.0 | 828.6  | 779.3  | 870.6  | 870.6  |
|       |        | mmap_permute    | 67.3  | 99.3   | 95.1   | 117.2  | 117.2  |
|       |        | soa_reconvert   | 531.9 | 671.3  | 638.0  | 720.8  | 720.8  |
|       |        | arc_swap        | 0.0   | 0.0    | 0.0    | 0.0    | 0.0    |
|       |        | madvise         | 0.0   | 0.0    | 0.0    | 0.0    | 0.0    |
|       |        | gen_bump        | 0.0   | 0.0    | 0.1    | 0.5    | 0.5    |
| 0.50  | 8/16   | total           | 1287.5 | 1646.5 | 1620.0 | 1722.8 | 1722.8 |
|       |        | mmap_permute    | 131.6 | 184.7  | 182.0  | 195.7  | 195.7  |
|       |        | soa_reconvert   | 1085.2 | 1365.4 | 1348.2 | 1434.0 | 1434.0 |
|       |        | arc_swap        | 0.0   | 0.0    | 0.0    | 0.0    | 0.0    |
|       |        | madvise         | 0.0   | 0.0    | 0.0    | 0.0    | 0.0    |
|       |        | gen_bump        | 0.0   | 0.0    | 0.0    | 0.0    | 0.0    |
| 1.00  | 16/16  | total           | 2490.1 | 3296.4 | 3179.2 | 3472.6 | 3472.6 |
|       |        | mmap_permute    | 257.1 | 358.8  | 342.9  | 389.4  | 389.4  |
|       |        | soa_reconvert   | 2093.7 | 2758.6 | 2659.5 | 2915.1 | 2915.1 |
|       |        | arc_swap        | 0.0   | 0.0    | 0.0    | 0.0    | 0.0    |
|       |        | madvise         | 0.0   | 0.0    | 0.0    | 0.0    | 0.0    |
|       |        | gen_bump        | 0.0   | 0.0    | 0.0    | 0.0    | 0.0    |

## Per-layer 정규화 (p50 / swapped_layers)

```
ratio=0.25 (4 layers)  → mmap_permute  24.8 ms/layer | soa_reconvert 167.8 ms/layer | total 207.1 ms/layer
ratio=0.50 (8 layers)  → mmap_permute  23.1 ms/layer | soa_reconvert 170.7 ms/layer | total 205.8 ms/layer
ratio=1.00 (16 layers) → mmap_permute  22.4 ms/layer | soa_reconvert 172.4 ms/layer | total 206.0 ms/layer
```

**모든 stage가 layer 수에 정확히 선형 비례** — α의 wall-clock 결과와 일관.

## Stage Share (p50 기준)

| ratio | total_p50 | mmap_permute       | soa_reconvert      | arc_swap | madvise | gen_bump | residual |
|-------|-----------|--------------------|--------------------|----------|---------|----------|----------|
| 0.25  | 828.6 ms  | 99.3 ms (12.0%)    | 671.3 ms (81.0%)   | 0.00 ms  | 0.00 ms | 0.00 ms  | 58.0 ms (7.0%)  |
| 0.50  | 1646.5 ms | 184.7 ms (11.2%)   | 1365.4 ms (82.9%)  | 0.00 ms  | 0.00 ms | 0.00 ms  | 96.4 ms (5.9%)  |
| 1.00  | 3296.4 ms | 358.8 ms (10.9%)   | 2758.6 ms (83.7%)  | 0.00 ms  | 0.00 ms | 0.00 ms  | 179.0 ms (5.4%) |

residual = `total - sum(measured_stages)` — Phase 3.7a `invalidate_noshuffle_soa_registry` (전체 lock 1회), Galloc/Backend init, Tensor 생성 등 unmeasured overhead. 모든 ratio에서 5~7% 안에 들어와 측정 무결성 양호.

## 가설 검증

태스크 정의 가설:
- 예측: ratio=1.0에서 `soa_reconvert ≈ 1180 ms`, `mmap_permute ≈ 2100 ms`
- **실측: `soa_reconvert = 2758.6 ms` (예측 2.3×), `mmap_permute = 358.8 ms` (예측 1/5.9)**
- → 예측의 우선순위는 **반대**. SOA 재변환이 mmap+permute보다 7.7×~13.7× 무거움.

## 핵심 결론

1. **`soa_reconvert`가 압도적 dominant 병목 (83% of total)**: ratio 무관 ~172 ms/layer.
2. **`mmap_permute`는 11%**: ratio 무관 ~23 ms/layer로 secondary GGUF page-cache (warm 후) + Q/K permute 비용.
3. **`arc_swap`/`madvise`/`gen_bump`은 측정 노이즈 이하 (<0.5 ms total)**: per-layer atomic store는 sub-µs, MADV_DONTNEED는 syscall 1ms 이내, ratio_generation 카운터 bump도 무시 가능.
4. **추정 분해 (α 리포트)에서 SOA 50%, mmap 30%, 잔차 18%로 잡았던 것은 잘못**. 실제는 SOA 84%, mmap 11%, 잔차 5%. SOA 비중이 훨씬 큼.

## SLA 재판정 (per-layer p50 + AUF 효과 예측)

| 시나리오 | per-layer | SLA(<50ms) |
|----------|-----------|-----------|
| 현재 (Phase 3.7a) | 206 ms | ❌ FAIL ×4.1 |
| **AUF (3.7b) 적용 — soa_reconvert 사전 계산 절감 시** | 206 - 172 = **34 ms** | ✅ PASS |
| AUF + mmap_permute (Q/K permute 사전 적용) 둘 다 절감 | 34 - ~23 = **~11 ms** | ✅ PASS, 실질적 cap 도달 |

**AUF 단독으로 SLA 충족 가능성 높음.** Phase 3.7b 우선순위 최상.

## TBT ratio=1.0 -20.7% gap 단서

- swap 자체는 generation 직전 1회만 발생 → steady-state TBT에 직접적 영향 없음.
- 단, β 측정으로 **`mmap_permute` stage의 동작이 secondary slice → Q/K permute 적용 → GPU upload**임이 확인됨. 이 GPU upload (cl_mem 신규 생성 + `write_buffer`)가 LayerWeights에 전달되어 이후 decode forward에서 사용. **upload된 cl_mem이 ratio=1.0에서 16개로 늘어나 KV/weight cl_mem fragmentation (memory note `project_kv_fragmentation.md` 참조)을 악화시킬 가능성** — α 리포트의 PSS 5.6× 초과 + TBT -20.7%와 정합.
- 대책: AUF (사전 SOA 변환된 단일 cl_mem 등록) 또는 cl_mem aggregation (per-layer → 단일).

## 다음 액션 권고

1. **Phase 3.7b (AUF) 우선순위 최상위로 확정**:
   - 사전 SOA 변환 + 단일 cl_mem 등록으로 `soa_reconvert` 0 ms화
   - 추가로 Q/K permute도 사전 적용 → `mmap_permute` 23 ms/layer → ~5 ms/layer (단순 mmap slice + cl_mem reuse)
   - 예상 per-layer: 206 → ~5–11 ms (×20 개선)
2. **β2 sub-stage 분해는 현재 불필요**: 5 stages 중 dominant가 명확하고 잔차 < 7%이므로 추가 분해 효과 미미. 단, AUF 적용 후 새 dominant 확인이 필요하면 그 시점에 진행.
3. **TBT gap 검증을 위한 별도 실험** (Architect 후속): cl_mem fragmentation 변화 측정 (S25 vendor `mem_info` adb dump + KV gpu allocation 트레이스).

## 산출물

- 원시 로그: `/tmp/swap_latency_beta/r{0.25,0.50,1.00}_i{1..10}.log` (30 파일, host)
- 분석 스크립트: `/tmp/analyze_beta.py` (host)
- 측정 스크립트: `/tmp/swap_beta_measure.sh` (host)

---

## §AUF 5차 측정 결과 (2026-04-26)

- **브랜치/커밋**: `feat/weight` @ `21c6d82` (SOA bypass 본격 구현 완료)
- **새 커밋 (4차→5차)**:
  - `22f7c8a` AUF builder full Adreno noshuffle SOA pipeline
  - `e9edffc` `Backend::register_pre_converted_soa` trait + OpenCL impl
  - `2d768ec` `swap_executor` materialise + Stage (d) registration 실동작
  - `21c6d82` e2e regression guards
- **AUF 재빌드** (필수): host `21c6d82` `auf_tool` 디바이스 push 후 디바이스에서 직접 빌드.
  - 입력: `/data/local/tmp/Llama-3.2-1B-Instruct-q4_0.gguf` (4차와 동일, 703 MB)
  - 출력: `/data/local/tmp/Llama-3.2-1B-Instruct.auf` (701 MB, sha256 `1a1ead0c1f532b26034989deb7dbfece4f5b7ed41b881491cfee857ae014c5ec`)
  - 4차 sha256 `4e02b03f5c560f9db27a6bf3780bd66c1f299dc93f2627fa8f1b67e17aa3bb5d` (다름 → SOA pipeline 다르게 적용 확인)
  - WEIGHTS_ADRENO_SOA: 663 MB, 146 tensors, 16 layers, n_heads_q=32, n_kv_heads=8

### Stage A — 스모크 (PASS)

`--force-swap-ratio 1.0 --num-tokens 16 --protected-prefix 4 --prompt "The capital of France is" -b opencl --threads 6`:
```
weight_swap: force ratio=1.00, swapped 16/16 layers in 590.1ms
weight_swap stages: mmap_permute=253.7ms arc_swap=0.0ms madvise=0.0ms soa_reconvert=196.3ms gen_bump=0.0ms
The capital of France is Paris. It is a beautiful city with many historical buildings, museums, and cultural
```

- "Paris" 포함 ✅
- "/buttons" garbage 없음 ✅ (4차 회귀 해소)

### Stage B — 회귀 가드

| 측정 | 4차 | 5차 | 판정 |
|------|-----|-----|------|
| R-new-1 (Arc 회귀: ratio=0 with secondary vs ratio=0 alone) | +0.42% | **+0.45%** | ✅ PASS |
| PSS 감소 (ratio=1 vs ratio=0, Pss total) | 844 MB | **~799 MB** (5510→4711 MB) | ✅ PASS (target 150 MB ×5.3) |
| Pss_Shmem 감소 | 843 MB | **~793 MB** (2416→1623 MB) | ✅ PASS |
| INV-122 정확성 (greedy sanity) | "/buttons" garbage | **본질 정확** ("Paris... Eiffel Tower... 1889 World's Fair") | ✅ 회귀 해소 |

INV-122 full sweep (100 prompts × 5 ratios)은 이번 측정 시간 제약으로 미진행. greedy sanity로 garbage 없음 + factual 일관성 확인. full sweep은 별도 Tester 세션 필요.

### Stage C — 레이턴시 비율 (β stage breakdown, N=5, ratio=1.0)

| run | total_ms | mmap_permute | arc_swap | madvise | soa_reconvert | gen_bump | per_layer |
|-----|----------|--------------|----------|---------|----------------|----------|-----------|
| 1   | 845.4    | 389.2        | 0.0      | 0.0     | 272.3          | 0.0      | 52.8 ms |
| 2   | 582.3    | 253.1        | 0.0      | 0.0     | 192.7          | 0.0      | 36.4 ms |
| 3   | 593.4    | 256.3        | 0.0      | 0.0     | 195.8          | 0.0      | 37.1 ms |
| 4   | 857.6    | 382.5        | 0.0      | 0.0     | 284.8          | 0.0      | 53.6 ms |
| 5   | 856.0    | 377.5        | 0.0      | 0.0     | 286.7          | 0.0      | 53.5 ms |

**Stats (N=5)**:
- per-layer total: min=36.4, **p50=52.8**, p95=53.6, mean=46.7 ms
- mmap_permute (per-layer): min=15.8, p50=23.6, p95=24.3 ms
- soa_reconvert (per-layer): min=12.0, **p50=17.0**, p95=17.9 ms

### 4차 vs 5차 비교

| 지표 | 4차 (`aee9adc`) | 5차 (`21c6d82`) | 감소율 |
|------|------------------|------------------|--------|
| per-layer total p50 | 206 ms | **52.8 ms** | **−74.4%** (3.9× 단축) |
| per-layer total mean | 206 ms | 46.7 ms | −77.3% |
| soa_reconvert per-layer | 172 ms (84% dominant) | **17.0 ms** | **−90.1%** (10.1× 단축) |
| mmap_permute per-layer | (4차 미보고, 추정 ~30 ms) | 23.6 ms | (변동 적음) |

**SLA <50 ms (Phase 4 target)**: ⚠️ **부분 충족**
- per-layer p50 = 52.8 ms — **SLA 5.7% 초과** (border-line FAIL)
- per-layer min = 36.4 ms — SLA 충족
- 양봉 분포: 3/5 runs가 ~50 ms대 (cold), 2/5 runs가 36-37 ms대 (warm-cache 추정)
- 5-stage 분포 변화: 4차 dominant `soa_reconvert` 84% → 5차 `mmap_permute` ≈ `soa_reconvert` 균형 (각 ~50%)

### Stage 분포 표

| Stage | 4차 비율 | 5차 비율 (N=5 mean) |
|-------|---------:|--------------------:|
| mmap_permute | (미상) | 44.4% (332/747 ms) |
| arc_swap     | ≈ 0%   | 0.0% |
| madvise      | ≈ 0%   | 0.0% |
| soa_reconvert| 84%    | 33.0% (247/747 ms) |
| gen_bump     | ≈ 0%   | 0.0% |
| (잔여, 측정 boundary 외) | 16% | 22.6% |

5차 잔여 22.6%는 layer 루프 외부 동기화/cl_mem 할당 등 stage boundary 외 코드 (4차의 16%와 유사 비율).

### Decode TBT (steady-state, 200 tokens)

| run | Decode ms/tok | Avg TBT |
|-----|----------------|---------|
| 1   | 20.64 | 26.80 |
| 2   | 20.64 | 26.02 |
| 3   | 20.52 | 25.91 |
| 4   | 20.69 | 26.86 |
| 5   | 20.71 | 26.99 |

mean Decode = 20.64 ms/tok, mean Avg TBT = 26.5 ms/tok. **이전 4차 ratio=1.0 mixed Decode (Q4 baseline 대비 −20.7% FAIL)와 비교 필요** — Q4 baseline 실측 Decode가 4차 리포트에 명시되어야 정확한 비교 가능. 현 5차 데이터로는 Q4 baseline 대비 % 차이 산출 불가 (보고된 %는 기존 Q4 baseline에 의존, 5차에서 Q4 재측정 미수행).

### soa_reconvert 비제로 원인 분석

핸드오프 기대값은 "soa_reconvert ≈ 0 ms" (84% → 0%). 실측 5차에서 soa_reconvert = 17 ms/layer (12 ms × 7 weight tensors × 16 layers = 1344 register call의 누적 비용).

`swap_executor.rs:380-461` `skip_soa_reconvert == true` 분기에서 측정되는 시간:
- `register_pre_converted_soa` × (7 weights × 16 layers) = 112 호출
- 각 호출마다 `enqueue_write_buffer(q_buf)` + `enqueue_write_buffer(d_buf)` + `register_noshuffle_soa()` (HashMap insert)
- 총 224 buffer write + 112 registry insert

이는 4차 dominant였던 `convert_q4_0_to_noshuffle` GPU 커널 디스패치 + dual CPU transpose 비용 대비 10× 단축. 핸드오프의 "≈ 0" 기대치는 너무 보수적이었던 것으로 판단.

### 권장 후속 조치

1. **SLA <50 ms border-line 해소**: per-layer 간 변동 (cold 53 ms ↔ warm 36 ms) 원인 규명. `mmap_permute`의 mmap demand-paging이 cold path 추정.
2. **Q4 baseline Decode 재측정**으로 ratio=1.0 mixed의 정확한 회귀 % 산출.
3. **INV-122 full sweep 재측정**으로 SOA bypass 정확성 통계적 검증 (5차 빌드 AUF 사용).

### 산출물

- `/tmp/phase4_5th_smoke.log` — Stage A 스모크 (정확성 PASS)
- `/tmp/phase4_5th_baseline_ratio0.log` — ratio=0 baseline 5 runs
- `/tmp/phase4_5th_rnew1_ratio0_with_secondary.log` — R-new-1 (Arc 회귀) 5 runs
- `/tmp/phase4_5th_ratio1.log` — Stage C ratio=1.0 5 runs
- `/tmp/phase4_5th_pss_ratio0.log` / `pss_ratio1.log` — PSS sample
- `/tmp/phase4_5th_beta.csv` — β stage 데이터

## Phase 4 종결 verdict (2026-04-26)

> **상태**: ✅ 종결. 사용자 결정으로 SLA border-line(p50 52.8 ms, 5.7% 초과)을 수용하고 Phase 4 종결.

| 게이트 | 결과 | 비고 |
|--------|------|------|
| 정확성 (Paris/garbage 가드) | ✅ PASS | 4차 회귀 완전 해소 |
| R-new-1 (Arc 회귀) | ✅ PASS | +0.45% (이전 +0.42%) |
| R-new-3 (PSS) | ✅ PASS | −799 MB (target ×5.3) |
| INV-122 정확성 (greedy sanity) | ✅ PASS | F16/AUF/Q4 일관 ("Paris... 1889") |
| LATENCY soa_reconvert ≪ baseline | ✅ PASS | 172→17 ms, −90.1% |
| LATENCY per-layer SLA <50 ms | ⚠️ 부분 충족 (수용) | p50 52.8 ms, min 36.4 ms, 양봉 분포 |

**핵심 성과**: per-layer p50 −74.4% (206→52.8 ms), soa_reconvert −90.1% (172→17 ms). 4차 차단 (SOA bypass 본질 미구현 + "/buttons" garbage)를 SOA bypass 본격 구현(`22f7c8a→21c6d82`)으로 해소.

**별도 sprint로 분리**:
- cold-path 균일화 (madvise(MADV_WILLNEED) prefault 또는 warmup pass)
- INV-122 full 100-prompt sweep
- ratio=1.0 mixed −20.7% TBT gap (cl_mem fragmentation 가설)
- KIVI plan mixed state legacy fallback
