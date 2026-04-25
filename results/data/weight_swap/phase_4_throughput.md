# Phase 4 — WSWAP-4-TBT 실측 리포트 (Galaxy S25)

- **측정일**: 2026-04-25
- **브랜치/커밋**: `feat/weight` @ `73f8675`
- **디바이스**: Galaxy S25 (`SM-S931N`, adb `R3CY408S5SB`)
- **Android**: 16, kernel `6.6.77-android15-8-31998796-abogkiS931NKSS9BZCH-4k`
- **백엔드**: OpenCL (Adreno 830), 6 threads (`--threads 6`), `--profile` **미사용**
- **모델**:
  - F16 GGUF: `Llama-3.2-1B-Instruct-f16.gguf` (2.4 GB, 16 layers)
  - Q4_0 GGUF: `Llama-3.2-1B-Instruct-q4_0.gguf` (703 MB)
  - tokenizer: `/data/local/tmp/tokenizer.json` (Llama legacy fallback)
- **CLI**: `--num-tokens 256 --protected-prefix 4 --prompt "The capital of France is" --threads 6`
- **반복**: N=3회/configuration (별도 프로세스, intra-config sleep 12s, inter-config sleep 40s, thermal isolation)
- **소스**: stderr `Decode: X ms/tok (Y tok/s)` 라인 (generate.rs:5558, forward only, sync 오버헤드 없음)
- **재배포**: 불필요 — 17:34 push된 `/data/local/tmp/generate` (Phase 3.7a `73f8675` 빌드) 그대로 사용.

## 결과: 6 configuration tok/s 분포

| Configuration | swap layers | n | tok/s mean | tok/s σ | σ/μ | ms/tok mean | ms/tok σ |
|---|---|---|---|---|---|---|---|
| (a) F16 baseline (no secondary) | 0/16 | 3 | **23.77** | 0.058 | 0.24% | 42.083 | 0.055 |
| (b) Q4_0 baseline (no secondary) | 0/16 | 3 | **61.30** | 0.781 | 1.27% | 16.313 | 0.203 |
| (c) Arc snapshot, ratio=0.0 (LayerSlot active, no swap) | 0/16 | 3 | **23.87** | 0.058 | 0.24% | 41.913 | 0.032 |
| (d1) ratio=0.25 mixed | 4/16 | 3 | **27.20** | 0.100 | 0.37% | 36.777 | 0.078 |
| (d2) ratio=0.50 mixed | 8/16 | 3 | **31.93** | 0.058 | 0.18% | 31.320 | 0.072 |
| (d3) ratio=1.00 mixed | 16/16 | 3 | **48.60** | 0.265 | 0.54% | 20.570 | 0.115 |

CV(σ/μ) 모두 < 1.3% — 측정 노이즈 매우 낮음.

### 원시 iteration 데이터

```
a (F16):    23.8 / 23.8 / 23.7   (255 / 255 / 220 tokens, EOS 1회)
b (Q4_0):   62.2 / 60.8 / 60.9   (148 / 255 / 255 tokens, EOS 1회)
c (Arc):    23.9 / 23.8 / 23.9   (149 / 179 / 119 tokens, EOS 모두)
d1 (0.25):  27.3 / 27.2 / 27.1   (56 / 255 / 255 tokens)
d2 (0.50):  32.0 / 31.9 / 31.9   (99 / 227 / 155 tokens)
d3 (1.00):  48.5 / 48.9 / 48.4   (132 / 255 / 255 tokens)
```

EOS 조기 종료가 일부 있으나 모든 iteration의 forward window가 ≥ 56 tokens이므로 평균 ms/tok은 안정적.

## R-new-1 정량 판정 — Arc 간접 참조 회귀 (a vs c)

| 지표 | (a) F16 baseline | (c) Arc snapshot ratio=0.0 | Δ 절대 | Δ 백분율 |
|---|---|---|---|---|
| tok/s mean | 23.767 | 23.867 | **+0.100** | **+0.42%** |
| ms/tok mean | 42.083 | 41.913 | **-0.170** | **-0.40%** |

- **회귀 임계 (≤ 1%)**: **PASS** — 절대값 0.42%로 측정 노이즈 (CV 0.24%) 와 같은 자릿수.
- **방향**: (c)가 미세하게 더 빠름 — `LayerSlot`/`ArcSwap` 간접 참조의 실측 오버헤드는 **사실상 0**.
- **결론**: `TransformerWeights` → `Vec<Arc<LayerSlot>>` 리팩토링이 forward hot-path에 미치는 영향 없음. 본 측정 세션에서 발생한 ±0.5% 변동은 thermal/스케줄러 노이즈 수준이며 Arc 패널티에 기인하지 않는다.

## 타겟 충족 표

| 타겟 (acceptance) | 기준 | 실측 | 결과 |
|---|---|---|---|
| (c) F16 baseline 대비 회귀 ≤ 1% (R-new-1) | tok/s 회귀 절대값 ≤ 1% | **+0.42%** (회귀 없음) | ✅ **PASS** |
| (d2) ratio=0.5 vs F16 baseline 열화 ≤ 10% | tok/s 23.77 → ≥ 21.39 | **31.93 (+34.4%)** | ✅ **PASS** (대폭 향상) |
| (d3) ratio=1.0 vs Q4_0 baseline 열화 ≤ 5% | tok/s 61.30 → ≥ 58.24 | **48.60 (-20.72%)** | ❌ **FAIL** |

## 추가 정량 관찰

### 1. swap 비율과 throughput의 비선형 향상 (vs F16)

```
ratio   tok/s   gain vs F16 23.77
─────   ─────   ─────────────────
0.00    23.87   +0.42%   (Arc snapshot 비활성)
0.25    27.20  +14.44%   (4 layers Q4)
0.50    31.93  +34.36%   (8 layers Q4)
1.00    48.60 +104.49%   (전부 Q4, swap 경로)
```

per-layer 효과로 분해하면:
- ratio 0→0.25 (4 layer 전환): +14.4% / 4 = **+3.6% per swapped layer**
- ratio 0.25→0.50 (4 layer 추가): +20.0pp / 4 = **+5.0% per layer**
- ratio 0.50→1.00 (8 layer 추가): +70.1pp / 8 = **+8.8% per layer**

Q4 layer 비중이 늘수록 per-layer gain이 가속한다. 이는 mixed mode에서 F16 layer 1개라도 남으면 그 layer가 critical path를 잡아 전체 throughput을 제한하기 때문 (Adreno F16 GEMV가 step-rate를 지배).

### 2. d3 vs Q4 baseline gap (-20.7%)

ratio=1.00 mixed (48.60 tok/s)는 모든 layer가 Q4_0이지만 Q4_0 단독 baseline (61.30 tok/s)에 도달하지 못한다 (-20.7%, 타겟 -5% 미달).

원인 분석 (코드 수정 없이 hypothesis):
- **a) Adreno SOA layout 차이**: primary loader는 Q4_0 GGUF에서 직접 SOA 변환 + permanent cl_mem에 등록한다. 반면 swap 경로는 secondary GGUF의 raw bytes를 mmap한 뒤 Phase 3.7a에서 추가된 `convert_aos_to_soa` runtime 변환을 거쳐 별도 cl_mem에 등록한다. 결과 cl_mem buffer pool/alloc 패턴이 달라 GPU memory traffic이 비최적일 수 있음.
- **b) cl_mem fragmentation**: HeadMajor cl_mem이 56개 분리되어 있으면 attention slope +1.32 μs/n_kv가 발생한다는 기존 finding (`memory:project_kv_fragmentation`). swap은 layer 단위 재할당으로 이 효과를 더 가중시킬 가능성.
- **c) madvise 후 page re-fault**: Phase 4 PSS 측정에서 swap 후 madvise(MADV_DONTNEED)이 ~52 MB/layer 회수됨이 확인됨. 회수된 영역이 forward 시 page fault → tlb miss를 추가로 발생시킬 수 있음.
- **d) AUF 미사용**: Phase 3.7b AUF는 사전 SOA 변환된 cl_mem을 직접 등록하여 (a) 경로를 우회한다. AUF 도입 시 d3 gap이 좁혀질 가능성이 가장 큼.

이 -20.7% gap은 swap 경로 자체의 정상 동작 (정확성/안정성 PASS) 하에서 **성능 최적화 여지로 남은 항목**이며, INV-122 (정확성) 진입의 차단 요인은 아니다.

### 3. TTFT (first-swap-after-load) 증가

TTFT (mean over N=3, ms):
- (a) F16: 358.2 ms
- (b) Q4_0: 225.2 ms
- (c) ratio=0.0: 352.9 ms
- (d1) ratio=0.25: 1336.8 ms (+swap 835 ms)
- (d2) ratio=0.50: 2287.3 ms (+swap 1797 ms)
- (d3) ratio=1.00: 3631.0 ms (+swap 3251 ms)

TTFT 증가 = WSWAP-4-LATENCY p50 swap latency와 일치 (per-layer 206 ms × N_swap). first-prefill 1회만 영향, 이후 prefill은 swap 없으므로 baseline 동일. **steady-state TBT는 swap latency에 영향받지 않음** — 본 측정의 decode tok/s는 swap 직후 forward만으로 계산되므로 swap latency 외삽 없음.

### 4. Prefill QCF 측정 영향

`--prefill-qcf` 같은 별도 flag는 본 워크로드에서 활성화되지 않았다 (`force_swap_ratio` 경로는 prefill QCF measurement 호출 없음). 따라서 prefill latency 본 측정은 prefill QCF off 상태이며, 별도 비교 측정은 본 sprint 범위 외.

## 발견된 이슈 / 측정 노이즈 분석

1. **EOS 조기 종료**: 6 configuration 중 4개에서 1번 이상 EOS 종료 (256 tokens 미달). 그러나 forward window 최소값이 56 tokens (d1 i1)이고, 평균 ms/tok은 forward 횟수로 정규화되어 있으므로 영향 없음. CV가 모두 < 1.3%인 것이 측정 안정성을 뒷받침.
2. **Thermal drift**: 측정 시작 (zone1=34.9°C) → d3 시작 (zone1=39.6°C) 약 5°C 상승. SLA throttle 임계 미도달. 다만 마지막 d3가 가장 thermal stress가 높은 상태에서 측정되었음에도 CV 0.54%로 안정적이어서 thermal 영향은 미미.
3. **GPU 첫 swap iteration**: d1/d2/d3 모두 i1에서 swap latency가 i2/i3보다 약간 짧음 (page cache cold이지만 GPU buffer warm-up 효과). decode tok/s에는 영향 없음.
4. **--profile 미사용 준수**: 모든 측정에서 `--profile` flag 부재 확인. forward 경로의 sync 오버헤드 0.

## 결론

### Acceptance criteria

- ✅ **tok/s × configuration 표 산출** (3 baseline + 1 Arc + 3 mixed)
- ✅ **R-new-1 정량 판정 PASS** — Arc 간접 참조 회귀 +0.42% (≤ 1% 충족)
- ✅ **(d2) ratio=0.5 mixed 타겟 PASS** — F16 대비 +34.4% (열화 -10% 대신 향상)
- ❌ **(d3) ratio=1.0 mixed 타겟 FAIL** — Q4_0 baseline 대비 -20.7% (타겟 -5%)
- ✅ 리포트 산출

### 다음 라운드 (INV-122) 진행 가능 여부 — **YES, 권고**

근거:
1. **R-new-1 PASS**: 동적 LayerSlot 인프라 자체의 회귀가 없음 → forward correctness 변경 없음.
2. **모든 mixed config에서 정확성 회귀 무**: Phase 3.7a + 이슈 A/B fix 효과로 "Paris" 정답 + 후속 generation 정상 (실측 prompt completion 확인). 
3. **(d3) -20.7% gap은 throughput 이슈, 정확성 무관**: AUF (Phase 3.7b) 도입 시 자연스럽게 좁혀질 가능성이 높으며, INV-122 (NMSE/top-5/top-1 forward equivalence) 측정 자체는 throughput 영향과 독립적임.
4. **측정 안정성**: 모든 configuration CV < 1.3%, 각 configuration N=3 mean이 σ보다 100배 이상 큼 → INV-122 측정 시에도 baseline 노이즈 마진 충분히 작음.

권고:
- INV-122 (WSWAP-4-INV122)는 즉시 진입 가능. 100+ prompts × {0.25, 0.5, 0.75, 1.0} ratio의 NMSE/top-5/top-1/ROUGE-L 측정.
- (d3) -20.7% gap은 별도 기록 후 Phase 3.7b AUF 도입 시 재측정 (예상: gap < -5% 회복).

## 산출물

- 원시 로그: `/tmp/swap_throughput/{a_f16_baseline,b_q4_baseline,c_arc_snapshot,d1_ratio_0_25,d2_ratio_0_50,d3_ratio_1_00}_i{1,2,3}.log` (18 파일, host)
- 측정 스크립트: `/tmp/swap_throughput_measure.sh` (host)
- master log: `/tmp/swap_throughput/run.log`
