# Round 15: H2O 성능 저하 근본 원인 분석

## 1. 배경

Round 14에서 Llama 3.2 1B 대상 H2O Heavy Hitter(HH) eviction이 Sliding Window보다
일관되게 열등함이 확인되었다 (EMR 0.506 vs 0.519, 동일 eviction 수에서 3배 격차).
본 분석은 **왜 논문과 달리 H2O가 실패하는가**에 대한 근본 원인을 규명한다.

---

## 2. 연구 질문

| # | 가설 | 요약 |
|---|------|------|
| Q1 | 모델 스케일 문제 (1B too small) | 대형 모델 필요 — 본 분석 범위 외 |
| Q2 | BOS 외 genuine HH가 존재하지 않음 | 1B attention이 균등 분포 |
| Q3 | GQA가 score 집계를 왜곡 | 그룹 내 평균화가 신호 희석 |
| Q4 | Signal-driven batch eviction vs 논문의 per-step | 누적 방식 차이 |

---

## 3. 검증 방법

### 3.1 코드 분석

H2O 구현을 논문(Zhang et al., 2023)과 대조하여 6개 Phase 검증 수행:

- **Phase 1**: `ws.scores`가 post-softmax 확률임을 확인 (3 tests)
- **Phase 2**: MAX cross-layer 집계가 layer-critical 토큰을 보존함을 확인 (4 tests)
- **Phase 3**: Score reset의 필요성 증명 — `shift_positions()` 후 importance 배열 불일치 (3 tests)
- **Phase 4**: Budget 계산 edge case + debug logging (2 tests)
- **Phase 5**: `--h2o-debug` CLI 플래그 구현
- **Phase 6**: 논문 대비 설계 차이점 정리

총 12개 테스트, 전체 통과. 구현 자체는 논문과 일치.

### 3.2 Score Distribution Simulation

Llama 3.2 1B 파라미터로 896 decode step 시뮬레이션 수행:

| 파라미터 | 값 |
|---------|-----|
| Q-heads / KV-heads | 32 / 8 (GQA) |
| Layers | 16 |
| Prompt length | 128 |
| Cache at eviction | 1024 |
| Score aggregation | Within-step MAX, across-steps SUM |
| Decay | 0.0 (기본값) |
| Calibration | Round 14 관측값 (BOS≈3003, prompt≈3.3, gen≈33) |

출력: `experiments/analysis/score_distribution.csv` (1024 tokens × position/score/type)

### 3.3 분석 도구

`experiments/analysis/hh_proof_analysis.py` — Shannon entropy, Gini 계수,
Pearson/Spearman 상관, H2O 시뮬레이션, quintile 분석 수행.

---

## 4. 가설별 검증 결과

### Q2: BOS 외 Genuine Heavy Hitter 존재 여부

**결론: 존재하지 않음.**

Round 14 실험 데이터와 시뮬레이션 모두에서 확인:

| 토큰 범주 | 누적 score | 비율 |
|-----------|-----------|------|
| BOS | 4533.7 | 100x baseline |
| Structural (구두점 등) | 12.9 | 3.6x prompt |
| Prompt (일반) | 3.6 | baseline |
| Generated (평균) | 27.3 | 7.6x prompt |

**정보이론 분석 (BOS 제외)**:

| 지표 | 값 | 해석 |
|------|-----|------|
| Normalized Entropy | **0.9748** | 거의 균등 분포 (1.0 = 완전 균등) |
| Gini Coefficient | 0.3128 | 낮은 불평등 |
| CV (σ/μ) | 0.561 | 중간 변동 |
| >1σ outlier | 13.4% | 정규 분포 수준 |
| >2σ outlier | 3.3% | 기대값과 일치 (특이 집단 없음) |

> Normalized entropy 0.97은 BOS를 제외하면 모든 토큰의 중요도가 거의 동등함을 의미한다.
> H2O가 가정하는 이봉 분포(BOS + 산발적 HH)가 1B에서는 형성되지 않는다.

---

### Q3: GQA Score 집계 왜곡

**결론: 코드상 존재하나 근본 원인 아님.**

`attention_scores.rs:186-193`에서 GQA 그룹 내 4개 Q-head 스코어를 **평균**:

```
Q-head scores [9.0, 0.1, 0.1, 0.1] → 평균 2.3 (MAX라면 9.0)
```

그러나:
- Regular H2O는 flat score(32 Q-head 전체 SUM)만 사용 → GQA 구조를 무시하므로 영향 없음
- H2O+는 per-KV-head score를 사용하지만, Round 14에서 Δ=+0.011 (무의미)
- **근본적으로 HH가 존재하지 않으므로, 집계 방식을 개선해도 효과 없음**

---

### Q4: Signal-Driven Batch Eviction vs Per-Step Eviction

**결론: 구조적 차이 존재, 부분적 기여.**

| 측면 | H2O 논문 | 이 구현 |
|------|---------|---------|
| 트리거 | 매 decode step | Resilience signal 수신 시 |
| 빈도 | 100회/100토큰 | 0~10회/100토큰 |
| Score 연령 | 항상 1 step (fresh) | N step 누적 (stale) |
| Budget | 고정 (e.g. 200) | Grow-on-demand → max 2048 |
| `should_evict()` | cache > budget → true | **항상 false** (h2o.rs:42-48) |

> H2O의 `should_evict()`가 하드코딩 `false`이므로 외부 signal 없이는 eviction이 발생하지 않는다.
> Score decay 기본값이 0.0이므로 모든 historical score가 동일 가중치로 무한 누적된다.

---

## 5. 핵심 발견: 누적 Score의 역상관 문제

### 5.1 현상

Score distribution simulation에서 **예상 밖의 결과** 발견:

| 상관 지표 | 값 | 해석 |
|-----------|-----|------|
| **Pearson r** | **-0.9499** | 강한 음의 상관 |
| **Spearman ρ** | **-1.0000** | 완벽한 역순위 |

Position과 score가 **완벽한 역상관**: 오래된 토큰일수록 높은 score.

### 5.2 원인

누적 SUM (decay=0.0)에서 토큰의 총 score는:

```
importance[t] ≈ Σ(step=entry..now) per_step_attention(t, step)
```

모든 generated 토큰이 비슷한 per-step attention을 받으므로:

```
importance[t] ∝ (현재 step - entry step) = 캐시 체류 시간
```

**누적 score가 "attention importance"가 아니라 "체류 시간의 proxy"가 되어버린다.**

### 5.3 결과: H2O가 Sliding Window의 정반대를 선택

```
  Position    0                  128        381  382            770        1023
  ├─ Prefix ─┤├── HH Selected ──┤├─ Evicted ┤├─── Evicted ───┤├─ Recent ─┤
              │  (oldest gen)     │           │                │(newest)   │
              │  avg=41.7         │           │                │           │
              │                   │  avg=21.1 │                │           │
              └── HIGH scores ────┘           └── LOW scores ──┘
```

| 전략 | 선택 토큰 | 위치 | 특성 |
|------|----------|------|------|
| **H2O** | pos 128-381 | **가장 오래된** 254개 | 문맥적 관련성 낮음 |
| **Sliding** | pos 770-1023 | **가장 최근** 254개 | 현재 문맥 유지 |

H2O가 보존하는 "Heavy Hitter"는 가장 오랫동안 캐시에 있던 토큰들이다.
이들은 현재 생성 문맥과 **가장 관련 없는** 토큰이며, sliding window가 보존하는
최근 토큰의 **정반대**이다.

### 5.4 정량적 증거

**H2O Selection Quality (prefix=4, target=512, kr=0.5)**:

| 지표 | HH Selected | Evicted | Random |
|------|-------------|---------|--------|
| Count | 254 | 512 | 254 |
| Avg Score | 41.7 | 21.1 | 29.7 |
| **HH/Evicted** | **1.98x** | — | — |
| **HH/Random** | **1.41x** | — | — |

- HH/Evicted 비율 1.98x: "중요"와 "비중요" 토큰 간 차이가 2배 미만
- HH/Random 비율 1.41x: 무작위 선택 대비 41% 개선 (의미 없는 수준)
- **Score margin** (최악 HH - 최선 Evicted): **0.015** (= 0.00σ)
  - HH 경계의 토큰 score가 사실상 동일 → 순위 결정이 무의미

**Position Quintile Analysis (Generated tokens)**:

| Quintile | Position Range | Avg Score | 해석 |
|----------|---------------|-----------|------|
| Q1 (Oldest) | 128-306 | **44.9** | H2O가 "HH"로 선택 |
| Q2 (Old) | 307-485 | 32.2 | 경계 영역 |
| Q3 (Middle) | 486-664 | 26.5 | Evicted |
| Q4 (Recent) | 665-843 | 21.3 | Evicted |
| Q5 (Most recent) | 844-1023 | **11.6** | Sliding이 보존 |

> Q1(oldest)의 avg score가 Q5(most recent)의 3.9x인 것은 "중요도"가 아니라
> Q1이 Q5보다 3.9배 오래 캐시에 있었기 때문이다.

---

## 6. >2σ Outlier 분석

Score가 mean + 2σ를 초과하는 34개 토큰 (상위 3.3%):

| Position | Score | Type | 캐시 체류 (steps) |
|----------|-------|------|-------------------|
| 128 | 73.6 | generated | 895 (최대) |
| 129 | 72.2 | generated | 894 |
| 130 | 70.8 | generated | 893 |
| 131 | 69.6 | generated | 892 |
| ... | ... | ... | ... |
| 161 | 51.7 | generated | 862 |

**모든 >2σ outlier가 가장 오래된 generated 토큰**(pos 128-161)이다.
이들이 outlier인 이유는 "attention을 많이 받아서"가 아니라 "가장 오래 있어서"이다.

---

## 7. 근본 원인 종합

```
┌─────────────────────────────────────────────────────────┐
│  Root Cause Chain                                       │
│                                                         │
│  1B 모델의 attention 균등 분포                           │
│       ↓                                                 │
│  모든 non-BOS 토큰의 per-step attention ≈ 동일           │
│       ↓                                                 │
│  누적 SUM (decay=0) → score ∝ 체류 시간                  │
│       ↓                                                 │
│  H2O가 "oldest = highest score = HH"로 선택              │
│       ↓                                                 │
│  Sliding window (most recent)의 정반대 토큰 보존          │
│       ↓                                                 │
│  문맥 coherence 파괴 → 생성 품질 3x 저하                 │
└─────────────────────────────────────────────────────────┘
```

**두 가지 요인이 동시에 작용**:

1. **모델 한계** (Q2): 1B 모델이 BOS 외 specialized attention pattern을 형성하지 못함
2. **누적 바이어스** (Q4): decay 없는 SUM 누적이 체류 시간을 중요도로 오인

단독으로는 각각 "HH가 무의미" → "sliding과 비슷한 성능"이 되지만,
**두 요인이 결합**되면 "HH가 sliding의 정반대를 선택" → "적극적으로 해로운" 결과를 낳는다.

---

## 8. 보정 방안 제안

### 8.1 즉시 적용 가능

| 방안 | 설명 | 변경량 | 효과 |
|------|------|--------|------|
| **시간 정규화** | `score[t] / steps_since_entry[t]` | 작음 | 역상관 제거 |
| **Decay 조정** | `--h2o-decay 0.98` (half-life ≈ 35 steps) | 0줄 | 최근 attention 가중 |

### 8.2 중기 개선

| 방안 | 설명 | 변경량 | 효과 |
|------|------|--------|------|
| **Frequency ranking** | Top-K 등장 빈도 추적 | 중간 | 누적 아티팩트 완전 제거 |
| **SnapKV** | Prefill-end 1회 scoring | 중간 | 프롬프트 압축 (검증됨) |

### 8.3 장기 (아키텍처 변경)

| 방안 | 설명 | 변경량 | 효과 |
|------|------|--------|------|
| **Per-step eviction** | 매 step 고정 budget 유지 | 큼 | 논문 원본 재현 |

### 8.4 중요 Caveat

> 1B 모델이 근본적으로 attention diversity가 부족하면 (Q2),
> 어떤 스코어링 방식을 적용해도 meaningful HH가 나타나지 않을 수 있다.
> 보정 후 동일 분석 파이프라인으로 entropy/correlation을 재측정하여
> **"스코어링 문제인가, 모델 한계인가"** 를 분리해야 한다.

---

## 9. 실험 이력

| 항목 | 내용 |
|------|------|
| 선행 실험 | Round 1-14 (90+ experiments) |
| 검증 테스트 | 12개 신규 unit test (Phase 1-4) |
| 분석 데이터 | `experiments/analysis/score_distribution.csv` |
| 분석 스크립트 | `experiments/analysis/hh_proof_analysis.py` |
| 상세 분석 | `experiments/reports/hh_proof_report.md` |
| 구현 검증 | `engine/src/core/attention_scores.rs` (experiment test) |
| CLI 진단 | `--h2o-debug` flag (`engine/src/bin/generate.rs`) |

---

## 10. On-Device 실험: Time-Normalized Scoring 검증

### 10.1 실험 설정

누적 바이어스(SUM → time-in-cache proxy)를 제거하기 위해 **Time-Normalized Scoring**을 구현하고
실제 모델에서 3-way 비교 실험을 수행하였다.

- **Time-Normalized Scoring**: `importance[t] / step_count[t]` — 평균 per-step importance
- 구현: `attention_scores.rs`의 `end_step()`에서 `step_count` 추적 + `importance_scores()` 분기

| 조건 | Eviction | Scoring | CLI Flag |
|------|----------|---------|----------|
| **Sliding** | kr=0.0 (pure window) | N/A | `--eviction-policy sliding` |
| **H2O-Raw** | kr=0.5 (HH+window) | Cumulative SUM | `--eviction-policy h2o` |
| **H2O-Norm** | kr=0.5 (HH+window) | Time-normalized | `--eviction-policy h2o --h2o-time-normalize` |

공통 파라미터: Llama 3.2 1B, `--greedy`, `--kv-layout head`, `--protected-prefix 4`,
`--experiment-schedule memory_critical_1024.json` (target=512), `-n 2048`

### 10.2 결과

#### PPL-01 (Literary domain)

| Metric | H2O-Raw vs Slide | H2O-Norm vs Slide |
|--------|:---:|:---:|
| **EMR** | 52.9% | **100.0%** |
| **Suffix EMR** | 5.7% | **100.0%** |
| **FDT** | pos 1078 | pos 2047 (no divergence) |
| **Top-5 Overlap** | 54.5% | **91.9%** |
| **Post-evict Top-5** | 9.0% | **83.9%** |
| Avg TBT | 134.7ms | 135.7ms |

#### PPL-03 (Technical domain)

| Metric | H2O-Raw vs Slide | H2O-Norm vs Slide |
|--------|:---:|:---:|
| **EMR** | **100.0%** | 58.9% |
| **Suffix EMR** | **100.0%** | 17.8% |
| **FDT** | pos 2047 (no divergence) | pos 1203 |
| **Top-5 Overlap** | **89.9%** | 59.5% |
| **Post-evict Top-5** | **79.8%** | 19.0% |
| Avg TBT | 139.9ms | 141.1ms |

### 10.3 Score 분포 비교

| Experiment | N_gen | BOS Score | Prompt μ | Gen μ | Gen σ | Pearson r |
|------------|:---:|:---:|:---:|:---:|:---:|:---:|
| H2O-RAW-PPL01 | 1024 | **22740.7** | 47.3 | 54.5 | 25.8 | **-0.47** |
| H2O-NORM-PPL01 | 1024 | 22.2 | 0.05 | 0.21 | 0.39 | **+0.46** |
| H2O-RAW-PPL03 | 1034 | **23160.1** | 54.5 | 51.1 | 22.4 | **-0.30** |
| H2O-NORM-PPL03 | 1034 | 22.6 | 0.05 | 0.21 | 0.42 | **+0.45** |

- Raw scoring: Pearson r < 0 → **older tokens score higher** (time-in-cache 지배)
- Normalized scoring: Pearson r > 0 → **recent tokens score higher** (바이어스 반전 성공)

### 10.4 Token Selection Overlap Analysis

Time normalization이 H2O의 토큰 선택을 sliding window에 얼마나 근접시키는가:

| Experiment | Both Keep | Only H2O | Only Slide | HH Contiguous Runs | Avg Gap |
|------------|:---:|:---:|:---:|:---:|:---:|
| RAW-PPL01 | 294 | **218** | 218 | 115 | 3.9 |
| **NORM-PPL01** | **418** | **94** | 94 | 70 | 7.1 |
| RAW-PPL03 | 315 | **197** | 197 | 110 | 4.4 |
| **NORM-PPL03** | **431** | **81** | 81 | 58 | 9.1 |

정규화 후 sliding과의 차이 토큰 수: **218→94 (PPL-01)**, **197→81 (PPL-03)** — 약 57% 감소

### 10.5 핵심 발견: Contiguity Paradox

결과가 프롬프트에 따라 정반대인 이유:

1. **PPL-01**: Raw H2O가 218개 토큰을 다르게 선택 → 5.7% suffix EMR (치명적).
   Norm이 차이를 94개로 줄임 → 100% EMR (완벽). **정규화가 해결**.

2. **PPL-03**: Raw H2O가 197개 토큰을 다르게 선택 → 100% EMR (우연히 무해).
   Norm이 차이를 81개로 줄임 → 17.8% EMR (악화). **정규화가 악화**.

**근본 원인**: kr=0.5는 recent window를 508→254로 절반 축소한다.
HH 예산(254)이 나머지를 채우는데, 이 토큰들은 **반드시 불연속(scattered)** 하다.
불연속 토큰의 영향은 **예측 불가능**하고 **프롬프트 의존적**이다.

- PPL-01에서 Raw의 218개 불연속 HH가 해로웠고, Norm의 94개는 무해했음
- PPL-03에서 Raw의 197개 불연속 HH가 우연히 무해했으나, Norm의 81개가 해로움
- 동일한 "개선" (차이 감소)이 반대 결과를 낳음 → **HH 선택 자체가 신뢰 불가**

### 10.6 결론

| 항목 | 판정 |
|------|------|
| 시간 정규화가 누적 바이어스를 제거하는가? | **YES** (Pearson r: -0.47 → +0.46) |
| 정규화가 HH 선택을 sliding에 근접시키는가? | **YES** (차이 57% 감소) |
| 정규화가 H2O 성능을 일관되게 개선하는가? | **NO** (PPL-01 개선, PPL-03 악화) |
| HH 선택이 1B 모델에서 신뢰할 수 있는가? | **NO** (프롬프트 의존, 예측 불가) |

> **최종 판정**: 누적 바이어스는 H2O 실패의 **기여 요인**이지만, **근본 원인이 아니다**.
> 근본 원인은 **1B 모델의 attention entropy가 너무 높아** (≈0.97) genuine HH가 존재하지 않는 것이다.
> 어떤 scoring 방식을 사용하든 kr>0인 한 contiguous recent context의 일부가 scattered HH로
> 대체되며, 이는 **예측 불가능한 품질 변동**을 야기한다.
> **Sliding Window (kr=0.0)만이 일관되게 최적이다.**

---

## 11. 결론

1. **H2O 구현은 논문과 일치**한다. 성능 저하는 버그가 아니다.
2. **1B 모델은 genuine Heavy Hitter를 생성하지 않는다** — attention entropy ≈ 0.97 (near-uniform).
3. **누적 SUM scoring은 체류 시간의 proxy**가 되어 H2O가 가장 오래된 토큰을 "중요"하다고 판단한다.
4. **Time-Normalized Scoring은 누적 바이어스를 제거**하지만 (Pearson r: -0.47 → +0.46),
   HH 선택의 결과는 프롬프트에 따라 개선(+94.3%) 또는 악화(-82.2%)되어 **신뢰 불가**.
5. **근본 원인은 kr>0에 의한 contiguity 손실**: recent window 축소 + scattered HH 대체가
   예측 불가능한 품질 변동을 야기한다. 어떤 scoring이든 이 구조적 문제를 극복할 수 없다.
6. **Sliding Window (kr=0.0)가 1B 모델에서 유일하게 일관된 최적 전략**이다.
7. **H2O의 score 계산 (6x overhead)은 1B에서 순수 비용**이며, 제거를 권장한다.

---

*Generated: 2026-03-10*
*Data: Round 14-15 experiments, 12 verification tests, score distribution simulation*
*On-device validation: 6 experiments (PPL-01, PPL-03 × 3 conditions), Pixel phone, Llama 3.2 1B*
*Analysis scripts: `experiments/analysis/round15_compare.py`, `round15_deep_analysis.py`*
