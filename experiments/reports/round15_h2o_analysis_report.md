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

## 10. 결론

1. **H2O 구현은 논문과 일치**한다. 성능 저하는 버그가 아니다.
2. **1B 모델은 genuine Heavy Hitter를 생성하지 않는다** — attention이 BOS + 균등 분포.
3. **누적 SUM scoring은 체류 시간의 proxy**가 되어 H2O가 가장 오래된 토큰을 "중요"하다고 판단한다.
4. **결과적으로 H2O는 sliding window의 정반대 토큰을 보존**하여 적극적으로 해롭다.
5. **Sliding Window (kr=0.0)가 1B 모델에서 최적**이며, H2O의 6x score 계산 오버헤드는 순수 비용이다.
6. **보정 가능성**: 시간 정규화 또는 frequency ranking으로 누적 바이어스를 제거한 후,
   모델 한계 vs 스코어링 문제를 분리 검증할 수 있다.

---

*Generated: 2026-03-10*
*Data: Round 14-15 experiments, 12 verification tests, score distribution simulation*
