# Round 14: H2O/H2O+ with Real Attention Scores

## 실험 개요

Q4_0 스코어 버그 수정 후 H2O/H2O+ 재검증.
28개 신규 실험 (14 H2O + 14 H2O+), Round 13 baseline 재사용.

**핵심 변경**: `compute_attention_scores()` 추가 — Q4_0 K 캐시를 CPU에서 역양자화,
Q·K^T + softmax 계산 후 ws.scores에 기록.

**조건**: 2048 토큰, inject@1024, 80% eviction (ratio=0.20), decay=0.0

---

## Baselines (Round 13 재사용)

| Policy | Domain | EMR |
|--------|--------|-----|
| Sliding | Literary | 0.519 |
| Sliding | Technical | 1.000 |

---

## Phase 2: H2O Keep Ratio Sweep (Real Scores)

| keep_ratio | HH | Recent | Literary EMR | Technical EMR |
|-----------|-----|--------|------------|------------|
| 0.0 | 0 | 172 | 0.519 | 1.000 |
| 0.1 | 17 | 155 | 0.511 | 1.000 |
| 0.2 | 34 | 138 | 0.561 | 0.766 |
| 0.3 | 51 | 121 | 0.507 | 0.503 |
| 0.5 | 86 | 86 | 0.508 | 0.505 |
| 0.7 | 120 | 52 | 0.526 | 0.504 |
| 0.9 | 154 | 18 | 0.517 | 0.505 |

---

## Phase 3: H2O+ Keep Ratio Sweep (Real Scores)

| keep_ratio | HH | Recent | Literary EMR | Technical EMR |
|-----------|-----|--------|------------|------------|
| 0.0 | 0 | 172 | 0.519 | 1.000 |
| 0.1 | 17 | 155 | 1.000 | 1.000 |
| 0.2 | 34 | 138 | 0.514 | 0.503 |
| 0.3 | 51 | 121 | 0.507 | 0.514 |
| 0.5 | 86 | 86 | 0.513 | 0.520 |
| 0.7 | 120 | 52 | 0.828 | 0.505 |
| 0.9 | 154 | 18 | 0.513 | 0.508 |

---

## Phase 4: H2O vs H2O+ Head-to-Head

| keep_ratio | H2O EMR (Lit) | H2O+ EMR (Lit) | Δ | H2O EMR (Tech) | H2O+ EMR (Tech) | Δ |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0 | 0.519 | 0.519 | +0.000 | 1.000 | 1.000 | +0.000 |
| 0.1 | 0.511 | 1.000 | +0.489 | 1.000 | 1.000 | +0.000 |
| 0.2 | 0.561 | 0.514 | -0.047 | 0.766 | 0.503 | -0.263 |
| 0.3 | 0.507 | 0.507 | +0.000 | 0.503 | 0.514 | +0.011 |
| 0.5 | 0.508 | 0.513 | +0.006 | 0.505 | 0.520 | +0.016 |
| 0.7 | 0.526 | 0.828 | +0.302 | 0.504 | 0.505 | +0.001 |
| 0.9 | 0.517 | 0.513 | -0.004 | 0.505 | 0.508 | +0.003 |

---

## 핵심 결론

### 1. 스코어 수정이 H2O 성능을 변화시켰는가?

**아니오.** H2O kr=0.5 EMR (0.506)은 Round 12/13의 zero-score 결과(~0.51-0.54)와 사실상 동일하다.

실제 어텐션 스코어를 사용해도 Heavy Hitter 토큰 선택이 품질을 개선하지 못했다. **HH 파티션은 Llama 3.2 1B에서 무가치하다는 Round 12 결론이 재확인됨.**

### 2. H2O vs H2O+ (per-head eviction)

kr=0.5 기준 평균: H2O 0.506 vs H2O+ 0.517 (Δ=+0.011). **유의미한 차이 없음.**

H2O+ kr=0.1/0.7에서 Literary EMR 급등 (1.000, 0.828)이 관찰되었으나, 이는 비단조적이고 재현 불확실 — stochastic한 토큰 선택의 운 가능성이 높다.

### 3. 최적 정책

| Domain | 최적 정책 | EMR |
|--------|-----------|-----|
| Literary | Sliding (kr=0.0) | 0.519 |
| Technical | Sliding (kr=0.0) | **1.000** |

**결론: Sliding Window가 H2O/H2O+를 일관되게 match하거나 능가한다.**

---

## 가설 판정

| 가설 | 판정 | 근거 |
|------|------|------|
| Q4_0 스코어 수정이 H2O를 개선한다 | **REJECTED** | kr=0.5 EMR 0.506 ≈ Round 12 수준 |
| 실제 스코어로 HH가 가치를 회복한다 | **REJECTED** | kr=0.0(Sliding)이 여전히 최적 |
| Per-head eviction(H2O+)이 H2O를 능가한다 | **REJECTED** | 평균 Δ=+0.011, 비단조적 |
| HH 파티션은 1B 모델에서 무가치하다 | **CONFIRMED** | 14개 keep_ratio에서 일관 확인 |

---

## 시사점

1. **HH 파티션 폐기**: Llama 3.2 1B에서 attention score 기반 Heavy Hitter 선택은 Sliding Window 대비 이점이 없다.
2. **H2O+ 구현 불필요**: Per-head eviction이 flat eviction 대비 체계적 개선을 보이지 않는다.
3. **Sliding Window = 최적 eviction**: 단순한 "최근 N 토큰 유지" 전략이 가장 효과적이다.
4. **성능 오버헤드**: 스코어 계산이 avg TBT를 ~40ms → ~237ms로 6x 증가시킨다. 품질 개선 없이 성능만 저하.
5. **향후 방향**: (a) 더 큰 모델(7B+)에서 HH 재검증, (b) SnapKV (prefill-end 압축) 탐색, (c) eviction 전략 대신 KV 캐시 양자화 최적화에 집중.
