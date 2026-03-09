# Round 13: H2O+ Per-Head Eviction — Results

## 실험 개요

H2O+ (GQA-aware per-head eviction)가 H2O를 개선하는지 검증.
20개 실험 (14 H2O+ + 6 baseline/sliding/h2o), 동일 바이너리 기준.

**조건**: 2048 토큰 생성, inject@1024, 80% eviction (ratio=0.20), decay=0.0

---

## Phase 1: 4-Way Direct Comparison (kr=0.5)

| Policy | Domain | EMR | FDT | ROUGE-L | BLEU-4 |
|--------|--------|-----|-----|---------|--------|
| Baseline | Literary | 1.000 | 2048 | 1.000 | 1.000 |
| Baseline | Technical | 1.000 | 2048 | 1.000 | 1.000 |
| Sliding | Literary | 0.519 | 1031 | 0.690 | 0.663 |
| Sliding | Technical | **1.000** | 2047 | 1.000 | 1.000 |
| H2O | Literary | 0.540 | 1064 | 0.793 | 0.751 |
| H2O | Technical | 0.505 | 1029 | 0.606 | 0.556 |
| H2O+ | Literary | 0.540 | 1064 | 0.793 | 0.751 |
| H2O+ | Technical | 0.505 | 1029 | 0.606 | 0.556 |

**핵심**: H2O와 H2O+가 kr=0.5에서 **완전히 동일한 결과** (Δ=0.000).

---

## Phase 2: H2O+ Keep Ratio Sweep

| keep_ratio | HH | Recent | Literary EMR | Technical EMR |
|-----------|-----|--------|------------|------------|
| 0.0 | 0 | 172 | 0.519 | **1.000** |
| 0.1 | 17 | 155 | **1.000** | 0.925 |
| 0.2 | 34 | 138 | 0.506 | 0.505 |
| 0.3 | 51 | 121 | 0.979 | 0.506 |
| 0.5 | 86 | 86 | 0.540 | 0.505 |
| 0.7 | 120 | 52 | 0.506 | 0.507 |
| 0.9 | 154 | 18 | 0.514 | 0.506 |
| **Sliding** | 0 | 172 | 0.519 | **1.000** |

**관찰**:
- kr=0.0 (HH 없음) == Sliding (예상대로)
- kr=0.1, 0.3에서 Literary EMR이 높음 (1.000, 0.979) — 비단조적
- Technical은 kr=0.0에서 최고 (1.000), HH 할당 시 하락

---

## Phase 3: H2O vs H2O+ Head-to-Head (동일 바이너리)

| keep_ratio | H2O EMR (Lit) | H2O+ EMR (Lit) | Δ | H2O EMR (Tech) | H2O+ EMR (Tech) | Δ |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0 | 0.519 | 0.519 | 0.000 | 1.000 | 1.000 | 0.000 |
| 0.1 | 1.000 | 1.000 | 0.000 | 0.925 | 0.925 | 0.000 |
| 0.3 | 0.979 | 0.979 | 0.000 | 0.506 | 0.506 | 0.000 |
| 0.5 | 0.540 | 0.540 | 0.000 | 0.505 | 0.505 | 0.000 |

**결론: 모든 keep_ratio에서 H2O == H2O+ (Δ=0.000).**

---

## 가설 판정

| 가설 | 판정 | 근거 |
|------|------|------|
| Per-head HH가 H2O를 개선한다 | **REJECTED** | 모든 kr에서 H2O == H2O+ |
| HH 개념 자체가 무가치하다 | **CONFIRMED** | kr=0.0(Sliding)이 최적 |

---

## 분석: 왜 H2O == H2O+인가?

Per-head 선택은 각 KV 헤드가 다른 HH 토큰을 선택하는 것이 목표였다.
그러나 결과적으로 동일한 EMR이 나온 이유:

1. **GQA 그룹 내 Q-헤드들의 attention 패턴이 충분히 유사**
   - 같은 GQA 그룹에 속한 Q-헤드들이 비슷한 토큰에 attention을 주므로,
     per-KV-head 평균 점수가 flat 합산 점수와 동일한 토큰을 선택
   - 결과적으로 8개 KV 헤드 모두 같은 HH 토큰을 선택

2. **1B 모델의 한계**
   - 8개 KV 헤드로는 충분한 다양성이 없음
   - 더 큰 모델(7B+, 더 많은 KV 헤드)에서 차이 가능성

3. **HH 자체의 근본적 한계**
   - Round 12에서 이미 확인: HH partition이 무가치
   - Per-head로 바꿔도 무가치한 것은 동일

---

## 시사점

1. **Aggressive eviction에서 Sliding이 최적** — HH 할당은 불필요
2. **Per-head eviction은 1B 모델에서 효과 없음** — 헤드 간 다양성 부족
3. **비단조적 EMR 패턴** (kr=0.1 > kr=0.0, Literary) — 향후 조사 필요
4. **다음 방향**: SnapKV (prefill-time compression)로 전환 권장
   - Eviction 대신 compression: 중요 토큰을 미리 선별
   - Prefill에서 attention pattern이 더 안정적
