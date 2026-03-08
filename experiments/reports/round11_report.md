# Round 11: Aggressive Eviction Deep Dive Report

## 실험 개요

77개 실험 수행. 80% eviction (keep_ratio=0.20) 조건에서 Sliding과 H2O 정책의
차이를 심화 분석.

| Sub | 실험 | 주요 변수 | 실험 수 |
|-----|------|----------|--------|
| A | Long Context PPL | 2048tok, 80% eviction, 5 domains | 15 |
| B | NIAH Redesign | passkey 3/5/10/20자리 + simple facts, 16 blocks | 54 |
| C | Position Sensitivity | PPL01 vs PPL03, inject@25/50/75% | 8 |

**Round 10 대비 변경점:**
- Eviction 강도: 50% → **80%** (keep_ratio 0.50 → 0.20)
- 컨텍스트 길이: 512-1024tok → **2048tok**
- NIAH: 2 needle → **6 needle** (passkey 길이 스케일링 + simple facts)
- 신규 CLI 플래그: `--experiment-eviction-ratio` (전략 오버라이드)

---

## Sub-A: Long Context PPL (2048tok, 80% eviction)

### 결과 테이블

| 프롬프트 | Sliding EMR | H2O EMR | Sliding TopK | H2O TopK | Evicted |
|---------|------------|---------|-------------|---------|---------|
| Literary | **0.992** | 0.517 | 0.885 | 0.545 | 860 |
| Encyclopedic | 0.528 | 0.526 | 0.544 | 0.546 | 869 |
| Technical | **1.000** | 0.511 | 0.911 | 0.538 | 868 |
| Conversational | **1.000** | 0.520 | 0.883 | 0.544 | 866 |
| News | **1.000** | 0.527 | 0.872 | 0.544 | 862 |
| **평균** | **0.904** | **0.520** | **0.819** | **0.543** | ~865 |

### 핵심 발견

1. **Sliding이 H2O를 압도적으로 능가**: EMR 0.904 vs 0.520. 80% eviction에서
   H2O는 모든 도메인에서 ~0.52로 붕괴하지만, Sliding은 4/5 도메인에서
   EMR ≥ 0.992 달성.

2. **H2O의 한계 노출**: 80% eviction에서 H2O의 attention-score 기반
   토큰 선택이 무력화됨. 남은 20% 토큰의 heavy-hitter 분포가
   실제 생성 품질과 괴리.

3. **Encyclopedic 도메인의 취약성 (Sliding)**: Round 10에서 Technical이
   가장 민감했으나, 2048tok에서는 **Encyclopedic(판구조론)이 유일한
   취약 도메인** (EMR=0.528). 사실적 참조 체인이 길어질수록
   초기 사실에 대한 의존도가 높아지기 때문.

4. **도메인 감도는 컨텍스트 길이에 따라 변화**:
   - 512tok: Technical 민감 (초기 정의 참조)
   - 2048tok: Encyclopedic 민감 (장거리 사실 참조)
   - 시사점: 단일 컨텍스트 길이로 도메인 감도를 일반화할 수 없음.

5. **~865 토큰 eviction**: 프롬프트(~40tok) + 1024 생성 토큰 시점에서
   eviction 발동. 전체 ~1064 토큰 중 80%인 ~865 토큰 제거.

### Round 10 vs Round 11 비교

| 조건 | Sliding EMR | H2O EMR | 차이 |
|------|-----------|---------|------|
| R10: 512tok, 50% evict | 0.642 | 0.520 | +0.122 |
| R10: 1024tok, 50% evict | 0.913 | 0.898 | +0.015 |
| **R11: 2048tok, 80% evict** | **0.904** | **0.520** | **+0.384** |

**80% eviction이 두 정책의 차이를 극대화함.** 50% eviction에서는 H2O도
선방했으나 (1024tok에서 0.898), 80%에서는 완전히 무너짐.

---

## Sub-B: NIAH Redesign (16 blocks, 80% eviction)

### Passkey 길이 스케일링

| Needle | 길이 | Baseline | Sliding | H2O | Evicted |
|--------|------|----------|---------|-----|---------|
| Pass-3dig | 3자리 (729) | 3/3 | 3/3 | 3/3 | ~888 |
| Pass-5dig | 5자리 (58291) | 3/3 | 3/3 | 3/3 | ~891 |
| Pass-10dig | 10자리 | 3/3 | 3/3 | 3/3 | ~894 |
| Pass-20dig | 20자리 | 3/3 | 3/3 | 3/3 | ~899 |
| Number(42) | 2자리 | 3/3 | 3/3 | 3/3 | ~890 |
| **Date(1969)** | 날짜 | **0/3** | **0/3** | **0/3** | ~892 |

### 핵심 발견

1. **숫자 패턴은 20자리까지 100% 강건**: 3자리부터 20자리까지
   모든 passkey가 모든 depth(10%, 50%, 90%), 모든 정책에서
   완벽하게 검색됨. ~890 토큰 eviction에도 무영향.

2. **날짜 형식은 1B 모델에서 실패**: "July 20, 1969"는
   baseline에서도 검색 실패. Round 10의 N-FACT("Crescentport")와
   동일한 패턴 — **1B 모델은 다중 토큰 자연어 정보 검색에 한계**.

3. **Eviction이 NIAH에 미치는 영향 없음**: 모든 성공 사례에서
   retrieval_score=1.000. 이는 NIAH 프롬프트의 구조적 특성 때문:
   질문이 프롬프트 끝에 위치하여 needle과 질문 모두
   protected prefix 또는 recent window에 포함됨.

4. **NIAH는 eviction 벤치마크로 부적합 (현재 설계)**: 숫자 검색은
   너무 쉽고 (100% pass), 자연어 검색은 모델 능력 자체의 한계.
   향후 더 큰 모델(7B+) 또는 중간 난이도 태스크 필요.

---

## Sub-C: Position Sensitivity (2048tok, 80% eviction)

### EMR by Injection Position

| 도메인 | 정책 | P25% | P50% | P75% |
|--------|------|------|------|------|
| Literary | Sliding | 0.261 | **0.992** | **1.000** |
| Literary | H2O | 0.258 | 0.517 | 0.767 |
| Technical | Sliding | 0.254 | **1.000** | 0.767 |
| Technical | H2O | 0.254 | 0.511 | 0.753 |

### 상세 메트릭

| 도메인 | 정책 | 위치 | EMR | FDT | ROUGE-L | TopK | Evicted |
|--------|------|------|-----|-----|---------|------|---------|
| Literary | Sliding | P25% | 0.261 | 514 | 0.461 | 0.312 | 450 |
| Literary | Sliding | P50% | 0.992 | 1038 | 0.991 | 0.885 | 860 |
| Literary | Sliding | P75% | 1.000 | 2047 | 1.000 | 0.945 | 1269 |
| Literary | H2O | P25% | 0.258 | 514 | 0.396 | 0.306 | 450 |
| Literary | H2O | P50% | 0.517 | 1031 | 0.613 | 0.545 | 860 |
| Literary | H2O | P75% | 0.767 | 1568 | 0.844 | 0.773 | 1269 |
| Technical | Sliding | P25% | 0.254 | 513 | 0.326 | 0.285 | 458 |
| Technical | Sliding | P50% | 1.000 | 2047 | 1.000 | 0.911 | 868 |
| Technical | Sliding | P75% | 0.767 | 1569 | 0.993 | 0.783 | 1277 |
| Technical | H2O | P25% | 0.254 | 513 | 0.337 | 0.279 | 458 |
| Technical | H2O | P50% | 0.511 | 1036 | 0.527 | 0.538 | 868 |
| Technical | H2O | P75% | 0.753 | 1542 | 0.772 | 0.762 | 1277 |

### 핵심 발견

1. **Eviction 위치가 가장 강력한 품질 결정 인자**:
   - P25%: 양쪽 정책 모두 EMR ~0.25 (출력의 75%가 영향)
   - P50%: Sliding EMR 0.99-1.00, H2O EMR ~0.51
   - P75%: 양쪽 모두 양호 (EMR 0.75-1.00)
   **도메인보다 위치가 압도적으로 중요.**

2. **P25% eviction = catastrophic**: 80% eviction이 토큰 512 위치에서
   발생하면, 이후 1536 토큰의 생성이 모두 영향. FDT가
   eviction 직후 (514)에서 시작하여 전체 품질 저하.

3. **Sliding의 우위는 P50% 이후에서만 유효**: P25%에서는
   Sliding(0.254)과 H2O(0.254)가 동일하게 실패. Sliding의 강점은
   "최근 토큰이 충분히 있을 때"만 발휘됨.

4. **P75%에서의 도메인 차이**: Literary는 Sliding P75%에서 EMR=1.000
   (FDT=2047, 완전 동일), Technical은 EMR=0.767 (FDT=1569).
   기술 텍스트는 eviction 직후에도 초기 정의를 참조하여
   즉시 diverge하지만, 문학 텍스트는 자연스러운 continuation이 가능.

---

## 종합 결론

| # | 발견 | 근거 |
|---|------|------|
| 1 | **80% eviction에서 Sliding >> H2O** | EMR 0.904 vs 0.520 (Sub-A) |
| 2 | **H2O는 aggressive eviction에서 붕괴** | 모든 도메인 EMR ~0.52로 수렴 |
| 3 | **Eviction 위치가 도메인보다 중요** | P25% EMR=0.25, P75% EMR≥0.75 (Sub-C) |
| 4 | **도메인 감도는 컨텍스트 길이 의존적** | 512tok: Technical 민감 → 2048tok: Encyclopedic 민감 |
| 5 | **숫자 passkey는 20자리까지 eviction 강건** | 모든 조건 100% 검색 성공 (Sub-B) |
| 6 | **1B 모델은 자연어 정보 검색에 한계** | DATE2, FACT 모두 baseline 실패 |
| 7 | **NIAH는 현재 설계로는 변별력 부족** | 숫자=항상 성공, 자연어=항상 실패 |

### 논문 시사점

1. **Eviction 정책 비교 시 강도(ratio)를 반드시 변화**: 50% eviction에서는
   H2O도 선방하지만 (R10: EMR 0.898), 80%에서 무너짐 (R11: EMR 0.520).
   정책의 실제 강건성은 aggressive 조건에서만 드러남.

2. **Eviction 위치 ∝ 품질 영향**: 생성 후반부 eviction은 무해하지만
   초반부 eviction은 catastrophic. 시스템 설계 시 eviction 타이밍 최적화 필요.

3. **Sliding Window의 단순성이 장점**: 복잡한 attention-score 기반
   eviction(H2O)보다 단순한 "최근 N 토큰 유지"가 실제로 우수.
   이는 autoregressive 생성에서 최근 컨텍스트의 중요성을 반영.
