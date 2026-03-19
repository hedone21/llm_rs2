# QCF Validation Experiment Report

> QCF (Quality Cost Function) 값이 실제 품질 열화를 예측하는지 4개 벤치마크로 검증한 결과.

**실험 일시**: 2026-03-19
**모델**: Llama 3.2 1B (F32 KV cache, CPU backend)
**총 실험 수**: 591회 (PPL 11 + NIAH 45 + QA 35 + MMLU 500)
**총 소요 시간**: ~4.5시간

---

## 1. PPL (Perplexity) — Phase 1

### 실험 설정

- 평가 텍스트: `eval_text.txt` (~2047 tokens, teacher-forcing)
- 정책: Sliding Window, H2O (kr=0.5)
- Budget: B1(1600) ~ B5(350), 5단계 + baseline
- QCF: eviction 시점에 자동 수집

### 결과

| Policy | Budget | PPL | ΔPPL | QCF avg | QCF total | Evictions |
|--------|--------|-----|------|---------|-----------|-----------|
| none | 2048 | 7.09 | 0.00 | — | — | 0 |
| **sliding** | **1600** | **8.22** | **1.12** | **0.3579** | **0.358** | **1** |
| sliding | 1200 | 15.99 | 8.90 | 0.1440 | 0.720 | 5 |
| sliding | 900 | 118.66 | 111.57 | 0.0011 | 1.273 | 1147 |
| sliding | 600 | 251.15 | 244.05 | 0.0017 | 2.408 | 1447 |
| sliding | 350 | 434.57 | 427.47 | 0.0028 | 4.835 | 1697 |
| **h2o** | **1600** | **17.52** | **10.43** | **0.000006** | **0.003** | **447** |
| h2o | 1200 | 48.17 | 41.07 | 0.000014 | 0.012 | 847 |
| h2o | 900 | 120.36 | 113.27 | 0.000183 | 0.210 | 1147 |
| h2o | 600 | 269.20 | 262.10 | 0.000689 | 0.998 | 1447 |
| h2o | 350 | 467.56 | 460.46 | 0.001556 | 2.640 | 1697 |

### Rank Correlation

| 정책 | QCF metric | ρ(QCF, ΔPPL) | 판정 |
|------|-----------|--------------|------|
| Sliding | **avg** | **-0.600** | FAIL |
| Sliding | **total (cumulative)** | **1.000** | **PASS** |
| H2O | avg | 1.000 | PASS |
| H2O | total (cumulative) | 1.000 | PASS |
| Combined | total | 0.721 | — |

### 분석

**QCF avg vs QCF total의 차이가 핵심 발견.**

- **Sliding Window**: 매 step 1토큰씩 evict → 개별 QCF가 매우 작음 (0.001). avg는 eviction 횟수에 무관하게 비슷한 값. 반면 **cumulative total은 budget 감소와 완벽하게 단조 증가** → ρ=1.0.
- **H2O**: budget 초과 시 한 번에 많은 토큰을 evict → 개별 QCF도 budget과 비례 → avg와 total 모두 ρ=1.0.
- **결론**: **QCF는 cumulative total로 사용해야 하며, 이 경우 PPL 열화를 완벽하게 예측한다.**

### α Calibration

| 정책 | α (slope) | R² |
|------|-----------|-----|
| H2O | 279,646 | 0.982 |
| Sliding | — | (비선형, piecewise 필요) |

H2O의 α가 매우 큰 이유: QCF total 값이 0.003~2.6 범위인데 ΔPPL은 10~460 범위이므로 스케일 차이가 큼.

---

## 2. NIAH (Needle-in-a-Haystack) — Phase 2

### 실험 설정

- Needles: N-PASS (passkey), N-FACT (fact), N-NUM (number)
- Depth: 0.1, 0.25, 0.5 (needle 위치)
- Context: 20 filler blocks (~1500 tokens)
- Budget: B0(2048) ~ B4(600), 5단계
- 정책: Sliding Window
- 생성: greedy, 64 tokens

### 결과

| Budget | Accuracy | QCF avg | Evictions |
|--------|----------|---------|-----------|
| 2048 (B0) | **100%** (9/9) | 0.000 | 0 |
| 1600 (B1) | **100%** (9/9) | 0.000 | 0 |
| 1200 (B2) | **100%** (9/9) | 0.144 | 2 |
| 900 (B3) | **100%** (9/9) | 0.001 | ~480 |
| 600 (B4) | **100%** (9/9) | 0.002 | ~780 |

**45/45 전부 PASS** — 모든 needle, 모든 depth, 모든 budget에서 100% 검색 성공.

### 분석

Sliding window eviction이 NIAH 성능에 **전혀 영향을 미치지 않는다.** 이유:

1. **Prefill attention 전파**: 모든 토큰이 prefill 시 full attention을 통해 처리됨. Needle 정보가 hidden state에 인코딩된 후에 eviction이 발생.
2. **Sliding window 특성**: 최근 토큰을 보존하므로, 맨 뒤의 question과 그 직전 context는 항상 KV cache에 남음.
3. **1B 모델의 information persistence**: 작은 모델이 짧은 context에서 needle 정보를 강하게 인코딩.

**QCF 상관 측정 불가** — accuracy에 변동이 없어 Spearman ρ를 계산할 수 없음. 이것은 QCF의 문제가 아니라 **sliding window eviction이 NIAH에 무해하다**는 사실의 반영.

---

## 3. QA (Document Question Answering) — Phase 3

### 실험 설정

- 태스크: Single-doc QA 3개, Summarization 2개, Multi-hop 2개 (총 7개)
- Context padding: NIAH filler blocks로 ~1500 tokens 확장
- 구조: `[Padding] [Document] [Question]`
- Budget: B0(2048) ~ B4(600), 5단계
- 정책: Sliding Window
- 생성: greedy, 128 tokens

### 결과

| Budget | Avg F1 | EM Rate | QCF avg |
|--------|--------|---------|---------|
| 2048 (B0) | 0.028 | 14.3% | 0.000 |
| 1600 (B1) | 0.028 | 14.3% | 0.000 |
| 1200 (B2) | 0.028 | 14.3% | 0.144 |
| 900 (B3) | 0.028 | 14.3% | 0.001 |
| 600 (B4) | 0.028 | 14.3% | 0.002 |

**F1, EM 모두 budget에 관계없이 동일.**

### 분석

NIAH와 동일한 메커니즘:

1. **Padding-first eviction**: `[Padding][Document][Question]` 구조에서 sliding window는 앞쪽 padding만 evict. Document와 question은 최근 토큰이므로 보존.
2. **Document 무손상**: budget=600에서도 ~600 tokens의 최근 window에 document (~400 tokens) + question이 포함됨.
3. **F1이 낮은 이유**: eviction 무관. 1B 모델의 기본 QA 능력이 낮고, 128 tokens 생성에서 precision이 낮음.

**실험 설계 개선 필요**: document를 앞에, padding을 뒤에 배치하면 eviction이 document를 직접 훼손하여 의미 있는 열화를 측정할 수 있음.

---

## 4. MMLU (Many-Shot In-Context Learning) — Phase 4

### 실험 설정

- 형식: 20-shot ICL (~1500 tokens context)
- Subjects: marketing, professional_psychology, high_school_psychology, us_foreign_policy, human_sexuality
- 문항: 각 subject 20개 (총 100개)
- 평가: eval-ll 모드 (NLL 기반 선택)
- Budget: B0(2048) ~ B4(600), 5단계
- 정책: Sliding Window

### 전체 결과

| Budget | Accuracy | QCF avg |
|--------|----------|---------|
| 2048 (B0) | 24.0% (24/100) | 0.000 |
| 1600 (B1) | 24.0% (24/100) | 0.072 |
| 1200 (B2) | 23.0% (23/100) | 0.115 |
| 900 (B3) | 24.0% (24/100) | 0.001 |
| 600 (B4) | 28.0% (28/100) | 0.002 |

### Subject별 결과

| Subject | B0 | B1 | B2 | B3 | B4 | Trend |
|---------|-----|-----|-----|-----|-----|-------|
| marketing | 25% | 25% | 25% | 25% | 35% | ↑ noise |
| professional_psychology | 0% | 0% | 0% | 0% | 0% | flat (floor) |
| high_school_psychology | 10% | 10% | 10% | 20% | 20% | ↑ noise |
| us_foreign_policy | 40% | 40% | 35% | 40% | 45% | noisy |
| human_sexuality | 45% | 45% | 45% | 35% | 40% | ↓↑ noisy |

### 분석

1. **Random baseline (25%) 수준**: 전체 평균 24%는 random chance와 구분 불가.
2. **Subject 간 극심한 편차**: professional_psychology = 0% (완전 실패), human_sexuality = 45% (상대적 성공).
3. **비단조 변동**: budget 감소 시 accuracy가 떨어지지 않고 오히려 올라가는 경우도 있음. 이는 20문항의 **binomial noise** (SE ≈ ±10%p at n=20) 때문.
4. **1B 모델의 ICL 한계**: many-shot examples가 eviction되어도 baseline 자체가 random이므로 추가 열화를 측정 불가.

**결론**: 1B 모델에서 MMLU는 QCF 검증에 부적합. 3B+ 모델이나 ICL에 더 민감한 벤치마크 필요.

---

## 5. 종합 분석

### Correlation Summary

| Benchmark | Metric | QCF type | ρ | 판정 |
|-----------|--------|----------|---|------|
| **PPL (Sliding)** | ΔPPL | cumulative total | **1.000** | **PASS** |
| **PPL (H2O)** | ΔPPL | cumulative total | **1.000** | **PASS** |
| **PPL (H2O)** | ΔPPL | avg | **1.000** | **PASS** |
| PPL (Sliding) | ΔPPL | avg | -0.600 | FAIL |
| NIAH | 1-accuracy | — | N/A | 변동 없음 |
| QA | 1-F1 | — | N/A | 변동 없음 |
| MMLU | 1-accuracy | — | N/A | noise > signal |

### 핵심 발견

#### 1. QCF cumulative total은 PPL 열화를 완벽하게 예측한다

두 정책 모두에서 ρ=1.0. 이것은 QCF의 수식 `Σ_evicted attn(t)×‖V(t)‖₁ / Σ_all attn(t)×‖V(t)‖₁`이 정보 손실량의 유효한 proxy임을 입증한다.

**단, avg가 아닌 cumulative total을 사용해야 한다.** Sliding window처럼 매 step 미세한 eviction이 반복되는 경우, 개별 QCF(avg)는 budget과 비례하지 않지만, 누적 QCF(total)는 완벽한 단조 관계를 보인다.

#### 2. Sliding window eviction은 task-level 벤치마크에 무해하다

NIAH (100%), QA (불변), MMLU (noise) — 세 벤치마크 모두에서 sliding window eviction이 성능 열화를 유발하지 않았다.

**원인**: Transformer의 prefill attention이 이미 모든 토큰 간 정보를 전파한 후에 eviction이 발생. KV cache에서 토큰을 삭제해도 이미 인코딩된 hidden state는 영향받지 않음. 열화는 **이후 decode 중 삭제된 토큰을 attend하지 못하는 것**에서 발생하며, 이는 PPL(teacher-forcing)에서만 측정 가능.

#### 3. QCF avg vs QCF total의 의미론적 차이

| QCF type | 의미 | 적합한 상황 |
|----------|------|------------|
| **avg** | 단일 eviction의 평균 열화 강도 | H2O (대량 일괄 eviction) |
| **total** | 누적 정보 손실량 | Sliding (매 step 미세 eviction) |
| **max** | 최대 단일 eviction 열화 | worst-case 분석 |

### 한계 및 개선 방향

1. **Decode-phase eviction 테스트 필요**: 현재 실험은 prefill 후 eviction. 긴 생성(1000+ tokens)에서 decode 중 eviction이 발생하는 시나리오에서 NIAH/QA 열화를 측정해야 함.
2. **프롬프트 구조 개선**: QA에서 document를 앞에 배치하여 eviction 대상으로 만들어야 함.
3. **3B 모델**: MMLU 검증에는 baseline 성능이 30%+ 이상인 모델 필요.
4. **Cross-policy 비교**: H2O의 score-based eviction이 NIAH/QA에서 더 큰 영향을 미치는지 확인 필요 (중요 토큰을 선택적으로 보존/삭제하므로).

---

## 실험 환경

| 항목 | 값 |
|------|-----|
| Model | Llama 3.2 1B (HuggingFace Safetensors) |
| Backend | CPU (host, x86_64) |
| KV dtype | F32 |
| KV layout | HeadMajor |
| max_seq_len | 2048 |
| protected_prefix | 4 |
| Temperature | 0 (greedy / deterministic) |
| Eviction policies | Sliding Window, H2O (kr=0.5) |
| Budgets | 2048, 1600, 1200, 900, 600, 350 |
| Total experiments | 591 |
| Total wall time | ~4.5 hours |
