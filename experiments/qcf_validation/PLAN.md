# QCF Validation Experiment Plan

> QCF(Quality Cost Function) 값이 실제 품질 열화를 예측하는지 검증한다.
> 핵심 가설: **"QCF가 높을수록, PPL·벤치마크 점수가 나빠진다"** (단조 상관)

---

## 시나리오: 중간 열화 (Mid-Inference Degradation)

실제 on-device 시나리오를 반영한다:

```
[정상 구간]              [열화 시점]        [열화된 상태로 계속]
Token 0 ─── ... ─── Token K ──eviction──→ Token K+1 ─── ... ─── Token N
                              ↑
                         KV budget 초과
                         QCF 측정 시점
```

- 프롬프트의 앞부분은 정상 처리됨
- KV budget에 도달하면 eviction 발생 → 일부 토큰의 KV가 삭제됨
- 이후 추론은 열화된 KV cache로 수행
- **핵심**: 중요 정보(needle, 문서, few-shot examples)가 열화 구간 이전에 위치

---

## 공통 설정

| 항목 | 값 |
|------|-----|
| 모델 | Llama 3.2 1B (`models/llama3.2-1b`) |
| max_seq_len | 2048 |
| Backend | CPU (host) |
| KV type | F32 (V-norm 접근 필요) |
| KV layout | HeadMajor |
| Eviction policy | Sliding Window (기본), H2O (비교) |
| protected_prefix | 4 |
| Temperature | 0 (greedy) |

### Budget 수준 (5단계)

| Level | KV Budget | 열화 정도 | 의미 |
|-------|-----------|----------|------|
| B0 | 2048 (없음) | Baseline | 열화 없음 |
| B1 | 1600 | Mild | ~20% eviction |
| B2 | 1200 | Moderate | ~40% eviction |
| B3 | 900 | Aggressive | ~55% eviction |
| B4 | 600 | Severe | ~70% eviction |
| B5 | 350 | Extreme | ~80% eviction |

---

## Benchmark 1: Perplexity (PPL)

### 목적

QCF와 PPL 증가량의 rank correlation 검증. α calibration 데이터 수집.

### 프롬프트 구성

```
[eval_text.txt 전체] ← ~2047 tokens, teacher-forcing
```

- budget < 2047이면 처리 중간에 eviction 발생
- 이후 토큰의 NLL이 증가 → PPL 상승

### 실험 매트릭스

| 정책 | Budget | 측정 | 실험 수 |
|------|--------|------|---------|
| None | 2048 | PPL (baseline) | 1 |
| Sliding | B1~B5 | PPL, cumulative QCF | 5 |
| H2O (kr=0.5) | B1~B5 | PPL, cumulative QCF | 5 |

**총 11회** | 소요: ~5분

### 분석

```python
# 각 실험의 (avg_QCF, ΔPPL) 쌍 수집
# Spearman ρ 계산
# scatter plot: x=cumulative QCF, y=ΔPPL
# α = slope of linear regression (QCF → ΔPPL)
```

### 성공 기준

- ρ(QCF, ΔPPL) ≥ 0.85

---

## Benchmark 2: NIAH (Needle-in-a-Haystack)

### 목적

KV cache eviction이 정보 검색 능력에 미치는 영향을 QCF가 예측하는지 검증.

### 프롬프트 구성

```
[Filler 앞부분] [Needle] [Filler 뒷부분] [Question]
 ←── depth ───→                          ←── 생성 ──→
```

- `assemble_niah.py`로 조합
- Filler blocks 6~8개 사용하여 ~1500-1800 tokens 구성
- Needle이 앞쪽(depth=0.1~0.25)에 배치되면 eviction 대상이 될 가능성 높음
- Question은 항상 맨 뒤 → eviction 이후 시점에서 needle 검색 시도

### 중간 열화 시나리오

```
예: budget=900, 프롬프트=1600 tokens

Token 0 ─── [Filler] ─── Token 400 [Needle] ─── [Filler] ─── Token 900 ──eviction──→
                                                                    ↑
                                                              budget 초과
                                                          앞쪽 filler + needle eviction 가능

─── [나머지 Filler] ─── Token 1600 [Question] ──→ 생성 시작
                                                  → needle이 eviction되었으면 검색 실패
```

### 실험 매트릭스

| Needle | Depth | Blocks | Budget | 실험 수 |
|--------|-------|--------|--------|---------|
| N-PASS (passkey) | 0.1, 0.25, 0.5 | 7 | B0~B4 | 15 |
| N-FACT (fact) | 0.1, 0.25, 0.5 | 7 | B0~B4 | 15 |
| N-NUM (number) | 0.1, 0.25, 0.5 | 7 | B0~B4 | 15 |

**총 45회** (baseline 포함) | 소요: ~25분

### 측정

- **Retrieval Accuracy**: 생성 텍스트에 `expected_answer`가 포함되면 1, 아니면 0
- **QCF**: eviction 시점에 수집된 cumulative QCF
- 각 budget 수준에서 accuracy = (성공 수) / (needle 수 × depth 수)

### 분석

```python
# budget별로 (avg_QCF, accuracy) 집계
# ρ(QCF, 1 - accuracy) 계산
# heatmap: x=budget, y=depth, color=accuracy
# QCF threshold 발견: accuracy가 급락하는 QCF 값
```

### 성공 기준

- ρ(QCF, 1 - accuracy) ≥ 0.70
- depth=0.1 (앞쪽 needle)에서 budget 감소 시 가장 먼저 실패

---

## Benchmark 3: QA (Document QA + Summarization + Multi-hop)

### 목적

문서 이해 태스크에서 QCF가 F1/EM 열화를 예측하는지 검증.

### 프롬프트 구성

문서가 ~200-400 tokens으로 짧으므로, **context padding**으로 총 길이를 ~1500 tokens로 확장한다:

```
[Padding context — 관련 없는 배경 텍스트]
[Document — 정답이 포함된 핵심 문서]
[Question]
```

- Padding = NIAH의 filler_blocks 재활용 (4~5개, ~800 tokens)
- Document는 padding 뒤에 배치 → 앞쪽 padding이 먼저 eviction 대상
- 심한 eviction에서는 document 앞부분도 eviction 대상

### 중간 열화 시나리오

```
예: budget=900, 총 1500 tokens

Token 0 ─── [Padding: 800 tokens] ─── Token 800 [Document 시작] ─── Token 900 ──eviction──→
                                                                          ↑
                                                                    budget 초과
                                                                padding 전체 + document 일부 evicted

─── [Document 나머지] ─── Token 1200 [Question] ──→ 생성 시작
                                                    → document 앞부분 정보 손실 → F1 하락
```

### 실험 매트릭스

| 태스크 | 문항 | Budget | 반복 | 실험 수 |
|--------|------|--------|------|---------|
| Single-doc QA (3문항) | QA-SD-01~03 | B0~B4 | 1 | 15 |
| Summarization (2문항) | QA-SUM-01~02 | B0~B4 | 1 | 10 |
| Multi-hop (2문항) | QA-MH-01~02 | B0~B4 | 1 | 10 |

**총 35회** | 소요: ~20분

### 측정

- **F1 Score**: 생성 텍스트와 expected_answer의 token-level F1
- **Exact Match (EM)**: 정답 문자열이 생성에 포함되는지
- **ROUGE-L**: longest common subsequence
- **QCF**: eviction 시점의 cumulative QCF

### 분석

```python
# budget별로 (avg_QCF, avg_F1) 집계
# ρ(QCF, 1 - F1) 계산
# 태스크 유형별 분리 분석 (single-doc vs multi-hop)
```

### 성공 기준

- ρ(QCF, 1 - F1) ≥ 0.60
- Multi-hop이 Single-doc보다 eviction에 민감할 것 (정보가 분산)

---

## Benchmark 4: MMLU (Many-Shot ICL)

### 목적

Few-shot in-context learning이 KV cache eviction으로 열화되는 양상을 QCF가 예측하는지 검증.

### 핵심 아이디어

```
기존 MMLU 5-shot: ~500 tokens → eviction 미발생 → 테스트 불가

해결: Many-shot (15~20-shot)으로 context를 ~1500 tokens로 확장
      → budget에 의해 앞쪽 examples가 eviction
      → 사실상 shot 수가 줄어드는 효과
      → ICL 성능 열화 측정
```

### 프롬프트 구성

```
[System instruction]
[Example 1] ← eviction 대상 (가장 먼저)
[Example 2] ← eviction 대상
...
[Example 15] ← eviction 대상
[Example 16~20] ← 최근 window에 남음
[Test question] ← 항상 남음
```

- 각 example: ~70-80 tokens (question + 4 choices + answer)
- 20-shot ≈ 1400-1600 tokens + test question
- Budget 감소 → 앞쪽 examples eviction → 실질 shot 수 감소

### 중간 열화 시나리오

```
예: 20-shot, budget=900

Token 0 ─── [Examples 1-12: ~900 tokens] ─── Token 900 ──eviction──→
                                                   ↑
                                             budget 초과
                                         Examples 1-8 정도 evicted

─── [Examples 13-20 + Test Question] ──→ 답변 생성
                                         → 실질 ~8-shot으로 축소
                                         → accuracy 하락
```

### Subject 선택 기준

1B 모델에서 baseline accuracy가 측정 가능한 (>30%) 과목 선택:

| Subject | 예상 20-shot Accuracy | 문항 수 | 선택 |
|---------|----------------------|---------|------|
| marketing | ~40% | 234 | ✅ |
| professional_psychology | ~35% | 612 | ✅ |
| high_school_psychology | ~40% | 545 | ✅ |
| us_foreign_policy | ~35% | 100 | ✅ |
| human_sexuality | ~35% | 131 | ✅ |

→ 총 **100문항 샘플** (각 subject에서 20문항)

### 실험 매트릭스

| Shot 수 | Budget | Eviction 효과 | QCF | 실험 수 |
|---------|--------|--------------|-----|---------|
| 20-shot | 2048 (B0) | 없음 | 0 | 100 |
| 20-shot | 1600 (B1) | ~3 examples evicted | low | 100 |
| 20-shot | 1200 (B2) | ~8 examples evicted | mid | 100 |
| 20-shot | 900 (B3) | ~12 examples evicted | high | 100 |
| 20-shot | 600 (B4) | ~16 examples evicted | very high | 100 |

**총 500회** | 소요: ~1.5시간 (프롬프트당 ~10초, 답 1토큰)

### 측정

- **Accuracy**: 첫 생성 토큰이 정답 선택지 (A/B/C/D)와 일치하는 비율
- **QCF**: 각 실험의 eviction 시점 cumulative QCF

### 분석

```python
# budget별로 (avg_QCF, accuracy) 집계
# binomial CI 계산 (100 samples → SE ≈ ±5%)
# ρ(QCF, accuracy) 계산
# bar chart: x=budget, y=accuracy (with error bars)
```

### 성공 기준

- ρ(QCF, 1 - accuracy) ≥ 0.60
- B0 → B4에서 accuracy 단조 감소 (최소 5%p 차이)

### 데이터 준비

```bash
# MMLU 데이터 다운로드 (HuggingFace)
# https://huggingface.co/datasets/cais/mmlu
# 또는 직접: https://people.eecs.berkeley.edu/~hendrycks/data.tar
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('cais/mmlu', 'marketing', split='test')
print(len(ds))
"
```

---

## 통합 분석

### Cross-Benchmark Correlation Matrix

모든 벤치마크의 (QCF, metric) 쌍을 통합하여:

```
         PPL    NIAH   QA-F1   MMLU
QCF     ρ=?    ρ=?    ρ=?     ρ=?
```

### 핵심 출력물

1. **Scatter plots** (4개): 각 벤치마크별 QCF vs quality metric
2. **Correlation table**: Spearman ρ + p-value (4 benchmarks × 2 policies)
3. **α calibration table**: 각 benchmark의 linear regression 결과
4. **Threshold analysis**: QCF 값별 "안전 구간" / "위험 구간" 분류

### 논문 Contribution

| 주장 | 검증 벤치마크 | 요구 ρ |
|------|-------------|--------|
| QCF가 언어 모델링 열화를 예측 | PPL | ≥ 0.85 |
| QCF가 정보 검색 실패를 예측 | NIAH | ≥ 0.70 |
| QCF가 문서 이해 열화를 예측 | QA | ≥ 0.60 |
| QCF가 ICL 성능 열화를 예측 | MMLU | ≥ 0.60 |

**"QCF는 추가 forward pass 없이, 4종의 상이한 품질 지표에서 열화를 유의하게 예측한다"**

---

## 실행 계획

```
Phase 0: 데이터 준비 (MMLU 다운로드, 프롬프트 구성)     ~30분
Phase 1: PPL sweep                                      ~5분
Phase 2: NIAH sweep                                     ~25분
Phase 3: QA sweep                                       ~20분
Phase 4: MMLU sweep                                     ~90분
Phase 5: 통합 분석 + 시각화                              ~30분
                                                  총 ~3.5시간
```

### 필요한 구현 (실험 전)

1. **MMLU runner**: 20-shot 프롬프트 구성 + 첫 토큰 정답 판정 스크립트
2. **QA runner**: padding + document + question 구성 + F1 평가 스크립트
3. **NIAH runner**: `assemble_niah.py` 확장 (budget별 batch 실행)
4. **QCF 수집**: PPL 모드 외 생성 모드에서도 QCF 출력 (generate.rs 수정 필요 여부 확인)
5. **통합 분석**: `analyze.py` (ρ 계산, scatter plot, α fitting)

---

## 디렉토리 구조

```
experiments/qcf_validation/
├── PLAN.md                     ← 이 문서
├── scripts/
│   ├── run_ppl.sh              ← Phase 1
│   ├── run_niah.sh             ← Phase 2
│   ├── run_qa.sh               ← Phase 3
│   ├── run_mmlu.py             ← Phase 4 (프롬프트 구성 + 실행 + 판정)
│   └── analyze.py              ← Phase 5 (통합 분석)
├── results/
│   ├── ppl/                    ← PPL JSON 결과
│   ├── niah/                   ← NIAH 결과
│   ├── qa/                     ← QA 결과
│   └── mmlu/                   ← MMLU 결과
├── plots/                      ← 시각화 출력
└── REPORT.md                   ← 최종 결과
```
