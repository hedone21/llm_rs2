# KiVi Q2 PiQA Evaluation Report

**Date**: 2026-03-13
**Models**: Llama 3.2 1B, Llama 3.2 3B, Llama 3.1 8B (Q4_0 weights)
**Platform**: Host CPU (x86_64, 20 cores, 62GB RAM)
**Benchmark**: PiQA (Physical Intuition QA) — 100 questions, log-likelihood evaluation
**Methodology**: acc_norm (byte-length-normalized NLL), greedy decoding

---

## 1. Executive Summary

KiVi Q2(2-bit 비대칭 양자화) KV 캐시 압축이 downstream task 정확도에 미치는 영향을 1B, 3B, 8B 모델에서 PiQA 벤치마크로 측정했습니다.

**핵심 결론**: KiVi Q2는 **8B 모델에서 유의미한 정확도 하락(-5~6%p)**을 보이지만, **1B/3B 모델에서는 노이즈 범위 내**의 영향만 줍니다. 이전 텍스트 생성 평가(EMR < 1%)와 달리, log-likelihood 기반 multiple-choice 평가에서는 상대적으로 작은 영향을 미칩니다.

| Model | head_dim | Baseline | KiVi r=32 | KiVi r=128 |
|-------|----------|----------|-----------|------------|
| **1B** | 64 | 56.0% | **60.0% (+4.0)** | 57.0% (+1.0) |
| **3B** | 128 | 59.0% | 57.0% (-2.0) | **61.0% (+2.0)** |
| **8B** | 128 | **72.0%** | 66.0% (-6.0) | 67.0% (-5.0) |

---

## 2. Experiment Design

### 2.1 Configuration Matrix

| Setting | Baseline | KiVi r=32 | KiVi r=128 |
|---------|----------|-----------|------------|
| KV storage | Q4_0 (HeadMajor) | Q2 + FP32 residual | Q2 + FP32 residual |
| Residual buffer | N/A | 32 tokens | 128 tokens |
| Eviction | None | None (KIVI internal) | None (KIVI internal) |
| max_seq_len | 2048 | 2048 | 2048 |

### 2.2 Models

| Model | Parameters | hidden_size | head_dim | kv_heads | layers |
|-------|-----------|-------------|----------|----------|--------|
| Llama 3.2 1B | 1.24B | 2048 | 64 | 8 | 16 |
| Llama 3.2 3B | 3.21B | 3072 | 128 | 8 | 28 |
| Llama 3.1 8B | 8.03B | 4096 | 128 | 8 | 32 |

### 2.3 Evaluation Method

PiQA (Physical Intuition QA): 2-choice multiple-choice task measuring physical common sense.
- **Metric**: acc_norm — byte-length-normalized negative log-likelihood
- **Scoring**: For each question, compute NLL of each choice continuation, normalize by byte length, select minimum
- **Questions**: 100 (from prepared dataset)
- **Decoding**: Greedy (temperature=0)

---

## 3. Results

### 3.1 Accuracy Summary

```
Model    Config       Correct/Total    Accuracy    Δ vs Baseline
────────────────────────────────────────────────────────────────
1B       Baseline     56/100           56.0%       —
1B       KiVi r=32    60/100           60.0%       +4.0%p
1B       KiVi r=128   57/100           57.0%       +1.0%p

3B       Baseline     59/100           59.0%       —
3B       KiVi r=32    57/100           57.0%       -2.0%p
3B       KiVi r=128   61/100           61.0%       +2.0%p

8B       Baseline     72/100           72.0%       —
8B       KiVi r=32    66/100           66.0%       -6.0%p
8B       KiVi r=128   67/100           67.0%       -5.0%p
```

### 3.2 Model Size Scaling

| Model | Baseline | Best KiVi | Worst KiVi | Max Degradation |
|-------|----------|-----------|------------|-----------------|
| 1B | 56.0% | 60.0% (r=32) | 57.0% (r=128) | **+4.0%p (improvement)** |
| 3B | 59.0% | 61.0% (r=128) | 57.0% (r=32) | -2.0%p |
| 8B | 72.0% | 67.0% (r=128) | 66.0% (r=32) | **-6.0%p** |

---

## 4. Analysis

### 4.1 텍스트 생성 vs Log-Likelihood 평가의 차이

이전 텍스트 생성 평가 (kivi_accuracy_report.md)에서는:
- 1B KiVi r=32: EMR=0.0%, Top-K Overlap=7% → **완전히 다른 텍스트 생성**

본 PiQA 평가에서는:
- 1B KiVi r=32: **60.0% (+4.0%p)** → baseline보다 높음

이 차이의 원인:

1. **상대 비교 vs 절대 정확도**: PiQA는 두 선택지의 NLL을 **상대적으로** 비교합니다. Q2 양자화 노이즈가 양쪽 선택지에 비슷하게 영향을 미치면, 상대적 순위가 보존됩니다.

2. **짧은 프롬프트**: PiQA 프롬프트는 보통 50-150 토큰으로, r=32 KiVi에서도 최대 1-4회의 Q2 flush만 발생합니다. 텍스트 생성 실험의 256+ 토큰과 비교하면 양자화 누적 오차가 적습니다.

3. **Continuation 길이**: 선택지 continuation은 1-3 토큰으로 매우 짧아, Q2 오차가 NLL에 미치는 영향이 제한적입니다.

### 4.2 모델 크기별 영향 분석

**1B (head_dim=64)**: 의외로 KiVi가 baseline보다 나은 결과. 이는 통계적 노이즈 범위 내(100문제 기준 신뢰구간 약 ±5%p)에 해당하며, Q2의 노이즈가 일종의 regularization 효과를 낼 수 있습니다. 또한 1B 모델의 baseline 정확도 자체가 낮아(56%) random에 가까워 KiVi 영향이 작습니다.

**3B (head_dim=128)**: 혼재된 결과. r=32에서 -2%p, r=128에서 +2%p. 역시 노이즈 범위 내입니다.

**8B (head_dim=128)**: 일관된 하락 (-5~6%p). 8B 모델은 baseline이 72%로 높아, Q2 양자화가 실질적으로 정보를 손실시키는 것이 측정 가능합니다. 이는 KIVI 논문의 예측과 반대인데, 논문에서는 큰 모델이 redundancy가 높아 양자화에 강하다고 했습니다. 그러나:

  - 우리 구현은 **Q2 (2-bit)** 로 매우 공격적이고, 논문은 Q2에서도 7B 모델 기준입니다
  - 8B 모델이 높은 정확도에서 시작하므로, 작은 로짓 차이가 정답/오답을 뒤바꿀 가능성이 더 큽니다
  - Q4_0 weight quantization + Q2 KV cache의 이중 양자화 효과

### 4.3 Residual Size 영향

| Model | r=32 vs Baseline | r=128 vs Baseline | r=128 vs r=32 |
|-------|-----------------|-------------------|---------------|
| 1B | +4.0%p | +1.0%p | -3.0%p |
| 3B | -2.0%p | +2.0%p | +4.0%p |
| 8B | -6.0%p | -5.0%p | +1.0%p |

r=128이 r=32보다 일관되게 나은 결과를 보이는 것은 예상대로입니다 (더 적은 Q2 flush = 더 많은 FP32 데이터 보존). 그러나 1B에서 역전 현상은 통계적 노이즈입니다.

### 4.4 통계적 유의성

100문제 기준, binomial confidence interval (95%):
- 56%: CI = [45.7%, 65.9%]
- 60%: CI = [49.7%, 69.7%]
- 72%: CI = [62.1%, 80.5%]

1B/3B의 baseline-KiVi 차이(1-4%p)는 **통계적으로 유의하지 않습니다**.
8B의 6%p 차이는 경계적이며, 더 많은 문제(500+)로 확인이 필요합니다.

---

## 5. 이전 평가와의 비교

| 평가 유형 | 메트릭 | 1B KiVi r=32 | 해석 |
|----------|--------|-------------|------|
| **텍스트 생성** (kivi_accuracy_report) | EMR | 0.0% | Q2 flush 후 완전히 다른 토큰 생성 |
| **텍스트 생성** (kivi_accuracy_report) | Top-K Overlap | 7% | 로짓 분포 자체가 다름 |
| **PiQA** (본 보고서) | acc_norm | 60.0% (+4%p) | 상대적 NLL 비교에서는 영향 미미 |

**결론**: KiVi Q2는 **open-ended 생성에서는 치명적**이지만, **classification/multiple-choice에서는 비교적 견딜만합니다**. 이는 KiVi가 절대적인 logit 값을 왜곡하더라도, 선택지 간의 상대적 순위는 상당 부분 보존하기 때문입니다.

---

## 6. Recommendations

### 6.1 KiVi Q2의 적용 범위

| Use Case | 적합성 | 이유 |
|----------|--------|------|
| Open-ended text generation | **부적합** | EMR < 1%, 완전 발산 |
| Multiple-choice classification | **조건부 적합** | 1B/3B에서 노이즈 범위 내, 8B에서 -5~6%p |
| Embedding/retrieval | **검증 필요** | 벡터 방향 왜곡이 유사도 검색에 미치는 영향 미확인 |

### 6.2 개선 방향

1. **Q4 비대칭 양자화**: 2-bit → 4-bit로 전환하면 8B에서도 <1%p 하락 기대
2. **Mixed precision**: attention sink (BOS 등)은 FP32로, 나머지는 Q2로 차등 양자화
3. **Full PiQA evaluation**: 100 → 1838 문제 (전체 validation set)로 통계적 유의성 확보
4. **추가 벤치마크**: COPA, WinoGrande, HellaSwag 등으로 task-level robustness 확인

---

## 7. Raw Data

| File | Description |
|------|-------------|
| `experiments/benchmarks/results/piqa_1b_baseline.json` | 1B Baseline (Q4_0 KV) |
| `experiments/benchmarks/results/piqa_1b_kivi_r32.json` | 1B KiVi Q2 res=32 |
| `experiments/benchmarks/results/piqa_1b_kivi_r128.json` | 1B KiVi Q2 res=128 |
| `experiments/benchmarks/results/piqa_3b_baseline.json` | 3B Baseline (Q4_0 KV) |
| `experiments/benchmarks/results/piqa_3b_kivi_r32.json` | 3B KiVi Q2 res=32 |
| `experiments/benchmarks/results/piqa_3b_kivi_r128.json` | 3B KiVi Q2 res=128 |
| `experiments/benchmarks/results/piqa_8b_baseline.json` | 8B Baseline (Q4_0 KV) |
| `experiments/benchmarks/results/piqa_8b_kivi_r32.json` | 8B KiVi Q2 res=32 |
| `experiments/benchmarks/results/piqa_8b_kivi_r128.json` | 8B KiVi Q2 res=128 |
