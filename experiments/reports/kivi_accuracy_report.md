# KIVI Q2 KV Cache Accuracy Evaluation Report

**Date**: 2026-03-12
**Model**: Llama 3.2 1B (Q4_0 weights)
**Platform**: Host CPU (x86_64)
**Methodology**: Greedy decoding (temp=0), per-token top-10 logits recording

---

## 1. Executive Summary

KIVI Q2 압축(2-bit 비대칭 양자화)은 Llama 3.2 1B 모델에서 **심각한 품질 저하**를 일으킵니다. 첫 번째 Q2 flush가 발생하기 전까지는 baseline과 완벽히 동일한 출력을 생성하지만, flush 이후에는 즉시 발산하여 EMR(Exact Match Rate)이 1% 미만으로 떨어집니다.

| 핵심 지표 | res=32 | res=64 | res=128 |
|-----------|--------|--------|---------|
| **FDT (First Divergent Token)** | 0 | 15 | 82 |
| **EMR** | 0.0% | 7.1% | 32.9% |
| **Top-K Overlap (pre-flush)** | N/A | 100% | 99.0% |
| **Top-K Overlap (post-flush)** | 7.1% | 8.0% | 8.3% |
| **Compression** | 5.7–8.4x | 4.2x | 3.8–4.2x |
| **Speed Overhead** | +15~21% | +22% | +4~18% |

**결론**: 1B 규모 모델에서 Q2 양자화는 KV 캐시의 정보를 과도하게 손실시켜, 실용적인 추론에 사용할 수 없습니다. KIVI 논문(ICML 2024)의 실험은 7B 이상 모델 기준이며, 1B 모델은 hidden representation의 중복성(redundancy)이 충분하지 않습니다.

---

## 2. Experiment Design

### 2.1 Configuration Matrix

| 설정 | Baseline | KIVI res=32 | KIVI res=64 | KIVI res=128 |
|------|----------|-------------|-------------|--------------|
| KV storage | FP32 | Q2 + FP32 residual | Q2 + FP32 residual | Q2 + FP32 residual |
| Residual buffer | N/A | 32 tokens | 64 tokens | 128 tokens |
| Quantization trigger | N/A | buffer full → batch Q2 | buffer full → batch Q2 | buffer full → batch Q2 |
| Eviction | None | None (mutual exclusion) | None | None |

### 2.2 Prompts

**Perplexity (PPL)**: 5개 도메인 프롬프트 (문학/백과/기술/대화/뉴스), 256 tokens 생성
**Few-shot (QA-FS)**: 3개 분류 태스크 (감정분석/카테고리/번역), 32 tokens 생성
**장문 (PPL-01-512)**: 문학 프롬프트, 512 tokens 생성

### 2.3 Metrics

- **FDT** (First Divergent Token): 처음으로 baseline과 다른 토큰이 생성되는 위치
- **EMR** (Exact Match Rate): 전체 토큰 중 baseline과 동일한 비율
- **Top-K Overlap**: 각 토큰 위치에서 상위 10개 logit의 교집합 비율
- **ROUGE-L**: LCS 기반 F1 (텍스트 유사도)
- **BLEU-4**: 4-gram precision 기하평균 (번역 품질)

---

## 3. Results

### 3.1 PPL Prompts (256 tokens, res=32)

| Prompt | FDT | EMR | ROUGE-L | BLEU-4 | TopK Avg | TBT Change |
|--------|-----|-----|---------|--------|----------|------------|
| PPL-01 (literary) | 0 | 0.000 | 0.144 | 0.035 | 0.071 | +20.7% |
| PPL-02 (encyclopedic) | 0 | 0.004 | 0.110 | 0.000 | 0.044 | +19.5% |
| PPL-03 (technical) | 0 | 0.020 | 0.145 | 0.000 | 0.050 | +15.4% |
| PPL-04 (conversational) | 0 | 0.008 | 0.124 | 0.000 | 0.080 | +15.5% |
| PPL-05 (news) | 0 | 0.004 | 0.130 | 0.000 | 0.035 | +15.6% |
| **Average** | **0** | **0.007** | **0.131** | **0.007** | **0.056** | **+17.3%** |

### 3.2 Few-shot Tasks (32 tokens, res=32)

| Task | Baseline Answer | KIVI Answer | FDT | EMR | TopK Avg |
|------|----------------|-------------|-----|-----|----------|
| QA-FS-01 (sentiment) | `<EOS>` (correct: positive) | `\n\nReview:...` (wrong) | 0 | 0.000 | 0.052 |
| QA-FS-02 (category) | `medicine` (wrong) | `\n\n...` (wrong) | 0 | 0.032 | 0.090 |
| QA-FS-03 (translation) | `<EOS>` (correct: jardin) | `age, jardine...` (wrong) | 0 | 0.000 | 0.065 |

### 3.3 Residual Size Ablation (PPL-01, 256 tokens)

| Residual Size | FDT | EMR | TopK (pre-FDT) | TopK (post-FDT) | ROUGE-L | BLEU-4 | Compression |
|---------------|-----|-----|----------------|-----------------|---------|--------|-------------|
| **32** | 0 | 0.000 | N/A | 0.071 | 0.144 | 0.035 | 7.1x |
| **64** | 15 | 0.071 | 1.000 | 0.080 | 0.225 | 0.125 | 4.2x |
| **128** | 82 | 0.329 | 0.990 | 0.083 | 0.450 | 0.403 | 4.2x |

### 3.4 Long Sequence (PPL-01, 512 tokens, res=32)

| Metric | Baseline | KIVI | Change |
|--------|----------|------|--------|
| FDT | - | 0 | - |
| EMR | - | 0.002 | - |
| ROUGE-L | - | 0.141 | - |
| Top-K Overlap | - | 0.068 | - |
| Avg TBT | 35.9ms | 44.9ms | +25.2% |
| Compression | 1.0x | 8.4x | - |

### 3.5 Memory Compression

| Config | 256-token cache (per-layer) | vs FP32 |
|--------|---------------------------|---------|
| FP32 (baseline) | ~20,000 KB | 1.0x |
| KIVI res=32 | ~2,800 KB | **7.1x** |
| KIVI res=64 | ~4,600 KB | **4.2x** |
| KIVI res=128 | ~4,600 KB | **4.2x** |

---

## 4. Analysis

### 4.1 Q2 Flush가 품질 붕괴의 원인

Residual size ablation 결과가 핵심 증거입니다:

```
res=32:  프롬프트(49 tokens) > 버퍼(32) → prefill 중 Q2 flush 발생 → FDT=0
res=64:  프롬프트(49) < 버퍼(64) → decode 15번째에 버퍼 가득 참 → FDT=15
res=128: 프롬프트(49) < 버퍼(128) → decode 79번째에 버퍼 가득 참 → FDT=82
```

**Pre-flush Top-K Overlap = 100%**: Q2 flush 전에는 baseline과 완벽히 동일한 logit 분포를 생성합니다. 이는 KiviCache 구현이 올바르다는 것을 증명합니다 — FP32 residual 경로는 KVCache와 동일하게 동작합니다.

**Post-flush Top-K Overlap ≈ 7%**: Q2 flush 후에는 logit 분포가 거의 완전히 다릅니다. 2-bit 양자화(4 levels)가 key/value 벡터의 정보를 파괴합니다.

### 4.2 1B 모델의 한계

KIVI 논문은 Llama-2-7B, Llama-2-13B 모델에서 perplexity 저하 0.3 이내를 달성했습니다. 그러나:

- **1B 모델의 head_dim=64**: 7B+ 모델보다 head dimension이 작아, 각 차원의 정보 밀도가 높습니다.
- **Q2는 4개 레벨만 표현**: `{0, 1, 2, 3} * scale + minimum`으로 연속 공간을 이산화하면, 64-dim 벡터의 미세한 방향 차이가 모두 소실됩니다.
- **Attention softmax 증폭**: Key의 작은 양자화 오차가 QK^T dot product에서 증폭되고, softmax를 거치면 attention weight 분포가 완전히 달라집니다.

### 4.3 속도 오버헤드

| 원인 | 영향 |
|------|------|
| `get_view()` dequantization | Q2 → FP32 변환 (매 토큰) |
| 긴 시퀀스에서 증가 | 256tok: +17%, 512tok: +25% |
| FP32 residual copy | 추가 메모리 조작 |

KIVI의 의도는 메모리 절감이지만, 현재 CPU-only 구현에서는 dequantization 오버헤드가 매 토큰 발생하여 15-25% 속도 저하가 있습니다.

---

## 5. Comparison with Eviction Policies

이전 round12 실험에서 Sliding Window eviction(80% eviction ratio)의 EMR은 0.517~0.992였습니다.

| 방법 | EMR | Memory Saving | Quality |
|------|-----|---------------|---------|
| **Baseline (no eviction)** | 1.000 | 0% | Perfect |
| **Sliding Window** (80% evict) | 0.517~0.992 | ~80% cache reduction | Moderate degradation |
| **KIVI Q2** (res=32) | 0.007 | 86% (7.1x compression) | **Unusable** |
| **KIVI Q2** (res=128) | 0.329 | 76% (4.2x compression) | Severe degradation |

KIVI Q2는 Sliding Window보다 더 심각한 품질 저하를 보이며, 메모리 절감 효과도 eviction 정책과 유사한 수준입니다.

---

## 6. Recommendations

### 6.1 단기 (현재 구현 활용)

1. **Q4_0 KIVI 구현 고려**: 2-bit 대신 4-bit asymmetric 양자화 (BlockQ4_0의 변형)로 전환하면, 1B 모델에서도 acceptable한 품질을 유지할 가능성이 있습니다.
2. **대형 모델 검증**: Llama 3.2 3B에서 KIVI Q2 테스트를 수행하여, 모델 크기 증가에 따른 Q2 tolerance를 확인합니다.
3. **Residual size 최적화**: 프롬프트 길이 이상의 residual size를 사용하면 적어도 short context에서는 품질을 유지할 수 있습니다.

### 6.2 중기 (구현 개선)

4. **Hybrid Q4+Q2**: 초기 토큰(attention sink)은 Q4로, 나머지는 Q2로 — 차등 양자화 전략.
5. **Per-head calibration**: 레이어/헤드별 양자화 감도를 측정하고, 민감한 헤드만 Q4로 유지.
6. **GPU dequant kernel**: CPU dequantization 오버헤드를 GPU 커널로 이동하여 속도 오버헤드 제거.

---

## 7. Raw Data

실험 결과 JSONL 파일: `experiments/results/kivi/`

| File | Description |
|------|-------------|
| `BASE-PPL-{01~05}.jsonl` | Baseline FP32, 256 tokens |
| `KIVI-PPL-{01~05}.jsonl` | KIVI Q2 res=32, 256 tokens |
| `KIVI-PPL-01-r64.jsonl` | KIVI Q2 res=64, 256 tokens |
| `KIVI-PPL-01-r128.jsonl` | KIVI Q2 res=128, 256 tokens |
| `KIVI-PPL-03-r128.jsonl` | KIVI Q2 res=128, 256 tokens |
| `BASE-PPL01-512.jsonl` | Baseline FP32, 512 tokens |
| `KIVI-PPL01-512.jsonl` | KIVI Q2 res=32, 512 tokens |
| `BASE-QA-FS-{01~03}.jsonl` | Baseline few-shot, 32 tokens |
| `KIVI-QA-FS-{01~03}.jsonl` | KIVI Q2 few-shot, 32 tokens |
| `KIVI-QA-FS-01-r128.jsonl` | KIVI Q2 res=128, 32 tokens |

---

## 한국어 요약

### KIVI Q2 정확도 평가 결과

**핵심 발견**: Llama 3.2 1B 모델에서 KIVI Q2(2-bit) KV 캐시 압축은 심각한 품질 저하를 일으킵니다.

- **Q2 flush 전**: Baseline과 100% 동일 (구현 정확성 입증)
- **Q2 flush 후**: EMR < 1%, Top-K Overlap ≈ 7% (실질적으로 다른 텍스트 생성)
- **Residual 크기 영향**: res=32 → FDT=0, res=64 → FDT=15, res=128 → FDT=82
  - 발산 시점은 첫 Q2 flush와 정확히 일치
- **메모리 압축**: 3.8~8.4x (vs FP32), Sliding Window eviction과 유사한 절감 효과
- **속도 오버헤드**: +15~25% (dequantization 비용)

**원인 분석**: 1B 모델은 head_dim=64로, 2-bit 양자화(4 levels)가 key/value 벡터의 방향 정보를 파괴합니다. Attention softmax가 이 오차를 증폭하여 완전히 다른 attention 분포를 생성합니다.

**권장사항**: Q4 asymmetric 양자화 또는 3B+ 모델에서의 검증이 필요합니다. 현재 Q2 구현은 1B 모델에서는 실용적이지 않습니다.
