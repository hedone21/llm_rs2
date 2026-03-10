# H2O Eviction 정확도 벤치마크 리포트

**날짜**: 2026-03-10
**목적**: Base(무제거) vs Sliding Window vs H2O 세 정책의 정확도 비교
**모델**: Llama 3.2 1B, Llama 3.2 3B (Q4_0 양자화)
**환경**: 호스트 PC (x86_64 AVX2, CPU 백엔드)

---

## 1. 실험 설계

| 파라미터 | 값 |
|----------|-----|
| 디코드 토큰 | 256 |
| Max seq len | 512 |
| Eviction 트리거 | 디코드 토큰 128 (memory_critical) |
| Eviction ratio | 0.50 (캐시의 50% 유지) |
| H2O keep_ratio | 0.5 |
| H2O decay | 0.0 |
| H2O scoring | 시간 정규화 (기본값, `score / steps_since_entry`) |
| 샘플링 | greedy (temperature=0) |
| 프롬프트 | PPL-01 (Literary), PPL-03 (Technical) |

**Eviction 흐름**: 프롬프트 (~50 토큰) prefill → 128 토큰 디코드 → memory_critical 신호 → 캐시 50% 제거 → 128 토큰 추가 디코드

---

## 2. 결과 요약

### 2.1 정확도 메트릭 비교

| Model | Prompt | Policy | EMR | FDT | ROUGE-L | BLEU-4 | Top-K Overlap |
|-------|--------|--------|----:|----:|--------:|-------:|--------------:|
| **1B** | PPL01 | Base | 1.000 | — | 1.000 | 1.000 | 1.000 |
| | | Sliding | 0.514 | 129 | 0.602 | 0.544 | 0.555 |
| | | **H2O** | **0.529** | **134** | **0.653** | **0.607** | **0.573** |
| **1B** | PPL03 | Base | 1.000 | — | 1.000 | 1.000 | 1.000 |
| | | Sliding | 0.510 | 129 | 0.550 | 0.466 | 0.525 |
| | | **H2O** | 0.510 | 129 | 0.551 | **0.499** | 0.524 |
| **3B** | PPL01 | Base | 1.000 | — | 1.000 | 1.000 | 1.000 |
| | | Sliding | 0.510 | 129 | 0.628 | 0.561 | **0.621** |
| | | **H2O** | 0.510 | 129 | **0.638** | **0.569** | 0.610 |
| **3B** | PPL03 | Base | 1.000 | — | 1.000 | 1.000 | 1.000 |
| | | Sliding | 0.514 | 129 | **0.620** | 0.496 | **0.614** |
| | | **H2O** | **0.522** | 129 | 0.585 | **0.508** | 0.604 |

### 2.2 속도 메트릭 비교

| Model | Prompt | Policy | Avg TBT (ms) | Throughput (t/s) | Overhead |
|-------|--------|--------|-------------:|-----------------:|---------:|
| **1B** | PPL01 | Base | 29.8 | 33.6 | — |
| | | Sliding | 30.2 | 33.1 | +1.5% |
| | | H2O | 35.2 | 28.4 | +18.4% |
| **1B** | PPL03 | Base | 30.8 | 32.4 | — |
| | | Sliding | 31.1 | 32.1 | +1.0% |
| | | H2O | 38.6 | 25.9 | +25.3% |
| **3B** | PPL01 | Base | 87.3 | 11.5 | — |
| | | Sliding | 86.3 | 11.6 | -1.2% |
| | | H2O | 99.4 | 10.1 | +13.9% |
| **3B** | PPL03 | Base | 90.0 | 11.1 | — |
| | | Sliding | 88.8 | 11.3 | -1.3% |
| | | H2O | 104.3 | 9.6 | +15.9% |

### 2.3 Eviction 상세

| Model | Prompt | Policy | Evicted Tokens | Final Cache | Cache Util |
|-------|--------|--------|---------------:|------------:|-----------:|
| 1B | PPL01 | Sliding | 89 | 215 | 42.0% |
| 1B | PPL01 | H2O | 89 | 215 | 42.0% |
| 1B | PPL03 | Sliding | 94 | 220 | 43.0% |
| 1B | PPL03 | H2O | 94 | 220 | 43.0% |
| 3B | PPL01 | Sliding | 89 | 215 | 42.0% |
| 3B | PPL01 | H2O | 89 | 215 | 42.0% |
| 3B | PPL03 | Sliding | 94 | 220 | 43.0% |
| 3B | PPL03 | H2O | 94 | 220 | 43.0% |

---

## 3. 분석

### 3.1 H2O vs Sliding Window 정확도

**1B 모델:**
- **PPL01 (Literary)**: H2O가 전 메트릭에서 Sliding을 상회. EMR +0.015, FDT +5, ROUGE-L +0.051, BLEU-4 +0.063. H2O가 eviction 직후에도 5토큰 더 오래 baseline을 유지한 점이 주목할 만하다 (FDT 134 vs 129).
- **PPL03 (Technical)**: 두 정책이 거의 동등. EMR 동일(0.510), ROUGE-L 차이 0.001, Top-K 차이 0.001. BLEU-4에서만 H2O가 +0.033 우위.
- **평균**: H2O가 Sliding 대비 약간 우위 또는 동등. 시간 정규화 도입 전 Round 15에서 H2O가 열위(EMR 0.506 vs 0.519)였던 것과 대조적.

**3B 모델:**
- **PPL01 (Literary)**: ROUGE-L, BLEU-4에서 H2O 미세 우위 (+0.010, +0.008). Top-K에서는 Sliding 우위 (0.621 vs 0.610).
- **PPL03 (Technical)**: EMR에서 H2O 우위 (0.522 vs 0.514), BLEU-4에서도 우위 (+0.012). 반면 ROUGE-L, Top-K에서는 Sliding이 우위.
- **평균**: 메트릭별로 엇갈리며, 실질적 동등.

### 3.2 모델 크기별 Eviction 내성

| 메트릭 | 1B 평균 | 3B 평균 | 차이 |
|--------|--------:|--------:|-----:|
| EMR | 0.516 | 0.514 | -0.002 |
| ROUGE-L | 0.589 | 0.618 | **+0.029** |
| BLEU-4 | 0.529 | 0.534 | +0.005 |
| Top-K Overlap (post-FDT) | 0.070 | 0.215 | **+0.145** |

**3B가 eviction 후 예측 분포를 더 잘 보존한다.** 특히 post-FDT Top-K Overlap에서 3B(0.215)가 1B(0.070)보다 3배 높다. 이는 더 큰 모델이 KV 캐시 손실에도 불구하고 유사한 토큰 분포를 유지할 수 있음을 시사한다.

### 3.3 H2O 속도 오버헤드

H2O는 attention score 누적 연산으로 인해 Sliding 대비 **13-25%의 TBT 오버헤드**가 발생한다.

| Model | H2O Overhead (vs Base) | Sliding Overhead (vs Base) |
|-------|----------------------:|---------------------------:|
| 1B | +18.4% ~ +25.3% | +1.0% ~ +1.5% |
| 3B | +13.9% ~ +15.9% | -1.2% ~ -1.3% |

- 3B에서 Sliding이 Base보다 미세하게 빠른 것은 eviction 후 캐시 크기 축소로 인한 attention 연산 감소 효과.
- H2O 오버헤드는 3B(~15%)가 1B(~22%)보다 상대적으로 작음 — matmul 비중이 커져 score 누적의 상대 비용이 줄어듦.

### 3.4 시간 정규화 효과 (Round 15 대비)

| | Round 15 (raw SUM) | 이번 벤치마크 (time-normalized) |
|---|---|---|
| H2O vs Sliding (1B EMR) | -0.013 (열위) | +0.010 (우위) |
| 원인 | 누적 바이어스: 오래된 토큰 과대평가 | 단위 시간당 중요도로 공정 비교 |

**시간 정규화가 H2O의 토큰 선택 품질을 실질적으로 개선**했다. Raw SUM 방식에서 "가장 오래 캐시에 머문 토큰"을 선택하던 역방향 편향이 제거됨.

---

## 4. 결론

1. **시간 정규화 후 H2O는 1B에서도 Sliding과 동등하거나 약간 우수** — 이전 Round 15 분석에서 발견된 누적 바이어스 문제가 해결됨.
2. **3B 모델은 eviction에 더 견고** — post-FDT Top-K Overlap이 1B 대비 3배 높아, 큰 모델이 캐시 손실을 더 잘 보상.
3. **H2O는 정확도 이점이 있으나 13-25% 속도 오버헤드 수반** — attention score 추적 비용. 정확도가 중요한 시나리오에서 선택적 사용 권장.
4. **두 모델 공통으로 eviction 직후(FDT 이후) 급격한 발산** 발생 — Suffix EMR이 0.008-0.032로 매우 낮아, 한번 발산하면 복구가 어려움.

---

## 5. 실험 파일

| 파일 | 설명 |
|------|------|
| `experiments/run_accuracy_bench.sh` | 벤치마크 실행 스크립트 |
| `experiments/results/accuracy_bench/*.jsonl` | 실험 원본 데이터 (12개) |
| `experiments/results/accuracy_bench/*.scores.csv` | H2O score 진단 데이터 (4개) |
| `experiments/analysis/compare.py` | 비교 분석 도구 |

---

## 6. 부수 성과

이번 벤치마크 수행 과정에서 다음 코드 변경이 이루어짐:

- **Multi-shard safetensors 로딩** (`engine/src/models/llama/llama_model.rs`): `model.safetensors.index.json`을 파싱하여 여러 shard 파일에서 텐서를 로드. 3B 이상 모델 지원.
- **k_dim 계산 버그 수정** (`engine/src/bin/generate.rs`): `hidden_size / 4` 하드코딩 → `num_kv_heads * head_dim`. 1B에서는 우연히 맞았으나 3B에서 panic 발생 원인이었음.
