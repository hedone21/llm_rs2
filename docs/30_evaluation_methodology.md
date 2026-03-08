# KV Cache Eviction 평가 방법론

KV cache eviction 기법의 품질을 학술적으로 평가하기 위한 방법론 문서.
관련 논문 조사 결과와 llm.rs 프로젝트에 적합한 평가 전략을 정의한다.

## 1. 관련 연구 조사

### 1.1 논문별 평가 벤치마크 요약

| 논문 | 연도 | 벤치마크 | 주요 메트릭 |
|------|------|---------|------------|
| **H2O** (Zhang et al.) | NeurIPS'23 | COPA, RTE, Winogrande 등 6개 zero-shot + XSUM, CNN/DM, AlpacaEval | Accuracy, ROUGE, Win rate |
| **StreamingLLM** (Xiao et al.) | ICLR'24 | PG19 (4M tokens) | Perplexity |
| **ScissorHands** (Liu et al.) | NeurIPS'23 | WikiText-2, C4 + Winogrande, MathQA 등 | Perplexity, Zero-shot accuracy |
| **FastGen** (Ge et al.) | ICLR'24 | AlpacaEval, GSM8K, HumanEval, NQ, TQA | Win rate, F1, Pass@1 |
| **SnapKV** (Li et al.) | NeurIPS'24 | LongBench 16개 + NIAH (380k) | F1, ROUGE, NIAH accuracy |
| **PyramidKV** (Cai et al.) | TMLR'25 | LongBench 16개 + NIAH (8k) | F1, ROUGE, accuracy |
| **Quest** (Tang et al.) | ICML'24 | PG19, Passkey (100k), LongBench 6개 | Perplexity, F1, ROUGE |
| **KIVI** (Yuan et al.) | ICML'24 | LongBench 8개 + CoQA, TruthfulQA, GSM8K + NIAH | F1, ROUGE, EM, BLEU |
| **MInference** (Jiang et al.) | NeurIPS'24 | RULER (128k), NIAH (1M), InfiniteBench (214k) | Task accuracy, Perplexity |

### 1.2 평가 방법론의 진화

**Phase 1 — 2023 초기 (H2O, ScissorHands)**:
- 퍼플렉시티 (PG19, WikiText-2) + 표준 길이 NLU zero-shot 태스크
- 장문 맥락(long-context)에 특화된 평가 없음
- 주로 고정 길이 입력에서 출력 품질 측정

**Phase 2 — 2024 중기 (SnapKV, PyramidKV, Quest)**:
- **LongBench** (16개 데이터셋, 5k-15k 토큰)가 사실상 표준 벤치마크로 정착
- **Needle-in-a-Haystack (NIAH)** — 장문 맥락 내 정보 검색 능력의 직관적 평가
- 다양한 카테고리: 단일문서 QA, 다중문서 QA, 요약, Few-shot, 합성 태스크, 코드

**Phase 3 — 2024-2025 후기 (SCBench, "Pitfalls", "Hold Onto That Thought")**:
- 다중 턴 시나리오, 지시 따르기(instruction following) 평가
- Chain-of-Thought 추론 중 eviction 영향 측정
- 기존 벤치마크의 한계 지적: 표준 벤치마크가 품질을 과대평가할 수 있음

### 1.3 주요 벤치마크 상세

#### Perplexity (언어 모델링 품질)

**사용 데이터**: PG19 (Project Gutenberg 장편 소설), WikiText-2, C4

**핵심 원리**: Teacher-forcing으로 ground-truth 토큰을 입력하고 각 위치에서의
log-probability를 측정. PPL = exp(-1/N * sum(log P(t_i | t_<i))).

**KV cache eviction에서의 의미**: Eviction 후에도 다음 토큰 예측 분포가
얼마나 유지되는지 측정. PPL_evicted / PPL_baseline 비율이 1.0에 가까울수록 좋음.

**대표 논문**: StreamingLLM (PG19, 4M 토큰까지), ScissorHands (WikiText-2, C4),
Quest (PG19, 0-32k 프롬프트 길이 변화)

#### Needle-in-a-Haystack (NIAH)

**핵심 원리**: 긴 컨텍스트 내에 특정 정보(needle)를 삽입하고, 모델이 이를
정확히 검색할 수 있는지 측정. 검색 정확도를 깊이(depth)와 길이(length)에 따라
2D 히트맵으로 시각화.

**Passkey 변형**: 랜덤 숫자/문자열을 needle로 사용 (예: "The pass key is 89721").
모델이 정확한 passkey를 출력하면 성공.

**KV cache eviction에서의 의미**: Eviction이 needle의 KV 엔트리를 제거하면
검색 실패. H2O가 needle을 "heavy hitter"로 식별하여 보존하는지,
sliding window가 오래된 needle을 제거하는지 직접 비교 가능.

**대표 논문**: SnapKV (380k까지), PyramidKV (8k), MInference (1M까지),
Quest (Passkey, 100k)

#### LongBench

**구성**: 6개 카테고리, 16개 데이터셋

| 카테고리 | 데이터셋 | 메트릭 |
|---------|---------|--------|
| Single-doc QA | NarrativeQA, Qasper, MultiFieldQA | F1 |
| Multi-doc QA | HotpotQA, 2WikiMQA, Musique | F1 |
| Summarization | GovReport, QMSum, MultiNews | ROUGE |
| Few-shot | TREC, TriviaQA, SAMSum | Accuracy/F1/ROUGE |
| Synthetic | PassageCount, PassageRetrieval | Accuracy |
| Code | LCC, RepoBench-P | Similarity |

**KV cache eviction에서의 의미**: 다양한 태스크 유형에서 eviction이 미치는
영향을 종합적으로 평가. 특히 문서 QA (관련 정보 검색 필요)와 요약 (전체 맥락
파악 필요)이 eviction에 민감.

**대표 논문**: SnapKV, PyramidKV, KIVI, Quest, Ada-KV (모두 LongBench 사용)

### 1.4 공통 베이스라인

| 순위 | 베이스라인 | 설명 | 사용 빈도 |
|------|----------|------|----------|
| 1 | Full KV cache | 압축 없음 (상한선) | 모든 논문 |
| 2 | Sliding window | 최근 N 토큰만 유지 | 대부분 |
| 3 | H2O | Heavy hitter + recent | SnapKV 이후 표준 |
| 4 | Random eviction | 랜덤 제거 (하한선/ablation) | H2O, ScissorHands |
| 5 | StreamingLLM | Attention sink + recent | 추론 논문 |

### 1.5 주요 메타 벤치마크 (2024-2025)

- **SCBench** (ICLR'25): KV cache 생명주기 벤치마크. 12개 태스크, 다중 턴/다중 요청
  시나리오. 8개 카테고리의 방법론 비교.
- **"The Pitfalls of KV Cache Compression"** (2024): 다중 지시 환경에서의 실패 모드
  식별. 시스템 프롬프트 누출을 구체적 메트릭으로 측정.
- **"Hold Onto That Thought"** (NeurIPS'25): 추론 태스크 (GSM8K, MATH-500, FOLIO,
  DROP)에서 KV cache 압축 평가. 128-512 cache budget에서 테스트.
- **"KV Cache Compression, But What Must We Give in Return?"** (EMNLP'24 Findings):
  10+ 방법론의 7개 카테고리 장문 태스크 종합 벤치마크.

---

## 2. llm.rs 평가 전략

### 2.1 설계 원칙

**대상 모델**: Llama 3.2 1B (8 KV heads, 64 head_dim, 16 layers)
**컨텍스트**: 최대 2048-8192 토큰 (RoPE 기반)
**제약**: 온디바이스 추론, 외부 ML 프레임워크 없이 자동 평가 가능해야 함

**3-Tier 평가 체계**:

| Tier | 벤치마크 | 측정 대상 | 자동화 수준 |
|------|---------|----------|------------|
| 1 | **Perplexity** | 토큰 예측 분포 품질 | 완전 자동 (top_logits 기반) |
| 2 | **NIAH** | 정보 검색 정확도 | 완전 자동 (문자열 매칭) |
| 3 | **LongBench-style QA** | 태스크 수행 능력 | 반자동 (F1/ROUGE 계산) |

### 2.2 Tier 1: Perplexity 평가

#### 목적
Eviction 후에도 모델의 다음 토큰 예측 분포가 얼마나 유지되는지 측정.

#### 방법론
1. **Baseline 실행**: Full KV cache로 N 토큰 생성 → per-token top_logits 기록
2. **Eviction 실행**: 동일 프롬프트, eviction 적용 → per-token top_logits 기록
3. **비교 메트릭**:
   - **Probability of Baseline Token**: Eviction 실행에서 baseline 토큰의 logit 순위/확률
   - **Top-K Overlap**: 기존 메트릭 활용 (per-token 평균 교집합)
   - **Entropy Ratio**: Eviction 후 logit 엔트로피 변화 (불확실성 증가 여부)
   - **KL Divergence**: Top-K softmax 분포 간 발산

#### 프롬프트 설계
다양한 도메인의 텍스트로 continuation 품질을 측정:
- **PPL-01 (Literary)**: 공공 도메인 문학 작품 발췌
- **PPL-02 (Encyclopedic)**: 백과사전/위키 스타일 설명문
- **PPL-03 (Technical)**: 기술/과학 텍스트
- **PPL-04 (Conversational)**: 대화체 텍스트
- **PPL-05 (News)**: 뉴스 기사 스타일

각 프롬프트는 ~30-50 토큰, 생성 길이 512-2048 토큰.

#### 실험 매트릭스
- **생성 길이**: 512, 1024, 2048 토큰
- **Eviction 정책**: none (baseline), sliding, h2o
- **Eviction 시점**: 25%, 50%, 75% 위치
- **KV 버짓**: 128, 256, 512, 1024 토큰

### 2.3 Tier 2: Needle-in-a-Haystack (NIAH)

#### 목적
Eviction이 컨텍스트 내 특정 정보의 보존에 미치는 영향을 직접 측정.

#### 방법론
1. **프롬프트 구성**: [배경 텍스트] + [Needle] + [배경 텍스트] + [질문]
2. **신호 주입**: 생성 초기(1-5 토큰)에 memory_critical → 즉시 eviction 발동
3. **검증**: 생성된 텍스트에 needle 정보가 포함되어 있는지 확인
4. **메트릭**: Retrieval Accuracy (binary: 정확/부정확)

#### Needle 유형
- **Passkey**: 랜덤 숫자 코드 (예: "The special code is 58291")
- **Fact**: 구체적 사실 (예: "The capital of Freedonia is Silverton")
- **Name**: 인명 (예: "The lead researcher was Dr. Elena Vasquez")
- **Date**: 날짜 (예: "The experiment was conducted on March 15, 1987")

#### 변수
- **Needle 깊이 (Depth)**: 프롬프트 내 needle 위치 — 10%, 25%, 50%, 75%, 90%
- **컨텍스트 길이**: 256, 512, 1024 토큰 (프롬프트 토큰)
- **Eviction 정책**: sliding (needle 위치에 따라 제거 여부 결정적) vs h2o (attention 기반)
- **KV 버짓**: 다양한 cache 크기에서 needle 보존율

#### 핵심 가설
- **Sliding window**: Needle이 window 밖에 있으면 반드시 제거 → 검색 실패
- **H2O**: Needle에 대한 attention score가 높으면 보존 → 검색 성공 가능
- **깊이 효과**: 깊은 needle(50%+)이 sliding에서 더 잘 보존 (최근 window에 가까움)

### 2.4 Tier 3: LongBench-style QA

#### 목적
실제 태스크 수행 능력에 eviction이 미치는 영향을 다각적으로 측정.

#### 태스크 카테고리

**A. Single-document QA (단일 문서 질의응답)**:
- 지문 + 질문 형식
- 정답이 지문 내에 명시적으로 존재
- 메트릭: F1 score (토큰 수준 겹침)

**B. Summarization (요약)**:
- 지문 + "Summarize the above passage" 지시
- 전체 맥락 파악이 필요하므로 eviction에 민감
- 메트릭: ROUGE-L

**C. Few-shot Classification (소수 예제 분류)**:
- 예제 + 테스트 입력 형식
- 예제의 KV가 eviction되면 성능 저하
- 메트릭: Accuracy

**D. Multi-hop Reasoning (다중 단계 추론)**:
- 여러 사실을 조합해야 하는 질문
- 중간 사실의 KV가 eviction되면 추론 실패
- 메트릭: F1 score

#### 프롬프트 설계
Llama 3.2 1B의 능력 범위 내에서 설계:
- 지문 길이: 200-500 토큰 (1B 모델이 처리 가능한 범위)
- 질문: 직접적이고 명확한 질문 (복잡한 추론 최소화)
- 정답: 짧고 명확한 텍스트 (F1 계산 용이)

---

## 3. 평가 메트릭 정의

### 3.1 Perplexity 메트릭

| 메트릭 | 수식 | 의미 |
|--------|------|------|
| **Baseline Token Rank** | rank(baseline_token) in eviction top_logits | 낮을수록 좋음 (1 = 최선) |
| **Baseline Token in Top-K** | 1 if baseline_token in top-K else 0 | Top-K 안에 있는 비율 |
| **Top-K Overlap** | \|top_K_base ∩ top_K_exp\| / K | 기존 메트릭 재활용 |
| **Entropy Ratio** | H(exp_logits) / H(base_logits) | 1.0 = 동일, >1.0 = 불확실성 증가 |

### 3.2 NIAH 메트릭

| 메트릭 | 수식 | 의미 |
|--------|------|------|
| **Retrieval Accuracy** | 1 if needle in generated_text else 0 | 검색 성공 여부 |
| **Retrieval Score** | longest_common_subsequence(needle, output) / len(needle) | 부분 검색 점수 |
| **Depth-Length Heatmap** | accuracy[depth][length] | 깊이×길이 2D 시각화 |

### 3.3 QA 메트릭

| 메트릭 | 수식 | 의미 |
|--------|------|------|
| **F1 Score** | 2PR/(P+R), P=precision, R=recall (토큰 수준) | QA 정확도 |
| **Exact Match (EM)** | 1 if normalize(pred) == normalize(gold) else 0 | 완전 일치 |
| **ROUGE-L** | LCS 기반 F1 | 요약 품질 |

---

## 4. 실험 실행 계획

### 4.1 Round 10: Perplexity 평가

5개 도메인 프롬프트 × 3개 생성 길이 × 3개 eviction 조건 = **45 실험**

```
프롬프트:  PPL-01 ~ PPL-05
생성 길이: 512, 1024, 2048
Eviction:  none (baseline), sliding (inject@50%), h2o (inject@50%)
```

### 4.2 Round 11: NIAH 평가

3개 needle 유형 × 5개 깊이 × 3개 길이 × 2개 eviction = **90 실험**
(실행 가능성을 고려하여 축소 가능)

```
Needle:    passkey, fact, name
깊이:      10%, 25%, 50%, 75%, 90%
길이:      256, 512, 1024
Eviction:  sliding, h2o (baseline은 항상 100% 정확하므로 생략 가능)
```

### 4.3 Round 12: LongBench-style QA

4개 카테고리 × 3개 프롬프트 × 3개 eviction = **36 실험**

```
카테고리:  single_doc_qa, summarization, few_shot, multi_hop
프롬프트:  각 카테고리 3개씩
Eviction:  none (baseline), sliding (inject@50%), h2o (inject@50%)
```

---

## 5. 참고문헌

### 핵심 논문
- Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", NeurIPS 2023
- Xiao et al., "Efficient Streaming Language Models with Attention Sinks", ICLR 2024
- Liu et al., "ScissorHands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time", NeurIPS 2023
- Ge et al., "Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs", ICLR 2024
- Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation", NeurIPS 2024
- Cai et al., "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling", TMLR 2025
- Tang et al., "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference", ICML 2024
- Yuan et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024
- Jiang et al., "MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention", NeurIPS 2024

### 벤치마크 논문
- Bai et al., "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding", ACL 2024
- Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context Language Models?", 2024
- Zhang et al., "InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens", 2024

### 메타 분석 / 서베이
- "KV Cache Compression, But What Must We Give in Return?", EMNLP 2024 Findings
- "The Pitfalls of KV Cache Compression", 2024
- SCBench: "A KV Cache-Centric Analysis of Efficient Long-Context Methods", ICLR 2025
- "Hold Onto That Thought: KV Cache Compression on Reasoning Tasks", NeurIPS 2025
- NACL: "A General and Effective KV Cache Eviction Framework for LLMs at Inference Time", ACL 2024
