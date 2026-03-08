# Benchmark Prompts

KV Cache Eviction 평가를 위한 벤치마크 프롬프트.
학술 논문의 평가 방법론을 기반으로 설계되었다.

방법론 상세: [docs/30_evaluation_methodology.md](../../docs/30_evaluation_methodology.md)

## 프롬프트 카테고리

### 1. Perplexity (PPL-01 ~ PPL-05)

**목적**: 다양한 도메인에서 텍스트 continuation 품질 측정

| ID | 도메인 | 설명 |
|----|--------|------|
| PPL-01 | Literary | 문학 작품 스타일 서사 |
| PPL-02 | Encyclopedic | 백과사전/과학 설명문 |
| PPL-03 | Technical | 알고리즘/자료구조 기술 문서 |
| PPL-04 | Conversational | 대화체 일상 설명 |
| PPL-05 | News | 뉴스 기사 보도문 |

**사용법**:
```bash
# Baseline (full cache)
cargo run --release --bin generate -- \
  -m $MODEL -p "$PROMPT" -n 512 --greedy \
  --experiment-output results/PPL-01-base.jsonl

# Eviction (signal injection at 50%)
cargo run --release --bin generate -- \
  -m $MODEL -p "$PROMPT" -n 512 --greedy \
  --eviction-policy h2o --h2o-keep-ratio 0.5 --h2o-recent-window 128 \
  --experiment-schedule configs/memory_critical_256.json \
  --experiment-output results/PPL-01-h2o.jsonl
```

**메트릭**: EMR, Top-K Overlap, ROUGE-L, BLEU-4

### 2. Needle-in-a-Haystack (NIAH)

**목적**: Eviction이 특정 정보 보존에 미치는 영향 직접 측정

**구성 요소**:
- `filler_blocks` (F-01 ~ F-08): 배경 텍스트 8개 블록
- `needles` (N-PASS ~ N-STAT): 검색 대상 정보 5종
- `depth_ratios`: needle 삽입 깊이 [10%, 25%, 50%, 75%, 90%]

**프롬프트 조합 방법**:
1. 원하는 컨텍스트 길이에 맞게 filler_blocks 선택
2. depth_ratio 위치에 needle 삽입
3. 프롬프트 마지막에 question 배치
4. 생성 초반(1-5 토큰)에 memory_critical 신호 주입

**예시 (depth=25%, 4 blocks)**:
```
[F-01]
[N-PASS: "The special access code for the research database is 58291..."]
[F-02]
[F-03]
[F-04]
What is the special access code for the research database? The access code is
```

**메트릭**: Retrieval Accuracy (binary), Retrieval Score (LCS ratio)

**핵심 가설**:
- Sliding window: needle이 window 밖이면 반드시 검색 실패
- H2O: needle의 attention score가 높으면 보존 → 검색 성공 가능

### 3. QA (LongBench-style)

**목적**: 실제 태스크 수행에서 eviction 영향을 다각적으로 측정

| 카테고리 | 프롬프트 수 | 메트릭 |
|---------|-----------|--------|
| Single-doc QA | 3 (QA-SD-01~03) | F1, EM |
| Summarization | 2 (QA-SUM-01~02) | ROUGE-L |
| Few-shot | 3 (QA-FS-01~03) | Accuracy |
| Multi-hop | 2 (QA-MH-01~02) | F1 |

**사용법**:
```bash
# document + question 을 하나의 prompt로 결합
PROMPT="$DOCUMENT\n\nQuestion: $QUESTION\nAnswer:"

cargo run --release --bin generate -- \
  -m $MODEL -p "$PROMPT" -n 128 --greedy \
  --eviction-policy h2o \
  --experiment-schedule configs/memory_critical_early.json \
  --experiment-output results/QA-SD-01-h2o.jsonl
```

## 기존 프롬프트와의 관계

| 기존 | 새 카테고리 | 비고 |
|------|-----------|------|
| P1 (AI history) | PPL-02 (encyclopedic)로 대체 | 도메인 유사 |
| P2 (Networks) | PPL-03 (technical)로 대체 | 도메인 유사 |
| P3 (Narrative) | PPL-01 (literary)로 대체 | 도메인 유사 |
| — | NIAH (신규) | 검색 정확도 평가 추가 |
| — | QA (신규) | 태스크 수행 평가 추가 |

## 파일 구조

```
experiments/prompts/
├── README.md                    # 이 문서
└── benchmark_prompts.json       # 모든 벤치마크 프롬프트 (3 카테고리)
```
