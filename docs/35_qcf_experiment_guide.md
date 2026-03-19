# QCF Experiment Guide — 실험 에이전트 인수인계 문서

> **목적**: 이 문서 하나로 실험 에이전트가 llm.rs의 QCF 검증 실험을 독립적으로 수행할 수 있도록 한다.
> **마지막 업데이트**: 2026-03-19

---

## Changelog

| 날짜 | 변경 내용 |
|------|----------|
| 2026-03-19 | 초판 작성. Round 1 결과 포함. proxy→qcf 리네이밍 반영. |
| 2026-03-18 | QCF 시스템 구현 (eviction 수식 수정, layer importance 추가) |
| 2026-03-18 | 실험 스크립트 구현 (run_ppl, run_niah, run_qa, run_mmlu) |

---

## 1. QCF 시스템 개요

### QCF란?

**Quality Cost Function** — 각 lossy action이 추론 품질에 미치는 열화량을 **추가 forward pass 없이** 추정하는 수치.

```
D(action) = α × Q(action)

Q: QCF 값 (action 적용 과정에서 부수적으로 수집)
α: 오프라인 calibration으로 결정된 변환 계수
D: 예상 PPL 증가량
```

### 지원하는 QCF 유형

| QCF 유형 | 수식 | 적용 대상 |
|----------|------|----------|
| **Eviction** | `Σ_evicted attn(t)×‖V(t)‖₁ / Σ_all attn(t)×‖V(t)‖₁` | H2O, SnapKV |
| **Sliding** | `prune_count / total_active` | Sliding Window, StreamingLLM |
| **Quantization** | `0.6×NMSE(K) + 0.4×NMSE(V)` | KIVI flush |
| **Layer Skip** | `Σ importance(skipped) / Σ importance(all)` | SWIFT (미구현) |

### 핵심 결론 (Round 1에서 확인됨)

- **cumulative QCF total**을 사용해야 한다. avg는 sliding window에서 비단조.
- Sliding: 매 step 1토큰 evict → 개별 QCF ~0.001로 매우 작지만 total은 단조 증가.
- H2O: 한 번에 많은 토큰 evict → avg와 total 모두 단조.
- **ρ(QCF_total, ΔPPL) = 1.000** (두 정책 모두 완벽한 rank correlation).

---

## 2. 바이너리 인터페이스

### 빌드

```bash
cd /home/go/Workspace/llm_rs2
cargo build --release -p llm_rs2 --bin generate
# 바이너리: target/release/generate
```

### 실행 모드

#### (A) PPL 평가 — QCF 자동 수집

```bash
target/release/generate \
  --model-path models/llama3.2-1b \
  --ppl <text_file_path> \
  --backend cpu \
  --kv-type f32 \
  --temperature 0 \
  --kv-layout head \
  --max-seq-len 2048 \
  --eviction-policy sliding \
  --kv-budget 900 \
  --protected-prefix 4
```

**출력 (stdout, JSON)**:
```json
{
  "ppl": 118.66,
  "total_nll": 4010.66,
  "token_count": 2047,
  "tokens_per_second": 69.15,
  "wall_time_s": 29.60,
  "qcf_metrics": [
    {
      "step": 900,
      "action": "sliding",
      "raw_value": 0.0011,
      "tokens_affected": 1,
      "cache_pos_before": 901,
      "cache_pos_after": 900
    }
  ],
  "eviction_count": 1147,
  "config": { ... }
}
```

**QCF 요약 계산**:
```python
metrics = result.get("qcf_metrics", result.get("proxy_metrics", []))
qcf_total = sum(m["raw_value"] for m in metrics)
qcf_avg = qcf_total / len(metrics) if metrics else 0
```

#### (B) Eval-LL — 다지선다 NLL 평가 (MMLU용)

```bash
target/release/generate \
  --model-path models/llama3.2-1b \
  --eval-ll \
  --eval-batch batch.json \
  --backend cpu \
  --kv-type f32 \
  --temperature 0 \
  --eviction-policy sliding \
  --kv-budget 900 \
  --protected-prefix 4
```

**입력 (batch.json)**:
```json
[
  {
    "id": "q1",
    "prompt": "Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:",
    "choices": [" A", " B", " C", " D"]
  }
]
```

**출력 (stdout, JSON)**:
```json
{
  "results": [
    {
      "id": "q1",
      "choice_nlls": [1.234, 2.345, 1.567, 3.456],
      "predicted": 0,
      "n_prompt_tokens": 127,
      "eviction_count": 3,
      "evicted_tokens": 450
    }
  ],
  "config": { ... },
  "wall_time_s": 12.34
}
```

#### (C) 텍스트 생성 — NIAH/QA 답변 수집

```bash
target/release/generate \
  --model-path models/llama3.2-1b \
  --prompt "..." \
  -n 64 \
  --backend cpu \
  --kv-type f32 \
  --temperature 0 \
  --eviction-policy sliding \
  --kv-budget 900 \
  --protected-prefix 4 \
  2>/dev/null
```

**출력 (stdout)**: prompt + 생성된 텍스트 (streaming, `\r`으로 업데이트)

**주의**: QCF는 생성 모드에서 수집되지 않음. 같은 프롬프트로 PPL 모드를 별도 실행하여 QCF 수집.

### 핵심 CLI 플래그

| 플래그 | 기본값 | 실험에서의 용도 |
|--------|--------|---------------|
| `--ppl <file>` | — | PPL 평가 + QCF 수집 |
| `--eval-ll --eval-batch <file>` | — | MMLU NLL 평가 |
| `--eviction-policy` | `none` | `sliding`, `h2o`, `h2o_plus`, `d2o`, `streaming` |
| `--kv-budget` | 0 (무제한) | 결정론적 eviction 트리거 지점 |
| `--kv-budget-ratio` | 0.0 | prompt 비율 기반 budget (kv-budget과 상호배타) |
| `--kv-type` | `f16` | **QCF V-norm 접근에 `f32` 필요** |
| `--protected-prefix` | 자동 | score-based → 4, sliding → prompt 길이 |
| `--h2o-keep-ratio` | 0.5 | H2O heavy-hitter 비율 |
| `--h2o-decay` | 0.0 | 중요도 점수 지수 감소 |
| `--h2o-tracked-layers` | 0 (전체) | score 추적 레이어 수 |
| `--d2o-keep-ratio` | 0.75 | D2O 유지 비율 |
| `--d2o-beta` | 0.7 | D2O EMA 임계값 |
| `--temperature` | 0.8 | **실험에서 반드시 `0` 사용** |
| `--kv-layout` | `head` | HeadMajor 레이아웃 |
| `--eviction-window` | 1024 | sliding window 크기 |
| `--eviction-target-ratio` | 0.75 | eviction 시 유지 비율 |
| `--sink-size` | 4 | StreamingLLM sink tokens |
| `--kivi` | false | KIVI Q2 압축 활성화 |
| `-n` | 20 | 생성 토큰 수 |

---

## 3. 모델 정보

### Llama 3.2 1B

| 파라미터 | 값 |
|---------|-----|
| hidden_size (dim) | 2048 |
| num_attention_heads (n_heads_q) | 32 |
| num_key_value_heads (n_kv_heads) | 8 |
| head_dim | 64 |
| num_hidden_layers | 16 |
| intermediate_size (FFN) | 8192 |
| vocab_size | 128256 |
| max_seq_len (inference) | 2048 |
| rope_theta | 500000.0 |

```bash
# 모델 다운로드
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama3.2-1b
```

---

## 4. 실험 스크립트

### 위치

```
experiments/qcf_validation/
├── run_all.sh                   # 마스터 실행
├── PLAN.md                      # 실험 계획
├── REPORT.md                    # 결과 리포트
├── scripts/
│   ├── common.py                # 공통 유틸 (바이너리 호출, JSON 파싱, F1, QCF 요약)
│   ├── run_ppl.py               # Phase 1: PPL sweep
│   ├── run_niah.py              # Phase 2: NIAH
│   ├── run_qa.py                # Phase 3: QA
│   ├── run_mmlu.py              # Phase 4: MMLU
│   └── analyze.py               # Phase 5: 통합 분석
├── results/                     # 실험 결과 JSON
├── plots/                       # 시각화 PNG
└── data/mmlu/                   # MMLU 데이터
```

### 실행

```bash
./experiments/qcf_validation/run_all.sh              # 전체 (~4.5시간)
./experiments/qcf_validation/run_all.sh --phase 1    # PPL (~15분)
./experiments/qcf_validation/run_all.sh --phase 2    # NIAH (~40분)
./experiments/qcf_validation/run_all.sh --phase 3    # QA (~30분)
./experiments/qcf_validation/run_all.sh --phase 4    # MMLU (~3시간)
./experiments/qcf_validation/run_all.sh --phase 5    # 분석만
./experiments/qcf_validation/run_all.sh --skip-mmlu  # MMLU 제외 (~1시간)
```

### Budget 수준

| Key | Budget | 열화 정도 |
|-----|--------|----------|
| B0 | 2048 | Baseline (eviction 없음) |
| B1 | 1600 | Mild (~20% eviction) |
| B2 | 1200 | Moderate (~40%) |
| B3 | 900 | Aggressive (~55%) |
| B4 | 600 | Severe (~70%) |
| B5 | 350 | Extreme (~80%) |

---

## 5. Round 1 결과 (2026-03-19)

### PPL — **핵심 성공**

| Policy | Budget | PPL | ΔPPL | QCF total | Evictions |
|--------|--------|-----|------|-----------|-----------|
| none | 2048 | 7.09 | 0.00 | 0.000 | 0 |
| sliding | 1600 | 8.22 | 1.12 | 0.358 | 1 |
| sliding | 1200 | 15.99 | 8.90 | 0.720 | 5 |
| sliding | 900 | 118.66 | 111.57 | 1.273 | 1147 |
| sliding | 600 | 251.15 | 244.05 | 2.408 | 1447 |
| sliding | 350 | 434.57 | 427.47 | 4.835 | 1697 |
| h2o | 1600 | 17.52 | 10.43 | 0.003 | 447 |
| h2o | 1200 | 48.17 | 41.07 | 0.012 | 847 |
| h2o | 900 | 120.36 | 113.27 | 0.210 | 1147 |
| h2o | 600 | 269.20 | 262.10 | 0.998 | 1447 |
| h2o | 350 | 467.56 | 460.46 | 2.640 | 1697 |

**Rank Correlation**:

| Policy | QCF metric | ρ(QCF, ΔPPL) |
|--------|-----------|--------------|
| Sliding | avg | -0.600 (FAIL) |
| **Sliding** | **total** | **1.000 (PASS)** |
| H2O | avg | 1.000 (PASS) |
| **H2O** | **total** | **1.000 (PASS)** |
| Combined | total | 0.721 |

### NIAH — 100% PASS (45/45)

Sliding window eviction은 NIAH에 무해. Prefill attention이 needle 정보를 hidden state에 인코딩한 후 eviction 발생.

### QA — F1 불변

Document가 뒤쪽에 배치되어 eviction 대상이 아님. 프롬프트 구조 개선 필요.

### MMLU — Noise > Signal

전체 accuracy 24~28% (random 수준). 1B 모델의 ICL 능력 부족.

---

## 6. 알려진 함정 & 주의사항

### 반드시 지켜야 할 것

1. **`--kv-type f32` 사용**: QCF V-norm 계산은 F32에서만 동작. F16/Q4에서는 QCF=0.
2. **`--temperature 0`**: 결정론적 실험 필수.
3. **QCF total 사용**: avg는 sliding에서 비단조. 항상 cumulative sum.
4. **`--protected-prefix 4`**: 미지정 시 sliding은 전체 prompt 보호 → eviction 안 일어남.
5. **JSON 키 호환**: `qcf_metrics` (신버전) 또는 `proxy_metrics` (구버전) 둘 다 확인.

### QCF avg가 비단조인 이유

```
Sliding B1: 1회 eviction, ~400 tokens evict → QCF avg = 0.358
Sliding B3: 1147회 eviction, 매번 1 token → QCF avg = 0.001

avg = "1회 평균" → 횟수와 무관하게 비슷
total = "누적" → 횟수×크기에 비례, 단조 증가
```

### Sliding window eviction의 특성

- Prefill 후 eviction → 이미 attention으로 정보 전파됨
- Decode에서만 영향 (evict된 토큰을 attend 불가)
- Teacher-forcing (PPL)에서만 측정 가능, NIAH/QA 같은 짧은 생성에서는 영향 미미

### 1B 모델의 한계

- MMLU baseline ~25% (random chance)
- ICL 능력 미약 → many-shot eviction 효과 없음
- QA F1 기본 0.02~0.04 수준

---

## 7. Round 2 실험 설계 제안

### 7-1. α Calibration

Round 1 PPL 데이터로 `D = α × QCF_total` fitting. Policy별 분리.

### 7-2. Decode-Phase Eviction

짧은 프롬프트 (~200 tokens) + 긴 생성 (1000+ tokens). Budget=500 → decode 중 eviction. Token Match Rate 측정.

### 7-3. QA 프롬프트 구조 수정

`[Document][Padding][Question]` → document가 eviction 대상. 또는 budget을 document보다 작게.

### 7-4. 3B 모델 / 대안 벤치마크

MMLU는 3B+ 모델 필요. 또는 HellaSwag/ARC-Easy (1B에서 ~40-50%).

### 7-5. Cross-Policy 비교

동일 budget에서 H2O vs Sliding의 NIAH/QA 영향 비교.

---

## 8. Eviction 정책 상세

| 정책 | CLI | 동작 | QCF 수집 |
|------|-----|------|---------|
| none | `--eviction-policy none` | eviction 없음 | — |
| sliding | `--eviction-policy sliding --kv-budget N` | budget 초과 시 oldest 1토큰 삭제 | `compute_sliding_qcf()` |
| streaming | `--eviction-policy streaming --sink-size 4` | sink 보존 + sliding | `compute_sliding_qcf()` |
| h2o | `--eviction-policy h2o --h2o-keep-ratio 0.5` | 3-partition: prefix+HH+recent | `compute_eviction_qcf()` |
| h2o_plus | `--eviction-policy h2o_plus` | per-head GQA-aware H2O | `compute_eviction_qcf()` |
| d2o | `--eviction-policy d2o --d2o-keep-ratio 0.75` | H2O + cosine merge | 별도 handler |

---

## 9. 벤치마크 프롬프트

### benchmark_prompts.json

`experiments/prompts/benchmark_prompts.json`:

| 카테고리 | 항목 | 용도 |
|---------|------|------|
| perplexity | PPL-01~05 (5개 도메인) | PPL 평가 |
| niah | 10 needles + 8 filler blocks | 정보 검색 |
| qa | 3 single-doc + 2 summarization + 2 multi-hop + 3 few-shot | 문서 이해 |

### NIAH 프롬프트 조합

```bash
python experiments/prompts/assemble_niah.py --needle N-PASS --depth 0.25 --blocks 20
```

| Blocks | ~Tokens |
|--------|---------|
| 7 | 530 |
| 12 | 890 |
| 20 | 1500 |

### MMLU 데이터

```bash
python experiments/qcf_validation/scripts/run_mmlu.py --download
# → experiments/qcf_validation/data/mmlu/{subject}.json
# 구조: {"subject": "...", "test": [{"question","choices","answer"}], "dev": [...]}
```

---

## 10. 분석 방법

### Spearman Rank Correlation

```python
def spearman_rho(x, y):
    n = len(x)
    if n < 3: return float('nan')
    def rank(arr):
        s = sorted(range(len(arr)), key=lambda i: arr[i])
        r = [0.0]*len(arr)
        for ri, i in enumerate(s): r[i] = ri + 1.0
        return r
    rx, ry = rank(x), rank(y)
    d2 = sum((a-b)**2 for a,b in zip(rx,ry))
    return 1 - 6*d2/(n*(n*n-1))
```

### QCF 지표 선택

| 상황 | 지표 | 이유 |
|------|------|------|
| Sliding window | **total** | avg 비단조 |
| H2O | avg 또는 total | 둘 다 단조 |
| Cross-policy | **total** | 일관된 비교 |

### 성공 기준

| 벤치마크 | ρ 기준 |
|---------|--------|
| PPL | ≥ 0.85 |
| NIAH | ≥ 0.70 |
| QA | ≥ 0.60 |
| MMLU | ≥ 0.60 |

---

## 11. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| QCF 전부 0 | `--kv-type f16` 사용 | `--kv-type f32` |
| Eviction 없음 | budget 미설정 | `--kv-budget N` (N < prompt length) |
| Sliding eviction 없음 | protected-prefix가 prompt 전체 | `--protected-prefix 4` 명시 |
| PPL JSON 파싱 실패 | stderr 섞임 | stdout만 파싱, stderr 분리 |
| MMLU 0% | 어려운 과목 | human_sexuality, us_foreign_policy 선택 |
| `qcf_metrics` 키 없음 | 구버전 바이너리 | `proxy_metrics`도 확인 |
| 모델 로딩 실패 | 경로 오류 | `models/llama3.2-1b/` 확인 |
| Timeout | 긴 텍스트 + CPU | `--max-seq-len` 축소 또는 timeout 증가 |
