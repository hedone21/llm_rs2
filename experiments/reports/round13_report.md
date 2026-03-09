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

## 근본 원인: Q4_0 어텐션 스코어 미기록 (Critical Bug)

**H2O == H2O+의 진짜 원인은 어텐션 스코어가 전혀 수집되지 않았기 때문이다.**

### 버그 상세

`llama_layer.rs:431`에서 KV 캐시 dtype이 F32가 아니면 `attention_gen()` 경로로 분기:

```rust
if (backend.name() == "OpenCL" && use_gpu_attn) || k_cache.dtype() != DType::F32 {
    backend.attention_gen(...)  // ← ws.scores에 기록하지 않음
} else {
    // CPU F32 경로 — 이 경로만 ws.scores에 post-softmax 점수 기록
}
```

**Q4_0 KV 캐시** (실험에서 사용)는 항상 `attention_gen()` 경로를 타며,
이 함수는 어텐션 출력만 계산하고 **post-softmax 스코어를 ws.scores에 기록하지 않는다**.

### 결과

- `AttentionScoreAccumulator`가 받는 `ws.scores`는 항상 **전부 0.0**
- flat importance = 0, head importance = 0 → 모든 토큰이 동일 중요도
- H2O/H2O+ 모두 "중요도 0인 토큰 중 선택" → 동일한 (사실상 무작위) 결과
- **Round 12의 "HH 무가치" 결론도 이 버그의 영향을 받음**

### 진단 과정

1. H2O+ 진단: 8개 KV 헤드가 100% 동일한 HH 토큰 선택
2. 스코어 진단: `flat nonzero=0/4096, head nonzero=0/32768`
3. 누적기 진단: `accumulate_layer_gqa` 입력 `nonzero=0/131072`
4. 코드 추적: `attention_gen()` 경로에서 scores 미출력 확인

### 시사점

- **Round 12 결론 재검토 필요**: H2O의 HH가 "무가치"한 것이 아니라, 스코어가 0이어서 HH 선택 자체가 작동하지 않았을 가능성
- **수정 방향**: `attention_gen()` 경로에서도 post-softmax 스코어를 출력하거나, 별도 F32 스코어 계산 패스 추가
- **수정 후 Round 12/13 재실험 필요**

---

## 시사점

1. **Q4_0 스코어 버그 수정이 최우선** — 스코어 없이는 score-based eviction 평가 불가
2. **Round 12 "HH 무가치" 결론 재검토 필요** — 실제 스코어로 재실험해야 유효한 결론
3. **비단조적 EMR 패턴** (kr=0.1 > kr=0.0, Literary) — 스코어가 0이므로 사실상 무작위 변동
4. **수정 후 실험 로드맵**:
   - Step 1: `attention_gen()` 경로에서 F32 스코어 출력 구현
   - Step 2: Round 12 keep_ratio sweep 재실험 (H2O with real scores)
   - Step 3: Round 13 H2O vs H2O+ 재비교 (with real scores)
   - Step 4: 결과에 따라 SnapKV 또는 H2O+ 최적화 방향 결정
