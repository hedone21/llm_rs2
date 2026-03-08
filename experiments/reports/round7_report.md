# Round 7: 개선된 H2O 검증 리포트

## 변경 사항 요약

| 항목 | 기존 (Round 4) | 개선 (Round 7) |
|------|---------------|----------------|
| Layer 집계 | SUM (모든 레이어 점수 합산) | **per-layer MAX** (레이어별 독립성) |
| Budget 산출 | `min_keep = prefix + rw + 16` 하드코딩 | **`available * keep_ratio`** 비율 기반 |
| HH:Recent 비율 | ~1:8 (HH=16 고정, Recent=128) | **50:50** (keep_ratio=0.5) |
| Tracked layers | 마지막 3개 (3/16) | **전체 레이어** (0=all) |
| Decay | 0.1 (스텝별 10% 감소) | **0.0** (감소 없음) |
| end_step() | 없음 (직접 importance에 합산) | **step→cumulative 분리** |

---

## 실험 결과

### Group A: 기본 비교 (512 tokens, Memory Critical@256)

| ID | keep_ratio | EMR | FDT | ROUGE-L | BLEU-4 | 비고 |
|----|-----------|-----|-----|---------|--------|------|
| M-C-256-sl (old) | — | **0.687** | 351 | — | — | Sliding Window |
| M-C-256-h2o (old) | 0.5 (rw=128) | 0.593 | 302 | 0.859 | — | 기존 H2O |
| **N-01** (new) | 0.5 | 0.511 | 257 | 0.781 | 0.735 | 개선 H2O |

### Group B: keep_ratio 변수 (512 tokens, inject@256)

| ID | keep_ratio | EMR | FDT | ROUGE-L | effective recent | 비고 |
|----|-----------|-----|-----|---------|-----------------|------|
| H-01/04/07 (old) | 0.3/0.5/0.7 | 0.593 | 302 | 0.859 | 128 (고정) | **kr 무효과** |
| **N-02** (new) | 0.3 | 0.509 | 257 | 0.759 | 84 | |
| **N-01** (new) | 0.5 | 0.511 | 257 | 0.781 | 60 | |
| **N-03** (new) | 0.7 | 0.507 | 257 | 0.760 | 36 | |

### Group C: 주입 위치 변수 (512 tokens, kr=0.5)

| ID | Inject@pos | Old EMR | New EMR | Old FDT | New FDT | ROUGE-L | 변화 |
|----|-----------|---------|---------|---------|---------|---------|------|
| P-128 / **N-04** | 128 | **1.000** | 0.295 | 511 | 145 | 0.467 | **⬇ 대폭 하락** |
| P-256 / **N-01** | 256 | 0.593 | 0.511 | 302 | 257 | 0.781 | ⬇ 하락 |
| P-384 / **N-05** | 384 | 0.885 | 0.769 | 444 | 387 | 0.942 | ⬇ 하락 |
| P-448 / **N-06** | 448 | **0.890** | **0.890** | 454 | 454 | 0.973 | ➡ 동일 |

### Group D: 1024 토큰 (kr=0.5)

| ID | Inject@pos | Old EMR | New EMR | Old FDT | New FDT | ROUGE-L | 변화 |
|----|-----------|---------|---------|---------|---------|---------|------|
| R-C-512-h2o / **N-07** | 512 | 0.546 | **0.990** | 557 | 569 | 0.988 | **⬆ 대폭 향상** |
| R-C-256-h2o / **N-08** | 256 | 0.297 | 0.255 | 302 | 257 | 0.708 | ⬇ 하락 |

---

## Budget 분석

Memory Critical은 `target_ratio=0.50` (cache 50% 유지).

| ID | cache 크기 | keep | HH budget | Recent budget | Old recent |
|----|-----------|------|-----------|---------------|------------|
| N-04 (inject@128) | 146 | 73 | 27 | **28** | 128 (eviction 안 됨) |
| N-01 (inject@256) | 274 | 137 | 59 | **60** | 128 |
| N-05 (inject@384) | 402 | 201 | 91 | **92** | 128 |
| N-06 (inject@448) | 466 | 233 | 107 | **108** | 128 |
| N-07 (inject@512, 1024tok) | 530 | 265 | 123 | **124** | 128 |

---

## 핵심 발견

### 1. HH 선택이 정상 작동 (대형 cache에서 극적 개선)

**N-07 (1024tok, inject@512)**: EMR 0.546 → **0.990** (+81.4%p)

- 동일한 266개 토큰 eviction으로 기존 대비 극적 품질 향상
- per-layer MAX 집계 + 전체 레이어 추적 + 논문식 budget split이 효과를 발휘
- **Suffix EMR = 0.978**: eviction 후에도 거의 완벽한 토큰 일치
- 유효 recent=124, hh=123 → HH 123개 토큰이 실제로 중요 토큰을 올바르게 보존

### 2. 소형 cache에서 품질 하락 (recent window 부족)

| 시점 | Old recent | New recent | Old EMR | New EMR |
|------|-----------|-----------|---------|---------|
| inject@128 | 128 (eviction 안 됨) | 28 | 1.000 | 0.295 |
| inject@256 | 128 | 60 | 0.593 | 0.511 |
| inject@384 | 128 | 92 | 0.885 | 0.769 |
| inject@448 | 128 | 108 | 0.890 | 0.890 |
| inject@512 (1024tok) | 128 | 124 | 0.546 | 0.990 |

**패턴**: recent ≥ ~120일 때 개선, recent < 100일 때 하락.

**원인**: 기존 `recent_window=128`은 절대적 최소 보장(floor)이었지만, 새 budget 기반 설계는
`target_ratio=0.50`으로 공격적 eviction 시 recent 할당이 비례적으로 줄어듦.

### 3. keep_ratio가 이제 실제 효과를 가짐 (방향은 반직관적)

| keep_ratio | HH | Recent | EMR |
|-----------|-----|--------|-----|
| 0.3 | 35 | **84** | 0.509 |
| 0.5 | 59 | **60** | 0.511 |
| 0.7 | 83 | **36** | 0.507 |

- 기존: kr 변화에 EMR 무변화 (0.593 고정) — kr이 budget에 영향 못 미침
- 개선: kr이 HH:Recent 비율을 직접 결정 — **효과가 있으나 미미** (0.507~0.511)
- kr=0.3이 최다 recent(84)를 주지만 EMR 최고가 아님 → 이 cache 크기에서는 HH/Recent 구분보다 **절대적 recent 수**가 중요

### 4. 전환점: cache ≈ 500+ 토큰

- cache 530 (N-07): recent=124 ≈ old 128 → HH 메커니즘이 추가 품질 보존 → **EMR 0.990**
- cache 274 (N-01): recent=60 < old 128 → recent 부족이 HH 이점을 상쇄 → **EMR 0.511**

---

## 결론

### 성공: per-layer MAX + 전체 레이어 추적은 효과적

N-07에서 **EMR 0.546 → 0.990** 향상은 논문의 핵심 메커니즘(per-layer 독립 scoring, 모든 레이어 추적)이
정상 동작함을 입증. HH 토큰 선택이 실제로 중요 토큰을 올바르게 식별하고 있음.

### 문제: 일괄 eviction에서 recent window 최소 보장 필요

논문은 **per-step 1토큰 eviction**을 가정하여 50:50 split이 자연스럽게 동작.
우리 시스템은 **일괄 50% eviction** (target_ratio=0.50)이므로,
고정 budget 비율만으로는 소형 cache에서 recent window가 과도하게 줄어듦.

### 개선 방향

`recent_budget`에 **최소 보장(floor)**을 추가:

```rust
let min_recent = self.min_recent_window; // e.g., 128
let recent_budget = (available - hh_budget).max(min_recent);
let hh_budget = available.saturating_sub(recent_budget); // HH는 나머지
```

이렇게 하면:
- **대형 cache**: 기존처럼 50:50 split → HH 메커니즘 활용 (N-07 수준 품질)
- **소형 cache**: recent가 최소 128 보장 → 기존 수준 이상 품질 유지

---

## 수치 요약

| 비교 항목 | 기존 H2O | 개선 H2O | Sliding | 비고 |
|----------|---------|---------|---------|------|
| 512tok, inject@256 | 0.593 | 0.511 | 0.687 | ⬇ recent 부족 |
| 512tok, inject@128 | 1.000 | 0.295 | — | ⬇⬇ critical |
| 512tok, inject@448 | 0.890 | 0.890 | — | ➡ 동일 |
| **1024tok, inject@512** | **0.546** | **0.990** | — | **⬆⬆ 극적 개선** |
| 1024tok, inject@256 | 0.297 | 0.255 | — | ⬇ recent 부족 |
