# Round 13: H2O+ (Per-Head Eviction) 정확도 비교

## 1. 목표

4가지 eviction 정책의 정확도를 직접 비교:

| 정책 | 설명 | per-head? |
|------|------|-----------|
| **none** (baseline) | eviction 없음, 전체 KV 캐시 유지 | - |
| **sliding** | 최근 N 토큰 유지, FIFO 제거 | No |
| **h2o** | 3-partition (prefix + HH + recent), 모든 헤드 동일 토큰 | No |
| **h2o_plus** | 3-partition, KV-헤드별 독립 HH 선택 (GQA-aware) | **Yes** |

**핵심 질문**: Per-head HH 선택이 H2O의 성능 저하를 복구하는가?

Round 12 결론: H2O의 HH partition은 모든 헤드에서 동일한 토큰을 선택하므로 무가치했다.
H2O+는 각 KV-헤드가 독립적으로 다른 HH 토큰을 선택한다. 이것이 HH의 가치를 복구하는지 검증한다.

---

## 2. Round 12 결과 요약 (배경)

```
80% eviction (inject@1024, eviction_ratio=0.20):
  Sliding:       EMR 0.992 / 1.000  (Literary / Technical)
  H2O (kr=0.5):  EMR 0.517 / 0.511  ← HH 무가치
  H2O (kr=0.0):  EMR 0.992 / 1.000  ← recent만 유지하면 Sliding과 동일
```

결론: budget의 50%를 HH에 할당하면 recent window가 절반으로 줄어, 오히려 성능 하락.

---

## 3. 실험 설계

### Phase 1: 4-Way Direct Comparison (핵심)

**동일 조건에서 4개 정책 비교.**

| ID | 정책 | keep_ratio | 비고 |
|----|------|-----------|------|
| BASE-PPL01 | none | - | Round 11 재사용 |
| BASE-PPL03 | none | - | Round 11 재사용 |
| SL-PPL01 | sliding | - | Round 11 재사용 |
| SL-PPL03 | sliding | - | Round 11 재사용 |
| H2O-PPL01 | h2o | 0.5 | Round 12 KR-50 재사용 |
| H2O-PPL03 | h2o | 0.5 | Round 12 KR-50 재사용 |
| **H2OP-50-PPL01** | h2o_plus | 0.5 | **신규** |
| **H2OP-50-PPL03** | h2o_plus | 0.5 | **신규** |

**공통 조건**:
- 프롬프트: PPL01 (Literary), PPL03 (Technical)
- 토큰: 2048, inject@1024, 80% eviction (ratio=0.20)
- decay: 0.0 (Round 12에서 decay 무관 확인)
- tracked_layers: 0 (all layers)

**신규 실험**: 2개 (H2OP-50 × 2 prompts)

**분석**: 4개 정책의 EMR, FDT, ROUGE-L 비교표 + 차트

### Phase 2: H2O+ Keep Ratio Sweep

**H2O+에서 최적 keep_ratio 탐색.**

H2O+의 per-head 선택이 HH에 가치를 부여한다면, keep_ratio > 0에서 Sliding보다 우수할 수 있다.

| ID | keep_ratio | HH | Recent | 비고 |
|----|-----------|-----|--------|------|
| H2OP-00-PPLxx | 0.0 | 0 | 172 | Sliding과 동일해야 함 |
| H2OP-10-PPLxx | 0.1 | 17 | 155 | |
| H2OP-20-PPLxx | 0.2 | 34 | 138 | |
| H2OP-30-PPLxx | 0.3 | 52 | 120 | |
| H2OP-50-PPLxx | 0.5 | 86 | 86 | Phase 1 재사용 |
| H2OP-70-PPLxx | 0.7 | 120 | 52 | |
| H2OP-90-PPLxx | 0.9 | 155 | 17 | |

**공통 조건**: 2048tok, inject@1024, 80% eviction, decay=0.0

**신규 실험**: 6 ratios × 2 prompts = 12개 (H2OP-50 Phase 1에서 재사용)

**분석**:
1. EMR vs keep_ratio 곡선 (H2O vs H2O+ 오버레이, Sliding 수평선)
2. H2O+의 최적 keep_ratio
3. H2O+ keep_ratio=0.0이 Sliding과 동일한지 확인

### Phase 3: H2O vs H2O+ Head-to-Head (동일 keep_ratio 비교)

Phase 1+2 데이터를 재사용하여 matched comparison:

| keep_ratio | H2O EMR | H2O+ EMR | 차이 |
|-----------|---------|----------|------|
| 0.0 | Round 12 KR-00 | Phase 2 | |
| 0.1 | Round 12 KR-10 | Phase 2 | |
| 0.2 | Round 12 KR-20 | Phase 2 | |
| 0.3 | Round 12 KR-30 | Phase 2 | |
| 0.5 | Round 12 KR-50 | Phase 1 | |
| 0.7 | Round 12 KR-70 | Phase 2 | |
| 0.9 | Round 12 KR-90 | Phase 2 | |

**신규 실험**: 0개 (모두 Phase 1+2 + Round 12 데이터 재사용)

**분석**: keep_ratio별 H2O vs H2O+ EMR 차이 → per-head 선택의 가치를 정량화

---

## 4. 실험 매트릭스 요약

| Phase | 신규 | 재사용 | 코드 변경 | 즉시 실행 |
|-------|------|--------|----------|----------|
| Phase 1 (4-Way) | 2 | 6 | 없음 | ✅ |
| Phase 2 (KR Sweep) | 12 | 2 | 없음 | ✅ |
| Phase 3 (Head-to-Head) | 0 | 14 | 없음 | ✅ |
| **합계** | **14** | **22** | | |

**예상 실행 시간**: 14 × ~90초 ≈ **21분**

---

## 5. 판정 기준

### 시나리오 A: Per-Head가 HH를 복구 (기대)

```
증거 패턴:
  H2OP-50 EMR >> H2O-50 EMR (≥ 0.10 차이)
  H2OP-50 EMR ≈ Sliding EMR
  H2O+에서 최적 keep_ratio ∈ [0.3, 0.7]

결론:
  "Per-head HH 선택은 H2O의 HH partition에 의미있는 가치를 부여한다.
   모든 헤드가 동일 토큰을 선택하는 것이 H2O 실패의 원인이었다."

시사점:
  - GQA 구조에서 per-head eviction이 핵심
  - H2O+가 Sliding을 능가하면 HH 개념 자체는 유효
```

### 시나리오 B: Per-Head로도 HH 무가치

```
증거 패턴:
  H2OP-50 EMR ≈ H2O-50 EMR (차이 < 0.05)
  H2O+ keep_ratio=0.0이 최고 EMR
  Per-head여도 HH가 recent window 축소를 보상 못함

결론:
  "Per-head HH 선택도 autoregressive 생성에서 무가치.
   문제는 HH 선택 방식이 아니라 HH 개념 자체에 있다."

시사점:
  - Aggressive eviction에서 Sliding이 최적
  - HH-based eviction은 1B 모델에서 근본적으로 부적합
```

### 시나리오 C: 부분적 개선

```
증거 패턴:
  H2OP-50 EMR > H2O-50 EMR (0.03~0.10 차이)
  H2OP-50 EMR < Sliding EMR
  최적 keep_ratio ∈ [0.1, 0.2]

결론:
  "Per-head 선택이 소량의 HH에는 가치를 부여하나,
   budget의 대부분은 여전히 recent에 할당해야 한다."

시사점:
  - H2O+ keep_ratio=0.1이 Sliding보다 약간 우수할 가능성
  - 소수 정예 HH + 대다수 recent이 최적 전략
```

---

## 6. 재사용 데이터

### Round 11 (baseline, sliding)
```
experiments/results/round11/PPL01-2048-base.jsonl
experiments/results/round11/PPL03-2048-base.jsonl
experiments/results/round11/PPL01-2048-sl.jsonl
experiments/results/round11/PPL03-2048-sl.jsonl
```

### Round 12 (H2O keep ratio sweep)
```
experiments/results/round12/KR-00-PPL01.jsonl   # h2o kr=0.0
experiments/results/round12/KR-00-PPL03.jsonl
experiments/results/round12/KR-10-PPL01.jsonl   # h2o kr=0.1
experiments/results/round12/KR-10-PPL03.jsonl
experiments/results/round12/KR-20-PPL01.jsonl   # h2o kr=0.2
experiments/results/round12/KR-20-PPL03.jsonl
experiments/results/round12/KR-30-PPL01.jsonl   # h2o kr=0.3
experiments/results/round12/KR-30-PPL03.jsonl
experiments/results/round12/KR-70-PPL01.jsonl   # h2o kr=0.7
experiments/results/round12/KR-70-PPL03.jsonl
experiments/results/round12/KR-90-PPL01.jsonl   # h2o kr=0.9
experiments/results/round12/KR-90-PPL03.jsonl
```

Round 12의 H2O kr=0.5는 Round 11의 h2o.jsonl과 동일 (재사용).

---

## 7. H2O+ CLI 사용법

```bash
# H2O+ (per-head eviction), keep_ratio=0.5, decay=0.0
./target/release/generate -m $MODEL -p "$PROMPT" -n 2048 \
  --max-seq-len 4096 --greedy \
  --eviction-policy h2o_plus \
  --h2o-keep-ratio 0.5 --h2o-decay 0.0 \
  --experiment-schedule configs/memory_critical_1024.json \
  --experiment-eviction-ratio 0.20 \
  --experiment-output results/H2OP-50-PPL01.jsonl \
  --experiment-logits-topk 10 --experiment-sample-interval 10
```

H2O+는 내부적으로 `AttentionScoreAccumulator::new_gqa()`를 사용하여
Q-헤드를 KV 그룹별로 묶어 per-KV-head 2D 중요도를 추적한다.
제거 시 `force_evict_with_head_scores()`를 통해 각 KV-헤드가 독립적으로 HH를 선택한다.

---

## 8. 파일 목록

### 생성될 파일

| 파일 | 용도 |
|------|------|
| `experiments/PLAN_round13.md` | 실험 계획 (이 문서) |
| `experiments/run_round13.sh` | 실행 스크립트 |
| `experiments/results/round13/*.jsonl` | 최대 14개 결과 |
| `experiments/reports/round13_report.md` | 결과 보고서 (실험 후) |

### 수정된 파일

| 파일 | 변경 | 규모 |
|------|------|------|
| `engine/src/bin/generate.rs` | signal-driven eviction에서 head scores 전달 | ~10줄 |

---

## 9. 성공 기준

이 실험이 완료되면 다음 질문에 정량적으로 답할 수 있어야 한다:

1. **Per-head HH 선택이 H2O를 개선하는가?**
   → H2O+ vs H2O EMR 차이로 판정

2. **H2O+가 Sliding을 능가하는가?**
   → H2O+ EMR vs Sliding EMR

3. **H2O+의 최적 keep_ratio는?**
   → Phase 2 sweep 결과

4. **Per-head eviction의 한계는?**
   → 시나리오 A/B/C 중 어디에 해당하는지
