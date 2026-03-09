# Round 14: H2O/H2O+ with Real Attention Scores

## 1. 배경

Round 12-13에서 "H2O의 HH가 무가치하다"는 결론을 내렸으나,
근본 원인은 **Q4_0 KV 캐시에서 어텐션 스코어가 전혀 수집되지 않았던 버그**였다.

`attention_gen()` (Q4_0/F16용)은 `ws.scores`에 기록하지 않아,
모든 중요도 점수가 0으로 남았다. H2O/H2O+는 사실상 랜덤 제거를 수행하고 있었다.

**수정**: `compute_attention_scores()` — Q4_0/F16 K 캐시를 CPU에서 역양자화,
Q·K^T + softmax 계산 후 `ws.scores`에 기록. (`need_scores` 플래그로 조건부 활성화)

## 2. 목표

스코어 버그 수정 후 **Round 12/13의 핵심 질문을 재검증**:

1. **H2O의 HH는 정말 무가치한가?** (Round 12 결론 재검토)
2. **Per-head HH 선택(H2O+)이 H2O를 개선하는가?** (Round 13 재검토)
3. **최적 keep_ratio는?**

## 3. 실험 설계

### Phase 1: Baselines (재사용 가능)

Baseline과 Sliding은 스코어에 의존하지 않으므로 Round 13 결과 재사용.

| ID | 정책 | 출처 |
|----|------|------|
| BASE-PPL01/03 | none | Round 13 |
| SL-PPL01/03 | sliding | Round 13 |

### Phase 2: H2O Keep Ratio Sweep (with real scores)

| ID | keep_ratio | HH | Recent | 비고 |
|----|-----------|-----|--------|------|
| H2O-00-PPLxx | 0.0 | 0 | 172 | Sliding과 동일해야 |
| H2O-10-PPLxx | 0.1 | 17 | 155 | |
| H2O-20-PPLxx | 0.2 | 34 | 138 | |
| H2O-30-PPLxx | 0.3 | 51 | 121 | |
| H2O-50-PPLxx | 0.5 | 86 | 86 | |
| H2O-70-PPLxx | 0.7 | 120 | 52 | |
| H2O-90-PPLxx | 0.9 | 154 | 18 | |

신규: 7 ratios × 2 prompts = **14 실험**

### Phase 3: H2O+ Keep Ratio Sweep (with real scores)

동일 keep_ratio 범위로 H2O+도 sweep:

| ID | keep_ratio | 비고 |
|----|-----------|------|
| H2OP-00-PPLxx | 0.0 | |
| H2OP-10-PPLxx | 0.1 | |
| H2OP-20-PPLxx | 0.2 | |
| H2OP-30-PPLxx | 0.3 | |
| H2OP-50-PPLxx | 0.5 | |
| H2OP-70-PPLxx | 0.7 | |
| H2OP-90-PPLxx | 0.9 | |

신규: 7 ratios × 2 prompts = **14 실험**

### Phase 4: H2O vs H2O+ Head-to-Head

Phase 2+3 데이터로 keep_ratio별 비교. 신규 실험 0.

## 4. 실험 조건

- 프롬프트: PPL01 (Literary), PPL03 (Technical)
- 토큰: 2048, inject@1024, 80% eviction (ratio=0.20)
- decay: 0.0
- KV 캐시: Q4_0 (스코어 수정 적용)

## 5. 총 실험 수

| Phase | 신규 | 재사용 |
|-------|------|--------|
| Phase 1 (Baselines) | 0 | 4 |
| Phase 2 (H2O sweep) | 14 | 0 |
| Phase 3 (H2O+ sweep) | 14 | 0 |
| Phase 4 (Head-to-Head) | 0 | 28 |
| **합계** | **28** | **32** |

예상 시간: 28 × ~120초 ≈ **56분** (스코어 계산 오버헤드 포함)

## 6. 판정 기준

### 시나리오 A: Real Scores가 HH를 복구
```
H2O kr=0.5 EMR >> 이전 Round 12 값 (0.517)
H2O kr=0.5 EMR ≈ Sliding EMR
→ "HH는 유효하나, 이전에 스코어 버그로 작동하지 않았다"
```

### 시나리오 B: Real Scores로도 HH 무가치
```
H2O kr=0.5 EMR ≈ 이전 값 (0.517)
H2O kr=0.0이 최고
→ "HH 개념 자체가 1B 모델에서 무효. 스코어와 무관"
```

### 시나리오 C: Per-Head가 추가 개선
```
H2O+ kr=0.5 EMR > H2O kr=0.5 EMR
→ "Per-head 선택이 HH에 가치 추가"
```
