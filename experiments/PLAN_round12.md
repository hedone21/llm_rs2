# Round 12: H2O 성능 저하 원인 분석 (Root Cause Analysis)

## 1. 문제 정의

Round 11에서 80% eviction 시 H2O가 Sliding보다 크게 열등:

| 조건 | Sliding EMR | H2O EMR | 차이 |
|------|-----------|---------|------|
| 2048tok, 80% evict | **0.904** | **0.520** | -0.384 |

H2O는 attention score 기반으로 "중요한" 토큰을 선별하여 보존하므로
이론적으로 Sliding(단순 최근 N 토큰)보다 우수해야 한다. 왜 그렇지 않은가?

## 2. 사전 분석: Eviction 시점의 수치적 상황

Round 11 Sub-A (2048tok, inject@1024) 기준:

```
시점: 토큰 1024 생성 후
current_pos ≈ 1064 (프롬프트 ~40 + 생성 1024)
target_ratio = 0.20 (80% eviction)
target_len = floor(1064 × 0.20) = 212
protected_prefix = 40 (프롬프트 길이)
available = 212 - 40 = 172 슬롯

┌─────────────────────────────────────────────────────────┐
│ Sliding Window                                          │
│                                                         │
│ [prefix 40] [         recent 172 토큰           ]       │
│              ←── 토큰 892~1063 (최근 172개) ──→         │
│                                                         │
│ recent window = 172 토큰                                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ H2O (keep_ratio=0.5)                                    │
│                                                         │
│ [prefix 40] [HH 86 토큰] [   recent 86 토큰    ]       │
│              score 기반    ←── 토큰 978~1063 ──→        │
│                                                         │
│ recent window = 86 토큰 (Sliding의 50%)                 │
│ heavy hitters = 86 토큰 (토큰 40~977에서 선별)          │
└─────────────────────────────────────────────────────────┘
```

**핵심 차이**: H2O의 recent window(86)는 Sliding의 recent window(172)의 **절반**.

---

## 3. 가설 목록

### H1: Recent Window 축소가 주원인 (Budget Split)

**주장**: H2O가 available budget의 50%를 heavy hitter에 할당하면서
recent window가 절반으로 줄어든다. Autoregressive 생성에서
각 토큰은 직전 토큰들에 강하게 의존하므로, recent 연속성 파괴가
heavy hitter 보존의 이득을 압도한다.

**예측**: `keep_ratio=0.0`(HH 없음, 100% recent)이면 Sliding과
동일한 EMR이 나와야 함.

**신뢰도**: ★★★★★ (매우 높음)

### H2: 1B 모델의 Attention Score가 부정확

**주장**: 1B 모델은 attention pattern이 noisy하여
score 기반 토큰 선택이 무작위 선택과 다르지 않다.
Heavy hitter로 선택된 토큰이 실제로 "중요한" 토큰이 아닐 수 있다.

**예측**: H2O(score 기반)와 Random(무작위 선택)의 EMR이 유사.

**신뢰도**: ★★★☆☆ (중간)

### H3: Decay 파라미터가 Heavy Hitter 선택을 무력화

**주장**: Round 11의 `h2o_decay=0.1`에서, 100 스텝 전 토큰의
누적 점수는 `0.9^100 ≈ 2.6e-5`로 사실상 0이다.
따라서 heavy hitter는 최근 ~50개 토큰에서만 선택되며,
이는 recent window와 중복되어 budget이 낭비된다.

**수학적 분석**:
```
importance[t] = Σ_{s=0}^{N-1} (1-decay)^(N-1-s) × step_score[s][t]

decay=0.1일 때, 스텝 s에서의 기여도 감쇠:
  10 스텝 전: 0.9^10  = 0.349 (35%)
  50 스텝 전: 0.9^50  = 0.005 (0.5%)
 100 스텝 전: 0.9^100 = 2.6e-5 (≈0%)
1024 스텝 전: 0.9^1024 ≈ 0 (완전 소실)

→ 실질적 유효 기억 범위: ~50 스텝

cf. decay=0.0일 때:
  모든 스텝의 기여도 동일 (1.0)
  → 1024 스텝 동안 꾸준히 attention 받은 토큰이 최고 점수
  → attention sink(프롬프트 토큰)에 편향 가능성
     (단, prefix가 이미 보호되므로 evictable 범위 밖)
```

**예측**: `decay=0.0`(무감쇠)이면 heavy hitter가 더 의미있는
토큰을 선택하여 EMR 개선. `decay=0.9`(극단 감쇠)이면 heavy hitter가
recent window와 완전 중복.

**신뢰도**: ★★★★☆ (높음)

### H5: Heavy Hitter가 생성 품질에 본질적으로 무관

**주장**: Autoregressive 생성에서 "attention이 높은 과거 토큰"을
보존하는 것이 다음 토큰 예측 품질에 실질적으로 기여하지 않는다.
모델은 주로 recent context에서 패턴을 이어가며,
중간에 삽입된 비연속적인 heavy hitter 토큰은 오히려
positional encoding 일관성을 깨뜨릴 수 있다.

**예측**: 동일한 recent window 크기에서도 heavy hitter 유무가
EMR에 유의미한 영향을 주지 않음.

**신뢰도**: ★★★☆☆ (중간, H1이 확정되면 상승)

---

## 4. 실험 설계

### 실행 가능성에 따른 분류

**핵심 인사이트**: Phase 1, 3, 4는 기존 CLI 파라미터만으로 실행 가능.
코드 변경 없이 즉시 시작할 수 있다.

| Phase | 코드 변경 | 즉시 실행 |
|-------|----------|----------|
| Phase 1 (H1) | 불필요 | ✅ |
| Phase 2 (H2) | `--experiment-shuffle-scores` 필요 | ❌ |
| Phase 3 (H3) | 불필요 | ✅ |
| Phase 4 (H5) | 불필요 | ✅ |

---

### Phase 1: Keep Ratio Sweep (H1 검증) — 최우선

H2O의 budget split이 주원인인지 직접 검증. **코드 변경 불필요.**

#### 실험 매트릭스

| ID | keep_ratio | HH 슬롯 | Recent 슬롯 | 비고 |
|----|-----------|---------|------------|------|
| KR-00 | 0.0 | 0 | 172 | ★ 핵심: Sliding과 같아야 함 |
| KR-10 | 0.1 | 17 | 155 | |
| KR-20 | 0.2 | 34 | 138 | |
| KR-30 | 0.3 | 52 | 120 | |
| KR-50 | 0.5 | 86 | 86 | Round 11 재현 (재사용) |
| KR-70 | 0.7 | 120 | 52 | |
| KR-90 | 0.9 | 155 | 17 | |
| KR-100 | 1.0 | 172 | 0 | 극단: recent 없음 |

**공통 조건**:
- 프롬프트: PPL01 (Literary), PPL03 (Technical)
- 토큰: 2048, inject@1024, 80% eviction (ratio=0.20)
- decay: 0.1, tracked_layers: 0 (all)

**대조군** (Round 11 재사용):
- PPL01-2048-base / PPL03-2048-base (baseline)
- PPL01-2048-sl / PPL03-2048-sl (Sliding)
- PPL01-2048-h2o / PPL03-2048-h2o (KR-50)

**신규 실험**: 7 ratios × 2 prompts = **14개**

**분석 방법**:
1. EMR vs keep_ratio 곡선 플롯 (Sliding을 수평선으로 표시)
2. KR-00과 Sliding EMR의 차이 (== 0이면 H1 확정)
3. 최적 keep_ratio 탐색 (EMR이 최대인 지점)
4. EMR 단조성 확인: keep_ratio 증가 → EMR 감소?

**판정 기준**:
```
IF |KR-00 EMR - Sliding EMR| < 0.02:
    → H1 확정: Budget split이 주원인

IF KR-00이 keep_ratio 중 최고 EMR:
    → H5 강화: Heavy hitter가 무가치 또는 유해

IF 최적 keep_ratio ∈ [0.1, 0.3]:
    → 시나리오 C: H2O 개선 가능 (소수 HH + 대다수 recent)

IF 모든 keep_ratio에서 EMR 편차 < 0.05:
    → Score 자체가 무의미 → Phase 2 필수
```

---

### Phase 2: Score Quality Audit (H2 검증) — 조건부 실행

Score 기반 선택이 무작위보다 나은지 검증. **코드 변경 필요.**

> Phase 1에서 H1이 명확히 확정되면 Phase 2는 선택적.
> Phase 1에서 "모든 keep_ratio의 EMR이 유사"하면 Phase 2는 필수.

#### 코드 변경

**`--experiment-shuffle-scores` 플래그** (generate.rs):
- Eviction 시 importance score를 랜덤 셔플 후 H2O에 전달
- H2O의 3-partition 알고리즘은 동일, 입력만 랜덤
- `rand` crate 의존성 추가 (또는 간단한 Fisher-Yates 직접 구현)

```rust
#[arg(long)]
experiment_shuffle_scores: bool,

// Eviction 처리부에서:
let effective_scores = if args.experiment_shuffle_scores {
    let mut shuffled = acc.importance_scores().to_vec();
    // Fisher-Yates shuffle on evictable range only
    let range = protected_prefix..current_pos;
    for i in range.clone().rev() {
        let j = protected_prefix + (simple_rng_next() % (i - protected_prefix + 1));
        shuffled.swap(i, j);
    }
    shuffled
} else {
    acc.importance_scores().to_vec()
};
```

#### 실험 매트릭스

| ID | 방법 | keep_ratio | 비고 |
|----|------|-----------|------|
| RND-50-PPL01 | shuffle | 0.5 | 무작위 HH 선택 |
| RND-50-PPL03 | shuffle | 0.5 | 무작위 HH 선택 |
| H2O-50-PPL01 | score 기반 | 0.5 | Phase 1 KR-50 재사용 |
| H2O-50-PPL03 | score 기반 | 0.5 | Phase 1 KR-50 재사용 |

**신규 실험**: **2개** (RND-50 × 2 prompts)

**판정 기준**:
```
IF |Random EMR - H2O EMR| < 0.02:
    → H2 확정: 1B 모델의 score가 무의미

IF Random EMR < H2O EMR (차이 > 0.05):
    → Score가 유효하나 budget split 손실을 보상 못함

IF Random EMR > H2O EMR:
    → 이상: Score가 오히려 해로움 (추가 조사 필요)
```

---

### Phase 3: Decay Parameter Sweep (H3 검증)

Decay가 heavy hitter 선택 품질에 미치는 영향. **코드 변경 불필요.**

#### 실험 매트릭스

| ID | decay | 유효 기억 범위 | 설명 |
|----|-------|--------------|------|
| D-000 | 0.00 | ∞ (전체) | 무감쇠: 전체 이력 누적 |
| D-001 | 0.01 | ~300 스텝 | 약한 감쇠 |
| D-005 | 0.05 | ~60 스텝 | |
| D-010 | 0.10 | ~30 스텝 | Round 11 기본값 (재사용) |
| D-050 | 0.50 | ~6 스텝 | 강한 감쇠 |
| D-090 | 0.90 | ~1 스텝 | 극단: 직전 스텝만 유효 |

유효 기억 범위 = `log(0.05) / log(1-decay)` (점수가 5% 이상 남는 스텝 수)

**공통 조건**: PPL01, PPL03, 2048tok, 80% evict, keep_ratio=0.5

**신규 실험**: 5 decays × 2 prompts = **10개** (D-010은 Round 11 재사용)

**분석 방법**:
1. EMR vs decay 곡선 (2개 프롬프트)
2. decay=0.0이 최고인지, decay=0.9가 최고인지
3. decay에 따른 EMR 변화량 (큰 변화 = H3 유력, 작은 변화 = H3 기각)

**판정 기준**:
```
IF max(EMR) - min(EMR) < 0.05 across all decays:
    → H3 기각: Decay는 주원인이 아님

IF decay=0.0이 최고 EMR:
    → Decay가 너무 커서 유효한 HH를 찾지 못했음
    → 개선점: decay=0으로 설정 권장

IF decay=0.9가 최고 EMR:
    → HH가 recent에 가까울수록 유리 = recent window 자체가 핵심
    → H1과 일관된 결론
```

---

### Phase 4: Sliding Window 축소 비교 (H5 검증)

**핵심 실험**: 동일한 recent window 크기에서 H2O와 Sliding을 비교.
**코드 변경 불필요.**

H2O(keep_ratio=0.5)의 recent window는 86 토큰.
Sliding의 eviction ratio를 강화하여 동일하게 86 토큰만 유지하면,
heavy hitter 86개의 가치를 직접 측정할 수 있다.

#### 실험 설계

| ID | 정책 | eviction_ratio | 유지 토큰 | 구성 |
|----|------|---------------|----------|------|
| SL-172 | Sliding | 0.20 | ~212 | prefix 40 + recent 172 (재사용) |
| SL-086 | Sliding | 0.118 | ~126 | prefix 40 + recent 86 |
| H2O-50 | H2O | 0.20 | ~212 | prefix 40 + HH 86 + recent 86 (재사용) |

```
SL-086 eviction_ratio 계산:
  target_len = prefix + recent = 40 + 86 = 126
  ratio = 126 / 1064 ≈ 0.118
```

**세 실험의 비교 구조**:
```
SL-172:  [prefix 40] [         recent 172          ]  total=212
SL-086:  [prefix 40] [   recent 86    ]               total=126
H2O-50:  [prefix 40] [HH 86] [recent 86]              total=212
                       ↑
                  이 86개의 가치를 측정
```

**핵심 비교**: `SL-086 vs H2O-50`
- 둘 다 **동일한 86 토큰의 recent window** 보유
- 차이: H2O-50에만 추가로 86개 heavy hitter 존재
- `H2O-50 EMR > SL-086 EMR` → heavy hitter가 가치 있음
- `H2O-50 EMR ≈ SL-086 EMR` → heavy hitter가 무가치 (H5 확정)

**신규 실험**: 2 prompts × 1 = **2개** (SL-086)

**판정 기준**:
```
IF |SL-086 EMR - H2O-50 EMR| < 0.02:
    → H5 확정: 86개 heavy hitter가 품질에 무기여
    → 212 슬롯 중 86개를 HH에 "낭비"한 것

IF SL-086 EMR < H2O-50 EMR (차이 > 0.05):
    → HH가 부분적 유효: 동일 recent에서 H2O가 우수
    → 문제는 HH 자체가 아니라 recent window 축소와의 트레이드오프

IF SL-086 EMR > H2O-50 EMR:
    → HH가 해로움: 비연속적 토큰 삽입이 모델을 혼란
```

---

## 5. 실행 계획

### 전체 실험 수 요약

| Phase | 가설 | 신규 | 재사용 | 코드 변경 | 즉시 실행 |
|-------|------|------|--------|----------|----------|
| Phase 1 | H1 (Budget Split) | 14 | 6 | 없음 | ✅ |
| Phase 2 | H2 (Score 품질) | 2 | 2 | shuffle 플래그 | ❌ |
| Phase 3 | H3 (Decay) | 10 | 2 | 없음 | ✅ |
| Phase 4 | H5 (HH 무가치) | 2 | 4 | 없음 | ✅ |
| **합계** | | **28** | **14** | | |

**예상 실행 시간**: 28 × ~90초 ≈ **42분** (전체 실행 시)

### 실행 순서

```
즉시 실행 가능 (코드 변경 없음):
═══════════════════════════════════════════════
 Phase 1: Keep Ratio Sweep (14개)   ← 최우선
 Phase 3: Decay Sweep (10개)        ← Phase 1과 함께 실행
 Phase 4: Sliding 축소 (2개)        ← Phase 1과 함께 실행
═══════════════════════════════════════════════
                     │
                     ▼
              Phase 1 결과 분석
                     │
         ┌───────────┼───────────┐
         │           │           │
    H1 명확     H1 불확실     시나리오 C
    (KR-00≈SL)  (EMR 유사)   (최적 KR 존재)
         │           │           │
         ▼           ▼           ▼
     결론 도출    Phase 2     결론 도출
     (조기 종료)  (코드 변경    (Phase 3
                  + 2개 실험)   결과 참조)
```

### 조기 종료 조건

```
[최선] Phase 1에서 KR-00 ≈ Sliding (차이 < 0.02):
  → "Budget split이 주원인"으로 결론
  → Phase 3, 4 데이터로 부가 분석
  → Phase 2 불필요
  → 코드 변경 없이 완료

[차선] Phase 1의 모든 keep_ratio에서 EMR 편차 < 0.05:
  → "Score 자체가 무의미" 방향
  → Phase 2 필수 (코드 변경 + 2개 실험)
```

### 단일 스크립트 실행 (Phase 1 + 3 + 4)

Phase 1, 3, 4를 하나의 `run_round12.sh`로 통합. 총 26개 실험.

**예상 실행 시간**: 26 × ~90초 ≈ **39분**

---

## 6. 분석 스크립트

### round12_analyze.py 구조

```python
def analyze_keep_ratio_sweep():
    """Phase 1: KR-00 ~ KR-100의 EMR을 Sliding과 비교"""
    # 1. 각 keep_ratio별 EMR, FDT, ROUGE-L 계산
    # 2. EMR vs keep_ratio 곡선
    # 3. KR-00 vs Sliding 차이
    # 4. 최적 keep_ratio 탐색
    # 5. EMR 단조성 검정

def analyze_decay_sweep():
    """Phase 3: Decay별 EMR 곡선"""
    # 1. 각 decay별 EMR 계산
    # 2. EMR vs decay 곡선
    # 3. 최적 decay 탐색

def analyze_reduced_sliding():
    """Phase 4: Sliding(172) vs Sliding(86) vs H2O(86+86)"""
    # 1. 세 조건의 EMR 비교
    # 2. HH 가치 = H2O EMR - SL-086 EMR

def analyze_random_baseline():
    """Phase 2 (조건부): Random vs H2O"""
    # 1. Random vs H2O EMR 비교

def generate_report():
    """종합 보고서 생성"""
    # Phase 1~4 결과 통합
    # 가설 판정 결과 요약
    # 결론 및 논문 시사점
```

---

## 7. 예상 결론 시나리오

### 시나리오 A: Budget Split 주원인 (가장 유력, 70%)

```
증거 패턴:
  Phase 1: KR-00 ≈ Sliding EMR, EMR이 keep_ratio 감소와 단조 증가
  Phase 3: 모든 decay에서 EMR 유사 (decay는 부차적)
  Phase 4: SL-086 ≈ H2O-50 (HH 86개가 무가치)

결론:
  "H2O의 50:50 budget split은 autoregressive 생성에 부적합.
   recent context 연속성이 heavy hitter 보존보다 압도적으로 중요."

논문 시사점:
  - H2O 원논문의 50:50 split은 prefill PPL 기준 → decode에 부적합
  - 제안: keep_ratio=0.0~0.1 (거의 전부 recent으로 배분)
  - 결론적으로 aggressive eviction에서 Sliding이 H2O를 지배
```

### 시나리오 B: Score 무의미 + Budget Split (20%)

```
증거 패턴:
  Phase 1: 모든 keep_ratio에서 EMR 유사 (score 무관)
  Phase 2: Random ≈ H2O (score가 random과 동일)
  Phase 3: 모든 decay에서 EMR 유사

결론:
  "1B 모델의 attention score는 token importance를 유의미하게
   구별하지 못함. Score 기반 선택 자체가 무효."

논문 시사점:
  - 모델 크기별 H2O 적용 임계점 존재
  - 1B에서는 Sliding 최적, 7B+에서 H2O 재평가 필요
```

### 시나리오 C: 최적 keep_ratio 존재 (10%)

```
증거 패턴:
  Phase 1: KR-10 또는 KR-20에서 EMR이 Sliding 초과
  Phase 3: decay=0.0에서 EMR 개선 (장기 기억이 유효)
  Phase 4: H2O-50 > SL-086 (HH가 부분적 유효)

결론:
  "소수의 진정한 heavy hitter(~10-20%)가 품질에 기여.
   논문의 기본값(50%)은 과도하며, recent 우선 배분이 핵심."

논문 시사점:
  - Adaptive keep_ratio: eviction 강도에 따라 동적 조절
  - Aggressive eviction에서는 recent 극대화 + 최소 HH
```

---

## 8. 코드 변경 상세 (Phase 2 전용)

Phase 2 실행이 필요한 경우에만 적용.

### `--experiment-shuffle-scores` 플래그

**파일**: `engine/src/bin/generate.rs`

```rust
/// Shuffle importance scores before H2O eviction (random baseline ablation).
#[arg(long)]
experiment_shuffle_scores: bool,
```

**Eviction 처리부**:
```rust
let result = if let Some(ref acc) = score_accumulator {
    let scores = if args.experiment_shuffle_scores {
        let mut shuffled = acc.importance_scores().to_vec();
        // Fisher-Yates shuffle on evictable range [prefix..current_pos)
        let prefix = actual_protected_prefix;
        let pos = kv_caches[0].current_pos;
        for i in (prefix + 1..pos).rev() {
            let j = prefix + (rng_state % (i - prefix + 1));
            shuffled.swap(i, j);
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        }
        cache_manager.force_evict_with_scores(&mut kv_caches, effective_ratio, &shuffled)
    } else {
        cache_manager.force_evict_with_scores(
            &mut kv_caches, effective_ratio, acc.importance_scores(),
        )
    };
    // ...
};
```

**영향 파일**: `generate.rs`만 수정 (1개 파일, ~20줄)

---

## 9. 파일 목록

### 생성될 파일

| 파일 | 용도 |
|------|------|
| `experiments/run_round12.sh` | 실험 스크립트 (Phase 1+3+4) |
| `experiments/analysis/round12_analyze.py` | 분석 스크립트 |
| `experiments/reports/round12_report.md` | 결과 보고서 |
| `experiments/results/round12/*.jsonl` | 최대 28개 결과 |

### 수정될 파일 (Phase 2 진행 시)

| 파일 | 변경 | 규모 |
|------|------|------|
| `engine/src/bin/generate.rs` | shuffle 플래그 1개 | ~20줄 |

### Round 11 재사용 파일

| 파일 | 용도 |
|------|------|
| PPL01-2048-base.jsonl | baseline |
| PPL03-2048-base.jsonl | baseline |
| PPL01-2048-sl.jsonl | Sliding 대조군 |
| PPL03-2048-sl.jsonl | Sliding 대조군 |
| PPL01-2048-h2o.jsonl | KR-50 / D-010 |
| PPL03-2048-h2o.jsonl | KR-50 / D-010 |

---

## 10. 성공 기준

이 실험이 완료되면 다음 질문에 정량적으로 답할 수 있어야 한다:

1. **H2O가 Sliding보다 못한 주된 이유는?**
   → Budget split / Score 품질 / Decay 중 정량적 기여도

2. **keep_ratio를 조절하면 H2O 개선이 가능한가?**
   → 최적 keep_ratio 수치 또는 "불가능" 판정

3. **1B 모델에서 attention score 기반 eviction이 유효한가?**
   → Random 비교로 score 유효성 판정 (Phase 2 진행 시)

4. **동일 recent window에서 heavy hitter의 가치는?**
   → SL-086 vs H2O-50 비교로 정량화

---

## 부록: CLI 예시

```bash
# Phase 1: KR-00 (keep_ratio=0.0, PPL01)
./target/release/generate -m $MODEL -p "$PPL01" -n 2048 \
  --max-seq-len 4096 --greedy \
  --eviction-policy h2o --h2o-keep-ratio 0.0 --h2o-decay 0.1 \
  --experiment-schedule $CONFIGS/memory_critical_1024.json \
  --experiment-eviction-ratio 0.20 \
  --experiment-output $RESULTS/KR-00-PPL01.jsonl \
  --experiment-logits-topk 10 --experiment-sample-interval 10

# Phase 3: D-000 (decay=0.0, PPL01)
./target/release/generate -m $MODEL -p "$PPL01" -n 2048 \
  --max-seq-len 4096 --greedy \
  --eviction-policy h2o --h2o-keep-ratio 0.5 --h2o-decay 0.0 \
  --experiment-schedule $CONFIGS/memory_critical_1024.json \
  --experiment-eviction-ratio 0.20 \
  --experiment-output $RESULTS/D-000-PPL01.jsonl \
  --experiment-logits-topk 10 --experiment-sample-interval 10

# Phase 4: SL-086 (Sliding, ratio=0.118, PPL01)
./target/release/generate -m $MODEL -p "$PPL01" -n 2048 \
  --max-seq-len 4096 --greedy \
  --eviction-policy sliding --eviction-window 4096 \
  --experiment-schedule $CONFIGS/memory_critical_1024.json \
  --experiment-eviction-ratio 0.118 \
  --experiment-output $RESULTS/SL-086-PPL01.jsonl \
  --experiment-logits-topk 10 --experiment-sample-interval 10

# Phase 2 (조건부): Random shuffle (PPL01)
./target/release/generate -m $MODEL -p "$PPL01" -n 2048 \
  --max-seq-len 4096 --greedy \
  --eviction-policy h2o --h2o-keep-ratio 0.5 --h2o-decay 0.1 \
  --experiment-shuffle-scores \
  --experiment-schedule $CONFIGS/memory_critical_1024.json \
  --experiment-eviction-ratio 0.20 \
  --experiment-output $RESULTS/RND-50-PPL01.jsonl \
  --experiment-logits-topk 10 --experiment-sample-interval 10
```
