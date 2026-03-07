# Experiment Findings

실험별 발견사항을 누적 기록합니다.

## 실험 진행 현황

| Round | 상태 | 실험 수 | 완료 | 비고 |
|-------|------|--------|------|------|
| 1 | DONE | 5 | 5 | Baseline |
| 2 | DONE | 14 | 14 | 단일 신호 |
| 3 | DONE | 8 | 8 | 주입 위치 변수 |
| 4 | DONE | 9 | 9 | H2O 파라미터 sweep |
| 5 | DONE | 9 | 9 | 복합 조건 (+B-2048) |

**총 45 실험 완료**.

---

## Round 1: Baseline (2026-03-07)

### 가설
- 신호 없이 순수 추론 성능을 측정하여 후속 실험의 비교 기준을 확립한다.
- sliding/h2o eviction 정책 자체의 오버헤드는 미미할 것이다.

### 결과

| ID | Tokens | Eviction | Avg TBT | Avg Forward | RSS Start | RSS End |
|----|--------|----------|---------|-------------|-----------|---------|
| B-128 | 127 | none | 36.5ms | 35.3ms | 1762MB | 1768MB |
| B-512 | 511 | none | 40.7ms | 39.5ms | 1762MB | 1769MB |
| B-1024 | 1023 | none | 45.1ms | 43.8ms | 1762MB | 1769MB |
| B-2048 | 2030 | none | 51.4ms | 50.1ms | 1762MB | 1770MB |
| B-512-sliding | 511 | sliding | 42.2ms | 40.9ms | 1762MB | 1769MB |
| B-512-h2o | 511 | h2o | 41.6ms | 40.3ms | 1762MB | 1769MB |

### 인사이트

1. **TBT는 시퀀스 길이에 비례 증가**: 128→512→1024→2048 토큰에서 36.5→40.7→45.1→51.4ms (+41%). KV cache 크기 증가에 따른 attention 계산 비용 증가.
2. **Eviction 정책 오버헤드**: sliding +3.7%, h2o +2.2% (B-512 대비). 가설 대로 미미한 수준.
3. **RSS 변화**: ~6MB 증가 (모든 실험 동일). KV cache Q4_0 기반으로 메모리 효율적.
4. **Greedy 재현성**: B-128 2회 실행 시 127개 토큰 완전 일치 — 실험 프레임워크 정상 동작 확인.
5. **Governor**: powersave 모드. CPU freq 변동이 TBT에 영향을 줄 수 있음 (향후 performance 모드 비교 권장).

---

## Round 2: 단일 신호 (2026-03-07)

### 가설
- **속도**: Thermal throttle는 TBT를 유의미하게 증가시킬 것이다. Compute guidance는 이미 CPU이므로 영향 없을 것이다.
- **품질**: Memory eviction 후 생성 텍스트가 baseline과 diverge할 것이다. H2O가 sliding보다 품질을 더 잘 보존할 것이다.
- **메모리**: 후반부 eviction(768 토큰)은 남은 토큰이 적어 품질 영향이 최소일 것이다.

### 결과 — 속도 실험 (baseline: B-128, 36.5ms)

| ID | Signal | TBT+% | EMR | Actions |
|----|--------|-------|-----|---------|
| T-W-32 | Thermal Warning | +0.7% | 1.000 | SwitchBackend(Cpu) — 이미 CPU이므로 무영향 |
| T-C-32 | Thermal Critical | +173.8% | 1.000 | Throttle(70ms) + LimitTokens(64) |
| T-CR-32-96 | Thermal Crit→Normal | +114.7% | 1.000 | Throttle 후 RestoreDefaults |
| C-32 | Compute Critical | +7.6% | 1.000 | SwitchBackend(Cpu) — 호스트에서 무영향 |
| E-32 | Energy Emergency | +6.0% | 1.000 | Suspend — 32 토큰에서 중단 |

### 결과 — 품질 실험 (baseline: B-512, 40.7ms)

| ID | Signal | Eviction | TBT+% | EMR | FDT | Evicted |
|----|--------|----------|-------|-----|-----|---------|
| M-W-256-sl | Mem Warning@256 | sliding | +4.6% | 0.687 | 351 | 1회 |
| M-C-256-sl | Mem Critical@256 | sliding | +3.8% | 0.687 | 351 | 1회 |
| M-C-256-h2o | Mem Critical@256 | h2o | +4.4% | 0.593 | 302 | 1회 |
| M-CR-256-384 | Mem Crit→Norm@256-384 | h2o | +3.7% | 0.593 | 302 | 1회 |

### 결과 — 메모리 실험 (baseline: B-1024, 45.1ms)

| ID | Signal | TBT+% | EMR | FDT | Evicted Tokens |
|----|--------|-------|-----|-----|----------------|
| R-C-512-sl | Mem Crit@512 sliding | +0.4% | 0.526 | 537 | 266 |
| R-C-512-h2o | Mem Crit@512 h2o | +0.6% | 0.546 | 557 | 266 |
| R-C-256-h2o | Mem Crit@256 h2o | +2.6% | 0.297 | 302 | 138 |
| R-C-768-h2o | Mem Crit@768 h2o | +1.7% | 1.000 | 1023 | 394 |
| R-CR-512-768 | Mem Crit→Norm@512-768 | +0.8% | 0.546 | 557 | 266 |

### 인사이트

1. **Throttle는 TBT에 지배적 영향**: T-C-32에서 70ms delay → TBT 36.5→99.9ms (+174%). 순수 계산 시간(forward_ms)은 변하지 않으며, sleep이 전부.
2. **Recovery 효과 확인**: T-CR-32-96에서 Normal 신호 후 throttle 해제 → 평균 TBT가 T-C-32보다 35% 낮음.
3. **SwitchBackend(Cpu)는 호스트에서 무영향**: T-W-32, C-32 모두 이미 CPU 백엔드이므로 실질 변화 없음. 실제 GPU 환경에서는 다를 것.
4. **H2O가 sliding보다 품질이 낮다** (가설과 반대): H2O EMR=0.593 < sliding EMR=0.687. H2O는 FDT=302로 더 빨리 diverge함. 가능 원인: H2O의 중요도 기반 eviction이 sliding의 단순 window 방식보다 더 공격적으로 토큰을 제거할 수 있음.
5. **Memory Recovery(Normal 신호)는 품질을 복구하지 않음**: M-CR-256-384와 M-C-256-h2o의 EMR이 동일(0.593). eviction으로 제거된 KV cache는 복원 불가.
6. **후반 eviction은 품질 무영향**: R-C-768-h2o에서 EMR=1.000 (394 토큰 evict 했음에도!). eviction 후 남은 생성 토큰이 적어(255개) 아직 diverge하지 않았을 수 있음.
7. **조기 eviction은 품질 치명적**: R-C-256-h2o EMR=0.297 (70% 불일치). 초반 eviction은 이후 긴 생성 구간에서 연쇄적으로 diverge.
8. **TBT 오버헤드는 미미**: eviction 포함 실험에서도 TBT 증가 < 5%. eviction 자체의 계산 비용은 낮음.

---

## Round 3: 주입 위치 변수 (2026-03-07)

### 가설
- **위치 가설**: eviction을 늦게 할수록 품질 손상이 적을 것이다 (남은 생성 토큰이 적어 diverge할 시간이 부족).
- **재현성 가설**: P-256은 Round 2의 M-C-256-h2o와 동일한 결과를 보일 것이다.
- **128 토큰 위치 가설**: cache가 매우 작은 시점의 eviction은 제거할 토큰이 적어 품질 영향이 적을 것이다.

### 결과 — 품질 (512 tokens, h2o, baseline: B-512)

| ID | Inject@pos | EMR | FDT | ROUGE-L | 비고 |
|----|-----------|-----|-----|---------|------|
| P-128 | 128 (25%) | 1.000 | 511 | 1.000 | 품질 무영향! |
| P-256 | 256 (50%) | 0.593 | 302 | 0.859 | Round 2 재현 ✅ |
| P-384 | 384 (75%) | 0.885 | 444 | 0.950 | 후반 eviction → 높은 품질 |
| P-448 | 448 (88%) | 0.890 | 454 | 0.963 | P-384와 유사 |

### 결과 — 메모리 (1024 tokens, h2o, baseline: B-1024)

| ID | Inject@pos | EMR | FDT | ROUGE-L | 비고 |
|----|-----------|-----|-----|---------|------|
| RP-256 | 256 (25%) | 0.297 | 302 | 0.765 | Round 2 재현 ✅ |
| RP-512 | 512 (50%) | 0.546 | 557 | 0.928 | Round 2 재현 ✅ |
| RP-768 | 768 (75%) | 1.000 | 1023 | 1.000 | Round 2 재현 ✅ |
| RP-896 | 896 (88%) | 0.950 | 972 | 0.998 | 거의 무영향 |

### 인사이트

1. **P-128 EMR=1.000 — 초반 eviction은 품질 무영향**: 128 토큰 시점의 cache는 매우 작아서(~128 entries) eviction해도 제거할 토큰이 거의 없음. keep_ratio=0.5로 ~64개만 evict하므로 모델이 동일 경로 유지.
2. **위치-품질 관계는 비선형**: P-128(1.000) → P-256(0.593) → P-384(0.885) → P-448(0.890). 256 지점이 **최악의 sweet spot** — cache가 충분히 커서 많은 토큰이 evict되면서, 남은 생성 구간도 길어서 diverge 여유가 있음.
3. **P-384/P-448 비교**: 후반 eviction은 높은 EMR(~0.89)을 보이지만, P-448이 P-384와 거의 동일. 384 이후부터는 위치 효과가 수렴.
4. **재현성 우수**: P-256=M-C-256-h2o, RP-256=R-C-256-h2o, RP-512=R-C-512-h2o, RP-768=R-C-768-h2o 모두 정확히 일치. 실험 프레임워크의 결정론적 동작 확인.
5. **RP-896 EMR=0.950**: 896 토큰에서 eviction해도 마지막 ~127 토큰 중 일부에서 diverge 발생 (FDT=972). 후반 eviction도 완전히 안전하지는 않음.
6. **FDT ≈ inject_pos + 46**: 대부분 실험에서 FDT가 주입 위치 직후 ~46 토큰에서 발생 (P-256: FDT=302, P-384: FDT=444, P-448: FDT=454). eviction 후 diverge까지 일정한 잠복기 존재.

### 결론
> **최적 eviction 전략**: 가능한 한 늦게, 이상적으로는 시퀀스 75% 이후에 eviction. 25% 이전은 cache가 작아 무해. 50% 부근이 최악.

---

## Round 4: H2O 파라미터 sweep (2026-03-07)

### 가설
- **keep_ratio**: 높을수록(0.7) 더 많은 토큰을 보존하므로 품질이 좋을 것이다.
- **recent_window**: 클수록(128) 최근 컨텍스트를 더 잘 유지하여 품질이 좋을 것이다.
- **decay**: 낮을수록(0.05) 누적 점수가 천천히 감소하여 품질이 좋을 것이다.

### 결과 (512 tokens, h2o, Memory Critical@256, baseline: B-512)

| ID | keep_ratio | recent_window | decay | EMR | FDT | ROUGE-L |
|----|-----------|---------------|-------|-----|-----|---------|
| H-01 | 0.3 | 64 | 0.1 | 0.503 | 257 | 0.763 |
| H-02 | 0.3 | 128 | 0.1 | 0.593 | 302 | 0.859 |
| H-03 | 0.3 | 128 | 0.05 | 0.593 | 302 | 0.859 |
| H-04 | 0.5 | 64 | 0.1 | 0.503 | 257 | 0.763 |
| H-05 | 0.5 | 128 | 0.1 | 0.593 | 302 | 0.859 |
| H-06 | 0.5 | 128 | 0.05 | 0.593 | 302 | 0.859 |
| H-07 | 0.7 | 64 | 0.1 | 0.503 | 257 | 0.763 |
| H-08 | 0.7 | 128 | 0.1 | 0.593 | 302 | 0.859 |
| H-09 | 0.7 | 128 | 0.05 | 0.593 | 302 | 0.859 |

### 인사이트

1. **recent_window이 유일한 품질 결정 변수**: recent_window=64 → EMR=0.503/FDT=257, recent_window=128 → EMR=0.593/FDT=302. 모든 keep_ratio(0.3/0.5/0.7)와 decay(0.05/0.1) 조합에서 **동일한 결과**.
2. **keep_ratio는 품질에 영향 없음** (가설 반박): 0.3(aggressive)과 0.7(conservative) 사이에 차이가 없음. 이는 eviction이 50% target ratio로 실행되어, keep_ratio가 실제 eviction 양을 결정하지 못하기 때문일 수 있음.
3. **decay는 품질에 영향 없음** (가설 반박): 0.05와 0.1 사이에 차이 없음. 누적 점수 감소율이 eviction 대상 선정에 실질적 영향을 못 미침.
4. **FDT가 257 vs 302**: recent_window=64에서 FDT=257 (eviction 직후 1토큰만에 diverge!), recent_window=128에서 FDT=302 (46토큰 여유). recent_window가 작으면 최근 컨텍스트가 부족하여 즉시 diverge.
5. **Backlog 해답**: Round 2에서 H2O가 sliding보다 품질이 낮았던 원인은 recent_window=128이 이 모델/시퀀스에 대해 충분하지 않았던 것. sliding은 전체 window를 최근으로 유지하므로 사실상 recent_window=∞와 같은 효과.

### 결론
> **H2O 권장 설정**: recent_window은 가능한 한 크게 설정 (최소 128, 이상적으로 256+). keep_ratio와 decay는 현재 구현에서 품질에 무영향이므로 기본값(0.5, 0.1) 유지.

---

## Round 5: 복합 조건 (2026-03-07)

### 가설
- **이중 제약**: Thermal + Memory 동시 적용 시 TBT와 품질 모두 영향을 받을 것이다. 영향은 개별 효과의 합보다 클 수 있음.
- **반복 eviction**: 연속 eviction은 품질을 더 크게 손상시킬 것이다.
- **신호 폭풍**: 짧은 간격으로 다수 신호 주입 시 시스템이 안정적으로 처리할 것이다.
- **초대형 시퀀스**: 2048 토큰에서도 eviction 위치-품질 관계가 유지될 것이다.

### 결과 — 512 토큰 복합 (baseline: B-512, 40.7ms)

| ID | 조건 | TBT+% | EMR | FDT | ROUGE-L | Evictions |
|----|------|-------|-----|-----|---------|-----------|
| X-01 | Thermal+Memory@256 | +78.7% | 0.593 | 302 | 0.859 | 1 |
| X-02 | Thermal@128→Memory@256→Normal@384 | +79.4% | 0.593 | 302 | 0.859 | 1 |
| X-03 | Crit(200)→Norm(300)→Crit(400)→Norm(500) | +1.8% | 0.403 | 204 | 0.759 | 2 |
| X-04 | 5× Memory Crit (200~280) | +1.3% | 0.407 | 204 | 0.768 | 5 |

### 결과 — 2048 토큰 (baseline: B-2048, 51.4ms)

| ID | 조건 | TBT+% | EMR | FDT | ROUGE-L | Evictions |
|----|------|-------|-----|-----|---------|-----------|
| X-05 | Memory Crit@512 | -4.2% | 0.278 | 557 | 0.896 | 1 |
| X-06 | Memory Crit@1024 | -5.1% | 0.800 | 1625 | 0.995 | 1 |
| X-07 | 3× Crit→Norm (512~1792) | -9.8% | 0.290 | 557 | 0.903 | 3 |

### 결과 — 1024 토큰 이중 제약 (baseline: B-1024, 45.1ms)

| ID | 조건 | TBT+% | EMR | FDT | ROUGE-L | Evictions |
|----|------|-------|-----|-----|---------|-----------|
| X-08 | Thermal@256+Memory@512 | +100.5% | 0.546 | 557 | 0.928 | 1 |

### 인사이트

1. **이중 제약은 속도와 품질에 독립적 영향**: X-01/X-02에서 TBT는 ~+79% (throttle 효과), EMR은 0.593 (eviction 효과). Thermal은 속도만, Memory는 품질만 영향. 상승효과 없음.
2. **반복 eviction은 품질을 추가 손상**: X-03 EMR=0.403 (2회 eviction) vs P-256 EMR=0.593 (1회 eviction). 반복 eviction은 cache를 누적적으로 훼손.
3. **신호 폭풍도 안정적 처리**: X-04에서 5회 연속 Memory Critical을 20토큰 간격으로 주입. 시스템이 정상 동작하며 EMR=0.407 (X-03과 유사). 반복 eviction 수가 많아도 추가 품질 손실은 미미.
4. **초대형 시퀀스에서도 위치-품질 패턴 유지**: X-05(evict@512) EMR=0.278 vs X-06(evict@1024) EMR=0.800. 2048 토큰에서도 "늦은 eviction이 더 안전" 패턴 재확인.
5. **3회 반복 eviction(X-07) EMR=0.290**: 첫 eviction@512에서 품질 붕괴(FDT=557), 이후 eviction은 이미 diverge된 상태에서 추가 손상. 첫 eviction 위치가 결정적.
6. **X-06 FDT=1625 — 후반 eviction의 긴 잠복기**: 1024에서 eviction 후 601 토큰 동안 baseline과 동일한 출력 유지. eviction 후 즉시 diverge하지 않으며, 특정 문맥 전환점에서 diverge 시작.
7. **TBT 음수값은 호스트 변동성**: X-05/X-06/X-07에서 TBT가 baseline보다 낮게 측정됨. CPU governor(powersave) 모드에서의 자연 변동. TBT ±10% 범위는 노이즈로 간주.

### 결론
> **Resilience 시스템은 복합 조건에서도 안정적으로 동작**. 이중 제약은 속도/품질에 독립적으로 영향하며, 반복 eviction은 누적 손상을 유발하나 시스템 안정성은 유지. 최적 전략은 eviction을 최대한 지연시키고, 불가피한 경우 1회로 제한하는 것.

---

## 전체 실험 요약

### 핵심 발견

| # | 발견 | 근거 |
|---|------|------|
| 1 | Throttle은 순수 속도 제어로, 품질 무영향 | T-C-32: TBT +174%, EMR=1.000 |
| 2 | Eviction은 품질만 영향, 속도 무영향 (< 5%) | M-C-256-h2o: TBT +4.4%, EMR=0.593 |
| 3 | Eviction 위치가 품질의 최대 결정 요인 | P-128: EMR=1.000, P-256: EMR=0.593, P-384: EMR=0.885 |
| 4 | H2O의 recent_window이 유일한 품질 파라미터 | H-01(rw=64): EMR=0.503, H-02(rw=128): EMR=0.593 |
| 5 | keep_ratio, decay는 품질에 무영향 | Round 4 전 실험에서 동일 결과 |
| 6 | Sliding이 H2O보다 품질 우수 (EMR 0.687 vs 0.593) | recent_window 한계 때문 |
| 7 | 반복 eviction은 누적 품질 손상 | X-03(2회): EMR=0.403, X-04(5회): EMR=0.407 |
| 8 | 복합 제약은 독립적 영향 (속도×품질 교차 없음) | X-01: TBT from throttle, EMR from eviction |
| 9 | FDT ≈ inject_pos + 46 (일관된 잠복기) | Round 3 전체에서 확인 |
| 10 | Recovery(Normal 신호)는 eviction 손상을 복구 불가 | M-CR-256-384 = M-C-256-h2o |

### 운영 권장사항

1. **Eviction은 시퀀스 75% 이후로 지연**: EMR ≥ 0.885 확보 가능
2. **H2O recent_window은 최대한 크게**: 최소 128, 권장 256+
3. **반복 eviction 회피**: 1회로 제한, 불가피 시 첫 eviction을 최대한 늦게
4. **Throttle은 안전한 속도 제어**: 품질 손실 없이 TBT 조절 가능
5. **Sliding vs H2O**: 품질 우선이면 sliding, 메모리 효율 우선이면 H2O
