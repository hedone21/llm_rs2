# Experiment Findings

실험별 발견사항을 누적 기록합니다.

## 실험 진행 현황

| Round | 상태 | 실험 수 | 완료 | 비고 |
|-------|------|--------|------|------|
| 1 | DONE | 5 | 5 | Baseline |
| 2 | DONE | 14 | 14 | 단일 신호 |
| 3 | TODO | 8 | 0 | 주입 위치 변수 |
| 4 | TODO | 9 | 0 | H2O 파라미터 sweep |
| 5 | TODO | 8 | 0 | 복합 조건 |

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
| B-512-sliding | 511 | sliding | 42.2ms | 40.9ms | 1762MB | 1769MB |
| B-512-h2o | 511 | h2o | 41.6ms | 40.3ms | 1762MB | 1769MB |

### 인사이트

1. **TBT는 시퀀스 길이에 비례 증가**: 128→512→1024 토큰에서 36.5→40.7→45.1ms (+23%). KV cache 크기 증가에 따른 attention 계산 비용 증가.
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
6. **후반 eviction은 품질 무영향**: R-C-768-h2o에서 EMR=1.000 (394 토큰 evict 했음에도!). 이는 eviction 후 남은 생성 토큰이 적어(255개) 아직 diverge하지 않았을 수 있음. 또는 768 토큰 시점에서 모델이 이미 출력을 완성했을 수 있음.
7. **조기 eviction은 품질 치명적**: R-C-256-h2o EMR=0.297 (70% 불일치). 초반 eviction은 이후 긴 생성 구간에서 연쇄적으로 diverge.
8. **TBT 오버헤드는 미미**: eviction 포함 실험에서도 TBT 증가 < 5%. eviction 자체의 계산 비용은 낮음.

### 다음 실험 방향 (Round 3)
- **FDT와 eviction 위치의 관계**를 더 세밀하게 측정 (128, 256, 384, 448 토큰 위치에서 eviction)
- **H2O vs sliding 품질 비교**를 더 다양한 조건에서 검증
- H2O 파라미터(keep_ratio, recent_window, decay)가 품질에 미치는 영향 측정 (Round 4)
