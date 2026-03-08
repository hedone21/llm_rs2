# Round 12: H2O 성능 저하 원인 분석 Report

## 실험 개요

H2O가 Sliding보다 열등한 원인을 4가지 가설로 분석.
26개 신규 실험 + Round 11 baseline 재사용.

| Phase | 가설 | 실험 수 | 핵심 변수 |
|-------|------|---------|----------|
| 1 | H1: Budget Split | 14 | keep_ratio 0.0~1.0 |
| 3 | H3: Decay | 10 | decay 0.0~0.9 |
| 4 | H5: HH Value | 2 | Sliding(86) vs H2O(86+86) |

---

## Phase 1: Keep Ratio Sweep (H1)

| keep_ratio | HH | Recent | Literary EMR | Technical EMR |
|-----------|-----|--------|------------|------------|
| 0.0 | 0 | 172 | **0.992** | **1.000** |
| 0.1 | 17 | 155 | 0.513 | 0.531 |
| 0.2 | 34 | 138 | 1.000 | 0.509 |
| 0.3 | 51 | 121 | 0.514 | 0.509 |
| 0.5 | 86 | 86 | 0.517 | 0.511 |
| 0.7 | 120 | 52 | 0.514 | 0.505 |
| 0.9 | 154 | 18 | 0.512 | 0.507 |
| 1.0 | 172 | 0 | 0.506 | 0.503 |
| Sliding | 0 | 172 | **0.992** | **1.000** |


## Phase 3: Decay Sweep (H3)

| decay | 유효 범위 | Literary EMR | Technical EMR |
|-------|----------|------------|------------|
| 0.00 | ∞ | 0.517 | 0.511 |
| 0.01 | ~298 steps | 0.517 | 0.511 |
| 0.05 | ~58 steps | 0.517 | 0.511 |
| 0.10 | ~28 steps | 0.517 | 0.511 |
| 0.50 | ~4 steps | 0.517 | 0.511 |
| 0.90 | ~1 steps | 0.517 | 0.511 |


## Phase 4: Sliding Reduction (H5)

| Config | Recent | HH | Total | Literary EMR | Technical EMR |
|--------|--------|-----|-------|------------|------------|
| SL-172 | 172 | 0 | 212 | 0.992 | 1.000 |
| SL-086 | 86 | 0 | 126 | 0.507 | 0.509 |
| H2O-50 | 86 | 86 | 212 | 0.517 | 0.511 |


## 가설 판정 결과

| 가설 | 판정 | 근거 |
|------|------|------|
| H1: Budget Split | CONFIRMED | KR-00 vs Sliding EMR 비교 |
| H3: Decay | REJECTED | Decay별 EMR 범위 |
| H5: HH Value | CONFIRMED | SL-086 vs H2O-50 비교 |

