# 28. Resilience 실험 프레임워크 가이드

## 목적

가상의 제약 조건(Thermal, Memory, Compute, Energy)을 토큰 위치 기반으로 주입하여 Resilience 시스템이 LLM 추론의 **속도**, **품질**, **리소스**에 미치는 영향을 정량적으로 측정한다.

## 실험 모드 CLI

### 플래그

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--experiment-schedule` | path | - | 토큰 위치 기반 신호 스케줄 JSON |
| `--experiment-output` | path | - | per-token 결과 JSONL 출력 경로 |
| `--experiment-logits-topk` | usize | 10 | 각 토큰의 top-K logit 기록 수 |
| `--experiment-sample-interval` | usize | 1 | 시스템 메트릭 샘플링 간격 (N 토큰마다) |
| `--greedy` | flag | false | temperature=0 강제 (재현성 확보) |

### 기본 실행 예시

```bash
# Baseline (신호 없음)
./target/release/generate -m /path/to/model \
  -p "The history of artificial intelligence began in" \
  -n 512 --greedy \
  --experiment-schedule experiments/configs/baseline.json \
  --experiment-output experiments/results/B-512.jsonl \
  --experiment-sample-interval 1 --experiment-logits-topk 10

# Memory eviction 실험
./target/release/generate -m /path/to/model \
  -p "The history of artificial intelligence began in" \
  -n 512 --greedy \
  --eviction-policy h2o --h2o-keep-ratio 0.5 --h2o-recent-window 128 --h2o-decay 0.1 \
  --experiment-schedule experiments/configs/memory_critical_256.json \
  --experiment-output experiments/results/M-C-256-h2o.jsonl \
  --experiment-sample-interval 1 --experiment-logits-topk 10
```

## 스케줄 JSON 포맷

```json
{
  "name": "experiment-id",
  "description": "설명",
  "signals": [
    {
      "at_token": 256,
      "signal": {
        "memory_pressure": {
          "level": "critical",
          "available_bytes": 30000000,
          "reclaim_target_bytes": 100000000
        }
      }
    }
  ]
}
```

### 지원 신호 타입

| 타입 | 키 | 레벨 | 효과 |
|------|-----|------|------|
| Memory Pressure | `memory_pressure` | normal/warning/critical | Warning: Evict(0.85), Critical: Evict(0.50) |
| Thermal Alert | `thermal_alert` | normal/warning/critical | Warning: SwitchBackend, Critical: Throttle(70ms)+LimitTokens(64) |
| Compute Guidance | `compute_guidance` | normal/warning/critical | SwitchBackend 권장 |
| Energy Constraint | `energy_constraint` | normal/warning/critical/emergency | Emergency: Suspend+RejectNew |

같은 `at_token`에 여러 신호를 배치하여 복합 조건을 실험할 수 있다.

## JSONL 출력 스키마

### Per-Token 레코드 (매 줄)

```json
{
  "pos": 0,
  "token_id": 892,
  "text": " time",
  "tbt_ms": 45.2,
  "forward_ms": 44.8,
  "signal": "Memory(Critical)",
  "actions": ["Evict(0.5)"],
  "cache_pos": 162,
  "throttle_ms": 0,
  "top_logits": [[892, 16.67], [3347, 16.20]],
  "sys": {
    "rss_mb": 1768.09,
    "cpu_pct": 625.4,
    "cpu_mhz": [2427, 3571],
    "thermal_mc": [70000]
  }
}
```

- `signal`: 주입된 신호 (없으면 필드 없음 또는 null)
- `sys`: `--experiment-sample-interval` 간격으로 수집 (나머지는 null)

### Summary 레코드 (마지막 줄)

```json
{
  "_summary": true,
  "total_tokens": 511,
  "ttft_ms": 2860.86,
  "avg_tbt_ms": 40.74,
  "avg_forward_ms": 39.50,
  "total_throttle_ms": 0,
  "eviction_count": 1,
  "evicted_tokens_total": 138,
  "final_cache_pos": 416,
  "max_seq_len": 2048,
  "prompt": "...",
  "schedule_name": "M-C-256",
  "eviction_policy": "h2o",
  "backend": "cpu",
  "sample_interval": 1
}
```

## 품질 메트릭

| 메트릭 | 설명 | 범위 |
|--------|------|------|
| **FDT** (First Divergent Token) | 실험이 baseline과 처음 달라지는 토큰 위치 | 0~N (높을수록 좋음) |
| **EMR** (Exact Match Rate) | 전체 토큰 일치율 | 0~1 (1.0 = 완벽) |
| **Suffix EMR** | FDT 이후 토큰 일치율 (복구 능력) | 0~1 |
| **ROUGE-L** | LCS 기반 F1 (순서 유지 유사도) | 0~1 |
| **BLEU-4** | 1~4-gram precision 기하평균 | 0~1 |
| **Top-K Overlap** | logit 분포 안정성 (조기 경고 지표) | 0~1 |

## 분석 스크립트

### 개별 비교

```bash
python3 experiments/analysis/compare.py \
  --baseline experiments/results/B-512.jsonl \
  --experiment experiments/results/M-C-256-h2o.jsonl
```

### Round 요약 보고서

```bash
python3 experiments/analysis/round_report.py --round 2    # 특정 round
python3 experiments/analysis/round_report.py --round all   # 전체
```

### 시각화

```bash
# TBT 시계열 (baseline 밴드 + 실험 라인 + 신호/eviction 마커)
python3 experiments/analysis/plot_tbt_timeline.py \
  --baseline experiments/results/B-512.jsonl \
  --experiments experiments/results/P-128.jsonl experiments/results/P-256.jsonl \
  --output experiments/reports/plots/comparison.png

# RSS 시계열
python3 experiments/analysis/plot_rss_timeline.py \
  --experiments experiments/results/RP-256.jsonl experiments/results/RP-512.jsonl \
  --output experiments/reports/plots/rss.png
```

## 실험 Round 설계

| Round | 목적 | 실험 수 | 변수 |
|-------|------|--------|------|
| 1 | Baseline | 5 | 토큰 수별 기준선 |
| 2 | 단일 신호 | 14 | 각 신호 타입의 개별 영향 |
| 3 | 주입 위치 | 8 | eviction 위치별 품질/메모리 |
| 4 | H2O sweep | 9 | keep_ratio × recent_window × decay |
| 5 | 복합 조건 | 9 | 이중 제약, 반복 eviction, 대규모 |

### Round 실행

```bash
bash experiments/run_round1.sh  # ~2분
bash experiments/run_round2.sh  # ~6분
bash experiments/run_round3.sh  # ~5분
bash experiments/run_round4.sh  # ~4분
bash experiments/run_round5.sh  # ~12분 (2048 토큰 포함)
```

### JSONL 검증

```bash
python3 experiments/validate_jsonl.py
```

## 워크플로우

```
 가설 수립 → 스케줄 JSON 작성 → 실험 실행 → 분석 → 인사이트 기록 → 다음 Round
```

1. `experiments/configs/`에 스케줄 JSON 작성
2. `run_roundN.sh` 실행 (또는 직접 CLI 실행)
3. `compare.py` / `round_report.py`로 분석
4. `plot_*.py`로 시각화
5. `experiments/FINDINGS.md`에 발견사항 기록

## 디렉토리 구조

```
experiments/
├── PLAN.md              # 실험 계획서
├── FINDINGS.md          # 누적 발견사항
├── validate_jsonl.py    # JSONL 검증 스크립트
├── run_round[1-5].sh    # Round별 실행 스크립트
├── configs/             # 신호 스케줄 JSON
├── results/             # 실험 결과 JSONL
├── reports/plots/       # 시각화 PNG
└── analysis/            # 분석 스크립트
    ├── quality_metrics.py
    ├── compare.py
    ├── round_report.py
    ├── plot_tbt_timeline.py
    └── plot_rss_timeline.py
```
