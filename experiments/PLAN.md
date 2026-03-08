# Resilience 실험 계획서

## 1. 목적

가상의 제약 조건(Thermal, Memory, Compute)을 주입하여 Resilience 시스템이 LLM 추론에 미치는 영향을 **정량적으로** 측정한다.

| 측정 영역 | 질문 |
|-----------|------|
| **속도** | 제약 발생 시 토큰 생성 속도가 얼마나 저하되는가? 복구까지 몇 토큰이 걸리는가? |
| **품질** | KV cache eviction이 생성 텍스트의 정확도를 얼마나 손상시키는가? |
| **리소스** | Resilience 동작이 CPU, 메모리, 온도에 어떤 변화를 만드는가? |

## 2. 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│  generate --experiment-schedule schedule.json                    │
│                                                                  │
│  Token 0 ────── Token 1 ────── ... ────── Token N                │
│     │              │                         │                   │
│     │        [at_token=30?]                  │                   │
│     │         inject signal ───► ResilienceManager               │
│     │                             │                              │
│     │                        poll() → actions                    │
│     │                        Throttle / Evict / Suspend          │
│     │              │                         │                   │
│     ▼              ▼                         ▼                   │
│  Record per-token:                                               │
│    token_id, tbt_ms, forward_ms, actions, cache_pos, top_logits │
│  Record per-N-tokens (system sampling):                          │
│    process_rss_mb, cpu_util_pct, cpu_freq_mhz, thermal_mc       │
│                                                                  │
│  ──► experiment_output.jsonl                                     │
└──────────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│  experiments/analysis/compare.py                                 │
│                                                                  │
│  baseline.jsonl  ←── 비교 ──►  experiment.jsonl                  │
│                                                                  │
│  속도 메트릭:  TBT delta, throughput ratio, recovery latency     │
│  품질 메트릭:  FDT, EMR, ROUGE-L, BLEU-4, Top-K Overlap        │
│  리소스 메트릭: RSS delta, CPU freq/util 변화, cache efficiency  │
│                                                                  │
│  ──► report_EXP-001.md + plots/                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 3. 구현: generate.rs experiment mode

### 3.1 새 CLI 플래그

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--experiment-schedule` | path | - | 토큰 위치 기반 신호 스케줄 JSON |
| `--experiment-output` | path | - | per-token 결과 JSONL 출력 경로 |
| `--experiment-logits-topk` | usize | 10 | 각 토큰의 top-K logit 기록 수 |
| `--experiment-sample-interval` | usize | 1 | 시스템 메트릭 샘플링 간격 (N 토큰마다) |
| `--greedy` | flag | false | temperature=0 강제 (재현성 확보) |

`--experiment-sample-interval`로 시스템 메트릭 수집 빈도를 제어한다.
- `1` = 매 토큰 (호스트 CPU에서 오버헤드 < 0.5%, 기본값)
- `5` = 5 토큰마다 (디바이스 환경에서 권장)
- `0` = 시스템 메트릭 비활성화 (순수 속도/품질만 측정)

### 3.2 스케줄 JSON 포맷

```json
{
  "name": "thermal_critical_at_30_recover_60",
  "description": "Token 30에서 Thermal Critical, Token 60에서 복구",
  "signals": [
    {
      "at_token": 30,
      "signal": {
        "thermal_alert": {
          "level": "critical",
          "temperature_mc": 48000,
          "throttling_active": true,
          "throttle_ratio": 0.3
        }
      }
    },
    {
      "at_token": 60,
      "signal": {
        "thermal_alert": {
          "level": "normal",
          "temperature_mc": 35000,
          "throttling_active": false,
          "throttle_ratio": 1.0
        }
      }
    }
  ]
}
```

같은 `at_token`에 여러 신호를 배치할 수 있다 (복합 조건 실험).

### 3.3 토큰 위치 기반 주입 메커니즘

```rust
// experiment mode: 내부 mpsc 채널로 ResilienceManager 생성
let (experiment_tx, rx) = std::sync::mpsc::channel();
let resilience_manager = ResilienceManager::new(rx);

// 추론 루프 내, resilience checkpoint 직전:
if let Some(schedule) = &experiment_schedule {
    for entry in schedule.signals_at(decode_token_index) {
        experiment_tx.send(entry.signal.clone()).ok();
    }
}
// 이후 기존 rm.poll() → execute_action() 흐름 그대로 활용
```

- 기존 `--enable-resilience` (외부 transport)와 **독립적으로 동작**
- 외부 프로세스(signal_injector) 불필요
- `--experiment-schedule` 지정 시 자동으로 resilience manager 활성화

### 3.4 시스템 메트릭 수집

```rust
struct SystemSampler {
    interval: usize,           // N 토큰마다 샘플링
    prev_cpu_times: Option<(u64, u64)>,  // (user+sys, total) for CPU util delta
}

impl SystemSampler {
    /// interval=0이면 None 반환, 아니면 N번째 토큰마다 수집
    fn sample(&mut self, token_pos: usize) -> Option<SystemMetrics> {
        if self.interval == 0 || token_pos % self.interval != 0 {
            return None;
        }
        Some(SystemMetrics {
            process_rss_mb: self.read_rss(),        // /proc/self/statm
            cpu_util_pct: self.read_cpu_util(),      // /proc/self/stat delta
            cpu_freq_mhz: self.read_cpu_freqs(),     // /sys/devices/system/cpu/cpu*/cpufreq
            thermal_mc: self.read_thermal(),          // /sys/class/thermal/thermal_zone*
            gpu_freq_hz: self.read_gpu_freq(),        // /sys/class/kgsl (Android) or null
            gpu_util_pct: self.read_gpu_util(),       // /sys/class/kgsl or null
        })
    }
}
```

읽기 소스:

| 메트릭 | Linux 경로 | 예상 비용 | 비고 |
|--------|-----------|----------|------|
| Process RSS | `/proc/self/statm` | ~2μs | 항상 사용 가능 |
| CPU Utilization | `/proc/self/stat` (delta 계산) | ~3μs | 이전 샘플과의 차이로 계산 |
| CPU Frequency | `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` | ~20μs × cores | 코어별 현재 주파수 |
| Thermal | `/sys/class/thermal/thermal_zone*/temp` | ~10μs × zones | 밀리도(m°C) 단위 |
| GPU Frequency | `/sys/class/kgsl/kgsl-3d0/gpuclk` (Adreno) | ~5μs | Android only, 없으면 null |
| GPU Utilization | `/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage` | ~5μs | Android only, 없으면 null |

호스트(x86_64)에서 8코어 + 4 thermal zone 기준:
- interval=1 (매 토큰): ~200μs/token, TBT ~45ms 대비 **0.4% 오버헤드**
- interval=5: ~40μs/token 평균, **0.09% 오버헤드**

### 3.5 Per-Token JSONL 출력

각 줄이 하나의 토큰 기록:

```jsonl
{"pos":0,"token_id":791,"text":" began","tbt_ms":45.2,"forward_ms":44.8,"signal":null,"actions":[],"cache_pos":25,"throttle_ms":0,"top_logits":[[791,12.3],[450,11.1],[23,10.8]],"sys":{"rss_mb":2048.3,"cpu_pct":94.2,"cpu_mhz":[4200,4200,4200,4200,2400,2400,2400,2400],"thermal_mc":[48200,51000,47800],"gpu_mhz":null,"gpu_pct":null}}
{"pos":1,"token_id":302,"text":" the","tbt_ms":44.9,"forward_ms":44.6,"signal":null,"actions":[],"cache_pos":26,"throttle_ms":0,"top_logits":[[302,13.1],[891,12.0]],"sys":null}
{"pos":4,"token_id":567,"text":" early","tbt_ms":45.1,"forward_ms":44.7,"signal":null,"actions":[],"cache_pos":29,"throttle_ms":0,"top_logits":[[567,12.8],[123,11.5]],"sys":{"rss_mb":2048.5,"cpu_pct":93.8,"cpu_mhz":[4200,4200,4200,4200,2400,2400,2400,2400],"thermal_mc":[48300,51100,47900],"gpu_mhz":null,"gpu_pct":null}}
```

- `sys` 필드: `--experiment-sample-interval N`에 따라 N 토큰마다 기록, 나머지는 `null`
- interval=1이면 매 줄에 `sys` 존재, interval=5이면 pos 0,5,10,... 에서만 존재

#### 필드 사전

| 필드 | 타입 | 설명 |
|------|------|------|
| `pos` | int | 디코드 토큰 순서 (0-based) |
| `token_id` | int | 생성된 토큰 ID |
| `text` | string | 디코딩된 텍스트 |
| `tbt_ms` | float | Time Between Tokens, throttle delay 포함 |
| `forward_ms` | float | 순수 forward pass 시간 (throttle 제외) |
| `signal` | string? | 이 위치에서 주입된 신호 요약 (null = 없음) |
| `actions` | string[] | ResilienceManager 반환 액션 목록 |
| `cache_pos` | int | KV cache 물리적 위치 (eviction 후 감소) |
| `throttle_ms` | int | 적용된 throttle sleep (ms) |
| `top_logits` | [int,float][] | 상위 K개 [token_id, logit_value] |
| `sys.rss_mb` | float? | 프로세스 RSS (MB) |
| `sys.cpu_pct` | float? | 프로세스 CPU 사용률 (%) |
| `sys.cpu_mhz` | int[]? | 코어별 CPU 주파수 (MHz) |
| `sys.thermal_mc` | int[]? | thermal zone별 온도 (m°C) |
| `sys.gpu_mhz` | int? | GPU 주파수 (MHz), 없으면 null |
| `sys.gpu_pct` | float? | GPU 사용률 (%), 없으면 null |

#### Summary 레코드 (마지막 줄)

```jsonl
{
  "_summary": true,
  "total_tokens": 128,
  "ttft_ms": 1200.5,
  "avg_tbt_ms": 52.3,
  "avg_forward_ms": 45.1,
  "total_throttle_ms": 2100,
  "eviction_count": 2,
  "evicted_tokens_total": 45,
  "final_cache_pos": 83,
  "max_seq_len": 2048,
  "prompt": "The history of artificial intelligence began in",
  "schedule_name": "thermal_critical_at_30",
  "eviction_policy": "h2o",
  "h2o_keep_ratio": 0.5,
  "h2o_recent_window": 128,
  "h2o_decay": 0.1,
  "backend": "cpu",
  "sample_interval": 1,
  "sys_start": {
    "rss_mb": 2020.1,
    "cpu_mhz": [4200,4200,4200,4200,2400,2400,2400,2400],
    "thermal_mc": [45000,48000,44000],
    "governor": "performance"
  },
  "sys_end": {
    "rss_mb": 2048.3,
    "cpu_mhz": [4200,4200,4200,4200,2400,2400,2400,2400],
    "thermal_mc": [48200,51000,47800]
  }
}
```

## 4. 측정 메트릭

### 4.1 속도 메트릭

| 메트릭 | 수식 | 의미 |
|--------|------|------|
| **TBT Ratio** | `avg_tbt_exp / avg_tbt_base` | 1.0 = 영향 없음, 2.0 = 2배 느림 |
| **Forward-Only Ratio** | `avg_forward_exp / avg_forward_base` | throttle 제외한 순수 계산 영향 |
| **Throttle Overhead** | `total_throttle_ms / total_time_ms` | 전체 시간 중 sleep 비율 |
| **Throughput Ratio** | `(tokens/sec)_exp / (tokens/sec)_base` | 처리량 비교 |
| **Recovery Latency** | Normal 신호 후 TBT가 baseline ±10% 이내로 돌아오는 토큰 수 | 복구 속도 |

### 4.2 품질 메트릭

#### Phase 1 (필수, 직접 구현)

| 메트릭 | 수식 | 의미 |
|--------|------|------|
| **FDT** (First Divergent Token) | `min(i) where exp[i] != base[i]` | 품질 영향 시작점. 높을수록 좋음 |
| **EMR** (Exact Match Rate) | `count(exp[i]==base[i]) / N` | 전체 토큰 일치율. 1.0 = 완벽 |
| **Suffix EMR** | `count(match for i>=FDT) / (N-FDT)` | eviction 이후 복구 능력 |
| **ROUGE-L** | LCS 기반 F1 | 순서 유지 유사도. Lin (2004) |
| **BLEU-4** | 1~4-gram precision 기하평균 + brevity penalty | N-gram 정밀도. Papineni et al. (2002) |
| **Top-K Overlap** | `\|topK_base ∩ topK_exp\| / K` (per-token 평균) | logit 분포 안정성. 조기 경고 지표 |

> **Top-K Overlap이 중요한 이유**: 토큰이 같아도 logit 분포가 불안정해지면 곧 diverge할 위험 신호.
> FDT 이전 토큰들의 Top-K Overlap 추이를 보면 "아직 같은 토큰을 선택하지만 불안정해지고 있다"를 포착 가능.

#### Phase 2 (확장, 외부 모델 필요)

| 메트릭 | 설명 | 도구 |
|--------|------|------|
| **BERTScore** | 토큰별 contextual embedding 매칭 P/R/F1 | `pip install bert-score` |
| **Cosine Similarity** | 문장 임베딩 유사도 | `sentence-transformers` |
| **KL Divergence** | top_logits에서 softmax 후 분포 발산 계산 | 직접 구현 |
| **METEOR** | Unigram 매칭 (exact + stem + synonym) | `nltk` |

### 4.3 리소스 메트릭

| 메트릭 | 수식 | 의미 |
|--------|------|------|
| **RSS Delta** | `max(rss) - min(rss)` 또는 eviction 전후 차이 | 메모리 절감 효과 |
| **RSS Peak** | `max(rss_mb)` | 최대 메모리 사용 |
| **CPU Freq Stability** | `std(cpu_freq) / mean(cpu_freq)` | 주파수 변동률 (CV). 높으면 실험 오염 가능 |
| **CPU Util Avg** | `mean(cpu_pct)` | 평균 CPU 사용률 |
| **CPU Util at Eviction** | eviction 발생 토큰의 cpu_pct | eviction CPU 비용 |
| **Thermal Rise** | `thermal_end - thermal_start` (m°C) | 추론 중 온도 상승 |
| **Thermal Peak** | `max(thermal_mc)` | 최대 온도 |
| **Cache Utilization** | `final_cache_pos / max_seq_len` | KV cache 사용률 |
| **Eviction Count** | 총 eviction 횟수 | 안정성 지표 |
| **Eviction Ratio** | `evicted_tokens / total_generated` | 토큰당 eviction 비율 |
| **Cache Efficiency** | `EMR / (1 - cache_utilization)` | 메모리 절감 대비 품질 유지 |

> **CPU Freq Stability**: 이 값이 0.05 이상이면 호스트 CPU governor가 주파수를 변동시키고 있으므로
> TBT 측정의 신뢰성이 떨어진다. 분석 보고서에 경고 표시.

## 5. 실험 매트릭스

### 설계 원칙: 목적별 토큰 수 분리

KV cache 메모리는 토큰 수에 비례한다. Llama 3.2 1B (kv_heads=8, head_dim=64, layers=16, Q4_0) 기준:

| 토큰 수 | KV cache 크기 | evict 50% 절감 | 실행 시간 (~45ms/tok) | 적합한 측정 |
|---------|--------------|---------------|---------------------|------------|
| 128 | ~4MB | ~2MB | ~6초 | 속도(throttle), 동작 검증 |
| 512 | ~16MB | ~8MB | ~23초 | 품질(eviction 영향) |
| 1024 | ~32MB | ~16MB | ~46초 | 메모리(RSS 변화 관측) |
| 2048 | ~64MB | ~32MB | ~92초 | 메모리(대규모 eviction) |

**규칙**:
- **속도 실험** (Thermal, Compute, Energy): 128 토큰이면 충분. 품질 무관.
- **품질 실험** (Memory eviction): 512 토큰. cache가 충분히 쌓인 후 eviction해야 품질 차이 관측 가능.
- **메모리 실험** (RSS 절감 관측): 1024 토큰. eviction 전후 RSS delta가 수십 MB 수준이어야 의미 있음.
- **신호 주입 위치**: 토큰 수의 **비율**로 결정. 25%, 50%, 75% 지점.

### 공통 조건

- **모델**: Llama 3.2 1B (`/home/go/Workspace/models/llama3.2-1b`)
- **백엔드**: CPU (호스트 x86_64 AVX2)
- **Sampling**: `--greedy` (temperature=0, 재현성 확보)
- **sample-interval**: 1 (매 토큰 시스템 메트릭 수집)
- **top-k logits**: 10

### Round 1: Baseline (5 실험)

각 토큰 수별 baseline 확보. 모든 후속 실험의 비교 기준.

| ID | Eviction | Tokens | 실행 시간 | 측정 목적 |
|----|----------|--------|----------|-----------|
| B-128 | none | 128 | ~6초 | 속도 baseline (짧은) |
| B-512 | none | 512 | ~23초 | 품질 baseline (중간) |
| B-1024 | none | 1024 | ~46초 | 메모리 baseline (긴) |
| B-512-sliding | sliding(1024) | 512 | ~23초 | sliding 오버헤드 |
| B-512-h2o | h2o(0.5/128/0.1) | 512 | ~23초 | h2o 오버헤드 |

### Round 2: 단일 신호 (14 실험)

#### 속도 실험 — Thermal / Compute / Energy (128 토큰)

eviction 없으므로 품질 영향 = 0. 순수 속도 영향만 측정.

| ID | 신호 | 주입 위치 | Eviction | Tokens | 측정 |
|----|------|----------|----------|--------|------|
| T-W-32 | Thermal Warning | 32 (25%) | none | 128 | 약한 throttle |
| T-C-32 | Thermal Critical | 32 (25%) | none | 128 | 강한 throttle |
| T-CR-32-96 | Thermal Crit→Normal | 32→96 (25%→75%) | none | 128 | throttle + 복구 |
| C-32 | Compute(CPU) | 32 (25%) | none | 128 | backend 권장 |
| E-32 | Energy Emergency | 32 (25%) | none | 128 | suspend 동작 |

#### 품질 실험 — Memory eviction (512 토큰)

cache가 충분히 쌓인 후(~256 토큰) eviction. 품질 + 속도 동시 측정.

| ID | 신호 | 주입 위치 | Eviction | Tokens | 측정 |
|----|------|----------|----------|--------|------|
| M-W-256-sl | Memory Warning | 256 (50%) | sliding | 512 | 약한 eviction |
| M-C-256-sl | Memory Critical | 256 (50%) | sliding | 512 | 강한 eviction (sliding) |
| M-C-256-h2o | Memory Critical | 256 (50%) | h2o | 512 | 강한 eviction (h2o) |
| M-CR-256-384 | Memory Crit→Normal | 256→384 (50%→75%) | h2o | 512 | eviction + 복구 |

#### 메모리 실험 — 대규모 eviction (1024 토큰)

RSS 변화가 수십 MB 수준으로 명확히 관측 가능.

| ID | 신호 | 주입 위치 | Eviction | Tokens | 측정 |
|----|------|----------|----------|--------|------|
| R-C-512-sl | Memory Critical | 512 (50%) | sliding | 1024 | RSS delta (sliding) |
| R-C-512-h2o | Memory Critical | 512 (50%) | h2o | 1024 | RSS delta (h2o) |
| R-C-256-h2o | Memory Critical | 256 (25%) | h2o | 1024 | 조기 eviction RSS |
| R-C-768-h2o | Memory Critical | 768 (75%) | h2o | 1024 | 후반 eviction RSS |
| R-CR-512-768 | Memory Crit→Normal | 512→768 | h2o | 1024 | RSS 복구 관측 |

### Round 3: 주입 위치 변수 (8 실험)

동일 신호, 다른 위치. 위치별 품질/메모리 영향 비교.

**품질 관점 (512 토큰, h2o)**:

| ID | 주입 위치 | 비율 | 목적 |
|----|----------|------|------|
| P-128 | 128 | 25% | 초반 eviction (cache 작음 → evict 적음 → 품질 영향 적음?) |
| P-256 | 256 | 50% | 중반 eviction (= Round 2 M-C-256-h2o, 재현성 확인) |
| P-384 | 384 | 75% | 후반 eviction (cache 큼 → evict 많음 → 품질?) |
| P-448 | 448 | 87.5% | 말미 eviction (남은 토큰 적음 → 복구 시간 부족?) |

**메모리 관점 (1024 토큰, h2o)**:

| ID | 주입 위치 | 비율 | 목적 |
|----|----------|------|------|
| RP-256 | 256 | 25% | 조기 eviction → RSS 조기 절감 |
| RP-512 | 512 | 50% | (= Round 2 R-C-512-h2o, 재현성 확인) |
| RP-768 | 768 | 75% | 후반 eviction → RSS 최대 절감 |
| RP-896 | 896 | 87.5% | 말미 eviction → 잠깐의 RSS 절감 |

### Round 4: H2O 파라미터 sweep (9 실험)

Memory Critical at token 256 고정, 512 토큰 생성. h2o 파라미터 3축 변경.

| ID | keep_ratio | recent_window | decay | 목적 |
|----|-----------|---------------|-------|------|
| H-01 | 0.3 | 64 | 0.1 | aggressive, small window |
| H-02 | 0.3 | 128 | 0.1 | aggressive, large window |
| H-03 | 0.3 | 128 | 0.05 | aggressive, slow decay |
| H-04 | 0.5 | 64 | 0.1 | moderate, small window |
| H-05 | 0.5 | 128 | 0.1 | **moderate default** |
| H-06 | 0.5 | 128 | 0.05 | moderate, slow decay |
| H-07 | 0.7 | 64 | 0.1 | conservative, small window |
| H-08 | 0.7 | 128 | 0.1 | conservative, large window |
| H-09 | 0.7 | 128 | 0.05 | conservative, slow decay |

### Round 5: 복합 조건 (8 실험)

| ID | 신호 조합 | Eviction | Tokens | 목적 |
|----|----------|----------|--------|------|
| X-01 | Thermal Crit(256) + Memory Crit(256) | h2o | 512 | 동시 이중 제약 |
| X-02 | Thermal Crit(128) → Memory Crit(256) → Normal(384) | h2o | 512 | 연쇄 제약 |
| X-03 | 반복 Crit(200)→Norm(300)→Crit(400)→Norm(500) | h2o | 512 (evict_window=256) | 반복 eviction |
| X-04 | 5× Memory Crit 연속 (200,220,240,260,280) | h2o | 512 | 신호 폭풍 |
| X-05 | Memory Critical at 512 | h2o | 2048 | 초대형 시퀀스 |
| X-06 | Memory Critical at 1024 | h2o | 2048 | 초대형 후반 eviction |
| X-07 | 반복 Crit→Norm (3회, 512/768/1024) | h2o | 1536 | 대규모 반복 복구 |
| X-08 | Thermal Crit(256) + Memory Crit(512) | h2o | 1024 | 이중 제약 + 메모리 |

**총 44 실험** (Round 1~5). 예상 총 실행 시간: ~35분.

## 6. 분석 도구

### 6.1 compare.py — 개별 실험 비교

```bash
python experiments/analysis/compare.py \
  --baseline experiments/results/B-512.jsonl \
  --experiment experiments/results/M-C-256-h2o.jsonl \
  --output experiments/reports/M-C-256-h2o_report.md
```

출력 예시:

```
=====================================================================
  M-C-256-h2o vs B-512
  MemoryPressure(Critical) at token 256/512  |  h2o (0.5/128/0.1)
=====================================================================

  -- Speed -------------------------------------------------------
  Avg TBT:           45.2ms -> 48.7ms   (+7.7%)
  Avg Forward:       45.2ms -> 46.1ms   (+2.0%)
  Throttle:          0ms total (0.0% of wall time)
  Throughput:        22.1 -> 20.5 t/s   (-7.2%)
  Recovery Latency:  N/A (no Normal signal)

  -- Quality ------------------------------------------------------
  First Divergent Token:   258 / 512
  Exact Match Rate:        0.831  (425/512)
  Suffix EMR (post-FDT):   0.654  (166/254)
  ROUGE-L F1:              0.889
  BLEU-4:                  0.752
  Top-K Overlap (avg):     0.85
  Top-K Overlap (pre-FDT): 0.97
  Top-K Overlap (post-FDT):0.73

  -- Resources ----------------------------------------------------
  Evictions:          1 (128 tokens removed)
  Cache Utilization:  0.188  (384/2048)
  RSS:                2020MB -> 2068MB  (peak: 2080MB, post-evict: 2052MB)
  RSS Saved:          -16MB after eviction
  CPU Freq Stability: 0.002  (OK)
  CPU Util (avg):     93.8%
  CPU Util at Evict:  98.2%  (+4.4% spike)
  Thermal Rise:       +3200 m°C

=====================================================================
```

### 6.2 round_report.py — Round 전체 요약

```bash
python experiments/analysis/round_report.py --round 1
```

출력 예시:

```
Round 2 Summary — Single Signal (2026-03-07)
==========================================================================================================================================
 ID              Signal             Evict    Tokens  TBT +%  FWD +%  EMR    FDT   ROUGE-L  RSS Peak  RSS Saved  Evictions  CPU CV
==========================================================================================================================================
 Speed (baseline: B-128)
 T-W-32          Therm.Warn@32      none     128     +5.3%  +0.0%  1.000   128   1.000    2020MB    -          0          0.002
 T-C-32          Therm.Crit@32      none     128    +15.6%  +0.0%  1.000   128   1.000    2020MB    -          0          0.002
 T-CR-32-96      Therm.C->N@32-96   none     128     +8.2%  +0.0%  1.000   128   1.000    2020MB    -          0          0.002
 C-32            Compute@32         none     128     +0.0%  +0.0%  1.000   128   1.000    2020MB    -          0          0.002
 E-32            Energy.Emrg@32     none     128      N/A    N/A    N/A     32    N/A      2020MB    -          0          0.002

 Quality (baseline: B-512)
 M-W-256-sl      Mem.Warn@256       sliding  512     +3.1%  +1.8%  0.922   260   0.941    2048MB    -8MB       1          0.003
 M-C-256-sl      Mem.Crit@256       sliding  512     +5.7%  +3.2%  0.812   258   0.872    2048MB    -12MB      1          0.003
 M-C-256-h2o     Mem.Crit@256       h2o      512     +4.3%  +2.0%  0.873   260   0.904    2046MB    -10MB      1          0.003
 M-CR-256-384    Mem.C->N@256-384   h2o      512     +3.8%  +1.5%  0.891   260   0.918    2046MB    -10MB      1          0.002

 Memory (baseline: B-1024)
 R-C-512-sl      Mem.Crit@512       sliding  1024    +4.8%  +2.5%  0.785   514   0.841    2080MB    -16MB      1          0.003
 R-C-512-h2o     Mem.Crit@512       h2o      1024    +3.9%  +1.8%  0.847   516   0.889    2076MB    -14MB      1          0.003
 R-C-256-h2o     Mem.Crit@256       h2o      1024    +2.1%  +1.2%  0.901   258   0.928    2044MB    -8MB       1          0.002
 R-C-768-h2o     Mem.Crit@768       h2o      1024    +5.2%  +2.8%  0.812   770   0.861    2092MB    -22MB      1          0.003
 R-CR-512-768    Mem.C->N@512-768   h2o      1024    +3.5%  +1.5%  0.865   514   0.901    2076MB    -14MB      1          0.002
==========================================================================================================================================

Key Observations:
 - Thermal: speed only (TBT +5~16%), zero quality impact, zero RSS impact
 - Memory eviction: H2O > Sliding on quality (EMR 0.87 vs 0.81 at 512 tok)
 - RSS Saved: -8~22MB proportional to cache size at eviction point
 - Later eviction = more tokens removed = larger RSS save but worse quality
 - CPU spike at eviction: +4~5% momentary, returns to baseline within 2 tokens
```

### 6.3 plot_tbt_timeline.py — TBT 시계열 그래프

```bash
python experiments/analysis/plot_tbt_timeline.py \
  --baseline experiments/results/B-01.jsonl \
  --experiments experiments/results/T-02.jsonl experiments/results/M-03.jsonl \
  --output experiments/reports/plots/round1_tbt.png
```

그래프 내용:
- X축: 토큰 위치, Y축: TBT (ms)
- baseline을 회색 밴드로, 실험을 컬러 라인으로
- 신호 주입 시점을 수직선으로 표시
- eviction 이벤트를 마커로 표시

### 6.4 quality_metrics.py — 품질 메트릭 라이브러리

```python
def compute_fdt(baseline_tokens, experiment_tokens) -> int
def compute_emr(baseline_tokens, experiment_tokens) -> float
def compute_suffix_emr(baseline_tokens, experiment_tokens, fdt) -> float
def compute_rouge_l(baseline_text, experiment_text) -> dict  # P, R, F1
def compute_bleu4(baseline_text, experiment_text) -> float
def compute_topk_overlap(baseline_logits, experiment_logits, k=10) -> list[float]
```

## 7. 실행 워크플로우

```bash
# 0. 빌드
cargo build --release --bin generate

# 1. Baseline (토큰 수별)
cargo run --release --bin generate -- \
  --model-path /home/go/Workspace/models/llama3.2-1b \
  --prompt "The history of artificial intelligence began in" \
  -n 512 --backend cpu --greedy \
  --experiment-output experiments/results/B-512.jsonl

cargo run --release --bin generate -- \
  --model-path /home/go/Workspace/models/llama3.2-1b \
  --prompt "The history of artificial intelligence began in" \
  -n 1024 --backend cpu --greedy \
  --experiment-output experiments/results/B-1024.jsonl

# 2. 품질 실험 (512 토큰, eviction at 256)
cargo run --release --bin generate -- \
  --model-path /home/go/Workspace/models/llama3.2-1b \
  --prompt "The history of artificial intelligence began in" \
  -n 512 --backend cpu --greedy \
  --eviction-policy h2o --h2o-keep-ratio 0.5 --h2o-recent-window 128 \
  --experiment-schedule experiments/configs/memory_critical_at_256.json \
  --experiment-output experiments/results/M-C-256-h2o.jsonl

# 3. 메모리 실험 (1024 토큰, eviction at 512)
cargo run --release --bin generate -- \
  --model-path /home/go/Workspace/models/llama3.2-1b \
  --prompt "The history of artificial intelligence began in" \
  -n 1024 --backend cpu --greedy \
  --eviction-policy h2o --h2o-keep-ratio 0.5 --h2o-recent-window 128 \
  --experiment-schedule experiments/configs/memory_critical_at_512.json \
  --experiment-output experiments/results/R-C-512-h2o.jsonl

# 4. 개별 비교 (품질)
python experiments/analysis/compare.py \
  --baseline experiments/results/B-512.jsonl \
  --experiment experiments/results/M-C-256-h2o.jsonl \
  --output experiments/reports/M-C-256-h2o_report.md

# 5. 개별 비교 (메모리)
python experiments/analysis/compare.py \
  --baseline experiments/results/B-1024.jsonl \
  --experiment experiments/results/R-C-512-h2o.jsonl \
  --output experiments/reports/R-C-512-h2o_report.md

# 6. Round 전체 분석
python experiments/analysis/round_report.py --round 2

# 7. 시각화
python experiments/analysis/plot_tbt_timeline.py \
  --baseline experiments/results/B-512.jsonl \
  --experiments experiments/results/M-C-256-h2o.jsonl experiments/results/M-C-256-sl.jsonl \
  --output experiments/reports/plots/round2_quality_tbt.png

python experiments/analysis/plot_tbt_timeline.py \
  --baseline experiments/results/B-1024.jsonl \
  --experiments experiments/results/R-C-512-h2o.jsonl experiments/results/R-C-512-sl.jsonl \
  --output experiments/reports/plots/round2_memory_tbt.png
```

## 8. 디렉토리 구조

```
experiments/
├── PLAN.md                         # 이 문서
├── FINDINGS.md                     # 누적 발견사항
├── configs/                        # 신호 스케줄 JSON (토큰 위치 기반)
│   ├── baseline.json               # 빈 스케줄 {"name":"baseline","signals":[]}
│   ├── thermal_warning_32.json     # 속도: 128 토큰용
│   ├── thermal_critical_32.json
│   ├── thermal_crit_32_recover_96.json
│   ├── compute_cpu_32.json
│   ├── energy_emergency_32.json
│   ├── memory_warning_256.json     # 품질: 512 토큰용
│   ├── memory_critical_256_sliding.json
│   ├── memory_critical_256_h2o.json
│   ├── memory_crit_256_recover_384.json
│   ├── memory_critical_512.json    # 메모리: 1024 토큰용
│   ├── memory_critical_256_long.json
│   ├── memory_critical_768.json
│   ├── memory_crit_512_recover_768.json
│   └── ...
├── results/                        # 실험 결과 JSONL
│   ├── B-128.jsonl
│   ├── B-512.jsonl
│   ├── B-1024.jsonl
│   ├── M-C-256-h2o.jsonl
│   ├── R-C-512-h2o.jsonl
│   └── ...
├── reports/                        # 분석 보고서
│   ├── M-C-256-h2o_report.md
│   ├── round2_summary.md
│   └── plots/
│       ├── round2_quality_tbt.png
│       └── round2_memory_rss.png
├── analysis/                       # 분석 스크립트
│   ├── compare.py                  # baseline vs experiment
│   ├── round_report.py             # round 요약
│   ├── plot_tbt_timeline.py        # TBT 시계열 그래프
│   ├── plot_rss_timeline.py        # RSS 시계열 그래프
│   ├── quality_metrics.py          # ROUGE, BLEU, FDT, EMR 라이브러리
│   └── requirements.txt            # rouge-score, nltk, matplotlib, bert-score
└── prompts/                        # 실험용 프롬프트
    ├── factual.txt                 # "The history of artificial intelligence began in"
    ├── creative.txt                # "Once upon a time in a distant galaxy"
    └── technical.txt               # "To implement a binary search tree in Rust"
```

## 9. 관리 체계

### FINDINGS.md 포맷

```markdown
## Round N (날짜)

### 가설
- 가설 1: ...
- 가설 2: ...

### 결과
- [확인/반박] 가설 1: 수치 근거
- [확인/반박] 가설 2: 수치 근거

### 인사이트
- 발견 1
- 발견 2

### 다음 실험 방향
- Round N+1에서 확인할 사항
```

### 반복 사이클

```
 가설 수립
     │
     ▼
 스케줄 JSON 작성 (configs/)
     │
     ▼
 실험 실행 (cargo run ... --experiment-*)
     │
     ▼
 분석 (compare.py / round_report.py / plots)
     │
     ▼
 인사이트 기록 (FINDINGS.md)
     │
     ▼
 다음 Round 설계 ─────────────► 가설 수립
```

## 10. 벤치마크 평가 (Round 10-12)

> 방법론 상세: [docs/30_evaluation_methodology.md](../docs/30_evaluation_methodology.md)
> 프롬프트: [experiments/prompts/](prompts/)

기존 Round 1-9는 Resilience 시스템의 동작 검증에 집중했다.
Round 10-12는 KV cache eviction의 **디코딩 정확도**를 학술적으로 평가한다.

### 평가 프레임워크

3-Tier 체계로 관련 논문 (H2O, StreamingLLM, SnapKV, PyramidKV 등)의
표준 평가 방법론을 채택.

| Tier | 벤치마크 | 참고 논문 | 측정 대상 |
|------|---------|----------|----------|
| 1 | **Perplexity** | StreamingLLM, ScissorHands, Quest | 토큰 예측 분포 품질 |
| 2 | **NIAH** | SnapKV, PyramidKV, MInference | 정보 검색 정확도 |
| 3 | **LongBench-style QA** | SnapKV, PyramidKV, KIVI | 태스크 수행 능력 |

### Round 10: Perplexity 평가

**프롬프트**: PPL-01 ~ PPL-05 (5개 도메인)

| 도메인 | ID | 설명 |
|--------|-----|------|
| Literary | PPL-01 | 문학 작품 스타일 서사 |
| Encyclopedic | PPL-02 | 백과사전/과학 설명문 |
| Technical | PPL-03 | 알고리즘/자료구조 기술 문서 |
| Conversational | PPL-04 | 대화체 일상 설명 |
| News | PPL-05 | 뉴스 기사 보도문 |

**실험 매트릭스** (5 프롬프트 × 3 길이 × 3 정책 = 45 실험):

| 변수 | 값 |
|------|------|
| 생성 길이 | 512, 1024, 2048 토큰 |
| Eviction 정책 | none (baseline), sliding, h2o |
| 신호 주입 | 50% 위치에서 memory_critical |

**메트릭**:
- EMR (기존)
- Top-K Overlap (기존)
- ROUGE-L, BLEU-4 (기존)
- Baseline Token Rank (신규): eviction 실행에서 baseline 토큰의 순위
- Entropy Ratio (신규): eviction 후 logit 엔트로피 변화

### Round 11: Needle-in-a-Haystack (NIAH)

**프롬프트**: assemble_niah.py로 자동 조합

| 변수 | 값 |
|------|------|
| Needle 유형 | passkey, fact, name, date, statistic (5종) |
| Needle 깊이 | 10%, 25%, 50%, 75%, 90% |
| 컨텍스트 길이 | 4 blocks (~256 tok), 6 blocks (~512 tok), 8 blocks (~1024 tok) |
| Eviction 정책 | sliding, h2o |
| 신호 주입 | 생성 1 토큰에서 memory_critical (즉시 eviction) |

**핵심 가설**:
- Sliding window: needle이 window 밖이면 반드시 검색 실패
- H2O: needle의 attention score가 높으면 보존 → 검색 성공 가능
- 깊은 needle(75-90%)은 sliding에서도 보존 (최근 window에 가까움)

**메트릭**: Retrieval Accuracy (binary), Retrieval Score (LCS ratio)

**전체**: 5 needle × 5 depth × 3 length × 2 policy = **150 실험**
(축소 운영: passkey + fact 2종 × 5 depth × 2 length × 2 policy = **40 실험**)

### Round 12: LongBench-style QA

**프롬프트**: benchmark_prompts.json의 qa 섹션

| 카테고리 | 프롬프트 수 | 메트릭 |
|---------|-----------|--------|
| Single-doc QA | 3 | F1, EM |
| Summarization | 2 | ROUGE-L |
| Few-shot | 3 | Accuracy |
| Multi-hop | 2 | F1 |

**실험 매트릭스** (10 프롬프트 × 3 정책 = 30 실험):

| 변수 | 값 |
|------|------|
| Eviction 정책 | none (baseline), sliding, h2o |
| 신호 주입 | 생성 5 토큰에서 memory_critical |

**메트릭**: F1, Exact Match, ROUGE-L

### 베이스라인 비교 (논문 표준)

모든 Round에서 다음 4개 베이스라인을 비교:

| # | 베이스라인 | 설명 |
|---|----------|------|
| 1 | Full KV cache | 압축 없음 (상한선) |
| 2 | Sliding window | 최근 N 토큰만 유지 (SlidingWindowPolicy) |
| 3 | H2O | Heavy hitter + recent (H2OPolicy) |
| 4 | Random eviction | 랜덤 제거 (하한선, ablation용 — 향후 구현) |

## 11. 구현 순서

### Phase A: CLI 기반 실험 인프라 (Step 1~5)

| Step | 내용 | 산출물 | 담당 |
|------|------|--------|------|
| **1** | generate.rs experiment mode | CLI 플래그, 주입 로직, SystemSampler, JSONL 출력 | Rust Dev |
| **2a** | Round 1~2 스케줄 JSON 작성 | `experiments/configs/` ~15개 파일 | Rust Dev |
| **2b** | compare.py + quality_metrics.py + plot 스크립트 | CLI 분석 도구 (matplotlib → PNG) | Frontend Dev |
| **3** | Baseline 실행 (128/512/1024) + JSONL 검증 | `B-128.jsonl`, `B-512.jsonl`, `B-1024.jsonl` | Tester |
| **4** | Round 2 전체 실험 + 분석 | 보고서 + FINDINGS | Tester |
| **5** | Round 3~5 반복 | 점진적 인사이트 누적 | Tester |

Step 2a, 2b는 병렬 진행. **Step 1이 가장 크고 핵심적인 구현.**

### Phase B: 대시보드 연계 (Step 6~7)

Round 2 결과가 축적된 후, 인터랙티브 분석 필요성이 생기면 착수.

| Step | 내용 | 산출물 | 담당 |
|------|------|--------|------|
| **6** | JSONL 파서 + REST API | `dashboard/backend/experiment_parser.py`, `/api/experiments` | Frontend Dev |
| **7** | Dashboard Experiments 탭 | 실험 목록, TBT/RSS 시계열 (Plotly), 비교 모드 | Frontend Dev |

대시보드 탭 기능:
- 실험 목록 테이블 (ID, Signal, Eviction, Tokens, EMR, TBT%, RSS)
- 개별 상세: TBT 시계열 + RSS 시계열 (Plotly.js 인터랙티브)
- 비교 모드: baseline vs experiment 겹쳐 보기 (기존 Compare 탭 패턴 활용)
- 신호 주입 시점 수직선 + eviction 이벤트 마커

각 Step 완료 시 커밋.
