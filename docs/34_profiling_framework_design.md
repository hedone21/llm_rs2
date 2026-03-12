# 34. Inference Profiling Framework 설계

> **Status**: Draft
> **Author**: Architect
> **Date**: 2026-03-12

## 1. 동기 및 목표

현재 프로파일링은 `OpProfiler`(연산별 타이밍)와 외부 스크립트(`android_profile.py`)로 분산되어 있다.
연구자가 H2O 캐시 정책의 내부 동작, 어텐션 스코어 분포, 에빅션 이벤트 등을 **한 번의 추론 실행으로 수집하고 시각화**할 수 있는 통합 프로파일링 프레임워크가 필요하다.

### 설계 원칙

| 원칙 | 설명 |
|------|------|
| **Zero-cost when off** | `Option<InferenceProfiler>` — None이면 분기 예측으로 완전 스킵 |
| **모듈 기반 확장** | 새 프로브 = 새 파일 + `InferenceProfiler` 필드 추가 |
| **단일 출력** | 모든 프로브 데이터를 하나의 JSON 파일로 통합 export |
| **시각화 분리** | Rust는 데이터 수집만, 시각화는 Python 스크립트 |

## 2. 아키텍처

### 2.1 모듈 구조

```
engine/src/profile/
├── mod.rs              # InferenceProfiler, ProfileConfig, JSON export
├── ops.rs              # OpProfiler (기존 llama_layer.rs에서 이동)
├── scores.rs           # ScoreTracker — H2O 어텐션 스코어 스냅샷
├── latency.rs          # LatencyTracker — 토큰별 디코드 지연
├── entropy.rs          # EntropyTracker — 어텐션 엔트로피
└── cache.rs            # CacheTracker — KV 캐시 사용률

scripts/
└── visualize_attention.py  # 히트맵 + 파티션 시각화
```

### 2.2 핵심 구조체

```rust
/// 통합 프로파일러 — 모든 프로브를 포함
pub struct InferenceProfiler {
    pub ops: OpProfiler,              // 연산별 타이밍
    pub scores: ScoreTracker,         // H2O 어텐션 스코어 히스토리
    pub latency: LatencyTracker,      // 토큰별 지연
    pub entropy: Option<EntropyTracker>,  // 어텐션 엔트로피 (선택)
    pub cache: Option<CacheTracker>,      // 캐시 사용률 (선택)
    config: ProfileConfig,
}

pub struct ProfileConfig {
    /// 스코어 스냅샷 간격 (1 = 매 스텝, 10 = 10스텝마다)
    pub score_snapshot_interval: usize,
    /// per-head 스코어 추적 (H2O+ 분석용)
    pub track_per_head: bool,
    /// 활성화할 프로브 목록
    pub enabled_probes: Vec<String>,
    /// 출력 디렉토리
    pub output_dir: PathBuf,
}
```

### 2.3 사용 흐름

```
CLI: --profile [--profile-dir ./results] [--profile-interval 1] [--profile-probes scores,entropy]
         │
         ▼
  InferenceProfiler::new(ProfileConfig)
         │
    ┌────┴────────────────────────────┐
    │ Decode Loop (token-by-token)    │
    │                                 │
    │  profiler.on_step_begin(step)   │
    │       │                         │
    │  model.forward_into(...)        │
    │       │                         │
    │  profiler.on_step_end(step,     │
    │       token_id,                 │
    │       &score_accumulator,       │
    │       forward_ms)               │
    │       │                         │
    │  [eviction 발생 시]             │
    │  profiler.on_eviction(...)      │
    └─────────────────────────────────┘
         │
  profiler.export_json(path)?
         │
  python scripts/visualize_attention.py results/profile_*.json
```

## 3. 프로브 상세 설계

### 3.1 ScoreTracker — H2O 어텐션 스코어 (핵심)

**목적**: 디코드 과정에서 각 캐시 포지션의 어텐션 중요도가 어떻게 변화하는지 시계열로 기록한다.

```rust
pub struct ScoreTracker {
    snapshots: Vec<ScoreSnapshot>,
    evictions: Vec<EvictionEvent>,
    snapshot_interval: usize,
    track_per_head: bool,
}

pub struct ScoreSnapshot {
    pub step: usize,               // 디코드 스텝 번호
    pub token_id: u32,             // 이 스텝에서 생성된 토큰
    pub cache_len: usize,          // 현재 캐시 길이
    pub importance: Vec<f32>,      // [cache_len] 포지션별 중요도
    pub head_importance: Option<Vec<f32>>,  // [n_kv_heads * cache_len] (선택)
}

pub struct EvictionEvent {
    pub step: usize,               // 발생 스텝
    pub policy: String,            // "h2o", "h2o_plus", "d2o"
    pub before_len: usize,         // 에빅션 전 캐시 길이
    pub after_len: usize,          // 에빅션 후 캐시 길이
    pub kept_positions: Vec<usize>,    // 유지된 포지션 인덱스
    pub evicted_count: usize,      // 제거된 토큰 수
    pub partition: PartitionInfo,  // 에빅션 시점의 파티션 정보
}

pub struct PartitionInfo {
    pub prefix_end: usize,         // [0..prefix_end) = Protected Prefix
    pub hh_count: usize,           // Heavy Hitter 개수
    pub recent_start: usize,       // [recent_start..) = Recent Window
}
```

**데이터 수집 시점**:
- `on_step_end()`: `step % snapshot_interval == 0`일 때 `AttentionScoreAccumulator`에서 현재 importance를 clone
- `on_eviction()`: `CacheManager::maybe_evict_with_scores()` 호출 전후의 캐시 상태 기록

**메모리 추정** (max_seq_len=2048, 500 토큰 생성, interval=1):
- 스냅샷당: ~8KB (2048 * 4 bytes)
- 총: ~4MB (500 스냅샷) — 연구용으로 충분히 합리적

### 3.2 LatencyTracker — 토큰별 지연

```rust
pub struct LatencyTracker {
    records: Vec<LatencyRecord>,
}

pub struct LatencyRecord {
    pub step: usize,
    pub forward_us: u64,      // forward pass 소요 시간
    pub sample_us: u64,       // 샘플링 소요 시간
    pub total_us: u64,        // 전체 (forward + sample + overhead)
    pub cache_len: usize,     // 어텐션 대상 시퀀스 길이
}
```

**목적**: cache_len 증가에 따른 어텐션 연산 비용 변화, 에빅션 후 성능 회복 등을 분석.

### 3.3 OpProfiler (기존, 이동)

현재 `llama_layer.rs`에 있는 `OpProfiler`를 `profile/ops.rs`로 이동하고, 기존 위치에서 re-export.

변경 사항 없이 위치만 정리. `print_report()` 외에 `to_json()` 메서드 추가.

### 3.4 EntropyTracker — 어텐션 엔트로피 (제안 항목)

```rust
pub struct EntropyTracker {
    /// [step][layer] = 해당 레이어의 평균 어텐션 엔트로피
    records: Vec<Vec<f32>>,
}
```

**어텐션 엔트로피**: `H = -Σ p(t) * log2(p(t))` (post-softmax 확률 분포)
- **낮은 엔트로피**: 소수 토큰에 집중 → Heavy Hitter가 명확
- **높은 엔트로피**: 분산된 어텐션 → 에빅션이 위험

**수집 방법**: `accumulate_layer_gqa()` 호출 시 scores 버퍼에서 per-head 엔트로피 계산.
단, 엔트로피 계산의 log 연산 비용이 있으므로 선택적 활성화.

### 3.5 CacheTracker — KV 캐시 사용률 (제안 항목)

```rust
pub struct CacheTracker {
    records: Vec<CacheRecord>,
}

pub struct CacheRecord {
    pub step: usize,
    pub capacity: usize,       // 물리 버퍼 크기
    pub current_pos: usize,    // 현재 사용 중인 위치
    pub utilization: f32,      // current_pos / capacity
    pub memory_bytes: usize,   // 실제 메모리 사용량
}
```

## 4. H2O 어텐션 스코어 시각화

### 4.1 히트맵 (핵심 시각화)

```
           Decode Step →
     0   10   20   30   40   50  ...
   ┌────────────────────────────────┐
 0 │████ ████ ████ ████ ████ ████  │ ← Protected Prefix (항상 높음)
 1 │████ ████ ████ ████ ████ ████  │
 2 │████ ████ ████ ████ ████ ████  │
   │---  ---  ---  ---  ---  ---   │ ← prefix 경계
 3 │░░░░ ░░░░ ██░░ ████ ████ ████  │ ← Heavy Hitter (점수 상승)
 4 │░░░░ ████ ████ ░░░░ ░░░░       │ ← 에빅션됨
 5 │░░░░ ░░░░ ░░░░ ░░░░ ░░░░ ░░░░  │
   │                    ┊ eviction  │
 ..│          ...                   │
 N │               ░░░░ ░░░░ ████  │ ← Recent Window
   └────────────────────────────────┘

   색상: ■ 높은 중요도 → ░ 낮은 중요도
   │ 에빅션 이벤트 (수직 점선)
   ─ 파티션 경계 (수평 점선)
```

**X축**: Decode step (시간)
**Y축**: Cache position (토큰)
**색상**: Importance score (viridis colormap, log scale)
**오버레이**:
- 수직 점선: 에빅션 이벤트 발생 시점
- 수평 점선: Prefix / HH / Recent 파티션 경계
- 에빅션 후 캐시 포지션 재매핑 표시

### 4.2 파티션 진화 차트

```
  Cache
  Size
   ▲
   │  ┌──────────────────────────────────┐
   │  │         Recent Window            │
   │  │                                  │
   │  ├──────────────────────────────────┤ ← 에빅션
   │  │       Heavy Hitters              │
   │  │                                  │
   │  ├──────────────────────────────────┤
   │  │     Protected Prefix             │
   │  └──────────────────────────────────┘
   └──────────────────────────────────────► Step
```

Stacked area chart: 각 파티션이 캐시에서 차지하는 비율 변화.

### 4.3 시각화 스크립트

`scripts/visualize_attention.py`:

```
사용법:
  python scripts/visualize_attention.py results/profile_h2o_*.json
  python scripts/visualize_attention.py results/profile_h2o_*.json --per-head
  python scripts/visualize_attention.py results/profile_h2o_*.json --output plots/

출력:
  1. attention_heatmap.png       — 전체 어텐션 스코어 히트맵
  2. partition_evolution.png     — 파티션 비율 변화 (stacked area)
  3. eviction_detail.png         — 에빅션 전후 스코어 비교
  4. head_heatmap_{h}.png        — per-head 히트맵 (--per-head 옵션)
```

## 5. JSON 출력 형식

```jsonc
{
  "metadata": {
    "model": "Llama-3.2-1B",
    "backend": "cpu",
    "eviction_policy": "h2o",
    "max_seq_len": 2048,
    "prompt_len": 128,
    "generated_tokens": 500,
    "timestamp": "2026-03-12T14:30:00Z"
  },
  "ops": {
    "count": 8000,  // 16 layers * 500 tokens
    "breakdown": {
      "matmul_qkv": { "total_us": 124500, "avg_us": 15, "pct": 18.5 },
      // ...
    }
  },
  "latency": {
    "records": [
      { "step": 0, "forward_us": 5200, "cache_len": 128 },
      { "step": 1, "forward_us": 5250, "cache_len": 129 },
      // ...
    ]
  },
  "scores": {
    "snapshot_interval": 1,
    "n_kv_heads": 8,
    "max_seq_len": 2048,
    "snapshots": [
      {
        "step": 0,
        "token_id": 1234,
        "cache_len": 128,
        "importance": [0.82, 0.15, 0.03, ...],  // [cache_len]
        "head_importance": [[0.9, 0.1, ...], ...]  // [n_kv_heads][cache_len] (선택)
      }
    ],
    "evictions": [
      {
        "step": 120,
        "policy": "h2o",
        "before_len": 256,
        "after_len": 200,
        "evicted_count": 56,
        "partition": { "prefix_end": 4, "hh_count": 98, "recent_start": 102 }
      }
    ]
  },
  "entropy": {  // 선택적
    "records": [
      { "step": 0, "per_layer": [3.2, 2.8, 4.1, ...] }
    ]
  },
  "cache": {  // 선택적
    "records": [
      { "step": 0, "capacity": 512, "current_pos": 128, "utilization": 0.25 }
    ]
  }
}
```

## 6. generate.rs 통합

### 6.1 CLI 인터페이스

```
--profile                    프로파일링 활성화 (기본 프로브: ops, latency, scores)
--profile-dir <DIR>          출력 디렉토리 (기본: results/profile/)
--profile-interval <N>       스코어 스냅샷 간격 (기본: 1)
--profile-probes <LIST>      활성화할 프로브 (쉼표 구분: ops,scores,latency,entropy,cache)
--profile-per-head           per-head 스코어 추적 (H2O+ 분석용)
```

### 6.2 통합 코드 (개념)

```rust
// 생성
let profiler = if args.profile {
    Some(InferenceProfiler::new(ProfileConfig {
        score_snapshot_interval: args.profile_interval,
        track_per_head: args.profile_per_head,
        enabled_probes: args.profile_probes.clone(),
        output_dir: args.profile_dir.clone(),
    }))
} else {
    None
};

// 디코드 루프
for step in 0..num_tokens {
    if let Some(ref mut p) = profiler { p.on_step_begin(step); }

    let t0 = Instant::now();
    model.forward_into(LlamaModelForwardArgs {
        profiler: profiler.as_mut().map(|p| &mut p.ops),  // OpProfiler 전달
        score_accumulator: score_accumulator.as_mut(),
        ..
    })?;
    let forward_us = t0.elapsed().as_micros() as u64;

    score_accumulator.as_mut().map(|acc| acc.end_step());

    if let Some(ref mut p) = profiler {
        p.on_step_end(step, token_id, &score_accumulator, forward_us);
    }

    // 에빅션 발생 시
    if eviction_happened {
        if let Some(ref mut p) = profiler {
            p.on_eviction(EvictionEvent { step, .. });
        }
    }
}

// 완료 후 export
if let Some(ref profiler) = profiler {
    profiler.export_json()?;
    profiler.ops.print_report();  // stderr 출력도 유지
}
```

### 6.3 제로 오버헤드 보장

| 상태 | 비용 |
|------|------|
| `--profile` 미지정 | `Option::None` 체크 = 단일 분기 (branch predicted, ~0ns) |
| `--profile` 지정 | 스냅샷 clone + Vec push = ~1-5μs/step (forward ~5ms 대비 0.1%) |
| per-head 활성화 | 추가 clone = ~10μs/step (여전히 0.2% 미만) |

기존 `Option<OpProfiler>` 패턴과 동일. Cargo feature flag 불필요 — 런타임 분기만으로 충분.

## 7. 새 프로브 추가 절차 (확장 가이드)

1. `engine/src/profile/` 에 새 파일 생성 (e.g., `my_probe.rs`)
2. Tracker 구조체 정의 + `to_json() -> serde_json::Value` 메서드 구현
3. `mod.rs`의 `InferenceProfiler`에 `pub my_probe: Option<MyProbeTracker>` 필드 추가
4. `on_step_begin/end()` 또는 전용 hook에서 데이터 수집 로직 추가
5. `export_json()`에서 `my_probe` 섹션 추가
6. CLI에 `--profile-probes` 목록에 이름 등록
7. (선택) `visualize_attention.py`에 시각화 코드 추가

**규칙**: 각 프로브는 독립적. 다른 프로브에 의존하지 않는다.

## 8. 추가 프로파일링 항목 제안

### 8.1 높은 연구 가치 (권장)

| 항목 | 설명 | 인사이트 |
|------|------|----------|
| **Attention Entropy** | `H = -Σ p·log₂(p)` per head/layer | 어텐션 집중도. 낮으면 HH 명확, 높으면 에빅션 위험 |
| **Layer-wise Activation Norms** | 각 레이어 출력의 L2 norm | 레이어 기여도 분석, 프루닝 후보 식별 |
| **Eviction Quality Score** | 에빅션 전후 어텐션 패턴 변화율 | 에빅션 정책 효과 정량화 (KL divergence) |
| **Token Throughput Curve** | 시간에 따른 tok/s 변화 (rolling avg) | 서멀 쓰로틀링 감지, 캐시 크기 영향 분석 |

### 8.2 보통 연구 가치

| 항목 | 설명 | 인사이트 |
|------|------|----------|
| **Quantization Error** | Q4_0 KV 캐시 양자화 오차 추적 | 정밀도 vs 메모리 트레이드오프 검증 |
| **Heavy Hitter Stability** | HH 토큰의 스텝별 변화율 | HH 집합이 안정적인지 → 에빅션 빈도 결정 근거 |
| **Softmax Temperature** | 어텐션 스코어의 최대값 분포 | 수치 안정성 모니터링 |
| **Memory Pressure Timeline** | 시스템 메모리 + 캐시 메모리 시계열 | 에빅션 트리거 조건 분석 |

### 8.3 장기적 확장

| 항목 | 설명 |
|------|------|
| **Cross-layer Attention Flow** | 특정 토큰이 레이어를 통과하며 받는 어텐션 변화 |
| **Token Importance Ranking** | 에빅션 후 실제 품질 변화와 importance score의 상관 분석 |
| **D2O Merge Statistics** | 토큰 병합 빈도, 유사도 분포, 병합이 품질에 미치는 영향 |

## 9. 리스크 및 대응 방안

| 리스크 | 발생 가능성 | 영향도 | 대응 방안 |
|--------|------------|--------|-----------|
| **메모리 사용 증가** (장시간 프로파일링 시 스냅샷 누적) | 중 | 중 | `snapshot_interval` 파라미터로 제어. max_seq_len=2048, 1000스텝 = ~8MB (합리적) |
| **프로파일 실행 시 성능 저하** | 낮 | 낮 | 측정 결과 0.1~0.2% 오버헤드. `Vec::clone()` 최적화 가능 (ring buffer) |
| **OpProfiler 이동 시 import 깨짐** | 높 | 낮 | 기존 위치에서 `pub use profile::ops::OpProfiler;` re-export |
| **JSON 파일 크기** (per-head + 긴 시퀀스) | 중 | 낮 | per-head는 opt-in, 압축 export 옵션 제공 |
| **시각화 Python 의존성** | 낮 | 낮 | matplotlib/numpy만 사용 (기존 스크립트와 동일) |
| **에빅션 이벤트 수집 시 캐시 상태 동기화** | 중 | 중 | `CacheManager`에 hook 추가, 에빅션 전후 상태를 원자적으로 캡처 |

### 롤백 전략

- 프로브별 독립 모듈이므로 문제 발생 시 해당 프로브만 비활성화 가능
- `Option<InferenceProfiler>` 패턴이므로 전체 비활성화도 CLI flag 하나로 가능
- OpProfiler re-export로 기존 코드 호환성 유지

## 10. 구현 우선순위

| 순서 | 항목 | 의존성 |
|------|------|--------|
| 1 | `profile/` 모듈 생성 + `InferenceProfiler` 골격 | 없음 |
| 2 | `OpProfiler` 이동 + re-export | #1 |
| 3 | `ScoreTracker` 구현 | #1 |
| 4 | `LatencyTracker` 구현 | #1 |
| 5 | `generate.rs` 통합 + CLI 플래그 | #1-4 |
| 6 | JSON export | #5 |
| 7 | `visualize_attention.py` 히트맵 | #6 |
| 8 | `EntropyTracker` (선택) | #1 |
| 9 | `CacheTracker` (선택) | #1 |
