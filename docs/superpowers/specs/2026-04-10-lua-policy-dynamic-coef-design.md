# LuaPolicy Dynamic Coefficient Design

> **목적**: LuaPolicy가 하드코딩된 계수 대신 런타임에 EWMA로 학습되는 동적 계수를 사용하도록 한다.
> Rust가 메트릭 수집 + pressure 계산 + trigger 판정 + relief 학습을 담당하고,
> Lua는 이 데이터를 `ctx.coef`로 받아 액션 결정만 수행한다.
>
> **배경**: 논문 C3 §3.5 + §4.8. Manager가 기기/워크로드별 수동 튜닝 없이 자동 적응.
>
> **관련 문서**: `../papers/pact2026/plan/impl-request-v23.md`

---

## 1. 결정 사항 요약

| 항목 | 결정 |
|------|------|
| 정책 로직 주체 | LuaPolicy (HierarchicalPolicy deprecated) |
| 메트릭 소스 | SystemSignal (Monitor 스레드가 수집) |
| TBT 계산 | `1000.0 / throughput` 역산 (Engine 수정 없음) |
| PI Controller | deprecated (`#[cfg(feature = "hierarchical")]`) |
| HierarchicalPolicy | deprecated (동일 feature flag) |
| Relief 학습 | Rust 쪽 EWMA (Lua는 읽기만) |
| Relief 영속화 | 초기 구현에 포함 (JSON save/load) |
| Relief 초기값 | config 파일에서 설정 가능 |
| Main app trigger | 초기 구현 제외 (TBT/mem/temp 3개로 시작) |
| Deprecated 테스트 | `#[cfg(feature = "hierarchical")]` 뒤로 이동, 코드 보존 |

---

## 2. 아키텍처

```
Monitor Thread(s)
    │ SystemSignal
    ▼
┌─────────────────────── LuaPolicy ───────────────────────┐
│                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────┐ │
│  │ SignalState   │   │ TriggerEngine│   │ EwmaRelief  │ │
│  │              │   │              │   │ Table        │ │
│  │ 최근 Signal  │──▶│ TBT tracker  │   │             │ │
│  │ 캐시 (4종)   │   │ mem/temp 판정│   │ per-action  │ │
│  │ + pressure   │   │ hysteresis   │   │ 6D relief   │ │
│  │ 계산 (6D)    │   │              │   │ EWMA 학습   │ │
│  └──────────────┘   └──────┬───────┘   │ save/load   │ │
│                            │           └──────┬──────┘ │
│                            ▼                  │        │
│                     ┌─────────────┐           │        │
│                     │ build_ctx() │◀──────────┘        │
│                     │             │                     │
│                     │ ctx.engine  │ ← EngineStatus     │
│                     │ ctx.active  │ ← active actions   │
│                     │ ctx.signal  │ ← 최근 Signal 원시값│
│                     │ ctx.coef    │                     │
│                     │  .relief    │ ← EWMA 학습값      │
│                     │  .pressure  │ ← 6D 정규화 압력   │
│                     │  .trigger   │ ← 3개 trigger 판정 │
│                     └──────┬──────┘                     │
│                            │                           │
│                            ▼                           │
│                     Lua decide(ctx)                    │
│                            │                           │
│                            ▼                           │
│                     Vec<EngineCommand>                  │
│                            │                           │
│                     ┌──────┴──────┐                    │
│                     │ Observation │                     │
│                     │ before/after│                     │
│                     │ → EWMA 갱신 │                     │
│                     └─────────────┘                    │
└──────────────────────────────────────────────────────────┘
```

### 데이터 흐름

1. Monitor 스레드가 SystemSignal을 mpsc로 전송
2. main loop가 `policy.process_signal(signal)` 호출
3. LuaPolicy 내부:
   - SignalState: signal 캐시 + 6D pressure 계산
   - TriggerEngine: trigger 판정 (hysteresis 포함)
   - build_ctx(): ctx 테이블 조립 (engine + active + signal + coef)
   - Lua decide(ctx) 호출 → Vec<EngineCommand>
4. 액션 실행 후 Observation이 before/after pressure 비교 → EwmaReliefTable 갱신

---

## 3. Deprecated 처리

`hierarchical` feature flag 뒤로 이동하는 대상:

| 파일 | 내용 |
|------|------|
| `pipeline.rs` | `HierarchicalPolicy` impl 부분 |
| `pi_controller.rs` | 전체 |
| `supervisory.rs` | 전체 |
| `selector.rs` | 전체 |
| `action_registry.rs` | 전체 |
| `evaluator.rs` | 전체 |
| `relief/linear.rs` | RLS 구현체 |
| `types.rs` | `PressureVector`, `FeatureVector`, `ActionMeta` 등 (HierarchicalPolicy 전용 타입) |
| 관련 테스트 전부 | `#[cfg(test)]` 내에서 `#[cfg(feature)]` 중첩 |

**유지 대상**: `PolicyStrategy` trait, `LuaPolicy`, `OperatingMode`.

`relief/mod.rs`의 `ReliefEstimator` trait도 deprecated 처리한다. `EwmaReliefTable`은 이 trait을 구현하지 않고 `lua_policy.rs` 내부의 독립 구조체로 존재한다 (EWMA는 FeatureVector 입력이 필요 없으므로 trait 인터페이스가 맞지 않음).

`Cargo.toml`에서 `hierarchical`은 default features에 포함하지 않는다.

---

## 4. LuaPolicy 내부 컴포넌트

### 4.1 SignalState

최근 수신된 SystemSignal을 variant별로 1개씩 캐시하고, 6D pressure를 계산한다.

```rust
struct SignalState {
    memory: Option<MemorySnapshot>,     // available_bytes, total_bytes
    compute: Option<ComputeSnapshot>,   // cpu_usage_pct, gpu_usage_pct
    thermal: Option<ThermalSnapshot>,   // temperature_mc, throttling
    energy: Option<EnergySnapshot>,     // level
}

struct MemorySnapshot {
    available_bytes: u64,
    total_bytes: u64,
}

struct ComputeSnapshot {
    cpu_usage_pct: f64,
    gpu_usage_pct: f64,
}

struct ThermalSnapshot {
    temperature_mc: i32,
    throttling_active: bool,
}

struct EnergySnapshot {
    level: Level,
}
```

**6D Pressure 계산**:

```rust
struct Pressure6D {
    gpu: f32,       // gpu_usage_pct / 100.0
    cpu: f32,       // cpu_usage_pct / 100.0
    memory: f32,    // 1.0 - (available / total)
    thermal: f32,   // (temp_mc/1000 - TEMP_SAFE) / (TEMP_CRITICAL - TEMP_SAFE), clamped 0~1
    latency: f32,   // tbt_degradation_ratio (TriggerEngine에서 가져옴)
    main_app: f32,  // 0.0 (초기 구현 미사용)
}
```

thermal 정규화 상수 (`TEMP_SAFE`, `TEMP_CRITICAL`)는 config에서 설정 가능.

### 4.2 TriggerEngine

3개 독립 trigger + hysteresis.

```rust
struct TriggerEngine {
    config: TriggerConfig,
    tbt_tracker: TbtTracker,
    state: TriggerState,
}

struct TriggerConfig {
    tbt_enter: f64,     // default 0.30
    tbt_exit: f64,      // default 0.10
    mem_enter: f64,     // default 0.80
    mem_exit: f64,      // default 0.60
    temp_enter: f64,    // default 0.70
    temp_exit: f64,     // default 0.50
}

struct TbtTracker {
    ewma: f64,
    baseline: Option<f64>,
    warmup_count: u32,
    warmup_target: u32,     // default 20
}

struct TriggerState {
    tbt_degraded: bool,
    mem_low: bool,
    temp_high: bool,
}
```

**TBT 계산**: `1000.0 / throughput`. `throughput == 0`이면 (idle/prefill) trigger 판정 건너뜀.

**Hysteresis**: 각 trigger에 enter/exit threshold 분리. 예: `tbt_degraded`가 false → degradation_ratio > tbt_enter → true. true인 상태에서 degradation_ratio < tbt_exit → false.

### 4.3 EwmaReliefTable

Per-action 6D EWMA 학습 + 영속화.

```rust
struct EwmaReliefTable {
    entries: HashMap<String, ReliefEntry>,
    alpha: f32,     // default 0.875 (Jacobson TCP RTT)
}

struct ReliefEntry {
    relief: [f32; 6],       // [gpu, cpu, memory, thermal, latency, main_app]
    observation_count: u32,
}
```

**API**:

- `predict(action: &str) -> [f32; 6]`: 학습값 반환. count==0이면 config의 default 값.
- `observe(action: &str, observed: [f32; 6])`: 첫 관측은 직접 대입, 이후 EWMA 갱신.
- `save(path) / load(path)`: JSON serde.

**초기값**: config 파일(`policy_config.toml`)의 `[adaptation.default_relief]` 섹션에서 액션별 지정. 지정되지 않은 액션은 하드코딩 fallback (`[0.0; 6]`).

### 4.4 Observation

액션 실행 전후의 pressure 변화를 측정하여 EWMA를 갱신한다.

```rust
struct ObservationContext {
    action: String,              // 단일 액션일 때만 기록
    before: Pressure6D,
    timestamp: Instant,
}

const OBSERVATION_DELAY_SECS: f64 = 3.0;
```

- 액션 실행 직전: `before` 스냅샷 저장
- 다음 `process_signal()` 호출 시: 3초 경과했으면 `after` 스냅샷 → `observed_relief = before - after` → `ewma_table.observe()`
- 복수 액션 동시 적용: observation 건너뜀 (로깅만)

---

## 5. Lua ctx 구조

```lua
ctx = {
  engine = {                    -- EngineStatus (기존 유지)
    device = "opencl",
    throughput = 12.5,
    kv_util = 0.73,
    cache_tokens = 1496,
    state = "running",
    phase = "decode",
    partition_ratio = 0.0,
    -- ...
  },

  active = {"kv_evict_h2o"},    -- 활성 액션 (기존 유지)

  signal = {                    -- SystemSignal 원시값 (신규)
    memory = { available = 1048576, total = 8388608 },
    compute = { cpu_pct = 45.2, gpu_pct = 82.1 },
    thermal = { temp_c = 41.3, throttling = false },
  },

  coef = {                      -- 동적 계수 (신규)
    pressure = {
      gpu = 0.82, cpu = 0.45, memory = 0.25,
      thermal = 0.41, latency = 0.35, main_app = 0.0,
    },
    trigger = {
      tbt_degraded = true,
      mem_low = false,
      temp_high = false,
    },
    relief = {
      switch_hw     = { gpu=0.5, cpu=-0.3, mem=0.0, therm=0.3, lat=-0.1, qos=0.0 },
      kv_evict_h2o  = { gpu=0.1, cpu=0.0,  mem=0.4, therm=0.1, lat=0.0,  qos=0.0 },
      throttle      = { gpu=0.0, cpu=0.3,  mem=0.0, therm=0.2, lat=-0.2, qos=0.0 },
      layer_skip    = { gpu=0.2, cpu=0.1,  mem=0.0, therm=0.1, lat=-0.1, qos=0.0 },
      -- ...
    },
  },
}
```

---

## 6. Config 변경

`policy_config.toml`에 추가:

```toml
[adaptation]
ewma_alpha = 0.875
relief_table_path = "relief_table.json"

# Thermal normalization
temp_safe_c = 35.0
temp_critical_c = 50.0

[adaptation.trigger]
tbt_enter = 0.30
tbt_exit = 0.10
tbt_warmup_tokens = 20
mem_enter = 0.80
mem_exit = 0.60
temp_enter = 0.70
temp_exit = 0.50

[adaptation.default_relief]
# 각 액션의 초기 relief 값 [gpu, cpu, memory, thermal, latency, main_app_qos]
# 지정하지 않은 액션은 [0, 0, 0, 0, 0, 0]으로 시작
switch_hw = [0.5, -0.3, 0.0, 0.3, -0.1, 0.0]
kv_evict_h2o = [0.1, 0.0, 0.4, 0.1, 0.0, 0.0]
kv_evict_sliding = [0.1, 0.0, 0.3, 0.1, 0.0, 0.0]
throttle = [0.0, 0.3, 0.0, 0.2, -0.2, 0.0]
set_target_tbt = [0.0, 0.2, 0.0, 0.1, -0.1, 0.0]
layer_skip = [0.2, 0.1, 0.0, 0.1, -0.1, 0.0]
kv_quant_dynamic = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
kv_merge_d2o = [0.1, 0.0, 0.3, 0.1, 0.0, 0.0]
set_partition_ratio = [0.3, -0.2, 0.0, 0.1, 0.0, 0.0]
```

---

## 7. 영향 범위

| 파일 | 변경 | 유형 |
|------|------|------|
| `manager/Cargo.toml` | `hierarchical` feature flag 추가, default에서 제외 | 수정 |
| `manager/src/lib.rs` | deprecated 모듈에 `#[cfg(feature)]` | 수정 |
| `manager/src/pipeline.rs` | `HierarchicalPolicy`에 `#[cfg(feature)]`, trait 유지 | 수정 |
| `manager/src/types.rs` | HierarchicalPolicy 전용 타입에 `#[cfg(feature)]` | 수정 |
| `manager/src/lua_policy.rs` | SignalState, TriggerEngine, EwmaReliefTable, Observation 추가. build_ctx() 확장. process_signal()에서 SystemSignal 활용 | 수정 (대폭) |
| `manager/src/config.rs` | `AdaptationConfig` 파싱 추가. 기존 `PolicyConfig`에 `#[cfg(feature)]` | 수정 |
| `manager/src/pi_controller.rs` | `#[cfg(feature = "hierarchical")]` 전체 | 수정 |
| `manager/src/supervisory.rs` | 동일 | 수정 |
| `manager/src/selector.rs` | 동일 | 수정 |
| `manager/src/action_registry.rs` | 동일 | 수정 |
| `manager/src/evaluator.rs` | 동일 | 수정 |
| `manager/src/relief/linear.rs` | 동일 | 수정 |
| `manager/scripts/policy_example.lua` | ctx.coef 활용 스크립트로 교체 | 수정 |
| `policy_config.toml` | `[adaptation]` 섹션 추가 | 수정 |

---

## 8. 테스트

### Unit Test

- **TbtTracker**: warmup → baseline 설정 → degradation_ratio 정확성. throughput=0 skip
- **TriggerEngine**: 각 trigger 독립 발동, hysteresis enter/exit
- **EwmaReliefTable**: 첫 관측 직접 대입, 반복 수렴, save/load round-trip, config default 적용
- **SignalState**: SystemSignal 4종 → 6D pressure 계산 정확성
- **Observation**: 단일 액션 시 EWMA 갱신, 복수 액션 시 건너뜀

### Integration Test

- **LuaPolicy e2e**: SystemSignal 시퀀스 → ctx.coef 정상 구성 → decide() → EngineCommand
- **Feature flag**: `cargo build` (default) 빌드 성공
- **Feature flag**: `cargo build --features hierarchical` 기존 테스트 통과

### 검증 데이터 (논문 §4.8용)

```csv
timestamp, action, predicted_relief[6], actual_relief[6], observation_count
```
