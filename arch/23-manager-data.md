# Manager Data Types and Configuration Schema -- Architecture

> spec/23-manager-data.md의 구현 상세. 컴포넌트 중심 기술.

---

## 1. TOML Config 계층 구조

### 설계 결정

Config 시스템은 2계층 구조를 따른다:

1. **Top-level `Config`** -- 모니터 4종 + 외부 모니터 + 적응 설정 + (선택) 정책 엔진을 구성
2. **AdaptationConfig** -- LuaPolicy 적응 엔진 설정 (EWMA, triggers, relief defaults) (**신규 2026-04**)
3. **PolicyConfig** -- PI Controller, Supervisory, Selector, Relief Model, Actions, Exclusion Groups의 6개 하위 설정 (**`#[cfg(feature = "hierarchical")]` 뒤로 이동**)

모든 모니터 섹션은 `Option<T>` 래핑이며, 미지정 시 `None` (비활성). 정책도 `Option<PolicyConfig>`. Config 전체는 `#[serde(default)]`로 부분 TOML 파일 허용.

```mermaid
classDiagram
    class Config {
        +manager: ManagerConfig
        +memory: Option~MemoryMonitorConfig~
        +thermal: Option~ThermalMonitorConfig~
        +compute: Option~ComputeMonitorConfig~
        +energy: Option~EnergyMonitorConfig~
        +external: Option~ExternalMonitorConfig~
        +adaptation: AdaptationConfig
        +policy: Option~PolicyConfig~ [hierarchical]
        +from_file(path) Result~Config~
    }
    class AdaptationConfig {
        +ewma_alpha: f32
        +relief_table_path: String
        +temp_safe_c: f32
        +temp_critical_c: f32
        +trigger: TriggerConfig
        +default_relief: HashMap~String, Vec~f32~~
    }
    class TriggerConfig {
        +tbt_enter: f64
        +tbt_exit: f64
        +tbt_warmup_tokens: u32
        +mem_enter: f64
        +mem_exit: f64
        +temp_enter: f64
        +temp_exit: f64
    }
    class PolicyConfig {
        <<hierarchical feature>>
        +pi_controller: PiControllerConfig
        +supervisory: SupervisoryConfig
        +selector: SelectorConfig
        +relief_model: ReliefModelConfig
        +actions: HashMap~String, ActionConfig~
        +exclusion_groups: HashMap~String, Vec~String~~
    }
    Config --> AdaptationConfig
    AdaptationConfig --> TriggerConfig
    Config --> PolicyConfig : [hierarchical]
    PolicyConfig --> PiControllerConfig
    PolicyConfig --> SupervisoryConfig
    PolicyConfig --> SelectorConfig
    PolicyConfig --> ReliefModelConfig
    PolicyConfig --> ActionConfig
```

### 인터페이스

**`Config`** (`manager/src/config.rs`)

```rust
pub struct Config {
    pub manager: ManagerConfig,
    pub memory: Option<MemoryMonitorConfig>,
    pub thermal: Option<ThermalMonitorConfig>,
    pub compute: Option<ComputeMonitorConfig>,
    pub energy: Option<EnergyMonitorConfig>,
    pub external: Option<ExternalMonitorConfig>,
    pub adaptation: AdaptationConfig,          // 신규 (2026-04): LuaPolicy 적응 설정

    #[cfg(feature = "hierarchical")]
    pub policy: Option<PolicyConfig>,          // DEPRECATED
}

impl Config {
    pub fn from_file(path: &Path) -> anyhow::Result<Self>;
}
```

- **Pre**: `path`가 유효한 TOML 파일
- **Post**: 각 섹션은 `#[serde(default)]`로 기본값 보충됨
- **Error**: TOML 파싱 실패 시 `anyhow::Error` 전파 (MGR-DAT-C02)

### Config 검증 규칙

| 규칙 | 보장 방법 | 관련 INV |
|------|----------|---------|
| 비어있는 actions HashMap = 빈 ActionRegistry | `PolicyConfig::default()` | MGR-DAT-C01 |
| TOML 파싱 실패 = 에러 전파 | `Config::from_file()` → `anyhow::Result` | MGR-DAT-C02 |
| 모니터 미지정 = 비활성 | `Option<T>` + `None` default | 아키텍처 |
| Supervisory 임계값 부등식 | Default 값으로 정적 보장 | INV-034~036 |

### Config 키 레퍼런스

#### `[manager]` -- ManagerConfig

| TOML 키 | 타입 | 기본값 | spec ID |
|---------|------|--------|---------|
| `poll_interval_ms` | u64 | 1000 | MGR-DAT-021 |

#### `[memory]` -- MemoryMonitorConfig (Descending: 낮을수록 심각)

| TOML 키 | 타입 | 기본값 | spec ID |
|---------|------|--------|---------|
| `enabled` | bool | true | MGR-DAT-022 |
| `poll_interval_ms` | Option\<u64\> | None | MGR-DAT-022 |
| `warning_pct` | f64 | 40.0 | MGR-DAT-022 |
| `critical_pct` | f64 | 20.0 | MGR-DAT-022 |
| `emergency_pct` | f64 | 10.0 | MGR-DAT-022 |
| `hysteresis_pct` | f64 | 5.0 | MGR-DAT-022 |

#### `[thermal]` -- ThermalMonitorConfig (Ascending: 높을수록 심각)

| TOML 키 | 타입 | 기본값 | spec ID |
|---------|------|--------|---------|
| `enabled` | bool | true | MGR-DAT-023 |
| `poll_interval_ms` | Option\<u64\> | None | MGR-DAT-023 |
| `zone_types` | Vec\<String\> | [] | MGR-DAT-023 |
| `warning_mc` | i32 | 60000 | MGR-DAT-023 |
| `critical_mc` | i32 | 75000 | MGR-DAT-023 |
| `emergency_mc` | i32 | 85000 | MGR-DAT-023 |
| `hysteresis_mc` | i32 | 5000 | MGR-DAT-023 |

#### `[compute]` -- ComputeMonitorConfig (Ascending, Emergency 없음)

| TOML 키 | 타입 | 기본값 | spec ID |
|---------|------|--------|---------|
| `enabled` | bool | true | MGR-DAT-024 |
| `poll_interval_ms` | Option\<u64\> | None | MGR-DAT-024 |
| `warning_pct` | f64 | 70.0 | MGR-DAT-024 |
| `critical_pct` | f64 | 90.0 | MGR-DAT-024 |
| `hysteresis_pct` | f64 | 5.0 | MGR-DAT-024 |

#### `[energy]` -- EnergyMonitorConfig (Descending)

| TOML 키 | 타입 | 기본값 | spec ID |
|---------|------|--------|---------|
| `enabled` | bool | true | MGR-DAT-025 |
| `poll_interval_ms` | Option\<u64\> | None | MGR-DAT-025 |
| `warning_pct` | f64 | 30.0 | MGR-DAT-025 |
| `critical_pct` | f64 | 15.0 | MGR-DAT-025 |
| `emergency_pct` | f64 | 5.0 | MGR-DAT-025 |
| `warning_power_budget_mw` | u32 | 3000 | MGR-DAT-025 |
| `critical_power_budget_mw` | u32 | 1500 | MGR-DAT-025 |
| `emergency_power_budget_mw` | u32 | 500 | MGR-DAT-025 |
| `ignore_when_charging` | bool | true | MGR-DAT-025 |

#### `[external]` -- ExternalMonitorConfig

| TOML 키 | 타입 | 기본값 | spec ID |
|---------|------|--------|---------|
| `enabled` | bool | false | MGR-DAT-026 |
| `transport` | String | "stdin" | MGR-DAT-026 |

#### `[adaptation]` -- AdaptationConfig (신규 2026-04, LuaPolicy)

| TOML 키 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `ewma_alpha` | f32 | 0.875 | EWMA 평활 계수 (Jacobson TCP RTT) |
| `relief_table_path` | String | "" (비활성) | Relief table JSON 저장/복원 경로 |
| `temp_safe_c` | f32 | 35.0 | Thermal 정규화 하한 (Celsius) |
| `temp_critical_c` | f32 | 50.0 | Thermal 정규화 상한 (Celsius) |

#### `[adaptation.trigger]` -- TriggerConfig (신규 2026-04)

| TOML 키 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `tbt_enter` | f64 | 0.30 | TBT degradation trigger 진입 (baseline 대비 30% 악화) |
| `tbt_exit` | f64 | 0.10 | TBT degradation trigger 해제 |
| `tbt_warmup_tokens` | u32 | 20 | Baseline 확정 전 warmup 토큰 수 |
| `mem_enter` | f64 | 0.80 | Memory pressure trigger 진입 |
| `mem_exit` | f64 | 0.60 | Memory pressure trigger 해제 |
| `temp_enter` | f64 | 0.70 | Temperature trigger 진입 (정규화 값) |
| `temp_exit` | f64 | 0.50 | Temperature trigger 해제 |

#### `[adaptation.default_relief]` -- Per-Action 6D Default Relief

키는 액션 이름 (snake_case), 값은 6-element 배열 `[gpu, cpu, memory, thermal, latency, main_app_qos]`.

```toml
[adaptation.default_relief]
switch_hw = [0.3, 0.0, 0.0, 0.2, -0.1, 0.0]
throttle = [0.2, 0.1, 0.0, 0.15, -0.2, 0.0]
kv_evict_sliding = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
```

---

#### `[policy.pi_controller]` -- PiControllerConfig **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

| TOML 키 | 타입 | 기본값 | spec ID |
|---------|------|--------|---------|
| `compute_kp` | f32 | 1.5 | MGR-DAT-031 |
| `compute_ki` | f32 | 0.3 | MGR-DAT-031 |
| `compute_setpoint` | f32 | 0.70 | MGR-DAT-031 |
| `memory_kp` | f32 | 2.0 | MGR-DAT-031 |
| `memory_ki` | f32 | 0.5 | MGR-DAT-031 |
| `memory_setpoint` | f32 | 0.75 | MGR-DAT-031 |
| `thermal_kp` | f32 | 1.0 | MGR-DAT-031 |
| `thermal_ki` | f32 | 0.2 | MGR-DAT-031 |
| `thermal_setpoint` | f32 | 0.80 | MGR-DAT-031 |
| `integral_clamp` | f32 | 2.0 | MGR-DAT-031 |
| `memory_gain_zones` | Vec\<GainZone\> | [] | MGR-DAT-031 |

#### `[policy.supervisory]` -- SupervisoryConfig **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

| TOML 키 | 타입 | 기본값 | spec ID |
|---------|------|--------|---------|
| `warning_threshold` | f32 | 0.4 | MGR-DAT-032 |
| `critical_threshold` | f32 | 0.7 | MGR-DAT-032 |
| `warning_release` | f32 | 0.25 | MGR-DAT-032 |
| `critical_release` | f32 | 0.50 | MGR-DAT-032 |
| `hold_time_secs` | f32 | 4.0 | MGR-DAT-032 |

#### `[policy.selector]` -- SelectorConfig **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

| TOML 키 | 타입 | 기본값 | spec ID |
|---------|------|--------|---------|
| `latency_budget` | f32 | 0.5 | MGR-DAT-033 |
| `algorithm` | String | "exhaustive" | MGR-DAT-033 |

#### `[policy.relief_model]` -- ReliefModelConfig **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

| TOML 키 | 타입 | 기본값 | spec ID |
|---------|------|--------|---------|
| `forgetting_factor` | f32 | 0.995 | MGR-DAT-034 |
| `prior_weight` | u32 | 5 | MGR-DAT-034 |
| `storage_dir` | String | "~/.llm_rs/models" | MGR-DAT-034 |

#### `[policy.actions.<name>]` -- ActionConfig **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

| TOML 키 | 타입 | 기본값 | spec ID |
|---------|------|--------|---------|
| `lossy` | bool | false | MGR-DAT-035 |
| `reversible` | bool | false | MGR-DAT-035 |
| `default_cost` | f32 | 1.0 | MGR-DAT-035 |

#### `[policy.exclusion_groups]` **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

| TOML 키 | 타입 | 기본값 | spec ID |
|---------|------|--------|---------|
| `<group_name>` | Vec\<String\> | (없음) | MGR-DAT-036 |

---

## 2. SystemSignal 데이터 타입

### 설계 결정

`SystemSignal`은 `shared/` 크레이트 (`llm_shared`)에 정의되어 Manager와 Engine 양쪽에서 사용한다. 각 variant는 도메인별 센서 데이터와 `level: Level` 필드를 모두 포함한다.

### 인터페이스

**`SystemSignal`** (`shared/src/lib.rs`)

```rust
#[serde(rename_all = "snake_case")]
pub enum SystemSignal {
    MemoryPressure {
        level: Level,
        available_bytes: u64,
        total_bytes: u64,
        reclaim_target_bytes: u64,
    },
    ComputeGuidance {
        level: Level,
        recommended_backend: RecommendedBackend,
        reason: ComputeReason,
        cpu_usage_pct: f64,
        gpu_usage_pct: f64,
    },
    ThermalAlert {
        level: Level,
        temperature_mc: i32,
        throttling_active: bool,
        throttle_ratio: f64,
    },
    EnergyConstraint {
        level: Level,
        reason: EnergyReason,
        power_budget_mw: u32,
    },
}

impl SystemSignal {
    pub fn level(&self) -> Level;  // 모든 variant에서 level 추출
}
```

**보조 타입** (`shared/src/lib.rs`)

```rust
#[derive(PartialOrd, Ord)]  // Normal < Warning < Critical < Emergency
pub enum Level { Normal, Warning, Critical, Emergency }

pub enum RecommendedBackend { Cpu, Gpu, Any }
pub enum ComputeReason { CpuBottleneck, GpuBottleneck, CpuAvailable, GpuAvailable, BothLoaded, Balanced }
pub enum EnergyReason { BatteryLow, BatteryCritical, PowerLimit, ThermalPower, Charging, None }
```

- 모든 보조 타입은 `from_dbus_str(&str) -> Option<Self>` 메서드를 제공 (D-Bus 문자열 역직렬화)
- 모든 타입은 `serde(rename_all = "snake_case")` 적용

### 데이터 흐름

```mermaid
flowchart LR
    MemMon[MemoryMonitor<br>/proc/meminfo] --> |SystemSignal| Bus[mpsc channel]
    ThermMon[ThermalMonitor<br>/sys/class/thermal/] --> |SystemSignal| Bus
    CompMon[ComputeMonitor<br>/proc/stat] --> |SystemSignal| Bus
    EneMon[EnergyMonitor<br>/sys/class/power_supply/] --> |SystemSignal| Bus
    ExtMon[ExternalMonitor<br>stdin/unix socket] --> |SystemSignal| Bus
    Bus --> PolicyEngine["LuaPolicy (기본)<br/>또는 HierarchicalPolicy [hierarchical]"]
    PolicyEngine --> |EngineDirective| Emitter[Emitter]
```

---

## 3. Monitor 인터페이스

### 설계 결정

각 Monitor는 독립 OS 스레드에서 실행되며, `mpsc::Sender<SystemSignal>`을 통해 정책 엔진에 시그널을 전달한다. 공유 상태 없이 채널로만 통신하여 Monitor 간 장애 전파를 차단한다 (INV-013).

### 인터페이스

**`Monitor` trait** (`manager/src/monitor/mod.rs`)

```rust
pub trait Monitor: Send + 'static {
    fn run(&mut self, tx: mpsc::Sender<SystemSignal>, shutdown: Arc<AtomicBool>) -> anyhow::Result<()>;
    fn initial_signal(&self) -> Option<SystemSignal>;
    fn name(&self) -> &str;
}
```

- **Pre**: `shutdown`이 false 상태
- **Post**: `shutdown`이 true가 되면 `Ok(())` 반환
- **Side effect**: 임계값 교차 시 `tx`로 `SystemSignal` 전송

### Monitor별 데이터 소스

| Monitor | 데이터 소스 | Direction | 임계값 단위 |
|---------|-----------|-----------|-----------|
| MemoryMonitor | `/proc/meminfo` | Descending (낮을수록 심각) | % (available memory) |
| ThermalMonitor | `/sys/class/thermal/` | Ascending (높을수록 심각) | millidegree Celsius |
| ComputeMonitor | `/proc/stat` (CPU delta) | Ascending (높을수록 심각) | % (CPU usage) |
| EnergyMonitor | `/sys/class/power_supply/` | Descending (낮을수록 심각) | % (battery) + mW (budget) |
| ExternalMonitor | stdin 또는 Unix socket | N/A (직접 JSON 주입) | N/A |

모든 Monitor (External 제외)는 내부적으로 `ThresholdEvaluator`를 사용하여 raw 측정값을 `Level`로 변환한다. Hysteresis 적용으로 Level 간 진동(flickering)을 방지한다.

---

## 4. Core Data Types (Policy Engine)

### 설계 결정

Manager 정책 엔진의 핵심 타입은 두 계층으로 분리되어 있다:

- **공용**: `OperatingMode` (`manager/src/types.rs`) — LuaPolicy와 HierarchicalPolicy 모두 사용
- **HierarchicalPolicy 전용** (`#[cfg(feature = "hierarchical")]`): `ActionId`, `PressureVector`, `ReliefVector`, `FeatureVector` 등 (`manager/src/types.rs` 내 `hierarchical_types` 모듈)
- **LuaPolicy 전용**: `Pressure6D`, `SignalState`, `TriggerState`, `ReliefEntry` 등 (`manager/src/lua_policy.rs` 내 비공개 타입)

### ActionId 및 도메인 매핑 **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

**`ActionId`** (`manager/src/types.rs`, `hierarchical_types` 모듈 내)

```rust
#[serde(rename_all = "snake_case")]
pub enum ActionId {
    SwitchHw, Throttle, KvOffloadDisk,
    KvEvictSliding, KvEvictH2o, KvQuantDynamic, LayerSkip,
}

impl ActionId {
    pub fn from_str(s: &str) -> Option<ActionId>;
    pub fn all() -> &'static [ActionId];      // 7종
    pub fn primary_domain(&self) -> Domain;    // INV-045
}
```

- `primary_domain()` 매핑: SwitchHw/Throttle/LayerSkip -> `Domain::Compute`, 나머지 -> `Domain::Memory` (INV-045)

### Pressure / Relief 벡터 **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

> LuaPolicy는 `Pressure6D`(6D)와 `[f32; 6]` relief 배열을 사용한다 (§4a 참조).

```rust
pub struct PressureVector { pub compute: f32, pub memory: f32, pub thermal: f32 }
impl PressureVector {
    pub fn max(&self) -> f32;
    pub fn any_domain_exceeds(&self, reference: &PressureVector, factor: f32) -> bool;
}
impl Sub for PressureVector { type Output = ReliefVector; }  // latency=0.0 (INV-050)

pub struct ReliefVector { pub compute: f32, pub memory: f32, pub thermal: f32, pub latency: f32 }
impl ReliefVector { pub fn zero() -> Self; }
impl Add for ReliefVector { ... }
impl AddAssign for ReliefVector { ... }
```

### FeatureVector (Relief Estimator 입력) **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

```rust
pub const FEATURE_DIM: usize = 13;
pub struct FeatureVector { pub values: [f32; FEATURE_DIM] }
```

13개 feature 인덱스 상수: `feature::KV_OCCUPANCY(0)`, `IS_GPU(1)`, `TOKEN_PROGRESS(2)`, `IS_PREFILL(3)`, `KV_DTYPE_NORM(4)`, `TBT_RATIO(5)`, `TOKENS_GENERATED_NORM(6)`, `ACTIVE_SWITCH_HW(7)`, `ACTIVE_THROTTLE(8)`, `ACTIVE_KV_OFFLOAD(9)`, `ACTIVE_EVICTION(10)`, `ACTIVE_LAYER_SKIP(11)`, `ACTIVE_KV_QUANT(12)`.

### Action 관련 타입

| 타입 | 역할 | 위치 | Feature |
|------|------|------|---------|
| `OperatingMode { Normal, Warning, Critical }` | 시스템 운영 모드 (PartialOrd/Ord) | `types.rs` | 공용 |
| `ActionKind { Lossless, Lossy }` | 품질 영향 분류 | `types.rs` | hierarchical |
| `Domain { Compute, Memory, Thermal }` | 압력 도메인 (내부 전용, serde 없음) | `types.rs` | hierarchical |
| `ActionMeta { id, kind, reversible, param_range, exclusion_group, default_cost }` | 액션 메타데이터 | `types.rs` | hierarchical |
| `ParamRange { param_name, min, max }` | 파라미터 범위 (Serialize/Deserialize) | `types.rs` | hierarchical |
| `ActionParams { values: HashMap<String, f32> }` | 파라미터 집합 | `types.rs` | hierarchical |
| `ActionCommand { action, operation }` | Selector 출력 | `types.rs` | hierarchical |
| `Operation { Apply(ActionParams), Release }` | 적용/해제 | `types.rs` | hierarchical |

### 4a. LuaPolicy 전용 데이터 타입 (신규 2026-04)

> 요구사항: MGR-DAT-070~076, MGR-090~093, INV-086~092

**파일**: `manager/src/lua_policy.rs` (비공개 타입)

#### EngineStatus 신규 필드 (Phase 1, 2026-04)

> 요구사항: MSG-060 #17~18, MSG-067~069, MGR-DAT-075, MGR-DAT-076, INV-091, INV-092

`shared::EngineStatus`에 2개 필드가 추가되어 LuaPolicy `ctx.engine`으로 노출된다. 상세 흐름은 `arch/20-manager.md` §10.7 참조.

```rust
// shared/src/lib.rs (수정 위치 — 본 문서는 참조만)
pub struct EngineStatus {
    // ... 기존 16 필드 ...
    #[serde(default)]
    pub self_cpu_pct: f64,   // MGR-DAT-075, [0,1] clamp
    #[serde(default)]
    pub self_gpu_pct: f64,   // MGR-DAT-076, Phase 1 = 0.0
}
```

| 필드 | serde | 구버전 역직렬화 | 소비 위치 |
|------|-------|----------------|----------|
| `self_cpu_pct` | `#[serde(default)]` → 0.0 | OK | `LuaPolicy::build_ctx()` → `ctx.engine.cpu_pct` |
| `self_gpu_pct` | `#[serde(default)]` → 0.0 | OK | `LuaPolicy::build_ctx()` → `ctx.engine.gpu_pct` (항상 0.0) |

Pressure6D / EwmaReliefTable / TriggerEngine 계산에는 두 필드를 사용하지 않는다 (설계 결정: `arch/20-manager.md` §10.7).

#### Pressure6D

```rust
struct Pressure6D {
    gpu: f32,       // GPU 사용률 (0~1)
    cpu: f32,       // CPU 사용률 (0~1)
    memory: f32,    // 메모리 사용률 (0~1)
    thermal: f32,   // 정규화 온도 (0~1)
    latency: f32,   // TBT degradation ratio (0~inf)
    main_app: f32,  // 예약 (0.0)
}
```

HierarchicalPolicy의 `PressureVector`(3D: compute, memory, thermal)를 6D로 확장. GPU/CPU를 분리하고 latency/main_app 차원을 추가하여 Lua 스크립트에 세밀한 압력 정보를 제공한다.

#### TriggerState

```rust
struct TriggerState {
    tbt_degraded: bool,  // TBT baseline 대비 악화
    mem_low: bool,       // 메모리 부족
    temp_high: bool,     // 온도 과열
}
```

#### ReliefEntry (EWMA 학습 데이터)

```rust
struct ReliefEntry {
    relief: [f32; 6],         // 6D relief vector (RELIEF_DIMS)
    observation_count: u32,   // 관측 횟수
}
```

#### AdaptationConfig / TriggerConfig

```rust
// manager/src/config.rs
pub struct AdaptationConfig {
    pub ewma_alpha: f32,                              // default 0.875
    pub relief_table_path: String,                    // default ""
    pub temp_safe_c: f32,                             // default 35.0
    pub temp_critical_c: f32,                         // default 50.0
    pub trigger: TriggerConfig,
    pub default_relief: HashMap<String, Vec<f32>>,    // per-action 6D prior
}

pub struct TriggerConfig {
    pub tbt_enter: f64,          // default 0.30
    pub tbt_exit: f64,           // default 0.10
    pub tbt_warmup_tokens: u32,  // default 20
    pub mem_enter: f64,          // default 0.80
    pub mem_exit: f64,           // default 0.60
    pub temp_enter: f64,         // default 0.70
    pub temp_exit: f64,          // default 0.50
}
```

---

## 5. ActionRegistry **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

### 설계 결정

`ActionRegistry`는 TOML 설정에서 등록된 액션의 메타데이터와 배타 그룹을 통합 관리한다. PolicyConfig에 포함된 액션만 등록되며, 알 수 없는 이름은 무시한다.

### 인터페이스

**`ActionRegistry`** (`manager/src/action_registry.rs`)

```rust
pub struct ActionRegistry {
    actions: HashMap<ActionId, ActionMeta>,
    exclusion_groups: HashMap<String, Vec<ActionId>>,
}

impl ActionRegistry {
    pub fn from_config(config: &PolicyConfig) -> Self;
    pub fn get(&self, action: &ActionId) -> Option<&ActionMeta>;
    pub fn all_actions(&self) -> impl Iterator<Item = &ActionMeta>;
    pub fn lossy_actions(&self) -> Vec<ActionId>;
    pub fn lossless_actions(&self) -> Vec<ActionId>;
    pub fn exclusion_groups(&self) -> &HashMap<String, Vec<ActionId>>;
    pub fn is_excluded(&self, a: &ActionId, b: &ActionId) -> bool;
    pub fn default_cost(&self, action: &ActionId) -> f32;  // 미등록 = 1.0
}
```

- **`from_config()` 처리 흐름**: 순회 -> `ActionId::from_str()` 필터 -> kind 분류 -> `default_param_range()` 할당 -> exclusion group 매핑

### default_param_range 하드코딩 테이블

`default_param_range()` 함수 (`manager/src/action_registry.rs`) -- 액션별 기본 파라미터 범위.

| ActionId | param_name | min | max |
|----------|-----------|-----|-----|
| KvEvictSliding | keep_ratio | 0.3 | 0.9 |
| KvEvictH2o | keep_ratio | 0.3 | 0.9 |
| KvQuantDynamic | target_bits | 4.0 | 8.0 |
| LayerSkip | skip_layers | 1.0 | 8.0 |
| Throttle | delay_ms | 0.0 | 100.0 |
| SwitchHw | (None) | -- | -- |
| KvOffloadDisk | (None) | -- | -- |

---

## 6. 코드-스펙 차이 (Known Divergence)

| 항목 | 스펙 | 코드 | 영향 |
|------|------|------|------|
| SystemSignal level 필드 | raw 전용 (level 없음) | 모든 variant에 `level: Level` 포함 | Monitor가 Level 산출 담당. 스펙 갱신 권장 |
| ActionId 8종 | 8종 (KvMergeD2o 포함) | 8종 (KvMergeD2o 구현 완료, hierarchical 뒤) | 스펙-코드 일치 |
| EnergyMonitor hysteresis | 하드코딩 2.0 | Config 필드 없음, 코드 내 고정값 | 일치 (하드코딩 의도) |
| default_relief 8종 | 8종 (KvMergeD2o 포함) | 8종 (KvMergeD2o 포함, hierarchical 뒤) | 스펙-코드 일치 |
| ReliefVector 4D | 4D (compute, memory, thermal, latency) | LuaPolicy: 6D `[f32; 6]` (gpu, cpu, memory, thermal, latency, main_app) | LuaPolicy 전용 변경. 기존 4D는 hierarchical 뒤에 유지 |
| 정책 기본값 | HierarchicalPolicy 기본 | LuaPolicy 기본 (`lua` feature default) | 2026-04 변경. hierarchical은 opt-in feature |
| PressureVector 3D | 3D (compute, memory, thermal) | LuaPolicy: `Pressure6D` 6D (gpu/cpu 분리 + latency + main_app) | LuaPolicy 전용 변경 |