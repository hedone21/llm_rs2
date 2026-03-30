# Manager Data Types and Configuration Schema -- Architecture

> spec/23-manager-data.md의 구현 상세.

## 코드 매핑

### 3.1 SystemSignal Fields (Manager Consumption)

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-DAT-010 | `shared/src/lib.rs` | `enum SystemSignal` | **코드-스펙 차이**: 코드의 SystemSignal에 `level: Level` 필드 포함. 스펙은 raw 전용 |
| MGR-DAT-011 | `shared/src/lib.rs` | `SystemSignal::MemoryPressure { level, available_bytes, total_bytes, reclaim_target_bytes }` | level 필드 존재 (스펙과 차이) |
| MGR-DAT-012 | `shared/src/lib.rs` | `SystemSignal::ThermalAlert { level, temperature_mc, throttling_active, throttle_ratio }` | |
| MGR-DAT-013 | `shared/src/lib.rs` | `SystemSignal::ComputeGuidance { level, recommended_backend, reason, cpu_usage_pct, gpu_usage_pct }` | |
| MGR-DAT-014 | `shared/src/lib.rs` | `SystemSignal::EnergyConstraint { level, reason, battery_pct, power_budget_mw }` | |
| MGR-DAT-015 | `shared/src/lib.rs` | `enum Level { Normal, Warning, Critical, Emergency }` | serde `rename_all = "snake_case"`, Ord derive |
| MGR-DAT-016 | `shared/src/lib.rs` | `enum RecommendedBackend`, `enum ComputeReason`, `enum EnergyReason` | 각각 serde `rename_all = "snake_case"` |

### 3.2 Config TOML Schema

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-DAT-020 | `manager/src/config.rs:7-17` | `struct Config` | 모든 섹션 `#[serde(default)]` |
| MGR-DAT-021 | `manager/src/config.rs:26-39` | `struct ManagerConfig` | `poll_interval_ms: u64 = 1000` |
| MGR-DAT-022 | `manager/src/config.rs:44-66` | `struct MemoryMonitorConfig` | Descending. enabled, poll_interval_ms, warning/critical/emergency_pct, hysteresis_pct |
| MGR-DAT-023 | `manager/src/config.rs:70-95` | `struct ThermalMonitorConfig` | Ascending. zone_types, warning/critical/emergency_mc, hysteresis_mc |
| MGR-DAT-024 | `manager/src/config.rs:100-119` | `struct ComputeMonitorConfig` | Ascending. emergency 없음 (f64::MAX 비활성) |
| MGR-DAT-025 | `manager/src/config.rs:124-153` | `struct EnergyMonitorConfig` | Descending. ignore_when_charging, power_budget_mw 3단계 |
| MGR-DAT-026 | `manager/src/config.rs:156-171` | `struct ExternalMonitorConfig` | enabled=false, transport="stdin" |

### 3.3 PolicyConfig TOML Schema

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-DAT-030 | `manager/src/config.rs:174-183` | `struct PolicyConfig` | pi_controller, supervisory, selector, relief_model, actions, exclusion_groups |
| MGR-DAT-031 | `manager/src/config.rs:186-221` | `struct PiControllerConfig` | compute/memory/thermal kp/ki/setpoint, integral_clamp, memory_gain_zones |
| MGR-DAT-032 | `manager/src/config.rs:224-244` | `struct SupervisoryConfig` | warning/critical threshold/release, hold_time_secs |
| MGR-DAT-033 | `manager/src/config.rs:247-261` | `struct SelectorConfig` | latency_budget, algorithm |
| MGR-DAT-034 | `manager/src/config.rs:264-280` | `struct ReliefModelConfig` | forgetting_factor, prior_weight, storage_dir |
| MGR-DAT-035 | `manager/src/config.rs:283-304` | `struct ActionConfig` | lossy, reversible, default_cost |
| MGR-DAT-036 | `manager/src/config.rs:181` | `exclusion_groups: HashMap<String, Vec<String>>` | PolicyConfig 필드 |

### 3.4 Core Data Types

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-DAT-040 | `manager/src/types.rs:5-15` | `enum ActionId` | **7종** (KvMergeD2o 미포함). 스펙 8종 |
| MGR-DAT-040 | `manager/src/types.rs:18-31` | `ActionId::from_str()` | snake_case 문자열 매핑 |
| MGR-DAT-040 | `manager/src/types.rs:34-44` | `ActionId::all()` | 7종 반환 |
| MGR-DAT-040 | `manager/src/types.rs:47-56` | `ActionId::primary_domain()` | SwitchHw/Throttle/LayerSkip→Compute, 나머지→Memory |
| MGR-DAT-041 | `manager/src/types.rs:59-64` | `enum ActionKind { Lossless, Lossy }` | serde rename_all |
| MGR-DAT-042 | `manager/src/types.rs:67-72` | `enum Domain { Compute, Memory, Thermal }` | serde derive 없음 (내부 전용) |
| MGR-DAT-043 | `manager/src/types.rs:75-80` | `enum OperatingMode { Normal, Warning, Critical }` | PartialOrd/Ord derive |
| MGR-DAT-044 | `manager/src/types.rs:83-101` | `struct PressureVector { compute, memory, thermal: f32 }` | `max()`, `any_domain_exceeds()` 메서드 |
| MGR-DAT-044 | `manager/src/types.rs:103-113` | `impl Sub<PressureVector> for PressureVector` | Output = ReliefVector (latency=0.0) |
| MGR-DAT-045 | `manager/src/types.rs:116-149` | `struct ReliefVector { compute, memory, thermal, latency: f32 }` | Add/AddAssign impl. Serialize/Deserialize |
| MGR-DAT-046 | `manager/src/types.rs:152-182` | `struct FeatureVector { values: [f32; 13] }` + `mod feature` | FEATURE_DIM=13, 상수 인덱스 |
| MGR-DAT-050 | `manager/src/types.rs:185-193` | `struct ActionMeta` | id, kind, reversible, param_range, exclusion_group, default_cost |
| MGR-DAT-051 | `manager/src/types.rs:195-200` | `struct ParamRange { param_name, min, max }` | Serialize/Deserialize |
| MGR-DAT-052 | `manager/src/types.rs:203-206` | `struct ActionParams { values: HashMap<String, f32> }` | Default: 빈 HashMap |
| MGR-DAT-053 | `manager/src/types.rs:209-219` | `struct ActionCommand`, `enum Operation { Apply, Release }` | |

### 3.5 ActionMeta and ActionRegistry

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-DAT-054 | `manager/src/action_registry.rs` | `ActionRegistry` — `from_config()` | 설정에 포함된 액션만 등록 |
| MGR-DAT-055 | `manager/src/action_registry.rs:17-56` | `from_config()` 구현 | 6단계: 순회 → from_str → kind → param_range → exclusion |
| MGR-DAT-056 | `manager/src/action_registry.rs` | `default_param_range()` 함수 | 하드코딩 테이블. KvMergeD2o 분기 없음 (variant 없음) |
| MGR-DAT-057 | `manager/src/action_registry.rs` | `get`, `all_actions`, `lossy_actions`, `lossless_actions`, `exclusion_groups`, `is_excluded`, `default_cost` | 7개 메서드 |

### 3.6 ReliefEstimator Default Relief Prior

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-DAT-060 | `manager/src/relief/linear.rs` | `default_relief()` 함수 | **7종** (KvMergeD2o 없음). 스펙 8종 |

### Constraints

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-DAT-C01 | `manager/src/config.rs:174-183` | `PolicyConfig::default()` → actions 빈 HashMap | ActionRegistry 빈 상태 → 빈 조합 반환 |
| MGR-DAT-C02 | `manager/src/config.rs:20-23` | `Config::from_file()` → `anyhow::Result` | TOML 파싱 실패 시 에러 전파 |

## 코드-스펙 차이 (Known Divergence)

| 항목 | 스펙 | 코드 | 영향 |
|------|------|------|------|
| SystemSignal level 필드 (MGR-DAT-010) | SystemSignal에 level 필드 없음 (raw 전용) | 모든 SystemSignal variant에 `level: Level` 필드 존재 | Monitor가 Level 산출을 담당. 스펙은 Emitter 전용 |
| ActionId 7 vs 8종 (MGR-DAT-040) | 8종 (KvMergeD2o 포함) | 7종 (KvMergeD2o 없음) | 코드에 variant 추가 필요 시 스펙 변경 불필요 |
| EnergyMonitor hysteresis | 하드코딩 2.0 (스펙 일치) | EnergyMonitorConfig에 필드 없음, 코드 내 고정값 | 일치 (하드코딩 의도) |
| default_relief 7 vs 8종 (MGR-DAT-060) | 8종 (KvMergeD2o 포함) | 7종 | ActionId 추가 시 함께 추가 |

## Config

### 최상위 Config

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `manager.poll_interval_ms` | u64 | 1000 | MGR-DAT-021 |

### MemoryMonitorConfig (`[memory]`)

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `memory.enabled` | bool | true | MGR-DAT-022 |
| `memory.poll_interval_ms` | Option\<u64\> | None | MGR-DAT-022 |
| `memory.warning_pct` | f64 | 40.0 | MGR-DAT-022 |
| `memory.critical_pct` | f64 | 20.0 | MGR-DAT-022 |
| `memory.emergency_pct` | f64 | 10.0 | MGR-DAT-022 |
| `memory.hysteresis_pct` | f64 | 5.0 | MGR-DAT-022 |

### ThermalMonitorConfig (`[thermal]`)

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `thermal.enabled` | bool | true | MGR-DAT-023 |
| `thermal.poll_interval_ms` | Option\<u64\> | None | MGR-DAT-023 |
| `thermal.zone_types` | Vec\<String\> | [] | MGR-DAT-023 |
| `thermal.warning_mc` | i32 | 60000 | MGR-DAT-023 |
| `thermal.critical_mc` | i32 | 75000 | MGR-DAT-023 |
| `thermal.emergency_mc` | i32 | 85000 | MGR-DAT-023 |
| `thermal.hysteresis_mc` | i32 | 5000 | MGR-DAT-023 |

### ComputeMonitorConfig (`[compute]`)

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `compute.enabled` | bool | true | MGR-DAT-024 |
| `compute.poll_interval_ms` | Option\<u64\> | None | MGR-DAT-024 |
| `compute.warning_pct` | f64 | 70.0 | MGR-DAT-024 |
| `compute.critical_pct` | f64 | 90.0 | MGR-DAT-024 |
| `compute.hysteresis_pct` | f64 | 5.0 | MGR-DAT-024 |

### EnergyMonitorConfig (`[energy]`)

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `energy.enabled` | bool | true | MGR-DAT-025 |
| `energy.poll_interval_ms` | Option\<u64\> | None | MGR-DAT-025 |
| `energy.warning_pct` | f64 | 30.0 | MGR-DAT-025 |
| `energy.critical_pct` | f64 | 15.0 | MGR-DAT-025 |
| `energy.emergency_pct` | f64 | 5.0 | MGR-DAT-025 |
| `energy.warning_power_budget_mw` | u32 | 3000 | MGR-DAT-025 |
| `energy.critical_power_budget_mw` | u32 | 1500 | MGR-DAT-025 |
| `energy.emergency_power_budget_mw` | u32 | 500 | MGR-DAT-025 |
| `energy.ignore_when_charging` | bool | true | MGR-DAT-025 |

### ExternalMonitorConfig (`[external]`)

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `external.enabled` | bool | false | MGR-DAT-026 |
| `external.transport` | String | "stdin" | MGR-DAT-026 |

### PiControllerConfig (`[policy.pi_controller]`)

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `policy.pi_controller.compute_kp` | f32 | 1.5 | MGR-DAT-031 |
| `policy.pi_controller.compute_ki` | f32 | 0.3 | MGR-DAT-031 |
| `policy.pi_controller.compute_setpoint` | f32 | 0.70 | MGR-DAT-031 |
| `policy.pi_controller.memory_kp` | f32 | 2.0 | MGR-DAT-031 (비사용) |
| `policy.pi_controller.memory_ki` | f32 | 0.5 | MGR-DAT-031 (비사용) |
| `policy.pi_controller.memory_setpoint` | f32 | 0.75 | MGR-DAT-031 (비사용) |
| `policy.pi_controller.thermal_kp` | f32 | 1.0 | MGR-DAT-031 |
| `policy.pi_controller.thermal_ki` | f32 | 0.2 | MGR-DAT-031 |
| `policy.pi_controller.thermal_setpoint` | f32 | 0.80 | MGR-DAT-031 |
| `policy.pi_controller.integral_clamp` | f32 | 2.0 | MGR-DAT-031 |
| `policy.pi_controller.memory_gain_zones` | Vec\<GainZone\> | [] | MGR-DAT-031 (비사용) |

### SupervisoryConfig (`[policy.supervisory]`)

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `policy.supervisory.warning_threshold` | f32 | 0.4 | MGR-DAT-032 |
| `policy.supervisory.critical_threshold` | f32 | 0.7 | MGR-DAT-032 |
| `policy.supervisory.warning_release` | f32 | 0.25 | MGR-DAT-032 |
| `policy.supervisory.critical_release` | f32 | 0.50 | MGR-DAT-032 |
| `policy.supervisory.hold_time_secs` | f32 | 4.0 | MGR-DAT-032 |

### SelectorConfig (`[policy.selector]`)

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `policy.selector.latency_budget` | f32 | 0.5 | MGR-DAT-033 |
| `policy.selector.algorithm` | String | "exhaustive" | MGR-DAT-033 |

### ReliefModelConfig (`[policy.relief_model]`)

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `policy.relief_model.forgetting_factor` | f32 | 0.995 | MGR-DAT-034 |
| `policy.relief_model.prior_weight` | u32 | 5 | MGR-DAT-034 |
| `policy.relief_model.storage_dir` | String | "~/.llm_rs/models" | MGR-DAT-034 |

### ActionConfig (`[policy.actions.<name>]`)

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `policy.actions.<name>.lossy` | bool | false | MGR-DAT-035 |
| `policy.actions.<name>.reversible` | bool | false | MGR-DAT-035 |
| `policy.actions.<name>.default_cost` | f32 | 1.0 | MGR-DAT-035 |

### exclusion_groups (`[policy.exclusion_groups]`)

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `policy.exclusion_groups.<name>` | Vec\<String\> | (없음) | MGR-DAT-036 |

## 주요 Struct/Trait 매핑

| spec 개념 | Rust 타입 | 위치 |
|-----------|----------|------|
| SystemSignal | `enum SystemSignal` | `shared/src/lib.rs` |
| Level | `enum Level` | `shared/src/lib.rs` |
| RecommendedBackend | `enum RecommendedBackend` | `shared/src/lib.rs` |
| ComputeReason | `enum ComputeReason` | `shared/src/lib.rs` |
| EnergyReason | `enum EnergyReason` | `shared/src/lib.rs` |
| Config | `struct Config` | `manager/src/config.rs:7-17` |
| ManagerConfig | `struct ManagerConfig` | `manager/src/config.rs:26-39` |
| MemoryMonitorConfig | `struct MemoryMonitorConfig` | `manager/src/config.rs:44-66` |
| ThermalMonitorConfig | `struct ThermalMonitorConfig` | `manager/src/config.rs:70-95` |
| ComputeMonitorConfig | `struct ComputeMonitorConfig` | `manager/src/config.rs:100-119` |
| EnergyMonitorConfig | `struct EnergyMonitorConfig` | `manager/src/config.rs:124-153` |
| ExternalMonitorConfig | `struct ExternalMonitorConfig` | `manager/src/config.rs:156-171` |
| PolicyConfig | `struct PolicyConfig` | `manager/src/config.rs:174-183` |
| PiControllerConfig | `struct PiControllerConfig` | `manager/src/config.rs:186-221` |
| SupervisoryConfig | `struct SupervisoryConfig` | `manager/src/config.rs:224-244` |
| SelectorConfig | `struct SelectorConfig` | `manager/src/config.rs:247-261` |
| ReliefModelConfig | `struct ReliefModelConfig` | `manager/src/config.rs:264-280` |
| ActionConfig | `struct ActionConfig` | `manager/src/config.rs:283-304` |
| GainZone | `struct GainZone` | `manager/src/pi_controller.rs:5-8` |
| ActionId | `enum ActionId` | `manager/src/types.rs:5-15` |
| ActionKind | `enum ActionKind` | `manager/src/types.rs:59-64` |
| Domain | `enum Domain` | `manager/src/types.rs:67-72` |
| OperatingMode | `enum OperatingMode` | `manager/src/types.rs:75-80` |
| PressureVector | `struct PressureVector` | `manager/src/types.rs:83-101` |
| ReliefVector | `struct ReliefVector` | `manager/src/types.rs:116-149` |
| FeatureVector | `struct FeatureVector` | `manager/src/types.rs:152-182` |
| ActionMeta | `struct ActionMeta` | `manager/src/types.rs:185-193` |
| ParamRange | `struct ParamRange` | `manager/src/types.rs:195-200` |
| ActionParams | `struct ActionParams` | `manager/src/types.rs:203-206` |
| ActionCommand | `struct ActionCommand` | `manager/src/types.rs:209-213` |
| Operation | `enum Operation` | `manager/src/types.rs:215-218` |
| ActionRegistry | `struct ActionRegistry` | `manager/src/action_registry.rs:7-10` |

## default_param_range 하드코딩 테이블

코드 위치: `manager/src/action_registry.rs` — `default_param_range()` 함수.

| ActionId | param_name | min | max | 코드 존재 |
|----------|-----------|-----|-----|----------|
| KvEvictSliding | keep_ratio | 0.3 | 0.9 | O |
| KvEvictH2o | keep_ratio | 0.3 | 0.9 | O |
| KvQuantDynamic | target_bits | 4.0 | 8.0 | O |
| LayerSkip | skip_layers | 1.0 | 8.0 | O |
| Throttle | delay_ms | 0.0 | 100.0 | O |
| SwitchHw | (None) | -- | -- | O |
| KvOffloadDisk | (None) | -- | -- | O |
| KvMergeD2o | keep_ratio | 0.3 | 0.9 | **X** (variant 없음) |