# Manager Data Types and Configuration Schema

> **TL;DR**: Manager 내부 데이터 타입, TOML 설정 스키마, SystemSignal 필드의 Manager 내부 소비 관점을 완전 정의한다.
> (1) SystemSignal 4종(MemoryPressure, ThermalAlert, ComputeGuidance, EnergyConstraint)의 전체 필드와 Manager 소비 방식.
> (2) Config TOML 스키마: 최상위 Config, ManagerConfig, MonitorConfig 4종, ExternalMonitorConfig의 키·타입·기본값·범위.
> (3) PolicyConfig TOML 스키마: PiControllerConfig, SupervisoryConfig, SelectorConfig, ReliefModelConfig, ActionConfig, exclusion_groups.
> (4) 핵심 데이터 타입: ActionId 8종, ActionKind, Domain, OperatingMode, PressureVector, ReliefVector, FeatureVector, ActionMeta, ParamRange, ActionParams, ActionCommand.
> (5) ActionRegistry 8종 액션 메타데이터 테이블, default_param_range, default_relief prior.

## 1. Purpose and Scope

이 문서는 Manager 내부 **데이터 타입, TOML 설정 스키마, SystemSignal 필드의 Manager 내부 사용/해석 관점**을 정의한다. `20-manager.md` MGR-014~042, `21-manager-state.md` MGR-073~086, `22-manager-algorithms.md` MGR-ALG-013~014에서 위임된 데이터 정의를 이 파일에서 완성한다.

**이 파일이 명세하는 것:**

- SystemSignal 4종 필드 — Manager가 이를 어떻게 소비·해석하는지 중심
- Config / ManagerConfig / PolicyConfig TOML 스키마 (키 이름, 타입, 기본값, 허용 범위 완전 명시)
- PiControllerConfig, SupervisoryConfig, SelectorConfig, ReliefModelConfig
- MonitorConfig 4종 + ExternalMonitorConfig
- ActionConfig, ActionRegistry 8종 액션 메타데이터
- PressureVector, ReliefVector, FeatureVector 의미
- ActionId, ActionKind, OperatingMode, Domain enum
- ParamRange, ActionMeta, ActionParams, ActionCommand struct
- ReliefEstimator default_relief prior 테이블

**이 파일이 명세하지 않는 것:**

- Manager 아키텍처 개요 → `20-manager.md`
- 상태 머신 전이 테이블 → `21-manager-state.md`
- 알고리즘 수식/의사코드 → `22-manager-algorithms.md`
- 와이어 포맷, 프로토콜 메시지 필드 → `11-protocol-messages.md`

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **SystemSignal** | Monitor가 생성하는 도메인별 시스템 상태 메시지. 4종: MemoryPressure, ThermalAlert, ComputeGuidance, EnergyConstraint. |
| **Level** | 4단계 심각도: Normal < Warning < Critical < Emergency. |
| **Domain** | 압력 도메인: Compute, Memory, Thermal. |
| **OperatingMode** | Manager 운영 모드: Normal < Warning < Critical. Emergency 없음. |
| **PressureVector** | 3차원 압력 벡터 (compute, memory, thermal). 각 원소는 [0, 1] 범위의 pressure intensity. |
| **ReliefVector** | 4차원 완화 효과 벡터 (compute, memory, thermal, latency). 양수는 완화, latency 음수는 악화. |
| **FeatureVector** | 13차원 시스템 상태 벡터. ReliefEstimator 입력. |
| **ActionId** | 적응형 액션 식별자 열거형. 8종. |
| **ActionKind** | 액션 손실 분류: Lossless(품질 유지) / Lossy(품질 저하 수반). |
| **ActionMeta** | 액션 메타데이터: id, kind, reversible, param_range, exclusion_group, default_cost. |
| **ActionRegistry** | 액션 메타데이터 저장소. PolicyConfig로부터 초기화된다. |
| **ParamRange** | 액션 파라미터 범위: param_name, min, max. |
| **GainZone** | PI gain scheduling 구간: above(measurement 하한), kp(해당 구간 비례 이득). |

## 3. Specification

### 3.1 SystemSignal Fields (Manager Consumption) [MGR-DAT-010 ~ MGR-DAT-019]

**[MGR-DAT-010]** SystemSignal은 Monitor Layer가 생성하여 mpsc 채널로 Policy Layer에 전달하는 도메인별 시스템 상태 메시지이다. **raw 센서 데이터만 포함하며, `level` 필드를 포함하지 않는다.** *(MUST)*

> **11번과의 관계**: `11-protocol-messages.md` MSG-100~104는 D-Bus 와이어 포맷을 정의하며, D-Bus 경로에서는 `level` 필드가 포함된다. D-Bus Emitter가 raw 값에서 level을 계산하여 와이어 메시지에 추가한다. 내부 SystemSignal(이 문서)과 D-Bus SystemSignal(11번)은 구조가 다르다.
>
> **설계 원칙**: Monitor는 raw 센서 데이터를 수집·전달하는 것이 유일한 책임이다. 심각도 평가(Level)는 Monitor의 책임이 아니며, D-Bus 전송이 필요한 경우에만 Emitter Layer에서 수행한다. Policy Layer는 raw 필드에서 직접 압력을 계산한다 (MGR-021, MGR-024).

**[MGR-DAT-011]** MemoryPressure — Memory 도메인 상태를 전달한다. *(MUST)*

| 필드 | 타입 | 범위 | Manager 소비 |
|------|------|------|-------------|
| `available_bytes` | u64 | >= 0 | `m = clamp(1 - available_bytes / total_bytes, 0, 1)` → **pressure.memory에 직접 매핑** (PI 미경유, MGR-ALG-013a). |
| `total_bytes` | u64 | > 0 | 정규화 분모. 0이면 m=0 (방어적 처리). |

Memory 압력은 `available_bytes / total_bytes`로 직접 계산하며, PI 평활화를 거치지 않는다 (MGR-ALG-013a).

**[MGR-DAT-012]** ThermalAlert — Thermal 도메인 상태를 전달한다. *(MUST)*

| 필드 | 타��� | 범위 | Manager 소비 |
|------|------|------|-------------|
| `temperature_mc` | i32 | 밀리섭씨 | `m = clamp(temperature_mc / 85000, 0, 1)` → pi_thermal.update(m, dt) (MGR-ALG-014). |
| `throttling_active` | bool | true/false | 로깅/모니터링 목적. |
| `throttle_ratio` | f64 | [0.0, 1.0] | 1.0=스로틀 없음, 0.0=완전 스로틀. |

정규화 기준: 85000mc(85°C)를 1.0으로 매핑.

**[MGR-DAT-013]** ComputeGuidance — Compute 도메인 상태를 전달한다. *(MUST)*

| 필드 | 타입 | 범위 | Manager 소비 |
|------|------|------|-------------|
| `recommended_backend` | RecommendedBackend | Cpu/Gpu/Any | SwitchHw 액션 결정에 향후 활용 가능. |
| `reason` | ComputeReason | 6종 | 로깅 목적. |
| `cpu_usage_pct` | f64 | [0.0, 100.0] | `m_cpu = clamp(cpu_usage_pct / 100, 0, 1)` |
| `gpu_usage_pct` | f64 | [0.0, 100.0] | `m_gpu = clamp(gpu_usage_pct / 100, 0, 1)` |

`m = max(m_cpu, m_gpu)` → pi_compute.update(m, dt) (MGR-ALG-014).

**[MGR-DAT-014]** EnergyConstraint — Energy 상태를 전달한다. 별도 PI 인스턴스 없이 compute PI에 보조 기여한다 (MGR-029). *(MUST)*

| 필드 | 타입 | 범위 | Manager 소비 |
|------|------|------|-------------|
| `reason` | EnergyReason | 6종 | 로깅 목적. |
| `battery_pct` | f64 | [0.0, 100.0] | `m = clamp(1 - battery_pct/100, 0, 1) * 0.5` → `max(pressure.compute, m)` → pi_compute (MGR-ALG-015). *(MUST)* |
| `power_budget_mw` | u32 | >= 0 | 전력 예산. 향후 활용 가능. |

**[MGR-DAT-015]** Level enum — 4단계 심각도. **D-Bus 전송 경로 전용**이다. 내부 SystemSignal에는 포함되지 않는다. D-Bus Emitter가 raw 값에서 ThresholdEvaluator를 통해 Level을 산출하여 D-Bus 와이어 메시지에 포함한다 (`11-protocol-messages.md` MSG-100~104). *(MUST)*

| 값 | 순서 | D-Bus 용도 |
|----|------|-----------|
| Normal | 0 | 정상. |
| Warning | 1 | 경고. Engine Strategy가 참조. |
| Critical | 2 | 위험. Engine Strategy가 참조. |
| Emergency | 3 | 극한. Engine이 자율 Suspend (SYS-055). |

serde: `rename_all = "snake_case"`. Ord derive로 순서가 보장된다.

**[MGR-DAT-016]** 보조 열거형 — ComputeGuidance 및 EnergyConstraint 필드에 사용되는 enum이다. *(MUST)*

RecommendedBackend:

| 값 | 의미 |
|----|------|
| Cpu | CPU 사용 권장 |
| Gpu | GPU 사용 권장 |
| Any | 제한 없음 |

ComputeReason (6종):

| 값 | 의미 |
|----|------|
| CpuBottleneck | CPU 병목 |
| GpuBottleneck | GPU 병목 |
| CpuAvailable | CPU 여유 |
| GpuAvailable | GPU 여유 |
| BothLoaded | 양쪽 모두 부하 |
| Balanced | 균형 |

EnergyReason (6종):

| 값 | 의미 |
|----|------|
| BatteryLow | 배터리 부족 |
| BatteryCritical | 배터리 위험 |
| PowerLimit | 전력 제한 |
| ThermalPower | 열 관련 전력 제한 |
| Charging | 충전 중 |
| None | 제약 없음 |

serde: 모든 보조 열거형은 `rename_all = "snake_case"`.

---

### 3.2 Config TOML Schema [MGR-DAT-020 ~ MGR-DAT-029]

**[MGR-DAT-020]** 최상위 Config 구조 — TOML 파일의 최상위 구조이다. 모든 섹션은 `#[serde(default)]`로 생략 가능하다. *(MUST)*

```toml
[manager]          # ManagerConfig (항상 존재, default 적용)
[memory]           # Option<MemoryMonitorConfig> (생략 시 None)
[thermal]          # Option<ThermalMonitorConfig> (생략 시 None)
[compute]          # Option<ComputeMonitorConfig> (생략 시 None)
[energy]           # Option<EnergyMonitorConfig> (생략 시 None)
[external]         # Option<ExternalMonitorConfig> (생략 시 None)
[policy]           # Option<PolicyConfig> (생략 시 None)
```

- `Config::default()`에서 모든 Optional 섹션은 None이다. ManagerConfig만 Default trait으로 기본값이 적용된다.
- `Config::from_file(path)`: TOML 파일에서 로드한다. 파싱 실패 시 anyhow::Result를 전파한다 (MGR-DAT-C02).

**[MGR-DAT-021]** ManagerConfig — 전역 기본 설정. *(MUST)*

| TOML 키 | 타입 | 기본값 | 범위 | 설명 |
|---------|------|--------|------|------|
| `poll_interval_ms` | u64 | 1000 | > 0 | 기본 폴링 주기 (밀리초). Monitor별 개별 설정이 없을 때 이 값을 사용. |

**[MGR-DAT-022]** MemoryMonitorConfig — Memory 도메인 Monitor 설정. 임계값 방향은 **Descending** (값이 낮을수록 심각). *(MUST)*

| TOML 키 | 타입 | 기본값 | 범위 | 설명 |
|---------|------|--------|------|------|
| `enabled` | bool | true | -- | Monitor 활성화 여부 |
| `poll_interval_ms` | Option\<u64\> | None | > 0 | 개별 폴링 주기. None이면 ManagerConfig.poll_interval_ms 사용. **Memory는 100ms 이하 권장** (MGR-022). |
| `warning_pct` | f64 | 40.0 | (0, 100) | 가용 메모리 퍼센트 이하 시 Warning |
| `critical_pct` | f64 | 20.0 | (0, 100) | 가용 메모리 퍼센트 이하 시 Critical |
| `emergency_pct` | f64 | 10.0 | (0, 100) | 가용 메모리 퍼센트 이하 시 Emergency |
| `hysteresis_pct` | f64 | 5.0 | > 0 | 회복 오프셋. Descending: threshold + hysteresis 초과 시 회복. |

불변식: `emergency_pct < critical_pct < warning_pct`.

**[MGR-DAT-023]** ThermalMonitorConfig — Thermal 도메인 Monitor 설정. 임계값 방향은 **Ascending** (값이 높을수록 심각). *(MUST)*

| TOML 키 | 타입 | 기본값 | 범위 | 설명 |
|---------|------|--------|------|------|
| `enabled` | bool | true | -- | Monitor 활성화 여부 |
| `poll_interval_ms` | Option\<u64\> | None | > 0 | 개별 폴링 주기. None이면 ManagerConfig.poll_interval_ms 사용. |
| `zone_types` | Vec\<String\> | [] (빈 배열) | -- | 모니터링할 thermal zone 타입 필터. 빈 배열이면 모든 zone 사용. |
| `warning_mc` | i32 | 60000 | > 0 | 밀리섭씨 이상 시 Warning |
| `critical_mc` | i32 | 75000 | > warning_mc | 밀리섭씨 이상 시 Critical |
| `emergency_mc` | i32 | 85000 | > critical_mc | 밀리섭씨 이상 시 Emergency |
| `hysteresis_mc` | i32 | 5000 | > 0 | 회복 오프셋. Ascending: threshold - hysteresis 이하 시 회복. |

불변식: `warning_mc < critical_mc < emergency_mc`.

**[MGR-DAT-024]** ComputeMonitorConfig — Compute 도메인 Monitor 설정. 임계값 방향은 **Ascending**. **Emergency 레벨이 없다.** *(MUST)*

| TOML 키 | 타입 | 기본값 | 범위 | 설명 |
|---------|------|--------|------|------|
| `enabled` | bool | true | -- | Monitor 활성화 여부 |
| `poll_interval_ms` | Option\<u64\> | None | > 0 | 개별 폴링 주기. None이면 ManagerConfig.poll_interval_ms 사용. |
| `warning_pct` | f64 | 70.0 | (0, 100) | CPU/GPU 사용률 퍼센트 이상 시 Warning |
| `critical_pct` | f64 | 90.0 | (0, 100) | CPU/GPU 사용률 퍼센트 이상 시 Critical |
| `hysteresis_pct` | f64 | 5.0 | > 0 | 회복 오프셋 |

Emergency threshold는 코드에서 f64::MAX로 설정되어 도달 불가하다. ComputeMonitorConfig에 emergency 필드는 존재하지 않는다.

불변식: `warning_pct < critical_pct`.

**[MGR-DAT-025]** EnergyMonitorConfig — Energy 도메인 Monitor 설정. 임계값 방향은 **Descending** (배터리 퍼센트 기준). *(MUST)*

| TOML 키 | 타입 | 기본값 | 범위 | 설명 |
|---------|------|--------|------|------|
| `enabled` | bool | true | -- | Monitor 활성화 여부 |
| `poll_interval_ms` | Option\<u64\> | None | > 0 | 개별 폴링 주기. None이면 ManagerConfig.poll_interval_ms 사용. |
| `warning_pct` | f64 | 30.0 | (0, 100) | 배터리 퍼센트 이하 시 Warning |
| `critical_pct` | f64 | 15.0 | (0, 100) | 배터리 퍼센트 이하 시 Critical |
| `emergency_pct` | f64 | 5.0 | (0, 100) | 배터리 퍼센트 이하 시 Emergency |
| `warning_power_budget_mw` | u32 | 3000 | > 0 | Warning 레벨 전력 예산 (밀리와트) |
| `critical_power_budget_mw` | u32 | 1500 | > 0 | Critical 레벨 전력 예산 |
| `emergency_power_budget_mw` | u32 | 500 | > 0 | Emergency 레벨 전력 예산 |
| `ignore_when_charging` | bool | true | -- | 충전 중 항상 Normal Level 유지 |

Energy hysteresis는 **TOML에서 설정할 수 없다**. EnergyMonitor 내부에서 2.0(%)으로 하드코딩되어 있다.

불변식: `emergency_pct < critical_pct < warning_pct`.

**[MGR-DAT-026]** ExternalMonitorConfig — 외부 신호 주입을 위한 연구/테스트 전용 Monitor 설정. *(MUST)*

| TOML 키 | 타입 | 기본값 | 범위 | 설명 |
|---------|------|--------|------|------|
| `enabled` | bool | false | -- | Monitor 활성화 여부 |
| `transport` | String | "stdin" | "stdin" 또는 "unix:\<path\>" | 외부 신호 수신 전송 매체 |

연구/테스트 용도 (MGR-020).

---

### 3.3 PolicyConfig TOML Schema [MGR-DAT-030 ~ MGR-DAT-039]

**[MGR-DAT-030]** PolicyConfig 구조 — `[policy]` 섹션 아래의 계층형 설정이다. *(MUST)*

```toml
[policy.pi_controller]     # PiControllerConfig
[policy.supervisory]        # SupervisoryConfig
[policy.selector]           # SelectorConfig
[policy.relief_model]       # ReliefModelConfig
[policy.actions.<name>]     # HashMap<String, ActionConfig>
[policy.exclusion_groups]   # HashMap<String, Vec<String>>
```

설정 로딩 우선순위: (1) `--policy-config` CLI 인수 파일, (2) config.toml의 `[policy]` 섹션, (3) PolicyConfig::default() (MGR-042).

**[MGR-DAT-031]** PiControllerConfig — PI Controller 도메인별 파라미터. *(MUST)*

| TOML 키 | 타입 | 기본값 | 범위 | 설명 |
|---------|------|--------|------|------|
| `compute_kp` | f32 | 1.5 | > 0 | Compute PI 비례 이득 |
| `compute_ki` | f32 | 0.3 | >= 0 | Compute PI 적분 이득 |
| `compute_setpoint` | f32 | 0.70 | [0, 1] | Compute PI setpoint |
| `memory_kp` | f32 | 2.0 | > 0 | Memory PI 비례 이득 (비사용, 아래 참고) |
| `memory_ki` | f32 | 0.5 | >= 0 | Memory PI 적분 이득 (비사용, 아래 참고) |
| `memory_setpoint` | f32 | 0.75 | [0, 1] | Memory PI setpoint (비사용, 아래 참고) |
| `thermal_kp` | f32 | 1.0 | > 0 | Thermal PI 비례 이득 |
| `thermal_ki` | f32 | 0.2 | >= 0 | Thermal PI 적분 이득 |
| `thermal_setpoint` | f32 | 0.80 | [0, 1] | Thermal PI setpoint |
| `integral_clamp` | f32 | 2.0 | > 0 | 적분항 상한 (모든 PI 인스턴스 공유) |
| `memory_gain_zones` | Vec\<GainZone\> | [] (빈 배열) | -- | Memory 도메인 gain scheduling 구간 (비사용, 아래 참고) |

> **비규범 주석 (Memory PI 필드)**: `memory_kp`, `memory_ki`, `memory_setpoint`, `memory_gain_zones` 필드는 PiControllerConfig에 정의되어 있으나, **Memory 도메인은 PI Controller를 사용하지 않는다** (MGR-ALG-013a). Memory는 임계값 기반 직접 매핑을 사용하며, 이 필드들은 PI 인스턴스 생성 시 초기화되지만 update()가 호출되지 않는다. 초기 설계에서 3개 도메인 모두 PI Controller를 사용했으나, Memory 도메인의 OOM 즉각 대응 필요성이 확인되어 직접 매핑으로 전환했다. 기존 설정 호환성을 위해 필드를 유지한다.

GainZone struct:

| 필드 | 타입 | 설명 |
|------|------|------|
| `above` | f32 | measurement가 이 값 이상일 때 해당 kp 적용 |
| `kp` | f32 | 해당 구간의 비례 이득 |

gain_zones는 `above` 기준 오름차순 정렬되어야 한다 (MUST). 현재 memory 도메인에만 정의 가능하나, Memory PI가 비활성이므로 실질적으로 미사용이다.

**[MGR-DAT-032]** SupervisoryConfig — Supervisory 모드 전환 임계값 설정. *(MUST)*

| TOML 키 | 타입 | 기본값 | 범위 | 설명 |
|---------|------|--------|------|------|
| `warning_threshold` | f32 | 0.4 | (0, 1) | peak pressure >= 이 값이면 Warning 진입 |
| `critical_threshold` | f32 | 0.7 | (0, 1) | peak pressure >= 이 값이면 Critical 진입 |
| `warning_release` | f32 | 0.25 | (0, 1) | peak < 이 값이 hold_time 지속 시 Normal 복귀 |
| `critical_release` | f32 | 0.50 | (0, 1) | peak < 이 값이 hold_time 지속 시 Warning 복귀 |
| `hold_time_secs` | f32 | 4.0 | > 0 | 디에스컬레이션 안정화 대기 시간 (초) |

불변식:
- `warning_release < warning_threshold` *(MUST)*
- `critical_release < critical_threshold` *(MUST)*
- `warning_threshold < critical_threshold` *(MUST)*

**[MGR-DAT-033]** SelectorConfig — Action Selector 설정. *(MUST)*

| TOML 키 | 타입 | 기본값 | 범위 | 설명 |
|---------|------|--------|------|------|
| `latency_budget` | f32 | 0.5 | > 0 | 조합의 총 latency 악화 허용 한도. `total_relief.latency < -latency_budget`이면 해당 조합 제외 (MGR-ALG-033). |
| `algorithm` | String | "exhaustive" | "exhaustive" | 탐색 알고리즘. 현재 전수 탐색만 지원. |

**[MGR-DAT-034]** ReliefModelConfig — Relief Estimator 모델 설정. *(MUST)*

| TOML 키 | 타입 | 기본값 | 범위 | 설명 |
|---------|------|--------|------|------|
| `forgetting_factor` | f32 | 0.995 | (0, 1] | RLS forgetting factor. 1에 가까울수록 과거 데이터 가중치 유지. |
| `prior_weight` | u32 | 5 | > 0 | Prior 정규화 가중치. P 행렬 초기값 (prior_weight * I). |
| `storage_dir` | String | "~/.llm_rs/models" | 유효 경로 | 모델 영속화 디렉토리 (MGR-048). |

**[MGR-DAT-035]** ActionConfig — 액션별 메타데이터 설정. `[policy.actions.<action_name>]` 형식으로 개별 액션을 설정한다. *(MUST)*

| TOML 키 | 타입 | 기본값 | 범위 | 설명 |
|---------|------|--------|------|------|
| `lossy` | bool | false | -- | true이면 Lossy (ActionKind::Lossy), false이면 Lossless (ActionKind::Lossless) |
| `reversible` | bool | false | -- | 가역 여부. true이면 Release 가능. |
| `default_cost` | f32 | 1.0 | >= 0 | QCF 값 미수신 시 사용되는 기본 비용. ActionRegistry.default_cost()를 통해 접근. |

`action_name`은 ActionId의 snake_case 문자열이다: `switch_hw`, `throttle`, `kv_offload_disk`, `kv_evict_sliding`, `kv_evict_h2o`, `kv_merge_d2o`, `kv_quant_dynamic`, `layer_skip`.

알 수 없는 `action_name`은 무시된다 (ActionId::from_str() 실패 시 skip).

**[MGR-DAT-036]** exclusion_groups — 배타 그룹 설정. `[policy.exclusion_groups]` 섹션에서 정의한다. 키는 그룹 이름, 값은 action_name 문자열 배열이다. *(MUST)*

```toml
[policy.exclusion_groups]
eviction = ["kv_evict_sliding", "kv_evict_h2o", "kv_merge_d2o"]
```

동일 배타 그룹의 액션은 동시에 선택되지 않는다 (MGR-ALG-033). 알 수 없는 action_name은 무시된다.

---

### 3.4 Core Data Types [MGR-DAT-040 ~ MGR-DAT-049]

**[MGR-DAT-040]** ActionId enum — 적응형 액션 식별자. 8종을 정의한다. *(MUST)*

| 값 | snake_case | primary_domain | 코드 구현 상태 |
|----|-----------|---------------|--------------|
| SwitchHw | `switch_hw` | Compute | 구현 |
| Throttle | `throttle` | Compute | 구현 |
| KvOffloadDisk | `kv_offload_disk` | Memory | 구현 |
| KvEvictSliding | `kv_evict_sliding` | Memory | 구현 |
| KvEvictH2o | `kv_evict_h2o` | Memory | 구현 |
| KvMergeD2o | `kv_merge_d2o` | Memory | **미구현** (스펙 전용) |
| KvQuantDynamic | `kv_quant_dynamic` | Memory | 구현 |
| LayerSkip | `layer_skip` | Compute | 구현 |

- serde: `rename_all = "snake_case"`. JSON 직렬화/역직렬화 시 snake_case 문자열 사용.
- `ActionId::from_str(s)`: snake_case 문자열 → ActionId 변환. 실패 시 None.
- `ActionId::all()`: 현재 코드는 7종 반환 (KvMergeD2o 제외). 스펙은 8종을 정의한다.
- `ActionId::primary_domain()`: SwitchHw / Throttle / LayerSkip → Compute, 나머지 → Memory.

> **코드-스펙 차이**: `types.rs`의 ActionId enum에 KvMergeD2o variant가 없다. D2O (Dynamic Discriminative Operations)는 구현 복잡도가 높아 Engine 측 KV cache merging 로직이 미완성이다. 프로토콜(`11-protocol-messages.md` MSG-034b)과 `20-manager.md` MGR-028에서 정의되어 있으며, 스펙에서 8종을 유지하여 향후 구현 시 스펙 변경 없이 코드를 추가할 수 있게 한다.

**[MGR-DAT-041]** ActionKind enum — 액션의 품질 영향 분류. *(MUST)*

| 값 | 의미 |
|----|------|
| Lossless | 품질 저하 없이 리소스 완화. cost = 0. |
| Lossy | 품질 저하를 수반. cost = QCF 값 또는 default_cost. |

serde: `rename_all = "snake_case"`. Warning 모드에서 Lossy 액션은 선택되지 않는다.

**[MGR-DAT-042]** Domain enum — 압력 도메인. *(MUST)*

| 값 | 의미 | PressureVector 대응 |
|----|------|-------------------|
| Compute | 연산 부하 | pressure.compute |
| Memory | 메모리 압박 | pressure.memory |
| Thermal | 열 상태 | pressure.thermal |

serde derive 없음 (내부 전용). Energy는 별도 Domain이 아니며, compute에 합산된다 (MGR-029).

**[MGR-DAT-043]** OperatingMode enum — Manager 운영 모드. *(MUST)*

| 값 | 순서 | 허용 액션 종류 |
|----|------|-------------|
| Normal | 0 | 없음 (액션 발행 안 함) |
| Warning | 1 | Lossless만 |
| Critical | 2 | Lossless + Lossy |

PartialOrd/Ord derive: Normal < Warning < Critical. Emergency 없음. Monitor의 Emergency Level은 Supervisory에서 Critical로 처리된다.

**[MGR-DAT-044]** PressureVector — 3차원 압력 벡터. *(MUST)*

| 필드 | 타입 | 범위 | 소스 |
|------|------|------|------|
| `compute` | f32 | [0, 1] | PI Controller 출력 (pi_compute) |
| `memory` | f32 | [0, 1] | 임계값 기반 직접 매핑 (MGR-ALG-013a) |
| `thermal` | f32 | [0, 1] | PI Controller 출력 (pi_thermal) |

메서드:

| 메서드 | 반환 | 설명 |
|--------|------|------|
| `max()` | f32 | 3개 필드 중 최대값 (= peak pressure) |
| `any_domain_exceeds(reference, factor)` | bool | 하나라도 `reference * factor` 초과 시 true |

연산: `PressureVector - PressureVector` → `ReliefVector` (latency = 0.0).

Default: 모든 필드 0.0.

**[MGR-DAT-045]** ReliefVector — 4차원 완화 효과 벡터. *(MUST)*

| 필드 | 타입 | 범위 | 의미 |
|------|------|------|------|
| `compute` | f32 | 양수=완화 | compute 도메인 압력 감소 효과 |
| `memory` | f32 | 양수=완화 | memory 도메인 압력 감소 효과 |
| `thermal` | f32 | 양수=완화 | thermal 도메인 압력 감소 효과 |
| `latency` | f32 | 음수=악화 | 지연 시간 영향. 음수이면 latency 증가(악화). |

Default: 모든 필드 0.0. Add/AddAssign: 필드별 합산. ActionSelector에서 조합의 총 relief를 계산할 때 사용.

serde: Serialize + Deserialize.

**[MGR-DAT-046]** FeatureVector — 13차원 시스템 상태 벡터. ReliefEstimator 입력. *(MUST)*

FEATURE_DIM = 13. `values: [f32; 13]`.

| 인덱스 | 상수명 | 의미 | 값 범위 |
|--------|--------|------|--------|
| 0 | KV_OCCUPANCY | KV 캐시 점유율 | [0, 1] |
| 1 | IS_GPU | GPU 활성 여부 | 0.0 또는 1.0 |
| 2 | TOKEN_PROGRESS | 토큰 진행률 | [0, ~] |
| 3 | IS_PREFILL | Prefill 단계 여부 | (미사용, 0.0) |
| 4 | KV_DTYPE_NORM | KV dtype 정규화 | (미사용, 0.0) |
| 5 | TBT_RATIO | 처리량 비율 | [0, ~] |
| 6 | TOKENS_GENERATED_NORM | 생성 토큰 수 정규화 | [0, ~] |
| 7 | ACTIVE_SWITCH_HW | SwitchHw 활성 | (미사용, 0.0) |
| 8 | ACTIVE_THROTTLE | Throttle 활성 | (미사용, 0.0) |
| 9 | ACTIVE_KV_OFFLOAD | KvOffload 활성 | (미사용, 0.0) |
| 10 | ACTIVE_EVICTION | Eviction 활성 | 0.0 또는 1.0 |
| 11 | ACTIVE_LAYER_SKIP | LayerSkip 활성 | 0.0 또는 1.0 |
| 12 | ACTIVE_KV_QUANT | KvQuant 활성 | (미사용, 0.0) |

인덱스 매핑 상세(EngineStatus → FeatureVector)는 `21-manager-state.md` MGR-080에서 정의한다. 이 문서에서는 각 인덱스의 **의미**만 기술한다.

---

### 3.5 ActionMeta and ActionRegistry [MGR-DAT-050 ~ MGR-DAT-059]

**[MGR-DAT-050]** ActionMeta struct — 액션 메타데이터. *(MUST)*

| 필드 | 타입 | 설명 |
|------|------|------|
| `id` | ActionId | 액션 식별자 |
| `kind` | ActionKind | Lossless / Lossy |
| `reversible` | bool | 가역 여부 |
| `param_range` | Option\<ParamRange\> | 파라미터 범위. None이면 파라미터 없는 액션. |
| `exclusion_group` | Option\<String\> | 배타 그룹 이름. None이면 배타 제약 없음. |
| `default_cost` | f32 | 기본 QCF 비용 |

**[MGR-DAT-051]** ParamRange struct — 액션 파라미터의 허용 범위. *(MUST)*

| 필드 | 타입 | 설명 |
|------|------|------|
| `param_name` | String | 파라미터 이름 (예: "keep_ratio", "delay_ms") |
| `min` | f32 | 파라미터 최소값 |
| `max` | f32 | 파라미터 최대값 |

serde: Serialize + Deserialize.

**[MGR-DAT-052]** ActionParams struct — 액션 파라미터 값. Directive 생성 시 ActionSelector가 결정한 파라미터를 담는다. *(MUST)*

| 필드 | 타입 | 설명 |
|------|------|------|
| `values` | HashMap\<String, f32\> | 파라미터 이름-값 쌍 |

serde: Serialize + Deserialize. Default: 빈 HashMap.

**[MGR-DAT-053]** ActionCommand struct — ActionSelector 출력. *(MUST)*

| 필드 | 타입 | 설명 |
|------|------|------|
| `action` | ActionId | 대상 액션 |
| `operation` | Operation | Apply(ActionParams) 또는 Release |

Operation enum:

| 변형 | 설명 |
|------|------|
| Apply(ActionParams) | 액션 적용 (파라미터 포함) |
| Release | 액션 해제 |

**[MGR-DAT-054]** ActionRegistry 8종 액션 메타데이터 테이블 — ActionRegistry는 `from_config()`로 PolicyConfig에서 초기화된다. 설정에 포함된 액션만 등록된다 (최대 8종). 아래는 권장 기본 메타데이터이다. *(MUST)*

| ActionId | kind | reversible | param_name | param min | param max | 배타 그룹 | primary_domain | default_cost |
|----------|------|------------|------------|-----------|-----------|----------|---------------|-------------|
| SwitchHw | Lossless | true | (없음) | -- | -- | (없음) | Compute | 1.0 |
| Throttle | Lossless | true | delay_ms | 0.0 | 100.0 | (없음) | Compute | 1.0 |
| KvOffloadDisk | Lossless | false | (없음) | -- | -- | (없음) | Memory | 1.0 |
| KvEvictSliding | Lossy | false | keep_ratio | 0.3 | 0.9 | eviction | Memory | 1.0 |
| KvEvictH2o | Lossy | false | keep_ratio | 0.3 | 0.9 | eviction | Memory | 1.0 |
| KvMergeD2o | Lossy | false | keep_ratio | 0.3 | 0.9 | eviction | Memory | 1.0 |
| KvQuantDynamic | Lossy | false | target_bits | 4.0 | 8.0 | (없음) | Memory | 1.0 |
| LayerSkip | Lossy | true | skip_layers | 1.0 | 8.0 | (없음) | Compute | 1.0 |

> kind, reversible, default_cost는 TOML ActionConfig에서 설정. param_range는 `default_param_range()`에서 하드코딩. 배타 그룹은 TOML `exclusion_groups`에서 설정.

**[MGR-DAT-055]** ActionRegistry 생성 (from_config) — PolicyConfig로부터 ActionRegistry를 구성하는 절차이다. *(MUST)*

1. `PolicyConfig.actions`의 각 `(name, ActionConfig)` 엔트리를 순회한다.
2. `ActionId::from_str(name)` 실패 시 skip한다.
3. `ActionConfig.lossy` → ActionKind 변환: true → Lossy, false → Lossless.
4. `default_param_range(id)` 하드코딩 범위를 적용한다.
5. `exclusion_group`은 초기 None으로 설정한다.
6. `PolicyConfig.exclusion_groups`의 각 `(group_name, members)` 엔트리를 순회하여, members의 ActionId를 파싱하고 해당 ActionMeta.exclusion_group에 group_name을 설정한다.

**[MGR-DAT-056]** default_param_range 하드코딩 테이블 — 액션별 기본 파라미터 범위이다. *(MUST)*

| ActionId | param_name | min | max |
|----------|-----------|-----|-----|
| KvEvictSliding | keep_ratio | 0.3 | 0.9 |
| KvEvictH2o | keep_ratio | 0.3 | 0.9 |
| KvQuantDynamic | target_bits | 4.0 | 8.0 |
| LayerSkip | skip_layers | 1.0 | 8.0 |
| Throttle | delay_ms | 0.0 | 100.0 |
| SwitchHw | (없음 -- None) | -- | -- |
| KvOffloadDisk | (없음 -- None) | -- | -- |
| KvMergeD2o | keep_ratio | 0.3 | 0.9 |

SwitchHw와 KvOffloadDisk는 파라미터 없는 액션이다 (param_range = None).

KvMergeD2o는 코드에 ActionId variant가 없으므로 `default_param_range()`에 분기가 없다. 스펙에서 keep_ratio [0.3, 0.9]을 정의한다.

> 파라미터 범위는 Engine의 물리적 제약에 의해 결정된다 (keep_ratio 0.3 미만이면 품질 붕괴, target_bits 4 미만은 양자화 정밀도 한계 등). 향후 TOML 설정 가능하게 확장할 수 있다.

**[MGR-DAT-057]** ActionRegistry 주요 메서드. *(MUST)*

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `get` | `get(&ActionId) -> Option<&ActionMeta>` | 액션 메타 조회 |
| `all_actions` | `all_actions() -> impl Iterator<Item = &ActionMeta>` | 등록된 모든 액션 순회 |
| `lossy_actions` | `lossy_actions() -> Vec<ActionId>` | Lossy 액션 목록 |
| `lossless_actions` | `lossless_actions() -> Vec<ActionId>` | Lossless 액션 목록 |
| `exclusion_groups` | `exclusion_groups() -> &HashMap<String, Vec<ActionId>>` | 배타 그룹 맵 |
| `is_excluded` | `is_excluded(&ActionId, &ActionId) -> bool` | 두 액션의 배타 관계 확인 |
| `default_cost` | `default_cost(&ActionId) -> f32` | 기본 QCF 비용. 미등록 액션은 1.0. |

---

### 3.6 ReliefEstimator Default Relief Prior [MGR-DAT-060]

**[MGR-DAT-060]** ReliefEstimator가 Absent/Initialized 상태일 때 predict()가 반환하는 기본 ReliefVector이다 (MGR-086 재확인). *(MUST)*

| ActionId | compute | memory | thermal | latency |
|----------|---------|--------|---------|---------|
| SwitchHw | 0.5 | 0.0 | 0.3 | -0.1 |
| Throttle | 0.3 | 0.0 | 0.2 | -0.3 |
| KvOffloadDisk | 0.0 | 0.4 | 0.0 | -0.2 |
| KvEvictSliding | 0.0 | 0.7 | 0.0 | 0.0 |
| KvEvictH2o | 0.0 | 0.6 | 0.0 | 0.0 |
| KvMergeD2o | 0.0 | 0.6 | 0.0 | 0.0 |
| KvQuantDynamic | 0.0 | 0.3 | 0.0 | 0.0 |
| LayerSkip | 0.3 | 0.0 | 0.1 | -0.2 |

이 테이블은 `21-manager-state.md` MGR-086과 동일하다. 데이터 관점에서 재기술한다.

---

## 4. Alternative Behavior

해당 없음. 이 문서는 데이터 정의 문서이다. 데이터 처리의 대안 동작은 해당 알고리즘 스펙(`22-manager-algorithms.md`)에서 다룬다.

## 5. Constraints

**[MGR-DAT-C01]** PolicyConfig가 None인 경우(TOML에서 `[policy]` 섹션 미지정), PolicyConfig::default()가 적용된다. 이 경우 actions HashMap이 비어 있으므로 ActionRegistry에 액션이 등록되지 않아 ActionSelector가 빈 조합을 반환한다. *(MUST)*

**[MGR-DAT-C02]** TOML 파싱 실패 시 Manager 프로세스가 시작하지 않는다 (anyhow::Result 전파). *(MUST)*

## 6. Examples

### 6.1 전체 TOML 설정 예시

```toml
[manager]
poll_interval_ms = 500

[memory]
enabled = true
poll_interval_ms = 100
warning_pct = 40.0
critical_pct = 20.0
emergency_pct = 10.0
hysteresis_pct = 5.0

[thermal]
enabled = true
zone_types = ["x86_pkg_temp"]
warning_mc = 60000
critical_mc = 75000
emergency_mc = 85000
hysteresis_mc = 5000

[compute]
enabled = true
warning_pct = 70.0
critical_pct = 90.0
hysteresis_pct = 5.0

[energy]
enabled = true
warning_pct = 30.0
critical_pct = 15.0
emergency_pct = 5.0
warning_power_budget_mw = 3000
critical_power_budget_mw = 1500
emergency_power_budget_mw = 500
ignore_when_charging = true

[external]
enabled = false
transport = "stdin"

[policy.pi_controller]
compute_kp = 1.5
compute_ki = 0.3
compute_setpoint = 0.70
thermal_kp = 1.0
thermal_ki = 0.2
thermal_setpoint = 0.80
integral_clamp = 2.0

[policy.supervisory]
warning_threshold = 0.4
critical_threshold = 0.7
warning_release = 0.25
critical_release = 0.50
hold_time_secs = 4.0

[policy.selector]
latency_budget = 0.5
algorithm = "exhaustive"

[policy.relief_model]
forgetting_factor = 0.995
prior_weight = 5
storage_dir = "~/.llm_rs/models"

[policy.actions.switch_hw]
lossy = false
reversible = true
default_cost = 1.0

[policy.actions.throttle]
lossy = false
reversible = true
default_cost = 1.0

[policy.actions.kv_offload_disk]
lossy = false
reversible = false
default_cost = 1.0

[policy.actions.kv_evict_sliding]
lossy = true
reversible = false
default_cost = 1.0

[policy.actions.kv_evict_h2o]
lossy = true
reversible = false
default_cost = 1.0

[policy.actions.kv_quant_dynamic]
lossy = true
reversible = false
default_cost = 1.0

[policy.actions.layer_skip]
lossy = true
reversible = true
default_cost = 1.0

[policy.exclusion_groups]
eviction = ["kv_evict_sliding", "kv_evict_h2o", "kv_merge_d2o"]
```

### 6.2 최소 설정 (모든 default 사용)

```toml
[manager]
poll_interval_ms = 500

[memory]
enabled = true
```

모든 Optional 섹션은 생략 가능하며, 지정 시 각 필드도 개별 생략 가능하다 (`#[serde(default)]`).

## 7. Rationale (non-normative)

### 왜 Memory PI 필드가 PiControllerConfig에 잔존하는가

초기 설계에서 3개 도메인 모두 PI Controller를 사용했으나, Memory 도메인의 OOM 즉각 대응 필요성이 확인되어 임계값 기반 직접 매핑으로 전환했다 (MGR-ALG-013a). 기존 설정 호환성을 위해 필드를 제거하지 않았다.

### 왜 Energy hysteresis가 하드코딩인가

배터리 잔량은 이산적(1% 단위)이므로 사용자 설정 오류 가능성이 높다. 2% 고정값이 대부분의 디바이스에서 적절하다.

### 왜 default_param_range가 하드코딩인가

파라미터 범위는 Engine의 물리적 제약에 의해 결정된다 (keep_ratio 0.3 미만이면 품질 붕괴, target_bits 4 미만은 양자화 정밀도 한계 등). 사용자가 임의 범위를 설정하면 Engine이 Rejected를 반환할 가능성이 높다. 향후 TOML 설정 가능하게 확장할 수 있다.

### 왜 ActionId에 KvMergeD2o가 코드에 없는가

D2O (Dynamic Discriminative Operations)는 구현 복잡도가 높다. 프로토콜(`11-protocol-messages.md` MSG-034b)과 `20-manager.md` MGR-028에는 정의되어 있으나, Engine 측 KV cache merging 로직이 미완성이다. 스펙에서 8종을 유지하여 향후 구현 시 스펙 변경 없이 코드를 추가할 수 있게 한다.

### default_relief prior 값의 한계

MGR-DAT-060의 default_relief 값은 도메인 전문 지식 기반 초기 추정치이다. Prior 결정 전략(실측 calibration, 모델별 프리셋 등)은 미정이며, 현재 값과 S25 실측 간 괴리가 존재한다 (SwitchHw memory prior=0.0 vs 실측 -810MB, Eviction prior=0.6 vs null 등). 상세 분석과 C3 온라인 학습에 의한 보정 설계는 `21-manager-state.md` §7 Rationale "default_relief prior 값의 출처와 한계" 참조.

### 왜 Energy는 별도 Domain이 아닌가

Energy 상태는 독립된 압력 차원으로 모델링하기보다, 기존 compute 압력에 floor를 적용하는 방식이 효과적이다. 배터리 부족 시 연산 부하를 낮추는 것이 주된 대응이며, 이는 compute 도메인의 압력 증가와 동일한 효과를 갖기 때문이다 (MGR-029).
