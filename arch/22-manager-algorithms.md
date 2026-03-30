# Manager Algorithms -- Architecture

> spec/22-manager-algorithms.md의 구현 상세.

## 코드 매핑

### 3.1 PI Controller

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-ALG-010 | `manager/src/pi_controller.rs` | `PiController` struct | 단일 도메인용 PI |
| MGR-ALG-010 | `manager/src/pipeline.rs:189-229` | `update_pressure()` — 도메인별 분기 | Memory는 PI 경유하나 직접 매핑으로 전환 필요 (아래 차이 참고) |
| MGR-ALG-011 | `manager/src/pi_controller.rs:50-58` | `effective_kp()` 메서드 | `gain_zones` 역순 탐색 |
| MGR-ALG-012 | `manager/src/pi_controller.rs` | `can_act` 플래그 + `integral_clamp` | Anti-windup 이중 메커니즘 |
| MGR-ALG-013 | `manager/src/pipeline.rs:111-129` | `pi_compute`, `pi_thermal` 인스턴스 생성 | `PiControllerConfig` 기반 |
| MGR-ALG-013a | `manager/src/pipeline.rs:191-203` | Memory 처리 | **코드-스펙 차이**: 코드는 `pi_memory.update(m, dt)` 호출 (PI 경유). 스펙은 직접 매핑 (`pressure.memory = m`) |
| MGR-ALG-014 | `manager/src/pipeline.rs:189-229` | `update_pressure()` — SystemSignal 매칭 | 도메인별 정규화 변환 |
| MGR-ALG-015 | `manager/src/pipeline.rs:221-228` | EnergyConstraint 처리 | **코드-스펙 차이**: `level_to_measurement(level)` 사용. 스펙은 `battery_pct` raw 값 사용 |
| MGR-ALG-016 | `manager/src/pipeline.rs:177-186` | `elapsed_dt()` 메서드 | 도메인별 독립 dt, clamp [0.001, 10.0], 기본 0.1s |
| INV-030 | `manager/src/pi_controller.rs` | `can_act == false` 시 integral 갱신 skip | `if self.can_act { integral += ... }` |
| INV-031 | `manager/src/pi_controller.rs` | `integral.clamp(0.0, self.integral_clamp)` | update() 내부 |

### 3.2 Supervisory Judgment Logic

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-ALG-020 | `manager/src/supervisory.rs` | `evaluate(&PressureVector) -> OperatingMode` | peak pressure 기반 |
| MGR-ALG-021 | `manager/src/supervisory.rs` | `next_mode()` — match `current_mode` | 전이 테이블 구현 |
| MGR-ALG-022 | `manager/src/types.rs:91-93` | `PressureVector::max()` | `compute.max(memory).max(thermal)` |
| MGR-ALG-023 | `manager/src/config.rs:234-244` | `SupervisoryConfig::default()` | 기본값 일치 확인 |
| MGR-ALG-024 | `manager/src/pipeline.rs:312-320` | `needs_action` 판정 | Normal→false, else mode_changed OR 1.2x |
| INV-034 | `manager/src/config.rs:234-244` | Default 값으로 보증 | `warning_release(0.25) < warning_threshold(0.4)` |
| INV-035 | `manager/src/config.rs:234-244` | Default 값으로 보증 | `critical_release(0.50) < critical_threshold(0.7)` |
| INV-036 | `manager/src/config.rs:234-244` | Default 값으로 보증 | `warning_threshold(0.4) < critical_threshold(0.7)` |

### 3.3 ActionSelector Exhaustive Search

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-ALG-030 | `manager/src/selector.rs` | `ActionSelector::select()` | 4-phase: filter → cost → find_optimal → parametrize |
| MGR-ALG-031 | `manager/src/selector.rs` | `filter_candidates()` | Warning: Lossy 제외, 이미 활성 제외, available 필터 |
| MGR-ALG-032 | `manager/src/selector.rs` | `compute_cost()` | Lossless=0, Lossy=QCF or INFINITY |
| MGR-ALG-033 | `manager/src/selector.rs` | `find_optimal()` — 2^N mask 순회 | exclusion conflict + latency budget + cost 최소화 |
| MGR-ALG-034 | `manager/src/selector.rs` | `has_exclusion_conflict()` | O(N^2) 쌍별 검사 |
| MGR-ALG-035 | `manager/src/selector.rs` | `parametrize()` | 선형 보간: `max - intensity * (max - min)` |
| MGR-ALG-036 | `manager/src/pipeline.rs:483-521` | `action_to_engine_command()` | ActionCommand → EngineCommand 변환 |
| MGR-ALG-037 | (상수) | O(2^N * N^2), N <= 8 → 상수 시간 | 최대 16384 연산 |
| INV-037 | `manager/src/selector.rs` | `filter_candidates()` — mode==Warning 시 Lossy continue | |
| INV-038 | `manager/src/selector.rs` | `filter_candidates()` — active_actions 체크 | |
| INV-039 | `manager/src/selector.rs` | `compute_cost()` — Lossless → 0.0 | |
| INV-040 | `manager/src/selector.rs` | `compute_cost()` — QCF 없으면 INFINITY | |
| INV-041 | `manager/src/selector.rs` | `has_exclusion_conflict()` → skip | |
| INV-042 | `manager/src/selector.rs` | `total_relief.latency < -latency_budget` → skip | |
| INV-043 | `manager/src/selector.rs` | `best_mask` 존재 시 `best_effort_mask`보다 우선 | |
| INV-044 | `manager/src/selector.rs` | `value.clamp(range.min, range.max)` | |
| INV-045 | `manager/src/types.rs:47-56` | `ActionId::primary_domain()` | SwitchHw/Throttle/LayerSkip→Compute, 나머지→Memory |

### 3.4 ReliefEstimator Online Learning

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-ALG-040 | `manager/src/relief/mod.rs:8-27` | `trait ReliefEstimator` | predict, observe, save, load, observation_count |
| MGR-ALG-041 | `manager/src/relief/linear.rs:17-29` | `struct LinearModel` | W(4xD), b(4), P(DxD), lambda, observation_count |
| MGR-ALG-042 | `manager/src/relief/linear.rs` | `predict()` — count==0 시 default_relief | |
| MGR-ALG-043 | `manager/src/relief/linear.rs` | `default_relief()` 함수 | 하드코딩 prior 테이블 |
| MGR-ALG-044 | `manager/src/relief/linear.rs` | `update()` — RLS gain vector + W/b/P 갱신 | lr_bias=0.1, EMA bias |
| MGR-ALG-045 | `manager/src/relief/linear.rs` | `new_model()` — P=100*I | |
| MGR-ALG-046 | `manager/src/config.rs:272-279` | `ReliefModelConfig.forgetting_factor = 0.995` | lambda 기본값 |
| MGR-ALG-047 | `manager/src/relief/linear.rs` | `save()`/`load()` — JSON serde | action_to_key: ActionId↔snake_case |
| INV-046 | `manager/src/relief/linear.rs` | RLS gain vector k 계산 | P_phi / denom |
| INV-047 | `manager/src/relief/linear.rs` | bias EMA (lr=0.1) — W 갱신 후 잔여 오차 | |
| INV-048 | `manager/src/relief/linear.rs` | P 초기값 100*I | `new_model()` 내부 |
| INV-049 | `manager/src/config.rs:275` | `forgetting_factor = 0.995` | (0, 1] 범위 |

### 3.5 FeatureVector Transform

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-ALG-050 | `manager/src/types.rs:167-182` | `mod feature` — 13개 상수 인덱스 | FEATURE_DIM = 13 |
| MGR-ALG-051 | `manager/src/pipeline.rs:413-465` | `update_engine_state()` | Heartbeat만 갱신 |
| MGR-ALG-052 | `manager/src/types.rs:160-165` | `FeatureVector::zeros()` | 13차원 zero vector |
| MGR-ALG-053 | `manager/src/pipeline.rs:455-464` | `ActionId::from_str()` filter_map | 인식 불가 skip |

### 3.6 Observation Relief Calculation

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-ALG-060 | `manager/src/pipeline.rs:49-58` | `struct ObservationContext` | pressure_before, feature_vec, applied_actions, applied_at |
| MGR-ALG-061 | `manager/src/pipeline.rs:232-253` | `update_observation()` | `OBSERVATION_DELAY_SECS=3.0` |
| MGR-ALG-062 | `manager/src/pipeline.rs:358-364,232-253` | pending_observation 수명 관리 | Directive 발행 시 Set, 3s 후 Clear, 새 Directive 시 교체 |
| MGR-ALG-063 | `manager/src/pipeline.rs:358-365` | Directive 발행 시 ObservationContext 기록 | `last_acted_pressure` 동시 갱신 |
| MGR-ALG-064 | `manager/src/pipeline.rs:383-403` | `process_de_escalation` 로직 | `result.is_none()` 확인 후 생성 |
| INV-050 | `manager/src/types.rs:103-113` | `PressureVector::sub()` → `ReliefVector { latency: 0.0 }` | latency 항상 0.0 |
| INV-051 | `manager/src/pipeline.rs:242-245` | `for action in ctx.applied_actions` — 동일 relief 귀속 | 개별 기여 분리 불가 |

### 3.7 Process Signal Pipeline

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-ALG-070 | `manager/src/pipeline.rs:301-407` | `process_signal()` 전체 구현 | 7단계 순차 |
| MGR-ALG-071 | `manager/src/pipeline.rs:325-334` | QCF proxy — `registry.default_cost()` 사용 | 실시간 QCF 미구현 |
| MGR-ALG-072 | (trace 예시 — 코드 없음) | | |

## 코드-스펙 차이 (Known Divergence)

| 항목 | 스펙 | 코드 | 영향 |
|------|------|------|------|
| Memory 압력 계산 (MGR-ALG-013a) | `pressure.memory = m` (직접 매핑, PI 미경유) | `self.pi_memory.update(m, dt)` (PI 경유) | 코드가 PI 평활화를 적용. 스펙 의도는 즉각 반영 |
| EnergyConstraint (MGR-ALG-015) | raw `battery_pct` → `clamp(1 - battery_pct/100, 0, 1) * 0.5` | `level_to_measurement(level) * 0.5` | 코드는 4단계 이산 Level 변환, 스펙은 연속 raw 변환 |

## Config

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `policy.pi_controller.compute_kp` | f32 | 1.5 | MGR-ALG-013 |
| `policy.pi_controller.compute_ki` | f32 | 0.3 | MGR-ALG-013 |
| `policy.pi_controller.compute_setpoint` | f32 | 0.70 | MGR-ALG-013 |
| `policy.pi_controller.thermal_kp` | f32 | 1.0 | MGR-ALG-013 |
| `policy.pi_controller.thermal_ki` | f32 | 0.2 | MGR-ALG-013 |
| `policy.pi_controller.thermal_setpoint` | f32 | 0.80 | MGR-ALG-013 |
| `policy.pi_controller.integral_clamp` | f32 | 2.0 | MGR-ALG-012 |
| `policy.pi_controller.memory_kp` | f32 | 2.0 | MGR-ALG-013 (비사용) |
| `policy.pi_controller.memory_ki` | f32 | 0.5 | MGR-ALG-013 (비사용) |
| `policy.pi_controller.memory_setpoint` | f32 | 0.75 | MGR-ALG-013 (비사용) |
| `policy.pi_controller.memory_gain_zones` | Vec\<GainZone\> | [] | MGR-ALG-011 (비사용) |
| `policy.supervisory.warning_threshold` | f32 | 0.4 | MGR-ALG-023 |
| `policy.supervisory.critical_threshold` | f32 | 0.7 | MGR-ALG-023 |
| `policy.supervisory.warning_release` | f32 | 0.25 | MGR-ALG-023 |
| `policy.supervisory.critical_release` | f32 | 0.50 | MGR-ALG-023 |
| `policy.supervisory.hold_time_secs` | f32 | 4.0 | MGR-ALG-023 |
| `policy.selector.latency_budget` | f32 | 0.5 | MGR-ALG-033 |
| `policy.selector.algorithm` | String | "exhaustive" | MGR-ALG-030 |
| `policy.relief_model.forgetting_factor` | f32 | 0.995 | MGR-ALG-046 |
| `policy.relief_model.prior_weight` | u32 | 5 | MGR-ALG-045 |
| `policy.relief_model.storage_dir` | String | "~/.llm_rs/models" | MGR-ALG-047 |

## 주요 상수

| 상수 | 값 | 코드 위치 | spec/ 근거 |
|------|---|----------|-----------|
| `OBSERVATION_DELAY_SECS` | 3.0 | `manager/src/pipeline.rs:62` | MGR-ALG-061 |
| `FEATURE_DIM` | 13 | `manager/src/types.rs:152` | MGR-ALG-050 |
| `level_to_measurement` mapping | Normal=0.0, Warning=0.55, Critical=0.80, Emergency=1.0 | `manager/src/pipeline.rs:524-531` | MGR-ALG-015 (코드-스펙 차이) |

## 주요 Struct/Trait 매핑

| spec 개념 | Rust 타입 | 위치 |
|-----------|----------|------|
| PI Controller | `struct PiController` | `manager/src/pi_controller.rs:17-27` |
| GainZone | `struct GainZone` | `manager/src/pi_controller.rs:5-8` |
| SupervisoryLayer | `struct SupervisoryLayer` | `manager/src/supervisory.rs:13-22` |
| ActionSelector | `struct ActionSelector` (unit struct) | `manager/src/selector.rs:14` |
| ReliefEstimator | `trait ReliefEstimator` | `manager/src/relief/mod.rs:8-27` |
| OnlineLinearEstimator | `struct OnlineLinearEstimator` | `manager/src/relief/linear.rs` |
| LinearModel | `struct LinearModel` | `manager/src/relief/linear.rs:17-29` |
| ObservationContext | `struct ObservationContext` | `manager/src/pipeline.rs:49-58` |
| HierarchicalPolicy | `struct HierarchicalPolicy` | `manager/src/pipeline.rs:75-104` |
| PolicyStrategy | `trait PolicyStrategy` | `manager/src/pipeline.rs:34-46` |