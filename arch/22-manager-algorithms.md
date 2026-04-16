# Manager Algorithms -- Architecture

> spec/22-manager-algorithms.md (WHAT) → 구현 상세 (HOW).
> 컴포넌트 중심으로 설계 결정, 인터페이스, 처리 흐름, 예외 처리, 코드-스펙 차이를 기술한다.

---

## 1. PiController (MGR-ALG-010~016, INV-030~031) **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

> **DEPRECATED (2026-04)**: `hierarchical` feature flag 뒤로 이동. LuaPolicy는 PI Controller 대신 `SignalState.pressure_with_thermal()`로 직접 pressure를 계산한다 (§8 참조).

**파일**: `manager/src/pi_controller.rs`

### 1.1 설계 결정

spec은 "도메인별 독립 압력 계산기"를 요구한다 (MGR-ALG-010). 구현은 단일 `PiController` struct를 도메인별로 인스턴스화하여 사용한다. Compute/Thermal은 PI 제어, Memory는 PI 경유하되 **코드-스펙 차이**가 있다 (아래 1.6 참조).

Gain scheduling (MGR-ALG-011)은 `gain_zones: Vec<GainZone>`으로 구현하며, builder 패턴 (`with_gain_zones()`)으로 선택적 주입한다. 현재 Memory 인스턴스만 gain scheduling을 사용하고, Compute/Thermal은 고정 Kp이다.

Anti-windup (MGR-ALG-012)은 이중 메커니즘으로 구현한다:
- `can_act` 플래그: 해소 가능한 액션 부재 시 적분 동결 (INV-030)
- `integral_clamp`: 적분 상한 제한 (INV-031)

### 1.2 인터페이스

```rust
pub struct GainZone {
    pub above: f32,   // measurement 임계값
    pub kp: f32,      // 이 구간의 비례 이득
}

pub struct PiController {
    kp: f32,
    ki: f32,
    setpoint: f32,
    integral: f32,
    integral_clamp: f32,
    can_act: bool,
    gain_zones: Vec<GainZone>,
}

impl PiController {
    pub fn new(kp: f32, ki: f32, setpoint: f32, integral_clamp: f32) -> Self
    // POST: integral = 0.0, can_act = true, gain_zones = []

    pub fn with_gain_zones(mut self, zones: Vec<GainZone>) -> Self
    // Builder 패턴. PRE: zones는 above 기준 오름차순 정렬

    pub fn update(&mut self, measurement: f32, dt: f32) -> f32
    // PRE: measurement in [0, 1], dt > 0
    // POST: output in [0, 1] (INV-031)
    // POST: can_act == false 이면 integral 불변 (INV-030)

    pub fn set_can_act(&mut self, can_act: bool)
    // Anti-windup 플래그 설정

    pub fn reset_integral(&mut self)
    // POST: integral = 0.0
}
```

### 1.3 처리 흐름

```mermaid
flowchart TD
    A[update 호출] --> B["error = max(measurement - setpoint, 0)"]
    B --> C{can_act?}
    C -->|yes| D["integral = clamp(integral + error*dt, 0, integral_clamp)"]
    C -->|no| E[integral 불변]
    D --> F["kp = effective_kp(measurement)"]
    E --> F
    F --> G["output = clamp(kp*error + ki*integral, 0, 1)"]
    G --> H[return output]
```

`effective_kp(measurement)` — `gain_zones`를 역순 탐색하여 `measurement >= zone.above`인 첫 zone의 `kp`를 반환. zones가 비어있으면 기본 `self.kp` 반환.

### 1.4 인스턴스 구성 (pipeline.rs 내)

| 인스턴스 | kp | ki | setpoint | integral_clamp | gain_zones | spec ID |
|---------|----|----|----------|---------------|------------|---------|
| `pi_compute` | 1.5 | 0.3 | 0.70 | 2.0 | 없음 | MGR-ALG-013 |
| `pi_thermal` | 1.0 | 0.2 | 0.80 | 2.0 | 없음 | MGR-ALG-013 |
| `pi_memory` | 2.0 | 0.5 | 0.75 | 2.0 | config 주입 | MGR-ALG-013 |

인스턴스 생성은 `HierarchicalPolicy::new()` (`manager/src/pipeline.rs`)에서 `PiControllerConfig` 기반으로 수행한다.

### 1.5 Measurement Normalization (update_pressure)

`HierarchicalPolicy::update_pressure()` (`manager/src/pipeline.rs`)이 SystemSignal에서 도메인별 measurement를 추출한다.

| SystemSignal | 측정값 변환 | PI 인스턴스 | spec ID |
|-------------|-----------|------------|---------|
| MemoryPressure | `m = clamp(1 - available/total, 0, 1)` | `pi_memory` | MGR-ALG-014 |
| ThermalAlert | `m = clamp(temperature_mc / 85000, 0, 1)` | `pi_thermal` | MGR-ALG-014 |
| ComputeGuidance | `m = max(cpu_pct/100, gpu_pct/100)` | `pi_compute` | MGR-ALG-014 |
| EnergyConstraint | `m = level_to_measurement(level) * 0.5` | `pi_compute` (보조) | MGR-ALG-015 |

`elapsed_dt()` 메서드 (MGR-ALG-016): 도메인별 `last_signal_time` HashMap을 유지하여 실측 dt를 계산한다. 첫 신호 시 default 0.1s, 결과는 `[0.001, 10.0]`으로 clamp.

### 1.6 코드-스펙 차이

| 항목 | 스펙 (WHAT) | 코드 (HOW) | 영향 |
|------|------------|-----------|------|
| **Memory 압력 계산** (MGR-ALG-013a) | `pressure.memory = m` (직접 매핑, PI 미경유) | `self.pi_memory.update(m, dt)` (PI 경유) | 코드가 PI 평활화를 적용하여 OOM 상황에서 반응이 지연됨. 스펙 의도는 즉각 반영. |
| **EnergyConstraint** (MGR-ALG-015) | raw `battery_pct` → `clamp(1 - battery_pct/100, 0, 1) * 0.5` | `level_to_measurement(level) * 0.5` (4단계 이산 변환) | 코드는 Level enum (Normal=0.0, Warning=0.55, Critical=0.80, Emergency=1.0)으로 변환. 스펙은 연속 raw 값 사용. |

### 1.7 테스트 커버리지

`manager/src/pi_controller.rs` 내 12개 유닛 테스트:
- zero error 출력, proportional-only, 적분 누적, anti-windup (can_act/clamp), 출력 범위, spike vs sustained, reset, gain zones (empty/low/mid/high/critical).

---

## 2. SupervisoryLayer (MGR-ALG-020~024, INV-032~036) **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

> **DEPRECATED (2026-04)**: `hierarchical` feature flag 뒤로 이동. LuaPolicy는 `TriggerEngine`(3 triggers + hysteresis)으로 대체한다 (§9 참조).

**파일**: `manager/src/supervisory.rs`

### 2.1 설계 결정

spec은 "peak pressure 기반 3-state 운영 모드 판정, 비대칭 hysteresis"를 요구한다 (MGR-ALG-020~021). 구현 전략:

- **에스컬레이션**: 즉시 전환. Normal에서 Critical 직행 가능 (INV-032).
- **디에스컬레이션**: `hold_time` 동안 release threshold 미만 유지 후 1단계씩 하강 (INV-033). `stable_since: Option<Instant>` 필드로 추적.
- **테스트 용이성**: `evaluate_at(&mut self, pressure, now: Instant)` 메서드로 시간 주입 가능. 공개 `evaluate()` 메서드는 `Instant::now()` 위임.

### 2.2 인터페이스

```rust
pub struct SupervisoryLayer {
    mode: OperatingMode,
    warning_threshold: f32,
    critical_threshold: f32,
    warning_release: f32,
    critical_release: f32,
    hold_time: Duration,
    stable_since: Option<Instant>,
}

impl SupervisoryLayer {
    pub fn new(config: &SupervisoryConfig) -> Self
    // POST: mode = Normal, stable_since = None

    pub fn mode(&self) -> OperatingMode

    pub fn evaluate(&mut self, pressure: &PressureVector) -> OperatingMode
    // POST: mode 갱신 및 반환

    pub fn evaluate_at(&mut self, pressure: &PressureVector, now: Instant) -> OperatingMode
    // 테스트용 시간 주입 버전
}
```

### 2.3 상태 전이 흐름

```mermaid
stateDiagram-v2
    [*] --> Normal
    Normal --> Warning : peak >= warning_threshold (즉시)
    Normal --> Critical : peak >= critical_threshold (즉시, 직행)
    Warning --> Critical : peak >= critical_threshold (즉시)
    Warning --> Normal : peak < warning_release & hold_time 경과
    Critical --> Warning : peak < critical_release & hold_time 경과

    note right of Warning
        warning_release <= peak < warning_threshold 구간:
        stable_since 리셋, Warning 유지
    end note

    note right of Critical
        critical_release <= peak < critical_threshold 구간:
        stable_since 리셋, Critical 유지
    end note
```

**디에스컬레이션 상세**: peak가 release threshold 미만이 될 때 `stable_since = Some(now)` 설정. 이후 호출에서 `now - stable_since >= hold_time`이면 1단계 하강. 중간에 peak가 재상승하면 `stable_since = None`으로 리셋 (hold_time 재시작).

### 2.4 needs_action 판정 (MGR-ALG-024)

`HierarchicalPolicy::process_signal()` 내부에서 수행:

```rust
let needs_action = match mode {
    OperatingMode::Normal => false,
    _ => mode != self.prev_mode
        || self.pressure.any_domain_exceeds(&self.last_acted_pressure, 1.2),
};
```

- Normal 모드: 항상 false.
- Warning/Critical: 모드 변경 시 또는 **어느 한 도메인이든** 마지막 액션 시점 대비 20% 이상 압력 증가 시 true.

### 2.5 Config

| 파라미터 | 기본값 | 불변식 |
|---------|--------|--------|
| `warning_threshold` | 0.4 | INV-036: `< critical_threshold` |
| `critical_threshold` | 0.7 | |
| `warning_release` | 0.25 | INV-034: `< warning_threshold` |
| `critical_release` | 0.50 | INV-035: `< critical_threshold` |
| `hold_time_secs` | 4.0 | |

불변식 보증: `SupervisoryConfig::default()` 값으로 보증. 런타임 검증은 없음 (config 로딩 시 위반 가능).

### 2.6 테스트 커버리지

`manager/src/supervisory.rs` 내 8개 유닛 테스트:
- 정상 유지, Warning/Critical 에스컬레이션, 직행, hold_time 미충족/경과, spike 리셋, stepwise 디에스컬레이션.

---

## 3. ActionSelector (MGR-ALG-030~037, INV-037~045) **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

> **DEPRECATED (2026-04)**: `hierarchical` feature flag 뒤로 이동. LuaPolicy는 Lua 스크립트 내에서 직접 액션을 선택한다.

**파일**: `manager/src/selector.rs`

### 3.1 설계 결정

spec은 "전수 탐색 후 최소 비용 조합 선택"을 요구한다 (MGR-ALG-030). 구현 전략:

- **Stateless unit struct**: `ActionSelector`는 필드 없는 unit struct. 모든 상태는 `select()` 호출 인자로 전달된다. 이로써 테스트에서 쉽게 mock 주입 가능.
- **2^N bitmask 순회**: N <= 8 제약 (MGR-ALG-037) 하에 최대 256 조합이므로 상수 시간에 완료된다 (O(2^N * N^2) = 최대 16384 연산).
- **best-effort 폴백**: 어떤 조합도 전체 pressure를 해소하지 못하면 coverage 최대화 조합을 반환 (INV-043).

### 3.2 인터페이스

```rust
pub struct ActionSelector;  // unit struct, stateless

impl ActionSelector {
    pub fn select(
        registry: &ActionRegistry,
        estimator: &dyn ReliefEstimator,
        pressure: &PressureVector,
        mode: OperatingMode,
        engine_state: &FeatureVector,
        qcf_values: &HashMap<ActionId, f32>,
        latency_budget: f32,
        active_actions: &[ActionId],
        available_actions: &[ActionId],
    ) -> Vec<ActionCommand>
    // PRE: mode != Normal (호출측에서 보장; 내부적으로 pressure=0 조기 반환)
    // POST: 각 ActionCommand.params 값 in [range.min, range.max] (INV-044)
    // POST: Warning 모드 → 반환에 Lossy 액션 없음 (INV-037)
    // POST: active_actions에 포함된 액션 없음 (INV-038)
    // POST: Lossless cost = 0 (INV-039)
    // POST: QCF 없는 Lossy = INFINITY (INV-040)
}
```

### 3.3 4-Phase 처리 흐름

```mermaid
flowchart TD
    P0{pressure 전부 <= 0?} -->|yes| R0[return empty]
    P0 -->|no| P1

    subgraph "Phase 1: filter_candidates"
        P1[registry.all_actions] --> F1{Warning & Lossy?}
        F1 -->|yes| SKIP1[제외]
        F1 -->|no| F2{active_actions에 포함?}
        F2 -->|yes| SKIP2[제외]
        F2 -->|no| F3{available 비어있지 않고 미포함?}
        F3 -->|yes| SKIP3[제외]
        F3 -->|no| KEEP[후보에 추가]
    end

    KEEP --> P2

    subgraph "Phase 2: predict + cost"
        P2[각 후보에 대해] --> P2a["relief = estimator.predict(action, state)"]
        P2a --> P2b["cost = compute_cost(action, registry, qcf)"]
    end

    P2b --> P3

    subgraph "Phase 3: find_optimal"
        P3["2^N mask 순회"] --> C1{exclusion conflict?}
        C1 -->|yes| NEXT[skip]
        C1 -->|no| C2{latency < -budget?}
        C2 -->|yes| NEXT
        C2 -->|no| C3{pressure 충족?}
        C3 -->|yes| C4["cost < best_cost → update best_mask"]
        C3 -->|no| C5["coverage > best_coverage → update best_effort_mask"]
    end

    C4 --> P4
    C5 --> P4

    subgraph "Phase 4: parametrize"
        P4["best_mask or best_effort_mask"] --> P4a["각 action에 대해 parametrize()"]
        P4a --> P4b["ActionCommand 생성"]
    end

    P4b --> RET[return commands]
```

### 3.4 핵심 내부 함수

| 함수 | spec ID | 역할 |
|------|---------|------|
| `filter_candidates()` | MGR-ALG-031 | Warning/active/available 필터. INV-037, INV-038 보증. |
| `compute_cost()` | MGR-ALG-032 | Lossless=0.0, Lossy=QCF값 또는 INFINITY. INV-039, INV-040 보증. |
| `find_optimal()` | MGR-ALG-033 | 2^N bitmask 순회. exclusion (INV-041), latency budget (INV-042), best-effort (INV-043). |
| `has_exclusion_conflict()` | MGR-ALG-034 | O(N^2) 쌍별 검사 via `registry.is_excluded()`. |
| `parametrize()` | MGR-ALG-035 | 선형 보간: `value = max - intensity * (max - min)`, clamp. INV-044 보증. |

### 3.5 ActionCommand → EngineCommand 변환 (MGR-ALG-036)

**파일**: `manager/src/pipeline.rs` — 모듈 레벨 함수 `action_to_engine_command()`

| ActionId | EngineCommand | 파라미터 |
|----------|-------------|---------|
| SwitchHw | `SwitchHw { device: "cpu" }` | 고정 |
| Throttle | `Throttle { delay_ms }` | `params["delay_ms"] as u64` |
| KvEvictSliding | `KvEvictSliding { keep_ratio }` | `params["keep_ratio"]` |
| KvEvictH2o | `KvEvictH2o { keep_ratio }` | `params["keep_ratio"]` |
| KvOffloadDisk | `KvEvictSliding { keep_ratio: 0.8 }` | fallback 고정 |
| KvQuantDynamic | `KvQuantDynamic { target_bits }` | `params["target_bits"] as u8` |
| LayerSkip | `LayerSkip { skip_ratio }` | `skip_layers / total_layers`, clamp [0, 1] |
| (any, Release) | None | restore directive에서 일괄 처리 |

### 3.6 primary_domain 매핑 (INV-045)

**파일**: `manager/src/types.rs` — `ActionId::primary_domain()`

| ActionId | Domain |
|----------|--------|
| SwitchHw, Throttle, LayerSkip | Compute |
| KvOffloadDisk, KvEvictSliding, KvEvictH2o, KvQuantDynamic | Memory |

### 3.7 테스트 커버리지

`manager/src/selector.rs` 내 10개 유닛 테스트:
- Warning lossless only, Critical minimum cost, cross-domain single action, exclusion group, latency budget, best-effort, empty candidates, parametrize proportional, no action at zero pressure, active exclusion, available_actions filtering.

`MockEstimator` (테스트 전용): `HashMap<ActionId, ReliefVector>` 기반 predict, observe/save/load는 no-op.

---

## 4. ReliefEstimator / OnlineLinearEstimator (MGR-ALG-040~047, INV-046~049) **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

> **DEPRECATED (2026-04)**: `hierarchical` feature flag 뒤로 이동. LuaPolicy는 `EwmaReliefTable`(6D EWMA)로 대체한다 (§10 참조).

**파일**: `manager/src/relief/mod.rs` (trait), `manager/src/relief/linear.rs` (구현체)

### 4.1 설계 결정

spec은 "Strategy 패턴 인터페이스 + RLS 온라인 선형 회귀 구현"을 요구한다 (MGR-ALG-040~041).

- **trait ReliefEstimator**: `predict`, `observe`, `save`, `load`, `observation_count` 5개 메서드. `Send + Sync` bound로 스레드 안전 보장.
- **OnlineLinearEstimator**: 액션별 독립 `LinearModel`을 `HashMap<ActionId, LinearModel>`로 관리.
- **지연 초기화**: `observe()` 호출 시에만 `ensure_model()`로 모델 생성. `predict()`는 모델 미존재 또는 `observation_count == 0` 시 `default_relief()` 반환 (MGR-ALG-042).
- **default_relief**: 하드코딩 prior 테이블 (MGR-ALG-043). 학습 데이터 없는 cold start 시 합리적 초기 예측 제공.

### 4.2 인터페이스

```rust
// --- trait (relief/mod.rs) ---
pub trait ReliefEstimator: Send + Sync {
    fn predict(&self, action: &ActionId, state: &FeatureVector) -> ReliefVector;
    fn observe(&mut self, action: &ActionId, state: &FeatureVector, actual: &ReliefVector);
    fn save(&self, path: &Path) -> io::Result<()>;
    fn load(&mut self, path: &Path) -> io::Result<()>;
    fn observation_count(&self, action: &ActionId) -> u32;
}

// --- 구현체 (relief/linear.rs) ---
pub struct OnlineLinearEstimator {
    models: HashMap<ActionId, LinearModel>,
    feature_dim: usize,        // FEATURE_DIM = 13
    forgetting_factor: f32,    // lambda, default 0.995
}

impl OnlineLinearEstimator {
    pub fn new(feature_dim: usize, forgetting_factor: f32) -> Self
    pub fn default_config() -> Self
    // POST: feature_dim = 13, forgetting_factor = 0.995
}
```

### 4.3 LinearModel 내부 구조

```rust
struct LinearModel {        // 비공개, Serialize/Deserialize
    weights: Vec<Vec<f32>>, // 4 x D 행렬
    bias: [f32; 4],         // 4차원 bias
    p_matrix: Vec<Vec<f32>>,// D x D 역공분산 행렬
    forgetting_factor: f32, // lambda
    observation_count: u32,
}
```

- **초기화** (MGR-ALG-045, INV-048): `W = zeros(4, D)`, `b = [0; 4]`, `P = 100 * I(D)`.
- P의 큰 초기값은 초기 관측에 높은 학습률을 부여한다.

### 4.4 RLS Update 처리 흐름 (MGR-ALG-044)

```mermaid
flowchart TD
    A["update(phi, actual)"] --> B["P_phi = P × phi"]
    B --> C["denom = lambda + phi^T × P_phi"]
    C --> D["k = P_phi / denom  (gain vector, INV-046)"]
    D --> E["for dim in 0..4"]
    E --> F["error = actual[dim] - (W[dim]·phi + b[dim])"]
    F --> G["W[dim] += k × error"]
    G --> H["residual = actual[dim] - (W_new[dim]·phi + b[dim])"]
    H --> I["b[dim] += 0.1 × residual  (EMA, INV-047)"]
    I --> J["P = (P - k × phi^T × P) / lambda"]
    J --> K["observation_count += 1"]
```

**핵심 수식**:
- Gain vector: `k = P*phi / (lambda + phi^T*P*phi)` (INV-046)
- Bias 갱신: EMA (lr=0.1), W 갱신 후 잔여 오차 기반 (INV-047)
- P 갱신: `P = (P - k * phi^T * P) / lambda`
- lambda = 0.995 → 반감기 ~138 관측 (INV-049)

### 4.5 Default Relief Prior 테이블 (MGR-ALG-043)

| ActionId | compute | memory | thermal | latency |
|----------|---------|--------|---------|---------|
| SwitchHw | 0.5 | 0.0 | 0.3 | -0.1 |
| Throttle | 0.3 | 0.0 | 0.2 | -0.3 |
| KvOffloadDisk | 0.0 | 0.4 | 0.0 | -0.2 |
| KvEvictSliding | 0.0 | 0.7 | 0.0 | 0.0 |
| KvEvictH2o | 0.0 | 0.6 | 0.0 | 0.0 |
| KvQuantDynamic | 0.0 | 0.3 | 0.0 | 0.0 |
| KvMergeD2o | 0.0 | 0.6 | 0.0 | 0.0 |
| LayerSkip | 0.3 | 0.0 | 0.1 | -0.2 |

### 4.6 Persistence (MGR-ALG-047)

- `save()`: `SavedEstimator { feature_dim, forgetting_factor, models: HashMap<String, LinearModel> }` → JSON 직렬화.
- `load()`: JSON → 파싱, `key_to_action()`으로 문자열 → ActionId 변환 (인식 불가 키는 skip).
- `action_to_key()`: ActionId ↔ snake_case 문자열 (예: `SwitchHw` ↔ `"switch_hw"`).
- 파일 부재 시 빈 models로 시작 (비치명적).

### 4.7 테스트 커버리지

`manager/src/relief/linear.rs` 내 7개 유닛 테스트:
- default relief 반환, 각 액션별 default 값 확인, observe 후 수렴 방향, 액션 독립성, save/load roundtrip, observation_count 일치, 수렴 품질.

---

## 5. 데이터 타입 (MGR-ALG-050~053, INV-045, INV-050) **[DEPRECATED: `#[cfg(feature = "hierarchical")]` 일부]**

> **일부 DEPRECATED (2026-04)**: `PressureVector`, `ReliefVector`, `FeatureVector`, `ActionId`, `ActionMeta`, `ParamRange`, `ActionParams`, `ActionCommand`, `Operation`은 `hierarchical` feature flag 뒤로 이동. `OperatingMode`는 공용 유지.

**파일**: `manager/src/types.rs`

### 5.1 PressureVector

```rust
pub struct PressureVector {
    pub compute: f32,
    pub memory: f32,
    pub thermal: f32,
}

impl PressureVector {
    pub fn max(&self) -> f32
    // MGR-ALG-022: compute.max(memory).max(thermal)

    pub fn any_domain_exceeds(&self, reference: &PressureVector, factor: f32) -> bool
    // MGR-ALG-024: 도메인별 factor 배 초과 검사
}

impl Sub for PressureVector {
    type Output = ReliefVector;
    // INV-050: latency = 0.0 (PressureVector에 latency 차원 없음)
}
```

### 5.2 ReliefVector

```rust
pub struct ReliefVector {
    pub compute: f32,
    pub memory: f32,
    pub thermal: f32,
    pub latency: f32,   // 음수 = 악화
}

impl ReliefVector {
    pub fn zero() -> Self
}
// Add, AddAssign 구현
```

### 5.3 FeatureVector (MGR-ALG-050~052)

```rust
pub const FEATURE_DIM: usize = 13;

pub struct FeatureVector {
    pub values: [f32; FEATURE_DIM],
}

impl FeatureVector {
    pub fn zeros() -> Self
    // MGR-ALG-052: 13차원 zero vector
}

pub mod feature {
    pub const KV_OCCUPANCY: usize = 0;
    pub const IS_GPU: usize = 1;
    pub const TOKEN_PROGRESS: usize = 2;
    pub const IS_PREFILL: usize = 3;          // 미사용, 확장 예약
    pub const KV_DTYPE_NORM: usize = 4;       // 미사용, 확장 예약
    pub const TBT_RATIO: usize = 5;
    pub const TOKENS_GENERATED_NORM: usize = 6;
    pub const ACTIVE_SWITCH_HW: usize = 7;    // 미사용, 확장 예약
    pub const ACTIVE_THROTTLE: usize = 8;     // 미사용, 확장 예약
    pub const ACTIVE_KV_OFFLOAD: usize = 9;   // 미사용, 확장 예약
    pub const ACTIVE_EVICTION: usize = 10;
    pub const ACTIVE_LAYER_SKIP: usize = 11;
    pub const ACTIVE_KV_QUANT: usize = 12;    // 미사용, 확장 예약
}
```

### 5.4 ActionId

```rust
pub enum ActionId {
    SwitchHw, Throttle, KvOffloadDisk,
    KvEvictSliding, KvEvictH2o, KvQuantDynamic, LayerSkip,
}

impl ActionId {
    pub fn from_str(s: &str) -> Option<ActionId>   // MGR-ALG-053
    pub fn all() -> &'static [ActionId]
    pub fn primary_domain(&self) -> Domain            // INV-045
}
```

### 5.5 기타 타입

| 타입 | 설명 |
|------|------|
| `ActionKind` | `Lossless` / `Lossy` |
| `Domain` | `Compute` / `Memory` / `Thermal` |
| `OperatingMode` | `Normal` / `Warning` / `Critical` (PartialOrd 도출) |
| `ActionMeta` | id, kind, reversible, param_range, exclusion_group, default_cost |
| `ParamRange` | param_name, min, max |
| `ActionParams` | `values: HashMap<String, f32>` |
| `ActionCommand` | action: ActionId, operation: Operation |
| `Operation` | `Apply(ActionParams)` / `Release` |

---

## 6. HierarchicalPolicy 오케스트레이션 (MGR-ALG-060~072) **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

> **DEPRECATED (2026-04)**: `hierarchical` feature flag 뒤로 이동. LuaPolicy가 기본 정책.

**파일**: `manager/src/pipeline.rs`

### 6.1 설계 결정

spec은 "SystemSignal → PI → Supervisory → ActionSelector → EngineDirective" 7단계 순차 파이프라인을 요구한다 (MGR-ALG-070). `HierarchicalPolicy`는 이 4개 컴포넌트와 관측 컨텍스트를 통합 관리하는 오케스트레이터이다.

**PolicyStrategy trait**: 정책 계층의 공통 인터페이스. 향후 규칙 기반 정책 등으로 교체 가능 (OCP).

### 6.2 인터페이스

```rust
pub trait PolicyStrategy: Send {
    fn process_signal(&mut self, signal: &SystemSignal) -> Option<EngineDirective>;
    fn update_engine_state(&mut self, msg: &EngineMessage);
    fn mode(&self) -> OperatingMode;
    fn save_model(&self) {}  // default no-op
}

pub struct HierarchicalPolicy {
    // 컴포넌트
    pi_compute: PiController,
    pi_memory: PiController,
    pi_thermal: PiController,
    supervisory: SupervisoryLayer,
    registry: ActionRegistry,
    estimator: OnlineLinearEstimator,
    // 상태
    engine_state: FeatureVector,
    pressure: PressureVector,
    prev_mode: OperatingMode,
    last_acted_pressure: PressureVector,
    pending_observation: Option<ObservationContext>,
    latency_budget: f32,
    dt: f32,
    relief_model_path: Option<String>,
    available_actions: Vec<ActionId>,
    active_actions_reported: Vec<ActionId>,
    last_signal_time: HashMap<&'static str, Instant>,
}

impl HierarchicalPolicy {
    pub fn new(config: &PolicyConfig) -> Self
    pub fn set_relief_model_path(&mut self, path: String)
    pub fn set_dt(&mut self, dt: f32)
    pub fn pressure(&self) -> &PressureVector
}
```

### 6.3 process_signal 7단계 파이프라인 (MGR-ALG-070)

```mermaid
flowchart TD
    S1["1 update_pressure(signal)"] --> S2["2 supervisory.evaluate(pressure) -> mode"]
    S2 --> S3["3 update_observation() -- 3s 경과 시 실측 relief 관측"]
    S3 --> S4{"4 needs_action?"}
    S4 -->|no| S7
    S4 -->|yes| S5["5 QCF proxy: registry.default_cost() (MGR-ALG-071)"]
    S5 --> S6["6 ActionSelector::select() -> commands"]
    S6 --> S6a{commands 비어있음?}
    S6a -->|yes| S7
    S6a -->|no| S6b["convert_to_engine_commands()"]
    S6b --> S6c["ObservationContext 기록 (MGR-ALG-063)"]
    S6c --> S6d["last_acted_pressure 갱신"]
    S6d --> S6e["result = Some(EngineDirective)"]
    S6e --> S7

    S7{"7 prev_mode > mode? (디에스컬레이션)"}
    S7 -->|no| S8["prev_mode = mode; return result"]
    S7 -->|yes| S7a{"(prev, curr)"}
    S7a -->|"Critical→Warning"| S7b["result = build_lossy_release_directive() (if result is None)"]
    S7a -->|"*→Normal"| S7c["result = build_restore_directive() (if result is None)"]
    S7b --> S8
    S7c --> S8
```

### 6.4 ObservationContext 수명 관리 (MGR-ALG-060~064)

```rust
struct ObservationContext {
    pressure_before: PressureVector,
    feature_vec: FeatureVector,
    applied_actions: Vec<ActionId>,
    applied_at: Instant,
}
```

| 이벤트 | 전이 | spec ID |
|--------|------|---------|
| Directive 발행 | `None → Some(ctx)` | MGR-ALG-063 |
| 3.0s 경과 | `Some → None` (observe 호출 후) | MGR-ALG-061 |
| 3.0s 미경과 | `Some` 유지 | MGR-ALG-062 |
| 새 Directive (관찰 중) | 이전 폐기, 새 ctx로 교체 | MGR-ALG-062 |

**실측 relief 계산**: `actual_relief = pressure_before - current_pressure` (INV-050: latency=0.0). 적용된 모든 액션에 동일 relief 귀속 (INV-051: 개별 기여 분리 불가).

### 6.5 update_engine_state (MGR-ALG-051)

Heartbeat 메시지에서만 FeatureVector를 갱신한다. Capability/Response 메시지는 무시.

갱신 필드: `KV_OCCUPANCY`, `IS_GPU`, `TOKEN_PROGRESS`, `TBT_RATIO`, `TOKENS_GENERATED_NORM`, `ACTIVE_EVICTION`, `ACTIVE_LAYER_SKIP`. 미사용 인덱스 (3, 4, 7, 8, 9, 12)는 기본값 0.0 유지.

또한 `available_actions`, `active_actions_reported`를 `ActionId::from_str()` 파싱으로 갱신 (MGR-ALG-053).

### 6.6 QCF Proxy (MGR-ALG-071)

실시간 QCF가 미구현이므로, `registry.default_cost(&id)`로 lossy 액션의 cost를 대체한다.

### 6.7 디에스컬레이션 Directive (MGR-ALG-064)

- Critical → Warning: `RestoreDefaults` (lossy 해제, lossless는 다음 사이클에서 재선택)
- * → Normal: `RestoreDefaults` (전체 복원)
- 동일 사이클에서 needs_action 경로에서 이미 directive가 생성된 경우 디에스컬레이션은 무시 (`result.is_none()` 확인).

### 6.8 테스트 커버리지

`manager/src/pipeline.rs` 내 유닛 테스트:
- Normal pressure 무응답, pressure 누적, (통합 테스트는 mock_engine/mock_manager 바이너리).

---

## 7. ActionRegistry **[DEPRECATED: `#[cfg(feature = "hierarchical")]`]**

> **DEPRECATED (2026-04)**: `hierarchical` feature flag 뒤로 이동.

**파일**: `manager/src/action_registry.rs`

### 7.1 설계 결정

`PolicyConfig`에서 액션 메타데이터와 배타 그룹을 파싱하여 통합 레지스트리로 관리한다. ActionSelector가 이 레지스트리에 의존하여 후보 필터링, cost 계산, exclusion 검사, 파라미터 결정을 수행한다.

### 7.2 인터페이스

```rust
pub struct ActionRegistry {
    actions: HashMap<ActionId, ActionMeta>,
    exclusion_groups: HashMap<String, Vec<ActionId>>,
}

impl ActionRegistry {
    pub fn from_config(config: &PolicyConfig) -> Self
    pub fn get(&self, action: &ActionId) -> Option<&ActionMeta>
    pub fn all_actions(&self) -> impl Iterator<Item = &ActionMeta>
    pub fn lossy_actions(&self) -> Vec<ActionId>
    pub fn is_excluded(&self, a: &ActionId, b: &ActionId) -> bool
    pub fn default_cost(&self, action: &ActionId) -> f32
}
```

`default_param_range(id)` (모듈 레벨 함수): ActionId별 하드코딩 ParamRange 반환. 예: `KvEvictSliding` → `ParamRange { param_name: "keep_ratio", min: 0.3, max: 0.9 }`.

---

## 8. Config 종합 **[DEPRECATED 섹션 — hierarchical feature 전용]**

> 아래 config 키들은 `hierarchical` feature 활성 시에만 유효하다. LuaPolicy의 config는 §12 참조.

| config 키 | 타입 | 기본값 | spec 근거 |
|-----------|------|--------|-----------|
| `policy.pi_controller.compute_kp` | f32 | 1.5 | MGR-ALG-013 |
| `policy.pi_controller.compute_ki` | f32 | 0.3 | MGR-ALG-013 |
| `policy.pi_controller.compute_setpoint` | f32 | 0.70 | MGR-ALG-013 |
| `policy.pi_controller.thermal_kp` | f32 | 1.0 | MGR-ALG-013 |
| `policy.pi_controller.thermal_ki` | f32 | 0.2 | MGR-ALG-013 |
| `policy.pi_controller.thermal_setpoint` | f32 | 0.80 | MGR-ALG-013 |
| `policy.pi_controller.integral_clamp` | f32 | 2.0 | MGR-ALG-012 |
| `policy.pi_controller.memory_kp` | f32 | 2.0 | MGR-ALG-013 |
| `policy.pi_controller.memory_ki` | f32 | 0.5 | MGR-ALG-013 |
| `policy.pi_controller.memory_setpoint` | f32 | 0.75 | MGR-ALG-013 |
| `policy.pi_controller.memory_gain_zones` | `Vec<GainZone>` | [] | MGR-ALG-011 |
| `policy.supervisory.warning_threshold` | f32 | 0.4 | MGR-ALG-023 |
| `policy.supervisory.critical_threshold` | f32 | 0.7 | MGR-ALG-023 |
| `policy.supervisory.warning_release` | f32 | 0.25 | MGR-ALG-023 |
| `policy.supervisory.critical_release` | f32 | 0.50 | MGR-ALG-023 |
| `policy.supervisory.hold_time_secs` | f32 | 4.0 | MGR-ALG-023 |
| `policy.selector.latency_budget` | f32 | 0.5 | MGR-ALG-033 |
| `policy.selector.algorithm` | String | "exhaustive" | MGR-ALG-030 |
| `policy.relief_model.forgetting_factor` | f32 | 0.995 | MGR-ALG-046 |
| `policy.relief_model.prior_weight` | u32 | 5 | MGR-ALG-045 |
| `policy.relief_model.storage_dir` | String | `"~/.llm_rs/models"` | MGR-ALG-047 |

---

## 9. 주요 상수

### 9.1 HierarchicalPolicy 상수 [DEPRECATED: `#[cfg(feature = "hierarchical")]`]

| 상수 | 값 | 코드 위치 | spec 근거 |
|------|---|----------|-----------|
| `OBSERVATION_DELAY_SECS` | 3.0 | `manager/src/pipeline.rs` | MGR-ALG-061 |
| `FEATURE_DIM` | 13 | `manager/src/types.rs` | MGR-ALG-050 |
| `level_to_measurement` | Normal=0.0, Warning=0.55, Critical=0.80, Emergency=1.0 | `manager/src/pipeline.rs` | MGR-ALG-015 (코드-스펙 차이) |
| `lr_bias` | 0.1 | `manager/src/relief/linear.rs` | MGR-ALG-044 |
| `P_init` | 100.0 * I | `manager/src/relief/linear.rs` | MGR-ALG-045 (INV-048) |
| `SEQ_COUNTER` | AtomicU64, 1부터 | `manager/src/pipeline.rs` | (내부 구현) |

### 9.2 LuaPolicy 상수

| 상수 | 값 | 코드 위치 | 설명 |
|------|---|----------|------|
| `RELIEF_DIMS` | 6 | `manager/src/lua_policy.rs` | 6D relief: gpu, cpu, memory, thermal, latency, main_app_qos |
| `OBSERVATION_DELAY_SECS` | 3.0 | `manager/src/lua_policy.rs` | 관측 settling 시간 |

---

## 10. 코드-스펙 차이 종합 (Known Divergence)

| 항목 | 스펙 (WHAT) | 코드 (HOW) | 영향 | 심각도 |
|------|------------|-----------|------|--------|
| Memory 압력 계산 (MGR-ALG-013a) | `pressure.memory = m` (직접 매핑, PI 미경유) | `self.pi_memory.update(m, dt)` (PI 경유) | PI 평활화로 OOM 즉각 대응 지연 | 높음 |
| EnergyConstraint (MGR-ALG-015) | raw `battery_pct` → 연속 변환 | `level_to_measurement(level)` → 4단계 이산 | 중간 배터리 수준(30~70%)에서 세밀한 압력 반영 불가 | 중간 |
| KvMergeD2o default relief | spec 테이블에 포함 (memory=0.6) | ActionId enum에 KvMergeD2o 정의, default_relief 포함 | 스펙-코드 일치 | 해소 |

---

## 11. 컴포넌트 의존성 다이어그램

### 11.1 HierarchicalPolicy [DEPRECATED]

```mermaid
flowchart LR
    subgraph pipeline.rs
        HP["HierarchicalPolicy<br/>[DEPRECATED]"]
    end

    subgraph pi_controller.rs
        PI[PiController]
        GZ[GainZone]
    end

    subgraph supervisory.rs
        SL[SupervisoryLayer]
    end

    subgraph selector.rs
        AS[ActionSelector]
    end

    subgraph relief/
        RE["trait ReliefEstimator"]
        OLE[OnlineLinearEstimator]
        LM[LinearModel]
    end

    subgraph action_registry.rs
        AR[ActionRegistry]
    end

    subgraph types.rs
        PV[PressureVector]
        RV[ReliefVector]
        FV[FeatureVector]
        AI[ActionId]
        AC[ActionCommand]
    end

    subgraph config.rs
        PC[PolicyConfig]
        PIC[PiControllerConfig]
        SC[SupervisoryConfig]
    end

    HP --> PI
    HP --> SL
    HP --> AS
    HP --> OLE
    HP --> AR
    HP --> PV
    HP --> FV

    PI --> GZ
    SL --> PV
    AS --> AR
    AS --> RE
    AS --> PV
    AS --> FV
    AS --> AI
    AS --> AC

    OLE -.->|implements| RE
    OLE --> LM
    OLE --> AI
    OLE --> FV
    OLE --> RV

    AR --> AI
    AR --> PC

    HP --> PC
    PI --> PIC
    SL --> SC
```

### 11.2 LuaPolicy (기본 정책)

```mermaid
flowchart LR
    subgraph lua_policy.rs
        LP[LuaPolicy]
        SS[SignalState]
        P6D[Pressure6D]
        TE[TriggerEngine]
        TBT[TbtTracker]
        ERT[EwmaReliefTable]
        OC[ObservationContext]
    end

    subgraph config.rs
        AC[AdaptationConfig]
        TC[TriggerConfig]
    end

    subgraph pipeline.rs
        PS["trait PolicyStrategy"]
    end

    LP -.->|implements| PS
    LP --> SS
    LP --> TE
    LP --> ERT
    LP --> OC
    LP --> AC

    SS --> P6D
    TE --> TBT
    TE --> TC
    OC --> P6D
    OC --> ERT
```

---

## 12. LuaPolicy 적응 알고리즘 (신규 2026-04)

> LuaPolicy의 Rust 측 적응 엔진 알고리즘. HierarchicalPolicy의 PI+Supervisory+Selector+RLS를 대체한다.

### 12.1 Pressure6D 직접 정규화 (SignalState → Pressure6D)

**기존 대비 차이**: HierarchicalPolicy는 PI Controller(적분+비례)를 경유하여 pressure를 평활화했다. LuaPolicy는 직접 정규화 매핑으로 **즉각적 반응**을 보장한다.

**계산 규칙**:

| 차원 | 입력 | 변환식 | 범위 |
|------|------|--------|------|
| `gpu` | `SignalState.gpu_pct` (0~100) | `gpu_pct / 100.0` | [0, 1] |
| `cpu` | `SignalState.cpu_pct` (0~100) | `cpu_pct / 100.0` | [0, 1] |
| `memory` | `mem_available / mem_total` | `1.0 - available/total` | [0, 1] |
| `thermal` | `temp_mc` (millidegree) | `(temp_c - safe) / (critical - safe)` | [0, 1] |
| `latency` | `TbtTracker.degradation_ratio()` | 직접 사용 (baseline 미확정 시 0.0) | [0, +inf) |
| `main_app` | (예약) | 0.0 | 0.0 |

**Thermal 정규화**: `temp_safe_c`(기본 35.0)과 `temp_critical_c`(기본 50.0)로 선형 보간. safe 이하 → 0.0, critical 이상 → 1.0.

### 12.2 TBT Baseline 추적 (EWMA)

**파일**: `manager/src/lua_policy.rs` — `TbtTracker`

Engine heartbeat의 `throughput` (tokens/sec)에서 TBT(Time Between Tokens)를 역산하고 EWMA로 추적한다.

**알고리즘**:

```
입력: throughput (tok/s, > 0 필수. <= 0 이면 skip)
tbt_ms = 1000.0 / throughput

if warmup_count == 0:
    ewma = tbt_ms                          # 첫 관측: 직접 대입
else:
    ewma = 0.875 * ewma + 0.125 * tbt_ms   # alpha=0.875 (Jacobson TCP RTT)

warmup_count += 1
if baseline is None and warmup_count >= warmup_target (default 20):
    baseline = ewma                        # baseline 확정

degradation_ratio = (ewma - baseline) / baseline   # baseline 대비 악화 비율
```

**설계 근거**: Alpha=0.875 (= 7/8)는 Jacobson의 TCP RTT 추정에서 유래. 빠른 적응과 노이즈 필터링의 균형. 반감기 ~5.2 관측.

### 12.3 Trigger Hysteresis

**파일**: `manager/src/lua_policy.rs` — `TriggerEngine`

각 trigger는 독립적인 **enter/exit threshold** 쌍으로 hysteresis를 구현한다.

**상태 전이**:

```
비활성 상태:
    if value > enter_threshold → 활성화

활성 상태:
    if value < exit_threshold → 비활성화
    (enter_threshold > exit_threshold 이므로 진동 방지)
```

**3개 Trigger**:

| Trigger | 입력 | enter | exit | Hysteresis gap |
|---------|------|-------|------|---------------|
| `tbt_degraded` | `degradation_ratio` | 0.30 (30% 악화) | 0.10 (10%) | 0.20 |
| `mem_low` | `pressure.memory` | 0.80 | 0.60 | 0.20 |
| `temp_high` | `pressure.thermal` (정규화) | 0.70 | 0.50 | 0.20 |

**TBT trigger 특이사항**:
- `throughput <= 0` 이면 관측 건너뜀 (0으로 나누기 방지)
- `baseline` 미확정 시 (`warmup_count < warmup_target`) trigger 갱신 안 함

### 12.4 EWMA Relief Learning

**파일**: `manager/src/lua_policy.rs` — `EwmaReliefTable`

액션별 6D relief 벡터를 EWMA로 학습한다.

**알고리즘**:

```
observe(action, observed[6]):
    entry = entries.get_or_create(action)
    if entry.observation_count == 0:
        entry.relief = observed             # 첫 관측: 직접 대입 (prior 대체)
    else:
        for i in 0..6:
            entry.relief[i] = alpha * entry.relief[i] + (1 - alpha) * observed[i]
    entry.observation_count += 1

predict(action) -> [f32; 6]:
    if entries.contains(action): return entries[action].relief
    if defaults.contains(action): return defaults[action]   # cold start prior
    return [0.0; 6]
```

**HierarchicalPolicy RLS 대비 차이**:

| 측면 | RLS (OnlineLinearEstimator) | EWMA (EwmaReliefTable) |
|------|---------------------------|------------------------|
| 입력 | 13D FeatureVector (상태 의존) | 없음 (상태 무관) |
| 모델 | 4 x D 가중치 행렬 + bias | 6D 스칼라 |
| 차원 | 4D (compute, memory, thermal, latency) | 6D (gpu, cpu, memory, thermal, latency, main_app) |
| 학습률 | 적응적 (P 행렬, forgetting factor) | 고정 alpha=0.875 |
| 복잡도 | O(D^2) per update | O(6) per update |
| 장점 | 상태 조건부 예측 가능 | 단순, 빠른 수렴, Lua 친화적 |

### 12.5 Observation (3초 Settling)

**파일**: `manager/src/lua_policy.rs` — `ObservationContext`

**알고리즘**:

```
process_signal() 내 try_complete_observation():
    if observation is None: return
    if elapsed < 3.0s: return (observation 유지)

    after = pressure_with_thermal()   # 현재 pressure 계산
    observed[0..4] = before[0..4] - after[0..4]   # 감소=양수 (호전)
    observed[5]    = after[5] - before[5]         # main_app: 증가=양수 (호전)
    relief_table.observe(action, observed)
    observation = None

decide() 반환 후:
    if commands 비어있지 않음:
        첫 번째 커맨드의 action_name으로 ObservationContext 생성
        (여러 커맨드가 반환되어도 단일 액션만 추적)
```

**단일 액션 제한 근거**: 여러 액션이 동시 적용되면 개별 기여를 분리할 수 없다. HierarchicalPolicy도 동일한 제한이 있었다 (INV-051). LuaPolicy는 이를 명시적으로 단일 액션만 추적하여 EWMA 오염을 방지한다.

---

## 13. LuaPolicy Config (AdaptationConfig)

**파일**: `manager/src/config.rs`

| config 키 | 타입 | 기본값 | 설명 |
|-----------|------|--------|------|
| `adaptation.ewma_alpha` | f32 | 0.875 | EWMA 평활 계수 (Jacobson TCP RTT) |
| `adaptation.relief_table_path` | String | "" (비활성) | Relief table JSON 저장 경로 |
| `adaptation.temp_safe_c` | f32 | 35.0 | Thermal 정규화 하한 (Celsius) |
| `adaptation.temp_critical_c` | f32 | 50.0 | Thermal 정규화 상한 (Celsius) |
| `adaptation.trigger.tbt_enter` | f64 | 0.30 | TBT degradation trigger 진입 threshold |
| `adaptation.trigger.tbt_exit` | f64 | 0.10 | TBT degradation trigger 해제 threshold |
| `adaptation.trigger.tbt_warmup_tokens` | u32 | 20 | TBT baseline 확정 전 warmup 토큰 수 |
| `adaptation.trigger.mem_enter` | f64 | 0.80 | Memory trigger 진입 threshold |
| `adaptation.trigger.mem_exit` | f64 | 0.60 | Memory trigger 해제 threshold |
| `adaptation.trigger.temp_enter` | f64 | 0.70 | Temperature trigger 진입 threshold |
| `adaptation.trigger.temp_exit` | f64 | 0.50 | Temperature trigger 해제 threshold |
| `adaptation.default_relief` | `HashMap<String, Vec<f32>>` | {} | Per-action 6D default relief (cold start prior) |

---

## 14. LuaPolicy DPP 정책 결정 (policy_default.lua v2.1.0)

> Lua 스크립트에서 실행되는 DPP(Drift-Plus-Penalty) 기반 정책 결정 알고리즘의 아키텍처.
> §12가 Rust 측 적응 엔진(Pressure6D, TBT, Trigger, EWMA Relief)을 기술한다면, 본 섹션은 Lua 측 정책 결정 로직을 기술한다.

**파일**: `manager/scripts/policy_default.lua`

### 14.1 설계 결정

Lyapunov Drift-Plus-Penalty 프레임워크를 Lua 스크립트로 구현한다. HierarchicalPolicy의 3단계 파이프라인(PI → Supervisory → ActionSelector)을 **단일 DPP score 최적화**로 대체하는 것이 핵심 설계 변경이다.

**대체 관계**:

| HierarchicalPolicy 컴포넌트 | DPP 대체 메커니즘 | 근거 |
|---------------------------|-----------------|------|
| PI Controller (적분+비례) | 직접 정규화 (§12.1) | OOM 즉각 대응 지연 제거 |
| SupervisoryLayer (3-state 모드) | Z_k > 0 여부 + trigger gate | 연속적 압력 반영, 이산 모드 전환 제거 |
| ActionSelector (2^N 전수 탐색) | DPP score argmax O(N) | 탐색 공간 축소, joint registry로 조합 제한 |
| RLS 13D→4D | EWMA 6D scalar (§12.4) | Lua 친화적 단순성 |
| latency budget | hard floor + V·lat soft penalty | 안전성 보장과 유연성 병행 |

**Safety Override 분리**: DPP 최적화와 무관하게 compute/thermal emergency 시 throttle을 강제 삽입한다. DPP 내부에 안전 로직을 혼합하지 않아 **관심사 분리**를 보장한다.

### 14.2 DPP 상수

```lua
DPP = {
    V           = 1.0,   -- latency penalty weight
    C           = 0.30,  -- latency hard floor (normal)
    C_EMERGENCY = 0.50,  -- latency hard floor (emergency)
    W_WARN      = 1.0,   -- threshold excess weights (§4.2.3)
    W_CRIT      = 2.0,
    W_EMERG     = 4.0,
    UCB         = 1.0,   -- LinUCB exploration bonus weight
}
```

| 상수 | 역할 | 설계 근거 |
|------|------|----------|
| `V` | latency soft penalty 가중치 | V가 클수록 latency 보존 우선, 작을수록 resource relief 우선 |
| `C` / `C_EMERGENCY` | latency hard floor | C=0.30: 30% 이상 TBT 악화 액션 차단. Emergency 시 0.50으로 완화하여 강한 조치 허용 |
| `W_WARN/CRIT/EMERG` | threshold 초과 가중치 | 1:2:4 기하급수적 증가로 emergency 도메인에 우선 대응 |
| `UCB` | LinUCB exploration bonus 가중치 | DPP score에 additive bonus로 탐색 촉진. §15 참조 |

### 14.3 도메인별 Threshold (THETA)

```lua
THETA = {
    mem   = { warn = 0.60, crit = 0.80, emerg = 0.90 },
    cpu   = { warn = 0.70, crit = 0.85, emerg = 0.95 },
    gpu   = { warn = 0.70, crit = 0.85, emerg = 0.95 },
    therm = { warn = 0.70, crit = 0.85, emerg = 0.95 },
}
```

**Rust Monitor 정렬**: mem threshold는 `MemoryMonitorConfig::default()`의 available 40/20/10% → used 60/80/90%와 일치한다. cpu/gpu/therm는 `ComputeMonitorConfig`/`ThermalMonitorConfig` 기본값에 정렬되며, emerg=0.95는 정책 레이어 자체 정의이다.

**도메인 키 매핑**: Lua `PRESSURE_KEY` 테이블이 DPP 도메인 키(mem, cpu, gpu, therm)를 Rust 측 pressure 키(memory, cpu, gpu, thermal)로 변환한다.

### 14.4 Z_k 계산 (Multi-Threshold Excess Virtual Queue)

```
Z_k(p, θ) = W_WARN · max(0, p − θ.warn)
           + W_CRIT · max(0, p − θ.crit)
           + W_EMERG · max(0, p − θ.emerg)
```

**특성**:
- 4개 도메인(mem, cpu, gpu, therm) 각각에 독립 계산
- pressure가 모든 threshold 미만이면 Z_k = 0 → 해당 도메인 무시
- 여러 threshold를 동시 초과하면 **누적** 가중 (예: p=0.92에서 mem 도메인은 W_WARN·0.32 + W_CRIT·0.12 + W_EMERG·0.02 = 0.64)
- Z_k가 클수록 해당 도메인의 resource relief를 더 강하게 요구

**Emergency 판별**:
- `has_compute_emergency`: cpu/gpu/therm 중 하나라도 p >= θ.emerg → Safety Override 활성화 (mem 제외)
- `any_emergency`: 모든 도메인 중 하나라도 p >= θ.emerg → latency hard floor 완화

### 14.5 decide() 처리 흐름

```mermaid
flowchart TD
    START["decide(ctx)"] --> JVAL{"0. Joint action 검증<br/>ctx.is_joint_valid()"}
    JVAL -->|"위반"| ERR["Lua error raise"]
    JVAL -->|"통과"| TRIG{"1. Trigger gate<br/>tbt_degraded ∨ mem_low ∨ temp_high?"}
    TRIG -->|"모두 비활성"| TRIG_R{"active 있음?"}
    TRIG_R -->|yes| RESTORE["restore_defaults 반환"]
    TRIG_R -->|no| NOOP["빈 목록 반환"]

    TRIG -->|"하나 이상 활성"| ZK["2. Z_k 계산<br/>4개 도메인 × compute_Zk()"]
    ZK --> EMERG["emergency 판별<br/>has_compute_emergency, any_emergency"]
    EMERG --> ZK_CHK{"3. 모든 Z_k = 0?"}
    ZK_CHK -->|yes| ZK_R{"active 있음?"}
    ZK_R -->|yes| RESTORE2["restore_defaults 반환"]
    ZK_R -->|no| NOOP2["빈 목록 반환"]

    ZK_CHK -->|no| CAND["4. Candidate 열거<br/>single actions + joint actions"]
    CAND --> FLOOR["latency hard floor 결정<br/>any_emergency ? −C_EMERGENCY : −C"]
    FLOOR --> ARGMAX["5. DPP score argmax<br/>score = Σ_k Z_k·r_k − V·max(−lat, 0)<br/>+ UCB·ucb_bonus<br/>hard floor 통과 + score > 0 + 비활성"]
    ARGMAX --> BUILD{"best_action 존재?"}
    BUILD -->|no| SAFETY
    BUILD -->|yes| CMD["build_cmd() → primary_cmds<br/>(joint이면 component별 빌드)"]
    CMD --> SAFETY{"6. Safety Override<br/>has_compute_emergency<br/>∧ throttle 비활성?"}
    SAFETY -->|yes| THR["throttle 강제 추가<br/>(primary에 이미 있으면 skip)"]
    SAFETY -->|no| RET["primary_cmds 반환"]
    THR --> RET
```

**단계별 상세**:

**0단계 — Joint Action 검증**: `ctx.is_joint_valid`가 존재하면 `JOINT_ACTIONS` registry의 모든 joint action에 대해 component 목록을 검증한다. `is_joint_valid(components)`는 Rust 측 `exclusion_groups` (§14.7)를 기반으로 같은 배타 그룹에 속한 액션이 2개 이상 포함되면 `false`를 반환하며, 이 경우 Lua error를 raise하여 joint action 등록 자체의 논리적 오류를 조기에 감지한다. 이는 런타임 방어가 아닌 **정책 스크립트 개발 시 검증**을 목적으로 한다.

**1단계 — Trigger Gate**: Rust 측 `TriggerEngine` (§12.3)이 계산한 3개 trigger를 검사한다. 모든 trigger가 비활성이면 DPP 계산을 건너뛰고 즉시 restore 또는 no-op을 반환한다. 이는 불필요한 DPP 연산을 방지하는 **조기 종료** 최적화이다.

**2단계 — Z_k 계산**: `PRESSURE_KEY` 매핑을 통해 4개 도메인의 pressure를 읽고 `compute_Zk()`를 적용한다. Emergency 판별은 원본 pressure와 THETA.emerg의 직접 비교이다 (Z_k 값이 아님).

**3단계 — Z_k 전부 0 검사**: Trigger가 활성이더라도 모든 pressure가 threshold 미만이면 개입이 불필요하다. 이 검사는 trigger hysteresis의 exit 지연으로 인한 불필요한 액션 발행을 방지한다.

**4단계 — Candidate 열거**: `ctx.coef.relief` 테이블에서 single action을 수집하되, `JOINT_ACTIONS` registry에 속한 component는 제외한다 (joint action이 이미 포함하므로 중복 방지). Joint action의 relief는 학습된 값 우선, 없으면 component 선형 합 fallback (§14.8).

**5단계 — DPP Score Argmax**:

```
score(a) = Σ_k [ Z_k · r_k(a) ]  −  V · max(−lat(a), 0)  +  UCB · ucb_bonus(a)
           ─────────────────────     ─────────────────────     ──────────────────
           resource relief term       latency penalty term      exploration bonus
```

- **Hard floor 필터**: `lat(a) >= lat_floor` 위반 시 candidate에서 제외 (safe set)
- **Score > 0 필터**: 음수 score는 latency 비용이 resource 이득을 초과 → 무개입이 나음
- **Active 필터**: `is_active_any()`로 이미 활성인 액션 중복 적용 방지
- **탐색 복잡도**: O(|singles| + |joints|) — HierarchicalPolicy의 O(2^N) 대비 선형

**6단계 — Safety Override**: DPP 결과와 **독립적으로** compute/thermal emergency 시 throttle을 강제 삽입한다. Memory emergency는 제외한다 — throttle은 compute 부하를 낮추는 액션이며 메모리 부족은 kv_evict 계열로 처리해야 한다.

### 14.6 Pressure Level → Parameter 매핑 (build_cmd)

`build_cmd(action, level, cpu_pressure)`는 선택된 액션과 현재 pressure level에 따라 EngineDirective 파라미터를 결정한다.

**Pressure Level 판정** (memory pressure 기반):

| Level | 조건 | 근거 |
|-------|------|------|
| emergency | mem_pressure >= 0.90 | MemoryMonitor: available <= 10% |
| critical | mem_pressure >= 0.80 | MemoryMonitor: available <= 20% |
| warning | mem_pressure >= 0.60 | MemoryMonitor: available <= 40% |
| normal | otherwise | — |

**Level → 파라미터 테이블 (LEVEL_PARAMS)**:

| Level | keep_ratio | target_bits | 용도 |
|-------|-----------|-------------|------|
| emergency | 0.25 | 2 | kv_evict: 25%만 유지. kv_quant: 2-bit 양자화 |
| critical | 0.50 | 4 | kv_evict: 50% 유지. kv_quant: 4-bit |
| warning | 0.70 | 8 | kv_evict: 70% 유지. kv_quant: 8-bit |
| normal | 0.85 | 16 | kv_evict: 85% 유지. kv_quant: 16-bit (사실상 무변환) |

**액션별 build_cmd 분기**:

| 액션 | 파라미터 | 계산 |
|------|---------|------|
| `kv_evict_h2o`, `kv_evict_sliding`, `kv_merge_d2o` | `keep_ratio` | LEVEL_PARAMS[level].keep_ratio |
| `kv_quant_dynamic` | `target_bits` | LEVEL_PARAMS[level].target_bits |
| `throttle` | `delay_ms` | `max(floor(cpu_pressure * 200 / 20) * 20, 20)` — 20ms 양자화 |
| `set_target_tbt` | `target_ms` | 150 (하드코딩) |
| `layer_skip` | `skip_ratio` | 0.25 (하드코딩) |
| `switch_hw` | `device` | "cpu" (하드코딩) |
| `set_partition_ratio` | `ratio` | 0.5 (하드코딩) |

**Throttle 양자화 근거**: 20ms 단위로 양자화하여 미세한 pressure 변동이 매 tick마다 다른 directive를 생성하는 것을 방지한다 (chattering 억제).

### 14.7 Joint Action Registry

**설계 결정**: HierarchicalPolicy는 2^N 부분 집합을 전수 탐색했으나, 실제 유효한 조합은 소수이다. DPP는 **수동 등록된 arity <= 2 조합**만 허용하여 탐색 공간을 O(N)으로 제한한다.

**등록된 Joint Actions**:

| Joint Key | Components | 사용 시나리오 |
|-----------|-----------|-------------|
| `throttle_plus_layer_skip` | throttle + layer_skip | Compute 위기: 처리 지연 + 레이어 건너뛰기 동시 적용 |

**제거된 Joint Actions**:

| Joint Key | Components | 제거 근거 |
|-----------|-----------|----------|
| ~~`kv_evict_plus_quant`~~ | ~~kv_evict_sliding + kv_quant_dynamic~~ | `kv_quality` 배타 그룹 위반 — 두 액션 모두 KV 캐시 정확도를 훼손하므로 동시 적용 시 이중 품질 저하. `exclusion_groups`에 의해 차단됨 |

**Exclusion Groups**: Rust 측 `LuaPolicy`에 `exclusion_groups: HashMap<String, Vec<String>>` 필드가 있으며, `new_with_exclusions()` 생성자로 주입한다. 이 맵은 같은 품질 차원을 훼손하는 액션들을 그룹으로 묶어 동시 발행을 금지한다 (예: `"kv_quality" → ["kv_evict_sliding", "kv_evict_h2o", "kv_quant_dynamic", "kv_merge_d2o"]`).

**`ctx.is_joint_valid(action_list)`**: build_ctx()에서 Lua closure로 노출된다. 인자로 받은 action 이름 배열에 대해 exclusion_groups의 모든 그룹을 검사하고, 같은 그룹에 속한 액션이 2개 이상이면 `false`를 반환한다. decide() 진입 시 모든 등록된 joint action의 components를 이 함수로 검증하며, 위반 시 Lua error를 raise한다 (§14.5 Step 0).

**Relief 계산**:
1. 학습된 relief가 `ctx.coef.relief[joint_key]`에 존재하면 직접 사용
2. 없으면 **component 선형 합 fallback**: `r_joint[k] = Σ r_comp[k]`
3. 선형 합은 component 간 상호작용을 무시하므로 과대 추정 가능성이 있으나, cold start 시 안전한 근사치

**Active 검사**: Joint action은 component 중 **하나라도** active이면 skip한다 (`is_active_any()`). 이는 부분 중복 적용을 방지한다.

**Candidate 열거 시 중복 방지**: `JOINT_ACTIONS` registry에 속한 single action은 candidate에서 제외된다. 이는 joint와 single이 동시에 후보에 오르는 것을 방지한다.

### 14.8 HierarchicalPolicy 대비 차이 종합

| 측면 | HierarchicalPolicy | LuaPolicy DPP |
|------|-------------------|---------------|
| 압력 계산 | PI Controller (적분+비례) | 직접 정규화 (즉각 반응, §12.1) |
| 모드 판정 | SupervisoryLayer (Normal/Warning/Critical 3-state) | Z_k > 0 여부 + trigger gate (연속적) |
| 액션 선택 | 2^N 전수 탐색 + cost 최소화 | DPP score argmax O(N) |
| 조합 전략 | 임의 부분 집합 | Joint registry (arity <= 2, 수동 등록) |
| Relief 학습 | RLS 13D→4D (상태 조건부) | EWMA 6D scalar (상태 무관, §12.4) |
| 안전 보장 | latency budget + exclusion group | hard floor + Safety Override (DPP 외부) |
| 디에스컬레이션 | 모드 전이 시 restore directive | trigger gate off 시 restore (§14.5 단계 1) |
| 구현 언어 | Rust (컴파일 타임 고정) | Lua (런타임 스크립트, hot-reload 가능) |
| 관측 | 단일 액션 제한 (INV-051) | 동일 — 단일 액션만 추적 (§12.5) |

### 14.9 코드-스펙 매핑

본 섹션은 `spec/22-manager-algorithms.md`의 MGR-DPP-xxx ID에 대응한다.

| spec ID | 설명 | 코드 위치 |
|---------|------|----------|
| DPP 상수 | V, C, C_EMERGENCY, W_* | `policy_default.lua` 26-33행 부근 `DPP` 테이블 |
| THETA | 도메인별 warn/crit/emerg threshold | `policy_default.lua` 39-44행 부근 `THETA` 테이블 |
| Z_k 계산 | compute_Zk() | `policy_default.lua` `compute_Zk` 함수 |
| decide() 전체 흐름 | 6단계 파이프라인 | `policy_default.lua` `decide` 함수 |
| Safety Override | compute/thermal emergency throttle | `decide()` 내 단계 8 |
| Joint actions | 조합 등록 + relief fallback | `JOINT_ACTIONS` 테이블 + `joint_relief` 함수 |
| build_cmd | pressure level → 파라미터 변환 | `build_cmd` 함수 + `LEVEL_PARAMS` 테이블 |

## 15. LinUCB Exploration Bonus

DPP score에 exploration bonus를 추가하여 cold-start bias를 완화한다. 대응 스펙: `spec/22-manager-algorithms.md` §3.9 (MGR-LUCB-010~070).

**파일**: `manager/src/lua_policy.rs` (LinUcbTable, build_feature_vec, build_ctx)

### 15.1 설계 결정

**간결한 P-matrix-only 설계를 채택한 이유**:

| 관점 | Full LinUCB (V⁻¹ + b + theta) | P-matrix-only (채택) |
|------|-------------------------------|---------------------|
| 평균 추정 | theta = V⁻¹ b (자체 추정) | EwmaReliefTable 위임 (이미 수렴 빠름) |
| Uncertainty 추정 | sigma = sqrt(phi^T V⁻¹ phi) | 동일: sigma = sqrt(phi^T P phi) |
| 보관 상태 | D*D + 6*D + 6*D + count | D*D만 (P matrix) |
| 메모리 per action | ~660 bytes (D=13) | ~676 bytes (13^2 f32 = 169 * 4) |
| 구현 복잡도 | b/theta 갱신 + theta=V⁻¹ b 재계산 | Sherman-Morrison P 갱신만 |
| DPP 통합 | relief mean + sigma 분리 노출 | ucb_bonus 스칼라로 단순 additive |

핵심 이유: EwmaReliefTable의 mean 추정이 이미 잘 동작하고 (alpha=0.875, 3~5회 수렴), 부족한 것은 exploration뿐이다. P matrix만 유지하면 기존 EWMA 인프라를 그대로 사용하면서 `ucb_bonus` 스칼라 하나로 DPP score에 additive bonus를 더할 수 있다. Pessimistic safe set, 도메인별 sigma 분리, beta_t 스케줄 등은 불필요한 복잡성으로 판단하여 배제했다.

**왜 hierarchical OnlineLinearEstimator(13D->4D RLS)를 재사용하지 않는가**:
- OnlineLinearEstimator는 `#[cfg(feature = "hierarchical")]` 경로 전용이다.
- 13D feature는 PI output, supervisory mode 등 hierarchical 전용 신호에 의존한다.
- LuaPolicy는 PI를 사용하지 않으므로 (§12.1 직접 정규화) hierarchical feature를 구성할 수 없다.

### 15.2 LinUcbTable 구조체

```rust
pub const LINUCB_FEATURE_DIM: usize = 13;

struct LinUcbTable {
    /// action_name -> D*D P matrix (flat row-major, length = feature_dim^2)
    matrices: HashMap<String, Vec<f32>>,
    feature_dim: usize,   // 13
}
```

| 메서드 | 시그니처 | 동작 |
|--------|---------|------|
| `new()` | `-> Self` | 빈 HashMap, feature_dim=13 |
| `ensure_matrix(action)` | `&mut self, &str` | P 없으면 identity I_13 초기화 (최대 불확실성) |
| `ucb_bonus(action, phi)` | `&self, &str, &[f32] -> f32` | `sqrt(max(0, phi^T P phi))`. P 없으면 `1.0` 반환 (cold-start 최대 탐색) |
| `update(action, phi)` | `&mut self, &str, &[f32]` | Sherman-Morrison: `P <- P - (P phi phi^T P) / (1 + phi^T P phi)` |

설계 결정:
- **P = A⁻¹ 직접 보관**: predict 시 P가 직접 필요하고, Sherman-Morrison으로 P를 incremental update한다. A를 보관하고 매번 역행렬을 계산하는 것보다 효율적 (O(D^2) vs O(D^3)).
- **HashMap 키**: EwmaReliefTable과 동일한 action name 키 구조.
- **flattened row-major**: D=13이므로 169개 f32. 캐시 친화적.
- **b/theta 미보관**: mean relief는 EWMA가 담당하므로 reward 모델링(b, theta)이 불필요.

Sherman-Morrison update (update 메서드 내부):

```
p_phi = P * phi           // D-vector, O(D^2)
denom = 1 + phi^T * p_phi // scalar, O(D)
P -= (p_phi * p_phi^T) / denom  // D*D update, O(D^2)
```

총 update 비용: O(D^2) = O(169). OBSERVATION_DELAY=3초 간격이므로 무시할 수 있는 비용이다.

### 15.3 Feature Vector (13D)

`build_feature_vec()` -- `LuaPolicy` 메서드. 현재 engine_state + trigger_engine에서 13차원 벡터를 구성한다.

```mermaid
flowchart LR
    ES["EngineStatus<br/>(Heartbeat)"] --> KV["[0] kv_occupancy"]
    ES --> GPU["[1] is_gpu"]
    ES --> TOK["[2,6] token_progress"]
    ES --> PF["[3] is_prefill"]
    ES --> DT["[4] kv_dtype_norm"]
    ES --> AA["[7-12] active_* flags"]
    TE["TriggerEngine"] --> TBT["[5] tbt_ratio"]
    KV & GPU & TOK & PF & DT & TBT & AA --> PHI["phi in R^13"]
```

| Index | Name | Source | Range | 계산 |
|-------|------|--------|-------|------|
| 0 | KV_OCCUPANCY | EngineStatus | [0,1] | `kv_cache_utilization.clamp(0, 1)` |
| 1 | IS_GPU | EngineStatus | {0, 1} | `active_device.contains("opencl") -> 1.0` |
| 2 | TOKEN_PROGRESS | EngineStatus | [0,1] | `min(tokens_generated / 2048, 1.0)` |
| 3 | IS_PREFILL | EngineStatus | {0, 1} | `phase == "prefill" -> 1.0` |
| 4 | KV_DTYPE_NORM | EngineStatus | {0.0, 0.5, 1.0} | `f32->0.0, f16->0.5, else->1.0` |
| 5 | TBT_RATIO | TriggerEngine | [0,inf) | `tbt_degradation_ratio().unwrap_or(0.0)` |
| 6 | TOKENS_GEN_NORM | EngineStatus | [0,1] | = TOKEN_PROGRESS (동일 값) |
| 7 | ACTIVE_SWITCH_HW | EngineStatus | {0, 1} | `"switch_hw" in active_actions` |
| 8 | ACTIVE_THROTTLE | EngineStatus | {0, 1} | `"throttle" or "set_target_tbt" in active_actions` |
| 9 | ACTIVE_KV_OFFLOAD | EngineStatus | {0, 1} | `"kv_offload_disk" in active_actions` |
| 10 | ACTIVE_KV_EVICT | EngineStatus | {0, 1} | `"kv_evict_h2o" or "kv_evict_sliding" or "kv_merge_d2o"` |
| 11 | ACTIVE_LAYER_SKIP | EngineStatus | {0, 1} | `"layer_skip" in active_actions` |
| 12 | ACTIVE_KV_QUANT | EngineStatus | {0, 1} | `"kv_quant_dynamic" in active_actions` |

**EngineStatus 미수신 시**: 모든 engine 파생 feature = 0.0 (안전 기본값). TBT_RATIO는 TriggerEngine에서 독립 계산.

**인덱스 2와 6이 동일한 이유**: 초기 설계에서 구분된 의미를 가졌으나 현재는 동일 값. 호환성 유지를 위해 변경하지 않는다.

### 15.4 ucb_bonus 처리 흐름

```mermaid
flowchart TD
    BC["build_ctx()"] --> BF["build_feature_vec() -> phi[13]"]
    BF --> FS["self.feature_state = phi"]
    FS --> LOOP["for action in action_names"]
    LOOP --> UB["linucb.ucb_bonus(action, &phi)"]
    UB --> CHK{"P matrix 존재?"}
    CHK -->|no| DEF["return 1.0<br/>(cold-start 최대 탐색)"]
    CHK -->|yes| CALC["v = P * phi<br/>val = phi^T * v<br/>return sqrt(max(0, val))"]
    DEF & CALC --> SCALE["ucb = result * linucb_alpha"]
    SCALE --> SET["entry.set('ucb_bonus', ucb)"]
    SET --> REL["ctx.coef.relief[action].ucb_bonus = ucb"]
```

**linucb_alpha**: `AdaptationConfig.linucb_alpha` (기본값 `0.5`). Rust 측에서 ucb_bonus에 곱해진 후 Lua에 전달된다. Lua 측에서는 `DPP.UCB` (기본값 `1.0`)로 한번 더 가중한다. 실질적 가중치 = `linucb_alpha * DPP.UCB`.

### 15.5 Sherman-Morrison Update 흐름

```mermaid
sequenceDiagram
    participant OBS as try_complete_observation()
    participant EWMA as EwmaReliefTable
    participant LUCB as LinUcbTable

    Note over OBS: OBSERVATION_DELAY(3s) 경과
    OBS->>OBS: compute actual_relief = before - after
    OBS->>EWMA: observe(action, actual_relief)
    OBS->>LUCB: update(action, &obs.feature_vec)

    Note over LUCB: ensure_matrix(action)
    Note over LUCB: p_phi = P * phi    [O(D^2)]
    Note over LUCB: denom = 1 + phi^T * p_phi    [O(D)]
    Note over LUCB: P <- P - (p_phi * p_phi^T) / denom    [O(D^2)]
```

**feature_vec 스냅샷**: `ObservationContext`에 큐잉 시점의 phi를 저장한다 (`feature_vec: [f32; LINUCB_FEATURE_DIM]`). update 시 이 스냅샷을 사용하므로 "이 상태에서 이 action을 적용하면" 이라는 인과 관계가 보존된다.

**P matrix 단조 감소**: lambda=1.0 (망각 없음). 관측이 누적될수록 P의 대각 성분이 감소하여 ucb_bonus가 줄어든다. 이는 exploitation 수렴을 보장한다.

**EWMA와 동시 호출**: `relief_table.observe()`와 `linucb.update()`는 같은 시점에 호출되며, 각각 mean과 uncertainty를 독립적으로 갱신한다.

### 15.6 linucb_alpha (고정 가중치)

beta_t 스케줄 대신 고정 `linucb_alpha`를 사용한다.

```
Lua에서 받는 ucb_bonus = linucb.ucb_bonus(action, phi) * linucb_alpha
DPP score += DPP.UCB * ucb_bonus
```

| 파라미터 | 위치 | 기본값 | 역할 |
|----------|------|--------|------|
| `linucb_alpha` | `AdaptationConfig` (Rust, `config.rs`) | 0.5 | Rust 측 ucb_bonus 스케일 |
| `DPP.UCB` | `policy_default.lua` (Lua) | 1.0 | Lua 측 score 가중치 |

**beta_t 스케줄을 채택하지 않은 이유**: P matrix 자체가 관측 누적에 따라 단조 감소하므로 ucb_bonus가 자연스럽게 줄어든다. 별도의 observation_count 기반 beta_t 감쇠는 이중 감소가 되어 exploration이 과도하게 빠르게 소멸한다. 고정 alpha가 더 안정적이다.

### 15.7 Lua 인터페이스

**build_ctx() 출력 구조**:

```lua
ctx.coef.relief[action] = {
    gpu = ...,       -- EWMA mean (기존)
    cpu = ...,
    memory = ...,
    thermal = ...,
    lat = ...,
    qos = ...,
    ucb_bonus = 0.42 -- LinUCB bonus * linucb_alpha (신규)
}
```

ucb_bonus는 기존 relief 테이블 **내부** 필드로 노출된다. 별도 `relief_ub` 테이블이 아님. 이는 Lua 측에서 `r.ucb_bonus`로 바로 접근 가능하게 하여 코드 단순성을 유지한다.

**Cold-start 동작**: P matrix 미존재 시 `ucb_bonus(action, phi) = 1.0`을 반환한다. `linucb_alpha=0.5` 적용 후 `DPP.UCB=1.0`과 곱하면 score에 +0.5가 더해져 미탐색 action에 대한 탐색을 촉진한다.

### 15.8 DPP Score 통합 (Additive Bonus)

`policy_default.lua` v2.1.0의 score 수식:

```lua
local score = resource_term - latency_penalty + DPP.UCB * (r.ucb_bonus or 0)
```

기존 DPP score에 단순 additive bonus를 더하는 방식이다. Pessimistic safe set (`is_safe_ucb`)이나 도메인별 sigma 분리는 사용하지 않는다.

| 방식 | 장점 | 단점 |
|------|------|------|
| **Additive bonus (채택)** | 구현 단순, 기존 score 해석 유지 | UCB가 latency 안전성에 반영 안 됨 |
| Pessimistic safe set | latency 보수적 보장 | 복잡, DPP.UCB 외 별도 beta_t 필요 |
| 도메인별 sigma 분리 | 정밀한 exploration | relief_ub 테이블 추가, Lua 복잡도 증가 |

**Hard floor와의 관계**: ucb_bonus는 hard floor 검사(`lat >= lat_floor`) 이후 score에 더해진다. 따라서 latency hard floor는 여전히 보장된다.

### 15.9 EwmaReliefTable 대비 차이 테이블 (갱신)

| 측면 | EwmaReliefTable | LinUcbTable |
|------|----------------|-------------|
| 역할 | Mean 추정 (점추정) | Exploration bonus (uncertainty) |
| 모델 | 지수가중이동평균 (EWMA) | P matrix (= A⁻¹) Sherman-Morrison |
| 입력 | action name + relief[6] | action name + phi[13] |
| 출력 | mean relief[6] (6D) | ucb_bonus 스칼라 (1D) |
| 상태 per action | mean[6] + obs_count | P matrix [13x13] = 169 f32 |
| 메모리 per action | ~28 bytes | ~676 bytes |
| 갱신 비용 | O(6) | O(D^2) = O(169) |
| Cold start | default prior (INV-087) | P = I -> ucb_bonus = 1.0 (최대 탐색) |
| 수렴 | 빠름 (alpha=0.875, 3~5회) | P 단조 감소 (관측마다 shrink) |
| DPP 사용 | `relief[domain]` -> resource term | `ucb_bonus` -> additive bonus in score |
| 갱신 시점 | observe() -- relief delta 필요 | update() -- phi만 필요 (reward 미사용) |
| Lua 노출 | `relief[action].gpu/cpu/...` | `relief[action].ucb_bonus` |

### 15.10 Phase별 구현 현황

**Phase 1 (완료)**: LinUcbTable + DPP additive bonus

구현 파일:
- `manager/src/lua_policy.rs`: `LinUcbTable` struct, `build_feature_vec()`, build_ctx() ucb_bonus 노출
- `manager/scripts/policy_default.lua` v2.1.0: `DPP.UCB` 상수, score에 additive bonus 적용
- `manager/src/config.rs`: `AdaptationConfig.linucb_alpha` (기본 0.5)

테스트 (`lua_policy.rs` 내 유닛 테스트):
- `linucb_cold_start_bonus`: P 미존재 시 ucb_bonus = 1.0
- `linucb_update_reduces_bonus`: observe 후 ucb_bonus 감소 검증
- `linucb_multiple_updates_convergence`: 다수 관측 후 bonus 수렴 검증

**Phase 2 (미착수)**: Pessimistic safe set + 도메인별 sigma 분리

현재 Phase 1의 additive bonus가 충분한지 시뮬레이터 결과를 확인한 후 진행 여부를 결정한다. 필요 시 `policy_safe_lucb.lua`로 별도 작성하여 AB 비교한다.