# Manager Algorithms

> **TL;DR**: Manager Policy Layer 내부 알고리즘의 상세 명세.
> **HierarchicalPolicy (3.1~3.7)**: (1) 도메인별 압력 계산: Compute/Thermal은 PI Controller (gain scheduling, anti-windup), Memory는 임계값 기반 직접 매핑 (OOM 즉각 대응). 전략 패턴으로 교체 가능. (2) Supervisory: peak pressure 기반 3-state 운영 모드 판정. (3) ActionSelector: 2^N 전수 탐색 cross-domain 조합 최적화. (4) ReliefEstimator: RLS 온라인 선형 회귀로 액션 효과 예측.
> **LuaPolicy DPP (3.8)**: Lyapunov Drift-Plus-Penalty 기반 온라인 자원 관리. Multi-threshold virtual queue + latency penalty로 다중 도메인 압력 해소를 선형 스칼라화. EwmaReliefTable로 relief 학습. Production 기본 정책.
> 각 알고리즘은 의사코드, Contract (PRE/POST/INV), example trace를 포함한다.

## 1. Purpose and Scope

이 문서는 Manager Policy Layer의 **알고리즘 상세**를 정의한다. `20-manager.md` MGR-024~030에서 위임된 PI Controller, Supervisory, ActionSelector, ReliefEstimator의 수식, 의사코드, 불변식을 완성한다.

**이 파일이 명세하는 것:**

- PI Controller 수식, gain scheduling, anti-windup 메커니즘
- Supervisory 판정 로직 (전이 테이블은 `21-manager-state.md` 참조)
- ActionSelector 전수 탐색 알고리즘 (후보 필터링, cost 함수, 제약 만족, 파라미터 결정)
- ReliefEstimator RLS 온라인 선형 회귀 수식
- FeatureVector 변환 상세
- Observation relief 계산
- EnergyConstraint max-floor 처리
- process_signal 전체 파이프라인
- LuaPolicy DPP (Drift-Plus-Penalty) 결정 알고리즘 — multi-threshold virtual queue, safe set, joint action, safety override, trigger engine

**이 파일이 명세하지 않는 것:**

- Manager 아키텍처 개요 → `20-manager.md`
- 상태 머신 전이 테이블 → `21-manager-state.md`
- 설정 스키마, SystemSignal 필드 → `23-manager-data.md`
- 프로토콜 메시지, 시퀀스 → `10-protocol.md`, `11-protocol-messages.md`, `12-protocol-sequences.md`

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **measurement** | 원시 센서 값을 [0, 1]로 정규화한 값. PI Controller 입력. |
| **error** | max(measurement - setpoint, 0). setpoint 초과분. |
| **pressure intensity** | PI Controller 출력 [0, 1]. 단일 도메인의 리소스 압력 정도. |
| **PressureVector** | 3차원 벡터 (compute, memory, thermal). 각 원소는 해당 도메인의 pressure intensity. |
| **peak pressure** | PressureVector.max(). 전체 도메인 중 최대 압력 값. |
| **ReliefVector** | 4차원 벡터 (compute, memory, thermal, latency). 액션의 도메인별 완화 효과. latency 음수는 악화. |
| **FeatureVector** | 13차원 시스템 상태 벡터. ReliefEstimator 입력. EngineStatus Heartbeat에서 갱신. |
| **coverage** | 조합의 relief가 pressure를 충족하는 정도. best-effort 비교에 사용. |
| **GainZone** | gain scheduling 구간 정의. measurement가 `above` 이상일 때 해당 `kp`를 적용. |
| **RLS** | Recursive Least Squares. 온라인 선형 회귀 갱신 알고리즘. |

## 3. Specification

### 3.1 PI Controller [MGR-ALG-010 ~ MGR-ALG-016]

#### MGR-ALG-010: Pressure Calculation Strategy

도메인별 독립 압력 계산기가 원시 측정값 [0, 1]을 연속 pressure intensity [0, 1]로 변환한다. Compute와 Thermal은 PI Controller를, Memory는 임계값 기반 직접 매핑을 사용한다. *(MUST)*

> **전략 패턴**: 도메인별 압력 계산 전략은 교체 가능하다. 현재 구성은 PI(Compute, Thermal) + 임계값(Memory)이지만, PID, 학습 기반 등 다른 전략으로 대체할 수 있다. 각 전략은 동일한 인터페이스(measurement → pressure intensity)를 따른다.

##### PI Controller (Compute, Thermal)

```
PRE:  measurement in [0, 1], dt > 0
POST: output in [0, 1]

function update(measurement, dt) -> pressure_intensity:
    error = max(measurement - setpoint, 0)

    if can_act:
        integral = clamp(integral + error * dt, 0, integral_clamp)

    kp_eff = effective_kp(measurement)
    output = clamp(kp_eff * error + ki * integral, 0, 1)
    return output
```

- `error`: 단방향. measurement < setpoint 이면 error = 0, 출력 = 0.
- `kp_eff`: gain scheduling 적용 시 `effective_kp()`, 미적용 시 고정 `kp`.
- `output`: pressure intensity. 0 = 압력 없음, 1 = 최대 압력.

#### MGR-ALG-011: Gain Scheduling (effective_kp)

measurement 크기에 따라 비례 이득(Kp)을 가변 적용한다. *(MUST)*

```
PRE:  gain_zones sorted ascending by 'above' field
POST: return kp >= 0

function effective_kp(measurement) -> kp:
    for zone in gain_zones.reverse():
        if measurement >= zone.above:
            return zone.kp
    return base_kp
```

- gain_zones가 비어있으면 항상 base_kp 반환 (고정 Kp와 동일).
- 현재 memory 도메인만 gain scheduling을 지원한다 (설정: `memory_gain_zones`). compute와 thermal은 고정 Kp.

#### MGR-ALG-012: Anti-Windup

적분항의 무한 누적을 방지하는 두 가지 메커니즘을 제공한다. *(MUST)*

```
INV-030: can_act = false 일 때 integral 값은 변하지 않는다.
INV-031: integral in [0, integral_clamp]  (항상 유지)
```

- `can_act` 플래그: 해소 가능한 액션이 존재하지 않으면 false로 설정하여 적분 누적을 동결한다.
- `integral_clamp`: 적분 상한. 장시간 pressure가 해소되지 않더라도 적분이 무한히 커지지 않는다.
- `reset_integral()`: integral을 0으로 초기화한다 (모드 전이 시 등에 사용).

#### MGR-ALG-013: PI 인스턴스 (Compute, Thermal)

| 인스턴스 | kp (default) | ki (default) | setpoint (default) | integral_clamp | gain_zones |
|---------|------|------|------|------|------|
| pi_compute | 1.5 | 0.3 | 0.70 | 2.0 | (없음) |
| pi_thermal | 1.0 | 0.2 | 0.80 | 2.0 | (없음) |

#### MGR-ALG-013a: Memory 임계값 기반 직접 매핑

Memory 도메인은 PI Controller를 사용하지 않는다. OOM은 초 단위로 발생하므로 PI의 평활화 지연이 치명적이다. 대신 Monitor의 측정값을 직접 압력에 매핑한다. *(MUST)*

```
function update_memory_pressure(signal):
    m = clamp(1 - available_bytes / total_bytes, 0, 1)
    pressure.memory = m    // 평활화 없이 직접 매핑
```

> **참고 (non-normative)**: 현재 구현은 raw measurement 직접 전달(passthrough)이다. 향후 비선형 매핑(구간별 기울기), Level 기반 이산 매핑 등으로 전략을 교체할 수 있다. 핵심 제약은 **PI 평활화를 거치지 않는다**는 것이다.

각 인스턴스는 독립적으로 상태(integral)를 유지하며 서로 영향을 주지 않는다. *(MUST)*

#### MGR-ALG-014: Measurement Normalization (update_pressure)

SystemSignal에서 도메인별 측정값을 추출하여 [0, 1]로 정규화한 뒤 해당 PI 인스턴스에 입력한다. *(MUST)*

| SystemSignal | 측정값 변환 | 압력 계산 전략 |
|-------------|-----------|--------------|
| MemoryPressure | `m = clamp(1 - available_bytes / total_bytes, 0, 1)` | **직접 매핑** (MGR-ALG-013a) |
| ThermalAlert | `m = clamp(temperature_mc / 85000, 0, 1)` | PI Controller (pi_thermal) |
| ComputeGuidance | `m = max(clamp(cpu_usage_pct/100, 0, 1), clamp(gpu_usage_pct/100, 0, 1))` | PI Controller (pi_compute) |
| EnergyConstraint | `m = clamp(1 - battery_pct / 100, 0, 1)` | compute PI 보조 (MGR-ALG-015) |

- total_bytes = 0 인 경우 m = 0 (방어적 처리).
- Thermal 정규화 기준: 85000 mc (85 C) = 1.0.
- Compute: CPU와 GPU 사용률 중 최대값.

#### MGR-ALG-015: EnergyConstraint Raw Processing

EnergyConstraint는 별도 PI 인스턴스 없이 compute PI에 보조 기여한다. **level이 아닌 raw 값(battery_pct)에서 직접 측정값을 산출한다.** *(MUST)*

```
PRE:  battery_pct in [0, 100] (EnergyConstraint raw 필드)
POST: compute pressure 갱신

energy_measurement = clamp(1.0 - battery_pct / 100.0, 0, 1) * 0.5
combined = max(pressure.compute, energy_measurement)
compute_pressure = pi_compute.update(combined, dt)
```

| battery_pct | energy_measurement | 의미 |
|-------------|-------------------|------|
| 100% | 0.0 | 배터리 완충, compute 영향 없음 |
| 70% | 0.15 | 정상 사용 |
| 30% | 0.35 | 배터리 부족, compute floor 상승 |
| 5% | 0.475 | 극단 부족, compute 강한 제약 |

> **설계 원칙**: Monitor는 raw 데이터만 전달하고, Policy가 raw 값에서 직접 압력을 계산한다. 이전 설계에서 사용하던 `level_to_measurement(level)` 경로는 폐기한다. 전력 상태의 시간 스케일(분~시간)이 PI의 초 단위 제어와 상이하므로 별도 도메인 대신 compute 보조 신호로 반영한다. 0.5 가중치는 energy가 compute를 과도하게 지배하지 않도록 한다.

#### MGR-ALG-016: Elapsed dt Calculation

도메인별 실측 dt를 계산하여 PI 적분의 시간 정확도를 보장한다. *(MUST)*

```
PRE:  domain in {"memory", "thermal", "compute"}
POST: dt in [0.001, 10.0]

function elapsed_dt(domain) -> dt:
    now = clock()
    if last_signal_time[domain] exists:
        dt = now - last_signal_time[domain]
    else:
        dt = default_dt   // 0.1 s
    last_signal_time[domain] = now
    return clamp(dt, 0.001, 10.0)
```

- 도메인별 독립 dt 계산.
- 첫 신호 시 default_dt (0.1 s) 사용.
- [0.001, 10.0] 범위 clamp로 이상값 방지.

**Example Trace (PI Controller — Thermal):**

```
pi_thermal: kp=1.0, ki=0.2, setpoint=0.80, integral_clamp=2.0

t=0s: measurement=0.70 (59500mc / 85000)
      error = max(0.70 - 0.80, 0) = 0
      output = 0.0

t=1s: measurement=0.85 (72250mc), dt=1.0
      error = 0.85 - 0.80 = 0.05
      integral = 0 + 0.05 * 1.0 = 0.05
      output = clamp(1.0 * 0.05 + 0.2 * 0.05, 0, 1) = 0.06

t=2s: measurement=0.90 (76500mc), dt=1.0
      error = 0.10
      integral = 0.05 + 0.10 * 1.0 = 0.15
      output = clamp(1.0 * 0.10 + 0.2 * 0.15, 0, 1) = 0.13

t=3s: measurement=0.95 (80750mc), dt=1.0
      error = 0.15
      integral = 0.15 + 0.15 * 1.0 = 0.30
      output = clamp(1.0 * 0.15 + 0.2 * 0.30, 0, 1) = 0.21
```

**Example Trace (Memory 직접 매핑):**

```
t=0s: available=1600MB, total=2000MB
      m = 1 - 1600/2000 = 0.20
      pressure.memory = 0.20

t=0.1s: available=800MB (외부 앱이 800MB 할당)
      m = 1 - 800/2000 = 0.60
      pressure.memory = 0.60    ← PI 지연 없이 즉시 반영

t=0.2s: available=200MB (추가 할당)
      m = 1 - 200/2000 = 0.90
      pressure.memory = 0.90    ← 100ms 만에 Supervisory Critical 진입 가능
```

---

### 3.2 Supervisory Judgment Logic [MGR-ALG-020 ~ MGR-ALG-024]

#### MGR-ALG-020: Supervisory Evaluate

PressureVector를 받아 시스템 운영 모드를 결정한다. *(MUST)*

```
PRE:  pressure is PressureVector
POST: mode in {Normal, Warning, Critical}

function evaluate(pressure) -> OperatingMode:
    peak = pressure.max()
    mode = next_mode(peak, now())
    return mode
```

#### MGR-ALG-021: next_mode Judgment

전이 테이블(`21-manager-state.md` MGR-055)에 따라 다음 모드를 결정한다. *(MUST)*

```
function next_mode(peak, now) -> OperatingMode:
    match current_mode:
        Normal:
            if peak >= critical_threshold:
                stable_since = None; return Critical
            if peak >= warning_threshold:
                stable_since = None; return Warning
            stable_since = None; return Normal

        Warning:
            if peak >= critical_threshold:
                stable_since = None; return Critical
            if peak >= warning_threshold:
                stable_since = None; return Warning
            if peak < warning_release:
                return try_de_escalate(now, Normal)
            // warning_release <= peak < warning_threshold
            stable_since = None; return Warning

        Critical:
            if peak >= critical_threshold:
                stable_since = None; return Critical
            if peak < critical_release:
                return try_de_escalate(now, Warning)
            // critical_release <= peak < critical_threshold
            stable_since = None; return Critical

function try_de_escalate(now, target_mode) -> OperatingMode:
    if stable_since is None:
        stable_since = now
        return current_mode       // 첫 확인, 유지
    if now - stable_since >= hold_time:
        stable_since = None
        return target_mode        // hold_time 경과, 1단계 하강
    return current_mode           // hold_time 미충족, 유지
```

```
INV-032: 에스컬레이션은 즉시. Normal -> Critical 직행 가능.
INV-033: 디에스컬레이션은 반드시 1단계씩. Critical -> Normal 직행 불가. (MGR-C03 재확인)
```

#### MGR-ALG-022: Peak Pressure Calculation

```
function peak(pressure: PressureVector) -> float:
    return max(pressure.compute, pressure.memory, pressure.thermal)
```

#### MGR-ALG-023: Supervisory Default Parameters

| 파라미터 | 기본값 | 용도 |
|---------|--------|------|
| warning_threshold | 0.40 | peak >= 이 값이면 Warning 진입 |
| critical_threshold | 0.70 | peak >= 이 값이면 Critical 진입 |
| warning_release | 0.25 | peak < 이 값이 hold_time 지속 시 Normal 복귀 |
| critical_release | 0.50 | peak < 이 값이 hold_time 지속 시 Warning 복귀 |
| hold_time | 4.0 s | 디에스컬레이션 안정화 대기 |

```
INV-034: warning_release < warning_threshold
INV-035: critical_release < critical_threshold
INV-036: warning_threshold < critical_threshold
```

#### MGR-ALG-024: needs_action Judgment

액션 발행 필요 여부를 판정한다. *(MUST)*

```
PRE:  mode in {Normal, Warning, Critical}
POST: needs_action: bool

function evaluate_needs_action(mode, prev_mode, pressure, last_acted_pressure):
    if mode == Normal:
        return false
    mode_changed = (mode != prev_mode)
    pressure_increased = pressure.any_domain_exceeds(last_acted_pressure, 1.2)
    return mode_changed OR pressure_increased

function any_domain_exceeds(pressure, reference, factor) -> bool:
    return pressure.compute > reference.compute * factor
        OR pressure.memory  > reference.memory  * factor
        OR pressure.thermal > reference.thermal * factor
```

- Normal 모드에서는 액션을 발행하지 않는다.
- 모드 변경 없이도 압력이 20% 이상 증가하면 추가 액션이 필요하다.

**Example Trace (Supervisory):**

```
config: warning=0.4, critical=0.7, w_release=0.25, c_release=0.50, hold=4.0s

t=0s:  peak=0.1  -> Normal  (0.1 < 0.4)
t=1s:  peak=0.5  -> Warning (0.5 >= 0.4, 즉시 에스컬레이션)
t=2s:  peak=0.8  -> Critical(0.8 >= 0.7, Warning 경유 없이 직행)
t=3s:  peak=0.3  -> Critical(0.3 < 0.50, stable_since = t3)
t=4s:  peak=0.2  -> Critical(elapsed = 1s < 4s hold_time)
t=7s:  peak=0.2  -> Warning (elapsed = 4s >= 4s, 1단계 하강)
t=8s:  peak=0.1  -> Warning (0.1 < 0.25, stable_since = t8)
t=12s: peak=0.1  -> Normal  (elapsed = 4s >= 4s, 1단계 하강)
```

---

### 3.3 ActionSelector Exhaustive Search [MGR-ALG-030 ~ MGR-ALG-037]

#### MGR-ALG-030: ActionSelector Overall Flow

Cross-domain 조합 최적화를 수행한다. Stateless: 모든 상태는 호출 인자로 전달된다. *(MUST)*

```
PRE:  registry, estimator, pressure, mode, engine_state, qcf_values,
      latency_budget, active_actions, available_actions 가 유효
POST: ActionCommand 목록 (빈 목록 가능)

function select(...) -> list of ActionCommand:
    // Phase 0: 조기 종료
    if pressure.compute <= 0 AND pressure.memory <= 0 AND pressure.thermal <= 0:
        return []

    // Phase 1: 후보 필터링
    candidate_ids = filter_candidates(registry, mode, active_actions, available_actions)
    if candidate_ids is empty:
        return []

    // Phase 2: 후보별 relief 예측 + cost 계산
    candidates = []
    for action in candidate_ids:
        relief = estimator.predict(action, engine_state)
        cost = compute_cost(action, registry, qcf_values)
        candidates.append({action, relief, cost})

    // Phase 3: 최적 조합 탐색 (전수 탐색)
    selected = find_optimal(candidates, pressure, registry, latency_budget)

    // Phase 4: 파라미터 결정
    commands = []
    for action in selected:
        params = parametrize(action, pressure, registry)
        commands.append(ActionCommand{action, Apply(params)})
    return commands
```

#### MGR-ALG-031: Phase 1 -- Candidate Filtering (filter_candidates)

등록된 액션에서 현재 상태에 적합한 후보를 선별한다. *(MUST)*

```
PRE:  registry에 등록된 모든 액션
POST: 필터링된 ActionId 목록

function filter_candidates(registry, mode, active_actions, available_actions):
    result = []
    for meta in registry.all_actions():
        // Rule 1: Warning 모드에서 Lossy 제외
        if mode == Warning AND meta.kind == Lossy:
            continue
        // Rule 2: 이미 활성 중인 액션 제외
        if meta.id in active_actions:
            continue
        // Rule 3: Engine이 보고한 실행 가능 목록 필터
        //         (비어있으면 필터링 안 함 -- backward compatibility)
        if available_actions is not empty AND meta.id not in available_actions:
            continue
        result.append(meta.id)
    return result
```

```
INV-037: Warning 모드에서 Lossy 액션은 선택되지 않는다.
INV-038: 이미 활성 중인 액션은 재선택되지 않는다.
```

#### MGR-ALG-032: Phase 2 -- Cost Calculation (compute_cost)

```
PRE:  action in registered actions
POST: cost >= 0

function compute_cost(action, registry, qcf_values):
    meta = registry.get(action)
    if meta is None:
        return INFINITY
    if meta.kind == Lossless:
        return 0.0
    // Lossy: QCF 값 사용. 없으면 INFINITY (사실상 선택 불가)
    if action in qcf_values:
        return qcf_values[action]
    return INFINITY
```

```
INV-039: Lossless 액션의 cost는 항상 0이다.
INV-040: QCF 값이 없는 Lossy 액션은 INFINITY cost로 사실상 선택되지 않는다.
```

#### MGR-ALG-033: Phase 3 -- Exhaustive Search (find_optimal)

모든 부분 집합(2^N)을 탐색하여 제약을 만족하고 cost가 최소인 조합을 선택한다. *(MUST)*

```
PRE:  candidates 목록 (N개), N <= 8 (등록 가능 액션 최대 8종)
POST: 최적 ActionId 조합

function find_optimal(candidates, pressure, registry, latency_budget):
    N = len(candidates)
    total_masks = 2^N

    best_cost = INFINITY
    best_mask = None

    best_coverage = -INFINITY
    best_effort_mask = None

    for mask in 0..total_masks:
        selected = indices where bit is set in mask

        // Constraint 1: Exclusion group -- 동일 그룹 액션 동시 선택 금지
        if has_exclusion_conflict(selected, candidates, registry):
            continue

        // 합산 relief 계산
        total_relief = sum of candidates[i].relief for i in selected

        // Constraint 2: Latency budget -- 안전 제약
        if total_relief.latency < -latency_budget:
            continue

        // Constraint 3: Pressure 충족 검사
        satisfies = total_relief.compute >= pressure.compute
                AND total_relief.memory  >= pressure.memory
                AND total_relief.thermal >= pressure.thermal

        if satisfies:
            total_cost = sum of candidates[i].cost for i in selected
            if total_cost < best_cost:
                best_cost = total_cost
                best_mask = mask
        else:
            // Best-effort: coverage 최대화
            coverage = min(total_relief.compute, pressure.compute)
                     + min(total_relief.memory,  pressure.memory)
                     + min(total_relief.thermal, pressure.thermal)
            if coverage > best_coverage:
                best_coverage = coverage
                best_effort_mask = mask

    // 완전 해소 가능 -> cost 최소 조합 우선
    // 불가능 -> coverage 최대 조합 (best-effort)
    // 동일 coverage의 best-effort 조합이 여러 개이면 마지막으로 발견된 mask를 선택한다 (구현 순서 의존)
    chosen = best_mask if best_mask exists, else best_effort_mask
    return actions from chosen mask
```

```
INV-041: 동일 배타 그룹의 액션은 하나의 조합에 동시 포함되지 않는다. (INV-016 재확인)
INV-042: 조합의 총 latency 악화가 latency_budget을 초과하면 해당 조합은 배제된다.
INV-043: 완전 해소 가능한 조합이 존재하면, best-effort 조합보다 항상 우선한다.
```

#### MGR-ALG-034: Exclusion Conflict Check

```
function has_exclusion_conflict(indices, candidates, registry) -> bool:
    for i in 0..len(indices):
        for j in (i+1)..len(indices):
            a = candidates[indices[i]].action
            b = candidates[indices[j]].action
            if registry.is_excluded(a, b):
                return true
    return false
```

- `is_excluded`: 두 액션이 동일 배타 그룹에 속하면 true.

#### MGR-ALG-035: Phase 4 -- Parameter Determination (parametrize)

액션의 primary domain에 대한 pressure intensity에 따라 파라미터를 선형 보간한다. *(MUST)*

```
PRE:  action에 param_range 존재
POST: ActionParams with value in [range.min, range.max]

function parametrize(action, pressure, registry):
    meta = registry.get(action)
    if meta is None OR meta.param_range is None:
        return default ActionParams

    range = meta.param_range
    intensity = pressure[action.primary_domain()]
    intensity = clamp(intensity, 0, 1)

    // 선형 보간: intensity=0 -> range.max (보수적)
    //            intensity=1 -> range.min (공격적)
    value = range.max - intensity * (range.max - range.min)
    value = clamp(value, range.min, range.max)

    return ActionParams{range.param_name: value}
```

| ActionId | primary_domain | param_name | range [min, max] | intensity 0 | intensity 1 |
|----------|---------------|------------|-------------------|-------------|-------------|
| Throttle | Compute | delay_ms | [0.0, 100.0] | 100.0 (보수적) | 0.0 (공격적) |
| KvEvictSliding | Memory | keep_ratio | [0.3, 0.9] | 0.9 (보수적) | 0.3 (공격적) |
| KvEvictH2o | Memory | keep_ratio | [0.3, 0.9] | 0.9 (보수적) | 0.3 (공격적) |
| KvMergeD2o | Memory | keep_ratio | [0.3, 0.9] | 0.9 (보수적) | 0.3 (공격적) |
| KvQuantDynamic | Memory | target_bits | [4.0, 8.0] | 8.0 (보수적) | 4.0 (공격적) |
| LayerSkip | Compute | skip_layers | [1.0, 8.0] | 8.0 (보수적) | 1.0 (공격적) |
| SwitchHw | Compute | (없음) | -- | -- | -- |
| KvOffloadDisk | Memory | (없음) | -- | -- | -- |

```
INV-044: parametrize 출력 value는 항상 [range.min, range.max] 범위 내이다.
INV-045: primary_domain 매핑: SwitchHw, Throttle, LayerSkip -> Compute.
         KvOffloadDisk, KvEvictSliding, KvEvictH2o, KvMergeD2o, KvQuantDynamic -> Memory.
```

#### MGR-ALG-036: ActionCommand to EngineCommand Conversion

ActionCommand를 와이어 포맷의 EngineCommand로 변환한다. *(MUST)*

| ActionId | EngineCommand | 파라미터 추출 |
|----------|-------------|-------------|
| SwitchHw | SwitchHw { device: "cpu" } | 고정 |
| Throttle | Throttle { delay_ms } | params["delay_ms"] as u64 |
| KvEvictSliding | KvEvictSliding { keep_ratio } | params["keep_ratio"] |
| KvEvictH2o | KvEvictH2o { keep_ratio } | params["keep_ratio"] |
| KvMergeD2o | KvMergeD2o { keep_ratio } | params["keep_ratio"] |
| KvOffloadDisk | KvEvictSliding { keep_ratio: 0.8 } | fallback 고정값 |
| KvQuantDynamic | KvQuantDynamic { target_bits } | params["target_bits"] as u8 |
| LayerSkip | LayerSkip { skip_ratio } | skip_layers / total_layers, clamp [0, 1] |

- KvOffloadDisk는 현재 fallback으로 KvEvictSliding(keep_ratio=0.8)을 사용한다.
- Operation::Release는 None을 반환한다 (restore directive에서 일괄 처리).

#### MGR-ALG-037: Complexity Analysis

- 등록 가능 액션 최대 8종 (MGR-028 참조) -> 2^8 = 256 조합.
- 각 조합에서 exclusion 검사 O(N^2), relief 합산 O(N).
- 총 복잡도: O(2^N * N^2), N <= 8 -> 상수 시간 (16384 연산 상한).
- ActionSelector는 stateless이므로 메모리 오버헤드 없음.

**Example Trace (ActionSelector):**

```
상황: mode=Critical, memory pressure=0.7, compute pressure=0.0
등록 액션: switch_hw (lossless), kv_evict_sliding (lossy, qcf=0.5),
           kv_evict_h2o (lossy, qcf=2.0)
배타 그룹: eviction = {kv_evict_sliding, kv_evict_h2o, kv_merge_d2o}
latency_budget = 1.0

Phase 1 (filter): Critical -> lossy 허용. active=[], available=[]
  candidates = [switch_hw, kv_evict_sliding, kv_evict_h2o]

Phase 2 (predict + cost):
  switch_hw:        relief=(0.5, 0.0, 0.3, -0.1), cost=0.0
  kv_evict_sliding: relief=(0.0, 0.7, 0.0,  0.0), cost=0.5
  kv_evict_h2o:     relief=(0.0, 0.6, 0.0,  0.0), cost=2.0

Phase 3 (find_optimal): 2^3 = 8 masks
  mask=000 (empty):       relief=(0,0,0,0)          -> 미충족, coverage=0
  mask=001 (switch_hw):   relief=(0.5,0,0.3,-0.1)   -> 미충족 (mem 0 < 0.7)
  mask=010 (sliding):     relief=(0,0.7,0,0)         -> 충족, cost=0.5
  mask=011 (sw+sliding):  relief=(0.5,0.7,0.3,-0.1)  -> 충족, cost=0.5
  mask=100 (h2o):         relief=(0,0.6,0,0)          -> 미충족 (0.6 < 0.7)
  mask=101 (sw+h2o):      relief=(0.5,0.6,0.3,-0.1)  -> 미충족
  mask=110 (sliding+h2o): EXCLUSION CONFLICT -> skip
  mask=111 (all):         EXCLUSION CONFLICT -> skip

  best: mask=010, cost=0.5 (sliding 단독)

Phase 4 (parametrize): kv_evict_sliding, primary_domain=Memory, intensity=0.7
  value = 0.9 - 0.7 * (0.9 - 0.3) = 0.9 - 0.42 = 0.48
  -> ActionCommand{KvEvictSliding, Apply{keep_ratio: 0.48}}
```

---

### 3.4 ReliefEstimator Online Learning [MGR-ALG-040 ~ MGR-ALG-047]

#### MGR-ALG-040: ReliefEstimator Interface

액션 효과를 온라인 학습으로 예측하는 Strategy 인터페이스이다. 현재 구현체: OnlineLinearEstimator. *(MUST)*

```
trait ReliefEstimator:
    predict(action, state: FeatureVector) -> ReliefVector
    observe(action, state: FeatureVector, actual: ReliefVector)
    save(path)
    load(path)
    observation_count(action) -> u32
```

- 액션별 독립 모델: `models: HashMap<ActionId, LinearModel>`.
- predict는 읽기 전용. observe만 모델을 생성/갱신한다.

#### MGR-ALG-041: LinearModel Structure

```
모델: relief = W * phi + b

  W: 4 x D 가중치 행렬 (4 relief 차원 x D feature)
  b: 4차원 bias 벡터
  P: D x D 역공분산 행렬 (RLS)
  lambda: 망각 인수 (forgetting factor)
  observation_count: 누적 관측 횟수

D = FEATURE_DIM = 13
```

#### MGR-ALG-042: Predict Algorithm

```
PRE:  phi is D-dim feature vector
POST: ReliefVector (4-dim)

function predict(action, state) -> ReliefVector:
    model = models[action]
    if model is None OR model.observation_count == 0:
        return default_relief(action)     // 사전 지식 기반 prior

    for dim in 0..4:
        vals[dim] = dot(model.weights[dim], state.values) + model.bias[dim]

    return ReliefVector{compute: vals[0], memory: vals[1],
                        thermal: vals[2], latency: vals[3]}
```

- predict 호출만으로는 모델을 생성하지 않는다 (observe 시에만 생성).

#### MGR-ALG-043: Default Relief Prior

학습 데이터가 없을 때(Absent/Initialized 상태) 사용하는 도메인 기반 사전 지식. *(MUST)*

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

> **참고 (non-normative)**: `21-manager-state.md` MGR-086 테이블과 동일하다.

#### MGR-ALG-044: RLS Update Algorithm

실측값으로 모델을 갱신하는 Recursive Least Squares 알고리즘이다. *(MUST)*

```
PRE:  phi is D-dim, actual is ReliefVector, lambda > 0
POST: W, b, P 갱신, observation_count++

function update(phi, actual):
    actual_arr = [actual.compute, actual.memory, actual.thermal, actual.latency]
    lr_bias = 0.1

    // ---- RLS Gain Vector ----
    // P_phi = P * phi   (D-vector)
    for i in 0..D:
        P_phi[i] = sum(P[i][j] * phi[j]  for j in 0..D)

    // denom = lambda + phi^T * P * phi   (scalar)
    denom = lambda + sum(phi[i] * P_phi[i]  for i in 0..D)

    // k = P_phi / denom   (D-vector, gain vector)
    for i in 0..D:
        k[i] = P_phi[i] / denom

    // ---- 4 relief 차원 각각 가중치 갱신 ----
    for dim in 0..4:
        // 현재 W로 예측
        predicted = dot(W[dim], phi) + b[dim]
        error = actual_arr[dim] - predicted

        // W 갱신: W[dim] += k * error
        for i in 0..D:
            W[dim][i] += k[i] * error

        // Bias 갱신: EMA (갱신된 W로 재예측 후 잔여 오차)
        predicted_new = dot(W[dim], phi) + b[dim]
        residual = actual_arr[dim] - predicted_new
        b[dim] += lr_bias * residual

    // ---- P matrix 갱신 ----
    // phi^T * P: D-vector
    for j in 0..D:
        phi_t_P[j] = sum(phi[i] * P[i][j]  for i in 0..D)

    // P = (P - k * phi_t_P) / lambda
    for i in 0..D:
        for j in 0..D:
            P[i][j] = (P[i][j] - k[i] * phi_t_P[j]) / lambda

    observation_count += 1
```

```
INV-046: RLS gain vector k는 P matrix와 phi의 함수이다. lambda는 망각 인수로
         과거 관측의 가중치를 점진적으로 감소시킨다.
INV-047: bias는 W 갱신 후의 잔여 오차에 EMA(lr=0.1)를 적용하여 별도 갱신한다.
         RLS의 P matrix는 W만 다룬다.
INV-048: P matrix는 D x D 대칭 양정치 행렬이다. 초기값은 100 * I (큰 초기 불확실성).
```

#### MGR-ALG-045: Model Initialization

```
function new_model(feature_dim, forgetting_factor):
    W = zeros(4, feature_dim)
    b = [0, 0, 0, 0]
    P = 100.0 * I(feature_dim)     // identity x 100
    lambda = forgetting_factor
    observation_count = 0
```

- P의 큰 초기값(100 x I)은 초기 관측에 높은 학습률을 부여한다.
- observe() 호출 시에만 모델을 생성한다 (predict만으로는 생성하지 않음).

#### MGR-ALG-046: Forgetting Factor

```
기본값: lambda = 0.995

의미: 각 갱신 시 이전 관측의 유효 가중치가 lambda^t로 감쇠.
- lambda = 0.995 -> 반감기 ~ 138 관측  (ln(0.5) / ln(0.995))
- 시스템 상태 변화(하드웨어, 모델, 워크로드)에 적응.

INV-049: lambda in (0, 1]. lambda = 1.0이면 forgetting 없음 (표준 RLS).
```

#### MGR-ALG-047: Model Persistence

```
function save(path):
    saved = {feature_dim, forgetting_factor,
             models: {action_key: model  for each model}}
    write_json(path, saved)

function load(path):
    saved = read_json(path)
    feature_dim = saved.feature_dim
    forgetting_factor = saved.forgetting_factor
    models = {key_to_action(key): model  for key, model in saved.models}
```

- action_to_key 매핑: ActionId <-> snake_case 문자열 (예: SwitchHw <-> "switch_hw").
- 파일 부재 시 빈 models로 시작 (비치명적).
- 프로세스 종료 시 save, 시작 시 load (`20-manager.md` MGR-048 참조).

**Example Trace (RLS, simplified D=3):**

```
D=3, lambda=0.995, lr_bias=0.1
초기: W=[[0,0,0], [0,0,0], [0,0,0], [0,0,0]]
      b=[0, 0, 0, 0]
      P=100*I

Observe #1: phi=[1, 0, 0], actual=[0.5, 0.7, 0, 0]

  P_phi = [100, 0, 0]
  denom = 0.995 + 1*100 = 100.995
  k = [0.990, 0, 0]

  dim=0 (compute):
    predicted = 0+0 = 0, error = 0.5
    W[0] += [0.990*0.5, 0, 0] = [0.495, 0, 0]
    predicted_new = 0.495, residual = 0.005
    b[0] += 0.1*0.005 = 0.0005

  dim=1 (memory):
    predicted = 0, error = 0.7
    W[1] += [0.990*0.7, 0, 0] = [0.693, 0, 0]
    predicted_new = 0.693, residual = 0.007
    b[1] += 0.1*0.007 = 0.0007

  P update:
    phi_t_P = [100, 0, 0]
    P[0][0] = (100 - 0.990*100) / 0.995 = 1.005 / 0.995 ~ 1.010
    (다른 원소는 0 또는 미미한 변화)

  -> 1회 관측만으로 W[0][0] ~ 0.495 (actual compute 0.5에 근접)
```

---

### 3.5 FeatureVector Transform [MGR-ALG-050 ~ MGR-ALG-053]

#### MGR-ALG-050: FeatureVector Dimensions

13차원 시스템 상태 벡터. ReliefEstimator의 입력으로 사용된다. *(MUST)*

| Index | Name | Source (EngineStatus) | Transform |
|-------|------|------|------|
| 0 | KV_OCCUPANCY | kv_cache_utilization | 직접 (0.0~1.0) |
| 1 | IS_GPU | active_device | contains("opencl") -> 1.0, else 0.0 |
| 2 | TOKEN_PROGRESS | kv_cache_tokens | / 2048.0, min 1.0 |
| 3 | IS_PREFILL | (미사용) | 0.0 |
| 4 | KV_DTYPE_NORM | (미사용) | 0.0 |
| 5 | TBT_RATIO | actual_throughput | / 100.0, clamp [0, 1] |
| 6 | TOKENS_GENERATED_NORM | tokens_generated | / 2048.0, min 1.0 |
| 7 | ACTIVE_SWITCH_HW | (미사용) | 0.0 |
| 8 | ACTIVE_THROTTLE | (미사용) | 0.0 |
| 9 | ACTIVE_KV_OFFLOAD | (미사용) | 0.0 |
| 10 | ACTIVE_EVICTION | eviction_policy | != "none" AND != "" -> 1.0, else 0.0 |
| 11 | ACTIVE_LAYER_SKIP | skip_ratio | > 0.0 -> 1.0, else 0.0 |
| 12 | ACTIVE_KV_QUANT | (미사용) | 0.0 |

> **참고 (non-normative)**: `21-manager-state.md` MGR-080 참조. Index 3, 4, 7, 8, 9, 12는 향후 확장 예약.

#### MGR-ALG-051: FeatureVector Update Timing

EngineMessage가 Heartbeat일 때만 갱신한다. *(MUST)*

```
PRE:  msg is EngineMessage
POST: engine_state, available_actions, active_actions_reported 갱신

function update_engine_state(msg):
    if msg is not Heartbeat:
        return
    status = msg.status

    v = engine_state.values
    v[0]  = status.kv_cache_utilization
    v[1]  = if "opencl" in status.active_device then 1.0 else 0.0
    v[2]  = min(status.kv_cache_tokens / 2048.0, 1.0)
    v[5]  = clamp(status.actual_throughput / 100.0, 0, 1)
    v[6]  = min(status.tokens_generated / 2048.0, 1.0)
    v[10] = if status.eviction_policy != "" AND != "none" then 1.0 else 0.0
    v[11] = if status.skip_ratio > 0.0 then 1.0 else 0.0

    available_actions = parse_action_ids(status.available_actions)
    active_actions_reported = parse_action_ids(status.active_actions)
```

#### MGR-ALG-052: FeatureVector Initial Value

기본값: 13차원 zero vector. Engine 연결 전 또는 Heartbeat 미수신 상태의 가정: GPU 미사용, KV 비어있음, 모든 액션 비활성. *(MUST)*

#### MGR-ALG-053: ActionId Parsing

```
function parse_action_ids(strings: list of string) -> list of ActionId:
    return strings.filter_map(s -> ActionId.from_str(s))
    // 인식 불가 문자열은 skip (비치명적)
```

---

### 3.6 Observation Relief Calculation [MGR-ALG-060 ~ MGR-ALG-064]

#### MGR-ALG-060: ObservationContext

액션 효과 관측을 위한 컨텍스트 구조이다. *(MUST)*

```
struct ObservationContext:
    pressure_before: PressureVector    // 액션 적용 직전 압력
    feature_vec: FeatureVector         // 액션 적용 시점 시스템 상태
    applied_actions: list of ActionId  // 적용된 액션 목록
    applied_at: Instant                // 적용 시각
```

#### MGR-ALG-061: Observation Relief Calculation

OBSERVATION_DELAY (3.0 s) 경과 후 실측 relief를 계산하여 estimator를 갱신한다. *(MUST)*

```
OBSERVATION_DELAY = 3.0 s

function update_observation():
    if pending_observation is None:
        return
    ctx = pending_observation

    if elapsed(ctx.applied_at) <= OBSERVATION_DELAY:
        return     // 대기 계속

    // 실측 relief = 적용 전 pressure - 현재 pressure
    actual_relief = ctx.pressure_before - current_pressure
    // ReliefVector = {compute: before.compute - now.compute,
    //                 memory:  before.memory  - now.memory,
    //                 thermal: before.thermal - now.thermal,
    //                 latency: 0.0}

    // 적용된 각 액션에 대해 동일 relief로 observe
    for action in ctx.applied_actions:
        estimator.observe(action, ctx.feature_vec, actual_relief)

    pending_observation = None
```

```
INV-050: 관찰 relief의 latency 차원은 항상 0.0이다.
         (PressureVector 차감에 latency 없음)
INV-051: 여러 액션이 동시에 적용된 경우, 전체 relief가 각 액션에 귀속된다.
         (한계: 개별 기여 분리 불가)
```

#### MGR-ALG-062: pending_observation Lifecycle

| Event | Transition |
|-------|-----------|
| Directive 발행 | None -> Some(ObservationContext) |
| 3.0 s 경과 | Some -> None (observe 호출 후) |
| 3.0 s 미경과 | Some 유지 |
| 새 Directive (관찰 중) | 이전 관찰 **폐기**, 새 ObservationContext로 **교체** |

> **참고 (non-normative)**: `21-manager-state.md` MGR-077과 동일하다.

#### MGR-ALG-063: State Update on Directive Issuance

Directive 발행 시 관찰 컨텍스트를 기록하고 last_acted_pressure를 갱신한다. *(MUST)*

```
function on_directive_issued(commands, current_pressure, engine_state):
    pending_observation = ObservationContext {
        pressure_before: current_pressure,
        feature_vec: engine_state.clone(),
        applied_actions: commands.map(c -> c.action),
        applied_at: now()
    }
    last_acted_pressure = current_pressure
```

#### MGR-ALG-064: De-escalation Directive Generation

동일 사이클에서 needs_action 경로에서 이미 Directive가 생성된 경우, 디에스컬레이션 Directive는 발행하지 않는다. *(MUST)*

```
function process_de_escalation(prev_mode, current_mode, existing_result):
    if prev_mode <= current_mode:
        return existing_result
    if existing_result is Some:
        return existing_result     // needs_action 경로 우선

    match (prev_mode, current_mode):
        (Critical, Warning):
            return Some(Directive{RestoreDefaults})
        (_, Normal):
            return Some(Directive{RestoreDefaults})
        _:
            return None
```

---

### 3.7 Process Signal Pipeline [MGR-ALG-070 ~ MGR-ALG-072]

#### MGR-ALG-070: process_signal Full Pipeline

SystemSignal을 처리하여 EngineDirective를 생성하는 전체 파이프라인이다. *(MUST)*

```
PRE:  signal is SystemSignal
POST: Option<EngineDirective>

function process_signal(signal) -> Option<EngineDirective>:
    // Step 1: PI Controller 갱신 -> pressure 갱신
    update_pressure(signal)

    // Step 2: Supervisory -> 운영 모드 결정
    mode = supervisory.evaluate(pressure)

    // Step 3: 관찰 갱신 (3초 경과 시 실측 relief -> estimator.observe)
    update_observation()

    // Step 4: needs_action 판정
    needs_action = evaluate_needs_action(mode, prev_mode,
                                         pressure, last_acted_pressure)

    // Step 5: 액션 선택 및 Directive 생성
    result = None
    if needs_action:
        qcf_values = {id: registry.default_cost(id)
                      for id in registry.lossy_actions()}
        commands = ActionSelector.select(
            registry, estimator, pressure, mode, engine_state,
            qcf_values, latency_budget, active_actions_reported,
            available_actions
        )
        if commands is not empty:
            engine_commands = convert_to_engine_commands(commands)
            if engine_commands is not empty:
                directive = EngineDirective{
                    seq_id: next_seq_id(),
                    commands: engine_commands
                }
                on_directive_issued(commands, pressure, engine_state)
                result = Some(directive)

    // Step 6: 디에스컬레이션 처리
    result = process_de_escalation(prev_mode, mode, result)

    // Step 7: prev_mode 갱신
    prev_mode = mode
    return result
```

> **참고 (non-normative)**: `21-manager-state.md` MGR-075의 6단계와 동일하다 (Step 7은 상태 갱신).

#### MGR-ALG-071: QCF Proxy (Current Implementation)

Engine이 QCF 실시간 보고를 구현하기 전의 대체 메커니즘이다. *(MUST)*

```
function get_qcf_values(registry) -> map of ActionId to float:
    result = {}
    for id in registry.lossy_actions():
        result[id] = registry.default_cost(id)   // ActionConfig.default_cost (기본 1.0)
    return result
```

> **참고 (non-normative)**: QCF 실시간 보고가 Engine에 구현되면, 이 proxy는 Engine이 보고한 QCF 값으로 대체될 예정이다.

#### MGR-ALG-072: Full Pipeline Example Trace

```
초기 상태: mode=Normal, all pressures=0, pending_observation=None
Memory: 임계값 직접 매핑 (PI 미사용, 100ms 폴링)

t=0.0s: signal=MemoryPressure(available=1600MB, total=2GB)
  Step 1: m = 1 - 1600/2000 = 0.20
          pressure.memory = 0.20 (직접 매핑)
  Step 2: peak = 0.20 -> Normal (< 0.4)
  Step 4: mode=Normal -> needs_action = false
  Result: None

t=0.1s: signal=MemoryPressure(available=800MB, total=2GB) [외부 앱 대량 할당]
  Step 1: m = 1 - 800/2000 = 0.60
          pressure.memory = 0.60 (즉시 반영, PI 지연 없음)
  Step 2: peak = 0.60 -> Warning (>= 0.4, 즉시 에스컬레이션)
  Step 4: mode_changed (Normal -> Warning) -> needs_action = true
  Step 5: ActionSelector -> [lossless 후보만]
          -> Directive 생성, pending_observation = Some
  Result: Some(Directive)
  ⏱ 감지→Directive: ~150ms (폴링 100ms + recv_timeout 50ms)

t=0.2s: signal=MemoryPressure(available=200MB, total=2GB) [추가 할당]
  Step 1: pressure.memory = 0.90
  Step 2: peak = 0.90 -> Critical (>= 0.7, 즉시 에스컬레이션)
  Step 4: mode_changed (Warning -> Critical) -> needs_action = true
  Step 5: ActionSelector -> [lossless + lossy 후보, QCF 비용 기반]
          -> Directive 생성 (e.g., KvEvictH2o + Throttle)
  Result: Some(Directive)

t=3.2s: signal=MemoryPressure (여전히 높음)
  Step 3: t3.2 - t0.2 = 3.0s >= OBSERVATION_DELAY
          actual_relief = pressure_before - pressure_current
          estimator.observe(action, feature_vec, actual_relief)
          pending_observation = None
```

### 3.8 LuaPolicy DPP Algorithm [MGR-DPP-010 ~ MGR-DPP-070]

Production 정책 엔진인 `policy_default.lua` v2.1.0의 핵심 결정 알고리즘이다. Lyapunov Drift-Plus-Penalty (DPP) 프레임워크를 사용하여 다중 도메인 압력 해소와 latency 비용 간 최적 트레이드오프를 선형 스칼라화로 해결한다.

> **HierarchicalPolicy와의 관계**: 3.1~3.7의 HierarchicalPolicy (PI + Supervisory + ActionSelector + RLS)는 `#[cfg(feature = "hierarchical")]` 경로에서만 활성화된다. LuaPolicy DPP는 기본 정책 경로이며, relief 학습에 EwmaReliefTable (MGR-ALG-080~083, INV-086~090)을 사용한다. 두 정책은 `Policy` 트레이트를 통해 교체 가능하다.

#### MGR-DPP-010: DPP Objective Function

DPP는 모든 candidate action에 대해 아래 목적 함수를 평가하고, 최대값을 달성하는 action을 선택한다. *(MUST)*

```
PRE:  Z_k >= 0 for all k, V >= 0, r_k(a) is predicted relief for domain k
POST: a* is the action maximizing the objective among A_safe

a* = argmax_{a ∈ A_safe} [ Σ_k Z_k · r_k(a)
                            - V · max(-lat(a), 0)
                            + UCB · ucb_bonus(a) ]
```

- `Z_k`: 도메인 k의 virtual queue 값 (압력의 가중 초과량). 높을수록 해당 도메인 해소에 높은 가치 부여.
- `r_k(a)`: action a의 도메인 k에 대한 예측 relief. EwmaReliefTable.predict(a)에서 획득.
- `lat(a)`: action a의 예측 latency 영향. 음수이면 latency 악화.
- `V`: latency penalty weight. latency 보호와 압력 해소 간 트레이드오프 조절.
- `UCB`: UCB bonus weight (기본 1.0). LinUCB confidence width를 score에 additive 반영.
- `ucb_bonus(a)`: LinUCB가 산출하는 exploration bonus. `σ(a) · linucb_alpha`. LinUCB 비활성 시 0. 상세는 §3.9 참조.
- `A_safe`: safe set. latency floor 제약을 만족하는 action 집합 (§3.8 MGR-DPP-020).

**INV-093**: DPP objective는 Z_k=0인 도메인의 relief를 무시한다. 압력이 warn 이하인 도메인은 결정에 영향을 미치지 않는다.

#### MGR-DPP-011: Multi-Threshold Excess Virtual Queue

도메인 k의 virtual queue Z_k는 3단계 임계값 초과량의 가중합이다. *(MUST)*

```
PRE:  p_k in [0, 1], θ_warn < θ_crit < θ_emerg, all θ in (0, 1)
POST: Z_k >= 0

function compute_virtual_queue(p_k, thresholds) -> Z_k:
    Z_k = W_WARN  * max(0, p_k - θ_warn)
        + W_CRIT  * max(0, p_k - θ_crit)
        + W_EMERG * max(0, p_k - θ_emerg)
    return Z_k
```

상수:

| 가중치 | 값 | 의미 |
|--------|-----|------|
| W_WARN | 1.0 | warning 초과분 기본 가중 |
| W_CRIT | 2.0 | critical 초과분 2배 가중 |
| W_EMERG | 4.0 | emergency 초과분 4배 가중 |

**INV-094**: Z_k는 항상 비음수이다. p_k <= θ_warn이면 Z_k = 0이다.

**INV-095**: 가중치 관계는 W_WARN < W_CRIT < W_EMERG를 유지한다. emergency 도메인은 항상 warning/critical보다 큰 Z_k를 가진다.

#### MGR-DPP-012: Domain Thresholds

도메인별 3단계 임계값 정의이다. *(MUST)*

| 도메인 (k) | θ_warn | θ_crit | θ_emerg | 비고 |
|------------|--------|--------|---------|------|
| mem | 0.60 | 0.80 | 0.90 | 메모리: 낮은 warn으로 조기 대응 |
| cpu | 0.70 | 0.85 | 0.95 | Compute |
| gpu | 0.70 | 0.85 | 0.95 | Compute |
| therm | 0.70 | 0.85 | 0.95 | Thermal |

**INV-096**: 모든 도메인에서 θ_warn < θ_crit < θ_emerg가 성립한다.

#### MGR-DPP-013: DPP Constants

DPP 프레임워크의 전역 상수이다. *(MUST)*

| 상수 | 값 | 의미 |
|------|-----|------|
| V | 1.0 | latency penalty weight. 클수록 latency 보호 우선. |
| C | 0.30 | normal latency floor. lat(a) >= -C 를 만족해야 A_safe에 포함. |
| C_EMERGENCY | 0.50 | emergency latency floor. emergency 도메인 존재 시 완화된 floor. |
| UCB | 1.0 | LinUCB exploration bonus weight. ucb_bonus(a) 항의 가중치. |

#### MGR-DPP-020: Safe Set Construction

Candidate action의 safe set A_safe를 구성한다. latency floor 제약을 만족하는 action만 포함한다. *(MUST)*

```
PRE:  candidates is set of all candidate actions
      has_emergency = any domain k where p_k >= θ_emerg
POST: A_safe ⊆ candidates, all a in A_safe satisfy latency floor

function build_safe_set(candidates, has_emergency) -> A_safe:
    floor = if has_emergency then -C_EMERGENCY else -C
    A_safe = { a ∈ candidates | lat(a) >= floor }
    return A_safe
```

- `lat(a)`: EwmaReliefTable.predict(a)의 latency 차원 (index 4). 음수 = latency 악화.
- Emergency 시 floor을 완화하여 더 공격적인 action 허용 (-0.50 vs -0.30).

**INV-097**: A_safe가 비어있으면 no-op을 반환한다. latency floor을 위반하는 action은 절대 선택되지 않는다.

#### MGR-DPP-030: Joint Action Registry

2개 이하의 single action을 조합한 joint action을 정의한다. *(MUST)*

```
joint_actions = {
    "throttle_plus_layer_skip":  ["throttle", "layer_skip"]
}
```

> **v2.1.0 변경**: `kv_evict_plus_quant` (`kv_evict_sliding` + `kv_quant_dynamic`) 제거. 두 action은 `kv_quality` exclusion group에 속하므로 동시 발행이 금지된다 (INV-041). Joint action으로 등록하면 exclusion 위반이 되어 제거하였다.

**ctx.is_joint_valid 검증**: `decide()` 진입 시, 모든 joint action의 component를 exclusion_groups로 검증한다. *(MUST)*

```
PRE:  joint_actions registry, exclusion_groups registry
POST: all joint actions are valid, or error raised

function validate_joint_actions(joint_actions, exclusion_groups):
    for name, components in joint_actions:
        if not ctx.is_joint_valid(components):
            error("joint action '" .. name .. "' violates exclusion groups")
```

Joint action의 relief 추정:

```
PRE:  joint action J = {a1, a2}
POST: predicted relief vector for J

function predict_joint_relief(J) -> relief:
    if ewma_table.has_entry(J.name):
        return ewma_table.predict(J.name)    // 학습된 joint relief
    else:
        // fallback: component 선형 합
        return ewma_table.predict(a1) + ewma_table.predict(a2)
```

**INV-098**: Joint action은 최대 2개의 component action으로 구성된다.

**INV-099**: Joint relief fallback은 component relief의 선형 합이다. 학습된 joint relief가 존재하면 이를 우선 사용한다.

#### MGR-DPP-040: Safety Override

DPP 결정 외부에서 적용되는 강제 규칙이다. DPP가 선택한 action과 무관하게 적용된다. *(MUST)*

```
PRE:  dpp_result is the action selected by DPP objective
      domain_levels is map of domain -> pressure_level
POST: final_action with safety overrides applied

function apply_safety_override(dpp_result, domain_levels) -> final_action:
    final_action = dpp_result

    // Compute 또는 Thermal이 emergency인 경우에만 throttle 강제 추가
    if domain_levels["cpu"] == EMERGENCY
       or domain_levels["gpu"] == EMERGENCY
       or domain_levels["therm"] == EMERGENCY:
        if "throttle" not in final_action:
            final_action = final_action + "throttle"

    // 주의: Memory emergency는 throttle을 강제하지 않음
    // (throttle은 compute 액션이므로 memory 해소에 무효)

    return final_action
```

**INV-100**: Safety Override는 compute/thermal emergency에서만 throttle을 추가한다. Memory emergency는 Safety Override 대상이 아니다.

**INV-101**: Safety Override는 DPP 결과에 throttle을 **추가**만 한다. DPP가 선택한 action을 제거하지 않는다.

#### MGR-DPP-050: Pressure Level to Parameter Mapping

Pressure level에 따라 action의 세부 파라미터를 결정한다. *(MUST)*

| Pressure Level | keep_ratio | target_bits | 적용 시나리오 |
|---------------|------------|-------------|-------------|
| emergency | 0.25 | 2 | 최대 공격: KV 75% evict, 2-bit quant |
| critical | 0.50 | 4 | 강한 해소: KV 50% evict, 4-bit quant |
| warning | 0.70 | 8 | 완만한 해소: KV 30% evict, 8-bit quant |
| normal | 0.85 | 16 | 최소 개입 (거의 무손실) |

Pressure level 판정:

```
function pressure_level(p_k, thresholds) -> level:
    if p_k >= θ_emerg: return EMERGENCY
    if p_k >= θ_crit:  return CRITICAL
    if p_k >= θ_warn:  return WARNING
    return NORMAL
```

#### MGR-DPP-060: Decision Pipeline (Full 6-Step)

LuaPolicy의 `decide()` 호출 시 실행되는 전체 결정 파이프라인이다. *(MUST)*

```
PRE:  pressures is map of domain -> pressure value [0,1]
      triggers is set of active TriggerEngine triggers
      ewma_table is EwmaReliefTable instance
      active_actions is set of currently active actions
POST: list of EngineCommand (possibly empty)

function decide(pressures, triggers, ewma_table, active_actions) -> commands:

    // ── Step 1: Trigger 검사 ──
    if triggers is empty:
        if active_actions is not empty:
            return restore_commands(active_actions)
        return []

    // ── Step 2: Virtual Queue 계산 + Emergency 판별 ──
    has_emergency = false
    Z = {}
    for k in {mem, cpu, gpu, therm}:
        Z[k] = compute_virtual_queue(pressures[k], thresholds[k])
        if pressures[k] >= thresholds[k].emerg:
            has_emergency = true

    // ── Step 3: 전체 Z_k = 0 검사 ──
    if all Z[k] == 0:
        if active_actions is not empty:
            return restore_commands(active_actions)
        return []

    // ── Step 4: Candidate 열거 (single + joint) ──
    candidates = enumerate_single_actions()
                 ∪ enumerate_joint_actions()

    // ── Step 5: DPP Score 계산 + argmax ──
    A_safe = build_safe_set(candidates, has_emergency)
    if A_safe is empty:
        return []

    best = None, best_score = -∞
    for a in A_safe:
        relief = predict_relief(a)    // single or joint
        resource_term = Σ_k Z[k] * relief[k]
        latency_penalty = V * max(-relief.latency, 0)
        ucb = UCB * (relief.ucb_bonus or 0)   // LinUCB bonus (§3.9)
        score = resource_term - latency_penalty + ucb
        if score > best_score:
            best = a, best_score = score

    // ── Step 6: Safety Override ──
    domain_levels = { k: pressure_level(pressures[k], thresholds[k])
                      for k in {mem, cpu, gpu, therm} }
    final = apply_safety_override(best, domain_levels)

    // 파라미터 결정: 최고 압력 도메인의 level 사용
    peak_level = max(domain_levels.values())
    params = parameter_mapping(peak_level)

    return build_commands(final, params)
```

**INV-102**: Step 1에서 trigger가 없으면 Step 2~6을 실행하지 않는다. Trigger가 없는 상태에서 새 action을 발행하지 않는다.

**INV-103**: Step 5에서 score 비교는 strict greater-than이다. 동점 시 먼저 열거된 candidate가 선택된다.

#### MGR-DPP-061: TriggerEngine Hysteresis

TriggerEngine은 hysteresis 기반 3개 trigger를 관리한다. *(MUST)*

```
triggers = {
    tbt_degraded: { enter: 0.30, exit: 0.10 },   // TBT 열화율
    mem_low:      { enter: 0.80, exit: 0.60 },   // 메모리 사용률
    temp_high:    { enter: 0.70, exit: 0.50 }    // 온도 사용률
}
```

```
PRE:  value in [0, 1], trigger has state {active: bool, enter: float, exit: float}
POST: trigger.active updated

function update_trigger(trigger, value):
    if trigger.active:
        if value < trigger.exit:
            trigger.active = false
    else:
        if value >= trigger.enter:
            trigger.active = true
```

- `tbt_degraded`의 value: `(current_tbt - baseline_tbt) / baseline_tbt`. TBT baseline은 EWMA (alpha=0.875, warmup=20 tokens).

**INV-104**: Hysteresis 간격(enter - exit)은 양수이다. enter > exit가 항상 성립한다.

**INV-105**: TBT baseline EWMA warmup 기간(20 tokens) 동안 `tbt_degraded` trigger는 활성화되지 않는다.

#### MGR-DPP-062: EwmaReliefTable Integration

DPP는 EwmaReliefTable을 통해 relief를 예측하고 학습한다. 상세 수식은 MGR-ALG-080~083 및 INV-086~090을 참조한다. *(MUST)*

```
Relief 벡터: 6D = [gpu, cpu, memory, thermal, latency, main_app_qos]
EWMA alpha: 0.875
Observation delay: 3초
Cold start: default prior 사용 (observation_count == 0 → 직접 대입, INV-087)
```

DPP objective에서 사용하는 도메인 매핑:

| DPP 도메인 k | Relief 벡터 index | 용도 |
|-------------|------------------|------|
| gpu | 0 | Z_gpu · r_gpu(a) |
| cpu | 1 | Z_cpu · r_cpu(a) |
| mem | 2 | Z_mem · r_mem(a) |
| therm | 3 | Z_therm · r_therm(a) |
| (latency) | 4 | V · max(-lat(a), 0) penalty |
| (main_app_qos) | 5 | (현재 DPP objective에 미사용) |

#### MGR-DPP-070: Example Trace — Memory Critical + Thermal Warning

```
상태: mem=0.82, cpu=0.40, gpu=0.50, therm=0.72
      triggers = {mem_low: active, temp_high: active}
      active_actions = {}

Step 1: triggers not empty → 진행

Step 2: Virtual Queue 계산
  Z_mem   = 1.0 * max(0, 0.82 - 0.60)   // 0.22
          + 2.0 * max(0, 0.82 - 0.80)   // 0.04
          + 4.0 * max(0, 0.82 - 0.90)   // 0.00
          = 0.26
  Z_cpu   = 1.0 * max(0, 0.40 - 0.70)   = 0.00
  Z_gpu   = 1.0 * max(0, 0.50 - 0.70)   = 0.00
  Z_therm = 1.0 * max(0, 0.72 - 0.70)   // 0.02
          + 2.0 * max(0, 0.72 - 0.85)   // 0.00
          + 4.0 * max(0, 0.72 - 0.95)   // 0.00
          = 0.02
  has_emergency = false (모든 도메인 < θ_emerg)

Step 3: Z_mem=0.26, Z_therm=0.02 → 진행 (not all zero)

Step 4: Candidates (예시 relief from EwmaReliefTable, ucb_bonus from LinUCB §3.9)
  kv_evict_sliding:  relief = [0.0, 0.0, 0.35, 0.0, -0.15, 0.0], ucb_bonus = 0.05
  kv_quant_dynamic:  relief = [0.0, 0.0, 0.20, 0.0, -0.10, 0.0], ucb_bonus = 0.02
  throttle:          relief = [0.15, 0.10, 0.0, 0.25, -0.20, 0.0], ucb_bonus = 0.08

  > v2.1.0: kv_evict_plus_quant 제거 (kv_quality exclusion group 위반)

Step 5: DPP Score (V=1.0, UCB=1.0)
  kv_evict_sliding:
    resource_term = 0.26*0.35 + 0.02*0.0 = 0.091
    latency_penalty = 1.0 * max(0.15, 0) = 0.15
    ucb = 1.0 * 0.05 = 0.05
    score = 0.091 - 0.15 + 0.05 = -0.009

  kv_quant_dynamic:
    resource_term = 0.26*0.20 = 0.052
    latency_penalty = 1.0 * max(0.10, 0) = 0.10
    ucb = 1.0 * 0.02 = 0.02
    score = 0.052 - 0.10 + 0.02 = -0.028

  throttle:
    resource_term = 0.00*0.15 + 0.00*0.10 + 0.26*0.0 + 0.02*0.25 = 0.005
    latency_penalty = 1.0 * max(0.20, 0) = 0.20
    ucb = 1.0 * 0.08 = 0.08
    score = 0.005 - 0.20 + 0.08 = -0.115

  Safe set (floor = -C = -0.30):
    모든 candidate의 lat(a) >= -0.30 → 전원 포함

  Best: kv_evict_sliding (score = -0.009)

Step 6: Safety Override
  domain_levels = {mem: CRITICAL, cpu: NORMAL, gpu: NORMAL, therm: WARNING}
  compute/thermal emergency 없음 → override 없음

Result: [KvEvictSliding { keep_ratio: 0.50 }]
  (peak_level = CRITICAL → keep_ratio=0.50)
```

> **해석**: v2.1.0에서 UCB bonus가 추가되어, LinUCB exploration이 score에 반영된다. kv_evict_sliding은 관측이 적어 ucb_bonus가 높고 (0.05), 이것이 latency penalty를 상쇄하여 kv_quant_dynamic보다 높은 score를 달성했다. LinUCB 비활성 시 (ucb_bonus=0) kv_quant_dynamic이 선택되어 기존 동작과 동일하다.

### 3.9 LinUCB Exploration Bonus [MGR-LUCB-010 ~ MGR-LUCB-070]

DPP (§3.8)의 score에 LinUCB (Linear Upper Confidence Bound) exploration bonus를 additive로 통합한다. LinUCB는 DPP를 대체하지 않으며, EwmaReliefTable의 점추정(mean)을 유지하면서 uncertainty(σ)를 추가하여 exploration을 개선한다. Safe set은 기존 DPP (§3.8 MGR-DPP-020)를 그대로 사용한다.

> **EwmaReliefTable과의 관계**: Mean 추정은 EWMA가 담당하고 (빠른 수렴, 단순), uncertainty 추정은 LinUCB P matrix가 담당한다 (정확한 confidence width). `ucb_bonus = σ · linucb_alpha`가 DPP score에 additive로 더해진다.

#### MGR-LUCB-010: LinUCB + DPP Additive Bonus

DPP objective에 LinUCB exploration bonus를 additive 항으로 추가한다. *(MUST)*

```
PRE:  Z_k >= 0 for all k, V >= 0
      r̂_k(a): EwmaReliefTable 점추정 (mean)
      ucb_bonus(a): LinUCB confidence width σ(a) · linucb_alpha
POST: a* is the action maximizing the objective among A_safe

score(a) = Σ_k Z_k · r̂_k(a)
           - V · max(-lât(a), 0)
           + UCB · ucb_bonus(a)

a* = argmax_{a ∈ A_safe} score(a)
```

- `ucb_bonus(a)`: `σ(a, φ) · linucb_alpha`. σ가 크면 (관측 적음) bonus가 크다 → exploration 유도.
- `UCB`: DPP 상수 (기본 1.0). §3.8 MGR-DPP-013 참조.
- `A_safe`: 기존 DPP safe set (§3.8 MGR-DPP-020). Pessimistic safe set은 사용하지 않는다.
- DPP 상수 (V, C, C_EMERGENCY, W_*, THETA)는 §3.8과 동일하게 적용한다.

**INV-106**: `linucb_alpha=0`이면 `ucb_bonus(a)=0`이 되어, objective는 §3.8 MGR-DPP-010의 기존 DPP objective와 수학적으로 동일하다.

#### MGR-LUCB-011: LinUcbTable 인터페이스

Per-action P matrix (= V^{-1}) 관리 구조체이다. *(MUST)*

```
PRE:  action name, feature vector φ ∈ R^D (D = LINUCB_FEATURE_DIM = 13)
POST: scalar ucb_bonus

struct LinUcbTable:
    // Per-action D×D P matrix (= V⁻¹). action 미등록 시 None.
    tables: HashMap<String, [[f32; D]; D]>

    ucb_bonus(action: str, φ: R^D) -> f32
    update(action: str, φ: R^D)
```

- `ucb_bonus`: σ = sqrt(max(0, φᵀ · P · φ)). P가 없으면 (cold start) 1.0을 반환하여 최대 탐색을 유도한다.
- `update`: Sherman-Morrison으로 P matrix를 갱신한다 (§3.9 MGR-LUCB-013 참조).
- b, theta 벡터는 관리하지 않는다. Mean 추정은 EWMA가 전담한다.

**INV-107**: `ucb_bonus`는 읽기 전용이다. P matrix를 변경하지 않는다. (MGR-C08 정신 계승)

**INV-108**: σ >= 0. `max(0, φᵀ·P·φ)` + `sqrt`로 비음수가 보장된다.

#### MGR-LUCB-012: Feature Vector 정의 (13D)

LinUCB에 입력되는 context feature vector이다. HierarchicalPolicy FeatureVector와 동일한 레이아웃이다. *(MUST)*

```
PRE:  SignalState, Heartbeat, active_actions available
POST: φ ∈ R^D, D = 13, all elements in [0, 1]

LINUCB_FEATURE_DIM = 13

φ = [
    0:  KV_OCCUPANCY,       // Heartbeat.kv_cache_utilization      [0,1]
    1:  IS_GPU,             // 1.0 if GPU backend, 0.0 otherwise   {0,1}
    2:  TOKEN_PROGRESS,     // min(tokens_generated / max_tokens, 1.0) [0,1]
    3:  IS_PREFILL,         // 1.0 if prefill phase, 0.0 otherwise {0,1}
    4:  KV_DTYPE_NORM,      // kv_dtype encoding normalized        [0,1]
    5:  TBT_RATIO,          // current_tbt / baseline_tbt          [0,1] (clamped)
    6:  TOKENS_GEN_NORM,    // min(tokens_generated / 1000, 1.0)   [0,1]
    7:  ACTIVE_SWITCH_HW,   // 1.0 if switch_hw active             {0,1}
    8:  ACTIVE_THROTTLE,    // 1.0 if throttle active               {0,1}
    9:  ACTIVE_KV_OFFLOAD,  // 1.0 if kv_offload active             {0,1}
    10: ACTIVE_EVICTION,    // 1.0 if kv_evict_* active             {0,1}
    11: ACTIVE_LAYER_SKIP,  // 1.0 if layer_skip active             {0,1}
    12: ACTIVE_KV_QUANT,    // 1.0 if kv_quant_dynamic active       {0,1}
]
```

- Feature dimension D = 13. Context 7D (indices 0~6) + active action indicators 6D (indices 7~12).
- 모든 feature는 [0, 1] 범위로 정규화된다.
- Active action indicator는 현재 실행 중인 action의 존재 여부를 이진 인코딩한다.

**INV-109**: Feature vector의 모든 원소는 [0, 1] 범위이다.

#### MGR-LUCB-013: P Matrix 관리 (Sherman-Morrison)

Per-action P matrix (= V^{-1})의 갱신 규칙이다. b, theta는 관리하지 않는다. *(MUST)*

```
초기화 (action 첫 등록 시):
    P = (1/λ) · I_D       // D × D, λ = 1.0 → P = I_D
                           // V = λI → P = V⁻¹ = (1/λ)I

갱신 (update 호출 시):
    PRE:  φ ∈ R^D, P is current inverse
    POST: P updated via Sherman-Morrison

    // Sherman-Morrison: (V + φφᵀ)⁻¹ = V⁻¹ - (V⁻¹φ)(φᵀV⁻¹) / (1 + φᵀV⁻¹φ)
    Pφ = P · φ
    denom = 1.0 + φᵀ · Pφ
    P ← P - (Pφ · Pφᵀ) / denom
```

- λ (ridge regularization): 1.0 고정. forgetting factor 없음.
- V를 직접 저장하지 않고, P = V^{-1}만 유지하여 O(D^2) 갱신 (D=13이므로 169 flops).
- P는 항상 positive semi-definite이다 (Sherman-Morrison은 positive definiteness를 보존한다).

**INV-110**: P는 항상 positive semi-definite이다. Sherman-Morrison update는 positive definiteness를 보존한다.

#### MGR-LUCB-014: Confidence Width σ 계산

Action의 uncertainty를 나타내는 스칼라 σ이다. *(MUST)*

```
PRE:  φ ∈ R^D, P = V⁻¹ (positive semi-definite)
POST: σ >= 0

σ(a, φ) = sqrt(max(0, φᵀ · P_a · φ))
```

- P가 존재하지 않으면 (action 미등록, cold start): σ = 1.0 (최대 탐색).
- 관측이 증가하면 P가 shrink하여 σ가 감소한다 (uncertainty 감소).
- `max(0, ...)`: 부동소수점 오차로 인한 미세 음수를 방지한다.

#### MGR-LUCB-015: linucb_alpha 가중치

UCB bonus의 고정 가중치이다. β_t 스케줄은 사용하지 않는다. *(MUST)*

```
ucb_bonus(a) = σ(a, φ) · linucb_alpha
```

| 설정 | 기본값 | 위치 | 의미 |
|------|--------|------|------|
| linucb_alpha | 0.5 | AdaptationConfig | exploration 강도. σ에 곱해지는 고정 가중치. |

- β_t 시간 감쇠 스케줄은 없다. P matrix의 자연 shrink로 σ가 감소하여, 동일 φ에 대해 ucb_bonus가 단조 감소한다.
- `linucb_alpha=0`이면 ucb_bonus=0이 되어 기존 DPP와 동일하게 동작한다 (§3.9 MGR-LUCB-070).

**INV-111**: `linucb_alpha >= 0`. AdaptationConfig에서 검증한다.

**INV-112**: 동일 φ에 대해, P matrix update 후 σ는 단조 감소한다. Sherman-Morrison은 P를 monotonically shrink하므로 `φᵀPφ`는 비증가이다.

#### MGR-LUCB-020: Safe Set — 기존 DPP 유지 [REMOVED]

~~Pessimistic safe set은 구현하지 않는다.~~ 코드는 기존 DPP safe set (§3.8 MGR-DPP-020)을 그대로 사용한다. LinUCB는 safe set을 변경하지 않고 score의 additive bonus로만 영향을 미친다.

> **v2.1.0 설계 결정**: Pessimistic safe set (A_safe_ucb ⊆ A_safe)은 초기 설계에 있었으나, cold start 시 σ가 과도하게 커서 대부분의 action이 safe set에서 배제되는 문제가 발생하였다. 대신 기존 DPP safe set을 유지하고, exploration은 additive bonus로 유도한다.

#### MGR-LUCB-021: [REMOVED]

~~Optimistic reward selection은 별도 단계로 존재하지 않는다.~~ DPP score에 `+ UCB · ucb_bonus(a)` 항으로 통합되었다 (§3.9 MGR-LUCB-010).

#### MGR-LUCB-030: EwmaReliefTable과의 하이브리드 관계

EWMA와 LinUCB의 역할 분담이다. *(MUST)*

```
DPP score 계산 시:
    mean[k] = EwmaReliefTable.predict(a)[k]      // EWMA 점추정
    ucb     = UCB * σ(a, φ) * linucb_alpha        // LinUCB exploration bonus

관측 시:
    EwmaReliefTable.observe(a, actual_relief)     // EWMA 갱신
    LinUcbTable.update(a, φ)                      // LinUCB P matrix 갱신 (동시)
```

- Mean은 EWMA가 제공한다. LinUcbTable은 mean을 계산하지 않는다 (P matrix만 관리).
- σ는 LinUCB P matrix가 제공한다. EWMA는 uncertainty를 제공할 수 없다.
- 두 테이블은 독립적으로 갱신되며, 동일 시점에 호출된다.

**INV-115**: EWMA observe와 LinUCB update는 동일 시점에 호출된다. 두 테이블 간 학습 데이터 불일치는 없다.

#### MGR-LUCB-040: Lua 인터페이스

LinUCB 추정 결과를 Lua 정책에 노출하는 인터페이스이다. *(MUST)*

```
build_ctx() 시:
    ctx.coef.relief[action_name].ucb_bonus = σ(a, φ) · linucb_alpha
```

- `ucb_bonus` 필드는 per-action relief 테이블에 직접 포함된다. 별도 `relief_ub` 테이블은 없다.
- Lua DPP score 계산에서 `+ DPP.UCB * (r.ucb_bonus or 0)`으로 참조한다.
- `ucb_bonus`가 nil이면 0으로 처리 (LinUCB 미활성 시 자연 fallback).

#### MGR-LUCB-050: Phase 로드맵

단계적 도입 계획이다. *(SHOULD)*

| Phase | 내용 | 상태 | 검증 방법 |
|-------|------|------|----------|
| Phase 1 | Rust LinUcbTable 구현 + Lua ctx.coef.relief[a].ucb_bonus 노출 + policy_default.lua v2.1.0 통합 | **완료** | 유닛 테스트: ucb_bonus 수렴, Sherman-Morrison 정합성 |
| Phase 2 | 시뮬레이터 AB 비교 | 미착수 | EWMA-only vs UCB 강화: cumulative relief, exploration rate |
| Phase 3 | 온디바이스 AB 테스트 | 미착수 | TBT 회귀 없음 확인 |

#### MGR-LUCB-060: Example Trace — LinUCB Additive Bonus

```
상태: mem=0.82, cpu=0.40, gpu=0.50, therm=0.72
      triggers = {mem_low: active, temp_high: active}
      active_actions = {}
      linucb_alpha = 0.5

Feature vector φ (13D):
  [0.65, 1.0, 0.30, 0.0, 0.5, 1.2, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  (kv_occ=0.65, is_gpu=1.0, token_prog=0.30, is_prefill=0, kv_dtype=0.5,
   tbt_ratio=1.2→clamped 1.0, tokens_gen=0.15, all active_actions=0)

Step 1~4: §3.8 MGR-DPP-060과 동일 (Z 계산, candidate 열거)
  Z_mem=0.26, Z_therm=0.02, Z_cpu=0.00, Z_gpu=0.00

Step 5: DPP Score + LinUCB Bonus (V=1.0, UCB=1.0)

  kv_evict_sliding (P matrix 5회 update 후):
    EWMA mean = [0.0, 0.0, 0.35, 0.0, -0.15, 0.0]
    σ = sqrt(φᵀ·P·φ) = 0.42
    ucb_bonus = 0.42 * 0.5 = 0.21
    resource_term = 0.26*0.35 + 0.02*0.0 = 0.091
    latency_penalty = 1.0 * max(0.15, 0) = 0.15
    ucb = 1.0 * 0.21 = 0.21
    score = 0.091 - 0.15 + 0.21 = 0.151

  kv_quant_dynamic (P matrix 20회 update 후):
    EWMA mean = [0.0, 0.0, 0.20, 0.0, -0.10, 0.0]
    σ = 0.18
    ucb_bonus = 0.18 * 0.5 = 0.09
    resource_term = 0.26*0.20 = 0.052
    latency_penalty = 1.0 * max(0.10, 0) = 0.10
    ucb = 1.0 * 0.09 = 0.09
    score = 0.052 - 0.10 + 0.09 = 0.042

  throttle (P matrix 미등록, cold start):
    EWMA mean = [0.15, 0.10, 0.0, 0.25, -0.20, 0.0]
    σ = 1.0 (cold start default)
    ucb_bonus = 1.0 * 0.5 = 0.50
    resource_term = 0.02*0.25 = 0.005
    latency_penalty = 1.0 * max(0.20, 0) = 0.20
    ucb = 1.0 * 0.50 = 0.50
    score = 0.005 - 0.20 + 0.50 = 0.305

  Safe set (§3.8 MGR-DPP-020, floor = -C = -0.30):
    모든 candidate의 lat(a) >= -0.30 → 전원 포함

  Best: throttle (score = 0.305)

Step 6: Safety Override
  domain_levels = {mem: CRITICAL, cpu: NORMAL, gpu: NORMAL, therm: WARNING}
  compute/thermal emergency 없음 → override 없음

Result: [Throttle]
```

> **해석**: throttle은 cold start (P 미등록)로 σ=1.0이 되어 ucb_bonus=0.50이 부여되었다. 이 exploration bonus가 latency penalty를 크게 상쇄하여 최고 score를 달성했다. 기존 DPP (ucb_bonus=0)에서는 throttle의 score가 -0.195로 최하위였으나, LinUCB bonus로 탐색이 유도되었다. throttle을 실행하고 relief를 관측하면 P matrix가 update되어 σ가 감소하고, 이후에는 EWMA mean 기반 결정에 수렴한다.

#### MGR-LUCB-070: LinUCB 비활성화 (Fallback)

`linucb_alpha=0`으로 설정하면 기존 §3.8 DPP로 완전 복귀한다. *(MUST)*

```
비활성화 조건:
    - linucb_alpha = 0 (AdaptationConfig)

동작:
    ucb_bonus(a) = σ(a, φ) · 0 = 0 for all a
    score(a) = Σ_k Z_k · r̂_k(a) - V · max(-lât(a), 0) + UCB · 0
             = Σ_k Z_k · r̂_k(a) - V · max(-lât(a), 0)   // §3.8 동일
    A_safe: §3.8 MGR-DPP-020 그대로                       // 변경 없음
```

**INV-116**: `linucb_alpha=0`이면, DPP 결정은 §3.8과 비트 단위 동일한 결과를 산출한다. 하위 호환을 보장한다.

## 4. Alternative Behavior

- **Engine 미연결 상태**: engine_state가 기본값(전체 0.0)인 상태에서 Monitor 신호만으로 process_signal()이 동작한다. ReliefEstimator는 default_relief prior 기반으로 예측한다. 정상은 아니나 프로토콜 위반이 아니다.

- **pending_observation 중복**: 관찰 대기 중 새 Directive가 발행되면 이전 관찰을 폐기하고 새 ObservationContext로 교체한다. 이전 액션의 relief는 학습되지 않는다.

- **QCF 값 전부 부재**: 모든 Lossy 액션의 qcf_values가 비어있으면 전부 INFINITY cost가 되어 Lossy 액션은 사실상 선택되지 않는다. Lossless 액션만 선택 가능하다.

- **후보 전무 (Warning + Lossy Only)**: Warning 모드에서 등록된 액션이 모두 Lossy이면 filter_candidates가 빈 목록을 반환하여 Directive가 생성되지 않는다.

## 5. Constraints

**[MGR-C06]** PI Controller의 output은 항상 [0, 1] 범위 내이다. *(MUST)*

**[MGR-C07]** ActionSelector는 stateless이다. 호출 간 내부 상태를 유지하지 않는다. *(MUST)*

**[MGR-C08]** ReliefEstimator의 predict는 읽기 전용이다. 모델을 생성하거나 변경하지 않는다. *(MUST)*

**[MGR-C09]** 동일 배타 그룹의 액션은 하나의 Directive에 동시 포함되지 않는다. *(MUST)* (INV-041 재확인)

**[MGR-C10]** Normal 모드에서는 액션을 발행하지 않는다. *(MUST)*

## 6. Examples

### 6.1 Gain Scheduling Effect

```
pi_compute: base_kp=1.5, ki=0.3, setpoint=0.70
gain_zones = [{above: 0.85, kp: 4.0}, {above: 0.92, kp: 10.0}]

measurement=0.80:
  effective_kp = 1.5 (base, 0.80 < 0.85)
  error = 0.10
  output ~ 1.5 * 0.10 = 0.15

measurement=0.88:
  effective_kp = 4.0 (zone 1, 0.88 >= 0.85)
  error = 0.13
  output ~ 4.0 * 0.13 = 0.52

measurement=0.95:
  effective_kp = 10.0 (zone 2, 0.95 >= 0.92)
  error = 0.20
  output ~ clamp(10.0 * 0.20, 0, 1) = 1.0
```

> **Rationale (non-normative)**: 메모리 사용률 85% 이상에서는 OOM 위험이 급격히 증가하므로 비례 이득을 높여 빠르게 Critical 모드에 진입시킨다.

### 6.2 Anti-Windup in Practice

```
pi_compute: kp=1.5, ki=0.3, setpoint=0.70, integral_clamp=2.0

t=0~10s: measurement=0.90 sustained, but no lossless action available
         -> can_act set to false
         -> integral frozen at current value (no accumulation)

t=11s:   SwitchHw becomes available -> can_act set to true
         -> integral starts accumulating again
         -> output reflects actual accumulated pressure, not inflated value
```

### 6.3 Cross-Domain Pressure Resolution

```
pressure = (compute=0.6, memory=0.7, thermal=0.0)
mode = Critical

Available actions and their predicted relief:
  SwitchHw:       relief=(0.5, 0.0, 0.3, -0.1), cost=0.0
  KvEvictSliding: relief=(0.0, 0.7, 0.0,  0.0), cost=0.5
  Throttle:       relief=(0.3, 0.0, 0.2, -0.3), cost=0.0

Optimal: {SwitchHw, KvEvictSliding}
  total_relief = (0.5, 0.7, 0.3, -0.1)
  satisfies: 0.5 < 0.6 (compute) -> NO

Best-effort: {SwitchHw, KvEvictSliding, Throttle}
  total_relief = (0.8, 0.7, 0.5, -0.4)
  latency check: -0.4 >= -0.5 (budget) -> OK
  satisfies: all >= pressure -> YES, cost = 0.0 + 0.5 + 0.0 = 0.5
  -> Selected
```

### 6.4 RLS Convergence

```
OnlineLinearEstimator, D=13, lambda=0.995

Action: KvEvictSliding
Repeated observation: state=[0.5, 1.0, 0.3, 0, ...], actual_relief=(0, 0.8, 0, 0)

After  1 observe: predict -> (0, ~0.79, 0, 0)   (빠른 초기 수렴, P 크기 때문)
After 10 observe: predict -> (0, ~0.80, 0, 0)   (수렴)
After 30 observe: predict -> (0, ~0.80, 0, 0)   (안정)

Different state=[0.8, 0.0, 0.6, 0, ...]:
  predict -> (0, ~0.48, 0, 0)   (다른 상태에서는 다른 예측)
```

## 7. Rationale (non-normative)

### 왜 PI이고 PID가 아닌가

미분항(D)은 측정값의 순간 변화율에 반응한다. 센서 노이즈(특히 `/proc/meminfo`의 캐시 변동, GPU 사용률 스파이크)가 미분항을 크게 흔들어 불필요한 액션 발행을 유발할 수 있다. PI만으로 충분한 응답 특성을 확보하면서 노이즈 민감성을 회피한다.

### 왜 gain scheduling이 memory만인가

메모리 사용률은 85~95% 구간에서 OOM 위험이 비선형으로 급증한다. 고정 Kp로는 이 구간에서 충분한 반응 속도를 얻기 어렵다. Compute와 thermal은 상대적으로 선형적인 위험 증가를 보이므로 고정 Kp로 충분하다.

### 왜 전수 탐색인가

등록 가능 액션이 최대 8종(2^8 = 256 조합)이므로 전수 탐색이 실용적이다. 그리디 알고리즘은 cross-domain relief(예: SwitchHw가 compute + thermal을 동시 해소)를 놓칠 수 있다. 전수 탐색은 전역 최적을 보장한다.

### 왜 RLS이고 gradient descent가 아닌가

RLS는 매 관측마다 closed-form 갱신으로 빠르게 수렴한다 (learning rate tuning 불필요). 관측이 드물고(OBSERVATION_DELAY=3초), 빠른 cold-start 수렴이 중요한 온라인 학습 환경에 적합하다. SGD는 learning rate 선택에 민감하고, 적은 관측에서 불안정할 수 있다.

### 왜 bias를 EMA로 별도 갱신하는가

RLS는 선형 모델 W * phi의 가중치를 갱신한다. bias를 phi에 augment(차원 추가)하면 P matrix가 (D+1) x (D+1)로 확대되어 계산량이 증가한다. 별도 EMA로 bias를 갱신하면 P matrix 크기를 D x D로 유지하면서 bias 적응을 달성한다.

### 왜 Memory는 PI가 아닌 임계값 기반인가

Memory pressure(OOM)는 초 단위로 발생한다. ZRAM 환경에서도 swap thrashing은 즉각적이다 (S25 실측: 11GB instant pressure → 5초 내 OOM kill). PI Controller의 I항 누적과 P항 평활화는 수 초의 응답 지연을 유발하여 메모리 위기 시 치명적이다. 반면 Compute(CPU spike는 ms 변동)와 Thermal(열 관성이 크므로 수 초~수십 초 변동)은 PI 평활화가 noise 제거에 효과적이다.

이를 위해 MemoryMonitor는 100ms 이하 주기로 폴링해야 하며(MGR-022), 임계값 기반 직접 매핑(MGR-ALG-013a)으로 측정값이 상승하는 즉시 압력을 반영한다. 폴링(≤100ms) + 직접 매핑(평활화 없음) + 메인 루프 recv_timeout(50ms)의 조합으로 Supervisory가 최대 ~150ms 내에 Warning/Critical 전이를 판정할 수 있다.

### 왜 도메인별 전략이 교체 가능한가

도메인별 압력 계산의 최적 전략은 하드웨어, 모델 크기, 운영 환경에 따라 달라진다. 전략 패턴으로 추상화하면:
- 모든 도메인에 PI를 적용하는 구성 (현재 코드)
- Memory만 임계값 기반인 구성 (권장 설계)
- PID, 적응형 게인, ML 기반 등 향후 전략

을 코드 변경 최소화로 전환할 수 있다. PolicyStrategy trait(MGR-023)이 정책 전체를, 도메인별 전략이 개별 압력 계산을 각각 추상화한다.

### 왜 OBSERVATION_DELAY가 3초인가

1초 Monitor 폴링 주기의 3배로, 액션 적용 후 시스템이 안정화되기에 충분한 시간이다. PI Controller의 응답 시간(수초)과 OS 메모리 회수/열 방출의 지연을 고려한 경험적 값이다. Supervisory hold_time(4초)보다 짧아야 관측 완료 후 디에스컬레이션을 판단할 수 있다.
