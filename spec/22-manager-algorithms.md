# Manager Algorithms

> **TL;DR**: Manager Policy Layer 내부 4개 핵심 알고리즘의 상세 명세.
> (1) 도메인별 압력 계산: Compute/Thermal은 PI Controller (gain scheduling, anti-windup), Memory는 임계값 기반 직접 매핑 (OOM 즉각 대응). 전략 패턴으로 교체 가능.
> (2) Supervisory: peak pressure 기반 3-state 운영 모드 판정.
> (3) ActionSelector: 2^N 전수 탐색 cross-domain 조합 최적화.
> (4) ReliefEstimator: RLS 온라인 선형 회귀로 액션 효과 예측.
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

> **참고**: 현재 구현은 raw measurement 직접 전달(passthrough)이다. 향후 비선형 매핑(구간별 기울기), Level 기반 이산 매핑 등으로 전략을 교체할 수 있다. 핵심 제약은 **PI 평활화를 거치지 않는다**는 것이다.

각 인스턴스는 독립적으로 상태(integral)를 유지하며 서로 영향을 주지 않는다. *(MUST)*

#### MGR-ALG-014: Measurement Normalization (update_pressure)

SystemSignal에서 도메인별 측정값을 추출하여 [0, 1]로 정규화한 뒤 해당 PI 인스턴스에 입력한다. *(MUST)*

| SystemSignal | 측정값 변환 | 압력 계산 전략 |
|-------------|-----------|--------------|
| MemoryPressure | `m = clamp(1 - available_bytes / total_bytes, 0, 1)` | **직접 매핑** (MGR-ALG-013a) |
| ThermalAlert | `m = clamp(temperature_mc / 85000, 0, 1)` | PI Controller (pi_thermal) |
| ComputeGuidance | `m = max(clamp(cpu_usage_pct/100, 0, 1), clamp(gpu_usage_pct/100, 0, 1))` | PI Controller (pi_compute) |
| EnergyConstraint | MGR-ALG-015 참조 | pi_compute |

- total_bytes = 0 인 경우 m = 0 (방어적 처리).
- Thermal 정규화 기준: 85000 mc (85 C) = 1.0.
- Compute: CPU와 GPU 사용률 중 최대값.

#### MGR-ALG-015: EnergyConstraint Max-Floor Processing

EnergyConstraint는 별도 PI 인스턴스 없이 compute PI에 보조 기여한다. *(MUST)*

```
PRE:  level in {Normal, Warning, Critical, Emergency}
POST: compute pressure 갱신

energy_measurement = level_to_measurement(level) * 0.5
combined = max(pressure.compute, energy_measurement)
compute_pressure = pi_compute.update(combined, dt)
```

level_to_measurement 매핑:

| Level | measurement |
|-------|-------------|
| Normal | 0.0 |
| Warning | 0.55 |
| Critical | 0.80 |
| Emergency | 1.0 |

> **Rationale (non-normative)**: 전력 상태의 시간 스케일(분~시간)이 PI의 초 단위 제어와 상이하므로 별도 도메인 대신 compute 보조 신호로 반영한다. 0.5 가중치는 energy가 compute를 과도하게 지배하지 않도록 한다.

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

> **참고**: `21-manager-state.md` MGR-086 테이블과 동일하다.

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

> **참고**: `21-manager-state.md` MGR-080 참조. Index 3, 4, 7, 8, 9, 12는 향후 확장 예약.

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

> **참고**: `21-manager-state.md` MGR-077과 동일하다.

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

> **참고**: `21-manager-state.md` MGR-075의 6단계와 동일하다 (Step 7은 상태 갱신).

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
