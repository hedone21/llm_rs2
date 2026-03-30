# Manager State Machines

> **TL;DR**: Manager 내부 5개 상태 머신의 전이 조건을 빠짐없이 열거한다. (1) OperatingMode 3-state (Normal/Warning/Critical): 에스컬레이션 즉시, 디에스컬레이션 hold_time 후 1단계. (2) ConnectionState 3-state (Listening/Connected/Disconnected): 양방향 채널 연결 수명주기. (3) ThresholdEvaluator 4-level (Normal/Warning/Critical/Emergency): 히스테리시스 기반 Level 판정. (4) PolicyPipeline 복합 상태: needs_action 판정, pending_observation 수명, 디에스컬레이션 Directive 조건. (5) ReliefEstimator 모델 수명: Absent/Initialized/Learning.

## 1. Purpose and Scope

이 문서는 Manager 내부 **5개 상태 머신**의 전이 테이블, 조건, 불변식을 빠짐없이 열거한다. 각 상태 머신에 대해 모든 상태, 모든 이벤트, 모든 전이를 정의한다.

**이 파일이 명세하는 것:**

- OperatingMode 상태 머신 (Supervisory Layer)
- ConnectionState 상태 머신 (Emitter Layer)
- ThresholdEvaluator 상태 머신 (Monitor Layer)
- PolicyPipeline 내부 복합 상태 (Policy Layer)
- ReliefEstimator 모델 수명 상태 (Policy Layer)

**이 파일이 명세하지 않는 것:**

- Manager 아키텍처 개요 → `20-manager.md`
- PI/Supervisory/Selector/Relief 알고리즘 수식 → `22-manager-algorithms.md`
- 데이터 스키마 → `23-manager-data.md`
- 프로토콜 시퀀스 → `12-protocol-sequences.md`

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **State Machine** | 유한 개의 상태, 이벤트, 전이 규칙으로 구성된 모델. |
| **Escalation** | 더 높은 심각도로의 상태 전이 (Normal→Warning, Warning→Critical 등). |
| **De-escalation** | 더 낮은 심각도로의 상태 전이. |
| **Hysteresis** | 에스컬레이션과 디에스컬레이션에 서로 다른 임계값을 적용하여 빈번한 전이(flapping)를 방지하는 기법. |
| **Hold Time** | 디에스컬레이션 전 해당 조건을 유지해야 하는 최소 시간. |
| **Peak Pressure** | `PressureVector.max()` — 전체 도메인 중 최대 압력 값. |

## 3. Specification

### 3.1 OperatingMode State Machine [MGR-050 ~ MGR-059]

Supervisory Layer의 핵심 상태 머신이다. PressureVector의 peak pressure를 입력으로 받아 OperatingMode를 결정한다.

**[MGR-050]** 상태 정의 — 3개 상태. 순서 관계: Normal < Warning < Critical. *(MUST)*

```
               즉시 에스컬레이션
          ┌────────────────────────┐
          │                        ▼
    ┌─────────┐   즉시    ┌─────────┐   즉시    ┌──────────┐
    │ Normal  │─────────→│ Warning │─────────→│ Critical │
    └─────────┘          └─────────┘          └──────────┘
          ▲   hold_time후   │  ▲   hold_time후   │
          └─────────────────┘  └─────────────────┘
              1단계 하강             1단계 하강
```

**[MGR-051]** 에스컬레이션 규칙 — 즉시 전이. 단계 건너뛰기가 가능하다 (Normal→Critical 직접 가능). *(MUST)*

- `peak >= critical_threshold` → Critical (현재 모드 무관)
- `peak >= warning_threshold AND mode < Warning` → Warning

**[MGR-052]** 디에스컬레이션 규칙 — hold_time 안정화 후 **1단계**씩만 하강. *(MUST)*

- Critical→Warning: `peak < critical_release` for hold_time
- Warning→Normal: `peak < warning_release` for hold_time
- Critical→Normal **직접 불가** (반드시 Warning 경유)

**[MGR-053]** stable_since 타이머 — 디에스컬레이션 조건 충족 시작 시점을 기록한다. 조건 미충족(재에스컬레이션 또는 압력 재상승) 시 None으로 리셋한다. *(MUST)*

**[MGR-054]** 기본 파라미터 (SupervisoryConfig default): *(SHOULD)*

| 파라미터 | 기본값 |
|---------|--------|
| warning_threshold | 0.4 |
| critical_threshold | 0.7 |
| warning_release | 0.25 |
| critical_release | 0.50 |
| hold_time | 4.0초 |

**[MGR-055]** 전이 테이블 (완전 열거): *(MUST)*

| 현재 상태 | 조건 | 다음 상태 | 부수 효과 |
|----------|------|----------|----------|
| Normal | peak >= critical_threshold | Critical | stable_since = None |
| Normal | peak >= warning_threshold | Warning | stable_since = None |
| Normal | peak < warning_threshold | Normal | stable_since = None |
| Warning | peak >= critical_threshold | Critical | stable_since = None |
| Warning | peak >= warning_threshold | Warning | stable_since = None |
| Warning | warning_release <= peak < warning_threshold | Warning | stable_since = None |
| Warning | peak < warning_release, stable_since=None | Warning | stable_since = now |
| Warning | peak < warning_release, elapsed < hold_time | Warning | (유지) |
| Warning | peak < warning_release, elapsed >= hold_time | Normal | stable_since = None |
| Critical | peak >= critical_threshold | Critical | stable_since = None |
| Critical | critical_release <= peak < critical_threshold | Critical | stable_since = None |
| Critical | peak < critical_release, stable_since=None | Critical | stable_since = now |
| Critical | peak < critical_release, elapsed < hold_time | Critical | (유지) |
| Critical | peak < critical_release, elapsed >= hold_time | Warning | stable_since = None |

> **참고 (non-normative)**: Manager와 Engine 간 모드 매핑:
> - Manager Normal ↔ Engine Normal
> - Manager Warning ↔ Engine Degraded
> - Manager Critical ↔ Engine Minimal

### 3.2 ConnectionState State Machine [MGR-060 ~ MGR-066]

Manager의 Engine 연결 상태를 관리하는 상태 머신이다. 프로토콜 수준 정의는 PROTO-040~046에 있으며, 여기서는 Manager 내부 구현 관점의 전이를 명세한다.

**[MGR-060]** 상태 정의 — 3개 상태. 초기 상태: Listening. *(MUST)*

| 상태 | 설명 |
|------|------|
| Listening | 소켓 바인드 완료, 클라이언트 대기. |
| Connected | Engine 연결 수립, Reader 스레드 활성. |
| Disconnected | 연결 끊김, 재연결 대기. |

```
                ┌──────────────┐
                │  Listening   │ ◀── 초기 상태 (bind + listen)
                └──────┬───────┘
                       │ accept()
                       ▼
                ┌──────────────┐
       ┌──────▶│  Connected   │
       │       └──────┬───────┘
       │              │ write error / reader EOF / inbox disconnected
       │              ▼
       │       ┌──────────────┐
       └───────┤ Disconnected │
 ensure_connected()   └──────────────┘
 (non-blocking accept)
```

> **참고 (non-normative)**: `01-architecture.md` SYS-093은 4-state (Active 포함)를 도시하나, 코드에서 Active는 Connected의 논리적 하위 조건(Capability 수신 여부)이며 별도 상태가 아니다.

**[MGR-061]** 전이 테이블 (완전 열거): *(MUST)*

| 현재 상태 | 이벤트 | 다음 상태 | 부수 효과 |
|----------|--------|----------|----------|
| Listening | accept() 성공 | Connected | Reader 스레드 시작, SyncSender(64) 생성 |
| Listening | accept() 실패/timeout | Listening | (유지) |
| Connected | write 오류 | Disconnected | -- |
| Connected | Reader EOF (inbox drop) | Disconnected | Reader 스레드 종료 |
| Connected | 정상 통신 | Connected | (유지) |
| Disconnected | emit_directive() 호출 → ensure_connected() → non-blocking accept 성공 | Connected | 새 Reader 스레드 시작 |
| Disconnected | ensure_connected() → accept 실패 | Disconnected | (유지) |

**[MGR-062]** Listening → Connected 전환 — `wait_for_client(timeout, shutdown)`. 블로킹 대기. timeout 초과 또는 shutdown 시 프로세스를 종료한다. *(MUST)*

**[MGR-063]** 재연결 메커니즘 — Disconnected 상태에서 `emit_directive()` 호출 시 `ensure_connected()`로 non-blocking accept를 시도한다. 성공 시 Connected 전환, 새 Capability 수신을 기대한다 (PROTO-045). 실패 시 Disconnected 유지, directive skip (비치명적 Ok 반환). *(MUST)*

**[MGR-064]** Reader 스레드 수명 — Connected 진입 시 시작, Connected 이탈 시 종료. ReaderHandle (RAII)로 관리된다. *(MUST)*

**[MGR-065]** inbox 용량 — `mpsc::sync_channel(64)`. 가득 차면 Reader 스레드가 블로킹되어 자연적 흐름 제어가 발생한다. *(MUST)* (SEQ-090 참조)

**[MGR-066]** TCP 동일 패턴 — TcpConnectionState는 ConnectionState와 동일한 3-state 전이를 따른다. *(MUST)*

### 3.3 ThresholdEvaluator State Machine [MGR-067 ~ MGR-073]

**D-Bus Emitter 전용** 상태 머신이다. D-Bus 전송 경로에서 raw 센서 값을 4-level (Normal/Warning/Critical/Emergency)로 변환하여 D-Bus 와이어 메시지에 포함한다. 내부 SystemSignal에는 Level이 포함되지 않으며, Policy Layer에서도 사용하지 않는다 (MGR-021).

> **참고 (non-normative)**: Emergency는 D-Bus 경로에서 Engine이 자율적으로 Suspend하는 트리거이다 (SYS-055, SYS-085a). Manager의 OperatingMode(Normal/Warning/Critical)에는 Emergency가 존재하지 않는다.

**[MGR-067]** 상태 정의 — 4개 상태 (Level enum). 초기 상태: Normal. *(MUST)*

**[MGR-068]** Direction — 두 가지 방향을 지원한다: *(MUST)*

| Direction | 의미 | 사용 Monitor |
|-----------|------|-------------|
| Ascending | 값이 높을수록 위험 | ThermalMonitor, ComputeMonitor |
| Descending | 값이 낮을수록 위험 | MemoryMonitor, EnergyMonitor |

**[MGR-069]** Thresholds 파라미터: *(MUST)*

| 필드 | 설명 |
|------|------|
| warning | Warning 에스컬레이션 임계값 |
| critical | Critical 에스컬레이션 임계값 |
| emergency | Emergency 에스컬레이션 임계값. `f64::MAX`(Ascending) 또는 `f64::MIN`(Descending) 설정 시 해당 레벨 비활성. |
| hysteresis | 회복 방향 오프셋. |

**[MGR-070]** 에스컬레이션 규칙 (Ascending 방향) — 즉시 전이. 단계 건너뛰기 가능. *(MUST)*

```
if value >= emergency_up AND current < Emergency → Emergency
if value >= critical_up AND current < Critical   → Critical
if value >= warning_up AND current < Warning     → Warning
```

**[MGR-071]** 회복 규칙 (Ascending 방향) — recovery threshold = threshold - hysteresis. 교차 시 가능한 최저 Level로 직접 하강. *(MUST)*

```
warning_down  = warning_up - hysteresis
critical_down = critical_up - hysteresis
emergency_down = emergency_up - hysteresis

if current == Emergency AND value < emergency_down:
    if value < warning_down    → Normal
    elif value < critical_down → Warning
    else                       → Critical

if current == Critical AND value < critical_down:
    if value < warning_down → Normal
    else                    → Warning

if current == Warning AND value < warning_down:
    → Normal
```

**[MGR-072]** Descending 방향 — 부등호 반전. recovery threshold = threshold + hysteresis. *(MUST)*

에스컬레이션: `value <= threshold` (값이 임계값 이하로 떨어지면 위험).
회복: `value > threshold + hysteresis` (값이 회복 임계값 위로 올라가야 안전).

**[MGR-073]** 전이 테이블 (Ascending 방향, 완전 열거): *(MUST)*

`warning_down = warning_up - hysteresis`, `critical_down = critical_up - hysteresis`, `emergency_down = emergency_up - hysteresis`

| 현재 Level | 값 범위 | 다음 Level |
|-----------|--------|-----------|
| Normal | value < warning_up | Normal |
| Normal | warning_up <= value < critical_up | Warning |
| Normal | critical_up <= value < emergency_up | Critical |
| Normal | value >= emergency_up | Emergency |
| Warning | value < warning_down | Normal |
| Warning | warning_down <= value < critical_up | Warning |
| Warning | critical_up <= value < emergency_up | Critical |
| Warning | value >= emergency_up | Emergency |
| Critical | value < warning_down | Normal |
| Critical | warning_down <= value < critical_down | Warning |
| Critical | critical_down <= value < emergency_up | Critical |
| Critical | value >= emergency_up | Emergency |
| Emergency | value < warning_down | Normal |
| Emergency | warning_down <= value < critical_down | Warning |
| Emergency | critical_down <= value < emergency_down | Critical |
| Emergency | value >= emergency_down | Emergency |

Monitor별 기본 임계값:

| Monitor | 방향 | warning | critical | emergency | hysteresis |
|---------|------|---------|----------|-----------|------------|
| Memory | Descending | 40.0% | 20.0% | 10.0% | 5.0% |
| Thermal | Ascending | 60000 mc | 75000 mc | 85000 mc | 5000 mc |
| Compute | Ascending | 70.0% | 90.0% | f64::MAX | 5.0% |
| Energy | Descending | 30.0% | 15.0% | 5.0% | 2.0 (하드코딩) |

> **참고 (non-normative)**: Energy hysteresis는 EnergyMonitorConfig에 필드가 없으며, 코드 내 고정값 2.0이다.

### 3.4 PolicyPipeline Internal State [MGR-074 ~ MGR-082]

HierarchicalPolicy의 내부 동적 상태와 전이 조건이다. 단일 enum 상태 머신이 아닌, 여러 독립 상태 변수의 조합으로 동작한다.

**[MGR-074]** 복합 상태 개요: *(MUST)*

| 상태 변수 | 타입 | 의미 |
|----------|------|------|
| pressure | PressureVector | 현재 압력 벡터 (compute, memory, thermal). Compute/Thermal은 PI 출력, Memory는 임계값 직접 매핑. |
| prev_mode | OperatingMode | 이전 사이클의 모드 |
| pending_observation | Option\<ObservationContext\> | 관찰 대기 중인 액션 컨텍스트 |
| last_acted_pressure | PressureVector | 마지막 액션 발행 시점의 압력 |
| engine_state | FeatureVector | 최신 Heartbeat 기반 엔진 상태 (13차원) |
| available_actions | Vec\<ActionId\> | Engine이 보고한 가용 액션 |
| active_actions_reported | Vec\<ActionId\> | Engine이 보고한 활성 액션 |
| last_signal_time | HashMap\<str, Instant\> | 도메인별 마지막 신호 시각 (실제 dt 계산) |

**[MGR-075]** process_signal() 실행 흐름 — 6단계 순차 처리: *(MUST)*

```
function process_signal(signal) -> Option<EngineDirective>:
    // 1. PI Controller 갱신
    update_pressure(signal)

    // 2. Supervisory → mode 결정
    mode = supervisory.evaluate(pressure)

    // 3. 관찰 갱신
    update_observation()

    // 4. needs_action 판정
    needs_action = evaluate_needs_action(mode)

    // 5. 액션 선택 및 Directive 생성 (needs_action=true 시)
    result = None
    if needs_action:
        result = select_and_build_directive(mode)

    // 6. De-escalation 처리 (prev_mode > mode 시)
    if prev_mode > mode AND result is None:
        result = build_de_escalation_directive(prev_mode, mode)

    prev_mode = mode
    return result
```

**[MGR-076]** needs_action 판정 조건: *(MUST)*

| mode | mode_changed | pressure > 1.2x last_acted | needs_action |
|------|-------------|---------------------------|-------------|
| Normal | * | * | false |
| Warning | true | * | true |
| Warning | false | true | true |
| Warning | false | false | false |
| Critical | true | * | true |
| Critical | false | true | true |
| Critical | false | false | false |

- `mode_changed` = (current_mode != prev_mode)
- `pressure > 1.2x last_acted` = PressureVector.any_domain_exceeds(last_acted_pressure, 1.2)

**[MGR-077]** pending_observation 수명: *(MUST)*

| 이벤트 | 전이 |
|--------|------|
| Directive 전송 | None → Some(ObservationContext { pressure_before, feature_vec, applied_actions, applied_at=now }) |
| 3.0초 경과 | Some → None. relief = pressure_before - pressure_current. estimator.observe() 호출. |
| 3.0초 미경과 | Some 유지 (관찰 대기) |
| 새 Directive (관찰 중) | 이전 관찰 폐기, 새 ObservationContext로 교체 |

**[MGR-078]** 디에스컬레이션 Directive 생성 조건: *(MUST)*

동일 사이클에서 needs_action 경로에서 이미 Directive가 생성된 경우, 디에스컬레이션 Directive는 발행하지 않는다.

| prev_mode | current_mode | 조건 | Directive |
|-----------|-------------|------|-----------|
| Critical | Warning | needs_action 경로에서 Directive 미생성 | build_lossy_release_directive() → RestoreDefaults |
| Critical | Normal | needs_action 경로에서 Directive 미생성 | build_restore_directive() → RestoreDefaults |
| Warning | Normal | needs_action 경로에서 Directive 미생성 | build_restore_directive() → RestoreDefaults |
| * | (동일 또는 상위) | -- | (없음, 에스컬레이션 시 needs_action 경로에서 처리) |

**[MGR-079]** last_acted_pressure 갱신 — Directive 전송 시 현재 pressure를 last_acted_pressure에 복사한다. needs_action 1.2x 비교의 기준점이 된다. *(MUST)*

**[MGR-080]** engine_state (FeatureVector) 갱신 매핑 — Heartbeat(EngineStatus) → FeatureVector 인덱스: *(MUST)*

| 인덱스 | 이름 | 소스 | 변환 |
|--------|------|------|------|
| 0 | KV_OCCUPANCY | kv_cache_utilization | 직접 |
| 1 | IS_GPU | active_device | "opencl" 포함 시 1.0, 아니면 0.0 |
| 2 | TOKEN_PROGRESS | kv_cache_tokens | / 2048 |
| 3 | IS_PREFILL | (미사용) | 0.0 |
| 4 | KV_DTYPE_NORM | (미사용) | 0.0 |
| 5 | TBT_RATIO | actual_throughput | / 100.0 |
| 6 | TOKENS_GENERATED_NORM | tokens_generated | / 2048 |
| 7 | ACTIVE_SWITCH_HW | (미사용) | 0.0 |
| 8 | ACTIVE_THROTTLE | (미사용) | 0.0 |
| 9 | ACTIVE_KV_OFFLOAD | (미사용) | 0.0 |
| 10 | ACTIVE_EVICTION | eviction_policy | != "none" 시 1.0, 아니면 0.0 |
| 11 | ACTIVE_LAYER_SKIP | skip_ratio | > 0.0 시 1.0, 아니면 0.0 |
| 12 | ACTIVE_KV_QUANT | (미사용) | 0.0 |

**[MGR-081]** available_actions / active_actions 파싱 — EngineStatus의 available_actions/active_actions 문자열 배열을 `ActionId::from_str()`로 파싱한다. 인식 불가 문자열은 skip한다. *(MUST)*

**[MGR-082]** last_signal_time 기반 실제 dt 계산 — 도메인별 이전 신호 시각과 현재 시각의 차이를 PI dt로 사용한다. 최초 신호 시 기본값 (config dt, 0.1초). *(MUST)*

### 3.5 ReliefEstimator Lifecycle [MGR-083 ~ MGR-087]

OnlineLinearEstimator의 모델 수명과 상태 전이이다.

**[MGR-083]** 모델 단위 — 액션별(ActionId) 독립 LinearModel. `models: HashMap<ActionId, LinearModel>`. *(MUST)*

**[MGR-084]** 모델 수명 상태: *(MUST)*

| 상태 | 조건 | predict 동작 |
|------|------|-------------|
| Absent | models에 해당 ActionId 없음 | default_relief(action) 반환 (하드코딩 prior) |
| Initialized | model 존재, observation_count=0 | default_relief(action) 반환 |
| Learning | model 존재, observation_count >= 1 | 학습된 W x phi + b 반환 |

**[MGR-085]** 전이 테이블 (완전 열거): *(MUST)*

| 현재 | 이벤트 | 다음 |
|------|--------|------|
| Absent | observe(action, ...) 호출 | Learning (ensure_model → RLS update, count=1) |
| Absent | predict(action, ...) 호출 | Absent (default 반환, 모델 생성 안 함) |
| Initialized | observe(action, ...) 호출 | Learning (count=1) |
| Initialized | predict(action, ...) 호출 | Initialized (default 반환) |
| Learning | observe(action, ...) 호출 | Learning (count++) |
| Learning | predict(action, ...) 호출 | Learning (학습 모델 사용) |

**[MGR-086]** default_relief 테이블 (하드코딩 prior): *(MUST)*

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

**[MGR-087]** 영속화: *(SHOULD)*

- `save(path)`: models HashMap을 JSON 직렬화
- `load(path)`: 파일에서 복원
- 프로세스 종료 시 save, 시작 시 load
- 파일 부재 시 빈 models로 시작

## 4. Alternative Behavior

- **Heartbeat 수신 없이 Directive**: engine_state가 기본값(전체 0.0) 상태에서 Monitor 신호만으로 process_signal()이 동작 가능하다. 정상은 아니나 프로토콜 위반이 아니다. ReliefEstimator는 기본 상태 기반으로 예측한다.

- **pending_observation 중복**: 관찰 대기 중 새 Directive가 발행되면 이전 관찰을 폐기하고 새 ObservationContext로 교체한다. 이전 액션의 relief는 학습되지 않는다.

- **ThresholdEvaluator Emergency 비활성**: emergency를 `f64::MAX` (Ascending) 또는 `f64::MIN` (Descending)으로 설정하면 Emergency 레벨에 도달하지 않는다. ComputeMonitor는 이 방식으로 Emergency를 비활성화한다.

## 5. Constraints

**[MGR-C03]** OperatingMode 디에스컬레이션은 반드시 1단계씩 하강한다. Critical→Normal 직접 전이는 불가하다. *(MUST)*

**[MGR-C04]** ThresholdEvaluator의 에스컬레이션은 단계 건너뛰기가 가능하지만, 회복은 recovery threshold 교차를 반드시 요구한다. *(MUST)*

**[MGR-C05]** ConnectionState 전이에서 Connected → Listening 직접 전이는 불가하다. 반드시 Disconnected를 경유한다 (또는 프로세스 재시작). *(MUST)*

## 6. Examples

### 6.1 OperatingMode 전이 시나리오

기본 파라미터(warning_threshold=0.4, critical_threshold=0.7, warning_release=0.25, critical_release=0.50, hold_time=4.0초) 기준:

```
t=0s:  peak=0.1 → Normal
t=1s:  peak=0.5 → Warning    (즉시 에스컬레이션: 0.5 >= 0.4)
t=2s:  peak=0.8 → Critical   (즉시 에스컬레이션: 0.8 >= 0.7)
t=3s:  peak=0.3 → Critical   (0.3 < 0.50=critical_release, stable_since=now)
t=4s:  peak=0.2 → Critical   (stable 1초, < hold_time 4초)
t=7s:  peak=0.2 → Warning    (stable 4초, 디에스컬레이션 1단계)
t=8s:  peak=0.1 → Warning    (0.1 < 0.25=warning_release, stable_since=now)
t=12s: peak=0.1 → Normal     (stable 4초, 디에스컬레이션 1단계)
```

### 6.2 ThresholdEvaluator Ascending 시나리오

Thermal 기본값 (warning=60000mc, critical=75000mc, emergency=85000mc, hysteresis=5000mc):

```
t=0: temp=50000 → Normal
t=1: temp=62000 → Warning     (>= 60000=warning_up)
t=2: temp=77000 → Critical    (>= 75000=critical_up)
t=3: temp=68000 → Warning     (< 70000=critical_down, >= 55000=warning_down)
t=4: temp=53000 → Normal      (< 55000=warning_down)
```

### 6.3 ConnectionState 재연결 시나리오

```
Listening → accept → Connected → Reader EOF → Disconnected
→ emit_directive → ensure_connected → accept → Connected (새 Capability 대기)
```

### 6.4 PolicyPipeline 관찰 시나리오

```
t=0s: process_signal → Warning, needs_action=true
      → Directive(Throttle) → pending_observation=Some
t=1s: process_signal → Warning, needs_action=false
      → (관찰 대기 중)
t=3s: process_signal → Warning, 관찰 3초 경과
      → relief 계산 → estimator.observe() → pending_observation=None
```

## 7. Rationale (non-normative)

### 왜 에스컬레이션은 즉시이고 디에스컬레이션은 지연되는가

리소스 압력은 빠르게 대응해야 하지만(시스템 OOM, 과열), 해소는 일시적일 수 있다. 성급한 디에스컬레이션은 액션 해제 → 재압력 → 재적용 사이클(flapping)을 유발한다. 비가역 lossy 액션(eviction)의 경우 불필요한 품질 손실이 누적된다.

### 왜 hold_time이 4초인가

OBSERVATION_DELAY_SECS(3초)보다 길어야 이전 액션의 relief가 측정된 후에만 디에스컬레이션한다. 1초 여유는 PI 안정화와 Monitor 샘플링 주기(1초) 1회분을 포함한다.

### 왜 ThresholdEvaluator의 회복은 히스테리시스이고 Supervisory의 디에스컬레이션은 hold_time인가

ThresholdEvaluator는 원시 센서 값에 대해 작동하며 빈도가 높다(1초 주기). 값 기반 히스테리시스가 적합하다. Supervisory는 PI 출력(이미 평활화된 값)에 대해 작동하므로 시간 기반 안정화(hold_time)가 적합하다.

### 왜 needs_action에서 1.2x 비교인가

모드 변경 없이도 압력이 유의미하게 증가(20% 이상)하면 추가 액션이 필요할 수 있다. 1.0x이면 미세 변동에도 반응하여 불필요한 Directive가 발행된다. 값은 경험적 튜닝에 의한 것이다.

### default_relief prior 값의 출처와 한계

현재 MGR-086의 default_relief 값은 도메인 전문 지식에 기반한 **초기 추정치**이다. prior 값의 최적 결정 전략(실측 기반 calibration, 모델 크기별 프리셋, 논문 참조 등)은 아직 미정이며 향후 결정이 필요하다.

실측 relief profile(S25, Qwen 2.5 1.5B, 3.5K seq)과의 주요 괴리:
- SwitchHw: prior memory=0.0이나 실측은 -810MB (★★★). GPU→CPU 전환의 메모리 해제 효과를 prior가 반영하지 못함.
- KvEvictH2o/Sliding: prior memory=0.6/0.7이나 실측은 null (0MB). 소형 모델에서 KV가 RSS의 3%에 불과.
- KvOffloadDisk: prior memory=0.4이나 실측은 +33MB (역효과). 복사 기반 백업이므로 relief가 아닌 overhead.

이 괴리는 C3(온라인 학습)의 존재 이유이다. Prior가 부정확하더라도 RLS가 수 회 관측 후 실측 기반으로 보정한다. Prior는 학습 데이터가 없는 cold-start 구간에서만 사용되며, 이 기간의 suboptimal 선택은 시스템 안전성에 영향을 주지 않는다 (Warning 모드에서는 lossless만 허용, Critical에서만 lossy 사용).

### 왜 ReliefEstimator가 predict 호출만으로는 모델을 생성하지 않는가

predict는 읽기 전용 연산이다. 실제 관측 데이터 없이 모델을 초기화하면 RLS의 P 행렬이 큰 초기값(100xI)으로 설정되어 첫 observe에서 과도한 가중치 갱신이 발생할 수 있다. observe 호출 시에만 생성하여 항상 관측 데이터와 함께 초기화한다.

### 왜 디에스컬레이션 Directive가 needs_action 경로의 Directive보다 우선순위가 낮은가

동일 사이클에서 needs_action이 true이면 에스컬레이션 또는 압력 증가에 대한 능동 대응이 필요한 상황이다. 디에스컬레이션 Directive(RestoreDefaults)와 동시에 새 액션 Directive를 발행하면 Engine이 한 사이클 내에서 해제 후 재적용을 수행하게 되어 비효율적이다. needs_action 경로가 이미 현재 모드에 맞는 최적 액션을 선택했으므로 추가 디에스컬레이션 Directive는 불필요하다.
