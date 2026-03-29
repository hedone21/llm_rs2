# Protocol Sequences

> **TL;DR**: Manager ↔ Engine 간 정규 상호작용 시퀀스를 시간 축에서 정의한다. 세션 수명주기 4단계(Handshake → Steady-State → Pressure Response → Termination), 압력 에스컬레이션/디에스컬레이션 시퀀스, QCF 요청 시퀀스(Critical 전환 시 2 round-trip), 관찰·학습 사이클, 재연결·오류 시퀀스, 백프레셔 흐름 제어, D-Bus 시퀀스를 완전 구체적으로 기술한다. 와이어 포맷은 `10-protocol.md`, 메시지 필드는 `11-protocol-messages.md`에서 다룬다.

## 1. Purpose and Scope

이 문서는 Manager ↔ Engine 간 정규 상호작용 시퀀스를 정의한다. "언제, 어떤 순서로, 어떤 메시지가 교환되는가"를 시간 축에서 명세한다.

**이 파일이 명세하는 것:**

- 세션 수명주기 단계 (Handshake, Steady-State, Pressure Response, Termination)
- 각 단계의 메시지 교환 순서와 타이밍 제약
- 압력 에스컬레이션/디에스컬레이션 시퀀스
- 액션 효과 관찰·학습 시퀀스
- 재연결 시퀀스
- 프로토콜 수준 오류 시퀀스
- 흐름 제어(백프레셔) 시퀀스
- QCF 요청 시퀀스 (Critical 전환 시)
- D-Bus 전송 시퀀스

**이 파일이 명세하지 않는 것:**

- 와이어 포맷, 프레이밍, 직렬화 규칙 → `10-protocol.md`
- 개별 메시지의 필드명, 타입, 범위 → `11-protocol-messages.md`
- Manager 내부 알고리즘 (PI Controller, Supervisory, ActionSelector, ReliefEstimator) → `22-manager-algorithms.md`
- Engine 내부 명령 처리 로직 (eviction, quantization, layer skip) → `32-engine-algorithms.md`

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **Sequence** | 특정 시나리오에서 교환되는 메시지들의 시간 순서 있는 나열. |
| **Phase** | 세션 내 구분되는 시간 구간. Handshake, Steady-State, Pressure Response, Termination의 4단계. |
| **Escalation** | OperatingMode가 상위로 전이하는 것. Normal → Warning → Critical. |
| **De-escalation** | OperatingMode가 하위로 전이하는 것. Critical → Warning → Normal. |
| **Observation Window** | 액션 적용 후 실제 효과를 측정하기까지 대기하는 시간. 현재 3.0초. |
| **Round-trip** | Directive 전송부터 대응 Response 수신까지의 교환 단위. |
| **Monitor Signal** | Manager의 Monitor 스레드가 OS/하드웨어 상태를 샘플링하여 생성하는 SystemSignal. |
| **Drain** | 채널에 누적된 모든 메시지를 한 번에 소비하는 동작. |
| **OperatingMode 매핑** | Manager의 Normal/Warning/Critical은 Engine의 Normal/Degraded/Minimal에 대응한다. `00-overview.md` §2 참조. Manager가 모드를 전이하면 Directive를 통해 Engine의 동작 수준이 변경된다. |

## 3. Specification

### 3.1 Session Lifecycle Phases [SEQ-010 ~ SEQ-013]

**[SEQ-010]** Phase 1 — **Connection & Handshake**. Engine이 Manager에 연결하고 Capability를 전송한다. Manager가 연결을 수락하고 Capability를 수신한다. *(MUST)*

**[SEQ-011]** Phase 2 — **Steady-State Monitoring**. Engine이 주기적으로 Heartbeat를 전송한다. Manager는 PI Controller를 갱신한다. OperatingMode가 Normal이면 Directive를 전송하지 않는다. *(MUST)*

**[SEQ-012]** Phase 3 — **Pressure Response**. Monitor 신호가 압력 상승을 감지한다. Manager가 모드 전이를 결정하고 Directive를 전송한다. Engine이 명령을 실행하고 Response를 전송한다. Manager가 Observation Window 후 효과를 측정한다. *(MUST)*

**[SEQ-013]** Phase 4 — **Session Termination**. Engine 프로세스 종료 또는 연결 끊김. Manager는 Disconnected 상태로 전이한다. *(MUST)*

```
┌──────────────────────────────────────────────────────────────────┐
│                      Session Lifecycle                           │
│                                                                  │
│  Phase 1          Phase 2              Phase 3        Phase 4    │
│  Handshake        Steady-State         Pressure       Termin.    │
│ ┌──────────┐    ┌──────────────┐    ┌─────────────┐  ┌───────┐  │
│ │ Connect  │    │  Heartbeat   │    │ Escalation  │  │  EOF  │  │
│ │ Capabil. │───→│  Monitoring  │───→│ Directive   │──│  or   │  │
│ │ 1st HB   │    │  (Normal)    │    │ Response    │  │ Close │  │
│ └──────────┘    └──────────────┘    │ Observation │  └───────┘  │
│                        ↑            │ De-escalat. │             │
│                        └────────────└─────────────┘             │
│                         (return to Normal)                       │
└──────────────────────────────────────────────────────────────────┘
```

Phase 2와 Phase 3은 세션 수명 동안 반복된다. 압력이 해소되면 Phase 3에서 Phase 2로 복귀한다.

### 3.2 Connection & Handshake Sequence [SEQ-020 ~ SEQ-025]

**[SEQ-020]** Manager는 `bind()` + `listen()` 완료 후 Listening 상태에서 `wait_for_client(timeout)`으로 클라이언트 연결을 대기한다. 기본 타임아웃은 60초이다. *(MUST)*

**[SEQ-021]** Engine은 `Transport::connect()`를 호출하여 Manager에 연결한다. Manager는 `accept()`로 연결을 수락하고 Connected 상태로 전이한다. Manager는 Reader thread를 시작하여 Engine 메시지를 비동기로 수신한다. *(MUST)*

**[SEQ-022]** Engine은 연결 직후 `EngineMessage::Capability`를 **반드시** 전송한다. 이것이 세션의 첫 메시지이다. Capability는 세션당 정확히 1회 전송한다 (INV-015 참조). *(MUST)*

**[SEQ-023]** Manager는 Capability를 수신하여 `available_devices`, `max_kv_tokens`, `bytes_per_kv_token`, `num_layers` 등을 캐시한다. Manager는 Capability 수신 전까지 Directive를 전송하지 않아야 한다 (PROTO-044). *(SHOULD)*

**[SEQ-024]** Engine은 Capability 전송 후 `heartbeat_interval` (기본 1000ms) 경과 시 첫 Heartbeat를 전송한다. Heartbeat는 `CommandExecutor::poll()` 내에서 경과 시간을 확인하여 전송 여부를 결정한다. *(MUST)*

**[SEQ-025]** Manager는 첫 Heartbeat 수신 후 `update_engine_state()`를 호출하여 FeatureVector를 초기화한다. 이 시점부터 Manager의 PolicyPipeline이 Engine 상태를 반영한 압력 계산을 수행할 수 있다. *(MUST)*

```
  Manager                                    Engine
    │                                          │
    │  bind() + listen()                       │
    │  state: Listening                        │
    │                                          │
    │◄─────── TCP/Unix connect ────────────────│  Transport::connect()
    │  accept()                                │
    │  state: Connected                        │
    │  spawn Reader thread                     │
    │                                          │
    │◄─────── Capability ──────────────────────│  EngineMessage::Capability
    │  cache available_devices,                │  (세션의 첫 메시지, 정확히 1회)
    │  max_kv_tokens, ...                      │
    │                                          │
    │         ... heartbeat_interval 경과 ...   │
    │                                          │
    │◄─────── Heartbeat ───────────────────────│  EngineMessage::Heartbeat
    │  update_engine_state()                   │  (poll()에서 경과 시간 확인)
    │  FeatureVector 초기화                     │
    │                                          │
```

### 3.3 Steady-State Monitoring Sequence [SEQ-030 ~ SEQ-035]

**[SEQ-030]** Engine은 `CommandExecutor::poll()` 호출 시 `heartbeat_interval` (기본 1000ms) 경과 여부를 확인하고, 경과하면 `send_heartbeat(kv_snap)`을 호출한다. Heartbeat에는 현재 KV 캐시 상태, 디바이스, 처리량, 활성 액션 등 16개 필드가 포함된다. *(MUST)*

**[SEQ-031]** Manager의 Reader thread는 수신한 EngineMessage를 `sync_channel(64)` inbox에 push한다. Manager 메인 루프는 `try_recv` 반복으로 inbox의 모든 pending 메시지를 drain한다. Heartbeat 수신 시 `update_engine_state()`를 호출하여 FeatureVector를 갱신한다. *(MUST)*

**[SEQ-032]** Manager 메인 루프는 Engine 메시지 drain 후 `recv_timeout(50ms)`로 Monitor 채널에서 SystemSignal을 대기한다. Normal 모드에서는 `needs_action = false`이므로 Directive를 생성하지 않는다. *(MUST)*

**[SEQ-033]** Heartbeat와 Monitor 신호는 독립적으로 도착한다. Manager 메인 루프는 **Engine 메시지를 먼저 drain한 후** Monitor 신호를 처리한다. 이 순서는 Engine 상태를 최신화한 후 압력을 계산하기 위함이다. *(MUST)*

- 처리 우선순위: Engine messages > Monitor signals (메인 루프 코드 순서에 의한 암묵적 우선순위)

**[SEQ-034]** Engine의 `poll()`은 토큰 생성당 1회 호출된다. Heartbeat는 `poll()` 내에서 경과 시간 기반으로 전송 결정되므로, 실제 Heartbeat 주기는 토큰 생성 속도에 의존한다. *(MUST)*

**[SEQ-035]** Monitor 신호가 50ms 내에 도착하지 않으면 Manager 메인 루프는 timeout하여 다음 루프 반복으로 넘어간다 (`continue`). Engine 측은 Directive가 없으면 다음 토큰 생성 시 `poll()`을 재호출한다. *(MUST)*

```
  Manager                     Engine                     Monitor
    │                           │                           │
    │  ┌─ try_recv drain ──┐    │                           │
    │  │ (Engine msgs)     │    │                           │
    │◄─┤                   ├────│  Heartbeat (매 1000ms)    │
    │  │ update_engine_    │    │  (poll()에서 경과 확인)     │
    │  │ state()           │    │                           │
    │  └───────────────────┘    │                           │
    │                           │                           │
    │  recv_timeout(50ms) ──────┼───────────────────────────│
    │◄──────────────────────────┼───── SystemSignal ────────│
    │  process_signal()         │                           │
    │  mode=Normal → no action  │                           │
    │                           │                           │
    │  ... 50ms timeout ...     │                           │
    │  continue (no signal)     │                           │
    │                           │                           │
    ├── 1초 구간 ──────────────────────────────────────────── │
    │  Heartbeat 1회 + Monitor signals 0~N회                 │
```

### 3.4 Pressure Escalation Sequence [SEQ-040 ~ SEQ-049]

**[SEQ-040]** 트리거: Monitor 신호가 Manager 메인 루프의 `recv_timeout(50ms)`를 통해 도착한다. `process_signal(signal)`이 호출된다. *(MUST)*

**[SEQ-041]** 압력 갱신: `update_pressure(signal)`이 호출된다. 도메인별 PI Controller가 측정값을 입력받아 pressure를 계산한다. 도메인 매핑: *(MUST)*

- `MemoryPressure` → 임계값 직접 매핑 → `pressure.memory` (PI 미사용, MGR-ALG-013a 참조)
- `ThermalAlert` → `pi_thermal` → `pressure.thermal`
- `ComputeGuidance` → `pi_compute` → `pressure.compute`
- `EnergyConstraint` → `pi_compute` (보조 기여, 상세는 `22-manager-algorithms.md` 참조)

**[SEQ-042]** 모드 전이 판정: `supervisory.evaluate(&pressure)`가 현재 pressure를 기반으로 OperatingMode를 결정한다. *(MUST)*

- Normal → Warning: pressure가 `warning_threshold`를 초과
- Warning → Critical: pressure가 `critical_threshold`를 초과
- 상세 임계값 정의는 `22-manager-algorithms.md` 범위

**[SEQ-043]** 액션 필요성 판단 — `needs_action` 조건: *(MUST)*

```
needs_action ← CASE mode OF
    Normal   → false
    Warning  → mode_changed OR pressure.any_domain_exceeds(last_acted_pressure, 1.2)
    Critical → mode_changed OR pressure.any_domain_exceeds(last_acted_pressure, 1.2)
```

- `mode_changed`: 현재 모드가 이전 모드(`prev_mode`)와 다른 경우
- `any_domain_exceeds(ref, factor)`: 어느 한 도메인의 pressure가 `ref` × `factor`를 초과하는 경우

**[SEQ-044]** 액션 선택: `ActionSelector::select()`를 호출한다. 입력: registry, estimator, pressure, mode, engine_state (FeatureVector), qcf_values, latency_budget, active_actions_reported, available_actions. 출력: ActionCommand 리스트. *(MUST)*

> **참고** (non-normative): `qcf_values`는 현재 Manager의 `DegradationEstimator`가 내부 계산한 값을 사용한다. 향후 `QcfRequest`/`QcfResponse` 메시지 (MSG-014 확장)가 구현되면 Engine으로부터 어텐션 기반 QCF 프록시를 수신하여 사용할 수 있다. QCF 값의 산출 로직과 액션 선택 최적화에의 활용은 `22-manager-algorithms.md` 범위이다.

**[SEQ-045]** Directive 생성: `next_seq_id()`로 단조 증가 seq_id를 생성한다 (초기값 1, AtomicU64). ActionSelector 결과를 EngineCommand로 변환하여 `EngineDirective { seq_id, commands }` 구성. *(MUST)*

**[SEQ-046]** Directive 전송: `emitter.emit_directive(&directive)`를 호출한다. 내부적으로 `write_manager_message()` → length-prefixed JSON으로 전송. Connected 상태가 아니면 전송을 skip하고 `Ok`를 반환한다 (비치명적). *(MUST)*

**[SEQ-047]** Engine Directive 수신: `MessageLoop` thread가 `transport.recv()`로 수신하여 `cmd_tx.send(directive)`로 전달한다. Engine의 다음 `poll()` 호출 시 `cmd_rx.try_recv()`로 수신한다. *(MUST)*

**[SEQ-048]** Engine 명령 실행: `apply_command(cmd, plan)` × N회 호출. 각 명령에 대해 `CommandResult` (Ok / Partial / Rejected) 생성. 전체 결과를 `ExecutionPlan`에 누적. *(MUST)*

**[SEQ-049]** Response 전송: 각 Directive에 대해 **정확히 1개** `CommandResponse`를 전송한다 (INV-022). `seq_id`는 수신한 Directive의 seq_id와 일치해야 한다 (INV-023). `results` 배열의 길이는 Directive의 `commands` 배열 길이와 같아야 한다 (INV-024). Response는 `resp_tx` → MessageLoop thread → `transport.send()`로 전송된다. *(MUST)*

```
  Monitor          Manager                              Engine
    │                │                                     │
    │─ SystemSignal ─│                                     │
    │                │ process_signal()                     │
    │                │  ① update_pressure()                 │
    │                │  ② supervisory.evaluate() → mode     │
    │                │  ③ update_observation()              │
    │                │  ④ needs_action? → true              │
    │                │  ⑤ ActionSelector::select()          │
    │                │  ⑥ EngineDirective { seq, cmds }     │
    │                │                                     │
    │                │──── Directive(seq=N) ───────────────→│
    │                │                                     │  poll() → try_recv
    │                │                                     │  apply_command() × N
    │                │                                     │  ExecutionPlan
    │                │◄─── Response(seq=N) ────────────────│
    │                │  log response                       │
    │                │                                     │
```

### 3.5 Observation & Relief Measurement Sequence [SEQ-050 ~ SEQ-054]

**[SEQ-050]** 관찰 컨텍스트 기록: Directive 생성 직후 (전송과 동시), Manager는 `ObservationContext`를 저장한다: *(MUST)*

```
ObservationContext {
    pressure_before: current pressure snapshot,
    feature_vec:     current engine FeatureVector,
    applied_actions: list of ActionId applied,
    applied_at:      current timestamp
}
```

**[SEQ-051]** 대기 기간: `OBSERVATION_DELAY_SECS = 3.0`초. 이 기간 동안 Heartbeat 약 3개를 수신한다 (1000ms 주기 기준). Manager 메인 루프는 계속 실행되며, 이 기간 중 새 Monitor 신호가 도착해도 `pending_observation`이 존재하는 한 관찰을 업데이트하지 않는다 (단, 새 Directive를 생성하면 기존 관찰을 덮어쓴다). *(MUST)*

**[SEQ-052]** 효과 측정: 3.0초 경과 후 다음 `process_signal()` 호출 시 `update_observation()`이 실행된다. `actual_relief = pressure_before − current_pressure`를 계산한다. *(MUST)*

**[SEQ-053]** Relief Estimator 학습: `estimator.observe(action, feature_vec, actual_relief)`를 각 적용 액션에 대해 호출한다. 온라인 선형 모델이 갱신된다. *(MUST)*

**[SEQ-054]** 관찰 해제: `pending_observation = None`. 다음 액션 선택 시 갱신된 estimator를 사용한다. *(MUST)*

```
  Manager                               Engine
    │                                     │
    │── Directive(seq=N) ────────────────→│
    │  record ObservationContext           │
    │  { pressure_before, features,       │
    │    actions, timestamp }             │
    │                                     │
    │◄── Heartbeat ───────────────────────│  t + ~1s
    │◄── Heartbeat ───────────────────────│  t + ~2s
    │◄── Heartbeat ───────────────────────│  t + ~3s
    │                                     │
    │  update_observation()               │  t + 3.0s (OBSERVATION_DELAY_SECS)
    │  actual_relief =                    │
    │    pressure_before − pressure_now   │
    │  estimator.observe(action,          │
    │    feature_vec, actual_relief)      │
    │  pending_observation = None          │
    │                                     │
```

### 3.6 De-escalation Sequence [SEQ-060 ~ SEQ-064]

**[SEQ-060]** 모드 하강 감지: `supervisory.evaluate(&pressure)`가 이전 모드(`prev_mode`)보다 낮은 모드를 반환한다. `prev_mode > mode`가 참이면 디에스컬레이션 시퀀스가 시작된다. *(MUST)*

**[SEQ-061]** Critical → Warning 전이: `build_lossy_release_directive()`가 호출된다. `RestoreDefaults` 명령을 포함하는 Directive를 전송한다. 이는 모든 활성 액션을 일괄 해제한다. 다음 사이클에서 Warning 모드의 ActionSelector가 lossless 액션만 재선택한다. *(MUST)*

**[SEQ-062]** Warning 또는 Critical → Normal 전이: `build_restore_directive()`가 호출된다. `RestoreDefaults` 명령을 포함하는 Directive를 전송한다. 모든 활성 액션이 해제된다. *(MUST)*

**[SEQ-063]** `RestoreDefaults` 효과: Engine은 `active_actions`를 전체 초기화하고, `throttle_delay_ms`를 0으로, `compute_level`/`memory_level`을 Normal로 복원한다. *(MUST)*

**[SEQ-064]** 점진적 해제: Critical → Normal 전이는 2단계로 진행한다. 각 단계에서 별도의 Directive를 전송한다. *(SHOULD)*

1. Critical → Warning: `RestoreDefaults` 전송 → 모든 액션 해제 → 다음 사이클에서 Warning 모드 lossless 액션 재선택
2. Warning → Normal: `RestoreDefaults` 전송 → 모든 액션 해제

> **참고**: 현재 구현에서 `build_lossy_release_directive()`와 `build_restore_directive()`는 모두 `RestoreDefaults`만 전송한다. 세분화된 per-action release는 향후 확장 가능하다.

```
  Manager                               Engine
    │                                     │
    │  ── Critical 모드 운영 중 ──         │
    │                                     │
    │  supervisory.evaluate() → Warning   │
    │  prev_mode(Critical) > mode(Warning)│
    │  build_lossy_release_directive()    │
    │                                     │
    │── Directive [RestoreDefaults] ─────→│  ① lossy+lossless 모두 해제
    │◄── Response(Ok) ────────────────────│
    │                                     │
    │  ... 다음 process_signal() ...       │
    │  mode=Warning, needs_action=true    │
    │  ActionSelector → lossless만 선택    │
    │                                     │
    │── Directive [Throttle] ────────────→│  ② Warning용 lossless 재적용
    │◄── Response(Ok) ────────────────────│
    │                                     │
    │  ... pressure 해소 ...               │
    │                                     │
    │  supervisory.evaluate() → Normal    │
    │  build_restore_directive()          │
    │                                     │
    │── Directive [RestoreDefaults] ─────→│  ③ 모든 액션 해제
    │◄── Response(Ok) ────────────────────│
    │                                     │
```

### 3.7 Reconnection Sequence [SEQ-070 ~ SEQ-075]

**[SEQ-070]** 끊김 감지 (Manager): Reader thread가 EOF를 감지하면 종료된다. `inbox`의 `Receiver::try_recv()`가 `Disconnected`를 반환한다. Manager는 `state = Disconnected`로 전이한다. *(MUST)*

**[SEQ-071]** 끊김 감지 (Engine): `MessageLoop`의 `transport.recv()`가 `Disconnected`를 반환하면 메시지 루프 thread가 종료된다. *(MUST)*

**[SEQ-072]** Manager 독립 동작: Disconnected 상태에서 Policy 루프는 계속 실행된다 (Monitor 신호 처리, 압력 계산, 모드 전이 수행). `emit_directive()` 호출 시 `ensure_connected()`가 non-blocking `accept()`를 시도한다. 연결이 없으면 전송을 skip하고 `Ok`를 반환한다. *(MUST)*

**[SEQ-073]** 재연결 수립: 새 Engine이 `connect()`를 호출한다. Manager의 `ensure_connected()`가 non-blocking `accept()` 성공 → Connected 상태로 전이. 새 Reader thread를 시작한다. *(MUST)*

**[SEQ-074]** 새 Capability: 재연결된 Engine은 **반드시** 새 Capability를 전송해야 한다 (PROTO-045). Manager는 이전 세션의 Policy 상태 (pressure, mode, estimator)를 **유지**한다. Engine 관련 캐시 (available_devices 등)만 새 Capability로 갱신된다. *(MUST)*

**[SEQ-075]** seq_id 연속성: Manager의 `SEQ_COUNTER`는 프로세스 수명(static AtomicU64)이므로 재연결 시에도 이전 값에서 계속 증가한다 (PROTO-074). *(MUST)*

```
  Manager                                    Engine (1st)
    │                                          │
    │  state: Connected                        │
    │◄─────── Heartbeat ───────────────────────│
    │                                          │
    │  ╳ EOF ──────────────────────────────────│  프로세스 종료 / 연결 끊김
    │  Reader thread → Disconnected 감지        │
    │  state: Disconnected                     │
    │                                          │
    │  Policy 루프 계속 실행                     │
    │  emit_directive() → ensure_connected()   │
    │  accept() → WouldBlock → skip            │
    │                                          │
    │                                          Engine (2nd)
    │                                          │
    │◄─────── connect ─────────────────────────│  새 Engine 연결
    │  ensure_connected() → accept() 성공       │
    │  state: Connected                        │
    │  spawn Reader thread                     │
    │                                          │
    │◄─────── Capability ──────────────────────│  새 세션 첫 메시지 (MUST)
    │  update available_devices, ...           │
    │  (Policy 상태: pressure/mode 유지)        │
    │                                          │
    │◄─────── Heartbeat ───────────────────────│
    │  정상 흐름 재개                            │
    │  seq_id는 이전 값에서 계속 증가            │
    │                                          │
```

### 3.8 Error Sequences [SEQ-080 ~ SEQ-088]

**[SEQ-080]** JSON ParseError: 수신측은 해당 프레임을 skip하고 연결을 유지한다. 다음 프레임은 정상 처리한다 (PROTO-061). *(MUST)*

- Engine `MessageLoop`: warn 로그 출력 → `continue` (다음 recv)
- Manager Reader thread: warn 로그 출력 → `continue` (다음 read)

**[SEQ-081]** 페이로드 크기 초과: 64KB 초과 프레임 수신 시 ParseError로 처리한다 (PROTO-060). 연결을 유지한다. *(SHOULD)*

> **참고**: Engine Transport에 `MAX_PAYLOAD_SIZE = 64 * 1024` 가드가 구현되어 있다. Manager 측 Channel에는 현재 이 가드가 미적용이다.

**[SEQ-082]** 쓰기 오류: *(MUST)*

- Manager: `write_manager_message()` 또는 `write_signal()` 실패 시 `state → Disconnected`. 에러를 전파하지 않고 `Ok`를 반환한다 (비치명적).
- Engine: `MessageLoop`의 `transport.send()` 실패 시 메시지 루프 thread가 종료된다.

**[SEQ-083]** EOF (연결 끊김): PROTO-062 적용. SEQ-070~071의 끊김 감지 시퀀스로 진행한다. *(MUST)*

**[SEQ-084]** Rejected 응답 처리: `CommandResult::Rejected` 수신은 프로토콜 오류가 아니다 (PROTO-065). Manager는 로그에 기록한다. *(MUST)*

> **설계 의도** (non-normative): `docs/37_protocol_design.md` §9-2는 3회 연속 Rejected 시 해당 액션을 후보에서 제외하도록 권장한다. 현재 미구현.

**[SEQ-085]** Partial 응답 처리: `CommandResult::Partial { achieved, reason }` 수신. Manager는 `achieved` 값을 Relief 계산에 반영할 수 있다 (`22-manager-algorithms.md` 범위). *(MAY)*

**[SEQ-086]** 알 수 없는 메시지 타입: PROTO-064 적용. serde 역직렬화 실패 시 ParseError 경로 (SEQ-080)로 처리한다. *(MUST)*

**[SEQ-087]** Heartbeat 부재 시나리오: Manager는 특정 Heartbeat 주기를 가정하지 않는다 (PROTO-070). Heartbeat가 오래 도착하지 않아도 연결을 유지한다. *(MUST)*

> **설계 의도** (non-normative): `docs/37_protocol_design.md` §9-1은 3초 Heartbeat 타임아웃을 제안한다. 현재 미구현.

**[SEQ-088]** Directive 응답 부재 시나리오: Manager는 현재 Response 타임아웃을 적용하지 않는다. 다음 Directive는 새 seq_id로 독립 전송한다. *(MUST)*

> **설계 의도** (non-normative): `docs/37_protocol_design.md` §9-1은 500ms 응답 타임아웃을 제안한다. 현재 미구현.

### 3.9 Backpressure Sequences [SEQ-090 ~ SEQ-093]

**[SEQ-090]** Engine Heartbeat 과다: Manager의 inbox (`sync_channel(64)`)가 가득 차면 Reader thread의 `inbox_tx.send()`가 블로킹된다. Reader thread가 블로킹되면 `stream.read_exact()`가 호출되지 않아 Engine의 `transport.send()`도 블로킹된다. 이는 자연적 흐름 제어(natural backpressure)이다. *(MUST)*

**[SEQ-091]** Manager 메인 루프 drain: `try_recv` 반복으로 inbox의 모든 pending EngineMessage를 한 번에 소비한다. Heartbeat N개가 연속으로 수신되면 마지막 상태가 유효하다 (각 Heartbeat마다 `update_engine_state()`가 호출되어 FeatureVector를 덮어쓴다). *(MUST)*

**[SEQ-092]** Engine Directive drain: `CommandExecutor::poll()`에서 `cmd_rx.try_recv()` 반복으로 모든 pending Directive를 drain한다. 다중 Directive가 누적되면 순서대로 처리하며, **각 Directive에 대해 별도의 Response**를 전송한다. *(MUST)*

**[SEQ-093]** Monitor 신호 축적: Monitor 채널은 unbounded `mpsc::channel()`이다. Monitor thread가 블로킹되지 않는다. Manager 메인 루프는 `recv_timeout(50ms)`로 한 번에 하나의 신호만 처리한다. *(MUST)*

### 3.10 QCF Request Sequence [SEQ-095 ~ SEQ-098]

Critical 전환 시 Manager가 Engine에 QCF 비용을 요청하고, 응답을 받아 최적 액션 조합을 선택하는 시퀀스.

```
Manager                                    Engine
  │                                          │
  │  [Supervisory: Critical 전환 감지]        │
  │                                          │
  │── Directive(seq=N, [RequestQcf]) ───────►│
  │                                          │── QCF 계산 (읽기 전용 스캔)
  │◄── Response(seq=N, [Ok]) ───────────────│
  │◄── QcfEstimate(estimates) ──────────────│
  │                                          │
  │  [ActionSelector: estimates를 비용으로    │
  │   사용하여 최소 품질 저하 조합 선택]       │
  │                                          │
  │── Directive(seq=N+1, [actions...]) ─────►│
  │                                          │── 액션 실행
  │◄── Response(seq=N+1, [results...]) ─────│
  │                                          │
```

**[SEQ-095]** Manager는 Supervisory가 Critical 모드로 전환할 때 `RequestQcf`를 포함한 Directive를 전송한다. *(MUST)*

**[SEQ-096]** Engine은 RequestQcf Directive에 대해 먼저 CommandResponse(Ok)를 전송하고, 그 다음 QcfEstimate를 별도 EngineMessage로 전송한다. 순서: Response → QcfEstimate. *(MUST)*

**[SEQ-097]** Manager는 QcfEstimate를 수신한 후 ActionSelector를 실행한다. ActionSelector는 `estimates`의 값을 lossy 액션의 비용으로 사용한다. QcfEstimate에 포함되지 않은 lossy 액션은 ActionRegistry의 default_cost를 사용한다. *(MUST)*

**[SEQ-098]** QcfEstimate 수신 타임아웃: Manager는 Response 수신 후 일정 시간(SHOULD: 1초) 내에 QcfEstimate가 도착하지 않으면 default_cost로 fallback하여 ActionSelector를 실행한다. *(SHOULD)*

#### 대안 동작

| 조건 | 동작 |
|------|------|
| Engine 미연결 상태에서 Critical 전환 | RequestQcf 전송 불가. default_cost로 즉시 선택. |
| QcfEstimate의 estimates가 비어있음 | 모든 lossy 액션에 default_cost 적용. |
| Warning→Critical이 아닌 Normal→Critical 직행 | 동일하게 RequestQcf를 전송한다. |

### 3.11 D-Bus Sequence [SEQ-100 ~ SEQ-104]

**[SEQ-100]** D-Bus 연결: Engine은 `zbus::blocking::Connection::system()`으로 System Bus에 연결한다. `org.llm.Manager1` 경로에 대한 Proxy를 생성하고 `receive_all_signals()`로 시그널 수신을 시작한다. *(MUST)*

**[SEQ-101]** SystemSignal 수신: D-Bus signal 도착 시 member name을 확인한다. `"Directive"` member이면 시그널 본문의 JSON 문자열을 `ManagerMessage`로 직접 역직렬화한다 (네이티브 Directive 경로). 그 외 member는 SystemSignal 변환 경로로 처리한다. *(MUST)*

**[SEQ-102]** SystemSignal → EngineDirective 변환 매핑 테이블: *(MUST)*

| Signal | Level | EngineCommand(s) |
|--------|-------|-------------------|
| MemoryPressure | Normal | `RestoreDefaults` |
| MemoryPressure | Warning | `KvEvictSliding { keep_ratio: 0.85 }` |
| MemoryPressure | Critical | `KvEvictH2o { keep_ratio: 0.50 }` |
| MemoryPressure | Emergency | `KvEvictH2o { keep_ratio: 0.25 }` (SYS-055: Suspend 불필요, 공격적 eviction으로 대응) |
| ComputeGuidance | Normal | `RestoreDefaults` |
| ComputeGuidance | Warning | `Throttle { delay_ms: 30 }` + `SwitchHw { device }` |
| ComputeGuidance | Critical | `Throttle { delay_ms: 70 }` + `SwitchHw { device }` |
| ComputeGuidance | Emergency | `Suspend` |
| ThermalAlert | Normal | `RestoreDefaults` |
| ThermalAlert | Warning | `Throttle { delay_ms: 30 }` + `PrepareComputeUnit { device: "cpu" }` |
| ThermalAlert | Critical | `Throttle { delay_ms: 70 }` + `SwitchHw { device: "cpu" }` |
| ThermalAlert | Emergency | `Suspend` |
| EnergyConstraint | Normal | `RestoreDefaults` |
| EnergyConstraint | Warning | `SwitchHw { device: "cpu" }` |
| EnergyConstraint | Critical | `SwitchHw { device: "cpu" }` + `Throttle { delay_ms: 70 }` |
| EnergyConstraint | Emergency | `Suspend` |

> **참고**: ComputeGuidance의 `device`는 `recommended_backend` 필드에서 결정된다 (`Cpu` → `"cpu"`, `Gpu` → `"gpu"`, `Any` → `"any"`).

**[SEQ-103]** D-Bus 비대칭: Engine → Manager Response는 best-effort D-Bus signal (`"EngineMessage"` member로 JSON 직렬화하여 emit)이다. 프로토콜 보장이 없다. *(MUST)*

**[SEQ-104]** D-Bus에서의 INV-022 완화: D-Bus 경로에서는 Response 전달이 보장되지 않으므로, Manager는 Response를 기대하지 않는다. D-Bus Transport의 seq_id는 자체 카운터(`next_seq_id`, 초기값 1)로 관리된다. *(MUST)*

```
  D-Bus System Bus             Engine
       │                         │
       │  ── D-Bus signal ──────→│  member: "MemoryPressure"
       │                         │  body: (level, avail, total, reclaim)
       │                         │
       │                         │  parse_memory_pressure()
       │                         │  signal_to_manager_message()
       │                         │  → EngineDirective { seq, commands }
       │                         │
       │                         │  apply_command() × N
       │                         │  ExecutionPlan
       │                         │
       │  ◄── D-Bus signal ──────│  member: "EngineMessage"
       │  (best-effort)          │  body: (json_string)
       │                         │
```

## 4. Alternative Behavior

**Engine 독립 동작**: Manager 미연결 시 Engine은 모든 ResourceLevel을 Normal로 간주한다 (SYS-050). `poll()`에서 `cmd_rx`가 비어 있으므로 ExecutionPlan은 기본값 (no-op)이다.

**Manager 독립 동작**: Engine 미연결 시 Emitter 호출은 skip된다 (비치명적 `Ok` 반환). Policy 루프는 계속 실행되어 압력 계산, 모드 전이, 관찰 업데이트를 수행한다.

**다중 Directive 누적**: Engine이 느린 토큰 생성 (예: 대형 모델)으로 `poll()` 간격이 길면 여러 Directive가 `cmd_rx`에 누적될 수 있다. 순서대로 처리하며, 각각 별도의 Response를 전송한다.

**Heartbeat 수신 없는 Directive**: Manager가 Monitor 신호만으로 Directive를 생성할 수 있다 (engine_state가 기본값 상태에서). 정상은 아니나 프로토콜 위반이 아니다.

## 5. Constraints

**[CON-030]** 순서 보장: 동일 TCP/Unix 연결 내 메시지 순서는 보장된다 (STREAM 소켓). Frame 재배열이 발생하지 않는다. *(MUST)*

**[CON-031]** 단방향 순서: Manager→Engine과 Engine→Manager는 독립 스트림이다. 교차 순서 보장이 없다 (예: Directive A에 대한 Response와 Directive B의 상대적 순서는 보장되지 않는다). *(MUST)*

**[CON-032]** Capability 단일성: 세션당 정확히 1회 (INV-015). 2회 이상 전송 시 동작은 미정의이다. *(MUST)*

**[CON-033]** Response 필수성: 모든 Directive에 대해 정확히 1개 Response (INV-022). 누락 시 Manager는 해당 seq_id를 영원히 pending으로 유지한다 (현재 구현: 로깅만). *(MUST)*

## 6. Examples

### 6.1 정상 세션 trace

Capability → Heartbeat ×3 → Directive(Throttle) → Response(Ok) → Heartbeat ×3 → RestoreDefaults → Response(Ok)

```
t=0ms     Engine → Manager
{
  "type": "capability",
  "available_devices": ["cpu", "opencl"],
  "active_device": "opencl",
  "max_kv_tokens": 2048,
  "bytes_per_kv_token": 256,
  "num_layers": 16
}

t=1000ms  Engine → Manager
{
  "type": "heartbeat",
  "active_device": "opencl",
  "compute_level": "normal",
  "actual_throughput": 12.5,
  "memory_level": "normal",
  "kv_cache_bytes": 524288,
  "kv_cache_tokens": 256,
  "kv_cache_utilization": 0.125,
  "memory_lossless_min": 1.0,
  "memory_lossy_min": 0.01,
  "state": "running",
  "tokens_generated": 50,
  "available_actions": ["throttle", "switch_hw", "layer_skip"],
  "active_actions": [],
  "eviction_policy": "none",
  "kv_dtype": "f16",
  "skip_ratio": 0.0
}

t=2000ms  Engine → Manager
{
  "type": "heartbeat",
  "active_device": "opencl",
  "compute_level": "normal",
  "actual_throughput": 13.0,
  "memory_level": "normal",
  "kv_cache_bytes": 786432,
  "kv_cache_tokens": 384,
  "kv_cache_utilization": 0.1875,
  "memory_lossless_min": 1.0,
  "memory_lossy_min": 0.01,
  "state": "running",
  "tokens_generated": 100,
  "available_actions": ["throttle", "switch_hw", "layer_skip"],
  "active_actions": [],
  "eviction_policy": "none",
  "kv_dtype": "f16",
  "skip_ratio": 0.0
}

t=3000ms  Engine → Manager
{
  "type": "heartbeat",
  "active_device": "opencl",
  "compute_level": "normal",
  "actual_throughput": 13.2,
  "memory_level": "normal",
  "kv_cache_bytes": 1048576,
  "kv_cache_tokens": 512,
  "kv_cache_utilization": 0.25,
  "memory_lossless_min": 1.0,
  "memory_lossy_min": 0.01,
  "state": "running",
  "tokens_generated": 150,
  "available_actions": ["throttle", "switch_hw", "layer_skip"],
  "active_actions": [],
  "eviction_policy": "none",
  "kv_dtype": "f16",
  "skip_ratio": 0.0
}

t=3050ms  Manager → Engine  (Monitor: ComputeGuidance Warning 수신)
{
  "type": "directive",
  "seq_id": 1,
  "commands": [
    {"type": "throttle", "delay_ms": 30}
  ]
}

          Manager state: mode Normal → Warning

t=3100ms  Engine → Manager
{
  "type": "response",
  "seq_id": 1,
  "results": [{"status": "ok"}]
}

t=4000ms  Engine → Manager
{
  "type": "heartbeat",
  "active_device": "opencl",
  "compute_level": "normal",
  "actual_throughput": 10.0,
  "memory_level": "normal",
  "kv_cache_bytes": 1048576,
  "kv_cache_tokens": 530,
  "kv_cache_utilization": 0.259,
  "memory_lossless_min": 1.0,
  "memory_lossy_min": 0.01,
  "state": "running",
  "tokens_generated": 160,
  "available_actions": ["throttle", "switch_hw", "layer_skip"],
  "active_actions": ["throttle"],
  "eviction_policy": "none",
  "kv_dtype": "f16",
  "skip_ratio": 0.0
}

t=5000ms  Engine → Manager (Heartbeat)
t=6000ms  Engine → Manager (Heartbeat)

t=6050ms  Manager state: Observation complete (3s elapsed since t=3050ms)
          actual_relief = pressure_before − pressure_now
          estimator.observe(throttle, features, relief)

t=9050ms  Manager → Engine  (Monitor: ComputeGuidance Normal 수신)
{
  "type": "directive",
  "seq_id": 2,
  "commands": [
    {"type": "restore_defaults"}
  ]
}

          Manager state: mode Warning → Normal

t=9100ms  Engine → Manager
{
  "type": "response",
  "seq_id": 2,
  "results": [{"status": "ok"}]
}
```

### 6.2 압력 에스컬레이션 trace

Normal → Warning(Throttle) → Critical(RequestQcf → QcfEstimate → KvEvictH2o + Throttle) → Warning(RestoreDefaults) → Normal(RestoreDefaults)

> **시나리오**: ComputeGuidance 압력이 점진적으로 상승하여 Warning → Critical로 에스컬레이션된 후 해소되는 과정. Warning 단계에서는 compute 특화 lossless 액션(Throttle)을 선택한다. Critical 전환 시에는 먼저 RequestQcf로 Engine에 QCF 비용을 요청하고, QcfEstimate를 받은 뒤 cross-domain 액션(KvEvictH2o + Throttle)을 조합 선택한다 (SEQ-095~098 참조).

```
t=0ms     Initial state: mode=Normal, pressure=(0.0, 0.0, 0.0)
                                               (compute, memory, thermal)

t=1000ms  Monitor: ComputeGuidance Warning
          PI update → pressure.compute=0.55
          supervisory → Warning
          needs_action=true (mode_changed)
          ActionSelector → [Throttle{30}]

          Manager → Engine
          {
            "type": "directive", "seq_id": 1,
            "commands": [{"type": "throttle", "delay_ms": 30}]
          }

          Engine → Manager
          {
            "type": "response", "seq_id": 1,
            "results": [{"status": "ok"}]
          }

t=3000ms  Monitor: ComputeGuidance Critical + MemoryPressure Warning (복합 압력)
          PI update → pressure.compute=0.82, pressure.memory=0.48
          supervisory → Critical
          needs_action=true (mode_changed)

          ── QCF 요청 (SEQ-095~098) ──
          Manager → Engine
          {
            "type": "directive", "seq_id": 2,
            "commands": [{"type": "request_qcf"}]
          }

          Engine → Manager
          {"type": "response", "seq_id": 2, "results": [{"status": "ok"}]}

          Engine → Manager
          {
            "type": "qcf_estimate",
            "estimates": {"kv_evict_h2o": 0.12, "kv_merge_d2o": 0.08, "layer_skip": 0.35}
          }

          ActionSelector(estimates를 비용으로 사용)
            → [KvEvictH2o{0.50}, Throttle{50}]
            (cross-domain: memory 액션 + compute 액션 조합)

          Manager → Engine
          {
            "type": "directive", "seq_id": 3,
            "commands": [
              {"type": "kv_evict_h2o", "keep_ratio": 0.50},
              {"type": "throttle", "delay_ms": 50}
            ]
          }

          Engine → Manager
          {
            "type": "response", "seq_id": 3,
            "results": [{"status": "ok"}, {"status": "ok"}]
          }

t=8000ms  Monitor: ComputeGuidance Warning (압력 감소)
          PI update → pressure.compute=0.40, pressure.memory=0.20
          supervisory → Warning
          De-escalation: Critical → Warning
          build_lossy_release_directive()

          Manager → Engine
          {
            "type": "directive", "seq_id": 4,
            "commands": [{"type": "restore_defaults"}]
          }

          Engine → Manager
          {
            "type": "response", "seq_id": 4,
            "results": [{"status": "ok"}]
          }

t=9000ms  Next process_signal: mode=Warning, needs_action=true
          ActionSelector → [Throttle{30}] (lossless only)

          Manager → Engine
          {
            "type": "directive", "seq_id": 5,
            "commands": [{"type": "throttle", "delay_ms": 30}]
          }

t=15000ms Monitor: ComputeGuidance Normal (압력 해소)
          PI update → pressure.compute=0.12, pressure.memory=0.08
          supervisory → Normal
          De-escalation: Warning → Normal
          build_restore_directive()

          Manager → Engine
          {
            "type": "directive", "seq_id": 6,
            "commands": [{"type": "restore_defaults"}]
          }

          Engine → Manager
          {
            "type": "response", "seq_id": 6,
            "results": [{"status": "ok"}]
          }
```

### 6.3 재연결 trace

Connected → EOF → Disconnected → 새 Engine 연결 → Capability → 정상 흐름

```
t=0ms     State: Connected, mode=Warning, seq_id counter=8

t=100ms   Engine 프로세스 종료 → EOF
          Reader thread 종료
          Manager try_recv() → Disconnected 감지
          State: Disconnected

t=500ms   Monitor: MemoryPressure Warning
          process_signal() → Directive 생성 (seq_id=7)
          emit_directive() → ensure_connected() → accept() WouldBlock
          전송 skip (Ok 반환)
          Policy 상태 유지: mode=Warning, pressure 갱신됨

t=2000ms  새 Engine 프로세스 시작 → connect()
          ensure_connected() → accept() 성공
          State: Connected
          spawn Reader thread

t=2010ms  Engine → Manager
          {
            "type": "capability",
            "available_devices": ["cpu"],
            "active_device": "cpu",
            "max_kv_tokens": 1024,
            "bytes_per_kv_token": 128,
            "num_layers": 12
          }

          Manager: available_devices 갱신, Policy 상태(mode, pressure) 유지

t=3010ms  Engine → Manager (첫 Heartbeat)
          {
            "type": "heartbeat",
            "active_device": "cpu",
            ...
          }

t=3050ms  Monitor: MemoryPressure Warning
          process_signal() → Directive 생성 (seq_id=8, 이전 7에서 계속)

          Manager → Engine
          {
            "type": "directive", "seq_id": 8,
            "commands": [{"type": "throttle", "delay_ms": 30}]
          }
```

### 6.4 Rejected 처리 trace

Directive(KvStreaming) → Response(Rejected) → 다음 Directive에서 KvStreaming 제외

```
t=0ms     mode=Warning
          ActionSelector → [KvStreaming{sink_size: 4, window_size: 256}]

          Manager → Engine
          {
            "type": "directive", "seq_id": 10,
            "commands": [
              {"type": "kv_streaming", "sink_size": 4, "window_size": 256}
            ]
          }

t=50ms    Engine → Manager
          {
            "type": "response", "seq_id": 10,
            "results": [
              {
                "status": "rejected",
                "reason": "KvStreaming not yet implemented"
              }
            ]
          }

          Manager: log "Engine rejected KvStreaming: KvStreaming not yet implemented"

t=1000ms  다음 process_signal()
          needs_action=true (pressure 증가)
          ActionSelector → [Throttle{30}] (KvStreaming 회피는 22-manager-algorithms.md 범위)

          Manager → Engine
          {
            "type": "directive", "seq_id": 11,
            "commands": [{"type": "throttle", "delay_ms": 30}]
          }

          Engine → Manager
          {
            "type": "response", "seq_id": 11,
            "results": [{"status": "ok"}]
          }
```

## 7. Rationale (non-normative)

**왜 Engine 메시지를 먼저 drain하는가**: Manager 메인 루프에서 Engine 상태를 최신화한 후 Monitor 신호를 처리해야 정확한 압력 계산이 가능하다. 구 상태로 Directive를 생성하면 부적절한 액션 선택 위험이 있다. 코드 순서 (main.rs): `while try_recv_engine_message` → `recv_timeout(50ms)`.

**왜 3초 관찰 대기인가**: Monitor 샘플링 주기 ~1000ms, PI 안정화 + 실제 OS 수준 효과 반영 시간을 고려하면 최소 2~3 사이클이 필요하다. 너무 짧으면 noise가 학습되고, 너무 길면 반응이 지연된다. `OBSERVATION_DELAY_SECS = 3.0`은 Heartbeat 약 3개분의 시간이다.

**왜 RestoreDefaults로 일괄 해제인가**: 개별 액션 해제 명령 없이 단일 `RestoreDefaults`로 모든 활성 액션을 초기화한다. 간결성과 구현 단순성을 우선한다. 세분화된 per-action release는 `build_lossy_release_directive()` 메서드만 변경하면 향후 추가 가능하다.

**왜 docs/37의 타임아웃이 미구현인가**: 현재 시스템은 1:1 로컬 IPC로 메시지 전달 지연이 극히 낮아 타임아웃 없이도 안정적이다. Heartbeat 타임아웃(3초)과 Directive 응답 타임아웃(500ms)은 향후 네트워크 전송 또는 다중 Engine 지원 시 구현 필요하다.

**왜 D-Bus에서 INV-022가 완화되는가**: D-Bus는 fire-and-forget 특성이다. Engine → Manager D-Bus signal은 best-effort이므로 Response 전달을 보장할 수 없다. 프로토콜은 이를 수용하여 D-Bus 경로에서 Response 의무를 면제한다.

**왜 Monitor 채널은 unbounded인가**: Monitor thread는 OS 시스템 콜을 수행하므로 블로킹되면 샘플링 주기가 불규칙해진다. Unbounded 채널로 Monitor의 독립성을 보장한다. Manager 메인 루프는 `recv_timeout(50ms)`로 한 번에 하나씩 처리하여 Policy 계산 지연을 제한한다.

**왜 seq_id가 프로세스 수명인가**: `AtomicU64` static 카운터이므로 재연결 시에도 초기화되지 않는다. 이는 Manager가 여러 Engine 세션에 걸쳐 Directive를 추적할 수 있게 하며, seq_id 충돌을 방지한다.

**왜 Manager가 직접 액션을 선택하고 명령하는가**: 현재 구현은 Manager-selects-and-commands 모델로, Manager의 `ActionSelector`가 구체적 `EngineCommand`를 생성하여 Directive에 포함한다 (SEQ-044~046). 논문의 policy-design에서 제안하는 Budget/Spending 모델 — Manager는 resource budget만 전달하고 Engine이 자율적으로 액션을 선택 — 은 HR-1로 추적 중이다. Budget/Spending 모델이 채택되면 §3.4 시퀀스 (SEQ-044~046)의 Directive 구조와 §6 trace를 갱신해야 한다. 이 아키텍처 결정은 논문 기여 C2 (Cross-domain Action Selection)의 실행 위치에 직접 영향을 준다.
