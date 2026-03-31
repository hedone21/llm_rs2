# Test Tools

> **TL;DR**: mock_engine과 mock_manager는 Manager-Engine 프로토콜 적합성을 E2E로 검증하는 바이너리이다. mock_engine은 Engine 역할을 시뮬레이션하여 Manager의 Directive 전송/처리를 검증하고, mock_manager는 Manager 역할을 시뮬레이션하여 Engine의 프로토콜 구현을 검증한다. 두 도구는 `spec/12-protocol-sequences.md`의 SEQ-xxx 시퀀스를 구현해야 하며, `spec/41-invariants.md`의 INV-020~026 프로토콜 불변식을 준수해야 한다.

## 1. Purpose and Scope

이 문서는 프로토콜 E2E 검증을 위한 두 테스트 바이너리(mock_engine, mock_manager)의 요구사항을 정의한다.

**이 파일이 명세하는 것:**

- mock_engine이 구현해야 하는 프로토콜 시퀀스와 메시지 처리
- mock_manager가 구현해야 하는 프로토콜 시퀀스와 메시지 처리
- 두 도구의 CLI 인터페이스 요구사항
- 두 도구 간 상호운용 요구사항

**이 파일이 명세하지 않는 것:**

- 프로토콜 와이어 포맷, 메시지 필드 상세 --> `10-protocol.md`, `11-protocol-messages.md`
- 프로토콜 시퀀스 정의 --> `12-protocol-sequences.md`
- Production Engine/Manager 내부 알고리즘 --> `22-*`, `32-*`

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **mock_engine** | Manager에 연결하여 Engine 역할을 시뮬레이션하는 테스트 바이너리. `manager/src/bin/mock_engine.rs`. |
| **mock_manager** | Engine이 연결할 수 있는 Manager 역할을 시뮬레이션하는 테스트 바이너리. `manager/src/bin/mock_manager.rs`. |
| **프로토콜 적합성** | `spec/12-protocol-sequences.md`의 SEQ-xxx 시퀀스와 `spec/41-invariants.md`의 INV-020~026을 준수하는 것. |
| **시나리오 모드** | JSON 파일에 기술된 일련의 동작을 시간 순서대로 재생하는 운영 모드. |
| **인터랙티브 모드** | CLI 플래그로 단일 동작을 지정하는 운영 모드. |

## 3. Specification

### 3.1 mock_engine [TOOL-010 ~ TOOL-025]

#### 3.1.1 Handshake [TOOL-010 ~ TOOL-012]

**[TOOL-010]** mock_engine은 Unix 소켓으로 Manager에 연결한 후 `EngineMessage::Capability`를 세션의 첫 메시지로 전송해야 한다 (SEQ-022). Capability의 5개 필드(`available_devices`, `active_device`, `max_kv_tokens`, `bytes_per_kv_token`, `num_layers`)는 CLI 플래그 또는 기본값으로 구성한다. *(MUST)*

**[TOOL-011]** mock_engine은 Capability 전송 후 `heartbeat_interval` 경과 시 첫 Heartbeat를 전송해야 한다 (SEQ-024). *(MUST)*

**[TOOL-012]** mock_engine은 Capability를 세션당 정확히 1회 전송해야 한다 (INV-015). *(MUST)*

#### 3.1.2 Steady-State [TOOL-013 ~ TOOL-015]

**[TOOL-013]** mock_engine은 `heartbeat_interval` (CLI `--heartbeat-ms`, 기본 100ms) 주기로 `EngineMessage::Heartbeat(EngineStatus)`를 전송해야 한다 (SEQ-030). *(MUST)*

**[TOOL-014]** EngineStatus의 16개 필드는 mock_engine의 내부 상태를 반영해야 한다. 특히: *(MUST)*

- `active_device`: 현재 활성 디바이스 (SwitchHw 반영)
- `kv_cache_utilization`: 현재 KV 점유율 (eviction 반영)
- `state`: Running/Suspended (Suspend/Resume 반영)
- `tokens_generated`: 누적 토큰 수
- `skip_ratio`: 현재 레이어 스킵 비율 (LayerSkip 반영)
- `eviction_policy`: 현재 eviction 정책 (eviction command 반영)
- `active_actions`: 현재 활성 중인 액션 목록 (RestoreDefaults 시 초기화)
- `available_actions`: 사용 가능한 액션 목록

**[TOOL-015]** mock_engine은 Directive가 없는 동안에도 Heartbeat를 계속 전송해야 한다 (SEQ-035). *(MUST)*

#### 3.1.3 Command Processing [TOOL-016 ~ TOOL-019]

**[TOOL-016]** mock_engine은 EngineCommand 13종을 모두 처리해야 한다. 각 command에 대해 내부 상태를 갱신하고 적절한 `CommandResult`를 반환한다. *(MUST)*

| EngineCommand | 상태 변경 | 기본 결과 |
|---------------|----------|----------|
| KvEvictSliding { keep_ratio } | kv_occupancy *= keep_ratio (clamp 0.01~1.0), eviction_policy = "sliding" | Ok |
| KvEvictH2o { keep_ratio } | kv_occupancy *= keep_ratio (clamp 0.01~1.0), eviction_policy = "h2o" | Ok |
| KvStreaming { sink_size, window_size } | eviction_policy = "streaming" | Ok |
| KvMergeD2o { keep_ratio } | kv_occupancy *= keep_ratio (clamp 0.01~1.0), eviction_policy = "d2o" | Ok |
| KvQuantDynamic { target_bits } | (로그만 출력) | Ok |
| Throttle { delay_ms } | throttle_delay_ms = delay_ms | Ok |
| LayerSkip { skip_ratio } | skip_ratio = skip_ratio | Ok |
| SwitchHw { device } | active_device = device | Ok |
| PrepareComputeUnit { device } | (no-op, 로그만 출력) | Ok |
| RestoreDefaults | throttle=0, skip_ratio=0.0, eviction_policy="none", active_actions=[] | Ok |
| Suspend | state = Suspended | Ok |
| Resume | state = Running | Ok |
| RequestQcf | (QcfEstimate 전송 트리거) | Ok |

**[TOOL-017]** mock_engine은 각 Directive에 대해 정확히 1개 `CommandResponse`를 전송해야 한다 (INV-022). *(MUST)*

**[TOOL-018]** CommandResponse의 `seq_id`는 수신한 Directive의 `seq_id`와 일치해야 한다 (INV-023). `results` 배열의 길이는 Directive의 `commands` 배열 길이와 같아야 한다 (INV-024). *(MUST)*

**[TOOL-019]** mock_engine은 `RequestQcf` command를 수신하면, CommandResponse(Ok) 전송 후 별도의 `EngineMessage::QcfEstimate`를 전송해야 한다. 순서: Response --> QcfEstimate (SEQ-096). QcfEstimate의 `estimates`는 모의 값(각 lossy action에 대해 고정 비용)을 포함한다. *(MUST)*

#### 3.1.4 CLI Interface [TOOL-020 ~ TOOL-022]

**[TOOL-020]** mock_engine은 다음 CLI 플래그를 지원해야 한다: *(MUST)*

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--socket` | String | `/tmp/llm_manager.sock` | Manager Unix 소켓 경로 |
| `--heartbeat-ms` | u64 | 100 | Heartbeat 전송 주기 (ms) |
| `--kv-occupancy` | f32 | 0.5 | 초기 KV 캐시 점유율 (0.0~1.0) |
| `--device` | String | `opencl` | 초기 활성 디바이스 |
| `--duration-secs` | u64 | 30 | 실행 시간 (초) |

**[TOOL-021]** mock_engine은 세션 종료 시 Summary를 stdout에 출력해야 한다: Heartbeat 전송 횟수, Directive 수신 횟수, 최종 상태. *(SHOULD)*

**[TOOL-022]** mock_engine은 수신한 Directive와 적용 결과를 stdout에 로그 출력해야 한다. *(SHOULD)*

#### 3.1.5 Wire Format [TOOL-023]

**[TOOL-023]** mock_engine은 `10-protocol.md` PROTO-010에 정의된 length-prefixed JSON 와이어 포맷(4-byte BE u32 length + UTF-8 JSON payload)을 사용해야 한다. *(MUST)*

### 3.2 mock_manager [TOOL-030 ~ TOOL-048]

#### 3.2.1 Unix Socket Server [TOOL-030 ~ TOOL-034]

**[TOOL-030]** mock_manager는 Unix 소켓 경로에 `bind()` + `listen()`하여 Engine 연결을 대기해야 한다 (SEQ-020). *(MUST)*

**[TOOL-031]** mock_manager는 Engine 연결을 `accept()`하고 Reader thread를 시작하여 EngineMessage를 비동기로 수신해야 한다 (SEQ-021). *(MUST)*

**[TOOL-032]** mock_manager는 Engine으로부터 `EngineMessage::Capability`를 수신하고, `available_devices`, `max_kv_tokens`, `bytes_per_kv_token`, `num_layers`를 캐시해야 한다 (SEQ-023). Capability 수신 전까지 Directive를 전송하지 않아야 한다. *(MUST)*

**[TOOL-033]** mock_manager는 Engine으로부터 `EngineMessage::Heartbeat`를 수신하고, EngineStatus의 주요 필드를 stdout에 출력해야 한다. *(MUST)*

**[TOOL-034]** mock_manager는 Engine으로부터 `EngineMessage::Response`를 수신하고, `seq_id`와 `results`를 stdout에 출력해야 한다. Response의 `seq_id`가 전송한 Directive의 `seq_id`와 일치하는지 검증해야 한다 (INV-023). *(MUST)*

#### 3.2.2 Directive Transmission [TOOL-035 ~ TOOL-039]

**[TOOL-035]** mock_manager는 CLI/시나리오에서 지정한 EngineCommand를 `EngineDirective`로 구성하여 Engine에 전송해야 한다 (SEQ-045, SEQ-046). *(MUST)*

**[TOOL-036]** mock_manager는 `seq_id`를 1부터 시작하여 단조 증가시켜야 한다 (INV-020, INV-021). 동일 `seq_id`를 재사용하지 않는다. *(MUST)*

**[TOOL-037]** mock_manager는 Response를 수신하면 `results` 배열의 길이가 전송한 `commands` 배열의 길이와 같은지 검증해야 한다 (INV-024). 불일치 시 경고를 출력한다. *(MUST)*

**[TOOL-038]** mock_manager는 `EngineMessage::QcfEstimate` 수신을 지원해야 한다 (SEQ-096, SEQ-097). QcfEstimate 수신 시 `estimates`를 stdout에 출력한다. *(MUST)*

**[TOOL-039]** mock_manager는 `10-protocol.md` PROTO-010에 정의된 length-prefixed JSON 와이어 포맷을 사용해야 한다. *(MUST)*

#### 3.2.3 CLI Interface [TOOL-040 ~ TOOL-044]

**[TOOL-040]** mock_manager는 다음 CLI 플래그를 지원해야 한다: *(MUST)*

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--socket` | String | `/tmp/llm_manager.sock` | Unix 소켓 Listen 경로 |
| `--timeout-secs` | u64 | 60 | Engine 연결 대기 타임아웃 (초) |

**[TOOL-041]** mock_manager는 인터랙티브 모드를 지원해야 한다: CLI 플래그로 단일 command를 지정하여 Directive로 전송한다. *(MUST)*

| 플래그 | 설명 |
|--------|------|
| `--command` | EngineCommand 타입 (e.g., `kv_evict_sliding`, `throttle`, `request_qcf`) |
| `--keep-ratio` | eviction/merge 계열 command의 keep_ratio 파라미터 |
| `--delay-ms` | Throttle command의 delay_ms 파라미터 |
| `--skip-ratio` | LayerSkip command의 skip_ratio 파라미터 |
| `--device` | SwitchHw/PrepareComputeUnit의 device 파라미터 |
| `--target-bits` | KvQuantDynamic의 target_bits 파라미터 |
| `--sink-size` | KvStreaming의 sink_size 파라미터 |
| `--window-size` | KvStreaming의 window_size 파라미터 |

**[TOOL-042]** mock_manager는 시나리오 모드를 지원해야 한다: JSON 파일에 기술된 command 시퀀스를 순차 실행한다. *(SHOULD)*

**[TOOL-043]** 시나리오 JSON 형식: *(SHOULD)*

```json
{
  "name": "escalation_test",
  "description": "Optional description",
  "steps": [
    {
      "delay_ms": 2000,
      "commands": [
        {"type": "throttle", "delay_ms": 50}
      ]
    },
    {
      "delay_ms": 1000,
      "commands": [
        {"type": "request_qcf"}
      ]
    }
  ]
}
```

각 step은 `delay_ms` 대기 후 `commands`를 단일 Directive로 전송한다. Response를 수신하고 결과를 출력한다.

**[TOOL-044]** mock_manager의 인터랙티브 모드에서 `--command` 없이 실행하면 Capability + Heartbeat를 수신하고 상태를 출력한 후 종료한다 (연결 검증 모드). *(SHOULD)*

#### 3.2.4 D-Bus 경로 (기존) [TOOL-045 ~ TOOL-046]

**[TOOL-045]** mock_manager는 기존 D-Bus 시그널 발행 기능을 유지해야 한다. `--dbus` 플래그로 D-Bus 모드를 활성화한다. D-Bus 모드와 Unix 모드는 상호 배타적이다. *(SHOULD)*

**[TOOL-046]** D-Bus 모드에서 mock_manager는 4종 시그널(`MemoryPressure`, `ComputeGuidance`, `ThermalAlert`, `EnergyConstraint`)을 발행하는 기존 기능을 유지한다. 기존 CLI 플래그(`--signal`, `--level`, `--scenario` 등)는 D-Bus 모드에서만 유효하다. *(SHOULD)*

> **Rationale**: D-Bus 경로는 Engine의 DbusListener 테스트에 여전히 필요하다. Unix 경로는 양방향 프로토콜 E2E 검증에 필요하다. 두 경로의 테스트 목적이 다르므로 모드 분리로 공존시킨다.

#### 3.2.5 Wire Format [TOOL-047]

**[TOOL-047]** mock_manager는 Unix 모드에서 ManagerMessage를 length-prefixed JSON으로 전송하고, EngineMessage를 length-prefixed JSON으로 수신해야 한다. 프레이밍은 PROTO-010과 동일하다. *(MUST)*

#### 3.2.6 Validation Output [TOOL-048]

**[TOOL-048]** mock_manager는 프로토콜 불변식 위반을 감지하면 stderr에 경고를 출력해야 한다. 감지 대상: *(SHOULD)*

- INV-023 위반: Response.seq_id != 전송 Directive.seq_id
- INV-024 위반: len(results) != len(commands)
- INV-015 위반: Capability가 0회 또는 2회 이상 수신됨

### 3.3 Inter-tool Requirements [TOOL-050 ~ TOOL-052]

**[TOOL-050]** mock_engine과 mock_manager(Unix 모드)는 직접 연결하여 양방향 프로토콜 E2E를 검증할 수 있어야 한다. mock_manager가 Listen하고 mock_engine이 연결한다. *(MUST)*

**[TOOL-051]** mock_engine과 mock_manager 간 메시지 교환은 production Manager/Engine과 동일한 와이어 포맷을 사용해야 한다. 도구 전용 메시지나 확장은 없다. *(MUST)*

**[TOOL-052]** mock_engine은 production Manager에, mock_manager(Unix 모드)는 production Engine에도 연결하여 사용할 수 있어야 한다. 도구와 production 컴포넌트 간 호환성을 유지한다. *(MUST)*

## 4. Constraints

**[TOOL-060]** mock_engine과 mock_manager는 `llm_manager` 크레이트의 바이너리로 빌드된다. `llm_shared` 크레이트에만 의존한다 (mock_manager Unix 모드). D-Bus 모드는 추가로 `zbus` 크레이트에 의존한다. *(MUST)*

**[TOOL-061]** mock_engine과 mock_manager는 테스트 도구이므로 production 코드 경로에 영향을 주지 않는다. `src/bin/` 디렉토리의 별도 바이너리로 분리한다. *(MUST)*

## 5. Rationale (non-normative)

**왜 별도 spec 파일인가**: 테스트 도구는 production 컴포넌트가 아니지만, 프로토콜 적합성 검증의 핵심 수단이다. spec에서 관리하면 프로토콜 변경 시 도구 갱신이 추적 가능하다. production 스펙(10-12)과 분리하여 관심사 혼합을 방지한다.

**왜 TOOL prefix인가**: `TEST-xxx`는 테스트 케이스(assertion)와 혼동될 수 있다. `TOOL-xxx`는 테스트 "도구(바이너리)"의 요구사항임이 명확하다.

**왜 D-Bus와 Unix 모드를 공존시키는가**: D-Bus 경로는 Engine의 `DbusListener` 컴포넌트 테스트에 필요하고, Unix 경로는 양방향 프로토콜(Capability/Heartbeat/Directive/Response/QcfEstimate) E2E 검증에 필요하다. 용도가 다르므로 하나로 대체할 수 없다.

**왜 mock_engine에 QcfEstimate가 필요한가**: production Engine은 RequestQcf 수신 시 Response(Ok) 후 별도 QcfEstimate를 전송한다 (SEQ-096). mock_engine이 이를 구현하지 않으면 Manager의 QCF 요청 시퀀스(SEQ-095~098)를 E2E 테스트할 수 없다.

**왜 mock_manager에 seq_id 검증이 필요한가**: INV-020~024는 프로토콜의 핵심 불변식이다. mock_manager가 이를 검증하면 production Engine의 프로토콜 적합성을 도구 수준에서 즉시 확인할 수 있다.
