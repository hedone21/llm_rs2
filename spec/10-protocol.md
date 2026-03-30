# Wire Protocol

> **TL;DR**: Manager ↔ Engine 간 IPC 와이어 프로토콜을 정의한다. 4바이트 Big-Endian 길이 접두사 + UTF-8 JSON 페이로드 프레이밍, serde internally-tagged enum 직렬화 규칙, Unix Socket / TCP / D-Bus 전송 계층, 3-state 연결 수명주기, 오류 처리, 타이밍·백프레셔 제약을 명세한다. 개별 메시지 필드 정의는 `11-protocol-messages.md`, 정규 상호작용 시퀀스는 `12-protocol-sequences.md`에서 다룬다.

## 1. Purpose and Scope

이 문서는 Manager ↔ Engine 간 와이어 프로토콜의 전송 계층, 프레이밍, 직렬화 규칙, 연결 관리를 정의한다.

**이 파일이 명세하는 것:**

- 프레임 구조 (Wire Format)
- JSON 직렬화 규칙 (Serialization Convention)
- 전송 계층 구현체 (Transport Layer)
- 연결 수명주기와 상태 머신 (Connection Lifecycle)
- 메시지 흐름 방향 (Message Flow Direction)
- 프로토콜 수준 오류 처리 (Error Handling)
- 타이밍 제약과 흐름 제어 (Timing and Backpressure)

**이 파일이 명세하지 않는 것:**

- 개별 메시지의 필드 정의 → `11-protocol-messages.md`
- 정규 상호작용 시퀀스 → `12-protocol-sequences.md`
- Manager/Engine 내부 처리 로직 → `22-manager-algorithms.md`, `32-engine-algorithms.md`

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **Frame** | 하나의 length-prefixed JSON 메시지 단위. 길이 접두사 4바이트 + 페이로드로 구성된다. |
| **Envelope** | 최상위 메시지 래퍼. Manager→Engine 방향은 `ManagerMessage`, Engine→Manager 방향은 `EngineMessage`. |
| **Payload** | Frame 내 UTF-8 JSON 바이트 시퀀스. |
| **Session** | 하나의 TCP/Unix 연결 수명. 연결 수립부터 종료까지를 가리킨다. |
| **Transport** | 바이트 스트림을 제공하는 전송 계층 구현체. |
| **Channel** | Manager 측 양방향 통신 추상. Emitter(송신) + EngineReceiver(수신)를 조합한다. |

## 3. Specification

### 3.1 Wire Format [PROTO-010 ~ PROTO-014]

**[PROTO-010]** 모든 메시지는 다음 Frame 구조로 전송된다: *(MUST)*

```
+-------------------+-----------------------------+
| Length (4 bytes)  |   Payload (N bytes)         |
| Big-Endian u32    |   UTF-8 JSON                |
+-------------------+-----------------------------+
```

**[PROTO-011]** Length 필드는 Payload 바이트 수를 나타낸다. Length 필드 자체(4바이트)는 포함하지 않는다. *(MUST)*

**[PROTO-012]** 최대 페이로드 크기는 **65,536 바이트 (64 KB)**이다. 수신측은 이 크기를 초과하는 프레임을 ParseError로 거부해야 한다. *(SHOULD)*

> **참고 (non-normative)**: Engine Transport(`engine/src/resilience/transport.rs`)에서 `MAX_PAYLOAD_SIZE = 64 * 1024` 가드가 구현되어 있다. Manager 측 Channel(`unix_socket.rs`, `tcp.rs`)의 `read_engine_message()`에는 현재 이 가드가 미적용되어 있다. 향후 Manager 측에도 동일 가드를 추가하는 것을 권장한다.

**[PROTO-013]** Payload는 UTF-8 인코딩이다. BOM(Byte Order Mark)을 포함해서는 안 된다. *(MUST NOT)*

**[PROTO-014]** Length 필드는 Big-Endian 바이트 순서이다. *(MUST)*

#### 의사코드: write_frame

```
function write_frame(writer, message):
    json_bytes ← serialize_to_json(message)    // UTF-8 compact JSON
    length ← byte_count(json_bytes)            // payload 바이트 수
    writer.write(to_big_endian_u32(length))    // 4바이트 BE 길이
    writer.write(json_bytes)                   // JSON 페이로드
    writer.flush()
```

#### 의사코드: read_frame

```
function read_frame(reader):
    len_buf ← reader.read_exact(4 bytes)
    length ← from_big_endian_u32(len_buf)

    if length > 65536:
        raise ParseError("payload too large")

    payload ← reader.read_exact(length bytes)
    message ← deserialize_from_json(payload)   // UTF-8 → 메시지 타입
    return message
```

#### Example: Heartbeat Frame 바이트 덤프

Heartbeat 메시지 (간략화):

```json
{"type":"heartbeat","active_device":"cpu","compute_level":"normal","actual_throughput":15.0,"memory_level":"normal","kv_cache_bytes":1048576,"kv_cache_tokens":512,"kv_cache_utilization":0.25,"memory_lossless_min":1.0,"memory_lossy_min":0.01,"state":"running","tokens_generated":100}
```

위 JSON이 243바이트라고 가정하면 (실제 크기는 필드 값에 따라 변동):

```
Offset  Hex                                         ASCII
0000    00 00 00 F3                                  ....         (length = 243)
0004    7B 22 74 79 70 65 22 3A 22 68 65 61 72 74 .. {"type":"heart..
```

### 3.2 Serialization Convention [PROTO-020 ~ PROTO-026]

**[PROTO-020]** JSON 직렬화는 serde_json compact 형식을 사용한다. Pretty print(줄바꿈/들여쓰기)를 사용해서는 안 된다. *(MUST NOT)*

**[PROTO-021]** Envelope 및 주요 enum 타입은 **internally tagged** 직렬화를 사용한다. 태그 필드와 변형별 값은 다음 테이블을 따른다: *(MUST)*

| 타입 | 태그 속성 | 태그 필드 | rename_all | 변형 (tag value) |
|------|----------|----------|------------|-----------------|
| EngineCommand | `tag = "type"` | `"type"` | `snake_case` | `"throttle"`, `"layer_skip"`, `"kv_evict_h2o"`, `"kv_evict_sliding"`, `"kv_merge_d2o"`, `"kv_streaming"`, `"kv_quant_dynamic"`, `"request_qcf"`, `"restore_defaults"`, `"switch_hw"`, `"prepare_compute_unit"`, `"suspend"`, `"resume"` |
| ManagerMessage | `tag = "type"` | `"type"` | `snake_case` | `"directive"` |
| EngineMessage | `tag = "type"` | `"type"` | `snake_case` | `"capability"`, `"heartbeat"`, `"response"`, `"qcf_estimate"` |
| CommandResult | `tag = "status"` | `"status"` | `snake_case` | `"ok"`, `"partial"`, `"rejected"` |
| SystemSignal | (없음) | — | `snake_case` | **externally tagged** (serde 기본). D-Bus 전송 전용. |

> **Internally tagged**: 태그 필드가 JSON 객체 내부에 포함된다. 나머지 필드는 동일 레벨에 flat merge된다.
>
> **Externally tagged** (SystemSignal): 변형 이름이 JSON 객체의 최외곽 키가 된다. 예: `{"memory_pressure": {...}}`.
>
> **EngineCommand 가용 변형**: 위 테이블의 EngineCommand 변형 13종은 프로토콜 수준에서 정의된 전체 집합이다. 이 중 리소스 관리 액션(`throttle`, `layer_skip`, `kv_evict_*`, `kv_streaming`, `kv_quant_dynamic`, `switch_hw`)의 런타임 유효 여부는 Engine이 Heartbeat의 `available_actions` 필드로 보고하는 가용 목록에 의해 결정된다. Engine은 현재 상태(eviction 정책 설정, KV 캐시 데이터 타입 등)에 기반하여 가용 액션을 동적으로 계산하고 매 Heartbeat마다 보고한다 (`11-protocol-messages.md` EngineStatus 참조). Manager의 Action Selector는 Engine이 보고한 `available_actions`에 포함된 액션만 Directive에 포함해야 한다 *(SHOULD)*. 제어 명령(`restore_defaults`, `prepare_compute_unit`, `suspend`, `resume`, `request_qcf`)은 `available_actions`와 무관하게 전송할 수 있다.

**[PROTO-022]** 구조체 필드명은 snake_case 그대로 JSON 키로 사용된다. serde rename이 없는 struct의 필드명이 곧 와이어 이름이다. *(MUST)*

**[PROTO-023]** 정수(u8, u32, u64, usize, i32)는 JSON number로 직렬화된다. 따옴표로 감싸지 않는다. 부동소수점(f32, f64)도 JSON number로 직렬화된다. *(MUST)*

**[PROTO-024]** 문자열(String)은 JSON string으로 직렬화된다. *(MUST)*

**[PROTO-025]** 배열(Vec<T>)은 JSON array로 직렬화된다. *(MUST)*

**[PROTO-026]** `#[serde(default)]`가 적용된 필드는 JSON에서 생략 가능하다. 생략 시 타입별 기본값이 적용된다: *(MAY)*

| 타입 | 기본값 |
|------|--------|
| `Vec<T>` | 빈 배열 `[]` |
| `String` | 빈 문자열 `""` |
| `f32` | `0.0` |

#### Enum rename_all 적용 대상

다음 열거형 타입에 `rename_all = "snake_case"`가 적용된다:

| 열거형 | 변형 수 | 사용처 |
|--------|--------|--------|
| Level | 4 | SystemSignal (D-Bus) |
| ResourceLevel | 3 | EngineStatus |
| EngineState | 3 | EngineStatus |
| RecommendedBackend | 3 | SystemSignal.ComputeGuidance |
| ComputeReason | 6 | SystemSignal.ComputeGuidance |
| EnergyReason | 6 | SystemSignal.EnergyConstraint |
| EngineCommand | 13 | EngineDirective.commands |
| ManagerMessage | 1 | Manager → Engine envelope |
| EngineMessage | 4 | Engine → Manager envelope |
| CommandResult | 3 | CommandResponse.results |
| SystemSignal | 4 | D-Bus 전송 경로 |

### 3.3 Transport Layer [PROTO-030 ~ PROTO-036]

**[PROTO-030]** 기본 전송 매체는 **Unix Domain Socket** (STREAM 타입)이다. 소켓 경로는 설정 가능하다. *(MUST)*

**[PROTO-031]** 대체 전송 매체로 **TCP loopback**을 지원한다. Android SELinux 환경에서 Unix socket `bind()`가 제한될 때 사용한다. 주소 형식은 `host:port` (예: `"127.0.0.1:9200"`). *(MAY)*

**[PROTO-032]** 추가 전송 매체로 **D-Bus System Bus**를 지원한다. *(MAY)*

- Well-known name: `org.llm.Manager1`
- Object path: `/org/llm/Manager1`
- Interface: `org.llm.Manager1`
- 방향: Manager → Engine (SystemSignal 4종 + native Directive). Engine → Manager는 best-effort D-Bus signal (프로토콜 보장 없음).
- 활성화: Engine CLI `--dbus` 플래그.
- D-Bus Transport는 수신한 SystemSignal을 내부적으로 ManagerMessage(EngineDirective)로 변환한다.
- D-Bus Transport는 SystemSignal 외에 `"Directive"` D-Bus 시그널도 수신할 수 있다. 이 경우 시그널 본문의 JSON 문자열을 ManagerMessage로 직접 역직렬화한다 (SystemSignal 변환 경로를 거치지 않음).

**[PROTO-033]** 전송 매체 선택은 CLI 플래그로 결정된다: *(MUST)*

| 측 | 플래그 | 값 형식 | 예시 |
|----|--------|---------|------|
| Engine | `--resilience-transport` | `unix:<path>`, `tcp:<host:port>`, `dbus` | `unix:/tmp/llm.sock` |
| Manager | `--transport` | `unix`, `tcp`, `dbus` | `--transport unix` |

**[PROTO-034]** Unix Socket과 TCP는 양방향이다 (ManagerMessage ↔ EngineMessage). D-Bus는 비대칭이다: Manager → Engine은 SystemSignal 4종 및 native Directive, Engine → Manager는 best-effort D-Bus signal (프로토콜 보장 없음). *(MUST)*

**[PROTO-035]** 와이어 포맷은 전송 매체에 무관하다. Unix Socket과 TCP 모두 동일한 length-prefixed JSON 프레이밍(PROTO-010)을 사용한다. *(MUST)*

**[PROTO-036]** MockTransport는 테스트용 mpsc 채널 기반 구현이다. 프로덕션 와이어 포맷을 사용하지 않는다. *(non-normative)*

### 3.4 Connection Lifecycle [PROTO-040 ~ PROTO-046]

**[PROTO-040]** Manager가 서버 역할(bind + listen)을 하고, Engine이 클라이언트 역할(connect)을 한다. *(MUST)*

**[PROTO-041]** 연결 모델은 **1:1 단일 클라이언트**이다. 동시에 여러 Engine이 연결할 수 없다. *(MUST)*

**[PROTO-042]** Manager 측 연결 상태는 3-state 머신으로 모델링된다: *(MUST)*

#### 상태 전이 다이어그램

```
                  ┌──────────────┐
                  │  Listening   │ ◄── 초기 상태 (bind + listen)
                  └──────┬───────┘
                         │ accept()
                         ▼
                  ┌──────────────┐
         ┌──────►│  Connected   │
         │       └──────┬───────┘
         │              │ write error / reader EOF / inbox disconnected
         │              ▼
         │       ┌──────────────┐
         └───────┤ Disconnected │
  ensure_connected()  └──────────────┘
  (non-blocking accept)
```

#### 상태 전이 테이블

| 현재 상태 | 이벤트 | 다음 상태 | 조건 |
|-----------|--------|-----------|------|
| Listening | `accept()` 성공 | Connected | `wait_for_client()` (초기 연결 대기) |
| Connected | Writer 쓰기 오류 | Disconnected | `emit()` / `emit_directive()` 내 write 실패 |
| Connected | Reader thread EOF | Disconnected | `try_recv()`에서 inbox `Disconnected` 감지 |
| Disconnected | `ensure_connected()` → `accept()` 성공 | Connected | 다음 `emit()` 호출 시 non-blocking accept 시도 |

**[PROTO-043]** Engine 측은 `Transport::connect()`를 호출하여 연결한다. 연결 실패 시 `ConnectionFailed` 에러를 반환한다. *(MUST)*

**[PROTO-044]** 연결 직후 Engine은 `EngineMessage::Capability`를 전송해야 한다. Manager는 Capability 수신 전까지 Directive를 전송하지 않아야 한다. *(Engine: MUST, Manager: SHOULD)*

**[PROTO-045]** Manager가 Disconnected 상태에서 다음 `emit()` 호출 시 non-blocking `accept()`를 시도한다. 재연결 시 Engine은 새 Capability를 전송해야 한다. *(MUST)*

**[PROTO-046]** 연결 종료는 양측 모두 소켓 close로 수행한다. Manager는 Drop 시 Unix 소켓 파일을 삭제한다. *(MUST)*

### 3.5 Message Flow Direction [PROTO-050 ~ PROTO-052]

**[PROTO-050]** Manager → Engine 방향은 `ManagerMessage` envelope를 사용한다. 현재 1종 변형: *(MUST)*

| 변형 | 설명 |
|------|------|
| Directive(EngineDirective) | 명령 배치 전송 |

**[PROTO-051]** Engine → Manager 방향은 `EngineMessage` envelope를 사용한다. 현재 4종 변형: *(MUST)*

| 변형 | 설명 |
|------|------|
| Capability(EngineCapability) | 세션당 1회 능력 보고 |
| Heartbeat(EngineStatus) | 주기적 상태 보고 |
| Response(CommandResponse) | Directive 실행 응답 |
| QcfEstimate(QcfEstimate) | Critical 전환 시 QCF 비용 응답 (SYS-012) |

**[PROTO-052]** D-Bus 전송 경로에서는 Manager → Engine 방향으로 `SystemSignal` 4종을 전달한다. Engine → Manager 방향 메시지는 best-effort D-Bus signal로 전송한다. *(MAY)*

### 3.6 Error Handling [PROTO-060 ~ PROTO-065]

**[PROTO-060]** 페이로드 크기 초과 — 수신측은 64KB 초과 프레임을 `ParseError`로 거부한다. 연결은 유지한다. *(SHOULD)*

**[PROTO-061]** JSON 파싱 실패 — 수신측은 ParseError를 로그 기록 후 해당 프레임을 skip한다. Engine Transport의 `MessageLoop`에서 `continue`로 다음 프레임을 읽는다. 연결은 유지한다. *(MUST)*

**[PROTO-062]** 연결 끊김 감지 — EOF 수신 시 `TransportError::Disconnected`를 발생시킨다. *(MUST)*

- **Manager**: Reader thread 종료 → inbox `Disconnected` → state 전이 (Connected → Disconnected).
- **Engine**: `MessageLoop` 종료.

**[PROTO-063]** 쓰기 오류 — *(MUST)*

- **Manager**: state를 Disconnected로 전이한다. 에러를 호출자에게 전파하지 않는다 (비치명적).
- **Engine**: `MessageLoop`가 종료된다.

**[PROTO-064]** 알 수 없는 메시지 타입 — serde 역직렬화 실패로 ParseError가 발생한다. PROTO-061과 동일한 경로로 처리된다. *(MUST)*

**[PROTO-065]** `CommandResult` 수준의 Rejected/Partial은 **프로토콜 오류가 아닌 비즈니스 응답**이다. 연결에 영향을 주지 않는다. *(non-normative)*

### 3.7 Timing and Backpressure [PROTO-070 ~ PROTO-075]

**[PROTO-070]** Heartbeat 주기는 Engine 설정 가능하다. 현재 기본값은 **1000ms**이다. Manager는 특정 주기를 가정하지 않는다. *(SHOULD)*

**[PROTO-071]** Manager 수신 버퍼는 `sync_channel(64)`이다. Reader thread가 최대 64개 `EngineMessage`를 버퍼링한다. 버퍼가 가득 차면 reader thread가 블로킹되어 자연적 흐름 제어가 발생한다. *(MUST)*

**[PROTO-072]** Manager 메인 루프는 `recv_timeout(50ms)` 간격으로 Engine 메시지를 확인한다. *(SHOULD)*

**[PROTO-073]** Engine의 `CommandExecutor::poll()`은 `try_recv` 반복으로 모든 pending `ManagerMessage`를 한 번에 드레인한다. *(MUST)*

**[PROTO-074]** seq_id 생성: `AtomicU64`, 초기값 1, `fetch_add(1, Relaxed)`. 세션 간 리셋하지 않는다 (프로세스 수명). *(MUST)*

#### 불변식

- **[INV-020]** seq_id는 단조 증가한다: `seq_id(N+1) > seq_id(N)`. *(MUST)*
- **[INV-021]** 동일 seq_id를 재사용해서는 안 된다. *(MUST NOT)*

**[PROTO-075]** 각 `EngineDirective`에 대해 Engine은 정확히 1개의 `CommandResponse`를 전송한다. `CommandResponse.seq_id`는 대응 `EngineDirective.seq_id`와 일치해야 한다. *(MUST)*

#### 불변식

- **[INV-022]** 모든 Directive는 정확히 1개의 Response를 유발한다. *(MUST)*
- **[INV-023]** `CommandResponse.seq_id == EngineDirective.seq_id`. *(MUST)*
- **[INV-024]** `CommandResponse.results` 배열 길이 == `EngineDirective.commands` 배열 길이. *(MUST)*

## 4. Alternative Behavior

- **D-Bus 전송 (PROTO-032)**: 양방향 프로토콜(ManagerMessage ↔ EngineMessage)을 완전히 지원하지 않는다. SystemSignal 4종만 Manager → Engine으로 전달하며, D-Bus Transport가 내부적으로 EngineDirective로 변환한다. Engine → Manager는 best-effort D-Bus signal.

- **Engine 독립 동작**: Manager 미연결 시 Engine은 모든 ResourceLevel을 Normal로 간주한다 (Fail-Safe, SYS-050 참조). 추론은 중단 없이 계속된다.

- **Manager 독립 동작**: Engine 미연결 시 Manager는 Emitter 호출을 skip한다 (비치명적). Policy 루프는 계속 실행되어 모니터링 데이터를 수집한다.

## 5. Constraints

- **[CON-010]** No async: 모든 I/O는 blocking 또는 non-blocking polling이다. tokio, async-std 등 비동기 런타임을 사용하지 않는다 (SYS-064 참조). *(MUST NOT)*

- **[CON-011]** JSON 전용: 바이너리 직렬화(bincode, protobuf 등)를 사용하지 않는다. serde_json만 허용한다 (SYS-065 참조). *(MUST NOT)*

- **[CON-012]** 단일 연결: 1 Manager : 1 Engine. 다중 Engine 연결을 지원하지 않는다 (SYS-093 참조). *(MUST NOT)*

## 6. Examples

### 6.1 정상 프레임 바이트 덤프 — Heartbeat

```json
{"type":"heartbeat","active_device":"cpu","compute_level":"normal","actual_throughput":15.0,"memory_level":"normal","kv_cache_bytes":1048576,"kv_cache_tokens":512,"kv_cache_utilization":0.25,"memory_lossless_min":1.0,"memory_lossy_min":0.01,"state":"running","tokens_generated":100,"available_actions":["throttle","switch_hw","layer_skip"],"active_actions":["throttle"],"eviction_policy":"none","kv_dtype":"f16","skip_ratio":0.0}
```

### 6.2 EngineCommand Internally Tagged 예시

**Throttle:**
```json
{"type":"throttle","delay_ms":50}
```

**KvEvictH2o:**
```json
{"type":"kv_evict_h2o","keep_ratio":0.48}
```

**SwitchHw:**
```json
{"type":"switch_hw","device":"opencl"}
```

**RestoreDefaults:**
```json
{"type":"restore_defaults"}
```

**Suspend:**
```json
{"type":"suspend"}
```

### 6.3 연결 상태 전이 Trace

| Step | Manager 상태 | 이벤트 | 비고 |
|------|-------------|--------|------|
| 1 | Listening | `bind()` + `listen()` 완료 | 소켓 파일 생성됨 |
| 2 | Listening → Connected | Engine `connect()` → Manager `accept()` | Reader thread 시작 |
| 3 | Connected | Engine → Capability 전송 | Manager 수신 |
| 4 | Connected | Manager → Directive(seq=1) 전송 | Engine 수신·처리·Response 전송 |
| 5 | Connected → Disconnected | Engine 프로세스 종료 → EOF | Reader thread 종료 |
| 6 | Disconnected | Manager `emit()` 호출 → `ensure_connected()` | non-blocking accept 시도 |
| 7 | Disconnected → Connected | 새 Engine `connect()` → Manager `accept()` | 새 Reader thread 시작 |
| 8 | Connected | 새 Engine → Capability 전송 | 새 세션 시작 |

## 7. Rationale (non-normative)

### 왜 length-prefixed인가

단순성과 구현 용이성. No-async 환경에서 프레임 경계가 명확하다. 구분자 기반 프레이밍(newline-delimited JSON 등)은 페이로드 내 구분자를 이스케이프해야 하는 복잡성이 있다.

### 왜 JSON인가

디버깅 용이성. `socat`이나 `nc`로 와이어 메시지를 직접 읽을 수 있다. 타입 안전성은 serde에 위임한다. 메시지 빈도가 1~10 msg/sec 수준이므로 직렬화 오버헤드가 병목이 아니다.

### 왜 internally tagged enum인가

serde의 기본 지원 태그 방식 중 하나로, 수신측에서 `"type"` 필드 값으로 빠르게 분기할 수 있다. Adjacently tagged(`{"type":"...", "content":{...}}`)보다 JSON이 평탄하여 크기가 작다.

### 왜 64KB 제한인가

일반적인 메시지 크기: Heartbeat ~500B, Directive ~200B, Capability ~150B. 64KB는 충분한 여유를 제공하면서 OOM을 방지한다. 악의적이거나 오작동하는 피어로부터의 메모리 고갈을 방어하는 최소 가드이다.

### 왜 Manager가 서버인가

Manager는 장기 실행 서비스(시스템 부팅 시 시작)이고, Engine은 추론 세션별로 시작·종료된다. Manager bind → Engine connect 패턴이 자연스럽다. Engine 재시작 시 Manager 소켓에 재연결하면 된다.

### 왜 sync_channel(64)인가

Backpressure 메커니즘. Engine이 과도한 heartbeat를 전송해도 reader thread가 block되어 자연적 흐름 제어가 발생한다. 무제한 버퍼(`mpsc::channel`)는 메모리 누수 위험이 있다.
