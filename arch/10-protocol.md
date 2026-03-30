# Wire Protocol -- Architecture

> spec/10-protocol.md의 구현 상세.

## 코드 매핑

### 3.1 Wire Format [PROTO-010 ~ PROTO-014]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| PROTO-010 | `engine/src/resilience/transport.rs` | `write_frame()`, `read_frame()` | 4B BE u32 + UTF-8 JSON |
| PROTO-010 | `manager/src/channel/unix_socket.rs` | `write_manager_message()`, `read_engine_message()` | 동일 프레이밍 |
| PROTO-010 | `manager/src/channel/tcp.rs` | `write_manager_message()`, `read_engine_message()` | 동일 프레이밍 |
| PROTO-012 | `engine/src/resilience/transport.rs:245` | `const MAX_PAYLOAD_SIZE: u32 = 64 * 1024` | Engine측 64KB 가드 |
| PROTO-012 | `engine/src/resilience/transport.rs:267` | `if len > MAX_PAYLOAD_SIZE { ... ParseError }` | |
| PROTO-012 | (Manager측) | **미구현** — `read_engine_message()`에 크기 가드 없음 | spec에서 향후 추가 권장 |
| INV-020 | `manager/src/pipeline.rs:65-68` | `SEQ_COUNTER: AtomicU64::new(1)` + `fetch_add(1, Relaxed)` | |
| INV-021 | `manager/src/pipeline.rs:65-68` | AtomicU64 단조 증가로 보장 | |

### 3.2 Serialization Convention [PROTO-020 ~ PROTO-026]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| PROTO-020 | `engine/src/resilience/transport.rs` | `serde_json::to_vec()` — compact JSON | |
| PROTO-021 | `shared/src/lib.rs` | `#[serde(tag = "type", rename_all = "snake_case")]` 등 | 타입별 serde 어노테이션 |
| PROTO-022 | `shared/src/lib.rs` | snake_case 필드명 그대로 JSON 키 | rename 미사용 |
| PROTO-026 | `shared/src/lib.rs` | `#[serde(default)]` — EngineStatus 필드 12~16 등 | |

#### serde 어노테이션 상세 — `shared/src/lib.rs`

| 타입 | 라인 | serde 어노테이션 |
|------|------|-----------------|
| `Level` | 7 | `#[serde(rename_all = "snake_case")]` |
| `RecommendedBackend` | 17 | `#[serde(rename_all = "snake_case")]` |
| `ComputeReason` | 26 | `#[serde(rename_all = "snake_case")]` |
| `EnergyReason` | 38 | `#[serde(rename_all = "snake_case")]` |
| `SystemSignal` | 106 | `#[serde(rename_all = "snake_case")]` — externally tagged (serde 기본) |
| `ResourceLevel` | 151 | `#[serde(rename_all = "snake_case")]` |
| `EngineState` | 160 | `#[serde(rename_all = "snake_case")]` |
| `EngineCommand` | 170 | `#[serde(tag = "type", rename_all = "snake_case")]` |
| `EngineDirective` | 205 | (struct, internally tagged via ManagerMessage) |
| `ManagerMessage` | 213 | `#[serde(tag = "type", rename_all = "snake_case")]` |
| `EngineCapability` | 219 | (struct) |
| `EngineStatus` | 229 | (struct, `#[serde(default)]` on 필드 12~16) |
| `CommandResult` | 261 | `#[serde(tag = "status", rename_all = "snake_case")]` |
| `CommandResponse` | 269 | (struct) |
| `EngineMessage` | 277 | `#[serde(tag = "type", rename_all = "snake_case")]` |

### 3.3 Transport Layer [PROTO-030 ~ PROTO-036]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| PROTO-030 | `engine/src/resilience/transport.rs` | `UnixSocketTransport` — Unix Stream | Engine 클라이언트 |
| PROTO-030 | `manager/src/channel/unix_socket.rs` | `UnixSocketChannel` — bind + listen | Manager 서버 |
| PROTO-031 | `engine/src/resilience/transport.rs` | `TcpTransport` | |
| PROTO-031 | `manager/src/channel/tcp.rs` | `TcpChannel` | |
| PROTO-032 | `engine/src/resilience/dbus_transport.rs` | `DbusTransport` — `zbus::blocking::Connection::system()` | |
| PROTO-032 | `manager/src/emitter/dbus.rs` | `DbusEmitter` | D-Bus 시그널 emit |
| PROTO-036 | `engine/src/resilience/transport.rs` | `MockTransport` — mpsc 채널 | 테스트 전용 |

#### Transport 파일 경로 전체

| 측 | 전송 매체 | 파일 |
|----|----------|------|
| Engine | Unix Socket | `engine/src/resilience/transport.rs` (UnixSocketTransport) |
| Engine | TCP | `engine/src/resilience/transport.rs` (TcpTransport) |
| Engine | D-Bus | `engine/src/resilience/dbus_transport.rs` (DbusTransport) |
| Engine | Mock | `engine/src/resilience/transport.rs` (MockTransport) |
| Manager | Unix Socket | `manager/src/channel/unix_socket.rs` (UnixSocketChannel) |
| Manager | TCP | `manager/src/channel/tcp.rs` (TcpChannel) |
| Manager | D-Bus | `manager/src/emitter/dbus.rs` (DbusEmitter) |

### 3.4 Connection Lifecycle [PROTO-040 ~ PROTO-046]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| PROTO-040 | `manager/src/channel/unix_socket.rs` | `bind()` + `listen()` | Manager가 서버 |
| PROTO-040 | `engine/src/resilience/transport.rs` | `connect()` | Engine이 클라이언트 |
| PROTO-042 | `manager/src/channel/unix_socket.rs` | `state` 필드: Listening/Connected/Disconnected | 3-state |
| PROTO-043 | `engine/src/resilience/transport.rs` | `Transport::connect()` → `ConnectionFailed` | |
| PROTO-044 | `engine/src/resilience/executor.rs` | 연결 직후 Capability 전송 | |
| PROTO-045 | `manager/src/channel/unix_socket.rs` | `ensure_connected()` — non-blocking accept | |

### 3.5 Message Flow Direction [PROTO-050 ~ PROTO-052]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| PROTO-050 | `shared/src/lib.rs:213` | `ManagerMessage::Directive(EngineDirective)` | M→E |
| PROTO-051 | `shared/src/lib.rs:277` | `EngineMessage` — 4종 변형 | E→M |
| PROTO-052 | `engine/src/resilience/dbus_transport.rs` | SystemSignal 4종 수신 | D-Bus 경로 |

### 3.6 Error Handling [PROTO-060 ~ PROTO-065]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| PROTO-060 | `engine/src/resilience/transport.rs:267` | `MAX_PAYLOAD_SIZE` 초과 시 `ParseError` | Engine측만 구현 |
| PROTO-061 | `engine/src/resilience/transport.rs` | MessageLoop 내 `continue` | |
| PROTO-062 | `engine/src/resilience/transport.rs` | EOF → `TransportError::Disconnected` | |

### 3.7 Timing and Backpressure [PROTO-070 ~ PROTO-075]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| PROTO-070 | `engine/src/bin/generate.rs:670` | `heartbeat_interval = Duration::from_millis(1000)` | 기본 1000ms |
| PROTO-071 | `manager/src/channel/unix_socket.rs:152` | `sync_channel(64)` | |
| PROTO-071 | `manager/src/channel/tcp.rs:144` | `sync_channel(64)` | |
| PROTO-072 | `manager/src/main.rs:193` | `recv_timeout(Duration::from_millis(50))` | |
| PROTO-073 | `engine/src/resilience/executor.rs` | `poll()` 내 `try_recv` 반복 | |
| PROTO-074 | `manager/src/pipeline.rs:65-68` | `static SEQ_COUNTER: AtomicU64::new(1)`, `fetch_add(1, Relaxed)` | 프로세스 수명 |
| PROTO-074 | `engine/src/resilience/dbus_transport.rs:20` | `next_seq_id: u64` (초기값 1, 자체 카운터) | D-Bus 경로 전용 |
| PROTO-075 | `engine/src/resilience/executor.rs` | Directive당 1개 CommandResponse 전송 | INV-022 |
| INV-022 | `engine/src/resilience/executor.rs` | `poll()` 내 각 Directive → 1 Response | |
| INV-023 | `engine/src/resilience/executor.rs` | `CommandResponse.seq_id = directive.seq_id` | |
| INV-024 | `engine/src/resilience/executor.rs` | `results.len() == commands.len()` | |

## Config

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| (heartbeat_interval은 현재 하드코딩) | `Duration` | `1000ms` | PROTO-070 |
| (MAX_PAYLOAD_SIZE는 상수) | `u32` | `64 * 1024` | PROTO-012 |
| (sync_channel 용량은 하드코딩) | `usize` | `64` | PROTO-071 |
| (recv_timeout은 하드코딩) | `Duration` | `50ms` | PROTO-072 |

## CLI

| 플래그 | 설명 | spec/ 근거 |
|--------|------|-----------|
| `--resilience-transport` (Engine) | `dbus`, `unix:<path>`, `tcp:<host:port>` (기본: `dbus`) | PROTO-033 |
| `--transport` (Manager) | `dbus`, `unix:<path>`, `tcp:<host:port>` (기본: `dbus`) | PROTO-033 |
| `--client-timeout` (Manager) | Unix socket 클라이언트 대기 타임아웃(초) (기본: 60) | PROTO-040 |