# Protocol Sequences -- Architecture

> spec/12-protocol-sequences.md의 구현 상세.

## 코드 매핑

### 3.1 Session Lifecycle Phases [SEQ-010 ~ SEQ-013]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SEQ-010 | `manager/src/channel/unix_socket.rs` | `wait_for_client(timeout)` + `accept()` | Phase 1: Handshake |
| SEQ-010 | `engine/src/resilience/transport.rs` | `Transport::connect()` | |
| SEQ-011 | `manager/src/main.rs` | 메인 루프 — Monitor signal + Engine message drain | Phase 2: Steady-State |
| SEQ-012 | `manager/src/pipeline.rs` | `process_signal()` — PI + Supervisory + Selector | Phase 3: Pressure Response |
| SEQ-013 | `manager/src/channel/unix_socket.rs` | EOF → Disconnected 전이 | Phase 4: Termination |

### 3.2 Connection & Handshake [SEQ-020 ~ SEQ-025]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SEQ-020 | `manager/src/channel/unix_socket.rs` | `wait_for_client(timeout)` — 기본 60초 | CLI `--client-timeout` |
| SEQ-021 | `manager/src/channel/unix_socket.rs` | `accept()` → Connected, Reader thread spawn | |
| SEQ-022 | `engine/src/resilience/executor.rs` | 연결 직후 `send_capability()` 호출 | 세션 첫 메시지 |
| SEQ-023 | `manager/src/pipeline.rs` | Capability → `available_devices` 등 캐시 | |
| SEQ-024 | `engine/src/resilience/executor.rs:156` | `if self.last_heartbeat.elapsed() >= self.heartbeat_interval` | 경과 시간 기반 |
| SEQ-025 | `manager/src/pipeline.rs` | `update_engine_state()` → FeatureVector 초기화 | |

### 3.3 Steady-State Monitoring [SEQ-030 ~ SEQ-035]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SEQ-030 | `engine/src/resilience/executor.rs:93,156` | `heartbeat_interval: Duration` + 경과 확인 | `poll()` 내 |
| SEQ-031 | `manager/src/channel/unix_socket.rs:152` | `sync_channel(64)` inbox | Reader thread → Main |
| SEQ-031 | `manager/src/main.rs` | `try_recv` 반복으로 drain | |
| SEQ-032 | `manager/src/main.rs:193` | `recv_timeout(Duration::from_millis(50))` | Monitor 채널 |
| SEQ-033 | `manager/src/main.rs` | Engine drain 먼저 → Monitor signal 처리 | 코드 순서 |
| SEQ-034 | `engine/src/resilience/executor.rs` | `poll()` — 토큰 생성당 1회 | |
| SEQ-035 | `manager/src/main.rs:193` | `recv_timeout(50ms)` → timeout 시 `continue` | |

### 3.4 Pressure Escalation [SEQ-040 ~ SEQ-049]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SEQ-040 | `manager/src/pipeline.rs` | `process_signal(signal)` | 메인 진입점 |
| SEQ-041 | `manager/src/pipeline.rs` | `update_pressure(signal)` → 도메인별 PI/직접 매핑 | |
| SEQ-041 | `manager/src/pi_controller.rs` | `PiController::update()` | Compute, Thermal |
| SEQ-042 | `manager/src/supervisory.rs` | `supervisory.evaluate(&pressure)` → OperatingMode | |
| SEQ-043 | `manager/src/pipeline.rs` | `needs_action` 조건 — mode_changed OR pressure exceeds | |
| SEQ-044 | `manager/src/selector.rs` | `ActionSelector::select()` | cross-domain 탐색 |
| SEQ-045 | `manager/src/pipeline.rs:65-68` | `next_seq_id()` → `SEQ_COUNTER.fetch_add(1, Relaxed)` | |
| SEQ-046 | `manager/src/channel/unix_socket.rs` | `emit_directive()` → `write_manager_message()` | |
| SEQ-047 | `engine/src/resilience/transport.rs` | MessageLoop thread → `cmd_tx.send()` | |
| SEQ-047 | `engine/src/resilience/executor.rs` | `poll()` → `cmd_rx.try_recv()` | |
| SEQ-048 | `engine/src/resilience/executor.rs` | `apply_command(cmd, plan)` × N | |
| SEQ-049 | `engine/src/resilience/executor.rs` | Directive당 1개 CommandResponse → `resp_tx` | INV-022~024 |

### 3.5 Observation & Relief [SEQ-050 ~ SEQ-054]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SEQ-050 | `manager/src/pipeline.rs` | `ObservationContext` 저장 | Directive 생성 직후 |
| SEQ-051 | `manager/src/pipeline.rs` | `OBSERVATION_DELAY_SECS = 3.0` | 하드코딩 상수 |
| SEQ-052 | `manager/src/pipeline.rs` | `update_observation()` — `actual_relief = before − now` | |
| SEQ-053 | `manager/src/relief/` | `estimator.observe(action, feature_vec, actual_relief)` | 온라인 학습 |

### 3.6 De-escalation [SEQ-060 ~ SEQ-064]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SEQ-060 | `manager/src/supervisory.rs` | `evaluate()` — 이전 모드 > 현재 모드 | |
| SEQ-061 | `manager/src/pipeline.rs` | `build_lossy_release_directive()` → RestoreDefaults | |
| SEQ-062 | `manager/src/pipeline.rs` | `build_restore_directive()` → RestoreDefaults | |

### 3.7 Reconnection [SEQ-070 ~ SEQ-075]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SEQ-070 | `manager/src/channel/unix_socket.rs` | Reader EOF → `inbox` Disconnected → state 전이 | |
| SEQ-071 | `engine/src/resilience/transport.rs` | MessageLoop `recv()` → Disconnected → thread 종료 | |
| SEQ-072 | `manager/src/channel/unix_socket.rs` | `ensure_connected()` — non-blocking accept | |
| SEQ-073 | `manager/src/channel/unix_socket.rs` | accept 성공 → Connected, 새 Reader spawn | |
| SEQ-074 | `engine/src/resilience/executor.rs` | 재연결 시 새 Capability 전송 | |
| SEQ-075 | `manager/src/pipeline.rs:65` | `static SEQ_COUNTER` — 프로세스 수명, 리셋 없음 | |

### 3.8 Error Sequences [SEQ-080 ~ SEQ-088]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SEQ-080 | `engine/src/resilience/transport.rs` | MessageLoop — warn 로그 + `continue` | |
| SEQ-081 | `engine/src/resilience/transport.rs:267` | `MAX_PAYLOAD_SIZE` 가드 | Manager측 미구현 |
| SEQ-082 | `manager/src/channel/unix_socket.rs` | write 실패 → Disconnected, 에러 미전파 | |
| SEQ-087 | `manager/src/main.rs` | Heartbeat 타임아웃 **미구현** | docs/37 제안: 3초 |
| SEQ-088 | `manager/src/pipeline.rs` | Response 타임아웃 **미구현** | docs/37 제안: 500ms |

### 3.9 Backpressure [SEQ-090 ~ SEQ-093]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SEQ-090 | `manager/src/channel/unix_socket.rs:152` | `sync_channel(64)` — 가득 차면 Reader 블로킹 | 자연적 배압 |
| SEQ-091 | `manager/src/main.rs` | `try_recv` 반복 drain | 마지막 Heartbeat 유효 |
| SEQ-092 | `engine/src/resilience/executor.rs` | `poll()` — `try_recv` 반복으로 다중 Directive drain | 각각 별도 Response |
| SEQ-093 | `manager/src/main.rs` | Monitor 채널: unbounded `mpsc::channel()` | Monitor 블로킹 방지 |

### 3.10 QCF Request Sequence [SEQ-095 ~ SEQ-098]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SEQ-095 | `manager/src/pipeline.rs` | Critical 전환 시 `RequestQcf` Directive 전송 | |
| SEQ-096 | `engine/src/resilience/executor.rs` | RequestQcf → Response(Ok) → QcfEstimate 전송 | |
| SEQ-097 | `manager/src/selector.rs` | QcfEstimate → lossy 액션 비용으로 사용 | |
| SEQ-098 | `manager/src/pipeline.rs` | QcfEstimate 타임아웃 **미구현** | spec SHOULD: 1초 |

### 3.11 D-Bus Sequence [SEQ-100 ~ SEQ-104]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SEQ-100 | `engine/src/resilience/dbus_transport.rs` | `zbus::blocking::Connection::system()` | `org.llm.Manager1` |
| SEQ-101 | `engine/src/resilience/dbus_transport.rs` | `"Directive"` member → JSON → ManagerMessage | 네이티브 경로 |
| SEQ-102 | `engine/src/resilience/dbus_transport.rs` | `signal_to_manager_message()` — Level별 변환 테이블 | |
| SEQ-103 | `engine/src/resilience/dbus_transport.rs` | Engine → Manager: best-effort D-Bus signal | 보장 없음 |
| SEQ-104 | `engine/src/resilience/dbus_transport.rs:20` | `next_seq_id: u64` (초기값 1) — 자체 카운터 | INV-022 완화 |

### Constraints [CON-030 ~ CON-033]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| CON-030 | STREAM 소켓 특성 | 프레임 순서 보장 | |
| CON-032 | `engine/src/resilience/executor.rs` | Capability 1회 전송 (세션당) | INV-015 |
| CON-033 | `engine/src/resilience/executor.rs` | Directive당 1 Response | INV-022 |

## Config

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| (heartbeat_interval은 하드코딩) | `Duration` | `1000ms` | SEQ-030 |
| (OBSERVATION_DELAY_SECS는 하드코딩) | `f64` | `3.0` | SEQ-051 |
| (sync_channel 용량은 하드코딩) | `usize` | `64` | SEQ-090 |
| (recv_timeout은 하드코딩) | `Duration` | `50ms` | SEQ-032 |
| (client_timeout은 CLI 파라미터) | `u64` (초) | `60` | SEQ-020 |

## CLI

| 플래그 | 설명 | spec/ 근거 |
|--------|------|-----------|
| `--client-timeout` (Manager) | Unix socket 클라이언트 대기 타임아웃(초) | SEQ-020 |
| `--enable-resilience` (Engine) | Resilience 서브시스템 활성화 | SEQ-010 (Handshake 전제) |
| `--resilience-transport` (Engine) | 전송 매체 선택 | SEQ-100 (D-Bus 경로) |

## 미구현 사항 (spec 대비)

| spec/ ID | 내용 | 현재 상태 |
|----------|------|----------|
| SEQ-081 | Manager측 64KB 페이로드 크기 가드 | 미구현 (Engine측만 구현) |
| SEQ-087 | Heartbeat 타임아웃 (docs/37 제안: 3초) | 미구현 |
| SEQ-088 | Directive Response 타임아웃 (docs/37 제안: 500ms) | 미구현 |
| SEQ-098 | QcfEstimate 수신 타임아웃 (spec SHOULD: 1초) | 미구현 |
| MSG-014 | QcfEstimate 구조체 (shared/src/lib.rs) | 미정의 |