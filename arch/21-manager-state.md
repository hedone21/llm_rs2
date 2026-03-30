# Manager State Machines -- Architecture

> spec/21-manager-state.md의 구현 상세.

## 코드 매핑

### 3.1 OperatingMode State Machine

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-050 | `manager/src/types.rs:75-80` | `enum OperatingMode { Normal, Warning, Critical }` | `PartialOrd`/`Ord` derive |
| MGR-051 | `manager/src/supervisory.rs` | `SupervisoryLayer::evaluate()` → `next_mode()` | 에스컬레이션 즉시 전이 |
| MGR-052 | `manager/src/supervisory.rs` | `try_de_escalate()` | hold_time 후 1단계 하강 |
| MGR-053 | `manager/src/supervisory.rs` | `stable_since: Option<Instant>` 필드 | 디에스컬레이션 타이머 |
| MGR-054 | `manager/src/config.rs:234-244` | `SupervisoryConfig::default()` | `warning_threshold=0.4`, `critical_threshold=0.7`, `hold_time_secs=4.0` 등 |
| MGR-055 | `manager/src/supervisory.rs` | `evaluate()` 내부 match 분기 | 전이 테이블 완전 구현 |
| INV-032 | `manager/src/supervisory.rs` | `peak >= critical_threshold` → Critical 직행 | Normal→Critical 가능 |
| INV-033 | `manager/src/supervisory.rs` | `try_de_escalate()` → `current_mode`에서 1단계만 하강 | Critical→Normal 직행 불가 |

### 3.2 ConnectionState State Machine

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-060 | `manager/src/channel/unix_socket.rs:37-44` | `enum ConnectionState { Listening, Connected, Disconnected }` | 초기: Listening |
| MGR-061 | `manager/src/channel/unix_socket.rs` | `emit_directive()`, `ensure_connected()` 내부 상태 전이 | 전이 테이블 구현 |
| MGR-062 | `manager/src/channel/unix_socket.rs` | `wait_for_client(timeout, shutdown)` | 블로킹 대기, timeout/shutdown 시 반환 |
| MGR-063 | `manager/src/channel/unix_socket.rs` | `ensure_connected()` — non-blocking accept | Disconnected → Connected 재연결 |
| MGR-064 | `manager/src/channel/unix_socket.rs:14-29` | `ReaderHandle` RAII struct | `Drop` 시 handle take (join 안 함) |
| MGR-065 | `manager/src/channel/unix_socket.rs` | `mpsc::sync_channel(64)` | SyncSender 용량 64 |
| MGR-066 | `manager/src/channel/tcp.rs` | `TcpChannel` — `TcpConnectionState` | 동일 3-state 패턴 |

### 3.3 ThresholdEvaluator State Machine

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-067 | `manager/src/evaluator.rs` | `ThresholdEvaluator.current: Level` | 4-level 상태 (Level enum from llm_shared) |
| MGR-068 | `manager/src/evaluator.rs:4-9` | `enum Direction { Ascending, Descending }` | |
| MGR-069 | `manager/src/evaluator.rs:12-19` | `struct Thresholds { warning, critical, emergency, hysteresis }` | f64 타입 |
| MGR-070 | `manager/src/evaluator.rs` | `evaluate()` 메서드 — Ascending 에스컬레이션 | 단계 건너뛰기 가능 |
| MGR-071 | `manager/src/evaluator.rs` | `evaluate()` 메서드 — recovery threshold 적용 | `threshold - hysteresis` |
| MGR-072 | `manager/src/evaluator.rs` | Descending 방향 — 부등호 반전 구현 | `threshold + hysteresis` |
| MGR-073 | `manager/src/evaluator.rs` | `evaluate()` 전체 전이 테이블 구현 | Ascending/Descending 양방향 |

### Monitor별 기본 ThresholdEvaluator 파라미터

| Monitor | 코드 위치 | 기본값 |
|---------|----------|--------|
| Memory | `manager/src/config.rs:55-66` | Descending: w=40%, c=20%, e=10%, hyst=5% |
| Thermal | `manager/src/config.rs:83-95` | Ascending: w=60000mc, c=75000mc, e=85000mc, hyst=5000mc |
| Compute | `manager/src/config.rs:110-119` | Ascending: w=70%, c=90%, emergency=f64::MAX (비활성), hyst=5% |
| Energy | `manager/src/config.rs:139-153` | Descending: w=30%, c=15%, e=5%. Hysteresis=2.0 **하드코딩** |

### 3.4 PolicyPipeline Internal State

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-074 | `manager/src/pipeline.rs:75-104` | `HierarchicalPolicy` struct 필드 | pressure, prev_mode, pending_observation 등 |
| MGR-075 | `manager/src/pipeline.rs:301-407` | `PolicyStrategy::process_signal()` 구현 | 7단계 순차 처리 |
| MGR-076 | `manager/src/pipeline.rs:312-320` | `needs_action` 판정 | Normal→false, 그 외→mode_changed OR 1.2x |
| MGR-077 | `manager/src/pipeline.rs:232-253,358-364` | `update_observation()`, `ObservationContext` 생성 | OBSERVATION_DELAY_SECS=3.0 |
| MGR-078 | `manager/src/pipeline.rs:383-403` | de-escalation 분기 | `result.is_none()` 확인 후 생성 |
| MGR-079 | `manager/src/pipeline.rs:365` | `self.last_acted_pressure = self.pressure` | Directive 전송 시 갱신 |
| MGR-080 | `manager/src/pipeline.rs:413-465` | `update_engine_state()` | FeatureVector 13차원 매핑 |
| MGR-081 | `manager/src/pipeline.rs:455-464` | `ActionId::from_str()` filter_map | 인식 불가 문자열 skip |
| MGR-082 | `manager/src/pipeline.rs:177-186` | `elapsed_dt()` 메서드 | `last_signal_time` HashMap, clamp [0.001, 10.0] |

### 3.5 ReliefEstimator Lifecycle

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-083 | `manager/src/relief/linear.rs` | `models: HashMap<ActionId, LinearModel>` | 액션별 독립 모델 |
| MGR-084 | `manager/src/relief/linear.rs` | `predict()` — `observation_count == 0` 시 default_relief 반환 | Absent/Initialized → default |
| MGR-085 | `manager/src/relief/linear.rs` | `observe()` — `ensure_model()` → RLS update | Absent → Learning (count=1) |
| MGR-086 | `manager/src/relief/linear.rs` | `default_relief()` 함수 | 하드코딩 prior 테이블 |
| MGR-087 | `manager/src/relief/linear.rs` | `save()` → JSON, `load()` → HashMap 복원 | serde_json 사용 |

## Config

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `policy.supervisory.warning_threshold` | f32 | 0.4 | MGR-054 |
| `policy.supervisory.critical_threshold` | f32 | 0.7 | MGR-054 |
| `policy.supervisory.warning_release` | f32 | 0.25 | MGR-054 |
| `policy.supervisory.critical_release` | f32 | 0.50 | MGR-054 |
| `policy.supervisory.hold_time_secs` | f32 | 4.0 | MGR-054 |

## 주요 Struct/Trait 매핑

| spec 개념 | Rust 타입 | 위치 |
|-----------|----------|------|
| OperatingMode | `enum OperatingMode` | `manager/src/types.rs:75-80` |
| ConnectionState | `enum ConnectionState` | `manager/src/channel/unix_socket.rs:37-44` |
| TcpConnectionState | `enum TcpConnectionState` | `manager/src/channel/tcp.rs` |
| ThresholdEvaluator | `struct ThresholdEvaluator` | `manager/src/evaluator.rs:26-30` |
| Direction | `enum Direction` | `manager/src/evaluator.rs:4-9` |
| Thresholds | `struct Thresholds` | `manager/src/evaluator.rs:12-19` |
| Level | `enum Level` | `shared/src/lib.rs` (llm_shared 크레이트) |
| ObservationContext | `struct ObservationContext` | `manager/src/pipeline.rs:49-58` |
| SupervisoryLayer | `struct SupervisoryLayer` | `manager/src/supervisory.rs:13-22` |
| ReaderHandle | `struct ReaderHandle` (RAII) | `manager/src/channel/unix_socket.rs:18-29` |