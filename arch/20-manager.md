# Manager Overview -- Architecture

> spec/20-manager.md의 구현 상세.

## 코드 매핑

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MGR-010 | `manager/src/main.rs` | 독립 바이너리 `llm_manager` | Cargo.toml `[[bin]]` |
| MGR-011 | (아키텍처적 보증) | Manager 프로세스 분리 | Engine 크레이트에 Manager 의존 없음 |
| MGR-012 | `manager/src/main.rs:163-215` | main loop: Monitor rx → policy → emitter | 3-layer 순차 실행 |
| MGR-013 | `manager/src/main.rs:116` | `create_transport()` — 단일 채널 생성 | 1:1 통신 |
| MGR-014 | `manager/src/monitor/mod.rs` | `Monitor` trait: `run()`, `initial_signal()`, `name()` | 4+1개 구현체 |
| MGR-015 | `manager/src/monitor/mod.rs:24-40` | `trait Monitor` 정의 | `run`, `initial_signal`, `name` 3개 메서드 |
| MGR-016 | `manager/src/monitor/memory.rs` | `MemoryMonitor` — `/proc/meminfo` 파싱 | `available_bytes`, `total_bytes` |
| MGR-017 | `manager/src/monitor/thermal.rs` | `ThermalMonitor` — `/sys/class/thermal/` 읽기 | `temperature_mc`, `throttle_ratio` |
| MGR-018 | `manager/src/monitor/compute.rs` | `ComputeMonitor` — `/proc/stat` CPU delta | `cpu_usage_pct`, `gpu_usage_pct` |
| MGR-019 | `manager/src/monitor/energy.rs` | `EnergyMonitor` — `/sys/class/power_supply/` | `battery_pct`, `power_budget_mw` |
| MGR-020 | `manager/src/monitor/external.rs` | `ExternalMonitor` — stdin/unix JSON Lines | 연구/테스트 전용 |
| MGR-021 | `manager/src/evaluator.rs` | `ThresholdEvaluator` struct | D-Bus Emitter 전용. Direction, Thresholds |
| MGR-022 | `manager/src/config.rs:31` | `ManagerConfig.poll_interval_ms = 1000` | Monitor별 `poll_interval_ms` Override 가능 |
| MGR-023 | `manager/src/pipeline.rs:34-46` | `trait PolicyStrategy` | `process_signal`, `update_engine_state`, `mode`, `save_model` |
| MGR-024 | `manager/src/pipeline.rs:189-229` | `HierarchicalPolicy::update_pressure()` | 도메인별 분기: Memory/Thermal/Compute/Energy |
| MGR-025 | `manager/src/supervisory.rs` | `SupervisoryLayer::evaluate()` | peak pressure → OperatingMode |
| MGR-026 | `manager/src/selector.rs` | `ActionSelector::select()` | Stateless, 2^N 전수 탐색 |
| MGR-027 | `manager/src/relief/mod.rs` | `trait ReliefEstimator` | `predict`, `observe`, `save`, `load`, `observation_count` |
| MGR-027 | `manager/src/relief/linear.rs` | `OnlineLinearEstimator` 구현체 | RLS 온라인 선형 회귀 |
| MGR-028 | `manager/src/action_registry.rs` | `ActionRegistry::from_config()` | PolicyConfig에서 초기화. 최대 8종 |
| MGR-029 | `manager/src/pipeline.rs:221-228` | `EnergyConstraint` → compute PI 보조 | **코드-스펙 차이**: 코드는 `level_to_measurement(level)` 사용, 스펙은 raw `battery_pct` 직접 사용 |
| MGR-030 | `manager/src/pipeline.rs:62` | `OBSERVATION_DELAY_SECS = 3.0` | `update_observation()` 메서드 |
| MGR-031 | `manager/src/emitter/mod.rs:13-38` | `trait Emitter` | `emit`, `emit_initial`, `emit_directive`, `name` |
| MGR-032 | `manager/src/channel/mod.rs:14-22` | `trait EngineReceiver` | `try_recv`, `is_connected` |
| MGR-033 | `manager/src/channel/unix_socket.rs` | `UnixSocketChannel` — `ConnectionState` 3-state | `Listening`, `Connected`, `Disconnected` |
| MGR-033 | `manager/src/channel/tcp.rs` | `TcpChannel` — 동일 3-state 패턴 | |
| MGR-034 | `manager/src/emitter/dbus.rs` | `DbusEmitter` | `#[cfg(feature = "dbus")]` |
| MGR-035 | `manager/src/main.rs:163-215` | main loop 구현 | Engine drain → `recv_timeout(50ms)` → policy |
| MGR-036 | `manager/src/main.rs:170-191` | `while let Some(msg) = transport.try_recv_engine_message()` | Engine 메시지 우선 drain |
| MGR-037 | `manager/src/main.rs:172-190` | `match &msg` — Heartbeat/Response/Capability 분기 | `update_engine_state()` for Heartbeat |
| MGR-038 | `manager/src/main.rs:193-210` | `rx.recv_timeout(50ms)` → `policy.process_signal()` | Directive 생성 시 `emit_directive()` |
| MGR-039 | `manager/src/main.rs:128-378` | `spawn_monitors()` + main thread | mpsc 채널 통신 |
| MGR-040 | `manager/src/main.rs:352-378` | `spawn_monitors()` | 독립 OS 스레드, `mpsc::Sender` 전달 |
| MGR-041 | `manager/src/main.rs:58-62,217-228` | `SHUTDOWN` AtomicBool + signal handler | save → join → exit |
| MGR-042 | `manager/src/main.rs:238-268` | `load_policy_config()` | 우선순위: CLI > config.toml > default |
| MGR-043 | `manager/src/main.rs:270-307` | `create_transport()` | `unix:`, `tcp:`, `dbus` 분기 |
| MGR-044 | `manager/src/main.rs:320-350` | `build_monitors()` | `enabled` 플래그 검사 |
| MGR-045 | `manager/src/main.rs:124-159` | `initial_signals` 수집 → `process_signal()` 투입 | 초기 Directive 생성 가능 |
| MGR-046 | `manager/Cargo.toml` | `name = "llm_manager"` | 바이너리명 |
| MGR-047 | `manager/src/main.rs:64-85` | `Args` struct (clap) | `--config`, `--transport`, `--client-timeout`, `--policy-config` |
| MGR-048 | `manager/src/main.rs:143-144,221-222` | `set_relief_model_path()`, `policy.save_model()` | 시작 시 load, 종료 시 save |
| MGR-C01 | (아키텍처적 보증) | `std::thread` + `std::sync::mpsc` 사용 | Cargo.toml에 async 런타임 의존 없음 |
| MGR-C02 | `manager/Cargo.toml` | `llm_shared` 의존 | IPC 타입 (`SystemSignal`, `EngineDirective` 등)에 한정 |

## 코드-스펙 차이 (Known Divergence)

| 항목 | 스펙 | 코드 | 영향 |
|------|------|------|------|
| EnergyConstraint 처리 (MGR-029) | raw `battery_pct`에서 `m = clamp(1 - battery_pct/100, 0, 1) * 0.5` | `level_to_measurement(level) * 0.5` — Level enum 기반 변환 | 스펙은 raw 값 직접 사용을 명세. 코드는 Level 기반으로 4단계 이산 변환. 향후 raw 전환 필요 |
| ActionId 7종 vs 8종 (MGR-028) | 8종 (`KvMergeD2o` 포함) | 7종 (`KvMergeD2o` variant 없음) | 스펙 향후 호환용. Engine 측 merge 미구현 |

## Config

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `manager.poll_interval_ms` | u64 | 1000 | MGR-022 |

## CLI

| 플래그 | 설명 | spec/ 근거 |
|--------|------|-----------|
| `--config` / `-c` | TOML 설정 파일 경로 | MGR-047 |
| `--transport` / `-t` | 전송 매체 (`dbus`, `unix:<path>`, `tcp:<host:port>`) | MGR-043, MGR-047 |
| `--client-timeout` | Engine 연결 대기 시간 (초) | MGR-047 |
| `--policy-config` | 별도 정책 설정 TOML 경로 | MGR-042, MGR-047 |

## 모듈 구조

```
manager/src/
├── main.rs                  # 바이너리 진입점, main loop, CLI
├── lib.rs                   # 모듈 re-export
├── config.rs                # Config, ManagerConfig, PolicyConfig 등 TOML 스키마
├── types.rs                 # ActionId, PressureVector, ReliefVector, FeatureVector 등
├── pipeline.rs              # PolicyStrategy trait, HierarchicalPolicy 구현
├── pi_controller.rs         # PiController, GainZone
├── supervisory.rs           # SupervisoryLayer (OperatingMode FSM)
├── selector.rs              # ActionSelector (stateless 전수 탐색)
├── action_registry.rs       # ActionRegistry (ActionMeta 저장소)
├── evaluator.rs             # ThresholdEvaluator (D-Bus Emitter 전용)
├── monitor/
│   ├── mod.rs               # Monitor trait 정의
│   ├── memory.rs            # MemoryMonitor
│   ├── thermal.rs           # ThermalMonitor
│   ├── compute.rs           # ComputeMonitor
│   ├── energy.rs            # EnergyMonitor
│   └── external.rs          # ExternalMonitor (연구/테스트)
├── emitter/
│   ├── mod.rs               # Emitter trait 정의
│   ├── unix_socket.rs       # UnixSocketEmitter
│   └── dbus.rs              # DbusEmitter (#[cfg(feature = "dbus")])
├── channel/
│   ├── mod.rs               # EngineReceiver trait, EngineChannel trait
│   ├── unix_socket.rs       # UnixSocketChannel (Emitter + EngineReceiver)
│   └── tcp.rs               # TcpChannel (Emitter + EngineReceiver)
├── relief/
│   ├── mod.rs               # ReliefEstimator trait 정의
│   └── linear.rs            # OnlineLinearEstimator (RLS)
└── bin/
    ├── mock_engine.rs       # Manager 테스트용 모의 Engine
    └── mock_manager.rs      # Engine 테스트용 모의 Manager
```