# Engine State Machines -- Architecture

> spec/31-engine-state.md의 구현 상세.

## 코드 매핑

### 3.1 OperatingMode FSM

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ST-010 | `engine/src/resilience/state.rs` | `OperatingMode` enum: Normal, Degraded, Minimal, Suspended | 4-state |
| ENG-ST-011 | `engine/src/resilience/state.rs` | `OperatingMode::from_levels(memory, compute, thermal, energy)` — worst-wins 순수 함수 | `max()` 비교 |
| ENG-ST-012 | `engine/src/resilience/state.rs` | `from_levels()` 내부 match 구현 | Level::Normal→Normal, Warning→Degraded, Critical→Minimal, Emergency→Suspended |
| ENG-ST-013 | `engine/src/resilience/state.rs` | 12 전이 — `from_levels()` 순수 함수이므로 이전 상태 무관, 모든 조합 가능 | |
| ENG-ST-014 | `engine/src/resilience/manager.rs` | `ResilienceManager::process_signal()` 내 `from_levels()` 호출 | Strategy 경로(D-Bus 레거시) 전용. CommandExecutor에서는 미사용 |
| ENG-ST-015 | `engine/src/resilience/dbus_transport.rs` | `signal_to_manager_message()`: Emergency → `EngineCommand::Suspend` 변환 | SYS-055 자율 전이 |

### 3.2 EngineState FSM

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ST-020 | `shared/src/lib.rs` (또는 shared 크레이트 내) | `EngineState` enum: Idle, Running, Suspended | 3-state, shared 크레이트에 정의 (ENG-ST-025) |
| ENG-ST-021 | `engine/src/resilience/executor.rs`, `engine/src/bin/generate.rs` | 전이 6종: Idle→Running (`set_running()`), Running→Suspended (`apply_command(Suspend)`), Suspended→Running (`apply_command(Resume)`), Running→Idle (loop 종료), Idle→Idle (자기 전이), Suspended→Idle (프로세스 종료) | |
| ENG-ST-022 | — | Idle→Suspended: 코드상 전이 발생 가능하나 의미 없는 전이 | SHOULD NOT |
| ENG-ST-023 | `engine/src/resilience/executor.rs` `send_heartbeat()` | `Heartbeat.state = engine_state` — EngineState를 Manager에 보고 | |
| ENG-ST-024 | — | EngineState와 OperatingMode는 독립 FSM | EngineState: 프로토콜 수준, OperatingMode: Strategy 경로 내부 |
| ENG-ST-025 | `shared/src/lib.rs` | EngineState는 shared 크레이트에 정의 (Manager-Engine 프로토콜 공유). OperatingMode는 `engine/src/resilience/state.rs`에 정의 (Engine 내부 전용) | |

### 3.3 CommandExecutor State

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ST-030 | `engine/src/resilience/executor.rs` `CommandExecutor` struct | 내부 필드 11종: `engine_state`, `compute_level`, `memory_level`, `active_device`, `throttle_delay_ms`, `active_actions`, `throughput_ema`, `last_token_time`, `tokens_generated`, `last_heartbeat`, `heartbeat_interval` | |
| ENG-ST-031 | `engine/src/resilience/executor.rs` | `active_actions: Vec<String>` 관리 — `EngineCommand`별 추가/제거/초기화 규칙 | RestoreDefaults → `clear()` |
| ENG-ST-032 | `engine/src/resilience/executor.rs` `compute_available_actions()` | 기본 3종 + eviction_policy 조건 + kv_dtype 조건 | Heartbeat 전송 시 호출 |
| ENG-ST-033 | `engine/src/resilience/executor.rs` `apply_command()` | EngineCommand 13종 → ExecutionPlan 필드 1:1 매핑 | CommandResult 반환 |
| ENG-ST-034 | `engine/src/resilience/executor.rs` `poll()` | 7단계: Heartbeat 체크 → try_recv drain → directives 수집 → apply_command 순차 → Suspend plan 초기화 → throttle 동기화 → plan 반환 | |
| ENG-ST-035 | `engine/src/resilience/executor.rs` `poll()` | Superseding 규칙: 동일 poll() 내 복수 Directive 순서 처리, 후행이 선행 plan 필드를 덮어씀 | |

### EngineCommand 13종 → ExecutionPlan 매핑

| 명령 | Plan 필드 | CommandResult | 코드 위치 |
|------|----------|---------------|----------|
| Throttle { delay_ms } | `plan.throttle_delay_ms` | Ok | `executor.rs` `apply_command()` |
| LayerSkip { skip_ratio } | `plan.layer_skip = Some(ratio)` | Ok | `executor.rs` |
| KvEvictH2o { keep_ratio } | `plan.evict = Some(EvictPlan { H2o, ratio, Critical })` | Ok | `executor.rs` |
| KvEvictSliding { keep_ratio } | `plan.evict = Some(EvictPlan { Sliding, ratio, Critical })` | Ok | `executor.rs` |
| KvMergeD2o { keep_ratio } | `plan.evict = Some(EvictPlan { D2o, ratio, Critical })` | Ok | `executor.rs` (미구현: 향후 추가) |
| KvStreaming { .. } | (plan 변경 없음) | Rejected("KvStreaming not yet implemented") | `executor.rs` |
| KvQuantDynamic { target_bits } | `plan.kv_quant_bits = Some(bits)` | Ok | `executor.rs` |
| RequestQcf | `plan.request_qcf = true` | Ok | `executor.rs` (QcfEstimate 별도 EngineMessage 전송) |
| RestoreDefaults | `plan.restore_defaults = true`, `plan.throttle_delay_ms = 0` | Ok | `executor.rs` |
| SwitchHw { device } | `plan.switch_device = Some(device)` | Ok | `executor.rs` |
| PrepareComputeUnit { device } | `plan.prepare_device = Some(device)` | Ok | `executor.rs` |
| Suspend | `plan.suspended = true` | Ok | `executor.rs` |
| Resume | `plan.resumed = true`, `plan.throttle_delay_ms = 0` | Ok | `executor.rs` |

### 3.4 ExecutionPlan Lifecycle

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ST-040 | `engine/src/resilience/executor.rs` | `ExecutionPlan` struct: 9 필드 (evict, switch_device, prepare_device, throttle_delay_ms, suspended, resumed, layer_skip, kv_quant_bits, restore_defaults) | `#[derive(Default)]` |
| ENG-ST-041 | `engine/src/resilience/executor.rs` | `EvictPlan` struct: `target_ratio: f32`, `level: ResourceLevel`, `method: EvictMethod` | EvictMethod: H2o, Sliding, Streaming |
| ENG-ST-042 | `engine/src/bin/generate.rs` decode loop | `let plan = executor.poll(...)` → 즉시 소비 → 다음 iteration에서 재할당으로 폐기 | 1회성 수명 |
| ENG-ST-043 | `engine/src/bin/generate.rs` decode loop | 소비 순서 9단계: evict → switch_device → prepare_device → kv_quant_bits → layer_skip → restore_defaults → suspended → resumed → throttle_delay_ms | |

### ExecutionPlan 필드 상세

| 필드 | 타입 | 기본값 | 코드 위치 |
|------|------|--------|----------|
| evict | `Option<EvictPlan>` | None | `executor.rs` |
| switch_device | `Option<String>` | None | `executor.rs` |
| prepare_device | `Option<String>` | None | `executor.rs` |
| throttle_delay_ms | u64 | 0 | `executor.rs` |
| suspended | bool | false | `executor.rs` |
| resumed | bool | false | `executor.rs` |
| layer_skip | `Option<f32>` | None | `executor.rs` |
| kv_quant_bits | `Option<u8>` | None | `executor.rs` |
| restore_defaults | bool | false | `executor.rs` |

### EvictMethod enum

`H2o`, `Sliding`, `Streaming` — Engine 내부 타입, shared 프로토콜에 미포함.

코드 위치: `engine/src/resilience/executor.rs`

### 3.5 ResilienceManager State (D-Bus 레거시)

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ST-050 | `engine/src/resilience/manager.rs` `ResilienceManager` struct | 내부 필드: `mode: OperatingMode`, `current_levels: SignalLevels` (4종 Level), `strategies: HashMap` (4종 Strategy) | |
| ENG-ST-051 | `engine/src/resilience/manager.rs` `poll()` | try_recv drain → process_signal() × N → resolve_conflicts() | |
| ENG-ST-052 | `engine/src/resilience/strategy/` (memory.rs, compute.rs, thermal.rs, energy.rs) | 각 Strategy의 `react()` 구현 — Level별 ResilienceAction 반환 | |
| ENG-ST-053 | `engine/src/resilience/strategy/` 또는 `manager.rs` | `ResilienceStrategy` trait: `react(&SystemSignal, OperatingMode) -> Vec<ResilienceAction>` + `name() -> &str` | `Send + Sync` 바운드 |
| ENG-ST-054 | `engine/src/resilience/manager.rs` | `InferenceContext` struct + `execute_action()` 함수 | `generate.rs`에서 현재 미사용 |
| ENG-ST-055 | — | ResilienceManager(Strategy) vs CommandExecutor(Directive) 관계 — 독립 경로, DbusTransport가 브리징 | |

### Strategy 4종 코드 위치

| Strategy | 코드 위치 | domain |
|----------|----------|--------|
| MemoryStrategy | `engine/src/resilience/strategy/memory.rs` | MemoryPressure |
| ComputeStrategy | `engine/src/resilience/strategy/compute.rs` (추정) | ComputeGuidance |
| ThermalStrategy | `engine/src/resilience/strategy/thermal.rs` (추정) | ThermalAlert |
| EnergyStrategy | `engine/src/resilience/strategy/energy.rs` (추정) | EnergyConstraint |

### 3.6 resolve_conflicts()

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ST-060 | `engine/src/resilience/manager.rs` `resolve_conflicts()` | 7규칙 single-pass scan | |
| ENG-ST-061 | `engine/src/resilience/manager.rs` | R1: Suspend → `[Suspend]` 반환. R2: RestoreDefaults 억제. R3: Evict min ratio. R4: SwitchBackend CPU wins. R5: LimitTokens min. R6: Throttle max. R7: RejectNew any | |
| ENG-ST-063 | `engine/src/resilience/signal.rs` 또는 `manager.rs` | `ResilienceAction` enum 7종: Evict, SwitchBackend, LimitTokens, Throttle, Suspend, RejectNew, RestoreDefaults | Strategy 경로 내부 타입 |

### ResilienceAction 7종

| Action | 코드 위치 |
|--------|----------|
| Evict { target_ratio: f32 } | `engine/src/resilience/` |
| SwitchBackend { to: BackendType } | `engine/src/resilience/` |
| LimitTokens { max_tokens: usize } | `engine/src/resilience/` |
| Throttle { delay_ms: u64 } | `engine/src/resilience/` |
| Suspend | `engine/src/resilience/` |
| RejectNew | `engine/src/resilience/` |
| RestoreDefaults | `engine/src/resilience/` |

### 3.7 KVSnapshot Reporting

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-ST-070 | `engine/src/resilience/executor.rs` 또는 `engine/src/bin/generate.rs` | `KVSnapshot` struct: 7 필드 (total_bytes, total_tokens, capacity, protected_prefix, kv_dtype, eviction_policy, skip_ratio) | Heartbeat의 EngineStatus 필드 채움 |

### KVSnapshot 필드 출처

| 필드 | 타입 | 출처 코드 |
|------|------|----------|
| total_bytes | u64 | 전 layer KV buffer 크기 합산 |
| total_tokens | usize | `kv_caches[0].current_pos()` |
| capacity | usize | `kv_caches[0].capacity()` |
| protected_prefix | usize | CLI `--protected-prefix` 또는 정책 기본값 |
| kv_dtype | String | CLI `--kv-type` / KIVI 모드 ("f16", "q4", "q2") |
| eviction_policy | String | CLI `--eviction-policy` |
| skip_ratio | f32 | 현재 layer skip 비율 |

### Invariants 코드 매핑

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| INV-070 | `engine/src/resilience/state.rs` `from_levels()` | 순수 함수 — 입력 4종 Level만 참조, `&self` 없음 | |
| INV-071 | `engine/src/resilience/executor.rs` `apply_command()` | EngineState 전이는 `apply_command(Suspend)`, `apply_command(Resume)`, `set_running()` 내부에서만 발생 | 외부 직접 변경 불가 (private 필드) |
| INV-072 | `engine/src/resilience/manager.rs` `resolve_conflicts()` | `if has_suspend: return vec![Suspend]` | R1 규칙 |
| INV-073 | `engine/src/resilience/manager.rs` `resolve_conflicts()` | `if has_restore AND no other constraints: return vec![RestoreDefaults]` | R2 규칙 |
| INV-074 | `engine/src/resilience/executor.rs` `poll()` step 5 | `if plan.suspended { plan.evict = None; plan.switch_device = None; plan.prepare_device = None; }` | |
| INV-075 | `engine/src/resilience/executor.rs` `apply_command(Resume)` | `compute_level = Normal; memory_level = Normal; throttle_delay_ms = 0;` | |
| INV-076 | `engine/src/resilience/executor.rs` `apply_command(RestoreDefaults)` | `active_actions.clear(); throttle_delay_ms = 0; compute_level = Normal; memory_level = Normal;` | |

## CLI

(Engine 상태 머신 관련 CLI는 spec/30-engine.md (arch/30-engine.md)에서 관리한다. 이 문서에서는 상태 머신 내부 동작만 기술한다.)

## Config

(Engine 상태 머신은 별도 config 파일을 사용하지 않는다. 하드코딩 값은 다음과 같다.)

| 항목 | 값 | 타입 | spec/ 근거 |
|------|---|------|-----------|
| heartbeat_interval | 1000ms | Duration | ENG-ST-030, ENG-033 |
| throughput_ema alpha | 0.1 | f32 | ENG-ST-030 |
| EvictPlan.level 기본값 | Critical | ResourceLevel | ENG-ST-033, ENG-ST-041 |
