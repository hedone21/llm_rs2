# Protocol Messages -- Architecture

> spec/11-protocol-messages.md의 구현 상세.

## 코드 매핑

모든 메시지 타입은 `shared/src/lib.rs` 단일 파일에 정의되어 있다.

### 3.1 Envelope Types [MSG-010 ~ MSG-014]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MSG-010 | `shared/src/lib.rs:213` | `pub enum ManagerMessage` — `#[serde(tag = "type", rename_all = "snake_case")]` | 1종: Directive |
| MSG-011 | `shared/src/lib.rs:277` | `pub enum EngineMessage` — `#[serde(tag = "type", rename_all = "snake_case")]` | 4종: Capability, Heartbeat, Response, QcfEstimate |
| MSG-014 | — | QcfEstimate는 `shared/src/lib.rs`에 **미정의** | spec에 명세되었으나 코드 미구현 |

### 3.2 EngineDirective [MSG-020 ~ MSG-022]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MSG-020 | `shared/src/lib.rs:205` | `pub struct EngineDirective { pub seq_id: u64, pub commands: Vec<EngineCommand> }` | |
| MSG-021 | `manager/src/pipeline.rs:65-68` | `SEQ_COUNTER: AtomicU64::new(1)` | 단조 증가 보장 |

### 3.3 EngineCommand [MSG-030 ~ MSG-041]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MSG-030 | `shared/src/lib.rs:170` | `pub enum EngineCommand` — `#[serde(tag = "type", rename_all = "snake_case")]` | 13종 |

#### EngineCommand 변형 — shared/src/lib.rs 내 위치

| Tag Value | Variant | 필드 | 라인 (approx) |
|-----------|---------|------|--------------|
| `"throttle"` | Throttle | `delay_ms: u64` | ~172 |
| `"layer_skip"` | LayerSkip | `skip_ratio: f32` | ~174 |
| `"kv_evict_h2o"` | KvEvictH2o | `keep_ratio: f32` | ~176 |
| `"kv_evict_sliding"` | KvEvictSliding | `keep_ratio: f32` | ~178 |
| `"kv_merge_d2o"` | KvMergeD2o | `keep_ratio: f32` | ~180 |
| `"kv_streaming"` | KvStreaming | `sink_size: usize, window_size: usize` | ~182 |
| `"kv_quant_dynamic"` | KvQuantDynamic | `target_bits: u8` | ~185 |
| `"request_qcf"` | RequestQcf | (없음) | ~187 |
| `"restore_defaults"` | RestoreDefaults | (없음) | ~189 |
| `"switch_hw"` | SwitchHw | `device: String` | ~191 |
| `"prepare_compute_unit"` | PrepareComputeUnit | `device: String` | ~193 |
| `"suspend"` | Suspend | (없음) | ~195 |
| `"resume"` | Resume | (없음) | ~197 |

#### 명령 실행 매핑 (Engine측)

| EngineCommand | 실행 위치 | 메서드 |
|---------------|----------|--------|
| Throttle | `engine/src/resilience/executor.rs` | `apply_command()` → `plan.throttle_delay_ms` |
| LayerSkip | `engine/src/resilience/executor.rs` | `apply_command()` → `plan.skip_ratio` |
| KvEvictH2o | `engine/src/resilience/executor.rs` | `apply_command()` → eviction 실행 |
| KvEvictSliding | `engine/src/resilience/executor.rs` | `apply_command()` → eviction 실행 |
| KvMergeD2o | `engine/src/resilience/executor.rs` | `apply_command()` → D2O merge |
| KvStreaming | `engine/src/resilience/executor.rs` | `apply_command()` → Rejected (미구현) |
| KvQuantDynamic | `engine/src/resilience/executor.rs` | `apply_command()` → KIVI bits 전환 |
| RequestQcf | `engine/src/resilience/executor.rs` | `apply_command()` → QCF 계산 |
| RestoreDefaults | `engine/src/resilience/executor.rs` | `apply_command()` → 전체 리셋 |
| SwitchHw | `engine/src/resilience/executor.rs` | `apply_command()` → backend 전환 |
| PrepareComputeUnit | `engine/src/resilience/executor.rs` | `apply_command()` → 워밍업 |
| Suspend | `engine/src/resilience/executor.rs` | `apply_command()` → plan override |
| Resume | `engine/src/resilience/executor.rs` | `apply_command()` → resume |

### 3.4 EngineCapability [MSG-050 ~ MSG-052]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MSG-050 | `shared/src/lib.rs:219` | `pub struct EngineCapability` — 5필드 | |

#### 필드 → 코드

| 필드 | 타입 (Rust) |
|------|-----------|
| `available_devices` | `Vec<String>` |
| `active_device` | `String` |
| `max_kv_tokens` | `usize` |
| `bytes_per_kv_token` | `usize` |
| `num_layers` | `usize` |

### 3.5 EngineStatus (Heartbeat) [MSG-060 ~ MSG-066]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MSG-060 | `shared/src/lib.rs:229` | `pub struct EngineStatus` — 16필드 | |
| MSG-061 | `shared/src/lib.rs` | 필드 12~16에 `#[serde(default)]` | 하위 호환 |

#### 필드 → 코드 (serde(default) 여부)

| # | 필드 | 타입 (Rust) | serde(default) |
|---|------|-----------|---------------|
| 1 | `active_device` | `String` | X |
| 2 | `compute_level` | `ResourceLevel` | X |
| 3 | `actual_throughput` | `f32` | X |
| 4 | `memory_level` | `ResourceLevel` | X |
| 5 | `kv_cache_bytes` | `u64` | X |
| 6 | `kv_cache_tokens` | `usize` | X |
| 7 | `kv_cache_utilization` | `f32` | X |
| 8 | `memory_lossless_min` | `f32` | X |
| 9 | `memory_lossy_min` | `f32` | X |
| 10 | `state` | `EngineState` | X |
| 11 | `tokens_generated` | `usize` | X |
| 12 | `available_actions` | `Vec<String>` | O |
| 13 | `active_actions` | `Vec<String>` | O |
| 14 | `eviction_policy` | `String` | O |
| 15 | `kv_dtype` | `String` | O |
| 16 | `skip_ratio` | `f32` | O |

### 3.6 CommandResponse [MSG-070 ~ MSG-073]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MSG-070 | `shared/src/lib.rs:269` | `pub struct CommandResponse { pub seq_id: u64, pub results: Vec<CommandResult> }` | |
| INV-025 | `engine/src/resilience/executor.rs` | `results.len() == commands.len()` 보장 | |
| INV-026 | `engine/src/resilience/executor.rs` | 수신 seq_id에 대해서만 Response 전송 | |

### 3.7 CommandResult [MSG-080 ~ MSG-083]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MSG-080 | `shared/src/lib.rs:261` | `pub enum CommandResult` — `#[serde(tag = "status", rename_all = "snake_case")]` | 3종 |

#### CommandResult 변형

| Tag Value | Variant | 추가 필드 |
|-----------|---------|----------|
| `"ok"` | Ok | (없음) |
| `"partial"` | Partial | `achieved: f32, reason: String` |
| `"rejected"` | Rejected | `reason: String` |

### 3.8 QcfEstimate [MSG-085 ~ MSG-087]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MSG-085 | — | `shared/src/lib.rs`에 **미정의** | spec에 명세되었으나 struct 미구현. EngineMessage에 QcfEstimate 변형만 존재할 수 있음 |

### 3.9 Supporting Enums [MSG-090 ~ MSG-095]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MSG-090 | `shared/src/lib.rs:151` | `pub enum ResourceLevel` — Normal, Warning, Critical | PartialOrd derive |
| MSG-091 | `shared/src/lib.rs:160` | `pub enum EngineState` — Idle, Running, Suspended | |
| MSG-092 | `shared/src/lib.rs:7` | `pub enum Level` — Normal, Warning, Critical, Emergency | PartialOrd, Ord derive |
| MSG-093 | `shared/src/lib.rs:17` | `pub enum RecommendedBackend` — Cpu, Gpu, Any | |
| MSG-094 | `shared/src/lib.rs:26` | `pub enum ComputeReason` — 6종 | |
| MSG-095 | `shared/src/lib.rs:38` | `pub enum EnergyReason` — 6종 | `None` 변형에 명시적 `#[serde(rename = "none")]` |

### 3.10 D-Bus SystemSignal [MSG-100 ~ MSG-104]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| MSG-100 | `shared/src/lib.rs:106` | `pub enum SystemSignal` — 4종 (externally tagged) | |
| MSG-101 | `shared/src/lib.rs` | `MemoryPressure { level, available_bytes, total_bytes, reclaim_target_bytes }` | |
| MSG-102 | `shared/src/lib.rs` | `ComputeGuidance { level, recommended_backend, reason, cpu_usage_pct, gpu_usage_pct }` | |
| MSG-103 | `shared/src/lib.rs` | `ThermalAlert { level, temperature_mc, throttling_active, throttle_ratio }` | |
| MSG-104 | `shared/src/lib.rs` | `EnergyConstraint { level, reason, power_budget_mw }` | |

### Constraints [CON-020 ~ CON-022]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| CON-020 | `shared/src/lib.rs` | serde 어노테이션이 와이어 포맷 결정 | 필드명 변경 금지 |
| CON-021 | `shared/src/lib.rs` | `#[serde(default)]`로 신규 필드 추가 가능 | 하위 호환 유지 |
| INV-027 | `shared/src/lib.rs` | serde 어노테이션 변경 = 프로토콜 버전 변경 | |
| INV-028 | `shared/src/lib.rs` | 새 필드 시 `#[serde(default)]` 필수 | |

## Config

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| (해당 없음 — 메시지 정의는 코드 내 serde derive로 결정) | | | |

## CLI

| 플래그 | 설명 | spec/ 근거 |
|--------|------|-----------|
| (해당 없음 — 메시지 타입은 CLI와 무관) | | |