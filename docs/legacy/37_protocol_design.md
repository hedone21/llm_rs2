# 37. Manager ↔ Engine Protocol Design

> **Status**: Design (v1.0)
> **Author**: Architect
> **Date**: 2026-03-20
> **Supersedes**: `20_dbus_ipc_spec.md`의 wire protocol 및 메시지 포맷 섹션

---

## 1. 개요

이 문서는 Manager(`llm_manager` crate)와 Engine(`llm_rs2` crate) 간의 양방향 통신 프로토콜을 정의한다.

기존 `20_dbus_ipc_spec.md`는 단방향 `SystemSignal` 기반 프로토콜(Manager → Engine 일방 전달)을 명세했다. 이 문서는 그 프로토콜을 양방향 `ManagerMessage` ↔ `EngineMessage` 커맨드 프로토콜로 대체한다.

**Supersedes**: `20_dbus_ipc_spec.md`의 wire protocol 및 메시지 포맷 섹션. D-Bus 시그널 이름(`MemoryPressure`, `ComputeGuidance`, `ThermalAlert`, `EnergyConstraint`)과 인터페이스 정의(`org.llm.Manager1`)는 하위 호환을 위해 유지하되, 실질적 데이터 교환은 이 문서의 프로토콜을 따른다.

**구현 위치**: `shared/src/lib.rs` — Manager와 Engine이 공통으로 의존하는 타입들이 정의되어 있으며, 이미 다음 타입들이 구현되어 있다: `EngineCapability`, `EngineStatus`, `EngineDirective`, `EngineCommand`, `CommandResponse`, `CommandResult`, `EngineMessage`, `ManagerMessage`.

---

## 2. 프로토콜 아키텍처

```
Engine                                    Manager
──────                                    ───────
  │                                          │
  │── EngineMessage::Capability ───────────→│  (세션 시작 시 1회)
  │                                          │
  │── EngineMessage::Heartbeat ────────────→│  (주기적, ~100ms)
  │                                          │
  │←─────────────── ManagerMessage::QcfRequest ──│  (on-demand, pressure 임계 초과 시)
  │── EngineMessage::QcfResponse ──────────→│
  │                                          │
  │←──────────── ManagerMessage::Directive ──│  (액션 필요 시)
  │── EngineMessage::Response ─────────────→│
  │                                          │
```

4가지 메시지 흐름: Capability(등록), Heartbeat(상태 보고), QCF(품질 비용 조회), Directive(명령 실행)

---

## 3. 메시지 타입 상세

### 3-1. Engine → Manager: Capability (세션 시작 시 1회)

`EngineMessage::Capability(EngineCapability)` — Engine이 연결 직후 자신의 능력을 보고한다. Manager는 이 정보로 이후 Directive에서 유효한 `device`와 파라미터 범위를 결정한다.

```rust
// shared/src/lib.rs (구현 완료)
pub struct EngineCapability {
    pub available_devices: Vec<String>,  // ["cpu", "opencl"]
    pub active_device: String,           // "cpu"
    pub max_kv_tokens: usize,            // 2048
    pub bytes_per_kv_token: usize,       // 256 (F16, 2 heads × 64 dim × 2 bytes)
    pub num_layers: usize,               // 16
}
```

**확장 예정 (policy_design 설계 반영 시)**: `ActionCapability` 목록을 추가하여 각 액션의 lossy/lossless 구분과 파라미터 범위를 명시할 수 있다.

```rust
// 향후 확장 구조 (미구현)
struct ActionCapability {
    id: ActionId,
    kind: ActionKind,                      // Lossless | Lossy
    param_range: Option<ParamRange>,       // 예: keep_ratio 0.3~0.9
    exclusion_group: Option<String>,       // 상호 배타 그룹
}

enum ActionId {
    SwitchHw,
    Throttle,
    KvOffloadDisk,
    KvEvictSliding,
    KvEvictH2o,
    KvQuantDynamic,
    LayerSkip,
    SnapkvCompress,
}

enum ActionKind { Lossless, Lossy }
```

---

### 3-2. Engine → Manager: Heartbeat (주기적, ~100ms)

`EngineMessage::Heartbeat(EngineStatus)` — Engine의 현재 상태를 주기적으로 보고한다.

```rust
// shared/src/lib.rs (구현 완료)
pub struct EngineStatus {
    pub active_device: String,           // 현재 활성 컴퓨트 유닛 ("cpu", "opencl")
    pub compute_level: ResourceLevel,    // 현재 적용된 컴퓨트 제약 수준
    pub actual_throughput: f32,          // 실측 TPS (tokens/sec)
    pub memory_level: ResourceLevel,     // 현재 적용된 메모리 제약 수준
    pub kv_cache_bytes: u64,             // KV 캐시 실제 사용 바이트
    pub kv_cache_tokens: usize,          // KV 캐시에 저장된 토큰 수
    pub kv_cache_utilization: f32,       // 0.0~1.0 (kv_cache_tokens / max_kv_tokens)
    pub memory_lossless_min: f32,        // 품질 손실 없이 확보 가능한 최소 메모리 비율
    pub memory_lossy_min: f32,           // lossy 액션까지 허용 시 확보 가능한 최소
    pub state: EngineState,              // Idle | Running | Suspended
    pub tokens_generated: usize,         // 현재 세션 누적 생성 토큰 수
}

pub enum EngineState {
    Idle,
    Running,
    Suspended,
}

// 3단계 압력 수준 (4단계 Level과 구분됨)
pub enum ResourceLevel {
    Normal,
    Warning,
    Critical,
}
```

**TBT 관련 확장 예정**: 현재 `actual_throughput`(TPS)으로 지연 정보를 표현하지만, 향후 `tbt_ema_ms`(Token Between Tokens EMA)와 `tbt_baseline_ms`를 분리하여 Manager가 latency 영향을 정규화할 수 있도록 확장할 수 있다:

```rust
// 향후 확장 (미구현)
tbt_ema_ms: f32,         // 최근 토큰 TBT EMA
tbt_baseline_ms: f32,    // 액션 미적용 시 기준 TBT
// Manager는 (tbt_ema - tbt_baseline) / tbt_baseline으로 latency overhead 정규화
```

---

### 3-3. Manager → Engine: Directive / Response

Manager가 Engine에 명령을 내린다. `EngineDirective`는 `seq_id`로 식별되며, Engine은 처리 결과를 `CommandResponse`로 반환한다.

```rust
// shared/src/lib.rs (구현 완료)

// Manager → Engine
pub enum ManagerMessage {
    Directive(EngineDirective),
    // 향후: QcfRequest(QcfRequest),
}

pub struct EngineDirective {
    pub seq_id: u64,
    pub commands: Vec<EngineCommand>,
}

pub enum EngineCommand {
    /// 컴퓨트 리소스 수준 설정 + 목표 처리량
    SetComputeLevel {
        level: ResourceLevel,
        target_throughput: f32,
        deadline_ms: Option<u64>,
    },
    /// 컴퓨트 유닛 전환 (즉시)
    SwitchComputeUnit { device: String },
    /// 컴퓨트 유닛 사전 준비 (전환 전 warm-up)
    PrepareComputeUnit { device: String },
    /// 메모리 압력 수준 설정 + 증발 목표 비율
    SetMemoryLevel {
        level: ResourceLevel,
        target_ratio: f32,
        deadline_ms: Option<u64>,
    },
    /// 즉시 추론 중단
    Suspend,
    /// Suspended 상태에서 재개
    Resume,
}

// Engine → Manager
pub enum EngineMessage {
    Capability(EngineCapability),
    Heartbeat(EngineStatus),
    Response(CommandResponse),
    // 향후: QcfResponse(QcfResponse),
}

pub struct CommandResponse {
    pub seq_id: u64,               // EngineDirective.seq_id와 매칭
    pub results: Vec<CommandResult>,
}

pub enum CommandResult {
    Ok,
    Partial { achieved: f32, reason: String },
    Rejected { reason: String },
}
```

**`Partial` vs `Rejected`**: `Partial`은 명령이 부분적으로 실행되었음을 뜻한다 (예: eviction 목표 0.5를 달성했지만 0.4까지만 확보됨). `Rejected`는 실행 자체가 불가한 경우다 (예: 단일 백엔드 환경에서 `SwitchComputeUnit` 요청).

---

### 3-4. QCF Request / Response (on-demand, 향후 확장)

Manager가 액션 선택 전 품질 비용을 조회하는 흐름. 현재 `shared/src/lib.rs`에 미구현이며, `36_policy_design.md`의 Selector 컴포넌트가 필요로 하는 인터페이스다.

Manager가 QCF를 요청하는 시점: pressure level이 Warning 이상으로 올라가서 액션 선택이 필요할 때만. Normal 상태에서는 요청하지 않는다.

```rust
// 향후 추가 예정 (shared/src/lib.rs)

// Manager → Engine (ManagerMessage 확장)
struct QcfRequest {
    request_id: u64,
    candidates: Vec<QcfCandidate>,
}

struct QcfCandidate {
    action: ActionId,
    tentative_params: ActionParams,  // 이 파라미터로 적용 시의 QCF 추정 요청
}

// Engine → Manager (EngineMessage 확장)
struct QcfResponse {
    request_id: u64,
    estimates: Vec<QcfEstimate>,
}

struct QcfEstimate {
    action: ActionId,
    qcf_value: f32,       // proxy 추정값 (0.0 = 영향 없음)
    confidence: f32,      // 추정 신뢰도 (0.0~1.0)
}
```

QCF 계산은 Engine 내부의 기존 QCF 모듈(`engine/src/core/qcf/`)을 활용한다. Engine은 attention weights 등 내부 데이터에 접근하여 스칼라 값으로 요약 보고하며, Manager는 이 값을 받아 Selector의 latency budget과 비교한다.

---

## 4. 상위 메시지 봉투 (Envelope)

모든 메시지는 단일 enum으로 래핑하여 전송한다. serde의 `#[serde(tag = "type")]` 어노테이션으로 JSON 직렬화 시 `"type"` 필드가 추가된다.

```rust
// Engine → Manager 방향 (구현 완료)
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineMessage {
    Capability(EngineCapability),   // type: "capability"
    Heartbeat(EngineStatus),        // type: "heartbeat"
    Response(CommandResponse),      // type: "response"
}

// Manager → Engine 방향 (구현 완료)
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ManagerMessage {
    Directive(EngineDirective),     // type: "directive"
}
```

JSON 직렬화 예시:

```json
// EngineMessage::Heartbeat 직렬화
{
  "type": "heartbeat",
  "active_device": "cpu",
  "compute_level": "normal",
  "actual_throughput": 15.3,
  "memory_level": "warning",
  "kv_cache_bytes": 4194304,
  "kv_cache_tokens": 512,
  "kv_cache_utilization": 0.25,
  "memory_lossless_min": 1.0,
  "memory_lossy_min": 0.01,
  "state": "running",
  "tokens_generated": 128
}

// ManagerMessage::Directive 직렬화
{
  "type": "directive",
  "seq_id": 42,
  "commands": [
    {
      "type": "set_memory_level",
      "level": "warning",
      "target_ratio": 0.85
    }
  ]
}
```

---

## 5. Wire Format

기존 `20_dbus_ipc_spec.md`의 Unix socket wire format을 유지한다. 전송 내용이 `SystemSignal`이 아닌 `EngineMessage` 또는 `ManagerMessage`의 serde JSON 직렬화로 변경되는 것이 핵심이다.

```
┌──────────┬─────────────────────────────────────┐
│ 4 bytes  │ N bytes                             │
│ BE u32 N │ UTF-8 JSON (EngineMessage 또는      │
│          │            ManagerMessage)           │
└──────────┴─────────────────────────────────────┘
```

- BE u32: 페이로드 길이 (빅엔디안 4바이트 unsigned integer)
- UTF-8 JSON: `serde_json::to_string()` 직렬화 결과

**Transport 구현:**

| Transport | 구현 방식 | 플랫폼 |
|-----------|----------|--------|
| UnixSocket | 위 wire format 직접 사용 | Android / Linux 범용 |
| D-Bus | `SystemSignal`을 수신하여 내부에서 `ManagerMessage::Directive`로 변환하는 어댑터 레이어 유지 | Linux |

D-Bus 어댑터 레이어는 하위 호환성을 위해 유지한다. `SystemSignal::MemoryPressure { level: Critical, reclaim_target_bytes: N }` 수신 시 `EngineCommand::SetMemoryLevel { level: Critical, target_ratio: 0.5 }`로 변환하는 방식이다.

---

## 6. Feature Vector 스키마

`36_policy_design.md`의 Relief Estimator가 사용하는 `EngineStatus` → Feature Vector 변환 규칙. 벡터 크기는 13으로 고정한다.

```rust
// 향후 Manager 내부 구현 (미구현)
struct FeatureVector {
    values: [f32; 13],
}
```

| Index | Name | Range | 변환 규칙 |
|-------|------|-------|-----------|
| 0 | kv_occupancy | 0.0~1.0 | `status.kv_cache_utilization` 직접 |
| 1 | is_gpu | 0 or 1 | `active_device == "opencl"` ? 1.0 : 0.0 |
| 2 | token_progress | 0.0~1.0 | `kv_cache_tokens / max_kv_tokens` |
| 3 | compute_level_norm | 0.0~1.0 | Normal=0.0, Warning=0.5, Critical=1.0 |
| 4 | memory_level_norm | 0.0~1.0 | Normal=0.0, Warning=0.5, Critical=1.0 |
| 5 | tbt_ratio | 0.0~∞ | `1.0 / actual_throughput / baseline_tps` (baseline=0이면 1.0) |
| 6 | tokens_generated_norm | 0.0~1.0 | `tokens_generated / max_kv_tokens` |
| 7 | is_running | 0 or 1 | `state == Running` ? 1.0 : 0.0 |
| 8 | memory_lossless_min | 0.0~1.0 | `status.memory_lossless_min` 직접 |
| 9 | memory_lossy_min | 0.0~1.0 | `status.memory_lossy_min` 직접 |
| 10~12 | (예약) | 0.0 | 향후 QCF 응답 값 또는 활성 액션 플래그 |

---

## 7. Configuration Schema

### 7-1. policy_config.toml

`36_policy_design.md`에 명시된 Manager의 정책 설정 파일. 이 문서에서는 프로토콜과 직접 연관된 파라미터만 발췌한다.

```toml
[pi_controller]
compute_kp = 1.5
compute_ki = 0.3
compute_setpoint = 0.70    # 목표 kv_cache_utilization (compute 측면)
memory_kp = 2.0
memory_ki = 0.5
memory_setpoint = 0.75     # 목표 kv_cache_utilization (memory 측면)
thermal_kp = 1.0
thermal_ki = 0.2
thermal_setpoint = 0.80
integral_clamp = 2.0

[supervisory]
warning_threshold = 0.4    # PI output이 이 값 초과 시 Warning 모드
critical_threshold = 0.7   # PI output이 이 값 초과 시 Critical 모드
warning_release = 0.25     # Warning 해제 임계값
critical_release = 0.50    # Critical 해제 임계값
hold_time_secs = 4.0       # 히스테리시스 유지 시간

[selector]
latency_budget = 0.5       # 허용 TBT 증가율 (50%)
algorithm = "exhaustive"   # "exhaustive" | "greedy"

[relief_model]
forgetting_factor = 0.995
prior_weight = 5           # 초기 가상 관측 수 (베이지안 prior)
storage_dir = "~/.llm_rs/models"

# 각 액션의 기본 QCF 추정치 (alpha) 및 reversibility
[actions.switch_hw]
alpha = 0.0
reversible = true

[actions.throttle]
alpha = 0.0
reversible = true

[actions.kv_evict_sliding]
alpha = 0.12
reversible = false

[actions.kv_evict_h2o]
alpha = 0.15
reversible = false

[actions.kv_quant_dynamic]
alpha = 0.08
reversible = true

[actions.layer_skip]
alpha = 0.25
reversible = true

[actions.snapkv_compress]
alpha = 0.10
reversible = false

[exclusion_groups]
eviction = ["kv_evict_sliding", "kv_evict_h2o"]
```

### 7-2. Relief Model 저장 포맷

세션 간 누적 학습 데이터. `DegradationEstimator`(`engine/src/core/qcf/estimator.rs`)의 EMA 가중치와 RLS P 행렬을 Manager가 외부 JSON으로 저장한다.

```json
{
    "device_id": "pixel",
    "model_id": "llama3.2-1b",
    "updated_at": "2026-03-20T12:00:00Z",
    "models": {
        "switch_hw": {
            "weights": [[...], [...], [...], [...]],
            "bias": [0.0, 0.0, 0.0, 0.0],
            "p_matrix": [[...]],
            "observation_count": 42
        },
        "kv_evict_sliding": { "...": "..." }
    }
}
```

---

## 8. 타이밍 다이어그램

세션 시작부터 압력 발생, 액션 적용, 해제까지의 전체 흐름:

```
시간 ──────────────────────────────────────────────────────────→

Engine:  [Capability]──[hb]──[hb]──[hb]──[hb]──────[Response]──[hb]──[hb]──[Response]─
                │        │    │    │    │               ▲          │    │        ▲
Manager: [reg]──       [PI]──[PI]──[PI]──[PI]          │        [PI]──[PI]      │
                                          │             │                       │
                                       Warning          │                    Normal
                                          │             │                       │
                                    [Supervisory]       │                [Release cmd]
                                          │             │
                                    [Selector]──────[Directive]
                                          │
                                    (QCF Request → 향후 추가)
```

| 단계 | 설명 |
|------|------|
| `[Capability]` | 연결 직후 Engine이 능력 보고 |
| `[hb]` | ~100ms 간격 Heartbeat |
| `[PI]` | Manager 내부 PI 컨트롤러가 Heartbeat 데이터로 pressure 계산 |
| `[Supervisory]` | PI output이 warning_threshold 초과 → Warning 모드 진입 |
| `[Selector]` | 최적 액션 선택 (alpha 기반 QCF 추정, latency budget 검사) |
| `[Directive]` | 선택된 액션을 `EngineCommand`로 전달 |
| `[Response]` | Engine이 실행 결과 보고 (`CommandResult::Ok` / `Partial` / `Rejected`) |
| `[Release cmd]` | pressure 해소 후 reversible 액션을 `Resume` 또는 반전 명령으로 해제 |

---

## 9. 에러 처리

### 9-1. 타임아웃

| 상황 | 조건 | Manager 행동 |
|------|------|-------------|
| Heartbeat 타임아웃 | 3초 이상 `EngineMessage` 없음 | Engine 연결 끊김 판단, 모든 적용 액션을 내부 상태에서 해제 |
| Directive 응답 타임아웃 | 500ms 내 `CommandResponse` 없음 | 해당 명령 실패로 간주, `seq_id` 재사용 불가 |
| QCF 타임아웃 (향후) | 500ms 내 `QcfResponse` 없음 | 해당 액션 QCF를 `f32::INFINITY`로 가정 → Selector에서 제외 |

### 9-2. Rejected 처리

`CommandResult::Rejected` 수신 시 해당 액션을 일시적으로 후보에서 제외한다. 구체적 정책은 Manager 구현에 위임하지만, 권장 기본값은 3회 연속 선택 skip이다.

### 9-3. 연결 실패

Engine 연결 불가 시 Manager는 자체 PolicyEngine 루프를 계속 실행하되 Emitter 호출을 skip한다. Engine은 Manager 연결 없이도 독립 동작하며 모든 `ResourceLevel`을 `Normal`로 간주한다 (Fail-Safe 원칙, `20_dbus_ipc_spec.md` 8.2절 계승).

---

## 10. 기존 프로토콜과의 호환성

| 기존 (`20_dbus_ipc_spec.md`) | 신규 (이 문서) | 호환 방안 |
|------------------------------|--------------|-----------|
| `SystemSignal` (단방향, 4종) | `ManagerMessage` / `EngineMessage` (양방향) | D-Bus 어댑터: `SystemSignal` → `EngineCommand` 변환 |
| 4개 `Level` enum (Normal~Emergency) | `ResourceLevel` (3단계: Normal/Warning/Critical) + `Suspend` 명령 | Emergency → `Suspend` 명령으로 매핑 |
| `Emitter` trait: `emit(signal)` | `Emitter` trait 확장: `emit_directive(msg)` 추가 | `emit()` 유지, `emit_directive()` 선택적 구현 |
| `SignalListener<T>`: 수신만 | 양방향 transport: 송수신 모두 | `UnixSocketTransport` 확장 필요 |

**legacy_passthrough 모드**: Manager CLI에 `--legacy-passthrough` 플래그를 유지한다. 이 모드에서는 Monitor가 생성한 `SystemSignal`을 변환 없이 바로 Emitter로 전달하여 기존 동작을 유지한다.

`Level` → `ResourceLevel` 매핑 규칙:

| 기존 `Level` | 신규 `ResourceLevel` / 명령 |
|--------------|--------------------------|
| `Normal` | `ResourceLevel::Normal` + reversible 액션 Release |
| `Warning` | `ResourceLevel::Warning` + `SetMemoryLevel { target_ratio: 0.85 }` |
| `Critical` | `ResourceLevel::Critical` + `SetMemoryLevel { target_ratio: 0.50 }` |
| `Emergency` | `EngineCommand::Suspend` + `SetMemoryLevel { target_ratio: 0.25 }` |

---

## 11. shared crate 타입 현황 및 추가 예정

### 11-1. 구현 완료 (`shared/src/lib.rs`)

모두 `#[derive(Serialize, Deserialize, Debug, Clone)]` 적용됨.

| 타입 | 설명 |
|------|------|
| `ResourceLevel` | 3단계 압력 수준 |
| `EngineState` | Idle / Running / Suspended |
| `EngineCapability` | 연결 시 Engine 능력 보고 |
| `EngineStatus` | 주기적 상태 heartbeat |
| `EngineCommand` | Manager → Engine 개별 명령 (6종) |
| `EngineDirective` | `seq_id` + `Vec<EngineCommand>` 배치 |
| `CommandResult` | Ok / Partial / Rejected |
| `CommandResponse` | `seq_id` + `Vec<CommandResult>` |
| `EngineMessage` | Engine→Manager 봉투 enum |
| `ManagerMessage` | Manager→Engine 봉투 enum |

기존 `SystemSignal` 관련 타입(`Level`, `SystemSignal`, `RecommendedBackend`, `ComputeReason`, `EnergyReason`)은 D-Bus 하위 호환을 위해 **유지**된다.

### 11-2. 향후 추가 예정

`36_policy_design.md`의 정책 시스템 구현 시 아래 타입들을 추가한다:

```rust
// ActionCapability 관련
ActionCapability, ActionId, ActionKind, ParamRange

// QCF 조회 관련
QcfRequest, QcfCandidate, QcfResponse, QcfEstimate, ActionParams

// EngineMessage / ManagerMessage에 변형 추가
EngineMessage::QcfResponse(QcfResponse)
ManagerMessage::QcfRequest(QcfRequest)
```

---

## 12. 관련 문서

- `36_policy_design.md` — Hierarchical Policy 시스템 설계 (이 프로토콜의 사용 컨텍스트)
- `20_dbus_ipc_spec.md` — (구) D-Bus IPC 명세 (D-Bus 시그널 이름 및 인터페이스 정의는 유지)
- `27_manager_architecture.md` — Manager 서비스 내부 아키텍처 (3-layer, OCP PolicyEngine)
- `22_resilience_integration.md` — (구) generate.rs 통합 설계 (Engine 측 수신 처리)
- `docs/PROJECT_CONTEXT.md` — 전체 구현 상태 및 개발 치트시트
