# System Architecture -- Architecture

> spec/01-architecture.md의 구현 상세.

## 코드 매핑

### 3.1 Component Decomposition [SYS-070 ~ SYS-079]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SYS-070 | `Cargo.toml` | `members = ["engine", "shared", "manager"]` | |
| SYS-071 | `engine/Cargo.toml`, `manager/Cargo.toml` | 양측 `llm_shared` path 의존 | Engine ⊥ Manager |
| SYS-072 | `shared/Cargo.toml` | 의존: `serde`, `serde_json` only | |
| INV-010 | `Cargo.toml` | 크레이트 의존 그래프에서 Engine↔Manager 경로 없음 | |
| INV-011 | `shared/src/lib.rs` | 내부 Engine/Manager 타입 미참조 | |

### 3.2 Engine Internal Architecture [SYS-080 ~ SYS-085]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SYS-080 | `engine/src/lib.rs` | 모듈 선언 — `core`, `backend`, `models`, `layers`, `memory`, `buffer`, `resilience` | |
| SYS-081 | `engine/src/core/backend.rs` | `pub trait Backend` (17+ ops) | matmul, softmax, RoPE 등 |
| SYS-082 | `engine/src/core/kv_cache.rs` | `pub trait KVCacheOps` | KVCache, KiviCache, OffloadKVCache |
| SYS-083 | `engine/src/core/pressure/mod.rs` | `CachePressurePipeline`, `CachePressureHandler` trait | |
| SYS-083 | `engine/src/core/cache_manager.rs` | `CacheManager` — Pipeline 래퍼 | |
| SYS-084 | `engine/src/resilience/transport.rs` | `pub trait Transport` | connect, recv, send |
| SYS-085 | `engine/src/resilience/executor.rs` | `CommandExecutor` | ManagerMessage → ExecutionPlan |
| SYS-085a | `engine/src/resilience/manager.rs` | `ResilienceManager` | Strategy 기반 (D-Bus 경로) |
| INV-012 | `engine/src/core/backend.rs` | Backend trait이 유일한 hw 추상화점 | |

#### Engine 서브시스템 → 코드 모듈 매핑

| 서브시스템 | 주요 코드 경로 |
|-----------|--------------|
| Model | `engine/src/models/llama/llama_model.rs` |
| Core | `engine/src/core/` (tensor, buffer, math_utils, sampling, shape) |
| Backend | `engine/src/backend/cpu/`, `engine/src/backend/opencl/` |
| KV Cache | `engine/src/core/kv_cache.rs` |
| Cache Management | `engine/src/core/cache_manager.rs`, `engine/src/core/pressure/` |
| Resilience | `engine/src/resilience/` (executor, transport, manager, strategy) |
| QCF | `engine/src/core/qcf/` |
| Eval | `engine/src/bin/generate.rs` (eval loop) |

#### Transport trait 구현체 → 코드 파일

| 구현체 | 코드 위치 |
|--------|----------|
| UnixSocketTransport | `engine/src/resilience/transport.rs` |
| TcpTransport | `engine/src/resilience/transport.rs` |
| DbusTransport | `engine/src/resilience/dbus_transport.rs` |
| MockTransport | `engine/src/resilience/transport.rs` |

#### KV Cache 구현체 → 코드 파일

| 구현체 | 코드 위치 |
|--------|----------|
| KVCache | `engine/src/core/kv_cache.rs` |
| KiviCache | `engine/src/core/kivi_cache.rs` |
| OffloadKVCache | `engine/src/core/offload_kv_cache.rs` |

#### CachePressureHandler 구현체

| Handler | 코드 위치 | 상태 |
|---------|----------|------|
| EvictionHandler | `engine/src/core/pressure/eviction_handler.rs` | 활성 |
| D2OHandler | `engine/src/core/pressure/d2o_handler.rs` | 활성 |
| SwapHandler | `engine/src/core/pressure/swap_handler.rs` | 활성 |
| QuantizeHandler | `engine/src/core/pressure/quantize_handler.rs` | 활성 (간접) |
| MergeHandler | `engine/src/core/pressure/merge_handler.rs` | 스텁 |
| SparseHandler | `engine/src/core/pressure/sparse_handler.rs` | 스텁 |

### 3.3 Manager Internal Architecture [SYS-086 ~ SYS-089]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SYS-086 | `manager/src/main.rs` | 3-layer: Monitor → Policy → Emitter | |
| SYS-087 | `manager/src/monitor/` | Monitor trait + 도메인별 구현 | |
| SYS-088 | `manager/src/pipeline.rs` | `HierarchicalPolicy` (PI + Supervisory + Selector) | |
| SYS-088a | `manager/src/selector.rs` | `ActionSelector` — cross-domain 조합 탐색 | |
| SYS-089 | `manager/src/emitter/` | `Emitter` trait + 구현체 | |
| INV-013 | `manager/src/main.rs` | 각 Monitor는 독립 `std::thread::spawn` | |

#### Manager Monitor → 코드 파일

| Monitor | 코드 위치 |
|---------|----------|
| MemoryMonitor | `manager/src/monitor/memory.rs` |
| ThermalMonitor | `manager/src/monitor/thermal.rs` |
| ComputeMonitor | `manager/src/monitor/compute.rs` |
| EnergyMonitor | `manager/src/monitor/energy.rs` |
| ExternalMonitor | `manager/src/monitor/external.rs` |

#### Manager Emitter → 코드 파일

| Emitter | 코드 위치 |
|---------|----------|
| UnixSocketEmitter (Channel) | `manager/src/channel/unix_socket.rs` |
| TcpChannel | `manager/src/channel/tcp.rs` |
| DbusEmitter | `manager/src/emitter/dbus.rs` |

#### Manager Policy → 코드 파일

| 컴포넌트 | 코드 위치 |
|---------|----------|
| PI Controller | `manager/src/pi_controller.rs` |
| Supervisory | `manager/src/supervisory.rs` |
| ActionSelector | `manager/src/selector.rs` |
| ActionRegistry | `manager/src/action_registry.rs` |
| ReliefEstimator | `manager/src/relief/mod.rs`, `manager/src/relief/linear.rs` |
| HierarchicalPolicy | `manager/src/pipeline.rs` |
| PolicyConfig | `manager/src/config.rs` |
| FeatureVector/Types | `manager/src/types.rs` |

### 3.4 IPC Topology [SYS-090 ~ SYS-094]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SYS-090 | `shared/src/lib.rs` | `ManagerMessage` (M→E), `EngineMessage` (E→M) | |
| SYS-092 | `engine/src/resilience/transport.rs` | `write_frame()`, `read_frame()` — 4B BE length + JSON | |
| SYS-092 | `manager/src/channel/unix_socket.rs` | `write_manager_message()`, `read_engine_message()` | |
| SYS-093 | `manager/src/channel/unix_socket.rs` | 단일 `TcpListener::accept()` | 1:1 연결 |
| SYS-094 | `engine/src/resilience/dbus_transport.rs` | D-Bus System Bus 수신 | `org.llm.Manager1` |
| INV-014 | `manager/src/pipeline.rs:65-68` | `static SEQ_COUNTER: AtomicU64::new(1)`, `fetch_add(1, Relaxed)` | |
| INV-015 | `engine/src/resilience/executor.rs` | Capability 세션당 1회 전송 | |

### 3.5 Action Pool [SYS-095 ~ SYS-099]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SYS-095 | `manager/src/action_registry.rs` | `ActionRegistry` — 액션 메타데이터 | |
| SYS-096 | `manager/src/config.rs` | `PolicyConfig.exclusion_groups` (TOML HashMap) | |
| SYS-098 | `engine/src/core/qcf/` | lossy action → QcfMetric 생성 | |
| SYS-099 | `manager/src/selector.rs` | 모드별 허용 액션 제약 | Warning: lossless only |
| INV-016 | `manager/src/selector.rs` | 배타 그룹 검증 | |

### 3.6 Threading Model

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| INV-018 | `engine/src/bin/generate.rs` | 단일 스레드 추론 루프 | |
| (스레딩) | `manager/src/main.rs` | Monitor: `thread::spawn` × 4~5, Main: Policy loop | |
| (동기화) | `manager/src/channel/unix_socket.rs:152` | `sync_channel(64)` — Reader → Main | 배압 제어 |
| (동기화) | `manager/src/main.rs:58` | `static SHUTDOWN: AtomicBool` | SIGINT/SIGTERM |

## Config

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `policy.exclusion_groups` | HashMap<String, Vec<String>> | `{}` (빈 맵) | SYS-096 |
| (기타 PolicyConfig 키는 arch/20-manager.md 또는 arch/22-manager-algorithms.md에서 상세화) | | | |

## CLI

(CLI 플래그는 arch/00-overview.md에서 통합 관리)