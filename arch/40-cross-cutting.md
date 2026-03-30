# Cross-cutting Concerns -- Architecture

> spec/40-cross-cutting.md의 구현 상세.

## 코드 매핑

### 3.1 Fail-Safety

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| CROSS-010 (독립성) | Cargo.toml 의존 구조 | Engine, Manager = 별도 바이너리 (별도 OS 프로세스) | Shared가 유일한 공유 의존성 |
| CROSS-011 (Engine 독립) | `engine/src/bin/generate.rs` | `command_executor = None` 시 resilience checkpoint skip, 순수 추론 | Manager 연결 끊김 시 MessageLoop 종료 |
| CROSS-011 | `engine/src/resilience/executor.rs` | `CommandExecutor` — `poll()` 내부에서 Heartbeat 자동 전송 | Manager 연결 끊김 후에도 추론 계속 |
| CROSS-012 (Manager 독립) | `manager/src/pipeline.rs` | `emit_directive()` → `ensure_connected()` — accept 실패 시 skip, Ok 반환 | Policy 루프 계속 |
| CROSS-012 | `manager/src/emitter/unix_socket.rs`, `manager/src/emitter/dbus.rs` | Emitter — 쓰기 오류 시 Disconnected 전이, Ok 반환 | |
| CROSS-013 (Emergency) | `engine/src/resilience/dbus_transport.rs` | D-Bus 경로: Emergency SystemSignal → Suspend EngineCommand | Engine 자율 대응 |
| CROSS-013 | `engine/src/resilience/strategy/memory.rs` | MemoryStrategy: Emergency → Evict(0.25) + RejectNew | |
| CROSS-013 | `engine/src/resilience/strategy/thermal.rs` | ThermalStrategy: Emergency → Suspend | |
| CROSS-013 | `engine/src/resilience/strategy/energy.rs` | EnergyStrategy: Emergency → Suspend + RejectNew | |

### 3.2 Shared Crate Boundary

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| CROSS-020 | `engine/Cargo.toml`, `manager/Cargo.toml`, `shared/Cargo.toml` | Engine→Shared, Manager→Shared 의존만 존재 | Engine↔Manager 직접 의존 없음 |
| CROSS-021 | `shared/src/lib.rs` | ManagerMessage, EngineMessage, EngineDirective, EngineCommand, CommandResponse, CommandResult, EngineCapability, EngineStatus, QcfEstimate, ResourceLevel, EngineState, Level, SystemSignal 등 | |
| CROSS-022 | `shared/src/lib.rs` | serde 어노테이션: `tag="type"`, `rename_all="snake_case"` (internally tagged) | SystemSignal은 externally tagged |

### 3.3 Protocol Version Compatibility

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| CROSS-030 | `shared/src/lib.rs` | 새 필드에 `#[serde(default)]` 적용 | EngineStatus 5개 필드 등 |
| CROSS-031 | (정적 보증) | 기존 필드 삭제/변경 금지 — 코드 리뷰로 보장 | |
| CROSS-032 | `shared/Cargo.toml` | 외부 의존성: serde, serde_json만 | |

### 3.4 Logging Strategy

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| CROSS-040 | `engine/src/bin/generate.rs`, `manager/src/main.rs` | `env_logger` 또는 `RUST_LOG` 환경 변수 | 별도 프로세스 → 별도 로그 스트림 |
| CROSS-041 | `manager/src/pipeline.rs`, `manager/src/supervisory.rs` | `log::info!`, `log::warn!` 등 | 모드 전이, Directive 발행, Response 수신 |
| CROSS-042 | `engine/src/resilience/executor.rs`, `engine/src/resilience/transport.rs` | `log::info!`, `log::warn!` 등 | EngineState 전이, ParseError, Transport 연결 |

### 3.5 Error Propagation Strategy

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| CROSS-050 (Manager) | `manager/src/emitter/unix_socket.rs` | 쓰기 오류 → Disconnected, Ok 반환 | Policy 루프 계속 |
| CROSS-050 | `manager/src/channel/unix_socket.rs`, `manager/src/channel/tcp.rs` | JSON ParseError → warn 로그, frame skip | |
| CROSS-051 (Engine) | `engine/src/resilience/transport.rs` | Transport 끊김 → MessageLoop 종료 | 마지막 상태 유지, 추론 계속 |
| CROSS-051 | `engine/src/resilience/executor.rs` | Command 실행 실패 → Rejected/Partial 반환 | 추론 중단 없음 |
| CROSS-052 | `shared/src/lib.rs` | `CommandResult { Completed, Rejected, Partial }` | 비즈니스 응답, 연결 미영향 |

### 3.6 Timing Constraints

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| CROSS-060 (Heartbeat) | `engine/src/resilience/executor.rs` | 1000ms 간격 하드코딩 | poll() 내부 타이머 |
| CROSS-060 (recv_timeout) | `manager/src/pipeline.rs` | 50ms recv_timeout | 메인 루프 |
| CROSS-060 (MemoryMonitor) | `manager/src/monitor/memory.rs` | <=100ms 폴링 | `/proc/meminfo` |
| CROSS-060 (sync_channel) | `manager/src/channel/unix_socket.rs:152`, `manager/src/channel/tcp.rs:144` | `mpsc::sync_channel(64)` | 배압 제어 |
| CROSS-060 (MAX_PAYLOAD) | `engine/src/resilience/transport.rs:245` | `MAX_PAYLOAD_SIZE = 64 * 1024` | |
| CROSS-061 | (아키텍처 제약) | 타이밍 관계: MemoryMonitor(100ms) + recv_timeout(50ms) = ~150ms 대응 | |

### 3.7 Platform Dependencies

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| CROSS-070 (NEON) | `engine/src/backend/cpu/neon.rs` | `#[cfg(target_arch = "aarch64")]` | CpuBackendNeon |
| CROSS-070 (x86) | `engine/src/backend/cpu/x86.rs` | `#[cfg(target_arch = "x86_64")]` | CpuBackendAVX2 |
| CROSS-071 (OpenCL) | `engine/src/backend/opencl/mod.rs` | `#[cfg(feature = "opencl")]` | OpenCLBackend |
| CROSS-071 | `engine/kernels/*.cl` | ~80 kernel 파일 | MatMul, RoPE, Softmax 등 |
| CROSS-072 (OS 인터페이스) | `manager/src/monitor/memory.rs` | `/proc/meminfo` 파싱 | |
| CROSS-072 | `manager/src/monitor/compute.rs` | `/proc/stat` CPU delta | |
| CROSS-072 | `manager/src/monitor/thermal.rs` | `/sys/class/thermal/` | |
| CROSS-072 | `manager/src/monitor/energy.rs` | `/sys/class/power_supply/` | |
| CROSS-072 | `manager/src/channel/unix_socket.rs` | Unix Domain Socket — Manager 서버 | |
| CROSS-072 | `engine/src/resilience/transport.rs` | Unix Domain Socket — Engine 클라이언트 | |
| CROSS-073 (SELinux) | `manager/src/channel/tcp.rs` | TCP loopback fallback `127.0.0.1:port` | CLI: `--transport tcp:<host:port>` |

### 3.8 Memory Management Strategy

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| CROSS-080 (mmap) | `engine/src/models/` | Safetensors mmap 로딩 | OS 페이지 캐시, demand paging |
| CROSS-081 (madvise) | `engine/src/core/kv_cache.rs:1015` | `madvise_dontneed()` — `MADV_DONTNEED` (Linux) | high_water_pos 범위 제한 |
| CROSS-081 | `engine/src/core/kv_cache.rs:363` | `shrink_to_fit()` — 재할당으로 물리 메모리 해제 | dynamic cache 전용 |
| CROSS-082 (zero-copy UMA) | `engine/src/buffer/unified_buffer.rs` | `CL_MEM_ALLOC_HOST_PTR` (SharedBuffer) | GPU pin — madvise 무효 |
| CROSS-082 | `engine/src/buffer/madviseable_gpu_buffer.rs` | `CL_MEM_USE_HOST_PTR` (MadviseableGPUBuffer) | is_host_managed=true — madvise 가능 |

### 3.9 Security Model

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| CROSS-090 (1:1 신뢰) | `manager/src/channel/unix_socket.rs` | 단일 accept, 1:1 클라이언트 | 인증/암호화 없음 |
| CROSS-091 (크기 제한) | `engine/src/resilience/transport.rs:245` | `MAX_PAYLOAD_SIZE = 64KB` | Manager 측 미적용 |
| CROSS-092 (소켓 정리) | `manager/src/channel/unix_socket.rs` | Drop 시 소켓 파일 삭제 | |

## Config

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| (하드코딩) Heartbeat 주기 | u64 | 1000ms | CROSS-060 |
| (하드코딩) recv_timeout | Duration | 50ms | CROSS-060 |
| (하드코딩) sync_channel 버퍼 | usize | 64 | CROSS-060 |
| (하드코딩) MAX_PAYLOAD_SIZE | u32 | 64KB | CROSS-060 |

## CLI

| 플래그 | 설명 | spec/ 근거 |
|--------|------|-----------|
| `--transport` (Manager) | 전송 방식: unix/tcp | CROSS-073 |
| `--resilience-transport` (Engine) | 전송 방식: dbus/unix/tcp | CROSS-073 |
| `--client-timeout` (Manager) | 초기 연결 대기 (기본 60초) | CROSS-060 |