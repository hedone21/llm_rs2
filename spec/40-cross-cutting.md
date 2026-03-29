# Cross-cutting Concerns

> **TL;DR**: llm_rs2 시스템의 프로세스 간 공유 관심사를 종합한다.
> Fail-Safety (Manager/Engine 독립성), Shared 크레이트 경계 (IPC 타입만 공유),
> 프로토콜 버전 호환 (serde default 규칙), 로깅 전략 (RUST_LOG 기반),
> 에러 전파 전략 (Manager 비치명적/Engine fail-open), 성능 제약 (타이밍 상수),
> 플랫폼 의존성 (ARM64/NEON, OpenCL, /proc /sys, SELinux TCP fallback),
> 메모리 관리 전략 (mmap, madvise, zero-copy UMA),
> 보안 고려 (로컬 IPC 1:1 신뢰)를 정의한다.

## 1. Purpose and Scope

이 문서는 개별 컴포넌트 스펙(20~33번)이나 프로토콜 스펙(10~12번)에서 다루기 어려운 **시스템 횡단(cross-cutting) 관심사**를 종합한다. 각 관심사는 이미 기존 스펙에서 부분적으로 정의되어 있으며, 이 문서는 흩어진 내용을 한 곳에서 조망하고 추가 지침을 제공한다.

**이 파일이 명세하는 것:**

- Fail-Safety 원칙과 독립성 보증
- Shared 크레이트의 경계와 의존 규칙
- 프로토콜 버전 호환 정책
- 로깅 전략
- 에러 전파 전략
- 성능 관련 타이밍 제약
- 플랫폼 의존성과 이식성
- 메모리 관리 전략
- 보안 모델

**이 파일이 명세하지 않는 것:**

- 개별 컴포넌트 아키텍처 → `20-manager.md` ~ `33-engine-data.md`
- 프로토콜 와이어 포맷, 메시지, 시퀀스 → `10-protocol.md` ~ `12-protocol-sequences.md`
- 불변식 카탈로그 → `41-invariants.md`

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **Cross-cutting Concern** | 단일 컴포넌트에 국한되지 않고 여러 컴포넌트에 걸쳐 적용되는 설계 관심사. |
| **Fail-open** | 외부 의존(Manager 연결 등)이 불가할 때 핵심 기능(추론)을 기본 모드로 계속 수행하는 전략. |
| **Graceful Degradation** | 장애 시 전체 중단 대신 기능을 축소하여 서비스를 유지하는 전략. |
| **Wire Compatibility** | Manager와 Engine이 독립 배포될 때 JSON 메시지의 역직렬화가 실패하지 않는 상태. |
| **UMA** | Unified Memory Architecture. CPU와 GPU가 동일 물리 메모리를 공유하는 아키텍처. |

## 3. Specification

### 3.1 Fail-Safety [CROSS-010 ~ CROSS-013]

**[CROSS-010]** Manager/Engine 독립성 원칙. *(MUST)*

시스템은 항상 2개의 독립 OS 프로세스(Engine, Manager)로 구성된다 (SYS-001, INV-001). 어느 한쪽의 장애가 다른 쪽의 핵심 기능을 중단시켜서는 안 된다.

| 장애 시나리오 | 영향받는 프로세스 | 보장 동작 | 근거 |
|-------------|---------------|----------|------|
| Manager 크래시/연결 끊김 | Engine | 추론 루프 계속 (fail-open). 모든 ResourceLevel을 Normal로 간주. | SYS-050, INV-005 |
| Engine 크래시/연결 끊김 | Manager | 모니터링 루프 계속. Emitter 호출 skip (비치명적 Ok). 재연결 시도. | SYS-051, INV-006 |
| IPC 연결 끊김 (양방향) | 양측 | Graceful degradation: Engine = Resilience 비활성, Manager = emit skip. | SYS-052 |
| Resilience 서브시스템 내부 장애 | Engine | 추론은 기본 모드로 계속. Resilience 실패가 추론 크래시로 전파되지 않음. | SYS-053 |
| Monitor 스레드 panic/무한루프 | Manager | 다른 Monitor 스레드에 전파되지 않음. 각 Monitor는 독립 OS 스레드, 공유 상태 없음. | SYS-054, INV-013 |

**[CROSS-011]** Engine 독립 동작. *(MUST)*

- Manager 미연결 시: `command_executor = None`. Resilience checkpoint 건너뜀. 순수 추론만 수행.
- Manager 연결 끊김 시: 마지막 상태 유지, 추론 계속. MessageLoop thread 종료.
- D-Bus 경로: Emergency Level 수신 시 Engine이 자율적으로 Suspend (SYS-055, ENG-ST-015).

**[CROSS-012]** Manager 독립 동작. *(MUST)*

- Engine 미연결 시: Policy 루프 계속 실행 (Monitor 신호 처리, 압력 계산, 모드 전이).
- `emit_directive()` 호출 시 `ensure_connected()` 실행 -- accept 실패 시 skip, `Ok` 반환.
- 재연결 시 Policy 상태(pressure, mode, estimator) 유지. Engine 캐시(available_devices 등)만 갱신.

**[CROSS-013]** Emergency 자율 대응. *(MUST)*

Emergency는 Manager OperatingMode에 포함되지 않는다. Emergency 대응 경로:

| 경로 | 메커니즘 | 대응 |
|------|---------|------|
| 양방향 프로토콜 (Unix Socket/TCP) | Manager가 Emergency를 pressure=1.0으로 변환하여 Critical Directive + 필요시 Suspend 전송 | Manager가 제어 |
| D-Bus 경로 | DbusTransport가 Emergency SystemSignal을 Suspend EngineCommand로 변환 | Engine 자율 |

도메인별 Emergency 대응 (SYS-055):

| 도메인 | Emergency 대응 | 근거 |
|--------|---------------|------|
| Thermal | 즉시 Suspend | 과열 보호, 하드웨어 손상 방지 |
| Energy | 즉시 Suspend + 새 요청 거부 | 배터리 보호, 전원 차단 방지 |
| Memory | 공격적 eviction (25%) + 새 요청 거부 | OOM killer 회피. Suspend 불필요 |

### 3.2 Shared Crate Boundary [CROSS-020 ~ CROSS-022]

**[CROSS-020]** Shared 크레이트는 IPC 타입만 정의한다. *(MUST)*

```
Engine ──depends──> Shared <──depends── Manager
Engine _|_ Manager   (직접 의존 없음)
```

- INV-010: Engine과 Manager 간 직접 코드 의존 금지. Shared가 유일한 공유 의존성.
- INV-011: Shared는 Engine이나 Manager의 내부 구현에 의존 금지.
- SYS-072: Shared의 외부 의존성은 serde/serde_json만 허용.
- MGR-C02: Manager의 Shared 의존은 IPC 타입에 한정.

**[CROSS-021]** Shared에 정의된 타입 목록. *(MUST)*

| 카테고리 | 타입 |
|---------|------|
| Envelope | ManagerMessage, EngineMessage |
| Directive | EngineDirective, EngineCommand (13종) |
| Response | CommandResponse, CommandResult (3종) |
| Engine 보고 | EngineCapability (5필드), EngineStatus (16필드), QcfEstimate |
| 열거형 | ResourceLevel (3), EngineState (3), Level (4), RecommendedBackend (3), ComputeReason (6), EnergyReason (6) |
| D-Bus 전용 | SystemSignal (4종, externally tagged) |

**[CROSS-022]** Shared 타입의 serde 어노테이션은 와이어 포맷을 결정한다. *(MUST)*

- 모든 주요 enum: `tag = "type"`, `rename_all = "snake_case"` (internally tagged)
- SystemSignal: externally tagged (serde 기본)
- CommandResult: `tag = "status"`, `rename_all = "snake_case"`

### 3.3 Protocol Version Compatibility [CROSS-030 ~ CROSS-032]

**[CROSS-030]** 하위 호환 변경 (backward-compatible). *(MAY)*

| 변경 유형 | 하위 호환 조건 | 예시 |
|----------|-------------|------|
| 새 필드 추가 | `#[serde(default)]` 적용 필수 (INV-028) | EngineStatus에 available_actions 추가 |
| 새 enum variant 추가 | 구 버전 수신측에서 ParseError 발생하며 skip (PROTO-061) | EngineCommand에 새 액션 추가 |
| 기본값이 있는 필드 추가 | `#[serde(default)]`로 생략 시 기본값 적용 | skip_ratio (default 0.0) |

**[CROSS-031]** 비호환 변경 (breaking). *(MUST NOT)*

| 변경 유형 | 영향 | 금지 근거 |
|----------|------|----------|
| 기존 필드 삭제 | 구 버전 피어의 역직렬화 실패 | CON-022 |
| 기존 필드명 변경 | 와이어 포맷 변경 | CON-020 |
| 태그 값(type/status) 변경 | 메시지 라우팅 실패 | CON-020 |
| serde 어노테이션 변경 | 와이어 포맷 변경 | INV-027 |

**[CROSS-032]** 프로토콜 버전 관리 규칙. *(MUST)*

- INV-027: Shared 크레이트의 serde 어노테이션 변경은 프로토콜 버전 변경에 해당.
- INV-028: 새 필드 추가 시 반드시 `#[serde(default)]` 적용.
- Engine과 Manager는 독립 배포 가능(SYS-073). 따라서 와이어 호환은 필수.
- 현재 명시적 버전 협상 프로토콜은 없음. `#[serde(default)]`와 unknown field 무시가 유일한 호환 메커니즘.

### 3.4 Logging Strategy [CROSS-040 ~ CROSS-042]

**[CROSS-040]** RUST_LOG 환경 변수 기반 로깅. *(SHOULD)*

- Manager와 Engine은 별도 프로세스이므로 별도 로그 스트림을 생성한다.
- 구조화 로깅: 타임스탬프, 레벨, 모듈 경로, 메시지.
- 프로토콜 수준 이벤트(ParseError, Disconnected 등)는 warn 레벨로 기록.
- Engine 상태 전이(EngineState 변경)는 info 레벨로 기록.
- Heartbeat, 일상적 Monitor 신호는 debug/trace 레벨.

**[CROSS-041]** Manager 로그 항목. *(SHOULD)*

| 이벤트 | 레벨 | 내용 |
|--------|------|------|
| 모드 전이 | info | `"mode transition: {prev} -> {next}"` |
| Directive 발행 | info | seq_id, commands 요약 |
| Response 수신 | debug | seq_id, results |
| Rejected 응답 | warn | action, reason |
| 연결 상태 전이 | info | Listening/Connected/Disconnected |
| Monitor 장애 | error | Monitor 이름, 에러 내용 |
| Relief observation | debug | action, actual_relief |

**[CROSS-042]** Engine 로그 항목. *(SHOULD)*

| 이벤트 | 레벨 | 내용 |
|--------|------|------|
| EngineState 전이 | info | `"state: {prev} -> {next}"` |
| Directive 수신 | debug | seq_id, commands |
| Command 실행 결과 | debug | command, result |
| ParseError | warn | 프레임 내용 일부 |
| Transport 연결/끊김 | info | transport 이름, 사유 |
| Heartbeat 전송 | trace | 16필드 요약 |

### 3.5 Error Propagation Strategy [CROSS-050 ~ CROSS-052]

**[CROSS-050]** Manager 에러 전파: 비치명적 (emit-and-skip). *(MUST)*

| 에러 유형 | 처리 | 영향 |
|----------|------|------|
| Emitter 쓰기 오류 | state를 Disconnected로 전이, Ok 반환 | Policy 루프 계속 |
| JSON ParseError | warn 로그, 프레임 skip | 연결 유지 |
| Monitor 스레드 panic | 해당 Monitor만 종료, 다른 Monitor 독립 | 도메인 정보 결여 |
| Relief 모델 로드 실패 | 빈 models로 시작, 비치명적 | cold-start |
| config.toml 부재 | `PolicyConfig::default()` 사용 | 기본 동작 |

**[CROSS-051]** Engine 에러 전파: fail-open (추론 계속). *(MUST)*

| 에러 유형 | 처리 | 영향 |
|----------|------|------|
| Transport 연결 실패 | `ConnectionFailed` 에러, Resilience 비활성 | 단독 추론 |
| Transport 끊김 | MessageLoop 종료 | 마지막 상태 유지 |
| JSON ParseError | warn 로그, `continue` (다음 recv) | 연결 유지 |
| Command 실행 실패 | `Rejected` 또는 `Partial` 반환 | 추론 중단 없음 |
| Resilience 서브시스템 panic | 추론 루프가 catch 없이 계속 (SYS-053) | 기본 모드 추론 |

**[CROSS-052]** CommandResult 수준 에러. *(non-normative)*

- `Rejected`/`Partial`은 프로토콜 오류가 아니라 비즈니스 응답이다 (PROTO-065).
- 연결에 영향을 주지 않는다.
- Manager는 로그 기록만 수행한다 (향후 3회 연속 Rejected 시 후보 제외 권장).

### 3.6 Timing Constraints [CROSS-060 ~ CROSS-061]

**[CROSS-060]** 시스템 타이밍 상수 종합. *(MUST/SHOULD)*

| 상수 | 값 | 위치 | 강도 | 근거 |
|------|---|------|------|------|
| Heartbeat 주기 | 1000ms | Engine (하드코딩) | SHOULD | Manager 50ms 폴링 대비 느슨, Engine 부하 최소화 (PROTO-070) |
| Manager recv_timeout | 50ms | Manager 메인 루프 | SHOULD | Monitor 신호 수신 간격 (PROTO-072) |
| MemoryMonitor 폴링 | <=100ms | Manager MemoryMonitor | MUST | OOM 즉각 대응 (MGR-022) |
| 기타 Monitor 폴링 | 1000ms (기본) | Manager | SHOULD | Monitor별 개별 설정 가능 (MGR-022) |
| Observation delay | 3.0초 | Manager Policy | SHOULD | 액션 효과 측정 대기 (SEQ-051) |
| Supervisory hold_time | 4.0초 | Manager Supervisory | SHOULD | 디에스컬레이션 안정화 대기 (MGR-ALG-023) |
| sync_channel 버퍼 | 64 | Manager Reader thread | MUST | 배압 제어, OOM 방지 (PROTO-071) |
| MAX_PAYLOAD_SIZE | 64KB | Engine Transport | SHOULD | 악성/오작동 피어 메모리 가드 (PROTO-012) |
| Client timeout | 60초 | Manager 초기 연결 대기 | SHOULD | CLI --client-timeout (MGR-047) |

**[CROSS-061]** 타이밍 관계 제약. *(SHOULD)*

```
MemoryMonitor 폴링 (<=100ms)
  + recv_timeout (50ms)
  + 직접 매핑 (지연 없음)
  = ~150ms 이내 Supervisory 모드 전이 가능

Observation delay (3.0s) < Supervisory hold_time (4.0s)
  -> 관측 완료 후 디에스컬레이션 판단 가능

Heartbeat (1000ms) vs Observation delay (3.0s)
  -> 관찰 기간 중 ~3개 Heartbeat 수신
```

### 3.7 Platform Dependencies [CROSS-070 ~ CROSS-073]

**[CROSS-070]** ARM64/NEON 의존성. *(MUST)*

- NEON SIMD 최적화는 ARM64에서만 활성화 (INV-002, SYS-022~023).
- `CpuBackendNeon`: `vdotq_s32` 등 NEON 벡터 명령. `#[cfg(target_arch = "aarch64")]`.
- x86_64: AVX2 또는 스칼라 폴백.
- 기타 아키텍처: 스칼라 폴백 (`CpuBackendCommon`).

**[CROSS-071]** OpenCL 의존성. *(MAY)*

- Feature gate: `opencl`. `#[cfg(feature = "opencl")]`.
- 지원 커널: MatMul (F32, F16, Q4_0, Q8_0), RoPE, Softmax (SYS-025).
- Adreno GPU 최적화: `CL_MEM_ALLOC_HOST_PTR` (zero-copy UMA, SYS-026).
- 빌드: `opencl` feature 비활성 시 GPU 코드 전체 제거.

**[CROSS-072]** OS 인터페이스 의존성. *(MUST)*

| 인터페이스 | 용도 | 사용 프로세스 |
|-----------|------|-------------|
| `/proc/meminfo` | 가용/전체 메모리 | Manager (MemoryMonitor) |
| `/proc/stat` | CPU 사용률 (delta) | Manager (ComputeMonitor) |
| `/sys/class/thermal/` | 온도, 스로틀링 | Manager (ThermalMonitor) |
| `/sys/class/power_supply/` | 배터리, 충전 | Manager (EnergyMonitor) |
| Unix Domain Socket | IPC 기본 전송 | Manager (서버), Engine (클라이언트) |
| D-Bus System Bus | 대체 전송 | Manager (Emitter), Engine (Listener) |

**[CROSS-073]** SELinux 대응. *(MAY)*

- Android SELinux 환경에서 Unix socket `bind()` 제한 가능.
- TCP loopback (`127.0.0.1:port`) fallback 제공 (PROTO-031).
- CLI: `--transport tcp:<host:port>` (Manager), `--resilience-transport tcp:<host:port>` (Engine).

### 3.8 Memory Management Strategy [CROSS-080 ~ CROSS-083]

**[CROSS-080]** 모델 가중치 로딩: mmap. *(MUST)*

- Safetensors 파일을 mmap으로 매핑하여 메모리 효율적 로딩.
- OS 페이지 캐시 활용. 실제 접근 시점에 물리 메모리 할당 (demand paging).

**[CROSS-081]** KV 캐시 메모리 관리: madvise. *(SHOULD)*

- `release_unused_pages()`: eviction 후 미사용 KV 캐시 페이지를 OS에 반환.
- `MADV_DONTNEED` (Linux): 페이지를 즉시 해제. 재접근 시 zero-fill.
- `MADV_FREE` (macOS): 페이지를 lazy 해제. 재접근 시 기존 데이터 또는 zero.
- `shrink_to_fit()`: high_water_pos 기반 물리적 KV 버퍼 축소.

**[CROSS-082]** Zero-copy UMA. *(SHOULD, OpenCL + ARM UMA)*

- `CL_MEM_ALLOC_HOST_PTR` (SharedBuffer) 또는 `CL_MEM_USE_HOST_PTR` (MadviseableGPUBuffer).
- CPU와 GPU가 동일 물리 메모리 공유. 데이터 복사 없이 양측 접근.
- MadviseableGPUBuffer: 앱 소유 메모리 + `CL_MEM_USE_HOST_PTR`로 GPU 노출. `is_host_managed=true`이므로 madvise 가능.
- Adreno GPU 핀 문제: `CL_MEM_ALLOC_HOST_PTR` 버퍼는 OS가 메모리 회수 불가 (GPU 핀). MadviseableGPUBuffer의 `shrink_to_fit` 대안으로 물리 메모리 반환.

**[CROSS-083]** 메모리 예산 참고. *(non-normative)*

```
Llama 3.2 1B, seq=2048:
- 모델 가중치 (Q4_0): ~1.2 GB
- KV 캐시 (F16):      ~64 MB  (16 layers x 4 MB)
- KV 캐시 (F32):      ~128 MB
- 워크스페이스:         ~128 KB (공유)
- 총합:               ~1.3-1.4 GB

타겟 디바이스: 4~12 GB RAM
```

### 3.9 Security Model [CROSS-090 ~ CROSS-092]

**[CROSS-090]** 로컬 IPC 1:1 신뢰 모델. *(MUST)*

- Manager와 Engine은 동일 디바이스의 로컬 IPC로만 통신.
- 1:1 단일 클라이언트 연결 (SYS-093, PROTO-041).
- 인증/암호화 메커니즘 없음. 양측이 신뢰할 수 있는 환경을 전제.
- 네트워크 전송 지원 시에도 `127.0.0.1` loopback으로 한정 (PROTO-031).

**[CROSS-091]** 메시지 크기 제한. *(SHOULD)*

- MAX_PAYLOAD_SIZE = 64KB (PROTO-012).
- 악의적이거나 오작동하는 피어로부터의 메모리 고갈 방어.
- Engine Transport에 구현 완료. Manager 측 미적용 (향후 추가 권장).

**[CROSS-092]** Unix Socket 파일 정리. *(MUST)*

- Manager Drop 시 Unix 소켓 파일 삭제 (PROTO-046).
- 소켓 파일 경로는 CLI 설정 가능.

## 4. Alternative Behavior

- **D-Bus 전용 운영**: Unix Socket/TCP 미사용 시 D-Bus 경로만 활성. Emergency 자율 대응(CROSS-013)은 D-Bus 경로에서 Engine이 직접 수행한다.
- **Manager 미사용**: Engine 단독 실행. Resilience 비활성, CacheManager 자체 압력 대응만 수행 (CROSS-011).
- **OpenCL 미사용**: `opencl` feature 비활성 시 CPU 전용 빌드. Zero-copy UMA(CROSS-082)는 해당 없음. Galloc 할당자 사용.

## 5. Constraints

- **[CON-040]** Shared 크레이트의 와이어 포맷 호환을 보장하기 위해 CROSS-031에 정의된 비호환 변경을 수행해서는 안 된다. *(MUST NOT)*
- **[CON-041]** 모든 Manager 에러 경로는 비치명적이어야 한다. Policy 루프가 패닉으로 종료되어서는 안 된다. *(MUST NOT)*
- **[CON-042]** 모든 Engine 에러 경로는 fail-open이어야 한다. Resilience 장애가 추론 루프를 크래시시켜서는 안 된다. *(MUST NOT)*

## 6. Examples

### 예시 1: Manager 크래시 후 Engine 독립 동작

```
시간  | 이벤트                              | Engine 상태
------|-------------------------------------|---------------------------
t=0   | 정상 동작: Heartbeat + Directive     | Running, Degraded
t=5   | Manager 프로세스 크래시               | MessageLoop EOF 감지
t=5+  | Engine 내부: command_executor 해제   | Running, Normal (기본 모드)
t=6   | 추론 계속 (Resilience 비활성)          | Running, Normal
t=30  | Manager 재시작, 새 소켓 bind          | (Engine 인지 못함)
t=31  | Engine 재시작 또는 재연결 시           | Capability 재전송, 정상 복귀
```

### 예시 2: 프로토콜 하위 호환 확장

```
v1 Engine (필드 11개):
  {"type":"heartbeat", ...(11 필드)...}

v2 Manager (필드 16개 기대):
  available_actions: #[serde(default)] -> []
  active_actions:    #[serde(default)] -> []
  eviction_policy:   #[serde(default)] -> ""
  kv_dtype:          #[serde(default)] -> ""
  skip_ratio:        #[serde(default)] -> 0.0

결과: v2 Manager가 v1 Engine의 메시지를 정상 역직렬화.
      누락 필드는 기본값으로 채워짐.
```

### 예시 3: 타이밍 관계 -- Memory Emergency 대응

```
t=0ms    MemoryMonitor: /proc/meminfo 폴링 -> available < critical
t=0ms    MemoryMonitor: SystemSignal(MemoryPressure, Critical) 전송
t=50ms   Manager 메인 루프: recv_timeout(50ms) 만료, 신호 수신
t=50ms   PI/Supervisory: 즉시 Critical 모드 전이
t=50ms   ActionSelector: KvEvictH2o(0.48) + Throttle(30ms) 선택
t=51ms   Emitter: Directive(seq=N) 전송
t=~150ms Engine: poll() -> ExecutionPlan 실행
         -> ~150ms 이내 대응 완료
```

## 7. Rationale (non-normative)

### 왜 Fail-Safety를 별도 섹션으로 종합하는가

Fail-Safety 원칙은 00-overview(SYS-050~055), 01-architecture(INV-005~006), 10-protocol(PROTO-060~063), 20-manager(MGR-C01), 30-engine(ENG-070)에 분산되어 있다. 각 스펙은 자신의 컴포넌트 관점에서 기술하므로, 시스템 전체의 장애 대응 그림을 파악하려면 5개 문서를 교차 참조해야 한다. 이 문서에서 한 곳에 종합함으로써 장애 시나리오별 보장을 일목요연하게 확인할 수 있다.

### 왜 명시적 버전 협상이 없는가

현재 Manager와 Engine은 동일 소스에서 빌드된 바이너리를 전제한다. `#[serde(default)]`와 unknown field skip이 충분한 하위 호환을 제공한다. 향후 독립 릴리스 주기가 도입되면 Capability에 protocol_version 필드를 추가하는 것을 검토한다 (Open Issue O-02).

### 왜 madvise를 zero-copy UMA에서 분리하는가

`CL_MEM_ALLOC_HOST_PTR`로 할당된 GPU 핀 메모리는 OS가 회수할 수 없다. 반면 `CL_MEM_USE_HOST_PTR`(MadviseableGPUBuffer)은 앱 소유 메모리이므로 madvise가 적용 가능하다. 두 전략의 메모리 반환 특성이 상이하므로 별도 항목으로 기술한다.

### 실측 relief와 설계 의도의 괴리

01-architecture SYS-095의 도메인 매핑은 설계 의도이다. 실측 profile(JOURNAL 세션 8)에서는 SwitchHw만 cross-domain(Compute+Thermal) 효과가 유의미하게 확인되었으며, Eviction/Offload의 Memory relief는 물리 메모리 수준에서 null/negligible이었다. ReliefEstimator의 온라인 학습(SYS-019)이 이 괴리를 runtime에 보정하는 메커니즘이다.
