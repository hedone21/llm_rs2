# System Architecture

> **TL;DR**: llm_rs2는 3개의 Cargo 크레이트(Engine, Manager, Shared)로 구성된 workspace이다. Engine은 추론 전담, Manager는 리소스 모니터링 전담 프로세스이며, Shared가 양측의 IPC 인터페이스를 정의한다. Manager ↔ Engine 간 통신은 Unix Socket(기본) 또는 D-Bus 위의 length-prefixed JSON으로 수행된다. Engine 내부는 Model, Core, KV Cache, Cache Management, Resilience, QCF, Eval, Backend 서브시스템으로 분해되고, Manager 내부는 Monitor → Policy → Emitter 3-layer 파이프라인을 따른다. Policy의 Action Selector는 3개 도메인(compute, memory, thermal)의 압력을 동시에 고려하는 cross-domain 조합 탐색을 수행하며, ReliefEstimator가 온라인 학습으로 액션 효과를 예측한다. 8개의 등록 액션(3 lossless + 5 lossy)이 Action Pool을 구성하며, 상호 배타 규칙과 QCF 보고 의무가 적용된다.

## 1. Purpose and Scope

이 문서는 시스템의 컴포넌트 분해와 컴포넌트 간 경계를 정의한다. 이 문서를 읽으면 각 컴포넌트가 무엇을 하고, 어떤 인터페이스로 연결되는지, 어떤 스레딩 모델을 따르는지 알 수 있다.

**이 파일이 명세하는 것:**

- 3-crate 분해와 의존 관계 (Component Decomposition)
- Engine 내부 서브시스템 구조 (상위 수준)
- Manager 내부 3-layer 구조 (상위 수준)
- Manager ↔ Engine IPC 토폴로지 (연결 모델, 메시지 흐름)
- Action Pool 체계 (액션 분류, 상호 배타, QCF 연동)
- 스레딩 모델과 데이터 흐름

**이 파일이 명세하지 않는 것:**

- 프로토콜 와이어 포맷 상세 → `10-protocol.md`
- 메시지별 필드 정의 → `11-protocol-messages.md`
- 정규 상호작용 시퀀스 → `12-protocol-sequences.md`
- Manager 알고리즘 상세 (PI, Selector, Relief) → `22-manager-algorithms.md`
- Engine 알고리즘 상세 (eviction, quantization) → `32-engine-algorithms.md`
- 개별 상태 머신 → `21-manager-state.md`, `31-engine-state.md`

## 2. Definitions

### 아키텍처 용어

| 용어 | 정의 |
|------|------|
| **Crate** | Rust 컴파일 단위. 바이너리 또는 라이브러리 형태. |
| **Workspace** | 여러 크레이트를 단일 프로젝트로 관리하는 Cargo 구조. 공통 의존성 해석과 빌드 프로파일을 공유한다. |
| **Feature Gate** | 컴파일 타임에 특정 기능을 활성화/비활성화하는 조건부 컴파일 메커니즘. |

### 컴포넌트 용어

| 용어 | 정의 |
|------|------|
| **Engine Crate** | `llm_rs2` 패키지. LLM 추론 엔진 바이너리를 생성한다. |
| **Manager Crate** | `llm_manager` 패키지. 리소스 모니터링 서비스 바이너리를 생성한다. |
| **Shared Crate** | `llm_shared` 패키지. IPC 메시지 타입 정의 라이브러리. |

### 계층 용어

| 용어 | 정의 |
|------|------|
| **Monitor Layer** | Manager 내부 첫 번째 계층. 4+1개 병렬 스레드에서 OS 센서를 수집하여 SystemSignal을 생성한다. |
| **Policy Layer** | Manager 내부 두 번째 계층. PI Controller → Supervisory → Action Selector 파이프라인으로 구성된다. |
| **Emitter Layer** | Manager 내부 세 번째 계층. 생성된 디렉티브를 Engine에 전송한다. |
| **Resilience Subsystem** | Engine 내부 서브시스템. Manager로부터 수신한 디렉티브를 해석하고, 전략 패턴으로 액션을 실행한다. |

### IPC 용어

| 용어 | 정의 |
|------|------|
| **Unix Socket Channel** | Unix 도메인 소켓 기반 양방향 통신 채널. 기본 전송 매체. |
| **D-Bus System Bus** | Linux D-Bus 시스템 버스. 단방향 전송 매체 (SystemSignal 전송 지원). |
| **Wire Format** | 4바이트 BE u32 길이 접두사 + UTF-8 JSON 페이로드. |

### 스레딩 용어

| 용어 | 정의 |
|------|------|
| **Main Thread** | 프로세스의 메인 실행 흐름. Engine에서는 추론 루프, Manager에서는 Policy 루프를 실행한다. |
| **Monitor Thread** | Manager에서 각 도메인 센서를 수집하는 전용 스레드. |
| **SignalListener Thread** | Engine에서 D-Bus 신호를 수신하는 선택적 스레드. |
| **Reader Thread** | 양방향 채널에서 Engine → Manager 메시지를 비동기적으로 읽는 스레드. |

## 3. Specification

### 3.1 Component Decomposition [SYS-070 ~ SYS-079]

**[SYS-070]** 시스템은 3개의 Cargo 크레이트로 구성된 workspace이다: *(MUST)*

| 크레이트 | 패키지명 | 유형 | 역할 |
|---------|---------|------|------|
| engine | `llm_rs2` | 바이너리 | LLM 추론 엔진 — 모델 로딩, forward pass, KV 캐시, 백엔드 연산, Resilience 서브시스템 |
| manager | `llm_manager` | 바이너리 | 리소스 모니터 서비스 — 센서 수집, 정책 평가, 디렉티브 전송 |
| shared | `llm_shared` | 라이브러리 | 공유 타입 라이브러리 — IPC 메시지 타입 정의 |

**[SYS-071]** 크레이트 간 의존 관계는 다음을 따른다: *(MUST)*

```
Engine ──depends──→ Shared
Manager ──depends──→ Shared
Engine ⊥ Manager   (직접 의존 없음)
```

- Engine은 Shared에 의존한다.
- Manager는 Shared에 의존한다.
- Engine과 Manager는 서로를 직접 의존하지 않는다.

**[SYS-072]** Shared 크레이트는 외부 의존성을 최소화해야 한다. IPC 직렬화를 위한 serde/serde_json만 필수 의존성으로 허용한다. *(SHOULD)*

**[SYS-073]** 각 바이너리 크레이트(engine, manager)는 독립적으로 빌드하고 배포할 수 있어야 한다. *(MUST)*

#### 불변식

- **[INV-010]** Engine과 Manager 간에 직접적인 코드 의존이 존재해서는 안 된다. Shared가 유일한 공유 의존성이다. *(MUST NOT)*
- **[INV-011]** Shared 크레이트는 Engine이나 Manager의 내부 구현에 의존해서는 안 된다. *(MUST NOT)*

### 3.2 Engine Internal Architecture [SYS-080 ~ SYS-085]

**[SYS-080]** Engine은 다음 서브시스템으로 분해된다: *(MUST)*

| 서브시스템 | 역할 | 주요 추상화 |
|-----------|------|-----------|
| **Model** | 모델 로딩, 레이어 실행, forward pass | 모델 정의, 레이어 가중치 |
| **Core** | 텐서 연산, 메모리 관리의 기반 | Tensor, Buffer trait, Memory trait |
| **Backend** | 하드웨어별 연산 디스패치 | Backend trait |
| **KV Cache** | KV 캐시 구현체 관리 | KVCacheOps trait, PrefetchableCache trait |
| **Cache Management** | 캐시 압력 대응 파이프라인 | CacheManager, CachePressurePipeline |
| **Resilience** | Manager 디렉티브 수신 및 액션 실행 | ResilienceManager, Transport trait, ResilienceStrategy trait |
| **QCF** | 품질 비용 추적 | QcfMetric, DegradationEstimator |
| **Eval** | 추론 루프 실행 | eval_loop, StepHook |

> **참고**: 위 8개 서브시스템은 engine 크레이트의 22+ 코드 모듈을 논리적으로 그루핑한 것이다. 예를 들어 Core 서브시스템은 tensor, buffer, memory, math_utils, sampling, shape 등 다수 모듈을 포함한다. 개별 모듈의 상세는 33-engine-data.md에서 기술한다.

**[SYS-081]** Backend trait은 모든 하드웨어 가속기 연산의 유일한 추상화점이다. 모든 텐서 연산(matmul, softmax, RoPE 등)은 Backend trait을 통해 디스패치된다. *(MUST)*

Backend trait이 정의하는 연산 범주:

| 범주 | 연산 |
|------|------|
| 행렬 연산 | matmul, matmul_transposed |
| 활성화/정규화 | silu_mul, gelu_tanh_mul, rms_norm, softmax |
| 위치 인코딩 | rope_inplace |
| 어텐션 | attention_gen (선택적 기본 구현) |
| 메모리 | copy_from, copy_into, read_buffer, write_buffer, cast, gather, buffer_shift, copy_slice |
| 산술 | add_assign, scale, add_row_bias |

**[SYS-082]** KVCacheOps trait은 모든 KV 캐시 구현의 공통 인터페이스이다. 다음 구현체가 존재한다: *(MUST)*

| 구현체 | 설명 | 메모리 레이아웃 |
|--------|------|---------------|
| KVCache | 표준 인메모리 캐시 (F32/F16/Q4_0). Eviction 지원. | SeqMajor 또는 HeadMajor |
| KiviCache | KIVI 다중 비트 압축 (Q2/Q4/Q8 + FP32 잔차) | HeadMajor |
| OffloadKVCache | 레이어별 디스크 오프로드 + prefetch 파이프라인 | SeqMajor 전용 |

KVCacheOps trait이 정의하는 인터페이스:

| 메서드 | 역할 |
|--------|------|
| current_pos / set_current_pos | 현재 물리 슬롯 위치 관리 |
| capacity | 최대 토큰 용량 |
| kv_heads / head_dim | KV 헤드 수, 헤드 차원 |
| layout | 메모리 레이아웃 (SeqMajor / HeadMajor) |
| kv_dtype | 데이터 타입 |
| memory_usage_bytes | 현재 메모리 사용량 |
| update | 새 K/V 텐서를 캐시에 기록 |
| get_view | 어텐션 연산을 위한 K/V 텐서 뷰 반환 |
| needs_attn_scores / set_attn_scores | 어텐션 스코어 피드백 (importance 기반 eviction용) |

> 기본 구현이 제공되는 선택적(optional) 메서드 3개가 추가로 존재한다: `get_buffers_mut` (K/V 텐서 직접 접근), `advance_pos` (위치 n만큼 전진), `ensure_capacity` (최소 토큰 용량 보장).

**[SYS-083]** CacheManager는 CachePressurePipeline을 통해 캐시 압력에 대응한다. 파이프라인은 여러 Handler를 순차 실행하되, 각 Handler는 자신의 최소 활성 레벨(min_level) 이상일 때만 실행된다. *(MUST)*

파이프라인 Handler 목록:

| Handler | 역할 | 상태 |
|---------|------|------|
| EvictionHandler | 기존 EvictionPolicy를 어댑터 패턴으로 래핑. H2O, SlidingWindow, StreamingLLM 등의 eviction 실행. | 활성 |
| D2OHandler | H2O 변형. Eviction 시 토큰 병합 보상 (scatter-reduce merge). | 활성 |
| SwapHandler | LRU 전략으로 오래된 토큰을 디스크로 오프로드. | 활성 |
| QuantizeHandler | 압력 수준에 따라 KIVI 양자화 비트 수 조절 (Normal=유지, Warning=8bit, Critical=4bit, Emergency=2bit). KiviCache 전용; 표준 KVCache에서는 NoOp. | 활성 (간접) |
| MergeHandler | 유사 인접 토큰 병합 (cosine similarity 기반). | 스텁 |
| SparseHandler | 스파스 어텐션 마스크 적용. KV 데이터 미수정. | 스텁 |

**[SYS-084]** Resilience 서브시스템은 Transport trait을 통해 Manager와 통신한다. Transport trait은 연결, 수신, 전송 인터페이스를 추상화한다. *(MUST)*

Transport trait 구현체:

| 구현체 | 전송 매체 | 방향 | 용도 |
|--------|----------|------|------|
| UnixSocketTransport | Unix 도메인 소켓 | 양방향 | 기본 (Android/Linux) |
| TcpTransport | TCP 소켓 | 양방향 | OS 비의존 |
| DbusTransport | D-Bus 시스템 버스 | 단방향 (수신만) | Android/Linux 시스템 통합 |
| MockTransport | mpsc 채널 | 양방향 | 테스트 |

**[SYS-085]** Engine Resilience 서브시스템의 명령 수신 경로는 Directive 기반(CommandExecutor)이다. *(MUST)*

| 컴포넌트 | 입력 | 출력 |
|---------|------|------|
| CommandExecutor | ManagerMessage::Directive | ExecutionPlan + CommandResponse |

CommandExecutor는 Manager로부터 수신한 EngineDirective를 디코딩하여 ExecutionPlan을 생성하고, 실행 결과를 CommandResponse로 반환한다.

**[SYS-085a]** Engine Resilience 서브시스템은 D-Bus 경로에서 SystemSignal을 직접 수신하여 자율적으로 대응하는 Strategy 기반 경로를 추가로 지원한다. *(MAY)*

| 컴포넌트 | 입력 | 출력 |
|---------|------|------|
| ResilienceManager | SystemSignal (D-Bus) | ResilienceAction[] |

- ResilienceManager는 도메인별 Strategy(Memory, Thermal, Compute, Energy)를 보유하며, 수신 신호의 Level에 따라 해당 Strategy에 대응을 위임한다.
- 각 Strategy는 Level별로 미리 정의된 대응 액션(SwitchBackend, Throttle, LimitTokens, Evict, Suspend 등)을 반환한다.
- Emergency Level 수신 시 Thermal/Energy Strategy는 `Suspend`를, Memory Strategy는 공격적 eviction(25%)을 반환한다 (SYS-055).
- 다수 Strategy의 출력이 충돌하면 `resolve_conflicts()`가 병합한다. **Suspend는 모든 다른 액션에 우선한다**.
- 이 경로는 Manager 없이 D-Bus로 운영할 때 Engine의 자율 안전 메커니즘을 제공한다 (SYS-050 참조).

> **참고**: Directive 경로(SYS-085)와 Strategy 경로(SYS-085a)는 전송 매체에 따라 배타적으로 활성화된다. 양방향 프로토콜(Unix Socket/TCP) 사용 시 Directive 경로가, D-Bus 사용 시 Strategy 경로가 활성화된다.

#### 불변식

- **[INV-012]** Backend trait은 Engine 내 모든 하드웨어 가속기 연산의 유일한 추상화점이다. Backend를 거치지 않는 하드웨어 직접 호출이 존재해서는 안 된다. *(MUST NOT)*

### 3.3 Manager Internal Architecture [SYS-086 ~ SYS-089]

**[SYS-086]** Manager는 3-layer 파이프라인 구조를 따른다: *(MUST)*

```
Layer 1: Monitor Layer ──SystemSignal──→ Layer 2: Policy Layer ──EngineDirective──→ Layer 3: Emitter Layer
                                                ↑                                          │
                                      EngineMessage (Heartbeat) ←──────────────────────────┘
```

**[SYS-087]** Monitor Layer는 도메인별 독립 스레드에서 OS 센서를 수집한다: *(MUST)*

| Monitor | 도메인 | 센서 소스 | 출력 SystemSignal |
|---------|--------|----------|------------------|
| MemoryMonitor | 메모리 | `/proc/meminfo` | MemoryPressure { available_bytes, total_bytes } |
| ThermalMonitor | 열 | `/sys/class/thermal/` | ThermalAlert { temperature_mc, throttling_active, throttle_ratio } |
| ComputeMonitor | 연산 | `/proc/stat` (CPU delta) | ComputeGuidance { recommended_backend, reason, cpu_usage_pct, gpu_usage_pct } |
| EnergyMonitor | 전력 | `/sys/class/power_supply/` | EnergyConstraint { reason, battery_pct, power_budget_mw } |
| ExternalMonitor | 외부 | stdin/socket | (임의 SystemSignal, 연구/테스트용) |

각 Monitor는 Monitor trait을 구현한다. trait 인터페이스:

| 메서드 | 역할 |
|--------|------|
| run(tx, shutdown) | 스레드 메인 루프. SystemSignal을 mpsc로 전송. shutdown 플래그로 종료. |
| initial_signal() | 시작 시 초기 상태를 보고하는 선택적 신호. |
| name() | Monitor 식별자. |

**[SYS-088]** Policy Layer는 메인 스레드에서 순차 실행되는 3단계 계층형 파이프라인이다: *(MUST)*

```
SystemSignal → PI Controller → PressureVector → Supervisory → OperatingMode (Manager) → Action Selector → ActionCommand[]
```

| 단계 | 입력 | 출력 | 역할 |
|------|------|------|------|
| 압력 계산기 (도메인별) | 원시 측정값 [0, 1] | PressureVector (compute, memory, thermal) | Compute/Thermal: PI Controller (P항+I항, anti-windup, gain scheduling). Memory: 임계값 기반 직접 매핑 (OOM 즉각 대응, ≤100ms 폴링). 전략 패턴으로 교체 가능. |
| Supervisory | PressureVector | OperatingMode (Manager): Normal/Warning/Critical | 운영 모드 결정. 에스컬레이션은 즉시(단계 건너뛰기 가능), 디에스컬레이션은 hold_time 후 단계적. 히스테리시스. |
| Action Selector | OperatingMode + PressureVector + Engine 상태 | ActionCommand 벡터 | Cross-domain 최적 액션 조합 선택. 전수 탐색(2^N), 비용 최소화, 제약 만족. |

> **Manager ↔ Engine OperatingMode 매핑**: Supervisory는 Manager 측 OperatingMode (Normal/Warning/Critical)를 출력한다. Engine 측 OperatingMode (Normal/Degraded/Minimal/Suspended)와의 매핑은 다음과 같다:
>
> | Manager OperatingMode | Engine OperatingMode | 전환 트리거 |
> |---|---|---|
> | Normal | Normal | 압력 해소 + hold_time 경과 |
> | Warning | Degraded | Lossless 액션 Directive 수신 시 |
> | Critical | Minimal | Lossy 액션 Directive 수신 시 |
> | (해당 없음) | Suspended | Suspend 명령 또는 Emergency Level |

**[SYS-088a]** Action Selector는 단일 도메인이 아닌 전체 도메인(compute, memory, thermal)의 압력을 동시에 고려하여 최소 degradation 액션 조합을 선택해야 한다 (cross-domain selection). 각 액션의 multi-domain relief 예측을 ReliefEstimator에서 획득하고, 모든 도메인의 잔여 압력을 해소하는 조합 중 최소 QCF 비용 조합을 선택한다. *(MUST)*

> **Energy 신호 처리**: EnergyConstraint 신호는 독립 PI Controller가 아닌 compute PI에 가중 합산(0.5×)된다. 전력 상태는 시간 스케일(분~시간)이 PI의 초 단위 제어와 상이하여, 별도 PI 도메인보다 compute 보조 신호로 반영하는 것이 적절하다.

보조 컴포넌트:

| 컴포넌트 | 역할 |
|---------|------|
| ActionRegistry | 액션 메타데이터 저장소. ID, 종류(Lossless/Lossy), 가역 여부, 파라미터 범위, 배타 그룹, 기본 비용. |
| ReliefEstimator | 액션별 릴리프 예측. 온라인 선형 회귀(RLS, λ=0.995) 기반 학습. 13차원 특성 벡터 → 4차원 ReliefVector. 액션 실행 후 3초 관찰 딜레이를 두고 실측 relief로 모델을 업데이트한다 ([SYS-019] 참조). |

**[SYS-089]** Emitter Layer는 Policy가 결정한 디렉티브를 Engine에 전송한다. Emitter trait을 구현한다: *(MUST)*

| 메서드 | 역할 |
|--------|------|
| emit(signal) | SystemSignal 전송 (D-Bus) |
| emit_initial(signals) | 초기 상태 일괄 전송 |
| emit_directive(directive) | EngineDirective 전송 (양방향 프로토콜) |
| name() | Emitter 식별자 반환 |

Emitter 구현체:

| 구현체 | 전송 매체 | 방향 | 용도 |
|--------|----------|------|------|
| UnixSocketEmitter | Unix 도메인 소켓 | 양방향 (EngineReceiver 겸용) | 기본 |
| TcpChannel | TCP 소켓 | 양방향 | OS 비의존 |
| DbusEmitter | D-Bus 시스템 버스 | 단방향 (전송만) | Android/Linux 시스템 통합 |

#### 불변식

- **[INV-013]** Monitor 스레드 하나의 장애(panic, 무한루프)가 다른 Monitor 스레드에 전파되어서는 안 된다. 각 Monitor는 독립 OS 스레드에서 실행되며, 공유 상태 없이 mpsc로만 통신한다. *(MUST NOT)*

### 3.4 IPC Topology [SYS-090 ~ SYS-094]

**[SYS-090]** Manager ↔ Engine 간 통신은 양방향이다: *(MUST)*

| 방향 | 메시지 타입 | 내용 |
|------|-----------|------|
| Manager → Engine | ManagerMessage | Directive(EngineDirective) |
| Engine → Manager | EngineMessage | Capability, Heartbeat, Response |

**[SYS-091]** 메시지 흐름은 4가지로 분류된다: *(MUST)*

| 흐름 | 방향 | 빈도 | 설명 |
|------|------|------|------|
| Capability 등록 | Engine → Manager | 세션당 1회 | Engine이 디바이스 정보, KV 용량 등을 보고한다. |
| Heartbeat 상태 보고 | Engine → Manager | 주기적 (~100ms) | Engine이 현재 상태 (디바이스, 리소스 레벨, KV 사용량, 활성 액션 등)를 보고한다. |
| Directive 명령 | Manager → Engine | 이벤트 구동 | Manager가 리소스 상황에 따라 EngineCommand 묶음을 전송한다. |
| Response 응답 | Engine → Manager | Directive당 1회 | Engine이 Directive 실행 결과(Ok/Partial/Rejected)를 보고한다. |

**[SYS-092]** 기본 전송 매체는 Unix Socket이다. 와이어 포맷은 length-prefixed JSON이다: *(MUST)*

```
┌──────────────┬──────────────────────────────────┐
│ 4 bytes      │ N bytes                          │
│ BE u32 (= N) │ UTF-8 JSON (ManagerMessage       │
│              │             또는 EngineMessage)   │
└──────────────┴──────────────────────────────────┘
```

- 길이 필드는 Big-Endian unsigned 32비트 정수이다.
- 페이로드는 UTF-8 인코딩된 JSON이다.
- 직렬화 라이브러리: serde_json.

**[SYS-093]** 연결 모델은 1:1, 단일 클라이언트이다. 하나의 Manager에 하나의 Engine만 연결된다. *(MUST)*

연결 상태 전이:

```
                accept()
Listening ──────────────→ Connected ──────→ Active
    ↑                        │                  │
    │                        │ write_err        │ write_err
    │                        │ reader_eof       │ reader_eof
    │                        ▼                  ▼
    └───── ensure_connected ── Disconnected ←────┘
              (emit 시 재연결 시도)
```

| 상태 | 설명 |
|------|------|
| Listening | Manager가 소켓에 바인드하고 연결 대기. non-blocking accept(). |
| Connected | 연결 수립 완료. Reader 스레드 활성. |
| Active | Capability 수신 후 정상 통신 중. |
| Disconnected | 연결 끊김. 다음 emit 시 재연결 시도. |

**[SYS-094]** 추가 전송 경로로 D-Bus System Bus를 지원한다. `--dbus` 플래그로 활성화된다. *(MAY)*

- D-Bus 경로에서는 SystemSignal만 전송한다 (단방향, Manager → Engine).
- 양방향 프로토콜(Heartbeat, Response)은 지원하지 않는다.
- D-Bus 인터페이스: `org.llm.Manager1`.
- 신호명: `MemoryPressure`, `ComputeGuidance`, `ThermalAlert`, `EnergyConstraint`.

#### 불변식

- **[INV-014]** EngineDirective의 `seq_id`는 세션 내에서 단조 증가해야 한다. 같은 세션에서 이전보다 작거나 같은 `seq_id`가 전송되어서는 안 된다. *(MUST)*
- **[INV-015]** Capability 메시지는 세션당 1회만 전송된다. 연결 수립 직후 Engine이 전송하며, 동일 세션에서 재전송하지 않는다. *(MUST)*

### 3.5 Action Pool Architecture [SYS-095 ~ SYS-099]

**[SYS-095]** Action Pool은 시스템이 리소스 압박에 대응하기 위해 사용하는 액션의 집합이다. 각 액션은 Lossless 또는 Lossy로 분류된다. *(MUST)*

현재 등록된 액션 (ActionRegistry):

| ID | 이름 | 종류 | 가역 | 대상 도메인 (설계 의도) | EngineCommand | 파라미터 |
|----|------|------|------|------------------------|---------------|---------|
| W1 | SwitchHw | Lossless | Yes | Compute, Thermal | `SwitchHw { device }` | device: 디바이스 식별자 |
| W2 | KvOffloadDisk | Lossless | Yes | Memory | (Engine 내부 처리) | offload_ratio: [0, 1] |
| W3 | Throttle | Lossless | Yes | Compute | `Throttle { delay_ms }` | delay_ms: 0~100 |
| C1 | LayerSkip | Lossy | Yes | Compute | `LayerSkip { skip_ratio }` | skip_ratio: [0, 1] |
| C4 | KvEvictH2o | Lossy | No | Memory | `KvEvictH2o { keep_ratio }` | keep_ratio: [0, 1] |
| C5 | KvEvictSliding | Lossy | No | Memory | `KvEvictSliding { keep_ratio }` | keep_ratio: [0, 1] |
| C7 | KvMergeD2o | Lossy | No | Memory | `KvMergeD2o { keep_ratio }` | keep_ratio: [0, 1] |
| C6 | KvQuantDynamic | Lossy | Yes | Memory | `KvQuantDynamic { target_bits }` | target_bits: 2, 4, 8 |
| C8 | KvStreaming | Lossy | No | Memory | `KvStreaming { sink_size, window_size }` | sink_size, window_size: 정수 |

> **도메인 매핑 참고 (non-normative)**: 위 '대상 도메인'은 액션이 해소하도록 **설계된** 논리적 도메인이다. 실측 relief profile(JOURNAL 세션 8)에서는 SwitchHw만 cross-domain(Compute+Thermal) 효과가 유의미하게 확인되었으며, Eviction/Offload 등의 Memory 도메인 릴리프는 물리 메모리 수준에서 null/negligible로 관측되었다. 설계 의도와 실측 간 괴리의 보정 메커니즘은 §7 Rationale을 참조하라.

> **참고**: KvStreaming (C8)은 EngineCommand로 존재하나 ActionRegistry에 미등록 상태이다. StreamingLLM과 동일 동작(sink + sliding window)이며, 등록 시 C4/C5/C7과 상호 배타 그룹(eviction)을 형성해야 한다. ID C2, C3은 이전 설계에서 삭제된 액션에 할당되었으며 현재 미사용이다.

추가 EngineCommand (액션이 아닌 제어 명령):

| 명령 | 용도 |
|------|------|
| RestoreDefaults | 모든 활성 액션을 해제하고 기본 상태로 복귀 |
| PrepareComputeUnit { device } | Backend 전환 사전 준비 (워밍업) |
| Suspend | 추론 일시 중지 |
| Resume | 추론 재개 |

**[SYS-096]** 상호 배타 규칙: 동일 배타 그룹의 액션은 동시에 활성화할 수 없다. 배타 그룹은 정책 설정(TOML `PolicyConfig.exclusion_groups`)에서 정의된다. 아래는 표준 설정의 배타 그룹이다: *(MUST)*

| 배타 그룹 | 액션 | 이유 |
|----------|------|------|
| eviction | C4 (KvEvictH2o), C5 (KvEvictSliding), C7 (KvMergeD2o) | Eviction 정책은 단일 선택. 서로 다른 eviction 전략을 동시 적용하면 KV 캐시 일관성이 깨진다. |

> 설정 예시: `[policy.exclusion_groups]` / `eviction = ["kv_evict_sliding", "kv_evict_h2o", "kv_merge_d2o"]`. 코드 기본값은 빈 HashMap이며, 표준 설정 파일에서 위 프리셋을 정의한다.

**[SYS-097]** 조합 규칙: 배타 그룹이 다른 액션은 동시 활성이 가능하다. *(MAY)*

유효한 조합 예시:
- Eviction (C4 또는 C5) + KvQuantDynamic (C6) + Throttle (W3) + SwitchHw (W1)
- LayerSkip (C1) + Throttle (W3)

무효한 조합 예시:
- KvEvictH2o (C4) + KvEvictSliding (C5) → 동일 배타 그룹 위반
- KvMergeD2o (C7) + KvEvictH2o (C4) → 동일 배타 그룹 위반

**[SYS-098]** 모든 lossy action 실행 시 QcfMetric을 보고해야 한다. QCF 수집이 활성화된 상태에서 lossy action이 QcfMetric 없이 실행되어서는 안 된다 ([SYS-040] 참조). *(MUST)*

**[SYS-099]** Action Selector의 모드별 액션 선택 제약: *(MUST)*

| OperatingMode (Manager) | Engine 대응 모드 | 허용 액션 | 근거 |
|------------------------|----------------|----------|------|
| Normal | Normal | (없음, 액션 불필요) | 압력 없음 |
| Warning | Degraded | Lossless만 (W1, W2, W3) | 품질 손실 0 보장 |
| Critical | Minimal | Lossless + Lossy 전체 | QCF 기반 비용 최소화 |
| (Emergency Level 또는 Suspend 명령) | Suspended | (액션 선택 불가, 추론 일시 중지) | Emergency 시 자율 진입 (SYS-055), 또는 명시적 Suspend 명령 |

#### 불변식

- **[INV-016]** 동일 배타 그룹의 액션이 동시에 활성화되어서는 안 된다. *(MUST NOT)*
- **[INV-017]** QCF 수집 활성 상태에서 lossy action 실행 시 QcfMetric이 반드시 생성되어야 한다 ([INV-004] 재확인). *(MUST)*

### 3.6 Threading Model [SYS-070 보충]

**Engine 스레딩:**

| 스레드 | 역할 | 수명 | 통신 |
|--------|------|------|------|
| Main (추론) | Prefill → Decode 루프 실행. 매 토큰 생성 시 Resilience poll. | 추론 세션 동안 | — |
| SignalListener (선택적) | D-Bus 신호 수신 → mpsc로 전달. | Engine 수명 (resilience feature 활성 시) | mpsc::Sender\<SystemSignal\> |
| MessageLoop (선택적) | Transport로 ManagerMessage 수신 → mpsc로 전달. EngineMessage 전송. | Engine 수명 (양방향 프로토콜 활성 시) | mpsc::Receiver\<ManagerMessage\>, mpsc::Sender\<EngineMessage\> |

**Manager 스레딩:**

| 스레드 | 역할 | 수명 | 통신 |
|--------|------|------|------|
| Main (Policy) | mpsc에서 SystemSignal 수신 → PI → Supervisory → Selector → emit. Heartbeat 수신 처리. | 프로그램 수명 | mpsc::Receiver\<SystemSignal\>, try_recv (Engine 메시지) |
| Monitor (N개, 4~5) | 도메인별 센서 수집 → SystemSignal 전송. | 프로그램 수명 | mpsc::Sender\<SystemSignal\> |
| Reader (채널당 1개) | Unix/TCP 소켓에서 EngineMessage 블로킹 읽기 → inbox mpsc에 전달. | 연결당 (재연결 시 재생성) | mpsc::SyncSender\<EngineMessage\> (버퍼 64) |

**동기화 메커니즘:**
- `mpsc::channel`: Monitor → Main (SystemSignal), MessageLoop → Engine Main (ManagerMessage)
- `mpsc::sync_channel(64)`: Reader → Main (EngineMessage, 배압 제어)
- `Arc<AtomicBool>`: 전역 셧다운 플래그 (SIGINT/SIGTERM 핸들러)
- 명시적 뮤텍스 없음 (Policy 업데이트는 단일 스레드)

#### 불변식

- **[INV-018]** 추론 루프(Prefill/Decode)는 단일 스레드에서 실행된다. Forward pass 중 다른 스레드가 모델 상태나 KV 캐시에 동시 접근해서는 안 된다. *(MUST NOT)*

### 3.7 Data Flow Overview [SYS-070 보충]

시스템의 데이터 흐름은 세 가지 주요 경로로 구분된다.

#### 추론 경로 (Inference Path)

```
모델 파일 (safetensors)
    │ 로딩 (mmap + 역양자화)
    ▼
모델 가중치 (in-memory)
    │
    ▼
Prefill: 입력 프롬프트 전체 처리
    │ KV Cache 초기화
    ▼
Decode 루프:
    │ ┌──────────────────────────────────────────────┐
    │ │  for each layer:                             │
    │ │    RoPE(Q, start_pos) → Attention(Q, K, V)  │
    │ │    → FFN(SiLU/GeLU) → Residual Add          │
    │ │  RMS Norm → Logits → Sampling                │
    │ │  KV Cache update                             │
    │ └──────────────────────────────────────────────┘
    │ Resilience poll (매 토큰)
    ▼
토큰 출력 (stdout)
```

#### 적응 경로 (Adaptation Path)

```
OS 센서 (/proc, /sys)
    │ Monitor 스레드 (주기적 폴링)
    ▼
SystemSignal (mpsc → Main)
    │
    ▼
PI Controller ──→ PressureVector (compute, memory, thermal)
    │
    ▼
Supervisory ──→ OperatingMode/Manager (Normal / Warning / Critical)
    │
    ▼
Action Selector ──→ ActionCommand[] (최적 조합)
    │                     │
    │  ReliefEstimator     │  ActionRegistry
    │  (예측)              │  (메타데이터)
    ▼                     ▼
EngineDirective (seq_id + commands[])
    │ Emitter (Unix Socket)
    ▼
Engine Resilience ──→ ExecutionPlan 실행
    │
    ▼
CommandResponse (Ok / Partial / Rejected)
    │ Engine → Manager
    ▼
Relief observation (3초 후 실측 → ReliefEstimator 업데이트)
```

#### 품질 추적 경로 (Quality Tracking Path)

```
Lossy Action 실행
    │
    ▼
어텐션 중요도 스코어 수집
    │ (needs_attn_scores → set_attn_scores)
    ▼
QcfMetric 생성
    │ action, raw_value, normalized_value, per_head, tokens_affected
    ▼
DegradationEstimator
    │ 피스와이즈 선형 곡선 + EMA 보정
    ▼
PPL 증가량 예측 (degradation)
    │
    ▼
Action Selector 비용 함수: D = Σ default_cost(action)
    │ (현재 구현: Engine 미연결 시 config 정적값 사용.
    │  Engine 연결 시 실시간 QCF 비용 사용 경로는
    │  22-manager-algorithms.md에서 상세화)
    ▼
(선택적) ReliefEstimator 관측 업데이트
```

## 4. Alternative Behavior

### D-Bus Transport

D-Bus System Bus를 통해 SystemSignal을 전달한다. 현재는 단방향(Manager → Engine)으로, 양방향 프로토콜(Heartbeat, Response, Capability)은 지원하지 않는다. Engine 측에서는 `resilience` feature가 활성화되어야 D-Bus 수신이 가능하다.

### 단독 Engine 실행

Manager 없이 Engine만 실행할 수 있다. 이 경우:
- Resilience 서브시스템은 비활성 상태이다.
- 외부 리소스 적응 없이 기본 모드로 추론을 수행한다.
- KV 캐시 압력은 Engine 내부의 CacheManager가 자체적으로 처리한다 (OS 메모리 감시 기반).

## 5. Constraints

**프로세스 분리**: Engine과 Manager는 반드시 별도 OS 프로세스로 실행된다. 단일 프로세스 내 통합은 허용하지 않는다 ([SYS-001]).

**IPC 직렬화**: JSON (serde_json) 전용. 바이너리 직렬화 포맷(bincode, MessagePack 등)은 사용하지 않는다 ([SYS-065]).

**단일 세션**: 동시에 하나의 Engine만 Manager에 연결된다 ([SYS-093]).

**No async runtime**: 모든 스레딩은 `std::thread`와 `mpsc`로 구현한다 ([SYS-064]).

## 6. Examples

### 예시 1: 시스템 배포 토폴로지

Android 디바이스에서의 프로세스 배치:

```
┌─────────────────────────────────────────────────────┐
│ Android Device (Snapdragon 8 Gen 3, 12 GB RAM)      │
│                                                     │
│  ┌─────────────────┐    Unix Socket    ┌──────────┐ │
│  │ llm_manager     │◄────────────────►│ llm_rs2  │ │
│  │                 │ /tmp/llm.sock     │          │ │
│  │ [Monitor x5]   │                   │ [Model]  │ │
│  │ [Policy]       │                   │ [KV$]    │ │
│  │ [Emitter]      │                   │ [Backend]│ │
│  └─────────────────┘                   └──────────┘ │
│         │                                   │       │
│    /proc, /sys                        OpenCL (GPU)  │
│    (OS 센서)                          CPU (NEON)    │
└─────────────────────────────────────────────────────┘
```

### 예시 2: 정상 추론 시퀀스

Manager와 Engine이 연결된 후 정상 추론이 진행되는 시퀀스:

```
Engine                          Manager
  │                                │
  │──── Capability ───────────────→│  (디바이스: ["cpu","gpu"], max_kv: 2048)
  │                                │
  │  ┌─ Decode 루프 ─────────────┐ │
  │  │                           │ │
  │  │←── Heartbeat ────────────→│ │  (state: Running, kv_tokens: 50, device: gpu)
  │  │  (토큰 1)                 │ │
  │  │                           │ │
  │  │←── Heartbeat ────────────→│ │  (state: Running, kv_tokens: 51)
  │  │  (토큰 2)                 │ │
  │  │                           │ │
  │  │   ... (반복) ...          │ │
  │  │                           │ │
  │  └───────────────────────────┘ │
  │                                │
  │  (추론 완료)                    │
```

### 예시 3: 리소스 압박 대응 시퀀스

메모리 압박이 발생하여 Manager가 Directive를 전송하는 시퀀스:

```
Engine                          Manager
  │                                │
  │←── Heartbeat ─────────────────→│  (memory_level: Warning, kv_utilization: 0.85)
  │                                │
  │                                │  [PI: memory_pressure = 0.72]
  │                                │  [Supervisory: Critical]
  │                                │  [Selector: KvEvictH2o(keep=0.5)]
  │                                │
  │←── Directive(seq=1) ──────────│  commands: [KvEvictH2o { keep_ratio: 0.5 }]
  │                                │
  │  (KV 캐시 50% eviction 실행)   │
  │  (QcfMetric 생성)              │
  │                                │
  │──── Response(seq=1) ──────────→│  results: [Ok]
  │                                │
  │←── Heartbeat ─────────────────→│  (memory_level: Normal, kv_utilization: 0.42)
  │                                │
  │                                │  [PI: memory_pressure = 0.15]
  │                                │  [Supervisory: hold_time 대기...]
  │                                │  [Relief observation: actual = 0.57]
  │                                │  [ReliefEstimator update]
```

## 7. Rationale (non-normative)

### 왜 2-프로세스 분리인가

프로세스 분리는 세 가지 핵심 이점을 제공한다:

1. **Fail-safety**: 한쪽 프로세스의 크래시가 다른 쪽에 영향을 주지 않는다. 특히 Manager 크래시 시에도 Engine은 추론을 지속할 수 있어, 사용자 경험이 보호된다.
2. **독립 업데이트**: Manager의 정책 로직만 업데이트할 때 Engine을 재시작할 필요가 없다. 반대도 마찬가지이다.
3. **리소스 격리**: Monitor 스레드의 센서 폴링과 Policy 평가가 추론 hot path의 지연에 영향을 주지 않는다.

### 왜 Cargo workspace 3-crate 구조인가

- **Shared 독립**: IPC 타입이 독립 크레이트에 있어 Engine과 Manager의 순환 의존을 원천 차단한다.
- **독립 빌드**: 각 바이너리를 독립적으로 빌드/배포할 수 있다.
- **타입 일관성**: Shared의 타입 정의를 양측이 동일하게 참조하여, 직렬화/역직렬화 불일치를 방지한다.

### 왜 계층형 정책인가

기존 임계값 기반 접근의 문제점:
- CPU 69% → 71%에서 Warning으로 즉시 점프. 5초간 71%와 30초간 71%를 구분하지 못함.
- 도메인 독립 판단: Backend 전환이 compute와 thermal을 동시에 해소하는 것을 고려하지 못함.
- 품질 보장 없음: lossy action의 실제 비용을 알지 못함.

계층형 정책은 이를 해결한다:
- PI Controller의 I항이 누적 압력을 반영한다 (시간 인식).
- Action Selector가 cross-domain relief를 고려한다.
- QCF 기반 비용 함수가 품질 보장을 제공한다.

### 왜 Energy는 별도 PI 도메인이 아닌가

EnergyMonitor는 배터리 잔량, 충전 상태, 전력 예산 등 시간 스케일이 분~시간 단위인 메트릭을 수집한다. PI Controller는 초 단위의 빠른 피드백 루프를 실행하므로, Energy를 4번째 독립 PI 도메인으로 두면 저빈도 신호(배터리 1% 감소)가 I항에 부적절하게 누적된다. 대신 EnergyConstraint 신호를 Compute PI에 가중 합산(0.5×)하여, 전력 부족 시 연산 강도를 줄이는 간접 대응을 수행한다. 향후 배터리 잔량에 기반한 장기 전략(long-term policy)이 필요하면 Supervisory 상위에 별도 계층을 추가할 수 있다.

### 왜 설계 의도 도메인과 실측이 다른가

SYS-095의 '대상 도메인'은 각 액션이 해소할 것으로 **설계된** 논리적 도메인이다. 그러나 실측 relief profile에서는 SwitchHw만이 cross-domain(Compute+Thermal) 효과를 보였고, Eviction/Offload 등은 물리 메모리 변화가 null/negligible이었다. 이는 소형 모델(1.5B)과 짧은 시퀀스에서 KV 캐시가 전체 메모리 대비 미미하기 때문이다. ReliefEstimator의 온라인 학습([SYS-019])이 이 괴리를 런타임에 보정한다: 정적 도메인 매핑 대신 실측 기반 relief 예측을 학습하여 Action Selector에 제공한다.

### 왜 Action Pool 체계인가

- **Lossless 우선**: Warning 모드에서 lossless만 허용하여, 불필요한 품질 손실을 방지한다.
- **Lossy 최소화**: Critical에서만 lossy를 허용하되, QCF 비용 함수로 최소 비용 조합을 선택한다.
- **조합 최적화**: 단일 액션이 아닌 액션 조합을 탐색하여, 각 도메인의 압력을 동시에 해소한다.
- **상호 배타**: 호환 불가능한 액션 조합(예: 서로 다른 eviction 정책)을 제도적으로 방지한다.
