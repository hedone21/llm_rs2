# Engine Overview

> **TL;DR**: Engine은 LLM 추론을 실행하는 독립 프로세스이다. 8개 서브시스템(Model, Core, Backend, KV Cache, Cache Management, Resilience, QCF, Eval)으로 구성되며, Manager의 Directive에 반응하여 런타임 적응을 수행한다. Transport trait 4종 구현체로 Manager와 통신하고, Main(추론) + MessageLoop + Rayon 3종 스레드로 동작한다. Manager 없이 단독 실행이 가능하며, Manager 장애 시에도 추론을 계속한다.

## 1. Purpose and Scope

이 문서는 Engine 프로세스의 **책임, 8개 서브시스템 분해, Transport trait과 구현체, 스레딩 모델, CLI 인터페이스**를 상위 수준에서 정의한다.

**이 파일이 명세하는 것:**

- Engine 프로세스의 단일 책임과 독립성 보증
- 8개 서브시스템의 책임, 코드 모듈 매핑, 인터페이스, 의존 관계
- Transport trait과 4종 구현체
- 스레딩 모델과 Resilience Checkpoint
- CLI 인터페이스 (40+ 플래그 중 아키텍처 관련)
- 서브시스템 의존 그래프

**이 파일이 명세하지 않는 것:**

- 상태 머신 전이 테이블 → `31-engine-state.md`
- Eviction 정책별 알고리즘, quantization 수식, layer skip 선택, QCF 계산 → `32-engine-algorithms.md`
- 데이터 타입, 설정 스키마 상세 → `33-engine-data.md`
- Manager 내부 → `20-manager.md` ~ `23-manager-data.md`
- 프로토콜 메시지, 시퀀스 → `10-protocol.md` ~ `12-protocol-sequences.md`

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **Subsystem** | Engine 내부의 논리적 기능 단위. 1개 이상의 Rust 모듈로 구현된다. |
| **Backend** | 수치 연산을 수행하는 하드웨어 추상화 계층. `Backend` trait으로 추상화된다. |
| **Transport** | Manager-Engine 간 양방향 바이트 스트림 추상화. `Transport` trait으로 추상화된다. |
| **MessageLoop** | Transport를 소유하고 mpsc 채널로 브리징하는 전용 스레드. |
| **Inference Loop** | 토큰 단위로 forward pass + sampling + resilience poll을 반복하는 메인 루프. |
| **Resilience Checkpoint** | Inference Loop 내에서 토큰당 1회 실행되는 `CommandExecutor.poll()` 호출 지점. |
| **ExecutionPlan** | 단일 `poll()` 호출의 결과물. Inference Loop가 즉시 소비한다. 수명 규칙 → `31-engine-state.md` ENG-ST-042. |

## 3. Specification

### 3.1 Engine Process Responsibility [ENG-010]

**[ENG-010]** Engine 프로세스는 LLM 추론 실행의 단일 책임을 갖는다. 모델 로딩, 토큰화, forward pass, 샘플링, KV 캐시 관리, QCF 메트릭 수집을 수행하며 Manager의 Directive에 반응한다. *(MUST)*

- Engine은 독립 프로세스로 실행되며, Manager 없이도 단독 동작이 가능하다 (SYS-052 참조)
- Manager 연결이 없을 때 `command_executor = None` 상태로 추론만 수행한다
- Manager 연결이 끊겨도 마지막 상태를 유지하며 추론을 계속한다 (SYS-053 참조)

### 3.2 Subsystem Decomposition [ENG-011 ~ ENG-018]

8개 서브시스템은 22+ 코드 모듈의 논리적 그루핑이다 (SYS-080 참조). 서브시스템과 코드 모듈은 1:1 대응이 아니다.

**[ENG-011] Model Subsystem** *(MUST)*

| 항목 | 내용 |
|------|------|
| 책임 | 모델 가중치 로딩, forward pass 실행, 토큰 임베딩/언임베딩 |
| 코드 모듈 | `models/` (TransformerModel, config, mappers), `layers/` (attention, llama_layer, transformer_layer, workspace) |
| 주요 인터페이스 | `TransformerModel::forward()` -- logits 텐서 반환 |
| 의존 | Backend, KV Cache, Memory |

**[ENG-012] Core Subsystem** *(MUST)*

| 항목 | 내용 |
|------|------|
| 책임 | 텐서 연산 기본 타입, 메모리 할당, 수학 유틸리티, 샘플링 |
| 코드 모듈 | `core/tensor.rs`, `core/buffer.rs`, `core/shape.rs`, `core/memory.rs`, `core/math_utils.rs`, `core/sampling.rs`, `core/quant.rs`, `core/thread_pool.rs` |
| 주요 인터페이스 | `Tensor`, `Buffer` (trait), `DType` enum, `Memory` (trait), `SamplingConfig` |
| 의존 | 없음 (최하위 계층) |

**[ENG-013] Backend Subsystem** *(MUST)*

| 항목 | 내용 |
|------|------|
| 책임 | 하드웨어별 수치 연산 구현 (matmul, RMS norm, RoPE, softmax, attention 등) |
| 코드 모듈 | `core/backend.rs` (Backend trait), `backend/cpu/` (Neon, AVX2, Common), `backend/opencl/` (OpenCLBackend) |
| Backend trait | ~20 메서드: `matmul`, `matmul_transposed`, `rms_norm`, `silu_mul`, `gelu_tanh_mul`, `rope`, `softmax`, `masked_softmax`, `attention_qkv` 등 |

Backend 구현체:

| 구현체 | feature gate | 타겟 아키텍처 | 특징 |
|--------|-------------|--------------|------|
| CpuBackendNeon | 기본 | aarch64 | NEON SIMD + Rayon 병렬, F16 네이티브 |
| CpuBackendAVX2 | 기본 | x86_64 | AVX2 SIMD |
| CpuBackendCommon | 기본 | 기타 | 스칼라 폴백 |
| OpenCLBackend | `opencl` | 모든 아키텍처 | GPU kernel (OpenCL), plan-based decode. Adreno/Mali 타겟. |
| CudaBackend | `cuda` | aarch64 (Jetson) | cudarc + llama.cpp PTX 커널, Tensor Core, MMVQ/MMQ. sm_72+ 전용. → `34-engine-cuda.md` |

> **참고 (non-normative)**: `--backend cpu` 시 GPU secondary(OpenCL 또는 CUDA)를 자동 초기화한다. SwitchHw 명령으로 런타임 전환이 가능하다. `opencl`과 `cuda` feature는 상호 배타적이다.

**[ENG-014] KV Cache Subsystem** *(MUST)*

| 항목 | 내용 |
|------|------|
| 책임 | Key-Value 캐시 저장, 조회, 용량 관리 |
| 코드 모듈 | `core/kv_cache.rs` (KVCacheOps trait, KVCache struct), `core/kivi_cache.rs` (KiviCache), `core/kv_migrate.rs` |
| KVCacheOps trait | 14 메서드: `current_pos`, `set_current_pos`, `capacity`, `kv_heads`, `head_dim`, `layout`, `kv_dtype`, `memory_usage_bytes`, `update`, `get_view`, `get_buffers_mut`, `advance_pos`, `ensure_capacity`, `needs_attn_scores` / `set_attn_scores` (일부 default 구현) |
| KVLayout | `Head` (head-major), `Seq` (seq-major) |

KVCacheOps 구현체:

| 구현체 | 설명 | DType | 지원 기능 |
|--------|------|-------|----------|
| KVCache | 표준 캐시 | F32, F16, Q4_0 | 모든 eviction 정책, offload |
| KiviCache | KIVI Q2 + residual buffer | F32 입력, 내부 Q2 | AWQE (`needs_attn_scores=true`), eviction 미지원 |
| OffloadKVCache | Disk/Raw offload | F16, F32 | seq-major only, `--kv-offload` |

**[ENG-015] Cache Management Subsystem** *(MUST)*

| 항목 | 내용 |
|------|------|
| 책임 | 메모리 압력 기반 KV 캐시 관리 오케스트레이션 |
| 코드 모듈 | `core/cache_manager.rs` (CacheManager), `core/pressure/` (CachePressurePipeline, 6종 Handler), `core/eviction/` (EvictionPolicy trait, 5종 구현), `core/attention_scores.rs` |
| CacheManager | CachePressurePipeline 래핑, `force_evict_by_policy()` (resilience dispatch), SystemMonitor 의존 |
| CachePressurePipeline | PressureLevel 기반 Handler 체인. PressureStageConfig: min_level + handler |

CachePressureHandler 6종:

| Handler | 구현 상태 | 설명 |
|---------|----------|------|
| EvictionHandler | 구현 완료 | EvictionPolicy 래핑 어댑터 |
| D2OHandler | 구현 완료 | D2O (Dynamic Token Dropping) |
| SwapHandler | 구현 완료 | Disk offload |
| QuantizeHandler | 간접 활성 | KiviCache 경로에서 간접 호출. `handle()`은 NoOp |

EvictionPolicy 구현체 5종:

| 구현체 | 설명 |
|--------|------|
| H2OPolicy | Heavy-Hitter Oracle, score 기반 eviction |
| H2OPlusPolicy | H2O + per-head GQA-aware eviction |
| SlidingWindowPolicy | 고정 윈도우 eviction |
| NoEvictionPolicy | eviction 미수행 (기본값) |
| StreamingLLMPolicy | CLI `--eviction-policy streaming` |

> **8종 액션과의 관계** (SYS-095 참조):
>
> | 액션 | Engine 실행 경로 |
> |------|-----------------|
> | SwitchHw | `plan.switch_device` -- Backend 런타임 교체 |
> | Throttle | `plan.throttle_delay_ms` -- 토큰 간 sleep |
> | KvEvictH2o | `EvictMethod::H2o` -- `CacheManager::force_evict_by_policy()` |
> | KvEvictSliding | `EvictMethod::Sliding` -- `CacheManager::force_evict_by_policy()` |
> | KvStreaming | `EvictMethod::Streaming` -- `StreamingLLMPolicy::new(sink_size, window_size).evict()` 즉석 호출. `ActionId::KvEvictStreaming` 등록. C4/C5/C7과 eviction 배타 그룹 |
> | KvQuantDynamic | `plan.kv_quant_bits` -- KIVI 경로에서 소비 |
> | KvMergeD2o | `EvictMethod::D2o` -- `CacheManager::force_evict_with_scores(target_ratio)` via Pipeline D2OHandler. 전제: `--eviction-policy d2o`. C4/C5/C7/C8 eviction 배타 그룹 |
> | LayerSkip | `plan.layer_skip` -- SkipConfig 갱신 |

**[ENG-016] Resilience Subsystem** *(MUST)*

| 항목 | 내용 |
|------|------|
| 책임 | Manager Directive 수신/응답, 시스템 신호 반응, ExecutionPlan 생성 |
| 코드 모듈 | `resilience/` 전체 (executor.rs, transport.rs, manager.rs, state.rs, signal.rs, strategy/, dbus_transport.rs) |

Resilience 서브시스템은 두 경로를 포함한다 (SYS-085, SYS-085a 참조).

**경로 A: Directive 경로 (주 경로)**

```
Manager --> Transport --> MessageLoop(thread) --> mpsc --> CommandExecutor --> ExecutionPlan --> Inference Loop
                                                  CommandExecutor <-- KVSnapshot
                                                  CommandExecutor --> mpsc --> MessageLoop --> Transport --> Manager
```

| 컴포넌트 | 책임 |
|----------|------|
| Transport (trait) | Manager-Engine 양방향 바이트 스트림 |
| MessageLoop | Transport를 소유하는 전용 스레드. blocking recv + try_recv drain |
| CommandExecutor | ManagerMessage 수신, EngineCommand 해석, ExecutionPlan 구축. 전략 로직 없음. Heartbeat 주기 전송 |

- CommandExecutor는 EngineCommand 11종을 1:1로 ExecutionPlan 필드에 매핑한다
- Suspend가 포함되면 다른 모든 plan 필드를 초기화한다 (evict=None, switch_device=None, throttle=0)
- 동일 `poll()` 내 복수 Directive는 순서대로 처리하며 후행이 선행을 덮어쓴다 (superseding)
- 상태 전이 상세 → `31-engine-state.md` ENG-ST-030 ~ ENG-ST-039

**경로 B: Strategy 경로 (D-Bus 레거시)**

```
D-Bus system bus --> DbusTransport --> signal_to_manager_message() --> ManagerMessage --> (경로 A 합류)
                                       (내부: Level --> EngineCommand 변환)
```

| 컴포넌트 | 책임 |
|----------|------|
| DbusTransport | D-Bus 시스템 버스 연결, legacy SystemSignal 파싱, ManagerMessage 변환 |
| ResilienceManager | 4종 SystemSignal 수신, SignalLevels 캐시, OperatingMode 계산, 4종 Strategy 위임, `resolve_conflicts()` |
| Strategy 4종 | MemoryStrategy, ComputeStrategy, ThermalStrategy, EnergyStrategy |

- DbusTransport 내부에서 SystemSignal을 EngineCommand로 변환한다 (`signal_to_manager_message`)
- ResilienceManager는 `generate.rs` 메인 루프에서 **직접 사용되지 않는다**. DbusTransport가 변환한 결과가 `ManagerMessage::Directive`로 CommandExecutor에 전달된다
- Emergency level signal 수신 시 DbusTransport가 `EngineCommand::Suspend`로 변환하여 Engine이 자율적으로 Suspended 상태에 진입한다 (SYS-055)
- ResilienceManager/Strategy는 독립적으로도 사용 가능하나 (`manager.rs`의 `InferenceContext` + `execute_action`), 현재 `generate.rs`에서는 이 경로를 사용하지 않는다
- OperatingMode FSM 상세 → `31-engine-state.md` ENG-ST-010 ~ ENG-ST-015

**공존 규칙** (SYS-085):

- Directive 경로가 주(primary)이다. Manager 프로세스 존재 시 항상 이 경로를 사용한다 *(MUST)*
- Strategy 경로는 D-Bus feature gate (`#[cfg(feature = "resilience")]`) 뒤에 위치한다
- 두 경로가 동시 활성되지 않는다. `generate.rs`에서 transport 종류에 따라 하나만 선택한다 *(MUST)*
- DbusTransport 사용 시 D-Bus signal이 ManagerMessage로 변환되어 CommandExecutor로 전달되므로 사실상 Directive 경로로 합류한다

**[ENG-017] QCF Subsystem** *(MUST)*

| 항목 | 내용 |
|------|------|
| 책임 | 손실 액션의 품질 비용 추정 (QCF 메트릭 수집, degradation estimation) |
| 코드 모듈 | `core/qcf/` (mod.rs, eviction_qcf.rs, quant_qcf.rs, skip_qcf.rs, estimator.rs, layer_importance.rs) |
| QcfMetric | action, raw_value, normalized_value, per_head, tokens_affected |
| QcfMode | Attn, Caote, Both -- CLI `--qcf-mode` |
| DegradationEstimator | QCF를 추정 PPL 증가량으로 변환 (알고리즘 상세 → `32-engine-algorithms.md`) |

**[ENG-018] Eval Subsystem** *(MUST)*

| 항목 | 내용 |
|------|------|
| 책임 | Log-likelihood 평가 (downstream task accuracy 측정) |
| 코드 모듈 | `eval/` (eval_loop.rs, hook.rs, eviction_hook.rs, kivi_hook.rs, output.rs, qcf_helpers.rs) |
| StepHook trait | KVCacheOps 구현체별 평가 로직 추상화 (EvictionHook, KiviHook) |
| CLI | `--eval-ll`, `--eval-batch`, `--eval-continuation` |

#### Cross-subsystem Engine Components

서브시스템 경계를 횡단하는 핵심 컴포넌트 (ENG-ALG/INV 추적용):

| 컴포넌트 | 핵심 타입/함수 | ENG-ALG | INV |
|----------|---------------|---------|-----|
| GPU Plan Partition (`backend/opencl/plan.rs`) | PartitionStep, FfnVariant, build_partitioned_ffn | ENG-ALG-200 | INV-082, INV-120 |

### 3.3 Transport Trait and Implementations [ENG-020 ~ ENG-024]

**[ENG-020]** Transport trait은 Manager-Engine 간 양방향 바이트 스트림을 추상화한다. *(MUST)*

```
trait Transport: Send + 'static
    connect() -> Result<(), TransportError>
    recv()    -> Result<ManagerMessage, TransportError>   // blocking
    send(msg: &EngineMessage) -> Result<(), TransportError>
    name()    -> &str
```

| 메서드 | 설명 |
|--------|------|
| `connect()` | 연결 수립. 1회 호출 |
| `recv()` | Manager에서 Engine으로 메시지 블로킹 수신 |
| `send()` | Engine에서 Manager로 메시지 전송 (Capability, Heartbeat, Response) |
| `name()` | 로깅용 이름 |

TransportError 변종: `ConnectionFailed(String)`, `Disconnected`, `ParseError(String)`, `Io(io::Error)`.

**[ENG-021]** 와이어 포맷 (UnixSocket, TCP 공통): *(MUST)*

- `[4 bytes BE u32 length][UTF-8 JSON payload]`
- JSON serde: serde_json
- SYS-065 참조

**[ENG-022]** Transport 구현체 4종: *(MUST)*

| 구현체 | feature gate | CLI 선택 | 프로토콜 |
|--------|-------------|----------|----------|
| UnixSocketTransport | `#[cfg(unix)]` | `--resilience-transport unix:/path` | 길이 접두 JSON |
| TcpTransport | 기본 | `--resilience-transport tcp:addr:port` | 길이 접두 JSON |
| DbusTransport | `resilience` | `--resilience-transport dbus` | D-Bus signal 변환 JSON |
| MockTransport | (테스트 전용) | -- | mpsc 채널 |

**[ENG-023]** MessageLoop: *(MUST)*

- `MessageLoop::spawn(transport)` 호출 시 `(cmd_rx, resp_tx, JoinHandle)` 반환
- 전용 스레드 생성. 스레드 이름: `"{transport_name}-loop"`
- 루프 동작: (1) `resp_rx`에서 `try_recv` drain하여 `transport.send()` (2) blocking `transport.recv()`로 수신하여 `cmd_tx.send()`
- 종료 조건: Disconnected, `cmd_tx` 수신측 drop, send 오류

**[ENG-024]** DbusTransport 변환 규칙: *(MUST)*

| D-Bus signal | Level | 변환 결과 EngineCommand |
|-------------|-------|----------------------|
| MemoryPressure | Normal | RestoreDefaults |
| MemoryPressure | Warning | KvEvictSliding { keep_ratio: 0.85 } |
| MemoryPressure | Critical | KvEvictH2o { keep_ratio: 0.50 } |
| MemoryPressure | Emergency | KvEvictH2o { keep_ratio: 0.25 } (SYS-055: Suspend 불필요, 공격적 eviction) |
| ComputeGuidance | Normal | RestoreDefaults |
| ComputeGuidance | Warning | Throttle { delay_ms: 30 } + SwitchHw { device } |
| ComputeGuidance | Critical | Throttle { delay_ms: 70 } + SwitchHw { device } |
| ComputeGuidance | Emergency | Suspend |
| ThermalAlert | Normal | RestoreDefaults |
| ThermalAlert | Warning | Throttle { delay_ms: 30 } + PrepareComputeUnit { device: "cpu" } |
| ThermalAlert | Critical | Throttle { delay_ms: 70 } + SwitchHw { device: "cpu" } |
| ThermalAlert | Emergency | Suspend |
| EnergyConstraint | Normal | RestoreDefaults |
| EnergyConstraint | Warning | SwitchHw { device: "cpu" } |
| EnergyConstraint | Critical | SwitchHw { device: "cpu" } + Throttle { delay_ms: 70 } |
| EnergyConstraint | Emergency | Suspend |

DbusTransport는 새 "Directive" D-Bus signal도 지원한다. JSON body를 직접 ManagerMessage로 역직렬화하며, legacy signal보다 우선 처리한다.

### 3.4 Threading Model [ENG-030 ~ ENG-033]

**[ENG-030]** 스레드 구조: *(MUST)*

| 스레드 | 생성 조건 | 역할 |
|--------|----------|------|
| Main (추론) | 항상 | model load, prefill, decode loop (per-token: forward + sample + resilience poll) |
| MessageLoop | `--enable-resilience` 또는 `--experiment-schedule` | Transport recv/send 브리징 |
| Rayon pool | 항상 (전역) | matmul 병렬화. `--threads N` (0=auto-detect) |

**[ENG-031]** Main 스레드 흐름: *(MUST)*

```
 1. CLI Args 파싱
 2. Rayon 스레드 풀 초기화
 3. Backend 생성 (cpu / opencl / hybrid)
 4. Model 로딩 (safetensors -> Tensor)
 5. KV Cache 생성 (KVCache / KiviCache / OffloadKVCache)
 6. CacheManager 생성 (EvictionPolicy + SystemMonitor + Pipeline)
 7. CommandExecutor 생성:
    - experiment_schedule -> 내부 mpsc 채널
    - enable_resilience   -> transport
    - 둘 다 없으면         -> None
 8. [KIVI 경로] run_kivi() 분기
 9. [Offload 경로] run_offload() 분기
10. Prefill (prompt encoding, chunked prefill 옵션)
11. Decode loop:
    a. forward pass
    b. sampling
    c. Experiment: inject directives
    d. Resilience checkpoint (ENG-032)
    e. Eviction dispatch (plan.evict -> cache_manager.force_evict_by_policy)
    f. Device switch (plan.switch_device -> backend 교체)
    g. Throttle (plan.throttle_delay_ms -> sleep)
    h. Suspend/Resume (plan.suspended -> 대기 루프)
    i. QCF metric 수집
    j. Profiling/Logging
```

**[ENG-032]** Resilience Checkpoint 상세: *(MUST)*

```
// 토큰당 1회, decode loop 내
kv_snap = KVSnapshot {
    total_bytes,        // 전 layer KV buffer 합산
    total_tokens,       // kv_caches[0].current_pos
    capacity,           // kv_caches[0].capacity()
    protected_prefix,   // CLI --protected-prefix 또는 정책 기본값
    kv_dtype,           // "f16", "q4", "q2"
    eviction_policy,    // CLI --eviction-policy
    skip_ratio,         // 현재 layer skip 비율
}
plan = executor.poll(&kv_snap)
// plan 소비: evict, switch_device, throttle_delay_ms, suspended, resumed,
//            layer_skip, kv_quant_bits, restore_defaults
```

KVSnapshot 필드 상세 → `31-engine-state.md` ENG-ST-070.

**[ENG-033]** Heartbeat: *(MUST)*

- heartbeat_interval: 1000ms 하드코딩 (`generate.rs`)
- Heartbeat는 `executor.poll()` 내에서 경과 시간 체크 후 전송한다
- Heartbeat 내용: EngineStatus (16필드, `active_actions`/`available_actions` 포함)
- EngineStatus 필드 상세 → `11-protocol-messages.md` MSG-050 ~ MSG-066

### 3.5 CLI Interface [ENG-040 ~ ENG-045]

**[ENG-040]** Resilience 관련 CLI 플래그: *(MUST)*

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--enable-resilience` | bool | false | Resilience 활성화 (CommandExecutor 생성) |
| `--resilience-transport` | String | "dbus" | Transport 종류: "dbus", "unix:/path", "tcp:addr:port" |
| `--experiment-schedule` | Option&lt;String&gt; | None | 실험 스케줄 JSON (내부 mpsc 채널, transport 불필요) |

**[ENG-041]** Backend 관련 CLI 플래그: *(MUST)*

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--backend` | String | "cpu" | "cpu", "opencl", "cuda" |
| `--switch-threshold` | usize | 0 | hybrid 모드 자동 전환 토큰 수 (0=비활성) |
| `--gpu-attn` | bool | false | GPU attention kernel 사용 |
| `--zero-copy` | bool | false | CPU-GPU 공유 메모리 |

**[ENG-042]** KV Cache 관련 CLI 플래그: *(MUST)*

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--kv-type` | String | "f16" | KV 캐시 dtype: f32, f16, q4 |
| `--kv-layout` | String | "head" | head-major 또는 seq-major |
| `--eviction-policy` | String | "none" | none, sliding, streaming, h2o, h2o_plus, d2o |
| `--kv-budget` | usize | 0 | 최대 KV 토큰 수 (0=무제한) |
| `--kv-budget-ratio` | f32 | 0.0 | prompt 대비 비율 |
| `--kivi` | bool | false | KIVI Q2 캐시 활성화 |
| `--kv-offload` | String | "none" | none, raw, disk |
| `--protected-prefix` | Option&lt;usize&gt; | None | eviction에서 보호할 접두 토큰 수 |

**[ENG-043]** QCF/Eval CLI 플래그: *(MUST)*

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--qcf-mode` | String | "attn" | "attn", "caote", "both" |
| `--eval-ll` | bool | false | Log-likelihood 평가 모드 |
| `--eval-batch` | Option&lt;String&gt; | None | 평가 배치 JSON 파일 |
| `--profile` | bool | false | 프로파일링 활성화 |

**[ENG-044]** Inference 관련 CLI 플래그 (주요): *(MUST)*

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--model-path` | String | "models/llama3.2-1b" | 모델 경로 |
| `--num-tokens` | usize | 20 | 생성 토큰 수 |
| `--max-seq-len` | usize | 2048 | 최대 시퀀스 길이 |
| `--threads` | usize | 0 | 스레드 수 (0=auto) |
| `--weight-dtype` | String | "f16" | 가중치 dtype |
| `--skip-ratio` | Option&lt;f32&gt; | None | Layer skip 비율 |
| `--greedy` | bool | false | 탐욕 샘플링 |

**[ENG-045]** Experiment 모드: *(MUST)*

- `--experiment-schedule`: ExperimentSchedule JSON 로딩. 토큰 위치별 Directive 주입
- `--experiment-output`: JSONL 출력 경로
- `--experiment-eviction-ratio`: resilience eviction 신호의 target_ratio 오버라이드
- Experiment 모드에서는 외부 Transport 대신 내부 mpsc 채널로 CommandExecutor를 구동한다

### 3.6 Subsystem Dependency Graph [ENG-050]

**[ENG-050]** 서브시스템 의존 그래프: *(MUST)*

```
Model --------- Backend ---- Core (Tensor, Buffer, Memory, DType)
  |               |
  |               +---- OpenCL (feature: opencl)
  |               +---- CPU (Neon/AVX2/Common)
  |
  +---- KV Cache -- Core
  |        |
  |        +---- Cache Management -- EvictionPolicy
  |                    |                   |
  |                    +-- CachePressurePipeline -- 6종 Handler
  |                    +-- SystemMonitor
  |
  +---- QCF -- Core (attention_scores)
  |
  +---- Eval -- Model + KV Cache + QCF

Resilience -- Transport -- MessageLoop
    |              |
    |              +-- UnixSocket / TCP / D-Bus / Mock
    |              +-- mpsc channels
    |
    +-- CommandExecutor -- ExecutionPlan --> Inference Loop
    |
    +-- ResilienceManager + Strategy (D-Bus 레거시, generate.rs에서 미사용)
```

## 4. Alternative Behavior

**[ENG-060]** 단독 실행 모드:

- `--enable-resilience` 없이 실행 시 `command_executor = None`
- Inference Loop에서 resilience checkpoint를 건너뛴다
- KV 캐시 관리는 CacheManager의 자체 메모리 감시(SystemMonitor)로만 수행한다

**[ENG-061]** D-Bus Transport (레거시):

- `--resilience-transport dbus` 시 DbusTransport 사용
- `resilience` feature gate 필요 (`#[cfg(feature = "resilience")]`)
- Manager 프로세스 없이 OS D-Bus signal에 직접 반응하는 자율 모드
- `signal_to_manager_message()`로 EngineDirective를 자동 생성한다

## 5. Constraints

**[ENG-070]** Engine 프로세스는 async 런타임을 사용하지 않는다. *(MUST NOT)* (SYS-064 참조)

**[ENG-071]** 모든 프로세스 간 메시지는 JSON 직렬화한다. *(MUST)* (SYS-065 참조)

**[ENG-072]** 단일 세션 -- Engine은 동시에 1개 Manager만 연결한다. *(MUST)*

**[ENG-073]** `Transport::recv()`는 blocking이다. MessageLoop 전용 스레드에서만 호출한다. *(MUST)*

**Invariants:**

| ID | 불변식 |
|----|--------|
| INV-060 | `CommandExecutor.poll()`은 토큰당 최대 1회 호출된다 |
| INV-061 | ExecutionPlan은 생성한 `poll()` 호출이 반환한 즉시 소비되며, 다음 `poll()` 전에 폐기된다 |
| INV-062 | Suspend가 포함된 ExecutionPlan에서 evict, switch_device, prepare_device는 None이다 |
| INV-063 | MessageLoop 스레드는 Transport의 유일한 소유자이다 |
| INV-064 | heartbeat_interval 내에 최소 1회 Heartbeat가 전송된다 (`poll()` 호출이 있을 때) |
| INV-065 | Backend trait 구현체는 `Send + Sync`이다 |

## 6. Examples

### 6.1 Engine 단독 실행 (Manager 미연결)

```
1. ./generate --model-path models/qwen2.5-1.5b --num-tokens 100
2. CLI 파싱 -> enable_resilience=false
3. Backend 생성: CpuBackendNeon
4. Model 로딩
5. KV Cache 생성: KVCache (F16, head-major)
6. command_executor = None
7. Prefill
8. Decode loop:
   - forward + sample (resilience checkpoint 건너뜀)
   - 100 tokens 생성 후 종료
```

### 6.2 Manager 연결 상태에서 Directive 처리

```
1. ./generate --model-path models/qwen2.5-1.5b --enable-resilience --resilience-transport unix:/tmp/llm.sock
2. CommandExecutor 생성, MessageLoop 시작
3. EngineCapability 전송
4. Decode loop 진입 (engine_state = Running)
5. Token 50: poll() -> ManagerMessage::Directive(KvEvictH2o { keep_ratio: 0.7 })
   -> plan.evict = Some(EvictPlan { H2o, 0.7, Critical })
   -> cache_manager.force_evict_by_policy()
6. Token 51: poll() -> empty -> plan = default (throttle 유지)
```

### 6.3 Experiment 모드

```
1. ./generate --model-path models/qwen2.5-1.5b --experiment-schedule schedule.json
2. 내부 mpsc 채널로 CommandExecutor 생성
3. Token 10: schedule에 따라 Directive 주입 -> plan.evict
4. Token 50: schedule에 따라 Directive 주입 -> plan.suspended = true
```

## 7. Rationale (non-normative)

### 왜 8개 서브시스템인가

22+ 코드 모듈을 기능별로 그루핑하여 아키텍처 이해와 변경 영향 분석을 용이하게 한다. 그루핑은 논리적이며 코드 모듈과 1:1 대응이 아니다.

### 왜 Resilience에 두 경로가 공존하는가

Directive 경로는 Manager가 cross-domain 최적화를 수행한 결과를 Engine에 전달하는 주 경로이다. Strategy 경로는 Manager 없이 D-Bus signal에 직접 반응해야 하는 레거시 환경을 지원한다. DbusTransport가 signal을 ManagerMessage로 변환하여 CommandExecutor에 전달하므로 최종적으로 단일 실행 경로(CommandExecutor)로 수렴한다.

### 왜 CommandExecutor에 전략 로직이 없는가

Engine은 Manager의 cross-domain 최적화 결과를 신뢰하고 1:1로 실행한다. 전략 로직을 Manager에 집중하여 관심사를 분리하고, Engine의 복잡성을 최소화한다.

### 왜 std::thread이고 async가 아닌가

Engine의 주 작업은 CPU-bound forward pass이다. MessageLoop의 blocking recv는 전용 스레드에서 수행하므로 추론에 영향을 주지 않는다. 스레드 수가 3 이하이므로 OS 스레드 오버헤드는 미미하다 (SYS-064 참조).

### 왜 heartbeat_interval이 1000ms 하드코딩인가

현재 구현에서는 설정 가능한 인터페이스를 제공하지 않는다. Manager의 50ms 폴링 주기 대비 충분히 느슨하여 Engine 부하를 최소화하면서도 Manager가 Engine 상태를 주기적으로 파악할 수 있다 (PROTO-071 참조).

## 부록 A. QNN OpPackage cdylib (M1, 2026-05-09)

> **TL;DR**: QNN GPU OpPackage 인터페이스를 구현하는 별도 cdylib 산출물 `crates/qnn_oppkg/`. Engine/Manager/Shared **어느 것에도 의존하지 않으며** (`engine`도 `qnn_oppkg`에 의존하지 않음), QNN runtime이 dlopen하여 호출하는 외부 계약 컴포넌트. INV-001/010/011의 "2 독립 프로세스 + Shared" 골격은 보존된다 — qnn_oppkg는 **Engine 프로세스의 일부도 Manager 프로세스의 일부도 아닌** 외부 라이브러리 산출물이다.

### A.1 정의

| 용어 | 정의 |
|------|------|
| **OpPackage** | QNN(Qualcomm Neural Network) SDK가 정의한 외부 op 등록 인터페이스. `QnnOpPackage_InitInterface`를 export한 cdylib을 SDK가 dlopen하고 `registerOpPackage`로 op들을 그래프에 주입한다. |
| **OpDescriptor** | (op_type, kernel_name, kernel_source, build_options, build_layout fn)의 정적 등록 단위. |
| **OpImplLayout** | 런타임에 op_config로부터 산출되는 (mem_objects, args, workgroup) 묶음. `OpDescriptor::build_layout`이 생성. |
| **PoC crate** | `crates/qnn_oppkg_poc/` — Phase R 검증용. 2 ops 보존. M1 기간 동안 회귀 안전망으로 유지. |

### A.2 Crate 산출물 [ENG-QNN-010]

**[ENG-QNN-010]** `crates/qnn_oppkg/`는 cdylib 산출물이다. `engine`(`llm_rs2`), `manager`(`llm_manager`), `shared`(`llm_shared`) 어느 크레이트도 `qnn_oppkg`에 의존하지 않는다. 역방향 의존도 없다 (qnn_oppkg는 workspace member로만 추가되며 cargo dependency edge는 형성되지 않는다). *(MUST)*

**[ENG-QNN-011]** `qnn_oppkg`는 `engine/kernels/*.cl` 파일을 `include_str!`로만 임베드한다. 커널 인라인 작성을 금지하며, kernel source는 production `.cl` 파일을 단일 진실의 원천(SSOT)으로 한다. *(MUST)*

> **Rationale (non-normative)**: 인라인 kernel은 production `.cl`과 drift 위험. `include_str!`은 빌드 시점에 production 자산을 그대로 복사하여 SSOT를 강제한다.

**[ENG-QNN-012]** M1 범위에서 cdylib은 다음 5개 op을 노출한다: `CustomMatMulF16F32`, `CustomAdd`, `CustomRmsNorm`, `CustomSoftmax`, `CustomSiluMul`. 추가 op(Q4_0, FlashAttn, RoPE, KvScatter)은 M2 이후 범위. *(MUST)*

### A.3 cdylib 외부 계약 [ENG-QNN-020 ~ ENG-QNN-024]

**[ENG-QNN-020]** cdylib은 다음 export symbol을 제공한다 (QNN runtime이 dlopen하여 호출): *(MUST)*

| Symbol | 시그니처 (요약) | 용도 |
|--------|-----------------|------|
| `QnnOpPackage_InitInterface` | `(Qnn_ApiVersion_t version) -> Qnn_OpPackage_Interface_t` | V1.4 + V2.0 fn pointer table 반환. SDK가 op 등록·dispatch·해제 시 이 table을 통해 cdylib 내부 함수를 호출한다. |
| `pkg_get_info` (내부 fn ptr 경유) | `() -> &QnnOpPackage_Info_t` | packageName, numOperations, operationNames, backendApiVersion, coreApiVersion 메타데이터. |
| `pkg_create_op_impl` (내부 fn ptr 경유) | `(node, *mut op_impl) -> Qnn_ErrorHandle_t` | op_type 디스패치 + state pointer 생성. |
| `pkg_free_op_impl` (내부 fn ptr 경유) | `(op_impl) -> Qnn_ErrorHandle_t` | state pointer 해제. **leak 금지** (M1.8). |

**[ENG-QNN-021]** 정적 op 등록 슬라이스 `OPS: &[OpDescriptor]`의 길이는 `numOperations`와 비트 단위 일치한다. M1 범위에서 둘 다 5이다. *(MUST)*

**[ENG-QNN-022]** `OPS` 슬라이스 내 모든 `OpDescriptor.op_type` 값은 슬라이스 내에서 고유하다 (중복 등록 금지). *(MUST)*

**[ENG-QNN-023]** cdylib의 `backendApiVersion`은 `3.7.0`이다 (Phase R에서 결정적 fix). 이 값은 QNN GPU API와 일치하며 변경 시 SDK가 cdylib을 reject한다. *(MUST)*

**[ENG-QNN-024]** `pkg_create_op_impl` dispatcher는 표 기반(`OPS.iter().find(|d| d.op_type == requested)`)이며 if-else 체인을 사용하지 않는다. 이는 op 추가 시 단일 행 등록만으로 디스패치가 확장되도록 보장한다 (Open-Closed). *(SHOULD)*

### A.4 OpPackage 내부 추상화 [ENG-QNN-030]

**[ENG-QNN-030]** OpPackage 내부는 다음 4개 모듈로 분해된다: *(MUST)*

| 모듈 | 책임 |
|------|------|
| `args.rs` | `ArgSpec`(InputTensor/OutputTensor/LocalMem/Int/UInt/ULong/Float), `MemoryObjectSpec`(data_type, flat_dims, flat_offsets), `OpImplLayout`(mem_objects + args + workgroup) — pure data |
| `op_impl.rs` | `build_op_state(descriptor, layout) -> *mut GpuOperation` — 정해진 leak 패턴으로 state 생성. `free_op_impl_state(*mut)` reverse-mapping table 사용. |
| `registry.rs` | `OPS` 정적 슬라이스. `find_descriptor(op_type)` lookup. |
| `static_info.rs` / `interface.rs` | cdylib 메타데이터 + V1.4/V2.0 fn pointer table 채움 |

### A.5 Constraints [ENG-QNN-C01 ~ ENG-QNN-C04]

**[ENG-QNN-C01]** `qnn_oppkg`는 `engine`, `manager`, `shared` 중 어느 크레이트도 의존하지 않는다. 양방향 모두. *(MUST NOT)* (INV-001/010/011 보존)

**[ENG-QNN-C02]** PoC crate `qnn_oppkg_poc`는 M1 기간 동안 회귀 안전망으로 보존된다. 같은 workspace에 공존하며 packageName과 op_type이 분리되어 있어 SDK 등록 충돌이 없다. *(SHOULD)*

**[ENG-QNN-C03]** `engine/kernels/*.cl` 파일은 OpPackage 통합을 이유로 수정하지 않는다 (이미 Adreno 디바이스에서 검증된 자산). *(MUST NOT)*

**[ENG-QNN-C04]** `pkg_free_op_impl`은 PoC의 leak 패턴을 production에서 그대로 사용하지 않는다. M1.8에서 reverse-mapping table + `Box::from_raw` drop으로 정상화된다. 100회 register/free leak test에서 VmRSS slope < 1 KB/iter을 만족해야 한다. *(MUST)*

### A.6 Invariants

| ID | 한줄 요약 |
|----|-----------|
| INV-151 | qnn_oppkg cdylib은 engine/manager/shared와 cargo dependency edge를 형성하지 않는다. (ENG-QNN-C01) |
| INV-152 | `OPS.len() == numOperations` 정적 일치. (ENG-QNN-021) |
| INV-153 | `OPS` 슬라이스 내 op_type 고유성. (ENG-QNN-022) |
| INV-154 | cdylib `backendApiVersion == 3.7.0`. (ENG-QNN-023) |
| INV-155 | 100회 register/free 후 VmRSS slope < 1 KB/iter. (ENG-QNN-C04) |

상세는 `spec/41-invariants.md` §3.22 참조.

### A.7 Examples (non-normative)

```bash
# Android cross-build
python scripts/run_device.py -d s25 --skip-exec build -p qnn_oppkg

# QNN runtime이 cdylib 로드 (개념)
QnnBackend_loadOpPackage(
    backend_handle,
    "/data/local/tmp/libqnn_oppkg.so",   # cdylib path
    "QnnOpPackage_InitInterface",        # ENG-QNN-020 의무 symbol
    NULL                                  // target = QNN_GPU
)
```

### A.8 Rationale (non-normative)

- **왜 별도 crate인가**: cdylib은 `crate-type = ["cdylib"]`을 요구하며 Engine 바이너리(`bin = []`)와 한 크레이트에 둘 수 없다. 또한 QNN runtime이 dlopen하는 외부 계약 단위는 Engine 프로세스 내부 코드와 lifecycle이 다르다.
- **왜 의존성을 0으로 두는가**: qnn_oppkg가 engine 코드를 import하면 (a) 빌드 그래프 거대화, (b) cdylib 사이즈 증가, (c) INV-001/010/011 위반 우려. 5개 op은 모두 self-contained이며 engine의 Tensor/Buffer 추상화를 필요로 하지 않는다 (kernel은 cl_mem과 raw arg만 받음).
- **왜 PoC crate를 보존하는가**: M1 회귀 발생 시 즉각적 rollback path. 두 crate는 독립 packageName으로 SDK에 동시 등록 가능하여 A/B 비교도 지원한다.

### A.9 INV-155 정밀화 (2026-05-09 M2 진입 시점)

M1.8 실측 결과 cdylib 자체 leak는 0이지만(STATE_MAP entries == 0 사후 일관) QNN driver가 op_impl 외 영역(GPU compiled kernel cache, command buffer 풀)에서 잔여물을 유지하여 VmRSS slope이 ~1.1 KB/iter로 측정되었다. 이는 cdylib 책임 외 영역이며 INV-155의 본래 의도(cdylib 자체 leak 부재 보증)와 다른 차원이다. 본 절은 INV-155를 **2-tier**로 분리한다.

**[ENG-QNN-C04']** INV-155 보강(2026-05-09):
- **Primary (cdylib own leak)**: 100회 register/free 후 `STATE_MAP::lock().len() == 0`. cdylib이 보유한 모든 `Box<State>`는 `pkg_free_op_impl`에 의해 소진된다. *(MUST)*
- **Secondary (driver residual tolerance)**: `/proc/self/status::VmRSS` slope (last 50 iter linear regression) < 3 KB/iter. 1.1 KB/iter 실측치는 driver 영역으로 cdylib 책임이 아니지만, 회귀 detector로 임계값을 설정한다. *(SHOULD)*

> **Rationale (non-normative)**: M1.0 spec은 1 KB/iter을 가정했으나 실측은 1.1 KB/iter였다. 임계값을 3 KB/iter로 완화하되 primary detector(STATE_MAP=0)는 그대로 유지하여 cdylib 자체 회귀는 즉시 검출한다.

INV-155의 한 줄 요약은 `41-invariants.md` §3.22에서 갱신된다.

## 부록 B. QNN OpPackage M2 — Layer-level Graph (2026-05-09)

> **TL;DR**: 단일 OpPackage graph로 Qwen 1 transformer layer (12~13 op)을 build/execute한다. 추가 5 op (`CustomMatMulQ40F32`, `CustomFlashAttn`, `CustomRope`, `CustomKvScatter`, `CustomDeqQ40`) + layer graph builder + KV cache integration. **production engine code 변경 0** 유지 (M3에서 backend trait 통합).

### B.1 정의

| 용어 | 정의 |
|------|------|
| **Layer graph** | 단일 transformer layer (RMSNorm → QKV matmul → RoPE → KV scatter → FlashAttn → out_proj → residual → RMSNorm → ffn_gate → silu_mul → ffn_up → ffn_down → residual)를 단일 QNN graph로 빌드한 것. 13 op nodes 이내. |
| **Graph builder** | `(model_weights, kv_cache, layer_idx) → QnnGraph_Handle_t`를 생성하는 호스트 측 빌더. `crates/qnn_oppkg/src/graph/`. |
| **External KV buffer** | `[1, kv_heads, 2048, head_dim]` F16 max-padded 고정 shape KV cache. graph 외부에 host alloc되며 `QNN_TENSOR_TYPE_APP_WRITE`로 graph에 노출. |
| **Intermediate alias** | OpPackage가 SiluMul처럼 in-place kernel을 OOP graph 안에서 사용할 때, kernel 입력/출력 cl_mem을 동일 buffer에 매핑하면서도 graph topology에서 별개 tensor edge로 보이게 하는 ArgSpec 패턴. |
| **OOP refactor** | Out-Of-Place. SiluMul kernel이 입력과 출력을 분리하도록 production `.cl`에 `kernel_silu_mul_oop` 추가하는 변경. **본 spec에서는 채택하지 않음**(B.4 결정). |

### B.2 Op 등록 [ENG-QNN-101 ~ ENG-QNN-105]

**[ENG-QNN-101]** M2 종료 시 cdylib은 다음 op 10개를 노출한다 (M1의 5개 + M2 신규 5개): *(MUST)*

| op_type | Kernel source | 데이터 타입 | 비고 |
|---------|--------------|-----------|------|
| (M1) `CustomMatMulF16F32` | `mul_mv_f16_f32.cl` | F16 weight, F32 act | — |
| (M1) `CustomAdd` | `simple_ops.cl` | F32 | — |
| (M1) `CustomRmsNorm` | `simple_ops.cl` | F32 | — |
| (M1) `CustomSoftmax` | `simple_ops.cl` | F32 | — |
| (M1) `CustomSiluMul` | `simple_ops.cl` | F32 | in-place; B.4 intermediate alias로 OOP 노출 |
| (M2) `CustomMatMulQ40F32` | `mul_mv_q4_0_f32.cl` (또는 production 등가물) | Q4_0 weight raw bytes, F32 act | qkv/ffn/lm_head hot path |
| (M2) `CustomFlashAttn` | `flash_attn_*.cl` (production 선택) | F16 KV, F32 QO | online softmax; max-padded mask 입력 |
| (M2) `CustomRope` | `simple_ops.cl` 또는 `rope.cl` | F32 | `pos`는 Int 스칼라 arg |
| (M2) `CustomKvScatter` | `simple_ops.cl::kernel_kv_scatter_f32_to_f16` | F32 입력 → F16 출력 | HeadMajor 스트라이드 가정 |
| (M2) `CustomDeqQ40` | `simple_ops.cl::kernel_dequant_q4_0` | Q4_0 → F32 | fallback dequant; Q4 GEMV 미지원 시 |

**[ENG-QNN-102]** M2 op 등록은 M1과 동일한 dispatch table 패턴(`OPS: &[OpDescriptor]`)을 따르며, 신규 op 추가는 **OPS 슬라이스에 1행 추가 + ops/ 모듈 1개 추가**만으로 완료된다 (Open-Closed). M1의 `pkg_create_op_impl` table-based dispatcher는 변경하지 않는다 (ENG-QNN-024 보존). *(MUST)*

**[ENG-QNN-103]** M2 신규 op은 모두 `engine/kernels/*.cl`을 `include_str!`로 임베드한다 (ENG-QNN-011 보존). 단, **M2 단계에서 `engine/kernels/`에 신규 .cl 파일을 추가할 수 없다**: production이 이미 보유한 kernel 자산만 사용한다. production이 보유하지 않은 op (예: 별도 RoPE kernel이 없는 경우)은 기존 kernel을 인자만 다르게 호출하여 표현한다. *(MUST)*

> **Rationale (non-normative)**: ENG-QNN-C03("engine/kernels/*.cl 미수정")은 M2에서도 유지된다. 신규 .cl 추가는 production에 영향을 주는 변경(빌드 그래프 + OpenCL 백엔드 신규 자산)이므로 M3 backend trait 통합 시점에 함께 평가한다.

**[ENG-QNN-104]** Q4_0 weight tensor는 `MemoryObjectSpec::data_type = RawBytes(block_size=18, element_count=N/32)`로 노출한다. 32 element block당 18 byte (2 byte F16 scale + 16 byte 4-bit nibbles). 신규 enum variant `RawBytes { block_size: u32, element_count: u32 }` 또는 동등한 abstraction을 `args.rs::MemoryObjectSpec`에 추가한다. *(MUST)*

**[ENG-QNN-105]** RoPE의 `pos` (현재 token 위치)는 `ArgSpec::Int(i32)` 스칼라 arg로 graph 외부에서 주입한다. KV scatter의 `pos`도 동일 패턴. graph build 시점에 fixed 값이 아니므로 매 forward call마다 graph executor가 갱신한다 (QNN graph executor가 `Qnn_OpConfig_t` 인자를 동적으로 update할 수 있는지는 SDK 능력 + M2.1 검증 사항). *(SHOULD)*

### B.3 Layer Graph 구조 [ENG-QNN-110 ~ ENG-QNN-114]

**[ENG-QNN-110]** Layer graph는 다음 입력/출력 인터페이스를 갖는다: *(MUST)*

| 방향 | Tensor | Shape | Dtype | 비고 |
|------|--------|-------|-------|------|
| 입력 | `x` | `[1, dim]` | F32 | residual 입력 |
| 입력 | `kv_cache_k` | `[1, kv_heads, 2048, head_dim]` | F16 | external (APP_WRITE), in/out |
| 입력 | `kv_cache_v` | `[1, kv_heads, 2048, head_dim]` | F16 | external (APP_WRITE), in/out |
| 입력 | `attn_mask` | `[1, 2048]` | F32 | max-padded mask, valid range 외 -inf |
| 입력 (scalar) | `pos` | i32 | — | 현재 token 위치 |
| 입력 | `weights_*` | static | F16/Q4_0 | RMSNorm γ, Wq, Wk, Wv, Wo, RMSNorm γ', Wgate, Wup, Wdown |
| 출력 | `y` | `[1, dim]` | F32 | layer 출력 (다음 layer의 `x`) |
| 출력 (in-place) | `kv_cache_k`, `kv_cache_v` | (위와 동일) | F16 | KvScatter가 graph 안에서 in-place write |

**[ENG-QNN-111]** Layer graph는 13 op nodes 이내로 구성된다 (Qwen2.5-1.5B 기준): *(MUST)*

```
1. RmsNorm(x) → x_norm
2. MatMulQ40F32(x_norm, Wq) → q
3. MatMulQ40F32(x_norm, Wk) → k
4. MatMulQ40F32(x_norm, Wv) → v
5. Rope(q, pos) → q_rot
6. Rope(k, pos) → k_rot
7. KvScatter(k_rot, kv_cache_k, pos)  // in-place write
8. KvScatter(v,    kv_cache_v, pos)   // in-place write
9. FlashAttn(q_rot, kv_cache_k, kv_cache_v, attn_mask, pos) → attn_out
10. MatMulQ40F32(attn_out, Wo) → o_proj
11. Add(x, o_proj) → residual_1     // Add는 in-place; graph topology에서는 OOP
12. RmsNorm(residual_1) → ffn_in
... (FFN: gate / silu_mul / up / down / Add)
```

> 11 ops로 압축 가능 여부는 op fusion 결정 (M2.5).

**[ENG-QNN-112]** Layer graph 빌드는 `crates/qnn_oppkg/src/graph/layer.rs::build_layer_graph(weights, kv_cache_handles, layer_idx) -> QnnGraph_Handle_t` 함수가 단일 진입점이다. 빌더는 `args.rs::OpImplLayout`을 노드별로 생성하여 SDK의 `QnnGraph_addNode` 호출로 그래프를 구성한다. *(MUST)*

**[ENG-QNN-113]** `graphFinalize` 호출은 layer당 1회이며 시간은 200 ms 미만이어야 한다 (디바이스 측정). *(MUST)*

**[ENG-QNN-114]** KV cache buffer 2개(K, V)는 graph 외부에 host alloc되며 `QNN_TENSOR_TYPE_APP_WRITE` (또는 SDK 동등 type)로 graph에 노출된다. graph는 buffer의 ownership을 갖지 않으며 lifecycle은 호출자(graph builder의 caller)가 관리한다. *(MUST)*

### B.4 SiluMul OOP 결정 [ENG-QNN-120]

**[ENG-QNN-120]** SiluMul kernel은 production에서 in-place(`output == input2`)로 작성되어 있어 graph 안에서 host-readable 출력을 생성하지 않는다. M2에서는 다음 옵션 중 **C (intermediate alias 패턴)**을 채택한다: *(MUST)*

| 옵션 | 정확성 | 성능 | production 변경 | abstraction 복잡도 | 채택 |
|------|--------|------|----------------|------------------|------|
| **A**: production `.cl`에 `kernel_silu_mul_oop` 추가 | OK | 동등 (1 GEMV) | **engine/kernels/ 변경** (ENG-QNN-C03 위반) | 낮음 | ✗ |
| **B**: silu_oop + elementwise_mul_oop 2 단계 분해 | OK | -10~20% (커널 2개 + intermediate buffer write) | 신규 mul kernel 필요(C03 위반) 또는 production simple_ops 사용 | 중간 | ✗ |
| **C**: intermediate alias — kernel은 in-place, graph topology에서 input2를 output edge로 alias | OK | 동등 | 0 | OpPackage abstraction에 alias 추가 | ✓ |

> **Rationale (non-normative)**: 옵션 A는 ENG-QNN-C03을 직접 위반. 옵션 B는 mul kernel 추가가 필요하고 성능 손실. 옵션 C는 SDK가 `QNN_TENSOR_TYPE_NATIVE` (graph 내부 tensor)로 동일 backing buffer를 input/output 양쪽에 매핑할 수 있는지 의존하지만, 가능성이 가장 높고 production 자산 보존.

**옵션 C 구현 메모(non-normative)**: `args.rs::ArgSpec`에 `OutputTensorAliased { input_index: u32 }` variant를 추가하거나, `OpDescriptor::build_layout`이 OpImplLayout 생성 시 input/output mem_objects를 동일 cl_mem으로 채워 SDK에 노출. SDK가 동일 buffer alias를 거부하면 옵션 B로 fallback (M2.4 검증 게이트).

### B.5 Production code 변경 0 유지 [ENG-QNN-130 ~ ENG-QNN-132]

**[ENG-QNN-130]** M2 단계 종료 시점까지 `engine/`, `manager/`, `shared/` 어느 크레이트의 소스 라인도 변경되지 않는다 (test 추가 제외 — 단, test가 production 모듈을 import해도 production 빌드 산출물에 영향이 없어야 한다). *(MUST)*

**[ENG-QNN-131]** Layer graph builder, KV cache integration, microbench는 모두 `crates/qnn_oppkg/` 내부 또는 `crates/qnn_oppkg/src/bin/`에 위치한다. 별도 crate 분리는 M2에서 수행하지 않는다 (B.7 결정). *(MUST)*

**[ENG-QNN-132]** M2 검증은 다음 두 산출물로 한정한다: (a) `crates/qnn_oppkg/tests/spec/` host/device test, (b) `crates/qnn_oppkg/src/bin/microbench_qnn_layer_graph.rs` 디바이스 microbench. production engine 바이너리(`generate`, `test_backend`)는 호출하지 않는다. *(MUST)*

### B.6 Pass Gate [ENG-QNN-140 ~ ENG-QNN-143]

**[ENG-QNN-140]** **Accuracy gate**: Qwen2.5-1.5B의 layer 0을 OpPackage graph로 실행한 결과 `y`가 CPU NEON reference 대비 `max_abs_err < 1e-2`(F16 tolerance)를 만족한다. 동일 입력 (`x`, `kv_cache`, `pos`)에 대해 1회 forward + 1회 KV write 후 양쪽이 비교 가능. *(MUST)*

**[ENG-QNN-141]** **TBT gate**: 1 layer를 production OpenCL 경로로 실행하는 baseline 대비 OpPackage 경로의 token-time 비율 ≤ 1.10. baseline은 동일 디바이스(Galaxy S25), 동일 ratio(GPU 100%), 동일 model. 측정 단위는 wall-clock (ENG-QNN feedback: profile-events 금지). *(MUST)*

**[ENG-QNN-142]** **graphFinalize gate**: 1 layer graph build → finalize 시간 ≤ 200 ms (디바이스). *(MUST)*

**[ENG-QNN-143]** **Memory gate**: layer graph 1회 build → execute 100회 → release 사이클의 VmRSS slope < 3 KB/iter. INV-155 v2(secondary tier) 동일 임계값. *(MUST)*

### B.7 Layer Graph Builder 위치 결정 [ENG-QNN-150]

**[ENG-QNN-150]** Layer graph builder는 `crates/qnn_oppkg/src/graph/`에 위치한다. 별도 crate (`qnn_oppkg_graph`)으로 분리하지 않는다. *(MUST)*

| 옵션 | 채택 | Rationale |
|------|------|-----------|
| A: `crates/qnn_oppkg/src/graph/` | ✓ | M2 단계는 op 등록 + graph build 모두 cdylib 외부 계약의 일부. crate 분리 시 cyclic dep (graph가 ops에 의존, ops가 args에 의존) 단순화 효과 미미. |
| B: 별도 crate `crates/qnn_oppkg_graph/` | ✗ | M3에서 production engine 통합 시 backend trait이 graph builder를 직접 호출하면, graph crate은 engine과 cdylib 양쪽에 link되어야 함 — INV-151 위반 위험. |
| C: `engine/src/backend/qnn_oppkg/` | ✗ | production code 변경 0 위반 (ENG-QNN-130). M3 영역. |

### B.8 Constraints [ENG-QNN-C10 ~ ENG-QNN-C14]

**[ENG-QNN-C10]** M2에서 `engine/kernels/*.cl` 파일은 추가/수정/삭제하지 않는다 (ENG-QNN-C03 보존). *(MUST NOT)*

**[ENG-QNN-C11]** M2에서 SiluMul OOP refactor는 abstraction 계층(intermediate alias)에서 해결하며 production kernel을 수정하지 않는다 (B.4 옵션 C). *(MUST NOT)* (production .cl 변경 금지)

**[ENG-QNN-C12]** KV cache는 max-padded fixed shape `[1, kv_heads, 2048, head_dim]`만 지원한다. dynamic seq_len은 M2 범위 외. attention mask는 max-padded이며 valid range 외 token에 `-INFINITY` 주입. *(MUST)*

**[ENG-QNN-C13]** M2 graph는 layer 단위로만 빌드한다. multi-layer (전체 모델) graph는 M3 이후 범위. *(MUST)*

**[ENG-QNN-C14]** Q4_0 weight raw bytes abstraction(`RawBytes` MemoryObjectSpec)은 18 byte/block × N/32 blocks 가정만 지원한다. Q8_0, Q5_K 등 다른 quantization은 M2 범위 외이며 별도 abstraction을 추후 추가한다. *(MUST)*

### B.9 Invariants

| ID | 한줄 요약 |
|----|-----------|
| INV-156 | M2 cdylib `OPS.len() == 10` (M1 5개 + M2 5개). (ENG-QNN-101) |
| INV-157 | M2 모든 신규 op은 production `.cl` 파일을 `include_str!`로 임베드하며 신규 .cl 파일을 추가하지 않는다. (ENG-QNN-103, ENG-QNN-C10) |
| INV-158 | Layer graph node 수 ≤ 13. (ENG-QNN-111) |
| INV-159 | KV cache는 max-padded fixed shape `[1, kv_heads, 2048, head_dim]`. (ENG-QNN-C12) |
| INV-160 | M2 종료 시점 `engine/`, `manager/`, `shared/` 소스 변경 라인 수 == 0. (ENG-QNN-130) |
| INV-161 | Layer 0 OpPackage 출력의 `max_abs_err < 1e-2` (vs CPU NEON reference). (ENG-QNN-140) |
| INV-162 | OpPackage 1 layer TBT ≤ baseline × 1.10. (ENG-QNN-141) |
| INV-163 | `graphFinalize` ≤ 200 ms. (ENG-QNN-142) |
| INV-164 | SiluMul는 production `.cl` 수정 없이 intermediate alias 패턴으로 graph-safe. (ENG-QNN-120) |
| INV-165 | Q4_0 weight는 `MemoryObjectSpec::RawBytes(block_size=18)`로 노출. (ENG-QNN-104) |

상세는 `spec/41-invariants.md` §3.23 참조.

### B.10 Sub-task Pass-gate Map (M2.1 ~ M2.6)

| Sub-task | 산출물 | 검증 게이트 | 통과 조건 |
|----------|--------|------------|----------|
| M2.1 Q4_0 GEMV op | `ops/matmul_q4_0_f32.rs` + RawBytes abstraction | unit test (host) | qkv 단일 op accuracy max_abs_err < 1e-3 |
| M2.2 Rope + KvScatter op | `ops/rope.rs`, `ops/kv_scatter.rs` | unit test (host/device) | scalar pos arg 정상 갱신, F32→F16 cast 정확 |
| M2.3 FlashAttn op | `ops/flash_attn.rs` | device test | max-padded mask attention vs CPU 참조 max_abs_err < 1e-2 |
| M2.4 SiluMul intermediate alias | `args.rs::ArgSpec::OutputTensorAliased` | device test | graph 안에서 host-readable output 생성 + accuracy 보존 |
| M2.5 Layer graph builder | `graph/layer.rs::build_layer_graph` | accuracy + finalize gate | INV-161, INV-163 PASS |
| M2.6 TBT + memory gate | `bin/microbench_qnn_layer_graph.rs` | device microbench | INV-162, INV-143 PASS |

> 본 게이트 모두 PASS 시 M3 (backend trait 통합) 진입 가능.

### B.11 M3 진입 전 한계

- **Multi-layer graph 미지원**: 1 layer 단위 build/execute만 검증. 16 layer 모두 동일 graph 재사용 가능 여부는 M3 검증 사항.
- **Dynamic shape 미지원**: KV cache는 max-padded only. prefill (variable seq_len) 경로는 M3 이후.
- **Backend trait 미통합**: production `Backend` trait의 op 라우팅이 OpPackage graph로 분기하지 않음. `generate` 바이너리에서 호출 불가.
- **Multi-quantization 미지원**: Q4_0만. Q8_0/Q5_K는 별도 RawBytes variant 필요.

### B.12 Examples (non-normative)

```rust
// Layer graph 사용 (개념)
let graph = qnn_oppkg::graph::build_layer_graph(
    &weights[layer_idx],
    &kv_cache_handles,
    layer_idx,
)?;
qnn_oppkg::graph::execute(graph, &x, pos)?;
// kv_cache_handles는 in-place로 갱신됨
```

```bash
# 디바이스 microbench
python scripts/run_device.py -d s25 build -p qnn_oppkg --bin microbench_qnn_layer_graph
python scripts/run_device.py -d s25 exec microbench_qnn_layer_graph -- --layers 1 --iters 100
```
