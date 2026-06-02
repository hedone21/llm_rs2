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
| ResilienceManager | 4종 SystemSignal 수신, SignalLevels 캐시, OperatingMode 계산, 3종 Strategy 위임, `resolve_conflicts()` |
| Strategy 3종 | ComputeStrategy, ThermalStrategy, EnergyStrategy (α-W-3: `MemoryStrategy` 삭제 → graded `Pressure`) |

> **α-W-3 갱신 (`arch/pipeline_stage_design_v2.md` §5.4 drift-sync)**: `MemoryStrategy` 삭제(memory 압력은 graded `Pressure` scalar 경로, `LocalPressureSource` 융합) → strategy 3종. strategy 출력 어휘는 `ResilienceAction`(폐기) → `EngineCommand`(`shared/`)로 통일. manager-less 자율 경로는 `LocalPolicy`(front-door ①)로 재정위. 상세: `31-engine-state.md` §3.5/§3.6.

- DbusTransport 내부에서 SystemSignal을 EngineCommand로 변환한다 (`signal_to_manager_message`)
- ResilienceManager는 `generate.rs` 메인 루프에서 **직접 사용되지 않는다**. DbusTransport가 변환한 결과가 `ManagerMessage::Directive`로 CommandExecutor에 전달된다
- Emergency level signal 수신 시 DbusTransport가 `EngineCommand::Suspend`로 변환하여 Engine이 자율적으로 Suspended 상태에 진입한다 (SYS-055)
- ResilienceManager/Strategy 의 manager-less 정책 적용은 `CommandDispatcher` → `LoopControl`(②control) / `registry.submit`(①KV/weight·③switch)로 일원화된다 (α-W-3, §5.4; 구 `InferenceContext` + `execute_action` 직접 작용 폐기)
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

## 부록 C. QNN-GPU OpPackage M3 Backend (2026-05-10)

> **TL;DR**: M2 layer graph cdylib 자산을 production engine의 `Backend` trait 신규 구현체 `QnnOppkgBackend`로 통합한다. `engine/src/backend/qnn_oppkg/`에 신규 모듈, `--backend qnn_oppkg | qnngpu` opt-in flag, 28× layer graph cache, rpcmem-backed KV. **Pass-gate**: 32-token greedy decode token sequence가 OpenCL backend와 100% 일치 + TBT ≤ 1.20× (GREEN ≤1.10× / YELLOW 1.10~1.20×). M2 INV-160(production change == 0)은 본 단계에서 자연 만료한다.

### C.1 정의

| 용어 | 정의 |
|------|------|
| **qnn_oppkg backend** | `engine/src/backend/qnn_oppkg/` 신규 모듈. `Backend` trait을 OpenCL과 동일 시그니처로 구현하며, fast path는 `execute_layer_graph`를 통해 M2 cdylib의 layer graph를 1회 dispatch한다. |
| **Layer graph cache** | model load 시점에 `N`개 layer (Qwen2.5-1.5B의 경우 28) 각각에 대해 1회 `graphFinalize` 후 `Vec<Arc<LayerGraph>>` 형태로 process lifetime 동안 재사용하는 캐시. eager prebuild (D1 결정). |
| **execute_layer_graph fast path** | `Backend` trait의 신규 method. transformer.rs `forward_into` layer loop에서 `supports_layer_graph()`가 true이면 trait method 1회 호출로 layer 1개의 14 op을 dispatch. 기존 trait method (matmul/rope/etc)는 fallback 경로로만 잔존. |
| **rpcmem KV** | Phase R R-A2/R-Y에서 검증된 DMA-BUF heap allocator. fd→mmap host pointer로 graph 외부 노출과 host-side eviction/quant 정책 동시 가능. |
| **Backend secondary** | `--switch-hw` round-trip을 위해 qnn_oppkg backend가 OpenCL backend를 secondary로 보유. CPU fallback은 본 단계에서 별도. |

### C.2 Backend Module 산출물 [ENG-QNN-201 ~ ENG-QNN-210]

**[ENG-QNN-201]** `engine/src/backend/qnn_oppkg/` 디렉토리를 신규 추가한다. `QnnOppkgBackend`는 `Backend` trait의 모든 필수 method (`matmul`, `matmul_transposed`, `rms_norm`, `rms_norm_oop`, `rope_inplace`, `attention_gen`, `kv_scatter_f32_to_f16_batch`, `flash_attention_prefill`)를 OpenCL backend와 동일 시그니처로 구현한다. *(MUST)*

**[ENG-QNN-202]** Backend dispatch는 `--backend qnn_oppkg | qnngpu` flag로 활성화한다. default off (opt-in). flag 미지정 시 OpenCL/CPU/CUDA 동작은 변경되지 않는다. `feature = "qnn"` cargo flag가 활성일 때만 dispatch에 등장한다. *(MUST)*

**[ENG-QNN-203]** Layer graph cache는 model load 시점에 `N` layer × `graphFinalize` 1회 후 process lifetime 동안 재사용한다. 캐시 invalidation은 weight swap path에서만 발동(M4 영역). `N`은 model 메타데이터의 `n_layers`로 결정 (Qwen2.5-1.5B = 28). *(MUST)*

**[ENG-QNN-204]** KV cache는 rpcmem(DMA-BUF heap)-backed buffer로 alloc되며 mmap된 host pointer를 graph builder의 `KvCacheHandle`에 전달한다. 16 layer × `[1, kv_heads, 2048, head_dim] F16` shape (= layer당 ~1 MB at kv_heads=2, head_dim=128 → 합 ~16 MB)을 보유. M2 INV-159 max-padded fixed shape를 그대로 보존한다. *(MUST)*

**[ENG-QNN-205]** Weight buffer는 `LayerSlot` (`engine/src/models/weights/slot.rs`) snapshot에서 source pointer를 추출하여 graph build 시점에 weight handle로 baked. swap (LayerSlot generation 변경) 발생 시 graph weight handle rebind가 필요하나, M3 단계에서는 swap을 차단(또는 LayerSlot generation 고정 가정)하고, M4 chunk dispatcher가 rebind 정책을 다룬다. *(MUST)*

**[ENG-QNN-206]** Backend secondary는 OpenCL backend로 둔다. `SwitchHw` round-trip 시 qnn_oppkg primary ↔ OpenCL secondary 사이를 이동할 수 있어야 한다. CPU secondary는 본 단계 범위 외 (Qwen NEON path 미검증, MEMORY 참조). *(SHOULD)*

**[ENG-QNN-207]** `transformer.rs::forward_into` layer loop는 `backend.supports_layer_graph()`가 true이면 `backend.execute_layer_graph(layer_idx, &x, &mut kv_cache, pos, &mut x_out)` 1회 호출로 layer 1개를 처리한다. trait method (matmul 등) 호출은 fallback debug 경로로만 남으며, fast path 정상 동작 시 호출 횟수는 0이어야 한다. *(MUST)*

**[ENG-QNN-208]** Prefill (variable seq_len) 경로는 본 단계 범위 외이며 OpenCL backend로 fallback한다 (`--qnn-prefill-fallback opencl`, default 동작). decode-only가 본 단계 범위. *(MUST)*

**[ENG-QNN-209]** `graphFinalize`는 layer당 ≤ 200 ms (INV-163 보존), N=28 layer 직렬 build 시 model load wall-clock 증가량 ≤ ~33 s. eager prebuild (default true, `--qnn-graph-cache-prebuild=true`)가 D1 결정에 따라 채택된다. *(MUST)*

**[ENG-QNN-210]** Attention mask는 host에서 매 token mask buffer만 갱신 후 graph push하는 M2 검증 패턴(D2 결정)을 그대로 사용한다. mask buffer는 model load 시 1회 alloc하여 process lifetime 동안 재사용하며, 매 token에서 `mask[pos]`만 update한다. *(SHOULD)*

### C.3 Backend trait 신규 method [ENG-QNN-211 ~ ENG-QNN-220]

**[ENG-QNN-211]** `Backend` trait에 다음 두 method를 추가한다. 둘 다 default 구현을 가져 기존 backend(CPU/OpenCL/CUDA)는 변경 없이 컴파일된다. *(MUST)*

```rust
fn supports_layer_graph(&self) -> bool { false }

fn execute_layer_graph(
    &self,
    layer_idx: usize,
    x: &Tensor,
    kv_cache: &mut KVCache,
    pos: usize,
    x_out: &mut Tensor,
) -> Result<()> {
    bail!("backend does not implement execute_layer_graph")
}
```

**[ENG-QNN-212]** `supports_layer_graph()`는 idempotent하며 backend 내부 cache 상태에 의존하지 않는다 (model load 후 항상 true 또는 항상 false). transformer.rs는 forward 진입 시점에 1회만 호출하여 분기 결정. *(MUST)*

**[ENG-QNN-213]** `execute_layer_graph`의 사전조건: (a) `layer_idx < n_layers`, (b) `kv_cache`는 INV-159 shape 준수, (c) `pos < 2048`, (d) `x.shape == [1, dim]` F32, (e) `x_out.shape == [1, dim]` F32. 위반 시 `Err`. *(MUST)*

**[ENG-QNN-214]** `execute_layer_graph`의 사후조건: (a) `kv_cache_k[layer_idx][pos]` / `kv_cache_v[layer_idx][pos]` 갱신, (b) `x_out`에 layer 출력 기록, (c) `pos`는 caller가 layer loop 후 increment (backend는 변경하지 않음). INV-014/018 보존. *(MUST)*

**[ENG-QNN-215]** `execute_layer_graph`는 caller 측 `&mut KVCache` 참조의 lifetime 동안만 KV buffer에 접근한다. graph cache는 KV buffer 소유권을 갖지 않는다 (M2 INV-114 정신 보존). *(MUST)*

**[ENG-QNN-216]** trait fallback 호출 instrumentation: qnn_oppkg backend는 fast path 활성 상태에서 `matmul`/`rope_inplace`/`attention_gen`/`kv_scatter_*` 호출 시 debug build에서 panic하거나 release build에서 metric 카운트한다. instrumentation count > 0은 RED 신호 (graph fast path 미발동). *(SHOULD)*

**[ENG-QNN-217]** `enqueue_write_async`/`wait_event_blocking`/`supports_async_transfer` 3 method는 M4 chunk dispatcher가 활용한다. qnn_oppkg backend는 본 단계에서 OpenCL backend의 구현을 위임(secondary backend 경유) 또는 자체 cl_event 큐를 보유한다. M3 단계 선택은 구현 단계 결정사항. *(SHOULD)*

**[ENG-QNN-218]** `invalidate_noshuffle_soa_registry`/`ensure_noshuffle_soa_registered`/`alloc_pre_converted_soa_tensor` 3 weight swap hook은 본 단계에서 noop 또는 secondary backend로 위임한다. M4 chunk swap path가 graph weight rebind 정책을 결정 후 본격 활용. *(SHOULD)*

**[ENG-QNN-219]** Backend dispatch는 `--backend` flag의 default fallback (`_ => bail!("Unknown backend")`)을 보존한다. `qnn_oppkg | qnngpu`는 `feature = "qnn"`이 활성 + flag가 명시 지정된 경우에만 dispatch된다. unknown backend는 여전히 `bail!`한다. *(MUST)*

**[ENG-QNN-220]** `--backend qnn_oppkg`에 대한 CLI Args 추가: (a) `--qnn-graph-cache-prebuild` (default true, D1), (b) `--qnn-allow-fallback` (default false, fast path 실패 시 OpenCL fallback 허용 여부, debug 용도). *(SHOULD)*

### C.4 Layer Graph Contract [ENG-QNN-221 ~ ENG-QNN-230]

**[ENG-QNN-221]** Layer graph는 14 op nodes로 구성된다 (M2 INV-158의 13에서 +1 — RoPE OOP를 Q/K 각각 분리 + Add residual 2개 명시). *(MUST)*

```
1. RmsNormPre(x) → x_norm
2. MatMulQ40F32(x_norm, Wq) → q
3. MatMulQ40F32(x_norm, Wk) → k
4. MatMulQ40F32(x_norm, Wv) → v
5. RopeOOP(q, pos) → q_rot           (kernel_rope_simple_oop)
6. RopeOOP(k, pos) → k_rot           (kernel_rope_simple_oop)
7. KvScatter(k_rot → KV_K, pos)      (multi-output: KV slot WRITE)
8. KvScatter(v    → KV_V, pos)
9. FlashAttn(q_rot, KV_K, KV_V, mask, pos) → attn_out
10. MatMulQ40F32(attn_out, Wo) → o_proj
11. Add(x, o_proj) → r1
12. RmsNormPost(r1) → ffn_in
13. (Gate/Up/SiluMul/Down 합성 노드 또는 4개 분리)
14. Add(r1, ffn_out) → y
```

> 13의 Gate/Up/SiluMul/Down은 op fusion 결정에 따라 1~4 nodes. INV-185는 **build-time** const `LAYER_NODE_COUNT == 14`를 강제하며, fusion 선택이 14를 위반하면 const도 함께 갱신해야 한다.

**[ENG-QNN-222]** RoPE는 production `rope_inplace` 대신 `kernel_rope_simple_oop` (M2.B에서 OpPackage 호환을 위해 추가된 OOP variant)을 사용한다. 이는 M2 단계에서 `engine/kernels/simple_ops.cl`에 추가되었으므로 INV-157을 위반하지 않는다. *(MUST)*

**[ENG-QNN-223]** KvScatter는 multi-output (k_rot → KV_K, v → KV_V) 패턴을 사용한다. 두 KV slot은 graph external buffer (rpcmem-backed, ENG-QNN-204) 이며 graph 내부 tensor가 아니다. 동일 buffer를 graph 안에서 read-after-write 하는 FlashAttn 노드의 input edge로 연결할 때 SDK가 hazard를 자동 처리해야 한다 (M2.E에서 검증). *(MUST)*

**[ENG-QNN-224]** SiluMul은 M2 INV-164의 `OutputTensorAliased` 패턴을 그대로 사용한다. SDK가 옵션 C를 거부하면 옵션 B (silu_oop + mul_oop 2단계 분해) fallback 발동. *(MUST)*

**[ENG-QNN-225]** KV layout view transform: production KVCache는 HeadMajor `[head, pos, dim]` cl_mem stride를 가지며, graph 입력은 `[1, kv_heads, 2048, head_dim]` shape를 기대한다. stride가 동일한 메모리 layout이므로 **buffer reshape 비용 0**의 view transform만으로 입력 전달이 가능해야 한다. 위반 시 (메모리 copy 필요) RED. *(MUST)*

**[ENG-QNN-226]** Layer graph 입출력 인터페이스는 M2 ENG-QNN-110의 표를 그대로 재사용한다. 단, `weights_*` static 입력은 28 layer 각각의 LayerSlot snapshot에서 추출된다. *(MUST)*

**[ENG-QNN-227]** `pos: i32` scalar arg는 매 forward call마다 graph executor가 갱신한다 (M2 ENG-QNN-105). RoPE Q/K, KvScatter K/V, FlashAttn 5 노드가 동일 pos를 공유한다. *(MUST)*

**[ENG-QNN-228]** Attention mask buffer는 model load 시 1회 alloc된 `[2048] F16` 또는 `[1, 2048] F32` (production reference와 동일 dtype)이며, 매 token에서 `mask[pos]`에 valid mark를 push한다. mask buffer는 graph 외부 alloc + APP_WRITE. *(SHOULD)*

**[ENG-QNN-229]** Q4_0 weight tensor는 M2 INV-165 (`MemoryObjectSpec::RawBytes { block_size: 18, element_count: N/32 }`)를 그대로 사용한다. weight handle은 LayerSlot snapshot의 cl_mem (또는 rpcmem) 주소를 가리키며 graph build 시점에 baked. *(MUST)*

**[ENG-QNN-230]** Layer graph는 `crates/qnn_oppkg::graph::layer::build_layer_graph`를 직접 재사용한다 (M2 산출물). engine 측은 caller 위치만 변경되며 graph builder 자체는 수정하지 않는다. *(MUST)*

### C.5 Pass Gate [ENG-QNN-231 ~ ENG-QNN-240]

**[ENG-QNN-231]** **Accuracy gate (절대)**: Qwen2.5-1.5B Q4_0 GGUF, 32-token greedy decode (top-1, 동일 RNG seed=42, 동일 prompt), `--backend qnn_oppkg`로 생성된 token sequence가 `--backend opencl` 결과와 100% 일치한다. 1개 token이라도 다르면 RED. *(MUST)*

**[ENG-QNN-232]** **Single-layer accuracy gate**: layer 0 isolation 시 OpenCL backend의 layer 0 출력 `y` 대비 `max_abs_err < 1e-2` (F16 tolerance). M2 INV-161의 production 확장. *(MUST)*

**[ENG-QNN-233]** **TBT gate**: Galaxy S25, 동일 prompt 5회 평균 (warm-up 3회 제외), wall-clock 측정. baseline = `--backend opencl` decode TBT. (a) GREEN ≤ 1.10×, (b) YELLOW 1.10×~1.20×, (c) RED > 1.20×. RED 시 사용자 호출. *(MUST)*

**[ENG-QNN-234]** **VmRSS gate**: 32-token decode 후 `/proc/self/status::VmRSS` ≤ baseline × 1.10. baseline은 `--backend opencl` 기준. 추가 비용 = rpcmem KV (~16 MB) + graph metadata (~100 MB) + secondary OpenCL context (~수십 MB). *(MUST)*

**[ENG-QNN-235]** **VmRSS slope gate**: 32-token decode 동안 token당 VmRSS 증가 < 50 KB/token. INV-155 v2 secondary tier (3 KB/iter)와는 다른 영역 (token 단위 leak detector). *(SHOULD)*

**[ENG-QNN-236]** **graphFinalize gate**: layer당 ≤ 200 ms (M2 INV-163 보존). 28 layer 직렬 build wall-clock ≤ 33 s. *(MUST)*

**[ENG-QNN-237]** **Regression gate**: `cargo test --workspace --features opencl` (qnn 비활성) 0건 회귀. `cargo test --workspace --features qnn,opencl` 도 0건 회귀 (qnn enable이지만 backend 미선택 시 OpenCL 정상). INV-169 검증 핵심 게이트. *(MUST)*

**[ENG-QNN-238]** **OpenCL TBT 무회귀 gate**: `--backend opencl` decode TBT가 M3.0 진입 직전 baseline 대비 ≤ 1.05× (5% tolerance). Backend trait 신규 method 추가가 hot path overhead를 도입하지 않음을 검증. *(MUST)*

**[ENG-QNN-239]** **Trait fallback count gate**: qnn_oppkg backend로 32 token decode 후 trait method (matmul/rope/etc) 호출 instrumentation count == 0. fast path 정상 발동 검증. *(SHOULD)*

**[ENG-QNN-240]** TBT 측정은 wall-clock only (`CL_QUEUE_PROFILING_ENABLE`/`--profile-events` 금지, M2 ENG-QNN-141 정신 보존). 6T 스레드만 사용 (Galaxy S25 벤치 가이드). *(MUST)*

### C.6 Constraints

**[ENG-QNN-C20]** `--backend qnn_oppkg`는 default off. unknown backend 거부 (`bail!`)는 그대로 유지된다. *(MUST)* (INV-170)

**[ENG-QNN-C21]** M3 단계에서 Q4_0 weight 외 quantization (Q8_0/Q5_K/F16) 지원은 범위 외이며 OpenCL backend로 fallback한다. M2 ENG-QNN-C14 보존. *(MUST)*

**[ENG-QNN-C22]** prefill (variable seq_len)은 본 단계 범위 외 — OpenCL backend로 fallback. M2 ENG-QNN-C12/C13 보존. *(MUST)*

**[ENG-QNN-C23]** D7 결정에 따라 production 변경은 최소화하되, 필요 시 `engine/kernels/*.cl` 포함 수정 가능. 단 OpenCL backend 정확성/TBT 무회귀 (INV-169)가 핵심 게이트로 작동. M2 INV-160 (production change == 0)는 본 단계에서 자연 만료. *(MUST)*

**[ENG-QNN-C24]** `crates/qnn_oppkg/`의 host metadata (`LayerConfig` 등)에 대해 `engine` 크레이트가 cargo dependency edge를 형성한다. cdylib 자체 (binary artifact) 의존이 아닌 host-side rust crate 의존이며, INV-151 (cdylib ⊥ engine)의 본래 정신은 보존된다 — cdylib은 여전히 dlopen 산출물이고 engine bin은 cdylib을 link하지 않는다. *(MUST)*

### C.7 Invariants

| ID | 한줄 요약 |
|----|-----------|
| INV-166 | qnn_oppkg backend는 `Backend` trait의 모든 필수 method를 OpenCL과 동일 시그니처로 구현. (ENG-QNN-201) |
| INV-167 | Layer graph는 N(=28)회 graphFinalize 후 process lifetime 동안 재사용. invalidation은 weight swap path에서만. (ENG-QNN-203) |
| INV-168 | KV cache shape는 M2와 동일 `[1, kv_heads, 2048, head_dim] F16` max-padded. M3 dynamic seq_len 미지원. (ENG-QNN-204, INV-159 보존) |
| INV-169 | OpenCL backend 정확성/TBT 무회귀 — `cargo test --workspace --features opencl` 0건 + `--backend opencl` decode TBT ≤ 1.05× baseline. (ENG-QNN-237/238) |
| INV-170 | `--backend qnn_oppkg`는 default off. unknown backend는 `bail!`. (ENG-QNN-219, ENG-QNN-C20) |
| INV-171 | KV cache는 rpcmem(DMA-BUF heap)-backed + host pointer expose. (ENG-QNN-204) |
| INV-172 | Qwen2.5-1.5B Q4_0 32-token greedy decode = OpenCL backend top-1 token sequence 100% 일치. (ENG-QNN-231) |
| INV-173 | TBT 측정은 wall-clock only (`--profile-events` 금지). (ENG-QNN-240) |
| INV-174 | qnn_oppkg backend의 `supports_layer_graph()`는 model load 후 항상 true 또는 항상 false (idempotent). (ENG-QNN-212) |
| INV-175 | qnn_oppkg backend로 32 token decode 후 trait method (matmul/rope/etc) 호출 count == 0 (fast path 발동 보증). (ENG-QNN-239) |
| INV-176 | Layer graph node 수 == 14 (M2의 13에서 +1, build-time const `LAYER_NODE_COUNT`와 동기화). (ENG-QNN-221) |
| INV-177 | KV layout view transform은 buffer copy 비용 0 — production HeadMajor stride와 graph 입력 stride 동일. (ENG-QNN-225) |
| INV-178 | 32-token decode 동안 VmRSS slope < 50 KB/token. (ENG-QNN-235) |
| INV-179 | TBT GREEN ≤ 1.10× / YELLOW 1.10~1.20× / RED > 1.20× (Galaxy S25, 5회 평균, warm-up 3회 제외). (ENG-QNN-233) |
| INV-180 | qnn_oppkg backend는 `crates/qnn_oppkg`의 host metadata에 cargo dependency edge를 형성하지만, cdylib binary는 dlopen 산출물로 engine binary가 link하지 않는다. (ENG-QNN-C24, INV-151 본래 정신 보존) |

상세는 `spec/41-invariants.md` §3.24 참조.

### C.8 Examples (non-normative)

```bash
# qnn_oppkg backend로 32-token greedy decode
python scripts/run_device.py -d s25 generate -- \
    --backend qnn_oppkg \
    --model-path models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
    --prompt-file fixtures/prompt_a.txt \
    --max-new-tokens 32 \
    --seed 42

# accuracy 비교 (vs OpenCL)
diff \
  <(... --backend opencl ... --max-new-tokens 32 --seed 42) \
  <(... --backend qnn_oppkg ... --max-new-tokens 32 --seed 42)
# 빈 diff 이면 INV-172 PASS
```

### C.9 Rationale (non-normative)

- **왜 layer graph 단위 dispatch인가**: M2에서 검증된 14-node single-layer graph는 transformer.rs forward_gen 1 layer와 1:1 매핑된다. graph 단위 dispatch는 (a) trait method 호출 overhead 제거, (b) QNN graph optimizer (kernel fusion 등) 활용, (c) layer 단위 weight swap hook 자연 정렬을 모두 얻는다.
- **왜 default off인가**: M3 단계는 OpenCL backend의 무회귀가 핵심 게이트다. opt-in flag로 선택적 활성하여 회귀 위험을 격리한다.
- **왜 INV-160 약화인가**: M2까지 production 변경 0이 외형적 격리 보증이었으나, M3은 backend trait 통합이 본질이므로 trait 신규 method 추가가 불가피. 대체 게이트(INV-169 무회귀)가 동일 정신을 보존한다.
- **왜 eager prebuild인가**: 28 layer × graphFinalize ~33s를 매 token 첫 사용 시점에 분산하면 lazy compilation cache miss로 인한 TBT spike를 유발. eager는 초기 load wall-clock을 희생하여 production-grade decode TBT를 확보. D1 결정.

## 부록 D. QNN-GPU OpPackage M4 Async Swap (2026-05-10, placeholder)

> **TL;DR**: M3 layer graph cache 자산을 활용하여 14-node DAG의 정적 phase analyzer + chunk swap dispatcher를 도입한다. cache-fit phase 진입 시 weight chunk를 `enqueue_write_async`로 dispatch하고 DDR-heavy phase 시작 직전에 wait. 본 부록은 M3.4 메인 게이트 통과 후 본격 채워지며, 여기서는 M3 단계에서 미리 잡아두어야 할 seam ID와 placeholder만 명시한다.

### D.1 정의 (placeholder)

| 용어 | 정의 |
|------|------|
| **Phase analyzer** | 14-node static DAG를 DDR-heavy 7개 / cache-fit 9개로 분류하는 build-time const table. runtime 비용 0. |
| **Chunk swap dispatcher** | weight tensor를 1/2/4/8/16 MB chunk로 분할하여 cache-fit phase 동안 `enqueue_write_async`로 dispatch + DDR-heavy phase 직전 `wait_event_blocking`. |
| **Hide ratio** | `1 - (overlapped time / forward time)`. swap pause time을 forward 동안 얼마나 가려냈는지 비율. |
| **Graph weight rebind** | chunk swap으로 GPU-side weight buffer 주소가 변경된 경우, graph cache의 weight handle을 업데이트하는 SDK 호출 (옵션 A) 또는 graph 재build (옵션 B fallback). |

### D.2 신규 ID 예약 [ENG-QNN-301 ~ ENG-QNN-320]

| ID | placeholder |
|----|-------------|
| ENG-QNN-301 | Phase analyzer (14-node static DAG) DDR-heavy 7 nodes / cache-fit 9 nodes 분류 |
| ENG-QNN-302 | Chunk swap dispatcher — chunk size sweep {1, 2, 4, 8, 16} MB, default 4 MB 가설 |
| ENG-QNN-303 | Hide ratio ≥ 20% 1점 PASS 게이트 (M4 메인) |
| ENG-QNN-304 | `Backend::enqueue_write_async` + `wait_event_blocking` 활용 |
| ENG-QNN-305 | Phase 6.5 인프라 재사용 (LayerSlot/SecondaryMmap/IntraForwardSwapHook 확장/SwapExecutor `build_chunk_from_mmap_async` 추가) |
| ENG-QNN-306 | Graph weight handle rebind 정책 (옵션 A SDK API / 옵션 B 재build, M4.1 spike 후 결정) |
| ENG-QNN-307 | Chunk swap은 qnn_oppkg backend 한정. CPU/OpenCL은 NoOp fallback |
| ENG-QNN-308 ~ ENG-QNN-320 | M4.1~M4.3 단계 진입 시 본문 채움 |

### D.3 Invariants (placeholder)

| ID | 한줄 요약 |
|----|-----------|
| INV-181 | Chunk swap은 qnn_oppkg backend 한정. CPU/OpenCL은 NoOp fallback. |
| INV-182 | Chunk dispatch 시작은 cache-fit phase 진입 시점. DDR-heavy phase에서는 dispatch 금지. |
| INV-183 | Chunk size sweep = {1, 2, 4, 8, 16} MB. |
| INV-184 | Phase analyzer는 14-node static DAG 기반 const table — runtime 비용 0. |
| INV-185 | `LAYER_NODE_COUNT == 14` (qnn_oppkg::graph) 동기화 build-time check. |
| INV-186 | `wait_event_blocking`은 다음 token decode 시작 전에 호출 (forward 도중 main queue 비차단). |
| INV-187 | Hide ratio ≥ 20% 1점에서라도 PASS면 GREEN. |
| INV-188 | Swap on/off 토큰 시퀀스 100% 일치 (정확성). |

상세는 `spec/41-invariants.md` §3.25 참조.

### D.4 Constraints (placeholder)

본 부록의 본문은 M4.0 진입 시 (M3.4 메인 게이트 통과 + Auto-Gate) 채운다. `arch/weight_swap.md` §11에 컴포넌트 도식만 placeholder로 작성.

## 부록 E. RpcmemAllocator (Sprint 2a Phase 2, 2026-05-26)

> **TL;DR**: `libcdsprpc.so` 3 심볼(`rpcmem_alloc`/`rpcmem_free`/`rpcmem_to_fd`)을 단일 책임 모듈 `engine/src/memory/rpcmem/allocator.rs`로 격리하고, OpenCL backend 가 `--opencl-rpcmem` 활성 시 KV cache zero-copy + RpcmemSecondaryStore precision swap 두 consumer 에게 동일 `Arc<RpcmemAllocator>` 인스턴스를 주입한다. `qnn_oppkg` backend 가 보유하던 `libQnnGpu.so` / `libqnn_oppkg.so` dlopen 경로는 본 단계 이후 dead path 가 되며 Sprint 2b 에서 backend 자체와 함께 제거된다. INV-001/010/011(2 프로세스 + Shared) 와 INV-170(qnn_oppkg opt-in) 정신은 보존되며, libcdsprpc.so 만 production code path 로 승격된다.

### E.1 정의

| 용어 | 정의 |
|------|------|
| **rpcmem heap** | Qualcomm FastRPC 가 노출하는 DMA-BUF heap. `libcdsprpc.so` 의 `rpcmem_alloc`/`rpcmem_free` 로 alloc/free 하며, `rpcmem_to_fd` 로 DMA-BUF fd 를 얻는다. CPU↔GPU↔HTP 가 동일 물리 페이지를 공유 (Adreno UMA). |
| **RpcmemAllocator** | `libcdsprpc.so` 의 dlopen 핸들 + 3 fn-pointer 캐시를 보유한 단일 책임 모듈. process lifetime 동안 단일 인스턴스. |
| **`--opencl-rpcmem`** | OpenCL backend 옵션. 활성 시 (a) `OpenCLMemory::alloc_kv` 가 rpcmem + `CL_MEM_USE_HOST_PTR` alias 경로 사용, (b) precision swap 의 secondary store 가 `RpcmemSecondaryStore` 변형 사용. default off (opt-in). |
| **RpcmemAliasBuffer** | rpcmem host pointer 를 OpenCL `CL_MEM_USE_HOST_PTR` alias 로 감싼 zero-copy 버퍼. 기존 `engine/src/memory/rpcmem/opencl_alias.rs` 모듈을 재사용. |
| **Self-secondary** | AUF multi-dtype variant 에서 primary 와 secondary 가 같은 AUF 파일을 공유하는 precision swap path (W-AUF-2). RpcmemAllocator 주입 path 가 GGUF secondary 와 동일하다. |

### E.2 Allocator 모듈 책임 [ENG-RPCMEM-010 ~ ENG-RPCMEM-013]

**[ENG-RPCMEM-010]** `engine/src/memory/rpcmem/allocator.rs` 는 `libcdsprpc.so` 의 `dlopen` 과 `rpcmem_alloc` / `rpcmem_free` / `rpcmem_to_fd` 3 심볼 lookup 을 수행하는 단일 진입점이다. cdsprpc symbol 을 직접 `libloading::Symbol`로 가져오는 다른 production 모듈을 추가 금지한다 (테스트/microbench 는 제외). *(MUST)*

**[ENG-RPCMEM-011]** `RpcmemAllocator::new()` 는 호스트(non-Android) 타겟에서 컴파일에 포함되지 않거나 (`#[cfg(target_os = "android")]`) 호출 시 `Err`을 반환한다. 호출자(OpenCLBackend init / RpcmemSecondaryStore loader) 는 `Err` 시 `--opencl-rpcmem` 을 자동으로 비활성화 (warning 1회 stderr) 하여 host build 가 계속 진행되어야 한다. *(MUST)*

**[ENG-RPCMEM-012]** `RpcmemAllocator` 는 `Send + Sync` 이며 process lifetime 동안 단일 인스턴스가 `Arc<RpcmemAllocator>` 로 OpenCLBackend 와 RpcmemSecondaryStore 양쪽에 공유된다. Backend init 시점에 1회 alloc 후 model load / swap path 가 동일 Arc 를 clone 한다. *(MUST)*

**[ENG-RPCMEM-013]** `RpcmemAllocator` 의 Drop 은 `libcdsprpc.so` 의 `dlclose` 만 수행하며 outstanding `host_ptr` 의 `rpcmem_free` 는 호출하지 않는다. buffer lifetime (`QnnOppkgKvBuffer` 후신 / `RpcmemLayerRegion`) 이 각자의 Drop 에서 `rpcmem_free` 를 호출한다 (allocator lifetime ⊃ 모든 buffer lifetime 이 보장되어야 한다 — INV-RPCMEM-005). *(MUST)*

### E.3 OpenCL Backend Wire-up [ENG-RPCMEM-020 ~ ENG-RPCMEM-024]

**[ENG-RPCMEM-020]** `OpenCLBackend::new_with_profile_events` 의 시그니처는 변경하지 않는다. 신규 `OpenCLBackend::new_with_options(profile_events_enabled, opencl_rpcmem)` 를 추가하고 기존 `new`/`new_with_profile_events` 는 `opencl_rpcmem = false` 로 위임한다. (호환성 우선) *(SHOULD)*

**[ENG-RPCMEM-021]** `opencl_rpcmem = true` 시 `OpenCLBackend` 는 `Arc<RpcmemAllocator>` 를 lazy init (첫 alloc 시점에 한 번) 또는 `new_with_options` 진입 시점에 eager init 한다. 본 단계는 eager init 을 선택한다 (lazy 의 첫 alloc 지점은 hot path 라 init failure visibility 낮음). `RpcmemAllocator::new()` 가 `Err` 일 경우 `OpenCLBackend::new_with_options` 가 `opencl_rpcmem = false` 로 강등하고 stderr 에 1회 경고를 출력한다 (ENG-RPCMEM-011 호환). *(MUST)*

**[ENG-RPCMEM-022]** `OpenCLMemory` 는 `opencl_rpcmem` 활성 시 `Arc<RpcmemAllocator>` 를 보유한다. `alloc_kv(size, dtype)` 는:
1. `opencl_rpcmem == false` → 기존 `UnifiedBuffer` (Phase 0) 경로 유지.
2. `opencl_rpcmem == true` → `RpcmemAllocator::alloc(size)` 호출 → `CL_MEM_USE_HOST_PTR` alias 생성 → KV-shape `RpcmemKvBuffer` (구 `QnnOppkgKvBuffer` 의 후신, Sprint 2a Phase 3 에서 신설 또는 기존 코드 이동) 반환.
3. rpcmem alloc 실패 시 `UnifiedBuffer` fallback (per-buffer fallback, session 전체 abort 금지 — INV-RPCMEM-003). *(MUST)*

**[ENG-RPCMEM-023]** `OpenCLMemory::alloc` (activation tensor) 은 `opencl_rpcmem` 값과 무관하게 기존 `OpenCLBuffer`/`UnifiedBuffer` 경로만 사용한다. rpcmem 은 KV cache 와 precision swap secondary 전용. activation 은 short-lived 라 zero-copy benefit 이 없으며 rpcmem heap 단편화 위험만 증가. *(MUST)*

**[ENG-RPCMEM-024]** `OpenCLBackend::get_extension(EXT_RPCMEM_ALLOCATOR)` 가 `opencl_rpcmem` 활성 시 `Arc<RpcmemAllocator>` 의 raw view (`&dyn Any`) 를 반환한다. RpcmemSecondaryStore loader 가 본 extension 으로 allocator 핸들을 획득한다 (현 `EXT_QNN_OPPKG` downcast 패턴을 그대로 따름). 비활성 시 `None`. *(MUST)*

### E.4 Precision Swap Wire-up [ENG-RPCMEM-030 ~ ENG-RPCMEM-033]

**[ENG-RPCMEM-030]** `RpcmemSecondaryStore::from_gguf` / `RpcmemSecondaryStore::from_auf_self_secondary` 의 마지막 parameter `(RpcmemAllocFn, RpcmemFreeFn)` 을 `Arc<RpcmemAllocator>` 로 교체한다. 내부 `build_layer_region` 은 `allocator.alloc(total)` 을 호출하고 `RpcmemLayerRegion` 은 `Arc<RpcmemAllocator>` (또는 `RpcmemFreeFn` 추출) 를 보유하여 Drop 시 free 한다. fn-pointer 직접 보유 대신 `Arc<RpcmemAllocator>` 보유를 권장 (lifetime 종속 명시화). *(MUST)*

**[ENG-RPCMEM-031]** `secondary_mmap.rs::try_open_rpcmem_secondary` 와 `try_open_rpcmem_self_secondary_for_auf` 의 backend lookup 분기는 다음 우선순위로 변경된다:
1. `EXT_RPCMEM_ALLOCATOR` (신규) 가 `Some(allocator)` → 본 allocator 사용.
2. (Sprint 2a 유지) `EXT_QNN_OPPKG` 가 `Some(qnn_oppkg)` → 기존 `QnnOppkgRuntime::rpcmem_fns()` 사용. (Sprint 2b backend 제거 시 본 분기도 삭제)
3. 둘 다 `None` → `SecondaryUnavailable` 로 GGUF/AUF 일반 mmap path 로 fallback.

`backend_supports_rpcmem_secondary(backend)` 는 두 extension 중 하나라도 `Some` 일 때 true 를 반환한다. *(MUST)*

**[ENG-RPCMEM-032]** RpcmemSecondaryStore 는 OpenCLBackend 가 보유한 `Arc<RpcmemAllocator>` 와 동일 인스턴스를 받는다 (clone 으로 공유). 같은 process 안에서 두 개의 RpcmemAllocator 가 alloc 되어 별도 dlopen 핸들을 갖는 상태는 금지된다 (INV-RPCMEM-002). *(MUST)*

**[ENG-RPCMEM-033]** `RpcmemSecondaryStore::backend_weak` 로 보유되는 `Weak<dyn Backend>` 는 `OpenCLBackend` 를 가리키며, LISWAP-6 Phase 1 alias cache populate 시 `Backend::alloc_alias_weight_buffer` 호출 경로는 기존(M3) 그대로 유지된다. 즉 본 부록의 변경은 alloc 분리만 다루며 alias 생성 경로(`opencl/mod.rs::alloc_alias_weight_buffer`) 는 무변경. *(MUST)*

### E.5 CLI / Session Wire-up [ENG-RPCMEM-040 ~ ENG-RPCMEM-042]

**[ENG-RPCMEM-040]** `engine/src/session/cli/mod.rs::CliArgs` 에 `pub opencl_rpcmem: bool` (default false, `--opencl-rpcmem`) flag 를 추가한다. doc comment 는 (a) Android-only, (b) KV cache + precision swap secondary 둘 다 활성화됨, (c) Sprint 2a verification gate 측정용임을 명시. *(MUST)*

**[ENG-RPCMEM-041]** Sprint 2a (본 부록 도입 단계) 에서 `--opencl-rpcmem` 와 `--backend qnn_oppkg | qnngpu` 가 동시 지정될 경우 stderr 경고 1회 후 `--opencl-rpcmem` 가 무시된다 (qnn_oppkg backend 가 자체 rpcmem path 보유 — 중복 활성 방지). Sprint 2b 에서 `qnn_oppkg` backend 가 제거되면 본 mutex 도 삭제된다. *(MUST)*

**[ENG-RPCMEM-042]** `session/init.rs::build_runtime` 의 `"opencl"` 분기는 `OpenCLBackend::new_with_options(profile_events, args.opencl_rpcmem)` 로 변경되며, 반환된 backend 를 `EXT_RPCMEM_ALLOCATOR` extension 으로 노출한다. `"qnn_oppkg"` 분기는 무변경 (Sprint 2b 에서 삭제). `"cpu"` / `"cuda"` 분기는 `--opencl-rpcmem` 를 무시한다 (stderr 경고 1회). *(MUST)*

### E.6 Constraints [ENG-RPCMEM-C01 ~ ENG-RPCMEM-C04]

**[ENG-RPCMEM-C01]** `RpcmemAllocator` 는 `libQnnGpu.so` 또는 `libqnn_oppkg.so` 를 dlopen 하지 않는다. QNN-GPU 의존 제거가 본 모듈의 핵심 목적. *(MUST NOT)*

**[ENG-RPCMEM-C02]** Sprint 2a Phase 2 본 단계에서 `engine/src/backend/qnn_oppkg/` 의 어떤 코드도 수정/삭제하지 않는다. 두 path (qnn_oppkg backend / `--opencl-rpcmem`) 가 공존하며 회귀 검증 안전망으로 유지된다. Sprint 2b backend 제거 단계에서 비로소 삭제. *(MUST NOT)*

**[ENG-RPCMEM-C03]** `--opencl-rpcmem` 활성 시에도 fast feasibility 측정 (Phase 10 HeteroLLM 재현) 결과의 raw `clientBuf` slow path (0.04 GB/s) 는 회피한다. 즉 `RpcmemAliasBuffer` (`CL_MEM_USE_HOST_PTR` alias) 만 사용하며 OpenCL backend 가 별도의 `clientBuf` import 를 추가하지 않는다. *(MUST NOT)*

**[ENG-RPCMEM-C04]** `RpcmemAllocator` 모듈은 INV-LAYER-001/002 를 준수한다. `memory/rpcmem/allocator.rs` 는 L2 (`memory/`) 에 위치하며 L3 (`models/`, `pressure/`, `inference/`) 의 어떤 모듈도 import 하지 않는다. allocator 가 노출하는 API 는 `unsafe fn alloc(&self, size) -> Result<(*mut u8, RawFd)>` / `unsafe fn free(&self, host_ptr)` 의 raw byte interface 만 유지한다. *(MUST)*

### E.7 Invariants

| ID | 한줄 요약 |
|----|-----------|
| INV-RPCMEM-001 | RpcmemAllocator 는 Android target 에서만 컴파일된다 (host 는 컴파일 자체에서 제외 또는 `new()` 가 `Err`). |
| INV-RPCMEM-002 | `--opencl-rpcmem` 활성 시 OpenCLBackend 와 RpcmemSecondaryStore 는 동일 `Arc<RpcmemAllocator>` 인스턴스를 공유한다 (single allocator per session). |
| INV-RPCMEM-003 | rpcmem alloc 실패는 per-buffer fallback (UnifiedBuffer 또는 SecondaryUnavailable) — session abort 금지. |
| INV-RPCMEM-004 | `RpcmemAllocator` 는 `libQnnGpu.so` / `libqnn_oppkg.so` 를 dlopen 하지 않는다 (libcdsprpc.so 만). |
| INV-RPCMEM-005 | `RpcmemAllocator::Drop` 시점에 모든 rpcmem buffer (RpcmemKvBuffer / RpcmemLayerRegion) 는 이미 drop 되어 있어야 한다 (lifetime 포함 관계). |
| INV-RPCMEM-006 | `--opencl-rpcmem` 와 `--backend qnn_oppkg` 동시 지정 시 본 sprint(2a)에서는 전자가 무시되며 stderr 경고 1회 출력. |
| INV-RPCMEM-007 | `OpenCLMemory::alloc` (activation) 은 `opencl_rpcmem` 값과 무관하게 rpcmem heap 을 사용하지 않는다 (KV/secondary 전용). |
| INV-RPCMEM-008 | `RpcmemAliasBuffer` 의 backing host_ptr 은 allocator 가 alloc 한 rpcmem 영역 안에만 존재한다 (raw `clientBuf` import 금지 — Phase 10 결과 회피). |

상세는 `spec/41-invariants.md` §3.27 참조.

### E.8 Rationale (non-normative)

#### E.8.1 왜 별도 backend 가 아닌 OpenCLBackend 의 옵션인가

`qnn_oppkg` backend 는 raw QNN-GPU graph execution path (M3 layer graph) 와 rpcmem KV 두 가지를 묶어 도입되었으나 paper main evidence 측정 (`project_qnn_oppkg_m2_complete_20260510.md`, `project_swap_overhead_opencl_complete_20260509.md`) 결과:
- M3 graph path 의 production code 진입은 fast path 미사용 — `--backend qnn_oppkg fast off` 가 production default.
- 실측 가속의 95% 가 rpcmem DMA-BUF + USE_HOST_PTR alias (KV zero-copy) 에서 기인.

따라서 `qnn_oppkg` 라는 backend ID 를 유지할 정당성이 약하며, OpenCLBackend 가 옵션으로 rpcmem path 를 활성화하는 것이 SOLID 의 SRP / OCP 양쪽을 만족한다.

#### E.8.2 왜 lazy init 이 아닌 eager init 인가

`OpenCLBackend::new_with_options` 진입 시점에 `RpcmemAllocator::new()` 가 실패하면 `--opencl-rpcmem` 가 자동 강등된다. lazy init (첫 KV alloc 시점) 은 model load 후의 hot path 에서 dlopen 실패가 드러나 디버깅 비용이 증가한다. eager init 의 비용은 dlopen 3 symbol lookup + zero allocation 으로 ~수 ms 수준.

#### E.8.3 왜 `--opencl-rpcmem` 가 단일 flag 로 KV + secondary 양쪽을 활성화하는가

두 consumer 는 사실상 같은 allocator 위에서 동작하며, KV 만 끄거나 secondary 만 끄는 모드는 실측 use case 가 없다. flag 분리는 표면적 유연성을 위해 인터페이스를 늘릴 뿐 실측 측정 매트릭스를 줄이지 못한다. 측정 sprint 가 종결되어 production opt-in 으로 굳어진 후 분리 필요성이 입증되면 분리한다 (YAGNI).

#### E.8.4 왜 Sprint 2b 이전에 backend 를 삭제하지 않는가

회귀 안전망. `--opencl-rpcmem` 측정이 `--backend qnn_oppkg` 와 동등 또는 우월함이 디바이스에서 검증된 후에야 backend 모듈을 삭제한다. 본 단계는 두 path 의 공존 + 단일 sprint verification gate 통과를 목적으로 한다.

