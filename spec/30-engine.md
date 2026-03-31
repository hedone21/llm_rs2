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
| OpenCLBackend | `opencl` | 모든 아키텍처 | GPU kernel, plan-based decode |

> **참고 (non-normative)**: `--backend hybrid` 시 CPU(primary) + GPU(secondary)를 `Arc<dyn Backend>`로 동시 보유한다. SwitchHw 명령으로 런타임 전환이 가능하다.

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
| MergeHandler | 스텁 | 미구현 |
| SparseHandler | 스텁 | 미구현 |

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
| `--backend` | String | "cpu" | "cpu", "opencl", "hybrid" |
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
