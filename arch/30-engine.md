# Engine Overview -- Architecture

> spec/30-engine.md의 구현 상세.

## 코드 매핑

### 3.1 Engine Process Responsibility

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-010 | `engine/src/bin/generate.rs` | `main()` 함수 — CLI 파싱, 모델 로딩, 추론 루프 통합 | Manager 없이 단독 동작: `command_executor = None` |

### 3.2 Subsystem Decomposition

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-011 (Model) | `engine/src/models/` (TransformerModel, config, mappers), `engine/src/layers/` (attention, llama_layer, transformer_layer, workspace) | `TransformerModel::forward()` — logits 텐서 반환 | Backend, KV Cache, Memory에 의존 |
| ENG-012 (Core) | `engine/src/core/tensor.rs`, `core/buffer.rs`, `core/shape.rs`, `core/memory.rs`, `core/math_utils.rs`, `core/sampling.rs`, `core/quant.rs`, `core/thread_pool.rs` | `Tensor`, `Buffer` trait, `DType` enum, `Memory` trait, `SamplingConfig` | 최하위 계층, 무의존 |
| ENG-013 (Backend) | `engine/src/core/backend.rs` (Backend trait), `engine/src/backend/cpu/` (Neon, AVX2, Common), `engine/src/backend/opencl/` (OpenCLBackend) | Backend trait ~20 메서드: `matmul`, `rms_norm`, `silu_mul`, `rope`, `softmax`, `attention_qkv` 등 | |
| ENG-014 (KV Cache) | `engine/src/core/kv_cache.rs` (KVCacheOps trait, KVCache struct), `engine/src/core/kivi_cache.rs` (KiviCache), `engine/src/core/kv_migrate.rs` | KVCacheOps trait 14 메서드 | |
| ENG-015 (Cache Management) | `engine/src/core/cache_manager.rs` (CacheManager), `engine/src/core/pressure/` (Pipeline, 6종 Handler), `engine/src/core/eviction/` (EvictionPolicy trait, 5종), `engine/src/core/attention_scores.rs` | CacheManager → CachePressurePipeline 래핑, `force_evict_by_policy()` | SystemMonitor 의존 |
| ENG-016 (Resilience) | `engine/src/resilience/` 전체 (executor.rs, transport.rs, manager.rs, state.rs, signal.rs, strategy/, dbus_transport.rs) | Directive 경로 + Strategy 경로 (D-Bus 레거시) | |
| ENG-017 (QCF) | `engine/src/core/qcf/` (mod.rs, eviction_qcf.rs, quant_qcf.rs, skip_qcf.rs, estimator.rs, layer_importance.rs) | QcfMetric, QcfMode, DegradationEstimator | |
| ENG-018 (Eval) | `engine/src/eval/` (eval_loop.rs, hook.rs, eviction_hook.rs, kivi_hook.rs, output.rs, qcf_helpers.rs) | StepHook trait — KVCacheOps 구현체별 평가 로직 추상화 | |

### Backend 구현체

| 구현체 | feature gate | 타겟 아키텍처 | 코드 위치 |
|--------|-------------|--------------|----------|
| CpuBackendNeon | 기본 | aarch64 | `engine/src/backend/cpu/neon.rs` |
| CpuBackendAVX2 | 기본 | x86_64 | `engine/src/backend/cpu/x86.rs` |
| CpuBackendCommon | 기본 | 기타 | `engine/src/backend/cpu/common.rs` |
| OpenCLBackend | `opencl` | 모든 아키텍처 | `engine/src/backend/opencl/mod.rs` |

### KVCacheOps 구현체

| 구현체 | DType | 코드 위치 | 비고 |
|--------|-------|----------|------|
| KVCache | F32, F16, Q4_0 | `engine/src/core/kv_cache.rs` | 모든 eviction 정책, offload 지원 |
| KiviCache | F32 입력, 내부 Q2 | `engine/src/core/kivi_cache.rs` | AWQE (`needs_attn_scores=true`), eviction 미지원 |
| OffloadKVCache | F16, F32 | `engine/src/core/kv_cache.rs` 또는 별도 모듈 | seq-major only, `--kv-offload` |

### CachePressureHandler 구현체 6종

| Handler | 구현 상태 | 코드 위치 |
|---------|----------|----------|
| EvictionHandler | 구현 완료 | `engine/src/core/pressure/eviction_handler.rs` |
| D2OHandler | 구현 완료 | `engine/src/core/pressure/d2o_handler.rs` |
| SwapHandler | 구현 완료 | `engine/src/core/pressure/swap_handler.rs` |
| QuantizeHandler | 간접 활성 | `engine/src/core/pressure/quantize_handler.rs` |
| MergeHandler | 스텁 | `engine/src/core/pressure/merge_handler.rs` |
| SparseHandler | 스텁 | `engine/src/core/pressure/sparse_handler.rs` |

### EvictionPolicy 구현체 5종

| 구현체 | CLI 값 | 코드 위치 |
|--------|--------|----------|
| NoEvictionPolicy | `none` | `engine/src/core/eviction/mod.rs` |
| SlidingWindowPolicy | `sliding` | `engine/src/core/eviction/mod.rs` 또는 별도 파일 |
| StreamingLLMPolicy | `streaming` | `engine/src/core/eviction/` |
| H2OPolicy | `h2o` | `engine/src/core/eviction/h2o.rs` |
| H2OPlusPolicy | `h2o_plus` | `engine/src/core/eviction/h2o.rs` 또는 별도 파일 |

### 8종 액션의 Engine 실행 경로

| 액션 | 실행 경로 | 구현 상태 |
|------|----------|----------|
| SwitchHw | `plan.switch_device` — Backend 런타임 교체 | 구현 완료 |
| Throttle | `plan.throttle_delay_ms` — 토큰 간 sleep | 구현 완료 |
| KvEvictH2o | `EvictMethod::H2o` — `CacheManager::force_evict_by_policy()` | 구현 완료 |
| KvEvictSliding | `EvictMethod::Sliding` — `CacheManager::force_evict_by_policy()` | 구현 완료 |
| KvStreaming | `CommandResult::Rejected` 반환 | 프로토콜 정의 완료, Engine 실행 미구현 |
| KvQuantDynamic | `plan.kv_quant_bits` — KIVI 경로에서 소비 | 구현 완료 |
| KvMergeD2o | — | 스펙 전용, 코드 미등록, 미구현 |
| LayerSkip | `plan.layer_skip` — SkipConfig 갱신 | 구현 완료 |

### 3.3 Transport Trait and Implementations

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-020 | `engine/src/resilience/transport.rs` | `Transport` trait: `connect()`, `recv()`, `send()`, `name()` | `Send + 'static` 바운드 |
| ENG-021 | `engine/src/resilience/transport.rs` 내 impl | `[4B BE u32 length][UTF-8 JSON]` wire format | serde_json 직렬화 |
| ENG-023 | `engine/src/resilience/transport.rs` | `MessageLoop::spawn(transport)` → `(cmd_rx, resp_tx, JoinHandle)` | 스레드 이름: `"{transport_name}-loop"` |
| ENG-024 | `engine/src/resilience/dbus_transport.rs` | `signal_to_manager_message()` — SystemSignal → EngineCommand 변환 | feature gate: `resilience` |

### Transport 구현체 4종

| 구현체 | feature gate | CLI 선택 | 코드 위치 |
|--------|-------------|----------|----------|
| UnixSocketTransport | `#[cfg(unix)]` | `--resilience-transport unix:/path` | `engine/src/resilience/transport.rs` |
| TcpTransport | 기본 | `--resilience-transport tcp:addr:port` | `engine/src/resilience/transport.rs` |
| DbusTransport | `resilience` | `--resilience-transport dbus` | `engine/src/resilience/dbus_transport.rs` |
| MockTransport | (테스트 전용) | — | `engine/src/resilience/transport.rs` |

### TransportError 변종

`ConnectionFailed(String)`, `Disconnected`, `ParseError(String)`, `Io(io::Error)`.

### 3.4 Threading Model

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-030 | `engine/src/bin/generate.rs` | Main(추론) + MessageLoop + Rayon pool 3종 스레드 | |
| ENG-031 | `engine/src/bin/generate.rs` `main()` | 10단계 Main 스레드 흐름 (CLI→Rayon→Backend→Model→KV→CacheMgr→Executor→Prefill→Decode) | |
| ENG-032 | `engine/src/bin/generate.rs` decode loop 내 | `KVSnapshot` 구성 → `executor.poll(&kv_snap)` → plan 소비 | 토큰당 1회 |
| ENG-033 | `engine/src/resilience/executor.rs` | `poll()` 내 `elapsed >= interval` 체크 → `send_heartbeat()` | `heartbeat_interval = 1000ms` 하드코딩 |

### Resilience Directive 경로

```
Manager --> Transport --> MessageLoop(thread) --> mpsc --> CommandExecutor --> ExecutionPlan --> Inference Loop
```

| 컴포넌트 | 코드 위치 | 책임 |
|----------|----------|------|
| Transport (trait) | `engine/src/resilience/transport.rs` | Manager-Engine 양방향 바이트 스트림 |
| MessageLoop | `engine/src/resilience/transport.rs` | Transport 소유 전용 스레드, blocking recv + try_recv drain |
| CommandExecutor | `engine/src/resilience/executor.rs` | ManagerMessage 수신 → EngineCommand 해석 → ExecutionPlan 구축 |

### Resilience Strategy 경로 (D-Bus 레거시)

```
D-Bus system bus --> DbusTransport --> signal_to_manager_message() --> ManagerMessage --> CommandExecutor
```

| 컴포넌트 | 코드 위치 | 책임 |
|----------|----------|------|
| DbusTransport | `engine/src/resilience/dbus_transport.rs` | D-Bus signal 파싱, ManagerMessage 변환 |
| ResilienceManager | `engine/src/resilience/manager.rs` | 4종 SystemSignal 수신, OperatingMode 계산, Strategy 위임 |
| Strategy 4종 | `engine/src/resilience/strategy/` (memory.rs, compute.rs, thermal.rs, energy.rs) | domain별 반응 로직 |

### D-Bus signal → EngineCommand 변환 테이블 (ENG-024)

| D-Bus signal | Level | 변환 결과 |
|-------------|-------|----------|
| MemoryPressure | Normal | RestoreDefaults |
| MemoryPressure | Warning | KvEvictSliding { keep_ratio: 0.85 } |
| MemoryPressure | Critical | KvEvictH2o { keep_ratio: 0.50 } |
| MemoryPressure | Emergency | KvEvictH2o { keep_ratio: 0.25 } |
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

### 3.6 Subsystem Dependency Graph (ENG-050)

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

### Invariants 코드 매핑

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| INV-060 | `engine/src/bin/generate.rs` decode loop | `executor.poll(&kv_snap)` 호출이 토큰당 1회 | loop iteration 당 1회 호출 |
| INV-061 | `engine/src/bin/generate.rs` decode loop | `poll()` 반환값을 즉시 소비, 변수 재할당으로 폐기 | `let plan = executor.poll(...)` |
| INV-062 | `engine/src/resilience/executor.rs` `poll()` | step 5: `plan.suspended == true` → evict/switch_device/prepare_device = None | |
| INV-063 | `engine/src/resilience/transport.rs` `MessageLoop::spawn()` | Transport를 `move` 클로저로 이동 — 단일 소유자 | `std::thread::spawn(move \|\| { ... })` |
| INV-064 | `engine/src/resilience/executor.rs` `poll()` | `elapsed >= heartbeat_interval` → `send_heartbeat()` | |
| INV-065 | `engine/src/core/backend.rs` Backend trait | trait 바운드: `Send + Sync` | `dyn Backend` 공유에 필수 |

### Constraints 코드 매핑

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-070 | 프로젝트 전체 | async 런타임 미사용, `std::thread` + `mpsc` 채널 | Cargo.toml에 tokio 등 미포함 |
| ENG-071 | `engine/src/resilience/transport.rs` | `serde_json::to_string()` / `from_str()` | |
| ENG-072 | `engine/src/resilience/executor.rs` | 단일 `cmd_rx` 채널 — 1개 Transport에서만 수신 | |
| ENG-073 | `engine/src/resilience/transport.rs` MessageLoop | `transport.recv()` blocking 호출은 전용 스레드에서만 | |

## Feature Gates

| feature | 영향 모듈 | 설명 | spec/ 근거 |
|---------|----------|------|-----------|
| `opencl` | `engine/src/backend/opencl/` | OpenCL GPU 백엔드 활성화 (기본 활성) | ENG-013 |
| `resilience` | `engine/src/resilience/dbus_transport.rs` | D-Bus Transport, Strategy 경로 | ENG-016, ENG-022, ENG-061 |
| `#[cfg(unix)]` | `engine/src/resilience/transport.rs` | UnixSocketTransport (Unix 전용) | ENG-022 |

## CLI

### Resilience 관련 (ENG-040)

| 플래그 | 타입 | 기본값 | spec/ 근거 |
|--------|------|--------|-----------|
| `--enable-resilience` | bool | false | ENG-040 |
| `--resilience-transport` | String | "dbus" | ENG-040 |
| `--experiment-schedule` | Option\<String\> | None | ENG-040, ENG-045 |

### Backend 관련 (ENG-041)

| 플래그 | 타입 | 기본값 | spec/ 근거 |
|--------|------|--------|-----------|
| `--backend` | String | "cpu" | ENG-041 |
| `--switch-threshold` | usize | 0 | ENG-041 |
| `--gpu-attn` | bool | false | ENG-041 |
| `--zero-copy` | bool | false | ENG-041 |

### KV Cache 관련 (ENG-042)

| 플래그 | 타입 | 기본값 | spec/ 근거 |
|--------|------|--------|-----------|
| `--kv-type` | String | "f16" | ENG-042 |
| `--kv-layout` | String | "head" | ENG-042 |
| `--eviction-policy` | String | "none" | ENG-042 |
| `--kv-budget` | usize | 0 | ENG-042 |
| `--kv-budget-ratio` | f32 | 0.0 | ENG-042 |
| `--kivi` | bool | false | ENG-042 |
| `--kv-offload` | String | "none" | ENG-042 |
| `--protected-prefix` | Option\<usize\> | None | ENG-042 |

### QCF/Eval 관련 (ENG-043)

| 플래그 | 타입 | 기본값 | spec/ 근거 |
|--------|------|--------|-----------|
| `--qcf-mode` | String | "attn" | ENG-043 |
| `--eval-ll` | bool | false | ENG-043 |
| `--eval-batch` | Option\<String\> | None | ENG-043 |
| `--profile` | bool | false | ENG-043 |

### Inference 관련 (ENG-044)

| 플래그 | 타입 | 기본값 | spec/ 근거 |
|--------|------|--------|-----------|
| `--model-path` | String | "models/llama3.2-1b" | ENG-044 |
| `--num-tokens` | usize | 20 | ENG-044 |
| `--max-seq-len` | usize | 2048 | ENG-044 |
| `--threads` | usize | 0 (auto) | ENG-044 |
| `--weight-dtype` | String | "f16" | ENG-044 |
| `--skip-ratio` | Option\<f32\> | None | ENG-044 |
| `--greedy` | bool | false | ENG-044 |

### Experiment 모드 (ENG-045)

| 플래그 | 타입 | 기본값 | spec/ 근거 |
|--------|------|--------|-----------|
| `--experiment-schedule` | Option\<String\> | None | ENG-045 |
| `--experiment-output` | Option\<String\> | None | ENG-045 |
| `--experiment-eviction-ratio` | Option\<f32\> | None | ENG-045 |

## Config

(Engine은 설정 파일 기반 config를 사용하지 않는다. 모든 설정은 CLI 플래그로 전달된다.)
