# Engine Data Types -- Architecture

> spec/33-engine-data.md의 구현 상세.

## 코드 매핑

### 3.1 KVCacheOps Trait

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-DAT-010 | `engine/src/core/kv_cache.rs:17` | `pub trait KVCacheOps: Send` | Generic monomorphization `<C: KVCacheOps>`, dyn Trait 아님 |
| ENG-DAT-010 | `engine/src/core/kv_cache.rs` | 14+ 메서드: `current_pos`, `update`, `get_view`, `ensure_capacity`, `needs_attn_scores`, `set_attn_scores` 등 | |

### 3.2 PrefetchableCache

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-DAT-011 | `engine/src/core/kv_cache.rs:92` | `pub trait PrefetchableCache: KVCacheOps` | OffloadKVCache 전용 extension |

### 3.3 KVCache 구현체

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-DAT-012 | `engine/src/core/kv_cache.rs:121` | `pub struct KVCache` — F16/F32/Q4_0 | dynamic grow/shrink, HeadMajor/SeqMajor |
| ENG-DAT-012 | `engine/src/core/kv_cache.rs:116` | `pub enum KVLayout { SeqMajor, HeadMajor }` | offset/stride 계산 분기 |
| ENG-DAT-012 | `engine/src/core/kv_cache.rs` | `new_dynamic()`, `grow()`, `shrink_to_fit()`, `prune_prefix()`, `shift_positions()`, `compact_keep_positions()`, `release_unused_pages()` | |

### 3.4 KiviCache 구현체

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-DAT-013 | `engine/src/core/kivi_cache.rs:164` | `pub struct KiviCache` — Q2/Q4/Q8 + FP32 residual | `kv_dtype()` 항상 F32, `get_buffers_mut()` 항상 None |
| ENG-DAT-013 | `engine/src/core/kivi_cache.rs` | QuantizedBlocks enum: Q2(BlockQ2_0), Q4(BlockKVQ4), Q8(BlockKVQ8) | QKKV=32 단위 블록 |
| ENG-DAT-013 | `engine/src/core/kivi_cache.rs` | GPU 모드: `new_gpu()`, gpu_res_k/v, gpu_attn_k/v, gpu_q2k/v | 6종 GPU 버퍼 |

### 3.5 OffloadKVCache

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| (ENG-DAT-012 변형) | `engine/src/core/offload/mod.rs:37` | `pub struct OffloadKVCache` | PrefetchableCache 구현, `--kv-offload` |

### 3.6 Buffer Trait 계층

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-DAT-020 | `engine/src/core/buffer.rs:30` | `pub trait Buffer: Send + Sync` | as_ptr, cl_mem, map_for_cpu, is_host_managed 등 |
| ENG-DAT-020 (SharedBuffer) | `engine/src/buffer/shared_buffer.rs:8` | `pub struct SharedBuffer` — `Vec<u8>` 기반 | is_host_managed=true, cl_mem=None |
| ENG-DAT-020 (MadviseableGPU) | `engine/src/buffer/madviseable_gpu_buffer.rs:15` | `pub struct MadviseableGPUBuffer` | is_host_managed=true, `CL_MEM_USE_HOST_PTR` |
| ENG-DAT-020 (UnifiedBuffer) | `engine/src/buffer/unified_buffer.rs:31` | `pub struct UnifiedBuffer` | is_host_managed=false, `CL_MEM_ALLOC_HOST_PTR` |
| ENG-DAT-020 (OpenCLBuffer) | `engine/src/backend/opencl/buffer.rs:12` | `pub struct OpenCLBuffer` | is_host_managed=false, GPU-only |

### 3.7 DType

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-DAT-021 | `engine/src/core/buffer.rs:7` | `pub enum DType { Q4_0, Q4_1, F16, BF16, F32, U8 }` | |

### 3.8 Backend Trait

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-DAT-030 | `engine/src/core/backend.rs:5` | `pub trait Backend: Send + Sync` — ~20 메서드 | matmul, rms_norm, silu_mul, rope, attention_gen 등 |
| ENG-DAT-030 (CpuBackendNeon) | `engine/src/backend/cpu/neon.rs` | `#[cfg(target_arch = "aarch64")]` NEON | |
| ENG-DAT-030 (CpuBackendAVX2) | `engine/src/backend/cpu/x86.rs` | `#[cfg(target_arch = "x86_64")]` AVX2+FMA | |
| ENG-DAT-030 (CpuBackendCommon) | `engine/src/backend/cpu/common.rs` | 스칼라 fallback | |
| ENG-DAT-030 (OpenCLBackend) | `engine/src/backend/opencl/mod.rs` | `#[cfg(feature = "opencl")]` GPU kernel | |

### 3.9 Tensor

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-DAT-031 | `engine/src/core/tensor.rs:8` | `pub struct Tensor { shape, buffer: Arc<dyn Buffer>, backend: Arc<dyn Backend> }` | Clone = shallow copy (Arc 공유) |
| ENG-DAT-031 | `engine/src/core/shape.rs` | `Shape` struct — 차원 정보 | |

### 3.10 ExecutionPlan / EvictPlan / KVSnapshot

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-DAT-040 | `engine/src/resilience/executor.rs:13` | `pub struct ExecutionPlan` — 9 필드 | poll() 반환, 즉시 소비, 1회성 |
| ENG-DAT-040 | `engine/src/resilience/executor.rs:44` | `pub struct EvictPlan { target_ratio, level, method }` | |
| ENG-DAT-040 | `engine/src/resilience/executor.rs:36` | `pub enum EvictMethod { H2o, Sliding, Streaming }` | Streaming: Rejected 반환 |
| ENG-DAT-040 | `engine/src/resilience/executor.rs:55` | `pub struct KVSnapshot` — 7 필드 | total_bytes, total_tokens, capacity 등 |

### 3.11 EvictionPolicy Trait

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-DAT-050 | `engine/src/core/eviction/mod.rs:12` | `pub trait EvictionPolicy: Send + Sync` — 5 메서드 | should_evict, evict, evict_with_scores, evict_with_head_scores, name |
| ENG-DAT-050 (H2OPolicy) | `engine/src/core/eviction/h2o.rs` | "h2o" — score 기반 3-partition | |
| ENG-DAT-050 (H2OPlusPolicy) | `engine/src/core/eviction/h2o_plus.rs` | "h2o_plus" — per-KV-head (HeadMajor 전용) | |
| ENG-DAT-050 (SlidingWindow) | `engine/src/core/eviction/sliding_window.rs` | "sliding_window" — FIFO prune_prefix | |
| ENG-DAT-050 (StreamingLLM) | `engine/src/core/eviction/streaming_llm.rs` | "streaming" — sink + sliding 결합 | |
| ENG-DAT-050 (NoEviction) | `engine/src/core/eviction/no_eviction.rs` | "none" — should_evict 항상 false | |

### 3.12 QcfMetric / QcfConfig

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-DAT-060 | `engine/src/core/qcf/mod.rs:28` | `pub struct QcfMetric { action, raw_value, normalized_value, per_head, tokens_affected }` | normalized_value unbounded (1 이상 가능) |
| ENG-DAT-061 | `engine/src/core/qcf/mod.rs:66` | `pub struct QcfConfig { enabled, mode, aggregation, d_max, epsilon }` | |
| ENG-DAT-061 | `engine/src/core/qcf/mod.rs:45` | `pub enum QcfMode { Attn, Caote, Both }` | |
| ENG-DAT-061 | `engine/src/core/qcf/mod.rs:93` | `pub enum AggregationMode { Mean, Defensive { temperature } }` | |

### 3.13 SkipConfig

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-DAT-062 | `engine/src/core/skip_config.rs` | `pub struct SkipConfig { attn_skip: HashSet<usize>, mlp_skip: HashSet<usize> }` | uniform_init, validate, skip_attn, skip_mlp |

### 3.14 ImportanceTable / ImportanceEntry

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-DAT-063 | `engine/src/core/qcf/layer_importance.rs:23` | `pub struct ImportanceEntry { layer_id, sublayer, importance, opr }` | |
| ENG-DAT-063 | `engine/src/core/qcf/layer_importance.rs:37` | `pub struct ImportanceTable { entries, total_importance }` | compute_qcf, estimate_qcf_for_count |

### 3.15 PressureLevel / ActionResult / HandlerContext

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| ENG-DAT-080 | `engine/src/core/pressure/mod.rs` | `PressureLevel = llm_shared::Level` — Normal < Warning < Critical < Emergency | Ord derive |
| ENG-DAT-080 | `engine/src/core/pressure/mod.rs:67` | `pub enum ActionResult { NoOp, Evicted, Quantized, Merged, Swapped, Sparsified }` | |
| ENG-DAT-080 | `engine/src/core/pressure/mod.rs:36` | `pub struct HandlerContext` — caches, importance, head_importance, pressure_level, mem_available, target_ratio, qcf_sink, layer_ratios | |
| ENG-DAT-080 | `engine/src/core/pressure/mod.rs:135` | `pub struct PressureStageConfig { min_level, handler: Box<dyn CachePressureHandler> }` | |

## Config

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| `kivi.bits` | u8 | 4 | ENG-DAT-013 |
| `kivi.residual_size` | usize | 32 | ENG-DAT-013 |
| `kivi.group_size` | usize | 32 (QKKV) | ENG-DAT-013 |
| `kivi.awqe_enabled` | bool | false | ENG-DAT-013 |
| `qcf.enabled` | bool | true | ENG-DAT-061 |
| `qcf.mode` | QcfMode | Attn | ENG-DAT-061 |
| `qcf.aggregation` | AggregationMode | Mean | ENG-DAT-061 |
| `qcf.d_max` | f32 | 5.0 | ENG-DAT-061 |
| `qcf.epsilon` | f32 | 1e-8 | ENG-DAT-061 |

## CLI

| 플래그 | 설명 | spec/ 근거 |
|--------|------|-----------|
| `--model-path` | 모델 경로 | ENG-DAT-070 |
| `--prompt` / `--prompt-file` | 입력 프롬프트 | ENG-DAT-070 |
| `--num-tokens` | 생성 토큰 수 | ENG-DAT-070 |
| `--backend` | cpu / opencl / hybrid | ENG-DAT-070 |
| `--max-seq-len` | 최대 시퀀스 길이 | ENG-DAT-070 |
| `--threads` | 스레드 수 (0=auto) | ENG-DAT-070 |
| `--use-rayon` | Rayon vs SpinPool | ENG-DAT-070 |
| `--temperature` | 샘플링 온도 | ENG-DAT-070 |
| `--top-p` | Top-p 샘플링 | ENG-DAT-070 |
| `--top-k` | Top-k 샘플링 | ENG-DAT-070 |
| `--greedy` | 그리디 샘플링 | ENG-DAT-070 |
| `--weight-dtype` | 모델 가중치 타입 (f16/q4) | ENG-DAT-070 |
| `--kv-type` | KV 캐시 타입 (f32/f16/q4) | ENG-DAT-070 |
| `--kv-layout` | KV 레이아웃 (head/seq) | ENG-DAT-070 |
| `--initial-kv-capacity` | 초기 KV 용량 | ENG-DAT-070 |
| `--kv-budget` / `--kv-budget-ratio` | KV 예산 | ENG-DAT-070 |
| `--kivi` | KIVI 양자화 활성 | ENG-DAT-070 |
| `--kivi-residual-size` | KIVI residual 크기 | ENG-DAT-070 |
| `--eviction-policy` | eviction 정책 | ENG-DAT-070 |
| `--eviction-window` | sliding/streaming 윈도우 | ENG-DAT-070 |
| `--sink-size` | StreamingLLM sink 토큰 수 | ENG-DAT-070 |
| `--protected-prefix` | eviction 보호 prefix | ENG-DAT-070 |
| `--memory-threshold-mb` | eviction 트리거 메모리 | ENG-DAT-070 |
| `--eviction-target-ratio` | eviction 보존 비율 | ENG-DAT-070 |
| `--h2o-keep-ratio` | H2O heavy hitter 비율 | ENG-DAT-070 |
| `--h2o-tracked-layers` | H2O score 추적 레이어 | ENG-DAT-070 |
| `--h2o-decay` | H2O score 감쇠 | ENG-DAT-070 |
| `--d2o-keep-ratio` | D2O heavy hitter 비율 | ENG-DAT-070 |
| `--d2o-ema-alpha` / `--d2o-ema-beta` | D2O EMA 가중치 | ENG-DAT-070 |
| `--d2o-layer-alloc` | D2O per-layer 동적 할당 | ENG-DAT-070 |
| `--skip-layers` / `--skip-ratio` | layer skip | ENG-DAT-070 |
| `--qcf-mode` | QCF proxy 모드 | ENG-DAT-070 |
| `--enable-resilience` | Resilience 활성 | ENG-DAT-070 |
| `--resilience-transport` | Resilience transport | ENG-DAT-070 |
| `--kv-offload` | KV offload 모드 | ENG-DAT-070 |
| `--gpu-attn` | GPU attention 커널 | ENG-DAT-070 |
| `--prefill-chunk-size` | chunked prefill | ENG-DAT-070 |
| `--profile` / `--profile-dir` | 프로파일링 | ENG-DAT-070 |
| `--eval-ll` / `--ppl` | 평가 모드 | ENG-DAT-070 |
| `--experiment-schedule` | 실험 스케줄 | ENG-DAT-070 |