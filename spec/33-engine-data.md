# Engine Data Types

> **TL;DR**: Engine 내부 데이터 타입, KV 캐시 구조, Backend/Buffer trait, CLI 설정을 정의한다. KVCacheOps trait (14+ 메서드)과 3종 구현체(KVCache, KiviCache, OffloadKVCache), Buffer trait 계층과 4종 구현체(SharedBuffer, MadviseableGPUBuffer, UnifiedBuffer, OpenCLBuffer), Backend trait (~20 메서드), Tensor 구조, ExecutionPlan/EvictPlan/KVSnapshot, EvictionPolicy trait과 5종 구현체, QcfMetric/QcfConfig, SkipConfig, ImportanceTable, CachePressurePipeline 데이터 타입, Engine CLI 60+ 플래그를 기술한다. 필드명/타입/범위/기본값을 정의하되 struct 레이아웃은 자유이다.

## 1. Purpose and Scope

이 문서는 Engine 내부의 **데이터 타입, 인터페이스, 설정**을 정의한다. 의미와 관계, 필드명/타입/범위/기본값을 기술한다.

**이 파일이 명세하는 것:**

- KVCacheOps trait과 구현체 (KVCache, KiviCache)
- PrefetchableCache extension trait
- Buffer trait 계층과 4종 구현체
- DType enum
- Backend trait과 구현체
- Tensor 구조
- ExecutionPlan, EvictPlan, KVSnapshot (데이터 관점)
- EvictionPolicy trait과 5종 구현체
- QcfMetric, QcfConfig, QcfMode, AggregationMode
- SkipConfig, ImportanceTable
- CachePressurePipeline 관련 타입 (PressureLevel, ActionResult, HandlerContext)
- Engine CLI 플래그 전체

**이 파일이 명세하지 않는 것:**

- Engine 아키텍처 개요 → `30-engine.md`
- 상태 머신 전이 테이블 → `31-engine-state.md`
- 알고리즘 상세 (eviction 수식, QCF 계산 등) → `32-engine-algorithms.md`
- Manager 알고리즘/데이터 → `20-manager.md` ~ `23-manager-data.md`
- 프로토콜 메시지/시퀀스 → `10-protocol.md` ~ `12-protocol-sequences.md`

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **KVLayout** | KV 캐시의 메모리 레이아웃. SeqMajor 또는 HeadMajor. |
| **Dynamic Cache** | `Memory` trait을 보유한 KVCache. 런타임에 grow/shrink가 가능하다. |
| **Static Cache** | `memory = None`인 KVCache. 고정 용량이며 shrink_to_fit 불가. |
| **Monomorphization** | `<C: KVCacheOps>` 제네릭으로 컴파일 타임에 구체 타입을 결정. dyn Trait 아님, zero overhead. |
| **UMA** | Unified Memory Architecture. CPU와 GPU가 물리 메모리를 공유하는 아키텍처 (Qualcomm Snapdragon 등). |

## 3. Specification

### 3.1 KVCacheOps Trait [ENG-DAT-010]

**[ENG-DAT-010]** KVCacheOps는 KV 캐시 연산의 추상 인터페이스이다. Generic monomorphization (`<C: KVCacheOps>`)으로 사용되며 dyn Trait이 아니다. *(MUST)*

| 메서드 | 시그니처 (의사) | 설명 |
|--------|----------------|------|
| `current_pos` | `() -> usize` | 현재 유효 토큰 수 |
| `set_current_pos` | `(usize)` | 위치 카운터 직접 설정 (probe 되돌리기용) |
| `capacity` | `() -> usize` | 물리 버퍼 토큰 용량 |
| `kv_heads` | `() -> usize` | KV head 수 |
| `head_dim` | `() -> usize` | head당 차원 |
| `layout` | `() -> KVLayout` | 메모리 레이아웃 (SeqMajor / HeadMajor) |
| `kv_dtype` | `() -> DType` | 호출자가 `update()`에 전달할 dtype. KiviCache는 F32 반환 (내부 양자화) |
| `memory_usage_bytes` | `() -> usize` | 현재 저장 KV 데이터의 바이트 수 |
| `update` | `(&Tensor, &Tensor) -> Result<()>` | 새 K/V 추가. 입력: `[batch, seq_len, kv_heads, head_dim]` |
| `get_view` | `() -> (Tensor, Tensor)` | attention 계산용 K/V 텐서. `[0..current_pos]` 범위 |
| `get_buffers_mut` | `() -> Option<(&mut Tensor, &mut Tensor)>` | zero-copy scatter write용 직접 접근. KiviCache는 None 반환 |
| `advance_pos` | `(usize)` | 데이터 복사 없이 position 전진 (`get_buffers_mut`과 함께 사용) |
| `ensure_capacity` | `(usize) -> Result<bool>` | 최소 토큰 수 보장, 필요 시 grow. bool = 버퍼 변경 여부 |
| `needs_attn_scores` | `() -> bool` | decode 시 post-softmax score 계산 필요 여부 (KiviCache AWQE용) |
| `set_attn_scores` | `(&[f32], n_heads_q, stride, valid_len)` | 최근 decode step의 attention score 저장 |

---

### 3.2 PrefetchableCache Extension Trait [ENG-DAT-011]

**[ENG-DAT-011]** PrefetchableCache는 OffloadKVCache 전용 확장 trait이다. 레이어 간 I/O 파이프라인을 제공한다. *(MAY)*

| 메서드 | 설명 |
|--------|------|
| `preload` | 외부 저장소 → 메모리 버퍼 로드 |
| `release_buffers` | 메모리 버퍼 해제 (동시 2 레이어만 활성) |
| `reset_preload` | 토큰 경계에서 preloaded 플래그 리셋 |
| `retain_preload` | cross-token 버퍼 유지 (다음 토큰의 preload skip) |

---

### 3.3 KVCache 구현체 [ENG-DAT-012]

**[ENG-DAT-012]** KVCache는 일반 KV 캐시 구현체이다. F16/F32/Q4_0 dtype을 지원하며, dynamic grow/shrink가 가능하다. *(MUST)*

**필드**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `k_buffer` | Tensor | K 저장소 |
| `v_buffer` | Tensor | V 저장소 |
| `current_pos` | usize | 유효 토큰 수 |
| `high_water_pos` | usize | 최대 current_pos 기록 (madvise 범위 제한, ENG-ALG-071 참조) |
| `max_seq_len` | usize | 절대 상한 |
| `capacity` | usize | 현재 물리 용량 (dynamic grow 시 변동) |
| `kv_heads` | usize | KV head 수 |
| `head_dim` | usize | head당 차원 |
| `layout` | KVLayout | SeqMajor 또는 HeadMajor |
| `memory` | Option\<Arc\<dyn Memory\>\> | dynamic grow/shrink용 할당자. None = 고정 용량 |

**KVLayout**:

| Layout | 메모리 순서 | offset(pos, head) | 용도 |
|--------|-----------|-------------------|------|
| SeqMajor | `[batch, seq_pos, kv_heads, head_dim]` | `pos * kv_heads * head_dim + head * head_dim` | Sliding window (연속 prune) |
| HeadMajor | `[batch, kv_heads, seq_pos, head_dim]` | `head * capacity * head_dim + pos * head_dim` | H2O/H2O+ (per-head eviction) |

**Dynamic grow**: `new_dynamic()` 생성자. 초기 용량에서 시작, 2배 확장 (`max_seq_len` 상한). Layout별 데이터 복사 필요 (HeadMajor는 per-head copy).

**주요 메서드** (KVCacheOps 외):

| 메서드 | 설명 |
|--------|------|
| `offset(pos, head)` | layout 기반 요소 오프셋 |
| `pos_stride()` | 연속 position 간 요소 거리 |
| `head_stride()` | 연속 head 간 요소 거리 |
| `q4_block_offset(pos, head, bpp)` | Q4_0 블록 인덱스 |
| `grow(min_capacity)` | 2배 확장, alloc_kv + 데이터 복사 |
| `shrink_to_fit()` | `next_power_of_2(current_pos)`로 축소 (ENG-ALG-072 참조) |
| `prune_prefix(count)` | 앞 count개 제거, shift, release_unused_pages |
| `shift_positions(src, dst, count)` | layout-aware 위치 이동 |
| `shift_positions_for_head(head, ...)` | HeadMajor 전용 per-head 이동 |
| `compact_keep_positions(keep, start)` | 연속 구간 batch 최적화 compaction (ENG-ALG-010 참조) |
| `compact_keep_positions_for_head(head, ...)` | per-head compaction |
| `release_unused_pages()` | shrink_to_fit 또는 madvise (ENG-ALG-071, ENG-ALG-072 참조) |
| `memory_usage_bytes()` | current_pos 기반 실제 사용량 |

---

### 3.4 KiviCache 구현체 [ENG-DAT-013]

**[ENG-DAT-013]** KiviCache는 KIVI 비대칭 양자화 KV 캐시 구현체이다. Q2/Q4/Q8 + FP32 residual 이중 구조. *(MUST)*

**필드**:

| 필드 | 타입 | 범위/기본값 | 설명 |
|------|------|------------|------|
| `bits` | u8 | {2, 4, 8} | 현재 양자화 bit-width |
| `qk` | QuantizedBlocks | - | Key blocks (per-channel) |
| `qv` | QuantizedBlocks | - | Value blocks (per-token) |
| `q2_tokens` | usize | res_cap의 배수 | 양자화 저장소의 토큰 수 |
| `res_k` | Vec\<f32\> | - | FP32 residual K `[kv_heads * res_cap * head_dim]` |
| `res_v` | Vec\<f32\> | - | FP32 residual V `[kv_heads * res_cap * head_dim]` |
| `res_pos` | usize | [0, res_cap) | residual 유효 토큰 수 |
| `res_cap` | usize | QKKV=32의 배수 | residual 용량 (CLI: `--kivi-residual-size`) |
| `attn_k_buf` | Vec\<f32\> | - | Pre-allocated attention K 출력 `[max_seq_len * kv_heads * head_dim]` |
| `attn_v_buf` | Vec\<f32\> | - | Pre-allocated attention V 출력 |
| `q2_deq_tokens` | usize | - | incremental dequant 추적 (ENG-ALG-022 참조) |
| `kv_heads` | usize | - | KV head 수 |
| `head_dim` | usize | - | head당 차원 |
| `max_seq_len` | usize | - | 최대 시퀀스 길이 |
| `group_size` | usize | 32 (= QKKV) | 양자화 그룹 크기 |
| `flush_proxies` | Vec\<QcfMetric\> | - | flush 시 수집된 QCF proxy |
| `awqe_enabled` | bool | false | AWQE proxy 활성화 여부 |
| `last_attn_scores` | Option\<AttnScoresSnapshot\> | None | 최근 attention score 스냅샷 |

**GPU mode 추가 필드** (CPU mode에서는 모두 None):

| 필드 | 타입 | 설명 |
|------|------|------|
| `gpu_backend` | Option\<Arc\<dyn Backend\>\> | GPU 백엔드 |
| `gpu_memory` | Option\<Arc\<dyn Memory\>\> | GPU 메모리 할당자 |
| `gpu_res_k`, `gpu_res_v` | Option\<Tensor\> | GPU residual 버퍼 |
| `gpu_attn_k`, `gpu_attn_v` | Option\<Tensor\> | GPU attention 출력 버퍼 |
| `gpu_q2k`, `gpu_q2v` | Option\<Tensor\> | GPU 양자화 저장소 (U8) |
| `gpu_q2k_blocks`, `gpu_q2v_blocks` | usize | GPU 양자화 블록 수 |

**AttnScoresSnapshot**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `scores` | Vec\<f32\> | `[n_heads_q * stride]`, post-softmax |
| `n_heads_q` | usize | query head 수 |
| `stride` | usize | = max_seq_len allocation |
| `valid_len` | usize | 스냅샷 시점의 유효 position 수 |

**QuantizedBlocks**: enum { Q2(Vec\<BlockQ2_0\>), Q4(Vec\<BlockKVQ4\>), Q8(Vec\<BlockKVQ8\>) }. 모두 QKKV=32 단위 블록.

---

### 3.5 Buffer Trait 계층 [ENG-DAT-020]

**[ENG-DAT-020]** Buffer trait은 모든 물리 메모리 표현의 추상 인터페이스이다. `Send + Sync` 요구. *(MUST)*

| 메서드 | 시그니처 (의사) | 기본값 | 설명 |
|--------|----------------|--------|------|
| `as_any` | `() -> &dyn Any` | - | 다운캐스트용 |
| `dtype` | `() -> DType` | - | 데이터 타입 |
| `size` | `() -> usize` | - | 바이트 단위 총 크기 |
| `as_ptr` | `() -> *const u8` | - | CPU 읽기 전용 포인터 |
| `as_mut_ptr` | `() -> *mut u8` | - | CPU 쓰기 포인터 |
| `cl_mem` | `() -> Option<&Mem>` | - | OpenCL 핸들 (없으면 None) |
| `sync_device` | `() -> Result<()>` | - | 디바이스 동기화 |
| `map_for_cpu` | `() -> Result<()>` | Ok(()) | CPU 접근 매핑 (GPU→CPU 전환) |
| `unmap_for_gpu` | `() -> Result<()>` | Ok(()) | GPU 접근 매핑 (CPU→GPU 전환) |
| `is_mapped` | `() -> bool` | true | 현재 CPU 매핑 상태 |
| `is_host_managed` | `() -> bool` | true | 앱이 메모리 소유 (madvise 유효 여부 판단, ENG-ALG-071 참조) |

**구현체 4종**:

| 구현체 | 설명 | is_host_managed | cl_mem |
|--------|------|----------------|--------|
| SharedBuffer | `Vec<u8>` 기반 CPU 전용 | true | None |
| MadviseableGPUBuffer | `Vec<u8>` + `CL_MEM_USE_HOST_PTR` (ENG-ALG-073 참조) | true | Some |
| UnifiedBuffer | `CL_MEM_ALLOC_HOST_PTR` (드라이버 관리) | false | Some |
| OpenCLBuffer | GPU-only (`CL_MEM_READ_WRITE`) | false | Some |

---

### 3.6 DType [ENG-DAT-021]

**[ENG-DAT-021]** DType enum은 텐서 요소의 데이터 타입을 정의한다. *(MUST)*

| Variant | 바이트 크기 | 설명 |
|---------|-----------|------|
| Q4_0 | 블록 단위 | Block quantized (4-bit, 1 바이트/요소) |
| Q4_1 | 블록 단위 | Block quantized (4-bit, variant 1) |
| F16 | 2 | IEEE 754 half-precision |
| BF16 | 2 | Brain floating-point |
| F32 | 4 | IEEE 754 single-precision |
| U8 | 1 | Unsigned 8-bit integer |

---

### 3.7 Backend Trait [ENG-DAT-030]

**[ENG-DAT-030]** Backend trait은 하드웨어 추상화 계층이다. 모든 수치 연산을 디스패치한다. *(MUST)*

| 범주 | 메서드 | 설명 |
|------|--------|------|
| **Identity** | `name()`, `device()` | 백엔드 이름/디바이스 식별 |
| **Basic Math** | `matmul`, `matmul_transposed`, `matmul_slice` | 행렬 곱 (3종) |
| **In-place** | `add_assign`, `scale`, `add_row_bias` | 원소별 연산 |
| **Activation/Norm** | `silu_mul`, `gelu_tanh_mul`, `rms_norm`, `rms_norm_oop`, `add_rms_norm_oop`, `softmax` | 활성화/정규화 |
| **Positional** | `rope_inplace` | RoPE 위치 인코딩 |
| **Attention** | `attention_gen` | 단일 쿼리 attention (GQA 인식), `scores_out` 옵션 |
| **Memory** | `copy_from`, `copy_into`, `read_buffer`, `write_buffer`, `copy_slice`, `buffer_shift`, `gather` | 데이터 이동/복사 |
| **Cast** | `cast`, `kv_scatter_f32_to_f16` | 타입 변환 (fused F32→F16 scatter 포함) |
| **Sync** | `synchronize`, `flush` | GPU 동기화/큐 플러시 |

**구현체 2종**:

| 구현체 | 조건 | 특징 |
|--------|------|------|
| CpuBackend | `aarch64` → CpuBackendNeon, `x86_64` → CpuBackendAVX2, 기타 → CpuBackendCommon | SIMD 가속, Rayon 병렬 |
| OpenCLBackend | `opencl` feature gate | GPU kernel, plan-based decode |

---

### 3.8 Tensor [ENG-DAT-031]

**[ENG-DAT-031]** Tensor는 shape + 물리 버퍼 + 연산 디스패치 대상의 3-tuple이다. *(MUST)*

**필드**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `shape` | Shape | 차원 정보 (e.g., `[1, 2048, 8, 64]`) |
| `buffer` | Arc\<dyn Buffer\> | 물리 메모리 (공유 가능) |
| `backend` | Arc\<dyn Backend\> | 연산 디스패치 대상 |

| 메서드 | 설명 |
|--------|------|
| `shape()`, `buffer()`, `backend()` | 필드 접근 |
| `dtype()`, `size()`, `numel()` | 메타데이터 |
| `as_ptr()`, `as_mut_ptr()` | 원시 포인터 |
| `as_slice::<T>()`, `as_mut_slice::<T>()` | 타입드 슬라이스 접근 |
| `reshape(new_shape)` | shape만 변경, 버퍼 불변 (numel 동일 필수) |
| `to_device(backend)` | 다른 백엔드로 복사 (같은 이름이면 no-op) |

**Clone**: `Arc<dyn Buffer>` 공유 (shallow copy). 버퍼 데이터를 복제하지 않는다.

---

### 3.9 ExecutionPlan (데이터 관점) [ENG-DAT-040]

**[ENG-DAT-040]** ExecutionPlan은 단일 `poll()` 호출의 결과물이며, Inference Loop가 즉시 소비한다. 수명 규칙은 `31-engine-state.md` ENG-ST-042 참조. *(MUST)*

**필드**:

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `evict` | Option\<EvictPlan\> | None | eviction 계획 |
| `switch_device` | Option\<String\> | None | 전환 대상 디바이스 (e.g., "opencl") |
| `prepare_device` | Option\<String\> | None | pre-warm 대상 디바이스 |
| `throttle_delay_ms` | u64 | 0 | 토큰 간 딜레이 (0 = 없음) |
| `suspended` | bool | false | 추론 일시중지 |
| `resumed` | bool | false | 일시중지 해제 |
| `layer_skip` | Option\<f32\> | None | skip ratio (0.0 = skip 없음) |
| `kv_quant_bits` | Option\<u8\> | None | KIVI 양자화 bits (2, 4, 8) |
| `restore_defaults` | bool | false | 모든 action 상태 초기화 |

**EvictPlan**:

| 필드 | 타입 | 범위 | 설명 |
|------|------|------|------|
| `target_ratio` | f32 | 0.0-1.0 | 보존 비율 |
| `level` | ResourceLevel | - | Warning = lossless only, Critical = lossy OK |
| `method` | EvictMethod | - | H2o, Sliding, Streaming |
| `streaming_params` | Option\<StreamingParams\> | - | Streaming 전용 파라미터. 나머지 method에서는 None |

**EvictMethod**: `enum { H2o, Sliding, Streaming }`

- `Streaming`: StreamingLLM 정책 실행. `streaming_params`에서 sink_size, window_size를 추출하여 `StreamingLLMPolicy::new(sink_size, window_size).evict()` 호출

**StreamingParams**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `sink_size` | usize | Attention sink 토큰 수 (시퀀스 선두 유지) |
| `window_size` | usize | Recent window 크기 (시퀀스 후미 유지) |

**KVSnapshot**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `total_bytes` | u64 | KV 캐시 총 바이트 |
| `total_tokens` | usize | 유효 토큰 수 |
| `capacity` | usize | 물리 용량 |
| `protected_prefix` | usize | 보호 prefix 수 |
| `kv_dtype` | String | "f16", "q4", "q2" 등 |
| `eviction_policy` | String | "none", "h2o", "sliding", "d2o" 등 |
| `skip_ratio` | f32 | 현재 layer skip 비율 |

---

### 3.10 EvictionPolicy Trait [ENG-DAT-050]

**[ENG-DAT-050]** EvictionPolicy trait은 eviction 정책의 추상 인터페이스이다. `Send + Sync` 요구. *(MUST)*

| 메서드 | 시그니처 (의사) | 설명 |
|--------|----------------|------|
| `should_evict` | `(&KVCache, mem_available: usize) -> bool` | eviction 트리거 판단 |
| `evict` | `(&mut KVCache, target_len: usize) -> Result<()>` | score 없는 eviction |
| `name` | `() -> &str` | 정책 이름 |
| `evict_with_scores` | `(&mut KVCache, target_len, importance: &[f32]) -> Result<()>` | flat importance array 기반 eviction |
| `evict_with_head_scores` | `(&mut KVCache, target_len, flat, head, n_kv_heads) -> Result<()>` | per-KV-head importance 기반 eviction |

**구현체 4종**:

| 구현체 | name() | 설명 |
|--------|--------|------|
| H2OPolicy | "h2o" | score 기반 3-partition (ENG-ALG-010 참조). `keep_ratio`, `protected_prefix`, `decay` 파라미터 |
| H2OPlusPolicy | "h2o_plus" | per-KV-head eviction (HeadMajor 전용) |
| SlidingWindowPolicy | "sliding_window" | FIFO prune_prefix (ENG-ALG-011 참조) |
| StreamingLLMPolicy | "streaming" | Attention sink + sliding window 결합. `sink_size` + `window_size` 파라미터 |
| NoEvictionPolicy | "none" | `should_evict` 항상 false |

---

### 3.11 QcfMetric [ENG-DAT-060]

**[ENG-DAT-060]** QcfMetric은 단일 QCF 측정의 결과를 표현한다. *(MUST)*

| 필드 | 타입 | 범위 | 설명 |
|------|------|------|------|
| `action` | String | - | 액션 식별자. 가능한 값: `"h2o"`, `"sliding_attn"`, `"eviction_attn"`, `"eviction_caote"`, `"sliding_caote"`, `"kivi"`, `"kivi_opr"`, `"kivi_awqe"`, `"swift"` |
| `raw_value` | f32 | 주로 [0, 1] | 집계된 QCF 값. Eviction: `evicted_imp / total_imp` (clamp 0~1). Non-eviction: unbounded 가능 |
| `normalized_value` | f32 | [0, +inf) | 정규화 값. Eviction: `evicted_imp / remaining_imp` (**unbounded, 1 이상 가능**). Non-eviction: raw_value와 동일 |
| `per_head` | Option\<Vec\<f32\>\> | - | per-KV-head QCF. Layout: `[n_kv_heads]` |
| `tokens_affected` | usize | - | 액션이 영향을 미친 토큰 수 |

> **코드와 기존 스펙 불일치 (00-overview SYS-043)**: SYS-043은 normalized_value를 "[0,1] 정규화"로 기술하나, 실제 코드에서 eviction의 `normalized_value = evicted_importance / remaining_importance`이며 1 이상이 가능하다. 이 스펙은 코드를 따른다.

---

### 3.12 QcfConfig [ENG-DAT-061]

**[ENG-DAT-061]** QcfConfig는 QCF 수집의 런타임 설정이다. *(MUST)*

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | bool | true | QCF 수집 활성화 |
| `mode` | QcfMode | Attn | proxy 모드 |
| `aggregation` | AggregationMode | Mean | head 집계 방식 |
| `d_max` | f32 | 5.0 | 최대 degradation 추정값 |
| `epsilon` | f32 | 1e-8 | division-by-zero 가드 |

**QcfMode**: `enum { Attn, Caote, Both }`

**AggregationMode**: `enum { Mean, Defensive { temperature: f32 } }`

- `Mean`: 단순 평균
- `Defensive`: softmax-weighted mean (ENG-ALG-041 참조). temperature가 낮을수록 worst-case head 강조

---

### 3.13 SkipConfig [ENG-DAT-062]

**[ENG-DAT-062]** SkipConfig는 SWIFT 기반 layer skip 설정이다. *(MUST)*

**필드**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `attn_skip` | HashSet\<usize\> | attention skip 레이어 인덱스 |
| `mlp_skip` | HashSet\<usize\> | MLP skip 레이어 인덱스 |

**메서드**:

| 메서드 | 설명 |
|--------|------|
| `new()` | 빈 설정 (skip 없음) |
| `uniform_init(num_layers, skip_ratio)` | 균등 분배 초기화 (ENG-ALG-030 참조) |
| `validate(num_layers) -> bool` | SWIFT 제약 검증 (layer 0, L-1 보호) |
| `skip_attn(layer_id)` | 해당 레이어 attention skip 여부 |
| `skip_mlp(layer_id)` | 해당 레이어 MLP skip 여부 |
| `total_skips()` | 전체 skip된 sub-layer 수 |
| `is_active()` | 하나라도 skip이면 true |

---

### 3.14 ImportanceTable / ImportanceEntry [ENG-DAT-063]

**[ENG-DAT-063]** ImportanceTable은 prefill 시 1회 계산된 per-layer importance를 저장한다. *(MUST)*

**ImportanceEntry**:

| 필드 | 타입 | 범위 | 설명 |
|------|------|------|------|
| `layer_id` | usize | [0, L) | 레이어 인덱스 |
| `sublayer` | SubLayer | Full, Attention, Mlp | sub-layer 구분 |
| `importance` | f32 | [0, 1] | `1 - cosine_similarity(input, output)`. 0=identity, 1=orthogonal |
| `opr` | f32 | [0, +inf) | `||output-input|| / ||input||` |

**ImportanceTable**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `entries` | Vec\<ImportanceEntry\> | 레이어별 importance 목록 |
| `total_importance` | f32 | 모든 entry의 importance 합 |

**메서드**:

| 메서드 | 설명 |
|--------|------|
| `compute_qcf(skip_set)` | 주어진 skip set의 QCF 값 [0, 1] (ENG-ALG-032 참조) |
| `compute_opr_skip(skip_set)` | skip된 레이어들의 OPR 합 |
| `estimate_qcf_for_count(n, L)` | importance 최저 n개 선택, QCF + skip set 반환 |

---

### 3.15 Engine CLI 플래그 [ENG-DAT-070]

**[ENG-DAT-070]** `generate` 바이너리의 모든 CLI 인수를 정의한다. *(MUST)*

#### 3.15.1 기본 설정

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--model-path` | String | "models/llama3.2-1b" | 모델 경로 |
| `--prompt` | String | "Hello, world! I am a" | 입력 프롬프트 |
| `--prompt-file` | String? | None | 프롬프트 파일 경로 (`--prompt` 대체) |
| `--num-tokens` | usize | 20 | 생성 토큰 수 |
| `--backend` | String | "cpu" | "cpu", "opencl", "hybrid" |
| `--switch-threshold` | usize | 0 | hybrid CPU→GPU 전환 토큰 수 |
| `--zero-copy` | bool | false | zero-copy shared memory |
| `--max-seq-len` | usize | 2048 | 최대 시퀀스 길이 |
| `--threads` | usize | 0 | 스레드 수 (0=auto) |
| `--use-rayon` | bool | false | Rayon vs SpinPool 토글 |

#### 3.15.2 샘플링

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--temperature` | f32 | 0.8 | 샘플링 온도 |
| `--top-p` | f32 | 0.9 | Top-p (nucleus) 샘플링 |
| `--top-k` | usize | 40 | Top-k 샘플링 |
| `--repetition-penalty` | f32 | 1.1 | 반복 페널티 |
| `--repetition-window` | usize | 64 | 반복 페널티 윈도우 |
| `--greedy` | bool | false | 그리디 샘플링 (temperature=0) |
| `--ignore-eos` | bool | false | EOS 무시, 계속 생성 |

#### 3.15.3 GPU / Attention

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--gpu-attn` | bool | false | GPU attention 커널 사용 |
| `--no-gpu-plan` | bool | false | GPU 커널 플랜 비활성 |
| `--no-prefill-ws` | bool | false | PrefillWorkspace 비활성 |
| `--prefill-chunk-size` | usize | 0 | Chunked prefill 크기 (0=비활성, ENG-ALG-080 참조) |

#### 3.15.4 KV 캐시

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--weight-dtype` | String | "f16" | 모델 가중치 타입 ("f16", "q4") |
| `--kv-type` | String | "f16" | KV 캐시 타입 ("f32", "f16", "q4") |
| `--kv-layout` | String | "head" | "head" (HeadMajor) 또는 "seq" (SeqMajor) |
| `--initial-kv-capacity` | usize | 0 | 초기 KV 용량 (0=auto) |
| `--kv-budget` | usize | 0 | KV 예산 (토큰, 0=무제한) |
| `--kv-budget-ratio` | f32 | 0.0 | KV 예산 (prompt 대비 비율) |

#### 3.15.5 KIVI 양자화

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--kivi` | bool | false | KIVI 양자화 활성 |
| `--kivi-residual-size` | usize | 32 | KIVI residual 버퍼 크기 |

#### 3.15.6 Eviction 정책

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--eviction-policy` | String | "none" | "none", "sliding", "streaming", "h2o", "h2o_plus", "d2o" |
| `--eviction-window` | usize | 1024 | sliding/streaming 윈도우 크기 |
| `--sink-size` | usize | 4 | StreamingLLM attention sink 토큰 수 |
| `--streaming-window` | usize | 0 | StreamingLLM recent 윈도우 (0=auto) |
| `--protected-prefix` | usize? | None | eviction 보호 prefix (기본: 정책별 상이) |
| `--memory-threshold-mb` | usize | 256 | eviction 트리거 메모리 임계값(MB) |
| `--eviction-target-ratio` | f32 | 0.75 | eviction 보존 비율 |

#### 3.15.7 H2O 설정

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--h2o-keep-ratio` | f32 | 0.5 | H2O heavy hitter 보존 비율 |
| `--h2o-tracked-layers` | usize | 0 | H2O score 추적 레이어 수 (0=전체) |
| `--h2o-decay` | f32 | 0.0 | H2O score 지수 감쇠 (0=없음) |
| `--h2o-debug` | bool | false | H2O 디버그 출력 |
| `--h2o-raw-scores` | bool | false | time-normalized scoring 비활성 |

#### 3.15.8 D2O 설정

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--d2o-keep-ratio` | f32 | 0.75 | D2O heavy hitter 비율 |
| `--d2o-ema-alpha` | f32 | 0.5 | D2O EMA old-threshold 가중치 |
| `--d2o-ema-beta` | f32 | 0.5 | D2O EMA new-mean 가중치 |
| `--d2o-layer-alloc` | bool | false | D2O per-layer 동적 할당 |
| `--d2o-protected-layers` | Vec\<usize\>? | None | D2O 보호 레이어 |

#### 3.15.9 Layer Skip

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--skip-layers` | Vec\<usize\>? | None | 명시적 skip 레이어 |
| `--skip-ratio` | f32? | None | skip 비율 (uniform_init, ENG-ALG-030 참조) |
| `--dump-importance` | bool | false | importance table 출력 후 종료 |

#### 3.15.10 QCF

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--qcf-mode` | String | "attn" | "attn", "caote", "both" |

#### 3.15.11 Resilience

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--enable-resilience` | bool | false | Resilience Manager 활성 |
| `--resilience-transport` | String | "dbus" | "dbus", "unix:\<path\>", "tcp:\<addr\>" |
| `--experiment-eviction-ratio` | f32? | None | resilience eviction ratio override |

#### 3.15.12 KV Offload

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--kv-offload` | String | "none" | "none", "raw", "disk" |
| `--offload-path` | String | "" | disk offload 경로 |
| `--max-prefetch-depth` | usize | 4 | offload prefetch depth |

#### 3.15.13 프로파일링

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--profile` | bool | false | 프로파일링 활성 |
| `--profile-dir` | String | "results/profile" | 프로파일 출력 디렉토리 |
| `--profile-interval` | usize | 1 | Score snapshot 간격 |
| `--profile-probes` | String | "ops,latency,scores" | 프로파일 프로브 목록 |
| `--profile-per-head` | bool | false | per-KV-head score 추적 |

#### 3.15.14 평가

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--eval-ll` | bool | false | log-likelihood 평가 모드 |
| `--eval-continuation` | String? | None | 평가 continuation 텍스트 |
| `--eval-batch` | String? | None | 평가 배치 JSON |
| `--ppl` | String? | None | perplexity 평가 참조 텍스트 |

#### 3.15.15 실험

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--experiment-schedule` | String? | None | 실험 스케줄 JSON |
| `--experiment-output` | String? | None | 실험 출력 JSONL |
| `--experiment-logits-topk` | usize | 10 | 실험 top-K logits 기록 수 |
| `--experiment-sample-interval` | usize | 1 | 시스템 메트릭 샘플 간격 |

---

### 3.15.16 Weight Loading / Dynamic Weight Swap

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--model-path` | String | (기존) | Primary 가중치 파일. 초기 로딩 시 모든 decoder layer가 이 파일의 dtype으로 로드된다. |
| `--model-path-secondary` | String? | None | Secondary 가중치 파일 (낮은 정밀도, e.g. Q4_0). 제공 시 디스크에 mmap만 되고, 런타임 swap 대상으로 예약된다. (ENG-DAT-090) |
| `--force-swap-ratio` | `Option<f32>` | None | 디버그 전용. Manager 없이 prefill 종료 시 `ResilienceAction::SwapWeights { ratio }`를 직접 트리거한다. 값은 `[0.0, 1.0]`. (ENG-ALG-211 debug hook) |

---

### 3.16 PressureLevel / ActionResult [ENG-DAT-080]

**[ENG-DAT-080]** CachePressurePipeline의 pressure level과 handler 결과 타입을 정의한다. *(MUST)*

**PressureLevel**: `type PressureLevel = llm_shared::Level` -- `Normal < Warning < Critical < Emergency` (Ord derive).

**ActionResult**:

| Variant | 필드 | 설명 |
|---------|------|------|
| `NoOp` | - | 액션 미수행 |
| `Evicted` | `tokens_removed: usize`, `new_pos: usize` | 토큰 제거됨 |
| `Quantized` | - | KV 정밀도 감소 (stub) |
| `Swapped` | `tokens_swapped: usize` | KV 토큰 offload됨 |
| `WeightSwapped` | `layers: Vec<(usize, DType, DType)>`, `freed_bytes: u64`, `latency_ms: u64` | Weight layer dtype 교체됨. `(layer_idx, from_dtype, to_dtype)` 튜플 목록. `freed_bytes`는 `madvise(DONTNEED)` 후 회수된 상주 바이트. `latency_ms`는 swap 전체 실행 시간. (WSWAP-2, INV-123 검증 기반) |

**HandlerContext**: 각 CachePressureHandler에 전달되는 컨텍스트.

| 필드 | 타입 | 설명 |
|------|------|------|
| `caches` | &mut [KVCache] | 모든 레이어의 KV 캐시 |
| `importance` | Option\<&[f32]\> | per-token importance scores |
| `head_importance` | Option\<&[f32]\> | per-KV-head importance scores |
| `n_kv_heads` | usize | KV head 수 |
| `pressure_level` | PressureLevel | 현재 pressure level |
| `mem_available` | usize | 가용 메모리 바이트 |
| `target_ratio` | Option\<f32\> | 외부 override (resilience) |
| `qcf_sink` | Option\<&mut Vec\<QcfMetric\>\> | QCF 메트릭 수집 대상 |
| `layer_ratios` | Option\<&[(f32, f32)]\> | D2O layer allocation |

---

### 3.17 LoadConfig — Weight Loading with Dynamic Swap [ENG-DAT-090]

**[ENG-DAT-090]** `LoadConfig`는 가중치 로딩의 런타임 설정이다. 초기 로딩은 단일 `default_dtype`을 사용하고, `secondary_source`가 제공되면 런타임 swap 대상 파일로 예약한다. *(MUST)*

**필수 필드**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `primary_source` | PathBuf | Primary 가중치 파일 경로. 기존 `--model-path`가 연결되는 지점. 초기 모든 decoder layer 로딩 소스. |
| `default_dtype` | DType | Primary 파일이 공급하는 기본 dtype. Loader가 파일 헤더에서 추론하며 초기 로딩 시점의 `LayerSlot::current_dtype` 초깃값. |

**신규 필드** (Dynamic Swap 지원):

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `secondary_source` | `Option<PathBuf>` | None | Secondary 가중치 파일 경로. Primary보다 낮은 정밀도 dtype(e.g. Q4_0)이어야 의미가 있다. **초기 로딩에 사용되지 않는다** — mmap 핸들만 `TransformerWeights`에 보관되어 나중에 `SwapExecutor`가 layer 단위로 참조한다. |

**의미론**:

- `secondary_source == None`이면 swap 경로 비활성. `ResilienceAction::SwapWeights`는 즉시 NoOp 반환.
- `secondary_source == Some(path)`이면 로더가 primary에 더해 secondary 파일의 metadata(n_layer, n_head, n_kv_head, hidden_size, intermediate_size, head_dim)를 검증하고 mmap을 열어둔다. mmap handle은 `LayerSlot::secondary_mmap_handle`로 공유된다 (ENG-DAT-092).
- 초기 로딩 후 모든 decoder layer의 `current_dtype`은 `default_dtype`과 같고, `Primary` 전용 상태이다.
- Cross-layer tensor(embedding, final_norm, lm_head)는 항상 primary에서 로드된다. **Swap 대상이 아니다.**
- Backend 무관. OpenCL 백엔드의 `rewrap_weights_for_dual_access()`는 swap 직후의 새 Buffer에 대해서도 동일 규칙으로 호출된다.

**ENG-DAT-091 [DEPRECATED 2026-04-24]**: 구 TOML `LayerDtypeProfile` 스키마는 정적 per-layer profile 노선과 함께 폐기되었다. 동적 swap은 Manager 신호 기반이며 TOML 입력이 없다. `quantize_profile` 바이너리도 동시 폐기. ID는 재사용하지 않는다.

---

### 3.18 LayerSlot — Swappable Weight Slot [ENG-DAT-092]

**[ENG-DAT-092]** `LayerSlot`은 decoder block 한 layer의 가중치 묶음을 런타임에 교체 가능한 단위로 캡슐화한다. Forward pass는 이 slot을 통해서만 layer weight에 접근한다. *(MUST)*

**필드**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `current_dtype` | `DType` | 현재 slot이 보유한 weight dtype. 초기값은 `LoadConfig::default_dtype`. swap 후 secondary dtype으로 전환. `INV-124` 대상. |
| `weights` | `Arc<LayerWeights>` 또는 동치 snapshot 타입 | Layer의 QKV/O/gate/up/down + attn_norm/ffn_norm tensor 묶음. **Lock-free snapshot 교체**가 핵심 요건. 기본 권장 구현은 `arc_swap::ArcSwap<LayerWeights>`. 대안은 `RwLock<Arc<LayerWeights>>` 또는 custom epoch 기반 swap. 최종 선택은 Senior Implementer PoC의 decode latency 측정 후 확정한다. |
| `secondary_mmap_handle` | `Option<Arc<SecondaryMmap>>` | Secondary 파일의 mmap view. Layer 범위의 tensor slice 정보(offset+len+dtype)를 포함. `None`이면 이 layer는 swap 불가. |
| `generation` | `AtomicU64` | swap 발생 시마다 1씩 증가. Forward 진입 시점의 값을 캡처하여 swap 도중 재진입을 감지한다. INV-120과 동일 패턴. |

**후조건**:
- Forward 읽기 경로는 `weights` snapshot을 `Arc::clone`으로 획득 후 사용. 진행 중 swap이 발생해도 해당 forward는 이전 snapshot을 계속 사용하고, **다음 layer 진입 시 최신 snapshot을 본다**.
- `current_dtype`은 snapshot 교체와 **동일 원자 단계**에 갱신되어야 한다(INV-124).

---

### 3.19 TransformerWeights — Slot Collection [ENG-DAT-093]

**[ENG-DAT-093]** `TransformerWeights`는 모델 전체의 per-layer slot과 cross-layer tensor, 공유 secondary mmap handle을 보관한다. *(MUST)*

**필드**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `layers` | `Vec<LayerSlot>` | Decoder block layer 목록. 길이 = `num_layers`. 순서는 layer index와 동일. |
| `embedding` | `Arc<Tensor>` | 임베딩 테이블. **Swap 대상 아님**. `LoadConfig::default_dtype`으로 로드된 상태 유지. |
| `final_norm` | `Arc<Tensor>` | 최종 RMSNorm weight. **Swap 대상 아님**. |
| `lm_head` | `Option<Arc<Tensor>>` | 출력 projection. tie_word_embeddings 모델(Qwen 등)은 `None`이며 forward는 `embedding`을 재사용. **Swap 대상 아님**. |
| `secondary_mmap` | `Option<Arc<SecondaryMmap>>` | Secondary 파일 mmap 소유권 핸들. 모든 `LayerSlot::secondary_mmap_handle`은 여기서 clone된 Arc를 공유. 생성자에서 1회 설정 후 모델 lifetime 동안 **drop 금지**(INV-125). |
| `ratio_generation` | `AtomicU64` | 전체 swap 세대 카운터. `SwapExecutor`가 ratio 기반 다중 layer를 한 배치로 교체할 때 한 번 증가. Forward plan 재빌드 트리거 키 (ENG-ALG-200의 `ratio_generation`과 의미 통합). |

**불변식**:
- `secondary_mmap.is_some()` ⇒ `secondary_mmap` handle은 `TransformerWeights::drop()` 시점까지 살아있다 (INV-125).
- 각 `LayerSlot::secondary_mmap_handle.is_some()` ⇒ 해당 Arc는 `TransformerWeights::secondary_mmap`과 동일 pointee.

---

### 3.20 SecondaryMmap — Layer Tensor View [ENG-DAT-094]

**[ENG-DAT-094]** `SecondaryMmap`은 secondary GGUF 파일의 메모리 매핑과, layer별 tensor slice 인덱스를 보관하는 read-only 핸들이다. *(MUST)*

**필드**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `mmap` | `memmap2::Mmap` 또는 동치 | Secondary 파일 전체 read-only mmap. |
| `layer_index` | `Vec<LayerTensorSlice>` | 각 decoder layer에 속한 tensor들의 `(name, offset, len, dtype, shape)` 목록. GGUF header 파싱 1회로 구축. |
| `cross_layer_offsets` | `HashMap<String, (u64, u64, DType)>` | 참고용. Swap 대상이 아니지만 metadata 검증에 사용. |

**의미론**:
- `SwapExecutor`는 `layer_index[i]`를 조회해 해당 layer에 필요한 tensor의 raw byte slice를 얻고, Q/K permutation 등 primary loader와 동일한 후처리를 적용하여 새 `LayerWeights`를 생성한다.
- `mmap` 자체는 swap 후에도 drop되지 않는다. `madvise(DONTNEED)`는 swap된 **primary** 페이지 영역에만 적용된다 (INV-125의 보호 대상이 아닌 쪽).

## 4. Alternative Behavior

해당 없음. 이 문서는 데이터 정의 문서이다. 데이터 처리의 대안 동작은 `32-engine-algorithms.md`에서 다룬다.

## 5. Constraints

**[ENG-DAT-C01]** KVCacheOps는 dyn Trait이 아닌 generic monomorphization으로 사용한다. 런타임 dispatch overhead가 없어야 한다. *(MUST)*

**[ENG-DAT-C02]** KiviCache의 `kv_dtype()`은 항상 F32를 반환한다. 호출자는 F32 텐서로 `update()`를 호출하며, 내부에서 양자화가 수행된다. *(MUST)*

**[ENG-DAT-C03]** KiviCache의 `get_buffers_mut()`은 항상 None을 반환한다. zero-copy scatter write가 불가능하다. *(MUST)*

**[ENG-DAT-C04]** KVLayout 변경은 KV 캐시 재생성을 요구한다. 런타임 layout 전환은 지원하지 않는다. *(MUST NOT)*

**[ENG-DAT-C05]** ~~(폐기)~~ EvictMethod::Streaming은 구현 완료되었다. executor.rs에서 `KvStreaming` 수신 시 `EvictPlan { method: Streaming, streaming_params: Some(StreamingParams { sink_size, window_size }), target_ratio: 0.0, pressure_level: Critical }`을 생성하고 `CommandResult::Ok`를 반환한다. generate.rs에서 `StreamingLLMPolicy::new(sink_size, window_size).evict()` 즉석 호출로 실행한다. *(MUST)*

**[ENG-DAT-C06]** QcfMetric의 `action` 필드 값은 이 스펙에 명시된 9종 문자열 중 하나여야 한다. *(MUST)*

**[ENG-DAT-C07]** Tensor의 `reshape()`은 numel이 동일한 경우에만 허용된다. *(MUST)*

**[ENG-DAT-C08]** Swap 대상 layer index는 `[0, num_layers)` 범위 내여야 하며, 이 범위 밖 인덱스를 담은 swap 요청은 NoOp으로 처리된다. *(MUST)*

**[ENG-DAT-C09]** `LoadConfig::secondary_source == None`이면 `ResilienceAction::SwapWeights`는 무조건 NoOp을 반환한다. 로딩 경로는 단일 primary 파일만 열고, `LayerSlot::secondary_mmap_handle`은 모두 `None`이다. *(MUST)*

**[ENG-DAT-C10]** Primary와 secondary 가중치 파일의 모델 메타데이터(GGUF metadata의 `n_layer`, `n_head`, `n_kv_head`, `hidden_size`, `intermediate_size`, `head_dim`) 및 각 layer tensor의 shape은 모두 일치해야 한다. 불일치 시 loader는 에러 반환하고 swap 경로도 비활성화된다. *(MUST)*

**[ENG-DAT-C11]** Cross-layer tensor(embedding, final_norm, lm_head)는 `TransformerWeights`의 swap 대상에서 제외된다. 이들의 dtype은 모델 lifetime 동안 `LoadConfig::default_dtype`으로 고정이다. *(MUST)*

**[ENG-DAT-C12]** `TransformerWeights::secondary_mmap`이 `Some`인 동안 해당 `Arc<SecondaryMmap>`은 drop될 수 없다. `LayerSlot::secondary_mmap_handle`의 모든 clone이 drop되어도 `TransformerWeights`가 최후 소유권을 보존한다. *(MUST)*

## 6. Examples

### 6.1 KVCache 생성 및 사용

```
// HeadMajor, F16, 8 heads, 64 dim, max 2048 tokens
cache = KVCache::new_dynamic(kv_heads=8, head_dim=64, max_seq_len=2048,
                              layout=HeadMajor, dtype=F16, memory=cpu_memory,
                              initial_capacity=256)

// Prefill: 100 tokens
cache.ensure_capacity(100)       // capacity=256, 충분
cache.update(k_tensor, v_tensor) // current_pos=100, high_water_pos=100

// Decode: 200 tokens 추가
for _ in 0..200:
    cache.ensure_capacity(current_pos + 1)
    if current_pos + 1 > capacity:
        cache.grow(current_pos + 1)  // capacity: 256 -> 512
    cache.update(k_token, v_token)

// Eviction: 300 -> 200 tokens
h2o_evict(cache, target_len=200, ...)
// current_pos=200, release_unused_pages() -> shrink_to_fit 또는 madvise
```

### 6.2 KiviCache Lifecycle

```
// Q4, residual_size=64
kivi = KiviCache::new(bits=4, kv_heads=8, head_dim=64,
                       max_seq_len=2048, residual_size=64)

// kv_dtype() returns F32 (호출자는 F32 텐서 전달)
assert kivi.kv_dtype() == F32

// Token 1~64: residual에 누적
for t in 1..=64:
    kivi.update(k_f32, v_f32)  // res_pos: 1..64

// Token 64: flush 발생
// res_pos(64) >= res_cap(64) -> flush_residual()
// q2_tokens=64, res_pos=0

// Token 65~128: 다시 누적 -> flush
// q2_tokens=128

// get_view(): incremental dequant + residual copy
(k_view, v_view) = kivi.get_view()
// k_view.shape = [1, 128+res_pos, 8, 64]
```

### 6.3 CLI 조합 예시

```bash
# H2O eviction + KIVI Q4 + Resilience
./generate --model-path models/qwen2.5-1.5b \
  --eviction-policy h2o --h2o-keep-ratio 0.5 \
  --kivi --kivi-residual-size 64 \
  --kv-layout head --kv-type f16 \
  --enable-resilience --resilience-transport "unix:/tmp/llm.sock" \
  --num-tokens 500 --max-seq-len 4096

# D2O + Layer Skip + Chunked Prefill + GPU
./generate --model-path models/llama3.2-1b \
  --eviction-policy d2o --d2o-keep-ratio 0.75 --d2o-layer-alloc \
  --skip-ratio 0.3 \
  --prefill-chunk-size 512 \
  --backend opencl --gpu-attn \
  --num-tokens 1000
```

## 7. Rationale (non-normative)

### 왜 KVCacheOps가 dyn Trait이 아닌가

KV 캐시 연산은 추론 루프의 hot path에 있다. 매 토큰, 매 레이어에서 `update()`와 `get_view()`가 호출된다. dyn Trait의 vtable 간접 호출은 분기 예측 실패와 인라인 불가로 인한 성능 저하를 유발한다. Generic monomorphization은 컴파일 타임에 구체 타입을 결정하여 zero-cost 추상화를 보장한다.

### 왜 KiviCache가 kv_dtype에서 F32를 반환하는가

KiviCache의 residual buffer가 FP32이므로, 호출자는 항상 FP32 텐서로 update()를 수행한다. 양자화는 KiviCache 내부에서 flush 시 발생하며 호출자에게 투명하다. 이는 KVCacheOps 인터페이스의 균일성을 보장한다 -- 호출자 코드가 캐시 구현체에 무관하게 동작한다.

### 왜 SeqMajor와 HeadMajor 2종인가

SeqMajor는 `prune_prefix()`(sliding window)에서 연속 메모리 이동으로 최적이다. HeadMajor는 per-head eviction(H2O/H2O+)에서 head 단위 독립 처리가 가능하여 cache-friendly이다. Eviction 정책에 따라 적합한 layout이 다르므로 두 가지를 모두 지원한다.

### 왜 Buffer trait에 is_host_managed()가 필요한가

madvise(MADV_DONTNEED)는 앱이 소유한 anonymous private mapping에서만 물리 페이지를 해제한다. GPU 드라이버가 할당한 메모리(UnifiedBuffer, OpenCLBuffer)에서는 효과가 없거나 정의되지 않은 동작을 유발한다. `is_host_managed()`는 madvise 안전성을 런타임에 판단하는 유일한 수단이다.

### 왜 CLI 플래그가 60+ 개인가

llm_rs2는 연구 프로토타입으로서 eviction 정책, 양자화, layer skip, QCF 모드, 프로파일링 등 다양한 실험 설정을 하나의 바이너리에서 조합해야 한다. 각 플래그는 독립적인 실험 변수를 제어하며, 플래그 조합으로 실험 매트릭스를 구성한다. 프로덕션 배포 시에는 설정 파일(TOML/JSON)로 축약할 수 있다.
