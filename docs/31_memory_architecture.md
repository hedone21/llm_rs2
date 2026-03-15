# Chapter 31: Memory Architecture Overview

> **이전**: [30. 평가 방법론](30_evaluation_methodology.md) | **다음**: [32. KV 캐시 오프로드](32_kv_offload.md)

이 문서는 llm.rs 프레임워크의 메모리 관리 시스템을 총체적으로 설명합니다. 버퍼 할당부터 KV 캐시 관리까지, 추론 중 메모리가 어떻게 할당·사용·해제되는지를 하나의 관점에서 조망합니다.

---

## 31.1 Memory Stack 개요

llm.rs의 메모리 시스템은 4개 계층으로 구성됩니다:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 4: KV Cache 정책 (Eviction / Offload)             │ ← docs 11, 32
│    CacheManager, EvictionPolicy, OffloadKVCache,        │
│    PrefetchController, PreloadPool                      │
├─────────────────────────────────────────────────────────┤
│  Layer 3: KV Cache 데이터 구조                           │ ← §31.3
│    KVCache (SeqMajor/HeadMajor), KVCacheOps trait,     │
│    PrefetchableCache trait, OffloadStore trait           │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Tensor + Workspace                             │ ← §31.4
│    Tensor (Shape + Buffer + Backend),                    │
│    LayerWorkspace (pre-allocated intermediates)          │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Buffer + Allocator                             │ ← docs 08, §31.2
│    Buffer trait, Memory trait,                           │
│    SharedBuffer, UnifiedBuffer, OpenCLBuffer,            │
│    Galloc, OpenCLMemory                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 31.2 Layer 1: Buffer & Allocator

### 핵심 Trait

**Buffer** (`core/buffer.rs`): 물리 메모리 추상화

```rust
pub trait Buffer: Send + Sync {
    fn dtype(&self) -> DType;
    fn size(&self) -> usize;
    fn as_ptr(&self) -> *const u8;      // CPU 접근 포인터 (GPU-only면 null)
    fn as_mut_ptr(&self) -> *mut u8;
    fn cl_mem(&self) -> Option<&Mem>;   // OpenCL 핸들 (CPU-only면 None)
    fn sync_device(&self) -> Result<()>;

    // Zero-copy 지원 (ARM SoC용)
    fn map_for_cpu(&self) -> Result<()> { Ok(()) }
    fn unmap_for_gpu(&self) -> Result<()> { Ok(()) }
    fn is_mapped(&self) -> bool { true }
}
```

**Memory** (`core/memory.rs`): 할당자 추상화

```rust
pub trait Memory: Send + Sync {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>>;
    fn used_memory(&self) -> usize;
}
```

### Buffer 구현체

| 구현 | 파일 | CPU 포인터 | GPU 핸들 | 용도 |
|------|------|-----------|---------|------|
| `SharedBuffer` | `buffer/shared_buffer.rs` | `Vec<u8>` 직접 | None | 호스트 전용, 테스트 |
| `UnifiedBuffer` | `buffer/unified_buffer.rs` | `CL_MEM_ALLOC_HOST_PTR` 매핑 | 있음 | ARM SoC zero-copy |
| `OpenCLBuffer` | `backend/opencl/buffer.rs` | null (CPU 접근 불가) | 있음 | GPU 전용 (discrete) |

### Allocator 구현체

| 구현 | 반환 타입 | 용도 |
|------|----------|------|
| `Galloc` | `SharedBuffer` | CPU 전용 테스트 |
| `OpenCLMemory` (`use_zero_copy=true`) | `UnifiedBuffer` | ARM SoC (기본) |
| `OpenCLMemory` (`use_zero_copy=false`) | `OpenCLBuffer` | Discrete GPU |

### Zero-Copy 매핑 상태 머신

ARM SoC에서 `UnifiedBuffer`는 CPU↔GPU 동일 물리 메모리를 공유합니다:

```
생성 (unmapped, GPU 접근)
    ↓ map_for_cpu()
CPU 읽기/쓰기 가능 (is_mapped=true)
    ↓ unmap_for_gpu()
GPU 커널 실행 가능 (is_mapped=false)
    ↓ map_for_cpu()
    ...
```

`UnifiedBuffer` 내부: `AtomicBool`(lock-free 상태 확인) + `Mutex<*mut u8>`(포인터 보호)

> **상세**: [Chapter 8: Memory Management](08_memory_management.md) — 데이터 전송 패턴, copy_slice, 동기화 전략

---

## 31.3 Layer 2: Tensor & Workspace

### Tensor

`Tensor` (`core/tensor.rs`)는 `Shape + Arc<dyn Buffer> + Arc<dyn Backend>`를 묶는 논리적 배열입니다.

```rust
pub struct Tensor {
    shape: Shape,                // 다차원 레이아웃
    buffer: Arc<dyn Buffer>,     // 참조 계수 기반 물리 메모리
    backend: Arc<dyn Backend>,   // 연산 디스패치용
}
```

**핵심 특성**:
- `Clone`은 데이터 복사 없이 `Arc` 증가만 수행
- `as_slice::<f32>()`로 타입 안전 슬라이싱
- Buffer가 `Send + Sync`이므로 스레드 간 공유 안전

### LayerWorkspace

`LayerWorkspace` (`layers/workspace.rs`)는 레이어별 중간 버퍼를 사전 할당하여 토큰당 할당 오버헤드를 제거합니다.

```rust
pub struct LayerWorkspace {
    pub q: Tensor,          // [batch, 1, q_dim]
    pub k: Tensor,          // [batch, 1, k_dim]
    pub v: Tensor,          // [batch, 1, v_dim]
    pub out_attn: Tensor,   // [batch, 1, q_dim]
    pub gate: Tensor,       // [batch, 1, ffn_hidden]
    pub up: Tensor,         // [batch, 1, ffn_hidden]
    pub down: Tensor,       // [batch, 1, dim]
    pub residual: Tensor,   // [batch, 1, dim]
    pub attn_out: Tensor,   // [batch, 1, dim]
    pub scores: Vec<f32>,   // [n_heads × max_seq_len] CPU 측
}
```

모델 초기화 시 한 번 생성되어 모든 토큰에서 재사용됩니다.

---

## 31.4 Layer 3: KV Cache 데이터 구조

### KVCacheOps Trait

모든 KV 캐시 구현이 준수하는 핵심 인터페이스:

```rust
pub trait KVCacheOps: Send {
    fn current_pos(&self) -> usize;        // 유효 토큰 수
    fn capacity(&self) -> usize;           // 물리 버퍼 용량
    fn kv_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn layout(&self) -> KVLayout;          // SeqMajor / HeadMajor
    fn kv_dtype(&self) -> DType;
    fn memory_usage_bytes(&self) -> usize;
    fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()>;
    fn get_view(&mut self) -> (Tensor, Tensor);
}
```

제네릭 모노모피즘(`<C: KVCacheOps>`)으로 사용되어 `dyn Trait` 오버헤드 없이 연속 배열(`Vec<C>`) 할당이 가능합니다.

### PrefetchableCache Trait

오프로드 가능한 캐시를 위한 확장:

```rust
pub trait PrefetchableCache: KVCacheOps {
    fn preload(&mut self) -> Result<()>;   // 저장소 → RAM 로드
    fn release_buffers(&mut self);          // RAM 해제
    fn reset_preload(&mut self);            // 토큰 경계 리셋
    fn retain_preload(&mut self) {}         // 교차 토큰 유지
}
```

### KVLayout

```rust
pub enum KVLayout {
    SeqMajor,    // [batch, seq_pos, kv_heads, head_dim] — 위치 연속
    HeadMajor,   // [batch, kv_heads, seq_pos, head_dim] — 헤드 연속
}
```

| 레이아웃 | 장점 | 필요 조건 |
|---------|------|----------|
| SeqMajor | 순차 쓰기 효율, 오프로드 호환 | 기본값, 전체 호환 |
| HeadMajor | 헤드별 eviction (H2O+) | per-head 정책 사용 시 |

### KV Cache 구현체 비교

| 구현 | 품질 | 메모리 전략 | Eviction | 레이아웃 | 파일 |
|------|------|-----------|---------|---------|------|
| **KVCache** | 무손실 | 인메모리 (동적 성장) | 지원 | 양쪽 | `core/kv_cache.rs` |
| **OffloadKVCache** | 무손실 | RawStore + 레이어별 프리페치 | 미지원 | SeqMajor만 | `core/offload/mod.rs` |

---

## 31.5 Layer 4: KV Cache 정책

### 정책 분류

```
KV Cache 정책
├── Eviction (토큰 제거로 메모리 확보)
│   ├── NoEvictionPolicy        — 아무것도 안 함 (기본)
│   ├── SlidingWindowPolicy     — 최근 N 토큰 유지
│   ├── H2OPolicy              — 3-partition (prefix + heavy hitters + recent)
│   └── D2OHandler              — merge compensation (pipeline handler)
│
└── Offloading (외부 저장소로 이동)
    └── OffloadKVCache + RawStore  — 무압축 인메모리 오프로드
```

### Eviction vs Offloading

| 속성 | Eviction | Offloading |
|------|---------|-----------|
| 목적 | 오래된 토큰 제거 | 비활성 레이어 저장소 이전 |
| 품질 영향 | 있음 (토큰 유실) | 없음 (무손실) |
| 트리거 | Resilience 시그널 | 항상 활성 (decode 시) |
| 메모리 절감 | 토큰 수 비례 | 활성 레이어 수 비례 |
| 호환성 | 상호 비호환 | 상호 비호환 |
| 적용 단위 | 전체 캐시 | 레이어별 |

> **Eviction 상세**: [Chapter 11: KV Cache 관리 전략](11_kv_cache_management.md)
>
> **Offloading 상세**: [Chapter 32: KV 캐시 오프로드](32_kv_offload.md)

---

## 31.6 추론 중 메모리 흐름

### Prefill 단계

```
Input Tokens [batch, seq_len]
    ↓ Embedding lookup
Hidden [batch, seq_len, dim]
    ↓
┌─────────── Layer Loop (i = 0..num_layers) ───────────┐
│                                                       │
│  RMSNorm → QKV Matmul → RoPE                        │
│      ↓                                               │
│  kv_cache[i].update(K, V)     ← KV 데이터 저장       │
│      ↓                                               │
│  kv_cache[i].get_view()       ← K, V 텐서 반환       │
│      ↓                                               │
│  Attention (Q × K^T → softmax → × V)                │
│      ↓                                               │
│  Output Proj → Residual → FFN → Residual             │
│                                                       │
│  [LayerWorkspace: q, k, v, gate, up, down 재사용]     │
└───────────────────────────────────────────────────────┘
    ↓
Final RMSNorm → LM Head → Logits [batch, vocab_size]
    ↓ D2H (GPU → CPU, argmax용)
Next Token
```

### Decode 단계 — 모드별 차이

**표준 모드** (`forward_into`):
- 16개 레이어 × 인메모리 KV 버퍼 상주
- 메모리: `num_layers × seq_len × kv_heads × head_dim × dtype_size × 2`

**오프로드 모드** (`forward_into_offload`):
- `depth`개 레이어만 attn 버퍼 보유 (나머지는 RawStore에 저장)
- 백그라운드 스레드에서 다음 레이어 프리로드
- 메모리: `depth × seq_len × kv_heads × head_dim × dtype_size × 2` + store 데이터

---

## 31.7 메모리 예산 (Llama 3.2 1B, seq=2048)

### 모델 가중치 (고정, ~1.2 GB Q4_0)

| 구성요소 | 크기 |
|---------|------|
| 임베딩 + LM Head | ~256 MB |
| 16 × Transformer Layer (Q4_0) | ~960 MB |

### KV Cache (가변, dtype/모드별)

| 모드 | KV DType | 활성 메모리 | 총 KV 데이터 | 비고 |
|------|---------|-----------|------------|------|
| 표준 | F16 | 64 MB | 64 MB | 16L × 4MB |
| 표준 | F32 | 128 MB | 128 MB | 16L × 8MB |
| 오프로드 (Raw, depth=2) | F16 | ~16 MB | 64 MB | 2L attn + 8MB out |
| 오프로드 (Raw, depth=2) | F32 | ~32 MB | 128 MB | 2L attn + 16MB out |

### Workspace (고정, 레이어 공유)

| 구성요소 | F16 크기 | F32 크기 |
|---------|---------|---------|
| q + k + v + out_attn | ~32 KB | ~64 KB |
| gate + up + down | ~48 KB | ~96 KB |
| residual + attn_out | ~16 KB | ~32 KB |
| scores (CPU) | ~32 KB | ~32 KB |

Workspace는 decode 시 단일 토큰용이므로 미미합니다.

---

## 31.8 OffloadStore 추상화

레이어별 KV 데이터의 저장/로드를 추상화하는 trait:

```rust
pub trait OffloadStore: Send {
    fn store(&mut self, k_data: &[u8], v_data: &[u8], num_tokens: usize) -> Result<()>;
    fn load_into(&self, k_buf: &mut [u8], v_buf: &mut [u8]) -> Result<usize>;
    fn append_token(&mut self, k_token: &[u8], v_token: &[u8]) -> Result<()>;
    fn storage_size(&self) -> usize;
    fn stored_tokens(&self) -> usize;
    fn clear(&mut self);
}
```

현재 유일한 구현체는 `RawStore` (무압축 `Vec<u8>` 저장)입니다.

> **참고**: DiskStore(디스크 I/O)와 ZramStore(LZ4 압축)는 2026-03-15 리팩토링에서 제거되었습니다.
> 고엔트로피 F16/F32 KV 활성화 데이터에서 압축 효율이 낮고 복잡성만 증가시키기 때문입니다.

---

## 31.9 파일 구조 요약

```
engine/src/
├── core/
│   ├── buffer.rs           Buffer trait, DType enum
│   ├── memory.rs           Memory trait
│   ├── tensor.rs           Tensor struct
│   ├── backend.rs          Backend trait (17+ ops)
│   ├── kv_cache.rs         KVCache, KVCacheOps, PrefetchableCache, KVLayout
│   ├── cache_manager.rs    CacheManager (pressure pipeline)
│   ├── eviction/           Eviction 정책 (NoEviction, SlidingWindow, H2O)
│   ├── pressure/           CachePressure 핸들러 (D2O, stubs)
│   ├── offload/
│   │   ├── mod.rs          OffloadKVCache + tests (2071줄)
│   │   ├── store.rs        OffloadStore trait (28줄)
│   │   ├── raw_store.rs    RawStore (160줄)
│   │   ├── prefetch.rs     PrefetchController (279줄)
│   │   └── preload_pool.rs PreloadPool (241줄)
│   └── ...
├── buffer/
│   ├── shared_buffer.rs    SharedBuffer (CPU Vec)
│   └── unified_buffer.rs   UnifiedBuffer (zero-copy)
├── memory/
│   └── galloc.rs           Galloc allocator
├── layers/
│   └── workspace.rs        LayerWorkspace
└── backend/opencl/
    ├── buffer.rs           OpenCLBuffer
    └── memory.rs           OpenCLMemory allocator
```

---

## 31.10 관련 문서

| 문서 | 내용 | 관계 |
|------|------|------|
| [08. Memory Management](08_memory_management.md) | GPU/CPU 버퍼 전송, copy_slice, 동기화 | Layer 1 상세 |
| [10. Model Inference](10_model_inference.md) | Forward pass, LayerWorkspace 사용 | Layer 2 상세 |
| [11. KV Cache 관리 전략](11_kv_cache_management.md) | Eviction 정책 (SlidingWindow, H2O, D2O) | Layer 4-A |
| [32. KV 캐시 오프로드](32_kv_offload.md) | OffloadKVCache, RawStore, 프리페치 파이프라인 | Layer 4-B |
