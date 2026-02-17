# Chapter 8: Memory Management

> **이전**: [07. 커널 구현](07_kernel_implementation.md) | **다음**: [09. 어텐션 메커니즘](09_attention_mechanism.md)

## 8.1 Overview

이 문서는 프레임워크의 메모리 관리 전략을 설명합니다. CPU 측(Galloc, SharedBuffer)과 GPU 측(OpenCLMemory, OpenCLBuffer)의 할당/전송/동기화를 다룹니다.

### 핵심 개념
- **Galloc + SharedBuffer**: CPU 힙 메모리 할당 및 관리
- **OpenCLBuffer**: GPU 메모리에 할당된 버퍼
- **Zero-copy 지향**: 가능한 한 GPU↔CPU 전송 최소화
- **비동기 복사**: `queue.finish()` 호출 최소화

---

## 8.2 CPU 측 메모리 관리

### Galloc (CPU 메모리 할당자)

**파일**: `src/memory/galloc.rs`

`Galloc`은 CPU용 `Memory` trait 구현체이다. 상태를 갖지 않는(stateless) 단순한 할당자로, 매 호출마다 새 `SharedBuffer`를 생성한다.

```rust
pub struct Galloc;

impl Memory for Galloc {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        let buf = SharedBuffer::new(size, dtype);
        Ok(Arc::new(buf))
    }

    fn used_memory(&self) -> usize {
        0  // 추적 미구현
    }
}
```

메모리 추적이 구현되어 있지 않으므로 `used_memory()`는 항상 0을 반환한다. `Arc`의 레퍼런스 카운팅이 자동으로 해제를 처리하는 "allocate and forget" 방식이다.

### SharedBuffer (CPU 전용 Buffer)

**파일**: `src/buffer/shared_buffer.rs`

`SharedBuffer`는 `Vec<u8>`로 힙 메모리를 관리하는 CPU 전용 `Buffer` 구현체이다.

```rust
pub struct SharedBuffer {
    data: Vec<u8>,
    size: usize,
    dtype: DType,
}
```

Buffer trait 구현:

- **`as_ptr()`**: `self.data.as_ptr()` — Vec 내부 데이터의 읽기 포인터
- **`as_mut_ptr()`**: `self.data.as_ptr() as *mut u8` — `&self`에서 가변 포인터를 반환 (unsafe)
- **`cl_mem()`**: 항상 `None` — GPU 메모리 없음
- **`sync_device()`**: no-op — CPU 메모리는 항상 동기화 상태

### CPU vs GPU 메모리 비교

| 속성 | CPU (Galloc + SharedBuffer) | GPU (OpenCLMemory + OpenCLBuffer) |
|------|---------------------------|----------------------------------|
| 할당 방식 | `Vec<u8>::new()` | `OclBuffer::builder().build()` |
| CPU 포인터 | 직접 접근 가능 | null (또는 zero-copy map) |
| GPU 핸들 | None | `cl_mem()` 반환 |
| 메모리 추적 | 미구현 (항상 0) | 미구현 (항상 0) |
| 동기화 | 불필요 | `queue.finish()` 또는 `read()` |

---

## 8.3 OpenCL Buffer 구조

### 파일 위치
`src/backend/opencl/buffer.rs`

### OpenCLBuffer 구조체
```rust
pub struct OpenCLBuffer {
    pub buffer: OclBuffer<u8>,  // ocl crate의 버퍼
    dtype: DType,
    size: usize,
}
```

### Buffer Trait 구현
```rust
impl Buffer for OpenCLBuffer {
    fn dtype(&self) -> DType { self.dtype }
    fn size(&self) -> usize { self.size }
    
    // CPU 메모리에 직접 접근 불가 (GPU 버퍼)
    fn as_ptr(&self) -> *const u8 { std::ptr::null() }
    fn as_mut_ptr(&mut self) -> *mut u8 { std::ptr::null_mut() }
    
    fn as_any(&self) -> &dyn Any { self }
}
```

---

## 8.4 메모리 할당

### OpenCLMemory 구조체
```rust
pub struct OpenCLMemory {
    context: Context,
    queue: Queue,
}

impl Memory for OpenCLMemory {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        let buffer = OclBuffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(size)
            .build()?;
        
        Ok(Arc::new(OpenCLBuffer {
            buffer,
            dtype,
            size,
        }))
    }
}
```

---

## 8.5 데이터 전송 패턴

### 8.4.1 Host-to-Device (H2D)
```rust
// copy_from() 내에서
if !src_ptr.is_null() {
    let src_slice = std::slice::from_raw_parts(src_ptr, size);
    ocl::core::enqueue_write_buffer(
        &self.queue,
        ocl_buf.buffer.as_core(),
        true,  // blocking write
        0,     // offset
        src_slice,
        None::<&Event>,
        None::<&mut Event>,
    )?;
}
```

### 8.4.2 Device-to-Device (D2D)
```rust
// 최적화: 비동기 복사 (queue.finish() 없음)
ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
    &self.queue,
    src_buf.buffer.as_core(),
    dst_buf.buffer.as_core(),
    0,  // src_offset
    0,  // dst_offset
    src_buf.size(),
    None::<&Event>,
    None::<&mut Event>,
)?;
// queue.finish() 호출하지 않음!
```

### 8.4.3 Device-to-Host (D2H)
```rust
// read_buffer() 구현
fn read_buffer(&self, t: &Tensor, dst: &mut [u8]) -> Result<()> {
    let buf = t.buffer().downcast_ref::<OpenCLBuffer>()?;
    buf.buffer.read(dst).queue(&self.queue).enq()?;
    Ok(())
}
```

---

## 8.6 copy_slice() 상세 구현

### 시그니처
```rust
fn copy_slice(
    &self,
    src: &Tensor, 
    dst: &mut Tensor,
    src_offset: usize,  // element offset
    dst_offset: usize,  // element offset
    count: usize        // element count
) -> Result<()>
```

### 타입 크기 계산
```rust
let type_size = match src.dtype() {
    DType::F32 => 4,
    DType::F16 => 2,
    DType::U8 => 1,
    DType::Q4_0 => 18,  // BlockQ4_0 크기
    _ => bail!("Unsupported dtype"),
};

let src_byte_off = src_offset * type_size;
let dst_byte_off = dst_offset * type_size;
let byte_len = count * type_size;
```

### 복사 케이스 분기
```
┌─────────────────────────────────────────────────┐
│           copy_slice() 분기                      │
├─────────────────────────────────────────────────┤
│ src=OpenCL, dst=OpenCL → enqueue_copy_buffer    │
│ src=CPU,    dst=OpenCL → enqueue_write_buffer   │
│ src=OpenCL, dst=CPU    → enqueue_read_buffer    │
└─────────────────────────────────────────────────┘
```

---

## 8.7 동기화 전략

### 성능 최적화: 지연 동기화
```rust
// ❌ 이전 (매 복사 후 동기화)
ocl::core::enqueue_copy_buffer(...)?;
self.queue.finish()?;  // 성능 저하!

// ✅ 현재 (필요할 때만 동기화)
ocl::core::enqueue_copy_buffer(...)?;
// queue.finish() 호출 안 함
```

### 명시적 동기화가 필요한 경우
```rust
// 1. 결과를 CPU에서 읽어야 할 때
fn read_buffer(&self, t: &Tensor, dst: &mut [u8]) -> Result<()> {
    buf.buffer.read(dst).queue(&self.queue).enq()?;
    // read()가 내부적으로 동기화함
    Ok(())
}

// 2. 벤치마크/프로파일링 시
fn synchronize(&self) -> Result<()> {
    ocl::core::finish(&self.queue)?;
    Ok(())
}
```

---

## 8.8 메모리 레이아웃

### Tensor Shape → Buffer Size
```rust
// F32 Tensor
size = shape.numel() * 4

// Q4_0 Tensor (32개 값 → 18 bytes)
size = (shape.numel() / 32) * 18
```

### KV Cache 메모리 레이아웃
```
K-Cache: [max_seq_len, kv_heads, head_dim]
V-Cache: [max_seq_len, kv_heads, head_dim]

// 메모리 주소 계산
offset(t, h, d) = (t * kv_heads + h) * head_dim + d
```

---

## 8.9 성능 최적화 정리

| 최적화 | 구현 | 효과 |
|--------|------|------|
| 비동기 D2D 복사 | `queue.finish()` 제거 | 5-10% 향상 |
| GPU-only Attention | CPU fallback 제거 | 36% 향상 |
| 버퍼 재사용 | `copy_from`에서 동일 컨텍스트 사용 | 메모리 할당 감소 |

---

## 8.10 메모리 흐름도

### Token Generation 메모리 흐름
```
┌──────────────────────────────────────────────────────────┐
│                     GPU Memory                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Input Token (H2D once)                                  │
│       ↓                                                  │
│  Embedding → hidden [1, dim]                             │
│       ↓                                                  │
│  ┌─────────────────────────────────────┐                │
│  │ Layer 0                              │                │
│  │   RMSNorm ─→ QKV Matmul ─→ RoPE     │                │
│  │       ↓             ↓     ↓         │                │
│  │   KV Cache Update  Attention (GPU!) │                │
│  │       ↓                   ↓         │                │
│  │   Output Proj ─→ Residual           │                │
│  │       ↓                             │                │
│  │   FFN (Gate/Up/Down)                │                │
│  └─────────────────────────────────────┘                │
│       ↓                                                  │
│  ... (Layer 1-15) ...                                   │
│       ↓                                                  │
│  Final RMSNorm → LM Head Matmul                         │
│       ↓                                                  │
│  Logits [1, vocab_size] (D2H for argmax)                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**D2H 발생 지점**: Output logits만 CPU로 전송 (argmax 계산용)
