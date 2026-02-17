# Chapter 6: OpenCL Backend Implementation

> **이전**: [05. 토크나이저 및 샘플링](05_tokenizer_and_sampling.md) | **다음**: [07. 커널 구현](07_kernel_implementation.md)

## 6.1 Overview

OpenCL 백엔드는 Adreno GPU를 타겟으로 최적화된 GPU 가속 추론 엔진입니다.

### Key Design Decisions
1. **커널 캐싱**: 모든 커널 객체를 `Mutex<KernelCache>`에 저장하여 재사용
2. **비동기 실행**: 불필요한 `queue.finish()` 호출 제거
3. **Fast Math 최적화**: 컴파일러 플래그를 통한 성능 향상

---

## 6.2 OpenCLBackend 구조체

### 파일 위치
`src/backend/opencl/mod.rs`

### 구조 정의
```rust
pub struct OpenCLBackend {
    // OpenCL Context Objects
    pub context: Context,
    pub queue: Queue,
    pub device: Device,
    
    // Compiled Programs
    pub program: Program,              // mul_mv_f32_f32.cl
    pub simple_ops_program: Program,   // simple_ops.cl
    pub q4_0_program: Program,         // mul_mv_q4_0_f32.cl
    pub get_rows_program: Program,     // get_rows.cl
    
    // Cached Kernels (Thread-safe)
    kernels: Mutex<KernelCache>,
}
```

### KernelCache 구조
```rust
struct KernelCache {
    kernel_mul_mat_f32_f32: CoreKernel,    // F32 matmul
    kernel_mul_mat_q4_0_f32: CoreKernel,   // Q4_0 matmul
    kernel_rms_norm_opt: CoreKernel,       // RMS Normalization
    kernel_softmax_opt: CoreKernel,        // Softmax
    kernel_rope_simple: CoreKernel,        // RoPE
    kernel_silu_mul_simple: CoreKernel,    // SiLU * Y
    kernel_add_assign_simple: CoreKernel,  // X += Y
    kernel_scale_simple: CoreKernel,       // X *= scalar
    kernel_get_rows_q4_0: CoreKernel,      // Embedding Q4_0
    kernel_get_rows_f32: CoreKernel,       // Embedding F32
    kernel_attn_gen: CoreKernel,           // GPU Attention
}

// Thread-safety for raw kernel pointers
unsafe impl Send for KernelCache {}
unsafe impl Sync for KernelCache {}
```

---

## 6.3 초기화 흐름 (`new()`)

### Step-by-Step
```
1. Platform::default()
     └─ 시스템의 기본 OpenCL 플랫폼 선택

2. Device::list(platform, DEVICE_TYPE_GPU)
     └─ GPU 디바이스 목록에서 첫 번째 선택

3. Context::builder().platform().devices().build()
     └─ OpenCL 컨텍스트 생성

4. Queue::new(&context, device, None)
     └─ 커맨드 큐 생성 (In-order execution)

5. Program::builder()
     .devices(device)
     .src(kernel_source)
     .cmplr_opt(CL_FAST_MATH_OPTS)
     .build(&context)
     └─ 각 커널 파일 컴파일 (4개)

6. ocl::core::create_kernel(&program, "kernel_name")
     └─ 11개 커널 객체 생성 및 캐싱
```

### 컴파일러 최적화 플래그
```
-cl-std=CL2.0           : OpenCL 2.0 표준
-cl-mad-enable          : a*b+c를 MAD 명령어로 융합
-cl-unsafe-math-optimizations : 부동소수점 최적화 허용
-cl-finite-math-only    : NaN/Inf 검사 생략
-cl-fast-relaxed-math   : 수학 함수 근사치 허용
```

---

## 6.4 커널 디스패치 패턴

### 일반적인 디스패치 흐름
```rust
fn some_operation(&self, ...) -> Result<()> {
    // 1. 버퍼 다운캐스트
    let buf = tensor.buffer()
        .as_any()
        .downcast_ref::<OpenCLBuffer>()
        .ok_or(anyhow!("Not OpenCL buffer"))?;
    
    // 2. 커널 획득 (Mutex lock)
    let kernels = self.kernels.lock()
        .map_err(|e| anyhow!("Lock poisoned: {}", e))?;
    let kernel = &kernels.kernel_xxx;
    
    // 3. 인자 설정
    unsafe {
        ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(buf.as_core()))?;
        ocl::core::set_kernel_arg(kernel, 1, ArgVal::scalar(&param))?;
        // ... more args
    }
    
    // 4. 실행
    unsafe {
        ocl::core::enqueue_kernel(
            &self.queue,
            kernel,
            1,                    // work_dim
            None,                 // global_work_offset
            &[global_size],       // global_work_size
            Some([local_size]),   // local_work_size
            None::<&Event>,       // wait_events
            None::<&mut Event>,   // completion_event
        )?;
    }
    
    // 5. 락 자동 해제 (함수 종료 시)
    Ok(())
}
```

---

## 6.5 워크그룹 크기 설정

### Adreno GPU 최적화
```
Local Size = 64  (Adreno subgroup size)
```

### 커널별 워크 크기
| Kernel | Global Size | Local Size | Notes |
|--------|------------|------------|-------|
| `rms_norm_opt` | rows * 64 | 64 | 1 workgroup per row |
| `softmax_opt` | rows * 64 | 64 | 1 workgroup per row |
| `rope_simple` | seq*heads*(dim/2) | - | No local memory |
| `attn_gen` | num_heads * 64 | 64 | 1 workgroup per head |
| `mul_mat_q4_0` | ((N+3)/4)*64, M, 1 | 64,1,1 | 3D dispatch |
| `mul_mat_f32` | N*64, (M+3)/4, 1 | 64,1,1 | 3D dispatch |

---

## 6.6 메모리 관리

### copy_from() 구현
```rust
fn copy_from(&self, src: &Tensor) -> Result<Tensor> {
    // 1. 새 버퍼 할당
    let buffer = memory.alloc(size, dtype)?;
    
    // 2. KernelCache 복제 (락 획득 → 클론 → 락 해제)
    let kernel_cache = {
        let src_kernels = self.kernels.lock()?;
        KernelCache {
            kernel_xxx: src_kernels.kernel_xxx.clone(),
            // ... all 11 kernels
        }
    };
    
    // 3. 새 백엔드 인스턴스 생성
    let new_backend = OpenCLBackend {
        context: self.context.clone(),
        kernels: Mutex::new(kernel_cache),
        // ...
    };
    
    // 4. 복사 수행
    if src is OpenCLBuffer {
        // Device-to-Device: enqueue_copy_buffer
        // 비동기 - queue.finish() 호출 안 함
    } else {
        // Host-to-Device: enqueue_write_buffer (blocking)
    }
    
    Ok(Tensor::new(shape, buffer, Arc::new(new_backend)))
}
```

---

## 6.7 Backend Trait 구현

### 필수 메서드
```rust
impl Backend for OpenCLBackend {
    fn name(&self) -> &str { "OpenCL" }
    fn device(&self) -> &str { "GPU" }
    
    fn matmul(&self, a, b, out) -> Result<()>;
    fn matmul_transposed(&self, a, b, out) -> Result<()>;
    fn rms_norm(&self, x, weight, eps) -> Result<()>;
    fn rope_inplace(&self, x, start_pos, theta) -> Result<()>;
    fn softmax(&self, x) -> Result<()>;
    fn silu_mul(&self, x, y) -> Result<()>;
    fn add_assign(&self, x, y) -> Result<()>;
    fn scale(&self, x, val) -> Result<()>;
    fn gather(&self, src, indices, dst) -> Result<()>;
    fn copy_from(&self, src) -> Result<Tensor>;
    fn copy_slice(&self, src, dst, ...) -> Result<()>;
    fn attention_gen(&self, q, k, v, out, ...) -> Result<()>;
    
    fn read_buffer(&self, t, dst) -> Result<()>;
    fn synchronize(&self) -> Result<()>;
}
```

---

## 6.8 에러 처리

### 일반적인 에러 케이스
```rust
// 버퍼 타입 불일치
.ok_or(anyhow!("X is not OpenCL buffer"))?

// 락 포이즈닝
.map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?

// OpenCL 에러
ocl::core::enqueue_kernel(...)?  // 자동으로 ocl::Error 전파
```

### 커널 컴파일 실패 대응
```rust
let program = match Program::builder()...build() {
    Ok(p) => p,
    Err(e) => {
        eprintln!("WARN: Failed to compile kernel: {}", e);
        // Dummy 커널로 폴백
        Program::builder()
            .src("__kernel void kernel_xxx() {}")
            .build(&context)?
    }
};
```
