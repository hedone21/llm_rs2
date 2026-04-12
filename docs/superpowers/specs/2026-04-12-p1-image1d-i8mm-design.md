# P1 성능 최적화: GPU image1d_buffer_t + CPU i8mm

## 개요

Q4_0 decode 성능을 llama.cpp 수준에 근접시키기 위한 두 P1 작업의 설계.

- **GPU P1**: gemv_noshuffle 커널의 global buffer → image1d_buffer_t 전환 (텍스처 캐시 활용)
- **CPU P1**: ARM i8mm (`smmla`) 기반 2-row dot product 추가

두 작업은 독립적이며 하나의 설계로 관리하되, 구현은 GPU → CPU 순서로 진행한다.

## 현재 성능 기준선 (Llama 3.2 1B, Adreno 830)

| 경로 | llm.rs | llama.cpp | 달성률 |
|------|--------|-----------|--------|
| GPU Q4 decode | 33.3 tok/s | 45.2 tok/s | 74% |
| CPU Q4 decode | 13.6 tok/s | 36.0 tok/s | 38% |

---

## GPU P1: image1d_buffer_t 전환

### 배경

현재 `gemv_noshuffle_q4_0.cl`은 llama.cpp 원본의 `image1d_buffer_t` 접근을 `global uint*` / `global float4*`로 대체한 MVP 상태. global buffer 읽기는 L1/L2 캐시만 사용하지만, `image1d_buffer_t`는 GPU의 텍스처 캐시(TP, Texture Processor)를 활용하여 대역폭을 크게 향상시킨다. MVP에서 +2%에 그친 이유가 이 차이.

### 커널 변경 (`gemv_noshuffle_q4_0.cl`)

시그니처 변경:

```cl
// 변경 전 (MVP)
__kernel void kernel_gemv_noshuffle_q4_0(
    global uint   * src0_q,
    global half2  * src0_d,
    global float4 * src1,
    global float  * dst,
    int ne00, int ne01)

// 변경 후
__kernel void kernel_gemv_noshuffle_q4_0(
    __read_only image1d_buffer_t src0_q,  // R32UI
    global half2  * src0_d,               // global 유지
    __read_only image1d_buffer_t src1,    // RGBA32F
    global float  * dst,
    int ne00, int ne01)
```

데이터 접근 변경:
- `src0_q[offset]` → `read_imageui(src0_q, offset).x`
- `src1[offset]` → `read_imagef(src1, offset)`
- `src0_d[offset]` → 변경 없음 (global half2*)

dequantize 매크로, reduction 로직, dispatch 크기는 모두 동일 유지. 컴파일타임 상수 (`LINE_STRIDE_A`, `BLOCK_STRIDE_A`, `SIMDGROUP_WIDTH`)도 기존과 동일하게 `-D` define으로 전달.

### Rust 측 변경 (`backend/opencl/mod.rs`)

#### NoshuffleSoaEntry 확장

```rust
pub struct NoshuffleSoaEntry {
    pub q_buf: Mem,    // SOA nibbles buffer (기존)
    pub d_buf: Mem,    // SOA scales buffer (기존)
    pub q_img: Mem,    // image1d_buffer_t wrapping q_buf (신규)
    pub ne00: usize,
    pub ne01: usize,
}
```

#### Image 생성

| 대상 | 포맷 | width | buffer | 생성 시점 |
|------|------|-------|--------|----------|
| weight (`src0_q`) | `CL_R, CL_UNSIGNED_INT32` | 총 uint 개수 | `q_buf` | `convert_q4_0_to_noshuffle()` 시 1회 |
| activation (`src1`) | `CL_RGBA, CL_FLOAT` | `ne00 / 4` | `src1_buf` | `matmul_q4_0_noshuffle()` 매 호출 |

`ocl-core` API 사용:

```rust
use ocl::core::{ImageDescriptor, ImageFormat, MemObjectType, ImageChannelOrder, ImageChannelDataType};

let img_fmt = ImageFormat::new(
    ImageChannelOrder::R,
    ImageChannelDataType::UnsignedInt32,
);
let img_desc = ImageDescriptor::new(
    MemObjectType::Image1dBuffer,
    width, 0, 0, 0, 0, 0,
    Some(q_buf.clone()),
);
let q_img = unsafe {
    ocl::core::create_image(&context, MemFlags::READ_ONLY, &img_fmt, &img_desc, None::<&[u32]>, None)?
};
```

#### Dispatch 변경

```rust
// matmul_q4_0_noshuffle() 내 커널 arg 설정
set_kernel_arg(kernel, 0, ArgVal::mem(&entry.q_img))?;  // image
set_kernel_arg(kernel, 1, ArgVal::mem(&entry.d_buf))?;  // global
set_kernel_arg(kernel, 2, ArgVal::mem(&act_img))?;       // image
set_kernel_arg(kernel, 3, ArgVal::mem(dst_buf))?;        // global
```

### 리스크 및 대응

- **`CL_DEVICE_IMAGE_MAX_BUFFER_SIZE` 제한**: weight 버퍼가 image width 상한을 초과할 수 있음. 초과 시 해당 weight은 noshuffle 비활성화 (기존 matmul_q4_0 fallback).
- **Activation image 매 호출 생성 오버헤드**: `clCreateImage`은 lightweight wrapper (메모리 복사 없음). 프로파일 후 필요시 캐시 추가.

---

## CPU P1: i8mm 2-row dot product

### 배경

ARM i8mm 확장 (`FEAT_I8MM`, Cortex-A78+ / Cortex-X2+)의 `smmla` (signed 8-bit matrix multiply-accumulate) 명령어는 2x8x2 정수 행렬곱을 1사이클에 수행. 현재 `sdot` 경로는 1-row 단위로 처리하지만, i8mm는 2 weight row를 동시에 처리하여 throughput을 2배로 올린다.

### 새 함수: `vec_dot_q4_0_q8_0_i8mm`

```rust
/// 2-row i8mm dot product.
/// weight 2행 × activation 1행 → 결과 2개 동시 산출.
pub unsafe fn vec_dot_q4_0_q8_0_i8mm(
    n: usize,
    s0: &mut f32,
    s1: &mut f32,
    vx0: *const BlockQ4_0,
    vx1: *const BlockQ4_0,
    vy: *const BlockQ8_0,
)
```

연산 패턴 (llama.cpp `__ARM_FEATURE_MATMUL_INT8` 경로 기반):

```
for each block i:
  x0_l, x0_h = unpack_q4(weight_row0[i])  // low/high nibbles → int8
  x1_l, x1_h = unpack_q4(weight_row1[i])
  y_l, y_h   = load_q8(activation[i])

  // Interleave for vmmlaq (2x8 × 2x8 → 2x2)
  l0 = vzip1_s64(x0_l, x1_l)    // [x0_low8, x1_low8]
  l1 = vzip2_s64(x0_l, x1_l)    // [x0_high8, x1_high8]
  l2 = vzip1_s64(x0_h, x1_h)
  l3 = vzip2_s64(x0_h, x1_h)

  r0 = vzip1_s64(y_l, y_l)      // activation 복제
  r1 = vzip2_s64(y_l, y_l)
  r2 = vzip1_s64(y_h, y_h)
  r3 = vzip2_s64(y_h, y_h)

  // smmla: acc[0]=dot(x0,y), acc[2]=dot(x1,y) (유효 결과)
  acc = smmla(smmla(smmla(smmla(zero, l0, r0), l1, r1), l2, r2), l3, r3)

  scale = [d_x0*d_y, _, d_x1*d_y, _]
  sumv = vmlaq_f32(sumv, vcvtq_f32_s32(acc), scale)

s0 = sumv[0] + sumv[1]  // 실제로는 sumv[0]만 유효 (scale[1]=0 or 무시)
s1 = sumv[2] + sumv[3]
```

인라인 어셈블리 (`smmla`):

```rust
std::arch::asm!(
    "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
    acc = inout(vreg) acc,
    a = in(vreg) l0,
    b = in(vreg) r0,
);
```

### matmul 루프 변경 (`matmul_transposed_q4_0`)

```rust
if std::arch::is_aarch64_feature_detected!("i8mm") {
    // 2-row 단위 루프
    let mut j = 0;
    while j + 1 < n {
        vec_dot_q4_0_q8_0_i8mm(k, &mut s0, &mut s1,
            weight_ptr.add(j * nb_k), weight_ptr.add((j+1) * nb_k), act_ptr);
        out[i*n + j]   = s0;
        out[i*n + j+1] = s1;
        j += 2;
    }
    if j < n {  // 홀수 나머지 → sdot fallback
        vec_dot_q4_0_q8_0_sdot(k, &mut sum, weight_ptr.add(j * nb_k), act_ptr);
        out[i*n + j] = sum;
    }
} else if std::arch::is_aarch64_feature_detected!("dotprod") {
    // 기존 sdot 1-row 경로
} else {
    // 기본 경로
}
```

병렬화: 기존 `par_chunks_mut` 구조 유지. chunk 내에서 2-row 단위 처리.

### 리스크 및 대응

- **signed/unsigned 시맨틱**: Q4_0 nibble은 unsigned (0~15), `-8` 오프셋으로 signed (-8~7)로 변환. `smmla`는 signed 연산이므로 변환 후 사용하면 정확. 단위 테스트로 검증.
- **호스트 테스트 제약**: macOS Apple Silicon (M1/M2/M3)은 i8mm 미지원. 테스트는 `cfg` + 런타임 감지로 조건부 실행, Android 디바이스에서 실제 검증.

---

## 테스트 전략 (TDD)

### GPU P1 테스트

| 테스트 | 내용 | 실행 환경 |
|--------|------|----------|
| `test_image1d_buffer_creation` | image1d_buffer_t 생성 성공 + 포맷 검증 | 호스트 (macOS OpenCL) |
| `test_noshuffle_q4_0_correctness` (기존 확장) | image 경로 수치 정확성, CPU reference 대비 | Adreno 디바이스 |
| `test_noshuffle_image_max_buffer_size` | width 상한 초과 시 graceful fallback | 호스트 + 디바이스 |

### CPU P1 테스트

| 테스트 | 내용 | 실행 환경 |
|--------|------|----------|
| `test_i8mm_dot_basic` | 2-row dot product vs f64 reference (n=32,64,128) | i8mm 지원 디바이스 |
| `test_i8mm_dot_large` | 대형 차원 (n=2048, 4096) 정확성 | i8mm 지원 디바이스 |
| `test_i8mm_matmul_even_n` | 짝수 n에서 2-row 루프 정확성 | i8mm 지원 디바이스 |
| `test_i8mm_matmul_odd_n` | 홀수 n에서 나머지 처리 정확성 | i8mm 지원 디바이스 |
| `test_i8mm_matmul_m_gt_1` | prefill (m>1) 시 정확성 | i8mm 지원 디바이스 |
| `test_i8mm_runtime_detection` | feature 감지 분기 동작 확인 | 모든 환경 |

---

## 구현 순서

1. GPU P1: 테스트 작성 → 커널 + Rust 구현 → 디바이스 검증
2. CPU P1: 테스트 작성 → i8mm 함수 + matmul 루프 → 디바이스 검증

## 성공 기준

| 항목 | GPU P1 | CPU P1 |
|------|--------|--------|
| 정확성 | 기존 noshuffle 테스트 전수 통과 | reference 대비 오차 < 1e-4 |
| 성능 목표 | GPU Q4 decode >= 40 tok/s | CPU Q4 decode >= 18 tok/s |
| llama.cpp 달성률 | >= 88% (현재 74%) | >= 50% (현재 38%) |
| 회귀 없음 | F16/Q8_0 경로 성능 변화 없음 | dotprod 경로 성능 변화 없음 |

## 주요 파일

| 파일 | 변경 |
|------|------|
| `engine/kernels/gemv_noshuffle_q4_0.cl` | global → image1d_buffer_t |
| `engine/src/backend/opencl/mod.rs` | NoshuffleSoaEntry + image 생성 + dispatch |
| `engine/src/backend/cpu/neon.rs` | i8mm dot + matmul 2-row 루프 |

## 참조

- llama.cpp noshuffle 커널: `/tmp/llama-cpp-build/ggml/src/ggml-opencl/kernels/gemv_noshuffle.cl`
- llama.cpp image1d dispatch: `/tmp/llama-cpp-build/ggml/src/ggml-opencl/ggml-opencl.cpp:4710-4750`
- llama.cpp i8mm Q4_0: `/tmp/llama-cpp-build/ggml/src/ggml-cpu/arch/arm/quants.c:261-331`
- ocl-core ImageDescriptor: `ocl-core-0.11.5/src/types/structs.rs:1073`
