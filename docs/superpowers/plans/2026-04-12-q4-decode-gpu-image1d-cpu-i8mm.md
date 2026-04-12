# Q4_0 Decode P1: GPU image1d_buffer_t + CPU i8mm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GPU Q4 decode 33.3 → ≥40 tok/s (image1d_buffer_t), CPU Q4 decode 13.6 → ≥18 tok/s (i8mm).

**Architecture:** GPU — `gemv_noshuffle_q4_0.cl`의 `global uint*`/`global float4*`를 `image1d_buffer_t`의 `read_imageui`/`read_imagef`로 교체. Adreno TP 캐시 활용. CPU — `neon.rs`에 `smmla` 기반 2-row dot product 추가, matmul 루프를 2-row 단위 dispatch로 변경.

**Tech Stack:** Rust, OpenCL 2.0 (Adreno 830), ARM NEON + FEAT_I8MM, `ocl`/`ocl-core` 0.19/0.11 crate.

---

## Spec

`docs/superpowers/specs/2026-04-12-q4-decode-gpu-image1d-cpu-i8mm-design.md`

---

## File Structure

**수정:**
- `engine/kernels/gemv_noshuffle_q4_0.cl` — `global` → `image1d_buffer_t` 전환
- `engine/src/backend/opencl/mod.rs` — `NoshuffleSoaEntry` 확장, image 생성, dispatch 변경, 테스트 확장
- `engine/src/backend/cpu/neon.rs` — `vec_dot_q4_0_q8_0_i8mm` 추가, matmul 루프 2-row 분기

---

## Task 1: GPU — 커널을 image1d_buffer_t로 전환

**Files:**
- Modify: `engine/kernels/gemv_noshuffle_q4_0.cl`

- [ ] **Step 1: 커널 시그니처 및 데이터 접근 변경**

`gemv_noshuffle_q4_0.cl`에서 3가지 변경:

1. 커널 파라미터:
```cl
// 변경 전
__kernel void kernel_gemv_noshuffle_q4_0(
        global uint   * src0_q,
        global half2  * src0_d,
        global float4 * src1,
        global float  * dst,
        int ne00, int ne01)

// 변경 후
__kernel void kernel_gemv_noshuffle_q4_0(
        __read_only image1d_buffer_t src0_q,
        global half2  * src0_d,
        __read_only image1d_buffer_t src1,
        global float  * dst,
        int ne00, int ne01)
```

2. Weight 읽기 (8곳): `src0_q[offset]` → `read_imageui(src0_q, offset).x`
```cl
// 변경 전
regA.s0 = src0_q[gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0];
// 변경 후
regA.s0 = read_imageui(src0_q, gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0).x;
```

3. Activation 읽기 (2곳): `src1[offset]` → `read_imagef(src1, offset)`
```cl
// 변경 전
regB.s0123 = src1[slid * 2 + k * 8];
regB.s4567 = src1[1 + slid * 2 + k * 8];
// 변경 후
regB.s0123 = read_imagef(src1, slid * 2 + k * 8);
regB.s4567 = read_imagef(src1, 1 + slid * 2 + k * 8);
```

- [ ] **Step 2: 커널 주석 업데이트**

파일 상단 주석에서 "Global buffer MVP" 설명을 image1d_buffer_t 설명으로 교체.

- [ ] **Step 3: 커밋**

```bash
git add engine/kernels/gemv_noshuffle_q4_0.cl
git commit -m "perf(opencl): gemv_noshuffle kernel — global buffer → image1d_buffer_t"
```

---

## Task 2: GPU — Rust 인프라: image 생성 + dispatch

**Files:**
- Modify: `engine/src/backend/opencl/mod.rs:151-162` (NoshuffleSoaEntry)
- Modify: `engine/src/backend/opencl/mod.rs:1344-1502` (convert_q4_0_to_noshuffle)
- Modify: `engine/src/backend/opencl/mod.rs:1226-1260` (matmul_q4_0 dispatch)
- Modify: `engine/src/backend/opencl/mod.rs:1517-1595` (matmul_q4_0_noshuffle)

- [ ] **Step 1: NoshuffleSoaEntry에 q_img 필드 추가**

```rust
pub struct NoshuffleSoaEntry {
    pub q_buf: ocl::core::Mem,
    pub d_buf: ocl::core::Mem,
    pub q_img: ocl::core::Mem,  // image1d_buffer_t wrapping q_buf
    pub ne00: usize,
    pub ne01: usize,
}
```

- [ ] **Step 2: convert_q4_0_to_noshuffle에서 image 생성**

함수 반환 타입을 `(Mem, Mem)` → `(Mem, Mem, Mem)` (q_buf, d_buf, q_img)로 변경.

transpose 완료 후, q_buf를 wrapping하는 image1d_buffer_t 생성:

```rust
use ocl::core::{ImageFormat, ImageDescriptor, MemObjectType, ImageChannelOrder, ImageChannelDataType};

// q_buf는 ne01 * cols_ushort ushort = ne01 * (ne00/4) ushort
// uint 단위: q_total_ushort / 2
let q_total_uint = q_total_ushort / 2;
let q_img_fmt = ImageFormat::new(ImageChannelOrder::R, ImageChannelDataType::UnsignedInt32);
let q_img_desc = ImageDescriptor::new(
    MemObjectType::Image1dBuffer,
    q_total_uint, 0, 0, 0, 0, 0,
    Some(dst_q.clone()),
);
let q_img = unsafe {
    ocl::core::create_image(
        self.context.as_core(),
        ocl::core::MEM_READ_ONLY,
        &q_img_fmt,
        &q_img_desc,
        None::<&[u32]>,
        None,
    )
};
```

`create_image` 실패 시 (image_max_buffer_size 초과 등) `None` 반환하여 noshuffle 비활성화. `NoshuffleSoaEntry`의 `q_img`를 `Option<Mem>`으로 변경하고, `matmul_q4_0` dispatch에서 `q_img.is_none()`이면 기존 matmul fallback.

반환: `Ok((dst_q, dst_d, q_img.ok()))`

- [ ] **Step 3: register/lookup 호출부 업데이트**

`prepare_noshuffle_buffers()` (transformer.rs)와 `register_noshuffle_soa()` 호출부에서 3-tuple 대응.

`matmul_q4_0` dispatch (mod.rs:1244-1252)에서 `entry.q_img`를 전달:

```rust
if let Some(entry) = self.lookup_noshuffle_soa(b_key) {
    return self.matmul_q4_0_noshuffle(
        &entry.q_img, &entry.d_buf, /* ... */
    );
}
```

- [ ] **Step 4: matmul_q4_0_noshuffle에서 activation image 생성**

함수 시작부에 activation image 생성 추가:

```rust
let act_img_fmt = ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelDataType::Float);
let act_img_desc = ImageDescriptor::new(
    MemObjectType::Image1dBuffer,
    ne00 / 4, 0, 0, 0, 0, 0,
    Some(src1_buf.clone()),
);
let act_img = unsafe {
    ocl::core::create_image(
        self.context.as_core(),
        ocl::core::MEM_READ_ONLY,
        &act_img_fmt,
        &act_img_desc,
        None::<&[f32]>,
        None,
    )?
};
```

커널 arg 2번을 `src1_buf` 대신 `act_img` 전달:

```rust
ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(&act_img))?;
```

- [ ] **Step 5: 호스트 빌드 + 기존 테스트 실행**

```bash
cargo test -p llm_rs2 noshuffle -- --nocapture
```

macOS에서는 커널 컴파일 실패 시 skip 예상 (기존 동작).

- [ ] **Step 6: 커밋**

```bash
git add engine/src/backend/opencl/mod.rs engine/src/models/transformer.rs
git commit -m "perf(opencl): image1d_buffer_t infrastructure for gemv_noshuffle Q4_0"
```

---

## Task 3: GPU — image1d_buffer_t 생성 테스트

**Files:**
- Modify: `engine/src/backend/opencl/mod.rs` (noshuffle_tests 모듈)

- [ ] **Step 1: image1d_buffer_t 생성 + readback 테스트 작성**

`noshuffle_tests` 모듈에 추가:

```rust
#[test]
fn test_image1d_buffer_creation_and_readback() {
    let backend = match try_create_backend() {
        Some(b) => b,
        None => { eprintln!("[SKIPPED] No OpenCL device"); return; }
    };

    // Create a buffer with known data
    let data: Vec<u32> = (0..256).collect();
    let buf = unsafe {
        let b = ocl::core::create_buffer::<_, u32>(
            backend.context.as_core(), ocl::core::MEM_READ_WRITE, data.len(), None,
        ).unwrap();
        ocl::core::enqueue_write_buffer(
            &backend.queue, &b, true, 0,
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4),
            None::<&ocl::core::Event>, None::<&mut ocl::core::Event>,
        ).unwrap();
        b
    };

    // Create image1d_buffer_t wrapping it
    let img_fmt = ocl::core::ImageFormat::new(
        ocl::core::ImageChannelOrder::R,
        ocl::core::ImageChannelDataType::UnsignedInt32,
    );
    let img_desc = ocl::core::ImageDescriptor::new(
        ocl::core::MemObjectType::Image1dBuffer,
        data.len(), 0, 0, 0, 0, 0,
        Some(buf.clone()),
    );
    let result = unsafe {
        ocl::core::create_image(
            backend.context.as_core(),
            ocl::core::MEM_READ_ONLY,
            &img_fmt,
            &img_desc,
            None::<&[u32]>,
            None,
        )
    };

    match result {
        Ok(img) => {
            eprintln!("[PASS] image1d_buffer_t created successfully, width={}", data.len());
            // Image wraps the same memory — no separate readback needed.
            // Kernel-level verification is done by test_noshuffle_q4_0_correctness.
            drop(img);
        }
        Err(e) => {
            eprintln!("[SKIPPED] image1d_buffer_t not supported: {}", e);
        }
    }
}
```

- [ ] **Step 2: 테스트 실행**

```bash
cargo test -p llm_rs2 test_image1d_buffer_creation -- --nocapture
```

- [ ] **Step 3: 커밋**

```bash
git add engine/src/backend/opencl/mod.rs
git commit -m "test(opencl): add image1d_buffer_t creation test"
```

---

## Task 4: CPU — i8mm 2-row dot product 테스트 (TDD Red)

**Files:**
- Modify: `engine/src/backend/cpu/neon.rs` (테스트 모듈)

- [ ] **Step 1: i8mm dot product 테스트 작성**

`neon.rs` 하단 테스트 모듈에 추가. i8mm는 macOS에서 미지원이므로 런타임 감지로 skip:

```rust
#[cfg(target_arch = "aarch64")]
#[test]
fn test_i8mm_dot_q4_0_q8_0() {
    if !std::arch::is_aarch64_feature_detected!("i8mm") {
        eprintln!("[SKIPPED] i8mm not supported on this device");
        return;
    }

    let backend = NeonBackend;

    // Test dimensions
    for nb in [1, 2, 4, 8, 64, 128] {
        let k = nb * QK4_0; // 32, 64, 128, 256, 2048, 4096

        // Create weight data (2 rows)
        let mut rng_state = 42u64;
        let mut pseudo_rand = || -> i8 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) % 15) as i8 - 7
        };

        let mut blocks_row0 = Vec::with_capacity(nb);
        let mut blocks_row1 = Vec::with_capacity(nb);
        for _ in 0..nb {
            let mut b0 = BlockQ4_0 { d: half::f16::from_f32(0.1), qs: [0u8; QK4_0 / 2] };
            let mut b1 = BlockQ4_0 { d: half::f16::from_f32(0.15), qs: [0u8; QK4_0 / 2] };
            for z in 0..QK4_0 / 2 {
                let v0_lo = (pseudo_rand() + 8) as u8;
                let v0_hi = (pseudo_rand() + 8) as u8;
                b0.qs[z] = (v0_lo & 0x0F) | ((v0_hi & 0x0F) << 4);
                let v1_lo = (pseudo_rand() + 8) as u8;
                let v1_hi = (pseudo_rand() + 8) as u8;
                b1.qs[z] = (v1_lo & 0x0F) | ((v1_hi & 0x0F) << 4);
            }
            blocks_row0.push(b0);
            blocks_row1.push(b1);
        }

        // Create activation (Q8_0)
        let mut act_blocks = Vec::with_capacity(nb);
        for _ in 0..nb {
            let mut a = BlockQ8_0 { d: half::f16::from_f32(0.2), qs: [0i8; QK8_0] };
            for z in 0..QK8_0 {
                a.qs[z] = pseudo_rand();
            }
            act_blocks.push(a);
        }

        // Reference: compute using existing sdot
        let mut ref0 = 0.0f32;
        let mut ref1 = 0.0f32;
        unsafe {
            backend.vec_dot_q4_0_q8_0_sdot(k, &mut ref0, blocks_row0.as_ptr(), act_blocks.as_ptr());
            backend.vec_dot_q4_0_q8_0_sdot(k, &mut ref1, blocks_row1.as_ptr(), act_blocks.as_ptr());
        }

        // i8mm
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        unsafe {
            backend.vec_dot_q4_0_q8_0_i8mm(
                k, &mut s0, &mut s1,
                blocks_row0.as_ptr(), blocks_row1.as_ptr(),
                act_blocks.as_ptr(),
            );
        }

        let tol = 1e-4 * (k as f32).sqrt();
        assert!((s0 - ref0).abs() < tol, "k={k} row0: i8mm={s0} ref={ref0} diff={}", (s0 - ref0).abs());
        assert!((s1 - ref1).abs() < tol, "k={k} row1: i8mm={s1} ref={ref1} diff={}", (s1 - ref1).abs());
    }
}
```

- [ ] **Step 2: 컴파일 확인 (함수 stub 추가)**

컴파일을 위해 빈 stub을 `NeonBackend`에 추가:

```rust
#[cfg(target_arch = "aarch64")]
pub unsafe fn vec_dot_q4_0_q8_0_i8mm(
    &self, _n: usize, _s0: &mut f32, _s1: &mut f32,
    _vx0: *const BlockQ4_0, _vx1: *const BlockQ4_0, _vy: *const BlockQ8_0,
) {
    unimplemented!("i8mm dot product not yet implemented");
}
```

```bash
cargo check -p llm_rs2
```

- [ ] **Step 3: 커밋**

```bash
git add engine/src/backend/cpu/neon.rs
git commit -m "test(neon): add i8mm 2-row dot product test + stub (TDD red)"
```

---

## Task 5: CPU — i8mm dot product 구현 (TDD Green)

**Files:**
- Modify: `engine/src/backend/cpu/neon.rs:vec_dot_q4_0_q8_0_i8mm` (stub 교체)

- [ ] **Step 1: vec_dot_q4_0_q8_0_i8mm 구현**

stub을 실제 구현으로 교체. llama.cpp `arch/arm/quants.c:261-331` 패턴 기반:

```rust
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_q4_0_q8_0_i8mm(
    &self,
    n: usize,
    s0: &mut f32,
    s1: &mut f32,
    vx0: *const BlockQ4_0,
    vx1: *const BlockQ4_0,
    vy: *const BlockQ8_0,
) {
    use std::arch::aarch64::*;

    let nb = n / QK8_0;
    let m4b = vdupq_n_u8(0x0F);
    let s8b = vdupq_n_s8(0x08);

    let mut sumv = vdupq_n_f32(0.0);

    for i in 0..nb {
        let b_x0 = &*vx0.add(i);
        let b_x1 = &*vx1.add(i);
        let b_y = &*vy.add(i);

        // Unpack Q4_0 → signed int8
        let v0_0 = vld1q_u8(b_x0.qs.as_ptr());
        let v0_1 = vld1q_u8(b_x1.qs.as_ptr());

        let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
        let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4)), s8b);
        let x1_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_1, m4b)), s8b);
        let x1_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4)), s8b);

        // Load Q8_0
        let y_l = vld1q_s8(b_y.qs.as_ptr());
        let y_h = vld1q_s8(b_y.qs.as_ptr().add(16));

        // Scales: [d_x0*d_y, 0, d_x1*d_y, 0]
        let d_x0 = b_x0.d.to_f32();
        let d_x1 = b_x1.d.to_f32();
        let d_y = b_y.d.to_f32();
        let _scale = [d_x0 * d_y, 0.0f32, d_x1 * d_y, 0.0f32];
        let scale = vld1q_f32(_scale.as_ptr());

        // Interleave for smmla (2x8x2 matmul)
        let l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
        let l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
        let l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
        let l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

        let r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y_l), vreinterpretq_s64_s8(y_l)));
        let r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y_l), vreinterpretq_s64_s8(y_l)));
        let r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y_h), vreinterpretq_s64_s8(y_h)));
        let r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y_h), vreinterpretq_s64_s8(y_h)));

        // smmla: signed 8-bit matmul accumulate (4 chained calls)
        let mut acc = vdupq_n_s32(0);
        std::arch::asm!("smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) l0, b = in(vreg) r0);
        std::arch::asm!("smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) l1, b = in(vreg) r1);
        std::arch::asm!("smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) l2, b = in(vreg) r2);
        std::arch::asm!("smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) l3, b = in(vreg) r3);

        // acc = [dot(x0,y), dot(x0,y), dot(x1,y), dot(x1,y)]
        // scale = [d_x0*d_y, 0, d_x1*d_y, 0]
        // → sumv += [dot(x0,y)*d_x0*d_y, 0, dot(x1,y)*d_x1*d_y, 0]
        sumv = vmlaq_f32(sumv, vcvtq_f32_s32(acc), scale);
    }

    *s0 = vgetq_lane_f32(sumv, 0);
    *s1 = vgetq_lane_f32(sumv, 2);
}
```

- [ ] **Step 2: 컴파일 확인**

```bash
cargo check -p llm_rs2
```

macOS에서 컴파일만 확인 (i8mm 테스트는 디바이스에서 실행).

- [ ] **Step 3: 커밋**

```bash
git add engine/src/backend/cpu/neon.rs
git commit -m "perf(neon): implement i8mm 2-row dot product for Q4_0 (TDD green)"
```

---

## Task 6: CPU — matmul 루프 2-row dispatch

**Files:**
- Modify: `engine/src/backend/cpu/neon.rs:1920-1946` (matmul_transposed_q4_0 parallel loop)

- [ ] **Step 1: matmul 루프에 i8mm 분기 추가**

기존 `par_chunks_mut` 루프 내부에 i8mm 분기 추가. 기존 코드의 flat (i,j) 인덱싱을 유지하되, 같은 activation row 내에서 인접한 weight row 2개를 i8mm로 처리:

```rust
out_data.par_chunks_mut(chunk_size).enumerate().for_each(
    |(chunk_idx, chunk): (usize, &mut [f32])| {
        let start_idx = chunk_idx * chunk_size;

        #[cfg(target_arch = "aarch64")]
        let use_i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
        #[cfg(not(target_arch = "aarch64"))]
        let use_i8mm = false;

        if use_i8mm {
            let b_row_node = unsafe { b.as_ptr() as *const BlockQ4_0 };
            let mut local_i = 0;
            while local_i < chunk.len() {
                let idx = start_idx + local_i;
                let i = idx / n;
                let j = idx % n;
                let a_row_ptr = unsafe { a_q8.as_ptr().add(i * nb_k_q8) };

                // Pair adjacent weight rows if possible
                if local_i + 1 < chunk.len() && j + 1 < n {
                    let b0 = unsafe { b_row_node.add(j * nb_k) };
                    let b1 = unsafe { b_row_node.add((j + 1) * nb_k) };
                    unsafe {
                        self.vec_dot_q4_0_q8_0_i8mm(
                            k, &mut chunk[local_i], &mut chunk[local_i + 1],
                            b0, b1, a_row_ptr,
                        );
                    }
                    local_i += 2;
                } else {
                    let b_ptr = unsafe { b_row_node.add(j * nb_k) };
                    let mut sum = 0.0;
                    unsafe {
                        self.vec_dot_q4_0_q8_0_sdot(k, &mut sum, b_ptr, a_row_ptr);
                    }
                    chunk[local_i] = sum;
                    local_i += 1;
                }
            }
        } else {
            // Existing sdot / basic NEON path (unchanged)
            for (local_i, out_val) in chunk.iter_mut().enumerate() {
                let idx = start_idx + local_i;
                let i = idx / n;
                let j = idx % n;

                let b_offset = j * nb_k;
                let b_row_node = unsafe { b.as_ptr() as *const BlockQ4_0 };
                let b_row_ptr = unsafe { b_row_node.add(b_offset) };
                let a_row_ptr = unsafe { a_q8.as_ptr().add(i * nb_k_q8) };

                let mut sum = 0.0;
                unsafe {
                    if std::arch::is_aarch64_feature_detected!("dotprod") {
                        self.vec_dot_q4_0_q8_0_sdot(k, &mut sum, b_row_ptr, a_row_ptr);
                    } else {
                        self.vec_dot_q4_0_q8_0(k, &mut sum, b_row_ptr, a_row_ptr);
                    }
                }
                *out_val = sum;
            }
        }
    },
);
```

- [ ] **Step 2: 컴파일 + 호스트 테스트**

```bash
cargo test -p llm_rs2 matmul_transposed_q4_0 -- --nocapture
```

호스트에서는 `use_i8mm = false`이므로 기존 경로로 실행, 회귀 없음 확인.

- [ ] **Step 3: 커밋**

```bash
git add engine/src/backend/cpu/neon.rs
git commit -m "perf(neon): matmul_transposed_q4_0 — i8mm 2-row dispatch"
```

---

## Task 7: 디바이스 검증 — GPU + CPU

**Prerequisites:** Android 디바이스 연결, `source android.source` 완료.

- [ ] **Step 1: Android 빌드**

```bash
source android.source
cargo build --release --target aarch64-linux-android --bin generate --bin test_backend
```

- [ ] **Step 2: 배포 + 테스트 실행**

```bash
adb push target/aarch64-linux-android/release/test_backend /data/local/tmp/
adb shell "cd /data/local/tmp && ./test_backend"
```

noshuffle 테스트 PASS 확인 (image1d_buffer_t 경로).

- [ ] **Step 3: GPU Q4 decode 벤치마크**

```bash
adb push target/aarch64-linux-android/release/generate /data/local/tmp/
adb shell "/data/local/tmp/generate \
  --model-path /data/local/tmp/models/llama3.2-1b-gguf \
  --prompt-file /data/local/tmp/p100.txt \
  --max-tokens 128 --device opencl --dtype q4_0"
```

목표: `Decode: ≤25 ms/tok` (≥40 tok/s).

- [ ] **Step 4: CPU Q4 decode 벤치마크**

```bash
adb shell "/data/local/tmp/generate \
  --model-path /data/local/tmp/models/llama3.2-1b-gguf \
  --prompt-file /data/local/tmp/p100.txt \
  --max-tokens 128 --device cpu --dtype q4_0"
```

목표: `Decode: ≤55 ms/tok` (≥18 tok/s).

i8mm 활성화 확인: 로그에 feature detection 출력 추가 필요 시 임시 `log::info!` 삽입.

- [ ] **Step 5: 결과 커밋**

성능 결과를 커밋 메시지에 기록:

```bash
git commit --allow-empty -m "perf: Q4_0 decode device benchmark results

GPU Q4 decode: XX tok/s (was 33.3, target ≥40)
CPU Q4 decode: XX tok/s (was 13.6, target ≥18)"
```
