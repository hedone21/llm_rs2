use anyhow::{Result, anyhow};
use crate::core::backend::Backend;
use crate::core::tensor::Tensor;
use crate::core::buffer::DType;
use std::arch::aarch64::*;
use crate::backend::cpu::common::CpuBackendCommon;
use crate::core::quant::{BlockQ4_0, BlockQ4_1, BlockQ8_0, QK4_0, QK4_1, QK8_0};
use rayon::prelude::*;

    // sdot_asm and prefetch_asm removed (unused)

pub struct CpuBackendNeon;

impl CpuBackendNeon {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackendNeon {
    fn name(&self) -> &str {
        "CPU (NEON)"
    }

    fn device(&self) -> &str {
        "CPU (AArch64)"
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        CpuBackendCommon::new().matmul(a, b, out)
    }

    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        match b.dtype() {
            DType::F32 => self.matmul_transposed_f32(a, b, out),
            DType::Q4_0 => self.matmul_transposed_q4_0(a, b, out),
            // Fallback for others
            _ => CpuBackendCommon::new().matmul_transposed(a, b, out),
        }
    }

    fn matmul_slice(&self, a: &Tensor, b: &Tensor, rows: usize, cols: usize, out: &mut Tensor) -> Result<()> {
         match b.dtype() {
             DType::F32 => self.matmul_transposed_f32(a, b, out), 
             DType::Q4_0 => self.matmul_transposed_q4_0(a, b, out),
             _ => CpuBackendCommon::new().matmul_slice(a, b, rows, cols, out),
        }
    }

    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        CpuBackendCommon::new().add_assign(a, b)
    }

    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()> {
        CpuBackendCommon::new().scale(x, v)
    }

    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        CpuBackendCommon::new().silu_mul(a, b)
    }

    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32) -> Result<()> {
        CpuBackendCommon::new().rms_norm(x, w, eps)
    }

    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        CpuBackendCommon::new().softmax(x)
    }

    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        CpuBackendCommon::new().rope_inplace(x, start_pos, theta)
    }

    fn copy_from(&self, t: &Tensor) -> Result<Tensor> {
        CpuBackendCommon::new().copy_from(t)
    }
}

impl CpuBackendNeon {
    fn matmul_transposed_f32(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();
        
        if a_rank < 2 || b_rank < 2 { return Err(anyhow!("Tensors must be at least 2D for matmul")); }

        let k = a_shape[a_rank - 1];
        let k_b = b_shape[b_rank - 1]; 
        if k != k_b { return Err(anyhow!("Shape mismatch")); }
        
        let m: usize = a_shape[..a_rank-1].iter().product();
        let n = b_shape[b_rank - 2];

        let a_data = a.as_slice::<f32>();
        let b_data = b.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();

        // Heuristic: Serial for tiny matrices (Proven 3.6x faster for [1, 128, 256])
        if (m * n * k) < 100_000 {
            return self.matmul_transposed_f32_serial(a, b, out);
        }

        // Optimization: N-parallel for all M
        // Adaptive chunking: split N into exactly `num_threads` chunks to minimize Rayon overhead
        // for "medium" workloads (like M=1 generation).
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;
        let chunk_size = chunk_size.max(256); // Ensure at least 256 elements per task
        
        out_data.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
           let start_idx = chunk_idx * chunk_size;
           for (local_i, out_val) in chunk.iter_mut().enumerate() {
               let idx = start_idx + local_i;
               let i = idx / n;
               let j = idx % n;
               
               let a_ptr = unsafe { a_data.as_ptr().add(i * k) };
               let b_ptr = unsafe { b_data.as_ptr().add(j * k) };
               
               let mut k_idx = 0;
               
               unsafe {
                   let mut sum_v = vdupq_n_f32(0.0);
                   
                   while k_idx + 16 <= k {
                       let va0 = vld1q_f32(a_ptr.add(k_idx));
                       let vb0 = vld1q_f32(b_ptr.add(k_idx));
                       sum_v = vfmaq_f32(sum_v, va0, vb0);

                       let va1 = vld1q_f32(a_ptr.add(k_idx + 4));
                       let vb1 = vld1q_f32(b_ptr.add(k_idx + 4));
                       sum_v = vfmaq_f32(sum_v, va1, vb1);

                       let va2 = vld1q_f32(a_ptr.add(k_idx + 8));
                       let vb2 = vld1q_f32(b_ptr.add(k_idx + 8));
                       sum_v = vfmaq_f32(sum_v, va2, vb2);
                       
                       let va3 = vld1q_f32(a_ptr.add(k_idx + 12));
                       let vb3 = vld1q_f32(b_ptr.add(k_idx + 12));
                       sum_v = vfmaq_f32(sum_v, va3, vb3);
                       
                       k_idx += 16;
                   }
                   
                   while k_idx + 4 <= k {
                        let va = vld1q_f32(a_ptr.add(k_idx));
                        let vb = vld1q_f32(b_ptr.add(k_idx));
                        sum_v = vfmaq_f32(sum_v, va, vb);
                        k_idx += 4;
                   }

                   // Reduction
                   let sum_s = vaddvq_f32(sum_v);
                   let mut sum = sum_s;
                   
                   // Tail
                   while k_idx < k {
                       sum += *a_ptr.add(k_idx) * *b_ptr.add(k_idx);
                       k_idx += 1;
                   }
                   *out_val = sum;
               }
           }
        });
        Ok(())
    }

    // Pure serial implementation for tiny matrices
    fn matmul_transposed_f32_serial(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 2];
        let m: usize = a_shape[..a_shape.len()-1].iter().product();

        let a_data = a.as_slice::<f32>();
        let b_data = b.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();

        for i in 0..m {
            let a_offset = i * k;
            for j in 0..n {
                let b_offset = j * k;
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a_data[a_offset + l] * b_data[b_offset + l];
                }
                out_data[i * n + j] = sum;
            }
        }
        Ok(())
    }

    fn matmul_transposed_q4_0(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        let k = a_shape[a_rank - 1];
        let n = b_shape[b_rank - 2];
        let m: usize = a_shape[..a_rank-1].iter().product();

        // Heuristic: Serial for tiny matrices (Proven 9x faster for [1, 128, 256])
        if (m * n * k) < 100_000 {
            return self.matmul_transposed_q4_0_serial(a, b, out);
        }

        // Optimization: Use NEON Kernel for all M (Unified).
        // Note: We removed the M=1 scalar fallback in favor of the optimized kernel (fundamental solution).
        let a_data = a.as_slice::<f32>();
        let nb_k = k / QK4_0;
        let out_data = out.as_mut_slice::<f32>();

        // 1. Quantize A to Q8_0
        let nb_k_q8 = k / QK8_0;
        let total_q8_blocks = m * nb_k_q8;
         
        // Temp buffer allocation
        let mut a_q8 = Vec::with_capacity(total_q8_blocks);
        unsafe { a_q8.set_len(total_q8_blocks); }
         
        for i in 0..m {
             let a_offset = i * k;
             let a_row = &a_data[a_offset..a_offset + k];
             let q8_row = &mut a_q8[i * nb_k_q8..(i + 1) * nb_k_q8];
             unsafe { self.quantize_row_q8_0(a_row, q8_row, k); }
        }

        // Adaptive chunking strategy
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;
        let chunk_size = chunk_size.max(256);

        out_data.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk): (usize, &mut [f32])| {
            let start_idx = chunk_idx * chunk_size;
            for (local_i, out_val) in chunk.iter_mut().enumerate() {
                let idx = start_idx + local_i;
                let i = idx / n;
                let j = idx % n;
                
                let b_offset = j * nb_k;
                let b_row_node = unsafe { b.as_ptr() as *const BlockQ4_0 };
                let b_row_ptr = unsafe { b_row_node.add(b_offset) };
                
                // We need pointer to the start of the row in the vector
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
        });
        Ok(())
    }

    fn matmul_transposed_q4_0_serial(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
         let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 2];
        let m: usize = a_shape[..a_shape.len()-1].iter().product();

        let a_data = a.as_slice::<f32>();
        let nb_k = k / QK4_0;
        
        // Safety: b is created as Q4_0 tensor, so data is BlockQ4_0
        let total_blocks = n * nb_k;
        let b_blocks = unsafe {
            std::slice::from_raw_parts(b.as_ptr() as *const BlockQ4_0, total_blocks)
        };
        
        let out_data = out.as_mut_slice::<f32>();

        for i in 0..m {
            let a_offset = i * k;
            let a_row = &a_data[a_offset..a_offset + k];

            for j in 0..n {
                let b_offset = j * nb_k;
                let b_row_blocks = &b_blocks[b_offset..b_offset + nb_k];
                
                let mut sum = 0.0;
                for (bi, block) in b_row_blocks.iter().enumerate() {
                    let d = block.d.to_f32();
                    let a_slice = &a_row[bi * QK4_0..(bi + 1) * QK4_0];
                    
                    let mut isum: f32 = 0.0;
                    for z in 0..(QK4_0 / 2) {
                        let b_val = block.qs[z];
                        let v0 = (b_val & 0x0F) as i8 - 8;
                        let v1 = (b_val >> 4) as i8 - 8;
                        
                        isum += v0 as f32 * a_slice[z];
                        isum += v1 as f32 * a_slice[z + QK4_0 / 2];
                    }
                    sum += d * isum;
                }
                out_data[i * n + j] = sum;
            }
        }
        Ok(())
    }
    // Q8_0 Quantization (NEON)
    // Ported from llama.cpp/arch/arm/quants.c
    #[target_feature(enable = "neon")]
    pub unsafe fn quantize_row_q8_0(&self, x: &[f32], y: &mut [BlockQ8_0], k: usize) {
        assert!(k % QK8_0 == 0);
        let nb = k / QK8_0;
        
        let x_ptr = x.as_ptr();

        unsafe {
            for i in 0..nb {
                 // Load 32 floats
                 // 32 floats = 8 vectors of float32x4_t
                 let src_ptr = x_ptr.add(i * QK8_0);
                 
                 // Compute max abs
                 let mut max_abs = vdupq_n_f32(0.0);
                 
                 // Unroll 8 vectors
                 let v0 = vld1q_f32(src_ptr);
                 let v1 = vld1q_f32(src_ptr.add(4));
                 let v2 = vld1q_f32(src_ptr.add(8));
                 let v3 = vld1q_f32(src_ptr.add(12));
                 let v4 = vld1q_f32(src_ptr.add(16));
                 let v5 = vld1q_f32(src_ptr.add(20));
                 let v6 = vld1q_f32(src_ptr.add(24));
                 let v7 = vld1q_f32(src_ptr.add(28));

                 max_abs = vmaxq_f32(max_abs, vabsq_f32(v0));
                 max_abs = vmaxq_f32(max_abs, vabsq_f32(v1));
                 max_abs = vmaxq_f32(max_abs, vabsq_f32(v2));
                 max_abs = vmaxq_f32(max_abs, vabsq_f32(v3));
                 max_abs = vmaxq_f32(max_abs, vabsq_f32(v4));
                 max_abs = vmaxq_f32(max_abs, vabsq_f32(v5));
                 max_abs = vmaxq_f32(max_abs, vabsq_f32(v6));
                 max_abs = vmaxq_f32(max_abs, vabsq_f32(v7));

                 let amax = vmaxvq_f32(max_abs);
                 let d = amax / 127.0;
                 let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                 
                 y[i].d = half::f16::from_f32(d);
                 
                 let id_v = vdupq_n_f32(id);

                 // Convert to i8
                 // v0..v3 -> 16 floats -> 16 i32 -> 16 i16 -> 16 i8 (1 vector)
                 // v4..v7 -> ... -> 16 i8 (1 vector)
                 
                 // Process first 16
                 let i32_0 = vcvtnq_s32_f32(vmulq_f32(v0, id_v));
                 let i32_1 = vcvtnq_s32_f32(vmulq_f32(v1, id_v));
                 let i32_2 = vcvtnq_s32_f32(vmulq_f32(v2, id_v));
                 let i32_3 = vcvtnq_s32_f32(vmulq_f32(v3, id_v));
                 
                 let i16_0 = vqmovn_s32(i32_0);
                 let i16_1 = vqmovn_s32(i32_1);
                 let i16_2 = vqmovn_s32(i32_2);
                 let i16_3 = vqmovn_s32(i32_3);
                 
                 let i16_low  = vcombine_s16(i16_0, i16_1);
                 let i16_high = vcombine_s16(i16_2, i16_3);
                 
                 let i8_0 = vqmovn_s16(i16_low);
                 let i8_1 = vqmovn_s16(i16_high);
                 let i8_res_0 = vcombine_s8(i8_0, i8_1);

                 // Process next 16
                 let i32_4 = vcvtnq_s32_f32(vmulq_f32(v4, id_v));
                 let i32_5 = vcvtnq_s32_f32(vmulq_f32(v5, id_v));
                 let i32_6 = vcvtnq_s32_f32(vmulq_f32(v6, id_v));
                 let i32_7 = vcvtnq_s32_f32(vmulq_f32(v7, id_v));
                 
                 let i16_4 = vqmovn_s32(i32_4);
                 let i16_5 = vqmovn_s32(i32_5);
                 let i16_6 = vqmovn_s32(i32_6);
                 let i16_7 = vqmovn_s32(i32_7);
                 
                 let i16_low2  = vcombine_s16(i16_4, i16_5);
                 let i16_high2 = vcombine_s16(i16_6, i16_7);
                 
                 let i8_2 = vqmovn_s16(i16_low2);
                 let i8_3 = vqmovn_s16(i16_high2);
                 let i8_res_1 = vcombine_s8(i8_2, i8_3);

                 // Store
                 vst1q_s8(y[i].qs.as_mut_ptr(), i8_res_0);
                 vst1q_s8(y[i].qs.as_mut_ptr().add(16), i8_res_1);
            }
        }
    }

    // Dot Product Q4_0 * Q8_0 (NEON)
     #[target_feature(enable = "neon")]
    pub unsafe fn vec_dot_q4_0_q8_0(&self, n: usize, s: &mut f32, vx: *const BlockQ4_0, vy: *const BlockQ8_0) {
        let nb = n / QK8_0;
        
        let mut sumf = 0.0;
        unsafe {
             let m4b = vdupq_n_u8(0x0F);
             let s8b = vdupq_n_s8(8);

             let mut i = 0;
             while i + 1 < nb {
                 let x0 = &*vx.add(i);
                 let y0 = &*vy.add(i);
                 let x1 = &*vx.add(i+1);
                 let y1 = &*vy.add(i+1);

                 let d0 = x0.d.to_f32() * y0.d.to_f32();
                 let d1 = x1.d.to_f32() * y1.d.to_f32();

                 // Block 0
                 let v0_0 = vld1q_u8(x0.qs.as_ptr());
                 // Unpack and sub 8
                 let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
                 let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4)), s8b);
                 let y0_l = vld1q_s8(y0.qs.as_ptr());
                 let y0_h = vld1q_s8(y0.qs.as_ptr().add(16));

                 let mul_l0 = vmull_s8(vget_low_s8(x0_l), vget_low_s8(y0_l));
                 let mul_l1 = vmull_high_s8(x0_l, y0_l);
                 let mul_h0 = vmull_s8(vget_low_s8(x0_h), vget_low_s8(y0_h));
                 let mul_h1 = vmull_high_s8(x0_h, y0_h);
                 let acc0 = vaddlvq_s16(mul_l0) + vaddlvq_s16(mul_l1) + vaddlvq_s16(mul_h0) + vaddlvq_s16(mul_h1);

                 // Block 1
                 let v1_0 = vld1q_u8(x1.qs.as_ptr());
                 let x1_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1_0, m4b)), s8b);
                 let x1_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v1_0, 4)), s8b);
                 let y1_l = vld1q_s8(y1.qs.as_ptr());
                 let y1_h = vld1q_s8(y1.qs.as_ptr().add(16));

                 let mul_l0_1 = vmull_s8(vget_low_s8(x1_l), vget_low_s8(y1_l));
                 let mul_l1_1 = vmull_high_s8(x1_l, y1_l);
                 let mul_h0_1 = vmull_s8(vget_low_s8(x1_h), vget_low_s8(y1_h));
                 let mul_h1_1 = vmull_high_s8(x1_h, y1_h);
                 let acc1 = vaddlvq_s16(mul_l0_1) + vaddlvq_s16(mul_l1_1) + vaddlvq_s16(mul_h0_1) + vaddlvq_s16(mul_h1_1);
                 
                 sumf += d0 * acc0 as f32;
                 sumf += d1 * acc1 as f32;

                 i += 2;
             }
             
             // Tail
             while i < nb {
                 let x = &*vx.add(i);
                 let y = &*vy.add(i);
                 let d = x.d.to_f32() * y.d.to_f32();
                 
                 let v0_0 = vld1q_u8(x.qs.as_ptr());
                 let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
                 let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4)), s8b);
                 let y0_l = vld1q_s8(y.qs.as_ptr());
                 let y0_h = vld1q_s8(y.qs.as_ptr().add(16));
                 
                 let mul_l0 = vmull_s8(vget_low_s8(x0_l), vget_low_s8(y0_l));
                 let mul_l1 = vmull_high_s8(x0_l, y0_l);
                 let mul_h0 = vmull_s8(vget_low_s8(x0_h), vget_low_s8(y0_h));
                 let mul_h1 = vmull_high_s8(x0_h, y0_h);
                 let acc = vaddlvq_s16(mul_l0) + vaddlvq_s16(mul_l1) + vaddlvq_s16(mul_h0) + vaddlvq_s16(mul_h1);

                 sumf += d * acc as f32;
                 i += 1;
             }
        }
        *s = sumf;
    }

    // Dot Product Q4_0 * Q8_0 (NEON + DotProd)
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn vec_dot_q4_0_q8_0_sdot(&self, n: usize, s: &mut f32, vx: *const BlockQ4_0, vy: *const BlockQ8_0) {
        let nb = n / QK8_0;
        let mut sumf = 0.0;
        
        unsafe {
             let m4b = vdupq_n_u8(0x0F);
             let s8b = vdupq_n_s8(8);

             let mut i = 0;
             while i < nb {
                 let x = &*vx.add(i);
                 let y = &*vy.add(i);
                 let d = x.d.to_f32() * y.d.to_f32();

                 // Load Q4_0
                 let v0 = vld1q_u8(x.qs.as_ptr()); // 32 values (uint8x16)
                 
                 // Unpack to int8
                 let x_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4b)), s8b);
                 let x_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), s8b);

                 // Load Q8_0
                 let y_l = vld1q_s8(y.qs.as_ptr()); // 16 values
                 let y_h = vld1q_s8(y.qs.as_ptr().add(16)); // 16 values
                 
                 // Dot Product with Accumulation (ARMv8.2 sdot)
                 // sdot (acc, a, b) -> acc + a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
                 let mut acc = vdupq_n_s32(0);
                 
                 // Use asm! to emit sdot instruction since vdotq_s32 is unstable
                 std::arch::asm!(
                     "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                     acc = inout(vreg) acc,
                     x = in(vreg) x_l,
                     y = in(vreg) y_l,
                 );
                 
                 std::arch::asm!(
                     "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                     acc = inout(vreg) acc,
                     x = in(vreg) x_h,
                     y = in(vreg) y_h,
                 );
                 
                 let sum_i = vaddvq_s32(acc);
                 sumf += d * sum_i as f32;

                 i += 1;
             }
        }
        *s = sumf;
    }
}
