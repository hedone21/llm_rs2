use anyhow::{Result, anyhow};
use crate::core::backend::Backend;
use crate::core::tensor::Tensor;
use crate::core::buffer::DType;
use rayon::prelude::*;
use crate::core::quant::{BlockQ4_0, BlockQ4_1, BlockQ8_0, QK4_0, QK4_1, QK8_0};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use crate::backend::cpu::common::CpuBackendCommon;

pub struct CpuBackendAVX2;

impl CpuBackendAVX2 {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackendAVX2 {
    fn name(&self) -> &str {
        "CPU (AVX2)"
    }

    fn device(&self) -> &str {
        "CPU"
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        // Fallback or SIMD logic for standard matmul?
        // Let's rely on common for now as matmul is less critical than matmul_transposed for inference
        // Or specific AVX impl if needed.
        CpuBackendCommon::new().matmul(a, b, out)
    }

    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        match b.dtype() {
            DType::F32 => self.matmul_transposed_f32(a, b, out),
            DType::Q4_0 => self.matmul_transposed_q4_0(a, b, out),
            DType::Q4_1 => self.matmul_transposed_q4_1(a, b, out),
            _ => Err(anyhow!("Unsupported dtype for matmul_transposed: {:?}", b.dtype())),
        }
    }

    fn matmul_slice(&self, a: &Tensor, b: &Tensor, rows: usize, cols: usize, out: &mut Tensor) -> Result<()> {
         match b.dtype() {
             DType::F32 => self.matmul_transposed_f32(a, b, out), 
             DType::Q4_0 => self.matmul_transposed_q4_0(a, b, out),
             DType::Q4_1 => self.matmul_transposed_q4_1(a, b, out),
             _ => Err(anyhow!("Unsupported dtype for matmul_slice: {:?}", b.dtype())),
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

impl CpuBackendAVX2 {
    #[cfg(not(target_arch = "x86_64"))]
    fn matmul_transposed_f32(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        CpuBackendCommon::new().matmul_transposed_f32(a, b, out)
    }

    #[cfg(target_arch = "x86_64")]
    fn matmul_transposed_f32(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        if !is_x86_feature_detected!("avx2") {
            return CpuBackendCommon::new().matmul_transposed_f32(a, b, out);
        }

        unsafe {
            let a_shape = a.shape().dims();
            let b_shape = b.shape().dims();
            let a_rank = a_shape.len();
            let b_rank = b_shape.len();
            
            let k = a_shape[a_rank - 1];
            // Flatten M dimensions
            let m: usize = a_shape[..a_rank-1].iter().product();
            let n = b_shape[b_rank - 2];

            let a_data = a.as_slice::<f32>();
            let b_data = b.as_slice::<f32>();
            let out_data = out.as_mut_slice::<f32>();

            // Heuristic: For very small matrices, threading overhead outweighs AVX gains.
            // Fallback to scalar/common implementation which might be serial or lighter weight.
            // Threshold tuned to 100k.
             if (m * n * k) < 100_000 {
                return self.matmul_transposed_f32_serial(a, b, out);
            }

            // Optimization for small M (e.g. generation M=1..8)
            // Parallelize over N instead of M to use all cores.
            // Use small chunks (64) to avoid false sharing while maintaining good load balance (64 tasks for N=4096).
            if m < 8 {
                 out_data.par_iter_mut().enumerate().for_each(|(idx, out_val)| {
                    let i = idx / n;
                    let j = idx % n;
                    
                    let a_ptr = unsafe { a_data.as_ptr().add(i * k) };
                    let b_ptr = unsafe { b_data.as_ptr().add(j * k) };
                    
                    let mut k_idx = 0;
                    
                    unsafe {
                        let mut sum_v = _mm256_setzero_ps();
                        
                        while k_idx + 32 <= k {
                            let va0 = _mm256_loadu_ps(a_ptr.add(k_idx));
                            let vb0 = _mm256_loadu_ps(b_ptr.add(k_idx));
                            sum_v = _mm256_fmadd_ps(va0, vb0, sum_v);

                            let va1 = _mm256_loadu_ps(a_ptr.add(k_idx + 8));
                            let vb1 = _mm256_loadu_ps(b_ptr.add(k_idx + 8));
                            sum_v = _mm256_fmadd_ps(va1, vb1, sum_v);

                            let va2 = _mm256_loadu_ps(a_ptr.add(k_idx + 16));
                            let vb2 = _mm256_loadu_ps(b_ptr.add(k_idx + 16));
                            sum_v = _mm256_fmadd_ps(va2, vb2, sum_v);

                            let va3 = _mm256_loadu_ps(a_ptr.add(k_idx + 24));
                            let vb3 = _mm256_loadu_ps(b_ptr.add(k_idx + 24));
                            sum_v = _mm256_fmadd_ps(va3, vb3, sum_v);
                            
                            k_idx += 32;
                        }
                        
                        while k_idx + 8 <= k {
                             let va = _mm256_loadu_ps(a_ptr.add(k_idx));
                             let vb = _mm256_loadu_ps(b_ptr.add(k_idx));
                             sum_v = _mm256_fmadd_ps(va, vb, sum_v);
                             k_idx += 8;
                        }

                        let mut temp = [0.0f32; 8];
                        _mm256_storeu_ps(temp.as_mut_ptr(), sum_v);
                        let mut sum = temp.iter().sum::<f32>();
                        
                        // Tail
                        while k_idx < k {
                            sum += *a_ptr.add(k_idx) * *b_ptr.add(k_idx);
                            k_idx += 1;
                        }
                        
                        *out_val = sum;
                    }
                 });
                 return Ok(());
            }
            
            // For M in [8, 64), Blocked AVX2 (Block=8) reduces parallelism too much (e.g. M=8 -> 1 task).
            // Scalar (Row-parallel) is better.
            if m < 64 {
                 return CpuBackendCommon::new().matmul_transposed_f32(a, b, out);
            }

            // Block Size M=8 (Process 8 rows of A at once to reuse B)
            // This reduces B matrix bandwidth by 8x.
            const BLOCK_M: usize = 8;
            
            out_data.par_chunks_mut(n * BLOCK_M).enumerate().for_each(|(chunk_idx, out_chunk)| {
                let start_m = chunk_idx * BLOCK_M;
                let rows_in_chunk = std::cmp::min(BLOCK_M, m - start_m);

                // Prepare pointers for A rows
                let mut a_ptrs = [std::ptr::null::<f32>(); BLOCK_M];
                for r in 0..rows_in_chunk {
                     a_ptrs[r] = a_data.as_ptr().add((start_m + r) * k);
                }

                // Iterate over B rows (N)
                for j in 0..n {
                    let b_ptr = b_data.as_ptr().add(j * k);
                    
                    // Initialize accumulators
                    let mut sums = [_mm256_setzero_ps(); BLOCK_M];

                    let mut k_idx = 0;
                    // Main AVX Loop
                    while k_idx + 8 <= k {
                        let vb = _mm256_loadu_ps(b_ptr.add(k_idx));
                        
                        // FMA for each active row in block
                        // We unroll the loop over rows manually for the compiler or use a loop (since const)
                        for r in 0..rows_in_chunk {
                            let va = _mm256_loadu_ps(a_ptrs[r].add(k_idx));
                            sums[r] = _mm256_fmadd_ps(va, vb, sums[r]);
                        }
                        k_idx += 8;
                    }

                    // Reduce sums and handle tail
                    for r in 0..rows_in_chunk {
                         // Horizontal sum of AVX register
                        let mut temp = [0.0f32; 8];
                        _mm256_storeu_ps(temp.as_mut_ptr(), sums[r]);
                        let mut final_sum = temp.iter().sum::<f32>();
                        
                        // Scalar tail
                        let ar_ptr = a_ptrs[r];
                        let mut tail_k = k_idx;
                        while tail_k < k {
                             final_sum += *ar_ptr.add(tail_k) * *b_ptr.add(tail_k);
                             tail_k += 1;
                        }

                        // Write to output
                        // out_chunk is flattened [BLOCK_M * N]
                        // r-th row of chunk, j-th col
                        out_chunk[r * n + j] = final_sum;
                    }
                }
            });
            Ok(())
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn matmul_transposed_q4_0(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        CpuBackendCommon::new().matmul_transposed_q4_0(a, b, out)
    }

    #[cfg(target_arch = "x86_64")]

    // -------------------------------------------------------------------------
    // AVX2 Kernel Helpers (Ported from llama.cpp/arch/x86/quants.c)
    // -------------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    unsafe fn sum_i16_pairs_float(&self, x: __m256i) -> __m256 {
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let summed_pairs = _mm256_madd_epi16(ones, x);
            _mm256_cvtepi32_ps(summed_pairs)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mul_sum_us8_pairs_float(&self, ax: __m256i, sy: __m256i) -> __m256 {
        unsafe {
            // Perform multiplication and create 16-bit values
            let dot = _mm256_maddubs_epi16(ax, sy);
            self.sum_i16_pairs_float(dot)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mul_sum_i8_pairs_float(&self, x: __m256i, y: __m256i) -> __m256 {
        unsafe {
            // Get absolute values of x vectors
            let ax = _mm256_sign_epi8(x, x);
            // Sign the values of the y vectors
            let sy = _mm256_sign_epi8(y, x);
            self.mul_sum_us8_pairs_float(ax, sy)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn bytes_from_nibbles_32(&self, rsi: *const u8) -> __m256i {
        unsafe {
            // Load 16 bytes from memory
            let tmp = _mm_loadu_si128(rsi as *const __m128i);
            // Expand to 32 bytes: [low nibbles | high nibbles]
            // _mm256_set_m128i(hi, lo) -> inserts lo at 0, hi at 1.
            // We want (tmp >> 4) and tmp.
            let bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp); 
            let low_mask = _mm256_set1_epi8(0x0F);
            _mm256_and_si256(low_mask, bytes)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn hsum_float_8(&self, x: __m256) -> f32 {
        unsafe {
            let mut res = _mm256_extractf128_ps(x, 1);
            res = _mm_add_ps(res, _mm256_castps256_ps128(x));
            res = _mm_add_ps(res, _mm_movehl_ps(res, res));
            res = _mm_add_ss(res, _mm_movehdup_ps(res));
            _mm_cvtss_f32(res)
        }
    }

    // -------------------------------------------------------------------------
    // Core Kernel
    // -------------------------------------------------------------------------

    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn vec_dot_q4_0_q8_0(&self, n: usize, s: &mut f32, vx: *const BlockQ4_0, vy: *const BlockQ8_0) {
        let nb = n / QK8_0; 
        
        unsafe {
            let mut acc = _mm256_setzero_ps();
            
            // Offset to shift values from [0..15] to [-8..7]
            let off = _mm256_set1_epi8(8);

            for i in 0..nb {
                let x = &*vx.add(i);
                let y = &*vy.add(i);

                // Compute combined scale
                let d_val = x.d.to_f32() * y.d.to_f32();
                let d = _mm256_set1_ps(d_val);

                // Unpack Q4_0 nibbles -> bytes
                let mut qx = self.bytes_from_nibbles_32(x.qs.as_ptr());
                // Shift range
                qx = _mm256_sub_epi8(qx, off);

                // Load Q8_0 bytes
                let qy = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);

                // Dot product (i8*i8 accumulation)
                let q = self.mul_sum_i8_pairs_float(qx, qy);

                // Accumulate with scale
                acc = _mm256_fmadd_ps(d, q, acc);
            }

            *s = self.hsum_float_8(acc);
        }
    }

    // -------------------------------------------------------------------------
    // AVX2 Quantization Kernel
    // -------------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    pub unsafe fn quantize_row_q8_0(&self, x: &[f32], y: &mut [BlockQ8_0], k: usize) {
        assert!(k % QK8_0 == 0);
        let nb = k / QK8_0;
        
        let x_ptr = x.as_ptr();
        unsafe {
            // Offset for permutation to fix byte ordering after packing
            let perm_idx = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            let sign_bit = _mm256_set1_ps(-0.0);

            for i in 0..nb {
                let src = x_ptr.add(i * QK8_0);
                
                // Load 32 floats (4x AVX registers)
                let mut v0 = _mm256_loadu_ps(src);
                let mut v1 = _mm256_loadu_ps(src.add(8));
                let mut v2 = _mm256_loadu_ps(src.add(16));
                let mut v3 = _mm256_loadu_ps(src.add(24));

                // Compute max(abs(v))
                let mut max_abs = _mm256_andnot_ps(sign_bit, v0);
                max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v1));
                max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v2));
                max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v3));

                // Reduce max_abs (8 floats -> 1 float)
                let max4 = _mm_max_ps(_mm256_extractf128_ps(max_abs, 1), _mm256_castps256_ps128(max_abs));
                let max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
                let max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
                let max_scalar = _mm_cvtss_f32(max4);

                // Scale
                let d = max_scalar / 127.0;
                let id = if max_scalar != 0.0 { 127.0 / max_scalar } else { 0.0 };
                
                y[i].d = half::f16::from_f32(d);

                let mul = _mm256_set1_ps(id);

                // Apply multiplier
                v0 = _mm256_mul_ps(v0, mul);
                v1 = _mm256_mul_ps(v1, mul);
                v2 = _mm256_mul_ps(v2, mul);
                v3 = _mm256_mul_ps(v3, mul);

                // Round to nearest integer
                v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST as i32);
                v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST as i32);
                v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST as i32);
                v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST as i32);

                // Convert to i32
                let mut i0 = _mm256_cvtps_epi32(v0);
                let i1 = _mm256_cvtps_epi32(v1);
                let mut i2 = _mm256_cvtps_epi32(v2);
                let i3 = _mm256_cvtps_epi32(v3);

                // Pack i32 -> i16 -> i8
                i0 = _mm256_packs_epi32(i0, i1); 
                i2 = _mm256_packs_epi32(i2, i3);
                i0 = _mm256_packs_epi16(i0, i2);

                // Fix order
                i0 = _mm256_permutevar8x32_epi32(i0, perm_idx);

                // Store
                _mm256_storeu_si256(y[i].qs.as_mut_ptr() as *mut __m256i, i0);
            }
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    fn matmul_transposed_q4_0(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
         if !is_x86_feature_detected!("avx2") {
            return CpuBackendCommon::new().matmul_transposed_q4_0(a, b, out);
        }
        
         let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        let k = a_shape[a_rank - 1];
        let n = b_shape[b_rank - 2];
        let m: usize = a_shape[..a_rank-1].iter().product();

        // Heuristic: For very small matrices, threading overhead outweighs AVX gains.
        // Bumped to 100k.
        if (m * n * k) < 100_000 {
             return CpuBackendCommon::new().matmul_transposed_q4_0(a, b, out);
        }

        let a_data = a.as_slice::<f32>();
        let nb_k = k / QK4_0;
        let total_blocks = n * nb_k;
        
        let out_data = out.as_mut_slice::<f32>();

        // M < 4: Fine-grained (parallelize N) to utilize cores because M is too small.
        // This is the optimized generation path.
        if m < 4 {
             // 1. Quantize A to Q8_0
             let nb_k_q8 = k / QK8_0;
             let total_q8_blocks = m * nb_k_q8;
             
             // Temp buffer allocation (naive vec for now)
             let mut a_q8 = vec![BlockQ8_0 { d: half::f16::from_f32(0.0), qs: [0; QK8_0] }; total_q8_blocks];
             
             // We can quantize rows in parallel depending on m, but m is small here.
             // Just serial loop over m is fine, or par_iter if m >= 2? M < 4. Serial is fine.
             for i in 0..m {
                 let a_offset = i * k;
                 let a_row = &a_data[a_offset..a_offset + k];
                 let q8_row = &mut a_q8[i * nb_k_q8..(i + 1) * nb_k_q8];
                 unsafe { self.quantize_row_q8_0(a_row, q8_row, k); }
             }

             out_data.par_iter_mut().enumerate().for_each(|(idx, out_val)| {
                let i = idx / n;
                let j = idx % n;
                
                let b_offset = j * nb_k;
                let b_row_node = unsafe { b.as_ptr() as *const BlockQ4_0 };
                let b_row_ptr = unsafe { b_row_node.add(b_offset) };
                
                let a_row_ptr = unsafe { a_q8.as_ptr().add(i * nb_k_q8) };

                let mut sum = 0.0;
                unsafe {
                    self.vec_dot_q4_0_q8_0(k, &mut sum, b_row_ptr, a_row_ptr);
                }
                *out_val = sum;
             });
             return Ok(());
        } else {
             // For M >= 4, use common implementation which handles larger matrices well.
             return CpuBackendCommon::new().matmul_transposed_q4_0(a, b, out);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn matmul_transposed_q4_1(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
         CpuBackendCommon::new().matmul_transposed_q4_1(a, b, out)
    }

    // Pure serial implementation for tiny matrices to avoid Rayon overhead
    fn matmul_transposed_f32_serial(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let _a_rank = a_shape.len();
        let b_rank = b_shape.len();
        
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_rank - 2];
        let m: usize = a_shape[..a_shape.len()-1].iter().product();

        let a_data = a.as_slice::<f32>();
        let b_data = b.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();

        // Serial loop
        for i in 0..m {
            let a_offset = i * k;
            for j in 0..n {
                let b_offset = j * k;
                let mut sum = 0.0;
                // Auto-vectorization should handle this reasonably well
                for l in 0..k {
                    sum += a_data[a_offset + l] * b_data[b_offset + l];
                }
                out_data[i * n + j] = sum;
            }
        }
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

    #[cfg(target_arch = "x86_64")]
    fn matmul_transposed_q4_1(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
         if !is_x86_feature_detected!("avx2") {
            return CpuBackendCommon::new().matmul_transposed_q4_1(a, b, out);
        }
        CpuBackendCommon::new().matmul_transposed_q4_1(a, b, out)
    }
}
