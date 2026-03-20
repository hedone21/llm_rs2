use crate::backend::cpu::common::CpuBackendCommon;
use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::quant::{BlockQ4_0, BlockQ4_1, BlockQ8_0, QK4_0, QK4_1, QK8_0};
use crate::core::tensor::Tensor;
use crate::core::thread_pool::{self, WorkFn};
use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::arch::aarch64::*;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

/// Runtime toggle: when true, F16 matmul uses Rayon instead of SpinPool.
/// Set via `--use-rayon` CLI flag for A/B benchmarking.
pub static USE_RAYON: AtomicBool = AtomicBool::new(false);

// sdot_asm and prefetch_asm removed (unused)

pub struct CpuBackendNeon;

impl CpuBackendNeon {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackendNeon {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
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
            DType::F16 => self.matmul_transposed_f16(a, b, out),
            DType::Q4_0 => self.matmul_transposed_q4_0(a, b, out),
            _ => CpuBackendCommon::new().matmul_transposed(a, b, out),
        }
    }

    fn matmul_slice(
        &self,
        a: &Tensor,
        b: &Tensor,
        _rows: usize,
        _cols: usize,
        out: &mut Tensor,
    ) -> Result<()> {
        match b.dtype() {
            DType::F32 => self.matmul_transposed_f32(a, b, out),
            DType::F16 => self.matmul_transposed_f16(a, b, out),
            DType::Q4_0 => self.matmul_transposed_q4_0(a, b, out),
            _ => CpuBackendCommon::new().matmul_transposed(a, b, out),
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

    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        CpuBackendCommon::new().cast(src, dst)
    }

    fn attention_gen(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        num_heads_q: usize,
        num_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
    ) -> Result<()> {
        CpuBackendCommon::new().attention_gen(
            q,
            k_cache,
            v_cache,
            out,
            num_heads_q,
            num_heads_kv,
            head_dim,
            cache_seq_len,
        )
    }
}

impl CpuBackendNeon {
    fn matmul_transposed_f32(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        if a_rank < 2 || b_rank < 2 {
            return Err(anyhow!("Tensors must be at least 2D for matmul"));
        }

        let k = a_shape[a_rank - 1];
        let k_b = b_shape[b_rank - 1];
        if k != k_b {
            return Err(anyhow!("Shape mismatch"));
        }

        let m: usize = a_shape[..a_rank - 1].iter().product();
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

        out_data
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
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
        let m: usize = a_shape[..a_shape.len() - 1].iter().product();

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

    /// NEON F16 matmul: A(F32) x B^T(F16) -> Out(F32)
    /// Multi-row GEMV: processes NR=4 output rows simultaneously,
    /// sharing activation loads across rows for better ILP and reduced overhead.
    fn matmul_transposed_f16(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        let k = a_shape[a_rank - 1];
        let n = b_shape[b_rank - 2];
        let m: usize = a_shape[..a_rank - 1].iter().product();

        let a_data = a.as_slice::<f32>();
        let b_data = b.as_slice::<half::f16>();
        let out_data = out.as_mut_slice::<f32>();

        if (m * n * k) < 100_000 {
            return self.matmul_transposed_f16_serial(a_data, b_data, out_data, m, n, k);
        }

        const NR: usize = 4;

        if USE_RAYON.load(AtomicOrdering::Relaxed) {
            // Rayon path: par_chunks_mut for A/B benchmarking vs SpinPool
            let b_ptr_base = b_data.as_ptr() as usize; // usize for Send
            for i in 0..m {
                let a_ptr = unsafe { a_data.as_ptr().add(i * k) } as usize;
                let row_out = &mut out_data[i * n..(i + 1) * n];
                row_out
                    .par_chunks_mut(NR)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let j = chunk_idx * NR;
                        let bp = b_ptr_base as *const u16;
                        let ap = a_ptr as *const f32;
                        if chunk.len() == NR {
                            let b_ptrs = [
                                unsafe { bp.add(j * k) },
                                unsafe { bp.add((j + 1) * k) },
                                unsafe { bp.add((j + 2) * k) },
                                unsafe { bp.add((j + 3) * k) },
                            ];
                            let mut results = [0.0f32; NR];
                            unsafe { Self::vec_dot_f16_f32_4rows(k, ap, b_ptrs, &mut results) };
                            chunk.copy_from_slice(&results);
                        } else {
                            for t in 0..chunk.len() {
                                chunk[t] =
                                    unsafe { Self::vec_dot_f16_f32(k, ap, bp.add((j + t) * k)) };
                            }
                        }
                    });
            }
        } else {
            // SpinPool path (default)
            let pool = thread_pool::get_pool();

            let n_threads = rayon::current_num_threads();
            let target_chunks = n_threads * 8;
            let rows_per_chunk = ((n + target_chunks - 1) / target_chunks + NR - 1) / NR * NR;
            let rows_per_chunk = rows_per_chunk.max(NR);
            let n_chunks = (n + rows_per_chunk - 1) / rows_per_chunk;

            if m > 1 {
                // N-major GEMM: single dispatch, B stays in L1 across M rows.
                // chunk_id = n_chunk_idx * m + m_row — consecutive ids share B rows.
                let total_tasks = n_chunks * m;
                let ctx = F16GemmCtx {
                    a_base: a_data.as_ptr(),
                    b_base: b_data.as_ptr() as *const u16,
                    out_base: out_data.as_mut_ptr(),
                    m,
                    n,
                    k,
                    rows_per_chunk,
                };
                unsafe {
                    pool.dispatch(
                        total_tasks,
                        f16_gemm_chunk,
                        &ctx as *const F16GemmCtx as *const u8,
                    );
                }
            } else {
                // M=1: existing GEMV path
                let ctx = F16GemvCtx {
                    a_ptr: a_data.as_ptr(),
                    b_base: b_data.as_ptr() as *const u16,
                    out_ptr: out_data.as_mut_ptr(),
                    k,
                    n,
                    rows_per_chunk,
                };
                unsafe {
                    pool.dispatch(
                        n_chunks,
                        f16_gemv_chunk,
                        &ctx as *const F16GemvCtx as *const u8,
                    );
                }
            }
        }

        Ok(())
    }

    fn matmul_transposed_f16_serial(
        &self,
        a_data: &[f32],
        b_data: &[half::f16],
        out_data: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        const NR: usize = 4;
        let b_ptr_base = b_data.as_ptr() as *const u16;

        for i in 0..m {
            let a_ptr = unsafe { a_data.as_ptr().add(i * k) };
            let mut j = 0;
            while j + NR <= n {
                let b_ptrs = [
                    unsafe { b_ptr_base.add(j * k) },
                    unsafe { b_ptr_base.add((j + 1) * k) },
                    unsafe { b_ptr_base.add((j + 2) * k) },
                    unsafe { b_ptr_base.add((j + 3) * k) },
                ];
                let mut results = [0.0f32; NR];
                unsafe {
                    Self::vec_dot_f16_f32_4rows(k, a_ptr, b_ptrs, &mut results);
                }
                out_data[i * n + j..i * n + j + NR].copy_from_slice(&results);
                j += NR;
            }
            while j < n {
                let b_ptr = unsafe { (b_data.as_ptr() as *const u16).add(j * k) };
                out_data[i * n + j] = unsafe { Self::vec_dot_f16_f32(k, a_ptr, b_ptr) };
                j += 1;
            }
        }
        Ok(())
    }

    /// NEON multi-row dot product: compute 4 dot products simultaneously.
    /// A(F32)[K] · B0..B3(F16)[K] → 4 output scalars.
    /// Uses 16-element stride with software prefetch for weight data.
    /// 16 accumulators (4 rows × 4 each) for maximum ILP.
    #[target_feature(enable = "neon")]
    unsafe fn vec_dot_f16_f32_4rows(
        k: usize,
        a_ptr: *const f32,
        b_ptrs: [*const u16; 4],
        out: &mut [f32; 4],
    ) {
        // 4 rows × 4 accumulators = 16 registers (leaves 16 for temps)
        let mut s00 = vdupq_n_f32(0.0);
        let mut s01 = vdupq_n_f32(0.0);
        let mut s02 = vdupq_n_f32(0.0);
        let mut s03 = vdupq_n_f32(0.0);
        let mut s10 = vdupq_n_f32(0.0);
        let mut s11 = vdupq_n_f32(0.0);
        let mut s12 = vdupq_n_f32(0.0);
        let mut s13 = vdupq_n_f32(0.0);
        let mut s20 = vdupq_n_f32(0.0);
        let mut s21 = vdupq_n_f32(0.0);
        let mut s22 = vdupq_n_f32(0.0);
        let mut s23 = vdupq_n_f32(0.0);
        let mut s30 = vdupq_n_f32(0.0);
        let mut s31 = vdupq_n_f32(0.0);
        let mut s32 = vdupq_n_f32(0.0);
        let mut s33 = vdupq_n_f32(0.0);

        let mut idx = 0;
        // Prefetch distance: 128 bytes ahead (~2 cache lines = 64 F16 values)
        const PF: usize = 64;

        // Main loop: 16 elements per iteration × 4 rows
        while idx + 16 <= k {
            // Software prefetch weight data for all 4 rows
            std::arch::asm!(
                "prfm pldl1strm, [{b0}]",
                "prfm pldl1strm, [{b1}]",
                "prfm pldl1strm, [{b2}]",
                "prfm pldl1strm, [{b3}]",
                b0 = in(reg) b_ptrs[0].add(idx + PF),
                b1 = in(reg) b_ptrs[1].add(idx + PF),
                b2 = in(reg) b_ptrs[2].add(idx + PF),
                b3 = in(reg) b_ptrs[3].add(idx + PF),
            );

            // Load 16 F32 activations (shared)
            let a0 = vld1q_f32(a_ptr.add(idx));
            let a1 = vld1q_f32(a_ptr.add(idx + 4));
            let a2 = vld1q_f32(a_ptr.add(idx + 8));
            let a3 = vld1q_f32(a_ptr.add(idx + 12));

            // Macro: load 16 F16 → 4×F32, FMA into 4 accumulators
            macro_rules! dot16_row {
                ($bp:expr, $sa:ident, $sb:ident, $sc:ident, $sd:ident) => {
                    let bra: uint16x8_t = vld1q_u16($bp.add(idx));
                    let brb: uint16x8_t = vld1q_u16($bp.add(idx + 8));
                    let c0: float32x4_t;
                    let c1: float32x4_t;
                    let c2: float32x4_t;
                    let c3: float32x4_t;
                    std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) c0, i = in(vreg) bra);
                    std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) c1, i = in(vreg) bra);
                    std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) c2, i = in(vreg) brb);
                    std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) c3, i = in(vreg) brb);
                    $sa = vfmaq_f32($sa, a0, c0);
                    $sb = vfmaq_f32($sb, a1, c1);
                    $sc = vfmaq_f32($sc, a2, c2);
                    $sd = vfmaq_f32($sd, a3, c3);
                };
            }
            dot16_row!(b_ptrs[0], s00, s01, s02, s03);
            dot16_row!(b_ptrs[1], s10, s11, s12, s13);
            dot16_row!(b_ptrs[2], s20, s21, s22, s23);
            dot16_row!(b_ptrs[3], s30, s31, s32, s33);

            idx += 16;
        }

        // 8-element tail
        while idx + 8 <= k {
            let a0 = vld1q_f32(a_ptr.add(idx));
            let a1 = vld1q_f32(a_ptr.add(idx + 4));

            macro_rules! dot8_row {
                ($bp:expr, $s0:ident, $s1:ident) => {
                    let br: uint16x8_t = vld1q_u16($bp.add(idx));
                    let bl: float32x4_t;
                    let bh: float32x4_t;
                    std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) bl, i = in(vreg) br);
                    std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) bh, i = in(vreg) br);
                    $s0 = vfmaq_f32($s0, a0, bl);
                    $s1 = vfmaq_f32($s1, a1, bh);
                };
            }
            dot8_row!(b_ptrs[0], s00, s01);
            dot8_row!(b_ptrs[1], s10, s11);
            dot8_row!(b_ptrs[2], s20, s21);
            dot8_row!(b_ptrs[3], s30, s31);
            idx += 8;
        }

        // Reduce 4 accumulators per row → scalar
        out[0] = vaddvq_f32(vaddq_f32(vaddq_f32(s00, s01), vaddq_f32(s02, s03)));
        out[1] = vaddvq_f32(vaddq_f32(vaddq_f32(s10, s11), vaddq_f32(s12, s13)));
        out[2] = vaddvq_f32(vaddq_f32(vaddq_f32(s20, s21), vaddq_f32(s22, s23)));
        out[3] = vaddvq_f32(vaddq_f32(vaddq_f32(s30, s31), vaddq_f32(s32, s33)));

        // Scalar tail
        while idx < k {
            let a_val = *a_ptr.add(idx);
            for r in 0..4 {
                out[r] += a_val * half::f16::from_bits(*b_ptrs[r].add(idx)).to_f32();
            }
            idx += 1;
        }
    }

    /// Single-row NEON dot product: A(F32) · B(F16) → scalar F32.
    /// Fused fcvtl + vfmaq_f32 — no intermediate F32 buffer needed.
    ///
    /// # Safety
    /// `a_ptr` must point to at least `k` f32 values, `b_ptr` to at least `k` u16 (F16) values.
    #[target_feature(enable = "neon")]
    pub unsafe fn vec_dot_f16_f32(k: usize, a_ptr: *const f32, b_ptr: *const u16) -> f32 {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let mut k_idx = 0;

        // Main loop: 16 elements per iteration
        while k_idx + 16 <= k {
            let b_raw0: uint16x8_t = vld1q_u16(b_ptr.add(k_idx));
            let b_raw1: uint16x8_t = vld1q_u16(b_ptr.add(k_idx + 8));

            let b_f32_0: float32x4_t;
            let b_f32_1: float32x4_t;
            let b_f32_2: float32x4_t;
            let b_f32_3: float32x4_t;

            std::arch::asm!("fcvtl {out:v}.4s, {inp:v}.4h", out = lateout(vreg) b_f32_0, inp = in(vreg) b_raw0);
            std::arch::asm!("fcvtl2 {out:v}.4s, {inp:v}.8h", out = lateout(vreg) b_f32_1, inp = in(vreg) b_raw0);
            std::arch::asm!("fcvtl {out:v}.4s, {inp:v}.4h", out = lateout(vreg) b_f32_2, inp = in(vreg) b_raw1);
            std::arch::asm!("fcvtl2 {out:v}.4s, {inp:v}.8h", out = lateout(vreg) b_f32_3, inp = in(vreg) b_raw1);

            let a0 = vld1q_f32(a_ptr.add(k_idx));
            let a1 = vld1q_f32(a_ptr.add(k_idx + 4));
            let a2 = vld1q_f32(a_ptr.add(k_idx + 8));
            let a3 = vld1q_f32(a_ptr.add(k_idx + 12));

            sum0 = vfmaq_f32(sum0, a0, b_f32_0);
            sum1 = vfmaq_f32(sum1, a1, b_f32_1);
            sum2 = vfmaq_f32(sum2, a2, b_f32_2);
            sum3 = vfmaq_f32(sum3, a3, b_f32_3);

            k_idx += 16;
        }

        // 8-element tail
        while k_idx + 8 <= k {
            let b_raw: uint16x8_t = vld1q_u16(b_ptr.add(k_idx));
            let b_lo: float32x4_t;
            let b_hi: float32x4_t;
            std::arch::asm!("fcvtl {out:v}.4s, {inp:v}.4h", out = lateout(vreg) b_lo, inp = in(vreg) b_raw);
            std::arch::asm!("fcvtl2 {out:v}.4s, {inp:v}.8h", out = lateout(vreg) b_hi, inp = in(vreg) b_raw);

            let a0 = vld1q_f32(a_ptr.add(k_idx));
            let a1 = vld1q_f32(a_ptr.add(k_idx + 4));

            sum0 = vfmaq_f32(sum0, a0, b_lo);
            sum1 = vfmaq_f32(sum1, a1, b_hi);

            k_idx += 8;
        }

        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);
        let mut sum = vaddvq_f32(sum0);

        while k_idx < k {
            sum += *a_ptr.add(k_idx) * half::f16::from_bits(*b_ptr.add(k_idx)).to_f32();
            k_idx += 1;
        }

        sum
    }

    /// NEON F16 weighted accumulate: out[i] += weight * v_f16[i]
    /// Fused fcvtl + vfmaq_f32 — operates directly on F16 data without temp buffer.
    ///
    /// # Safety
    /// `out_ptr` must point to at least `k` f32 values (read-write),
    /// `v_ptr` to at least `k` u16 (F16) values.
    #[target_feature(enable = "neon")]
    pub unsafe fn vec_mad_f16(k: usize, out_ptr: *mut f32, v_ptr: *const u16, weight: f32) {
        let w_v = vdupq_n_f32(weight);
        let mut idx = 0;

        // Main loop: 16 elements per iteration
        while idx + 16 <= k {
            let v_raw0: uint16x8_t = vld1q_u16(v_ptr.add(idx));
            let v_raw1: uint16x8_t = vld1q_u16(v_ptr.add(idx + 8));

            let v0: float32x4_t;
            let v1: float32x4_t;
            let v2: float32x4_t;
            let v3: float32x4_t;
            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) v0, i = in(vreg) v_raw0);
            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) v1, i = in(vreg) v_raw0);
            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) v2, i = in(vreg) v_raw1);
            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) v3, i = in(vreg) v_raw1);

            vst1q_f32(
                out_ptr.add(idx),
                vfmaq_f32(vld1q_f32(out_ptr.add(idx)), w_v, v0),
            );
            vst1q_f32(
                out_ptr.add(idx + 4),
                vfmaq_f32(vld1q_f32(out_ptr.add(idx + 4)), w_v, v1),
            );
            vst1q_f32(
                out_ptr.add(idx + 8),
                vfmaq_f32(vld1q_f32(out_ptr.add(idx + 8)), w_v, v2),
            );
            vst1q_f32(
                out_ptr.add(idx + 12),
                vfmaq_f32(vld1q_f32(out_ptr.add(idx + 12)), w_v, v3),
            );

            idx += 16;
        }

        // 8-element tail
        while idx + 8 <= k {
            let v_raw: uint16x8_t = vld1q_u16(v_ptr.add(idx));
            let vlo: float32x4_t;
            let vhi: float32x4_t;
            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) vlo, i = in(vreg) v_raw);
            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) vhi, i = in(vreg) v_raw);

            vst1q_f32(
                out_ptr.add(idx),
                vfmaq_f32(vld1q_f32(out_ptr.add(idx)), w_v, vlo),
            );
            vst1q_f32(
                out_ptr.add(idx + 4),
                vfmaq_f32(vld1q_f32(out_ptr.add(idx + 4)), w_v, vhi),
            );

            idx += 8;
        }

        // Scalar tail
        while idx < k {
            *out_ptr.add(idx) += weight * half::f16::from_bits(*v_ptr.add(idx)).to_f32();
            idx += 1;
        }
    }

    /// NEON bulk F16→F32 conversion using fcvtl.
    /// 16 elements per iteration (2× vld1q_u16 + 4× fcvtl/fcvtl2 + 4× vst1q_f32).
    ///
    /// # Safety
    /// `src` must point to at least `n` u16 (F16) values,
    /// `dst` to at least `n` f32 values (write-only).
    #[target_feature(enable = "neon")]
    pub unsafe fn bulk_f16_to_f32(src: *const u16, dst: *mut f32, n: usize) {
        let mut i = 0;
        while i + 16 <= n {
            let raw0: uint16x8_t = vld1q_u16(src.add(i));
            let raw1: uint16x8_t = vld1q_u16(src.add(i + 8));
            let f0: float32x4_t;
            let f1: float32x4_t;
            let f2: float32x4_t;
            let f3: float32x4_t;
            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) f0, i = in(vreg) raw0);
            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) f1, i = in(vreg) raw0);
            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) f2, i = in(vreg) raw1);
            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) f3, i = in(vreg) raw1);
            vst1q_f32(dst.add(i), f0);
            vst1q_f32(dst.add(i + 4), f1);
            vst1q_f32(dst.add(i + 8), f2);
            vst1q_f32(dst.add(i + 12), f3);
            i += 16;
        }
        while i + 8 <= n {
            let raw: uint16x8_t = vld1q_u16(src.add(i));
            let lo: float32x4_t;
            let hi: float32x4_t;
            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) lo, i = in(vreg) raw);
            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) hi, i = in(vreg) raw);
            vst1q_f32(dst.add(i), lo);
            vst1q_f32(dst.add(i + 4), hi);
            i += 8;
        }
        while i < n {
            *dst.add(i) = half::f16::from_bits(*src.add(i)).to_f32();
            i += 1;
        }
    }

    fn matmul_transposed_q4_0(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        let k = a_shape[a_rank - 1];
        let n = b_shape[b_rank - 2];
        let m: usize = a_shape[..a_rank - 1].iter().product();

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
        unsafe {
            a_q8.set_len(total_q8_blocks);
        }

        for i in 0..m {
            let a_offset = i * k;
            let a_row = &a_data[a_offset..a_offset + k];
            let q8_row = &mut a_q8[i * nb_k_q8..(i + 1) * nb_k_q8];
            unsafe {
                self.quantize_row_q8_0(a_row, q8_row, k);
            }
        }

        // Adaptive chunking strategy
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;
        let chunk_size = chunk_size.max(256);

        out_data.par_chunks_mut(chunk_size).enumerate().for_each(
            |(chunk_idx, chunk): (usize, &mut [f32])| {
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
            },
        );
        Ok(())
    }

    fn matmul_transposed_q4_0_serial(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: &mut Tensor,
    ) -> Result<()> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();

        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 2];
        let m: usize = a_shape[..a_shape.len() - 1].iter().product();

        let a_data = a.as_slice::<f32>();
        let nb_k = k / QK4_0;

        // Safety: b is created as Q4_0 tensor, so data is BlockQ4_0
        let total_blocks = n * nb_k;
        let b_blocks =
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const BlockQ4_0, total_blocks) };

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

                let i16_low = vcombine_s16(i16_0, i16_1);
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

                let i16_low2 = vcombine_s16(i16_4, i16_5);
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
    pub unsafe fn vec_dot_q4_0_q8_0(
        &self,
        n: usize,
        s: &mut f32,
        vx: *const BlockQ4_0,
        vy: *const BlockQ8_0,
    ) {
        let nb = n / QK8_0;

        let mut sumf = 0.0;
        unsafe {
            let m4b = vdupq_n_u8(0x0F);
            let s8b = vdupq_n_s8(8);

            let mut i = 0;
            while i + 1 < nb {
                let x0 = &*vx.add(i);
                let y0 = &*vy.add(i);
                let x1 = &*vx.add(i + 1);
                let y1 = &*vy.add(i + 1);

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
                let acc0 = vaddlvq_s16(mul_l0)
                    + vaddlvq_s16(mul_l1)
                    + vaddlvq_s16(mul_h0)
                    + vaddlvq_s16(mul_h1);

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
                let acc1 = vaddlvq_s16(mul_l0_1)
                    + vaddlvq_s16(mul_l1_1)
                    + vaddlvq_s16(mul_h0_1)
                    + vaddlvq_s16(mul_h1_1);

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
                let acc = vaddlvq_s16(mul_l0)
                    + vaddlvq_s16(mul_l1)
                    + vaddlvq_s16(mul_h0)
                    + vaddlvq_s16(mul_h1);

                sumf += d * acc as f32;
                i += 1;
            }
        }
        *s = sumf;
    }

    // Dot Product Q4_0 * Q8_0 (NEON + DotProd)
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn vec_dot_q4_0_q8_0_sdot(
        &self,
        n: usize,
        s: &mut f32,
        vx: *const BlockQ4_0,
        vy: *const BlockQ8_0,
    ) {
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

// --- SpinPool work context for F16 GEMV ---

#[repr(C)]
struct F16GemvCtx {
    a_ptr: *const f32,
    b_base: *const u16,
    out_ptr: *mut f32,
    k: usize,
    n: usize,
    rows_per_chunk: usize,
}

// --- N-major GEMM context (prefill, M > 1) ---

/// Context for N-major GEMM dispatch.
/// Instead of dispatching M times (one per activation row), we dispatch once
/// with M × n_chunks tasks. Consecutive chunk_ids share the same B rows,
/// so the same core reuses B data from L1 across all M activation rows.
#[repr(C)]
struct F16GemmCtx {
    a_base: *const f32, // A[M, K] row-major
    b_base: *const u16, // B[N, K] row-major (F16, transposed)
    out_base: *mut f32, // Out[M, N] row-major
    m: usize,
    n: usize,
    k: usize,
    rows_per_chunk: usize, // NR-aligned N rows per chunk
}

unsafe impl Send for F16GemmCtx {}
unsafe impl Sync for F16GemmCtx {}

// --- Fused multi-matmul dispatch (QKV / gate+up) ---

/// Context for fused multi-matmul dispatch.
/// Multiple matmuls share input but have different weights/outputs.
/// Chunks are distributed across all matmuls in a single dispatch.
#[repr(C)]
struct FusedGemvCtx {
    a_ptr: *const f32,
    k: usize,
    rows_per_chunk: usize,
    n_matmuls: usize,
    // Per-matmul params (up to 3 for QKV)
    b_bases: [*const u16; 3],
    out_ptrs: [*mut f32; 3],
    ns: [usize; 3],
    chunk_offsets: [usize; 3], // cumulative chunk offset per matmul
}

/// Work function for fused multi-matmul: chunk_id spans across all matmuls.
///
/// # Safety
/// Called by SpinPool workers with valid FusedGemvCtx pointer.
unsafe fn fused_gemv_chunk(ctx_ptr: *const u8, chunk_id: usize) {
    const NR: usize = 4;
    let ctx = &*(ctx_ptr as *const FusedGemvCtx);

    // Determine which matmul this chunk belongs to.
    // chunk_offsets[i] = cumulative start chunk for matmul i.
    // For n_matmuls==2, chunk_offsets[2] is unused — route via n_matmuls check.
    let (mat_idx, local_chunk) = if chunk_id < ctx.chunk_offsets[1] {
        (0, chunk_id - ctx.chunk_offsets[0])
    } else if ctx.n_matmuls == 2 || chunk_id < ctx.chunk_offsets[2] {
        (1, chunk_id - ctx.chunk_offsets[1])
    } else {
        (2, chunk_id - ctx.chunk_offsets[2])
    };

    let b_base = ctx.b_bases[mat_idx];
    let out_ptr = ctx.out_ptrs[mat_idx];
    let n = ctx.ns[mat_idx];

    let j_start = local_chunk * ctx.rows_per_chunk;
    let j_end = (j_start + ctx.rows_per_chunk).min(n);

    let mut j = j_start;
    while j + NR <= j_end {
        let b_ptrs = [
            b_base.add(j * ctx.k),
            b_base.add((j + 1) * ctx.k),
            b_base.add((j + 2) * ctx.k),
            b_base.add((j + 3) * ctx.k),
        ];
        let mut results = [0.0f32; NR];
        CpuBackendNeon::vec_dot_f16_f32_4rows(ctx.k, ctx.a_ptr, b_ptrs, &mut results);
        std::ptr::copy_nonoverlapping(results.as_ptr(), out_ptr.add(j), NR);
        j += NR;
    }
    while j < j_end {
        *out_ptr.add(j) = CpuBackendNeon::vec_dot_f16_f32(ctx.k, ctx.a_ptr, b_base.add(j * ctx.k));
        j += 1;
    }
}

/// Dispatch multiple F16 matmuls (sharing the same input) as a single SpinPool dispatch.
/// Reduces dispatch overhead from N dispatches to 1.
///
/// # Safety
/// All tensor pointers must be valid. Input must be F32, weights F16, outputs F32.
/// All matmuls must share the same K dimension.
pub unsafe fn fused_matmul_f16(
    a_data: *const f32,
    k: usize,
    matmuls: &[(*const u16, *mut f32, usize)], // (weight_base, out_ptr, n_rows)
) {
    use crate::core::thread_pool;
    const NR: usize = 4;

    let pool = thread_pool::get_pool();
    let n_threads = rayon::current_num_threads();
    let total_rows: usize = matmuls.iter().map(|m| m.2).sum();
    let target_chunks = n_threads * 8;
    let rows_per_chunk = ((total_rows + target_chunks - 1) / target_chunks + NR - 1) / NR * NR;
    let rows_per_chunk = rows_per_chunk.max(NR);

    let mut b_bases = [std::ptr::null::<u16>(); 3];
    let mut out_ptrs = [std::ptr::null_mut::<f32>(); 3];
    let mut ns = [0usize; 3];
    let mut chunk_offsets = [0usize; 3];
    let mut total_chunks = 0;

    for (i, &(b, o, n)) in matmuls.iter().enumerate().take(3) {
        b_bases[i] = b;
        out_ptrs[i] = o;
        ns[i] = n;
        chunk_offsets[i] = total_chunks;
        total_chunks += (n + rows_per_chunk - 1) / rows_per_chunk;
    }

    let ctx = FusedGemvCtx {
        a_ptr: a_data,
        k,
        rows_per_chunk,
        n_matmuls: matmuls.len(),
        b_bases,
        out_ptrs,
        ns,
        chunk_offsets,
    };

    pool.dispatch(
        total_chunks,
        fused_gemv_chunk,
        &ctx as *const FusedGemvCtx as *const u8,
    );
}

/// Work function for SpinPool: processes a coarse chunk of output rows.
/// Each chunk contains multiple NR=4 row blocks to reduce atomic contention.
///
/// # Safety
/// Called by SpinPool workers with valid F16GemvCtx pointer.
unsafe fn f16_gemv_chunk(ctx_ptr: *const u8, chunk_id: usize) {
    const NR: usize = 4;
    let ctx = &*(ctx_ptr as *const F16GemvCtx);
    let j_start = chunk_id * ctx.rows_per_chunk;
    let j_end = (j_start + ctx.rows_per_chunk).min(ctx.n);

    let mut j = j_start;
    while j + NR <= j_end {
        let b_ptrs = [
            ctx.b_base.add(j * ctx.k),
            ctx.b_base.add((j + 1) * ctx.k),
            ctx.b_base.add((j + 2) * ctx.k),
            ctx.b_base.add((j + 3) * ctx.k),
        ];
        let mut results = [0.0f32; NR];
        CpuBackendNeon::vec_dot_f16_f32_4rows(ctx.k, ctx.a_ptr, b_ptrs, &mut results);
        std::ptr::copy_nonoverlapping(results.as_ptr(), ctx.out_ptr.add(j), NR);
        j += NR;
    }
    // Scalar tail within chunk
    while j < j_end {
        *ctx.out_ptr.add(j) =
            CpuBackendNeon::vec_dot_f16_f32(ctx.k, ctx.a_ptr, ctx.b_base.add(j * ctx.k));
        j += 1;
    }
}

/// N-major GEMM work function: processes one (n_chunk, m_row) pair.
/// Consecutive chunk_ids share the same N-chunk (B rows), enabling L1 reuse
/// when the same core picks up sequential tasks via SpinPool's fetch_add.
///
/// # Safety
/// Called by SpinPool workers with valid F16GemmCtx pointer.
unsafe fn f16_gemm_chunk(ctx_ptr: *const u8, chunk_id: usize) {
    const NR: usize = 4;
    let ctx = &*(ctx_ptr as *const F16GemmCtx);

    // N-major mapping: chunk_id = n_chunk_idx * m + m_row
    // Consecutive chunk_ids iterate m_row first → same B rows stay in L1
    let n_chunk_idx = chunk_id / ctx.m;
    let m_row = chunk_id % ctx.m;

    let j_start = n_chunk_idx * ctx.rows_per_chunk;
    let j_end = (j_start + ctx.rows_per_chunk).min(ctx.n);

    let a_ptr = ctx.a_base.add(m_row * ctx.k);
    let out_ptr = ctx.out_base.add(m_row * ctx.n);

    let mut j = j_start;
    while j + NR <= j_end {
        let b_ptrs = [
            ctx.b_base.add(j * ctx.k),
            ctx.b_base.add((j + 1) * ctx.k),
            ctx.b_base.add((j + 2) * ctx.k),
            ctx.b_base.add((j + 3) * ctx.k),
        ];
        let mut results = [0.0f32; NR];
        CpuBackendNeon::vec_dot_f16_f32_4rows(ctx.k, a_ptr, b_ptrs, &mut results);
        std::ptr::copy_nonoverlapping(results.as_ptr(), out_ptr.add(j), NR);
        j += NR;
    }
    // Scalar tail within chunk
    while j < j_end {
        *out_ptr.add(j) = CpuBackendNeon::vec_dot_f16_f32(ctx.k, a_ptr, ctx.b_base.add(j * ctx.k));
        j += 1;
    }
}
