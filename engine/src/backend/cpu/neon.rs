use crate::backend::cpu::common::CpuBackendCommon;
use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::quant::{BlockQ4_0, BlockQ4_1, BlockQ8_0, QK4_0, QK4_1, QK8_0};
use crate::core::tensor::Tensor;
use crate::core::thread_pool::{self, WorkFn};
use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::arch::aarch64::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

/// Runtime toggle: when true, F16 matmul uses Rayon instead of SpinPool.
/// Set via `--use-rayon` CLI flag for A/B benchmarking.
pub static USE_RAYON: AtomicBool = AtomicBool::new(false);

// Reusable Q8_0 workspace for matmul_transposed_q4_0.
// Eliminates per-matmul Vec<BlockQ8_0> allocation (~112 allocs/token).
// Safety: single-threaded inference model — only the main thread quantizes into this buffer
// before par_chunks_mut reads it via raw pointer.
thread_local! {
    static Q8_WORKSPACE: RefCell<Vec<BlockQ8_0>> = const { RefCell::new(Vec::new()) };
}

/// Wrapper to send `*const BlockQ8_0` into Rayon parallel closures.
/// Safety: the pointer targets a thread-local buffer that is fully initialized
/// before the parallel region and only read (never written) during it.
#[derive(Clone, Copy)]
struct SendSyncPtr(*const BlockQ8_0);
unsafe impl Send for SendSyncPtr {}
unsafe impl Sync for SendSyncPtr {}
impl SendSyncPtr {
    #[inline(always)]
    fn ptr(self) -> *const BlockQ8_0 {
        self.0
    }
}

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
        let a_data = a.as_mut_slice::<f32>();
        let b_data = b.as_slice::<f32>();
        unsafe { Self::swiglu_neon(a_data, b_data) };
        Ok(())
    }

    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32, add_unit: bool) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let x_data = x.as_mut_slice::<f32>();
        let w_data = w.as_slice::<f32>();
        x_data.par_chunks_mut(dim).for_each(|row| {
            unsafe { Self::rms_norm_neon(row, w_data, eps, add_unit) };
        });
        Ok(())
    }

    fn rms_norm_oop(
        &self,
        x: &Tensor,
        out: &mut Tensor,
        w: &Tensor,
        eps: f32,
        add_unit: bool,
    ) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let x_data = x.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();
        let w_data = w.as_slice::<f32>();
        for (x_row, out_row) in x_data.chunks(dim).zip(out_data.chunks_mut(dim)) {
            unsafe { Self::rms_norm_oop_neon(x_row, out_row, w_data, eps, add_unit) };
        }
        Ok(())
    }

    fn add_rms_norm_oop(
        &self,
        x: &mut Tensor,
        residual: &Tensor,
        out: &mut Tensor,
        w: &Tensor,
        eps: f32,
        add_unit: bool,
    ) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let x_data = x.as_mut_slice::<f32>();
        let res_data = residual.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();
        let w_data = w.as_slice::<f32>();
        for ((x_row, res_row), out_row) in x_data
            .chunks_mut(dim)
            .zip(res_data.chunks(dim))
            .zip(out_data.chunks_mut(dim))
        {
            unsafe { Self::add_rms_norm_oop_neon(x_row, res_row, out_row, w_data, eps, add_unit) };
        }
        Ok(())
    }

    fn gelu_tanh_mul(&self, gate: &mut Tensor, up: &Tensor) -> Result<()> {
        let gate_data = gate.as_mut_slice::<f32>();
        let up_data = up.as_slice::<f32>();
        unsafe { Self::gelu_tanh_mul_neon(gate_data, up_data) };
        Ok(())
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

    #[allow(clippy::too_many_arguments)]
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
        scores_out: Option<&mut [f32]>,
    ) -> Result<()> {
        // F16 KV cache: use NEON-optimized path (direct F16·F32 dot, vectorized softmax)
        if k_cache.dtype() == DType::F16 {
            return Self::attention_gen_f16_neon(
                q,
                k_cache,
                v_cache,
                out,
                num_heads_q,
                num_heads_kv,
                head_dim,
                cache_seq_len,
                scores_out,
            );
        }
        // F32 / Q4_0: delegate to common implementation
        CpuBackendCommon::new().attention_gen(
            q,
            k_cache,
            v_cache,
            out,
            num_heads_q,
            num_heads_kv,
            head_dim,
            cache_seq_len,
            scores_out,
        )
    }
}

impl CpuBackendNeon {
    /// NEON-optimized attention for F16 KV cache.
    ///
    /// Three optimizations over the generic common.rs F16 path:
    /// 1. Direct F16·F32 dot via `vec_dot_f16_f32` — eliminates intermediate F32 buffer
    /// 2. Vectorized softmax using NEON `v_expf` — 4-wide exp instead of scalar
    /// 3. Stack-allocated scores buffer for typical sequence lengths (<=4096)
    #[allow(clippy::too_many_arguments)]
    fn attention_gen_f16_neon(
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        num_heads_q: usize,
        num_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
        scores_out: Option<&mut [f32]>,
    ) -> Result<()> {
        let q_data = q.as_slice::<f32>();
        let out_data = out.as_mut_slice::<f32>();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = num_heads_q / num_heads_kv;

        let k_data = k_cache.as_slice::<half::f16>();
        let v_data = v_cache.as_slice::<half::f16>();

        // Wrap raw pointer for Send+Sync (Rayon par_chunks_mut capture).
        // Safety: each head h writes to non-overlapping region [h*stride .. h*stride+cache_seq_len].
        #[derive(Clone, Copy)]
        struct SendPtr(*mut f32);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}

        let scores_ptr = scores_out.as_ref().map(|s| SendPtr(s.as_ptr() as *mut f32));
        let scores_stride = scores_out
            .as_ref()
            .map(|s| s.len() / num_heads_q)
            .unwrap_or(0);

        // Detect layout: HeadMajor [batch, kv_heads, capacity, head_dim]
        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == num_heads_kv && k_shape[1] != k_shape[2];
        let capacity = if is_head_major { k_shape[2] } else { 0 };

        // Stack threshold: avoid heap alloc for typical decode sequences
        const STACK_SCORES_MAX: usize = 4096;

        out_data
            .par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(h, out_h)| {
                let kv_h = h / gqa_ratio;
                let q_off = h * head_dim;
                let q_ptr = unsafe { q_data.as_ptr().add(q_off) };

                // Fix 3: stack-allocated scores for typical lengths, heap fallback
                let mut stack_scores = [0.0f32; STACK_SCORES_MAX];
                let mut heap_scores: Vec<f32>;
                let scores: &mut [f32] = if cache_seq_len <= STACK_SCORES_MAX {
                    &mut stack_scores[..cache_seq_len]
                } else {
                    heap_scores = vec![0.0f32; cache_seq_len];
                    &mut heap_scores[..]
                };

                // --- Q * K^T: direct F16·F32 dot (Fix 1) ---
                // Process 4 timesteps at once using vec_dot_f16_f32_4rows
                let k_f16_ptr = k_data.as_ptr() as *const u16;
                let full_4 = cache_seq_len / 4;
                for chunk in 0..full_4 {
                    let t_base = chunk * 4;
                    let mut b_ptrs = [std::ptr::null::<u16>(); 4];
                    for r in 0..4 {
                        let t = t_base + r;
                        let off = if is_head_major {
                            (kv_h * capacity + t) * head_dim
                        } else {
                            (t * num_heads_kv + kv_h) * head_dim
                        };
                        b_ptrs[r] = unsafe { k_f16_ptr.add(off) };
                    }
                    let mut dots = [0.0f32; 4];
                    unsafe {
                        Self::vec_dot_f16_f32_4rows(head_dim, q_ptr, b_ptrs, &mut dots);
                    }
                    for r in 0..4 {
                        scores[t_base + r] = dots[r] * scale;
                    }
                }
                // Remaining timesteps (0-3)
                for t in (full_4 * 4)..cache_seq_len {
                    let off = if is_head_major {
                        (kv_h * capacity + t) * head_dim
                    } else {
                        (t * num_heads_kv + kv_h) * head_dim
                    };
                    let k_ptr = unsafe { k_f16_ptr.add(off) };
                    let dot =
                        unsafe { Self::vec_dot_f16_f32(head_dim, q_ptr, k_ptr) };
                    scores[t] = dot * scale;
                }

                // --- Softmax with NEON v_expf (Fix 2) ---
                // Pass 0: sanitize NaN in Q*K^T scores.
                // NaN can arise from dot(Q,K) when Q contains NaN (propagated
                // from a previous layer whose softmax produced NaN).  Replace
                // NaN with -inf so softmax assigns zero probability to those
                // positions instead of poisoning the entire distribution.
                unsafe {
                    let s_ptr = scores.as_mut_ptr();
                    let mut i = 0;
                    while i + 4 <= cache_seq_len {
                        let v = vld1q_f32(s_ptr.add(i));
                        // NaN != NaN → comparison yields 0 for NaN lanes
                        let nan_mask = vmvnq_u32(vceqq_f32(v, v));
                        // Replace NaN lanes with -inf
                        let clean = vbslq_f32(nan_mask, vdupq_n_f32(f32::NEG_INFINITY), v);
                        vst1q_f32(s_ptr.add(i), clean);
                        i += 4;
                    }
                    while i < cache_seq_len {
                        if (*s_ptr.add(i)).is_nan() {
                            *s_ptr.add(i) = f32::NEG_INFINITY;
                        }
                        i += 1;
                    }
                }

                // Pass 1: find max
                let mut max_val = f32::NEG_INFINITY;
                unsafe {
                    let s_ptr = scores.as_ptr();
                    let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
                    let mut i = 0;
                    while i + 4 <= cache_seq_len {
                        max_v = vmaxq_f32(max_v, vld1q_f32(s_ptr.add(i)));
                        i += 4;
                    }
                    max_val = vmaxvq_f32(max_v);
                    while i < cache_seq_len {
                        max_val = max_val.max(*s_ptr.add(i));
                        i += 1;
                    }
                }

                // Guard: if all logits were NaN (now all -inf), max_val stays
                // -inf.  exp(-inf - (-inf)) = exp(NaN) would poison everything.
                // Fall back to uniform distribution in that case.
                if max_val.is_infinite() && max_val.is_sign_negative() {
                    let uniform = 1.0 / cache_seq_len as f32;
                    for s in scores[..cache_seq_len].iter_mut() {
                        *s = uniform;
                    }
                } else {
                // Pass 2: exp(x - max) and sum
                // Use scalar exp() per lane — v_expf has bit-manipulation precision
                // issues on certain ARM cores (Apple Silicon, Cortex-A720) that
                // corrupt softmax distributions at longer sequence lengths (300+).
                // Same rationale as v_silu / v_tanh scalar fallbacks.
                let mut sum_exp = 0.0f32;
                unsafe {
                    let s_ptr = scores.as_mut_ptr();
                    let max_v = vdupq_n_f32(max_val);
                    let mut sum_v = vdupq_n_f32(0.0);
                    let mut i = 0;
                    while i + 4 <= cache_seq_len {
                        let x = vsubq_f32(vld1q_f32(s_ptr.add(i)), max_v);
                        // Scalar exp per lane for correctness
                        let mut vals = [0.0f32; 4];
                        vst1q_f32(vals.as_mut_ptr(), x);
                        vals[0] = vals[0].exp();
                        vals[1] = vals[1].exp();
                        vals[2] = vals[2].exp();
                        vals[3] = vals[3].exp();
                        let e = vld1q_f32(vals.as_ptr());
                        vst1q_f32(s_ptr.add(i), e);
                        sum_v = vaddq_f32(sum_v, e);
                        i += 4;
                    }
                    sum_exp = vaddvq_f32(sum_v);
                    while i < cache_seq_len {
                        let e = (*s_ptr.add(i) - max_val).exp();
                        *s_ptr.add(i) = e;
                        sum_exp += e;
                        i += 1;
                    }
                }

                // Pass 3: normalize
                // Guard: if sum_exp is 0, NaN, or inf, fall back to uniform.
                if sum_exp.is_nan() || sum_exp <= 0.0 || sum_exp.is_infinite() {
                    let uniform = 1.0 / cache_seq_len as f32;
                    for s in scores[..cache_seq_len].iter_mut() {
                        *s = uniform;
                    }
                } else {
                let inv_sum = 1.0 / sum_exp;
                unsafe {
                    let s_ptr = scores.as_mut_ptr();
                    let inv_v = vdupq_n_f32(inv_sum);
                    let mut i = 0;
                    while i + 4 <= cache_seq_len {
                        let x = vld1q_f32(s_ptr.add(i));
                        vst1q_f32(s_ptr.add(i), vmulq_f32(x, inv_v));
                        i += 4;
                    }
                    while i < cache_seq_len {
                        *s_ptr.add(i) *= inv_sum;
                        i += 1;
                    }
                }
                } // end sum_exp guard
                } // end max_val guard

                // Copy scores for diagnostics if requested
                if let Some(SendPtr(ptr)) = scores_ptr {
                    unsafe {
                        let dst = std::slice::from_raw_parts_mut(
                            ptr.add(h * scores_stride),
                            cache_seq_len,
                        );
                        dst.copy_from_slice(&scores[..cache_seq_len]);
                    }
                }

                // --- Weighted V sum: fused F16→F32 convert + FMA ---
                for d in 0..head_dim {
                    out_h[d] = 0.0;
                }
                let v_f16_ptr = v_data.as_ptr() as *const u16;
                for t in 0..cache_seq_len {
                    let w = scores[t];
                    if w == 0.0 {
                        continue;
                    }
                    let off = if is_head_major {
                        (kv_h * capacity + t) * head_dim
                    } else {
                        (t * num_heads_kv + kv_h) * head_dim
                    };
                    // Fused F16→F32 + weighted accumulation with NEON
                    unsafe {
                        let vp = v_f16_ptr.add(off);
                        let o_ptr = out_h.as_mut_ptr();
                        let w_v = vdupq_n_f32(w);
                        let mut i = 0;
                        while i + 16 <= head_dim {
                            // Load 16 F16 values → 4 F32x4 vectors
                            let raw0: uint16x8_t = vld1q_u16(vp.add(i));
                            let raw1: uint16x8_t = vld1q_u16(vp.add(i + 8));
                            let f0: float32x4_t;
                            let f1: float32x4_t;
                            let f2: float32x4_t;
                            let f3: float32x4_t;
                            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) f0, i = in(vreg) raw0);
                            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) f1, i = in(vreg) raw0);
                            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) f2, i = in(vreg) raw1);
                            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) f3, i = in(vreg) raw1);

                            vst1q_f32(
                                o_ptr.add(i),
                                vfmaq_f32(vld1q_f32(o_ptr.add(i)), w_v, f0),
                            );
                            vst1q_f32(
                                o_ptr.add(i + 4),
                                vfmaq_f32(vld1q_f32(o_ptr.add(i + 4)), w_v, f1),
                            );
                            vst1q_f32(
                                o_ptr.add(i + 8),
                                vfmaq_f32(vld1q_f32(o_ptr.add(i + 8)), w_v, f2),
                            );
                            vst1q_f32(
                                o_ptr.add(i + 12),
                                vfmaq_f32(vld1q_f32(o_ptr.add(i + 12)), w_v, f3),
                            );
                            i += 16;
                        }
                        while i + 8 <= head_dim {
                            let raw: uint16x8_t = vld1q_u16(vp.add(i));
                            let fl: float32x4_t;
                            let fh: float32x4_t;
                            std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) fl, i = in(vreg) raw);
                            std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) fh, i = in(vreg) raw);
                            vst1q_f32(
                                o_ptr.add(i),
                                vfmaq_f32(vld1q_f32(o_ptr.add(i)), w_v, fl),
                            );
                            vst1q_f32(
                                o_ptr.add(i + 4),
                                vfmaq_f32(vld1q_f32(o_ptr.add(i + 4)), w_v, fh),
                            );
                            i += 8;
                        }
                        while i < head_dim {
                            *o_ptr.add(i) += w * half::f16::from_bits(*vp.add(i)).to_f32();
                            i += 1;
                        }
                    }
                }
            });

        Ok(())
    }

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

        // M=1 decode: serial path uses F32×F16 dot directly (vec_dot_f16_f32_4rows),
        // avoiding F32→F16 conversion + heap alloc + SpinPool dispatch overhead.
        // Threshold: n*k < 2M covers Wk/Wv/Wq/Wo but keeps FFN gate/up/down parallel.
        let serial_threshold = if m == 1 { 2_000_000 } else { 100_000 };
        if (m * n * k) < serial_threshold {
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

            // Pre-convert A from F32 to F16 for the F16 matmul kernels
            // (one-time cost per forward-pass call; amortized across N).
            let mut a_f16_buf: Vec<u16> = vec![0u16; m * k];
            unsafe {
                for i in 0..m {
                    Self::f32_to_f16_neon(
                        a_data.as_ptr().add(i * k),
                        a_f16_buf.as_mut_ptr().add(i * k),
                        k,
                    );
                }
            }

            if m > 1 {
                // N-major GEMM with MR=4: groups of 4 A rows share B loads via F16 FMA.
                // chunk_id = n_chunk_idx * m_groups + m_group, where m_groups = ceil(m/4)
                let m_groups = (m + 3) / 4;
                let total_tasks = n_chunks * m_groups;
                let ctx = F16GemmCtx {
                    a_base: a_data.as_ptr(),
                    a_f16_base: a_f16_buf.as_ptr(),
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
                // M=1: decode GEMV path (native F16 FMA)
                let ctx = F16GemvCtx {
                    a_ptr: a_data.as_ptr(),
                    a_f16_ptr: a_f16_buf.as_ptr(),
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

    // --- NEON vectorized exp/silu/swiglu (ported from llama.cpp ggml_v_expf) ---

    /// NEON vectorized exp(x) — polynomial approximation from ARM optimized routine.
    /// Maximum error: 1.45358 + 0.5 ULP. Matches llama.cpp's ggml_v_expf().
    #[inline(always)]
    unsafe fn v_expf(x: float32x4_t) -> float32x4_t {
        let r = vdupq_n_f32(f32::from_bits(0x4b00_0000)); // 0x1.8p23
        let z = vfmaq_f32(r, x, vdupq_n_f32(f32::from_bits(0x3fb8_aa3b))); // 0x1.715476p+0
        let n = vsubq_f32(z, r);
        let b = vfmsq_f32(
            vfmsq_f32(x, n, vdupq_n_f32(f32::from_bits(0x3eb1_7200))), // 0x1.62e4p-1
            n,
            vdupq_n_f32(f32::from_bits(0x35bf_be8e)), // 0x1.7f7d1cp-20
        );
        let e = vshlq_n_u32::<23>(vreinterpretq_u32_f32(z));
        let k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1.0))));
        let c = vcagtq_f32(n, vdupq_n_f32(126.0));
        let u = vmulq_f32(b, b);
        let j = vfmaq_f32(
            vmulq_f32(vdupq_n_f32(f32::from_bits(0x3f7f_fff6)), b), // 0x1.ffffecp-1
            vfmaq_f32(
                vfmaq_f32(
                    vdupq_n_f32(f32::from_bits(0x3f7f_fedb)), // 0x1.fffdb6p-2
                    vdupq_n_f32(f32::from_bits(0x3e2a_af33)), // 0x1.555e66p-3
                    b,
                ),
                vfmaq_f32(
                    vdupq_n_f32(f32::from_bits(0x3d2b_9f17)), // 0x1.573e2ep-5
                    vdupq_n_f32(f32::from_bits(0x3c07_2010)), // 0x1.0e4020p-7
                    b,
                ),
                u,
            ),
            u,
        );
        // Fast path: no overflow/underflow
        if vpaddd_u64(vreinterpretq_u64_u32(c)) == 0 {
            return vfmaq_f32(k, j, k);
        }
        // Slow path: handle overflow/underflow
        let d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x8200_0000));
        let s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f00_0000)));
        let s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
        vbslq_f32(
            vcagtq_f32(n, vdupq_n_f32(192.0)),
            vmulq_f32(s1, s1),
            vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)),
        )
    }

    /// NEON vectorized silu: x / (1 + exp(-x))
    /// Uses scalar exp() per lane for correctness on all ARM64 implementations.
    #[inline(always)]
    unsafe fn v_silu(x: float32x4_t) -> float32x4_t {
        // Scalar exp per lane — avoids v_expf bit-manipulation issues on Apple Silicon
        let mut vals = [0.0f32; 4];
        vst1q_f32(vals.as_mut_ptr(), x);
        for v in vals.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
        vld1q_f32(vals.as_ptr())
    }

    /// NEON vectorized swiglu: silu(a) * b
    #[target_feature(enable = "neon")]
    unsafe fn swiglu_neon(a: &mut [f32], b: &[f32]) {
        let n = a.len();
        let mut i = 0;
        while i + 4 <= n {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vld1q_f32(b.as_ptr().add(i));
            let result = vmulq_f32(Self::v_silu(av), bv);
            vst1q_f32(a.as_mut_ptr().add(i), result);
            i += 4;
        }
        // Scalar tail
        while i < n {
            let x = a[i];
            a[i] = (x / (1.0 + (-x).exp())) * b[i];
            i += 1;
        }
    }

    /// NEON vectorized tanh — scalar per lane for correctness on all ARM64.
    /// Uses scalar tanh() to avoid v_expf bit-manipulation issues
    /// (same approach as v_silu — see Apple Silicon / Cortex-A720 errata).
    #[inline(always)]
    unsafe fn v_tanh(x: float32x4_t) -> float32x4_t {
        let mut vals = [0.0f32; 4];
        vst1q_f32(vals.as_mut_ptr(), x);
        for v in vals.iter_mut() {
            *v = v.tanh();
        }
        vld1q_f32(vals.as_ptr())
    }

    /// NEON vectorized GELU-tanh-mul: gate[i] = gelu_tanh(gate[i]) * up[i]
    /// gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    #[target_feature(enable = "neon")]
    unsafe fn gelu_tanh_mul_neon(gate: &mut [f32], up: &[f32]) {
        let n = gate.len();
        let sqrt_2_over_pi = vdupq_n_f32((2.0_f32 / std::f32::consts::PI).sqrt());
        let coeff = vdupq_n_f32(0.044715);
        let half = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);

        let mut i = 0;
        while i + 4 <= n {
            let x = vld1q_f32(gate.as_ptr().add(i));
            let u = vld1q_f32(up.as_ptr().add(i));

            // x^3
            let x2 = vmulq_f32(x, x);
            let x3 = vmulq_f32(x2, x);
            // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
            let inner = vmulq_f32(sqrt_2_over_pi, vfmaq_f32(x, coeff, x3));
            // 0.5 * x * (1 + tanh(inner)) * up
            let tanh_val = Self::v_tanh(inner);
            let result = vmulq_f32(vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, tanh_val)), u);
            vst1q_f32(gate.as_mut_ptr().add(i), result);
            i += 4;
        }
        // Scalar tail
        let sqrt_2_over_pi_s: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
        while i < n {
            let x = gate[i];
            let inner = sqrt_2_over_pi_s * (x + 0.044715 * x * x * x);
            gate[i] = 0.5 * x * (1.0 + inner.tanh()) * up[i];
            i += 1;
        }
    }

    /// NEON vectorized RMS norm with add_unit support.
    /// Pass 1: sum of squares (2-accumulator unroll)
    /// Pass 2: x[i] = x[i] * scale * w_eff, where w_eff = (1+w[i]) or w[i]
    #[target_feature(enable = "neon")]
    unsafe fn rms_norm_neon(row: &mut [f32], w: &[f32], eps: f32, add_unit: bool) {
        let dim = row.len();
        let ptr = row.as_mut_ptr();
        let w_ptr = w.as_ptr();

        // Pass 1: sum of squares with 2-accumulator unroll
        let mut sum_v0 = vdupq_n_f32(0.0);
        let mut sum_v1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 8 <= dim {
            let v0 = vld1q_f32(ptr.add(i));
            let v1 = vld1q_f32(ptr.add(i + 4));
            sum_v0 = vfmaq_f32(sum_v0, v0, v0);
            sum_v1 = vfmaq_f32(sum_v1, v1, v1);
            i += 8;
        }
        while i + 4 <= dim {
            let v = vld1q_f32(ptr.add(i));
            sum_v0 = vfmaq_f32(sum_v0, v, v);
            i += 4;
        }
        sum_v0 = vaddq_f32(sum_v0, sum_v1);
        let mut sum_sq = vaddvq_f32(sum_v0);
        while i < dim {
            sum_sq += *ptr.add(i) * *ptr.add(i);
            i += 1;
        }

        let scale = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        let scale_v = vdupq_n_f32(scale);

        // Pass 2: x[i] = x[i] * scale * w_eff
        i = 0;
        if add_unit {
            let one_v = vdupq_n_f32(1.0);
            while i + 4 <= dim {
                let v = vld1q_f32(ptr.add(i));
                let wv = vld1q_f32(w_ptr.add(i));
                let w_eff = vaddq_f32(one_v, wv);
                vst1q_f32(ptr.add(i), vmulq_f32(vmulq_f32(v, scale_v), w_eff));
                i += 4;
            }
            while i < dim {
                *ptr.add(i) = (*ptr.add(i) * scale) * (1.0 + *w_ptr.add(i));
                i += 1;
            }
        } else {
            while i + 4 <= dim {
                let v = vld1q_f32(ptr.add(i));
                let wv = vld1q_f32(w_ptr.add(i));
                vst1q_f32(ptr.add(i), vmulq_f32(vmulq_f32(v, scale_v), wv));
                i += 4;
            }
            while i < dim {
                *ptr.add(i) = (*ptr.add(i) * scale) * *w_ptr.add(i);
                i += 1;
            }
        }
    }

    /// NEON single-pass out-of-place RMS norm.
    /// Pass 1: read x → copy to out, compute sum-of-squares (saves one memcpy pass)
    /// Pass 2: out[i] = out[i] * scale * w_eff
    #[target_feature(enable = "neon")]
    unsafe fn rms_norm_oop_neon(x: &[f32], out: &mut [f32], w: &[f32], eps: f32, add_unit: bool) {
        let dim = x.len();
        let x_ptr = x.as_ptr();
        let out_ptr = out.as_mut_ptr();
        let w_ptr = w.as_ptr();

        // Pass 1: copy x → out, accumulate sum-of-squares
        let mut sum_v0 = vdupq_n_f32(0.0);
        let mut sum_v1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 8 <= dim {
            let v0 = vld1q_f32(x_ptr.add(i));
            let v1 = vld1q_f32(x_ptr.add(i + 4));
            vst1q_f32(out_ptr.add(i), v0);
            vst1q_f32(out_ptr.add(i + 4), v1);
            sum_v0 = vfmaq_f32(sum_v0, v0, v0);
            sum_v1 = vfmaq_f32(sum_v1, v1, v1);
            i += 8;
        }
        while i + 4 <= dim {
            let v = vld1q_f32(x_ptr.add(i));
            vst1q_f32(out_ptr.add(i), v);
            sum_v0 = vfmaq_f32(sum_v0, v, v);
            i += 4;
        }
        sum_v0 = vaddq_f32(sum_v0, sum_v1);
        let mut sum_sq = vaddvq_f32(sum_v0);
        while i < dim {
            let val = *x_ptr.add(i);
            *out_ptr.add(i) = val;
            sum_sq += val * val;
            i += 1;
        }

        let scale = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        let scale_v = vdupq_n_f32(scale);

        // Pass 2: out[i] *= scale * w_eff
        i = 0;
        if add_unit {
            let one_v = vdupq_n_f32(1.0);
            while i + 4 <= dim {
                let v = vld1q_f32(out_ptr.add(i));
                let wv = vld1q_f32(w_ptr.add(i));
                let w_eff = vaddq_f32(one_v, wv);
                vst1q_f32(out_ptr.add(i), vmulq_f32(vmulq_f32(v, scale_v), w_eff));
                i += 4;
            }
            while i < dim {
                *out_ptr.add(i) = (*out_ptr.add(i) * scale) * (1.0 + *w_ptr.add(i));
                i += 1;
            }
        } else {
            while i + 4 <= dim {
                let v = vld1q_f32(out_ptr.add(i));
                let wv = vld1q_f32(w_ptr.add(i));
                vst1q_f32(out_ptr.add(i), vmulq_f32(vmulq_f32(v, scale_v), wv));
                i += 4;
            }
            while i < dim {
                *out_ptr.add(i) = (*out_ptr.add(i) * scale) * *w_ptr.add(i);
                i += 1;
            }
        }
    }

    /// NEON fused add + out-of-place RMS norm.
    /// Pass 1: x[i] += res[i], copy sum to out, accumulate sum-of-squares
    /// Pass 2: out[i] = out[i] * scale * w_eff
    #[target_feature(enable = "neon")]
    unsafe fn add_rms_norm_oop_neon(
        x: &mut [f32],
        res: &[f32],
        out: &mut [f32],
        w: &[f32],
        eps: f32,
        add_unit: bool,
    ) {
        let dim = x.len();
        let x_ptr = x.as_mut_ptr();
        let res_ptr = res.as_ptr();
        let out_ptr = out.as_mut_ptr();
        let w_ptr = w.as_ptr();

        // Pass 1: x += res, copy to out, accumulate sum-of-squares
        let mut sum_v0 = vdupq_n_f32(0.0);
        let mut sum_v1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 8 <= dim {
            let xv0 = vld1q_f32(x_ptr.add(i));
            let xv1 = vld1q_f32(x_ptr.add(i + 4));
            let rv0 = vld1q_f32(res_ptr.add(i));
            let rv1 = vld1q_f32(res_ptr.add(i + 4));
            let s0 = vaddq_f32(xv0, rv0);
            let s1 = vaddq_f32(xv1, rv1);
            vst1q_f32(x_ptr.add(i), s0);
            vst1q_f32(x_ptr.add(i + 4), s1);
            vst1q_f32(out_ptr.add(i), s0);
            vst1q_f32(out_ptr.add(i + 4), s1);
            sum_v0 = vfmaq_f32(sum_v0, s0, s0);
            sum_v1 = vfmaq_f32(sum_v1, s1, s1);
            i += 8;
        }
        while i + 4 <= dim {
            let xv = vld1q_f32(x_ptr.add(i));
            let rv = vld1q_f32(res_ptr.add(i));
            let s = vaddq_f32(xv, rv);
            vst1q_f32(x_ptr.add(i), s);
            vst1q_f32(out_ptr.add(i), s);
            sum_v0 = vfmaq_f32(sum_v0, s, s);
            i += 4;
        }
        sum_v0 = vaddq_f32(sum_v0, sum_v1);
        let mut sum_sq = vaddvq_f32(sum_v0);
        while i < dim {
            let val = *x_ptr.add(i) + *res_ptr.add(i);
            *x_ptr.add(i) = val;
            *out_ptr.add(i) = val;
            sum_sq += val * val;
            i += 1;
        }

        let scale = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        let scale_v = vdupq_n_f32(scale);

        // Pass 2: out[i] *= scale * w_eff
        i = 0;
        if add_unit {
            let one_v = vdupq_n_f32(1.0);
            while i + 4 <= dim {
                let v = vld1q_f32(out_ptr.add(i));
                let wv = vld1q_f32(w_ptr.add(i));
                let w_eff = vaddq_f32(one_v, wv);
                vst1q_f32(out_ptr.add(i), vmulq_f32(vmulq_f32(v, scale_v), w_eff));
                i += 4;
            }
            while i < dim {
                *out_ptr.add(i) = (*out_ptr.add(i) * scale) * (1.0 + *w_ptr.add(i));
                i += 1;
            }
        } else {
            while i + 4 <= dim {
                let v = vld1q_f32(out_ptr.add(i));
                let wv = vld1q_f32(w_ptr.add(i));
                vst1q_f32(out_ptr.add(i), vmulq_f32(vmulq_f32(v, scale_v), wv));
                i += 4;
            }
            while i < dim {
                *out_ptr.add(i) = (*out_ptr.add(i) * scale) * *w_ptr.add(i);
                i += 1;
            }
        }
    }

    // --- F16 matmul kernels: native F16 FMA (FMLA .8H) with F16 accumulators ---

    /// Convert F32 slice to F16 using NEON fcvtn. One-time cost before the
    /// F16 inner loop (matmul_transposed_f16 pre-converts A row).
    #[target_feature(enable = "neon")]
    unsafe fn f32_to_f16_neon(src: *const f32, dst: *mut u16, len: usize) {
        let mut i = 0;
        while i + 8 <= len {
            let v0 = vld1q_f32(src.add(i));
            let v1 = vld1q_f32(src.add(i + 4));
            let h0: uint16x4_t;
            let h1: uint16x4_t;
            std::arch::asm!("fcvtn {o:v}.4h, {i:v}.4s", o = lateout(vreg) h0, i = in(vreg) v0);
            std::arch::asm!("fcvtn {o:v}.4h, {i:v}.4s", o = lateout(vreg) h1, i = in(vreg) v1);
            vst1_u16(dst.add(i), h0);
            vst1_u16(dst.add(i + 4), h1);
            i += 8;
        }
        while i < len {
            *dst.add(i) = half::f16::from_f32(*src.add(i)).to_bits();
            i += 1;
        }
    }

    /// RM=4 × NR=4 tiled GEMM with native F16 FMA and F16 accumulators.
    /// Processes 4 A rows × 4 B rows = 16 output cells per call.
    ///
    /// Pipeline (8 K-elem/iter): Load=8(4cy), FMA=16(4cy) → balanced @ 4cy
    /// Output: 16 / 4cy = 4.0 per cycle.
    ///
    /// F16 accumulators (uint16x8_t) use half the registers of F32,
    /// enabling the 4×4 tile that would be impossible with F32 (32 regs needed).
    #[target_feature(enable = "neon")]
    unsafe fn vec_dot_f16_native_4x4(
        k: usize,
        a_ptrs: [*const u16; 4],
        b_ptrs: [*const u16; 4],
        out: &mut [[f32; 4]; 4], // out[a_row][b_col]
    ) {
        // 16 F16 accumulators: acc_AxBy = A row x × B col y
        let mut a0b0: uint16x8_t = vdupq_n_u16(0);
        let mut a0b1: uint16x8_t = vdupq_n_u16(0);
        let mut a0b2: uint16x8_t = vdupq_n_u16(0);
        let mut a0b3: uint16x8_t = vdupq_n_u16(0);
        let mut a1b0: uint16x8_t = vdupq_n_u16(0);
        let mut a1b1: uint16x8_t = vdupq_n_u16(0);
        let mut a1b2: uint16x8_t = vdupq_n_u16(0);
        let mut a1b3: uint16x8_t = vdupq_n_u16(0);
        let mut a2b0: uint16x8_t = vdupq_n_u16(0);
        let mut a2b1: uint16x8_t = vdupq_n_u16(0);
        let mut a2b2: uint16x8_t = vdupq_n_u16(0);
        let mut a2b3: uint16x8_t = vdupq_n_u16(0);
        let mut a3b0: uint16x8_t = vdupq_n_u16(0);
        let mut a3b1: uint16x8_t = vdupq_n_u16(0);
        let mut a3b2: uint16x8_t = vdupq_n_u16(0);
        let mut a3b3: uint16x8_t = vdupq_n_u16(0);

        let mut idx = 0;

        // K-loop macro: process 8 K-elements (1 K-step)
        macro_rules! k_step {
            ($off:expr) => {
                let av0: uint16x8_t = vld1q_u16(a_ptrs[0].add($off));
                let av1: uint16x8_t = vld1q_u16(a_ptrs[1].add($off));
                let av2: uint16x8_t = vld1q_u16(a_ptrs[2].add($off));
                let av3: uint16x8_t = vld1q_u16(a_ptrs[3].add($off));

                macro_rules! col {
                                    ($bi:expr, $c0:ident, $c1:ident, $c2:ident, $c3:ident) => {
                                        let bv: uint16x8_t = vld1q_u16(b_ptrs[$bi].add($off));
                                        std::arch::asm!(
                                            "fmla {d0:v}.8h, {a0:v}.8h, {b:v}.8h",
                                            "fmla {d1:v}.8h, {a1:v}.8h, {b:v}.8h",
                                            "fmla {d2:v}.8h, {a2:v}.8h, {b:v}.8h",
                                            "fmla {d3:v}.8h, {a3:v}.8h, {b:v}.8h",
                                            d0 = inout(vreg) $c0, d1 = inout(vreg) $c1,
                                            d2 = inout(vreg) $c2, d3 = inout(vreg) $c3,
                                            a0 = in(vreg) av0, a1 = in(vreg) av1,
                                            a2 = in(vreg) av2, a3 = in(vreg) av3,
                                            b = in(vreg) bv,
                                        );
                                    };
                                }
                col!(0, a0b0, a1b0, a2b0, a3b0);
                col!(1, a0b1, a1b1, a2b1, a3b1);
                col!(2, a0b2, a1b2, a2b2, a3b2);
                col!(3, a0b3, a1b3, a2b3, a3b3);
            };
        }

        // Main loop: 2x unrolled (16 K-elements per iteration)
        // Halves branch overhead; gives OOO engine 48 instructions to schedule
        while idx + 16 <= k {
            k_step!(idx);
            k_step!(idx + 8);
            idx += 16;
        }

        // 8-element tail
        if idx + 8 <= k {
            k_step!(idx);
            idx += 8;
        }

        // Reduce F16 accumulators → F32 scalars
        macro_rules! reduce_f16 {
            ($acc:expr) => {{
                let lo: float32x4_t;
                let hi: float32x4_t;
                std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) lo, i = in(vreg) $acc);
                std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) hi, i = in(vreg) $acc);
                vaddvq_f32(vaddq_f32(lo, hi))
            }};
        }

        out[0] = [
            reduce_f16!(a0b0),
            reduce_f16!(a0b1),
            reduce_f16!(a0b2),
            reduce_f16!(a0b3),
        ];
        out[1] = [
            reduce_f16!(a1b0),
            reduce_f16!(a1b1),
            reduce_f16!(a1b2),
            reduce_f16!(a1b3),
        ];
        out[2] = [
            reduce_f16!(a2b0),
            reduce_f16!(a2b1),
            reduce_f16!(a2b2),
            reduce_f16!(a2b3),
        ];
        out[3] = [
            reduce_f16!(a3b0),
            reduce_f16!(a3b1),
            reduce_f16!(a3b2),
            reduce_f16!(a3b3),
        ];

        // Scalar tail (K not multiple of 8)
        while idx < k {
            for ar in 0..4 {
                let a_val = half::f16::from_bits(*a_ptrs[ar].add(idx)).to_f32();
                for bc in 0..4 {
                    out[ar][bc] += a_val * half::f16::from_bits(*b_ptrs[bc].add(idx)).to_f32();
                }
            }
            idx += 1;
        }
    }

    /// Native F16 FMA GEMV: 1 A row × 4 B rows → 4 F32 outputs.
    ///
    /// Matches llama.cpp's `ggml_vec_dot_f16` instruction stream (`FMLA Vd.8H`)
    /// while retaining our NR=4 A-reuse. Primary decode (M=1 GEMV) F16 matmul
    /// kernel, used by `f16_gemv_chunk` / `fused_gemv_chunk` and the
    /// `f16_gemm_chunk` tail-group fallback.
    ///
    /// Layout: 16 F16 accumulators = 4 B rows × 4 K-slots.
    /// K step = 32 per iter (llama.cpp `GGML_F16_STEP`).
    ///
    /// Instructions per iter: 4 A loads + 16 B loads + 16 FMA = 36 instrs.
    /// The earlier FMLAL-based kernel (widening F16×F16→F32) needed a pair
    /// of `fmlal`/`fmlal2` for the same flops, roughly 1.4× more instructions
    /// in the memory-bound GEMV regime, so the HW prefetcher stalled waiting
    /// for the load front to advance.
    ///
    /// F16 accumulator overflow: max K = 8960 (Qwen down_proj intermediate),
    /// 32 partial lanes → K_eff = 280. Worst case ≤280 × |10| × |2| = 5,600,
    /// well under F16 max 65,504 (~12× margin).
    #[target_feature(enable = "neon")]
    unsafe fn vec_dot_f16_native_gemv_4rows(
        k: usize,
        a_f16: *const u16,
        b_ptrs: [*const u16; 4],
        out: &mut [f32; 4],
    ) {
        // 16 F16 accumulators: acc[row][k_slot] for row 0..3, k_slot 0..3.
        // Typed as uint16x8_t because inline `asm!` treats the bits opaquely;
        // `fmla .8h` reinterprets them as half-precision floats. This matches
        // `vec_dot_f16_native_4x4` and keeps us off the unstable `float16x8_t`
        // intrinsic surface.
        let mut r0s0: uint16x8_t = vdupq_n_u16(0);
        let mut r0s1: uint16x8_t = vdupq_n_u16(0);
        let mut r0s2: uint16x8_t = vdupq_n_u16(0);
        let mut r0s3: uint16x8_t = vdupq_n_u16(0);
        let mut r1s0: uint16x8_t = vdupq_n_u16(0);
        let mut r1s1: uint16x8_t = vdupq_n_u16(0);
        let mut r1s2: uint16x8_t = vdupq_n_u16(0);
        let mut r1s3: uint16x8_t = vdupq_n_u16(0);
        let mut r2s0: uint16x8_t = vdupq_n_u16(0);
        let mut r2s1: uint16x8_t = vdupq_n_u16(0);
        let mut r2s2: uint16x8_t = vdupq_n_u16(0);
        let mut r2s3: uint16x8_t = vdupq_n_u16(0);
        let mut r3s0: uint16x8_t = vdupq_n_u16(0);
        let mut r3s1: uint16x8_t = vdupq_n_u16(0);
        let mut r3s2: uint16x8_t = vdupq_n_u16(0);
        let mut r3s3: uint16x8_t = vdupq_n_u16(0);

        let mut idx = 0;

        // Main loop: 32 F16 elements per iteration (llama.cpp GGML_F16_STEP=32).
        // One A load set (4 × 8 lanes) is shared across all 4 B rows.
        while idx + 32 <= k {
            // Load 4 A vectors (32 f16 lanes total)
            let av0: uint16x8_t = vld1q_u16(a_f16.add(idx));
            let av1: uint16x8_t = vld1q_u16(a_f16.add(idx + 8));
            let av2: uint16x8_t = vld1q_u16(a_f16.add(idx + 16));
            let av3: uint16x8_t = vld1q_u16(a_f16.add(idx + 24));

            // Per-row macro: load 4 B vectors, issue 4 FMA .8h (independent
            // accumulators, independent operands → max ILP).
            macro_rules! fma_row {
                ($bp:expr, $s0:ident, $s1:ident, $s2:ident, $s3:ident) => {
                    let bv0: uint16x8_t = vld1q_u16($bp.add(idx));
                    let bv1: uint16x8_t = vld1q_u16($bp.add(idx + 8));
                    let bv2: uint16x8_t = vld1q_u16($bp.add(idx + 16));
                    let bv3: uint16x8_t = vld1q_u16($bp.add(idx + 24));
                    std::arch::asm!(
                        "fmla {d0:v}.8h, {a0:v}.8h, {b0:v}.8h",
                        "fmla {d1:v}.8h, {a1:v}.8h, {b1:v}.8h",
                        "fmla {d2:v}.8h, {a2:v}.8h, {b2:v}.8h",
                        "fmla {d3:v}.8h, {a3:v}.8h, {b3:v}.8h",
                        d0 = inout(vreg) $s0, d1 = inout(vreg) $s1,
                        d2 = inout(vreg) $s2, d3 = inout(vreg) $s3,
                        a0 = in(vreg) av0, a1 = in(vreg) av1,
                        a2 = in(vreg) av2, a3 = in(vreg) av3,
                        b0 = in(vreg) bv0, b1 = in(vreg) bv1,
                        b2 = in(vreg) bv2, b3 = in(vreg) bv3,
                    );
                };
            }
            fma_row!(b_ptrs[0], r0s0, r0s1, r0s2, r0s3);
            fma_row!(b_ptrs[1], r1s0, r1s1, r1s2, r1s3);
            fma_row!(b_ptrs[2], r2s0, r2s1, r2s2, r2s3);
            fma_row!(b_ptrs[3], r3s0, r3s1, r3s2, r3s3);

            idx += 32;
        }

        // 16-element tail (one half-step): 2 slots per row
        if idx + 16 <= k {
            let av0: uint16x8_t = vld1q_u16(a_f16.add(idx));
            let av1: uint16x8_t = vld1q_u16(a_f16.add(idx + 8));
            macro_rules! fma_row16 {
                ($bp:expr, $s0:ident, $s1:ident) => {
                    let bv0: uint16x8_t = vld1q_u16($bp.add(idx));
                    let bv1: uint16x8_t = vld1q_u16($bp.add(idx + 8));
                    std::arch::asm!(
                        "fmla {d0:v}.8h, {a0:v}.8h, {b0:v}.8h",
                        "fmla {d1:v}.8h, {a1:v}.8h, {b1:v}.8h",
                        d0 = inout(vreg) $s0, d1 = inout(vreg) $s1,
                        a0 = in(vreg) av0, a1 = in(vreg) av1,
                        b0 = in(vreg) bv0, b1 = in(vreg) bv1,
                    );
                };
            }
            fma_row16!(b_ptrs[0], r0s0, r0s1);
            fma_row16!(b_ptrs[1], r1s0, r1s1);
            fma_row16!(b_ptrs[2], r2s0, r2s1);
            fma_row16!(b_ptrs[3], r3s0, r3s1);
            idx += 16;
        }

        // 8-element tail: 1 slot per row
        if idx + 8 <= k {
            let av0: uint16x8_t = vld1q_u16(a_f16.add(idx));
            macro_rules! fma_row8 {
                ($bp:expr, $s0:ident) => {
                    let bv0: uint16x8_t = vld1q_u16($bp.add(idx));
                    std::arch::asm!(
                        "fmla {d:v}.8h, {a:v}.8h, {b:v}.8h",
                        d = inout(vreg) $s0,
                        a = in(vreg) av0,
                        b = in(vreg) bv0,
                    );
                };
            }
            fma_row8!(b_ptrs[0], r0s0);
            fma_row8!(b_ptrs[1], r1s0);
            fma_row8!(b_ptrs[2], r2s0);
            fma_row8!(b_ptrs[3], r3s0);
            idx += 8;
        }

        // Reduce: 4 F16 partials per row → F32 scalar.
        // Each partial is converted via fcvtl/fcvtl2 to two f32x4 vectors and
        // summed in F32 (preserves precision over an in-F16 pre-sum).
        macro_rules! reduce_f16_partial {
            ($acc:expr) => {{
                let lo: float32x4_t;
                let hi: float32x4_t;
                std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) lo, i = in(vreg) $acc);
                std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) hi, i = in(vreg) $acc);
                vaddq_f32(lo, hi)
            }};
        }
        let r0 = vaddq_f32(
            vaddq_f32(reduce_f16_partial!(r0s0), reduce_f16_partial!(r0s1)),
            vaddq_f32(reduce_f16_partial!(r0s2), reduce_f16_partial!(r0s3)),
        );
        let r1 = vaddq_f32(
            vaddq_f32(reduce_f16_partial!(r1s0), reduce_f16_partial!(r1s1)),
            vaddq_f32(reduce_f16_partial!(r1s2), reduce_f16_partial!(r1s3)),
        );
        let r2 = vaddq_f32(
            vaddq_f32(reduce_f16_partial!(r2s0), reduce_f16_partial!(r2s1)),
            vaddq_f32(reduce_f16_partial!(r2s2), reduce_f16_partial!(r2s3)),
        );
        let r3 = vaddq_f32(
            vaddq_f32(reduce_f16_partial!(r3s0), reduce_f16_partial!(r3s1)),
            vaddq_f32(reduce_f16_partial!(r3s2), reduce_f16_partial!(r3s3)),
        );
        out[0] = vaddvq_f32(r0);
        out[1] = vaddvq_f32(r1);
        out[2] = vaddvq_f32(r2);
        out[3] = vaddvq_f32(r3);

        // Scalar tail (K not a multiple of 8)
        while idx < k {
            let a_val = half::f16::from_bits(*a_f16.add(idx)).to_f32();
            for r in 0..4 {
                out[r] += a_val * half::f16::from_bits(*b_ptrs[r].add(idx)).to_f32();
            }
            idx += 1;
        }
    }

    /// Native F16 FMA single-row dot product: A(F16) · B(F16) → scalar F32.
    ///
    /// Used for the `j < j_end` column tail of `f16_gemv_chunk` /
    /// `fused_gemv_chunk` when the chunk width is not a multiple of NR=4
    /// (e.g. tensor partition CPU slices with non-aligned out_dim), and for
    /// the MR=4 column tail / MR<4 tail group in `f16_gemm_chunk`.
    ///
    /// Uses 4 F16 K-slot accumulators following llama.cpp's pattern.
    #[target_feature(enable = "neon")]
    unsafe fn vec_dot_f16_native_gemv_1row(k: usize, a_f16: *const u16, b_ptr: *const u16) -> f32 {
        let mut s0: uint16x8_t = vdupq_n_u16(0);
        let mut s1: uint16x8_t = vdupq_n_u16(0);
        let mut s2: uint16x8_t = vdupq_n_u16(0);
        let mut s3: uint16x8_t = vdupq_n_u16(0);

        let mut idx = 0;

        while idx + 32 <= k {
            let av0: uint16x8_t = vld1q_u16(a_f16.add(idx));
            let av1: uint16x8_t = vld1q_u16(a_f16.add(idx + 8));
            let av2: uint16x8_t = vld1q_u16(a_f16.add(idx + 16));
            let av3: uint16x8_t = vld1q_u16(a_f16.add(idx + 24));
            let bv0: uint16x8_t = vld1q_u16(b_ptr.add(idx));
            let bv1: uint16x8_t = vld1q_u16(b_ptr.add(idx + 8));
            let bv2: uint16x8_t = vld1q_u16(b_ptr.add(idx + 16));
            let bv3: uint16x8_t = vld1q_u16(b_ptr.add(idx + 24));
            std::arch::asm!(
                "fmla {d0:v}.8h, {a0:v}.8h, {b0:v}.8h",
                "fmla {d1:v}.8h, {a1:v}.8h, {b1:v}.8h",
                "fmla {d2:v}.8h, {a2:v}.8h, {b2:v}.8h",
                "fmla {d3:v}.8h, {a3:v}.8h, {b3:v}.8h",
                d0 = inout(vreg) s0, d1 = inout(vreg) s1,
                d2 = inout(vreg) s2, d3 = inout(vreg) s3,
                a0 = in(vreg) av0, a1 = in(vreg) av1,
                a2 = in(vreg) av2, a3 = in(vreg) av3,
                b0 = in(vreg) bv0, b1 = in(vreg) bv1,
                b2 = in(vreg) bv2, b3 = in(vreg) bv3,
            );
            idx += 32;
        }

        if idx + 16 <= k {
            let av0: uint16x8_t = vld1q_u16(a_f16.add(idx));
            let av1: uint16x8_t = vld1q_u16(a_f16.add(idx + 8));
            let bv0: uint16x8_t = vld1q_u16(b_ptr.add(idx));
            let bv1: uint16x8_t = vld1q_u16(b_ptr.add(idx + 8));
            std::arch::asm!(
                "fmla {d0:v}.8h, {a0:v}.8h, {b0:v}.8h",
                "fmla {d1:v}.8h, {a1:v}.8h, {b1:v}.8h",
                d0 = inout(vreg) s0, d1 = inout(vreg) s1,
                a0 = in(vreg) av0, a1 = in(vreg) av1,
                b0 = in(vreg) bv0, b1 = in(vreg) bv1,
            );
            idx += 16;
        }

        if idx + 8 <= k {
            let av0: uint16x8_t = vld1q_u16(a_f16.add(idx));
            let bv0: uint16x8_t = vld1q_u16(b_ptr.add(idx));
            std::arch::asm!(
                "fmla {d:v}.8h, {a:v}.8h, {b:v}.8h",
                d = inout(vreg) s0,
                a = in(vreg) av0,
                b = in(vreg) bv0,
            );
            idx += 8;
        }

        // Reduce 4 F16 partials → F32 scalar (sum in F32 for precision)
        macro_rules! reduce_f16_partial {
            ($acc:expr) => {{
                let lo: float32x4_t;
                let hi: float32x4_t;
                std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", o = lateout(vreg) lo, i = in(vreg) $acc);
                std::arch::asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = lateout(vreg) hi, i = in(vreg) $acc);
                vaddq_f32(lo, hi)
            }};
        }
        let v = vaddq_f32(
            vaddq_f32(reduce_f16_partial!(s0), reduce_f16_partial!(s1)),
            vaddq_f32(reduce_f16_partial!(s2), reduce_f16_partial!(s3)),
        );
        let mut sum = vaddvq_f32(v);

        while idx < k {
            sum += half::f16::from_bits(*a_f16.add(idx)).to_f32()
                * half::f16::from_bits(*b_ptr.add(idx)).to_f32();
            idx += 1;
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

        let a_data = a.as_slice::<f32>();
        let nb_k = k / QK4_0;
        let out_data = out.as_mut_slice::<f32>();
        let nb_k_q8 = k / QK8_0;
        let total_q8_blocks = m * nb_k_q8;

        // Reuse thread-local workspace to avoid per-matmul allocation.
        let a_q8_ptr: SendSyncPtr = Q8_WORKSPACE.with(|ws| {
            let mut a_q8 = ws.borrow_mut();
            if a_q8.len() < total_q8_blocks {
                let additional = total_q8_blocks - a_q8.len();
                a_q8.reserve(additional);
                unsafe {
                    a_q8.set_len(total_q8_blocks);
                }
            }

            for i in 0..m {
                let a_offset = i * k;
                let a_row = &a_data[a_offset..a_offset + k];
                let q8_row = &mut a_q8[i * nb_k_q8..(i + 1) * nb_k_q8];
                unsafe {
                    self.quantize_row_q8_0(a_row, q8_row, k);
                }
            }

            SendSyncPtr(a_q8.as_ptr())
        });

        // Hoist feature detection outside closure
        #[cfg(target_arch = "aarch64")]
        let use_i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
        #[cfg(not(target_arch = "aarch64"))]
        let use_i8mm = false;

        #[cfg(target_arch = "aarch64")]
        let use_dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        #[cfg(not(target_arch = "aarch64"))]
        let use_dotprod = false;

        let num_threads = rayon::current_num_threads();

        if m == 1 {
            // === GEMV: adaptive chunk size, flat N-parallel ===
            let target_tasks = num_threads * 4;
            let chunk_size = {
                let cs = (n + target_tasks - 1) / target_tasks;
                (cs + 1) & !1 // round up to even for i8mm
            };

            out_data.par_chunks_mut(chunk_size).enumerate().for_each(
                |(chunk_idx, chunk): (usize, &mut [f32])| {
                    let start_idx = chunk_idx * chunk_size;
                    let b_base = b.as_ptr() as *const BlockQ4_0;
                    let a_ptr = unsafe { a_q8_ptr.ptr() };

                    if use_i8mm {
                        let mut li = 0;
                        while li < chunk.len() {
                            let j = start_idx + li;
                            if li + 1 < chunk.len() && j + 1 < n {
                                let (left, right) = chunk[li..].split_at_mut(1);
                                unsafe {
                                    self.vec_dot_q4_0_q8_0_i8mm(
                                        k, &mut left[0], &mut right[0],
                                        b_base.add(j * nb_k), b_base.add((j + 1) * nb_k), a_ptr,
                                    );
                                }
                                li += 2;
                            } else {
                                let mut sum = 0.0;
                                unsafe { self.vec_dot_q4_0_q8_0_sdot(k, &mut sum, b_base.add(j * nb_k), a_ptr) }
                                chunk[li] = sum;
                                li += 1;
                            }
                        }
                    } else if use_dotprod {
                        for (li, out_val) in chunk.iter_mut().enumerate() {
                            let mut sum = 0.0;
                            unsafe { self.vec_dot_q4_0_q8_0_sdot(k, &mut sum, b_base.add((start_idx + li) * nb_k), a_ptr) }
                            *out_val = sum;
                        }
                    } else {
                        for (li, out_val) in chunk.iter_mut().enumerate() {
                            let mut sum = 0.0;
                            unsafe { self.vec_dot_q4_0_q8_0(k, &mut sum, b_base.add((start_idx + li) * nb_k), a_ptr) }
                            *out_val = sum;
                        }
                    }
                },
            );
        } else {
            // === GEMM (M>1): N-major tiled ===
            // Weight rows are shared across M activation rows within each N-chunk,
            // improving cache reuse. Mirrors the F16 GEMM tiling strategy.
            const NR: usize = 2; // i8mm natural pair
            let target_n_chunks = num_threads * 4;
            let rows_per_chunk = ((n + target_n_chunks - 1) / target_n_chunks + NR - 1) / NR * NR;
            let rows_per_chunk = rows_per_chunk.max(NR);
            let n_chunks = (n + rows_per_chunk - 1) / rows_per_chunk;

            let b_base_usize = b.as_ptr() as usize;
            let a_q8_usize = a_q8_ptr.ptr() as usize;
            let out_ptr_usize = out_data.as_mut_ptr() as usize;

            (0..n_chunks).into_par_iter().for_each(|chunk_idx| {
                let j_start = chunk_idx * rows_per_chunk;
                let j_end = (j_start + rows_per_chunk).min(n);
                let b_base = b_base_usize as *const BlockQ4_0;
                let a_q8_base = a_q8_usize as *const BlockQ8_0;
                let out_base = out_ptr_usize as *mut f32;
                let backend = CpuBackendNeon::new();

                let mut j = j_start;
                while j < j_end {
                    if use_i8mm && j + 1 < j_end {
                        // i8mm: 2 weight rows × all M activation rows
                        let b0 = unsafe { b_base.add(j * nb_k) };
                        let b1 = unsafe { b_base.add((j + 1) * nb_k) };
                        for i in 0..m {
                            let a_row = unsafe { a_q8_base.add(i * nb_k_q8) };
                            let mut s0 = 0.0f32;
                            let mut s1 = 0.0f32;
                            unsafe {
                                backend.vec_dot_q4_0_q8_0_i8mm(k, &mut s0, &mut s1, b0, b1, a_row);
                                *out_base.add(i * n + j) = s0;
                                *out_base.add(i * n + j + 1) = s1;
                            }
                        }
                        j += 2;
                    } else {
                        let b_ptr = unsafe { b_base.add(j * nb_k) };
                        for i in 0..m {
                            let a_row = unsafe { a_q8_base.add(i * nb_k_q8) };
                            let mut sum = 0.0f32;
                            unsafe {
                                if use_dotprod {
                                    backend.vec_dot_q4_0_q8_0_sdot(k, &mut sum, b_ptr, a_row);
                                } else {
                                    backend.vec_dot_q4_0_q8_0(k, &mut sum, b_ptr, a_row);
                                }
                                *out_base.add(i * n + j) = sum;
                            }
                        }
                        j += 1;
                    }
                }
            });
        }
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
    // 2-block unrolling with vector FMA accumulation (matches llama.cpp pattern).
    // Key optimisation: dot products stay in int32x4_t lanes, converted to float32x4_t
    // and accumulated via vmlaq_n_f32, avoiding per-block scalar reduction (vaddvq_s32).
    // Final horizontal reduction happens once after the loop.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn vec_dot_q4_0_q8_0_sdot(
        &self,
        n: usize,
        s: &mut f32,
        vx: *const BlockQ4_0,
        vy: *const BlockQ8_0,
    ) {
        let nb = n / QK8_0;

        unsafe {
            let m4b = vdupq_n_u8(0x0F);
            let s8b = vdupq_n_s8(8);

            let mut sumv0 = vdupq_n_f32(0.0);
            let mut sumv1 = vdupq_n_f32(0.0);

            // Main loop: process 2 blocks per iteration
            let mut i = 0;
            while i + 1 < nb {
                let x0 = &*vx.add(i);
                let x1 = &*vx.add(i + 1);
                let y0 = &*vy.add(i);
                let y1 = &*vy.add(i + 1);

                let d0 = x0.d.to_f32() * y0.d.to_f32();
                let d1 = x1.d.to_f32() * y1.d.to_f32();

                // Load and unpack Q4_0 block 0
                let v0_0 = vld1q_u8(x0.qs.as_ptr());
                let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
                let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4)), s8b);

                // Load and unpack Q4_0 block 1
                let v0_1 = vld1q_u8(x1.qs.as_ptr());
                let x1_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_1, m4b)), s8b);
                let x1_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4)), s8b);

                // Load Q8_0 block 0
                let y0_l = vld1q_s8(y0.qs.as_ptr());
                let y0_h = vld1q_s8(y0.qs.as_ptr().add(16));

                // Load Q8_0 block 1
                let y1_l = vld1q_s8(y1.qs.as_ptr());
                let y1_h = vld1q_s8(y1.qs.as_ptr().add(16));

                // Dot product block 0: sdot low then high into p_0
                let mut p_0 = vdupq_n_s32(0);
                std::arch::asm!(
                    "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                    acc = inout(vreg) p_0,
                    x = in(vreg) x0_l,
                    y = in(vreg) y0_l,
                );
                std::arch::asm!(
                    "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                    acc = inout(vreg) p_0,
                    x = in(vreg) x0_h,
                    y = in(vreg) y0_h,
                );

                // Dot product block 1: sdot low then high into p_1
                let mut p_1 = vdupq_n_s32(0);
                std::arch::asm!(
                    "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                    acc = inout(vreg) p_1,
                    x = in(vreg) x1_l,
                    y = in(vreg) y1_l,
                );
                std::arch::asm!(
                    "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                    acc = inout(vreg) p_1,
                    x = in(vreg) x1_h,
                    y = in(vreg) y1_h,
                );

                // Vector FMA: accumulate int32x4_t → float32x4_t with scalar scale
                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
                sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);

                i += 2;
            }

            // Horizontal reduction (single pass at end)
            let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

            // Handle odd remainder block (when nb is odd)
            if i < nb {
                let x = &*vx.add(i);
                let y = &*vy.add(i);
                let d = x.d.to_f32() * y.d.to_f32();

                let v0 = vld1q_u8(x.qs.as_ptr());
                let x_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4b)), s8b);
                let x_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), s8b);

                let y_l = vld1q_s8(y.qs.as_ptr());
                let y_h = vld1q_s8(y.qs.as_ptr().add(16));

                let mut acc = vdupq_n_s32(0);
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

                sumf += d * vaddvq_s32(acc) as f32;
            }

            *s = sumf;
        }
    }

    /// 4-row sdot dot product: weight 4행 × activation 1행 → 결과 4개.
    /// Activation Q8_0 blocks are loaded once and reused across 4 weight rows,
    /// reducing memory bandwidth by ~4x in the GEMV (M=1) regime.
    ///
    /// Each iteration processes 2 Q4_0/Q8_0 block pairs (64 values) to keep
    /// the 8 accumulator registers busy and hide sdot latency.
    ///
    /// # Safety
    /// - `vx[0..4]` each point to `n / QK8_0` valid `BlockQ4_0` blocks
    /// - `vy` points to `n / QK8_0` valid `BlockQ8_0` blocks
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn vec_dot_q4_0_q8_0_sdot_4rows(
        &self,
        n: usize,
        vx: [*const BlockQ4_0; 4],
        vy: *const BlockQ8_0,
        out: &mut [f32; 4],
    ) {
        let nb = n / QK8_0;

        unsafe {
            let m4b = vdupq_n_u8(0x0F);
            let s8b = vdupq_n_s8(8);

            // 8 accumulators: 2 per row (for 2-block unrolling)
            let mut sum0a = vdupq_n_f32(0.0);
            let mut sum0b = vdupq_n_f32(0.0);
            let mut sum1a = vdupq_n_f32(0.0);
            let mut sum1b = vdupq_n_f32(0.0);
            let mut sum2a = vdupq_n_f32(0.0);
            let mut sum2b = vdupq_n_f32(0.0);
            let mut sum3a = vdupq_n_f32(0.0);
            let mut sum3b = vdupq_n_f32(0.0);

            let mut i = 0;
            while i + 1 < nb {
                // Load activation Q8_0 — shared across all 4 weight rows
                let y0 = &*vy.add(i);
                let y1 = &*vy.add(i + 1);
                let y0_l = vld1q_s8(y0.qs.as_ptr());
                let y0_h = vld1q_s8(y0.qs.as_ptr().add(16));
                let y1_l = vld1q_s8(y1.qs.as_ptr());
                let y1_h = vld1q_s8(y1.qs.as_ptr().add(16));

                // Process each of the 4 weight rows
                macro_rules! do_row {
                    ($row:expr, $suma:ident, $sumb:ident) => {
                        let x0 = &*vx[$row].add(i);
                        let x1 = &*vx[$row].add(i + 1);

                        // Block 0: unpack Q4_0 nibbles
                        let v0_0 = vld1q_u8(x0.qs.as_ptr());
                        let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
                        let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4)), s8b);

                        let d0 = x0.d.to_f32() * y0.d.to_f32();

                        let mut p_0 = vdupq_n_s32(0);
                        std::arch::asm!(
                            "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                            acc = inout(vreg) p_0,
                            x = in(vreg) x0_l,
                            y = in(vreg) y0_l,
                        );
                        std::arch::asm!(
                            "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                            acc = inout(vreg) p_0,
                            x = in(vreg) x0_h,
                            y = in(vreg) y0_h,
                        );
                        $suma = vmlaq_n_f32($suma, vcvtq_f32_s32(p_0), d0);

                        // Block 1: unpack Q4_0 nibbles
                        let v0_1 = vld1q_u8(x1.qs.as_ptr());
                        let x1_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_1, m4b)), s8b);
                        let x1_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4)), s8b);

                        let d1 = x1.d.to_f32() * y1.d.to_f32();

                        let mut p_1 = vdupq_n_s32(0);
                        std::arch::asm!(
                            "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                            acc = inout(vreg) p_1,
                            x = in(vreg) x1_l,
                            y = in(vreg) y1_l,
                        );
                        std::arch::asm!(
                            "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                            acc = inout(vreg) p_1,
                            x = in(vreg) x1_h,
                            y = in(vreg) y1_h,
                        );
                        $sumb = vmlaq_n_f32($sumb, vcvtq_f32_s32(p_1), d1);
                    };
                }

                do_row!(0, sum0a, sum0b);
                do_row!(1, sum1a, sum1b);
                do_row!(2, sum2a, sum2b);
                do_row!(3, sum3a, sum3b);

                i += 2;
            }

            // Handle odd remainder block
            if i < nb {
                let y0 = &*vy.add(i);
                let y0_l = vld1q_s8(y0.qs.as_ptr());
                let y0_h = vld1q_s8(y0.qs.as_ptr().add(16));

                macro_rules! do_row_tail {
                    ($row:expr, $suma:ident) => {
                        let x0 = &*vx[$row].add(i);
                        let v0_0 = vld1q_u8(x0.qs.as_ptr());
                        let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
                        let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4)), s8b);
                        let d0 = x0.d.to_f32() * y0.d.to_f32();
                        let mut p_0 = vdupq_n_s32(0);
                        std::arch::asm!(
                            "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                            acc = inout(vreg) p_0,
                            x = in(vreg) x0_l,
                            y = in(vreg) y0_l,
                        );
                        std::arch::asm!(
                            "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
                            acc = inout(vreg) p_0,
                            x = in(vreg) x0_h,
                            y = in(vreg) y0_h,
                        );
                        $suma = vmlaq_n_f32($suma, vcvtq_f32_s32(p_0), d0);
                    };
                }

                do_row_tail!(0, sum0a);
                do_row_tail!(1, sum1a);
                do_row_tail!(2, sum2a);
                do_row_tail!(3, sum3a);
            }

            // Horizontal reduction
            out[0] = vaddvq_f32(vaddq_f32(sum0a, sum0b));
            out[1] = vaddvq_f32(vaddq_f32(sum1a, sum1b));
            out[2] = vaddvq_f32(vaddq_f32(sum2a, sum2b));
            out[3] = vaddvq_f32(vaddq_f32(sum3a, sum3b));
        }
    }

    /// 2-row i8mm dot product: weight 2행 × activation 1행 → 결과 2개.
    /// ARM FEAT_I8MM (smmla) 기반. i8mm 미지원 디바이스에서는 호출하지 않을 것.
    ///
    /// `smmla` 시맨틱 (2x8x2 matrix multiply-accumulate):
    ///   acc[0] += dot(a[0:7],  b[0:7])   — weight row 0 × activation low
    ///   acc[1] += dot(a[0:7],  b[8:15])  — weight row 0 × activation high (duplicate)
    ///   acc[2] += dot(a[8:15], b[0:7])   — weight row 1 × activation low
    ///   acc[3] += dot(a[8:15], b[8:15])  — weight row 1 × activation high (duplicate)
    ///
    /// GEMV에서 activation row가 1개이므로 b의 양쪽 절반에 동일 데이터를 복제.
    /// scale = [d_x0*d_y, 0, d_x1*d_y, 0]으로 중복 lane을 제거하여
    /// sumv[0] = row 0 결과, sumv[2] = row 1 결과를 누적.
    ///
    /// # Safety
    /// - `vx0`, `vx1`은 각각 `n / QK8_0`개의 `BlockQ4_0`을 가리키는 유효 포인터
    /// - `vy`는 `n / QK8_0`개의 `BlockQ8_0`을 가리키는 유효 포인터
    /// - 호출자가 FEAT_I8MM 지원을 런타임에 확인해야 함
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
        unsafe {
            let nb = n / QK8_0;
            let m4b = vdupq_n_u8(0x0F);
            let s8b = vdupq_n_s8(0x08);

            let mut sumv = vdupq_n_f32(0.0);

            for i in 0..nb {
                let b_x0 = &*vx0.add(i);
                let b_x1 = &*vx1.add(i);
                let b_y = &*vy.add(i);

                // Unpack Q4_0 nibbles → signed int8 (subtract bias 8)
                let v0_0 = vld1q_u8(b_x0.qs.as_ptr());
                let v0_1 = vld1q_u8(b_x1.qs.as_ptr());

                let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
                let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4)), s8b);
                let x1_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_1, m4b)), s8b);
                let x1_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4)), s8b);

                // Load Q8_0 activation
                let y_l = vld1q_s8(b_y.qs.as_ptr());
                let y_h = vld1q_s8(b_y.qs.as_ptr().add(16));

                // Scale: [d_x0*d_y, 0, d_x1*d_y, 0] — zero lanes mask out duplicate smmla results
                let d_x0 = b_x0.d.to_f32();
                let d_x1 = b_x1.d.to_f32();
                let d_y = b_y.d.to_f32();
                let _scale = [d_x0 * d_y, 0.0f32, d_x1 * d_y, 0.0f32];
                let scale = vld1q_f32(_scale.as_ptr());

                // Interleave weight rows: a[0:7] = x0 half, a[8:15] = x1 half
                let l0 = vreinterpretq_s8_s64(vzip1q_s64(
                    vreinterpretq_s64_s8(x0_l),
                    vreinterpretq_s64_s8(x1_l),
                ));
                let l1 = vreinterpretq_s8_s64(vzip2q_s64(
                    vreinterpretq_s64_s8(x0_l),
                    vreinterpretq_s64_s8(x1_l),
                ));
                let l2 = vreinterpretq_s8_s64(vzip1q_s64(
                    vreinterpretq_s64_s8(x0_h),
                    vreinterpretq_s64_s8(x1_h),
                ));
                let l3 = vreinterpretq_s8_s64(vzip2q_s64(
                    vreinterpretq_s64_s8(x0_h),
                    vreinterpretq_s64_s8(x1_h),
                ));

                // Duplicate activation for both halves (GEMV: single activation row replicated)
                let r0 = vreinterpretq_s8_s64(vzip1q_s64(
                    vreinterpretq_s64_s8(y_l),
                    vreinterpretq_s64_s8(y_l),
                ));
                let r1 = vreinterpretq_s8_s64(vzip2q_s64(
                    vreinterpretq_s64_s8(y_l),
                    vreinterpretq_s64_s8(y_l),
                ));
                let r2 = vreinterpretq_s8_s64(vzip1q_s64(
                    vreinterpretq_s64_s8(y_h),
                    vreinterpretq_s64_s8(y_h),
                ));
                let r3 = vreinterpretq_s8_s64(vzip2q_s64(
                    vreinterpretq_s64_s8(y_h),
                    vreinterpretq_s64_s8(y_h),
                ));

                // 4 chained smmla: accumulate 32 int8 dot products per weight row.
                // .arch_extension i8mm enables smmla even when the default target
                // does not include +i8mm (runtime detection guards the call site).
                let mut acc = vdupq_n_s32(0);
                std::arch::asm!(
                    ".arch_extension i8mm",
                    "smmla {acc:v}.4s, {a0:v}.16b, {b0:v}.16b",
                    "smmla {acc:v}.4s, {a1:v}.16b, {b1:v}.16b",
                    "smmla {acc:v}.4s, {a2:v}.16b, {b2:v}.16b",
                    "smmla {acc:v}.4s, {a3:v}.16b, {b3:v}.16b",
                    acc = inout(vreg) acc,
                    a0 = in(vreg) l0,
                    b0 = in(vreg) r0,
                    a1 = in(vreg) l1,
                    b1 = in(vreg) r1,
                    a2 = in(vreg) l2,
                    b2 = in(vreg) r2,
                    a3 = in(vreg) l3,
                    b3 = in(vreg) r3,
                );

                // acc = [dot(x0,y), dot(x0,y), dot(x1,y), dot(x1,y)]
                // vmlaq with scale zeroes duplicate lanes → sumv[0] += x0, sumv[2] += x1
                sumv = vmlaq_f32(sumv, vcvtq_f32_s32(acc), scale);
            }

            *s0 = vgetq_lane_f32(sumv, 0);
            *s1 = vgetq_lane_f32(sumv, 2);
        }
    }
}

// --- SpinPool work context for F16 GEMV ---

#[repr(C)]
struct F16GemvCtx {
    a_ptr: *const f32,
    a_f16_ptr: *const u16, // Pre-converted A in F16 for the F16 matmul kernels
    b_base: *const u16,
    out_ptr: *mut f32,
    k: usize,
    n: usize,
    rows_per_chunk: usize,
}

// --- SpinPool work context for Q4_0 GEMV ---

#[repr(C)]
struct Q4GemvCtx {
    a_q8_ptr: *const BlockQ8_0, // Pre-quantized activation in Q8_0
    b_base: *const BlockQ4_0,   // Weight matrix [N, K/32] in Q4_0
    out_ptr: *mut f32,          // Output [N]
    k: usize,
    n: usize,
    nb_k: usize,               // K / QK4_0 = number of Q4_0 blocks per row
    rows_per_chunk: usize,
}

unsafe impl Send for Q4GemvCtx {}
unsafe impl Sync for Q4GemvCtx {}

// --- N-major GEMM context (prefill, M > 1) ---

/// Context for N-major GEMM dispatch.
/// Instead of dispatching M times (one per activation row), we dispatch once
/// with M × n_chunks tasks. Consecutive chunk_ids share the same B rows,
/// so the same core reuses B data from L1 across all M activation rows.
#[repr(C)]
struct F16GemmCtx {
    a_base: *const f32,     // A[M, K] row-major
    a_f16_base: *const u16, // Pre-converted A[M, K] in F16 for F16 matmul kernels
    b_base: *const u16,     // B[N, K] row-major (F16, transposed)
    out_base: *mut f32,     // Out[M, N] row-major
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
    a_f16_ptr: *const u16, // Pre-converted A in F16 for the F16 matmul kernels
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
        CpuBackendNeon::vec_dot_f16_native_gemv_4rows(ctx.k, ctx.a_f16_ptr, b_ptrs, &mut results);
        std::ptr::copy_nonoverlapping(results.as_ptr(), out_ptr.add(j), NR);
        j += NR;
    }
    while j < j_end {
        *out_ptr.add(j) = CpuBackendNeon::vec_dot_f16_native_gemv_1row(
            ctx.k,
            ctx.a_f16_ptr,
            b_base.add(j * ctx.k),
        );
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

    // Pre-convert A from F32 to F16 for the F16 matmul kernels
    let mut a_f16_buf: Vec<u16> = vec![0u16; k];
    CpuBackendNeon::f32_to_f16_neon(a_data, a_f16_buf.as_mut_ptr(), k);

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
        a_f16_ptr: a_f16_buf.as_ptr(),
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

/// Fused multi-matmul for Q4_0 decode (M=1): single Q8_0 quantization + single Rayon dispatch.
/// Combines gate+up (or QKV) into one parallel region, avoiding redundant quantization
/// and Rayon dispatch overhead.
///
/// `matmuls`: slice of (weight_base, out_ptr, n_rows) — up to 3 matmuls.
///
/// # Safety
/// All weight pointers must be valid BlockQ4_0 arrays. All out_ptrs must have capacity >= n_rows.
/// `a_data` must point to a valid f32 slice of length `k`.
pub unsafe fn fused_matmul_q4_0(
    a_data: *const f32,
    k: usize,
    matmuls: &[(*const BlockQ4_0, *mut f32, usize)],
) {
    let nb_k = k / QK4_0;
    let nb_k_q8 = k / QK8_0;
    let backend = CpuBackendNeon::new();

    // 1. Quantize activation to Q8_0 — once for all matmuls (reuse thread-local workspace)
    let a_q8_usize: usize = Q8_WORKSPACE.with(|ws| {
        let mut a_q8 = ws.borrow_mut();
        if a_q8.len() < nb_k_q8 {
            let additional = nb_k_q8 - a_q8.len();
            a_q8.reserve(additional);
            a_q8.set_len(nb_k_q8);
        }
        let a_row = std::slice::from_raw_parts(a_data, k);
        backend.quantize_row_q8_0(a_row, &mut a_q8[..nb_k_q8], k);
        a_q8.as_ptr() as usize
    });

    // 2. Build fused context — cast pointers to usize for Rayon Send
    let mut b_bases = [0usize; 3];
    let mut out_ptrs = [0usize; 3];
    let mut ns = [0usize; 3];
    let n_matmuls = matmuls.len();

    for (i, &(b, o, n)) in matmuls.iter().enumerate().take(3) {
        b_bases[i] = b as usize;
        out_ptrs[i] = o as usize;
        ns[i] = n;
    }

    // Build cumulative row offsets for matmul boundaries
    let mut row_offsets = [0usize; 4];
    for i in 0..n_matmuls {
        row_offsets[i + 1] = row_offsets[i] + ns[i];
    }
    let total_rows = row_offsets[n_matmuls];

    // Adaptive chunk size: target ~4 tasks per thread
    let num_threads = rayon::current_num_threads();
    let target_tasks = num_threads * 4;
    let chunk_size = {
        let cs = (total_rows + target_tasks - 1) / target_tasks;
        (cs + 1) & !1 // round up to even for i8mm 2-row pairing
    };
    let n_chunks = (total_rows + chunk_size - 1) / chunk_size;

    #[cfg(target_arch = "aarch64")]
    let use_i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
    #[cfg(not(target_arch = "aarch64"))]
    let use_i8mm = false;

    // 3. Rayon dispatch: write directly to output pointers (no intermediate buffer)
    (0..n_chunks).into_par_iter().for_each(|chunk_idx| {
        let global_start = chunk_idx * chunk_size;
        let global_end = (global_start + chunk_size).min(total_rows);
        let a_q8_ptr = a_q8_usize as *const BlockQ8_0;

        if use_i8mm {
            let mut gi = global_start;
            while gi < global_end {
                let mut mat_idx = 0;
                while mat_idx + 1 < n_matmuls && gi >= row_offsets[mat_idx + 1] {
                    mat_idx += 1;
                }
                let j = gi - row_offsets[mat_idx];
                let b_base = b_bases[mat_idx] as *const BlockQ4_0;
                let out_base = out_ptrs[mat_idx] as *mut f32;

                // i8mm 2-row: pair adjacent rows within same matmul
                if gi + 1 < global_end && gi + 1 < row_offsets[mat_idx + 1] {
                    let b0 = unsafe { b_base.add(j * nb_k) };
                    let b1 = unsafe { b_base.add((j + 1) * nb_k) };
                    let mut s0 = 0.0f32;
                    let mut s1 = 0.0f32;
                    unsafe {
                        backend.vec_dot_q4_0_q8_0_i8mm(
                            k, &mut s0, &mut s1, b0, b1, a_q8_ptr,
                        );
                        *out_base.add(j) = s0;
                        *out_base.add(j + 1) = s1;
                    }
                    gi += 2;
                } else {
                    let b_ptr = unsafe { b_base.add(j * nb_k) };
                    let mut sum = 0.0f32;
                    unsafe {
                        backend.vec_dot_q4_0_q8_0_sdot(k, &mut sum, b_ptr, a_q8_ptr);
                        *out_base.add(j) = sum;
                    }
                    gi += 1;
                }
            }
        } else {
            for gi in global_start..global_end {
                let mut mat_idx = 0;
                while mat_idx + 1 < n_matmuls && gi >= row_offsets[mat_idx + 1] {
                    mat_idx += 1;
                }
                let j = gi - row_offsets[mat_idx];
                let b_base = b_bases[mat_idx] as *const BlockQ4_0;
                let out_base = out_ptrs[mat_idx] as *mut f32;
                let b_ptr = unsafe { b_base.add(j * nb_k) };

                let mut sum = 0.0f32;
                unsafe {
                    if std::arch::is_aarch64_feature_detected!("dotprod") {
                        backend.vec_dot_q4_0_q8_0_sdot(k, &mut sum, b_ptr, a_q8_ptr);
                    } else {
                        backend.vec_dot_q4_0_q8_0(k, &mut sum, b_ptr, a_q8_ptr);
                    }
                    *out_base.add(j) = sum;
                }
            }
        }
    });
}

/// Work function for SpinPool: Q4_0 GEMV with batched dot product.
/// Uses i8mm (smmla, 2-row) when available, falls back to sdot (4-row).
///
/// # Safety
/// Called by SpinPool workers with valid Q4GemvCtx pointer.
unsafe fn q4_gemv_chunk(ctx_ptr: *const u8, chunk_id: usize) {
    let ctx = &*(ctx_ptr as *const Q4GemvCtx);
    let j_start = chunk_id * ctx.rows_per_chunk;
    let j_end = (j_start + ctx.rows_per_chunk).min(ctx.n);

    let backend = CpuBackendNeon::new();

    #[cfg(target_arch = "aarch64")]
    let use_i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
    #[cfg(not(target_arch = "aarch64"))]
    let use_i8mm = false;

    if use_i8mm {
        // i8mm path: 2-row batches using smmla instruction
        let mut j = j_start;
        while j + 1 < j_end {
            let mut s0 = 0.0f32;
            let mut s1 = 0.0f32;
            backend.vec_dot_q4_0_q8_0_i8mm(
                ctx.k,
                &mut s0,
                &mut s1,
                ctx.b_base.add(j * ctx.nb_k),
                ctx.b_base.add((j + 1) * ctx.nb_k),
                ctx.a_q8_ptr,
            );
            *ctx.out_ptr.add(j) = s0;
            *ctx.out_ptr.add(j + 1) = s1;
            j += 2;
        }
        if j < j_end {
            let mut sum = 0.0f32;
            backend.vec_dot_q4_0_q8_0_sdot(
                ctx.k,
                &mut sum,
                ctx.b_base.add(j * ctx.nb_k),
                ctx.a_q8_ptr,
            );
            *ctx.out_ptr.add(j) = sum;
        }
    } else {
        // sdot path: 4-row batches
        const NR: usize = 4;
        let mut j = j_start;
        while j + NR <= j_end {
            let vx = [
                ctx.b_base.add(j * ctx.nb_k),
                ctx.b_base.add((j + 1) * ctx.nb_k),
                ctx.b_base.add((j + 2) * ctx.nb_k),
                ctx.b_base.add((j + 3) * ctx.nb_k),
            ];
            let mut results = [0.0f32; NR];
            backend.vec_dot_q4_0_q8_0_sdot_4rows(ctx.k, vx, ctx.a_q8_ptr, &mut results);
            std::ptr::copy_nonoverlapping(results.as_ptr(), ctx.out_ptr.add(j), NR);
            j += NR;
        }
        while j < j_end {
            let mut sum = 0.0f32;
            backend.vec_dot_q4_0_q8_0_sdot(
                ctx.k,
                &mut sum,
                ctx.b_base.add(j * ctx.nb_k),
                ctx.a_q8_ptr,
            );
            *ctx.out_ptr.add(j) = sum;
            j += 1;
        }
    }
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
        CpuBackendNeon::vec_dot_f16_native_gemv_4rows(ctx.k, ctx.a_f16_ptr, b_ptrs, &mut results);
        std::ptr::copy_nonoverlapping(results.as_ptr(), ctx.out_ptr.add(j), NR);
        j += NR;
    }
    while j < j_end {
        *ctx.out_ptr.add(j) = CpuBackendNeon::vec_dot_f16_native_gemv_1row(
            ctx.k,
            ctx.a_f16_ptr,
            ctx.b_base.add(j * ctx.k),
        );
        j += 1;
    }
}

/// N-major GEMM with RM=4 tiling + native F16 FMA.
/// Processes (n_chunk, m_group) where m_group = group of 4 A rows.
/// 4×4 micro-tile: B loaded once, shared across 4 A rows.
/// F16 accumulators enable 16-output tile in 16 registers.
///
/// Pipeline: Load=8(4cy), FMA=16(4cy) → balanced @ 4cy, 4.0 output/cy
///
/// # Safety
/// Called by SpinPool workers with valid F16GemmCtx pointer.
unsafe fn f16_gemm_chunk(ctx_ptr: *const u8, chunk_id: usize) {
    const NR: usize = 4;
    const MR: usize = 4;
    let ctx = &*(ctx_ptr as *const F16GemmCtx);

    // MR=4 mapping: chunk_id = n_chunk_idx * m_groups + m_group
    let m_groups = (ctx.m + MR - 1) / MR;
    let n_chunk_idx = chunk_id / m_groups;
    let m_group = chunk_id % m_groups;
    let m_row0 = m_group * MR;
    let m_remaining = ctx.m - m_row0; // 1..=4 rows in this group

    let j_start = n_chunk_idx * ctx.rows_per_chunk;
    let j_end = (j_start + ctx.rows_per_chunk).min(ctx.n);

    if m_remaining >= MR {
        // Full MR=4 tile: use native F16 FMA 4×4 kernel
        let a_ptrs = [
            ctx.a_f16_base.add(m_row0 * ctx.k),
            ctx.a_f16_base.add((m_row0 + 1) * ctx.k),
            ctx.a_f16_base.add((m_row0 + 2) * ctx.k),
            ctx.a_f16_base.add((m_row0 + 3) * ctx.k),
        ];
        let out_ptrs = [
            ctx.out_base.add(m_row0 * ctx.n),
            ctx.out_base.add((m_row0 + 1) * ctx.n),
            ctx.out_base.add((m_row0 + 2) * ctx.n),
            ctx.out_base.add((m_row0 + 3) * ctx.n),
        ];

        let mut j = j_start;
        while j + NR <= j_end {
            let b_ptrs = [
                ctx.b_base.add(j * ctx.k),
                ctx.b_base.add((j + 1) * ctx.k),
                ctx.b_base.add((j + 2) * ctx.k),
                ctx.b_base.add((j + 3) * ctx.k),
            ];
            let mut results = [[0.0f32; NR]; MR];
            CpuBackendNeon::vec_dot_f16_native_4x4(ctx.k, a_ptrs, b_ptrs, &mut results);
            for r in 0..MR {
                std::ptr::copy_nonoverlapping(results[r].as_ptr(), out_ptrs[r].add(j), NR);
            }
            j += NR;
        }
        // Tail: remaining B columns (< NR)
        while j < j_end {
            for r in 0..MR {
                *out_ptrs[r].add(j) = CpuBackendNeon::vec_dot_f16_native_gemv_1row(
                    ctx.k,
                    a_ptrs[r],
                    ctx.b_base.add(j * ctx.k),
                );
            }
            j += 1;
        }
    } else {
        // Tail group (1-3 rows): fallback to per-row GEMV kernel
        for r in 0..m_remaining {
            let a_f16 = ctx.a_f16_base.add((m_row0 + r) * ctx.k);
            let out_ptr = ctx.out_base.add((m_row0 + r) * ctx.n);
            let mut j = j_start;
            while j + NR <= j_end {
                let b_ptrs = [
                    ctx.b_base.add(j * ctx.k),
                    ctx.b_base.add((j + 1) * ctx.k),
                    ctx.b_base.add((j + 2) * ctx.k),
                    ctx.b_base.add((j + 3) * ctx.k),
                ];
                let mut results = [0.0f32; NR];
                CpuBackendNeon::vec_dot_f16_native_gemv_4rows(ctx.k, a_f16, b_ptrs, &mut results);
                std::ptr::copy_nonoverlapping(results.as_ptr(), out_ptr.add(j), NR);
                j += NR;
            }
            while j < j_end {
                *out_ptr.add(j) = CpuBackendNeon::vec_dot_f16_native_gemv_1row(
                    ctx.k,
                    a_f16,
                    ctx.b_base.add(j * ctx.k),
                );
                j += 1;
            }
        }
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    /// Scalar F32 reference dot product (ground truth).
    fn ref_dot_f32(a_f16: &[half::f16], b_f16: &[half::f16]) -> f32 {
        assert_eq!(a_f16.len(), b_f16.len());
        let mut sum = 0.0f32;
        for (&a, &b) in a_f16.iter().zip(b_f16.iter()) {
            sum += a.to_f32() * b.to_f32();
        }
        sum
    }

    /// Deterministic pseudo-random F16 vector in range [lo, hi].
    /// Uses a splitmix64-style hash to avoid pulling in `rand` as dev-dep.
    fn gen_f16_vec(len: usize, seed: u64, lo: f32, hi: f32) -> Vec<half::f16> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
        let mut out = Vec::with_capacity(len);
        let span = hi - lo;
        for _ in 0..len {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            // Map to [0, 1)
            let u = ((z >> 11) as f64 / (1u64 << 53) as f64) as f32;
            out.push(half::f16::from_f32(lo + u * span));
        }
        out
    }

    /// Assert `actual ≈ expected` under the standard numpy-style bound
    /// `|a - e| ≤ atol + rtol × |e|`. F16 accumulators have a non-trivial
    /// additive noise floor (`atol`) from ULP rounding during reduction, plus
    /// a multiplicative precision limit (`rtol`) from mantissa truncation.
    fn assert_close(actual: &[f32], expected: &[f32], rtol: f32, atol: f32, ctx: &str) {
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let bound = atol + rtol * e.abs();
            let diff = (a - e).abs();
            assert!(
                diff <= bound,
                "{}: element {} diff={:e} > bound={:e} (actual={:e}, expected={:e})",
                ctx,
                i,
                diff,
                bound,
                a,
                e
            );
        }
    }

    /// Test that `vec_dot_f16_native_gemv_4rows` matches a scalar F32 reference
    /// and stays within F16 precision tolerances across the K × N matrix of
    /// shapes encountered in Qwen 2.5 decode (including NR-unaligned shapes
    /// from tensor partition CPU slices).
    #[test]
    fn test_f16_native_gemv_4rows_matches_ref() {
        // (K, N) pairs cover Qwen 2.5-1.5B decode matmul shapes plus
        // partition-unaligned and small-K tail cases:
        //   K=1536 → Q/K/V/O/Gate/Up (hidden_size)
        //   K=8960 → Down (intermediate_size)
        //   K in {32,33,40,64,72,256,1024} → K-step-32 main/tail/scalar-tail
        //   N in {4,5,7,16} → NR=4 aligned + unaligned tail columns
        //   N in {1536,8960} → Qwen output dims
        //   N in {2688,2776} → typical tensor partition CPU-slice widths
        let cases: &[(usize, usize)] = &[
            (32, 4),
            (32, 5),
            (32, 7),
            (32, 16),
            (33, 4),
            (33, 7),
            (40, 16),
            (64, 16),
            (64, 2688),
            (72, 7),
            (256, 16),
            (256, 1536),
            (1024, 1536),
            (1024, 2776),
            (1536, 1536),
            (1536, 2688),
            (1536, 8960),
            (8960, 1536),
            (8960, 2691),
        ];

        for &(k, n) in cases {
            // A in RMS-norm-like range (~unit magnitude).
            let a = gen_f16_vec(k, 0xA1A1_u64 ^ (k as u64), -1.5, 1.5);
            // B in Qwen-weight-like range (~0.25 typical magnitude).
            let b_all = gen_f16_vec(n * k, 0xB2B2_u64 ^ (k as u64 * 131 + n as u64), -0.5, 0.5);

            // Reference: scalar F32 for each of N rows
            let mut expected = vec![0.0f32; n];
            for j in 0..n {
                expected[j] = ref_dot_f32(&a, &b_all[j * k..(j + 1) * k]);
            }

            // Run the NR=4 kernel + 1-row tail, mirroring f16_gemv_chunk dispatch
            let mut out = vec![0.0f32; n];
            let a_ptr = a.as_ptr() as *const u16;
            let b_ptr_base = b_all.as_ptr() as *const u16;
            let mut j = 0;
            while j + 4 <= n {
                let b_ptrs = unsafe {
                    [
                        b_ptr_base.add(j * k),
                        b_ptr_base.add((j + 1) * k),
                        b_ptr_base.add((j + 2) * k),
                        b_ptr_base.add((j + 3) * k),
                    ]
                };
                let mut res = [0.0f32; 4];
                unsafe {
                    CpuBackendNeon::vec_dot_f16_native_gemv_4rows(k, a_ptr, b_ptrs, &mut res);
                }
                out[j..j + 4].copy_from_slice(&res);
                j += 4;
            }
            while j < n {
                out[j] = unsafe {
                    CpuBackendNeon::vec_dot_f16_native_gemv_1row(k, a_ptr, b_ptr_base.add(j * k))
                };
                j += 1;
            }

            // F16 accumulation precision envelope: rtol × |expected| + atol.
            // rtol ≈ 1e-2 covers the multiplicative F16 mantissa loss
            // (~11 bits); atol scales with sqrt(K) × input_scale to absorb
            // near-zero cancellation cases where the accumulated F16 noise
            // floor dominates relative error. Empirical constants tuned for
            // Qwen-range inputs (|a| ≤ 1.5, |b| ≤ 0.5).
            let atol = 3e-3 * (k as f32).sqrt() + 1e-2;
            assert_close(&out, &expected, 1e-2, atol, &format!("k={}, n={}", k, n));
        }
    }

    /// Test that `vec_dot_f16_native_gemv_1row` matches scalar F32 reference
    /// in isolation (it is exercised indirectly by the 4rows test above, but
    /// this covers the standalone code path used by e.g. the GEMM column tail).
    #[test]
    fn test_f16_native_gemv_1row_matches_ref() {
        let k_cases = [32usize, 33, 40, 64, 72, 256, 1024, 1536, 8960];
        for &k in &k_cases {
            let a = gen_f16_vec(k, 0xC3C3_u64 ^ (k as u64), -1.5, 1.5);
            let b = gen_f16_vec(k, 0xD4D4_u64 ^ (k as u64), -0.5, 0.5);
            let expected = ref_dot_f32(&a, &b);
            let actual = unsafe {
                CpuBackendNeon::vec_dot_f16_native_gemv_1row(
                    k,
                    a.as_ptr() as *const u16,
                    b.as_ptr() as *const u16,
                )
            };
            // Same F16 accumulation precision budget as the 4rows test.
            let atol = 3e-3 * (k as f32).sqrt() + 1e-2;
            assert_close(
                std::slice::from_ref(&actual),
                std::slice::from_ref(&expected),
                1e-2,
                atol,
                &format!("k={}", k),
            );
        }
    }

    #[test]
    fn test_i8mm_dot_q4_0_q8_0() {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            eprintln!("[SKIPPED] i8mm not supported on this device");
            return;
        }

        let backend = CpuBackendNeon;

        // Test various dimensions
        for nb in [1, 2, 4, 8, 64, 128] {
            let k = nb * QK4_0; // 32, 64, 128, 256, 2048, 4096

            // Create deterministic test data
            let mut rng_state = 42u64;
            let mut pseudo_rand = || -> i8 {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng_state >> 33) % 15) as i8 - 7
            };

            let mut blocks_row0 = Vec::with_capacity(nb);
            let mut blocks_row1 = Vec::with_capacity(nb);
            for blk in 0..nb {
                let d0 = half::f16::from_f32(0.1 + 0.01 * blk as f32);
                let d1 = half::f16::from_f32(0.15 + 0.01 * blk as f32);
                let mut qs0 = [0u8; QK4_0 / 2];
                let mut qs1 = [0u8; QK4_0 / 2];
                for z in 0..QK4_0 / 2 {
                    let v0_lo = (pseudo_rand() + 8) as u8;
                    let v0_hi = (pseudo_rand() + 8) as u8;
                    qs0[z] = (v0_lo & 0x0F) | ((v0_hi & 0x0F) << 4);
                    let v1_lo = (pseudo_rand() + 8) as u8;
                    let v1_hi = (pseudo_rand() + 8) as u8;
                    qs1[z] = (v1_lo & 0x0F) | ((v1_hi & 0x0F) << 4);
                }
                blocks_row0.push(BlockQ4_0 { d: d0, qs: qs0 });
                blocks_row1.push(BlockQ4_0 { d: d1, qs: qs1 });
            }

            // Create activation (Q8_0)
            let mut act_blocks = Vec::with_capacity(nb);
            for blk in 0..nb {
                let d = half::f16::from_f32(0.2 + 0.005 * blk as f32);
                let mut qs = [0i8; QK8_0];
                for z in 0..QK8_0 {
                    qs[z] = pseudo_rand();
                }
                act_blocks.push(BlockQ8_0 { d, qs });
            }

            // Reference: existing sdot function
            let mut ref0 = 0.0f32;
            let mut ref1 = 0.0f32;
            unsafe {
                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    backend.vec_dot_q4_0_q8_0_sdot(
                        k,
                        &mut ref0,
                        blocks_row0.as_ptr(),
                        act_blocks.as_ptr(),
                    );
                    backend.vec_dot_q4_0_q8_0_sdot(
                        k,
                        &mut ref1,
                        blocks_row1.as_ptr(),
                        act_blocks.as_ptr(),
                    );
                } else {
                    backend.vec_dot_q4_0_q8_0(
                        k,
                        &mut ref0,
                        blocks_row0.as_ptr(),
                        act_blocks.as_ptr(),
                    );
                    backend.vec_dot_q4_0_q8_0(
                        k,
                        &mut ref1,
                        blocks_row1.as_ptr(),
                        act_blocks.as_ptr(),
                    );
                }
            }

            // i8mm
            let mut s0 = 0.0f32;
            let mut s1 = 0.0f32;
            unsafe {
                backend.vec_dot_q4_0_q8_0_i8mm(
                    k,
                    &mut s0,
                    &mut s1,
                    blocks_row0.as_ptr(),
                    blocks_row1.as_ptr(),
                    act_blocks.as_ptr(),
                );
            }

            let tol = 1e-4 * (k as f32).sqrt();
            assert!(
                (s0 - ref0).abs() < tol,
                "k={k} row0: i8mm={s0} ref={ref0} diff={}",
                (s0 - ref0).abs()
            );
            assert!(
                (s1 - ref1).abs() < tol,
                "k={k} row1: i8mm={s1} ref={ref1} diff={}",
                (s1 - ref1).abs()
            );
            eprintln!(
                "[PASS] k={k}: row0 diff={:.6}, row1 diff={:.6}",
                (s0 - ref0).abs(),
                (s1 - ref1).abs()
            );
        }
    }
}
