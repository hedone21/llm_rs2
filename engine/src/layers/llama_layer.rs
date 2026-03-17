use crate::backend::cpu::CpuBackend;
use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::kv_cache::{KVCache, KVCacheOps, KVLayout};
use crate::core::memory::Memory;
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use crate::memory::galloc::Galloc;
use anyhow::Result;
use rayon::prelude::*;
use std::sync::Arc;

// Re-export OpProfiler from its canonical location for backward compatibility.
pub use crate::profile::ops::OpProfiler;

// --- x86_64 AVX2 SIMD helpers for attention ---

/// Dot product: sum(a[i] * b[i]) for i in 0..len, using AVX2+FMA.
/// head_dim=64 → 8 AVX2 iterations (4x unrolled = 2 outer iterations).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_f32_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;

        // 4x unrolled: 32 floats per iteration
        while i + 32 <= len {
            let a0 = _mm256_loadu_ps(a.add(i));
            let b0 = _mm256_loadu_ps(b.add(i));
            sum = _mm256_fmadd_ps(a0, b0, sum);

            let a1 = _mm256_loadu_ps(a.add(i + 8));
            let b1 = _mm256_loadu_ps(b.add(i + 8));
            sum = _mm256_fmadd_ps(a1, b1, sum);

            let a2 = _mm256_loadu_ps(a.add(i + 16));
            let b2 = _mm256_loadu_ps(b.add(i + 16));
            sum = _mm256_fmadd_ps(a2, b2, sum);

            let a3 = _mm256_loadu_ps(a.add(i + 24));
            let b3 = _mm256_loadu_ps(b.add(i + 24));
            sum = _mm256_fmadd_ps(a3, b3, sum);

            i += 32;
        }

        while i + 8 <= len {
            let a0 = _mm256_loadu_ps(a.add(i));
            let b0 = _mm256_loadu_ps(b.add(i));
            sum = _mm256_fmadd_ps(a0, b0, sum);
            i += 8;
        }

        // Horizontal sum: 256→128→scalar
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result128 = _mm_add_ss(sums, shuf2);
        let mut result = _mm_cvtss_f32(result128);

        // Scalar tail
        while i < len {
            result += *a.add(i) * *b.add(i);
            i += 1;
        }

        result
    }
}

/// Weighted accumulation: out[i] += weight * v[i] for i in 0..len, using AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn weighted_accum_f32_avx2(out: *mut f32, v: *const f32, weight: f32, len: usize) {
    unsafe {
        use std::arch::x86_64::*;
        let w = _mm256_set1_ps(weight);
        let mut i = 0;

        // 4x unrolled: 32 floats per iteration
        while i + 32 <= len {
            let o0 = _mm256_loadu_ps(out.add(i));
            let v0 = _mm256_loadu_ps(v.add(i));
            _mm256_storeu_ps(out.add(i), _mm256_fmadd_ps(w, v0, o0));

            let o1 = _mm256_loadu_ps(out.add(i + 8));
            let v1 = _mm256_loadu_ps(v.add(i + 8));
            _mm256_storeu_ps(out.add(i + 8), _mm256_fmadd_ps(w, v1, o1));

            let o2 = _mm256_loadu_ps(out.add(i + 16));
            let v2 = _mm256_loadu_ps(v.add(i + 16));
            _mm256_storeu_ps(out.add(i + 16), _mm256_fmadd_ps(w, v2, o2));

            let o3 = _mm256_loadu_ps(out.add(i + 24));
            let v3 = _mm256_loadu_ps(v.add(i + 24));
            _mm256_storeu_ps(out.add(i + 24), _mm256_fmadd_ps(w, v3, o3));

            i += 32;
        }

        while i + 8 <= len {
            let o0 = _mm256_loadu_ps(out.add(i));
            let v0 = _mm256_loadu_ps(v.add(i));
            _mm256_storeu_ps(out.add(i), _mm256_fmadd_ps(w, v0, o0));
            i += 8;
        }

        // Scalar tail
        while i < len {
            *out.add(i) += weight * *v.add(i);
            i += 1;
        }
    }
}

pub struct LlamaLayer {
    // Attention
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,

    // MLP
    pub w_gate: Tensor, // silu_mul gate
    pub w_up: Tensor,
    pub w_down: Tensor,

    // Norms
    pub attention_norm: Tensor,
    pub ffn_norm: Tensor,
}

impl LlamaLayer {
    pub fn forward<C: KVCacheOps>(&self, args: LlamaLayerForwardArgs<C>) -> Result<()> {
        let x = args.x;
        let kv_cache = args.kv_cache;
        let start_pos = args.start_pos;
        let backend = args.backend;
        let memory = args.memory;
        let rms_norm_eps = args.rms_norm_eps;
        let rope_theta = args.rope_theta;
        let workspace = args.workspace;
        let use_gpu_attn = args.use_gpu_attn;

        let batch_size = x.shape().dims()[0];
        let seq_len = x.shape().dims()[1];
        let dim = x.shape().dims()[2];
        let head_dim = args.head_dim;

        let need_scores = args.need_scores;

        if seq_len == 1
            && let Some(ws) = workspace
        {
            return self.forward_gen(LlamaForwardGenArgs {
                x,
                kv_cache,
                start_pos,
                backend,
                memory,
                ws,
                rms_norm_eps,
                rope_theta,
                use_gpu_attn,
                need_scores,
                head_dim,
                profiler: args.profiler,
            });
        }

        // Standard forward path (Prefill or dynamic generation)
        let residual = backend.copy_from(x)?;
        backend.rms_norm(x, &self.attention_norm, rms_norm_eps)?;

        let q_dim = self.wq.shape().dims()[0];
        let k_dim = self.wk.shape().dims()[0];
        let v_dim = self.wv.shape().dims()[0];

        let mut q = self.alloc_temp(vec![batch_size, seq_len, q_dim], memory, backend)?;
        let mut k = self.alloc_temp(vec![batch_size, seq_len, k_dim], memory, backend)?;
        let mut v = self.alloc_temp(vec![batch_size, seq_len, v_dim], memory, backend)?;

        backend.matmul_transposed(x, &self.wq, &mut q)?;
        backend.matmul_transposed(x, &self.wk, &mut k)?;
        backend.matmul_transposed(x, &self.wv, &mut v)?;

        let n_heads_q = q_dim / head_dim;
        let n_heads_kv = k_dim / head_dim;

        let mut q_rope = Tensor::new(
            Shape::new(vec![batch_size, seq_len, n_heads_q, head_dim]),
            q.buffer().clone(),
            backend.clone(),
        );
        let mut k_rope = Tensor::new(
            Shape::new(vec![batch_size, seq_len, n_heads_kv, head_dim]),
            k.buffer().clone(),
            backend.clone(),
        );

        backend.rope_inplace(&mut q_rope, start_pos, rope_theta)?;
        backend.rope_inplace(&mut k_rope, start_pos, rope_theta)?;

        // Cast to target dtype if KV cache is not F32
        let kv_dtype = kv_cache.kv_dtype();
        if kv_dtype != DType::F32 {
            let n_elem = seq_len * n_heads_kv * head_dim;
            let buf_size = match kv_dtype {
                DType::F16 => n_elem * 2,
                DType::Q4_0 => {
                    (n_elem / crate::core::quant::QK4_0)
                        * std::mem::size_of::<crate::core::quant::BlockQ4_0>()
                }
                _ => n_elem * 4,
            };
            let k_cast_buf = memory.alloc(buf_size, kv_dtype)?;
            let mut k_cast = Tensor::new(k_rope.shape().clone(), k_cast_buf, backend.clone());
            backend.cast(&k_rope, &mut k_cast)?;
            let v_cast_buf = memory.alloc(buf_size, kv_dtype)?;
            let mut v_cast = Tensor::new(v.shape().clone(), v_cast_buf, backend.clone());
            backend.cast(&v, &mut v_cast)?;
            kv_cache.update(&k_cast, &v_cast)?;
        } else {
            kv_cache.update(&k_rope, &v)?;
        }

        let cache_seq_len = kv_cache.current_pos();
        let (k_cache, v_cache) = kv_cache.get_view();

        let mut out_attn = self.alloc_temp(vec![batch_size, seq_len, q_dim], memory, backend)?;

        let is_opencl = backend.name() == "OpenCL";
        let mut out_vec = Vec::new();

        {
            // Helper to cast slice
            fn as_u8_mut(v: &mut [f32]) -> &mut [u8] {
                unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
            }

            let mut q_vec = Vec::new();
            let mut k_vec = Vec::new();
            let mut v_vec = Vec::new();

            let (q_data, k_data, v_data, out_ptr) = if is_opencl {
                // Helper to read tensor to F32 vec, handling Dequantization if needed
                let read_to_f32 = |t: &Tensor, vec: &mut Vec<f32>| -> Result<()> {
                    if t.dtype() == DType::Q4_0 {
                        use crate::core::quant::{BlockQ4_0, QK4_0};
                        let numel = t.numel();
                        let n_blocks = numel / QK4_0;
                        let byte_size = n_blocks * std::mem::size_of::<BlockQ4_0>();

                        let mut byte_vec = vec![0u8; byte_size];
                        backend.read_buffer(t, &mut byte_vec)?;

                        vec.resize(numel, 0.0);
                        let blocks = unsafe {
                            std::slice::from_raw_parts(
                                byte_vec.as_ptr() as *const BlockQ4_0,
                                n_blocks,
                            )
                        };

                        for i in 0..n_blocks {
                            let mut tmp = [0.0f32; QK4_0];
                            blocks[i].dequantize(&mut tmp);
                            vec[i * QK4_0..(i + 1) * QK4_0].copy_from_slice(&tmp);
                        }
                    } else if t.dtype() == DType::F16 {
                        let numel = t.numel();
                        let byte_size = numel * 2;
                        let mut byte_vec = vec![0u8; byte_size];
                        backend.read_buffer(t, &mut byte_vec)?;
                        vec.resize(numel, 0.0);
                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            crate::backend::cpu::neon::CpuBackendNeon::bulk_f16_to_f32(
                                byte_vec.as_ptr() as *const u16,
                                vec.as_mut_ptr(),
                                numel,
                            );
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            let f16_slice = unsafe {
                                std::slice::from_raw_parts(
                                    byte_vec.as_ptr() as *const half::f16,
                                    numel,
                                )
                            };
                            for i in 0..numel {
                                vec[i] = f16_slice[i].to_f32();
                            }
                        }
                    } else {
                        vec.resize(t.numel(), 0.0);
                        backend.read_buffer(t, as_u8_mut(vec))?;
                    }
                    Ok(())
                };

                read_to_f32(&q_rope, &mut q_vec)?;
                read_to_f32(&k_cache, &mut k_vec)?;
                read_to_f32(&v_cache, &mut v_vec)?;

                out_vec.resize(out_attn.numel(), 0.0);

                (&q_vec[..], &k_vec[..], &v_vec[..], &mut out_vec[..])
            } else if k_cache.dtype() == DType::Q4_0 {
                // Q4_0: dequantize KV cache to F32 temp buffers
                use crate::core::quant::{BlockQ4_0, QK4_0};
                // HeadMajor: need full buffer (heads are non-contiguous)
                let n_elems = if kv_cache.layout() == KVLayout::HeadMajor {
                    n_heads_kv * kv_cache.capacity() * head_dim
                } else {
                    cache_seq_len * n_heads_kv * head_dim
                };
                let n_blocks = n_elems / QK4_0;
                let k_blocks = unsafe {
                    std::slice::from_raw_parts(k_cache.as_ptr() as *const BlockQ4_0, n_blocks)
                };
                let v_blocks = unsafe {
                    std::slice::from_raw_parts(v_cache.as_ptr() as *const BlockQ4_0, n_blocks)
                };
                k_vec.resize(n_elems, 0.0f32);
                v_vec.resize(n_elems, 0.0f32);
                for bi in 0..n_blocks {
                    let mut tmp = [0.0f32; QK4_0];
                    k_blocks[bi].dequantize(&mut tmp);
                    k_vec[bi * QK4_0..(bi + 1) * QK4_0].copy_from_slice(&tmp);
                    v_blocks[bi].dequantize(&mut tmp);
                    v_vec[bi * QK4_0..(bi + 1) * QK4_0].copy_from_slice(&tmp);
                }
                (
                    q_rope.as_slice::<f32>(),
                    &k_vec[..],
                    &v_vec[..],
                    out_attn.as_mut_slice::<f32>(),
                )
            } else if k_cache.dtype() == DType::F16 {
                // F16: convert KV cache to F32 temp buffers using NEON bulk conversion
                // HeadMajor: need full buffer (heads are non-contiguous)
                let n_elems = if kv_cache.layout() == KVLayout::HeadMajor {
                    n_heads_kv * kv_cache.capacity() * head_dim
                } else {
                    cache_seq_len * n_heads_kv * head_dim
                };
                let k_f16_ptr = k_cache.as_ptr() as *const u16;
                let v_f16_ptr = v_cache.as_ptr() as *const u16;
                k_vec.resize(n_elems, 0.0f32);
                v_vec.resize(n_elems, 0.0f32);
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    crate::backend::cpu::neon::CpuBackendNeon::bulk_f16_to_f32(
                        k_f16_ptr,
                        k_vec.as_mut_ptr(),
                        n_elems,
                    );
                    crate::backend::cpu::neon::CpuBackendNeon::bulk_f16_to_f32(
                        v_f16_ptr,
                        v_vec.as_mut_ptr(),
                        n_elems,
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    let k_f16 = unsafe {
                        std::slice::from_raw_parts(k_f16_ptr as *const half::f16, n_elems)
                    };
                    let v_f16 = unsafe {
                        std::slice::from_raw_parts(v_f16_ptr as *const half::f16, n_elems)
                    };
                    for i in 0..n_elems {
                        k_vec[i] = k_f16[i].to_f32();
                        v_vec[i] = v_f16[i].to_f32();
                    }
                }
                (
                    q_rope.as_slice::<f32>(),
                    &k_vec[..],
                    &v_vec[..],
                    out_attn.as_mut_slice::<f32>(),
                )
            } else {
                (
                    q_rope.as_slice::<f32>(),
                    k_cache.as_slice::<f32>(),
                    v_cache.as_slice::<f32>(),
                    out_attn.as_mut_slice::<f32>(),
                )
            };

            use crate::layers::attention::flash_attention_forward_strided;

            let is_head_major_pf = kv_cache.layout() == KVLayout::HeadMajor;
            let kv_capacity = kv_cache.capacity();

            let chunk_q_stride = seq_len * n_heads_q * head_dim;
            let chunk_out_stride = seq_len * n_heads_q * head_dim;
            // KV Cache is strided by capacity (physical buffer size), not max_seq_len
            let chunk_k_stride = kv_capacity * n_heads_kv * head_dim;

            // Layout-dependent strides
            let (k_pos_stride, kv_head_stride) = if is_head_major_pf {
                (head_dim, kv_capacity * head_dim)
            } else {
                (n_heads_kv * head_dim, head_dim)
            };

            // Iterate over batch.
            // We use chunks_mut for out_ptr to satisfy borrow checker.
            for (b, out_batch) in out_ptr.chunks_mut(chunk_out_stride).enumerate() {
                let q_start = b * chunk_q_stride;
                let k_start = b * chunk_k_stride;
                let v_start = b * chunk_k_stride; // V has same layout as K in cache

                let q_slice = &q_data[q_start..q_start + chunk_q_stride];

                // For HeadMajor, heads are non-contiguous so we need the full buffer per batch.
                // For SeqMajor, only valid positions are needed (contiguous from start).
                let k_valid_len = if is_head_major_pf {
                    n_heads_kv * kv_capacity * head_dim
                } else {
                    cache_seq_len * n_heads_kv * head_dim
                };
                let k_slice = &k_data[k_start..k_start + k_valid_len];
                let v_slice = &v_data[v_start..v_start + k_valid_len];

                flash_attention_forward_strided(
                    q_slice,
                    k_slice,
                    v_slice,
                    out_batch,
                    n_heads_q,
                    n_heads_kv,
                    seq_len,
                    cache_seq_len,
                    head_dim,
                    n_heads_q * head_dim, // q_stride
                    k_pos_stride,         // k_stride (position stride)
                    k_pos_stride,         // v_stride (same as k)
                    n_heads_q * head_dim, // out_stride
                    kv_head_stride,       // kv_head_stride
                    start_pos,            // q_start_pos for causal mask
                    32,                   // br
                    32,                   // bc
                );
            }
        }

        if is_opencl {
            // Create temp CPU tensor from result and copy back
            // Using Galloc directly
            let size_bytes = out_vec.len() * 4;
            let buf = Galloc::new().alloc(size_bytes, DType::F32)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_vec.as_ptr(),
                    buf.as_mut_ptr() as *mut f32,
                    out_vec.len(),
                );
            }
            let cpu_out = Tensor::new(out_attn.shape().clone(), buf, Arc::new(CpuBackend::new()));
            out_attn = backend.copy_from(&cpu_out)?;
        }

        let mut attn_out_projected =
            self.alloc_temp(vec![batch_size, seq_len, dim], memory, backend)?;
        backend.matmul_transposed(&out_attn, &self.wo, &mut attn_out_projected)?;

        backend.add_assign(&mut attn_out_projected, &residual)?;
        *x = attn_out_projected;

        let residual_ffn = backend.copy_from(x)?;
        backend.rms_norm(x, &self.ffn_norm, rms_norm_eps)?;

        let ffn_hidden = self.w_up.shape().dims()[0];
        let mut gate = self.alloc_temp(vec![batch_size, seq_len, ffn_hidden], memory, backend)?;
        let mut up = self.alloc_temp(vec![batch_size, seq_len, ffn_hidden], memory, backend)?;

        backend.matmul_transposed(x, &self.w_gate, &mut gate)?;
        backend.matmul_transposed(x, &self.w_up, &mut up)?;

        backend.silu_mul(&mut gate, &up)?;

        let mut down = self.alloc_temp(vec![batch_size, seq_len, dim], memory, backend)?;
        backend.matmul_transposed(&gate, &self.w_down, &mut down)?;

        backend.add_assign(&mut down, &residual_ffn)?;
        *x = down;

        Ok(())
    }

    fn alloc_temp(
        &self,
        shape: Vec<usize>,
        memory: &dyn Memory,
        backend: &Arc<dyn Backend>,
    ) -> Result<Tensor> {
        let size: usize = shape.iter().product();
        let buf = memory.alloc(size * 4, DType::F32)?;
        Ok(Tensor::new(Shape::new(shape), buf, backend.clone()))
    }

    /// Fast path for single token generation using pre-allocated workspace.
    fn forward_gen<C: KVCacheOps>(&self, mut args: LlamaForwardGenArgs<C>) -> Result<()> {
        let x = args.x;
        let kv_cache = args.kv_cache;
        let start_pos = args.start_pos;
        let backend = args.backend;
        let memory = args.memory;
        let ws = args.ws;
        let rms_norm_eps = args.rms_norm_eps;
        let rope_theta = args.rope_theta;
        let use_gpu_attn = args.use_gpu_attn;

        let batch_size = x.shape().dims()[0];
        let head_dim = args.head_dim;
        let mut profiler = args.profiler.as_deref_mut();
        let is_opencl = backend.name() == "OpenCL";

        macro_rules! prof_start {
            () => {
                if profiler.is_some() {
                    // Drain GPU queue before timing to get accurate per-op measurement
                    if is_opencl {
                        backend.synchronize().ok();
                    }
                    std::time::Instant::now()
                } else {
                    // Dummy instant (never read)
                    std::time::Instant::now()
                }
            };
        }
        macro_rules! prof_record {
            ($t:expr, $field:ident) => {
                if let Some(ref mut p) = profiler {
                    // Wait for GPU kernel to actually complete before recording time
                    if is_opencl {
                        backend.synchronize().ok();
                    }
                    p.$field += $t.elapsed().as_micros() as u64;
                }
            };
        }

        // 1. Attention Norm — out-of-place: ws.residual = norm(x), x preserved for skip connection
        let t = prof_start!();
        backend.rms_norm_oop(x, &mut ws.residual, &self.attention_norm, rms_norm_eps)?;
        prof_record!(t, rms_norm);

        // 2. QKV Projections from normalized x (ws.residual) — fused dispatch for F16 CPU
        let t = prof_start!();
        #[cfg(target_arch = "aarch64")]
        let is_cpu_f16 = backend.name().contains("CPU") && self.wq.dtype() == DType::F16;
        #[cfg(not(target_arch = "aarch64"))]
        let is_cpu_f16 = false;
        let is_decode = x
            .shape()
            .dims()
            .iter()
            .take(x.shape().dims().len() - 1)
            .product::<usize>()
            == 1;
        if is_cpu_f16 && is_decode {
            // Fused QKV: single SpinPool dispatch for all 3 matmuls
            #[cfg(target_arch = "aarch64")]
            {
                let k = ws.residual.shape().dims()[ws.residual.shape().dims().len() - 1];
                unsafe {
                    crate::backend::cpu::neon::fused_matmul_f16(
                        ws.residual.as_ptr() as *const f32,
                        k,
                        &[
                            (
                                self.wq.as_ptr() as *const u16,
                                ws.q.as_mut_ptr() as *mut f32,
                                self.wq.shape().dims()[0],
                            ),
                            (
                                self.wk.as_ptr() as *const u16,
                                ws.k.as_mut_ptr() as *mut f32,
                                self.wk.shape().dims()[0],
                            ),
                            (
                                self.wv.as_ptr() as *const u16,
                                ws.v.as_mut_ptr() as *mut f32,
                                self.wv.shape().dims()[0],
                            ),
                        ],
                    );
                }
            }
        } else {
            crate::core::thread_pool::get_pool().begin_batch();
            backend.matmul_transposed(&ws.residual, &self.wq, &mut ws.q)?;
            backend.matmul_transposed(&ws.residual, &self.wk, &mut ws.k)?;
            backend.matmul_transposed(&ws.residual, &self.wv, &mut ws.v)?;
            crate::core::thread_pool::get_pool().end_batch();
        }
        prof_record!(t, matmul_qkv);

        // 3. RoPE
        let t = prof_start!();
        let q_dim = self.wq.shape().dims()[0];
        let k_dim = self.wk.shape().dims()[0];
        let n_heads_q = q_dim / head_dim;
        let n_heads_kv = k_dim / head_dim;

        let mut q_rope = Tensor::new(
            Shape::new(vec![batch_size, 1, n_heads_q, head_dim]),
            ws.q.buffer().clone(),
            backend.clone(),
        );
        let mut k_rope = Tensor::new(
            Shape::new(vec![batch_size, 1, n_heads_kv, head_dim]),
            ws.k.buffer().clone(),
            backend.clone(),
        );

        backend.rope_inplace(&mut q_rope, start_pos, rope_theta)?;
        backend.rope_inplace(&mut k_rope, start_pos, rope_theta)?;
        prof_record!(t, rope);

        // 4. KV Cache Update - cast to target dtype if needed
        let t = prof_start!();
        let kv_dtype = kv_cache.kv_dtype();
        use crate::core::kv_cache::KVLayout;
        if kv_dtype == DType::F16
            && is_opencl
            && is_decode
            && kv_cache.layout() == KVLayout::HeadMajor
        {
            // GPU F16 HeadMajor: fused cast+scatter kernel (1 dispatch instead of 2+16)
            let pos = kv_cache.current_pos();
            let cap = kv_cache.capacity();
            if let Some((k_buf, v_buf)) = kv_cache.get_buffers_mut() {
                backend.kv_scatter_f32_to_f16(&k_rope, &ws.v, k_buf, v_buf, head_dim, cap, pos)?;
            }
            kv_cache.advance_pos(1);
        } else if kv_dtype != DType::F32 {
            let n_elem = n_heads_kv * head_dim;
            let buf_size = match kv_dtype {
                DType::F16 => n_elem * 2,
                DType::Q4_0 => {
                    (n_elem / crate::core::quant::QK4_0)
                        * std::mem::size_of::<crate::core::quant::BlockQ4_0>()
                }
                _ => n_elem * 4,
            };
            let k_shape = k_rope.shape().clone();
            let v_shape = Shape::new(vec![batch_size, 1, n_heads_kv, head_dim]);
            if ws.k_cast.is_none() {
                let buf = memory.alloc(buf_size, kv_dtype)?;
                ws.k_cast = Some(Tensor::new(k_shape.clone(), buf, backend.clone()));
            }
            if ws.v_cast.is_none() {
                let buf = memory.alloc(buf_size, kv_dtype)?;
                ws.v_cast = Some(Tensor::new(v_shape.clone(), buf, backend.clone()));
            }
            let k_cast = ws.k_cast.as_mut().unwrap();
            let v_cast = ws.v_cast.as_mut().unwrap();
            backend.cast(&k_rope, k_cast)?;
            backend.cast(&ws.v, v_cast)?;
            kv_cache.update(k_cast, v_cast)?;
        } else {
            kv_cache.update(&k_rope, &ws.v)?;
        }
        prof_record!(t, kv_update);

        // 5. Attention - use GPU kernel for OpenCL
        let t = prof_start!();
        let cache_seq_len = kv_cache.current_pos();
        let (k_cache, v_cache) = kv_cache.get_view();

        let need_scores = args.need_scores;

        if (backend.name() == "OpenCL" && use_gpu_attn) || k_cache.dtype() != DType::F32 {
            // GPU attention or F16 KV cache - use backend's dtype-aware implementation
            backend.attention_gen(
                &q_rope,
                &k_cache,
                &v_cache,
                &mut ws.out_attn,
                n_heads_q,
                n_heads_kv,
                head_dim,
                cache_seq_len,
            )?;

            // Separate score computation pass for non-F32 KV cache.
            // attention_gen() does NOT write to ws.scores, so we compute
            // Q·K^T + softmax on CPU for score accumulation.
            if need_scores {
                Self::compute_attention_scores(
                    &q_rope,
                    &k_cache,
                    &mut ws.scores,
                    n_heads_q,
                    n_heads_kv,
                    head_dim,
                    cache_seq_len,
                    kv_cache.layout() == KVLayout::HeadMajor,
                    kv_cache.capacity(),
                    backend,
                )?;
            }
        } else {
            // CPU attention path (Fallback for OpenCL or native CPU F32)
            let mut q_vec = Vec::new();
            let mut k_vec = Vec::new();
            let mut v_vec = Vec::new();
            let mut out_vec = Vec::new();

            let is_opencl = backend.name() == "OpenCL";

            let (q_data, k_data, v_data, out_ptr) = if is_opencl {
                q_vec.resize(q_rope.size() / 4, 0.0);
                k_vec.resize(k_cache.size() / 4, 0.0);
                v_vec.resize(v_cache.size() / 4, 0.0);
                out_vec.resize(ws.out_attn.size() / 4, 0.0);

                fn as_u8_mut(v: &mut [f32]) -> &mut [u8] {
                    unsafe {
                        std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4)
                    }
                }

                backend.read_buffer(&q_rope, as_u8_mut(&mut q_vec))?;
                backend.read_buffer(&k_cache, as_u8_mut(&mut k_vec))?;
                backend.read_buffer(&v_cache, as_u8_mut(&mut v_vec))?;

                (&q_vec[..], &k_vec[..], &v_vec[..], &mut out_vec[..])
            } else {
                (
                    q_rope.as_slice::<f32>(),
                    k_cache.as_slice::<f32>(),
                    v_cache.as_slice::<f32>(),
                    ws.out_attn.as_mut_slice::<f32>(),
                )
            };

            // Re-interpret out_attn (or out_vec) as raw slice, will fill with zeros first
            // Note: out_ptr is &mut [f32]
            for x in out_ptr.iter_mut() {
                *x = 0.0;
            }

            let scale = 1.0 / (head_dim as f32).sqrt();
            let n_rep = n_heads_q / n_heads_kv;
            let is_head_major = kv_cache.layout() == KVLayout::HeadMajor;
            let kv_capacity = kv_cache.capacity();

            // Calculate stride before mutable borrow
            let stride = ws.scores.len() / n_heads_q;

            // Use pre-allocated scores buffer: [n_heads_q, max_seq_len] (conceptually)
            // But we can just use linear indexing: h * max_seq_len + t
            let all_scores = &mut ws.scores;
            // Ensure we have enough space (should be guaranteed by new LayerWorkspace)
            if all_scores.len() < n_heads_q * cache_seq_len {
                // Fallback or panic, but expecting correct size
                // Dynamic resize just in case (e.g. if max_seq_len valid but cache grew?)
                // Actually cache_seq_len <= max_seq_len.
            }

            // Parallelize over heads
            // 1. Prepare mutable slices for scores and output
            //    scores: [n_heads_q, max_seq_len] -> split into chunks of max_seq_len
            //    out:    [n_heads_q, head_dim] -> split into chunks of head_dim

            // Parallelize over heads for longer sequences, Serial for short (to avoid overhead)
            let use_parallel = cache_seq_len >= 256;

            if use_parallel {
                let scores_chunks = ws.scores.par_chunks_mut(stride).take(n_heads_q);
                let out_chunks = out_ptr.par_chunks_mut(head_dim).take(n_heads_q);

                scores_chunks
                    .zip(out_chunks)
                    .enumerate()
                    .for_each(|(h, (scores_h, out_h))| {
                        let kv_h = h / n_rep;

                        // Unsafe access for performance
                        unsafe {
                            let q_off = h * head_dim;
                            let q_ptr = q_data.as_ptr().add(q_off);

                            // --- Step 1: Q * K^T (NEON Vectorized) ---
                            for t in 0..cache_seq_len {
                                let k_off = if is_head_major {
                                    (kv_h * kv_capacity + t) * head_dim
                                } else {
                                    (t * n_heads_kv + kv_h) * head_dim
                                };
                                let k_ptr = k_data.as_ptr().add(k_off);

                                #[cfg(target_arch = "aarch64")]
                                let score = {
                                    use std::arch::aarch64::*;
                                    let mut sum_v = vdupq_n_f32(0.0);

                                    // head_dim = 64, process 16 elements per iteration (4 unrolled)
                                    let mut i = 0;
                                    while i + 16 <= head_dim {
                                        let q0 = vld1q_f32(q_ptr.add(i));
                                        let k0 = vld1q_f32(k_ptr.add(i));
                                        sum_v = vfmaq_f32(sum_v, q0, k0);

                                        let q1 = vld1q_f32(q_ptr.add(i + 4));
                                        let k1 = vld1q_f32(k_ptr.add(i + 4));
                                        sum_v = vfmaq_f32(sum_v, q1, k1);

                                        let q2 = vld1q_f32(q_ptr.add(i + 8));
                                        let k2 = vld1q_f32(k_ptr.add(i + 8));
                                        sum_v = vfmaq_f32(sum_v, q2, k2);

                                        let q3 = vld1q_f32(q_ptr.add(i + 12));
                                        let k3 = vld1q_f32(k_ptr.add(i + 12));
                                        sum_v = vfmaq_f32(sum_v, q3, k3);

                                        i += 16;
                                    }

                                    // Tail (if head_dim not multiple of 16)
                                    while i + 4 <= head_dim {
                                        let q0 = vld1q_f32(q_ptr.add(i));
                                        let k0 = vld1q_f32(k_ptr.add(i));
                                        sum_v = vfmaq_f32(sum_v, q0, k0);
                                        i += 4;
                                    }

                                    // Horizontal reduction
                                    let mut score = vaddvq_f32(sum_v);

                                    // Scalar tail
                                    while i < head_dim {
                                        score += *q_ptr.add(i) * *k_ptr.add(i);
                                        i += 1;
                                    }
                                    score
                                };

                                #[cfg(target_arch = "x86_64")]
                                let score = dot_f32_avx2(q_ptr, k_ptr, head_dim);

                                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                                let score = {
                                    let mut score = 0.0;
                                    for i in 0..head_dim {
                                        score += *q_ptr.add(i) * *k_ptr.add(i);
                                    }
                                    score
                                };

                                *scores_h.get_unchecked_mut(t) = score * scale;
                            }

                            // --- Step 2: Softmax ---
                            let active_scores = &mut scores_h[0..cache_seq_len];

                            let mut max_val = f32::NEG_INFINITY;
                            for i in 0..cache_seq_len {
                                let s = *active_scores.get_unchecked(i);
                                if s > max_val {
                                    max_val = s;
                                }
                            }

                            let mut sum_exp = 0.0;
                            for i in 0..cache_seq_len {
                                let s = (*active_scores.get_unchecked(i) - max_val).exp();
                                *active_scores.get_unchecked_mut(i) = s;
                                sum_exp += s;
                            }

                            let inv_sum = 1.0 / sum_exp;
                            for i in 0..cache_seq_len {
                                *active_scores.get_unchecked_mut(i) *= inv_sum;
                            }

                            // --- Step 3: Score * V (SIMD Vectorized) ---
                            // Zero out output
                            #[cfg(target_arch = "aarch64")]
                            {
                                use std::arch::aarch64::*;
                                let zero = vdupq_n_f32(0.0);
                                let mut i = 0;
                                while i + 4 <= head_dim {
                                    vst1q_f32(out_h.as_mut_ptr().add(i), zero);
                                    i += 4;
                                }
                                while i < head_dim {
                                    *out_h.get_unchecked_mut(i) = 0.0;
                                    i += 1;
                                }
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            for x in out_h.iter_mut() {
                                *x = 0.0;
                            }

                            for t in 0..cache_seq_len {
                                let weight = *active_scores.get_unchecked(t);
                                let v_off = if is_head_major {
                                    (kv_h * kv_capacity + t) * head_dim
                                } else {
                                    (t * n_heads_kv + kv_h) * head_dim
                                };
                                let v_ptr = v_data.as_ptr().add(v_off);

                                #[cfg(target_arch = "aarch64")]
                                {
                                    use std::arch::aarch64::*;
                                    let out_ptr_h = out_h.as_mut_ptr();
                                    let w = vdupq_n_f32(weight);

                                    let mut i = 0;
                                    while i + 16 <= head_dim {
                                        let v0 = vld1q_f32(v_ptr.add(i));
                                        let o0 = vld1q_f32(out_ptr_h.add(i));
                                        vst1q_f32(out_ptr_h.add(i), vfmaq_f32(o0, w, v0));

                                        let v1 = vld1q_f32(v_ptr.add(i + 4));
                                        let o1 = vld1q_f32(out_ptr_h.add(i + 4));
                                        vst1q_f32(out_ptr_h.add(i + 4), vfmaq_f32(o1, w, v1));

                                        let v2 = vld1q_f32(v_ptr.add(i + 8));
                                        let o2 = vld1q_f32(out_ptr_h.add(i + 8));
                                        vst1q_f32(out_ptr_h.add(i + 8), vfmaq_f32(o2, w, v2));

                                        let v3 = vld1q_f32(v_ptr.add(i + 12));
                                        let o3 = vld1q_f32(out_ptr_h.add(i + 12));
                                        vst1q_f32(out_ptr_h.add(i + 12), vfmaq_f32(o3, w, v3));

                                        i += 16;
                                    }

                                    while i + 4 <= head_dim {
                                        let v0 = vld1q_f32(v_ptr.add(i));
                                        let o0 = vld1q_f32(out_ptr_h.add(i));
                                        vst1q_f32(out_ptr_h.add(i), vfmaq_f32(o0, w, v0));
                                        i += 4;
                                    }

                                    while i < head_dim {
                                        *out_h.get_unchecked_mut(i) += weight * *v_ptr.add(i);
                                        i += 1;
                                    }
                                }

                                #[cfg(target_arch = "x86_64")]
                                weighted_accum_f32_avx2(
                                    out_h.as_mut_ptr(),
                                    v_ptr,
                                    weight,
                                    head_dim,
                                );

                                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                                for i in 0..head_dim {
                                    *out_h.get_unchecked_mut(i) += weight * *v_ptr.add(i);
                                }
                            }
                        }
                    });
            } else {
                // Serial execution for short sequences
                let scores_chunks = ws.scores.chunks_mut(stride).take(n_heads_q);
                let out_chunks = out_ptr.chunks_mut(head_dim).take(n_heads_q);

                scores_chunks
                    .zip(out_chunks)
                    .enumerate()
                    .for_each(|(h, (scores_h, out_h))| {
                        let kv_h = h / n_rep;
                        unsafe {
                            let q_off = h * head_dim;
                            let q_ptr = q_data.as_ptr().add(q_off);

                            // --- Step 1: Q * K^T (NEON Vectorized) ---
                            for t in 0..cache_seq_len {
                                let k_off = if is_head_major {
                                    (kv_h * kv_capacity + t) * head_dim
                                } else {
                                    (t * n_heads_kv + kv_h) * head_dim
                                };
                                let k_ptr = k_data.as_ptr().add(k_off);

                                #[cfg(target_arch = "aarch64")]
                                let score = {
                                    use std::arch::aarch64::*;
                                    let mut sum_v = vdupq_n_f32(0.0);
                                    let mut i = 0;
                                    while i + 16 <= head_dim {
                                        let q0 = vld1q_f32(q_ptr.add(i));
                                        let k0 = vld1q_f32(k_ptr.add(i));
                                        sum_v = vfmaq_f32(sum_v, q0, k0);
                                        let q1 = vld1q_f32(q_ptr.add(i + 4));
                                        let k1 = vld1q_f32(k_ptr.add(i + 4));
                                        sum_v = vfmaq_f32(sum_v, q1, k1);
                                        let q2 = vld1q_f32(q_ptr.add(i + 8));
                                        let k2 = vld1q_f32(k_ptr.add(i + 8));
                                        sum_v = vfmaq_f32(sum_v, q2, k2);
                                        let q3 = vld1q_f32(q_ptr.add(i + 12));
                                        let k3 = vld1q_f32(k_ptr.add(i + 12));
                                        sum_v = vfmaq_f32(sum_v, q3, k3);
                                        i += 16;
                                    }
                                    while i + 4 <= head_dim {
                                        let q0 = vld1q_f32(q_ptr.add(i));
                                        let k0 = vld1q_f32(k_ptr.add(i));
                                        sum_v = vfmaq_f32(sum_v, q0, k0);
                                        i += 4;
                                    }
                                    let mut score = vaddvq_f32(sum_v);
                                    while i < head_dim {
                                        score += *q_ptr.add(i) * *k_ptr.add(i);
                                        i += 1;
                                    }
                                    score
                                };
                                #[cfg(target_arch = "x86_64")]
                                let score = dot_f32_avx2(q_ptr, k_ptr, head_dim);

                                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                                let score = {
                                    let mut score = 0.0;
                                    for i in 0..head_dim {
                                        score += *q_ptr.add(i) * *k_ptr.add(i);
                                    }
                                    score
                                };
                                *scores_h.get_unchecked_mut(t) = score * scale;
                            }

                            // --- Step 2: Softmax ---
                            let active_scores = &mut scores_h[0..cache_seq_len];
                            let mut max_val = f32::NEG_INFINITY;
                            for i in 0..cache_seq_len {
                                let s = *active_scores.get_unchecked(i);
                                if s > max_val {
                                    max_val = s;
                                }
                            }
                            let mut sum_exp = 0.0;
                            for i in 0..cache_seq_len {
                                let s = (*active_scores.get_unchecked(i) - max_val).exp();
                                *active_scores.get_unchecked_mut(i) = s;
                                sum_exp += s;
                            }
                            let inv_sum = 1.0 / sum_exp;
                            for i in 0..cache_seq_len {
                                *active_scores.get_unchecked_mut(i) *= inv_sum;
                            }

                            // --- Step 3: Score * V (NEON Vectorized) ---
                            #[cfg(target_arch = "aarch64")]
                            {
                                use std::arch::aarch64::*;
                                let zero = vdupq_n_f32(0.0);
                                let mut i = 0;
                                while i + 4 <= head_dim {
                                    vst1q_f32(out_h.as_mut_ptr().add(i), zero);
                                    i += 4;
                                }
                                while i < head_dim {
                                    *out_h.get_unchecked_mut(i) = 0.0;
                                    i += 1;
                                }
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            for x in out_h.iter_mut() {
                                *x = 0.0;
                            }

                            for t in 0..cache_seq_len {
                                let weight = *active_scores.get_unchecked(t);
                                let v_off = if is_head_major {
                                    (kv_h * kv_capacity + t) * head_dim
                                } else {
                                    (t * n_heads_kv + kv_h) * head_dim
                                };
                                let v_ptr = v_data.as_ptr().add(v_off);
                                #[cfg(target_arch = "aarch64")]
                                let out_ptr_h = out_h.as_mut_ptr();

                                #[cfg(target_arch = "aarch64")]
                                {
                                    use std::arch::aarch64::*;
                                    let w = vdupq_n_f32(weight);
                                    let mut i = 0;
                                    while i + 16 <= head_dim {
                                        let v0 = vld1q_f32(v_ptr.add(i));
                                        let o0 = vld1q_f32(out_ptr_h.add(i));
                                        vst1q_f32(out_ptr_h.add(i), vfmaq_f32(o0, w, v0));
                                        let v1 = vld1q_f32(v_ptr.add(i + 4));
                                        let o1 = vld1q_f32(out_ptr_h.add(i + 4));
                                        vst1q_f32(out_ptr_h.add(i + 4), vfmaq_f32(o1, w, v1));
                                        let v2 = vld1q_f32(v_ptr.add(i + 8));
                                        let o2 = vld1q_f32(out_ptr_h.add(i + 8));
                                        vst1q_f32(out_ptr_h.add(i + 8), vfmaq_f32(o2, w, v2));
                                        let v3 = vld1q_f32(v_ptr.add(i + 12));
                                        let o3 = vld1q_f32(out_ptr_h.add(i + 12));
                                        vst1q_f32(out_ptr_h.add(i + 12), vfmaq_f32(o3, w, v3));
                                        i += 16;
                                    }
                                    while i + 4 <= head_dim {
                                        let v0 = vld1q_f32(v_ptr.add(i));
                                        let o0 = vld1q_f32(out_ptr_h.add(i));
                                        vst1q_f32(out_ptr_h.add(i), vfmaq_f32(o0, w, v0));
                                        i += 4;
                                    }
                                    while i < head_dim {
                                        *out_h.get_unchecked_mut(i) += weight * *v_ptr.add(i);
                                        i += 1;
                                    }
                                }
                                #[cfg(target_arch = "x86_64")]
                                weighted_accum_f32_avx2(
                                    out_h.as_mut_ptr(),
                                    v_ptr,
                                    weight,
                                    head_dim,
                                );

                                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                                for i in 0..head_dim {
                                    *out_h.get_unchecked_mut(i) += weight * *v_ptr.add(i);
                                }
                            }
                        }
                    });
            } // End of: if use_parallel { ... } else { ... }

            if is_opencl {
                // Determine size from the actual out_vec we used
                let size_bytes = out_vec.len() * 4;
                let buf = Galloc::new().alloc(size_bytes, DType::F32)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        out_vec.as_ptr(),
                        buf.as_mut_ptr() as *mut f32,
                        out_vec.len(),
                    );
                }
                let cpu_out = Tensor::new(
                    ws.out_attn.shape().clone(),
                    buf,
                    Arc::new(CpuBackend::new()),
                );
                // Use backend.copy_from to transfer back to GPU tensor ws.out_attn
                // Note: ws.out_attn is &mut Tensor, so this updates the GPU buffer contents
                ws.out_attn = backend.copy_from(&cpu_out)?;
            }
        }

        // 6. Output Projection
        prof_record!(t, attention);

        let t = prof_start!();
        backend.matmul_transposed(&ws.out_attn, &self.wo, &mut ws.attn_out)?;
        prof_record!(t, matmul_wo);

        // 7. Residual 1 — accumulate attention result into x (skip connection)
        let t = prof_start!();
        backend.add_assign(x, &ws.attn_out)?;
        prof_record!(t, add_assign);

        // 8. FFN Norm — out-of-place: ws.residual = norm(x), x preserved
        let t = prof_start!();
        backend.rms_norm_oop(x, &mut ws.residual, &self.ffn_norm, rms_norm_eps)?;
        prof_record!(t, rms_norm);

        // 9. FFN — fused gate+up dispatch for F16 CPU (1 dispatch instead of 2)
        let t = prof_start!();
        if is_cpu_f16 && is_decode {
            #[cfg(target_arch = "aarch64")]
            {
                let k = ws.residual.shape().dims()[ws.residual.shape().dims().len() - 1];
                unsafe {
                    crate::backend::cpu::neon::fused_matmul_f16(
                        ws.residual.as_ptr() as *const f32,
                        k,
                        &[
                            (
                                self.w_gate.as_ptr() as *const u16,
                                ws.gate.as_mut_ptr() as *mut f32,
                                self.w_gate.shape().dims()[0],
                            ),
                            (
                                self.w_up.as_ptr() as *const u16,
                                ws.up.as_mut_ptr() as *mut f32,
                                self.w_up.shape().dims()[0],
                            ),
                        ],
                    );
                }
            }
        } else {
            crate::core::thread_pool::get_pool().begin_batch();
            backend.matmul_transposed(&ws.residual, &self.w_gate, &mut ws.gate)?;
            backend.matmul_transposed(&ws.residual, &self.w_up, &mut ws.up)?;
            crate::core::thread_pool::get_pool().end_batch();
        }
        prof_record!(t, matmul_ffn);

        let t = prof_start!();
        backend.silu_mul(&mut ws.gate, &ws.up)?;
        prof_record!(t, silu_mul);

        let t = prof_start!();
        backend.matmul_transposed(&ws.gate, &self.w_down, &mut ws.down)?;
        prof_record!(t, matmul_ffn);

        // 10. Residual 2 — accumulate FFN result into x (skip connection)
        let t = prof_start!();
        backend.add_assign(x, &ws.down)?;
        prof_record!(t, add_assign);

        if let Some(ref mut p) = profiler {
            p.count += 1;
        }

        Ok(())
    }
}

impl LlamaLayer {
    /// Compute post-softmax attention scores for non-F32 KV cache (Q4_0, F16).
    /// This is a score-only pass — does NOT compute the attention output.
    /// Scores are written to `scores_out` in [n_heads_q, stride] layout.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn compute_attention_scores(
        q: &Tensor,
        k_cache: &Tensor,
        scores_out: &mut [f32],
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
        is_head_major: bool,
        capacity: usize,
        backend: &Arc<dyn Backend>,
    ) -> Result<()> {
        let stride = scores_out.len() / n_heads_q;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_rep = n_heads_q / n_heads_kv;

        // Read Q to CPU (always F32)
        let q_data: Vec<f32> = if backend.name() == "OpenCL" {
            let mut buf = vec![0.0f32; q.size() / 4];
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 4)
            };
            backend.read_buffer(q, bytes)?;
            buf
        } else {
            q.as_slice::<f32>().to_vec()
        };

        match k_cache.dtype() {
            DType::Q4_0 => {
                use crate::core::quant::{BlockQ4_0, QK4_0};
                let blocks_per_row = head_dim / QK4_0;

                // Read K cache to CPU
                let k_bytes = if backend.name() == "OpenCL" {
                    let mut buf = vec![0u8; k_cache.size()];
                    backend.read_buffer(k_cache, &mut buf)?;
                    buf
                } else {
                    let ptr = k_cache.as_ptr();
                    let len = k_cache.size();
                    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
                };
                let k_blocks = unsafe {
                    std::slice::from_raw_parts(
                        k_bytes.as_ptr() as *const BlockQ4_0,
                        k_bytes.len() / std::mem::size_of::<BlockQ4_0>(),
                    )
                };

                let score_chunks: Vec<&mut [f32]> =
                    scores_out.chunks_mut(stride).take(n_heads_q).collect();

                // Process each Q head
                for (h, scores_h) in score_chunks.into_iter().enumerate() {
                    let kv_h = h / n_rep;
                    let q_off = h * head_dim;
                    let q_vec = &q_data[q_off..q_off + head_dim];
                    let mut kv_f32 = vec![0.0f32; head_dim];

                    for t in 0..cache_seq_len {
                        let block_off = if is_head_major {
                            (kv_h * capacity + t) * blocks_per_row
                        } else {
                            (t * n_heads_kv + kv_h) * blocks_per_row
                        };
                        // Dequantize K row
                        for bi in 0..blocks_per_row {
                            let mut tmp = [0.0f32; QK4_0];
                            k_blocks[block_off + bi].dequantize(&mut tmp);
                            kv_f32[bi * QK4_0..(bi + 1) * QK4_0].copy_from_slice(&tmp);
                        }
                        let score: f32 = q_vec
                            .iter()
                            .zip(kv_f32[..head_dim].iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        scores_h[t] = score * scale;
                    }

                    // Softmax
                    let active = &mut scores_h[..cache_seq_len];
                    let max_v = active.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut sum_e = 0.0f32;
                    for s in active.iter_mut() {
                        *s = (*s - max_v).exp();
                        sum_e += *s;
                    }
                    let inv = 1.0 / sum_e;
                    for s in active.iter_mut() {
                        *s *= inv;
                    }
                }
            }
            DType::F16 => {
                // Read K cache to CPU as raw bytes
                let k_bytes: Vec<u8> = if backend.name() == "OpenCL" {
                    let mut buf = vec![0u8; k_cache.size()];
                    backend.read_buffer(k_cache, &mut buf)?;
                    buf
                } else {
                    let ptr = k_cache.as_ptr();
                    let len = k_cache.size();
                    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
                };
                let k_raw = k_bytes.as_ptr() as *const u16;

                let score_chunks: Vec<&mut [f32]> =
                    scores_out.chunks_mut(stride).take(n_heads_q).collect();

                for (h, scores_h) in score_chunks.into_iter().enumerate() {
                    let kv_h = h / n_rep;
                    let q_off = h * head_dim;

                    for t in 0..cache_seq_len {
                        let off = if is_head_major {
                            (kv_h * capacity + t) * head_dim
                        } else {
                            (t * n_heads_kv + kv_h) * head_dim
                        };

                        #[cfg(target_arch = "aarch64")]
                        let score = unsafe {
                            crate::backend::cpu::neon::CpuBackendNeon::vec_dot_f16_f32(
                                head_dim,
                                q_data.as_ptr().add(q_off),
                                k_raw.add(off),
                            )
                        };
                        #[cfg(not(target_arch = "aarch64"))]
                        let score = {
                            let k_f16 = unsafe {
                                std::slice::from_raw_parts(
                                    k_raw.add(off) as *const half::f16,
                                    head_dim,
                                )
                            };
                            let q_vec = &q_data[q_off..q_off + head_dim];
                            q_vec
                                .iter()
                                .zip(k_f16.iter())
                                .map(|(&a, &b)| a * b.to_f32())
                                .sum::<f32>()
                        };
                        scores_h[t] = score * scale;
                    }

                    let active = &mut scores_h[..cache_seq_len];
                    let max_v = active.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut sum_e = 0.0f32;
                    for s in active.iter_mut() {
                        *s = (*s - max_v).exp();
                        sum_e += *s;
                    }
                    let inv = 1.0 / sum_e;
                    for s in active.iter_mut() {
                        *s *= inv;
                    }
                }
            }
            _ => {
                // F32 should not reach here (handled by inline attention path)
            }
        }
        Ok(())
    }
}

pub struct LlamaForwardGenArgs<'a, C: KVCacheOps = KVCache> {
    pub x: &'a mut Tensor,
    pub kv_cache: &'a mut C,
    pub start_pos: usize,
    pub backend: &'a Arc<dyn Backend>,
    pub memory: &'a dyn Memory,
    pub ws: &'a mut super::workspace::LayerWorkspace,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub use_gpu_attn: bool,
    /// When true, compute attention scores into ws.scores even for non-F32 KV cache.
    /// Required for H2O/H2O+ score accumulation with Q4_0/F16 KV cache.
    pub need_scores: bool,
    pub head_dim: usize,
    /// Optional per-op profiler for timing breakdown.
    pub profiler: Option<&'a mut OpProfiler>,
}

pub struct LlamaLayerForwardArgs<'a, C: KVCacheOps = KVCache> {
    pub x: &'a mut Tensor,
    pub kv_cache: &'a mut C,
    pub start_pos: usize,
    pub backend: &'a Arc<dyn Backend>,
    pub memory: &'a dyn Memory,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub workspace: Option<&'a mut super::workspace::LayerWorkspace>,
    pub use_gpu_attn: bool,
    pub need_scores: bool,
    pub head_dim: usize,
    /// Optional per-op profiler for timing breakdown.
    pub profiler: Option<&'a mut OpProfiler>,
}

// OpProfiler has been moved to crate::profile::ops.
// Re-exported via `pub use crate::profile::ops::OpProfiler;` at the top of this file.

// ═══════════════════════════════════════════════════════════════════
// Phase 1: Verify attention scores are post-softmax probabilities
// ═══════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::shared_buffer::SharedBuffer;

    /// Replicate the softmax computation used in forward_gen (F32 inline path)
    /// to verify the mathematical property: sum(softmax(Q·K^T / sqrt(d))) ≈ 1.0.
    ///
    /// This tests the SAME algorithm used in lines 548-636 of forward_gen().
    #[test]
    fn test_inline_softmax_produces_valid_probabilities() {
        let head_dim = 64;
        let cache_seq_len = 8;
        let n_heads_q = 4;
        let n_heads_kv = 2;
        let n_rep = n_heads_q / n_heads_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Create Q: [n_heads_q, head_dim] with random-ish values
        let mut q_data = vec![0.0f32; n_heads_q * head_dim];
        for (i, v) in q_data.iter_mut().enumerate() {
            *v = ((i as f32 * 0.1).sin()) * 0.5;
        }

        // Create K cache: [cache_seq_len, n_heads_kv, head_dim] (SeqMajor)
        let mut k_data = vec![0.0f32; cache_seq_len * n_heads_kv * head_dim];
        for (i, v) in k_data.iter_mut().enumerate() {
            *v = ((i as f32 * 0.07).cos()) * 0.5;
        }

        let stride = cache_seq_len; // minimal stride = cache_seq_len
        let mut scores = vec![0.0f32; n_heads_q * stride];

        // Replicate the forward_gen inline path (non-NEON scalar path)
        for h in 0..n_heads_q {
            let kv_h = h / n_rep;
            let q_off = h * head_dim;
            let scores_h = &mut scores[h * stride..(h + 1) * stride];

            // Step 1: Q * K^T * scale
            for t in 0..cache_seq_len {
                let k_off = (t * n_heads_kv + kv_h) * head_dim; // SeqMajor
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_data[q_off + d] * k_data[k_off + d];
                }
                scores_h[t] = dot * scale;
            }

            // Step 2: Softmax (same algorithm as forward_gen lines 616-636)
            let active = &mut scores_h[..cache_seq_len];
            let max_val = active.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0f32;
            for s in active.iter_mut() {
                *s = (*s - max_val).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / sum_exp;
            for s in active.iter_mut() {
                *s *= inv_sum;
            }
        }

        // VERIFY: each head's scores sum to ~1.0
        for h in 0..n_heads_q {
            let head_scores = &scores[h * stride..h * stride + cache_seq_len];
            let sum: f32 = head_scores.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Head {} score sum = {} (expected ~1.0)",
                h,
                sum
            );
            // All values in [0, 1]
            for (t, &s) in head_scores.iter().enumerate() {
                assert!(
                    s >= 0.0 && s <= 1.0,
                    "Head {} token {} score = {} (must be in [0,1])",
                    h,
                    t,
                    s
                );
            }
        }
    }

    /// Verify compute_attention_scores() for F16 KV cache produces post-softmax.
    #[test]
    fn test_compute_attention_scores_f16_post_softmax() {
        let backend = Arc::new(CpuBackend::new());
        let head_dim = 64;
        let n_heads_q = 4;
        let n_heads_kv = 2;
        let capacity = 16;
        let cache_seq_len = 8;

        // Q tensor: [1, 1, n_heads_q * head_dim] = [1, 1, 256]
        let q_buf = Arc::new(SharedBuffer::new(n_heads_q * head_dim * 4, DType::F32));
        let q = Tensor::new(
            Shape::new(vec![1, 1, n_heads_q * head_dim]),
            q_buf,
            backend.clone(),
        );
        // Fill Q with values
        unsafe {
            let q_slice =
                std::slice::from_raw_parts_mut(q.as_mut_ptr() as *mut f32, n_heads_q * head_dim);
            for (i, v) in q_slice.iter_mut().enumerate() {
                *v = ((i as f32 * 0.1).sin()) * 0.3;
            }
        }

        // K cache tensor: [1, capacity, n_heads_kv, head_dim] in F16
        let k_buf = Arc::new(SharedBuffer::new(
            capacity * n_heads_kv * head_dim * 2, // F16 = 2 bytes
            DType::F16,
        ));
        let k_cache = Tensor::new(
            Shape::new(vec![1, capacity, n_heads_kv, head_dim]),
            k_buf,
            backend.clone(),
        );
        // Fill K with F16 values
        unsafe {
            let k_slice = std::slice::from_raw_parts_mut(
                k_cache.as_mut_ptr() as *mut half::f16,
                capacity * n_heads_kv * head_dim,
            );
            for (i, v) in k_slice.iter_mut().enumerate() {
                *v = half::f16::from_f32(((i as f32 * 0.07).cos()) * 0.3);
            }
        }

        let mut scores = vec![0.0f32; n_heads_q * capacity];

        LlamaLayer::compute_attention_scores(
            &q,
            &k_cache,
            &mut scores,
            n_heads_q,
            n_heads_kv,
            head_dim,
            cache_seq_len,
            false, // SeqMajor
            capacity,
            &(backend as Arc<dyn Backend>),
        )
        .unwrap();

        // Verify post-softmax: sum ≈ 1.0 per head
        let stride = scores.len() / n_heads_q;
        for h in 0..n_heads_q {
            let head_scores = &scores[h * stride..h * stride + cache_seq_len];
            let sum: f32 = head_scores.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "F16 Head {} score sum = {} (expected ~1.0)",
                h,
                sum
            );
            for (t, &s) in head_scores.iter().enumerate() {
                assert!(
                    s >= 0.0 && s <= 1.0,
                    "F16 Head {} token {} score = {} (must be in [0,1])",
                    h,
                    t,
                    s
                );
            }
        }
    }

    /// Verify that scores fed to accumulate_layer() are genuine post-softmax.
    /// Simulate the full pipeline: compute scores → accumulate → verify.
    #[test]
    fn test_accumulator_receives_post_softmax_scores() {
        use crate::core::attention_scores::AttentionScoreAccumulator;

        let n_heads_q = 4;
        let cache_seq_len = 8;
        let stride = cache_seq_len;

        // Simulate post-softmax scores (manually constructed)
        let mut scores = vec![0.0f32; n_heads_q * stride];
        for h in 0..n_heads_q {
            let head_scores = &mut scores[h * stride..h * stride + cache_seq_len];
            // Create a valid probability distribution
            let raw: Vec<f32> = (0..cache_seq_len)
                .map(|t| ((t as f32 + h as f32) * 0.5).exp())
                .collect();
            let sum: f32 = raw.iter().sum();
            for (t, &r) in raw.iter().enumerate() {
                head_scores[t] = r / sum;
            }
        }

        // Verify input is valid softmax
        for h in 0..n_heads_q {
            let sum: f32 = scores[h * stride..h * stride + cache_seq_len].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Feed to accumulator
        let mut acc = AttentionScoreAccumulator::new(16, n_heads_q, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();
        acc.accumulate_layer(&scores, stride, cache_seq_len, n_heads_q);
        acc.end_step();

        // Per-token importance = sum across heads of softmax probs
        // Each head sums to 1.0 → total per-token sum across all tokens = n_heads_q
        let imp = acc.importance_scores();
        let total: f32 = imp[..cache_seq_len].iter().sum();
        assert!(
            (total - n_heads_q as f32).abs() < 1e-4,
            "Total importance = {} (expected {} = n_heads_q * 1.0 per head)",
            total,
            n_heads_q
        );
    }
}
