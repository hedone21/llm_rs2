use super::*;

impl TransformerLayer {
    /// Fast path for single token generation using pre-allocated workspace.
    pub(super) fn forward_gen<C: KVCacheOps>(&self, mut args: ForwardGenArgs<C>) -> Result<()> {
        // SWIFT: if both sub-layers are skipped, early return (identity)
        if args.skip_attn && args.skip_mlp {
            return Ok(());
        }

        let _skip_attn = args.skip_attn;
        let _skip_mlp = args.skip_mlp;
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

        let rms_norm_add_unit = args.rms_norm_add_unit;
        let use_gelu_tanh = args.use_gelu_tanh;

        // 1. Attention Norm — out-of-place: ws.residual = norm(x), x preserved for skip connection
        let t = prof_start!();
        backend.rms_norm_oop(
            x,
            &mut ws.residual,
            &self.attention_norm,
            rms_norm_eps,
            rms_norm_add_unit,
        )?;
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
        if is_opencl {
            backend.flush()?;
        }
        prof_record!(t, matmul_qkv);

        // QKV bias extension point
        if let Some(ref bias) = self.qkv_bias {
            backend.add_row_bias(&mut ws.q, &bias.bq)?;
            backend.add_row_bias(&mut ws.k, &bias.bk)?;
            backend.add_row_bias(&mut ws.v, &bias.bv)?;
        }

        // 3. RoPE
        let t = prof_start!();
        let q_dim = self.wq.shape().dims()[0];
        let k_dim = self.wk.shape().dims()[0];
        let n_heads_q = q_dim / head_dim;
        let n_heads_kv = k_dim / head_dim;

        // Gemma3 QK-Norm: per-head RMSNorm on Q and K before RoPE
        if let Some(ref q_norm_w) = self.q_norm {
            // ws.q shape: [batch, 1, n_heads_q * head_dim] → reshape to [batch*n_heads_q, head_dim]
            let total_q_heads = batch_size * n_heads_q;
            let saved_shape = ws.q.shape().clone();
            ws.q.reshape(Shape::new(vec![total_q_heads, head_dim]));
            backend.rms_norm(&mut ws.q, q_norm_w, rms_norm_eps, true)?;
            ws.q.reshape(saved_shape);
        }
        if let Some(ref k_norm_w) = self.k_norm {
            // ws.k shape: [batch, 1, n_heads_kv * head_dim] → reshape to [batch*n_heads_kv, head_dim]
            let total_k_heads = batch_size * n_heads_kv;
            let saved_shape = ws.k.shape().clone();
            ws.k.reshape(Shape::new(vec![total_k_heads, head_dim]));
            backend.rms_norm(&mut ws.k, k_norm_w, rms_norm_eps, true)?;
            ws.k.reshape(saved_shape);
        }

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
            // Ensure capacity before direct scatter to prevent out-of-bounds GPU write
            kv_cache.ensure_capacity(kv_cache.current_pos() + 1)?;
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
            super::update_kv_cache(kv_cache, &k_rope, &ws.v, backend)?;
        }
        prof_record!(t, kv_update);

        // 5. Attention - use GPU kernel for OpenCL
        let t = prof_start!();
        let cache_seq_len = kv_cache.current_pos();

        // Sliding window attention (Gemma3 local layers):
        // Restrict attention to the most recent `window_size` tokens.
        let effective_cache_len = if let Some(true) = args.is_local_attn {
            let window = args.local_attn_window.unwrap_or(usize::MAX);
            cache_seq_len.min(window)
        } else {
            cache_seq_len
        };
        // Physical start offset in the KV cache for the window.
        let kv_start_pos = cache_seq_len - effective_cache_len;

        let (k_cache, v_cache) = kv_cache.get_view();

        // AWQE: cache가 attention scores를 요구하면 score 계산 강제
        let need_scores = args.need_scores || kv_cache.needs_attn_scores();

        // Use dtype-aware attention for non-F32 KV caches (F16, Q4_0).
        // On OpenCL: also guard that KV buffers are actual GPU buffers (not CPU-only
        // KiviCache with SharedBuffer) — CPU-only caches must use the F32 fallback.
        #[cfg(feature = "opencl")]
        let kv_is_gpu = k_cache.buffer().cl_mem().is_some();
        #[cfg(not(feature = "opencl"))]
        let kv_is_gpu = true;
        let use_typed_attn = if backend.name() == "OpenCL" {
            (use_gpu_attn || k_cache.dtype() != DType::F32) && kv_is_gpu
        } else {
            // CPU backend: always use typed attention for non-F32 KV cache.
            // CpuBackend::attention_gen handles F16/Q4_0 via dequantization.
            k_cache.dtype() != DType::F32
        };
        if use_typed_attn {
            // GPU attention or F16 KV cache - use backend's dtype-aware implementation.
            // When need_scores is true, attention_gen() writes post-softmax scores
            // directly into ws.scores, eliminating the separate CPU score recomputation.
            backend.attention_gen(
                &q_rope,
                &k_cache,
                &v_cache,
                &mut ws.out_attn,
                n_heads_q,
                n_heads_kv,
                head_dim,
                effective_cache_len,
                if need_scores {
                    Some(&mut ws.scores)
                } else {
                    None
                },
            )?;
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
            if all_scores.len() < n_heads_q * effective_cache_len {
                // Fallback or panic, but expecting correct size
                // Dynamic resize just in case (e.g. if max_seq_len valid but cache grew?)
                // Actually effective_cache_len <= cache_seq_len <= max_seq_len.
            }

            // Parallelize over heads
            // 1. Prepare mutable slices for scores and output
            //    scores: [n_heads_q, max_seq_len] -> split into chunks of max_seq_len
            //    out:    [n_heads_q, head_dim] -> split into chunks of head_dim

            // Parallelize over heads for longer sequences, Serial for short (to avoid overhead)
            let use_parallel = effective_cache_len >= 256;

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
                            // t iterates over [0, effective_cache_len); physical position = kv_start_pos + t
                            for t in 0..effective_cache_len {
                                let phys_t = kv_start_pos + t;
                                let k_off = if is_head_major {
                                    (kv_h * kv_capacity + phys_t) * head_dim
                                } else {
                                    (phys_t * n_heads_kv + kv_h) * head_dim
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
                            let active_scores = &mut scores_h[0..effective_cache_len];

                            let mut max_val = f32::NEG_INFINITY;
                            for i in 0..effective_cache_len {
                                let s = *active_scores.get_unchecked(i);
                                if s > max_val {
                                    max_val = s;
                                }
                            }

                            let mut sum_exp = 0.0;
                            for i in 0..effective_cache_len {
                                let s = (*active_scores.get_unchecked(i) - max_val).exp();
                                *active_scores.get_unchecked_mut(i) = s;
                                sum_exp += s;
                            }

                            let inv_sum = 1.0 / sum_exp;
                            for i in 0..effective_cache_len {
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

                            for t in 0..effective_cache_len {
                                let phys_t = kv_start_pos + t;
                                let weight = *active_scores.get_unchecked(t);
                                let v_off = if is_head_major {
                                    (kv_h * kv_capacity + phys_t) * head_dim
                                } else {
                                    (phys_t * n_heads_kv + kv_h) * head_dim
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
                            // t iterates over [0, effective_cache_len); physical position = kv_start_pos + t
                            for t in 0..effective_cache_len {
                                let phys_t = kv_start_pos + t;
                                let k_off = if is_head_major {
                                    (kv_h * kv_capacity + phys_t) * head_dim
                                } else {
                                    (phys_t * n_heads_kv + kv_h) * head_dim
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
                            let active_scores = &mut scores_h[0..effective_cache_len];
                            let mut max_val = f32::NEG_INFINITY;
                            for i in 0..effective_cache_len {
                                let s = *active_scores.get_unchecked(i);
                                if s > max_val {
                                    max_val = s;
                                }
                            }
                            let mut sum_exp = 0.0;
                            for i in 0..effective_cache_len {
                                let s = (*active_scores.get_unchecked(i) - max_val).exp();
                                *active_scores.get_unchecked_mut(i) = s;
                                sum_exp += s;
                            }
                            let inv_sum = 1.0 / sum_exp;
                            for i in 0..effective_cache_len {
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

                            for t in 0..effective_cache_len {
                                let phys_t = kv_start_pos + t;
                                let weight = *active_scores.get_unchecked(t);
                                let v_off = if is_head_major {
                                    (kv_h * kv_capacity + phys_t) * head_dim
                                } else {
                                    (phys_t * n_heads_kv + kv_h) * head_dim
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

        // Store post-softmax scores for KiviCache AWQE (used during next flush).
        {
            let stride = ws.scores.len() / n_heads_q;
            kv_cache.set_attn_scores(&ws.scores, n_heads_q, stride, effective_cache_len);
        }

        // 6. Output Projection
        prof_record!(t, attention);

        let t = prof_start!();
        backend.matmul_transposed(&ws.out_attn, &self.wo, &mut ws.attn_out)?;
        prof_record!(t, matmul_wo);

        // 7+8. Post-attention residual + pre-FFN norm
        let t = prof_start!();
        if rms_norm_add_unit {
            // Gemma3: apply post-attention norm (ffn_norm) to attn_out before residual add,
            // then apply pre_ffn_norm to get ws.residual for FFN input.
            backend.rms_norm(&mut ws.attn_out, &self.ffn_norm, rms_norm_eps, true)?;
            backend.add_assign(x, &ws.attn_out)?;
            if let Some(ref pfn) = self.pre_ffn_norm {
                backend.rms_norm_oop(x, &mut ws.residual, pfn, rms_norm_eps, true)?;
            } else {
                // Fallback: no pre_ffn_norm (should not happen for Gemma3)
                backend.copy_into(x, &mut ws.residual)?;
            }
        } else {
            // Llama/Qwen2: fused add + norm (ffn_norm as pre-FFN norm)
            backend.add_rms_norm_oop(
                x,
                &ws.attn_out,
                &mut ws.residual,
                &self.ffn_norm,
                rms_norm_eps,
                false,
            )?;
        }
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
        if is_opencl {
            backend.flush()?;
        }
        prof_record!(t, matmul_ffn);

        let t = prof_start!();
        if use_gelu_tanh {
            backend.gelu_tanh_mul(&mut ws.gate, &ws.up)?;
        } else {
            backend.silu_mul(&mut ws.gate, &ws.up)?;
        }
        prof_record!(t, silu_mul);

        let t = prof_start!();
        backend.matmul_transposed(&ws.gate, &self.w_down, &mut ws.down)?;
        prof_record!(t, matmul_ffn);

        // 10. Residual 2 — accumulate FFN result into x (skip connection)
        let t = prof_start!();
        // Gemma3: apply post-FFN norm to FFN output before residual add.
        if let Some(ref pfn) = self.post_ffn_norm {
            backend.rms_norm(&mut ws.down, pfn, rms_norm_eps, true)?;
        }
        backend.add_assign(x, &ws.down)?;
        prof_record!(t, add_assign);

        if let Some(ref mut p) = profiler {
            p.count += 1;
        }

        Ok(())
    }
}

impl TransformerLayer {
    /// Compute post-softmax attention scores for non-F32 KV cache (Q4_0, F16).
    /// This is a score-only pass — does NOT compute the attention output.
    /// Scores are written to `scores_out` in [n_heads_q, stride] layout.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    pub(super) fn compute_attention_scores(
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
