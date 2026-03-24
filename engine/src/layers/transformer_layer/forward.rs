use super::*;

impl TransformerLayer {
    /// Standard forward path (Prefill or dynamic generation)
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_prefill<C: KVCacheOps>(
        &self,
        x: &mut Tensor,
        kv_cache: &mut C,
        start_pos: usize,
        backend: &Arc<dyn Backend>,
        memory: &dyn Memory,
        rms_norm_eps: f32,
        rope_theta: f32,
        _use_gpu_attn: bool,
        _need_scores: bool,
        head_dim: usize,
        batch_size: usize,
        seq_len: usize,
        dim: usize,
        _skip_attn: bool,
        _skip_mlp: bool,
    ) -> Result<()> {
        // Standard forward path (Prefill or dynamic generation)
        let residual = backend.copy_from(x)?;
        backend.rms_norm(x, &self.attention_norm, rms_norm_eps, false)?;

        let q_dim = self.wq.shape().dims()[0];
        let k_dim = self.wk.shape().dims()[0];
        let v_dim = self.wv.shape().dims()[0];

        let mut q = self.alloc_temp(vec![batch_size, seq_len, q_dim], memory, backend)?;
        let mut k = self.alloc_temp(vec![batch_size, seq_len, k_dim], memory, backend)?;
        let mut v = self.alloc_temp(vec![batch_size, seq_len, v_dim], memory, backend)?;

        backend.matmul_transposed(x, &self.wq, &mut q)?;
        backend.matmul_transposed(x, &self.wk, &mut k)?;
        backend.matmul_transposed(x, &self.wv, &mut v)?;

        // QKV bias extension point
        if let Some(ref bias) = self.qkv_bias {
            backend.add_row_bias(&mut q, &bias.bq)?;
            backend.add_row_bias(&mut k, &bias.bk)?;
            backend.add_row_bias(&mut v, &bias.bv)?;
        }

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
        let kv_capacity = kv_cache.capacity();
        let kv_layout = kv_cache.layout();
        let (k_cache, v_cache) = kv_cache.get_view();

        let mut out_attn = self.alloc_temp(vec![batch_size, seq_len, q_dim], memory, backend)?;

        let is_opencl = backend.name() == "OpenCL";

        // Try GPU flash attention first (avoids GPU→CPU→GPU roundtrip)
        let gpu_dispatched = if is_opencl {
            if let Some(ocl_backend) = backend
                .as_any()
                .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            {
                ocl_backend.flash_attention_prefill(
                    &q_rope,
                    &k_cache,
                    &v_cache,
                    &mut out_attn,
                    n_heads_q,
                    n_heads_kv,
                    seq_len,
                    cache_seq_len,
                    head_dim,
                    kv_capacity,
                    batch_size,
                    kv_layout == KVLayout::HeadMajor,
                )?
            } else {
                false
            }
        } else {
            false
        };

        if !gpu_dispatched {
            // ---- CPU flash attention path (also used as GPU fallback) ----
            let mut out_vec = Vec::new();

            {
                // Helper to cast slice
                fn as_u8_mut(v: &mut [f32]) -> &mut [u8] {
                    unsafe {
                        std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4)
                    }
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
                    let n_elems = if kv_layout == KVLayout::HeadMajor {
                        n_heads_kv * kv_capacity * head_dim
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
                    let n_elems = if kv_layout == KVLayout::HeadMajor {
                        n_heads_kv * kv_capacity * head_dim
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

                let is_head_major_pf = kv_layout == KVLayout::HeadMajor;

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
                let cpu_out =
                    Tensor::new(out_attn.shape().clone(), buf, Arc::new(CpuBackend::new()));
                out_attn = backend.copy_from(&cpu_out)?;
            }
        }

        let mut attn_out_projected =
            self.alloc_temp(vec![batch_size, seq_len, dim], memory, backend)?;
        backend.matmul_transposed(&out_attn, &self.wo, &mut attn_out_projected)?;

        backend.add_assign(&mut attn_out_projected, &residual)?;
        *x = attn_out_projected;

        let residual_ffn = backend.copy_from(x)?;
        backend.rms_norm(x, &self.ffn_norm, rms_norm_eps, false)?;

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
}
