use crate::core::tensor::Tensor;
use crate::core::backend::Backend;
use crate::core::memory::Memory;
use crate::core::kv_cache::KVCache;
use crate::core::shape::Shape;
use crate::core::buffer::DType;
use anyhow::Result;
use crate::memory::galloc::Galloc;
use crate::backend::cpu::CpuBackend;
use std::sync::Arc;
use rayon::prelude::*;

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
    pub fn forward(&self, args: LlamaLayerForwardArgs) -> Result<()> {
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
        let head_dim = 64;

        if seq_len == 1 {
            if let Some(ws) = workspace {
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
                });
            }
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
             backend.clone()
        );
        let mut k_rope = Tensor::new(
             Shape::new(vec![batch_size, seq_len, n_heads_kv, head_dim]),
             k.buffer().clone(),
             backend.clone()
        );

        backend.rope_inplace(&mut q_rope, start_pos, rope_theta)?;
        backend.rope_inplace(&mut k_rope, start_pos, rope_theta)?;

        // Cast to target dtype if KV cache is not F32
        let kv_dtype = kv_cache.k_buffer.dtype();
        if kv_dtype != DType::F32 {
            let n_elem = seq_len * n_heads_kv * head_dim;
            let buf_size = match kv_dtype {
                DType::F16 => n_elem * 2,
                DType::Q4_0 => (n_elem / crate::core::quant::QK4_0) * std::mem::size_of::<crate::core::quant::BlockQ4_0>(),
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
        
        let cache_seq_len = kv_cache.current_pos; 
        let (k_cache, v_cache) = kv_cache.get_view(0); 
        
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
                q_vec.resize(q_rope.size() / 4, 0.0);
                k_vec.resize(k_cache.size() / 4, 0.0);
                v_vec.resize(v_cache.size() / 4, 0.0);
                out_vec.resize(out_attn.size() / 4, 0.0);
    
                backend.read_buffer(&q_rope, as_u8_mut(&mut q_vec))?;
                backend.read_buffer(&k_cache, as_u8_mut(&mut k_vec))?;
                backend.read_buffer(&v_cache, as_u8_mut(&mut v_vec))?;
    
                (&q_vec[..], &k_vec[..], &v_vec[..], &mut out_vec[..])
            } else if k_cache.dtype() == DType::Q4_0 {
                // Q4_0: dequantize KV cache to F32 temp buffers
                use crate::core::quant::{BlockQ4_0, QK4_0};
                let n_elems = cache_seq_len * n_heads_kv * head_dim;
                let n_blocks = n_elems / QK4_0;
                let k_blocks = unsafe { std::slice::from_raw_parts(k_cache.as_ptr() as *const BlockQ4_0, n_blocks) };
                let v_blocks = unsafe { std::slice::from_raw_parts(v_cache.as_ptr() as *const BlockQ4_0, n_blocks) };
                k_vec.resize(n_elems, 0.0f32);
                v_vec.resize(n_elems, 0.0f32);
                for bi in 0..n_blocks {
                    let mut tmp = [0.0f32; QK4_0];
                    k_blocks[bi].dequantize(&mut tmp);
                    k_vec[bi * QK4_0..(bi + 1) * QK4_0].copy_from_slice(&tmp);
                    v_blocks[bi].dequantize(&mut tmp);
                    v_vec[bi * QK4_0..(bi + 1) * QK4_0].copy_from_slice(&tmp);
                }
                (q_rope.as_slice::<f32>(), &k_vec[..], &v_vec[..], out_attn.as_mut_slice::<f32>())
            } else if k_cache.dtype() == DType::F16 {
                // F16: convert KV cache to F32 temp buffers
                let n_elems = cache_seq_len * n_heads_kv * head_dim;
                let k_f16 = k_cache.as_slice::<half::f16>();
                let v_f16 = v_cache.as_slice::<half::f16>();
                k_vec.resize(n_elems, 0.0f32);
                v_vec.resize(n_elems, 0.0f32);
                for i in 0..n_elems {
                    k_vec[i] = k_f16[i].to_f32();
                    v_vec[i] = v_f16[i].to_f32();
                }
                (q_rope.as_slice::<f32>(), &k_vec[..], &v_vec[..], out_attn.as_mut_slice::<f32>())
            } else {
                 (q_rope.as_slice::<f32>(), k_cache.as_slice::<f32>(), v_cache.as_slice::<f32>(), out_attn.as_mut_slice::<f32>())
            };
            
            use crate::layers::attention::flash_attention_forward;

            let chunk_q_stride = seq_len * n_heads_q * head_dim;
            let chunk_out_stride = seq_len * n_heads_q * head_dim;
            // KV Cache is strided by max_seq_len because it is allocated as [Batch, MaxSeq, ...]
            let chunk_k_stride = kv_cache.max_seq_len * n_heads_kv * head_dim; 

            // Iterate over batch. 
            // We use chunks_mut for out_ptr to satisfy borrow checker.
            for (b, out_batch) in out_ptr.chunks_mut(chunk_out_stride).enumerate() {
                 let q_start = b * chunk_q_stride;
                 let k_start = b * chunk_k_stride;
                 let v_start = b * chunk_k_stride; // V has same layout as K in cache (usually)

                 let q_slice = &q_data[q_start..q_start + chunk_q_stride];
                 
                 // K/V Cache: We only want the VALID part [0..cache_seq_len]
                 // But the buffer for this batch is huge (max_seq_len).
                 // We pass the slice covering valid data.
                 // Strides passed to flash_attention must match this slice logic.
                 // If we pass a slice starting at k_start, the "row 0" is k_data[k_start].
                 // The "row 1" is k_data[k_start + k_stride].
                 // k_stride = n_heads_kv * head_dim.
                 // This matches the cache layout (dense in seq dimension).
                 // The valid data length is cache_seq_len.
                 
                 let k_valid_len = cache_seq_len * n_heads_kv * head_dim;
                 let k_slice = &k_data[k_start..k_start + k_valid_len];
                 let v_slice = &v_data[v_start..v_start + k_valid_len];

                 flash_attention_forward(
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
                     n_heads_kv * head_dim, // k_stride
                     n_heads_kv * head_dim, // v_stride
                     n_heads_q * head_dim, // out_stride
                     start_pos, // q_start_pos for causal mask
                     32, // br
                     32, // bc
                 );
            }
        }

        if is_opencl {
             // Create temp CPU tensor from result and copy back
             // Using Galloc directly
             let size_bytes = out_vec.len() * 4;
             let buf = Galloc::new().alloc(size_bytes, DType::F32)?;
             unsafe { std::ptr::copy_nonoverlapping(out_vec.as_ptr(), buf.as_mut_ptr() as *mut f32, out_vec.len()); }
             let cpu_out = Tensor::new(out_attn.shape().clone(), buf, Arc::new(CpuBackend::new()));
             out_attn = backend.copy_from(&cpu_out)?;
        }
        
        let mut attn_out_projected = self.alloc_temp(vec![batch_size, seq_len, dim], memory, backend)?;
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

    fn alloc_temp(&self, shape: Vec<usize>, memory: &dyn Memory, backend: &Arc<dyn Backend>) -> Result<Tensor> {
        let size: usize = shape.iter().product();
        let buf = memory.alloc(size * 4, DType::F32)?;
        Ok(Tensor::new(Shape::new(shape), buf, backend.clone()))
    }

    /// Fast path for single token generation using pre-allocated workspace.
    fn forward_gen(&self, args: LlamaForwardGenArgs) -> Result<()> {
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
        let head_dim = 64;
        
        // 1. Attention Norm
        // x and ws.residual are tensors. Use copy_from for OpenCL compatibility (allocs new buffer but correct)

        ws.residual = backend.copy_from(x)?;

        backend.rms_norm(x, &self.attention_norm, rms_norm_eps)?;
        
        // 2. QKV Projections

        backend.matmul_transposed(x, &self.wq, &mut ws.q)?;
        backend.matmul_transposed(x, &self.wk, &mut ws.k)?;
        backend.matmul_transposed(x, &self.wv, &mut ws.v)?;
        
        // 3. RoPE

        let q_dim = self.wq.shape().dims()[0];
        let k_dim = self.wk.shape().dims()[0];
        let n_heads_q = q_dim / head_dim;
        let n_heads_kv = k_dim / head_dim;
        
        let mut q_rope = Tensor::new(
            Shape::new(vec![batch_size, 1, n_heads_q, head_dim]),
            ws.q.buffer().clone(),
            backend.clone()
        );
        let mut k_rope = Tensor::new(
            Shape::new(vec![batch_size, 1, n_heads_kv, head_dim]),
            ws.k.buffer().clone(),
            backend.clone()
        );
        
        backend.rope_inplace(&mut q_rope, start_pos, rope_theta)?;
        backend.rope_inplace(&mut k_rope, start_pos, rope_theta)?;
        
        // 4. KV Cache Update - cast to target dtype if needed
        let kv_dtype = kv_cache.k_buffer.dtype();
        if kv_dtype != DType::F32 {
            let n_elem = n_heads_kv * head_dim;
            let buf_size = match kv_dtype {
                DType::F16 => n_elem * 2,
                DType::Q4_0 => (n_elem / crate::core::quant::QK4_0) * std::mem::size_of::<crate::core::quant::BlockQ4_0>(),
                _ => n_elem * 4, // fallback
            };
            let k_cast_buf = memory.alloc(buf_size, kv_dtype)?;
            let mut k_cast = Tensor::new(k_rope.shape().clone(), k_cast_buf, backend.clone());
            backend.cast(&k_rope, &mut k_cast)?;
            let v_cast_buf = memory.alloc(buf_size, kv_dtype)?;
            let mut v_cast = Tensor::new(
                Shape::new(vec![batch_size, 1, n_heads_kv, head_dim]),
                v_cast_buf, backend.clone()
            );
            backend.cast(&ws.v, &mut v_cast)?;
            kv_cache.update(&k_cast, &v_cast)?;
        } else {
            kv_cache.update(&k_rope, &ws.v)?;
        }
        
        // 5. Attention - use GPU kernel for OpenCL
        let cache_seq_len = kv_cache.current_pos;
        let (k_cache, v_cache) = kv_cache.get_view(0);

        if (backend.name() == "OpenCL" && use_gpu_attn) || k_cache.dtype() != DType::F32 {
            // GPU attention or F16 KV cache - use backend's dtype-aware implementation
            backend.attention_gen(&q_rope, &k_cache, &v_cache, &mut ws.out_attn,
                                  n_heads_q, n_heads_kv, head_dim, cache_seq_len)?;
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
                     unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
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
                     ws.out_attn.as_mut_slice::<f32>()
                 )
            };

            // Re-interpret out_attn (or out_vec) as raw slice, will fill with zeros first
            // Note: out_ptr is &mut [f32]
            for x in out_ptr.iter_mut() { *x = 0.0; }

            let scale = 1.0 / (head_dim as f32).sqrt();
            let n_rep = n_heads_q / n_heads_kv;
            
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

                scores_chunks.zip(out_chunks).enumerate().for_each(|(h, (scores_h, out_h))| {
                    let kv_h = h / n_rep;
                    
                    // Unsafe access for performance
                    unsafe {
                        let q_off = h * head_dim;
                        let q_ptr = q_data.as_ptr().add(q_off);
                        
                        // --- Step 1: Q * K^T (NEON Vectorized) ---
                        for t in 0..cache_seq_len {
                            let k_off = (t * n_heads_kv + kv_h) * head_dim;
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
                            
                            #[cfg(not(target_arch = "aarch64"))]
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
                            if s > max_val { max_val = s; }
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
                        for x in out_h.iter_mut() { *x = 0.0; }
                        
                        for t in 0..cache_seq_len {
                            let weight = *active_scores.get_unchecked(t);
                            let v_off = (t * n_heads_kv + kv_h) * head_dim;
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
                            
                            #[cfg(not(target_arch = "aarch64"))]
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

                scores_chunks.zip(out_chunks).enumerate().for_each(|(h, (scores_h, out_h))| {
                    let kv_h = h / n_rep;
                     unsafe {
                        let q_off = h * head_dim;
                        let q_ptr = q_data.as_ptr().add(q_off);
                        
                        // --- Step 1: Q * K^T (NEON Vectorized) ---
                        for t in 0..cache_seq_len {
                            let k_off = (t * n_heads_kv + kv_h) * head_dim;
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
                            #[cfg(not(target_arch = "aarch64"))]
                            let score = {
                                let mut score = 0.0;
                                for i in 0..head_dim { score += *q_ptr.add(i) * *k_ptr.add(i); }
                                score
                            };
                            *scores_h.get_unchecked_mut(t) = score * scale;
                        }
                        
                        // --- Step 2: Softmax ---
                        let active_scores = &mut scores_h[0..cache_seq_len];
                        let mut max_val = f32::NEG_INFINITY;
                        for i in 0..cache_seq_len {
                            let s = *active_scores.get_unchecked(i);
                            if s > max_val { max_val = s; }
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
                            while i + 4 <= head_dim { vst1q_f32(out_h.as_mut_ptr().add(i), zero); i += 4; }
                            while i < head_dim { *out_h.get_unchecked_mut(i) = 0.0; i += 1; }
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        for x in out_h.iter_mut() { *x = 0.0; }
                        
                        for t in 0..cache_seq_len {
                            let weight = *active_scores.get_unchecked(t);
                            let v_off = (t * n_heads_kv + kv_h) * head_dim;
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
                            #[cfg(not(target_arch = "aarch64"))]
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
                unsafe { std::ptr::copy_nonoverlapping(out_vec.as_ptr(), buf.as_mut_ptr() as *mut f32, out_vec.len()); }
                let cpu_out = Tensor::new(ws.out_attn.shape().clone(), buf, Arc::new(CpuBackend::new()));
                // Use backend.copy_from to transfer back to GPU tensor ws.out_attn
                // Note: ws.out_attn is &mut Tensor, so this updates the GPU buffer contents
                ws.out_attn = backend.copy_from(&cpu_out)?;
            }
        }
        
        // 6. Output Projection

        backend.matmul_transposed(&ws.out_attn, &self.wo, &mut ws.attn_out)?;
        
        // 7. Residual 1

        backend.add_assign(&mut ws.attn_out, &ws.residual)?;
        
        // Copy to x for next stage
        *x = backend.copy_from(&ws.attn_out)?;
        
        // 8. FFN Norm

        ws.residual = backend.copy_from(x)?;
        backend.rms_norm(x, &self.ffn_norm, rms_norm_eps)?;
        
        // 9. FFN

        backend.matmul_transposed(x, &self.w_gate, &mut ws.gate)?;
        backend.matmul_transposed(x, &self.w_up, &mut ws.up)?;
        backend.silu_mul(&mut ws.gate, &ws.up)?;
        backend.matmul_transposed(&ws.gate, &self.w_down, &mut ws.down)?;
        
        // 10. Residual 2

        backend.add_assign(&mut ws.down, &ws.residual)?;
        
        // Copy to x for next layer
        *x = backend.copy_from(&ws.down)?;

        
        Ok(())
    }
}

pub struct LlamaForwardGenArgs<'a> {
    pub x: &'a mut Tensor,
    pub kv_cache: &'a mut KVCache,
    pub start_pos: usize,
    pub backend: &'a Arc<dyn Backend>,
    pub memory: &'a dyn Memory,
    pub ws: &'a mut super::workspace::LayerWorkspace,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub use_gpu_attn: bool,
}

pub struct LlamaLayerForwardArgs<'a> {
    pub x: &'a mut Tensor,
    pub kv_cache: &'a mut KVCache,
    pub start_pos: usize,
    pub backend: &'a Arc<dyn Backend>,
    pub memory: &'a dyn Memory,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub workspace: Option<&'a mut super::workspace::LayerWorkspace>,
    pub use_gpu_attn: bool,
}
