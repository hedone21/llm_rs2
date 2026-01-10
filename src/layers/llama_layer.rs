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
    pub fn forward(
        &self,
        x: &mut Tensor,
        kv_cache: &mut KVCache,
        start_pos: usize,
        backend: &Arc<dyn Backend>,
        memory: &dyn Memory,
        rms_norm_eps: f32,
        rope_theta: f32,
        workspace: Option<&mut super::workspace::LayerWorkspace>,
    ) -> Result<()> {
        let batch_size = x.shape().dims()[0];
        let seq_len = x.shape().dims()[1];
        let dim = x.shape().dims()[2];
        let head_dim = 64;

        if seq_len == 1 {
            if let Some(ws) = workspace {
                return self.forward_gen(x, kv_cache, start_pos, backend, ws, rms_norm_eps, rope_theta);
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

        kv_cache.update(&k_rope, &v)?;
        
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
            } else {
                 (q_rope.as_slice::<f32>(), k_cache.as_slice::<f32>(), v_cache.as_slice::<f32>(), out_attn.as_mut_slice::<f32>())
            };
            
            let scale = 1.0 / (head_dim as f32).sqrt();
            
            for b in 0..batch_size {
                for h in 0..n_heads_q {
                    let kv_h = h / (n_heads_q / n_heads_kv); 
                    
                    for t in 0..seq_len {
                         let q_off = ((b * seq_len + t) * n_heads_q + h) * head_dim;
                         let q_vec = &q_data[q_off..q_off + head_dim];
                         
                         let mut scores = vec![0.0; cache_seq_len];
                         for ct in 0..cache_seq_len {
                             let global_q_pos = start_pos + t;
                             let global_k_pos = ct;
                             
                             if global_k_pos > global_q_pos {
                                 scores[ct] = f32::NEG_INFINITY;
                                 continue;
                             }
                             
                             let k_off = ((b * kv_cache.max_seq_len + ct) * n_heads_kv + kv_h) * head_dim;
                             let k_vec = &k_data[k_off..k_off + head_dim];
                             
                             let score: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                             scores[ct] = score * scale;
                         }
                         
                         let max_val = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                         let mut sum_exp = 0.0;
                         for s in scores.iter_mut() {
                             *s = (*s - max_val).exp();
                             sum_exp += *s;
                         }
                         for s in scores.iter_mut() { *s /= sum_exp; }
                         
                         let mut out_vec = vec![0.0; head_dim];
                         for ct in 0..cache_seq_len {
                             let weight = scores[ct];
                             let v_off = ((b * kv_cache.max_seq_len + ct) * n_heads_kv + kv_h) * head_dim;
                             let v_vec = &v_data[v_off..v_off + head_dim];
                             
                             for d in 0..head_dim {
                                 out_vec[d] += weight * v_vec[d];
                             }
                         }
                         
                         let out_off = ((b * seq_len + t) * n_heads_q + h) * head_dim;
                         for d in 0..head_dim {
                             out_ptr[out_off + d] = out_vec[d];
                         }
                    }
                }
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
    fn forward_gen(
        &self,
        x: &mut Tensor,
        kv_cache: &mut KVCache,
        start_pos: usize,
        backend: &Arc<dyn Backend>,
        ws: &mut super::workspace::LayerWorkspace,
        rms_norm_eps: f32,
        rope_theta: f32,
    ) -> Result<()> {
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
        
        // 4. KV Cache Update

        kv_cache.update(&k_rope, &ws.v)?;
        
        // 5. Attention
        // 5. Attention
        let cache_seq_len = kv_cache.current_pos;
        let (k_cache, v_cache) = kv_cache.get_view(0);

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
                out_vec.resize(ws.out_attn.size() / 4, 0.0);
    
                backend.read_buffer(&q_rope, as_u8_mut(&mut q_vec))?;
                backend.read_buffer(&k_cache, as_u8_mut(&mut k_vec))?;
                backend.read_buffer(&v_cache, as_u8_mut(&mut v_vec))?;

    
                (&q_vec[..], &k_vec[..], &v_vec[..], &mut out_vec[..])
            } else {
                 (q_rope.as_slice::<f32>(), k_cache.as_slice::<f32>(), v_cache.as_slice::<f32>(), ws.out_attn.as_mut_slice::<f32>())
            };
        
            let scale = 1.0 / (head_dim as f32).sqrt();

            for h in 0..n_heads_q {
                let kv_h = h / (n_heads_q / n_heads_kv);
                let q_off = h * head_dim;
                let q_vec = &q_data[q_off..q_off + head_dim];
                
                let scores = &mut ws.scores[..cache_seq_len];
                for ct in 0..cache_seq_len {
                    let k_off = (ct * n_heads_kv + kv_h) * head_dim;
                    let k_vec = &k_data[k_off..k_off + head_dim];
                    
                    let score: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                    scores[ct] = score * scale;
                }
                
                let max_val = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum_exp = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max_val).exp();
                    sum_exp += *s;
                }
                for s in scores.iter_mut() { *s /= sum_exp; }
                
                let out_off = h * head_dim;
                for d in 0..head_dim {
                    out_ptr[out_off + d] = 0.0;
                }
                for ct in 0..cache_seq_len {
                    let weight = scores[ct];
                    let v_off = (ct * n_heads_kv + kv_h) * head_dim;
                    let v_vec = &v_data[v_off..v_off + head_dim];
                    
                    for d in 0..head_dim {
                        out_ptr[out_off + d] += weight * v_vec[d];
                    }
                }
            } // end h loop
        } // end borrow scope
        

        if is_opencl {
             // copy back

             let size_bytes = out_vec.len() * 4;
             let buf = Galloc::new().alloc(size_bytes, DType::F32)?;
             unsafe { std::ptr::copy_nonoverlapping(out_vec.as_ptr(), buf.as_mut_ptr() as *mut f32, out_vec.len()); }
             let cpu_out = Tensor::new(ws.out_attn.shape().clone(), buf, Arc::new(CpuBackend::new()));
             ws.out_attn = backend.copy_from(&cpu_out)?;
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
