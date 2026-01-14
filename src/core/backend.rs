use anyhow::Result;
use crate::core::tensor::Tensor;

pub trait Backend: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    fn name(&self) -> &str;
    fn device(&self) -> &str;

    // Basic Math
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()>;
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()>;
    fn matmul_slice(&self, a: &Tensor, b: &Tensor, rows: usize, cols: usize, out: &mut Tensor) -> Result<()>;

    // In-place operations
    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()>;
    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()>;

    // Activation & Norm
    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()>;
    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32) -> Result<()>;
    fn softmax(&self, x: &mut Tensor) -> Result<()>;

    // Rotate
    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()>;

    // Single-query attention for generation (GQA-aware)
    // Q: [num_heads_q, head_dim], K/V cache: [cache_seq_len, num_heads_kv, head_dim]
    // Output: [num_heads_q, head_dim]
    fn attention_gen(&self, q: &Tensor, k_cache: &Tensor, v_cache: &Tensor, out: &mut Tensor, 
                     num_heads_q: usize, num_heads_kv: usize, head_dim: usize, cache_seq_len: usize) -> Result<()> {
        // Default CPU implementation  
        let q_data = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const f32, q.size()/4) };
        let k_data = unsafe { std::slice::from_raw_parts(k_cache.as_ptr() as *const f32, k_cache.size()/4) };
        let v_data = unsafe { std::slice::from_raw_parts(v_cache.as_ptr() as *const f32, v_cache.size()/4) };
        let out_data = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, out.size()/4) };
        
        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = num_heads_q / num_heads_kv;
        
        for h in 0..num_heads_q {
            let kv_h = h / gqa_ratio;
            let q_off = h * head_dim;
            let q_vec = &q_data[q_off..q_off + head_dim];
            
            // Compute scores
            let mut scores = vec![0.0f32; cache_seq_len];
            for t in 0..cache_seq_len {
                let k_off = (t * num_heads_kv + kv_h) * head_dim;
                let k_vec = &k_data[k_off..k_off + head_dim];
                let score: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                scores[t] = score * scale;
            }
            
            // Softmax
            let max_val = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0;
            for s in scores.iter_mut() { *s = (*s - max_val).exp(); sum_exp += *s; }
            for s in scores.iter_mut() { *s /= sum_exp; }
            
            // Weighted sum of V
            let out_off = h * head_dim;
            for d in 0..head_dim { out_data[out_off + d] = 0.0; }
            for t in 0..cache_seq_len {
                let weight = scores[t];
                let v_off = (t * num_heads_kv + kv_h) * head_dim;
                let v_vec = &v_data[v_off..v_off + head_dim];
                for d in 0..head_dim { out_data[out_off + d] += weight * v_vec[d]; }
            }
        }
        Ok(())
    }

    // Memory Ops
    fn copy_from(&self, t: &Tensor) -> Result<Tensor>;
    fn read_buffer(&self, t: &Tensor, dst: &mut [u8]) -> Result<()> {
        let src_ptr = unsafe { t.buffer().as_ptr() as *const u8 };
        if src_ptr.is_null() {
            anyhow::bail!("Cannot read null buffer (not mapped)");
        }
        unsafe { std::ptr::copy_nonoverlapping(src_ptr, dst.as_mut_ptr(), dst.len()); }
        Ok(())
    }

    // Synchronization (for benchmarking)
    fn synchronize(&self) -> Result<()> { Ok(()) }
    
    // Embedding Lookup / Gather
    // src: [Rows, Cols] (Embeddings)
    // indices: [NumIndices] (Indices)
    // dst: [NumIndices, Cols] (Output)
    fn gather(&self, src: &Tensor, indices: &Tensor, dst: &mut Tensor) -> Result<()> {
        // Default CPU implementation
        let src_data = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.size()/4) };
        let idx_data = unsafe { std::slice::from_raw_parts(indices.as_ptr() as *const u32, indices.size()/4) }; // U8 buffer, U32 data
        let dst_data = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.size()/4) };
        
        let cols = src.shape().dims()[1];
        
        // Validation?
        
        for (i, &idx) in idx_data.iter().enumerate() {
            let offset = idx as usize * cols;
            let target_offset = i * cols;
            if offset + cols <= src_data.len() && target_offset + cols <= dst_data.len() {
                 dst_data[target_offset..target_offset+cols].copy_from_slice(&src_data[offset..offset+cols]);
            }
        }
        Ok(())
    }

    // New API: Copy slice from src to dst
    // src_offset/dst_offset are ELEMENT offsets (not bytes) if Tensor is typed, but here for simplicity let's assume they are ELEMENT offsets relative to Tensor's DType?
    // Actually, Tensors are somewhat untyped regarding pointer arithmetic in this trait unless we know DType size.
    // Let's assume count is number of ELEMENTS. And offsets are ELEMENTS.
    // Caller must ensure types match.
    fn copy_slice(&self, src: &Tensor, dst: &mut Tensor, src_offset: usize, dst_offset: usize, count: usize) -> Result<()> {
        let type_size = match src.dtype() {
            crate::core::buffer::DType::F32 => 4,
            crate::core::buffer::DType::F16 => 2,
            crate::core::buffer::DType::U8 => 1,
            crate::core::buffer::DType::Q4_0 => std::mem::size_of::<crate::core::quant::BlockQ4_0>(),            _ => 1, // Fallback
        };
        
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();
        
        if src_ptr.is_null() || dst_ptr.is_null() {
            anyhow::bail!("Null pointer in copy_slice (default impl), likely OpenCL buffer mismatch");
        }
        
        unsafe {
            let src_u8 = src_ptr as *const u8;
            let dst_u8 = dst_ptr as *mut u8;
            
            // Calculate byte offsets
            let src_byte_offset = src_offset * type_size;
            let dst_byte_offset = dst_offset * type_size;
            let byte_count = count * type_size;
            
            std::ptr::copy_nonoverlapping(src_u8.add(src_byte_offset), dst_u8.add(dst_byte_offset), byte_count);
        }
        Ok(())
    }
}
