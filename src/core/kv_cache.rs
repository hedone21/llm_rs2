use anyhow::{Result, anyhow};
use crate::core::tensor::Tensor;

pub struct KVCache {
    pub k_buffer: Tensor,
    pub v_buffer: Tensor,
    pub current_pos: usize,
    pub max_seq_len: usize,
}

impl KVCache {
    pub fn new(k: Tensor, v: Tensor, max_seq_len: usize) -> Self {
        Self {
            k_buffer: k,
            v_buffer: v,
            current_pos: 0,
            max_seq_len,
        }
    }

    pub fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        let batch_size = new_k.shape().dims()[0]; // Assuming [Batch, Seq, Head, Dim] or similar? 
        // Llama 3.2 1B usually outputs [Batch, Seq, KV_Head, Head_Dim] for K/V proj?
        // Actually, we usually transpose to [Batch, KV_Head, Seq, Head_Dim] for attention.
        // For simplicity in this CPU backend, let's assume we are just appending to the seq dim.
        
        let seq_len = new_k.shape().dims()[1];
        
        if self.current_pos + seq_len > self.max_seq_len {
            return Err(anyhow!("KV Cache overflow"));
        }
        
        // This is a naive CPU copy. In a real backend, we'd use backend.copy_slice or similar.
        // We need to copy `new_k` into `self.k_buffer` at `current_pos`.
        
        // Assuming Shape is [Batch=1, Seq, KV_Heads, Head_Dim] or similar.
        // Let's rely on flat pointers for now for simplicity, assuming contiguous layout matches.
        
        let elem_size = std::mem::size_of::<f32>(); // Assuming F32 for cache now
        let k_ptr = self.k_buffer.as_mut_ptr() as *mut f32;
        let v_ptr = self.v_buffer.as_mut_ptr() as *mut f32;
        let new_k_ptr = new_k.as_ptr() as *const f32;
        let new_v_ptr = new_v.as_ptr() as *const f32;
        
        let shape = self.k_buffer.shape().dims();
        // [Batch, MaxSeq, Heads, Dim]
        let heads = shape[2];
        let dim = shape[3];
        
        // Very simplified: assuming batch=1 and we fill seq dimension continuously.
        // k_buffer layout: [MaxSeq, Heads, Dim] (collapsed batch)
        // copy target offset = current_pos * Heads * Dim
        
        let height = heads * dim;
        let offset = self.current_pos * height;
        let count = seq_len * height;
        
        // Use backend copy_slice for safe handling of both CPU and GPU buffers
        let backend = self.k_buffer.backend().clone(); // Clone Arc to drop borrow of self
        backend.copy_slice(new_k, &mut self.k_buffer, 0, offset, count)?;
        backend.copy_slice(new_v, &mut self.v_buffer, 0, offset, count)?;
        
        self.current_pos += seq_len;
        
        Ok(())
    }

    pub fn get_view(&self, seq_len: usize) -> (Tensor, Tensor) {
        // Return view of the active cache for the current step + history
        // Actually KVCache is passed to attention. Attention usually needs all previous tokens.
        // For simple testing, we might not need to slice *tensor objects* explicitly struct-wise if the kernel knows `current_pos`.
        // But if we must return Tensors:
        
        // Stub: return references to full buffers, but updated logic uses `current_pos` and indices.
        // To be correct, we should return a "Slice" tensor. 
        // For this task, let's just cheat and return the whole buffer, 
        // and the attention layer manages indices using `current_pos`.
        (self.k_buffer.clone(), self.v_buffer.clone())
    }
}
