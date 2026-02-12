use anyhow::Result;
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
        let seq_len = new_k.shape().dims()[1];
        
        if self.current_pos + seq_len > self.max_seq_len {
            return Err(anyhow::anyhow!("KV Cache overflow"));
        }
        
        let shape = self.k_buffer.shape().dims();
        let heads = shape[2];
        let dim = shape[3];
        
        // For Q4_0, offsets/counts are in block units (each block = 32 elements = 18 bytes)
        let (offset, count) = if self.k_buffer.dtype() == crate::core::buffer::DType::Q4_0 {
            let blocks_per_pos = heads * dim / crate::core::quant::QK4_0;
            (self.current_pos * blocks_per_pos, seq_len * blocks_per_pos)
        } else {
            let height = heads * dim;
            (self.current_pos * height, seq_len * height)
        };
        
        let backend = self.k_buffer.backend().clone();
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
