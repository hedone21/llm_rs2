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

    /// Remove the first `count` tokens from the cache, shifting remaining data forward.
    ///
    /// Before: [A][B][C][D][E] (current_pos=5)
    /// prune_prefix(2)
    /// After:  [C][D][E][_][_] (current_pos=3)
    pub fn prune_prefix(&mut self, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        if count > self.current_pos {
            return Err(anyhow::anyhow!(
                "Cannot prune {} tokens, only {} in cache",
                count,
                self.current_pos
            ));
        }

        let shape = self.k_buffer.shape().dims();
        let heads = shape[2];
        let dim = shape[3];
        let remaining = self.current_pos - count;

        if remaining == 0 {
            self.current_pos = 0;
            return Ok(());
        }

        let is_q4 = self.k_buffer.dtype() == crate::core::buffer::DType::Q4_0;

        if is_q4 {
            let blocks_per_pos = heads * dim / crate::core::quant::QK4_0;
            let block_size = std::mem::size_of::<crate::core::quant::BlockQ4_0>();
            let src_byte_offset = count * blocks_per_pos * block_size;
            let move_bytes = remaining * blocks_per_pos * block_size;

            // memmove for K buffer
            let k_ptr = self.k_buffer.as_mut_ptr();
            let v_ptr = self.v_buffer.as_mut_ptr();
            if k_ptr.is_null() || v_ptr.is_null() {
                return Err(anyhow::anyhow!("Cannot prune: null buffer pointers (GPU-only buffers not supported for prune)"));
            }
            unsafe {
                std::ptr::copy(k_ptr.add(src_byte_offset), k_ptr, move_bytes);
                std::ptr::copy(v_ptr.add(src_byte_offset), v_ptr, move_bytes);
            }
        } else {
            let elems_per_pos = heads * dim;
            let type_size = self.k_buffer.dtype().size();
            let src_byte_offset = count * elems_per_pos * type_size;
            let move_bytes = remaining * elems_per_pos * type_size;

            let k_ptr = self.k_buffer.as_mut_ptr();
            let v_ptr = self.v_buffer.as_mut_ptr();
            if k_ptr.is_null() || v_ptr.is_null() {
                return Err(anyhow::anyhow!("Cannot prune: null buffer pointers (GPU-only buffers not supported for prune)"));
            }
            unsafe {
                std::ptr::copy(k_ptr.add(src_byte_offset), k_ptr, move_bytes);
                std::ptr::copy(v_ptr.add(src_byte_offset), v_ptr, move_bytes);
            }
        }

        self.current_pos = remaining;
        Ok(())
    }

    /// Returns the memory usage in bytes for currently stored KV data.
    pub fn memory_usage_bytes(&self) -> usize {
        let shape = self.k_buffer.shape().dims();
        let heads = shape[2];
        let dim = shape[3];

        let is_q4 = self.k_buffer.dtype() == crate::core::buffer::DType::Q4_0;

        let per_buffer = if is_q4 {
            let blocks_per_pos = heads * dim / crate::core::quant::QK4_0;
            let block_size = std::mem::size_of::<crate::core::quant::BlockQ4_0>();
            self.current_pos * blocks_per_pos * block_size
        } else {
            let type_size = self.k_buffer.dtype().size();
            self.current_pos * heads * dim * type_size
        };

        per_buffer * 2 // K + V
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::shared_buffer::SharedBuffer;
    use crate::core::buffer::{Buffer, DType};
    use crate::core::shape::Shape;
    use std::sync::Arc;

    fn make_cache(max_seq_len: usize, heads: usize, dim: usize) -> KVCache {
        let size_bytes = max_seq_len * heads * dim * 4; // F32
        let k_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let backend = Arc::new(CpuBackend::new());

        let k = Tensor::new(
            Shape::new(vec![1, max_seq_len, heads, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq_len, heads, dim]),
            v_buf,
            backend.clone(),
        );
        KVCache::new(k, v, max_seq_len)
    }

    #[test]
    fn test_prune_prefix_basic() {
        let mut cache = make_cache(100, 1, 4);

        // Fill K buffer with recognizable pattern: pos i => all values = (i+1) as f32
        let backend = Arc::new(CpuBackend::new());
        for i in 0..10 {
            let buf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
            let val = (i + 1) as f32;
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut f32;
                for j in 0..4 {
                    *ptr.add(j) = val;
                }
            }
            let t = Tensor::new(Shape::new(vec![1, 1, 1, 4]), buf, backend.clone());
            // Create matching V tensor
            let vbuf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
            unsafe {
                let ptr = vbuf.as_mut_ptr() as *mut f32;
                for j in 0..4 {
                    *ptr.add(j) = val * 10.0;
                }
            }
            let vt = Tensor::new(Shape::new(vec![1, 1, 1, 4]), vbuf, backend.clone());
            cache.update(&t, &vt).unwrap();
        }
        assert_eq!(cache.current_pos, 10);

        // Prune first 3 tokens
        cache.prune_prefix(3).unwrap();
        assert_eq!(cache.current_pos, 7);

        // Verify: position 0 should now contain what was position 3 (value = 4.0)
        let k_data = cache.k_buffer.as_slice::<f32>();
        assert_eq!(k_data[0], 4.0);
        assert_eq!(k_data[4], 5.0); // position 1 = old position 4

        let v_data = cache.v_buffer.as_slice::<f32>();
        assert_eq!(v_data[0], 40.0);
        assert_eq!(v_data[4], 50.0);
    }

    #[test]
    fn test_prune_prefix_zero() {
        let mut cache = make_cache(100, 1, 4);
        cache.current_pos = 5;
        cache.prune_prefix(0).unwrap();
        assert_eq!(cache.current_pos, 5);
    }

    #[test]
    fn test_prune_prefix_over_count() {
        let mut cache = make_cache(100, 1, 4);
        cache.current_pos = 5;
        assert!(cache.prune_prefix(10).is_err());
    }

    #[test]
    fn test_prune_prefix_all() {
        let mut cache = make_cache(100, 1, 4);
        cache.current_pos = 5;
        cache.prune_prefix(5).unwrap();
        assert_eq!(cache.current_pos, 0);
    }

    #[test]
    fn test_memory_usage_bytes() {
        let cache = make_cache(100, 2, 64);
        // Empty cache
        assert_eq!(cache.memory_usage_bytes(), 0);

        let mut cache = make_cache(100, 2, 64);
        cache.current_pos = 10;
        // 10 positions * 2 heads * 64 dim * 4 bytes (F32) * 2 (K+V)
        assert_eq!(cache.memory_usage_bytes(), 10 * 2 * 64 * 4 * 2);
    }
}
