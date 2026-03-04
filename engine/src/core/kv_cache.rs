use crate::core::memory::Memory;
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use anyhow::Result;
use std::sync::Arc;

pub struct KVCache {
    pub k_buffer: Tensor,
    pub v_buffer: Tensor,
    pub current_pos: usize,
    pub max_seq_len: usize,
    capacity: usize,
    kv_heads: usize,
    head_dim: usize,
    memory: Option<Arc<dyn Memory>>,
}

impl KVCache {
    /// Create a KVCache with full pre-allocation (capacity = max_seq_len).
    /// Growth is disabled. This preserves backward compatibility.
    pub fn new(k: Tensor, v: Tensor, max_seq_len: usize) -> Self {
        let shape = k.shape().dims();
        let kv_heads = shape[2];
        let head_dim = shape[3];
        Self {
            k_buffer: k,
            v_buffer: v,
            current_pos: 0,
            max_seq_len,
            capacity: max_seq_len,
            kv_heads,
            head_dim,
            memory: None,
        }
    }

    /// Create a KVCache with dynamic grow-on-demand allocation.
    /// Starts with `initial_capacity` and doubles up to `max_seq_len`.
    pub fn new_dynamic(
        k: Tensor,
        v: Tensor,
        initial_capacity: usize,
        max_seq_len: usize,
        kv_heads: usize,
        head_dim: usize,
        memory: Arc<dyn Memory>,
    ) -> Self {
        Self {
            k_buffer: k,
            v_buffer: v,
            current_pos: 0,
            max_seq_len,
            capacity: initial_capacity,
            kv_heads,
            head_dim,
            memory: Some(memory),
        }
    }

    /// Current physical buffer capacity in tokens.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Grow the KV cache buffers to at least `min_capacity` tokens.
    /// Uses doubling strategy clamped to `max_seq_len`.
    fn grow(&mut self, min_capacity: usize) -> Result<()> {
        let memory = self.memory.as_ref().ok_or_else(|| {
            anyhow::anyhow!("KVCache::grow() called on a non-dynamic cache (created with new())")
        })?;

        let new_cap = (self.capacity * 2).max(min_capacity).min(self.max_seq_len);
        let dtype = self.k_buffer.dtype();
        let backend = self.k_buffer.backend().clone();

        let n_values = new_cap * self.kv_heads * self.head_dim;
        let buf_size = match dtype {
            crate::core::buffer::DType::Q4_0 => {
                (n_values / crate::core::quant::QK4_0)
                    * std::mem::size_of::<crate::core::quant::BlockQ4_0>()
            }
            _ => n_values * dtype.size(),
        };

        let new_shape = Shape::new(vec![1, new_cap, self.kv_heads, self.head_dim]);

        // Allocate new buffers
        let new_k_buf = memory.alloc(buf_size, dtype)?;
        let mut new_k = Tensor::new(new_shape.clone(), new_k_buf, backend.clone());
        let new_v_buf = memory.alloc(buf_size, dtype)?;
        let mut new_v = Tensor::new(new_shape, new_v_buf, backend.clone());

        // Copy existing data
        if self.current_pos > 0 {
            let copy_count = if dtype == crate::core::buffer::DType::Q4_0 {
                let blocks_per_pos = self.kv_heads * self.head_dim / crate::core::quant::QK4_0;
                self.current_pos * blocks_per_pos
            } else {
                self.current_pos * self.kv_heads * self.head_dim
            };
            backend.copy_slice(&self.k_buffer, &mut new_k, 0, 0, copy_count)?;
            backend.copy_slice(&self.v_buffer, &mut new_v, 0, 0, copy_count)?;
        }

        log::info!(
            "[KVCache] Growing capacity: {} → {} tokens",
            self.capacity,
            new_cap
        );

        self.k_buffer = new_k;
        self.v_buffer = new_v;
        self.capacity = new_cap;

        Ok(())
    }

    pub fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        let seq_len = new_k.shape().dims()[1];

        if self.current_pos + seq_len > self.capacity {
            if self.current_pos + seq_len > self.max_seq_len {
                return Err(anyhow::anyhow!("KV Cache overflow"));
            }
            self.grow(self.current_pos + seq_len)?;
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

    pub fn get_view(&self, _seq_len: usize) -> (Tensor, Tensor) {
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

        let (src_offset, move_count) = if is_q4 {
            let bpp = heads * dim / crate::core::quant::QK4_0;
            (count * bpp, remaining * bpp)
        } else {
            let epp = heads * dim;
            (count * epp, remaining * epp)
        };

        let backend = self.k_buffer.backend().clone();
        backend.buffer_shift(&mut self.k_buffer, src_offset, 0, move_count)?;
        backend.buffer_shift(&mut self.v_buffer, src_offset, 0, move_count)?;

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
    use crate::memory::galloc::Galloc;
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

    fn make_dynamic_cache(
        initial_capacity: usize,
        max_seq_len: usize,
        heads: usize,
        dim: usize,
    ) -> KVCache {
        let memory: Arc<dyn Memory> = Arc::new(Galloc::new());
        let backend = Arc::new(CpuBackend::new());

        let size_bytes = initial_capacity * heads * dim * 4;
        let k_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));

        let k = Tensor::new(
            Shape::new(vec![1, initial_capacity, heads, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, initial_capacity, heads, dim]),
            v_buf,
            backend.clone(),
        );
        KVCache::new_dynamic(k, v, initial_capacity, max_seq_len, heads, dim, memory)
    }

    fn make_token_tensor(value: f32, heads: usize, dim: usize) -> (Tensor, Tensor) {
        let backend = Arc::new(CpuBackend::new());
        let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
        unsafe {
            let ptr = buf.as_mut_ptr() as *mut f32;
            for j in 0..(heads * dim) {
                *ptr.add(j) = value;
            }
        }
        let k = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend.clone());
        let vbuf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
        unsafe {
            let ptr = vbuf.as_mut_ptr() as *mut f32;
            for j in 0..(heads * dim) {
                *ptr.add(j) = value * 10.0;
            }
        }
        let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), vbuf, backend);
        (k, v)
    }

    #[test]
    fn test_prune_prefix_basic() {
        let mut cache = make_cache(100, 1, 4);

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

        cache.prune_prefix(3).unwrap();
        assert_eq!(cache.current_pos, 7);

        let k_data = cache.k_buffer.as_slice::<f32>();
        assert_eq!(k_data[0], 4.0);
        assert_eq!(k_data[4], 5.0);

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
        assert_eq!(cache.memory_usage_bytes(), 0);

        let mut cache = make_cache(100, 2, 64);
        cache.current_pos = 10;
        assert_eq!(cache.memory_usage_bytes(), 10 * 2 * 64 * 4 * 2);
    }

    #[test]
    fn test_cache_creation() {
        let cache = make_cache(64, 4, 8);
        assert_eq!(cache.current_pos, 0);
        assert_eq!(cache.max_seq_len, 64);
        assert_eq!(cache.capacity(), 64);
        assert_eq!(cache.k_buffer.shape().dims(), &[1, 64, 4, 8]);
        assert_eq!(cache.v_buffer.shape().dims(), &[1, 64, 4, 8]);
    }

    #[test]
    fn test_update_overflow() {
        let mut cache = make_cache(4, 1, 4);
        let backend = Arc::new(CpuBackend::new());

        for _ in 0..4 {
            let buf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
            let t = Tensor::new(Shape::new(vec![1, 1, 1, 4]), buf, backend.clone());
            let vbuf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
            let vt = Tensor::new(Shape::new(vec![1, 1, 1, 4]), vbuf, backend.clone());
            cache.update(&t, &vt).unwrap();
        }
        assert_eq!(cache.current_pos, 4);

        let buf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
        let t = Tensor::new(Shape::new(vec![1, 1, 1, 4]), buf, backend.clone());
        let vbuf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
        let vt = Tensor::new(Shape::new(vec![1, 1, 1, 4]), vbuf, backend.clone());
        assert!(cache.update(&t, &vt).is_err());
    }

    #[test]
    fn test_get_view() {
        let mut cache = make_cache(100, 2, 4);
        cache.current_pos = 10;
        let (k_view, v_view) = cache.get_view(10);
        assert_eq!(k_view.shape().dims(), &[1, 100, 2, 4]);
        assert_eq!(v_view.shape().dims(), &[1, 100, 2, 4]);
        assert_eq!(k_view.dtype(), DType::F32);
        assert_eq!(v_view.dtype(), DType::F32);
    }

    #[test]
    fn test_new_backward_compat() {
        let cache = make_cache(128, 4, 64);
        assert_eq!(cache.capacity(), 128);
        assert_eq!(cache.max_seq_len, 128);
        assert!(cache.memory.is_none());
    }

    #[test]
    fn test_dynamic_growth_basic() {
        let heads = 1;
        let dim = 4;
        let mut cache = make_dynamic_cache(4, 64, heads, dim);

        assert_eq!(cache.capacity(), 4);

        // Insert 4 tokens (fills initial capacity)
        for i in 0..4 {
            let (k, v) = make_token_tensor((i + 1) as f32, heads, dim);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 4);
        assert_eq!(cache.capacity(), 4);

        // Insert 1 more → triggers growth
        let (k, v) = make_token_tensor(5.0, heads, dim);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.current_pos, 5);
        assert!(cache.capacity() >= 5);

        // Verify data integrity
        let k_data = cache.k_buffer.as_slice::<f32>();
        assert_eq!(k_data[0], 1.0); // first token
        assert_eq!(k_data[4], 2.0); // second token (offset by dim=4)
        assert_eq!(k_data[16], 5.0); // fifth token
    }

    #[test]
    fn test_dynamic_growth_doubling() {
        let heads = 1;
        let dim = 4;
        let mut cache = make_dynamic_cache(4, 256, heads, dim);

        // Fill 4 tokens
        for i in 0..4 {
            let (k, v) = make_token_tensor(i as f32, heads, dim);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.capacity(), 4);

        // Insert 1 more → 4 * 2 = 8
        let (k, v) = make_token_tensor(100.0, heads, dim);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.capacity(), 8);

        // Fill up to 8
        for _ in 0..3 {
            let (k, v) = make_token_tensor(0.0, heads, dim);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 8);
        assert_eq!(cache.capacity(), 8);

        // Insert 1 more → 8 * 2 = 16
        let (k, v) = make_token_tensor(0.0, heads, dim);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.capacity(), 16);
    }

    #[test]
    fn test_dynamic_growth_capped() {
        let heads = 1;
        let dim = 4;
        let mut cache = make_dynamic_cache(4, 6, heads, dim);

        // Fill 4 tokens
        for i in 0..4 {
            let (k, v) = make_token_tensor(i as f32, heads, dim);
            cache.update(&k, &v).unwrap();
        }

        // Insert 1 more → min(4*2, 6) = 6
        let (k, v) = make_token_tensor(0.0, heads, dim);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.capacity(), 6);
    }

    #[test]
    fn test_dynamic_overflow() {
        let heads = 1;
        let dim = 4;
        let mut cache = make_dynamic_cache(4, 6, heads, dim);

        // Fill to max_seq_len
        for i in 0..6 {
            let (k, v) = make_token_tensor(i as f32, heads, dim);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 6);

        // One more should overflow
        let (k, v) = make_token_tensor(0.0, heads, dim);
        assert!(cache.update(&k, &v).is_err());
    }

    #[test]
    fn test_dynamic_with_eviction() {
        let heads = 1;
        let dim = 4;
        let mut cache = make_dynamic_cache(4, 32, heads, dim);

        // Fill 8 tokens (triggers growth from 4 → 8)
        for i in 0..8 {
            let (k, v) = make_token_tensor((i + 1) as f32, heads, dim);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 8);
        assert_eq!(cache.capacity(), 8);

        // Prune first 4 tokens
        cache.prune_prefix(4).unwrap();
        assert_eq!(cache.current_pos, 4);

        // Verify data shifted correctly: pos 0 should have value 5.0
        let k_data = cache.k_buffer.as_slice::<f32>();
        assert_eq!(k_data[0], 5.0);
        assert_eq!(k_data[4], 6.0);

        // Can insert more without growing (capacity=8, current_pos=4)
        for i in 0..4 {
            let (k, v) = make_token_tensor((100 + i) as f32, heads, dim);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 8);
        assert_eq!(cache.capacity(), 8); // no growth needed
    }

    #[test]
    fn test_non_dynamic_grow_fails() {
        let mut cache = make_cache(4, 1, 4);
        // Fill to capacity
        let backend = Arc::new(CpuBackend::new());
        for _ in 0..4 {
            let buf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
            let t = Tensor::new(Shape::new(vec![1, 1, 1, 4]), buf, backend.clone());
            let vbuf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
            let vt = Tensor::new(Shape::new(vec![1, 1, 1, 4]), vbuf, backend.clone());
            cache.update(&t, &vt).unwrap();
        }
        // Non-dynamic cache: capacity == max_seq_len, so overflow is immediate
        let buf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
        let t = Tensor::new(Shape::new(vec![1, 1, 1, 4]), buf, backend.clone());
        let vbuf = Arc::new(SharedBuffer::new(4 * 4, DType::F32));
        let vt = Tensor::new(Shape::new(vec![1, 1, 1, 4]), vbuf, backend.clone());
        assert!(cache.update(&t, &vt).is_err());
    }
}
