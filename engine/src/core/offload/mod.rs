//! KV cache layer-wise offload system.
//!
//! Two lossless strategies share a common `OffloadStore` interface:
//! - `DiskStore`: raw KV data → temporary files
//! - `ZramStore`: byte-shuffle → LZ4 compression → in-memory storage
//!
//! `OffloadKVCache` implements `KVCacheOps` and manages the lifecycle:
//! migration (KVCache → offload), per-token append during decode,
//! and on-demand data loading for attention computation.

pub mod disk_store;
pub mod preprocess;
pub mod store;
pub mod zram_store;

use crate::backend::cpu::CpuBackend;
use crate::buffer::shared_buffer::SharedBuffer;
use crate::core::buffer::{Buffer, DType};
use crate::core::kv_cache::{KVCacheOps, KVLayout};
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use anyhow::Result;
use std::sync::Arc;

use self::store::OffloadStore;

/// KV cache implementation that offloads data to an external store (disk or compressed memory).
///
/// Implements `KVCacheOps` so it can be used as a drop-in replacement in the
/// generic `forward_into<C: KVCacheOps>` path.
///
/// Phase 3 design: lazy attn buffers + per-layer prefetch support.
/// Only 2 layers hold attn buffers simultaneously (current + next), saving ~75% memory
/// vs pre-allocating all layers.
///
/// SeqMajor layout only. No eviction support.
pub struct OffloadKVCache {
    /// Layer index (for debug/logging).
    layer_id: usize,
    /// Number of KV heads.
    kv_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Data type (F16 or F32).
    dtype: DType,
    /// Current number of valid tokens in the store.
    current_pos: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Bytes per token for K (or V): kv_heads × head_dim × dtype.size().
    token_bytes: usize,
    /// The backing store (DiskStore or ZramStore).
    store: Box<dyn OffloadStore>,
    /// Lazy-allocated attention buffer for K. Allocated on preload(), freed on release_buffers().
    attn_k_buf: Option<Vec<u8>>,
    /// Lazy-allocated attention buffer for V. Allocated on preload(), freed on release_buffers().
    attn_v_buf: Option<Vec<u8>>,
    /// Whether preload() has been called and data is ready in attn buffers.
    preloaded: bool,
    /// Reusable output SharedBuffer for K (R-P2: avoid per-call allocation).
    out_k_buf: Option<Arc<SharedBuffer>>,
    /// Reusable output SharedBuffer for V (R-P2: avoid per-call allocation).
    out_v_buf: Option<Arc<SharedBuffer>>,
    /// Shared CPU backend for creating output tensors.
    out_backend: Arc<CpuBackend>,
}

impl OffloadKVCache {
    /// Create a new OffloadKVCache wrapping the given store.
    /// Attn buffers are lazy-allocated (None until preload() or get_view()).
    pub fn new(
        layer_id: usize,
        kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        max_seq_len: usize,
        store: Box<dyn OffloadStore>,
    ) -> Self {
        let token_bytes = kv_heads * head_dim * dtype.size();

        Self {
            layer_id,
            kv_heads,
            head_dim,
            dtype,
            current_pos: 0,
            max_seq_len,
            token_bytes,
            store,
            attn_k_buf: None,
            attn_v_buf: None,
            preloaded: false,
            out_k_buf: None,
            out_v_buf: None,
            out_backend: Arc::new(CpuBackend::new()),
        }
    }

    /// Migrate data from an existing KVCache (raw buffer copy).
    /// The source KVCache's data is read and stored in the offload store.
    pub fn migrate_from_raw(
        &mut self,
        k_data: &[u8],
        v_data: &[u8],
        num_tokens: usize,
    ) -> Result<()> {
        self.store.store(k_data, v_data, num_tokens)?;
        self.current_pos = num_tokens;
        Ok(())
    }

    /// Get a reference to the underlying store (for compression ratio queries etc).
    pub fn store(&self) -> &dyn OffloadStore {
        self.store.as_ref()
    }

    /// Pre-load KV data from store into attn buffers.
    /// Called by the prefetch pipeline to overlap I/O with compute.
    /// If attn buffers are not allocated, they are lazily created.
    pub fn preload(&mut self) -> Result<()> {
        let total_bytes = self.current_pos * self.token_bytes;
        if total_bytes == 0 {
            self.preloaded = true;
            return Ok(());
        }

        let max_bytes = self.max_seq_len * self.token_bytes;
        let k_buf = self.attn_k_buf.get_or_insert_with(|| vec![0u8; max_bytes]);
        let v_buf = self.attn_v_buf.get_or_insert_with(|| vec![0u8; max_bytes]);

        self.store
            .load_into(&mut k_buf[..total_bytes], &mut v_buf[..total_bytes])?;
        self.preloaded = true;
        Ok(())
    }

    /// Release attn buffers to free memory (R-P1: only 2 layers active at once).
    pub fn release_buffers(&mut self) {
        self.attn_k_buf = None;
        self.attn_v_buf = None;
        self.preloaded = false;
    }

    /// Reset preloaded flag (R-P5: call at token boundary).
    pub fn reset_preload(&mut self) {
        self.preloaded = false;
    }

    /// Allocate or reuse a SharedBuffer of the given size.
    fn reuse_or_alloc_out_buf(
        slot: &mut Option<Arc<SharedBuffer>>,
        needed_bytes: usize,
        dtype: DType,
    ) -> Arc<SharedBuffer> {
        if let Some(buf) = slot.as_ref()
            && buf.size() >= needed_bytes
        {
            return buf.clone();
        }
        let buf = Arc::new(SharedBuffer::new(needed_bytes, dtype));
        *slot = Some(buf.clone());
        buf
    }
}

impl KVCacheOps for OffloadKVCache {
    fn current_pos(&self) -> usize {
        self.current_pos
    }

    fn capacity(&self) -> usize {
        self.max_seq_len
    }

    fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn layout(&self) -> KVLayout {
        KVLayout::SeqMajor
    }

    fn kv_dtype(&self) -> DType {
        self.dtype
    }

    fn memory_usage_bytes(&self) -> usize {
        self.store.storage_size()
    }

    fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        let seq_len = new_k.shape().dims()[1];
        let total_after = self.current_pos + seq_len;
        if total_after > self.max_seq_len {
            return Err(anyhow::anyhow!(
                "OffloadKVCache[{}] overflow: {} + {} > {}",
                self.layer_id,
                self.current_pos,
                seq_len,
                self.max_seq_len
            ));
        }

        // Read raw data from tensors
        let k_ptr = new_k.buffer().as_ptr();
        let v_ptr = new_v.buffer().as_ptr();
        let expected_bytes = seq_len * self.token_bytes;

        let k_data = unsafe { std::slice::from_raw_parts(k_ptr, expected_bytes) };
        let v_data = unsafe { std::slice::from_raw_parts(v_ptr, expected_bytes) };

        if seq_len == 1 {
            // Decode path: single token append
            self.store.append_token(k_data, v_data)?;

            // If preloaded, also append to attn buffers (avoid re-load in get_view)
            if self.preloaded
                && let (Some(k_buf), Some(v_buf)) = (&mut self.attn_k_buf, &mut self.attn_v_buf)
            {
                let offset = self.current_pos * self.token_bytes;
                k_buf[offset..offset + self.token_bytes].copy_from_slice(k_data);
                v_buf[offset..offset + self.token_bytes].copy_from_slice(v_data);
            }
        } else {
            // Prefill or multi-token: batch store
            if self.current_pos == 0 {
                self.store.store(k_data, v_data, seq_len)?;
            } else {
                // Append token by token (slower but correct)
                for s in 0..seq_len {
                    let offset = s * self.token_bytes;
                    self.store.append_token(
                        &k_data[offset..offset + self.token_bytes],
                        &v_data[offset..offset + self.token_bytes],
                    )?;
                }
            }
        }

        self.current_pos = total_after;
        Ok(())
    }

    fn get_view(&mut self) -> (Tensor, Tensor) {
        let total = self.current_pos;
        let backend: Arc<dyn crate::core::backend::Backend> = self.out_backend.clone();

        if total == 0 {
            let buf = Arc::new(SharedBuffer::new(0, self.dtype));
            let shape = Shape::new(vec![1, 0, self.kv_heads, self.head_dim]);
            let t = Tensor::new(shape.clone(), buf.clone(), backend.clone());
            return (t.clone(), t);
        }

        let total_bytes = total * self.token_bytes;
        let max_bytes = self.max_seq_len * self.token_bytes;

        // Lazy-allocate attn buffers if needed
        let k_attn = self.attn_k_buf.get_or_insert_with(|| vec![0u8; max_bytes]);
        let v_attn = self.attn_v_buf.get_or_insert_with(|| vec![0u8; max_bytes]);

        if !self.preloaded {
            // Synchronous fallback: load from store
            if let Err(e) = self
                .store
                .load_into(&mut k_attn[..total_bytes], &mut v_attn[..total_bytes])
            {
                log::error!("OffloadKVCache[{}] load failed: {}", self.layer_id, e);
                let buf = Arc::new(SharedBuffer::new(0, self.dtype));
                let shape = Shape::new(vec![1, 0, self.kv_heads, self.head_dim]);
                let t = Tensor::new(shape.clone(), buf.clone(), backend.clone());
                return (t.clone(), t);
            }
        }
        // Reset preloaded after consumption
        self.preloaded = false;

        let shape = Shape::new(vec![1, total, self.kv_heads, self.head_dim]);

        // R-P2: Reuse output SharedBuffers when possible
        let k_buf = Self::reuse_or_alloc_out_buf(&mut self.out_k_buf, total_bytes, self.dtype);
        let v_buf = Self::reuse_or_alloc_out_buf(&mut self.out_v_buf, total_bytes, self.dtype);
        unsafe {
            std::ptr::copy_nonoverlapping(k_attn.as_ptr(), k_buf.as_mut_ptr(), total_bytes);
            std::ptr::copy_nonoverlapping(v_attn.as_ptr(), v_buf.as_mut_ptr(), total_bytes);
        }

        let k_tensor = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v_tensor = Tensor::new(shape, v_buf, backend);

        (k_tensor, v_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::kv_cache::KVCacheOps;

    fn make_test_tensor(
        seq_len: usize,
        kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        fill_val: u8,
    ) -> Tensor {
        let elem_bytes = dtype.size();
        let n_bytes = seq_len * kv_heads * head_dim * elem_bytes;
        let buf = Arc::new(SharedBuffer::new(n_bytes, dtype));
        unsafe {
            std::ptr::write_bytes(buf.as_mut_ptr(), fill_val, n_bytes);
        }
        let backend: Arc<dyn crate::core::backend::Backend> = Arc::new(CpuBackend::new());
        Tensor::new(
            Shape::new(vec![1, seq_len, kv_heads, head_dim]),
            buf,
            backend,
        )
    }

    fn make_f32_tensor_with_data(
        data: &[f32],
        seq_len: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Tensor {
        let n_bytes = data.len() * 4;
        let buf = Arc::new(SharedBuffer::new(n_bytes, DType::F32));
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.as_mut_ptr() as *mut f32, data.len());
        }
        let backend: Arc<dyn crate::core::backend::Backend> = Arc::new(CpuBackend::new());
        Tensor::new(
            Shape::new(vec![1, seq_len, kv_heads, head_dim]),
            buf,
            backend,
        )
    }

    #[test]
    fn test_offload_kvcache_ops_disk() {
        let dir = std::env::temp_dir().join("llm_rs2_test_offload_ops");
        let token_bytes = 2 * 32 * 2; // 2 heads × 32 dim × F16
        let store = disk_store::DiskStore::new(&dir, 0, token_bytes).unwrap();

        let mut cache = OffloadKVCache::new(0, 2, 32, DType::F16, 256, Box::new(store));

        assert_eq!(cache.current_pos(), 0);
        assert_eq!(cache.capacity(), 256);
        assert_eq!(cache.kv_dtype(), DType::F16);
        assert_eq!(cache.layout(), KVLayout::SeqMajor);

        // Update with 4 tokens
        let k = make_test_tensor(4, 2, 32, DType::F16, 0xAB);
        let v = make_test_tensor(4, 2, 32, DType::F16, 0xCD);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.current_pos(), 4);

        // Get view
        let (k_view, v_view) = cache.get_view();
        assert_eq!(k_view.shape().dims(), &[1, 4, 2, 32]);
        assert_eq!(v_view.shape().dims(), &[1, 4, 2, 32]);

        // Verify data
        let k_out =
            unsafe { std::slice::from_raw_parts(k_view.buffer().as_ptr(), 4 * token_bytes) };
        assert!(k_out.iter().all(|&b| b == 0xAB), "K data corrupted");

        let v_out =
            unsafe { std::slice::from_raw_parts(v_view.buffer().as_ptr(), 4 * token_bytes) };
        assert!(v_out.iter().all(|&b| b == 0xCD), "V data corrupted");
    }

    #[test]
    fn test_offload_kvcache_ops_zram() {
        let token_bytes = 2 * 32 * 2;
        let store = zram_store::ZramStore::new(token_bytes, 2, 64);

        let mut cache = OffloadKVCache::new(0, 2, 32, DType::F16, 256, Box::new(store));

        let k = make_test_tensor(4, 2, 32, DType::F16, 0xAB);
        let v = make_test_tensor(4, 2, 32, DType::F16, 0xCD);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.current_pos(), 4);

        let (k_view, v_view) = cache.get_view();
        assert_eq!(k_view.shape().dims(), &[1, 4, 2, 32]);

        let k_out =
            unsafe { std::slice::from_raw_parts(k_view.buffer().as_ptr(), 4 * token_bytes) };
        assert!(k_out.iter().all(|&b| b == 0xAB));

        let v_out =
            unsafe { std::slice::from_raw_parts(v_view.buffer().as_ptr(), 4 * token_bytes) };
        assert!(v_out.iter().all(|&b| b == 0xCD));
    }

    #[test]
    fn test_offload_kvcache_decode_loop() {
        let dir = std::env::temp_dir().join("llm_rs2_test_decode_loop");
        let kv_heads = 2;
        let head_dim = 32;
        let token_bytes = kv_heads * head_dim * 2; // F16
        let store = disk_store::DiskStore::new(&dir, 0, token_bytes).unwrap();

        let mut cache =
            OffloadKVCache::new(0, kv_heads, head_dim, DType::F16, 512, Box::new(store));

        // Simulate prefill (4 tokens) then 100 decode steps
        let k_prefill = make_test_tensor(4, kv_heads, head_dim, DType::F16, 0x11);
        let v_prefill = make_test_tensor(4, kv_heads, head_dim, DType::F16, 0x22);
        cache.update(&k_prefill, &v_prefill).unwrap();

        for i in 0..100u8 {
            let k = make_test_tensor(1, kv_heads, head_dim, DType::F16, i);
            let v = make_test_tensor(1, kv_heads, head_dim, DType::F16, i + 128);
            cache.update(&k, &v).unwrap();
        }

        assert_eq!(cache.current_pos(), 104);

        let (k_view, v_view) = cache.get_view();
        assert_eq!(k_view.shape().dims(), &[1, 104, kv_heads, head_dim]);
        assert_eq!(v_view.shape().dims(), &[1, 104, kv_heads, head_dim]);

        // Verify first 4 tokens still have prefill data
        let k_out =
            unsafe { std::slice::from_raw_parts(k_view.buffer().as_ptr(), 104 * token_bytes) };
        assert!(
            k_out[..token_bytes].iter().all(|&b| b == 0x11),
            "prefill K data corrupted"
        );
    }

    #[test]
    fn test_offload_kvcache_f32_bit_exact() {
        // Verify F32 data survives the full round-trip (DiskStore)
        let dir = std::env::temp_dir().join("llm_rs2_test_f32_exact");
        let kv_heads = 2;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4; // F32
        let store = disk_store::DiskStore::new(&dir, 0, token_bytes).unwrap();

        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        let k_vals: Vec<f32> = (0..kv_heads * head_dim)
            .map(|i| 1.0 + i as f32 * 0.01)
            .collect();
        let v_vals: Vec<f32> = (0..kv_heads * head_dim)
            .map(|i| 2.0 + i as f32 * 0.01)
            .collect();

        let k = make_f32_tensor_with_data(&k_vals, 1, kv_heads, head_dim);
        let v = make_f32_tensor_with_data(&v_vals, 1, kv_heads, head_dim);
        cache.update(&k, &v).unwrap();

        let (k_view, v_view) = cache.get_view();
        let k_out = k_view.as_slice::<f32>();
        let v_out = v_view.as_slice::<f32>();

        for i in 0..k_vals.len() {
            assert_eq!(
                k_out[i], k_vals[i],
                "K[{i}] mismatch: {} vs {}",
                k_out[i], k_vals[i]
            );
            assert_eq!(
                v_out[i], v_vals[i],
                "V[{i}] mismatch: {} vs {}",
                v_out[i], v_vals[i]
            );
        }
    }

    #[test]
    fn test_offload_kvcache_f32_zram_bit_exact() {
        // Verify F32 data survives ZramStore round-trip (lossless!)
        let kv_heads = 2;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4;
        let store = zram_store::ZramStore::new(token_bytes, 4, 32);

        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        let k_vals: Vec<f32> = (0..kv_heads * head_dim)
            .map(|i| std::f32::consts::PI + i as f32 * 0.001)
            .collect();
        let v_vals: Vec<f32> = (0..kv_heads * head_dim)
            .map(|i| std::f32::consts::E + i as f32 * 0.001)
            .collect();

        let k = make_f32_tensor_with_data(&k_vals, 1, kv_heads, head_dim);
        let v = make_f32_tensor_with_data(&v_vals, 1, kv_heads, head_dim);
        cache.update(&k, &v).unwrap();

        let (k_view, v_view) = cache.get_view();
        let k_out = k_view.as_slice::<f32>();
        let v_out = v_view.as_slice::<f32>();

        for i in 0..k_vals.len() {
            assert_eq!(k_out[i], k_vals[i], "K[{i}] not bit-exact");
            assert_eq!(v_out[i], v_vals[i], "V[{i}] not bit-exact");
        }
    }

    #[test]
    fn test_offload_kvcache_overflow() {
        let kv_heads = 1;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 2;
        let store = zram_store::ZramStore::new(token_bytes, 2, 8);

        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F16, 8, Box::new(store));

        // Fill to max
        for _ in 0..8 {
            let k = make_test_tensor(1, kv_heads, head_dim, DType::F16, 0);
            let v = make_test_tensor(1, kv_heads, head_dim, DType::F16, 0);
            cache.update(&k, &v).unwrap();
        }

        // One more should fail
        let k = make_test_tensor(1, kv_heads, head_dim, DType::F16, 0);
        let v = make_test_tensor(1, kv_heads, head_dim, DType::F16, 0);
        assert!(cache.update(&k, &v).is_err());
    }

    #[test]
    fn test_offload_kvcache_empty_view() {
        let store = zram_store::ZramStore::new(64, 2, 8);
        let mut cache = OffloadKVCache::new(0, 2, 4, DType::F16, 64, Box::new(store));

        let (k, v) = cache.get_view();
        assert_eq!(k.shape().dims(), &[1, 0, 2, 4]);
        assert_eq!(v.shape().dims(), &[1, 0, 2, 4]);
    }

    #[test]
    fn test_offload_kvcache_memory_usage() {
        let token_bytes = 8 * 64 * 2;
        let store = zram_store::ZramStore::new(token_bytes, 2, 64);
        let cache = OffloadKVCache::new(0, 8, 64, DType::F16, 256, Box::new(store));
        assert_eq!(cache.memory_usage_bytes(), 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Integration Tests: BASE (KVCache) vs OffloadKVCache comparison
    // ════════════════════════════════════════════════════════════════════════

    use crate::core::kv_cache::KVCache;
    use crate::core::memory::Memory;
    use crate::memory::galloc::Galloc;

    /// Create a standard KVCache (BASE) for comparison.
    fn make_base_kvcache(
        kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
    ) -> KVCache {
        let memory = Galloc::new();
        let elem_size = dtype.size();
        let buf_bytes = max_seq_len * kv_heads * head_dim * elem_size;
        let k_buf = memory.alloc(buf_bytes, dtype).unwrap();
        let v_buf = memory.alloc(buf_bytes, dtype).unwrap();
        let shape = Shape::new(vec![1, max_seq_len, kv_heads, head_dim]);
        let backend: Arc<dyn crate::core::backend::Backend> = Arc::new(CpuBackend::new());
        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend);
        KVCache::new(k, v, max_seq_len)
    }

    /// Generate realistic F32 KV tensor data for a given seq_len.
    fn make_realistic_f32_tensor(
        seq_len: usize,
        kv_heads: usize,
        head_dim: usize,
        seed: f32,
    ) -> Tensor {
        let n = seq_len * kv_heads * head_dim;
        let mut data = vec![0.0f32; n];
        for i in 0..n {
            // Simulate attention-like values with reasonable range
            data[i] = seed + (i as f32 * 0.001).sin() * 0.5;
        }
        make_f32_tensor_with_data(&data, seq_len, kv_heads, head_dim)
    }

    /// Generate realistic F16 KV tensor data.
    fn make_realistic_f16_tensor(
        seq_len: usize,
        kv_heads: usize,
        head_dim: usize,
        seed: u16,
    ) -> Tensor {
        let n = seq_len * kv_heads * head_dim;
        let n_bytes = n * 2;
        let buf = Arc::new(SharedBuffer::new(n_bytes, DType::F16));
        unsafe {
            let ptr = buf.as_mut_ptr() as *mut u16;
            for i in 0..n {
                // F16 values near 1.0 (0x3C00) with variation
                let val = seed.wrapping_add((i as u16) % 512);
                *ptr.add(i) = val;
            }
        }
        let backend: Arc<dyn crate::core::backend::Backend> = Arc::new(CpuBackend::new());
        Tensor::new(
            Shape::new(vec![1, seq_len, kv_heads, head_dim]),
            buf,
            backend,
        )
    }

    /// Compare get_view outputs between two KVCacheOps implementations.
    /// Returns (matches, total_elements).
    fn compare_views<A: KVCacheOps, B: KVCacheOps>(
        a: &mut A,
        b: &mut B,
        dtype: DType,
    ) -> (bool, usize) {
        let (k_a, v_a) = a.get_view();
        let (k_b, v_b) = b.get_view();

        let total_bytes = a.current_pos() * a.kv_heads() * a.head_dim() * dtype.size();
        if total_bytes == 0 {
            return (true, 0);
        }

        let k_a_data = unsafe { std::slice::from_raw_parts(k_a.buffer().as_ptr(), total_bytes) };
        let k_b_data = unsafe { std::slice::from_raw_parts(k_b.buffer().as_ptr(), total_bytes) };
        let v_a_data = unsafe { std::slice::from_raw_parts(v_a.buffer().as_ptr(), total_bytes) };
        let v_b_data = unsafe { std::slice::from_raw_parts(v_b.buffer().as_ptr(), total_bytes) };

        let k_match = k_a_data == k_b_data;
        let v_match = v_a_data == v_b_data;

        let elems = total_bytes / dtype.size();
        (k_match && v_match, elems)
    }

    #[test]
    fn test_integration_base_vs_disk_f32_accuracy() {
        let kv_heads = 8;
        let head_dim = 64;
        let max_seq_len = 256;
        let prefill_len = 16;
        let decode_steps = 64;

        let dir = std::env::temp_dir().join("llm_rs2_integ_disk_f32");
        let token_bytes = kv_heads * head_dim * 4;
        let disk = disk_store::DiskStore::new(&dir, 0, token_bytes).unwrap();

        let mut base = make_base_kvcache(kv_heads, head_dim, max_seq_len, DType::F32);
        let mut offload = OffloadKVCache::new(
            0,
            kv_heads,
            head_dim,
            DType::F32,
            max_seq_len,
            Box::new(disk),
        );

        // Prefill
        let k_pf = make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 1.0);
        let v_pf = make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 2.0);
        base.update(&k_pf, &v_pf).unwrap();
        offload.update(&k_pf, &v_pf).unwrap();

        // Decode
        for i in 0..decode_steps {
            let k = make_realistic_f32_tensor(1, kv_heads, head_dim, 0.1 * i as f32);
            let v = make_realistic_f32_tensor(1, kv_heads, head_dim, 0.2 * i as f32);
            base.update(&k, &v).unwrap();
            offload.update(&k, &v).unwrap();
        }

        assert_eq!(base.current_pos(), offload.current_pos());
        let (exact, elems) = compare_views(&mut base, &mut offload, DType::F32);
        assert!(
            exact,
            "DiskStore F32: BASE vs Offload NOT bit-exact! ({elems} elements)"
        );
        println!(
            "[PASS] DiskStore F32: bit-exact match, {} tokens, {} elements",
            base.current_pos(),
            elems
        );
    }

    #[test]
    fn test_integration_base_vs_disk_f16_accuracy() {
        let kv_heads = 8;
        let head_dim = 64;
        let max_seq_len = 256;
        let prefill_len = 16;
        let decode_steps = 64;

        let dir = std::env::temp_dir().join("llm_rs2_integ_disk_f16");
        let token_bytes = kv_heads * head_dim * 2;
        let disk = disk_store::DiskStore::new(&dir, 0, token_bytes).unwrap();

        let mut base = make_base_kvcache(kv_heads, head_dim, max_seq_len, DType::F16);
        let mut offload = OffloadKVCache::new(
            0,
            kv_heads,
            head_dim,
            DType::F16,
            max_seq_len,
            Box::new(disk),
        );

        // Prefill
        let k_pf = make_realistic_f16_tensor(prefill_len, kv_heads, head_dim, 0x3C00);
        let v_pf = make_realistic_f16_tensor(prefill_len, kv_heads, head_dim, 0x3E00);
        base.update(&k_pf, &v_pf).unwrap();
        offload.update(&k_pf, &v_pf).unwrap();

        // Decode
        for i in 0..decode_steps {
            let k = make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3C00 + (i as u16 * 3));
            let v = make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3E00 + (i as u16 * 5));
            base.update(&k, &v).unwrap();
            offload.update(&k, &v).unwrap();
        }

        let (exact, elems) = compare_views(&mut base, &mut offload, DType::F16);
        assert!(
            exact,
            "DiskStore F16: BASE vs Offload NOT bit-exact! ({elems} elements)"
        );
        println!(
            "[PASS] DiskStore F16: bit-exact match, {} tokens, {} elements",
            base.current_pos(),
            elems
        );
    }

    #[test]
    fn test_integration_base_vs_zram_f32_accuracy() {
        let kv_heads = 8;
        let head_dim = 64;
        let max_seq_len = 256;
        let prefill_len = 16;
        let decode_steps = 64;

        let token_bytes = kv_heads * head_dim * 4;
        let zram = zram_store::ZramStore::new(token_bytes, 4, 64);

        let mut base = make_base_kvcache(kv_heads, head_dim, max_seq_len, DType::F32);
        let mut offload = OffloadKVCache::new(
            0,
            kv_heads,
            head_dim,
            DType::F32,
            max_seq_len,
            Box::new(zram),
        );

        // Prefill
        let k_pf = make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 1.0);
        let v_pf = make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 2.0);
        base.update(&k_pf, &v_pf).unwrap();
        offload.update(&k_pf, &v_pf).unwrap();

        // Decode
        for i in 0..decode_steps {
            let k = make_realistic_f32_tensor(1, kv_heads, head_dim, 0.1 * i as f32);
            let v = make_realistic_f32_tensor(1, kv_heads, head_dim, 0.2 * i as f32);
            base.update(&k, &v).unwrap();
            offload.update(&k, &v).unwrap();
        }

        let (exact, elems) = compare_views(&mut base, &mut offload, DType::F32);
        assert!(
            exact,
            "ZramStore F32: BASE vs Offload NOT bit-exact! ({elems} elements)"
        );
        println!(
            "[PASS] ZramStore F32: bit-exact match, {} tokens, {} elements",
            base.current_pos(),
            elems
        );
    }

    #[test]
    fn test_integration_base_vs_zram_f16_accuracy() {
        let kv_heads = 8;
        let head_dim = 64;
        let max_seq_len = 256;
        let prefill_len = 16;
        let decode_steps = 64;

        let token_bytes = kv_heads * head_dim * 2;
        let zram = zram_store::ZramStore::new(token_bytes, 2, 64);

        let mut base = make_base_kvcache(kv_heads, head_dim, max_seq_len, DType::F16);
        let mut offload = OffloadKVCache::new(
            0,
            kv_heads,
            head_dim,
            DType::F16,
            max_seq_len,
            Box::new(zram),
        );

        let k_pf = make_realistic_f16_tensor(prefill_len, kv_heads, head_dim, 0x3C00);
        let v_pf = make_realistic_f16_tensor(prefill_len, kv_heads, head_dim, 0x3E00);
        base.update(&k_pf, &v_pf).unwrap();
        offload.update(&k_pf, &v_pf).unwrap();

        for i in 0..decode_steps {
            let k = make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3C00 + (i as u16 * 3));
            let v = make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3E00 + (i as u16 * 5));
            base.update(&k, &v).unwrap();
            offload.update(&k, &v).unwrap();
        }

        let (exact, elems) = compare_views(&mut base, &mut offload, DType::F16);
        assert!(
            exact,
            "ZramStore F16: BASE vs Offload NOT bit-exact! ({elems} elements)"
        );
        println!(
            "[PASS] ZramStore F16: bit-exact match, {} tokens, {} elements",
            base.current_pos(),
            elems
        );
    }

    #[test]
    fn test_integration_speed_and_compression() {
        // Benchmark: BASE vs DiskStore vs ZramStore
        // Measures update + get_view timing and ZramStore compression ratio
        let kv_heads = 8;
        let head_dim = 64;
        let max_seq_len = 512;
        let prefill_len = 32;
        let decode_steps = 128;

        let dir = std::env::temp_dir().join("llm_rs2_integ_bench");

        // --- BASE (KVCache) ---
        let base_start = std::time::Instant::now();
        let mut base = make_base_kvcache(kv_heads, head_dim, max_seq_len, DType::F32);
        {
            let k = make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 1.0);
            let v = make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 2.0);
            base.update(&k, &v).unwrap();
        }
        for i in 0..decode_steps {
            let k = make_realistic_f32_tensor(1, kv_heads, head_dim, 0.1 * i as f32);
            let v = make_realistic_f32_tensor(1, kv_heads, head_dim, 0.2 * i as f32);
            base.update(&k, &v).unwrap();
            let _ = KVCacheOps::get_view(&mut base);
        }
        let base_ms = base_start.elapsed().as_secs_f64() * 1000.0;

        // --- DiskStore ---
        let token_bytes = kv_heads * head_dim * 4;
        let disk_start = std::time::Instant::now();
        let disk = disk_store::DiskStore::new(&dir, 0, token_bytes).unwrap();
        let mut disk_cache = OffloadKVCache::new(
            0,
            kv_heads,
            head_dim,
            DType::F32,
            max_seq_len,
            Box::new(disk),
        );
        {
            let k = make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 1.0);
            let v = make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 2.0);
            disk_cache.update(&k, &v).unwrap();
        }
        for i in 0..decode_steps {
            let k = make_realistic_f32_tensor(1, kv_heads, head_dim, 0.1 * i as f32);
            let v = make_realistic_f32_tensor(1, kv_heads, head_dim, 0.2 * i as f32);
            disk_cache.update(&k, &v).unwrap();
            let _ = disk_cache.get_view();
        }
        let disk_ms = disk_start.elapsed().as_secs_f64() * 1000.0;
        let disk_size = disk_cache.memory_usage_bytes();

        // --- ZramStore F32 ---
        let zram_f32_start = std::time::Instant::now();
        let zram = zram_store::ZramStore::new(token_bytes, 4, 64);
        let mut zram_f32_cache = OffloadKVCache::new(
            0,
            kv_heads,
            head_dim,
            DType::F32,
            max_seq_len,
            Box::new(zram),
        );
        {
            let k = make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 1.0);
            let v = make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 2.0);
            zram_f32_cache.update(&k, &v).unwrap();
        }
        for i in 0..decode_steps {
            let k = make_realistic_f32_tensor(1, kv_heads, head_dim, 0.1 * i as f32);
            let v = make_realistic_f32_tensor(1, kv_heads, head_dim, 0.2 * i as f32);
            zram_f32_cache.update(&k, &v).unwrap();
            let _ = zram_f32_cache.get_view();
        }
        let zram_f32_ms = zram_f32_start.elapsed().as_secs_f64() * 1000.0;
        let zram_f32_storage = zram_f32_cache.memory_usage_bytes();
        let raw_f32_size = zram_f32_cache.current_pos() * kv_heads * head_dim * 4 * 2; // K+V
        let zram_f32_ratio = raw_f32_size as f64 / zram_f32_storage.max(1) as f64;

        // --- ZramStore F16 ---
        let token_bytes_f16 = kv_heads * head_dim * 2;
        let zram_f16_start = std::time::Instant::now();
        let zram16 = zram_store::ZramStore::new(token_bytes_f16, 2, 64);
        let mut zram_f16_cache = OffloadKVCache::new(
            0,
            kv_heads,
            head_dim,
            DType::F16,
            max_seq_len,
            Box::new(zram16),
        );
        {
            let k = make_realistic_f16_tensor(prefill_len, kv_heads, head_dim, 0x3C00);
            let v = make_realistic_f16_tensor(prefill_len, kv_heads, head_dim, 0x3E00);
            zram_f16_cache.update(&k, &v).unwrap();
        }
        for i in 0..decode_steps {
            let k = make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3C00 + (i as u16 * 3));
            let v = make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3E00 + (i as u16 * 5));
            zram_f16_cache.update(&k, &v).unwrap();
            let _ = zram_f16_cache.get_view();
        }
        let zram_f16_ms = zram_f16_start.elapsed().as_secs_f64() * 1000.0;
        let zram_f16_storage = zram_f16_cache.memory_usage_bytes();
        let raw_f16_size = zram_f16_cache.current_pos() * kv_heads * head_dim * 2 * 2;
        let zram_f16_ratio = raw_f16_size as f64 / zram_f16_storage.max(1) as f64;

        // Print report
        let total_tokens = prefill_len + decode_steps;
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║  KV Cache Offload Integration Benchmark                     ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Config: kv_heads={kv_heads}, head_dim={head_dim}, tokens={total_tokens}");
        println!("║  Prefill={prefill_len}, Decode={decode_steps}");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║  {:20} {:>10} {:>10} {:>10} ║",
            "Method", "Time(ms)", "Storage", "Ratio"
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║  {:20} {:>10.2} {:>9}B {:>10} ║",
            "BASE (KVCache F32)", base_ms, raw_f32_size, "1.00x"
        );
        println!(
            "║  {:20} {:>10.2} {:>9}B {:>10} ║",
            "DiskStore F32", disk_ms, disk_size, "N/A"
        );
        println!(
            "║  {:20} {:>10.2} {:>9}B {:>9.2}x ║",
            "ZramStore F32", zram_f32_ms, zram_f32_storage, zram_f32_ratio
        );
        println!(
            "║  {:20} {:>10.2} {:>9}B {:>9.2}x ║",
            "ZramStore F16", zram_f16_ms, zram_f16_storage, zram_f16_ratio
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Disk overhead: {:.1}x slower", disk_ms / base_ms);
        println!("║  Zram F32 overhead: {:.1}x slower", zram_f32_ms / base_ms);
        println!("║  Zram F16 compression: {:.2}x", zram_f16_ratio);
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        // Accuracy verification (all must be bit-exact)
        let mut base2 = make_base_kvcache(kv_heads, head_dim, max_seq_len, DType::F32);
        let disk2 = disk_store::DiskStore::new(&dir.join("verify"), 0, token_bytes).unwrap();
        let mut disk2_cache = OffloadKVCache::new(
            0,
            kv_heads,
            head_dim,
            DType::F32,
            max_seq_len,
            Box::new(disk2),
        );
        let zram2 = zram_store::ZramStore::new(token_bytes, 4, 64);
        let mut zram2_cache = OffloadKVCache::new(
            0,
            kv_heads,
            head_dim,
            DType::F32,
            max_seq_len,
            Box::new(zram2),
        );

        let k = make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 1.0);
        let v = make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 2.0);
        base2.update(&k, &v).unwrap();
        disk2_cache.update(&k, &v).unwrap();
        zram2_cache.update(&k, &v).unwrap();

        for i in 0..decode_steps {
            let k = make_realistic_f32_tensor(1, kv_heads, head_dim, 0.1 * i as f32);
            let v = make_realistic_f32_tensor(1, kv_heads, head_dim, 0.2 * i as f32);
            base2.update(&k, &v).unwrap();
            disk2_cache.update(&k, &v).unwrap();
            zram2_cache.update(&k, &v).unwrap();
        }

        let (disk_exact, _) = compare_views(&mut base2, &mut disk2_cache, DType::F32);
        assert!(disk_exact, "DiskStore F32 accuracy: NOT bit-exact vs BASE!");

        let (zram_exact, _) = compare_views(&mut base2, &mut zram2_cache, DType::F32);
        assert!(zram_exact, "ZramStore F32 accuracy: NOT bit-exact vs BASE!");

        // ZramStore compression should be meaningful
        assert!(
            zram_f16_ratio >= 1.3,
            "ZramStore F16 compression ratio {zram_f16_ratio:.2}x too low"
        );
        assert!(
            zram_f32_ratio >= 1.2,
            "ZramStore F32 compression ratio {zram_f32_ratio:.2}x too low"
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Preload / Prefetch Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_preload_skips_io_in_get_view() {
        // After preload(), get_view() should NOT call store.load_into() again.
        // Verify by comparing data: preloaded get_view == non-preloaded get_view.
        let kv_heads = 2;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4;
        let store = zram_store::ZramStore::new(token_bytes, 4, 32);
        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        // Store some data
        let k = make_f32_tensor_with_data(
            &(0..kv_heads * head_dim)
                .map(|i| i as f32)
                .collect::<Vec<_>>(),
            1,
            kv_heads,
            head_dim,
        );
        let v = make_f32_tensor_with_data(
            &(0..kv_heads * head_dim)
                .map(|i| 100.0 + i as f32)
                .collect::<Vec<_>>(),
            1,
            kv_heads,
            head_dim,
        );
        cache.update(&k, &v).unwrap();

        // Preload then get_view
        cache.preload().unwrap();
        assert!(cache.preloaded);
        let (k1, v1) = cache.get_view();
        assert!(!cache.preloaded, "preloaded should be reset after get_view");

        // Non-preloaded get_view (sync fallback)
        let (k2, v2) = cache.get_view();

        let k1_data = k1.as_slice::<f32>();
        let k2_data = k2.as_slice::<f32>();
        let v1_data = v1.as_slice::<f32>();
        let v2_data = v2.as_slice::<f32>();
        assert_eq!(k1_data, k2_data, "preloaded vs sync K mismatch");
        assert_eq!(v1_data, v2_data, "preloaded vs sync V mismatch");
    }

    #[test]
    fn test_preload_update_append_to_attn_buf() {
        // When preloaded, update(seq_len=1) should append to attn buffers too.
        let kv_heads = 1;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4;
        let store = zram_store::ZramStore::new(token_bytes, 4, 32);
        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        // Prefill
        let k0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let v0: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        cache
            .update(
                &make_f32_tensor_with_data(&k0, 1, kv_heads, head_dim),
                &make_f32_tensor_with_data(&v0, 1, kv_heads, head_dim),
            )
            .unwrap();

        // Preload (loads 1 token into attn buf)
        cache.preload().unwrap();

        // Decode: append 1 token while preloaded
        let k1: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let v1: Vec<f32> = vec![50.0, 60.0, 70.0, 80.0];
        cache
            .update(
                &make_f32_tensor_with_data(&k1, 1, kv_heads, head_dim),
                &make_f32_tensor_with_data(&v1, 1, kv_heads, head_dim),
            )
            .unwrap();
        assert_eq!(cache.current_pos(), 2);

        // get_view should return both tokens without re-loading
        // (preloaded + appended to attn buf)
        let (k_view, v_view) = cache.get_view();
        let k_data = k_view.as_slice::<f32>();
        let v_data = v_view.as_slice::<f32>();

        assert_eq!(&k_data[..4], &k0);
        assert_eq!(&k_data[4..8], &k1);
        assert_eq!(&v_data[..4], &v0);
        assert_eq!(&v_data[4..8], &v1);
    }

    #[test]
    fn test_release_buffers_frees_memory() {
        let kv_heads = 2;
        let head_dim = 32;
        let token_bytes = kv_heads * head_dim * 2;
        let store = zram_store::ZramStore::new(token_bytes, 2, 64);
        let mut cache =
            OffloadKVCache::new(0, kv_heads, head_dim, DType::F16, 256, Box::new(store));

        // Trigger allocation via preload
        let k = make_test_tensor(4, kv_heads, head_dim, DType::F16, 0xAA);
        let v = make_test_tensor(4, kv_heads, head_dim, DType::F16, 0xBB);
        cache.update(&k, &v).unwrap();
        cache.preload().unwrap();
        assert!(cache.attn_k_buf.is_some());
        assert!(cache.attn_v_buf.is_some());

        // Release
        cache.release_buffers();
        assert!(cache.attn_k_buf.is_none());
        assert!(cache.attn_v_buf.is_none());
        assert!(!cache.preloaded);

        // get_view should still work (sync fallback with fresh allocation)
        let (k_view, _) = cache.get_view();
        assert_eq!(k_view.shape().dims(), &[1, 4, kv_heads, head_dim]);
    }

    #[test]
    fn test_preload_concurrent_split_at_mut() {
        // Simulate the per-layer prefetch pattern: split_at_mut + thread::scope.
        let kv_heads = 2;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4;

        let mut caches: Vec<OffloadKVCache> = (0..4)
            .map(|i| {
                let store = zram_store::ZramStore::new(token_bytes, 4, 32);
                let mut c =
                    OffloadKVCache::new(i, kv_heads, head_dim, DType::F32, 64, Box::new(store));
                // Prefill each cache with 2 tokens
                let k = make_realistic_f32_tensor(2, kv_heads, head_dim, i as f32);
                let v = make_realistic_f32_tensor(2, kv_heads, head_dim, i as f32 + 10.0);
                c.update(&k, &v).unwrap();
                c
            })
            .collect();

        // Preload layer 0
        caches[0].preload().unwrap();

        // Simulate pipeline: compute layer i while preloading layer i+1
        for i in 0..3 {
            let (left, right) = caches.split_at_mut(i + 1);
            let current = &mut left[i];
            let next = &mut right[0];

            std::thread::scope(|s| {
                let handle = s.spawn(|| next.preload());
                // "Compute" on current layer (just read the view)
                let (k, v) = current.get_view();
                assert_eq!(k.shape().dims(), &[1, 2, kv_heads, head_dim]);
                assert_eq!(v.shape().dims(), &[1, 2, kv_heads, head_dim]);
                handle.join().unwrap().unwrap();
            });

            // Release previous layer's buffers
            if i > 0 {
                caches[i - 1].release_buffers();
            }
        }

        // Final layer
        let last = caches.len() - 1;
        let (k, v) = caches[last].get_view();
        assert_eq!(k.shape().dims(), &[1, 2, kv_heads, head_dim]);
        assert_eq!(v.shape().dims(), &[1, 2, kv_heads, head_dim]);
    }

    #[test]
    fn test_preload_empty_cache() {
        // preload() on empty cache should succeed
        let store = zram_store::ZramStore::new(32, 4, 8);
        let mut cache = OffloadKVCache::new(0, 1, 4, DType::F32, 64, Box::new(store));

        cache.preload().unwrap();
        assert!(cache.preloaded);
        // attn buffers should NOT be allocated for empty cache
        assert!(cache.attn_k_buf.is_none());

        let (k, v) = cache.get_view();
        assert_eq!(k.shape().dims(), &[1, 0, 1, 4]);
        assert_eq!(v.shape().dims(), &[1, 0, 1, 4]);
    }

    #[test]
    fn test_reset_preload() {
        let kv_heads = 1;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4; // F32
        let store = zram_store::ZramStore::new(token_bytes, 4, 8);
        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        let k = make_f32_tensor_with_data(&[1.0, 2.0, 3.0, 4.0], 1, 1, 4);
        let v = make_f32_tensor_with_data(&[5.0, 6.0, 7.0, 8.0], 1, 1, 4);
        cache.update(&k, &v).unwrap();

        cache.preload().unwrap();
        assert!(cache.preloaded);

        cache.reset_preload();
        assert!(!cache.preloaded);

        // get_view should still work (sync fallback)
        let (k_view, _) = cache.get_view();
        assert_eq!(k_view.as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_out_buf_reuse() {
        // R-P2: verify SharedBuffer reuse across get_view calls
        let kv_heads = 2;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4;
        let store = zram_store::ZramStore::new(token_bytes, 4, 32);
        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        let k = make_realistic_f32_tensor(1, kv_heads, head_dim, 1.0);
        let v = make_realistic_f32_tensor(1, kv_heads, head_dim, 2.0);
        cache.update(&k, &v).unwrap();

        // First get_view allocates out buffers
        let _ = cache.get_view();
        assert!(cache.out_k_buf.is_some());
        let ptr1 = cache.out_k_buf.as_ref().unwrap().as_ptr();

        // Second get_view should reuse same buffer (same size)
        let _ = cache.get_view();
        let ptr2 = cache.out_k_buf.as_ref().unwrap().as_ptr();
        assert_eq!(ptr1, ptr2, "SharedBuffer should be reused");
    }
}
