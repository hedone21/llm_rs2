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
pub mod prefetch;
pub mod preload_pool;
pub mod preprocess;
pub mod raw_store;
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
    /// Number of tokens written to attn_buf but not yet flushed to store.
    /// When preloaded, decode updates go to attn_buf only; store catches up on release.
    store_behind: usize,
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
            store_behind: 0,
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
        if self.preloaded {
            return Ok(()); // Already loaded, skip redundant work
        }

        // If store_behind > 0, attn_buf has newer data than store — flush first.
        if self.store_behind > 0 {
            self.flush_deferred()?;
        }

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

    /// Flush deferred tokens from attn_buf to store.
    /// Called before releasing buffers or when store needs to be up-to-date.
    fn flush_deferred(&mut self) -> Result<()> {
        if self.store_behind == 0 {
            return Ok(());
        }
        let (Some(k_buf), Some(v_buf)) = (&self.attn_k_buf, &self.attn_v_buf) else {
            anyhow::bail!(
                "OffloadKVCache[{}] flush_deferred: attn_buf is None but store_behind={}",
                self.layer_id,
                self.store_behind
            );
        };
        let start = (self.current_pos - self.store_behind) * self.token_bytes;
        let end = self.current_pos * self.token_bytes;
        for i in (start..end).step_by(self.token_bytes) {
            self.store.append_token(
                &k_buf[i..i + self.token_bytes],
                &v_buf[i..i + self.token_bytes],
            )?;
        }
        self.store_behind = 0;
        Ok(())
    }

    /// Release attn buffers to free memory (R-P1: only 2 layers active at once).
    pub fn release_buffers(&mut self) {
        if self.store_behind > 0
            && let Err(e) = self.flush_deferred()
        {
            log::error!(
                "OffloadKVCache[{}] flush_deferred failed: {}",
                self.layer_id,
                e
            );
        }
        self.attn_k_buf = None;
        self.attn_v_buf = None;
        self.preloaded = false;
    }

    /// Reset preloaded flag (R-P5: call at token boundary).
    pub fn reset_preload(&mut self) {
        self.preloaded = false;
    }

    /// Re-arm preloaded flag for cross-token buffer retention.
    /// Safe to call after `get_view()`: attn buffers still hold valid data
    /// because `get_view()` copies out (does not consume the source).
    pub fn retain_preload(&mut self) {
        if self.attn_k_buf.is_some() && self.attn_v_buf.is_some() {
            self.preloaded = true;
        }
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
            if self.preloaded
                && let (Some(k_buf), Some(v_buf)) = (&mut self.attn_k_buf, &mut self.attn_v_buf)
            {
                // Deferred write: attn_buf is authoritative, skip store write.
                // Store will catch up in flush_deferred() on release_buffers().
                let offset = self.current_pos * self.token_bytes;
                k_buf[offset..offset + self.token_bytes].copy_from_slice(k_data);
                v_buf[offset..offset + self.token_bytes].copy_from_slice(v_data);
                self.store_behind += 1;
            } else {
                // Non-retained: write to store immediately
                self.store.append_token(k_data, v_data)?;
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

        // Flush deferred tokens before borrowing attn buffers (avoids borrow conflict).
        // Track whether attn_buf already has valid data (deferred tokens in it).
        let attn_buf_valid = !self.preloaded && self.store_behind > 0;
        if attn_buf_valid {
            // attn_buf holds valid data — just sync store, no load_into needed.
            if let Err(e) = self.flush_deferred() {
                log::error!(
                    "OffloadKVCache[{}] flush_deferred in get_view failed: {}",
                    self.layer_id,
                    e
                );
            }
        }

        // Lazy-allocate attn buffers if needed
        let k_attn = self.attn_k_buf.get_or_insert_with(|| vec![0u8; max_bytes]);
        let v_attn = self.attn_v_buf.get_or_insert_with(|| vec![0u8; max_bytes]);

        if !self.preloaded && !attn_buf_valid {
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

impl crate::core::kv_cache::PrefetchableCache for OffloadKVCache {
    fn preload(&mut self) -> Result<()> {
        self.preload()
    }

    fn release_buffers(&mut self) {
        self.release_buffers();
    }

    fn reset_preload(&mut self) {
        self.reset_preload();
    }

    fn retain_preload(&mut self) {
        self.retain_preload();
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
    //  Phase 3 Benchmark: Sync vs Preload, Memory Analysis
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_bench_preload_vs_sync_and_memory() {
        // Llama 3.2 1B-scale parameters
        let kv_heads = 8;
        let head_dim = 64;
        let num_layers = 16;
        let max_seq_len = 2048;
        let prefill_len = 64;
        let decode_steps = 128;

        let token_bytes_f16 = kv_heads * head_dim * 2; // 1024 bytes/token
        let token_bytes_f32 = kv_heads * head_dim * 4; // 2048 bytes/token

        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║  Phase 3 KV Offload Benchmark: Sync vs Preload + Memory        ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Config: kv_heads={}, head_dim={}, layers={}, max_seq={}",
            kv_heads, head_dim, num_layers, max_seq_len
        );
        println!("║  Prefill={}, Decode={}", prefill_len, decode_steps);
        println!("╠══════════════════════════════════════════════════════════════════╣");

        let total_tokens = prefill_len + decode_steps;

        // ── Helper: create N-layer caches and run decode simulation ──
        let run_decode_bench = |mode: &str,
                                dtype: DType,
                                use_preload: bool|
         -> (f64, f64, f64, usize, usize) {
            let token_bytes = kv_heads * head_dim * dtype.size();
            let mut caches: Vec<OffloadKVCache> = (0..num_layers)
                .map(|layer_id| {
                    let store: Box<dyn store::OffloadStore> = match mode {
                        "zram" => {
                            Box::new(zram_store::ZramStore::new(token_bytes, dtype.size(), 64))
                        }
                        "disk" => {
                            let dir = std::env::temp_dir()
                                .join(format!("llm_rs2_bench3_{}_{:?}_{}", mode, dtype, layer_id));
                            Box::new(
                                disk_store::DiskStore::new(&dir, layer_id, token_bytes).unwrap(),
                            )
                        }
                        _ => unreachable!(),
                    };
                    OffloadKVCache::new(layer_id, kv_heads, head_dim, dtype, max_seq_len, store)
                })
                .collect();

            // Prefill
            let k_pf = if dtype == DType::F32 {
                make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 1.0)
            } else {
                make_realistic_f16_tensor(prefill_len, kv_heads, head_dim, 0x3C00)
            };
            let v_pf = if dtype == DType::F32 {
                make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 2.0)
            } else {
                make_realistic_f16_tensor(prefill_len, kv_heads, head_dim, 0x3E00)
            };
            for c in caches.iter_mut() {
                c.update(&k_pf, &v_pf).unwrap();
            }

            // Decode loop
            let decode_start = std::time::Instant::now();
            let mut total_io_us = 0u64;
            let mut total_getview_us = 0u64;

            for step in 0..decode_steps {
                // Generate per-step token data
                let k_tok = if dtype == DType::F32 {
                    make_realistic_f32_tensor(1, kv_heads, head_dim, 0.1 * step as f32)
                } else {
                    make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3C00 + (step as u16 * 3))
                };
                let v_tok = if dtype == DType::F32 {
                    make_realistic_f32_tensor(1, kv_heads, head_dim, 0.2 * step as f32)
                } else {
                    make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3E00 + (step as u16 * 5))
                };

                if use_preload {
                    // Reset preload flags
                    for c in caches.iter_mut() {
                        c.reset_preload();
                    }

                    // Update all layers
                    for c in caches.iter_mut() {
                        c.update(&k_tok, &v_tok).unwrap();
                    }

                    // Preload layer 0
                    caches[0].preload().unwrap();

                    // Pipeline: compute(layer i) || preload(layer i+1)
                    for i in 0..num_layers {
                        if i + 1 < num_layers {
                            let (left, right) = caches.split_at_mut(i + 1);
                            let current = &mut left[i];
                            let next = &mut right[0];

                            std::thread::scope(|s| {
                                let io_start = std::time::Instant::now();
                                let handle = s.spawn(|| next.preload().unwrap());

                                let gv_start = std::time::Instant::now();
                                let _ = current.get_view();
                                total_getview_us += gv_start.elapsed().as_micros() as u64;

                                handle.join().unwrap();
                                total_io_us += io_start.elapsed().as_micros() as u64;
                            });

                            if i > 0 {
                                caches[i - 1].release_buffers();
                            }
                        } else {
                            let gv_start = std::time::Instant::now();
                            let _ = caches[i].get_view();
                            total_getview_us += gv_start.elapsed().as_micros() as u64;
                            caches[i].release_buffers();
                        }
                    }
                } else {
                    // Sync: update + get_view for all layers
                    for c in caches.iter_mut() {
                        c.update(&k_tok, &v_tok).unwrap();
                    }
                    for c in caches.iter_mut() {
                        let gv_start = std::time::Instant::now();
                        let _ = c.get_view();
                        total_getview_us += gv_start.elapsed().as_micros() as u64;
                    }
                }
            }
            let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
            let avg_per_token_ms = decode_ms / decode_steps as f64;
            let avg_getview_us = total_getview_us as f64 / (decode_steps * num_layers) as f64;

            // Memory analysis
            let store_mem: usize = caches.iter().map(|c| c.memory_usage_bytes()).sum();
            let attn_buf_mem: usize = caches
                .iter()
                .map(|c| {
                    c.attn_k_buf.as_ref().map_or(0, |b| b.len())
                        + c.attn_v_buf.as_ref().map_or(0, |b| b.len())
                })
                .sum();

            (
                decode_ms,
                avg_per_token_ms,
                avg_getview_us,
                store_mem,
                attn_buf_mem,
            )
        };

        // ── Baseline: KVCache F32 ──
        let base_start = std::time::Instant::now();
        {
            let mut bases: Vec<KVCache> = (0..num_layers)
                .map(|_| make_base_kvcache(kv_heads, head_dim, max_seq_len, DType::F32))
                .collect();
            let k_pf = make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 1.0);
            let v_pf = make_realistic_f32_tensor(prefill_len, kv_heads, head_dim, 2.0);
            for c in bases.iter_mut() {
                c.update(&k_pf, &v_pf).unwrap();
            }
            for step in 0..decode_steps {
                let k = make_realistic_f32_tensor(1, kv_heads, head_dim, 0.1 * step as f32);
                let v = make_realistic_f32_tensor(1, kv_heads, head_dim, 0.2 * step as f32);
                for c in bases.iter_mut() {
                    c.update(&k, &v).unwrap();
                    let _ = KVCacheOps::get_view(c);
                }
            }
        }
        let base_ms = base_start.elapsed().as_secs_f64() * 1000.0;
        let base_per_tok = base_ms / decode_steps as f64;
        // BASE memory: 16 layers × max_seq_len × kv_heads × head_dim × 4 × 2 (K+V)
        let base_mem = num_layers * max_seq_len * kv_heads * head_dim * 4 * 2;

        // ── Baseline: KVCache F16 ──
        let base_f16_start = std::time::Instant::now();
        {
            let mut bases: Vec<KVCache> = (0..num_layers)
                .map(|_| make_base_kvcache(kv_heads, head_dim, max_seq_len, DType::F16))
                .collect();
            let k_pf = make_realistic_f16_tensor(prefill_len, kv_heads, head_dim, 0x3C00);
            let v_pf = make_realistic_f16_tensor(prefill_len, kv_heads, head_dim, 0x3E00);
            for c in bases.iter_mut() {
                c.update(&k_pf, &v_pf).unwrap();
            }
            for step in 0..decode_steps {
                let k =
                    make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3C00 + (step as u16 * 3));
                let v =
                    make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3E00 + (step as u16 * 5));
                for c in bases.iter_mut() {
                    c.update(&k, &v).unwrap();
                    let _ = KVCacheOps::get_view(c);
                }
            }
        }
        let base_f16_ms = base_f16_start.elapsed().as_secs_f64() * 1000.0;
        let base_f16_per_tok = base_f16_ms / decode_steps as f64;
        let base_f16_mem = num_layers * max_seq_len * kv_heads * head_dim * 2 * 2;

        // ── ZramStore F16: Sync ──
        let (zram_f16_sync_ms, zram_f16_sync_pt, zram_f16_sync_gv, zram_f16_store, _) =
            run_decode_bench("zram", DType::F16, false);

        // ── ZramStore F16: Preload ──
        let (
            zram_f16_pre_ms,
            zram_f16_pre_pt,
            zram_f16_pre_gv,
            zram_f16_pre_store,
            zram_f16_pre_attn,
        ) = run_decode_bench("zram", DType::F16, true);

        // ── ZramStore F32: Sync ──
        let (zram_f32_sync_ms, zram_f32_sync_pt, zram_f32_sync_gv, zram_f32_store, _) =
            run_decode_bench("zram", DType::F32, false);

        // ── ZramStore F32: Preload ──
        let (
            zram_f32_pre_ms,
            zram_f32_pre_pt,
            zram_f32_pre_gv,
            zram_f32_pre_store,
            zram_f32_pre_attn,
        ) = run_decode_bench("zram", DType::F32, true);

        // ── DiskStore F16: Sync ──
        let (disk_f16_sync_ms, disk_f16_sync_pt, disk_f16_sync_gv, disk_f16_store, _) =
            run_decode_bench("disk", DType::F16, false);

        // ── DiskStore F16: Preload ──
        let (disk_f16_pre_ms, disk_f16_pre_pt, disk_f16_pre_gv, _, disk_f16_pre_attn) =
            run_decode_bench("disk", DType::F16, true);

        let raw_f16_kv = total_tokens * token_bytes_f16 * 2 * num_layers; // K+V, all layers
        let raw_f32_kv = total_tokens * token_bytes_f32 * 2 * num_layers;

        // ═══ Performance Report ═══
        println!("║");
        println!(
            "║  {:30} {:>8} {:>10} {:>10} {:>8}",
            "Configuration", "Total", "Per-Token", "get_view", "vs BASE"
        );
        println!(
            "║  {:30} {:>8} {:>10} {:>10} {:>8}",
            "", "(ms)", "(ms/tok)", "(μs/call)", ""
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!(
            "║  {:30} {:>8.1} {:>10.3} {:>10} {:>8}",
            "BASE KVCache F32", base_ms, base_per_tok, "—", "1.0x"
        );
        println!(
            "║  {:30} {:>8.1} {:>10.3} {:>10} {:>8}",
            "BASE KVCache F16",
            base_f16_ms,
            base_f16_per_tok,
            "—",
            format!("{:.1}x", base_f16_ms / base_ms)
        );
        println!("║  ──────────────────────────────────────────────────────────────");
        println!(
            "║  {:30} {:>8.1} {:>10.3} {:>10.1} {:>8}",
            "ZramStore F16 (sync)",
            zram_f16_sync_ms,
            zram_f16_sync_pt,
            zram_f16_sync_gv,
            format!("{:.1}x", zram_f16_sync_ms / base_f16_ms)
        );
        println!(
            "║  {:30} {:>8.1} {:>10.3} {:>10.1} {:>8}",
            "ZramStore F16 (preload)",
            zram_f16_pre_ms,
            zram_f16_pre_pt,
            zram_f16_pre_gv,
            format!("{:.1}x", zram_f16_pre_ms / base_f16_ms)
        );
        let f16_speedup = zram_f16_sync_ms / zram_f16_pre_ms;
        println!("║    ↳ preload speedup vs sync: {:.2}x", f16_speedup);
        println!("║  ──────────────────────────────────────────────────────────────");
        println!(
            "║  {:30} {:>8.1} {:>10.3} {:>10.1} {:>8}",
            "ZramStore F32 (sync)",
            zram_f32_sync_ms,
            zram_f32_sync_pt,
            zram_f32_sync_gv,
            format!("{:.1}x", zram_f32_sync_ms / base_ms)
        );
        println!(
            "║  {:30} {:>8.1} {:>10.3} {:>10.1} {:>8}",
            "ZramStore F32 (preload)",
            zram_f32_pre_ms,
            zram_f32_pre_pt,
            zram_f32_pre_gv,
            format!("{:.1}x", zram_f32_pre_ms / base_ms)
        );
        let f32_speedup = zram_f32_sync_ms / zram_f32_pre_ms;
        println!("║    ↳ preload speedup vs sync: {:.2}x", f32_speedup);
        println!("║  ──────────────────────────────────────────────────────────────");
        println!(
            "║  {:30} {:>8.1} {:>10.3} {:>10.1} {:>8}",
            "DiskStore F16 (sync)",
            disk_f16_sync_ms,
            disk_f16_sync_pt,
            disk_f16_sync_gv,
            format!("{:.1}x", disk_f16_sync_ms / base_f16_ms)
        );
        println!(
            "║  {:30} {:>8.1} {:>10.3} {:>10.1} {:>8}",
            "DiskStore F16 (preload)",
            disk_f16_pre_ms,
            disk_f16_pre_pt,
            disk_f16_pre_gv,
            format!("{:.1}x", disk_f16_pre_ms / base_f16_ms)
        );
        let disk_speedup = disk_f16_sync_ms / disk_f16_pre_ms;
        println!("║    ↳ preload speedup vs sync: {:.2}x", disk_speedup);

        // ═══ Memory Report ═══
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!(
            "║  MEMORY ANALYSIS (at {} tokens, {} layers)",
            total_tokens, num_layers
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!(
            "║  {:30} {:>10} {:>10} {:>10} {:>8}",
            "Configuration", "KV Data", "Attn Buf", "Total", "vs BASE"
        );
        println!("║  ──────────────────────────────────────────────────────────────");
        println!(
            "║  {:30} {:>9}K {:>10} {:>9}K {:>8}",
            "BASE KVCache F32",
            base_mem / 1024,
            "—",
            base_mem / 1024,
            "1.0x"
        );
        println!(
            "║  {:30} {:>9}K {:>10} {:>9}K {:>8}",
            "BASE KVCache F16",
            base_f16_mem / 1024,
            "—",
            base_f16_mem / 1024,
            "1.0x"
        );
        println!("║  ──────────────────────────────────────────────────────────────");

        let zram_f16_total = zram_f16_pre_store + zram_f16_pre_attn;
        let zram_f16_savings = 1.0 - (zram_f16_total as f64 / base_f16_mem as f64);
        println!(
            "║  {:30} {:>9}K {:>9}K {:>9}K {:>7.0}%",
            "ZramStore F16 (preload)",
            zram_f16_pre_store / 1024,
            zram_f16_pre_attn / 1024,
            zram_f16_total / 1024,
            zram_f16_savings * 100.0
        );
        println!(
            "║    ↳ store compression: {:.2}x (raw {}K → {}K)",
            raw_f16_kv as f64 / zram_f16_pre_store.max(1) as f64,
            raw_f16_kv / 1024,
            zram_f16_pre_store / 1024,
        );

        let zram_f32_total = zram_f32_pre_store + zram_f32_pre_attn;
        let zram_f32_savings = 1.0 - (zram_f32_total as f64 / base_mem as f64);
        println!(
            "║  {:30} {:>9}K {:>9}K {:>9}K {:>7.0}%",
            "ZramStore F32 (preload)",
            zram_f32_pre_store / 1024,
            zram_f32_pre_attn / 1024,
            zram_f32_total / 1024,
            zram_f32_savings * 100.0
        );
        println!(
            "║    ↳ store compression: {:.2}x (raw {}K → {}K)",
            raw_f32_kv as f64 / zram_f32_pre_store.max(1) as f64,
            raw_f32_kv / 1024,
            zram_f32_pre_store / 1024,
        );

        let disk_f16_total = disk_f16_store + disk_f16_pre_attn;
        let disk_f16_savings = 1.0 - (disk_f16_total as f64 / base_f16_mem as f64);
        println!(
            "║  {:30} {:>9}K {:>9}K {:>9}K {:>7.0}%",
            "DiskStore F16 (preload)",
            disk_f16_store / 1024,
            disk_f16_pre_attn / 1024,
            disk_f16_total / 1024,
            disk_f16_savings * 100.0
        );
        println!("║    ↳ NOTE: disk store KB = file size (data on disk, not in RAM)",);

        // Attn buffer analysis
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║  ATTN BUFFER ANALYSIS (preload mode)");
        println!("║  ──────────────────────────────────────────────────────────────");
        let max_attn_f16 = 2 * max_seq_len * token_bytes_f16; // 2 layers (K+V)
        let max_attn_f32 = 2 * max_seq_len * token_bytes_f32;
        let naive_attn_f16 = num_layers * max_seq_len * token_bytes_f16 * 2;
        let naive_attn_f32 = num_layers * max_seq_len * token_bytes_f32 * 2;
        println!(
            "║  F16: active={} layers × {}K = {}K  (naive 16 layers = {}K, saving {:.0}%)",
            2,
            max_seq_len * token_bytes_f16 / 1024,
            max_attn_f16 / 1024,
            naive_attn_f16 / 1024,
            (1.0 - max_attn_f16 as f64 / naive_attn_f16 as f64) * 100.0,
        );
        println!(
            "║  F32: active={} layers × {}K = {}K  (naive 16 layers = {}K, saving {:.0}%)",
            2,
            max_seq_len * token_bytes_f32 / 1024,
            max_attn_f32 / 1024,
            naive_attn_f32 / 1024,
            (1.0 - max_attn_f32 as f64 / naive_attn_f32 as f64) * 100.0,
        );
        println!("╚══════════════════════════════════════════════════════════════════╝\n");
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

    #[test]
    fn test_preload_idempotent() {
        let token_bytes = 2 * 32 * 2;
        let store = zram_store::ZramStore::new(token_bytes, 2, 64);
        let mut cache = OffloadKVCache::new(0, 2, 32, DType::F16, 256, Box::new(store));

        let k = make_test_tensor(4, 2, 32, DType::F16, 0xAB);
        let v = make_test_tensor(4, 2, 32, DType::F16, 0xCD);
        cache.update(&k, &v).unwrap();

        // First preload
        cache.preload().unwrap();
        let k_ptr1 = cache.attn_k_buf.as_ref().unwrap().as_ptr();

        // Second preload (should be idempotent — skips due to guard)
        cache.preload().unwrap();
        let k_ptr2 = cache.attn_k_buf.as_ref().unwrap().as_ptr();
        assert_eq!(k_ptr1, k_ptr2, "preload should be idempotent");

        // Data should still be correct after double preload
        let (k_view, v_view) = cache.get_view();
        assert_eq!(k_view.shape().dims(), &[1, 4, 2, 32]);
        assert_eq!(v_view.shape().dims(), &[1, 4, 2, 32]);

        let k_out =
            unsafe { std::slice::from_raw_parts(k_view.buffer().as_ptr(), 4 * token_bytes) };
        assert!(
            k_out.iter().all(|&b| b == 0xAB),
            "K data corrupted after double preload"
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Adaptive Prefetch Benchmark: depth=1 vs adaptive with simulated work
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_bench_adaptive_prefetch() {
        use crate::core::offload::prefetch::PrefetchController;

        let kv_heads = 8;
        let head_dim = 64;
        let num_layers = 16;
        let max_seq_len = 2048;
        let prefill_len = 128;
        let decode_steps = 32;

        println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
        println!("║  Adaptive Prefetch Benchmark: depth=1 vs adaptive                       ║");
        println!("╠═══════════════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Config: kv_heads={}, head_dim={}, layers={}, prefill={}, decode={}",
            kv_heads, head_dim, num_layers, prefill_len, decode_steps
        );
        println!("╠══════════════════════════════════════════════════════════════════╣");

        // Simulate realistic timing: on ARM devices, layer forward takes ~2-4ms
        // and zram decompression takes ~2-6ms per layer. On the host, zram preload
        // is near-instant, so we inject artificial delays to exercise the adaptive
        // depth controller and measure overlap effectiveness.
        //
        // simulate_forward: ~2ms (simulates matrix ops per layer)
        // simulate_preload_extra: ~3ms (simulates zram decompression overhead)
        let simulate_forward = || {
            std::thread::sleep(std::time::Duration::from_millis(2));
        };
        let simulate_preload_extra = std::time::Duration::from_millis(3);

        /// Run decode loop with fire-and-forget prefetch strategy.
        /// Returns (total_ms, preload_calls, preload_skips, active_bufs_at_end, final_depth)
        fn run_with_depth(
            num_layers: usize,
            max_seq_len: usize,
            kv_heads: usize,
            head_dim: usize,
            prefill_len: usize,
            decode_steps: usize,
            max_depth: usize,
            simulate_forward: &dyn Fn(),
            preload_extra_delay: std::time::Duration,
        ) -> (f64, usize, usize, usize, usize) {
            let token_bytes = kv_heads * head_dim * 2;
            let mut caches: Vec<OffloadKVCache> = (0..num_layers)
                .map(|layer_id| {
                    let store: Box<dyn store::OffloadStore> =
                        Box::new(zram_store::ZramStore::new(token_bytes, 2, 64));
                    OffloadKVCache::new(
                        layer_id,
                        kv_heads,
                        head_dim,
                        DType::F16,
                        max_seq_len,
                        store,
                    )
                })
                .collect();

            // Prefill
            let k_pf = make_realistic_f16_tensor(prefill_len, kv_heads, head_dim, 0x3C00);
            let v_pf = make_realistic_f16_tensor(prefill_len, kv_heads, head_dim, 0x3E00);
            for c in caches.iter_mut() {
                c.update(&k_pf, &v_pf).unwrap();
            }

            let mut prefetch = PrefetchController::new(max_depth, num_layers);
            let mut preload_calls = 0usize;
            let mut preload_skips = 0usize;

            let decode_start = std::time::Instant::now();
            let caches_ptr = caches.as_mut_ptr();

            for step in 0..decode_steps {
                let k_tok =
                    make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3C00 + (step as u16 * 3));
                let v_tok =
                    make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3E00 + (step as u16 * 5));

                for c in caches.iter_mut() {
                    c.update(&k_tok, &v_tok).unwrap();
                }

                let depth = prefetch.depth();

                // Fire-and-forget prefetch (mirrors forward_into_offload)
                std::thread::scope(|s| {
                    // Sync preload [0..depth)
                    for j in 0..depth.min(num_layers) {
                        let was = unsafe { (*caches_ptr.add(j)).preloaded };
                        let t0 = std::time::Instant::now();
                        unsafe { (*caches_ptr.add(j)).preload().unwrap() };
                        if !was {
                            std::thread::sleep(preload_extra_delay);
                        }
                        prefetch.record_preload(t0.elapsed());
                        preload_calls += 1;
                        if was {
                            preload_skips += 1;
                        }
                    }

                    // Pending preload handles: each returns preload duration
                    type Handle<'s> =
                        Option<std::thread::ScopedJoinHandle<'s, std::time::Duration>>;
                    let mut pending: Vec<Handle<'_>> = (0..num_layers).map(|_| None).collect();

                    // Fire initial background preloads [depth..2*depth)
                    for j in depth..(2 * depth).min(num_layers) {
                        let cache_j = unsafe { &mut *caches_ptr.add(j) };
                        let delay = preload_extra_delay;
                        preload_calls += 1;
                        pending[j] = Some(s.spawn(move || {
                            let t0 = std::time::Instant::now();
                            cache_j.preload().unwrap();
                            std::thread::sleep(delay);
                            t0.elapsed()
                        }));
                    }

                    // Layer loop
                    for i in 0..num_layers {
                        // Join pending preload for this layer
                        if let Some(handle) = pending[i].take() {
                            let preload_dur = handle.join().unwrap();
                            prefetch.record_preload(preload_dur);
                        }

                        // Fire preload for layer i + depth
                        let far_idx = i + depth;
                        if far_idx < num_layers && pending[far_idx].is_none() {
                            let cache_far = unsafe { &mut *caches_ptr.add(far_idx) };
                            let delay = preload_extra_delay;
                            preload_calls += 1;
                            pending[far_idx] = Some(s.spawn(move || {
                                let t0 = std::time::Instant::now();
                                cache_far.preload().unwrap();
                                std::thread::sleep(delay);
                                t0.elapsed()
                            }));
                        }

                        // Forward (get_view + simulated work)
                        let current = unsafe { &mut *caches_ptr.add(i) };
                        let fwd_t0 = std::time::Instant::now();
                        let _ = current.get_view();
                        simulate_forward();
                        prefetch.record_forward(fwd_t0.elapsed());

                        // Cross-token retention for first `depth` layers
                        if i < depth {
                            unsafe { (*caches_ptr.add(i)).retain_preload() };
                        }

                        // Release previous layer (skip retained)
                        if i > 0 && (i - 1) >= depth {
                            unsafe { (*caches_ptr.add(i - 1)).release_buffers() };
                        }
                    }

                    // Join remaining
                    for handle in pending.into_iter().flatten() {
                        let preload_dur = handle.join().unwrap();
                        prefetch.record_preload(preload_dur);
                    }

                    // Release last layer (unless retained)
                    if num_layers >= 1 && (num_layers - 1) >= depth {
                        caches[num_layers - 1].release_buffers();
                    }
                });

                prefetch.adjust();
            }

            let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

            // Count active buffers at end
            let active_bufs = caches
                .iter()
                .filter(|c| c.attn_k_buf.is_some() || c.attn_v_buf.is_some())
                .count();

            (
                decode_ms,
                preload_calls,
                preload_skips,
                active_bufs,
                prefetch.depth(),
            )
        }

        // Run BASE: no offload, pure forward only (theoretical minimum)
        let base_ms = {
            let start = std::time::Instant::now();
            for _step in 0..decode_steps {
                for _layer in 0..num_layers {
                    simulate_forward();
                }
            }
            start.elapsed().as_secs_f64() * 1000.0
        };

        // Run with depth=1 (fixed)
        let (d1_ms, _d1_calls, d1_skips, d1_bufs, d1_depth) = run_with_depth(
            num_layers,
            max_seq_len,
            kv_heads,
            head_dim,
            prefill_len,
            decode_steps,
            1,
            &simulate_forward,
            simulate_preload_extra,
        );

        // Run with adaptive (max_depth=4)
        let (da_ms, _da_calls, _da_skips, da_bufs, da_depth) = run_with_depth(
            num_layers,
            max_seq_len,
            kv_heads,
            head_dim,
            prefill_len,
            decode_steps,
            4,
            &simulate_forward,
            simulate_preload_extra,
        );

        // Run with depth=4 (fixed max)
        let (d4_ms, _d4_calls, _d4_skips, _d4_bufs, d4_depth) = run_with_depth(
            num_layers,
            max_seq_len,
            kv_heads,
            head_dim,
            prefill_len,
            decode_steps,
            4,
            &simulate_forward,
            simulate_preload_extra,
        );

        let base_per_tok = base_ms / decode_steps as f64;
        let d1_per_tok = d1_ms / decode_steps as f64;
        let da_per_tok = da_ms / decode_steps as f64;
        let d4_per_tok = d4_ms / decode_steps as f64;

        println!("║");
        println!(
            "║  {:25} {:>10} {:>10} {:>8} {:>10}",
            "Strategy", "Total(ms)", "ms/tok", "Depth", "vs BASE"
        );
        println!("╠═══════════════════════════════════════════════════════════════════════════╣");
        println!(
            "║  {:25} {:>10.1} {:>10.2} {:>8} {:>10}",
            "BASE (no offload)", base_ms, base_per_tok, "-", "-"
        );
        println!(
            "║  {:25} {:>10.1} {:>10.2} {:>8} {:>+9.1}%",
            "Fixed depth=1",
            d1_ms,
            d1_per_tok,
            d1_depth,
            (d1_ms / base_ms - 1.0) * 100.0
        );
        println!(
            "║  {:25} {:>10.1} {:>10.2} {:>8} {:>+9.1}%",
            "Adaptive (max=4)",
            da_ms,
            da_per_tok,
            da_depth,
            (da_ms / base_ms - 1.0) * 100.0
        );
        println!(
            "║  {:25} {:>10.1} {:>10.2} {:>8} {:>+9.1}%",
            "Fixed depth=4",
            d4_ms,
            d4_per_tok,
            d4_depth,
            (d4_ms / base_ms - 1.0) * 100.0
        );
        println!("╠═══════════════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Offload overhead: depth=1 {:.1}%, adaptive {:.1}% (vs BASE)",
            (d1_ms / base_ms - 1.0) * 100.0,
            (da_ms / base_ms - 1.0) * 100.0,
        );
        println!(
            "║  Adaptive vs depth=1: {:.1}% faster",
            (1.0 - da_ms / d1_ms) * 100.0,
        );
        println!("╚═══════════════════════════════════════════════════════════════════════════╝\n");

        // Correctness assertions:
        // 1. depth=1 with retention: first token preloads layer 0 (no skip),
        //    subsequent tokens find layer 0 retained (skip). Total skips = decode_steps - 1.
        assert_eq!(
            d1_skips,
            decode_steps - 1,
            "depth=1 should skip retained preloads on tokens 2+"
        );

        // 2. Active buffers at end should be bounded (not all layers active)
        assert!(
            d1_bufs <= 3,
            "depth=1: too many active buffers at end: {d1_bufs}"
        );
        assert!(
            da_bufs <= 6,
            "adaptive: too many active buffers at end: {da_bufs}"
        );

        // 3. Adaptive should not be catastrophically slower (< 2x overhead)
        assert!(
            da_ms < d1_ms * 2.5,
            "adaptive is too slow: {da_ms:.1}ms vs depth=1 {d1_ms:.1}ms"
        );
    }

    #[test]
    fn test_bench_pool_vs_scope() {
        // Compare thread::scope (spawn per layer) vs PreloadPool (persistent workers).
        // Measures real preload cost without artificial delays — isolates threading overhead.
        use crate::core::offload::prefetch::PrefetchController;
        use crate::core::offload::preload_pool::{self, PreloadPool};

        let kv_heads = 8;
        let head_dim = 64;
        let num_layers = 16;
        let max_seq_len = 2048;
        let prefill_len = 128;
        let decode_steps = 64;
        let max_depth = 4;
        let token_bytes = kv_heads * head_dim * 2;

        println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
        println!("║  PreloadPool vs thread::scope Benchmark                                 ║");
        println!("╠═══════════════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Config: layers={num_layers}, depth={max_depth}, decode={decode_steps}, \
             kv_heads={kv_heads}, head_dim={head_dim}"
        );
        println!("╠═══════════════════════════════════════════════════════════════════════════╣");

        // Helper: create and prefill caches
        let make_caches = || -> Vec<OffloadKVCache> {
            let mut caches: Vec<OffloadKVCache> = (0..num_layers)
                .map(|layer_id| {
                    let store: Box<dyn store::OffloadStore> =
                        Box::new(zram_store::ZramStore::new(token_bytes, 2, 64));
                    OffloadKVCache::new(
                        layer_id,
                        kv_heads,
                        head_dim,
                        DType::F16,
                        max_seq_len,
                        store,
                    )
                })
                .collect();
            let k_pf = make_realistic_f16_tensor(prefill_len, kv_heads, head_dim, 0x3C00);
            let v_pf = make_realistic_f16_tensor(prefill_len, kv_heads, head_dim, 0x3E00);
            for c in caches.iter_mut() {
                c.update(&k_pf, &v_pf).unwrap();
            }
            caches
        };

        // ─── thread::scope baseline ───
        let scope_ms = {
            let mut caches = make_caches();
            let caches_ptr = caches.as_mut_ptr();
            let mut prefetch = PrefetchController::new(max_depth, num_layers);

            let start = std::time::Instant::now();
            for step in 0..decode_steps {
                let k_tok =
                    make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3C00 + (step as u16 * 3));
                let v_tok =
                    make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3E00 + (step as u16 * 5));
                for c in caches.iter_mut() {
                    c.update(&k_tok, &v_tok).unwrap();
                }
                let depth = prefetch.depth();

                std::thread::scope(|s| {
                    for j in 0..depth.min(num_layers) {
                        unsafe { (*caches_ptr.add(j)).preload().unwrap() };
                    }
                    type Handle<'s> =
                        Option<std::thread::ScopedJoinHandle<'s, std::time::Duration>>;
                    let mut pending: Vec<Handle<'_>> = (0..num_layers).map(|_| None).collect();
                    for j in depth..(2 * depth).min(num_layers) {
                        let cache_j = unsafe { &mut *caches_ptr.add(j) };
                        pending[j] = Some(s.spawn(move || {
                            let t0 = std::time::Instant::now();
                            cache_j.preload().unwrap();
                            t0.elapsed()
                        }));
                    }
                    for i in 0..num_layers {
                        if let Some(handle) = pending[i].take() {
                            prefetch.record_preload(handle.join().unwrap());
                        }
                        let far_idx = i + depth;
                        if far_idx < num_layers && pending[far_idx].is_none() {
                            let cache_far = unsafe { &mut *caches_ptr.add(far_idx) };
                            pending[far_idx] = Some(s.spawn(move || {
                                let t0 = std::time::Instant::now();
                                cache_far.preload().unwrap();
                                t0.elapsed()
                            }));
                        }
                        let current = unsafe { &mut *caches_ptr.add(i) };
                        let fwd_t0 = std::time::Instant::now();
                        let _ = current.get_view();
                        prefetch.record_forward(fwd_t0.elapsed());
                        if i < depth {
                            unsafe { (*caches_ptr.add(i)).retain_preload() };
                        }
                        if i > 0 && (i - 1) >= depth {
                            unsafe { (*caches_ptr.add(i - 1)).release_buffers() };
                        }
                    }
                    for handle in pending.into_iter().flatten() {
                        let _ = handle.join();
                    }
                    if num_layers >= 1 && (num_layers - 1) >= depth {
                        caches[num_layers - 1].release_buffers();
                    }
                });
                prefetch.adjust();
            }
            start.elapsed().as_secs_f64() * 1000.0
        };

        // ─── PreloadPool ───
        let pool_ms = {
            let mut caches = make_caches();
            let caches_ptr = caches.as_mut_ptr();
            let mut prefetch = PrefetchController::new(max_depth, num_layers);
            let pool = PreloadPool::new(max_depth);

            let start = std::time::Instant::now();
            for step in 0..decode_steps {
                let k_tok =
                    make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3C00 + (step as u16 * 3));
                let v_tok =
                    make_realistic_f16_tensor(1, kv_heads, head_dim, 0x3E00 + (step as u16 * 5));
                for c in caches.iter_mut() {
                    c.update(&k_tok, &v_tok).unwrap();
                }
                let depth = prefetch.depth();

                // Sync preload [0..depth)
                for j in 0..depth.min(num_layers) {
                    unsafe { (*caches_ptr.add(j)).preload().unwrap() };
                }

                // Pending receivers
                let mut pending: Vec<
                    Option<std::sync::mpsc::Receiver<preload_pool::PreloadResult>>,
                > = (0..num_layers).map(|_| None).collect();

                // Fire initial bg preloads [depth..2*depth)
                #[allow(clippy::needless_range_loop)]
                for j in depth..(2 * depth).min(num_layers) {
                    pending[j] = Some(unsafe {
                        pool.submit(
                            caches_ptr.add(j) as *mut (),
                            preload_pool::preload_erased::<OffloadKVCache>,
                        )
                    });
                }

                // Layer loop
                for i in 0..num_layers {
                    if let Some(rx) = pending[i].take() {
                        if let Ok(r) = rx.recv() {
                            prefetch.record_preload(r.duration);
                        }
                    }
                    let far_idx = i + depth;
                    if far_idx < num_layers && pending[far_idx].is_none() {
                        pending[far_idx] = Some(unsafe {
                            pool.submit(
                                caches_ptr.add(far_idx) as *mut (),
                                preload_pool::preload_erased::<OffloadKVCache>,
                            )
                        });
                    }
                    let current = unsafe { &mut *caches_ptr.add(i) };
                    let fwd_t0 = std::time::Instant::now();
                    let _ = current.get_view();
                    prefetch.record_forward(fwd_t0.elapsed());
                    if i < depth {
                        unsafe { (*caches_ptr.add(i)).retain_preload() };
                    }
                    if i > 0 && (i - 1) >= depth {
                        unsafe { (*caches_ptr.add(i - 1)).release_buffers() };
                    }
                }
                for rx in pending.into_iter().flatten() {
                    let _ = rx.recv();
                }
                if num_layers >= 1 && (num_layers - 1) >= depth {
                    caches[num_layers - 1].release_buffers();
                }

                prefetch.adjust();
            }
            start.elapsed().as_secs_f64() * 1000.0
        };

        let scope_per_tok = scope_ms / decode_steps as f64;
        let pool_per_tok = pool_ms / decode_steps as f64;
        let speedup_pct = (1.0 - pool_ms / scope_ms) * 100.0;

        println!("║");
        println!("║  {:25} {:>10} {:>10}", "Strategy", "Total(ms)", "ms/tok");
        println!("╠═══════════════════════════════════════════════════════════════════════════╣");
        println!(
            "║  {:25} {:>10.1} {:>10.3}",
            "thread::scope (per-tok)", scope_ms, scope_per_tok
        );
        println!(
            "║  {:25} {:>10.1} {:>10.3}",
            "PreloadPool (persistent)", pool_ms, pool_per_tok
        );
        println!("╠═══════════════════════════════════════════════════════════════════════════╣");
        println!("║  Pool speedup: {speedup_pct:+.1}% vs thread::scope");
        println!("╚═══════════════════════════════════════════════════════════════════════════╝\n");

        // Pool should not be significantly slower than scope
        assert!(
            pool_ms < scope_ms * 1.2,
            "pool should not be >20% slower: pool={pool_ms:.1}ms, scope={scope_ms:.1}ms"
        );
    }

    #[test]
    fn test_retain_preload_cross_token() {
        // Verify cross-token retention: retain_preload() after get_view()
        // allows next token's update() to dual-write, and preload() to skip.
        let kv_heads = 1;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4; // F32
        let store = zram_store::ZramStore::new(token_bytes, 4, 32);
        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        // Token 0: prefill
        let k0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let v0: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        cache
            .update(
                &make_f32_tensor_with_data(&k0, 1, kv_heads, head_dim),
                &make_f32_tensor_with_data(&v0, 1, kv_heads, head_dim),
            )
            .unwrap();

        // Preload (first time, loads from store)
        cache.preload().unwrap();
        assert!(cache.preloaded);

        // get_view resets preloaded
        let _ = cache.get_view();
        assert!(!cache.preloaded);

        // retain_preload: re-arms preloaded since buffers still exist
        cache.retain_preload();
        assert!(cache.preloaded);
        assert!(cache.attn_k_buf.is_some());
        assert!(cache.attn_v_buf.is_some());

        // Token 1: update with preloaded=true → dual-write
        let k1: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let v1: Vec<f32> = vec![50.0, 60.0, 70.0, 80.0];
        cache
            .update(
                &make_f32_tensor_with_data(&k1, 1, kv_heads, head_dim),
                &make_f32_tensor_with_data(&v1, 1, kv_heads, head_dim),
            )
            .unwrap();
        assert_eq!(cache.current_pos(), 2);

        // preload() should be a no-op (already preloaded)
        cache.preload().unwrap();
        assert!(cache.preloaded);

        // get_view should return both tokens correctly
        let (k_view, v_view) = cache.get_view();
        let k_data = k_view.as_slice::<f32>();
        let v_data = v_view.as_slice::<f32>();
        assert_eq!(&k_data[..4], &k0);
        assert_eq!(&k_data[4..8], &k1);
        assert_eq!(&v_data[..4], &v0);
        assert_eq!(&v_data[4..8], &v1);
    }

    #[test]
    fn test_retain_preload_depth_decrease() {
        // When depth decreases, previously retained layers should be released normally.
        let kv_heads = 1;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4;
        let store = zram_store::ZramStore::new(token_bytes, 4, 32);
        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        // Setup: prefill + preload + get_view + retain
        let k0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let v0: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        cache
            .update(
                &make_f32_tensor_with_data(&k0, 1, kv_heads, head_dim),
                &make_f32_tensor_with_data(&v0, 1, kv_heads, head_dim),
            )
            .unwrap();
        cache.preload().unwrap();
        let _ = cache.get_view();
        cache.retain_preload();
        assert!(cache.preloaded);

        // Simulate depth decrease: release_buffers clears retained state
        cache.release_buffers();
        assert!(!cache.preloaded);
        assert!(cache.attn_k_buf.is_none());
        assert!(cache.attn_v_buf.is_none());

        // preload() should load from store again
        cache.preload().unwrap();
        assert!(cache.preloaded);
        let (k_view, _) = cache.get_view();
        assert_eq!(k_view.as_slice::<f32>(), &k0);
    }

    #[test]
    fn test_retain_preload_guards_none_bufs() {
        // retain_preload() with no buffers should NOT set preloaded=true.
        let kv_heads = 1;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4;
        let store = zram_store::ZramStore::new(token_bytes, 4, 32);
        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        // No data, no buffers
        assert!(!cache.preloaded);
        assert!(cache.attn_k_buf.is_none());
        cache.retain_preload();
        assert!(
            !cache.preloaded,
            "retain_preload should not arm without buffers"
        );

        // After release_buffers, retain should also be safe
        let k: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let v: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        cache
            .update(
                &make_f32_tensor_with_data(&k, 1, kv_heads, head_dim),
                &make_f32_tensor_with_data(&v, 1, kv_heads, head_dim),
            )
            .unwrap();
        cache.preload().unwrap();
        cache.release_buffers();
        cache.retain_preload();
        assert!(
            !cache.preloaded,
            "retain_preload after release should not arm"
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Deferred Store Write Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_deferred_write_skips_store() {
        // When preloaded, decode update should NOT write to store (store_behind increases).
        let kv_heads = 1;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4;
        let store = zram_store::ZramStore::new(token_bytes, 4, 32);
        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        // Prefill 1 token
        let k0 = make_f32_tensor_with_data(&[1.0, 2.0, 3.0, 4.0], 1, kv_heads, head_dim);
        let v0 = make_f32_tensor_with_data(&[5.0, 6.0, 7.0, 8.0], 1, kv_heads, head_dim);
        cache.update(&k0, &v0).unwrap();
        assert_eq!(cache.store.stored_tokens(), 1);
        assert_eq!(cache.store_behind, 0);

        // Preload + retain
        cache.preload().unwrap();

        // Decode while preloaded: should defer store write
        let k1 = make_f32_tensor_with_data(&[10.0, 20.0, 30.0, 40.0], 1, kv_heads, head_dim);
        let v1 = make_f32_tensor_with_data(&[50.0, 60.0, 70.0, 80.0], 1, kv_heads, head_dim);
        cache.update(&k1, &v1).unwrap();

        // Store should NOT have the new token yet
        assert_eq!(
            cache.store.stored_tokens(),
            1,
            "store should still have 1 token"
        );
        assert_eq!(cache.store_behind, 1, "store_behind should be 1");
        assert_eq!(cache.current_pos(), 2);
    }

    #[test]
    fn test_deferred_flush_on_release() {
        // release_buffers() should flush deferred tokens to store.
        let kv_heads = 1;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4;
        let store = zram_store::ZramStore::new(token_bytes, 4, 32);
        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        // Prefill + preload
        let k0 = make_f32_tensor_with_data(&[1.0, 2.0, 3.0, 4.0], 1, kv_heads, head_dim);
        let v0 = make_f32_tensor_with_data(&[5.0, 6.0, 7.0, 8.0], 1, kv_heads, head_dim);
        cache.update(&k0, &v0).unwrap();
        cache.preload().unwrap();

        // 3 deferred decode tokens
        for i in 0..3 {
            let val = 10.0 * (i + 1) as f32;
            let k = make_f32_tensor_with_data(
                &[val, val + 1.0, val + 2.0, val + 3.0],
                1,
                kv_heads,
                head_dim,
            );
            let v = make_f32_tensor_with_data(
                &[val + 10.0, val + 11.0, val + 12.0, val + 13.0],
                1,
                kv_heads,
                head_dim,
            );
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.store_behind, 3);
        assert_eq!(cache.store.stored_tokens(), 1);

        // Release: should flush all deferred tokens
        cache.release_buffers();
        assert_eq!(cache.store_behind, 0);
        assert_eq!(cache.store.stored_tokens(), 4); // 1 original + 3 deferred

        // Verify data integrity: reload and check
        cache.preload().unwrap();
        let (k_view, v_view) = cache.get_view();
        let k_data = k_view.as_slice::<f32>();
        let v_data = v_view.as_slice::<f32>();
        assert_eq!(&k_data[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&k_data[4..8], &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(&v_data[..4], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&v_data[4..8], &[20.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn test_deferred_preload_after_behind() {
        // preload() with store_behind > 0 should flush then reload from store.
        let kv_heads = 1;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4;
        let store = zram_store::ZramStore::new(token_bytes, 4, 32);
        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        // Prefill + preload
        let k0 = make_f32_tensor_with_data(&[1.0, 2.0, 3.0, 4.0], 1, kv_heads, head_dim);
        let v0 = make_f32_tensor_with_data(&[5.0, 6.0, 7.0, 8.0], 1, kv_heads, head_dim);
        cache.update(&k0, &v0).unwrap();
        cache.preload().unwrap();

        // Deferred token
        let k1 = make_f32_tensor_with_data(&[10.0, 20.0, 30.0, 40.0], 1, kv_heads, head_dim);
        let v1 = make_f32_tensor_with_data(&[50.0, 60.0, 70.0, 80.0], 1, kv_heads, head_dim);
        cache.update(&k1, &v1).unwrap();
        assert_eq!(cache.store_behind, 1);

        // Reset preloaded (simulates token boundary)
        cache.preloaded = false;

        // preload() should flush then reload
        cache.preload().unwrap();
        assert!(cache.preloaded);
        assert_eq!(cache.store_behind, 0);
        assert_eq!(cache.store.stored_tokens(), 2);

        // Verify data
        let (k_view, v_view) = cache.get_view();
        let k_data = k_view.as_slice::<f32>();
        let v_data = v_view.as_slice::<f32>();
        assert_eq!(&k_data[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&k_data[4..8], &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(&v_data[..4], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&v_data[4..8], &[50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn test_deferred_get_view_with_store_behind() {
        // get_view() when !preloaded && store_behind > 0: attn_buf is valid, just flush.
        let kv_heads = 1;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4;
        let store = zram_store::ZramStore::new(token_bytes, 4, 32);
        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        // Prefill + preload
        let k0 = make_f32_tensor_with_data(&[1.0, 2.0, 3.0, 4.0], 1, kv_heads, head_dim);
        let v0 = make_f32_tensor_with_data(&[5.0, 6.0, 7.0, 8.0], 1, kv_heads, head_dim);
        cache.update(&k0, &v0).unwrap();
        cache.preload().unwrap();

        // Deferred token
        let k1 = make_f32_tensor_with_data(&[10.0, 20.0, 30.0, 40.0], 1, kv_heads, head_dim);
        let v1 = make_f32_tensor_with_data(&[50.0, 60.0, 70.0, 80.0], 1, kv_heads, head_dim);
        cache.update(&k1, &v1).unwrap();

        // get_view consumes preloaded (sets to false), but data is in attn_buf
        let (k_view, v_view) = cache.get_view();
        let k_data = k_view.as_slice::<f32>();
        assert_eq!(&k_data[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&k_data[4..8], &[10.0, 20.0, 30.0, 40.0]);
        let v_data = v_view.as_slice::<f32>();
        assert_eq!(&v_data[..4], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&v_data[4..8], &[50.0, 60.0, 70.0, 80.0]);

        // Now preloaded=false, store_behind=1 (get_view with preloaded=true doesn't flush).
        // retain_preload re-arms preloaded. Next deferred update increments store_behind to 2.
        cache.retain_preload(); // re-arm after get_view

        let k2 = make_f32_tensor_with_data(&[100.0, 200.0, 300.0, 400.0], 1, kv_heads, head_dim);
        let v2 = make_f32_tensor_with_data(&[500.0, 600.0, 700.0, 800.0], 1, kv_heads, head_dim);
        cache.update(&k2, &v2).unwrap();
        assert_eq!(cache.store_behind, 2); // 1 from first deferred + 1 from second

        // Manually set preloaded=false to trigger the store_behind path in get_view
        cache.preloaded = false;

        let (k_view2, _) = cache.get_view();
        let k_data2 = k_view2.as_slice::<f32>();
        assert_eq!(k_data2.len(), 3 * 4); // 3 tokens × 4 floats
        assert_eq!(&k_data2[8..12], &[100.0, 200.0, 300.0, 400.0]);
    }

    #[test]
    fn test_deferred_write_with_raw_store() {
        // Verify deferred write works with RawStore too.
        let kv_heads = 1;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4;
        let store = raw_store::RawStore::new(token_bytes);
        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        // Prefill
        let k0 = make_f32_tensor_with_data(&[1.0, 2.0, 3.0, 4.0], 1, kv_heads, head_dim);
        let v0 = make_f32_tensor_with_data(&[5.0, 6.0, 7.0, 8.0], 1, kv_heads, head_dim);
        cache.update(&k0, &v0).unwrap();
        cache.preload().unwrap();

        // 5 deferred tokens
        for i in 1..=5 {
            let val = i as f32 * 10.0;
            let k = make_f32_tensor_with_data(
                &[val, val + 1.0, val + 2.0, val + 3.0],
                1,
                kv_heads,
                head_dim,
            );
            let v = make_f32_tensor_with_data(
                &[val + 50.0, val + 51.0, val + 52.0, val + 53.0],
                1,
                kv_heads,
                head_dim,
            );
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.store_behind, 5);
        assert_eq!(cache.store.stored_tokens(), 1);
        assert_eq!(cache.current_pos(), 6);

        // Release flushes everything
        cache.release_buffers();
        assert_eq!(cache.store.stored_tokens(), 6);
        assert_eq!(cache.store_behind, 0);

        // Reload and verify all 6 tokens
        cache.preload().unwrap();
        let (k_view, v_view) = cache.get_view();
        let k_data = k_view.as_slice::<f32>();
        let v_data = v_view.as_slice::<f32>();
        assert_eq!(k_data.len(), 6 * 4);
        assert_eq!(&k_data[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&k_data[4..8], &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(&v_data[..4], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&v_data[4..8], &[60.0, 61.0, 62.0, 63.0]);
    }

    #[test]
    fn test_non_retained_update_writes_store_immediately() {
        // Without preload, update should write to store directly (store_behind stays 0).
        let kv_heads = 1;
        let head_dim = 4;
        let token_bytes = kv_heads * head_dim * 4;
        let store = raw_store::RawStore::new(token_bytes);
        let mut cache = OffloadKVCache::new(0, kv_heads, head_dim, DType::F32, 64, Box::new(store));

        // Decode without preload
        for i in 0..5 {
            let val = i as f32;
            let k = make_f32_tensor_with_data(&[val, val, val, val], 1, kv_heads, head_dim);
            let v = make_f32_tensor_with_data(&[val, val, val, val], 1, kv_heads, head_dim);
            cache.update(&k, &v).unwrap();
            assert_eq!(cache.store.stored_tokens(), i + 1);
            assert_eq!(cache.store_behind, 0);
        }
    }
}
