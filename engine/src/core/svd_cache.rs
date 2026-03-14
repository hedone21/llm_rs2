//! SVD-compressed K cache + V offload KV cache.
//!
//! Combines two strategies:
//! - **K**: SVD lossy compression (in-memory, per-head basis + coefficients)
//! - **V**: Lossless offload via existing `OffloadStore` (ZramStore/DiskStore)
//!
//! Design based on compress_lab findings: K is low-rank (CosSim 0.93 at k=10),
//! V is high-rank (CosSim 0.73) → compress K lossy, offload V lossless.

use crate::backend::cpu::CpuBackend;
use crate::buffer::shared_buffer::SharedBuffer;
use crate::core::buffer::{Buffer, DType};
use crate::core::kv_cache::{KVCacheOps, KVLayout};
use crate::core::offload::store::OffloadStore;
use crate::core::shape::Shape;
use crate::core::svd_math;
use crate::core::tensor::Tensor;
use anyhow::Result;
use std::sync::Arc;

/// Configuration for SVD K-cache compression.
pub struct SvdConfig {
    /// SVD rank (number of basis vectors per head). Default 10.
    pub rank_k: usize,
}

impl Default for SvdConfig {
    fn default() -> Self {
        Self { rank_k: 10 }
    }
}

/// Per-head SVD compression state.
struct SvdHeadState {
    /// Basis vectors: flattened `[k × head_dim]` (f32).
    basis: Vec<f32>,
    /// Projection coefficients: flattened `[compressed_tokens × k]` (f32).
    coeffs: Vec<f32>,
    /// Number of tokens with computed coefficients.
    compressed_tokens: usize,
    /// Actual rank achieved (may be < config.rank_k if matrix rank is low).
    actual_k: usize,
}

/// Adapter that wraps an `OffloadStore` for V-only storage.
///
/// Since `OffloadStore::store()` and `append_token()` expect both K and V data,
/// this adapter passes dummy K data (zeros) and only stores real V data.
struct VStoreAdapter {
    inner: Box<dyn OffloadStore>,
    /// Dummy K token buffer (token_bytes zeros).
    dummy_k_token: Vec<u8>,
}

impl VStoreAdapter {
    fn new(store: Box<dyn OffloadStore>, token_bytes: usize) -> Self {
        Self {
            inner: store,
            dummy_k_token: vec![0u8; token_bytes],
        }
    }

    fn store_v(&mut self, v_data: &[u8], num_tokens: usize) -> Result<()> {
        let token_bytes = self.dummy_k_token.len();
        let dummy_k = vec![0u8; num_tokens * token_bytes];
        self.inner.store(&dummy_k, v_data, num_tokens)
    }

    fn append_v_token(&mut self, v_token: &[u8]) -> Result<()> {
        self.inner.append_token(&self.dummy_k_token, v_token)
    }

    fn load_v_into(&self, v_buf: &mut [u8]) -> Result<usize> {
        let mut dummy_k = vec![0u8; v_buf.len()];
        self.inner.load_into(&mut dummy_k, v_buf)
    }

    fn storage_size(&self) -> usize {
        self.inner.storage_size()
    }

    #[allow(dead_code)]
    fn stored_tokens(&self) -> usize {
        self.inner.stored_tokens()
    }
}

/// SVD-compressed K cache + V offload KV cache.
///
/// K data is stored as per-head SVD basis + coefficients (lossy, in-memory).
/// V data is stored losslessly via an `OffloadStore` (ZramStore or DiskStore).
///
/// SeqMajor layout only. No eviction support.
pub struct SvdOffloadKVCache {
    layer_id: usize,
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    current_pos: usize,
    config: SvdConfig,

    // K: SVD compression state (per-head)
    k_heads: Vec<SvdHeadState>,
    // K reconstruction buffer (f32, incremental)
    attn_k_buf: Vec<f32>, // [max_seq × kv_heads × head_dim]
    k_reconstructed_tokens: usize,

    // V: offload storage
    v_store: VStoreAdapter,
    attn_v_buf: Option<Vec<u8>>, // lazy, allocated on preload()/get_view()
    v_preloaded: bool,
    v_token_bytes: usize, // kv_heads * head_dim * sizeof(F16)

    // Prefill raw K buffer for SVD computation
    // Stored temporarily during prefill, cleared after basis computation
    prefill_k_f32: Option<Vec<f32>>, // [prefill_tokens × kv_heads × head_dim]
    prefill_tokens: usize,

    // Output buffer reuse (OffloadKVCache pattern)
    out_k_buf: Option<Arc<SharedBuffer>>,
    out_v_buf: Option<Arc<SharedBuffer>>,
    out_backend: Arc<CpuBackend>,
}

impl SvdOffloadKVCache {
    /// Create a new SvdOffloadKVCache.
    ///
    /// - `layer_id`: transformer layer index
    /// - `kv_heads`: number of KV heads
    /// - `head_dim`: dimension per head
    /// - `max_seq_len`: maximum sequence length
    /// - `config`: SVD compression configuration
    /// - `v_store`: OffloadStore for V data (ZramStore or DiskStore)
    pub fn new(
        layer_id: usize,
        kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        config: SvdConfig,
        v_store: Box<dyn OffloadStore>,
    ) -> Self {
        let v_token_bytes = kv_heads * head_dim * DType::F16.size(); // V stored as F16
        let attn_k_size = max_seq_len * kv_heads * head_dim;
        let rank_k = config.rank_k;

        // Initialize per-head states with empty basis
        let k_heads = (0..kv_heads)
            .map(|_| SvdHeadState {
                basis: Vec::new(),
                coeffs: Vec::with_capacity(max_seq_len * rank_k),
                compressed_tokens: 0,
                actual_k: 0,
            })
            .collect();

        Self {
            layer_id,
            kv_heads,
            head_dim,
            max_seq_len,
            current_pos: 0,
            config,
            k_heads,
            attn_k_buf: vec![0.0f32; attn_k_size],
            k_reconstructed_tokens: 0,
            v_store: VStoreAdapter::new(v_store, v_token_bytes),
            attn_v_buf: None,
            v_preloaded: false,
            v_token_bytes,
            prefill_k_f32: None,
            prefill_tokens: 0,
            out_k_buf: None,
            out_v_buf: None,
            out_backend: Arc::new(CpuBackend::new()),
        }
    }

    /// Compute SVD basis from prefill K data and project all prefill tokens.
    fn compute_basis_from_prefill(&mut self) {
        let prefill_data = match self.prefill_k_f32.take() {
            Some(d) => d,
            None => return,
        };
        let num_tokens = self.prefill_tokens;
        if num_tokens == 0 {
            return;
        }

        let rank_k = self.config.rank_k;

        for h in 0..self.kv_heads {
            // Extract this head's data: [num_tokens × head_dim]
            let mut head_data = vec![0.0f32; num_tokens * self.head_dim];
            for t in 0..num_tokens {
                let src = t * self.kv_heads * self.head_dim + h * self.head_dim;
                let dst = t * self.head_dim;
                head_data[dst..dst + self.head_dim]
                    .copy_from_slice(&prefill_data[src..src + self.head_dim]);
            }

            // Compute Gram matrix A^T A
            let ata = svd_math::compute_gram_matrix(&head_data, num_tokens, self.head_dim);

            // Eigendecomposition → top-k basis vectors
            let (_, eigvecs) = svd_math::svd_eigen_f32(&ata, self.head_dim, rank_k);
            let actual_k = eigvecs.len() / self.head_dim;

            if actual_k == 0 {
                // Degenerate case: no significant components
                self.k_heads[h].basis = vec![0.0; rank_k * self.head_dim];
                self.k_heads[h].actual_k = 0;
                self.k_heads[h].coeffs.resize(num_tokens * rank_k, 0.0);
                self.k_heads[h].compressed_tokens = num_tokens;
                continue;
            }

            // Store basis (pad to rank_k if fewer eigenvectors found)
            let mut basis = vec![0.0f32; rank_k * self.head_dim];
            basis[..actual_k * self.head_dim].copy_from_slice(&eigvecs[..actual_k * self.head_dim]);
            self.k_heads[h].basis = basis;
            self.k_heads[h].actual_k = actual_k;

            // Project all prefill tokens
            for t in 0..num_tokens {
                let token = &head_data[t * self.head_dim..(t + 1) * self.head_dim];
                let coeffs =
                    svd_math::project_token(token, &self.k_heads[h].basis, rank_k, self.head_dim);
                self.k_heads[h].coeffs.extend_from_slice(&coeffs);
            }
            self.k_heads[h].compressed_tokens = num_tokens;
        }

        self.prefill_tokens = 0;
    }

    /// Project a single decode token's K data onto the SVD basis (per-head).
    fn project_decode_token(&mut self, k_f32: &[f32]) {
        let rank_k = self.config.rank_k;
        for h in 0..self.kv_heads {
            let token = &k_f32[h * self.head_dim..(h + 1) * self.head_dim];
            let coeffs =
                svd_math::project_token(token, &self.k_heads[h].basis, rank_k, self.head_dim);
            self.k_heads[h].coeffs.extend_from_slice(&coeffs);
            self.k_heads[h].compressed_tokens += 1;
        }
    }

    /// Incrementally reconstruct K data into attn_k_buf.
    /// Only reconstructs tokens from k_reconstructed_tokens to current_pos.
    fn reconstruct_k_incremental(&mut self) {
        let total = self.current_pos;
        if total <= self.k_reconstructed_tokens {
            return;
        }
        let start = self.k_reconstructed_tokens;
        let count = total - start;
        let rank_k = self.config.rank_k;

        for h in 0..self.kv_heads {
            let head = &self.k_heads[h];
            if head.actual_k == 0 {
                // Zero fill for degenerate heads
                for t in start..total {
                    let pos = t * self.kv_heads * self.head_dim + h * self.head_dim;
                    for d in 0..self.head_dim {
                        self.attn_k_buf[pos + d] = 0.0;
                    }
                }
                continue;
            }
            svd_math::reconstruct_into(
                &head.basis[..rank_k * self.head_dim],
                &head.coeffs,
                start,
                count,
                rank_k,
                self.head_dim,
                self.kv_heads,
                h,
                &mut self.attn_k_buf,
            );
        }
        self.k_reconstructed_tokens = total;
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

    /// Pre-load V data from store into attn_v_buf.
    pub fn preload(&mut self) -> Result<()> {
        let total_bytes = self.current_pos * self.v_token_bytes;
        if total_bytes == 0 {
            self.v_preloaded = true;
            return Ok(());
        }

        let max_bytes = self.max_seq_len * self.v_token_bytes;
        let v_buf = self.attn_v_buf.get_or_insert_with(|| vec![0u8; max_bytes]);
        self.v_store.load_v_into(&mut v_buf[..total_bytes])?;
        self.v_preloaded = true;
        Ok(())
    }

    /// Release V attn buffer to free memory (only 2 layers active at once).
    pub fn release_buffers(&mut self) {
        self.attn_v_buf = None;
        self.v_preloaded = false;
    }

    /// Reset preloaded flag at token boundary.
    pub fn reset_preload(&mut self) {
        self.v_preloaded = false;
    }
}

impl KVCacheOps for SvdOffloadKVCache {
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
        // Caller passes F16 data (same as OffloadKVCache F16 mode)
        DType::F16
    }

    fn memory_usage_bytes(&self) -> usize {
        // K: basis + coeffs (f32)
        let k_bytes: usize = self
            .k_heads
            .iter()
            .map(|h| h.basis.len() * 4 + h.coeffs.len() * 4)
            .sum();
        // V: offload store size
        let v_bytes = self.v_store.storage_size();
        k_bytes + v_bytes
    }

    fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        let seq_len = new_k.shape().dims()[1];
        let total_after = self.current_pos + seq_len;
        if total_after > self.max_seq_len {
            return Err(anyhow::anyhow!(
                "SvdOffloadKVCache[{}] overflow: {} + {} > {}",
                self.layer_id,
                self.current_pos,
                seq_len,
                self.max_seq_len
            ));
        }

        // Read raw F16 bytes
        let k_ptr = new_k.buffer().as_ptr();
        let v_ptr = new_v.buffer().as_ptr();
        let k_bytes = seq_len * self.kv_heads * self.head_dim * DType::F16.size();
        let v_bytes = seq_len * self.v_token_bytes;

        let k_data = unsafe { std::slice::from_raw_parts(k_ptr, k_bytes) };
        let v_data = unsafe { std::slice::from_raw_parts(v_ptr, v_bytes) };

        // Convert K from F16 to f32
        let k_elems = seq_len * self.kv_heads * self.head_dim;
        let mut k_f32 = vec![0.0f32; k_elems];
        let k_f16 =
            unsafe { std::slice::from_raw_parts(k_data.as_ptr() as *const half::f16, k_elems) };
        for (i, &f16_val) in k_f16.iter().enumerate() {
            k_f32[i] = f16_val.to_f32();
        }

        if seq_len > 1 {
            // Prefill path: accumulate K data for batch SVD computation
            if self.current_pos == 0 {
                // First prefill: store K f32 data, store V to offload
                self.prefill_k_f32 = Some(k_f32);
                self.prefill_tokens = seq_len;
                self.v_store.store_v(v_data, seq_len)?;
            } else {
                // Multi-batch prefill: append to existing
                if let Some(ref mut buf) = self.prefill_k_f32 {
                    buf.extend_from_slice(&k_f32);
                    self.prefill_tokens += seq_len;
                }
                for s in 0..seq_len {
                    let offset = s * self.v_token_bytes;
                    self.v_store
                        .append_v_token(&v_data[offset..offset + self.v_token_bytes])?;
                }
            }

            self.current_pos = total_after;

            // Compute SVD basis after all prefill tokens are collected
            // (called at the end of prefill, before decode starts)
            self.compute_basis_from_prefill();
        } else {
            // Decode path: single token
            // If basis not yet computed (edge case: decode without prefill),
            // initialize with identity-like basis
            if self.k_heads[0].basis.is_empty() {
                let rank_k = self.config.rank_k;
                for h in 0..self.kv_heads {
                    let mut basis = vec![0.0f32; rank_k * self.head_dim];
                    let actual_k = rank_k.min(self.head_dim);
                    for j in 0..actual_k {
                        basis[j * self.head_dim + j] = 1.0;
                    }
                    self.k_heads[h].basis = basis;
                    self.k_heads[h].actual_k = actual_k;
                }
            }

            // Project K onto basis
            self.project_decode_token(&k_f32);

            // Store V
            self.v_store.append_v_token(v_data)?;

            // If preloaded, also append V to attn buffer
            if self.v_preloaded
                && let Some(ref mut v_buf) = self.attn_v_buf
            {
                let offset = self.current_pos * self.v_token_bytes;
                v_buf[offset..offset + self.v_token_bytes].copy_from_slice(v_data);
            }

            self.current_pos = total_after;
        }

        Ok(())
    }

    fn get_view(&mut self) -> (Tensor, Tensor) {
        let total = self.current_pos;
        let backend: Arc<dyn crate::core::backend::Backend> = self.out_backend.clone();

        if total == 0 {
            let buf = Arc::new(SharedBuffer::new(0, DType::F16));
            let shape = Shape::new(vec![1, 0, self.kv_heads, self.head_dim]);
            let t = Tensor::new(shape.clone(), buf.clone(), backend.clone());
            return (t.clone(), t);
        }

        // K: incremental reconstruction (f32 → F16)
        self.reconstruct_k_incremental();

        let k_elems = total * self.kv_heads * self.head_dim;
        let k_f16_bytes = k_elems * DType::F16.size();
        let k_buf = Self::reuse_or_alloc_out_buf(&mut self.out_k_buf, k_f16_bytes, DType::F16);
        // Convert f32 → F16
        unsafe {
            let k_f16_ptr = k_buf.as_mut_ptr() as *mut half::f16;
            for i in 0..k_elems {
                *k_f16_ptr.add(i) = half::f16::from_f32(self.attn_k_buf[i]);
            }
        }

        // V: load from store
        let v_total_bytes = total * self.v_token_bytes;
        let max_v_bytes = self.max_seq_len * self.v_token_bytes;
        let v_attn = self
            .attn_v_buf
            .get_or_insert_with(|| vec![0u8; max_v_bytes]);

        if !self.v_preloaded
            && let Err(e) = self.v_store.load_v_into(&mut v_attn[..v_total_bytes])
        {
            log::error!("SvdOffloadKVCache[{}] V load failed: {}", self.layer_id, e);
            let buf = Arc::new(SharedBuffer::new(0, DType::F16));
            let shape = Shape::new(vec![1, 0, self.kv_heads, self.head_dim]);
            let t = Tensor::new(shape.clone(), buf.clone(), backend.clone());
            return (t.clone(), t);
        }
        self.v_preloaded = false;

        let v_buf = Self::reuse_or_alloc_out_buf(&mut self.out_v_buf, v_total_bytes, DType::F16);
        unsafe {
            std::ptr::copy_nonoverlapping(v_attn.as_ptr(), v_buf.as_mut_ptr(), v_total_bytes);
        }

        let shape = Shape::new(vec![1, total, self.kv_heads, self.head_dim]);
        let k_tensor = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v_tensor = Tensor::new(shape, v_buf, backend);

        (k_tensor, v_tensor)
    }
}

impl crate::core::kv_cache::PrefetchableCache for SvdOffloadKVCache {
    fn preload(&mut self) -> Result<()> {
        self.preload()
    }

    fn release_buffers(&mut self) {
        self.release_buffers();
    }

    fn reset_preload(&mut self) {
        self.reset_preload();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::shared_buffer::SharedBuffer;
    use crate::core::offload::zram_store::ZramStore;

    fn make_f16_input(seq_len: usize, kv_heads: usize, head_dim: usize, base: f32) -> Tensor {
        let n = seq_len * kv_heads * head_dim;
        let buf = Arc::new(SharedBuffer::new(n * 2, DType::F16));
        unsafe {
            let ptr = buf.as_mut_ptr() as *mut half::f16;
            for i in 0..n {
                *ptr.add(i) = half::f16::from_f32(base + (i as f32) * 0.01);
            }
        }
        let backend: Arc<dyn crate::core::backend::Backend> = Arc::new(CpuBackend::new());
        Tensor::new(
            Shape::new(vec![1, seq_len, kv_heads, head_dim]),
            buf,
            backend,
        )
    }

    fn create_test_cache(
        kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        rank_k: usize,
    ) -> SvdOffloadKVCache {
        let token_bytes = kv_heads * head_dim * 2; // F16
        let store = Box::new(ZramStore::new(token_bytes, 2, 64));
        SvdOffloadKVCache::new(0, kv_heads, head_dim, max_seq, SvdConfig { rank_k }, store)
    }

    #[test]
    fn test_svd_cache_basic() {
        let cache = create_test_cache(2, 64, 256, 10);
        assert_eq!(cache.current_pos(), 0);
        assert_eq!(cache.kv_dtype(), DType::F16);
        assert_eq!(cache.layout(), KVLayout::SeqMajor);
        assert_eq!(cache.capacity(), 256);
    }

    #[test]
    fn test_svd_cache_prefill_shape() {
        let kv_heads = 2;
        let head_dim = 64;
        let mut cache = create_test_cache(kv_heads, head_dim, 256, 10);

        // Prefill 16 tokens
        let k = make_f16_input(16, kv_heads, head_dim, 0.1);
        let v = make_f16_input(16, kv_heads, head_dim, 1.0);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.current_pos(), 16);

        let (k_view, v_view) = cache.get_view();
        assert_eq!(k_view.shape().dims(), &[1, 16, kv_heads, head_dim]);
        assert_eq!(v_view.shape().dims(), &[1, 16, kv_heads, head_dim]);
    }

    #[test]
    fn test_svd_cache_prefill_then_decode() {
        let kv_heads = 2;
        let head_dim = 32;
        let mut cache = create_test_cache(kv_heads, head_dim, 256, 8);

        // Prefill 32 tokens
        let k = make_f16_input(32, kv_heads, head_dim, 0.5);
        let v = make_f16_input(32, kv_heads, head_dim, 2.0);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.current_pos(), 32);

        // Decode 10 tokens
        for i in 0..10 {
            let k = make_f16_input(1, kv_heads, head_dim, 0.5 + i as f32 * 0.1);
            let v = make_f16_input(1, kv_heads, head_dim, 2.0 + i as f32 * 0.1);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos(), 42);

        let (k_view, v_view) = cache.get_view();
        assert_eq!(k_view.shape().dims(), &[1, 42, kv_heads, head_dim]);
        assert_eq!(v_view.shape().dims(), &[1, 42, kv_heads, head_dim]);
    }

    #[test]
    fn test_svd_cache_k_cosine_similarity() {
        let kv_heads = 1;
        let head_dim = 64;
        let prefill_len = 64;
        let decode_len = 36;
        let rank_k = 10;

        let mut cache = create_test_cache(kv_heads, head_dim, 256, rank_k);

        // Create structured data (low-rank for good SVD compression)
        let total = prefill_len + decode_len;
        let mut original_k_f32 = Vec::with_capacity(total * kv_heads * head_dim);

        // Prefill
        {
            let n = prefill_len * kv_heads * head_dim;
            let buf = Arc::new(SharedBuffer::new(n * 2, DType::F16));
            let mut k_f32_vals = Vec::with_capacity(n);
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut half::f16;
                for t in 0..prefill_len {
                    for h in 0..kv_heads {
                        for d in 0..head_dim {
                            // Low-rank pattern: primarily uses first few dimensions
                            let val = ((t as f32 + 1.0) * (d as f32 + 1.0) * 0.001).sin() * 0.5;
                            let f16_val = half::f16::from_f32(val);
                            let idx = t * kv_heads * head_dim + h * head_dim + d;
                            *ptr.add(idx) = f16_val;
                            k_f32_vals.push(f16_val.to_f32());
                        }
                    }
                }
            }
            original_k_f32.extend_from_slice(&k_f32_vals);
            let backend: Arc<dyn crate::core::backend::Backend> = Arc::new(CpuBackend::new());
            let k = Tensor::new(
                Shape::new(vec![1, prefill_len, kv_heads, head_dim]),
                buf,
                backend.clone(),
            );
            let v = make_f16_input(prefill_len, kv_heads, head_dim, 1.0);
            cache.update(&k, &v).unwrap();
        }

        // Decode
        for t_idx in 0..decode_len {
            let n = kv_heads * head_dim;
            let buf = Arc::new(SharedBuffer::new(n * 2, DType::F16));
            let mut k_f32_vals = Vec::with_capacity(n);
            let t = prefill_len + t_idx;
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut half::f16;
                for h in 0..kv_heads {
                    for d in 0..head_dim {
                        let val = ((t as f32 + 1.0) * (d as f32 + 1.0) * 0.001).sin() * 0.5;
                        let f16_val = half::f16::from_f32(val);
                        *ptr.add(h * head_dim + d) = f16_val;
                        k_f32_vals.push(f16_val.to_f32());
                    }
                }
            }
            original_k_f32.extend_from_slice(&k_f32_vals);
            let backend: Arc<dyn crate::core::backend::Backend> = Arc::new(CpuBackend::new());
            let k = Tensor::new(
                Shape::new(vec![1, 1, kv_heads, head_dim]),
                buf,
                backend.clone(),
            );
            let v = make_f16_input(1, kv_heads, head_dim, 1.0);
            cache.update(&k, &v).unwrap();
        }

        // Get reconstructed K
        let (k_view, _) = cache.get_view();
        let k_f16_data = unsafe {
            std::slice::from_raw_parts(
                k_view.buffer().as_ptr() as *const half::f16,
                total * kv_heads * head_dim,
            )
        };
        let reconstructed: Vec<f32> = k_f16_data.iter().map(|v| v.to_f32()).collect();

        // Compute cosine similarity
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for i in 0..original_k_f32.len() {
            let a = original_k_f32[i] as f64;
            let b = reconstructed[i] as f64;
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        let cos_sim = dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-15);
        assert!(
            cos_sim > 0.9,
            "K cosine similarity = {cos_sim:.4}, expected > 0.9"
        );
    }

    #[test]
    fn test_svd_cache_v_lossless() {
        let kv_heads = 2;
        let head_dim = 32;
        let mut cache = create_test_cache(kv_heads, head_dim, 256, 8);

        // Prefill 8 tokens with known V data
        let v_input = make_f16_input(8, kv_heads, head_dim, 3.14);
        let k_input = make_f16_input(8, kv_heads, head_dim, 0.1);
        let v_original: Vec<u8> = {
            let n = 8 * kv_heads * head_dim * 2;
            let ptr = v_input.buffer().as_ptr();
            unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
        };
        cache.update(&k_input, &v_input).unwrap();

        // Get view and compare V
        let (_, v_view) = cache.get_view();
        let v_out_bytes = {
            let n = 8 * kv_heads * head_dim * 2;
            let ptr = v_view.buffer().as_ptr();
            unsafe { std::slice::from_raw_parts(ptr, n) }
        };

        // V should be lossless (exact byte match)
        assert_eq!(v_original, v_out_bytes, "V data should be lossless");
    }

    #[test]
    fn test_svd_cache_memory_usage() {
        let kv_heads = 8;
        let head_dim = 64;
        let rank_k = 10;
        let mut cache = create_test_cache(kv_heads, head_dim, 512, rank_k);

        assert_eq!(cache.memory_usage_bytes(), 0);

        // Prefill 128 tokens
        let k = make_f16_input(128, kv_heads, head_dim, 0.5);
        let v = make_f16_input(128, kv_heads, head_dim, 1.0);
        cache.update(&k, &v).unwrap();

        let mem = cache.memory_usage_bytes();
        let raw_kv = 128 * kv_heads * head_dim * 2 * 2; // K+V F16
        // SVD should use significantly less memory than raw K+V
        assert!(
            mem < raw_kv,
            "SVD memory {mem} should be less than raw {raw_kv}"
        );
    }

    #[test]
    fn test_svd_cache_overflow() {
        let kv_heads = 1;
        let head_dim = 32;
        let mut cache = create_test_cache(kv_heads, head_dim, 64, 4);

        // Prefill to max
        let k = make_f16_input(64, kv_heads, head_dim, 0.1);
        let v = make_f16_input(64, kv_heads, head_dim, 1.0);
        cache.update(&k, &v).unwrap();

        // One more should fail
        let k = make_f16_input(1, kv_heads, head_dim, 0.1);
        let v = make_f16_input(1, kv_heads, head_dim, 1.0);
        assert!(cache.update(&k, &v).is_err());
    }

    #[test]
    fn test_svd_cache_preload_release() {
        let kv_heads = 2;
        let head_dim = 32;
        let mut cache = create_test_cache(kv_heads, head_dim, 256, 8);

        let k = make_f16_input(16, kv_heads, head_dim, 0.5);
        let v = make_f16_input(16, kv_heads, head_dim, 1.0);
        cache.update(&k, &v).unwrap();

        // Preload V
        cache.preload().unwrap();
        assert!(cache.attn_v_buf.is_some());

        // Release
        cache.release_buffers();
        assert!(cache.attn_v_buf.is_none());
        assert!(!cache.v_preloaded);
    }

    #[test]
    fn test_svd_cache_incremental_reconstruction() {
        let kv_heads = 1;
        let head_dim = 32;
        let mut cache = create_test_cache(kv_heads, head_dim, 256, 8);

        // Prefill 16 tokens
        let k = make_f16_input(16, kv_heads, head_dim, 0.5);
        let v = make_f16_input(16, kv_heads, head_dim, 1.0);
        cache.update(&k, &v).unwrap();

        // First get_view: reconstruct all 16
        let (k1, _) = cache.get_view();
        assert_eq!(cache.k_reconstructed_tokens, 16);
        let k1_data: Vec<u8> = {
            let n = 16 * kv_heads * head_dim * 2;
            unsafe { std::slice::from_raw_parts(k1.buffer().as_ptr(), n) }.to_vec()
        };

        // Add 4 more tokens
        for _ in 0..4 {
            let k = make_f16_input(1, kv_heads, head_dim, 0.6);
            let v = make_f16_input(1, kv_heads, head_dim, 1.1);
            cache.update(&k, &v).unwrap();
        }

        // Second get_view: should only reconstruct tokens 16..20
        let (k2, _) = cache.get_view();
        assert_eq!(cache.k_reconstructed_tokens, 20);
        let k2_data = unsafe {
            std::slice::from_raw_parts(k2.buffer().as_ptr(), 20 * kv_heads * head_dim * 2)
        };

        // First 16 tokens should be identical
        assert_eq!(
            &k1_data[..16 * kv_heads * head_dim * 2],
            &k2_data[..16 * kv_heads * head_dim * 2],
            "Incremental reconstruction changed existing tokens"
        );
    }
}
