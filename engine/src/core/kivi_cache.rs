//! KIVI-style KV cache with asymmetric 2-bit quantization (ICML 2024).
//!
//! Key insight: recent R tokens stay in FP32 (residual buffer). When full,
//! they are batch-quantized to Q2 and flushed to compressed storage.
//!
//! - **Key cache**: per-channel quantization (groups across tokens within each channel).
//! - **Value cache**: per-token quantization (groups within one token's head_dim).
//!
//! `kv_dtype()` returns `F32` so LlamaLayer passes F32 data to `update()`.
//! `get_view()` returns dequantized F32 tensors for the existing attention path.

use crate::backend::cpu::CpuBackend;
use crate::buffer::shared_buffer::SharedBuffer;
use crate::core::buffer::{Buffer, DType};
use crate::core::kv_cache::{KVCacheOps, KVLayout};
use crate::core::quant::{BlockQ2_0, QK2_0};
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use anyhow::Result;
use std::sync::Arc;

/// KIVI KV cache: Q2 compressed storage + FP32 residual buffer.
///
/// SeqMajor layout only (Phase 1). Data layout for residual:
/// `[kv_heads, res_cap, head_dim]` (head-first for easier per-channel Key quantization).
///
/// Q2 storage layout (after flush):
/// - Key: per-channel blocks. For each head, `head_dim` channels, each channel
///   quantized across tokens in groups of QK2_0 (32).
/// - Value: per-token blocks. For each head, each token's `head_dim` values
///   quantized in groups of QK2_0 (32).
pub struct KiviCache {
    // Q2 compressed storage (raw block data)
    q2_k: Vec<BlockQ2_0>,
    q2_v: Vec<BlockQ2_0>,
    /// Number of tokens in Q2 storage (always a multiple of `group_size`).
    pub q2_tokens: usize,

    // FP32 residual buffer — layout: [kv_heads][res_cap][head_dim]
    res_k: Vec<f32>,
    res_v: Vec<f32>,
    /// Number of valid tokens in residual buffer.
    pub res_pos: usize,
    /// Residual buffer capacity (R). Must be a multiple of QK2_0 (32).
    res_cap: usize,

    // Pre-allocated output buffers for get_view() dequantization
    attn_k_buf: Vec<f32>,
    attn_v_buf: Vec<f32>,

    // Dimensions
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,

    // Group size for quantization (= QK2_0 = 32)
    group_size: usize,
}

impl KiviCache {
    /// Create a new KiviCache.
    ///
    /// - `kv_heads`: number of KV heads
    /// - `head_dim`: dimension per head (must be a multiple of QK2_0)
    /// - `max_seq_len`: maximum sequence length
    /// - `residual_size`: FP32 residual buffer size in tokens (must be a multiple of QK2_0)
    pub fn new(kv_heads: usize, head_dim: usize, max_seq_len: usize, residual_size: usize) -> Self {
        assert!(
            residual_size.is_multiple_of(QK2_0) && residual_size > 0,
            "residual_size ({residual_size}) must be a positive multiple of {QK2_0}"
        );
        assert!(
            head_dim.is_multiple_of(QK2_0),
            "head_dim ({head_dim}) must be a multiple of {QK2_0}"
        );

        let res_elems = kv_heads * residual_size * head_dim;
        // Estimate max Q2 blocks: (max_seq_len / group_size) * kv_heads * head_dim / QK2_0
        // But we allocate conservatively and grow as needed
        let max_q2_tokens = max_seq_len; // upper bound
        let blocks_per_flush_k = kv_heads * head_dim; // per-channel: head_dim channels × (group_size/QK2_0=1) blocks
        let blocks_per_flush_v = kv_heads * (head_dim / QK2_0) * (residual_size / QK2_0);
        // Total max flushes
        let max_flushes = max_q2_tokens / residual_size;
        let max_k_blocks = max_flushes * blocks_per_flush_k;
        let max_v_blocks = max_flushes * blocks_per_flush_v;

        let attn_buf_size = max_seq_len * kv_heads * head_dim;

        Self {
            q2_k: Vec::with_capacity(max_k_blocks),
            q2_v: Vec::with_capacity(max_v_blocks),
            q2_tokens: 0,
            res_k: vec![0.0; res_elems],
            res_v: vec![0.0; res_elems],
            res_pos: 0,
            res_cap: residual_size,
            attn_k_buf: vec![0.0; attn_buf_size],
            attn_v_buf: vec![0.0; attn_buf_size],
            kv_heads,
            head_dim,
            max_seq_len,
            group_size: QK2_0,
        }
    }

    /// Total number of valid tokens (Q2 + residual).
    fn total_tokens(&self) -> usize {
        self.q2_tokens + self.res_pos
    }

    /// Flush residual buffer to Q2 storage.
    ///
    /// Key quantization: per-channel (for each head, each of `head_dim` channels
    /// is quantized across `group_size` tokens as one Q2 block).
    ///
    /// Value quantization: per-token (for each head, each token's `head_dim`
    /// values are quantized in groups of QK2_0).
    fn flush_residual(&mut self) {
        let gs = self.group_size; // = QK2_0
        assert!(self.res_pos >= gs, "not enough tokens to flush");

        // How many full groups to flush
        let n_groups = self.res_pos / gs;
        let flush_tokens = n_groups * gs;

        // === Key: per-channel quantization ===
        // Residual layout: [kv_heads][res_cap][head_dim]
        // For each head, for each channel (0..head_dim), gather `gs` token values
        // and quantize as one Q2 block.
        for h in 0..self.kv_heads {
            let head_base = h * self.res_cap * self.head_dim;
            for group in 0..n_groups {
                let tok_start = group * gs;
                for ch in 0..self.head_dim {
                    let mut vals = [0.0f32; QK2_0];
                    for (t, v) in vals.iter_mut().enumerate().take(gs) {
                        *v = self.res_k[head_base + (tok_start + t) * self.head_dim + ch];
                    }
                    self.q2_k.push(BlockQ2_0::quantize(&vals));
                }
            }
        }

        // === Value: per-token quantization ===
        // For each head, for each token, quantize head_dim values in groups of QK2_0.
        let blocks_per_token = self.head_dim / QK2_0;
        for h in 0..self.kv_heads {
            let head_base = h * self.res_cap * self.head_dim;
            for t in 0..flush_tokens {
                let tok_base = head_base + t * self.head_dim;
                for b in 0..blocks_per_token {
                    let start = tok_base + b * QK2_0;
                    let chunk: &[f32; QK2_0] = self.res_v[start..start + QK2_0].try_into().unwrap();
                    self.q2_v.push(BlockQ2_0::quantize(chunk));
                }
            }
        }

        self.q2_tokens += flush_tokens;

        // Shift remaining tokens (if any) to front of residual
        let remaining = self.res_pos - flush_tokens;
        if remaining > 0 {
            for h in 0..self.kv_heads {
                let head_base = h * self.res_cap * self.head_dim;
                let src_start = head_base + flush_tokens * self.head_dim;
                let dst_start = head_base;
                let count = remaining * self.head_dim;
                self.res_k
                    .copy_within(src_start..src_start + count, dst_start);
                self.res_v
                    .copy_within(src_start..src_start + count, dst_start);
            }
        }
        self.res_pos = remaining;
    }

    /// Assemble full K/V view by dequantizing Q2 blocks and copying residual data
    /// into the pre-allocated attention buffers.
    fn assemble_view(&mut self) {
        let gs = self.group_size;

        // === Dequantize Q2 Key (per-channel) ===
        if self.q2_tokens > 0 {
            let groups_per_flush = self.res_cap / gs;
            let n_flushes = self.q2_tokens / self.res_cap;
            let mut block_idx = 0;
            let mut channel_buf = [0.0f32; QK2_0];

            for flush in 0..n_flushes {
                let tok_base = flush * self.res_cap;
                for h in 0..self.kv_heads {
                    for g in 0..groups_per_flush {
                        let tok_start = tok_base + g * gs;
                        for ch in 0..self.head_dim {
                            self.q2_k[block_idx].dequantize(&mut channel_buf);
                            block_idx += 1;
                            for (t, &val) in channel_buf.iter().enumerate().take(gs) {
                                let pos = tok_start + t;
                                let out_idx =
                                    pos * self.kv_heads * self.head_dim + h * self.head_dim + ch;
                                self.attn_k_buf[out_idx] = val;
                            }
                        }
                    }
                }
            }

            // === Dequantize Q2 Value (per-token) ===
            let blocks_per_token = self.head_dim / QK2_0;
            let mut v_block_idx = 0;
            let mut deq_buf = [0.0f32; QK2_0];

            for flush in 0..n_flushes {
                let tok_base = flush * self.res_cap;
                for h in 0..self.kv_heads {
                    for t in 0..self.res_cap {
                        let pos = tok_base + t;
                        let out_base = pos * self.kv_heads * self.head_dim + h * self.head_dim;
                        for b in 0..blocks_per_token {
                            self.q2_v[v_block_idx].dequantize(&mut deq_buf);
                            v_block_idx += 1;
                            let start = out_base + b * QK2_0;
                            self.attn_v_buf[start..start + QK2_0].copy_from_slice(&deq_buf);
                        }
                    }
                }
            }
        }

        // === Copy residual FP32 data ===
        if self.res_pos > 0 {
            let token_offset = self.q2_tokens;
            for h in 0..self.kv_heads {
                let res_base = h * self.res_cap * self.head_dim;
                for t in 0..self.res_pos {
                    let pos = token_offset + t;
                    let out_base = pos * self.kv_heads * self.head_dim + h * self.head_dim;
                    let res_start = res_base + t * self.head_dim;
                    // K
                    self.attn_k_buf[out_base..out_base + self.head_dim]
                        .copy_from_slice(&self.res_k[res_start..res_start + self.head_dim]);
                    // V
                    self.attn_v_buf[out_base..out_base + self.head_dim]
                        .copy_from_slice(&self.res_v[res_start..res_start + self.head_dim]);
                }
            }
        }
    }
}

impl KVCacheOps for KiviCache {
    fn current_pos(&self) -> usize {
        self.total_tokens()
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
        DType::F32
    }

    fn memory_usage_bytes(&self) -> usize {
        let q2_bytes = (self.q2_k.len() + self.q2_v.len()) * std::mem::size_of::<BlockQ2_0>();
        let res_bytes = self.res_pos * self.kv_heads * self.head_dim * 4 * 2; // K + V
        q2_bytes + res_bytes
    }

    fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        let seq_len = new_k.shape().dims()[1];
        let total_after = self.total_tokens() + seq_len;
        if total_after > self.max_seq_len {
            return Err(anyhow::anyhow!(
                "KiviCache overflow: {} + {} > {}",
                self.total_tokens(),
                seq_len,
                self.max_seq_len
            ));
        }

        // Input layout: [batch=1, seq_len, kv_heads, head_dim] (SeqMajor, F32)
        let k_data = new_k.as_slice::<f32>();
        let v_data = new_v.as_slice::<f32>();

        // Append each token to residual buffer
        for s in 0..seq_len {
            // If residual is full, flush to Q2
            if self.res_pos >= self.res_cap {
                self.flush_residual();
            }

            // Copy this token into residual buffer
            // Input: seq-major [s * kv_heads * head_dim + h * head_dim + d]
            // Residual: [h * res_cap * head_dim + res_pos * head_dim + d]
            for h in 0..self.kv_heads {
                let src_base = s * self.kv_heads * self.head_dim + h * self.head_dim;
                let dst_base = h * self.res_cap * self.head_dim + self.res_pos * self.head_dim;
                self.res_k[dst_base..dst_base + self.head_dim]
                    .copy_from_slice(&k_data[src_base..src_base + self.head_dim]);
                self.res_v[dst_base..dst_base + self.head_dim]
                    .copy_from_slice(&v_data[src_base..src_base + self.head_dim]);
            }
            self.res_pos += 1;
        }

        Ok(())
    }

    fn get_view(&mut self) -> (Tensor, Tensor) {
        let total = self.total_tokens();
        if total == 0 {
            // Return empty tensors
            let backend: Arc<dyn crate::core::backend::Backend> = Arc::new(CpuBackend::new());
            let buf = Arc::new(SharedBuffer::new(0, DType::F32));
            let shape = Shape::new(vec![1, 0, self.kv_heads, self.head_dim]);
            let t = Tensor::new(shape.clone(), buf.clone(), backend.clone());
            return (t.clone(), t);
        }

        // Assemble full view (dequantize Q2 + copy residual)
        self.assemble_view();

        let buf_size = total * self.kv_heads * self.head_dim;

        // Create tensors wrapping the output buffers
        let backend: Arc<dyn crate::core::backend::Backend> = Arc::new(CpuBackend::new());
        let shape = Shape::new(vec![1, total, self.kv_heads, self.head_dim]);

        let k_byte_size = buf_size * 4;
        let k_buf = Arc::new(SharedBuffer::new(k_byte_size, DType::F32));
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.attn_k_buf.as_ptr(),
                k_buf.as_mut_ptr() as *mut f32,
                buf_size,
            );
        }
        let k_tensor = Tensor::new(shape.clone(), k_buf, backend.clone());

        let v_buf = Arc::new(SharedBuffer::new(k_byte_size, DType::F32));
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.attn_v_buf.as_ptr(),
                v_buf.as_mut_ptr() as *mut f32,
                buf_size,
            );
        }
        let v_tensor = Tensor::new(shape, v_buf, backend);

        (k_tensor, v_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::shared_buffer::SharedBuffer;
    use crate::core::buffer::Buffer;

    fn make_input_tensor(
        seq_len: usize,
        kv_heads: usize,
        head_dim: usize,
        base_val: f32,
    ) -> Tensor {
        let n = seq_len * kv_heads * head_dim;
        let mut data = vec![0.0f32; n];
        for i in 0..n {
            data[i] = base_val + (i as f32) * 0.01;
        }
        let buf = Arc::new(SharedBuffer::new(n * 4, DType::F32));
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.as_mut_ptr() as *mut f32, n);
        }
        let backend: Arc<dyn crate::core::backend::Backend> = Arc::new(CpuBackend::new());
        Tensor::new(
            Shape::new(vec![1, seq_len, kv_heads, head_dim]),
            buf,
            backend,
        )
    }

    #[test]
    fn test_kivi_cache_basic() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        assert_eq!(cache.current_pos(), 0);
        assert_eq!(cache.kv_dtype(), DType::F32);
        assert_eq!(cache.layout(), KVLayout::SeqMajor);

        // Add 1 token
        let k = make_input_tensor(1, kv_heads, head_dim, 1.0);
        let v = make_input_tensor(1, kv_heads, head_dim, 2.0);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.current_pos(), 1);
        assert_eq!(cache.res_pos, 1);
        assert_eq!(cache.q2_tokens, 0);
    }

    #[test]
    fn test_kivi_cache_residual_only() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Fill 16 tokens (within residual capacity)
        for i in 0..16 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1 + 100.0);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos(), 16);
        assert_eq!(cache.q2_tokens, 0);
        assert_eq!(cache.res_pos, 16);

        // get_view should return exact FP32 data (no quantization involved)
        let (k_view, _v_view) = cache.get_view();
        assert_eq!(k_view.shape().dims(), &[1, 16, kv_heads, head_dim]);

        let k_out = k_view.as_slice::<f32>();
        // First token, first head, first value
        let expected = 0.0 * 0.1 + 0.0 * 0.01; // base_val=0, i=0
        assert!(
            (k_out[0] - expected).abs() < 1e-5,
            "k_out[0]={}, expected={}",
            k_out[0],
            expected
        );
    }

    #[test]
    fn test_kivi_cache_flush_and_quantize() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Fill exactly res_cap tokens to trigger flush on the next insert
        for i in 0..res_cap {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.res_pos, res_cap);
        assert_eq!(cache.q2_tokens, 0);

        // Add one more token → triggers flush
        let k = make_input_tensor(1, kv_heads, head_dim, 99.0);
        let v = make_input_tensor(1, kv_heads, head_dim, 99.0);
        cache.update(&k, &v).unwrap();

        assert_eq!(cache.q2_tokens, res_cap);
        assert_eq!(cache.res_pos, 1);
        assert_eq!(cache.current_pos(), res_cap + 1);
    }

    #[test]
    fn test_kivi_cache_get_view_after_flush() {
        let kv_heads = 1;
        let head_dim = 32; // minimum for QK2_0
        let max_seq = 128;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Values: sequential so we can verify round-trip
        let mut all_k = Vec::new();
        let mut all_v = Vec::new();

        for i in 0..33 {
            let base_k = i as f32 * 0.1;
            let base_v = i as f32 * 0.1 + 50.0;
            let k = make_input_tensor(1, kv_heads, head_dim, base_k);
            let v = make_input_tensor(1, kv_heads, head_dim, base_v);

            // Save expected values
            all_k.extend_from_slice(k.as_slice::<f32>());
            all_v.extend_from_slice(v.as_slice::<f32>());

            cache.update(&k, &v).unwrap();
        }

        // 32 tokens flushed to Q2, 1 in residual
        assert_eq!(cache.q2_tokens, 32);
        assert_eq!(cache.res_pos, 1);

        let (k_view, v_view) = cache.get_view();
        let k_out = k_view.as_slice::<f32>();
        let v_out = v_view.as_slice::<f32>();

        assert_eq!(k_out.len(), 33 * kv_heads * head_dim);

        // Q2 tokens (0..32) should be approximately correct (Q2 error expected)
        let range_k = 3.2f32; // rough range of input values
        let max_q2_err = range_k / 6.0 + 0.1; // Q2 max error ≈ scale/2 + epsilon

        for i in 0..(32 * kv_heads * head_dim) {
            let err = (all_k[i] - k_out[i]).abs();
            assert!(
                err < max_q2_err,
                "K Q2 error at {i}: expected={}, got={}, err={err}",
                all_k[i],
                k_out[i]
            );
        }

        // Last token (residual, index 32) should be exact (FP32)
        let res_start = 32 * kv_heads * head_dim;
        for d in 0..head_dim {
            assert!(
                (all_k[res_start + d] - k_out[res_start + d]).abs() < 1e-5,
                "K residual mismatch at d={d}"
            );
            assert!(
                (all_v[res_start + d] - v_out[res_start + d]).abs() < 1e-5,
                "V residual mismatch at d={d}"
            );
        }
    }

    #[test]
    fn test_kivi_cache_multi_token_update() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Add 8 tokens at once (prefill-like)
        let k = make_input_tensor(8, kv_heads, head_dim, 1.0);
        let v = make_input_tensor(8, kv_heads, head_dim, 2.0);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.current_pos(), 8);
        assert_eq!(cache.res_pos, 8);
    }

    #[test]
    fn test_kivi_cache_overflow() {
        let kv_heads = 1;
        let head_dim = 32;
        let max_seq = 64;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Fill to max
        for _ in 0..64 {
            let k = make_input_tensor(1, kv_heads, head_dim, 0.0);
            let v = make_input_tensor(1, kv_heads, head_dim, 0.0);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos(), 64);

        // One more should fail
        let k = make_input_tensor(1, kv_heads, head_dim, 0.0);
        let v = make_input_tensor(1, kv_heads, head_dim, 0.0);
        assert!(cache.update(&k, &v).is_err());
    }

    #[test]
    fn test_kivi_cache_memory_usage() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        assert_eq!(cache.memory_usage_bytes(), 0);
    }

    #[test]
    fn test_kivi_cache_compression_ratio() {
        let kv_heads = 8;
        let head_dim = 64;
        let max_seq = 512;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Fill 128 tokens: flush at 33,65,97 → 96 Q2 + 32 residual
        for _ in 0..128 {
            let k = make_input_tensor(1, kv_heads, head_dim, 0.5);
            let v = make_input_tensor(1, kv_heads, head_dim, 0.5);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 96);
        assert_eq!(cache.res_pos, 32);
        assert_eq!(cache.current_pos(), 128);

        let kivi_bytes = cache.memory_usage_bytes();
        // FP32 would be: 128 * 8 * 64 * 4 bytes * 2 (K+V) = 524288 bytes
        let fp32_bytes = 128 * kv_heads * head_dim * 4 * 2;

        // Q2 portion has high compression, residual is FP32.
        // With 96 Q2 tokens: 96/128 * 10.67x compression + 32/128 * 1x
        // Should still show significant compression overall.
        let ratio = fp32_bytes as f64 / kivi_bytes as f64;
        assert!(
            ratio > 3.0,
            "compression ratio {ratio:.1}x too low (expected >3x)"
        );
    }
}
