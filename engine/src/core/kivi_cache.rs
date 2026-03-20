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
use crate::core::quant::{BlockKVQ4, BlockKVQ8, BlockQ2_0, QKKV};
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use anyhow::Result;
use std::sync::Arc;

/// Enum wrapping different quantization bit-widths for KV cache blocks.
#[derive(Clone)]
enum QuantizedBlocks {
    Q2(Vec<BlockQ2_0>),
    Q4(Vec<BlockKVQ4>),
    Q8(Vec<BlockKVQ8>),
}

impl QuantizedBlocks {
    fn new(bits: u8) -> Self {
        match bits {
            2 => QuantizedBlocks::Q2(Vec::new()),
            4 => QuantizedBlocks::Q4(Vec::new()),
            8 => QuantizedBlocks::Q8(Vec::new()),
            _ => panic!("unsupported bits: {bits}"),
        }
    }

    fn with_capacity(bits: u8, cap: usize) -> Self {
        match bits {
            2 => QuantizedBlocks::Q2(Vec::with_capacity(cap)),
            4 => QuantizedBlocks::Q4(Vec::with_capacity(cap)),
            8 => QuantizedBlocks::Q8(Vec::with_capacity(cap)),
            _ => panic!("unsupported bits: {bits}"),
        }
    }

    fn clear(&mut self) {
        match self {
            QuantizedBlocks::Q2(v) => v.clear(),
            QuantizedBlocks::Q4(v) => v.clear(),
            QuantizedBlocks::Q8(v) => v.clear(),
        }
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        match self {
            QuantizedBlocks::Q2(v) => v.len(),
            QuantizedBlocks::Q4(v) => v.len(),
            QuantizedBlocks::Q8(v) => v.len(),
        }
    }

    fn memory_bytes(&self) -> usize {
        match self {
            QuantizedBlocks::Q2(v) => v.len() * std::mem::size_of::<BlockQ2_0>(),
            QuantizedBlocks::Q4(v) => v.len() * std::mem::size_of::<BlockKVQ4>(),
            QuantizedBlocks::Q8(v) => v.len() * std::mem::size_of::<BlockKVQ8>(),
        }
    }

    /// Quantize a group of QKKV f32 values and push to storage.
    fn push_quantized(&mut self, src: &[f32; QKKV]) {
        match self {
            QuantizedBlocks::Q2(v) => v.push(BlockQ2_0::quantize(src)),
            QuantizedBlocks::Q4(v) => v.push(BlockKVQ4::quantize(src)),
            QuantizedBlocks::Q8(v) => v.push(BlockKVQ8::quantize(src)),
        }
    }

    /// Dequantize block at given index into output buffer.
    fn dequantize_block(&self, idx: usize, out: &mut [f32; QKKV]) {
        match self {
            QuantizedBlocks::Q2(v) => v[idx].dequantize(out),
            QuantizedBlocks::Q4(v) => v[idx].dequantize(out),
            QuantizedBlocks::Q8(v) => v[idx].dequantize(out),
        }
    }

    #[allow(dead_code)]
    fn capacity(&self) -> usize {
        match self {
            QuantizedBlocks::Q2(v) => v.capacity(),
            QuantizedBlocks::Q4(v) => v.capacity(),
            QuantizedBlocks::Q8(v) => v.capacity(),
        }
    }

    #[allow(dead_code)]
    fn bits(&self) -> u8 {
        match self {
            QuantizedBlocks::Q2(_) => 2,
            QuantizedBlocks::Q4(_) => 4,
            QuantizedBlocks::Q8(_) => 8,
        }
    }
}

/// KIVI KV cache: quantized compressed storage + FP32 residual buffer.
///
/// SeqMajor layout only (Phase 1). Data layout for residual:
/// `[kv_heads, res_cap, head_dim]` (head-first for easier per-channel Key quantization).
///
/// Quantized storage layout (after flush):
/// - Key: per-channel blocks. For each head, `head_dim` channels, each channel
///   quantized across tokens in groups of QKKV (32).
/// - Value: per-token blocks. For each head, each token's `head_dim` values
///   quantized in groups of QKKV (32).
///
/// Supports 2-bit (Q2), 4-bit (Q4), and 8-bit (Q8) quantization.
/// Dynamic bit transition via `transition_bits()`.
#[derive(Clone)]
pub struct KiviCache {
    /// Current quantization bit-width (2, 4, or 8).
    bits: u8,

    // Quantized compressed storage
    qk: QuantizedBlocks,
    qv: QuantizedBlocks,
    /// Number of tokens in quantized storage (always a multiple of `res_cap`).
    pub q2_tokens: usize,

    // FP32 residual buffer — layout: [kv_heads][res_cap][head_dim]
    res_k: Vec<f32>,
    res_v: Vec<f32>,
    /// Number of valid tokens in residual buffer.
    pub res_pos: usize,
    /// Residual buffer capacity (R). Must be a multiple of QKKV (32).
    res_cap: usize,

    // Pre-allocated output buffers for assemble_view() dequantization
    attn_k_buf: Vec<f32>,
    attn_v_buf: Vec<f32>,

    /// Number of quantized tokens already dequantized into attn_k_buf/attn_v_buf.
    /// Enables incremental dequantization — only new flushes are processed.
    q2_deq_tokens: usize,

    // Shared backend for get_view() tensors (avoid per-call allocation)
    out_backend: Arc<CpuBackend>,

    // Dimensions
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,

    // Group size for quantization (= QKKV = 32)
    group_size: usize,

    /// Accumulated flush proxy metrics (NMSE). Pushed during `flush_residual()`.
    flush_proxies: Vec<crate::core::qcf::QcfMetric>,
}

impl KiviCache {
    /// Create a new KiviCache with specified bit-width.
    ///
    /// - `kv_heads`: number of KV heads
    /// - `head_dim`: dimension per head (must be a multiple of QKKV)
    /// - `max_seq_len`: maximum sequence length
    /// - `residual_size`: FP32 residual buffer size in tokens (must be a multiple of QKKV)
    /// - `bits`: quantization bits (2, 4, or 8). Default: 2.
    pub fn new_with_bits(
        kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        residual_size: usize,
        bits: u8,
    ) -> Self {
        assert!(
            residual_size.is_multiple_of(QKKV) && residual_size > 0,
            "residual_size ({residual_size}) must be a positive multiple of {QKKV}"
        );
        assert!(
            head_dim.is_multiple_of(QKKV),
            "head_dim ({head_dim}) must be a multiple of {QKKV}"
        );
        assert!(
            bits == 2 || bits == 4 || bits == 8,
            "bits must be 2, 4, or 8, got {bits}"
        );

        let res_elems = kv_heads * residual_size * head_dim;
        let groups_per_flush = residual_size / QKKV;
        let max_flushes = max_seq_len / residual_size;
        let max_k_blocks = max_flushes * groups_per_flush * kv_heads * head_dim;
        let blocks_per_token = head_dim / QKKV;
        let max_v_blocks = max_flushes * kv_heads * residual_size * blocks_per_token;

        let attn_buf_size = max_seq_len * kv_heads * head_dim;

        Self {
            bits,
            qk: QuantizedBlocks::with_capacity(bits, max_k_blocks),
            qv: QuantizedBlocks::with_capacity(bits, max_v_blocks),
            q2_tokens: 0,
            res_k: vec![0.0; res_elems],
            res_v: vec![0.0; res_elems],
            res_pos: 0,
            res_cap: residual_size,
            attn_k_buf: vec![0.0; attn_buf_size],
            attn_v_buf: vec![0.0; attn_buf_size],
            q2_deq_tokens: 0,
            out_backend: Arc::new(CpuBackend::new()),
            kv_heads,
            head_dim,
            max_seq_len,
            group_size: QKKV,
            flush_proxies: Vec::new(),
        }
    }

    /// Create a new KiviCache with default 2-bit quantization.
    pub fn new(kv_heads: usize, head_dim: usize, max_seq_len: usize, residual_size: usize) -> Self {
        Self::new_with_bits(kv_heads, head_dim, max_seq_len, residual_size, 2)
    }

    /// Current quantization bit-width.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Take all accumulated flush proxy metrics (NMSE), draining the internal buffer.
    pub fn take_flush_proxies(&mut self) -> Vec<crate::core::qcf::QcfMetric> {
        std::mem::take(&mut self.flush_proxies)
    }

    /// Total number of valid tokens (Q2 + residual).
    fn total_tokens(&self) -> usize {
        self.q2_tokens + self.res_pos
    }

    /// Reset cache to empty state (reuse allocations).
    pub fn reset(&mut self) {
        self.qk.clear();
        self.qv.clear();
        self.q2_tokens = 0;
        self.res_k.fill(0.0);
        self.res_v.fill(0.0);
        self.res_pos = 0;
        self.attn_k_buf.fill(0.0);
        self.attn_v_buf.fill(0.0);
        self.q2_deq_tokens = 0;
        self.flush_proxies.clear();
    }

    /// Flush residual buffer to quantized storage.
    ///
    /// Key quantization: per-channel (for each head, each of `head_dim` channels
    /// is quantized across `group_size` tokens as one block).
    ///
    /// Value quantization: per-token (for each head, each token's `head_dim`
    /// values are quantized in groups of QKKV).
    fn flush_residual(&mut self) {
        let gs = self.group_size; // = QKKV
        assert!(self.res_pos >= gs, "not enough tokens to flush");
        debug_assert!(
            self.q2_tokens.is_multiple_of(self.res_cap),
            "q2_tokens ({}) must be a multiple of res_cap ({})",
            self.q2_tokens,
            self.res_cap
        );

        // How many full groups to flush
        let n_groups = self.res_pos / gs;
        let flush_tokens = n_groups * gs;

        // Compute NMSE proxy before quantization (FP32 originals still available)
        let qcf_config = crate::core::qcf::QcfConfig::default();
        let proxy_params = crate::core::qcf::FlushQcfParams {
            res_k: &self.res_k,
            res_v: &self.res_v,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            flush_tokens,
            res_cap: self.res_cap,
            bits: self.bits,
        };
        self.flush_proxies.push(crate::core::qcf::compute_flush_qcf(
            &proxy_params,
            &qcf_config,
        ));

        // === Key: per-channel quantization ===
        for h in 0..self.kv_heads {
            let head_base = h * self.res_cap * self.head_dim;
            for group in 0..n_groups {
                let tok_start = group * gs;
                for ch in 0..self.head_dim {
                    let mut vals = [0.0f32; QKKV];
                    for (t, v) in vals.iter_mut().enumerate().take(gs) {
                        *v = self.res_k[head_base + (tok_start + t) * self.head_dim + ch];
                    }
                    self.qk.push_quantized(&vals);
                }
            }
        }

        // === Value: per-token quantization ===
        let blocks_per_token = self.head_dim / QKKV;
        for h in 0..self.kv_heads {
            let head_base = h * self.res_cap * self.head_dim;
            for t in 0..flush_tokens {
                let tok_base = head_base + t * self.head_dim;
                for b in 0..blocks_per_token {
                    let start = tok_base + b * QKKV;
                    let chunk: &[f32; QKKV] = self.res_v[start..start + QKKV].try_into().unwrap();
                    self.qv.push_quantized(chunk);
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

    /// Transition quantized blocks to a new bit-width.
    ///
    /// Dequantizes existing blocks to FP32, then re-quantizes at the new bit-width.
    /// Error accumulates through the dequant→requant cycle.
    pub fn transition_bits(&mut self, new_bits: u8) -> Result<()> {
        assert!(
            new_bits == 2 || new_bits == 4 || new_bits == 8,
            "bits must be 2, 4, or 8, got {new_bits}"
        );
        if new_bits == self.bits {
            return Ok(());
        }
        if self.q2_tokens == 0 {
            // No quantized data — just switch the format
            self.qk = QuantizedBlocks::new(new_bits);
            self.qv = QuantizedBlocks::new(new_bits);
            self.bits = new_bits;
            return Ok(());
        }

        let gs = self.group_size;
        let n_groups_total = self.q2_tokens / gs;
        let blocks_per_token = self.head_dim / QKKV;

        // === Re-quantize Key blocks (per-channel) ===
        // Block order: for each flush, for each head, for each group, for each channel
        let total_k_blocks = n_groups_total * self.kv_heads * self.head_dim;
        let mut new_qk = QuantizedBlocks::with_capacity(new_bits, total_k_blocks);
        let mut buf = [0.0f32; QKKV];
        for i in 0..total_k_blocks {
            self.qk.dequantize_block(i, &mut buf);
            new_qk.push_quantized(&buf);
        }

        // === Re-quantize Value blocks (per-token) ===
        let total_v_blocks = self.q2_tokens * self.kv_heads * blocks_per_token;
        let mut new_qv = QuantizedBlocks::with_capacity(new_bits, total_v_blocks);
        for i in 0..total_v_blocks {
            self.qv.dequantize_block(i, &mut buf);
            new_qv.push_quantized(&buf);
        }

        self.qk = new_qk;
        self.qv = new_qv;
        self.bits = new_bits;
        // Invalidate dequant cache — assemble_view must re-dequantize everything
        self.q2_deq_tokens = 0;

        Ok(())
    }

    /// Assemble K/V view by dequantizing blocks and copying residual data
    /// into the pre-allocated attention buffers.
    ///
    /// **Incremental**: only dequantizes flushes that haven't been processed
    /// yet (tracked by `q2_deq_tokens`). Residual data is always re-copied
    /// because it changes every decode step.
    fn assemble_view(&mut self) {
        let gs = self.group_size;
        let groups_per_flush = self.res_cap / gs;

        // === Incremental quantized dequantization (only new flushes) ===
        if self.q2_tokens > self.q2_deq_tokens {
            debug_assert!(
                self.q2_tokens.is_multiple_of(self.res_cap),
                "q2_tokens ({}) must be a multiple of res_cap ({})",
                self.q2_tokens,
                self.res_cap
            );
            let old_flushes = self.q2_deq_tokens / self.res_cap;
            let new_flushes = self.q2_tokens / self.res_cap;

            let k_blocks_per_flush = groups_per_flush * self.kv_heads * self.head_dim;
            let blocks_per_token = self.head_dim / QKKV;
            let v_blocks_per_flush = self.kv_heads * self.res_cap * blocks_per_token;

            let mut k_block_idx = old_flushes * k_blocks_per_flush;
            let mut v_block_idx = old_flushes * v_blocks_per_flush;
            let mut channel_buf = [0.0f32; QKKV];
            let mut deq_buf = [0.0f32; QKKV];

            for flush in old_flushes..new_flushes {
                let tok_base = flush * self.res_cap;

                // Key: per-channel dequantization
                for h in 0..self.kv_heads {
                    for g in 0..groups_per_flush {
                        let tok_start = tok_base + g * gs;
                        for ch in 0..self.head_dim {
                            self.qk.dequantize_block(k_block_idx, &mut channel_buf);
                            k_block_idx += 1;
                            for (t, &val) in channel_buf.iter().enumerate().take(gs) {
                                let pos = tok_start + t;
                                let out_idx =
                                    pos * self.kv_heads * self.head_dim + h * self.head_dim + ch;
                                self.attn_k_buf[out_idx] = val;
                            }
                        }
                    }
                }

                // Value: per-token dequantization
                for h in 0..self.kv_heads {
                    for t in 0..self.res_cap {
                        let pos = tok_base + t;
                        let out_base = pos * self.kv_heads * self.head_dim + h * self.head_dim;
                        for b in 0..blocks_per_token {
                            self.qv.dequantize_block(v_block_idx, &mut deq_buf);
                            v_block_idx += 1;
                            let start = out_base + b * QKKV;
                            self.attn_v_buf[start..start + QKKV].copy_from_slice(&deq_buf);
                        }
                    }
                }
            }

            self.q2_deq_tokens = self.q2_tokens;
        }

        // === Copy residual FP32 data (always, since it changes every step) ===
        if self.res_pos > 0 {
            let token_offset = self.q2_tokens;
            for h in 0..self.kv_heads {
                let res_base = h * self.res_cap * self.head_dim;
                for t in 0..self.res_pos {
                    let pos = token_offset + t;
                    let out_base = pos * self.kv_heads * self.head_dim + h * self.head_dim;
                    let res_start = res_base + t * self.head_dim;
                    self.attn_k_buf[out_base..out_base + self.head_dim]
                        .copy_from_slice(&self.res_k[res_start..res_start + self.head_dim]);
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
        let q_bytes = self.qk.memory_bytes() + self.qv.memory_bytes();
        let res_bytes = self.res_pos * self.kv_heads * self.head_dim * 4 * 2; // K + V
        q_bytes + res_bytes
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
        let backend: Arc<dyn crate::core::backend::Backend> = self.out_backend.clone();
        if total == 0 {
            let buf = Arc::new(SharedBuffer::new(0, DType::F32));
            let shape = Shape::new(vec![1, 0, self.kv_heads, self.head_dim]);
            let t = Tensor::new(shape.clone(), buf.clone(), backend.clone());
            return (t.clone(), t);
        }

        // Incremental assemble (only dequantize new Q2 flushes + copy residual)
        self.assemble_view();

        let buf_size = total * self.kv_heads * self.head_dim;
        let byte_size = buf_size * 4;
        let shape = Shape::new(vec![1, total, self.kv_heads, self.head_dim]);

        let k_buf = Arc::new(SharedBuffer::new(byte_size, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(byte_size, DType::F32));
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.attn_k_buf.as_ptr(),
                k_buf.as_mut_ptr() as *mut f32,
                buf_size,
            );
            std::ptr::copy_nonoverlapping(
                self.attn_v_buf.as_ptr(),
                v_buf.as_mut_ptr() as *mut f32,
                buf_size,
            );
        }

        let k_tensor = Tensor::new(shape.clone(), k_buf, backend.clone());
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

    #[test]
    fn test_kivi_cache_incremental_deq() {
        // Verify that incremental dequantization produces identical results
        // to full dequantization across multiple flushes.
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);

        // Fill 64 tokens → 2 flushes (at token 32 and 64)
        for i in 0..65 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03 + 10.0);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 64);
        assert_eq!(cache.res_pos, 1);
        assert_eq!(cache.q2_deq_tokens, 0); // nothing dequantized yet

        // First get_view: dequantize all 64 Q2 tokens
        let (k1, v1) = cache.get_view();
        assert_eq!(cache.q2_deq_tokens, 64);
        let k1_data: Vec<f32> = k1.as_slice::<f32>().to_vec();
        let v1_data: Vec<f32> = v1.as_slice::<f32>().to_vec();

        // Add more tokens to trigger another flush
        for i in 65..97 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03 + 10.0);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 96);
        assert_eq!(cache.q2_deq_tokens, 64); // still at 64 from last get_view

        // Second get_view: should only dequantize flush 2 (tokens 64..95)
        let (k2, _v2) = cache.get_view();
        assert_eq!(cache.q2_deq_tokens, 96);
        let k2_data = k2.as_slice::<f32>();

        // First 64 tokens should be identical to the previous get_view
        let first_64_elems = 64 * kv_heads * head_dim;
        assert_eq!(&k1_data[..first_64_elems], &k2_data[..first_64_elems]);
        assert_eq!(
            &v1_data[..first_64_elems],
            &_v2.as_slice::<f32>()[..first_64_elems]
        );
    }

    #[test]
    fn test_kivi_cache_vec_capacity_no_realloc() {
        // With correct capacity calculations, no reallocation should occur
        // for sequences within max_seq_len.
        let kv_heads = 8;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 64; // larger residual to test groups_per_flush > 1

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        let initial_k_cap = cache.qk.capacity();
        let initial_v_cap = cache.qv.capacity();

        // Fill to max_seq_len (3 flushes: at 64, 128, 192 → 192 Q2 + 64 residual)
        for _ in 0..max_seq {
            let k = make_input_tensor(1, kv_heads, head_dim, 0.1);
            let v = make_input_tensor(1, kv_heads, head_dim, 0.2);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 192);
        assert_eq!(cache.res_pos, 64);

        // No reallocation should have occurred
        assert_eq!(
            cache.qk.capacity(),
            initial_k_cap,
            "K vec reallocated (initial={initial_k_cap}, now={})",
            cache.qk.capacity()
        );
        assert_eq!(
            cache.qv.capacity(),
            initial_v_cap,
            "V vec reallocated (initial={initial_v_cap}, now={})",
            cache.qv.capacity()
        );
    }

    /// Direct quality comparison: KIVI Q2 vs Baseline.
    ///
    /// Same synthetic KV data fed to both caches. Measures:
    /// - Cosine similarity (K and V separately)
    /// - MSE (mean squared error)
    /// - Memory usage
    ///
    /// Data pattern: sin-based low-rank structure (realistic for attention KV).
    #[test]
    fn test_compare_kivi_vs_baseline() {
        use crate::core::kv_cache::{KVCache, KVCacheOps};

        let kv_heads = 8;
        let head_dim = 64;
        let max_seq = 256;
        let prefill_len = 32;
        let decode_len = 96;
        let total = prefill_len + decode_len; // 128 tokens

        // Generate ground truth KV data (F32)
        // Pattern: sin-based with mild low-rank structure
        let mut ground_truth_k = vec![0.0f32; total * kv_heads * head_dim];
        let mut ground_truth_v = vec![0.0f32; total * kv_heads * head_dim];
        for t in 0..total {
            for h in 0..kv_heads {
                for d in 0..head_dim {
                    let idx = t * kv_heads * head_dim + h * head_dim + d;
                    // K: lower effective rank (attention key pattern)
                    ground_truth_k[idx] = ((t as f32 + 1.0) * (d as f32 + 1.0) * 0.003).sin() * 0.3
                        + ((h as f32 + 1.0) * (d as f32 + 1.0) * 0.007).cos() * 0.2;
                    // V: higher rank, more varied (value pattern)
                    ground_truth_v[idx] =
                        ((t as f32 * 0.1 + d as f32 * 0.05 + h as f32 * 0.3).sin()) * 0.4
                            + (t as f32 * 0.02).cos() * 0.1;
                }
            }
        }

        // Helper: create F32 tensor from slice
        let make_f32 = |data: &[f32], seq_len: usize| -> Tensor {
            let n = seq_len * kv_heads * head_dim;
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
        };

        // Helper: cosine similarity
        let cosine_sim = |a: &[f32], b: &[f32]| -> f64 {
            let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..a.len() {
                let (x, y) = (a[i] as f64, b[i] as f64);
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            dot / (na.sqrt() * nb.sqrt() + 1e-15)
        };

        // Helper: MSE
        let mse = |a: &[f32], b: &[f32]| -> f64 {
            let sum: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let d = (x - y) as f64;
                    d * d
                })
                .sum();
            sum / a.len() as f64
        };

        let elems_per_step = |seq_len: usize| seq_len * kv_heads * head_dim;

        // ═══════════════════════════════════════════════
        // 1. Baseline: standard KVCache (F32, lossless)
        // ═══════════════════════════════════════════════
        let baseline_k: Vec<f32>;
        let baseline_v: Vec<f32>;
        {
            let memory: Arc<dyn crate::core::memory::Memory> =
                Arc::new(crate::memory::galloc::Galloc);
            let backend: Arc<dyn crate::core::backend::Backend> = Arc::new(CpuBackend::new());
            let k_buf = memory
                .alloc(max_seq * kv_heads * head_dim * 4, DType::F32)
                .unwrap();
            let v_buf = memory
                .alloc(max_seq * kv_heads * head_dim * 4, DType::F32)
                .unwrap();
            let mut cache = KVCache::new(
                Tensor::new(
                    Shape::new(vec![1, max_seq, kv_heads, head_dim]),
                    k_buf,
                    backend.clone(),
                ),
                Tensor::new(
                    Shape::new(vec![1, max_seq, kv_heads, head_dim]),
                    v_buf,
                    backend.clone(),
                ),
                max_seq,
            );

            // Prefill
            let off = 0;
            let n = elems_per_step(prefill_len);
            let k = make_f32(&ground_truth_k[off..off + n], prefill_len);
            let v = make_f32(&ground_truth_v[off..off + n], prefill_len);
            cache.update(&k, &v).unwrap();

            // Decode
            for t in 0..decode_len {
                let off = (prefill_len + t) * kv_heads * head_dim;
                let n = elems_per_step(1);
                let k = make_f32(&ground_truth_k[off..off + n], 1);
                let v = make_f32(&ground_truth_v[off..off + n], 1);
                cache.update(&k, &v).unwrap();
            }

            let (kv, vv) = KVCacheOps::get_view(&mut cache);
            let n = total * kv_heads * head_dim;
            baseline_k = kv.as_slice::<f32>()[..n].to_vec();
            baseline_v = vv.as_slice::<f32>()[..n].to_vec();
        }

        // ═══════════════════════════════════════════════
        // 2. KIVI Q2 (res_cap=32)
        // ═══════════════════════════════════════════════
        let kivi_k: Vec<f32>;
        let kivi_v: Vec<f32>;
        let kivi_mem: usize;
        {
            let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, 32);

            // Prefill
            let off = 0;
            let n = elems_per_step(prefill_len);
            let k = make_f32(&ground_truth_k[off..off + n], prefill_len);
            let v = make_f32(&ground_truth_v[off..off + n], prefill_len);
            cache.update(&k, &v).unwrap();

            // Decode
            for t in 0..decode_len {
                let off = (prefill_len + t) * kv_heads * head_dim;
                let n = elems_per_step(1);
                let k = make_f32(&ground_truth_k[off..off + n], 1);
                let v = make_f32(&ground_truth_v[off..off + n], 1);
                cache.update(&k, &v).unwrap();
            }

            kivi_mem = cache.memory_usage_bytes();
            let (kv, vv) = cache.get_view();
            let n = total * kv_heads * head_dim;
            kivi_k = kv.as_slice::<f32>()[..n].to_vec();
            kivi_v = vv.as_slice::<f32>()[..n].to_vec();
        }

        // ═══════════════════════════════════════════════
        // 3. F16 baseline (quantization-only loss, no compression)
        // ═══════════════════════════════════════════════
        let f16_baseline_k: Vec<f32> = ground_truth_k
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let f16_baseline_v: Vec<f32> = ground_truth_v
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();

        // ═══════════════════════════════════════════════
        // Compute metrics
        // ═══════════════════════════════════════════════
        let n = total * kv_heads * head_dim;

        // KIVI vs Baseline (F32)
        let kivi_k_cos = cosine_sim(&baseline_k, &kivi_k);
        let kivi_v_cos = cosine_sim(&baseline_v, &kivi_v);
        let kivi_k_mse = mse(&baseline_k, &kivi_k);
        let kivi_v_mse = mse(&baseline_v, &kivi_v);

        // F16 vs F32 Baseline (pure precision loss)
        let f16_k_cos = cosine_sim(&baseline_k, &f16_baseline_k[..n]);
        let f16_v_cos = cosine_sim(&baseline_v, &f16_baseline_v[..n]);

        // Memory
        let baseline_mem = total * kv_heads * head_dim * 4 * 2; // F32 K+V

        // ═══════════════════════════════════════════════
        // Print comparison table
        // ═══════════════════════════════════════════════
        eprintln!();
        eprintln!("╔══════════════════════════════════════════════════════════════════╗");
        eprintln!("║           KIVI vs Baseline Quality Comparison                   ║");
        eprintln!(
            "║  Data: {total} tokens, {kv_heads} heads, {head_dim} dim, prefill={prefill_len} decode={decode_len}  ║"
        );
        eprintln!("╠══════════════════════════════════════════════════════════════════╣");
        eprintln!("║                      │  K CosSim  │  V CosSim  │  K MSE       ║");
        eprintln!("╠══════════════════════════════════════════════════════════════════╣");
        eprintln!(
            "║ F16 (precision only) │  {:.6}  │  {:.6}  │  {:.2e}  ║",
            f16_k_cos,
            f16_v_cos,
            mse(&baseline_k, &f16_baseline_k[..n])
        );
        eprintln!(
            "║ KIVI Q2 res=32       │  {:.6}  │  {:.6}  │  {:.2e}  ║",
            kivi_k_cos, kivi_v_cos, kivi_k_mse
        );
        eprintln!("╠══════════════════════════════════════════════════════════════════╣");
        eprintln!("║                      │  V MSE     │  Memory    │  Ratio       ║");
        eprintln!("╠══════════════════════════════════════════════════════════════════╣");
        eprintln!(
            "║ Baseline (F32)       │  0.00e+00  │  {:>7}B  │  1.00x       ║",
            baseline_mem
        );
        eprintln!(
            "║ KIVI Q2 res=32       │  {:.2e}  │  {:>7}B  │  {:.2}x       ║",
            kivi_v_mse,
            kivi_mem,
            baseline_mem as f64 / kivi_mem as f64
        );
        eprintln!("╚══════════════════════════════════════════════════════════════════╝");
        eprintln!();

        // ═══════════════════════════════════════════════
        // Assertions
        // ═══════════════════════════════════════════════

        // KIVI should have reasonable K cosine similarity
        assert!(kivi_k_cos > 0.9, "KIVI K CosSim {kivi_k_cos:.4} < 0.9");

        // KIVI should achieve good compression ratio
        let kivi_ratio = baseline_mem as f64 / kivi_mem as f64;
        assert!(
            kivi_ratio > 1.5,
            "KIVI ratio ({kivi_ratio:.1}x) should be > 1.5x"
        );
    }

    #[test]
    fn test_kivi_new_with_bits_q4() {
        let cache = KiviCache::new_with_bits(2, 64, 256, 32, 4);
        assert_eq!(cache.bits(), 4);
        assert_eq!(cache.current_pos(), 0);
    }

    #[test]
    fn test_kivi_new_with_bits_q8() {
        let cache = KiviCache::new_with_bits(2, 64, 256, 32, 8);
        assert_eq!(cache.bits(), 8);
    }

    #[test]
    fn test_kivi_q4_flush_and_view() {
        let kv_heads = 1;
        let head_dim = 32;
        let max_seq = 128;
        let res_cap = 32;

        let mut cache = KiviCache::new_with_bits(kv_heads, head_dim, max_seq, res_cap, 4);

        let mut all_k = Vec::new();
        for i in 0..33 {
            let base = i as f32 * 0.1;
            let k = make_input_tensor(1, kv_heads, head_dim, base);
            let v = make_input_tensor(1, kv_heads, head_dim, base + 50.0);
            all_k.extend_from_slice(k.as_slice::<f32>());
            cache.update(&k, &v).unwrap();
        }

        assert_eq!(cache.q2_tokens, 32);
        assert_eq!(cache.res_pos, 1);
        assert_eq!(cache.bits(), 4);

        let (k_view, _v_view) = cache.get_view();
        let k_out = k_view.as_slice::<f32>();

        // Q4 should have lower error than Q2
        let range = 3.2f32;
        let max_q4_err = range / 30.0 + 0.1;
        for i in 0..(32 * kv_heads * head_dim) {
            let err = (all_k[i] - k_out[i]).abs();
            assert!(
                err < max_q4_err,
                "K Q4 error at {i}: expected={}, got={}, err={err}",
                all_k[i],
                k_out[i]
            );
        }
    }

    #[test]
    fn test_kivi_transition_bits_8_to_4_to_2() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 256;
        let res_cap = 32;

        let mut cache = KiviCache::new_with_bits(kv_heads, head_dim, max_seq, res_cap, 8);

        // Fill 65 tokens → 2 flushes (flush at 32, 64) + 1 residual
        for i in 0..65 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.03 + 10.0);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 64);
        assert_eq!(cache.bits(), 8);

        // Get Q8 view as reference
        let n_q = 64 * kv_heads * head_dim;
        let (k8, _) = cache.get_view();
        let k8_data: Vec<f32> = k8.as_slice::<f32>()[..n_q].to_vec();

        // Transition 8 → 4
        cache.transition_bits(4).unwrap();
        assert_eq!(cache.bits(), 4);
        assert_eq!(cache.q2_tokens, 64);

        let (k4, _) = cache.get_view();
        let k4_data = k4.as_slice::<f32>();

        // Q4 should be close to Q8 (some error from requant)
        let mse_8_to_4: f32 = k8_data
            .iter()
            .zip(k4_data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / k8_data.len() as f32;

        // Transition 4 → 2
        cache.transition_bits(2).unwrap();
        assert_eq!(cache.bits(), 2);

        let (k2, _) = cache.get_view();
        let k2_data = k2.as_slice::<f32>();

        let mse_4_to_2: f32 = k4_data
            .iter()
            .zip(k2_data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / k4_data.len() as f32;

        // Both MSEs should be bounded (not NaN, not huge)
        assert!(mse_8_to_4 < 1.0, "MSE 8→4 too high: {mse_8_to_4}");
        assert!(mse_4_to_2 < 5.0, "MSE 4→2 too high: {mse_4_to_2}");

        // Q4→Q2 should have more error than Q8→Q4
        assert!(
            mse_4_to_2 > mse_8_to_4,
            "Expected more error in 4→2 than 8→4"
        );
    }

    #[test]
    fn test_kivi_transition_noop() {
        let mut cache = KiviCache::new_with_bits(1, 32, 128, 32, 4);
        // Fill 32 tokens
        for _ in 0..33 {
            let k = make_input_tensor(1, 1, 32, 0.5);
            let v = make_input_tensor(1, 1, 32, 0.5);
            cache.update(&k, &v).unwrap();
        }
        // Transition to same bits — should be no-op
        cache.transition_bits(4).unwrap();
        assert_eq!(cache.bits(), 4);
    }

    #[test]
    fn test_kivi_transition_empty() {
        let mut cache = KiviCache::new_with_bits(1, 32, 128, 32, 8);
        // No tokens — transition should just switch format
        cache.transition_bits(2).unwrap();
        assert_eq!(cache.bits(), 2);
        assert_eq!(cache.q2_tokens, 0);
    }

    #[test]
    fn test_take_flush_proxies_accumulates_multiple_flushes() {
        // residual_size=32 → flush triggers when res_pos reaches res_cap (32).
        // Flush #1 fires on token index 32 (before inserting it, res_pos==32).
        // Flush #2 fires on token index 65 (before inserting it, res_pos==32 again).
        // So we need 65+1=66 tokens to observe 2 flushes.
        let mut cache = KiviCache::new(1, 32, 256, 32);

        // Insert 66 tokens → 2 flushes should have occurred
        for i in 0..66 {
            let k = make_input_tensor(1, 1, 32, i as f32 * 0.1);
            let v = make_input_tensor(1, 1, 32, i as f32 * 0.1 + 5.0);
            cache.update(&k, &v).unwrap();
        }

        let proxies = cache.take_flush_proxies();
        assert_eq!(
            proxies.len(),
            2,
            "expected 2 flush proxies after 66 tokens with residual_size=32, got {}",
            proxies.len()
        );
        for p in &proxies {
            assert_eq!(p.action, "kivi");
            assert!(p.raw_value >= 0.0, "NMSE should be non-negative");
            assert_eq!(p.tokens_affected, 32);
        }

        // After take, buffer should be drained
        let proxies2 = cache.take_flush_proxies();
        assert!(proxies2.is_empty(), "buffer should be empty after take");
    }

    #[test]
    fn test_take_flush_proxies_empty_when_no_flush() {
        // residual_size=32 → no flush until 32 tokens are inserted
        let mut cache = KiviCache::new(1, 32, 256, 32);

        // Insert only 16 tokens (< residual_size) → no flush
        for i in 0..16 {
            let k = make_input_tensor(1, 1, 32, i as f32 * 0.1);
            let v = make_input_tensor(1, 1, 32, i as f32 * 0.1 + 5.0);
            cache.update(&k, &v).unwrap();
        }

        let proxies = cache.take_flush_proxies();
        assert!(
            proxies.is_empty(),
            "no flush should occur below residual_size"
        );
    }

    #[test]
    fn test_reset_clears_flush_proxies() {
        let mut cache = KiviCache::new(1, 32, 256, 32);

        // Flush fires when res_pos reaches res_cap before inserting the 33rd token.
        for i in 0..33 {
            let k = make_input_tensor(1, 1, 32, i as f32 * 0.1);
            let v = make_input_tensor(1, 1, 32, i as f32 * 0.1 + 5.0);
            cache.update(&k, &v).unwrap();
        }
        assert!(
            !cache.flush_proxies.is_empty(),
            "flush_proxies should have entries after flush"
        );

        cache.reset();
        assert!(
            cache.flush_proxies.is_empty(),
            "reset() must clear flush_proxies"
        );
    }
}
