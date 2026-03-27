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
//!
//! ## GPU mode
//!
//! When constructed with `new_gpu()`, KiviCache allocates persistent GPU buffers for
//! residual and attention data. The hot path (update, get_view) runs GPU kernels.
//! The cold path (quantize during flush) still runs on CPU since it is infrequent.
//! CPU-only mode is fully preserved for backward compatibility.

use crate::backend::cpu::CpuBackend;
use crate::buffer::shared_buffer::SharedBuffer;
use crate::core::backend::Backend;
use crate::core::buffer::{Buffer, DType};
use crate::core::kv_cache::{KVCacheOps, KVLayout};
use crate::core::memory::Memory;
use crate::core::quant::{BlockKVQ4, BlockKVQ8, BlockQ2_0, QKKV};
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use anyhow::Result;
use std::sync::Arc;

/// Snapshot of post-softmax attention scores from the previous decode step.
/// Used for AWQE (Attention-Weighted Quantization Error) during flush.
#[derive(Clone)]
struct AttnScoresSnapshot {
    /// Flattened scores: [n_heads_q * stride]. Post-softmax values.
    scores: Vec<f32>,
    /// Number of Q heads.
    n_heads_q: usize,
    /// Stride between heads (= max_seq_len allocation).
    stride: usize,
    /// Number of valid positions per head at snapshot time.
    valid_len: usize,
}

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
///
/// ## GPU mode
///
/// Set `gpu_backend` to `Some(...)` via `new_gpu()`. GPU buffers hold persistent
/// residual and attention F32 data. CPU residual vectors are still allocated for
/// the quantize step (cold path). All hot-path operations dispatch GPU kernels.
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

    /// Whether AWQE computation is enabled (set via set_awqe_enabled()).
    awqe_enabled: bool,
    /// Attention scores from the previous decode step (for AWQE).
    last_attn_scores: Option<AttnScoresSnapshot>,

    // ── GPU-native buffers (None = CPU-only mode, backward compatible) ──
    /// GPU backend handle. Some ↔ GPU mode enabled.
    gpu_backend: Option<Arc<dyn Backend>>,
    /// Memory allocator used to create GPU buffers.
    #[allow(dead_code)]
    gpu_memory: Option<Arc<dyn Memory>>,
    /// GPU F32 residual K buffer: [kv_heads, res_cap, head_dim]
    gpu_res_k: Option<Tensor>,
    /// GPU F32 residual V buffer: [kv_heads, res_cap, head_dim]
    gpu_res_v: Option<Tensor>,
    /// GPU F32 attention K output: [max_seq_len, kv_heads, head_dim]
    gpu_attn_k: Option<Tensor>,
    /// GPU F32 attention V output: [max_seq_len, kv_heads, head_dim]
    gpu_attn_v: Option<Tensor>,
    /// GPU byte buffer for Q2 key blocks (12 bytes per block).
    gpu_q2k: Option<Tensor>,
    /// GPU byte buffer for Q2 value blocks (12 bytes per block).
    gpu_q2v: Option<Tensor>,
    /// Number of Q2 key blocks written to `gpu_q2k`.
    gpu_q2k_blocks: usize,
    /// Number of Q2 value blocks written to `gpu_q2v`.
    gpu_q2v_blocks: usize,
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
            awqe_enabled: false,
            last_attn_scores: None,
            // GPU fields — None in CPU-only mode
            gpu_backend: None,
            gpu_memory: None,
            gpu_res_k: None,
            gpu_res_v: None,
            gpu_attn_k: None,
            gpu_attn_v: None,
            gpu_q2k: None,
            gpu_q2v: None,
            gpu_q2k_blocks: 0,
            gpu_q2v_blocks: 0,
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

    /// Enable/disable AWQE computation during flush.
    /// When enabled, `needs_attn_scores()` returns true, which causes
    /// the layer to compute attention scores even on GPU attention paths.
    pub fn set_awqe_enabled(&mut self, enabled: bool) {
        self.awqe_enabled = enabled;
    }

    /// Total number of valid tokens (Q2 + residual).
    fn total_tokens(&self) -> usize {
        self.q2_tokens + self.res_pos
    }

    /// Create a GPU-native KiviCache.
    ///
    /// When `backend` is an OpenCL backend, allocates persistent GPU buffers for
    /// residual and attention data. Falls back to CPU-only mode if GPU allocation fails.
    pub fn new_gpu(
        kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        residual_size: usize,
        bits: u8,
        backend: Arc<dyn Backend>,
        memory: Arc<dyn Memory>,
    ) -> Self {
        let mut cache = Self::new_with_bits(kv_heads, head_dim, max_seq_len, residual_size, bits);

        if backend.name() != "OpenCL" {
            return cache;
        }

        let res_elems = kv_heads * residual_size * head_dim;
        let attn_elems = max_seq_len * kv_heads * head_dim;

        // Q2 storage: max blocks needed
        let gs = QKKV;
        let groups_per_flush = residual_size / gs;
        let max_flushes = max_seq_len.div_ceil(residual_size);
        let blocks_per_token_v = head_dim / gs;
        let max_k_blocks = max_flushes * groups_per_flush * kv_heads * head_dim;
        let max_v_blocks = max_flushes * kv_heads * residual_size * blocks_per_token_v;

        let result = (|| -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
            let res_k_buf = memory.alloc(res_elems * 4, DType::F32)?;
            let res_k = Tensor::new(
                Shape::new(vec![kv_heads, residual_size, head_dim]),
                res_k_buf,
                backend.clone(),
            );
            let res_v_buf = memory.alloc(res_elems * 4, DType::F32)?;
            let res_v = Tensor::new(
                Shape::new(vec![kv_heads, residual_size, head_dim]),
                res_v_buf,
                backend.clone(),
            );

            let attn_k_buf = memory.alloc(attn_elems * 4, DType::F32)?;
            let attn_k = Tensor::new(
                Shape::new(vec![max_seq_len, kv_heads, head_dim]),
                attn_k_buf,
                backend.clone(),
            );
            let attn_v_buf = memory.alloc(attn_elems * 4, DType::F32)?;
            let attn_v = Tensor::new(
                Shape::new(vec![max_seq_len, kv_heads, head_dim]),
                attn_v_buf,
                backend.clone(),
            );

            // Q2 block storage: 12 bytes per block
            let q2k_bytes = max_k_blocks * 12;
            let q2k_buf = memory.alloc(q2k_bytes.max(12), DType::U8)?;
            let q2k = Tensor::new(
                Shape::new(vec![q2k_bytes.max(12)]),
                q2k_buf,
                backend.clone(),
            );
            let q2v_bytes = max_v_blocks * 12;
            let q2v_buf = memory.alloc(q2v_bytes.max(12), DType::U8)?;
            let q2v = Tensor::new(
                Shape::new(vec![q2v_bytes.max(12)]),
                q2v_buf,
                backend.clone(),
            );

            Ok((res_k, res_v, attn_k, attn_v, q2k, q2v))
        })();

        match result {
            Ok((res_k, res_v, attn_k, attn_v, q2k, q2v)) => {
                cache.gpu_backend = Some(backend);
                cache.gpu_memory = Some(memory);
                cache.gpu_res_k = Some(res_k);
                cache.gpu_res_v = Some(res_v);
                cache.gpu_attn_k = Some(attn_k);
                cache.gpu_attn_v = Some(attn_v);
                cache.gpu_q2k = Some(q2k);
                cache.gpu_q2v = Some(q2v);
                log::info!("KiviCache: GPU mode enabled");
            }
            Err(e) => {
                log::warn!("KiviCache: GPU alloc failed ({}), using CPU mode", e);
            }
        }

        cache
    }

    /// Returns `true` if GPU buffers are active.
    pub fn is_gpu(&self) -> bool {
        self.gpu_backend.is_some()
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
        // GPU position counters reset; buffers are reused (valid data region tracked by positions)
        self.gpu_q2k_blocks = 0;
        self.gpu_q2v_blocks = 0;
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
        self.flush_proxies.push(crate::core::qcf::compute_flush_opr(
            &proxy_params,
            &qcf_config,
        ));

        // AWQE: attention-weighted quantization error (V-only)
        if let Some(ref attn) = self.last_attn_scores {
            let gqa_group_size = if self.kv_heads > 0 {
                attn.n_heads_q / self.kv_heads
            } else {
                0
            };
            if gqa_group_size > 0 && self.q2_tokens < attn.valid_len {
                let awqe_params = crate::core::qcf::FlushAwqeParams {
                    res_v: &self.res_v,
                    kv_heads: self.kv_heads,
                    head_dim: self.head_dim,
                    flush_tokens,
                    res_cap: self.res_cap,
                    bits: self.bits,
                    attn_scores: &attn.scores,
                    n_heads_q: attn.n_heads_q,
                    scores_stride: attn.stride,
                    gqa_group_size,
                    flush_cache_start: self.q2_tokens,
                    scores_valid_len: attn.valid_len,
                };
                self.flush_proxies
                    .push(crate::core::qcf::compute_flush_awqe(
                        &awqe_params,
                        &qcf_config,
                    ));
            }
        }

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

    // ── GPU-mode private helpers ──────────────────────────────────────────────

    /// GPU update path: scatter input tokens into GPU residual buffer using
    /// `kivi_gather_update` kernel, flushing when the residual is full.
    fn update_gpu(&mut self, new_k: &Tensor, new_v: &Tensor, seq_len: usize) -> Result<()> {
        #[cfg(feature = "opencl")]
        {
            use crate::backend::opencl::OpenCLBackend;

            let backend_arc = self.gpu_backend.as_ref().unwrap().clone();
            let ocl = backend_arc
                .as_any()
                .downcast_ref::<OpenCLBackend>()
                .ok_or_else(|| anyhow::anyhow!("GPU mode requires OpenCLBackend"))?;

            let mut written = 0usize;
            while written < seq_len {
                if self.res_pos >= self.res_cap {
                    self.flush_residual_gpu()?;
                }
                // How many tokens we can write in this batch without overflow
                let batch = (seq_len - written).min(self.res_cap - self.res_pos);

                // We need to call the kernel for the [written..written+batch] slice of new_k/v.
                // The kernel signature is: input[seq_len, kv_heads, head_dim] → residual[kv_heads, res_cap, head_dim]
                // But our input tensor contains all seq_len tokens; we need a view of [batch] tokens starting at `written`.
                // Since GPU tensors may not support views, we create a lightweight wrapper pointing into the same buffer.
                // SAFETY: We rely on the kernel only reading `seq_len` tokens from the start of the input buffer.
                // To avoid GPU sub-buffer complexity, if written==0 and batch==seq_len we pass as-is.
                // Otherwise, we fall back to the CPU copy_slice approach per token.
                if written == 0 && batch == seq_len {
                    // Fast path: pass entire input at once
                    let gpu_res_k = self.gpu_res_k.as_mut().unwrap();
                    let gpu_res_v = self.gpu_res_v.as_mut().unwrap();
                    ocl.kivi_gather_update(
                        new_k,
                        gpu_res_k,
                        self.kv_heads,
                        self.res_cap,
                        self.head_dim,
                        batch,
                        self.res_pos,
                    )?;
                    ocl.kivi_gather_update(
                        new_v,
                        gpu_res_v,
                        self.kv_heads,
                        self.res_cap,
                        self.head_dim,
                        batch,
                        self.res_pos,
                    )?;
                } else {
                    // Slow path: token-by-token copy_slice for the sub-range
                    for s in written..written + batch {
                        let gpu_res_k = self.gpu_res_k.as_mut().unwrap();
                        let gpu_res_v = self.gpu_res_v.as_mut().unwrap();
                        for h in 0..self.kv_heads {
                            let src_off = s * self.kv_heads * self.head_dim + h * self.head_dim;
                            let dst_off = h * self.res_cap * self.head_dim
                                + (self.res_pos + (s - written)) * self.head_dim;
                            ocl.copy_slice(new_k, gpu_res_k, src_off, dst_off, self.head_dim)?;
                            ocl.copy_slice(new_v, gpu_res_v, src_off, dst_off, self.head_dim)?;
                        }
                    }
                }
                self.res_pos += batch;
                written += batch;
            }
            return Ok(());
        }
        #[allow(unreachable_code)]
        Err(anyhow::anyhow!(
            "update_gpu called but opencl feature not enabled"
        ))
    }

    /// GPU flush: read GPU residual → CPU quantize → upload Q2 blocks to GPU →
    /// dispatch GPU dequant kernels to fill attention buffers.
    fn flush_residual_gpu(&mut self) -> Result<()> {
        let gs = self.group_size;
        let n_groups = self.res_pos / gs;
        let flush_tokens = n_groups * gs;

        if flush_tokens == 0 {
            return Ok(());
        }

        // 1. Read GPU residual to CPU (needed for quantization)
        let res_bytes = self.kv_heads * self.res_cap * self.head_dim * 4;
        {
            let backend = self.gpu_backend.as_ref().unwrap();
            let gpu_res_k = self.gpu_res_k.as_ref().unwrap();
            let gpu_res_v = self.gpu_res_v.as_ref().unwrap();
            let k_dst = unsafe {
                std::slice::from_raw_parts_mut(self.res_k.as_mut_ptr() as *mut u8, res_bytes)
            };
            backend.read_buffer(gpu_res_k, k_dst)?;
            let v_dst = unsafe {
                std::slice::from_raw_parts_mut(self.res_v.as_mut_ptr() as *mut u8, res_bytes)
            };
            backend.read_buffer(gpu_res_v, v_dst)?;
        }

        // 2. Compute QCF proxy metrics (same as CPU path)
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
        self.flush_proxies.push(crate::core::qcf::compute_flush_opr(
            &proxy_params,
            &qcf_config,
        ));

        // AWQE (same logic as CPU flush_residual)
        if let Some(ref attn) = self.last_attn_scores {
            let gqa_group_size = if self.kv_heads > 0 {
                attn.n_heads_q / self.kv_heads
            } else {
                0
            };
            if gqa_group_size > 0 && self.q2_tokens < attn.valid_len {
                let awqe_params = crate::core::qcf::FlushAwqeParams {
                    res_v: &self.res_v,
                    kv_heads: self.kv_heads,
                    head_dim: self.head_dim,
                    flush_tokens,
                    res_cap: self.res_cap,
                    bits: self.bits,
                    attn_scores: &attn.scores,
                    n_heads_q: attn.n_heads_q,
                    scores_stride: attn.stride,
                    gqa_group_size,
                    flush_cache_start: self.q2_tokens,
                    scores_valid_len: attn.valid_len,
                };
                self.flush_proxies
                    .push(crate::core::qcf::compute_flush_awqe(
                        &awqe_params,
                        &qcf_config,
                    ));
            }
        }

        // 3. CPU quantize → fills self.qk / self.qv
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

        // 4. Upload Q2 blocks to GPU and run dequant kernels into attention buffers
        let tok_base = self.q2_tokens;
        self.upload_and_dequant_flush(flush_tokens, n_groups, tok_base)?;

        self.q2_tokens += flush_tokens;
        self.q2_deq_tokens = self.q2_tokens; // GPU dequant already done above

        // 5. Shift remaining residual tokens on GPU
        let remaining = self.res_pos - flush_tokens;
        if remaining > 0 {
            let backend = self.gpu_backend.as_ref().unwrap().clone();
            let gpu_res_k = self.gpu_res_k.as_mut().unwrap();
            for h in 0..self.kv_heads {
                let base = h * self.res_cap * self.head_dim;
                let src = base + flush_tokens * self.head_dim;
                let count = remaining * self.head_dim;
                backend.buffer_shift(gpu_res_k, src, base, count)?;
            }
            let gpu_res_v = self.gpu_res_v.as_mut().unwrap();
            for h in 0..self.kv_heads {
                let base = h * self.res_cap * self.head_dim;
                let src = base + flush_tokens * self.head_dim;
                let count = remaining * self.head_dim;
                backend.buffer_shift(gpu_res_v, src, base, count)?;
            }
        }
        self.res_pos = remaining;

        Ok(())
    }

    /// Upload newly quantized Q2 blocks to GPU buffers and dispatch dequant kernels.
    ///
    /// `flush_tokens`: number of tokens in this flush
    /// `n_groups`: number of Q2 key groups (= flush_tokens / group_size)
    /// `tok_base`: token offset in the attention buffer for this flush
    fn upload_and_dequant_flush(
        &mut self,
        flush_tokens: usize,
        n_groups: usize,
        tok_base: usize,
    ) -> Result<()> {
        let k_block_start = self.gpu_q2k_blocks;
        let v_block_start = self.gpu_q2v_blocks;

        let new_k_blocks = n_groups * self.kv_heads * self.head_dim;
        let blocks_per_token = self.head_dim / QKKV;
        let new_v_blocks = self.kv_heads * flush_tokens * blocks_per_token;

        // Serialize Q2 blocks to raw bytes and upload with byte offset.
        // BlockQ2_0 is repr(C) with fixed 12-byte size.
        const BLOCK_BYTES: usize = 12; // size_of::<BlockQ2_0>()

        let total_k_blocks_so_far = self.qk.len(); // after push in flush_residual_gpu
        let k_byte_offset = k_block_start * BLOCK_BYTES;
        let v_byte_offset = v_block_start * BLOCK_BYTES;

        // Collect the new blocks into contiguous byte slices
        let k_bytes =
            self.serialize_quantized_blocks_k(total_k_blocks_so_far - new_k_blocks, new_k_blocks);
        let total_v_blocks_so_far = self.qv.len();
        let v_bytes =
            self.serialize_quantized_blocks_v(total_v_blocks_so_far - new_v_blocks, new_v_blocks);

        #[cfg(feature = "opencl")]
        {
            use crate::backend::opencl::{OpenCLBackend, get_cl_mem};

            let backend_arc = self.gpu_backend.as_ref().unwrap().clone();
            if let Some(ocl) = backend_arc.as_any().downcast_ref::<OpenCLBackend>() {
                // Upload Q2 key blocks at byte offset
                {
                    let gpu_q2k = self.gpu_q2k.as_ref().unwrap();
                    let cl_mem = get_cl_mem(gpu_q2k.buffer().as_ref())?;
                    unsafe {
                        ocl::core::enqueue_write_buffer(
                            &ocl.queue,
                            cl_mem,
                            true,
                            k_byte_offset,
                            &k_bytes,
                            None::<&ocl::core::Event>,
                            None::<&mut ocl::core::Event>,
                        )?;
                    }
                }
                // Upload Q2 value blocks at byte offset
                {
                    let gpu_q2v = self.gpu_q2v.as_ref().unwrap();
                    let cl_mem = get_cl_mem(gpu_q2v.buffer().as_ref())?;
                    unsafe {
                        ocl::core::enqueue_write_buffer(
                            &ocl.queue,
                            cl_mem,
                            true,
                            v_byte_offset,
                            &v_bytes,
                            None::<&ocl::core::Event>,
                            None::<&mut ocl::core::Event>,
                        )?;
                    }
                }

                // Dispatch GPU dequant kernels
                {
                    let gpu_q2k = self.gpu_q2k.as_ref().unwrap();
                    let gpu_attn_k = self.gpu_attn_k.as_mut().unwrap();
                    ocl.kivi_dequantize_key_q2(
                        gpu_q2k,
                        gpu_attn_k,
                        self.kv_heads,
                        self.head_dim,
                        n_groups,
                        tok_base,
                        k_block_start,
                    )?;
                }
                {
                    let gpu_q2v = self.gpu_q2v.as_ref().unwrap();
                    let gpu_attn_v = self.gpu_attn_v.as_mut().unwrap();
                    ocl.kivi_dequantize_value_q2(
                        gpu_q2v,
                        gpu_attn_v,
                        self.kv_heads,
                        self.head_dim,
                        flush_tokens,
                        tok_base,
                        v_block_start,
                    )?;
                }
            }
        }

        self.gpu_q2k_blocks += new_k_blocks;
        self.gpu_q2v_blocks += new_v_blocks;

        Ok(())
    }

    /// Serialize Q2 key blocks [block_start..block_start+count] to raw bytes.
    fn serialize_quantized_blocks_k(&self, block_start: usize, count: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(count * 12);
        // Q4/Q8 GPU mode not yet supported; only Q2 blocks are serialized
        if let QuantizedBlocks::Q2(v) = &self.qk {
            for b in &v[block_start..block_start + count] {
                // SAFETY: BlockQ2_0 is repr(C), packed, 12 bytes
                let bytes =
                    unsafe { std::slice::from_raw_parts(b as *const BlockQ2_0 as *const u8, 12) };
                out.extend_from_slice(bytes);
            }
        }
        out
    }

    /// Serialize Q2 value blocks [block_start..block_start+count] to raw bytes.
    fn serialize_quantized_blocks_v(&self, block_start: usize, count: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(count * 12);
        if let QuantizedBlocks::Q2(v) = &self.qv {
            for b in &v[block_start..block_start + count] {
                let bytes =
                    unsafe { std::slice::from_raw_parts(b as *const BlockQ2_0 as *const u8, 12) };
                out.extend_from_slice(bytes);
            }
        }
        out
    }

    /// GPU assemble_view: ensure GPU attention buffers are up to date.
    ///
    /// - `q2_deq_tokens` is kept in sync with `q2_tokens` during `flush_residual_gpu`,
    ///   so no incremental dequant is needed here.
    /// - Residual scatter: uses `kivi_scatter_residual` kernel (always re-done each call).
    fn assemble_view_gpu(&mut self) -> Result<()> {
        #[cfg(feature = "opencl")]
        {
            use crate::backend::opencl::OpenCLBackend;

            if self.res_pos > 0 {
                let backend_arc = self.gpu_backend.as_ref().unwrap().clone();
                if let Some(ocl) = backend_arc.as_any().downcast_ref::<OpenCLBackend>() {
                    let gpu_res_k = self.gpu_res_k.as_ref().unwrap();
                    let gpu_attn_k = self.gpu_attn_k.as_mut().unwrap();
                    ocl.kivi_scatter_residual(
                        gpu_res_k,
                        gpu_attn_k,
                        self.kv_heads,
                        self.res_cap,
                        self.head_dim,
                        self.res_pos,
                        self.q2_tokens,
                    )?;
                    let gpu_res_v = self.gpu_res_v.as_ref().unwrap();
                    let gpu_attn_v = self.gpu_attn_v.as_mut().unwrap();
                    ocl.kivi_scatter_residual(
                        gpu_res_v,
                        gpu_attn_v,
                        self.kv_heads,
                        self.res_cap,
                        self.head_dim,
                        self.res_pos,
                        self.q2_tokens,
                    )?;
                }
            }
            return Ok(());
        }
        #[allow(unreachable_code)]
        Ok(())
    }

    /// GPU get_view: return Tensors backed by GPU attention buffers.
    fn get_view_gpu(&mut self) -> (Tensor, Tensor) {
        if let Err(e) = self.assemble_view_gpu() {
            log::warn!("KiviCache assemble_view_gpu error: {}", e);
        }

        let total = self.total_tokens();
        let backend = self.gpu_backend.as_ref().unwrap().clone();

        if total == 0 {
            let buf = Arc::new(SharedBuffer::new(0, DType::F32));
            let shape = Shape::new(vec![1, 0, self.kv_heads, self.head_dim]);
            let t = Tensor::new(shape.clone(), buf.clone(), backend);
            return (t.clone(), t);
        }

        let shape = Shape::new(vec![1, total, self.kv_heads, self.head_dim]);
        let k = Tensor::new(
            shape.clone(),
            self.gpu_attn_k.as_ref().unwrap().buffer().clone(),
            backend.clone(),
        );
        let v = Tensor::new(
            shape,
            self.gpu_attn_v.as_ref().unwrap().buffer().clone(),
            backend,
        );
        (k, v)
    }
}

impl KVCacheOps for KiviCache {
    fn current_pos(&self) -> usize {
        self.total_tokens()
    }

    fn set_current_pos(&mut self, _pos: usize) {
        // KiviCache position is derived from q2_tokens + res_pos; no-op.
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

        // GPU path: dispatch GPU kernels for residual update
        if self.gpu_backend.is_some() {
            return self.update_gpu(new_k, new_v, seq_len);
        }

        // === CPU path (unchanged) ===
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

    fn get_buffers_mut(&mut self) -> Option<(&mut Tensor, &mut Tensor)> {
        // GPU mode: return residual buffers to signal GPU tensor compatibility.
        // This tells update_kv_cache() to pass GPU tensors directly instead of
        // reading back to CPU. The actual data flow goes through update_gpu().
        if let (Some(k), Some(v)) = (&mut self.gpu_res_k, &mut self.gpu_res_v) {
            Some((k, v))
        } else {
            None
        }
    }

    fn get_view(&mut self) -> (Tensor, Tensor) {
        // GPU path: return tensors backed by GPU attention buffers
        if self.gpu_backend.is_some() {
            return self.get_view_gpu();
        }

        // === CPU path (unchanged) ===
        let total = self.total_tokens();
        let backend: Arc<dyn Backend> = self.out_backend.clone();
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

    fn needs_attn_scores(&self) -> bool {
        self.awqe_enabled
    }

    fn set_attn_scores(
        &mut self,
        scores: &[f32],
        n_heads_q: usize,
        stride: usize,
        valid_len: usize,
    ) {
        if !self.awqe_enabled {
            return;
        }
        let snapshot = self
            .last_attn_scores
            .get_or_insert_with(|| AttnScoresSnapshot {
                scores: Vec::new(),
                n_heads_q,
                stride,
                valid_len,
            });
        // Reuse allocation when possible
        snapshot.scores.clear();
        snapshot.scores.extend_from_slice(scores);
        snapshot.n_heads_q = n_heads_q;
        snapshot.stride = stride;
        snapshot.valid_len = valid_len;
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
        // Each flush pushes 2 QCF metrics (compute_flush_qcf → "kivi",
        // compute_flush_opr → "kivi_opr"). 2 flushes × 2 metrics = 4 total.
        assert_eq!(
            proxies.len(),
            4,
            "expected 4 flush proxies after 66 tokens with residual_size=32 (2 flushes × 2 metrics), got {}",
            proxies.len()
        );
        for p in &proxies {
            assert!(
                p.action == "kivi" || p.action == "kivi_opr",
                "unexpected action: {}",
                p.action
            );
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

    // ── GPU-mode tests ────────────────────────────────────────────────────────

    /// `new()` / `new_with_bits()` must always create CPU-mode caches.
    #[test]
    fn test_cpu_mode_is_default() {
        let cache = KiviCache::new(2, 64, 256, 32);
        assert!(!cache.is_gpu(), "new() must return CPU-mode cache");
        assert!(cache.gpu_backend.is_none());
        assert!(cache.gpu_res_k.is_none());
        assert!(cache.gpu_attn_k.is_none());
    }

    /// `new_gpu()` with a non-OpenCL backend must silently fall back to CPU mode.
    #[test]
    fn test_new_gpu_non_opencl_falls_back_to_cpu() {
        use crate::memory::galloc::Galloc;
        let cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let memory: Arc<dyn crate::core::memory::Memory> = Arc::new(Galloc::new());
        let cache = KiviCache::new_gpu(2, 64, 256, 32, 2, cpu_backend, memory);
        // CpuBackend.name() != "OpenCL" → must fall back to CPU mode
        assert!(
            !cache.is_gpu(),
            "new_gpu() with non-OpenCL backend must use CPU mode"
        );
    }

    /// `reset()` must clear GPU position counters.
    #[test]
    fn test_reset_clears_gpu_counters() {
        let mut cache = KiviCache::new(2, 64, 256, 32);
        // Manually poke GPU counters to non-zero values (simulating partial GPU use)
        cache.gpu_q2k_blocks = 42;
        cache.gpu_q2v_blocks = 17;
        cache.reset();
        assert_eq!(cache.gpu_q2k_blocks, 0, "reset() must zero gpu_q2k_blocks");
        assert_eq!(cache.gpu_q2v_blocks, 0, "reset() must zero gpu_q2v_blocks");
    }

    /// Existing CPU tests must not be affected by GPU fields being None.
    #[test]
    fn test_cpu_path_unaffected_by_gpu_fields() {
        let kv_heads = 2;
        let head_dim = 64;
        let max_seq = 128;
        let res_cap = 32;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        // Fill 33 tokens → triggers one flush
        for i in 0..33 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05 + 10.0);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.q2_tokens, 32);
        assert_eq!(cache.res_pos, 1);
        assert_eq!(
            cache.gpu_q2k_blocks, 0,
            "cpu mode must not touch gpu counters"
        );
        let (k_view, _) = cache.get_view();
        assert_eq!(k_view.shape().dims(), &[1, 33, kv_heads, head_dim]);
    }

    // ── AWQE tests ────────────────────────────────────────────────────────────

    /// Test 10: awqe_enabled=false → set_attn_scores is no-op, last_attn_scores stays None.
    #[test]
    fn test_kivi_awqe_disabled_no_scores_stored() {
        let mut cache = KiviCache::new(1, 32, 256, 32);
        assert!(!cache.awqe_enabled);

        // Call set_attn_scores with some data
        let scores = vec![0.1f32; 64];
        cache.set_attn_scores(&scores, 1, 64, 32);

        // awqe_enabled=false → must not store anything
        assert!(
            cache.last_attn_scores.is_none(),
            "set_attn_scores must be no-op when awqe_enabled=false"
        );
    }

    /// Test 11: awqe_enabled=true → set_attn_scores stores scores, flush produces kivi_awqe metric.
    #[test]
    fn test_kivi_awqe_enabled_scores_stored() {
        let kv_heads = 1;
        let head_dim = 32;
        let res_cap = 32;
        let n_heads_q = 1;
        let max_seq = 256;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        cache.set_awqe_enabled(true);

        // valid_len must cover flush tokens (q2_tokens=0 .. flush_tokens=32)
        // So valid_len >= 32
        let valid_len = 64usize;
        let stride = valid_len;
        let scores: Vec<f32> = (0..n_heads_q * stride)
            .map(|i| {
                if i < valid_len {
                    1.0 / valid_len as f32
                } else {
                    0.0
                }
            })
            .collect();

        // Store scores (simulating previous decode step)
        cache.set_attn_scores(&scores, n_heads_q, stride, valid_len);
        assert!(cache.last_attn_scores.is_some(), "scores must be stored");

        // Insert 33 tokens to trigger one flush at the 33rd token
        for i in 0..33 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05 + 1.0);
            cache.update(&k, &v).unwrap();
        }

        let proxies = cache.take_flush_proxies();
        // Must have NMSE, OPR, and AWQE
        let awqe_metric = proxies.iter().find(|m| m.action == "kivi_awqe");
        assert!(
            awqe_metric.is_some(),
            "flush_proxies must contain 'kivi_awqe' when awqe_enabled=true and scores present; got: {:?}",
            proxies.iter().map(|m| &m.action).collect::<Vec<_>>()
        );
        assert!(
            awqe_metric.unwrap().raw_value >= 0.0,
            "AWQE raw_value must be non-negative"
        );
    }

    /// Test 12: awqe_enabled=true but set_attn_scores never called → no AWQE in flush_proxies.
    #[test]
    fn test_kivi_awqe_no_scores_no_awqe() {
        let kv_heads = 1;
        let head_dim = 32;
        let res_cap = 32;
        let max_seq = 256;

        let mut cache = KiviCache::new(kv_heads, head_dim, max_seq, res_cap);
        cache.set_awqe_enabled(true);

        // Do NOT call set_attn_scores
        // Insert tokens to trigger a flush
        for i in 0..33 {
            let k = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.1);
            let v = make_input_tensor(1, kv_heads, head_dim, i as f32 * 0.05 + 1.0);
            cache.update(&k, &v).unwrap();
        }

        let proxies = cache.take_flush_proxies();
        let awqe_metric = proxies.iter().find(|m| m.action == "kivi_awqe");
        assert!(
            awqe_metric.is_none(),
            "flush_proxies must NOT contain 'kivi_awqe' when no scores were provided; got: {:?}",
            proxies.iter().map(|m| &m.action).collect::<Vec<_>>()
        );
        // But NMSE and OPR should still be present
        assert!(
            proxies.iter().any(|m| m.action == "kivi"),
            "NMSE proxy must still be present"
        );
        assert!(
            proxies.iter().any(|m| m.action == "kivi_opr"),
            "OPR proxy must still be present"
        );
    }

    /// Test 13: needs_attn_scores() mirrors awqe_enabled.
    #[test]
    fn test_kivi_needs_attn_scores() {
        use crate::core::kv_cache::KVCacheOps;

        let mut cache = KiviCache::new(1, 32, 256, 32);

        // Default: false
        assert!(
            !cache.needs_attn_scores(),
            "needs_attn_scores() must be false by default"
        );

        // After enabling
        cache.set_awqe_enabled(true);
        assert!(
            cache.needs_attn_scores(),
            "needs_attn_scores() must be true after set_awqe_enabled(true)"
        );

        // After disabling again
        cache.set_awqe_enabled(false);
        assert!(
            !cache.needs_attn_scores(),
            "needs_attn_scores() must be false after set_awqe_enabled(false)"
        );
    }
}
