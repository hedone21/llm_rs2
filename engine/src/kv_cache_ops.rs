//! KV cache 공통 trait 및 레이아웃 타입 (L2 shared identifier).
//!
//! `KVCacheOps` trait은 `KVCache` (F32/F16/Q4_0 + eviction)와 `KiviCache`
//! (Q2 + residual) 양쪽을 추상화하는 OCP extension point다. L1 backend
//! (`backend/opencl/plan.rs` 등)가 직접 import할 수 있도록 L2에 위치한다
//! (§13.8-G shared identifier promotion).
//!
//! `KVLayout`, `KiviRawBuffers`는 `KVCacheOps` method signature에서 사용되므로
//! 함께 위치한다.

use crate::buffer::DType;
use crate::tensor::Tensor;
use anyhow::Result;

/// KV cache 메모리 레이아웃.
///
/// - `SeqMajor`: `[batch, seq_pos, kv_heads, head_dim]` — positions contiguous across heads.
/// - `HeadMajor`: `[batch, kv_heads, seq_pos, head_dim]` — each head's positions contiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVLayout {
    SeqMajor,
    HeadMajor,
}

/// Raw GPU buffer references for native KIVI fused attention.
///
/// Provides direct access to quantized KV blocks and F32 residual buffers
/// without intermediate dequantization. Used by `attention_gen_kivi` kernel.
pub struct KiviRawBuffers<'a> {
    /// Quantized key blocks (Q2/Q4/Q8 packed).
    pub qk_buf: &'a Tensor,
    /// Quantized value blocks (Q2/Q4/Q8 packed).
    pub qv_buf: &'a Tensor,
    /// F32 residual keys [kv_heads, res_cap, head_dim].
    pub res_k: &'a Tensor,
    /// F32 residual values [kv_heads, res_cap, head_dim].
    pub res_v: &'a Tensor,
    /// Number of tokens in quantized storage.
    pub q_tokens: usize,
    /// Number of valid tokens in residual buffer.
    pub res_tokens: usize,
    /// Residual buffer capacity.
    pub res_cap: usize,
    /// Quantization bit-width (2, 4, or 8).
    pub bits: u8,
}

/// Trait abstracting KV cache operations for LlamaLayer/LlamaModel.
///
/// Implementors: `KVCache` (standard F32/F16/Q4_0 with eviction support),
/// `KiviCache` (KIVI Q2 + residual buffer, no eviction).
///
/// Generic monomorphization (`<C: KVCacheOps>`) is used instead of `dyn Trait`
/// to preserve contiguous slice access (`&mut [C]`) and zero runtime overhead.
pub trait KVCacheOps: Send {
    /// Number of valid tokens currently in the cache.
    fn current_pos(&self) -> usize;

    /// Override the current position counter.
    /// Used to undo a probe step's `update()` increment without modifying buffer contents.
    fn set_current_pos(&mut self, pos: usize);

    /// Physical buffer capacity in tokens.
    fn capacity(&self) -> usize;

    /// Number of KV heads.
    fn kv_heads(&self) -> usize;

    /// Dimension per head.
    fn head_dim(&self) -> usize;

    /// Memory layout.
    fn layout(&self) -> KVLayout;

    /// The DType that the caller should pass to `update()`.
    /// For KIVI, returns F32 (caller sends F32; KIVI quantizes internally).
    fn kv_dtype(&self) -> DType;

    /// Memory usage in bytes for currently stored KV data.
    fn memory_usage_bytes(&self) -> usize;

    /// Append new K/V data. Input shape: `[batch, seq_len, kv_heads, head_dim]`.
    fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()>;

    /// Get K/V tensors for attention computation.
    /// Returns `(k_tensor, v_tensor)` covering `[0..current_pos]`.
    /// `&mut self` allows internal buffer assembly (e.g. KIVI dequantization).
    fn get_view(&mut self) -> (Tensor, Tensor);

    /// Direct access to underlying K/V buffers for zero-overhead scatter writes.
    /// Returns None if the implementation doesn't support direct access (e.g. KIVI).
    fn get_buffers_mut(&mut self) -> Option<(&mut Tensor, &mut Tensor)> {
        None
    }

    /// Advance position counter without performing any data copy.
    /// Used with get_buffers_mut() when caller writes directly.
    fn advance_pos(&mut self, _n: usize) {}

    /// Ensure the cache has capacity for at least `min_tokens` total tokens.
    /// Grows the underlying buffers if needed. Returns true if buffers changed.
    fn ensure_capacity(&mut self, _min_tokens: usize) -> Result<bool> {
        Ok(false)
    }

    /// Whether this cache needs post-softmax attention scores computed
    /// during decode (even when no eviction policy requests them).
    /// Used by KiviCache for AWQE. Default: false.
    fn needs_attn_scores(&self) -> bool {
        false
    }

    /// Store post-softmax attention scores from the latest decode step.
    /// Used by KiviCache for AWQE (Attention-Weighted Quantization Error).
    /// Called after each decode step's attention; consumed during the next flush.
    fn set_attn_scores(
        &mut self,
        _scores: &[f32],
        _n_heads_q: usize,
        _stride: usize,
        _valid_len: usize,
    ) {
    }

    /// Get raw GPU buffers for native KIVI fused attention.
    ///
    /// Only KiviCache in GPU mode returns `Some`. Other caches return `None` (default).
    /// When `Some`, the caller can dispatch `attention_gen_kivi` directly,
    /// bypassing the intermediate F32 dequant + scatter step.
    fn get_kivi_raw_buffers(&self) -> Option<KiviRawBuffers<'_>> {
        None
    }

    // ── KIVI Plan support methods ──

    /// Current residual write position. Only meaningful for KiviCache.
    fn res_pos(&self) -> usize {
        0
    }

    /// Number of quantized tokens in compressed storage. Only meaningful for KiviCache.
    fn q2_tokens(&self) -> usize {
        0
    }

    /// Residual buffer capacity in tokens. Only meaningful for KiviCache.
    fn res_cap(&self) -> usize {
        0
    }

    /// Whether the residual buffer is full and needs flushing before next update.
    fn needs_flush(&self) -> bool {
        false
    }

    /// Flush residual to quantized storage if full. Returns Ok(true) if flushed.
    fn flush_if_needed(&mut self) -> Result<bool> {
        Ok(false)
    }
}
