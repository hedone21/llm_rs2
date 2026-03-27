use crate::core::buffer::DType;
use crate::core::memory::Memory;
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use anyhow::Result;
use std::sync::Arc;

// ── KVCacheOps trait (OCP extension point) ──────────────────────────────────

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
}

/// Extension trait for KV caches that support prefetch pipelines.
///
/// Implementors: `OffloadKVCache`.
/// Used by `forward_into_offload` to overlap I/O with compute.
pub trait PrefetchableCache: KVCacheOps {
    /// Pre-load data from external storage into memory buffers.
    fn preload(&mut self) -> Result<()>;

    /// Release memory buffers to free RAM (only 2 layers active at once).
    fn release_buffers(&mut self);

    /// Reset preloaded flag at token boundary.
    fn reset_preload(&mut self);

    /// Re-arm preloaded state for cross-token buffer retention.
    ///
    /// Call after `get_view()` on layers whose buffers should survive the token
    /// boundary. The next token's `update()` dual-writes into the still-valid
    /// attn buffers, so `preload()` becomes a no-op (early-return on
    /// `self.preloaded == true`).
    fn retain_preload(&mut self) {}
}

/// KV cache memory layout.
///
/// - `SeqMajor`: `[batch, seq_pos, kv_heads, head_dim]` — positions contiguous across heads.
/// - `HeadMajor`: `[batch, kv_heads, seq_pos, head_dim]` — each head's positions contiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVLayout {
    SeqMajor,
    HeadMajor,
}

pub struct KVCache {
    pub k_buffer: Tensor,
    pub v_buffer: Tensor,
    pub current_pos: usize,
    /// High-water mark: the maximum `current_pos` ever reached.
    ///
    /// Invariant: `current_pos <= high_water_pos <= capacity`.
    /// Used by `release_unused_pages()` to limit madvise to the
    /// region that was actually written, avoiding spurious page faults
    /// in the never-touched tail of the buffer.
    pub high_water_pos: usize,
    pub max_seq_len: usize,
    capacity: usize,
    kv_heads: usize,
    head_dim: usize,
    pub(crate) layout: KVLayout,
    memory: Option<Arc<dyn Memory>>,
}

impl KVCache {
    /// Create a KVCache with full pre-allocation (capacity = max_seq_len).
    /// Growth is disabled. Layout defaults to SeqMajor for backward compatibility.
    pub fn new(k: Tensor, v: Tensor, max_seq_len: usize) -> Self {
        let shape = k.shape().dims();
        let kv_heads = shape[2];
        let head_dim = shape[3];
        Self {
            k_buffer: k,
            v_buffer: v,
            current_pos: 0,
            high_water_pos: 0,
            max_seq_len,
            capacity: max_seq_len,
            kv_heads,
            head_dim,
            layout: KVLayout::SeqMajor,
            memory: None,
        }
    }

    /// Create a KVCache with dynamic grow-on-demand allocation.
    /// Starts with `initial_capacity` and doubles up to `max_seq_len`.
    /// Layout defaults to SeqMajor for backward compatibility.
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
            high_water_pos: 0,
            max_seq_len,
            capacity: initial_capacity,
            kv_heads,
            head_dim,
            layout: KVLayout::SeqMajor,
            memory: Some(memory),
        }
    }

    /// Current physical buffer capacity in tokens.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn layout(&self) -> KVLayout {
        self.layout
    }

    /// Set the KV cache layout. Must be called before any data is written.
    pub fn with_layout(mut self, layout: KVLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Element offset for position `pos` of head `head`.
    ///
    /// - SeqMajor: `pos * kv_heads * head_dim + head * head_dim`
    /// - HeadMajor: `head * capacity * head_dim + pos * head_dim`
    #[inline]
    pub fn offset(&self, pos: usize, head: usize) -> usize {
        match self.layout {
            KVLayout::SeqMajor => pos * self.kv_heads * self.head_dim + head * self.head_dim,
            KVLayout::HeadMajor => head * self.capacity * self.head_dim + pos * self.head_dim,
        }
    }

    /// Elements between consecutive positions for the same head.
    #[inline]
    pub fn pos_stride(&self) -> usize {
        match self.layout {
            KVLayout::SeqMajor => self.kv_heads * self.head_dim,
            KVLayout::HeadMajor => self.head_dim,
        }
    }

    /// Block offset for Q4_0 data at (pos, head).
    ///
    /// Returns the block index into a `&[BlockQ4_0]` slice.
    /// `blocks_per_pos` = head_dim / QK4_0 (e.g. 64/32 = 2).
    #[inline]
    pub fn q4_block_offset(&self, pos: usize, head: usize, blocks_per_pos: usize) -> usize {
        match self.layout {
            KVLayout::SeqMajor => (pos * self.kv_heads + head) * blocks_per_pos,
            KVLayout::HeadMajor => (head * self.capacity + pos) * blocks_per_pos,
        }
    }

    /// Elements between consecutive heads for the same position.
    #[inline]
    pub fn head_stride(&self) -> usize {
        match self.layout {
            KVLayout::SeqMajor => self.head_dim,
            KVLayout::HeadMajor => self.capacity * self.head_dim,
        }
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

        let new_shape = match self.layout {
            KVLayout::SeqMajor => Shape::new(vec![1, new_cap, self.kv_heads, self.head_dim]),
            KVLayout::HeadMajor => Shape::new(vec![1, self.kv_heads, new_cap, self.head_dim]),
        };

        // Allocate new buffers
        let new_k_buf = memory.alloc(buf_size, dtype)?;
        let mut new_k = Tensor::new(new_shape.clone(), new_k_buf, backend.clone());
        let new_v_buf = memory.alloc(buf_size, dtype)?;
        let mut new_v = Tensor::new(new_shape, new_v_buf, backend.clone());

        // Copy existing data
        if self.current_pos > 0 {
            match self.layout {
                KVLayout::SeqMajor => {
                    // Contiguous: all positions × heads × dim
                    let copy_count = if dtype == crate::core::buffer::DType::Q4_0 {
                        let blocks_per_pos =
                            self.kv_heads * self.head_dim / crate::core::quant::QK4_0;
                        self.current_pos * blocks_per_pos
                    } else {
                        self.current_pos * self.kv_heads * self.head_dim
                    };
                    backend.copy_slice(&self.k_buffer, &mut new_k, 0, 0, copy_count)?;
                    backend.copy_slice(&self.v_buffer, &mut new_v, 0, 0, copy_count)?;
                }
                KVLayout::HeadMajor => {
                    // Per-head copy: capacity changes head_stride
                    let old_head_stride = self.capacity * self.head_dim;
                    let new_head_stride = new_cap * self.head_dim;
                    let copy_per_head = self.current_pos * self.head_dim;

                    if dtype == crate::core::buffer::DType::Q4_0 {
                        let qk = crate::core::quant::QK4_0;
                        let old_hs = old_head_stride / qk;
                        let new_hs = new_head_stride / qk;
                        let cph = copy_per_head / qk;
                        for h in 0..self.kv_heads {
                            backend.copy_slice(
                                &self.k_buffer,
                                &mut new_k,
                                h * old_hs,
                                h * new_hs,
                                cph,
                            )?;
                            backend.copy_slice(
                                &self.v_buffer,
                                &mut new_v,
                                h * old_hs,
                                h * new_hs,
                                cph,
                            )?;
                        }
                    } else {
                        for h in 0..self.kv_heads {
                            backend.copy_slice(
                                &self.k_buffer,
                                &mut new_k,
                                h * old_head_stride,
                                h * new_head_stride,
                                copy_per_head,
                            )?;
                            backend.copy_slice(
                                &self.v_buffer,
                                &mut new_v,
                                h * old_head_stride,
                                h * new_head_stride,
                                copy_per_head,
                            )?;
                        }
                    }
                }
            }
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

    /// Append new K/V data to the cache.
    ///
    /// Input tensors are always seq-major `[batch, seq_len, kv_heads, head_dim]`
    /// (from the model's matmul output). For HeadMajor layout, this scatters
    /// per-head data to the correct positions.
    pub fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        let seq_len = new_k.shape().dims()[1];

        if self.current_pos + seq_len > self.capacity {
            if self.current_pos + seq_len > self.max_seq_len {
                return Err(anyhow::anyhow!("KV Cache overflow"));
            }
            self.grow(self.current_pos + seq_len)?;
        }

        let backend = self.k_buffer.backend().clone();
        let is_q4 = self.k_buffer.dtype() == crate::core::buffer::DType::Q4_0;

        match self.layout {
            KVLayout::SeqMajor => {
                // Contiguous: single copy per buffer
                let (offset, count) = if is_q4 {
                    let bpp = self.kv_heads * self.head_dim / crate::core::quant::QK4_0;
                    (self.current_pos * bpp, seq_len * bpp)
                } else {
                    let row = self.kv_heads * self.head_dim;
                    (self.current_pos * row, seq_len * row)
                };
                backend.copy_slice(new_k, &mut self.k_buffer, 0, offset, count)?;
                backend.copy_slice(new_v, &mut self.v_buffer, 0, offset, count)?;
            }
            KVLayout::HeadMajor => {
                // Input is seq-major: [batch, seq_len, kv_heads, head_dim]
                // Scatter to head-major: each head's data at head * capacity * head_dim + pos * head_dim
                if is_q4 {
                    let qk = crate::core::quant::QK4_0;
                    let src_row = self.kv_heads * self.head_dim / qk;
                    let dst_head_stride = self.capacity * self.head_dim / qk;
                    let blocks_per_head = self.head_dim / qk;
                    let dst_pos_blocks = self.current_pos * blocks_per_head;

                    for s in 0..seq_len {
                        for h in 0..self.kv_heads {
                            let src_off = s * src_row + h * blocks_per_head;
                            let dst_off =
                                h * dst_head_stride + dst_pos_blocks + s * blocks_per_head;
                            backend.copy_slice(
                                new_k,
                                &mut self.k_buffer,
                                src_off,
                                dst_off,
                                blocks_per_head,
                            )?;
                            backend.copy_slice(
                                new_v,
                                &mut self.v_buffer,
                                src_off,
                                dst_off,
                                blocks_per_head,
                            )?;
                        }
                    }
                } else {
                    // Direct memcpy for CPU buffers (avoids Backend::copy_slice overhead).
                    // For OpenCL buffers (as_ptr may be null or host-mapped),
                    // fall back to backend.copy_slice which handles GPU↔GPU copy.
                    let type_size = match self.k_buffer.dtype() {
                        crate::core::buffer::DType::F16 => 2,
                        crate::core::buffer::DType::F32 => 4,
                        _ => 0,
                    };

                    // Direct CPU memcpy is faster than GPU enqueue_copy_buffer for
                    // tiny copies (128 bytes per head). On ARM UMA (CL_MEM_ALLOC_HOST_PTR),
                    // host pointers directly access GPU memory without DMA transfer.
                    let k_dst = self.k_buffer.as_mut_ptr();
                    let k_src = new_k.as_ptr();
                    let can_direct_copy = type_size > 0 && !k_dst.is_null() && !k_src.is_null();

                    let src_row = self.kv_heads * self.head_dim;
                    let dst_head_stride = self.capacity * self.head_dim;

                    if can_direct_copy {
                        let bytes_per_head = self.head_dim * type_size;
                        let v_src = new_v.as_ptr();
                        let v_dst = self.v_buffer.as_mut_ptr();

                        unsafe {
                            for s in 0..seq_len {
                                for h in 0..self.kv_heads {
                                    let src_byte = (s * src_row + h * self.head_dim) * type_size;
                                    let dst_byte = (h * dst_head_stride
                                        + (self.current_pos + s) * self.head_dim)
                                        * type_size;
                                    std::ptr::copy_nonoverlapping(
                                        k_src.add(src_byte),
                                        k_dst.add(dst_byte),
                                        bytes_per_head,
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        v_src.add(src_byte),
                                        v_dst.add(dst_byte),
                                        bytes_per_head,
                                    );
                                }
                            }
                        }
                    } else {
                        for s in 0..seq_len {
                            for h in 0..self.kv_heads {
                                let src_off = s * src_row + h * self.head_dim;
                                let dst_off =
                                    h * dst_head_stride + (self.current_pos + s) * self.head_dim;
                                backend.copy_slice(
                                    new_k,
                                    &mut self.k_buffer,
                                    src_off,
                                    dst_off,
                                    self.head_dim,
                                )?;
                                backend.copy_slice(
                                    new_v,
                                    &mut self.v_buffer,
                                    src_off,
                                    dst_off,
                                    self.head_dim,
                                )?;
                            }
                        }
                    }
                }
            }
        }

        self.current_pos += seq_len;
        self.high_water_pos = self.high_water_pos.max(self.current_pos);

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

        let remaining = self.current_pos - count;

        if remaining == 0 {
            self.current_pos = 0;
            self.high_water_pos = 0;
            return Ok(());
        }

        let backend = self.k_buffer.backend().clone();
        let is_q4 = self.k_buffer.dtype() == crate::core::buffer::DType::Q4_0;

        match self.layout {
            KVLayout::SeqMajor => {
                let (src_offset, move_count) = if is_q4 {
                    let bpp = self.kv_heads * self.head_dim / crate::core::quant::QK4_0;
                    (count * bpp, remaining * bpp)
                } else {
                    let epp = self.kv_heads * self.head_dim;
                    (count * epp, remaining * epp)
                };
                backend.buffer_shift(&mut self.k_buffer, src_offset, 0, move_count)?;
                backend.buffer_shift(&mut self.v_buffer, src_offset, 0, move_count)?;
            }
            KVLayout::HeadMajor => {
                // Per-head shift: each head's data is contiguous
                if is_q4 {
                    let qk = crate::core::quant::QK4_0;
                    let head_stride = self.capacity * self.head_dim / qk;
                    let bph = self.head_dim / qk; // blocks per head per position
                    for h in 0..self.kv_heads {
                        let base = h * head_stride;
                        backend.buffer_shift(
                            &mut self.k_buffer,
                            base + count * bph,
                            base,
                            remaining * bph,
                        )?;
                        backend.buffer_shift(
                            &mut self.v_buffer,
                            base + count * bph,
                            base,
                            remaining * bph,
                        )?;
                    }
                } else {
                    let head_stride = self.capacity * self.head_dim;
                    for h in 0..self.kv_heads {
                        let base = h * head_stride;
                        backend.buffer_shift(
                            &mut self.k_buffer,
                            base + count * self.head_dim,
                            base,
                            remaining * self.head_dim,
                        )?;
                        backend.buffer_shift(
                            &mut self.v_buffer,
                            base + count * self.head_dim,
                            base,
                            remaining * self.head_dim,
                        )?;
                    }
                }
            }
        }

        self.current_pos = remaining;
        self.release_unused_pages();
        Ok(())
    }

    /// Returns the memory usage in bytes for currently stored KV data.
    pub fn memory_usage_bytes(&self) -> usize {
        let is_q4 = self.k_buffer.dtype() == crate::core::buffer::DType::Q4_0;

        let per_buffer = if is_q4 {
            let blocks_per_pos = self.kv_heads * self.head_dim / crate::core::quant::QK4_0;
            let block_size = std::mem::size_of::<crate::core::quant::BlockQ4_0>();
            self.current_pos * blocks_per_pos * block_size
        } else {
            let type_size = self.k_buffer.dtype().size();
            self.current_pos * self.kv_heads * self.head_dim * type_size
        };

        per_buffer * 2 // K + V
    }

    /// Layout-aware position shift for eviction policies.
    ///
    /// Moves `count` positions worth of data from `src_pos` to `dst_pos`
    /// within both K and V buffers. Handles per-head scattering for HeadMajor.
    pub fn shift_positions(&mut self, src_pos: usize, dst_pos: usize, count: usize) -> Result<()> {
        if count == 0 || src_pos == dst_pos {
            return Ok(());
        }

        let backend = self.k_buffer.backend().clone();
        let is_q4 = self.k_buffer.dtype() == crate::core::buffer::DType::Q4_0;

        match self.layout {
            KVLayout::SeqMajor => {
                let (src_off, dst_off, move_count) = if is_q4 {
                    let bpp = self.kv_heads * self.head_dim / crate::core::quant::QK4_0;
                    (src_pos * bpp, dst_pos * bpp, count * bpp)
                } else {
                    let epp = self.kv_heads * self.head_dim;
                    (src_pos * epp, dst_pos * epp, count * epp)
                };
                backend.buffer_shift(&mut self.k_buffer, src_off, dst_off, move_count)?;
                backend.buffer_shift(&mut self.v_buffer, src_off, dst_off, move_count)?;
            }
            KVLayout::HeadMajor => {
                if is_q4 {
                    let qk = crate::core::quant::QK4_0;
                    let head_stride = self.capacity * self.head_dim / qk;
                    let bph = self.head_dim / qk;
                    for h in 0..self.kv_heads {
                        let base = h * head_stride;
                        backend.buffer_shift(
                            &mut self.k_buffer,
                            base + src_pos * bph,
                            base + dst_pos * bph,
                            count * bph,
                        )?;
                        backend.buffer_shift(
                            &mut self.v_buffer,
                            base + src_pos * bph,
                            base + dst_pos * bph,
                            count * bph,
                        )?;
                    }
                } else {
                    let head_stride = self.capacity * self.head_dim;
                    for h in 0..self.kv_heads {
                        let base = h * head_stride;
                        backend.buffer_shift(
                            &mut self.k_buffer,
                            base + src_pos * self.head_dim,
                            base + dst_pos * self.head_dim,
                            count * self.head_dim,
                        )?;
                        backend.buffer_shift(
                            &mut self.v_buffer,
                            base + src_pos * self.head_dim,
                            base + dst_pos * self.head_dim,
                            count * self.head_dim,
                        )?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Move `count` positions from `src_pos` to `dst_pos` for a **single** KV head.
    ///
    /// Unlike `shift_positions()` which moves the same positions for all heads,
    /// this method only modifies data belonging to the specified `head`.
    /// Used by per-head eviction policies (e.g., H2O+) where each KV head
    /// independently selects which tokens to keep.
    ///
    /// **Requires HeadMajor layout** — panics on SeqMajor because per-head
    /// data is interleaved and cannot be moved independently.
    pub fn shift_positions_for_head(
        &mut self,
        head: usize,
        src_pos: usize,
        dst_pos: usize,
        count: usize,
    ) -> Result<()> {
        assert_eq!(
            self.layout,
            KVLayout::HeadMajor,
            "shift_positions_for_head requires HeadMajor layout"
        );

        if count == 0 || src_pos == dst_pos {
            return Ok(());
        }

        let backend = self.k_buffer.backend().clone();
        let is_q4 = self.k_buffer.dtype() == crate::core::buffer::DType::Q4_0;

        if is_q4 {
            let qk = crate::core::quant::QK4_0;
            let head_stride = self.capacity * self.head_dim / qk;
            let bph = self.head_dim / qk;
            let base = head * head_stride;
            backend.buffer_shift(
                &mut self.k_buffer,
                base + src_pos * bph,
                base + dst_pos * bph,
                count * bph,
            )?;
            backend.buffer_shift(
                &mut self.v_buffer,
                base + src_pos * bph,
                base + dst_pos * bph,
                count * bph,
            )?;
        } else {
            let head_stride = self.capacity * self.head_dim;
            let base = head * head_stride;
            backend.buffer_shift(
                &mut self.k_buffer,
                base + src_pos * self.head_dim,
                base + dst_pos * self.head_dim,
                count * self.head_dim,
            )?;
            backend.buffer_shift(
                &mut self.v_buffer,
                base + src_pos * self.head_dim,
                base + dst_pos * self.head_dim,
                count * self.head_dim,
            )?;
        }

        Ok(())
    }

    /// Release physical pages for the unused portion of KV buffers via `madvise(MADV_DONTNEED)`.
    ///
    /// After eviction reduces `current_pos`, the buffer region beyond `current_pos`
    /// is no longer needed. This method advises the OS to reclaim those physical pages,
    /// reducing RSS and increasing MemAvailable — without reallocating the buffer.
    ///
    /// Only applies to CPU heap buffers (`SharedBuffer` / `Vec<u8>`).
    /// GPU-managed buffers (`UnifiedBuffer` with `CL_MEM_ALLOC_HOST_PTR`) are skipped
    /// because their physical pages are pinned by the OpenCL driver.
    ///
    /// Returns the total bytes released (page-aligned) across K and V buffers.
    pub fn release_unused_pages(&mut self) -> usize {
        // Guard: GPU-managed buffer — driver pins pages, madvise is ineffective
        if self.k_buffer.buffer().cl_mem().is_some() {
            return 0;
        }

        let type_size = match self.k_buffer.dtype() {
            DType::F16 => 2,
            DType::F32 => 4,
            // Q4_0/Q4_1: block-quantized layout, skip for now (F16-only KV cache in practice)
            _ => return 0,
        };

        let hwm = self.high_water_pos;
        let mut total_released = 0usize;

        match self.layout {
            KVLayout::SeqMajor => {
                // Contiguous: [pos0_all_heads | pos1_all_heads | ... | posN_all_heads]
                // Only advise up to high_water_pos — never-touched tail pages are already free.
                let row_bytes = self.kv_heads * self.head_dim * type_size;
                let used_bytes = self.current_pos * row_bytes;
                let hwm_bytes = hwm * row_bytes;
                total_released += madvise_dontneed(self.k_buffer.as_ptr(), used_bytes, hwm_bytes);
                total_released += madvise_dontneed(self.v_buffer.as_ptr(), used_bytes, hwm_bytes);
            }
            KVLayout::HeadMajor => {
                // Per-head: [head0: pos0..posN | head1: pos0..posN | ...]
                let head_stride_bytes = self.capacity * self.head_dim * type_size;
                let used_per_head_bytes = self.current_pos * self.head_dim * type_size;
                let hwm_per_head_bytes = hwm * self.head_dim * type_size;
                for h in 0..self.kv_heads {
                    let base = h * head_stride_bytes;
                    let from = base + used_per_head_bytes;
                    let to = base + hwm_per_head_bytes;
                    total_released += madvise_dontneed(self.k_buffer.as_ptr(), from, to);
                    total_released += madvise_dontneed(self.v_buffer.as_ptr(), from, to);
                }
            }
        }

        if total_released > 0 {
            log::debug!(
                "[KVCache] released {} bytes of unused pages (pos={}, hwm={}, cap={})",
                total_released,
                self.current_pos,
                hwm,
                self.capacity,
            );
        }

        // After releasing, high_water_pos shrinks to current_pos.
        // Pages between current_pos and old hwm have been advised away.
        self.high_water_pos = self.current_pos;

        total_released
    }

    /// Compact the KV cache by moving `keep` positions to a contiguous region
    /// starting at `write_start`.
    ///
    /// `keep` must be sorted in ascending order. Consecutive source positions are
    /// merged into a single `shift_positions` call, reducing the number of
    /// `buffer_shift` invocations by up to 200x compared to per-token shifting.
    ///
    /// This method does **not** update `current_pos`; the caller is responsible
    /// for setting it to `write_start + keep.len()` after compaction.
    pub fn compact_keep_positions(&mut self, keep: &[usize], write_start: usize) -> Result<()> {
        if keep.is_empty() {
            return Ok(());
        }

        let mut write_pos = write_start;
        let mut batch_src_start = keep[0];
        let mut batch_dst_start = write_pos;
        let mut batch_count = 1usize;
        write_pos += 1;

        for &src_pos in &keep[1..] {
            if src_pos == batch_src_start + batch_count {
                // Extend current batch
                batch_count += 1;
            } else {
                // Flush current batch
                if batch_src_start != batch_dst_start {
                    self.shift_positions(batch_src_start, batch_dst_start, batch_count)?;
                }
                // Start new batch
                batch_src_start = src_pos;
                batch_dst_start = write_pos;
                batch_count = 1;
            }
            write_pos += 1;
        }

        // Flush final batch
        if batch_src_start != batch_dst_start {
            self.shift_positions(batch_src_start, batch_dst_start, batch_count)?;
        }

        Ok(())
    }

    /// Per-head variant of `compact_keep_positions` for H2O+ per-head eviction.
    ///
    /// Only moves data for the specified `head`; other heads are untouched.
    /// `keep` must be sorted in ascending order.
    ///
    /// **Requires HeadMajor layout** — delegates to `shift_positions_for_head`.
    pub fn compact_keep_positions_for_head(
        &mut self,
        head: usize,
        keep: &[usize],
        write_start: usize,
    ) -> Result<()> {
        if keep.is_empty() {
            return Ok(());
        }

        let mut write_pos = write_start;
        let mut batch_src_start = keep[0];
        let mut batch_dst_start = write_pos;
        let mut batch_count = 1usize;
        write_pos += 1;

        for &src_pos in &keep[1..] {
            if src_pos == batch_src_start + batch_count {
                // Extend current batch
                batch_count += 1;
            } else {
                // Flush current batch
                if batch_src_start != batch_dst_start {
                    self.shift_positions_for_head(
                        head,
                        batch_src_start,
                        batch_dst_start,
                        batch_count,
                    )?;
                }
                // Start new batch
                batch_src_start = src_pos;
                batch_dst_start = write_pos;
                batch_count = 1;
            }
            write_pos += 1;
        }

        // Flush final batch
        if batch_src_start != batch_dst_start {
            self.shift_positions_for_head(head, batch_src_start, batch_dst_start, batch_count)?;
        }

        Ok(())
    }
}

// ── madvise helpers ─────────────────────────────────────────────────────────

/// Advise the OS to release physical pages in `[from_offset..to_offset)` relative to `base_ptr`.
///
/// Addresses are rounded to page boundaries (start up, end down) to satisfy madvise requirements.
/// Returns the number of bytes actually advised for release.
fn madvise_dontneed(base_ptr: *const u8, from_offset: usize, to_offset: usize) -> usize {
    if from_offset >= to_offset || base_ptr.is_null() {
        return 0;
    }

    let page_size = page_size();
    let abs_start = base_ptr as usize + from_offset;
    let abs_end = base_ptr as usize + to_offset;

    // Round start UP (don't release partially-used page), end DOWN
    let aligned_start = round_up(abs_start, page_size);
    let aligned_end = round_down(abs_end, page_size);

    if aligned_start >= aligned_end {
        return 0;
    }

    let len = aligned_end - aligned_start;
    // SAFETY: The range [aligned_start..aligned_end) lies within the buffer's allocation.
    // MADV_DONTNEED on anonymous private mappings releases physical pages; re-access
    // triggers zero-fill page faults (safe for KV cache — overwritten before read).
    let ret =
        unsafe { libc::madvise(aligned_start as *mut libc::c_void, len, libc::MADV_DONTNEED) };
    if ret == 0 { len } else { 0 }
}

#[inline]
fn round_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

#[inline]
fn round_down(x: usize, align: usize) -> usize {
    x & !(align - 1)
}

/// Cached page size (4096 on most ARM64/x86_64 systems).
fn page_size() -> usize {
    static PAGE_SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGE_SIZE.get_or_init(|| {
        // SAFETY: sysconf(_SC_PAGESIZE) is always safe and returns a positive value on Linux.
        let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if ps > 0 { ps as usize } else { 4096 }
    })
}

// ── KVCacheOps implementation for KVCache ───────────────────────────────────

impl KVCacheOps for KVCache {
    fn current_pos(&self) -> usize {
        self.current_pos
    }

    fn set_current_pos(&mut self, pos: usize) {
        self.current_pos = pos;
        if pos == 0 {
            self.high_water_pos = 0;
        }
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn layout(&self) -> KVLayout {
        self.layout
    }

    fn kv_dtype(&self) -> DType {
        self.k_buffer.dtype()
    }

    fn memory_usage_bytes(&self) -> usize {
        self.memory_usage_bytes()
    }

    fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        self.update(new_k, new_v)
    }

    fn get_view(&mut self) -> (Tensor, Tensor) {
        (self.k_buffer.clone(), self.v_buffer.clone())
    }

    fn get_buffers_mut(&mut self) -> Option<(&mut Tensor, &mut Tensor)> {
        Some((&mut self.k_buffer, &mut self.v_buffer))
    }

    fn advance_pos(&mut self, n: usize) {
        self.current_pos += n;
        self.high_water_pos = self.high_water_pos.max(self.current_pos);
    }

    fn ensure_capacity(&mut self, min_tokens: usize) -> Result<bool> {
        if min_tokens <= self.capacity {
            return Ok(false);
        }
        if min_tokens > self.max_seq_len {
            return Err(anyhow::anyhow!(
                "KV Cache overflow: need {} tokens but max_seq_len={}",
                min_tokens,
                self.max_seq_len
            ));
        }
        self.grow(min_tokens)?;
        Ok(true)
    }
}

/// Get the maximum current_pos across all KV caches.
/// Returns 0 if the slice is empty.
#[inline]
pub fn max_cache_pos(caches: &[KVCache]) -> usize {
    caches.iter().map(|c| c.current_pos).max().unwrap_or(0)
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
    fn test_ensure_capacity_grows_when_needed() {
        let heads = 1;
        let dim = 4;
        let mut cache = make_dynamic_cache(4, 64, heads, dim);

        // Fill to capacity
        for i in 0..4 {
            let (k, v) = make_token_tensor(i as f32, heads, dim);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.capacity(), 4);
        assert_eq!(cache.current_pos(), 4);

        // ensure_capacity within current → no grow, returns false
        assert_eq!(cache.ensure_capacity(3).unwrap(), false);
        assert_eq!(cache.capacity(), 4);

        // ensure_capacity exactly at boundary → no grow
        assert_eq!(cache.ensure_capacity(4).unwrap(), false);

        // ensure_capacity beyond → grow, returns true
        assert_eq!(cache.ensure_capacity(5).unwrap(), true);
        assert!(cache.capacity() >= 5);

        // Data integrity after grow
        let k_data = cache.k_buffer.as_slice::<f32>();
        assert_eq!(k_data[0], 0.0);
        assert_eq!(k_data[dim], 1.0);
    }

    #[test]
    fn test_ensure_capacity_overflow() {
        let heads = 1;
        let dim = 4;
        let mut cache = make_dynamic_cache(4, 8, heads, dim);

        // Beyond max_seq_len → error
        assert!(cache.ensure_capacity(9).is_err());
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

    // ── Phase 0: Layout accessor tests ──

    #[test]
    fn test_layout_default_is_seq_major() {
        let cache = make_cache(64, 4, 8);
        assert_eq!(cache.layout(), KVLayout::SeqMajor);
    }

    #[test]
    fn test_offset_seq_major() {
        let cache = make_cache(64, 4, 8); // heads=4, dim=8, capacity=64
        // SeqMajor: pos * kv_heads * head_dim + head * head_dim
        assert_eq!(cache.offset(0, 0), 0);
        assert_eq!(cache.offset(0, 1), 8);
        assert_eq!(cache.offset(0, 3), 24);
        assert_eq!(cache.offset(1, 0), 32); // 1 * 4 * 8
        assert_eq!(cache.offset(5, 2), 5 * 32 + 16);
    }

    #[test]
    fn test_offset_head_major() {
        let cache = make_head_major_cache(64, 4, 8);

        // HeadMajor: head * capacity * head_dim + pos * head_dim
        assert_eq!(cache.offset(0, 0), 0);
        assert_eq!(cache.offset(1, 0), 8); // pos=1, head=0 → 0 + 1*8
        assert_eq!(cache.offset(0, 1), 512); // head=1 → 1 * 64 * 8
        assert_eq!(cache.offset(5, 2), 2 * 512 + 5 * 8);
    }

    #[test]
    fn test_strides_seq_major() {
        let cache = make_cache(64, 4, 8);
        assert_eq!(cache.pos_stride(), 32); // kv_heads * head_dim
        assert_eq!(cache.head_stride(), 8); // head_dim
    }

    #[test]
    fn test_strides_head_major() {
        let cache = make_head_major_cache(64, 4, 8);

        assert_eq!(cache.pos_stride(), 8); // head_dim
        assert_eq!(cache.head_stride(), 512); // capacity * head_dim
    }

    #[test]
    fn test_accessors() {
        let cache = make_cache(64, 4, 8);
        assert_eq!(cache.kv_heads(), 4);
        assert_eq!(cache.head_dim(), 8);
    }

    // ── Phase 1: HeadMajor internal operations ──

    fn make_head_major_cache(max_seq_len: usize, heads: usize, dim: usize) -> KVCache {
        let size_bytes = max_seq_len * heads * dim * 4;
        let k_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let backend = Arc::new(CpuBackend::new());
        let k = Tensor::new(
            Shape::new(vec![1, heads, max_seq_len, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, heads, max_seq_len, dim]), v_buf, backend);
        KVCache {
            k_buffer: k,
            v_buffer: v,
            current_pos: 0,
            high_water_pos: 0,
            max_seq_len,
            capacity: max_seq_len,
            kv_heads: heads,
            head_dim: dim,
            layout: KVLayout::HeadMajor,
            memory: None,
        }
    }

    fn make_head_major_dynamic_cache(
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
            Shape::new(vec![1, heads, initial_capacity, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, heads, initial_capacity, dim]),
            v_buf,
            backend,
        );
        KVCache {
            k_buffer: k,
            v_buffer: v,
            current_pos: 0,
            high_water_pos: 0,
            max_seq_len,
            capacity: initial_capacity,
            kv_heads: heads,
            head_dim: dim,
            layout: KVLayout::HeadMajor,
            memory: Some(memory),
        }
    }

    /// Read a value from head-major cache at (pos, head, d=0)
    fn hm_read(cache: &KVCache, pos: usize, head: usize) -> f32 {
        let off = cache.offset(pos, head);
        cache.k_buffer.as_slice::<f32>()[off]
    }

    fn hm_read_v(cache: &KVCache, pos: usize, head: usize) -> f32 {
        let off = cache.offset(pos, head);
        cache.v_buffer.as_slice::<f32>()[off]
    }

    #[test]
    fn test_hm_update_single_token() {
        let heads = 2;
        let dim = 4;
        let mut cache = make_head_major_cache(16, heads, dim);

        // Input: seq-major [1, 1, 2, 4] — head0=[1,1,1,1], head1=[2,2,2,2]
        let backend = Arc::new(CpuBackend::new());
        let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
        unsafe {
            let ptr = buf.as_mut_ptr() as *mut f32;
            for d in 0..dim {
                *ptr.add(d) = 1.0;
            } // head 0
            for d in 0..dim {
                *ptr.add(dim + d) = 2.0;
            } // head 1
        }
        let k = Tensor::new(
            Shape::new(vec![1, 1, heads, dim]),
            buf.clone(),
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend);
        cache.update(&k, &v).unwrap();

        assert_eq!(cache.current_pos, 1);
        // Head 0, pos 0 should be 1.0
        assert_eq!(hm_read(&cache, 0, 0), 1.0);
        // Head 1, pos 0 should be 2.0
        assert_eq!(hm_read(&cache, 0, 1), 2.0);
    }

    #[test]
    fn test_hm_update_multi_token() {
        let heads = 2;
        let dim = 2;
        let mut cache = make_head_major_cache(16, heads, dim);

        // 3 tokens: seq-major [1,1,3,2]
        let backend = Arc::new(CpuBackend::new());
        let n = 3 * heads * dim;
        let buf = Arc::new(SharedBuffer::new(n * 4, DType::F32));
        unsafe {
            let ptr = buf.as_mut_ptr() as *mut f32;
            // Token 0: h0=[10,11], h1=[20,21]
            *ptr.add(0) = 10.0;
            *ptr.add(1) = 11.0;
            *ptr.add(2) = 20.0;
            *ptr.add(3) = 21.0;
            // Token 1: h0=[30,31], h1=[40,41]
            *ptr.add(4) = 30.0;
            *ptr.add(5) = 31.0;
            *ptr.add(6) = 40.0;
            *ptr.add(7) = 41.0;
            // Token 2: h0=[50,51], h1=[60,61]
            *ptr.add(8) = 50.0;
            *ptr.add(9) = 51.0;
            *ptr.add(10) = 60.0;
            *ptr.add(11) = 61.0;
        }
        let k = Tensor::new(
            Shape::new(vec![1, 3, heads, dim]),
            buf.clone(),
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, 3, heads, dim]), buf, backend);
        cache.update(&k, &v).unwrap();

        assert_eq!(cache.current_pos, 3);
        // Head 0: [10, 30, 50] at positions 0, 1, 2
        assert_eq!(hm_read(&cache, 0, 0), 10.0);
        assert_eq!(hm_read(&cache, 1, 0), 30.0);
        assert_eq!(hm_read(&cache, 2, 0), 50.0);
        // Head 1: [20, 40, 60] at positions 0, 1, 2
        assert_eq!(hm_read(&cache, 0, 1), 20.0);
        assert_eq!(hm_read(&cache, 1, 1), 40.0);
        assert_eq!(hm_read(&cache, 2, 1), 60.0);
    }

    #[test]
    fn test_hm_prune_prefix() {
        let heads = 2;
        let dim = 2;
        let mut cache = make_head_major_cache(16, heads, dim);

        // Insert 5 tokens with distinct values per head
        let backend = Arc::new(CpuBackend::new());
        for i in 0..5 {
            let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut f32;
                for d in 0..dim {
                    *ptr.add(d) = (i * 10 + d) as f32;
                }
                for d in 0..dim {
                    *ptr.add(dim + d) = (i * 100 + d) as f32;
                }
            }
            let k = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                buf.clone(),
                backend.clone(),
            );
            let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend.clone());
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 5);

        // Head 0 pos 0: [0,1], pos 2: [20,21]
        assert_eq!(hm_read(&cache, 0, 0), 0.0);
        assert_eq!(hm_read(&cache, 2, 0), 20.0);

        // Prune first 2 tokens
        cache.prune_prefix(2).unwrap();
        assert_eq!(cache.current_pos, 3);

        // After prune: old pos 2 is now pos 0
        // Head 0: [20, 30, 40]
        assert_eq!(hm_read(&cache, 0, 0), 20.0);
        assert_eq!(hm_read(&cache, 1, 0), 30.0);
        assert_eq!(hm_read(&cache, 2, 0), 40.0);
        // Head 1: [200, 300, 400]
        assert_eq!(hm_read(&cache, 0, 1), 200.0);
        assert_eq!(hm_read(&cache, 1, 1), 300.0);
        assert_eq!(hm_read(&cache, 2, 1), 400.0);
    }

    #[test]
    fn test_hm_dynamic_growth() {
        let heads = 2;
        let dim = 2;
        let mut cache = make_head_major_dynamic_cache(4, 64, heads, dim);
        assert_eq!(cache.capacity(), 4);

        // Fill 4 tokens
        let backend = Arc::new(CpuBackend::new());
        for i in 0..4 {
            let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut f32;
                for d in 0..dim {
                    *ptr.add(d) = (i + 1) as f32;
                }
                for d in 0..dim {
                    *ptr.add(dim + d) = ((i + 1) * 10) as f32;
                }
            }
            let k = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                buf.clone(),
                backend.clone(),
            );
            let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend.clone());
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.capacity(), 4);

        // 5th token triggers grow
        let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
        unsafe {
            let ptr = buf.as_mut_ptr() as *mut f32;
            for d in 0..dim {
                *ptr.add(d) = 5.0;
            }
            for d in 0..dim {
                *ptr.add(dim + d) = 50.0;
            }
        }
        let k = Tensor::new(
            Shape::new(vec![1, 1, heads, dim]),
            buf.clone(),
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend);
        cache.update(&k, &v).unwrap();
        assert!(cache.capacity() >= 5);
        assert_eq!(cache.current_pos, 5);

        // Verify data integrity after growth
        assert_eq!(hm_read(&cache, 0, 0), 1.0);
        assert_eq!(hm_read(&cache, 3, 0), 4.0);
        assert_eq!(hm_read(&cache, 4, 0), 5.0);
        assert_eq!(hm_read(&cache, 0, 1), 10.0);
        assert_eq!(hm_read(&cache, 4, 1), 50.0);
    }

    #[test]
    fn test_hm_shift_positions() {
        let heads = 2;
        let dim = 2;
        let mut cache = make_head_major_cache(16, heads, dim);

        let backend = Arc::new(CpuBackend::new());
        for i in 0..5 {
            let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut f32;
                for d in 0..dim {
                    *ptr.add(d) = (i * 10 + d) as f32;
                }
                for d in 0..dim {
                    *ptr.add(dim + d) = (i * 100 + d) as f32;
                }
            }
            let k = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                buf.clone(),
                backend.clone(),
            );
            let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend.clone());
            cache.update(&k, &v).unwrap();
        }

        // shift_positions: move positions 3..5 to 1..3
        cache.shift_positions(3, 1, 2).unwrap();

        // Head 0: pos 1 now has old pos 3 value (30)
        assert_eq!(hm_read(&cache, 1, 0), 30.0);
        assert_eq!(hm_read(&cache, 2, 0), 40.0);
        // Head 1: pos 1 now has old pos 3 value (300)
        assert_eq!(hm_read(&cache, 1, 1), 300.0);
        assert_eq!(hm_read(&cache, 2, 1), 400.0);
    }

    /// Cross-layout: verify SeqMajor and HeadMajor produce same logical data
    #[test]
    fn test_cross_layout_equivalence() {
        let heads = 2;
        let dim = 4;
        let backend = Arc::new(CpuBackend::new());

        let mut sm_cache = make_cache(16, heads, dim);
        let mut hm_cache = make_head_major_cache(16, heads, dim);

        // Insert same 5 tokens into both
        for i in 0..5 {
            let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut f32;
                for d in 0..(heads * dim) {
                    *ptr.add(d) = (i * 100 + d) as f32;
                }
            }
            let k = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                buf.clone(),
                backend.clone(),
            );
            let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend.clone());
            sm_cache.update(&k, &v).unwrap();

            // Same data for hm
            let buf2 = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
            unsafe {
                let ptr = buf2.as_mut_ptr() as *mut f32;
                for d in 0..(heads * dim) {
                    *ptr.add(d) = (i * 100 + d) as f32;
                }
            }
            let k2 = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                buf2.clone(),
                backend.clone(),
            );
            let v2 = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf2, backend.clone());
            hm_cache.update(&k2, &v2).unwrap();
        }

        // Compare logical values at each (pos, head, d)
        let sm_k = sm_cache.k_buffer.as_slice::<f32>();
        let hm_k = hm_cache.k_buffer.as_slice::<f32>();
        for pos in 0..5 {
            for h in 0..heads {
                for d in 0..dim {
                    let sm_off = sm_cache.offset(pos, h) + d;
                    let hm_off = hm_cache.offset(pos, h) + d;
                    assert_eq!(
                        sm_k[sm_off], hm_k[hm_off],
                        "Mismatch at pos={}, head={}, d={}",
                        pos, h, d
                    );
                }
            }
        }

        // Prune and compare
        sm_cache.prune_prefix(2).unwrap();
        hm_cache.prune_prefix(2).unwrap();
        let sm_k = sm_cache.k_buffer.as_slice::<f32>();
        let hm_k = hm_cache.k_buffer.as_slice::<f32>();
        for pos in 0..3 {
            for h in 0..heads {
                for d in 0..dim {
                    let sm_off = sm_cache.offset(pos, h) + d;
                    let hm_off = hm_cache.offset(pos, h) + d;
                    assert_eq!(
                        sm_k[sm_off], hm_k[hm_off],
                        "After prune: mismatch at pos={}, head={}, d={}",
                        pos, h, d
                    );
                }
            }
        }
    }

    // ── Per-head shift tests ──

    #[test]
    fn test_shift_positions_for_head_basic() {
        let heads = 2;
        let dim = 2;
        let mut cache = make_head_major_cache(16, heads, dim);

        let backend = Arc::new(CpuBackend::new());
        for i in 0..5 {
            let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut f32;
                for d in 0..dim {
                    *ptr.add(d) = (i * 10 + d) as f32;
                }
                for d in 0..dim {
                    *ptr.add(dim + d) = (i * 100 + d) as f32;
                }
            }
            let k = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                buf.clone(),
                backend.clone(),
            );
            let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend.clone());
            cache.update(&k, &v).unwrap();
        }
        // Head 0: pos0=0, pos1=10, pos2=20, pos3=30, pos4=40
        // Head 1: pos0=0, pos1=100, pos2=200, pos3=300, pos4=400

        // Move pos 3 → pos 1 for HEAD 0 ONLY
        cache.shift_positions_for_head(0, 3, 1, 1).unwrap();

        // Head 0: pos 1 should now have value 30 (from pos 3)
        assert_eq!(hm_read(&cache, 1, 0), 30.0);
        // Head 1: pos 1 should be UNCHANGED (still 100)
        assert_eq!(hm_read(&cache, 1, 1), 100.0);

        // Head 0: pos 0 should be unchanged
        assert_eq!(hm_read(&cache, 0, 0), 0.0);
        // Head 1: pos 3 should be unchanged
        assert_eq!(hm_read(&cache, 3, 1), 300.0);
    }

    #[test]
    fn test_shift_positions_for_head_multi_count() {
        let heads = 2;
        let dim = 2;
        let mut cache = make_head_major_cache(16, heads, dim);

        let backend = Arc::new(CpuBackend::new());
        for i in 0..6 {
            let buf = Arc::new(SharedBuffer::new(heads * dim * 4, DType::F32));
            unsafe {
                let ptr = buf.as_mut_ptr() as *mut f32;
                for d in 0..dim {
                    *ptr.add(d) = (i * 10) as f32;
                }
                for d in 0..dim {
                    *ptr.add(dim + d) = (i * 100) as f32;
                }
            }
            let k = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                buf.clone(),
                backend.clone(),
            );
            let v = Tensor::new(Shape::new(vec![1, 1, heads, dim]), buf, backend.clone());
            cache.update(&k, &v).unwrap();
        }

        // Move positions 4..6 → 2..4 for head 1 only
        cache.shift_positions_for_head(1, 4, 2, 2).unwrap();

        // Head 1: pos 2 = old pos 4 = 400, pos 3 = old pos 5 = 500
        assert_eq!(hm_read(&cache, 2, 1), 400.0);
        assert_eq!(hm_read(&cache, 3, 1), 500.0);
        // Head 0: unchanged at pos 2, 3
        assert_eq!(hm_read(&cache, 2, 0), 20.0);
        assert_eq!(hm_read(&cache, 3, 0), 30.0);
    }

    #[test]
    #[should_panic(expected = "HeadMajor")]
    fn test_shift_positions_for_head_panics_seq_major() {
        let mut cache = make_cache(16, 2, 4);
        cache.current_pos = 5;
        cache.shift_positions_for_head(0, 3, 1, 1).unwrap();
    }

    #[test]
    fn test_shift_positions_for_head_noop() {
        let heads = 2;
        let dim = 2;
        let mut cache = make_head_major_cache(16, heads, dim);
        cache.current_pos = 5;

        // count=0 should be noop
        cache.shift_positions_for_head(0, 3, 1, 0).unwrap();
        // src==dst should be noop
        cache.shift_positions_for_head(0, 3, 3, 1).unwrap();
    }

    // ── madvise / release_unused_pages tests ──

    /// Helper: make a large F16 SeqMajor cache (buffer > 1 page).
    fn make_large_f16_cache(max_seq_len: usize, heads: usize, dim: usize) -> KVCache {
        let size_bytes = max_seq_len * heads * dim * 2; // F16
        let k_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F16));
        let v_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F16));
        let backend = Arc::new(CpuBackend::new());
        let k = Tensor::new(
            Shape::new(vec![1, max_seq_len, heads, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, max_seq_len, heads, dim]), v_buf, backend);
        KVCache::new(k, v, max_seq_len)
    }

    fn make_large_f16_headmajor_cache(max_seq_len: usize, heads: usize, dim: usize) -> KVCache {
        let size_bytes = max_seq_len * heads * dim * 2; // F16
        let k_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F16));
        let v_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F16));
        let backend = Arc::new(CpuBackend::new());
        let k = Tensor::new(
            Shape::new(vec![1, heads, max_seq_len, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, heads, max_seq_len, dim]), v_buf, backend);
        KVCache {
            k_buffer: k,
            v_buffer: v,
            current_pos: 0,
            high_water_pos: 0,
            max_seq_len,
            capacity: max_seq_len,
            kv_heads: heads,
            head_dim: dim,
            layout: KVLayout::HeadMajor,
            memory: None,
        }
    }

    /// Read VmRSS from /proc/self/status in bytes.
    fn read_rss_bytes() -> usize {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let kb: usize = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                return kb * 1024;
            }
        }
        0
    }

    #[test]
    fn test_release_unused_pages_returns_bytes_seqmajor() {
        // 8192 positions * 8 heads * 64 dim * 2 bytes = 8 MB per buffer
        let mut cache = make_large_f16_cache(8192, 8, 64);
        cache.current_pos = 8192;
        cache.high_water_pos = 8192;

        // Touch all pages to ensure physical allocation
        let ptr = cache.k_buffer.as_mut_ptr();
        if !ptr.is_null() {
            for i in (0..cache.k_buffer.size()).step_by(4096) {
                unsafe { std::ptr::write_volatile(ptr.add(i), 0xABu8) };
            }
        }

        // Evict most tokens: keep 100 out of 8192
        cache.current_pos = 100;
        let released = cache.release_unused_pages();

        // With SeqMajor, single contiguous region.
        // used = 100 * 8 * 64 * 2 = 102400 bytes (~25 pages)
        // total = 8192 * 8 * 64 * 2 = 8388608 bytes (~2048 pages)
        // Released should be significant (> 7 MB from K+V)
        assert!(
            released > 7_000_000,
            "Expected >7MB released, got {} bytes",
            released
        );
    }

    #[test]
    fn test_release_unused_pages_returns_bytes_headmajor() {
        // 8192 positions * 8 heads * 64 dim * 2 bytes = 8 MB per buffer
        let mut cache = make_large_f16_headmajor_cache(8192, 8, 64);
        cache.current_pos = 8192;
        cache.high_water_pos = 8192;

        // Touch all pages
        let ptr = cache.k_buffer.as_mut_ptr();
        if !ptr.is_null() {
            for i in (0..cache.k_buffer.size()).step_by(4096) {
                unsafe { std::ptr::write_volatile(ptr.add(i), 0xABu8) };
            }
        }

        cache.current_pos = 100;
        let released = cache.release_unused_pages();

        // HeadMajor: per head = 8192 * 64 * 2 = 1048576 bytes = 1MB
        // used per head = 100 * 64 * 2 = 12800 bytes (~3 pages used)
        // ~253 pages released per head, 8 heads = ~2024 pages per buffer, x2 (K+V)
        assert!(
            released > 7_000_000,
            "Expected >7MB released, got {} bytes",
            released
        );
    }

    #[test]
    fn test_release_unused_pages_rss_reduction() {
        // Allocate large cache, touch pages, then release and check RSS drops
        let mut cache = make_large_f16_cache(16384, 8, 64);
        cache.current_pos = 16384;
        cache.high_water_pos = 16384;

        // Touch all pages in both K and V
        for buf_ptr in [cache.k_buffer.as_mut_ptr(), cache.v_buffer.as_mut_ptr()] {
            if !buf_ptr.is_null() {
                let size = 16384 * 8 * 64 * 2;
                for i in (0..size).step_by(4096) {
                    unsafe { std::ptr::write_volatile(buf_ptr.add(i), 0xCDu8) };
                }
            }
        }

        let rss_before = read_rss_bytes();

        // Evict most: keep 10 tokens out of 16384
        cache.current_pos = 10;
        cache.release_unused_pages();

        let rss_after = read_rss_bytes();

        // KV buffer total = 16384 * 8 * 64 * 2 * 2 (K+V) = 32 MB
        // After eviction: ~0.02 MB used, ~32 MB released
        // RSS should drop by at least 20 MB (allowing margin for test runtime overhead)
        let rss_drop = rss_before.saturating_sub(rss_after);
        assert!(
            rss_drop > 20_000_000,
            "Expected RSS drop > 20MB, got {} MB drop (before={}, after={})",
            rss_drop / (1024 * 1024),
            rss_before / (1024 * 1024),
            rss_after / (1024 * 1024),
        );
    }

    #[test]
    fn test_release_unused_pages_regrowth_integrity() {
        // After release, new data written to madvise'd region should work
        let mut cache = make_large_f16_cache(4096, 2, 64);
        cache.current_pos = 4096;
        cache.high_water_pos = 4096;

        // Touch all pages
        let k_ptr = cache.k_buffer.as_mut_ptr();
        if !k_ptr.is_null() {
            for i in (0..cache.k_buffer.size()).step_by(4096) {
                unsafe { std::ptr::write_volatile(k_ptr.add(i), 0xABu8) };
            }
        }

        // Evict to 10 tokens and release
        cache.current_pos = 10;
        cache.release_unused_pages();

        // "Regrow": write new data into the released region
        cache.current_pos = 2000;
        let k_ptr = cache.k_buffer.as_mut_ptr();
        if !k_ptr.is_null() {
            let type_size = 2; // F16
            let row_size = 2 * 64 * type_size; // heads * dim * type_size
            // Write a marker at position 1000
            let offset = 1000 * row_size;
            unsafe {
                std::ptr::write_volatile(k_ptr.add(offset), 0x42u8);
                // Read it back
                let val = std::ptr::read_volatile(k_ptr.add(offset));
                assert_eq!(
                    val, 0x42u8,
                    "Data written to released region should be readable"
                );
            }
        }
    }

    #[test]
    fn test_release_unused_pages_noop_on_full_cache() {
        let mut cache = make_large_f16_cache(1024, 2, 64);
        cache.current_pos = 1024; // Full cache
        cache.high_water_pos = 1024;
        let released = cache.release_unused_pages();
        assert_eq!(released, 0, "Full cache should release 0 bytes");
    }

    #[test]
    fn test_release_unused_pages_noop_on_small_eviction() {
        let mut cache = make_large_f16_cache(1024, 2, 64);
        cache.current_pos = 1024;
        cache.high_water_pos = 1024;

        // Touch pages
        let ptr = cache.k_buffer.as_mut_ptr();
        if !ptr.is_null() {
            for i in (0..cache.k_buffer.size()).step_by(4096) {
                unsafe { std::ptr::write_volatile(ptr.add(i), 0xABu8) };
            }
        }

        // Evict just 1 token — likely < 1 page for most configs
        cache.current_pos = 1023;
        let released = cache.release_unused_pages();
        // 1 token = 2 * 64 * 2 = 256 bytes, well under page size → 0 released
        assert_eq!(released, 0, "Evicting < 1 page should release 0 bytes");
    }

    #[test]
    fn test_release_unused_pages_headmajor_boundary_safety() {
        // Verify HeadMajor madvise doesn't corrupt adjacent head data
        let heads = 4;
        let dim = 64;
        let cap = 4096;
        let mut cache = make_large_f16_headmajor_cache(cap, heads, dim);
        cache.current_pos = cap;
        cache.high_water_pos = cap;

        let type_size = 2; // F16
        let head_stride = cap * dim * type_size;

        // Write a marker at the START of each head region (position 0)
        let k_ptr = cache.k_buffer.as_mut_ptr();
        if k_ptr.is_null() {
            return;
        }
        for h in 0..heads {
            unsafe {
                let head_base = k_ptr.add(h * head_stride);
                std::ptr::write_volatile(head_base, (0x10 + h) as u8);
            }
        }

        // Evict to 10 tokens and release unused pages
        cache.current_pos = 10;
        cache.release_unused_pages();

        // Verify markers at head starts are preserved (they're in the "used" region)
        for h in 0..heads {
            unsafe {
                let head_base = k_ptr.add(h * head_stride);
                let val = std::ptr::read_volatile(head_base);
                assert_eq!(
                    val,
                    (0x10 + h) as u8,
                    "Head {} start marker should be preserved after madvise",
                    h
                );
            }
        }
    }

    #[test]
    fn test_prune_prefix_calls_release_unused_pages() {
        // prune_prefix should release pages (verified via RSS)
        // Use large buffers (32 MB total) and touch both K and V for clear signal.
        let mut cache = make_large_f16_cache(16384, 8, 64);
        cache.current_pos = 16384;
        cache.high_water_pos = 16384;

        // Touch all pages in both K and V buffers
        for buf_ptr in [cache.k_buffer.as_mut_ptr(), cache.v_buffer.as_mut_ptr()] {
            if !buf_ptr.is_null() {
                let size = 16384 * 8 * 64 * 2;
                for i in (0..size).step_by(4096) {
                    unsafe { std::ptr::write_volatile(buf_ptr.add(i), 0xABu8) };
                }
            }
        }

        let rss_before = read_rss_bytes();
        cache.prune_prefix(16000).unwrap(); // Keep 384 tokens
        let rss_after = read_rss_bytes();

        assert_eq!(cache.current_pos, 384);
        // 32 MB total, ~97% evicted → expect > 20 MB RSS drop
        let rss_drop = rss_before.saturating_sub(rss_after);
        assert!(
            rss_drop > 20_000_000,
            "Expected RSS drop > 20MB from prune_prefix, got {} MB (before={}, after={})",
            rss_drop / (1024 * 1024),
            rss_before / (1024 * 1024),
            rss_after / (1024 * 1024),
        );
    }

    #[test]
    fn test_release_unused_pages_noop_on_q4_dtype() {
        // Q4_0 dtype should be skipped (returns 0)
        use crate::core::quant::QK4_0;
        let heads = 2;
        let dim = 64;
        let max_seq = 1024;
        let blocks_per_pos = heads * dim / QK4_0;
        let block_size = std::mem::size_of::<crate::core::quant::BlockQ4_0>();
        let buf_size = max_seq * blocks_per_pos * block_size;
        let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::Q4_0));
        let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::Q4_0));
        let backend = Arc::new(CpuBackend::new());
        let k = Tensor::new(
            Shape::new(vec![1, max_seq, heads, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, max_seq, heads, dim]), v_buf, backend);
        let mut cache = KVCache::new(k, v, max_seq);
        cache.current_pos = 512;
        let released = cache.release_unused_pages();
        assert_eq!(released, 0, "Q4_0 dtype should not trigger madvise");
    }

    #[test]
    fn test_round_up_and_round_down() {
        assert_eq!(super::round_up(0, 4096), 0);
        assert_eq!(super::round_up(1, 4096), 4096);
        assert_eq!(super::round_up(4095, 4096), 4096);
        assert_eq!(super::round_up(4096, 4096), 4096);
        assert_eq!(super::round_up(4097, 4096), 8192);

        assert_eq!(super::round_down(0, 4096), 0);
        assert_eq!(super::round_down(1, 4096), 0);
        assert_eq!(super::round_down(4095, 4096), 0);
        assert_eq!(super::round_down(4096, 4096), 4096);
        assert_eq!(super::round_down(4097, 4096), 4096);
    }

    #[test]
    fn test_madvise_dontneed_null_ptr() {
        // Null pointer should return 0 without crashing
        let released = super::madvise_dontneed(std::ptr::null(), 0, 4096);
        assert_eq!(released, 0);
    }

    #[test]
    fn test_madvise_dontneed_empty_range() {
        let buf = vec![0u8; 8192];
        let released = super::madvise_dontneed(buf.as_ptr(), 4096, 4096);
        assert_eq!(released, 0, "from == to should release 0");
        let released = super::madvise_dontneed(buf.as_ptr(), 4096, 100);
        assert_eq!(released, 0, "from > to should release 0");
    }

    // ── compact_keep_positions tests ──

    /// Write value `val` at (pos, all heads, d=0) in a HeadMajor F32 cache.
    fn hm_write_pos(cache: &mut KVCache, pos: usize, val: f32) {
        let heads = cache.kv_heads;
        let dim = cache.head_dim;
        let cap = cache.capacity;
        let k_slice = cache.k_buffer.as_mut_slice::<f32>();
        let v_slice = cache.v_buffer.as_mut_slice::<f32>();
        for h in 0..heads {
            let base = h * cap * dim;
            k_slice[base + pos * dim] = val;
            v_slice[base + pos * dim] = val * 10.0;
        }
    }

    /// Read the K value at (pos, head=0, d=0) in a HeadMajor F32 cache.
    fn hm_read_k(cache: &KVCache, pos: usize) -> f32 {
        let off = cache.offset(pos, 0);
        cache.k_buffer.as_slice::<f32>()[off]
    }

    /// Read the V value at (pos, head=0, d=0) in a HeadMajor F32 cache.
    fn hm_read_v_d0(cache: &KVCache, pos: usize) -> f32 {
        let off = cache.offset(pos, 0);
        cache.v_buffer.as_slice::<f32>()[off]
    }

    /// Make a HeadMajor F32 cache pre-filled with distinct values (pos+1) at every position.
    fn make_filled_hm_cache(max_seq: usize, heads: usize, dim: usize) -> KVCache {
        let mut cache = make_head_major_cache(max_seq, heads, dim);
        for pos in 0..max_seq {
            hm_write_pos(&mut cache, pos, (pos + 1) as f32);
        }
        cache.current_pos = max_seq;
        cache
    }

    #[test]
    fn test_compact_keep_positions_consecutive() {
        // keep = [2, 3, 4] — a single consecutive run
        // write_start = 0
        // Expected: pos 0←2, 1←3, 2←4 in a single batch shift
        let mut cache = make_filled_hm_cache(10, 2, 4);
        let keep = vec![2usize, 3, 4];
        cache.compact_keep_positions(&keep, 0).unwrap();
        cache.current_pos = keep.len();

        // pos 0 should have old pos 2 value = 3.0
        assert_eq!(hm_read_k(&cache, 0), 3.0);
        assert_eq!(hm_read_k(&cache, 1), 4.0);
        assert_eq!(hm_read_k(&cache, 2), 5.0);
        assert_eq!(hm_read_v_d0(&cache, 0), 30.0);
    }

    #[test]
    fn test_compact_keep_positions_scattered() {
        // keep = [1, 3, 5, 7] — scattered, no consecutive pairs
        // write_start = 0
        let mut cache = make_filled_hm_cache(10, 2, 4);
        let keep = vec![1usize, 3, 5, 7];
        cache.compact_keep_positions(&keep, 0).unwrap();
        cache.current_pos = keep.len();

        // Each is shifted individually: dst 0←1, 1←3, 2←5, 3←7
        assert_eq!(hm_read_k(&cache, 0), 2.0); // was pos 1 → value 2
        assert_eq!(hm_read_k(&cache, 1), 4.0); // was pos 3 → value 4
        assert_eq!(hm_read_k(&cache, 2), 6.0); // was pos 5 → value 6
        assert_eq!(hm_read_k(&cache, 3), 8.0); // was pos 7 → value 8
    }

    #[test]
    fn test_compact_keep_positions_one_gap() {
        // Classic H2O scenario: keep HH at [0,2,4] and recent [7,8,9]
        // write_start = 0 (no protected prefix)
        // Batching: [0] in-place, [2] single, [4] single, [7,8,9] batch of 3
        let mut cache = make_filled_hm_cache(10, 2, 4);
        let keep = vec![0usize, 2, 4, 7, 8, 9];
        cache.compact_keep_positions(&keep, 0).unwrap();
        cache.current_pos = keep.len();

        assert_eq!(hm_read_k(&cache, 0), 1.0); // pos 0, in-place
        assert_eq!(hm_read_k(&cache, 1), 3.0); // pos 1 ← old pos 2
        assert_eq!(hm_read_k(&cache, 2), 5.0); // pos 2 ← old pos 4
        assert_eq!(hm_read_k(&cache, 3), 8.0); // pos 3 ← old pos 7
        assert_eq!(hm_read_k(&cache, 4), 9.0); // pos 4 ← old pos 8
        assert_eq!(hm_read_k(&cache, 5), 10.0); // pos 5 ← old pos 9
    }

    #[test]
    fn test_compact_keep_positions_all_in_place() {
        // keep = [0, 1, 2] with write_start = 0 — nothing to move
        let mut cache = make_filled_hm_cache(10, 2, 4);
        cache.compact_keep_positions(&[0, 1, 2], 0).unwrap();
        cache.current_pos = 3;

        // Values unchanged
        assert_eq!(hm_read_k(&cache, 0), 1.0);
        assert_eq!(hm_read_k(&cache, 1), 2.0);
        assert_eq!(hm_read_k(&cache, 2), 3.0);
    }

    #[test]
    fn test_compact_keep_positions_empty() {
        // Empty keep list → no-op, no panic
        let mut cache = make_filled_hm_cache(10, 2, 4);
        cache.compact_keep_positions(&[], 0).unwrap();
        // Values at original positions are untouched
        assert_eq!(hm_read_k(&cache, 0), 1.0);
    }

    #[test]
    fn test_compact_keep_positions_with_write_start() {
        // Simulate protected prefix: write_start = 2, keep = [4, 5, 8, 9]
        // 4,5 consecutive batch → shift to 2,3; 8,9 consecutive batch → shift to 4,5
        let mut cache = make_filled_hm_cache(12, 2, 4);
        let keep = vec![4usize, 5, 8, 9];
        cache.compact_keep_positions(&keep, 2).unwrap();
        cache.current_pos = 2 + keep.len();

        assert_eq!(hm_read_k(&cache, 2), 5.0); // old pos 4
        assert_eq!(hm_read_k(&cache, 3), 6.0); // old pos 5
        assert_eq!(hm_read_k(&cache, 4), 9.0); // old pos 8
        assert_eq!(hm_read_k(&cache, 5), 10.0); // old pos 9
        // Protected prefix (pos 0,1) untouched
        assert_eq!(hm_read_k(&cache, 0), 1.0);
        assert_eq!(hm_read_k(&cache, 1), 2.0);
    }

    #[test]
    fn test_compact_keep_positions_for_head() {
        // Per-head version: only head 0 is modified; head 1 untouched.
        // Cache: heads=2, dim=2, max_seq=10
        // Fill each head differently: head 0 = (pos+1)*1.0, head 1 = (pos+1)*100.0
        let heads = 2;
        let dim = 2;
        let cap = 10;
        let size_bytes = cap * heads * dim * 4;
        let k_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(size_bytes, DType::F32));
        let backend = Arc::new(CpuBackend::new());
        let k = Tensor::new(Shape::new(vec![1, heads, cap, dim]), k_buf, backend.clone());
        let v = Tensor::new(Shape::new(vec![1, heads, cap, dim]), v_buf, backend);
        let mut cache = KVCache {
            k_buffer: k,
            v_buffer: v,
            current_pos: cap,
            high_water_pos: cap,
            max_seq_len: cap,
            capacity: cap,
            kv_heads: heads,
            head_dim: dim,
            layout: KVLayout::HeadMajor,
            memory: None,
        };

        // Write distinct values per head
        {
            let k_slice = cache.k_buffer.as_mut_slice::<f32>();
            // head 0: stride = cap * dim = 20
            for pos in 0..cap {
                k_slice[pos * dim] = (pos + 1) as f32; // head 0
                k_slice[cap * dim + pos * dim] = (pos + 1) as f32 * 100.0; // head 1
            }
        }

        // Compact head 0 only: keep positions [2, 3, 7, 8, 9] → write_start=0
        let keep = vec![2usize, 3, 7, 8, 9];
        cache.compact_keep_positions_for_head(0, &keep, 0).unwrap();

        {
            let k_slice = cache.k_buffer.as_slice::<f32>();
            // Head 0 at pos 0 ← old pos 2 = 3.0; pos 1 ← old pos 3 = 4.0
            assert_eq!(k_slice[0 * dim], 3.0, "head0 pos0 should be old pos2");
            assert_eq!(k_slice[1 * dim], 4.0, "head0 pos1 should be old pos3");
            // pos 2,3,4 ← old pos 7,8,9 (consecutive batch)
            assert_eq!(k_slice[2 * dim], 8.0, "head0 pos2 should be old pos7");
            assert_eq!(k_slice[3 * dim], 9.0, "head0 pos3 should be old pos8");
            assert_eq!(k_slice[4 * dim], 10.0, "head0 pos4 should be old pos9");

            // Head 1 is completely untouched
            for pos in 0..cap {
                let expected = (pos + 1) as f32 * 100.0;
                assert_eq!(
                    k_slice[cap * dim + pos * dim],
                    expected,
                    "head1 pos{} should be unchanged",
                    pos
                );
            }
        }
    }

    // ── high_water_pos tracking tests ──

    /// Test that high_water_pos is correctly maintained across
    /// update → eviction → release_unused_pages flow.
    #[test]
    fn test_high_water_pos_tracking() {
        let mut cache = make_cache(100, 1, 4);

        // Initially both counters are zero
        assert_eq!(cache.current_pos, 0);
        assert_eq!(cache.high_water_pos, 0);

        // update() should advance high_water_pos monotonically
        for i in 0..10 {
            let (k, v) = make_token_tensor((i + 1) as f32, 1, 4);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 10);
        assert_eq!(cache.high_water_pos, 10);

        // Simulate eviction: current_pos drops, high_water_pos stays
        cache.current_pos = 5;
        assert_eq!(
            cache.high_water_pos, 10,
            "hwm must not decrease on eviction"
        );

        // release_unused_pages uses hwm as upper bound, then resets hwm to current_pos
        let released = cache.release_unused_pages();
        // Buffer is F32: 5 tokens * 1 head * 4 dim * 4 bytes = 80 bytes — below page granularity.
        // released may be 0 due to alignment, but high_water_pos must be reset.
        let _ = released; // don't assert bytes — page alignment makes it zero on small caches
        assert_eq!(
            cache.high_water_pos, 5,
            "hwm reset to current_pos after release"
        );

        // Continuing to update advances hwm again
        for i in 0..3 {
            let (k, v) = make_token_tensor((100 + i) as f32, 1, 4);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.current_pos, 8);
        assert_eq!(cache.high_water_pos, 8);

        // advance_pos also tracks hwm
        cache.advance_pos(4);
        assert_eq!(cache.current_pos, 12);
        assert_eq!(cache.high_water_pos, 12);

        // set_current_pos(0) resets hwm
        cache.set_current_pos(0);
        assert_eq!(cache.high_water_pos, 0, "set_current_pos(0) must reset hwm");

        // set_current_pos to a non-zero value does NOT reset hwm (only zero resets)
        cache.current_pos = 20;
        cache.high_water_pos = 20;
        cache.set_current_pos(10);
        assert_eq!(
            cache.high_water_pos, 20,
            "set_current_pos(non-zero) must not change hwm"
        );
    }

    /// Test that prune_prefix(all) resets high_water_pos via the remaining==0 path.
    #[test]
    fn test_high_water_pos_reset_on_prune_all() {
        let mut cache = make_cache(100, 1, 4);

        for i in 0..5 {
            let (k, v) = make_token_tensor((i + 1) as f32, 1, 4);
            cache.update(&k, &v).unwrap();
        }
        assert_eq!(cache.high_water_pos, 5);

        // prune_prefix(all) → remaining == 0 path
        cache.prune_prefix(5).unwrap();
        assert_eq!(cache.current_pos, 0);
        assert_eq!(cache.high_water_pos, 0, "prune_prefix(all) must reset hwm");
    }

    /// Test that release_unused_pages with hwm as upper bound narrows the madvise range
    /// compared to using capacity as the upper bound.
    #[test]
    fn test_high_water_pos_limits_madvise_range() {
        // Large cache: 8192 positions, but only touch the first 1024 positions.
        // high_water_pos = 1024 → madvise range is current_pos..1024 (not ..8192).
        let mut cache = make_large_f16_cache(8192, 8, 64);

        // Only touch first 1024 positions worth of pages
        let row_bytes = 8 * 64 * 2; // heads * dim * F16
        let touch_bytes = 1024 * row_bytes;
        let ptr = cache.k_buffer.as_mut_ptr();
        if !ptr.is_null() {
            for i in (0..touch_bytes).step_by(4096) {
                unsafe { std::ptr::write_volatile(ptr.add(i), 0xBBu8) };
            }
        }

        // Set hwm to 1024 (matching the touched region), current_pos to 100
        cache.current_pos = 1024;
        cache.high_water_pos = 1024;
        cache.current_pos = 100;

        // release_unused_pages should only advise 100..1024 (not 100..8192)
        let released = cache.release_unused_pages();
        // Expected range: (1024 - 100) * 8 * 64 * 2 * 2 (K+V) = ~2 MB
        // Full-capacity range would be (8192 - 100) * 8 * 64 * 2 * 2 = ~16 MB
        // We assert > 1 MB to confirm madvise fired over meaningful range
        assert!(
            released > 1_000_000,
            "Expected >1MB released (hwm-bounded), got {} bytes",
            released
        );
        // And hwm must be reset to current_pos after release
        assert_eq!(cache.high_water_pos, 100);
    }
}
