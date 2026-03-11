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
                    let src_row = self.kv_heads * self.head_dim / qk; // blocks per seq position in input
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
                    let src_row = self.kv_heads * self.head_dim; // elements per seq position in input
                    let dst_head_stride = self.capacity * self.head_dim;

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

        let remaining = self.current_pos - count;

        if remaining == 0 {
            self.current_pos = 0;
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
}

// ── KVCacheOps implementation for KVCache ───────────────────────────────────

impl KVCacheOps for KVCache {
    fn current_pos(&self) -> usize {
        self.current_pos
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
}
