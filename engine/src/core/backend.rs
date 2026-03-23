use crate::core::buffer::DType;
use crate::core::tensor::Tensor;
use anyhow::Result;

pub trait Backend: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    fn name(&self) -> &str;
    fn device(&self) -> &str;

    // Basic Math
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()>;
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()>;
    fn matmul_slice(
        &self,
        a: &Tensor,
        b: &Tensor,
        rows: usize,
        cols: usize,
        out: &mut Tensor,
    ) -> Result<()>;

    // In-place operations
    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()>;
    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()>;

    /// Broadcast-add a 1D bias to each row of a tensor.
    /// x: [..., dim], bias: [dim]. Adds bias to every row of x.
    fn add_row_bias(&self, x: &mut Tensor, bias: &Tensor) -> Result<()> {
        let x_data = x.as_mut_slice::<f32>();
        let b_data = bias.as_slice::<f32>();
        let dim = b_data.len();
        for row in x_data.chunks_mut(dim) {
            for (v, &b) in row.iter_mut().zip(b_data.iter()) {
                *v += b;
            }
        }
        Ok(())
    }

    // Activation & Norm
    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()>;
    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32) -> Result<()>;
    /// Out-of-place RMS norm: out = norm(x) * w. x is preserved.
    /// Default: copy x → out, then in-place rms_norm(out).
    fn rms_norm_oop(&self, x: &Tensor, out: &mut Tensor, w: &Tensor, eps: f32) -> Result<()> {
        self.copy_into(x, out)?;
        self.rms_norm(out, w, eps)
    }
    /// Fused add + out-of-place RMS norm: x += residual; out = norm(x) * w.
    /// Eliminates a separate add_assign dispatch.
    fn add_rms_norm_oop(
        &self,
        x: &mut Tensor,
        residual: &Tensor,
        out: &mut Tensor,
        w: &Tensor,
        eps: f32,
    ) -> Result<()> {
        self.add_assign(x, residual)?;
        self.rms_norm_oop(x, out, w, eps)
    }
    fn softmax(&self, x: &mut Tensor) -> Result<()>;

    // Rotate
    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()>;

    // Single-query attention for generation (GQA-aware)
    // Q: [num_heads_q, head_dim]
    // K/V cache: SeqMajor [cache_seq_len, num_heads_kv, head_dim] or HeadMajor [num_heads_kv, capacity, head_dim]
    // Output: [num_heads_q, head_dim]
    #[allow(clippy::too_many_arguments)]
    fn attention_gen(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        num_heads_q: usize,
        num_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
    ) -> Result<()> {
        // Default CPU implementation
        let q_data = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const f32, q.size() / 4) };
        let k_data = unsafe {
            std::slice::from_raw_parts(k_cache.as_ptr() as *const f32, k_cache.size() / 4)
        };
        let v_data = unsafe {
            std::slice::from_raw_parts(v_cache.as_ptr() as *const f32, v_cache.size() / 4)
        };
        let out_data =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, out.size() / 4) };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = num_heads_q / num_heads_kv;

        // Detect layout from shape: HeadMajor if shape[1] == num_heads_kv
        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == num_heads_kv && k_shape[1] != k_shape[2];
        let capacity = if is_head_major { k_shape[2] } else { 0 };

        for h in 0..num_heads_q {
            let kv_h = h / gqa_ratio;
            let q_off = h * head_dim;
            let q_vec = &q_data[q_off..q_off + head_dim];

            // Compute scores
            let mut scores = vec![0.0f32; cache_seq_len];
            #[allow(clippy::needless_range_loop)]
            for t in 0..cache_seq_len {
                let k_off = if is_head_major {
                    (kv_h * capacity + t) * head_dim
                } else {
                    (t * num_heads_kv + kv_h) * head_dim
                };
                let k_vec = &k_data[k_off..k_off + head_dim];
                let score: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                scores[t] = score * scale;
            }

            // Softmax
            let max_val = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - max_val).exp();
                sum_exp += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum_exp;
            }

            // Weighted sum of V
            let out_off = h * head_dim;
            for d in 0..head_dim {
                out_data[out_off + d] = 0.0;
            }
            #[allow(clippy::needless_range_loop)]
            for t in 0..cache_seq_len {
                let weight = scores[t];
                let v_off = if is_head_major {
                    (kv_h * capacity + t) * head_dim
                } else {
                    (t * num_heads_kv + kv_h) * head_dim
                };
                let v_vec = &v_data[v_off..v_off + head_dim];
                for d in 0..head_dim {
                    out_data[out_off + d] += weight * v_vec[d];
                }
            }
        }
        Ok(())
    }

    // Memory Ops
    fn copy_from(&self, t: &Tensor) -> Result<Tensor>;

    /// Fused F32→F16 cast + HeadMajor scatter for KV cache update.
    /// Replaces 2× cast + 16× copy_slice with a single GPU kernel.
    /// Default: falls back to separate cast + copy.
    #[allow(unused_variables, clippy::too_many_arguments)]
    fn kv_scatter_f32_to_f16(
        &self,
        k_src: &Tensor,
        v_src: &Tensor,
        k_dst: &mut Tensor,
        v_dst: &mut Tensor,
        head_dim: usize,
        capacity: usize,
        write_pos: usize,
    ) -> Result<()> {
        anyhow::bail!("kv_scatter_f32_to_f16 not implemented for this backend")
    }

    /// Copy data from src into dst buffer (same shape/size required).
    /// On GPU: just enqueue_copy_buffer, no new backend/kernel allocation.
    /// On CPU: memcpy.
    fn copy_into(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        let size = src.size();
        assert_eq!(size, dst.size(), "copy_into: size mismatch");
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();
        if !src_ptr.is_null() && !dst_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
            }
        }
        Ok(())
    }
    fn read_buffer(&self, t: &Tensor, dst: &mut [u8]) -> Result<()> {
        let src_ptr = t.buffer().as_ptr();
        if src_ptr.is_null() {
            anyhow::bail!("Cannot read null buffer (not mapped)");
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    /// Write host bytes into a backend buffer (CPU→GPU upload).
    /// Default: memcpy from src slice to tensor's mapped pointer.
    /// GPU backends should override with enqueue_write_buffer.
    fn write_buffer(&self, t: &mut Tensor, src: &[u8]) -> Result<()> {
        let dst_ptr = t.as_mut_ptr();
        if dst_ptr.is_null() {
            anyhow::bail!("Cannot write to null buffer (not mapped)");
        }
        assert_eq!(
            src.len(),
            t.size(),
            "write_buffer: size mismatch ({} vs {})",
            src.len(),
            t.size()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr, src.len());
        }
        Ok(())
    }

    // Type casting (e.g. F32 → F16)
    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()>;

    // Synchronization (for benchmarking)
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    /// Flush the command queue — submit enqueued commands to the device without waiting.
    /// On GPU backends this calls clFlush to prevent pipeline bubbles.
    fn flush(&self) -> Result<()> {
        Ok(())
    }

    // Embedding Lookup / Gather
    // src: [Rows, Cols] (Embeddings)
    // indices: [NumIndices] (Indices)
    // dst: [NumIndices, Cols] (Output)
    fn gather(&self, src: &Tensor, indices: &Tensor, dst: &mut Tensor) -> Result<()> {
        // Default CPU implementation — supports F32 and F16 src → F32 dst
        let idx_data = unsafe {
            std::slice::from_raw_parts(indices.as_ptr() as *const u32, indices.size() / 4)
        };
        let dst_data =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.size() / 4) };

        let cols = src.shape().dims()[1];
        let src_rows = src.shape().dims()[0];

        match src.dtype() {
            DType::F32 => {
                let src_data = unsafe {
                    std::slice::from_raw_parts(src.as_ptr() as *const f32, src_rows * cols)
                };
                for (i, &idx) in idx_data.iter().enumerate() {
                    let offset = idx as usize * cols;
                    let target_offset = i * cols;
                    if offset + cols <= src_data.len() && target_offset + cols <= dst_data.len() {
                        dst_data[target_offset..target_offset + cols]
                            .copy_from_slice(&src_data[offset..offset + cols]);
                    }
                }
            }
            DType::F16 => {
                let src_data = unsafe {
                    std::slice::from_raw_parts(src.as_ptr() as *const half::f16, src_rows * cols)
                };
                for (i, &idx) in idx_data.iter().enumerate() {
                    let offset = idx as usize * cols;
                    let target_offset = i * cols;
                    if offset + cols <= src_data.len() && target_offset + cols <= dst_data.len() {
                        for d in 0..cols {
                            dst_data[target_offset + d] = src_data[offset + d].to_f32();
                        }
                    }
                }
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "gather: unsupported src dtype {:?}",
                    src.dtype()
                ));
            }
        }
        Ok(())
    }

    /// Shift data within a single tensor (overlap-safe, like memmove).
    /// Offsets and count are in element units (Q4_0: 1 element = 1 block = 18 bytes).
    /// Default implementation uses CPU memmove via `std::ptr::copy`.
    fn buffer_shift(
        &self,
        tensor: &mut Tensor,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) -> Result<()> {
        let type_size = match tensor.dtype() {
            crate::core::buffer::DType::F32 => 4,
            crate::core::buffer::DType::F16 => 2,
            crate::core::buffer::DType::U8 => 1,
            crate::core::buffer::DType::Q4_0 => {
                std::mem::size_of::<crate::core::quant::BlockQ4_0>()
            }
            _ => 1,
        };

        let ptr = tensor.as_mut_ptr();
        if ptr.is_null() {
            anyhow::bail!("Null pointer in buffer_shift (default impl)");
        }
        unsafe {
            std::ptr::copy(
                ptr.add(src_offset * type_size),
                ptr.add(dst_offset * type_size),
                count * type_size,
            );
        }
        Ok(())
    }

    // New API: Copy slice from src to dst
    // src_offset/dst_offset are ELEMENT offsets (not bytes) if Tensor is typed, but here for simplicity let's assume they are ELEMENT offsets relative to Tensor's DType?
    // Actually, Tensors are somewhat untyped regarding pointer arithmetic in this trait unless we know DType size.
    // Let's assume count is number of ELEMENTS. And offsets are ELEMENTS.
    // Caller must ensure types match.
    fn copy_slice(
        &self,
        src: &Tensor,
        dst: &mut Tensor,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) -> Result<()> {
        let type_size = match src.dtype() {
            crate::core::buffer::DType::F32 => 4,
            crate::core::buffer::DType::F16 => 2,
            crate::core::buffer::DType::U8 => 1,
            crate::core::buffer::DType::Q4_0 => {
                std::mem::size_of::<crate::core::quant::BlockQ4_0>()
            }
            _ => 1, // Fallback
        };

        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();

        if src_ptr.is_null() || dst_ptr.is_null() {
            anyhow::bail!(
                "Null pointer in copy_slice (default impl), likely OpenCL buffer mismatch"
            );
        }

        unsafe {
            // Calculate byte offsets
            let src_byte_offset = src_offset * type_size;
            let dst_byte_offset = dst_offset * type_size;
            let byte_count = count * type_size;

            std::ptr::copy_nonoverlapping(
                src_ptr.add(src_byte_offset),
                dst_ptr.add(dst_byte_offset),
                byte_count,
            );
        }
        Ok(())
    }
}
