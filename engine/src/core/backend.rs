use crate::core::buffer::DType;
use crate::core::tensor::Tensor;
use anyhow::Result;

/// Opaque async GPU event handle returned by `enqueue_read_buffer_async`.
///
/// Used by tensor-partition's `LLMRS_PARTITION_ASYNC_READ` path to overlap
/// the residual DMA read with a subsequent GPU enqueue chain. Default for
/// non-OpenCL backends is a dummy (no-op wait) since the default async
/// fallback in `Backend` simply performs a synchronous blocking read.
#[derive(Default)]
pub struct GpuEvent {
    #[cfg(feature = "opencl")]
    pub(crate) inner: Option<ocl::core::Event>,
}

impl GpuEvent {
    /// Dummy event — `wait_event` on this is a no-op.
    pub fn dummy() -> Self {
        Self {
            #[cfg(feature = "opencl")]
            inner: None,
        }
    }
}

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

    /// Fused FFN gate + up matmul followed by SiLU gating:
    ///     out[i] = silu(gate·x)[i] * (up·x)[i]
    ///
    /// Both matmuls share the same activation `x`, so a GPU backend
    /// can bundle them into one kernel and reuse the quantised
    /// activation cache. The `gate_scratch`/`up_scratch` tensors are
    /// temporaries used by the default (non-fused) fallback; GPU
    /// overrides may ignore them and write `out` directly.
    ///
    /// Default: 3 op fallback (gate matmul → up matmul → silu_mul).
    fn matmul_ffn_gate_up_silu(
        &self,
        x: &Tensor,
        w_gate: &Tensor,
        w_up: &Tensor,
        out: &mut Tensor,
        up_scratch: &mut Tensor,
    ) -> Result<()> {
        self.matmul_transposed(x, w_gate, out)?;
        self.matmul_transposed(x, w_up, up_scratch)?;
        self.silu_mul(out, up_scratch)
    }

    /// GELU tanh approximation fused with elementwise multiply.
    /// gate[i] = gelu_tanh(gate[i]) * up[i]
    /// Used by Gemma 3 FFN (hidden_activation = "gelu_pytorch_tanh").
    /// Default: scalar CPU fallback — backends may override for performance.
    fn gelu_tanh_mul(&self, gate: &mut Tensor, up: &Tensor) -> Result<()> {
        let gate_data = gate.as_mut_slice::<f32>();
        let up_data = up.as_slice::<f32>();
        let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
        for (g, &u) in gate_data.iter_mut().zip(up_data.iter()) {
            let x = *g;
            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            *g = 0.5 * x * (1.0 + inner.tanh()) * u;
        }
        Ok(())
    }
    /// In-place RMS norm: x = x * w / rms(x).
    /// If `add_unit` is true (Gemma 3 style), applies `x * (1 + w) / rms(x)` instead.
    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32, add_unit: bool) -> Result<()>;
    /// Out-of-place RMS norm: out = norm(x) * w. x is preserved.
    /// Default: copy x → out, then in-place rms_norm(out).
    fn rms_norm_oop(
        &self,
        x: &Tensor,
        out: &mut Tensor,
        w: &Tensor,
        eps: f32,
        add_unit: bool,
    ) -> Result<()> {
        self.copy_into(x, out)?;
        self.rms_norm(out, w, eps, add_unit)
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
        add_unit: bool,
    ) -> Result<()> {
        self.add_assign(x, residual)?;
        self.rms_norm_oop(x, out, w, eps, add_unit)
    }

    /// Fused 3-input merge + out-of-place RMS norm for tensor-partition layer boundary.
    ///
    /// Replaces the decode-path per-layer merge sequence
    ///     (copy_slice gpu → buf, copy_slice cpu → staging, add_assign, residual += buf)
    /// plus the next layer's pre-attention RMSNorm with a single kernel dispatch,
    /// eliminating three inter-kernel barriers on the OpenCL in-order queue.
    ///
    /// Semantics:
    ///   residual_out[i] = prior_residual[i] + gpu_partial[i] + cpu_staging[i]
    ///   out[i]          = (residual_out[i] / rms(residual_out)) * w_eff[i]
    /// where `w_eff = (1 + norm_weight)` when `add_unit` is true (Gemma3) else `norm_weight`.
    ///
    /// Buffer aliasing: `residual_out` MAY alias `prior_residual` for an in-place residual
    /// update. `out` MUST NOT alias any input. The three partial inputs MUST be distinct
    /// buffers.
    ///
    /// Default implementation: composes the existing 3-step path for correctness and for
    /// backends that do not override (CPU, CUDA, and OpenCL fallback during bring-up).
    /// The OpenCL backend overrides with the fused kernel.
    #[allow(clippy::too_many_arguments)]
    fn fused_norm_merge(
        &self,
        prior_residual: &Tensor,
        gpu_partial: &Tensor,
        cpu_staging: &Tensor,
        norm_weight: &Tensor,
        out: &mut Tensor,
        residual_out: &mut Tensor,
        eps: f32,
        add_unit: bool,
    ) -> Result<()> {
        // Default path: residual_out = prior + gpu + cpu; out = norm(residual_out) * w.
        // Copy prior into residual_out, then add gpu and cpu contributions.
        self.copy_into(prior_residual, residual_out)?;
        self.add_assign(residual_out, gpu_partial)?;
        self.add_assign(residual_out, cpu_staging)?;
        self.rms_norm_oop(residual_out, out, norm_weight, eps, add_unit)
    }
    fn softmax(&self, x: &mut Tensor) -> Result<()>;

    // Rotate
    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()>;

    // Single-query attention for generation (GQA-aware)
    // Q: [num_heads_q, head_dim]
    // K/V cache: SeqMajor [cache_seq_len, num_heads_kv, head_dim] or HeadMajor [num_heads_kv, capacity, head_dim]
    // Output: [num_heads_q, head_dim]
    // scores_out: Optional [num_heads_q * stride] buffer for post-softmax attention scores.
    //             stride = scores_out.len() / num_heads_q. Each head's [0..cache_seq_len] filled.
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
        scores_out: Option<&mut [f32]>,
    ) -> Result<()> {
        // Default CPU implementation — supports F32 and F16 KV cache.
        use crate::core::buffer::DType;
        let q_data = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const f32, q.size() / 4) };
        let out_data =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, out.size() / 4) };

        // For F16 KV, dequantize to temporary F32 buffers.
        let kv_dtype = k_cache.dtype();
        let (k_f32_buf, v_f32_buf);
        let (k_data, v_data): (&[f32], &[f32]) = match kv_dtype {
            DType::F32 => (
                unsafe {
                    std::slice::from_raw_parts(k_cache.as_ptr() as *const f32, k_cache.size() / 4)
                },
                unsafe {
                    std::slice::from_raw_parts(v_cache.as_ptr() as *const f32, v_cache.size() / 4)
                },
            ),
            DType::F16 => {
                use half::f16;
                let k_f16 = unsafe {
                    std::slice::from_raw_parts(k_cache.as_ptr() as *const f16, k_cache.size() / 2)
                };
                let v_f16 = unsafe {
                    std::slice::from_raw_parts(v_cache.as_ptr() as *const f16, v_cache.size() / 2)
                };
                k_f32_buf = k_f16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();
                v_f32_buf = v_f16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();
                (k_f32_buf.as_slice(), v_f32_buf.as_slice())
            }
            _ => anyhow::bail!(
                "attention_gen default impl: unsupported KV dtype {:?}",
                kv_dtype
            ),
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = num_heads_q / num_heads_kv;

        // Detect layout from shape: HeadMajor if shape[1] == num_heads_kv
        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == num_heads_kv && k_shape[1] != k_shape[2];
        let capacity = if is_head_major { k_shape[2] } else { 0 };
        let scores_stride = scores_out
            .as_ref()
            .map(|s| s.len() / num_heads_q)
            .unwrap_or(0);
        let scores_ptr = scores_out.map(|s| s.as_mut_ptr());

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

            // Copy scores to output buffer if requested
            if let Some(ptr) = scores_ptr {
                unsafe {
                    let dst =
                        std::slice::from_raw_parts_mut(ptr.add(h * scores_stride), cache_seq_len);
                    dst.copy_from_slice(&scores[..cache_seq_len]);
                }
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

    /// Copy a weight tensor from CPU to this backend.
    ///
    /// Distinguishes static, read-only weights (uploaded once at model load
    /// time) from runtime activations. Backends that can benefit from
    /// device-local storage for weights (e.g. `cuda_embedded` with
    /// `--cuda-weights-device` to bypass Jetson UMA cache coherency issues)
    /// override this method. The default delegates to `copy_from`.
    fn copy_weight_from(&self, t: &Tensor) -> Result<Tensor> {
        self.copy_from(t)
    }

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
        // CPU fallback: F32→F16 cast + HeadMajor scatter per head.
        // k_src: [1, seq_len, kv_heads * head_dim] F32
        // k_dst: [1, kv_heads, capacity, head_dim] F16 HeadMajor
        use half::f16;
        let kv_heads = k_src.shape().dims().last().copied().unwrap_or(0) / head_dim;
        let src_f32 =
            unsafe { std::slice::from_raw_parts(k_src.as_ptr() as *const f32, k_src.size() / 4) };
        let dst_f16 = unsafe {
            std::slice::from_raw_parts_mut(k_dst.as_mut_ptr() as *mut f16, k_dst.size() / 2)
        };
        for h in 0..kv_heads {
            let src_off = h * head_dim;
            let dst_off = h * capacity * head_dim + write_pos * head_dim;
            for d in 0..head_dim {
                dst_f16[dst_off + d] = f16::from_f32(src_f32[src_off + d]);
            }
        }
        let v_src_f32 =
            unsafe { std::slice::from_raw_parts(v_src.as_ptr() as *const f32, v_src.size() / 4) };
        let v_dst_f16 = unsafe {
            std::slice::from_raw_parts_mut(v_dst.as_mut_ptr() as *mut f16, v_dst.size() / 2)
        };
        for h in 0..kv_heads {
            let src_off = h * head_dim;
            let dst_off = h * capacity * head_dim + write_pos * head_dim;
            for d in 0..head_dim {
                v_dst_f16[dst_off + d] = f16::from_f32(v_src_f32[src_off + d]);
            }
        }
        Ok(())
    }

    /// Whether this backend has a native batch F32→F16 KV scatter implementation.
    ///
    /// Returns `false` by default. Backends implementing a real kernel for
    /// `kv_scatter_f32_to_f16_batch` must override this to return `true`.
    ///
    /// The default implementation of `kv_scatter_f32_to_f16_batch` uses host
    /// pointer slices via `as_ptr()`/`as_mut_ptr()`, which segfaults on
    /// device-only buffers (e.g. OpenCL without zero-copy mapping). Callers
    /// must check this flag before dispatching the batch path and fall back to
    /// the per-position `cast + update` path otherwise.
    fn supports_kv_scatter_batch(&self) -> bool {
        false
    }

    /// Batch F32->F16 KV scatter for prefill: writes seq_len positions in one kernel launch.
    /// k_src/v_src: contiguous [seq_len, kv_heads * head_dim] F32
    /// k_dst/v_dst: [kv_heads, capacity, head_dim] F16 HeadMajor
    /// Default: host-pointer fallback — ONLY safe when callers guard on `supports_kv_scatter_batch()`.
    #[allow(clippy::too_many_arguments)]
    fn kv_scatter_f32_to_f16_batch(
        &self,
        k_src: &Tensor,
        v_src: &Tensor,
        k_dst: &mut Tensor,
        v_dst: &mut Tensor,
        n_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        write_pos_start: usize,
        seq_len: usize,
    ) -> Result<()> {
        // Default: iterate per position using single-position scatter
        use half::f16;
        let src_f32 =
            unsafe { std::slice::from_raw_parts(k_src.as_ptr() as *const f32, k_src.size() / 4) };
        let dst_f16 = unsafe {
            std::slice::from_raw_parts_mut(k_dst.as_mut_ptr() as *mut f16, k_dst.size() / 2)
        };
        let v_src_f32 =
            unsafe { std::slice::from_raw_parts(v_src.as_ptr() as *const f32, v_src.size() / 4) };
        let v_dst_f16 = unsafe {
            std::slice::from_raw_parts_mut(v_dst.as_mut_ptr() as *mut f16, v_dst.size() / 2)
        };
        for s in 0..seq_len {
            for h in 0..n_kv_heads {
                let src_off = (s * n_kv_heads + h) * head_dim;
                let dst_off = h * capacity * head_dim + (write_pos_start + s) * head_dim;
                for d in 0..head_dim {
                    dst_f16[dst_off + d] = f16::from_f32(src_f32[src_off + d]);
                    v_dst_f16[dst_off + d] = f16::from_f32(v_src_f32[src_off + d]);
                }
            }
        }
        Ok(())
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

    /// Non-blocking variant of `read_buffer`. Enqueues a DMA read and returns
    /// an opaque event handle that can be awaited via `wait_event`. Backends
    /// that support true async reads (OpenCL) override this. The default
    /// falls back to a synchronous blocking read and returns a dummy event.
    ///
    /// # Safety
    ///
    /// The caller must ensure `dst` remains valid until `wait_event` has
    /// returned for the returned event — OpenCL writes into `dst` on the
    /// device's timeline, not the host's.
    fn enqueue_read_buffer_async(&self, t: &Tensor, dst: &mut [u8]) -> Result<GpuEvent> {
        self.read_buffer(t, dst)?;
        Ok(GpuEvent::dummy())
    }

    /// Block until the event returned by `enqueue_read_buffer_async` has
    /// completed. Default: no-op (for backends that return a dummy event
    /// because the enqueue was already blocking).
    fn wait_event(&self, _evt: &GpuEvent) -> Result<()> {
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

    /// Write host bytes into a sub-range of a backend buffer starting at
    /// `dst_offset`. Used by callers that keep a max-capacity buffer but only
    /// need to upload the currently-valid prefix (e.g. `OffloadKVCache` staging
    /// a GPU KV view of length `current_pos` into a `max_seq_len`-sized buffer).
    ///
    /// Default: memcpy into the tensor's mapped pointer at `dst_offset`.
    /// GPU backends should override with a bounded `enqueue_write_buffer` to
    /// avoid requiring a full-capacity write.
    fn write_buffer_range(&self, t: &mut Tensor, src: &[u8], dst_offset: usize) -> Result<()> {
        let dst_ptr = t.as_mut_ptr();
        if dst_ptr.is_null() {
            anyhow::bail!("Cannot write to null buffer (not mapped)");
        }
        let end = dst_offset
            .checked_add(src.len())
            .ok_or_else(|| anyhow::anyhow!("write_buffer_range: offset+len overflow"))?;
        if end > t.size() {
            anyhow::bail!(
                "write_buffer_range: out of bounds ({} + {} > {})",
                dst_offset,
                src.len(),
                t.size()
            );
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr.add(dst_offset), src.len());
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

    /// GPU 백엔드 여부. CPU: false, OpenCL/CUDA: true.
    fn is_gpu(&self) -> bool {
        false
    }

    /// Discrete GPU (non-UMA) 여부. UMA(Adreno/Jetson): false, discrete(NVIDIA desktop): true.
    fn is_discrete_gpu(&self) -> bool {
        false
    }

    /// Maximum single buffer allocation size in bytes.
    /// GPU backends return CL_DEVICE_MAX_MEM_ALLOC_SIZE; CPU returns usize::MAX.
    fn max_single_alloc(&self) -> usize {
        usize::MAX
    }

    /// Invalidate backend-internal caches keyed by weight tensor `cl_mem`
    /// addresses after a runtime weight swap (ENG-ALG-221 / INV-130).
    ///
    /// The OpenCL backend overrides this to clear
    /// `noshuffle_soa_registry` (HashMap keyed by old `cl_mem` addresses).
    /// `SwapExecutor::execute_on_slots` is contracted to call this exactly
    /// once per non-empty swap batch — before bumping `ratio_generation` so
    /// that the subsequent `FullKernelPlan` rebuild observes a clean slate
    /// and re-registers SOA descriptors against the new cl_mem keys.
    ///
    /// Device-only: the Adreno noshuffle Q4_0 GEMV path is the sole consumer.
    /// CPU / CUDA defaults are no-op. Callers may invoke unconditionally.
    fn invalidate_noshuffle_soa_registry(&self) {}

    /// GPU prefill flash attention. Returns Ok(true) if GPU dispatched, Ok(false) for CPU fallback.
    /// Default: CPU fallback (returns false).
    #[allow(unused_variables, clippy::too_many_arguments)]
    fn flash_attention_prefill(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        n_heads_q: usize,
        n_heads_kv: usize,
        seq_len: usize,
        cache_seq_len: usize,
        head_dim: usize,
        kv_capacity: usize,
        batch_size: usize,
        is_head_major: bool,
    ) -> Result<bool> {
        Ok(false)
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

#[cfg(test)]
mod tests {
    use super::Backend;
    use crate::backend::cpu::common::CpuBackendCommon;
    use crate::core::buffer::DType;
    use crate::core::memory::Memory;
    use crate::core::shape::Shape;
    use crate::core::tensor::Tensor;
    use crate::memory::galloc::Galloc;
    use std::sync::Arc;

    fn make_cpu_tensor(data: &[f32]) -> Tensor {
        let memory = Galloc::new();
        let buf = memory.alloc(data.len() * 4, DType::F32).unwrap();
        let mut t = Tensor::new(
            Shape::new(vec![data.len()]),
            buf,
            Arc::new(CpuBackendCommon::new()),
        );
        t.as_mut_slice::<f32>().copy_from_slice(data);
        t
    }

    #[test]
    fn test_gelu_tanh_known_values() {
        let backend = CpuBackendCommon::new();

        // gelu_tanh(0.0) * 1.0 ≈ 0.0
        let mut gate = make_cpu_tensor(&[0.0f32]);
        let up = make_cpu_tensor(&[1.0f32]);
        backend.gelu_tanh_mul(&mut gate, &up).unwrap();
        let result = gate.as_slice::<f32>()[0];
        assert!(
            result.abs() < 1e-6,
            "gelu_tanh(0)*1 should be ~0, got {}",
            result
        );

        // gelu_tanh(1.0) * 1.0 ≈ 0.8412
        let mut gate = make_cpu_tensor(&[1.0f32]);
        let up = make_cpu_tensor(&[1.0f32]);
        backend.gelu_tanh_mul(&mut gate, &up).unwrap();
        let result = gate.as_slice::<f32>()[0];
        assert!(
            (result - 0.8412).abs() < 1e-3,
            "gelu_tanh(1)*1 should be ~0.8412, got {}",
            result
        );

        // gelu_tanh(-1.0) * 1.0 ≈ -0.1588
        let mut gate = make_cpu_tensor(&[-1.0f32]);
        let up = make_cpu_tensor(&[1.0f32]);
        backend.gelu_tanh_mul(&mut gate, &up).unwrap();
        let result = gate.as_slice::<f32>()[0];
        assert!(
            (result - (-0.1588)).abs() < 1e-3,
            "gelu_tanh(-1)*1 should be ~-0.1588, got {}",
            result
        );
    }
}
