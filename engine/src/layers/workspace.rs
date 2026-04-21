use crate::core::backend::Backend;
use crate::core::buffer::{Buffer, DType};
use crate::core::memory::Memory;
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use anyhow::Result;
use std::cell::UnsafeCell;
use std::sync::Arc;

/// Thin wrapper around `UnsafeCell<PartitionWorkspace>` that makes the
/// interior `Sync` for the restricted single-threaded-dispatch safety model
/// enforced by the OpenCL plan (`PartitionStep::run`) and forward_gen
/// (shared TLS thread). Do not use outside of that invariant.
pub struct PartitionWsCell(pub UnsafeCell<PartitionWorkspace>);

impl PartitionWsCell {
    pub fn new(ws: PartitionWorkspace) -> Self {
        Self(UnsafeCell::new(ws))
    }

    #[inline]
    pub fn get(&self) -> *mut PartitionWorkspace {
        self.0.get()
    }
}

// SAFETY: The LayerWorkspace single-threaded-dispatch contract (see
// `LayerWorkspace::partition_ws` doc) guarantees no aliased mutable access
// across threads. Both the forward_gen partition block and the plan-path
// `PartitionStep::run` execute on the same dispatch thread.
unsafe impl Send for PartitionWsCell {}
unsafe impl Sync for PartitionWsCell {}

/// Pre-allocated workspace for a single transformer layer.
/// Reused across all tokens to avoid per-token memory allocation overhead.
pub struct LayerWorkspace {
    pub q: Tensor,        // [batch, 1, q_dim]
    pub k: Tensor,        // [batch, 1, k_dim]
    pub v: Tensor,        // [batch, 1, v_dim]
    pub out_attn: Tensor, // [batch, 1, q_dim]
    pub gate: Tensor,     // [batch, 1, ffn_hidden]
    pub up: Tensor,       // [batch, 1, ffn_hidden]
    pub down: Tensor,     // [batch, 1, dim]
    pub residual: Tensor, // [batch, 1, dim]
    pub attn_out: Tensor, // [batch, 1, dim]
    /// Pre-allocated scores buffer for attention (max_seq_len size)
    pub scores: Vec<f32>,
    /// Score buffer offset: kv_start_pos for local attention layers, 0 for global.
    /// Used to correctly map ws.scores[t] to cache position (score_offset + t).
    pub score_offset: usize,
    /// Pre-allocated KV cache cast buffers (F32→F16 conversion).
    /// Avoids GPU memory allocation per token per layer.
    pub k_cast: Option<Tensor>,
    pub v_cast: Option<Tensor>,
    /// Pre-allocated scratch buffers for CPU-GPU tensor partition (decode only).
    /// None when tensor partition is disabled.
    ///
    /// Wrapped in `Arc<UnsafeCell<..>>` so the OpenCL plan path
    /// (`build_partitioned_layer_plan`) can share ownership with the
    /// forward_gen partition path — both mutate the workspace through the
    /// same allocation. Safety: single-threaded dispatch (same model as
    /// `KernelStep`). The Arc clone that the plan retains bumps the refcount
    /// so the workspace outlives the plan even if `LayerWorkspace` is
    /// reallocated mid-flight (e.g. on UMA switch). See
    /// `backend/opencl/plan.rs::PartitionPlanContext` for the dispatch-side
    /// invariant enforcement.
    pub partition_ws: Option<Arc<PartitionWsCell>>,
    /// Fused-merge carry slots: when `LLMRS_PARTITION_FUSED_MERGE=1`, the
    /// previous layer's partition FFN end leaves its `down_partial_gpu` and
    /// `cpu_merge_staging` (CPU partial uploaded) tensors here so the next
    /// layer's entry can fuse them with the residual + attn_norm into a
    /// single kernel. Cleared at the start of each forward_into call and
    /// whenever the path is disabled. The tensors are shallow clones (Arc
    /// buffers) of `partition_ws.down_partial_gpu` / `cpu_merge_staging`.
    pub partition_prev_gpu_partial: Option<Tensor>,
    pub partition_prev_cpu_staging: Option<Tensor>,
}

impl LayerWorkspace {
    /// Extract all buffer Arcs (for keeping GPU buffers alive during switch).
    pub fn take_buffers(&self) -> Vec<Arc<dyn crate::core::buffer::Buffer>> {
        let mut bufs = vec![
            self.q.buffer().clone(),
            self.k.buffer().clone(),
            self.v.buffer().clone(),
            self.out_attn.buffer().clone(),
            self.gate.buffer().clone(),
            self.up.buffer().clone(),
            self.down.buffer().clone(),
            self.residual.buffer().clone(),
            self.attn_out.buffer().clone(),
        ];
        if let Some(ref t) = self.k_cast {
            bufs.push(t.buffer().clone());
        }
        if let Some(ref t) = self.v_cast {
            bufs.push(t.buffer().clone());
        }
        if let Some(ref pw_cell) = self.partition_ws {
            // SAFETY: single-threaded dispatch (see field doc).
            let pw: &PartitionWorkspace = unsafe { &*pw_cell.get() };
            bufs.push(pw.gate_gpu.buffer().clone());
            bufs.push(pw.gate_cpu.buffer().clone());
            bufs.push(pw.up_gpu.buffer().clone());
            bufs.push(pw.up_cpu.buffer().clone());
            bufs.push(pw.residual_cpu.buffer().clone());
            bufs.push(pw.attn_out_cpu.buffer().clone());
            bufs.push(pw.x_cpu.buffer().clone());
            bufs.push(pw.down_partial_gpu.buffer().clone());
            bufs.push(pw.down_partial_cpu.buffer().clone());
            bufs.push(pw.cpu_merge_staging.buffer().clone());
        }
        bufs
    }

    /// Re-tag all tensors with a new backend without reallocating buffers.
    /// Used for UMA GPU↔CPU switch where buffers are already host-accessible.
    pub fn retag_backend(self, backend: Arc<dyn Backend>) -> Self {
        let retag = |t: Tensor| -> Tensor {
            Tensor::new(t.shape().clone(), t.buffer().clone(), backend.clone())
        };
        Self {
            q: retag(self.q),
            k: retag(self.k),
            v: retag(self.v),
            out_attn: retag(self.out_attn),
            gate: retag(self.gate),
            up: retag(self.up),
            down: retag(self.down),
            residual: retag(self.residual),
            attn_out: retag(self.attn_out),
            scores: self.scores,
            score_offset: self.score_offset,
            k_cast: self.k_cast.map(&retag),
            v_cast: self.v_cast.map(&retag),
            partition_ws: self.partition_ws.map(|pw_cell| {
                // `retag_backend` is only called on UMA GPU↔CPU switch, which
                // invalidates any active plan beforehand — so the Arc should
                // be unique here. Fall back to deep-cloning if unexpectedly
                // shared.
                let cell = Arc::try_unwrap(pw_cell).unwrap_or_else(|shared| {
                    // SAFETY: single-threaded dispatch invariant.
                    let pw: &PartitionWorkspace = unsafe { &*shared.get() };
                    PartitionWsCell::new(PartitionWorkspace {
                        gate_gpu: pw.gate_gpu.clone(),
                        gate_cpu: pw.gate_cpu.clone(),
                        up_gpu: pw.up_gpu.clone(),
                        up_cpu: pw.up_cpu.clone(),
                        residual_cpu: pw.residual_cpu.clone(),
                        attn_out_cpu: pw.attn_out_cpu.clone(),
                        x_cpu: pw.x_cpu.clone(),
                        down_partial_gpu: pw.down_partial_gpu.clone(),
                        down_partial_cpu: pw.down_partial_cpu.clone(),
                        cpu_merge_staging: pw.cpu_merge_staging.clone(),
                        ready_flag: pw.ready_flag.clone(),
                    })
                });
                let pw = cell.0.into_inner();
                Arc::new(PartitionWsCell::new(PartitionWorkspace {
                    gate_gpu: retag(pw.gate_gpu),
                    gate_cpu: retag(pw.gate_cpu),
                    up_gpu: retag(pw.up_gpu),
                    up_cpu: retag(pw.up_cpu),
                    residual_cpu: retag(pw.residual_cpu),
                    attn_out_cpu: retag(pw.attn_out_cpu),
                    x_cpu: retag(pw.x_cpu),
                    down_partial_gpu: retag(pw.down_partial_gpu),
                    down_partial_cpu: retag(pw.down_partial_cpu),
                    cpu_merge_staging: retag(pw.cpu_merge_staging),
                    ready_flag: retag(pw.ready_flag),
                }))
            }),
            partition_prev_gpu_partial: self.partition_prev_gpu_partial.map(&retag),
            partition_prev_cpu_staging: self.partition_prev_cpu_staging.map(&retag),
        }
    }

    pub fn new(
        config: WorkspaceConfig,
        memory: &dyn Memory,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let alloc = |shape: Vec<usize>| -> Result<Tensor> {
            let size: usize = shape.iter().product();
            let buf = memory.alloc(size * 4, DType::F32)?;
            Ok(Tensor::new(Shape::new(shape), buf, backend.clone()))
        };

        Ok(Self {
            q: alloc(vec![config.batch_size, 1, config.q_dim])?,
            k: alloc(vec![config.batch_size, 1, config.k_dim])?,
            v: alloc(vec![config.batch_size, 1, config.v_dim])?,
            out_attn: alloc(vec![config.batch_size, 1, config.q_dim])?,
            gate: alloc(vec![config.batch_size, 1, config.ffn_hidden])?,
            up: alloc(vec![config.batch_size, 1, config.ffn_hidden])?,
            down: alloc(vec![config.batch_size, 1, config.dim])?,
            residual: alloc(vec![config.batch_size, 1, config.dim])?,
            attn_out: alloc(vec![config.batch_size, 1, config.dim])?,
            scores: vec![0.0; config.n_heads * config.max_seq_len],
            score_offset: 0,
            k_cast: None, // Lazily initialized on first use with correct dtype
            v_cast: None,
            partition_ws: None, // Set externally when tensor partition is enabled
            partition_prev_gpu_partial: None,
            partition_prev_cpu_staging: None,
        })
    }

    /// Clear fused-merge carry slots. Called at the start of every
    /// `forward_into` to ensure layer 0 does not consume stale state from a
    /// previous token's final layer.
    pub fn reset_partition_prev(&mut self) {
        self.partition_prev_gpu_partial = None;
        self.partition_prev_cpu_staging = None;
    }
}

pub struct WorkspaceConfig {
    pub batch_size: usize,
    pub dim: usize,
    pub q_dim: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub ffn_hidden: usize,
    pub n_heads: usize,
    pub max_seq_len: usize,
}

/// Pre-allocated workspace for prefill (batch token processing).
/// Reuses GPU buffers across layers to avoid alloc/free churn that crashes NVIDIA's OpenCL driver.
pub struct PrefillWorkspace {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub out_attn: Tensor,
    pub attn_out_proj: Tensor,
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
    pub residual: Tensor,
    pub residual_ffn: Tensor,
    /// Lazily initialized cast buffers for F16/Q4 KV cache
    pub k_cast: Option<Tensor>,
    pub v_cast: Option<Tensor>,
    seq_len: usize,
}

impl PrefillWorkspace {
    pub fn new(
        config: &WorkspaceConfig,
        seq_len: usize,
        memory: &dyn Memory,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let alloc = |shape: Vec<usize>| -> Result<Tensor> {
            let size: usize = shape.iter().product();
            let buf = memory.alloc(size * 4, DType::F32)?;
            Ok(Tensor::new(Shape::new(shape), buf, backend.clone()))
        };
        let b = config.batch_size;
        Ok(Self {
            q: alloc(vec![b, seq_len, config.q_dim])?,
            k: alloc(vec![b, seq_len, config.k_dim])?,
            v: alloc(vec![b, seq_len, config.v_dim])?,
            out_attn: alloc(vec![b, seq_len, config.q_dim])?,
            attn_out_proj: alloc(vec![b, seq_len, config.dim])?,
            gate: alloc(vec![b, seq_len, config.ffn_hidden])?,
            up: alloc(vec![b, seq_len, config.ffn_hidden])?,
            down: alloc(vec![b, seq_len, config.dim])?,
            residual: alloc(vec![b, seq_len, config.dim])?,
            residual_ffn: alloc(vec![b, seq_len, config.dim])?,
            k_cast: None,
            v_cast: None,
            seq_len,
        })
    }

    /// Current sequence length this workspace is sized for.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}

/// Pre-allocated scratch tensors for partitioned FFN gate/up output (decode only).
///
/// Each pair (gpu, cpu) holds one partial result that together form the full
/// output dimension. Created once during partition setup and reused every token.
///
/// Only FFN gate/up are partitioned; attention and FFN down run GPU-only.
pub struct PartitionWorkspace {
    // --- FFN gate/up (Strategy B: CPU keeps its partial; GPU keeps its partial) ---
    /// GPU partial output for gate projection: [1, 1, gate_split_row]
    pub gate_gpu: Tensor,
    /// CPU partial output for gate projection: [1, 1, ffn_hidden - gate_split_row]
    pub gate_cpu: Tensor,
    /// GPU partial output for up projection: [1, 1, up_split_row]
    pub up_gpu: Tensor,
    /// CPU partial output for up projection: [1, 1, ffn_hidden - up_split_row]
    pub up_cpu: Tensor,
    /// CPU-side copy of residual for CPU matmul input: [1, 1, dim]
    /// UnifiedBuffer::as_ptr() returns null when unmapped, so we copy residual
    /// to this CPU buffer via read_buffer() before CPU matmul.
    pub residual_cpu: Tensor,
    /// CPU-side copy of attention output for Direction A (compute replication):
    /// [1, 1, dim]. When `LLMRS_PARTITION_REPLICATE_NORM=1` (default when
    /// partition is active), the partition block asynchronously DMA-reads
    /// `ws.attn_out` into this buffer and then the CPU runs its own
    /// `add_rms_norm_oop(x, attn_out_cpu, residual_cpu, ffn_norm, eps, false)`
    /// in parallel with the GPU's matching call on `ws.residual`. This
    /// advances the synchronization point from after `add_rms_norm_oop` to
    /// after `attn_out` is ready, allowing the host wait window to overlap
    /// with the GPU FFN chain enqueue.
    pub attn_out_cpu: Tensor,
    /// CPU-side copy of the layer input `x` for Direction A fallback.
    /// [1, 1, dim]. When `x` is not host-accessible (UMA `UnifiedBuffer`
    /// without a current `map()`, which is the common case on Adreno), the
    /// partition block asynchronously DMA-reads `x` into this buffer alongside
    /// the `attn_out` read, then the CPU's `add_rms_norm_oop` consumes it.
    /// Mirrors `attn_out_cpu`'s allocation pattern.
    pub x_cpu: Tensor,

    // --- FFN down (Strategy B) ---
    /// GPU partial output for down projection: [1, 1, hidden_size]
    /// Contains `W_down[:, :split_col] @ silu(gate_gpu)*up_gpu`.
    pub down_partial_gpu: Tensor,
    /// CPU partial output for down projection: [1, 1, hidden_size]
    /// Contains `W_down[:, split_col:] @ silu(gate_cpu)*up_cpu`.
    pub down_partial_cpu: Tensor,
    /// GPU-side staging buffer to upload the CPU partial for the final
    /// elementwise add: [1, 1, hidden_size].
    pub cpu_merge_staging: Tensor,
    /// Host-visible 4-byte flag (CL_MEM_ALLOC_HOST_PTR, permanent-mapped).
    /// Written by the `kernel_add_rms_norm_oop_f4_sigflag` kernel variant
    /// via `atomic_xchg` once the `residual` output is globally visible;
    /// `PartitionStep::run` spin-polls this flag instead of calling
    /// `clFinish(queue)` when `LLMRS_PARTITION_POLL_FLAG=1`. Eliminates
    /// per-layer driver round-trip latency (~0.29 ms on Adreno 830).
    pub ready_flag: Tensor,
}

impl PartitionWorkspace {
    /// Allocate partition workspace buffers for FFN gate/up partitioned projections.
    ///
    /// `ctx`: partition context from layer 0 (all layers share the same split geometry).
    /// `ffn_hidden`: FFN intermediate dimension (gate/up out_dim).
    /// `hidden_size`: model hidden dimension (residual size).
    /// `gpu_alloc`: allocator closure for GPU-side buffers.
    /// `gpu_backend`: backend for GPU-side tensors.
    /// `cpu_backend`: backend for CPU-side tensors.
    pub fn new(
        ctx: &crate::layers::tensor_partition::PartitionContext,
        ffn_hidden: usize,
        hidden_size: usize,
        gpu_alloc: &dyn Fn(usize, DType) -> Result<Arc<dyn Buffer>>,
        gpu_backend: Arc<dyn Backend>,
        cpu_backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        use crate::memory::galloc::Galloc;

        let cpu_mem = Galloc::new();

        // Helper: allocate a (gpu, cpu) tensor pair for a partitioned weight.
        let alloc_pair = |pw: &crate::layers::tensor_partition::PartitionedWeight,
                          out_dim: usize|
         -> Result<(Tensor, Tensor)> {
            let sr = pw.split_row;
            let cr = out_dim - sr;
            let gpu_buf = gpu_alloc(sr * 4, DType::F32)?;
            let cpu_buf = cpu_mem.alloc(cr * 4, DType::F32)?;
            Ok((
                Tensor::new(Shape::new(vec![1, 1, sr]), gpu_buf, gpu_backend.clone()),
                Tensor::new(Shape::new(vec![1, 1, cr]), cpu_buf, cpu_backend.clone()),
            ))
        };

        // FFN gate/up only
        let (gate_gpu, gate_cpu) = alloc_pair(&ctx.gate, ffn_hidden)?;
        let (up_gpu, up_cpu) = alloc_pair(&ctx.up, ffn_hidden)?;

        // FFN down (Strategy B whole-slice):
        //   down_partial_gpu  [1, 1, hidden]  GPU F32
        //   down_partial_cpu  [1, 1, hidden]  CPU F32
        //   cpu_merge_staging [1, 1, hidden]  GPU F32  (upload target for CPU partial)
        let down_partial_gpu = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            gpu_alloc(hidden_size * 4, DType::F32)?,
            gpu_backend.clone(),
        );
        let cpu_merge_staging = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            gpu_alloc(hidden_size * 4, DType::F32)?,
            gpu_backend.clone(),
        );
        // Phase 1a: permanent-map `cpu_merge_staging` UnifiedBuffer so the
        // plan-path merge can memcpy directly from `down_partial_cpu` into the
        // GPU-visible host pointer, eliminating the per-layer
        // `enqueue_write_buffer` (~0.13 ms/layer × 16 = 2.1 ms/tok on S25).
        // Opt-out via LLMRS_PARTITION_ZCOPY_MERGE=0 for A/B benchmarking.
        #[cfg(feature = "opencl")]
        if std::env::var("LLMRS_PARTITION_ZCOPY_MERGE")
            .map(|v| v != "0")
            .unwrap_or(true)
            && let Some(ub) = cpu_merge_staging
                .buffer()
                .as_any()
                .downcast_ref::<crate::buffer::unified_buffer::UnifiedBuffer>()
        {
            let _ = ub.map();
        }
        let down_partial_cpu = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            cpu_mem.alloc(hidden_size * 4, DType::F32)?,
            cpu_backend.clone(),
        );

        let residual_cpu = {
            let buf = cpu_mem.alloc(hidden_size * 4, DType::F32)?;
            Tensor::new(
                Shape::new(vec![1, 1, hidden_size]),
                buf,
                cpu_backend.clone(),
            )
        };
        let attn_out_cpu = {
            let buf = cpu_mem.alloc(hidden_size * 4, DType::F32)?;
            Tensor::new(
                Shape::new(vec![1, 1, hidden_size]),
                buf,
                cpu_backend.clone(),
            )
        };
        let x_cpu = {
            let buf = cpu_mem.alloc(hidden_size * 4, DType::F32)?;
            Tensor::new(Shape::new(vec![1, 1, hidden_size]), buf, cpu_backend)
        };
        // Ready-flag: 4 bytes, GPU-visible + CPU-visible via ALLOC_HOST_PTR.
        // Permanent-mapped so `as_mut_ptr()` yields the host pointer the
        // CPU polls while the sigflag kernel signals completion.
        let ready_flag = Tensor::new(
            Shape::new(vec![1]),
            gpu_alloc(4, DType::F32)?,
            gpu_backend.clone(),
        );
        #[cfg(feature = "opencl")]
        if let Some(ub) = ready_flag
            .buffer()
            .as_any()
            .downcast_ref::<crate::buffer::unified_buffer::UnifiedBuffer>()
        {
            let _ = ub.map();
        }

        Ok(Self {
            gate_gpu,
            gate_cpu,
            up_gpu,
            up_cpu,
            residual_cpu,
            attn_out_cpu,
            x_cpu,
            down_partial_gpu,
            down_partial_cpu,
            cpu_merge_staging,
            ready_flag,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::galloc::Galloc;

    fn llama_1b_config() -> WorkspaceConfig {
        WorkspaceConfig {
            batch_size: 1,
            dim: 2048,
            q_dim: 2048,
            k_dim: 512,
            v_dim: 512,
            ffn_hidden: 8192,
            n_heads: 32,
            max_seq_len: 2048,
        }
    }

    fn minimal_config() -> WorkspaceConfig {
        WorkspaceConfig {
            batch_size: 1,
            dim: 64,
            q_dim: 64,
            k_dim: 16,
            v_dim: 16,
            ffn_hidden: 128,
            n_heads: 1,
            max_seq_len: 32,
        }
    }

    #[test]
    fn test_workspace_allocation_shapes() {
        let memory = Galloc::new();
        let backend = Arc::new(CpuBackend::new());
        let cfg = llama_1b_config();
        let ws = LayerWorkspace::new(cfg, &memory, backend).unwrap();

        assert_eq!(ws.q.shape().dims(), &[1, 1, 2048]);
        assert_eq!(ws.k.shape().dims(), &[1, 1, 512]);
        assert_eq!(ws.v.shape().dims(), &[1, 1, 512]);
        assert_eq!(ws.out_attn.shape().dims(), &[1, 1, 2048]);
        assert_eq!(ws.gate.shape().dims(), &[1, 1, 8192]);
        assert_eq!(ws.up.shape().dims(), &[1, 1, 8192]);
        assert_eq!(ws.down.shape().dims(), &[1, 1, 2048]);
        assert_eq!(ws.residual.shape().dims(), &[1, 1, 2048]);
        assert_eq!(ws.attn_out.shape().dims(), &[1, 1, 2048]);
    }

    #[test]
    fn test_workspace_scores_size() {
        let memory = Galloc::new();
        let backend = Arc::new(CpuBackend::new());
        let cfg = llama_1b_config();
        let ws = LayerWorkspace::new(cfg, &memory, backend).unwrap();

        // scores = n_heads * max_seq_len = 32 * 2048 = 65536
        assert_eq!(ws.scores.len(), 32 * 2048);
        // All initialized to zero
        assert!(ws.scores.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_workspace_small_config() {
        let memory = Galloc::new();
        let backend = Arc::new(CpuBackend::new());
        let cfg = minimal_config();
        let ws = LayerWorkspace::new(cfg, &memory, backend).unwrap();

        assert_eq!(ws.q.shape().dims(), &[1, 1, 64]);
        assert_eq!(ws.k.shape().dims(), &[1, 1, 16]);
        assert_eq!(ws.gate.shape().dims(), &[1, 1, 128]);
        assert_eq!(ws.scores.len(), 1 * 32); // 1 head * 32 max_seq
    }

    #[test]
    fn test_workspace_tensors_writable() {
        let memory = Galloc::new();
        let backend = Arc::new(CpuBackend::new());
        let cfg = minimal_config();
        let ws = LayerWorkspace::new(cfg, &memory, backend).unwrap();

        // All tensors should have non-null writable pointers
        assert!(!ws.q.as_mut_ptr().is_null(), "q pointer is null");
        assert!(!ws.k.as_mut_ptr().is_null(), "k pointer is null");
        assert!(!ws.v.as_mut_ptr().is_null(), "v pointer is null");
        assert!(
            !ws.out_attn.as_mut_ptr().is_null(),
            "out_attn pointer is null"
        );
        assert!(!ws.gate.as_mut_ptr().is_null(), "gate pointer is null");
        assert!(!ws.up.as_mut_ptr().is_null(), "up pointer is null");
        assert!(!ws.down.as_mut_ptr().is_null(), "down pointer is null");
        assert!(
            !ws.residual.as_mut_ptr().is_null(),
            "residual pointer is null"
        );
        assert!(
            !ws.attn_out.as_mut_ptr().is_null(),
            "attn_out pointer is null"
        );

        // Verify buffer sizes are correct (numel * 4 bytes for F32)
        assert_eq!(ws.q.size(), 64 * 4);
        assert_eq!(ws.gate.size(), 128 * 4);
    }
}
