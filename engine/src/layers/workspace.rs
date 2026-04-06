use crate::core::backend::Backend;
use crate::core::buffer::{Buffer, DType};
use crate::core::memory::Memory;
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use anyhow::Result;
use std::sync::Arc;

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
    /// Pre-allocated KV cache cast buffers (F32→F16 conversion).
    /// Avoids GPU memory allocation per token per layer.
    pub k_cast: Option<Tensor>,
    pub v_cast: Option<Tensor>,
    /// Pre-allocated scratch buffers for CPU-GPU tensor partition (decode only).
    /// None when tensor partition is disabled.
    pub partition_ws: Option<PartitionWorkspace>,
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
        if let Some(ref pw) = self.partition_ws {
            bufs.push(pw.gate_gpu.buffer().clone());
            bufs.push(pw.gate_cpu.buffer().clone());
            bufs.push(pw.up_gpu.buffer().clone());
            bufs.push(pw.up_cpu.buffer().clone());
            bufs.push(pw.residual_cpu.buffer().clone());
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
            k_cast: self.k_cast.map(&retag),
            v_cast: self.v_cast.map(&retag),
            partition_ws: self.partition_ws.map(|pw| PartitionWorkspace {
                gate_gpu: retag(pw.gate_gpu),
                gate_cpu: retag(pw.gate_cpu),
                up_gpu: retag(pw.up_gpu),
                up_cpu: retag(pw.up_cpu),
                residual_cpu: retag(pw.residual_cpu),
            }),
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
            k_cast: None, // Lazily initialized on first use with correct dtype
            v_cast: None,
            partition_ws: None, // Set externally when tensor partition is enabled
        })
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

/// Pre-allocated scratch tensors for partitioned FFN output (decode only).
///
/// Each pair (gpu, cpu) holds one partial result that together form the full
/// output dimension. Created once during partition setup and reused every token.
pub struct PartitionWorkspace {
    /// GPU partial output for gate projection: [1, 1, split_row]
    pub gate_gpu: Tensor,
    /// CPU partial output for gate projection: [1, 1, out_dim - split_row]
    pub gate_cpu: Tensor,
    /// GPU partial output for up projection: [1, 1, split_row]
    pub up_gpu: Tensor,
    /// CPU partial output for up projection: [1, 1, out_dim - split_row]
    pub up_cpu: Tensor,
    /// CPU-side copy of residual for CPU matmul input: [1, 1, dim]
    /// UnifiedBuffer::as_ptr() returns null when unmapped, so we copy residual
    /// to this CPU buffer via read_buffer() before CPU matmul.
    pub residual_cpu: Tensor,
}

impl PartitionWorkspace {
    /// Allocate partition workspace buffers.
    ///
    /// `split_row`: number of output rows handled by GPU.
    /// `out_dim`: total output dimension of gate/up projections.
    /// `dim`: hidden dimension (for residual_cpu).
    /// `gpu_alloc`: allocator closure for GPU-side buffers. Caller decides the
    ///              concrete buffer type (MadviseableGPUBuffer for zero-copy,
    ///              UnifiedBuffer for standard OpenCL, CudaHostBuffer for CUDA, etc.).
    /// `gpu_backend`: backend for GPU-side tensors.
    /// `cpu_backend`: backend for CPU-side tensors.
    pub fn new(
        split_row: usize,
        out_dim: usize,
        dim: usize,
        gpu_alloc: &dyn Fn(usize, DType) -> Result<Arc<dyn Buffer>>,
        gpu_backend: Arc<dyn Backend>,
        cpu_backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        use crate::memory::galloc::Galloc;

        let cpu_rows = out_dim - split_row;
        let cpu_mem = Galloc::new();

        let gate_gpu_buf = gpu_alloc(split_row * 4, DType::F32)?;
        let gate_cpu_buf = cpu_mem.alloc(cpu_rows * 4, DType::F32)?;
        let up_gpu_buf = gpu_alloc(split_row * 4, DType::F32)?;
        let up_cpu_buf = cpu_mem.alloc(cpu_rows * 4, DType::F32)?;

        Ok(Self {
            gate_gpu: Tensor::new(
                Shape::new(vec![1, 1, split_row]),
                gate_gpu_buf,
                gpu_backend.clone(),
            ),
            gate_cpu: Tensor::new(
                Shape::new(vec![1, 1, cpu_rows]),
                gate_cpu_buf,
                cpu_backend.clone(),
            ),
            up_gpu: Tensor::new(Shape::new(vec![1, 1, split_row]), up_gpu_buf, gpu_backend),
            up_cpu: Tensor::new(
                Shape::new(vec![1, 1, cpu_rows]),
                up_cpu_buf,
                cpu_backend.clone(),
            ),
            residual_cpu: {
                let buf = cpu_mem.alloc(dim * 4, DType::F32)?;
                Tensor::new(Shape::new(vec![1, 1, dim]), buf, cpu_backend)
            },
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
