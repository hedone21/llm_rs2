use crate::core::tensor::Tensor;
use crate::core::shape::Shape;
use crate::core::buffer::DType;
use crate::core::memory::Memory;
use crate::core::backend::Backend;
use anyhow::Result;
use std::sync::Arc;

/// Pre-allocated workspace for a single transformer layer.
/// Reused across all tokens to avoid per-token memory allocation overhead.
pub struct LayerWorkspace {
    pub q: Tensor,          // [batch, 1, q_dim]
    pub k: Tensor,          // [batch, 1, k_dim]
    pub v: Tensor,          // [batch, 1, v_dim]
    pub out_attn: Tensor,   // [batch, 1, q_dim]
    pub gate: Tensor,       // [batch, 1, ffn_hidden]
    pub up: Tensor,         // [batch, 1, ffn_hidden]
    pub down: Tensor,       // [batch, 1, dim]
    pub residual: Tensor,   // [batch, 1, dim]
    pub attn_out: Tensor,   // [batch, 1, dim]
    /// Pre-allocated scores buffer for attention (max_seq_len size)
    pub scores: Vec<f32>,
}

impl LayerWorkspace {
    pub fn new(config: WorkspaceConfig, memory: &dyn Memory, backend: Arc<dyn Backend>) -> Result<Self> {
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
