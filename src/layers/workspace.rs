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
    pub fn new(
        batch_size: usize,
        dim: usize,
        q_dim: usize,
        k_dim: usize,
        v_dim: usize,
        ffn_hidden: usize,
        n_heads: usize, // New argument
        max_seq_len: usize,
        memory: &dyn Memory,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let alloc = |shape: Vec<usize>| -> Result<Tensor> {
            let size: usize = shape.iter().product();
            let buf = memory.alloc(size * 4, DType::F32)?;
            Ok(Tensor::new(Shape::new(shape), buf, backend.clone()))
        };

        Ok(Self {
            q: alloc(vec![batch_size, 1, q_dim])?,
            k: alloc(vec![batch_size, 1, k_dim])?,
            v: alloc(vec![batch_size, 1, v_dim])?,
            out_attn: alloc(vec![batch_size, 1, q_dim])?,
            gate: alloc(vec![batch_size, 1, ffn_hidden])?,
            up: alloc(vec![batch_size, 1, ffn_hidden])?,
            down: alloc(vec![batch_size, 1, dim])?,
            residual: alloc(vec![batch_size, 1, dim])?,
            attn_out: alloc(vec![batch_size, 1, dim])?,
            scores: vec![0.0; n_heads * max_seq_len], // Resized
        })
    }
}
