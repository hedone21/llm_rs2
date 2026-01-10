use anyhow::Result;
use crate::core::tensor::Tensor;

pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn device(&self) -> &str;

    // Basic Math
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()>;
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()>;
    fn matmul_slice(&self, a: &Tensor, b: &Tensor, rows: usize, cols: usize, out: &mut Tensor) -> Result<()>;

    // In-place operations
    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()>;
    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()>;

    // Activation & Norm
    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()>;
    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32) -> Result<()>;
    fn softmax(&self, x: &mut Tensor) -> Result<()>;

    // Rotate
    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()>;

    // Memory Ops
    fn copy_from(&self, t: &Tensor) -> Result<Tensor>;
}
