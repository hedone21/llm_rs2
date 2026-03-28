use crate::core::buffer::{Buffer, DType};
use anyhow::Result;
use std::sync::Arc;

pub trait Memory: Send + Sync {
    /// Allocate a buffer of specific size and type
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>>;

    /// Allocate a KV cache buffer — madvise-capable on GPU (UMA) backends.
    /// Default: delegates to alloc() (CPU backends already return host-managed buffers).
    fn alloc_kv(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        self.alloc(size, dtype)
    }

    /// Total used memory in bytes
    fn used_memory(&self) -> usize;
}
