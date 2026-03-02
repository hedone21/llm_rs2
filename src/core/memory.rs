use crate::core::buffer::{Buffer, DType};
use anyhow::Result;
use std::sync::Arc;

pub trait Memory: Send + Sync {
    /// Allocate a buffer of specific size and type
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>>;

    /// Total used memory in bytes
    fn used_memory(&self) -> usize;
}
