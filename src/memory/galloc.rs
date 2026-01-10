use anyhow::Result;
use std::sync::Arc;
use crate::core::memory::Memory;
use crate::core::buffer::{Buffer, DType};
use crate::buffer::shared_buffer::SharedBuffer;

pub struct Galloc;

impl Galloc {
    pub fn new() -> Self {
        Self
    }
}

impl Memory for Galloc {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        let buf = SharedBuffer::new(size, dtype);
        Ok(Arc::new(buf))
    }

    fn used_memory(&self) -> usize {
        0 // Tracking not implemented
    }
}
