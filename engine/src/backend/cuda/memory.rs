//! CUDA Unified Memory allocator implementing the `Memory` trait.

use crate::buffer::cuda_buffer::CudaBuffer;
use crate::core::buffer::{Buffer, DType};
use crate::core::memory::Memory;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// Memory allocator that creates CUDA Unified Memory buffers.
///
/// Tracks total allocated bytes for memory pressure reporting.
/// Requires a CUDA context to be current on the calling thread
/// (set up by `CudaBackend::new()`).
pub struct CudaMemory {
    used: Mutex<usize>,
}

impl CudaMemory {
    /// Create a new CudaMemory allocator.
    ///
    /// The CUDA context must already be initialized (via `CudaBackend::new()`).
    pub fn new() -> Self {
        Self {
            used: Mutex::new(0),
        }
    }
}

impl Memory for CudaMemory {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        let buf = CudaBuffer::new(size, dtype)?;
        *self.used.lock().unwrap() += size;
        Ok(Arc::new(buf))
    }

    fn used_memory(&self) -> usize {
        *self.used.lock().unwrap()
    }
}
