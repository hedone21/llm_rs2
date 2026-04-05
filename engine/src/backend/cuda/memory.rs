//! CUDA memory allocator implementing the `Memory` trait.
//!
//! Phase 3: uses CudaHostBuffer (cuMemHostAlloc with DEVICEMAP) for pinned
//! zero-copy buffers accessible from both CPU and GPU via cuBLAS.

use crate::buffer::cuda_buffer::CudaHostBuffer;
use crate::core::buffer::{Buffer, DType};
use crate::core::memory::Memory;
use anyhow::Result;
use std::sync::Arc;

/// Memory allocator for the CUDA backend.
///
/// Allocates pinned host memory (cuMemHostAlloc) with GPU device mapping.
/// On Jetson (UMA), this provides zero-copy access from both CPU and GPU.
/// The device pointer can be passed directly to cuBLAS for GPU-accelerated compute.
pub struct CudaMemory;

impl Default for CudaMemory {
    fn default() -> Self {
        Self
    }
}

impl CudaMemory {
    pub fn new() -> Self {
        Self
    }
}

impl Memory for CudaMemory {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        let buf = CudaHostBuffer::new(size, dtype)?;
        Ok(Arc::new(buf))
    }

    fn used_memory(&self) -> usize {
        // TODO: track cumulative allocations if needed for pressure monitoring.
        0
    }
}
