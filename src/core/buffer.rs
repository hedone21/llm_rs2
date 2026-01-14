use anyhow::Result;
#[cfg(feature = "opencl")]
use ocl::core::Mem;

/// Data Type Enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Q4_0,
    Q4_1,
    F16,
    BF16,
    F32,
    U8,
}

impl DType {
    pub fn size(&self) -> usize {
        match self {
            DType::Q4_0 | DType::Q4_1 => 1, // Actually block quantized, handled separately usually
            DType::F16 | DType::BF16 => 2,
            DType::F32 => 4,
            DType::U8 => 1,
        }
    }
}

use std::any::Any;

/// Buffer Trait: Physical Memory Layer
pub trait Buffer: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    /// Returns the data type
    fn dtype(&self) -> DType;

    /// Returns the total size in bytes
    fn size(&self) -> usize;

    /// Read-only pointer for CPU access
    fn as_ptr(&self) -> *const u8;

    /// Mutable pointer for CPU access
    fn as_mut_ptr(&self) -> *mut u8;

    /// OpenCL memory handle (None for CPU-only buffers)
    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&Mem>;

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()>;

    /// Ensure data is synced to the desired device
    fn sync_device(&self) -> Result<()>;

    // --- Zero-Copy Shared Memory Support ---

    /// Map buffer for CPU access (makes GPU access invalid until unmapped).
    /// Default implementation is no-op for CPU-only buffers.
    fn map_for_cpu(&self) -> Result<()> {
        Ok(()) // No-op for regular buffers
    }

    /// Unmap buffer for GPU access (makes CPU pointer invalid).
    /// Default implementation is no-op for CPU-only buffers.
    fn unmap_for_gpu(&self) -> Result<()> {
        Ok(()) // No-op for regular buffers
    }

    /// Check if buffer is currently mapped for CPU access.
    /// Default: true (CPU-only buffers are always "mapped").
    fn is_mapped(&self) -> bool {
        true
    }
}
