//! ClWrappedBuffer: Zero-alloc GPU wrapper for existing CPU buffers.
//!
//! Wraps an existing `Arc<dyn Buffer>` (Galloc, MmapBuffer, etc.) with a
//! `CL_MEM_USE_HOST_PTR` OpenCL handle. The original buffer owns the memory;
//! the CL handle just maps it for GPU access on UMA devices (Adreno, Mali).
//!
//! Key properties:
//! - `as_ptr()` / `as_mut_ptr()` → delegates to the original buffer (always valid)
//! - `cl_mem()` → returns the CL handle (valid for GPU kernels)
//! - **Zero additional memory allocation** — no Vec, no copy
//! - Original buffer's lifetime is tied via Arc (memory stays valid)

use crate::core::buffer::{Buffer, DType};
use anyhow::{Result, anyhow};
use ocl::core::Mem;
use std::any::Any;
use std::sync::Arc;

pub struct ClWrappedBuffer {
    /// Original buffer that owns the memory
    inner: Arc<dyn Buffer>,
    /// CL handle wrapping inner's host pointer
    cl_buffer: Mem,
    size: usize,
    dtype: DType,
}

// SAFETY: The inner buffer and CL handle are independently thread-safe.
unsafe impl Send for ClWrappedBuffer {}
unsafe impl Sync for ClWrappedBuffer {}

impl ClWrappedBuffer {
    /// Wrap an existing buffer with a CL_MEM_USE_HOST_PTR handle.
    ///
    /// The buffer's `as_ptr()` must return a valid, non-null pointer.
    /// On ARM UMA, this is zero-cost: GPU accesses the same physical DRAM.
    pub fn new(
        context: &ocl::Context,
        inner: Arc<dyn Buffer>,
        dtype: DType,
    ) -> Result<Self> {
        let ptr = inner.as_ptr();
        let size = inner.size();
        if ptr.is_null() {
            return Err(anyhow!("Cannot wrap null-pointer buffer with CL handle"));
        }

        let cl_buffer = unsafe {
            ocl::core::create_buffer(
                context.as_core(),
                ocl::core::MEM_READ_WRITE | ocl::core::MEM_USE_HOST_PTR,
                size,
                Some(std::slice::from_raw_parts(ptr, size)),
            )
        }
        .map_err(|e| anyhow!("Failed to create CL_MEM_USE_HOST_PTR wrapper: {}", e))?;

        Ok(Self {
            inner,
            cl_buffer,
            size,
            dtype,
        })
    }

    /// Get a reference to the cl_mem for use in `get_cl_mem()`.
    pub fn cl_mem_ref(&self) -> &Mem {
        &self.cl_buffer
    }
}

impl Buffer for ClWrappedBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn size(&self) -> usize {
        self.size
    }

    fn as_ptr(&self) -> *const u8 {
        self.inner.as_ptr()
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.inner.as_mut_ptr()
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&Mem> {
        Some(&self.cl_buffer)
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    fn sync_device(&self) -> Result<()> {
        Ok(())
    }

    fn is_host_managed(&self) -> bool {
        self.inner.is_host_managed()
    }
}
