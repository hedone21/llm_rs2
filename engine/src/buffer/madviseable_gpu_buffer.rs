use crate::core::buffer::{Buffer, DType};
use anyhow::{Result, anyhow};
use ocl::Context;
use ocl::core::Mem;
use std::any::Any;

/// MadviseableGPUBuffer: GPU-accessible buffer with madvise support.
///
/// Allocates host memory (aligned Vec) and wraps it with `CL_MEM_USE_HOST_PTR`
/// so that:
/// - GPU kernels access the same physical pages (zero-copy on UMA)
/// - `madvise(MADV_DONTNEED)` works because the app owns the memory (not driver-pinned)
///
/// This is used for KV cache buffers where eviction needs to release physical pages.
pub struct MadviseableGPUBuffer {
    /// App-managed host memory (madvise target)
    host_data: Vec<u8>,
    /// CL buffer wrapping host_data via CL_MEM_USE_HOST_PTR
    cl_buffer: Mem,
    size: usize,
    dtype: DType,
}

impl MadviseableGPUBuffer {
    /// Create a new MadviseableGPUBuffer.
    ///
    /// Allocates `size` bytes of host memory and creates a CL buffer
    /// pointing to it with `CL_MEM_USE_HOST_PTR`.
    pub fn new(context: &Context, size: usize, dtype: DType) -> Result<Self> {
        let host_data = vec![0u8; size];

        let cl_buffer = unsafe {
            ocl::core::create_buffer(
                context.as_core(),
                ocl::core::MEM_READ_WRITE | ocl::core::MEM_USE_HOST_PTR,
                size,
                Some(&host_data),
            )
        }
        .map_err(|e| anyhow!("Failed to create CL_MEM_USE_HOST_PTR buffer: {}", e))?;

        Ok(Self {
            host_data,
            cl_buffer,
            size,
            dtype,
        })
    }
}

impl Buffer for MadviseableGPUBuffer {
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
        self.host_data.as_ptr()
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.host_data.as_ptr() as *mut u8
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
        true // App-managed memory, madvise effective
    }
}

// SAFETY: host_data (Vec) is Send+Sync, cl_buffer (Mem) is Send+Sync.
// The CL buffer points to host_data which lives as long as this struct.
unsafe impl Send for MadviseableGPUBuffer {}
unsafe impl Sync for MadviseableGPUBuffer {}
