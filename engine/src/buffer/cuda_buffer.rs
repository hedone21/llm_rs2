//! CUDA Unified Memory buffer for Jetson UMA devices.
//!
//! Uses `cuMemAllocManaged` (via cudarc's `result` API) with `CU_MEM_ATTACH_GLOBAL`
//! so the returned pointer is accessible from both CPU and GPU without explicit copies.
//! This is the CUDA equivalent of OpenCL's `CL_MEM_ALLOC_HOST_PTR` zero-copy on Adreno.

use crate::core::buffer::{Buffer, DType};
use anyhow::{Result, anyhow};
use cudarc::driver::{result as cuda_result, sys as cuda_sys};
use std::any::Any;

/// A CUDA Unified Memory buffer.
///
/// Manages a raw `CUdeviceptr` allocated with `cuMemAllocManaged`.
/// On Jetson (UMA), this pointer is directly dereferenceable from CPU code.
/// Drop calls `cuMemFree` to release the allocation.
pub struct CudaBuffer {
    /// Raw CUDA device pointer (also valid as host pointer on UMA).
    dev_ptr: cuda_sys::CUdeviceptr,
    /// Total allocation size in bytes.
    size: usize,
    /// Logical data type for this buffer.
    dtype: DType,
}

// SAFETY: The underlying CUDA unified memory is thread-safe.
// cuMemAllocManaged with CU_MEM_ATTACH_GLOBAL makes the pointer accessible
// from any thread/stream. Access synchronization is the caller's responsibility
// (same as OpenCL zero-copy buffers).
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    /// Allocate `size` bytes of CUDA Unified Memory.
    ///
    /// Uses `CU_MEM_ATTACH_GLOBAL` so any stream/device can access the memory.
    /// On Jetson (UMA), the pointer is CPU-accessible without mapping.
    ///
    /// # Requirements
    /// A CUDA context must be current on the calling thread (typically set up
    /// by `CudaContext::new()` in `CudaBackend::new()`).
    pub fn new(size: usize, dtype: DType) -> Result<Self> {
        if size == 0 {
            return Err(anyhow!("CudaBuffer: cannot allocate 0 bytes"));
        }
        // SAFETY: cuMemAllocManaged requires a valid CUDA context on the current thread.
        // CudaBackend::new() ensures this via CudaContext::new() which calls cuInit+cuCtxCreate.
        // The returned pointer is uninitialized -- callers must write before reading.
        let dev_ptr = unsafe {
            cuda_result::malloc_managed(size, cuda_sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL)
                .map_err(|e| anyhow!("cuMemAllocManaged({size} bytes) failed: {e}"))?
        };
        Ok(Self {
            dev_ptr,
            size,
            dtype,
        })
    }

    /// Return the raw `CUdeviceptr` value (for passing to CUDA kernels).
    pub fn device_ptr(&self) -> cuda_sys::CUdeviceptr {
        self.dev_ptr
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        // SAFETY: We own this allocation (created in new()) and only free it once (here in Drop).
        // Any async work must have been synchronized before dropping.
        unsafe {
            let _ = cuda_result::memory_free(self.dev_ptr);
        }
    }
}

impl Buffer for CudaBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn size(&self) -> usize {
        self.size
    }

    /// CPU-accessible pointer on UMA (Jetson).
    ///
    /// SAFETY: On Jetson (UMA), `CUdeviceptr` from `cuMemAllocManaged` is a valid
    /// host pointer. On discrete GPUs this would segfault -- but Phase 1 targets UMA only.
    fn as_ptr(&self) -> *const u8 {
        self.dev_ptr as *const u8
    }

    /// Mutable CPU-accessible pointer on UMA (Jetson).
    fn as_mut_ptr(&self) -> *mut u8 {
        self.dev_ptr as *mut u8
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&ocl::core::Mem> {
        None
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    /// On UMA, host/device coherence is automatic; no explicit sync needed
    /// for data visibility. Kernel completion is handled by `Backend::synchronize()`.
    fn sync_device(&self) -> Result<()> {
        Ok(())
    }

    /// UMA: madvise is NOT effective on driver-managed unified memory.
    fn is_host_managed(&self) -> bool {
        false
    }

    /// This buffer is GPU-accessible.
    fn is_gpu_buffer(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Allocation tests require a real CUDA device.
    // Run with: cargo test --no-default-features --features cuda -- cuda_buffer

    #[test]
    fn test_cuda_buffer_dtype_accessors() {
        // Unit test that doesn't need a real device -- just checks compile-time logic.
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::F16.size(), 2);
    }
}
