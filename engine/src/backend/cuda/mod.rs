//! CUDA backend for Jetson (SM >= 7.2, UMA).
//!
//! Phase 1: all compute ops delegate to CpuBackend via the shared UMA pointer.
//! `copy_from` allocates CudaBuffer (Unified Memory) and memcpy's from source.
//! Future phases will add cuBLAS GEMV and custom CUDA kernels.

pub mod memory;

use crate::buffer::cuda_buffer::CudaBuffer;
use crate::core::backend::Backend;
use crate::core::buffer::Buffer;
use crate::core::tensor::Tensor;
use anyhow::{Result, anyhow};
use cudarc::driver::sys as cuda_sys;
use cudarc::driver::{CudaContext, result as cuda_result};
use std::any::Any;
use std::sync::Arc;

/// CUDA backend targeting Jetson UMA devices.
///
/// Phase 1: all math ops are CPU-fallback via Unified Memory pointers.
/// The CpuBackendCommon is stateless and used for all compute.
#[derive(Clone)]
pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    device_name: String,
    compute_capability: (i32, i32),
    is_uma: bool,
}

impl CudaBackend {
    /// Initialize CUDA backend on device 0.
    ///
    /// Requirements:
    /// - Compute Capability >= 7.2 (Jetson AGX Xavier / Volta+)
    /// - Managed Memory support
    pub fn new() -> Result<Self> {
        Self::with_ordinal(0)
    }

    /// Initialize CUDA backend on a specific device ordinal.
    pub fn with_ordinal(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)
            .map_err(|e| anyhow!("CUDA context creation failed (ordinal={ordinal}): {e}"))?;

        // Query compute capability
        let cc_major = ctx
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| anyhow!("Failed to query CC major: {e}"))?;
        let cc_minor = ctx
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| anyhow!("Failed to query CC minor: {e}"))?;

        if cc_major < 7 || (cc_major == 7 && cc_minor < 2) {
            return Err(anyhow!(
                "CUDA backend requires SM >= 7.2 (Jetson Xavier+), got sm_{cc_major}{cc_minor}"
            ));
        }

        // Check managed memory support
        let managed_mem = ctx
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)
            .unwrap_or(0);
        if managed_mem == 0 {
            return Err(anyhow!("Device does not support managed (unified) memory"));
        }

        // Check UMA (unified addressing)
        let unified_addr = ctx
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)
            .unwrap_or(0);
        let is_uma = unified_addr != 0;

        // Get device name
        let cu_device = cuda_result::device::get(ordinal as i32)
            .map_err(|e| anyhow!("Failed to get CUDA device: {e}"))?;
        let device_name = cuda_result::device::get_name(cu_device)
            .unwrap_or_else(|_| format!("CUDA Device {ordinal}"));

        eprintln!(
            "[CUDA] Device: {} (sm_{}{}, UMA={}, managed_mem={})",
            device_name, cc_major, cc_minor, is_uma, managed_mem
        );

        Ok(Self {
            ctx,
            device_name,
            compute_capability: (cc_major, cc_minor),
            is_uma,
        })
    }

    /// Access the underlying CUDA context (for buffer allocation, kernel launches, etc).
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Compute capability as (major, minor).
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }
}

/// Internal helper: get a CPU backend for fallback compute.
/// CpuBackendCommon is stateless, so this is cheap.
fn cpu_fallback() -> crate::backend::cpu::common::CpuBackendCommon {
    crate::backend::cpu::common::CpuBackendCommon::new()
}

impl Backend for CudaBackend {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "CUDA"
    }

    fn device(&self) -> &str {
        &self.device_name
    }

    fn is_gpu(&self) -> bool {
        true
    }

    fn is_discrete_gpu(&self) -> bool {
        !self.is_uma
    }

    // --- Math ops: CPU fallback via UMA pointers ---
    // On Jetson (UMA), CudaBuffer::as_ptr() returns a CPU-accessible pointer,
    // so CpuBackend can directly operate on the data without copies.

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        cpu_fallback().matmul(a, b, out)
    }

    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        cpu_fallback().matmul_transposed(a, b, out)
    }

    fn matmul_slice(
        &self,
        a: &Tensor,
        b: &Tensor,
        rows: usize,
        cols: usize,
        out: &mut Tensor,
    ) -> Result<()> {
        cpu_fallback().matmul_slice(a, b, rows, cols, out)
    }

    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        cpu_fallback().add_assign(a, b)
    }

    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()> {
        cpu_fallback().scale(x, v)
    }

    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        cpu_fallback().silu_mul(a, b)
    }

    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32, add_unit: bool) -> Result<()> {
        cpu_fallback().rms_norm(x, w, eps, add_unit)
    }

    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        cpu_fallback().softmax(x)
    }

    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        cpu_fallback().rope_inplace(x, start_pos, theta)
    }

    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        cpu_fallback().cast(src, dst)
    }

    fn copy_from(&self, src: &Tensor) -> Result<Tensor> {
        let size = src.size();
        let cuda_buf = CudaBuffer::new(size, src.dtype())?;

        // Copy data from source to unified memory.
        // SAFETY: Both pointers are valid. src from any Buffer (CPU or GPU),
        // dst from cuMemAllocManaged (valid host pointer on UMA).
        let src_ptr = src.as_ptr();
        let dst_ptr = cuda_buf.as_mut_ptr();
        if !src_ptr.is_null() && !dst_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
            }
        }

        // Return tensor with this backend. Clone is cheap (Arc<CudaContext> refcount).
        Ok(Tensor::new(
            src.shape().clone(),
            Arc::new(cuda_buf),
            Arc::new(self.clone()),
        ))
    }

    fn synchronize(&self) -> Result<()> {
        self.ctx
            .default_stream()
            .synchronize()
            .map_err(|e| anyhow!("CUDA synchronize failed: {e}"))
    }
}
