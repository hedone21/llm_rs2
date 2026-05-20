//! Memory layer (L2) — physical memory resources grouped by source.
//!
//! - [`galloc`]: generic heap allocator wrapper.
//! - [`host`]: host-managed buffers (heap Vec, mmap views).
//! - [`opencl`]: OpenCL cl_mem owners (device-only, unified, sub-region, host_ptr_pool, noshuffle SOA).
//! - [`cuda`]: CUDA driver allocations (managed/device/pinned + mmap alias).
//! - [`rpcmem`]: rpcmem (DMA-BUF heap) view adapters for cross-backend sharing.
//!
//! See ARCHITECTURE.md §13.8-D and the proud-strolling-whale plan (B안).
pub mod galloc;
pub mod host;
#[cfg(feature = "opencl")]
pub mod opencl;
#[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
pub mod cuda;
#[cfg(feature = "opencl")]
pub mod rpcmem;
pub mod secondary;
