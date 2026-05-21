//! OpenCL cl_mem owners + view adapters (L2).
//!
//! - [`device`]: device-only `OpenCLBuffer` (`clCreateBuffer` with `MEM_READ_WRITE`).
//! - [`unified`]: host-mappable `UnifiedBuffer` (`CL_MEM_ALLOC_HOST_PTR`).
//! - [`sub`]: zero-copy sub-region (`ClSubBuffer`, `clCreateSubBuffer`).
//! - [`host_ptr_pool_buffer`]: pool slot view (`HostPtrPoolBuffer`) — the
//!   allocator (`HostPtrPool`/`HostPtrPoolGuard`) stays in `backend/opencl/`
//!   because it directly imports `OpenCLBackend`.
//! - [`noshuffle`]: Q4_0 SOA replacement (`NoshuffleWeightBuffer`).
//!
//! See ARCHITECTURE.md §13.8-D / proud-strolling-whale plan (B안).
pub mod device;
pub mod host_ptr_pool_buffer;
pub mod noshuffle;
pub mod sub;
pub mod unified;
