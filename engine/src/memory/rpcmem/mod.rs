//! rpcmem (DMA-BUF heap) view adapters (L2).
//!
//! - [`opencl_alias`]: `RpcmemAliasBuffer` — rpcmem fd → cl_mem alias
//!   (CL_MEM_USE_HOST_PTR) for the OpenCL backend.
//!
//! `RpcmemLayerRegion` (fd allocator/lifecycle owner) stays in
//! `models/weights/rpcmem_secondary.rs` because it is structurally bound to
//! `RpcmemSecondaryStore` (D5'). The QNN consumer (`QnnBuffer::Rpcmem`) stays
//! in `backend/qnn_oppkg/` as a backend-internal view.
pub mod opencl_alias;
