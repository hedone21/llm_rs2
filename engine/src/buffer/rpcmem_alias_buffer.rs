//! `RpcmemAliasBuffer` — LISWAP-6 alias `cl_mem` for swapped weight tensors.
//!
//! Wraps an OpenCL `CL_MEM_USE_HOST_PTR` alias whose backing storage is a
//! per-layer rpcmem region owned by `RpcmemSecondaryStore`. The buffer itself
//! does not own the host memory — it pins the secondary mmap (`Arc<SecondaryMmap>`)
//! and the layer region (`Arc<RpcmemLayerRegion>`) so the rpcmem allocation
//! is freed only after every alias has been dropped (Drop ordering: cl_mem
//! → Arc<SecondaryMmap> → Arc<RpcmemLayerRegion> → rpcmem_free).
//!
//! Spec: ENG-DAT-094, INV-143 (alias lifetime via Arc retention).

#![cfg(feature = "opencl")]

use crate::core::buffer::{Buffer, DType};
use crate::models::weights::SecondaryMmap;
use anyhow::Result;
use ocl::core::Mem;
use std::any::Any;
use std::sync::Arc;

/// Alias `cl_mem` whose host pointer points into a per-layer rpcmem region.
///
/// **Read-only**: backed by the secondary GGUF mmap copy in rpcmem; weight
/// swap never writes through this alias. `as_mut_ptr()` returns the host
/// pointer (matching `as_ptr()`) only because some downstream paths
/// (`copy_weight_from`) consult `as_mut_ptr()` for the raw address; mutation
/// through it is contractually forbidden.
pub struct RpcmemAliasBuffer {
    /// CL alias handle (CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR).
    cl_buffer: Mem,
    /// Host pointer into the rpcmem region (`region_base + offset`).
    host_ptr: *mut u8,
    /// Tensor byte length.
    size: usize,
    /// Tensor dtype.
    dtype: DType,
    /// Lifetime guard 1: keep secondary store alive (it owns the layer-region map).
    _secondary_arc: Arc<SecondaryMmap>,
    /// Lifetime guard 2: keep this layer's rpcmem region alive even if the
    /// secondary store evicts the entry from its HashMap (defensive — current
    /// implementation never evicts, but the guarantee is explicit here).
    _layer_region: Arc<crate::models::weights::rpcmem_secondary::RpcmemLayerRegion>,
}

// SAFETY:
// - `cl_buffer` (`ocl::core::Mem`) is internally `Send + Sync` per `ocl` crate.
// - `host_ptr` is read-only; the rpcmem region is single-allocator and pinned
//   for the lifetime of `_layer_region`.
// - The two `Arc` lifetime guards prevent UAF: secondary store + layer region
//   both stay alive until the last alias is dropped.
unsafe impl Send for RpcmemAliasBuffer {}
unsafe impl Sync for RpcmemAliasBuffer {}

impl RpcmemAliasBuffer {
    /// Construct an alias from an existing `cl_mem` and the lifetime guards.
    ///
    /// Caller (typically the `OpenCLBackend::alloc_alias_weight_buffer`
    /// override) is responsible for creating the `cl_mem` with
    /// `CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY` against the same `host_ptr`.
    pub fn new(
        cl_buffer: Mem,
        host_ptr: *mut u8,
        size: usize,
        dtype: DType,
        secondary: Arc<SecondaryMmap>,
        layer_region: Arc<crate::models::weights::rpcmem_secondary::RpcmemLayerRegion>,
    ) -> Self {
        Self {
            cl_buffer,
            host_ptr,
            size,
            dtype,
            _secondary_arc: secondary,
            _layer_region: layer_region,
        }
    }
}

impl Buffer for RpcmemAliasBuffer {
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
        self.host_ptr as *const u8
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        // Returned only so size-of-pointer consumers (`copy_weight_from`
        // diagnostics) can identify the buffer; mutation through this
        // pointer is contractually forbidden (the underlying rpcmem region
        // mirrors a read-only mmap snapshot).
        self.host_ptr
    }

    fn cl_mem(&self) -> Option<&Mem> {
        Some(&self.cl_buffer)
    }

    fn sync_device(&self) -> Result<()> {
        // rpcmem DMA-BUF + USE_HOST_PTR alias share the same physical pages
        // on Adreno UMA; no explicit cache flush needed (matches
        // QnnOppkgKvBuffer semantics).
        Ok(())
    }

    fn is_host_managed(&self) -> bool {
        true
    }

    fn is_gpu_buffer(&self) -> bool {
        true
    }
}
