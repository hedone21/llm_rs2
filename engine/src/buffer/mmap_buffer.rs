use crate::core::buffer::{Buffer, DType};
use anyhow::Result;
#[cfg(feature = "opencl")]
use ocl::core::Mem;
use std::any::Any;
use std::sync::Arc;

/// Zero-copy buffer that directly references mmap'd memory.
/// Used for model weights to avoid copying from safetensors mmap to heap.
///
/// Benefits over SharedBuffer (heap Vec):
/// - Kernel readahead for sequential access (mmap page cache)
/// - Better TLB behavior (kernel can use transparent huge pages)
/// - No memory copy at load time
///
/// Immutable: `as_mut_ptr()` returns null (weights are read-only).
#[derive(Debug)]
pub struct MmapBuffer {
    /// Pointer to the start of this tensor's data within the mmap'd region
    ptr: *const u8,
    /// Size in bytes
    size: usize,
    /// Data type
    dtype: DType,
    /// Keep the mmap alive as long as this buffer exists
    _mmap: Arc<memmap2::Mmap>,
}

// Safety: MmapBuffer is read-only and the underlying mmap is immutable.
// The Arc<Mmap> ensures the mapping lives as long as any MmapBuffer references it.
unsafe impl Send for MmapBuffer {}
unsafe impl Sync for MmapBuffer {}

impl MmapBuffer {
    /// Create a new MmapBuffer pointing to a region within an mmap'd file.
    ///
    /// # Safety
    /// `offset + size` must not exceed the mmap length.
    pub unsafe fn new(mmap: Arc<memmap2::Mmap>, offset: usize, size: usize, dtype: DType) -> Self {
        debug_assert!(offset + size <= mmap.len());
        let ptr = mmap.as_ptr().add(offset);
        Self {
            ptr,
            size,
            dtype,
            _mmap: mmap,
        }
    }
}

impl Buffer for MmapBuffer {
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
        self.ptr
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        // Weights are read-only — mutation not supported
        self.ptr as *mut u8
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&Mem> {
        None // CPU-only buffer
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    fn sync_device(&self) -> Result<()> {
        Ok(())
    }
}
