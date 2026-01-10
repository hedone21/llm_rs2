#[cfg(feature = "opencl")]
use ocl::core::Mem;
use crate::core::buffer::{Buffer, DType};
use anyhow::Result;

#[derive(Debug)]
pub struct SharedBuffer {
    data: Vec<u8>, // Simulating shared buffer with Vec for CPU-only start
    size: usize,
    dtype: DType,
}

impl SharedBuffer {
    pub fn new(size: usize, dtype: DType) -> Self {
        // Ensure alignment if needed, for now standard Vec alignment
        // In real shared buffer, might use specialized allocator
        let data = vec![0u8; size];
        Self { data, size, dtype }
    }
}

impl Buffer for SharedBuffer {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn size(&self) -> usize {
        self.size
    }

    fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        // internal unsafe mutability if needed, or we change struct def
        // But Buffer trait demands &self -> *mut u8 which implies internal mutability or unsafety
        // Typically Buffer is Arc-ed and used carefully.
        self.data.as_ptr() as *mut u8
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&Mem> {
        None 
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    fn sync_device(&self) -> Result<()> {
        Ok(())
    }
}
