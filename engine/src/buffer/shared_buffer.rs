use crate::core::buffer::{Buffer, DType};
use anyhow::Result;
#[cfg(feature = "opencl")]
use ocl::core::Mem;
use std::any::Any;

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

    /// Create a SharedBuffer that takes ownership of an existing Vec<u8>.
    pub fn from_vec(data: Vec<u8>, dtype: DType) -> Self {
        let size = data.len();
        Self { data, size, dtype }
    }
}

impl Buffer for SharedBuffer {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_buffer_creation() {
        let buffer = SharedBuffer::new(1024, DType::F32);
        assert_eq!(buffer.size(), 1024);
        assert_eq!(buffer.dtype(), DType::F32);

        let ptr = buffer.as_ptr();
        assert!(!ptr.is_null());

        let mut_ptr = buffer.as_mut_ptr();
        assert!(!mut_ptr.is_null());
    }

    #[test]
    fn test_shared_buffer_zero_size() {
        let buffer = SharedBuffer::new(0, DType::F16);
        assert_eq!(buffer.size(), 0);
        assert_eq!(buffer.dtype(), DType::F16);
        // Zero-sized allocation usually does not return a null pointer in Rust Vec
        let ptr = buffer.as_ptr();
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_shared_buffer_mutability_semantics() {
        let buffer = SharedBuffer::new(4, DType::U8);

        unsafe {
            let mut_ptr = buffer.as_mut_ptr();
            *mut_ptr.add(0) = 42;
            *mut_ptr.add(3) = 99;

            let read_ptr = buffer.as_ptr();
            assert_eq!(*read_ptr.add(0), 42);
            assert_eq!(*read_ptr.add(3), 99);
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_cl_mem_with_feature_opencl() {
        let buffer = SharedBuffer::new(16, DType::F32);
        assert!(buffer.cl_mem().is_none());
    }

    #[cfg(not(feature = "opencl"))]
    #[test]
    fn test_cl_mem_without_feature_opencl() {
        let buffer = SharedBuffer::new(16, DType::F32);
        assert!(buffer.cl_mem().is_none());
    }

    #[test]
    fn test_sync_device() {
        let buffer = SharedBuffer::new(16, DType::F32);
        assert!(buffer.sync_device().is_ok()); // Should always be Ok(())
    }
}
