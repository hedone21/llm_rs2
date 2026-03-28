use crate::core::buffer::{Buffer, DType};
use anyhow::Result;
use ocl::core::Mem;
use ocl::{Buffer as OclBuffer, Queue};
use std::any::Any;
use std::ptr;

// unsafe impl Send for OpenCLBuffer {} // OclBuffer is Send/Sync usually?
// Check ocl docs. OclBuffer is Send/Sync.
// We don't need unsafe impl if fields are Send/Sync.

pub struct OpenCLBuffer {
    pub buffer: OclBuffer<u8>,
    queue: Queue,
    dtype: DType,
    size: usize,
}

impl OpenCLBuffer {
    pub fn new(queue: Queue, buffer: OclBuffer<u8>, size: usize, dtype: DType) -> Result<Self> {
        Ok(Self {
            buffer,
            queue,
            dtype,
            size,
        })
    }
}

impl Buffer for OpenCLBuffer {
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
        // Warning: Direct access not supported without mapping
        // Logic should use copy_from or map explicitly.
        // For now, returning null to satisfy trait.
        ptr::null()
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        ptr::null_mut()
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&Mem> {
        Some(self.buffer.as_core())
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    fn sync_device(&self) -> Result<()> {
        self.queue.finish()?;
        Ok(())
    }

    fn is_host_managed(&self) -> bool {
        false // Device-only GPU memory
    }
}
