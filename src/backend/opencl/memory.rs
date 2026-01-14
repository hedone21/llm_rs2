use crate::core::memory::Memory;
use crate::core::buffer::{Buffer, DType};
use crate::buffer::unified_buffer::UnifiedBuffer;
use super::buffer::OpenCLBuffer;
use ocl::{Context, Queue};
use ocl::flags::MemFlags;
use anyhow::Result;
use std::sync::{Arc, Mutex};


pub struct OpenCLMemory {
    context: Context,
    queue: Queue,
    used_memory: Mutex<usize>,
    /// If true, use UnifiedBuffer (zero-copy shared memory)
    /// If false, use OpenCLBuffer (device-only, faster)
    use_zero_copy: bool,
}

impl OpenCLMemory {
    pub fn new(context: Context, queue: Queue, use_zero_copy: bool) -> Self {
        Self {
            context,
            queue,
            used_memory: Mutex::new(0),
            use_zero_copy,
        }
    }
}

impl Memory for OpenCLMemory {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        let buffer: Arc<dyn Buffer> = if self.use_zero_copy {
            // Zero-copy shared memory (CPU-GPU accessible, but slower GPU kernels)
            Arc::new(UnifiedBuffer::new(self.queue.clone(), size, dtype)?)
        } else {
            // Device-only memory (faster GPU kernels, requires explicit copies)
            let ocl_buffer = ocl::Buffer::<u8>::builder()
                .queue(self.queue.clone())
                .flags(MemFlags::new().read_write())
                .len(size)
                .build()?;
            Arc::new(OpenCLBuffer::new(self.queue.clone(), ocl_buffer, size, dtype)?)
        };
        
        {
            let mut mem = self.used_memory.lock().unwrap();
            *mem += size;
        }

        Ok(buffer)
    }

    fn used_memory(&self) -> usize {
        *self.used_memory.lock().unwrap()
    }
}
