use crate::core::memory::Memory;
use crate::core::buffer::{Buffer, DType};
use super::buffer::OpenCLBuffer;
use ocl::{Context, Queue};
use ocl::flags::MemFlags;
use anyhow::Result;
use std::sync::{Arc, Mutex};


pub struct OpenCLMemory {
    context: Context,
    queue: Queue,
    used_memory: Mutex<usize>,
}

impl OpenCLMemory {
    pub fn new(context: Context, queue: Queue) -> Self {
        Self {
            context,
            queue,
            used_memory: Mutex::new(0),
        }
    }
}

impl Memory for OpenCLMemory {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        // Standard allocation (No host ptr for now)
        let buffer = ocl::Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .flags(MemFlags::new().read_write())
            .len(size)
            .build()?;


        let cl_buffer = OpenCLBuffer::new(self.queue.clone(), buffer, size, dtype)?;
        
        {
            let mut mem = self.used_memory.lock().unwrap();
            *mem += size;
        }

        Ok(Arc::new(cl_buffer))
    }

    fn used_memory(&self) -> usize {
        *self.used_memory.lock().unwrap()
    }
}
