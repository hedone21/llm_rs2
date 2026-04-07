use super::buffer::OpenCLBuffer;
use crate::buffer::unified_buffer::UnifiedBuffer;
use crate::core::buffer::{Buffer, DType};
use crate::core::memory::Memory;
use anyhow::Result;
use ocl::flags::MemFlags;
use ocl::{Context, Queue};
use std::sync::{Arc, Mutex};

pub struct OpenCLMemory {
    #[allow(dead_code)]
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
            Arc::new(OpenCLBuffer::new(
                self.queue.clone(),
                ocl_buffer,
                size,
                dtype,
            )?)
        };

        {
            let mut mem = self.used_memory.lock().unwrap();
            *mem += size;
        }

        Ok(buffer)
    }

    fn alloc_kv(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        if self.use_zero_copy {
            // KV cache: CL_MEM_ALLOC_HOST_PTR (driver-managed, single VMA).
            // Starts UNMAPPED — GPU writes via cl_mem during forward pass.
            // Map for CPU access only during SwitchHw (via kv_migrate).
            // Note: OpenCL spec says writing to a mapped buffer via cl_mem is UB.
            let buffer = UnifiedBuffer::new(self.queue.clone(), size, dtype)?;
            let buf: Arc<dyn Buffer> = Arc::new(buffer);
            {
                let mut mem = self.used_memory.lock().unwrap();
                *mem += size;
            }
            Ok(buf)
        } else {
            self.alloc(size, dtype)
        }
    }

    fn used_memory(&self) -> usize {
        *self.used_memory.lock().unwrap()
    }
}
