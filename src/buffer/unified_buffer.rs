use crate::core::buffer::{Buffer, DType};
use ocl::{Buffer as OclBuffer, Queue, flags};
use ocl::core::Mem;
use anyhow::{Result, anyhow};
use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::ptr;

/// UnifiedBuffer: CPU-GPU Zero-Copy Shared Memory
/// 
/// Uses CL_MEM_ALLOC_HOST_PTR to allocate memory that can be accessed
/// by both CPU and GPU without explicit data copies.
/// 
/// # Usage Pattern
/// ```ignore
/// // 1. Create buffer (starts mapped for CPU initialization)
/// let buffer = UnifiedBuffer::new(...)?;
/// 
/// // 2. Write data via as_mut_ptr() while mapped
/// let ptr = buffer.as_mut_ptr();
/// // ... CPU operations ...
/// 
/// // 3. Unmap for GPU access
/// buffer.unmap_for_gpu()?;
/// // ... GPU kernel execution ...
/// 
/// // 4. Map again for CPU reads
/// buffer.map_for_cpu()?;
/// ```
pub struct UnifiedBuffer {
    cl_buffer: OclBuffer<u8>,
    queue: Queue,
    /// Mapped pointer (valid only when is_mapped is true)
    mapped_ptr: Mutex<*mut u8>,
    size: usize,
    dtype: DType,
    is_mapped: AtomicBool,
}

// SAFETY: OpenCL buffer handles are thread-safe.
// Access to mapped_ptr is protected by Mutex.
unsafe impl Send for UnifiedBuffer {}
unsafe impl Sync for UnifiedBuffer {}

impl UnifiedBuffer {
    /// Create a new UnifiedBuffer with CL_MEM_ALLOC_HOST_PTR flag.
    /// The buffer starts in **unmapped** state (GPU-accessible).
    /// Call `map_for_cpu()` when CPU access is needed.
    pub fn new(queue: Queue, size: usize, dtype: DType) -> Result<Self> {
        // Create buffer with ALLOC_HOST_PTR for zero-copy
        let cl_buffer = OclBuffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR)
            .len(size)
            .build()
            .map_err(|e| anyhow!("Failed to create unified buffer: {}", e))?;

        // Start unmapped (GPU-accessible) for optimal kernel performance
        Ok(Self {
            cl_buffer,
            queue,
            mapped_ptr: Mutex::new(ptr::null_mut()),
            size,
            dtype,
            is_mapped: AtomicBool::new(false), // Starts unmapped
        })
    }

    /// Map the buffer for CPU access using high-level API.
    pub fn map(&self) -> Result<*mut u8> {
        // Check if already mapped
        if self.is_mapped.load(Ordering::Acquire) {
            let guard = self.mapped_ptr.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
            return Ok(*guard);
        }

        // Map the buffer using high-level API
        let mapped = unsafe {
            self.cl_buffer.map()
                .read()
                .write()
                .len(self.size)
                .enq()
                .map_err(|e| anyhow!("Failed to map buffer: {}", e))?
        };

        let raw_ptr = mapped.as_ptr() as *mut u8;
        // Forget the MemMap to keep the mapping alive
        std::mem::forget(mapped);

        let mut guard = self.mapped_ptr.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        *guard = raw_ptr;
        self.is_mapped.store(true, Ordering::Release);

        Ok(raw_ptr)
    }

    /// Unmap the buffer for GPU access.
    /// Uses queue.finish() to ensure previous operations are complete.
    pub fn unmap(&self) -> Result<()> {
        if !self.is_mapped.load(Ordering::Acquire) {
            return Ok(()); // Already unmapped
        }

        let mut guard = self.mapped_ptr.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let ptr = *guard;
        
        if ptr.is_null() {
            return Ok(());
        }

        // Use queue.finish() to ensure all operations on mapped memory are complete
        // Then the next map will get fresh data
        self.queue.finish()?;

        *guard = ptr::null_mut();
        self.is_mapped.store(false, Ordering::Release);

        Ok(())
    }

    /// Get the underlying OpenCL buffer for kernel execution.
    pub fn cl_buffer(&self) -> &OclBuffer<u8> {
        &self.cl_buffer
    }
}

impl Buffer for UnifiedBuffer {
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
        if !self.is_mapped.load(Ordering::Acquire) {
            return ptr::null();
        }
        match self.mapped_ptr.lock() {
            Ok(guard) => *guard as *const u8,
            Err(_) => ptr::null(),
        }
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        if !self.is_mapped.load(Ordering::Acquire) {
            return ptr::null_mut();
        }
        match self.mapped_ptr.lock() {
            Ok(guard) => *guard,
            Err(_) => ptr::null_mut(),
        }
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&Mem> {
        Some(self.cl_buffer.as_core())
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    fn sync_device(&self) -> Result<()> {
        self.queue.finish()?;
        Ok(())
    }

    fn map_for_cpu(&self) -> Result<()> {
        self.map()?;
        Ok(())
    }

    fn unmap_for_gpu(&self) -> Result<()> {
        self.unmap()
    }

    fn is_mapped(&self) -> bool {
        self.is_mapped.load(Ordering::Acquire)
    }
}

impl Drop for UnifiedBuffer {
    fn drop(&mut self) {
        // Sync to ensure any operations are complete
        let _ = self.queue.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocl::{Platform, Device, Context, Queue};

    fn create_test_queue() -> Result<Queue> {
        // Use try to avoid panic when no OpenCL is available
        let platform = Platform::list().into_iter().next()
            .ok_or_else(|| anyhow!("No OpenCL platform available"))?;
        let device = Device::first(platform)?;
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;
        let queue = Queue::new(&context, device, None)?;
        Ok(queue)
    }

    #[test]
    fn test_alloc_unified_buffer() {
        let queue = match create_test_queue() {
            Ok(q) => q,
            Err(_) => return, // Skip if no OpenCL device
        };

        let buffer = UnifiedBuffer::new(queue, 1024, DType::F32).unwrap();
        assert_eq!(buffer.size(), 1024);
        assert!(buffer.is_mapped()); // Starts mapped
    }

    #[test]
    fn test_map_returns_valid_ptr() {
        let queue = match create_test_queue() {
            Ok(q) => q,
            Err(_) => return,
        };

        let buffer = UnifiedBuffer::new(queue, 1024, DType::F32).unwrap();
        let ptr = buffer.as_mut_ptr();
        
        assert!(!ptr.is_null());
        assert!(buffer.is_mapped());
        
        buffer.unmap().unwrap();
        assert!(!buffer.is_mapped());
    }

    #[test]
    fn test_unmap_and_remap() {
        let queue = match create_test_queue() {
            Ok(q) => q,
            Err(_) => return,
        };

        let buffer = UnifiedBuffer::new(queue, 1024, DType::F32).unwrap();
        
        // Initially mapped
        assert!(buffer.is_mapped());
        
        // Unmap
        buffer.unmap().unwrap();
        assert!(!buffer.is_mapped());
        
        // Remap
        buffer.map().unwrap();
        assert!(buffer.is_mapped());
    }

    #[test]
    fn test_map_write_unmap_cycle() {
        let queue = match create_test_queue() {
            Ok(q) => q,
            Err(_) => return,
        };

        let buffer = UnifiedBuffer::new(queue, 256, DType::F32).unwrap();
        
        // Write while mapped
        let ptr = buffer.as_mut_ptr();
        unsafe {
            let f32_ptr = ptr as *mut f32;
            for i in 0..64 {
                *f32_ptr.add(i) = i as f32;
            }
        }
        
        // Unmap (simulating GPU access)
        buffer.unmap().unwrap();
        
        // Map again and verify
        let ptr = buffer.map().unwrap();
        unsafe {
            let f32_ptr = ptr as *const f32;
            assert_eq!(*f32_ptr.add(0), 0.0);
            assert_eq!(*f32_ptr.add(10), 10.0);
            assert_eq!(*f32_ptr.add(63), 63.0);
        }
    }
}
