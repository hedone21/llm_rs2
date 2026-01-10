use std::sync::Arc;
use anyhow::{Result, anyhow};
use crate::core::shape::Shape;
use crate::core::buffer::{Buffer, DType};
use crate::core::backend::Backend;

#[derive(Clone)]
pub struct Tensor {
    shape: Shape,
    buffer: Arc<dyn Buffer>,
    backend: Arc<dyn Backend>,
}

impl Tensor {
    pub fn new(shape: Shape, buffer: Arc<dyn Buffer>, backend: Arc<dyn Backend>) -> Self {
        Self { shape, buffer, backend }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn buffer(&self) -> &Arc<dyn Buffer> {
        &self.buffer
    }
    
    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    pub fn dtype(&self) -> DType {
        self.buffer.dtype()
    }

    pub fn size(&self) -> usize {
        self.buffer.size()
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    // Accessors
    pub fn as_ptr(&self) -> *const u8 {
        self.buffer.as_ptr()
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.buffer.as_mut_ptr()
    }

    pub fn as_slice<T>(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.as_ptr() as *const T,
                self.size() / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn as_mut_slice<T>(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.as_mut_ptr() as *mut T,
                self.size() / std::mem::size_of::<T>(),
            )
        }
    }

    // Operations delegated to backend
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        // Implementation would need an output tensor created via backend/memory
        // For now, this requires a way to allocate output. 
        // We usually pass 'out' or create it. 
        // Let's assume we'll use a lower-level API or the user creates 'out'.
        // Or we can add a helper if Backend supports allocation (it usually doesn't directly, Memory does).
        Err(anyhow!("Use backend.matmul directly for now"))
    }
    
    pub fn to_device(&mut self, backend: Arc<dyn Backend>) -> Result<()> {
        // If current backend is same, no-op
        if self.backend.name() == backend.name() {
            return Ok(());
        }
        
        // Use new backend to copy
        let new_tensor = backend.copy_from(self)?;
        *self = new_tensor;
        Ok(())
    }
}
