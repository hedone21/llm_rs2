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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::buffer::Buffer;
    use std::any::Any;

    // Dummy Buffer for testing
    struct DummyBuffer {
        size: usize,
        dtype: DType,
        data: Vec<u8>
    }
    impl DummyBuffer {
        fn new(size: usize, dtype: DType) -> Self {
            Self { size, dtype, data: vec![0; size] }
        }
    }
    impl Buffer for DummyBuffer {
        fn as_any(&self) -> &dyn Any { self }
        fn dtype(&self) -> DType { self.dtype }
        fn size(&self) -> usize { self.size }
        fn as_ptr(&self) -> *const u8 { self.data.as_ptr() }
        fn as_mut_ptr(&self) -> *mut u8 { 
            // This is meant as a dummy buffer, so we cast to mut ptr. 
            // Safe enough for these tests which don't actually mutate.
            self.data.as_ptr() as *mut u8 
        }
        #[cfg(feature = "opencl")]
        fn cl_mem(&self) -> Option<&ocl::core::Mem> { None }
        #[cfg(not(feature = "opencl"))]
        fn cl_mem(&self) -> Option<()> { None }
        fn sync_device(&self) -> Result<()> { Ok(()) }
        fn map_for_cpu(&self) -> Result<()> { Ok(()) }
        fn unmap_for_gpu(&self) -> Result<()> { Ok(()) }
    }

    // Dummy Backend for testing
    struct DummyBackend;
    impl Backend for DummyBackend {
        fn as_any(&self) -> &dyn Any { self }
        fn name(&self) -> &str { "dummy" }
        fn device(&self) -> &str { "cpu" }
        // Basic Math
        fn matmul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> Result<()> { Ok(()) }
        fn matmul_transposed(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> Result<()> { Ok(()) }
        fn matmul_slice(&self, _a: &Tensor, _b: &Tensor, _rows: usize, _cols: usize, _out: &mut Tensor) -> Result<()> { Ok(()) }
        // In-place operations
        fn add_assign(&self, _a: &mut Tensor, _b: &Tensor) -> Result<()> { Ok(()) }
        fn scale(&self, _x: &mut Tensor, _v: f32) -> Result<()> { Ok(()) }
        // Activation & Norm
        fn silu_mul(&self, _a: &mut Tensor, _b: &Tensor) -> Result<()> { Ok(()) }
        fn rms_norm(&self, _x: &mut Tensor, _weight: &Tensor, _epsilon: f32) -> Result<()> { Ok(()) }
        fn softmax(&self, _x: &mut Tensor) -> Result<()> { Ok(()) }
        // Rotate
        fn rope_inplace(&self, _x: &mut Tensor, _start_pos: usize, _theta: f32) -> Result<()> { Ok(()) }
        // Memory Ops
        fn copy_from(&self, _source: &Tensor) -> Result<Tensor> { Ok(_source.clone()) }
        // Type casting
        fn cast(&self, _src: &Tensor, _dst: &mut Tensor) -> Result<()> { Ok(()) }
    }

    #[test]
    fn test_tensor_creation_and_metadata() {
        let shape = Shape::new(vec![2, 3]);
        // 6 f32 elements = 24 bytes
        let buffer = Arc::new(DummyBuffer::new(24, DType::F32));
        let backend = Arc::new(DummyBackend);

        let tensor = Tensor::new(shape, buffer, backend);
        
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.size(), 24);
        assert_eq!(tensor.numel(), 6);
    }

    #[test]
    fn test_tensor_as_slice_bounds() {
        let shape = Shape::new(vec![5]);
        // 5 f32 elements = 20 bytes
        let buffer = Arc::new(DummyBuffer::new(20, DType::F32));
        let backend = Arc::new(DummyBackend);

        let mut tensor = Tensor::new(shape, buffer, backend);
        
        let slice = tensor.as_slice::<f32>();
        // Even with null pointers, internally it slices from the raw ptr with len = 20 / 4 = 5
        // We can't actually read from it, but we can verify the length of the slice representation.
        assert_eq!(slice.len(), 5);
        
        let mut_slice = tensor.as_mut_slice::<f32>();
        assert_eq!(mut_slice.len(), 5);
    }
    
    #[test]
    fn test_tensor_matmul_unimplemented() {
        let shape = Shape::new(vec![2, 2]);
        let buffer = Arc::new(DummyBuffer::new(16, DType::F32));
        let backend = Arc::new(DummyBackend);
        let tensor1 = Tensor::new(shape.clone(), buffer.clone(), backend.clone());
        let tensor2 = Tensor::new(shape, buffer, backend);
        
        assert!(tensor1.matmul(&tensor2).is_err());
    }

    struct DummyBackendSameName;
    impl Backend for DummyBackendSameName {
        fn as_any(&self) -> &dyn Any { self }
        fn name(&self) -> &str { "dummy" }
        fn device(&self) -> &str { "cpu" }
        // Basic Math
        fn matmul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> Result<()> { Ok(()) }
        fn matmul_transposed(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> Result<()> { Ok(()) }
        fn matmul_slice(&self, _a: &Tensor, _b: &Tensor, _rows: usize, _cols: usize, _out: &mut Tensor) -> Result<()> { Ok(()) }
        // In-place operations
        fn add_assign(&self, _a: &mut Tensor, _b: &Tensor) -> Result<()> { Ok(()) }
        fn scale(&self, _x: &mut Tensor, _v: f32) -> Result<()> { Ok(()) }
        // Activation & Norm
        fn silu_mul(&self, _a: &mut Tensor, _b: &Tensor) -> Result<()> { Ok(()) }
        fn rms_norm(&self, _x: &mut Tensor, _weight: &Tensor, _epsilon: f32) -> Result<()> { Ok(()) }
        fn softmax(&self, _x: &mut Tensor) -> Result<()> { Ok(()) }
        // Rotate
        fn rope_inplace(&self, _x: &mut Tensor, _start_pos: usize, _theta: f32) -> Result<()> { Ok(()) }
        // Memory Ops
        fn copy_from(&self, _source: &Tensor) -> Result<Tensor> { Err(anyhow::anyhow!("Should not be called")) }
        // Type casting
        fn cast(&self, _src: &Tensor, _dst: &mut Tensor) -> Result<()> { Ok(()) }
    }

    #[test]
    fn test_tensor_to_device() {
        let shape = Shape::new(vec![2, 2]);
        let buffer = Arc::new(DummyBuffer::new(16, DType::F32));
        let backend = Arc::new(DummyBackend);
        let mut tensor = Tensor::new(shape, buffer, backend);

        let same_backend = Arc::new(DummyBackendSameName);
        assert!(tensor.to_device(same_backend).is_ok()); // Should return Ok(()) without calling copy_from
    }
}
