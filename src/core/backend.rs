use anyhow::Result;
use crate::core::tensor::Tensor;

pub trait Backend: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    fn name(&self) -> &str;
    fn device(&self) -> &str;

    // Basic Math
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()>;
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()>;
    fn matmul_slice(&self, a: &Tensor, b: &Tensor, rows: usize, cols: usize, out: &mut Tensor) -> Result<()>;

    // In-place operations
    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()>;
    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()>;

    // Activation & Norm
    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()>;
    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32) -> Result<()>;
    fn softmax(&self, x: &mut Tensor) -> Result<()>;

    // Rotate
    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()>;

    // Memory Ops
    fn copy_from(&self, t: &Tensor) -> Result<Tensor>;
    fn read_buffer(&self, t: &Tensor, dst: &mut [u8]) -> Result<()> {
        let src_ptr = unsafe { t.buffer().as_ptr() as *const u8 };
        if src_ptr.is_null() {
            anyhow::bail!("Cannot read null buffer (not mapped)");
        }
        unsafe { std::ptr::copy_nonoverlapping(src_ptr, dst.as_mut_ptr(), dst.len()); }
        Ok(())
    }

    // Synchronization (for benchmarking)
    fn synchronize(&self) -> Result<()> { Ok(()) }
    
    // Embedding Lookup / Gather
    // src: [Rows, Cols] (Embeddings)
    // indices: [NumIndices] (Indices)
    // dst: [NumIndices, Cols] (Output)
    fn gather(&self, src: &Tensor, indices: &Tensor, dst: &mut Tensor) -> Result<()> {
        // Default CPU implementation
        let src_data = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.size()/4) };
        let idx_data = unsafe { std::slice::from_raw_parts(indices.as_ptr() as *const u32, indices.size()/4) }; // U8 buffer, U32 data
        let dst_data = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.size()/4) };
        
        let cols = src.shape().dims()[1];
        
        // Validation?
        
        for (i, &idx) in idx_data.iter().enumerate() {
            let offset = idx as usize * cols;
            let target_offset = i * cols;
            if offset + cols <= src_data.len() && target_offset + cols <= dst_data.len() {
                 dst_data[target_offset..target_offset+cols].copy_from_slice(&src_data[offset..offset+cols]);
            }
        }
        Ok(())
    }

    // New API: Copy slice from src to dst
    // src_offset/dst_offset are ELEMENT offsets (not bytes) if Tensor is typed, but here for simplicity let's assume they are ELEMENT offsets relative to Tensor's DType?
    // Actually, Tensors are somewhat untyped regarding pointer arithmetic in this trait unless we know DType size.
    // Let's assume count is number of ELEMENTS. And offsets are ELEMENTS.
    // Caller must ensure types match.
    fn copy_slice(&self, src: &Tensor, dst: &mut Tensor, src_offset: usize, dst_offset: usize, count: usize) -> Result<()> {
        let type_size = match src.dtype() {
            crate::core::buffer::DType::F32 => 4,
            crate::core::buffer::DType::F16 => 2,
            crate::core::buffer::DType::U8 => 1,
            crate::core::buffer::DType::Q4_0 => std::mem::size_of::<crate::core::quant::BlockQ4_0>(),            _ => 1, // Fallback
        };
        
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();
        
        if src_ptr.is_null() || dst_ptr.is_null() {
            anyhow::bail!("Null pointer in copy_slice (default impl), likely OpenCL buffer mismatch");
        }
        
        unsafe {
            let src_u8 = src_ptr as *const u8;
            let dst_u8 = dst_ptr as *mut u8;
            
            // Calculate byte offsets
            let src_byte_offset = src_offset * type_size;
            let dst_byte_offset = dst_offset * type_size;
            let byte_count = count * type_size;
            
            std::ptr::copy_nonoverlapping(src_u8.add(src_byte_offset), dst_u8.add(dst_byte_offset), byte_count);
        }
        Ok(())
    }
}
