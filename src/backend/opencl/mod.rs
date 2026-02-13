#![allow(unused_unsafe)]
use anyhow::{Result, anyhow};
use std::sync::{Arc, Mutex};
use crate::core::backend::Backend;
use crate::core::tensor::Tensor;
use crate::core::memory::Memory;
use ocl::{Context, Device, Platform, Queue, Program, flags};
use ocl::core::Kernel as CoreKernel;
use crate::core::buffer::DType;
use crate::core::buffer::Buffer;
use crate::buffer::unified_buffer::UnifiedBuffer;

pub mod buffer;
pub mod memory;

/// Helper function to get the OpenCL memory handle from a tensor buffer.
/// Works with both UnifiedBuffer and legacy OpenCLBuffer.
fn get_cl_mem(buf: &dyn Buffer) -> Result<&ocl::core::Mem> {
    // First try UnifiedBuffer
    if let Some(unified) = buf.as_any().downcast_ref::<UnifiedBuffer>() {
        return Ok(unified.cl_buffer().as_core());
    }
    // Then try legacy OpenCLBuffer
    if let Some(ocl_buf) = buf.as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>() {
        return Ok(ocl_buf.buffer.as_core());
    }
    Err(anyhow!("Buffer is not an OpenCL buffer type"))
}

// Compiler optimization flags for fast math (matching llama.cpp)
const CL_FAST_MATH_OPTS: &str = "-cl-std=CL2.0 -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math";

/// Cached kernel objects wrapped in a struct for interior mutability.
/// Uses Mutex to make the raw kernel pointers thread-safe.
struct KernelCache {
    kernel_mul_mat_f32_f32: CoreKernel,
    kernel_mul_mat_q4_0_f32: CoreKernel,
    kernel_rms_norm_opt: CoreKernel,
    kernel_softmax_opt: CoreKernel,
    kernel_rope_simple: CoreKernel,
    kernel_silu_mul_simple: CoreKernel,
    kernel_add_assign_simple: CoreKernel,
    kernel_scale_simple: CoreKernel,
    kernel_get_rows_q4_0: CoreKernel,
    kernel_get_rows_f32: CoreKernel,
    kernel_attn_gen: CoreKernel,
    kernel_cast_f32_to_f16: CoreKernel,
    kernel_attn_gen_half: CoreKernel,
    kernel_quantize_f32_to_q4_0: CoreKernel,
}

// SAFETY: OpenCL kernel objects are thread-safe for clSetKernelArg + clEnqueueNDRangeKernel
// when protected by a mutex. The underlying cl_kernel is not mutated by these operations,
// only the kernel arguments are set before each call.
unsafe impl Send for KernelCache {}
unsafe impl Sync for KernelCache {}

/// OpenCL Backend with cached kernel objects for performance.
/// Kernels are created once during initialization and reused across all calls.
pub struct OpenCLBackend {
    pub context: Context,
    pub queue: Queue,
    pub device: Device,
    // Programs
    pub program: Program,
    pub simple_ops_program: Program,
    pub q4_0_program: Program,
    pub quant_q4_0_program: Program,
    pub get_rows_program: Program,
    
    // Cached kernels protected by mutex for thread safety
    kernels: Mutex<KernelCache>,
}

// Manual Debug impl
impl std::fmt::Debug for OpenCLBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenCLBackend")
            .field("device", &self.device)
            .finish()
    }
}

impl OpenCLBackend {
    pub fn new() -> Result<Self> {
        let platform = Platform::default();
        let device = Device::list(platform, Some(flags::DEVICE_TYPE_GPU))?
            .into_iter()
            .next()
            .unwrap_or_else(|| Device::first(platform).expect("No OpenCL devices found"));

        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;

        let queue = Queue::new(&context, device, None)?;

        log::info!("Initialized OpenCL Backend on device: {}", device.name()?);

        // Load matmul kernel with fast math
        let matmul_src = include_str!("../../../kernels/mul_mv_f32_f32.cl");
        let program = match Program::builder()
            .devices(device)
            .src(matmul_src)
            .cmplr_opt(CL_FAST_MATH_OPTS)
            .build(&context) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("WARN: Failed to compile matmul kernel: {}. Falling back to dummy.", e);
                    Program::builder()
                        .devices(device)
                        .src("__kernel void kernel_mul_mat_f32_f32() {}")
                        .build(&context)?
                }
            };
        
        // Load simple ops kernel with fast math
        let simple_ops_src = include_str!("../../../kernels/simple_ops.cl");
        let simple_ops_program = Program::builder()
            .devices(device)
            .src(simple_ops_src)
            .cmplr_opt(CL_FAST_MATH_OPTS)
            .build(&context)?;
        
        // Load Q4_0 matmul kernel with fast math
        let q4_0_src = include_str!("../../../kernels/mul_mv_q4_0_f32.cl");
        let q4_0_program = match Program::builder()
            .devices(device)
            .src(q4_0_src)
            .cmplr_opt(CL_FAST_MATH_OPTS)
            .build(&context) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("WARN: Failed to compile Q4_0 kernel: {}. Using dummy.", e);
                    Program::builder()
                        .devices(device)
                        .src("__kernel void kernel_mul_mat_q4_0_f32() {}")
                        .build(&context)?
                }
            };

        // Load get_rows kernel with fast math
        let get_rows_src = include_str!("../../../kernels/get_rows.cl");
        let get_rows_program = Program::builder()
            .devices(device)
            .src(get_rows_src)
            .cmplr_opt(CL_FAST_MATH_OPTS)
            .build(&context)?;

        // Load quantize Q4_0 kernel
        let q4_quant_src = include_str!("../../../kernels/quantize_q4_0.cl");
        let quant_q4_0_program = Program::builder()
            .devices(device)
            .src(q4_quant_src)
            .cmplr_opt(CL_FAST_MATH_OPTS)
            .build(&context)?;

        // Create and cache all kernel objects once
        let kernel_cache = KernelCache {
            kernel_mul_mat_f32_f32: ocl::core::create_kernel(&program, "kernel_mul_mat_f32_f32")?,
            kernel_mul_mat_q4_0_f32: ocl::core::create_kernel(&q4_0_program, "kernel_mul_mat_q4_0_f32")?,
            kernel_rms_norm_opt: ocl::core::create_kernel(&simple_ops_program, "kernel_rms_norm_opt")?,
            kernel_softmax_opt: ocl::core::create_kernel(&simple_ops_program, "kernel_softmax_opt")?,
            kernel_rope_simple: ocl::core::create_kernel(&simple_ops_program, "kernel_rope_simple")?,
            kernel_silu_mul_simple: ocl::core::create_kernel(&simple_ops_program, "kernel_silu_mul_simple")?,
            kernel_add_assign_simple: ocl::core::create_kernel(&simple_ops_program, "kernel_add_assign_simple")?,
            kernel_scale_simple: ocl::core::create_kernel(&simple_ops_program, "kernel_scale_simple")?,
            kernel_get_rows_q4_0: ocl::core::create_kernel(&get_rows_program, "kernel_get_rows_q4_0")?,
            kernel_get_rows_f32: ocl::core::create_kernel(&get_rows_program, "kernel_get_rows_f32")?,
            kernel_attn_gen: ocl::core::create_kernel(&simple_ops_program, "kernel_attn_gen")?,
            kernel_cast_f32_to_f16: ocl::core::create_kernel(&simple_ops_program, "kernel_cast_f32_to_f16")?,
            kernel_attn_gen_half: ocl::core::create_kernel(&simple_ops_program, "kernel_attn_gen_half")?,
            kernel_quantize_f32_to_q4_0: ocl::core::create_kernel(&quant_q4_0_program, "kernel_quantize_f32_to_q4_0")?,
        };

        log::info!("OpenCL kernels cached successfully");

        Ok(Self {
            context,
            queue,
            device,
            program,
            simple_ops_program,
            q4_0_program,
            quant_q4_0_program,
            get_rows_program,
            kernels: Mutex::new(kernel_cache),
        })
    }

    pub fn matmul_q4_0(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        
        let (m, k) = if a_dims.len() == 3 {
            (a_dims[0] * a_dims[1], a_dims[2])
        } else {
            (a_dims[0], a_dims[1])
        };
        let n = b_dims[0];

        let a_buf = get_cl_mem(a.buffer().as_ref())
            .map_err(|_| anyhow!("A is not OpenCL buffer"))?;
        let b_buf = get_cl_mem(b.buffer().as_ref())
            .map_err(|_| anyhow!("B is not OpenCL buffer"))?;
        let out_buf = get_cl_mem(out.buffer().as_ref())
            .map_err(|_| anyhow!("Out is not OpenCL buffer"))?;

        let kernels = self.kernels.lock().map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel = &kernels.kernel_mul_mat_q4_0_f32;
        
        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1;
        let ne10 = k as i32;
        let ne12 = (k * m) as i32;
        let ne0 = n as i32;
        let ne1 = (n * m) as i32;
        let r2 = 1;
        let r3 = 1;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(b_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(a_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;
            
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&ne02))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&ne10))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&ne12))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&ne0))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&ne1))?;
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&r3))?;

            let local_work_size: [usize; 3] = [64, 1, 1];
            let group_size_0 = (n + 3) / 4;
            let global_work_size: [usize; 3] = [group_size_0 * local_work_size[0], m, 1];

            ocl::core::enqueue_kernel(
                &self.queue, kernel, 3, None, &global_work_size, Some(local_work_size),
                None::<&ocl::core::Event>, None::<&mut ocl::core::Event>
            )?;
        }
        Ok(())
    }
}

impl Backend for OpenCLBackend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn read_buffer(&self, t: &Tensor, dst: &mut [u8]) -> Result<()> {
        let buf = get_cl_mem(t.buffer().as_ref())
             .map_err(|_| anyhow::anyhow!("Not OpenCL buffer"))?;
        unsafe {
            ocl::core::enqueue_read_buffer(&self.queue, buf, true, 0, dst,
                None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }
    
    fn name(&self) -> &str { "OpenCL" }
    fn device(&self) -> &str { "GPU" }

    fn copy_from(&self, src: &Tensor) -> Result<Tensor> {
        let size = src.size();
        // Use device-only memory for copies (faster)
        let memory = crate::backend::opencl::memory::OpenCLMemory::new(self.context.clone(), self.queue.clone(), false);
        let buffer = memory.alloc(size, src.dtype())?;
        
        // Create new kernel cache by cloning kernel objects
        let src_kernels = self.kernels.lock().map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel_cache = KernelCache {
            kernel_mul_mat_f32_f32: src_kernels.kernel_mul_mat_f32_f32.clone(),
            kernel_mul_mat_q4_0_f32: src_kernels.kernel_mul_mat_q4_0_f32.clone(),
            kernel_rms_norm_opt: src_kernels.kernel_rms_norm_opt.clone(),
            kernel_softmax_opt: src_kernels.kernel_softmax_opt.clone(),
            kernel_rope_simple: src_kernels.kernel_rope_simple.clone(),
            kernel_silu_mul_simple: src_kernels.kernel_silu_mul_simple.clone(),
            kernel_add_assign_simple: src_kernels.kernel_add_assign_simple.clone(),
            kernel_scale_simple: src_kernels.kernel_scale_simple.clone(),
            kernel_get_rows_q4_0: src_kernels.kernel_get_rows_q4_0.clone(),
            kernel_get_rows_f32: src_kernels.kernel_get_rows_f32.clone(),
            kernel_attn_gen: src_kernels.kernel_attn_gen.clone(),
            kernel_cast_f32_to_f16: src_kernels.kernel_cast_f32_to_f16.clone(),
            kernel_attn_gen_half: src_kernels.kernel_attn_gen_half.clone(),
            kernel_quantize_f32_to_q4_0: src_kernels.kernel_quantize_f32_to_q4_0.clone(),
        };
        drop(src_kernels); // Release lock before potentially blocking operations
        
        let new_tensor = Tensor::new(src.shape().clone(), buffer.clone(), Arc::new(Self {
            context: self.context.clone(),
            queue: self.queue.clone(),
            device: self.device.clone(),
            program: self.program.clone(),
            simple_ops_program: self.simple_ops_program.clone(),
            q4_0_program: self.q4_0_program.clone(),
            quant_q4_0_program: self.quant_q4_0_program.clone(),
            get_rows_program: self.get_rows_program.clone(),
            kernels: Mutex::new(kernel_cache),
        }));

        // Device-to-Device Copy - try using get_cl_mem for both buffer types
        if let (Ok(src_mem), Ok(dst_mem)) = (get_cl_mem(src.buffer().as_ref()), get_cl_mem(buffer.as_ref())) {
             unsafe {
                 ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                    &self.queue, src_mem, dst_mem,
                    0, 0, size,
                    None::<&ocl::core::Event>, None::<&mut ocl::core::Event>
                 )?;
             }
             return Ok(new_tensor);
        }

        // Host-to-Device Copy - source is CPU, destination is GPU
        let src_ptr = src.as_ptr();
        if !src_ptr.is_null() {
            let src_slice = unsafe { std::slice::from_raw_parts(src_ptr, size) };
            if let Ok(dst_mem) = get_cl_mem(buffer.as_ref()) {
                 unsafe {
                     ocl::core::enqueue_write_buffer(
                        &self.queue, dst_mem, true, 0, src_slice,
                        None::<&ocl::core::Event>, None::<&mut ocl::core::Event>
                     )?;
                 }
            } else {
                 return Err(anyhow!("Failed to get cl_mem handle for destination buffer"));
            }
        }
        
        Ok(new_tensor)
    }

    fn synchronize(&self) -> Result<()> {
        ocl::core::finish(&self.queue)?;
        Ok(())
    }
    
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        
        let (m, k) = if a_dims.len() == 3 { (a_dims[0] * a_dims[1], a_dims[2]) } else { (a_dims[0], a_dims[1]) };
        let n = b_dims[0];
        
        if b_dims[1] != k { return Err(anyhow!("Dimension mismatch: A[{},{}] * B^T[{},{}]", m, k, n, b_dims[1])); }

        let a_buf = get_cl_mem(a.buffer().as_ref()).map_err(|_| anyhow!("A is not OpenCL buffer"))?;
        let b_buf = get_cl_mem(b.buffer().as_ref()).map_err(|_| anyhow!("B is not OpenCL buffer"))?;
        let c_buf = get_cl_mem(out.buffer().as_ref()).map_err(|_| anyhow!("Out is not OpenCL buffer"))?;
            
        let ne00 = k as i32; let ne01 = n as i32; let ne02 = 1i32;
        let nb00 = 4u64; let nb01 = k as u64 * 4; let nb02 = n as u64 * k as u64 * 4; let nb03 = nb02;
        let ne10 = k as i32; let ne11 = m as i32; let ne12 = 1i32;
        let nb10 = 4u64; let nb11 = k as u64 * 4; let nb12 = m as u64 * k as u64 * 4; let nb13 = nb12;
        let ne0 = n as i32; let ne1 = m as i32; let r2 = 1i32; let r3 = 1i32;

        let kernels = self.kernels.lock().map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel = &kernels.kernel_mul_mat_f32_f32;
        
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(b_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(a_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(c_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;
            
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&ne02))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&nb00))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&nb01))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&nb02))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&nb03))?;
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&ne10))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&ne11))?;
            ocl::core::set_kernel_arg(kernel, 15, ocl::core::ArgVal::scalar(&ne12))?;
            ocl::core::set_kernel_arg(kernel, 16, ocl::core::ArgVal::scalar(&nb10))?;
            ocl::core::set_kernel_arg(kernel, 17, ocl::core::ArgVal::scalar(&nb11))?;
            ocl::core::set_kernel_arg(kernel, 18, ocl::core::ArgVal::scalar(&nb12))?;
            ocl::core::set_kernel_arg(kernel, 19, ocl::core::ArgVal::scalar(&nb13))?;
            ocl::core::set_kernel_arg(kernel, 20, ocl::core::ArgVal::scalar(&ne0))?;
            ocl::core::set_kernel_arg(kernel, 21, ocl::core::ArgVal::scalar(&ne1))?;
            ocl::core::set_kernel_arg(kernel, 22, ocl::core::ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(kernel, 23, ocl::core::ArgVal::scalar(&r3))?;
            
            let local_size = 64usize;
            let global_work_size: [usize; 3] = [n * local_size, (m + 3) / 4, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];
            
            ocl::core::enqueue_kernel(&self.queue, kernel, 3, None, &global_work_size, Some(local_work_size),
                None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }

    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
         if b.dtype() == DType::Q4_0 { return self.matmul_q4_0(a, b, out); }
         self.matmul(a, b, out)
    }

    fn rms_norm(&self, x: &mut Tensor, weight: &Tensor, epsilon: f32) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len()-1].iter().product();
        
        let x_buf = get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let w_buf = get_cl_mem(weight.buffer().as_ref()).map_err(|_| anyhow!("Weight is not OpenCL buffer"))?;
        
        let kernels = self.kernels.lock().map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel = &kernels.kernel_rms_norm_opt;
        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();
        
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(w_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&epsilon))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::local::<f32>(&local_mem_size))?;
            
            let global_work_size: [usize; 3] = [rows * local_size, 1, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];
            
            ocl::core::enqueue_kernel(&self.queue, kernel, 1, None, &global_work_size, Some(local_work_size), 
                None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }

    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        let dims = x.shape().dims();
        let (seq_len, num_heads, head_dim) = if dims.len() == 4 { (dims[1], dims[2], dims[3]) } 
            else if dims.len() == 3 { (dims[0], dims[1], dims[2]) }
            else { return Err(anyhow!("RoPE expects 3 or 4 dims")); };
        
        let x_buf = get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        
        let kernels = self.kernels.lock().map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel = &kernels.kernel_rope_simple;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(num_heads as i32)))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&(seq_len as i32)))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&(start_pos as i32)))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&theta))?;
            
            let work_size = seq_len * num_heads * (head_dim / 2);
            ocl::core::enqueue_kernel(&self.queue, kernel, 1, None, &[work_size, 1, 1], None::<[usize; 3]>,
                None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }

    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        match (src.dtype(), dst.dtype()) {
            (DType::F32, DType::F16) => {
                let src_buf = get_cl_mem(src.buffer().as_ref())?;
                let dst_buf = get_cl_mem(dst.buffer().as_ref())?;
                let num_elements: usize = src.shape().dims().iter().product();
                let kernels = self.kernels.lock().map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
                let kernel = &kernels.kernel_cast_f32_to_f16;
                unsafe {
                    ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(src_buf))?;
                    ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(dst_buf))?;
                    ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(num_elements as i32)))?;
                    let gws: [usize; 3] = [((num_elements + 63) / 64) * 64, 1, 1];
                    let lws: [usize; 3] = [64, 1, 1];
                    ocl::core::enqueue_kernel(&self.queue, kernel, 1, None, &gws, Some(lws),
                        None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
                }
                Ok(())
            },
            (DType::F32, DType::Q4_0) => {
                let src_buf = get_cl_mem(src.buffer().as_ref())?;
                let dst_buf = get_cl_mem(dst.buffer().as_ref())?;
                let num_elements: usize = src.shape().dims().iter().product();
                
                if num_elements % 32 != 0 {
                    return Err(anyhow!("Q4_0 cast requires size multiple of 32"));
                }
                
                let kernels = self.kernels.lock().map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
                let kernel = &kernels.kernel_quantize_f32_to_q4_0;
                let num_blocks = num_elements / 32;
                
                unsafe {
                    ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(src_buf))?;
                    ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(dst_buf))?;
                    ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(num_elements as i32)))?;
                    
                    let local_size = 64;
                    let global_size = ((num_blocks + local_size - 1) / local_size) * local_size;
                    
                    ocl::core::enqueue_kernel(&self.queue, kernel, 1, None, &[global_size, 1, 1], Some([local_size, 1, 1]),
                        None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
                }
                Ok(())
            },
            _ => Err(anyhow!("OpenCL cast: unsupported {:?} -> {:?}", src.dtype(), dst.dtype())),
        }
    }

    fn attention_gen(&self, q: &Tensor, k_cache: &Tensor, v_cache: &Tensor, out: &mut Tensor,
                     num_heads_q: usize, num_heads_kv: usize, head_dim: usize, cache_seq_len: usize) -> Result<()> {
        let q_buf = get_cl_mem(q.buffer().as_ref())
            .map_err(|_| anyhow!("Q is not OpenCL buffer"))?;
        let k_buf = get_cl_mem(k_cache.buffer().as_ref())
            .map_err(|_| anyhow!("K is not OpenCL buffer"))?;
        let v_buf = get_cl_mem(v_cache.buffer().as_ref())
            .map_err(|_| anyhow!("V is not OpenCL buffer"))?;
        let o_buf = get_cl_mem(out.buffer().as_ref())
            .map_err(|_| anyhow!("Out is not OpenCL buffer"))?;
        
        let scale = 1.0 / (head_dim as f32).sqrt();
        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();
        
        let kernels = self.kernels.lock().map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        
        // Select kernel based on KV cache dtype
        let kernel = match k_cache.dtype() {
            DType::F16 => &kernels.kernel_attn_gen_half,
            _ => &kernels.kernel_attn_gen,
        };
        
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(q_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(k_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(v_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::mem(o_buf))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&(num_heads_q as i32)))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&(num_heads_kv as i32)))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&(cache_seq_len as i32)))?;
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&scale))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::local::<f32>(&local_mem_size))?;
            
            let global_work_size: [usize; 3] = [num_heads_q * local_size, 1, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];
            
            ocl::core::enqueue_kernel(&self.queue, kernel, 1, None, &global_work_size, Some(local_work_size),
                None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }

    fn silu_mul(&self, x: &mut Tensor, y: &Tensor) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();
        
        let x_buf = get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let y_buf = get_cl_mem(y.buffer().as_ref()).map_err(|_| anyhow!("Y is not OpenCL buffer"))?;
        
        let kernels = self.kernels.lock().map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel = &kernels.kernel_silu_mul_simple;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(y_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(size as i32)))?;
            
            ocl::core::enqueue_kernel(&self.queue, kernel, 1, None, &[size, 1, 1], None::<[usize; 3]>,
                None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }

    fn add_assign(&self, x: &mut Tensor, y: &Tensor) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();
        
        let x_buf = get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let y_buf = get_cl_mem(y.buffer().as_ref()).map_err(|_| anyhow!("Y is not OpenCL buffer"))?;
        
        let kernels = self.kernels.lock().map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel = &kernels.kernel_add_assign_simple;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(y_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(size as i32)))?;
            
            ocl::core::enqueue_kernel(&self.queue, kernel, 1, None, &[size, 1, 1], None::<[usize; 3]>,
                None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }

    fn matmul_slice(&self, a: &Tensor, b: &Tensor, _rows: usize, _cols: usize, out: &mut Tensor) -> Result<()> {
        self.matmul_transposed(a, b, out)
    }

    fn scale(&self, x: &mut Tensor, val: f32) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();
        
        let x_buf = get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        
        let kernels = self.kernels.lock().map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel = &kernels.kernel_scale_simple;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&val))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(size as i32)))?;
            
            ocl::core::enqueue_kernel(&self.queue, kernel, 1, None, &[size, 1, 1], None::<[usize; 3]>,
                None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }

    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len()-1].iter().product();
        
        let x_buf = get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        
        let kernels = self.kernels.lock().map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel = &kernels.kernel_softmax_opt;
        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();
        
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::local::<f32>(&local_mem_size))?;
            
            let global_work_size: [usize; 3] = [rows * local_size, 1, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];
            
            ocl::core::enqueue_kernel(&self.queue, kernel, 1, None, &global_work_size, Some(local_work_size),
                None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }

    fn gather(&self, src: &Tensor, indices: &Tensor, dst: &mut Tensor) -> Result<()> {
         let dims = src.shape().dims();
         let k = dims[dims.len() - 1];
         
         let src_buf = get_cl_mem(src.buffer().as_ref()).map_err(|_| anyhow!("Src is not OpenCL buffer"))?;
         let idx_buf = get_cl_mem(indices.buffer().as_ref()).map_err(|_| anyhow!("Indices is not OpenCL buffer"))?;
         let dst_buf = get_cl_mem(dst.buffer().as_ref()).map_err(|_| anyhow!("Dst is not OpenCL buffer"))?;
            
         let ne00 = k as i32;
         let nb01 = match src.dtype() {
             DType::F32 => (k * 4) as u64,
             DType::Q4_0 => (k / 32 * 18) as u64,
             _ => return Err(anyhow!("Unsupported src dtype for gather: {:?}", src.dtype())),
         };
         let nb02 = nb01 * dims[0] as u64; 
         let nb03 = nb02;
         let ne10 = indices.size() as i32; 
         let nb10 = 4u64;
         let nb11 = nb10 * ne10 as u64;
         let nb12 = nb11;
         let nb1 = (k * 4) as u64;
         let nb2 = nb1 * ne10 as u64; 
         let nb3 = nb2;

         let kernels = self.kernels.lock().map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
         let kernel = match src.dtype() {
             DType::Q4_0 => &kernels.kernel_get_rows_q4_0,
             DType::F32 => &kernels.kernel_get_rows_f32,
             _ => return Err(anyhow!("Unsupported dtype")),
         };
         
         unsafe {
             ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(src_buf))?;
             ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
             ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(idx_buf))?;
             ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
             ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(dst_buf))?;
             ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;
             ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
             ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&nb01))?;
             ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&nb02))?;
             ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&nb03))?;
             ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&ne10))?;
             ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&nb10))?;
             ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&nb11))?;
             ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&nb12))?;
             ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&nb1))?;
             ocl::core::set_kernel_arg(kernel, 15, ocl::core::ArgVal::scalar(&nb2))?;
             ocl::core::set_kernel_arg(kernel, 16, ocl::core::ArgVal::scalar(&nb3))?;

             let local_size = 64usize;
             let num_indices = indices.size();
             let global_work_size: [usize; 3] = [num_indices * local_size, 1, 1];
             let local_work_size: [usize; 3] = [local_size, 1, 1];
             
             ocl::core::enqueue_kernel(&self.queue, kernel, 1, None, &global_work_size, Some(local_work_size), 
                 None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
         }
         Ok(())
    }

    fn copy_slice(&self, src: &Tensor, dst: &mut Tensor, src_offset: usize, dst_offset: usize, count: usize) -> Result<()> {
         let type_size = match src.dtype() {
            DType::F32 => 4, DType::F16 => 2, DType::U8 => 1, DType::Q4_0 => 18,
             _ => return Err(anyhow!("Unsupported dtype for copy_slice: {:?}", src.dtype())),
         };
         
         let src_m = get_cl_mem(src.buffer().as_ref());
         let dst_m = get_cl_mem(dst.buffer().as_ref());
         
         if let (Ok(sb), Ok(db)) = (src_m.as_ref(), dst_m.as_ref()) {
             let src_byte_off = src_offset * type_size;
             let dst_byte_off = dst_offset * type_size;
             let byte_len = count * type_size;
             
             unsafe {
                 ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                     &self.queue, *sb, *db,
                     src_byte_off, dst_byte_off, byte_len,
                     None::<&ocl::core::Event>, None::<&mut ocl::core::Event>
                 )?;
             }
             return Ok(());
         }
         
         if let Ok(db) = dst_m {
             let src_ptr = src.as_ptr();
             if !src_ptr.is_null() {
                 let src_byte_off = src_offset * type_size;
                 let dst_byte_off = dst_offset * type_size;
                 let byte_len = count * type_size;
                 unsafe {
                     let src_u8 = src_ptr as *const u8;
                     ocl::core::enqueue_write_buffer(&self.queue, db, true, dst_byte_off,
                         std::slice::from_raw_parts(src_u8.add(src_byte_off), byte_len),
                         None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
                 }
                 return Ok(());
             }
         }
         
         if let Ok(sb) = src_m {
             let dst_ptr = dst.as_mut_ptr();
             if !dst_ptr.is_null() {
                 let src_byte_off = src_offset * type_size;
                 let dst_byte_off = dst_offset * type_size;
                 let byte_len = count * type_size;
                 unsafe {
                     let dst_u8 = dst_ptr as *mut u8;
                     ocl::core::enqueue_read_buffer(&self.queue, sb, true, src_byte_off,
                         std::slice::from_raw_parts_mut(dst_u8.add(dst_byte_off), byte_len),
                         None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
                 }
                 return Ok(());
             }
         }

         Err(anyhow!("Unsupported copy_slice combination in OpenCL backend or null pointers"))
    }
}
