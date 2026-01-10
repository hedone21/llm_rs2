use anyhow::{Result, anyhow};
use std::sync::Arc;
use crate::core::backend::Backend;
use crate::core::tensor::Tensor;
use crate::core::memory::Memory;
use ocl::{Context, Device, Platform, Queue, Program, Kernel, flags};
use crate::core::buffer::DType;
use crate::core::buffer::Buffer;

pub mod buffer;
pub mod memory;

#[derive(Debug)]
pub struct OpenCLBackend {
    pub context: Context,
    pub queue: Queue,
    pub device: Device,
    pub program: Program,
    pub simple_ops_program: Program,
    pub q4_0_program: Program,
    pub get_rows_program: Program,
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

        // Load matmul kernel
        let matmul_src = include_str!("../../../kernels/mul_mv_f32_f32.cl");
        let program = match Program::builder()
            .devices(device)
            .src(matmul_src)
            .cmplr_opt("-cl-std=CL2.0")
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
        
        // Load simple ops kernel
        let simple_ops_src = include_str!("../../../kernels/simple_ops.cl");
        let simple_ops_program = Program::builder()
            .devices(device)
            .src(simple_ops_src)
            .build(&context)?;
        
        // Load Q4_0 matmul kernel
        let q4_0_src = include_str!("../../../kernels/mul_mv_q4_0_f32.cl");
        let q4_0_program = match Program::builder()
            .devices(device)
            .src(q4_0_src)
            .cmplr_opt("-cl-std=CL2.0")
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


        // Load get_rows kernel
        let get_rows_src = include_str!("../../../kernels/get_rows.cl");
        let get_rows_program = Program::builder()
            .devices(device)
            .src(get_rows_src)
            .cmplr_opt("-cl-std=CL2.0")
            .build(&context)?;

        Ok(Self {
            context,
            queue,
            device,
            program,
            simple_ops_program,
            q4_0_program,
            get_rows_program,
        })
    }

    pub fn matmul_q4_0(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        let m = a_dims[0];
        let k = a_dims[1];
        let n = b_dims[0]; // b is [N, K]
        
        // Kernel signature:
        // kernel void kernel_mul_mat_q4_0_f32( ... )

        let a_buf = a.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("A is not OpenCL buffer"))?;
        let b_buf = b.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("B is not OpenCL buffer"))?;
        let out_buf = out.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("Out is not OpenCL buffer"))?;

        let kernel = ocl::core::create_kernel(&self.q4_0_program, "kernel_mul_mat_q4_0_f32")?;
        
        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1;
        
        let ne10 = k as i32; // Stride of A in elements (float)
        let ne12 = (k * m) as i32;
        
        let ne0 = n as i32; // Stride of Out in elements? dst[r1*ne0 + ...]
        let ne1 = (n * m) as i32;
        
        let r2 = 1;
        let r3 = 1;

        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(b_buf.buffer.as_core()))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(a_buf.buffer.as_core()))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(out_buf.buffer.as_core()))?;
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;
            
            ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&ne02))?;
            
            ocl::core::set_kernel_arg(&kernel, 9, ocl::core::ArgVal::scalar(&ne10))?;
            ocl::core::set_kernel_arg(&kernel, 10, ocl::core::ArgVal::scalar(&ne12))?;
            
            ocl::core::set_kernel_arg(&kernel, 11, ocl::core::ArgVal::scalar(&ne0))?;
            ocl::core::set_kernel_arg(&kernel, 12, ocl::core::ArgVal::scalar(&ne1))?;
            ocl::core::set_kernel_arg(&kernel, 13, ocl::core::ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(&kernel, 14, ocl::core::ArgVal::scalar(&r3))?;

            // Work size
            // dim 0: (N + 3) / 4 * 64
            // dim 1: M
            // dim 2: 1
            let local_work_size: [usize; 3] = [64, 1, 1];
            let group_size_0 = (n + 3) / 4;
            let global_work_size: [usize; 3] = [
                group_size_0 * local_work_size[0],
                m,
                1
            ];

            ocl::core::enqueue_kernel(
                &self.queue,
                &kernel,
                3,
                None,
                &global_work_size,
                Some(local_work_size),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>
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
        let buf = t.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
             .ok_or(anyhow::anyhow!("Not OpenCL buffer"))?;
        buf.buffer.read(dst).queue(&self.queue).enq()?;
        Ok(())
    }
    fn name(&self) -> &str {
        "OpenCL"
    }

    fn device(&self) -> &str {
        "GPU"
    }

    fn copy_from(&self, src: &Tensor) -> Result<Tensor> {
        let size = src.size();
        let memory = crate::backend::opencl::memory::OpenCLMemory::new(self.context.clone(), self.queue.clone());
        let buffer = memory.alloc(size, src.dtype())?;
        
        let new_tensor = Tensor::new(src.shape().clone(), buffer.clone(), Arc::new(Self {
            context: self.context.clone(),
            queue: self.queue.clone(),
            device: self.device.clone(),
            program: self.program.clone(),
            simple_ops_program: self.simple_ops_program.clone(),
            q4_0_program: self.q4_0_program.clone(),
            get_rows_program: self.get_rows_program.clone(),
        }));

        // Case 1: Source is OpenCL Tensor (Device-to-Device Copy)
        if let Some(src_buf_ocl) = src.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>() {
            if let Some(dst_buf_ocl) = buffer.as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>() {
                 unsafe {
                     ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                        &self.queue,
                        src_buf_ocl.buffer.as_core(),
                        dst_buf_ocl.buffer.as_core(),
                        0,
                        0,
                        src_buf_ocl.size(), // bytes
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>
                     )?;
                 }
                 // Synchronize to ensure copy is done (debugging stability)
                 self.queue.finish()?; 
                 return Ok(new_tensor);
            }
        }

        // Case 2: Source is CPU Tensor (Host-to-Device Copy)
        let src_ptr = src.as_ptr();
        if !src_ptr.is_null() {
            let src_slice = unsafe { std::slice::from_raw_parts(src_ptr, size) };
            if let Some(ocl_buf) = buffer.as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>() {
                 unsafe {
                     ocl::core::enqueue_write_buffer(
                        &self.queue,
                        ocl_buf.buffer.as_core(),
                        true, // blocking write
                        0,
                        src_slice,
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>
                     )?;
                 }
            } else {
                 return Err(anyhow!("Failed to downcast buffer in copy_from"));
            }
        } else {
             // If source pointer is null and it wasn't an OpenCL buffer, we might have an issue.
             // But for now, we assume if as_ptr is null, it might be some other backend we don't support or unmapped.
             // Warn?
             // log::warn!("copy_from called with null pointer and not OpenCL buffer");
        }
        
        Ok(new_tensor)
    }

    fn synchronize(&self) -> Result<()> {
        ocl::core::finish(&self.queue)?;
        Ok(())
    }
    
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        // For matmul_transposed: A[M,K] * B^T[N,K] = C[M,N]
        // B is stored as [N, K] (already transposed)
        // llama.cpp kernel: src0=B[N,K], src1=A[M,K], dst=C[M,N]
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        
        let m = a_dims[0];  // A rows
        let k = a_dims[1];  // A cols = B cols (inner dimension)
        let n = b_dims[0];  // B rows (output cols)
        
        // Verify dimensions
        if b_dims[1] != k {
            return Err(anyhow!("Dimension mismatch: A[{},{}] * B^T[{},{}]", m, k, n, b_dims[1]));
        }

        let a_buf = a.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("A is not OpenCL buffer"))?;
        let b_buf = b.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("B is not OpenCL buffer"))?;
        let c_buf = out.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("Out is not OpenCL buffer"))?;
            
        // Kernel args for llama.cpp mul_mv_f32_f32:
        // src0 = B[N, K], src1 = A[M, K], dst = C[M, N]
        // ne00 = K (inner dim), ne01 = N (src0 rows), ne02 = 1 (batch)
        // ne10 = K, ne11 = M (src1 rows), ne12 = 1
        // ne0 = N (output cols), ne1 = M (output rows)
        
        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1i32;
        let nb00 = 4u64;                    // sizeof(float)
        let nb01 = k as u64 * 4;            // stride between B rows
        let nb02 = n as u64 * k as u64 * 4; // batch stride
        let nb03 = nb02;
        
        let ne10 = k as i32;
        let ne11 = m as i32;
        let ne12 = 1i32;
        let nb10 = 4u64;
        let nb11 = k as u64 * 4;            // stride between A rows
        let nb12 = m as u64 * k as u64 * 4;
        let nb13 = nb12;
        
        let ne0 = n as i32;  // output cols
        let ne1 = m as i32;  // output rows
        let r2 = 1i32;
        let r3 = 1i32;

        let kernel = ocl::core::create_kernel(&self.program, "kernel_mul_mat_f32_f32")?;
        
        unsafe {
            let b_mem = b_buf.buffer.as_core();
            let a_mem = a_buf.buffer.as_core();
            let c_mem = c_buf.buffer.as_core();
            
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(b_mem))?;  // src0 = B
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(a_mem))?;  // src1 = A
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(c_mem))?;  // dst = C
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;
            
            ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&ne02))?;
            ocl::core::set_kernel_arg(&kernel, 9, ocl::core::ArgVal::scalar(&nb00))?;
            ocl::core::set_kernel_arg(&kernel, 10, ocl::core::ArgVal::scalar(&nb01))?;
            ocl::core::set_kernel_arg(&kernel, 11, ocl::core::ArgVal::scalar(&nb02))?;
            ocl::core::set_kernel_arg(&kernel, 12, ocl::core::ArgVal::scalar(&nb03))?;
            
            ocl::core::set_kernel_arg(&kernel, 13, ocl::core::ArgVal::scalar(&ne10))?;
            ocl::core::set_kernel_arg(&kernel, 14, ocl::core::ArgVal::scalar(&ne11))?;
            ocl::core::set_kernel_arg(&kernel, 15, ocl::core::ArgVal::scalar(&ne12))?;
            ocl::core::set_kernel_arg(&kernel, 16, ocl::core::ArgVal::scalar(&nb10))?;
            ocl::core::set_kernel_arg(&kernel, 17, ocl::core::ArgVal::scalar(&nb11))?;
            ocl::core::set_kernel_arg(&kernel, 18, ocl::core::ArgVal::scalar(&nb12))?;
            ocl::core::set_kernel_arg(&kernel, 19, ocl::core::ArgVal::scalar(&nb13))?;
            
            ocl::core::set_kernel_arg(&kernel, 20, ocl::core::ArgVal::scalar(&ne0))?;
            ocl::core::set_kernel_arg(&kernel, 21, ocl::core::ArgVal::scalar(&ne1))?;
            ocl::core::set_kernel_arg(&kernel, 22, ocl::core::ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(&kernel, 23, ocl::core::ArgVal::scalar(&r3))?;
            
            // Global work size:
            // - dim 0: n work groups (one per B row), each with local_size threads
            // - dim 1: ceil(m/4) work groups (4 A rows per work group)
            // - dim 2: 1 (batch)
            // Each work group processes one B row against 4 A rows
            // llama.cpp uses local size = subgroup size (64 for Adreno)
            let local_size = 64usize;  // Adreno subgroup size
            let global_work_size: [usize; 3] = [
                n * local_size,  // one group per B row
                (m + 3) / 4,     // ceil(M/4) groups
                1
            ];
            let local_work_size: [usize; 3] = [local_size, 1, 1];
            
            ocl::core::enqueue_kernel(
                &self.queue,
                &kernel,
                3,
                None,
                &global_work_size,
                Some(local_work_size),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>
            )?;
        }

        Ok(())
    }

    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
         if b.dtype() == DType::Q4_0 {
             return self.matmul_q4_0(a, b, out);
         }
         // The mul_mv_f32_f32 kernel expects B to be transposed already [N, K],
         // which matches matmul_transposed semantics. Delegate to matmul.
         self.matmul(a, b, out)
    }



    fn rms_norm(&self, x: &mut Tensor, weight: &Tensor, epsilon: f32) -> Result<()> {
        let dims = x.shape().dims();
        let rows = if dims.len() > 1 { dims[0] } else { 1 };
        let dim = dims[dims.len() - 1];
        
        let x_buf = x.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("X is not OpenCL buffer"))?;
        let w_buf = weight.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("Weight is not OpenCL buffer"))?;
        
        let kernel = ocl::core::create_kernel(&self.simple_ops_program, "kernel_rms_norm_simple")?;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(x_buf.buffer.as_core()))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(w_buf.buffer.as_core()))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(x_buf.buffer.as_core()))?; // output = x (inplace)
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&epsilon))?;
            
            ocl::core::enqueue_kernel(&self.queue, &kernel, 1, None, &[rows, 1, 1], None::<[usize; 3]>, None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }

    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        let dims = x.shape().dims();
        let total_elements = dims.iter().product::<usize>();
        let head_dim = dims[dims.len() - 1];
        
        let x_buf = x.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("X is not OpenCL buffer"))?;
        
        let kernel = ocl::core::create_kernel(&self.simple_ops_program, "kernel_rope_simple")?;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(x_buf.buffer.as_core()))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&(start_pos as i32)))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&theta))?;
            
            let work_size = total_elements / 2;  // pairs
            ocl::core::enqueue_kernel(&self.queue, &kernel, 1, None, &[work_size, 1, 1], None::<[usize; 3]>, None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }

    fn silu_mul(&self, x: &mut Tensor, y: &Tensor) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();
        
        let x_buf = x.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("X is not OpenCL buffer"))?;
        let y_buf = y.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("Y is not OpenCL buffer"))?;
        
        let kernel = ocl::core::create_kernel(&self.simple_ops_program, "kernel_silu_mul_simple")?;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(x_buf.buffer.as_core()))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(y_buf.buffer.as_core()))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&(size as i32)))?;
            
            ocl::core::enqueue_kernel(&self.queue, &kernel, 1, None, &[size, 1, 1], None::<[usize; 3]>, None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }

    fn add_assign(&self, x: &mut Tensor, y: &Tensor) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();
        
        let x_buf = x.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("X is not OpenCL buffer"))?;
        let y_buf = y.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("Y is not OpenCL buffer"))?;
        
        let kernel = ocl::core::create_kernel(&self.simple_ops_program, "kernel_add_assign_simple")?;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(x_buf.buffer.as_core()))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(y_buf.buffer.as_core()))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&(size as i32)))?;
            
            ocl::core::enqueue_kernel(&self.queue, &kernel, 1, None, &[size, 1, 1], None::<[usize; 3]>, None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }

    fn matmul_slice(&self, a: &Tensor, b: &Tensor, _rows: usize, _cols: usize, out: &mut Tensor) -> Result<()> {
        // For now, delegate to matmul_transposed
        self.matmul_transposed(a, b, out)
    }

    fn scale(&self, x: &mut Tensor, val: f32) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();
        
        let x_buf = x.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("X is not OpenCL buffer"))?;
        
        let kernel = ocl::core::create_kernel(&self.simple_ops_program, "kernel_scale_simple")?;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(x_buf.buffer.as_core()))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::scalar(&val))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&(size as i32)))?;
            
            ocl::core::enqueue_kernel(&self.queue, &kernel, 1, None, &[size, 1, 1], None::<[usize; 3]>, None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }

    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        let dims = x.shape().dims();
        let rows = if dims.len() > 1 { dims[0] } else { 1 };
        let dim = dims[dims.len() - 1];
        
        let x_buf = x.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("X is not OpenCL buffer"))?;
        
        let kernel = ocl::core::create_kernel(&self.simple_ops_program, "kernel_softmax_simple")?;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(x_buf.buffer.as_core()))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(x_buf.buffer.as_core()))?; // output = x (inplace)
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            
            ocl::core::enqueue_kernel(&self.queue, &kernel, 1, None, &[rows, 1, 1], None::<[usize; 3]>, None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        Ok(())
    }

    fn gather(&self, src: &Tensor, indices: &Tensor, dst: &mut Tensor) -> Result<()> {
         // Check if input is Q4_0 or F32
         let dims = src.shape().dims();
         let k = dims[dims.len() - 1]; // Hidden size
         
         let src_buf = src.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("Src is not OpenCL buffer"))?;
         let idx_buf = indices.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("Indices is not OpenCL buffer"))?;
         let dst_buf = dst.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
            .ok_or(anyhow!("Dst is not OpenCL buffer"))?;
            
         // Kernel arguments
         // src0, off0, src1, off1, dst, offd, ne00, nb01, nb02, nb03, ne10, nb10, nb11, nb12, nb1, nb2, nb3
         
         let ne00 = k as i32;
         let nb01 = match src.dtype() {
             DType::F32 => (k * 4) as u64,
             DType::Q4_0 => (k / 32 * 18) as u64, // 18 bytes per block
             _ => return Err(anyhow!("Unsupported src dtype for gather: {:?}", src.dtype())),
         };
         let nb02 = nb01 * dims[0] as u64; 
         let nb03 = nb02;

         let ne10 = indices.size() as i32; 
         // src1 indices strides
         let nb10 = 4u64; // int32
         let nb11 = nb10 * ne10 as u64;
         let nb12 = nb11;
         
         // dst strides
         let nb1 = (k * 4) as u64; // F32
         let nb2 = nb1 * ne10 as u64; 
         let nb3 = nb2;

         let kernel_name = match src.dtype() {
             DType::Q4_0 => "kernel_get_rows_q4_0",
             DType::F32 => "kernel_get_rows_f32",
             _ => return Err(anyhow!("Unsupported dtype")),
         };

         let kernel = ocl::core::create_kernel(&self.get_rows_program, kernel_name)?;
         
         unsafe {
             ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(src_buf.buffer.as_core()))?;
             ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
             ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(idx_buf.buffer.as_core()))?;
             ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
             ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(dst_buf.buffer.as_core()))?;
             ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;
             
             ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
             ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&nb01))?;
             ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&nb02))?;
             ocl::core::set_kernel_arg(&kernel, 9, ocl::core::ArgVal::scalar(&nb03))?;
             
             ocl::core::set_kernel_arg(&kernel, 10, ocl::core::ArgVal::scalar(&ne10))?;
             ocl::core::set_kernel_arg(&kernel, 11, ocl::core::ArgVal::scalar(&nb10))?;
             ocl::core::set_kernel_arg(&kernel, 12, ocl::core::ArgVal::scalar(&nb11))?;
             ocl::core::set_kernel_arg(&kernel, 13, ocl::core::ArgVal::scalar(&nb12))?;
             
             ocl::core::set_kernel_arg(&kernel, 14, ocl::core::ArgVal::scalar(&nb1))?;
             ocl::core::set_kernel_arg(&kernel, 15, ocl::core::ArgVal::scalar(&nb2))?;
             ocl::core::set_kernel_arg(&kernel, 16, ocl::core::ArgVal::scalar(&nb3))?;

             // Work size
             let local_size = 64usize;
             let num_indices = indices.size();
             let global_work_size: [usize; 3] = [num_indices * local_size, 1, 1];
             let local_work_size: [usize; 3] = [local_size, 1, 1];
             
             ocl::core::enqueue_kernel(
                 &self.queue, &kernel, 1, None, &global_work_size, Some(local_work_size), 
                 None::<&ocl::core::Event>, None::<&mut ocl::core::Event>
             )?;
         }
         Ok(())
    }

    fn copy_slice(&self, src: &Tensor, dst: &mut Tensor, src_offset: usize, dst_offset: usize, count: usize) -> Result<()> {
         // Identify element size
         let type_size = match src.dtype() {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::U8 => 1,
            // For Q4_0, since it's blocks, offsets/count are in BLOCKS not float elements?
            // KVCache is usually F32 or F16.
            // If DType is Q4_0, sizeof(BlockQ4_0).
             DType::Q4_0 => 18, // 32/2 + 2 = 18 bytes
             _ => return Err(anyhow!("Unsupported dtype for copy_slice: {:?}", src.dtype())),
         };
         
         let src_buf = src.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>();
         let dst_buf = dst.buffer().as_any().downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>();
         
         if let (Some(sb), Some(db)) = (src_buf, dst_buf) {
             // Device to Device copy
             let src_byte_off = src_offset * type_size;
             let dst_byte_off = dst_offset * type_size;
             let byte_len = count * type_size;
             
             unsafe {
                 ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                     &self.queue,
                     sb.buffer.as_core(),
                     db.buffer.as_core(),
                     src_byte_off,
                     dst_byte_off,
                     byte_len,
                     None::<&ocl::core::Event>,
                     None::<&mut ocl::core::Event>
                 )?;
             }
             self.queue.finish()?; // Sync for safety
             return Ok(());
         }
         
         // Fallback to default CPU copy likely fails if buffers are NULL, but try default if not both OpenCL
         // Actually default calls as_ptr(), if that returns null calling default will error.
         // Let's implement Host <-> Device copy here if needed, or error out?
         // KVCache new_k (src) might be CPU if we didn't put it on device? 
         // In forward_gen, k_rope IS on device. So it should be dev->dev.
         
         // If one is CPU and other is GPU?
         // If src is CPU and dst is GPU: enqueue_write_buffer
         // If src is GPU and dst is CPU: enqueue_read_buffer
         
         if let Some(db) = dst_buf {
             // Host to Device
             let src_ptr = src.as_ptr();
             if !src_ptr.is_null() {
                 let src_byte_off = src_offset * type_size;
                 let dst_byte_off = dst_offset * type_size;
                 let byte_len = count * type_size;
                 
                 unsafe {
                     let src_u8 = src_ptr as *const u8;
                     ocl::core::enqueue_write_buffer(
                         &self.queue,
                         db.buffer.as_core(),
                         true,
                         dst_byte_off,
                         std::slice::from_raw_parts(src_u8.add(src_byte_off), byte_len),
                         None::<&ocl::core::Event>,
                         None::<&mut ocl::core::Event>
                     )?;
                 }
                 return Ok(());
             }
         }
         
         if let Some(sb) = src_buf {
             // Device to Host
             let dst_ptr = dst.as_mut_ptr();
             if !dst_ptr.is_null() {
                 let src_byte_off = src_offset * type_size;
                 let dst_byte_off = dst_offset * type_size;
                 let byte_len = count * type_size;

                 unsafe {
                     let dst_u8 = dst_ptr as *mut u8;
                     ocl::core::enqueue_read_buffer(
                         &self.queue,
                         sb.buffer.as_core(),
                         true,
                         src_byte_off,
                         std::slice::from_raw_parts_mut(dst_u8.add(dst_byte_off), byte_len),
                         None::<&ocl::core::Event>,
                         None::<&mut ocl::core::Event>
                     )?;
                 }
                 return Ok(());
             }
         }

         // Delegate to default (which will likely fail if pointers are null)
         // We can't easily call default implementation from here without a helper or trait setup, 
         // but we can copy the logic.
         
         Err(anyhow!("Unsupported copy_slice combination in OpenCL backend or null pointers"))
    }
}

