#![allow(unused_unsafe)]
use crate::buffer::unified_buffer::UnifiedBuffer;
use crate::core::backend::Backend;
use crate::core::buffer::Buffer;
use crate::core::buffer::DType;
use crate::core::memory::Memory;
use crate::core::tensor::Tensor;
use anyhow::{Result, anyhow};
use ocl::core::Kernel as CoreKernel;
use ocl::{Context, Device, Platform, Program, Queue, flags};
use std::sync::{Arc, Mutex};

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
    if let Some(ocl_buf) = buf
        .as_any()
        .downcast_ref::<crate::backend::opencl::buffer::OpenCLBuffer>()
    {
        return Ok(ocl_buf.buffer.as_core());
    }
    Err(anyhow!("Buffer is not an OpenCL buffer type"))
}

// Fast math flags (excluding -cl-std which is determined at runtime)
const CL_FAST_MATH_FLAGS: &str =
    "-cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math";

/// Build compiler options string based on device's OpenCL C version.
/// Checks CL_DEVICE_OPENCL_C_VERSION (e.g. "OpenCL C 1.2") instead of the API version,
/// since a device can report OpenCL 3.0 API while only supporting CL C 1.2.
fn build_cl_opts(device: &Device) -> String {
    let cl_c_version_str = device
        .info(ocl::core::DeviceInfo::OpenclCVersion)
        .map(|v| v.to_string())
        .unwrap_or_default();
    // Parse "OpenCL C X.Y ..." → extract major.minor
    let supports_cl_c_2 = cl_c_version_str
        .split_whitespace()
        .nth(2) // "X.Y"
        .and_then(|ver| {
            let mut parts = ver.split('.');
            let major: u32 = parts.next()?.parse().ok()?;
            let minor: u32 = parts.next()?.parse().ok()?;
            Some((major, minor) >= (2, 0))
        })
        .unwrap_or(false);
    if supports_cl_c_2 {
        format!("-cl-std=CL2.0 {}", CL_FAST_MATH_FLAGS)
    } else {
        CL_FAST_MATH_FLAGS.to_string()
    }
}

/// Cached kernel objects wrapped in a struct for interior mutability.
/// Uses Mutex to make the raw kernel pointers thread-safe.
struct KernelCache {
    kernel_mul_mat_f32_f32: CoreKernel,
    kernel_mul_mat_f16_f32: CoreKernel,
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
    pub f16_program: Program,
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
        // --- Platform selection via OCL_PLATFORM env var ---
        let platform = if let Ok(name) = std::env::var("OCL_PLATFORM") {
            let name_lower = name.to_lowercase();
            Platform::list()
                .into_iter()
                .find(|p| {
                    p.name()
                        .unwrap_or_default()
                        .to_lowercase()
                        .contains(&name_lower)
                })
                .ok_or_else(|| anyhow!("No OpenCL platform matching '{}'", name))?
        } else {
            Platform::default()
        };

        // --- Device type selection via OCL_DEVICE_TYPE env var ---
        let device_type = match std::env::var("OCL_DEVICE_TYPE").as_deref().unwrap_or("gpu") {
            "cpu" => Some(flags::DEVICE_TYPE_CPU),
            "all" => None,
            _ => Some(flags::DEVICE_TYPE_GPU),
        };

        let device = if let Some(dtype) = device_type {
            Device::list(platform, Some(dtype))?
                .into_iter()
                .next()
                .unwrap_or_else(|| Device::first(platform).expect("No OpenCL devices found"))
        } else {
            Device::first(platform)?
        };

        // --- Log platform/device info ---
        let platform_name = platform.name().unwrap_or_else(|_| "unknown".into());
        let device_name = device.name().unwrap_or_else(|_| "unknown".into());
        let device_version = device
            .version()
            .map(|v| v.to_string())
            .unwrap_or_else(|_| "unknown".into());
        let cl_c_version = device
            .info(ocl::core::DeviceInfo::OpenclCVersion)
            .map(|v| v.to_string())
            .unwrap_or_else(|_| "unknown".into());
        let extensions = device
            .info(ocl::core::DeviceInfo::Extensions)
            .map(|v| v.to_string())
            .unwrap_or_default();
        let has_khr_subgroups = extensions.contains("cl_khr_subgroups");
        let has_intel_subgroups = extensions.contains("cl_intel_subgroups");

        log::info!("OpenCL Platform: {}", platform_name);
        log::info!(
            "OpenCL Device: {} (API {}, CL C: {})",
            device_name,
            device_version,
            cl_c_version
        );
        log::info!(
            "Subgroup support: cl_khr_subgroups={}, cl_intel_subgroups={}",
            has_khr_subgroups,
            has_intel_subgroups
        );

        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;

        let queue = Queue::new(&context, device, None)?;

        // --- Build compiler options based on device CL version ---
        let cl_opts = build_cl_opts(&device);
        log::info!("OpenCL compiler options: {}", cl_opts);

        // === Load kernel programs with fallback ===

        // Matmul F32: try original, fallback to nosub, then dummy
        let matmul_src = include_str!("../../../kernels/mul_mv_f32_f32.cl");
        let matmul_fallback_src = include_str!("../../../kernels/fallback/mul_mv_f32_f32_nosub.cl");
        let program = match Program::builder()
            .devices(device)
            .src(matmul_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => p,
            Err(e) => {
                log::warn!("mul_mv_f32_f32.cl failed: {}. Trying fallback.", e);
                match Program::builder()
                    .devices(device)
                    .src(matmul_fallback_src)
                    .cmplr_opt(&cl_opts)
                    .build(&context)
                {
                    Ok(p) => {
                        log::info!("Using fallback/mul_mv_f32_f32_nosub.cl");
                        p
                    }
                    Err(e2) => {
                        log::warn!("Fallback matmul also failed: {}. Using dummy.", e2);
                        Program::builder()
                            .devices(device)
                            .src("__kernel void kernel_mul_mat_f32_f32() {}")
                            .build(&context)?
                    }
                }
            }
        };

        // Simple ops: try original, fallback to nosub file
        let simple_ops_src = include_str!("../../../kernels/simple_ops.cl");
        let simple_ops_fallback_src = include_str!("../../../kernels/fallback/simple_ops_nosub.cl");
        let simple_ops_program = match Program::builder()
            .devices(device)
            .src(simple_ops_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("simple_ops.cl compiled (subgroup path)");
                p
            }
            Err(e) => {
                log::warn!(
                    "simple_ops.cl failed: {}. Using fallback/simple_ops_nosub.cl",
                    e
                );
                Program::builder()
                    .devices(device)
                    .src(simple_ops_fallback_src)
                    .cmplr_opt(&cl_opts)
                    .build(&context)?
            }
        };

        // Q4_0 matmul: try original, fallback to nosub, then dummy
        let q4_0_src = include_str!("../../../kernels/mul_mv_q4_0_f32.cl");
        let q4_0_fallback_src = include_str!("../../../kernels/fallback/mul_mv_q4_0_f32_nosub.cl");
        let q4_0_program = match Program::builder()
            .devices(device)
            .src(q4_0_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => p,
            Err(e) => {
                log::warn!("mul_mv_q4_0_f32.cl failed: {}. Trying fallback.", e);
                match Program::builder()
                    .devices(device)
                    .src(q4_0_fallback_src)
                    .cmplr_opt(&cl_opts)
                    .build(&context)
                {
                    Ok(p) => {
                        log::info!("Using fallback/mul_mv_q4_0_f32_nosub.cl");
                        p
                    }
                    Err(e2) => {
                        log::warn!("Fallback Q4_0 also failed: {}. Using dummy.", e2);
                        Program::builder()
                            .devices(device)
                            .src("__kernel void kernel_mul_mat_q4_0_f32() {}")
                            .build(&context)?
                    }
                }
            }
        };

        // Get rows: try original, fallback to dummy
        let get_rows_src = include_str!("../../../kernels/get_rows.cl");
        let get_rows_program = match Program::builder()
            .devices(device)
            .src(get_rows_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => p,
            Err(e) => {
                log::warn!("get_rows.cl failed: {}. Using dummy.", e);
                Program::builder()
                    .devices(device)
                    .src("__kernel void kernel_get_rows_q4_0() {} __kernel void kernel_get_rows_f32() {}")
                    .build(&context)?
            }
        };

        // Quantize Q4_0: try original, fallback to dummy
        let q4_quant_src = include_str!("../../../kernels/quantize_q4_0.cl");
        let quant_q4_0_program = match Program::builder()
            .devices(device)
            .src(q4_quant_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => p,
            Err(e) => {
                log::warn!("quantize_q4_0.cl failed: {}. Using dummy.", e);
                Program::builder()
                    .devices(device)
                    .src("__kernel void kernel_quantize_f32_to_q4_0() {}")
                    .build(&context)?
            }
        };

        // F16 GEMV: Adreno-optimized multi-row kernel (Q4 pattern ported to F16)
        let f16_src = include_str!("../../../kernels/mul_mv_f16_f32.cl");
        let f16_program = match Program::builder()
            .devices(device)
            .src(f16_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("mul_mv_f16_f32.cl compiled (GEMV optimized)");
                p
            }
            Err(e) => {
                log::warn!("mul_mv_f16_f32.cl failed: {}. Using dummy.", e);
                Program::builder()
                    .devices(device)
                    .src("__kernel void kernel_mul_mat_f16_f32() {}")
                    .build(&context)?
            }
        };

        // Create and cache all kernel objects once
        let kernel_cache = KernelCache {
            kernel_mul_mat_f32_f32: ocl::core::create_kernel(&program, "kernel_mul_mat_f32_f32")?,
            kernel_mul_mat_f16_f32: ocl::core::create_kernel(
                &f16_program,
                "kernel_mul_mat_f16_f32",
            )?,
            kernel_mul_mat_q4_0_f32: ocl::core::create_kernel(
                &q4_0_program,
                "kernel_mul_mat_q4_0_f32",
            )?,
            kernel_rms_norm_opt: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_rms_norm_opt",
            )?,
            kernel_softmax_opt: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_softmax_opt",
            )?,
            kernel_rope_simple: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_rope_simple",
            )?,
            kernel_silu_mul_simple: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_silu_mul_simple",
            )?,
            kernel_add_assign_simple: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_add_assign_simple",
            )?,
            kernel_scale_simple: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_scale_simple",
            )?,
            kernel_get_rows_q4_0: ocl::core::create_kernel(
                &get_rows_program,
                "kernel_get_rows_q4_0",
            )?,
            kernel_get_rows_f32: ocl::core::create_kernel(
                &get_rows_program,
                "kernel_get_rows_f32",
            )?,
            kernel_attn_gen: ocl::core::create_kernel(&simple_ops_program, "kernel_attn_gen")?,
            kernel_cast_f32_to_f16: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_cast_f32_to_f16",
            )?,
            kernel_attn_gen_half: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_attn_gen_half",
            )?,
            kernel_quantize_f32_to_q4_0: ocl::core::create_kernel(
                &quant_q4_0_program,
                "kernel_quantize_f32_to_q4_0",
            )?,
        };

        log::info!("OpenCL kernels cached successfully");

        Ok(Self {
            context,
            queue,
            device,
            program,
            simple_ops_program,
            q4_0_program,
            f16_program,
            quant_q4_0_program,
            get_rows_program,
            kernels: Mutex::new(kernel_cache),
        })
    }

    /// F16 weight matmul: A(F32) x B^T(F16) -> Out(F32)
    /// Uses the mul_mat_f16_f32 kernel. B is the F16 weight matrix (row-major, transposed).
    pub fn matmul_f16(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();

        let (m, k) = if a_dims.len() == 3 {
            (a_dims[0] * a_dims[1], a_dims[2])
        } else {
            (a_dims[0], a_dims[1])
        };
        let n = b_dims[0];

        let a_buf =
            get_cl_mem(a.buffer().as_ref()).map_err(|_| anyhow!("A is not OpenCL buffer"))?;
        let b_buf =
            get_cl_mem(b.buffer().as_ref()).map_err(|_| anyhow!("B is not OpenCL buffer"))?;
        let out_buf =
            get_cl_mem(out.buffer().as_ref()).map_err(|_| anyhow!("Out is not OpenCL buffer"))?;

        let kernels = self
            .kernels
            .lock()
            .map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel = &kernels.kernel_mul_mat_f16_f32;

        // Same 15-arg signature as Q4 GEMV kernel
        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1i32;
        let ne10 = k as i32;
        let ne12 = 1i32;
        let ne0 = n as i32;
        let ne1 = m as i32;
        let r2 = 1i32;
        let r3 = 1i32;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(b_buf))?; // src0=weight(F16)
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(a_buf))?; // src1=activation(F32)
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

            // N_DST=4 rows per workgroup, 64 threads (1 subgroup)
            let local_work_size: [usize; 3] = [64, 1, 1];
            let group_size_0 = n.div_ceil(4);
            let global_work_size: [usize; 3] = [group_size_0 * local_work_size[0], m, 1];

            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                3,
                None,
                &global_work_size,
                Some(local_work_size),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
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

        let a_buf =
            get_cl_mem(a.buffer().as_ref()).map_err(|_| anyhow!("A is not OpenCL buffer"))?;
        let b_buf =
            get_cl_mem(b.buffer().as_ref()).map_err(|_| anyhow!("B is not OpenCL buffer"))?;
        let out_buf =
            get_cl_mem(out.buffer().as_ref()).map_err(|_| anyhow!("Out is not OpenCL buffer"))?;

        let kernels = self
            .kernels
            .lock()
            .map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
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
            let group_size_0 = n.div_ceil(4);
            let global_work_size: [usize; 3] = [group_size_0 * local_work_size[0], m, 1];

            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                3,
                None,
                &global_work_size,
                Some(local_work_size),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
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
        let buf =
            get_cl_mem(t.buffer().as_ref()).map_err(|_| anyhow::anyhow!("Not OpenCL buffer"))?;
        unsafe {
            ocl::core::enqueue_read_buffer(
                &self.queue,
                buf,
                true,
                0,
                dst,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
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
        // Use device-only memory for copies (faster)
        let memory = crate::backend::opencl::memory::OpenCLMemory::new(
            self.context.clone(),
            self.queue.clone(),
            false,
        );
        let buffer = memory.alloc(size, src.dtype())?;

        // Share the source tensor's backend (Arc clone = cheap reference count)
        // instead of creating a new backend + cloning 15 kernel objects.
        // The Mutex<KernelCache> already provides thread safety.
        let new_tensor = Tensor::new(src.shape().clone(), buffer.clone(), src.backend().clone());

        // Device-to-Device Copy - try using get_cl_mem for both buffer types
        if let (Ok(src_mem), Ok(dst_mem)) = (
            get_cl_mem(src.buffer().as_ref()),
            get_cl_mem(buffer.as_ref()),
        ) {
            unsafe {
                ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                    &self.queue,
                    src_mem,
                    dst_mem,
                    0,
                    0,
                    size,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
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
                        &self.queue,
                        dst_mem,
                        true,
                        0,
                        src_slice,
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
                }
            } else {
                return Err(anyhow!(
                    "Failed to get cl_mem handle for destination buffer"
                ));
            }
        }

        Ok(new_tensor)
    }

    fn copy_into(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        let size = src.size();
        assert_eq!(size, dst.size(), "copy_into: size mismatch");

        if let (Ok(src_mem), Ok(dst_mem)) = (
            get_cl_mem(src.buffer().as_ref()),
            get_cl_mem(dst.buffer().as_ref()),
        ) {
            // GPU→GPU: enqueue_copy_buffer + flush (submit without waiting)
            unsafe {
                ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                    &self.queue,
                    src_mem,
                    dst_mem,
                    0,
                    0,
                    size,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            // Flush: submit queued commands immediately for better GPU pipelining
            ocl::core::flush(&self.queue)?;
        } else {
            // Fallback: CPU memcpy
            let src_ptr = src.as_ptr();
            let dst_ptr = dst.as_mut_ptr();
            if !src_ptr.is_null() && !dst_ptr.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
                }
            }
        }
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        ocl::core::finish(&self.queue)?;
        Ok(())
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();

        let (m, k) = if a_dims.len() == 3 {
            (a_dims[0] * a_dims[1], a_dims[2])
        } else {
            (a_dims[0], a_dims[1])
        };
        let n = b_dims[0];

        if b_dims[1] != k {
            return Err(anyhow!(
                "Dimension mismatch: A[{},{}] * B^T[{},{}]",
                m,
                k,
                n,
                b_dims[1]
            ));
        }

        let a_buf =
            get_cl_mem(a.buffer().as_ref()).map_err(|_| anyhow!("A is not OpenCL buffer"))?;
        let b_buf =
            get_cl_mem(b.buffer().as_ref()).map_err(|_| anyhow!("B is not OpenCL buffer"))?;
        let c_buf =
            get_cl_mem(out.buffer().as_ref()).map_err(|_| anyhow!("Out is not OpenCL buffer"))?;

        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1i32;
        let nb00 = 4u64;
        let nb01 = k as u64 * 4;
        let nb02 = n as u64 * k as u64 * 4;
        let nb03 = nb02;
        let ne10 = k as i32;
        let ne11 = m as i32;
        let ne12 = 1i32;
        let nb10 = 4u64;
        let nb11 = k as u64 * 4;
        let nb12 = m as u64 * k as u64 * 4;
        let nb13 = nb12;
        let ne0 = n as i32;
        let ne1 = m as i32;
        let r2 = 1i32;
        let r3 = 1i32;

        let kernels = self
            .kernels
            .lock()
            .map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
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
            let global_work_size: [usize; 3] = [n * local_size, m.div_ceil(4), 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];

            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                3,
                None,
                &global_work_size,
                Some(local_work_size),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        match b.dtype() {
            DType::Q4_0 => self.matmul_q4_0(a, b, out),
            DType::F16 => self.matmul_f16(a, b, out),
            _ => self.matmul(a, b, out), // F32 path
        }
    }

    fn rms_norm(&self, x: &mut Tensor, weight: &Tensor, epsilon: f32) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let w_buf = get_cl_mem(weight.buffer().as_ref())
            .map_err(|_| anyhow!("Weight is not OpenCL buffer"))?;

        let kernels = self
            .kernels
            .lock()
            .map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
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

            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                1,
                None,
                &global_work_size,
                Some(local_work_size),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        let dims = x.shape().dims();
        let (seq_len, num_heads, head_dim) = if dims.len() == 4 {
            (dims[1], dims[2], dims[3])
        } else if dims.len() == 3 {
            (dims[0], dims[1], dims[2])
        } else {
            return Err(anyhow!("RoPE expects 3 or 4 dims"));
        };

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;

        let kernels = self
            .kernels
            .lock()
            .map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel = &kernels.kernel_rope_simple;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(num_heads as i32)))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&(seq_len as i32)))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&(start_pos as i32)))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&theta))?;

            let work_size = seq_len * num_heads * (head_dim / 2);
            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                1,
                None,
                &[work_size, 1, 1],
                None::<[usize; 3]>,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        match (src.dtype(), dst.dtype()) {
            (DType::F32, DType::F16) => {
                let src_buf = get_cl_mem(src.buffer().as_ref())?;
                let dst_buf = get_cl_mem(dst.buffer().as_ref())?;
                let num_elements: usize = src.shape().dims().iter().product();
                let kernels = self
                    .kernels
                    .lock()
                    .map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
                let kernel = &kernels.kernel_cast_f32_to_f16;
                unsafe {
                    ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(src_buf))?;
                    ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(dst_buf))?;
                    ocl::core::set_kernel_arg(
                        kernel,
                        2,
                        ocl::core::ArgVal::scalar(&(num_elements as i32)),
                    )?;
                    let gws: [usize; 3] = [num_elements.div_ceil(64) * 64, 1, 1];
                    let lws: [usize; 3] = [64, 1, 1];
                    ocl::core::enqueue_kernel(
                        &self.queue,
                        kernel,
                        1,
                        None,
                        &gws,
                        Some(lws),
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
                }
                Ok(())
            }
            (DType::F32, DType::Q4_0) => {
                let src_buf = get_cl_mem(src.buffer().as_ref())?;
                let dst_buf = get_cl_mem(dst.buffer().as_ref())?;
                let num_elements: usize = src.shape().dims().iter().product();

                if !num_elements.is_multiple_of(32) {
                    return Err(anyhow!("Q4_0 cast requires size multiple of 32"));
                }

                let kernels = self
                    .kernels
                    .lock()
                    .map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
                let kernel = &kernels.kernel_quantize_f32_to_q4_0;
                let num_blocks = num_elements / 32;

                unsafe {
                    ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(src_buf))?;
                    ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(dst_buf))?;
                    ocl::core::set_kernel_arg(
                        kernel,
                        2,
                        ocl::core::ArgVal::scalar(&(num_elements as i32)),
                    )?;

                    let local_size = 64;
                    let global_size = num_blocks.div_ceil(local_size) * local_size;

                    ocl::core::enqueue_kernel(
                        &self.queue,
                        kernel,
                        1,
                        None,
                        &[global_size, 1, 1],
                        Some([local_size, 1, 1]),
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
                }
                Ok(())
            }
            _ => Err(anyhow!(
                "OpenCL cast: unsupported {:?} -> {:?}",
                src.dtype(),
                dst.dtype()
            )),
        }
    }

    fn attention_gen(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        num_heads_q: usize,
        num_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
    ) -> Result<()> {
        let q_buf =
            get_cl_mem(q.buffer().as_ref()).map_err(|_| anyhow!("Q is not OpenCL buffer"))?;
        let k_buf =
            get_cl_mem(k_cache.buffer().as_ref()).map_err(|_| anyhow!("K is not OpenCL buffer"))?;
        let v_buf =
            get_cl_mem(v_cache.buffer().as_ref()).map_err(|_| anyhow!("V is not OpenCL buffer"))?;
        let o_buf =
            get_cl_mem(out.buffer().as_ref()).map_err(|_| anyhow!("Out is not OpenCL buffer"))?;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();

        // Detect layout from shape: HeadMajor [batch, kv_heads, capacity, head_dim]
        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == num_heads_kv && k_shape[1] != k_shape[2];
        let capacity = if is_head_major { k_shape[2] } else { 0 };

        let (kv_pos_stride, kv_head_stride) = if is_head_major {
            (head_dim as i32, (capacity * head_dim) as i32)
        } else {
            ((num_heads_kv * head_dim) as i32, head_dim as i32)
        };

        let kernels = self
            .kernels
            .lock()
            .map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;

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
            ocl::core::set_kernel_arg(
                kernel,
                6,
                ocl::core::ArgVal::scalar(&(num_heads_kv as i32)),
            )?;
            ocl::core::set_kernel_arg(
                kernel,
                7,
                ocl::core::ArgVal::scalar(&(cache_seq_len as i32)),
            )?;
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&scale))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&kv_pos_stride))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&kv_head_stride))?;
            ocl::core::set_kernel_arg(
                kernel,
                11,
                ocl::core::ArgVal::local::<f32>(&local_mem_size),
            )?;

            let global_work_size: [usize; 3] = [num_heads_q * local_size, 1, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];

            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                1,
                None,
                &global_work_size,
                Some(local_work_size),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    fn silu_mul(&self, x: &mut Tensor, y: &Tensor) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let y_buf =
            get_cl_mem(y.buffer().as_ref()).map_err(|_| anyhow!("Y is not OpenCL buffer"))?;

        let kernels = self
            .kernels
            .lock()
            .map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel = &kernels.kernel_silu_mul_simple;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(y_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(size as i32)))?;

            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                1,
                None,
                &[size, 1, 1],
                None::<[usize; 3]>,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    fn add_assign(&self, x: &mut Tensor, y: &Tensor) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let y_buf =
            get_cl_mem(y.buffer().as_ref()).map_err(|_| anyhow!("Y is not OpenCL buffer"))?;

        let kernels = self
            .kernels
            .lock()
            .map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel = &kernels.kernel_add_assign_simple;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(y_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(size as i32)))?;

            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                1,
                None,
                &[size, 1, 1],
                None::<[usize; 3]>,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    fn matmul_slice(
        &self,
        a: &Tensor,
        b: &Tensor,
        _rows: usize,
        _cols: usize,
        out: &mut Tensor,
    ) -> Result<()> {
        self.matmul_transposed(a, b, out)
    }

    fn scale(&self, x: &mut Tensor, val: f32) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;

        let kernels = self
            .kernels
            .lock()
            .map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel = &kernels.kernel_scale_simple;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&val))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(size as i32)))?;

            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                1,
                None,
                &[size, 1, 1],
                None::<[usize; 3]>,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;

        let kernels = self
            .kernels
            .lock()
            .map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
        let kernel = &kernels.kernel_softmax_opt;
        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::local::<f32>(&local_mem_size))?;

            let global_work_size: [usize; 3] = [rows * local_size, 1, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];

            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                1,
                None,
                &global_work_size,
                Some(local_work_size),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    fn gather(&self, src: &Tensor, indices: &Tensor, dst: &mut Tensor) -> Result<()> {
        let dims = src.shape().dims();
        let k = dims[dims.len() - 1];

        let src_buf =
            get_cl_mem(src.buffer().as_ref()).map_err(|_| anyhow!("Src is not OpenCL buffer"))?;
        let idx_buf = get_cl_mem(indices.buffer().as_ref())
            .map_err(|_| anyhow!("Indices is not OpenCL buffer"))?;
        let dst_buf =
            get_cl_mem(dst.buffer().as_ref()).map_err(|_| anyhow!("Dst is not OpenCL buffer"))?;

        let ne00 = k as i32;
        let nb01 = match src.dtype() {
            DType::F32 => (k * 4) as u64,
            DType::Q4_0 => (k / 32 * 18) as u64,
            _ => {
                return Err(anyhow!(
                    "Unsupported src dtype for gather: {:?}",
                    src.dtype()
                ));
            }
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

        let kernels = self
            .kernels
            .lock()
            .map_err(|e| anyhow!("Kernel lock poisoned: {}", e))?;
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

            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                1,
                None,
                &global_work_size,
                Some(local_work_size),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    fn buffer_shift(
        &self,
        tensor: &mut Tensor,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) -> Result<()> {
        if src_offset == dst_offset || count == 0 {
            return Ok(());
        }

        // If CPU pointer is available, use CPU memmove (SharedBuffer, mapped UnifiedBuffer)
        let ptr = tensor.as_mut_ptr();
        if !ptr.is_null() {
            let type_size = match tensor.dtype() {
                DType::F32 => 4,
                DType::F16 => 2,
                DType::U8 => 1,
                DType::Q4_0 => std::mem::size_of::<crate::core::quant::BlockQ4_0>(),
                _ => 1,
            };
            unsafe {
                std::ptr::copy(
                    ptr.add(src_offset * type_size),
                    ptr.add(dst_offset * type_size),
                    count * type_size,
                );
            }
            return Ok(());
        }

        // GPU path: use enqueue_copy_buffer via cl_mem handle.
        // No queue.finish() after enqueue — the in-order command queue guarantees
        // that subsequent kernel dispatches on the same queue are serialized after
        // this copy, so explicit synchronization would stall the GPU pipeline.
        let buf_mem = get_cl_mem(tensor.buffer().as_ref())?;
        let type_size = match tensor.dtype() {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::U8 => 1,
            DType::Q4_0 => std::mem::size_of::<crate::core::quant::BlockQ4_0>(),
            _ => {
                return Err(anyhow!(
                    "Unsupported dtype for buffer_shift: {:?}",
                    tensor.dtype()
                ));
            }
        };

        let src_byte = src_offset * type_size;
        let dst_byte = dst_offset * type_size;
        let byte_count = count * type_size;

        // OpenCL spec: clEnqueueCopyBuffer with overlapping src/dst regions in the
        // same buffer is undefined behavior. Detect overlap and use a temp buffer.
        let no_overlap = (src_byte + byte_count <= dst_byte) || (dst_byte + byte_count <= src_byte);

        if no_overlap {
            unsafe {
                ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                    &self.queue,
                    buf_mem,
                    buf_mem,
                    src_byte,
                    dst_byte,
                    byte_count,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        } else {
            // Overlap: 2-pass copy via temporary GPU buffer.
            // Safe to drop temp after enqueue: clReleaseMemObject defers deallocation
            // until pending commands referencing this buffer complete (OpenCL spec).
            let temp = ocl::Buffer::<u8>::builder()
                .queue(self.queue.clone())
                .len(byte_count)
                .build()?;
            let temp_mem = temp.as_core();

            unsafe {
                ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                    &self.queue,
                    buf_mem,
                    temp_mem,
                    src_byte,
                    0,
                    byte_count,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
                ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                    &self.queue,
                    temp_mem,
                    buf_mem,
                    0,
                    dst_byte,
                    byte_count,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        }

        Ok(())
    }

    fn copy_slice(
        &self,
        src: &Tensor,
        dst: &mut Tensor,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) -> Result<()> {
        let type_size = match src.dtype() {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::U8 => 1,
            DType::Q4_0 => std::mem::size_of::<crate::core::quant::BlockQ4_0>(),
            _ => {
                return Err(anyhow!(
                    "Unsupported dtype for copy_slice: {:?}",
                    src.dtype()
                ));
            }
        };

        let src_m = get_cl_mem(src.buffer().as_ref());
        let dst_m = get_cl_mem(dst.buffer().as_ref());

        if let (Ok(sb), Ok(db)) = (src_m.as_ref(), dst_m.as_ref()) {
            let src_byte_off = src_offset * type_size;
            let dst_byte_off = dst_offset * type_size;
            let byte_len = count * type_size;

            unsafe {
                ocl::core::enqueue_copy_buffer::<u8, _, _, _>(
                    &self.queue,
                    *sb,
                    *db,
                    src_byte_off,
                    dst_byte_off,
                    byte_len,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
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
                    ocl::core::enqueue_write_buffer(
                        &self.queue,
                        db,
                        true,
                        dst_byte_off,
                        std::slice::from_raw_parts(src_ptr.add(src_byte_off), byte_len),
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
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
                    ocl::core::enqueue_read_buffer(
                        &self.queue,
                        sb,
                        true,
                        src_byte_off,
                        std::slice::from_raw_parts_mut(dst_ptr.add(dst_byte_off), byte_len),
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
                }
                return Ok(());
            }
        }

        Err(anyhow!(
            "Unsupported copy_slice combination in OpenCL backend or null pointers"
        ))
    }
}

#[cfg(test)]
mod gpu_buffer_shift_tests {
    use super::*;
    use crate::core::shape::Shape;

    fn try_create_backend() -> Option<Arc<OpenCLBackend>> {
        OpenCLBackend::new().ok().map(Arc::new)
    }

    /// Allocate a GPU-only (non-zero-copy) buffer, write data, return Tensor.
    fn make_gpu_tensor(backend: &Arc<OpenCLBackend>, data: &[f32], shape: Vec<usize>) -> Tensor {
        let byte_len = data.len() * 4;
        let mem = memory::OpenCLMemory::new(
            backend.context.clone(),
            backend.queue.clone(),
            false, // device-only
        );
        let buf = mem.alloc(byte_len, DType::F32).unwrap();
        let tensor = Tensor::new(Shape::new(shape), buf, backend.clone());

        // Write data to GPU buffer
        let cl_mem = get_cl_mem(tensor.buffer().as_ref()).unwrap();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                cl_mem,
                true,
                0,
                bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }
        tensor
    }

    /// Read f32 data back from a GPU tensor.
    fn read_gpu_tensor(backend: &Arc<OpenCLBackend>, tensor: &Tensor) -> Vec<f32> {
        let n = tensor.buffer().size() / 4;
        let mut result = vec![0.0f32; n];
        let cl_mem = get_cl_mem(tensor.buffer().as_ref()).unwrap();
        unsafe {
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                cl_mem,
                true,
                0,
                std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, n * 4),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }
        result
    }

    #[test]
    fn test_buffer_shift_gpu_no_overlap_f32() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // [1, 2, 3, 4, 5, 6, 7, 8] — shift elements [4..8] to [0..4] (no overlap)
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut tensor = make_gpu_tensor(&backend, &data, vec![8]);

        // src_offset=4, dst_offset=0, count=4 → no overlap (count <= src_offset)
        backend.buffer_shift(&mut tensor, 4, 0, 4).unwrap();

        let result = read_gpu_tensor(&backend, &tensor);
        assert_eq!(&result[0..4], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_buffer_shift_gpu_overlap_f32() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // [1, 2, 3, 4, 5, 6, 7, 8] — shift [2..8] to [0..6] (overlap: count=6 > src=2)
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut tensor = make_gpu_tensor(&backend, &data, vec![8]);

        backend.buffer_shift(&mut tensor, 2, 0, 6).unwrap();

        let result = read_gpu_tensor(&backend, &tensor);
        assert_eq!(&result[0..6], &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_buffer_shift_gpu_zero_count() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        let data: Vec<f32> = (1..=4).map(|x| x as f32).collect();
        let mut tensor = make_gpu_tensor(&backend, &data, vec![4]);

        // No-op
        backend.buffer_shift(&mut tensor, 2, 0, 0).unwrap();

        let result = read_gpu_tensor(&backend, &tensor);
        assert_eq!(&result[..], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_buffer_shift_gpu_same_offset() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        let data: Vec<f32> = (1..=4).map(|x| x as f32).collect();
        let mut tensor = make_gpu_tensor(&backend, &data, vec![4]);

        // src == dst → no-op
        backend.buffer_shift(&mut tensor, 1, 1, 2).unwrap();

        let result = read_gpu_tensor(&backend, &tensor);
        assert_eq!(&result[..], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_prune_prefix_opencl_buffer() {
        use crate::core::kv_cache::KVCache;

        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        let heads = 1;
        let dim = 4;
        let max_seq = 16;
        let mem = memory::OpenCLMemory::new(backend.context.clone(), backend.queue.clone(), false);

        let k_buf = mem.alloc(max_seq * heads * dim * 4, DType::F32).unwrap();
        let v_buf = mem.alloc(max_seq * heads * dim * 4, DType::F32).unwrap();
        let k = Tensor::new(
            Shape::new(vec![1, max_seq, heads, dim]),
            k_buf,
            backend.clone() as Arc<dyn Backend>,
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq, heads, dim]),
            v_buf,
            backend.clone() as Arc<dyn Backend>,
        );
        let mut cache = KVCache::new(k, v, max_seq);

        // Fill 8 positions: pos i → all values = (i+1) as f32
        let cpu_mem = crate::buffer::shared_buffer::SharedBuffer::new(heads * dim * 4, DType::F32);
        for i in 0..8 {
            let val = (i + 1) as f32;
            let kb = Arc::new(crate::buffer::shared_buffer::SharedBuffer::new(
                heads * dim * 4,
                DType::F32,
            ));
            let vb = Arc::new(crate::buffer::shared_buffer::SharedBuffer::new(
                heads * dim * 4,
                DType::F32,
            ));
            unsafe {
                let kp = kb.as_mut_ptr() as *mut f32;
                let vp = vb.as_mut_ptr() as *mut f32;
                for j in 0..(heads * dim) {
                    *kp.add(j) = val;
                    *vp.add(j) = val * 10.0;
                }
            }
            let kt = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                kb,
                backend.clone() as Arc<dyn Backend>,
            );
            let vt = Tensor::new(
                Shape::new(vec![1, 1, heads, dim]),
                vb,
                backend.clone() as Arc<dyn Backend>,
            );
            cache.update(&kt, &vt).unwrap();
        }
        assert_eq!(cache.current_pos, 8);

        // Prune first 3 tokens
        cache.prune_prefix(3).unwrap();
        assert_eq!(cache.current_pos, 5);

        // Read K buffer back and verify: pos 0 should be old pos 3 (value 4.0)
        let k_data = {
            let cl_mem = get_cl_mem(cache.k_buffer.buffer().as_ref()).unwrap();
            let n = 8 * heads * dim; // read enough elements
            let mut buf = vec![0.0f32; n];
            unsafe {
                ocl::core::enqueue_read_buffer(
                    &backend.queue,
                    cl_mem,
                    true,
                    0,
                    std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n * 4),
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )
                .unwrap();
            }
            buf
        };

        // pos 0 = old pos 3 = value 4.0
        assert_eq!(k_data[0], 4.0);
        // pos 1 = old pos 4 = value 5.0
        assert_eq!(k_data[dim], 5.0);
        // pos 4 = old pos 7 = value 8.0
        assert_eq!(k_data[4 * dim], 8.0);

        drop(cpu_mem); // suppress unused warning
    }
}
