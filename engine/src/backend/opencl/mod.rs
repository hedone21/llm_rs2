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
use std::cell::UnsafeCell;
#[cfg(test)]
use std::sync::Arc;

pub mod buffer;
pub mod gpu_score;
pub mod memory;
pub mod plan;

/// Helper function to get the OpenCL memory handle from a tensor buffer.
/// Works with both UnifiedBuffer and legacy OpenCLBuffer.
pub fn get_cl_mem(buf: &dyn Buffer) -> Result<&ocl::core::Mem> {
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
    kernel_gelu_tanh_mul: CoreKernel,
    kernel_add_assign_simple: CoreKernel,
    kernel_add_row_bias: CoreKernel,
    kernel_scale_simple: CoreKernel,
    kernel_get_rows_q4_0: CoreKernel,
    kernel_get_rows_f32: CoreKernel,
    kernel_get_rows_f16: CoreKernel,
    kernel_attn_gen: CoreKernel,
    kernel_cast_f32_to_f16: CoreKernel,
    kernel_attn_gen_half: CoreKernel,
    kernel_quantize_f32_to_q4_0: CoreKernel,
    kernel_kv_scatter_f32_to_f16: CoreKernel,
    kernel_rms_norm_oop: CoreKernel,
    kernel_add_rms_norm_oop: CoreKernel,
    kernel_flash_attn_f32: Option<CoreKernel>,
    kernel_flash_attn_f32_f16: Option<CoreKernel>,
    // GEMM kernels for prefill (tiled matrix multiply, M > 1)
    kernel_mul_mm_f16_f32: Option<CoreKernel>,
    kernel_mul_mm_f32_f32: Option<CoreKernel>,
    // KIVI Q2 kernels (optional — dequantize/scatter for GPU-native KiviCache)
    kernel_kivi_deq_value_q2: Option<CoreKernel>,
    kernel_kivi_deq_key_q2: Option<CoreKernel>,
    kernel_kivi_scatter_residual: Option<CoreKernel>,
    kernel_kivi_gather_update: Option<CoreKernel>,
    // F16 GEMV N_DST=4 variant for large-N matmuls (lm_head, FFN)
    kernel_mul_mat_f16_f32_l4: Option<CoreKernel>,
    // true when f16 kernel is the nosub fallback (1D work group, N_DST=4 rows/WG)
    f16_is_nosub: bool,
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

    // Flash attention programs (optional — compiled with head_dim-specific defines)
    pub flash_attn_f32_program: Option<Program>,
    pub flash_attn_f32_f16_program: Option<Program>,

    // F16 GEMV N_DST=4 for large-N decode (lm_head, FFN)
    pub f16_l4_program: Option<Program>,

    // GEMM programs for prefill (tiled matmul, optional — fallback to GEMV)
    pub gemm_f16_program: Option<Program>,
    pub gemm_f32_program: Option<Program>,

    // Cached kernels — inference is single-threaded, no lock needed.
    // UnsafeCell avoids Mutex overhead (~170us/lock on Adreno).
    kernels: UnsafeCell<KernelCache>,

    // true on UMA devices (Adreno, Mali): CL_MEM_ALLOC_HOST_PTR for zero-copy.
    // false on discrete GPUs (NVIDIA): device-only buffers for correct behavior.
    pub use_zero_copy: bool,

    // Pre-allocated 1-element dummy buffer for attention_gen when scores_out is None.
    // Avoids per-call GPU buffer allocation (16 layers x every token).
    dummy_score_buf: ocl::core::Mem,

    // GPU-side attention score accumulator (optional).
    // When active, attention_gen writes to a persistent GPU buffer and
    // reduce_layer aggregates scores without CPU readback.
    // UnsafeCell follows the same pattern as `kernels` — single-threaded access.
    gpu_score_acc: UnsafeCell<Option<gpu_score::GpuScoreAccumulator>>,

    // Cached compiler options for building additional programs (e.g., score_reduce.cl).
    cl_opts: String,
}

// SAFETY: OpenCLBackend is only accessed from the inference thread.
// The UnsafeCell is safe because we guarantee single-threaded access
// during kernel dispatch (same as llama.cpp's lock-free approach).
unsafe impl Send for OpenCLBackend {}
unsafe impl Sync for OpenCLBackend {}

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
        // Log memory limits
        if let Ok(global_mem) = device.info(ocl::core::DeviceInfo::GlobalMemSize) {
            log::info!(
                "GPU Global Memory: {} MB",
                global_mem.to_string().trim().parse::<u64>().unwrap_or(0) / (1024 * 1024)
            );
        }
        if let Ok(max_alloc) = device.info(ocl::core::DeviceInfo::MaxMemAllocSize) {
            log::info!(
                "GPU Max Alloc Size: {} MB",
                max_alloc.to_string().trim().parse::<u64>().unwrap_or(0) / (1024 * 1024)
            );
        }

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
                    .src("__kernel void kernel_get_rows_q4_0() {} __kernel void kernel_get_rows_f32() {} __kernel void kernel_get_rows_f16() {}")
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
        let f16_fallback_src = include_str!("../../../kernels/fallback/mul_mv_f16_f32_nosub.cl");
        let mut f16_is_nosub = false;
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
                log::warn!("mul_mv_f16_f32.cl failed: {}. Trying fallback.", e);
                match Program::builder()
                    .devices(device)
                    .src(f16_fallback_src)
                    .cmplr_opt(&cl_opts)
                    .build(&context)
                {
                    Ok(p) => {
                        log::info!("Using fallback/mul_mv_f16_f32_nosub.cl");
                        f16_is_nosub = true;
                        p
                    }
                    Err(e2) => {
                        log::warn!("Fallback F16 also failed: {}. Using dummy.", e2);
                        Program::builder()
                            .devices(device)
                            .src("__kernel void kernel_mul_mat_f16_f32() {}")
                            .build(&context)?
                    }
                }
            }
        };

        // F16 GEMV N_DST=4 variant for large-N decode (lm_head, FFN)
        let f16_l4_src = include_str!("../../../kernels/mul_mv_f16_f32_l4.cl");
        let f16_l4_program = match Program::builder()
            .devices(device)
            .src(f16_l4_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("mul_mv_f16_f32_l4.cl compiled (GEMV N_DST=4 for large N)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "mul_mv_f16_f32_l4.cl failed: {}. Large-N will use N_DST=2 fallback.",
                    e
                );
                None
            }
        };

        // Flash attention kernels — compiled with head_dim-specific defines
        // DK/DV=64 matches Llama 3.2 1B head_dim; BLOCK_M=64 (WG), BLOCK_N=32 (KV tile)
        let flash_attn_defines = "-DDK=64 -DDV=64 -DBLOCK_M=64 -DBLOCK_N=32";
        let flash_attn_f32_opts = format!("{} {}", cl_opts, flash_attn_defines);

        let flash_attn_f32_src = include_str!("../../../kernels/flash_attn_f32.cl");
        let flash_attn_f32_program = match Program::builder()
            .devices(device)
            .src(flash_attn_f32_src)
            .cmplr_opt(&flash_attn_f32_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("flash_attn_f32.cl compiled (BLOCK_M=64, BLOCK_N=32, DK=64)");
                Some(p)
            }
            Err(e) => {
                log::warn!("flash_attn_f32.cl failed: {}. GPU prefill F32 disabled.", e);
                None
            }
        };

        let flash_attn_f32_f16_src = include_str!("../../../kernels/flash_attn_f32_f16.cl");
        let flash_attn_f32_f16_program = match Program::builder()
            .devices(device)
            .src(flash_attn_f32_f16_src)
            .cmplr_opt(&flash_attn_f32_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("flash_attn_f32_f16.cl compiled (Q=F32, KV=F16)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "flash_attn_f32_f16.cl failed: {}. GPU prefill F16 KV disabled.",
                    e
                );
                None
            }
        };

        // GEMM kernels for prefill — tiled matmul (BM=64, BN=64, BK=16)
        let gemm_f16_src = include_str!("../../../kernels/mul_mm_f16_f32_l4_lm.cl");
        let gemm_f16_program = match Program::builder()
            .devices(device)
            .src(gemm_f16_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("mul_mm_f16_f32_l4_lm.cl compiled (GEMM prefill F16)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "mul_mm_f16_f32_l4_lm.cl failed: {}. Prefill will use GEMV.",
                    e
                );
                None
            }
        };

        let gemm_f32_src = include_str!("../../../kernels/mul_mm_f32_f32_l4_lm.cl");
        let gemm_f32_program = match Program::builder()
            .devices(device)
            .src(gemm_f32_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("mul_mm_f32_f32_l4_lm.cl compiled (GEMM prefill F32)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "mul_mm_f32_f32_l4_lm.cl failed: {}. Prefill will use GEMV.",
                    e
                );
                None
            }
        };

        // KIVI Q2 kernels (optional — dequantize/scatter for GPU-native KiviCache)
        let kivi_q2_src = include_str!("../../../kernels/kivi_q2.cl");
        let kivi_q2_kernels = Program::builder()
            .devices(device)
            .src(kivi_q2_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
            .ok();
        if kivi_q2_kernels.is_some() {
            log::info!("kivi_q2.cl compiled (KIVI Q2 dequant/scatter kernels)");
        } else {
            log::warn!("kivi_q2.cl failed to compile. KIVI Q2 GPU kernels disabled.");
        }

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
            kernel_gelu_tanh_mul: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_gelu_tanh_mul",
            )?,
            kernel_add_assign_simple: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_add_assign_simple",
            )?,
            kernel_add_row_bias: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_add_row_bias",
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
            kernel_get_rows_f16: ocl::core::create_kernel(
                &get_rows_program,
                "kernel_get_rows_f16",
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
            kernel_kv_scatter_f32_to_f16: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_kv_scatter_f32_to_f16",
            )?,
            kernel_rms_norm_oop: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_rms_norm_oop",
            )?,
            kernel_add_rms_norm_oop: ocl::core::create_kernel(
                &simple_ops_program,
                "kernel_add_rms_norm_oop",
            )?,
            kernel_flash_attn_f32: flash_attn_f32_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32").ok()),
            kernel_flash_attn_f32_f16: flash_attn_f32_f16_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32_f16").ok()),
            kernel_mul_mm_f16_f32: gemm_f16_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_mul_mm_f16_f32_l4_lm").ok()),
            kernel_mul_mm_f32_f32: gemm_f32_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_mul_mm_f32_f32_l4_lm").ok()),
            kernel_kivi_deq_value_q2: kivi_q2_kernels
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kivi_dequantize_value_q2").ok()),
            kernel_kivi_deq_key_q2: kivi_q2_kernels
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kivi_dequantize_key_q2").ok()),
            kernel_kivi_scatter_residual: kivi_q2_kernels
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kivi_scatter_residual").ok()),
            kernel_kivi_gather_update: kivi_q2_kernels
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kivi_gather_update").ok()),
            kernel_mul_mat_f16_f32_l4: f16_l4_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_mul_mat_f16_f32_l4").ok()),
            f16_is_nosub,
        };

        log::info!("OpenCL kernels cached successfully");

        // Auto-detect UMA (integrated GPU) vs discrete GPU for zero-copy decision.
        // CL_DEVICE_HOST_UNIFIED_MEMORY is the standard way; fall back to name heuristic.
        let use_zero_copy = {
            let unified_mem = device
                .info(ocl::core::DeviceInfo::HostUnifiedMemory)
                .map(|v| v.to_string().trim().to_lowercase())
                .unwrap_or_default();
            if unified_mem == "true" || unified_mem == "1" {
                true
            } else {
                let name_lower = device_name.to_lowercase();
                name_lower.contains("adreno")
                    || name_lower.contains("mali")
                    || name_lower.contains("powervr")
            }
        };
        log::info!(
            "Memory mode: {} (zero_copy={})",
            if use_zero_copy {
                "UMA/integrated"
            } else {
                "discrete"
            },
            use_zero_copy
        );

        // Pre-allocate 1-element dummy buffer for attention_gen (scores_out=None path)
        let dummy_score_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                context.as_core(),
                ocl::core::MEM_READ_WRITE,
                1,
                None,
            )?
        };

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
            flash_attn_f32_program,
            flash_attn_f32_f16_program,
            f16_l4_program,
            gemm_f16_program,
            gemm_f32_program,
            kernels: UnsafeCell::new(kernel_cache),
            use_zero_copy,
            dummy_score_buf,
            gpu_score_acc: UnsafeCell::new(None),
            cl_opts: cl_opts.clone(),
        })
    }

    /// Initialize the GPU-side attention score accumulator.
    ///
    /// Compiles `score_reduce.cl` and allocates persistent GPU buffers for
    /// score accumulation. If compilation fails, the accumulator is not
    /// created (graceful degradation — falls back to CPU score path).
    ///
    /// Must be called before the decode loop starts.
    pub fn init_gpu_score_acc(
        &self,
        n_heads_q: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        decay: f32,
    ) -> Result<()> {
        let acc = gpu_score::GpuScoreAccumulator::new(
            self.queue.as_core(),
            self.context.as_core(),
            self.device.as_core(),
            &self.cl_opts,
            n_heads_q,
            n_kv_heads,
            max_seq_len,
            decay,
        )?;
        // SAFETY: single-threaded access (same as kernels UnsafeCell pattern)
        unsafe {
            *self.gpu_score_acc.get() = Some(acc);
        }
        log::info!(
            "GPU score accumulator initialized (n_heads_q={}, n_kv_heads={}, max_seq={})",
            n_heads_q,
            n_kv_heads,
            max_seq_len
        );
        Ok(())
    }

    /// Get a reference to the GPU score accumulator (if initialized).
    pub fn gpu_score_acc(&self) -> Option<&gpu_score::GpuScoreAccumulator> {
        // SAFETY: single-threaded access
        unsafe { (*self.gpu_score_acc.get()).as_ref() }
    }

    /// Get a mutable reference to the GPU score accumulator (if initialized).
    ///
    /// # Safety
    /// Uses UnsafeCell — caller must ensure single-threaded access.
    /// The `&self -> &mut T` pattern is intentional (same as kernels UnsafeCell).
    #[allow(clippy::mut_from_ref)]
    pub fn gpu_score_acc_mut(&self) -> Option<&mut gpu_score::GpuScoreAccumulator> {
        // SAFETY: single-threaded inference, same as kernels UnsafeCell pattern
        unsafe { (*self.gpu_score_acc.get()).as_mut() }
    }

    /// Tiled GEMM for F16 weights: A(F32,[M,K]) x B^T(F16,[N,K]) -> Out(F32,[M,N])
    /// Uses mul_mm_f16_f32_l4_lm kernel with BM=64, BN=64, BK=16 tiling.
    fn matmul_gemm_f16(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        let (m, k) = if a_dims.len() == 3 {
            (a_dims[0] * a_dims[1], a_dims[2])
        } else {
            (a_dims[0], a_dims[1])
        };
        let n = b_dims[0];

        let a_buf = get_cl_mem(a.buffer().as_ref())?;
        let b_buf = get_cl_mem(b.buffer().as_ref())?;
        let out_buf = get_cl_mem(out.buffer().as_ref())?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = kernels
            .kernel_mul_mm_f16_f32
            .as_ref()
            .ok_or_else(|| anyhow!("GEMM F16 kernel not available"))?;

        // All strides in ELEMENT counts (kernel divides by LOAD_VEC internally)
        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1i32;
        let ne11 = m as i32;
        let ne12 = 1i32;
        let stride_a = k as i32;
        let stride_b = k as i32;
        let stride_d = n as i32;
        let batch_stride_a = (n * k) as i32;
        let batch_stride_b = (m * k) as i32;
        let batch_stride_d = (m * n) as i32;
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
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&ne11))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&ne12))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&stride_a))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&stride_b))?;
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&stride_d))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&batch_stride_a))?;
            ocl::core::set_kernel_arg(kernel, 15, ocl::core::ArgVal::scalar(&batch_stride_b))?;
            ocl::core::set_kernel_arg(kernel, 16, ocl::core::ArgVal::scalar(&batch_stride_d))?;
            ocl::core::set_kernel_arg(kernel, 17, ocl::core::ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(kernel, 18, ocl::core::ArgVal::scalar(&r3))?;

            // BM=64, BN=64, TM=4, TN=8 → 128 threads/WG
            // group_id(0) tiles N (weight rows), group_id(1) tiles M (activation rows)
            let local_work_size: [usize; 3] = [128, 1, 1];
            let global_work_size: [usize; 3] = [n.div_ceil(64) * 128, m.div_ceil(64), 1];

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

    /// Tiled GEMM for F32 weights: A(F32,[M,K]) x B^T(F32,[N,K]) -> Out(F32,[M,N])
    fn matmul_gemm_f32(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        let (m, k) = if a_dims.len() == 3 {
            (a_dims[0] * a_dims[1], a_dims[2])
        } else {
            (a_dims[0], a_dims[1])
        };
        let n = b_dims[0];

        let a_buf = get_cl_mem(a.buffer().as_ref())?;
        let b_buf = get_cl_mem(b.buffer().as_ref())?;
        let out_buf = get_cl_mem(out.buffer().as_ref())?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = kernels
            .kernel_mul_mm_f32_f32
            .as_ref()
            .ok_or_else(|| anyhow!("GEMM F32 kernel not available"))?;

        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1i32;
        let ne11 = m as i32;
        let ne12 = 1i32;
        let stride_a = k as i32;
        let stride_b = k as i32;
        let stride_d = n as i32;
        let batch_stride_a = (n * k) as i32;
        let batch_stride_b = (m * k) as i32;
        let batch_stride_d = (m * n) as i32;
        let r2 = 1i32;
        let r3 = 1i32;

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
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&ne11))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&ne12))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&stride_a))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&stride_b))?;
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&stride_d))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&batch_stride_a))?;
            ocl::core::set_kernel_arg(kernel, 15, ocl::core::ArgVal::scalar(&batch_stride_b))?;
            ocl::core::set_kernel_arg(kernel, 16, ocl::core::ArgVal::scalar(&batch_stride_d))?;
            ocl::core::set_kernel_arg(kernel, 17, ocl::core::ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(kernel, 18, ocl::core::ArgVal::scalar(&r3))?;

            let local_work_size: [usize; 3] = [128, 1, 1];
            let global_work_size: [usize; 3] = [n.div_ceil(64) * 128, m.div_ceil(64), 1];

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

    /// F16 weight matmul: A(F32) x B^T(F16) -> Out(F32)
    /// Routes to GEMM kernel for prefill (M > 4) or GEMV kernel for decode.
    pub fn matmul_f16(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();

        let (m, k) = if a_dims.len() == 3 {
            (a_dims[0] * a_dims[1], a_dims[2])
        } else {
            (a_dims[0], a_dims[1])
        };
        let n = b_dims[0];

        // Use tiled GEMM for prefill (M > 8), GEMV for decode/short sequences.
        // GEMM BM=64 tiling wastes work for small M; GEMV N_DST=4 is optimal there.
        const GEMM_THRESHOLD: usize = 8;
        if m > GEMM_THRESHOLD {
            let kernels = unsafe { &*self.kernels.get() };
            if kernels.kernel_mul_mm_f16_f32.is_some() {
                return self.matmul_gemm_f16(a, b, out);
            }
        }

        // GEMV path (decode or GEMM unavailable)
        let a_buf =
            get_cl_mem(a.buffer().as_ref()).map_err(|_| anyhow!("A is not OpenCL buffer"))?;
        let b_buf =
            get_cl_mem(b.buffer().as_ref()).map_err(|_| anyhow!("B is not OpenCL buffer"))?;
        let out_buf =
            get_cl_mem(out.buffer().as_ref()).map_err(|_| anyhow!("Out is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let f16_nosub = kernels.f16_is_nosub;

        // nosub fallback uses 1D work group — L4 kernel also requires subgroups
        let use_l4 = if f16_nosub {
            false
        } else {
            const LARGE_N_THRESHOLD: usize = 4096;
            n > LARGE_N_THRESHOLD && kernels.kernel_mul_mat_f16_f32_l4.is_some()
        };
        let kernel = if use_l4 {
            kernels.kernel_mul_mat_f16_f32_l4.as_ref().unwrap()
        } else {
            &kernels.kernel_mul_mat_f16_f32
        };

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

            if f16_nosub {
                // Nosub kernel: 1D local work group, N_DST=4 rows per WG
                const NOSUB_N_DST: usize = 4;
                let n_groups = n.div_ceil(NOSUB_N_DST);
                let local_work_size: [usize; 3] = [64, 1, 1];
                let global_work_size: [usize; 3] = [n_groups * 64, m, 1];
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
            } else {
                const N_SIMDGROUP: usize = 4;
                let local_work_size: [usize; 3] = [64, N_SIMDGROUP, 1];
                if use_l4 {
                    // N_DST=4: 256 rows/WG
                    const ROWS_PER_WG_L4: usize = 256; // N_SIMDWIDTH(64) * N_DST(4)
                    let n_groups = n.div_ceil(ROWS_PER_WG_L4);
                    let global_work_size: [usize; 3] = [n_groups * 64, N_SIMDGROUP, m];
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
                } else {
                    // N_DST=2: 128 rows/WG
                    const ROWS_PER_WG: usize = 128; // N_SIMDWIDTH(64) * N_DST(2)
                    let n_groups = n.div_ceil(ROWS_PER_WG);
                    let global_work_size: [usize; 3] = [n_groups * 64, N_SIMDGROUP, m];
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
            }
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

        let kernels = unsafe { &*self.kernels.get() };
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

    /// GPU flash attention for prefill. Dispatches flash_attn_f32 or flash_attn_f32_f16 kernel.
    /// Returns Ok(true) if GPU kernel was dispatched, Ok(false) if unsupported (caller should
    /// fall back to CPU).
    ///
    /// Q: [batch, seq_len, n_heads_q, head_dim] F32 (on GPU)
    /// K/V cache: HeadMajor [1, kv_heads, capacity, head_dim] F32 or F16 (on GPU)
    /// Output: [batch, seq_len, n_heads_q * head_dim] F32 (on GPU)
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_prefill(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        n_heads_q: usize,
        n_heads_kv: usize,
        seq_len: usize,
        cache_seq_len: usize,
        head_dim: usize,
        kv_capacity: usize,
        batch_size: usize,
        is_head_major: bool,
    ) -> Result<bool> {
        // Compiled with DK=DV=64; only support matching head_dim
        if head_dim != 64 {
            return Ok(false);
        }

        let kv_dtype = k_cache.dtype();
        let kernels = unsafe { &*self.kernels.get() };
        let kernel = match kv_dtype {
            DType::F32 => match &kernels.kernel_flash_attn_f32 {
                Some(k) => k,
                None => return Ok(false),
            },
            DType::F16 => match &kernels.kernel_flash_attn_f32_f16 {
                Some(k) => k,
                None => return Ok(false),
            },
            _ => return Ok(false), // Q4_0 not supported by flash attn kernel
        };

        let q_buf = get_cl_mem(q.buffer().as_ref())?;
        let k_buf = get_cl_mem(k_cache.buffer().as_ref())?;
        let v_buf = get_cl_mem(v_cache.buffer().as_ref())?;
        let o_buf = get_cl_mem(out.buffer().as_ref())?;

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let kv_elem_size: usize = match kv_dtype {
            DType::F16 => 2,
            _ => 4,
        };

        // Q strides in bytes (F32, row-major [batch, seq_len, n_heads_q, head_dim])
        let q_nb1 = (n_heads_q * head_dim * 4) as u64; // position stride
        let q_nb2 = (head_dim * 4) as u64; // head stride
        let q_nb3 = seq_len as u64 * q_nb1; // batch stride

        // KV strides in bytes
        let (k_nb1, k_nb2, k_nb3) = if is_head_major {
            // HeadMajor: [batch, kv_heads, capacity, head_dim]
            let pos_stride = (head_dim * kv_elem_size) as u64;
            let head_stride = (kv_capacity * head_dim * kv_elem_size) as u64;
            let batch_stride = n_heads_kv as u64 * head_stride;
            (pos_stride, head_stride, batch_stride)
        } else {
            // SeqMajor: [batch, capacity, kv_heads, head_dim]
            let pos_stride = (n_heads_kv * head_dim * kv_elem_size) as u64;
            let head_stride = (head_dim * kv_elem_size) as u64;
            let batch_stride = kv_capacity as u64 * pos_stride;
            (pos_stride, head_stride, batch_stride)
        };
        let (v_nb1, v_nb2, v_nb3) = (k_nb1, k_nb2, k_nb3);

        // Output strides in bytes (F32, [batch, seq_len, n_heads_q, head_dim])
        let o_nb1 = (head_dim * 4) as u64; // head stride
        let o_nb2 = (n_heads_q * head_dim * 4) as u64; // position stride
        let o_nb3 = seq_len as u64 * o_nb2; // batch stride

        let n_q = seq_len as i32;
        let n_kv = cache_seq_len as i32;
        let is_causal = 1i32;
        let n_head = n_heads_q as i32;
        let n_head_kv_arg = n_heads_kv as i32;
        let max_bias = 0.0f32;
        let m0 = 0.0f32;
        let m1 = 0.0f32;
        let n_head_log2 = 0i32;
        let logit_softcap = 0.0f32;
        let zero_u64 = 0u64;
        let zero_i32 = 0i32;

        unsafe {
            // Q, K, V, O buffers + offsets (args 0-7)
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(q_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(k_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(v_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::mem(o_buf))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&zero_u64))?;
            // Scalar params (args 8-12)
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&scale))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&n_q))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&n_kv))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&is_causal))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&n_head))?;
            // Q strides (args 13-15)
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&q_nb1))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&q_nb2))?;
            ocl::core::set_kernel_arg(kernel, 15, ocl::core::ArgVal::scalar(&q_nb3))?;
            // K strides (args 16-18)
            ocl::core::set_kernel_arg(kernel, 16, ocl::core::ArgVal::scalar(&k_nb1))?;
            ocl::core::set_kernel_arg(kernel, 17, ocl::core::ArgVal::scalar(&k_nb2))?;
            ocl::core::set_kernel_arg(kernel, 18, ocl::core::ArgVal::scalar(&k_nb3))?;
            // V strides (args 19-21)
            ocl::core::set_kernel_arg(kernel, 19, ocl::core::ArgVal::scalar(&v_nb1))?;
            ocl::core::set_kernel_arg(kernel, 20, ocl::core::ArgVal::scalar(&v_nb2))?;
            ocl::core::set_kernel_arg(kernel, 21, ocl::core::ArgVal::scalar(&v_nb3))?;
            // O strides (args 22-24)
            ocl::core::set_kernel_arg(kernel, 22, ocl::core::ArgVal::scalar(&o_nb1))?;
            ocl::core::set_kernel_arg(kernel, 23, ocl::core::ArgVal::scalar(&o_nb2))?;
            ocl::core::set_kernel_arg(kernel, 24, ocl::core::ArgVal::scalar(&o_nb3))?;
            // ALiBi params — unused (args 25-29)
            ocl::core::set_kernel_arg(kernel, 25, ocl::core::ArgVal::scalar(&max_bias))?;
            ocl::core::set_kernel_arg(kernel, 26, ocl::core::ArgVal::scalar(&m0))?;
            ocl::core::set_kernel_arg(kernel, 27, ocl::core::ArgVal::scalar(&m1))?;
            ocl::core::set_kernel_arg(kernel, 28, ocl::core::ArgVal::scalar(&n_head_log2))?;
            ocl::core::set_kernel_arg(kernel, 29, ocl::core::ArgVal::scalar(&logit_softcap))?;
            // n_head_kv (arg 30)
            ocl::core::set_kernel_arg(kernel, 30, ocl::core::ArgVal::scalar(&n_head_kv_arg))?;
            // mask = NULL (args 31-37)
            ocl::core::set_kernel_arg(kernel, 31, ocl::core::ArgVal::mem_null())?;
            ocl::core::set_kernel_arg(kernel, 32, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 33, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 34, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 35, ocl::core::ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 36, ocl::core::ArgVal::scalar(&zero_i32))?;
            ocl::core::set_kernel_arg(kernel, 37, ocl::core::ArgVal::scalar(&zero_i32))?;
            // sinks = NULL (args 38-39)
            ocl::core::set_kernel_arg(kernel, 38, ocl::core::ArgVal::mem_null())?;
            ocl::core::set_kernel_arg(kernel, 39, ocl::core::ArgVal::scalar(&zero_u64))?;

            // Work size: [ceil(n_q/BLOCK_M) * BLOCK_M, n_heads_q * batch_size, 1]
            const BLOCK_M: usize = 64;
            let n_groups_q = seq_len.div_ceil(BLOCK_M);
            let global_work_size: [usize; 3] = [n_groups_q * BLOCK_M, n_heads_q * batch_size, 1];
            let local_work_size: [usize; 3] = [BLOCK_M, 1, 1];

            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                2,
                None,
                &global_work_size,
                Some(local_work_size),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }

        Ok(true)
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
        // On UMA devices (Adreno): CL_MEM_ALLOC_HOST_PTR for zero-copy shared memory.
        // On discrete GPUs (NVIDIA): device-only buffers for correct behavior.
        let memory = crate::backend::opencl::memory::OpenCLMemory::new(
            self.context.clone(),
            self.queue.clone(),
            self.use_zero_copy,
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
            // GPU→GPU: enqueue_copy_buffer (in-order queue ensures correctness)
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

    fn write_buffer(&self, t: &mut Tensor, src: &[u8]) -> Result<()> {
        assert_eq!(
            src.len(),
            t.size(),
            "write_buffer: size mismatch ({} vs {})",
            src.len(),
            t.size()
        );
        if let Ok(dst_mem) = get_cl_mem(t.buffer().as_ref()) {
            // GPU buffer: blocking write to ensure src data is consumed before return.
            // For small transfers (e.g., 4-byte token id), overhead is negligible.
            unsafe {
                ocl::core::enqueue_write_buffer(
                    &self.queue,
                    dst_mem,
                    true,
                    0,
                    src,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        } else {
            // Mapped / host-ptr buffer: direct memcpy
            let dst_ptr = t.as_mut_ptr();
            if dst_ptr.is_null() {
                anyhow::bail!("write_buffer: null pointer in destination tensor");
            }
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr, src.len());
            }
        }
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        ocl::core::finish(&self.queue)?;
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        ocl::core::flush(&self.queue)?;
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

        // Use tiled GEMM for prefill (M > 8), GEMV for decode
        const GEMM_THRESHOLD: usize = 8;
        if m > GEMM_THRESHOLD {
            let kernels = unsafe { &*self.kernels.get() };
            if kernels.kernel_mul_mm_f32_f32.is_some() {
                return self.matmul_gemm_f32(a, b, out);
            }
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

        let kernels = unsafe { &*self.kernels.get() };
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

    fn rms_norm(
        &self,
        x: &mut Tensor,
        weight: &Tensor,
        epsilon: f32,
        add_unit: bool,
    ) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let w_buf = get_cl_mem(weight.buffer().as_ref())
            .map_err(|_| anyhow!("Weight is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_rms_norm_opt;
        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();
        let add_unit_i32: i32 = if add_unit { 1 } else { 0 };

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(w_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&epsilon))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&add_unit_i32))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::local::<f32>(&local_mem_size))?;

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

    fn rms_norm_oop(
        &self,
        x: &Tensor,
        out: &mut Tensor,
        weight: &Tensor,
        epsilon: f32,
        add_unit: bool,
    ) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let out_buf =
            get_cl_mem(out.buffer().as_ref()).map_err(|_| anyhow!("Out is not OpenCL buffer"))?;
        let w_buf = get_cl_mem(weight.buffer().as_ref())
            .map_err(|_| anyhow!("Weight is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_rms_norm_oop;
        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();
        let add_unit_i32: i32 = if add_unit { 1 } else { 0 };

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(w_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&epsilon))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&add_unit_i32))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::local::<f32>(&local_mem_size))?;

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

    fn add_rms_norm_oop(
        &self,
        x: &mut Tensor,
        residual: &Tensor,
        out: &mut Tensor,
        weight: &Tensor,
        epsilon: f32,
        add_unit: bool,
    ) -> Result<()> {
        let dims = x.shape().dims();
        let dim = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let res_buf = get_cl_mem(residual.buffer().as_ref())
            .map_err(|_| anyhow!("Residual is not OpenCL buffer"))?;
        let out_buf =
            get_cl_mem(out.buffer().as_ref()).map_err(|_| anyhow!("Out is not OpenCL buffer"))?;
        let w_buf = get_cl_mem(weight.buffer().as_ref())
            .map_err(|_| anyhow!("Weight is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_add_rms_norm_oop;
        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();
        let add_unit_i32: i32 = if add_unit { 1 } else { 0 };

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(res_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::mem(w_buf))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&epsilon))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&add_unit_i32))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::local::<f32>(&local_mem_size))?;

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

        let kernels = unsafe { &*self.kernels.get() };
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
                let kernels = unsafe { &*self.kernels.get() };
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

                let kernels = unsafe { &*self.kernels.get() };
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

    fn kv_scatter_f32_to_f16(
        &self,
        k_src: &Tensor,
        v_src: &Tensor,
        k_dst: &mut Tensor,
        v_dst: &mut Tensor,
        head_dim: usize,
        capacity: usize,
        write_pos: usize,
    ) -> Result<()> {
        let k_src_mem = get_cl_mem(k_src.buffer().as_ref())?;
        let v_src_mem = get_cl_mem(v_src.buffer().as_ref())?;
        let k_dst_mem = get_cl_mem(k_dst.buffer().as_ref())?;
        let v_dst_mem = get_cl_mem(v_dst.buffer().as_ref())?;

        let n_elems: usize = k_src.shape().dims().iter().product(); // kv_heads * head_dim

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_kv_scatter_f32_to_f16;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(k_src_mem))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(v_src_mem))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(k_dst_mem))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::mem(v_dst_mem))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&(capacity as i32)))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&(write_pos as i32)))?;

            let gws: [usize; 3] = [n_elems.div_ceil(64) * 64, 1, 1];
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
        scores_out: Option<&mut [f32]>,
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

        let kernels = unsafe { &*self.kernels.get() };

        // Select kernel based on KV cache dtype
        let kernel = match k_cache.dtype() {
            DType::F16 => &kernels.kernel_attn_gen_half,
            _ => &kernels.kernel_attn_gen,
        };

        // GPU score accumulator path: when active, use persistent GPU score buffer
        // and skip CPU readback. Scores are reduced on-device and read only at eviction.
        // SAFETY: single-threaded access (same as kernels UnsafeCell pattern)
        let gpu_acc = unsafe { &*self.gpu_score_acc.get() };
        let use_gpu_acc = gpu_acc.as_ref().is_some_and(|acc| acc.is_active());

        let (write_scores, score_stride_val) = if use_gpu_acc {
            let acc = gpu_acc.as_ref().unwrap();
            (1i32, acc.score_stride() as i32)
        } else {
            (
                scores_out.is_some() as i32,
                scores_out
                    .as_ref()
                    .map(|s| (s.len() / num_heads_q) as i32)
                    .unwrap_or(0),
            )
        };

        // GPU score buffer selection:
        // 1. GPU acc active -> persistent score_buf (no per-call alloc, no readback)
        // 2. scores_out requested -> per-call GPU alloc + CPU readback
        // 3. Neither -> dummy 1-element buffer
        let score_buf = if use_gpu_acc {
            None // using gpu_acc's persistent buffer
        } else if scores_out.is_some() {
            Some(unsafe {
                ocl::core::create_buffer::<_, f32>(
                    self.context.as_core(),
                    ocl::core::MEM_READ_WRITE | ocl::core::MEM_ALLOC_HOST_PTR,
                    num_heads_q * score_stride_val as usize,
                    None,
                )?
            })
        } else {
            None
        };
        let s_buf = if use_gpu_acc {
            gpu_acc.as_ref().unwrap().score_buf_mem()
        } else {
            score_buf.as_ref().unwrap_or(&self.dummy_score_buf)
        };

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(q_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(k_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(v_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::mem(o_buf))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(s_buf))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&(num_heads_q as i32)))?;
            ocl::core::set_kernel_arg(
                kernel,
                7,
                ocl::core::ArgVal::scalar(&(num_heads_kv as i32)),
            )?;
            ocl::core::set_kernel_arg(
                kernel,
                8,
                ocl::core::ArgVal::scalar(&(cache_seq_len as i32)),
            )?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&scale))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&kv_pos_stride))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&kv_head_stride))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&write_scores))?;
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&score_stride_val))?;
            ocl::core::set_kernel_arg(
                kernel,
                14,
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

        // Post-kernel score handling:
        // GPU acc path: run reduce_layer kernel (no CPU readback)
        // Legacy path: blocking GPU->CPU readback of scores
        if use_gpu_acc {
            // SAFETY: single-threaded access
            let acc = unsafe { &*self.gpu_score_acc.get() };
            if let Some(acc) = acc.as_ref() {
                acc.reduce_layer(self.queue.as_core(), cache_seq_len)?;
            }
        } else if let (Some(scores), Some(buf)) = (scores_out, &score_buf) {
            unsafe {
                ocl::core::enqueue_read_buffer(
                    &self.queue,
                    buf,
                    true, // blocking read
                    0,
                    scores,
                    None::<ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        }

        Ok(())
    }

    fn silu_mul(&self, x: &mut Tensor, y: &Tensor) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();
        let size4 = size / 4;

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let y_buf =
            get_cl_mem(y.buffer().as_ref()).map_err(|_| anyhow!("Y is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_silu_mul_simple;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(y_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(size4 as i32)))?;

            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                1,
                None,
                &[size4, 1, 1],
                None::<[usize; 3]>,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    fn gelu_tanh_mul(&self, x: &mut Tensor, y: &Tensor) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();
        let size4 = size / 4;

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let y_buf =
            get_cl_mem(y.buffer().as_ref()).map_err(|_| anyhow!("Y is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_gelu_tanh_mul;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(y_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(size4 as i32)))?;

            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                1,
                None,
                &[size4, 1, 1],
                None::<[usize; 3]>,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    fn add_assign(&self, x: &mut Tensor, y: &Tensor) -> Result<()> {
        let size = x.shape().dims().iter().product::<usize>();
        let size4 = size / 4;

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let y_buf =
            get_cl_mem(y.buffer().as_ref()).map_err(|_| anyhow!("Y is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_add_assign_simple;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(y_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(size4 as i32)))?;

            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                1,
                None,
                &[size4, 1, 1],
                None::<[usize; 3]>,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    fn add_row_bias(&self, x: &mut Tensor, bias: &Tensor) -> Result<()> {
        let x_dims = x.shape().dims();
        let dim = x_dims[x_dims.len() - 1];
        let total: usize = x_dims.iter().product();

        let x_buf =
            get_cl_mem(x.buffer().as_ref()).map_err(|_| anyhow!("X is not OpenCL buffer"))?;
        let b_buf =
            get_cl_mem(bias.buffer().as_ref()).map_err(|_| anyhow!("Bias is not OpenCL buffer"))?;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = &kernels.kernel_add_row_bias;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(b_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&(total as i32)))?;

            let gws = total.div_ceil(64) * 64;
            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                1,
                None,
                &[gws, 1, 1],
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

        let kernels = unsafe { &*self.kernels.get() };
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

        let kernels = unsafe { &*self.kernels.get() };
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
            DType::F16 => (k * 2) as u64,
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
        let ne10 = (indices.size() / 4) as i32; // u32 element count, not byte size
        let nb10 = 4u64;
        let nb11 = nb10 * ne10 as u64;
        let nb12 = nb11;
        let nb1 = (k * 4) as u64;
        let nb2 = nb1 * ne10 as u64;
        let nb3 = nb2;

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = match src.dtype() {
            DType::Q4_0 => &kernels.kernel_get_rows_q4_0,
            DType::F32 => &kernels.kernel_get_rows_f32,
            DType::F16 => &kernels.kernel_get_rows_f16,
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
            let num_indices = indices.size() / 4; // u32 element count, not byte size
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

// ── KIVI Q2 dispatch functions (OpenCLBackend-specific, not part of Backend trait) ──
impl OpenCLBackend {
    /// Dequantize Q2 value blocks on GPU (per-token layout).
    /// `q2_buf`: raw Q2 block data on GPU (uchar buffer)
    /// `attn_v`: F32 attention V buffer on GPU [max_seq, kv_heads, head_dim]
    #[allow(clippy::too_many_arguments)]
    pub fn kivi_dequantize_value_q2(
        &self,
        q2_buf: &Tensor,
        attn_v: &mut Tensor,
        kv_heads: usize,
        head_dim: usize,
        flush_tokens: usize,
        tok_base: usize,
        block_offset: usize,
    ) -> Result<()> {
        let kernels = unsafe { &*self.kernels.get() };
        let kernel = kernels
            .kernel_kivi_deq_value_q2
            .as_ref()
            .ok_or_else(|| anyhow!("KIVI Q2 kernel not available"))?;

        let q2_mem = get_cl_mem(q2_buf.buffer().as_ref())?;
        let attn_v_mem = get_cl_mem(attn_v.buffer().as_ref())?;

        let total_blocks = kv_heads * flush_tokens * (head_dim / 32);
        let kv_heads_i = kv_heads as i32;
        let head_dim_i = head_dim as i32;
        let flush_tokens_i = flush_tokens as i32;
        let tok_base_i = tok_base as i32;
        let block_offset_i = block_offset as i32;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(q2_mem))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(attn_v_mem))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&kv_heads_i))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&head_dim_i))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&flush_tokens_i))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&tok_base_i))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&block_offset_i))?;

            let global_work_size: [usize; 3] = [total_blocks, 1, 1];
            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                3,
                None,
                &global_work_size,
                None::<[usize; 3]>,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    /// Dequantize Q2 key blocks on GPU (per-channel scatter layout).
    /// `q2_buf`: raw Q2 block data on GPU (uchar buffer)
    /// `attn_k`: F32 attention K buffer on GPU [max_seq, kv_heads, head_dim]
    #[allow(clippy::too_many_arguments)]
    pub fn kivi_dequantize_key_q2(
        &self,
        q2_buf: &Tensor,
        attn_k: &mut Tensor,
        kv_heads: usize,
        head_dim: usize,
        groups_per_flush: usize,
        tok_base: usize,
        block_offset: usize,
    ) -> Result<()> {
        let kernels = unsafe { &*self.kernels.get() };
        let kernel = kernels
            .kernel_kivi_deq_key_q2
            .as_ref()
            .ok_or_else(|| anyhow!("KIVI Q2 kernel not available"))?;

        let q2_mem = get_cl_mem(q2_buf.buffer().as_ref())?;
        let attn_k_mem = get_cl_mem(attn_k.buffer().as_ref())?;

        let total_blocks = kv_heads * groups_per_flush * head_dim;
        let kv_heads_i = kv_heads as i32;
        let head_dim_i = head_dim as i32;
        let groups_per_flush_i = groups_per_flush as i32;
        let tok_base_i = tok_base as i32;
        let block_offset_i = block_offset as i32;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(q2_mem))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(attn_k_mem))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&kv_heads_i))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&head_dim_i))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&groups_per_flush_i))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&tok_base_i))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&block_offset_i))?;

            let global_work_size: [usize; 3] = [total_blocks, 1, 1];
            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                3,
                None,
                &global_work_size,
                None::<[usize; 3]>,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    /// Scatter residual F32 buffer [kv_heads, res_cap, head_dim] into SeqMajor
    /// attention buffer [max_seq, kv_heads, head_dim].
    #[allow(clippy::too_many_arguments)]
    pub fn kivi_scatter_residual(
        &self,
        residual: &Tensor,
        attn: &mut Tensor,
        kv_heads: usize,
        res_cap: usize,
        head_dim: usize,
        res_pos: usize,
        tok_base: usize,
    ) -> Result<()> {
        let kernels = unsafe { &*self.kernels.get() };
        let kernel = kernels
            .kernel_kivi_scatter_residual
            .as_ref()
            .ok_or_else(|| anyhow!("KIVI Q2 kernel not available"))?;

        let residual_mem = get_cl_mem(residual.buffer().as_ref())?;
        let attn_mem = get_cl_mem(attn.buffer().as_ref())?;

        let total = kv_heads * res_pos * head_dim;
        let kv_heads_i = kv_heads as i32;
        let res_cap_i = res_cap as i32;
        let head_dim_i = head_dim as i32;
        let res_pos_i = res_pos as i32;
        let tok_base_i = tok_base as i32;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(residual_mem))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(attn_mem))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&kv_heads_i))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&res_cap_i))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&head_dim_i))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&res_pos_i))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&tok_base_i))?;

            let global_work_size: [usize; 3] = [total, 1, 1];
            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                3,
                None,
                &global_work_size,
                None::<[usize; 3]>,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    /// Gather new K/V tokens from SeqMajor input [seq_len, kv_heads, head_dim]
    /// into head-first residual buffer [kv_heads, res_cap, head_dim].
    #[allow(clippy::too_many_arguments)]
    pub fn kivi_gather_update(
        &self,
        input: &Tensor,
        residual: &mut Tensor,
        kv_heads: usize,
        res_cap: usize,
        head_dim: usize,
        seq_len: usize,
        res_pos: usize,
    ) -> Result<()> {
        let kernels = unsafe { &*self.kernels.get() };
        let kernel = kernels
            .kernel_kivi_gather_update
            .as_ref()
            .ok_or_else(|| anyhow!("KIVI Q2 kernel not available"))?;

        let input_mem = get_cl_mem(input.buffer().as_ref())?;
        let residual_mem = get_cl_mem(residual.buffer().as_ref())?;

        let total = seq_len * kv_heads * head_dim;
        let kv_heads_i = kv_heads as i32;
        let res_cap_i = res_cap as i32;
        let head_dim_i = head_dim as i32;
        let seq_len_i = seq_len as i32;
        let res_pos_i = res_pos as i32;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(input_mem))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(residual_mem))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::scalar(&kv_heads_i))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&res_cap_i))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&head_dim_i))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&seq_len_i))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::scalar(&res_pos_i))?;

            let global_work_size: [usize; 3] = [total, 1, 1];
            ocl::core::enqueue_kernel(
                &self.queue,
                kernel,
                3,
                None,
                &global_work_size,
                None::<[usize; 3]>,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
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
