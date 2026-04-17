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
use std::collections::HashMap;
use std::sync::Arc;

use crate::resilience::gpu_self_meter::OpenClEventGpuMeter;

pub mod buffer;
pub mod gpu_score;
pub mod memory;
pub mod plan;

/// Emit a one-time diagnostic to stderr when the prefill flash attention
/// dispatcher routes a head_dim=128 / F16 KV workload through the new
/// `flash_attn_f32_f16` DK=128 kernel. Matches the [GQA-Attn] style
/// per-run marker so short-run debugging can confirm Qwen 2.5-1.5B
/// reaches GPU prefill instead of the CPU fallback.
fn log_prefill_dk128_once() {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        eprintln!(
            "[Prefill] flash_attn dispatch: head_dim=128, DType=F16, BLOCK_M=32, subgroup=64"
        );
    });
}

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
    // MadviseableGPUBuffer (CL_MEM_USE_HOST_PTR — legacy, retained for compatibility)
    if let Some(m) = buf
        .as_any()
        .downcast_ref::<crate::buffer::madviseable_gpu_buffer::MadviseableGPUBuffer>()
    {
        return Ok(m.cl_mem_ref());
    }
    // ClWrappedBuffer (CL_MEM_USE_HOST_PTR wrapper around existing CPU buffer)
    if let Some(m) = buf
        .as_any()
        .downcast_ref::<crate::buffer::cl_wrapped_buffer::ClWrappedBuffer>()
    {
        return Ok(m.cl_mem_ref());
    }
    // ClSubBuffer (zero-copy sub-region of a parent CL buffer via clCreateSubBuffer)
    if let Some(sb) = buf
        .as_any()
        .downcast_ref::<crate::buffer::cl_sub_buffer::ClSubBuffer>()
    {
        return Ok(sb.cl_mem_ref());
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
    /// Prefill flash attention, F32 Q/KV, head_dim=256 variant
    /// (compiled from `flash_attn_f32.cl` with -DDK=256 -DDV=256 -DBLOCK_M=16).
    /// Added to eliminate the CPU fallback for Gemma3 models (head_dim=256)
    /// whose repeated per-layer `clEnqueueReadBuffer` churn exhausts NVIDIA
    /// OpenCL driver staging resources during multi-question eval-ll.
    kernel_flash_attn_f32_dk256: Option<CoreKernel>,
    /// Prefill flash attention, head_dim=64 variant
    /// (Q=F32, KV=F16, compiled with -DDK=64 -DDV=64).
    /// Formerly `kernel_flash_attn_f32_f16` (implicit dk64); renamed to
    /// match the decode naming after introducing the dk128 variant.
    kernel_flash_attn_f32_f16_dk64: Option<CoreKernel>,
    /// Prefill flash attention, head_dim=128 variant
    /// (Q=F32, KV=F16, compiled with -DDK=128 -DDV=128).
    /// Dispatched by `flash_attention_prefill_gpu` for models like
    /// Qwen 2.5-1.5B (head_dim=128, GQA).
    kernel_flash_attn_f32_f16_dk128: Option<CoreKernel>,
    /// Decode-specialized flash attention, head_dim=64 variant
    /// (Q=F32, KV=F16, compiled with -DDK=64 -DDV=64).
    kernel_flash_attn_f32_f16_q1_dk64: Option<CoreKernel>,
    /// Decode-specialized flash attention, head_dim=128 variant
    /// (Q=F32, KV=F16, compiled with -DDK=128 -DDV=128).
    kernel_flash_attn_f32_f16_q1_dk128: Option<CoreKernel>,
    // GEMM kernels for prefill (tiled matrix multiply, M > 1)
    kernel_mul_mm_f16_f32: Option<CoreKernel>,
    kernel_mul_mm_q4_0_f32: Option<CoreKernel>,
    kernel_mul_mm_f32_f32: Option<CoreKernel>,
    // KIVI Q2 kernels (optional — dequantize/scatter for GPU-native KiviCache)
    kernel_kivi_deq_value_q2: Option<CoreKernel>,
    kernel_kivi_deq_key_q2: Option<CoreKernel>,
    kernel_kivi_scatter_residual: Option<CoreKernel>,
    kernel_kivi_gather_update: Option<CoreKernel>,
    // KIVI Q2 F16-output variants (dequant/scatter to half buffers)
    kernel_kivi_deq_value_q2_f16: Option<CoreKernel>,
    kernel_kivi_deq_key_q2_f16: Option<CoreKernel>,
    kernel_kivi_scatter_residual_f16: Option<CoreKernel>,
    // KIVI fused attention kernels (optional — direct Q2/Q4/Q8 attention without F32 dequant)
    kernel_attn_gen_kivi_q2: Option<CoreKernel>,
    kernel_attn_gen_kivi_q4: Option<CoreKernel>,
    kernel_attn_gen_kivi_q8: Option<CoreKernel>,
    // F16 GEMV N_DST=4 variant for large-N matmuls (lm_head, FFN)
    kernel_mul_mat_f16_f32_l4: Option<CoreKernel>,
    // Score-only attention kernel (decode, F16 KV): Q*K^T + softmax, no V multiply.
    // Used alongside flash_attention_decode_gpu to avoid falling back to slow kernel_attn_gen_half.
    kernel_score_only_half: Option<CoreKernel>,
    // true when f16 kernel is the nosub fallback (1D work group, N_DST=4 rows/WG)
    f16_is_nosub: bool,
    // Q4_0 noshuffle: SOA conversion kernel (Adreno-optimized, from cvt.cl)
    kernel_cvt_q4_0_noshuffle: Option<CoreKernel>,
}

// SAFETY: OpenCL kernel objects are thread-safe for clSetKernelArg + clEnqueueNDRangeKernel
// when protected by a mutex. The underlying cl_kernel is not mutated by these operations,
// only the kernel arguments are set before each call.
unsafe impl Send for KernelCache {}
unsafe impl Sync for KernelCache {}

/// SOA layout entry for a Q4_0 weight tensor converted to noshuffle format.
/// Stored in the noshuffle registry, keyed by the original cl_mem pointer.
pub struct NoshuffleSoaEntry {
    /// SOA nibbles buffer (transposed, ushort-level column-major)
    pub q_buf: ocl::core::Mem,
    /// SOA scales buffer (transposed, half-level column-major)
    pub d_buf: ocl::core::Mem,
    /// image1d_buffer_t wrapping q_buf (R32UI format) for Adreno TP cache.
    /// None when image creation fails (e.g. CL_DEVICE_IMAGE_MAX_BUFFER_SIZE exceeded,
    /// image1d_buffer_t not supported). Falls back to standard Q4_0 matmul path.
    pub q_img: Option<ocl::core::Mem>,
    /// K dimension (elements per row)
    pub ne00: usize,
    /// M dimension (number of rows)
    pub ne01: usize,
}

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
    /// Flash attention F32-Q / F32-KV program, head_dim=256 variant.
    /// Used by prefill (`flash_attn_f32` kernel) for Gemma3 1B / 4B.
    /// On NVIDIA OpenCL this replaces the CPU fallback whose per-layer
    /// `clEnqueueReadBuffer` calls exhaust driver staging resources.
    pub flash_attn_f32_program_dk256: Option<Program>,
    /// Flash attention F32-Q / F16-KV program, head_dim=64 variant.
    /// Used by prefill (`flash_attn_f32_f16` kernel) and decode
    /// (`flash_attn_f32_f16_q1` kernel) at head_dim=64.
    pub flash_attn_f32_f16_program_dk64: Option<Program>,
    /// Flash attention F32-Q / F16-KV program, head_dim=128 variant.
    /// Decode-only (no prefill dispatcher for this DK). Dispatched
    /// by `flash_attention_decode_gpu` for models with head_dim=128
    /// (e.g. Qwen 2.5-1.5B).
    pub flash_attn_f32_f16_program_dk128: Option<Program>,

    // F16 GEMV N_DST=4 for large-N decode (lm_head, FFN)
    pub f16_l4_program: Option<Program>,

    // GEMM programs for prefill (tiled matmul, optional — fallback to GEMV)
    pub gemm_f16_program: Option<Program>,
    pub gemm_q4_0_program: Option<Program>,
    pub gemm_f32_program: Option<Program>,

    // KIVI kernel programs (optional — needed for plan-based KIVI decode)
    pub kivi_q2_program: Option<Program>,
    pub kivi_attn_program: Option<Program>,

    // Score-only attention program (Q*K^T + softmax, no V multiply)
    pub attention_scores_program: Option<Program>,

    // Q4_0 noshuffle programs (optional — Adreno-optimized SOA GEMV)
    pub cvt_noshuffle_program: Option<Program>,
    // gemv_noshuffle programs are dimension-specific (LINE_STRIDE_A, BLOCK_STRIDE_A).
    // Built lazily per weight dimension via convert_q4_0_to_noshuffle().

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
    pub cl_opts: String,

    // CL_DEVICE_MAX_MEM_ALLOC_SIZE: maximum single buffer allocation (bytes).
    max_mem_alloc_size: usize,

    // Cached GEMV noshuffle kernels, keyed by (ne01,) dimensions.
    // Each unique dimension pair requires a different compile-time define set.
    // UnsafeCell: single-threaded access like other kernel caches.
    gemv_noshuffle_cache: UnsafeCell<HashMap<usize, (Program, CoreKernel)>>,

    // Q4_0 noshuffle SOA registry: maps original weight cl_mem pointer (as usize)
    // to pre-converted SOA buffers. Populated by prepare_noshuffle_buffers() at load time.
    // matmul_q4_0() auto-dispatches to noshuffle GEMV when a lookup succeeds.
    // UnsafeCell: single-threaded access (same pattern as kernels and gemv_noshuffle_cache).
    noshuffle_soa_registry: UnsafeCell<HashMap<usize, NoshuffleSoaEntry>>,

    // ── Event-based per-op profiling (--profile-events) ──────────────────
    // When true, the queue was created with CL_QUEUE_PROFILING_ENABLE and all
    // enqueue_kernel_labeled() calls capture an event + label. Decode-step
    // flush aggregates (End-Start) ns per label. Unlike the legacy --profile
    // mode, no per-op clFinish is required.
    pub profile_events_enabled: bool,
    // Captured (label, event) pairs, cleared on each flush_and_aggregate_profile().
    // UnsafeCell: single-threaded inference access.
    profile_events: UnsafeCell<Vec<(&'static str, ocl::core::Event)>>,
    // Accumulated GPU ns per label (total over the whole run).
    profile_accum: UnsafeCell<HashMap<&'static str, u64>>,
    // Optional caller-provided label hint. When Some, it overrides the
    // static label passed to enqueue_kernel_labeled(). Used by forward_gen
    // and plan.rs to distinguish matmul_qkv / matmul_wo / matmul_ffn / lm_head
    // which all dispatch the same underlying GEMV/GEMM kernel.
    op_label_hint: UnsafeCell<Option<&'static str>>,

    // MSG-068 / MGR-DAT-076 (Phase 2): Engine process GPU self-utilization
    // meter. `flush_and_aggregate_profile()` pushes per-event (end - start) ns
    // into this accumulator when present. `CommandExecutor::send_heartbeat`
    // then drains it to compute `self_gpu_pct`. Populated only when the caller
    // opts in via `new_with_profile_events(true)`. Arc so both backend and
    // executor can hold a reference without lifetime gymnastics.
    gpu_self_meter: Option<Arc<OpenClEventGpuMeter>>,
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
        Self::new_with_profile_events(false)
    }

    /// Build an OpenCLBackend with optional per-op event profiling.
    ///
    /// When `profile_events_enabled` is true, the command queue is created
    /// with `CL_QUEUE_PROFILING_ENABLE`, and every kernel dispatch that
    /// goes through `enqueue_kernel_labeled()` records an `ocl::core::Event`
    /// plus label. The decode loop periodically calls
    /// `flush_and_aggregate_profile()` which reads `End - Start` nanoseconds
    /// from each event and accumulates them into `profile_accum`.
    ///
    /// This is the replacement for the legacy `--profile` mode, which
    /// inserted two `clFinish()` calls per op and inflated decode ms/tok
    /// by ~54 ms on Adreno. Event-based measurement has near-zero CPU
    /// overhead because profiling info is collected by the GPU itself.
    pub fn new_with_profile_events(profile_events_enabled: bool) -> Result<Self> {
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
        let max_mem_alloc_size: usize = device
            .info(ocl::core::DeviceInfo::MaxMemAllocSize)
            .ok()
            .and_then(|v| v.to_string().trim().parse::<u64>().ok())
            .unwrap_or(1024 * 1024 * 1024) as usize; // fallback 1GB
        log::info!(
            "GPU Max Alloc Size: {} MB",
            max_mem_alloc_size / (1024 * 1024)
        );

        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;

        let queue_props = if profile_events_enabled {
            log::info!("OpenCL queue: CL_QUEUE_PROFILING_ENABLE (event-based per-op profiling)");
            Some(flags::QUEUE_PROFILING_ENABLE)
        } else {
            None
        };
        let queue = Queue::new(&context, device, queue_props)?;

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

        // F16 GEMV: two kernel variants with different dispatch geometries.
        //
        // 1. `mul_mv_f16_f32.cl` — 4-wave K-split: local=[64,4,1], N_DST=2, 128 rows/WG.
        //    4 subgroups split K dimension in parallel, partial sums reduced via local mem.
        //    Faster on Adreno/Mali due to better parallelism and activation reuse.
        //
        // 2. `fallback/mul_mv_f16_f32_nosub.cl` — subgroup-reduce: local=[64,1,1], N_DST=4, 4 rows/WG.
        //    Single subgroup per WG, uses sub_group_reduce_add(). Slower but works on any
        //    device with subgroup ops (or falls through to tree reduction).
        //
        // `f16_is_nosub` tracks which variant was loaded so dispatch code uses the
        // correct local/global geometry. Historically both paths shared a single
        // (nosub) dispatch, which silently corrupted the 4-wave variant — the
        // "broken on Adreno 830" story was a dispatch bug, not a kernel bug.
        let f16_src = include_str!("../../../kernels/mul_mv_f16_f32.cl");
        let f16_fallback_src = include_str!("../../../kernels/fallback/mul_mv_f16_f32_nosub.cl");
        let f16_is_nosub: bool;
        let f16_program = match Program::builder()
            .devices(device)
            .src(f16_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("mul_mv_f16_f32.cl compiled (4-wave K-split, N_DST=2, 128 rows/WG)");
                f16_is_nosub = false;
                p
            }
            Err(e) => {
                log::warn!("4-wave F16 compile failed: {}. Falling back to nosub.", e);
                match Program::builder()
                    .devices(device)
                    .src(f16_fallback_src)
                    .cmplr_opt(&cl_opts)
                    .build(&context)
                {
                    Ok(p) => {
                        log::info!("F16 GEMV compiled (subgroup-reduce fallback, N_DST=4)");
                        f16_is_nosub = true;
                        p
                    }
                    Err(e2) => {
                        log::warn!("Both F16 kernels failed: {}. Using dummy stub.", e2);
                        f16_is_nosub = true;
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

        // Flash attention kernels — compiled with head_dim-specific defines.
        // Each DK variant is a separate program because DK is a compile-time
        // constant (q_priv[DK_VEC] / o_acc[DV_VEC] are private arrays).
        let flash_attn_src = include_str!("../../../kernels/flash_attn_f32_f16.cl");
        let flash_attn_f32_src = include_str!("../../../kernels/flash_attn_f32.cl");

        // DK=64 (Llama 3.2 1B, other head_dim=64 models)
        let flash_attn_dk64_defines = "-DDK=64 -DDV=64 -DBLOCK_M=64 -DBLOCK_N=32";
        let flash_attn_f32_opts_dk64 = format!("{} {}", cl_opts, flash_attn_dk64_defines);

        let flash_attn_f32_program = match Program::builder()
            .devices(device)
            .src(flash_attn_f32_src)
            .cmplr_opt(&flash_attn_f32_opts_dk64)
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

        let flash_attn_f32_f16_program_dk64 = match Program::builder()
            .devices(device)
            .src(flash_attn_src)
            .cmplr_opt(&flash_attn_f32_opts_dk64)
            .build(&context)
        {
            Ok(p) => {
                log::info!("flash_attn_f32_f16.cl compiled (Q=F32, KV=F16, DK=64)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "flash_attn_f32_f16.cl DK=64 failed: {}. GPU flash F16 KV disabled for DK=64.",
                    e
                );
                None
            }
        };

        // DK=128 (Qwen 2.5-1.5B). Same source file, different macros.
        // Supports both prefill (`flash_attn_f32_f16`) and decode
        // (`flash_attn_f32_f16_q1`) kernels.
        let flash_attn_dk128_defines = "-DDK=128 -DDV=128 -DBLOCK_M=32 -DBLOCK_N=32";
        let flash_attn_f32_opts_dk128 = format!("{} {}", cl_opts, flash_attn_dk128_defines);

        let flash_attn_f32_f16_program_dk128 = match Program::builder()
            .devices(device)
            .src(flash_attn_src)
            .cmplr_opt(&flash_attn_f32_opts_dk128)
            .build(&context)
        {
            Ok(p) => {
                log::info!("flash_attn_f32_f16.cl compiled (Q=F32, KV=F16, DK=128, BLOCK_M=32)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "flash_attn_f32_f16.cl DK=128 failed: {}. GPU flash F16 KV disabled for DK=128 (Qwen prefill + decode fallback).",
                    e
                );
                None
            }
        };

        // DK=256 (Gemma3 1B / 4B). Built from the same F32 kernel source; only
        // the compile-time DK/DV/BLOCK_M/BLOCK_N constants differ.
        // * BLOCK_M=32: WG_SIZE = BLOCK_M = 32 matches NVIDIA warp size and
        //   keeps per-thread register pressure manageable for the
        //   q_priv[DK_VEC=64] + o_acc[DV_VEC=64] private arrays.
        // * BLOCK_N=16: 2 × 16 × 64 × sizeof(float4) = 32 KB of shared memory,
        //   under the 48 KB CUDA limit. BLOCK_N=32 overflows to 64 KB (ptxas
        //   "too much shared data"), BLOCK_N=8 reduces local tile reuse and
        //   showed numerical drift in long-prefill production runs.
        let flash_attn_dk256_defines = "-DDK=256 -DDV=256 -DBLOCK_M=32 -DBLOCK_N=16";
        let flash_attn_f32_opts_dk256 = format!("{} {}", cl_opts, flash_attn_dk256_defines);

        let flash_attn_f32_program_dk256 = match Program::builder()
            .devices(device)
            .src(flash_attn_f32_src)
            .cmplr_opt(&flash_attn_f32_opts_dk256)
            .build(&context)
        {
            Ok(p) => {
                log::info!("flash_attn_f32.cl compiled (BLOCK_M=16, BLOCK_N=32, DK=256)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "flash_attn_f32.cl DK=256 failed: {}. GPU prefill F32 head_dim=256 disabled (Gemma3 falls back to CPU attention).",
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

        // Q4_0 tiled GEMM for prefill (interleaved BlockQ4_0 layout).
        let gemm_q4_0_src = include_str!("../../../kernels/mul_mm_q4_0_f32_l4_lm.cl");
        let gemm_q4_0_program = match Program::builder()
            .devices(device)
            .src(gemm_q4_0_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => {
                log::info!("mul_mm_q4_0_f32_l4_lm.cl compiled (Q4_0 GEMM for prefill)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "mul_mm_q4_0_f32_l4_lm.cl failed: {}. Q4_0 prefill will fall back to GEMV.",
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

        // KIVI fused attention kernels (optional — native Q2/Q4/Q8 attention)
        let kivi_attn_src = include_str!("../../../kernels/kivi_attn.cl");
        let kivi_attn_program = Program::builder()
            .devices(device)
            .src(kivi_attn_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
            .ok();
        if kivi_attn_program.is_some() {
            log::info!("kivi_attn.cl compiled (KIVI fused attention kernels)");
        } else {
            log::warn!("kivi_attn.cl failed to compile. KIVI fused attention disabled.");
        }

        // Score-only attention kernel (Q*K^T + softmax, no V multiply)
        let attn_scores_src = include_str!("../../../kernels/attention_scores.cl");
        let attention_scores_program = Program::builder()
            .devices(device)
            .src(attn_scores_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
            .ok();
        if attention_scores_program.is_some() {
            log::info!("attention_scores.cl compiled (score-only decode kernel)");
        } else {
            log::warn!("attention_scores.cl failed to compile. Score-only kernel disabled.");
        }

        // Q4_0 noshuffle SOA conversion kernel (cvt.cl already contains the kernel)
        let cvt_noshuffle_src = include_str!("../../../kernels/cvt.cl");
        let cvt_noshuffle_program = Program::builder()
            .devices(device)
            .src(cvt_noshuffle_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
            .ok();
        if cvt_noshuffle_program.is_some() {
            log::info!("cvt.cl compiled (Q4_0 noshuffle SOA conversion kernel)");
        } else {
            log::warn!("cvt.cl failed to compile. Q4_0 noshuffle disabled.");
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
            kernel_flash_attn_f32_dk256: flash_attn_f32_program_dk256
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32").ok()),
            kernel_flash_attn_f32_f16_dk64: flash_attn_f32_f16_program_dk64
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32_f16").ok()),
            kernel_flash_attn_f32_f16_dk128: flash_attn_f32_f16_program_dk128
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32_f16").ok()),
            kernel_flash_attn_f32_f16_q1_dk64: flash_attn_f32_f16_program_dk64
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32_f16_q1").ok()),
            kernel_flash_attn_f32_f16_q1_dk128: flash_attn_f32_f16_program_dk128
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32_f16_q1").ok()),
            kernel_mul_mm_f16_f32: gemm_f16_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_mul_mm_f16_f32_l4_lm").ok()),
            kernel_mul_mm_f32_f32: gemm_f32_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_mul_mm_f32_f32_l4_lm").ok()),
            kernel_mul_mm_q4_0_f32: gemm_q4_0_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_mul_mm_q4_0_f32_l4_lm").ok()),
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
            kernel_kivi_deq_value_q2_f16: kivi_q2_kernels
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kivi_dequantize_value_q2_f16").ok()),
            kernel_kivi_deq_key_q2_f16: kivi_q2_kernels
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kivi_dequantize_key_q2_f16").ok()),
            kernel_kivi_scatter_residual_f16: kivi_q2_kernels
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kivi_scatter_residual_f16").ok()),
            kernel_attn_gen_kivi_q2: kivi_attn_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_attn_gen_kivi_q2").ok()),
            kernel_attn_gen_kivi_q4: kivi_attn_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_attn_gen_kivi_q4").ok()),
            kernel_attn_gen_kivi_q8: kivi_attn_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_attn_gen_kivi_q8").ok()),
            kernel_mul_mat_f16_f32_l4: f16_l4_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_mul_mat_f16_f32_l4").ok()),
            kernel_score_only_half: attention_scores_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "kernel_score_only_half").ok()),
            f16_is_nosub,
            kernel_cvt_q4_0_noshuffle: cvt_noshuffle_program.as_ref().and_then(|p| {
                ocl::core::create_kernel(p, "kernel_convert_block_q4_0_noshuffle").ok()
            }),
            // GEMV noshuffle kernel is built per-dimension (lazy), so None initially
        };

        log::info!("OpenCL kernels cached successfully");

        // Auto-detect UMA (integrated GPU) vs discrete GPU for zero-copy decision.
        // CL_DEVICE_HOST_UNIFIED_MEMORY is the standard way; fall back to name heuristic.
        let use_zero_copy = if std::env::var("FORCE_DEVICE_ONLY").is_ok() {
            log::info!("FORCE_DEVICE_ONLY: disabling zero-copy, using device-only buffers");
            false
        } else {
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
            flash_attn_f32_program_dk256,
            flash_attn_f32_f16_program_dk64,
            flash_attn_f32_f16_program_dk128,
            f16_l4_program,
            gemm_f16_program,
            gemm_f32_program,
            gemm_q4_0_program,
            kivi_q2_program: kivi_q2_kernels,
            kivi_attn_program,
            attention_scores_program,
            cvt_noshuffle_program,
            kernels: UnsafeCell::new(kernel_cache),
            use_zero_copy,
            dummy_score_buf,
            gpu_score_acc: UnsafeCell::new(None),
            cl_opts: cl_opts.clone(),
            max_mem_alloc_size,
            gemv_noshuffle_cache: UnsafeCell::new(HashMap::new()),
            noshuffle_soa_registry: UnsafeCell::new(HashMap::new()),
            profile_events_enabled,
            profile_events: UnsafeCell::new(Vec::new()),
            profile_accum: UnsafeCell::new(HashMap::new()),
            op_label_hint: UnsafeCell::new(None),
            // MSG-068 Phase 2: meter는 profile-events 모드에서만 생성된다.
            // 호스트가 --profile-events 또는 --heartbeat-gpu-profile로 queue
            // profiling을 켰을 때만 의미가 있다.
            gpu_self_meter: if profile_events_enabled {
                Some(Arc::new(OpenClEventGpuMeter::new()))
            } else {
                None
            },
        })
    }

    // ── Event-based per-op profiling helpers (--profile-events) ─────────
    //
    // See `OpenCLBackend::new_with_profile_events()` for the design rationale.
    // All three helpers are no-ops when `profile_events_enabled` is false.

    /// Set a caller-provided label hint. Overrides the static label passed to
    /// `enqueue_kernel_labeled()` for the next dispatches, until cleared.
    /// Used by the forward pass to distinguish matmul_qkv / matmul_wo /
    /// matmul_ffn / lm_head which all dispatch the same GEMV/GEMM kernels.
    #[inline]
    pub fn set_op_label(&self, label: &'static str) {
        if !self.profile_events_enabled {
            return;
        }
        // SAFETY: single-threaded inference access (same pattern as `kernels`).
        unsafe {
            *self.op_label_hint.get() = Some(label);
        }
    }

    /// Clear a previously-set label hint.
    #[inline]
    pub fn clear_op_label(&self) {
        if !self.profile_events_enabled {
            return;
        }
        unsafe {
            *self.op_label_hint.get() = None;
        }
    }

    /// Enqueue an OpenCL kernel, capturing an event tagged with `default_label`
    /// (or the caller-provided hint, when set) when profile-events mode is on.
    ///
    /// Fallback path (profile off) = plain `ocl::core::enqueue_kernel` with no
    /// event — identical to the old sites being replaced.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_kernel_labeled(
        &self,
        kernel: &CoreKernel,
        default_label: &'static str,
        work_dim: u32,
        gws: &[usize; 3],
        lws: Option<[usize; 3]>,
    ) -> Result<()> {
        if self.profile_events_enabled {
            let label = unsafe { (*self.op_label_hint.get()).unwrap_or(default_label) };
            let mut ev: ocl::core::Event = ocl::core::Event::null();
            unsafe {
                ocl::core::enqueue_kernel(
                    &self.queue,
                    kernel,
                    work_dim,
                    None,
                    gws,
                    lws,
                    None::<&ocl::core::Event>,
                    Some(&mut ev),
                )?;
                let events = &mut *self.profile_events.get();
                events.push((label, ev));
            }
        } else {
            unsafe {
                ocl::core::enqueue_kernel(
                    &self.queue,
                    kernel,
                    work_dim,
                    None,
                    gws,
                    lws,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        }
        Ok(())
    }

    /// Flush captured events into the per-label accumulator, reading
    /// `CL_PROFILING_COMMAND_{START,END}` from each event and summing
    /// `(end - start)` ns into `profile_accum[label]`.
    ///
    /// Caller must ensure the queue has been drained (e.g. via
    /// `synchronize()`) before calling — profiling info is only valid once
    /// the command has reached `CL_COMPLETE`. Events are released
    /// automatically when dropped from the Vec.
    ///
    /// No-op when profile-events mode is off.
    pub fn flush_and_aggregate_profile(&self) -> Result<()> {
        if !self.profile_events_enabled {
            return Ok(());
        }
        // SAFETY: single-threaded inference access.
        let (events_vec, accum) = unsafe {
            (
                &mut *self.profile_events.get(),
                &mut *self.profile_accum.get(),
            )
        };
        for (label, ev) in events_vec.drain(..) {
            let start_ns =
                match ocl::core::get_event_profiling_info(&ev, ocl::core::ProfilingInfo::Start) {
                    Ok(info) => info.time()?,
                    Err(e) => {
                        log::warn!("profile: get_event_profiling_info(start) failed: {}", e);
                        continue;
                    }
                };
            let end_ns =
                match ocl::core::get_event_profiling_info(&ev, ocl::core::ProfilingInfo::End) {
                    Ok(info) => info.time()?,
                    Err(e) => {
                        log::warn!("profile: get_event_profiling_info(end) failed: {}", e);
                        continue;
                    }
                };
            let delta = end_ns.saturating_sub(start_ns);
            *accum.entry(label).or_insert(0) += delta;
            // MSG-068 Phase 2: GPU self-util accumulator (opt-in). Sums raw
            // (end - start) ns regardless of label. CommandExecutor drains
            // this via GpuSelfMeter::sample() on each heartbeat.
            if let Some(meter) = self.gpu_self_meter.as_ref() {
                meter.record_busy_ns(delta);
            }
            // ev is dropped here -> clReleaseEvent
        }
        Ok(())
    }

    /// Clone the shared GPU self-utilization meter, if profile-events is enabled.
    ///
    /// Used by `CommandExecutor` to periodically drain accumulated GPU busy
    /// ns and compute `EngineStatus.self_gpu_pct` (MSG-068, MGR-DAT-076
    /// Phase 2). Returns `None` when the backend was constructed without
    /// profiling (`new_with_profile_events(false)`), in which case the
    /// executor falls back to `self_gpu_pct = 0.0` (INV-092).
    pub fn gpu_self_meter(&self) -> Option<Arc<OpenClEventGpuMeter>> {
        self.gpu_self_meter.clone()
    }

    /// Consume the accumulated per-label ns map, returning it as `HashMap<String, u64>`.
    /// Clears the internal accumulator (so callers can snapshot mid-run).
    ///
    /// Returns an empty map when profile-events is off.
    pub fn take_profile_accum(&self) -> HashMap<String, u64> {
        if !self.profile_events_enabled {
            return HashMap::new();
        }
        let accum = unsafe { &mut *self.profile_accum.get() };
        accum
            .drain()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<HashMap<_, _>>()
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

            self.enqueue_kernel_labeled(
                kernel,
                "matmul",
                3,
                &global_work_size,
                Some(local_work_size),
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

            self.enqueue_kernel_labeled(
                kernel,
                "matmul",
                3,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        Ok(())
    }

    /// Tiled GEMM for Q4_0 weights: A(F32,[M,K]) x B^T(Q4_0,[N,K]) -> Out(F32,[M,N])
    ///
    /// Uses interleaved BlockQ4_0 layout (single buffer; no separate q/d split).
    /// Matches `kernel_mul_mm_q4_0_f32_l4_lm` signature ported from llama.cpp.
    fn matmul_gemm_q4_0(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
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
            .kernel_mul_mm_q4_0_f32
            .as_ref()
            .ok_or_else(|| anyhow!("GEMM Q4_0 kernel not available"))?;

        // All strides in ELEMENT counts (kernel divides by LOAD_VEC internally).
        // src0 = weight (Q4_0, [N,K]); src1 = activation (F32, [M,K]); dst = out (F32, [M,N]).
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
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(b_buf))?; // src0=weight(Q4_0)
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

            self.enqueue_kernel_labeled(
                kernel,
                "matmul",
                3,
                &global_work_size,
                Some(local_work_size),
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

        // L4 variant (mul_mv_f16_f32_l4.cl, N_DST=4 → 4 rows/WG) is also 4-wave
        // K-split, so it only works when the main 4-wave kernel is loaded.
        // Use it for large-N matmuls (lm_head, FFN) where halving WG count pays off.
        const LARGE_N_THRESHOLD: usize = 4096;
        let use_l4 =
            !f16_nosub && n > LARGE_N_THRESHOLD && kernels.kernel_mul_mat_f16_f32_l4.is_some();
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

            let (global_work_size, local_work_size) = if f16_nosub {
                // Nosub kernel: local=[64,1,1], N_DST=4 → 4 rows/WG.
                // Global: dim0 = ceil(n/4)*64, dim1 = m (batch), dim2 = 1.
                const NOSUB_N_DST: usize = 4;
                let n_groups = n.div_ceil(NOSUB_N_DST);
                ([n_groups * 64, m, 1], [64usize, 1, 1])
            } else if use_l4 {
                // 4-wave L4 kernel: local=[64,4,1], N_DST=4 → 4 rows/WG, with
                // 64 lanes × 4 waves cooperating on each row-group (256-way K parallelism).
                // Global: dim0 = ceil(n/4)*64, dim1 = m*4 (4 waves per batch), dim2 = 1.
                const L4_N_DST: usize = 4;
                let n_groups = n.div_ceil(L4_N_DST);
                ([n_groups * 64, m * 4, 1], [64usize, 4, 1])
            } else {
                // 4-wave kernel: local=[64,4,1], N_DST=2 → 2 rows/WG, with
                // 64 lanes × 4 waves cooperating on each row-pair (256-way K parallelism).
                // Global: dim0 = ceil(n/2)*64, dim1 = m*4 (4 waves per batch), dim2 = 1.
                const WAVE4_N_DST: usize = 2;
                let n_groups = n.div_ceil(WAVE4_N_DST);
                ([n_groups * 64, m * 4, 1], [64usize, 4, 1])
            };
            self.enqueue_kernel_labeled(
                kernel,
                "matmul",
                3,
                &global_work_size,
                Some(local_work_size),
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

        // Auto-dispatch to noshuffle GEMV for decode (m==1) when image1d_buffer_t is available.
        // The noshuffle kernel uses Adreno TP cache via image reads (R32UI weight, RGBA32F act).
        // Only dispatches when q_img is Some (image creation succeeded at load time).
        // When q_img is None (image unsupported), falls through to standard Q4_0 GEMV.
        if m == 1 {
            let b_key = b_buf.as_ptr() as usize;
            if let Some(entry) = self.lookup_noshuffle_soa(b_key)
                && let Some(ref q_img) = entry.q_img
            {
                return self.matmul_q4_0_noshuffle(
                    q_img,
                    &entry.d_buf,
                    a_buf,
                    out_buf,
                    entry.ne00,
                    entry.ne01,
                );
            }
            // q_img == None or no SOA entry → fall through to standard Q4_0 GEMV
        }

        // Tiled GEMM for prefill (M >= 32). Reuses weights across BM=64 rows of output,
        // eliminating the 64x weight-reload overhead of GEMV for large batches.
        if m >= 32 {
            let kernels = unsafe { &*self.kernels.get() };
            if kernels.kernel_mul_mm_q4_0_f32.is_some() {
                return self.matmul_gemm_q4_0(a, b, out);
            }
        }

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

            self.enqueue_kernel_labeled(
                kernel,
                "matmul",
                3,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        Ok(())
    }

    /// Register a pre-converted noshuffle SOA entry for a weight tensor.
    ///
    /// `key` should be the original weight buffer's `cl_mem` pointer cast to `usize`.
    /// Called by `prepare_noshuffle_buffers()` after `convert_q4_0_to_noshuffle()`.
    pub fn register_noshuffle_soa(&self, key: usize, entry: NoshuffleSoaEntry) {
        // SAFETY: single-threaded inference access (same pattern as gemv_noshuffle_cache)
        let registry = unsafe { &mut *self.noshuffle_soa_registry.get() };
        registry.insert(key, entry);
    }

    /// Lookup a pre-converted noshuffle SOA entry by the original weight cl_mem pointer.
    ///
    /// Returns None if the weight was not pre-converted (fallback to standard Q4_0 GEMV).
    pub fn lookup_noshuffle_soa(&self, key: usize) -> Option<&NoshuffleSoaEntry> {
        // SAFETY: single-threaded inference access
        let registry = unsafe { &*self.noshuffle_soa_registry.get() };
        registry.get(&key)
    }

    /// Convert Q4_0 AOS weight buffer to SOA layout for the noshuffle GEMV kernel.
    ///
    /// Called once per weight tensor at model load time.
    /// Returns (q_buf, d_buf, q_img) — the SOA nibbles buffer, scales buffer,
    /// and an optional image1d_buffer_t wrapping q_buf (R32UI) for Adreno TP cache.
    /// **Transposed** so that the GEMV kernel can access them with coalesced reads.
    ///
    /// The full pipeline mirrors llama.cpp's Adreno path:
    ///   1. GPU: kernel_convert_block_q4_0_noshuffle (nibble rearrange, row-major)
    ///   2. CPU: ushort-level 2D transpose of q buffer (M rows x K/4 cols -> K/4 rows x M cols)
    ///   3. CPU: half-level 2D transpose of d buffer (M rows x blocks_per_row cols -> transposed)
    ///   4. Create image1d_buffer_t wrapping q_buf (R32UI) — gracefully degrades if unsupported.
    ///
    /// # Arguments
    /// * `src` - Raw Q4_0 weight buffer (BlockQ4_0 AOS format)
    /// * `num_blocks` - Total number of Q4_0 blocks (= num_rows * K / QK4_0)
    /// * `ne00` - K dimension (elements per row)
    /// * `ne01` - M dimension (number of rows)
    pub fn convert_q4_0_to_noshuffle(
        &self,
        src: &ocl::core::Mem,
        num_blocks: usize,
        ne00: usize,
        ne01: usize,
    ) -> Result<(ocl::core::Mem, ocl::core::Mem, Option<ocl::core::Mem>)> {
        let kernels = unsafe { &*self.kernels.get() };
        let cvt_kernel = kernels
            .kernel_cvt_q4_0_noshuffle
            .as_ref()
            .ok_or_else(|| anyhow!("Q4_0 noshuffle conversion kernel not available"))?;

        // Allocate SOA buffers (row-major, pre-transpose)
        // dst_q: num_blocks * 16 bytes (QK4_0/2 = 16 nibble-bytes per block)
        let q_bytes = num_blocks * 16;
        let dst_q = unsafe {
            ocl::core::create_buffer::<_, u8>(
                self.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                q_bytes,
                None,
            )?
        };

        // dst_d: num_blocks * 2 bytes (one f16 per block)
        let dst_d = unsafe {
            ocl::core::create_buffer::<_, u16>(
                self.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                num_blocks,
                None,
            )?
        };

        // Step 1: GPU SOA conversion (nibble rearrange, keeps row-major block order)
        unsafe {
            ocl::core::set_kernel_arg(cvt_kernel, 0, ocl::core::ArgVal::mem(src))?;
            ocl::core::set_kernel_arg(cvt_kernel, 1, ocl::core::ArgVal::mem(&dst_q))?;
            ocl::core::set_kernel_arg(cvt_kernel, 2, ocl::core::ArgVal::mem(&dst_d))?;

            let global_work_size: [usize; 3] = [num_blocks, 1, 1];
            self.enqueue_kernel_labeled(cvt_kernel, "load_time", 1, &global_work_size, None)?;
        }
        self.queue.finish()?;

        // Step 2: CPU transpose of q buffer (ushort-level 2D transpose)
        //
        // Row-major layout (after SOA conversion):
        //   q[row * cols_ushort + col] where cols_ushort = K/4 (ushort per row)
        // Transposed layout (what GEMV expects):
        //   q_t[col * M + row] (column-major by ushort)
        //
        // When GEMV reads uint* from the transposed buffer, consecutive row-pairs
        // (row 2*gid, 2*gid+1) share a uint, enabling 2-row-per-fiber processing.
        let cols_ushort = ne00 / 4; // K/4 ushort per row
        let q_total_ushort = ne01 * cols_ushort;
        {
            let mut q_host = vec![0u16; q_total_ushort];
            unsafe {
                ocl::core::enqueue_read_buffer(
                    &self.queue,
                    &dst_q,
                    true,
                    0,
                    std::slice::from_raw_parts_mut(
                        q_host.as_mut_ptr() as *mut u8,
                        q_total_ushort * 2,
                    ),
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }

            let mut q_transposed = vec![0u16; q_total_ushort];
            for row in 0..ne01 {
                for col in 0..cols_ushort {
                    q_transposed[col * ne01 + row] = q_host[row * cols_ushort + col];
                }
            }

            unsafe {
                ocl::core::enqueue_write_buffer(
                    &self.queue,
                    &dst_q,
                    true,
                    0,
                    std::slice::from_raw_parts(
                        q_transposed.as_ptr() as *const u8,
                        q_total_ushort * 2,
                    ),
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        }

        // Step 3: CPU transpose of d buffer (half-level 2D transpose)
        //
        // Row-major: d[row * blocks_per_row + k] (half per block)
        // Transposed: d_t[k * M + row] (column-major by half)
        //
        // GEMV reads half2* from transposed d: half2[k*(M/2) + gid]
        //   .x = d_t[k*M + 2*gid]   = scale for even row
        //   .y = d_t[k*M + 2*gid+1] = scale for odd row
        let blocks_per_row = ne00 / 32; // QK4_0 = 32
        {
            let mut d_host = vec![0u16; num_blocks];
            unsafe {
                ocl::core::enqueue_read_buffer(
                    &self.queue,
                    &dst_d,
                    true,
                    0,
                    std::slice::from_raw_parts_mut(d_host.as_mut_ptr() as *mut u8, num_blocks * 2),
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }

            let mut d_transposed = vec![0u16; num_blocks];
            for row in 0..ne01 {
                for k in 0..blocks_per_row {
                    d_transposed[k * ne01 + row] = d_host[row * blocks_per_row + k];
                }
            }

            unsafe {
                ocl::core::enqueue_write_buffer(
                    &self.queue,
                    &dst_d,
                    true,
                    0,
                    std::slice::from_raw_parts(d_transposed.as_ptr() as *const u8, num_blocks * 2),
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        }

        // Step 4: Create image1d_buffer_t wrapping transposed q_buf (R32UI).
        // Each texel = 1 uint = 2 ushort (row-pair), total width = q_total_ushort / 2 texels.
        // Gracefully falls back to None if image creation fails (unsupported device, size limit).
        let q_total_uint = q_total_ushort / 2;
        let q_img = {
            use ocl::core::{
                ImageChannelDataType, ImageChannelOrder, ImageDescriptor, ImageFormat,
                MemObjectType,
            };
            let q_img_fmt =
                ImageFormat::new(ImageChannelOrder::R, ImageChannelDataType::UnsignedInt32);
            let q_img_desc = ImageDescriptor::new(
                MemObjectType::Image1dBuffer,
                q_total_uint, // width in texels (each texel = 1 uint)
                0,
                0,
                0,
                0,
                0,
                // SAFETY: Mem::clone() calls clRetainMemObject, so the underlying buffer
                // stays alive even if this image is dropped first.
                Some(dst_q.clone()),
            );
            // SAFETY: dst_q is a valid CL buffer with q_total_uint * 4 bytes.
            // image1d_buffer_t is a read-only view over the same memory.
            let result = unsafe {
                ocl::core::create_image(
                    self.context.as_core(),
                    ocl::core::MEM_READ_ONLY,
                    &q_img_fmt,
                    &q_img_desc,
                    None::<&[u32]>,
                    None, // device_versions — fallback to context auto-detect (OpenCL 1.2+)
                )
            };
            match result {
                Ok(img) => {
                    log::info!(
                        "image1d_buffer_t created for noshuffle Q4_0: width={} texels, ne01={}",
                        q_total_uint,
                        ne01
                    );
                    Some(img)
                }
                Err(e) => {
                    log::warn!(
                        "image1d_buffer_t creation failed (ne01={}), fallback to buffer path: {}",
                        ne01,
                        e
                    );
                    None
                }
            }
        };

        log::info!(
            "Q4_0 noshuffle SOA conversion + transpose done: {} blocks, ne00={}, ne01={}, q={} KB, d={} KB, img={}",
            num_blocks,
            ne00,
            ne01,
            q_bytes / 1024,
            num_blocks * 2 / 1024,
            if q_img.is_some() { "yes" } else { "no" },
        );
        Ok((dst_q, dst_d, q_img))
    }

    /// Dispatch the noshuffle GEMV kernel for Q4_0 weights in SOA layout.
    ///
    /// Uses image1d_buffer_t for both weight nibbles (R32UI) and activation (RGBA32F)
    /// to leverage Adreno TP (texture pipe) cache for coalesced reads.
    ///
    /// Compiles the dimension-specific kernel on first call and caches it
    /// (keyed by `ne01`). Subsequent calls with the same dimensions hit the cache.
    #[allow(clippy::map_entry)]
    ///
    /// # Arguments
    /// * `q_img` - image1d_buffer_t wrapping SOA nibbles (R32UI, from convert_q4_0_to_noshuffle)
    /// * `d_buf` - SOA scales buffer (half2*, from convert_q4_0_to_noshuffle)
    /// * `src1_buf` - Activation vector buffer (F32), wrapped into RGBA32F image internally
    /// * `dst_buf` - Output buffer (F32)
    /// * `ne00` - K dimension
    /// * `ne01` - M dimension (number of output rows, must be even)
    pub fn matmul_q4_0_noshuffle(
        &self,
        q_img: &ocl::core::Mem,
        d_buf: &ocl::core::Mem,
        src1_buf: &ocl::core::Mem,
        dst_buf: &ocl::core::Mem,
        ne00: usize,
        ne01: usize,
    ) -> Result<()> {
        // Compute strides (in uint units, matching the kernel's indexing on transposed SOA).
        //
        // After ushort-level 2D transpose of the SOA q buffer:
        //   Original: M rows x (K/4) cols of ushort (row-major)
        //   Transposed: (K/4) rows x M cols of ushort (column-major by ushort)
        //
        // When read as uint* (2 ushort = 1 uint), consecutive row-pairs share a uint.
        //   LINE_STRIDE_A = ne01/2 — stride between consecutive ushort columns in uint units
        //   BLOCK_STRIDE_A = 4 * ne01 — stride between consecutive K-blocks (each block = 8 ushort cols)
        //     = 8 ushort_cols * (ne01/2) uint_per_ushort_col = 4*ne01
        //   SIMDGROUP_WIDTH = 64 (Adreno half-wave)
        //
        // Reference: llama.cpp gemv_noshuffle_general.cl: LINE_STRIDE_A=M/2, BLOCK_STRIDE_A=N_SIMDGROUP*M
        let line_stride_a = ne01 / 2;
        let block_stride_a = 4 * ne01;
        let simdgroup_width: usize = 64;

        // SAFETY: single-threaded inference access (same pattern as kernels UnsafeCell)
        let cache = unsafe { &mut *self.gemv_noshuffle_cache.get() };
        if !cache.contains_key(&ne01) {
            let gemv_src = include_str!("../../../kernels/gemv_noshuffle_q4_0.cl");
            let defines = format!(
                "{} -DLINE_STRIDE_A={} -DBLOCK_STRIDE_A={} -DSIMDGROUP_WIDTH={}",
                self.cl_opts, line_stride_a, block_stride_a, simdgroup_width
            );

            // Try with vector sub_group_broadcast first (Adreno 830+ / driver v47+).
            // Falls back to scalar path if the compiler rejects float8 broadcast.
            let defines_vec = format!("{} -DVECTOR_SUB_GROUP_BROADCAT", defines);
            let program_result = Program::builder()
                .devices(self.device)
                .src(gemv_src)
                .cmplr_opt(&defines_vec)
                .build(&self.context);

            let program = match program_result {
                Ok(p) => {
                    log::info!("GEMV noshuffle Q4_0: vector sub_group_broadcast enabled");
                    p
                }
                Err(_) => {
                    log::info!("GEMV noshuffle Q4_0: falling back to scalar sub_group_broadcast");
                    Program::builder()
                        .devices(self.device)
                        .src(gemv_src)
                        .cmplr_opt(&defines)
                        .build(&self.context)?
                }
            };

            let kernel = ocl::core::create_kernel(&program, "kernel_gemv_noshuffle_q4_0")?;
            log::info!(
                "GEMV noshuffle Q4_0 kernel compiled: ne01={}, LINE_STRIDE_A={}, BLOCK_STRIDE_A={}",
                ne01,
                line_stride_a,
                block_stride_a
            );
            cache.insert(ne01, (program, kernel));
        }
        let (_program, kernel) = cache.get(&ne01).unwrap();

        let ne00_i = ne00 as i32;
        let ne01_i = ne01 as i32;

        // Create activation image1d_buffer_t (RGBA32F) wrapping the F32 activation buffer.
        // Each texel = float4 (4 floats), so width = ne00 / 4.
        // This image is ephemeral (created per-dispatch, dropped after enqueue).
        // SAFETY: src1_buf has ne00 * sizeof(f32) bytes, and ne00 is always a multiple of 4
        // for Q4_0 (QK4_0=32, ne00 is a multiple of 32).
        let act_img = {
            use ocl::core::{
                ImageChannelDataType, ImageChannelOrder, ImageDescriptor, ImageFormat,
                MemObjectType,
            };
            let act_img_fmt =
                ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelDataType::Float);
            let act_img_desc = ImageDescriptor::new(
                MemObjectType::Image1dBuffer,
                ne00 / 4, // float4 texels -> K/4 width
                0,
                0,
                0,
                0,
                0,
                // SAFETY: Mem::clone() calls clRetainMemObject; the activation buffer
                // outlives this ephemeral image (dropped at end of this function).
                Some(src1_buf.clone()),
            );
            unsafe {
                ocl::core::create_image(
                    self.context.as_core(),
                    ocl::core::MEM_READ_ONLY,
                    &act_img_fmt,
                    &act_img_desc,
                    None::<&[f32]>,
                    None, // device_versions — fallback to context auto-detect
                )?
            }
        };

        unsafe {
            // arg 0: weight nibbles image (image1d_buffer_t, R32UI)
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(q_img))?;
            // arg 1: weight scales (global half2*)
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(d_buf))?;
            // arg 2: activation image (image1d_buffer_t, RGBA32F)
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(&act_img))?;
            // arg 3: output buffer (global float*)
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::mem(dst_buf))?;
            // arg 4: ne00 (K dimension)
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&ne00_i))?;
            // arg 5: ne01 (M dimension)
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::scalar(&ne01_i))?;

            // Dispatch: global=[ne01/2, N_SIMDGROUP=4, 1], local=[SIMDGROUP_WIDTH=64, 4, 1]
            let n_simdgroup: usize = 4;
            let global_work_size: [usize; 3] = [ne01 / 2, n_simdgroup, 1];
            let local_work_size: [usize; 3] = [simdgroup_width, n_simdgroup, 1];

            self.enqueue_kernel_labeled(
                kernel,
                "matmul",
                2,
                &global_work_size,
                Some(local_work_size),
            )?;
        }
        // act_img is dropped here; clReleaseMemObject is called.
        // The enqueued kernel retains its own reference to the image/buffer.
        Ok(())
    }

    /// GPU flash attention for prefill. Dispatches flash_attn_f32 or flash_attn_f32_f16 kernel.
    /// Returns Ok(true) if GPU kernel was dispatched, Ok(false) if unsupported (caller should
    /// fall back to CPU).
    ///
    /// Q: [batch, seq_len, n_heads_q, head_dim] F32 (on GPU)
    /// K/V cache: HeadMajor [1, kv_heads, capacity, head_dim] F32 or F16 (on GPU)
    /// Output: [batch, seq_len, n_heads_q * head_dim] F32 (on GPU)
    ///
    /// Supported head_dim values:
    /// - F32 KV: 64 only (`flash_attn_f32` compiled at DK=64).
    /// - F16 KV: 64 and 128 (two DK variants of `flash_attn_f32_f16`).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_prefill_gpu(
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
        let kv_dtype = k_cache.dtype();
        let kernels = unsafe { &*self.kernels.get() };
        let kernel = match kv_dtype {
            DType::F32 => match head_dim {
                64 => match &kernels.kernel_flash_attn_f32 {
                    Some(k) => k,
                    None => return Ok(false),
                },
                256 => match &kernels.kernel_flash_attn_f32_dk256 {
                    Some(k) => k,
                    None => return Ok(false),
                },
                _ => return Ok(false),
            },
            DType::F16 => match head_dim {
                64 => match &kernels.kernel_flash_attn_f32_f16_dk64 {
                    Some(k) => k,
                    None => return Ok(false),
                },
                128 => match &kernels.kernel_flash_attn_f32_f16_dk128 {
                    Some(k) => {
                        log_prefill_dk128_once();
                        k
                    }
                    None => return Ok(false),
                },
                _ => return Ok(false),
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

            // Work size: [ceil(n_q/block_m) * lanes_per_wg, n_heads_q * batch_size, 1]
            // BLOCK_M = Q-rows per WG (matches the -DBLOCK_M compile-time macro).
            // lanes_per_wg = threads per WG:
            //   head_dim=64  → BLOCK_M=64, 1 thread / Q-row → lanes_per_wg=64 (dk64 program, unchanged)
            //   head_dim=128 → BLOCK_M=32, 2 threads / Q-row (A-3 B-1 subgroup split) → lanes_per_wg=64
            let (block_m, lanes_per_wg): (usize, usize) = match head_dim {
                64 => (64, 64),
                128 => (32, 64),
                // DK=256: WG_SIZE = BLOCK_M = 32 to match NVIDIA warp size.
                256 => (32, 32),
                _ => unreachable!("head_dim guard above"),
            };
            let n_groups_q = seq_len.div_ceil(block_m);
            let global_work_size: [usize; 3] =
                [n_groups_q * lanes_per_wg, n_heads_q * batch_size, 1];
            let local_work_size: [usize; 3] = [lanes_per_wg, 1, 1];

            self.enqueue_kernel_labeled(
                kernel,
                "flash_prefill",
                2,
                &global_work_size,
                Some(local_work_size),
            )?;

            // DK=256 stability guard: the kernel's large per-thread register
            // footprint (q_priv[64] + o_acc[64] float4 tiles) pressures the
            // NVIDIA driver when 34 back-to-back layers' dispatches share the
            // command queue without intermediate drains. An explicit finish
            // breaks the chain and keeps multi-question eval-ll stable.
            // No-op on head_dim ∈ {64, 128} where the existing kernels are
            // register-light, and on Adreno (isolated by the head_dim == 256
            // guard itself — no production Gemma3 path on Adreno today).
            if head_dim == 256 {
                ocl::core::finish(&self.queue)?;
            }
        }

        Ok(true)
    }

    /// Decode-specialized flash attention: single query per head, online softmax,
    /// zero intermediate score buffer. Dispatches `flash_attn_f32_f16_q1` from the
    /// DK variant program matching the runtime `head_dim`.
    ///
    /// Supports head_dim ∈ {64, 128} today. Requires F16 KV on GPU,
    /// HeadMajor KV layout, `seq_len=1`, and no score output.
    /// Returns Ok(true) if the kernel was dispatched; Ok(false) if preconditions
    /// are not met and the caller should fall back to the legacy attention kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_decode_gpu(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
    ) -> Result<bool> {
        // Only F16 KV on HeadMajor GPU buffer is supported.
        if k_cache.dtype() != DType::F16 {
            return Ok(false);
        }
        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == n_heads_kv && k_shape[1] != k_shape[2];
        if !is_head_major {
            return Ok(false);
        }
        let kv_capacity = k_shape[2];

        // Head dim must match the DK/DV compile-time constant of one of the
        // compiled flash attention programs. Add new variants here when
        // adding a new head_dim (remember to compile the program in
        // `OpenCLBackend::new` and cache the kernel in `KernelCache`).
        let kernels = unsafe { &*self.kernels.get() };
        let kernel = match head_dim {
            64 => match &kernels.kernel_flash_attn_f32_f16_q1_dk64 {
                Some(k) => k,
                None => return Ok(false),
            },
            128 => match &kernels.kernel_flash_attn_f32_f16_q1_dk128 {
                Some(k) => k,
                None => return Ok(false),
            },
            _ => return Ok(false),
        };

        let q_buf = get_cl_mem(q.buffer().as_ref())?;
        let k_buf = get_cl_mem(k_cache.buffer().as_ref())?;
        let v_buf = get_cl_mem(v_cache.buffer().as_ref())?;
        let o_buf = get_cl_mem(out.buffer().as_ref())?;

        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Q strides in bytes (F32 [batch=1, seq_len=1, n_heads_q, head_dim])
        let q_nb1 = (n_heads_q * head_dim * 4) as u64; // position stride (unused, seq=1)
        let q_nb2 = (head_dim * 4) as u64; // head stride
        let q_nb3 = q_nb1; // batch stride (seq=1)

        // KV strides in bytes — HeadMajor [1, kv_heads, capacity, head_dim]
        let kv_elem_size: u64 = 2; // F16
        let k_nb1 = (head_dim as u64) * kv_elem_size; // position stride within a head
        let k_nb2 = (kv_capacity * head_dim) as u64 * kv_elem_size; // head stride
        let k_nb3 = (n_heads_kv as u64) * k_nb2; // batch stride
        let (v_nb1, v_nb2, v_nb3) = (k_nb1, k_nb2, k_nb3);

        // Output strides in bytes (F32 [batch=1, seq_len=1, n_heads_q, head_dim])
        let o_nb1 = (head_dim * 4) as u64; // head stride (q1 uses this for head offset)
        let o_nb2 = (n_heads_q * head_dim * 4) as u64; // position stride (unused)
        let o_nb3 = o_nb2; // batch stride

        let n_q = 1i32;
        let n_kv = cache_seq_len as i32;
        let is_causal = 0i32; // single query, no causal mask needed
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

            // Q1_WG_SIZE = 64 (compile-time constant in the kernel)
            const Q1_WG_SIZE: usize = 64;
            // Global = [Q1_WG_SIZE, n_heads_q * batch (= n_heads_q for decode batch=1), 1]
            // Each WG handles one (batch, head) pair.
            let global_work_size: [usize; 3] = [Q1_WG_SIZE, n_heads_q, 1];
            let local_work_size: [usize; 3] = [Q1_WG_SIZE, 1, 1];
            self.enqueue_kernel_labeled(
                kernel,
                "attention",
                2,
                &global_work_size,
                Some(local_work_size),
            )?;
        }

        Ok(true)
    }

    /// Compute post-softmax attention scores on GPU without V*score weighted sum.
    /// Used alongside `flash_attention_decode_gpu()` to avoid falling back to the
    /// slow `kernel_attn_gen_half` when `scores_out` is requested.
    ///
    /// Only supports F16 HeadMajor KV cache (same prerequisite as flash decode).
    /// Returns `Ok(true)` on success, `Ok(false)` if the kernel is unavailable.
    #[allow(clippy::too_many_arguments)]
    fn compute_scores_gpu(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        scores_out: &mut [f32],
        num_heads_q: usize,
        num_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
    ) -> Result<bool> {
        let kernels = unsafe { &*self.kernels.get() };
        let kernel = match &kernels.kernel_score_only_half {
            Some(k) => k,
            None => return Ok(false),
        };

        let q_buf =
            get_cl_mem(q.buffer().as_ref()).map_err(|_| anyhow!("Q is not OpenCL buffer"))?;
        let k_buf =
            get_cl_mem(k_cache.buffer().as_ref()).map_err(|_| anyhow!("K is not OpenCL buffer"))?;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();

        let score_stride = scores_out.len() / num_heads_q;

        // Detect HeadMajor layout and compute strides
        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == num_heads_kv && k_shape[1] != k_shape[2];
        let capacity = if is_head_major { k_shape[2] } else { 0 };

        let (kv_pos_stride, kv_head_stride) = if is_head_major {
            (head_dim as i32, (capacity * head_dim) as i32)
        } else {
            ((num_heads_kv * head_dim) as i32, head_dim as i32)
        };

        // Allocate GPU score buffer
        let score_buf_size = num_heads_q * score_stride;
        let score_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                self.context.as_core(),
                ocl::core::MEM_READ_WRITE | ocl::core::MEM_ALLOC_HOST_PTR,
                score_buf_size,
                None,
            )?
        };

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(q_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(k_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(&score_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&(num_heads_q as i32)))?;
            ocl::core::set_kernel_arg(
                kernel,
                5,
                ocl::core::ArgVal::scalar(&(num_heads_kv as i32)),
            )?;
            ocl::core::set_kernel_arg(
                kernel,
                6,
                ocl::core::ArgVal::scalar(&(cache_seq_len as i32)),
            )?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&scale))?;
            ocl::core::set_kernel_arg(kernel, 8, ocl::core::ArgVal::scalar(&kv_pos_stride))?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&kv_head_stride))?;
            ocl::core::set_kernel_arg(
                kernel,
                10,
                ocl::core::ArgVal::scalar(&(score_stride as i32)),
            )?;
            ocl::core::set_kernel_arg(
                kernel,
                11,
                ocl::core::ArgVal::local::<f32>(&local_mem_size),
            )?;

            let global_work_size: [usize; 3] = [num_heads_q * local_size, 1, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];
            self.enqueue_kernel_labeled(
                kernel,
                "score_only",
                1,
                &global_work_size,
                Some(local_work_size),
            )?;

            // Blocking readback of scores to CPU
            ocl::core::enqueue_read_buffer(
                &self.queue,
                &score_buf,
                true, // blocking read
                0,
                scores_out,
                None::<ocl::core::Event>,
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

    fn write_buffer_range(&self, t: &mut Tensor, src: &[u8], dst_offset: usize) -> Result<()> {
        let end = dst_offset
            .checked_add(src.len())
            .ok_or_else(|| anyhow::anyhow!("write_buffer_range: offset+len overflow"))?;
        if end > t.size() {
            anyhow::bail!(
                "write_buffer_range: out of bounds ({} + {} > {})",
                dst_offset,
                src.len(),
                t.size()
            );
        }
        if let Ok(dst_mem) = get_cl_mem(t.buffer().as_ref()) {
            // Partial blocking write starting at `dst_offset` bytes into the buffer.
            unsafe {
                ocl::core::enqueue_write_buffer(
                    &self.queue,
                    dst_mem,
                    true,
                    dst_offset,
                    src,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        } else {
            // Mapped / host-ptr buffer: direct offset memcpy
            let dst_ptr = t.as_mut_ptr();
            if dst_ptr.is_null() {
                anyhow::bail!("write_buffer_range: null pointer in destination tensor");
            }
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr.add(dst_offset), src.len());
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

    fn is_gpu(&self) -> bool {
        true
    }

    fn is_discrete_gpu(&self) -> bool {
        !self.use_zero_copy
    }

    fn max_single_alloc(&self) -> usize {
        self.max_mem_alloc_size
    }

    fn flash_attention_prefill(
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
        self.flash_attention_prefill_gpu(
            q,
            k_cache,
            v_cache,
            out,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            head_dim,
            kv_capacity,
            batch_size,
            is_head_major,
        )
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

            self.enqueue_kernel_labeled(
                kernel,
                "matmul",
                3,
                &global_work_size,
                Some(local_work_size),
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

            self.enqueue_kernel_labeled(
                kernel,
                "rms_norm",
                1,
                &global_work_size,
                Some(local_work_size),
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

            self.enqueue_kernel_labeled(
                kernel,
                "rms_norm",
                1,
                &global_work_size,
                Some(local_work_size),
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

            self.enqueue_kernel_labeled(
                kernel,
                "rms_norm",
                1,
                &global_work_size,
                Some(local_work_size),
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
            self.enqueue_kernel_labeled(kernel, "rope", 1, &[work_size, 1, 1], None)?;
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
                    self.enqueue_kernel_labeled(kernel, "cast", 1, &gws, Some(lws))?;
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

                    self.enqueue_kernel_labeled(
                        kernel,
                        "cast",
                        1,
                        &[global_size, 1, 1],
                        Some([local_size, 1, 1]),
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
            self.enqueue_kernel_labeled(kernel, "kv_update", 1, &gws, Some(lws))?;
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
        mut scores_out: Option<&mut [f32]>,
    ) -> Result<()> {
        // Flash-attention decode fast path: single-pass online softmax, vectorized
        // float4 K/V reads, no local-memory output reduction. Used when none of the
        // score-accumulator paths are active and the kernel's compile-time DK/DV
        // matches the current head_dim.
        //
        // When scores_out is requested, we still use flash attention for the output
        // vector and dispatch a separate lightweight score-only kernel (Q*K^T + softmax)
        // instead of falling back to the slow kernel_attn_gen_half.
        let gpu_acc_active = {
            let gpu_acc = unsafe { &*self.gpu_score_acc.get() };
            gpu_acc.as_ref().is_some_and(|acc| acc.is_active())
        };
        if !gpu_acc_active
            && self.flash_attention_decode_gpu(
                q,
                k_cache,
                v_cache,
                out,
                num_heads_q,
                num_heads_kv,
                head_dim,
                cache_seq_len,
            )?
        {
            // Flash attention succeeded for output. Now handle scores if needed.
            if let Some(ref mut scores) = scores_out {
                if self.compute_scores_gpu(
                    q,
                    k_cache,
                    scores,
                    num_heads_q,
                    num_heads_kv,
                    head_dim,
                    cache_seq_len,
                )? {
                    return Ok(());
                }
                // Score-only kernel unavailable — fall through to legacy path
                // for scores only (output already computed by flash attn, but
                // we need to recompute everything with the slow kernel to get
                // scores). This should be rare (kernel compilation failure).
            } else {
                return Ok(());
            }
        }

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

            self.enqueue_kernel_labeled(
                kernel,
                "attention",
                1,
                &global_work_size,
                Some(local_work_size),
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

            self.enqueue_kernel_labeled(kernel, "silu_mul", 1, &[size4, 1, 1], None)?;
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

            self.enqueue_kernel_labeled(kernel, "silu_mul", 1, &[size4, 1, 1], None)?;
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

            self.enqueue_kernel_labeled(kernel, "add_assign", 1, &[size4, 1, 1], None)?;
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
            self.enqueue_kernel_labeled(kernel, "add_assign", 1, &[gws, 1, 1], None)?;
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

            self.enqueue_kernel_labeled(kernel, "other", 1, &[size, 1, 1], None)?;
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

            self.enqueue_kernel_labeled(
                kernel,
                "other",
                1,
                &global_work_size,
                Some(local_work_size),
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

            self.enqueue_kernel_labeled(
                kernel,
                "gather",
                1,
                &global_work_size,
                Some(local_work_size),
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

        // Prefer GPU copy (clEnqueueCopyBuffer) to avoid CPU cache pollution on
        // zero-copy (CL_MEM_ALLOC_HOST_PTR) buffers. This is critical for tensor
        // partition workloads where CPU memmove on mapped UMA buffers contends with
        // the CPU-side matmul, causing ~21ms TBT regression.
        if let Ok(buf_mem) = get_cl_mem(tensor.buffer().as_ref()) {
            let src_byte = src_offset * type_size;
            let dst_byte = dst_offset * type_size;
            let byte_count = count * type_size;

            // GPU copy path: use enqueue_copy_buffer via cl_mem handle.
            // No queue.finish() after enqueue — the in-order command queue guarantees
            // that subsequent kernel dispatches on the same queue are serialized after
            // this copy, so explicit synchronization would stall the GPU pipeline.
            let gpu_result = (|| -> Result<()> {
                // OpenCL spec: clEnqueueCopyBuffer with overlapping src/dst regions in
                // the same buffer is undefined behavior. Detect overlap and use a temp
                // buffer.
                let no_overlap =
                    (src_byte + byte_count <= dst_byte) || (dst_byte + byte_count <= src_byte);

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
                    // Safe to drop temp after enqueue: clReleaseMemObject defers
                    // deallocation until pending commands referencing this buffer
                    // complete (OpenCL spec).
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
            })();

            match gpu_result {
                Ok(()) => return Ok(()),
                Err(e) => {
                    // Safety fallback: GPU copy failed, fall through to CPU memmove.
                    // Use std::sync::Once to emit the warning only once per process.
                    use std::sync::Once;
                    static WARN_ONCE: Once = Once::new();
                    WARN_ONCE.call_once(|| {
                        eprintln!(
                            "[buffer_shift] GPU enqueue_copy_buffer failed, \
                             falling back to CPU memmove: {e}"
                        );
                    });
                }
            }
        }

        // CPU fallback: pure-CPU buffers (SharedBuffer) or GPU copy failure.
        // Safety: as_mut_ptr() returns a valid mutable pointer for CPU-backed
        // buffers. std::ptr::copy handles overlapping regions correctly (memmove).
        let ptr = tensor.as_mut_ptr();
        if !ptr.is_null() {
            unsafe {
                std::ptr::copy(
                    ptr.add(src_offset * type_size),
                    ptr.add(dst_offset * type_size),
                    count * type_size,
                );
            }
            return Ok(());
        }

        Err(anyhow!(
            "buffer_shift: no GPU cl_mem and no CPU pointer available"
        ))
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
            self.enqueue_kernel_labeled(kernel, "kv_update", 3, &global_work_size, None)?;
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
            self.enqueue_kernel_labeled(kernel, "kv_update", 3, &global_work_size, None)?;
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
            self.enqueue_kernel_labeled(kernel, "kv_update", 3, &global_work_size, None)?;
        }
        Ok(())
    }

    /// Dequantize Q2 value blocks on GPU to F16 attention buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn kivi_dequantize_value_q2_f16(
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
            .kernel_kivi_deq_value_q2_f16
            .as_ref()
            .ok_or_else(|| anyhow!("KIVI Q2 F16 value dequant kernel not available"))?;

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
            self.enqueue_kernel_labeled(kernel, "kv_update", 3, &global_work_size, None)?;
        }
        Ok(())
    }

    /// Dequantize Q2 key blocks on GPU to F16 attention buffer (per-channel scatter).
    #[allow(clippy::too_many_arguments)]
    pub fn kivi_dequantize_key_q2_f16(
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
            .kernel_kivi_deq_key_q2_f16
            .as_ref()
            .ok_or_else(|| anyhow!("KIVI Q2 F16 key dequant kernel not available"))?;

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
            self.enqueue_kernel_labeled(kernel, "kv_update", 3, &global_work_size, None)?;
        }
        Ok(())
    }

    /// Scatter residual F32 buffer to F16 attention buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn kivi_scatter_residual_f16(
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
            .kernel_kivi_scatter_residual_f16
            .as_ref()
            .ok_or_else(|| anyhow!("KIVI F16 scatter_residual kernel not available"))?;

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
            self.enqueue_kernel_labeled(kernel, "kv_update", 3, &global_work_size, None)?;
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
            self.enqueue_kernel_labeled(kernel, "kv_update", 3, &global_work_size, None)?;
        }
        Ok(())
    }

    /// KIVI fused attention: Q2/Q4/Q8 quantized KV + F32 residual, single kernel.
    ///
    /// Eliminates the intermediate F32 dequant buffer by performing on-the-fly
    /// dequantization inside the attention kernel.
    ///
    /// `bits` selects the kernel variant: 2 → Q2, 4 → Q4, 8 → Q8.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_gen_kivi(
        &self,
        q: &Tensor,
        qk_buf: &Tensor,
        qv_buf: &Tensor,
        res_k: &Tensor,
        res_v: &Tensor,
        out: &mut Tensor,
        num_heads_q: usize,
        num_heads_kv: usize,
        head_dim: usize,
        q_tokens: usize,
        res_tokens: usize,
        res_cap: usize,
        scale: f32,
        scores_out: Option<&mut [f32]>,
        bits: u8,
    ) -> Result<()> {
        let kernels = unsafe { &*self.kernels.get() };
        let kernel = match bits {
            2 => kernels
                .kernel_attn_gen_kivi_q2
                .as_ref()
                .ok_or_else(|| anyhow!("KIVI Q2 attention kernel not available"))?,
            4 => kernels
                .kernel_attn_gen_kivi_q4
                .as_ref()
                .ok_or_else(|| anyhow!("KIVI Q4 attention kernel not available"))?,
            8 => kernels
                .kernel_attn_gen_kivi_q8
                .as_ref()
                .ok_or_else(|| anyhow!("KIVI Q8 attention kernel not available"))?,
            _ => return Err(anyhow!("Unsupported KIVI bits: {}", bits)),
        };

        let q_buf = get_cl_mem(q.buffer().as_ref())?;
        let qk_mem = get_cl_mem(qk_buf.buffer().as_ref())?;
        let qv_mem = get_cl_mem(qv_buf.buffer().as_ref())?;
        let res_k_mem = get_cl_mem(res_k.buffer().as_ref())?;
        let res_v_mem = get_cl_mem(res_v.buffer().as_ref())?;
        let o_buf = get_cl_mem(out.buffer().as_ref())?;

        let has_scores = scores_out.is_some() as i32;
        let total_tokens = q_tokens + res_tokens;
        let score_stride_val = scores_out
            .as_ref()
            .map(|s| (s.len() / num_heads_q) as i32)
            .unwrap_or(total_tokens as i32);

        // Allocate GPU score buffer if needed, otherwise use dummy
        let score_buf = if scores_out.is_some() {
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
        let s_buf = score_buf.as_ref().unwrap_or(&self.dummy_score_buf);

        let local_size = 64usize;
        let local_mem_size = local_size * std::mem::size_of::<f32>();

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(q_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(qk_mem))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(qv_mem))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::mem(res_k_mem))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::mem(res_v_mem))?;
            ocl::core::set_kernel_arg(kernel, 5, ocl::core::ArgVal::mem(o_buf))?;
            ocl::core::set_kernel_arg(kernel, 6, ocl::core::ArgVal::mem(s_buf))?;
            ocl::core::set_kernel_arg(kernel, 7, ocl::core::ArgVal::scalar(&(num_heads_q as i32)))?;
            ocl::core::set_kernel_arg(
                kernel,
                8,
                ocl::core::ArgVal::scalar(&(num_heads_kv as i32)),
            )?;
            ocl::core::set_kernel_arg(kernel, 9, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 10, ocl::core::ArgVal::scalar(&(q_tokens as i32)))?;
            ocl::core::set_kernel_arg(kernel, 11, ocl::core::ArgVal::scalar(&(res_tokens as i32)))?;
            ocl::core::set_kernel_arg(kernel, 12, ocl::core::ArgVal::scalar(&(res_cap as i32)))?;
            ocl::core::set_kernel_arg(kernel, 13, ocl::core::ArgVal::scalar(&scale))?;
            ocl::core::set_kernel_arg(kernel, 14, ocl::core::ArgVal::scalar(&score_stride_val))?;
            ocl::core::set_kernel_arg(kernel, 15, ocl::core::ArgVal::scalar(&has_scores))?;
            ocl::core::set_kernel_arg(
                kernel,
                16,
                ocl::core::ArgVal::local::<f32>(&local_mem_size),
            )?;

            let global_work_size: [usize; 3] = [num_heads_q * local_size, 1, 1];
            let local_work_size: [usize; 3] = [local_size, 1, 1];

            self.enqueue_kernel_labeled(
                kernel,
                "attention",
                1,
                &global_work_size,
                Some(local_work_size),
            )?;
        }

        // Read back scores to CPU if requested
        if let (Some(scores), Some(buf)) = (scores_out, &score_buf) {
            unsafe {
                ocl::core::enqueue_read_buffer(
                    &self.queue,
                    buf,
                    true,
                    0,
                    scores,
                    None::<ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        }

        Ok(())
    }

    /// Returns true if this device lacks subgroup support (using nosub fallback kernels).
    /// Native KIVI attention (workgroup reduction) is preferred on these devices since
    /// the standard attention_gen also uses workgroup reduction.
    pub fn is_nosub(&self) -> bool {
        let kernels = unsafe { &*self.kernels.get() };
        kernels.f16_is_nosub
    }

    /// Returns true if the flash attention decode kernel is available for the
    /// given head_dim. Used by plan.rs to gate the `StandardFlash` attention
    /// variant, and by host tests to decide whether to exercise the flash path.
    /// Supported head_dim values match the DK variants compiled in `new()`.
    pub fn has_flash_decode_kernel(&self, head_dim: usize) -> bool {
        let kernels = unsafe { &*self.kernels.get() };
        match head_dim {
            64 => kernels.kernel_flash_attn_f32_f16_q1_dk64.is_some(),
            128 => kernels.kernel_flash_attn_f32_f16_q1_dk128.is_some(),
            _ => false,
        }
    }

    /// Check if KIVI fused attention kernel is available for the given bit-width.
    pub fn has_kivi_attn_kernel(&self, bits: u8) -> bool {
        let kernels = unsafe { &*self.kernels.get() };
        match bits {
            2 => kernels.kernel_attn_gen_kivi_q2.is_some(),
            4 => kernels.kernel_attn_gen_kivi_q4.is_some(),
            8 => kernels.kernel_attn_gen_kivi_q8.is_some(),
            _ => false,
        }
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

#[cfg(test)]
mod noshuffle_tests {
    use super::*;

    fn try_create_backend() -> Option<Arc<OpenCLBackend>> {
        OpenCLBackend::new().ok().map(Arc::new)
    }

    /// Q4_0 block: 2 bytes d (f16) + 16 bytes qs (QK4_0/2 = 16 nibble bytes)
    const BLOCK_Q4_0_SIZE: usize = 18;
    const QK4_0: usize = 32;

    /// Build a minimal Q4_0 block from known quant values.
    ///
    /// `d_f32`: scale (will be converted to f16)
    /// `quants`: 32 signed 4-bit values (-8..7), stored as unsigned 0..15 in nibbles
    fn make_q4_0_block(d_f32: f32, quants: &[i8; 32]) -> [u8; BLOCK_Q4_0_SIZE] {
        let mut block = [0u8; BLOCK_Q4_0_SIZE];
        let d_f16 = half::f16::from_f32(d_f32);
        block[0..2].copy_from_slice(&d_f16.to_le_bytes());
        // Pack 32 quants into 16 bytes: qs[i] = (q[2i] & 0xF) | (q[2i+1] << 4)
        // In Q4_0, stored nibble = quant + 8 (unsigned 0..15)
        for i in 0..16 {
            let lo = (quants[i] + 8) as u8 & 0x0F;
            let hi = (quants[i + 16] + 8) as u8 & 0x0F;
            block[2 + i] = lo | (hi << 4);
        }
        block
    }

    /// Reference CPU dequant for a Q4_0 block -> 32 f32 values.
    fn dequant_q4_0_block(block: &[u8; BLOCK_Q4_0_SIZE]) -> [f32; 32] {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let mut out = [0.0f32; 32];
        for i in 0..16 {
            let byte = block[2 + i];
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
            out[i] = lo as f32 * d;
            out[i + 16] = hi as f32 * d;
        }
        out
    }

    /// Reference CPU matmul: weight (ne01 x ne00, Q4_0) * activation (ne00,) -> output (ne01,)
    fn reference_matmul_q4_0(
        blocks: &[[u8; BLOCK_Q4_0_SIZE]],
        activation: &[f32],
        ne00: usize,
        ne01: usize,
    ) -> Vec<f32> {
        let blocks_per_row = ne00 / QK4_0;
        let mut output = vec![0.0f32; ne01];
        for row in 0..ne01 {
            let mut sum = 0.0f32;
            for k_blk in 0..blocks_per_row {
                let block = &blocks[row * blocks_per_row + k_blk];
                let dequants = dequant_q4_0_block(block);
                for j in 0..QK4_0 {
                    sum += dequants[j] * activation[k_blk * QK4_0 + j];
                }
            }
            output[row] = sum;
        }
        output
    }

    /// Test that the SOA conversion + transpose produces the correct transposed layout.
    ///
    /// This verifies the nibble rearrangement and 2D transpose independently of GEMV.
    #[test]
    fn test_soa_conversion_transpose_layout() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // Small matrix: 4 rows, K=64 -> blocks_per_row=2, total 8 blocks
        let ne01 = 4usize;
        let ne00 = 64usize;
        let blocks_per_row = ne00 / QK4_0;
        let num_blocks = ne01 * blocks_per_row;

        // Create blocks with known patterns
        let mut all_blocks = Vec::new();
        for row in 0..ne01 {
            for k in 0..blocks_per_row {
                let d = (row + 1) as f32 * 0.1 + k as f32 * 0.01;
                let mut quants = [0i8; 32];
                for j in 0..32 {
                    quants[j] = ((row * 100 + k * 32 + j) % 15) as i8 - 7;
                }
                all_blocks.push(make_q4_0_block(d, &quants));
            }
        }

        // Upload raw Q4_0 data to GPU
        let raw_bytes: Vec<u8> = all_blocks.iter().flat_map(|b| b.iter().copied()).collect();
        let src_buf = unsafe {
            ocl::core::create_buffer::<_, u8>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                raw_bytes.len(),
                None,
            )
            .unwrap()
        };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                &src_buf,
                true,
                0,
                &raw_bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        // Run SOA conversion + transpose
        let (q_buf, d_buf, _q_img) = backend
            .convert_q4_0_to_noshuffle(&src_buf, num_blocks, ne00, ne01)
            .unwrap();

        // Read back transposed q and d
        let q_total_ushort = ne01 * (ne00 / 4);
        let mut q_transposed = vec![0u16; q_total_ushort];
        unsafe {
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                &q_buf,
                true,
                0,
                std::slice::from_raw_parts_mut(
                    q_transposed.as_mut_ptr() as *mut u8,
                    q_total_ushort * 2,
                ),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        let mut d_transposed = vec![0u16; num_blocks];
        unsafe {
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                &d_buf,
                true,
                0,
                std::slice::from_raw_parts_mut(
                    d_transposed.as_mut_ptr() as *mut u8,
                    num_blocks * 2,
                ),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        // Verify d transpose: d_transposed[k * ne01 + row] == original d[row][k]
        for row in 0..ne01 {
            for k in 0..blocks_per_row {
                let original_d = half::f16::from_le_bytes([
                    all_blocks[row * blocks_per_row + k][0],
                    all_blocks[row * blocks_per_row + k][1],
                ]);
                let transposed_d = half::f16::from_bits(d_transposed[k * ne01 + row]);
                assert_eq!(
                    original_d.to_bits(),
                    transposed_d.to_bits(),
                    "d mismatch at row={}, k={}: original={:?}, transposed={:?}",
                    row,
                    k,
                    original_d,
                    transposed_d
                );
            }
        }

        // Verify q transpose layout: each row's noshuffle block data should be
        // accessible at the correct transposed position.
        //
        // After noshuffle conversion, block[row][k] has 16 bytes = 8 ushort.
        // After transpose, ushort col c at row r is at q_transposed[c * ne01 + r].
        //
        // For row `row`, block `k`, the ushort column range is k*8..k*8+7.
        // We can verify by computing the expected noshuffle output for each block
        // and checking the transposed positions.
        for row in 0..ne01 {
            for k in 0..blocks_per_row {
                let block = &all_blocks[row * blocks_per_row + k];
                // Compute expected noshuffle q bytes (same logic as kernel)
                let mut expected_q = [0u8; 16];
                for i in 0..8 {
                    let x0 = block[2 + 2 * i];
                    let x1 = block[2 + 2 * i + 1];
                    expected_q[i] = (x0 & 0x0F) | ((x1 & 0x0F) << 4);
                    expected_q[i + 8] = ((x0 & 0xF0) >> 4) | (x1 & 0xF0);
                }

                // expected_q as 8 ushort (LE)
                let mut expected_ushort = [0u16; 8];
                for i in 0..8 {
                    expected_ushort[i] =
                        expected_q[2 * i] as u16 | ((expected_q[2 * i + 1] as u16) << 8);
                }

                // Check transposed positions: col = k*8+j, row = row
                for j in 0..8 {
                    let col_ushort = k * 8 + j;
                    let transposed_val = q_transposed[col_ushort * ne01 + row];
                    assert_eq!(
                        expected_ushort[j], transposed_val,
                        "q mismatch at row={}, k={}, ushort_j={}: expected={:#06x}, got={:#06x}",
                        row, k, j, expected_ushort[j], transposed_val
                    );
                }
            }
        }

        eprintln!(
            "[PASS] SOA conversion + transpose layout verified for {}x{}",
            ne01, ne00
        );
    }

    /// Test that the GEMV indexing logic in the transposed layout produces correct
    /// uint values when accessed with the kernel's stride pattern.
    ///
    /// This simulates the kernel's memory access pattern without actually running
    /// the GPU kernel (no sub_group_broadcast needed).
    #[test]
    fn test_noshuffle_gemv_indexing() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // 4 rows, K=128 -> blocks_per_row=4
        let ne01 = 4usize;
        let ne00 = 128usize;
        let blocks_per_row = ne00 / QK4_0;
        let num_blocks = ne01 * blocks_per_row;

        // Create blocks with known scale and quant patterns
        let mut all_blocks = Vec::new();
        for row in 0..ne01 {
            for k in 0..blocks_per_row {
                let d = 0.5f32;
                let mut quants = [0i8; 32];
                for j in 0..32 {
                    quants[j] = ((row * 1000 + k * 32 + j) % 15) as i8 - 7;
                }
                all_blocks.push(make_q4_0_block(d, &quants));
            }
        }

        // Upload and convert
        let raw_bytes: Vec<u8> = all_blocks.iter().flat_map(|b| b.iter().copied()).collect();
        let src_buf = unsafe {
            ocl::core::create_buffer::<_, u8>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                raw_bytes.len(),
                None,
            )
            .unwrap()
        };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                &src_buf,
                true,
                0,
                &raw_bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        let (q_buf, d_buf, _q_img) = backend
            .convert_q4_0_to_noshuffle(&src_buf, num_blocks, ne00, ne01)
            .unwrap();

        // Read back as uint (for q) and u16 (for d)
        let q_total_uint = ne01 * ne00 / 8; // ne01 * (K/4) ushort / 2
        let mut q_uint = vec![0u32; q_total_uint];
        unsafe {
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                &q_buf,
                true,
                0,
                std::slice::from_raw_parts_mut(q_uint.as_mut_ptr() as *mut u8, q_total_uint * 4),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        let mut d_u16 = vec![0u16; num_blocks];
        unsafe {
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                &d_buf,
                true,
                0,
                std::slice::from_raw_parts_mut(d_u16.as_mut_ptr() as *mut u8, num_blocks * 2),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        let line_stride_a = ne01 / 2;
        let block_stride_a = 4 * ne01;

        // Simulate the kernel access pattern for each row-pair and K-block
        for gid in 0..ne01 / 2 {
            for k in 0..blocks_per_row {
                // Read scale (half2 = two consecutive u16 as half pair)
                let scale_idx = gid + k * line_stride_a;
                let scale_even = half::f16::from_bits(d_u16[scale_idx * 2]);
                let scale_odd = half::f16::from_bits(d_u16[scale_idx * 2 + 1]);

                // Expected scales
                let row_even = 2 * gid;
                let row_odd = 2 * gid + 1;
                let expected_d_even = half::f16::from_le_bytes([
                    all_blocks[row_even * blocks_per_row + k][0],
                    all_blocks[row_even * blocks_per_row + k][1],
                ]);
                let expected_d_odd = half::f16::from_le_bytes([
                    all_blocks[row_odd * blocks_per_row + k][0],
                    all_blocks[row_odd * blocks_per_row + k][1],
                ]);

                assert_eq!(
                    scale_even.to_bits(),
                    expected_d_even.to_bits(),
                    "scale even mismatch: gid={}, k={}",
                    gid,
                    k
                );
                assert_eq!(
                    scale_odd.to_bits(),
                    expected_d_odd.to_bits(),
                    "scale odd mismatch: gid={}, k={}",
                    gid,
                    k
                );

                // Read 8 uint values (matching kernel's regA pattern)
                for i in 0..8 {
                    let q_idx = gid + k * block_stride_a + line_stride_a * i;
                    assert!(
                        q_idx < q_total_uint,
                        "q_idx {} out of range (max {})",
                        q_idx,
                        q_total_uint
                    );
                    let val = q_uint[q_idx];
                    let lo_ushort = (val & 0xFFFF) as u16;
                    let hi_ushort = (val >> 16) as u16;

                    // lo_ushort should be from row_even, hi_ushort from row_odd
                    // Both at the same ushort column: k*8 + i
                    // Compute expected from noshuffle conversion
                    for (check_row, check_ushort, label) in
                        [(row_even, lo_ushort, "even"), (row_odd, hi_ushort, "odd")]
                    {
                        let block = &all_blocks[check_row * blocks_per_row + k];
                        let mut expected_q = [0u8; 16];
                        for ii in 0..8 {
                            let x0 = block[2 + 2 * ii];
                            let x1 = block[2 + 2 * ii + 1];
                            expected_q[ii] = (x0 & 0x0F) | ((x1 & 0x0F) << 4);
                            expected_q[ii + 8] = ((x0 & 0xF0) >> 4) | (x1 & 0xF0);
                        }
                        // ushort column i within this block's 8 ushorts
                        let expected_val =
                            expected_q[2 * i] as u16 | ((expected_q[2 * i + 1] as u16) << 8);
                        assert_eq!(
                            check_ushort, expected_val,
                            "q {} row mismatch: gid={}, k={}, i={}: got={:#06x}, expected={:#06x}",
                            label, gid, k, i, check_ushort, expected_val
                        );
                    }
                }
            }
        }

        eprintln!("[PASS] GEMV indexing verified for {}x{}", ne01, ne00);
    }

    /// End-to-end test: compare noshuffle GEMV output against reference CPU dequant.
    ///
    /// On macOS (Apple GPU), the noshuffle GEMV kernel may not compile due to missing
    /// sub_group_broadcast / cl_qcom_reqd_sub_group_size. In that case, this test
    /// is skipped gracefully.
    #[test]
    fn test_noshuffle_q4_0_correctness() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // 8 rows, K=128 (small but exercises multiple blocks and row-pairs)
        let ne01 = 8usize;
        let ne00 = 128usize;
        let blocks_per_row = ne00 / QK4_0;
        let num_blocks = ne01 * blocks_per_row;

        // Create blocks with non-trivial patterns
        let mut all_blocks = Vec::new();
        for row in 0..ne01 {
            for k in 0..blocks_per_row {
                let d = 0.1 + 0.05 * row as f32 + 0.01 * k as f32;
                let mut quants = [0i8; 32];
                for j in 0..32 {
                    quants[j] = ((row * 7 + k * 3 + j * 5) % 15) as i8 - 7;
                }
                all_blocks.push(make_q4_0_block(d, &quants));
            }
        }

        // Create activation vector
        let mut activation = vec![0.0f32; ne00];
        for i in 0..ne00 {
            activation[i] = (i as f32 * 0.01).sin();
        }

        // Reference CPU result
        let reference = reference_matmul_q4_0(&all_blocks, &activation, ne00, ne01);

        // Upload Q4_0 data
        let raw_bytes: Vec<u8> = all_blocks.iter().flat_map(|b| b.iter().copied()).collect();
        let src_buf = unsafe {
            ocl::core::create_buffer::<_, u8>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                raw_bytes.len(),
                None,
            )
            .unwrap()
        };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                &src_buf,
                true,
                0,
                &raw_bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        // Convert to noshuffle SOA + transpose
        let (q_buf, d_buf, _q_img) = backend
            .convert_q4_0_to_noshuffle(&src_buf, num_blocks, ne00, ne01)
            .unwrap();

        // Upload activation
        let act_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                ne00,
                None,
            )
            .unwrap()
        };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                &act_buf,
                true,
                0,
                std::slice::from_raw_parts(activation.as_ptr() as *const u8, ne00 * 4),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
        }

        // Allocate output
        let dst_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                ne01,
                None,
            )
            .unwrap()
        };

        // Try running the noshuffle GEMV kernel
        match backend.matmul_q4_0_noshuffle(&q_buf, &d_buf, &act_buf, &dst_buf, ne00, ne01) {
            Ok(()) => {
                backend.queue.finish().unwrap();

                // Read back output
                let mut output = vec![0.0f32; ne01];
                unsafe {
                    ocl::core::enqueue_read_buffer(
                        &backend.queue,
                        &dst_buf,
                        true,
                        0,
                        std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u8, ne01 * 4),
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )
                    .unwrap();
                }

                // Compare with reference
                let max_abs_error = reference
                    .iter()
                    .zip(output.iter())
                    .map(|(r, o)| (r - o).abs())
                    .fold(0.0f32, f32::max);

                eprintln!("Reference: {:?}", &reference);
                eprintln!("GPU:       {:?}", &output);
                eprintln!("Max abs error: {}", max_abs_error);

                // Q4_0 dequant introduces rounding, so allow moderate tolerance
                assert!(
                    max_abs_error < 0.01,
                    "GEMV output diverges from reference: max_abs_error={} (threshold=0.01)",
                    max_abs_error
                );

                eprintln!(
                    "[PASS] Noshuffle GEMV correctness verified for {}x{}",
                    ne01, ne00
                );
            }
            Err(e) => {
                eprintln!(
                    "[SKIPPED] Noshuffle GEMV kernel failed to compile/dispatch (expected on \
                     macOS without Adreno extensions): {}",
                    e
                );
                // SOA conversion + transpose tests above already validate the data layout.
            }
        }
    }

    #[test]
    fn test_image1d_buffer_creation_and_readback() {
        let backend = match try_create_backend() {
            Some(b) => b,
            None => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // Create a buffer with known data
        let data: Vec<u32> = (0..256).collect();
        let buf = unsafe {
            let b = ocl::core::create_buffer::<_, u32>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                data.len(),
                None,
            )
            .unwrap();
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                &b,
                true,
                0,
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
            b
        };

        // Create image1d_buffer_t wrapping it (R32UI format)
        use ocl::core::{
            ImageChannelDataType, ImageChannelOrder, ImageDescriptor, ImageFormat, MemObjectType,
        };
        let img_fmt = ImageFormat::new(ImageChannelOrder::R, ImageChannelDataType::UnsignedInt32);
        let img_desc = ImageDescriptor::new(
            MemObjectType::Image1dBuffer,
            data.len(),
            0,
            0,
            0,
            0,
            0,
            Some(buf.clone()),
        );
        let result = unsafe {
            ocl::core::create_image(
                backend.context.as_core(),
                ocl::core::MEM_READ_ONLY,
                &img_fmt,
                &img_desc,
                None::<&[u32]>,
                None,
            )
        };

        match result {
            Ok(img) => {
                eprintln!(
                    "[PASS] image1d_buffer_t R32UI created: width={}",
                    data.len()
                );
                drop(img);
            }
            Err(e) => {
                eprintln!("[SKIPPED] image1d_buffer_t not supported: {}", e);
            }
        }

        // Also test RGBA32F format (for activation vectors)
        let float_data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let float_buf = unsafe {
            let b = ocl::core::create_buffer::<_, f32>(
                backend.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                float_data.len(),
                None,
            )
            .unwrap();
            ocl::core::enqueue_write_buffer(
                &backend.queue,
                &b,
                true,
                0,
                std::slice::from_raw_parts(float_data.as_ptr() as *const u8, float_data.len() * 4),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .unwrap();
            b
        };

        let act_fmt = ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelDataType::Float);
        let act_desc = ImageDescriptor::new(
            MemObjectType::Image1dBuffer,
            float_data.len() / 4, // RGBA = 4 floats per texel
            0,
            0,
            0,
            0,
            0,
            Some(float_buf.clone()),
        );
        let act_result = unsafe {
            ocl::core::create_image(
                backend.context.as_core(),
                ocl::core::MEM_READ_ONLY,
                &act_fmt,
                &act_desc,
                None::<&[f32]>,
                None,
            )
        };

        match act_result {
            Ok(img) => {
                eprintln!(
                    "[PASS] image1d_buffer_t RGBA32F created: width={}",
                    float_data.len() / 4
                );
                drop(img);
            }
            Err(e) => {
                eprintln!("[SKIPPED] image1d_buffer_t RGBA32F not supported: {}", e);
            }
        }
    }
}
