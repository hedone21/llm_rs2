//! CUDA backend for Jetson (SM >= 7.2, UMA).
//!
//! Phase 4: NVRTC custom kernels — all ops run on GPU, no CPU fallback
//! except Q4_0 matmul (cuBLAS doesn't support Q4 natively).
//!
//! Key components:
//! - `CudaHostBuffer` (cuMemHostAlloc + DEVICEMAP): zero-copy CPU+GPU access
//! - cuBLAS `cublasGemmEx`: F32xF32, F16xF16 matmul
//! - NVRTC custom kernels: rms_norm, rope, softmax, silu_mul, attention_gen, etc.
//! - F32xF16 matmul: cast F32->F16 via custom kernel, then cuBLAS F16xF16

pub mod kernels;
pub mod memory;
pub mod profiler;

use crate::buffer::cuda_buffer::{CudaBuffer, CudaDeviceBuffer, CudaHostBuffer};
use crate::core::backend::Backend;
use crate::core::buffer::{Buffer, DType};
use crate::core::tensor::Tensor;
use anyhow::{Context, Result, anyhow};
use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
use cudarc::driver::sys as cuda_sys;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg, result as cuda_result};
use kernels::CudaKernels;
use profiler::{CudaOpProfiler, CudaOpTag};
use std::any::Any;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::Arc;

/// Wrapper around a raw cuBLAS handle for safe sharing via Arc.
///
/// cuBLAS handles are thread-safe for concurrent read operations and
/// serialized internally for writes, so sharing via Arc is safe.
struct CublasHandle {
    handle: cublas_sys::cublasHandle_t,
}

// SAFETY: cuBLAS handles are thread-safe -- the library serializes concurrent
// API calls internally. We only share via Arc (no mutable aliasing).
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        // SAFETY: We own this handle (created in CudaBackend::new) and
        // only destroy it once (here in Drop).
        unsafe {
            let _ = cublas_result::destroy_handle(self.handle);
        }
    }
}

/// CUDA backend targeting Jetson UMA devices.
///
/// Phase 4: cuBLAS matmul + NVRTC custom kernels for all other ops.
/// CPU fallback only for Q4_0 matmul and matmul_slice.
#[derive(Clone)]
pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    device_name: String,
    compute_capability: (i32, i32),
    is_uma: bool,
    cublas: Arc<CublasHandle>,
    kernels: Arc<CudaKernels>,
    /// Cached F16 cast buffer for matmul_transposed F32×F16 path.
    cast_cache: Arc<std::sync::Mutex<Option<CudaHostBuffer>>>,
    /// Per-op CUDA event profiler. `None` = profiling off (zero
    /// overhead on the launch path: one `Mutex::lock` + `Option::is_none`
    /// check). Enabled via `enable_profiler`.
    profiler: Arc<std::sync::Mutex<Option<CudaOpProfiler>>>,
    /// Caller-side label hint for the next `matmul_transposed` dispatch.
    /// Mirrors `OpenCLBackend::op_label_hint`; set by `set_op_label`
    /// (forward_gen / transformer.rs) to distinguish matmul_qkv /
    /// matmul_wo / matmul_ffn / lm_head.
    ///
    /// SAFETY: only accessed from the single inference thread, same
    /// pattern as OpenCL. Wrapped in `Arc` so `Clone` on the backend is
    /// cheap and shares the same hint slot.
    op_label_hint: Arc<OpLabelHint>,
}

/// Wrapper around `UnsafeCell<Option<&'static str>>` so we can
/// `unsafe impl Send + Sync` without exposing the cell on the backend
/// struct.
pub(crate) struct OpLabelHint(UnsafeCell<Option<&'static str>>);

impl OpLabelHint {
    fn new() -> Self {
        Self(UnsafeCell::new(None))
    }

    #[inline]
    fn get(&self) -> Option<&'static str> {
        // SAFETY: single-threaded inference access.
        unsafe { *self.0.get() }
    }

    #[inline]
    fn set(&self, label: Option<&'static str>) {
        // SAFETY: single-threaded inference access.
        unsafe {
            *self.0.get() = label;
        }
    }
}

// SAFETY: same rationale as `OpenCLBackend::op_label_hint` — the
// inference loop is single-threaded and the hint is only read/written
// on that thread. Any cross-thread access would be via `clone()` +
// `set_op_label` from a fresh thread, which is not a supported
// pattern.
unsafe impl Send for OpLabelHint {}
unsafe impl Sync for OpLabelHint {}

impl CudaBackend {
    /// Initialize CUDA backend on device 0.
    ///
    /// Requirements:
    /// - Compute Capability >= 7.2 (Jetson AGX Xavier / Volta+)
    /// - Managed Memory support
    pub fn new() -> Result<Self> {
        Self::with_ordinal(0)
    }

    /// Initialize CUDA backend on a specific device ordinal.
    pub fn with_ordinal(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)
            .map_err(|e| anyhow!("CUDA context creation failed (ordinal={ordinal}): {e}"))?;

        // Query compute capability
        let cc_major = ctx
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| anyhow!("Failed to query CC major: {e}"))?;
        let cc_minor = ctx
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| anyhow!("Failed to query CC minor: {e}"))?;

        if cc_major < 7 || (cc_major == 7 && cc_minor < 2) {
            return Err(anyhow!(
                "CUDA backend requires SM >= 7.2 (Jetson Xavier+), got sm_{cc_major}{cc_minor}"
            ));
        }

        // Check managed memory support
        let managed_mem = ctx
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)
            .unwrap_or(0);
        if managed_mem == 0 {
            return Err(anyhow!("Device does not support managed (unified) memory"));
        }

        // Check UMA: integrated GPU (e.g., Jetson) shares physical memory with CPU.
        // CU_DEVICE_ATTRIBUTE_INTEGRATED = 1 means integrated (UMA), 0 means discrete.
        let integrated = ctx
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)
            .unwrap_or(0);
        let is_uma = integrated != 0;

        // Get device name
        let cu_device = cuda_result::device::get(ordinal as i32)
            .map_err(|e| anyhow!("Failed to get CUDA device: {e}"))?;
        let device_name = cuda_result::device::get_name(cu_device)
            .unwrap_or_else(|_| format!("CUDA Device {ordinal}"));

        // Create cuBLAS handle and bind to the same stream as custom kernels.
        // Without this, cuBLAS uses stream 0 while custom kernels use
        // ctx.default_stream(), causing cross-stream ordering issues.
        let cublas_handle =
            cublas_result::create_handle().map_err(|e| anyhow!("cublasCreate_v2 failed: {e}"))?;
        unsafe {
            cublas_result::set_stream(cublas_handle, ctx.default_stream().cu_stream() as _)
                .map_err(|e| anyhow!("cublasSetStream failed: {e}"))?;
        }

        // Compile custom NVRTC kernels
        let cc = (cc_major, cc_minor);
        let kernels = CudaKernels::compile(&ctx, cc)?;

        eprintln!(
            "[CUDA] Device: {} (sm_{}{}, UMA={}, managed_mem={}, cuBLAS=ready, kernels=ready)",
            device_name, cc_major, cc_minor, is_uma, managed_mem
        );

        let backend = Self {
            ctx,
            device_name,
            compute_capability: (cc_major, cc_minor),
            is_uma,
            cublas: Arc::new(CublasHandle {
                handle: cublas_handle,
            }),
            kernels: Arc::new(kernels),
            cast_cache: Arc::new(std::sync::Mutex::new(None)),
            profiler: Arc::new(std::sync::Mutex::new(None)),
            op_label_hint: Arc::new(OpLabelHint::new()),
        };

        // Run self-test to verify kernel launch + arg passing
        backend.self_test()?;

        Ok(backend)
    }

    /// Access the underlying CUDA context (for buffer allocation, kernel launches, etc).
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Compute capability as (major, minor).
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    // ── Per-op CUDA event profiler (--cuda-profile) ─────────────────
    //
    // The fast path when profiling is disabled is a single
    // `Mutex::lock()` + `Option::is_none()` check per launch site;
    // when disabled the helper returns early before touching the CUDA
    // event API. See `profiler.rs` for the timing model.

    /// Enable the per-op CUDA event profiler with a fixed-size event
    /// pool. Idempotent — a second call replaces the existing state.
    pub fn enable_profiler(&self, pool_size: usize) -> Result<()> {
        let p = CudaOpProfiler::new(pool_size).context("allocating CudaOpProfiler event pool")?;
        *self.profiler.lock().unwrap() = Some(p);
        eprintln!("[CUDA-Profile] event-based profiler enabled (pool_size={pool_size})");
        Ok(())
    }

    /// Whether profiling is currently active.
    pub fn profiler_enabled(&self) -> bool {
        self.profiler.lock().unwrap().is_some()
    }

    /// Drain pending events, synchronising the last one, and return a
    /// snapshot of the current aggregate map. Returns `None` when the
    /// profiler is disabled.
    ///
    /// The internal aggregate keeps accumulating across calls — to
    /// reset between runs use `reset_profiler()`.
    pub fn flush_profiler(&self) -> Result<Option<HashMap<&'static str, (u64, f64)>>> {
        let mut guard = self.profiler.lock().unwrap();
        let Some(p) = guard.as_mut() else {
            return Ok(None);
        };
        p.flush()?;
        Ok(Some(p.report().clone()))
    }

    /// Number of start attempts dropped because the event pool was
    /// exhausted between flushes. Returns 0 when disabled.
    pub fn profiler_dropped(&self) -> u64 {
        self.profiler
            .lock()
            .unwrap()
            .as_ref()
            .map(|p| p.dropped())
            .unwrap_or(0)
    }

    /// Clear aggregate + in-flight state. Pool stays allocated.
    pub fn reset_profiler(&self) {
        if let Some(p) = self.profiler.lock().unwrap().as_mut() {
            p.reset();
        }
    }

    /// Caller-side label hint for the next `matmul_transposed`
    /// dispatch. See `OpenCLBackend::set_op_label` for the design.
    /// No-op when the profiler is disabled — the hint is cheap to
    /// set even so to keep the call sites uniform.
    #[inline]
    pub fn set_op_label(&self, label: &'static str) {
        self.op_label_hint.set(Some(label));
    }

    /// Clear a previously-set label hint.
    #[inline]
    pub fn clear_op_label(&self) {
        self.op_label_hint.set(None);
    }

    /// Begin timing `tag` (resolving against the current label hint)
    /// on `stream`. Returns the raw CUstream for use by the caller
    /// and a `bool` indicating whether a matching `end_op` is needed
    /// (true when the start event actually recorded).
    ///
    /// Disabled-profiler fast path: returns `(stream_raw, false)`
    /// without acquiring the mutex's inner state — one lock +
    /// `is_none()` check.
    #[inline]
    fn begin_op(&self, tag: CudaOpTag, stream: cuda_sys::CUstream) -> Result<bool> {
        let mut guard = self.profiler.lock().unwrap();
        let Some(p) = guard.as_mut() else {
            return Ok(false);
        };
        let label = self
            .op_label_hint
            .get()
            .unwrap_or_else(|| tag.profile_label());
        p.record_start(label, stream)
    }

    /// Pair with a successful `begin_op`. Called only when
    /// `begin_op` returned `Ok(true)`.
    #[inline]
    fn end_op(&self, stream: cuda_sys::CUstream) -> Result<()> {
        let mut guard = self.profiler.lock().unwrap();
        if let Some(p) = guard.as_mut() {
            p.record_end(stream)?;
        }
        Ok(())
    }

    /// Attempt to get a CUdeviceptr from a Buffer.
    ///
    /// Returns Some(device_ptr) if the buffer is a CudaHostBuffer or CudaBuffer,
    /// None otherwise (e.g., SharedBuffer/MmapBuffer without CUDA registration).
    fn get_device_ptr(buf: &dyn Buffer) -> Option<cuda_sys::CUdeviceptr> {
        buf.as_any()
            .downcast_ref::<CudaHostBuffer>()
            .map(|hb| hb.device_ptr())
            .or_else(|| {
                buf.as_any()
                    .downcast_ref::<CudaBuffer>()
                    .map(|cb| cb.device_ptr())
            })
            .or_else(|| {
                buf.as_any()
                    .downcast_ref::<CudaDeviceBuffer>()
                    .map(|db| db.device_ptr())
            })
    }

    /// Get device pointer or return error with context.
    #[allow(dead_code)]
    fn require_device_ptr(buf: &dyn Buffer, name: &str) -> Result<cuda_sys::CUdeviceptr> {
        Self::get_device_ptr(buf)
            .ok_or_else(|| anyhow!("{name}: buffer has no CUDA device pointer"))
    }

    /// Run a basic self-test to verify kernel launch + arg passing works.
    /// Tests add_assign, scale, rms_norm with known data.
    pub fn self_test(&self) -> Result<()> {
        use crate::buffer::cuda_buffer::CudaHostBuffer;
        use crate::core::buffer::DType;

        // === Test 1: add_assign [1,2,3] + [4,5,6] = [5,7,9] ===
        let a_buf = CudaHostBuffer::new(12, DType::F32)?;
        let b_buf = CudaHostBuffer::new(12, DType::F32)?;
        unsafe {
            let a = std::slice::from_raw_parts_mut(a_buf.as_mut_ptr() as *mut f32, 3);
            a.copy_from_slice(&[1.0, 2.0, 3.0]);
            let b = std::slice::from_raw_parts_mut(b_buf.as_mut_ptr() as *mut f32, 3);
            b.copy_from_slice(&[4.0, 5.0, 6.0]);
        }
        let a_ptr = a_buf.device_ptr();
        let b_ptr = b_buf.device_ptr();
        let k: i32 = 3;
        let stream = self.ctx.default_stream();
        unsafe {
            stream
                .launch_builder(&self.kernels.add_assign)
                .arg(&a_ptr)
                .arg(&b_ptr)
                .arg(&k)
                .launch(Self::launch_config_1d(3))
                .map_err(|e| anyhow!("self-test add_assign launch failed: {e}"))?;
        }
        self.synchronize()?;
        let result = unsafe { std::slice::from_raw_parts(a_buf.as_ptr() as *const f32, 3) };
        eprint!(
            "[CUDA self-test] add_assign: [{:.1}, {:.1}, {:.1}]",
            result[0], result[1], result[2]
        );
        if (result[0] - 5.0).abs() > 0.01
            || (result[1] - 7.0).abs() > 0.01
            || (result[2] - 9.0).abs() > 0.01
        {
            eprintln!(" FAIL (expected [5.0, 7.0, 9.0])");
            return Err(anyhow!("self-test add_assign FAILED"));
        }
        eprintln!(" OK");

        // === Test 2: scale [2,4,6] * 0.5 = [1,2,3] ===
        let s_buf = CudaHostBuffer::new(12, DType::F32)?;
        unsafe {
            let s = std::slice::from_raw_parts_mut(s_buf.as_mut_ptr() as *mut f32, 3);
            s.copy_from_slice(&[2.0, 4.0, 6.0]);
        }
        let s_ptr = s_buf.device_ptr();
        let v: f32 = 0.5;
        unsafe {
            stream
                .launch_builder(&self.kernels.scale)
                .arg(&s_ptr)
                .arg(&v)
                .arg(&k)
                .launch(Self::launch_config_1d(3))
                .map_err(|e| anyhow!("self-test scale launch failed: {e}"))?;
        }
        self.synchronize()?;
        let result = unsafe { std::slice::from_raw_parts(s_buf.as_ptr() as *const f32, 3) };
        eprint!(
            "[CUDA self-test] scale:      [{:.1}, {:.1}, {:.1}]",
            result[0], result[1], result[2]
        );
        if (result[0] - 1.0).abs() > 0.01
            || (result[1] - 2.0).abs() > 0.01
            || (result[2] - 3.0).abs() > 0.01
        {
            eprintln!(" FAIL (expected [1.0, 2.0, 3.0])");
            return Err(anyhow!("self-test scale FAILED"));
        }
        eprintln!(" OK");

        // === Test 3: rms_norm [3,4] with weight [1,1] ===
        // rms = sqrt((9+16)/2) = sqrt(12.5) = 3.536
        // rms_inv = 1/3.536 = 0.2828
        // result = [3*0.2828, 4*0.2828] = [0.849, 1.131]
        let x_buf = CudaHostBuffer::new(8, DType::F32)?;
        let w_buf = CudaHostBuffer::new(8, DType::F32)?;
        unsafe {
            let x = std::slice::from_raw_parts_mut(x_buf.as_mut_ptr() as *mut f32, 2);
            x.copy_from_slice(&[3.0, 4.0]);
            let w = std::slice::from_raw_parts_mut(w_buf.as_mut_ptr() as *mut f32, 2);
            w.copy_from_slice(&[1.0, 1.0]);
        }
        let x_ptr = x_buf.device_ptr();
        let w_ptr = w_buf.device_ptr();
        let ncols: i32 = 2;
        let eps: f32 = 1e-5;
        let add_unit: i32 = 0;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1), // min warp size
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.kernels.rms_norm)
                .arg(&x_ptr) // dst
                .arg(&x_ptr) // x (in-place)
                .arg(&w_ptr) // weight
                .arg(&ncols)
                .arg(&eps)
                .arg(&add_unit)
                .launch(cfg)
                .map_err(|e| anyhow!("self-test rms_norm launch failed: {e}"))?;
        }
        self.synchronize()?;
        let result = unsafe { std::slice::from_raw_parts(x_buf.as_ptr() as *const f32, 2) };
        let expected_0 = 3.0 / (12.5f32 + 1e-5).sqrt();
        let expected_1 = 4.0 / (12.5f32 + 1e-5).sqrt();
        eprint!(
            "[CUDA self-test] rms_norm:   [{:.4}, {:.4}]",
            result[0], result[1]
        );
        if (result[0] - expected_0).abs() > 0.01 || (result[1] - expected_1).abs() > 0.01 {
            eprintln!(" FAIL (expected [{:.4}, {:.4}])", expected_0, expected_1);
            return Err(anyhow!(
                "self-test rms_norm FAILED: got [{}, {}]",
                result[0],
                result[1]
            ));
        }
        eprintln!(" OK");

        Ok(())
    }

    /// Standard 1D launch config: ceil(n/256) blocks, 256 threads.
    fn launch_config_1d(n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// Internal helper: get a CPU backend for fallback compute.
/// Uses the platform-native backend (Neon on aarch64, AVX2 on x86_64).
fn cpu_fallback() -> crate::backend::cpu::CpuBackend {
    crate::backend::cpu::CpuBackend::new()
}

/// Wrap a single kernel-launch block with optional per-op profiling.
///
/// `self_:expr` is typically `self`; `tag:expr` is a `CudaOpTag`; the
/// block `$body` performs the `launch_builder(...).launch(cfg)` call
/// on the default stream. When the profiler is disabled the pair of
/// `begin_op` / `end_op` collapses into two quick mutex lock +
/// `is_none()` checks — no CUDA API calls.
///
/// NOTE: the block must launch on the **default stream** so that
/// start/end event ordering is sequential. Custom streams would need
/// a dedicated record path.
macro_rules! cuda_profile {
    ($self_:expr, $tag:expr, $stream:expr, $body:block) => {{
        let __stream_raw: cuda_sys::CUstream = $stream.cu_stream();
        let __timed = $self_.begin_op($tag, __stream_raw)?;
        let __res: Result<()> = (|| $body)();
        if __timed {
            $self_.end_op(__stream_raw)?;
        }
        __res?;
    }};
}

impl Backend for CudaBackend {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "CUDA"
    }

    fn device(&self) -> &str {
        &self.device_name
    }

    fn is_gpu(&self) -> bool {
        true
    }

    fn is_discrete_gpu(&self) -> bool {
        !self.is_uma
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
        _is_head_major: bool,
    ) -> Result<bool> {
        let kv_dtype = k_cache.dtype();

        // Only support F32/F16 KV and head_dim in {64, 128, 256}
        if !matches!(head_dim, 64 | 128 | 256) || kv_dtype == DType::Q4_0 {
            self.synchronize()?;
            return Ok(false);
        }

        let q_ptr = Self::get_device_ptr(q.buffer().as_ref());
        let k_ptr = Self::get_device_ptr(k_cache.buffer().as_ref());
        let v_ptr = Self::get_device_ptr(v_cache.buffer().as_ref());
        let out_ptr = Self::get_device_ptr(out.buffer().as_ref());

        if let (Some(qp), Some(kp), Some(vp), Some(op)) = (q_ptr, k_ptr, v_ptr, out_ptr) {
            let block_m: u32 = 32;
            let cfg = LaunchConfig {
                grid_dim: (
                    seq_len.div_ceil(block_m as usize) as u32,
                    (n_heads_q * batch_size) as u32,
                    1,
                ),
                block_dim: (block_m, 1, 1),
                shared_mem_bytes: 0,
            };

            let nhq = n_heads_q as i32;
            let nkv = n_heads_kv as i32;
            let sl = seq_len as i32;
            let csl = cache_seq_len as i32;
            let cap = kv_capacity as i32;
            let bs = batch_size as i32;

            let kernel = match (kv_dtype, head_dim) {
                (DType::F32, 64) => &self.kernels.flash_prefill_f32_dk64,
                (DType::F32, 128) => &self.kernels.flash_prefill_f32_dk128,
                (DType::F32, 256) => &self.kernels.flash_prefill_f32_dk256,
                (DType::F16, 64) => &self.kernels.flash_prefill_f16kv_dk64,
                (DType::F16, 128) => &self.kernels.flash_prefill_f16kv_dk128,
                (DType::F16, 256) => &self.kernels.flash_prefill_f16kv_dk256,
                _ => {
                    self.synchronize()?;
                    return Ok(false);
                }
            };

            let stream = self.ctx.default_stream();
            // SAFETY: qp is valid F32 device ptr for Q [batch, seq_len, n_heads_q, head_dim].
            // kp/vp are valid F32 or F16 device ptrs for KV [batch, n_heads_kv, capacity, head_dim].
            // op is valid F32 device ptr for output, same layout as Q.
            // All dimensions are checked by callers (transformer layer).
            cuda_profile!(self, CudaOpTag::FlashAttentionPrefill, stream, {
                unsafe {
                    stream
                        .launch_builder(kernel)
                        .arg(&qp)
                        .arg(&kp)
                        .arg(&vp)
                        .arg(&op)
                        .arg(&nhq)
                        .arg(&nkv)
                        .arg(&sl)
                        .arg(&csl)
                        .arg(&cap)
                        .arg(&bs)
                        .launch(cfg)
                        .map_err(|e| anyhow!("flash_attn_prefill launch failed: {e}"))?;
                }
                Ok(())
            });
            self.synchronize()?;
            Ok(true)
        } else {
            self.synchronize()?;
            Ok(false)
        }
    }

    // --- Math ops ---

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        cpu_fallback().matmul(a, b, out)
    }

    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        // Sync before cuBLAS: other ops may have written to input buffers
        // on a different stream than cuBLAS uses.
        self.synchronize()?;
        let a_dtype = a.dtype();
        let b_dtype = b.dtype();

        // Q4_0 weights always go through CPU fallback
        if b_dtype == DType::Q4_0 || a_dtype == DType::Q4_0 {
            return cpu_fallback().matmul_transposed(a, b, out);
        }

        // Try to get device pointers for cuBLAS
        let a_dev = Self::get_device_ptr(a.buffer().as_ref());
        let b_dev = Self::get_device_ptr(b.buffer().as_ref());
        let out_dev = Self::get_device_ptr(out.buffer().as_ref());

        // All three buffers must have device pointers for cuBLAS
        if let (Some(a_ptr), Some(b_ptr), Some(out_ptr)) = (a_dev, b_dev, out_dev) {
            let a_dims = a.shape().dims();
            let b_dims = b.shape().dims();
            let a_rank = a_dims.len();
            let b_rank = b_dims.len();

            let k = a_dims[a_rank - 1] as i32;
            let m: i32 = a_dims[..a_rank - 1].iter().product::<usize>() as i32;
            let n = b_dims[b_rank - 2] as i32;

            if a_dtype == DType::F32 && b_dtype == DType::F32 {
                // Pure F32: use sgemm
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;
                let stream = self.ctx.default_stream();
                // SAFETY: All device pointers are valid CudaHostBuffer/CudaBuffer allocations.
                // Dimensions are checked by shape accessors. cuBLAS handle is valid.
                cuda_profile!(self, CudaOpTag::Matmul, stream, {
                    unsafe {
                        cublas_result::sgemm(
                            self.cublas.handle,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            b_ptr as *const f32,
                            k,
                            a_ptr as *const f32,
                            k,
                            &beta,
                            out_ptr as *mut f32,
                            n,
                        )
                        .map_err(|e| anyhow!("cublasSgemm failed: {e}"))?;
                    }
                    Ok(())
                });
                self.synchronize()?;
                Ok(())
            } else if a_dtype == DType::F32 && b_dtype == DType::F16 {
                // F32 activation x F16 weight: cast A to F16, then F16xF16 cuBLAS.
                // Reuse cached buffer to avoid per-call allocation.
                let k_elements = (m * k) as usize;
                let needed = k_elements * 2;
                let a_f16_buf = {
                    let mut cache = self.cast_cache.lock().unwrap();
                    match cache.take() {
                        Some(buf) if buf.size() >= needed => buf,
                        _ => CudaHostBuffer::new(needed, DType::F16)?,
                    }
                };
                let a_f16_ptr = a_f16_buf.device_ptr();

                // Launch F32->F16 cast kernel
                let cfg = Self::launch_config_1d(k_elements);
                let stream = self.ctx.default_stream();
                let k_i32 = k_elements as i32;
                // SAFETY: a_ptr is valid F32 device memory of size k_elements*4.
                // a_f16_ptr is valid F16 device memory of size k_elements*2.
                // Cast kernel reads k_elements floats and writes k_elements halfs.
                cuda_profile!(self, CudaOpTag::Cast, stream, {
                    unsafe {
                        stream
                            .launch_builder(&self.kernels.cast_f32_f16)
                            .arg(&a_ptr)
                            .arg(&a_f16_ptr)
                            .arg(&k_i32)
                            .launch(cfg)
                            .map_err(|e| anyhow!("cast_f32_to_f16 kernel launch failed: {e}"))?;
                    }
                    Ok(())
                });

                // F16xF16 cuBLAS GemmEx
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;
                // SAFETY: a_f16_ptr and b_ptr are valid F16 device memory.
                // out_ptr is valid F32 device memory. Dimensions match.
                cuda_profile!(self, CudaOpTag::Matmul, stream, {
                    unsafe {
                        cublas_result::gemm_ex(
                            self.cublas.handle,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha as *const f32 as *const std::ffi::c_void,
                            b_ptr as *const std::ffi::c_void,
                            cublas_sys::cudaDataType_t::CUDA_R_16F,
                            k,
                            a_f16_ptr as *const std::ffi::c_void,
                            cublas_sys::cudaDataType_t::CUDA_R_16F,
                            k,
                            &beta as *const f32 as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            cublas_sys::cudaDataType_t::CUDA_R_32F,
                            n,
                            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                        )
                        .map_err(|e| anyhow!("cublasGemmEx (F32->F16 x F16) failed: {e}"))?;
                    }
                    Ok(())
                });
                self.synchronize()?;
                *self.cast_cache.lock().unwrap() = Some(a_f16_buf);
                Ok(())
            } else if a_dtype == DType::F16 && b_dtype == DType::F16 {
                // Both F16: use GemmEx with F32 compute for accuracy
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;
                let stream = self.ctx.default_stream();
                // SAFETY: Both pointers are valid F16 device memory. out_ptr is valid F32.
                cuda_profile!(self, CudaOpTag::Matmul, stream, {
                    unsafe {
                        cublas_result::gemm_ex(
                            self.cublas.handle,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha as *const f32 as *const std::ffi::c_void,
                            b_ptr as *const std::ffi::c_void,
                            cublas_sys::cudaDataType_t::CUDA_R_16F,
                            k,
                            a_ptr as *const std::ffi::c_void,
                            cublas_sys::cudaDataType_t::CUDA_R_16F,
                            k,
                            &beta as *const f32 as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            cublas_sys::cudaDataType_t::CUDA_R_32F,
                            n,
                            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                        )
                        .map_err(|e| anyhow!("cublasGemmEx (F16xF16) failed: {e}"))?;
                    }
                    Ok(())
                });
                self.synchronize()?;
                Ok(())
            } else {
                // Unsupported dtype combination -> CPU fallback
                cpu_fallback().matmul_transposed(a, b, out)
            }
        } else {
            // No device pointers available -> CPU fallback
            cpu_fallback().matmul_transposed(a, b, out)
        }
    }

    fn matmul_slice(
        &self,
        a: &Tensor,
        b: &Tensor,
        rows: usize,
        cols: usize,
        out: &mut Tensor,
    ) -> Result<()> {
        cpu_fallback().matmul_slice(a, b, rows, cols, out)
    }

    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        let n = a.numel();
        let a_ptr = Self::require_device_ptr(a.buffer().as_ref(), "add_assign a")?;
        let b_ptr = Self::require_device_ptr(b.buffer().as_ref(), "add_assign b")?;
        let k = n as i32;
        let stream = self.ctx.default_stream();
        // SAFETY: a_ptr and b_ptr are valid F32 device memory of n elements each.
        cuda_profile!(self, CudaOpTag::AddAssign, stream, {
            unsafe {
                stream
                    .launch_builder(&self.kernels.add_assign)
                    .arg(&a_ptr)
                    .arg(&b_ptr)
                    .arg(&k)
                    .launch(Self::launch_config_1d(n))
                    .map_err(|e| anyhow!("add_assign kernel launch failed: {e}"))?;
            }
            Ok(())
        });
        self.synchronize()?;
        Ok(())
    }

    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()> {
        let n = x.numel();
        let x_ptr = Self::require_device_ptr(x.buffer().as_ref(), "scale x")?;
        let k = n as i32;
        let stream = self.ctx.default_stream();
        // SAFETY: x_ptr is valid F32 device memory of n elements.
        cuda_profile!(self, CudaOpTag::Scale, stream, {
            unsafe {
                stream
                    .launch_builder(&self.kernels.scale)
                    .arg(&x_ptr)
                    .arg(&v)
                    .arg(&k)
                    .launch(Self::launch_config_1d(n))
                    .map_err(|e| anyhow!("scale kernel launch failed: {e}"))?;
            }
            Ok(())
        });
        self.synchronize()?;
        Ok(())
    }

    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        let n = a.numel();
        let a_ptr = Self::require_device_ptr(a.buffer().as_ref(), "silu_mul gate")?;
        let b_ptr = Self::require_device_ptr(b.buffer().as_ref(), "silu_mul up")?;
        let k = n as i32;
        let stream = self.ctx.default_stream();
        // SAFETY: a_ptr (gate) and b_ptr (up) are valid F32 device memory of n elements.
        cuda_profile!(self, CudaOpTag::SiluMul, stream, {
            unsafe {
                stream
                    .launch_builder(&self.kernels.silu_mul)
                    .arg(&a_ptr)
                    .arg(&b_ptr)
                    .arg(&k)
                    .launch(Self::launch_config_1d(n))
                    .map_err(|e| anyhow!("silu_mul kernel launch failed: {e}"))?;
            }
            Ok(())
        });
        self.synchronize()?;
        Ok(())
    }

    fn gelu_tanh_mul(&self, gate: &mut Tensor, up: &Tensor) -> Result<()> {
        let n = gate.numel();
        let gate_ptr = Self::require_device_ptr(gate.buffer().as_ref(), "gelu_tanh_mul gate")?;
        let up_ptr = Self::require_device_ptr(up.buffer().as_ref(), "gelu_tanh_mul up")?;
        let k = n as i32;
        let stream = self.ctx.default_stream();
        // SAFETY: gate_ptr and up_ptr are valid F32 device memory of n elements.
        cuda_profile!(self, CudaOpTag::GeluTanhMul, stream, {
            unsafe {
                stream
                    .launch_builder(&self.kernels.gelu_tanh_mul)
                    .arg(&gate_ptr)
                    .arg(&up_ptr)
                    .arg(&k)
                    .launch(Self::launch_config_1d(n))
                    .map_err(|e| anyhow!("gelu_tanh_mul kernel launch failed: {e}"))?;
            }
            Ok(())
        });
        self.synchronize()?;
        Ok(())
    }

    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32, add_unit: bool) -> Result<()> {
        let dims = x.shape().dims().to_vec();
        let rank = dims.len();
        let ncols = dims[rank - 1] as i32;
        let nrows: usize = dims[..rank - 1].iter().product::<usize>().max(1);
        let x_ptr = Self::require_device_ptr(x.buffer().as_ref(), "rms_norm x")?;
        let w_ptr = Self::require_device_ptr(w.buffer().as_ref(), "rms_norm w")?;
        let add_unit_i32: i32 = if add_unit { 1 } else { 0 };
        let block_size = (ncols as usize).min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.ctx.default_stream();
        // SAFETY: x_ptr (in-place dst+src) and w_ptr are valid F32 device memory.
        // nrows * ncols == total elements.
        cuda_profile!(self, CudaOpTag::RmsNorm, stream, {
            unsafe {
                stream
                    .launch_builder(&self.kernels.rms_norm)
                    .arg(&x_ptr) // dst
                    .arg(&x_ptr) // x (in-place)
                    .arg(&w_ptr) // weight
                    .arg(&ncols)
                    .arg(&eps)
                    .arg(&add_unit_i32)
                    .launch(cfg)
                    .map_err(|e| anyhow!("rms_norm kernel launch failed: {e}"))?;
            }
            Ok(())
        });
        self.synchronize()?;
        Ok(())
    }

    fn rms_norm_oop(
        &self,
        x: &Tensor,
        out: &mut Tensor,
        w: &Tensor,
        eps: f32,
        add_unit: bool,
    ) -> Result<()> {
        let dims = x.shape().dims().to_vec();
        let rank = dims.len();
        let ncols = dims[rank - 1] as i32;
        let nrows: usize = dims[..rank - 1].iter().product::<usize>().max(1);
        let x_ptr = Self::require_device_ptr(x.buffer().as_ref(), "rms_norm_oop x")?;
        let out_ptr = Self::require_device_ptr(out.buffer().as_ref(), "rms_norm_oop out")?;
        let w_ptr = Self::require_device_ptr(w.buffer().as_ref(), "rms_norm_oop w")?;
        let add_unit_i32: i32 = if add_unit { 1 } else { 0 };
        let block_size = (ncols as usize).min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.ctx.default_stream();
        // SAFETY: out_ptr (dst), x_ptr (src), w_ptr are valid F32 device memory.
        cuda_profile!(self, CudaOpTag::RmsNorm, stream, {
            unsafe {
                stream
                    .launch_builder(&self.kernels.rms_norm)
                    .arg(&out_ptr) // dst
                    .arg(&x_ptr) // x (read-only)
                    .arg(&w_ptr) // weight
                    .arg(&ncols)
                    .arg(&eps)
                    .arg(&add_unit_i32)
                    .launch(cfg)
                    .map_err(|e| anyhow!("rms_norm_oop kernel launch failed: {e}"))?;
            }
            Ok(())
        });
        self.synchronize()?;
        Ok(())
    }

    fn add_rms_norm_oop(
        &self,
        x: &mut Tensor,
        residual: &Tensor,
        out: &mut Tensor,
        w: &Tensor,
        eps: f32,
        add_unit: bool,
    ) -> Result<()> {
        // Fused: x += residual; out = rms_norm(x) * w
        self.add_assign(x, residual)?;
        self.rms_norm_oop(x, out, w, eps, add_unit)
    }

    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        let dims = x.shape().dims().to_vec();
        let rank = dims.len();
        let ncols = dims[rank - 1] as i32;
        let nrows: usize = dims[..rank - 1].iter().product::<usize>().max(1);
        let x_ptr = Self::require_device_ptr(x.buffer().as_ref(), "softmax x")?;
        let block_size = (ncols as usize).min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.ctx.default_stream();
        // SAFETY: x_ptr is valid F32 device memory of nrows*ncols elements.
        cuda_profile!(self, CudaOpTag::Softmax, stream, {
            unsafe {
                stream
                    .launch_builder(&self.kernels.softmax)
                    .arg(&x_ptr)
                    .arg(&ncols)
                    .launch(cfg)
                    .map_err(|e| anyhow!("softmax kernel launch failed: {e}"))?;
            }
            Ok(())
        });
        self.synchronize()?;
        Ok(())
    }

    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        let dims = x.shape().dims().to_vec();
        let rank = dims.len();
        if rank < 3 {
            return Err(anyhow!(
                "rope_inplace: expected at least 3 dims, got {rank}"
            ));
        }
        let head_dim = dims[rank - 1];
        let n_heads = dims[rank - 2];
        let seq_len: usize = dims[..rank - 1]
            .iter()
            .rev()
            .skip(1)
            .next()
            .copied()
            .unwrap_or(1);

        let x_ptr = Self::require_device_ptr(x.buffer().as_ref(), "rope_inplace x")?;
        let hd = head_dim as i32;
        let nh = n_heads as i32;
        let sl = seq_len as i32;
        let sp = start_pos as i32;
        let cfg = LaunchConfig {
            grid_dim: ((seq_len * n_heads) as u32, 1, 1),
            block_dim: ((head_dim / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.ctx.default_stream();
        cuda_profile!(self, CudaOpTag::Rope, stream, {
            unsafe {
                stream
                    .launch_builder(&self.kernels.rope_inplace)
                    .arg(&x_ptr)
                    .arg(&hd)
                    .arg(&nh)
                    .arg(&sl)
                    .arg(&sp)
                    .arg(&theta)
                    .launch(cfg)
                    .map_err(|e| anyhow!("rope_inplace kernel launch failed: {e}"))?;
            }
            Ok(())
        });
        self.synchronize()?;
        Ok(())
    }

    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        let src_dtype = src.dtype();
        let dst_dtype = dst.dtype();
        let n = src.numel();
        let src_ptr = Self::get_device_ptr(src.buffer().as_ref());
        let dst_ptr = Self::get_device_ptr(dst.buffer().as_ref());

        match (src_dtype, dst_dtype, src_ptr, dst_ptr) {
            (DType::F32, DType::F16, Some(sp), Some(dp)) => {
                let k = n as i32;
                let stream = self.ctx.default_stream();
                cuda_profile!(self, CudaOpTag::Cast, stream, {
                    unsafe {
                        stream
                            .launch_builder(&self.kernels.cast_f32_f16)
                            .arg(&sp)
                            .arg(&dp)
                            .arg(&k)
                            .launch(Self::launch_config_1d(n))
                            .map_err(|e| anyhow!("cast_f32_f16 kernel launch failed: {e}"))?;
                    }
                    Ok(())
                });
                self.synchronize()?; // CPU may read dst immediately
                Ok(())
            }
            (DType::F16, DType::F32, Some(sp), Some(dp)) => {
                let k = n as i32;
                let stream = self.ctx.default_stream();
                cuda_profile!(self, CudaOpTag::Cast, stream, {
                    unsafe {
                        stream
                            .launch_builder(&self.kernels.cast_f16_f32)
                            .arg(&sp)
                            .arg(&dp)
                            .arg(&k)
                            .launch(Self::launch_config_1d(n))
                            .map_err(|e| anyhow!("cast_f16_f32 kernel launch failed: {e}"))?;
                    }
                    Ok(())
                });
                self.synchronize()?;
                Ok(())
            }
            _ => {
                // Unsupported dtype or missing device ptr: sync then CPU fallback
                self.synchronize()?;
                cpu_fallback().cast(src, dst)
            }
        }
    }

    fn add_row_bias(&self, x: &mut Tensor, bias: &Tensor) -> Result<()> {
        let dims = x.shape().dims().to_vec();
        let rank = dims.len();
        let ncols = dims[rank - 1] as i32;
        let total = x.numel();
        let x_ptr = Self::require_device_ptr(x.buffer().as_ref(), "add_row_bias x")?;
        let bias_ptr = Self::require_device_ptr(bias.buffer().as_ref(), "add_row_bias bias")?;
        let total_i32 = total as i32;
        let stream = self.ctx.default_stream();
        // SAFETY: x_ptr is valid F32 device ptr of total elements; bias_ptr is valid F32 of ncols.
        cuda_profile!(self, CudaOpTag::AddRowBias, stream, {
            unsafe {
                stream
                    .launch_builder(&self.kernels.add_row_bias)
                    .arg(&x_ptr)
                    .arg(&bias_ptr)
                    .arg(&ncols)
                    .arg(&total_i32)
                    .launch(Self::launch_config_1d(total))
                    .map_err(|e| anyhow!("add_row_bias kernel launch failed: {e}"))?;
            }
            Ok(())
        });
        self.synchronize()?;
        Ok(())
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
        let ks_ptr = Self::get_device_ptr(k_src.buffer().as_ref());
        let vs_ptr = Self::get_device_ptr(v_src.buffer().as_ref());
        let kd_ptr = Self::get_device_ptr(k_dst.buffer().as_ref());
        let vd_ptr = Self::get_device_ptr(v_dst.buffer().as_ref());

        if let (Some(ks), Some(vs), Some(kd), Some(vd)) = (ks_ptr, vs_ptr, kd_ptr, vd_ptr) {
            // Infer kv_heads from k_src shape: [..., kv_heads, head_dim]
            let dims = k_src.shape().dims().to_vec();
            let kv_heads = if dims.len() >= 2 {
                dims[dims.len() - 2]
            } else {
                1
            };
            let hd = head_dim as i32;
            let cap = capacity as i32;
            let wp = write_pos as i32;
            let cfg = LaunchConfig {
                grid_dim: (kv_heads as u32, 1, 1),
                block_dim: (head_dim as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let stream = self.ctx.default_stream();
            // SAFETY: ks/vs are valid F32 device ptrs; kd/vd are valid F16 device ptrs.
            // Dimensions match kv_heads * head_dim for src, kv_heads * capacity * head_dim for dst.
            cuda_profile!(self, CudaOpTag::KvScatter, stream, {
                unsafe {
                    stream
                        .launch_builder(&self.kernels.kv_scatter)
                        .arg(&ks)
                        .arg(&vs)
                        .arg(&kd)
                        .arg(&vd)
                        .arg(&hd)
                        .arg(&cap)
                        .arg(&wp)
                        .launch(cfg)
                        .map_err(|e| anyhow!("kv_scatter kernel launch failed: {e}"))?;
                }
                Ok(())
            });
            self.synchronize()?;
            Ok(())
        } else {
            // Missing device ptr: sync then CPU fallback
            self.synchronize()?;
            cpu_fallback()
                .kv_scatter_f32_to_f16(k_src, v_src, k_dst, v_dst, head_dim, capacity, write_pos)
        }
    }

    fn supports_kv_scatter_batch(&self) -> bool {
        true
    }

    fn kv_scatter_f32_to_f16_batch(
        &self,
        k_src: &Tensor,
        v_src: &Tensor,
        k_dst: &mut Tensor,
        v_dst: &mut Tensor,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        write_pos_start: usize,
        seq_len: usize,
    ) -> Result<()> {
        let ks = Self::get_device_ptr(k_src.buffer().as_ref());
        let vs = Self::get_device_ptr(v_src.buffer().as_ref());
        let kd = Self::get_device_ptr(k_dst.buffer().as_ref());
        let vd = Self::get_device_ptr(v_dst.buffer().as_ref());

        if let (Some(ks), Some(vs), Some(kd), Some(vd)) = (ks, vs, kd, vd) {
            let hd = head_dim as i32;
            let cap = capacity as i32;
            let wps = write_pos_start as i32;
            let sl = seq_len as i32;
            let kvh = kv_heads as i32;
            let cfg = LaunchConfig {
                grid_dim: (kv_heads as u32, seq_len as u32, 1),
                block_dim: (head_dim as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let stream = self.ctx.default_stream();
            cuda_profile!(self, CudaOpTag::KvScatter, stream, {
                unsafe {
                    stream
                        .launch_builder(&self.kernels.kv_scatter_batch)
                        .arg(&ks)
                        .arg(&vs)
                        .arg(&kd)
                        .arg(&vd)
                        .arg(&kvh)
                        .arg(&hd)
                        .arg(&cap)
                        .arg(&wps)
                        .arg(&sl)
                        .launch(cfg)
                        .map_err(|e| anyhow!("kv_scatter_batch launch failed: {e}"))?;
                }
                Ok(())
            });
            self.synchronize()?;
            Ok(())
        } else {
            self.synchronize()?;
            cpu_fallback().kv_scatter_f32_to_f16_batch(
                k_src,
                v_src,
                k_dst,
                v_dst,
                kv_heads,
                head_dim,
                capacity,
                write_pos_start,
                seq_len,
            )
        }
    }

    fn gather(&self, src: &Tensor, indices: &Tensor, dst: &mut Tensor) -> Result<()> {
        // GPU kernel only supports F16 embedding -> F32 output.
        // For other dtypes, sync and fall back to CPU.
        if src.dtype() != DType::F16 {
            self.synchronize()?;
            return cpu_fallback().gather(src, indices, dst);
        }

        let src_ptr = Self::get_device_ptr(src.buffer().as_ref());
        let idx_ptr = Self::get_device_ptr(indices.buffer().as_ref());
        let dst_ptr = Self::get_device_ptr(dst.buffer().as_ref());

        if let (Some(sp), Some(ip), Some(dp)) = (src_ptr, idx_ptr, dst_ptr) {
            let dims = dst.shape().dims().to_vec();
            let rank = dims.len();
            let dim = dims[rank - 1];
            let n_tokens: usize = dims[..rank - 1].iter().product::<usize>().max(1);
            let dim_i32 = dim as i32;
            let y_blocks = dim.div_ceil(256) as u32;
            let cfg = LaunchConfig {
                grid_dim: (n_tokens as u32, y_blocks, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let stream = self.ctx.default_stream();
            // SAFETY: sp is valid F16 embed device ptr; ip is valid i32 indices; dp is valid F32 output.
            cuda_profile!(self, CudaOpTag::Gather, stream, {
                unsafe {
                    stream
                        .launch_builder(&self.kernels.gather_f16)
                        .arg(&sp)
                        .arg(&ip)
                        .arg(&dp)
                        .arg(&dim_i32)
                        .launch(cfg)
                        .map_err(|e| anyhow!("gather_f16 kernel launch failed: {e}"))?;
                }
                Ok(())
            });
            self.synchronize()?;
            Ok(())
        } else {
            self.synchronize()?;
            cpu_fallback().gather(src, indices, dst)
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
        scores_out: Option<&mut [f32]>,
    ) -> Result<()> {
        let kv_dtype = k_cache.dtype();
        let q_ptr = Self::get_device_ptr(q.buffer().as_ref());
        let k_ptr = Self::get_device_ptr(k_cache.buffer().as_ref());
        let v_ptr = Self::get_device_ptr(v_cache.buffer().as_ref());
        let out_ptr = Self::get_device_ptr(out.buffer().as_ref());

        // If scores_out is requested or device pointers are unavailable, use CPU fallback.
        if scores_out.is_some() {
            self.synchronize()?;
            return cpu_fallback().attention_gen(
                q,
                k_cache,
                v_cache,
                out,
                num_heads_q,
                num_heads_kv,
                head_dim,
                cache_seq_len,
                scores_out,
            );
        }

        if let (Some(qp), Some(kp), Some(vp), Some(op)) = (q_ptr, k_ptr, v_ptr, out_ptr) {
            // Shared memory: cache_seq_len floats for scores
            let shmem = (cache_seq_len * std::mem::size_of::<f32>()) as u32;
            let block_size = 256u32;
            let cfg = LaunchConfig {
                grid_dim: (num_heads_q as u32, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: shmem,
            };
            let nhq = num_heads_q as i32;
            let nkv = num_heads_kv as i32;
            let hd = head_dim as i32;
            // KV cache is HeadMajor: capacity is derived from the k_cache tensor shape.
            let k_dims = k_cache.shape().dims().to_vec();
            let cap = if k_dims.len() >= 3 {
                k_dims[k_dims.len() - 2] as i32
            } else {
                cache_seq_len as i32
            };
            let csl = cache_seq_len as i32;
            let stream = self.ctx.default_stream();

            match kv_dtype {
                DType::F32 => {
                    // SAFETY: qp, kp, vp, op are valid F32 device ptrs with correct dimensions.
                    cuda_profile!(self, CudaOpTag::Attention, stream, {
                        unsafe {
                            stream
                                .launch_builder(&self.kernels.flash_attn_f32)
                                .arg(&qp)
                                .arg(&kp)
                                .arg(&vp)
                                .arg(&op)
                                .arg(&nhq)
                                .arg(&nkv)
                                .arg(&hd)
                                .arg(&cap)
                                .arg(&csl)
                                .launch(cfg)
                                .map_err(|e| {
                                    anyhow!("attention_gen_f32 kernel launch failed: {e}")
                                })?;
                        }
                        Ok(())
                    });
                    self.synchronize()?;
                    Ok(())
                }
                DType::F16 => {
                    // SAFETY: kp and vp are valid F16 device ptrs; qp and op are F32.
                    cuda_profile!(self, CudaOpTag::Attention, stream, {
                        unsafe {
                            stream
                                .launch_builder(&self.kernels.flash_attn_f16kv)
                                .arg(&qp)
                                .arg(&kp)
                                .arg(&vp)
                                .arg(&op)
                                .arg(&nhq)
                                .arg(&nkv)
                                .arg(&hd)
                                .arg(&cap)
                                .arg(&csl)
                                .launch(cfg)
                                .map_err(|e| {
                                    anyhow!("attention_gen_f16kv kernel launch failed: {e}")
                                })?;
                        }
                        Ok(())
                    });
                    self.synchronize()?;
                    Ok(())
                }
                _ => {
                    // Q4_0 or unsupported: sync then CPU fallback
                    self.synchronize()?;
                    cpu_fallback().attention_gen(
                        q,
                        k_cache,
                        v_cache,
                        out,
                        num_heads_q,
                        num_heads_kv,
                        head_dim,
                        cache_seq_len,
                        None,
                    )
                }
            }
        } else {
            // Missing device pointers: sync then CPU fallback
            self.synchronize()?;
            cpu_fallback().attention_gen(
                q,
                k_cache,
                v_cache,
                out,
                num_heads_q,
                num_heads_kv,
                head_dim,
                cache_seq_len,
                None,
            )
        }
    }

    // --- Memory ops ---

    fn synchronize(&self) -> Result<()> {
        self.ctx
            .default_stream()
            .synchronize()
            .map_err(|e| anyhow!("CUDA synchronize failed: {e}"))
    }

    fn read_buffer(&self, t: &Tensor, dst: &mut [u8]) -> Result<()> {
        self.synchronize()?;
        if let Some(db) = t.buffer().as_any().downcast_ref::<CudaDeviceBuffer>() {
            db.copy_to_host(dst.as_mut_ptr(), dst.len())?;
            return Ok(());
        }
        let src_ptr = t.buffer().as_ptr();
        if src_ptr.is_null() {
            anyhow::bail!("Cannot read null buffer");
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn copy_from(&self, src: &Tensor) -> Result<Tensor> {
        self.synchronize()?;
        let size = src.size();
        let src_ptr = src.as_ptr();

        if self.is_discrete_gpu() {
            // Discrete GPU: use managed memory (cuMemAllocManaged).
            // CUDA driver auto-migrates pages to VRAM on first GPU access.
            let managed_buf = CudaBuffer::new(size, src.dtype())?;
            if !src_ptr.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(src_ptr, managed_buf.as_mut_ptr(), size);
                }
            }
            Ok(Tensor::new(
                src.shape().clone(),
                Arc::new(managed_buf),
                Arc::new(self.clone()),
            ))
        } else {
            // UMA (Jetson): pinned host memory is zero-copy.
            let cuda_buf = CudaHostBuffer::new(size, src.dtype())?;
            let dst_ptr = cuda_buf.as_mut_ptr();
            if !src_ptr.is_null() && !dst_ptr.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
                }
            }
            Ok(Tensor::new(
                src.shape().clone(),
                Arc::new(cuda_buf),
                Arc::new(self.clone()),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    /// Reference implementation of tiled online-softmax flash attention (matches CUDA kernel logic).
    /// Used to verify numerical stability invariants without requiring a GPU.
    fn ref_flash_attn_prefill(
        q: &[f32],       // [seq_len, n_heads_q, head_dim]
        k: &[f32],       // [n_heads_kv, capacity, head_dim]
        v: &[f32],       // [n_heads_kv, capacity, head_dim]
        out: &mut [f32], // [seq_len, n_heads_q, head_dim]
        n_heads_q: usize,
        n_heads_kv: usize,
        seq_len: usize,
        cache_seq_len: usize,
        kv_capacity: usize,
        head_dim: usize,
        block_n: usize, // KV tile size
    ) {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = n_heads_q / n_heads_kv;

        for s in 0..seq_len {
            let causal_limit = (cache_seq_len as i64) - (seq_len as i64) + (s as i64);

            for h_q in 0..n_heads_q {
                let h_kv = h_q / gqa_ratio;
                let q_off = (s * n_heads_q + h_q) * head_dim;

                let mut o_acc = vec![0.0f32; head_dim];
                let mut m_i: f32 = f32::NEG_INFINITY;
                let mut l_i: f32 = 0.0;

                // Process KV in tiles of block_n, pairs of 2
                for kv_start in (0..cache_seq_len).step_by(block_n) {
                    let kv_end = (kv_start + block_n).min(cache_seq_len);
                    let tile_len = kv_end - kv_start;

                    let mut p = 0;
                    while p + 1 < tile_len {
                        let kp0 = kv_start + p;
                        let kp1 = kv_start + p + 1;

                        let s0 = if (kp0 as i64) <= causal_limit {
                            let k_off = h_kv * kv_capacity * head_dim + kp0 * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q[q_off + d] * k[k_off + d];
                            }
                            dot * scale
                        } else {
                            f32::NEG_INFINITY
                        };

                        let s1 = if (kp1 as i64) <= causal_limit {
                            let k_off = h_kv * kv_capacity * head_dim + kp1 * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q[q_off + d] * k[k_off + d];
                            }
                            dot * scale
                        } else {
                            f32::NEG_INFINITY
                        };

                        let m_new = m_i.max(s0.max(s1));
                        if m_new > f32::NEG_INFINITY {
                            let exp0 = (s0 - m_new).exp();
                            let exp1 = (s1 - m_new).exp();
                            let rescale = if m_i > f32::NEG_INFINITY {
                                (m_i - m_new).exp()
                            } else {
                                0.0
                            };
                            l_i = l_i * rescale + exp0 + exp1;
                            for d in 0..head_dim {
                                let v0 = v[h_kv * kv_capacity * head_dim + kp0 * head_dim + d];
                                let v1 = v[h_kv * kv_capacity * head_dim + kp1 * head_dim + d];
                                o_acc[d] = o_acc[d] * rescale + exp0 * v0 + exp1 * v1;
                            }
                            m_i = m_new;
                        }
                        p += 2;
                    }

                    // Odd remainder
                    if p < tile_len {
                        let kp = kv_start + p;
                        let s = if (kp as i64) <= causal_limit {
                            let k_off = h_kv * kv_capacity * head_dim + kp * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q[q_off + d] * k[k_off + d];
                            }
                            dot * scale
                        } else {
                            f32::NEG_INFINITY
                        };

                        let m_new = m_i.max(s);
                        if m_new > f32::NEG_INFINITY {
                            let exp_s = (s - m_new).exp();
                            let rescale = if m_i > f32::NEG_INFINITY {
                                (m_i - m_new).exp()
                            } else {
                                0.0
                            };
                            l_i = l_i * rescale + exp_s;
                            for d in 0..head_dim {
                                o_acc[d] = o_acc[d] * rescale
                                    + exp_s * v[h_kv * kv_capacity * head_dim + kp * head_dim + d];
                            }
                            m_i = m_new;
                        }
                    }
                }

                let o_off = (s * n_heads_q + h_q) * head_dim;
                if l_i > 0.0 {
                    let inv_l = 1.0 / l_i;
                    for d in 0..head_dim {
                        out[o_off + d] = o_acc[d] * inv_l;
                    }
                } else {
                    for d in 0..head_dim {
                        out[o_off + d] = 0.0;
                    }
                }
            }
        }
    }

    /// Naive attention (no tiling) for reference comparison.
    fn ref_naive_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        n_heads_q: usize,
        n_heads_kv: usize,
        seq_len: usize,
        cache_seq_len: usize,
        kv_capacity: usize,
        head_dim: usize,
    ) {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = n_heads_q / n_heads_kv;
        for s in 0..seq_len {
            let causal_limit = (cache_seq_len as i64) - (seq_len as i64) + (s as i64);
            for h_q in 0..n_heads_q {
                let h_kv = h_q / gqa_ratio;
                let q_off = (s * n_heads_q + h_q) * head_dim;
                // Compute all scores
                let mut scores = vec![f32::NEG_INFINITY; cache_seq_len];
                for t in 0..cache_seq_len {
                    if (t as i64) <= causal_limit {
                        let k_off = h_kv * kv_capacity * head_dim + t * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_off + d] * k[k_off + d];
                        }
                        scores[t] = dot * scale;
                    }
                }
                // Softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                let mut exps = vec![0.0f32; cache_seq_len];
                if max_s > f32::NEG_INFINITY {
                    for t in 0..cache_seq_len {
                        exps[t] = (scores[t] - max_s).exp();
                        sum_exp += exps[t];
                    }
                }
                // Weighted sum
                let o_off = (s * n_heads_q + h_q) * head_dim;
                if sum_exp > 0.0 {
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for t in 0..cache_seq_len {
                            acc += exps[t] * v[h_kv * kv_capacity * head_dim + t * head_dim + d];
                        }
                        out[o_off + d] = acc / sum_exp;
                    }
                } else {
                    for d in 0..head_dim {
                        out[o_off + d] = 0.0;
                    }
                }
            }
        }
    }

    // --- Test cases ---

    #[test]
    fn test_flash_prefill_no_nan_basic() {
        // Basic case: seq_len=4, head_dim=64, 2 heads, tile_size=8
        let head_dim = 64;
        let seq_len = 4;
        let n_heads_q = 2;
        let n_heads_kv = 2;
        let capacity = 16;
        let cache_seq_len = seq_len; // fresh prefill

        let q = vec![0.1f32; seq_len * n_heads_q * head_dim];
        let k = vec![0.2f32; n_heads_kv * capacity * head_dim];
        let v = vec![0.3f32; n_heads_kv * capacity * head_dim];
        let mut out_flash = vec![0.0f32; seq_len * n_heads_q * head_dim];
        let mut out_naive = vec![0.0f32; seq_len * n_heads_q * head_dim];

        ref_flash_attn_prefill(
            &q,
            &k,
            &v,
            &mut out_flash,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
            8,
        );
        ref_naive_attention(
            &q,
            &k,
            &v,
            &mut out_naive,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
        );

        for i in 0..out_flash.len() {
            assert!(!out_flash[i].is_nan(), "NaN at index {i}");
            assert!(
                (out_flash[i] - out_naive[i]).abs() < 1e-5,
                "mismatch at {i}: flash={} naive={}",
                out_flash[i],
                out_naive[i]
            );
        }
    }

    #[test]
    fn test_flash_prefill_no_nan_head_dim_256() {
        // Gemma3 4B config: head_dim=256, GQA
        let head_dim = 256;
        let seq_len = 7; // NOT multiple of BLOCK_M=32 -> tests padding
        let n_heads_q = 8;
        let n_heads_kv = 4; // GQA ratio 2
        let capacity = 32;
        let cache_seq_len = seq_len;

        let q: Vec<f32> = (0..seq_len * n_heads_q * head_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let k: Vec<f32> = (0..n_heads_kv * capacity * head_dim)
            .map(|i| ((i % 83) as f32 - 41.0) * 0.01)
            .collect();
        let v: Vec<f32> = (0..n_heads_kv * capacity * head_dim)
            .map(|i| ((i % 71) as f32 - 35.0) * 0.01)
            .collect();
        let mut out_flash = vec![0.0f32; seq_len * n_heads_q * head_dim];
        let mut out_naive = vec![0.0f32; seq_len * n_heads_q * head_dim];

        ref_flash_attn_prefill(
            &q,
            &k,
            &v,
            &mut out_flash,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
            8,
        );
        ref_naive_attention(
            &q,
            &k,
            &v,
            &mut out_naive,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
        );

        for i in 0..out_flash.len() {
            assert!(!out_flash[i].is_nan(), "NaN at index {i}");
            assert!(
                (out_flash[i] - out_naive[i]).abs() < 1e-4,
                "mismatch at {i}: flash={} naive={}",
                out_flash[i],
                out_naive[i]
            );
        }
    }

    #[test]
    fn test_flash_prefill_no_nan_single_token() {
        // Edge case: seq_len=1 (single token prefill)
        let head_dim = 64;
        let seq_len = 1;
        let n_heads_q = 4;
        let n_heads_kv = 4;
        let capacity = 8;
        let cache_seq_len = 1;

        let q = vec![1.0f32; seq_len * n_heads_q * head_dim];
        let k = vec![1.0f32; n_heads_kv * capacity * head_dim];
        let v = vec![0.5f32; n_heads_kv * capacity * head_dim];
        let mut out = vec![0.0f32; seq_len * n_heads_q * head_dim];

        ref_flash_attn_prefill(
            &q,
            &k,
            &v,
            &mut out,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
            8,
        );

        for i in 0..out.len() {
            assert!(!out[i].is_nan(), "NaN at index {i}");
            // With single token, output should equal v (softmax of single element = 1.0)
            assert!(
                (out[i] - 0.5).abs() < 1e-5,
                "expected 0.5, got {} at {i}",
                out[i]
            );
        }
    }

    #[test]
    fn test_flash_prefill_no_nan_odd_tile_boundary() {
        // cache_seq_len that creates odd tile remainder for all block_n values
        for &block_n in &[8, 16, 32] {
            let head_dim = 64;
            let seq_len = 5;
            let n_heads_q = 2;
            let n_heads_kv = 2;
            let cache_seq_len = block_n + 3; // ensures odd remainder
            let capacity = cache_seq_len + 4;

            let q: Vec<f32> = (0..seq_len * n_heads_q * head_dim)
                .map(|i| (i as f32 * 0.01).sin())
                .collect();
            let k: Vec<f32> = (0..n_heads_kv * capacity * head_dim)
                .map(|i| (i as f32 * 0.02).cos())
                .collect();
            let v: Vec<f32> = (0..n_heads_kv * capacity * head_dim)
                .map(|i| (i as f32 * 0.03).sin())
                .collect();
            let mut out_flash = vec![0.0f32; seq_len * n_heads_q * head_dim];
            let mut out_naive = vec![0.0f32; seq_len * n_heads_q * head_dim];

            ref_flash_attn_prefill(
                &q,
                &k,
                &v,
                &mut out_flash,
                n_heads_q,
                n_heads_kv,
                seq_len,
                cache_seq_len,
                capacity,
                head_dim,
                block_n,
            );
            ref_naive_attention(
                &q,
                &k,
                &v,
                &mut out_naive,
                n_heads_q,
                n_heads_kv,
                seq_len,
                cache_seq_len,
                capacity,
                head_dim,
            );

            for i in 0..out_flash.len() {
                assert!(!out_flash[i].is_nan(), "NaN at block_n={block_n} index {i}");
                assert!(
                    (out_flash[i] - out_naive[i]).abs() < 1e-4,
                    "mismatch at block_n={block_n} index {i}: flash={} naive={}",
                    out_flash[i],
                    out_naive[i]
                );
            }
        }
    }

    #[test]
    fn test_flash_prefill_no_nan_all_masked() {
        // Edge case: cache_seq_len=0 -- no KV to attend, should produce zeros not NaN
        let head_dim = 64;
        let seq_len = 4;
        let n_heads_q = 2;
        let n_heads_kv = 2;
        let capacity = 8;
        let cache_seq_len = 0;

        let q = vec![1.0f32; seq_len * n_heads_q * head_dim];
        let k = vec![0.0f32; n_heads_kv * capacity * head_dim];
        let v = vec![0.0f32; n_heads_kv * capacity * head_dim];
        let mut out = vec![f32::NAN; seq_len * n_heads_q * head_dim]; // init with NaN to detect untouched

        ref_flash_attn_prefill(
            &q,
            &k,
            &v,
            &mut out,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
            8,
        );

        for i in 0..out.len() {
            assert!(!out[i].is_nan(), "NaN at index {i} (all-masked case)");
            assert_eq!(out[i], 0.0, "expected 0.0 for all-masked at index {i}");
        }
    }

    #[test]
    fn test_flash_prefill_continuation_context() {
        // Continuation: existing KV cache + new tokens
        let head_dim = 128;
        let seq_len = 3; // new tokens
        let n_heads_q = 4;
        let n_heads_kv = 2; // GQA
        let capacity = 32;
        let cache_seq_len = 10; // already 7 cached + 3 new = 10

        let q: Vec<f32> = (0..seq_len * n_heads_q * head_dim)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let k: Vec<f32> = (0..n_heads_kv * capacity * head_dim)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let v: Vec<f32> = (0..n_heads_kv * capacity * head_dim)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let mut out_flash = vec![0.0f32; seq_len * n_heads_q * head_dim];
        let mut out_naive = vec![0.0f32; seq_len * n_heads_q * head_dim];

        ref_flash_attn_prefill(
            &q,
            &k,
            &v,
            &mut out_flash,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
            16,
        );
        ref_naive_attention(
            &q,
            &k,
            &v,
            &mut out_naive,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            capacity,
            head_dim,
        );

        for i in 0..out_flash.len() {
            assert!(!out_flash[i].is_nan(), "NaN at index {i}");
            assert!(
                (out_flash[i] - out_naive[i]).abs() < 1e-4,
                "mismatch at {i}: flash={} naive={}",
                out_flash[i],
                out_naive[i]
            );
        }
    }
}
