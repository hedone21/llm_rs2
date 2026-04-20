use super::buffer::OpenCLBuffer;
use super::qcom_ext::{QcomCachePolicy, QcomCapabilities};
use crate::buffer::unified_buffer::UnifiedBuffer;
use crate::core::buffer::{Buffer, DType};
use crate::core::memory::Memory;
use anyhow::Result;
use ocl::flags::MemFlags;
use ocl::{Context, Queue};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

static KV_ALLOC_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Emit the `LLMRS_QCOM_IOCOHERENT=1` banner once so we can confirm
/// from device logs that the PoC path actually engaged.
fn log_iocoherent_request_once() {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        eprintln!(
            "[QCOM] LLMRS_QCOM_IOCOHERENT=1: requesting iocoherent host-cache policy for alloc_with_policy calls"
        );
    });
}

pub struct OpenCLMemory {
    #[allow(dead_code)]
    context: Context,
    queue: Queue,
    used_memory: Mutex<usize>,
    /// If true, use UnifiedBuffer (zero-copy shared memory)
    /// If false, use OpenCLBuffer (device-only, faster)
    use_zero_copy: bool,
    /// Qualcomm extension capabilities cached from the backend probe.
    /// Defaults to the all-false struct when not supplied (e.g.
    /// legacy construction paths / test harness) — in which case any
    /// `alloc_with_policy` call with a non-Default policy silently
    /// falls back to the standard `CL_MEM_ALLOC_HOST_PTR` path.
    qcom_caps: QcomCapabilities,
}

impl OpenCLMemory {
    pub fn new(context: Context, queue: Queue, use_zero_copy: bool) -> Self {
        Self {
            context,
            queue,
            used_memory: Mutex::new(0),
            use_zero_copy,
            qcom_caps: QcomCapabilities::default(),
        }
    }

    /// Construct an `OpenCLMemory` with a known Qualcomm capability
    /// set. Used by `OpenCLBackend::copy_from` and similar so that
    /// `alloc_with_policy` can honour iocoherent requests on Adreno.
    pub fn new_with_caps(
        context: Context,
        queue: Queue,
        use_zero_copy: bool,
        qcom_caps: QcomCapabilities,
    ) -> Self {
        Self {
            context,
            queue,
            used_memory: Mutex::new(0),
            use_zero_copy,
            qcom_caps,
        }
    }

    /// Internal: decide whether a given policy is actually honourable
    /// on the current device. Returns the effective policy.
    fn effective_policy(&self, requested: QcomCachePolicy) -> QcomCachePolicy {
        match requested {
            QcomCachePolicy::Default => QcomCachePolicy::Default,
            QcomCachePolicy::IoCoherent => {
                if self.qcom_caps.supports_iocoherent() {
                    QcomCachePolicy::IoCoherent
                } else {
                    log::debug!(
                        "QcomCachePolicy::IoCoherent requested but device lacks \
                         cl_qcom_ext_host_ptr_iocoherent — falling back to Default"
                    );
                    QcomCachePolicy::Default
                }
            }
            // All other host_cache_policy variants also need the base
            // cl_qcom_ext_host_ptr extension.
            _ => {
                if self.qcom_caps.ext_host_ptr {
                    requested
                } else {
                    log::debug!(
                        "QcomCachePolicy::{:?} requested but device lacks cl_qcom_ext_host_ptr — falling back to Default",
                        requested
                    );
                    QcomCachePolicy::Default
                }
            }
        }
    }

    /// Allocate a GPU buffer with an explicit Qualcomm host-cache
    /// policy. On Adreno devices that expose
    /// `cl_qcom_ext_host_ptr[_iocoherent]`, this uses the extension
    /// path; on all other devices it silently falls back to the
    /// standard `alloc()` behaviour.
    ///
    /// Milestone 1 of the Doppeladler adoption plan — the production
    /// forward paths still use `alloc()` / `alloc_kv()` and are
    /// unchanged. This entry point exists so PoC / microbench code
    /// can exercise the iocoherent allocation.
    pub fn alloc_with_policy(
        &self,
        size: usize,
        dtype: DType,
        policy: QcomCachePolicy,
    ) -> Result<Arc<dyn Buffer>> {
        let effective = self.effective_policy(policy);

        let buffer: Arc<dyn Buffer> = match effective {
            QcomCachePolicy::Default => {
                // Existing zero-copy / device-only branch.
                if self.use_zero_copy {
                    Arc::new(UnifiedBuffer::new(self.queue.clone(), size, dtype)?)
                } else {
                    let ocl_buffer = ocl::Buffer::<u8>::builder()
                        .queue(self.queue.clone())
                        .flags(MemFlags::new().read_write())
                        .len(size)
                        .build()?;
                    Arc::new(OpenCLBuffer::new(
                        self.queue.clone(),
                        ocl_buffer,
                        size,
                        dtype,
                    )?)
                }
            }
            _ => {
                if std::env::var_os("LLMRS_QCOM_IOCOHERENT").is_some() {
                    log_iocoherent_request_once();
                }
                // Try the extension path; on driver failure we
                // gracefully fall back to the standard path so a
                // single bad allocation does not abort inference.
                let page_size_hint = self.qcom_caps.page_size_bytes;
                match UnifiedBuffer::new_with_cache_policy(
                    self.queue.clone(),
                    size,
                    dtype,
                    effective,
                    page_size_hint,
                ) {
                    Ok(buf) => Arc::new(buf),
                    Err(e) => {
                        log::warn!(
                            "alloc_with_policy({:?}) failed: {} — falling back to Default",
                            effective,
                            e
                        );
                        Arc::new(UnifiedBuffer::new(self.queue.clone(), size, dtype)?)
                    }
                }
            }
        };

        {
            let mut mem = self.used_memory.lock().unwrap();
            *mem += size;
        }

        Ok(buffer)
    }
}

impl Memory for OpenCLMemory {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        // Production path is unchanged — Milestone 1 does NOT route
        // any existing alloc through the extension. Opt-in only via
        // alloc_with_policy().
        self.alloc_with_policy(size, dtype, QcomCachePolicy::Default)
    }

    fn alloc_kv(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        if self.use_zero_copy {
            // KV cache: CL_MEM_ALLOC_HOST_PTR (driver-managed, single VMA).
            // Starts UNMAPPED — GPU writes via cl_mem during forward pass.
            // Map for CPU access only during SwitchHw (via kv_migrate).
            // Note: OpenCL spec says writing to a mapped buffer via cl_mem is UB.
            let buffer = UnifiedBuffer::new(self.queue.clone(), size, dtype)?;

            if std::env::var_os("LLMRS_LOG_KV_VMA").is_some() {
                let idx = KV_ALLOC_COUNTER.fetch_add(1, Ordering::Relaxed);
                match buffer.map() {
                    Ok(host_ptr) => {
                        eprintln!(
                            "[KV_VMA] idx={:3} host_ptr={:p} size={}",
                            idx, host_ptr, size
                        );
                        let _ = buffer.unmap();
                    }
                    Err(e) => {
                        eprintln!("[KV_VMA] idx={} map failed: {}", idx, e);
                    }
                }
            }

            let buf: Arc<dyn Buffer> = Arc::new(buffer);
            {
                let mut mem = self.used_memory.lock().unwrap();
                *mem += size;
            }
            Ok(buf)
        } else {
            self.alloc(size, dtype)
        }
    }

    fn used_memory(&self) -> usize {
        *self.used_memory.lock().unwrap()
    }
}
