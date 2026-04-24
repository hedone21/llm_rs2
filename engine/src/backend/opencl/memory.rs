use super::buffer::OpenCLBuffer;
use crate::buffer::unified_buffer::UnifiedBuffer;
use crate::core::buffer::{Buffer, DType};
use crate::core::memory::Memory;
use anyhow::Result;
use ocl::flags::MemFlags;
use ocl::{Context, Queue};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

static KV_ALLOC_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Returns `true` if `LLMRS_LOG_WEIGHT_VMA` is set.
/// Cached after first call — no per-alloc env lookup.
fn log_weight_vma_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LLMRS_LOG_WEIGHT_VMA").is_ok())
}

pub struct OpenCLMemory {
    #[allow(dead_code)]
    context: Context,
    queue: Queue,
    used_memory: Mutex<usize>,
    /// If true, use UnifiedBuffer (zero-copy shared memory)
    /// If false, use OpenCLBuffer (device-only, faster)
    use_zero_copy: bool,
}

impl OpenCLMemory {
    pub fn new(context: Context, queue: Queue, use_zero_copy: bool) -> Self {
        Self {
            context,
            queue,
            used_memory: Mutex::new(0),
            use_zero_copy,
        }
    }
}

impl Memory for OpenCLMemory {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        let buffer: Arc<dyn Buffer> = if self.use_zero_copy {
            // Zero-copy shared memory (CPU-GPU accessible, but slower GPU kernels)
            let ub = UnifiedBuffer::new(self.queue.clone(), size, dtype)?;
            if log_weight_vma_enabled() {
                // cl_mem pointer: obtained via cl_mem().as_ptr() on the Mem wrapper.
                // host_ptr: map briefly to get the VMA address, then unmap immediately.
                // The map/unmap here is diagnostic-only; it does not affect GPU-read state
                // because the buffer is freshly allocated (no GPU writes in flight yet).
                let cl_hex = ub
                    .cl_mem()
                    .map(|m| format!("{:#x}", m.as_ptr() as usize))
                    .unwrap_or_else(|| "null".to_string());
                let host_hex = match ub.map() {
                    Ok(p) => {
                        let s = format!("{p:p}");
                        let _ = ub.unmap();
                        s
                    }
                    Err(_) => "null".to_string(),
                };
                eprintln!(
                    "[VMA] alloc path=zero_copy size={size} dtype={dtype:?} \
                     cl_mem={cl_hex} host_ptr={host_hex}"
                );
            }
            Arc::new(ub)
        } else {
            // Device-only memory (faster GPU kernels, requires explicit copies)
            let ocl_buffer = ocl::Buffer::<u8>::builder()
                .queue(self.queue.clone())
                .flags(MemFlags::new().read_write())
                .len(size)
                .build()?;
            let ob = OpenCLBuffer::new(self.queue.clone(), ocl_buffer, size, dtype)?;
            if log_weight_vma_enabled() {
                let cl_hex = ob
                    .cl_mem()
                    .map(|m| format!("{:#x}", m.as_ptr() as usize))
                    .unwrap_or_else(|| "null".to_string());
                eprintln!(
                    "[VMA] alloc path=device_only size={size} dtype={dtype:?} \
                     cl_mem={cl_hex} host_ptr=null"
                );
            }
            Arc::new(ob)
        };

        {
            let mut mem = self.used_memory.lock().unwrap();
            *mem += size;
        }

        Ok(buffer)
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
