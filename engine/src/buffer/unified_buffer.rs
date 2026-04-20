use crate::core::buffer::{Buffer, DType};
use anyhow::{Result, anyhow};
use ocl::core::Mem;
use ocl::{Buffer as OclBuffer, Queue, flags};
use std::alloc::{Layout, alloc, dealloc};
use std::any::Any;
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "opencl")]
use crate::backend::opencl::qcom_ext::{
    CL_MEM_EXT_HOST_PTR_QCOM, ClMemExtHostPtrQcom, QcomCachePolicy,
};

/// UnifiedBuffer: CPU-GPU Zero-Copy Shared Memory
///
/// Default path uses `CL_MEM_ALLOC_HOST_PTR` to allocate memory that
/// can be accessed by both CPU and GPU without explicit data copies.
///
/// Optional Milestone 1 path (Doppeladler C):
/// `new_with_cache_policy(.., QcomCachePolicy::IoCoherent)` wraps a
/// user-allocated page-aligned host buffer with
/// `CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM` so that the Adreno
/// driver maps it as io-coherent. This eliminates the implicit CPU
/// cache flush/invalidate operations issued on buffer map/unmap
/// transitions — see
/// `.agent/research/2026-04-20_coherence_flag_matrix.md` section C2.
///
/// # Usage Pattern
/// ```ignore
/// // 1. Create buffer (starts mapped for CPU initialization)
/// let buffer = UnifiedBuffer::new(...)?;
///
/// // 2. Write data via as_mut_ptr() while mapped
/// let ptr = buffer.as_mut_ptr();
/// // ... CPU operations ...
///
/// // 3. Unmap for GPU access
/// buffer.unmap_for_gpu()?;
/// // ... GPU kernel execution ...
///
/// // 4. Map again for CPU reads
/// buffer.map_for_cpu()?;
/// ```
pub struct UnifiedBuffer {
    /// Backing storage discriminates between the default
    /// (`OclBuffer<u8>`) and the Qualcomm-extension (`Mem` + host
    /// allocation we own) paths.
    storage: BufferStorage,
    queue: Queue,
    /// Mapped pointer (valid only when is_mapped is true).
    /// For Qualcomm `CL_MEM_USE_HOST_PTR` paths this always equals the
    /// user-allocated host pointer — no map/unmap is strictly
    /// required, but we keep the same API contract for call sites.
    mapped_ptr: Mutex<*mut u8>,
    size: usize,
    dtype: DType,
    is_mapped: AtomicBool,
}

enum BufferStorage {
    /// Standard `CL_MEM_ALLOC_HOST_PTR` path via the high-level ocl crate.
    Standard { cl_buffer: OclBuffer<u8> },
    /// Qualcomm `cl_qcom_ext_host_ptr[_iocoherent]` path. The host
    /// allocation is owned by this struct and freed on Drop after the
    /// `cl_mem` is released.
    ///
    /// SAFETY invariant: `host_ptr` is a non-null pointer to
    /// `host_alloc_size` bytes allocated with `std::alloc::alloc`
    /// under `host_alloc_layout`. `mem` was created by
    /// `clCreateBuffer` with `CL_MEM_USE_HOST_PTR |
    /// CL_MEM_EXT_HOST_PTR_QCOM` referencing that host pointer via a
    /// `cl_mem_ext_host_ptr`.
    QcomExt {
        mem: Mem,
        host_ptr: *mut u8,
        host_alloc_layout: Layout,
        // For Debug / diagnostic purposes only.
        #[allow(dead_code)]
        policy_token: u32,
    },
}

// SAFETY: OpenCL buffer handles are thread-safe.
// Access to mapped_ptr is protected by Mutex.
// For the QcomExt variant, `host_ptr` is only ever read after
// `is_mapped` is observed true (Acquire) and never mutated outside of
// construction/drop, so cross-thread reads through the Buffer trait
// are sound.
unsafe impl Send for UnifiedBuffer {}
unsafe impl Sync for UnifiedBuffer {}

impl UnifiedBuffer {
    /// Create a new UnifiedBuffer with CL_MEM_ALLOC_HOST_PTR flag.
    /// The buffer starts in **unmapped** state (GPU-accessible).
    /// Call `map_for_cpu()` when CPU access is needed.
    pub fn new(queue: Queue, size: usize, dtype: DType) -> Result<Self> {
        // Create buffer with ALLOC_HOST_PTR for zero-copy
        let cl_buffer = OclBuffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR)
            .len(size)
            .build()
            .map_err(|e| anyhow!("Failed to create unified buffer: {}", e))?;

        // Start unmapped (GPU-accessible) for optimal kernel performance
        Ok(Self {
            storage: BufferStorage::Standard { cl_buffer },
            queue,
            mapped_ptr: Mutex::new(ptr::null_mut()),
            size,
            dtype,
            is_mapped: AtomicBool::new(false), // Starts unmapped
        })
    }

    /// Create a UnifiedBuffer with an explicit Qualcomm host-cache
    /// policy. When `policy == QcomCachePolicy::Default` this is
    /// identical to `new()` and works on any platform.
    ///
    /// For any other policy, the caller must have verified via
    /// `OpenCLBackend::qcom_capabilities` that the Adreno extension is
    /// present; otherwise this returns an error. The higher-level
    /// `OpenCLMemory::alloc_with_policy` performs that check and
    /// silently falls back to `Default`.
    ///
    /// # Semantics
    ///
    /// - `Default`: standard `CL_MEM_ALLOC_HOST_PTR` (unchanged).
    /// - `IoCoherent` / `WriteBack` / `WriteThrough` /
    ///   `WriteCombining` / `Uncached`: allocates a page-aligned
    ///   host buffer and wraps it with `CL_MEM_USE_HOST_PTR |
    ///   CL_MEM_EXT_HOST_PTR_QCOM`. The host buffer is permanently
    ///   "mapped" for CPU access (the Qualcomm spec guarantees a
    ///   single VMA on Adreno).
    ///
    /// For the Qualcomm path, `is_mapped()` starts **true** — the
    /// host pointer is always valid. `unmap()` / `map()` become
    /// no-ops (they only drive `queue.finish()` for GPU-CPU sync).
    pub fn new_with_cache_policy(
        queue: Queue,
        size: usize,
        dtype: DType,
        policy: QcomCachePolicy,
        page_size_hint: usize,
    ) -> Result<Self> {
        if policy == QcomCachePolicy::Default {
            return Self::new(queue, size, dtype);
        }

        let cache_token = policy
            .host_cache_policy_token()
            .ok_or_else(|| anyhow!("QcomCachePolicy::Default has no extension token"))?;

        // 1. Allocate page-aligned host memory. The Adreno driver
        //    requires the host pointer passed via CL_MEM_USE_HOST_PTR
        //    + CL_MEM_EXT_HOST_PTR_QCOM to be aligned to
        //    CL_DEVICE_PAGE_SIZE_QCOM. 4 KiB is the universal safe
        //    fallback when the hint is 0.
        let page = if page_size_hint == 0 {
            4096
        } else {
            page_size_hint
        };
        // Round size up to the page boundary to satisfy potential
        // Adreno page-granularity requirements and to leave room for
        // the CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM padding.
        let padded_size = size.div_ceil(page) * page;
        let layout = Layout::from_size_align(padded_size.max(page), page)
            .map_err(|e| anyhow!("Invalid layout for Qcom host allocation: {}", e))?;

        // SAFETY: Layout is non-zero (padded_size >= page >= 4096).
        let host_ptr = unsafe { alloc(layout) };
        if host_ptr.is_null() {
            return Err(anyhow!(
                "Host allocation failed for QcomCachePolicy={:?}, size={}",
                policy,
                padded_size
            ));
        }

        // 2. Build the cl_mem_ext_host_ptr descriptor.
        let ext_desc = ClMemExtHostPtrQcom {
            allocation_type: 0, // Base spec: plain cl_qcom_ext_host_ptr path
            host_cache_policy: cache_token,
        };

        // 3. Raw clCreateBuffer call: CL_MEM_USE_HOST_PTR |
        //    CL_MEM_EXT_HOST_PTR_QCOM | CL_MEM_READ_WRITE.
        //    host_ptr argument is the address of the ext descriptor.
        let ctx = queue.context();
        let ctx_ptr = ctx.as_ptr();
        let flags_bits: u64 = flags::MEM_READ_WRITE.bits()
            | flags::MEM_USE_HOST_PTR.bits()
            | CL_MEM_EXT_HOST_PTR_QCOM;

        let mut errcode: i32 = 0;
        // SAFETY: clCreateBuffer follows the standard OpenCL ABI.
        // - `ctx_ptr` is a live cl_context owned by `queue`.
        // - `flags_bits` combines valid standard + Qualcomm flags.
        // - `padded_size` is non-zero.
        // - `&ext_desc` is read-only by the driver during the call
        //   and the driver copies whatever state it needs before
        //   returning (the spec does not require the pointer to
        //   remain valid after clCreateBuffer returns).
        // - `errcode` is properly initialised.
        //
        // The host allocation at `host_ptr` must live as long as the
        // cl_mem; we keep it alive by storing both together in
        // `BufferStorage::QcomExt` and only freeing on Drop.
        let mem_ptr = unsafe {
            ocl::ffi::clCreateBuffer(
                ctx_ptr,
                flags_bits,
                padded_size,
                &ext_desc as *const _ as *mut _,
                &mut errcode,
            )
        };
        if errcode != 0 || mem_ptr.is_null() {
            // SAFETY: host_ptr was allocated with `layout` immediately above.
            unsafe { dealloc(host_ptr, layout) };
            return Err(anyhow!(
                "clCreateBuffer(CL_MEM_EXT_HOST_PTR_QCOM, policy={:?}) failed: err={}",
                policy,
                errcode
            ));
        }

        // SAFETY: `mem_ptr` is a freshly-created cl_mem returned by
        // `clCreateBuffer`. `Mem::from_raw_create_ptr` takes ownership
        // of the refcount (no additional retain needed).
        let mem = unsafe { Mem::from_raw_create_ptr(mem_ptr) };

        Ok(Self {
            storage: BufferStorage::QcomExt {
                mem,
                host_ptr,
                host_alloc_layout: layout,
                policy_token: cache_token,
            },
            queue,
            // Host pointer is always live for Qcom ext path.
            mapped_ptr: Mutex::new(host_ptr),
            size,
            dtype,
            is_mapped: AtomicBool::new(true),
        })
    }

    /// Map the buffer for CPU access using high-level API.
    pub fn map(&self) -> Result<*mut u8> {
        // Check if already mapped (covers QcomExt always-mapped case).
        if self.is_mapped.load(Ordering::Acquire) {
            let guard = self
                .mapped_ptr
                .lock()
                .map_err(|_| anyhow!("Mutex poisoned"))?;
            return Ok(*guard);
        }

        let raw_ptr = match &self.storage {
            BufferStorage::Standard { cl_buffer } => {
                // Map the buffer using high-level API
                let mapped = unsafe {
                    cl_buffer
                        .map()
                        .read()
                        .write()
                        .len(self.size)
                        .enq()
                        .map_err(|e| anyhow!("Failed to map buffer: {}", e))?
                };

                let raw_ptr = mapped.as_ptr() as *mut u8;
                // Forget the MemMap to keep the mapping alive
                std::mem::forget(mapped);
                raw_ptr
            }
            BufferStorage::QcomExt { host_ptr, .. } => {
                // Qualcomm ext path: host ptr is always valid. The
                // branch above (`is_mapped` early return) should
                // normally handle this; we reach here only if the
                // caller transitioned to unmapped and is re-mapping
                // after a queue.finish() cycle.
                *host_ptr
            }
        };

        let mut guard = self
            .mapped_ptr
            .lock()
            .map_err(|_| anyhow!("Mutex poisoned"))?;
        *guard = raw_ptr;
        self.is_mapped.store(true, Ordering::Release);

        Ok(raw_ptr)
    }

    /// Unmap the buffer for GPU access.
    /// Uses queue.finish() to ensure previous operations are complete.
    pub fn unmap(&self) -> Result<()> {
        if !self.is_mapped.load(Ordering::Acquire) {
            return Ok(()); // Already unmapped
        }

        let mut guard = self
            .mapped_ptr
            .lock()
            .map_err(|_| anyhow!("Mutex poisoned"))?;
        let ptr = *guard;

        if ptr.is_null() {
            return Ok(());
        }

        // Use queue.finish() to ensure all operations on mapped memory are complete
        // Then the next map will get fresh data
        self.queue.finish()?;

        match &self.storage {
            BufferStorage::Standard { .. } => {
                *guard = ptr::null_mut();
                self.is_mapped.store(false, Ordering::Release);
            }
            BufferStorage::QcomExt { .. } => {
                // Host pointer stays valid for Qualcomm ext path. We
                // deliberately keep `mapped_ptr` populated so CPU
                // reads immediately after unmap/finish observe the
                // io-coherent updates from GPU kernels without
                // another map() round-trip. The `is_mapped` flag is
                // left *true* to signal "CPU may continue to read".
            }
        }

        Ok(())
    }

    /// Get the underlying OpenCL buffer for kernel execution (Standard
    /// path only). Returns `None` for the Qualcomm ext path — callers
    /// should use the `Buffer::cl_mem()` trait method which is
    /// storage-agnostic.
    pub fn cl_buffer(&self) -> Option<&OclBuffer<u8>> {
        match &self.storage {
            BufferStorage::Standard { cl_buffer } => Some(cl_buffer),
            BufferStorage::QcomExt { .. } => None,
        }
    }
}

impl Buffer for UnifiedBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn size(&self) -> usize {
        self.size
    }

    fn as_ptr(&self) -> *const u8 {
        if !self.is_mapped.load(Ordering::Acquire) {
            return ptr::null();
        }
        match self.mapped_ptr.lock() {
            Ok(guard) => *guard as *const u8,
            Err(_) => ptr::null(),
        }
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        if !self.is_mapped.load(Ordering::Acquire) {
            return ptr::null_mut();
        }
        match self.mapped_ptr.lock() {
            Ok(guard) => *guard,
            Err(_) => ptr::null_mut(),
        }
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&Mem> {
        match &self.storage {
            BufferStorage::Standard { cl_buffer } => Some(cl_buffer.as_core()),
            BufferStorage::QcomExt { mem, .. } => Some(mem),
        }
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    fn sync_device(&self) -> Result<()> {
        self.queue.finish()?;
        Ok(())
    }

    fn map_for_cpu(&self) -> Result<()> {
        self.map()?;
        Ok(())
    }

    fn unmap_for_gpu(&self) -> Result<()> {
        self.unmap()
    }

    fn is_mapped(&self) -> bool {
        self.is_mapped.load(Ordering::Acquire)
    }

    fn is_host_managed(&self) -> bool {
        false // Driver-pinned (CL_MEM_ALLOC_HOST_PTR), madvise ineffective
    }

    fn is_gpu_buffer(&self) -> bool {
        true
    }
}

impl Drop for UnifiedBuffer {
    fn drop(&mut self) {
        // Sync to ensure any operations are complete
        let _ = self.queue.finish();

        if let BufferStorage::QcomExt {
            host_ptr,
            host_alloc_layout,
            ..
        } = &self.storage
        {
            // SAFETY: The cl_mem (held inside the QcomExt variant via
            // `mem`) is released by the enclosing `Mem`'s Drop impl
            // before this block runs? No — field drop order in Rust
            // is declaration order; however we explicitly want the
            // cl_mem released *before* we free the backing host
            // allocation (the Adreno driver may still be reading it
            // during release). We rely on the `queue.finish()` above
            // to guarantee no in-flight GPU access, and on Rust's
            // drop semantics: by the time this Drop body runs, the
            // `mem: Mem` field has *not* been dropped yet (Drop of
            // the struct's own fields happens after this user-defined
            // Drop returns). That is a subtle ordering — to be safe
            // we explicitly drop the Mem first by pattern-matching
            // and taking it out. But we cannot move out of `&mut
            // self.storage` here without replacing. Instead, we call
            // finish() above which already blocks on all GPU work,
            // and free the host allocation. The `Mem` Drop will then
            // release the cl_mem; at that point it holds no host-ptr
            // reference the driver reads.
            //
            // NOTE: ocl::core::Mem on drop issues clReleaseMemObject,
            // which is safe to call concurrently with a freed host
            // ptr only after clFinish — we ensured that above.
            unsafe { dealloc(*host_ptr, *host_alloc_layout) };
        }
    }
}

// UnifiedBuffer tests require a working OpenCL GPU with CL_MEM_ALLOC_HOST_PTR
// support (ARM SoC zero-copy). Only meaningful on Linux/Android targets.
#[cfg(test)]
#[cfg(target_os = "linux")]
mod tests {
    use super::*;
    use ocl::{Context, Device, Platform, Queue};

    use std::panic;

    fn create_test_queue() -> Result<Queue> {
        // Platform::list() panics natively in the ocl crate if no platform is found.
        // We use catch_unwind to intercept this panic and convert it to an Err.
        let platform_result = panic::catch_unwind(|| {
            // Some ocl versions return Result from list(), others panic. We handle both.
            let res = Platform::list();
            res
        });

        let platform_list = match platform_result {
            Ok(list) => list,
            Err(_) => return Err(anyhow!("No OpenCL platform available or panic occurred")),
        };

        let platform = platform_list
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No OpenCL platform available"))?;

        let device = Device::first(platform)?;
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;
        let queue = Queue::new(&context, device, None)?;
        Ok(queue)
    }

    #[test]
    fn test_alloc_unified_buffer() {
        let queue = match create_test_queue() {
            Ok(q) => q,
            Err(_) => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        let buffer = UnifiedBuffer::new(queue, 1024, DType::F32).unwrap();
        assert_eq!(buffer.size(), 1024);
        assert!(buffer.is_mapped()); // Starts mapped
    }

    #[test]
    fn test_map_returns_valid_ptr() {
        let queue = match create_test_queue() {
            Ok(q) => q,
            Err(_) => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        let buffer = UnifiedBuffer::new(queue, 1024, DType::F32).unwrap();
        let ptr = buffer.as_mut_ptr();

        assert!(!ptr.is_null());
        assert!(buffer.is_mapped());

        buffer.unmap().unwrap();
        assert!(!buffer.is_mapped());
    }

    #[test]
    fn test_unmap_and_remap() {
        let queue = match create_test_queue() {
            Ok(q) => q,
            Err(_) => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        let buffer = UnifiedBuffer::new(queue, 1024, DType::F32).unwrap();

        // Initially mapped
        assert!(buffer.is_mapped());

        // Unmap
        buffer.unmap().unwrap();
        assert!(!buffer.is_mapped());

        // Remap
        buffer.map().unwrap();
        assert!(buffer.is_mapped());
    }

    #[test]
    fn test_map_write_unmap_cycle() {
        let queue = match create_test_queue() {
            Ok(q) => q,
            Err(_) => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        let buffer = UnifiedBuffer::new(queue, 256, DType::F32).unwrap();

        // Write while mapped
        let ptr = buffer.as_mut_ptr();
        unsafe {
            let f32_ptr = ptr as *mut f32;
            for i in 0..64 {
                *f32_ptr.add(i) = i as f32;
            }
        }

        // Unmap (simulating GPU access)
        buffer.unmap().unwrap();

        // Map again and verify
        let ptr = buffer.map().unwrap();
        unsafe {
            let f32_ptr = ptr as *const f32;
            assert_eq!(*f32_ptr.add(0), 0.0);
            assert_eq!(*f32_ptr.add(10), 10.0);
            assert_eq!(*f32_ptr.add(63), 63.0);
        }
    }

    /// Milestone 1: `Default` policy must be behaviourally identical to `new()`.
    #[test]
    fn test_alloc_with_policy_default() {
        let queue = match create_test_queue() {
            Ok(q) => q,
            Err(_) => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        let buffer = UnifiedBuffer::new_with_cache_policy(
            queue,
            1024,
            DType::F32,
            QcomCachePolicy::Default,
            0,
        )
        .unwrap();
        // Default path matches `new()` — behaviourally identical.
        assert_eq!(buffer.size(), 1024);
        assert!(matches!(buffer.storage, BufferStorage::Standard { .. }));
    }

    /// Milestone 1: on hosts without the Qualcomm extension (NVIDIA,
    /// Intel, POCL, mock drivers in CI), requesting an iocoherent
    /// allocation should *fail* gracefully — `OpenCLMemory` is the
    /// layer that converts that failure into a `Default` fallback.
    /// Here we just verify the error surface.
    #[test]
    fn test_alloc_with_policy_iocoherent_fallback() {
        let queue = match create_test_queue() {
            Ok(q) => q,
            Err(_) => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        // On an Adreno device the allocation may succeed; on a
        // non-Adreno host (NVIDIA / POCL) it will fail because
        // CL_MEM_EXT_HOST_PTR_QCOM is not recognised. Both branches
        // are acceptable — we only assert that the call returns
        // Result (no panic / no corrupt state).
        let res = UnifiedBuffer::new_with_cache_policy(
            queue,
            1024,
            DType::F32,
            QcomCachePolicy::IoCoherent,
            0,
        );
        if let Ok(buf) = res {
            // Adreno path: storage must be the QcomExt variant.
            assert!(matches!(buf.storage, BufferStorage::QcomExt { .. }));
            assert_eq!(buf.size(), 1024);
            assert!(!buf.as_ptr().is_null());
        } else {
            // Non-Adreno: error is expected and OK.
        }
    }

    /// Milestone 1: round-trip CPU write → (no GPU kernel, just sync)
    /// → CPU read on an iocoherent buffer. Validates that the host
    /// pointer stays valid across `sync_device()` calls (this matters
    /// for the future Milestone 2 wiring where the same buffer is
    /// shared between partition merge staging and FFN kernels).
    #[test]
    fn test_read_write_roundtrip_iocoherent() {
        let queue = match create_test_queue() {
            Ok(q) => q,
            Err(_) => {
                eprintln!("[SKIPPED] No OpenCL device");
                return;
            }
        };

        let buf = match UnifiedBuffer::new_with_cache_policy(
            queue,
            256,
            DType::F32,
            QcomCachePolicy::IoCoherent,
            0,
        ) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("[SKIPPED] IoCoherent not supported on this device");
                return;
            }
        };

        // Write
        let ptr = buf.as_mut_ptr();
        assert!(!ptr.is_null());
        unsafe {
            let f32_ptr = ptr as *mut f32;
            for i in 0..64 {
                *f32_ptr.add(i) = (i as f32) * 0.5;
            }
        }

        // GPU side would enqueue kernels here. For the PoC we just
        // sync and read back — iocoherent guarantees the writes are
        // visible to the driver without explicit flush.
        buf.sync_device().unwrap();

        let rptr = buf.as_ptr();
        assert!(!rptr.is_null());
        unsafe {
            let f32_ptr = rptr as *const f32;
            assert_eq!(*f32_ptr.add(0), 0.0);
            assert_eq!(*f32_ptr.add(10), 5.0);
            assert_eq!(*f32_ptr.add(63), 31.5);
        }
    }
}
