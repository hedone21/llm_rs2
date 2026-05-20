//! `Memory` impl for qnn_oppkg backend (M3.3 — rpcmem allocator).
//!
//! Spec: `spec/30-engine.md` 부록 C.2 (ENG-QNN-204, INV-171),
//! `arch/30-engine.md` §18.1.
//!
//! ## INV-171 — rpcmem-backed + host pointer expose
//! Android device path: `librpcmem.so`(libcdsprpc.so로 dlopen)으로 DMA-BUF heap
//! fd 확보 → host pointer (rpcmem_alloc은 mmap된 host 가상 주소를 반환) →
//! QNN runtime이 `QnnMem_register`로 graph-binding handle 등록. Eviction/quant
//! policy는 host pointer 통해 수행 가능 (LISWAP HostPtrPool과 동일 패턴).
//!
//! ## 본 단계 (M3.3) 적용 범위
//! - Android device: `rpcmem_alloc` + `rpcmem_to_fd` + `QnnMem_register` 본격
//!   도입. M2.H microbench lines 196-204의 alloc flow를 production wrapper로
//!   이식.
//! - host (linux x86_64): `posix_memalign` fallback으로 host buffer alloc.
//!   `qnn_mem_handle`은 0 (graph에서 사용 불가). 정확성 검증/단위 테스트 용도.

use crate::backend::qnn_oppkg::QnnOppkgBackend;
use crate::backend::qnn_oppkg::buffer::QnnOppkgBuffer;
use crate::buffer::{Buffer, DType};
use crate::memory::Memory;
use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// QNN OpPackage backend용 메모리 allocator.
///
/// 본 단계 (M3.3)는 다음을 보장한다:
/// - host fallback: `posix_memalign` 64-byte aligned. graph 호환 X (qnn_mem_handle=0).
/// - device path: rpcmem_alloc → fd → `QnnMem_register`로 graph-binding handle.
///
/// `used_memory()`는 누적 alloc 합 (dealloc 추적은 M3.4에서 도입; 본 단계는
/// process lifetime 동안만 alloc).
pub struct QnnOppkgMemory {
    /// Backend reference — runtime 핸들 공유 (`QnnMem_register` 시 사용).
    backend: Arc<QnnOppkgBackend>,
    /// 누적 alloc 바이트.
    used_bytes: AtomicUsize,
}

impl QnnOppkgMemory {
    pub fn new(backend: Arc<QnnOppkgBackend>) -> Self {
        Self {
            backend,
            used_bytes: AtomicUsize::new(0),
        }
    }
}

impl Memory for QnnOppkgMemory {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        if size == 0 {
            return Err(anyhow!("QnnOppkgMemory::alloc(size=0) invalid"));
        }
        #[cfg(target_os = "android")]
        {
            let buf = android_rpcmem_alloc(self.backend.as_ref(), dtype, size)?;
            self.used_bytes.fetch_add(size, Ordering::Relaxed);
            Ok(Arc::new(buf))
        }
        #[cfg(not(target_os = "android"))]
        {
            // host fallback — posix_memalign 64-byte aligned. graph 호환 X.
            // 본 path는 단위 테스트(host에서 buffer trait 동작 검증) 용도.
            let _ = &self.backend; // unused on host
            let buf = QnnOppkgBuffer::host_aligned_alloc(dtype, size)?;
            self.used_bytes.fetch_add(size, Ordering::Relaxed);
            Ok(Arc::new(buf))
        }
    }

    fn used_memory(&self) -> usize {
        self.used_bytes.load(Ordering::Relaxed)
    }
}

#[cfg(target_os = "android")]
fn android_rpcmem_alloc(
    backend: &QnnOppkgBackend,
    dtype: DType,
    size: usize,
) -> Result<QnnOppkgBuffer> {
    use crate::backend::qnn_oppkg::runtime::ffi;
    use std::ptr;

    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;

    let runtime = backend.runtime_arc();
    if !runtime.is_initialized() {
        return Err(anyhow!(
            "QnnOppkgMemory::alloc — runtime not initialized (libQnnGpu missing or init failed)"
        ));
    }
    let (rpcmem_alloc, _rpcmem_free, rpcmem_to_fd) = runtime.rpcmem_fns();

    // INV-171: alloc returns host-mapped pointer; same buffer is shared with
    // QNN graph via fd + memHandle. Host can read/write directly without DMA.
    let host_ptr =
        unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size as i32) };
    if host_ptr.is_null() {
        return Err(anyhow!(
            "rpcmem_alloc(size={}) returned NULL (heap exhaustion?)",
            size
        ));
    }
    let fd = unsafe { rpcmem_to_fd(host_ptr) };
    if fd < 0 {
        // Best-effort free, then propagate error.
        unsafe { _rpcmem_free(host_ptr) };
        return Err(anyhow!("rpcmem_to_fd returned {} (size={})", fd, size));
    }

    // QnnMem_register — converts (fd, host_ptr) into a Qnn_MemHandle_t. Graph
    // tensors then bind to this handle via `Qnn_TensorV1_t.memHandle`.
    //
    // Note: at alloc time we don't yet know the graph-side dimensions, so we
    // register with shape `[size]` (1-D bytes). layer_graph::build will
    // ensure tensor dims align with the consumer op's expected layout.
    let mut dims: [u32; 1] = [size as u32];
    let mem_descriptor = ffi::Qnn_MemDescriptor_t {
        memShape: ffi::Qnn_MemShape_t {
            numDim: 1,
            dimSize: dims.as_mut_ptr(),
            shapeConfig: ptr::null(),
        },
        // QnnMem_register only needs a generic byte-typed registration; the
        // tensor binding (build_tensor) overrides dataType per consumer.
        dataType: ffi::Qnn_DataType_t_QNN_DATATYPE_UINT_8,
        memType: ffi::Qnn_MemType_t_QNN_MEM_TYPE_DMA_BUF,
        __bindgen_anon_1: ffi::Qnn_MemDescriptor_t__bindgen_ty_1 {
            dmaBufInfo: ffi::Qnn_MemDmaBufInfo_t { fd, data: host_ptr },
        },
    };
    let mut mem_handle: ffi::Qnn_MemHandle_t = ptr::null_mut();
    let v = runtime.v();
    let mem_register = v
        .memRegister
        .ok_or_else(|| anyhow!("memRegister fn-pointer is NULL"))?;
    let err = unsafe { mem_register(runtime.context(), &mem_descriptor, 1, &mut mem_handle) };
    if err != 0 {
        unsafe { _rpcmem_free(host_ptr) };
        return Err(anyhow!(
            "QnnMem_register(size={}, fd={}) err=0x{:x}",
            size,
            fd,
            err
        ));
    }

    Ok(QnnOppkgBuffer::from_rpcmem(
        dtype,
        size,
        host_ptr as *mut u8,
        fd,
        size,
        mem_handle as u64,
    ))
}

#[cfg(test)]
mod tests {
    // QnnOppkgMemory는 backend Arc가 필요하지만, host에서 backend init은
    // Err이므로 alloc(0) 자체는 buffer.rs unit test (`host_aligned_alloc_zero_errs`)
    // 에서 검증된다. 본 module의 unit test는 향후 rpcmem allocator 도입 후
    // 추가 예정.

    #[cfg(not(target_os = "android"))]
    #[test]
    fn host_alloc_zero_returns_err() {
        // placeholder — buffer.rs `host_aligned_alloc_zero_errs`가 실제 검증 담당.
        // 본 stub은 mod test scaffold만 유지하여 M3.3 device path 도입 시
        // 추가 case가 같은 path로 들어오도록 한다.
    }
}
