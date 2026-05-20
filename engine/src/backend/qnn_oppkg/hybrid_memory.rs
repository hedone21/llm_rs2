//! `QnnOppkgHybridMemory` — QNN rpcmem + OpenCL dual-backed KV 메모리.
//!
//! Step 1 구현:
//! - `alloc()` → OpenCL secondary memory로 위임 (activation tensor는 cl_mem 유지)
//! - `alloc_kv()` → rpcmem + CL_MEM_USE_HOST_PTR alias로 `QnnOppkgKvBuffer` 생성
//!
//! KV cache buffer가 rpcmem DMA-BUF와 OpenCL cl_mem 두 핸들을 동시에 보유하므로
//! production OpenCL path (cl_mem 경유 read_buffer/write_buffer)가 무손상으로 동작한다.
//!
//! Step 2 (다음 세션): graph build/execute에서 `qnn_mem_handle()`을 외부 주입하여
//! bridge memcpy를 제거한다.

use crate::backend::qnn_oppkg::QnnOppkgBackend;
use crate::buffer::{Buffer, DType};
use crate::memory::Memory;
use anyhow::Result;
use std::sync::Arc;

pub struct QnnOppkgHybridMemory {
    /// activation tensor alloc 위임 대상 (OpenCL cl_mem 반환).
    ocl_memory: Arc<dyn Memory>,
    /// rpcmem alloc + QnnMem_register용 QNN backend reference.
    qnn_backend: Arc<QnnOppkgBackend>,
    /// CL_MEM_USE_HOST_PTR alias 생성용 OpenCL context.
    #[cfg(feature = "opencl")]
    ocl_context: ocl::Context,
}

impl QnnOppkgHybridMemory {
    #[cfg(feature = "opencl")]
    pub fn new(
        ocl_memory: Arc<dyn Memory>,
        qnn_backend: Arc<QnnOppkgBackend>,
        ocl_context: ocl::Context,
    ) -> Self {
        Self {
            ocl_memory,
            qnn_backend,
            ocl_context,
        }
    }

    #[cfg(not(feature = "opencl"))]
    pub fn new(ocl_memory: Arc<dyn Memory>, qnn_backend: Arc<QnnOppkgBackend>) -> Self {
        Self {
            ocl_memory,
            qnn_backend,
        }
    }
}

impl Memory for QnnOppkgHybridMemory {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        // activation tensor는 OpenCL cl_mem으로 할당 (production prefill/decode 무손상).
        self.ocl_memory.alloc(size, dtype)
    }

    fn alloc_kv(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        #[cfg(all(feature = "opencl", target_os = "android"))]
        {
            android_alloc_kv(&self.qnn_backend, &self.ocl_context, size, dtype)
        }
        #[cfg(not(all(feature = "opencl", target_os = "android")))]
        {
            // host build 또는 opencl feature 미사용: OpenCL memory에 위임.
            self.ocl_memory.alloc_kv(size, dtype)
        }
    }

    fn used_memory(&self) -> usize {
        self.ocl_memory.used_memory()
    }
}

/// Android + opencl feature 전용 — rpcmem + CL_MEM_USE_HOST_PTR dual alloc.
#[cfg(all(feature = "opencl", target_os = "android"))]
fn android_alloc_kv(
    backend: &QnnOppkgBackend,
    ocl_context: &ocl::Context,
    size: usize,
    dtype: DType,
) -> Result<Arc<dyn Buffer>> {
    use crate::backend::qnn_oppkg::kv_buffer::QnnOppkgKvBuffer;
    use crate::backend::qnn_oppkg::runtime::ffi;
    use anyhow::anyhow;
    use ocl::core::MEM_USE_HOST_PTR;
    use std::ptr;

    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;

    if size == 0 {
        return Err(anyhow!("QnnOppkgHybridMemory::alloc_kv(size=0) invalid"));
    }

    let runtime = backend.runtime_arc();
    if !runtime.is_initialized() {
        return Err(anyhow!(
            "QnnOppkgHybridMemory::alloc_kv — runtime not initialized"
        ));
    }
    let (rpcmem_alloc, rpcmem_free, rpcmem_to_fd) = runtime.rpcmem_fns();

    // 1. rpcmem_alloc — host-mapped DMA-BUF 할당.
    let host_ptr =
        unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size as i32) };
    if host_ptr.is_null() {
        return Err(anyhow!("rpcmem_alloc(size={size}) returned NULL"));
    }
    let fd = unsafe { rpcmem_to_fd(host_ptr) };
    if fd < 0 {
        unsafe { rpcmem_free(host_ptr) };
        return Err(anyhow!("rpcmem_to_fd returned {fd} (size={size})"));
    }

    // 2. QnnMem_register — graph binding handle 등록.
    let mut dims: [u32; 1] = [size as u32];
    let mem_descriptor = ffi::Qnn_MemDescriptor_t {
        memShape: ffi::Qnn_MemShape_t {
            numDim: 1,
            dimSize: dims.as_mut_ptr(),
            shapeConfig: ptr::null(),
        },
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
        unsafe { rpcmem_free(host_ptr) };
        return Err(anyhow!(
            "QnnMem_register(size={size}, fd={fd}) err=0x{err:x}"
        ));
    }

    // 3. OpenCL CL_MEM_USE_HOST_PTR alias — 동일 host_ptr 페이지를 참조.
    // SAFETY: host_ptr은 rpcmem_alloc이 반환한 유효 포인터 (size 바이트 이상).
    // CL_MEM_USE_HOST_PTR은 OpenCL driver가 host_ptr을 그대로 사용하므로 복사 없음.
    let cl_mem_slice = unsafe { std::slice::from_raw_parts_mut(host_ptr as *mut u8, size) };
    let cl_mem_obj = unsafe {
        ocl::core::create_buffer::<_, u8>(
            ocl_context.as_core(),
            MEM_USE_HOST_PTR,
            size,
            Some(cl_mem_slice),
        )
        .map_err(|e| {
            // best-effort cleanup
            unsafe { rpcmem_free(host_ptr) };
            anyhow!("clCreateBuffer USE_HOST_PTR failed (size={size}): {e}")
        })?
    };

    let buf = QnnOppkgKvBuffer::new(
        host_ptr as *mut u8,
        fd,
        mem_handle as u64,
        cl_mem_obj,
        size,
        dtype,
        rpcmem_free,
    );
    Ok(Arc::new(buf))
}
