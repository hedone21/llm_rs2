//! `QnnOppkgKvBuffer` — rpcmem + OpenCL dual-backed KV cache buffer.
//!
//! Step 1 (이 파일): rpcmem heap에 KV alloc + OpenCL `CL_MEM_USE_HOST_PTR`로
//! cl_mem alias를 만드는 dual buffer 도입. KV cache가 두 backend에서 동일
//! backing memory를 공유.
//!
//! Step 2 (다음 세션): Graph build/execute가 dual buffer의 `mem_handle`을
//! 외부 주입받게 변경하여 bridge memcpy를 제거한다.
//!
//! ## 안전성 노트
//! - `host_ptr`: rpcmem_alloc이 반환하는 mmap된 가상 주소. rpcmem DMA-BUF는
//!   CPU와 DSP/GPU가 동일 물리 페이지를 공유 (UMA).
//! - `cl_mem`: `CL_MEM_USE_HOST_PTR`로 생성한 alias. OpenCL driver가 동일
//!   host_ptr 페이지를 그대로 참조하므로 별도 memcpy 없음.
//! - `mem_handle`: QnnMem_register 결과. Step 2에서 graph에 주입 예정.
//! - `Send + Sync`: production OpenCL queue는 단일 스레드 접근 보장.

use crate::core::buffer::{Buffer, DType};
use anyhow::Result;
use std::any::Any;

/// rpcmem + OpenCL USE_HOST_PTR dual-backed KV cache buffer.
///
/// `cl_mem()`이 Some(...)을 반환하므로 OpenCL backend의 `read_buffer` /
/// `write_buffer`가 자동으로 호환된다 (downcast 경유 cl_mem 추출).
/// `as_ptr()`은 host_ptr을 반환하여 eviction / KV migrate 정책도 가능.
pub struct QnnOppkgKvBuffer {
    /// rpcmem_alloc이 반환한 host-mapped 가상 주소.
    host_ptr: *mut u8,
    /// rpcmem_to_fd 결과 (DMA-BUF fd).
    #[allow(dead_code)]
    fd: i32,
    /// QnnMem_register 결과. Step 2에서 graph에 주입 예정.
    mem_handle: u64,
    /// CL_MEM_USE_HOST_PTR alias — 동일 host_ptr 페이지를 참조.
    #[cfg(feature = "opencl")]
    cl_mem_obj: ocl::core::Mem,
    size: usize,
    dtype: DType,
    /// rpcmem_free fn-pointer — Drop에서 host_ptr 해제에 사용.
    /// Android에서만 유효. fn-pointer는 Copy이므로 thread-safe.
    #[cfg(target_os = "android")]
    rpcmem_free: unsafe extern "C" fn(*mut std::ffi::c_void),
}

impl QnnOppkgKvBuffer {
    #[cfg(all(feature = "opencl", target_os = "android"))]
    pub(crate) fn new(
        host_ptr: *mut u8,
        fd: i32,
        mem_handle: u64,
        cl_mem_obj: ocl::core::Mem,
        size: usize,
        dtype: DType,
        rpcmem_free: unsafe extern "C" fn(*mut std::ffi::c_void),
    ) -> Self {
        Self {
            host_ptr,
            fd,
            mem_handle,
            cl_mem_obj,
            size,
            dtype,
            rpcmem_free,
        }
    }

    /// Step 2에서 graph에 주입할 QNN mem handle.
    #[allow(dead_code)]
    pub(crate) fn qnn_mem_handle(&self) -> u64 {
        self.mem_handle
    }
}

// SAFETY: fn-pointer (rpcmem_free)는 Copy + stateless. host_ptr는 단일 소유.
// production OpenCL queue는 단일 스레드 접근 보장.
unsafe impl Send for QnnOppkgKvBuffer {}
unsafe impl Sync for QnnOppkgKvBuffer {}

impl Drop for QnnOppkgKvBuffer {
    fn drop(&mut self) {
        // cl_mem_obj (ocl::core::Mem)은 Drop 시 자동 clReleaseMemObject.
        // rpcmem_free는 Android에서만 수행.
        #[cfg(target_os = "android")]
        if !self.host_ptr.is_null() {
            // SAFETY: host_ptr은 rpcmem_alloc 경유로 할당된 유효 포인터.
            // Drop은 ownership 이동 1회만 호출되므로 double-free 없음.
            // rpcmem_free는 library symbol로 획득한 fn-pointer (런타임 유효).
            unsafe { (self.rpcmem_free)(self.host_ptr as *mut std::ffi::c_void) };
        }
    }
}

impl Buffer for QnnOppkgKvBuffer {
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
        self.host_ptr as *const u8
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.host_ptr
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&ocl::core::Mem> {
        Some(&self.cl_mem_obj)
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    fn sync_device(&self) -> Result<()> {
        // rpcmem DMA-BUF는 UMA에서 CPU-GPU가 동일 물리 페이지를 공유.
        // USE_HOST_PTR alias도 동일 물리 페이지 참조 → 추가 sync 불필요.
        Ok(())
    }

    fn is_host_managed(&self) -> bool {
        // rpcmem fd는 host(app)가 mmap을 관리 → eviction 정책 host-side 가능.
        true
    }

    fn is_gpu_buffer(&self) -> bool {
        // cl_mem이 있으므로 OpenCL backend에서 GPU buffer로 취급.
        true
    }
}
