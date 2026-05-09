//! `Buffer` impl for qnn_oppkg backend (M3.1 stub).
//!
//! Spec: `spec/30-engine.md` 부록 C.2 (ENG-QNN-204, INV-171),
//! `arch/30-engine.md` §18.1.
//!
//! M3.1: struct + `Buffer` trait 최소 골격만 마련. read/write/cl_mem 등은
//! M3.2에서 rpcmem allocator + view transform 본격 도입 후 채운다.

use crate::core::buffer::{Buffer, DType};
use anyhow::Result;
use std::any::Any;

/// rpcmem(DMA-BUF heap) backed buffer. host_ptr는 mmap된 가상 주소,
/// `qnn_mem_handle`은 QNN runtime이 register한 핸들 (M3.2에서 본격 채움).
pub struct QnnOppkgBuffer {
    dtype: DType,
    size: usize,
    /// mmap host pointer. M3.1 stub에서는 항상 null.
    host_ptr: *mut u8,
    /// QNN registered memory handle (M3.2에서 V2.0 `QnnMem_register` 결과).
    /// 본 단계에서는 placeholder 0.
    #[allow(dead_code)]
    qnn_mem_handle: u64,
}

// SAFETY: rpcmem fd 및 mmap 영역은 host pointer와 함께 thread-safe 사용을
// 가정한다. 실제 동시 접근 정책은 M3.2/M3.3에서 본격 검증하며, 현재 stub은
// host에서 alloc되지 않는다.
unsafe impl Send for QnnOppkgBuffer {}
unsafe impl Sync for QnnOppkgBuffer {}

impl QnnOppkgBuffer {
    /// M3.2에서 본격 alloc 도입. M3.1 단계는 placeholder.
    pub(crate) fn placeholder(dtype: DType, size: usize) -> Self {
        Self {
            dtype,
            size,
            host_ptr: std::ptr::null_mut(),
            qnn_mem_handle: 0,
        }
    }
}

impl Buffer for QnnOppkgBuffer {
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
        // qnn_oppkg buffer는 cl_mem이 아닌 rpcmem fd를 보유한다 (INV-171).
        None
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    fn sync_device(&self) -> Result<()> {
        // rpcmem 매핑은 디바이스/호스트가 같은 물리 페이지를 공유하므로 sync
        // 필요 없음. M3.3에서 cache flush 정책 검토 후 갱신.
        Ok(())
    }

    fn is_host_managed(&self) -> bool {
        // rpcmem fd는 host(app)가 mmap을 관리. eviction/quant 정책 host-side 가능.
        true
    }

    fn is_gpu_buffer(&self) -> bool {
        // QNN graph가 직접 접근 가능하므로 GPU buffer.
        true
    }
}
