//! `Buffer` impl for qnn_oppkg backend (M3.2 — host fallback alloc + rpcmem
//! skeleton).
//!
//! Spec: `spec/30-engine.md` 부록 C.2 (ENG-QNN-204, INV-171),
//! `arch/30-engine.md` §18.1.
//!
//! ## 구조
//! `host_ptr` + `qnn_mem_handle` pair. Android device path는 rpcmem fd → mmap
//! 으로 채워지고, host fallback path는 `posix_memalign` 만 사용한다
//! (qnn_mem_handle=0).
//!
//! ## 본 단계 (M3.2) 적용 범위
//! - host fallback alloc/dealloc 동작 (정확성 검증용)
//! - Buffer trait 모든 method 정상 구현 (cl_mem은 항상 None — INV-171: qnn 소스는
//!   rpcmem fd만 보유, OpenCL cl_mem 외부 공유는 R-Y에서 RED 확정)
//! - rpcmem path는 M3.3에서 본격 (현재 placeholder factory만 노출)

use crate::buffer::{Buffer, DType};
use anyhow::{Result, anyhow};
use std::alloc::{Layout, alloc, dealloc};
use std::any::Any;

/// rpcmem(DMA-BUF heap) backed buffer.
///
/// `host_ptr`는 mmap된 host 가상 주소 (Android) 또는 `posix_memalign` 결과
/// (host fallback). `qnn_mem_handle`은 V2.0 `QnnMem_register` 결과 (Android만
/// 유효, host는 0).
pub struct QnnOppkgBuffer {
    dtype: DType,
    size: usize,
    /// 본 buffer가 소유한 host alignment layout. `Drop`에서 해제 시 사용.
    /// rpcmem path는 `BackingStore::Rpcmem`에서 fd close + munmap을 처리한다.
    backing: BackingStore,
}

/// Buffer가 보유한 host memory의 출처.
enum BackingStore {
    /// host fallback — `std::alloc::alloc` 결과. 단위 테스트용.
    HostAligned { ptr: *mut u8, layout: Layout },
    /// Android rpcmem path (M3.3에서 활성화).
    /// fd / mmap_size / qnn_mem_handle 필드는 본 단계에서 unused.
    #[allow(dead_code)]
    Rpcmem {
        host_ptr: *mut u8,
        fd: i32,
        mmap_size: usize,
        qnn_mem_handle: u64,
    },
}

// SAFETY: rpcmem fd 및 mmap 영역은 `Buffer` trait 사용자(forward path)가
// 동시 R/W 시점을 직접 동기화한다. host fallback `HostAligned`도 동일.
// `Buffer` trait이 `Send + Sync`로 요구되므로 명시적으로 unsafe impl.
unsafe impl Send for QnnOppkgBuffer {}
unsafe impl Sync for QnnOppkgBuffer {}

impl QnnOppkgBuffer {
    /// Host fallback alloc — `posix_memalign` 64-byte aligned. `Drop`에서
    /// `dealloc` 호출하여 해제한다.
    ///
    /// graph 호환 X (`qnn_mem_handle == 0`). M3.3 device path와는 별개.
    pub(crate) fn host_aligned_alloc(dtype: DType, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(anyhow!("host_aligned_alloc(size=0) invalid"));
        }
        // 64 byte alignment: SIMD friendly + cache line 친화 + most rpcmem
        // heaps도 동일 alignment를 보장하므로 cross-path 호환.
        let layout =
            Layout::from_size_align(size, 64).map_err(|e| anyhow!("invalid layout: {e}"))?;
        // SAFETY: layout은 size>0, align>0 + power-of-2 검증됨.
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(anyhow!("host alloc failed (size={size})"));
        }
        Ok(Self {
            dtype,
            size,
            backing: BackingStore::HostAligned { ptr, layout },
        })
    }

    /// rpcmem-backed factory (M3.3 device 본문 진입 시 구현).
    ///
    /// 본 단계에서는 호출되지 않는다. signature만 노출하여 M3.3 작업이 본 path를
    /// 그대로 사용하도록 한다.
    #[cfg(target_os = "android")]
    #[allow(dead_code)]
    pub(crate) fn from_rpcmem(
        dtype: DType,
        size: usize,
        host_ptr: *mut u8,
        fd: i32,
        mmap_size: usize,
        qnn_mem_handle: u64,
    ) -> Self {
        Self {
            dtype,
            size,
            backing: BackingStore::Rpcmem {
                host_ptr,
                fd,
                mmap_size,
                qnn_mem_handle,
            },
        }
    }

    /// QNN registered memory handle. Android device path에서만 유효 (host
    /// fallback은 항상 0). M3.3 graph build에서 weight/KV tensor의
    /// `Qnn_Tensor_t.memHandle`로 baked.
    #[allow(dead_code)]
    pub(crate) fn qnn_mem_handle(&self) -> u64 {
        match &self.backing {
            BackingStore::HostAligned { .. } => 0,
            BackingStore::Rpcmem { qnn_mem_handle, .. } => *qnn_mem_handle,
        }
    }

    /// Raw host pointer accessor.
    fn host_ptr(&self) -> *mut u8 {
        match &self.backing {
            BackingStore::HostAligned { ptr, .. } => *ptr,
            BackingStore::Rpcmem { host_ptr, .. } => *host_ptr,
        }
    }
}

impl Drop for QnnOppkgBuffer {
    fn drop(&mut self) {
        match self.backing {
            BackingStore::HostAligned { ptr, layout } => {
                if !ptr.is_null() {
                    // SAFETY: `ptr` + `layout`은 `host_aligned_alloc`에서
                    // 유래. drop은 1회만 호출됨 (Rust ownership).
                    unsafe { dealloc(ptr, layout) };
                }
            }
            BackingStore::Rpcmem { .. } => {
                // M3.3 device 진입 시 rpcmem_free + close(fd) + munmap.
                // 본 단계는 host fallback path만 살아있으므로 unreachable.
            }
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
        self.host_ptr() as *const u8
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.host_ptr()
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&ocl::core::Mem> {
        // INV-171: qnn_oppkg buffer는 cl_mem이 아닌 rpcmem fd를 보유한다.
        // R-Y 결과 — cl_mem 외부 공유는 RED. graph 외부 공유는 rpcmem fd 경로만 사용.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_aligned_alloc_basic() {
        let buf = QnnOppkgBuffer::host_aligned_alloc(DType::F32, 1024).expect("alloc OK");
        assert_eq!(buf.size(), 1024);
        assert_eq!(buf.dtype(), DType::F32);
        assert!(!buf.as_ptr().is_null());
        assert_eq!(buf.qnn_mem_handle(), 0, "host fallback handle == 0");
        assert!(buf.is_host_managed());
        assert!(buf.is_gpu_buffer());
    }

    #[test]
    fn host_aligned_alloc_zero_errs() {
        let r = QnnOppkgBuffer::host_aligned_alloc(DType::F32, 0);
        assert!(r.is_err());
    }

    #[test]
    fn host_aligned_alloc_aligned_to_64() {
        let buf = QnnOppkgBuffer::host_aligned_alloc(DType::F32, 256).expect("alloc OK");
        let p = buf.as_ptr() as usize;
        assert_eq!(p % 64, 0, "host fallback alloc must be 64-byte aligned");
    }
}
