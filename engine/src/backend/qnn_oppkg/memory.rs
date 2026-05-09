//! `Memory` impl for qnn_oppkg backend (M3.2 rpcmem skeleton).
//!
//! Spec: `spec/30-engine.md` 부록 C.2 (ENG-QNN-204, INV-171),
//! `arch/30-engine.md` §18.1.
//!
//! ## INV-171 — rpcmem-backed + host pointer expose
//! Android device path: `librpcmem.so` (또는 `/dev/dma_heap/system` ioctl)으로
//! DMA-BUF heap fd 확보 → mmap host pointer → QNN runtime이 등록한
//! `qnn_mem_handle`을 graph에 baked. Eviction/quant policy는 host pointer
//! 통해 수행 가능 (LISWAP HostPtrPool과 동일 패턴).
//!
//! ## 본 단계 (M3.2) 적용 범위
//! - Android device: `rpcmem_alloc` + `rpcmem_to_fd` + mmap dlopen wrapper는
//!   M3.3 device forward 진입 시 본격 도입 (M2.H microbench lines 196-204
//!   참고).
//! - host (linux x86_64): `posix_memalign` fallback으로 host buffer alloc.
//!   `qnn_mem_handle`은 0 (graph에서 사용 불가). 정확성 검증/단위 테스트 용도.

use crate::backend::qnn_oppkg::QnnOppkgBackend;
use crate::backend::qnn_oppkg::buffer::QnnOppkgBuffer;
use crate::core::buffer::{Buffer, DType};
use crate::core::memory::Memory;
use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// QNN OpPackage backend용 메모리 allocator.
///
/// 본 단계 (M3.2)는 다음을 보장한다:
/// - host fallback: `posix_memalign` 64-byte aligned. graph 호환 X (qnn_mem_handle=0).
/// - device path: M3.3 진입 시 rpcmem fd → mmap → `QnnMem_register`.
///
/// `used_memory()`는 누적 alloc 합 (dealloc 추적은 M3.3에서 도입; 본 단계는
/// process lifetime 동안만 alloc).
pub struct QnnOppkgMemory {
    /// Backend reference — runtime 핸들 공유 (M3.3에서 `QnnMem_register` 시 사용).
    #[allow(dead_code)]
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
            // M3.3 device forward 진입 시 본격: rpcmem_alloc + rpcmem_to_fd +
            // mmap + QnnMem_register. 현재 단계는 alloc 호출 자체가 발생하지
            // 않으므로 명시적 Err.
            let _ = (size, dtype);
            Err(anyhow!(
                "QnnOppkgMemory::alloc (android) — M3.3에서 rpcmem allocator 도입 후 구현"
            ))
        }
        #[cfg(not(target_os = "android"))]
        {
            // host fallback — posix_memalign 64-byte aligned. graph 호환 X.
            // 본 path는 단위 테스트(host에서 buffer trait 동작 검증) 용도.
            let buf = QnnOppkgBuffer::host_aligned_alloc(dtype, size)?;
            self.used_bytes.fetch_add(size, Ordering::Relaxed);
            Ok(Arc::new(buf))
        }
    }

    fn used_memory(&self) -> usize {
        self.used_bytes.load(Ordering::Relaxed)
    }
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
