//! `Memory` impl for qnn_oppkg backend (M3.1 stub).
//!
//! Spec: `spec/30-engine.md` 부록 C.2 (ENG-QNN-204, INV-171),
//! `arch/30-engine.md` §18.1.
//!
//! M3.1: trait skeleton만. 실제 rpcmem(DMA-BUF heap) allocator는 M3.2에서
//! 본격 도입한다 (LISWAP 인프라의 `LLMRS_OPENCL_DMABUF_HEAP=1` 경로 재사용
//! 후보, 본 단계 범위 외).

use crate::backend::qnn_oppkg::QnnOppkgBackend;
use crate::backend::qnn_oppkg::buffer::QnnOppkgBuffer;
use crate::core::buffer::{Buffer, DType};
use crate::core::memory::Memory;
use anyhow::{Result, anyhow};
use std::sync::Arc;

/// QNN OpPackage backend용 메모리 allocator (M3.1 stub).
///
/// 실제 rpcmem(DMA-BUF heap) alloc + mmap host pointer 노출은 M3.2에서 본격
/// 도입. 본 단계는 backend handle만 보유하여 trait 형태를 갖춘다.
pub struct QnnOppkgMemory {
    /// Backend reference — runtime 핸들 공유 (M3.2에서 rpcmem fd 등록 시 사용).
    #[allow(dead_code)]
    backend: Arc<QnnOppkgBackend>,
}

impl QnnOppkgMemory {
    pub fn new(backend: Arc<QnnOppkgBackend>) -> Self {
        Self { backend }
    }
}

impl Memory for QnnOppkgMemory {
    fn alloc(&self, _size: usize, _dtype: DType) -> Result<Arc<dyn Buffer>> {
        // M3.2에서 rpcmem fd alloc + mmap + QnnMem_register 본격 구현.
        // M3.1 단계는 어떤 path도 alloc을 호출하지 않으므로 명확한 Err.
        Err(anyhow!(
            "QnnOppkgMemory::alloc() — M3.2에서 rpcmem allocator 도입 후 구현"
        ))
    }

    fn used_memory(&self) -> usize {
        0
    }
}

/// Stub allocator helper (M3.2 진입 전 unit test용).
#[allow(dead_code)]
pub(crate) fn placeholder_buffer(dtype: DType, size: usize) -> QnnOppkgBuffer {
    QnnOppkgBuffer::placeholder(dtype, size)
}
