//! HTP FastRPC backend — `Memory` trait impl (rpcmem allocator).
//!
//! `RpcmemBuffer::alloc` 을 `Memory::alloc` 으로 노출하여 엔진의 buffer
//! provisioning 경로가 rpcmem (DMA-BUF heap) 영역을 primary store 로 쓸 수
//! 있게 한다. 동일 [`HtpFastrpcHost`] Arc 를 backend 와 share 하여 FFI symbol
//! lifetime 을 보장한다.

#![cfg(feature = "htp_fastrpc")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;

use super::buffer::RpcmemBuffer;
use super::host::HtpFastrpcHost;
use crate::buffer::{Buffer, DType};
use crate::memory::Memory;

/// rpcmem-backed `Memory` allocator. backend 와 동일 host Arc 를 share.
pub struct HtpFastrpcMemory {
    /// FFI symbol holder. backend 와 share (host lifetime 보장).
    host: Arc<HtpFastrpcHost>,
    /// 누적 할당 바이트. **increment-only (MVP 근사)** — RpcmemBuffer Drop 시
    /// 감소시키지 않는다. decrement-on-drop 은 RpcmemBuffer struct 수정을
    /// 요구하므로 (S1 surgical 유지 정책) 도입하지 않았다. `used_memory` 는
    /// manager pressure 용 informational 값이라 단조 증가 근사를 허용한다.
    used: AtomicUsize,
}

impl HtpFastrpcMemory {
    pub fn new(host: Arc<HtpFastrpcHost>) -> Self {
        Self {
            host,
            used: AtomicUsize::new(0),
        }
    }
}

impl Memory for HtpFastrpcMemory {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        let buf = RpcmemBuffer::alloc(self.host.clone(), size, dtype)?;
        // increment-only 회계 (decrement-on-drop 미도입, struct 주석 참조).
        self.used.fetch_add(size, Ordering::Relaxed);
        Ok(Arc::new(buf) as Arc<dyn Buffer>)
    }

    fn used_memory(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }
}
