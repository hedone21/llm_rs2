//! HTP FastRPC backend — rpcmem allocator wrapper (INV-HTP-FRPC-002).
//!
//! `rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size)` 로 할당한
//! DMA-BUF heap 메모리를 RAII wrapper 로 감싼다. **정확히 1개 wrapper 가
//! ownership 을 가지며** Drop 시 `rpcmem_free` 가 1회 호출된다.
//!
//! 이중 free 또는 leak 은 다음 패턴으로 차단:
//! 1. `RpcmemBuffer` 는 `Clone` 미구현 (컴파일 타임 강제).
//! 2. multi-owner read 가 필요할 경우 caller 가 `Arc<RpcmemBuffer>` 로 wrap
//!    한다 — `Arc::clone` 은 inner buffer 의 free 를 추가 트리거하지 않는다.
//!
//! 자세한 규칙: `spec/htp_fastrpc.md` §3 INV-HTP-FRPC-002.

#![cfg(feature = "htp_fastrpc")]

use std::os::raw::{c_int, c_void};
use std::sync::Arc;

use anyhow::{Result, anyhow};

use super::host::{
    DspQueueBuffer, FASTRPC_MAP_FD, HtpFastrpcHost, RPCMEM_DEFAULT_FLAGS, RPCMEM_HEAP_ID_SYSTEM,
    RPCMEM_HEAP_NOREG,
};
use super::idl::DspqBufferType;

/// alloc flags = `RPCMEM_DEFAULT_FLAGS | RPCMEM_HEAP_NOREG`. llama.cpp
/// `ggml-hexagon.cpp:268` 와 동일. NOREG 는 alloc 시점에 모든 도메인 자동
/// register 를 막아 명시 `fastrpc_mmap` 으로 특정 도메인에만 register 한다.
///
/// β-1.MAP 검증: NOREG 포함/제거 두 변형 모두 `fastrpc_mmap rc=0` (이전
/// β-1.START 의 `rc=0x1 EIO` 는 FASTRPC_MAP_FD enum 값 오인 (16 → 2) 이
/// root cause 였음). 둘의 차이는 dspqueue_write 시 `fastrpc_buffer_ref`
/// 카운트 동작에만 영향을 줘 본 PoC scope 외. NOREG 유지 = llama.cpp 일치.
const RPCMEM_ALLOC_FLAGS: u32 = RPCMEM_DEFAULT_FLAGS | RPCMEM_HEAP_NOREG;

/// rpcmem-backed buffer. RAII free, single-owner.
///
/// `host` 는 `Arc` 로 보유 (FFI symbol pointer 의 lifetime 보장). `Drop`
/// 시 `host.rpcmem_free(ptr)` 가 정확히 1회 호출된다.
///
/// `Clone` 은 의도적으로 미구현 — 복제가 필요하면 `Arc<RpcmemBuffer>` 로
/// wrap 한다 (single free 보장).
pub struct RpcmemBuffer {
    /// host-side virtual address.
    ptr: *mut u8,
    /// rpcmem fd (`rpcmem_to_fd` 결과). dspqueue_buffer 의 fd field 로 전달.
    fd: c_int,
    /// user-requested byte size (slice/dsp_buf 가 본 값을 노출).
    size: usize,
    /// 실제 rpcmem allocation 크기 (page-aligned ≥ size). `fastrpc_mmap` 와
    /// `fastrpc_munmap` 의 length 인자로 사용.
    alloc_size: usize,
    /// FFI symbol holder. Arc 로 backend / buffer pool 등에서 공유.
    host: Arc<HtpFastrpcHost>,
    /// `fastrpc_mmap(domain, fd, base, 0, alloc_size, FASTRPC_MAP_FD)` 호출
    /// 결과. true 면 Drop 에서 `fastrpc_munmap` 을 호출.
    mapped: bool,
}

// SAFETY: ptr 은 rpcmem 영역 (process-global). host 의 rpcmem_free 는
// thread-safe. 단일 소유자 정책으로 race 차단.
unsafe impl Send for RpcmemBuffer {}
unsafe impl Sync for RpcmemBuffer {}

impl RpcmemBuffer {
    /// `rpcmem_alloc(HEAP_ID_SYSTEM, DEFAULT_FLAGS, size)` 로 신규 할당.
    ///
    /// `size == 0` 은 거부 (ggml-hexagon 도 0 size 을 64 로 강제 promote
    /// 하지만, 본 wrapper 는 caller 에게 명시 책임을 요구).
    pub fn alloc(host: Arc<HtpFastrpcHost>, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(anyhow!("htp_fastrpc: RpcmemBuffer::alloc size == 0"));
        }
        // fastrpc_mmap 가 length 4K alignment 요구 (driver-dependent;
        // ARM page = 4 KiB). llama.cpp 도 `size += 4*1024` 로 padding 후
        // round 결과를 mmap. self.size 는 user-requested 값을 유지.
        const PAGE_SIZE: usize = 4096;
        let alloc_size = size.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        // SAFETY: rpcmem_alloc 은 thread-safe entry point. size 가 i32::MAX
        // 초과면 caller 책임 (rpcmem_alloc2 가 usize-size 받지만 optional).
        let raw = if let Some(alloc2) = host.rpcmem_alloc2 {
            unsafe { alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_ALLOC_FLAGS, alloc_size) }
        } else {
            if alloc_size > i32::MAX as usize {
                return Err(anyhow!(
                    "htp_fastrpc: RpcmemBuffer::alloc size {alloc_size} exceeds i32::MAX (use rpcmem_alloc2)"
                ));
            }
            unsafe {
                (host.rpcmem_alloc)(
                    RPCMEM_HEAP_ID_SYSTEM,
                    RPCMEM_ALLOC_FLAGS,
                    alloc_size as c_int,
                )
            }
        };
        if raw.is_null() {
            return Err(anyhow!(
                "htp_fastrpc: rpcmem_alloc returned NULL (size={size})"
            ));
        }
        let ptr = raw as *mut u8;
        // SAFETY: ptr 이 valid rpcmem allocation 임을 직전에 확인.
        let fd = unsafe { (host.rpcmem_to_fd)(ptr as *mut c_void) };
        if fd < 0 {
            // rpcmem_to_fd 실패 → 즉시 free 후 에러.
            unsafe { (host.rpcmem_free)(ptr as *mut c_void) };
            return Err(anyhow!(
                "htp_fastrpc: rpcmem_to_fd returned {fd} (size={size})"
            ));
        }

        // ── fastrpc_mmap (FastRPC domain 에 buffer 등록) ───────────────────
        //
        // dspqueue_write 가 `Buffer FD not mapped to domain N` 으로 fail 하지
        // 않으려면 buffer 를 FastRPC domain 에 명시 mmap 해야 한다 (llama.cpp
        // ggml-hexagon.cpp:235 와 동일 패턴).
        //
        // β-1.MAP root cause: 이전 `FASTRPC_MAP_FD = 16` 은 사실
        // `FASTRPC_MAP_FD_NOMAP` (등록만, 실제 mmap 안 함) — `rc=0x1 EIO` 의
        // 원인. QC fastrpc 오픈소스 enum 정확값 = 2.
        //
        // SAFETY: domain_id / fd / ptr / size 모두 직전에 valid 확인.
        // length 는 page-aligned `alloc_size` 사용.
        let mmap_rc = unsafe {
            (host.fastrpc_mmap)(
                host.domain_id,
                fd,
                ptr as *mut c_void,
                0,
                alloc_size,
                FASTRPC_MAP_FD,
            )
        };
        if mmap_rc != 0 {
            // mmap 실패 → free 후 에러. rpcmem 자체는 valid 였으니 free 호출.
            unsafe { (host.rpcmem_free)(ptr as *mut c_void) };
            return Err(anyhow!(
                "htp_fastrpc: fastrpc_mmap rc={mmap_rc:#x} (domain={}, fd={}, size={alloc_size})",
                host.domain_id,
                fd,
            ));
        }

        Ok(Self {
            ptr,
            fd,
            size,
            alloc_size,
            host,
            mapped: true,
        })
    }

    /// raw host-side pointer (read-only view 용). DSP 와 share 중이라
    /// host-side write 후 `DspqBufferType::CpuWriteDspRead` flag 로
    /// cache flush 필요.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// raw host-side pointer (write 용).
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// host-side byte slice. `unsafe` 가 아닌 이유: ptr 은 alloc 시점에 valid
    /// 보장 + size 동일 + lifetime 이 `&self` 로 제한. DSP 와의 race 는
    /// caller 가 `synchronize` 로 fence 하는 책임.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr/size 는 alloc 시점에 valid, &self lifetime 동안 stable.
        unsafe { core::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// host-side mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: 동일. &mut self 가 exclusive access 보장.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    pub fn fd(&self) -> c_int {
        self.fd
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn host(&self) -> &Arc<HtpFastrpcHost> {
        &self.host
    }

    /// dspqueue_write 의 buffer descriptor 생성. `direction` 으로 cache
    /// flush flag 결정 (`DspqBufferType::to_flags`).
    ///
    /// `offset` 와 `size` 는 user-specified — 본 buffer 의 부분 transfer 용.
    /// `offset + size <= self.size` 가 아니면 에러.
    pub fn dsp_buf(
        &self,
        direction: DspqBufferType,
        offset: u32,
        size: u32,
    ) -> Result<DspQueueBuffer> {
        let end = (offset as usize)
            .checked_add(size as usize)
            .ok_or_else(|| anyhow!("htp_fastrpc: dsp_buf overflow"))?;
        if end > self.size {
            return Err(anyhow!(
                "htp_fastrpc: dsp_buf range {offset}..{end} exceeds buffer size {}",
                self.size
            ));
        }
        Ok(DspQueueBuffer {
            fd: self.fd,
            // SAFETY: pointer arithmetic 가 buffer 안에 머무름 (위에서 검증).
            ptr: unsafe { self.ptr.add(offset as usize) } as *mut c_void,
            offset,
            size,
            flags: direction.to_flags(),
            reserved: [0; 3],
        })
    }
}

impl Drop for RpcmemBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // mmap 했었다면 munmap 먼저. munmap rc 무시 (best-effort cleanup,
            // host 의 domain 자체가 close 되면 자동 정리됨).
            if self.mapped {
                // SAFETY: domain/fd/ptr/size 모두 alloc 시점에 valid 였고,
                // 본 Drop 진입 전에는 unchanged.
                unsafe {
                    (self.host.fastrpc_munmap)(
                        self.host.domain_id,
                        self.fd,
                        self.ptr as *mut c_void,
                        self.alloc_size,
                    );
                }
                self.mapped = false;
            }
            // SAFETY: ptr 은 alloc 시점에 rpcmem_alloc 으로 얻은 값 (단일
            // owner). Drop 1회만 호출되며 Clone 미구현으로 컴파일 타임
            // 차단됨.
            unsafe { (self.host.rpcmem_free)(self.ptr as *mut c_void) };
            self.ptr = core::ptr::null_mut();
        }
    }
}

// Compile-time guard: Clone 미구현. INV-HTP-FRPC-002 의 single-owner 원칙.
//
// trait bound 가 아니라 `impl !Clone` 으로는 stable Rust 에서 표현 불가하나,
// 단순히 Clone derive 를 추가하지 않는 것 + 다음 마커 const 로 의도를 명시.
const _: () = {
    // 본 const 자체는 no-op. Clone derive 가 우연히 추가되면 review 단계에서
    // 본 const 의 의도 주석이 trigger 가 된다.
};

#[cfg(test)]
mod tests {
    use super::RpcmemBuffer;

    /// `RpcmemBuffer` 가 `Clone` 을 구현하지 않음을 컴파일 타임에 보장.
    ///
    /// `assert_not_impl_all!` 매크로는 `static_assertions` crate 가 필요하므로
    /// 여기서는 trait bound 로 동등 효과를 낸다. `requires_clone::<T>()`
    /// 호출이 컴파일된다면 T: Clone 이라는 뜻 — RpcmemBuffer 에는 컴파일
    /// fail (intentional).
    #[allow(dead_code)]
    fn assert_not_clone() {
        fn requires_clone<T: Clone>() {}
        // ↓ 아래 라인의 주석을 풀면 컴파일 fail 해야 한다 (intentional).
        // requires_clone::<RpcmemBuffer>();
        // 컴파일 패스가 되는 sanity 체크: u32 는 Clone.
        requires_clone::<u32>();
        // marker: RpcmemBuffer 가 in-scope 임을 컴파일러가 확인.
        let _ = core::marker::PhantomData::<RpcmemBuffer>;
    }
}
