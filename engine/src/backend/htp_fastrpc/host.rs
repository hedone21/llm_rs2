//! HTP FastRPC backend — FFI binding + handle/session lifecycle
//! (INV-HTP-FRPC-001 / INV-HTP-FRPC-004).
//!
//! `libcdsprpc.so` (stock S25 `/vendor/lib64/libcdsprpc.so`, 497624 bytes) 를
//! dlopen 하여 17 개 symbol 을 dlsym 후, FastRPC handshake 4-step
//! (reserve_session → unsigned_pd → handle_open → dspqueue_create) 을 거쳐
//! HTP DSP-side skel 과 통신할 ready state 를 구성한다.
//!
//! 본 모듈은 17 symbol struct + lifecycle entry points 만 노출한다. op
//! dispatch (rmsnorm 등) 는 `mod.rs` 의 Backend trait impl 에서 본 모듈의
//! function pointer 를 직접 호출하는 형태로 진행된다 (Phase 4 작업).
//!
//! Path B 차용: ggml-hexagon `htp-drv.cpp:32-360` 의 17 typedef + dlsym 시퀀스
//! 와 llama.cpp `ggml-hexagon.cpp::ggml_hexagon_session::allocate` 의 4-step
//! handshake 를 직접 매핑.
//!
//! Target = `aarch64-linux-android` 만 active. 다른 target 은 [`HtpFastrpcHost::new`]
//! 가 `Error::BackendUnavailable` 류 에러 (PoC scope: anyhow context) 를 반환.

#![cfg(feature = "htp_fastrpc")]
#![allow(non_camel_case_types, non_upper_case_globals)]

use std::os::raw::{c_int, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow};

#[cfg(target_os = "android")]
use std::ffi::CString;

use super::error::{AEE_SUCCESS, map_aee_err};

// ── FastRPC opaque types ──

pub type RemoteHandle64 = u64;
pub type DspQueueT = *mut c_void;

// ── FastRPC session_control req_id constants ──
//
// 출처: Qualcomm fastrpc open-source `inc/remote.h` (BSD-3-Clause).
// dry-run prototype `engine/microbench/htp_fastrpc_dryrun.rs` 에서 검증
// 완료된 값들.

pub const FASTRPC_RESERVE_NEW_SESSION: u32 = 13;
pub const FASTRPC_GET_URI: u32 = 15;
pub const DSPRPC_CONTROL_UNSIGNED_MODULE: u32 = 2;

/// `remote_handle_control` / `remote_handle64_control` 의 latency QoS req.
pub const DSPRPC_CONTROL_LATENCY: u32 = 1;

/// CDSP domain 이름 (FastRPC `domain.h`).
pub const CDSP_DOMAIN_NAME: &str = "cdsp";

/// dspqueue 호출의 default timeout (microseconds). llama.cpp ggml-hexagon
/// 와 동일 값.
pub const DSPQUEUE_TIMEOUT: u32 = 10_000_000;

/// rpcmem heap id. llama.cpp `ggml-hexagon.cpp:268`. SYSTEM heap = 25.
pub const RPCMEM_HEAP_ID_SYSTEM: c_int = 25;
/// rpcmem default flags. llama.cpp 동일.
pub const RPCMEM_DEFAULT_FLAGS: u32 = 1;
/// `RPCMEM_HEAP_NOREG` — alloc 시점에 모든 FastRPC domain 자동 register 를
/// 막는다. 명시 `fastrpc_mmap` 으로 특정 domain 에만 register 할 때 필수.
/// 출처: QC fastrpc 오픈소스 `inc/rpcmem.h` (github.com/quic/fastrpc).
/// llama.cpp `ggml-hexagon.cpp:268` 가 본 값을 OR.
///
/// β-1.MAP 결정값: 0x40000000 (이전 0x80000000 은 `RPCMEM_HEAP_DEFAULT` 였음).
pub const RPCMEM_HEAP_NOREG: u32 = 0x4000_0000;

/// `RPCMEM_TRY_MAP_STATIC` — static map hint. SDK header 발견 정의값. 일부
/// fastrpc 경로에서 mmap 통과 조건. β-1.MAP fallback 후보로 보유.
pub const RPCMEM_TRY_MAP_STATIC: u32 = 0x0400_0000;

/// HTP architecture version. S25 = Hexagon v79 (NSP v79).
pub const HTP_ARCH_V79: u32 = 79;

/// `enum fastrpc_map_flags` — QC fastrpc 오픈소스 `inc/remote.h`.
/// β-1.MAP 결정값: 진짜 `FASTRPC_MAP_FD = 2`. 이전 `16` 은 `FASTRPC_MAP_FD_NOMAP`
/// 변종 ("등록만 하고 실제 mmap 은 하지 않음") 으로, `fastrpc_mmap rc=0x1 EIO`
/// 의 root cause 였음. llama.cpp 가 `FASTRPC_MAP_FD` 심볼 그대로 사용해
/// SDK header 의 enum 값에 의존했기에 본 PoC 도 우연히 16 을 잘못 가져왔던 것.
pub const FASTRPC_MAP_FD: u32 = 2;

// ── FastRPC request structs (mirror of remote.h) ──

/// `struct remote_rpc_reserve_new_session` — remote.h.
#[repr(C)]
#[derive(Debug)]
pub struct RemoteRpcReserveNewSession {
    pub domain_name: *mut u8,
    pub domain_name_len: u32,
    pub session_name: *mut u8,
    pub session_name_len: u32,
    /// [out] effective_domain_id (CDSP 의 경우 보통 3).
    pub effective_domain_id: u32,
    /// [out] session_id.
    pub session_id: u32,
}

/// `struct remote_rpc_control_unsigned_module` — remote.h.
#[repr(C)]
pub struct RemoteRpcControlUnsignedModule {
    pub domain: i32,
    pub enable: i32,
}

/// `struct remote_rpc_get_uri` — remote.h. session-encoded URI 획득용.
///
/// `FASTRPC_GET_URI` 가 reserve_session 으로 얻은 session_id 를
/// session-encoded URI 로 변환 (default domain → session-specific domain).
/// 본 step 누락 시 raw URI 로 handle_open 호출 → fallback to default
/// domain 3 → `Error 0xe (AEE_ENOSUCHMOD) domain >= 0` 로 fail (실측 검증).
#[repr(C)]
pub struct RemoteRpcGetUri {
    pub domain_name: *mut u8,
    pub domain_name_len: u32,
    pub session_id: u32,
    pub module_uri: *mut u8,
    pub module_uri_len: u32,
    pub uri: *mut u8,
    pub uri_len: u32,
}

/// `struct remote_rpc_control_latency` — remote.h.
#[repr(C)]
pub struct RemoteRpcControlLatency {
    pub enable: u32,
    pub latency: u32,
}

// ── dspqueue_buffer (transport companion) ──
//
// dspqueue header 의 struct 정의. ggml-hexagon `htp_req_buff_init` 로부터
// field layout 역추적 (fd / ptr / offset / size / flags). DSP-side 매핑이
// 필요해 `#[repr(C)]` 정확성 필수.

/// `struct dspqueue_buffer` — dspqueue_write/read 의 buffer descriptor.
///
/// llama.cpp `ggml-hexagon.cpp:2224-2255` (htp_req_buff_init) 에서 사용되는
/// field 만 명시. SDK 헤더가 stock S25 에 있지 않아 ggml-hexagon 의 사용
/// 패턴을 ground truth 로 채택.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct DspQueueBuffer {
    /// rpcmem fd (mandatory).
    pub fd: c_int,
    /// host virtual address (mandatory).
    pub ptr: *mut c_void,
    /// offset within the mapped region.
    pub offset: u32,
    /// byte size to transfer.
    pub size: u32,
    /// cache maintenance flags. `DSPQUEUE_BUFFER_FLAG_*`.
    pub flags: u32,
    /// pad / reserved. ggml-hexagon 은 memset(0) 후 명시 field 만 set.
    pub reserved: [u32; 3],
}

impl DspQueueBuffer {
    pub const fn zeroed() -> Self {
        Self {
            fd: 0,
            ptr: core::ptr::null_mut(),
            offset: 0,
            size: 0,
            flags: 0,
            reserved: [0; 3],
        }
    }
}

// SAFETY: ptr 은 rpcmem 의 host virtual address. ownership 은 RpcmemBuffer
// 가 가지며 본 struct 는 transport-time 참조만 한다. Send/Sync 는 host 가
// FFI handle 을 thread-safe 하게 관리하면 충족.
unsafe impl Send for DspQueueBuffer {}
unsafe impl Sync for DspQueueBuffer {}

// ── FFI function typedefs (17 symbols) ──

pub type RpcmemAllocFn =
    unsafe extern "C" fn(heapid: c_int, flags: u32, size: c_int) -> *mut c_void;
pub type RpcmemAlloc2Fn =
    unsafe extern "C" fn(heapid: c_int, flags: u32, size: usize) -> *mut c_void;
pub type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
pub type RpcmemToFdFn = unsafe extern "C" fn(po: *mut c_void) -> c_int;

pub type FastrpcMmapFn = unsafe extern "C" fn(
    domain: c_int,
    fd: c_int,
    addr: *mut c_void,
    offset: c_int,
    length: usize,
    flags: u32,
) -> c_int;
pub type FastrpcMunmapFn =
    unsafe extern "C" fn(domain: c_int, fd: c_int, addr: *mut c_void, length: usize) -> c_int;

pub type DspqueueCreateFn = unsafe extern "C" fn(
    domain: c_int,
    flags: u32,
    req_queue_size: u32,
    resp_queue_size: u32,
    packet_callback: *mut c_void,
    error_callback: *mut c_void,
    callback_context: *mut c_void,
    queue: *mut DspQueueT,
) -> c_int;
pub type DspqueueCloseFn = unsafe extern "C" fn(queue: DspQueueT) -> c_int;
pub type DspqueueExportFn = unsafe extern "C" fn(queue: DspQueueT, queue_id: *mut u64) -> c_int;
pub type DspqueueWriteFn = unsafe extern "C" fn(
    queue: DspQueueT,
    flags: u32,
    num_buffers: u32,
    buffers: *mut DspQueueBuffer,
    message_length: u32,
    message: *const u8,
    timeout_us: u32,
) -> c_int;
pub type DspqueueReadFn = unsafe extern "C" fn(
    queue: DspQueueT,
    flags: *mut u32,
    max_buffers: u32,
    num_buffers: *mut u32,
    buffers: *mut DspQueueBuffer,
    max_message_length: u32,
    message_length: *mut u32,
    message: *mut u8,
    timeout_us: u32,
) -> c_int;

pub type RemoteHandle64OpenFn =
    unsafe extern "C" fn(name: *const u8, ph: *mut RemoteHandle64) -> c_int;
pub type RemoteHandle64CloseFn = unsafe extern "C" fn(h: RemoteHandle64) -> c_int;
pub type RemoteHandle64ControlFn =
    unsafe extern "C" fn(h: RemoteHandle64, req: u32, data: *mut c_void, datalen: u32) -> c_int;
pub type RemoteSessionControlFn =
    unsafe extern "C" fn(req: u32, data: *mut c_void, datalen: u32) -> c_int;

/// `remote_register_buf_attr2` — rpcmem buffer 를 FastRPC call 에 사용 가능하도록
/// driver-internal table 에 등록. `fastrpc_mmap` 와는 별개 (mmap = SMMU 매핑,
/// register = ref count + zero-copy 가능 표시). dspqueue_write 가 buffer 를
/// attach 할 때 driver 가 본 table 을 lookup 해 ref count 를 +1; 미등록 buffer
/// 면 ref=-1 underflow 로 `AEE_EUNABLETOLOAD (0xe)` fail.
///
/// QC `inc/remote.h`: `void remote_register_buf_attr2(void* buf, size_t size,
/// int fd, int attr);`. `fd = -1` 로 호출하면 deregister.
pub type RemoteRegisterBufAttr2Fn =
    unsafe extern "C" fn(buf: *mut c_void, size: usize, fd: c_int, attr: c_int);

/// `remote_register_buf_attr2` 의 `attr` 인자 상수 (QC `inc/remote.h`).
///
/// PoC 본 sprint 는 `FASTRPC_ATTR_NONE = 0` 사용 (default, llama.cpp 동일).
pub const FASTRPC_ATTR_NONE: c_int = 0;
pub const FASTRPC_ATTR_NON_COHERENT: c_int = 2;
pub const FASTRPC_ATTR_COHERENT: c_int = 4;
pub const FASTRPC_ATTR_KEEP_MAP: c_int = 8;

// Note: `remote_handle64_invoke` 는 htp_iface_start/stop lifecycle IDL call
// 용으로만 사용 (op dispatch path 아님). signature 단순화: scalars + raw arg
// array. PoC scope 에서는 직접 사용하지 않고 host.rs 의 lifecycle helper
// 가 wrap.
pub type RemoteHandle64InvokeFn =
    unsafe extern "C" fn(h: RemoteHandle64, scalars: u32, pra: *mut c_void) -> c_int;

// ── HtpFastrpcHost ──

/// FastRPC handle/session lifecycle + 17 dlsym symbol container.
///
/// process lifetime 동안 정확히 1회 생성, drop 시 정확히 1회 teardown
/// (INV-HTP-FRPC-001). `Arc<HtpFastrpcHost>` 로 wrap 하여 backend / buffer /
/// session 간 공유.
///
/// PoC scope 에서는 `htp_iface_start` / `_stop` 의 진짜 IDL invocation 은
/// stub (Phase 4 작업). 본 모듈은 4-step handshake 까지만 진행하며,
/// queue/handle 은 setup 되어 있되 DSP-side skel 은 idle 상태.
pub struct HtpFastrpcHost {
    /// dlopen library handle. drop 시 dlclose.
    #[allow(dead_code)]
    lib: libloading::Library,

    /// 사용된 libcdsprpc.so 경로 (진단용).
    pub lib_path: String,

    // 17 FFI function pointers.
    pub rpcmem_alloc: RpcmemAllocFn,
    pub rpcmem_alloc2: Option<RpcmemAlloc2Fn>,
    pub rpcmem_free: RpcmemFreeFn,
    pub rpcmem_to_fd: RpcmemToFdFn,
    pub fastrpc_mmap: FastrpcMmapFn,
    pub fastrpc_munmap: FastrpcMunmapFn,
    /// `remote_register_buf_attr2` — optional. 미export device 는 None →
    /// `RpcmemBuffer::alloc` 가 silent skip (이전 동작 유지).
    pub remote_register_buf_attr2: Option<RemoteRegisterBufAttr2Fn>,
    pub dspqueue_create: DspqueueCreateFn,
    pub dspqueue_close: DspqueueCloseFn,
    pub dspqueue_export: DspqueueExportFn,
    pub dspqueue_write: DspqueueWriteFn,
    pub dspqueue_read: DspqueueReadFn,
    pub remote_handle64_open: RemoteHandle64OpenFn,
    pub remote_handle64_close: RemoteHandle64CloseFn,
    pub remote_handle64_control: RemoteHandle64ControlFn,
    pub remote_handle64_invoke: RemoteHandle64InvokeFn,
    pub remote_session_control: RemoteSessionControlFn,

    /// CDSP effective_domain_id (CDSP=3 default, FASTRPC_RESERVE 후 갱신).
    pub domain_id: i32,
    /// FastRPC session id.
    pub session_id: u32,
    /// Open 된 remote handle (htp_iface skel). 0 = invalid.
    pub handle: RemoteHandle64,
    /// dspqueue handle. null = invalid.
    pub queue: DspQueueT,
    /// dspqueue_export 로 얻은 queue id (DSP 측에서 import 시 사용).
    pub queue_id: u64,
    /// `htp_iface_start` 호출 후 true. Drop / shutdown 에서 stop 호출 여부
    /// 분기. `Acquire`/`Release` 로 publish.
    iface_started: AtomicBool,
}

/// FastRPC `remote_arg` (buf variant). `remote.h` 에서:
///
/// ```c
/// typedef struct remote_buf { void *pv; size_t nLen; } remote_buf;
/// typedef union remote_arg { remote_buf buf; uint32_t h; } remote_arg;
/// ```
///
/// aarch64-linux-android: pv (8B) + n_len (8B) = 16B, alignment 8.
/// htp_iface 의 lifecycle method 는 buf variant 만 사용한다 (handle variant
/// 는 dspqueue 등 다른 경로용).
#[repr(C)]
struct RemoteArgBuf {
    pv: *mut c_void,
    n_len: usize,
}

/// `REMOTE_SCALARS_MAKEX(nAttr, nMethod, nIn, nOut, noIn, noOut)` — remote.h.
///
/// ```c
/// #define REMOTE_SCALARS_MAKEX(nAttr,nMethod,nIn,nOut,noIn,noOut) \
///     ((((uint32_t)  (nAttr) & 0x07) << 29) | \
///      (((uint32_t)(nMethod) & 0x1f) << 24) | \
///      (((uint32_t)    (nIn) & 0xff) << 16) | \
///      (((uint32_t)   (nOut) & 0xff) <<  8) | \
///      (((uint32_t)   (noIn) & 0x0f) <<  4) | \
///       ((uint32_t)  (noOut)         & 0x0f))
/// ```
#[inline]
const fn remote_scalars_makex(
    attr: u32,
    method: u32,
    n_in: u32,
    n_out: u32,
    no_in: u32,
    no_out: u32,
) -> u32 {
    ((attr & 0x7) << 29)
        | ((method & 0x1f) << 24)
        | ((n_in & 0xff) << 16)
        | ((n_out & 0xff) << 8)
        | ((no_in & 0xf) << 4)
        | (no_out & 0xf)
}

// ── htp_iface method IDs (llama.cpp build-snapdragon/.../htp_iface_stub.c) ──
//
// `static const Method methods[4] = { ... }` 의 array index = method id.
// open/close 는 `remote_handle64_{open,close}` 가 직접 처리 → invoke 안 씀.
//
// | method        | mid | sc (host)             | primIn size | layout                                      |
// |---------------|-----|-----------------------|-------------|---------------------------------------------|
// | start         | 2   | MAKEX(0,2,1,0,0,0)    | 24 B        | u32 sess_id @0, u64 queue_id @8, u32 nhvx @16 |
// | stop          | 3   | MAKEX(0,3,0,0,0,0)    | 0 (no pra)  | —                                           |
// | enable_etm    | 4   | MAKEX(0,4,0,0,0,0)    | 0           | —                                           |
// | disable_etm   | 5   | MAKEX(0,5,0,0,0,0)    | 0           | —                                           |

const HTP_IFACE_MID_START: u32 = 2;
const HTP_IFACE_MID_STOP: u32 = 3;
const HTP_IFACE_MID_ENABLE_ETM: u32 = 4;
const HTP_IFACE_MID_DISABLE_ETM: u32 = 5;

// SAFETY: 모든 FFI symbol pointer 는 dlopen 으로부터 얻은 process-global
// 주소이며 thread-safe 호출이 보장된다 (libcdsprpc 자체가 multi-thread
// safe). handle/queue 는 backend instance 안에서 동기화 책임을 갖는다.
unsafe impl Send for HtpFastrpcHost {}
unsafe impl Sync for HtpFastrpcHost {}

impl HtpFastrpcHost {
    /// 새 host 인스턴스를 생성한다 (INV-HTP-FRPC-001 lifecycle setup).
    ///
    /// 시퀀스:
    /// 1. `/vendor/lib64/libcdsprpc.so` (fallback `/data/local/tmp/...`) dlopen
    /// 2. 17 symbol dlsym
    /// 3. `FASTRPC_RESERVE_NEW_SESSION` → effective_domain_id
    /// 4. `DSPRPC_CONTROL_UNSIGNED_MODULE(enable=1)`
    /// 5. `remote_handle64_open(uri)` → handle
    /// 6. `dspqueue_create(128KB req, 64KB resp)` → queue
    /// 7. `dspqueue_export(queue)` → queue_id
    ///
    /// PoC scope: `htp_iface_start` 의 IDL invocation 은 Phase 4 작업.
    ///
    /// Target = `aarch64-linux-android` 만 active.
    #[cfg(target_os = "android")]
    pub fn new(session_name: &str) -> Result<Arc<Self>> {
        Self::new_internal(session_name)
    }

    /// non-android target stub. dry-run prototype 과 동등하게 호스트 빌드는
    /// 통과하나 호출 시 unavailable 에러.
    #[cfg(not(target_os = "android"))]
    pub fn new(_session_name: &str) -> Result<Arc<Self>> {
        Err(anyhow!(
            "htp_fastrpc: backend unavailable on this target (aarch64-linux-android only)"
        ))
    }

    /// 내부 구현 (android target). dlopen + 17 dlsym + 4-step handshake +
    /// dspqueue create/export.
    #[cfg(target_os = "android")]
    fn new_internal(session_name: &str) -> Result<Arc<Self>> {
        // ── Step 1: dlopen libcdsprpc.so ──
        let candidates = [
            "/vendor/lib64/libcdsprpc.so",
            "/data/local/tmp/libcdsprpc.so",
            "libcdsprpc.so",
        ];
        let mut loaded: Option<(libloading::Library, String)> = None;
        for &path in &candidates {
            match unsafe { libloading::Library::new(path) } {
                Ok(lib) => {
                    loaded = Some((lib, path.to_string()));
                    break;
                }
                Err(_) => continue,
            }
        }
        let (lib, lib_path) = loaded
            .ok_or_else(|| anyhow!("htp_fastrpc: libcdsprpc.so not found in any candidate path"))?;

        // ── Step 2: dlsym 17 symbols ──
        //
        // SAFETY: 모든 symbol 은 libcdsprpc.so 의 공개 entry point. ggml-hexagon
        // htp-drv.cpp:338-354 와 동일 list.
        macro_rules! dlsym {
            ($lib:expr, $ty:ty, $sym:literal) => {{
                let s: libloading::Symbol<$ty> = unsafe {
                    $lib.get($sym).with_context(|| {
                        format!(
                            "htp_fastrpc: dlsym {} failed",
                            String::from_utf8_lossy($sym)
                        )
                    })?
                };
                *s
            }};
        }

        let rpcmem_alloc: RpcmemAllocFn = dlsym!(lib, RpcmemAllocFn, b"rpcmem_alloc\0");
        // rpcmem_alloc2 는 optional (구 driver 호환). htp-drv.cpp:339 의
        // `ignore=true` 와 동일.
        let rpcmem_alloc2: Option<RpcmemAlloc2Fn> = unsafe {
            lib.get::<RpcmemAlloc2Fn>(b"rpcmem_alloc2\0")
                .ok()
                .map(|s| *s)
        };
        let rpcmem_free: RpcmemFreeFn = dlsym!(lib, RpcmemFreeFn, b"rpcmem_free\0");
        let rpcmem_to_fd: RpcmemToFdFn = dlsym!(lib, RpcmemToFdFn, b"rpcmem_to_fd\0");
        let fastrpc_mmap: FastrpcMmapFn = dlsym!(lib, FastrpcMmapFn, b"fastrpc_mmap\0");
        let fastrpc_munmap: FastrpcMunmapFn = dlsym!(lib, FastrpcMunmapFn, b"fastrpc_munmap\0");
        // remote_register_buf_attr2 는 optional. libcdsprpc.so 의 strings export
        // 에 존재 (S25 vendor lib 검증) 하지만 구 driver 호환 위해 None tolerant.
        let remote_register_buf_attr2: Option<RemoteRegisterBufAttr2Fn> = unsafe {
            lib.get::<RemoteRegisterBufAttr2Fn>(b"remote_register_buf_attr2\0")
                .ok()
                .map(|s| *s)
        };
        let dspqueue_create: DspqueueCreateFn = dlsym!(lib, DspqueueCreateFn, b"dspqueue_create\0");
        let dspqueue_close: DspqueueCloseFn = dlsym!(lib, DspqueueCloseFn, b"dspqueue_close\0");
        let dspqueue_export: DspqueueExportFn = dlsym!(lib, DspqueueExportFn, b"dspqueue_export\0");
        let dspqueue_write: DspqueueWriteFn = dlsym!(lib, DspqueueWriteFn, b"dspqueue_write\0");
        let dspqueue_read: DspqueueReadFn = dlsym!(lib, DspqueueReadFn, b"dspqueue_read\0");
        let remote_handle64_open: RemoteHandle64OpenFn =
            dlsym!(lib, RemoteHandle64OpenFn, b"remote_handle64_open\0");
        let remote_handle64_close: RemoteHandle64CloseFn =
            dlsym!(lib, RemoteHandle64CloseFn, b"remote_handle64_close\0");
        let remote_handle64_control: RemoteHandle64ControlFn =
            dlsym!(lib, RemoteHandle64ControlFn, b"remote_handle64_control\0");
        let remote_handle64_invoke: RemoteHandle64InvokeFn =
            dlsym!(lib, RemoteHandle64InvokeFn, b"remote_handle64_invoke\0");
        let remote_session_control: RemoteSessionControlFn =
            dlsym!(lib, RemoteSessionControlFn, b"remote_session_control\0");

        // ── Step 3: FASTRPC_RESERVE_NEW_SESSION ──
        let mut domain_name = CDSP_DOMAIN_NAME.as_bytes().to_vec();
        domain_name.push(0); // null-terminate
        let mut session_name_bytes = session_name.as_bytes().to_vec();
        session_name_bytes.push(0);

        let mut reserve_req = RemoteRpcReserveNewSession {
            domain_name: domain_name.as_mut_ptr(),
            domain_name_len: (domain_name.len() - 1) as u32,
            session_name: session_name_bytes.as_mut_ptr(),
            session_name_len: (session_name_bytes.len() - 1) as u32,
            effective_domain_id: 0,
            session_id: 0,
        };
        let reserve_size = core::mem::size_of::<RemoteRpcReserveNewSession>() as u32;

        // SAFETY: req struct + size 가 일치하며 buffer 는 stack live.
        let rc = unsafe {
            remote_session_control(
                FASTRPC_RESERVE_NEW_SESSION,
                &mut reserve_req as *mut _ as *mut c_void,
                reserve_size,
            )
        };
        if rc != AEE_SUCCESS {
            return Err(map_aee_err(rc))
                .with_context(|| "htp_fastrpc: FASTRPC_RESERVE_NEW_SESSION");
        }
        let domain_id = reserve_req.effective_domain_id as i32;
        let session_id = reserve_req.session_id;

        // ── Step 4: DSPRPC_CONTROL_UNSIGNED_MODULE ──
        let mut unsigned_req = RemoteRpcControlUnsignedModule {
            domain: domain_id,
            enable: 1,
        };
        let unsigned_size = core::mem::size_of::<RemoteRpcControlUnsignedModule>() as u32;
        // SAFETY: 동일.
        let rc = unsafe {
            remote_session_control(
                DSPRPC_CONTROL_UNSIGNED_MODULE,
                &mut unsigned_req as *mut _ as *mut c_void,
                unsigned_size,
            )
        };
        if rc != AEE_SUCCESS {
            return Err(map_aee_err(rc))
                .with_context(|| "htp_fastrpc: DSPRPC_CONTROL_UNSIGNED_MODULE");
        }

        // ── Step 5a: FASTRPC_GET_URI — session-encoded URI 획득 ──
        //
        // 누락 시 raw URI 로 handle_open 호출 → default domain 3 fallback →
        // `Error 0xe (AEE_ENOSUCHMOD) domain >= 0` (Q-2.2-α Phase 6 실측에서 확인,
        // llama.cpp ggml-hexagon.cpp:1580-1606 와 동일 sequence 보정).
        let raw_uri = format!(
            "file:///libggml-htp-v{arch}.so?htp_iface_skel_handle_invoke&_modver=1.0",
            arch = HTP_ARCH_V79,
        );
        let mut raw_uri_bytes = raw_uri.as_bytes().to_vec();
        raw_uri_bytes.push(0);
        let mut session_uri_buf = vec![0u8; 256];

        let mut get_uri_req = RemoteRpcGetUri {
            domain_name: domain_name.as_mut_ptr(),
            domain_name_len: (domain_name.len() - 1) as u32,
            session_id,
            module_uri: raw_uri_bytes.as_mut_ptr(),
            module_uri_len: (raw_uri_bytes.len() - 1) as u32,
            uri: session_uri_buf.as_mut_ptr(),
            uri_len: session_uri_buf.len() as u32,
        };
        let get_uri_size = core::mem::size_of::<RemoteRpcGetUri>() as u32;
        // SAFETY: 모든 raw ptr 가 ScopedLive vec 의 backing. struct lifetime 호출 종료까지.
        let rc = unsafe {
            remote_session_control(
                FASTRPC_GET_URI,
                &mut get_uri_req as *mut _ as *mut c_void,
                get_uri_size,
            )
        };
        let uri = if rc == AEE_SUCCESS {
            // session_uri_buf 안에 null-terminated 문자열이 채워짐.
            let nul = session_uri_buf
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(session_uri_buf.len());
            String::from_utf8_lossy(&session_uri_buf[..nul]).into_owned()
        } else {
            // fallback to raw URI (single-session path)
            eprintln!("htp_fastrpc: FASTRPC_GET_URI failed rc={rc:#x}, fallback to raw URI");
            raw_uri.clone()
        };

        // ── Step 5b: remote_handle64_open(session_uri) ──
        let uri_c =
            CString::new(uri.as_str()).map_err(|e| anyhow!("htp_fastrpc: URI null byte: {}", e))?;
        let mut handle: RemoteHandle64 = 0;
        // SAFETY: uri_c live; handle out-pointer 유효.
        let rc =
            unsafe { remote_handle64_open(uri_c.as_ptr() as *const u8, &mut handle as *mut _) };
        if rc != AEE_SUCCESS {
            return Err(map_aee_err(rc))
                .with_context(|| format!("htp_fastrpc: remote_handle64_open uri={uri}"));
        }

        // ── Step 6 (best-effort): latency QoS ──
        //
        // 실패해도 fatal 아님. ggml-hexagon 동일 policy (warning only).
        {
            let mut latency_req = RemoteRpcControlLatency {
                enable: 1,
                latency: 0,
            };
            let lsz = core::mem::size_of::<RemoteRpcControlLatency>() as u32;
            // SAFETY: handle live, struct stack live.
            let _ = unsafe {
                remote_handle64_control(
                    handle,
                    DSPRPC_CONTROL_LATENCY,
                    &mut latency_req as *mut _ as *mut c_void,
                    lsz,
                )
            };
        }

        // ── Step 7: dspqueue_create(128KB req, 64KB resp) ──
        let mut queue: DspQueueT = core::ptr::null_mut();
        // SAFETY: callback/context null OK (we drive read/write explicitly,
        // ggml-hexagon 와 동일 policy).
        let rc = unsafe {
            dspqueue_create(
                domain_id,
                0,          // flags
                128 * 1024, // req queue
                64 * 1024,  // resp queue
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                &mut queue as *mut _,
            )
        };
        if rc != AEE_SUCCESS {
            // teardown handle before returning error
            let _ = unsafe { remote_handle64_close(handle) };
            return Err(map_aee_err(rc)).with_context(|| "htp_fastrpc: dspqueue_create");
        }

        // ── Step 8: dspqueue_export(queue, &queue_id) ──
        let mut queue_id: u64 = 0;
        // SAFETY: queue live.
        let rc = unsafe { dspqueue_export(queue, &mut queue_id as *mut _) };
        if rc != AEE_SUCCESS {
            let _ = unsafe { dspqueue_close(queue) };
            let _ = unsafe { remote_handle64_close(handle) };
            return Err(map_aee_err(rc)).with_context(|| "htp_fastrpc: dspqueue_export");
        }

        // `htp_iface_start(handle, sess_id, queue_id, n_hvx)` 는 caller
        // (HtpFastrpcBackend::new) 가 본 인스턴스의 [`try_start_iface`] 를
        // 명시 호출한다. n_hvx 정책 분리를 위해 new_internal 은 setup ready
        // state 까지만 보장.

        Ok(Arc::new(Self {
            lib,
            lib_path,
            rpcmem_alloc,
            rpcmem_alloc2,
            rpcmem_free,
            rpcmem_to_fd,
            fastrpc_mmap,
            fastrpc_munmap,
            remote_register_buf_attr2,
            dspqueue_create,
            dspqueue_close,
            dspqueue_export,
            dspqueue_write,
            dspqueue_read,
            remote_handle64_open,
            remote_handle64_close,
            remote_handle64_control,
            remote_handle64_invoke,
            remote_session_control,
            domain_id,
            session_id,
            handle,
            queue,
            queue_id,
            iface_started: AtomicBool::new(false),
        }))
    }

    /// `htp_iface_start(handle, sess_id, queue_id, n_hvx)` IDL invocation.
    ///
    /// 자동 생성 stub `htp_iface_stub.c::htp_iface_start` 의 packet schema 를
    /// Rust 로 직접 transcribe — `mid=2`, sc = `MAKEX(0,2,1,0,0,0)`, primIn 은
    /// 24 byte (u32 sess_id @0, u64 queue_id @8, u32 n_hvx @16). `remote_arg`
    /// 1 개를 buf variant 로 채워 `pv=primIn`, `n_len=24`.
    ///
    /// `n_hvx` 는 사용할 HVX vector unit 수. llama.cpp 의 `opt_nhvx` 기본값은
    /// 0 (= use all). PoC 는 0 을 권장 (DSP-side 가 device default 결정).
    ///
    /// 본 메서드는 idempotent: `iface_started` 가 이미 true 면 즉시 Ok.
    pub fn try_start_iface(&self, n_hvx: u32) -> Result<()> {
        if self.iface_started.load(Ordering::Acquire) {
            return Ok(());
        }
        if self.handle == 0 {
            return Err(anyhow!(
                "htp_fastrpc: try_start_iface called with invalid handle"
            ));
        }

        // primIn buffer — 정확히 24 byte. u64 alignment 유지.
        let mut prim_in: [u64; 3] = [0; 3];
        let prim_bytes =
            unsafe { core::slice::from_raw_parts_mut(prim_in.as_mut_ptr() as *mut u8, 24) };
        // u32 sess_id @ offset 0
        prim_bytes[0..4].copy_from_slice(&self.session_id.to_ne_bytes());
        // u64 queue_id @ offset 8 (offset 4..8 은 padding, primIn[0] 의 상위 4B)
        prim_bytes[8..16].copy_from_slice(&self.queue_id.to_ne_bytes());
        // u32 n_hvx @ offset 16
        prim_bytes[16..20].copy_from_slice(&n_hvx.to_ne_bytes());

        let mut pra = RemoteArgBuf {
            pv: prim_in.as_mut_ptr() as *mut c_void,
            n_len: 24,
        };
        let sc = remote_scalars_makex(0, HTP_IFACE_MID_START, 1, 0, 0, 0);

        // SAFETY: handle valid (4-step handshake 통과 후), pra 는 stack live
        // (호출 종료 후 즉시 무효). remote_handle64_invoke 는 동기 호출이라
        // pra 가 호출 본문 동안 유효.
        let rc = unsafe {
            (self.remote_handle64_invoke)(self.handle, sc, &mut pra as *mut _ as *mut c_void)
        };
        if rc != AEE_SUCCESS {
            return Err(map_aee_err(rc)).with_context(|| {
                format!(
                    "htp_fastrpc: htp_iface_start (sess={}, queue_id={:#x}, n_hvx={})",
                    self.session_id, self.queue_id, n_hvx
                )
            });
        }
        self.iface_started.store(true, Ordering::Release);
        Ok(())
    }

    /// `htp_iface_stop` IDL invocation. mid=3, sc = `MAKEX(0,3,0,0,0,0)`,
    /// pra = NULL (인자 없음). idempotent: 시작 안 된 상태에서 호출하면 no-op.
    ///
    /// 자동 생성 stub `_stub_method_1` 의 호출 패턴: `remote_handle64_invoke(_handle, sc, NULL)`.
    pub fn try_stop_iface(&self) -> Result<()> {
        if !self.iface_started.load(Ordering::Acquire) {
            return Ok(());
        }
        if self.handle == 0 {
            // 이미 close 된 handle — Drop 의 best-effort 호출에서 진입 가능.
            self.iface_started.store(false, Ordering::Release);
            return Ok(());
        }
        let sc = remote_scalars_makex(0, HTP_IFACE_MID_STOP, 0, 0, 0, 0);
        // SAFETY: handle valid, pra=NULL (인자 0 개).
        let rc = unsafe { (self.remote_handle64_invoke)(self.handle, sc, core::ptr::null_mut()) };
        // started flag 는 stop 호출 결과 무관 clear (재시도 방지).
        self.iface_started.store(false, Ordering::Release);
        if rc != AEE_SUCCESS {
            return Err(map_aee_err(rc)).with_context(|| "htp_fastrpc: htp_iface_stop");
        }
        Ok(())
    }

    /// `htp_iface_enable_etm` IDL invocation. mid=4. ETM = ARM/Hexagon
    /// Embedded Trace Macrocell — 하드웨어 trace unit. profiling 인프라
    /// 통합 시 사용. PoC scope 외라 wire-up 만 해두고 호출 site 는 없음.
    ///
    /// TODO(profiling): trace control flag + buffer routing 추가 후 호출.
    #[allow(dead_code)]
    pub fn try_enable_etm(&self) -> Result<()> {
        let sc = remote_scalars_makex(0, HTP_IFACE_MID_ENABLE_ETM, 0, 0, 0, 0);
        // SAFETY: handle valid (precondition).
        let rc = unsafe { (self.remote_handle64_invoke)(self.handle, sc, core::ptr::null_mut()) };
        if rc != AEE_SUCCESS {
            return Err(map_aee_err(rc)).with_context(|| "htp_fastrpc: htp_iface_enable_etm");
        }
        Ok(())
    }

    /// `htp_iface_disable_etm` IDL invocation. mid=5. ETM 비활성화. PoC
    /// scope 외 (enable_etm pair).
    #[allow(dead_code)]
    pub fn try_disable_etm(&self) -> Result<()> {
        let sc = remote_scalars_makex(0, HTP_IFACE_MID_DISABLE_ETM, 0, 0, 0, 0);
        // SAFETY: handle valid.
        let rc = unsafe { (self.remote_handle64_invoke)(self.handle, sc, core::ptr::null_mut()) };
        if rc != AEE_SUCCESS {
            return Err(map_aee_err(rc)).with_context(|| "htp_fastrpc: htp_iface_disable_etm");
        }
        Ok(())
    }

    /// `htp_iface_stop` 후 `remote_handle64_close` + `dspqueue_close` 를
    /// 호출하는 명시 teardown. `Drop` 도 동일 시퀀스를 수행하지만, error
    /// 보고가 필요한 경우 본 메서드를 명시 호출한다.
    pub fn shutdown(&mut self) -> Result<()> {
        // iface 가 started 상태면 stop 먼저 (lazy → close → dangling worker 회피).
        // stop 실패는 진단 출력만 하고 진행 (handle/queue close 우선).
        if let Err(e) = self.try_stop_iface() {
            eprintln!("htp_fastrpc: shutdown stop_iface failed (continuing close): {e}");
        }

        if !self.queue.is_null() {
            // SAFETY: queue 가 valid handle.
            let rc = unsafe { (self.dspqueue_close)(self.queue) };
            self.queue = core::ptr::null_mut();
            if rc != AEE_SUCCESS {
                return Err(map_aee_err(rc)).with_context(|| "htp_fastrpc: dspqueue_close");
            }
        }
        if self.handle != 0 {
            // SAFETY: handle 이 valid.
            let rc = unsafe { (self.remote_handle64_close)(self.handle) };
            self.handle = 0;
            if rc != AEE_SUCCESS {
                return Err(map_aee_err(rc)).with_context(|| "htp_fastrpc: remote_handle64_close");
            }
        }
        Ok(())
    }
}

impl Drop for HtpFastrpcHost {
    fn drop(&mut self) {
        // Best-effort teardown — Drop 안에서는 error 를 panic 으로 escalate
        // 하지 않는다 (INV-HTP-FRPC-001 lifecycle 의 1-shot 책임은 정상
        // 종료 경로에서 `shutdown` 으로 검증된다).
        //
        // iface_started 인 경우 DSP-side worker thread 가 dspqueue 를 detach
        // 한 뒤 handle/queue 를 close 해야 worker leak 이 없다. stop 결과는
        // 무시 (Drop 본질).
        let _ = self.try_stop_iface();
        if !self.queue.is_null() {
            // SAFETY: queue 가 valid 일 때만 호출.
            unsafe { (self.dspqueue_close)(self.queue) };
            self.queue = core::ptr::null_mut();
        }
        if self.handle != 0 {
            // SAFETY: handle 이 valid 일 때만 호출.
            unsafe { (self.remote_handle64_close)(self.handle) };
            self.handle = 0;
        }
        // lib 은 자동 drop → dlclose.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_sizes_sanity() {
        // RemoteRpcReserveNewSession on 64-bit (aarch64-linux-android):
        //   ptr(8) + u32(4) + pad(4) + ptr(8) + u32(4) + u32(4) + u32(4) + tail pad(4) = 40 B
        // C struct alignment rule: trailing pad 로 8 B alignment 보장.
        // 본 값은 ggml-hexagon (`ggml-hexagon.cpp:1561`) 에서도 동일하게
        // sizeof(n) 으로 전달된다.
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(core::mem::size_of::<RemoteRpcReserveNewSession>(), 40);
        }
        // RemoteRpcControlUnsignedModule: 2 i32 = 8 B (aligned).
        assert_eq!(core::mem::size_of::<RemoteRpcControlUnsignedModule>(), 8);
    }

    #[test]
    #[cfg(not(target_os = "android"))]
    fn new_returns_unavailable_on_host() {
        // 호스트 (non-android) 빌드에서는 stub 가 unavailable 에러를 반환.
        // HtpFastrpcHost 는 Debug 미구현 (raw fn ptr) 이라 unwrap_err 대신
        // match.
        let s = match HtpFastrpcHost::new("test") {
            Ok(_) => panic!("expected unavailable error on non-android"),
            Err(e) => format!("{e}"),
        };
        assert!(s.contains("unavailable"), "unexpected error: {s}");
    }

    #[test]
    fn dspq_buffer_zeroed_is_zero() {
        let b = DspQueueBuffer::zeroed();
        assert_eq!(b.fd, 0);
        assert!(b.ptr.is_null());
        assert_eq!(b.size, 0);
        assert_eq!(b.flags, 0);
    }

    /// `remote_scalars_makex` 값이 llama.cpp 자동 생성 stub 의
    /// `REMOTE_SCALARS_MAKEX` 와 일치하는지 검증. mid 별 sc encoding 은
    /// FastRPC wire format 의 핵심이라 transcribe 정확성을 단정한다.
    #[test]
    fn iface_method_sc_encoding() {
        // start(0, 2, 1, 0, 0, 0) = (2 << 24) | (1 << 16)
        assert_eq!(
            remote_scalars_makex(0, HTP_IFACE_MID_START, 1, 0, 0, 0),
            0x0201_0000
        );
        // stop(0, 3, 0, 0, 0, 0) = (3 << 24)
        assert_eq!(
            remote_scalars_makex(0, HTP_IFACE_MID_STOP, 0, 0, 0, 0),
            0x0300_0000
        );
        // enable_etm(0, 4, 0, 0, 0, 0) = (4 << 24)
        assert_eq!(
            remote_scalars_makex(0, HTP_IFACE_MID_ENABLE_ETM, 0, 0, 0, 0),
            0x0400_0000
        );
        // disable_etm(0, 5, 0, 0, 0, 0) = (5 << 24)
        assert_eq!(
            remote_scalars_makex(0, HTP_IFACE_MID_DISABLE_ETM, 0, 0, 0, 0),
            0x0500_0000
        );
    }

    #[test]
    fn remote_arg_buf_layout_size() {
        // aarch64 / x86_64 (LP64): pv(8) + n_len(8) = 16 B.
        assert_eq!(core::mem::size_of::<RemoteArgBuf>(), 16);
    }
}
