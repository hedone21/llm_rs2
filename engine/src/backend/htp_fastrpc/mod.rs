//! HTP FastRPC backend — module entry point.
//!
//! Q-2.2-α PoC Phase 3 산출물. 본 모듈은 5 파일 분해로 INV 1:1 매칭을
//! 갖는다:
//!
//! | 파일 | 책임 | INV |
//! |------|------|-----|
//! | `host.rs`   | dlopen + 17 dlsym + handshake + handle/queue lifecycle | INV-HTP-FRPC-001, 004 |
//! | `idl.rs`    | host↔DSP packet schema (llama.cpp htp-msg.h 동기화)     | INV-HTP-FRPC-003 |
//! | `buffer.rs` | rpcmem allocator wrapper (RAII free, single-owner)     | INV-HTP-FRPC-002 |
//! | `error.rs`  | AEEResult → engine error mapping                       | INV-HTP-FRPC-004 |
//! | `mod.rs`    | Backend trait impl placeholder + re-export             | INV-HTP-FRPC-005 (skeleton) |
//!
//! **PoC scope**: Phase 3 = host binding 5 파일 분리만. `Backend` trait impl
//! 의 진짜 method 구현 (rmsnorm 진짜 HVX 호출, write_buffer / synchronize,
//! cpu_companion 위임) 은 Phase 4 작업.
//!
//! 자세한 spec: `spec/htp_fastrpc.md`. 아키텍처: `arch/htp_fastrpc.md`.

pub mod buffer;
pub mod error;
pub mod host;
pub mod idl;

pub use buffer::RpcmemBuffer;
pub use error::{
    AEE_SUCCESS, HTP_STATUS_OK, KNOWN_QNN_DEVICE_CREATE_FAIL, aee_label, map_aee_err,
    map_htp_status,
};
pub use host::{
    CDSP_DOMAIN_NAME, DSPQUEUE_TIMEOUT, DspQueueBuffer, DspQueueT, FASTRPC_RESERVE_NEW_SESSION,
    HTP_ARCH_V79, HtpFastrpcHost, RPCMEM_DEFAULT_FLAGS, RPCMEM_HEAP_ID_SYSTEM, RemoteHandle64,
};
pub use idl::{
    DspqBufferType, HTP_MAX_DIMS, HTP_MAX_OP_PARAMS_SLOTS, HTP_MAX_PACKET_BUFFERS, HTP_OP_RMS_NORM,
    HTP_TYPE_F16, HTP_TYPE_F32, HTP_TYPE_Q4_0, HTP_TYPE_Q8_0, HtpGeneralReq, HtpGeneralRsp,
    HtpTensor, htp_tensor_from_shape, init_rmsnorm_req,
};

use std::sync::Arc;

use anyhow::Result;

/// HTP FastRPC backend (Phase 3 placeholder).
///
/// Phase 4 에서 `crate::backend::Backend` trait 구현 + `cpu_companion` 위임
/// 라우팅이 추가될 예정. 현재는 host lifecycle 보유만.
///
/// Drop 시 `host` 의 Arc 가 0 으로 감소하면 [`HtpFastrpcHost`] 의 RAII
/// teardown 으로 dspqueue_close + remote_handle64_close + dlclose 가
/// 순차 호출된다 (INV-HTP-FRPC-001).
pub struct HtpFastrpcBackend {
    /// FastRPC host (dlopen + handle/queue 보유). 추후 RpcmemBuffer 가
    /// 동일 Arc 를 share.
    #[allow(dead_code)]
    host: Arc<HtpFastrpcHost>,
}

impl HtpFastrpcBackend {
    /// 새 backend 인스턴스를 생성한다. host 가 4-step handshake 를 완료한
    /// 상태로 반환된다.
    ///
    /// Phase 4 에서 cpu_companion 주입 + Backend trait method dispatch 가
    /// 추가될 예정.
    pub fn new(session_name: &str) -> Result<Self> {
        let host = HtpFastrpcHost::new(session_name)?;
        Ok(Self { host })
    }

    /// 진단/테스트용 host 접근자.
    pub fn host(&self) -> &Arc<HtpFastrpcHost> {
        &self.host
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(target_os = "android"))]
    fn new_unavailable_on_host() {
        // HtpFastrpcBackend 는 Debug 미구현 (raw fn ptr 보유) 이라 unwrap_err
        // 대신 match 로 에러 추출.
        let s = match HtpFastrpcBackend::new("test") {
            Ok(_) => panic!("expected error on non-android host"),
            Err(e) => format!("{e}"),
        };
        assert!(s.contains("unavailable"), "unexpected: {s}");
    }

    #[test]
    fn re_exports_compile() {
        // 본 test 는 mod.rs 의 pub use 가 valid 한지 컴파일 타임에 검증만.
        let _: u32 = HTP_OP_RMS_NORM;
        let _: u32 = HTP_TYPE_F32;
        let _: i32 = AEE_SUCCESS;
    }
}
