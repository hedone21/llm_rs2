//! HTP FastRPC backend — host↔DSP packet schema (INV-HTP-FRPC-003).
//!
//! 본 모듈은 llama.cpp `ggml/src/ggml-hexagon/htp/htp-msg.h` (commit
//! 알 수 없음, 2026-05-26 snapshot) 의 schema 와 **byte-identical** 이다.
//! 호스트 Rust `#[repr(C)]` 와 DSP C struct 의 byte layout 이 1 bit 라도
//! 어긋나면 garbage 가 silently 처리된다 (DSP 측 OOB read/write 가
//! crash 가 아닌 wrong-answer 로 나타남). schema 변경 권한은 본 프로젝트
//! 에 없으며 llama.cpp upstream 을 따라간다.
//!
//! 출처: llama.cpp `ggml/src/ggml-hexagon/htp/htp-msg.h:118-150`. MIT
//! attribution 은 `arch/htp_fastrpc.md` §6 참조.

#![cfg(feature = "htp_fastrpc")]
#![allow(non_camel_case_types, non_upper_case_globals)]

// ── 상수 ──

/// `HTP_MAX_DIMS` — `htp_tensor.ne` / `nb` 의 길이.
pub const HTP_MAX_DIMS: usize = 4;

/// `HTP_MAX_OP_PARAMS` (bytes) — `htp_general_req.op_params` 의 byte 길이.
/// llama.cpp 정의: `HTP_MAX_OP_PARAMS / sizeof(int32_t) = 64 / 4 = 16` slot.
pub const HTP_MAX_OP_PARAMS_BYTES: usize = 64;
pub const HTP_MAX_OP_PARAMS_SLOTS: usize = HTP_MAX_OP_PARAMS_BYTES / 4;

/// `HTP_MAX_PACKET_BUFFERS` — 한 packet 의 dspqueue_buffer 최대 개수.
pub const HTP_MAX_PACKET_BUFFERS: u32 = 8;

// ── data type enum (htp_data_type) ──

pub const HTP_TYPE_F32: u32 = 0;
pub const HTP_TYPE_F16: u32 = 1;
pub const HTP_TYPE_Q4_0: u32 = 2;
pub const HTP_TYPE_Q8_0: u32 = 8;
pub const HTP_TYPE_I32: u32 = 26;
pub const HTP_TYPE_I64: u32 = 27;
pub const HTP_TYPE_MXFP4: u32 = 39;

// ── op enum (htp_op) ──
//
// 주의: 첫 4개 (MUL/ADD/SUB/DIV) 는 index 로도 쓰여 reorder 금지. PoC scope
// 는 RMS_NORM 만 사용하나 β 단계 sprint 를 위해 전체 enum 상수를 미리 노출.

pub const HTP_OP_MUL: u32 = 0;
pub const HTP_OP_ADD: u32 = 1;
pub const HTP_OP_SUB: u32 = 2;
pub const HTP_OP_DIV: u32 = 3;
pub const HTP_OP_MUL_MAT: u32 = 4;
pub const HTP_OP_MUL_MAT_ID: u32 = 5;
pub const HTP_OP_RMS_NORM: u32 = 6;
pub const HTP_OP_UNARY_SILU: u32 = 7;
pub const HTP_OP_UNARY_GELU: u32 = 8;
pub const HTP_OP_GLU_SWIGLU: u32 = 9;
pub const HTP_OP_GLU_SWIGLU_OAI: u32 = 10;
pub const HTP_OP_GLU_GEGLU: u32 = 11;
pub const HTP_OP_SOFTMAX: u32 = 12;
pub const HTP_OP_ADD_ID: u32 = 13;
pub const HTP_OP_ROPE: u32 = 14;
pub const HTP_OP_FLASH_ATTN_EXT: u32 = 15;
pub const HTP_OP_SET_ROWS: u32 = 16;
pub const HTP_OP_GET_ROWS: u32 = 17;
pub const HTP_OP_SCALE: u32 = 18;
pub const HTP_OP_CPY: u32 = 19;
pub const HTP_OP_ARGSORT: u32 = 20;
pub const HTP_OP_SQR: u32 = 21;
pub const HTP_OP_SQRT: u32 = 22;
pub const HTP_OP_SUM_ROWS: u32 = 23;
pub const HTP_OP_SSM_CONV: u32 = 24;

// ── op flags (htp_op flags) ──

pub const HTP_OPFLAGS_SKIP_QUANTIZE: u32 = 1 << 0;
pub const HTP_OPFLAGS_SKIP_COMPUTE: u32 = 1 << 1;
pub const HTP_OPFLAGS_EARLY_WAKEUP: u32 = 1 << 2;

// ── dspqueue_buffer flags ──
//
// 본 모듈에 두는 이유: `htp_general_req` 와 함께 enqueue 되는 동반 데이터라
// idl schema 의 일부로 묶는다. dspqueue 자체는 transport 라 host.rs 에 있다.
//
// QC fastrpc 오픈소스 `inc/dspqueue.h` 정확값 (β-1.QUEUE 정정, 2026-05-26):
//
//   DSPQUEUE_BUFFER_FLAG_REF                  = 0x04
//   DSPQUEUE_BUFFER_FLAG_DEREF                = 0x08
//   DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER         = 0x10
//   DSPQUEUE_BUFFER_FLAG_INVALIDATE_SENDER    = 0x20
//   DSPQUEUE_BUFFER_FLAG_FLUSH_RECIPIENT      = 0x40
//   DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT = 0x80
//
// β-1.MAP root cause 와 동일 클래스: 이전 `1 << 0` / `1 << 1` 은 driver 가
// 인식하지 못하는 reserved/invalid bit. logcat 에 `flags 0x3` 으로 그대로
// 흘러나오며 `dspqueue_write` 가 `AEE_EUNABLETOLOAD (0xe)` + driver-side
// `fastrpc_buffer_ref ref=-1` 로 fail.

pub const DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER: u32 = 1 << 4;
pub const DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT: u32 = 1 << 7;

/// dspqueue_buffer cache 정책 (ggml-hexagon `dspqbuf_type` 차용).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DspqBufferType {
    /// CPU 가 쓰고 DSP 가 읽음 (input/weight). flush + invalidate.
    CpuWriteDspRead,
    /// DSP 가 쓰고 CPU 가 읽음 (output). flush only.
    DspWriteCpuRead,
    /// const buffer (weight read-only). cache maintenance 없음.
    Constant,
}

impl DspqBufferType {
    pub fn to_flags(self) -> u32 {
        match self {
            Self::CpuWriteDspRead => {
                DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT
            }
            Self::DspWriteCpuRead => DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER,
            Self::Constant => 0,
        }
    }
}

// ── packet structs (byte-identical to llama.cpp htp-msg.h) ──

/// `struct htp_tensor` — DSP-side tensor descriptor.
///
/// `data` 는 host 측에서는 rpcmem buffer offset (0) 으로 두고 DSP-side
/// `htp_packet_callback` 이 mapped DSP ptr 로 patch 한다. host 측은
/// dspqueue_buffer 의 fd/ptr/offset 으로 ownership 을 전달.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct HtpTensor {
    pub data: u32,
    pub type_: u32,
    pub ne: [u32; HTP_MAX_DIMS],
    pub nb: [u32; HTP_MAX_DIMS],
}

impl HtpTensor {
    pub const fn zeroed() -> Self {
        Self {
            data: 0,
            type_: 0,
            ne: [0; HTP_MAX_DIMS],
            nb: [0; HTP_MAX_DIMS],
        }
    }
}

/// `struct htp_general_req` — host → DSP op request packet.
///
/// 모든 op 가 본 schema 를 share. rmsnorm 의 경우 `op = HTP_OP_RMS_NORM`,
/// `op_params[0] = epsilon (memcpy float)`, `src0 = input`, `dst = output`,
/// `n_bufs = 2`. 미사용 src1..src4 는 zero-init.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct HtpGeneralReq {
    pub op: u32,
    pub op_params: [i32; HTP_MAX_OP_PARAMS_SLOTS],
    pub flags: u32,
    pub src0: HtpTensor,
    pub src1: HtpTensor,
    pub src2: HtpTensor,
    pub src3: HtpTensor,
    pub src4: HtpTensor,
    pub dst: HtpTensor,
}

impl HtpGeneralReq {
    pub const fn zeroed() -> Self {
        Self {
            op: 0,
            op_params: [0; HTP_MAX_OP_PARAMS_SLOTS],
            flags: 0,
            src0: HtpTensor::zeroed(),
            src1: HtpTensor::zeroed(),
            src2: HtpTensor::zeroed(),
            src3: HtpTensor::zeroed(),
            src4: HtpTensor::zeroed(),
            dst: HtpTensor::zeroed(),
        }
    }
}

/// `struct htp_general_rsp` — DSP → host op response packet.
///
/// llama.cpp 정의 그대로 (htp-msg.h:143-150). pad 까지 포함해 정확히 64 B.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct HtpGeneralRsp {
    pub op: u32,
    pub status: u32,
    pub prof_usecs: u32,
    pub prof_cycles: u32,
    pub prof_pkts: u32,
    pub unused: [u8; 44],
}

impl HtpGeneralRsp {
    pub const fn zeroed() -> Self {
        Self {
            op: 0,
            status: 0,
            prof_usecs: 0,
            prof_cycles: 0,
            prof_pkts: 0,
            unused: [0; 44],
        }
    }
}

// ── compile-time size assertions ──
//
// schema 변경 시 컴파일 fail 로 즉시 발견. llama.cpp htp-msg.h 와 byte-
// identical 보장.

const _: () = {
    // HtpTensor = u32 + u32 + 4*u32 + 4*u32 = 40 B
    assert!(core::mem::size_of::<HtpTensor>() == 40);
    // op(4) + op_params(64) + flags(4) + 6 * HtpTensor(40) = 312 B
    assert!(core::mem::size_of::<HtpGeneralReq>() == 4 + 64 + 4 + 6 * 40);
    // op + status + 3 prof + 44 pad = 5*4 + 44 = 64 B
    assert!(core::mem::size_of::<HtpGeneralRsp>() == 64);
    // INV-HTP-FRPC-003: htp_general_req should be 8-byte aligned (cacheline-
    // friendly). 312 % 8 == 0.
    assert!(core::mem::size_of::<HtpGeneralReq>().is_multiple_of(8));
};

// ── helpers ──

/// `tensor` 의 shape/stride 를 `HtpTensor` 로 변환. `data` 는 0 (DSP
/// 측에서 patch).
pub fn htp_tensor_from_shape(
    type_: u32,
    ne: [u32; HTP_MAX_DIMS],
    nb: [u32; HTP_MAX_DIMS],
) -> HtpTensor {
    HtpTensor {
        data: 0,
        type_,
        ne,
        nb,
    }
}

/// rmsnorm op req 초기화. `op_params[0]` 슬롯에 epsilon 을 float 로 memcpy.
///
/// llama.cpp `unary-ops.c::rms_norm` 의 epsilon 디코드 방식과 일치.
pub fn init_rmsnorm_req(req: &mut HtpGeneralReq, eps: f32, src0: HtpTensor, dst: HtpTensor) {
    req.op = HTP_OP_RMS_NORM;
    req.flags = 0;
    // op_params[0] 슬롯에 float bit pattern 으로 기록.
    let eps_bits = eps.to_bits() as i32;
    req.op_params[0] = eps_bits;
    req.src0 = src0;
    req.src1 = HtpTensor::zeroed();
    req.src2 = HtpTensor::zeroed();
    req.src3 = HtpTensor::zeroed();
    req.src4 = HtpTensor::zeroed();
    req.dst = dst;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_sizes_match_spec() {
        assert_eq!(core::mem::size_of::<HtpTensor>(), 40);
        assert_eq!(core::mem::size_of::<HtpGeneralReq>(), 312);
        assert_eq!(core::mem::size_of::<HtpGeneralRsp>(), 64);
    }

    #[test]
    fn op_constants_match_llamacpp() {
        // llama.cpp htp-msg.h:46-72 그대로
        assert_eq!(HTP_OP_MUL, 0);
        assert_eq!(HTP_OP_ADD, 1);
        assert_eq!(HTP_OP_MUL_MAT, 4);
        assert_eq!(HTP_OP_RMS_NORM, 6);
        assert_eq!(HTP_OP_FLASH_ATTN_EXT, 15);
    }

    #[test]
    fn type_constants_match_llamacpp() {
        assert_eq!(HTP_TYPE_F32, 0);
        assert_eq!(HTP_TYPE_F16, 1);
        assert_eq!(HTP_TYPE_Q4_0, 2);
        assert_eq!(HTP_TYPE_Q8_0, 8);
    }

    #[test]
    fn init_rmsnorm_req_writes_eps_bits() {
        let mut req = HtpGeneralReq::zeroed();
        let src = htp_tensor_from_shape(HTP_TYPE_F32, [1536, 1, 1, 1], [4, 6144, 6144, 6144]);
        let dst = htp_tensor_from_shape(HTP_TYPE_F32, [1536, 1, 1, 1], [4, 6144, 6144, 6144]);
        init_rmsnorm_req(&mut req, 1e-5_f32, src, dst);
        assert_eq!(req.op, HTP_OP_RMS_NORM);
        // op_params[0] 의 bit pattern 이 1e-5 f32 와 일치
        let decoded = f32::from_bits(req.op_params[0] as u32);
        assert!((decoded - 1e-5_f32).abs() < f32::EPSILON);
        assert_eq!(req.src0.type_, HTP_TYPE_F32);
        assert_eq!(req.dst.type_, HTP_TYPE_F32);
        // 미사용 src1~src4 는 zero
        assert_eq!(req.src1.type_, 0);
        assert_eq!(req.src1.ne, [0; HTP_MAX_DIMS]);
    }

    #[test]
    fn dspq_buffer_type_flags() {
        assert_eq!(
            DspqBufferType::CpuWriteDspRead.to_flags(),
            DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT
        );
        assert_eq!(
            DspqBufferType::DspWriteCpuRead.to_flags(),
            DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
        );
        assert_eq!(DspqBufferType::Constant.to_flags(), 0);
    }
}
