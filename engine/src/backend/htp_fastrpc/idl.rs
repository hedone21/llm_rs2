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

/// MUL_MAT op req 초기화. src0=weight (constant, e.g. Q4_0), src1=input (F32),
/// dst=output (F32). op_params 슬롯 미사용 (flags 로만 제어).
///
/// llama.cpp `ggml-hexagon.cpp::init_binary_req<true>` 의 MUL_MAT 분기와 동일.
/// `skip_quantize=true` 이면 host 가 src1 을 미리 양자화했음을 알림 (PoC 에서는
/// 항상 false — DSP-side dynamic quantize 사용).
///
/// llama.cpp DSP-side 가 지원하는 weight dtype 은 Q4_0 / Q8_0 / MXFP4
/// (`matmul-ops.c::htp_mminit_vec_dot`). F32 weight 미지원.
pub fn init_matmul_req(
    req: &mut HtpGeneralReq,
    src0: HtpTensor,
    src1: HtpTensor,
    dst: HtpTensor,
    skip_quantize: bool,
) {
    req.op = HTP_OP_MUL_MAT;
    req.flags = if skip_quantize {
        HTP_OPFLAGS_SKIP_QUANTIZE
    } else {
        0
    };
    req.op_params = [0; HTP_MAX_OP_PARAMS_SLOTS];
    req.src0 = src0;
    req.src1 = src1;
    req.src2 = HtpTensor::zeroed();
    req.src3 = HtpTensor::zeroed();
    req.src4 = HtpTensor::zeroed();
    req.dst = dst;
}

/// Element-wise binary op (MUL/ADD/SUB/DIV) req 초기화. n_bufs=3 path.
///
/// llama.cpp `ggml-hexagon.cpp::init_binary_req<false>` (`_is_src0_constant=false`,
/// MUL_MAT 가 아닌 element-wise) 와 동일. src0/src1 모두 host activation 이라
/// `DSPQBUF_TYPE_CPU_WRITE_DSP_READ`. op_params 슬롯 미사용 (ggml 측에서 element-
/// wise op 는 op_params 비어 있음 — `proc_binary_req` 가 src/dst data ptr 만
/// 사용한다, `htp/main.c:602`).
pub fn init_binary_req(
    req: &mut HtpGeneralReq,
    op: u32,
    src0: HtpTensor,
    src1: HtpTensor,
    dst: HtpTensor,
) {
    debug_assert!(
        op == HTP_OP_MUL || op == HTP_OP_ADD || op == HTP_OP_SUB || op == HTP_OP_DIV,
        "init_binary_req: op must be one of MUL/ADD/SUB/DIV"
    );
    req.op = op;
    req.flags = 0;
    req.op_params = [0; HTP_MAX_OP_PARAMS_SLOTS];
    req.src0 = src0;
    req.src1 = src1;
    req.src2 = HtpTensor::zeroed();
    req.src3 = HtpTensor::zeroed();
    req.src4 = HtpTensor::zeroed();
    req.dst = dst;
}

/// Unary activation op (SILU/GELU) req 초기화. n_bufs=2 path.
///
/// llama.cpp `ggml-hexagon.cpp::init_unary_req` 의 `GGML_OP_UNARY` 분기와 동일.
/// `proc_activations_req` 가 op_params 를 그대로 사용하지만 ggml unary op 의
/// op_params 는 본질적으로 empty 라 zero-init OK (`htp/main.c:828`).
///
/// `src1` slot 은 미사용 (zero) — `htp/main.c:821-823` 가 `n_bufs==3` 일 때만
/// `octx.src1` 를 채운다.
pub fn init_unary_act_req(req: &mut HtpGeneralReq, op: u32, src0: HtpTensor, dst: HtpTensor) {
    debug_assert!(
        op == HTP_OP_UNARY_SILU || op == HTP_OP_UNARY_GELU,
        "init_unary_act_req: op must be UNARY_SILU or UNARY_GELU"
    );
    req.op = op;
    req.flags = 0;
    req.op_params = [0; HTP_MAX_OP_PARAMS_SLOTS];
    req.src0 = src0;
    req.src1 = HtpTensor::zeroed();
    req.src2 = HtpTensor::zeroed();
    req.src3 = HtpTensor::zeroed();
    req.src4 = HtpTensor::zeroed();
    req.dst = dst;
}

/// ROPE op params (ggml convention).
///
/// **DSP-side decode slot 매핑** (ggml.c::ggml_rope_ext + rope-ops.c::execute_op_rope_f32
/// 실측):
///   [0] n_past (unused, 0)
///   [1] n_dims (i32) — 회전할 dim (head_dim 이하)
///   [2] mode (i32) — 0=normal interleaved, 2=NeoX split, 8/40=Qwen-VL
///   [3] n_ctx (unused, 0)
///   [4] n_ctx_orig (i32) — YaRN extension 의 원래 context length
///   [5] freq_base (f32 bits) — Qwen2.5=1e6, Llama3=5e5
///   [6] freq_scale (f32 bits)
///   [7] ext_factor (f32 bits) — YaRN extrapolation factor
///   [8] attn_factor (f32 bits)
///   [9] beta_fast (f32 bits)
///   [10] beta_slow (f32 bits)
///   [11..14] sections[4] (Qwen-VL multimodal RoPE, 미사용 시 0)
#[derive(Clone, Copy, Debug)]
pub struct RopeParams {
    pub n_dims: i32,
    pub mode: i32,
    pub n_ctx_orig: i32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl RopeParams {
    /// Qwen2.5 vendor RoPE params (mode=0 normal interleaved, freq_base=1e6).
    pub fn qwen2_5() -> Self {
        Self {
            n_dims: 128,
            mode: 0,
            n_ctx_orig: 32768,
            freq_base: 1_000_000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

/// ROPE op req 초기화 (n_bufs=3 path — src2 freq_factors 미사용).
///
/// llama.cpp `ggml-hexagon.cpp::init_rope_req` 의 op_params packing 패턴 동일
/// (`memcpy(&req->op_params, &t->op_params, sizeof(t->op_params))`). ggml.c
/// `ggml_rope_ext` (line 4125-) 의 `params[15]` layout 을 그대로 따른다 — slot 0
/// 과 3 은 **legacy unused (n_past / n_ctx)** 라 반드시 0 유지.
///
/// DSP-side `proc_rope_req` 가 n_bufs=3 path 도 분기 처리 (`htp/main.c:857-908`,
/// `if (4 == n_bufs) octx.src2 = req->src2;` else 3-buf). src2 buffer 자체를
/// 보내지 않으면 freq_factors=NULL 동작.
///
/// 인자:
///   src0 = input `[head_dim, n_heads, n_tokens]` F32
///   src1 = positions `[n_tokens]` i32 (HTP_TYPE_I32)
///   dst  = output, src0 와 동일 shape
pub fn init_rope_req(
    req: &mut HtpGeneralReq,
    params: &RopeParams,
    src0: HtpTensor,
    src1: HtpTensor,
    dst: HtpTensor,
) {
    req.op = HTP_OP_ROPE;
    req.flags = 0;
    req.op_params = [0; HTP_MAX_OP_PARAMS_SLOTS];
    // slot 0 = n_past (unused, 0)
    req.op_params[1] = params.n_dims;
    req.op_params[2] = params.mode;
    // slot 3 = n_ctx (unused, 0)
    req.op_params[4] = params.n_ctx_orig;
    req.op_params[5] = params.freq_base.to_bits() as i32;
    req.op_params[6] = params.freq_scale.to_bits() as i32;
    req.op_params[7] = params.ext_factor.to_bits() as i32;
    req.op_params[8] = params.attn_factor.to_bits() as i32;
    req.op_params[9] = params.beta_fast.to_bits() as i32;
    req.op_params[10] = params.beta_slow.to_bits() as i32;
    // slot 11..14 = sections[4] (Qwen-VL multimodal RoPE), unused 시 0 유지.
    req.src0 = src0;
    req.src1 = src1;
    req.src2 = HtpTensor::zeroed(); // n_bufs=3 path: freq_factors 미사용
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
    fn init_matmul_req_packs_constants() {
        // Qwen2.5-1.5B Q-proj shape: W[N=1536, K=1536] Q4_0, x[K=1536] F32, y[N=1536] F32
        const K: u32 = 1536;
        const N: u32 = 1536;
        // Q4_0: nb[0] = sizeof(block_q4_0) = 18, nb[1] = (K/32)*18, nb[2..3] = N*nb[1]
        let row_bytes_w = (K / 32) * 18;
        let plane_bytes_w = N * row_bytes_w;
        let src0 = htp_tensor_from_shape(
            HTP_TYPE_Q4_0,
            [K, N, 1, 1],
            [18, row_bytes_w, plane_bytes_w, plane_bytes_w],
        );
        let src1 = htp_tensor_from_shape(HTP_TYPE_F32, [K, 1, 1, 1], [4, K * 4, K * 4, K * 4]);
        let dst = htp_tensor_from_shape(HTP_TYPE_F32, [N, 1, 1, 1], [4, N * 4, N * 4, N * 4]);

        let mut req = HtpGeneralReq::zeroed();
        init_matmul_req(&mut req, src0, src1, dst, false);

        assert_eq!(req.op, HTP_OP_MUL_MAT);
        assert_eq!(req.flags, 0, "skip_quantize=false → flags=0");
        assert_eq!(req.src0.type_, HTP_TYPE_Q4_0);
        assert_eq!(req.src1.type_, HTP_TYPE_F32);
        assert_eq!(req.dst.type_, HTP_TYPE_F32);
        // 미사용 src2..src4 zero
        assert_eq!(req.src2.type_, 0);
        assert_eq!(req.src2.ne, [0; HTP_MAX_DIMS]);
        assert_eq!(req.src3.ne, [0; HTP_MAX_DIMS]);
        assert_eq!(req.src4.ne, [0; HTP_MAX_DIMS]);
        // op_params 는 MUL_MAT 에서 미사용
        assert_eq!(req.op_params, [0; HTP_MAX_OP_PARAMS_SLOTS]);

        // skip_quantize=true 분기
        let mut req2 = HtpGeneralReq::zeroed();
        init_matmul_req(&mut req2, src0, src1, dst, true);
        assert_eq!(req2.flags, HTP_OPFLAGS_SKIP_QUANTIZE);
    }

    #[test]
    fn init_binary_req_mul_zeroes_unused_slots() {
        // Qwen2.5-1.5B SwiGLU element-wise mul: x[8960] F32 × y[8960] F32 → z[8960] F32
        const DIM: u32 = 8960;
        let nb = [4u32, DIM * 4, DIM * 4, DIM * 4];
        let src0 = htp_tensor_from_shape(HTP_TYPE_F32, [DIM, 1, 1, 1], nb);
        let src1 = htp_tensor_from_shape(HTP_TYPE_F32, [DIM, 1, 1, 1], nb);
        let dst = htp_tensor_from_shape(HTP_TYPE_F32, [DIM, 1, 1, 1], nb);

        let mut req = HtpGeneralReq::zeroed();
        init_binary_req(&mut req, HTP_OP_MUL, src0, src1, dst);

        assert_eq!(req.op, HTP_OP_MUL);
        assert_eq!(req.flags, 0);
        assert_eq!(req.src0.type_, HTP_TYPE_F32);
        assert_eq!(req.src1.type_, HTP_TYPE_F32);
        assert_eq!(req.dst.type_, HTP_TYPE_F32);
        assert_eq!(req.src0.ne[0], DIM);
        assert_eq!(req.src1.ne[0], DIM);
        // 미사용 src2..src4 zero
        assert_eq!(req.src2.type_, 0);
        assert_eq!(req.src2.ne, [0; HTP_MAX_DIMS]);
        assert_eq!(req.src3.ne, [0; HTP_MAX_DIMS]);
        assert_eq!(req.src4.ne, [0; HTP_MAX_DIMS]);
        // op_params 미사용
        assert_eq!(req.op_params, [0; HTP_MAX_OP_PARAMS_SLOTS]);
    }

    #[test]
    fn init_binary_req_add_uses_add_op() {
        // Qwen2.5-1.5B residual add: x[1536] F32 + y[1536] F32 → z[1536] F32
        const DIM: u32 = 1536;
        let nb = [4u32, DIM * 4, DIM * 4, DIM * 4];
        let t = htp_tensor_from_shape(HTP_TYPE_F32, [DIM, 1, 1, 1], nb);

        let mut req = HtpGeneralReq::zeroed();
        init_binary_req(&mut req, HTP_OP_ADD, t, t, t);

        assert_eq!(req.op, HTP_OP_ADD);
        assert_eq!(req.flags, 0);
        assert_eq!(req.src2.ne, [0; HTP_MAX_DIMS]);
    }

    #[test]
    fn init_unary_act_req_silu_packs_correctly() {
        // Qwen2.5-1.5B SiLU activation: x[8960] F32 → y[8960] F32
        const DIM: u32 = 8960;
        let nb = [4u32, DIM * 4, DIM * 4, DIM * 4];
        let src = htp_tensor_from_shape(HTP_TYPE_F32, [DIM, 1, 1, 1], nb);
        let dst = htp_tensor_from_shape(HTP_TYPE_F32, [DIM, 1, 1, 1], nb);

        let mut req = HtpGeneralReq::zeroed();
        init_unary_act_req(&mut req, HTP_OP_UNARY_SILU, src, dst);

        assert_eq!(req.op, HTP_OP_UNARY_SILU);
        assert_eq!(req.flags, 0);
        assert_eq!(req.src0.type_, HTP_TYPE_F32);
        assert_eq!(req.dst.type_, HTP_TYPE_F32);
        // n_bufs=2 path: src1 미사용 (zero)
        assert_eq!(req.src1.type_, 0);
        assert_eq!(req.src1.ne, [0; HTP_MAX_DIMS]);
        assert_eq!(req.src2.ne, [0; HTP_MAX_DIMS]);
        assert_eq!(req.src3.ne, [0; HTP_MAX_DIMS]);
        assert_eq!(req.src4.ne, [0; HTP_MAX_DIMS]);
        // op_params 미사용
        assert_eq!(req.op_params, [0; HTP_MAX_OP_PARAMS_SLOTS]);
    }

    #[test]
    fn init_rope_req_qwen2_5_packs_op_params_correctly() {
        // Qwen2.5-1.5B Q-rotation decode: input [head_dim=128, n_heads=12, n_tokens=1] F32
        //                                  positions [1] i32
        // ggml convention: slot 0=n_past, 1=n_dims, 2=mode, 3=n_ctx, 4=n_ctx_orig,
        //                  5=freq_base, 6=freq_scale, 7=ext_factor, 8=attn_factor,
        //                  9=beta_fast, 10=beta_slow, 11..14=sections.
        const HEAD_DIM: u32 = 128;
        const N_HEADS: u32 = 12;
        const N_TOKENS: u32 = 1;
        let nb_in = [
            4u32,
            HEAD_DIM * 4,
            HEAD_DIM * N_HEADS * 4,
            HEAD_DIM * N_HEADS * N_TOKENS * 4,
        ];
        let src0 = htp_tensor_from_shape(HTP_TYPE_F32, [HEAD_DIM, N_HEADS, N_TOKENS, 1], nb_in);
        let src1 = htp_tensor_from_shape(HTP_TYPE_I32, [N_TOKENS, 1, 1, 1], [4, 4, 4, 4]);
        let dst = htp_tensor_from_shape(HTP_TYPE_F32, [HEAD_DIM, N_HEADS, N_TOKENS, 1], nb_in);

        let params = RopeParams::qwen2_5();
        let mut req = HtpGeneralReq::zeroed();
        init_rope_req(&mut req, &params, src0, src1, dst);

        assert_eq!(req.op, HTP_OP_ROPE);
        assert_eq!(req.flags, 0);
        // slot 0 = n_past (unused)
        assert_eq!(req.op_params[0], 0, "n_past must be 0 (unused)");
        // i32 slots
        assert_eq!(req.op_params[1], 128, "n_dims");
        assert_eq!(req.op_params[2], 0, "mode=normal");
        // slot 3 = n_ctx (unused)
        assert_eq!(req.op_params[3], 0, "n_ctx must be 0 (unused)");
        assert_eq!(req.op_params[4], 32768, "n_ctx_orig");
        // f32-bitcast slots
        assert_eq!(
            req.op_params[5],
            1_000_000.0_f32.to_bits() as i32,
            "freq_base"
        );
        assert_eq!(req.op_params[6], 1.0_f32.to_bits() as i32, "freq_scale");
        assert_eq!(req.op_params[7], 0.0_f32.to_bits() as i32, "ext_factor");
        assert_eq!(req.op_params[8], 1.0_f32.to_bits() as i32, "attn_factor");
        assert_eq!(req.op_params[9], 32.0_f32.to_bits() as i32, "beta_fast");
        assert_eq!(req.op_params[10], 1.0_f32.to_bits() as i32, "beta_slow");
        // sections (slot 11..14) zero — Qwen2.5 는 multimodal RoPE 미사용
        for (idx, slot) in req.op_params[11..15].iter().enumerate() {
            assert_eq!(*slot, 0, "sections[{idx}] must be zero");
        }
        // slot 15 도 zero
        assert_eq!(req.op_params[15], 0);
        // src binding
        assert_eq!(req.src0.type_, HTP_TYPE_F32);
        assert_eq!(req.src0.ne[0], HEAD_DIM);
        assert_eq!(req.src1.type_, HTP_TYPE_I32, "positions tensor i32");
        assert_eq!(req.src1.ne[0], N_TOKENS);
        assert_eq!(req.dst.type_, HTP_TYPE_F32);
        // n_bufs=3 path — src2 미사용 (freq_factors NULL)
        assert_eq!(req.src2.type_, 0);
        assert_eq!(req.src2.ne, [0; HTP_MAX_DIMS]);
        assert_eq!(req.src3.ne, [0; HTP_MAX_DIMS]);
        assert_eq!(req.src4.ne, [0; HTP_MAX_DIMS]);
    }

    #[test]
    fn init_rope_req_f32_bit_pattern_roundtrips() {
        // f32::to_bits() as i32 → DSP-side *(float *)&op_params[i] read 시 원본 복원
        // 검증. wrong cast (예: as i32 빠뜨림 → truncation) 시 freq_base 등 wrong value.
        let src0 = htp_tensor_from_shape(HTP_TYPE_F32, [128, 12, 1, 1], [4, 512, 6144, 6144]);
        let src1 = htp_tensor_from_shape(HTP_TYPE_I32, [1, 1, 1, 1], [4, 4, 4, 4]);
        let dst = htp_tensor_from_shape(HTP_TYPE_F32, [128, 12, 1, 1], [4, 512, 6144, 6144]);

        let params = RopeParams::qwen2_5();
        let mut req = HtpGeneralReq::zeroed();
        init_rope_req(&mut req, &params, src0, src1, dst);

        // 각 f32 슬롯을 다시 decode 했을 때 원본 값과 일치 (ggml slot 5..10)
        assert_eq!(f32::from_bits(req.op_params[5] as u32), 1_000_000.0_f32);
        assert_eq!(f32::from_bits(req.op_params[6] as u32), 1.0_f32);
        assert_eq!(f32::from_bits(req.op_params[7] as u32), 0.0_f32);
        assert_eq!(f32::from_bits(req.op_params[8] as u32), 1.0_f32);
        assert_eq!(f32::from_bits(req.op_params[9] as u32), 32.0_f32);
        assert_eq!(f32::from_bits(req.op_params[10] as u32), 1.0_f32);
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
