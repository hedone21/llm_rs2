//! HTP FastRPC backend — module entry point + `Backend` trait impl.
//!
//! Q-2.2-α PoC Phase 4. 본 모듈은 5 파일 분해로 INV 1:1 매칭을 갖는다:
//!
//! | 파일 | 책임 | INV |
//! |------|------|-----|
//! | `host.rs`   | dlopen + 17 dlsym + handshake + handle/queue lifecycle | INV-HTP-FRPC-001, 004 |
//! | `idl.rs`    | host↔DSP packet schema (llama.cpp htp-msg.h 동기화)     | INV-HTP-FRPC-003 |
//! | `buffer.rs` | rpcmem allocator wrapper (RAII free, single-owner)     | INV-HTP-FRPC-002 |
//! | `error.rs`  | AEEResult → engine error mapping                       | INV-HTP-FRPC-004 |
//! | `mod.rs`    | Backend trait impl + cpu_companion 위임 라우팅          | INV-HTP-FRPC-005 |
//!
//! **PoC scope (Phase 4)**: rms_norm 1개 op 가 진짜 HVX 호출 path (RpcmemBuffer
//! backing 확인 시). 나머지 trait method 는 `cpu_companion` (CpuBackend
//! singleton) 으로 위임하거나 default 반환 유지. method 별 sprint 단계는
//! `papers/eurosys2027/_workspace/experiment/qnn_q22_dryrun_fastrpc_2026_05_26/backend_interface_matrix.md`
//! §3-3 참조.

pub mod buffer;
pub mod error;
pub mod host;
pub mod idl;
pub mod memory;
pub mod repack;

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
    DspqBufferType, HTP_MAX_DIMS, HTP_MAX_OP_PARAMS_SLOTS, HTP_MAX_PACKET_BUFFERS, HTP_OP_ADD,
    HTP_OP_FLASH_ATTN_EXT, HTP_OP_GET_ROWS, HTP_OP_MUL, HTP_OP_MUL_MAT, HTP_OP_RMS_NORM,
    HTP_OP_ROPE, HTP_OP_SOFTMAX, HTP_OP_UNARY_GELU, HTP_OP_UNARY_SILU, HTP_OPFLAGS_SKIP_QUANTIZE,
    HTP_TYPE_F16, HTP_TYPE_F32, HTP_TYPE_I32, HTP_TYPE_Q4_0, HTP_TYPE_Q8_0, HtpGeneralReq,
    HtpGeneralRsp, HtpTensor, RopeParams, SoftmaxParams, htp_tensor_from_shape, init_binary_req,
    init_get_rows_req, init_matmul_req, init_rmsnorm_req, init_rope_req, init_softmax_req,
    init_unary_act_req,
};

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};

use crate::backend::{Backend, GpuEvent, GpuScoreAccess, KiviAttentionBackend};
use crate::buffer::DType;
use crate::tensor::Tensor;

/// HTP FastRPC backend — `Backend` trait impl + cpu_companion 위임.
///
/// PoC scope: rms_norm 만 진짜 HTP 호출 path 를 갖고, 그 외 trait method 는
/// `cpu_companion` (CpuBackend Arc) 으로 위임된다 (matmul/silu_mul/softmax 등).
/// 진짜 호출 path 는 input/output Tensor 의 backing buffer 가 RpcmemBuffer
/// 일 때만 활성화되며, 그렇지 않으면 cpu_companion fallback 으로 graceful
/// degrade 한다.
///
/// Drop 시 `host` 의 Arc 가 0 으로 감소하면 [`HtpFastrpcHost`] 의 RAII
/// teardown 으로 dspqueue_close + remote_handle64_close + dlclose 가
/// 순차 호출된다 (INV-HTP-FRPC-001).
pub struct HtpFastrpcBackend {
    /// FastRPC host (dlopen + handle/queue 보유). 추후 RpcmemBuffer 가
    /// 동일 Arc 를 share.
    host: Arc<HtpFastrpcHost>,

    /// CPU companion. 모든 host-side fallback dispatch 가 본 backend 로 위임됨.
    /// `cpu_singleton()` 에서 process-wide Arc 를 share.
    cpu_companion: Arc<dyn Backend>,

    /// 현재 layer index (Phase 4 placeholder, β 단계에서 graph dispatch 시 활용).
    #[allow(dead_code)]
    layer_idx: AtomicUsize,
}

/// `htp_iface_start` 의 n_hvx default. llama.cpp `ggml-hexagon.cpp::opt_nhvx`
/// 의 기본값 0 = "use all" 와 동일. DSP-side 가 device 의 HVX unit 수에
/// 맞춰 선택. 환경 변수로 override 가능 (`HTP_FASTRPC_N_HVX`).
const HTP_FASTRPC_N_HVX_DEFAULT: u32 = 0;

// SAFETY: HtpFastrpcHost 가 Send+Sync (dlsym 결과 fn ptr 는 process-global).
// cpu_companion 은 Arc<dyn Backend> 로 자체 Send+Sync 보유. 나머지 atomic.
unsafe impl Send for HtpFastrpcBackend {}
unsafe impl Sync for HtpFastrpcBackend {}

impl HtpFastrpcBackend {
    /// 새 backend 인스턴스를 생성한다. host 가 4-step handshake 를 완료한
    /// 상태로 반환되며, best-effort `htp_iface_start` IDL invocation 을
    /// 시도한다 (실패해도 fatal 아님 — DSP-side skel 의 lazy init 에 의존).
    ///
    /// `cpu_companion` 은 process-wide `cpu_singleton()` 에서 share.
    ///
    /// 본 메서드는 `host.try_start_iface(n_hvx)` 를 명시 호출하여 DSP-side
    /// worker thread 가 dspqueue 를 attach 한 ready state 까지 보장한다. start
    /// 실패는 fatal — caller 가 명시적으로 error 를 받아 fallback 결정.
    ///
    /// `n_hvx` 는 환경 변수 `HTP_FASTRPC_N_HVX` 가 있으면 그 값, 없으면
    /// [`HTP_FASTRPC_N_HVX_DEFAULT`] (=0, "use all").
    pub fn new(session_name: &str) -> Result<Self> {
        let host = HtpFastrpcHost::new(session_name)?;
        let cpu_companion = crate::backend::cpu::cpu_singleton();

        let n_hvx = std::env::var("HTP_FASTRPC_N_HVX")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(HTP_FASTRPC_N_HVX_DEFAULT);
        host.try_start_iface(n_hvx).with_context(|| {
            format!("htp_fastrpc: HtpFastrpcBackend::new — try_start_iface(n_hvx={n_hvx}) failed")
        })?;

        Ok(Self {
            host,
            cpu_companion,
            layer_idx: AtomicUsize::new(0),
        })
    }

    /// 진단/테스트용 host 접근자.
    pub fn host(&self) -> &Arc<HtpFastrpcHost> {
        &self.host
    }

    /// rms_norm 진짜 HTP 호출 path. input/output Tensor backing 이 RpcmemBuffer
    /// 일 때만 진입한다. caller (`Backend::rms_norm`) 가 backing 확인 후 분기.
    ///
    /// PoC: in-place semantics (`x` 가 input/output). gamma weight `w` 는
    /// 별도 buffer (반드시 rpcmem 일 필요는 없으나 일관성 위해 동일 영역
    /// 권장).
    #[cfg(target_os = "android")]
    #[allow(dead_code)] // Phase 5 wire-up 대기 (RpcmemBuffer→Buffer trait 구현 + Tensor downcast).
    fn rms_norm_via_htp(
        &self,
        x_buf: &DspQueueBuffer,
        w_buf: &DspQueueBuffer,
        out_buf: &DspQueueBuffer,
        ne: [u32; HTP_MAX_DIMS],
        nb: [u32; HTP_MAX_DIMS],
        eps: f32,
    ) -> Result<()> {
        use std::os::raw::c_void;

        // 1) HtpGeneralReq 조립.
        let mut req = HtpGeneralReq::zeroed();
        let src0 = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
        let dst = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
        init_rmsnorm_req(&mut req, eps, src0, dst);
        // src1 슬롯에 gamma weight tensor 를 명시 (HTP rmsnorm op 가
        // gamma 를 별 buffer 로 받는 변종일 경우 — llama.cpp htp-msg.h 의
        // rmsnorm 사용 패턴에 따라 src1 또는 dst 슬롯 처리는 vendor-specific).
        req.src1 = htp_tensor_from_shape(HTP_TYPE_F32, [ne[0], 1, 1, 1], [4, nb[0], nb[0], nb[0]]);

        // 2) DspQueueBuffer 배열 구성: [input, weight, output]
        let mut bufs: [DspQueueBuffer; 3] = [*x_buf, *w_buf, *out_buf];

        // 3) dspqueue_write — host → DSP.
        // SAFETY: queue valid (host 가 보유 + drop 시 close 보장).
        let rc = unsafe {
            (self.host.dspqueue_write)(
                self.host.queue,
                0, // flags
                3, // num_buffers
                bufs.as_mut_ptr(),
                core::mem::size_of::<HtpGeneralReq>() as u32,
                &req as *const _ as *const u8,
                DSPQUEUE_TIMEOUT,
            )
        };
        if rc != AEE_SUCCESS {
            return Err(map_aee_err(rc));
        }

        // 4) dspqueue_read — DSP → host (blocking, 동기 response wait).
        let mut rsp = HtpGeneralRsp::zeroed();
        let mut rsp_buf_count: u32 = 0;
        let mut rsp_bufs: [DspQueueBuffer; 3] = [DspQueueBuffer::zeroed(); 3];
        let mut rsp_msg_len: u32 = 0;
        let mut rsp_flags: u32 = 0;

        // SAFETY: 같은 queue, out-pointer 들 유효.
        let rc = unsafe {
            (self.host.dspqueue_read)(
                self.host.queue,
                &mut rsp_flags as *mut u32,
                3, // max_buffers
                &mut rsp_buf_count as *mut u32,
                rsp_bufs.as_mut_ptr(),
                core::mem::size_of::<HtpGeneralRsp>() as u32,
                &mut rsp_msg_len as *mut u32,
                &mut rsp as *mut _ as *mut u8,
                DSPQUEUE_TIMEOUT,
            )
        };
        if rc != AEE_SUCCESS {
            return Err(map_aee_err(rc));
        }

        // 5) DSP-side op status 확인.
        if rsp.status != HTP_STATUS_OK {
            return Err(map_htp_status(rsp.status));
        }

        // touch unused refs to avoid clippy warnings on cross-platform stub.
        let _ = bufs.as_ptr() as *const c_void;
        Ok(())
    }

    /// S4/Y — Q4_0 weight × F32 activation matmul NPU dispatch (M≥1).
    ///
    /// caller (`matmul_transposed`) 가 (a) `htp_matmul_dispatchable` 게이트
    /// 통과, (b) a/b/out 전부 RpcmemBuffer-backed 확인 후 진입. weight `b` 는
    /// `copy_weight_from` 에서 이미 q4x4x2 layout 으로 repack 되어 있다고 가정.
    ///
    /// `m` = activation row 수 (decode GEMV=1, prefill GEMM=seq_len). M==1 이면
    /// microbench `run_htp` 의 GEMV dispatch 와 byte-identical 로 환원된다. M>1
    /// 은 ne1=M, plane stride=M*row 로 일반화 (llama.cpp HTP 의 general matmul
    /// 과 동일). Q4_0 matmul 을 prefill·decode 모두 NPU 로 보내 CPU fallback 의
    /// q4x4x2↔standard 레이아웃 비호환(garbage)을 구조적으로 제거한다.
    ///
    /// timing/prof 진단 없이 정확성 path 만 (n_bufs=3 dspqueue_write →
    /// dspqueue_read → status 확인).
    #[cfg(target_os = "android")]
    fn matmul_transposed_via_htp(
        &self,
        a_rpc: &RpcmemBuffer,
        b_rpc: &RpcmemBuffer,
        out_rpc: &RpcmemBuffer,
        n: usize,
        k: usize,
        m: usize,
        weight_dtype: DType,
    ) -> Result<()> {
        // ── Build packet ───────────────────────────────────────────────────
        //
        // ggml convention for weight `W[N, K]` (row-major): ne = (K, N, 1, 1).
        //   Q4_0: nb[0]=18(block_q4_0), nb[1]=(K/32)*18   — q4x4x2 repacked.
        //   F16 : nb[0]=2(f16),         nb[1]=K*2          — row-major bytes 직접.
        //   nb[2..3] = N * nb[1] (plane/file stride).
        let (htp_wtype, elem_bytes, row_bytes_w) = match weight_dtype {
            DType::Q4_0 => (HTP_TYPE_Q4_0, 18u32, (k as u32 / 32) * 18),
            DType::F16 => (HTP_TYPE_F16, 2u32, (k * 2) as u32),
            other => anyhow::bail!("htp via_htp: unsupported weight dtype {other:?}"),
        };
        let plane_bytes_w = (n as u32) * row_bytes_w;
        let ne_w = [k as u32, n as u32, 1, 1];
        let nb_w = [elem_bytes, row_bytes_w, plane_bytes_w, plane_bytes_w];

        // input F32 activation x[M, K] (row-major) — ne=(K, M, 1, 1).
        // M==1 이면 [4, K*4, K*4, K*4] 로 환원 (proven GEMV path).
        let plane_x = (m * k * 4) as u32;
        let ne_x = [k as u32, m as u32, 1, 1];
        let nb_x = [4u32, (k * 4) as u32, plane_x, plane_x];

        // output F32 y[M, N] (row-major) — ne=(N, M, 1, 1).
        let plane_y = (m * n * 4) as u32;
        let ne_y = [n as u32, m as u32, 1, 1];
        let nb_y = [4u32, (n * 4) as u32, plane_y, plane_y];

        let mut req = HtpGeneralReq::zeroed();
        let src0 = htp_tensor_from_shape(htp_wtype, ne_w, nb_w);
        let src1 = htp_tensor_from_shape(HTP_TYPE_F32, ne_x, nb_x);
        let dst = htp_tensor_from_shape(HTP_TYPE_F32, ne_y, nb_y);
        // skip_quantize=false: DSP-side 가 input(F32) 을 자체 처리 — Q4_0 path 는
        // dynamic Q8_0 양자화, F16 path 는 f32→fp16 변환 (`quantize_f32_f16`).
        init_matmul_req(&mut req, src0, src1, dst, false);

        // bufs ordering MUST match init_binary_req<true>:
        //   [0] weight (Constant), [1] input (CpuWriteDspRead), [2] output (DspWriteCpuRead)
        let bytes_w = b_rpc.size() as u32;
        let bytes_x = a_rpc.size() as u32;
        let bytes_y = out_rpc.size() as u32;
        let mut bufs: [DspQueueBuffer; 3] = [
            b_rpc.dsp_buf(DspqBufferType::Constant, 0, bytes_w)?,
            a_rpc.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes_x)?,
            out_rpc.dsp_buf(DspqBufferType::DspWriteCpuRead, 0, bytes_y)?,
        ];

        // dspqueue_write (host → DSP).
        // SAFETY: queue valid (host 보유 + drop 시 close), bufs/req live.
        let rc = unsafe {
            (self.host.dspqueue_write)(
                self.host.queue,
                0, // flags
                3, // num_buffers (weight + input + output)
                bufs.as_mut_ptr(),
                core::mem::size_of::<HtpGeneralReq>() as u32,
                &req as *const _ as *const u8,
                DSPQUEUE_TIMEOUT,
            )
        };
        if rc != AEE_SUCCESS {
            return Err(map_aee_err(rc).context("dspqueue_write"));
        }

        // dspqueue_read (DSP → host, blocking).
        let mut rsp = HtpGeneralRsp::zeroed();
        let mut rsp_buf_count: u32 = 0;
        let mut rsp_bufs: [DspQueueBuffer; 4] = [DspQueueBuffer::zeroed(); 4];
        let mut rsp_msg_len: u32 = 0;
        let mut rsp_flags: u32 = 0;
        // SAFETY: 같은 queue, out-pointer 들 유효.
        let rc = unsafe {
            (self.host.dspqueue_read)(
                self.host.queue,
                &mut rsp_flags as *mut u32,
                4, // max_buffers
                &mut rsp_buf_count as *mut u32,
                rsp_bufs.as_mut_ptr(),
                core::mem::size_of::<HtpGeneralRsp>() as u32,
                &mut rsp_msg_len as *mut u32,
                &mut rsp as *mut _ as *mut u8,
                DSPQUEUE_TIMEOUT,
            )
        };
        if rc != AEE_SUCCESS {
            return Err(map_aee_err(rc).context("dspqueue_read"));
        }
        // HTP_STATUS_OK = 1 (llama.cpp htp-msg.h:24). 0 은 uninitialized.
        if rsp.status != HTP_STATUS_OK {
            return Err(map_htp_status(rsp.status).context("DSP matmul status"));
        }
        let _ = (rsp_buf_count, rsp_msg_len, rsp_flags);
        Ok(())
    }
}

/// HTP NPU dispatch 가능 여부 (dtype/shape gating). buffer backing 확인은 별도.
///
/// `Some((n, k, m))` 이면 weight `b` 는 `[n, k]` Q4_0 (K%256==0), activation `a`
/// 는 F32 `[m, k]` (M=row 수: decode GEMV=1, prefill GEMM=seq_len), output 은
/// F32 `[m, n]`. 비-Q4_0 weight / K misalign / shape 불일치는 `None` (cpu
/// fallback). **M==1 뿐 아니라 M>1(prefill)도 dispatch** — Q4_0 matmul 을 전부
/// NPU 로 보내 CPU fallback 의 q4x4x2↔standard 비호환을 제거 (Y 설계).
///
/// host(non-android) 에서도 컴파일/테스트 가능 (cfg 게이트 없음). 단 non-android
/// 에서는 유일한 비-test 호출처(`matmul_transposed` 의 android 블록)가 cfg-out
/// 되어 dead — `htp_dispatch_log_once` 와 동일하게 allow.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
fn htp_matmul_dispatchable(a: &Tensor, b: &Tensor, out: &Tensor) -> Option<(usize, usize, usize)> {
    let bd = b.shape().dims();
    if bd.len() != 2 {
        return None;
    }
    let (n, k) = (bd[0], bd[1]);
    if b.dtype() != DType::Q4_0 || a.dtype() != DType::F32 || out.dtype() != DType::F32 {
        return None;
    }
    if !k.is_multiple_of(256) {
        return None;
    }
    // activation 은 정확히 [M, K] = M*K 원소 (M = row 수). K=0(division 보호) 또는
    // K 비배수면 reject. `||` 단락평가로 k==0 시 is_multiple_of 미호출.
    if k == 0 || !a.numel().is_multiple_of(k) {
        return None;
    }
    let m = a.numel() / k;
    if m == 0 || out.numel() != m * n {
        return None;
    }
    Some((n, k, m))
}

/// F16 weight 의 HTP NPU dispatch 가능 여부 (A 실험 — F16 row-major matmul).
///
/// `Some((n, k, m))` 이면 weight `b` 는 `[n, k]` F16 (row-major, K%64==0 = HVX
/// f16 벡터 64 elem 정렬), activation `a` 는 F32 `[m, k]`, output 은 F32 `[m, n]`.
/// Q4_0 와 달리 F16 rpcmem 은 **표준 layout** 이라 dispatch 불가 시 cpu fallback
/// 이 안전 (garbage 아님) — 호출처가 bail 대신 cpu 위임.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
fn htp_matmul_dispatchable_f16(
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
) -> Option<(usize, usize, usize)> {
    let bd = b.shape().dims();
    if bd.len() != 2 {
        return None;
    }
    let (n, k) = (bd[0], bd[1]);
    if b.dtype() != DType::F16 || a.dtype() != DType::F32 || out.dtype() != DType::F32 {
        return None;
    }
    // HVX f16 dot 는 128-byte(=64 f16) 벡터 단위. K 미정렬은 reject (cpu fallback).
    if k == 0 || !k.is_multiple_of(64) {
        return None;
    }
    if !a.numel().is_multiple_of(k) {
        return None;
    }
    let m = a.numel() / k;
    if m == 0 || out.numel() != m * n {
        return None;
    }
    Some((n, k, m))
}

/// 첫 1회 dispatch / fallback 분기를 eprintln 으로 노출 ("no silent caps").
/// NPU 가 실제 도는지 / silent cpu fallback 인지 device 에서 확인용. dispatch
/// 와 fallback 각각 `Once` 로 첫 발생만 출력.
///
/// host(non-android) 빌드에서는 `matmul_transposed` 의 android 블록이 cfg-out
/// 되어 호출처가 없으므로 dead_code allow.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
fn htp_dispatch_log_once(dispatched: bool, reason: &str) {
    use std::sync::Once;
    static DISPATCH_ONCE: Once = Once::new();
    static FALLBACK_ONCE: Once = Once::new();
    if dispatched {
        DISPATCH_ONCE.call_once(|| {
            eprintln!("[htp] matmul_transposed: NPU dispatch 활성 ({reason}, M≥1)");
        });
    } else {
        FALLBACK_ONCE.call_once(|| {
            eprintln!("[htp] matmul_transposed: cpu fallback — {reason}");
        });
    }
}

/// F16 weight 를 rpcmem(DMA-BUF heap)으로 올린 누적 바이트 — A 실험의 OOM ceiling
/// 진단용. `copy_weight_from` 의 F16 arm 이 alloc 마다 누적/로깅하므로, alloc 실패
/// 시 직전 누적값이 rpcmem 상한의 하한 추정치가 된다.
static F16_RPCMEM_TOTAL: AtomicUsize = AtomicUsize::new(0);

impl Backend for HtpFastrpcBackend {
    // ── Lifecycle / identity ──────────────────────────────────────────

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "htp_fastrpc"
    }

    fn device(&self) -> &str {
        "Hexagon HTP V79 (FastRPC)"
    }

    fn is_gpu(&self) -> bool {
        // NPU 는 별 카테고리. is_gpu()/is_discrete_gpu() 모두 false 유지.
        false
    }

    // ── Math / compute (cpu_companion 위임) ───────────────────────────

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        self.cpu_companion.matmul(a, b, out)
    }

    /// S4/Y — weight matmul (`a @ bᵀ → out`). b = Q4_0 weight (q4x4x2 repacked),
    /// a = F32 activation `[m, k]` (decode GEMV=1 row, prefill GEMM=seq_len rows).
    /// a/b/out 전부 RpcmemBuffer-backed 면 prefill·decode 모두 NPU dispatch. Q4_0
    /// 을 전부 NPU 로 보내 CPU fallback 의 q4x4x2↔standard 비호환을 제거한다.
    ///
    /// **silent garbage 차단 (pivot = weight 가 q4x4x2 인가)**: weight `b` 의
    /// buffer 가 RpcmemBuffer 면 그 weight 는 q4x4x2 repacked (copy_weight_from 이
    /// Q4_0 2D K%256==0 만 rpcmem+repack). CPU 는 q4x4x2 를 standard block_q4_0 로
    /// 읽어 garbage 이므로, q4x4x2 weight 는 **NPU dispatch (act/out 도 rpcmem +
    /// dtype/shape 게이트 통과) 아니면 loud error** — cpu fallback 절대 금지. weight
    /// 가 비-rpcmem(standard)이면 cpu fallback 안전. 이 구조로 silent cpu garbage
    /// 경로를 호출 그래프와 무관하게 봉쇄한다.
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            // weight 가 rpcmem 이면 NPU 대상 (copy_weight_from 이 Q4_0 q4x4x2 / F16
            // row-major 만 rpcmem). dtype 으로 분기 — pivot 은 "rpcmem layout 이
            // 비표준(Q4_0 q4x4x2)인가 표준(F16)인가".
            if let Some(b_rpc) = b.buffer().as_any().downcast_ref::<RpcmemBuffer>() {
                let a_rpc = a.buffer().as_any().downcast_ref::<RpcmemBuffer>();
                let out_rpc = out.buffer().as_any().downcast_ref::<RpcmemBuffer>();
                match b.dtype() {
                    // Q4_0 q4x4x2 = 비표준 layout → NPU dispatch 아니면 cpu 가
                    // q4x4x2 를 standard block_q4_0 로 읽어 garbage → loud bail.
                    DType::Q4_0 => {
                        if let (Some((n, k, m)), Some(a_rpc), Some(out_rpc)) =
                            (htp_matmul_dispatchable(a, b, out), a_rpc, out_rpc)
                        {
                            htp_dispatch_log_once(true, "Q4_0 q4x4x2");
                            return self.matmul_transposed_via_htp(
                                a_rpc,
                                b_rpc,
                                out_rpc,
                                n,
                                k,
                                m,
                                DType::Q4_0,
                            );
                        }
                        anyhow::bail!(
                            "htp: q4x4x2 Q4_0 weight matmul 은 NPU dispatch 필요 (act/out rpcmem + \
                             dtype/shape 게이트); act_rpcmem={}, out_rpcmem={}. cpu fallback 은 \
                             q4x4x2↔standard 비호환으로 garbage 라 차단",
                            a_rpc.is_some(),
                            out_rpc.is_some()
                        );
                    }
                    // F16 = 표준 row-major layout → dispatch 가능하면 NPU, 아니면
                    // cpu fallback 안전 (rpcmem F16 bytes = standard, garbage 아님).
                    DType::F16 => {
                        if let (Some((n, k, m)), Some(a_rpc), Some(out_rpc)) =
                            (htp_matmul_dispatchable_f16(a, b, out), a_rpc, out_rpc)
                        {
                            htp_dispatch_log_once(true, "F16 row-major");
                            return self.matmul_transposed_via_htp(
                                a_rpc,
                                b_rpc,
                                out_rpc,
                                n,
                                k,
                                m,
                                DType::F16,
                            );
                        }
                        htp_dispatch_log_once(
                            false,
                            "F16 weight dispatch 조건 미충족 → cpu fallback (safe, standard F16)",
                        );
                    }
                    // copy_weight_from 이 Q4_0/F16 만 rpcmem 화하므로 도달 불가.
                    // 방어적으로 cpu fallback (표준 layout 가정).
                    _ => {
                        htp_dispatch_log_once(
                            false,
                            "rpcmem weight 비-Q4_0/F16 → cpu fallback (방어)",
                        );
                    }
                }
            } else {
                // weight 가 rpcmem 아님 (standard, copy_weight_from 의 cpu_companion
                // arm) → cpu fallback 안전.
                htp_dispatch_log_once(false, "weight 가 rpcmem 아님 → cpu fallback (standard)");
            }
        }
        self.cpu_companion.matmul_transposed(a, b, out)
    }

    fn matmul_slice(
        &self,
        a: &Tensor,
        b: &Tensor,
        rows: usize,
        cols: usize,
        out: &mut Tensor,
    ) -> Result<()> {
        self.cpu_companion.matmul_slice(a, b, rows, cols, out)
    }

    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        self.cpu_companion.add_assign(a, b)
    }

    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()> {
        self.cpu_companion.scale(x, v)
    }

    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        self.cpu_companion.silu_mul(a, b)
    }

    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        self.cpu_companion.softmax(x)
    }

    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        self.cpu_companion.rope_inplace(x, start_pos, theta)
    }

    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        self.cpu_companion.cast(src, dst)
    }

    // ── rms_norm: 진짜 HTP 호출 path (RpcmemBuffer backing 시) ─────────

    /// In-place RMS norm. Tensor backing 이 RpcmemBuffer 면 진짜 HVX 호출,
    /// 그렇지 않으면 cpu_companion 위임.
    ///
    /// PoC: backing 확인은 `Buffer::as_any().downcast_ref::<RpcmemBuffer>()`
    /// 가 stable Rust 에서 직접 적용되지 않으므로 (RpcmemBuffer 자체가
    /// Buffer trait 미구현, Phase 5 wrap 작업 대기), 본 sprint 에서는
    /// **항상 cpu_companion 위임** 으로 종결한다. 진짜 HTP 호출 path
    /// (`rms_norm_via_htp`) 는 inline 코드로 빌드만 시키고 unused.
    ///
    /// Phase 5 진입 시: (a) `RpcmemBuffer` 를 `Buffer` trait 구현, (b)
    /// Tensor::buffer().as_any() 로 downcast, (c) 매치 시 dsp_buf() 추출 후
    /// `rms_norm_via_htp` 호출, (d) 미매치 시 cpu_companion fallback.
    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32, add_unit: bool) -> Result<()> {
        // PoC scope: RpcmemBuffer backing wrap 미완성 → cpu_companion 위임.
        // `rms_norm_via_htp` 는 Phase 5 wire-up 대기.
        self.cpu_companion.rms_norm(x, w, eps, add_unit)
    }

    // ── Memory ops (cpu_companion 위임) ───────────────────────────────

    fn copy_from(&self, t: &Tensor) -> Result<Tensor> {
        // PoC: weight upload path 는 cpu_companion 의 copy_from 사용
        // (SharedBuffer 기반). 진짜 rpcmem alloc + memcpy 는 β 단계 작업.
        self.cpu_companion.copy_from(t)
    }

    /// S5 — NPU dispatch 대상 weight (Q4_0 2D, K%256==0) 만 rpcmem (DMA-BUF
    /// heap) 으로 할당하면서 q4x4x2 layout 으로 repack (DSP-side
    /// `vec_dot_q4x4x2_q8x4x2_*` 가 expect). **그 외 dtype (F32/F16/Q8_0/Q6_K
    /// 등) 은 전부 cpu_companion** — `matmul_transposed` 게이트가 Q4_0 만
    /// dispatch 하므로 비-Q4_0 weight 는 어차피 cpu matmul. rpcmem 상주는 순수
    /// 낭비이자 system-heap OOM 압박이라 회피 (예: F16 lm_head ~445MB).
    ///
    /// 본 override 가 없으면 weight 가 SharedBuffer 에 상주 → `matmul_transposed`
    /// 의 RpcmemBuffer downcast 가 영원히 실패 → silent cpu fallback (NPU 미경유).
    ///
    /// host(non-android) 에서 `RpcmemBuffer::alloc` 은 runtime Err 이나 컴파일은
    /// OK — 이 경로는 android 에서만 reachable (host 는 `HtpFastrpcBackend::new`
    /// 자체가 Err 라 backend 가 생성되지 않음).
    fn copy_weight_from(&self, t: &Tensor) -> Result<Tensor> {
        let dims = t.shape().dims();
        match t.dtype() {
            // Q4_0 2D weight (K%256==0): rpcmem alloc + q4x4x2 repack (NPU 대상).
            DType::Q4_0 if dims.len() == 2 && dims[1].is_multiple_of(256) => {
                let (n, k) = (dims[0], dims[1]);
                let blocks = t.as_slice::<crate::quant::BlockQ4_0>();
                let repacked = repack::repack_q4_0_to_q4x4x2_matrix(blocks, n, k);
                let mut buf = RpcmemBuffer::alloc(self.host.clone(), repacked.len(), DType::Q4_0)?;
                // SAFETY: buf.as_mut_ptr() 은 직전 alloc 으로 valid, repacked.len()
                // byte allocated (alloc size 는 page-align ≥ repacked.len()).
                // 두 영역 non-overlapping (별도 alloc).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        repacked.as_ptr(),
                        buf.as_mut_ptr(),
                        repacked.len(),
                    );
                }
                Ok(Tensor::new(
                    t.shape().clone(),
                    Arc::new(buf),
                    self.cpu_companion.clone(),
                ))
            }
            // F16 2D weight (K%64==0): rpcmem alloc + row-major 복사 (repack 불필요
            // — DSP `vec_dot_f16_*` 가 row-major bytes 직접 사용). ★ A 실험: 전 F16
            // weight 를 rpcmem 상주시켜 OOM ceiling 측정. Q4_0(~703MB)의 16/4.5≈3.5×
            // (~2.5GB) 라 rpcmem(DMA-BUF) 한계 초과 가능 — 실패 시 누적 로그로 ceiling 캡처.
            DType::F16 if dims.len() == 2 && dims[1].is_multiple_of(64) => {
                let bytes = t.numel() * 2;
                let total = F16_RPCMEM_TOTAL.fetch_add(bytes, Ordering::Relaxed) + bytes;
                let mut buf = RpcmemBuffer::alloc(self.host.clone(), bytes, DType::F16)
                    .with_context(|| {
                        format!(
                            "htp: F16 weight rpcmem alloc 실패 {} B (직전까지 누적 {} MB) — \
                             rpcmem(DMA-BUF) ceiling 도달 추정",
                            bytes,
                            (total - bytes) / (1024 * 1024)
                        )
                    })?;
                let f16_slice = t.as_slice::<half::f16>();
                // SAFETY: f16 = 2-byte POD, f16_slice.len()*2 == bytes == buf alloc.
                let src_bytes =
                    unsafe { std::slice::from_raw_parts(f16_slice.as_ptr() as *const u8, bytes) };
                // SAFETY: buf 직전 alloc 으로 valid, src/dst non-overlapping, len 동일.
                unsafe {
                    std::ptr::copy_nonoverlapping(src_bytes.as_ptr(), buf.as_mut_ptr(), bytes);
                }
                eprintln!(
                    "[htp] F16 weight → rpcmem: [{},{}] {} B (누적 {} MB)",
                    dims[0],
                    dims[1],
                    bytes,
                    total / (1024 * 1024)
                );
                Ok(Tensor::new(
                    t.shape().clone(),
                    Arc::new(buf),
                    self.cpu_companion.clone(),
                ))
            }
            // 그 외 (F32/Q8_0/Q6_K/… , 비정렬·비2D Q4_0/F16): cpu_companion
            // (SharedBuffer). NPU dispatch 대상이 아니므로 rpcmem 상주 불필요.
            _ => self.cpu_companion.copy_weight_from(t),
        }
    }

    // ── attention/gather/flash 등 default-bodied trait method 는 trait
    //    body 가 CPU fallback 을 이미 포함하므로 override 안 함.
    //    명시 위임이 필요한 것은 cpu_companion 으로 호출. 단, default
    //    body 가 `self.copy_into / self.add_assign / self.rms_norm_oop`
    //    같이 self 를 재귀 호출하면 본 backend 의 cpu_companion 위임
    //    chain 으로 자동 routing 된다.

    // ── HTP 가 자체 처리 가능한 method 들의 위임 (β 단계 진짜 호출 후보) ──

    fn gelu_tanh_mul(&self, gate: &mut Tensor, up: &Tensor) -> Result<()> {
        self.cpu_companion.gelu_tanh_mul(gate, up)
    }

    fn fused_norm_merge(
        &self,
        prior_residual: &Tensor,
        gpu_partial: &Tensor,
        cpu_staging: &Tensor,
        norm_weight: &Tensor,
        out: &mut Tensor,
        residual_out: &mut Tensor,
        eps: f32,
        add_unit: bool,
    ) -> Result<()> {
        self.cpu_companion.fused_norm_merge(
            prior_residual,
            gpu_partial,
            cpu_staging,
            norm_weight,
            out,
            residual_out,
            eps,
            add_unit,
        )
    }

    fn kv_scatter_f32_to_f32_batch(
        &self,
        k_src: &Tensor,
        v_src: &Tensor,
        k_dst: &mut Tensor,
        v_dst: &mut Tensor,
        n_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        write_pos_start: usize,
        seq_len: usize,
    ) -> Result<()> {
        self.cpu_companion.kv_scatter_f32_to_f32_batch(
            k_src,
            v_src,
            k_dst,
            v_dst,
            n_kv_heads,
            head_dim,
            capacity,
            write_pos_start,
            seq_len,
        )
    }

    fn buffer_shift(
        &self,
        tensor: &mut Tensor,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) -> Result<()> {
        self.cpu_companion
            .buffer_shift(tensor, src_offset, dst_offset, count)
    }

    fn enqueue_read_buffer_async(&self, t: &Tensor, dst: &mut [u8]) -> Result<GpuEvent> {
        // PoC: sync fallback. supports_async_transfer() = false 라 caller 가
        // dispatcher path 진입을 회피한다.
        self.cpu_companion.enqueue_read_buffer_async(t, dst)
    }

    fn enqueue_write_async(&self, src: &Tensor) -> Result<(Tensor, GpuEvent)> {
        self.cpu_companion.enqueue_write_async(src)
    }

    fn enqueue_write_into_async(
        &self,
        dst: &Tensor,
        src: *const u8,
        len: usize,
    ) -> Result<GpuEvent> {
        self.cpu_companion.enqueue_write_into_async(dst, src, len)
    }

    // ── Synchronization ───────────────────────────────────────────────

    fn synchronize(&self) -> Result<()> {
        // PoC: 실 op dispatch path 가 dspqueue_read 까지 blocking 으로 진행
        // 되므로 추가 fence 가 필요 없다. 후속 sprint 에서 async path 도입
        // 시 본 method 에서 명시 fence (dspqueue_read flush 또는
        // remote_handle64_invoke 동기 호출) 를 트리거해야 한다.
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        // PoC: 동기 dispatch 라 별도 flush 필요 없음.
        Ok(())
    }

    // ── cpu_companion (B-5b Phase 2 Stage 1 contract) ─────────────────

    fn cpu_companion(&self) -> &dyn Backend {
        self.cpu_companion.as_ref()
    }

    // ── KIVI / score / extension hooks: HTP 미지원 ────────────────────
    //
    // 아래 method 는 default 본문이 OpenCL backend 의 hot-path downcast 회피
    // 용도로 도입된 것 (trait body 에서 None / no-op 으로 정의됨). HTP 는
    // 본 sprint scope 밖이므로 trait default 그대로 사용.
    //
    //   as_kivi_attention(), gpu_score_acc(), gpu_score_acc_mut(),
    //   profile_events_enabled(), set_op_label(), clear_op_label(),
    //   supports_layer_graph(), execute_layer_graph(),
    //   flash_attention_prefill() — default false/None/Err 유지.

    // ── Identity escape hatch (cold-path lookup) ──────────────────────

    fn get_extension(&self, _name: &str) -> Option<&dyn std::any::Any> {
        // PoC: HTP backend 는 cold-path extension 키 아직 노출 안 함.
        // β 단계에서 `EXT_HTP_FASTRPC` 키 + handle 노출 가능.
        None
    }

    // ── unused KIVI sub-trait helper to silence compiler ──────────────
    //
    // `KiviAttentionBackend` / `GpuScoreAccess` 는 별 trait 라 본 trait impl
    // 안에서 직접 method 정의 불필요. 사용 의도가 없음을 명시 위해
    // `_use_traits` PhantomData 패턴 대신 trait body 의 default `None`
    // 반환 유지.
}

// Compile-time use to keep imports happy (no functional effect).
#[allow(dead_code)]
const _: () = {
    let _ = |b: &HtpFastrpcBackend| {
        let _: &dyn KiviAttentionBackend;
        let _: &dyn GpuScoreAccess;
        let _: DType = DType::F32;
        let _: &HtpFastrpcBackend = b;
    };
};

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

    /// CpuBackend singleton 이 cpu_companion 으로 정상 연결됨을 컴파일 타임에
    /// 검증. 본 test 는 non-android 빌드에서도 fail 되어선 안 됨 — 단순히
    /// 타입 시스템 sanity.
    #[test]
    fn cpu_companion_type_compiles() {
        fn requires_backend<T: Backend + ?Sized>(_: &T) {}
        let cpu = crate::backend::cpu::cpu_singleton();
        requires_backend(cpu.as_ref());
        // singleton 이 valid name 을 반환 (non-android 도 동작).
        assert!(!cpu.name().is_empty());
    }

    // ── S4 guard predicate (htp_matmul_dispatchable) ──────────────────────
    //
    // host-컴파일 가능 — RpcmemBuffer backing 확인은 별도, dtype/shape gating
    // 만 검증. SharedBuffer 로 Tensor 를 합성해 dtype/shape 조합 케이스 커버.

    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;

    fn mk_tensor(dims: Vec<usize>, dtype: DType, byte_len: usize) -> Tensor {
        let buf = SharedBuffer::new(byte_len, dtype);
        let cpu = crate::backend::cpu::cpu_singleton();
        Tensor::new(Shape::new(dims), Arc::new(buf), cpu)
    }

    #[test]
    fn dispatchable_q4_0_m1_k256_some() {
        // weight [N=4, K=256] Q4_0, act F32 K=256 (M==1 decode), out F32 N=4.
        let k = 256;
        let n = 4;
        let blocks_per_row = k / 32; // 8
        let w_bytes = n * blocks_per_row * 18;
        let b = mk_tensor(vec![n, k], DType::Q4_0, w_bytes);
        let a = mk_tensor(vec![1, k], DType::F32, k * 4);
        let out = mk_tensor(vec![1, n], DType::F32, n * 4);
        assert_eq!(htp_matmul_dispatchable(&a, &b, &out), Some((n, k, 1)));
    }

    #[test]
    fn dispatchable_f16_weight_none() {
        // F16 weight → None (Q4_0 만 dispatch).
        let k = 256;
        let n = 4;
        let b = mk_tensor(vec![n, k], DType::F16, n * k * 2);
        let a = mk_tensor(vec![1, k], DType::F32, k * 4);
        let out = mk_tensor(vec![1, n], DType::F32, n * 4);
        assert_eq!(htp_matmul_dispatchable(&a, &b, &out), None);
    }

    #[test]
    fn dispatchable_prefill_m2_some() {
        // M==2 (prefill GEMM) → Y 설계에선 dispatch 대상 → Some((n, k, 2)).
        let k = 256;
        let n = 4;
        let blocks_per_row = k / 32;
        let w_bytes = n * blocks_per_row * 18;
        let b = mk_tensor(vec![n, k], DType::Q4_0, w_bytes);
        // act [2, K] → numel = 2*K, out [2, N] → numel = 2*N.
        let a = mk_tensor(vec![2, k], DType::F32, 2 * k * 4);
        let out = mk_tensor(vec![2, n], DType::F32, 2 * n * 4);
        assert_eq!(htp_matmul_dispatchable(&a, &b, &out), Some((n, k, 2)));
    }

    #[test]
    fn dispatchable_shape_mismatch_none() {
        // act numel 이 K 배수가 아니면 (M 비정수) → None.
        let k = 256;
        let n = 4;
        let w_bytes = n * (k / 32) * 18;
        let b = mk_tensor(vec![n, k], DType::Q4_0, w_bytes);
        let a = mk_tensor(vec![1, k + 1], DType::F32, (k + 1) * 4); // numel=257, 256 비배수
        let out = mk_tensor(vec![1, n], DType::F32, n * 4);
        assert_eq!(htp_matmul_dispatchable(&a, &b, &out), None);
        // out numel 이 m*n 과 불일치해도 None.
        let a2 = mk_tensor(vec![2, k], DType::F32, 2 * k * 4); // m=2
        let out2 = mk_tensor(vec![1, n], DType::F32, n * 4); // numel=n != 2*n
        assert_eq!(htp_matmul_dispatchable(&a2, &b, &out2), None);
    }

    #[test]
    fn dispatchable_k_misaligned_none() {
        // K=255 (not 256 multiple) → None. (Q4_0 block=32 정렬 무시하고 게이트만 확인)
        let k = 255;
        let n = 4;
        let b = mk_tensor(vec![n, k], DType::Q4_0, n * 8 * 18);
        let a = mk_tensor(vec![1, k], DType::F32, k * 4);
        let out = mk_tensor(vec![1, n], DType::F32, n * 4);
        assert_eq!(htp_matmul_dispatchable(&a, &b, &out), None);
    }

    #[test]
    fn dispatchable_b_not_2d_none() {
        // weight 가 1D → None (bd.len() != 2).
        let b = mk_tensor(vec![256], DType::Q4_0, 8 * 18);
        let a = mk_tensor(vec![1, 256], DType::F32, 256 * 4);
        let out = mk_tensor(vec![1, 1], DType::F32, 4);
        assert_eq!(htp_matmul_dispatchable(&a, &b, &out), None);
    }

    // ── A 실험: F16 dispatchable (htp_matmul_dispatchable_f16) ──────────────

    #[test]
    fn dispatchable_f16_k64_some() {
        // F16 weight, K%64==0, M=1 → Some((n, k, 1)).
        let k = 1536; // 64 배수
        let n = 2048;
        let b = mk_tensor(vec![n, k], DType::F16, n * k * 2);
        let a = mk_tensor(vec![1, k], DType::F32, k * 4);
        let out = mk_tensor(vec![1, n], DType::F32, n * 4);
        assert_eq!(htp_matmul_dispatchable_f16(&a, &b, &out), Some((n, k, 1)));
    }

    #[test]
    fn dispatchable_f16_prefill_m3_some() {
        // M==3 (prefill GEMM) → Some((n, k, 3)).
        let k = 1536;
        let n = 2048;
        let b = mk_tensor(vec![n, k], DType::F16, n * k * 2);
        let a = mk_tensor(vec![3, k], DType::F32, 3 * k * 4);
        let out = mk_tensor(vec![3, n], DType::F32, 3 * n * 4);
        assert_eq!(htp_matmul_dispatchable_f16(&a, &b, &out), Some((n, k, 3)));
    }

    #[test]
    fn dispatchable_f16_q4_weight_none() {
        // Q4_0 weight 는 F16 helper 에서 None (dtype gate).
        let k = 1536;
        let n = 2048;
        let b = mk_tensor(vec![n, k], DType::Q4_0, n * (k / 32) * 18);
        let a = mk_tensor(vec![1, k], DType::F32, k * 4);
        let out = mk_tensor(vec![1, n], DType::F32, n * 4);
        assert_eq!(htp_matmul_dispatchable_f16(&a, &b, &out), None);
    }

    #[test]
    fn dispatchable_f16_k_misaligned_none() {
        // K=1500 (64 비배수) → None (HVX f16 벡터 정렬 실패 → cpu fallback).
        let k = 1500;
        let n = 16;
        let b = mk_tensor(vec![n, k], DType::F16, n * k * 2);
        let a = mk_tensor(vec![1, k], DType::F32, k * 4);
        let out = mk_tensor(vec![1, n], DType::F32, n * 4);
        assert_eq!(htp_matmul_dispatchable_f16(&a, &b, &out), None);
    }
}
