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

    /// dspqueue 에 enqueue 됐으나 아직 drain(read) 하지 않은 in-flight DSP matmul
    /// op 수. async batch fusion 의 핵심: `enqueue_matmul_via_htp` 가 +1,
    /// `drain_pending` 가 0 까지 read 하여, 여러 op 를 묶어 한 번에 drain 함으로써
    /// ~100µs FastRPC dispatch floor 를 op 마다→batch 당 1회로 amortize 한다.
    /// forward pass 는 토큰당 단일 스레드 순차 실행이라 본 카운터는 매 matmul
    /// 경계에서 0 으로 복귀한다 (단일-op 경로) — batch 경로(FFN gate/up)만 일시 >1.
    /// host(non-android) 빌드에서는 android-only 메서드에서만 read 되므로 dead.
    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    op_pending: AtomicUsize,

    /// RoPE positions 용 lazily-allocated i32 rpcmem scratch. decode 는 토큰당
    /// 단일 position 이라 작은 버퍼를 1회 alloc 후 매 토큰 재기록(start_pos)한다.
    /// `rope_inplace` NPU override 에서만 read/write — env OFF 면 영원히 None.
    #[cfg(target_os = "android")]
    rope_pos_scratch: std::sync::Mutex<Option<RpcmemBuffer>>,
}

/// `htp_iface_start` 의 n_hvx default. llama.cpp `ggml-hexagon.cpp::opt_nhvx`
/// 의 기본값 0 = "use all" 와 동일. DSP-side 가 device 의 HVX unit 수에
/// 맞춰 선택. 환경 변수로 override 가능 (`HTP_FASTRPC_N_HVX`).
const HTP_FASTRPC_N_HVX_DEFAULT: u32 = 0;

/// v79 element-wise(unary/binary) op 의 dispatch 당 처리 element 상한(아래 안전치).
/// DSP 가 1 VTCM 타일(~12032 f32)만 처리하고 전체 텐서를 loop 하지 않으므로
/// flat pointwise helper 는 이 크기 이하로 분할 dispatch 한다(device 실측 근거).
#[cfg(target_os = "android")]
const HTP_OP_CHUNK: usize = 8192;

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
            op_pending: AtomicUsize::new(0),
            #[cfg(target_os = "android")]
            rope_pos_scratch: std::sync::Mutex::new(None),
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

    /// S4/Y — Q4_0/F16 weight × F32 activation matmul 을 dspqueue 에 **enqueue
    /// (write only)** 한다. drain(read) 은 [`drain_pending`] 가 담당 — 여러 op 를
    /// enqueue 후 한 번에 drain 하여 ~100µs FastRPC dispatch floor 를 amortize 하는
    /// async batch fusion 의 write half. **호출처는 반드시 후속 `drain_pending`
    /// (또는 enqueue+drain 을 묶은 `matmul_transposed_via_htp` / `flush` /
    /// `synchronize`)으로 in-flight op 를 회수해야 한다** — drain 없이 다음 토큰으로
    /// 넘어가면 출력 buffer 가 미완성(garbage)이다.
    ///
    /// caller 가 (a) dispatch 게이트(`htp_matmul_dispatchable[_f16]`) 통과, (b)
    /// a/b/out 전부 RpcmemBuffer-backed 확인 후 진입. weight `b` 는 Q4_0 이면
    /// `copy_weight_from` 에서 q4x4x2 repack 됨, F16 이면 row-major bytes.
    ///
    /// `m` = activation row 수 (decode GEMV=1, prefill GEMM=seq_len). M==1 이면
    /// microbench `run_htp` 의 GEMV dispatch 와 byte-identical 로 환원된다. M>1
    /// 은 ne1=M, plane stride=M*row 로 일반화. Q4_0 matmul 을 prefill·decode 모두
    /// NPU 로 보내 CPU fallback 의 q4x4x2↔standard 비호환(garbage)을 구조적 제거.
    #[cfg(target_os = "android")]
    fn enqueue_matmul_via_htp(
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

        // enqueue only — DSP 가 비동기 처리, 응답은 drain_pending 의 dspqueue_read 가
        // FIFO 순서로 회수. a/b/out_rpc rpcmem 데이터는 drain 까지 alive 필요 (호출처
        // workspace/weight Tensor 소유권 보장).
        self.enqueue_packet(&req, &mut bufs)
    }

    /// 임의의 op packet 을 dspqueue 에 **enqueue (write only)** 한다 — packet
    /// build(req)·buffer 배열(bufs) 은 op-specific 호출처가 준비하고, 본 helper 는
    /// `dspqueue_write` + `op_pending.fetch_add(1)` 만 공유 처리한다. matmul/silu/
    /// binary/rmsnorm/rope 의 모든 enqueue 가 본 단일 write 경로를 거치므로 step1
    /// async batch fusion (enqueue×N → drain×1) 불변이 op 종류와 무관하게 유지된다.
    ///
    /// `bufs.len()` = packet 의 num_buffers (n_bufs). drain 은 [`drain_pending`] 가
    /// op_pending 0 까지 회수. rpcmem 데이터 alive 는 호출처 소유권(workspace/weight
    /// Tensor) 책임 — 본 helper 반환 후 bufs/req stack-local drop OK (dspqueue_write
    /// 가 FIFO 로 복사 완료).
    #[cfg(target_os = "android")]
    fn enqueue_packet(&self, req: &HtpGeneralReq, bufs: &mut [DspQueueBuffer]) -> Result<()> {
        // SAFETY: queue valid (host 가 보유 + drop 시 close 보장). bufs/req 는 이
        // 호출 동안 live (caller stack). dspqueue_write 가 packet 을 FIFO 로 복사.
        let rc = unsafe {
            (self.host.dspqueue_write)(
                self.host.queue,
                0, // flags
                bufs.len() as u32,
                bufs.as_mut_ptr(),
                core::mem::size_of::<HtpGeneralReq>() as u32,
                req as *const _ as *const u8,
                DSPQUEUE_TIMEOUT,
            )
        };
        if rc != AEE_SUCCESS {
            return Err(map_aee_err(rc).context("dspqueue_write"));
        }
        // in-flight op +1 — drain_pending 가 동수 dspqueue_read 로 회수.
        self.op_pending.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// dspqueue 에 enqueue 된 in-flight matmul op 들을 전부 drain (blocking read).
    /// `op_pending` 이 0 이 될 때까지 `dspqueue_read` 를 반복 — DSP 가 FIFO 순서로
    /// 응답을 발행하므로 write i 응답이 read i 에 대응한다 (PoC `htp_batch_dispatch`
    /// 에서 device 검증, max_abs_err 보존). 각 응답의 `status` 를 확인.
    ///
    /// async batch fusion 의 flush 지점: `enqueue_matmul_via_htp` ×N → `drain_pending`
    /// ×1 로 dispatch floor 를 op 마다→batch 당 1회로 amortize. read 실패 시 카운터를
    /// 0 으로 리셋(stale op poisoning 방지)한 뒤 error 전파.
    #[cfg(target_os = "android")]
    fn drain_pending(&self) -> Result<()> {
        while self.op_pending.load(Ordering::Acquire) > 0 {
            // dspqueue_read (DSP → host, blocking) — 한 op 응답 회수.
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
                self.op_pending.store(0, Ordering::Release);
                return Err(map_aee_err(rc).context("dspqueue_read (drain)"));
            }
            // HTP_STATUS_OK = 1 (llama.cpp htp-msg.h:24). 0 은 uninitialized.
            if rsp.status != HTP_STATUS_OK {
                self.op_pending.store(0, Ordering::Release);
                return Err(map_htp_status(rsp.status).context("DSP matmul status (drain)"));
            }
            let _ = (rsp_buf_count, rsp_msg_len, rsp_flags);
            self.op_pending.fetch_sub(1, Ordering::AcqRel);
        }
        Ok(())
    }

    /// 단일-op matmul (enqueue + 즉시 drain). 기존 동기 dspqueue_write→read 쌍과
    /// **byte-identical** — `matmul_transposed` 의 비-batch 호출처가 사용. batch
    /// 경로(FFN gate/up)는 `enqueue_matmul_via_htp` 를 직접 ×N 호출 후 묶어서
    /// `drain_pending` 한다.
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
        self.enqueue_matmul_via_htp(a_rpc, b_rpc, out_rpc, n, k, m, weight_dtype)?;
        self.drain_pending()
    }

    // ── pointwise / norm / rope op NPU dispatch (enqueue + drain) ──────────
    //
    // 각 helper 는 (1) packet build (idl init_* + htp_tensor_from_shape),
    // (2) DspQueueBuffer 배열 (microbench bufs ordering mirror),
    // (3) enqueue_packet → drain_pending. 모두 F32 element-wise / 1D-flat 라
    // ne=[numel,1,1,1], nb=[4, numel*4, ...] 단일 row 로 환원 (decode M=1 의
    // pointwise 는 본질적으로 1D — CPU 도 flat slice 로 처리).
    //
    // ★ v79 element-wise(unary/binary) op 은 dispatch 당 ~12032 f32(1 VTCM 타일)
    //   만 처리하고 전체 텐서를 loop 하지 않는다(device 실측). numel > 상한이면
    //   silent 미처리(passthrough) → garbage. flat helper 는 HTP_OP_CHUNK 이하로
    //   분할 dispatch 한다. 8192 = 검증된 안전치(8960 microbench PASS) 미만 + 여유.

    /// SILU(src0) → dst (in-place 시 dst==src0). n_bufs=2 unary_act_req.
    /// microbench `htp_silu.rs` 의 bufs ordering: [0]=src0 CpuWriteDspRead,
    /// [1]=dst DspWriteCpuRead. in-place 면 동일 fd 의 두 view.
    ///
    /// **chunking 필수**: v79 unary op 은 dispatch 당 ~12032 f32(1 VTCM 타일)만
    /// 처리하고 전체 텐서를 loop 하지 않는다(device 실측: numel=44800 prefill 시
    /// idx 12032 부터 입력 passthrough = 미처리). 따라서 [`HTP_OP_CHUNK`] 이하
    /// 조각으로 분할해 dsp_buf offset 으로 sub-range 를 순차 dispatch 한다.
    #[cfg(target_os = "android")]
    fn silu_via_htp(&self, src0: &RpcmemBuffer, dst: &RpcmemBuffer, numel: usize) -> Result<()> {
        let mut off = 0usize;
        while off < numel {
            let n = (numel - off).min(HTP_OP_CHUNK);
            let bytes = (n * 4) as u32;
            let boff = (off * 4) as u32;
            let ne = [n as u32, 1, 1, 1];
            let nb = [4u32, bytes, bytes, bytes];
            let mut req = HtpGeneralReq::zeroed();
            let s0 = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
            let d = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
            init_unary_act_req(&mut req, HTP_OP_UNARY_SILU, s0, d);
            let mut bufs: [DspQueueBuffer; 2] = [
                src0.dsp_buf(DspqBufferType::CpuWriteDspRead, boff, bytes)?,
                dst.dsp_buf(DspqBufferType::DspWriteCpuRead, boff, bytes)?,
            ];
            self.enqueue_packet(&req, &mut bufs)?;
            self.drain_pending()?;
            off += n;
        }
        Ok(())
    }

    /// element-wise binary op(src0, src1) → dst (in-place 시 dst==src0).
    /// op ∈ {MUL, ADD}. n_bufs=3 binary_req. microbench `htp_add.rs`/`htp_mul.rs`
    /// bufs ordering: [0]=src0 CpuWriteDspRead, [1]=src1 CpuWriteDspRead,
    /// [2]=dst DspWriteCpuRead.
    #[cfg(target_os = "android")]
    fn binary_via_htp(
        &self,
        op: u32,
        src0: &RpcmemBuffer,
        src1: &RpcmemBuffer,
        dst: &RpcmemBuffer,
        numel: usize,
    ) -> Result<()> {
        // chunking: silu_via_htp 와 동일 사유(v79 ~12032 f32 상한). flat element-wise.
        let mut off = 0usize;
        while off < numel {
            let n = (numel - off).min(HTP_OP_CHUNK);
            let bytes = (n * 4) as u32;
            let boff = (off * 4) as u32;
            let ne = [n as u32, 1, 1, 1];
            let nb = [4u32, bytes, bytes, bytes];
            let mut req = HtpGeneralReq::zeroed();
            let s0 = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
            let s1 = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
            let d = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
            init_binary_req(&mut req, op, s0, s1, d);
            let mut bufs: [DspQueueBuffer; 3] = [
                src0.dsp_buf(DspqBufferType::CpuWriteDspRead, boff, bytes)?,
                src1.dsp_buf(DspqBufferType::CpuWriteDspRead, boff, bytes)?,
                dst.dsp_buf(DspqBufferType::DspWriteCpuRead, boff, bytes)?,
            ];
            self.enqueue_packet(&req, &mut bufs)?;
            self.drain_pending()?;
            off += n;
        }
        Ok(())
    }

    /// row-broadcast ADD: `dst[r,i] = src0[r,i] + bias[i]` (bias `[dim]` 이 rows
    /// 만큼 반복). ggml ADD repeat 의미 — `mul_row_broadcast_via_htp` 와 op 만 다름.
    #[cfg(target_os = "android")]
    fn add_row_broadcast_via_htp(
        &self,
        src0: &RpcmemBuffer,
        bias: &RpcmemBuffer,
        dst: &RpcmemBuffer,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        let row_bytes = (dim * 4) as u32;
        let plane = (rows * dim * 4) as u32;
        let ne0 = [dim as u32, rows as u32, 1, 1];
        let nb0 = [4u32, row_bytes, plane, plane];
        let ne1 = [dim as u32, 1, 1, 1];
        let nb1 = [4u32, row_bytes, row_bytes, row_bytes];
        let mut req = HtpGeneralReq::zeroed();
        let s0 = htp_tensor_from_shape(HTP_TYPE_F32, ne0, nb0);
        let s1 = htp_tensor_from_shape(HTP_TYPE_F32, ne1, nb1);
        let d = htp_tensor_from_shape(HTP_TYPE_F32, ne0, nb0);
        init_binary_req(&mut req, HTP_OP_ADD, s0, s1, d);
        let mut bufs: [DspQueueBuffer; 3] = [
            src0.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, plane)?,
            bias.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, row_bytes)?,
            dst.dsp_buf(DspqBufferType::DspWriteCpuRead, 0, plane)?,
        ];
        self.enqueue_packet(&req, &mut bufs)?;
        self.drain_pending()
    }

    /// row-broadcast MUL: `dst[r,i] = src0[r,i] * src1[i]` (src1=gamma `[dim]` 이
    /// rows 만큼 반복). ggml MUL 의 repeat 의미 — src1 의 outer dim(ne[1..])=1 이면
    /// src0 의 outer dim 에 broadcast. in-place 시 dst==src0. n_bufs=3.
    #[cfg(target_os = "android")]
    fn mul_row_broadcast_via_htp(
        &self,
        src0: &RpcmemBuffer,
        gamma: &RpcmemBuffer,
        dst: &RpcmemBuffer,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        let row_bytes = (dim * 4) as u32;
        let plane = (rows * dim * 4) as u32;
        // src0/dst: [dim, rows, 1, 1]. gamma: [dim, 1, 1, 1] (broadcast over rows).
        let ne0 = [dim as u32, rows as u32, 1, 1];
        let nb0 = [4u32, row_bytes, plane, plane];
        let ne1 = [dim as u32, 1, 1, 1];
        let nb1 = [4u32, row_bytes, row_bytes, row_bytes];
        let mut req = HtpGeneralReq::zeroed();
        let s0 = htp_tensor_from_shape(HTP_TYPE_F32, ne0, nb0);
        let s1 = htp_tensor_from_shape(HTP_TYPE_F32, ne1, nb1);
        let d = htp_tensor_from_shape(HTP_TYPE_F32, ne0, nb0);
        init_binary_req(&mut req, HTP_OP_MUL, s0, s1, d);
        let mut bufs: [DspQueueBuffer; 3] = [
            src0.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, plane)?,
            gamma.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, row_bytes)?,
            dst.dsp_buf(DspqBufferType::DspWriteCpuRead, 0, plane)?,
        ];
        self.enqueue_packet(&req, &mut bufs)?;
        self.drain_pending()
    }

    /// RMSNORM(src0) → dst (gamma 미적용 — output=input/rms(input)). in-place 시
    /// dst==src0. n_bufs=2. 후속 gamma 곱은 호출처가 별도 MUL 로 처리. `dim` 은
    /// per-row normalize 단위 (마지막 dim). rows = numel/dim 행을 ne[1] 로 묶는다.
    #[cfg(target_os = "android")]
    fn rmsnorm_via_htp(
        &self,
        src0: &RpcmemBuffer,
        dst: &RpcmemBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        let row_bytes = (dim * 4) as u32;
        let plane = (rows * dim * 4) as u32;
        let bytes = plane;
        // ggml convention ne[0]=innermost(=normalize 단위 dim), ne[1]=rows.
        let ne = [dim as u32, rows as u32, 1, 1];
        let nb = [4u32, row_bytes, plane, plane];
        let mut req = HtpGeneralReq::zeroed();
        let s0 = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
        let d = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
        init_rmsnorm_req(&mut req, eps, s0, d);
        let mut bufs: [DspQueueBuffer; 2] = [
            src0.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes)?,
            dst.dsp_buf(DspqBufferType::DspWriteCpuRead, 0, bytes)?,
        ];
        self.enqueue_packet(&req, &mut bufs)?;
        self.drain_pending()
    }

    /// ROPE(src0, positions) → dst (in-place 시 dst==src0). n_bufs=3. positions 는
    /// i32 rpcmem 버퍼 (decode 단일 토큰 = [start_pos]). microbench `htp_rope.rs`
    /// bufs ordering: [0]=src0 CpuWriteDspRead, [1]=positions CpuWriteDspRead,
    /// [2]=dst DspWriteCpuRead.
    #[cfg(target_os = "android")]
    fn rope_via_htp(
        &self,
        src0: &RpcmemBuffer,
        pos: &RpcmemBuffer,
        dst: &RpcmemBuffer,
        params: &RopeParams,
        head_dim: usize,
        n_heads: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let bytes_in = (head_dim * n_heads * n_tokens * 4) as u32;
        let ne_in = [head_dim as u32, n_heads as u32, n_tokens as u32, 1];
        let nb_in = [
            4u32,
            (head_dim * 4) as u32,
            (head_dim * n_heads * 4) as u32,
            bytes_in,
        ];
        let bytes_pos = (n_tokens * 4) as u32;
        let ne_pos = [n_tokens as u32, 1, 1, 1];
        let nb_pos = [4u32, bytes_pos, bytes_pos, bytes_pos];
        let mut req = HtpGeneralReq::zeroed();
        let s0 = htp_tensor_from_shape(HTP_TYPE_F32, ne_in, nb_in);
        let s1 = htp_tensor_from_shape(HTP_TYPE_I32, ne_pos, nb_pos);
        let d = htp_tensor_from_shape(HTP_TYPE_F32, ne_in, nb_in);
        init_rope_req(&mut req, params, s0, s1, d);
        let mut bufs: [DspQueueBuffer; 3] = [
            src0.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes_in)?,
            pos.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes_pos)?,
            dst.dsp_buf(DspqBufferType::DspWriteCpuRead, 0, bytes_in)?,
        ];
        self.enqueue_packet(&req, &mut bufs)?;
        self.drain_pending()
    }

    /// RoPE in-place dispatch. positions scratch(i32 rpcmem)를 lazily alloc/재기록
    /// 한 뒤 `rope_via_htp` 호출 (dst==src0 in-place). decode 토큰당 position 만
    /// 갱신하므로 scratch 는 n_tokens 가 커질 때만 재할당.
    #[cfg(target_os = "android")]
    fn rope_dispatch(
        &self,
        x_rpc: &RpcmemBuffer,
        start_pos: usize,
        n_tokens: usize,
        params: &RopeParams,
        head_dim: usize,
        n_heads: usize,
    ) -> Result<()> {
        let pos_bytes = n_tokens * 4;
        let mut guard = self
            .rope_pos_scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("htp: rope_pos_scratch mutex poisoned"))?;
        // 용량 부족 시 (재)할당. dtype 은 byte-view (U8) — i32 element 해석은 packet
        // 의 HTP_TYPE_I32 가 담당, buffer 는 byte 단위.
        let need_realloc = match guard.as_ref() {
            Some(buf) => buf.size() < pos_bytes,
            None => true,
        };
        if need_realloc {
            *guard = Some(RpcmemBuffer::alloc(
                self.host.clone(),
                pos_bytes,
                DType::U8,
            )?);
        }
        let pos_buf = guard.as_mut().expect("rope_pos_scratch alloc'd above");
        // positions = [start_pos, start_pos+1, ...] i32.
        // SAFETY: pos_buf 는 pos_bytes(=n_tokens*4) ≥ 만큼 alloc, write 범위 내.
        unsafe {
            let p = pos_buf.as_mut_ptr() as *mut i32;
            for t in 0..n_tokens {
                p.add(t).write((start_pos + t) as i32);
            }
        }
        // in-place: dst==src0==x_rpc. pos_buf 는 guard 가 살아있는 동안 valid.
        self.rope_via_htp(x_rpc, pos_buf, x_rpc, params, head_dim, n_heads, n_tokens)
    }

    /// decode flash attention (HTP_OP_FLASH_ATTN_EXT). q=F32, k/v=F16(strict, KV
    /// cache HeadMajor), out=F32. n_bufs=4 (Q/K/V/dst), mask/sinks 미사용.
    /// op_params=[scale=1/√head_dim, max_bias=0, logit_softcap=0]. microbench
    /// `htp_flash_attn_ext.rs` packet mirror — 단 K/V 는 **HeadMajor capacity stride**
    /// (nb[2]=head_dim*capacity*2, ne[1]=cache_seq_len) 로 실제 KV cache 를 가리킨다.
    /// dst [head_dim, 1, n_heads_q] = CPU attention_gen 의 out[h*head_dim+d] 와 동일.
    /// GQA(n_heads_q≠n_kv)는 DSP-side 가 처리. score 미반환이라 caller 가 scores
    /// None 일 때만 진입(eviction off).
    #[cfg(target_os = "android")]
    #[allow(clippy::too_many_arguments)]
    fn flash_attn_via_htp(
        &self,
        q_rpc: &RpcmemBuffer,
        k_rpc: &RpcmemBuffer,
        v_rpc: &RpcmemBuffer,
        out_rpc: &RpcmemBuffer,
        num_heads_q: usize,
        num_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
        capacity: usize,
    ) -> Result<()> {
        let hd = head_dim as u32;
        let csl = cache_seq_len as u32;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // gqa_ratio = kv-head 당 q-head 수. v79 flash_attn 은 **항상 kv_h=0 만 사용**
        // (GQA 미iterate, device 실측: nhkv=2 시 q-head 6~11=kv_h1 group 전부 garbage).
        // 우회: kv-head 를 1개씩 떼어 그 그룹의 gqa q-head 만 dispatch → DSP 가 보는
        // kv-head 는 1개(=올바른 그룹)뿐이라 kv_h=0 이 항상 정답. HeadMajor 에서
        // single head 의 valid 위치는 contiguous(offset g*capacity*head_dim) 라 gather 불요.
        let gqa = num_heads_q / num_heads_kv;
        let g32 = gqa as u32;
        for g in 0..num_heads_kv {
            // Q F32: group g 의 q-head [g*gqa .. (g+1)*gqa]. ne=[head_dim, gqa, 1, 1].
            let ne_q = [hd, g32, 1, 1];
            let nb_q = [4u32, hd * 4, hd * g32 * 4, hd * g32 * 4];
            // K/V F16: 단일 kv-head g, valid csl 위치(HeadMajor 내 contiguous).
            let ne_k = [hd, csl, 1, 1];
            let nb_k = [2u32, hd * 2, hd * csl * 2, hd * csl * 2];
            // dst F32: group g 의 out-head. ne=[head_dim, 1, gqa, 1] → out[h*head_dim+d].
            let ne_d = [hd, 1, g32, 1];
            let nb_d = [4u32, hd * 4, hd * 4, hd * g32 * 4];

            let mut req = HtpGeneralReq::zeroed();
            req.op = HTP_OP_FLASH_ATTN_EXT;
            req.flags = 0;
            req.op_params = [0; HTP_MAX_OP_PARAMS_SLOTS];
            req.op_params[0] = scale.to_bits() as i32;
            req.src0 = htp_tensor_from_shape(HTP_TYPE_F32, ne_q, nb_q);
            req.src1 = htp_tensor_from_shape(HTP_TYPE_F16, ne_k, nb_k);
            req.src2 = htp_tensor_from_shape(HTP_TYPE_F16, ne_k, nb_k);
            req.dst = htp_tensor_from_shape(HTP_TYPE_F32, ne_d, nb_d);

            let q_off = (g * gqa * head_dim * 4) as u32; // F32 q-head group
            let q_sz = (gqa * head_dim * 4) as u32;
            let k_off = (g * capacity * head_dim * 2) as u32; // F16 HeadMajor head g
            let k_sz = (cache_seq_len * head_dim * 2) as u32; // valid csl 위치만
            let d_off = (g * gqa * head_dim * 4) as u32; // F32 out-head group
            let d_sz = (gqa * head_dim * 4) as u32;
            let mut bufs: [DspQueueBuffer; 4] = [
                q_rpc.dsp_buf(DspqBufferType::CpuWriteDspRead, q_off, q_sz)?,
                k_rpc.dsp_buf(DspqBufferType::CpuWriteDspRead, k_off, k_sz)?,
                v_rpc.dsp_buf(DspqBufferType::CpuWriteDspRead, k_off, k_sz)?,
                out_rpc.dsp_buf(DspqBufferType::DspWriteCpuRead, d_off, d_sz)?,
            ];
            self.enqueue_packet(&req, &mut bufs)?;
            self.drain_pending()?;
        }
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

/// `a @ bᵀ → out` matmul 의 NPU dispatch plan. weight `b`/act `a`/out 이 전부
/// RpcmemBuffer-backed 이고 dtype/shape 게이트(`htp_matmul_dispatchable[_f16]`)를
/// 통과하면 `Some((a_rpc, b_rpc, out_rpc, n, k, m, weight_dtype))`. 하나라도
/// 미충족이면 `None` — 호출처(FFN batch override)는 None 시 개별 `matmul_transposed`
/// 로 fallback 하며, 그쪽이 Q4_0 q4x4x2 의 cpu garbage bail / F16 safe-cpu 불변을
/// 보존한다. `matmul_transposed` 의 dtype 분기와 동일 게이트를 재사용해 batch 경로의
/// 판정이 단일-op 경로와 어긋나지 않게 한다.
#[cfg(target_os = "android")]
#[allow(clippy::type_complexity)]
fn htp_matmul_dispatch_plan<'t>(
    a: &'t Tensor,
    b: &'t Tensor,
    out: &'t Tensor,
) -> Option<(
    &'t RpcmemBuffer,
    &'t RpcmemBuffer,
    &'t RpcmemBuffer,
    usize,
    usize,
    usize,
    DType,
)> {
    let b_rpc = b.buffer().as_any().downcast_ref::<RpcmemBuffer>()?;
    let a_rpc = a.buffer().as_any().downcast_ref::<RpcmemBuffer>()?;
    let out_rpc = out.buffer().as_any().downcast_ref::<RpcmemBuffer>()?;
    match b.dtype() {
        DType::Q4_0 => {
            let (n, k, m) = htp_matmul_dispatchable(a, b, out)?;
            Some((a_rpc, b_rpc, out_rpc, n, k, m, DType::Q4_0))
        }
        DType::F16 => {
            let (n, k, m) = htp_matmul_dispatchable_f16(a, b, out)?;
            Some((a_rpc, b_rpc, out_rpc, n, k, m, DType::F16))
        }
        _ => None,
    }
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

    /// step2 — FFN gate+up 를 한 batch 로 NPU dispatch (async batch fusion).
    ///
    /// gate/up 은 둘 다 normed FFN 입력 `x` 만 읽는 **독립** matmul 이고 출력 buffer
    /// (`out`/`up_scratch`)가 disjoint 라, enqueue×2 → drain×1 로 묶어 ~100µs FastRPC
    /// dispatch floor 를 op 마다(2회)→batch(1회)로 amortize 한다. 이후 `silu_mul` 이
    /// gate·up 둘 다 읽으므로 **반드시 drain 후** CPU 실행 (drain 이 DspWriteCpuRead
    /// coherency 까지 보장 — PoC `htp_batch_dispatch` 검증). attention/KV-cache 와
    /// 무관한 KV-free 블록이라 eviction/KIVI/D2O 정책과 결합이 없다 (fusion 경계를
    /// KV 바깥에 둔다는 설계 불변).
    ///
    /// gate·up 중 하나라도 NPU dispatch 불가(비-rpcmem / dtype·shape 미충족 / non-android)
    /// 면 trait default 와 동일한 개별 `matmul_transposed`×2 + `silu_mul` 로 fallback
    /// — 그쪽이 Q4_0 q4x4x2 cpu garbage bail / F16 safe-cpu 불변을 per-op 보존한다.
    fn matmul_ffn_gate_up_silu(
        &self,
        x: &Tensor,
        w_gate: &Tensor,
        w_up: &Tensor,
        out: &mut Tensor,
        up_scratch: &mut Tensor,
    ) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            // env LLMRS_DISABLE_HTP_FFN_BATCH=1 이면 batch 끄고 fallback(개별 dispatch).
            // A/B 측정으로 floor recovery 효과를 격리 + production 안전 토글
            // (기존 LLMRS_DISABLE_FLUSH_FFN 패턴과 동일 역할).
            let batch_enabled = std::env::var_os("LLMRS_DISABLE_HTP_FFN_BATCH").is_none();
            // gate·up 둘 다 NPU dispatch 가능할 때만 batch (그래야 enqueue 도중 한쪽이
            // 불가로 판명돼 half-enqueue 되는 상황을 회피). 판정은 단일-op 경로와 동일
            // 게이트(htp_matmul_dispatch_plan) 재사용.
            if batch_enabled
                && let (Some(g), Some(u)) = (
                    htp_matmul_dispatch_plan(x, w_gate, out),
                    htp_matmul_dispatch_plan(x, w_up, up_scratch),
                )
            {
                // 전용 Once — htp_dispatch_log_once 의 DISPATCH_ONCE 는 q_proj 가 먼저
                // 소비하므로 batch 진입을 별도 1회 로그로 확증 ("no silent caps").
                {
                    use std::sync::Once;
                    static FFN_BATCH_ONCE: Once = Once::new();
                    FFN_BATCH_ONCE.call_once(|| {
                        eprintln!(
                            "[htp] matmul_ffn_gate_up_silu: gate/up batch dispatch 활성 \
                             (enqueue×2→drain×1, floor 2→1)"
                        );
                    });
                }
                // enqueue gate → enqueue up (op_pending 0→2), drain 1회 (2 read).
                self.enqueue_matmul_via_htp(g.0, g.1, g.2, g.3, g.4, g.5, g.6)?;
                self.enqueue_matmul_via_htp(u.0, u.1, u.2, u.3, u.4, u.5, u.6)?;
                self.drain_pending()?;
                // gate/up 모두 완성·coherent → silu_mul(out ← silu(out)*up_scratch) CPU.
                return self.silu_mul(out, up_scratch);
            }
        }
        // fallback: trait default 동치. 개별 matmul_transposed 가 dispatch 분기/bail 보존.
        self.matmul_transposed(x, w_gate, out)?;
        self.matmul_transposed(x, w_up, up_scratch)?;
        self.silu_mul(out, up_scratch)
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

    /// `a += b` element-wise. env `LLMRS_HTP_NPU_ADD=1` + a/b 둘 다 RpcmemBuffer-backed
    /// 면 NPU `ADD(a, b) → a` (in-place). 그 외 cpu_companion (회귀 0).
    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            if std::env::var_os("LLMRS_HTP_NPU_ADD").is_some()
                && a.dtype() == DType::F32
                && b.dtype() == DType::F32
                && a.numel() == b.numel()
                && let (Some(a_rpc), Some(b_rpc)) = (
                    a.buffer().as_any().downcast_ref::<RpcmemBuffer>(),
                    b.buffer().as_any().downcast_ref::<RpcmemBuffer>(),
                )
            {
                // in-place: dst==src0==a_rpc (동일 fd 의 두 view).
                return self.binary_via_htp(HTP_OP_ADD, a_rpc, b_rpc, a_rpc, a.numel());
            }
        }
        self.cpu_companion.add_assign(a, b)
    }

    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()> {
        self.cpu_companion.scale(x, v)
    }

    /// row-broadcast bias add. env `LLMRS_HTP_NPU_ADD=1` + x/bias 둘 다 rpcmem-backed
    /// 면 NPU MUL/ADD 의 row-broadcast 경로(`ADD(x, bias)` broadcast). decode M=1 은
    /// x.numel()==bias.numel() 라 단순 element-wise ADD 로 환원되지만, rows>1 (prefill)
    /// 도 ggml ADD 의 src1 outer-dim=1 broadcast 로 일반화. 그 외 trait default(CPU).
    fn add_row_bias(&self, x: &mut Tensor, bias: &Tensor) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            if std::env::var_os("LLMRS_HTP_NPU_ADD").is_some()
                && x.dtype() == DType::F32
                && bias.dtype() == DType::F32
                && let (Some(x_rpc), Some(bias_rpc)) = (
                    x.buffer().as_any().downcast_ref::<RpcmemBuffer>(),
                    bias.buffer().as_any().downcast_ref::<RpcmemBuffer>(),
                )
            {
                let dim = bias.numel();
                if dim > 0 && x.numel().is_multiple_of(dim) {
                    let rows = x.numel() / dim;
                    return self.add_row_broadcast_via_htp(x_rpc, bias_rpc, x_rpc, rows, dim);
                }
            }
        }
        // trait default (CPU broadcast).
        let x_data = x.as_mut_slice::<f32>();
        let b_data = bias.as_slice::<f32>();
        let dim = b_data.len();
        for row in x_data.chunks_mut(dim) {
            for (v, &b) in row.iter_mut().zip(b_data.iter()) {
                *v += b;
            }
        }
        Ok(())
    }

    /// `a[i] = silu(a[i]) * b[i]`. env `LLMRS_HTP_NPU_SILU=1` + a/b 둘 다
    /// RpcmemBuffer-backed 면 NPU 2-op: `SILU(a) → a` (unary in-place) +
    /// `MUL(a, b) → a` (binary in-place). 그 외 cpu_companion (회귀 0).
    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            if std::env::var_os("LLMRS_HTP_NPU_SILU").is_some()
                && a.dtype() == DType::F32
                && b.dtype() == DType::F32
                && a.numel() == b.numel()
                && let (Some(a_rpc), Some(b_rpc)) = (
                    a.buffer().as_any().downcast_ref::<RpcmemBuffer>(),
                    b.buffer().as_any().downcast_ref::<RpcmemBuffer>(),
                )
            {
                let numel = a.numel();
                // SILU(a) → a (in-place), 이어서 MUL(a, b) → a (in-place). 둘 다
                // chunked(silu_via_htp/binary_via_htp 가 HTP_OP_CHUNK 분할) — v79 의
                // ~12032 f32 dispatch 상한을 넘는 prefill gate(M×ffn_hidden)도 정확.
                self.silu_via_htp(a_rpc, a_rpc, numel)?;
                return self.binary_via_htp(HTP_OP_MUL, a_rpc, b_rpc, a_rpc, numel);
            }
        }
        self.cpu_companion.silu_mul(a, b)
    }

    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        self.cpu_companion.softmax(x)
    }

    /// decode attention. env `LLMRS_HTP_NPU_ATTN=1` + scores 미요청(eviction off) +
    /// q F32 / k,v F16(HeadMajor) / out F32 + 전부 rpcmem 면 NPU flash_attn,
    /// 그 외 cpu_companion (회귀 0). DSP flash_attn 은 score 를 반환하지 않으므로
    /// scores_out=Some(H2O/D2O) 면 NPU 불가 → CPU. **v79 flash_attn 의 수치 정확성은
    /// microbench 에서 status 만 검증됨 — token-id 로 device 검증 필요(리스크).**
    #[allow(clippy::too_many_arguments)]
    fn attention_gen(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        num_heads_q: usize,
        num_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
        scores_out: Option<&mut [f32]>,
    ) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            let ks = k_cache.shape().dims();
            // HeadMajor 판정 = CPU attention_gen 과 동일 (ks[1]==n_kv && ks[1]!=ks[2]).
            let is_head_major = ks.len() >= 3 && ks[1] == num_heads_kv && ks[1] != ks[2];
            if std::env::var_os("LLMRS_HTP_NPU_ATTN").is_some()
                && scores_out.is_none()
                && is_head_major
                && q.dtype() == DType::F32
                && k_cache.dtype() == DType::F16
                && v_cache.dtype() == DType::F16
                && out.dtype() == DType::F32
                && let (Some(q_rpc), Some(k_rpc), Some(v_rpc), Some(out_rpc)) = (
                    q.buffer().as_any().downcast_ref::<RpcmemBuffer>(),
                    k_cache.buffer().as_any().downcast_ref::<RpcmemBuffer>(),
                    v_cache.buffer().as_any().downcast_ref::<RpcmemBuffer>(),
                    out.buffer().as_any().downcast_ref::<RpcmemBuffer>(),
                )
            {
                let capacity = ks[2];
                htp_dispatch_log_once(true, "flash_attn (per-kv-head)");
                return self.flash_attn_via_htp(
                    q_rpc,
                    k_rpc,
                    v_rpc,
                    out_rpc,
                    num_heads_q,
                    num_heads_kv,
                    head_dim,
                    cache_seq_len,
                    capacity,
                );
            }
        }
        self.cpu_companion.attention_gen(
            q,
            k_cache,
            v_cache,
            out,
            num_heads_q,
            num_heads_kv,
            head_dim,
            cache_seq_len,
            scores_out,
        )
    }

    /// In-place RoPE. env `LLMRS_HTP_NPU_ROPE=1` + x RpcmemBuffer-backed 면 NPU
    /// ROPE op (in-place), 그 외 cpu_companion (회귀 0).
    ///
    /// **rotation mode**: CPU 레퍼런스(`cpu/common.rs::rope_inplace`)는 NeoX-style
    /// split — `head_slice[i]` 를 `head_slice[i+half_dim]` 와 페어링하고 freq =
    /// `theta^(-2i/head_dim)`. 이는 ggml mode=2(GGML_ROPE_TYPE_NEOX) 와 동일하므로
    /// `RopeParams.mode=2`. (mode=0 normal interleaved 와 결과가 다르다 — 정확성
    /// 검증에서 mode 매핑이 핵심.)
    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            if std::env::var_os("LLMRS_HTP_NPU_ROPE").is_some() && x.dtype() == DType::F32 {
                let dims = x.shape().dims();
                // CPU rope_inplace 와 동일 dim 추출: 마지막=head_dim, 직전=n_heads,
                // 그 직전=seq_len. (decode: [batch, 1, n_heads, head_dim] → seq=1.)
                if dims.len() >= 3 {
                    let head_dim = dims[dims.len() - 1];
                    let n_heads = dims[dims.len() - 2];
                    let n_tokens = dims[dims.len() - 3];
                    if let Some(x_rpc) = x.buffer().as_any().downcast_ref::<RpcmemBuffer>() {
                        // positions [start_pos .. start_pos+n_tokens) i32 → lazily-alloc
                        // scratch 에 기록. NeoX freq_base=theta, mode=2 (CPU 레퍼런스 일치).
                        // device 실측 max_err 1e-5 (F32 rounding) — bit-exact 에 가까움.
                        let params = RopeParams {
                            n_dims: head_dim as i32,
                            mode: 2,
                            n_ctx_orig: 0,
                            freq_base: theta,
                            freq_scale: 1.0,
                            ext_factor: 0.0,
                            attn_factor: 1.0,
                            beta_fast: 0.0,
                            beta_slow: 0.0,
                        };
                        return self
                            .rope_dispatch(x_rpc, start_pos, n_tokens, &params, head_dim, n_heads);
                    }
                }
            }
        }
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
        #[cfg(target_os = "android")]
        {
            // add_unit=true(Gemma: (1+w)) 는 NPU MUL 로 직접 표현 불가 → cpu fallback.
            // add_unit=false(Qwen/Llama)만 NPU: RMSNORM(x)→x + MUL(x, gamma)→x.
            if !add_unit
                && std::env::var_os("LLMRS_HTP_NPU_RMSNORM").is_some()
                && x.dtype() == DType::F32
                && w.dtype() == DType::F32
                && let (Some(x_rpc), Some(w_rpc)) = (
                    x.buffer().as_any().downcast_ref::<RpcmemBuffer>(),
                    w.buffer().as_any().downcast_ref::<RpcmemBuffer>(),
                )
            {
                let dims = x.shape().dims();
                let dim = dims[dims.len() - 1];
                if dim > 0 && x.numel().is_multiple_of(dim) {
                    let rows = x.numel() / dim;
                    // RMSNORM(x)→x (gamma 미적용), 이어서 row-broadcast MUL(x, gamma)→x.
                    self.rmsnorm_via_htp(x_rpc, x_rpc, rows, dim, eps)?;
                    return self.mul_row_broadcast_via_htp(x_rpc, w_rpc, x_rpc, rows, dim);
                }
            }
        }
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
            // F32 1D weight (norm gamma, qkv bias): rpcmem alloc + byte 복사.
            // pointwise/norm op NPU override (rms_norm gamma MUL, add_row_bias) 가
            // weight 를 DSP 로 넘기려면 rpcmem-backed 여야 한다. 1D 로 한정해
            // lm_head/embedding 의 2D F32 (cpu matmul 대상) 는 건드리지 않는다.
            // env 게이트 OFF 여도 promote 자체는 무해 (rpcmem 약간 더 씀, op 가
            // downcast 성공해도 env OFF 면 cpu fallback 경로라 미사용).
            DType::F32 if dims.len() == 1 => {
                let bytes = t.numel() * 4;
                let mut buf = RpcmemBuffer::alloc(self.host.clone(), bytes, DType::F32)?;
                let f32_slice = t.as_slice::<f32>();
                // SAFETY: f32 = 4-byte POD, f32_slice.len()*4 == bytes == buf alloc.
                let src_bytes =
                    unsafe { std::slice::from_raw_parts(f32_slice.as_ptr() as *const u8, bytes) };
                // SAFETY: buf 직전 alloc 으로 valid, src/dst non-overlapping, len 동일.
                unsafe {
                    std::ptr::copy_nonoverlapping(src_bytes.as_ptr(), buf.as_mut_ptr(), bytes);
                }
                Ok(Tensor::new(
                    t.shape().clone(),
                    Arc::new(buf),
                    self.cpu_companion.clone(),
                ))
            }
            // 그 외 (F32 2D/Q8_0/Q6_K/… , 비정렬·비2D Q4_0/F16): cpu_companion
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
        // async batch fusion: enqueue 된 in-flight DSP matmul op 을 전부 drain
        // (full barrier). 단일-op 경로는 매 matmul 이 즉시 drain 하므로 여기서
        // op_pending==0 → no-op (동작 불변). batch 경로(FFN gate/up)는 자체적으로
        // drain 후 반환하므로 역시 여기 도달 시 0 — 즉 본 fence 는 forward 의
        // GPU-gated sync 지점이 HTP 로도 확장될 때를 위한 forward-compatible 안전망.
        #[cfg(target_os = "android")]
        {
            self.drain_pending()?;
        }
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        // HTP 는 비차단 submit(clFlush 류) 개념이 없다 — 출력 가시화의 유일 수단이
        // blocking dspqueue_read 이므로 flush == drain (pending op 회수). forward 의
        // batch 경계 flush 지점(QKV/FFN)이 HTP 로 확장되면 floor amortize 동작.
        #[cfg(target_os = "android")]
        {
            self.drain_pending()?;
        }
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
