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
    HTP_OP_MUL, HTP_OP_MUL_MAT, HTP_OP_RMS_NORM, HTP_OP_ROPE, HTP_OP_UNARY_GELU, HTP_OP_UNARY_SILU,
    HTP_OPFLAGS_SKIP_QUANTIZE, HTP_TYPE_F16, HTP_TYPE_F32, HTP_TYPE_I32, HTP_TYPE_Q4_0,
    HTP_TYPE_Q8_0, HtpGeneralReq, HtpGeneralRsp, HtpTensor, RopeParams, htp_tensor_from_shape,
    init_binary_req, init_matmul_req, init_rmsnorm_req, init_rope_req, init_unary_act_req,
};

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

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
}

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

    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
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
}
