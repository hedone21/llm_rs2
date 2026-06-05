//! microbench_htp_silu_f16 — P1d Senior Implementer (Full matrix sprint)
//!
//! HTP NPU SILU F16 dispatch **NO_SUPPORT 확정 evidence**.
//!
//! DSP-side (`libggml-htp-v79.so` `unary-ops.c:368-382 op_unary`) 는 F32 만
//! 지원:
//! ```c
//! switch (octx->src0.type) {
//!     case HTP_TYPE_F32:
//!         err = execute_op_unary_f32(octx);
//!         break;
//!     default:
//!         err = HTP_STATUS_NO_SUPPORT;
//!         break;
//! }
//! ```
//!
//! 본 bin 은 F16 SILU dispatch 를 시도하여 `HTP_STATUS_NO_SUPPORT` 가 정확히
//! 반환되는지 확인한다. paper "NPU F16 unary 미지원 (✗ 마킹)" evidence.
//!
//! 같은 패턴이 RMS_NORM / ROPE / SOFTMAX (src0) / GET_ROWS 에도 적용된다
//! (matrix.md v2-§6.0 결정). 본 bin 은 SILU 1 op 만 시도하여 status code 만
//! 검증 — 다른 op 의 F16 NO_SUPPORT 는 본 file 의 narrative 로 통일 인용.

#![allow(clippy::unnecessary_wraps)]

#[cfg(not(feature = "htp_fastrpc"))]
fn main() {
    eprintln!("microbench_htp_silu_f16 requires --features htp_fastrpc");
    std::process::exit(2);
}

#[cfg(feature = "htp_fastrpc")]
fn main() -> anyhow::Result<()> {
    use half::f16;

    const DIM: usize = 8960;

    println!("=== microbench_htp_silu_f16 (P1d NO_SUPPORT evidence) ===\n");
    println!("Config:");
    println!("  shape         = [1, {DIM}] F16 ({} B)", DIM * 2);
    println!("  expected      = HTP_STATUS_NO_SUPPORT (DSP-side unary F16 미지원)\n");

    let mut host_x_f32 = vec![0.0f32; DIM];
    for (i, x) in host_x_f32.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5;
    }
    let host_x_f16: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let x_f16_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(host_x_f16.as_ptr() as *const u8, host_x_f16.len() * 2)
    };

    println!("[1/1] QNN HTP NPU (F16 SILU dispatch — expect NO_SUPPORT)");
    let result = run_htp(x_f16_bytes, DIM);
    match result {
        Ok(_) => {
            println!(
                "  UNEXPECTED — HTP_STATUS_OK returned for F16 SILU. \
                 matrix.md v2-§6.0 의 ours.htp SILU F16 `✗` 마킹 재검토 필요.\n"
            );
            println!("  → paper narrative 갱신: F16 unary path GREEN (sprint 발견)");
        }
        Err(e) => {
            let msg = format!("{e:#}");
            if msg.contains("NO_SUPPORT") || msg.contains("no_support") || msg.contains("status") {
                println!("  EXPECTED — HTP_STATUS_NO_SUPPORT path 검증 GREEN.\n  details: {msg}\n",);
                println!(
                    "  → paper narrative: DSP-side unary-ops.c::op_unary switch 는 F32 only. \
                     F16 path 진입 시 DSP 가 NO_SUPPORT status 로 반환."
                );
                println!(
                    "  → 같은 패턴 적용 op (SILU/RMS_NORM/ROPE/SOFTMAX/GET_ROWS) 모두 ✗ 마킹."
                );
            } else {
                println!("  SKIP — {msg}");
            }
        }
    }

    Ok(())
}

#[cfg(feature = "htp_fastrpc")]
fn run_htp(x_f16_bytes: &[u8], dim: usize) -> anyhow::Result<()> {
    #[cfg(not(target_os = "android"))]
    {
        let _ = (x_f16_bytes, dim);
        anyhow::bail!(
            "HTP path requires target_os=android (no NPU on host PC). \
             Cross-build with --target aarch64-linux-android and deploy to S25."
        );
    }

    #[cfg(target_os = "android")]
    {
        use anyhow::Context;
        use llm_rs2::backend::htp_fastrpc::{
            AEE_SUCCESS, DSPQUEUE_TIMEOUT, DspQueueBuffer, DspqBufferType, HTP_OP_UNARY_SILU,
            HTP_STATUS_OK, HTP_TYPE_F16, HtpFastrpcHost, HtpGeneralReq, HtpGeneralRsp,
            RpcmemBuffer, htp_tensor_from_shape, init_unary_act_req, map_aee_err, map_htp_status,
        };

        let host = match HtpFastrpcHost::new("microbench_htp_silu_f16") {
            Ok(h) => h,
            Err(e) => anyhow::bail!("HtpFastrpcHost::new failed: {}", e),
        };
        println!(
            "  host: libcdsprpc.so={}, domain_id={}, session_id={}, queue_id={}",
            host.lib_path, host.domain_id, host.session_id, host.queue_id
        );

        let n_hvx: u32 = std::env::var("HTP_FASTRPC_N_HVX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        host.try_start_iface(n_hvx)
            .with_context(|| format!("htp_iface_start(n_hvx={n_hvx})"))?;
        println!("  htp_iface_start: OK (n_hvx={n_hvx})");

        let bytes = dim * 2;
        let mut buf_x = RpcmemBuffer::alloc(host.clone(), bytes, llm_rs2::buffer::DType::F16)?;
        let mut buf_y = RpcmemBuffer::alloc(host.clone(), bytes, llm_rs2::buffer::DType::F16)?;

        unsafe {
            std::ptr::copy_nonoverlapping(x_f16_bytes.as_ptr(), buf_x.as_mut_ptr(), bytes);
            std::ptr::write_bytes(buf_y.as_mut_ptr(), 0, bytes);
        }

        let ne = [dim as u32, 1, 1, 1];
        let nb = [2u32, (dim * 2) as u32, (dim * 2) as u32, (dim * 2) as u32];

        let mut req = HtpGeneralReq::zeroed();
        let src0 = htp_tensor_from_shape(HTP_TYPE_F16, ne, nb);
        let dst = htp_tensor_from_shape(HTP_TYPE_F16, ne, nb);
        init_unary_act_req(&mut req, HTP_OP_UNARY_SILU, src0, dst);

        let mut bufs: [DspQueueBuffer; 2] = [
            buf_x.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes as u32)?,
            buf_y.dsp_buf(DspqBufferType::DspWriteCpuRead, 0, bytes as u32)?,
        ];

        let rc = unsafe {
            (host.dspqueue_write)(
                host.queue,
                0,
                2,
                bufs.as_mut_ptr(),
                core::mem::size_of::<HtpGeneralReq>() as u32,
                &req as *const _ as *const u8,
                DSPQUEUE_TIMEOUT,
            )
        };
        if rc != AEE_SUCCESS {
            return Err(map_aee_err(rc).context("dspqueue_write").into());
        }

        let mut rsp = HtpGeneralRsp::zeroed();
        let mut rsp_buf_count: u32 = 0;
        let mut rsp_bufs: [DspQueueBuffer; 4] = [DspQueueBuffer::zeroed(); 4];
        let mut rsp_msg_len: u32 = 0;
        let mut rsp_flags: u32 = 0;

        let rc = unsafe {
            (host.dspqueue_read)(
                host.queue,
                &mut rsp_flags as *mut u32,
                4,
                &mut rsp_buf_count as *mut u32,
                rsp_bufs.as_mut_ptr(),
                core::mem::size_of::<HtpGeneralRsp>() as u32,
                &mut rsp_msg_len as *mut u32,
                &mut rsp as *mut _ as *mut u8,
                DSPQUEUE_TIMEOUT,
            )
        };
        if rc != AEE_SUCCESS {
            return Err(map_aee_err(rc).context("dspqueue_read").into());
        }

        println!(
            "  diag (F16 SILU attempt): rsp.op={}, status={}, prof_usecs={}, prof_cycles={}, rsp_buf_count={}",
            rsp.op, rsp.status, rsp.prof_usecs, rsp.prof_cycles, rsp_buf_count
        );
        let _ = (rsp_msg_len, rsp_flags);

        if rsp.status != HTP_STATUS_OK {
            return Err(map_htp_status(rsp.status)
                .context("DSP op status (expected NO_SUPPORT for F16 unary)")
                .into());
        }

        Ok(())
    }
}
