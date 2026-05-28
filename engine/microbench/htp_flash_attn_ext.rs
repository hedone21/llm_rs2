//! microbench_htp_flash_attn_ext — P1d Senior Implementer (Full matrix sprint)
//!
//! HTP NPU FLASH_ATTN_EXT dispatch — Qwen 2.5-1.5B decode attention (GQA).
//!
//! DSP-side (`libggml-htp-v79.so` `flash-attn-ops.c:610-666 op_flash_attn_ext`)
//! support check (line 618):
//! ```c
//! if ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32)
//!     || k->type != HTP_TYPE_F16 || v->type != HTP_TYPE_F16) {
//!     return HTP_STATUS_NO_SUPPORT;
//! }
//! ```
//! → Q ∈ {F16, F32}, K/V strict F16. dst ∈ {F32, F16}.
//!
//! Sprint goal: matrix.md v2-§6.0 의 ours.htp FLASH_ATTN_EXT cell `✗` 가정
//! 검증. dispatch 자체 GREEN (HTP_STATUS_OK) 이면 paper narrative 갱신,
//! NO_SUPPORT 또는 다른 status 면 `✗` 확정 (G1~G5 의 7/7 GREEN inherit 후
//! attention 만 미지원 = paper main evidence).
//!
//! Shape (Qwen 2.5-1.5B decode, seq=1024 ctx):
//!   Q  : `[head_dim=128, n_heads=12, n_tokens=1, 1]` F16
//!   K  : `[head_dim=128, ctx=1024, n_kv_heads=2, 1]` F16
//!   V  : `[head_dim=128, ctx=1024, n_kv_heads=2, 1]` F16
//!   dst: `[head_dim=128, n_tokens=1, n_heads=12, 1]` F32
//!
//! mask / sinks 미사용 (src3/src4 buffer 자체 미전송, n_bufs=4 attempt).
//!
//! op_params (3 f32 slots, ggml convention):
//!   [0] scale         = 1/sqrt(128) ≈ 0.0883
//!   [1] max_bias      = 0.0 (no ALiBi)
//!   [2] logit_softcap = 0.0 (no soft cap)
//!
//! Per backend per attempt: warmup 5 + measure 50 iter, median + mean + stddev.
//!
//! Build (Android cross):
//!   cargo build --release --features htp_fastrpc \
//!       --target aarch64-linux-android --bin microbench_htp_flash_attn_ext
//!
//! Run on device:
//!   adb -s R3CY408S5SB push target/aarch64-linux-android/release/microbench_htp_flash_attn_ext /data/local/tmp/
//!   adb -s R3CY408S5SB shell "LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64 \
//!              ADSP_LIBRARY_PATH=/data/local/tmp \
//!              /data/local/tmp/microbench_htp_flash_attn_ext"

#![allow(clippy::unnecessary_wraps)]

#[cfg(not(feature = "htp_fastrpc"))]
fn main() {
    eprintln!("microbench_htp_flash_attn_ext requires --features htp_fastrpc");
    std::process::exit(2);
}

#[cfg(feature = "htp_fastrpc")]
fn main() -> anyhow::Result<()> {
    use half::f16;

    const HEAD_DIM: usize = 128;
    const N_HEADS: usize = 12;
    const N_KV: usize = 2;
    const N_TOKENS: usize = 1;
    const CTX: usize = 1024;
    const WARMUP: usize = 5;
    const MEASURE: usize = 50;

    println!("=== microbench_htp_flash_attn_ext (P1d sprint) ===\n");
    println!("Config (Qwen 2.5-1.5B decode):");
    println!("  Q   = [head_dim={HEAD_DIM}, n_heads={N_HEADS}, n_tokens={N_TOKENS}] F16");
    println!("  K/V = [head_dim={HEAD_DIM}, ctx={CTX}, n_kv={N_KV}] F16");
    println!("  dst = [head_dim={HEAD_DIM}, n_tokens={N_TOKENS}, n_heads={N_HEADS}] F32");
    println!("  scale = 1/sqrt({HEAD_DIM})");
    println!("  warmup={WARMUP} measure={MEASURE}\n");

    // ── Deterministic synthetic data ──────────────────────────────────────
    let q_size = HEAD_DIM * N_HEADS * N_TOKENS;
    let kv_size = HEAD_DIM * CTX * N_KV;
    let dst_size = HEAD_DIM * N_TOKENS * N_HEADS;

    let mut host_q_f32 = vec![0.0f32; q_size];
    let mut host_k_f32 = vec![0.0f32; kv_size];
    let mut host_v_f32 = vec![0.0f32; kv_size];
    for (i, x) in host_q_f32.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5;
    }
    for (i, x) in host_k_f32.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0291 + 0.13).rem_euclid(1.0) - 0.5;
    }
    for (i, x) in host_v_f32.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0411 + 0.17).rem_euclid(1.0) - 0.5;
    }

    let host_q_f16: Vec<f16> = host_q_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_k_f16: Vec<f16> = host_k_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_v_f16: Vec<f16> = host_v_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let q_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(host_q_f16.as_ptr() as *const u8, host_q_f16.len() * 2)
    };
    let k_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(host_k_f16.as_ptr() as *const u8, host_k_f16.len() * 2)
    };
    let v_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(host_v_f16.as_ptr() as *const u8, host_v_f16.len() * 2)
    };

    println!("[1/1] QNN HTP NPU (FLASH_ATTN_EXT dispatch attempt)");
    let result = run_htp(
        q_bytes, k_bytes, v_bytes, dst_size, HEAD_DIM, N_HEADS, N_KV, N_TOKENS, CTX, WARMUP,
        MEASURE,
    );
    match result {
        Ok(stats) => {
            println!(
                "  GREEN — mean={:.2} us (median={:.2}, stddev={:.2}, n={})",
                stats.mean_us, stats.median_us, stats.stddev_us, MEASURE
            );
            println!(
                "  → matrix.md v2-§6.0 ours.htp FLASH_ATTN_EXT cell `✗` 가정 reject. \
                 paper narrative 갱신 필요 (HTP NPU 가 fused attention GREEN dispatch 가능)."
            );
        }
        Err(e) => {
            let msg = format!("{e:#}");
            println!("  FAIL — {msg}");
            if msg.contains("NO_SUPPORT") || msg.contains("no_support") {
                println!("  → DSP-side 가 명시적 HTP_STATUS_NO_SUPPORT 반환 (shape/dtype 제약).");
                println!(
                    "    NPU fused FA 미지원 가설은 단순 dtype check 가 아닌 runtime path 차원."
                );
            } else {
                println!(
                    "  → dispatch path 자체에서 fail (alloc/transport/lifecycle). \
                     matrix.md narrative 는 '✗ hard, init/dispatch 실패' 로 유지."
                );
            }
        }
    }

    Ok(())
}

#[cfg(feature = "htp_fastrpc")]
#[derive(Debug, Clone, Copy)]
struct TimingStats {
    mean_us: f64,
    median_us: f64,
    stddev_us: f64,
}

#[cfg(feature = "htp_fastrpc")]
#[allow(dead_code)] // host PC 빌드에서 run_htp 가 cfg-out
fn bench<F>(warmup: usize, measure: usize, mut iter_fn: F) -> anyhow::Result<TimingStats>
where
    F: FnMut() -> anyhow::Result<f64>,
{
    for _ in 0..warmup {
        let _ = iter_fn()?;
    }
    let mut samples = Vec::with_capacity(measure);
    for _ in 0..measure {
        samples.push(iter_fn()?);
    }
    Ok(summarize(&samples))
}

#[cfg(feature = "htp_fastrpc")]
#[allow(dead_code)]
fn summarize(samples: &[f64]) -> TimingStats {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let stddev = var.sqrt();
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    TimingStats {
        mean_us: mean,
        median_us: median,
        stddev_us: stddev,
    }
}

#[cfg(feature = "htp_fastrpc")]
#[allow(clippy::too_many_arguments)]
fn run_htp(
    q_bytes: &[u8],
    k_bytes: &[u8],
    v_bytes: &[u8],
    dst_size: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv: usize,
    n_tokens: usize,
    ctx: usize,
    warmup: usize,
    measure: usize,
) -> anyhow::Result<TimingStats> {
    #[cfg(not(target_os = "android"))]
    {
        let _ = (
            q_bytes, k_bytes, v_bytes, dst_size, head_dim, n_heads, n_kv, n_tokens, ctx, warmup,
            measure,
        );
        anyhow::bail!(
            "HTP path requires target_os=android (no NPU on host PC). \
             Cross-build with --target aarch64-linux-android and deploy to S25."
        );
    }

    #[cfg(target_os = "android")]
    {
        use std::time::Instant;

        use anyhow::Context;
        use llm_rs2::backend::htp_fastrpc::{
            AEE_SUCCESS, DSPQUEUE_TIMEOUT, DspQueueBuffer, DspqBufferType, HTP_MAX_OP_PARAMS_SLOTS,
            HTP_OP_FLASH_ATTN_EXT, HTP_STATUS_OK, HTP_TYPE_F16, HTP_TYPE_F32, HtpFastrpcHost,
            HtpGeneralReq, HtpGeneralRsp, RpcmemBuffer, htp_tensor_from_shape, map_aee_err,
            map_htp_status,
        };

        let host = match HtpFastrpcHost::new("microbench_htp_flash_attn_ext") {
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

        let bytes_q = q_bytes.len();
        let bytes_k = k_bytes.len();
        let bytes_v = v_bytes.len();
        let bytes_dst = dst_size * 4; // F32 output

        let mut buf_q = RpcmemBuffer::alloc(host.clone(), bytes_q)?;
        let mut buf_k = RpcmemBuffer::alloc(host.clone(), bytes_k)?;
        let mut buf_v = RpcmemBuffer::alloc(host.clone(), bytes_v)?;
        let mut buf_dst = RpcmemBuffer::alloc(host.clone(), bytes_dst)?;

        unsafe {
            std::ptr::copy_nonoverlapping(q_bytes.as_ptr(), buf_q.as_mut_ptr(), bytes_q);
            std::ptr::copy_nonoverlapping(k_bytes.as_ptr(), buf_k.as_mut_ptr(), bytes_k);
            std::ptr::copy_nonoverlapping(v_bytes.as_ptr(), buf_v.as_mut_ptr(), bytes_v);
            std::ptr::write_bytes(buf_dst.as_mut_ptr(), 0, bytes_dst);
        }

        // ── Build tensor shapes (ggml convention, ne0 = innermost) ────────
        //
        // Q: [head_dim, n_heads, n_tokens, 1]   stride bytes (F16)
        let ne_q = [head_dim as u32, n_heads as u32, n_tokens as u32, 1];
        let nb_q = [
            2u32,
            (head_dim * 2) as u32,
            (head_dim * n_heads * 2) as u32,
            (head_dim * n_heads * n_tokens * 2) as u32,
        ];
        // K: [head_dim, ctx, n_kv, 1]    F16
        let ne_k = [head_dim as u32, ctx as u32, n_kv as u32, 1];
        let nb_k = [
            2u32,
            (head_dim * 2) as u32,
            (head_dim * ctx * 2) as u32,
            (head_dim * ctx * n_kv * 2) as u32,
        ];
        // V: same shape as K
        let ne_v = ne_k;
        let nb_v = nb_k;
        // dst: [head_dim, n_tokens, n_heads, 1] F32 (output of flash-attn 후 permute)
        let ne_dst = [head_dim as u32, n_tokens as u32, n_heads as u32, 1];
        let nb_dst = [
            4u32,
            (head_dim * 4) as u32,
            (head_dim * n_tokens * 4) as u32,
            (head_dim * n_tokens * n_heads * 4) as u32,
        ];

        let scale: f32 = 1.0_f32 / (head_dim as f32).sqrt();
        let max_bias: f32 = 0.0;
        let logit_softcap: f32 = 0.0;

        let dispatch = |measure_timing: bool| -> anyhow::Result<f64> {
            let mut req = HtpGeneralReq::zeroed();
            req.op = HTP_OP_FLASH_ATTN_EXT;
            req.flags = 0;
            req.op_params = [0; HTP_MAX_OP_PARAMS_SLOTS];
            req.op_params[0] = scale.to_bits() as i32;
            req.op_params[1] = max_bias.to_bits() as i32;
            req.op_params[2] = logit_softcap.to_bits() as i32;

            req.src0 = htp_tensor_from_shape(HTP_TYPE_F16, ne_q, nb_q);
            req.src1 = htp_tensor_from_shape(HTP_TYPE_F16, ne_k, nb_k);
            req.src2 = htp_tensor_from_shape(HTP_TYPE_F16, ne_v, nb_v);
            // src3 (mask) / src4 (sinks) 미사용 — buffer 자체 미전송.
            req.dst = htp_tensor_from_shape(HTP_TYPE_F32, ne_dst, nb_dst);

            // n_bufs=4 attempt (Q + K + V + dst). 일부 DSP-side 가 strict 6 buffer
            // 를 요구하면 NO_SUPPORT 또는 dspqueue_write fail. 결과 status 로 판정.
            let mut bufs: [DspQueueBuffer; 4] = [
                buf_q.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes_q as u32)?,
                buf_k.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes_k as u32)?,
                buf_v.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes_v as u32)?,
                buf_dst.dsp_buf(DspqBufferType::DspWriteCpuRead, 0, bytes_dst as u32)?,
            ];

            let t = Instant::now();

            let rc = unsafe {
                (host.dspqueue_write)(
                    host.queue,
                    0,
                    4,
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
            let mut rsp_bufs: [DspQueueBuffer; 6] = [DspQueueBuffer::zeroed(); 6];
            let mut rsp_msg_len: u32 = 0;
            let mut rsp_flags: u32 = 0;

            let rc = unsafe {
                (host.dspqueue_read)(
                    host.queue,
                    &mut rsp_flags as *mut u32,
                    6,
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
            if rsp.status != HTP_STATUS_OK {
                return Err(map_htp_status(rsp.status).context("DSP op status").into());
            }
            if !measure_timing {
                println!(
                    "  diag: rsp.op={}, status={}, prof_usecs={}, prof_cycles={}, prof_pkts={}, rsp_buf_count={}",
                    rsp.op,
                    rsp.status,
                    rsp.prof_usecs,
                    rsp.prof_cycles,
                    rsp.prof_pkts,
                    rsp_buf_count,
                );
                let _ = (rsp_msg_len, rsp_flags);
            }

            let elapsed_us = t.elapsed().as_secs_f64() * 1e6;
            if measure_timing {
                Ok(elapsed_us)
            } else {
                Ok(0.0)
            }
        };

        // Correctness gate (single dispatch, status only — paper goal = dispatch GREEN/FAIL)
        let _ = dispatch(false)?;

        // Timing loop
        let timing = bench(warmup, measure, || dispatch(true))?;

        Ok(timing)
    }
}
