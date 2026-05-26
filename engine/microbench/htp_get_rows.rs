//! microbench_htp_get_rows — Q-2.2 β G5 sprint (HTP_OP_GET_ROWS)
//!
//! 2-way GET_ROWS microbench: CPU baseline (host reference gather) vs QNN HTP
//! NPU (FastRPC + `libggml-htp-v79.so` dspqueue path, n_bufs=3 strict path).
//!
//! 목적: matrix.md row 17 (GET_ROWS embed [151936,1536]) 의 Ours-NPU 셀 채움 +
//! dispatch correctness GREEN gate. ROPE sprint 의 n_bufs=3 base 위에서
//! op_params 미사용 + i32 idx buffer 패턴.
//!
//! 본 sprint = correctness/dispatch GREEN 우선. embed table 은 full vocab=151936
//! 대신 dummy vocab=1024 사용 (alloc 부담 회피, 1024 × 1536 × 4 = 6 MB OK).
//! full vocab 측정은 별 sprint.
//!
//! 측정 형식:
//!   1. CPU baseline (host-side row gather reference)
//!   2. QNN HTP NPU (`engine::backend::htp_fastrpc::HtpFastrpcHost` raw dispatch)
//!
//! Per backend: warmup 10 iter + measure 200 iter (default), median + mean
//! + stddev. wall-clock only.
//!
//! Correctness:
//!   baseline = CPU reference (bit-exact row copy).
//!   max_abs_err vs CPU. threshold = 1e-3 (실제는 0.0 bit-exact 기대).
//!
//! Shape:
//!   src0 = embed [hidden=1536, vocab=1024] F32 (6,291,456 B = 6 MB)
//!   src1 = idx   [n_tokens=1] i32 (4 B, alloc 64 B min)
//!   dst  = out   [hidden=1536, n_tokens=1] F32 (6,144 B)
//!
//! Build (host):
//!   cargo build --release --features htp_fastrpc --bin microbench_htp_get_rows
//!
//! Build (Android cross):
//!   cargo build --release --features htp_fastrpc \
//!       --target aarch64-linux-android --bin microbench_htp_get_rows
//!
//! Run on device:
//!   adb -s R3CY408S5SB push target/aarch64-linux-android/release/microbench_htp_get_rows /data/local/tmp/
//!   adb -s R3CY408S5SB shell "LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64 \
//!              ADSP_LIBRARY_PATH=/data/local/tmp \
//!              /data/local/tmp/microbench_htp_get_rows"

#![allow(clippy::unnecessary_wraps)]

#[cfg(not(feature = "htp_fastrpc"))]
fn main() {
    eprintln!("microbench_htp_get_rows requires --features htp_fastrpc");
    std::process::exit(2);
}

#[cfg(feature = "htp_fastrpc")]
fn main() -> anyhow::Result<()> {
    use std::time::Instant;

    // ── Configuration (Qwen2.5-1.5B hidden=1536, dummy vocab=1024) ─────────
    const HIDDEN: usize = 1536;
    const VOCAB: usize = 1024;
    const N_TOKENS: usize = 1;
    const IDX_VALUE: i32 = 42; // 임의 nontrivial row index (< VOCAB)
    const WARMUP: usize = 10;
    const MEASURE: usize = 200;
    const ERR_THRESHOLD: f32 = 1e-3;

    let n_embed = HIDDEN * VOCAB;
    let bytes_embed = n_embed * 4;
    let bytes_idx = (N_TOKENS * 4).max(64); // llama.cpp min size 64 B
    let bytes_dst = HIDDEN * N_TOKENS * 4;

    println!("=== microbench_htp_get_rows (Q-2.2 β G5 sprint) ===\n");
    println!("Config:");
    println!(
        "  embed  shape  = [{HIDDEN}, {VOCAB}] F32 ({bytes_embed} B = {} MB, vocab=1024 dummy)",
        bytes_embed / (1024 * 1024)
    );
    println!(
        "  idx    shape  = [{N_TOKENS}] i32 (alloc {bytes_idx} B, ≥64 B min), value={IDX_VALUE}"
    );
    println!("  dst    shape  = [{HIDDEN}, {N_TOKENS}] F32 ({bytes_dst} B)");
    println!("  warmup iter   = {WARMUP}");
    println!("  measure iter  = {MEASURE}");
    println!("  err thresh    = {ERR_THRESHOLD} (expect 0.0 bit-exact)\n");

    // ── Deterministic synthetic data ──────────────────────────────────────
    let mut host_embed = vec![0.0f32; n_embed];
    for (i, x) in host_embed.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0017).rem_euclid(1.0) - 0.5;
    }
    let host_idx = vec![IDX_VALUE; N_TOKENS];

    // ── [1/2] CPU baseline (host reference row gather) ────────────────────
    println!("[1/2] CPU baseline (host-side row gather reference)");
    let cpu_baseline = get_rows_ref(&host_embed, &host_idx, HIDDEN);
    let cpu_max = cpu_baseline.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("  baseline magnitude: max|y| = {cpu_max:.6}");

    let cpu_stats = bench(WARMUP, MEASURE, || {
        let t = Instant::now();
        let _out = get_rows_ref(&host_embed, &host_idx, HIDDEN);
        Ok(t.elapsed().as_secs_f64() * 1e6)
    })?;
    println!(
        "  mean={:.2} us (median={:.2}, stddev={:.2}, n={})\n",
        cpu_stats.mean_us, cpu_stats.median_us, cpu_stats.stddev_us, MEASURE
    );

    // ── [2/2] QNN HTP NPU ─────────────────────────────────────────────────
    println!("[2/2] QNN HTP NPU (FastRPC + libggml-htp-v79.so dspqueue)");
    let htp_result = run_htp(
        &host_embed,
        &host_idx,
        &cpu_baseline,
        HIDDEN,
        VOCAB,
        N_TOKENS,
        bytes_embed,
        bytes_idx,
        bytes_dst,
        WARMUP,
        MEASURE,
    );
    let htp_stats = match htp_result {
        Ok(stats) => {
            let pass = stats.max_abs_err < ERR_THRESHOLD;
            println!(
                "  mean={:.2} us (median={:.2}, stddev={:.2}, n={})",
                stats.timing.mean_us, stats.timing.median_us, stats.timing.stddev_us, MEASURE
            );
            println!(
                "  vs CPU: max_abs_err={:.3e}, max_rel_err={:.3e} — {}\n",
                stats.max_abs_err,
                stats.max_rel_err,
                if pass { "PASS" } else { "FAIL" }
            );
            Some(stats)
        }
        Err(e) => {
            println!("  SKIP — {e:#}\n");
            None
        }
    };

    // ── Summary ───────────────────────────────────────────────────────────
    println!("=== Summary ===");
    println!(
        "CPU:      {:>8.2} us/op (1.00x baseline)",
        cpu_stats.mean_us
    );
    if let Some(s) = &htp_stats {
        let ratio = s.timing.mean_us / cpu_stats.mean_us;
        let delta_pct = (1.0 / ratio - 1.0) * 100.0;
        println!(
            "QNN HTP:  {:>8.2} us/op ({:.2}x, {:+.0}% vs CPU), err={:.2e}",
            s.timing.mean_us, ratio, delta_pct, s.max_abs_err
        );
    } else {
        println!("QNN HTP:  SKIP (Android cross-build + S25 deploy required)");
    }

    Ok(())
}

// ── Host GET_ROWS reference (bit-exact row copy) ─────────────────────────
//
// ggml `ggml_get_rows`: dst[t, :] = src0[idx[t], :].
// memory layout: src0 stride [nb0=4, nb1=HIDDEN*4] → row t = idx[t] * HIDDEN.

#[cfg(feature = "htp_fastrpc")]
fn get_rows_ref(embed: &[f32], idx: &[i32], hidden: usize) -> Vec<f32> {
    let n_tokens = idx.len();
    let mut out = vec![0.0_f32; hidden * n_tokens];
    for t in 0..n_tokens {
        let row = idx[t] as usize;
        let src = &embed[row * hidden..(row + 1) * hidden];
        out[t * hidden..(t + 1) * hidden].copy_from_slice(src);
    }
    out
}

// ── Timing helpers ────────────────────────────────────────────────────────

#[cfg(feature = "htp_fastrpc")]
#[derive(Debug, Clone, Copy)]
struct TimingStats {
    mean_us: f64,
    median_us: f64,
    stddev_us: f64,
}

#[cfg(feature = "htp_fastrpc")]
#[derive(Debug, Clone, Copy)]
struct BackendStats {
    timing: TimingStats,
    max_abs_err: f32,
    max_rel_err: f32,
}

#[cfg(feature = "htp_fastrpc")]
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
#[allow(dead_code)] // run_htp 가 host PC 빌드에서 cfg-out
fn compute_err(test: &[f32], baseline: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (t, b) in test.iter().zip(baseline.iter()) {
        let abs = (t - b).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        let denom = b.abs().max(1e-30);
        let rel = abs / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    (max_abs, max_rel)
}

// ── QNN HTP NPU backend path (Option B — direct rpcmem + dspqueue) ────────

#[cfg(feature = "htp_fastrpc")]
#[allow(clippy::too_many_arguments)]
fn run_htp(
    host_embed: &[f32],
    host_idx: &[i32],
    cpu_baseline: &[f32],
    hidden: usize,
    vocab: usize,
    n_tokens: usize,
    bytes_embed: usize,
    bytes_idx: usize,
    bytes_dst: usize,
    warmup: usize,
    measure: usize,
) -> anyhow::Result<BackendStats> {
    #[cfg(not(target_os = "android"))]
    {
        let _ = (
            host_embed,
            host_idx,
            cpu_baseline,
            hidden,
            vocab,
            n_tokens,
            bytes_embed,
            bytes_idx,
            bytes_dst,
            warmup,
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
            AEE_SUCCESS, DSPQUEUE_TIMEOUT, DspQueueBuffer, DspqBufferType, HTP_STATUS_OK,
            HTP_TYPE_F32, HTP_TYPE_I32, HtpFastrpcHost, HtpGeneralReq, HtpGeneralRsp, RpcmemBuffer,
            htp_tensor_from_shape, init_get_rows_req, map_aee_err, map_htp_status,
        };

        // 4-step FastRPC handshake.
        let host = match HtpFastrpcHost::new("microbench_htp_get_rows") {
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

        // ── Allocate rpcmem buffers ─────────────────────────────────────────
        //
        // n_bufs=3 ordering (llama.cpp init_get_rows_req strict):
        //   bufs[0] = src0 (CpuWriteDspRead, embed table)
        //   bufs[1] = src1 (CpuWriteDspRead, idx i32)
        //   bufs[2] = dst  (DspWriteCpuRead, gathered rows)
        let mut buf_embed = RpcmemBuffer::alloc(host.clone(), bytes_embed)?;
        let mut buf_idx = RpcmemBuffer::alloc(host.clone(), bytes_idx)?;
        let mut buf_dst = RpcmemBuffer::alloc(host.clone(), bytes_dst)?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                host_embed.as_ptr() as *const u8,
                buf_embed.as_mut_ptr(),
                bytes_embed,
            );
            std::ptr::write_bytes(buf_idx.as_mut_ptr(), 0, bytes_idx);
            std::ptr::copy_nonoverlapping(
                host_idx.as_ptr() as *const u8,
                buf_idx.as_mut_ptr(),
                host_idx.len() * 4,
            );
            std::ptr::write_bytes(buf_dst.as_mut_ptr(), 0, bytes_dst);
        }

        // ── Build packet ───────────────────────────────────────────────────
        // ggml convention: ne[0]=innermost.
        let ne_embed = [hidden as u32, vocab as u32, 1, 1];
        let nb_embed = [
            4u32,
            (hidden * 4) as u32,
            (hidden * vocab * 4) as u32,
            (hidden * vocab * 4) as u32,
        ];
        let ne_idx = [n_tokens as u32, 1, 1, 1];
        let nb_idx = [4u32, 4, 4, 4];
        let ne_dst = [hidden as u32, n_tokens as u32, 1, 1];
        let nb_dst = [
            4u32,
            (hidden * 4) as u32,
            (hidden * n_tokens * 4) as u32,
            (hidden * n_tokens * 4) as u32,
        ];

        let dispatch = |measure_timing: bool| -> anyhow::Result<f64> {
            let mut req = HtpGeneralReq::zeroed();
            let src0 = htp_tensor_from_shape(HTP_TYPE_F32, ne_embed, nb_embed);
            let src1 = htp_tensor_from_shape(HTP_TYPE_I32, ne_idx, nb_idx);
            let dst = htp_tensor_from_shape(HTP_TYPE_F32, ne_dst, nb_dst);
            init_get_rows_req(&mut req, src0, src1, dst);

            let mut bufs: [DspQueueBuffer; 3] = [
                buf_embed.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes_embed as u32)?,
                buf_idx.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes_idx as u32)?,
                buf_dst.dsp_buf(DspqBufferType::DspWriteCpuRead, 0, bytes_dst as u32)?,
            ];

            let t = Instant::now();

            let rc = unsafe {
                (host.dspqueue_write)(
                    host.queue,
                    0,
                    3,
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
                    rsp_buf_count
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

        // Correctness gate
        let _ = dispatch(false)?;
        let n_elem = hidden * n_tokens;
        let htp_result: Vec<f32> = unsafe {
            let p = buf_dst.as_ptr() as *const f32;
            std::slice::from_raw_parts(p, n_elem).to_vec()
        };
        let (max_abs_err, max_rel_err) = compute_err(&htp_result, cpu_baseline);

        // Timing loop
        let timing = bench(warmup, measure, || dispatch(true))?;

        Ok(BackendStats {
            timing,
            max_abs_err,
            max_rel_err,
        })
    }
}
