//! microbench_htp_softmax — Q-2.2 β G4a sprint (HTP_OP_SOFTMAX)
//!
//! 2-way SOFTMAX microbench: CPU baseline (host reference) vs QNN HTP NPU
//! (FastRPC + `libggml-htp-v79.so` dspqueue path, n_bufs=2 mask-less path).
//!
//! 목적: matrix.md row 12 (SOFTMAX [12,1,nctx] F32) 의 Ours-NPU 셀 채움 +
//! dispatch correctness GREEN gate. RMSNorm sprint 의 n_bufs=2 base 위에서
//! op_params 2 slot packing 차이 (scale + max_bias 모두 f32 bit-cast).
//!
//! 측정 형식:
//!   1. CPU baseline (host-side per-row softmax reference)
//!   2. QNN HTP NPU (`engine::backend::htp_fastrpc::HtpFastrpcHost` raw dispatch)
//!
//! Per backend: warmup 10 iter + measure 200 iter (default), median + mean
//! + stddev. wall-clock only.
//!
//! Correctness:
//!   baseline = CPU reference output.
//!   max_abs_err / max_rel_err vs CPU. threshold = 1e-3 (F32 op — exp/sum 손실 분만).
//!
//! Shape (Qwen2.5-1.5B decode attention scores, n_ctx_max=2048):
//!   src0 = scores [k_len=2048, q_len=1, n_heads=12, 1] F32 (98,304 B)
//!   dst  = output, same shape
//!
//! SoftmaxParams (Qwen2.5 standard attention):
//!   scale = 1/sqrt(128), max_bias = 0 (non-ALiBi)
//!
//! Build (host):
//!   cargo build --release --features htp_fastrpc --bin microbench_htp_softmax
//!
//! Build (Android cross):
//!   cargo build --release --features htp_fastrpc \
//!       --target aarch64-linux-android --bin microbench_htp_softmax
//!
//! Run on device:
//!   adb -s R3CY408S5SB push target/aarch64-linux-android/release/microbench_htp_softmax /data/local/tmp/
//!   adb -s R3CY408S5SB shell "LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64 \
//!              ADSP_LIBRARY_PATH=/data/local/tmp \
//!              /data/local/tmp/microbench_htp_softmax"

#![allow(clippy::unnecessary_wraps)]

#[cfg(not(feature = "htp_fastrpc"))]
fn main() {
    eprintln!("microbench_htp_softmax requires --features htp_fastrpc");
    std::process::exit(2);
}

#[cfg(feature = "htp_fastrpc")]
fn main() -> anyhow::Result<()> {
    use std::time::Instant;

    // ── Configuration (Qwen2.5-1.5B decode attention scores) ──────────────
    const K_LEN: usize = 2048; // n_ctx_max
    const Q_LEN: usize = 1; // decode
    const N_HEADS: usize = 12;
    const WARMUP: usize = 10;
    const MEASURE: usize = 200;
    const ERR_THRESHOLD: f32 = 1e-3;

    let n_elem = K_LEN * Q_LEN * N_HEADS;
    let bytes_in = n_elem * 4;
    let bytes_out = bytes_in;

    let scale = 1.0_f32 / (128.0_f32).sqrt();
    let max_bias: f32 = 0.0;

    println!("=== microbench_htp_softmax (Q-2.2 β G4a sprint) ===\n");
    println!("Config:");
    println!("  input  shape  = [{K_LEN}, {Q_LEN}, {N_HEADS}] F32 ({bytes_in} B)");
    println!("  output shape  = same as input ({bytes_out} B)");
    println!("  scale         = {scale:.8} (1/sqrt(128))");
    println!("  max_bias      = {max_bias}");
    println!("  warmup iter   = {WARMUP}");
    println!("  measure iter  = {MEASURE}");
    println!("  err thresh    = {ERR_THRESHOLD}\n");

    // ── Deterministic synthetic data (centered [-5, 5] attention-score-like) ──
    // 평탄한 입력은 softmax 가 uniform 가까워 stress 안 됨 → centered range 사용.
    let mut host_in = vec![0.0f32; n_elem];
    for (i, x) in host_in.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0173).rem_euclid(10.0) - 5.0;
    }

    // ── [1/2] CPU baseline (host reference softmax) ───────────────────────
    println!("[1/2] CPU baseline (host-side per-row softmax reference)");
    let cpu_baseline = softmax_ref(&host_in, scale, K_LEN);
    let cpu_max = cpu_baseline.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let cpu_min = cpu_baseline
        .iter()
        .map(|v| v.abs())
        .fold(f32::MAX, f32::min);
    println!("  baseline magnitude: max|y|={cpu_max:.6}, min|y|={cpu_min:.3e}");
    // sanity: 각 row sum ≈ 1.0
    let n_rows = n_elem / K_LEN;
    let mut row_sum_err: f32 = 0.0;
    for r in 0..n_rows {
        let s: f32 = cpu_baseline[r * K_LEN..(r + 1) * K_LEN].iter().sum();
        let err = (s - 1.0).abs();
        if err > row_sum_err {
            row_sum_err = err;
        }
    }
    println!("  row sum |Σy-1| max: {row_sum_err:.3e} (should be ≤1e-5)");

    let cpu_stats = bench(WARMUP, MEASURE, || {
        let t = Instant::now();
        let _out = softmax_ref(&host_in, scale, K_LEN);
        Ok(t.elapsed().as_secs_f64() * 1e6)
    })?;
    println!(
        "  mean={:.2} us (median={:.2}, stddev={:.2}, n={})\n",
        cpu_stats.mean_us, cpu_stats.median_us, cpu_stats.stddev_us, MEASURE
    );

    // ── [2/2] QNN HTP NPU ─────────────────────────────────────────────────
    println!("[2/2] QNN HTP NPU (FastRPC + libggml-htp-v79.so dspqueue)");
    let htp_result = run_htp(
        &host_in,
        &cpu_baseline,
        K_LEN,
        Q_LEN,
        N_HEADS,
        bytes_in,
        bytes_out,
        scale,
        max_bias,
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

// ── Host softmax reference (per-row, scale 적용 후 numerically stable) ─────
//
// ggml `ggml_compute_forward_soft_max_f32` 동치:
//   1. scaled = x * scale (+ mask, ALiBi 미사용)
//   2. max  = max(scaled[row])
//   3. exp  = exp(scaled[row] - max)
//   4. sum  = Σ exp
//   5. y    = exp / sum
//
// row 단위 = ne[0] (innermost). 본 sprint shape [K_LEN, Q_LEN, N_HEADS] 에서
// row size = K_LEN, n_rows = Q_LEN * N_HEADS.

#[cfg(feature = "htp_fastrpc")]
fn softmax_ref(input: &[f32], scale: f32, k_len: usize) -> Vec<f32> {
    let n_rows = input.len() / k_len;
    let mut out = vec![0.0_f32; input.len()];
    for r in 0..n_rows {
        let row = &input[r * k_len..(r + 1) * k_len];
        // max(scaled) for numerical stability.
        let mut max = f32::NEG_INFINITY;
        for &x in row {
            let sx = x * scale;
            if sx > max {
                max = sx;
            }
        }
        let mut sum: f32 = 0.0;
        let dst = &mut out[r * k_len..(r + 1) * k_len];
        for (i, &x) in row.iter().enumerate() {
            let e = (x * scale - max).exp();
            dst[i] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for y in dst.iter_mut() {
            *y *= inv;
        }
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
    host_in: &[f32],
    cpu_baseline: &[f32],
    k_len: usize,
    q_len: usize,
    n_heads: usize,
    bytes_in: usize,
    bytes_out: usize,
    scale: f32,
    max_bias: f32,
    warmup: usize,
    measure: usize,
) -> anyhow::Result<BackendStats> {
    #[cfg(not(target_os = "android"))]
    {
        let _ = (
            host_in,
            cpu_baseline,
            k_len,
            q_len,
            n_heads,
            bytes_in,
            bytes_out,
            scale,
            max_bias,
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
            HTP_TYPE_F32, HtpFastrpcHost, HtpGeneralReq, HtpGeneralRsp, RpcmemBuffer,
            SoftmaxParams, htp_tensor_from_shape, init_softmax_req, map_aee_err, map_htp_status,
        };

        // 4-step FastRPC handshake.
        let host = match HtpFastrpcHost::new("microbench_htp_softmax") {
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
        // n_bufs=2 ordering (mask-less softmax):
        //   bufs[0] = src0 (CpuWriteDspRead, host activation scores)
        //   bufs[1] = dst  (DspWriteCpuRead, output)
        let mut buf_in = RpcmemBuffer::alloc(host.clone(), bytes_in)?;
        let mut buf_out = RpcmemBuffer::alloc(host.clone(), bytes_out)?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                host_in.as_ptr() as *const u8,
                buf_in.as_mut_ptr(),
                bytes_in,
            );
            std::ptr::write_bytes(buf_out.as_mut_ptr(), 0, bytes_out);
        }

        // ── Build packet ───────────────────────────────────────────────────
        // ggml convention: ne[0]=innermost (k_len).
        let ne = [k_len as u32, q_len as u32, n_heads as u32, 1];
        let nb = [
            4u32,
            (k_len * 4) as u32,
            (k_len * q_len * 4) as u32,
            (k_len * q_len * n_heads * 4) as u32,
        ];

        let params = SoftmaxParams { scale, max_bias };

        let dispatch = |measure_timing: bool| -> anyhow::Result<f64> {
            let mut req = HtpGeneralReq::zeroed();
            let src0 = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
            let dst = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
            init_softmax_req(&mut req, &params, src0, dst);

            let mut bufs: [DspQueueBuffer; 2] = [
                buf_in.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes_in as u32)?,
                buf_out.dsp_buf(DspqBufferType::DspWriteCpuRead, 0, bytes_out as u32)?,
            ];

            let t = Instant::now();

            let rc = unsafe {
                (host.dspqueue_write)(
                    host.queue,
                    0,
                    2, // num_buffers (input + output, n_bufs=2 mask-less)
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
            let mut rsp_bufs: [DspQueueBuffer; 3] = [DspQueueBuffer::zeroed(); 3];
            let mut rsp_msg_len: u32 = 0;
            let mut rsp_flags: u32 = 0;

            let rc = unsafe {
                (host.dspqueue_read)(
                    host.queue,
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
        let n_elem = k_len * q_len * n_heads;
        let htp_result: Vec<f32> = unsafe {
            let p = buf_out.as_ptr() as *const f32;
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
