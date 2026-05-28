//! microbench_htp_mul_f16 — P1d Senior Implementer (Full matrix sprint)
//!
//! HTP NPU MUL F16 dispatch. DSP-side (`libggml-htp-v79.so` `binary-ops.c:133,
//! 154, 175`) 는 `hvx_mul_f16_aaa` / `hvx_mul_f16_aau` / `hvx_mul_f16_uuu`
//! intrinsic path 를 갖는다.
//!
//! Shape (Qwen 2.5-1.5B SwiGLU gate · up elementwise):
//!   src0   gate[8960] F16 (17,920 B)
//!   src1   up[8960]   F16 (17,920 B)
//!   dst    z[8960]    F16 (17,920 B)
//!
//! Per backend: warmup 10 + measure 1000 iter.
//! Threshold: F16 round-trip + product ~1e-3 magnitude → 5e-3 absolute.

#![allow(clippy::unnecessary_wraps)]

#[cfg(not(feature = "htp_fastrpc"))]
fn main() {
    eprintln!("microbench_htp_mul_f16 requires --features htp_fastrpc");
    std::process::exit(2);
}

#[cfg(feature = "htp_fastrpc")]
fn main() -> anyhow::Result<()> {
    use std::time::Instant;

    use half::f16;

    const DIM: usize = 8960;
    const WARMUP: usize = 10;
    const MEASURE: usize = 1000;
    const ERR_THRESHOLD: f32 = 5e-3;

    let input_bytes_f16 = DIM * 2;

    println!("=== microbench_htp_mul_f16 (P1d sprint) ===\n");
    println!("Config:");
    println!("  shape         = [1, {DIM}] F16 ({input_bytes_f16} B per buf)");
    println!("  warmup iter   = {WARMUP}");
    println!("  measure iter  = {MEASURE}");
    println!("  err thresh    = {ERR_THRESHOLD}\n");

    let mut host_a_f32 = vec![0.0f32; DIM];
    let mut host_b_f32 = vec![0.0f32; DIM];
    for (i, x) in host_a_f32.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5;
    }
    for (i, x) in host_b_f32.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0291 + 0.13).rem_euclid(1.0) - 0.5;
    }

    let host_a_f16: Vec<f16> = host_a_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_b_f16: Vec<f16> = host_b_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let a_f16_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(host_a_f16.as_ptr() as *const u8, host_a_f16.len() * 2)
    };
    let b_f16_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(host_b_f16.as_ptr() as *const u8, host_b_f16.len() * 2)
    };

    println!("[1/2] CPU baseline (F32 elementwise mul on F16-rt values)");

    // F16 round-trip values 로 baseline 계산
    let host_a_rt: Vec<f32> = host_a_f16.iter().map(|h| h.to_f32()).collect();
    let host_b_rt: Vec<f32> = host_b_f16.iter().map(|h| h.to_f32()).collect();

    // CPU baseline: elementwise z[i] = a[i] * b[i].
    // Backend trait 에 pure elementwise mul 가 없어 (silu_mul 은 SiLU(a)*b)
    // host-side scalar loop 사용. NPU 와 동일 결과 비교용 reference.
    let cpu_baseline: Vec<f32> = host_a_rt
        .iter()
        .zip(host_b_rt.iter())
        .map(|(a, b)| a * b)
        .collect();
    let cpu_max = cpu_baseline.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("  baseline magnitude: max|y| = {cpu_max:.4}");

    let cpu_stats = bench(WARMUP, MEASURE, || {
        let t = Instant::now();
        let mut out = vec![0.0f32; DIM];
        for i in 0..DIM {
            out[i] = host_a_rt[i] * host_b_rt[i];
        }
        std::hint::black_box(out);
        Ok(t.elapsed().as_secs_f64() * 1e6)
    })?;
    println!(
        "  mean={:.2} us (median={:.2}, stddev={:.2}, n={})\n",
        cpu_stats.mean_us, cpu_stats.median_us, cpu_stats.stddev_us, MEASURE
    );

    println!("[2/2] QNN HTP NPU (F16 native, FastRPC + libggml-htp-v79.so)");
    let htp_result = run_htp(
        a_f16_bytes,
        b_f16_bytes,
        &cpu_baseline,
        DIM,
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
#[allow(dead_code)]
fn compute_err_f16_dst(test_f16_bytes: &[u8], baseline_f32: &[f32], n: usize) -> (f32, f32) {
    use half::f16;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let test_f16: &[f16] =
        unsafe { std::slice::from_raw_parts(test_f16_bytes.as_ptr() as *const f16, n) };
    for (t_h, b) in test_f16.iter().zip(baseline_f32.iter()) {
        let t = t_h.to_f32();
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

#[cfg(feature = "htp_fastrpc")]
fn run_htp(
    a_f16_bytes: &[u8],
    b_f16_bytes: &[u8],
    cpu_baseline: &[f32],
    dim: usize,
    warmup: usize,
    measure: usize,
) -> anyhow::Result<BackendStats> {
    #[cfg(not(target_os = "android"))]
    {
        let _ = (a_f16_bytes, b_f16_bytes, cpu_baseline, dim, warmup, measure);
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
            AEE_SUCCESS, DSPQUEUE_TIMEOUT, DspQueueBuffer, DspqBufferType, HTP_OP_MUL,
            HTP_STATUS_OK, HTP_TYPE_F16, HtpFastrpcHost, HtpGeneralReq, HtpGeneralRsp,
            RpcmemBuffer, htp_tensor_from_shape, init_binary_req, map_aee_err, map_htp_status,
        };

        let host = match HtpFastrpcHost::new("microbench_htp_mul_f16") {
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

        let mut buf_a = RpcmemBuffer::alloc(host.clone(), bytes)?;
        let mut buf_b = RpcmemBuffer::alloc(host.clone(), bytes)?;
        let mut buf_y = RpcmemBuffer::alloc(host.clone(), bytes)?;

        unsafe {
            std::ptr::copy_nonoverlapping(a_f16_bytes.as_ptr(), buf_a.as_mut_ptr(), bytes);
            std::ptr::copy_nonoverlapping(b_f16_bytes.as_ptr(), buf_b.as_mut_ptr(), bytes);
            std::ptr::write_bytes(buf_y.as_mut_ptr(), 0, bytes);
        }

        let ne = [dim as u32, 1, 1, 1];
        let nb = [2u32, (dim * 2) as u32, (dim * 2) as u32, (dim * 2) as u32];

        let dispatch = |measure_timing: bool| -> anyhow::Result<f64> {
            let mut req = HtpGeneralReq::zeroed();
            let src0 = htp_tensor_from_shape(HTP_TYPE_F16, ne, nb);
            let src1 = htp_tensor_from_shape(HTP_TYPE_F16, ne, nb);
            let dst = htp_tensor_from_shape(HTP_TYPE_F16, ne, nb);
            init_binary_req(&mut req, HTP_OP_MUL, src0, src1, dst);

            let mut bufs: [DspQueueBuffer; 3] = [
                buf_a.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes as u32)?,
                buf_b.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes as u32)?,
                buf_y.dsp_buf(DspqBufferType::DspWriteCpuRead, 0, bytes as u32)?,
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
                    "  diag (F16): rsp.op={}, status={}, prof_usecs={}, prof_cycles={}, prof_pkts={}, rsp_buf_count={}",
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

        let _ = dispatch(false)?;
        let htp_f16_bytes: Vec<u8> =
            unsafe { std::slice::from_raw_parts(buf_y.as_ptr(), bytes).to_vec() };
        let (max_abs_err, max_rel_err) = compute_err_f16_dst(&htp_f16_bytes, cpu_baseline, dim);

        let timing = bench(warmup, measure, || dispatch(true))?;

        Ok(BackendStats {
            timing,
            max_abs_err,
            max_rel_err,
        })
    }
}
