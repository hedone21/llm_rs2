//! microbench_htp_add — Q-2.2 β G1 sprint (HTP_OP_ADD)
//!
//! 2-way ADD microbench: CPU baseline (NEON `add_assign`) vs QNN HTP NPU
//! (FastRPC + `libggml-htp-v79.so` dspqueue path, n_bufs=3 binary_req).
//!
//! 목적: matrix.md row 1 (ADD `[1, 1536]` residual, Qwen2.5-1.5B) 의 Ours-NPU
//! 셀 채움 + dispatch correctness GREEN gate. RMSNORM/MUL_MAT sprint 의 host
//! binding + DspQueueBuffer 24B layout fix (`dde08248`) 위에서 n_bufs=3 path
//! 만 차이 (constant weight 없는 element-wise variant).
//!
//! 측정 형식:
//!   1. CPU baseline (`engine::backend::cpu::cpu_singleton`) — NEON F32 add
//!   2. QNN HTP NPU (`engine::backend::htp_fastrpc::HtpFastrpcBackend`)
//!
//! Per backend: warmup 10 iter + measure 1000 iter (default), median + mean
//! + stddev. wall-clock only.
//!
//! Correctness:
//!   baseline = CPU output.
//!   max_abs_err / max_rel_err vs CPU. threshold = 1e-3 (F32-only path).
//!
//! Shape (Qwen2.5-1.5B residual add):
//!   src0   x[1536] F32 (6,144 B)
//!   src1   y[1536] F32 (6,144 B)
//!   dst    z[1536] F32 (6,144 B)
//!
//! 데이터 시드: deterministic (rmsnorm 패턴 차용).
//!
//! Build (host):
//!   cargo build --release --features htp_fastrpc --bin microbench_htp_add
//!
//! Build (Android cross):
//!   cargo build --release --features htp_fastrpc \
//!       --target aarch64-linux-android --bin microbench_htp_add
//!
//! Run on device:
//!   adb -s R3CY408S5SB push target/aarch64-linux-android/release/microbench_htp_add /data/local/tmp/
//!   adb -s R3CY408S5SB shell "LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64 \
//!              ADSP_LIBRARY_PATH=/data/local/tmp \
//!              /data/local/tmp/microbench_htp_add"

#![allow(clippy::unnecessary_wraps)]

#[cfg(not(feature = "htp_fastrpc"))]
fn main() {
    eprintln!("microbench_htp_add requires --features htp_fastrpc");
    std::process::exit(2);
}

#[cfg(feature = "htp_fastrpc")]
fn main() -> anyhow::Result<()> {
    use std::sync::Arc;
    use std::time::Instant;

    use llm_rs2::backend::Backend;
    use llm_rs2::buffer::{Buffer, DType};
    use llm_rs2::memory::host::shared::SharedBuffer;
    use llm_rs2::shape::Shape;
    use llm_rs2::tensor::Tensor;

    // ── Configuration (Qwen2.5-1.5B residual add) ─────────────────────────
    const DIM: usize = 1536;
    const WARMUP: usize = 10;
    const MEASURE: usize = 1000;
    const ERR_THRESHOLD: f32 = 1e-3;

    let input_bytes = DIM * 4;

    println!("=== microbench_htp_add (Q-2.2 β G1 sprint) ===\n");
    println!("Config:");
    println!("  shape         = [1, {DIM}] F32 ({input_bytes} B per buf)");
    println!("  warmup iter   = {WARMUP}");
    println!("  measure iter  = {MEASURE}");
    println!("  err thresh    = {ERR_THRESHOLD}\n");

    // ── Deterministic synthetic data ──────────────────────────────────────
    let mut host_a = vec![0.0f32; DIM];
    let mut host_b = vec![0.0f32; DIM];
    for (i, x) in host_a.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5;
    }
    for (i, x) in host_b.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0291 + 0.13).rem_euclid(1.0) - 0.5;
    }

    // ── [1/2] CPU baseline ────────────────────────────────────────────────
    println!("[1/2] CPU baseline (NEON F32 add_assign)");
    let cpu_backend: Arc<dyn Backend> = llm_rs2::backend::cpu::cpu_singleton();

    let mk_f32_tensor = |data: &[f32], shape: Vec<usize>| -> Tensor {
        let buf = SharedBuffer::new(data.len() * 4, DType::F32);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buf.as_mut_ptr(),
                data.len() * 4,
            );
        }
        Tensor::new(Shape::new(shape), Arc::new(buf), cpu_backend.clone())
    };

    // CPU baseline: a += b
    let mut a_cpu = mk_f32_tensor(&host_a, vec![1, DIM]);
    let b_cpu = mk_f32_tensor(&host_b, vec![1, DIM]);
    cpu_backend.add_assign(&mut a_cpu, &b_cpu)?;
    let cpu_baseline: Vec<f32> = a_cpu.as_slice::<f32>().to_vec();
    let cpu_max = cpu_baseline.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("  baseline magnitude: max|y| = {cpu_max:.4}");

    let cpu_stats = bench(WARMUP, MEASURE, || {
        let mut a = mk_f32_tensor(&host_a, vec![1, DIM]);
        let t = Instant::now();
        cpu_backend.add_assign(&mut a, &b_cpu)?;
        Ok(t.elapsed().as_secs_f64() * 1e6)
    })?;
    println!(
        "  mean={:.2} us (median={:.2}, stddev={:.2}, n={})\n",
        cpu_stats.mean_us, cpu_stats.median_us, cpu_stats.stddev_us, MEASURE
    );

    // ── [2/2] QNN HTP NPU ─────────────────────────────────────────────────
    println!("[2/2] QNN HTP NPU (FastRPC + libggml-htp-v79.so dspqueue)");
    let htp_result = run_htp(&host_a, &host_b, &cpu_baseline, DIM, WARMUP, MEASURE);
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
fn run_htp(
    host_a: &[f32],
    host_b: &[f32],
    cpu_baseline: &[f32],
    dim: usize,
    warmup: usize,
    measure: usize,
) -> anyhow::Result<BackendStats> {
    #[cfg(not(target_os = "android"))]
    {
        let _ = (host_a, host_b, cpu_baseline, dim, warmup, measure);
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
            AEE_SUCCESS, DSPQUEUE_TIMEOUT, DspQueueBuffer, DspqBufferType, HTP_OP_ADD,
            HTP_STATUS_OK, HTP_TYPE_F32, HtpFastrpcHost, HtpGeneralReq, HtpGeneralRsp,
            RpcmemBuffer, htp_tensor_from_shape, init_binary_req, map_aee_err, map_htp_status,
        };

        // 4-step FastRPC handshake.
        let host = match HtpFastrpcHost::new("microbench_htp_add") {
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
        // n_bufs=3 ordering (llama.cpp init_binary_req<false>):
        //   bufs[0] = src0 (CpuWriteDspRead, host activation)
        //   bufs[1] = src1 (CpuWriteDspRead, host activation)
        //   bufs[2] = dst  (DspWriteCpuRead)
        let bytes = dim * 4;

        let mut buf_a = RpcmemBuffer::alloc(host.clone(), bytes, llm_rs2::buffer::DType::F32)?;
        let mut buf_b = RpcmemBuffer::alloc(host.clone(), bytes, llm_rs2::buffer::DType::F32)?;
        let mut buf_y = RpcmemBuffer::alloc(host.clone(), bytes, llm_rs2::buffer::DType::F32)?;

        unsafe {
            std::ptr::copy_nonoverlapping(host_a.as_ptr() as *const u8, buf_a.as_mut_ptr(), bytes);
            std::ptr::copy_nonoverlapping(host_b.as_ptr() as *const u8, buf_b.as_mut_ptr(), bytes);
            std::ptr::write_bytes(buf_y.as_mut_ptr(), 0, bytes);
        }

        // ── Build packet ───────────────────────────────────────────────────
        // F32 vector [dim] — ne = [dim, 1, 1, 1], element stride = 4 B.
        let ne = [dim as u32, 1, 1, 1];
        let nb = [4u32, (dim * 4) as u32, (dim * 4) as u32, (dim * 4) as u32];

        let dispatch = |measure_timing: bool| -> anyhow::Result<f64> {
            let mut req = HtpGeneralReq::zeroed();
            let src0 = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
            let src1 = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
            let dst = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
            init_binary_req(&mut req, HTP_OP_ADD, src0, src1, dst);

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
        let htp_result: Vec<f32> = unsafe {
            let p = buf_y.as_ptr() as *const f32;
            std::slice::from_raw_parts(p, dim).to_vec()
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
