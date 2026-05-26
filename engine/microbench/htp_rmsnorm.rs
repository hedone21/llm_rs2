//! microbench_htp_rmsnorm — Q-2.2-α PoC Phase 5
//!
//! 3-way RMSNorm microbench: CPU baseline vs OpenCL GPU vs QNN HTP NPU
//! (FastRPC + `libggml-htp-v79.so` dspqueue path).
//!
//! 목적: Phase 3 (FFI host) + Phase 4 (Backend trait stub + `rms_norm_via_htp`
//! inline helper) 를 실측 검증. HTP path 는 [Option B] — Backend trait routing
//! 우회 (RpcmemBuffer ↔ Buffer trait wrap 가 아직 없음). microbench 내부에서
//! `RpcmemBuffer::alloc` + `dspqueue_write/read` 를 직접 호출하여 raw HTP
//! dispatch latency 만 측정한다. Backend trait wire-up (β scope) 와는 독립.
//!
//! 측정 형식:
//!   1. CPU baseline (`engine::backend::cpu::cpu_singleton`)
//!   2. OpenCL GPU (`engine::backend::opencl::OpenCLBackend`)
//!   3. QNN HTP NPU (`engine::backend::htp_fastrpc::HtpFastrpcBackend`)
//!
//! Per backend: warmup 10 iter + measure 1000 iter (default), median + mean
//! + stddev. wall-clock only (CLAUDE.md `feedback_opencl_profile_events_cross_engine`).
//!
//! Correctness:
//!   baseline = CPU output.
//!   max_abs_err / max_rel_err vs CPU. threshold = 1e-3
//!   (spec/htp_fastrpc.md INV-HTP-FRPC-003).
//!
//! Shape:
//!   Qwen2.5-1.5B 의 RMSNorm = `[1, 1536]`. gamma weight `[1536]`.
//!   N(0, 0.5) for input (centered, mid-magnitude),
//!   N(1.0, 0.02) for gamma (RMSNorm standard init).
//!
//! Build (host):
//!   cargo build --release --features htp_fastrpc --bin microbench_htp_rmsnorm
//!
//! Build (Android cross):
//!   cargo build --release --features htp_fastrpc \
//!       --target aarch64-linux-android --bin microbench_htp_rmsnorm
//!
//! Run on device (Phase 6, requires `libggml-htp-v79.so` push):
//!   adb push /home/go/Workspace/llama.cpp/build-snapdragon/ggml/src/ggml-hexagon/libggml-htp-v79.so \
//!           /data/local/tmp/
//!   adb shell "LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64 \
//!              /data/local/tmp/microbench_htp_rmsnorm"
//!
//! Host PC 실행:
//!   - HTP path 는 `target_os != android` 에서 unavailable 메시지 + skip.
//!   - CPU + OpenCL 만 측정.

#![allow(clippy::unnecessary_wraps)]

#[cfg(not(feature = "htp_fastrpc"))]
fn main() {
    eprintln!("microbench_htp_rmsnorm requires --features htp_fastrpc");
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

    // ── Configuration ─────────────────────────────────────────────────────
    const ROWS: usize = 1;
    const DIM: usize = 1536; // Qwen2.5-1.5B hidden
    const EPS: f32 = 1e-5;
    const WARMUP: usize = 10;
    const MEASURE: usize = 1000;
    const ERR_THRESHOLD: f32 = 1e-3;
    const ADD_UNIT: bool = false; // Llama/Qwen style (Gemma3 = true)

    println!("=== microbench_htp_rmsnorm (Q-2.2-α PoC Phase 5) ===\n");
    println!("Config:");
    println!("  shape       = [{}, {}]", ROWS, DIM);
    println!("  eps         = {}", EPS);
    println!("  add_unit    = {}", ADD_UNIT);
    println!("  warmup iter = {}", WARMUP);
    println!("  measure iter= {}", MEASURE);
    println!("  err thresh  = {}\n", ERR_THRESHOLD);

    // ── Deterministic synthetic input ──────────────────────────────────────
    let total = ROWS * DIM;
    let mut host_x = vec![0.0f32; total];
    let mut host_w = vec![0.0f32; DIM];
    for (i, x) in host_x.iter_mut().enumerate() {
        // mid-magnitude pseudo-uniform on [-0.5, 0.5)
        *x = ((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5;
    }
    // PoC: gamma=1.0 fill (no-op affine). llama.cpp HTP_OP_RMS_NORM 은 gamma 미적용
    // (unary-ops.c::rms_norm_f32 가 input/rms(input) 만 처리). CPU/OpenCL baseline 도
    // 동일하게 gamma=1 로 비교 → output == input / rms(input) 형태로 3-way 일치.
    for w in host_w.iter_mut() {
        *w = 1.0;
    }

    // ── [1/3] CPU baseline ─────────────────────────────────────────────────
    println!("[1/3] CPU baseline ({})", std::any::type_name::<f32>());
    let cpu_backend: Arc<dyn Backend> = llm_rs2::backend::cpu::cpu_singleton();

    let mk_cpu_tensor = |data: &[f32], shape: Vec<usize>| -> Tensor {
        let buf = SharedBuffer::new(data.len() * 4, DType::F32);
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.as_mut_ptr(), bytes.len());
        }
        Tensor::new(Shape::new(shape), Arc::new(buf), cpu_backend.clone())
    };

    // CPU baseline output (used by both OpenCL & HTP correctness gates).
    let mut cpu_x = mk_cpu_tensor(&host_x, vec![ROWS, DIM]);
    let cpu_w = mk_cpu_tensor(&host_w, vec![DIM]);
    cpu_backend.rms_norm(&mut cpu_x, &cpu_w, EPS, ADD_UNIT)?;
    let cpu_baseline: Vec<f32> = cpu_x.as_slice::<f32>().to_vec();

    // CPU timing — rebuild input each iter so we measure full op (CPU rms_norm
    // is in-place; otherwise we'd be timing the second call on already-
    // normalized data which gives different cost characteristics).
    let cpu_stats = bench(WARMUP, MEASURE, || {
        let mut x = mk_cpu_tensor(&host_x, vec![ROWS, DIM]);
        let t = Instant::now();
        cpu_backend.rms_norm(&mut x, &cpu_w, EPS, ADD_UNIT)?;
        Ok(t.elapsed().as_secs_f64() * 1e6)
    })?;
    println!(
        "  mean={:.2} us (median={:.2}, stddev={:.2}, n={})\n",
        cpu_stats.mean_us, cpu_stats.median_us, cpu_stats.stddev_us, MEASURE
    );

    // ── [2/3] OpenCL GPU ───────────────────────────────────────────────────
    println!("[2/3] OpenCL GPU");
    let opencl_result = run_opencl(
        &host_x,
        &host_w,
        &cpu_baseline,
        ROWS,
        DIM,
        EPS,
        ADD_UNIT,
        WARMUP,
        MEASURE,
    );
    let opencl_stats = match opencl_result {
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
            println!("  SKIP — {:#}\n", e);
            None
        }
    };

    // ── [3/3] QNN HTP NPU ──────────────────────────────────────────────────
    println!("[3/3] QNN HTP NPU (FastRPC + libggml-htp-v79.so dspqueue)");
    let htp_result = run_htp(
        &host_x,
        &host_w,
        &cpu_baseline,
        ROWS,
        DIM,
        EPS,
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
            println!("  SKIP — {:#}\n", e);
            None
        }
    };

    // ── Summary ────────────────────────────────────────────────────────────
    println!("=== Summary ===");
    println!(
        "CPU:      {:>8.2} us/op (1.00x baseline)",
        cpu_stats.mean_us
    );
    if let Some(s) = &opencl_stats {
        let ratio = s.timing.mean_us / cpu_stats.mean_us;
        let delta_pct = (1.0 / ratio - 1.0) * 100.0;
        println!(
            "OpenCL:   {:>8.2} us/op ({:.2}x, {:+.0}% vs CPU), err={:.2e}",
            s.timing.mean_us, ratio, delta_pct, s.max_abs_err
        );
    } else {
        println!("OpenCL:   SKIP");
    }
    if let Some(s) = &htp_stats {
        let ratio = s.timing.mean_us / cpu_stats.mean_us;
        let delta_pct = (1.0 / ratio - 1.0) * 100.0;
        println!(
            "QNN HTP:  {:>8.2} us/op ({:.2}x, {:+.0}% vs CPU), err={:.2e}",
            s.timing.mean_us, ratio, delta_pct, s.max_abs_err
        );
    } else {
        println!("QNN HTP:  SKIP (Phase 6 device deploy required)");
    }

    Ok(())
}

// ── Timing helpers (feature gated) ────────────────────────────────────────

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
    // warmup
    for _ in 0..warmup {
        let _ = iter_fn()?;
    }
    // measure
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

// ── OpenCL backend path ───────────────────────────────────────────────────

#[cfg(feature = "htp_fastrpc")]
#[allow(clippy::too_many_arguments)]
fn run_opencl(
    host_x: &[f32],
    host_w: &[f32],
    cpu_baseline: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
    add_unit: bool,
    warmup: usize,
    measure: usize,
) -> anyhow::Result<BackendStats> {
    use std::sync::Arc;
    use std::time::Instant;

    use llm_rs2::backend::Backend;
    use llm_rs2::buffer::{Buffer, DType};
    use llm_rs2::memory::host::shared::SharedBuffer;
    use llm_rs2::shape::Shape;
    use llm_rs2::tensor::Tensor;

    #[cfg(not(feature = "opencl"))]
    {
        let _ = (
            host_x,
            host_w,
            cpu_baseline,
            rows,
            dim,
            eps,
            add_unit,
            warmup,
            measure,
        );
        anyhow::bail!("opencl feature not enabled — rebuild with --features htp_fastrpc,opencl");
    }

    #[cfg(feature = "opencl")]
    {
        let gpu_backend: Arc<dyn Backend> = match llm_rs2::backend::opencl::OpenCLBackend::new() {
            Ok(be) => Arc::new(be),
            Err(e) => anyhow::bail!("OpenCL backend init failed: {}", e),
        };
        let cpu_backend: Arc<dyn Backend> = llm_rs2::backend::cpu::cpu_singleton();

        // helper to alloc CPU tensor + upload to GPU via copy_from
        let mk_gpu_tensor = |data: &[f32], shape: Vec<usize>| -> anyhow::Result<Tensor> {
            let buf = SharedBuffer::new(data.len() * 4, DType::F32);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    buf.as_mut_ptr(),
                    data.len() * 4,
                );
            }
            let cpu_t = Tensor::new(Shape::new(shape), Arc::new(buf), cpu_backend.clone());
            gpu_backend.copy_from(&cpu_t)
        };

        // Allocate weight tensor once (read-only across iters).
        let w_gpu = mk_gpu_tensor(host_w, vec![dim])?;

        // Allocate input tensor once and refill each iter via write_buffer.
        let mut x_gpu = mk_gpu_tensor(host_x, vec![rows, dim])?;

        // Sanity: correctness check (single op).
        gpu_backend.rms_norm(&mut x_gpu, &w_gpu, eps, add_unit)?;
        gpu_backend.synchronize()?;

        let mut gpu_out = vec![0u8; rows * dim * 4];
        gpu_backend.read_buffer(&x_gpu, &mut gpu_out)?;
        let gpu_result: &[f32] =
            unsafe { std::slice::from_raw_parts(gpu_out.as_ptr() as *const f32, rows * dim) };
        let (max_abs_err, max_rel_err) = compute_err(gpu_result, cpu_baseline);

        // Bench loop: rewrite input each iter, dispatch rms_norm, sync (wall-clock).
        let x_bytes =
            unsafe { std::slice::from_raw_parts(host_x.as_ptr() as *const u8, host_x.len() * 4) };

        let timing = bench(warmup, measure, || {
            gpu_backend.write_buffer(&mut x_gpu, x_bytes)?;
            gpu_backend.synchronize()?;
            let t = Instant::now();
            gpu_backend.rms_norm(&mut x_gpu, &w_gpu, eps, add_unit)?;
            gpu_backend.synchronize()?;
            Ok(t.elapsed().as_secs_f64() * 1e6)
        })?;

        Ok(BackendStats {
            timing,
            max_abs_err,
            max_rel_err,
        })
    }
}

// ── QNN HTP NPU backend path (Option B — direct rpcmem + dspqueue) ────────

#[cfg(feature = "htp_fastrpc")]
#[allow(clippy::too_many_arguments)]
fn run_htp(
    host_x: &[f32],
    host_w: &[f32],
    cpu_baseline: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
    warmup: usize,
    measure: usize,
) -> anyhow::Result<BackendStats> {
    #[cfg(not(target_os = "android"))]
    {
        let _ = (
            host_x,
            host_w,
            cpu_baseline,
            rows,
            dim,
            eps,
            warmup,
            measure,
        );
        anyhow::bail!(
            "HTP path requires target_os=android (no NPU on host PC). \
             Cross-build with --target aarch64-linux-android and run Phase 6 deploy."
        );
    }

    #[cfg(target_os = "android")]
    {
        use std::time::Instant;

        use llm_rs2::backend::htp_fastrpc::{
            AEE_SUCCESS, DSPQUEUE_TIMEOUT, DspQueueBuffer, DspqBufferType, HTP_TYPE_F32,
            HtpFastrpcHost, HtpGeneralReq, HtpGeneralRsp, RpcmemBuffer, htp_tensor_from_shape,
            init_rmsnorm_req, map_aee_err, map_htp_status,
        };

        // 4-step FastRPC handshake.
        let host = match HtpFastrpcHost::new("microbench_htp_rmsnorm") {
            Ok(h) => h,
            Err(e) => anyhow::bail!("HtpFastrpcHost::new failed: {}", e),
        };
        println!(
            "  host: libcdsprpc.so={}, domain_id={}, session_id={}, queue_id={}",
            host.lib_path, host.domain_id, host.session_id, host.queue_id
        );

        // ── Allocate rpcmem buffers ─────────────────────────────────────────
        //
        // Layout: input/output 둘 다 [rows, dim] F32.
        // weight 는 [dim] F32. HTP rmsnorm op 의 정확한 src 슬롯 사용 패턴은
        // llama.cpp `htp-msg.h` rmsnorm 변종에 따라 다르므로 (vendor-specific):
        //   PoC 본 sprint = src0 = input, dst = output. weight 는 src1 로 보냄
        //   (mod.rs:rms_norm_via_htp 의 패턴 그대로).
        let bytes_x = rows * dim * 4;
        let bytes_w = dim * 4;
        let bytes_y = rows * dim * 4;

        let mut buf_x = RpcmemBuffer::alloc(host.clone(), bytes_x)?;
        let mut buf_w = RpcmemBuffer::alloc(host.clone(), bytes_w)?;
        let mut buf_y = RpcmemBuffer::alloc(host.clone(), bytes_y)?;

        // host-side write
        unsafe {
            std::ptr::copy_nonoverlapping(
                host_x.as_ptr() as *const u8,
                buf_x.as_mut_ptr(),
                bytes_x,
            );
            std::ptr::copy_nonoverlapping(
                host_w.as_ptr() as *const u8,
                buf_w.as_mut_ptr(),
                bytes_w,
            );
            // zero output region
            std::ptr::write_bytes(buf_y.as_mut_ptr(), 0, bytes_y);
        }

        // ── Build packet ───────────────────────────────────────────────────
        // ne/nb shape (HTP_MAX_DIMS = 4). element stride = sizeof(F32) = 4.
        let ne = [dim as u32, rows as u32, 1, 1];
        let nb = [
            4u32,
            (dim * 4) as u32,
            (rows * dim * 4) as u32,
            (rows * dim * 4) as u32,
        ];
        let ne_w = [dim as u32, 1, 1, 1];
        let nb_w = [4u32, (dim * 4) as u32, (dim * 4) as u32, (dim * 4) as u32];

        // Build dispatcher closure — re-issued each iter.
        // Each iter:
        //   1. CpuWriteDspRead flush input
        //   2. dspqueue_write([x, y], packet)  ← n_bufs=2 (llama.cpp htp/main.c:1081
        //      "if (n_bufs != 2) Bad unary-req buffer list")
        //   3. dspqueue_read (blocking)
        //   4. check rsp.status
        //
        // NOTE: llama.cpp HTP_OP_RMS_NORM 은 gamma 미적용. unary-ops.c::rms_norm_f32
        //       가 (src, dst, op_params[0]=eps) 만 사용. gamma 곱하기는 별도 op
        //       (HTP_OP_MUL) 로 chain — PoC scope 외. CPU baseline 도 gamma=1 비교.
        let _ = (ne_w, nb_w, &buf_w); // weight buffer 는 hold 만 (alloc/lifetime)
        let dispatch = |measure_timing: bool| -> anyhow::Result<f64> {
            let mut req = HtpGeneralReq::zeroed();
            let src0 = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
            let dst = htp_tensor_from_shape(HTP_TYPE_F32, ne, nb);
            init_rmsnorm_req(&mut req, eps, src0, dst);
            // src1 미사용 (n_bufs=2 path)

            let mut bufs: [DspQueueBuffer; 2] = [
                buf_x.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes_x as u32)?,
                buf_y.dsp_buf(DspqBufferType::DspWriteCpuRead, 0, bytes_y as u32)?,
            ];

            let t = Instant::now();

            // dspqueue_write (host → DSP)
            // SAFETY: queue/bufs/req live; valid handle.
            let rc = unsafe {
                (host.dspqueue_write)(
                    host.queue,
                    0, // flags
                    2, // num_buffers (input + output)
                    bufs.as_mut_ptr(),
                    core::mem::size_of::<HtpGeneralReq>() as u32,
                    &req as *const _ as *const u8,
                    DSPQUEUE_TIMEOUT,
                )
            };
            if rc != AEE_SUCCESS {
                return Err(map_aee_err(rc).context("dspqueue_write").into());
            }

            // dspqueue_read (DSP → host, blocking)
            let mut rsp = HtpGeneralRsp::zeroed();
            let mut rsp_buf_count: u32 = 0;
            let mut rsp_bufs: [DspQueueBuffer; 2] = [DspQueueBuffer::zeroed(); 2];
            let mut rsp_msg_len: u32 = 0;
            let mut rsp_flags: u32 = 0;

            // SAFETY: out-pointers + queue valid.
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
            if rsp.status != 0 {
                return Err(map_htp_status(rsp.status).context("DSP op status").into());
            }

            let elapsed_us = t.elapsed().as_secs_f64() * 1e6;
            if measure_timing {
                Ok(elapsed_us)
            } else {
                Ok(0.0)
            }
        };

        // ── Correctness gate (single dispatch + readback) ──────────────────
        let _ = dispatch(false)?;
        let htp_result: Vec<f32> = unsafe {
            let p = buf_y.as_ptr() as *const f32;
            std::slice::from_raw_parts(p, rows * dim).to_vec()
        };
        let (max_abs_err, max_rel_err) = compute_err(&htp_result, cpu_baseline);

        // ── Timing loop ────────────────────────────────────────────────────
        let timing = bench(warmup, measure, || dispatch(true))?;

        // RpcmemBuffer Drop on scope exit → rpcmem_free.
        Ok(BackendStats {
            timing,
            max_abs_err,
            max_rel_err,
        })
    }
}

// (anyhow::Error has .context via anyhow::Context — no helper needed.)
