//! microbench_htp_matmul_f16 — P1d Senior Implementer (Full matrix sprint)
//!
//! HTP NPU MUL_MAT F16 dispatch — F16 weight × F32 activation → F32 output.
//! Qwen 2.5-1.5B의 3 shape (mm_ffn / mm_lmh / mm_qkv) 모두 측정.
//!
//! DSP-side (`libggml-htp-v79.so` `matmul-ops.c:2360`) 는 F16 weight 가
//! `src0->type == HTP_TYPE_F16` 일 때 `f16-f32` (또는 VTCM fit 시 `f16-f16`)
//! path 로 분기한다. weight repack (Q4_0 의 q4x4x2 같은 변환) 불필요 —
//! F16 row-major bytes 직접 dispatch.
//!
//! 측정 shape (Qwen 2.5-1.5B, K=1536 고정):
//!   - mm_ffn  : K=1536  N=8960   (FFN gate / up)
//!   - mm_lmh  : K=1536  N=151936 (LM head, ~445 MB F16 weight)
//!   - mm_qkv  : K=1536  N=2048   (QKV fused, Q=1536+K=256+V=256)
//!
//! Sprint goal: P1d acceptance gate
//!   1. mm_ffn (K=1536 N=8960)   F16 GREEN
//!   2. mm_qkv (K=1536 N=2048)   F16 GREEN (K/V N=256 dim variance 검증의 합)
//!   3. mm_lmh (K=1536 N=151936) F16 GREEN gate — DSP allocation 한계 검증
//!
//! Per backend per shape: warmup 10 + measure 100 iter, median + mean + stddev.
//! wall-clock only (CLAUDE.md `feedback_opencl_profile_events_cross_engine`).
//!
//! Correctness (F16 weight × F32 activation):
//!   F16 round-trip + DSP HVX fp16 path 의 본질 손실은 ~1e-3 magnitude.
//!   threshold = 5e-3 (F32 baseline 대비, llama.cpp test-backend-ops F16
//!   MUL_MAT NMSE 5e-2 보다 strict).
//!
//! Build (Android cross):
//!   cargo build --release --features htp_fastrpc \
//!       --target aarch64-linux-android --bin microbench_htp_matmul_f16
//!
//! Run on device:
//!   adb -s R3CY408S5SB push target/aarch64-linux-android/release/microbench_htp_matmul_f16 /data/local/tmp/
//!   adb -s R3CY408S5SB shell "LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64 \
//!              ADSP_LIBRARY_PATH=/data/local/tmp \
//!              /data/local/tmp/microbench_htp_matmul_f16"
//!
//! Selective shape:
//!   HTP_MM_F16_SHAPES=mm_ffn,mm_qkv ./microbench_htp_matmul_f16  (default = all 3)

#![allow(clippy::unnecessary_wraps)]

#[cfg(not(feature = "htp_fastrpc"))]
fn main() {
    eprintln!("microbench_htp_matmul_f16 requires --features htp_fastrpc");
    std::process::exit(2);
}

#[cfg(feature = "htp_fastrpc")]
fn main() -> anyhow::Result<()> {
    use std::sync::Arc;

    use half::f16;
    use llm_rs2::backend::Backend;
    use llm_rs2::buffer::{Buffer, DType};
    use llm_rs2::memory::host::shared::SharedBuffer;
    use llm_rs2::shape::Shape;
    use llm_rs2::tensor::Tensor;

    const WARMUP: usize = 10;
    const MEASURE: usize = 100;
    // F16 weight + DSP HVX fp16 dot path 의 본질 손실. F32 baseline 대비
    // ~1e-3 magnitude 가 normal. threshold 5e-3 absolute (mm_lmh 큰 reduction
    // dim 누적 고려 시 7e-3, matrix.md v2-§7 tolerance 표 inherit).
    const ERR_THRESHOLD_DEFAULT: f32 = 5e-3;
    const ERR_THRESHOLD_LMH: f32 = 7e-3;

    // ── Shape inventory ───────────────────────────────────────────────────
    // (label, K, N, err_threshold)
    let all_shapes: &[(&str, usize, usize, f32)] = &[
        ("mm_ffn", 1536, 8960, ERR_THRESHOLD_DEFAULT),
        ("mm_qkv", 1536, 2048, ERR_THRESHOLD_DEFAULT),
        ("mm_lmh", 1536, 151936, ERR_THRESHOLD_LMH),
    ];

    // --shape <sid> CLI 우선 (드라이버가 셀별로 주입), 없으면 HTP_MM_F16_SHAPES env (하위호환).
    // 이전엔 env-only 라 드라이버의 --shape 를 silent ignore → 한 프로세스가 다중 shape 를
    // 순차 측정 → 드라이버의 last-`mean=` 파서가 항상 마지막(mm_lmh) 값을 모든 셀에 기록하는
    // 라벨 버그 (full_matrix_2026_05_28 의 ours.htp F16 "7.8ms 고정" 의 원인).
    let args: Vec<String> = std::env::args().collect();
    let cli_shape = args
        .windows(2)
        .find(|w| w[0] == "--shape")
        .map(|w| w[1].clone());
    let env_filter = cli_shape.or_else(|| std::env::var("HTP_MM_F16_SHAPES").ok());
    let shapes: Vec<&(&str, usize, usize, f32)> = match env_filter.as_deref() {
        Some(s) => {
            let wanted: Vec<&str> = s.split(',').map(|x| x.trim()).collect();
            all_shapes
                .iter()
                .filter(|t| wanted.contains(&t.0))
                .collect()
        }
        None => all_shapes.iter().collect(),
    };

    println!("=== microbench_htp_matmul_f16 (P1d sprint) ===\n");
    println!("Config:");
    println!("  warmup iter   = {WARMUP}");
    println!("  measure iter  = {MEASURE}");
    println!("  shapes        = {} cell\n", shapes.len());

    let cpu_backend: Arc<dyn Backend> = llm_rs2::backend::cpu::cpu_singleton();

    for tuple in shapes.iter() {
        let (label, k, n, err_threshold) = **tuple;
        let weight_bytes_f16 = n * k * 2;
        let input_bytes_f32 = k * 4;
        let output_bytes_f32 = n * 4;

        println!("──────────────────────────────────────────────────────");
        println!(
            "shape `{label}` (K={k}, N={n}) — weight F16 {} MB, input F32 {} KB, output F32 {} KB",
            weight_bytes_f16 as f32 / (1024.0 * 1024.0),
            input_bytes_f32 as f32 / 1024.0,
            output_bytes_f32 as f32 / 1024.0,
        );

        // ── Deterministic synthetic data ──────────────────────────────────
        let mut host_w_f32 = vec![0.0f32; n * k];
        let mut host_x = vec![0.0f32; k];
        for (i, w) in host_w_f32.iter_mut().enumerate() {
            *w = ((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5;
        }
        for (i, x) in host_x.iter_mut().enumerate() {
            *x = ((i as f32) * 0.0291 + 0.13).rem_euclid(1.0) - 0.5;
        }

        // Convert weight to F16 row-major bytes (n_rows × k columns, row-major).
        // DSP-side `matmul-ops.c::vec_dot_f16_*` 가 row-major bytes 를 그대로 사용.
        let host_w_f16: Vec<f16> = host_w_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let w_f16_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(host_w_f16.as_ptr() as *const u8, host_w_f16.len() * 2)
        };
        assert_eq!(w_f16_bytes.len(), weight_bytes_f16);

        // CPU baseline = F32 weight × F32 input (high precision reference).
        // F16 round-trip 손실은 별도 측정 — baseline 자체는 F32 정확도 유지.
        let mk_f32_tensor =
            |data: &[f32], shape: Vec<usize>, backend: &Arc<dyn Backend>| -> Tensor {
                let buf = SharedBuffer::new(data.len() * 4, DType::F32);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u8,
                        buf.as_mut_ptr(),
                        data.len() * 4,
                    );
                }
                Tensor::new(Shape::new(shape), Arc::new(buf), backend.clone())
            };

        // CPU baseline: F16 weight 의 F32 round-trip (F16 → F32) 으로 matmul.
        // → DSP F16 path 의 round-trip 손실과 동등한 baseline.
        let host_w_round: Vec<f32> = host_w_f16.iter().map(|h| h.to_f32()).collect();
        let x_cpu = mk_f32_tensor(&host_x, vec![1, k], &cpu_backend);
        let w_cpu = mk_f32_tensor(&host_w_round, vec![n, k], &cpu_backend);
        let mut out_cpu = mk_f32_tensor(&vec![0.0f32; n], vec![1, n], &cpu_backend);
        cpu_backend.matmul_transposed(&x_cpu, &w_cpu, &mut out_cpu)?;
        let cpu_baseline: Vec<f32> = out_cpu.as_slice::<f32>().to_vec();
        let cpu_max = cpu_baseline.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        println!("  CPU baseline (F16-rt → F32 matmul): max|y| = {cpu_max:.4}");

        // ── HTP NPU dispatch ──────────────────────────────────────────────
        let htp_result = run_htp(w_f16_bytes, &host_x, &cpu_baseline, n, k, WARMUP, MEASURE);
        match htp_result {
            Ok(stats) => {
                let pass = stats.max_abs_err < err_threshold;
                println!(
                    "  HTP F16: mean={:.2} us (median={:.2}, stddev={:.2}, n={})",
                    stats.timing.mean_us, stats.timing.median_us, stats.timing.stddev_us, MEASURE,
                );
                println!(
                    "    vs CPU(F16-rt): max_abs_err={:.3e}, max_rel_err={:.3e} — {} (thresh={:.2e})\n",
                    stats.max_abs_err,
                    stats.max_rel_err,
                    if pass { "PASS" } else { "FAIL" },
                    err_threshold,
                );
            }
            Err(e) => {
                println!("  HTP F16: SKIP — {e:#}\n");
            }
        }
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
#[allow(dead_code)]
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
    w_f16_bytes: &[u8],
    host_x: &[f32],
    cpu_baseline: &[f32],
    n: usize,
    k: usize,
    warmup: usize,
    measure: usize,
) -> anyhow::Result<BackendStats> {
    #[cfg(not(target_os = "android"))]
    {
        let _ = (w_f16_bytes, host_x, cpu_baseline, n, k, warmup, measure);
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
            HTP_TYPE_F16, HTP_TYPE_F32, HtpFastrpcHost, HtpGeneralReq, HtpGeneralRsp, RpcmemBuffer,
            htp_tensor_from_shape, init_matmul_req, map_aee_err, map_htp_status,
        };

        // 4-step FastRPC handshake.
        let host = match HtpFastrpcHost::new("microbench_htp_matmul_f16") {
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
        // n_bufs=3 ordering (init_binary_req<true>):
        //   bufs[0] = src0 = weight  (Constant)
        //   bufs[1] = src1 = input   (CpuWriteDspRead)
        //   bufs[2] = dst  = output  (DspWriteCpuRead)
        let bytes_w = w_f16_bytes.len();
        let bytes_x = k * 4;
        let bytes_y = n * 4;

        let mut buf_w = RpcmemBuffer::alloc(host.clone(), bytes_w)
            .with_context(|| format!("rpcmem alloc weight {bytes_w} B (F16 N={n} K={k})"))?;
        let mut buf_x = RpcmemBuffer::alloc(host.clone(), bytes_x)?;
        let mut buf_y = RpcmemBuffer::alloc(host.clone(), bytes_y)?;

        unsafe {
            std::ptr::copy_nonoverlapping(w_f16_bytes.as_ptr(), buf_w.as_mut_ptr(), bytes_w);
            std::ptr::copy_nonoverlapping(
                host_x.as_ptr() as *const u8,
                buf_x.as_mut_ptr(),
                bytes_x,
            );
            std::ptr::write_bytes(buf_y.as_mut_ptr(), 0, bytes_y);
        }

        // ── Build packet ───────────────────────────────────────────────────
        //
        // ggml convention for F16 weight W[N, K] (row-major, ne0=K innermost):
        //   ne = (K, N, 1, 1)
        //   nb[0] = sizeof(f16) = 2
        //   nb[1] = K * 2          — row stride
        //   nb[2..3] = N * nb[1]   — plane stride
        let row_bytes_w = (k * 2) as u32;
        let plane_bytes_w = (n as u32) * row_bytes_w;
        let ne_w = [k as u32, n as u32, 1, 1];
        let nb_w = [2u32, row_bytes_w, plane_bytes_w, plane_bytes_w];

        // input F32 vector x[K]
        let ne_x = [k as u32, 1, 1, 1];
        let nb_x = [4u32, (k * 4) as u32, (k * 4) as u32, (k * 4) as u32];

        // output F32 vector y[N]
        let ne_y = [n as u32, 1, 1, 1];
        let nb_y = [4u32, (n * 4) as u32, (n * 4) as u32, (n * 4) as u32];

        let dispatch = |measure_timing: bool| -> anyhow::Result<f64> {
            let mut req = HtpGeneralReq::zeroed();
            let src0 = htp_tensor_from_shape(HTP_TYPE_F16, ne_w, nb_w);
            let src1 = htp_tensor_from_shape(HTP_TYPE_F32, ne_x, nb_x);
            let dst = htp_tensor_from_shape(HTP_TYPE_F32, ne_y, nb_y);
            // skip_quantize=true 일 필요 없음 — F16 path 는 DSP-side 가
            // src1 F32 → fp16 quantize 를 자체 처리 (`quantize_f32_f16`).
            init_matmul_req(&mut req, src0, src1, dst, false);

            let mut bufs: [DspQueueBuffer; 3] = [
                buf_w.dsp_buf(DspqBufferType::Constant, 0, bytes_w as u32)?,
                buf_x.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes_x as u32)?,
                buf_y.dsp_buf(DspqBufferType::DspWriteCpuRead, 0, bytes_y as u32)?,
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
                    "  diag (F16, N={n} K={k}): rsp.op={}, status={}, prof_usecs={}, prof_cycles={}, prof_pkts={}, rsp_buf_count={}",
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

        // Correctness gate
        let _ = dispatch(false)?;
        let htp_result: Vec<f32> = unsafe {
            let p = buf_y.as_ptr() as *const f32;
            std::slice::from_raw_parts(p, n).to_vec()
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
