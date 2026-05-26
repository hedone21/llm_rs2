//! microbench_htp_rope — Q-2.2 β G3 sprint (HTP_OP_ROPE)
//!
//! 2-way ROPE microbench: CPU baseline (host reference) vs QNN HTP NPU
//! (FastRPC + `libggml-htp-v79.so` dspqueue path, n_bufs=3 path).
//!
//! 목적: matrix.md row 14 (ROPE head_dim=128 (q12,kv2) θ=1e6) 의 Ours-NPU 셀
//! 채움 + dispatch correctness GREEN gate. ADD/MUL/SILU sprint 의 n_bufs=3
//! base 위에서 op_params 9 slot packing 차이 (i32×3 + f32×6 bit-cast).
//!
//! 측정 형식:
//!   1. CPU baseline (host-side mode=0 normal RoPE reference)
//!   2. QNN HTP NPU (`engine::backend::htp_fastrpc::HtpFastrpcHost` raw dispatch)
//!
//! Per backend: warmup 10 iter + measure 1000 iter (default), median + mean
//! + stddev. wall-clock only.
//!
//! Correctness:
//!   baseline = CPU reference output.
//!   max_abs_err / max_rel_err vs CPU. threshold = 1e-3 (F32-only path —
//!   sin/cos floating point 자체 손실 분만 허용).
//!
//! Shape (Qwen2.5-1.5B Q-rotation, decode n=1):
//!   src0 = input     [head_dim=128, n_heads=12, n_tokens=1] F32 (6,144 B)
//!   src1 = positions [n_tokens=1] i32 (4 B, alloc 64 B minimum)
//!   dst  = output    same shape as src0
//!
//! RoPE params (Qwen2.5 vendor, mode=0 normal split):
//!   n_dims=128, mode=0, n_ctx_orig=32768
//!   freq_base=1e6, freq_scale=1.0
//!   ext_factor=0.0, attn_factor=1.0, beta_fast=32.0, beta_slow=1.0
//!
//! Build (host):
//!   cargo build --release --features htp_fastrpc --bin microbench_htp_rope
//!
//! Build (Android cross):
//!   cargo build --release --features htp_fastrpc \
//!       --target aarch64-linux-android --bin microbench_htp_rope
//!
//! Run on device:
//!   adb -s R3CY408S5SB push target/aarch64-linux-android/release/microbench_htp_rope /data/local/tmp/
//!   adb -s R3CY408S5SB shell "LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64 \
//!              ADSP_LIBRARY_PATH=/data/local/tmp \
//!              /data/local/tmp/microbench_htp_rope"

#![allow(clippy::unnecessary_wraps)]

#[cfg(not(feature = "htp_fastrpc"))]
fn main() {
    eprintln!("microbench_htp_rope requires --features htp_fastrpc");
    std::process::exit(2);
}

#[cfg(feature = "htp_fastrpc")]
fn main() -> anyhow::Result<()> {
    use std::time::Instant;

    // ── Configuration (Qwen2.5-1.5B Q-rotation decode) ────────────────────
    const HEAD_DIM: usize = 128;
    const N_HEADS: usize = 12;
    const N_TOKENS: usize = 1;
    const N_DIMS: usize = 128;
    const FREQ_BASE: f32 = 1_000_000.0;
    const FREQ_SCALE: f32 = 1.0;
    const POSITION: i32 = 5; // 임의 nontrivial position
    const WARMUP: usize = 10;
    const MEASURE: usize = 1000;
    const ERR_THRESHOLD: f32 = 1e-3;

    let n_elem = HEAD_DIM * N_HEADS * N_TOKENS;
    let bytes_in = n_elem * 4;
    let bytes_pos = (N_TOKENS * 4).max(64); // llama.cpp 가 min size 64 B 보장
    let bytes_out = bytes_in;

    println!("=== microbench_htp_rope (Q-2.2 β G3 sprint) ===\n");
    println!("Config:");
    println!("  input  shape  = [{HEAD_DIM}, {N_HEADS}, {N_TOKENS}] F32 ({bytes_in} B)");
    println!("  pos    shape  = [{N_TOKENS}] i32 (alloc {bytes_pos} B, ≥64 B min)");
    println!("  output shape  = same as input ({bytes_out} B)");
    println!(
        "  rope params   = n_dims={N_DIMS} mode=0 freq_base={FREQ_BASE} freq_scale={FREQ_SCALE}"
    );
    println!("  position      = {POSITION}");
    println!("  warmup iter   = {WARMUP}");
    println!("  measure iter  = {MEASURE}");
    println!("  err thresh    = {ERR_THRESHOLD}\n");

    // ── Deterministic synthetic data ──────────────────────────────────────
    let mut host_in = vec![0.0f32; n_elem];
    for (i, x) in host_in.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5;
    }
    let host_pos = vec![POSITION; N_TOKENS];

    // ── [1/2] CPU baseline (host reference RoPE) ──────────────────────────
    println!("[1/2] CPU baseline (host-side mode=0 normal RoPE reference)");
    let cpu_baseline = rope_ref(
        &host_in, &host_pos, N_DIMS, FREQ_BASE, FREQ_SCALE, HEAD_DIM, N_HEADS, N_TOKENS,
    );
    let cpu_max = cpu_baseline.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("  baseline magnitude: max|y| = {cpu_max:.4}");

    let cpu_stats = bench(WARMUP, MEASURE, || {
        let t = Instant::now();
        let _out = rope_ref(
            &host_in, &host_pos, N_DIMS, FREQ_BASE, FREQ_SCALE, HEAD_DIM, N_HEADS, N_TOKENS,
        );
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
        &host_pos,
        &cpu_baseline,
        HEAD_DIM,
        N_HEADS,
        N_TOKENS,
        bytes_in,
        bytes_pos,
        bytes_out,
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

// ── Host RoPE reference (mode=0 normal interleaved, ggml convention) ──────
//
// **중요**: ggml `mode=0` (NORMAL, GGML_ROPE_TYPE_NORMAL) = **interleaved pair**
// (i, i+1). NeoX (mode=2) 가 split-pair (i, i+n_dims/2). 본 sprint 안내 문서가
// 둘을 거꾸로 명시했었으나 llama.cpp `ggml/src/ggml-hexagon/htp/rope-ops.c`
// 실측으로 정정 (line 215-216 normal vs 173-174 NeoX 비교).
//
//   theta_i = position * freq_scale / freq_base^(2 i / n_dims), i ∈ [0, n_dims/2)
//
//   pair index k = 0..n_dims/2:  i = 2*k
//     y[i+0] = x[i+0] * cos(theta_k) - x[i+1] * sin(theta_k)
//     y[i+1] = x[i+0] * sin(theta_k) + x[i+1] * cos(theta_k)
//
// theta cache stride: rope-ops.c `rope_cache_init` 가 `for i0 += 2` 로 매 pair 마다
// 1 theta 생성 → cache[2k+0]=cos, cache[2k+1]=sin. element index i 에서 본 theta
// k = i/2.
//
// head_dim 의 tail (n_dims..head_dim) 은 passthrough (rotation skip).

#[cfg(feature = "htp_fastrpc")]
fn rope_ref(
    input: &[f32],
    positions: &[i32],
    n_dims: usize,
    freq_base: f32,
    freq_scale: f32,
    head_dim: usize,
    n_heads: usize,
    n_tokens: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    let half = n_dims / 2;
    for t in 0..n_tokens {
        let pos = positions[t] as f32 * freq_scale;
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            // mode=0 NORMAL: interleaved pair (i, i+1), pair index k = 0..half
            for k in 0..half {
                let theta = pos / freq_base.powf(2.0 * k as f32 / n_dims as f32);
                let (s, c) = theta.sin_cos();
                let i = 2 * k;
                let x0 = input[base + i];
                let x1 = input[base + i + 1];
                output[base + i] = x0 * c - x1 * s;
                output[base + i + 1] = x0 * s + x1 * c;
            }
            // tail passthrough (head_dim > n_dims case)
            for i in n_dims..head_dim {
                output[base + i] = input[base + i];
            }
        }
    }
    output
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
    host_pos: &[i32],
    cpu_baseline: &[f32],
    head_dim: usize,
    n_heads: usize,
    n_tokens: usize,
    bytes_in: usize,
    bytes_pos: usize,
    bytes_out: usize,
    warmup: usize,
    measure: usize,
) -> anyhow::Result<BackendStats> {
    #[cfg(not(target_os = "android"))]
    {
        let _ = (
            host_in,
            host_pos,
            cpu_baseline,
            head_dim,
            n_heads,
            n_tokens,
            bytes_in,
            bytes_pos,
            bytes_out,
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
            HTP_TYPE_F32, HTP_TYPE_I32, HtpFastrpcHost, HtpGeneralReq, HtpGeneralRsp, RopeParams,
            RpcmemBuffer, htp_tensor_from_shape, init_rope_req, map_aee_err, map_htp_status,
        };

        // 4-step FastRPC handshake.
        let host = match HtpFastrpcHost::new("microbench_htp_rope") {
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
        // n_bufs=3 ordering (llama.cpp init_rope_req, src2 freq_factors 미사용):
        //   bufs[0] = src0 (CpuWriteDspRead, host input activation)
        //   bufs[1] = src1 (CpuWriteDspRead, positions i32)
        //   bufs[2] = dst  (DspWriteCpuRead, output)
        let mut buf_in = RpcmemBuffer::alloc(host.clone(), bytes_in)?;
        let mut buf_pos = RpcmemBuffer::alloc(host.clone(), bytes_pos)?;
        let mut buf_out = RpcmemBuffer::alloc(host.clone(), bytes_out)?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                host_in.as_ptr() as *const u8,
                buf_in.as_mut_ptr(),
                bytes_in,
            );
            // positions: i32×N_TOKENS 만 채우고 나머지 padding 은 0.
            std::ptr::write_bytes(buf_pos.as_mut_ptr(), 0, bytes_pos);
            std::ptr::copy_nonoverlapping(
                host_pos.as_ptr() as *const u8,
                buf_pos.as_mut_ptr(),
                host_pos.len() * 4,
            );
            std::ptr::write_bytes(buf_out.as_mut_ptr(), 0, bytes_out);
        }

        // ── Build packet ───────────────────────────────────────────────────
        let ne_in = [head_dim as u32, n_heads as u32, n_tokens as u32, 1];
        let nb_in = [
            4u32,
            (head_dim * 4) as u32,
            (head_dim * n_heads * 4) as u32,
            (head_dim * n_heads * n_tokens * 4) as u32,
        ];
        let ne_pos = [n_tokens as u32, 1, 1, 1];
        let nb_pos = [4u32, 4, 4, 4];

        let params = RopeParams::qwen2_5();

        let dispatch = |measure_timing: bool| -> anyhow::Result<f64> {
            let mut req = HtpGeneralReq::zeroed();
            let src0 = htp_tensor_from_shape(HTP_TYPE_F32, ne_in, nb_in);
            let src1 = htp_tensor_from_shape(HTP_TYPE_I32, ne_pos, nb_pos);
            let dst = htp_tensor_from_shape(HTP_TYPE_F32, ne_in, nb_in);
            init_rope_req(&mut req, &params, src0, src1, dst);

            let mut bufs: [DspQueueBuffer; 3] = [
                buf_in.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes_in as u32)?,
                buf_pos.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes_pos as u32)?,
                buf_out.dsp_buf(DspqBufferType::DspWriteCpuRead, 0, bytes_out as u32)?,
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
        let n_elem = head_dim * n_heads * n_tokens;
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
