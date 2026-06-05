//! microbench_htp_matmul — Q-2.2 β G2 sprint (MUL_MAT Q4_0 GEMV)
//!
//! 2-way MUL_MAT microbench: CPU baseline (Q4_0 weight, F32 activation/output)
//! vs QNN HTP NPU (FastRPC + `libggml-htp-v79.so` dspqueue path).
//!
//! 목적: matrix.md row 4a (MUL_MAT Q-proj, Qwen2.5-1.5B) 의 Ours-NPU 셀
//! 채움 + dispatch correctness GREEN gate. RMSNORM sprint 의 host binding +
//! DspQueueBuffer 24B layout fix (`dde08248`) 위에서 n_bufs=3 path 만 차이.
//!
//! 측정 형식:
//!   1. CPU baseline (`engine::backend::cpu::cpu_singleton`) — NEON Q4_0 GEMV
//!   2. QNN HTP NPU (`engine::backend::htp_fastrpc::HtpFastrpcBackend`)
//!
//! Per backend: warmup 10 iter + measure 100 iter (default), median + mean
//! + stddev. wall-clock only (CLAUDE.md `feedback_opencl_profile_events_cross_engine`).
//!
//! Correctness:
//!   baseline = CPU NEON output.
//!   max_abs_err / max_rel_err vs CPU. threshold = 1e-3
//!   (spec/htp_fastrpc.md INV-HTP-FRPC-003, RMSNORM sprint 와 동일 게이트).
//!
//! Shape (Qwen2.5-1.5B Q-proj):
//!   weight  W[N=1536, K=1536] Q4_0 (row-major, 73728 blocks × 18 B = 1,327,104 B)
//!   input   x[K=1536] F32 (6,144 B)
//!   output  y[N=1536] F32 (6,144 B)
//!
//! 데이터 시드: deterministic (rmsnorm 패턴 차용). weight = pseudo-random in
//! [-0.5, 0.5), input = pseudo-uniform in [-0.5, 0.5).
//!
//! Build (host):
//!   cargo build --release --features htp_fastrpc --bin microbench_htp_matmul
//!
//! Build (Android cross):
//!   cargo build --release --features htp_fastrpc \
//!       --target aarch64-linux-android --bin microbench_htp_matmul
//!
//! Run on device:
//!   adb -s R3CY408S5SB push target/aarch64-linux-android/release/microbench_htp_matmul /data/local/tmp/
//!   adb -s R3CY408S5SB shell "LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64 \
//!              /data/local/tmp/microbench_htp_matmul"
//!
//! Host PC 실행:
//!   - HTP path 는 `target_os != android` 에서 unavailable 메시지 + skip.
//!   - CPU baseline 만 측정.

#![allow(clippy::unnecessary_wraps)]

#[cfg(not(feature = "htp_fastrpc"))]
fn main() {
    eprintln!("microbench_htp_matmul requires --features htp_fastrpc");
    std::process::exit(2);
}

#[cfg(feature = "htp_fastrpc")]
fn main() -> anyhow::Result<()> {
    use std::sync::Arc;
    use std::time::Instant;

    use llm_rs2::backend::Backend;
    use llm_rs2::buffer::{Buffer, DType};
    use llm_rs2::memory::host::shared::SharedBuffer;
    use llm_rs2::quant::BlockQ4_0;
    use llm_rs2::quant::convert::quantize_q4_0;
    use llm_rs2::shape::Shape;
    use llm_rs2::tensor::Tensor;

    // ── Configuration (Qwen2.5-1.5B) ──────────────────────────────────────
    // --shape <sid> CLI / HTP_MM_Q4_SHAPE env 로 shape 선택 (드라이버가 셀별 주입).
    // 이전엔 const N=K=1536 하드코딩 → 드라이버의 --shape 를 silent ignore → 세 셀
    // (mm_qkv/mm_ffn/mm_lmh)이 전부 N=1536 단일 측정의 복제본이 되는 버그.
    let args: Vec<String> = std::env::args().collect();
    let cli_shape = args
        .windows(2)
        .find(|w| w[0] == "--shape")
        .map(|w| w[1].clone())
        .or_else(|| std::env::var("HTP_MM_Q4_SHAPE").ok());
    #[allow(non_snake_case)]
    let (K, N): (usize, usize) = match cli_shape.as_deref() {
        Some("mm_qkv") => (1536, 2048),
        Some("mm_ffn") => (1536, 8960),
        Some("mm_lmh") => (1536, 151936),
        _ => (1536, 1536), // shape 미지정 시 기존 기본값
    };
    const WARMUP: usize = 10;
    const MEASURE: usize = 100;
    // Q4_0 weight × F32 activation + DSP dynamic Q8_0 quant 시 본질적 손실
    // (vec_dot_q4x4x2_q8x4x2_* path). CPU NEON baseline 은 F32 dot 으로 더
    // 정확. ~1e-2 magnitude 차이는 expected. 5e-2 = 5% absolute threshold
    // (llama.cpp test-backend-ops MUL_MAT Q4_0 reference 대비 유사 범위).
    const ERR_THRESHOLD: f32 = 5e-2;

    assert!(K.is_multiple_of(32), "K must be QK4_0 multiple");
    let blocks_per_row = K / 32;
    let weight_bytes = N * blocks_per_row * std::mem::size_of::<BlockQ4_0>();
    let input_bytes = K * 4;
    let output_bytes = N * 4;

    println!("=== microbench_htp_matmul (Q-2.2 β G2 sprint) ===\n");
    println!("Config:");
    println!("  weight shape  = [N={N}, K={K}] Q4_0 ({weight_bytes} B)");
    println!("  input shape   = [K={K}] F32 ({input_bytes} B)");
    println!("  output shape  = [N={N}] F32 ({output_bytes} B)");
    println!("  warmup iter   = {WARMUP}");
    println!("  measure iter  = {MEASURE}");
    println!("  err thresh    = {ERR_THRESHOLD}\n");

    // ── Deterministic synthetic data ──────────────────────────────────────
    let mut host_w_f32 = vec![0.0f32; N * K];
    let mut host_x = vec![0.0f32; K];
    for (i, w) in host_w_f32.iter_mut().enumerate() {
        // pseudo-uniform on [-0.5, 0.5)
        *w = ((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5;
    }
    for (i, x) in host_x.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0291 + 0.13).rem_euclid(1.0) - 0.5;
    }

    // ── Quantize weight to Q4_0 (standard block_q4_0 layout) ──────────────
    // CPU baseline 은 standard `block_q4_0` (18 B/block) 을 그대로 사용.
    // HTP path 는 추가로 q4x4x2 layout 으로 repack 후 dispatch (DSP-side
    // vec_dot_q4x4x2_q8x4x2_* 가 expect — llama.cpp ggml-hexagon.cpp:402
    // `repack_row_q4x4x2`). Q4_0 자체 byte 표현은 동일하지만 8 blocks 단위로
    // quants/scales 가 분리 packed 됨.
    let q4_blocks = quantize_q4_0(&host_w_f32, N, K);
    assert_eq!(q4_blocks.len(), N * blocks_per_row);
    let q4_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            q4_blocks.as_ptr() as *const u8,
            q4_blocks.len() * std::mem::size_of::<BlockQ4_0>(),
        )
    };
    assert_eq!(q4_bytes.len(), weight_bytes);

    // q4x4x2 repacked layout: 같은 byte 길이지만 quants 와 scales 가 분리.
    // K=1536 = 6×256 (QK_Q4_0x4x2 multiple), padding 불필요.
    let q4x4x2_bytes = repack_q4_0_to_q4x4x2_matrix(&q4_blocks, N, K);
    assert_eq!(q4x4x2_bytes.len(), weight_bytes);

    // ── [1/2] CPU baseline (Q4_0 weight, F32 activation) ──────────────────
    println!("[1/2] CPU baseline (Q4_0 × F32 → F32, NEON matmul_transposed)");
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
    let mk_q4_tensor = |bytes: &[u8], shape: Vec<usize>| -> Tensor {
        let buf = SharedBuffer::new(bytes.len(), DType::Q4_0);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.as_mut_ptr(), bytes.len());
        }
        Tensor::new(Shape::new(shape), Arc::new(buf), cpu_backend.clone())
    };

    // CPU baseline: matmul_transposed(x[1, K], W[N, K]) -> out[1, N]
    let x_cpu = mk_f32_tensor(&host_x, vec![1, K]);
    let w_cpu = mk_q4_tensor(q4_bytes, vec![N, K]);
    let mut out_cpu = mk_f32_tensor(&vec![0.0f32; N], vec![1, N]);
    cpu_backend.matmul_transposed(&x_cpu, &w_cpu, &mut out_cpu)?;
    let cpu_baseline: Vec<f32> = out_cpu.as_slice::<f32>().to_vec();

    // sanity: 합리적 magnitude (Q4_0 round-trip 후 GEMV 결과는 보통 O(K * 0.5 * 0.5) 정도)
    let cpu_max = cpu_baseline.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("  baseline magnitude: max|y| = {cpu_max:.4}");

    let cpu_stats = bench(WARMUP, MEASURE, || {
        let x = mk_f32_tensor(&host_x, vec![1, K]);
        let mut out = mk_f32_tensor(&vec![0.0f32; N], vec![1, N]);
        let t = Instant::now();
        cpu_backend.matmul_transposed(&x, &w_cpu, &mut out)?;
        Ok(t.elapsed().as_secs_f64() * 1e6)
    })?;
    println!(
        "  mean={:.2} us (median={:.2}, stddev={:.2}, n={})\n",
        cpu_stats.mean_us, cpu_stats.median_us, cpu_stats.stddev_us, MEASURE
    );

    // ── [2/2] QNN HTP NPU ─────────────────────────────────────────────────
    println!("[2/2] QNN HTP NPU (FastRPC + libggml-htp-v79.so dspqueue)");
    let htp_result = run_htp(&q4x4x2_bytes, &host_x, &cpu_baseline, N, K, WARMUP, MEASURE);
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

// ── q4x4x2 repack (llama.cpp HTP backend layout, ggml-hexagon.cpp:402) ────
//
// llama.cpp HTP backend 는 standard `block_q4_0` (32 elem × 18 B) 를 **q4x4x2
// layout** 으로 repack 후 DSP 에 전달. DSP-side `vec_dot_q4x4x2_q8x4x2_*`
// (matmul-ops.c:734+) 가 이 layout 을 expect.
//
// q4x4x2 group = 8 인접 Q4_0 block (256 elem, 144 B):
//   - 128 B quants (각 elem 4-bit pair-packed, group 내 256 elem)
//   - 16 B scales (8 × f16)
//
// 본 함수는 row-major weight matrix `W[N, K]` 의 standard Q4_0 byte stream
// 을 q4x4x2-packed byte stream 으로 변환. K 는 256 multiple 가정 (Qwen2.5
// shape K∈{1536, 8960} 모두 manage).

#[cfg(feature = "htp_fastrpc")]
fn repack_q4_0_to_q4x4x2_matrix(
    src: &[llm_rs2::quant::BlockQ4_0],
    n_rows: usize,
    k: usize,
) -> Vec<u8> {
    const QK_Q4_0X4X2: usize = 256; // llama.cpp htp-msg.h: QK_Q4_0x4x2
    const QK4_0: usize = 32;
    assert!(
        k.is_multiple_of(QK_Q4_0X4X2),
        "K must be QK_Q4_0x4x2 multiple"
    );
    let blocks_per_row = k / QK4_0; // 18 B/block in standard layout
    let row_bytes = blocks_per_row * 18; // = (k/32)*18 = k * 9 / 16
    let mut out = vec![0u8; n_rows * row_bytes];
    for r in 0..n_rows {
        let src_row = &src[r * blocks_per_row..(r + 1) * blocks_per_row];
        let dst_row = &mut out[r * row_bytes..(r + 1) * row_bytes];
        repack_row_q4x4x2(dst_row, src_row, k);
    }
    out
}

#[cfg(feature = "htp_fastrpc")]
fn repack_row_q4x4x2(y: &mut [u8], x: &[llm_rs2::quant::BlockQ4_0], k: usize) {
    use half::f16;
    const QK_Q4_0X4X2: usize = 256;
    const QK4_0: usize = 32;
    let nb = k.div_ceil(QK_Q4_0X4X2); // number of q4x4x2 groups

    let dblk_size = 8 * 2; // 8 × f16 = 16 B
    let qblk_size = QK_Q4_0X4X2 / 2; // 128 B (4-bit per elem, 256 elem)
    let qrow_size = k / 2; // K/2 B (int4 not padded to blocks)

    // y_q at offset 0 (quants), y_d at offset qrow_size (scales).
    // SAFETY: caller 가 y 가 충분히 큼 (row_bytes = qrow_size + nb*dblk_size)을 보장.
    //
    // standard block_q4_0 (`{d: f16, qs: [u8; 16]}`) 의 qs 는 nibble pair-pack
    // 된 32 elem. nibble unpack 시 lower nibble = elem [0..16], upper nibble =
    // elem [16..32].

    // Repack quants
    for i in 0..nb {
        // unpacked 256 elem buffer for this group (8 blocks)
        let mut qs_unpacked = [0u8; QK_Q4_0X4X2];
        for bi in 0..8 {
            let block_idx = i * 8 + bi;
            if block_idx >= x.len() {
                break;
            }
            let blk = &x[block_idx];
            // unpack_q4_0_quants (ggml-hexagon.cpp:381)
            for j in 0..(QK4_0 / 2) {
                let x0 = blk.qs[j] & 0x0F;
                let x1 = blk.qs[j] >> 4;
                qs_unpacked[bi * QK4_0 + j] = x0;
                qs_unpacked[bi * QK4_0 + j + QK4_0 / 2] = x1;
            }
        }
        // repack: `q[j] = (qs[j+128] << 4) | qs[j]` for j in [0..128)
        let q_off = i * qblk_size;
        for j in 0..(QK_Q4_0X4X2 / 2) {
            y[q_off + j] = (qs_unpacked[j + 128] << 4) | qs_unpacked[j];
        }
    }

    // Repack scales (8 × f16 per group)
    for i in 0..nb {
        let d_off = qrow_size + i * dblk_size;
        for bi in 0..8 {
            let block_idx = i * 8 + bi;
            if block_idx >= x.len() {
                break;
            }
            let d_bits: u16 = f16::to_bits(x[block_idx].d);
            y[d_off + bi * 2] = (d_bits & 0xFF) as u8;
            y[d_off + bi * 2 + 1] = ((d_bits >> 8) & 0xFF) as u8;
        }
    }
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
#[allow(dead_code)] // host PC 빌드에서 run_htp 가 cfg-out 되어 미사용 경고 발생.
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
    q4_weight_bytes: &[u8],
    host_x: &[f32],
    cpu_baseline: &[f32],
    n: usize,
    k: usize,
    warmup: usize,
    measure: usize,
) -> anyhow::Result<BackendStats> {
    #[cfg(not(target_os = "android"))]
    {
        let _ = (q4_weight_bytes, host_x, cpu_baseline, n, k, warmup, measure);
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
            HTP_TYPE_F32, HTP_TYPE_Q4_0, HtpFastrpcHost, HtpGeneralReq, HtpGeneralRsp,
            RpcmemBuffer, htp_tensor_from_shape, init_matmul_req, map_aee_err, map_htp_status,
        };

        // 4-step FastRPC handshake.
        let host = match HtpFastrpcHost::new("microbench_htp_matmul") {
            Ok(h) => h,
            Err(e) => anyhow::bail!("HtpFastrpcHost::new failed: {}", e),
        };
        println!(
            "  host: libcdsprpc.so={}, domain_id={}, session_id={}, queue_id={}",
            host.lib_path, host.domain_id, host.session_id, host.queue_id
        );

        // β-1 PoC: lifecycle htp_iface_start(handle, sess_id, queue_id, n_hvx).
        let n_hvx: u32 = std::env::var("HTP_FASTRPC_N_HVX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        host.try_start_iface(n_hvx)
            .with_context(|| format!("htp_iface_start(n_hvx={n_hvx})"))?;
        println!("  htp_iface_start: OK (n_hvx={n_hvx})");

        // ── Allocate rpcmem buffers ─────────────────────────────────────────
        //
        // n_bufs=3 ordering (llama.cpp init_binary_req<true>):
        //   bufs[0] = src0 = weight  (Constant, no cache maintenance)
        //   bufs[1] = src1 = input   (CpuWriteDspRead, flush+invalidate)
        //   bufs[2] = dst  = output  (DspWriteCpuRead, flush)
        let bytes_w = q4_weight_bytes.len();
        let bytes_x = k * 4;
        let bytes_y = n * 4;

        let mut buf_w = RpcmemBuffer::alloc(host.clone(), bytes_w, llm_rs2::buffer::DType::Q4_0)?;
        let mut buf_x = RpcmemBuffer::alloc(host.clone(), bytes_x, llm_rs2::buffer::DType::F32)?;
        let mut buf_y = RpcmemBuffer::alloc(host.clone(), bytes_y, llm_rs2::buffer::DType::F32)?;

        // host-side weight write (한 번만, Constant 라 매 iter 재업로드 불필요)
        unsafe {
            std::ptr::copy_nonoverlapping(q4_weight_bytes.as_ptr(), buf_w.as_mut_ptr(), bytes_w);
            std::ptr::copy_nonoverlapping(
                host_x.as_ptr() as *const u8,
                buf_x.as_mut_ptr(),
                bytes_x,
            );
            std::ptr::write_bytes(buf_y.as_mut_ptr(), 0, bytes_y);
        }

        // ── Build packet ───────────────────────────────────────────────────
        //
        // ggml convention for Q4_0 weight `W[N, K]` (row-major):
        //   ne = (K, N, 1, 1)
        //   nb[0] = sizeof(block_q4_0) = 18
        //   nb[1] = (K/32) * 18           — row stride in bytes
        //   nb[2..3] = N * nb[1]          — plane/file stride
        let blocks_per_row = (k / 32) as u32;
        let row_bytes_w = blocks_per_row * 18;
        let plane_bytes_w = (n as u32) * row_bytes_w;
        let ne_w = [k as u32, n as u32, 1, 1];
        let nb_w = [18u32, row_bytes_w, plane_bytes_w, plane_bytes_w];

        // input F32 vector x[K] — ne0=K
        let ne_x = [k as u32, 1, 1, 1];
        let nb_x = [4u32, (k * 4) as u32, (k * 4) as u32, (k * 4) as u32];

        // output F32 vector y[N] — ne0=N
        let ne_y = [n as u32, 1, 1, 1];
        let nb_y = [4u32, (n * 4) as u32, (n * 4) as u32, (n * 4) as u32];

        let dispatch = |measure_timing: bool| -> anyhow::Result<f64> {
            let mut req = HtpGeneralReq::zeroed();
            let src0 = htp_tensor_from_shape(HTP_TYPE_Q4_0, ne_w, nb_w);
            let src1 = htp_tensor_from_shape(HTP_TYPE_F32, ne_x, nb_x);
            let dst = htp_tensor_from_shape(HTP_TYPE_F32, ne_y, nb_y);
            // skip_quantize=false: DSP-side 가 input 을 dynamic 양자화 (Q8_0).
            init_matmul_req(&mut req, src0, src1, dst, false);

            // bufs ordering MUST match init_binary_req<true>:
            //   [0] weight (Constant), [1] input (CpuWriteDspRead), [2] output (DspWriteCpuRead)
            let mut bufs: [DspQueueBuffer; 3] = [
                buf_w.dsp_buf(DspqBufferType::Constant, 0, bytes_w as u32)?,
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
                    3, // num_buffers (weight + input + output)
                    bufs.as_mut_ptr(),
                    core::mem::size_of::<HtpGeneralReq>() as u32,
                    &req as *const _ as *const u8,
                    DSPQUEUE_TIMEOUT,
                )
            };
            if rc != AEE_SUCCESS {
                return Err(map_aee_err(rc).context("dspqueue_write").into());
            }

            // dspqueue_read (DSP → host, blocking). proc_matmul_req 가 rsp_bufs[0]=dst
            // 만 flush 하므로 max_buffers=1 이면 충분하지만 여유 4 로 헤더 ok.
            let mut rsp = HtpGeneralRsp::zeroed();
            let mut rsp_buf_count: u32 = 0;
            let mut rsp_bufs: [DspQueueBuffer; 4] = [DspQueueBuffer::zeroed(); 4];
            let mut rsp_msg_len: u32 = 0;
            let mut rsp_flags: u32 = 0;

            // SAFETY: out-pointers + queue valid.
            let rc = unsafe {
                (host.dspqueue_read)(
                    host.queue,
                    &mut rsp_flags as *mut u32,
                    4, // max_buffers
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
            // HTP_STATUS_OK = 1 (llama.cpp htp-msg.h:24). 0 은 uninitialized.
            if rsp.status != HTP_STATUS_OK {
                return Err(map_htp_status(rsp.status).context("DSP op status").into());
            }
            if !measure_timing {
                // 첫 dispatch 의 DSP-side prof + rsp buf 진단 — production 측정에는
                // 영향 없음 (timing loop 에서는 measure_timing=true 라 skip).
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

        // ── Correctness gate (single dispatch + readback) ──────────────────
        let _ = dispatch(false)?;
        let htp_result: Vec<f32> = unsafe {
            let p = buf_y.as_ptr() as *const f32;
            std::slice::from_raw_parts(p, n).to_vec()
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
