//! microbench_htp_batch_dispatch — FastRPC dspqueue async batching probe.
//!
//! 가설: 현 production `matmul_transposed_via_htp` 는 op 1개를 `dspqueue_write`
//! 직후 `dspqueue_read` blocking 으로 묶는 strict 1:1 동기 dispatch 라 op compute
//! 와 무관하게 op 당 ~100µs floor (FastRPC round-trip + DMA-BUF coherency + DSP
//! wake/signal 왕복) 를 갖는다. dspqueue 는 비동기 FIFO 이므로 "N개 write 를
//! 연속 발행 후 N개 read 로 한 번에 drain" 하면 DSP wake/signal 왕복이 op마다 →
//! 배치당 1회로 amortize 되어 floor 를 회수할 수 있다. 본 microbench 가 이를
//! 직접 측정으로 확정/반증한다.
//!
//! 측정 설계 (두 모드 공정 비교 — 같은 shape / warmup / measure iter):
//!   - Mode A (sync 1:1):   for i in 0..N { write(out[i]) → read → status }
//!   - Mode B (async batch): for i in 0..N { write(out[i]) } 전부 발행 후
//!     for _ in 0..N { read → status } 로 drain
//!
//! 두 모드 모두 동일 weight(공유) + input(공유) + output N개(Mode B 에서 동시
//! in-flight) 를 사용. wall-clock only.
//!
//! Shape (mm_qkv GEMV, Qwen2.5-1.5B): K=1536, N=2048, M=1.
//!   weight  W[N=2048, K=1536] Q4_0 (q4x4x2 repack)
//!   input   x[K=1536] F32
//!   output  y[N=2048] F32  × batch
//!
//! 배치 크기 리스트 N ∈ {1, 7, 14, 28} (기본값; env `HTP_BATCH_SIZES=7,14`
//! 또는 CLI `--batch-sizes 7,14` override).
//!
//! PASS 조건 (N≥7):
//!   (1) max_abs_err < 5e-2 (정확성)
//!   (2) ratio = B/A < 0.8  (floor 회수 실증)
//!   N=1 은 control (ratio≈1.0 기대, gate 미적용).
//!
//! Build (host): cargo build --release --features htp_fastrpc \
//!     --bin microbench_htp_batch_dispatch
//! Build (Android): cargo build --release --features htp_fastrpc \
//!     --target aarch64-linux-android --bin microbench_htp_batch_dispatch
//! Run (device): adb push + LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64 실행.
//!
//! Host PC 실행: HTP path 는 `target_os != android` 에서 SKIP (CPU baseline
//! 만 계산, batch 측정 없음).

#![allow(clippy::unnecessary_wraps)]

#[cfg(not(feature = "htp_fastrpc"))]
fn main() {
    eprintln!("microbench_htp_batch_dispatch requires --features htp_fastrpc");
    std::process::exit(2);
}

#[cfg(feature = "htp_fastrpc")]
fn main() -> anyhow::Result<()> {
    use std::sync::Arc;

    use llm_rs2::backend::Backend;
    use llm_rs2::backend::htp_fastrpc::repack::repack_q4_0_to_q4x4x2_matrix;
    use llm_rs2::buffer::{Buffer, DType};
    use llm_rs2::memory::host::shared::SharedBuffer;
    use llm_rs2::quant::BlockQ4_0;
    use llm_rs2::quant::convert::quantize_q4_0;
    use llm_rs2::shape::Shape;
    use llm_rs2::tensor::Tensor;

    // ── Configuration (mm_qkv GEMV) ──────────────────────────────────────
    #[allow(non_snake_case)]
    let (K, N): (usize, usize) = (1536, 2048);
    const WARMUP: usize = 5;
    const MEASURE: usize = 30;
    // Q4_0 weight × F32 activation + DSP dynamic Q8_0 quant 시 본질적 손실.
    // htp_matmul.rs 와 동일 5e-2 = 5% absolute threshold.
    const ERR_THRESHOLD: f32 = 5e-2;

    // 배치 크기 리스트 — CLI `--batch-sizes 7,14` / env `HTP_BATCH_SIZES=7,14`
    // override. 미지정 시 기본 [1, 7, 14, 28].
    let args: Vec<String> = std::env::args().collect();
    let batch_arg = args
        .windows(2)
        .find(|w| w[0] == "--batch-sizes")
        .map(|w| w[1].clone())
        .or_else(|| std::env::var("HTP_BATCH_SIZES").ok());
    let batch_sizes: Vec<usize> = match batch_arg {
        Some(s) => s
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .collect(),
        None => vec![1, 7, 14, 28],
    };
    let batch_sizes = if batch_sizes.is_empty() {
        vec![1, 7, 14, 28]
    } else {
        batch_sizes
    };

    assert!(K.is_multiple_of(32), "K must be QK4_0 multiple");
    let blocks_per_row = K / 32;
    let weight_bytes = N * blocks_per_row * std::mem::size_of::<BlockQ4_0>();

    println!("=== microbench_htp_batch_dispatch ===");
    println!("shape mm_qkv K={K} N={N} (Q4_0 GEMV)");
    println!(
        "  warmup={WARMUP} measure={MEASURE} err_thresh={ERR_THRESHOLD} \
         batch_sizes={batch_sizes:?}"
    );

    // ── Deterministic synthetic data (htp_matmul.rs 패턴 차용) ────────────
    let mut host_w_f32 = vec![0.0f32; N * K];
    let mut host_x = vec![0.0f32; K];
    for (i, w) in host_w_f32.iter_mut().enumerate() {
        *w = ((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5;
    }
    for (i, x) in host_x.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0291 + 0.13).rem_euclid(1.0) - 0.5;
    }

    // ── Quantize weight to Q4_0 (standard layout) + q4x4x2 repack ─────────
    let q4_blocks = quantize_q4_0(&host_w_f32, N, K);
    assert_eq!(q4_blocks.len(), N * blocks_per_row);
    let q4_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            q4_blocks.as_ptr() as *const u8,
            q4_blocks.len() * std::mem::size_of::<BlockQ4_0>(),
        )
    };
    assert_eq!(q4_bytes.len(), weight_bytes);
    let q4x4x2_bytes = repack_q4_0_to_q4x4x2_matrix(&q4_blocks, N, K);
    assert_eq!(q4x4x2_bytes.len(), weight_bytes);

    // ── CPU baseline (Q4_0 weight, F32 activation, NEON matmul_transposed) ─
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
    let x_cpu = mk_f32_tensor(&host_x, vec![1, K]);
    let w_cpu = mk_q4_tensor(q4_bytes, vec![N, K]);
    let mut out_cpu = mk_f32_tensor(&vec![0.0f32; N], vec![1, N]);
    cpu_backend.matmul_transposed(&x_cpu, &w_cpu, &mut out_cpu)?;
    let cpu_baseline: Vec<f32> = out_cpu.as_slice::<f32>().to_vec();
    let cpu_max = cpu_baseline.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("  cpu baseline magnitude: max|y| = {cpu_max:.4}");

    // ── HTP batch dispatch measurement (android-only) ────────────────────
    let result = run_htp_batch(
        &q4x4x2_bytes,
        &host_x,
        &cpu_baseline,
        N,
        K,
        WARMUP,
        MEASURE,
        &batch_sizes,
        ERR_THRESHOLD,
    );
    match result {
        Ok(rows) => {
            for r in &rows {
                let pass = r.passed(ERR_THRESHOLD);
                println!(
                    "  N={:>2}: A(sync)={:>8.2} us  B(batch)={:>8.2} us  ratio=B/A={:.2}  \
                     max_abs_err={:.2e}  [{}]",
                    r.n,
                    r.a_us,
                    r.b_us,
                    r.ratio(),
                    r.max_abs_err,
                    if pass { "PASS" } else { "FAIL" }
                );
            }
            println!("=== Summary ===");
            let recovery: Vec<String> = rows
                .iter()
                .filter(|r| r.n >= 7)
                .map(|r| format!("N={} ratio={:.2}", r.n, r.ratio()))
                .collect();
            println!("floor recovery (gate: B < 0.8*A): {}", recovery.join(", "));
        }
        Err(e) => {
            println!("  SKIP — {e:#}");
            println!("=== Summary ===");
            println!("floor recovery: SKIP (Android cross-build + S25 deploy required)");
        }
    }

    Ok(())
}

// ── Timing helpers ──────────────────────────────────────────────────────

#[cfg(feature = "htp_fastrpc")]
#[allow(dead_code)] // host PC 빌드에서 run_htp_batch 가 cfg-out → 미사용.
fn median_us(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

#[cfg(feature = "htp_fastrpc")]
#[allow(dead_code)] // host PC 빌드에서 run_htp_batch 가 cfg-out → 미사용.
fn max_abs_err(test: &[f32], baseline: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for (t, b) in test.iter().zip(baseline.iter()) {
        let abs = (t - b).abs();
        if abs > m {
            m = abs;
        }
    }
    m
}

#[cfg(feature = "htp_fastrpc")]
struct BatchRow {
    n: usize,
    /// Mode A (sync 1:1) median wall-clock for the whole N-op batch (µs).
    a_us: f64,
    /// Mode B (async batch) median wall-clock for the whole N-op batch (µs).
    b_us: f64,
    /// max |B_output - CPU_baseline| over the last batch (정확성).
    max_abs_err: f32,
}

#[cfg(feature = "htp_fastrpc")]
impl BatchRow {
    fn ratio(&self) -> f64 {
        if self.a_us > 0.0 {
            self.b_us / self.a_us
        } else {
            f64::NAN
        }
    }

    /// N≥7 에서 (정확성 < thresh) && (ratio < 0.8). N<7 은 control 이라
    /// 정확성만 게이트.
    fn passed(&self, err_thresh: f32) -> bool {
        let acc = self.max_abs_err < err_thresh;
        if self.n >= 7 {
            acc && self.ratio() < 0.8
        } else {
            acc
        }
    }
}

// ── HTP batch dispatch path (Option B — direct rpcmem + dspqueue) ─────────

#[cfg(feature = "htp_fastrpc")]
#[allow(clippy::too_many_arguments)]
fn run_htp_batch(
    q4_weight_bytes: &[u8],
    host_x: &[f32],
    cpu_baseline: &[f32],
    n: usize,
    k: usize,
    warmup: usize,
    measure: usize,
    batch_sizes: &[usize],
    err_thresh: f32,
) -> anyhow::Result<Vec<BatchRow>> {
    #[cfg(not(target_os = "android"))]
    {
        let _ = (
            q4_weight_bytes,
            host_x,
            cpu_baseline,
            n,
            k,
            warmup,
            measure,
            batch_sizes,
            err_thresh,
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
            HTP_TYPE_F32, HTP_TYPE_Q4_0, HtpFastrpcHost, HtpGeneralReq, HtpGeneralRsp,
            RpcmemBuffer, htp_tensor_from_shape, init_matmul_req, map_aee_err, map_htp_status,
        };

        // ── 4-step FastRPC handshake ─────────────────────────────────────
        let host = match HtpFastrpcHost::new("microbench_htp_batch_dispatch") {
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

        // ── Allocate shared weight + input + N output buffers ────────────
        //
        // n_bufs=3 ordering (llama.cpp init_binary_req<true>):
        //   bufs[0] = src0 = weight  (Constant, shared, no cache maintenance)
        //   bufs[1] = src1 = input   (CpuWriteDspRead, shared)
        //   bufs[2] = dst  = output  (DspWriteCpuRead, per-op — Mode B 에서
        //                             N개가 동시 in-flight 라 각각 alloc)
        let bytes_w = q4_weight_bytes.len();
        let bytes_x = k * 4;
        let bytes_y = n * 4;

        let mut buf_w = RpcmemBuffer::alloc(host.clone(), bytes_w, llm_rs2::buffer::DType::Q4_0)?;
        let mut buf_x = RpcmemBuffer::alloc(host.clone(), bytes_x, llm_rs2::buffer::DType::F32)?;

        // host-side weight + input write (Constant/공유, 매 iter 재업로드 불필요).
        // SAFETY: ptr/size 는 alloc 시점 valid, exclusive &mut.
        unsafe {
            std::ptr::copy_nonoverlapping(q4_weight_bytes.as_ptr(), buf_w.as_mut_ptr(), bytes_w);
            std::ptr::copy_nonoverlapping(
                host_x.as_ptr() as *const u8,
                buf_x.as_mut_ptr(),
                bytes_x,
            );
        }

        // 최대 batch size 만큼 output buffer Vec 미리 확보 (가장 큰 N).
        let max_n = *batch_sizes.iter().max().unwrap_or(&1);
        let mut out_bufs: Vec<RpcmemBuffer> = Vec::with_capacity(max_n);
        for _ in 0..max_n {
            let mut b = RpcmemBuffer::alloc(host.clone(), bytes_y, llm_rs2::buffer::DType::F32)?;
            // SAFETY: exclusive &mut, size 일치.
            unsafe { std::ptr::write_bytes(b.as_mut_ptr(), 0, bytes_y) };
            out_bufs.push(b);
        }

        // ── Build packet tensor descriptors (htp_matmul.rs M=1 식 그대로) ──
        let blocks_per_row = (k / 32) as u32;
        let row_bytes_w = blocks_per_row * 18;
        let plane_bytes_w = (n as u32) * row_bytes_w;
        let ne_w = [k as u32, n as u32, 1, 1];
        let nb_w = [18u32, row_bytes_w, plane_bytes_w, plane_bytes_w];
        let ne_x = [k as u32, 1, 1, 1];
        let nb_x = [4u32, (k * 4) as u32, (k * 4) as u32, (k * 4) as u32];
        let ne_y = [n as u32, 1, 1, 1];
        let nb_y = [4u32, (n * 4) as u32, (n * 4) as u32, (n * 4) as u32];

        let make_req = || -> HtpGeneralReq {
            let mut req = HtpGeneralReq::zeroed();
            let src0 = htp_tensor_from_shape(HTP_TYPE_Q4_0, ne_w, nb_w);
            let src1 = htp_tensor_from_shape(HTP_TYPE_F32, ne_x, nb_x);
            let dst = htp_tensor_from_shape(HTP_TYPE_F32, ne_y, nb_y);
            // skip_quantize=false: DSP-side 가 input 을 dynamic 양자화 (Q8_0).
            init_matmul_req(&mut req, src0, src1, dst, false);
            req
        };

        // 단일 write: weight(Constant) + input(CpuWriteDspRead) + out[idx](DspWriteCpuRead).
        // bufs 배열은 write 마다 새로 구성 (weight/input 은 같은 dsp_buf, out[i] 만 다름).
        let write_op = |idx: usize, req: &HtpGeneralReq| -> anyhow::Result<()> {
            let mut bufs: [DspQueueBuffer; 3] = [
                buf_w.dsp_buf(DspqBufferType::Constant, 0, bytes_w as u32)?,
                buf_x.dsp_buf(DspqBufferType::CpuWriteDspRead, 0, bytes_x as u32)?,
                out_bufs[idx].dsp_buf(DspqBufferType::DspWriteCpuRead, 0, bytes_y as u32)?,
            ];
            // SAFETY: queue/bufs/req live; valid handle.
            let rc = unsafe {
                (host.dspqueue_write)(
                    host.queue,
                    0,
                    3,
                    bufs.as_mut_ptr(),
                    core::mem::size_of::<HtpGeneralReq>() as u32,
                    req as *const _ as *const u8,
                    DSPQUEUE_TIMEOUT,
                )
            };
            if rc != AEE_SUCCESS {
                return Err(map_aee_err(rc).context("dspqueue_write").into());
            }
            Ok(())
        };

        // 단일 blocking read + status 체크. FIFO 순서로 응답 반환 — Mode B 에서
        // read i 가 write i 응답에 대응.
        let read_op = || -> anyhow::Result<()> {
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
            let _ = (rsp_msg_len, rsp_flags, rsp_buf_count);
            Ok(())
        };

        let req = make_req();

        // ── Mode A: sync 1:1 — write→read→status per op ──────────────────
        let mode_a = |cur_n: usize| -> anyhow::Result<f64> {
            let t = Instant::now();
            for i in 0..cur_n {
                write_op(i, &req)?;
                read_op()?;
            }
            Ok(t.elapsed().as_secs_f64() * 1e6)
        };

        // ── Mode B: async batch — N writes 전부 발행 후 N reads 로 drain ──
        let mode_b = |cur_n: usize| -> anyhow::Result<f64> {
            let t = Instant::now();
            for i in 0..cur_n {
                write_op(i, &req)?;
            }
            for _ in 0..cur_n {
                read_op()?;
            }
            Ok(t.elapsed().as_secs_f64() * 1e6)
        };

        // ── Per-batch-size measurement (공정 비교: 같은 warmup/measure) ───
        let mut rows = Vec::with_capacity(batch_sizes.len());
        for &cur_n in batch_sizes {
            // warmup (양 모드 모두 — DSP wake/cache 워밍).
            for _ in 0..warmup {
                let _ = mode_a(cur_n)?;
                let _ = mode_b(cur_n)?;
            }

            // Mode A measure.
            let mut a_samples = Vec::with_capacity(measure);
            for _ in 0..measure {
                a_samples.push(mode_a(cur_n)?);
            }
            let a_us = median_us(&mut a_samples);

            // Mode B measure — 마지막 batch 의 output 을 정확성 검증에 사용.
            let mut b_samples = Vec::with_capacity(measure);
            for _ in 0..measure {
                b_samples.push(mode_b(cur_n)?);
            }
            let b_us = median_us(&mut b_samples);

            // 정확성: Mode B 마지막 batch 의 out_bufs[0..cur_n] 전부를 CPU
            // baseline 과 비교 (각 op 은 동일 weight/input 이라 모두 같은 결과).
            let mut err = 0.0f32;
            for buf in out_bufs.iter().take(cur_n) {
                // SAFETY: buf alive, F32 n elements.
                let out: &[f32] =
                    unsafe { core::slice::from_raw_parts(buf.as_ptr() as *const f32, n) };
                let e = max_abs_err(out, cpu_baseline);
                if e > err {
                    err = e;
                }
            }

            rows.push(BatchRow {
                n: cur_n,
                a_us,
                b_us,
                max_abs_err: err,
            });
        }

        let _ = err_thresh; // 게이트는 호출부 print 에서 적용.
        // out_bufs / buf_w / buf_x Drop on scope exit → rpcmem_free.
        Ok(rows)
    }
}
