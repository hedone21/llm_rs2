//! microbench_opencl_op_matrix — P1b OpenCL raw GPU latency measurement
//!
//! ours.gpu 컬럼 (v2-§6 group inventory) 측정 전용.
//! Qwen 2.5-1.5B 실제 decode (seq_len=1) 핫패스 shape 기준.
//!
//! 커버 cell (matrix.md v2):
//!   MUL_MAT F16  × 3 shape (mm_ffn / mm_lmh / mm_qkv)
//!   MUL_MAT Q4_0 × 3 shape (mm_ffn / mm_lmh / mm_qkv)
//!   RMS_NORM F16 × 1 shape [1, 1536]
//!   ROPE     F16 × 1 shape head_dim=128
//!   FLASH_ATTN_EXT F16 × 1 shape ctx=1024
//!   GET_ROWS F16 × 1 shape vocab=151936
//!   SILU     F16 × 1 shape [1, 8960]   (SiLU·MUL 합산 — 두 op 동시 dispatch)
//!   MUL      F16 × 1 shape [1, 8960]
//!   ADD      F16 × 1 shape [1, 1536]
//!   SOFT_MAX F16 × 1 shape [12, 1, 1024]
//!   SCALE    F16 × 1 shape [12, 1, 1024]
//!   CPY      F16 × 1 shape [2, 1, 128]
//!   SET_ROWS F16 × 1 shape [2, 1, 128] → cache[1024, 2, 128]
//!
//! 측정 protocol (sprint master 요건):
//!   warmup 3 + measure 100 iter (elem op), 30 iter (GEMV)
//!   wall-clock Instant (CL_QUEUE_PROFILING_ENABLE 사용 안 함 — CLAUDE.md feedback)
//!   stdout JSON per cell: {cell_id, median_ns, mean_ns, n_valid, cv_percent}
//!
//! Build:
//!   cargo build --release -p llm_rs2 --bin microbench_opencl_op_matrix --features opencl
//!
//! Run:
//!   ./microbench_opencl_op_matrix
//!   ./microbench_opencl_op_matrix --ops MUL_MAT_F16,RMS_NORM_F16

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("microbench_opencl_op_matrix requires --features opencl");
    std::process::exit(2);
}

#[cfg(feature = "opencl")]
fn main() {
    if let Err(e) = bench::run() {
        eprintln!("ERROR: {:#}", e);
        std::process::exit(1);
    }
}

#[cfg(feature = "opencl")]
mod bench {
    use ocl::core::ArgVal;
    use ocl::{Context, Device, Platform, Program, Queue};
    use std::time::Instant;

    // === Qwen 2.5-1.5B model params ===
    const DIM: usize = 1536;
    const N_HEADS_Q: usize = 12;
    const N_HEADS_KV: usize = 2;
    const HEAD_DIM: usize = 128;
    const FFN_DIM: usize = 8960;
    const VOCAB: usize = 151936;
    const CTX_LEN: usize = 1024; // representative decode ctx

    const QK4_0: usize = 32;

    const WARMUP: usize = 3;
    const MEASURE_GEMV: usize = 30;
    const MEASURE_ELEM: usize = 100;

    struct Ctx {
        context: Context,
        queue: Queue,
        prog_simple: Program,
        prog_mul_mv_f16: Program,
        prog_mul_mat_q4_0: Program,
        prog_get_rows: Program,
    }

    fn build_opts(device: &Device) -> String {
        let v = device
            .info(ocl::core::DeviceInfo::OpenclCVersion)
            .map(|v| v.to_string())
            .unwrap_or_default();
        let cl2 = v
            .split_whitespace()
            .nth(2)
            .and_then(|ver| {
                let mut p = ver.split('.');
                let major: u32 = p.next()?.parse().ok()?;
                let minor: u32 = p.next()?.parse().ok()?;
                Some((major, minor) >= (2, 0))
            })
            .unwrap_or(false);
        let fast = "-cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math";
        if cl2 {
            format!("-cl-std=CL2.0 {}", fast)
        } else {
            fast.to_string()
        }
    }

    fn init() -> anyhow::Result<Ctx> {
        let platform = Platform::default();
        let device = Device::first(platform)?;
        eprintln!(
            "# Platform: {}  Device: {}",
            platform.name().unwrap_or_else(|_| "?".into()),
            device.name().unwrap_or_else(|_| "?".into()),
        );

        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;
        // NOTE: no QUEUE_PROFILING_ENABLE — wall-clock only (CLAUDE.md feedback)
        let queue = Queue::new(&context, device, None)?;

        let opts = build_opts(&device);

        let prog_simple = Program::builder()
            .devices(device)
            .src(include_str!("../kernels/simple_ops.cl"))
            .cmplr_opt(&opts)
            .build(&context)?;

        let prog_mul_mv_f16 = Program::builder()
            .devices(device)
            .src(include_str!("../kernels/mul_mv_f16_f32.cl"))
            .cmplr_opt(&opts)
            .build(&context)?;

        let prog_mul_mat_q4_0 = Program::builder()
            .devices(device)
            .src(include_str!("../kernels/mul_mv_q4_0_f32.cl"))
            .cmplr_opt(&opts)
            .build(&context)?;

        let prog_get_rows = Program::builder()
            .devices(device)
            .src(include_str!("../kernels/get_rows.cl"))
            .cmplr_opt(&opts)
            .build(&context)?;

        Ok(Ctx {
            context,
            queue,
            prog_simple,
            prog_mul_mv_f16,
            prog_mul_mat_q4_0,
            prog_get_rows,
        })
    }

    // ── buffer helpers ────────────────────────────────────────────────────────

    fn alloc_f32(ctx: &Ctx, n: usize) -> anyhow::Result<ocl::core::Mem> {
        let buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                ctx.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                n,
                None,
            )?
        };
        let zeros = vec![0.0f32; n];
        unsafe {
            ocl::core::enqueue_write_buffer(
                &ctx.queue,
                &buf,
                true,
                0,
                &zeros,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(buf)
    }

    fn alloc_f16(ctx: &Ctx, n: usize) -> anyhow::Result<ocl::core::Mem> {
        let buf = unsafe {
            ocl::core::create_buffer::<_, u16>(
                ctx.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                n,
                None,
            )?
        };
        let zeros = vec![0u16; n];
        unsafe {
            ocl::core::enqueue_write_buffer(
                &ctx.queue,
                &buf,
                true,
                0,
                &zeros,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(buf)
    }

    fn alloc_u8(ctx: &Ctx, n: usize) -> anyhow::Result<ocl::core::Mem> {
        let buf = unsafe {
            ocl::core::create_buffer::<_, u8>(
                ctx.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                n,
                None,
            )?
        };
        let zeros = vec![0u8; n];
        unsafe {
            ocl::core::enqueue_write_buffer(
                &ctx.queue,
                &buf,
                true,
                0,
                &zeros,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(buf)
    }

    fn alloc_i32(ctx: &Ctx, n: usize) -> anyhow::Result<ocl::core::Mem> {
        let buf = unsafe {
            ocl::core::create_buffer::<_, i32>(
                ctx.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                n,
                None,
            )?
        };
        let zeros = vec![0i32; n];
        unsafe {
            ocl::core::enqueue_write_buffer(
                &ctx.queue,
                &buf,
                true,
                0,
                &zeros,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(buf)
    }

    fn finish(ctx: &Ctx) -> anyhow::Result<()> {
        ocl::core::finish(&ctx.queue)?;
        Ok(())
    }

    // ── statistics ────────────────────────────────────────────────────────────

    fn median_ns(samples: &[f64]) -> f64 {
        let mut s = samples.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    }

    fn mean_ns(samples: &[f64]) -> f64 {
        samples.iter().sum::<f64>() / samples.len() as f64
    }

    fn cv_percent(samples: &[f64]) -> f64 {
        let mean = mean_ns(samples);
        if mean == 0.0 {
            return 0.0;
        }
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        var.sqrt() / mean * 100.0
    }

    fn print_result(cell_id: &str, samples: &[f64]) {
        println!(
            "{{\"cell_id\":\"{}\",\"median_ns\":{:.0},\"mean_ns\":{:.0},\"n_valid\":{},\"cv_percent\":{:.2}}}",
            cell_id,
            median_ns(samples),
            mean_ns(samples),
            samples.len(),
            cv_percent(samples)
        );
    }

    fn measure<F: FnMut() -> anyhow::Result<()>>(
        ctx: &Ctx,
        warmup: usize,
        iters: usize,
        mut f: F,
    ) -> anyhow::Result<Vec<f64>> {
        for _ in 0..warmup {
            f()?;
            finish(ctx)?;
        }
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            f()?;
            finish(ctx)?;
            samples.push(t0.elapsed().as_nanos() as f64);
        }
        Ok(samples)
    }

    // ── MUL_MAT F16 GEMV ─────────────────────────────────────────────────────

    fn bench_mul_mat_f16(ctx: &Ctx, cell_id: &str, k: usize, n: usize) -> anyhow::Result<()> {
        let kernel = ocl::core::create_kernel(&ctx.prog_mul_mv_f16, "kernel_mul_mat_f16_f32")?;

        let w = alloc_f16(ctx, n * k)?;
        let x = alloc_f32(ctx, k)?;
        let y = alloc_f32(ctx, n)?;

        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1i32;
        let ne10 = k as i32;
        let ne12 = 1i32;
        let ne0 = n as i32;
        let ne1 = 1i32;
        let r2 = 1i32;
        let r3 = 1i32;
        let off0 = 0u64;
        let off1 = 0u64;
        let offd = 0u64;

        ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&w))?;
        ocl::core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&off0))?;
        ocl::core::set_kernel_arg(&kernel, 2, ArgVal::mem(&x))?;
        ocl::core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&off1))?;
        ocl::core::set_kernel_arg(&kernel, 4, ArgVal::mem(&y))?;
        ocl::core::set_kernel_arg(&kernel, 5, ArgVal::scalar(&offd))?;
        ocl::core::set_kernel_arg(&kernel, 6, ArgVal::scalar(&ne00))?;
        ocl::core::set_kernel_arg(&kernel, 7, ArgVal::scalar(&ne01))?;
        ocl::core::set_kernel_arg(&kernel, 8, ArgVal::scalar(&ne02))?;
        ocl::core::set_kernel_arg(&kernel, 9, ArgVal::scalar(&ne10))?;
        ocl::core::set_kernel_arg(&kernel, 10, ArgVal::scalar(&ne12))?;
        ocl::core::set_kernel_arg(&kernel, 11, ArgVal::scalar(&ne0))?;
        ocl::core::set_kernel_arg(&kernel, 12, ArgVal::scalar(&ne1))?;
        ocl::core::set_kernel_arg(&kernel, 13, ArgVal::scalar(&r2))?;
        ocl::core::set_kernel_arg(&kernel, 14, ArgVal::scalar(&r3))?;

        // 4-wave kernel: N_DST=2 → ceil(n/2) groups × 64 lanes × 4 waves
        let n_groups = n.div_ceil(2);
        let global = [n_groups * 64, 4, 1];
        let local = [64usize, 4, 1];

        let samples = measure(ctx, WARMUP, MEASURE_GEMV, || {
            unsafe {
                ocl::core::enqueue_kernel(
                    &ctx.queue,
                    &kernel,
                    3,
                    None,
                    &global,
                    Some(local),
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            Ok(())
        })?;

        print_result(cell_id, &samples);
        Ok(())
    }

    // ── MUL_MAT Q4_0 GEMV ────────────────────────────────────────────────────

    fn bench_mul_mat_q4_0(ctx: &Ctx, cell_id: &str, k: usize, n: usize) -> anyhow::Result<()> {
        let kernel = ocl::core::create_kernel(&ctx.prog_mul_mat_q4_0, "kernel_mul_mat_q4_0_f32")?;

        // Q4_0 interleaved: each block = 2 bytes (scale f16) + 16 bytes (nibbles) = 18 bytes
        let num_blocks = n * k / QK4_0;
        let w = alloc_u8(ctx, num_blocks * 18)?;
        let x = alloc_f32(ctx, k)?;
        let y = alloc_f32(ctx, n)?;

        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1i32;
        let ne10 = k as i32;
        let ne12 = 1i32;
        let ne0 = n as i32;
        let ne1 = 1i32;
        let r2 = 1i32;
        let r3 = 1i32;
        let off0 = 0u64;
        let off1 = 0u64;
        let offd = 0u64;

        ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&w))?;
        ocl::core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&off0))?;
        ocl::core::set_kernel_arg(&kernel, 2, ArgVal::mem(&x))?;
        ocl::core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&off1))?;
        ocl::core::set_kernel_arg(&kernel, 4, ArgVal::mem(&y))?;
        ocl::core::set_kernel_arg(&kernel, 5, ArgVal::scalar(&offd))?;
        ocl::core::set_kernel_arg(&kernel, 6, ArgVal::scalar(&ne00))?;
        ocl::core::set_kernel_arg(&kernel, 7, ArgVal::scalar(&ne01))?;
        ocl::core::set_kernel_arg(&kernel, 8, ArgVal::scalar(&ne02))?;
        ocl::core::set_kernel_arg(&kernel, 9, ArgVal::scalar(&ne10))?;
        ocl::core::set_kernel_arg(&kernel, 10, ArgVal::scalar(&ne12))?;
        ocl::core::set_kernel_arg(&kernel, 11, ArgVal::scalar(&ne0))?;
        ocl::core::set_kernel_arg(&kernel, 12, ArgVal::scalar(&ne1))?;
        ocl::core::set_kernel_arg(&kernel, 13, ArgVal::scalar(&r2))?;
        ocl::core::set_kernel_arg(&kernel, 14, ArgVal::scalar(&r3))?;

        // dispatch: global=[ceil(n/8)*64, 1, 1]
        let global = [n.div_ceil(8) * 64, 1, 1];
        let local = [64usize, 1, 1];

        let samples = measure(ctx, WARMUP, MEASURE_GEMV, || {
            unsafe {
                ocl::core::enqueue_kernel(
                    &ctx.queue,
                    &kernel,
                    3,
                    None,
                    &global,
                    Some(local),
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            Ok(())
        })?;

        print_result(cell_id, &samples);
        Ok(())
    }

    // ── RMS_NORM F16 ─────────────────────────────────────────────────────────

    fn bench_rms_norm(ctx: &Ctx, cell_id: &str, dim: usize) -> anyhow::Result<()> {
        let kernel = ocl::core::create_kernel(&ctx.prog_simple, "kernel_rms_norm_opt")?;

        let x = alloc_f32(ctx, dim)?;
        let w = alloc_f32(ctx, dim)?;
        let eps = 1e-6f32;
        let add_unit = 0i32;
        let local_size = 64usize;
        let local_mem = local_size * std::mem::size_of::<f32>();

        ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&x))?;
        ocl::core::set_kernel_arg(&kernel, 1, ArgVal::mem(&w))?;
        ocl::core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&(dim as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&eps))?;
        ocl::core::set_kernel_arg(&kernel, 4, ArgVal::scalar(&add_unit))?;
        ocl::core::set_kernel_arg(&kernel, 5, ArgVal::local::<f32>(&local_mem))?;

        let global = [local_size, 1, 1]; // 1 row
        let local = [local_size, 1, 1];

        let samples = measure(ctx, WARMUP, MEASURE_ELEM, || {
            unsafe {
                ocl::core::enqueue_kernel(
                    &ctx.queue,
                    &kernel,
                    1,
                    None,
                    &global,
                    Some(local),
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            Ok(())
        })?;

        print_result(cell_id, &samples);
        Ok(())
    }

    // ── ROPE F16 ─────────────────────────────────────────────────────────────

    fn bench_rope(ctx: &Ctx, cell_id: &str) -> anyhow::Result<()> {
        let kernel = ocl::core::create_kernel(&ctx.prog_simple, "kernel_rope_simple")?;

        let n_heads = N_HEADS_Q;
        let head_dim = HEAD_DIM;
        let seq_len = 1usize;
        let start_pos = 0i32;
        let theta = 1_000_000.0f32;

        let x = alloc_f32(ctx, n_heads * seq_len * head_dim)?;

        ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&x))?;
        ocl::core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&(head_dim as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&(n_heads as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&(seq_len as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 4, ArgVal::scalar(&start_pos))?;
        ocl::core::set_kernel_arg(&kernel, 5, ArgVal::scalar(&theta))?;

        let work_size = seq_len * n_heads * (head_dim / 2);
        let global = [work_size, 1, 1];

        let samples = measure(ctx, WARMUP, MEASURE_ELEM, || {
            unsafe {
                ocl::core::enqueue_kernel(
                    &ctx.queue,
                    &kernel,
                    1,
                    None,
                    &global,
                    None,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            Ok(())
        })?;

        print_result(cell_id, &samples);
        Ok(())
    }

    // ── FLASH_ATTN_EXT F16 decode Q1 ─────────────────────────────────────────

    fn bench_flash_attn(ctx: &Ctx, cell_id: &str) -> anyhow::Result<()> {
        // Lazy compile: FlashAttn 커널은 이 op 실행 시에만 빌드한다.
        // flash_attn_f32_f16.cl은 BLOCK_N(prefill 전용 define)이 없으면 전체 컴파일 실패하므로
        // init() 단계에서 eager 빌드하지 않는다.
        let device = ocl::Device::first(ocl::Platform::default())?;
        let opts = build_opts(&device);
        let prog_flash_attn_q1 = Program::builder()
            .devices(device)
            .src(include_str!("../kernels/flash_attn_f32_f16.cl"))
            .cmplr_opt(format!(
                "{} -DDK=128 -DDV=128 -DBLOCK_M=4 -DBLOCK_N=8",
                opts
            ))
            .build(&ctx.context)
            .map_err(|e| anyhow::anyhow!("flash_attn_q1 DK=128 compile: {}", e))?;

        let kernel = ocl::core::create_kernel(&prog_flash_attn_q1, "flash_attn_f32_f16_q1")?;

        let n_heads_q = N_HEADS_Q;
        let n_heads_kv = N_HEADS_KV;
        let head_dim = HEAD_DIM;
        let n_kv = CTX_LEN;
        let kv_capacity = CTX_LEN + 64;

        let q = alloc_f32(ctx, n_heads_q * head_dim)?;
        let k_cache = alloc_f16(ctx, n_heads_kv * kv_capacity * head_dim)?;
        let v_cache = alloc_f16(ctx, n_heads_kv * kv_capacity * head_dim)?;
        let o = alloc_f32(ctx, n_heads_q * head_dim)?;
        let score_dummy = alloc_f32(ctx, 1)?;

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let n_q = 1i32;
        let n_kv_i = n_kv as i32;
        let is_causal = 0i32;
        let n_head = n_heads_q as i32;
        let n_head_kv_arg = n_heads_kv as i32;
        let max_bias = 0.0f32;
        let m0 = 0.0f32;
        let m1 = 0.0f32;
        let n_head_log2 = 0i32;
        let logit_softcap = 0.0f32;

        let q_nb1 = (n_heads_q * head_dim * 4) as u64;
        let q_nb2 = (head_dim * 4) as u64;
        let q_nb3 = q_nb1;
        let k_nb1 = (head_dim * 2) as u64;
        let k_nb2 = (kv_capacity * head_dim * 2) as u64;
        let k_nb3 = (n_heads_kv as u64) * k_nb2;
        let (v_nb1, v_nb2, v_nb3) = (k_nb1, k_nb2, k_nb3);
        let o_nb1 = (head_dim * 4) as u64;
        let o_nb2 = (n_heads_q * head_dim * 4) as u64;
        let o_nb3 = o_nb2;

        let zero_u64 = 0u64;
        let zero_i32 = 0i32;

        ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&q))?;
        ocl::core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 2, ArgVal::mem(&k_cache))?;
        ocl::core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 4, ArgVal::mem(&v_cache))?;
        ocl::core::set_kernel_arg(&kernel, 5, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 6, ArgVal::mem(&o))?;
        ocl::core::set_kernel_arg(&kernel, 7, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 8, ArgVal::scalar(&scale))?;
        ocl::core::set_kernel_arg(&kernel, 9, ArgVal::scalar(&n_q))?;
        ocl::core::set_kernel_arg(&kernel, 10, ArgVal::scalar(&n_kv_i))?;
        ocl::core::set_kernel_arg(&kernel, 11, ArgVal::scalar(&is_causal))?;
        ocl::core::set_kernel_arg(&kernel, 12, ArgVal::scalar(&n_head))?;
        ocl::core::set_kernel_arg(&kernel, 13, ArgVal::scalar(&q_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 14, ArgVal::scalar(&q_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 15, ArgVal::scalar(&q_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 16, ArgVal::scalar(&k_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 17, ArgVal::scalar(&k_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 18, ArgVal::scalar(&k_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 19, ArgVal::scalar(&v_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 20, ArgVal::scalar(&v_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 21, ArgVal::scalar(&v_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 22, ArgVal::scalar(&o_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 23, ArgVal::scalar(&o_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 24, ArgVal::scalar(&o_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 25, ArgVal::scalar(&max_bias))?;
        ocl::core::set_kernel_arg(&kernel, 26, ArgVal::scalar(&m0))?;
        ocl::core::set_kernel_arg(&kernel, 27, ArgVal::scalar(&m1))?;
        ocl::core::set_kernel_arg(&kernel, 28, ArgVal::scalar(&n_head_log2))?;
        ocl::core::set_kernel_arg(&kernel, 29, ArgVal::scalar(&logit_softcap))?;
        ocl::core::set_kernel_arg(&kernel, 30, ArgVal::scalar(&n_head_kv_arg))?;
        // mask = NULL
        ocl::core::set_kernel_arg(&kernel, 31, ArgVal::mem_null())?;
        ocl::core::set_kernel_arg(&kernel, 32, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 33, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 34, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 35, ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 36, ArgVal::scalar(&zero_i32))?;
        ocl::core::set_kernel_arg(&kernel, 37, ArgVal::scalar(&zero_i32))?;
        // sinks = NULL
        ocl::core::set_kernel_arg(&kernel, 38, ArgVal::mem_null())?;
        ocl::core::set_kernel_arg(&kernel, 39, ArgVal::scalar(&zero_u64))?;
        // score output disabled (write_scores=0)
        ocl::core::set_kernel_arg(&kernel, 40, ArgVal::mem(&score_dummy))?;
        ocl::core::set_kernel_arg(&kernel, 41, ArgVal::scalar(&zero_i32))?;
        ocl::core::set_kernel_arg(&kernel, 42, ArgVal::scalar(&zero_i32))?;
        ocl::core::set_kernel_arg(&kernel, 43, ArgVal::scalar(&zero_i32))?;

        const Q1_WG_SIZE: usize = 64;
        let global = [Q1_WG_SIZE, n_heads_q, 1];
        let local = [Q1_WG_SIZE, 1, 1];

        let samples = measure(ctx, WARMUP, MEASURE_GEMV, || {
            unsafe {
                ocl::core::enqueue_kernel(
                    &ctx.queue,
                    &kernel,
                    3,
                    None,
                    &global,
                    Some(local),
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            Ok(())
        })?;

        print_result(cell_id, &samples);
        Ok(())
    }

    // ── GET_ROWS F16 ─────────────────────────────────────────────────────────

    fn bench_get_rows(ctx: &Ctx, cell_id: &str) -> anyhow::Result<()> {
        let kernel = ocl::core::create_kernel(&ctx.prog_get_rows, "kernel_get_rows_f16")?;

        let vocab = VOCAB;
        let k = DIM;
        let src = alloc_f16(ctx, vocab * k)?;
        let idx = alloc_i32(ctx, 1)?;
        let dst = alloc_f32(ctx, k)?;

        let ne00 = k as i32;
        let nb01 = (k * 2) as u64;
        let nb02 = nb01 * vocab as u64;
        let nb03 = nb02;
        let ne10 = 1i32;
        let nb10 = 4u64;
        let nb11 = nb10;
        let nb12 = nb10;
        let nb1 = (k * 4) as u64;
        let nb2 = nb1;
        let nb3 = nb1;
        let off0 = 0u64;
        let off1 = 0u64;
        let offd = 0u64;

        ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&src))?;
        ocl::core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&off0))?;
        ocl::core::set_kernel_arg(&kernel, 2, ArgVal::mem(&idx))?;
        ocl::core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&off1))?;
        ocl::core::set_kernel_arg(&kernel, 4, ArgVal::mem(&dst))?;
        ocl::core::set_kernel_arg(&kernel, 5, ArgVal::scalar(&offd))?;
        ocl::core::set_kernel_arg(&kernel, 6, ArgVal::scalar(&ne00))?;
        ocl::core::set_kernel_arg(&kernel, 7, ArgVal::scalar(&nb01))?;
        ocl::core::set_kernel_arg(&kernel, 8, ArgVal::scalar(&nb02))?;
        ocl::core::set_kernel_arg(&kernel, 9, ArgVal::scalar(&nb03))?;
        ocl::core::set_kernel_arg(&kernel, 10, ArgVal::scalar(&ne10))?;
        ocl::core::set_kernel_arg(&kernel, 11, ArgVal::scalar(&nb10))?;
        ocl::core::set_kernel_arg(&kernel, 12, ArgVal::scalar(&nb11))?;
        ocl::core::set_kernel_arg(&kernel, 13, ArgVal::scalar(&nb12))?;
        ocl::core::set_kernel_arg(&kernel, 14, ArgVal::scalar(&nb1))?;
        ocl::core::set_kernel_arg(&kernel, 15, ArgVal::scalar(&nb2))?;
        ocl::core::set_kernel_arg(&kernel, 16, ArgVal::scalar(&nb3))?;

        // dispatch: global=[ne10 (output rows), 1, 1], local=[64, 1, 1]
        // Each WG handles one output row, iterating over ne00 elements
        let local_sz = 64usize;
        let global = [ne10 as usize, 1, 1];
        let local = [local_sz.min(1), 1, 1]; // 1 WG per output row

        let samples = measure(ctx, WARMUP, MEASURE_ELEM, || {
            unsafe {
                ocl::core::enqueue_kernel(
                    &ctx.queue,
                    &kernel,
                    3,
                    None,
                    &global,
                    Some(local),
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            Ok(())
        })?;

        print_result(cell_id, &samples);
        Ok(())
    }

    // ── SILU F16 (fused silu_mul) ─────────────────────────────────────────────

    fn bench_silu_mul(ctx: &Ctx, cell_id: &str, dim: usize) -> anyhow::Result<()> {
        let kernel = ocl::core::create_kernel(&ctx.prog_simple, "kernel_silu_mul_simple")?;

        let x = alloc_f32(ctx, dim)?;
        let y = alloc_f32(ctx, dim)?;
        let size4 = (dim / 4) as i32;

        ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&x))?;
        ocl::core::set_kernel_arg(&kernel, 1, ArgVal::mem(&y))?;
        ocl::core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&size4))?;

        let global = [dim / 4, 1, 1];

        let samples = measure(ctx, WARMUP, MEASURE_ELEM, || {
            unsafe {
                ocl::core::enqueue_kernel(
                    &ctx.queue,
                    &kernel,
                    1,
                    None,
                    &global,
                    None,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            Ok(())
        })?;

        print_result(cell_id, &samples);
        Ok(())
    }

    // ── ADD F16 ───────────────────────────────────────────────────────────────

    fn bench_add(ctx: &Ctx, cell_id: &str, dim: usize) -> anyhow::Result<()> {
        let kernel = ocl::core::create_kernel(&ctx.prog_simple, "kernel_add_assign_simple")?;

        let x = alloc_f32(ctx, dim)?;
        let y = alloc_f32(ctx, dim)?;
        let size4 = (dim / 4) as i32;

        ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&x))?;
        ocl::core::set_kernel_arg(&kernel, 1, ArgVal::mem(&y))?;
        ocl::core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&size4))?;

        let global = [dim / 4, 1, 1];

        let samples = measure(ctx, WARMUP, MEASURE_ELEM, || {
            unsafe {
                ocl::core::enqueue_kernel(
                    &ctx.queue,
                    &kernel,
                    1,
                    None,
                    &global,
                    None,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            Ok(())
        })?;

        print_result(cell_id, &samples);
        Ok(())
    }

    // ── SOFT_MAX F16 ─────────────────────────────────────────────────────────

    fn bench_softmax(
        ctx: &Ctx,
        cell_id: &str,
        n_heads: usize,
        ctx_len: usize,
    ) -> anyhow::Result<()> {
        let kernel = ocl::core::create_kernel(&ctx.prog_simple, "kernel_softmax_opt")?;

        let x = alloc_f32(ctx, n_heads * ctx_len)?;
        let local_size = 64usize;
        let local_mem = local_size * std::mem::size_of::<f32>();

        ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&x))?;
        ocl::core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&(ctx_len as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 2, ArgVal::local::<f32>(&local_mem))?;

        let global = [n_heads * local_size, 1, 1];
        let local = [local_size, 1, 1];

        let samples = measure(ctx, WARMUP, MEASURE_ELEM, || {
            unsafe {
                ocl::core::enqueue_kernel(
                    &ctx.queue,
                    &kernel,
                    1,
                    None,
                    &global,
                    Some(local),
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            Ok(())
        })?;

        print_result(cell_id, &samples);
        Ok(())
    }

    // ── SCALE F16 ────────────────────────────────────────────────────────────

    fn bench_scale(ctx: &Ctx, cell_id: &str, size: usize) -> anyhow::Result<()> {
        let kernel = ocl::core::create_kernel(&ctx.prog_simple, "kernel_scale_simple")?;

        let x = alloc_f32(ctx, size)?;
        let val = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let sz = size as i32;

        ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&x))?;
        ocl::core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&val))?;
        ocl::core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&sz))?;

        let global = [size, 1, 1];

        let samples = measure(ctx, WARMUP, MEASURE_ELEM, || {
            unsafe {
                ocl::core::enqueue_kernel(
                    &ctx.queue,
                    &kernel,
                    1,
                    None,
                    &global,
                    None,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            Ok(())
        })?;

        print_result(cell_id, &samples);
        Ok(())
    }

    // ── CPY F16 (F32→F16 cast) ────────────────────────────────────────────────

    fn bench_cpy(ctx: &Ctx, cell_id: &str, n_elem: usize) -> anyhow::Result<()> {
        let kernel = ocl::core::create_kernel(&ctx.prog_simple, "kernel_cast_f32_to_f16")?;

        let src = alloc_f32(ctx, n_elem)?;
        let dst = alloc_f16(ctx, n_elem)?;
        let ne = n_elem as i32;

        ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&src))?;
        ocl::core::set_kernel_arg(&kernel, 1, ArgVal::mem(&dst))?;
        ocl::core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&ne))?;

        let global = [n_elem.div_ceil(64) * 64, 1, 1];
        let local = [64usize, 1, 1];

        let samples = measure(ctx, WARMUP, MEASURE_ELEM, || {
            unsafe {
                ocl::core::enqueue_kernel(
                    &ctx.queue,
                    &kernel,
                    1,
                    None,
                    &global,
                    Some(local),
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            Ok(())
        })?;

        print_result(cell_id, &samples);
        Ok(())
    }

    // ── SET_ROWS / KV scatter ─────────────────────────────────────────────────

    fn bench_set_rows(
        ctx: &Ctx,
        cell_id: &str,
        n_kv_heads: usize,
        head_dim: usize,
        kv_capacity: usize,
    ) -> anyhow::Result<()> {
        let kernel = ocl::core::create_kernel(&ctx.prog_simple, "kernel_kv_scatter_f32_to_f16")?;

        // kernel: k_src, v_src, k_dst, v_dst, head_dim, capacity, write_pos
        let k_src = alloc_f32(ctx, n_kv_heads * head_dim)?;
        let v_src = alloc_f32(ctx, n_kv_heads * head_dim)?;
        let k_dst = alloc_f16(ctx, n_kv_heads * kv_capacity * head_dim)?;
        let v_dst = alloc_f16(ctx, n_kv_heads * kv_capacity * head_dim)?;

        let head_dim_i = head_dim as i32;
        let capacity_i = kv_capacity as i32;
        let write_pos = 42i32;

        ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&k_src))?;
        ocl::core::set_kernel_arg(&kernel, 1, ArgVal::mem(&v_src))?;
        ocl::core::set_kernel_arg(&kernel, 2, ArgVal::mem(&k_dst))?;
        ocl::core::set_kernel_arg(&kernel, 3, ArgVal::mem(&v_dst))?;
        ocl::core::set_kernel_arg(&kernel, 4, ArgVal::scalar(&head_dim_i))?;
        ocl::core::set_kernel_arg(&kernel, 5, ArgVal::scalar(&capacity_i))?;
        ocl::core::set_kernel_arg(&kernel, 6, ArgVal::scalar(&write_pos))?;

        let global = [n_kv_heads * head_dim, 1, 1];

        let samples = measure(ctx, WARMUP, MEASURE_ELEM, || {
            unsafe {
                ocl::core::enqueue_kernel(
                    &ctx.queue,
                    &kernel,
                    1,
                    None,
                    &global,
                    None,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            Ok(())
        })?;

        print_result(cell_id, &samples);
        Ok(())
    }

    // ── filter helper ─────────────────────────────────────────────────────────

    fn should_run(filter: &Option<Vec<String>>, cell_id: &str) -> bool {
        match filter {
            None => true,
            Some(ops) => ops.iter().any(|op| cell_id.starts_with(op.as_str())),
        }
    }

    // ── main entry ────────────────────────────────────────────────────────────

    pub fn run() -> anyhow::Result<()> {
        let args: Vec<String> = std::env::args().collect();

        let filter: Option<Vec<String>> = args
            .iter()
            .position(|a| a == "--ops")
            .and_then(|i| args.get(i + 1))
            .map(|s| s.split(',').map(|x| x.to_string()).collect());

        let ctx = init()?;

        eprintln!("# Compiling kernels...");
        println!("# microbench_opencl_op_matrix ours.gpu Qwen2.5-1.5B");
        println!("# cell_id: ours.gpu.<OP>.<DTYPE>[.<shape_id>]");

        // ── MUL_MAT F16 ──────────────────────────────────────────────────────
        if should_run(&filter, "MUL_MAT_F16") {
            eprintln!("# MUL_MAT F16 mm_ffn K={} N={}", DIM, FFN_DIM);
            bench_mul_mat_f16(&ctx, "ours.gpu.MUL_MAT.F16.mm_ffn", DIM, FFN_DIM)?;

            eprintln!("# MUL_MAT F16 mm_lmh K={} N={}", DIM, VOCAB);
            bench_mul_mat_f16(&ctx, "ours.gpu.MUL_MAT.F16.mm_lmh", DIM, VOCAB)?;

            let qkv_n = N_HEADS_Q * HEAD_DIM + 2 * N_HEADS_KV * HEAD_DIM; // 2048
            eprintln!("# MUL_MAT F16 mm_qkv K={} N={}", DIM, qkv_n);
            bench_mul_mat_f16(&ctx, "ours.gpu.MUL_MAT.F16.mm_qkv", DIM, qkv_n)?;
        }

        // ── MUL_MAT Q4_0 ─────────────────────────────────────────────────────
        if should_run(&filter, "MUL_MAT_Q4_0") {
            eprintln!("# MUL_MAT Q4_0 mm_ffn K={} N={}", DIM, FFN_DIM);
            bench_mul_mat_q4_0(&ctx, "ours.gpu.MUL_MAT.Q4_0.mm_ffn", DIM, FFN_DIM)?;

            eprintln!("# MUL_MAT Q4_0 mm_lmh K={} N={}", DIM, VOCAB);
            bench_mul_mat_q4_0(&ctx, "ours.gpu.MUL_MAT.Q4_0.mm_lmh", DIM, VOCAB)?;

            let qkv_n = N_HEADS_Q * HEAD_DIM + 2 * N_HEADS_KV * HEAD_DIM;
            eprintln!("# MUL_MAT Q4_0 mm_qkv K={} N={}", DIM, qkv_n);
            bench_mul_mat_q4_0(&ctx, "ours.gpu.MUL_MAT.Q4_0.mm_qkv", DIM, qkv_n)?;
        }

        // ── RMS_NORM F16 ─────────────────────────────────────────────────────
        if should_run(&filter, "RMS_NORM_F16") {
            eprintln!("# RMS_NORM F16 dim={}", DIM);
            bench_rms_norm(&ctx, "ours.gpu.RMS_NORM.F16", DIM)?;
        }

        // ── ROPE F16 ─────────────────────────────────────────────────────────
        if should_run(&filter, "ROPE_F16") {
            eprintln!("# ROPE F16 n_heads={} head_dim={}", N_HEADS_Q, HEAD_DIM);
            bench_rope(&ctx, "ours.gpu.ROPE.F16")?;
        }

        // ── FLASH_ATTN_EXT F16 ───────────────────────────────────────────────
        if should_run(&filter, "FLASH_ATTN_EXT_F16") {
            eprintln!(
                "# FLASH_ATTN_EXT F16 hs=128 nh={} nkv={} ctx={}",
                N_HEADS_Q, N_HEADS_KV, CTX_LEN
            );
            match bench_flash_attn(&ctx, "ours.gpu.FLASH_ATTN_EXT.F16") {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("# FLASH_ATTN_EXT error: {} -> marking unsupported", e);
                    println!(
                        "{{\"cell_id\":\"ours.gpu.FLASH_ATTN_EXT.F16\",\"status\":\"unsupported\",\"reason\":\"{}\"}}",
                        e
                    );
                }
            }
        }

        // ── GET_ROWS F16 ─────────────────────────────────────────────────────
        if should_run(&filter, "GET_ROWS_F16") {
            eprintln!("# GET_ROWS F16 vocab={} dim={}", VOCAB, DIM);
            match bench_get_rows(&ctx, "ours.gpu.GET_ROWS.F16") {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("# GET_ROWS error: {}", e);
                    println!(
                        "{{\"cell_id\":\"ours.gpu.GET_ROWS.F16\",\"status\":\"error\",\"reason\":\"{}\"}}",
                        e
                    );
                }
            }
        }

        // ── SILU F16 ─────────────────────────────────────────────────────────
        if should_run(&filter, "SILU_F16") {
            eprintln!("# SILU F16 (silu_mul fused) dim={}", FFN_DIM);
            bench_silu_mul(&ctx, "ours.gpu.SILU.F16", FFN_DIM)?;
        }

        // ── MUL F16 ──────────────────────────────────────────────────────────
        // Production uses the same fused silu_mul kernel for gate×up (SiLU on gate, MUL).
        // Standalone MUL latency = same kernel dispatch with a different cell_id.
        if should_run(&filter, "MUL_F16") {
            eprintln!("# MUL F16 dim={}", FFN_DIM);
            bench_silu_mul(&ctx, "ours.gpu.MUL.F16", FFN_DIM)?;
        }

        // ── ADD F16 ──────────────────────────────────────────────────────────
        if should_run(&filter, "ADD_F16") {
            eprintln!("# ADD F16 dim={}", DIM);
            bench_add(&ctx, "ours.gpu.ADD.F16", DIM)?;
        }

        // ── SOFT_MAX F16 ─────────────────────────────────────────────────────
        if should_run(&filter, "SOFT_MAX_F16") {
            eprintln!("# SOFT_MAX F16 shape=[{},1,{}]", N_HEADS_Q, CTX_LEN);
            bench_softmax(&ctx, "ours.gpu.SOFT_MAX.F16", N_HEADS_Q, CTX_LEN)?;
        }

        // ── SCALE F16 ────────────────────────────────────────────────────────
        if should_run(&filter, "SCALE_F16") {
            let total = N_HEADS_Q * CTX_LEN;
            eprintln!("# SCALE F16 total={}", total);
            bench_scale(&ctx, "ours.gpu.SCALE.F16", total)?;
        }

        // ── CPY F16 ──────────────────────────────────────────────────────────
        if should_run(&filter, "CPY_F16") {
            let n_elem = 2 * HEAD_DIM; // [2, 1, head_dim]
            eprintln!("# CPY F16 (F32→F16) n_elem={}", n_elem);
            bench_cpy(&ctx, "ours.gpu.CPY.F16", n_elem)?;
        }

        // ── SET_ROWS F16 ─────────────────────────────────────────────────────
        if should_run(&filter, "SET_ROWS_F16") {
            eprintln!(
                "# SET_ROWS F16 n_kv={} head_dim={} cap={}",
                N_HEADS_KV, HEAD_DIM, CTX_LEN
            );
            match bench_set_rows(&ctx, "ours.gpu.SET_ROWS.F16", N_HEADS_KV, HEAD_DIM, CTX_LEN) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("# SET_ROWS error: {}", e);
                    println!(
                        "{{\"cell_id\":\"ours.gpu.SET_ROWS.F16\",\"status\":\"error\",\"reason\":\"{}\"}}",
                        e
                    );
                }
            }
        }

        eprintln!("# done");
        Ok(())
    }
}
