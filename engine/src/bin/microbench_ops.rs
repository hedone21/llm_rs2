//! microbench_ops — llm_rs2 decode 경로의 OP-level 커널 cross-engine 벤치
//!
//! 타겟: Qwen 2.5-1.5B Q4_0 decode (seq_len=1, batch=1)
//!   dim=1536, n_heads=12, n_kv_heads=2, head_dim=128,
//!   n_layers=28, ffn_dim=8960, vocab=151936
//!
//! 측정 OP (모두 GEMV / 작은 elementwise):
//!   - Q4_0 matmul: Q/KV/O proj, FFN gate/up, FFN down, lm_head
//!   - small ops:   RMSNorm, RoPE(Q), RoPE(K), SiLU·mul, residual add
//!
//! 엔진 2종을 같은 shape 내에서 교차 실행(A,B,A,B,...)하여 thermal drift 상쇄:
//!   - llm_rs2  : engine/kernels/ production source
//!   - llama.cpp: .agent/research/microbench_ops/ vendored source
//!
//! 써멀 쿨다운:
//!   - 시작 전: /sys/class/thermal/thermal_zone0/temp <= 32000 mC 까지 대기(최대 10분)
//!   - Q4_0 GEMV shape 사이: 15 s
//!   - small op 사이: 5 s
//!
//! 출력: op별 min/median/mean μs + 최종 simulated ms/token 합계 markdown 표.
//!
//! 참고: 커널 .cl 파일은 수정하지 않음(CLAUDE.md 규칙). 템플릿:
//!   engine/src/bin/microbench_flash_attn.rs

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("microbench_ops requires --features opencl");
    std::process::exit(2);
}

#[cfg(feature = "opencl")]
mod bench {
    use ocl::core::{ArgVal, Event, Mem, ProfilingInfo};
    use ocl::flags;
    use ocl::{Context, Device, Platform, Program, Queue};
    use std::time::{Duration, Instant};

    // === Qwen 2.5-1.5B ===
    pub const DIM: usize = 1536;
    pub const N_HEADS_Q: usize = 12;
    pub const N_HEADS_KV: usize = 2;
    pub const HEAD_DIM: usize = 128;
    pub const N_LAYERS: usize = 28;
    pub const FFN_DIM: usize = 8960;
    pub const VOCAB: usize = 151936;

    pub const WARMUP: usize = 3;
    pub const ITERS: usize = 10;

    pub const QK4_0: usize = 32;
    pub const Q4_0_BLOCK_BYTES: usize = 18; // half(d) + 16 uchars

    /// Q4_0 matmul shape: y[N] = W[N,K] @ x[K]
    #[derive(Clone, Copy)]
    pub struct Q4Shape {
        pub name: &'static str,
        pub k: usize,
        pub n: usize,
        pub calls_per_tok: usize,
    }

    pub const Q4_SHAPES: &[Q4Shape] = &[
        Q4Shape {
            name: "Q proj",
            k: DIM,
            n: DIM,
            calls_per_tok: N_LAYERS,
        },
        Q4Shape {
            name: "K/V proj",
            k: DIM,
            n: N_HEADS_KV * HEAD_DIM, // 256
            calls_per_tok: 2 * N_LAYERS,
        },
        Q4Shape {
            name: "O proj",
            k: DIM,
            n: DIM,
            calls_per_tok: N_LAYERS,
        },
        Q4Shape {
            name: "gate/up",
            k: DIM,
            n: FFN_DIM,
            calls_per_tok: 2 * N_LAYERS,
        },
        Q4Shape {
            name: "down",
            k: FFN_DIM,
            n: DIM,
            calls_per_tok: N_LAYERS,
        },
        Q4Shape {
            name: "lm_head",
            k: DIM,
            n: VOCAB,
            calls_per_tok: 1,
        },
    ];

    /// 하나의 benchmark 결과 (한 engine × 한 OP).
    pub struct OpResult {
        pub engine: &'static str,
        pub op: String,
        pub shape: String,
        pub calls_per_tok: usize,
        pub samples: Vec<f64>, // μs
    }

    impl OpResult {
        pub fn min(&self) -> f64 {
            self.samples.iter().cloned().fold(f64::INFINITY, f64::min)
        }
        pub fn mean(&self) -> f64 {
            self.samples.iter().sum::<f64>() / self.samples.len() as f64
        }
        pub fn median(&self) -> f64 {
            let mut s = self.samples.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[s.len() / 2]
        }
    }

    pub struct State {
        pub context: Context,
        pub device: Device,
        pub queue: Queue,
        pub cl_opts: String,
        pub our_program: Program,
        pub llama_program_q4: Program,
        pub llama_program_rms: Program,
        pub llama_program_rope: Program,
        pub llama_program_silu: Program,
        pub llama_program_mul: Program,
        pub llama_program_add: Program,
        /// llama.cpp Ab_Bi_8x4 FFN kernel — new production kernel
        pub llama_program_ab_bi: Program,
        /// noshuffle GEMV program cache keyed by (ne00, ne01) — recompiled per-shape
        /// because LINE_STRIDE_A / BLOCK_STRIDE_A are compile-time defines.
        pub noshuffle_cache: std::cell::RefCell<
            std::collections::HashMap<(usize, usize), (Program, ocl::core::Kernel)>,
        >,
    }

    fn detect_cl_c2(device: &Device) -> bool {
        let v = device
            .info(ocl::core::DeviceInfo::OpenclCVersion)
            .map(|v| v.to_string())
            .unwrap_or_default();
        v.split_whitespace()
            .nth(2)
            .and_then(|ver| {
                let mut parts = ver.split('.');
                let major: u32 = parts.next()?.parse().ok()?;
                let minor: u32 = parts.next()?.parse().ok()?;
                Some((major, minor) >= (2, 0))
            })
            .unwrap_or(false)
    }

    pub fn init() -> anyhow::Result<State> {
        let platform = Platform::default();
        let device = Device::first(platform)?;
        println!("# microbench_ops — Qwen 2.5-1.5B Q4_0 decode OP breakdown");
        println!(
            "# Platform: {}",
            platform.name().unwrap_or_else(|_| "unknown".into())
        );
        println!(
            "# Device:   {}",
            device.name().unwrap_or_else(|_| "unknown".into())
        );

        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;
        let queue = Queue::new(&context, device, Some(flags::QUEUE_PROFILING_ENABLE))?;

        let cl2 = detect_cl_c2(&device);
        let fast_math = "-cl-mad-enable -cl-unsafe-math-optimizations \
                         -cl-finite-math-only -cl-fast-relaxed-math";
        let _ = cl2;
        let cl_opts = if cl2 {
            format!("-cl-std=CL2.0 {}", fast_math)
        } else {
            fast_math.to_string()
        };
        println!("# Build opts: {}", cl_opts);

        // llm_rs2 커널들 (production source)
        let our_src = format!(
            "{}\n{}\n",
            include_str!("../../kernels/simple_ops.cl"),
            include_str!("../../kernels/mul_mv_q4_0_f32.cl"),
        );
        let our_program = Program::builder()
            .devices(device)
            .src(our_src)
            .cmplr_opt(&cl_opts)
            .build(&context)?;

        // llama.cpp 벤더 커널 — 파일별로 별도 program (각각 fp16 pragma 헤더 포함)
        let llama_q4_src =
            include_str!("../../../.agent/research/microbench_ops/mul_mv_q4_0_f32_8x_flat.cl");
        let llama_program_q4 = Program::builder()
            .devices(device)
            .src(llama_q4_src)
            .cmplr_opt(&cl_opts)
            .build(&context)?;

        let llama_rms_src = include_str!("../../../.agent/research/microbench_ops/rms_norm.cl");
        let llama_program_rms = Program::builder()
            .devices(device)
            .src(llama_rms_src)
            .cmplr_opt(&cl_opts)
            .build(&context)?;

        let llama_rope_src = include_str!("../../../.agent/research/microbench_ops/rope.cl");
        let llama_program_rope = Program::builder()
            .devices(device)
            .src(llama_rope_src)
            .cmplr_opt(&cl_opts)
            .build(&context)?;

        let llama_silu_src = include_str!("../../../.agent/research/microbench_ops/silu.cl");
        let llama_program_silu = Program::builder()
            .devices(device)
            .src(llama_silu_src)
            .cmplr_opt(&cl_opts)
            .build(&context)?;

        let llama_mul_src = include_str!("../../../.agent/research/microbench_ops/mul.cl");
        let llama_program_mul = Program::builder()
            .devices(device)
            .src(llama_mul_src)
            .cmplr_opt(&cl_opts)
            .build(&context)?;

        let llama_add_src = include_str!("../../../.agent/research/microbench_ops/add.cl");
        let llama_program_add = Program::builder()
            .devices(device)
            .src(llama_add_src)
            .cmplr_opt(&cl_opts)
            .build(&context)?;

        let llama_ab_bi_src =
            include_str!("../../../.agent/research/microbench_ops/mul_mat_Ab_Bi_8x4.cl");
        let llama_program_ab_bi = Program::builder()
            .devices(device)
            .src(llama_ab_bi_src)
            .cmplr_opt(&cl_opts)
            .build(&context)?;

        Ok(State {
            context,
            device,
            queue,
            cl_opts,
            our_program,
            llama_program_q4,
            llama_program_rms,
            llama_program_rope,
            llama_program_silu,
            llama_program_mul,
            llama_program_add,
            llama_program_ab_bi,
            noshuffle_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
        })
    }

    // ---------- thermal helpers ----------

    fn read_thermal_mc() -> Option<i64> {
        std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp")
            .ok()
            .and_then(|s| s.trim().parse::<i64>().ok())
    }

    pub fn wait_for_cooldown(threshold_mc: i64, max_wait: Duration) {
        let start = Instant::now();
        loop {
            match read_thermal_mc() {
                Some(t) if t <= threshold_mc => {
                    println!("# cooldown OK: {}°C", t as f64 / 1000.0);
                    return;
                }
                Some(t) => {
                    if start.elapsed() > max_wait {
                        eprintln!(
                            "# WARN: cooldown timeout after {}s, current temp {}°C, proceeding anyway",
                            start.elapsed().as_secs(),
                            t as f64 / 1000.0
                        );
                        return;
                    }
                    println!(
                        "# waiting cooldown: {}°C > {}°C (threshold), elapsed {}s",
                        t as f64 / 1000.0,
                        threshold_mc as f64 / 1000.0,
                        start.elapsed().as_secs()
                    );
                    std::thread::sleep(Duration::from_secs(5));
                }
                None => {
                    eprintln!(
                        "# WARN: /sys/class/thermal/thermal_zone0/temp unavailable, skipping cooldown"
                    );
                    return;
                }
            }
        }
    }

    pub fn sleep_quiet(d: Duration) {
        if d.as_secs() > 0 {
            println!("# sleep {}s", d.as_secs());
        }
        std::thread::sleep(d);
    }

    // ---------- buffer helpers ----------

    pub fn alloc_bytes(ctx: &Context, bytes: usize) -> anyhow::Result<Mem> {
        // READ_WRITE 의도적으로 host-ptr 미사용 (perf만 측정, 데이터는 random OK).
        let buf = unsafe {
            ocl::core::create_buffer::<_, u8>(
                ctx.as_core(),
                ocl::core::MEM_READ_WRITE,
                bytes,
                None,
            )?
        };
        Ok(buf)
    }

    pub fn alloc_f32(ctx: &Context, count: usize) -> anyhow::Result<Mem> {
        let buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                ctx.as_core(),
                ocl::core::MEM_READ_WRITE,
                count,
                None,
            )?
        };
        Ok(buf)
    }

    /// 랜덤 바이트로 초기화.
    pub fn fill_random_bytes(
        queue: &Queue,
        buf: &Mem,
        bytes: usize,
        seed: u64,
    ) -> anyhow::Result<()> {
        let mut s = seed.wrapping_mul(6364136223846793005);
        let mut data = vec![0u8; bytes];
        for b in data.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *b = (s >> 33) as u8;
        }
        unsafe {
            ocl::core::enqueue_write_buffer(
                queue,
                buf,
                true,
                0,
                &data,
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
        Ok(())
    }

    /// Single dispatch 이벤트 timing — μs 반환.
    pub fn time_event(ev: &Event) -> anyhow::Result<f64> {
        let s = ocl::core::get_event_profiling_info(ev, ProfilingInfo::Start)?.time()?;
        let e = ocl::core::get_event_profiling_info(ev, ProfilingInfo::End)?.time()?;
        Ok((e.saturating_sub(s)) as f64 / 1000.0)
    }

    // ---------- Q4_0 matmul benchmark ----------

    /// llm_rs2 커널 한 번 dispatch. Arg 시그니처:
    ///   (src0, offset0, src1, offset1, dst, offsetd, ne00, ne01, ne02,
    ///    ne10, ne12, ne0, ne1, r2, r3)
    /// gws=[ceil_div(N,4)*64, M, 1], lws=[64,1,1]
    fn dispatch_our_q4(
        queue: &Queue,
        kernel: &ocl::core::Kernel,
        weight: &Mem,
        x: &Mem,
        out: &Mem,
        k: usize,
        n: usize,
    ) -> anyhow::Result<Event> {
        let m = 1i32;
        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1i32;
        let ne10 = k as i32;
        let ne12 = (k * m as usize) as i32;
        let ne0 = n as i32;
        let ne1 = (n * m as usize) as i32;
        let r2 = 1i32;
        let r3 = 1i32;
        let zero_u64 = 0u64;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(weight))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::mem(x))?;
            ocl::core::set_kernel_arg(kernel, 3, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ArgVal::mem(out))?;
            ocl::core::set_kernel_arg(kernel, 5, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 6, ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(kernel, 7, ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(kernel, 8, ArgVal::scalar(&ne02))?;
            ocl::core::set_kernel_arg(kernel, 9, ArgVal::scalar(&ne10))?;
            ocl::core::set_kernel_arg(kernel, 10, ArgVal::scalar(&ne12))?;
            ocl::core::set_kernel_arg(kernel, 11, ArgVal::scalar(&ne0))?;
            ocl::core::set_kernel_arg(kernel, 12, ArgVal::scalar(&ne1))?;
            ocl::core::set_kernel_arg(kernel, 13, ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(kernel, 14, ArgVal::scalar(&r3))?;

            let lws: [usize; 3] = [64, 1, 1];
            let gws: [usize; 3] = [n.div_ceil(4) * 64, m as usize, 1];
            let mut ev = Event::null();
            ocl::core::enqueue_kernel(
                queue,
                kernel,
                3,
                None,
                &gws,
                Some(lws),
                None::<&Event>,
                Some(&mut ev),
            )?;
            Ok(ev)
        }
    }

    /// llama.cpp 8x_flat dispatch. Arg 시그니처:
    ///   (src0_q, src0_d, src1, offset1, dst, offsetd, ne00, ne01, ne02,
    ///    ne10, ne12, ne0, ne1, r2, r3)
    /// gws=[ceil_div(N,8)*64, 1, 1], lws=[64,1,1]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_llama_q4(
        queue: &Queue,
        kernel: &ocl::core::Kernel,
        src0_q: &Mem,
        src0_d: &Mem,
        x: &Mem,
        out: &Mem,
        k: usize,
        n: usize,
    ) -> anyhow::Result<Event> {
        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02 = 1i32;
        let ne10 = k as i32;
        let ne12 = 1i32;
        let ne0 = n as i32;
        let ne1 = 1i32;
        let r2 = 1i32;
        let r3 = 1i32;
        let zero_u64 = 0u64;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(src0_q))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::mem(src0_d))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::mem(x))?;
            ocl::core::set_kernel_arg(kernel, 3, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ArgVal::mem(out))?;
            ocl::core::set_kernel_arg(kernel, 5, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 6, ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(kernel, 7, ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(kernel, 8, ArgVal::scalar(&ne02))?;
            ocl::core::set_kernel_arg(kernel, 9, ArgVal::scalar(&ne10))?;
            ocl::core::set_kernel_arg(kernel, 10, ArgVal::scalar(&ne12))?;
            ocl::core::set_kernel_arg(kernel, 11, ArgVal::scalar(&ne0))?;
            ocl::core::set_kernel_arg(kernel, 12, ArgVal::scalar(&ne1))?;
            ocl::core::set_kernel_arg(kernel, 13, ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(kernel, 14, ArgVal::scalar(&r3))?;

            let lws: [usize; 3] = [64, 1, 1];
            let gws: [usize; 3] = [n.div_ceil(8) * 64, 1, 1];
            let mut ev = Event::null();
            ocl::core::enqueue_kernel(
                queue,
                kernel,
                3,
                None,
                &gws,
                Some(lws),
                None::<&Event>,
                Some(&mut ev),
            )?;
            Ok(ev)
        }
    }

    // ---------- noshuffle (production) Q4_0 GEMV ----------

    /// Build / cache the noshuffle GEMV program for (ne00, ne01). Returns a
    /// freshly-created kernel (ocl::core::Kernel isn't Clone, so each call site
    /// gets its own handle from the cached Program).
    ///
    /// LINE_STRIDE_A and BLOCK_STRIDE_A are compile-time defines, so every
    /// unique shape needs its own program.
    fn noshuffle_kernel_for(
        state: &State,
        ne00: usize,
        ne01: usize,
    ) -> anyhow::Result<ocl::core::Kernel> {
        let key = (ne00, ne01);
        {
            let cache = state.noshuffle_cache.borrow();
            if let Some((program, _)) = cache.get(&key) {
                return Ok(ocl::core::create_kernel(
                    program,
                    "kernel_gemv_noshuffle_q4_0",
                )?);
            }
        }

        let line_stride_a = ne01 / 2;
        let block_stride_a = 4 * ne01;
        let simdgroup_width: usize = 64;
        let gemv_src = include_str!("../../kernels/gemv_noshuffle_q4_0.cl");
        let defines = format!(
            "{} -DLINE_STRIDE_A={} -DBLOCK_STRIDE_A={} -DSIMDGROUP_WIDTH={}",
            state.cl_opts, line_stride_a, block_stride_a, simdgroup_width
        );
        // Try with VECTOR_SUB_GROUP_BROADCAT first (Adreno 830+).
        let defines_vec = format!("{} -DVECTOR_SUB_GROUP_BROADCAT", defines);
        let program = match Program::builder()
            .devices(state.device)
            .src(gemv_src)
            .cmplr_opt(&defines_vec)
            .build(&state.context)
        {
            Ok(p) => {
                println!(
                    "# noshuffle GEMV (ne00={}, ne01={}): vector sub_group_broadcast enabled",
                    ne00, ne01
                );
                p
            }
            Err(_) => {
                println!(
                    "# noshuffle GEMV (ne00={}, ne01={}): scalar sub_group_broadcast fallback",
                    ne00, ne01
                );
                Program::builder()
                    .devices(state.device)
                    .src(gemv_src)
                    .cmplr_opt(&defines)
                    .build(&state.context)?
            }
        };

        let kernel_cached = ocl::core::create_kernel(&program, "kernel_gemv_noshuffle_q4_0")?;
        let kernel_out = ocl::core::create_kernel(&program, "kernel_gemv_noshuffle_q4_0")?;
        state
            .noshuffle_cache
            .borrow_mut()
            .insert(key, (program, kernel_cached));
        Ok(kernel_out)
    }

    /// Allocate SOA q/d buffers and an R32UI image1d_buffer over q, matching
    /// `matmul_q4_0_noshuffle()` layout expectations. Data is random (perf only).
    ///
    /// Returns (q_buf, d_buf, q_img).
    /// `q_img` is None when image1d_buffer creation fails (size limit, unsupported).
    fn alloc_noshuffle_weights(
        state: &State,
        ne00: usize,
        ne01: usize,
        seed: u64,
    ) -> anyhow::Result<(Mem, Mem, Option<Mem>)> {
        let num_blocks = ne01 * (ne00 / QK4_0);
        // q buffer: num_blocks * 16 bytes (same size as the SOA nibble buffer after transpose).
        let q_bytes = num_blocks * 16;
        let q_buf = alloc_bytes(&state.context, q_bytes)?;
        fill_random_bytes(&state.queue, &q_buf, q_bytes, seed ^ 0xE1E1)?;

        // d buffer: num_blocks * 2 bytes (half per (row, kblock) pair, SOA).
        let d_bytes = num_blocks * 2;
        let d_buf = alloc_bytes(&state.context, d_bytes)?;
        fill_random_bytes(&state.queue, &d_buf, d_bytes, seed ^ 0xE2E2)?;

        // image1d_buffer over q (R32UI): width = q_bytes / 4 uint texels.
        let q_total_uint = q_bytes / 4;
        let q_img = {
            use ocl::core::{
                ImageChannelDataType, ImageChannelOrder, ImageDescriptor, ImageFormat,
                MemObjectType,
            };
            let fmt = ImageFormat::new(ImageChannelOrder::R, ImageChannelDataType::UnsignedInt32);
            let desc = ImageDescriptor::new(
                MemObjectType::Image1dBuffer,
                q_total_uint,
                0,
                0,
                0,
                0,
                0,
                Some(q_buf.clone()),
            );
            let res = unsafe {
                ocl::core::create_image(
                    state.context.as_core(),
                    ocl::core::MEM_READ_ONLY,
                    &fmt,
                    &desc,
                    None::<&[u32]>,
                    None,
                )
            };
            match res {
                Ok(img) => Some(img),
                Err(e) => {
                    eprintln!(
                        "# WARN: image1d_buffer_t (R32UI, width={}) creation failed for \
                         noshuffle ne00={}, ne01={}: {} — skipping noshuffle for this shape",
                        q_total_uint, ne00, ne01, e
                    );
                    None
                }
            }
        };
        Ok((q_buf, d_buf, q_img))
    }

    /// Dispatch the noshuffle GEMV kernel. Creates (and drops) an ephemeral
    /// RGBA32F image1d_buffer over the activation buffer, matching
    /// `matmul_q4_0_noshuffle()`.
    ///
    /// Returns None if the activation image cannot be created.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_our_noshuffle(
        state: &State,
        kernel: &ocl::core::Kernel,
        q_img: &Mem,
        d_buf: &Mem,
        act_buf: &Mem,
        out_buf: &Mem,
        ne00: usize,
        ne01: usize,
    ) -> anyhow::Result<Option<Event>> {
        // ephemeral RGBA32F image over activation (width = ne00/4 float4 texels)
        let act_img = {
            use ocl::core::{
                ImageChannelDataType, ImageChannelOrder, ImageDescriptor, ImageFormat,
                MemObjectType,
            };
            let fmt = ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelDataType::Float);
            let desc = ImageDescriptor::new(
                MemObjectType::Image1dBuffer,
                ne00 / 4,
                0,
                0,
                0,
                0,
                0,
                Some(act_buf.clone()),
            );
            let res = unsafe {
                ocl::core::create_image(
                    state.context.as_core(),
                    ocl::core::MEM_READ_ONLY,
                    &fmt,
                    &desc,
                    None::<&[f32]>,
                    None,
                )
            };
            match res {
                Ok(img) => img,
                Err(e) => {
                    eprintln!(
                        "# WARN: activation image1d_buffer_t (RGBA32F, width={}) \
                         creation failed: {} — skipping noshuffle dispatch",
                        ne00 / 4,
                        e
                    );
                    return Ok(None);
                }
            }
        };

        let ne00_i = ne00 as i32;
        let ne01_i = ne01 as i32;
        let simdgroup_width: usize = 64;
        let n_simdgroup: usize = 4;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(q_img))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::mem(d_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::mem(&act_img))?;
            ocl::core::set_kernel_arg(kernel, 3, ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(kernel, 4, ArgVal::scalar(&ne00_i))?;
            ocl::core::set_kernel_arg(kernel, 5, ArgVal::scalar(&ne01_i))?;

            let gws: [usize; 3] = [ne01 / 2, n_simdgroup, 1];
            let lws: [usize; 3] = [simdgroup_width, n_simdgroup, 1];
            let mut ev = Event::null();
            ocl::core::enqueue_kernel(
                &state.queue,
                kernel,
                2,
                None,
                &gws,
                Some(lws),
                None::<&Event>,
                Some(&mut ev),
            )?;
            // act_img dropped at end of scope (clReleaseMemObject); the enqueued
            // kernel retains its own reference until it completes.
            Ok(Some(ev))
        }
    }

    // ---------- llama.cpp Ab_Bi_8x4 (new FFN GEMM) ----------

    /// Allocate Ab_Bi weight buffers:
    ///   q: ushort[K/4][N]  — 4 nibbles per ushort, transposed layout
    ///   d: half[K/32][N]   — 1 scale per 32-element block per row
    /// Random data — only timing matters.
    fn alloc_ab_bi_weights(
        state: &State,
        k: usize,
        n: usize,
        seed: u64,
    ) -> anyhow::Result<(Mem, Mem)> {
        let q_bytes = (k / 4) * n * 2;
        let d_bytes = (k / 32) * n * 2;
        let q_buf = alloc_bytes(&state.context, q_bytes)?;
        fill_random_bytes(&state.queue, &q_buf, q_bytes, seed ^ 0xAB1A)?;
        let d_buf = alloc_bytes(&state.context, d_bytes)?;
        fill_random_bytes(&state.queue, &d_buf, d_bytes, seed ^ 0xAB1D)?;
        Ok((q_buf, d_buf))
    }

    /// Dispatch llama's kernel_mul_mat_Ab_Bi_8x4.
    ///
    /// Kernel signature (decode, N=1 padded to 4):
    ///   args: (q, d, src1_image RGBA16F, dst f32, m=N, n=4, k=K, n_no_padding=1)
    ///   gws=[n_4, m_4, 1] = [1, N/4, 1],  lws=[1, 128, 1]  (subgroup 128)
    #[allow(clippy::too_many_arguments)]
    fn dispatch_llama_ab_bi(
        state: &State,
        kernel: &ocl::core::Kernel,
        q_buf: &Mem,
        d_buf: &Mem,
        act_buf_half4: &Mem, // K half4 texels of activation (padded N=4)
        out_buf: &Mem,
        k: usize,
        n: usize,
    ) -> anyhow::Result<Event> {
        // Activation image: RGBA16F, width = K texels (each holds 4 halves for N=4).
        let act_img = {
            use ocl::core::{
                ImageChannelDataType, ImageChannelOrder, ImageDescriptor, ImageFormat,
                MemObjectType,
            };
            let fmt = ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelDataType::HalfFloat);
            let desc = ImageDescriptor::new(
                MemObjectType::Image1dBuffer,
                k,
                0,
                0,
                0,
                0,
                0,
                Some(act_buf_half4.clone()),
            );
            unsafe {
                ocl::core::create_image(
                    state.context.as_core(),
                    ocl::core::MEM_READ_ONLY,
                    &fmt,
                    &desc,
                    None::<&[u16]>,
                    None,
                )?
            }
        };

        // N padded to 4 for decode
        let m_i = n as i32; // m = output dim
        let n_padded_i = 4i32;
        let k_i = k as i32;
        let n_no_padding_i = 1i32;

        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(q_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::mem(d_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::mem(&act_img))?;
            ocl::core::set_kernel_arg(kernel, 3, ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(kernel, 4, ArgVal::scalar(&m_i))?;
            ocl::core::set_kernel_arg(kernel, 5, ArgVal::scalar(&n_padded_i))?;
            ocl::core::set_kernel_arg(kernel, 6, ArgVal::scalar(&k_i))?;
            ocl::core::set_kernel_arg(kernel, 7, ArgVal::scalar(&n_no_padding_i))?;

            // gws = [n_4, m_4, 1] = [1, m/4, 1]; lws = [1, 128, 1] (subgroup 128 on Adreno).
            let gws: [usize; 3] = [1, n / 4, 1];
            let lws: [usize; 3] = [1, 128, 1];
            let mut ev = Event::null();
            ocl::core::enqueue_kernel(
                &state.queue,
                kernel,
                2,
                None,
                &gws,
                Some(lws),
                None::<&Event>,
                Some(&mut ev),
            )?;
            Ok(ev)
        }
    }

    /// Run llama.cpp Ab_Bi_8x4 for the given shape. Returns per-dispatch μs samples.
    /// Activations in half4 format (padded N=4) — random data, timing-only.
    pub fn bench_ab_bi_shape(state: &State, shape: Q4Shape) -> anyhow::Result<OpResult> {
        let Q4Shape {
            name,
            k,
            n,
            calls_per_tok,
        } = shape;

        let seed = 0xB100 ^ (k as u64) ^ (n as u64);
        let (q_buf, d_buf) = alloc_ab_bi_weights(state, k, n, seed)?;

        // Activation buffer: K half4 texels, random fill (timing only).
        let act_bytes = k * 8;
        let act_buf = alloc_bytes(&state.context, act_bytes)?;
        fill_random_bytes(&state.queue, &act_buf, act_bytes, seed ^ 0xAC7)?;

        let out_buf = alloc_f32(&state.context, n * 4)?; // m * n_padded

        let kernel =
            ocl::core::create_kernel(&state.llama_program_ab_bi, "kernel_mul_mat_Ab_Bi_8x4")?;

        // Warmup
        for _ in 0..WARMUP {
            let ev =
                dispatch_llama_ab_bi(state, &kernel, &q_buf, &d_buf, &act_buf, &out_buf, k, n)?;
            ocl::core::finish(&state.queue)?;
            let _ = time_event(&ev);
        }

        let mut samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let ev =
                dispatch_llama_ab_bi(state, &kernel, &q_buf, &d_buf, &act_buf, &out_buf, k, n)?;
            ocl::core::finish(&state.queue)?;
            samples.push(time_event(&ev)?);
        }

        Ok(OpResult {
            engine: "llama_ab_bi",
            op: name.to_string(),
            shape: format!("K={}, N={}", k, n),
            calls_per_tok,
            samples,
        })
    }

    pub fn bench_q4_shape(
        state: &State,
        shape: Q4Shape,
    ) -> anyhow::Result<(OpResult, OpResult, Option<OpResult>)> {
        let Q4Shape {
            name,
            k,
            n,
            calls_per_tok,
        } = shape;

        // Weight buffers
        // AOS (llm_rs2): (K/32) * 18 bytes per row, n rows
        let n_blocks_per_row = k / QK4_0;
        let aos_bytes = n * n_blocks_per_row * Q4_0_BLOCK_BYTES;
        let w_aos = alloc_bytes(&state.context, aos_bytes)?;
        fill_random_bytes(
            &state.queue,
            &w_aos,
            aos_bytes,
            0xA0A0 ^ (k as u64) ^ (n as u64),
        )?;

        // SOA (llama.cpp): 분리된 q (uchar, 16 bytes/block) + d (half, 2 bytes/block)
        let total_blocks = n * n_blocks_per_row;
        let soa_q_bytes = total_blocks * 16;
        let soa_d_bytes = total_blocks * 2;
        let w_soa_q = alloc_bytes(&state.context, soa_q_bytes)?;
        let w_soa_d = alloc_bytes(&state.context, soa_d_bytes)?;
        fill_random_bytes(
            &state.queue,
            &w_soa_q,
            soa_q_bytes,
            0xB0B0 ^ (k as u64) ^ (n as u64),
        )?;
        fill_random_bytes(
            &state.queue,
            &w_soa_d,
            soa_d_bytes,
            0xC0C0 ^ (k as u64) ^ (n as u64),
        )?;

        // x [K], out [N]
        let x_buf = alloc_f32(&state.context, k)?;
        let out_buf = alloc_f32(&state.context, n)?;
        fill_random_bytes(&state.queue, &x_buf, k * 4, 0xD0D0)?;

        // kernel handles
        let our_kernel = ocl::core::create_kernel(&state.our_program, "kernel_mul_mat_q4_0_f32")?;
        let llama_kernel =
            ocl::core::create_kernel(&state.llama_program_q4, "kernel_mul_mat_q4_0_f32_8x_flat")?;

        // Production noshuffle path: random SOA q/d + image1d_buffer over q.
        // ne01 must be even for the 2-row-per-fiber dispatch; all Q4_SHAPES are even.
        let noshuffle_ready = n % 2 == 0;
        let noshuffle_data = if noshuffle_ready {
            let seed = 0xD4D4 ^ (k as u64) ^ (n as u64);
            let (q_buf, d_buf, q_img) = alloc_noshuffle_weights(state, k, n, seed)?;
            q_img.map(|img| (q_buf, d_buf, img))
        } else {
            None
        };
        let noshuffle_kernel = if noshuffle_data.is_some() {
            Some(noshuffle_kernel_for(state, k, n)?)
        } else {
            None
        };

        // Warmup
        for _ in 0..WARMUP {
            let ev = dispatch_our_q4(&state.queue, &our_kernel, &w_aos, &x_buf, &out_buf, k, n)?;
            ocl::core::finish(&state.queue)?;
            let _ = time_event(&ev);
            let ev = dispatch_llama_q4(
                &state.queue,
                &llama_kernel,
                &w_soa_q,
                &w_soa_d,
                &x_buf,
                &out_buf,
                k,
                n,
            )?;
            ocl::core::finish(&state.queue)?;
            let _ = time_event(&ev);
            if let (Some((_, d_buf, q_img)), Some(ns_kernel)) =
                (noshuffle_data.as_ref(), noshuffle_kernel.as_ref())
                && let Some(ev_ns) =
                    dispatch_our_noshuffle(state, ns_kernel, q_img, d_buf, &x_buf, &out_buf, k, n)?
            {
                ocl::core::finish(&state.queue)?;
                let _ = time_event(&ev_ns);
            }
        }

        // Cross-run: ours-aos, ours-noshuffle, llama-8x_flat round-robin.
        let mut our_samples = Vec::with_capacity(ITERS);
        let mut noshuffle_samples: Option<Vec<f64>> =
            if noshuffle_data.is_some() && noshuffle_kernel.is_some() {
                Some(Vec::with_capacity(ITERS))
            } else {
                None
            };
        let mut llama_samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let ev_a = dispatch_our_q4(&state.queue, &our_kernel, &w_aos, &x_buf, &out_buf, k, n)?;
            ocl::core::finish(&state.queue)?;
            our_samples.push(time_event(&ev_a)?);

            if let (Some((_, d_buf, q_img)), Some(ns_kernel), Some(samples)) = (
                noshuffle_data.as_ref(),
                noshuffle_kernel.as_ref(),
                noshuffle_samples.as_mut(),
            ) && let Some(ev_ns) =
                dispatch_our_noshuffle(state, ns_kernel, q_img, d_buf, &x_buf, &out_buf, k, n)?
            {
                ocl::core::finish(&state.queue)?;
                samples.push(time_event(&ev_ns)?);
            }

            let ev_b = dispatch_llama_q4(
                &state.queue,
                &llama_kernel,
                &w_soa_q,
                &w_soa_d,
                &x_buf,
                &out_buf,
                k,
                n,
            )?;
            ocl::core::finish(&state.queue)?;
            llama_samples.push(time_event(&ev_b)?);
        }

        let shape_str = format!("K={}, N={}", k, n);
        let noshuffle_result =
            noshuffle_samples
                .filter(|s| !s.is_empty())
                .map(|samples| OpResult {
                    engine: "llm_rs2_noshuffle",
                    op: name.to_string(),
                    shape: shape_str.clone(),
                    calls_per_tok,
                    samples,
                });
        Ok((
            OpResult {
                engine: "llm_rs2",
                op: name.to_string(),
                shape: shape_str.clone(),
                calls_per_tok,
                samples: our_samples,
            },
            OpResult {
                engine: "llama.cpp",
                op: name.to_string(),
                shape: shape_str,
                calls_per_tok,
                samples: llama_samples,
            },
            noshuffle_result,
        ))
    }

    // ---------- RMSNorm ----------

    /// llm_rs2: kernel_rms_norm_oop(x, out, w, dim, eps, add_unit, local scratch)
    /// gws=[rows*64, 1, 1], lws=[64,1,1]. rows=1 for decode.
    fn dispatch_our_rms(
        queue: &Queue,
        kernel: &ocl::core::Kernel,
        x: &Mem,
        out: &Mem,
        w: &Mem,
        dim: usize,
    ) -> anyhow::Result<Event> {
        let local_size = 64usize;
        let local_mem_bytes = local_size * std::mem::size_of::<f32>();
        let eps = 1e-5f32;
        let add_unit: i32 = 0;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(x))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::mem(out))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::mem(w))?;
            ocl::core::set_kernel_arg(kernel, 3, ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(kernel, 4, ArgVal::scalar(&eps))?;
            ocl::core::set_kernel_arg(kernel, 5, ArgVal::scalar(&add_unit))?;
            ocl::core::set_kernel_arg(kernel, 6, ArgVal::local::<f32>(&local_mem_bytes))?;
            let gws: [usize; 3] = [local_size, 1, 1];
            let lws: [usize; 3] = [local_size, 1, 1];
            let mut ev = Event::null();
            ocl::core::enqueue_kernel(
                queue,
                kernel,
                1,
                None,
                &gws,
                Some(lws),
                None::<&Event>,
                Some(&mut ev),
            )?;
            Ok(ev)
        }
    }

    /// llama.cpp: kernel_rms_norm(src0, off0, dst, offd, ne00, ne01, ne02, ne03,
    ///                             nb01, nb02, nb03, eps, local sum)
    /// gws=[ne01*64, ne02, ne03], lws=[64,1,1]. Local mem = sizeof(float)*nth/sgs.
    ///
    /// 주의: 이 커널은 weight를 곱하지 않는다(pure RMS scale). llm_rs2는 weight 곱 포함.
    /// 본 벤치는 측정 대상 커널의 latency만 비교한다(사양에 따른 지정).
    fn dispatch_llama_rms(
        queue: &Queue,
        kernel: &ocl::core::Kernel,
        src0: &Mem,
        dst: &Mem,
        dim: usize,
    ) -> anyhow::Result<Event> {
        let ne00 = dim as i32;
        let ne01 = 1i32;
        let ne02 = 1i32;
        let ne03 = 1i32;
        let nb01 = (dim * 4) as u64;
        let nb02 = nb01;
        let nb03 = nb01;
        let eps = 1e-5f32;
        let nth = 64usize;
        let sgs = 64usize;
        let local_mem_bytes = (nth / sgs) * std::mem::size_of::<f32>();
        let zero_u64 = 0u64;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(src0))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::mem(dst))?;
            ocl::core::set_kernel_arg(kernel, 3, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(kernel, 5, ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(kernel, 6, ArgVal::scalar(&ne02))?;
            ocl::core::set_kernel_arg(kernel, 7, ArgVal::scalar(&ne03))?;
            ocl::core::set_kernel_arg(kernel, 8, ArgVal::scalar(&nb01))?;
            ocl::core::set_kernel_arg(kernel, 9, ArgVal::scalar(&nb02))?;
            ocl::core::set_kernel_arg(kernel, 10, ArgVal::scalar(&nb03))?;
            ocl::core::set_kernel_arg(kernel, 11, ArgVal::scalar(&eps))?;
            ocl::core::set_kernel_arg(kernel, 12, ArgVal::local::<f32>(&local_mem_bytes))?;
            let gws: [usize; 3] = [nth, 1, 1];
            let lws: [usize; 3] = [nth, 1, 1];
            let mut ev = Event::null();
            ocl::core::enqueue_kernel(
                queue,
                kernel,
                3,
                None,
                &gws,
                Some(lws),
                None::<&Event>,
                Some(&mut ev),
            )?;
            Ok(ev)
        }
    }

    pub fn bench_rms_norm(state: &State) -> anyhow::Result<(OpResult, OpResult)> {
        let x = alloc_f32(&state.context, DIM)?;
        let out = alloc_f32(&state.context, DIM)?;
        let w = alloc_f32(&state.context, DIM)?;
        fill_random_bytes(&state.queue, &x, DIM * 4, 0x1111)?;
        fill_random_bytes(&state.queue, &w, DIM * 4, 0x2222)?;

        let our_k = ocl::core::create_kernel(&state.our_program, "kernel_rms_norm_oop")?;
        let our_f4_k = ocl::core::create_kernel(&state.our_program, "kernel_rms_norm_oop_f4")?;
        let llama_k = ocl::core::create_kernel(&state.llama_program_rms, "kernel_rms_norm")?;

        for _ in 0..WARMUP {
            let ev = dispatch_our_rms(&state.queue, &our_k, &x, &out, &w, DIM)?;
            ocl::core::finish(&state.queue)?;
            let _ = time_event(&ev);
            let ev = dispatch_our_rms(&state.queue, &our_f4_k, &x, &out, &w, DIM)?;
            ocl::core::finish(&state.queue)?;
            let _ = time_event(&ev);
            let ev = dispatch_llama_rms(&state.queue, &llama_k, &x, &out, DIM)?;
            ocl::core::finish(&state.queue)?;
            let _ = time_event(&ev);
        }

        let mut our_s = Vec::with_capacity(ITERS);
        let mut our_f4_s = Vec::with_capacity(ITERS);
        let mut ll_s = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let ev = dispatch_our_rms(&state.queue, &our_k, &x, &out, &w, DIM)?;
            ocl::core::finish(&state.queue)?;
            our_s.push(time_event(&ev)?);
            let ev = dispatch_our_rms(&state.queue, &our_f4_k, &x, &out, &w, DIM)?;
            ocl::core::finish(&state.queue)?;
            our_f4_s.push(time_event(&ev)?);
            let ev = dispatch_llama_rms(&state.queue, &llama_k, &x, &out, DIM)?;
            ocl::core::finish(&state.queue)?;
            ll_s.push(time_event(&ev)?);
        }
        // Print f4 result inline (no API change to bench_rms_norm signature)
        let sorted: Vec<f64> = {
            let mut s = our_f4_s.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s
        };
        let med = sorted[sorted.len() / 2];
        let mean: f64 = our_f4_s.iter().sum::<f64>() / our_f4_s.len() as f64;
        println!(
            "#   our_f4         min/med/mean = {:.2}/{:.2}/{:.2} μs  (float4 vectorized)",
            sorted[0], med, mean
        );

        let shape_str = format!("[{}]", DIM);
        let calls = 2 * N_LAYERS;
        Ok((
            OpResult {
                engine: "llm_rs2",
                op: "RMSNorm".into(),
                shape: shape_str.clone(),
                calls_per_tok: calls,
                samples: our_s,
            },
            OpResult {
                engine: "llama.cpp",
                op: "RMSNorm".into(),
                shape: shape_str,
                calls_per_tok: calls,
                samples: ll_s,
            },
        ))
    }

    // ---------- RoPE ----------

    /// llm_rs2: kernel_rope_simple(x, head_dim, num_heads, seq_len, start_pos, theta)
    /// gws=[seq_len*num_heads*head_dim/2, 1, 1], lws=None
    fn dispatch_our_rope(
        queue: &Queue,
        kernel: &ocl::core::Kernel,
        x: &Mem,
        num_heads: usize,
    ) -> anyhow::Result<Event> {
        let head_dim = HEAD_DIM as i32;
        let num_heads_i = num_heads as i32;
        let seq_len = 1i32;
        let start_pos = 0i32;
        let theta = 1_000_000.0f32;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(x))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::scalar(&head_dim))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::scalar(&num_heads_i))?;
            ocl::core::set_kernel_arg(kernel, 3, ArgVal::scalar(&seq_len))?;
            ocl::core::set_kernel_arg(kernel, 4, ArgVal::scalar(&start_pos))?;
            ocl::core::set_kernel_arg(kernel, 5, ArgVal::scalar(&theta))?;
            let work_size = num_heads * (HEAD_DIM / 2);
            let gws: [usize; 3] = [work_size, 1, 1];
            let mut ev = Event::null();
            ocl::core::enqueue_kernel(
                queue,
                kernel,
                1,
                None,
                &gws,
                None,
                None::<&Event>,
                Some(&mut ev),
            )?;
            Ok(ev)
        }
    }

    /// llama.cpp: kernel_rope_norm_f32 — 33 args, gws=[ne01*nth, ne02, ne03], lws=[nth,1,1].
    /// decode: ne00=head_dim, ne01=num_heads, ne02=seq_len=1, ne03=1.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_llama_rope(
        queue: &Queue,
        kernel: &ocl::core::Kernel,
        x: &Mem,
        pos_buf: &Mem,
        num_heads: usize,
    ) -> anyhow::Result<Event> {
        let ne00 = HEAD_DIM as i32;
        let ne01 = num_heads as i32;
        let ne02 = 1i32;
        let ne03 = 1i32;
        let nb00 = 4u64;
        let nb01 = (HEAD_DIM * 4) as u64;
        let nb02 = (num_heads * HEAD_DIM * 4) as u64;
        let nb03 = nb02;
        let ne0 = ne00;
        let ne1 = ne01;
        let ne2 = ne02;
        let ne3 = ne03;
        let nb0 = nb00;
        let nb1 = nb01;
        let nb2 = nb02;
        let nb3 = nb03;
        let n_past = 0i32;
        let n_dims = HEAD_DIM as i32;
        let n_ctx_orig = 32768i32;
        let freq_base = 1_000_000.0f32;
        let freq_scale = 1.0f32;
        let ext_factor = 0.0f32;
        let attn_factor = 1.0f32;
        let beta_fast = 32.0f32;
        let beta_slow = 1.0f32;
        let zero_u64 = 0u64;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(x))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::mem(pos_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ArgVal::scalar(&zero_u64))?;
            // src2 = src0 (freq_factors unused; kernel에서 src2==src0이면 1.0 사용)
            ocl::core::set_kernel_arg(kernel, 4, ArgVal::mem(x))?;
            ocl::core::set_kernel_arg(kernel, 5, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 6, ArgVal::mem(x))?;
            ocl::core::set_kernel_arg(kernel, 7, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 8, ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(kernel, 9, ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(kernel, 10, ArgVal::scalar(&ne02))?;
            ocl::core::set_kernel_arg(kernel, 11, ArgVal::scalar(&ne03))?;
            ocl::core::set_kernel_arg(kernel, 12, ArgVal::scalar(&nb00))?;
            ocl::core::set_kernel_arg(kernel, 13, ArgVal::scalar(&nb01))?;
            ocl::core::set_kernel_arg(kernel, 14, ArgVal::scalar(&nb02))?;
            ocl::core::set_kernel_arg(kernel, 15, ArgVal::scalar(&nb03))?;
            ocl::core::set_kernel_arg(kernel, 16, ArgVal::scalar(&ne0))?;
            ocl::core::set_kernel_arg(kernel, 17, ArgVal::scalar(&ne1))?;
            ocl::core::set_kernel_arg(kernel, 18, ArgVal::scalar(&ne2))?;
            ocl::core::set_kernel_arg(kernel, 19, ArgVal::scalar(&ne3))?;
            ocl::core::set_kernel_arg(kernel, 20, ArgVal::scalar(&nb0))?;
            ocl::core::set_kernel_arg(kernel, 21, ArgVal::scalar(&nb1))?;
            ocl::core::set_kernel_arg(kernel, 22, ArgVal::scalar(&nb2))?;
            ocl::core::set_kernel_arg(kernel, 23, ArgVal::scalar(&nb3))?;
            ocl::core::set_kernel_arg(kernel, 24, ArgVal::scalar(&n_past))?;
            ocl::core::set_kernel_arg(kernel, 25, ArgVal::scalar(&n_dims))?;
            ocl::core::set_kernel_arg(kernel, 26, ArgVal::scalar(&n_ctx_orig))?;
            ocl::core::set_kernel_arg(kernel, 27, ArgVal::scalar(&freq_base))?;
            ocl::core::set_kernel_arg(kernel, 28, ArgVal::scalar(&freq_scale))?;
            ocl::core::set_kernel_arg(kernel, 29, ArgVal::scalar(&ext_factor))?;
            ocl::core::set_kernel_arg(kernel, 30, ArgVal::scalar(&attn_factor))?;
            ocl::core::set_kernel_arg(kernel, 31, ArgVal::scalar(&beta_fast))?;
            ocl::core::set_kernel_arg(kernel, 32, ArgVal::scalar(&beta_slow))?;

            let nth: usize = std::cmp::min(64, HEAD_DIM);
            let gws: [usize; 3] = [num_heads * nth, 1, 1];
            let lws: [usize; 3] = [nth, 1, 1];
            let mut ev = Event::null();
            ocl::core::enqueue_kernel(
                queue,
                kernel,
                3,
                None,
                &gws,
                Some(lws),
                None::<&Event>,
                Some(&mut ev),
            )?;
            Ok(ev)
        }
    }

    pub fn bench_rope(
        state: &State,
        num_heads: usize,
        label: &'static str,
        calls_per_tok: usize,
    ) -> anyhow::Result<(OpResult, OpResult)> {
        let n_elems = num_heads * HEAD_DIM;
        let x = alloc_f32(&state.context, n_elems)?;
        fill_random_bytes(&state.queue, &x, n_elems * 4, 0x3030)?;

        // llama.cpp requires pos buffer (int * n_tokens). decode seq_len=1.
        let pos_buf = alloc_bytes(&state.context, 4)?;
        fill_random_bytes(&state.queue, &pos_buf, 4, 0x4040)?;

        let our_k = ocl::core::create_kernel(&state.our_program, "kernel_rope_simple")?;
        let llama_k = ocl::core::create_kernel(&state.llama_program_rope, "kernel_rope_norm_f32")?;

        for _ in 0..WARMUP {
            let ev = dispatch_our_rope(&state.queue, &our_k, &x, num_heads)?;
            ocl::core::finish(&state.queue)?;
            let _ = time_event(&ev);
            let ev = dispatch_llama_rope(&state.queue, &llama_k, &x, &pos_buf, num_heads)?;
            ocl::core::finish(&state.queue)?;
            let _ = time_event(&ev);
        }

        let mut our_s = Vec::with_capacity(ITERS);
        let mut ll_s = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let ev = dispatch_our_rope(&state.queue, &our_k, &x, num_heads)?;
            ocl::core::finish(&state.queue)?;
            our_s.push(time_event(&ev)?);
            let ev = dispatch_llama_rope(&state.queue, &llama_k, &x, &pos_buf, num_heads)?;
            ocl::core::finish(&state.queue)?;
            ll_s.push(time_event(&ev)?);
        }

        let shape_str = format!("[1, {}, {}]", num_heads, HEAD_DIM);
        Ok((
            OpResult {
                engine: "llm_rs2",
                op: format!("RoPE ({})", label),
                shape: shape_str.clone(),
                calls_per_tok,
                samples: our_s,
            },
            OpResult {
                engine: "llama.cpp",
                op: format!("RoPE ({})", label),
                shape: shape_str,
                calls_per_tok,
                samples: ll_s,
            },
        ))
    }

    // ---------- SiLU·mul ----------

    /// llm_rs2: kernel_silu_mul_simple(x, y, size4) — in-place x[i] = silu(x[i]) * y[i], float4
    fn dispatch_our_silu_mul(
        queue: &Queue,
        kernel: &ocl::core::Kernel,
        x: &Mem,
        y: &Mem,
        size: usize,
    ) -> anyhow::Result<Event> {
        let size4 = (size / 4) as i32;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(x))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::mem(y))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::scalar(&size4))?;
            let gws: [usize; 3] = [size / 4, 1, 1];
            let mut ev = Event::null();
            ocl::core::enqueue_kernel(
                queue,
                kernel,
                1,
                None,
                &gws,
                None,
                None::<&Event>,
                Some(&mut ev),
            )?;
            Ok(ev)
        }
    }

    /// llama.cpp: kernel_silu_4 + kernel_mul_row (2-step, 합산 측정).
    ///   - silu_4(src0=gate, off0, dst=tmp, offd)  gws=[n/4], lws=[64,1,1]
    ///   - mul_row(src0=tmp, off0, src1=up, off1, dst=tmp, offd, ne=n/4)
    fn dispatch_llama_silu_mul(
        queue: &Queue,
        silu_k: &ocl::core::Kernel,
        mul_k: &ocl::core::Kernel,
        gate: &Mem,
        up: &Mem,
        tmp: &Mem,
        size: usize,
    ) -> anyhow::Result<(Event, Event)> {
        let n = size / 4;
        let zero_u64 = 0u64;
        unsafe {
            ocl::core::set_kernel_arg(silu_k, 0, ArgVal::mem(gate))?;
            ocl::core::set_kernel_arg(silu_k, 1, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(silu_k, 2, ArgVal::mem(tmp))?;
            ocl::core::set_kernel_arg(silu_k, 3, ArgVal::scalar(&zero_u64))?;
            let gws: [usize; 3] = [n, 1, 1];
            let lws: [usize; 3] = [64, 1, 1];
            let mut ev_a = Event::null();
            ocl::core::enqueue_kernel(
                queue,
                silu_k,
                3,
                None,
                &gws,
                Some(lws),
                None::<&Event>,
                Some(&mut ev_a),
            )?;

            // mul_row broadcast row — (src0, off0, src1, off1, dst, offd, ne)
            let ne = n as i32;
            ocl::core::set_kernel_arg(mul_k, 0, ArgVal::mem(tmp))?;
            ocl::core::set_kernel_arg(mul_k, 1, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(mul_k, 2, ArgVal::mem(up))?;
            ocl::core::set_kernel_arg(mul_k, 3, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(mul_k, 4, ArgVal::mem(tmp))?;
            ocl::core::set_kernel_arg(mul_k, 5, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(mul_k, 6, ArgVal::scalar(&ne))?;
            let gws2: [usize; 3] = [n, 1, 1];
            let lws2: [usize; 3] = [64, 1, 1];
            let mut ev_b = Event::null();
            ocl::core::enqueue_kernel(
                queue,
                mul_k,
                1,
                None,
                &gws2,
                Some(lws2),
                None::<&Event>,
                Some(&mut ev_b),
            )?;
            Ok((ev_a, ev_b))
        }
    }

    pub fn bench_silu_mul(state: &State) -> anyhow::Result<(OpResult, OpResult)> {
        let size = FFN_DIM;
        let gate = alloc_f32(&state.context, size)?;
        let up = alloc_f32(&state.context, size)?;
        let tmp = alloc_f32(&state.context, size)?;
        fill_random_bytes(&state.queue, &gate, size * 4, 0x5050)?;
        fill_random_bytes(&state.queue, &up, size * 4, 0x6060)?;

        let our_k = ocl::core::create_kernel(&state.our_program, "kernel_silu_mul_simple")?;
        let silu_k = ocl::core::create_kernel(&state.llama_program_silu, "kernel_silu_4")?;
        let mul_k = ocl::core::create_kernel(&state.llama_program_mul, "kernel_mul_row")?;

        for _ in 0..WARMUP {
            let ev = dispatch_our_silu_mul(&state.queue, &our_k, &gate, &up, size)?;
            ocl::core::finish(&state.queue)?;
            let _ = time_event(&ev);
            let (a, b) =
                dispatch_llama_silu_mul(&state.queue, &silu_k, &mul_k, &gate, &up, &tmp, size)?;
            ocl::core::finish(&state.queue)?;
            let _ = time_event(&a);
            let _ = time_event(&b);
        }

        let mut our_s = Vec::with_capacity(ITERS);
        let mut ll_s = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let ev = dispatch_our_silu_mul(&state.queue, &our_k, &gate, &up, size)?;
            ocl::core::finish(&state.queue)?;
            our_s.push(time_event(&ev)?);

            let (a, b) =
                dispatch_llama_silu_mul(&state.queue, &silu_k, &mul_k, &gate, &up, &tmp, size)?;
            ocl::core::finish(&state.queue)?;
            ll_s.push(time_event(&a)? + time_event(&b)?);
        }

        let shape_str = format!("[{}]", size);
        let calls = N_LAYERS;
        Ok((
            OpResult {
                engine: "llm_rs2",
                op: "SiLU·mul".into(),
                shape: shape_str.clone(),
                calls_per_tok: calls,
                samples: our_s,
            },
            OpResult {
                engine: "llama.cpp",
                op: "SiLU·mul".into(),
                shape: shape_str,
                calls_per_tok: calls,
                samples: ll_s,
            },
        ))
    }

    // ---------- residual add ----------

    /// llm_rs2: kernel_add_assign_simple(x_f4, y_f4, size4) — in-place x += y
    fn dispatch_our_add(
        queue: &Queue,
        kernel: &ocl::core::Kernel,
        x: &Mem,
        y: &Mem,
        size: usize,
    ) -> anyhow::Result<Event> {
        let size4 = (size / 4) as i32;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(x))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::mem(y))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::scalar(&size4))?;
            let gws: [usize; 3] = [size / 4, 1, 1];
            let mut ev = Event::null();
            ocl::core::enqueue_kernel(
                queue,
                kernel,
                1,
                None,
                &gws,
                None,
                None::<&Event>,
                Some(&mut ev),
            )?;
            Ok(ev)
        }
    }

    /// llama.cpp: kernel_add_row(src0_f4, off0, src1_f4, off1, dst_f4, offd, ne)
    ///   gws=[n/4], lws=[64,1,1] (bcast_row path; n % 4 == 0, n % 64 == 0 for DIM=1536)
    fn dispatch_llama_add(
        queue: &Queue,
        kernel: &ocl::core::Kernel,
        x: &Mem,
        y: &Mem,
        out: &Mem,
        size: usize,
    ) -> anyhow::Result<Event> {
        let n = (size / 4) as i32;
        let zero_u64 = 0u64;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(x))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::mem(y))?;
            ocl::core::set_kernel_arg(kernel, 3, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ArgVal::mem(out))?;
            ocl::core::set_kernel_arg(kernel, 5, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 6, ArgVal::scalar(&n))?;
            let gws: [usize; 3] = [size / 4, 1, 1];
            let lws: [usize; 3] = [64, 1, 1];
            let mut ev = Event::null();
            ocl::core::enqueue_kernel(
                queue,
                kernel,
                1,
                None,
                &gws,
                Some(lws),
                None::<&Event>,
                Some(&mut ev),
            )?;
            Ok(ev)
        }
    }

    pub fn bench_add(state: &State) -> anyhow::Result<(OpResult, OpResult)> {
        let size = DIM;
        let x = alloc_f32(&state.context, size)?;
        let y = alloc_f32(&state.context, size)?;
        let out = alloc_f32(&state.context, size)?;
        fill_random_bytes(&state.queue, &x, size * 4, 0x7070)?;
        fill_random_bytes(&state.queue, &y, size * 4, 0x8080)?;

        let our_k = ocl::core::create_kernel(&state.our_program, "kernel_add_assign_simple")?;
        let llama_k = ocl::core::create_kernel(&state.llama_program_add, "kernel_add_row")?;

        for _ in 0..WARMUP {
            let ev = dispatch_our_add(&state.queue, &our_k, &x, &y, size)?;
            ocl::core::finish(&state.queue)?;
            let _ = time_event(&ev);
            let ev = dispatch_llama_add(&state.queue, &llama_k, &x, &y, &out, size)?;
            ocl::core::finish(&state.queue)?;
            let _ = time_event(&ev);
        }

        let mut our_s = Vec::with_capacity(ITERS);
        let mut ll_s = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let ev = dispatch_our_add(&state.queue, &our_k, &x, &y, size)?;
            ocl::core::finish(&state.queue)?;
            our_s.push(time_event(&ev)?);
            let ev = dispatch_llama_add(&state.queue, &llama_k, &x, &y, &out, size)?;
            ocl::core::finish(&state.queue)?;
            ll_s.push(time_event(&ev)?);
        }

        let shape_str = format!("[{}]", size);
        let calls = 2 * N_LAYERS;
        Ok((
            OpResult {
                engine: "llm_rs2",
                op: "Add (residual)".into(),
                shape: shape_str.clone(),
                calls_per_tok: calls,
                samples: our_s,
            },
            OpResult {
                engine: "llama.cpp",
                op: "Add (residual)".into(),
                shape: shape_str,
                calls_per_tok: calls,
                samples: ll_s,
            },
        ))
    }

    // ---------- Report ----------

    /// Generic bench row. Q4_0 rows carry an optional third engine (noshuffle);
    /// small-op rows do not.
    pub struct BenchRow {
        pub ours: OpResult,
        pub llama: OpResult,
        pub noshuffle: Option<OpResult>,
    }

    fn best_label(ours_med: f64, ll_med: f64, ns_med: Option<f64>) -> &'static str {
        let mut best_name = "ours-aos";
        let mut best_val = ours_med;
        if ll_med < best_val {
            best_name = "llama";
            best_val = ll_med;
        }
        if let Some(ns) = ns_med
            && ns < best_val
        {
            best_name = "ours-noshuffle";
            // best_val tracked for future extension but currently terminal
        }
        best_name
    }

    fn print_raw_dump(rows: &[BenchRow]) {
        println!();
        println!("## Raw μs per op (min / median / mean)");
        println!("engine,op,shape,min_us,median_us,mean_us,samples");
        for row in rows {
            for r in [&row.ours, &row.llama] {
                println!(
                    "{},{},{},{:.2},{:.2},{:.2},{}",
                    r.engine,
                    r.op,
                    r.shape,
                    r.min(),
                    r.median(),
                    r.mean(),
                    r.samples.len(),
                );
            }
            if let Some(ns) = row.noshuffle.as_ref() {
                println!(
                    "{},{},{},{:.2},{:.2},{:.2},{}",
                    ns.engine,
                    ns.op,
                    ns.shape,
                    ns.min(),
                    ns.median(),
                    ns.mean(),
                    ns.samples.len(),
                );
            }
        }
    }

    pub fn print_markdown_summary(rows: &[BenchRow]) {
        // Q4_0 rows (3-way) and small-op rows (2-way) use separate tables for
        // readability; raw CSV dumps all engines.

        let (q4_rows, small_rows): (Vec<&BenchRow>, Vec<&BenchRow>) =
            rows.iter().partition(|r| r.noshuffle.is_some());

        if !q4_rows.is_empty() {
            println!();
            println!("## Q4_0 GEMV microbench (3-way: AoS vs noshuffle-image vs llama 8x_flat)");
            println!();
            println!(
                "| Op | Shape | ours-aos μs | ours-noshuffle μs | llama μs | best | 호출/tok | aos ms/tok | noshuffle ms/tok | llama ms/tok |"
            );
            println!("|---|---|---:|---:|---:|---|---:|---:|---:|---:|");
            let mut total_our_ms = 0.0f64;
            let mut total_ns_ms = 0.0f64;
            let mut total_ll_ms = 0.0f64;
            for row in &q4_rows {
                let our_med = row.ours.median();
                let ll_med = row.llama.median();
                let ns_med_opt = row.noshuffle.as_ref().map(|r| r.median());
                let best = best_label(our_med, ll_med, ns_med_opt);
                let our_contrib = our_med * row.ours.calls_per_tok as f64 / 1000.0;
                let ll_contrib = ll_med * row.llama.calls_per_tok as f64 / 1000.0;
                let ns_contrib = ns_med_opt.map(|m| m * row.ours.calls_per_tok as f64 / 1000.0);
                total_our_ms += our_contrib;
                total_ll_ms += ll_contrib;
                if let Some(c) = ns_contrib {
                    total_ns_ms += c;
                }
                let ns_med_str = ns_med_opt
                    .map(|m| format!("{:.2}", m))
                    .unwrap_or_else(|| "-".to_string());
                let ns_contrib_str = ns_contrib
                    .map(|c| format!("{:.2}", c))
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "| {} | {} | {:.2} | {} | {:.2} | {} | {} | {:.2} | {} | {:.2} |",
                    row.ours.op,
                    row.ours.shape,
                    our_med,
                    ns_med_str,
                    ll_med,
                    best,
                    row.ours.calls_per_tok,
                    our_contrib,
                    ns_contrib_str,
                    ll_contrib,
                );
            }
            println!(
                "| **Q4_0 합계** | | | | | | | **{:.2} ms** | **{:.2} ms** | **{:.2} ms** |",
                total_our_ms, total_ns_ms, total_ll_ms
            );
        }

        if !small_rows.is_empty() {
            println!();
            println!("## Small-op microbench (2-way)");
            println!();
            println!(
                "| Op | Shape | llm_rs2 μs | llama.cpp μs | ratio (llama/ours) | 호출/tok | llm_rs2 기여 ms/tok | llama 기여 ms/tok |"
            );
            println!("|---|---|---:|---:|---:|---:|---:|---:|");
            let mut total_our_ms = 0.0f64;
            let mut total_ll_ms = 0.0f64;
            for row in &small_rows {
                let our_med = row.ours.median();
                let ll_med = row.llama.median();
                let ratio = if our_med > 0.0 { ll_med / our_med } else { 0.0 };
                let our_contrib = our_med * row.ours.calls_per_tok as f64 / 1000.0;
                let ll_contrib = ll_med * row.llama.calls_per_tok as f64 / 1000.0;
                total_our_ms += our_contrib;
                total_ll_ms += ll_contrib;
                println!(
                    "| {} | {} | {:.2} | {:.2} | {:.2}x | {} | {:.2} | {:.2} |",
                    row.ours.op,
                    row.ours.shape,
                    our_med,
                    ll_med,
                    ratio,
                    row.ours.calls_per_tok,
                    our_contrib,
                    ll_contrib,
                );
            }
            println!(
                "| **Small-op 합계** | | | | | | **{:.2} ms** | **{:.2} ms** |",
                total_our_ms, total_ll_ms
            );
        }

        print_raw_dump(rows);
    }
}

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use std::time::Duration;

    // 시작 전 쿨다운
    bench::wait_for_cooldown(32_000, Duration::from_secs(600));

    let state = bench::init()?;

    let mut all: Vec<bench::BenchRow> = Vec::new();

    // Q4_0 matmul: 각 shape 사이 15초 쿨다운
    for (i, shape) in bench::Q4_SHAPES.iter().enumerate() {
        if i > 0 {
            bench::sleep_quiet(Duration::from_secs(15));
        }
        println!(
            "# Q4_0 matmul shape: {} (K={}, N={}, calls/tok={})",
            shape.name, shape.k, shape.n, shape.calls_per_tok
        );
        let (ours, llama, noshuffle) = bench::bench_q4_shape(&state, *shape)?;
        println!(
            "#   ours-aos        min/med/mean = {:.2}/{:.2}/{:.2} μs",
            ours.min(),
            ours.median(),
            ours.mean()
        );
        if let Some(ns) = noshuffle.as_ref() {
            println!(
                "#   ours-noshuffle  min/med/mean = {:.2}/{:.2}/{:.2} μs",
                ns.min(),
                ns.median(),
                ns.mean()
            );
        } else {
            println!("#   ours-noshuffle  unavailable (image1d_buffer creation failed)");
        }
        println!(
            "#   llama-8x_flat   min/med/mean = {:.2}/{:.2}/{:.2} μs",
            llama.min(),
            llama.median(),
            llama.mean()
        );

        // Ab_Bi_8x4 — only for FFN shapes (avoid huge buffers like lm_head N=151936).
        if shape.n % 4 == 0 && shape.k % 4 == 0 && shape.n <= 10000 {
            match bench::bench_ab_bi_shape(&state, *shape) {
                Ok(ab) => {
                    println!(
                        "#   llama-Ab_Bi_8x4 min/med/mean = {:.2}/{:.2}/{:.2} μs",
                        ab.min(),
                        ab.median(),
                        ab.mean()
                    );
                }
                Err(e) => {
                    println!("#   llama-Ab_Bi_8x4 FAILED: {}", e);
                }
            }
        }

        all.push(bench::BenchRow {
            ours,
            llama,
            noshuffle,
        });
    }

    // small ops — 각 op 사이 5초 쿨다운
    type SmallOpFn<'a> = Box<dyn Fn() -> anyhow::Result<(bench::OpResult, bench::OpResult)> + 'a>;
    let small_ops: Vec<(&str, SmallOpFn)> = vec![
        ("RMSNorm", Box::new(|| bench::bench_rms_norm(&state))),
        (
            "RoPE (Q)",
            Box::new(|| bench::bench_rope(&state, bench::N_HEADS_Q, "Q", bench::N_LAYERS)),
        ),
        (
            "RoPE (K)",
            Box::new(|| bench::bench_rope(&state, bench::N_HEADS_KV, "K", bench::N_LAYERS)),
        ),
        ("SiLU·mul", Box::new(|| bench::bench_silu_mul(&state))),
        ("Add", Box::new(|| bench::bench_add(&state))),
    ];

    for (i, (name, f)) in small_ops.iter().enumerate() {
        bench::sleep_quiet(Duration::from_secs(if i == 0 { 15 } else { 5 }));
        println!("# small op: {}", name);
        let (ours, llama) = f()?;
        println!(
            "#   llm_rs2  min/med/mean = {:.2}/{:.2}/{:.2} μs",
            ours.min(),
            ours.median(),
            ours.mean()
        );
        println!(
            "#   llama    min/med/mean = {:.2}/{:.2}/{:.2} μs",
            llama.min(),
            llama.median(),
            llama.mean()
        );
        all.push(bench::BenchRow {
            ours,
            llama,
            noshuffle: None,
        });
    }

    bench::print_markdown_summary(&all);
    Ok(())
}
