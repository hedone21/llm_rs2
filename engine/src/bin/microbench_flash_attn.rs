//! microbench_flash_attn — flash_attn_f32_f16_q1 격리 측정 (Option C+A)
//!
//! 목적 1 (Option C, 완료): KV stride layout 차이 격리 검증 → REJECT
//!   - HeadMajor (k_nb1=256B) vs PosMajor (k_nb1=512B): slope ratio ≈ 1.0
//!
//! 목적 2 (Option A, 본 확장): production 조건과 microbench-best 사이의
//! 차이 (per-layer 0.36 → 0.47 μs/n_kv) 분해. 4 variants를 단계적으로 추가:
//!   - Single        : 1 dispatch, mask=NULL, Q=const  (microbench baseline)
//!   - Repeat28      : 28 back-to-back dispatches, NULL mask, const Q
//!     → sequential cache pressure 효과 측정
//!   - Repeat28Mask  : 28 dispatches + causal F16 mask 버퍼 read
//!     → mask read overhead 측정 (k_idx마다 1 half load)
//!   - Repeat28MaskQ : 28 dispatches + mask + Q를 28-slot pool에서 rotate
//!     → Q cold-cache 효과 측정 (production은 매 layer 다른 Q)
//!
//! 목적 3 (H2 GPU 캐시 thrashing 검증, 본 확장): Q1 dispatch 사이에 대용량
//! 버퍼 read/write 커널을 삽입해 L2/SLC를 오염시킨 뒤 Q1 slope 변화 측정.
//! production에서 Q1 직전에 QKV/RoPE/scatter, 직후에 FFN(~20MB Q4_0 streaming)이
//! 실행되어 K/V가 cold-cache 상태로 밀려나는 효과를 흉내냄.
//!   - Repeat28Pollute8  : 각 Q1 전 8MB streaming (LLC 일부 eviction)
//!   - Repeat28Pollute32 : 각 Q1 전 32MB streaming (LLC 완전 eviction)
//! production과 microbench 사이 Q1 slope 갭(microbench TIE, production 5.70 vs 7.32 μs/n_kv)이
//! 주변 op의 캐시 thrashing으로 설명되는지 검증.
//!
//! 가설: incremental overhead 합이 production gap 0.11 μs/n_kv per layer
//! (= 3.1 μs/n_kv per token) 를 설명한다.
//!
//! 모든 variant × 2 layout × 4 n_kv → 12 조합 × MEASURE_ITERS = 30 measurements.

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("microbench_flash_attn requires --features opencl");
    std::process::exit(2);
}

#[cfg(feature = "opencl")]
mod bench {
    use ocl::core::{ArgVal, Event, Mem, ProfilingInfo};
    use ocl::flags;
    use ocl::{Context, Device, Platform, Program, Queue};

    // === 모델 파라미터 (Qwen 2.5-1.5B) ===
    pub const N_H_Q: usize = 12;
    pub const N_H_KV: usize = 2;
    pub const DK: usize = 128;
    pub const DV: usize = 128;
    pub const CAP: usize = 8192; // 최대 n_kv 4472 + 마진
    pub const Q1_WG_SIZE: usize = 64;
    pub const N_LAYERS: usize = 28; // Repeat28 — Qwen 2.5-1.5B 레이어 수

    pub const N_KV_VALUES: &[i32] = &[258, 1025, 2047, 4472];
    pub const WARMUP_ITERS: usize = 5;
    pub const MEASURE_ITERS: usize = 30;

    /// 변형 정의 — 누적적으로 production overhead 추가
    #[derive(Clone, Copy)]
    pub struct Variant {
        pub name: &'static str,
        pub num_dispatches: usize,
        pub use_mask: bool,
        pub vary_q: bool,
        /// Q1 dispatch마다 앞에 삽입할 "cache pollute" 커널의 읽기 buffer 크기(MB).
        /// 0 = 삽입 안 함. production에서 Q1 주변 FFN/QKV가 L2/SLC를 밀어내는 효과 흉내.
        pub pollute_mb: usize,
        /// true = dispatch마다 다른 cl_mem(K, V) 사용 (28×K + 28×V = 56 separate cl_mem).
        /// production과 동일한 cl_mem 개수. false = 공유 K/V (llama.cpp처럼 1 cl_mem).
        pub fragmented_kv: bool,
    }

    /// Cache-pollute 커널용 최대 버퍼 크기. 변수 정의가 아닌 alloc 상한.
    pub const POLLUTE_MAX_MB: usize = 64;

    pub const VARIANTS: &[Variant] = &[
        Variant {
            name: "Single",
            num_dispatches: 1,
            use_mask: false,
            vary_q: false,
            pollute_mb: 0,
            fragmented_kv: false,
        },
        Variant {
            name: "Repeat28",
            num_dispatches: N_LAYERS,
            use_mask: false,
            vary_q: false,
            pollute_mb: 0,
            fragmented_kv: false,
        },
        Variant {
            name: "Repeat28Mask",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: false,
            pollute_mb: 0,
            fragmented_kv: false,
        },
        Variant {
            name: "Repeat28MaskQ",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: true,
            pollute_mb: 0,
            fragmented_kv: false,
        },
        Variant {
            name: "Repeat28Pollute8",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: true,
            pollute_mb: 8,
            fragmented_kv: false,
        },
        Variant {
            name: "Repeat28Pollute32",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: true,
            pollute_mb: 32,
            fragmented_kv: false,
        },
        // 28 layer KV를 28 separate cl_mem으로 분리 — production 실제 구조.
        // Contiguous(Repeat28MaskQ) 대비 slope 차이 → cl_mem 분리 오버헤드 격리 측정.
        Variant {
            name: "Repeat28Frag",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: true,
            pollute_mb: 0,
            fragmented_kv: true,
        },
    ];

    /// fields hold GPU buffer ownership so they aren't dropped early.
    #[allow(dead_code)]
    pub struct State {
        pub context: Context,
        pub queue: Queue,
        /// (engine_name, kernel) — Cross-run에서 두 엔진(우리 vs llama.cpp)
        /// Q1 kernel을 같은 buffer 상에서 dispatch.
        pub kernels: Vec<(String, ocl::core::Kernel)>,
        pub q_pool_buf: Mem,
        pub k_buf: Mem,
        pub v_buf: Mem,
        /// 28 separate K cl_mem (production 구조 시뮬레이션, Repeat28Frag 전용)
        pub k_bufs_frag: Vec<Mem>,
        pub v_bufs_frag: Vec<Mem>,
        pub o_buf: Mem,
        pub mask_buf: Mem,
        pub pollute_buf: Mem,
        pub pollute_kernel: ocl::core::Kernel,
        pub q_slot_bytes: u64,
        pub gws: [usize; 3],
        pub lws: Option<[usize; 3]>,
    }

    /// Cache-pollute 커널: streaming RW로 L2/SLC를 강제 eviction.
    const POLLUTE_SRC: &str = r#"
__kernel void pollute(__global float4* buf, const int elems4) {
    const int gid = get_global_id(0);
    if (gid < elems4) {
        float4 v = buf[gid];
        buf[gid] = v + (float4)(1e-30f);
    }
}
"#;

    pub fn init() -> anyhow::Result<State> {
        let platform = Platform::default();
        let device = Device::first(platform)?;
        let platform_name = platform.name().unwrap_or_else(|_| "unknown".into());
        let device_name = device.name().unwrap_or_else(|_| "unknown".into());
        println!("# microbench_flash_attn — Option C (stride) + Option A (production conditions)");
        println!("# Platform: {}", platform_name);
        println!("# Device:   {}", device_name);
        println!(
            "# Model: n_h_q={} n_h_kv={} dk={} dv={} cap={} Q1_WG_SIZE={} layers={}",
            N_H_Q, N_H_KV, DK, DV, CAP, Q1_WG_SIZE, N_LAYERS
        );

        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;
        // LLMRS_MB_WALLCLOCK=1 → profile-events 플래그 없이 queue 생성.
        // Wall-clock 측정 (Instant::now() + queue.finish()) 을 대신 사용하여
        // production dispatch와 동일한 queue 설정으로 Q1 slope 격리 측정.
        let wallclock_mode = std::env::var_os("LLMRS_MB_WALLCLOCK").is_some();
        let queue_props = if wallclock_mode {
            None
        } else {
            Some(flags::QUEUE_PROFILING_ENABLE)
        };
        let queue = Queue::new(&context, device, queue_props)?;
        if wallclock_mode {
            println!("# Mode: wallclock (no CL_QUEUE_PROFILING_ENABLE, Instant-based)");
        } else {
            println!("# Mode: profile-events (CL_QUEUE_PROFILING_ENABLE)");
        }

        // 컴파일 옵션 — production과 동일
        let cl_c_version_str = device
            .info(ocl::core::DeviceInfo::OpenclCVersion)
            .map(|v| v.to_string())
            .unwrap_or_default();
        let supports_cl_c_2 = cl_c_version_str
            .split_whitespace()
            .nth(2)
            .and_then(|ver| {
                let mut parts = ver.split('.');
                let major: u32 = parts.next()?.parse().ok()?;
                let minor: u32 = parts.next()?.parse().ok()?;
                Some((major, minor) >= (2, 0))
            })
            .unwrap_or(false);
        let fast_math = "-cl-mad-enable -cl-unsafe-math-optimizations \
                         -cl-finite-math-only -cl-fast-relaxed-math";
        let cl_opts = if supports_cl_c_2 {
            format!("-cl-std=CL2.0 {}", fast_math)
        } else {
            fast_math.to_string()
        };
        let dk128_defines = format!("-DDK={} -DDV={} -DBLOCK_M=32 -DBLOCK_N=32", DK, DV);
        let full_opts = format!("{} {}", dk128_defines, cl_opts);
        println!("# Build opts: {}", full_opts);

        // 우리(llm_rs2) Q1: production source 그대로
        let our_src = include_str!("../../kernels/flash_attn_f32_f16.cl");
        let our_program = Program::builder()
            .devices(device)
            .src(our_src)
            .cmplr_opt(&full_opts)
            .build(&context)?;
        let our_kernel = ocl::core::create_kernel(&our_program, "flash_attn_f32_f16_q1")?;

        // llama.cpp Q1: vendored source (Phase A 재감사 후 cross-run 검증용).
        // SLM tree-reduce + barrier 패턴 (B-4 sub_group_reduce 미적용).
        let llama_src = include_str!(
            "../../../.agent/research/microbench_flash_attn/llamacpp_q1_flash_attn.cl"
        );
        let llama_program = Program::builder()
            .devices(device)
            .src(llama_src)
            .cmplr_opt(&full_opts)
            .build(&context)?;
        let llama_kernel = ocl::core::create_kernel(&llama_program, "flash_attn_f32_f16_q1")?;

        // KV: HeadMajor 기본 layout으로 alloc. 0.125 (F16 0x3000) 채움.
        let kv_elems = N_H_KV * CAP * DK;
        let kv_filler: u16 = 0x3000;
        let kv_host: Vec<u16> = vec![kv_filler; kv_elems];
        let alloc_kv = || -> anyhow::Result<Mem> {
            let buf = unsafe {
                ocl::core::create_buffer::<_, u16>(
                    context.as_core(),
                    ocl::core::MEM_READ_WRITE,
                    kv_elems,
                    None,
                )?
            };
            unsafe {
                ocl::core::enqueue_write_buffer(
                    &queue,
                    &buf,
                    true,
                    0,
                    std::slice::from_raw_parts(kv_host.as_ptr() as *const u8, kv_elems * 2),
                    None::<&Event>,
                    None::<&mut Event>,
                )?;
            }
            Ok(buf)
        };
        let k_buf = alloc_kv()?;
        let v_buf = alloc_kv()?;

        // Fragmented variant용 28 separate K/V buffers. 각 buffer는 1 layer KV size
        // (N_H_KV * CAP * DK half = 4 MB) = production의 alloc_kv 단위와 동일.
        // 56 separate cl_mem object를 만들어 driver bookkeeping 오버헤드를 격리.
        let mut k_bufs_frag: Vec<Mem> = Vec::with_capacity(N_LAYERS);
        let mut v_bufs_frag: Vec<Mem> = Vec::with_capacity(N_LAYERS);
        for _ in 0..N_LAYERS {
            k_bufs_frag.push(alloc_kv()?);
            v_bufs_frag.push(alloc_kv()?);
        }

        // Q pool: 28-slot, 각 slot = N_H_Q * DK F32. 슬롯별로 다른 값 채워서
        // Q variance variant에서 cold-cache 흉내.
        let q_elems_per_slot = N_H_Q * DK;
        let q_pool_elems = N_LAYERS * q_elems_per_slot;
        let q_pool_host: Vec<f32> = (0..q_pool_elems)
            .map(|i| 0.05 + ((i % 17) as f32) * 0.01)
            .collect();
        let q_pool_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                context.as_core(),
                ocl::core::MEM_READ_WRITE,
                q_pool_elems,
                None,
            )?
        };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &queue,
                &q_pool_buf,
                true,
                0,
                std::slice::from_raw_parts(q_pool_host.as_ptr() as *const u8, q_pool_elems * 4),
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
        let q_slot_bytes = (q_elems_per_slot * 4) as u64;

        // O 버퍼
        let o_elems = N_H_Q * DV;
        let o_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                context.as_core(),
                ocl::core::MEM_READ_WRITE,
                o_elems,
                None,
            )?
        };

        // Mask buffer: F16 zeros, 길이 = max n_kv.  causal mask는 decode에서
        // 모든 과거 가시 → 값 전부 0.0 (F16 = 0x0000). 커널이 매 k_idx마다
        // 1 half load 수행하는 비용만 측정 목적.
        let mask_elems = *N_KV_VALUES.iter().max().unwrap() as usize;
        let mask_host: Vec<u16> = vec![0u16; mask_elems];
        let mask_buf = unsafe {
            ocl::core::create_buffer::<_, u16>(
                context.as_core(),
                ocl::core::MEM_READ_WRITE,
                mask_elems,
                None,
            )?
        };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &queue,
                &mask_buf,
                true,
                0,
                std::slice::from_raw_parts(mask_host.as_ptr() as *const u8, mask_elems * 2),
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }

        // 정적 stride/scalar args 한 번만 set. arg 0/1/16-21/31/10 만 동적.
        let q_nb1 = (N_H_Q * DK * 4) as u64;
        let q_nb2 = (DK * 4) as u64;
        let q_nb3 = q_nb1;
        let o_nb1 = (DV * 4) as u64;
        let o_nb2 = (N_H_Q * DV * 4) as u64;
        let o_nb3 = o_nb2;

        let scale = 1.0f32 / (DK as f32).sqrt();
        let n_q = 1i32;
        let is_causal = 0i32;
        let n_head = N_H_Q as i32;
        let n_head_kv = N_H_KV as i32;
        let max_bias = 0.0f32;
        let m0 = 0.0f32;
        let m1 = 0.0f32;
        let n_head_log2 = 0i32;
        let logit_softcap = 0.0f32;
        let zero_u64 = 0u64;
        let mask_nb1 = 2u64; // F16 1 element stride (used as offset per k_idx)
        let mask_nb2 = (mask_elems * 2) as u64;
        let mask_nb3 = mask_nb2;
        let mask_ne = 1i32;

        // 두 kernel에 동일 정적 args 설정 — q_offset/n_kv/K-V strides/mask는 동적
        let setup_static = |k: &ocl::core::Kernel| -> anyhow::Result<()> {
            ocl::core::set_kernel_arg(k, 0, ArgVal::mem(&q_pool_buf))?;
            ocl::core::set_kernel_arg(k, 1, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(k, 2, ArgVal::mem(&k_buf))?;
            ocl::core::set_kernel_arg(k, 3, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(k, 4, ArgVal::mem(&v_buf))?;
            ocl::core::set_kernel_arg(k, 5, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(k, 6, ArgVal::mem(&o_buf))?;
            ocl::core::set_kernel_arg(k, 7, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(k, 8, ArgVal::scalar(&scale))?;
            ocl::core::set_kernel_arg(k, 9, ArgVal::scalar(&n_q))?;
            // arg 10 = n_kv (per-iter)
            ocl::core::set_kernel_arg(k, 11, ArgVal::scalar(&is_causal))?;
            ocl::core::set_kernel_arg(k, 12, ArgVal::scalar(&n_head))?;
            ocl::core::set_kernel_arg(k, 13, ArgVal::scalar(&q_nb1))?;
            ocl::core::set_kernel_arg(k, 14, ArgVal::scalar(&q_nb2))?;
            ocl::core::set_kernel_arg(k, 15, ArgVal::scalar(&q_nb3))?;
            // args 16-21 = K/V strides (per-layout)
            ocl::core::set_kernel_arg(k, 22, ArgVal::scalar(&o_nb1))?;
            ocl::core::set_kernel_arg(k, 23, ArgVal::scalar(&o_nb2))?;
            ocl::core::set_kernel_arg(k, 24, ArgVal::scalar(&o_nb3))?;
            ocl::core::set_kernel_arg(k, 25, ArgVal::scalar(&max_bias))?;
            ocl::core::set_kernel_arg(k, 26, ArgVal::scalar(&m0))?;
            ocl::core::set_kernel_arg(k, 27, ArgVal::scalar(&m1))?;
            ocl::core::set_kernel_arg(k, 28, ArgVal::scalar(&n_head_log2))?;
            ocl::core::set_kernel_arg(k, 29, ArgVal::scalar(&logit_softcap))?;
            ocl::core::set_kernel_arg(k, 30, ArgVal::scalar(&n_head_kv))?;
            // arg 31 = mask_void (per-variant)
            ocl::core::set_kernel_arg(k, 32, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(k, 33, ArgVal::scalar(&mask_nb1))?;
            ocl::core::set_kernel_arg(k, 34, ArgVal::scalar(&mask_nb2))?;
            ocl::core::set_kernel_arg(k, 35, ArgVal::scalar(&mask_nb3))?;
            ocl::core::set_kernel_arg(k, 36, ArgVal::scalar(&mask_ne))?;
            ocl::core::set_kernel_arg(k, 37, ArgVal::scalar(&mask_ne))?;
            ocl::core::set_kernel_arg(k, 38, ArgVal::mem_null())?;
            ocl::core::set_kernel_arg(k, 39, ArgVal::scalar(&zero_u64))?;
            Ok(())
        };
        setup_static(&our_kernel)?;
        setup_static(&llama_kernel)?;

        // Pollute buffer: POLLUTE_MAX_MB만큼 float4로 alloc. 초기값 non-zero.
        let pollute_elems4: usize = (POLLUTE_MAX_MB * 1024 * 1024) / 16;
        let pollute_host: Vec<f32> = vec![0.001; pollute_elems4 * 4];
        let pollute_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                context.as_core(),
                ocl::core::MEM_READ_WRITE,
                pollute_elems4 * 4,
                None,
            )?
        };
        unsafe {
            ocl::core::enqueue_write_buffer(
                &queue,
                &pollute_buf,
                true,
                0,
                std::slice::from_raw_parts(
                    pollute_host.as_ptr() as *const u8,
                    pollute_elems4 * 4 * 4,
                ),
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
        let pollute_program = Program::builder()
            .devices(device)
            .src(POLLUTE_SRC)
            .cmplr_opt(&cl_opts)
            .build(&context)?;
        let pollute_kernel = ocl::core::create_kernel(&pollute_program, "pollute")?;

        Ok(State {
            context,
            queue,
            kernels: vec![
                ("llm_rs2".to_string(), our_kernel),
                ("llama.cpp".to_string(), llama_kernel),
            ],
            q_pool_buf,
            k_buf,
            v_buf,
            k_bufs_frag,
            v_bufs_frag,
            o_buf,
            mask_buf,
            pollute_buf,
            pollute_kernel,
            q_slot_bytes,
            gws: [Q1_WG_SIZE, N_H_Q, 1],
            lws: Some([Q1_WG_SIZE, 1, 1]),
        })
    }

    /// 한 measurement: variant.num_dispatches 만큼 dispatch.
    /// 기본은 event profiling 합산, LLMRS_MB_WALLCLOCK=1이면 Instant + queue.finish()로
    /// wall-clock 측정. 반환값은 measurement 1회의 μs (= per-token attention 비용).
    fn measure_one(
        state: &State,
        kernel: &ocl::core::Kernel,
        variant: Variant,
    ) -> anyhow::Result<f64> {
        let wallclock_mode = std::env::var_os("LLMRS_MB_WALLCLOCK").is_some();
        let mut events: Vec<Event> = if wallclock_mode {
            Vec::new()
        } else {
            (0..variant.num_dispatches).map(|_| Event::null()).collect()
        };
        let pollute_elems4: i32 = ((variant.pollute_mb * 1024 * 1024) / 16) as i32;
        let pollute_gws: [usize; 3] = [pollute_elems4.max(1) as usize, 1, 1];
        let pollute_lws: Option<[usize; 3]> = Some([64, 1, 1]);
        if variant.pollute_mb > 0 {
            ocl::core::set_kernel_arg(&state.pollute_kernel, 0, ArgVal::mem(&state.pollute_buf))?;
            ocl::core::set_kernel_arg(&state.pollute_kernel, 1, ArgVal::scalar(&pollute_elems4))?;
        }
        let wc_start = if wallclock_mode {
            Some(std::time::Instant::now())
        } else {
            None
        };
        for i in 0..variant.num_dispatches {
            if variant.pollute_mb > 0 {
                unsafe {
                    ocl::core::enqueue_kernel(
                        &state.queue,
                        &state.pollute_kernel,
                        1,
                        None,
                        &pollute_gws,
                        pollute_lws,
                        None::<&Event>,
                        None::<&mut Event>,
                    )?;
                }
            }
            if variant.vary_q {
                let q_off = (i as u64) * state.q_slot_bytes;
                ocl::core::set_kernel_arg(kernel, 1, ArgVal::scalar(&q_off))?;
            }
            if variant.fragmented_kv {
                ocl::core::set_kernel_arg(kernel, 2, ArgVal::mem(&state.k_bufs_frag[i]))?;
                ocl::core::set_kernel_arg(kernel, 4, ArgVal::mem(&state.v_bufs_frag[i]))?;
            }
            unsafe {
                if wallclock_mode {
                    ocl::core::enqueue_kernel(
                        &state.queue,
                        kernel,
                        2,
                        None,
                        &state.gws,
                        state.lws,
                        None::<&Event>,
                        None::<&mut Event>,
                    )?;
                } else {
                    ocl::core::enqueue_kernel(
                        &state.queue,
                        kernel,
                        2,
                        None,
                        &state.gws,
                        state.lws,
                        None::<&Event>,
                        Some(&mut events[i]),
                    )?;
                }
            }
        }
        // Fragmented 종료 후 공유 K/V로 복원 (다음 variant 영향 방지)
        if variant.fragmented_kv {
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::mem(&state.k_buf))?;
            ocl::core::set_kernel_arg(kernel, 4, ArgVal::mem(&state.v_buf))?;
        }
        ocl::core::finish(&state.queue)?;
        if let Some(t0) = wc_start {
            return Ok(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        let mut sum_us = 0.0;
        for ev in &events {
            let s = ocl::core::get_event_profiling_info(ev, ProfilingInfo::Start)?.time()?;
            let e = ocl::core::get_event_profiling_info(ev, ProfilingInfo::End)?.time()?;
            sum_us += (e.saturating_sub(s)) as f64 / 1000.0;
        }
        Ok(sum_us)
    }

    pub fn measure_matrix(
        state: &State,
        layouts: &[(&'static str, (u64, u64, u64))],
    ) -> anyhow::Result<Vec<(String, String, String, Vec<(i32, f64)>)>> {
        // header
        println!();
        println!("engine,layout,variant,n_kv,median_us,mean_us,min_us,max_us,iters");

        let mut all_results: Vec<(String, String, String, Vec<(i32, f64)>)> = Vec::new();

        // engine 외부 루프: 같은 KV 데이터/같은 dispatch params로 두 kernel 직접 비교
        for (engine_name, kernel) in &state.kernels {
            for (layout_name, (k_nb1, k_nb2, k_nb3)) in layouts {
                ocl::core::set_kernel_arg(kernel, 16, ArgVal::scalar(k_nb1))?;
                ocl::core::set_kernel_arg(kernel, 17, ArgVal::scalar(k_nb2))?;
                ocl::core::set_kernel_arg(kernel, 18, ArgVal::scalar(k_nb3))?;
                ocl::core::set_kernel_arg(kernel, 19, ArgVal::scalar(k_nb1))?;
                ocl::core::set_kernel_arg(kernel, 20, ArgVal::scalar(k_nb2))?;
                ocl::core::set_kernel_arg(kernel, 21, ArgVal::scalar(k_nb3))?;

                for variant in VARIANTS {
                    if variant.use_mask {
                        ocl::core::set_kernel_arg(kernel, 31, ArgVal::mem(&state.mask_buf))?;
                    } else {
                        ocl::core::set_kernel_arg(kernel, 31, ArgVal::mem_null())?;
                    }
                    let zero_u64 = 0u64;
                    if !variant.vary_q {
                        ocl::core::set_kernel_arg(kernel, 1, ArgVal::scalar(&zero_u64))?;
                    }

                    let mut points: Vec<(i32, f64)> = Vec::new();

                    for &n_kv in N_KV_VALUES {
                        ocl::core::set_kernel_arg(kernel, 10, ArgVal::scalar(&n_kv))?;

                        for _ in 0..WARMUP_ITERS {
                            let _ = measure_one(state, kernel, *variant)?;
                        }

                        let mut samples: Vec<f64> = Vec::with_capacity(MEASURE_ITERS);
                        for _ in 0..MEASURE_ITERS {
                            samples.push(measure_one(state, kernel, *variant)?);
                        }

                        let mut sorted = samples.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let median = sorted[sorted.len() / 2];
                        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
                        let min = sorted[0];
                        let max = *sorted.last().unwrap();

                        println!(
                            "{},{},{},{},{:.2},{:.2},{:.2},{:.2},{}",
                            engine_name,
                            layout_name,
                            variant.name,
                            n_kv,
                            median,
                            mean,
                            min,
                            max,
                            samples.len()
                        );
                        points.push((n_kv, median));
                    }

                    all_results.push((
                        engine_name.clone(),
                        layout_name.to_string(),
                        variant.name.to_string(),
                        points,
                    ));
                }
            }
        }

        Ok(all_results)
    }

    pub fn print_slope_table(results: &[(String, String, String, Vec<(i32, f64)>)]) {
        println!();
        println!("# Slope (μs / n_kv) per (engine, layout, variant) — least squares");
        println!("# engine,layout,variant,slope_us_per_n_kv,intercept_us");
        for (engine, layout, variant, pts) in results {
            let (slope, intercept) = lsq(pts);
            println!(
                "# {},{},{},{:.4},{:.2}",
                engine, layout, variant, slope, intercept
            );
        }
    }

    fn lsq(pts: &[(i32, f64)]) -> (f64, f64) {
        let n = pts.len() as f64;
        let sx: f64 = pts.iter().map(|(x, _)| *x as f64).sum();
        let sy: f64 = pts.iter().map(|(_, y)| *y).sum();
        let sxx: f64 = pts.iter().map(|(x, _)| (*x as f64).powi(2)).sum();
        let sxy: f64 = pts.iter().map(|(x, y)| (*x as f64) * *y).sum();
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let intercept = (sy - slope * sx) / n;
        (slope, intercept)
    }

    pub fn print_cross_run_summary(results: &[(String, String, String, Vec<(i32, f64)>)]) {
        // (engine, variant) → slope (HeadMajor 만)
        let mut by_var: std::collections::BTreeMap<String, Vec<(String, f64)>> =
            std::collections::BTreeMap::new();
        for (engine, layout, variant, pts) in results {
            if layout != "HeadMajor" {
                continue;
            }
            let (slope, _) = lsq(pts);
            by_var
                .entry(variant.clone())
                .or_default()
                .push((engine.clone(), slope));
        }

        println!();
        println!("# Cross-run verdict — HeadMajor only, per variant");
        println!("# variant,engine_a,slope_a,engine_b,slope_b,ratio_a/b");
        for (variant, engines) in &by_var {
            if engines.len() != 2 {
                continue;
            }
            let (ea, sa) = &engines[0];
            let (eb, sb) = &engines[1];
            let ratio = sa / sb;
            let verdict = if ratio < 0.95 {
                format!(
                    "{} faster ({:.1}% advantage)",
                    ea,
                    (1.0 / ratio - 1.0) * 100.0
                )
            } else if ratio > 1.05 {
                format!("{} faster ({:.1}% advantage)", eb, (ratio - 1.0) * 100.0)
            } else {
                "within ±5% (TIE)".to_string()
            };
            println!(
                "# {:16},{},{:.4},{},{:.4},{:.3}x  →  {}",
                variant, ea, sa, eb, sb, ratio, verdict
            );
        }
    }
}

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    let state = bench::init()?;

    let kv_elem: u64 = 2;
    let head_major = (
        (bench::DK as u64) * kv_elem,
        (bench::CAP * bench::DK) as u64 * kv_elem,
        (bench::N_H_KV * bench::CAP * bench::DK) as u64 * kv_elem,
    );
    let pos_major = (
        (bench::N_H_KV * bench::DK) as u64 * kv_elem,
        (bench::DK as u64) * kv_elem,
        (bench::N_H_KV * bench::CAP * bench::DK) as u64 * kv_elem,
    );
    let layouts: &[(&'static str, (u64, u64, u64))] =
        &[("HeadMajor", head_major), ("PosMajor", pos_major)];

    let results = bench::measure_matrix(&state, layouts)?;
    bench::print_slope_table(&results);
    bench::print_cross_run_summary(&results);
    Ok(())
}
