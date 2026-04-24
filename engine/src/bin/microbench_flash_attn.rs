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
//! 목적 4 (production-like op chain, 2026-04-15 확장):
//!   pollute 기반 가설이 기각된 뒤(streaming 32MB, stride 18B 모두 slope 변화 < 0.1 μs/n_kv)
//!   남은 가설은 "production decode layer의 실제 op sequence + weight 점유 + scheduler 상태"이다.
//!   Q1 앞/뒤에 실제 production kernel (mul_mv_q4_0_f32, rope_simple, kv_scatter_f32_to_f16,
//!   rms_norm_opt_f4, silu_mul_simple) 을 단계적으로 삽입하여 Q1 slope 변화 측정.
//!   - Repeat28WithQKV         : Q1 앞 matmul_qkv (Q4_0 GEMV)
//!   - Repeat28WithQKVRope     : + rope(Q) + rope(K)
//!   - Repeat28WithQKVRopeKv   : + kv_scatter
//!   - Repeat28WithAttnFFN     : Q1 뒤 WO + rms_norm + gate/up + silu_mul + down
//!   - Repeat28FullLayer       : 전체 layer chain (production 가장 유사)
//!   140 Q4_0 weight buffers (28 layer × 5 matmul) 가 점유된 상태에서 측정하여
//!   weight working-set 영향까지 재현.
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
    /// Hidden dimension (Qwen 2.5-1.5B)
    pub const HIDDEN: usize = N_H_Q * DK; // 12 * 128 = 1536
    /// FFN 내부 hidden (Qwen 2.5-1.5B)
    pub const FFN_HIDDEN: usize = 8960;
    /// QKV 출력 = Q + K + V = n_h_q*dk + n_h_kv*dk + n_h_kv*dv
    pub const QKV_OUT: usize = (N_H_Q + 2 * N_H_KV) * DK; // 2048
    /// Q4_0 block size (elements per block)
    pub const Q4_0_QK: usize = 32;
    /// Q4_0 block bytes (half scale + 16 nibble bytes)
    pub const Q4_0_BLOCK_BYTES: usize = 18;

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
        /// true = 18B stride struct-pattern pollute (Q4_0 matmul 유사)
        /// false = float4 streaming (기존)
        pub pollute_stride: bool,
        /// Production decode layer chain 흉내용 추가 dispatch.
        /// Q1 앞에 matmul_qkv (Q4_0 GEMV, HIDDEN → QKV_OUT) 1회.
        pub insert_qkv: bool,
        /// Q1 앞에 rope(Q) + rope(K) 2회.
        pub insert_rope: bool,
        /// Q1 앞에 kv_scatter_f32_to_f16 1회.
        pub insert_kv_scatter: bool,
        /// Q1 뒤에 matmul_wo + add_rms_norm_oop + gate/up/down matmul + silu_mul 삽입.
        /// Layer chain 중 Q1 앞/뒤 전체 op 집합을 재현.
        pub insert_post_attn: bool,
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
            pollute_stride: false,
            insert_qkv: false,
            insert_rope: false,
            insert_kv_scatter: false,
            insert_post_attn: false,
        },
        Variant {
            name: "Repeat28",
            num_dispatches: N_LAYERS,
            use_mask: false,
            vary_q: false,
            pollute_mb: 0,
            fragmented_kv: false,
            pollute_stride: false,
            insert_qkv: false,
            insert_rope: false,
            insert_kv_scatter: false,
            insert_post_attn: false,
        },
        Variant {
            name: "Repeat28Mask",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: false,
            pollute_mb: 0,
            fragmented_kv: false,
            pollute_stride: false,
            insert_qkv: false,
            insert_rope: false,
            insert_kv_scatter: false,
            insert_post_attn: false,
        },
        Variant {
            name: "Repeat28MaskQ",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: true,
            pollute_mb: 0,
            fragmented_kv: false,
            pollute_stride: false,
            insert_qkv: false,
            insert_rope: false,
            insert_kv_scatter: false,
            insert_post_attn: false,
        },
        Variant {
            name: "Repeat28Pollute8",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: true,
            pollute_mb: 8,
            fragmented_kv: false,
            pollute_stride: false,
            insert_qkv: false,
            insert_rope: false,
            insert_kv_scatter: false,
            insert_post_attn: false,
        },
        Variant {
            name: "Repeat28Pollute32",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: true,
            pollute_mb: 32,
            fragmented_kv: false,
            pollute_stride: false,
            insert_qkv: false,
            insert_rope: false,
            insert_kv_scatter: false,
            insert_post_attn: false,
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
            pollute_stride: false,
            insert_qkv: false,
            insert_rope: false,
            insert_kv_scatter: false,
            insert_post_attn: false,
        },
        // Q4_0 matmul 18B stride access pattern — streaming(float4) 대비 slope 차이 검증.
        // Q4_0 block_q4_0 구조: half(2B) + 16 nibble bytes = 18B per block.
        Variant {
            name: "Repeat28PolluteStride8",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: true,
            pollute_mb: 8,
            fragmented_kv: false,
            pollute_stride: true,
            insert_qkv: false,
            insert_rope: false,
            insert_kv_scatter: false,
            insert_post_attn: false,
        },
        Variant {
            name: "Repeat28PolluteStride32",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: true,
            pollute_mb: 32,
            fragmented_kv: false,
            pollute_stride: true,
            insert_qkv: false,
            insert_rope: false,
            insert_kv_scatter: false,
            insert_post_attn: false,
        },
        // === Production-like op chain variants (단계별 누적) ===
        // Base: Repeat28MaskQ. Q1 주변에 실제 production decode layer에서
        // 실행되는 op들을 단계적으로 추가. 각 variant가 Q1 slope에 주는 영향
        // 격리 측정 → production gap +1.6 μs/n_kv 구성 요소 식별.
        //
        // Q1 앞: matmul_qkv (Q4_0 GEMV, HIDDEN → QKV_OUT)
        Variant {
            name: "Repeat28WithQKV",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: true,
            pollute_mb: 0,
            fragmented_kv: false,
            pollute_stride: false,
            insert_qkv: true,
            insert_rope: false,
            insert_kv_scatter: false,
            insert_post_attn: false,
        },
        // + rope(Q) + rope(K)
        Variant {
            name: "Repeat28WithQKVRope",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: true,
            pollute_mb: 0,
            fragmented_kv: false,
            pollute_stride: false,
            insert_qkv: true,
            insert_rope: true,
            insert_kv_scatter: false,
            insert_post_attn: false,
        },
        // + kv_scatter_f32_to_f16
        Variant {
            name: "Repeat28WithQKVRopeKv",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: true,
            pollute_mb: 0,
            fragmented_kv: false,
            pollute_stride: false,
            insert_qkv: true,
            insert_rope: true,
            insert_kv_scatter: true,
            insert_post_attn: false,
        },
        // Q1 뒤: matmul_wo + add_rms_norm + matmul_gate + matmul_up + silu_mul + matmul_down
        Variant {
            name: "Repeat28WithAttnFFN",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: true,
            pollute_mb: 0,
            fragmented_kv: false,
            pollute_stride: false,
            insert_qkv: false,
            insert_rope: false,
            insert_kv_scatter: false,
            insert_post_attn: true,
        },
        // 전체 layer chain: QKV + rope + kv_scatter + Q1 + WO + add_rms_norm + FFN
        Variant {
            name: "Repeat28FullLayer",
            num_dispatches: N_LAYERS,
            use_mask: true,
            vary_q: true,
            pollute_mb: 0,
            fragmented_kv: false,
            pollute_stride: false,
            insert_qkv: true,
            insert_rope: true,
            insert_kv_scatter: true,
            insert_post_attn: true,
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
        pub pollute_stride_kernel: ocl::core::Kernel,
        pub pollute_sink_buf: Mem,
        pub q_slot_bytes: u64,
        pub gws: [usize; 3],
        pub lws: Option<[usize; 3]>,
        // === Production-like op chain state (insert_* variants에서만 사용) ===
        /// Hidden state F32 [HIDDEN] (rms_norm/matmul 입력)
        pub hidden_buf: Mem,
        /// Residual buffer F32 [HIDDEN] (add_rms_norm_oop)
        pub residual_buf: Mem,
        /// RMSNorm weight F32 [HIDDEN]
        pub rms_weight_buf: Mem,
        /// QKV output F32 [QKV_OUT] (matmul_qkv 출력, rope/kv_scatter 입력)
        pub qkv_out_buf: Mem,
        /// K tmp F32 [N_H_KV * DK] (rope/kv_scatter K 입력)
        pub k_tmp_buf: Mem,
        /// V tmp F32 [N_H_KV * DK] (kv_scatter V 입력)
        pub v_tmp_buf: Mem,
        /// WO output F32 [HIDDEN]
        pub wo_out_buf: Mem,
        /// FFN gate/up output F32 [FFN_HIDDEN]
        pub ffn_gate_buf: Mem,
        pub ffn_up_buf: Mem,
        /// FFN down output F32 [HIDDEN]
        pub ffn_down_buf: Mem,
        /// Per-layer Q4_0 weight buffers (28 layers × 5 matmul = 140 buffers)
        pub w_qkv_bufs: Vec<Mem>, // [HIDDEN → QKV_OUT]
        pub w_wo_bufs: Vec<Mem>,   // [HIDDEN → HIDDEN]
        pub w_gate_bufs: Vec<Mem>, // [HIDDEN → FFN_HIDDEN]
        pub w_up_bufs: Vec<Mem>,   // [HIDDEN → FFN_HIDDEN]
        pub w_down_bufs: Vec<Mem>, // [FFN_HIDDEN → HIDDEN]
        /// Q4_0 matmul kernel (kernel_mul_mat_q4_0_f32, reused for all matmul)
        pub q4_0_matmul_kernel: ocl::core::Kernel,
        /// RoPE kernel (kernel_rope_simple, inplace)
        pub rope_kernel: ocl::core::Kernel,
        /// KV scatter kernel (kernel_kv_scatter_f32_to_f16)
        pub kv_scatter_kernel: ocl::core::Kernel,
        /// RMSNorm kernel (kernel_rms_norm_opt_f4)
        pub rms_norm_kernel: ocl::core::Kernel,
        /// SiLU mul kernel (kernel_silu_mul_simple)
        pub silu_mul_kernel: ocl::core::Kernel,
        /// Dummy 1-element F32 buffer for score output arg (write_scores=0 시 미사용)
        pub score_dummy_buf: Mem,
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

    /// Cache-pollute 커널 (stride variant): 18B stride struct read — Q4_0 block_q4_0 access 흉내.
    /// block_q4_0 = { half d; uint8_t qs[16]; } = 2 + 16 = 18 bytes per block.
    /// 각 thread가 struct 1개씩 읽어 volatile sum에 기여 (DCE 방지용 sink write 1 lane).
    const POLLUTE_STRIDE_SRC: &str = r#"
__kernel void pollute_stride(
    __global const uchar* buf,
    const int n_blocks,
    __global float* sink
) {
    const int gid = get_global_id(0);
    if (gid >= n_blocks) return;
    const int base = gid * 18;
    /* half scale — 2 bytes */
    float d = vload_half(0, (const __global half*)(buf + base));
    /* 16 nibble bytes — 모두 읽기 (prefetch 친화적 contiguous) */
    uchar acc = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc ^= buf[base + 2 + i];
    if (gid == 0) sink[0] = d * (float)acc;  /* DCE 방지 (write 1 lane) */
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

        // dummy score buffer (write_scores=0이면 미사용, 그러나 cl_mem은 valid 해야 함)
        let score_dummy_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                context.as_core(),
                ocl::core::MEM_READ_WRITE,
                1,
                None,
            )?
        };

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
        // args 40-43: score output (flash_attn_f32_f16.cl 커밋 3096de4에서 추가).
        // llama.cpp 커널에는 해당 인자가 없으므로 our_kernel 에만 설정.
        ocl::core::set_kernel_arg(&our_kernel, 40, ArgVal::mem(&score_dummy_buf))?;
        ocl::core::set_kernel_arg(&our_kernel, 41, ArgVal::scalar(&0i32))?; // score_layer_offset
        ocl::core::set_kernel_arg(&our_kernel, 42, ArgVal::scalar(&0i32))?; // score_stride
        ocl::core::set_kernel_arg(&our_kernel, 43, ArgVal::scalar(&0i32))?; // write_scores=0 (no-op)
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

        // Stride-pattern pollute: 18B struct (Q4_0 block_q4_0) per thread.
        // pollute_buf(uchar*)를 재사용하여 추가 alloc 없이 stride access 흉내.
        let pollute_stride_program = Program::builder()
            .devices(device)
            .src(POLLUTE_STRIDE_SRC)
            .cmplr_opt(&cl_opts)
            .build(&context)?;
        let pollute_stride_kernel =
            ocl::core::create_kernel(&pollute_stride_program, "pollute_stride")?;

        // Sink buffer: float 1개 (DCE 방지용 lane-0 write).
        let pollute_sink_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                context.as_core(),
                ocl::core::MEM_READ_WRITE,
                1,
                None,
            )?
        };

        // === Production-like op chain 리소스 alloc ===
        //
        // F32 activation buffers (hidden/residual/qkv_out/k_tmp/v_tmp/wo/ffn_gate/up/down).
        // rms_weight: RMSNorm weight (F32 [HIDDEN]).
        let alloc_f32 = |elems: usize| -> anyhow::Result<Mem> {
            let buf = unsafe {
                ocl::core::create_buffer::<_, f32>(
                    context.as_core(),
                    ocl::core::MEM_READ_WRITE,
                    elems,
                    None,
                )?
            };
            // zero-init (write 1 byte 0 패턴으로 충분 — 실제 값은 무의미, cache 상태만 중요)
            let zero: Vec<f32> = vec![0.0; elems];
            unsafe {
                ocl::core::enqueue_write_buffer(
                    &queue,
                    &buf,
                    true,
                    0,
                    std::slice::from_raw_parts(zero.as_ptr() as *const u8, elems * 4),
                    None::<&Event>,
                    None::<&mut Event>,
                )?;
            }
            Ok(buf)
        };
        let hidden_buf = alloc_f32(HIDDEN)?;
        let residual_buf = alloc_f32(HIDDEN)?;
        let rms_weight_buf = alloc_f32(HIDDEN)?;
        let qkv_out_buf = alloc_f32(QKV_OUT)?;
        let k_tmp_buf = alloc_f32(N_H_KV * DK)?;
        let v_tmp_buf = alloc_f32(N_H_KV * DK)?;
        let wo_out_buf = alloc_f32(HIDDEN)?;
        let ffn_gate_buf = alloc_f32(FFN_HIDDEN)?;
        let ffn_up_buf = alloc_f32(FFN_HIDDEN)?;
        let ffn_down_buf = alloc_f32(HIDDEN)?;

        // Q4_0 weight buffers: per-layer, per-matmul-type.
        // Production과 동일 구조(140개 buffer)로 driver bookkeeping/L2 working-set 재현.
        // Q4_0 layout: K*N/QK 블록, 블록당 18B (half scale + 16 nibble bytes).
        let alloc_q4_0 = |k_dim: usize, n_dim: usize| -> anyhow::Result<Mem> {
            let n_blocks = (k_dim * n_dim) / Q4_0_QK;
            let bytes = n_blocks * Q4_0_BLOCK_BYTES;
            let buf = unsafe {
                ocl::core::create_buffer::<_, u8>(
                    context.as_core(),
                    ocl::core::MEM_READ_WRITE,
                    bytes,
                    None,
                )?
            };
            // 초기값: zero. 정확한 결과값 불필요 (L2 state + dispatch pattern만 중요).
            let zero: Vec<u8> = vec![0u8; bytes];
            unsafe {
                ocl::core::enqueue_write_buffer(
                    &queue,
                    &buf,
                    true,
                    0,
                    &zero,
                    None::<&Event>,
                    None::<&mut Event>,
                )?;
            }
            Ok(buf)
        };
        let mut w_qkv_bufs: Vec<Mem> = Vec::with_capacity(N_LAYERS);
        let mut w_wo_bufs: Vec<Mem> = Vec::with_capacity(N_LAYERS);
        let mut w_gate_bufs: Vec<Mem> = Vec::with_capacity(N_LAYERS);
        let mut w_up_bufs: Vec<Mem> = Vec::with_capacity(N_LAYERS);
        let mut w_down_bufs: Vec<Mem> = Vec::with_capacity(N_LAYERS);
        for _ in 0..N_LAYERS {
            w_qkv_bufs.push(alloc_q4_0(HIDDEN, QKV_OUT)?);
            w_wo_bufs.push(alloc_q4_0(HIDDEN, HIDDEN)?);
            w_gate_bufs.push(alloc_q4_0(HIDDEN, FFN_HIDDEN)?);
            w_up_bufs.push(alloc_q4_0(HIDDEN, FFN_HIDDEN)?);
            w_down_bufs.push(alloc_q4_0(FFN_HIDDEN, HIDDEN)?);
        }

        // Production kernel 재사용 — include_str!로 원본 그대로 컴파일.
        // cl_opts는 flash_attn과 같은 fast-math 플래그만 전달 (DK/DV define 불필요).
        //
        // Note: simple_ops.cl은 subgroup 의존 — Adreno에서는 OK, 호스트 OpenCL 일부는
        // 실패할 수 있음. 실패 시 nosub fallback 사용.
        let simple_ops_src = include_str!("../../kernels/simple_ops.cl");
        let simple_ops_fallback_src = include_str!("../../kernels/fallback/simple_ops_nosub.cl");
        let simple_ops_program = match Program::builder()
            .devices(device)
            .src(simple_ops_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => p,
            Err(_) => Program::builder()
                .devices(device)
                .src(simple_ops_fallback_src)
                .cmplr_opt(&cl_opts)
                .build(&context)?,
        };
        let rope_kernel = ocl::core::create_kernel(&simple_ops_program, "kernel_rope_simple")?;
        let kv_scatter_kernel =
            ocl::core::create_kernel(&simple_ops_program, "kernel_kv_scatter_f32_to_f16")?;
        // rms_norm: opt_f4 (production 경로)는 HIDDEN % 4 == 0 조건 OK (1536)
        let rms_norm_kernel =
            ocl::core::create_kernel(&simple_ops_program, "kernel_rms_norm_opt_f4")?;
        let silu_mul_kernel =
            ocl::core::create_kernel(&simple_ops_program, "kernel_silu_mul_simple")?;

        // Q4_0 matmul kernel — production `kernel_mul_mat_q4_0_f32`. GEMV (m==1) 경로.
        let q4_0_src = include_str!("../../kernels/mul_mv_q4_0_f32.cl");
        let q4_0_fallback_src = include_str!("../../kernels/fallback/mul_mv_q4_0_f32_nosub.cl");
        let q4_0_program = match Program::builder()
            .devices(device)
            .src(q4_0_src)
            .cmplr_opt(&cl_opts)
            .build(&context)
        {
            Ok(p) => p,
            Err(_) => Program::builder()
                .devices(device)
                .src(q4_0_fallback_src)
                .cmplr_opt(&cl_opts)
                .build(&context)?,
        };
        let q4_0_matmul_kernel =
            ocl::core::create_kernel(&q4_0_program, "kernel_mul_mat_q4_0_f32")?;

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
            pollute_stride_kernel,
            pollute_sink_buf,
            q_slot_bytes,
            gws: [Q1_WG_SIZE, N_H_Q, 1],
            lws: Some([Q1_WG_SIZE, 1, 1]),
            hidden_buf,
            residual_buf,
            rms_weight_buf,
            qkv_out_buf,
            k_tmp_buf,
            v_tmp_buf,
            wo_out_buf,
            ffn_gate_buf,
            ffn_up_buf,
            ffn_down_buf,
            w_qkv_bufs,
            w_wo_bufs,
            w_gate_bufs,
            w_up_bufs,
            w_down_bufs,
            q4_0_matmul_kernel,
            rope_kernel,
            kv_scatter_kernel,
            rms_norm_kernel,
            silu_mul_kernel,
            score_dummy_buf,
        })
    }

    /// Production fake QKV matmul dispatch (Q4_0 GEMV, HIDDEN → QKV_OUT).
    /// Production `matmul_q4_0` (mul_mv_q4_0_f32.cl) 그대로 사용.
    /// layer_idx별로 다른 weight buffer를 바인딩하여 140-weight working-set 재현.
    fn dispatch_fake_matmul_q4_0(
        state: &State,
        weight_buf: &Mem,
        src_buf: &Mem,
        dst_buf: &Mem,
        k_dim: usize, // input dim
        n_dim: usize, // output dim
    ) -> anyhow::Result<()> {
        let kernel = &state.q4_0_matmul_kernel;
        let ne00 = k_dim as i32;
        let ne01 = n_dim as i32;
        let ne02 = 1i32;
        let ne10 = k_dim as i32;
        let ne12 = k_dim as i32; // m=1
        let ne0 = n_dim as i32;
        let ne1 = n_dim as i32; // m=1
        let r2 = 1i32;
        let r3 = 1i32;
        let zero_u64 = 0u64;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(weight_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::mem(src_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ArgVal::scalar(&zero_u64))?;
            ocl::core::set_kernel_arg(kernel, 4, ArgVal::mem(dst_buf))?;
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

            let local_work_size: [usize; 3] = [64, 1, 1];
            let group_size_0 = n_dim.div_ceil(4);
            let global_work_size: [usize; 3] = [group_size_0 * local_work_size[0], 1, 1];
            ocl::core::enqueue_kernel(
                &state.queue,
                kernel,
                3,
                None,
                &global_work_size,
                Some(local_work_size),
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
        Ok(())
    }

    /// RoPE (in-place on `x_buf`) — production `kernel_rope_simple` 그대로.
    fn dispatch_fake_rope(state: &State, x_buf: &Mem, num_heads: usize) -> anyhow::Result<()> {
        let kernel = &state.rope_kernel;
        let head_dim_i32 = DK as i32;
        let num_heads_i32 = num_heads as i32;
        let seq_len_i32 = 1i32;
        let start_pos_i32 = 0i32;
        let theta: f32 = 1_000_000.0; // Qwen rope_theta (정확한 값 불필요)
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::scalar(&head_dim_i32))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::scalar(&num_heads_i32))?;
            ocl::core::set_kernel_arg(kernel, 3, ArgVal::scalar(&seq_len_i32))?;
            ocl::core::set_kernel_arg(kernel, 4, ArgVal::scalar(&start_pos_i32))?;
            ocl::core::set_kernel_arg(kernel, 5, ArgVal::scalar(&theta))?;

            let work_size = num_heads * (DK / 2);
            let gws: [usize; 3] = [work_size, 1, 1];
            ocl::core::enqueue_kernel(
                &state.queue,
                kernel,
                1,
                None,
                &gws,
                None,
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
        Ok(())
    }

    /// KV scatter F32 → F16 (production `kernel_kv_scatter_f32_to_f16`).
    fn dispatch_fake_kv_scatter(state: &State) -> anyhow::Result<()> {
        let kernel = &state.kv_scatter_kernel;
        let head_dim_i32 = DK as i32;
        let capacity_i32 = CAP as i32;
        let write_pos_i32 = 0i32; // 항상 0 OK — cache state는 동일 slot 덮어쓰기
        let n_elems = N_H_KV * DK;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(&state.k_tmp_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::mem(&state.v_tmp_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::mem(&state.k_buf))?;
            ocl::core::set_kernel_arg(kernel, 3, ArgVal::mem(&state.v_buf))?;
            ocl::core::set_kernel_arg(kernel, 4, ArgVal::scalar(&head_dim_i32))?;
            ocl::core::set_kernel_arg(kernel, 5, ArgVal::scalar(&capacity_i32))?;
            ocl::core::set_kernel_arg(kernel, 6, ArgVal::scalar(&write_pos_i32))?;

            let gws: [usize; 3] = [n_elems.div_ceil(64) * 64, 1, 1];
            let lws: [usize; 3] = [64, 1, 1];
            ocl::core::enqueue_kernel(
                &state.queue,
                kernel,
                1,
                None,
                &gws,
                Some(lws),
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
        Ok(())
    }

    /// RMSNorm (in-place on `x_buf`, weight `rms_weight_buf`).
    /// Production `kernel_rms_norm_opt_f4` 그대로. HIDDEN=1536 → %4==0 OK.
    fn dispatch_fake_rms_norm(state: &State, x_buf: &Mem) -> anyhow::Result<()> {
        let kernel = &state.rms_norm_kernel;
        let dim_i32 = HIDDEN as i32;
        let eps: f32 = 1e-6;
        let add_unit: i32 = 0;
        let local_size: usize = 64;
        let local_mem_bytes = local_size * std::mem::size_of::<f32>();
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(x_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::mem(&state.rms_weight_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::scalar(&dim_i32))?;
            ocl::core::set_kernel_arg(kernel, 3, ArgVal::scalar(&eps))?;
            ocl::core::set_kernel_arg(kernel, 4, ArgVal::scalar(&add_unit))?;
            ocl::core::set_kernel_arg(kernel, 5, ArgVal::local::<f32>(&local_mem_bytes))?;

            let gws: [usize; 3] = [local_size, 1, 1]; // 1 row (decode)
            let lws: [usize; 3] = [local_size, 1, 1];
            ocl::core::enqueue_kernel(
                &state.queue,
                kernel,
                1,
                None,
                &gws,
                Some(lws),
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
        Ok(())
    }

    /// SiLU mul (gate = silu(gate) * up, in-place on `gate_buf`).
    /// Production `kernel_silu_mul_simple` — float4 elementwise.
    fn dispatch_fake_silu_mul(state: &State) -> anyhow::Result<()> {
        let kernel = &state.silu_mul_kernel;
        let size4 = (FFN_HIDDEN / 4) as i32;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(&state.ffn_gate_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::mem(&state.ffn_up_buf))?;
            ocl::core::set_kernel_arg(kernel, 2, ArgVal::scalar(&size4))?;

            let gws: [usize; 3] = [size4.max(1) as usize, 1, 1];
            ocl::core::enqueue_kernel(
                &state.queue,
                kernel,
                1,
                None,
                &gws,
                None,
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
        Ok(())
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
        // stride variant: n_blocks = (MB * 1024 * 1024) / 18
        let pollute_stride_n_blocks: i32 = if variant.pollute_mb > 0 {
            ((variant.pollute_mb * 1024 * 1024) / 18) as i32
        } else {
            1
        };
        let pollute_stride_gws: [usize; 3] = [pollute_stride_n_blocks.max(1) as usize, 1, 1];
        let pollute_stride_lws: Option<[usize; 3]> = Some([64, 1, 1]);
        if variant.pollute_mb > 0 {
            if variant.pollute_stride {
                ocl::core::set_kernel_arg(
                    &state.pollute_stride_kernel,
                    0,
                    ArgVal::mem(&state.pollute_buf),
                )?;
                ocl::core::set_kernel_arg(
                    &state.pollute_stride_kernel,
                    1,
                    ArgVal::scalar(&pollute_stride_n_blocks),
                )?;
                ocl::core::set_kernel_arg(
                    &state.pollute_stride_kernel,
                    2,
                    ArgVal::mem(&state.pollute_sink_buf),
                )?;
            } else {
                ocl::core::set_kernel_arg(
                    &state.pollute_kernel,
                    0,
                    ArgVal::mem(&state.pollute_buf),
                )?;
                ocl::core::set_kernel_arg(
                    &state.pollute_kernel,
                    1,
                    ArgVal::scalar(&pollute_elems4),
                )?;
            }
        }
        let wc_start = if wallclock_mode {
            Some(std::time::Instant::now())
        } else {
            None
        };
        for i in 0..variant.num_dispatches {
            if variant.pollute_mb > 0 {
                if variant.pollute_stride {
                    unsafe {
                        ocl::core::enqueue_kernel(
                            &state.queue,
                            &state.pollute_stride_kernel,
                            1,
                            None,
                            &pollute_stride_gws,
                            pollute_stride_lws,
                            None::<&Event>,
                            None::<&mut Event>,
                        )?;
                    }
                } else {
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
            }
            // === Production-like op chain: Q1 앞 dispatches ===
            if variant.insert_qkv {
                // matmul_qkv: hidden [HIDDEN] × W_qkv [HIDDEN → QKV_OUT] → qkv_out
                dispatch_fake_matmul_q4_0(
                    state,
                    &state.w_qkv_bufs[i],
                    &state.hidden_buf,
                    &state.qkv_out_buf,
                    HIDDEN,
                    QKV_OUT,
                )?;
            }
            if variant.insert_rope {
                // rope(Q) on qkv_out (첫 N_H_Q heads 영역).
                // kernel_rope_simple은 x_buf 시작부터 num_heads heads 처리.
                dispatch_fake_rope(state, &state.qkv_out_buf, N_H_Q)?;
                // rope(K) on k_tmp (K는 production에서 QKV output 중 K 슬라이스지만,
                // 측정 목적상 별도 buffer OK — access pattern은 동일 scale).
                dispatch_fake_rope(state, &state.k_tmp_buf, N_H_KV)?;
            }
            if variant.insert_kv_scatter {
                dispatch_fake_kv_scatter(state)?;
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
            // === Production-like op chain: Q1 뒤 dispatches ===
            if variant.insert_post_attn {
                // matmul_wo: o_buf (reshape to hidden) × W_wo → wo_out.
                // Note: o_buf는 [N_H_Q * DV] = HIDDEN elements (F32). matmul 입력으로
                // 그대로 사용. Q1 출력을 WO input으로 연결하여 production dependency 재현.
                dispatch_fake_matmul_q4_0(
                    state,
                    &state.w_wo_bufs[i],
                    &state.o_buf,
                    &state.wo_out_buf,
                    HIDDEN,
                    HIDDEN,
                )?;
                // add_rms_norm 대체: rms_norm (hidden += residual 단계는 생략,
                // L2 working-set + dispatch count만 재현).
                dispatch_fake_rms_norm(state, &state.hidden_buf)?;
                // FFN gate
                dispatch_fake_matmul_q4_0(
                    state,
                    &state.w_gate_bufs[i],
                    &state.hidden_buf,
                    &state.ffn_gate_buf,
                    HIDDEN,
                    FFN_HIDDEN,
                )?;
                // FFN up
                dispatch_fake_matmul_q4_0(
                    state,
                    &state.w_up_bufs[i],
                    &state.hidden_buf,
                    &state.ffn_up_buf,
                    HIDDEN,
                    FFN_HIDDEN,
                )?;
                // silu_mul (gate = silu(gate) * up)
                dispatch_fake_silu_mul(state)?;
                // FFN down
                dispatch_fake_matmul_q4_0(
                    state,
                    &state.w_down_bufs[i],
                    &state.ffn_gate_buf,
                    &state.ffn_down_buf,
                    FFN_HIDDEN,
                    HIDDEN,
                )?;
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
