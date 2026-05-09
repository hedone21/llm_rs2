//! M2.H 8th attempt — 6 sub-graph bisect (sub-graph composition drift).
//!
//! Background: M2.H 7th attempt (OOP variant + multi-output abstraction)
//! reports KvScatter outputs bit-equal vs reference but the 14-node chain
//! end-point still diverges by max_abs_err = 2.228 on `x_out`. graphAddNode
//! and graphFinalize both PASS, so the data-route inside the SDK-composed
//! graph deviates somewhere between sub-graphs. Standalone op tests are all
//! GREEN, so the drift comes from how composed sub-graphs forward NATIVE
//! intermediates between ops (rank handling, stride aliasing, output binding
//! mismatch, ...).
//!
//! 6 sub-graph plan (each end-point exported to host APP_READ):
//!   1. RmsNorm(pre) -> Q proj -> RoPE(Q)              => q_rot
//!   2. RmsNorm(pre) -> K proj -> V proj -> RoPE(K)
//!                   -> KvScatter                      => kcache, vcache
//!   3. (host inputs) q_rot + kcache + vcache          => attn_o
//!   4. (host inputs) attn_o + x                       => O proj + Add => x_attn
//!   5. (host inputs) x_attn                           => RmsNorm(post)
//!                   + gate/up proj + SiluMul          => silu_out
//!   6. (host inputs) silu_out + x_attn                => Down proj + Add => x_out
//!
//! Each sub-graph reuses the same rpcmem-backed weights/inputs as the 14-node
//! chain. Outputs are read back through APP_READ host buffers immediately
//! after `graphExecute` and compared to the matching stage of the raw-OpenCL
//! reference chain. The first sub-graph whose `max_abs_err >= 1e-2`
//! identifies the composition fault site.
//!
//! ## Pass-gate
//! - All 6 sub-graphs `max_abs_err < 1e-2` -> GREEN (no drift; fault must lie
//!   beyond OpPackage scope).
//! - First sub-graph >= 1e-2 -> root-cause locus printed.
//!
//! ## Build / deploy
//!
//! ```text
//! cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!   --bin microbench_qnn_qwen_layer_bisect2
//! cargo build --release -p qnn_oppkg --target aarch64-linux-android
//! adb push target/aarch64-linux-android/release/libqnn_oppkg.so /data/local/tmp/
//! python scripts/run_device.py -d galaxy_s25 microbench_qnn_qwen_layer_bisect2
//! ```

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_qwen_layer_bisect2 requires --features qnn");
    std::process::exit(2);
}

#[cfg(feature = "qnn")]
#[allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code
)]
mod qnn {
    include!(concat!(env!("OUT_DIR"), "/qnn_bindings.rs"));
}

#[cfg(feature = "qnn")]
const RMS_EPS: f32 = 1e-5;

#[cfg(feature = "qnn")]
const QK4_0: usize = 32;
#[cfg(feature = "qnn")]
const QS_PER_BLOCK: usize = 16;

#[cfg(feature = "qnn")]
fn main() -> anyhow::Result<()> {
    use libloading::{Library, Symbol};
    use std::ffi::{CString, c_void};
    use std::os::raw::c_uint;
    use std::ptr;

    use qnn::*;

    const PKG_PATH: &str = "/data/local/tmp/libqnn_oppkg.so";
    const PKG_PROVIDER: &str = "QnnOpPackage_InitInterface";
    const PKG_TARGET: &str = "GPU_QTI_AISW";

    // Qwen 2.5-1.5B reference — mirrors `qnn_oppkg::graph::LayerConfig::qwen2p5_1p5b()`.
    // Inlined here to keep the engine crate dependency-free of qnn_oppkg
    // (INV-160: production code unchanged).
    let dim: usize = 1536;
    let n_head: usize = 12;
    let n_kv_heads: usize = 2;
    let head_dim: usize = 128;
    let ffn_dim: usize = 8960;
    let kv_capacity: usize = 2048;
    let q_proj_out: usize = n_head * head_dim;
    let kv_proj_out: usize = n_kv_heads * head_dim;
    let pos: i32 = 0; // KV cache initially empty; this layer writes pos=0.
    let n_kv: usize = 1; // attention sees the freshly written token only.
    let theta: f32 = 1_000_000.0; // Qwen2.5 RoPE theta.

    println!("=== microbench_qnn_qwen_layer_bisect2 (M2.H 8th, 6 sub-graphs) ===\n");
    println!("Op Package:       {}", PKG_PATH);
    println!("Interface symbol: {}", PKG_PROVIDER);
    println!(
        "Layer dims: dim={}, n_head={}, n_kv_heads={}, head_dim={}, ffn_dim={}, kv_capacity={}",
        dim, n_head, n_kv_heads, head_dim, ffn_dim, kv_capacity
    );
    println!(
        "6 sub-graphs (each with host roundtrip):\n  \
         1: RmsNorm(pre) + Q proj + RoPE Q              => q_rot\n  \
         2: RmsNorm(pre) + K proj + V proj + RoPE K + KvScatter => kcache, vcache\n  \
         3: q_rot + kcache + vcache => FlashAttn        => attn_o\n  \
         4: attn_o + x => O proj + Add(residual1)       => x_attn\n  \
         5: x_attn => RmsNorm(post) + gate/up + SiluMul => silu_out\n  \
         6: silu_out + x_attn => Down + Add(residual2)  => x_out\n"
    );

    let backend_lib_path = std::env::var("QNN_BACKEND_LIB")
        .unwrap_or_else(|_| "/data/local/tmp/qnn/libQnnGpu.so".to_string());
    println!("Backend lib: {}\n", backend_lib_path);

    // ── Build raw-OpenCL reference toolchain (14 separate kernel objects) ────
    use ocl::{Context, Device, Platform, Program, Queue};

    let platform = Platform::default();
    let device = Device::first(platform)?;
    let cl_ctx = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let cl_q = Queue::new(&cl_ctx, device, None)?;

    let simple_src = include_str!("../../kernels/simple_ops.cl");
    let add_src = include_str!("../../kernels/add.cl");
    let q40_src = include_str!("../../kernels/mul_mv_q4_0_f32_8x_flat.cl");
    let fa_src = include_str!("../../kernels/flash_attn_f32_f16.cl");
    let cl_opts = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";
    let fa_opts = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math -DDK=128 -DDV=128 \
                   -DBLOCK_M=32 -DBLOCK_N=32";

    let prog_simple = Program::builder()
        .devices(device)
        .src(simple_src)
        .cmplr_opt(cl_opts)
        .build(&cl_ctx)?;
    let prog_add = Program::builder()
        .devices(device)
        .src(add_src)
        .cmplr_opt(cl_opts)
        .build(&cl_ctx)?;
    let prog_q40 = Program::builder()
        .devices(device)
        .src(q40_src)
        .cmplr_opt(cl_opts)
        .build(&cl_ctx)?;
    let prog_fa = Program::builder()
        .devices(device)
        .src(fa_src)
        .cmplr_opt(fa_opts)
        .build(&cl_ctx)?;

    let k_rms = ocl::core::create_kernel(&prog_simple, "kernel_rms_norm_simple")?;
    let k_q40 = ocl::core::create_kernel(&prog_q40, "kernel_mul_mat_q4_0_f32_8x_flat")?;
    let k_rope = ocl::core::create_kernel(&prog_simple, "kernel_rope_simple")?;
    let k_kvs = ocl::core::create_kernel(&prog_simple, "kernel_kv_scatter_f32_to_f16")?;
    let k_fa = ocl::core::create_kernel(&prog_fa, "flash_attn_f32_f16_q1")?;
    let k_silu = ocl::core::create_kernel(&prog_simple, "kernel_silu_mul_simple")?;
    let k_add = ocl::core::create_kernel(&prog_add, "kernel_add_row")?;

    // ── QNN backend + register OpPackage ─────────────────────────────────────
    let gpu_lib = unsafe { Library::new(&backend_lib_path) }?;
    type GetProvidersFn = unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let gp: Symbol<GetProvidersFn> = unsafe { gpu_lib.get(b"QnnInterface_getProviders\0")? };
    let mut provs: *mut *const QnnInterface_t = ptr::null_mut();
    let mut np: c_uint = 0;
    let err = unsafe { gp(&mut provs, &mut np) };
    anyhow::ensure!(err == 0 && np > 0, "GPU getProviders err=0x{:x}", err);
    let v = unsafe { (**provs).__bindgen_anon_1.v2_25 };

    // VERBOSE logger — NULL callback = QNN's default platform logger (Android logcat).
    // Inspect with: adb logcat -v threadtime QnnGpu:V QnnGpuOpPackage:V QnnGraph:V QnnDevice:V \*:S
    let mut logger: Qnn_LogHandle_t = ptr::null_mut();
    if let Some(log_create) = v.logCreate {
        let err = unsafe { log_create(None, QnnLog_Level_t_QNN_LOG_LEVEL_VERBOSE, &mut logger) };
        if err != 0 {
            eprintln!("logCreate err=0x{:x} (proceeding without logger)", err);
            logger = ptr::null_mut();
        } else {
            eprintln!("logCreate VERBOSE: OK");
        }
    }

    let mut be: Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe { (v.backendCreate.unwrap())(logger, ptr::null_mut(), &mut be) };
    anyhow::ensure!(err == 0, "backendCreate err=0x{:x}", err);
    println!("backend: OK");

    let pkg_path = CString::new(PKG_PATH).unwrap();
    let pkg_provider = CString::new(PKG_PROVIDER).unwrap();
    let pkg_target = CString::new(PKG_TARGET).unwrap();
    let reg_fn = v
        .backendRegisterOpPackage
        .ok_or_else(|| anyhow::anyhow!("backendRegisterOpPackage is NULL"))?;
    let err = unsafe {
        reg_fn(
            be,
            pkg_path.as_ptr(),
            pkg_provider.as_ptr(),
            pkg_target.as_ptr(),
        )
    };
    println!("registerOpPackage -> err=0x{:x}", err);
    anyhow::ensure!(err == 0, "registerOpPackage err=0x{:x}", err);

    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe { (v.contextCreate.unwrap())(be, ptr::null_mut(), ptr::null_mut(), &mut ctx) };
    anyhow::ensure!(err == 0, "contextCreate err=0x{:x}", err);

    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;
    type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(po: *const c_void) -> i32;
    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }?;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let _rpcmem_free: Symbol<RpcmemFreeFn> = unsafe { rpc_lib.get(b"rpcmem_free\0")? };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };

    let result = run_layer(
        &v,
        ctx,
        &cl_q,
        &cl_ctx,
        &k_rms,
        &k_q40,
        &k_rope,
        &k_kvs,
        &k_fa,
        &k_silu,
        &k_add,
        &rpcmem_alloc,
        &rpcmem_to_fd,
        RPCMEM_HEAP_ID_SYSTEM,
        RPCMEM_DEFAULT_FLAGS,
        dim,
        n_head,
        n_kv_heads,
        head_dim,
        ffn_dim,
        kv_capacity,
        q_proj_out,
        kv_proj_out,
        pos,
        n_kv,
        theta,
    );

    let pass = match result {
        Ok(metrics) => {
            let acc_thresh = 1e-2_f32;
            println!("\n=== Per-sub-graph max_abs_err vs raw-OpenCL reference ===");
            println!(
                "{:>2}  {:<24}  {:>14}  {}",
                "#", "endpoint", "max_abs_err", "verdict"
            );
            println!("{}", "-".repeat(64));
            let mut first_red: Option<usize> = None;
            for (i, (name, err)) in metrics.stages.iter().enumerate() {
                let pass_i = *err < acc_thresh;
                if !pass_i && first_red.is_none() {
                    first_red = Some(i);
                }
                println!(
                    "{:>2}  {:<24}  {:>14.6e}  {}",
                    i + 1,
                    name,
                    err,
                    if pass_i { "PASS" } else { "FAIL" }
                );
            }
            if let Some(idx) = first_red {
                let (name, err) = &metrics.stages[idx];
                println!(
                    "\n>>> First divergent sub-graph: #{} {} (max_abs_err={:.6e})",
                    idx + 1,
                    name,
                    err
                );
            } else {
                println!("\n>>> All 6 sub-graphs within {:.0e} threshold", acc_thresh);
            }
            first_red.is_none()
        }
        Err(e) => {
            println!("\nERROR: {}", e);
            false
        }
    };

    println!(
        "\n=== M2.H bisect2 verdict: {} ===",
        if pass {
            "GREEN (no sub-graph drift)"
        } else {
            "DRIFT IDENTIFIED"
        }
    );

    unsafe {
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(be);
    }

    if pass { Ok(()) } else { std::process::exit(1) }
}

/// Pack `m * k` row-major F32 weights into Q4_0 SOA form (`q_bytes`,
/// `d_halves`). Layout matches `kernel_mul_mat_q4_0_f32_8x_flat` expectations:
/// `q_bytes` stores 4-bit packed quants in 16 nibbles per 32-element block,
/// and `d_halves` stores per-block FP16 scales.
#[cfg(feature = "qnn")]
fn pack_q40_soa(weights: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u16>) {
    assert!(k.is_multiple_of(QK4_0), "K must be a multiple of 32");
    let num_blocks = n * k / QK4_0;
    let mut q = vec![0u8; num_blocks * QS_PER_BLOCK];
    let mut d = vec![0u16; num_blocks];

    for row in 0..n {
        for blk in 0..(k / QK4_0) {
            let base = row * k + blk * QK4_0;
            let block = &weights[base..base + QK4_0];

            // Q4_0 scale: max(abs(x)) / -8.
            let mut amax = 0.0f32;
            let mut max_signed = 0.0f32;
            for &x in block {
                if x.abs() > amax {
                    amax = x.abs();
                    max_signed = x;
                }
            }
            let scale = if amax > 0.0 { max_signed / -8.0 } else { 0.0 };
            let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };

            let blk_idx = row * (k / QK4_0) + blk;
            d[blk_idx] = f32_to_f16_bits(scale);

            // Quantise: q = clamp(round(x * inv) + 8, 0, 15).
            // Pack nibble pair (q[i + 16] << 4) | q[i] for i ∈ [0, 16).
            let mut quants = [0u8; QK4_0];
            for (i, &x) in block.iter().enumerate() {
                let raw = ((x * inv).round() as i32) + 8;
                quants[i] = raw.clamp(0, 15) as u8;
            }
            let q_off = blk_idx * QS_PER_BLOCK;
            for i in 0..QS_PER_BLOCK {
                q[q_off + i] = (quants[i + QS_PER_BLOCK] << 4) | (quants[i] & 0x0F);
            }
        }
    }

    (q, d)
}

#[cfg(feature = "qnn")]
struct LayerMetrics {
    /// Per-sub-graph (endpoint name, max_abs_err) ordered 1..=6.
    stages: Vec<(String, f32)>,
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "qnn")]
fn run_layer(
    v: &qnn::QnnInterface_ImplementationV2_25_t,
    ctx: qnn::Qnn_ContextHandle_t,
    cl_q: &ocl::Queue,
    cl_ctx: &ocl::Context,
    k_rms: &ocl::core::Kernel,
    k_q40: &ocl::core::Kernel,
    k_rope: &ocl::core::Kernel,
    k_kvs: &ocl::core::Kernel,
    k_fa: &ocl::core::Kernel,
    k_silu: &ocl::core::Kernel,
    k_add: &ocl::core::Kernel,
    rpcmem_alloc: &libloading::Symbol<unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void>,
    rpcmem_to_fd: &libloading::Symbol<unsafe extern "C" fn(*const std::ffi::c_void) -> i32>,
    heap_id: i32,
    flags: u32,
    dim: usize,
    n_head: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    kv_capacity: usize,
    q_proj_out: usize,
    kv_proj_out: usize,
    pos: i32,
    n_kv: usize,
    theta: f32,
) -> anyhow::Result<LayerMetrics> {
    use ocl::core::ArgVal;
    use qnn::*;
    use std::ffi::CString;
    use std::os::raw::c_char;
    use std::ptr;

    // ── 1. Generate identical inputs/weights for both paths ─────────────────
    // Random but deterministic. Q4_0 weights: pack on host so both paths
    // consume bit-identical SOA buffers.
    let mut host_x = vec![0.0f32; dim];
    for (i, x) in host_x.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5;
    }
    let mut host_rms_w_pre = vec![0.0f32; dim];
    let mut host_rms_w_post = vec![0.0f32; dim];
    for (i, w) in host_rms_w_pre.iter_mut().enumerate() {
        *w = ((i as f32) * 0.0091 + 0.13).rem_euclid(1.0) * 0.4 + 0.8;
    }
    for (i, w) in host_rms_w_post.iter_mut().enumerate() {
        *w = ((i as f32) * 0.0107 + 0.19).rem_euclid(1.0) * 0.4 + 0.8;
    }
    // Q4_0 weights: small magnitude so quantisation error stays bounded.
    fn gen_weights(n: usize, k: usize, seed: f32) -> Vec<f32> {
        let mut w = vec![0.0f32; n * k];
        for (i, x) in w.iter_mut().enumerate() {
            *x = (((i as f32) * 0.000_31 + seed).rem_euclid(1.0) - 0.5) * 0.1;
        }
        w
    }
    let w_q = gen_weights(q_proj_out, dim, 0.13);
    let w_k = gen_weights(kv_proj_out, dim, 0.21);
    let w_v = gen_weights(kv_proj_out, dim, 0.29);
    let w_o = gen_weights(dim, q_proj_out, 0.37);
    let w_gate = gen_weights(ffn_dim, dim, 0.43);
    let w_up = gen_weights(ffn_dim, dim, 0.51);
    let w_down = gen_weights(dim, ffn_dim, 0.59);

    // Pack to Q4_0 SOA on host (so raw OpenCL and OpPackage paths consume
    // identical byte buffers — quantisation error becomes a constant offset
    // shared by both paths).
    let (qq_q, qq_d) = pack_q40_soa(&w_q, q_proj_out, dim);
    let (qk_q, qk_d) = pack_q40_soa(&w_k, kv_proj_out, dim);
    let (qv_q, qv_d) = pack_q40_soa(&w_v, kv_proj_out, dim);
    let (qo_q, qo_d) = pack_q40_soa(&w_o, dim, q_proj_out);
    let (qg_q, qg_d) = pack_q40_soa(&w_gate, ffn_dim, dim);
    let (qu_q, qu_d) = pack_q40_soa(&w_up, ffn_dim, dim);
    let (qd_q, qd_d) = pack_q40_soa(&w_down, dim, ffn_dim);

    // ── 2. Path A: raw OpenCL — 14-stage chain ──────────────────────────────
    // Buffers: x, rms_w_pre, q4_0 (q+d) per weight, intermediates, kv cache.
    let kv_total = n_kv_heads * kv_capacity * head_dim;
    let mk_buf_f32_ro = |n| unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, n, None)
    };
    let mk_buf_f32_rw = |n| unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_WRITE, n, None)
    };
    let mk_buf_u8_ro = |n| unsafe {
        ocl::core::create_buffer::<_, u8>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, n, None)
    };
    let mk_buf_u16_ro = |n| unsafe {
        ocl::core::create_buffer::<_, u16>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, n, None)
    };
    let mk_buf_u16_rw = |n| unsafe {
        ocl::core::create_buffer::<_, u16>(cl_ctx.as_core(), ocl::core::MEM_READ_WRITE, n, None)
    };

    let buf_x = mk_buf_f32_ro(dim)?;
    let buf_rms_w_pre = mk_buf_f32_ro(dim)?;
    let buf_rms_w_post = mk_buf_f32_ro(dim)?;
    let buf_y1 = mk_buf_f32_rw(dim)?;
    let buf_qq = mk_buf_u8_ro(qq_q.len())?;
    let buf_qd = mk_buf_u16_ro(qq_d.len())?;
    let buf_kq = mk_buf_u8_ro(qk_q.len())?;
    let buf_kd = mk_buf_u16_ro(qk_d.len())?;
    let buf_vq = mk_buf_u8_ro(qv_q.len())?;
    let buf_vd = mk_buf_u16_ro(qv_d.len())?;
    let buf_oq = mk_buf_u8_ro(qo_q.len())?;
    let buf_od = mk_buf_u16_ro(qo_d.len())?;
    let buf_gq = mk_buf_u8_ro(qg_q.len())?;
    let buf_gd = mk_buf_u16_ro(qg_d.len())?;
    let buf_uq = mk_buf_u8_ro(qu_q.len())?;
    let buf_ud = mk_buf_u16_ro(qu_d.len())?;
    let buf_dq = mk_buf_u8_ro(qd_q.len())?;
    let buf_dd = mk_buf_u16_ro(qd_d.len())?;
    let buf_q = mk_buf_f32_rw(q_proj_out)?;
    let buf_k = mk_buf_f32_rw(kv_proj_out)?;
    let buf_v = mk_buf_f32_rw(kv_proj_out)?;
    let buf_kcache = mk_buf_u16_rw(kv_total)?;
    let buf_vcache = mk_buf_u16_rw(kv_total)?;
    let buf_attn_o = mk_buf_f32_rw(q_proj_out)?;
    let buf_o = mk_buf_f32_rw(dim)?;
    let buf_x_attn = mk_buf_f32_rw(dim)?;
    let buf_y2 = mk_buf_f32_rw(dim)?;
    let buf_gate = mk_buf_f32_rw(ffn_dim)?;
    let buf_up = mk_buf_f32_rw(ffn_dim)?;
    let buf_down = mk_buf_f32_rw(dim)?;
    let buf_x_out = mk_buf_f32_rw(dim)?;

    // Upload all host-side buffers (zero-init kv cache).
    let host_kv_zero = vec![0u16; kv_total];
    macro_rules! up_f32 {
        ($buf:expr, $host:expr) => {
            unsafe {
                ocl::core::enqueue_write_buffer(
                    cl_q,
                    $buf,
                    true,
                    0,
                    $host,
                    None::<ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        };
    }
    macro_rules! up_u8 {
        ($buf:expr, $host:expr) => {
            unsafe {
                ocl::core::enqueue_write_buffer(
                    cl_q,
                    $buf,
                    true,
                    0,
                    $host,
                    None::<ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        };
    }
    macro_rules! up_u16 {
        ($buf:expr, $host:expr) => {
            unsafe {
                ocl::core::enqueue_write_buffer(
                    cl_q,
                    $buf,
                    true,
                    0,
                    $host,
                    None::<ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        };
    }
    up_f32!(&buf_x, &host_x);
    up_f32!(&buf_rms_w_pre, &host_rms_w_pre);
    up_f32!(&buf_rms_w_post, &host_rms_w_post);
    up_u8!(&buf_qq, &qq_q);
    up_u16!(&buf_qd, &qq_d);
    up_u8!(&buf_kq, &qk_q);
    up_u16!(&buf_kd, &qk_d);
    up_u8!(&buf_vq, &qv_q);
    up_u16!(&buf_vd, &qv_d);
    up_u8!(&buf_oq, &qo_q);
    up_u16!(&buf_od, &qo_d);
    up_u8!(&buf_gq, &qg_q);
    up_u16!(&buf_gd, &qg_d);
    up_u8!(&buf_uq, &qu_q);
    up_u16!(&buf_ud, &qu_d);
    up_u8!(&buf_dq, &qd_q);
    up_u16!(&buf_dd, &qd_d);
    up_u16!(&buf_kcache, &host_kv_zero);
    up_u16!(&buf_vcache, &host_kv_zero);
    ocl::core::finish(cl_q)?;

    // Stage 1: RmsNorm(pre) — kernel_rms_norm_simple(x, w, y1, dim, eps)
    let dim_i: i32 = dim as i32;
    let ffn_dim_i: i32 = ffn_dim as i32;
    ocl::core::set_kernel_arg(k_rms, 0, ArgVal::mem(&buf_x))?;
    ocl::core::set_kernel_arg(k_rms, 1, ArgVal::mem(&buf_rms_w_pre))?;
    ocl::core::set_kernel_arg(k_rms, 2, ArgVal::mem(&buf_y1))?;
    ocl::core::set_kernel_arg(k_rms, 3, ArgVal::scalar(&dim_i))?;
    ocl::core::set_kernel_arg(k_rms, 4, ArgVal::scalar(&RMS_EPS))?;
    let one3 = [1usize, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_rms,
            1,
            None,
            &one3,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Helper for Q4_0 matmul stage. Mirrors microbench_qnn_oppkg_matmul_q40.
    let dispatch_q40 = |k_q40: &ocl::core::Kernel,
                        bq: &ocl::core::Mem,
                        bd: &ocl::core::Mem,
                        bx: &ocl::core::Mem,
                        by: &ocl::core::Mem,
                        m: usize,
                        n: usize,
                        k: usize|
     -> anyhow::Result<()> {
        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02: i32 = 1;
        let ne10 = k as i32;
        let ne12: i32 = 1;
        let ne0 = n as i32;
        let ne1 = m as i32;
        let r2: i32 = 1;
        let r3: i32 = 1;
        let off1: u64 = 0;
        let offd: u64 = 0;
        ocl::core::set_kernel_arg(k_q40, 0, ArgVal::mem(bq))?;
        ocl::core::set_kernel_arg(k_q40, 1, ArgVal::mem(bd))?;
        ocl::core::set_kernel_arg(k_q40, 2, ArgVal::mem(bx))?;
        ocl::core::set_kernel_arg(k_q40, 3, ArgVal::scalar(&off1))?;
        ocl::core::set_kernel_arg(k_q40, 4, ArgVal::mem(by))?;
        ocl::core::set_kernel_arg(k_q40, 5, ArgVal::scalar(&offd))?;
        ocl::core::set_kernel_arg(k_q40, 6, ArgVal::scalar(&ne00))?;
        ocl::core::set_kernel_arg(k_q40, 7, ArgVal::scalar(&ne01))?;
        ocl::core::set_kernel_arg(k_q40, 8, ArgVal::scalar(&ne02))?;
        ocl::core::set_kernel_arg(k_q40, 9, ArgVal::scalar(&ne10))?;
        ocl::core::set_kernel_arg(k_q40, 10, ArgVal::scalar(&ne12))?;
        ocl::core::set_kernel_arg(k_q40, 11, ArgVal::scalar(&ne0))?;
        ocl::core::set_kernel_arg(k_q40, 12, ArgVal::scalar(&ne1))?;
        ocl::core::set_kernel_arg(k_q40, 13, ArgVal::scalar(&r2))?;
        ocl::core::set_kernel_arg(k_q40, 14, ArgVal::scalar(&r3))?;
        let global = [n.div_ceil(8) * 64, 1, 1];
        let local = [64usize, 1, 1];
        unsafe {
            ocl::core::enqueue_kernel(
                cl_q,
                k_q40,
                3,
                None,
                &global,
                Some(local),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        ocl::core::finish(cl_q)?;
        Ok(())
    };

    // Stages 2-4: Q/K/V projection (M=1).
    dispatch_q40(k_q40, &buf_qq, &buf_qd, &buf_y1, &buf_q, 1, q_proj_out, dim)?;
    dispatch_q40(
        k_q40,
        &buf_kq,
        &buf_kd,
        &buf_y1,
        &buf_k,
        1,
        kv_proj_out,
        dim,
    )?;
    dispatch_q40(
        k_q40,
        &buf_vq,
        &buf_vd,
        &buf_y1,
        &buf_v,
        1,
        kv_proj_out,
        dim,
    )?;

    // Stage 5: RoPE(Q) — kernel_rope_simple(x, head_dim, num_heads, seq_len, start_pos, theta)
    let head_dim_i: i32 = head_dim as i32;
    let n_head_i: i32 = n_head as i32;
    let n_kv_heads_i: i32 = n_kv_heads as i32;
    let seq_len_i: i32 = 1;
    ocl::core::set_kernel_arg(k_rope, 0, ArgVal::mem(&buf_q))?;
    ocl::core::set_kernel_arg(k_rope, 1, ArgVal::scalar(&head_dim_i))?;
    ocl::core::set_kernel_arg(k_rope, 2, ArgVal::scalar(&n_head_i))?;
    ocl::core::set_kernel_arg(k_rope, 3, ArgVal::scalar(&seq_len_i))?;
    ocl::core::set_kernel_arg(k_rope, 4, ArgVal::scalar(&pos))?;
    ocl::core::set_kernel_arg(k_rope, 5, ArgVal::scalar(&theta))?;
    let pairs_q = n_head * (head_dim / 2);
    let global_rope_q = [pairs_q, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_rope,
            1,
            None,
            &global_rope_q,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stage 6: RoPE(K)
    ocl::core::set_kernel_arg(k_rope, 0, ArgVal::mem(&buf_k))?;
    ocl::core::set_kernel_arg(k_rope, 1, ArgVal::scalar(&head_dim_i))?;
    ocl::core::set_kernel_arg(k_rope, 2, ArgVal::scalar(&n_kv_heads_i))?;
    ocl::core::set_kernel_arg(k_rope, 3, ArgVal::scalar(&seq_len_i))?;
    ocl::core::set_kernel_arg(k_rope, 4, ArgVal::scalar(&pos))?;
    ocl::core::set_kernel_arg(k_rope, 5, ArgVal::scalar(&theta))?;
    let pairs_k = n_kv_heads * (head_dim / 2);
    let global_rope_k = [pairs_k, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_rope,
            1,
            None,
            &global_rope_k,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stage 7: KvScatter — kernel_kv_scatter_f32_to_f16(k_src, v_src, k_dst, v_dst, head_dim, capacity, write_pos)
    let capacity_i: i32 = kv_capacity as i32;
    ocl::core::set_kernel_arg(k_kvs, 0, ArgVal::mem(&buf_k))?;
    ocl::core::set_kernel_arg(k_kvs, 1, ArgVal::mem(&buf_v))?;
    ocl::core::set_kernel_arg(k_kvs, 2, ArgVal::mem(&buf_kcache))?;
    ocl::core::set_kernel_arg(k_kvs, 3, ArgVal::mem(&buf_vcache))?;
    ocl::core::set_kernel_arg(k_kvs, 4, ArgVal::scalar(&head_dim_i))?;
    ocl::core::set_kernel_arg(k_kvs, 5, ArgVal::scalar(&capacity_i))?;
    ocl::core::set_kernel_arg(k_kvs, 6, ArgVal::scalar(&pos))?;
    let global_kvs = [kv_proj_out, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_kvs,
            1,
            None,
            &global_kvs,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stage 8: FlashAttn — flash_attn_f32_f16_q1 (decode, n_kv=1)
    let buf_score_dummy = mk_buf_f32_rw(1)?;
    let q_nb1 = (n_head * head_dim * 4) as u64;
    let q_nb2 = (head_dim * 4) as u64;
    let q_nb3 = q_nb1;
    let k_nb1 = (head_dim * 2) as u64;
    let k_nb2 = (kv_capacity * head_dim * 2) as u64;
    let k_nb3 = (n_kv_heads as u64) * k_nb2;
    let o_nb1 = (head_dim * 4) as u64;
    let o_nb2 = (n_head * head_dim * 4) as u64;
    let o_nb3 = o_nb2;
    let zero_u64: u64 = 0;
    let zero_i32: i32 = 0;
    let n_kv_i: i32 = n_kv as i32;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let n_q: i32 = 1;
    let max_bias: f32 = 0.0;
    let m0: f32 = 0.0;
    let m1: f32 = 0.0;
    let n_head_log2: i32 = 0;
    let logit_softcap: f32 = 0.0;
    ocl::core::set_kernel_arg(k_fa, 0, ArgVal::mem(&buf_q))?;
    ocl::core::set_kernel_arg(k_fa, 1, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 2, ArgVal::mem(&buf_kcache))?;
    ocl::core::set_kernel_arg(k_fa, 3, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 4, ArgVal::mem(&buf_vcache))?;
    ocl::core::set_kernel_arg(k_fa, 5, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 6, ArgVal::mem(&buf_attn_o))?;
    ocl::core::set_kernel_arg(k_fa, 7, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 8, ArgVal::scalar(&scale))?;
    ocl::core::set_kernel_arg(k_fa, 9, ArgVal::scalar(&n_q))?;
    ocl::core::set_kernel_arg(k_fa, 10, ArgVal::scalar(&n_kv_i))?;
    ocl::core::set_kernel_arg(k_fa, 11, ArgVal::scalar(&zero_i32))?;
    ocl::core::set_kernel_arg(k_fa, 12, ArgVal::scalar(&n_head_i))?;
    ocl::core::set_kernel_arg(k_fa, 13, ArgVal::scalar(&q_nb1))?;
    ocl::core::set_kernel_arg(k_fa, 14, ArgVal::scalar(&q_nb2))?;
    ocl::core::set_kernel_arg(k_fa, 15, ArgVal::scalar(&q_nb3))?;
    ocl::core::set_kernel_arg(k_fa, 16, ArgVal::scalar(&k_nb1))?;
    ocl::core::set_kernel_arg(k_fa, 17, ArgVal::scalar(&k_nb2))?;
    ocl::core::set_kernel_arg(k_fa, 18, ArgVal::scalar(&k_nb3))?;
    ocl::core::set_kernel_arg(k_fa, 19, ArgVal::scalar(&k_nb1))?;
    ocl::core::set_kernel_arg(k_fa, 20, ArgVal::scalar(&k_nb2))?;
    ocl::core::set_kernel_arg(k_fa, 21, ArgVal::scalar(&k_nb3))?;
    ocl::core::set_kernel_arg(k_fa, 22, ArgVal::scalar(&o_nb1))?;
    ocl::core::set_kernel_arg(k_fa, 23, ArgVal::scalar(&o_nb2))?;
    ocl::core::set_kernel_arg(k_fa, 24, ArgVal::scalar(&o_nb3))?;
    ocl::core::set_kernel_arg(k_fa, 25, ArgVal::scalar(&max_bias))?;
    ocl::core::set_kernel_arg(k_fa, 26, ArgVal::scalar(&m0))?;
    ocl::core::set_kernel_arg(k_fa, 27, ArgVal::scalar(&m1))?;
    ocl::core::set_kernel_arg(k_fa, 28, ArgVal::scalar(&n_head_log2))?;
    ocl::core::set_kernel_arg(k_fa, 29, ArgVal::scalar(&logit_softcap))?;
    ocl::core::set_kernel_arg(k_fa, 30, ArgVal::scalar(&n_kv_heads_i))?;
    ocl::core::set_kernel_arg(k_fa, 31, ArgVal::mem_null())?;
    ocl::core::set_kernel_arg(k_fa, 32, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 33, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 34, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 35, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 36, ArgVal::scalar(&zero_i32))?;
    ocl::core::set_kernel_arg(k_fa, 37, ArgVal::scalar(&zero_i32))?;
    ocl::core::set_kernel_arg(k_fa, 38, ArgVal::mem_null())?;
    ocl::core::set_kernel_arg(k_fa, 39, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 40, ArgVal::mem(&buf_score_dummy))?;
    ocl::core::set_kernel_arg(k_fa, 41, ArgVal::scalar(&zero_i32))?;
    ocl::core::set_kernel_arg(k_fa, 42, ArgVal::scalar(&zero_i32))?;
    ocl::core::set_kernel_arg(k_fa, 43, ArgVal::scalar(&zero_i32))?;
    const Q1_WG: usize = 64;
    let global_fa = [Q1_WG, n_head, 1];
    let local_fa = [Q1_WG, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_fa,
            2,
            None,
            &global_fa,
            Some(local_fa),
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stage 9: O proj — Q4_0 matmul (M=1, N=dim, K=q_proj_out)
    dispatch_q40(
        k_q40,
        &buf_oq,
        &buf_od,
        &buf_attn_o,
        &buf_o,
        1,
        dim,
        q_proj_out,
    )?;

    // Stage 10: Add (residual #1) — kernel_add_row(o, x, x_attn, n4)
    let n4_dim: i32 = (dim / 4) as i32;
    let n4_ffn: i32 = (ffn_dim / 4) as i32;
    let zero_u64_h: u64 = 0;
    ocl::core::set_kernel_arg(k_add, 0, ArgVal::mem(&buf_o))?;
    ocl::core::set_kernel_arg(k_add, 1, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 2, ArgVal::mem(&buf_x))?;
    ocl::core::set_kernel_arg(k_add, 3, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 4, ArgVal::mem(&buf_x_attn))?;
    ocl::core::set_kernel_arg(k_add, 5, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 6, ArgVal::scalar(&n4_dim))?;
    let global_add_dim = [n4_dim as usize, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_add,
            1,
            None,
            &global_add_dim,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stage 11: RmsNorm(post)
    ocl::core::set_kernel_arg(k_rms, 0, ArgVal::mem(&buf_x_attn))?;
    ocl::core::set_kernel_arg(k_rms, 1, ArgVal::mem(&buf_rms_w_post))?;
    ocl::core::set_kernel_arg(k_rms, 2, ArgVal::mem(&buf_y2))?;
    ocl::core::set_kernel_arg(k_rms, 3, ArgVal::scalar(&dim_i))?;
    ocl::core::set_kernel_arg(k_rms, 4, ArgVal::scalar(&RMS_EPS))?;
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_rms,
            1,
            None,
            &one3,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stages 12-13: gate / up — Q4_0 matmul (M=1, N=ffn_dim, K=dim)
    dispatch_q40(k_q40, &buf_gq, &buf_gd, &buf_y2, &buf_gate, 1, ffn_dim, dim)?;
    dispatch_q40(k_q40, &buf_uq, &buf_ud, &buf_y2, &buf_up, 1, ffn_dim, dim)?;

    // Stage 14: SiluMul — kernel_silu_mul_simple(gate, up, ffn_dim/4) [in-place on gate]
    ocl::core::set_kernel_arg(k_silu, 0, ArgVal::mem(&buf_gate))?;
    ocl::core::set_kernel_arg(k_silu, 1, ArgVal::mem(&buf_up))?;
    ocl::core::set_kernel_arg(k_silu, 2, ArgVal::scalar(&n4_ffn))?;
    let global_silu = [n4_ffn as usize, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_silu,
            1,
            None,
            &global_silu,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stage 15: down proj — Q4_0 matmul (M=1, N=dim, K=ffn_dim)
    let _ = ffn_dim_i; // referenced symbol, used implicitly via cfg
    dispatch_q40(
        k_q40, &buf_dq, &buf_dd, &buf_gate, &buf_down, 1, dim, ffn_dim,
    )?;

    // Stage 16: Add (residual #2)
    ocl::core::set_kernel_arg(k_add, 0, ArgVal::mem(&buf_down))?;
    ocl::core::set_kernel_arg(k_add, 1, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 2, ArgVal::mem(&buf_x_attn))?;
    ocl::core::set_kernel_arg(k_add, 3, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 4, ArgVal::mem(&buf_x_out))?;
    ocl::core::set_kernel_arg(k_add, 5, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 6, ArgVal::scalar(&n4_dim))?;
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_add,
            1,
            None,
            &global_add_dim,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Read back the reference layer output + Graph A boundary tensors.
    let mut ref_x_out = vec![0.0f32; dim];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_x_out,
            true,
            0,
            &mut ref_x_out,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    // 2-graph boundary: q_rope (rank-3), k_cache (rank-4 full), v_cache (rank-4 full).
    let mut ref_q_rope = vec![0.0f32; q_proj_out];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_q,
            true,
            0,
            &mut ref_q_rope,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    let mut ref_kcache_full = vec![0u16; kv_total];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_kcache,
            true,
            0,
            &mut ref_kcache_full,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    let mut ref_vcache_full = vec![0u16; kv_total];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_vcache,
            true,
            0,
            &mut ref_vcache_full,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    // Per-stage references for sub-graph 3/4/5: attn_o, x_attn, silu_out.
    // SiluMul in raw OpenCL is in-place on buf_gate (so post-silu gate buffer
    // == silu_out). attn_o is FlashAttn output. x_attn is residual1 add output.
    let mut ref_attn_o = vec![0.0f32; q_proj_out];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_attn_o,
            true,
            0,
            &mut ref_attn_o,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    let mut ref_x_attn = vec![0.0f32; dim];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_x_attn,
            true,
            0,
            &mut ref_x_attn,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    let mut ref_silu_out = vec![0.0f32; ffn_dim];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_gate,
            true,
            0,
            &mut ref_silu_out,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // ── 3. Path B: QNN — 6 sub-graphs with host roundtrip per boundary ──────
    // rpcmem allocations. Graph A and Graph B share weights (Q4 buffers,
    // RMS weights, residual `x`) plus dedicated boundary buffers
    // (`q_rope`, `kcache`, `vcache`) that Graph A writes and Graph B reads.
    let bytes_x = (dim * 4) as i32;
    let bytes_rms = (dim * 4) as i32;
    let bytes_kv = (kv_total * 2) as i32;
    let bytes_q_rope = (q_proj_out * 4) as i32;
    let bytes_x_out = (dim * 4) as i32;
    let bytes_attn_o = (q_proj_out * 4) as i32;
    let bytes_x_attn = (dim * 4) as i32;
    let bytes_silu_out = (ffn_dim * 4) as i32;
    let bytes_q40 = |q: &[u8]| q.len() as i32;
    let bytes_q40d = |d: &[u16]| (d.len() * 2) as i32;
    let bytes_mask = 2_i32;
    let bytes_dummy = 4_i32;
    // FlashAttn sinks: kernel reads `sinks_ptr[head_idx]` for head_idx ∈
    // [0, n_head). The OpPackage path always binds a non-null mem object for
    // sinks (no `mem_null` equivalent), so the buffer must be sized to n_head
    // f32s. Host populates with -1e30 so `exp(sink - m_final) ≈ 0`,
    // matching the raw-OpenCL `mem_null()` baseline. A 1-element buffer
    // (`bytes_dummy`) was the source of the M2.H 0.5 ratio: kernel uses
    // m_i = sinks[head_idx] = 0 (zero-init) instead of -INFINITY, then
    // l_final += exp(0 - m_final) inflates the denominator.
    let bytes_sinks = (n_head * 4) as i32;

    // Graph A inputs/weights (RmsNorm pre, Q/K/V Q4_0 weights).
    let rpc_x = unsafe { rpcmem_alloc(heap_id, flags, bytes_x) };
    let rpc_rms_pre = unsafe { rpcmem_alloc(heap_id, flags, bytes_rms) };
    let rpc_qq = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40(&qq_q)) };
    let rpc_qd = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40d(&qq_d)) };
    let rpc_kq = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40(&qk_q)) };
    let rpc_kd = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40d(&qk_d)) };
    let rpc_vq = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40(&qv_q)) };
    let rpc_vd = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40d(&qv_d)) };
    // Graph A endpoints (host-readable) = Graph B inputs.
    let rpc_q_rope = unsafe { rpcmem_alloc(heap_id, flags, bytes_q_rope) };
    let rpc_kcache = unsafe { rpcmem_alloc(heap_id, flags, bytes_kv) };
    let rpc_vcache = unsafe { rpcmem_alloc(heap_id, flags, bytes_kv) };
    // Graph B-only weights.
    let rpc_rms_post = unsafe { rpcmem_alloc(heap_id, flags, bytes_rms) };
    let rpc_oq = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40(&qo_q)) };
    let rpc_od = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40d(&qo_d)) };
    let rpc_gq = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40(&qg_q)) };
    let rpc_gd = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40d(&qg_d)) };
    let rpc_uq = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40(&qu_q)) };
    let rpc_ud = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40d(&qu_d)) };
    let rpc_dq = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40(&qd_q)) };
    let rpc_dd = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40d(&qd_d)) };
    // Graph B endpoint + FlashAttn auxiliaries.
    let rpc_x_out = unsafe { rpcmem_alloc(heap_id, flags, bytes_x_out) };
    let rpc_mask = unsafe { rpcmem_alloc(heap_id, flags, bytes_mask) };
    let rpc_sinks = unsafe { rpcmem_alloc(heap_id, flags, bytes_sinks) };
    let rpc_score = unsafe { rpcmem_alloc(heap_id, flags, bytes_dummy) };
    // Sub-graph boundary buffers (attn_o, x_attn, silu_out) — each is the
    // produced endpoint of one sub-graph and the consumed input of the next.
    let rpc_attn_o = unsafe { rpcmem_alloc(heap_id, flags, bytes_attn_o) };
    let rpc_x_attn = unsafe { rpcmem_alloc(heap_id, flags, bytes_x_attn) };
    let rpc_silu_out = unsafe { rpcmem_alloc(heap_id, flags, bytes_silu_out) };
    anyhow::ensure!(
        !rpc_x.is_null()
            && !rpc_rms_pre.is_null()
            && !rpc_rms_post.is_null()
            && !rpc_q_rope.is_null()
            && !rpc_kcache.is_null()
            && !rpc_vcache.is_null()
            && !rpc_x_out.is_null()
            && !rpc_qq.is_null()
            && !rpc_qd.is_null()
            && !rpc_kq.is_null()
            && !rpc_kd.is_null()
            && !rpc_vq.is_null()
            && !rpc_vd.is_null()
            && !rpc_oq.is_null()
            && !rpc_od.is_null()
            && !rpc_gq.is_null()
            && !rpc_gd.is_null()
            && !rpc_uq.is_null()
            && !rpc_ud.is_null()
            && !rpc_dq.is_null()
            && !rpc_dd.is_null()
            && !rpc_mask.is_null()
            && !rpc_sinks.is_null()
            && !rpc_score.is_null()
            && !rpc_attn_o.is_null()
            && !rpc_x_attn.is_null()
            && !rpc_silu_out.is_null(),
        "rpcmem_alloc failed"
    );

    // Copy host data into rpcmem.
    unsafe {
        std::ptr::copy_nonoverlapping(
            host_x.as_ptr() as *const u8,
            rpc_x as *mut u8,
            bytes_x as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_rms_w_pre.as_ptr() as *const u8,
            rpc_rms_pre as *mut u8,
            bytes_rms as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_rms_w_post.as_ptr() as *const u8,
            rpc_rms_post as *mut u8,
            bytes_rms as usize,
        );
        // Zero-init Graph A outputs (q_rope) and KV cache before Graph A run.
        std::ptr::write_bytes(rpc_q_rope as *mut u8, 0, bytes_q_rope as usize);
        std::ptr::write_bytes(rpc_kcache as *mut u8, 0, bytes_kv as usize);
        std::ptr::write_bytes(rpc_vcache as *mut u8, 0, bytes_kv as usize);
        std::ptr::write_bytes(rpc_x_out as *mut u8, 0, bytes_x_out as usize);
        std::ptr::copy_nonoverlapping(qq_q.as_ptr(), rpc_qq as *mut u8, qq_q.len());
        std::ptr::copy_nonoverlapping(
            qq_d.as_ptr() as *const u8,
            rpc_qd as *mut u8,
            qq_d.len() * 2,
        );
        std::ptr::copy_nonoverlapping(qk_q.as_ptr(), rpc_kq as *mut u8, qk_q.len());
        std::ptr::copy_nonoverlapping(
            qk_d.as_ptr() as *const u8,
            rpc_kd as *mut u8,
            qk_d.len() * 2,
        );
        std::ptr::copy_nonoverlapping(qv_q.as_ptr(), rpc_vq as *mut u8, qv_q.len());
        std::ptr::copy_nonoverlapping(
            qv_d.as_ptr() as *const u8,
            rpc_vd as *mut u8,
            qv_d.len() * 2,
        );
        std::ptr::copy_nonoverlapping(qo_q.as_ptr(), rpc_oq as *mut u8, qo_q.len());
        std::ptr::copy_nonoverlapping(
            qo_d.as_ptr() as *const u8,
            rpc_od as *mut u8,
            qo_d.len() * 2,
        );
        std::ptr::copy_nonoverlapping(qg_q.as_ptr(), rpc_gq as *mut u8, qg_q.len());
        std::ptr::copy_nonoverlapping(
            qg_d.as_ptr() as *const u8,
            rpc_gd as *mut u8,
            qg_d.len() * 2,
        );
        std::ptr::copy_nonoverlapping(qu_q.as_ptr(), rpc_uq as *mut u8, qu_q.len());
        std::ptr::copy_nonoverlapping(
            qu_d.as_ptr() as *const u8,
            rpc_ud as *mut u8,
            qu_d.len() * 2,
        );
        std::ptr::copy_nonoverlapping(qd_q.as_ptr(), rpc_dq as *mut u8, qd_q.len());
        std::ptr::copy_nonoverlapping(
            qd_d.as_ptr() as *const u8,
            rpc_dd as *mut u8,
            qd_d.len() * 2,
        );
        std::ptr::write_bytes(rpc_mask as *mut u8, 0, bytes_mask as usize);
        // Sinks: -1e30 (≈ -INFINITY) per head — neutralises the kernel's
        // sink-attention path so output matches raw-OpenCL mem_null baseline.
        let neg_huge: f32 = -1.0e30f32;
        for i in 0..n_head {
            std::ptr::write_unaligned((rpc_sinks as *mut u8).add(i * 4) as *mut f32, neg_huge);
        }
        std::ptr::write_bytes(rpc_score as *mut u8, 0, bytes_dummy as usize);
        std::ptr::write_bytes(rpc_attn_o as *mut u8, 0, bytes_attn_o as usize);
        std::ptr::write_bytes(rpc_x_attn as *mut u8, 0, bytes_x_attn as usize);
        std::ptr::write_bytes(rpc_silu_out as *mut u8, 0, bytes_silu_out as usize);
    }

    let fd_x = unsafe { rpcmem_to_fd(rpc_x) };
    let fd_rms_pre = unsafe { rpcmem_to_fd(rpc_rms_pre) };
    let fd_rms_post = unsafe { rpcmem_to_fd(rpc_rms_post) };
    let fd_q_rope = unsafe { rpcmem_to_fd(rpc_q_rope) };
    let fd_kcache = unsafe { rpcmem_to_fd(rpc_kcache) };
    let fd_vcache = unsafe { rpcmem_to_fd(rpc_vcache) };
    let fd_x_out = unsafe { rpcmem_to_fd(rpc_x_out) };
    let fd_qq = unsafe { rpcmem_to_fd(rpc_qq) };
    let fd_qd = unsafe { rpcmem_to_fd(rpc_qd) };
    let fd_kq = unsafe { rpcmem_to_fd(rpc_kq) };
    let fd_kd = unsafe { rpcmem_to_fd(rpc_kd) };
    let fd_vq = unsafe { rpcmem_to_fd(rpc_vq) };
    let fd_vd = unsafe { rpcmem_to_fd(rpc_vd) };
    let fd_oq = unsafe { rpcmem_to_fd(rpc_oq) };
    let fd_od = unsafe { rpcmem_to_fd(rpc_od) };
    let fd_gq = unsafe { rpcmem_to_fd(rpc_gq) };
    let fd_gd = unsafe { rpcmem_to_fd(rpc_gd) };
    let fd_uq = unsafe { rpcmem_to_fd(rpc_uq) };
    let fd_ud = unsafe { rpcmem_to_fd(rpc_ud) };
    let fd_dq = unsafe { rpcmem_to_fd(rpc_dq) };
    let fd_dd = unsafe { rpcmem_to_fd(rpc_dd) };
    let fd_mask = unsafe { rpcmem_to_fd(rpc_mask) };
    let fd_sinks = unsafe { rpcmem_to_fd(rpc_sinks) };
    let fd_score = unsafe { rpcmem_to_fd(rpc_score) };
    let fd_attn_o = unsafe { rpcmem_to_fd(rpc_attn_o) };
    let fd_x_attn = unsafe { rpcmem_to_fd(rpc_x_attn) };
    let fd_silu_out = unsafe { rpcmem_to_fd(rpc_silu_out) };

    // ── 4. Shared tensor metadata (CStrings + dims; must outlive graphs) ────
    let qp = Qnn_QuantizeParams_t {
        encodingDefinition: Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
        quantizationEncoding: Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
        __bindgen_anon_1: Qnn_QuantizeParams_t__bindgen_ty_1 {
            scaleOffsetEncoding: Qnn_ScaleOffset_t {
                scale: 0.0,
                offset: 0,
            },
        },
    };
    let mk_tv1 = |ttype, dtype, rank: u32, dims_ptr: *mut u32| Qnn_TensorV1_t {
        id: 0,
        name: ptr::null(),
        type_: ttype,
        dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        dataType: dtype,
        quantizeParams: qp,
        rank,
        dimensions: dims_ptr,
        memType: Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
        __bindgen_anon_1: Qnn_TensorV1_t__bindgen_ty_1 {
            clientBuf: Qnn_ClientBuffer_t {
                data: ptr::null_mut(),
                dataSize: 0,
            },
        },
    };
    let build_tensor = |ttype, dtype, rank: u32, dims_ptr: *mut u32, name: *const c_char| {
        let mut t = Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_tv1(ttype, dtype, rank, dims_ptr),
            },
        };
        t.__bindgen_anon_1.v1.name = name;
        t
    };

    let app_w = Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE;
    let app_r = Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ;
    let native = Qnn_TensorType_t_QNN_TENSOR_TYPE_NATIVE;
    let f32_t = Qnn_DataType_t_QNN_DATATYPE_FLOAT_32;
    let f16_t = Qnn_DataType_t_QNN_DATATYPE_FLOAT_16;
    let u8_t = Qnn_DataType_t_QNN_DATATYPE_UINT_8;

    // Dimensions (mut Vec — pointers stored in Qnn_TensorV1_t.dimensions).
    let mut dims_x: Vec<u32> = vec![1, dim as u32];
    let mut dims_rms_pre = vec![dim as u32];
    let mut dims_rms_post = vec![dim as u32];
    let mut dims_y1 = vec![1u32, dim as u32];
    let mut dims_qq = vec![qq_q.len() as u32];
    let mut dims_qd = vec![qq_d.len() as u32];
    let mut dims_kq = vec![qk_q.len() as u32];
    let mut dims_kd = vec![qk_d.len() as u32];
    let mut dims_vq = vec![qv_q.len() as u32];
    let mut dims_vd = vec![qv_d.len() as u32];
    let mut dims_oq = vec![qo_q.len() as u32];
    let mut dims_od = vec![qo_d.len() as u32];
    let mut dims_gq = vec![qg_q.len() as u32];
    let mut dims_gd = vec![qg_d.len() as u32];
    let mut dims_uq = vec![qu_q.len() as u32];
    let mut dims_ud = vec![qu_d.len() as u32];
    let mut dims_dq = vec![qd_q.len() as u32];
    let mut dims_dd = vec![qd_d.len() as u32];
    let mut dims_q = vec![1u32, n_head as u32, head_dim as u32];
    let mut dims_kvec = vec![1u32, n_kv_heads as u32, head_dim as u32];
    let mut dims_vvec = vec![1u32, n_kv_heads as u32, head_dim as u32];
    let mut dims_q_rope = vec![1u32, n_head as u32, head_dim as u32];
    let mut dims_k_rope = vec![1u32, n_kv_heads as u32, head_dim as u32];
    let mut dims_kcache = vec![1u32, n_kv_heads as u32, kv_capacity as u32, head_dim as u32];
    let mut dims_vcache = dims_kcache.clone();
    let mut dims_attn_o = vec![1u32, n_head as u32, head_dim as u32];
    let mut dims_o = vec![1u32, dim as u32];
    let mut dims_x_attn = vec![1u32, dim as u32];
    let mut dims_y2 = vec![1u32, dim as u32];
    let mut dims_gate = vec![1u32, ffn_dim as u32];
    let mut dims_up = vec![1u32, ffn_dim as u32];
    let mut dims_silu_out = vec![1u32, ffn_dim as u32];
    let mut dims_down = vec![1u32, dim as u32];
    let mut dims_x_out = vec![1u32, dim as u32];
    let mut dims_mask = vec![n_kv as u32];
    let mut dims_sinks = vec![n_head as u32];
    let mut dims_score = vec![1u32];

    // Tensor name CStrings — separate copies per graph because both graphs
    // independently call tensorCreateGraphTensor (names are scoped per graph).
    let nm_x = CString::new("x").unwrap();
    let nm_rms_pre = CString::new("rms_pre").unwrap();
    let nm_rms_post = CString::new("rms_post").unwrap();
    let nm_y1 = CString::new("y1").unwrap();
    let nm_qq = CString::new("qq").unwrap();
    let nm_qd = CString::new("qd").unwrap();
    let nm_kq = CString::new("kq").unwrap();
    let nm_kd = CString::new("kd").unwrap();
    let nm_vq = CString::new("vq").unwrap();
    let nm_vd = CString::new("vd").unwrap();
    let nm_oq = CString::new("oq").unwrap();
    let nm_od = CString::new("od").unwrap();
    let nm_gq = CString::new("gq").unwrap();
    let nm_gd = CString::new("gd").unwrap();
    let nm_uq = CString::new("uq").unwrap();
    let nm_ud = CString::new("ud").unwrap();
    let nm_dq = CString::new("dq").unwrap();
    let nm_dd = CString::new("dd").unwrap();
    let nm_q = CString::new("q").unwrap();
    let nm_k = CString::new("k").unwrap();
    let nm_v = CString::new("v").unwrap();
    let nm_q_rope = CString::new("q_rope").unwrap();
    let nm_k_rope = CString::new("k_rope").unwrap();
    let nm_kcache = CString::new("kcache").unwrap();
    let nm_vcache = CString::new("vcache").unwrap();
    let nm_attn_o = CString::new("attn_o").unwrap();
    let nm_o = CString::new("o").unwrap();
    let nm_x_attn = CString::new("x_attn").unwrap();
    let nm_y2 = CString::new("y2").unwrap();
    let nm_gate = CString::new("gate").unwrap();
    let nm_up = CString::new("up").unwrap();
    let nm_silu_out = CString::new("silu_out").unwrap();
    let nm_down = CString::new("down").unwrap();
    let nm_x_out = CString::new("x_out").unwrap();
    let nm_mask = CString::new("mask").unwrap();
    let nm_sinks = CString::new("sinks").unwrap();
    let nm_score = CString::new("score").unwrap();

    // ── 5. Run 6 sub-graphs sequentially. Each one builds a fresh
    //    Qnn_GraphHandle_t in the same context, exports its endpoint(s) to
    //    host APP_READ rpcmem, reads them back, and accumulates max_abs_err
    //    against the matching raw-OpenCL reference stage.
    //
    //    Why a fresh graph per sub-graph: M2.H 7th confirmed that the SDK
    //    composer drifts somewhere across the 14-node DAG. Splitting into
    //    minimal sub-graphs (each <= 5 nodes) keeps each sub-graph small
    //    enough that the SDK composer cannot rearrange edges ambiguously
    //    while still exercising the chain compositions that fail at scale.
    //
    //    Pattern is intentionally repetitive (one block per sub-graph) so
    //    each block can be commented out individually for further bisecting.
    let mk_desc = |fd: i32,
                   host_data: *mut std::ffi::c_void,
                   dtype: Qnn_DataType_t,
                   dims: &[u32]|
     -> Qnn_MemDescriptor_t {
        Qnn_MemDescriptor_t {
            memShape: Qnn_MemShape_t {
                numDim: dims.len() as u32,
                dimSize: dims.as_ptr() as *mut u32,
                shapeConfig: ptr::null(),
            },
            dataType: dtype,
            memType: Qnn_MemType_t_QNN_MEM_TYPE_DMA_BUF,
            __bindgen_anon_1: Qnn_MemDescriptor_t__bindgen_ty_1 {
                dmaBufInfo: Qnn_MemDmaBufInfo_t {
                    fd,
                    data: host_data,
                },
            },
        }
    };
    let set_mh = |t: &mut Qnn_Tensor_t, h: *mut std::ffi::c_void| {
        t.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        t.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = h;
    };
    let pkg = CString::new("qnn_oppkg").unwrap();
    let make_op = |name: *const c_char,
                   typ: *const c_char,
                   ins: *mut Qnn_Tensor_t,
                   n_in: u32,
                   outs: *mut Qnn_Tensor_t,
                   n_out: u32,
                   params: *mut Qnn_Param_t,
                   n_params: u32|
     -> Qnn_OpConfig_t {
        Qnn_OpConfig_t {
            version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
            __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
                v1: Qnn_OpConfigV1_t {
                    name,
                    packageName: pkg.as_ptr(),
                    typeName: typ,
                    numOfParams: n_params,
                    params,
                    numOfInputs: n_in,
                    inputTensors: ins,
                    numOfOutputs: n_out,
                    outputTensors: outs,
                },
            },
        }
    };

    // Param-name CStrings (shared across sub-graphs).
    let pn_start_pos = CString::new("start_pos").unwrap();
    let pn_theta = CString::new("theta").unwrap();
    let pn_head_dim = CString::new("head_dim").unwrap();
    let pn_capacity = CString::new("capacity").unwrap();
    let pn_write_pos = CString::new("write_pos").unwrap();
    let pn_n_kv = CString::new("n_kv").unwrap();
    let pn_n_head = CString::new("n_head").unwrap();
    let pn_n_head_kv = CString::new("n_head_kv").unwrap();
    let pn_kv_capacity = CString::new("kv_capacity").unwrap();
    let pn_head_dim_fa = CString::new("head_dim").unwrap();

    // Op-type CStrings.
    let ot_rms = CString::new("CustomRmsNorm").unwrap();
    let ot_q40 = CString::new("CustomMatMulQ40F32").unwrap();
    let ot_rope = CString::new("CustomRope").unwrap();
    let ot_kvs = CString::new("CustomKvScatter").unwrap();
    let ot_fa = CString::new("CustomFlashAttn").unwrap();
    let ot_silu = CString::new("CustomSiluMul").unwrap();
    let ot_add = CString::new("CustomAdd").unwrap();

    // Helper: param builders.
    let mk_rope_params = |start: i32, th: f32| {
        [
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: pn_start_pos.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 { int32Value: start },
                    },
                },
            },
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: pn_theta.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 { floatValue: th },
                    },
                },
            },
        ]
    };

    let mut stages: Vec<(String, f32)> = Vec::with_capacity(6);

    // ───────────────────────────────────────────────────────────────────────
    // Sub-graph 1: RmsNorm(pre) -> Q proj -> RoPE Q  =>  q_rope
    // Tests: RmsNorm + matmul rank-3 reshape + RoPE OOP composition.
    // ───────────────────────────────────────────────────────────────────────
    {
        // Zero out endpoint before run.
        unsafe { std::ptr::write_bytes(rpc_q_rope as *mut u8, 0, bytes_q_rope as usize) };

        let mut t_x = build_tensor(app_w, f32_t, 2, dims_x.as_mut_ptr(), nm_x.as_ptr());
        let mut t_rms_pre = build_tensor(
            app_w,
            f32_t,
            1,
            dims_rms_pre.as_mut_ptr(),
            nm_rms_pre.as_ptr(),
        );
        let mut t_qq = build_tensor(app_w, u8_t, 1, dims_qq.as_mut_ptr(), nm_qq.as_ptr());
        let mut t_qd = build_tensor(app_w, f16_t, 1, dims_qd.as_mut_ptr(), nm_qd.as_ptr());
        let mut t_q_rope = build_tensor(
            app_r,
            f32_t,
            3,
            dims_q_rope.as_mut_ptr(),
            nm_q_rope.as_ptr(),
        );
        let mut t_y1 = build_tensor(native, f32_t, 2, dims_y1.as_mut_ptr(), nm_y1.as_ptr());
        let mut t_q = build_tensor(native, f32_t, 3, dims_q.as_mut_ptr(), nm_q.as_ptr());

        let g_name = CString::new("sg1").unwrap();
        let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
        let err =
            unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
        anyhow::ensure!(err == 0, "graphCreate(sg1) err=0x{:x}", err);

        let regs: &mut [(&str, &mut Qnn_Tensor_t)] = &mut [
            ("x", &mut t_x),
            ("rms_pre", &mut t_rms_pre),
            ("qq", &mut t_qq),
            ("qd", &mut t_qd),
            ("q_rope", &mut t_q_rope),
            ("y1", &mut t_y1),
            ("q", &mut t_q),
        ];
        for (label, t) in regs.iter_mut() {
            let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, *t) };
            anyhow::ensure!(err == 0, "sg1 tensorCreate({}) err=0x{:x}", label, err);
        }

        let mut rope_q_params = mk_rope_params(pos, theta);

        let mut in_rms = [t_x, t_rms_pre];
        let mut out_rms = [t_y1];
        let mut in_q = [t_qq, t_qd, t_y1];
        let mut out_q = [t_q];
        let mut in_rope = [t_q];
        let mut out_rope = [t_q_rope];

        let on_rms = CString::new("sg1_rms").unwrap();
        let on_q = CString::new("sg1_q").unwrap();
        let on_rope = CString::new("sg1_rope").unwrap();

        let nodes: [(
            *const c_char,
            *const c_char,
            *mut Qnn_Tensor_t,
            u32,
            *mut Qnn_Tensor_t,
            u32,
            *mut Qnn_Param_t,
            u32,
            &str,
        ); 3] = [
            (
                on_rms.as_ptr(),
                ot_rms.as_ptr(),
                in_rms.as_mut_ptr(),
                2,
                out_rms.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "sg1.RmsNorm",
            ),
            (
                on_q.as_ptr(),
                ot_q40.as_ptr(),
                in_q.as_mut_ptr(),
                3,
                out_q.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "sg1.Q proj",
            ),
            (
                on_rope.as_ptr(),
                ot_rope.as_ptr(),
                in_rope.as_mut_ptr(),
                1,
                out_rope.as_mut_ptr(),
                1,
                rope_q_params.as_mut_ptr(),
                2,
                "sg1.RoPE Q",
            ),
        ];
        for (nm, ty, ins, n_in, outs, n_out, params, n_params, label) in nodes {
            let op = make_op(nm, ty, ins, n_in, outs, n_out, params, n_params);
            let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
            anyhow::ensure!(err == 0, "graphAddNode({}) err=0x{:x}", label, err);
        }
        let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
        anyhow::ensure!(err == 0, "graphFinalize(sg1) err=0x{:x}", err);

        let descs = [
            mk_desc(fd_x, rpc_x, f32_t, &dims_x),
            mk_desc(fd_rms_pre, rpc_rms_pre, f32_t, &dims_rms_pre),
            mk_desc(fd_qq, rpc_qq, u8_t, &dims_qq),
            mk_desc(fd_qd, rpc_qd, f16_t, &dims_qd),
            mk_desc(fd_q_rope, rpc_q_rope, f32_t, &dims_q_rope),
        ];
        let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 5];
        let err = unsafe {
            (v.memRegister.unwrap())(ctx, descs.as_ptr(), descs.len() as u32, mh.as_mut_ptr())
        };
        anyhow::ensure!(err == 0, "memRegister(sg1) err=0x{:x}", err);

        let mut t_x_mh = t_x;
        set_mh(&mut t_x_mh, mh[0]);
        let mut t_rms_pre_mh = t_rms_pre;
        set_mh(&mut t_rms_pre_mh, mh[1]);
        let mut t_qq_mh = t_qq;
        set_mh(&mut t_qq_mh, mh[2]);
        let mut t_qd_mh = t_qd;
        set_mh(&mut t_qd_mh, mh[3]);
        let mut t_q_rope_mh = t_q_rope;
        set_mh(&mut t_q_rope_mh, mh[4]);

        let exec_in = [t_x_mh, t_rms_pre_mh, t_qq_mh, t_qd_mh];
        let mut exec_out = [t_q_rope_mh];
        let err = unsafe {
            (v.graphExecute.unwrap())(
                graph,
                exec_in.as_ptr(),
                exec_in.len() as u32,
                exec_out.as_mut_ptr(),
                exec_out.len() as u32,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        anyhow::ensure!(err == 0, "graphExecute(sg1) err=0x{:x}", err);
        unsafe {
            let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), mh.len() as u32);
        }

        let qnn_view = unsafe { std::slice::from_raw_parts(rpc_q_rope as *const f32, q_proj_out) };
        let max_abs = ref_q_rope
            .iter()
            .zip(qnn_view.iter())
            .map(|(r, q)| (r - q).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[sg1] q_rope[0..8] qnn={:?} ref={:?}",
            &qnn_view[..8],
            &ref_q_rope[..8]
        );
        println!("[sg1] q_rope max_abs_err = {:.6e}", max_abs);
        stages.push(("1.q_rot".into(), max_abs));
    }

    // ───────────────────────────────────────────────────────────────────────
    // Sub-graph 2: RmsNorm(pre) -> K proj -> V proj -> RoPE K -> KvScatter
    //                                                       =>  kcache, vcache
    // Tests: KvScatter multi-output + post-RoPE binding.
    // ───────────────────────────────────────────────────────────────────────
    {
        unsafe {
            std::ptr::write_bytes(rpc_kcache as *mut u8, 0, bytes_kv as usize);
            std::ptr::write_bytes(rpc_vcache as *mut u8, 0, bytes_kv as usize);
        }

        let mut t_x = build_tensor(app_w, f32_t, 2, dims_x.as_mut_ptr(), nm_x.as_ptr());
        let mut t_rms_pre = build_tensor(
            app_w,
            f32_t,
            1,
            dims_rms_pre.as_mut_ptr(),
            nm_rms_pre.as_ptr(),
        );
        let mut t_kq = build_tensor(app_w, u8_t, 1, dims_kq.as_mut_ptr(), nm_kq.as_ptr());
        let mut t_kd = build_tensor(app_w, f16_t, 1, dims_kd.as_mut_ptr(), nm_kd.as_ptr());
        let mut t_vq = build_tensor(app_w, u8_t, 1, dims_vq.as_mut_ptr(), nm_vq.as_ptr());
        let mut t_vd = build_tensor(app_w, f16_t, 1, dims_vd.as_mut_ptr(), nm_vd.as_ptr());
        let mut t_kcache = build_tensor(
            app_r,
            f16_t,
            4,
            dims_kcache.as_mut_ptr(),
            nm_kcache.as_ptr(),
        );
        let mut t_vcache = build_tensor(
            app_r,
            f16_t,
            4,
            dims_vcache.as_mut_ptr(),
            nm_vcache.as_ptr(),
        );
        let mut t_y1 = build_tensor(native, f32_t, 2, dims_y1.as_mut_ptr(), nm_y1.as_ptr());
        let mut t_k = build_tensor(native, f32_t, 3, dims_kvec.as_mut_ptr(), nm_k.as_ptr());
        let mut t_v = build_tensor(native, f32_t, 3, dims_vvec.as_mut_ptr(), nm_v.as_ptr());
        let mut t_k_rope = build_tensor(
            native,
            f32_t,
            3,
            dims_k_rope.as_mut_ptr(),
            nm_k_rope.as_ptr(),
        );

        let g_name = CString::new("sg2").unwrap();
        let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
        let err =
            unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
        anyhow::ensure!(err == 0, "graphCreate(sg2) err=0x{:x}", err);

        let regs: &mut [(&str, &mut Qnn_Tensor_t)] = &mut [
            ("x", &mut t_x),
            ("rms_pre", &mut t_rms_pre),
            ("kq", &mut t_kq),
            ("kd", &mut t_kd),
            ("vq", &mut t_vq),
            ("vd", &mut t_vd),
            ("kcache", &mut t_kcache),
            ("vcache", &mut t_vcache),
            ("y1", &mut t_y1),
            ("k", &mut t_k),
            ("v", &mut t_v),
            ("k_rope", &mut t_k_rope),
        ];
        for (label, t) in regs.iter_mut() {
            let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, *t) };
            anyhow::ensure!(err == 0, "sg2 tensorCreate({}) err=0x{:x}", label, err);
        }

        let mut rope_k_params = mk_rope_params(pos, theta);
        let mut kvs_params = [
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: pn_head_dim.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: head_dim as i32,
                        },
                    },
                },
            },
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: pn_capacity.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: kv_capacity as i32,
                        },
                    },
                },
            },
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: pn_write_pos.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 { int32Value: pos },
                    },
                },
            },
        ];

        let mut in_rms = [t_x, t_rms_pre];
        let mut out_rms = [t_y1];
        let mut in_k = [t_kq, t_kd, t_y1];
        let mut out_k = [t_k];
        let mut in_v = [t_vq, t_vd, t_y1];
        let mut out_v = [t_v];
        let mut in_rope = [t_k];
        let mut out_rope = [t_k_rope];
        let mut in_kvs = [t_k_rope, t_v];
        let mut out_kvs = [t_kcache, t_vcache];

        let on_rms = CString::new("sg2_rms").unwrap();
        let on_k = CString::new("sg2_k").unwrap();
        let on_v = CString::new("sg2_v").unwrap();
        let on_rope = CString::new("sg2_rope").unwrap();
        let on_kvs = CString::new("sg2_kvs").unwrap();

        let nodes: [(
            *const c_char,
            *const c_char,
            *mut Qnn_Tensor_t,
            u32,
            *mut Qnn_Tensor_t,
            u32,
            *mut Qnn_Param_t,
            u32,
            &str,
        ); 5] = [
            (
                on_rms.as_ptr(),
                ot_rms.as_ptr(),
                in_rms.as_mut_ptr(),
                2,
                out_rms.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "sg2.RmsNorm",
            ),
            (
                on_k.as_ptr(),
                ot_q40.as_ptr(),
                in_k.as_mut_ptr(),
                3,
                out_k.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "sg2.K proj",
            ),
            (
                on_v.as_ptr(),
                ot_q40.as_ptr(),
                in_v.as_mut_ptr(),
                3,
                out_v.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "sg2.V proj",
            ),
            (
                on_rope.as_ptr(),
                ot_rope.as_ptr(),
                in_rope.as_mut_ptr(),
                1,
                out_rope.as_mut_ptr(),
                1,
                rope_k_params.as_mut_ptr(),
                2,
                "sg2.RoPE K",
            ),
            (
                on_kvs.as_ptr(),
                ot_kvs.as_ptr(),
                in_kvs.as_mut_ptr(),
                2,
                out_kvs.as_mut_ptr(),
                2,
                kvs_params.as_mut_ptr(),
                3,
                "sg2.KvScatter",
            ),
        ];
        for (nm, ty, ins, n_in, outs, n_out, params, n_params, label) in nodes {
            let op = make_op(nm, ty, ins, n_in, outs, n_out, params, n_params);
            let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
            anyhow::ensure!(err == 0, "graphAddNode({}) err=0x{:x}", label, err);
        }
        let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
        anyhow::ensure!(err == 0, "graphFinalize(sg2) err=0x{:x}", err);

        let descs = [
            mk_desc(fd_x, rpc_x, f32_t, &dims_x),
            mk_desc(fd_rms_pre, rpc_rms_pre, f32_t, &dims_rms_pre),
            mk_desc(fd_kq, rpc_kq, u8_t, &dims_kq),
            mk_desc(fd_kd, rpc_kd, f16_t, &dims_kd),
            mk_desc(fd_vq, rpc_vq, u8_t, &dims_vq),
            mk_desc(fd_vd, rpc_vd, f16_t, &dims_vd),
            mk_desc(fd_kcache, rpc_kcache, f16_t, &dims_kcache),
            mk_desc(fd_vcache, rpc_vcache, f16_t, &dims_vcache),
        ];
        let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 8];
        let err = unsafe {
            (v.memRegister.unwrap())(ctx, descs.as_ptr(), descs.len() as u32, mh.as_mut_ptr())
        };
        anyhow::ensure!(err == 0, "memRegister(sg2) err=0x{:x}", err);

        let mut t_x_mh = t_x;
        set_mh(&mut t_x_mh, mh[0]);
        let mut t_rms_pre_mh = t_rms_pre;
        set_mh(&mut t_rms_pre_mh, mh[1]);
        let mut t_kq_mh = t_kq;
        set_mh(&mut t_kq_mh, mh[2]);
        let mut t_kd_mh = t_kd;
        set_mh(&mut t_kd_mh, mh[3]);
        let mut t_vq_mh = t_vq;
        set_mh(&mut t_vq_mh, mh[4]);
        let mut t_vd_mh = t_vd;
        set_mh(&mut t_vd_mh, mh[5]);
        let mut t_kcache_mh = t_kcache;
        set_mh(&mut t_kcache_mh, mh[6]);
        let mut t_vcache_mh = t_vcache;
        set_mh(&mut t_vcache_mh, mh[7]);

        let exec_in = [t_x_mh, t_rms_pre_mh, t_kq_mh, t_kd_mh, t_vq_mh, t_vd_mh];
        let mut exec_out = [t_kcache_mh, t_vcache_mh];
        let err = unsafe {
            (v.graphExecute.unwrap())(
                graph,
                exec_in.as_ptr(),
                exec_in.len() as u32,
                exec_out.as_mut_ptr(),
                exec_out.len() as u32,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        anyhow::ensure!(err == 0, "graphExecute(sg2) err=0x{:x}", err);
        unsafe {
            let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), mh.len() as u32);
        }

        let qnn_kc = unsafe { std::slice::from_raw_parts(rpc_kcache as *const u16, kv_total) };
        let qnn_vc = unsafe { std::slice::from_raw_parts(rpc_vcache as *const u16, kv_total) };
        let f16_diff = |r: u16, q: u16| -> f32 { (f16_bits_to_f32(r) - f16_bits_to_f32(q)).abs() };
        let max_abs_kc = ref_kcache_full
            .iter()
            .zip(qnn_kc.iter())
            .map(|(r, q)| f16_diff(*r, *q))
            .fold(0.0f32, f32::max);
        let max_abs_vc = ref_vcache_full
            .iter()
            .zip(qnn_vc.iter())
            .map(|(r, q)| f16_diff(*r, *q))
            .fold(0.0f32, f32::max);
        let max_abs = max_abs_kc.max(max_abs_vc);
        println!(
            "[sg2] kcache max_abs_err = {:.6e}, vcache max_abs_err = {:.6e}",
            max_abs_kc, max_abs_vc
        );
        stages.push(("2.kcache+vcache".into(), max_abs));
    }

    // ───────────────────────────────────────────────────────────────────────
    // Sub-graph 3: (host) q_rope + kcache + vcache => FlashAttn => attn_o
    // Tests: FlashAttn with host-bound F32 Q + F16 KV cache (max-padded
    // [1, kv_heads, capacity, head_dim]) — same shapes as sub-graph 2 wrote.
    //
    // Isolation: prime q_rope/kcache/vcache with the raw-OpenCL reference
    // values (overwrites any drift produced by sg1/sg2) so any error here is
    // purely from FlashAttn composition.
    // ───────────────────────────────────────────────────────────────────────
    {
        unsafe {
            std::ptr::copy_nonoverlapping(
                ref_q_rope.as_ptr() as *const u8,
                rpc_q_rope as *mut u8,
                bytes_q_rope as usize,
            );
            std::ptr::copy_nonoverlapping(
                ref_kcache_full.as_ptr() as *const u8,
                rpc_kcache as *mut u8,
                bytes_kv as usize,
            );
            std::ptr::copy_nonoverlapping(
                ref_vcache_full.as_ptr() as *const u8,
                rpc_vcache as *mut u8,
                bytes_kv as usize,
            );
            std::ptr::write_bytes(rpc_attn_o as *mut u8, 0, bytes_attn_o as usize);
        }

        let mut t_q_rope = build_tensor(
            app_w,
            f32_t,
            3,
            dims_q_rope.as_mut_ptr(),
            nm_q_rope.as_ptr(),
        );
        let mut t_kcache = build_tensor(
            app_w,
            f16_t,
            4,
            dims_kcache.as_mut_ptr(),
            nm_kcache.as_ptr(),
        );
        let mut t_vcache = build_tensor(
            app_w,
            f16_t,
            4,
            dims_vcache.as_mut_ptr(),
            nm_vcache.as_ptr(),
        );
        let mut t_mask = build_tensor(app_w, f16_t, 1, dims_mask.as_mut_ptr(), nm_mask.as_ptr());
        let mut t_sinks = build_tensor(app_w, f32_t, 1, dims_sinks.as_mut_ptr(), nm_sinks.as_ptr());
        let mut t_score = build_tensor(app_w, f32_t, 1, dims_score.as_mut_ptr(), nm_score.as_ptr());
        let mut t_attn_o = build_tensor(
            app_r,
            f32_t,
            3,
            dims_attn_o.as_mut_ptr(),
            nm_attn_o.as_ptr(),
        );

        let g_name = CString::new("sg3").unwrap();
        let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
        let err =
            unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
        anyhow::ensure!(err == 0, "graphCreate(sg3) err=0x{:x}", err);

        let regs: &mut [(&str, &mut Qnn_Tensor_t)] = &mut [
            ("q_rope", &mut t_q_rope),
            ("kcache", &mut t_kcache),
            ("vcache", &mut t_vcache),
            ("mask", &mut t_mask),
            ("sinks", &mut t_sinks),
            ("score", &mut t_score),
            ("attn_o", &mut t_attn_o),
        ];
        for (label, t) in regs.iter_mut() {
            let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, *t) };
            anyhow::ensure!(err == 0, "sg3 tensorCreate({}) err=0x{:x}", label, err);
        }

        let mut fa_params = [
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: pn_n_kv.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: n_kv as i32,
                        },
                    },
                },
            },
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: pn_n_head.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: n_head as i32,
                        },
                    },
                },
            },
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: pn_n_head_kv.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: n_kv_heads as i32,
                        },
                    },
                },
            },
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: pn_kv_capacity.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: kv_capacity as i32,
                        },
                    },
                },
            },
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: pn_head_dim_fa.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: head_dim as i32,
                        },
                    },
                },
            },
        ];

        let mut in_fa = [t_q_rope, t_kcache, t_vcache, t_mask, t_sinks, t_score];
        let mut out_fa = [t_attn_o];
        let on_fa = CString::new("sg3_fa").unwrap();
        let op = make_op(
            on_fa.as_ptr(),
            ot_fa.as_ptr(),
            in_fa.as_mut_ptr(),
            6,
            out_fa.as_mut_ptr(),
            1,
            fa_params.as_mut_ptr(),
            5,
        );
        let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
        anyhow::ensure!(err == 0, "graphAddNode(sg3.FlashAttn) err=0x{:x}", err);

        let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
        anyhow::ensure!(err == 0, "graphFinalize(sg3) err=0x{:x}", err);

        let descs = [
            mk_desc(fd_q_rope, rpc_q_rope, f32_t, &dims_q_rope),
            mk_desc(fd_kcache, rpc_kcache, f16_t, &dims_kcache),
            mk_desc(fd_vcache, rpc_vcache, f16_t, &dims_vcache),
            mk_desc(fd_mask, rpc_mask, f16_t, &dims_mask),
            mk_desc(fd_sinks, rpc_sinks, f32_t, &dims_sinks),
            mk_desc(fd_score, rpc_score, f32_t, &dims_score),
            mk_desc(fd_attn_o, rpc_attn_o, f32_t, &dims_attn_o),
        ];
        let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 7];
        let err = unsafe {
            (v.memRegister.unwrap())(ctx, descs.as_ptr(), descs.len() as u32, mh.as_mut_ptr())
        };
        anyhow::ensure!(err == 0, "memRegister(sg3) err=0x{:x}", err);

        let mut t_q_rope_mh = t_q_rope;
        set_mh(&mut t_q_rope_mh, mh[0]);
        let mut t_kcache_mh = t_kcache;
        set_mh(&mut t_kcache_mh, mh[1]);
        let mut t_vcache_mh = t_vcache;
        set_mh(&mut t_vcache_mh, mh[2]);
        let mut t_mask_mh = t_mask;
        set_mh(&mut t_mask_mh, mh[3]);
        let mut t_sinks_mh = t_sinks;
        set_mh(&mut t_sinks_mh, mh[4]);
        let mut t_score_mh = t_score;
        set_mh(&mut t_score_mh, mh[5]);
        let mut t_attn_o_mh = t_attn_o;
        set_mh(&mut t_attn_o_mh, mh[6]);

        let exec_in = [
            t_q_rope_mh,
            t_kcache_mh,
            t_vcache_mh,
            t_mask_mh,
            t_sinks_mh,
            t_score_mh,
        ];
        let mut exec_out = [t_attn_o_mh];
        let err = unsafe {
            (v.graphExecute.unwrap())(
                graph,
                exec_in.as_ptr(),
                exec_in.len() as u32,
                exec_out.as_mut_ptr(),
                exec_out.len() as u32,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        anyhow::ensure!(err == 0, "graphExecute(sg3) err=0x{:x}", err);
        unsafe {
            let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), mh.len() as u32);
        }

        let qnn_view = unsafe { std::slice::from_raw_parts(rpc_attn_o as *const f32, q_proj_out) };
        let max_abs = ref_attn_o
            .iter()
            .zip(qnn_view.iter())
            .map(|(r, q)| (r - q).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[sg3] attn_o[0..8] qnn={:?} ref={:?}",
            &qnn_view[..8],
            &ref_attn_o[..8]
        );
        println!("[sg3] attn_o max_abs_err = {:.6e}", max_abs);
        stages.push(("3.attn_o".into(), max_abs));
    }

    // ───────────────────────────────────────────────────────────────────────
    // Sub-graph 4: (host) attn_o + x => O proj + Add(residual1) => x_attn
    // Tests: Q4_0 matmul rank-3 input -> rank-2 output + Add chain.
    //
    // Isolation: prime attn_o with reference; x is the original input (already
    // in rpc_x).
    // ───────────────────────────────────────────────────────────────────────
    {
        unsafe {
            std::ptr::copy_nonoverlapping(
                ref_attn_o.as_ptr() as *const u8,
                rpc_attn_o as *mut u8,
                bytes_attn_o as usize,
            );
            std::ptr::write_bytes(rpc_x_attn as *mut u8, 0, bytes_x_attn as usize);
        }

        let mut t_attn_o = build_tensor(
            app_w,
            f32_t,
            3,
            dims_attn_o.as_mut_ptr(),
            nm_attn_o.as_ptr(),
        );
        let mut t_x = build_tensor(app_w, f32_t, 2, dims_x.as_mut_ptr(), nm_x.as_ptr());
        let mut t_oq = build_tensor(app_w, u8_t, 1, dims_oq.as_mut_ptr(), nm_oq.as_ptr());
        let mut t_od = build_tensor(app_w, f16_t, 1, dims_od.as_mut_ptr(), nm_od.as_ptr());
        let mut t_x_attn = build_tensor(
            app_r,
            f32_t,
            2,
            dims_x_attn.as_mut_ptr(),
            nm_x_attn.as_ptr(),
        );
        let mut t_o = build_tensor(native, f32_t, 2, dims_o.as_mut_ptr(), nm_o.as_ptr());

        let g_name = CString::new("sg4").unwrap();
        let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
        let err =
            unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
        anyhow::ensure!(err == 0, "graphCreate(sg4) err=0x{:x}", err);

        let regs: &mut [(&str, &mut Qnn_Tensor_t)] = &mut [
            ("attn_o", &mut t_attn_o),
            ("x", &mut t_x),
            ("oq", &mut t_oq),
            ("od", &mut t_od),
            ("x_attn", &mut t_x_attn),
            ("o", &mut t_o),
        ];
        for (label, t) in regs.iter_mut() {
            let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, *t) };
            anyhow::ensure!(err == 0, "sg4 tensorCreate({}) err=0x{:x}", label, err);
        }

        let mut in_o = [t_oq, t_od, t_attn_o];
        let mut out_o = [t_o];
        let mut in_add = [t_o, t_x];
        let mut out_add = [t_x_attn];

        let on_o = CString::new("sg4_o").unwrap();
        let on_add = CString::new("sg4_add").unwrap();

        let nodes: [(
            *const c_char,
            *const c_char,
            *mut Qnn_Tensor_t,
            u32,
            *mut Qnn_Tensor_t,
            u32,
            *mut Qnn_Param_t,
            u32,
            &str,
        ); 2] = [
            (
                on_o.as_ptr(),
                ot_q40.as_ptr(),
                in_o.as_mut_ptr(),
                3,
                out_o.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "sg4.O proj",
            ),
            (
                on_add.as_ptr(),
                ot_add.as_ptr(),
                in_add.as_mut_ptr(),
                2,
                out_add.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "sg4.Add(residual1)",
            ),
        ];
        for (nm, ty, ins, n_in, outs, n_out, params, n_params, label) in nodes {
            let op = make_op(nm, ty, ins, n_in, outs, n_out, params, n_params);
            let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
            anyhow::ensure!(err == 0, "graphAddNode({}) err=0x{:x}", label, err);
        }
        let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
        anyhow::ensure!(err == 0, "graphFinalize(sg4) err=0x{:x}", err);

        let descs = [
            mk_desc(fd_attn_o, rpc_attn_o, f32_t, &dims_attn_o),
            mk_desc(fd_x, rpc_x, f32_t, &dims_x),
            mk_desc(fd_oq, rpc_oq, u8_t, &dims_oq),
            mk_desc(fd_od, rpc_od, f16_t, &dims_od),
            mk_desc(fd_x_attn, rpc_x_attn, f32_t, &dims_x_attn),
        ];
        let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 5];
        let err = unsafe {
            (v.memRegister.unwrap())(ctx, descs.as_ptr(), descs.len() as u32, mh.as_mut_ptr())
        };
        anyhow::ensure!(err == 0, "memRegister(sg4) err=0x{:x}", err);

        let mut t_attn_o_mh = t_attn_o;
        set_mh(&mut t_attn_o_mh, mh[0]);
        let mut t_x_mh = t_x;
        set_mh(&mut t_x_mh, mh[1]);
        let mut t_oq_mh = t_oq;
        set_mh(&mut t_oq_mh, mh[2]);
        let mut t_od_mh = t_od;
        set_mh(&mut t_od_mh, mh[3]);
        let mut t_x_attn_mh = t_x_attn;
        set_mh(&mut t_x_attn_mh, mh[4]);

        let exec_in = [t_attn_o_mh, t_x_mh, t_oq_mh, t_od_mh];
        let mut exec_out = [t_x_attn_mh];
        let err = unsafe {
            (v.graphExecute.unwrap())(
                graph,
                exec_in.as_ptr(),
                exec_in.len() as u32,
                exec_out.as_mut_ptr(),
                exec_out.len() as u32,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        anyhow::ensure!(err == 0, "graphExecute(sg4) err=0x{:x}", err);
        unsafe {
            let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), mh.len() as u32);
        }

        let qnn_view = unsafe { std::slice::from_raw_parts(rpc_x_attn as *const f32, dim) };
        let max_abs = ref_x_attn
            .iter()
            .zip(qnn_view.iter())
            .map(|(r, q)| (r - q).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[sg4] x_attn[0..8] qnn={:?} ref={:?}",
            &qnn_view[..8],
            &ref_x_attn[..8]
        );
        println!("[sg4] x_attn max_abs_err = {:.6e}", max_abs);
        stages.push(("4.x_attn".into(), max_abs));
    }

    // ───────────────────────────────────────────────────────────────────────
    // Sub-graph 5: (host) x_attn => RmsNorm(post) + gate/up + SiluMul
    //                          => silu_out
    // Tests: SiluMul OOP + matmul fan-out (gate || up) chain.
    //
    // Isolation: prime x_attn with reference.
    // ───────────────────────────────────────────────────────────────────────
    {
        unsafe {
            std::ptr::copy_nonoverlapping(
                ref_x_attn.as_ptr() as *const u8,
                rpc_x_attn as *mut u8,
                bytes_x_attn as usize,
            );
            std::ptr::write_bytes(rpc_silu_out as *mut u8, 0, bytes_silu_out as usize);
        }

        let mut t_x_attn = build_tensor(
            app_w,
            f32_t,
            2,
            dims_x_attn.as_mut_ptr(),
            nm_x_attn.as_ptr(),
        );
        let mut t_rms_post = build_tensor(
            app_w,
            f32_t,
            1,
            dims_rms_post.as_mut_ptr(),
            nm_rms_post.as_ptr(),
        );
        let mut t_gq = build_tensor(app_w, u8_t, 1, dims_gq.as_mut_ptr(), nm_gq.as_ptr());
        let mut t_gd = build_tensor(app_w, f16_t, 1, dims_gd.as_mut_ptr(), nm_gd.as_ptr());
        let mut t_uq = build_tensor(app_w, u8_t, 1, dims_uq.as_mut_ptr(), nm_uq.as_ptr());
        let mut t_ud = build_tensor(app_w, f16_t, 1, dims_ud.as_mut_ptr(), nm_ud.as_ptr());
        let mut t_silu_out = build_tensor(
            app_r,
            f32_t,
            2,
            dims_silu_out.as_mut_ptr(),
            nm_silu_out.as_ptr(),
        );
        let mut t_y2 = build_tensor(native, f32_t, 2, dims_y2.as_mut_ptr(), nm_y2.as_ptr());
        let mut t_gate = build_tensor(native, f32_t, 2, dims_gate.as_mut_ptr(), nm_gate.as_ptr());
        let mut t_up = build_tensor(native, f32_t, 2, dims_up.as_mut_ptr(), nm_up.as_ptr());

        let g_name = CString::new("sg5").unwrap();
        let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
        let err =
            unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
        anyhow::ensure!(err == 0, "graphCreate(sg5) err=0x{:x}", err);

        let regs: &mut [(&str, &mut Qnn_Tensor_t)] = &mut [
            ("x_attn", &mut t_x_attn),
            ("rms_post", &mut t_rms_post),
            ("gq", &mut t_gq),
            ("gd", &mut t_gd),
            ("uq", &mut t_uq),
            ("ud", &mut t_ud),
            ("silu_out", &mut t_silu_out),
            ("y2", &mut t_y2),
            ("gate", &mut t_gate),
            ("up", &mut t_up),
        ];
        for (label, t) in regs.iter_mut() {
            let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, *t) };
            anyhow::ensure!(err == 0, "sg5 tensorCreate({}) err=0x{:x}", label, err);
        }

        let mut in_rms = [t_x_attn, t_rms_post];
        let mut out_rms = [t_y2];
        let mut in_gate = [t_gq, t_gd, t_y2];
        let mut out_gate = [t_gate];
        let mut in_up = [t_uq, t_ud, t_y2];
        let mut out_up = [t_up];
        let mut in_silu = [t_gate, t_up];
        let mut out_silu = [t_silu_out];

        let on_rms = CString::new("sg5_rms").unwrap();
        let on_gate = CString::new("sg5_gate").unwrap();
        let on_up = CString::new("sg5_up").unwrap();
        let on_silu = CString::new("sg5_silu").unwrap();

        let nodes: [(
            *const c_char,
            *const c_char,
            *mut Qnn_Tensor_t,
            u32,
            *mut Qnn_Tensor_t,
            u32,
            *mut Qnn_Param_t,
            u32,
            &str,
        ); 4] = [
            (
                on_rms.as_ptr(),
                ot_rms.as_ptr(),
                in_rms.as_mut_ptr(),
                2,
                out_rms.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "sg5.RmsNorm(post)",
            ),
            (
                on_gate.as_ptr(),
                ot_q40.as_ptr(),
                in_gate.as_mut_ptr(),
                3,
                out_gate.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "sg5.gate proj",
            ),
            (
                on_up.as_ptr(),
                ot_q40.as_ptr(),
                in_up.as_mut_ptr(),
                3,
                out_up.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "sg5.up proj",
            ),
            (
                on_silu.as_ptr(),
                ot_silu.as_ptr(),
                in_silu.as_mut_ptr(),
                2,
                out_silu.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "sg5.SiluMul",
            ),
        ];
        for (nm, ty, ins, n_in, outs, n_out, params, n_params, label) in nodes {
            let op = make_op(nm, ty, ins, n_in, outs, n_out, params, n_params);
            let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
            anyhow::ensure!(err == 0, "graphAddNode({}) err=0x{:x}", label, err);
        }
        let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
        anyhow::ensure!(err == 0, "graphFinalize(sg5) err=0x{:x}", err);

        let descs = [
            mk_desc(fd_x_attn, rpc_x_attn, f32_t, &dims_x_attn),
            mk_desc(fd_rms_post, rpc_rms_post, f32_t, &dims_rms_post),
            mk_desc(fd_gq, rpc_gq, u8_t, &dims_gq),
            mk_desc(fd_gd, rpc_gd, f16_t, &dims_gd),
            mk_desc(fd_uq, rpc_uq, u8_t, &dims_uq),
            mk_desc(fd_ud, rpc_ud, f16_t, &dims_ud),
            mk_desc(fd_silu_out, rpc_silu_out, f32_t, &dims_silu_out),
        ];
        let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 7];
        let err = unsafe {
            (v.memRegister.unwrap())(ctx, descs.as_ptr(), descs.len() as u32, mh.as_mut_ptr())
        };
        anyhow::ensure!(err == 0, "memRegister(sg5) err=0x{:x}", err);

        let mut t_x_attn_mh = t_x_attn;
        set_mh(&mut t_x_attn_mh, mh[0]);
        let mut t_rms_post_mh = t_rms_post;
        set_mh(&mut t_rms_post_mh, mh[1]);
        let mut t_gq_mh = t_gq;
        set_mh(&mut t_gq_mh, mh[2]);
        let mut t_gd_mh = t_gd;
        set_mh(&mut t_gd_mh, mh[3]);
        let mut t_uq_mh = t_uq;
        set_mh(&mut t_uq_mh, mh[4]);
        let mut t_ud_mh = t_ud;
        set_mh(&mut t_ud_mh, mh[5]);
        let mut t_silu_out_mh = t_silu_out;
        set_mh(&mut t_silu_out_mh, mh[6]);

        let exec_in = [
            t_x_attn_mh,
            t_rms_post_mh,
            t_gq_mh,
            t_gd_mh,
            t_uq_mh,
            t_ud_mh,
        ];
        let mut exec_out = [t_silu_out_mh];
        let err = unsafe {
            (v.graphExecute.unwrap())(
                graph,
                exec_in.as_ptr(),
                exec_in.len() as u32,
                exec_out.as_mut_ptr(),
                exec_out.len() as u32,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        anyhow::ensure!(err == 0, "graphExecute(sg5) err=0x{:x}", err);
        unsafe {
            let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), mh.len() as u32);
        }

        let qnn_view = unsafe { std::slice::from_raw_parts(rpc_silu_out as *const f32, ffn_dim) };
        let max_abs = ref_silu_out
            .iter()
            .zip(qnn_view.iter())
            .map(|(r, q)| (r - q).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[sg5] silu_out[0..8] qnn={:?} ref={:?}",
            &qnn_view[..8],
            &ref_silu_out[..8]
        );
        println!("[sg5] silu_out max_abs_err = {:.6e}", max_abs);
        stages.push(("5.silu_out".into(), max_abs));
    }

    // ───────────────────────────────────────────────────────────────────────
    // Sub-graph 6: (host) silu_out + x_attn => Down + Add(residual2) => x_out
    // Tests: SiluMul output rank handling into Down proj and residual chain.
    //
    // Isolation: prime silu_out and x_attn with reference values.
    // ───────────────────────────────────────────────────────────────────────
    {
        unsafe {
            std::ptr::copy_nonoverlapping(
                ref_silu_out.as_ptr() as *const u8,
                rpc_silu_out as *mut u8,
                bytes_silu_out as usize,
            );
            std::ptr::copy_nonoverlapping(
                ref_x_attn.as_ptr() as *const u8,
                rpc_x_attn as *mut u8,
                bytes_x_attn as usize,
            );
            std::ptr::write_bytes(rpc_x_out as *mut u8, 0, bytes_x_out as usize);
        }

        let mut t_silu_out = build_tensor(
            app_w,
            f32_t,
            2,
            dims_silu_out.as_mut_ptr(),
            nm_silu_out.as_ptr(),
        );
        let mut t_x_attn = build_tensor(
            app_w,
            f32_t,
            2,
            dims_x_attn.as_mut_ptr(),
            nm_x_attn.as_ptr(),
        );
        let mut t_dq = build_tensor(app_w, u8_t, 1, dims_dq.as_mut_ptr(), nm_dq.as_ptr());
        let mut t_dd = build_tensor(app_w, f16_t, 1, dims_dd.as_mut_ptr(), nm_dd.as_ptr());
        let mut t_x_out = build_tensor(app_r, f32_t, 2, dims_x_out.as_mut_ptr(), nm_x_out.as_ptr());
        let mut t_down = build_tensor(native, f32_t, 2, dims_down.as_mut_ptr(), nm_down.as_ptr());

        let g_name = CString::new("sg6").unwrap();
        let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
        let err =
            unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
        anyhow::ensure!(err == 0, "graphCreate(sg6) err=0x{:x}", err);

        let regs: &mut [(&str, &mut Qnn_Tensor_t)] = &mut [
            ("silu_out", &mut t_silu_out),
            ("x_attn", &mut t_x_attn),
            ("dq", &mut t_dq),
            ("dd", &mut t_dd),
            ("x_out", &mut t_x_out),
            ("down", &mut t_down),
        ];
        for (label, t) in regs.iter_mut() {
            let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, *t) };
            anyhow::ensure!(err == 0, "sg6 tensorCreate({}) err=0x{:x}", label, err);
        }

        let mut in_down = [t_dq, t_dd, t_silu_out];
        let mut out_down = [t_down];
        let mut in_add = [t_down, t_x_attn];
        let mut out_add = [t_x_out];

        let on_down = CString::new("sg6_down").unwrap();
        let on_add = CString::new("sg6_add").unwrap();

        let nodes: [(
            *const c_char,
            *const c_char,
            *mut Qnn_Tensor_t,
            u32,
            *mut Qnn_Tensor_t,
            u32,
            *mut Qnn_Param_t,
            u32,
            &str,
        ); 2] = [
            (
                on_down.as_ptr(),
                ot_q40.as_ptr(),
                in_down.as_mut_ptr(),
                3,
                out_down.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "sg6.Down proj",
            ),
            (
                on_add.as_ptr(),
                ot_add.as_ptr(),
                in_add.as_mut_ptr(),
                2,
                out_add.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "sg6.Add(residual2)",
            ),
        ];
        for (nm, ty, ins, n_in, outs, n_out, params, n_params, label) in nodes {
            let op = make_op(nm, ty, ins, n_in, outs, n_out, params, n_params);
            let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
            anyhow::ensure!(err == 0, "graphAddNode({}) err=0x{:x}", label, err);
        }
        let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
        anyhow::ensure!(err == 0, "graphFinalize(sg6) err=0x{:x}", err);

        let descs = [
            mk_desc(fd_silu_out, rpc_silu_out, f32_t, &dims_silu_out),
            mk_desc(fd_x_attn, rpc_x_attn, f32_t, &dims_x_attn),
            mk_desc(fd_dq, rpc_dq, u8_t, &dims_dq),
            mk_desc(fd_dd, rpc_dd, f16_t, &dims_dd),
            mk_desc(fd_x_out, rpc_x_out, f32_t, &dims_x_out),
        ];
        let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 5];
        let err = unsafe {
            (v.memRegister.unwrap())(ctx, descs.as_ptr(), descs.len() as u32, mh.as_mut_ptr())
        };
        anyhow::ensure!(err == 0, "memRegister(sg6) err=0x{:x}", err);

        let mut t_silu_out_mh = t_silu_out;
        set_mh(&mut t_silu_out_mh, mh[0]);
        let mut t_x_attn_mh = t_x_attn;
        set_mh(&mut t_x_attn_mh, mh[1]);
        let mut t_dq_mh = t_dq;
        set_mh(&mut t_dq_mh, mh[2]);
        let mut t_dd_mh = t_dd;
        set_mh(&mut t_dd_mh, mh[3]);
        let mut t_x_out_mh = t_x_out;
        set_mh(&mut t_x_out_mh, mh[4]);

        let exec_in = [t_silu_out_mh, t_x_attn_mh, t_dq_mh, t_dd_mh];
        let mut exec_out = [t_x_out_mh];
        let err = unsafe {
            (v.graphExecute.unwrap())(
                graph,
                exec_in.as_ptr(),
                exec_in.len() as u32,
                exec_out.as_mut_ptr(),
                exec_out.len() as u32,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        anyhow::ensure!(err == 0, "graphExecute(sg6) err=0x{:x}", err);
        unsafe {
            let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), mh.len() as u32);
        }

        let qnn_view = unsafe { std::slice::from_raw_parts(rpc_x_out as *const f32, dim) };
        let max_abs = ref_x_out
            .iter()
            .zip(qnn_view.iter())
            .map(|(r, q)| (r - q).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[sg6] x_out[0..8] qnn={:?} ref={:?}",
            &qnn_view[..8],
            &ref_x_out[..8]
        );
        println!("[sg6] x_out max_abs_err = {:.6e}", max_abs);
        stages.push(("6.x_out".into(), max_abs));
    }

    // ── 6. Return collected per-sub-graph results ──────────────────────────
    //    Sub-graphs 3..6 use raw-OpenCL reference values as inputs (primed at
    //    the top of each block) so divergence reflects QNN composition drift
    //    only — not propagation of upstream sub-graph drift. Sub-graphs 1+2
    //    consume the original `x` / RMS weights so any divergence there is
    //    intrinsic to RmsNorm + Q/K/V proj + RoPE/KvScatter composition.
    Ok(LayerMetrics { stages })
}

#[cfg(feature = "qnn")]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let frac = mant as f32 / 1024.0;
        return if sign == 1 {
            -frac * 2.0f32.powi(-14)
        } else {
            frac * 2.0f32.powi(-14)
        };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    let new_exp = (exp - 15 + 127) as u32;
    let new_mant = mant << 13;
    f32::from_bits((sign << 31) | (new_exp << 23) | new_mant)
}

#[cfg(feature = "qnn")]
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 {
        return sign << 15;
    }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 {
        return sign << 15;
    }
    if new_exp >= 31 {
        return (sign << 15) | (0x1f << 10);
    }
    let new_mant = (mant >> 13) as u16;
    (sign << 15) | ((new_exp as u16) << 10) | new_mant
}
