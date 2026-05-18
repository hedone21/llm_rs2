//! M2.H B1 — Reduced 3-op chain correctness probe.
//!
//! Hypothesis: M2.H 14-node single graph (`microbench_qnn_qwen_layer.rs`) shows
//! `max_abs_err = 1.37` and 6th attempt's 2-graph split shows the intermediate
//! `q_rope` host buffer is all-zeros. Both observations are consistent with
//! the SDK never forcing NATIVE intermediates to a host-visible buffer between
//! ops in the same graph. P0 (`disableMemoryOptimizations` +
//! `disableNodeOptimizations=1`) had no effect.
//!
//! This probe builds the smallest possible chain that still exercises:
//!   (1) NATIVE intermediate consumed by the next op (RmsNorm.y → MatMul.y)
//!   (2) NATIVE rank-reshape across nodes (MatMul output [1, n_head, head_dim]
//!       feeds RoPE which expects rank 3)
//!   (3) The `OP_INPUT_READWRITE` (RoPE) endpoint — the kernel mutates
//!       `inputs[0]` in place and the graph endpoint is RoPE's output edge.
//!
//! Topology:
//!     x_in  (APP_WRITE [1, dim])
//!       │
//!       ▼ RmsNorm(x_in, w_norm)         → y1     (NATIVE rank 2 [1, dim])
//!       │
//!       ▼ MatMulQ40F32(W_q, y1)         → q      (NATIVE rank 3 [1, n_head, head_dim])
//!       │
//!       ▼ RoPE(q)                       → q_rot  (APP_READ rank 3) ← endpoint
//!
//! Verdict bands (3 ops, identical kernels):
//!   `< 1e-3`            — chain composition is correct           → GREEN
//!   `[1e-3, 1e-1)`      — minor SDK-introduced drift             → YELLOW
//!   `>= 1e-1`           — fundamental chain-composition failure  → RED
//!
//! Interpretation:
//!   GREEN  → 14-node failure lives downstream (KvScatter / FlashAttn / O proj
//!            / FFN). The 3-op composition primitive is not the root cause.
//!   YELLOW → SDK applies a small NATIVE-buffer transform between ops; root
//!            cause is in the SDK's internal layout normalisation but small
//!            chains tolerate it.
//!   RED    → All chain regimes are intrinsically broken. OpPackage path has
//!            to be reconsidered (no chain composition). M2.H path is dead.
//!
//! Build (Android cross):
//!   cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!     --bin microbench_qnn_oppkg_reduced3_correct
//!
//! Pre-deploy:
//!   cargo build --release -p qnn_oppkg --target aarch64-linux-android
//!   adb push target/aarch64-linux-android/release/libqnn_oppkg.so /data/local/tmp/
//!
//! Run on device:
//!   adb shell "LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:/vendor/lib64 \
//!              /data/local/tmp/microbench_qnn_oppkg_reduced3_correct"

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_oppkg_reduced3_correct requires --features qnn");
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
const EPS: f32 = 1e-5;

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
    use std::time::Instant;

    use qnn::*;

    const PKG_PATH: &str = "/data/local/tmp/libqnn_oppkg.so";
    const PKG_PROVIDER: &str = "QnnOpPackage_InitInterface";
    const PKG_TARGET: &str = "GPU_QTI_AISW";

    println!("=== microbench_qnn_oppkg_reduced3_correct (M2.H B1) ===\n");
    println!("Op Package:       {}", PKG_PATH);
    println!("Interface symbol: {}", PKG_PROVIDER);
    println!("Chain topology:   RmsNorm -> MatMulQ40F32 -> RoPE (single graph, 3 nodes)");
    println!("Endpoint:         RoPE.outputs[0] = q_rot (APP_READ)");
    println!(
        "Hypothesis test:  Does the SDK propagate NATIVE intermediates between\n\
         \t\t  ops in a small chain?\n"
    );

    let backend_lib_path = std::env::var("QNN_BACKEND_LIB")
        .unwrap_or_else(|_| "/data/local/tmp/qnn/libQnnGpu.so".to_string());
    println!("Backend lib: {}\n", backend_lib_path);

    // Match qwen_layer dims so a GREEN/RED here is comparable to the 14-node test.
    let dim: usize = 1536;
    let n_head: usize = 12;
    let head_dim: usize = 128;
    let q_proj_out: usize = n_head * head_dim;
    let theta: f32 = 10000.0;
    let start_pos: i32 = 0;

    // ── Build raw-OpenCL reference kernels ───────────────────────────────────
    use ocl::{Context, Device, Platform, Program, Queue};

    let platform = Platform::default();
    let device = Device::first(platform)?;
    let cl_ctx = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let cl_q = Queue::new(&cl_ctx, device, None)?;

    let simple_src = include_str!("../kernels/simple_ops.cl");
    let q40_src = include_str!("../kernels/mul_mv_q4_0_f32_8x_flat.cl");
    let cl_opts = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";

    let prog_simple = Program::builder()
        .devices(device)
        .src(simple_src)
        .cmplr_opt(cl_opts)
        .build(&cl_ctx)?;
    let prog_q40 = Program::builder()
        .devices(device)
        .src(q40_src)
        .cmplr_opt(cl_opts)
        .build(&cl_ctx)?;

    let k_rms = ocl::core::create_kernel(&prog_simple, "kernel_rms_norm_simple")?;
    let k_q40 = ocl::core::create_kernel(&prog_q40, "kernel_mul_mat_q4_0_f32_8x_flat")?;
    let k_rope = ocl::core::create_kernel(&prog_simple, "kernel_rope_simple")?;

    // ── QNN-GPU backend + register OpPackage ─────────────────────────────────
    let gpu_lib = unsafe { Library::new(&backend_lib_path) }?;
    type GetProvidersFn = unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let gp: Symbol<GetProvidersFn> = unsafe { gpu_lib.get(b"QnnInterface_getProviders\0")? };
    let mut provs: *mut *const QnnInterface_t = ptr::null_mut();
    let mut np: c_uint = 0;
    let err = unsafe { gp(&mut provs, &mut np) };
    anyhow::ensure!(err == 0 && np > 0, "GPU getProviders err=0x{:x}", err);
    let v = unsafe { (**provs).__bindgen_anon_1.v2_25 };

    let mut logger: Qnn_LogHandle_t = ptr::null_mut();
    if let Some(log_create) = v.logCreate {
        let err = unsafe { log_create(None, QnnLog_Level_t_QNN_LOG_LEVEL_ERROR, &mut logger) };
        if err != 0 {
            eprintln!("logCreate err=0x{:x} (proceeding without logger)", err);
            logger = ptr::null_mut();
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
    println!("  OK\n");

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

    println!(
        "--- (dim={}, n_head={}, head_dim={}, q_proj_out={}) ---",
        dim, n_head, head_dim, q_proj_out
    );

    let t_total0 = Instant::now();
    let result = run_chain(
        &v,
        ctx,
        &cl_q,
        &cl_ctx,
        &k_rms,
        &k_q40,
        &k_rope,
        &rpcmem_alloc,
        &rpcmem_to_fd,
        RPCMEM_HEAP_ID_SYSTEM,
        RPCMEM_DEFAULT_FLAGS,
        dim,
        n_head,
        head_dim,
        q_proj_out,
        start_pos,
        theta,
    );
    let total_ms = t_total0.elapsed().as_secs_f64() * 1000.0;

    let pass = match result {
        Ok((max_err, finalize_ms)) => {
            let pass_thresh = 1e-3_f32;
            let yellow_lo = pass_thresh;
            let red_lo = 1e-1_f32;
            let band = if max_err < yellow_lo {
                "GREEN-band (chain composition correct)"
            } else if max_err < red_lo {
                "YELLOW-band (small SDK drift, <1e-1)"
            } else {
                "RED-band (chain composition broken)"
            };
            let pass = max_err < pass_thresh;
            println!("  graphFinalize    = {:.2} ms", finalize_ms);
            println!("  total run time   = {:.2} ms", total_ms);
            println!(
                "  max_abs_err      = {:.6e}  [{}]  {}",
                max_err,
                band,
                if pass { "PASS" } else { "FAIL" }
            );
            pass
        }
        Err(e) => {
            println!("  ERROR: {}", e);
            false
        }
    };

    println!(
        "\n=== M2.H B1 verdict: {} ===",
        if pass { "GREEN" } else { "RED-or-YELLOW" }
    );
    println!(
        "Implication:\n\
         \tGREEN  -> 14-node failure is downstream of (RmsNorm, Q proj, RoPE).\n\
         \t          Bisect by adding ops one at a time starting from KvScatter.\n\
         \tYELLOW -> small chain drift, propagates additively to 14 nodes\n\
         \t          (~14/3 = 4.7x). Investigate SDK NATIVE-buffer normalisation.\n\
         \tRED    -> all chain regimes broken. Drop OpPackage chain composition;\n\
         \t          use 1-graph-per-op or M1 path (no graph composition)."
    );

    unsafe {
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(be);
    }

    if pass { Ok(()) } else { std::process::exit(1) }
}

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
#[allow(clippy::too_many_arguments)]
fn run_chain(
    v: &qnn::QnnInterface_ImplementationV2_25_t,
    ctx: qnn::Qnn_ContextHandle_t,
    cl_q: &ocl::Queue,
    cl_ctx: &ocl::Context,
    k_rms: &ocl::core::Kernel,
    k_q40: &ocl::core::Kernel,
    k_rope: &ocl::core::Kernel,
    rpcmem_alloc: &libloading::Symbol<unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void>,
    rpcmem_to_fd: &libloading::Symbol<unsafe extern "C" fn(*const std::ffi::c_void) -> i32>,
    heap_id: i32,
    flags: u32,
    dim: usize,
    n_head: usize,
    head_dim: usize,
    q_proj_out: usize,
    start_pos: i32,
    theta: f32,
) -> anyhow::Result<(f32, f64)> {
    use ocl::core::ArgVal;
    use qnn::*;
    use std::ffi::CString;
    use std::os::raw::c_char;
    use std::ptr;
    use std::time::Instant;

    anyhow::ensure!(dim.is_multiple_of(QK4_0), "dim must be a multiple of 32");
    anyhow::ensure!(q_proj_out == n_head * head_dim, "q_proj_out mismatch");
    anyhow::ensure!(head_dim % 2 == 0, "head_dim must be even (RoPE pairs)");

    // ── Generate identical inputs for both paths ─────────────────────────────
    let mut host_x = vec![0.0f32; dim];
    for (i, x) in host_x.iter_mut().enumerate() {
        *x = (((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5) * 4.0;
    }

    let mut host_rms_w = vec![0.0f32; dim];
    for (i, w) in host_rms_w.iter_mut().enumerate() {
        *w = ((i as f32) * 0.0107 + 0.19).rem_euclid(1.0) * 0.4 + 0.8;
    }

    fn gen_weights(n: usize, k: usize, seed: f32) -> Vec<f32> {
        let mut w = vec![0.0f32; n * k];
        for (i, x) in w.iter_mut().enumerate() {
            *x = (((i as f32) * 0.000_31 + seed).rem_euclid(1.0) - 0.5) * 0.1;
        }
        w
    }
    let w_q = gen_weights(q_proj_out, dim, 0.13);
    let (qq_q, qq_d) = pack_q40_soa(&w_q, q_proj_out, dim);

    // ── Path A: raw OpenCL reference (3 sequential kernels) ──────────────────
    let buf_x = unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, dim, None)?
    };
    let buf_rms_w = unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, dim, None)?
    };
    let buf_y1 = unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_WRITE, dim, None)?
    };
    let buf_qq = unsafe {
        ocl::core::create_buffer::<_, u8>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_ONLY,
            qq_q.len(),
            None,
        )?
    };
    let buf_qd = unsafe {
        ocl::core::create_buffer::<_, u16>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_ONLY,
            qq_d.len(),
            None,
        )?
    };
    let buf_q = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_WRITE,
            q_proj_out,
            None,
        )?
    };

    unsafe {
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_x,
            true,
            0,
            &host_x,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_rms_w,
            true,
            0,
            &host_rms_w,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_qq,
            true,
            0,
            &qq_q,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_qd,
            true,
            0,
            &qq_d,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stage 1: RmsNorm
    let dim_i: i32 = dim as i32;
    ocl::core::set_kernel_arg(k_rms, 0, ArgVal::mem(&buf_x))?;
    ocl::core::set_kernel_arg(k_rms, 1, ArgVal::mem(&buf_rms_w))?;
    ocl::core::set_kernel_arg(k_rms, 2, ArgVal::mem(&buf_y1))?;
    ocl::core::set_kernel_arg(k_rms, 3, ArgVal::scalar(&dim_i))?;
    ocl::core::set_kernel_arg(k_rms, 4, ArgVal::scalar(&EPS))?;
    let global_rms = [1usize, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_rms,
            1,
            None,
            &global_rms,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stage 2: Q proj — Q4_0 matmul (M=1, N=q_proj_out, K=dim)
    let m: usize = 1;
    let n: usize = q_proj_out;
    let k_: usize = dim;
    let ne00 = k_ as i32;
    let ne01 = n as i32;
    let ne02: i32 = 1;
    let ne10 = k_ as i32;
    let ne12: i32 = 1;
    let ne0 = n as i32;
    let ne1 = m as i32;
    let r2: i32 = 1;
    let r3: i32 = 1;
    let off1: u64 = 0;
    let offd: u64 = 0;
    ocl::core::set_kernel_arg(k_q40, 0, ArgVal::mem(&buf_qq))?;
    ocl::core::set_kernel_arg(k_q40, 1, ArgVal::mem(&buf_qd))?;
    ocl::core::set_kernel_arg(k_q40, 2, ArgVal::mem(&buf_y1))?;
    ocl::core::set_kernel_arg(k_q40, 3, ArgVal::scalar(&off1))?;
    ocl::core::set_kernel_arg(k_q40, 4, ArgVal::mem(&buf_q))?;
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
    let global_q40 = [n.div_ceil(8) * 64, 1, 1];
    let local_q40 = [64usize, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_q40,
            3,
            None,
            &global_q40,
            Some(local_q40),
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stage 3: RoPE(Q) — kernel_rope_simple(x, head_dim, num_heads, seq_len, start_pos, theta)
    let head_dim_i: i32 = head_dim as i32;
    let n_head_i: i32 = n_head as i32;
    let seq_len_i: i32 = 1;
    ocl::core::set_kernel_arg(k_rope, 0, ArgVal::mem(&buf_q))?;
    ocl::core::set_kernel_arg(k_rope, 1, ArgVal::scalar(&head_dim_i))?;
    ocl::core::set_kernel_arg(k_rope, 2, ArgVal::scalar(&n_head_i))?;
    ocl::core::set_kernel_arg(k_rope, 3, ArgVal::scalar(&seq_len_i))?;
    ocl::core::set_kernel_arg(k_rope, 4, ArgVal::scalar(&start_pos))?;
    ocl::core::set_kernel_arg(k_rope, 5, ArgVal::scalar(&theta))?;
    let pairs_q = n_head * (head_dim / 2);
    let global_rope = [pairs_q, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_rope,
            1,
            None,
            &global_rope,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    let mut ref_q_rot = vec![0.0f32; q_proj_out];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_q,
            true,
            0,
            &mut ref_q_rot,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // ── Path B: QNN single graph with 3 chained ops ──────────────────────────
    let bytes_x = (dim * 4) as i32;
    let bytes_rms_w = (dim * 4) as i32;
    let bytes_qq = qq_q.len() as i32;
    let bytes_qd = (qq_d.len() * 2) as i32;
    let bytes_qrot = (q_proj_out * 4) as i32;

    let rpc_x = unsafe { rpcmem_alloc(heap_id, flags, bytes_x) };
    let rpc_rms_w = unsafe { rpcmem_alloc(heap_id, flags, bytes_rms_w) };
    let rpc_qq = unsafe { rpcmem_alloc(heap_id, flags, bytes_qq) };
    let rpc_qd = unsafe { rpcmem_alloc(heap_id, flags, bytes_qd) };
    let rpc_qrot = unsafe { rpcmem_alloc(heap_id, flags, bytes_qrot) };
    anyhow::ensure!(
        !rpc_x.is_null()
            && !rpc_rms_w.is_null()
            && !rpc_qq.is_null()
            && !rpc_qd.is_null()
            && !rpc_qrot.is_null(),
        "rpcmem_alloc failed"
    );

    unsafe {
        std::ptr::copy_nonoverlapping(
            host_x.as_ptr() as *const u8,
            rpc_x as *mut u8,
            bytes_x as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_rms_w.as_ptr() as *const u8,
            rpc_rms_w as *mut u8,
            bytes_rms_w as usize,
        );
        std::ptr::copy_nonoverlapping(qq_q.as_ptr(), rpc_qq as *mut u8, bytes_qq as usize);
        std::ptr::copy_nonoverlapping(
            qq_d.as_ptr() as *const u8,
            rpc_qd as *mut u8,
            bytes_qd as usize,
        );
        // Zero the output buffer so we can detect "SDK never wrote it".
        std::ptr::write_bytes(rpc_qrot as *mut u8, 0u8, bytes_qrot as usize);
    }

    let fd_x = unsafe { rpcmem_to_fd(rpc_x) };
    let fd_rms_w = unsafe { rpcmem_to_fd(rpc_rms_w) };
    let fd_qq = unsafe { rpcmem_to_fd(rpc_qq) };
    let fd_qd = unsafe { rpcmem_to_fd(rpc_qd) };
    let fd_qrot = unsafe { rpcmem_to_fd(rpc_qrot) };

    // Tensor dims. Match the rank pattern proven for matmul output reshape in
    // M2.H 3rd attempt (matmul output reshapes to rank-3 directly).
    let mut dims_x: Vec<u32> = vec![1u32, dim as u32];
    let mut dims_rms_w: Vec<u32> = vec![dim as u32];
    let mut dims_y1: Vec<u32> = vec![1u32, dim as u32];
    let mut dims_qq: Vec<u32> = vec![qq_q.len() as u32];
    let mut dims_qd: Vec<u32> = vec![qq_d.len() as u32];
    // q output of MatMul reshapes directly to rank-3 attention view.
    let mut dims_q: Vec<u32> = vec![1u32, n_head as u32, head_dim as u32];
    let mut dims_qrot: Vec<u32> = vec![1u32, n_head as u32, head_dim as u32];

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

    let nm_x = CString::new("b1_x").unwrap();
    let nm_rms_w = CString::new("b1_rms_w").unwrap();
    let nm_y1 = CString::new("b1_y1").unwrap();
    let nm_qq = CString::new("b1_qq").unwrap();
    let nm_qd = CString::new("b1_qd").unwrap();
    let nm_q = CString::new("b1_q").unwrap();
    let nm_qrot = CString::new("b1_q_rot").unwrap();

    let app_w = Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE;
    let app_r = Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ;
    let native = Qnn_TensorType_t_QNN_TENSOR_TYPE_NATIVE;
    let f32_t = Qnn_DataType_t_QNN_DATATYPE_FLOAT_32;
    let f16_t = Qnn_DataType_t_QNN_DATATYPE_FLOAT_16;
    let u8_t = Qnn_DataType_t_QNN_DATATYPE_UINT_8;

    let build = |ttype, dtype, rank: u32, dims_ptr: *mut u32, name: *const c_char| {
        let mut t = Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_tv1(ttype, dtype, rank, dims_ptr),
            },
        };
        t.__bindgen_anon_1.v1.name = name;
        t
    };

    // Endpoints: x, rms_w (APP_WRITE) and q_rot (APP_READ).
    // Intermediates: y1, q (NATIVE).
    // Weights: qq, qd (APP_WRITE).
    let mut t_x = build(app_w, f32_t, 2, dims_x.as_mut_ptr(), nm_x.as_ptr());
    let mut t_rms_w = build(app_w, f32_t, 1, dims_rms_w.as_mut_ptr(), nm_rms_w.as_ptr());
    let mut t_qq = build(app_w, u8_t, 1, dims_qq.as_mut_ptr(), nm_qq.as_ptr());
    let mut t_qd = build(app_w, f16_t, 1, dims_qd.as_mut_ptr(), nm_qd.as_ptr());
    let mut t_y1 = build(native, f32_t, 2, dims_y1.as_mut_ptr(), nm_y1.as_ptr());
    let mut t_q = build(native, f32_t, 3, dims_q.as_mut_ptr(), nm_q.as_ptr());
    let mut t_qrot = build(app_r, f32_t, 3, dims_qrot.as_mut_ptr(), nm_qrot.as_ptr());

    // ── Build graph ──────────────────────────────────────────────────────────
    let g_name = CString::new("b1_reduced3_graph").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    for (label, t) in [
        ("x", &mut t_x),
        ("rms_w", &mut t_rms_w),
        ("qq", &mut t_qq),
        ("qd", &mut t_qd),
        ("y1", &mut t_y1),
        ("q", &mut t_q),
        ("q_rot", &mut t_qrot),
    ] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", label, err);
    }

    // RoPE params.
    let pn_start_pos = CString::new("start_pos").unwrap();
    let pn_theta = CString::new("theta").unwrap();
    let mut rope_params = [
        Qnn_Param_t {
            paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            name: pn_start_pos.as_ptr(),
            __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                scalarParam: Qnn_Scalar_t {
                    dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                    __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                        int32Value: start_pos,
                    },
                },
            },
        },
        Qnn_Param_t {
            paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            name: pn_theta.as_ptr(),
            __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                scalarParam: Qnn_Scalar_t {
                    dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                    __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 { floatValue: theta },
                },
            },
        },
    ];

    let pkg = CString::new("qnn_oppkg").unwrap();
    let on_rms = CString::new("rms_b1").unwrap();
    let on_q_proj = CString::new("q_proj_b1").unwrap();
    let on_rope = CString::new("rope_b1").unwrap();
    let ot_rms = CString::new("CustomRmsNorm").unwrap();
    let ot_q40 = CString::new("CustomMatMulQ40F32").unwrap();
    let ot_rope = CString::new("CustomRope").unwrap();

    let mut in_rms = [t_x, t_rms_w];
    let mut out_rms = [t_y1];
    let mut in_qproj = [t_qq, t_qd, t_y1];
    let mut out_qproj = [t_q];
    let mut in_rope = [t_q];
    let mut out_rope = [t_qrot];

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
            "RmsNorm",
        ),
        (
            on_q_proj.as_ptr(),
            ot_q40.as_ptr(),
            in_qproj.as_mut_ptr(),
            3,
            out_qproj.as_mut_ptr(),
            1,
            ptr::null_mut(),
            0,
            "Q proj (Q4_0 matmul)",
        ),
        (
            on_rope.as_ptr(),
            ot_rope.as_ptr(),
            in_rope.as_mut_ptr(),
            1,
            out_rope.as_mut_ptr(),
            1,
            rope_params.as_mut_ptr(),
            2,
            "RoPE",
        ),
    ];
    for (nm, ty, ins, n_in, outs, n_out, params, n_params, label) in nodes {
        let op = make_op(nm, ty, ins, n_in, outs, n_out, params, n_params);
        let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
        anyhow::ensure!(err == 0, "graphAddNode({}) err=0x{:x}", label, err);
    }

    let t_fin0 = Instant::now();
    let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
    let finalize_ms = t_fin0.elapsed().as_secs_f64() * 1000.0;
    anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);

    // ── memRegister 5 host-backed tensors (intermediates y1/q are NATIVE) ────
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
    let descs = [
        mk_desc(fd_x, rpc_x, f32_t, &dims_x),
        mk_desc(fd_rms_w, rpc_rms_w, f32_t, &dims_rms_w),
        mk_desc(fd_qq, rpc_qq, u8_t, &dims_qq),
        mk_desc(fd_qd, rpc_qd, f16_t, &dims_qd),
        mk_desc(fd_qrot, rpc_qrot, f32_t, &dims_qrot),
    ];
    let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 5];
    let err = unsafe {
        (v.memRegister.unwrap())(ctx, descs.as_ptr(), descs.len() as u32, mh.as_mut_ptr())
    };
    anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);

    let set_mh = |t: &mut Qnn_Tensor_t, h: *mut std::ffi::c_void| {
        t.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        t.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = h;
    };
    let mut t_x_mh = t_x;
    set_mh(&mut t_x_mh, mh[0]);
    let mut t_rms_w_mh = t_rms_w;
    set_mh(&mut t_rms_w_mh, mh[1]);
    let mut t_qq_mh = t_qq;
    set_mh(&mut t_qq_mh, mh[2]);
    let mut t_qd_mh = t_qd;
    set_mh(&mut t_qd_mh, mh[3]);
    let mut t_qrot_mh = t_qrot;
    set_mh(&mut t_qrot_mh, mh[4]);

    let exec_inputs = [t_x_mh, t_rms_w_mh, t_qq_mh, t_qd_mh];
    let mut exec_outputs = [t_qrot_mh];

    let err = unsafe {
        (v.graphExecute.unwrap())(
            graph,
            exec_inputs.as_ptr(),
            exec_inputs.len() as u32,
            exec_outputs.as_mut_ptr(),
            exec_outputs.len() as u32,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);

    // ── Diagnostic + compare ─────────────────────────────────────────────────
    let qnn_qrot = unsafe { std::slice::from_raw_parts(rpc_qrot as *const f32, q_proj_out) };

    let nz_qnn = qnn_qrot.iter().filter(|&&x| x != 0.0).count();
    let nz_ref = ref_q_rot.iter().filter(|&&x| x != 0.0).count();
    eprintln!(
        "[diag] q_rot nonzero count: qnn={} ref={} total={}",
        nz_qnn, nz_ref, q_proj_out
    );
    eprintln!("[diag] ref_q_rot[0..8] = {:?}", &ref_q_rot[..8]);
    eprintln!("[diag] qnn_q_rot[0..8] = {:?}", &qnn_qrot[..8]);

    let mut max_abs = 0.0f32;
    for i in 0..q_proj_out {
        let d = (qnn_qrot[i] - ref_q_rot[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }

    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), mh.len() as u32);
    }

    Ok((max_abs, finalize_ms))
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
