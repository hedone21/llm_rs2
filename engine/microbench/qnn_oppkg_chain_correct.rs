//! M1.9 chain correctness — production OpPackage multi-node graph vs raw OpenCL.
//!
//! Chains the four out-of-place ops in a single QNN graph and compares against
//! four raw OpenCL kernel invocations (the same kernels the OpPackage embeds):
//!
//!   x ── RmsNorm ── y1 ── MatMul ── y2 ── Add ── y3 ── Softmax ── y4
//!         ▲                  ▲             ▲
//!       rms_w            W_matmul         bias
//!
//! Why **four** ops, not five:
//!
//! `CustomSiluMul` is **in-place** — its kernel mutates `inputs[0]` via
//! `OP_INPUT_READWRITE` and never writes to `outputs[0]`. The output tensor
//! exists only to satisfy the OpPackage's "claim the last mem_object" contract
//! (`build_op_state` lines 211-214). At graph level this means SiluMul's
//! `outputs[0]` slot carries no kernel-produced data; M1.7 reads its result by
//! aliasing `inputs[0]` and `outputs[0]` to the same fd at host-visible
//! `APP_READ`/`APP_WRITE` registration time.
//!
//! That trick only works when SiluMul is the **only** node — a single-graph
//! integration is impossible without driver-side support:
//!
//! * Placing SiluMul as the chain's last node forces its input (e.g.
//!   `Softmax`'s output, `chain_y4`) to be `APP_WRITE` (host-visible) so
//!   SiluMul's host-side fd alias works. But `APP_WRITE` is the input role;
//!   QNN-GPU rejects it as an output slot:
//!   `GPU_ERROR_INVALID_TYPE(10012) - Invalid OpConfig output tensor type for
//!   tensor: chain_y4` (logcat-confirmed).
//! * Placing SiluMul mid-chain leaves its `outputs[0]` driver-managed
//!   (`NATIVE`); the next node would then read garbage because the kernel
//!   never writes that slot.
//!
//! Either way, integrating an in-place op into a multi-node QNN graph requires
//! either (a) a host-side post-pass to copy `inputs[0]` → `outputs[0]`, or
//! (b) refactoring the OpPackage to be out-of-place. Both touch production
//! code and are out of scope for M1.9.
//!
//! M1.9 therefore validates compounded correctness on the four out-of-place
//! ops as a single graph; SiluMul's correctness is already covered by M1.7
//! (max_abs_err = 0 standalone).
//!
//! Shapes (rank-2 throughout to satisfy MatMul's `rank>=2` constraint):
//!   x          [1, dim]   FLOAT_32  (graph input, APP_WRITE)
//!   rms_w      [1, dim]   FLOAT_32  (graph input, APP_WRITE; rank-2 broadcast)
//!   W_matmul   [dim, dim] FLOAT_16  (graph input, APP_WRITE)
//!   bias       [1, dim]   FLOAT_32  (graph input, APP_WRITE)
//!   y1, y2, y3 [1, dim]   FLOAT_32  (NATIVE intermediates)
//!   y4         [1, dim]   FLOAT_32  (graph output, APP_READ)
//!
//! `dim = 256` mirrors M1.4's first matmul case (M=1, N=256, K=256). All
//! tensors keep the same flat element count, satisfying the float4
//! vectorisation constraint (`total % 4 == 0`) for every op.
//!
//! Reference path (raw OpenCL, identical kernel sources):
//!   1. kernel_rms_norm_simple(x, rms_w, t1, dim, eps)
//!   2. kernel_mul_mat_f16_f32(W_matmul, t1, t2, ...)
//!   3. kernel_add_row(t2, bias, t3, n4)
//!   4. kernel_softmax_simple(t3, t4, dim)
//!
//! Pass criterion: max_abs_err < 1e-3 (compounded tolerance). Bands:
//!   `< 1e-5`            — accumulated FP rounding only         → GREEN
//!   `[1e-5, 1e-3)`      — possible graph topology drift        → YELLOW
//!   `>= 1e-3`           — chain mapping bug                    → RED
//!
//! Build (Android cross):
//!   cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!     --bin microbench_qnn_oppkg_chain_correct
//!
//! Pre-deploy:
//!   cargo build --release -p qnn_oppkg --target aarch64-linux-android
//!   adb push target/aarch64-linux-android/release/libqnn_oppkg.so /data/local/tmp/
//!
//! Run on device:
//!   adb shell "LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:/vendor/lib64 \
//!              /data/local/tmp/microbench_qnn_oppkg_chain_correct"

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_oppkg_chain_correct requires --features qnn");
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
fn main() -> anyhow::Result<()> {
    use libloading::{Library, Symbol};
    use std::ffi::{CString, c_void};
    use std::os::raw::c_uint;
    use std::ptr;

    use qnn::*;

    const PKG_PATH: &str = "/data/local/tmp/libqnn_oppkg.so";
    const PKG_PROVIDER: &str = "QnnOpPackage_InitInterface";
    const PKG_TARGET: &str = "GPU_QTI_AISW";

    println!("=== microbench_qnn_oppkg_chain_correct (M1.9) ===\n");
    println!("Op Package:       {}", PKG_PATH);
    println!("Interface symbol: {}", PKG_PROVIDER);
    println!("Chain topology:   RmsNorm -> MatMulF16F32 -> Add -> Softmax (single graph)");
    println!(
        "Note:             SiluMul (in-place, M1.7) cannot be chained without graph-level\n\
         \x20                 alias support. Validated standalone in M1.7 (max_abs_err = 0).\n"
    );

    let backend_lib_path = std::env::var("QNN_BACKEND_LIB")
        .unwrap_or_else(|_| "/data/local/tmp/qnn/libQnnGpu.so".to_string());
    println!("Backend lib: {}\n", backend_lib_path);

    // ── Build raw-OpenCL reference toolchain (4 separate kernels) ─────────────
    use ocl::{Context, Device, Platform, Program, Queue};

    let platform = Platform::default();
    let device = Device::first(platform)?;
    let cl_ctx = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let cl_q = Queue::new(&cl_ctx, device, None)?;

    let simple_src = include_str!("../kernels/simple_ops.cl");
    let add_src = include_str!("../kernels/add.cl");
    let matmul_src = include_str!("../kernels/mul_mv_f16_f32.cl");
    let cl_opts = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";

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
    let prog_matmul = Program::builder()
        .devices(device)
        .src(matmul_src)
        .cmplr_opt(cl_opts)
        .build(&cl_ctx)?;

    let k_rms = ocl::core::create_kernel(&prog_simple, "kernel_rms_norm_simple")?;
    let k_matmul = ocl::core::create_kernel(&prog_matmul, "kernel_mul_mat_f16_f32")?;
    let k_add = ocl::core::create_kernel(&prog_add, "kernel_add_row")?;
    let k_softmax = ocl::core::create_kernel(&prog_simple, "kernel_softmax_simple")?;

    // ── QNN-GPU backend + register OpPackage ──────────────────────────────────
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

    let dim: usize = 256;
    println!("--- dim = {} (rows=1 throughout) ---", dim);

    let result = run_chain(
        &v,
        ctx,
        &cl_q,
        &cl_ctx,
        &k_rms,
        &k_matmul,
        &k_add,
        &k_softmax,
        &rpcmem_alloc,
        &rpcmem_to_fd,
        RPCMEM_HEAP_ID_SYSTEM,
        RPCMEM_DEFAULT_FLAGS,
        dim,
    );

    let pass = match result {
        Ok(max_err) => {
            let pass_thresh = 1e-3_f32;
            let yellow_lo = 1e-5_f32;
            let band = if max_err < yellow_lo {
                "GREEN-band (FP rounding only)"
            } else if max_err < pass_thresh {
                "YELLOW-band (graph drift, <1e-3)"
            } else {
                "RED-band"
            };
            let pass = max_err < pass_thresh;
            println!(
                "  max_abs_err = {:.6e}  [{}]  {}",
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
        "\n=== M1.9 verdict: {} ===",
        if pass { "GREEN" } else { "RED" }
    );

    unsafe {
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(be);
    }

    if pass { Ok(()) } else { std::process::exit(1) }
}

#[cfg(feature = "qnn")]
#[allow(clippy::too_many_arguments)]
fn run_chain(
    v: &qnn::QnnInterface_ImplementationV2_25_t,
    ctx: qnn::Qnn_ContextHandle_t,
    cl_q: &ocl::Queue,
    cl_ctx: &ocl::Context,
    k_rms: &ocl::core::Kernel,
    k_matmul: &ocl::core::Kernel,
    k_add: &ocl::core::Kernel,
    k_softmax: &ocl::core::Kernel,
    rpcmem_alloc: &libloading::Symbol<unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void>,
    rpcmem_to_fd: &libloading::Symbol<unsafe extern "C" fn(*const std::ffi::c_void) -> i32>,
    heap_id: i32,
    flags: u32,
    dim: usize,
) -> anyhow::Result<f32> {
    use ocl::core::ArgVal;
    use qnn::*;
    use std::ffi::CString;
    use std::os::raw::c_char;
    use std::ptr;

    // ── Generate identical inputs for both paths ──────────────────────────────
    let total = dim;
    anyhow::ensure!(total % 4 == 0, "dim must be multiple of 4");
    let n4: i32 = (total / 4) as i32;

    let mut host_x = vec![0.0f32; total];
    let mut host_rms_w = vec![0.0f32; dim];
    let mut host_w_f16 = vec![0u16; dim * dim];
    let mut host_bias = vec![0.0f32; total];
    for i in 0..total {
        host_x[i] = ((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5;
        host_bias[i] = ((i as f32) * 0.0061 + 0.21).rem_euclid(1.0) * 0.2 - 0.1;
    }
    for (i, w) in host_rms_w.iter_mut().enumerate() {
        *w = ((i as f32) * 0.0091 + 0.13).rem_euclid(1.0) * 0.5 + 0.5;
    }
    for (i, w) in host_w_f16.iter_mut().enumerate() {
        let val = ((i as f32) * 0.0007 + 0.13).rem_euclid(1.0) - 0.5;
        *w = f32_to_f16_bits(val);
    }

    // ── Path A: raw OpenCL — 4 sequential kernel launches ─────────────────────
    let buf_x = unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, total, None)?
    };
    let buf_rms_w = unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, dim, None)?
    };
    let buf_w = unsafe {
        ocl::core::create_buffer::<_, u16>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_ONLY,
            dim * dim,
            None,
        )?
    };
    let buf_bias = unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, total, None)?
    };
    let buf_t1 = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_WRITE,
            total,
            None,
        )?
    };
    let buf_t2 = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_WRITE,
            total,
            None,
        )?
    };
    let buf_t3 = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_WRITE,
            total,
            None,
        )?
    };
    let buf_t4 = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_WRITE,
            total,
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
            &buf_w,
            true,
            0,
            &host_w_f16,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_bias,
            true,
            0,
            &host_bias,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stage 1: RmsNorm — kernel_rms_norm_simple(x, weight, output, dim, eps)
    let dim_i: i32 = dim as i32;
    ocl::core::set_kernel_arg(k_rms, 0, ArgVal::mem(&buf_x))?;
    ocl::core::set_kernel_arg(k_rms, 1, ArgVal::mem(&buf_rms_w))?;
    ocl::core::set_kernel_arg(k_rms, 2, ArgVal::mem(&buf_t1))?;
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

    // Stage 2: MatMul — kernel_mul_mat_f16_f32; M=1, N=dim, K=dim.
    let m: usize = 1;
    let n: usize = dim;
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
    let off0: u64 = 0;
    let off1: u64 = 0;
    let offd: u64 = 0;
    ocl::core::set_kernel_arg(k_matmul, 0, ArgVal::mem(&buf_w))?;
    ocl::core::set_kernel_arg(k_matmul, 1, ArgVal::scalar(&off0))?;
    ocl::core::set_kernel_arg(k_matmul, 2, ArgVal::mem(&buf_t1))?;
    ocl::core::set_kernel_arg(k_matmul, 3, ArgVal::scalar(&off1))?;
    ocl::core::set_kernel_arg(k_matmul, 4, ArgVal::mem(&buf_t2))?;
    ocl::core::set_kernel_arg(k_matmul, 5, ArgVal::scalar(&offd))?;
    ocl::core::set_kernel_arg(k_matmul, 6, ArgVal::scalar(&ne00))?;
    ocl::core::set_kernel_arg(k_matmul, 7, ArgVal::scalar(&ne01))?;
    ocl::core::set_kernel_arg(k_matmul, 8, ArgVal::scalar(&ne02))?;
    ocl::core::set_kernel_arg(k_matmul, 9, ArgVal::scalar(&ne10))?;
    ocl::core::set_kernel_arg(k_matmul, 10, ArgVal::scalar(&ne12))?;
    ocl::core::set_kernel_arg(k_matmul, 11, ArgVal::scalar(&ne0))?;
    ocl::core::set_kernel_arg(k_matmul, 12, ArgVal::scalar(&ne1))?;
    ocl::core::set_kernel_arg(k_matmul, 13, ArgVal::scalar(&r2))?;
    ocl::core::set_kernel_arg(k_matmul, 14, ArgVal::scalar(&r3))?;
    let n_dst: usize = 2;
    let global_mm = [n.div_ceil(n_dst) * 64, m * 4, 1];
    let local_mm = [64usize, 4, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_matmul,
            3,
            None,
            &global_mm,
            Some(local_mm),
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stage 3: Add — kernel_add_row(t2, 0, bias, 0, t3, 0, n4)
    ocl::core::set_kernel_arg(k_add, 0, ArgVal::mem(&buf_t2))?;
    ocl::core::set_kernel_arg(k_add, 1, ArgVal::scalar(&off0))?;
    ocl::core::set_kernel_arg(k_add, 2, ArgVal::mem(&buf_bias))?;
    ocl::core::set_kernel_arg(k_add, 3, ArgVal::scalar(&off1))?;
    ocl::core::set_kernel_arg(k_add, 4, ArgVal::mem(&buf_t3))?;
    ocl::core::set_kernel_arg(k_add, 5, ArgVal::scalar(&offd))?;
    ocl::core::set_kernel_arg(k_add, 6, ArgVal::scalar(&n4))?;
    let global_add = [n4 as usize, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_add,
            1,
            None,
            &global_add,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stage 4: Softmax — kernel_softmax_simple(t3, t4, dim); rows = 1.
    ocl::core::set_kernel_arg(k_softmax, 0, ArgVal::mem(&buf_t3))?;
    ocl::core::set_kernel_arg(k_softmax, 1, ArgVal::mem(&buf_t4))?;
    ocl::core::set_kernel_arg(k_softmax, 2, ArgVal::scalar(&dim_i))?;
    let global_sm = [1usize, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_softmax,
            1,
            None,
            &global_sm,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    let mut ref_y = vec![0.0f32; total];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_t4,
            true,
            0,
            &mut ref_y,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // ── Path B: QNN single graph with 4 chained ops ───────────────────────────
    let bytes_x = (total * 4) as i32;
    let bytes_rms_w = (dim * 4) as i32;
    let bytes_w = (dim * dim * 2) as i32;
    let bytes_bias = (total * 4) as i32;
    let bytes_y4 = (total * 4) as i32;

    let rpc_x = unsafe { rpcmem_alloc(heap_id, flags, bytes_x) };
    let rpc_rms_w = unsafe { rpcmem_alloc(heap_id, flags, bytes_rms_w) };
    let rpc_w = unsafe { rpcmem_alloc(heap_id, flags, bytes_w) };
    let rpc_bias = unsafe { rpcmem_alloc(heap_id, flags, bytes_bias) };
    let rpc_y4 = unsafe { rpcmem_alloc(heap_id, flags, bytes_y4) };
    anyhow::ensure!(
        !rpc_x.is_null()
            && !rpc_rms_w.is_null()
            && !rpc_w.is_null()
            && !rpc_bias.is_null()
            && !rpc_y4.is_null(),
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
        std::ptr::copy_nonoverlapping(
            host_w_f16.as_ptr() as *const u8,
            rpc_w as *mut u8,
            bytes_w as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_bias.as_ptr() as *const u8,
            rpc_bias as *mut u8,
            bytes_bias as usize,
        );
    }

    let fd_x = unsafe { rpcmem_to_fd(rpc_x) };
    let fd_rms_w = unsafe { rpcmem_to_fd(rpc_rms_w) };
    let fd_w = unsafe { rpcmem_to_fd(rpc_w) };
    let fd_bias = unsafe { rpcmem_to_fd(rpc_bias) };
    let fd_y4 = unsafe { rpcmem_to_fd(rpc_y4) };

    let mut dims_x: Vec<u32> = vec![1, dim as u32];
    let mut dims_rms_w: Vec<u32> = vec![1, dim as u32];
    let mut dims_w: Vec<u32> = vec![dim as u32, dim as u32];
    let mut dims_bias: Vec<u32> = vec![1, dim as u32];
    let mut dims_y1: Vec<u32> = vec![1, dim as u32];
    let mut dims_y2: Vec<u32> = vec![1, dim as u32];
    let mut dims_y3: Vec<u32> = vec![1, dim as u32];
    let mut dims_y4: Vec<u32> = vec![1, dim as u32];

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

    let mk_tv1 = |ttype, dtype, dims_ptr: *mut u32, rank: u32| Qnn_TensorV1_t {
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

    let n_x = CString::new("chain_x").unwrap();
    let n_rms_w = CString::new("chain_rms_w").unwrap();
    let n_w = CString::new("chain_W").unwrap();
    let n_bias = CString::new("chain_bias").unwrap();
    let n_y1 = CString::new("chain_y1").unwrap();
    let n_y2 = CString::new("chain_y2").unwrap();
    let n_y3 = CString::new("chain_y3").unwrap();
    let n_y4 = CString::new("chain_y4").unwrap();

    let mut t_x = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                dims_x.as_mut_ptr(),
                2,
            ),
        },
    };
    t_x.__bindgen_anon_1.v1.name = n_x.as_ptr();
    let mut t_rms_w = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                dims_rms_w.as_mut_ptr(),
                2,
            ),
        },
    };
    t_rms_w.__bindgen_anon_1.v1.name = n_rms_w.as_ptr();
    let mut t_w = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
                dims_w.as_mut_ptr(),
                2,
            ),
        },
    };
    t_w.__bindgen_anon_1.v1.name = n_w.as_ptr();
    let mut t_bias = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                dims_bias.as_mut_ptr(),
                2,
            ),
        },
    };
    t_bias.__bindgen_anon_1.v1.name = n_bias.as_ptr();

    let mut t_y1 = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_NATIVE,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                dims_y1.as_mut_ptr(),
                2,
            ),
        },
    };
    t_y1.__bindgen_anon_1.v1.name = n_y1.as_ptr();
    let mut t_y2 = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_NATIVE,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                dims_y2.as_mut_ptr(),
                2,
            ),
        },
    };
    t_y2.__bindgen_anon_1.v1.name = n_y2.as_ptr();
    let mut t_y3 = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_NATIVE,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                dims_y3.as_mut_ptr(),
                2,
            ),
        },
    };
    t_y3.__bindgen_anon_1.v1.name = n_y3.as_ptr();
    let mut t_y4 = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                dims_y4.as_mut_ptr(),
                2,
            ),
        },
    };
    t_y4.__bindgen_anon_1.v1.name = n_y4.as_ptr();

    // ── Build graph ───────────────────────────────────────────────────────────
    let g_name = CString::new("chain_graph_4op").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    for (label, t) in [
        ("x", &mut t_x),
        ("rms_w", &mut t_rms_w),
        ("W", &mut t_w),
        ("bias", &mut t_bias),
        ("y1", &mut t_y1),
        ("y2", &mut t_y2),
        ("y3", &mut t_y3),
        ("y4", &mut t_y4),
    ] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", label, err);
    }

    let pkg = CString::new("qnn_oppkg").unwrap();
    let op_name_rms = CString::new("rms0").unwrap();
    let op_name_mm = CString::new("mm0").unwrap();
    let op_name_add = CString::new("add0").unwrap();
    let op_name_sm = CString::new("sm0").unwrap();
    let ot_rms = CString::new("CustomRmsNorm").unwrap();
    let ot_mm = CString::new("CustomMatMulF16F32").unwrap();
    let ot_add = CString::new("CustomAdd").unwrap();
    let ot_sm = CString::new("CustomSoftmax").unwrap();

    let mut in_rms = [t_x, t_rms_w];
    let mut out_rms = [t_y1];
    let mut in_mm = [t_w, t_y1];
    let mut out_mm = [t_y2];
    let mut in_add = [t_y2, t_bias];
    let mut out_add = [t_y3];
    let mut in_sm = [t_y3];
    let mut out_sm = [t_y4];

    let make_op = |name: *const c_char,
                   typ: *const c_char,
                   ins: *mut Qnn_Tensor_t,
                   n_in: u32,
                   outs: *mut Qnn_Tensor_t,
                   n_out: u32|
     -> Qnn_OpConfig_t {
        Qnn_OpConfig_t {
            version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
            __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
                v1: Qnn_OpConfigV1_t {
                    name,
                    packageName: pkg.as_ptr(),
                    typeName: typ,
                    numOfParams: 0,
                    params: ptr::null_mut(),
                    numOfInputs: n_in,
                    inputTensors: ins,
                    numOfOutputs: n_out,
                    outputTensors: outs,
                },
            },
        }
    };

    let nodes = [
        (
            op_name_rms.as_ptr(),
            ot_rms.as_ptr(),
            in_rms.as_mut_ptr(),
            2u32,
            out_rms.as_mut_ptr(),
            1u32,
            "RmsNorm",
        ),
        (
            op_name_mm.as_ptr(),
            ot_mm.as_ptr(),
            in_mm.as_mut_ptr(),
            2u32,
            out_mm.as_mut_ptr(),
            1u32,
            "MatMul",
        ),
        (
            op_name_add.as_ptr(),
            ot_add.as_ptr(),
            in_add.as_mut_ptr(),
            2u32,
            out_add.as_mut_ptr(),
            1u32,
            "Add",
        ),
        (
            op_name_sm.as_ptr(),
            ot_sm.as_ptr(),
            in_sm.as_mut_ptr(),
            1u32,
            out_sm.as_mut_ptr(),
            1u32,
            "Softmax",
        ),
    ];
    for (nm, ty, ins, n_in, outs, n_out, label) in nodes {
        let op = make_op(nm, ty, ins, n_in, outs, n_out);
        let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
        anyhow::ensure!(err == 0, "graphAddNode({}) err=0x{:x}", label, err);
    }

    let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
    anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);

    // ── memRegister: 5 host-backed tensors (4 inputs + y4 output) ─────────────
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
        mk_desc(fd_x, rpc_x, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_x),
        mk_desc(
            fd_rms_w,
            rpc_rms_w,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            &dims_rms_w,
        ),
        mk_desc(fd_w, rpc_w, Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, &dims_w),
        mk_desc(
            fd_bias,
            rpc_bias,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            &dims_bias,
        ),
        mk_desc(
            fd_y4,
            rpc_y4,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            &dims_y4,
        ),
    ];
    let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 5];
    let err = unsafe { (v.memRegister.unwrap())(ctx, descs.as_ptr(), 5, mh.as_mut_ptr()) };
    anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);

    let set_mh = |t: &mut Qnn_Tensor_t, h: *mut std::ffi::c_void| {
        t.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        t.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = h;
    };
    let mut t_x_mh = t_x;
    set_mh(&mut t_x_mh, mh[0]);
    let mut t_rms_w_mh = t_rms_w;
    set_mh(&mut t_rms_w_mh, mh[1]);
    let mut t_w_mh = t_w;
    set_mh(&mut t_w_mh, mh[2]);
    let mut t_bias_mh = t_bias;
    set_mh(&mut t_bias_mh, mh[3]);
    let mut t_y4_mh = t_y4;
    set_mh(&mut t_y4_mh, mh[4]);

    let exec_inputs = [t_x_mh, t_rms_w_mh, t_w_mh, t_bias_mh];
    let mut exec_outputs = [t_y4_mh];

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

    let mut max_abs = 0.0f32;
    unsafe {
        let test_y = std::slice::from_raw_parts(rpc_y4 as *const f32, total);
        for i in 0..total {
            let d = (test_y[i] - ref_y[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
    }

    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), 5);
    }

    Ok(max_abs)
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
