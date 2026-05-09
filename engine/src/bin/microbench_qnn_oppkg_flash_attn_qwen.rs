//! M2.H standalone replay — Qwen 2.5-1.5B dims (n_head=12, n_head_kv=2,
//! head_dim=128, kv_capacity=2048, n_kv=1, write_pos=0).
//!
//! Goal: replay the chain (sg3) FlashAttn binding pattern in a standalone
//! single-op graph. M2.F (n_head=32) is GREEN; sg3 reports a deterministic
//! `qnn_attn_o = ref × 0.507253` ratio. This bench isolates whether the 0.5
//! ratio is intrinsic to the Qwen dims (n_head=12, n_head_kv=2 with
//! n_kv=1/write_pos=0) or only emerges under chain composition.
//!
//! Pattern: `[1, n_head_kv, capacity, head_dim]` rank-4 KV cache fully
//! host-written — head 0/1 both populated at position 0 only, all other
//! capacity slots zero (mirrors KvScatter at write_pos=0 with n_kv=1).
//!
//! Build:
//!   cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!     --bin microbench_qnn_oppkg_flash_attn_qwen
//!
//! Pre-deploy (qnn_oppkg cdylib):
//!   cargo build --release -p qnn_oppkg --target aarch64-linux-android
//!   adb push target/aarch64-linux-android/release/libqnn_oppkg.so /data/local/tmp/
//!
//! Run:
//!   adb shell "LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:/vendor/lib64 \
//!              /data/local/tmp/microbench_qnn_oppkg_flash_attn_qwen"

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_oppkg_flash_attn_qwen requires --features qnn");
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
fn main() -> anyhow::Result<()> {
    use libloading::{Library, Symbol};
    use std::ffi::{CString, c_void};
    use std::os::raw::c_uint;
    use std::ptr;

    use qnn::*;

    const PKG_PATH: &str = "/data/local/tmp/libqnn_oppkg.so";
    const PKG_PROVIDER: &str = "QnnOpPackage_InitInterface";
    const PKG_TARGET: &str = "GPU_QTI_AISW";

    println!("=== microbench_qnn_oppkg_flash_attn_qwen (M2.H replay) ===\n");
    println!("Op Package: {}", PKG_PATH);

    let backend_lib_path = std::env::var("QNN_BACKEND_LIB")
        .unwrap_or_else(|_| "/data/local/tmp/qnn/libQnnGpu.so".to_string());
    println!("Backend lib: {}\n", backend_lib_path);

    use ocl::{Context, Device, Platform, Program, Queue};

    let platform = Platform::default();
    let device = Device::first(platform)?;
    let cl_ctx = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let cl_q = Queue::new(&cl_ctx, device, None)?;
    let kernel_src = include_str!("../../kernels/flash_attn_f32_f16.cl");
    let cl_program = Program::builder()
        .devices(device)
        .src(kernel_src)
        .cmplr_opt("-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math -DDK=128 -DDV=128 -DBLOCK_M=32 -DBLOCK_N=32")
        .build(&cl_ctx)?;
    let cl_kernel = ocl::core::create_kernel(&cl_program, "flash_attn_f32_f16_q1")?;

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
    anyhow::ensure!(err == 0, "registerOpPackage err=0x{:x}", err);
    println!("registerOpPackage: OK\n");

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

    // ── Sweep: Qwen dims at chain n_kv vs longer KV ─────────────────────────
    // Original failure mode: M2.H sg3 (n_head=12, n_head_kv=2, n_kv=1) showed
    // a deterministic `qnn_attn_o = ref × 0.507` ratio. Root cause: the
    // OpPackage descriptor sized `sinks` mem-object as 1 element, but the
    // kernel reads `sinks_ptr[head_idx]` for head_idx ∈ [0, n_head). With a
    // zero-initialised buffer, the kernel takes the `sinks_void != NULL`
    // branch and uses `m_i = 0` (instead of -INFINITY) plus
    // `l_final += exp(0 - m_final)`, which inflates the denominator.
    //
    // Fix: descriptor sinks mem_object now sized to `n_head` f32, host fills
    // with -1e30 so `exp(sink - m_final) = 0` — matches raw-OpenCL
    // `mem_null()` baseline byte-for-byte.
    //
    // (n_head=32 cases also exercised but graph creation hits a per-context
    // limit when too many graphs accumulate; tested separately in M2.F.)
    let n_head_kv: usize = 2;
    let head_dim: usize = 128;
    let kv_capacity: usize = 2048;
    let cases: &[(usize, usize)] = &[
        (12, 1),   // chain pattern: Qwen + write_pos=0 (was the 0.5-ratio case)
        (12, 128), // Qwen dims, longer KV
    ];

    let mut all_pass = true;
    for &(n_head, n_kv) in cases {
        println!(
            "--- (n_head={}, n_head_kv={}, head_dim={}, capacity={}, n_kv={}) ---",
            n_head, n_head_kv, head_dim, kv_capacity, n_kv
        );
        let result = run_case(
            &v,
            ctx,
            &cl_q,
            &cl_ctx,
            &cl_kernel,
            &rpcmem_alloc,
            &rpcmem_to_fd,
            RPCMEM_HEAP_ID_SYSTEM,
            RPCMEM_DEFAULT_FLAGS,
            n_head,
            n_head_kv,
            head_dim,
            kv_capacity,
            n_kv,
        );
        match result {
            Ok(max_err) => {
                let pass = max_err < 1e-2;
                println!(
                    "  O max_abs_err = {:.6e}  {}",
                    max_err,
                    if pass { "PASS" } else { "FAIL" }
                );
                if !pass {
                    all_pass = false;
                }
            }
            Err(e) => {
                println!("  ERROR: {}", e);
                all_pass = false;
            }
        }
    }

    println!(
        "\n=== verdict: {} ===",
        if all_pass { "GREEN" } else { "RED" }
    );

    unsafe {
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(be);
    }

    if all_pass {
        Ok(())
    } else {
        std::process::exit(1)
    }
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

#[cfg(feature = "qnn")]
#[allow(clippy::too_many_arguments)]
fn run_case(
    v: &qnn::QnnInterface_ImplementationV2_25_t,
    ctx: qnn::Qnn_ContextHandle_t,
    cl_q: &ocl::Queue,
    cl_ctx: &ocl::Context,
    cl_kernel: &ocl::core::Kernel,
    rpcmem_alloc: &libloading::Symbol<unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void>,
    rpcmem_to_fd: &libloading::Symbol<unsafe extern "C" fn(*const std::ffi::c_void) -> i32>,
    heap_id: i32,
    flags: u32,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    kv_capacity: usize,
    n_kv: usize,
) -> anyhow::Result<f32> {
    use ocl::core::ArgVal;
    use qnn::*;
    use std::ffi::CString;
    use std::ptr;

    let q_total = n_head * head_dim;
    let kv_total = n_head_kv * kv_capacity * head_dim;
    let o_total = n_head * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut host_q = vec![0.0f32; q_total];
    let mut host_k = vec![0u16; kv_total];
    let mut host_v = vec![0u16; kv_total];

    for i in 0..q_total {
        host_q[i] = ((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5;
    }
    // chain pattern: write_pos=0 with n_kv=1 — populate position 0 of *both*
    // KV heads with valid data, leave the remaining capacity-1 slots zero.
    for h in 0..n_head_kv {
        for k_idx in 0..n_kv {
            for d in 0..head_dim {
                let off = h * kv_capacity * head_dim + k_idx * head_dim + d;
                let i = off as f32;
                let kv = (i * 0.0241 + 0.13).rem_euclid(1.0) - 0.5;
                let vv = (i * 0.0317 + 0.21).rem_euclid(1.0) - 0.5;
                host_k[off] = f32_to_f16_bits(kv * 0.5);
                host_v[off] = f32_to_f16_bits(vv * 0.5);
            }
        }
    }

    // ── Path A: raw OpenCL reference ─────────────────────────────────────────
    let buf_q = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_ONLY,
            q_total,
            None,
        )?
    };
    let buf_k = unsafe {
        ocl::core::create_buffer::<_, u16>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_ONLY,
            kv_total,
            None,
        )?
    };
    let buf_v = unsafe {
        ocl::core::create_buffer::<_, u16>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_ONLY,
            kv_total,
            None,
        )?
    };
    let buf_o = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_WRITE,
            o_total,
            None,
        )?
    };
    let buf_score_dummy = unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_WRITE, 1, None)?
    };

    let host_o_init = vec![0.0f32; o_total];

    unsafe {
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_q,
            true,
            0,
            &host_q,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_k,
            true,
            0,
            &host_k,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_v,
            true,
            0,
            &host_v,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_o,
            true,
            0,
            &host_o_init,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    let q_nb1 = (n_head * head_dim * 4) as u64;
    let q_nb2 = (head_dim * 4) as u64;
    let q_nb3 = q_nb1;
    let k_nb1 = (head_dim * 2) as u64;
    let k_nb2 = (kv_capacity * head_dim * 2) as u64;
    let k_nb3 = (n_head_kv as u64) * k_nb2;
    let o_nb1 = (head_dim * 4) as u64;
    let o_nb2 = (n_head * head_dim * 4) as u64;
    let o_nb3 = o_nb2;

    let zero_u64: u64 = 0;
    let zero_i32: i32 = 0;
    let n_q: i32 = 1;
    let n_kv_i: i32 = n_kv as i32;
    let is_causal: i32 = 0;
    let n_head_i: i32 = n_head as i32;
    let n_head_kv_i: i32 = n_head_kv as i32;
    let max_bias: f32 = 0.0;
    let m0: f32 = 0.0;
    let m1: f32 = 0.0;
    let n_head_log2: i32 = 0;
    let logit_softcap: f32 = 0.0;

    ocl::core::set_kernel_arg(cl_kernel, 0, ArgVal::mem(&buf_q))?;
    ocl::core::set_kernel_arg(cl_kernel, 1, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(cl_kernel, 2, ArgVal::mem(&buf_k))?;
    ocl::core::set_kernel_arg(cl_kernel, 3, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(cl_kernel, 4, ArgVal::mem(&buf_v))?;
    ocl::core::set_kernel_arg(cl_kernel, 5, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(cl_kernel, 6, ArgVal::mem(&buf_o))?;
    ocl::core::set_kernel_arg(cl_kernel, 7, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(cl_kernel, 8, ArgVal::scalar(&scale))?;
    ocl::core::set_kernel_arg(cl_kernel, 9, ArgVal::scalar(&n_q))?;
    ocl::core::set_kernel_arg(cl_kernel, 10, ArgVal::scalar(&n_kv_i))?;
    ocl::core::set_kernel_arg(cl_kernel, 11, ArgVal::scalar(&is_causal))?;
    ocl::core::set_kernel_arg(cl_kernel, 12, ArgVal::scalar(&n_head_i))?;
    ocl::core::set_kernel_arg(cl_kernel, 13, ArgVal::scalar(&q_nb1))?;
    ocl::core::set_kernel_arg(cl_kernel, 14, ArgVal::scalar(&q_nb2))?;
    ocl::core::set_kernel_arg(cl_kernel, 15, ArgVal::scalar(&q_nb3))?;
    ocl::core::set_kernel_arg(cl_kernel, 16, ArgVal::scalar(&k_nb1))?;
    ocl::core::set_kernel_arg(cl_kernel, 17, ArgVal::scalar(&k_nb2))?;
    ocl::core::set_kernel_arg(cl_kernel, 18, ArgVal::scalar(&k_nb3))?;
    ocl::core::set_kernel_arg(cl_kernel, 19, ArgVal::scalar(&k_nb1))?;
    ocl::core::set_kernel_arg(cl_kernel, 20, ArgVal::scalar(&k_nb2))?;
    ocl::core::set_kernel_arg(cl_kernel, 21, ArgVal::scalar(&k_nb3))?;
    ocl::core::set_kernel_arg(cl_kernel, 22, ArgVal::scalar(&o_nb1))?;
    ocl::core::set_kernel_arg(cl_kernel, 23, ArgVal::scalar(&o_nb2))?;
    ocl::core::set_kernel_arg(cl_kernel, 24, ArgVal::scalar(&o_nb3))?;
    ocl::core::set_kernel_arg(cl_kernel, 25, ArgVal::scalar(&max_bias))?;
    ocl::core::set_kernel_arg(cl_kernel, 26, ArgVal::scalar(&m0))?;
    ocl::core::set_kernel_arg(cl_kernel, 27, ArgVal::scalar(&m1))?;
    ocl::core::set_kernel_arg(cl_kernel, 28, ArgVal::scalar(&n_head_log2))?;
    ocl::core::set_kernel_arg(cl_kernel, 29, ArgVal::scalar(&logit_softcap))?;
    ocl::core::set_kernel_arg(cl_kernel, 30, ArgVal::scalar(&n_head_kv_i))?;
    ocl::core::set_kernel_arg(cl_kernel, 31, ArgVal::mem_null())?;
    ocl::core::set_kernel_arg(cl_kernel, 32, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(cl_kernel, 33, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(cl_kernel, 34, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(cl_kernel, 35, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(cl_kernel, 36, ArgVal::scalar(&zero_i32))?;
    ocl::core::set_kernel_arg(cl_kernel, 37, ArgVal::scalar(&zero_i32))?;
    ocl::core::set_kernel_arg(cl_kernel, 38, ArgVal::mem_null())?;
    ocl::core::set_kernel_arg(cl_kernel, 39, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(cl_kernel, 40, ArgVal::mem(&buf_score_dummy))?;
    ocl::core::set_kernel_arg(cl_kernel, 41, ArgVal::scalar(&zero_i32))?;
    ocl::core::set_kernel_arg(cl_kernel, 42, ArgVal::scalar(&zero_i32))?;
    ocl::core::set_kernel_arg(cl_kernel, 43, ArgVal::scalar(&zero_i32))?;

    const Q1_WG_SIZE: usize = 64;
    let global = [Q1_WG_SIZE, n_head, 1];
    let local = [Q1_WG_SIZE, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            cl_kernel,
            2,
            None,
            &global,
            Some(local),
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    let mut ref_o = vec![0.0f32; o_total];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_o,
            true,
            0,
            &mut ref_o,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // ── Path B: QNN graph with CustomFlashAttn ───────────────────────────────
    let q_bytes = (q_total * 4) as i32;
    let kv_bytes = (kv_total * 2) as i32;
    let mask_elems = n_kv;
    let mask_bytes = (mask_elems * 2) as i32;
    let dummy_bytes: i32 = 4;
    // sinks buffer must be sized to `n_head * sizeof(f32)` because the kernel
    // reads `sinks_ptr[head_idx]` for head_idx ∈ [0, n_head). A 1-element
    // buffer triggers OOB reads for head_idx > 0 (driver may return zeros,
    // but UB).
    let sinks_bytes = (n_head * 4) as i32;
    let o_bytes = (o_total * 4) as i32;

    let rpc_q = unsafe { rpcmem_alloc(heap_id, flags, q_bytes) };
    let rpc_k = unsafe { rpcmem_alloc(heap_id, flags, kv_bytes) };
    let rpc_v = unsafe { rpcmem_alloc(heap_id, flags, kv_bytes) };
    let rpc_mask = unsafe { rpcmem_alloc(heap_id, flags, mask_bytes) };
    let rpc_sinks = unsafe { rpcmem_alloc(heap_id, flags, sinks_bytes) };
    let rpc_score = unsafe { rpcmem_alloc(heap_id, flags, dummy_bytes) };
    let rpc_o = unsafe { rpcmem_alloc(heap_id, flags, o_bytes) };
    anyhow::ensure!(
        !rpc_q.is_null()
            && !rpc_k.is_null()
            && !rpc_v.is_null()
            && !rpc_mask.is_null()
            && !rpc_sinks.is_null()
            && !rpc_score.is_null()
            && !rpc_o.is_null(),
        "rpcmem_alloc failed"
    );

    unsafe {
        std::ptr::copy_nonoverlapping(
            host_q.as_ptr() as *const u8,
            rpc_q as *mut u8,
            q_bytes as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_k.as_ptr() as *const u8,
            rpc_k as *mut u8,
            kv_bytes as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_v.as_ptr() as *const u8,
            rpc_v as *mut u8,
            kv_bytes as usize,
        );
        std::ptr::write_bytes(rpc_mask as *mut u8, 0, mask_bytes as usize);
        // sinks must NOT be zero — kernel branches on `sinks_void != NULL` and
        // treats sinks_ptr[head_idx] as `m_sink` (an attention sink logit). A
        // zero buffer makes m_sink=0 enter `m_final = max(m_i, 0)`, causing
        // `l_final += exp(0 - m_final)` which inflates the denominator. For
        // OpPackage parity with the raw-OpenCL `mem_null()` baseline we set
        // sinks to a very large negative value so `exp(sinks - m_final) ≈ 0`.
        // -1e30 is well below any plausible attention logit and `exp(-1e30) =
        // 0` exactly in IEEE-754 single precision.
        let neg_huge: f32 = -1.0e30f32;
        for i in 0..n_head {
            std::ptr::write_unaligned((rpc_sinks as *mut u8).add(i * 4) as *mut f32, neg_huge);
        }
        std::ptr::write_bytes(rpc_score as *mut u8, 0, dummy_bytes as usize);
        std::ptr::write_bytes(rpc_o as *mut u8, 0, o_bytes as usize);
    }

    let fd_q = unsafe { rpcmem_to_fd(rpc_q) };
    let fd_k = unsafe { rpcmem_to_fd(rpc_k) };
    let fd_v = unsafe { rpcmem_to_fd(rpc_v) };
    let fd_mask = unsafe { rpcmem_to_fd(rpc_mask) };
    let fd_sinks = unsafe { rpcmem_to_fd(rpc_sinks) };
    let fd_score = unsafe { rpcmem_to_fd(rpc_score) };
    let fd_o = unsafe { rpcmem_to_fd(rpc_o) };

    let mut dims_q = vec![1u32, n_head as u32, head_dim as u32];
    let mut dims_kv = vec![1u32, n_head_kv as u32, kv_capacity as u32, head_dim as u32];
    let mut dims_kv2 = dims_kv.clone();
    let mut dims_mask = vec![mask_elems as u32];
    // sinks: n_head halves of float32 — kernel indexes sinks_ptr[head_idx]
    let mut dims_sinks = vec![n_head as u32];
    let mut dims_dummy2 = vec![1u32];
    let mut dims_o = vec![1u32, n_head as u32, head_dim as u32];

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

    macro_rules! mk_tensor {
        ($ttype:expr, $dtype:expr, $rank:expr, $dims:expr) => {
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: Qnn_TensorV1_t {
                        id: 0,
                        name: ptr::null(),
                        type_: $ttype,
                        dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                        dataType: $dtype,
                        quantizeParams: qp,
                        rank: $rank,
                        dimensions: $dims,
                        memType: Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
                        __bindgen_anon_1: Qnn_TensorV1_t__bindgen_ty_1 {
                            clientBuf: Qnn_ClientBuffer_t {
                                data: ptr::null_mut(),
                                dataSize: 0,
                            },
                        },
                    },
                },
            }
        };
    }

    let name_q = CString::new("fa_q").unwrap();
    let name_k = CString::new("fa_k").unwrap();
    let name_v = CString::new("fa_v").unwrap();
    let name_mask = CString::new("fa_mask").unwrap();
    let name_sinks = CString::new("fa_sinks").unwrap();
    let name_score = CString::new("fa_score").unwrap();
    let name_o = CString::new("fa_o").unwrap();

    let mut t_q = mk_tensor!(
        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
        3,
        dims_q.as_mut_ptr()
    );
    t_q.__bindgen_anon_1.v1.name = name_q.as_ptr();

    let mut t_k = mk_tensor!(
        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
        4,
        dims_kv.as_mut_ptr()
    );
    t_k.__bindgen_anon_1.v1.name = name_k.as_ptr();

    let mut t_v = mk_tensor!(
        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
        4,
        dims_kv2.as_mut_ptr()
    );
    t_v.__bindgen_anon_1.v1.name = name_v.as_ptr();

    let mut t_mask = mk_tensor!(
        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
        1,
        dims_mask.as_mut_ptr()
    );
    t_mask.__bindgen_anon_1.v1.name = name_mask.as_ptr();

    let mut t_sinks = mk_tensor!(
        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
        1,
        dims_sinks.as_mut_ptr()
    );
    t_sinks.__bindgen_anon_1.v1.name = name_sinks.as_ptr();

    let mut t_score = mk_tensor!(
        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
        1,
        dims_dummy2.as_mut_ptr()
    );
    t_score.__bindgen_anon_1.v1.name = name_score.as_ptr();

    let mut t_o = mk_tensor!(
        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
        3,
        dims_o.as_mut_ptr()
    );
    t_o.__bindgen_anon_1.v1.name = name_o.as_ptr();

    let g_name = CString::new(format!("fa_graph_qwen_{}", n_kv)).unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    for (label, t) in [
        ("Q", &mut t_q),
        ("K", &mut t_k),
        ("V", &mut t_v),
        ("mask", &mut t_mask),
        ("sinks", &mut t_sinks),
        ("score", &mut t_score),
        ("O", &mut t_o),
    ] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", label, err);
    }

    let p_n_kv = CString::new("n_kv").unwrap();
    let p_n_head = CString::new("n_head").unwrap();
    let p_n_head_kv = CString::new("n_head_kv").unwrap();
    let p_kv_cap = CString::new("kv_capacity").unwrap();
    let p_head_dim = CString::new("head_dim").unwrap();

    macro_rules! mk_iparam {
        ($name:expr, $val:expr) => {
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: $name,
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 { int32Value: $val },
                    },
                },
            }
        };
    }

    let mut params = [
        mk_iparam!(p_n_kv.as_ptr(), n_kv as i32),
        mk_iparam!(p_n_head.as_ptr(), n_head as i32),
        mk_iparam!(p_n_head_kv.as_ptr(), n_head_kv as i32),
        mk_iparam!(p_kv_cap.as_ptr(), kv_capacity as i32),
        mk_iparam!(p_head_dim.as_ptr(), head_dim as i32),
    ];

    let op_name = CString::new("fa_op0").unwrap();
    let pkg = CString::new("qnn_oppkg").unwrap();
    let op_type = CString::new("CustomFlashAttn").unwrap();
    let mut inputs = [t_q, t_k, t_v, t_mask, t_sinks, t_score];
    let mut outputs = [t_o];
    let op = Qnn_OpConfig_t {
        version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
        __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
            v1: Qnn_OpConfigV1_t {
                name: op_name.as_ptr(),
                packageName: pkg.as_ptr(),
                typeName: op_type.as_ptr(),
                numOfParams: 5,
                params: params.as_mut_ptr(),
                numOfInputs: 6,
                inputTensors: inputs.as_mut_ptr(),
                numOfOutputs: 1,
                outputTensors: outputs.as_mut_ptr(),
            },
        },
    };
    let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
    anyhow::ensure!(err == 0, "graphAddNode err=0x{:x}", err);

    let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
    anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);

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

    let dims_q_u = vec![1u32, n_head as u32, head_dim as u32];
    let dims_kv_u = vec![1u32, n_head_kv as u32, kv_capacity as u32, head_dim as u32];
    let dims_mask_u = vec![mask_elems as u32];
    let dims_sinks_u = vec![n_head as u32];
    let dims_dummy_u = vec![1u32];
    let dims_o_u = vec![1u32, n_head as u32, head_dim as u32];

    let descs = [
        mk_desc(fd_q, rpc_q, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_q_u),
        mk_desc(
            fd_k,
            rpc_k,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
            &dims_kv_u,
        ),
        mk_desc(
            fd_v,
            rpc_v,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
            &dims_kv_u,
        ),
        mk_desc(
            fd_mask,
            rpc_mask,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
            &dims_mask_u,
        ),
        mk_desc(
            fd_sinks,
            rpc_sinks,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            &dims_sinks_u,
        ),
        mk_desc(
            fd_score,
            rpc_score,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            &dims_dummy_u,
        ),
        mk_desc(fd_o, rpc_o, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_o_u),
    ];
    let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 7];
    let err = unsafe { (v.memRegister.unwrap())(ctx, descs.as_ptr(), 7, mh.as_mut_ptr()) };
    anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);

    for (i, h) in mh.iter().enumerate().take(6) {
        inputs[i].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        inputs[i].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = *h;
    }
    outputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[6];

    let err = unsafe {
        (v.graphExecute.unwrap())(
            graph,
            inputs.as_ptr(),
            6,
            outputs.as_mut_ptr(),
            1,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);

    // Diagnostic: dump first 8 elements of QNN vs ref + ratio
    unsafe {
        let test_o = std::slice::from_raw_parts(rpc_o as *const f32, o_total);
        eprintln!("[diag] qnn O[0..8] = {:?}", &test_o[0..8.min(o_total)]);
        eprintln!("[diag] ref O[0..8] = {:?}", &ref_o[0..8.min(o_total)]);
        eprintln!(
            "[diag] qnn[i]/ref[i] [0..8] = {:?}",
            (0..8.min(o_total))
                .map(|i| if ref_o[i].abs() > 1e-30 {
                    test_o[i] / ref_o[i]
                } else {
                    f32::NAN
                })
                .collect::<Vec<_>>()
        );
        // Also dump ratio per-Q-head [head, d=0..4]
        for h in 0..n_head.min(12) {
            let off = h * head_dim;
            let ratios: Vec<f32> = (0..4)
                .map(|d| {
                    let r = ref_o[off + d];
                    let q = test_o[off + d];
                    if r.abs() > 1e-30 { q / r } else { f32::NAN }
                })
                .collect();
            eprintln!("[diag] head={} ratio[0..4] = {:?}", h, ratios);
        }
    }

    let mut max_abs = 0.0f32;
    unsafe {
        let test_o = std::slice::from_raw_parts(rpc_o as *const f32, o_total);
        for i in 0..o_total {
            let diff = (test_o[i] - ref_o[i]).abs();
            if diff > max_abs {
                max_abs = diff;
            }
        }
    }

    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), 7);
    }

    Ok(max_abs)
}
