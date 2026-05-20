//! M4 fundamental risk PoC — QNN OpPackage weight buffer mutability.
//!
//! 핵심 질문: graph build/finalize 후, host에서 weight buffer (rpcmem DMA_BUF)를
//! 변조하면 다음 graphExecute가 새 weight를 사용하는가?
//!
//! - YES → M4 async weight swap 가능 (GREEN)
//! - NO  → SDK가 weight를 internal cache로 copy → host write 무시 (RED)
//!
//! 단일 케이스 `CustomMatMulF16F32` (M=1, N=512, K=256, weight=256 KB).
//! Raw OpenCL reference는 weight A/B 각각 실행 → ref_A, ref_B.
//! QNN graph: 1회 finalize, rpcmem 1개 — weight A → execute → result_A,
//! 같은 rpcmem에 weight B memcpy → execute → result_B.
//!
//! 판정:
//!   max_abs_err(result_A, ref_A) < 1e-2  AND
//!   max_abs_err(result_B, ref_B) < 1e-2  → MUTABLE (GREEN)
//!   max_abs_err(result_B, ref_A) ≈ 0     → IMMUTABLE / cached (RED)
//!
//! Build (Android cross):
//!   cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!     --bin microbench_qnn_weight_mutability_poc
//!
//! Pre-deploy:
//!   cargo build --release -p qnn_oppkg --target aarch64-linux-android
//!   adb push target/aarch64-linux-android/release/libqnn_oppkg.so /data/local/tmp/
//!
//! Run on device:
//!   adb shell "LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:/vendor/lib64 \
//!              /data/local/tmp/microbench_qnn_weight_mutability_poc"

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_weight_mutability_poc requires --features qnn");
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

    println!("=== microbench_qnn_weight_mutability_poc (M4 risk check) ===\n");
    println!("Question: does host mutating weight rpcmem after finalize");
    println!("          change the next graphExecute output?\n");

    let backend_lib_path = std::env::var("QNN_BACKEND_LIB")
        .unwrap_or_else(|_| "/data/local/tmp/qnn/libQnnGpu.so".to_string());
    println!("Backend lib: {}", backend_lib_path);
    println!("Op Package: {}\n", PKG_PATH);

    // Case: small enough to be quick, large enough to differentiate.
    let (m, n, k) = (1usize, 512usize, 256usize);
    println!("Case: (M, N, K) = ({}, {}, {})", m, n, k);
    println!("  weight bytes = {} (F16)", n * k * 2);
    println!("  x bytes      = {} (F32)", m * k * 4);
    println!("  y bytes      = {} (F32)\n", m * n * 4);

    // ── Generate weight A, weight B (different seeds) and shared x ─────────────
    let host_w_a_f16 = make_weight_f16(n * k, /*seed*/ 0);
    let host_w_b_f16 = make_weight_f16(n * k, /*seed*/ 1);
    let host_x_f32 = make_x_f32(m * k);

    // ── Path A: raw OpenCL reference for weight A and weight B ─────────────────
    use ocl::{Context, Device, Platform, Program, Queue};

    let platform = Platform::default();
    let device = Device::first(platform)?;
    let cl_ctx = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let cl_q = Queue::new(&cl_ctx, device, None)?;
    let kernel_src = include_str!("../kernels/mul_mv_f16_f32.cl");
    let cl_program = Program::builder()
        .devices(device)
        .src(kernel_src)
        .cmplr_opt("-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math")
        .build(&cl_ctx)?;
    let cl_kernel = ocl::core::create_kernel(&cl_program, "kernel_mul_mat_f16_f32")?;

    let ref_a = run_opencl_ref(
        &cl_q,
        &cl_ctx,
        &cl_kernel,
        &host_w_a_f16,
        &host_x_f32,
        m,
        n,
        k,
    )?;
    let ref_b = run_opencl_ref(
        &cl_q,
        &cl_ctx,
        &cl_kernel,
        &host_w_b_f16,
        &host_x_f32,
        m,
        n,
        k,
    )?;

    let ref_ab_diff = max_abs_diff(&ref_a, &ref_b);
    println!(
        "[ref] max_abs_diff(ref_A, ref_B) = {:.6e}  (must be >> 0 to be a valid test)\n",
        ref_ab_diff
    );
    anyhow::ensure!(
        ref_ab_diff > 1e-2,
        "weight A and B produce nearly identical reference outputs (ref_ab_diff={:.3e}); test invalid",
        ref_ab_diff
    );

    // ── Path B: QNN-GPU backend + register OpPackage ──────────────────────────
    let gpu_lib = unsafe { Library::new(&backend_lib_path) }?;
    type GetProvidersFn = unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let gp: Symbol<GetProvidersFn> = unsafe { gpu_lib.get(b"QnnInterface_getProviders\0")? };
    let mut provs: *mut *const QnnInterface_t = ptr::null_mut();
    let mut np: c_uint = 0;
    let err = unsafe { gp(&mut provs, &mut np) };
    anyhow::ensure!(err == 0 && np > 0, "GPU getProviders err=0x{:x}", err);
    let v = unsafe { (**provs).__bindgen_anon_1.v2_25 };

    let mut be: Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe { (v.backendCreate.unwrap())(ptr::null_mut(), ptr::null_mut(), &mut be) };
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

    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe { (v.contextCreate.unwrap())(be, ptr::null_mut(), ptr::null_mut(), &mut ctx) };
    anyhow::ensure!(err == 0, "contextCreate err=0x{:x}", err);

    // rpcmem (DMA_BUF)
    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;
    type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(po: *const c_void) -> i32;
    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }?;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let _rpcmem_free: Symbol<RpcmemFreeFn> = unsafe { rpc_lib.get(b"rpcmem_free\0")? };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };

    let bytes_w = (n * k * 2) as i32; // F16
    let bytes_x = (m * k * 4) as i32; // F32
    let bytes_y = (m * n * 4) as i32; // F32
    let rpc_w = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes_w) };
    let rpc_x = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes_x) };
    let rpc_y = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes_y) };
    anyhow::ensure!(
        !rpc_w.is_null() && !rpc_x.is_null() && !rpc_y.is_null(),
        "rpcmem_alloc failed"
    );

    // Phase 1 prep: write weight A into rpc_w, write x into rpc_x.
    unsafe {
        std::ptr::copy_nonoverlapping(
            host_w_a_f16.as_ptr() as *const u8,
            rpc_w as *mut u8,
            bytes_w as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_x_f32.as_ptr() as *const u8,
            rpc_x as *mut u8,
            bytes_x as usize,
        );
    }

    let fd_w = unsafe { rpcmem_to_fd(rpc_w) };
    let fd_x = unsafe { rpcmem_to_fd(rpc_x) };
    let fd_y = unsafe { rpcmem_to_fd(rpc_y) };

    let mut dims_w: Vec<u32> = vec![n as u32, k as u32];
    let mut dims_x: Vec<u32> = vec![m as u32, k as u32];
    let mut dims_y: Vec<u32> = vec![m as u32, n as u32];

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
    let mk_tv1 = |ttype, dtype, dims_ptr: *mut u32| Qnn_TensorV1_t {
        id: 0,
        name: ptr::null(),
        type_: ttype,
        dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        dataType: dtype,
        quantizeParams: qp,
        rank: 2,
        dimensions: dims_ptr,
        memType: Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
        __bindgen_anon_1: Qnn_TensorV1_t__bindgen_ty_1 {
            clientBuf: Qnn_ClientBuffer_t {
                data: ptr::null_mut(),
                dataSize: 0,
            },
        },
    };

    let name_w = CString::new("w_mut_poc").unwrap();
    let name_x = CString::new("x_mut_poc").unwrap();
    let name_y = CString::new("y_mut_poc").unwrap();

    let mut t_w = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
                dims_w.as_mut_ptr(),
            ),
        },
    };
    t_w.__bindgen_anon_1.v1.name = name_w.as_ptr();
    let mut t_x = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                dims_x.as_mut_ptr(),
            ),
        },
    };
    t_x.__bindgen_anon_1.v1.name = name_x.as_ptr();
    let mut t_y = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                dims_y.as_mut_ptr(),
            ),
        },
    };
    t_y.__bindgen_anon_1.v1.name = name_y.as_ptr();

    let g_name = CString::new("matmul_mut_poc").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    for (label, t) in [("w", &mut t_w), ("x", &mut t_x), ("y", &mut t_y)] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", label, err);
    }

    let op_name = CString::new("matmul0_mut_poc").unwrap();
    let pkg = CString::new("qnn_oppkg").unwrap();
    let op_type = CString::new("CustomMatMulF16F32").unwrap();
    let mut inputs = [t_w, t_x];
    let mut outputs = [t_y];
    let op = Qnn_OpConfig_t {
        version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
        __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
            v1: Qnn_OpConfigV1_t {
                name: op_name.as_ptr(),
                packageName: pkg.as_ptr(),
                typeName: op_type.as_ptr(),
                numOfParams: 0,
                params: ptr::null_mut(),
                numOfInputs: 2,
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
    println!("graph finalize: OK\n");

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
        mk_desc(fd_w, rpc_w, Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, &dims_w),
        mk_desc(fd_x, rpc_x, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_x),
        mk_desc(fd_y, rpc_y, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_y),
    ];
    let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 3];
    let err = unsafe { (v.memRegister.unwrap())(ctx, descs.as_ptr(), 3, mh.as_mut_ptr()) };
    anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);

    inputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[0];
    inputs[1].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[1];
    outputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[2];

    // ── Phase 1: weight A → execute → result_A ───────────────────────────────
    println!("Phase 1: graphExecute with weight A");
    // Zero the output buffer to ensure we observe a fresh write.
    unsafe { std::ptr::write_bytes(rpc_y as *mut u8, 0u8, bytes_y as usize) };
    let err = unsafe {
        (v.graphExecute.unwrap())(
            graph,
            inputs.as_ptr(),
            2,
            outputs.as_mut_ptr(),
            1,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    anyhow::ensure!(err == 0, "graphExecute (Phase 1) err=0x{:x}", err);

    let mut result_a = vec![0.0f32; m * n];
    unsafe {
        let test_y = std::slice::from_raw_parts(rpc_y as *const f32, m * n);
        result_a.copy_from_slice(test_y);
    }
    let err_a_vs_ref_a = max_abs_diff(&result_a, &ref_a);
    let err_a_vs_ref_b = max_abs_diff(&result_a, &ref_b);
    println!(
        "  max_abs_err(result_A, ref_A) = {:.6e}  (expect << 1e-2)",
        err_a_vs_ref_a
    );
    println!(
        "  max_abs_err(result_A, ref_B) = {:.6e}  (sanity, expect ~ ref_ab_diff)\n",
        err_a_vs_ref_b
    );

    // ── Phase 2: host overwrites weight A → weight B in same rpcmem ──────────
    println!("Phase 2: host memcpy weight B into the SAME rpc_w (no re-register)");
    unsafe {
        std::ptr::copy_nonoverlapping(
            host_w_b_f16.as_ptr() as *const u8,
            rpc_w as *mut u8,
            bytes_w as usize,
        );
    }
    // Zero output again to detect stale-output false positive.
    unsafe { std::ptr::write_bytes(rpc_y as *mut u8, 0u8, bytes_y as usize) };
    let err = unsafe {
        (v.graphExecute.unwrap())(
            graph,
            inputs.as_ptr(),
            2,
            outputs.as_mut_ptr(),
            1,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    anyhow::ensure!(err == 0, "graphExecute (Phase 2) err=0x{:x}", err);

    let mut result_b = vec![0.0f32; m * n];
    unsafe {
        let test_y = std::slice::from_raw_parts(rpc_y as *const f32, m * n);
        result_b.copy_from_slice(test_y);
    }
    let err_b_vs_ref_b = max_abs_diff(&result_b, &ref_b);
    let err_b_vs_ref_a = max_abs_diff(&result_b, &ref_a);
    let err_a_vs_b = max_abs_diff(&result_a, &result_b);
    println!(
        "  max_abs_err(result_B, ref_B) = {:.6e}  (MUTABLE if << 1e-2)",
        err_b_vs_ref_b
    );
    println!(
        "  max_abs_err(result_B, ref_A) = {:.6e}  (IMMUTABLE if ~ 0)",
        err_b_vs_ref_a
    );
    println!(
        "  max_abs_err(result_A, result_B) = {:.6e}  (~ 0 means cached)\n",
        err_a_vs_b
    );

    // ── Verdict ──────────────────────────────────────────────────────────────
    let phase1_ok = err_a_vs_ref_a < 1e-2;
    let mutable = err_b_vs_ref_b < 1e-2 && err_b_vs_ref_a > ref_ab_diff * 0.5;
    let immutable = err_a_vs_b < 1e-3 && err_b_vs_ref_a < 1e-2;

    println!("=== Verdict ===");
    println!(
        "  Phase 1 (weight A correct):    {}",
        if phase1_ok { "OK" } else { "FAIL" }
    );
    let verdict = if !phase1_ok {
        "INCONCLUSIVE (Phase 1 failed)"
    } else if mutable {
        "MUTABLE (GREEN) — M4 async swap path is feasible"
    } else if immutable {
        "IMMUTABLE (RED) — SDK cached weight; M4 async swap blocked"
    } else {
        "UNCLEAR — partial mutation or other anomaly"
    };
    println!("  M4 fundamental risk:           {}\n", verdict);

    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), 3);
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(be);
    }

    if phase1_ok && mutable {
        Ok(())
    } else {
        std::process::exit(1)
    }
}

#[cfg(feature = "qnn")]
fn make_weight_f16(n: usize, seed: u32) -> Vec<u16> {
    let mut out = vec![0u16; n];
    // Different recipes per seed so OpenCL ref_A vs ref_B are clearly distinct.
    match seed {
        0 => {
            for (i, w) in out.iter_mut().enumerate() {
                let v = ((i as f32) * 0.0007 + 0.13).rem_euclid(1.0) - 0.5;
                *w = f32_to_f16_bits(v);
            }
        }
        _ => {
            for (i, w) in out.iter_mut().enumerate() {
                // Visibly different distribution and sign than seed 0.
                let v = ((i as f32) * 0.0019 + 0.41).rem_euclid(1.0) - 0.5;
                *w = f32_to_f16_bits(-v);
            }
        }
    }
    out
}

#[cfg(feature = "qnn")]
fn make_x_f32(n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (i, x) in out.iter_mut().enumerate() {
        *x = ((i as f32) * 0.011).rem_euclid(1.0) - 0.5;
    }
    out
}

#[cfg(feature = "qnn")]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    m
}

#[cfg(feature = "qnn")]
#[allow(clippy::too_many_arguments)]
fn run_opencl_ref(
    cl_q: &ocl::Queue,
    cl_ctx: &ocl::Context,
    cl_kernel: &ocl::core::Kernel,
    host_w_f16: &[u16],
    host_x_f32: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> anyhow::Result<Vec<f32>> {
    use ocl::core::ArgVal;

    let buf_w = unsafe {
        ocl::core::create_buffer::<_, u16>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, n * k, None)?
    };
    let buf_x = unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, m * k, None)?
    };
    let buf_y = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_WRITE,
            m * n,
            None,
        )?
    };
    unsafe {
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_w,
            true,
            0,
            host_w_f16,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_x,
            true,
            0,
            host_x_f32,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    let ne00 = k as i32;
    let ne01 = n as i32;
    let ne02: i32 = 1;
    let ne10 = k as i32;
    let ne12: i32 = 1;
    let ne0 = n as i32;
    let ne1 = m as i32;
    let r2: i32 = 1;
    let r3: i32 = 1;
    let off0: u64 = 0;
    let off1: u64 = 0;
    let offd: u64 = 0;
    ocl::core::set_kernel_arg(cl_kernel, 0, ArgVal::mem(&buf_w))?;
    ocl::core::set_kernel_arg(cl_kernel, 1, ArgVal::scalar(&off0))?;
    ocl::core::set_kernel_arg(cl_kernel, 2, ArgVal::mem(&buf_x))?;
    ocl::core::set_kernel_arg(cl_kernel, 3, ArgVal::scalar(&off1))?;
    ocl::core::set_kernel_arg(cl_kernel, 4, ArgVal::mem(&buf_y))?;
    ocl::core::set_kernel_arg(cl_kernel, 5, ArgVal::scalar(&offd))?;
    ocl::core::set_kernel_arg(cl_kernel, 6, ArgVal::scalar(&ne00))?;
    ocl::core::set_kernel_arg(cl_kernel, 7, ArgVal::scalar(&ne01))?;
    ocl::core::set_kernel_arg(cl_kernel, 8, ArgVal::scalar(&ne02))?;
    ocl::core::set_kernel_arg(cl_kernel, 9, ArgVal::scalar(&ne10))?;
    ocl::core::set_kernel_arg(cl_kernel, 10, ArgVal::scalar(&ne12))?;
    ocl::core::set_kernel_arg(cl_kernel, 11, ArgVal::scalar(&ne0))?;
    ocl::core::set_kernel_arg(cl_kernel, 12, ArgVal::scalar(&ne1))?;
    ocl::core::set_kernel_arg(cl_kernel, 13, ArgVal::scalar(&r2))?;
    ocl::core::set_kernel_arg(cl_kernel, 14, ArgVal::scalar(&r3))?;

    let n_dst: usize = 2;
    let global = [n.div_ceil(n_dst) * 64, m * 4, 1];
    let local = [64usize, 4, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            cl_kernel,
            3,
            None,
            &global,
            Some(local),
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    let mut ref_y = vec![0.0f32; m * n];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_y,
            true,
            0,
            &mut ref_y,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;
    Ok(ref_y)
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
