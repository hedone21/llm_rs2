//! M1.4 CustomMatMulF16F32 correctness — production OpPackage vs raw OpenCL.
//!
//! Reference: raw OpenCL `kernel_mul_mat_f16_f32` invoked directly on the same
//! kernel source the OpPackage embeds. Test: QNN backend registers
//! `libqnn_oppkg.so` and executes a graph with one `CustomMatMulF16F32` node.
//!
//! Cases (M, N, K) ∈ {(1, 512, 256), (4, 1536, 1536), (1, 8960, 1536)}.
//! Pass criterion: max_abs_err < 1e-2 (F16 weight × F32 x → F32 y).
//!
//! Build (Android cross):
//!   cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!     --bin microbench_qnn_oppkg_matmul_correct
//!
//! Pre-deploy:
//!   cargo build --release -p qnn_oppkg --target aarch64-linux-android
//!   adb push target/aarch64-linux-android/release/libqnn_oppkg.so /data/local/tmp/
//!
//! Run on device:
//!   adb shell "LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:/vendor/lib64 \
//!              /data/local/tmp/microbench_qnn_oppkg_matmul_correct"

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_oppkg_matmul_correct requires --features qnn");
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

    println!("=== microbench_qnn_oppkg_matmul_correct (M1.4) ===\n");
    println!("Op Package: {}", PKG_PATH);
    println!("Interface symbol: {}", PKG_PROVIDER);

    let backend_lib_path = std::env::var("QNN_BACKEND_LIB")
        .unwrap_or_else(|_| "/data/local/tmp/qnn/libQnnGpu.so".to_string());
    println!("Backend lib: {}\n", backend_lib_path);

    // ── Path A: raw OpenCL reference ──────────────────────────────────────────
    use ocl::{Context, Device, Platform, Program, Queue};

    let platform = Platform::default();
    let device = Device::first(platform)?;
    let cl_ctx = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let cl_q = Queue::new(&cl_ctx, device, None)?;
    let kernel_src = include_str!("../../kernels/mul_mv_f16_f32.cl");
    let cl_program = Program::builder()
        .devices(device)
        .src(kernel_src)
        .cmplr_opt("-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math")
        .build(&cl_ctx)?;
    let cl_kernel = ocl::core::create_kernel(&cl_program, "kernel_mul_mat_f16_f32")?;

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

    // rpcmem for DMA_BUF tensors
    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;
    type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(po: *const c_void) -> i32;
    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }?;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let _rpcmem_free: Symbol<RpcmemFreeFn> = unsafe { rpc_lib.get(b"rpcmem_free\0")? };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };

    let cases: &[(usize, usize, usize)] = &[(1, 512, 256), (4, 1536, 1536), (1, 8960, 1536)];
    let mut all_pass = true;

    for &(m, n, k) in cases {
        println!("--- (M, N, K) = ({}, {}, {}) ---", m, n, k);
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
            m,
            n,
            k,
        );
        match result {
            Ok(max_err) => {
                let pass = max_err < 1e-2;
                println!(
                    "  max_abs_err = {:.6e}  {}",
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
        "\n=== M1.4 verdict: {} ===",
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
    m: usize,
    n: usize,
    k: usize,
) -> anyhow::Result<f32> {
    use ocl::core::ArgVal;
    use qnn::*;
    use std::ffi::CString;
    use std::ptr;

    // ── Generate identical inputs for both paths ───────────────────────────────
    // Same recipe as microbench_qnngpu_matmul_tbt to keep paths apple-to-apple.
    let mut host_w_f16 = vec![0u16; n * k];
    let mut host_x_f32 = vec![0.0f32; m * k];
    for (i, w) in host_w_f16.iter_mut().enumerate() {
        let v = ((i as f32) * 0.0007 + 0.13).rem_euclid(1.0) - 0.5;
        *w = f32_to_f16_bits(v);
    }
    for (i, x) in host_x_f32.iter_mut().enumerate() {
        *x = ((i as f32) * 0.011).rem_euclid(1.0) - 0.5;
    }

    // ── Path A: raw OpenCL reference ──────────────────────────────────────────
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
            &host_w_f16,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_x,
            true,
            0,
            &host_x_f32,
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

    // ── Path B: QNN graph with CustomMatMulF16F32 ─────────────────────────────
    let bytes_w = (n * k * 2) as i32; // F16
    let bytes_x = (m * k * 4) as i32; // F32
    let bytes_y = (m * n * 4) as i32; // F32
    let rpc_w = unsafe { rpcmem_alloc(heap_id, flags, bytes_w) };
    let rpc_x = unsafe { rpcmem_alloc(heap_id, flags, bytes_x) };
    let rpc_y = unsafe { rpcmem_alloc(heap_id, flags, bytes_y) };
    anyhow::ensure!(
        !rpc_w.is_null() && !rpc_x.is_null() && !rpc_y.is_null(),
        "rpcmem_alloc failed for (M,N,K)=({},{},{})",
        m,
        n,
        k
    );

    unsafe {
        std::ptr::copy_nonoverlapping(
            host_w_f16.as_ptr() as *const u8,
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

    // weight [N, K] FLOAT_16, x [M, K] FLOAT_32, y [M, N] FLOAT_32.
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

    let name_w = CString::new(format!("w_{}_{}", n, k)).unwrap();
    let name_x = CString::new(format!("x_{}_{}", m, k)).unwrap();
    let name_y = CString::new(format!("y_{}_{}", m, n)).unwrap();

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

    let g_name = CString::new(format!("matmul_graph_{}_{}_{}", m, n, k)).unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    for (label, t) in [("w", &mut t_w), ("x", &mut t_x), ("y", &mut t_y)] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", label, err);
    }

    let op_name = CString::new(format!("matmul0_{}_{}_{}", m, n, k)).unwrap();
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
    anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);

    let mut max_abs = 0.0f32;
    unsafe {
        let test_y = std::slice::from_raw_parts(rpc_y as *const f32, m * n);
        for i in 0..(m * n) {
            let d = (test_y[i] - ref_y[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
    }

    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), 3);
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
