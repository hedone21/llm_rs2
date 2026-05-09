//! M1.7 CustomSiluMul correctness — production OpPackage vs raw OpenCL.
//!
//! Reference: raw OpenCL `kernel_silu_mul_simple` invoked directly on the same
//! kernel source the OpPackage embeds. The kernel is **in-place** —
//! `x[i] = silu(x[i]) * y[i]`. Test: QNN backend registers `libqnn_oppkg.so`
//! and executes a graph with one `CustomSiluMul` node.
//!
//! M1.7 selects the simple (one-thread-per-float4) kernel for the same reason
//! as M1.5/M1.6: avoid the unvalidated subgroup / LocalMem path. M1.7 scope
//! is correctness only.
//!
//! In-place mapping: the production kernel reads + writes `x`. The
//! `CustomSiluMul` op declares 2 inputs (x, y) and 1 output (x_inout, an
//! alias of x). Kernel arg 0 uses `OP_INPUT_READWRITE` so the kernel mutates
//! input tensor 0 directly. The output tensor exists so `build_op_state`'s
//! "claim the last mem object" contract is satisfied; at the QNN tensor
//! level x_inout points to the same DMA-BUF fd as x.
//!
//! Cases (rows, dim) ∈ {(1, 5632), (4, 8192), (16, 11008)}. Each `rows * dim`
//! is a multiple of 4 (kernel is float4-vectorised).
//!
//! Pass criterion: max_abs_err < 1e-4 (identical kernels → bit-equal output
//! is expected; threshold is loose to absorb potential -ffast-math reorder).
//!
//! Build (Android cross):
//!   cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!     --bin microbench_qnn_oppkg_silu_mul_correct
//!
//! Pre-deploy (CRITICAL — run_device.py does NOT auto-rebuild qnn_oppkg cdylib):
//!   cargo build --release -p qnn_oppkg --target aarch64-linux-android
//!   adb push target/aarch64-linux-android/release/libqnn_oppkg.so /data/local/tmp/
//!
//! Run on device:
//!   adb shell "LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:/vendor/lib64 \
//!              /data/local/tmp/microbench_qnn_oppkg_silu_mul_correct"

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_oppkg_silu_mul_correct requires --features qnn");
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

    println!("=== microbench_qnn_oppkg_silu_mul_correct (M1.7) ===\n");
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
    let kernel_src = include_str!("../../kernels/simple_ops.cl");
    let cl_program = Program::builder()
        .devices(device)
        .src(kernel_src)
        .cmplr_opt("-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math")
        .build(&cl_ctx)?;
    let cl_kernel = ocl::core::create_kernel(&cl_program, "kernel_silu_mul_simple")?;

    // ── Path B: QNN-GPU backend + register OpPackage ──────────────────────────
    let gpu_lib = unsafe { Library::new(&backend_lib_path) }?;
    type GetProvidersFn = unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let gp: Symbol<GetProvidersFn> = unsafe { gpu_lib.get(b"QnnInterface_getProviders\0")? };
    let mut provs: *mut *const QnnInterface_t = ptr::null_mut();
    let mut np: c_uint = 0;
    let err = unsafe { gp(&mut provs, &mut np) };
    anyhow::ensure!(err == 0 && np > 0, "GPU getProviders err=0x{:x}", err);
    let v = unsafe { (**provs).__bindgen_anon_1.v2_25 };

    // Logger — NULL callback = platform default to logcat.
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

    // Qwen2.5-1.5b ffn intermediate dim is 8960. Cases mirror typical ffn
    // shapes used in production (rows = decode tokens or prefill chunk).
    let cases: &[(usize, usize)] = &[(1, 5632), (4, 8192), (16, 11008)];
    let mut all_pass = true;

    for &(rows, dim) in cases {
        println!("--- (rows, dim) = ({}, {}) ---", rows, dim);
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
            rows,
            dim,
        );
        match result {
            Ok(max_err) => {
                let pass = max_err < 1e-4;
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
        "\n=== M1.7 verdict: {} ===",
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
    rows: usize,
    dim: usize,
) -> anyhow::Result<f32> {
    use ocl::core::ArgVal;
    use qnn::*;
    use std::ffi::CString;
    use std::ptr;

    // ── Generate identical inputs for both paths ──────────────────────────────
    let total = rows * dim;
    anyhow::ensure!(total % 4 == 0, "total ({}) must be multiple of 4", total);
    let size4 = total / 4;
    let mut host_x = vec![0.0f32; total];
    let mut host_y = vec![0.0f32; total];
    for i in 0..total {
        // Pseudo-random but deterministic. Range ~[-2, 2].
        host_x[i] = (((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5) * 4.0;
        host_y[i] = (((i as f32) * 0.0241 + 0.13).rem_euclid(1.0) - 0.5) * 4.0;
    }

    // ── Path A: raw OpenCL reference (in-place; buf_x is mutated) ─────────────
    let buf_x = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_WRITE,
            total,
            None,
        )?
    };
    let buf_y = unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, total, None)?
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
            &buf_y,
            true,
            0,
            &host_y,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    let size4_i: i32 = size4 as i32;
    // kernel_silu_mul_simple signature: (x, y, size4)
    ocl::core::set_kernel_arg(cl_kernel, 0, ArgVal::mem(&buf_x))?;
    ocl::core::set_kernel_arg(cl_kernel, 1, ArgVal::mem(&buf_y))?;
    ocl::core::set_kernel_arg(cl_kernel, 2, ArgVal::scalar(&size4_i))?;

    let global = [size4, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            cl_kernel,
            1,
            None,
            &global,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    let mut ref_x = vec![0.0f32; total];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_x,
            true,
            0,
            &mut ref_x,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // ── Path B: QNN graph with CustomSiluMul ──────────────────────────────────
    let bytes = (total * 4) as i32;
    let rpc_x = unsafe { rpcmem_alloc(heap_id, flags, bytes) };
    let rpc_y = unsafe { rpcmem_alloc(heap_id, flags, bytes) };
    anyhow::ensure!(
        !rpc_x.is_null() && !rpc_y.is_null(),
        "rpcmem_alloc failed for (rows,dim)=({},{})",
        rows,
        dim
    );

    unsafe {
        std::ptr::copy_nonoverlapping(
            host_x.as_ptr() as *const u8,
            rpc_x as *mut u8,
            bytes as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_y.as_ptr() as *const u8,
            rpc_y as *mut u8,
            bytes as usize,
        );
    }

    let fd_x = unsafe { rpcmem_to_fd(rpc_x) };
    let fd_y = unsafe { rpcmem_to_fd(rpc_y) };

    let mut dims_x: Vec<u32> = vec![rows as u32, dim as u32];
    let mut dims_y: Vec<u32> = vec![rows as u32, dim as u32];
    let mut dims_xout: Vec<u32> = vec![rows as u32, dim as u32];

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
    let mk_tv1 = |ttype, dims_ptr: *mut u32| Qnn_TensorV1_t {
        id: 0,
        name: ptr::null(),
        type_: ttype,
        dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
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

    let name_x = CString::new(format!("sm_x_{}_{}", rows, dim)).unwrap();
    let name_y = CString::new(format!("sm_y_{}_{}", rows, dim)).unwrap();
    let name_xout = CString::new(format!("sm_xout_{}_{}", rows, dim)).unwrap();

    let mut t_x = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                dims_x.as_mut_ptr(),
            ),
        },
    };
    t_x.__bindgen_anon_1.v1.name = name_x.as_ptr();
    let mut t_y = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                dims_y.as_mut_ptr(),
            ),
        },
    };
    t_y.__bindgen_anon_1.v1.name = name_y.as_ptr();
    let mut t_xout = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
                dims_xout.as_mut_ptr(),
            ),
        },
    };
    t_xout.__bindgen_anon_1.v1.name = name_xout.as_ptr();

    let g_name = CString::new(format!("sm_graph_{}_{}", rows, dim)).unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    for (label, t) in [("x", &mut t_x), ("y", &mut t_y), ("xout", &mut t_xout)] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", label, err);
    }

    let op_name = CString::new(format!("sm0_{}_{}", rows, dim)).unwrap();
    let pkg = CString::new("qnn_oppkg").unwrap();
    let op_type = CString::new("CustomSiluMul").unwrap();
    let mut inputs = [t_x, t_y];
    let mut outputs = [t_xout];
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
    // x_inout aliases x: same fd, same host data, distinct mem handle.
    let descs = [
        mk_desc(fd_x, rpc_x, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_x),
        mk_desc(fd_y, rpc_y, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_y),
        mk_desc(
            fd_x,
            rpc_x,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            &dims_xout,
        ),
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

    // Read back the in-place result from rpc_x. The kernel mutated input
    // tensor 0 via OP_INPUT_READWRITE, so rpc_x now holds silu(host_x) * host_y.
    let mut max_abs = 0.0f32;
    unsafe {
        let test_x = std::slice::from_raw_parts(rpc_x as *const f32, total);
        for i in 0..total {
            let d = (test_x[i] - ref_x[i]).abs();
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
