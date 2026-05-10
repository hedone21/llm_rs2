//! M2.B + M3.4 D-D.2 CustomRope correctness — production OpPackage vs raw OpenCL.
//!
//! Reference: raw OpenCL `kernel_rope_simple` invoked directly (in-place,
//! reads `start_pos` as a scalar kernel arg). The OpPackage uses the **OOP**
//! variant `kernel_rope_simple_oop` which now reads `start_pos` from
//! `pos_buf[0]` (M3.4 D-D.1: SCALAR op params get baked at graph finalize and
//! prevent multi-token decode). Numerically the two paths are identical: the
//! rotation formula and `start_pos` value match per case.
//!
//! Mapping (M3.4):
//!   inputs:  [0]=x_in F32 [seq_len, num_heads, head_dim]
//!            [1]=pos_buf I32 [1] — pos_buf[0] supplies start_pos at execute
//!   outputs: [0]=x_out F32 (claimed)
//!   params:  "theta" FLOAT_32 (build-time const)
//!
//! Cases (seq_len=1, num_heads=32, head_dim=128, theta=10000.0)
//!         × start_pos ∈ {0, 10, 100, 1000}. pos=0 reproduces M2.B GREEN
//! baseline; pos != 0 cases prove the runtime pos buffer drives RoPE rotation.
//!
//! Pass criterion: max_abs_err < 1e-4 (identical kernels → bit-equal output
//! is expected; threshold is loose to absorb potential -ffast-math reorder).

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_oppkg_rope_correct requires --features qnn");
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

    println!("=== microbench_qnn_oppkg_rope_correct (M2.B + M3.4 D-D.2) ===\n");
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
    let cl_kernel = ocl::core::create_kernel(&cl_program, "kernel_rope_simple")?;

    // ── Path B: QNN-GPU backend + register OpPackage ──────────────────────────
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

    // Llama 3.2 1B q-projection: seq_len=1, num_heads=32, head_dim=128 is too
    // wide; Llama 1B uses head_dim=64. For broader coverage we exercise a
    // Qwen-like (32, 128) shape since head_dim is the only RoPE-specific
    // dimension and the kernel handles both.
    let seq_len: usize = 1;
    let num_heads: usize = 32;
    let head_dim: usize = 128;
    let theta: f32 = 10000.0;
    let start_positions: &[i32] = &[0, 10, 100, 1000];
    let mut all_pass = true;

    for &start_pos in start_positions {
        println!(
            "--- (seq_len, num_heads, head_dim) = ({}, {}, {}), start_pos = {}, theta = {} ---",
            seq_len, num_heads, head_dim, start_pos, theta
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
            seq_len,
            num_heads,
            head_dim,
            start_pos,
            theta,
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
        "\n=== M3.4 D-D.2 verdict: {} ===",
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
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    start_pos: i32,
    theta: f32,
) -> anyhow::Result<f32> {
    use ocl::core::ArgVal;
    use qnn::*;
    use std::ffi::CString;
    use std::ptr;

    // ── Generate identical inputs for both paths ──────────────────────────────
    anyhow::ensure!(head_dim % 2 == 0, "head_dim ({}) must be even", head_dim);
    let total = seq_len * num_heads * head_dim;
    let pairs = seq_len * num_heads * (head_dim / 2);
    let mut host_x = vec![0.0f32; total];
    for i in 0..total {
        host_x[i] = (((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5) * 4.0;
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
    }
    ocl::core::finish(cl_q)?;

    let head_dim_i: i32 = head_dim as i32;
    let num_heads_i: i32 = num_heads as i32;
    let seq_len_i: i32 = seq_len as i32;
    // kernel_rope_simple signature: (x, head_dim, num_heads, seq_len, start_pos, theta)
    ocl::core::set_kernel_arg(cl_kernel, 0, ArgVal::mem(&buf_x))?;
    ocl::core::set_kernel_arg(cl_kernel, 1, ArgVal::scalar(&head_dim_i))?;
    ocl::core::set_kernel_arg(cl_kernel, 2, ArgVal::scalar(&num_heads_i))?;
    ocl::core::set_kernel_arg(cl_kernel, 3, ArgVal::scalar(&seq_len_i))?;
    ocl::core::set_kernel_arg(cl_kernel, 4, ArgVal::scalar(&start_pos))?;
    ocl::core::set_kernel_arg(cl_kernel, 5, ArgVal::scalar(&theta))?;

    let global = [pairs, 1, 1];
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

    // ── Path B: QNN graph with CustomRope ─────────────────────────────────────
    // OOP variant + pos buffer: distinct rpcmem allocations for input (x_in),
    // pos_buf (M3.4 D-D.1), and output (x_out).
    let bytes = (total * 4) as i32;
    let pos_bytes: i32 = 4; // INT_32 [1]
    let rpc_x = unsafe { rpcmem_alloc(heap_id, flags, bytes) };
    let rpc_pos = unsafe { rpcmem_alloc(heap_id, flags, pos_bytes) };
    let rpc_xout = unsafe { rpcmem_alloc(heap_id, flags, bytes) };
    anyhow::ensure!(
        !rpc_x.is_null() && !rpc_pos.is_null() && !rpc_xout.is_null(),
        "rpcmem_alloc failed for total={}",
        total
    );

    unsafe {
        std::ptr::copy_nonoverlapping(
            host_x.as_ptr() as *const u8,
            rpc_x as *mut u8,
            bytes as usize,
        );
        // Write the runtime pos value into pos_buf[0] (M3.4 D-D.1).
        *(rpc_pos as *mut i32) = start_pos;
        // Zero-init x_out so any unwritten elements would be obvious.
        std::ptr::write_bytes(rpc_xout as *mut u8, 0, bytes as usize);
    }

    let fd_x = unsafe { rpcmem_to_fd(rpc_x) };
    let fd_pos = unsafe { rpcmem_to_fd(rpc_pos) };
    let fd_xout = unsafe { rpcmem_to_fd(rpc_xout) };

    let mut dims_x: Vec<u32> = vec![seq_len as u32, num_heads as u32, head_dim as u32];
    let mut dims_pos: Vec<u32> = vec![1u32];
    let mut dims_xout: Vec<u32> = vec![seq_len as u32, num_heads as u32, head_dim as u32];

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
    let mk_tv1 = |ttype, dtype: Qnn_DataType_t, rank: u32, dims_ptr: *mut u32| Qnn_TensorV1_t {
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

    let name_x = CString::new(format!("rope_x_{}_{}_{}", num_heads, head_dim, start_pos)).unwrap();
    let name_pos =
        CString::new(format!("rope_pos_{}_{}_{}", num_heads, head_dim, start_pos)).unwrap();
    let name_xout = CString::new(format!(
        "rope_xout_{}_{}_{}",
        num_heads, head_dim, start_pos
    ))
    .unwrap();

    let mut t_x = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                3,
                dims_x.as_mut_ptr(),
            ),
        },
    };
    t_x.__bindgen_anon_1.v1.name = name_x.as_ptr();
    let mut t_pos = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                Qnn_DataType_t_QNN_DATATYPE_INT_32,
                1,
                dims_pos.as_mut_ptr(),
            ),
        },
    };
    t_pos.__bindgen_anon_1.v1.name = name_pos.as_ptr();
    let mut t_xout = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                3,
                dims_xout.as_mut_ptr(),
            ),
        },
    };
    t_xout.__bindgen_anon_1.v1.name = name_xout.as_ptr();

    let g_name = CString::new(format!("rope_graph_{}", start_pos)).unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    for (label, t) in [("x", &mut t_x), ("pos", &mut t_pos), ("xout", &mut t_xout)] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", label, err);
    }

    // Op params: scalar FLOAT_32 "theta" only. `start_pos` is now supplied via
    // the pos_buf input tensor at execute time (M3.4 D-D.1).
    let pname_theta = CString::new("theta").unwrap();
    let mut params = [Qnn_Param_t {
        paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
        name: pname_theta.as_ptr(),
        __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
            scalarParam: Qnn_Scalar_t {
                dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 { floatValue: theta },
            },
        },
    }];

    let op_name = CString::new(format!("rope0_{}", start_pos)).unwrap();
    let pkg = CString::new("qnn_oppkg").unwrap();
    let op_type = CString::new("CustomRope").unwrap();
    let mut inputs = [t_x, t_pos];
    let mut outputs = [t_xout];
    let op = Qnn_OpConfig_t {
        version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
        __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
            v1: Qnn_OpConfigV1_t {
                name: op_name.as_ptr(),
                packageName: pkg.as_ptr(),
                typeName: op_type.as_ptr(),
                numOfParams: 1,
                params: params.as_mut_ptr(),
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
    // OOP: input, pos_buf, and output are distinct rpcmem buffers.
    let descs = [
        mk_desc(fd_x, rpc_x, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_x),
        mk_desc(
            fd_pos,
            rpc_pos,
            Qnn_DataType_t_QNN_DATATYPE_INT_32,
            &dims_pos,
        ),
        mk_desc(
            fd_xout,
            rpc_xout,
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

    // Read back the OOP result from rpc_xout. The kernel wrote rope(host_x)
    // into x_out (a distinct buffer from x_in / rpc_x).
    let mut max_abs = 0.0f32;
    unsafe {
        let test_x = std::slice::from_raw_parts(rpc_xout as *const f32, total);
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
