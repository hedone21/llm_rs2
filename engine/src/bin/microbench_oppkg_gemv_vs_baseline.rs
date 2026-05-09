//! microbench_oppkg_gemv_vs_baseline — Phase R Wave 2: production GEMV inside OpPackage
//!
//! 비교: 직접 OpenCL `mul_mv_f16_f32` (R-B1 baseline) vs 같은 kernel을 OpPackage에
//! 등록해 QNN-GPU runtime이 build/launch (CustomMatMul op).
//!
//! 차원: Qwen2.5-1.5b FFN gate (M=1, K=1536, N=8960)
//! Pass: OpPackage TBT ≤ baseline × 1.10 (성능 무손실 또는 minor overhead)
//!
//! Build: cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!        --bin microbench_oppkg_gemv_vs_baseline

#[cfg(not(all(feature = "qnn", feature = "opencl")))]
fn main() {
    eprintln!("requires --features qnn,opencl");
    std::process::exit(2);
}

#[cfg(all(feature = "qnn", feature = "opencl"))]
#[allow(non_snake_case, non_camel_case_types, non_upper_case_globals, dead_code)]
mod qnn {
    include!(concat!(env!("OUT_DIR"), "/qnn_bindings.rs"));
}

#[cfg(all(feature = "qnn", feature = "opencl"))]
fn main() -> anyhow::Result<()> {
    use libloading::{Library, Symbol};
    use std::ffi::{CString, c_void};
    use std::os::raw::c_uint;
    use std::ptr;
    use std::time::Instant;

    use qnn::*;

    const PKG_PATH: &str = "/data/local/tmp/libqnn_oppkg_poc.so";
    const PKG_PROVIDER: &str = "QnnOpPackage_InitInterface";
    const PKG_TARGET: &str = "GPU";
    let n_iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(30);

    let m: usize = 1;
    let k: usize = 1536;
    let n: usize = 8960;

    println!("=== microbench_oppkg_gemv_vs_baseline ===");
    println!("Dim M={} K={} N={} (Qwen FFN gate)", m, k, n);

    // ── Baseline OpenCL (production mul_mv_f16_f32) ──
    use ocl::core::ArgVal;
    use ocl::{Context, Device, Platform, Program, Queue};
    let platform = Platform::default();
    let device = Device::first(platform)?;
    let cl_ctx = Context::builder().platform(platform).devices(device).build()?;
    let cl_q = Queue::new(&cl_ctx, device, None)?;
    let kernel_src = include_str!("../../kernels/mul_mv_f16_f32.cl");
    let cl_program = Program::builder()
        .devices(device)
        .src(kernel_src)
        .cmplr_opt("-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math")
        .build(&cl_ctx)?;
    let cl_kernel = ocl::core::create_kernel(&cl_program, "kernel_mul_mat_f16_f32")?;

    let mut host_w_f16 = vec![0u16; n * k];
    let mut host_x_f32 = vec![0.0f32; m * k];
    for i in 0..n * k {
        let v = ((i as f32) * 0.0007 + 0.13).rem_euclid(1.0) - 0.5;
        host_w_f16[i] = f32_to_f16_bits(v);
    }
    for i in 0..m * k {
        host_x_f32[i] = ((i as f32) * 0.011).rem_euclid(1.0) - 0.5;
    }

    let buf_w = unsafe {
        ocl::core::create_buffer::<_, u16>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, n * k, None)?
    };
    let buf_x = unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, m * k, None)?
    };
    let buf_y = unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_WRITE, m * n, None)?
    };
    unsafe {
        ocl::core::enqueue_write_buffer(&cl_q, &buf_w, true, 0, &host_w_f16, None::<ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        ocl::core::enqueue_write_buffer(&cl_q, &buf_x, true, 0, &host_x_f32, None::<ocl::core::Event>, None::<&mut ocl::core::Event>)?;
    }
    ocl::core::finish(&cl_q)?;

    let ne00 = k as i32;
    let ne01 = n as i32;
    let ne0 = n as i32;
    let ne1 = m as i32;
    let one_i32: i32 = 1;
    let zero_u64: u64 = 0;
    ocl::core::set_kernel_arg(&cl_kernel, 0, ArgVal::mem(&buf_w))?;
    ocl::core::set_kernel_arg(&cl_kernel, 1, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(&cl_kernel, 2, ArgVal::mem(&buf_x))?;
    ocl::core::set_kernel_arg(&cl_kernel, 3, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(&cl_kernel, 4, ArgVal::mem(&buf_y))?;
    ocl::core::set_kernel_arg(&cl_kernel, 5, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(&cl_kernel, 6, ArgVal::scalar(&ne00))?;
    ocl::core::set_kernel_arg(&cl_kernel, 7, ArgVal::scalar(&ne01))?;
    ocl::core::set_kernel_arg(&cl_kernel, 8, ArgVal::scalar(&one_i32))?;
    ocl::core::set_kernel_arg(&cl_kernel, 9, ArgVal::scalar(&ne00))?;
    ocl::core::set_kernel_arg(&cl_kernel, 10, ArgVal::scalar(&one_i32))?;
    ocl::core::set_kernel_arg(&cl_kernel, 11, ArgVal::scalar(&ne0))?;
    ocl::core::set_kernel_arg(&cl_kernel, 12, ArgVal::scalar(&ne1))?;
    ocl::core::set_kernel_arg(&cl_kernel, 13, ArgVal::scalar(&one_i32))?;
    ocl::core::set_kernel_arg(&cl_kernel, 14, ArgVal::scalar(&one_i32))?;
    let global = [(n + 1) / 2 * 64, m * 4, 1];
    let local = [64usize, 4, 1];
    let exec_baseline = || -> anyhow::Result<f64> {
        let t0 = Instant::now();
        unsafe {
            ocl::core::enqueue_kernel(&cl_q, &cl_kernel, 3, None, &global, Some(local),
                None::<&ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        }
        ocl::core::finish(&cl_q)?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    // ── OpPackage path: register, build graph, MEMHANDLE rpcmem ──
    let gpu_lib = unsafe { Library::new("/data/local/tmp/qnn/libQnnGpu.so") }?;
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
    let err = unsafe {
        (v.backendRegisterOpPackage.unwrap())(be, pkg_path.as_ptr(), pkg_provider.as_ptr(), pkg_target.as_ptr())
    };
    anyhow::ensure!(err == 0, "registerOpPackage err=0x{:x}", err);
    println!("registerOpPackage: OK");

    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe { (v.contextCreate.unwrap())(be, ptr::null_mut(), ptr::null_mut(), &mut ctx) };
    anyhow::ensure!(err == 0, "contextCreate err=0x{:x}", err);

    let g_name = CString::new("oppkg_gemv").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err = unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    // Tensors: weight [N, K] F16, x [M, K] F32, y [M, N] F32
    let dims_w: Vec<u32> = vec![n as u32, k as u32];
    let dims_x: Vec<u32> = vec![m as u32, k as u32];
    let dims_y: Vec<u32> = vec![m as u32, n as u32];
    let mk_v1 = |name: &CString, ttype: Qnn_TensorType_t, dt: Qnn_DataType_t, dims: &[u32]| -> Qnn_TensorV1_t {
        Qnn_TensorV1_t {
            id: 0,
            name: name.as_ptr(),
            type_: ttype,
            dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            dataType: dt,
            quantizeParams: Qnn_QuantizeParams_t {
                encodingDefinition: Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
                quantizationEncoding: Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
                __bindgen_anon_1: Qnn_QuantizeParams_t__bindgen_ty_1 {
                    scaleOffsetEncoding: Qnn_ScaleOffset_t { scale: 0.0, offset: 0 },
                },
            },
            rank: dims.len() as u32,
            dimensions: dims.as_ptr() as *mut u32,
            memType: Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
            __bindgen_anon_1: Qnn_TensorV1_t__bindgen_ty_1 {
                clientBuf: Qnn_ClientBuffer_t { data: ptr::null_mut(), dataSize: 0 },
            },
        }
    };
    let n_w = CString::new("W").unwrap();
    let n_x = CString::new("X").unwrap();
    let n_y = CString::new("Y").unwrap();
    let mut t_w = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1(&n_w, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, &dims_w),
        },
    };
    let mut t_x = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1(&n_x, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_x),
        },
    };
    let mut t_y = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1(&n_y, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_y),
        },
    };
    for (l, t) in [("W", &mut t_w), ("X", &mut t_x), ("Y", &mut t_y)] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", l, err);
    }
    let op_name = CString::new("mm0").unwrap();
    let pkg = CString::new("qnn_oppkg_poc").unwrap();
    let op_type = CString::new("CustomMatMul").unwrap();
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
    println!("graph finalize: OK");

    // rpcmem
    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;
    type AllocFn = unsafe extern "C" fn(i32, u32, i32) -> *mut c_void;
    type ToFdFn = unsafe extern "C" fn(*const c_void) -> i32;
    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }?;
    let rpc_alloc: Symbol<AllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let rpc_to_fd: Symbol<ToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };
    let bw = (n * k * 2) as i32;
    let bx = (m * k * 4) as i32;
    let by = (m * n * 4) as i32;
    let rpc_w = unsafe { rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bw) };
    let rpc_x = unsafe { rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bx) };
    let rpc_y = unsafe { rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, by) };
    anyhow::ensure!(!rpc_w.is_null() && !rpc_x.is_null() && !rpc_y.is_null());
    let fd_w = unsafe { rpc_to_fd(rpc_w) };
    let fd_x = unsafe { rpc_to_fd(rpc_x) };
    let fd_y = unsafe { rpc_to_fd(rpc_y) };
    unsafe {
        std::ptr::copy_nonoverlapping(host_w_f16.as_ptr() as *const u8, rpc_w as *mut u8, bw as usize);
        std::ptr::copy_nonoverlapping(host_x_f32.as_ptr() as *const u8, rpc_x as *mut u8, bx as usize);
    }
    let mk_desc = |fd: i32, host: *mut c_void, dt: Qnn_DataType_t, dims: &[u32]| -> Qnn_MemDescriptor_t {
        Qnn_MemDescriptor_t {
            memShape: Qnn_MemShape_t {
                numDim: dims.len() as u32,
                dimSize: dims.as_ptr() as *mut u32,
                shapeConfig: ptr::null(),
            },
            dataType: dt,
            memType: Qnn_MemType_t_QNN_MEM_TYPE_DMA_BUF,
            __bindgen_anon_1: Qnn_MemDescriptor_t__bindgen_ty_1 {
                dmaBufInfo: Qnn_MemDmaBufInfo_t { fd, data: host },
            },
        }
    };
    let descs = [
        mk_desc(fd_w, rpc_w, Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, &dims_w),
        mk_desc(fd_x, rpc_x, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_x),
        mk_desc(fd_y, rpc_y, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_y),
    ];
    let mut mh = [ptr::null_mut::<c_void>(); 3];
    let err = unsafe { (v.memRegister.unwrap())(ctx, descs.as_ptr(), 3, mh.as_mut_ptr()) };
    anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);

    inputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[0];
    inputs[1].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[1];
    outputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[2];

    let mut exec_oppkg = || -> anyhow::Result<f64> {
        let t0 = Instant::now();
        let err = unsafe {
            (v.graphExecute.unwrap())(graph, inputs.as_ptr(), 2, outputs.as_mut_ptr(), 1, ptr::null_mut(), ptr::null_mut())
        };
        anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    // Warmup
    println!("Warmup (5 iters)...");
    for _ in 0..5 {
        let _ = exec_baseline()?;
        let _ = exec_oppkg()?;
    }

    // Measure
    let measure = |label: &str, mut f: Box<dyn FnMut() -> anyhow::Result<f64>>| -> anyhow::Result<(f64, f64)> {
        let mut s: Vec<f64> = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            s.push(f()?);
        }
        let mean = s.iter().sum::<f64>() / n_iters as f64;
        let mut sorted = s.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let var = s.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_iters as f64;
        let cv = var.sqrt() / mean;
        println!("{:<40} mean={:.3}ms median={:.3}ms σ/mean={:.4}", label, mean, median, cv);
        Ok((mean, median))
    };
    let (b_mean, b_median) = measure("baseline OpenCL mul_mv_f16_f32", Box::new(exec_baseline))?;
    let (q_mean, q_median) = measure("OpPackage CustomMatMul (same kernel)", Box::new(exec_oppkg))?;

    println!("\n=== summary ===");
    let mean_ratio = q_mean / b_mean;
    let med_ratio = q_median / b_median;
    println!("mean ratio    q/b = {:.3}x (≤1.10 PASS)", mean_ratio);
    println!("median ratio  q/b = {:.3}x", med_ratio);

    // Correctness sanity
    let mut max_abs = 0.0f32;
    unsafe {
        let y_slice = std::slice::from_raw_parts(rpc_y as *const f32, m * n);
        // baseline result still in buf_y; read back
        let mut baseline_y = vec![0.0f32; m * n];
        ocl::core::enqueue_read_buffer(&cl_q, &buf_y, true, 0, &mut baseline_y, None::<ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        for i in 0..m * n {
            let d = (y_slice[i] - baseline_y[i]).abs();
            if d > max_abs { max_abs = d; }
        }
    }
    println!("correctness  max_abs(qnn vs cl) = {:.6}", max_abs);

    let pass = mean_ratio <= 1.10;
    println!("Verdict: {}", if pass { "✓ PASS — OpPackage performance preserved" } else { "✗ FAIL" });

    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), 3);
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(be);
    }
    if pass { Ok(()) } else { std::process::exit(1) }
}

#[cfg(all(feature = "qnn", feature = "opencl"))]
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 { return sign << 15; }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 { return sign << 15; }
    if new_exp >= 31 { return (sign << 15) | (0x1f << 10); }
    let new_mant = (mant >> 13) as u16;
    (sign << 15) | ((new_exp as u16) << 10) | new_mant
}
