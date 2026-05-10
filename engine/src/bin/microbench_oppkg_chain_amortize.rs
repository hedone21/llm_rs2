//! microbench_oppkg_chain_amortize — OpPackage graphExecute overhead 분산 분석
//!
//! N개의 MatMul을 chain으로 쌓은 단일 graph를 measure하여 per-op cost가
//! 어떻게 N에 따라 변하는지 본다. graphExecute 자체의 fixed overhead와
//! per-op cost를 분리.
//!
//! Chain: y_0 = x; y_{i+1} = W @ y_i  (square M=1, K=N=1024, F16 weight)
//!
//! Compare:
//!   - raw OpenCL: N enqueue_kernel + 1 finish
//!   - OpPackage:  graphExecute 1번 (graph 안에 N op)
//!
//! N values: 1, 4, 16, 64
//!
//! Build: cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!        --bin microbench_oppkg_chain_amortize

#[cfg(not(all(feature = "qnn", feature = "opencl")))]
fn main() {
    eprintln!("requires --features qnn,opencl");
    std::process::exit(2);
}

#[cfg(all(feature = "qnn", feature = "opencl"))]
#[allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code
)]
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

    let m: usize = 1;
    let k: usize = 1024;
    let n: usize = 1024; // square so chain dim matches
    let n_iters: usize = 30;
    let chain_lengths: [usize; 4] = [1, 4, 16, 64];

    println!("=== microbench_oppkg_chain_amortize ===");
    println!(
        "Per-op MatMul: M={} K={} N={} (square, F16w/F32x → F32y)\n",
        m, k, n
    );

    use ocl::core::ArgVal;
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

    let mut host_w_f16 = vec![0u16; n * k];
    for i in 0..n * k {
        let v = ((i as f32) * 0.0007 + 0.13).rem_euclid(1.0) - 0.5;
        host_w_f16[i] = f32_to_f16_bits(v);
    }
    let host_x_f32: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.011).rem_euclid(1.0) - 0.5)
        .collect();

    let buf_w = unsafe {
        ocl::core::create_buffer::<_, u16>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, n * k, None)?
    };
    unsafe {
        ocl::core::enqueue_write_buffer(
            &cl_q,
            &buf_w,
            true,
            0,
            &host_w_f16,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }

    // Set static kernel args
    let ne00 = k as i32;
    let ne01 = n as i32;
    let ne0 = n as i32;
    let ne1 = m as i32;
    let one_i32: i32 = 1;
    let zero_u64: u64 = 0;
    ocl::core::set_kernel_arg(&cl_kernel, 0, ArgVal::mem(&buf_w))?;
    ocl::core::set_kernel_arg(&cl_kernel, 1, ArgVal::scalar(&zero_u64))?;
    // arg 2 (x) and 4 (y) set per chain step
    ocl::core::set_kernel_arg(&cl_kernel, 3, ArgVal::scalar(&zero_u64))?;
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

    // QNN-GPU backend setup (shared across all N values)
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
        (v.backendRegisterOpPackage.unwrap())(
            be,
            pkg_path.as_ptr(),
            pkg_provider.as_ptr(),
            pkg_target.as_ptr(),
        )
    };
    anyhow::ensure!(err == 0, "registerOpPackage err=0x{:x}", err);

    println!(
        "{:<6} {:>16} {:>16} {:>14}",
        "N_op", "raw_ms (mean)", "oppkg_ms (mean)", "ratio q/b"
    );
    println!("{}", "-".repeat(60));

    for &n_op in chain_lengths.iter() {
        // ── Raw OpenCL chain ──
        // Buffers: ping-pong y_a, y_b
        let buf_ya = unsafe {
            ocl::core::create_buffer::<_, f32>(
                cl_ctx.as_core(),
                ocl::core::MEM_READ_WRITE,
                m * n,
                None,
            )?
        };
        let buf_yb = unsafe {
            ocl::core::create_buffer::<_, f32>(
                cl_ctx.as_core(),
                ocl::core::MEM_READ_WRITE,
                m * n,
                None,
            )?
        };
        // Initial y_a = x
        let mut x_padded = vec![0.0f32; m * n.max(k)];
        for i in 0..m * k {
            x_padded[i] = host_x_f32[i];
        }
        unsafe {
            ocl::core::enqueue_write_buffer(
                &cl_q,
                &buf_ya,
                true,
                0,
                &x_padded,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        ocl::core::finish(&cl_q)?;

        let exec_raw = || -> anyhow::Result<f64> {
            let t0 = Instant::now();
            let (mut a, mut b) = (&buf_ya, &buf_yb);
            for _ in 0..n_op {
                ocl::core::set_kernel_arg(&cl_kernel, 2, ArgVal::mem(a))?;
                ocl::core::set_kernel_arg(&cl_kernel, 4, ArgVal::mem(b))?;
                unsafe {
                    ocl::core::enqueue_kernel(
                        &cl_q,
                        &cl_kernel,
                        3,
                        None,
                        &global,
                        Some(local),
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
                }
                std::mem::swap(&mut a, &mut b);
            }
            ocl::core::finish(&cl_q)?;
            Ok(t0.elapsed().as_secs_f64() * 1000.0)
        };

        // ── OpPackage chain graph ──
        let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
        let err =
            unsafe { (v.contextCreate.unwrap())(be, ptr::null_mut(), ptr::null_mut(), &mut ctx) };
        anyhow::ensure!(err == 0);
        let g_name = CString::new(format!("chain_{}", n_op)).unwrap();
        let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
        let err =
            unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
        anyhow::ensure!(err == 0);

        let dims_w: Vec<u32> = vec![n as u32, k as u32];
        let dims_v: Vec<u32> = vec![m as u32, n as u32]; // square so y dim same
        let mk_v1 = |name: &CString,
                     ttype: Qnn_TensorType_t,
                     dt: Qnn_DataType_t,
                     dims: &[u32]|
         -> Qnn_TensorV1_t {
            Qnn_TensorV1_t {
                id: 0,
                name: name.as_ptr(),
                type_: ttype,
                dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                dataType: dt,
                quantizeParams: Qnn_QuantizeParams_t {
                    encodingDefinition: Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
                    quantizationEncoding:
                        Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
                    __bindgen_anon_1: Qnn_QuantizeParams_t__bindgen_ty_1 {
                        scaleOffsetEncoding: Qnn_ScaleOffset_t {
                            scale: 0.0,
                            offset: 0,
                        },
                    },
                },
                rank: dims.len() as u32,
                dimensions: dims.as_ptr() as *mut u32,
                memType: Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
                __bindgen_anon_1: Qnn_TensorV1_t__bindgen_ty_1 {
                    clientBuf: Qnn_ClientBuffer_t {
                        data: ptr::null_mut(),
                        dataSize: 0,
                    },
                },
            }
        };

        // Persistent storage for tensor names (1 W + (n_op+1) intermediates)
        let n_w = CString::new("W").unwrap();
        let mut int_names: Vec<CString> = Vec::with_capacity(n_op + 1);
        for i in 0..=n_op {
            int_names.push(CString::new(format!("y{}", i)).unwrap());
        }

        // Create W tensor (input to all ops)
        let mut t_w = Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_v1(
                    &n_w,
                    Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                    Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
                    &dims_w,
                ),
            },
        };
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, &mut t_w) };
        anyhow::ensure!(err == 0);

        // Create y0..yN tensors
        let mut y_tensors: Vec<Qnn_Tensor_t> = Vec::with_capacity(n_op + 1);
        for i in 0..=n_op {
            let ttype = if i == 0 {
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE
            } else if i == n_op {
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ
            } else {
                Qnn_TensorType_t_QNN_TENSOR_TYPE_NATIVE
            };
            // y_i dim: i==0 uses [M,K], rest [M,N]. Square so dims_v works for all.
            let dims = if i == 0 {
                vec![m as u32, k as u32]
            } else {
                dims_v.clone()
            };
            let mut t = Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_v1(
                        &int_names[i],
                        ttype,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                        &dims,
                    ),
                },
            };
            let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, &mut t) };
            anyhow::ensure!(err == 0);
            y_tensors.push(t);
        }

        // Add n_op nodes
        let pkg = CString::new("qnn_oppkg_poc").unwrap();
        let op_type = CString::new("CustomMatMul").unwrap();
        let mut op_names: Vec<CString> = Vec::with_capacity(n_op);
        for i in 0..n_op {
            op_names.push(CString::new(format!("mm_{}", i)).unwrap());
        }
        // Need to keep input/output tensor arrays alive during graphAddNode
        let mut inputs_holder: Vec<[Qnn_Tensor_t; 2]> = Vec::with_capacity(n_op);
        let mut outputs_holder: Vec<[Qnn_Tensor_t; 1]> = Vec::with_capacity(n_op);
        for i in 0..n_op {
            inputs_holder.push([t_w, y_tensors[i]]);
            outputs_holder.push([y_tensors[i + 1]]);
        }
        for i in 0..n_op {
            let op = Qnn_OpConfig_t {
                version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
                __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
                    v1: Qnn_OpConfigV1_t {
                        name: op_names[i].as_ptr(),
                        packageName: pkg.as_ptr(),
                        typeName: op_type.as_ptr(),
                        numOfParams: 0,
                        params: ptr::null_mut(),
                        numOfInputs: 2,
                        inputTensors: inputs_holder[i].as_mut_ptr(),
                        numOfOutputs: 1,
                        outputTensors: outputs_holder[i].as_mut_ptr(),
                    },
                },
            };
            let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
            anyhow::ensure!(err == 0, "graphAddNode[{}] err=0x{:x}", i, err);
        }
        let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
        anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);

        // rpcmem: W + y0 + y_n_op (only graph endpoints need MEMHANDLE)
        const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
        const RPCMEM_DEFAULT_FLAGS: u32 = 1;
        type AllocFn = unsafe extern "C" fn(i32, u32, i32) -> *mut c_void;
        type ToFdFn = unsafe extern "C" fn(*const c_void) -> i32;
        let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }?;
        let rpc_alloc: Symbol<AllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
        let rpc_to_fd: Symbol<ToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };
        let bw = (n * k * 2) as i32;
        let by0 = (m * k * 4) as i32;
        let byn = (m * n * 4) as i32;
        let rpc_w = unsafe { rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bw) };
        let rpc_y0 = unsafe { rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, by0) };
        let rpc_yn = unsafe { rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, byn) };
        anyhow::ensure!(!rpc_w.is_null() && !rpc_y0.is_null() && !rpc_yn.is_null());
        let fd_w = unsafe { rpc_to_fd(rpc_w) };
        let fd_y0 = unsafe { rpc_to_fd(rpc_y0) };
        let fd_yn = unsafe { rpc_to_fd(rpc_yn) };
        unsafe {
            std::ptr::copy_nonoverlapping(
                host_w_f16.as_ptr() as *const u8,
                rpc_w as *mut u8,
                bw as usize,
            );
            std::ptr::copy_nonoverlapping(
                host_x_f32.as_ptr() as *const u8,
                rpc_y0 as *mut u8,
                by0 as usize,
            );
        }
        let mk_desc =
            |fd: i32, host: *mut c_void, dt: Qnn_DataType_t, dims: &[u32]| -> Qnn_MemDescriptor_t {
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
        let dims_y0: Vec<u32> = vec![m as u32, k as u32];
        let dims_yn: Vec<u32> = vec![m as u32, n as u32];
        let descs = [
            mk_desc(fd_w, rpc_w, Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, &dims_w),
            mk_desc(
                fd_y0,
                rpc_y0,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                &dims_y0,
            ),
            mk_desc(
                fd_yn,
                rpc_yn,
                Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                &dims_yn,
            ),
        ];
        let mut mh = [ptr::null_mut::<c_void>(); 3];
        let err = unsafe { (v.memRegister.unwrap())(ctx, descs.as_ptr(), 3, mh.as_mut_ptr()) };
        anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);

        // Apply MEMHANDLE on W, y0, yn
        let mut t_w_mh = t_w;
        t_w_mh.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        t_w_mh.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[0];
        let mut t_y0_mh = y_tensors[0];
        t_y0_mh.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        t_y0_mh.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[1];
        let mut t_yn_mh = y_tensors[n_op];
        t_yn_mh.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        t_yn_mh.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[2];

        // graphExecute takes input tensors (graph inputs only) and output tensors (graph outputs).
        // Graph inputs: W, y0. Graph outputs: y_n_op.
        let exec_inputs = [t_w_mh, t_y0_mh];
        let mut exec_outputs = [t_yn_mh];

        let mut exec_oppkg = || -> anyhow::Result<f64> {
            let t0 = Instant::now();
            let err = unsafe {
                (v.graphExecute.unwrap())(
                    graph,
                    exec_inputs.as_ptr(),
                    2,
                    exec_outputs.as_mut_ptr(),
                    1,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            };
            anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);
            Ok(t0.elapsed().as_secs_f64() * 1000.0)
        };

        // Warmup
        for _ in 0..5 {
            let _ = exec_raw()?;
            let _ = exec_oppkg()?;
        }
        // Measure
        let mut raw_s = Vec::with_capacity(n_iters);
        let mut q_s = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            raw_s.push(exec_raw()?);
            q_s.push(exec_oppkg()?);
        }
        let raw_mean = raw_s.iter().sum::<f64>() / n_iters as f64;
        let q_mean = q_s.iter().sum::<f64>() / n_iters as f64;
        let ratio = q_mean / raw_mean;
        println!(
            "{:<6} {:>16.3} {:>16.3} {:>14.3}",
            n_op, raw_mean, q_mean, ratio
        );

        // Cleanup
        unsafe {
            let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), 3);
            let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        }
    }

    println!("\n=== Per-op cost analysis ===");
    println!("If overhead amortizes: ratio q/b decreases as N grows.");
    println!("If overhead is per-op: ratio stays ~constant.");

    unsafe {
        let _ = (v.backendFree.unwrap())(be);
    }
    Ok(())
}

#[cfg(all(feature = "qnn", feature = "opencl"))]
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
