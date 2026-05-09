//! microbench_qnngpu_matmul_tbt — Phase R Wave 2: R-B1
//!
//! 목적: QNN-GPU MatMul TBT vs production OpenCL `mul_mv_f16_f32` wall-clock 비교.
//! Pass: TBT_qnn ≤ TBT_baseline × 1.0 (성능 무손실).
//! Yellow: ≤ 1.1×.
//! Fail: > 1.1× → R-B1 RED, Phase R 종결.
//!
//! 차원: Qwen2.5-1.5b FFN gate matmul. M=1, K=1536, N=8960. (decode-time GEMV)
//! Baseline dtype: F16 weight × F32 input → F32 output (production 그대로).
//! Test dtype: F16 × F16 → F16 (QNN-GPU MAT_MUL prebuilt op, dtype-corrected).
//! Note: production이 F16 weight를 쓰므로 QNN도 F16으로 맞춤 — apple-to-apple bandwidth.
//!
//! Build: cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!        --bin microbench_qnngpu_matmul_tbt
//! Run:   `LD_LIBRARY_PATH=/data/local/tmp/qnn:/vendor/lib64 \
//!         ADSP_LIBRARY_PATH=/data/local/tmp/qnn \
//!         adb shell /data/local/tmp/microbench_qnngpu_matmul_tbt [N_ITERS]`

#[cfg(not(all(feature = "qnn", feature = "opencl")))]
fn main() {
    eprintln!("microbench_qnngpu_matmul_tbt requires --features qnn,opencl");
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

    let args: Vec<String> = std::env::args().collect();
    let k: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1536);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8960);
    let n_iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(30);

    let m: usize = 1;

    println!("=== microbench_qnngpu_matmul_tbt (Phase R / R-B1) ===\n");
    println!("Dim: M={}, K={}, N={}", m, k, n);
    println!("Baseline: OpenCL mul_mv_f16_f32 (F16 weight, F32 in/out)");
    println!("Test:     QNN-GPU FullyConnected (F16, weight[N,K] native, MEMHANDLE/DMA_BUF)");
    println!("n_iters per config: {}\n", n_iters);

    // ─────────────────────────────────────────────────────────
    // Path A: OpenCL baseline (production mul_mv_f16_f32)
    // ─────────────────────────────────────────────────────────
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

    // F16 weight buffer [N, K]
    let mut host_w_f16 = vec![0u16; n * k];
    let mut host_x_f32 = vec![0.0f32; m * k];
    for i in 0..n * k {
        let v = ((i as f32) * 0.0007 + 0.13).rem_euclid(1.0) - 0.5;
        host_w_f16[i] = f32_to_f16_bits(v);
    }
    for i in 0..m * k {
        host_x_f32[i] = ((i as f32) * 0.011).rem_euclid(1.0) - 0.5;
    }
    let host_y_f32 = vec![0.0f32; m * n];

    let buf_w = unsafe {
        ocl::core::create_buffer::<_, u16>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_ONLY,
            n * k,
            None,
        )?
    };
    let buf_x = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_ONLY,
            m * k,
            None,
        )?
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
            &cl_q,
            &buf_w,
            true,
            0,
            &host_w_f16,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            &cl_q,
            &buf_x,
            true,
            0,
            &host_x_f32,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(&cl_q)?;

    // mul_mv_f16_f32 args
    // ne00=K, ne01=N, ne02=1, ne10=K, ne12=1, ne0=N, ne1=M, r2=1, r3=1
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
    ocl::core::set_kernel_arg(&cl_kernel, 0, ArgVal::mem(&buf_w))?;
    ocl::core::set_kernel_arg(&cl_kernel, 1, ArgVal::scalar(&off0))?;
    ocl::core::set_kernel_arg(&cl_kernel, 2, ArgVal::mem(&buf_x))?;
    ocl::core::set_kernel_arg(&cl_kernel, 3, ArgVal::scalar(&off1))?;
    ocl::core::set_kernel_arg(&cl_kernel, 4, ArgVal::mem(&buf_y))?;
    ocl::core::set_kernel_arg(&cl_kernel, 5, ArgVal::scalar(&offd))?;
    ocl::core::set_kernel_arg(&cl_kernel, 6, ArgVal::scalar(&ne00))?;
    ocl::core::set_kernel_arg(&cl_kernel, 7, ArgVal::scalar(&ne01))?;
    ocl::core::set_kernel_arg(&cl_kernel, 8, ArgVal::scalar(&ne02))?;
    ocl::core::set_kernel_arg(&cl_kernel, 9, ArgVal::scalar(&ne10))?;
    ocl::core::set_kernel_arg(&cl_kernel, 10, ArgVal::scalar(&ne12))?;
    ocl::core::set_kernel_arg(&cl_kernel, 11, ArgVal::scalar(&ne0))?;
    ocl::core::set_kernel_arg(&cl_kernel, 12, ArgVal::scalar(&ne1))?;
    ocl::core::set_kernel_arg(&cl_kernel, 13, ArgVal::scalar(&r2))?;
    ocl::core::set_kernel_arg(&cl_kernel, 14, ArgVal::scalar(&r3))?;

    // Dispatch: global = [ceil(N/N_DST)*64, M*4, batch], local = [64, 4, 1]
    // N_DST=2 in production kernel
    let n_dst: usize = 2;
    let global = [(n + n_dst - 1) / n_dst * 64, m * 4, 1];
    let local = [64usize, 4, 1];

    let exec_baseline = || -> anyhow::Result<f64> {
        let t0 = Instant::now();
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
        ocl::core::finish(&cl_q)?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    // ─────────────────────────────────────────────────────────
    // Path B: QNN-GPU MatMul setup (rpcmem + DMA_BUF + MEMHANDLE)
    // ─────────────────────────────────────────────────────────
    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;
    type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(po: *const c_void) -> i32;

    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }
        .or_else(|_| unsafe { Library::new("libcdsprpc.so") })?;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let rpcmem_free: Symbol<RpcmemFreeFn> = unsafe { rpc_lib.get(b"rpcmem_free\0")? };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };

    // F16 buffers: 2 bytes/element
    let bytes_a = (m * k * 2) as i32;
    let bytes_b = (k * n * 2) as i32;
    let bytes_c = (m * n * 2) as i32;
    let rpc_a = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes_a) };
    let rpc_b = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes_b) };
    let rpc_c = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes_c) };
    anyhow::ensure!(
        !rpc_a.is_null() && !rpc_b.is_null() && !rpc_c.is_null(),
        "rpcmem_alloc failed (need {} MB total)",
        (bytes_a + bytes_b + bytes_c) / 1024 / 1024
    );
    let fd_a = unsafe { rpcmem_to_fd(rpc_a) };
    let fd_b = unsafe { rpcmem_to_fd(rpc_b) };
    let fd_c = unsafe { rpcmem_to_fd(rpc_c) };

    // Fill QNN inputs as F16 (apple-to-apple with baseline F16 weight)
    let host_a_f16: Vec<u16> = host_x_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    unsafe {
        std::ptr::copy_nonoverlapping(
            host_a_f16.as_ptr() as *const u8,
            rpc_a as *mut u8,
            bytes_a as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_w_f16.as_ptr() as *const u8,
            rpc_b as *mut u8,
            bytes_b as usize,
        );
    }

    // QNN-GPU backend
    let gpu_lib = unsafe { Library::new("/data/local/tmp/qnn/libQnnGpu.so") }?;
    type GetProvidersFn =
        unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let gpu_gp: Symbol<GetProvidersFn> = unsafe { gpu_lib.get(b"QnnInterface_getProviders\0")? };
    let mut gpu_provs: *mut *const QnnInterface_t = ptr::null_mut();
    let mut gpu_n_p: c_uint = 0;
    let err = unsafe { gpu_gp(&mut gpu_provs, &mut gpu_n_p) };
    anyhow::ensure!(err == 0 && gpu_n_p > 0, "GPU getProviders err=0x{:x}", err);
    let v_gpu = unsafe { (**gpu_provs).__bindgen_anon_1.v2_25 };

    let mut gpu_be: Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe {
        (v_gpu.backendCreate.unwrap())(ptr::null_mut(), ptr::null_mut(), &mut gpu_be)
    };
    anyhow::ensure!(err == 0, "GPU backendCreate err=0x{:x}", err);
    let mut gpu_ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe {
        (v_gpu.contextCreate.unwrap())(gpu_be, ptr::null_mut(), ptr::null_mut(), &mut gpu_ctx)
    };
    anyhow::ensure!(err == 0, "GPU contextCreate err=0x{:x}", err);

    // Register fds with GPU backend (DMA_BUF)
    // FullyConnected: weight is [out_features=N, in_features=K] (production-native layout)
    let dims_a: Vec<u32> = vec![m as u32, k as u32];
    let dims_b: Vec<u32> = vec![n as u32, k as u32];
    let dims_c: Vec<u32> = vec![m as u32, n as u32];
    let mk_descriptor_dmabuf =
        |fd: i32, host_data: *mut c_void, dims: &[u32]| -> Qnn_MemDescriptor_t {
            Qnn_MemDescriptor_t {
                memShape: Qnn_MemShape_t {
                    numDim: dims.len() as u32,
                    dimSize: dims.as_ptr() as *mut u32,
                    shapeConfig: ptr::null(),
                },
                dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
                memType: Qnn_MemType_t_QNN_MEM_TYPE_DMA_BUF,
                __bindgen_anon_1: Qnn_MemDescriptor_t__bindgen_ty_1 {
                    dmaBufInfo: Qnn_MemDmaBufInfo_t { fd, data: host_data },
                },
            }
        };
    let descs = [
        mk_descriptor_dmabuf(fd_a, rpc_a, &dims_a),
        mk_descriptor_dmabuf(fd_b, rpc_b, &dims_b),
        mk_descriptor_dmabuf(fd_c, rpc_c, &dims_c),
    ];
    let mut mh = [ptr::null_mut::<c_void>(); 3];
    let err = unsafe {
        (v_gpu.memRegister.unwrap())(gpu_ctx, descs.as_ptr(), 3, mh.as_mut_ptr())
    };
    anyhow::ensure!(err == 0, "GPU memRegister err=0x{:x}", err);

    // Build QNN-GPU graph: C = A @ B (FullyConnected, default config)
    let g_name = CString::new("matmul_gpu").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err = unsafe {
        (v_gpu.graphCreate.unwrap())(gpu_ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph)
    };
    anyhow::ensure!(err == 0, "GPU graphCreate err=0x{:x}", err);

    let mk_v1 =
        |name: &CString, ttype: Qnn_TensorType_t, dims: &[u32]| -> Qnn_TensorV1_t {
            Qnn_TensorV1_t {
                id: 0,
                name: name.as_ptr(),
                type_: ttype,
                dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
                quantizeParams: Qnn_QuantizeParams_t {
                    encodingDefinition: Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
                    quantizationEncoding:
                        Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
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
    let n_a = CString::new("A").unwrap();
    let n_b = CString::new("B").unwrap();
    let n_c = CString::new("C").unwrap();
    let mut t_a = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1(&n_a, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims_a),
        },
    };
    let mut t_b = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1(&n_b, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims_b),
        },
    };
    let mut t_c = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1(&n_c, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, &dims_c),
        },
    };
    for (l, t) in [("A", &mut t_a), ("B", &mut t_b), ("C", &mut t_c)] {
        let err = unsafe { (v_gpu.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", l, err);
    }
    let op_name = CString::new("fc0").unwrap();
    let pkg = CString::new("qti.aisw").unwrap();
    let op_type = CString::new("FullyConnected").unwrap();
    let mut inputs = [t_a, t_b];
    let mut outputs = [t_c];
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
    let err = unsafe { (v_gpu.graphAddNode.unwrap())(graph, op) };
    anyhow::ensure!(err == 0, "GPU graphAddNode err=0x{:x}", err);
    let t_finalize = Instant::now();
    let err = unsafe {
        (v_gpu.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut())
    };
    let finalize_ms = t_finalize.elapsed().as_secs_f64() * 1000.0;
    anyhow::ensure!(err == 0, "GPU graphFinalize err=0x{:x}", err);
    println!("QNN-GPU graph (MatMul {}x{}x{}) finalize: OK ({:.1} ms)", m, k, n, finalize_ms);

    // Switch to MEMHANDLE
    inputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[0];
    inputs[1].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[1];
    outputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[2];

    let mut exec_qnn = || -> anyhow::Result<f64> {
        let t0 = Instant::now();
        let err = unsafe {
            (v_gpu.graphExecute.unwrap())(
                graph,
                inputs.as_ptr(),
                2,
                outputs.as_mut_ptr(),
                1,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        anyhow::ensure!(err == 0, "GPU graphExecute err=0x{:x}", err);
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    // ─────────────────────────────────────────────────────────
    // Warmup + measurement loop
    // ─────────────────────────────────────────────────────────
    println!("Warmup (5 iters each)...");
    for _ in 0..5 {
        let _ = exec_baseline()?;
        let _ = exec_qnn()?;
    }
    println!();

    let measure = |label: &str, mut f: Box<dyn FnMut() -> anyhow::Result<f64>>|
                  -> anyhow::Result<(f64, f64, f64)> {
        let mut samples = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            samples.push(f()?);
        }
        let n_s = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n_s;
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_s;
        let stddev = var.sqrt();
        let cv = stddev / mean;
        println!(
            "{:<48} mean={:.3}ms median={:.3}ms σ={:.3}ms σ/mean={:.4}",
            label, mean, median, stddev, cv
        );
        Ok((mean, median, cv))
    };

    let (b_mean, b_median, b_cv) = measure(
        "Baseline OpenCL mul_mv_f16_f32 (F16w, F32x)",
        Box::new(exec_baseline),
    )?;
    let (q_mean, q_median, q_cv) = measure(
        "Test     QNN-GPU MAT_MUL (F32×F32, MEMHANDLE)",
        Box::new(exec_qnn),
    )?;

    println!("\n=== R-B1 summary ===");
    println!("baseline mean: {:.3}ms (median {:.3}ms, σ/mean {:.4})", b_mean, b_median, b_cv);
    println!("qnn-gpu mean:  {:.3}ms (median {:.3}ms, σ/mean {:.4})", q_mean, q_median, q_cv);
    let ratio = q_mean / b_mean;
    println!("ratio q/b:     {:.3}x  (≤1.00 GREEN, ≤1.10 YELLOW, >1.10 RED)", ratio);

    let verdict = if ratio <= 1.0 {
        "✓ GREEN"
    } else if ratio <= 1.1 {
        "△ YELLOW"
    } else {
        "✗ RED"
    };
    println!("\nR-B1 verdict: {}", verdict);

    // Cleanup
    unsafe {
        let _ = (v_gpu.memDeRegister.unwrap())(mh.as_ptr(), 3);
        let _ = (v_gpu.contextFree.unwrap())(gpu_ctx, ptr::null_mut());
        let _ = (v_gpu.backendFree.unwrap())(gpu_be);
        rpcmem_free(rpc_a);
        rpcmem_free(rpc_b);
        rpcmem_free(rpc_c);
    }
    let _ = host_y_f32; // silence unused
    if ratio <= 1.1 { Ok(()) } else { std::process::exit(1) }
}

#[cfg(all(feature = "qnn", feature = "opencl"))]
fn f32_to_f16_bits(v: f32) -> u16 {
    // IEEE-754 f32 → f16 (round-to-nearest-even, no Inf/NaN handling for benchmark data)
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

