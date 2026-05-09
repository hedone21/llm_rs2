//! microbench_htp_gpu_matmul_concurrent — Phase 32b-2: R3
//!
//! 목적: HTP MatMul (DDR-heavy) + GPU GEMV (DDR-heavy) 동시 실행 wall-clock.
//! Phase 10 C3는 compute-bound (busy loop)였지만 여기는 양쪽 모두 DDR-heavy.
//! HeteroInfer aggregate BW 패턴 검증.
//!
//! Configs (n=30):
//!   C1: GPU only (GEMV)
//!   C2: HTP only (MatMul)
//!   C3: GPU + HTP simultaneous (R3 key)
//!   C4: GPU sequential ×2 (sanity vs Phase 9 1.945x)
//!
//! Pass-gate (R3): C3 ≤ max(C1,C2) × 1.3
//!
//! Build: cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!        --bin microbench_htp_gpu_matmul_concurrent
//! Run:   `LD_LIBRARY_PATH=/data/local/tmp/qnn:/vendor/lib64 \
//!         ADSP_LIBRARY_PATH=/data/local/tmp/qnn \
//!         adb shell /data/local/tmp/microbench_htp_gpu_matmul_concurrent [N_ITERS]`

#[cfg(not(all(feature = "qnn", feature = "opencl")))]
fn main() {
    eprintln!("microbench_htp_gpu_matmul_concurrent requires --features qnn,opencl");
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
    use std::ffi::CString;
    use std::os::raw::c_uint;
    use std::ptr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Instant;

    use qnn::*;

    let args: Vec<String> = std::env::args().collect();
    let n_iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);

    let m: usize = 1;
    let k: usize = 1024;
    let n: usize = 4096;

    println!("=== microbench_htp_gpu_matmul_concurrent (Phase 32b-2 / R3) ===\n");
    println!("HTP MatMul: A[{},{}] × B[{},{}] (DDR-heavy: 16MB B + 4KB A + 16KB C)", m, k, k, n);
    println!("GPU GEMV:   same scale on OpenCL");
    println!("n_iters per config: {}\n", n_iters);

    // ── HTP setup ──
    let lib = unsafe { Library::new("/data/local/tmp/qnn/libQnnHtp.so") }
        .or_else(|_| unsafe { Library::new("libQnnHtp.so") })?;
    type GetProvidersFn =
        unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let get_providers: Symbol<GetProvidersFn> =
        unsafe { lib.get(b"QnnInterface_getProviders\0")? };
    let mut providers: *mut *const QnnInterface_t = ptr::null_mut();
    let mut num: c_uint = 0;
    let err = unsafe { get_providers(&mut providers, &mut num) };
    anyhow::ensure!(err == 0 && num > 0, "QnnInterface_getProviders err=0x{:x}", err);
    let v = unsafe { (**providers).__bindgen_anon_1.v2_25 };

    let mut backend: Qnn_BackendHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.backendCreate.unwrap())(ptr::null_mut(), ptr::null_mut(), &mut backend) };
    anyhow::ensure!(err == 0, "backendCreate err=0x{:x}", err);
    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe {
        (v.contextCreate.unwrap())(backend, ptr::null_mut(), ptr::null_mut(), &mut ctx)
    };
    anyhow::ensure!(err == 0, "contextCreate err=0x{:x}", err);

    let graph_name = CString::new("htp_matmul").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err = unsafe {
        (v.graphCreate.unwrap())(ctx, graph_name.as_ptr(), ptr::null_mut(), &mut graph)
    };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    let dims_a: Vec<u32> = vec![m as u32, k as u32];
    let dims_b: Vec<u32> = vec![k as u32, n as u32];
    let dims_c: Vec<u32> = vec![m as u32, n as u32];
    let mk_v1_raw = |name: &CString, ttype: Qnn_TensorType_t, dims: &[u32]| -> Qnn_TensorV1_t {
        Qnn_TensorV1_t {
            id: 0,
            name: name.as_ptr(),
            type_: ttype,
            dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
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
    let name_a = CString::new("A").unwrap();
    let name_b = CString::new("B").unwrap();
    let name_c = CString::new("C").unwrap();
    let mut t_a = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&name_a, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims_a),
        },
    };
    let mut t_b = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&name_b, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims_b),
        },
    };
    let mut t_c = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&name_c, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, &dims_c),
        },
    };
    for (l, t) in [("A", &mut t_a), ("B", &mut t_b), ("C", &mut t_c)] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreateGraphTensor({}) err=0x{:x}", l, err);
    }
    let op_name = CString::new("matmul0").unwrap();
    let pkg = CString::new("qti.aisw").unwrap();
    let op_type = CString::new("MatMul").unwrap();
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
    let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
    anyhow::ensure!(err == 0, "graphAddNode err=0x{:x}", err);
    let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
    anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);
    println!("HTP graph (MatMul {}x{}x{}) finalize: OK", m, k, n);

    let host_a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001) % 1.0 - 0.5).collect();
    let host_b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0007 + 0.13) % 1.0 - 0.5).collect();
    let mut host_c: Vec<f32> = vec![0.0; m * n];

    unsafe {
        inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
            data: host_a.as_ptr() as *mut _,
            dataSize: (host_a.len() * 4) as u32,
        };
        inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
            data: host_b.as_ptr() as *mut _,
            dataSize: (host_b.len() * 4) as u32,
        };
        outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
            data: host_c.as_mut_ptr() as *mut _,
            dataSize: (host_c.len() * 4) as u32,
        };
    }

    let mut exec_htp = || -> anyhow::Result<f64> {
        let t0 = Instant::now();
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
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    // ── OpenCL setup (GEMV: y = A @ x, A[N,K], x[K], y[N]) ──
    use ocl::core::ArgVal;
    use ocl::{Context, Device, Platform, Program, Queue};
    let platform = Platform::default();
    let device = Device::first(platform)?;
    let cl_ctx = Context::builder().platform(platform).devices(device).build()?;

    let gemv_src = format!(
        r#"
        #define K {}u
        __kernel void gemv(__global const float* A,
                           __global const float* x,
                           __global float* y) {{
            int i = get_global_id(0);
            float sum = 0.0f;
            const __global float* arow = A + (uint)i * K;
            for (uint k = 0; k < K; k++) {{
                sum += arow[k] * x[k];
            }}
            y[i] = sum;
        }}
        "#,
        k
    );
    let program = Program::builder().devices(device).src(&gemv_src).build(&cl_ctx)?;
    let kernel = ocl::core::create_kernel(&program, "gemv")?;
    let buf_a_cl = unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, k * n, None)?
    };
    let buf_x_cl = unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, k, None)?
    };
    let buf_y_cl = unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_WRITE, n, None)?
    };

    // Initial fill (does not count against measurement loop)
    let q_init = Queue::new(&cl_ctx, device, None)?;
    let host_b_t: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.0009) % 1.0 - 0.5).collect();
    let host_x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.011) % 1.0 - 0.5).collect();
    unsafe {
        ocl::core::enqueue_write_buffer(&q_init, &buf_a_cl, true, 0, &host_b_t, None::<ocl::core::Event>, None::<&mut ocl::core::Event>)?;
        ocl::core::enqueue_write_buffer(&q_init, &buf_x_cl, true, 0, &host_x, None::<ocl::core::Event>, None::<&mut ocl::core::Event>)?;
    }
    ocl::core::finish(&q_init)?;

    let exec_gpu = || -> anyhow::Result<f64> {
        let q = Queue::new(&cl_ctx, device, None)?;
        let t0 = Instant::now();
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&buf_a_cl))?;
            ocl::core::set_kernel_arg(&kernel, 1, ArgVal::mem(&buf_x_cl))?;
            ocl::core::set_kernel_arg(&kernel, 2, ArgVal::mem(&buf_y_cl))?;
            ocl::core::enqueue_kernel(
                &q,
                &kernel,
                1,
                None,
                &[n, 1, 1],
                None,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        ocl::core::finish(&q)?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    // Warmup
    println!("Warmup (5 iters each)...");
    for _ in 0..5 {
        let _ = exec_htp()?;
        let _ = exec_gpu()?;
    }
    let baseline_htp = exec_htp()?;
    let baseline_gpu = exec_gpu()?;
    println!("HTP single: {:.2} ms, GPU single: {:.2} ms\n", baseline_htp, baseline_gpu);

    // ── Configs ──
    enum Cfg { GpuOnly, HtpOnly, Concurrent, GpuSeqTwo }
    let configs: &[(&str, Cfg)] = &[
        ("C1: GPU GEMV only", Cfg::GpuOnly),
        ("C2: HTP MatMul only", Cfg::HtpOnly),
        ("C3: GPU + HTP simultaneous", Cfg::Concurrent),
        ("C4: GPU GEMV ×2 sequential (sanity)", Cfg::GpuSeqTwo),
    ];

    let mut summary: Vec<(String, f64, f64, f64)> = Vec::new();

    for (label, cfg) in configs {
        let mut samples = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let ms = match cfg {
                Cfg::GpuOnly => exec_gpu()?,
                Cfg::HtpOnly => exec_htp()?,
                Cfg::Concurrent => {
                    // GPU on bg thread, HTP on main thread, both start ASAP via barrier
                    let cl_ctx_clone = cl_ctx.clone();
                    let device_clone = device;
                    let buf_a_clone = buf_a_cl.clone();
                    let buf_x_clone = buf_x_cl.clone();
                    let buf_y_clone = buf_y_cl.clone();
                    let src_clone = gemv_src.clone();
                    let counter = Arc::new(AtomicUsize::new(0));
                    let counter2 = counter.clone();

                    let gpu_handle = thread::spawn(move || -> anyhow::Result<f64> {
                        let prog = Program::builder()
                            .devices(device_clone)
                            .src(&src_clone)
                            .build(&cl_ctx_clone)?;
                        let kern = ocl::core::create_kernel(&prog, "gemv")?;
                        let q = Queue::new(&cl_ctx_clone, device_clone, None)?;
                        counter2.fetch_add(1, Ordering::SeqCst);
                        while counter2.load(Ordering::SeqCst) < 2 {
                            std::hint::spin_loop();
                        }
                        let t0 = Instant::now();
                        unsafe {
                            ocl::core::set_kernel_arg(&kern, 0, ArgVal::mem(&buf_a_clone))?;
                            ocl::core::set_kernel_arg(&kern, 1, ArgVal::mem(&buf_x_clone))?;
                            ocl::core::set_kernel_arg(&kern, 2, ArgVal::mem(&buf_y_clone))?;
                            ocl::core::enqueue_kernel(
                                &q,
                                &kern,
                                1,
                                None,
                                &[n, 1, 1],
                                None,
                                None::<&ocl::core::Event>,
                                None::<&mut ocl::core::Event>,
                            )?;
                        }
                        ocl::core::finish(&q)?;
                        Ok(t0.elapsed().as_secs_f64() * 1000.0)
                    });

                    while counter.load(Ordering::SeqCst) < 1 {
                        std::hint::spin_loop();
                    }
                    let t0 = Instant::now();
                    counter.fetch_add(1, Ordering::SeqCst);
                    let _htp_ms = exec_htp()?;
                    let _gpu_ms = gpu_handle.join().unwrap()?;
                    t0.elapsed().as_secs_f64() * 1000.0
                }
                Cfg::GpuSeqTwo => {
                    let t0 = Instant::now();
                    let _ = exec_gpu()?;
                    let _ = exec_gpu()?;
                    t0.elapsed().as_secs_f64() * 1000.0
                }
            };
            samples.push(ms);
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
            "{:<42} mean={:.3} median={:.3} σ={:.3} σ/mean={:.3}",
            label, mean, median, stddev, cv
        );
        summary.push((label.to_string(), mean, median, cv));
    }

    let c1 = summary[0].1;
    let c2 = summary[1].1;
    let c3 = summary[2].1;
    let c4 = summary[3].1;
    let max_c1c2 = c1.max(c2);
    let serial_sum = c1 + c2;

    println!("\n=== Phase 32b-2 summary (R3 answer) ===");
    println!("{:<42} {:>10}", "Config", "mean");
    println!("{}", "-".repeat(60));
    for (label, mean, _, _) in &summary {
        println!("{:<42} {:>8.3}ms", label, mean);
    }
    println!();
    println!("max(C1, C2)        = {:.3} ms", max_c1c2);
    println!("C1 + C2 (serial)   = {:.3} ms", serial_sum);
    println!("C3 (concurrent)    = {:.3} ms", c3);
    println!("C4 (GPU seq ×2)    = {:.3} ms (Phase 9 ref: 2.005x baseline)", c4);
    println!();
    println!("C3 / max(C1,C2) = {:.3}x  (1.00 perfect parallel, ≤1.30 Pass)", c3 / max_c1c2);
    println!("C3 / (C1+C2)    = {:.3}x  (0.50 perfect parallel, 1.00 serial)", c3 / serial_sum);
    println!("C4 / C1         = {:.3}x  (sanity: GPU same-chip 직렬화 vs Phase 9 1.945x)", c4 / c1);

    let r3_pass = c3 / max_c1c2 <= 1.30;
    let r3_acceptable = c3 / max_c1c2 <= 1.50;
    println!(
        "\nR3 verdict: {}",
        if r3_pass { "✓ PASS" } else if r3_acceptable { "△ ACCEPTABLE" } else { "✗ FAIL" }
    );

    unsafe {
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(backend);
    }
    if r3_acceptable { Ok(()) } else { std::process::exit(1) }
}
