//! microbench_htp_gpu_parallel — LISWAP-5 Phase C: Q2 HTP+GPU 진정 H/W 병렬 검증
//!
//! 목적: Hexagon V79 HTP에 ~1ms compute 보내고, 동시에 Adreno 830 GPU에
//! ~1ms OpenCL kernel 보내서 wall-clock 측정. parallel ratio_to_C1 < 1.30 이면
//! 진정 H/W 동시 실행. (Phase 9 Vulkan은 multi-queue 1.945x 직렬 확정,
//! 이번엔 다른 chip 활용 — heterogeneous parallel)
//!
//! Configs (n=30):
//!   C1: GPU only baseline (OpenCL busy kernel)
//!   C2: HTP only baseline (large ElementWiseAdd graph)
//!   C3: GPU + HTP simultaneous launch ← Q2 핵심
//!   C4: GPU + GPU sequential ×2 (Phase 9 same-family 1.945x sanity)
//!   C5: HTP + GPU sync between (overhead penalty)
//!
//! Build: cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!        --bin microbench_htp_gpu_parallel
//! Run:   `LD_LIBRARY_PATH=/data/local/tmp/qnn:/vendor/lib64 \
//!         ADSP_LIBRARY_PATH=/data/local/tmp/qnn \
//!         adb shell /data/local/tmp/microbench_htp_gpu_parallel [N_ITERS]`

#[cfg(not(all(feature = "qnn", feature = "opencl")))]
fn main() {
    eprintln!("microbench_htp_gpu_parallel requires --features qnn,opencl");
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

    // ── HTP setup ──
    let htp_lib = unsafe { Library::new("/data/local/tmp/qnn/libQnnHtp.so") }
        .or_else(|_| unsafe { Library::new("libQnnHtp.so") })?;
    type GetProvidersFn = unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let get_providers: Symbol<GetProvidersFn> =
        unsafe { htp_lib.get(b"QnnInterface_getProviders\0")? };
    let mut providers: *mut *const QnnInterface_t = ptr::null_mut();
    let mut num: c_uint = 0;
    let err = unsafe { get_providers(&mut providers, &mut num) };
    anyhow::ensure!(
        err == 0 && num > 0,
        "QnnInterface_getProviders err=0x{:x}",
        err
    );
    let v = unsafe { (**providers).__bindgen_anon_1.v2_25 };

    let mut backend: Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe { (v.backendCreate.unwrap())(ptr::null_mut(), ptr::null_mut(), &mut backend) };
    anyhow::ensure!(err == 0, "backendCreate err=0x{:x}", err);

    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.contextCreate.unwrap())(backend, ptr::null_mut(), ptr::null_mut(), &mut ctx) };
    anyhow::ensure!(err == 0, "contextCreate err=0x{:x}", err);

    // HTP graph: large ElementWiseAdd to give the NPU enough work to be measurable.
    // We tune element count so a single graphExecute ≈ 1 ms.
    fn build_htp_add_graph(
        v: &QnnInterface_ImplementationV2_25_t,
        ctx: Qnn_ContextHandle_t,
        n_elements: usize,
    ) -> anyhow::Result<(
        Qnn_GraphHandle_t,
        Qnn_Tensor_t,
        Qnn_Tensor_t,
        Qnn_Tensor_t,
        std::ffi::CString,
        std::ffi::CString,
        std::ffi::CString,
        std::ffi::CString,
        std::ffi::CString,
        std::ffi::CString,
        Vec<u32>,
    )> {
        let graph_name = CString::new(format!("htp_busy_{}", n_elements)).unwrap();
        let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
        let err = unsafe {
            (v.graphCreate.unwrap())(ctx, graph_name.as_ptr(), ptr::null_mut(), &mut graph)
        };
        anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

        let dims = vec![n_elements as u32];
        fn mk_v1(name: &CString, ttype: Qnn_TensorType_t, dims: &[u32]) -> Qnn_TensorV1_t {
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
        }
        let name_a = CString::new("a").unwrap();
        let name_b = CString::new("b").unwrap();
        let name_c = CString::new("c").unwrap();
        let mut t_a = Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_v1(&name_a, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims),
            },
        };
        let mut t_b = Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_v1(&name_b, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims),
            },
        };
        let mut t_c = Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_v1(&name_c, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, &dims),
            },
        };
        for (l, t) in [("a", &mut t_a), ("b", &mut t_b), ("c", &mut t_c)] {
            let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
            anyhow::ensure!(err == 0, "tensorCreateGraphTensor({}) err=0x{:x}", l, err);
        }

        let op_name = CString::new("add").unwrap();
        let pkg = CString::new("qti.aisw").unwrap();
        let op_type = CString::new("ElementWiseAdd").unwrap();
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

        // We need to keep CStrings alive (they were referenced by tensor.name pointers
        // during create, but after finalize the backend has its own copy).
        let _ = (
            name_a.clone(),
            name_b.clone(),
            name_c.clone(),
            op_name.clone(),
            pkg.clone(),
            op_type.clone(),
        );
        Ok((
            graph, t_a, t_b, t_c, name_a, name_b, name_c, op_name, pkg, op_type, dims,
        ))
    }

    // Auto-tune HTP element count to ~1 ms / execute
    let mut htp_n = 65536usize;
    let mut htp_graph_state = build_htp_add_graph(&v, ctx, htp_n)?;
    let mut htp_a = vec![0.5f32; htp_n];
    let mut htp_b = vec![1.0f32; htp_n];
    let mut htp_c = vec![0.0f32; htp_n];

    let exec_htp = |v: &QnnInterface_ImplementationV2_25_t,
                    g: &mut (
        Qnn_GraphHandle_t,
        Qnn_Tensor_t,
        Qnn_Tensor_t,
        Qnn_Tensor_t,
        CString,
        CString,
        CString,
        CString,
        CString,
        CString,
        Vec<u32>,
    ),
                    a: &mut [f32],
                    b: &mut [f32],
                    c: &mut [f32]|
     -> anyhow::Result<()> {
        unsafe {
            g.1.__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
                data: a.as_mut_ptr() as *mut _,
                dataSize: (a.len() * 4) as u32,
            };
            g.2.__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
                data: b.as_mut_ptr() as *mut _,
                dataSize: (b.len() * 4) as u32,
            };
            g.3.__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
                data: c.as_mut_ptr() as *mut _,
                dataSize: (c.len() * 4) as u32,
            };
            let inputs = [g.1, g.2];
            let mut outputs = [g.3];
            let err = (v.graphExecute.unwrap())(
                g.0,
                inputs.as_ptr(),
                2,
                outputs.as_mut_ptr(),
                1,
                ptr::null_mut(),
                ptr::null_mut(),
            );
            anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);
        }
        Ok(())
    };

    println!(
        "Tuning HTP ElementWiseAdd size for ~1 ms / execute (start n={}):",
        htp_n
    );
    for _ in 0..6 {
        let t0 = Instant::now();
        exec_htp(&v, &mut htp_graph_state, &mut htp_a, &mut htp_b, &mut htp_c)?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  htp_n={}: {:.3} ms", htp_n, ms);
        if ms < 0.7 {
            htp_n *= 2;
        } else if ms > 1.5 {
            htp_n /= 2;
        } else {
            break;
        }
        // rebuild for new size
        htp_graph_state = build_htp_add_graph(&v, ctx, htp_n)?;
        htp_a = vec![0.5f32; htp_n];
        htp_b = vec![1.0f32; htp_n];
        htp_c = vec![0.0f32; htp_n];
    }
    let htp_per_iter_target_ms = {
        // measure final
        let t0 = Instant::now();
        exec_htp(&v, &mut htp_graph_state, &mut htp_a, &mut htp_b, &mut htp_c)?;
        t0.elapsed().as_secs_f64() * 1000.0
    };
    println!(
        "HTP per-iter ≈ {:.3} ms (n={})",
        htp_per_iter_target_ms, htp_n
    );

    // ── OpenCL setup ──
    use ocl::core::ArgVal;
    use ocl::{Context, Device, Platform, Program, Queue};

    let platform = Platform::default();
    let device = Device::first(platform)?;
    let cl_ctx = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    let busy_src = r#"
        __kernel void busy(__global float* out, const int iters) {
            int id = get_global_id(0);
            float v = (float)id;
            for (int i = 0; i < iters; i++) {
                v = v * 1.00001f + 0.5f;
                v -= 0.5f;
            }
            out[id] = v;
        }
    "#;
    let program = Program::builder()
        .devices(device)
        .src(busy_src)
        .build(&cl_ctx)?;
    let kernel = ocl::core::create_kernel(&program, "busy")?;
    const GSIZE: usize = 1024;
    let buf_a = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_WRITE,
            GSIZE,
            None,
        )?
    };
    let buf_b = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_WRITE,
            GSIZE,
            None,
        )?
    };

    // Tune iters → ~1ms
    let mut cl_iters: i32 = 100_000;
    println!("\nTuning OpenCL busy kernel iters to ~1 ms...");
    for _ in 0..6 {
        let q = Queue::new(&cl_ctx, device, None)?;
        let t0 = Instant::now();
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&buf_a))?;
            ocl::core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&cl_iters))?;
            ocl::core::enqueue_kernel(
                &q,
                &kernel,
                1,
                None,
                &[GSIZE, 1, 1],
                None,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        ocl::core::finish(&q)?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  cl_iters={}: {:.2} ms", cl_iters, ms);
        if ms < 0.7 {
            cl_iters *= 2;
        } else if ms > 1.5 {
            cl_iters = (cl_iters as f64 / (ms / 1.0)) as i32;
        } else {
            break;
        }
    }
    println!("OpenCL per-iter ≈ ~1 ms (cl_iters={})", cl_iters);

    // ── Run helpers ──
    let exec_gpu = |buf: &ocl::core::Mem| -> anyhow::Result<f64> {
        let q = Queue::new(&cl_ctx, device, None)?;
        let t0 = Instant::now();
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&cl_iters))?;
            ocl::core::enqueue_kernel(
                &q,
                &kernel,
                1,
                None,
                &[GSIZE, 1, 1],
                None,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        ocl::core::finish(&q)?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    // ── Configs ──
    enum Cfg {
        GpuOnly,
        HtpOnly,
        Parallel,
        GpuSeqTwo,
        HtpThenGpu,
    }
    let configs: &[(&str, Cfg)] = &[
        ("C1: GPU only baseline", Cfg::GpuOnly),
        ("C2: HTP only baseline", Cfg::HtpOnly),
        ("C3: GPU + HTP simultaneous (Q2 key)", Cfg::Parallel),
        ("C4: GPU sequential ×2 (Phase 9 sanity)", Cfg::GpuSeqTwo),
        ("C5: HTP then GPU (sequential)", Cfg::HtpThenGpu),
    ];

    println!(
        "\n=== Two-engine concurrent test (HTP+GPU, ~1ms × {} iters/cfg) ===\n",
        n_iters
    );

    let mut summary: Vec<(String, f64, f64, f64)> = Vec::new();

    for (label, cfg) in configs {
        // Warmup
        for _ in 0..5 {
            match cfg {
                Cfg::GpuOnly | Cfg::GpuSeqTwo => {
                    let _ = exec_gpu(&buf_a)?;
                }
                Cfg::HtpOnly => {
                    exec_htp(&v, &mut htp_graph_state, &mut htp_a, &mut htp_b, &mut htp_c)?
                }
                Cfg::Parallel | Cfg::HtpThenGpu => {
                    exec_htp(&v, &mut htp_graph_state, &mut htp_a, &mut htp_b, &mut htp_c)?;
                    let _ = exec_gpu(&buf_a)?;
                }
            }
        }

        let mut samples = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let ms = match cfg {
                Cfg::GpuOnly => exec_gpu(&buf_a)?,
                Cfg::HtpOnly => {
                    let t0 = Instant::now();
                    exec_htp(&v, &mut htp_graph_state, &mut htp_a, &mut htp_b, &mut htp_c)?;
                    t0.elapsed().as_secs_f64() * 1000.0
                }
                Cfg::Parallel => {
                    // GPU on background thread, HTP on main thread, both started ASAP.
                    let cl_ctx_clone = cl_ctx.clone();
                    let buf_clone = buf_b.clone();
                    let kernel_clone_iters = cl_iters;
                    let device_clone = device;
                    let kernel_program_src = busy_src.to_string();
                    let counter = Arc::new(AtomicUsize::new(0));
                    let counter_clone = counter.clone();

                    let gpu_handle = thread::spawn(move || -> anyhow::Result<f64> {
                        let prog = Program::builder()
                            .devices(device_clone)
                            .src(&kernel_program_src)
                            .build(&cl_ctx_clone)?;
                        let kern = ocl::core::create_kernel(&prog, "busy")?;
                        let q = Queue::new(&cl_ctx_clone, device_clone, None)?;
                        // Signal main thread we're ready
                        counter_clone.fetch_add(1, Ordering::SeqCst);
                        // Wait for main thread also ready
                        while counter_clone.load(Ordering::SeqCst) < 2 {
                            std::hint::spin_loop();
                        }
                        let t0 = Instant::now();
                        unsafe {
                            ocl::core::set_kernel_arg(&kern, 0, ArgVal::mem(&buf_clone))?;
                            ocl::core::set_kernel_arg(
                                &kern,
                                1,
                                ArgVal::scalar(&kernel_clone_iters),
                            )?;
                            ocl::core::enqueue_kernel(
                                &q,
                                &kern,
                                1,
                                None,
                                &[GSIZE, 1, 1],
                                None,
                                None::<&ocl::core::Event>,
                                None::<&mut ocl::core::Event>,
                            )?;
                        }
                        ocl::core::finish(&q)?;
                        Ok(t0.elapsed().as_secs_f64() * 1000.0)
                    });

                    // Wait for GPU thread to be ready
                    while counter.load(Ordering::SeqCst) < 1 {
                        std::hint::spin_loop();
                    }
                    let t0 = Instant::now();
                    counter.fetch_add(1, Ordering::SeqCst); // signal both go
                    exec_htp(&v, &mut htp_graph_state, &mut htp_a, &mut htp_b, &mut htp_c)?;
                    let _gpu_ms = gpu_handle.join().unwrap()?;
                    t0.elapsed().as_secs_f64() * 1000.0
                }
                Cfg::GpuSeqTwo => {
                    let t0 = Instant::now();
                    let _ = exec_gpu(&buf_a)?;
                    let _ = exec_gpu(&buf_b)?;
                    t0.elapsed().as_secs_f64() * 1000.0
                }
                Cfg::HtpThenGpu => {
                    let t0 = Instant::now();
                    exec_htp(&v, &mut htp_graph_state, &mut htp_a, &mut htp_b, &mut htp_c)?;
                    let _ = exec_gpu(&buf_a)?;
                    t0.elapsed().as_secs_f64() * 1000.0
                }
            };
            samples.push(ms);
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let stddev = var.sqrt();
        let cv = stddev / mean;
        println!(
            "{:<42} mean={:.3} median={:.3} σ={:.3} σ/mean={:.3}",
            label, mean, median, stddev, cv
        );
        summary.push((label.to_string(), mean, median, cv));
    }

    let baseline = summary[0].1;
    let htp_baseline = summary[1].1;
    let sum_baseline = baseline + htp_baseline;
    println!("\n=== Phase C summary (Q2 answer) ===");
    println!(
        "{:<42} {:>10} {:>10} {:>10} {:>10}",
        "Config", "mean", "median", "σ/mean", "ratio_C1"
    );
    println!("{}", "-".repeat(95));
    for (label, mean, median, cv) in &summary {
        println!(
            "{:<42} {:>8.3}ms {:>8.3}ms {:>9.3} {:>9.3}x",
            label,
            mean,
            median,
            cv,
            mean / baseline
        );
    }
    println!("\nGPU-only (C1)    = {:.3} ms (baseline)", baseline);
    println!("HTP-only (C2)    = {:.3} ms", htp_baseline);
    println!(
        "Sum (C1+C2)      = {:.3} ms (perfect serial upper bound)",
        sum_baseline
    );
    println!("Parallel (C3)    = {:.3} ms", summary[2].1);

    let parallel_ratio_to_sum = summary[2].1 / sum_baseline;
    let parallel_ratio_to_max = summary[2].1 / baseline.max(htp_baseline);
    println!(
        "C3 / (C1+C2)     = {:.3}x  (1.00 = perfect serial, 0.5 = perfect parallel)",
        parallel_ratio_to_sum
    );
    println!(
        "C3 / max(C1,C2)  = {:.3}x  (1.00 = perfect parallel, 2.00 = serial)",
        parallel_ratio_to_max
    );
    if parallel_ratio_to_max < 1.30 {
        println!("=> Q2 ANSWER: ✓ true H/W parallel — HTP+GPU heterogeneous concurrency confirmed");
    } else if parallel_ratio_to_max < 1.80 {
        println!("=> Q2 ANSWER: partial overlap (treated as ✗)");
    } else {
        println!("=> Q2 ANSWER: ✗ HTP+GPU also serializes (uncommon, paper finding)");
    }

    // Cleanup
    unsafe {
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(backend);
    }
    Ok(())
}
