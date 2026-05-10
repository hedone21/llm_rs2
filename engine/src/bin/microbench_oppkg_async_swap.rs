//! microbench_oppkg_async_swap — GPU forward 실행 중 weight swap async 검증
//!
//! 사용자 본질 질문: "GPU 연산 중에 memory write해도 별도 sync 없어?"
//!
//! 시나리오 (production weight swap):
//!   - W_a: 현 layer 가중치 (rpcmem, MEMHANDLE으로 graph에 등록)
//!   - W_b: 다음 layer 가중치 destination (rpcmem, host에서 mmap copy)
//!   - graphExecute(W_a)가 GPU에서 도는 동안 host가 W_b에 memcpy
//!   - parallel이면 wall-clock ≤ max(graphExecute, memcpy) × 1.1
//!
//! Configs (n=20):
//!   C1: graphExecute only (W_a)
//!   C2: host memcpy only (→ W_b)
//!   C3: graphExecute + memcpy concurrent (다른 thread, AtomicUsize barrier)
//!   C4: sequential (C1 → C2)
//!
//! Pass: C3 / max(C1,C2) ≤ 1.10 → async swap viable
//!
//! 차원: K=1536, N=8960 (Qwen FFN gate, F16 weight 27.5 MB)

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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Instant;

    use qnn::*;

    const PKG_PATH: &str = "/data/local/tmp/libqnn_oppkg_poc.so";
    const PKG_PROVIDER: &str = "QnnOpPackage_InitInterface";
    const PKG_TARGET: &str = "GPU";

    // CLI: [dim] [chain] [memcpy_mb]
    //   dim: square K=N (default 2048, "memory-bound large matmul")
    //   chain: chain_depth (default 8)
    //   memcpy_mb: swap chunk size in MB (default 8)
    let args: Vec<String> = std::env::args().collect();
    let dim: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let chain_depth: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let memcpy_mb: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    let m: usize = 1;
    let k: usize = dim;
    let n: usize = dim;
    let weight_bytes = n * k * 2; // F16
    let n_iters: usize = 20;

    println!("=== microbench_oppkg_async_swap ===");
    println!("dim K={} N={} (square)", k, n);
    println!(
        "Weight per op: F16 = {:.2} MB",
        weight_bytes as f64 / 1024.0 / 1024.0
    );
    println!("graphExecute chain depth: {}", chain_depth);
    println!("memcpy chunk: {} MB", memcpy_mb);
    println!("n_iters: {}\n", n_iters);

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

    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe { (v.contextCreate.unwrap())(be, ptr::null_mut(), ptr::null_mut(), &mut ctx) };
    anyhow::ensure!(err == 0);

    let g_name = CString::new("async_swap_chain").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0);

    let _ = weight_bytes;

    let dims_w: Vec<u32> = vec![n as u32, k as u32];
    let dims_v: Vec<u32> = vec![m as u32, n as u32];
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

    let n_w = CString::new("W").unwrap();
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

    let mut int_names: Vec<CString> = (0..=chain_depth)
        .map(|i| CString::new(format!("y{}", i)).unwrap())
        .collect();
    let mut y_tensors: Vec<Qnn_Tensor_t> = Vec::with_capacity(chain_depth + 1);
    for i in 0..=chain_depth {
        let ttype = if i == 0 {
            Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE
        } else if i == chain_depth {
            Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ
        } else {
            Qnn_TensorType_t_QNN_TENSOR_TYPE_NATIVE
        };
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
    let _ = int_names;

    let pkg_n = CString::new("qnn_oppkg_poc").unwrap();
    let op_type = CString::new("CustomMatMul").unwrap();
    let op_names: Vec<CString> = (0..chain_depth)
        .map(|i| CString::new(format!("mm_{}", i)).unwrap())
        .collect();
    let mut inputs_h: Vec<[Qnn_Tensor_t; 2]> = Vec::with_capacity(chain_depth);
    let mut outputs_h: Vec<[Qnn_Tensor_t; 1]> = Vec::with_capacity(chain_depth);
    for i in 0..chain_depth {
        inputs_h.push([t_w, y_tensors[i]]);
        outputs_h.push([y_tensors[i + 1]]);
    }
    for i in 0..chain_depth {
        let op = Qnn_OpConfig_t {
            version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
            __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
                v1: Qnn_OpConfigV1_t {
                    name: op_names[i].as_ptr(),
                    packageName: pkg_n.as_ptr(),
                    typeName: op_type.as_ptr(),
                    numOfParams: 0,
                    params: ptr::null_mut(),
                    numOfInputs: 2,
                    inputTensors: inputs_h[i].as_mut_ptr(),
                    numOfOutputs: 1,
                    outputTensors: outputs_h[i].as_mut_ptr(),
                },
            },
        };
        let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
        anyhow::ensure!(err == 0);
    }
    let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
    anyhow::ensure!(err == 0);
    println!("graphFinalize: OK");

    // rpcmem buffers
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
    let memcpy_bytes = (memcpy_mb * 1024 * 1024) as i32;
    let rpc_w_a = unsafe { rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bw) };
    let rpc_w_b = unsafe { rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, memcpy_bytes) };
    let rpc_y0 = unsafe { rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bx) };
    let rpc_yn = unsafe { rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, by) };
    anyhow::ensure!(
        !rpc_w_a.is_null() && !rpc_w_b.is_null() && !rpc_y0.is_null() && !rpc_yn.is_null()
    );

    // Fill rpc_w_a
    let mut host_w_a = vec![0u16; n * k];
    for i in 0..n * k {
        let val = ((i as f32) * 0.0007 + 0.13).rem_euclid(1.0) - 0.5;
        host_w_a[i] = f32_to_f16_bits(val);
    }
    let host_x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.011).rem_euclid(1.0) - 0.5)
        .collect();
    unsafe {
        std::ptr::copy_nonoverlapping(
            host_w_a.as_ptr() as *const u8,
            rpc_w_a as *mut u8,
            bw as usize,
        );
        std::ptr::copy_nonoverlapping(host_x.as_ptr() as *const u8, rpc_y0 as *mut u8, bx as usize);
    }
    // host source for memcpy → rpc_w_b. size = memcpy_bytes (independent of graph weight)
    let memcpy_elems = (memcpy_bytes / 2) as usize;
    let host_w_b_src: Vec<u16> = (0..memcpy_elems)
        .map(|i| ((i as f32 * 0.31) % 1.0 - 0.5) as f32)
        .map(f32_to_f16_bits)
        .collect();

    let fd_w_a = unsafe { rpc_to_fd(rpc_w_a) };
    let fd_y0 = unsafe { rpc_to_fd(rpc_y0) };
    let fd_yn = unsafe { rpc_to_fd(rpc_yn) };
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
    let descs = [
        mk_desc(
            fd_w_a,
            rpc_w_a,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
            &dims_w,
        ),
        mk_desc(
            fd_y0,
            rpc_y0,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            &dims_y0,
        ),
        mk_desc(fd_yn, rpc_yn, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_v),
    ];
    let mut mh = [ptr::null_mut::<c_void>(); 3];
    let err = unsafe { (v.memRegister.unwrap())(ctx, descs.as_ptr(), 3, mh.as_mut_ptr()) };
    anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);

    let mut t_w_mh = t_w;
    t_w_mh.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    t_w_mh.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[0];
    let mut t_y0_mh = y_tensors[0];
    t_y0_mh.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    t_y0_mh.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[1];
    let mut t_yn_mh = y_tensors[chain_depth];
    t_yn_mh.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    t_yn_mh.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[2];

    let exec_inputs = [t_w_mh, t_y0_mh];
    let mut exec_outputs = [t_yn_mh];

    let v_ptr = &v as *const _ as usize;
    let graph_ptr = graph as usize;
    let in_ptr = exec_inputs.as_ptr() as usize;
    let out_ptr = exec_outputs.as_mut_ptr() as usize;

    let exec_graph = || -> anyhow::Result<f64> {
        let v = unsafe { &*(v_ptr as *const QnnInterface_ImplementationV2_25_t) };
        let t0 = Instant::now();
        let err = unsafe {
            (v.graphExecute.unwrap())(
                graph_ptr as Qnn_GraphHandle_t,
                in_ptr as *const Qnn_Tensor_t,
                2,
                out_ptr as *mut Qnn_Tensor_t,
                1,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    let rpc_w_b_addr = rpc_w_b as usize;
    let host_w_b_src_ptr = host_w_b_src.as_ptr() as usize;
    let weight_size = memcpy_bytes as usize;
    let exec_memcpy = || -> f64 {
        let t0 = Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(
                host_w_b_src_ptr as *const u8,
                rpc_w_b_addr as *mut u8,
                weight_size,
            );
        }
        t0.elapsed().as_secs_f64() * 1000.0
    };

    // Warmup
    println!("Warmup (5 iters)...");
    for _ in 0..5 {
        let _ = exec_graph()?;
        let _ = exec_memcpy();
    }

    let measure = |label: &str,
                   mut f: Box<dyn FnMut() -> anyhow::Result<f64>>,
                   n_iters: usize|
     -> anyhow::Result<f64> {
        let mut s = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            s.push(f()?);
        }
        let mean = s.iter().sum::<f64>() / n_iters as f64;
        let mut sorted = s.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let var = s.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_iters as f64;
        let cv = var.sqrt() / mean;
        println!(
            "{:<42} mean={:.3}ms median={:.3}ms σ/mean={:.4}",
            label, mean, median, cv
        );
        Ok(mean)
    };

    let c1 = measure("C1: graphExecute only", Box::new(exec_graph), n_iters)?;
    let c2 = measure(
        "C2: host memcpy only",
        Box::new(move || Ok(exec_memcpy())),
        n_iters,
    )?;

    // C3: concurrent
    let mut c3_samples = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter2 = counter.clone();
        let h2 = thread::spawn(move || -> f64 {
            counter2.fetch_add(1, Ordering::SeqCst);
            while counter2.load(Ordering::SeqCst) < 2 {
                std::hint::spin_loop();
            }
            let t0 = Instant::now();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host_w_b_src_ptr as *const u8,
                    rpc_w_b_addr as *mut u8,
                    weight_size,
                );
            }
            t0.elapsed().as_secs_f64() * 1000.0
        });
        while counter.load(Ordering::SeqCst) < 1 {
            std::hint::spin_loop();
        }
        let t0 = Instant::now();
        counter.fetch_add(1, Ordering::SeqCst);
        let _ = exec_graph()?;
        let _ = h2.join().unwrap();
        c3_samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let c3_mean = c3_samples.iter().sum::<f64>() / n_iters as f64;
    let mut sorted_c3 = c3_samples.clone();
    sorted_c3.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let c3_med = sorted_c3[sorted_c3.len() / 2];
    let c3_var = c3_samples
        .iter()
        .map(|&x| (x - c3_mean).powi(2))
        .sum::<f64>()
        / n_iters as f64;
    let c3_cv = c3_var.sqrt() / c3_mean;
    println!(
        "{:<42} mean={:.3}ms median={:.3}ms σ/mean={:.4}",
        "C3: concurrent (graphExecute || memcpy)", c3_mean, c3_med, c3_cv
    );

    // C4: sequential
    let mut c4_samples = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        let _ = exec_graph()?;
        let _ = exec_memcpy();
        c4_samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let c4_mean = c4_samples.iter().sum::<f64>() / n_iters as f64;
    println!(
        "{:<42} mean={:.3}ms (sanity ≈ C1+C2)",
        "C4: sequential", c4_mean
    );

    let max_c1c2 = c1.max(c2);
    let serial_sum = c1 + c2;
    println!("\n=== Async swap viability ===");
    println!("C1 (graphExecute)  = {:.3} ms", c1);
    println!("C2 (memcpy 27MB)   = {:.3} ms", c2);
    println!("max(C1,C2)         = {:.3} ms", max_c1c2);
    println!("C1 + C2 (serial)   = {:.3} ms", serial_sum);
    println!("C3 (concurrent)    = {:.3} ms", c3_mean);
    println!("C4 (sequential)    = {:.3} ms", c4_mean);
    println!();
    println!(
        "C3 / max(C1,C2)    = {:.3}x  (≤1.10 → async OK)",
        c3_mean / max_c1c2
    );
    println!(
        "C3 / (C1+C2)       = {:.3}x  (≤0.70 → strong parallel; 1.0 = serialize)",
        c3_mean / serial_sum
    );

    let pass = c3_mean / max_c1c2 <= 1.10;
    let acceptable = c3_mean / max_c1c2 <= 1.30;
    println!(
        "\nVerdict: {}",
        if pass {
            "✓ PASS — async swap OK (no measurable contention)"
        } else if acceptable {
            "△ ACCEPTABLE — partial overlap"
        } else {
            "✗ FAIL — serialize or contention"
        }
    );

    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), 3);
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(be);
    }
    if acceptable {
        Ok(())
    } else {
        std::process::exit(1)
    }
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
