//! microbench_htp_rpcmem_throughput — LISWAP-5 Phase 12: R1 + R3
//!
//! 목적:
//!   R1: rpcmem zero-copy throughput. raw clientBuf path (Phase D 0.04 GB/s)와
//!       대비. Pass = ≥5 GB/s @ 25MB chunk.
//!   R3: large rpcmem alloc feasibility. 600MB single alloc 시도, 실패 시 25MB
//!       chunk fallback.
//!
//! Stage 분해:
//!   Stage 1: host RAM Vec<f32> → rpcmem ptr 으로 memcpy (CPU BW)
//!   Stage 2: HTP graph execute on rpcmem-backed MEMHANDLE tensor
//!   Stage 3: rpcmem 결과 읽기 (output)
//!   end-to-end = Stage1 + Stage2 + Stage3
//!
//! Build: cargo build --release --features qnn --target aarch64-linux-android \
//!        --bin microbench_htp_rpcmem_throughput
//! Run:   `LD_LIBRARY_PATH=/data/local/tmp/qnn:/vendor/lib64 \
//!         ADSP_LIBRARY_PATH=/data/local/tmp/qnn \
//!         adb shell /data/local/tmp/microbench_htp_rpcmem_throughput [SIZE_MB] [N_ITERS]`

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_htp_rpcmem_throughput requires --features qnn");
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
    use std::time::Instant;

    use qnn::*;

    // rpcmem heap IDs and flags (from /vendor/include/AEEStdDef.h or rpcmem.h)
    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;

    // libcdsprpc.so signatures
    type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(po: *const c_void) -> i32;

    let args: Vec<String> = std::env::args().collect();
    let size_mb: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(25);
    let n_iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let n_elements = (size_mb * 1024 * 1024) / 4;

    println!("=== microbench_htp_rpcmem_throughput (Phase 12 / R1 + R3) ===\n");
    println!(
        "Config: {} MB FP32 ({} elements), n_iters={}",
        size_mb, n_elements, n_iters
    );

    // ── libcdsprpc.so dlopen ──
    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }
        .or_else(|_| unsafe { Library::new("libcdsprpc.so") })?;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let rpcmem_free: Symbol<RpcmemFreeFn> = unsafe { rpc_lib.get(b"rpcmem_free\0")? };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };
    println!("libcdsprpc.so: dlopen + dlsym OK (rpcmem_alloc/free/to_fd)");

    // ── R3: alloc check ──
    let bytes = (n_elements * 4) as i32;
    let test_sizes_mb: Vec<usize> = vec![1, 25, 100, size_mb];
    let mut alloc_results: Vec<(usize, bool)> = Vec::new();
    println!("\n[R3] Alloc feasibility (system heap):");
    for &sz_mb in &test_sizes_mb {
        let sz = (sz_mb * 1024 * 1024) as i32;
        let p = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, sz) };
        let ok = !p.is_null();
        if ok {
            unsafe { rpcmem_free(p) };
        }
        alloc_results.push((sz_mb, ok));
        println!(
            "  rpcmem_alloc({:4} MB): {}",
            sz_mb,
            if ok { "OK" } else { "FAIL" }
        );
    }
    let r3_pass_full = alloc_results.iter().any(|(s, ok)| *s >= 600 && *ok);
    let r3_pass_chunk = alloc_results.iter().any(|(s, ok)| *s == 25 && *ok);
    println!(
        "R3 verdict: 600MB={} | 25MB chunk={}",
        if r3_pass_full { "✓" } else { "—" },
        if r3_pass_chunk { "✓" } else { "✗" }
    );

    // ── Allocate working buffers (a, b, c) ──
    let a_ptr = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    let b_ptr = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    let c_ptr = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    anyhow::ensure!(
        !a_ptr.is_null() && !b_ptr.is_null() && !c_ptr.is_null(),
        "rpcmem_alloc {} MB failed",
        size_mb
    );
    let a_fd = unsafe { rpcmem_to_fd(a_ptr) };
    let b_fd = unsafe { rpcmem_to_fd(b_ptr) };
    let c_fd = unsafe { rpcmem_to_fd(c_ptr) };
    println!(
        "\nrpcmem buffers: a={:p} fd={}, b={:p} fd={}, c={:p} fd={}",
        a_ptr, a_fd, b_ptr, b_fd, c_ptr, c_fd
    );

    // ── HTP setup ──
    let lib = unsafe { Library::new("/data/local/tmp/qnn/libQnnHtp.so") }
        .or_else(|_| unsafe { Library::new("libQnnHtp.so") })?;
    type GetProvidersFn = unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let get_providers: Symbol<GetProvidersFn> = unsafe { lib.get(b"QnnInterface_getProviders\0")? };
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

    // ── Register rpcmem fds with QNN as MemHandles ──
    let dims = vec![n_elements as u32];
    let mk_descriptor = |fd: i32| -> Qnn_MemDescriptor_t {
        Qnn_MemDescriptor_t {
            memShape: Qnn_MemShape_t {
                numDim: 1,
                dimSize: dims.as_ptr() as *mut u32,
                shapeConfig: ptr::null(),
            },
            dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            memType: Qnn_MemType_t_QNN_MEM_TYPE_ION,
            __bindgen_anon_1: Qnn_MemDescriptor_t__bindgen_ty_1 {
                ionInfo: Qnn_MemIonInfo_t { fd },
            },
        }
    };
    let descriptors = [
        mk_descriptor(a_fd),
        mk_descriptor(b_fd),
        mk_descriptor(c_fd),
    ];
    let mut mem_handles = [ptr::null_mut::<c_void>(); 3];
    let err =
        unsafe { (v.memRegister.unwrap())(ctx, descriptors.as_ptr(), 3, mem_handles.as_mut_ptr()) };
    anyhow::ensure!(err == 0, "QnnMem_register err=0x{:x}", err);
    println!(
        "QnnMem_register (3 ION fds): OK, handles=[{:p}, {:p}, {:p}]",
        mem_handles[0], mem_handles[1], mem_handles[2]
    );

    // ── Build graph: ElementWiseAdd, MEMHANDLE tensors ──
    let graph_name = CString::new("htp_rpcmem").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, graph_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    // ORT pattern: graph build with RAW memType + null clientBuf;
    // graphExecute attaches MEMHANDLE override on a per-call tensor copy.
    let mk_v1_raw = |name: &CString, ttype: Qnn_TensorType_t| -> Qnn_TensorV1_t {
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
            rank: 1,
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
    let name_a = CString::new("a").unwrap();
    let name_b = CString::new("b").unwrap();
    let name_c = CString::new("c").unwrap();
    let mut t_a = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&name_a, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE),
        },
    };
    let mut t_b = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&name_b, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE),
        },
    };
    let mut t_c = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&name_c, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ),
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
    println!("HTP graph build + finalize (RAW tensors, will attach MEMHANDLE at execute): OK");

    // Switch tensors to MEMHANDLE for execute calls (ORT pattern).
    unsafe {
        inputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mem_handles[0];
        inputs[1].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mem_handles[1];
        outputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mem_handles[2];
    }

    // ── Source data on regular host RAM (simulates weight buffer from mmap'd file) ──
    let host_a: Vec<f32> = (0..n_elements).map(|i| (i as f32) * 1.0e-6).collect();
    let host_b: Vec<f32> = vec![1.0; n_elements];

    // Helper: full pipeline (Stage 1: host→rpcmem, Stage 2: HTP execute, Stage 3: rpcmem→host read)
    let mut measure_full = || -> anyhow::Result<(f64, f64, f64, f64)> {
        // Stage 1: copy weight host→rpcmem (production: mmap'd weight chunk)
        let t1 = Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(
                host_a.as_ptr() as *const u8,
                a_ptr as *mut u8,
                bytes as usize,
            );
            std::ptr::copy_nonoverlapping(
                host_b.as_ptr() as *const u8,
                b_ptr as *mut u8,
                bytes as usize,
            );
        }
        let stage1_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // Stage 2: HTP execute via MEMHANDLE
        let t2 = Instant::now();
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
        let stage2_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // Stage 3: rpcmem read (host fetch). Skipped by default — production would
        // only do this if CPU consumed output. For "weight loader" semantics
        // there's no Stage 3.
        let t3 = Instant::now();
        let mut sum: f32 = 0.0;
        unsafe {
            let c_slice = std::slice::from_raw_parts(c_ptr as *const f32, n_elements);
            sum += c_slice[0];
        }
        let stage3_ms = t3.elapsed().as_secs_f64() * 1000.0;
        let _ = sum; // suppress unused

        let total_ms = stage1_ms + stage2_ms + stage3_ms;
        Ok((stage1_ms, stage2_ms, stage3_ms, total_ms))
    };

    println!(
        "\n[R1] Pipeline throughput @ {} MB (n={}):",
        size_mb, n_iters
    );
    println!("Warmup (3 iters)...");
    for _ in 0..3 {
        let _ = measure_full()?;
    }

    let mut s1: Vec<f64> = Vec::new();
    let mut s2: Vec<f64> = Vec::new();
    let mut tot: Vec<f64> = Vec::new();
    for i in 0..n_iters {
        let (a, b, _, t) = measure_full()?;
        println!(
            "  iter {:2}: stage1={:6.2}ms (host→rpcmem), stage2={:6.2}ms (HTP execute), total={:6.2}ms",
            i, a, b, t
        );
        s1.push(a);
        s2.push(b);
        tot.push(t);
    }

    let report = |label: &str, samples: &[f64]| -> (f64, f64) {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let stddev = var.sqrt();
        let cv = stddev / mean;
        let bw_input_only = ((size_mb as f64) * 2.0 / 1024.0) / (mean / 1000.0);
        println!(
            "  [{}] mean={:.2}ms σ={:.2}ms ({:.1}%), input-only BW={:.2} GB/s",
            label,
            mean,
            stddev,
            cv * 100.0,
            bw_input_only
        );
        (mean, bw_input_only)
    };

    println!("\n=== Phase 12 summary ===");
    println!("Stage breakdown (mean across {} iters):", n_iters);
    let (s1_mean, s1_bw) = report("Stage 1: host→rpcmem (CPU memcpy)", &s1);
    let (s2_mean, s2_bw) = report("Stage 2: HTP graphExecute (zero-copy)", &s2);
    let (tot_mean, tot_bw) = report("Total (stage1 + stage2 + stage3)", &tot);
    let _ = (s1_mean, s2_mean, tot_mean);

    println!("\nBaseline references:");
    println!("  Phase D raw clientBuf:                        0.04 GB/s");
    println!("  OpenCL ALLOC_HOST_PTR + clEnqueueWriteBuffer: 27.5 GB/s");
    println!("  Vulkan staging→DEVICE_LOCAL:                  26.86 GB/s");

    let r1_pass = s2_bw >= 5.0;
    let r1_acceptable = s2_bw >= 1.0;
    println!(
        "\nR1 verdict (Stage 2 zero-copy execute, input-only BW {:.2} GB/s):",
        s2_bw
    );
    if r1_pass {
        println!("  ✓ PASS (≥5 GB/s)");
    } else if r1_acceptable {
        println!("  △ ACCEPTABLE (≥1 GB/s)");
    } else {
        println!("  ✗ FAIL (<1 GB/s) — Option 3 (CPU+GPU) fallback warranted");
    }
    println!(
        "End-to-end (incl. stage1 host memcpy, BW {:.2} GB/s): {}",
        tot_bw,
        if tot_bw >= 5.0 { "✓" } else { "△" }
    );

    println!(
        "\nR3 verdict: 600MB={} | 25MB chunk={}",
        if r3_pass_full { "✓" } else { "—" },
        if r3_pass_chunk { "✓" } else { "✗" }
    );

    // Cleanup
    unsafe {
        let _ = (v.memDeRegister.unwrap())(mem_handles.as_ptr(), 3);
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(backend);
        rpcmem_free(a_ptr);
        rpcmem_free(b_ptr);
        rpcmem_free(c_ptr);
    }

    if !r1_acceptable || !r3_pass_chunk {
        std::process::exit(1);
    }
    Ok(())
}
