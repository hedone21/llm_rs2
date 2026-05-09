//! microbench_htp_throughput — LISWAP-5 Phase D: Q3 HTP throughput as weight loader
//!
//! 목적: Hexagon V79 HTP가 weight loader로 실용적인 throughput을 가지는지.
//! OpenCL ALLOC_HOST_PTR + clEnqueueWriteBuffer 27.5 GB/s baseline (Phase 0)와
//! Vulkan staging copy 0.979x (Phase 9)와 비교.
//!
//! 측정: large ElementWiseAdd (FP32) graphExecute 의 wall-clock.
//! per-execute에 host->device 전송 (input × 2) + compute + device->host (output × 1)
//! 포함. throughput = total_bytes / time.
//!
//! Build: cargo build --release --features qnn --target aarch64-linux-android \
//!        --bin microbench_htp_throughput
//! Run:   `LD_LIBRARY_PATH=/data/local/tmp/qnn:/vendor/lib64 \
//!         ADSP_LIBRARY_PATH=/data/local/tmp/qnn \
//!         adb shell /data/local/tmp/microbench_htp_throughput [SIZE_MB] [N_ITERS]`

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_htp_throughput requires --features qnn");
    std::process::exit(2);
}

#[cfg(feature = "qnn")]
#[allow(non_snake_case, non_camel_case_types, non_upper_case_globals, dead_code)]
mod qnn {
    include!(concat!(env!("OUT_DIR"), "/qnn_bindings.rs"));
}

#[cfg(feature = "qnn")]
fn main() -> anyhow::Result<()> {
    use libloading::{Library, Symbol};
    use std::ffi::CString;
    use std::os::raw::c_uint;
    use std::ptr;
    use std::time::Instant;

    use qnn::*;

    let args: Vec<String> = std::env::args().collect();
    let size_mb: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let n_iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    let n_elements = (size_mb * 1024 * 1024) / 4; // FP32

    println!("=== microbench_htp_throughput (LISWAP-5 Phase D / Q3) ===\n");
    println!("Config: {} MB FP32 ({} elements), n_iters={}", size_mb, n_elements, n_iters);

    // ── HTP setup ──
    let htp_lib = unsafe { Library::new("/data/local/tmp/qnn/libQnnHtp.so") }
        .or_else(|_| unsafe { Library::new("libQnnHtp.so") })?;
    type GetProvidersFn =
        unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let get_providers: Symbol<GetProvidersFn> =
        unsafe { htp_lib.get(b"QnnInterface_getProviders\0")? };
    let mut providers: *mut *const QnnInterface_t = ptr::null_mut();
    let mut num: c_uint = 0;
    let err = unsafe { get_providers(&mut providers, &mut num) };
    anyhow::ensure!(err == 0 && num > 0, "QnnInterface_getProviders err=0x{:x}", err);
    let v = unsafe { (**providers).__bindgen_anon_1.v2_25 };

    let mut backend: Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe {
        (v.backendCreate.unwrap())(ptr::null_mut(), ptr::null_mut(), &mut backend)
    };
    anyhow::ensure!(err == 0, "backendCreate err=0x{:x}", err);

    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe {
        (v.contextCreate.unwrap())(backend, ptr::null_mut(), ptr::null_mut(), &mut ctx)
    };
    anyhow::ensure!(err == 0, "contextCreate err=0x{:x}", err);

    let graph_name = CString::new("htp_throughput").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err = unsafe {
        (v.graphCreate.unwrap())(ctx, graph_name.as_ptr(), ptr::null_mut(), &mut graph)
    };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    let dims = vec![n_elements as u32];
    let make_v1 = |name: &CString, ttype: Qnn_TensorType_t| -> Qnn_TensorV1_t {
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
            rank: 1,
            dimensions: dims.as_ptr() as *mut u32,
            memType: Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
            __bindgen_anon_1: Qnn_TensorV1_t__bindgen_ty_1 {
                clientBuf: Qnn_ClientBuffer_t { data: ptr::null_mut(), dataSize: 0 },
            },
        }
    };
    let name_a = CString::new("a").unwrap();
    let name_b = CString::new("b").unwrap();
    let name_c = CString::new("c").unwrap();
    let mut t_a = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: make_v1(&name_a, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE),
        },
    };
    let mut t_b = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: make_v1(&name_b, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE),
        },
    };
    let mut t_c = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: make_v1(&name_c, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ),
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
    println!("HTP graph build: OK ({} MB tensors, ElementWiseAdd FP32)", size_mb);

    // ── Buffers ──
    let host_a: Vec<f32> = (0..n_elements).map(|i| (i as f32) * 1.0e-6).collect();
    let host_b: Vec<f32> = vec![1.0; n_elements];
    let mut host_c: Vec<f32> = vec![0.0; n_elements];

    let mut exec_once = || -> anyhow::Result<f64> {
        unsafe {
            inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
                data: host_a.as_ptr() as *mut _,
                dataSize: (n_elements * 4) as u32,
            };
            inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
                data: host_b.as_ptr() as *mut _,
                dataSize: (n_elements * 4) as u32,
            };
            outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
                data: host_c.as_mut_ptr() as *mut _,
                dataSize: (n_elements * 4) as u32,
            };
        }
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

    // Warmup
    println!("\nWarmup (3 iters)...");
    for _ in 0..3 {
        let _ = exec_once()?;
    }

    // Measure
    let mut samples = Vec::with_capacity(n_iters);
    println!("\nMeasure (n={}):", n_iters);
    for i in 0..n_iters {
        let ms = exec_once()?;
        samples.push(ms);
        println!("  iter {:2}: {:7.2} ms", i, ms);
    }

    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let stddev = var.sqrt();
    let mut sorted: Vec<f64> = samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let cv = stddev / mean;

    // Bandwidth: 2 inputs + 1 output = 3 × size
    let total_gb = (size_mb as f64) * 3.0 / 1024.0;
    let bw = total_gb / (mean / 1000.0);
    let bw_input_only = ((size_mb as f64) * 2.0 / 1024.0) / (mean / 1000.0);

    println!("\n=== Phase D summary (Q3 answer) ===");
    println!("HTP ElementWiseAdd FP32 {} MB:", size_mb);
    println!("  mean    : {:7.2} ms", mean);
    println!("  median  : {:7.2} ms", median);
    println!("  stddev  : {:7.2} ms ({:.1}% σ/mean)", stddev, cv * 100.0);
    println!("  effective BW (in×2 + out×1) = {:5.2} GB/s", bw);
    println!("  effective BW (in×2 only)    = {:5.2} GB/s", bw_input_only);

    // Compare to baselines (extrapolate to 600 MB linearly)
    let extrapolated_600mb = mean * (600.0 / size_mb as f64);
    println!("\nBaselines (600 MB H2D):");
    println!("  OpenCL ALLOC_HOST_PTR + clEnqueueWriteBuffer (Phase 0): 22.28 ms / 27.5 GB/s");
    println!("  Vulkan staging → DEVICE_LOCAL via vkCmdCopyBuffer (Phase 9): 21.81 ms / 26.86 GB/s");
    println!(
        "  HTP ElementWiseAdd extrapolated to 600 MB: ≈ {:.2} ms / {:.2} GB/s (input-only)",
        extrapolated_600mb,
        (600.0 * 2.0 / 1024.0) / (extrapolated_600mb / 1000.0)
    );

    let bw_vs_opencl = bw_input_only / 27.5;
    println!(
        "\nHTP input-only BW vs OpenCL 27.5 GB/s: {:.3}x (≥0.50 = 실용 가능, ≥0.10 = layer-limited)",
        bw_vs_opencl
    );

    if bw_input_only >= 5.0 {
        println!("=> Q3 ANSWER: ✓ practical (≥ 5 GB/s)");
    } else if bw_input_only >= 1.0 {
        println!("=> Q3 ANSWER: YELLOW (1~5 GB/s, 일부 layer 적용 가능)");
    } else {
        println!("=> Q3 ANSWER: ✗ too slow for production weight loader");
    }

    // Cleanup
    unsafe {
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(backend);
    }
    Ok(())
}
