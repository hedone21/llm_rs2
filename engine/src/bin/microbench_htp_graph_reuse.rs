//! microbench_htp_graph_reuse — LISWAP-5 Phase 11: R8 graph reuse + R5 25MB sanity
//!
//! 목적:
//!   R8: 같은 graph 객체로 다른 buffer pointer를 N회 execute할 때 first/re-execute
//!       wall-clock 비교. ratio < 1.10 이면 graph 재사용 가능 (production에서
//!       매 layer마다 graph re-finalize 필요 없음).
//!   R5 (sanity): 25MB tensor (per-layer scope) ElementWiseAdd 정확성 1e-4
//!       이내 검증.
//!
//! Pass-gate (Phase 11):
//!   - graph reuse ratio ≤ 1.10
//!   - 25MB FP32 ElementWiseAdd: 100% 정확
//!
//! Build: cargo build --release --features qnn --target aarch64-linux-android \
//!        --bin microbench_htp_graph_reuse
//! Run:   `LD_LIBRARY_PATH=/data/local/tmp/qnn:/vendor/lib64 \
//!         ADSP_LIBRARY_PATH=/data/local/tmp/qnn \
//!         adb shell /data/local/tmp/microbench_htp_graph_reuse [SIZE_MB] [N_ITERS]`

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_htp_graph_reuse requires --features qnn");
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
    use std::ffi::CString;
    use std::os::raw::c_uint;
    use std::ptr;
    use std::time::Instant;

    use qnn::*;

    let args: Vec<String> = std::env::args().collect();
    let size_mb: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(25);
    let n_iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let n_elements = (size_mb * 1024 * 1024) / 4; // FP32

    println!("=== microbench_htp_graph_reuse (Phase 11 / R8 + R5 sanity) ===\n");
    println!(
        "Config: {} MB FP32 ({} elements), n_iters={}",
        size_mb, n_elements, n_iters
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

    // Build graph for one fixed shape (same as production: each layer has same dim)
    let graph_name = CString::new("htp_reuse").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, graph_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    let dims = vec![n_elements as u32];
    let mk_v1 = |name: &CString, ttype: Qnn_TensorType_t| -> Qnn_TensorV1_t {
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
            v1: mk_v1(&name_a, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE),
        },
    };
    let mut t_b = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1(&name_b, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE),
        },
    };
    let mut t_c = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1(&name_c, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ),
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
    println!("HTP graph build + finalize: OK");

    // Allocate N distinct input buffer sets to model "different layer weights"
    const N_BUFFERS: usize = 8;
    let mut hosts_a: Vec<Vec<f32>> = (0..N_BUFFERS)
        .map(|seed| {
            (0..n_elements)
                .map(|i| (i + seed * 17) as f32 * 1.0e-6)
                .collect()
        })
        .collect();
    let mut hosts_b: Vec<Vec<f32>> = (0..N_BUFFERS)
        .map(|seed| {
            (0..n_elements)
                .map(|i| (i + seed * 31) as f32 * 0.5e-6 + 1.0)
                .collect()
        })
        .collect();
    let mut hosts_c: Vec<Vec<f32>> = (0..N_BUFFERS).map(|_| vec![0.0; n_elements]).collect();

    let exec_with_buffers = |v: &QnnInterface_ImplementationV2_25_t,
                             a: &mut [f32],
                             b: &mut [f32],
                             c: &mut [f32],
                             inputs: &mut [Qnn_Tensor_t; 2],
                             outputs: &mut [Qnn_Tensor_t; 1]|
     -> anyhow::Result<f64> {
        unsafe {
            inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
                data: a.as_mut_ptr() as *mut _,
                dataSize: (a.len() * 4) as u32,
            };
            inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
                data: b.as_mut_ptr() as *mut _,
                dataSize: (b.len() * 4) as u32,
            };
            outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
                data: c.as_mut_ptr() as *mut _,
                dataSize: (c.len() * 4) as u32,
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

    // ── R5 sanity: first execute is also accuracy check ──
    println!("\n[R5 sanity] First execute (buffer 0) — check correctness");
    let first_ms = exec_with_buffers(
        &v,
        &mut hosts_a[0],
        &mut hosts_b[0],
        &mut hosts_c[0],
        &mut inputs,
        &mut outputs,
    )?;
    println!("  first execute: {:.2} ms", first_ms);

    let mut mismatch = 0usize;
    let mut first_bad = None;
    for i in 0..n_elements {
        let exp = hosts_a[0][i] + hosts_b[0][i];
        if (hosts_c[0][i] - exp).abs() > 1e-4 * exp.abs().max(1.0) {
            mismatch += 1;
            if first_bad.is_none() {
                first_bad = Some((i, hosts_c[0][i], exp));
            }
        }
    }
    if mismatch == 0 {
        println!("  R5 ✓ correct ({} elements)", n_elements);
    } else {
        println!("  R5 ✗ MISMATCH {}/{}", mismatch, n_elements);
        if let Some((i, got, exp)) = first_bad {
            println!("    first bad: c[{}]={} (expected {})", i, got, exp);
        }
    }

    // ── R8: re-execute with different buffers ──
    println!(
        "\n[R8] Graph reuse: same graph, {} iters with rotating buffers",
        n_iters
    );
    let mut samples = Vec::with_capacity(n_iters);
    for it in 0..n_iters {
        let bi = it % N_BUFFERS;
        let ms = exec_with_buffers(
            &v,
            &mut hosts_a[bi],
            &mut hosts_b[bi],
            &mut hosts_c[bi],
            &mut inputs,
            &mut outputs,
        )?;
        samples.push(ms);
        println!("  iter {:2} (buf {}): {:.2} ms", it, bi, ms);
    }

    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let stddev = var.sqrt();
    let cv = stddev / mean;
    let mut sorted: Vec<f64> = samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let bw = ((size_mb as f64) * 3.0 / 1024.0) / (mean / 1000.0); // (a + b + c) bytes per execute

    println!("\n=== Phase 11 summary ===");
    println!(
        "R5 (correctness 1e-4 @ {} MB): {}",
        size_mb,
        if mismatch == 0 { "✓" } else { "✗" }
    );
    println!();
    println!("R8 (graph reuse, n={}):", n_iters);
    println!("  first execute     : {:.2} ms", first_ms);
    println!("  re-execute mean   : {:.2} ms", mean);
    println!("  re-execute median : {:.2} ms", median);
    println!(
        "  σ                 : {:.3} ms ({:.1}% σ/mean)",
        stddev,
        cv * 100.0
    );
    println!("  effective BW      : {:.3} GB/s (a+b+c)", bw);
    let reuse_ratio = first_ms / mean;
    println!("  first / re-exec   : {:.3}x (Pass: ≤ 1.10)", reuse_ratio);

    let r8_pass = reuse_ratio <= 1.10;
    let r5_pass = mismatch == 0;
    println!();
    println!("R5 verdict: {}", if r5_pass { "✓ PASS" } else { "✗ FAIL" });
    println!(
        "R8 verdict: {}",
        if r8_pass {
            "✓ PASS"
        } else if reuse_ratio <= 1.30 {
            "△ ACCEPTABLE"
        } else {
            "✗ FAIL"
        }
    );

    if !r5_pass {
        println!("\n=> Phase 11 BLOCKED on R5. Investigate tensor metadata / dtype.");
    } else if !r8_pass && reuse_ratio > 1.30 {
        println!(
            "\n=> Phase 11 BLOCKED on R8. Graph re-finalize per call required — LISWAP-5 throughput hit."
        );
    } else {
        println!("\n=> Phase 11 PASS-gate cleared. Proceed to Phase 12 (R1+R3 rpcmem).");
    }

    unsafe {
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(backend);
    }
    if r5_pass && reuse_ratio <= 1.30 {
        Ok(())
    } else {
        std::process::exit(1)
    }
}
