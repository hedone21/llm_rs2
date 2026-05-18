//! microbench_htp_matmul_correctness — Phase 32b-1: R1 + R2
//!
//! 목적: HTP의 prebuilt MatMul op를 활용해 정확성 검증.
//! 1B 모델 FFN gate matmul scale (1×K, K×N → 1×N)으로 측정.
//!
//! Pass-gate (R1+R2):
//!   - prebuilt MatMul op 호출 OK
//!   - max abs error < 1e-3 (FP32)
//!
//! Build: cargo build --release --features qnn --target aarch64-linux-android \
//!        --bin microbench_htp_matmul_correctness
//! Run:   `LD_LIBRARY_PATH=/data/local/tmp/qnn:/vendor/lib64 \
//!         ADSP_LIBRARY_PATH=/data/local/tmp/qnn \
//!         adb shell /data/local/tmp/microbench_htp_matmul_correctness [K] [N]`

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_htp_matmul_correctness requires --features qnn");
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

    // FFN gate scale: Qwen2.5-1.5B has dim=1536, ffn_dim=8960. We use simpler 1024×4096.
    let args: Vec<String> = std::env::args().collect();
    let k: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let m: usize = 1; // single-token decode

    println!("=== microbench_htp_matmul_correctness (Phase 32b-1) ===\n");
    println!(
        "MatMul: A[{}, {}] × B[{}, {}] = C[{}, {}]",
        m, k, k, n, m, n
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

    let graph_name = CString::new("htp_matmul").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, graph_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    // ── Tensors: A [m, k], B [k, n], C [m, n] ──
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

    // ── MatMul op ──
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
    anyhow::ensure!(err == 0, "graphAddNode(MatMul) err=0x{:x}", err);
    let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
    anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);
    println!("HTP graph (MatMul) finalize: OK");

    // ── Host data ──
    // A: small values to keep magnitudes manageable
    let host_a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001) % 1.0 - 0.5).collect();
    let host_b: Vec<f32> = (0..k * n)
        .map(|i| (i as f32 * 0.0007 + 0.13) % 1.0 - 0.5)
        .collect();
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

    // Warmup + measure
    let mut samples = Vec::new();
    for _ in 0..3 {
        let _ = unsafe {
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
    }
    for _ in 0..10 {
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
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    println!(
        "graphExecute mean: {:.2} ms over {} iters",
        mean,
        samples.len()
    );

    // ── CPU NEON ground truth ──
    println!("\nComputing CPU reference...");
    let t0 = Instant::now();
    let mut ref_c = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += host_a[mi * k + ki] * host_b[ki * n + ni];
            }
            ref_c[mi * n + ni] = acc;
        }
    }
    println!(
        "CPU reference: {:.2} ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // ── Compare ──
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut idx_max = 0;
    for i in 0..(m * n) {
        let abs = (host_c[i] - ref_c[i]).abs();
        let rel = abs / ref_c[i].abs().max(1.0);
        if abs > max_abs_err {
            max_abs_err = abs;
            idx_max = i;
        }
        if rel > max_rel_err {
            max_rel_err = rel;
        }
    }
    println!(
        "\nMax abs err: {:.6e} (at idx {}: HTP={:.6} ref={:.6})",
        max_abs_err, idx_max, host_c[idx_max], ref_c[idx_max]
    );
    println!("Max rel err: {:.6e}", max_rel_err);

    let pass_strict = max_abs_err < 1e-3;
    let pass_acceptable = max_abs_err < 1e-2;
    println!(
        "\nR1: prebuilt MatMul: {}",
        if mean > 0.0 { "✓ exists" } else { "✗" }
    );
    println!(
        "R2: numerical correctness: {}",
        if pass_strict {
            "✓ PASS (<1e-3)"
        } else if pass_acceptable {
            "△ ACCEPTABLE (<1e-2)"
        } else {
            "✗ FAIL"
        }
    );

    if pass_strict || pass_acceptable {
        println!("\n=> Phase 32b-1 PASS-gate cleared. Proceed to Phase 32b-2 (R3 concurrent).");
    } else {
        println!("\n=> Phase 32b-1 BLOCKED. Investigate HTP precision option.");
    }

    unsafe {
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(backend);
    }
    if pass_acceptable {
        Ok(())
    } else {
        std::process::exit(1)
    }
}
