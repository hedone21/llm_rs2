//! microbench_htp_correctness — LISWAP-5 Phase B: Q1 correctness via QNN HTP
//!
//! 목적: stock Galaxy S25 Hexagon V79 HTP에서 ElementWiseAdd 1024 elements 실행
//! 후 host에서 결과 검증. QNN backend create → graph build → execute 전 경로
//! 검증.
//!
//! Build: cargo build --release --features qnn --target aarch64-linux-android \
//!        --bin microbench_htp_correctness
//! Run:   `LD_LIBRARY_PATH=/vendor/lib64/snap:/vendor/lib64 \
//!         adb shell /data/local/tmp/microbench_htp_correctness`

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_htp_correctness requires --features qnn");
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

    use qnn::*;

    const N: usize = 1024;

    println!("=== microbench_htp_correctness (LISWAP-5 Phase B / Q1) ===\n");

    // ── 1. dlopen libQnnHtp.so + dlsym ──
    // Prefer SDK 2.33 .so we pushed to /data/local/tmp/qnn/, fall back to vendor.
    let lib = unsafe { Library::new("/data/local/tmp/qnn/libQnnHtp.so") }
        .or_else(|_| unsafe { Library::new("/vendor/lib64/snap/libQnnHtp.so") })
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
    let provider = unsafe { &**providers };
    let api = provider.apiVersion.coreApiVersion;
    println!(
        "Provider: backendId={}, providerName={:?}, api={}.{}.{}",
        provider.backendId,
        if provider.providerName.is_null() {
            "(null)".to_string()
        } else {
            unsafe {
                std::ffi::CStr::from_ptr(provider.providerName)
                    .to_string_lossy()
                    .into_owned()
            }
        },
        api.major,
        api.minor,
        api.patch
    );
    let v = unsafe { provider.__bindgen_anon_1.v2_25 };

    // ── 2. backendCreate ──
    let mut backend: Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe {
        (v.backendCreate.unwrap())(
            ptr::null_mut(), // logger
            ptr::null_mut(), // config
            &mut backend,
        )
    };
    anyhow::ensure!(err == 0, "backendCreate err=0x{:x}", err);
    println!("backendCreate: OK ({:p})", backend);

    // ── 3. backendGetApiVersion (sanity) ──
    let mut bv = Qnn_ApiVersion_t {
        coreApiVersion: Qnn_Version_t {
            major: 0,
            minor: 0,
            patch: 0,
        },
        backendApiVersion: Qnn_Version_t {
            major: 0,
            minor: 0,
            patch: 0,
        },
    };
    let err = unsafe { (v.backendGetApiVersion.unwrap())(&mut bv) };
    anyhow::ensure!(err == 0, "backendGetApiVersion err=0x{:x}", err);
    println!(
        "backendGetApiVersion: core={}.{}.{}, backend={}.{}.{}",
        bv.coreApiVersion.major,
        bv.coreApiVersion.minor,
        bv.coreApiVersion.patch,
        bv.backendApiVersion.major,
        bv.backendApiVersion.minor,
        bv.backendApiVersion.patch
    );

    // ── 4. contextCreate ──
    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe {
        (v.contextCreate.unwrap())(
            backend,
            ptr::null_mut(), // device (NULL = default HTP device)
            ptr::null_mut(), // config
            &mut ctx,
        )
    };
    anyhow::ensure!(err == 0, "contextCreate err=0x{:x}", err);
    println!("contextCreate: OK ({:p})", ctx);

    // ── 5. graphCreate ──
    let graph_name = CString::new("add_q1").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, graph_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);
    println!("graphCreate: OK ({:p})", graph);

    // ── 6. Tensors (input a, b, output c) ──
    fn make_tensor_v1(
        id: u32,
        name: &CString,
        ttype: Qnn_TensorType_t,
        dims: &[u32],
        data_ptr: *mut std::os::raw::c_void,
        data_size: u32,
    ) -> Qnn_TensorV1_t {
        Qnn_TensorV1_t {
            id,
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
                    data: data_ptr,
                    dataSize: data_size,
                },
            },
        }
    }

    let mut host_a: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
    let mut host_b: Vec<f32> = (0..N).map(|i| i as f32 * 0.25 + 1.0).collect();
    let mut host_c: Vec<f32> = vec![0.0; N];
    let dims = [N as u32];

    let name_a = CString::new("a").unwrap();
    let name_b = CString::new("b").unwrap();
    let name_c = CString::new("c").unwrap();

    // ORT pattern: tensor created with null clientBuf in graph build time;
    // actual buffer attached at graphExecute time.
    let mut t_a = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: make_tensor_v1(
                0,
                &name_a,
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                &dims,
                ptr::null_mut(),
                0,
            ),
        },
    };
    let mut t_b = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: make_tensor_v1(
                0,
                &name_b,
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                &dims,
                ptr::null_mut(),
                0,
            ),
        },
    };
    let mut t_c = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: make_tensor_v1(
                0,
                &name_c,
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
                &dims,
                ptr::null_mut(),
                0,
            ),
        },
    };

    for (label, t) in [("a", &mut t_a), ("b", &mut t_b), ("c", &mut t_c)] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(
            err == 0,
            "tensorCreateGraphTensor({}) err=0x{:x}",
            label,
            err
        );
        let id_after = unsafe { t.__bindgen_anon_1.v1.id };
        println!("tensorCreateGraphTensor({}): OK, id={}", label, id_after);
    }

    // ── 7. graphAddNode (ElementWiseAdd) ──
    let op_name = CString::new("add0").unwrap();
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
    anyhow::ensure!(err == 0, "graphAddNode(ElementWiseAdd) err=0x{:x}", err);
    println!("graphAddNode(ElementWiseAdd): OK");

    // ── 8. graphFinalize ──
    let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
    anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);
    println!("graphFinalize: OK");

    // ── 9. graphExecute (attach client buffers now) ──
    unsafe {
        inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
            data: host_a.as_mut_ptr() as *mut _,
            dataSize: (N * 4) as u32,
        };
        inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
            data: host_b.as_mut_ptr() as *mut _,
            dataSize: (N * 4) as u32,
        };
        outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
            data: host_c.as_mut_ptr() as *mut _,
            dataSize: (N * 4) as u32,
        };
    }
    let err = unsafe {
        (v.graphExecute.unwrap())(
            graph,
            inputs.as_ptr(),
            inputs.len() as u32,
            outputs.as_mut_ptr(),
            outputs.len() as u32,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);
    println!("graphExecute: OK");

    // ── 10. Verify ──
    let mut mismatch = 0usize;
    let mut first_bad = None;
    for i in 0..N {
        let exp = host_a[i] + host_b[i];
        if (host_c[i] - exp).abs() > 1e-4 * exp.abs().max(1.0) {
            mismatch += 1;
            if first_bad.is_none() {
                first_bad = Some((i, host_c[i], exp));
            }
        }
    }
    if mismatch == 0 {
        println!(
            "\n=== Q1 ANSWER: ✓ correct ({}/{} elements match) ===",
            N, N
        );
    } else {
        println!("\n=== Q1 ANSWER: ✗ MISMATCH {}/{} ===", mismatch, N);
        if let Some((i, got, exp)) = first_bad {
            println!("  first bad: c[{}]={} (expected {})", i, got, exp);
        }
    }

    // ── 11. Cleanup ──
    unsafe {
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(backend);
    }

    if mismatch == 0 {
        Ok(())
    } else {
        std::process::exit(1)
    }
}
