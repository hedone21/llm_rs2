//! microbench_oppkg_dynamic_shape — R-A3: KV cache dynamic shape PoC
//!
//! 검증:
//!   1. 다양한 K 차원 graph 별 graphFinalize cost
//!   2. graph의 tensor shape이 build 시 fixed인지, execute 시 변경 가능한지
//!   3. cross-shape execute 시도 (K=1024 graph에 K=512 data) — error 또는 ignore?
//!
//! 결과 해석:
//!   - 모든 K shape에서 finalize <10ms → per-decode-step rebuild 가능 (dynamic)
//!   - finalize >100ms → fixed shape graph만 가능, max-padded 또는 prebuilt set 필요

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("requires --features qnn");
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
    use std::ffi::{CString, c_void};
    use std::os::raw::c_uint;
    use std::ptr;
    use std::time::Instant;

    use qnn::*;

    const PKG_PATH: &str = "/data/local/tmp/libqnn_oppkg_poc.so";
    const PKG_PROVIDER: &str = "QnnOpPackage_InitInterface";
    const PKG_TARGET: &str = "GPU";

    let m: usize = 1;
    let n: usize = 1024;
    let k_values: [usize; 5] = [128, 512, 1024, 2048, 4096];
    let n_iters: usize = 5;

    println!("=== microbench_oppkg_dynamic_shape (R-A3) ===\n");
    println!("M={} N={} (square output dim), varying K", m, n);

    let gpu_lib = unsafe { Library::new("/data/local/tmp/qnn/libQnnGpu.so") }?;
    type GetProvidersFn = unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let gp: Symbol<GetProvidersFn> = unsafe { gpu_lib.get(b"QnnInterface_getProviders\0")? };
    let mut provs: *mut *const QnnInterface_t = ptr::null_mut();
    let mut np: c_uint = 0;
    let err = unsafe { gp(&mut provs, &mut np) };
    anyhow::ensure!(err == 0 && np > 0);
    let v = unsafe { (**provs).__bindgen_anon_1.v2_25 };

    let mut be: Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe { (v.backendCreate.unwrap())(ptr::null_mut(), ptr::null_mut(), &mut be) };
    anyhow::ensure!(err == 0);

    let pkg_path = CString::new(PKG_PATH).unwrap();
    let pkg_provider = CString::new(PKG_PROVIDER).unwrap();
    let pkg_target = CString::new(PKG_TARGET).unwrap();
    let err = unsafe {
        (v.backendRegisterOpPackage.unwrap())(be, pkg_path.as_ptr(), pkg_provider.as_ptr(), pkg_target.as_ptr())
    };
    anyhow::ensure!(err == 0, "registerOpPackage err=0x{:x}", err);

    println!("\n--- Test 1: graphFinalize cost vs K ---\n");
    println!("{:<8} {:>16} {:>16}", "K", "build (ms)", "finalize (ms)");
    println!("{}", "-".repeat(45));

    let mut finalize_times = Vec::new();
    for &k in k_values.iter() {
        let mut samples_b = Vec::new();
        let mut samples_f = Vec::new();
        for _ in 0..n_iters {
            let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
            unsafe { (v.contextCreate.unwrap())(be, ptr::null_mut(), ptr::null_mut(), &mut ctx) };

            let g_name = CString::new(format!("g_k{}", k)).unwrap();
            let mut graph: Qnn_GraphHandle_t = ptr::null_mut();

            let t_build0 = Instant::now();
            let err = unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
            anyhow::ensure!(err == 0);

            let dims_w: Vec<u32> = vec![n as u32, k as u32];
            let dims_x: Vec<u32> = vec![m as u32, k as u32];
            let dims_y: Vec<u32> = vec![m as u32, n as u32];
            let mk_v1 = |name: &CString, ttype: Qnn_TensorType_t, dt: Qnn_DataType_t, dims: &[u32]| -> Qnn_TensorV1_t {
                Qnn_TensorV1_t {
                    id: 0, name: name.as_ptr(), type_: ttype,
                    dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                    dataType: dt,
                    quantizeParams: Qnn_QuantizeParams_t {
                        encodingDefinition: Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
                        quantizationEncoding: Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
                        __bindgen_anon_1: Qnn_QuantizeParams_t__bindgen_ty_1 {
                            scaleOffsetEncoding: Qnn_ScaleOffset_t { scale: 0.0, offset: 0 },
                        },
                    },
                    rank: dims.len() as u32, dimensions: dims.as_ptr() as *mut u32,
                    memType: Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
                    __bindgen_anon_1: Qnn_TensorV1_t__bindgen_ty_1 {
                        clientBuf: Qnn_ClientBuffer_t { data: ptr::null_mut(), dataSize: 0 },
                    },
                }
            };
            let n_w = CString::new("W").unwrap();
            let n_x = CString::new("X").unwrap();
            let n_y = CString::new("Y").unwrap();
            let mut t_w = Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_v1(&n_w, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, &dims_w),
                },
            };
            let mut t_x = Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_v1(&n_x, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_x),
                },
            };
            let mut t_y = Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_v1(&n_y, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_y),
                },
            };
            for t in [&mut t_w, &mut t_x, &mut t_y] {
                let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
                anyhow::ensure!(err == 0);
            }
            let op_name = CString::new("mm").unwrap();
            let pkg = CString::new("qnn_oppkg_poc").unwrap();
            let op_type = CString::new("CustomMatMul").unwrap();
            let mut inputs = [t_w, t_x];
            let mut outputs = [t_y];
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
            anyhow::ensure!(err == 0);
            let build_ms = t_build0.elapsed().as_secs_f64() * 1000.0;

            let t_fin0 = Instant::now();
            let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
            anyhow::ensure!(err == 0);
            let fin_ms = t_fin0.elapsed().as_secs_f64() * 1000.0;

            samples_b.push(build_ms);
            samples_f.push(fin_ms);

            unsafe { (v.contextFree.unwrap())(ctx, ptr::null_mut()) };
        }
        let b_mean = samples_b.iter().sum::<f64>() / n_iters as f64;
        let f_mean = samples_f.iter().sum::<f64>() / n_iters as f64;
        finalize_times.push((k, b_mean, f_mean));
        println!("{:<8} {:>16.2} {:>16.2}", k, b_mean, f_mean);
    }

    println!("\n--- Decision ---");
    let max_fin = finalize_times.iter().map(|&(_, _, f)| f).fold(0.0, f64::max);
    if max_fin < 10.0 {
        println!("✓ GREEN: finalize <10ms — per-decode-step rebuild OK (dynamic)");
    } else if max_fin < 100.0 {
        println!("△ YELLOW: finalize {:.1}ms — prebuilt graph set OK, per-step rebuild marginal", max_fin);
    } else {
        println!("✗ RED: finalize {:.1}ms — must use max-padded fixed shape", max_fin);
    }
    println!();
    println!("Production interpretation:");
    println!("  decode TBT ~28ms. graphFinalize {:.1}ms = {:.1}% of TBT", max_fin, max_fin / 28.0 * 100.0);
    println!("  → If finalize > TBT, must avoid per-step rebuild (use max-padded or graph cache)");

    unsafe { (v.backendFree.unwrap())(be) };
    Ok(())
}
