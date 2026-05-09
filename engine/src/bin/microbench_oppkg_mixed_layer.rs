//! microbench_oppkg_mixed_layer — R-Y1: mixed-op layer graph
//!
//! Compare:
//!   - Homogeneous: 12 CustomMatMul ops in chain
//!   - Heterogeneous: 6 CustomMatMul + 6 CustomAdd interleaved
//!
//! production layer pattern simulation. graphFinalize + execute cost.
//! Hypothesis: mixed graph offers different optimizer surface than chain of identical ops.

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
    let k: usize = 1024;
    let n: usize = 1024;
    let n_iters: usize = 30;
    // Note: CustomAdd uses F32 inputs; CustomMatMul uses F16 weight + F32 input/output.
    // Build mixed graph alternating MatMul → Add. y_{i+1} = MatMul(W, y_i); z_{i+1} = Add(y_{i+1}, bias).

    println!("=== microbench_oppkg_mixed_layer (R-Y1) ===\n");
    println!("M={} K={} N={} (square)", m, k, n);
    println!("Mixed: 6 MatMul + 6 Add = 12 op layer pattern\n");

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
    anyhow::ensure!(err == 0);

    // Helper to build either homogeneous or mixed graph
    let build_and_measure = |label: &str, mixed: bool, num_ops: usize| -> anyhow::Result<()> {
        let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
        let err = unsafe { (v.contextCreate.unwrap())(be, ptr::null_mut(), ptr::null_mut(), &mut ctx) };
        anyhow::ensure!(err == 0);

        let g_name = CString::new(format!("{}_g", label)).unwrap();
        let mut graph: Qnn_GraphHandle_t = ptr::null_mut();

        let t_total = Instant::now();
        let err = unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
        anyhow::ensure!(err == 0);

        let dims_w: Vec<u32> = vec![n as u32, k as u32];
        let dims_v: Vec<u32> = vec![m as u32, n as u32];
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
        let n_b = CString::new("bias").unwrap();
        let mut t_w = Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_v1(&n_w, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, &dims_w),
            },
        };
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, &mut t_w) };
        anyhow::ensure!(err == 0);

        let mut t_b = Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_v1(&n_b, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_v),
            },
        };
        if mixed {
            let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, &mut t_b) };
            anyhow::ensure!(err == 0);
        }

        let int_names: Vec<CString> = (0..=num_ops).map(|i| CString::new(format!("y{}", i)).unwrap()).collect();
        let mut y_tensors: Vec<Qnn_Tensor_t> = Vec::with_capacity(num_ops + 1);
        for i in 0..=num_ops {
            let ttype = if i == 0 { Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE }
                        else if i == num_ops { Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ }
                        else { Qnn_TensorType_t_QNN_TENSOR_TYPE_NATIVE };
            let dims = if i == 0 { vec![m as u32, k as u32] } else { dims_v.clone() };
            let mut t = Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_v1(&int_names[i], ttype, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims),
                },
            };
            let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, &mut t) };
            anyhow::ensure!(err == 0);
            y_tensors.push(t);
        }

        let pkg = CString::new("qnn_oppkg_poc").unwrap();
        let op_type_mm = CString::new("CustomMatMul").unwrap();
        let op_type_add = CString::new("CustomAdd").unwrap();
        let op_names: Vec<CString> = (0..num_ops).map(|i| CString::new(format!("op_{}", i)).unwrap()).collect();
        let mut inputs_h: Vec<Vec<Qnn_Tensor_t>> = Vec::with_capacity(num_ops);
        let mut outputs_h: Vec<[Qnn_Tensor_t; 1]> = Vec::with_capacity(num_ops);
        for i in 0..num_ops {
            let is_add = mixed && (i % 2 == 1);
            if is_add {
                inputs_h.push(vec![y_tensors[i], t_b]);
            } else {
                inputs_h.push(vec![t_w, y_tensors[i]]);
            }
            outputs_h.push([y_tensors[i + 1]]);
        }
        for i in 0..num_ops {
            let is_add = mixed && (i % 2 == 1);
            let op = Qnn_OpConfig_t {
                version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
                __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
                    v1: Qnn_OpConfigV1_t {
                        name: op_names[i].as_ptr(),
                        packageName: pkg.as_ptr(),
                        typeName: if is_add { op_type_add.as_ptr() } else { op_type_mm.as_ptr() },
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
            anyhow::ensure!(err == 0, "graphAddNode[{}] err=0x{:x}", i, err);
        }
        let build_ms = t_total.elapsed().as_secs_f64() * 1000.0;

        let t_fin = Instant::now();
        let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
        anyhow::ensure!(err == 0);
        let fin_ms = t_fin.elapsed().as_secs_f64() * 1000.0;

        // rpcmem for endpoints
        const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
        const RPCMEM_DEFAULT_FLAGS: u32 = 1;
        type AllocFn = unsafe extern "C" fn(i32, u32, i32) -> *mut c_void;
        type ToFdFn = unsafe extern "C" fn(*const c_void) -> i32;
        let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }?;
        let rpc_alloc: Symbol<AllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
        let rpc_to_fd: Symbol<ToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };
        let bw = (n * k * 2) as i32;
        let bv = (m * n * 4) as i32;
        let bx = (m * k * 4) as i32;
        let rpc_w = unsafe { rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bw) };
        let rpc_b = unsafe { rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bv) };
        let rpc_y0 = unsafe { rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bx) };
        let rpc_yn = unsafe { rpc_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bv) };
        anyhow::ensure!(!rpc_w.is_null() && !rpc_b.is_null() && !rpc_y0.is_null() && !rpc_yn.is_null());
        let fd_w = unsafe { rpc_to_fd(rpc_w) };
        let fd_b = unsafe { rpc_to_fd(rpc_b) };
        let fd_y0 = unsafe { rpc_to_fd(rpc_y0) };
        let fd_yn = unsafe { rpc_to_fd(rpc_yn) };
        let mk_desc = |fd: i32, host: *mut c_void, dt: Qnn_DataType_t, dims: &[u32]| -> Qnn_MemDescriptor_t {
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
        let dims_x: Vec<u32> = vec![m as u32, k as u32];
        let descs_v = if mixed {
            vec![
                mk_desc(fd_w, rpc_w, Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, &dims_w),
                mk_desc(fd_b, rpc_b, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_v),
                mk_desc(fd_y0, rpc_y0, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_x),
                mk_desc(fd_yn, rpc_yn, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_v),
            ]
        } else {
            vec![
                mk_desc(fd_w, rpc_w, Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, &dims_w),
                mk_desc(fd_y0, rpc_y0, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_x),
                mk_desc(fd_yn, rpc_yn, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, &dims_v),
            ]
        };
        let mut mh = vec![ptr::null_mut::<c_void>(); descs_v.len()];
        let err = unsafe { (v.memRegister.unwrap())(ctx, descs_v.as_ptr(), descs_v.len() as u32, mh.as_mut_ptr()) };
        anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);

        let mut t_w_mh = t_w;
        t_w_mh.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        t_w_mh.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[0];
        let (mut t_b_mh, t_y0_idx, t_yn_idx) = if mixed {
            let mut tb = t_b;
            tb.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
            tb.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[1];
            (tb, 2, 3)
        } else {
            (t_b, 1, 2)
        };
        let mut t_y0_mh = y_tensors[0];
        t_y0_mh.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        t_y0_mh.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[t_y0_idx];
        let mut t_yn_mh = y_tensors[num_ops];
        t_yn_mh.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        t_yn_mh.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[t_yn_idx];

        let exec_inputs = if mixed { vec![t_w_mh, t_b_mh, t_y0_mh] } else { vec![t_w_mh, t_y0_mh] };
        let mut exec_outputs = [t_yn_mh];
        let v_ptr = &v as *const _ as usize;
        let graph_ptr = graph as usize;
        let in_ptr = exec_inputs.as_ptr() as usize;
        let in_count = exec_inputs.len() as u32;
        let out_ptr = exec_outputs.as_mut_ptr() as usize;

        let exec = || -> anyhow::Result<f64> {
            let v = unsafe { &*(v_ptr as *const QnnInterface_ImplementationV2_25_t) };
            let t0 = Instant::now();
            let err = unsafe {
                (v.graphExecute.unwrap())(
                    graph_ptr as Qnn_GraphHandle_t,
                    in_ptr as *const Qnn_Tensor_t, in_count,
                    out_ptr as *mut Qnn_Tensor_t, 1,
                    ptr::null_mut(), ptr::null_mut(),
                )
            };
            anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);
            Ok(t0.elapsed().as_secs_f64() * 1000.0)
        };

        for _ in 0..5 { let _ = exec()?; }
        let mut samples = Vec::with_capacity(n_iters);
        for _ in 0..n_iters { samples.push(exec()?); }
        let mean = samples.iter().sum::<f64>() / n_iters as f64;
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];

        println!(
            "{:<24} ops={:>2} build={:.2}ms finalize={:.2}ms exec mean={:.3}ms median={:.3}ms per-op={:.3}ms",
            label, num_ops, build_ms, fin_ms, mean, median, mean / num_ops as f64
        );

        unsafe {
            let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), mh.len() as u32);
            let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        }
        Ok(())
    };

    println!("Test pass:");
    println!();
    build_and_measure("Homogeneous (12 mm)", false, 12)?;
    build_and_measure("Mixed (6mm + 6add)", true, 12)?;
    build_and_measure("Homogeneous (16 mm)", false, 16)?;

    println!("\nNote: per-op cost difference shows graph optimizer effect on mixed workloads.");

    unsafe { (v.backendFree.unwrap())(be) };
    Ok(())
}
