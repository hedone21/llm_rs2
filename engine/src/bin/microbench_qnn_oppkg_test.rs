//! microbench_qnn_oppkg_test — Phase R Wave 2: drives qnn_oppkg_poc.so
//!
//! Goals:
//!   GA: registerOpPackage successfully loads /data/local/tmp/libqnn_oppkg_poc.so
//!   GB: graphAddNode with op_type="CustomAdd" passes validateOpConfig
//!   GC: graphExecute runs our OpenCL kernel (host_a + host_b → host_c)
//!   GD (deferred): cl_mem sharing with prebuilt op
//!
//! Build: cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!        --bin microbench_qnn_oppkg_test
//!
//! Pre-deploy steps (host):
//!   cargo build --release -p qnn_oppkg_poc --target aarch64-linux-android
//!   adb push target/aarch64-linux-android/release/libqnn_oppkg_poc.so /data/local/tmp/
//!
//! Run: adb shell "LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:/vendor/lib64 \
//!                  /data/local/tmp/microbench_qnn_oppkg_test"

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_oppkg_test requires --features qnn");
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

    use qnn::*;

    const PKG_PATH: &str = "/data/local/tmp/libqnn_oppkg_poc.so";
    const PKG_PROVIDER: &str = "QnnOpPackage_InitInterface";
    const PKG_TARGET: &str = "GPU_QTI_AISW";

    let n: usize = 1024;

    println!("=== microbench_qnn_oppkg_test ===\n");
    println!("Op Package .so: {}", PKG_PATH);
    println!("Interface symbol: {}", PKG_PROVIDER);
    println!("Vector length: {}\n", n);

    // For diagnosis: try CPU backend first (toggle via env)
    let backend_lib_path = std::env::var("QNN_BACKEND_LIB")
        .unwrap_or_else(|_| "/data/local/tmp/qnn/libQnnGpu.so".to_string());
    println!("Backend lib: {}", backend_lib_path);
    let gpu_lib = unsafe { Library::new(&backend_lib_path) }?;
    type GetProvidersFn =
        unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let gp: Symbol<GetProvidersFn> = unsafe { gpu_lib.get(b"QnnInterface_getProviders\0")? };
    let mut provs: *mut *const QnnInterface_t = ptr::null_mut();
    let mut np: c_uint = 0;
    let err = unsafe { gp(&mut provs, &mut np) };
    anyhow::ensure!(err == 0 && np > 0, "GPU getProviders err=0x{:x}", err);
    let v = unsafe { (**provs).__bindgen_anon_1.v2_25 };

    let mut be: Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe { (v.backendCreate.unwrap())(ptr::null_mut(), ptr::null_mut(), &mut be) };
    anyhow::ensure!(err == 0, "backendCreate err=0x{:x}", err);
    println!("backend: OK");
    println!(
        "  backendRegisterOpPackage fn = {:?}",
        v.backendRegisterOpPackage.map(|f| f as usize)
    );

    // GA: registerOpPackage
    let pkg_path = CString::new(PKG_PATH).unwrap();
    let pkg_provider = CString::new(PKG_PROVIDER).unwrap();
    let pkg_target = CString::new(PKG_TARGET).unwrap();
    let reg_fn = match v.backendRegisterOpPackage {
        Some(f) => f,
        None => {
            eprintln!("backendRegisterOpPackage fn pointer is NULL");
            std::process::exit(1);
        }
    };
    eprintln!("about to call registerOpPackage...");
    let err = unsafe {
        reg_fn(
            be,
            pkg_path.as_ptr(),
            pkg_provider.as_ptr(),
            pkg_target.as_ptr(),
        )
    };
    eprintln!("returned from registerOpPackage");
    println!("GA: registerOpPackage -> err = 0x{:x}", err);
    anyhow::ensure!(err == 0, "registerOpPackage err=0x{:x}", err);
    println!("    ✓ PASS");

    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe {
        (v.contextCreate.unwrap())(be, ptr::null_mut(), ptr::null_mut(), &mut ctx)
    };
    anyhow::ensure!(err == 0, "contextCreate err=0x{:x}", err);
    println!("context: OK");

    let g_name = CString::new("custom_add_graph").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err = unsafe {
        (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph)
    };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    let dims: Vec<u32> = vec![n as u32];
    let mk_v1 =
        |name: &CString, ttype: Qnn_TensorType_t, dims: &[u32]| -> Qnn_TensorV1_t {
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
    let n_a = CString::new("A").unwrap();
    let n_b = CString::new("B").unwrap();
    let n_c = CString::new("C").unwrap();
    let mut t_a = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1(&n_a, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims),
        },
    };
    let mut t_b = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1(&n_b, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims),
        },
    };
    let mut t_c = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1(&n_c, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, &dims),
        },
    };
    for (l, t) in [("A", &mut t_a), ("B", &mut t_b), ("C", &mut t_c)] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", l, err);
    }

    // GB: graphAddNode with our custom op_type
    let op_name = CString::new("add0").unwrap();
    let pkg = CString::new("qnn_oppkg_poc").unwrap();
    let op_type = CString::new("CustomAdd").unwrap();
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
    println!("GB: graphAddNode(CustomAdd) -> err = 0x{:x}", err);
    anyhow::ensure!(err == 0, "graphAddNode err=0x{:x}", err);
    println!("    ✓ PASS");

    let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
    println!("    graphFinalize -> err = 0x{:x}", err);
    anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);

    // Use rpcmem + DMA_BUF for IO (QNN-GPU requires MEMHANDLE in our setup)
    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;
    type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(po: *const c_void) -> i32;
    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }?;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let _rpcmem_free: Symbol<RpcmemFreeFn> = unsafe { rpc_lib.get(b"rpcmem_free\0")? };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };

    let bytes = (n * 4) as i32;
    let rpc_a = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    let rpc_b = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    let rpc_c = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    anyhow::ensure!(
        !rpc_a.is_null() && !rpc_b.is_null() && !rpc_c.is_null(),
        "rpcmem_alloc fail"
    );
    let fd_a = unsafe { rpcmem_to_fd(rpc_a) };
    let fd_b = unsafe { rpcmem_to_fd(rpc_b) };
    let fd_c = unsafe { rpcmem_to_fd(rpc_c) };

    // Fill inputs
    let host_a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 0.5).collect();
    let host_b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0007 + 0.13).collect();
    unsafe {
        std::ptr::copy_nonoverlapping(
            host_a.as_ptr() as *const u8,
            rpc_a as *mut u8,
            bytes as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_b.as_ptr() as *const u8,
            rpc_b as *mut u8,
            bytes as usize,
        );
    }

    let mk_desc = |fd: i32, host_data: *mut c_void| -> Qnn_MemDescriptor_t {
        Qnn_MemDescriptor_t {
            memShape: Qnn_MemShape_t {
                numDim: 1,
                dimSize: dims.as_ptr() as *mut u32,
                shapeConfig: ptr::null(),
            },
            dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            memType: Qnn_MemType_t_QNN_MEM_TYPE_DMA_BUF,
            __bindgen_anon_1: Qnn_MemDescriptor_t__bindgen_ty_1 {
                dmaBufInfo: Qnn_MemDmaBufInfo_t { fd, data: host_data },
            },
        }
    };
    let descs = [mk_desc(fd_a, rpc_a), mk_desc(fd_b, rpc_b), mk_desc(fd_c, rpc_c)];
    let mut mh = [ptr::null_mut::<c_void>(); 3];
    let err = unsafe {
        (v.memRegister.unwrap())(ctx, descs.as_ptr(), 3, mh.as_mut_ptr())
    };
    anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);
    inputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[0];
    inputs[1].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[1];
    outputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[2];

    // GC: graphExecute
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
    println!("GC: graphExecute -> err = 0x{:x}", err);
    anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);

    // Verify
    let mut max_abs = 0.0f32;
    let mut mismatch = 0usize;
    unsafe {
        let c_slice = std::slice::from_raw_parts(rpc_c as *const f32, n);
        for i in 0..n {
            let exp = host_a[i] + host_b[i];
            let d = (c_slice[i] - exp).abs();
            if d > max_abs {
                max_abs = d;
            }
            if d > 1e-4 {
                mismatch += 1;
            }
        }
    }
    println!(
        "    correctness: max_abs_err = {:.6}, mismatch (>1e-4) = {} / {}",
        max_abs, mismatch, n
    );
    let gc_pass = mismatch == 0;
    println!(
        "    {}",
        if gc_pass { "✓ PASS — custom OpenCL kernel ran inside QNN-GPU runtime" } else { "✗ FAIL" }
    );

    println!("\n=== Summary ===");
    println!("GA registerOpPackage: PASS");
    println!("GB graphAddNode:      PASS");
    println!("GC graphExecute:      {}", if gc_pass { "PASS" } else { "FAIL" });

    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), 3);
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(be);
    }
    if gc_pass { Ok(()) } else { std::process::exit(1) }
}
