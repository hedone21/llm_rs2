//! M2.H B4 — Prebuilt op chain composition correctness.
//!
//! Goal: distinguish between **SDK general limitation** vs **OpPackage-only
//! limitation** for graph composition. M2.H 6th + P0 found that an OpPackage
//! 5-op chain RED while standalone ops were GREEN; cause was suspected to be
//! either (a) SDK does not enforce host visibility on NATIVE intermediates
//! across nodes, or (b) OpPackage-specific incompatibility (Memory Object
//! contract, kernel layout assumptions, etc).
//!
//! This benchmark tests **prebuilt** ops only (no `registerOpPackage`). Chain:
//!
//!   x ── ElementWiseAdd ── t_mid ── ElementWiseAdd ── t_out
//!         ▲                            ▲
//!         y                            z
//!
//! Tensor types:
//!   x, y, z [1, dim] FLOAT_32   APP_WRITE  (graph inputs, host-visible)
//!   t_mid   [1, dim] FLOAT_32   NATIVE     (intermediate, driver-managed)
//!   t_out   [1, dim] FLOAT_32   APP_READ   (graph endpoint, host-visible)
//!
//! Why ElementWiseAdd, not MatMul + ReLU:
//!   - No params required (MatMul needs `transpose_in0/1`, etc).
//!   - Same 2 inputs, 1 output signature as `CustomAdd`, so the boilerplate
//!     reuses the M1.3/M1.9 patterns.
//!   - Failure here cleanly isolates "graph composition itself" from any
//!     param-binding fragility.
//!
//! Pass criterion: max_abs_err < 1e-5 for all dims tested. Bands:
//!   `< 1e-5`           — exact (only FP rounding)             → GREEN
//!   `[1e-5, 1e-3)`     — drift, possibly composition-related  → YELLOW
//!   `>= 1e-3` or graph never produces data → RED
//!
//! Verdict implications:
//!   GREEN — SDK handles prebuilt op graph composition correctly. The M2.H 6th
//!   failure is therefore **OpPackage-specific** (Memory Object, kernel layout,
//!   or OpPackage chain handling).
//!
//!   RED   — SDK itself does not deliver correct chain composition (NATIVE
//!   intermediates lose data across node boundaries even for prebuilt ops).
//!   In that case the OpPackage path inherits this fundamental SDK constraint
//!   and should be deprecated for chain workloads.
//!
//! Build (Android cross):
//!   cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!     --bin microbench_qnn_prebuilt_chain_correct
//!
//! Run on device:
//!   adb shell "LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:/vendor/lib64 \
//!              /data/local/tmp/microbench_qnn_prebuilt_chain_correct"

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_prebuilt_chain_correct requires --features qnn");
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
    use std::ffi::c_void;
    use std::os::raw::c_uint;
    use std::ptr;

    use qnn::*;

    println!("=== microbench_qnn_prebuilt_chain_correct (M2.H B4) ===\n");
    println!("Purpose: distinguish SDK general limitation vs OpPackage-only");
    println!("         limitation for graph composition.");
    println!("Chain:   x + y -> t_mid; t_mid + z -> t_out (prebuilt ElementWiseAdd x2)");
    println!("Package: \"qti.aisw\" (no registerOpPackage call)\n");

    let backend_lib_path = std::env::var("QNN_BACKEND_LIB")
        .unwrap_or_else(|_| "/data/local/tmp/qnn/libQnnGpu.so".to_string());
    println!("Backend lib: {}\n", backend_lib_path);

    let gpu_lib = unsafe { Library::new(&backend_lib_path) }?;
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
    println!("backend: OK (no OpPackage registration — prebuilt ops only)\n");

    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe { (v.contextCreate.unwrap())(be, ptr::null_mut(), ptr::null_mut(), &mut ctx) };
    anyhow::ensure!(err == 0, "contextCreate err=0x{:x}", err);

    // rpcmem for DMA_BUF tensors (same as M1.3)
    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;
    type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(po: *const c_void) -> i32;
    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }?;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let _rpcmem_free: Symbol<RpcmemFreeFn> = unsafe { rpc_lib.get(b"rpcmem_free\0")? };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };

    // Test sizes — multiples of 4 (float4 friendly), small to fit any GPU pipeline
    let cases: &[usize] = &[64, 256, 1024];
    let mut all_pass = true;

    for &n in cases {
        println!("--- dim = {} ---", n);
        let result = run_case(
            &v,
            ctx,
            &rpcmem_alloc,
            &rpcmem_to_fd,
            RPCMEM_HEAP_ID_SYSTEM,
            RPCMEM_DEFAULT_FLAGS,
            n,
        );
        match result {
            Ok(max_err) => {
                let pass = max_err < 1e-5;
                let band = if max_err < 1e-5 {
                    "GREEN"
                } else if max_err < 1e-3 {
                    "YELLOW"
                } else {
                    "RED"
                };
                println!(
                    "  max_abs_err = {:.6e}  [{}] {}",
                    max_err,
                    band,
                    if pass { "PASS" } else { "FAIL" }
                );
                if !pass {
                    all_pass = false;
                }
            }
            Err(e) => {
                println!("  ERROR: {}", e);
                all_pass = false;
            }
        }
    }

    let verdict = if all_pass { "GREEN" } else { "RED" };
    println!("\n=== M2.H B4 verdict: {} ===", verdict);
    if all_pass {
        println!("Implication: SDK handles prebuilt op chain composition correctly.");
        println!("             OpPackage path failure (M2.H 6th) is OpPackage-specific");
        println!("             (Memory Object / kernel layout / chain handling), not");
        println!("             a fundamental SDK limit. OpPackage path needs further work.");
    } else {
        println!("Implication: SDK does NOT correctly handle graph composition even for");
        println!("             prebuilt ops. NATIVE intermediates lose data across nodes.");
        println!("             OpPackage path inherits this constraint — recommend");
        println!("             deprecating OpPackage chain workloads.");
    }

    unsafe {
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(be);
    }

    if all_pass {
        Ok(())
    } else {
        std::process::exit(1)
    }
}

#[cfg(feature = "qnn")]
fn run_case(
    v: &qnn::QnnInterface_ImplementationV2_25_t,
    ctx: qnn::Qnn_ContextHandle_t,
    rpcmem_alloc: &libloading::Symbol<unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void>,
    rpcmem_to_fd: &libloading::Symbol<unsafe extern "C" fn(*const std::ffi::c_void) -> i32>,
    heap_id: i32,
    flags: u32,
    n: usize,
) -> anyhow::Result<f32> {
    use qnn::*;
    use std::ffi::CString;
    use std::ptr;

    // ── Allocate rpcmem buffers ──────────────────────────────────────────────
    let bytes = (n * 4) as i32;
    let rpc_x = unsafe { rpcmem_alloc(heap_id, flags, bytes) };
    let rpc_y = unsafe { rpcmem_alloc(heap_id, flags, bytes) };
    let rpc_z = unsafe { rpcmem_alloc(heap_id, flags, bytes) };
    let rpc_out = unsafe { rpcmem_alloc(heap_id, flags, bytes) };
    anyhow::ensure!(
        !rpc_x.is_null() && !rpc_y.is_null() && !rpc_z.is_null() && !rpc_out.is_null(),
        "rpcmem_alloc failed for n={}",
        n
    );

    // Distinct host data so the chain has a unique answer per index
    let host_x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 0.5).collect();
    let host_y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0007 + 0.13).collect();
    let host_z: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0005 - 0.21).collect();
    unsafe {
        std::ptr::copy_nonoverlapping(
            host_x.as_ptr() as *const u8,
            rpc_x as *mut u8,
            bytes as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_y.as_ptr() as *const u8,
            rpc_y as *mut u8,
            bytes as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_z.as_ptr() as *const u8,
            rpc_z as *mut u8,
            bytes as usize,
        );
        // Zero out the output buffer so a "graph never wrote" failure is visible
        std::ptr::write_bytes(rpc_out as *mut u8, 0, bytes as usize);
    }

    let fd_x = unsafe { rpcmem_to_fd(rpc_x) };
    let fd_y = unsafe { rpcmem_to_fd(rpc_y) };
    let fd_z = unsafe { rpcmem_to_fd(rpc_z) };
    let fd_out = unsafe { rpcmem_to_fd(rpc_out) };

    // ── Tensor descriptors ───────────────────────────────────────────────────
    // Use rank-2 [1, n] to match M1.9 chain conventions; ElementWiseAdd
    // broadcasts/aligns by rank.
    let mut dims = vec![1u32, n as u32];

    let qp = Qnn_QuantizeParams_t {
        encodingDefinition: Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
        quantizationEncoding: Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
        __bindgen_anon_1: Qnn_QuantizeParams_t__bindgen_ty_1 {
            scaleOffsetEncoding: Qnn_ScaleOffset_t {
                scale: 0.0,
                offset: 0,
            },
        },
    };
    let mk_tv1 = |ttype, dims_ptr: *mut u32| Qnn_TensorV1_t {
        id: 0,
        name: ptr::null(),
        type_: ttype,
        dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
        quantizeParams: qp,
        rank: 2,
        dimensions: dims_ptr,
        memType: Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
        __bindgen_anon_1: Qnn_TensorV1_t__bindgen_ty_1 {
            clientBuf: Qnn_ClientBuffer_t {
                data: ptr::null_mut(),
                dataSize: 0,
            },
        },
    };

    let dims_ptr = dims.as_mut_ptr();
    let name_x = CString::new(format!("pre_x_{}", n)).unwrap();
    let name_y = CString::new(format!("pre_y_{}", n)).unwrap();
    let name_z = CString::new(format!("pre_z_{}", n)).unwrap();
    let name_mid = CString::new(format!("pre_mid_{}", n)).unwrap();
    let name_out = CString::new(format!("pre_out_{}", n)).unwrap();

    let mut t_x = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, dims_ptr),
        },
    };
    t_x.__bindgen_anon_1.v1.name = name_x.as_ptr();
    let mut t_y = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, dims_ptr),
        },
    };
    t_y.__bindgen_anon_1.v1.name = name_y.as_ptr();
    let mut t_z = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, dims_ptr),
        },
    };
    t_z.__bindgen_anon_1.v1.name = name_z.as_ptr();
    let mut t_mid = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(Qnn_TensorType_t_QNN_TENSOR_TYPE_NATIVE, dims_ptr),
        },
    };
    t_mid.__bindgen_anon_1.v1.name = name_mid.as_ptr();
    let mut t_out = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, dims_ptr),
        },
    };
    t_out.__bindgen_anon_1.v1.name = name_out.as_ptr();

    // ── Build graph ──────────────────────────────────────────────────────────
    let g_name = CString::new(format!("prebuilt_chain_{}", n)).unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    for (label, t) in [
        ("x", &mut t_x),
        ("y", &mut t_y),
        ("z", &mut t_z),
        ("mid", &mut t_mid),
        ("out", &mut t_out),
    ] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", label, err);
    }

    // Op definitions — prebuilt ElementWiseAdd from "qti.aisw"
    let pkg = CString::new("qti.aisw").unwrap();
    let op_type = CString::new("ElementWiseAdd").unwrap();
    let op_name_a = CString::new(format!("ewa_a_{}", n)).unwrap();
    let op_name_b = CString::new(format!("ewa_b_{}", n)).unwrap();

    // Node A: x + y -> t_mid
    let mut in_a = [t_x, t_y];
    let mut out_a = [t_mid];
    let op_a = Qnn_OpConfig_t {
        version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
        __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
            v1: Qnn_OpConfigV1_t {
                name: op_name_a.as_ptr(),
                packageName: pkg.as_ptr(),
                typeName: op_type.as_ptr(),
                numOfParams: 0,
                params: ptr::null_mut(),
                numOfInputs: 2,
                inputTensors: in_a.as_mut_ptr(),
                numOfOutputs: 1,
                outputTensors: out_a.as_mut_ptr(),
            },
        },
    };
    let err = unsafe { (v.graphAddNode.unwrap())(graph, op_a) };
    anyhow::ensure!(err == 0, "graphAddNode(node_a) err=0x{:x}", err);

    // Node B: t_mid + z -> t_out
    let mut in_b = [t_mid, t_z];
    let mut out_b = [t_out];
    let op_b = Qnn_OpConfig_t {
        version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
        __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
            v1: Qnn_OpConfigV1_t {
                name: op_name_b.as_ptr(),
                packageName: pkg.as_ptr(),
                typeName: op_type.as_ptr(),
                numOfParams: 0,
                params: ptr::null_mut(),
                numOfInputs: 2,
                inputTensors: in_b.as_mut_ptr(),
                numOfOutputs: 1,
                outputTensors: out_b.as_mut_ptr(),
            },
        },
    };
    let err = unsafe { (v.graphAddNode.unwrap())(graph, op_b) };
    anyhow::ensure!(err == 0, "graphAddNode(node_b) err=0x{:x}", err);

    let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
    anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);

    // ── memRegister: 4 host-backed tensors (x, y, z inputs + out) ────────────
    let mk_desc = |fd: i32, host_data: *mut std::ffi::c_void| Qnn_MemDescriptor_t {
        memShape: Qnn_MemShape_t {
            numDim: 2,
            dimSize: dims.as_ptr() as *mut u32,
            shapeConfig: ptr::null(),
        },
        dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
        memType: Qnn_MemType_t_QNN_MEM_TYPE_DMA_BUF,
        __bindgen_anon_1: Qnn_MemDescriptor_t__bindgen_ty_1 {
            dmaBufInfo: Qnn_MemDmaBufInfo_t {
                fd,
                data: host_data,
            },
        },
    };
    let descs = [
        mk_desc(fd_x, rpc_x),
        mk_desc(fd_y, rpc_y),
        mk_desc(fd_z, rpc_z),
        mk_desc(fd_out, rpc_out),
    ];
    let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 4];
    let err = unsafe { (v.memRegister.unwrap())(ctx, descs.as_ptr(), 4, mh.as_mut_ptr()) };
    anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);

    let set_mh = |t: &mut Qnn_Tensor_t, h: *mut std::ffi::c_void| {
        t.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        t.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = h;
    };
    let mut t_x_mh = t_x;
    set_mh(&mut t_x_mh, mh[0]);
    let mut t_y_mh = t_y;
    set_mh(&mut t_y_mh, mh[1]);
    let mut t_z_mh = t_z;
    set_mh(&mut t_z_mh, mh[2]);
    let mut t_out_mh = t_out;
    set_mh(&mut t_out_mh, mh[3]);

    let exec_inputs = [t_x_mh, t_y_mh, t_z_mh];
    let mut exec_outputs = [t_out_mh];

    let err = unsafe {
        (v.graphExecute.unwrap())(
            graph,
            exec_inputs.as_ptr(),
            exec_inputs.len() as u32,
            exec_outputs.as_mut_ptr(),
            exec_outputs.len() as u32,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);

    // ── Compare against host reference ──────────────────────────────────────
    // Reference: out[i] = x[i] + y[i] + z[i]
    let mut max_abs = 0.0f32;
    unsafe {
        let out_slice = std::slice::from_raw_parts(rpc_out as *const f32, n);
        for i in 0..n {
            let expected = host_x[i] + host_y[i] + host_z[i];
            let d = (out_slice[i] - expected).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
    }

    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), 4);
    }

    Ok(max_abs)
}
