//! M1.8 INV-155 leak test — device microbench.
//!
//! 100회 register CustomAdd / graphFinalize / contextFree 반복하며 매 iter 후
//! `/proc/self/status::VmRSS`를 측정한다. last 50 iter VmRSS slope (linear
//! regression) < 1 KB/iter면 PASS.
//!
//! PoC의 leak 패턴 그대로면 매 iter당 OpImplState Box 누수로 slope가 수십
//! KB/iter로 나오고, M1.8 정상화 (`pkg_free_op_impl` reverse-mapping table)가
//! 적용되면 slope ≈ 0 (OS noise level)이 된다.
//!
//! Build:
//!   cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!     --bin microbench_qnn_oppkg_leak
//!
//! Pre-deploy (host):
//!   cargo build --release -p qnn_oppkg --target aarch64-linux-android
//!   adb push target/aarch64-linux-android/release/libqnn_oppkg.so /data/local/tmp/
//!
//! Run on device:
//!   adb shell "LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:/vendor/lib64 \
//!              /data/local/tmp/microbench_qnn_oppkg_leak"

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_oppkg_leak requires --features qnn");
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
fn read_vmrss_kb() -> u64 {
    let s = match std::fs::read_to_string("/proc/self/status") {
        Ok(s) => s,
        Err(_) => return 0,
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            if let Some(tok) = rest.trim().split_whitespace().next() {
                return tok.parse().unwrap_or(0);
            }
        }
    }
    0
}

/// Linear regression slope of `samples` (ys) over indices (xs = 0..n) in
/// the same units as `samples` (KB/iter).
#[cfg(feature = "qnn")]
fn slope_kb_per_iter(samples: &[u64]) -> f64 {
    let n = samples.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x = (n - 1.0) / 2.0;
    let mean_y = samples.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &y) in samples.iter().enumerate() {
        let dx = i as f64 - mean_x;
        let dy = y as f64 - mean_y;
        num += dx * dy;
        den += dx * dx;
    }
    if den == 0.0 { 0.0 } else { num / den }
}

#[cfg(feature = "qnn")]
fn main() -> anyhow::Result<()> {
    use libloading::{Library, Symbol};
    use std::ffi::{CString, c_void};
    use std::os::raw::c_uint;
    use std::ptr;

    use qnn::*;

    const PKG_PATH: &str = "/data/local/tmp/libqnn_oppkg.so";
    const PKG_PROVIDER: &str = "QnnOpPackage_InitInterface";
    const PKG_TARGET: &str = "GPU_QTI_AISW";
    // N_ITER override via env (default 100, INV-155 spec).
    let n_iter: usize = std::env::var("LEAK_ITER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    const N_ELEMS: usize = 1024;
    const SLOPE_GATE_KB_PER_ITER: f64 = 1.0;

    println!("=== microbench_qnn_oppkg_leak (M1.8 / INV-155) ===\n");
    println!("Op Package: {}", PKG_PATH);
    println!("Iterations: {}", n_iter);
    println!(
        "Slope gate: < {:.2} KB/iter (last 50%)\n",
        SLOPE_GATE_KB_PER_ITER
    );

    let backend_lib_path = std::env::var("QNN_BACKEND_LIB")
        .unwrap_or_else(|_| "/data/local/tmp/qnn/libQnnGpu.so".to_string());

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

    // registerOpPackage — 1회만 (전체 반복 동안 공유)
    let pkg_path = CString::new(PKG_PATH).unwrap();
    let pkg_provider = CString::new(PKG_PROVIDER).unwrap();
    let pkg_target = CString::new(PKG_TARGET).unwrap();
    let reg_fn = v
        .backendRegisterOpPackage
        .ok_or_else(|| anyhow::anyhow!("backendRegisterOpPackage is NULL"))?;
    let err = unsafe {
        reg_fn(
            be,
            pkg_path.as_ptr(),
            pkg_provider.as_ptr(),
            pkg_target.as_ptr(),
        )
    };
    anyhow::ensure!(err == 0, "registerOpPackage err=0x{:x}", err);
    println!("registerOpPackage: OK\n");

    // Diagnostics: STATE_MAP size hook from libqnn_oppkg.so.
    let oppkg_lib = unsafe { Library::new(PKG_PATH) }?;
    type StateMapLenFn = unsafe extern "C" fn() -> u64;
    let state_map_len: Symbol<StateMapLenFn> =
        unsafe { oppkg_lib.get(b"qnn_oppkg_state_map_len\0")? };

    // rpcmem
    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;
    type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(po: *const c_void) -> i32;
    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }?;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let rpcmem_free: Symbol<RpcmemFreeFn> = unsafe { rpc_lib.get(b"rpcmem_free\0")? };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };

    // Warm up: 5 iter 미측정 (lazy alloc, scratch pool stabilisation).
    for _ in 0..5 {
        run_one_iter(
            &v,
            be,
            &rpcmem_alloc,
            &rpcmem_free,
            &rpcmem_to_fd,
            RPCMEM_HEAP_ID_SYSTEM,
            RPCMEM_DEFAULT_FLAGS,
            N_ELEMS,
        )?;
    }

    let baseline = read_vmrss_kb();
    println!("Baseline VmRSS (post-warmup): {} KB\n", baseline);

    let mut samples: Vec<u64> = Vec::with_capacity(n_iter);
    for i in 0..n_iter {
        run_one_iter(
            &v,
            be,
            &rpcmem_alloc,
            &rpcmem_free,
            &rpcmem_to_fd,
            RPCMEM_HEAP_ID_SYSTEM,
            RPCMEM_DEFAULT_FLAGS,
            N_ELEMS,
        )?;
        let rss = read_vmrss_kb();
        samples.push(rss);
        // Report at 0%, 25%, 50%, 75%, 99% of n_iter.
        let report = i == 0
            || i + 1 == n_iter
            || i == n_iter / 4
            || i == n_iter / 2
            || i == (n_iter * 3) / 4;
        if report {
            let smlen = unsafe { state_map_len() };
            println!(
                "  iter {:>3}: VmRSS = {} KB (Δ={:+} KB)  STATE_MAP={}",
                i,
                rss,
                rss as i64 - baseline as i64,
                smlen,
            );
        }
    }

    // last half slope (≥ 50% of n_iter, min 50 samples).
    let half_start = samples
        .len()
        .saturating_sub(samples.len() / 2)
        .max(samples.len().saturating_sub(50));
    let last_half = &samples[half_start..];
    let slope = slope_kb_per_iter(last_half);
    let total_growth = (samples.last().copied().unwrap_or(0) as i64) - baseline as i64;

    let final_smlen = unsafe { state_map_len() };
    println!("\n--- Summary ---");
    println!(
        "Total VmRSS growth ({} iter): {:+} KB",
        n_iter, total_growth
    );
    println!(
        "Last-half slope ({} samples):  {:.4} KB/iter",
        last_half.len(),
        slope,
    );
    println!(
        "Gate:                          < {:.2} KB/iter",
        SLOPE_GATE_KB_PER_ITER
    );
    println!(
        "Final STATE_MAP entries:       {}  (cdylib leak proxy)",
        final_smlen
    );

    let pass = slope.abs() < SLOPE_GATE_KB_PER_ITER;
    println!(
        "\n=== M1.8 verdict: {} ===",
        if pass { "GREEN" } else { "RED" }
    );

    unsafe {
        let _ = (v.backendFree.unwrap())(be);
    }

    if pass { Ok(()) } else { std::process::exit(1) }
}

#[cfg(feature = "qnn")]
#[allow(clippy::too_many_arguments)]
fn run_one_iter(
    v: &qnn::QnnInterface_ImplementationV2_25_t,
    be: qnn::Qnn_BackendHandle_t,
    rpcmem_alloc: &libloading::Symbol<unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void>,
    rpcmem_free: &libloading::Symbol<unsafe extern "C" fn(*mut std::ffi::c_void)>,
    rpcmem_to_fd: &libloading::Symbol<unsafe extern "C" fn(*const std::ffi::c_void) -> i32>,
    heap_id: i32,
    flags: u32,
    n: usize,
) -> anyhow::Result<()> {
    use qnn::*;
    use std::ffi::CString;
    use std::ptr;

    // Per-iter context (so contextFree fully releases the graph + ops).
    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe { (v.contextCreate.unwrap())(be, ptr::null_mut(), ptr::null_mut(), &mut ctx) };
    anyhow::ensure!(err == 0, "contextCreate err=0x{:x}", err);

    let bytes = (n * 4) as i32;
    let rpc_x = unsafe { rpcmem_alloc(heap_id, flags, bytes) };
    let rpc_y = unsafe { rpcmem_alloc(heap_id, flags, bytes) };
    let rpc_out = unsafe { rpcmem_alloc(heap_id, flags, bytes) };
    anyhow::ensure!(
        !rpc_x.is_null() && !rpc_y.is_null() && !rpc_out.is_null(),
        "rpcmem_alloc failed",
    );

    // Fill x, y with deterministic values.
    let host_x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let host_y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0007).collect();
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
    }
    let fd_x = unsafe { rpcmem_to_fd(rpc_x) };
    let fd_y = unsafe { rpcmem_to_fd(rpc_y) };
    let fd_out = unsafe { rpcmem_to_fd(rpc_out) };

    let mut dims = vec![n as u32];
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
        rank: 1,
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
    let name_x = CString::new("x").unwrap();
    let name_y = CString::new("y").unwrap();
    let name_out = CString::new("out").unwrap();

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
    let mut t_out = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1(Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, dims_ptr),
        },
    };
    t_out.__bindgen_anon_1.v1.name = name_out.as_ptr();

    let g_name = CString::new("leak_g").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    for (label, t) in [("x", &mut t_x), ("y", &mut t_y), ("out", &mut t_out)] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", label, err);
    }

    let op_name = CString::new("add0").unwrap();
    let pkg = CString::new("qnn_oppkg").unwrap();
    let op_type = CString::new("CustomAdd").unwrap();
    let mut inputs = [t_x, t_y];
    let mut outputs = [t_out];
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

    // graphFinalize: createOpImpl 호출 → STATE_MAP에 entry +1
    let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
    anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);

    let mk_desc = |fd: i32, host_data: *mut std::ffi::c_void| Qnn_MemDescriptor_t {
        memShape: Qnn_MemShape_t {
            numDim: 1,
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
        mk_desc(fd_out, rpc_out),
    ];
    let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 3];
    let err = unsafe { (v.memRegister.unwrap())(ctx, descs.as_ptr(), 3, mh.as_mut_ptr()) };
    anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);

    inputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[0];
    inputs[1].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[1];
    outputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[2];

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

    // Teardown — order: memDeRegister → contextFree → rpcmem_free.
    // contextFree destroys graphs and triggers freeOpImpl on registered ops.
    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), 3);
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        rpcmem_free(rpc_x);
        rpcmem_free(rpc_y);
        rpcmem_free(rpc_out);
    }

    Ok(())
}
