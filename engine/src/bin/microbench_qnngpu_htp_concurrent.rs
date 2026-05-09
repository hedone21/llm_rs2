//! microbench_qnngpu_htp_concurrent — Phase R Wave 2 follow-up
//!
//! 목적: HTP MatMul + QNN-GPU MatMul 동시 실행이 H/W parallel 작동하는지 검증.
//! Phase 32b-2 (HTP + raw OpenCL)는 0.510× 확인. 본 측정은 raw OpenCL을 QNN-GPU로 교체.
//!
//! Configs (n=30):
//!   C1: HTP MatMul only
//!   C2: QNN-GPU MatMul only
//!   C3: 동시 launch (다른 thread, AtomicUsize barrier)
//!   C4: sequential (C1 → C2 wall-clock)
//!
//! Pass:
//!   - C3 / max(C1, C2) ≤ 1.30 → 부분 parallel (Phase 32b-2 기준)
//!   - C3 / (C1 + C2) ≤ 0.70 → 본격 H/W parallel (다른 chip)
//!   - > 1.0 → serialize (R-Y opt 3 sequential 결과가 misleading)
//!
//! 차원: 1×1024×4096 (DDR-heavy: 16 MB B). Phase 32b-2와 동일.
//!
//! Build: cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!        --bin microbench_qnngpu_htp_concurrent
//! Run:   `LD_LIBRARY_PATH=/data/local/tmp/qnn:/vendor/lib64 \
//!         ADSP_LIBRARY_PATH=/data/local/tmp/qnn \
//!         adb shell /data/local/tmp/microbench_qnngpu_htp_concurrent [N_ITERS]`

#[cfg(not(all(feature = "qnn", feature = "opencl")))]
fn main() {
    eprintln!("microbench_qnngpu_htp_concurrent requires --features qnn,opencl");
    std::process::exit(2);
}

#[cfg(all(feature = "qnn", feature = "opencl"))]
#[allow(non_snake_case, non_camel_case_types, non_upper_case_globals, dead_code)]
mod qnn {
    include!(concat!(env!("OUT_DIR"), "/qnn_bindings.rs"));
}

#[cfg(all(feature = "qnn", feature = "opencl"))]
fn main() -> anyhow::Result<()> {
    use libloading::{Library, Symbol};
    use std::ffi::{CString, c_void};
    use std::os::raw::c_uint;
    use std::ptr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Instant;

    use qnn::*;

    let args: Vec<String> = std::env::args().collect();
    let n_iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);

    let m: usize = 1;
    let k: usize = 1024;
    let n: usize = 4096;

    println!("=== microbench_qnngpu_htp_concurrent (Phase R follow-up) ===\n");
    println!("Dim: M={}, K={}, N={}  B size: {} MB", m, k, n, k * n * 4 / 1024 / 1024);
    println!("HTP: prebuilt MatMul (FP32, clientBuf)");
    println!("QNN-GPU: prebuilt MatMul (FP32, clientBuf)");
    println!("n_iters per config: {}\n", n_iters);

    type GetProvidersFn =
        unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;

    // ─────────────────────────────────────────────────────────
    // Common: build a MatMul graph (FP32 clientBuf) on a backend
    // Returns (graph, inputs, outputs, exec closure that returns wall-clock ms)
    // ─────────────────────────────────────────────────────────

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
                    scaleOffsetEncoding: Qnn_ScaleOffset_t { scale: 0.0, offset: 0 },
                },
            },
            rank: dims.len() as u32,
            dimensions: dims.as_ptr() as *mut u32,
            memType: Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
            __bindgen_anon_1: Qnn_TensorV1_t__bindgen_ty_1 {
                clientBuf: Qnn_ClientBuffer_t { data: ptr::null_mut(), dataSize: 0 },
            },
        }
    };

    let dims_a: Vec<u32> = vec![m as u32, k as u32];
    let dims_b: Vec<u32> = vec![k as u32, n as u32];
    let dims_c: Vec<u32> = vec![m as u32, n as u32];

    // ─────────────────────────────────────────────────────────
    // HTP backend
    // ─────────────────────────────────────────────────────────
    let htp_lib = unsafe { Library::new("/data/local/tmp/qnn/libQnnHtp.so") }?;
    let htp_gp: Symbol<GetProvidersFn> = unsafe { htp_lib.get(b"QnnInterface_getProviders\0")? };
    let mut htp_provs: *mut *const QnnInterface_t = ptr::null_mut();
    let mut htp_n_p: c_uint = 0;
    let err = unsafe { htp_gp(&mut htp_provs, &mut htp_n_p) };
    anyhow::ensure!(err == 0 && htp_n_p > 0, "HTP getProviders err=0x{:x}", err);
    let v_htp = unsafe { (**htp_provs).__bindgen_anon_1.v2_25 };

    let mut htp_be: Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe {
        (v_htp.backendCreate.unwrap())(ptr::null_mut(), ptr::null_mut(), &mut htp_be)
    };
    anyhow::ensure!(err == 0, "HTP backendCreate err=0x{:x}", err);
    let mut htp_ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe {
        (v_htp.contextCreate.unwrap())(htp_be, ptr::null_mut(), ptr::null_mut(), &mut htp_ctx)
    };
    anyhow::ensure!(err == 0, "HTP contextCreate err=0x{:x}", err);

    let htp_g_name = CString::new("htp_matmul").unwrap();
    let mut htp_graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err = unsafe {
        (v_htp.graphCreate.unwrap())(
            htp_ctx,
            htp_g_name.as_ptr(),
            ptr::null_mut(),
            &mut htp_graph,
        )
    };
    anyhow::ensure!(err == 0, "HTP graphCreate err=0x{:x}", err);

    let htp_n_a = CString::new("htp_A").unwrap();
    let htp_n_b = CString::new("htp_B").unwrap();
    let htp_n_c = CString::new("htp_C").unwrap();
    let mut htp_t_a = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&htp_n_a, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims_a),
        },
    };
    let mut htp_t_b = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&htp_n_b, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims_b),
        },
    };
    let mut htp_t_c = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&htp_n_c, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, &dims_c),
        },
    };
    for (l, t) in [("A", &mut htp_t_a), ("B", &mut htp_t_b), ("C", &mut htp_t_c)] {
        let err = unsafe { (v_htp.tensorCreateGraphTensor.unwrap())(htp_graph, t) };
        anyhow::ensure!(err == 0, "HTP tensorCreate({}) err=0x{:x}", l, err);
    }
    let htp_op_name = CString::new("htp_mm").unwrap();
    let pkg = CString::new("qti.aisw").unwrap();
    let op_type = CString::new("MatMul").unwrap();
    let mut htp_inputs = [htp_t_a, htp_t_b];
    let mut htp_outputs = [htp_t_c];
    let htp_op = Qnn_OpConfig_t {
        version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
        __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
            v1: Qnn_OpConfigV1_t {
                name: htp_op_name.as_ptr(),
                packageName: pkg.as_ptr(),
                typeName: op_type.as_ptr(),
                numOfParams: 0,
                params: ptr::null_mut(),
                numOfInputs: 2,
                inputTensors: htp_inputs.as_mut_ptr(),
                numOfOutputs: 1,
                outputTensors: htp_outputs.as_mut_ptr(),
            },
        },
    };
    let err = unsafe { (v_htp.graphAddNode.unwrap())(htp_graph, htp_op) };
    anyhow::ensure!(err == 0, "HTP graphAddNode err=0x{:x}", err);
    let err = unsafe {
        (v_htp.graphFinalize.unwrap())(htp_graph, ptr::null_mut(), ptr::null_mut())
    };
    anyhow::ensure!(err == 0, "HTP graphFinalize err=0x{:x}", err);
    println!("HTP graph (MatMul {}x{}x{}) finalize: OK", m, k, n);

    let host_a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001) % 1.0 - 0.5).collect();
    let host_b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0007 + 0.13) % 1.0 - 0.5).collect();
    let mut host_c_htp: Vec<f32> = vec![0.0; m * n];
    let host_c_gpu: Vec<f32> = vec![0.0; m * n];

    htp_inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
        data: host_a.as_ptr() as *mut _,
        dataSize: (host_a.len() * 4) as u32,
    };
    htp_inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
        data: host_b.as_ptr() as *mut _,
        dataSize: (host_b.len() * 4) as u32,
    };
    htp_outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.clientBuf = Qnn_ClientBuffer_t {
        data: host_c_htp.as_mut_ptr() as *mut _,
        dataSize: (host_c_htp.len() * 4) as u32,
    };

    // ─────────────────────────────────────────────────────────
    // QNN-GPU backend
    // ─────────────────────────────────────────────────────────
    let gpu_lib = unsafe { Library::new("/data/local/tmp/qnn/libQnnGpu.so") }?;
    let gpu_gp: Symbol<GetProvidersFn> = unsafe { gpu_lib.get(b"QnnInterface_getProviders\0")? };
    let mut gpu_provs: *mut *const QnnInterface_t = ptr::null_mut();
    let mut gpu_n_p: c_uint = 0;
    let err = unsafe { gpu_gp(&mut gpu_provs, &mut gpu_n_p) };
    anyhow::ensure!(err == 0 && gpu_n_p > 0, "GPU getProviders err=0x{:x}", err);
    let v_gpu = unsafe { (**gpu_provs).__bindgen_anon_1.v2_25 };

    let mut gpu_be: Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe {
        (v_gpu.backendCreate.unwrap())(ptr::null_mut(), ptr::null_mut(), &mut gpu_be)
    };
    anyhow::ensure!(err == 0, "GPU backendCreate err=0x{:x}", err);
    let mut gpu_ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe {
        (v_gpu.contextCreate.unwrap())(gpu_be, ptr::null_mut(), ptr::null_mut(), &mut gpu_ctx)
    };
    anyhow::ensure!(err == 0, "GPU contextCreate err=0x{:x}", err);

    let gpu_g_name = CString::new("gpu_matmul").unwrap();
    let mut gpu_graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err = unsafe {
        (v_gpu.graphCreate.unwrap())(
            gpu_ctx,
            gpu_g_name.as_ptr(),
            ptr::null_mut(),
            &mut gpu_graph,
        )
    };
    anyhow::ensure!(err == 0, "GPU graphCreate err=0x{:x}", err);

    let gpu_n_a = CString::new("gpu_A").unwrap();
    let gpu_n_b = CString::new("gpu_B").unwrap();
    let gpu_n_c = CString::new("gpu_C").unwrap();
    let mut gpu_t_a = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&gpu_n_a, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims_a),
        },
    };
    let mut gpu_t_b = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&gpu_n_b, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims_b),
        },
    };
    let mut gpu_t_c = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&gpu_n_c, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, &dims_c),
        },
    };
    for (l, t) in [("A", &mut gpu_t_a), ("B", &mut gpu_t_b), ("C", &mut gpu_t_c)] {
        let err = unsafe { (v_gpu.tensorCreateGraphTensor.unwrap())(gpu_graph, t) };
        anyhow::ensure!(err == 0, "GPU tensorCreate({}) err=0x{:x}", l, err);
    }
    let gpu_op_name = CString::new("gpu_mm").unwrap();
    let mut gpu_inputs = [gpu_t_a, gpu_t_b];
    let mut gpu_outputs = [gpu_t_c];
    let gpu_op = Qnn_OpConfig_t {
        version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
        __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
            v1: Qnn_OpConfigV1_t {
                name: gpu_op_name.as_ptr(),
                packageName: pkg.as_ptr(),
                typeName: op_type.as_ptr(),
                numOfParams: 0,
                params: ptr::null_mut(),
                numOfInputs: 2,
                inputTensors: gpu_inputs.as_mut_ptr(),
                numOfOutputs: 1,
                outputTensors: gpu_outputs.as_mut_ptr(),
            },
        },
    };
    let err = unsafe { (v_gpu.graphAddNode.unwrap())(gpu_graph, gpu_op) };
    anyhow::ensure!(err == 0, "GPU graphAddNode err=0x{:x}", err);
    let err = unsafe {
        (v_gpu.graphFinalize.unwrap())(gpu_graph, ptr::null_mut(), ptr::null_mut())
    };
    anyhow::ensure!(err == 0, "GPU graphFinalize err=0x{:x}", err);
    println!("QNN-GPU graph (MatMul {}x{}x{}) finalize: OK", m, k, n);

    // QNN-GPU requires MEMHANDLE (DMA_BUF). clientBuf is HTP-only.
    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;
    type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(po: *const c_void) -> i32;
    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }
        .or_else(|_| unsafe { Library::new("libcdsprpc.so") })?;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let _rpcmem_free: Symbol<RpcmemFreeFn> = unsafe { rpc_lib.get(b"rpcmem_free\0")? };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };

    let bytes_a = (m * k * 4) as i32;
    let bytes_b = (k * n * 4) as i32;
    let bytes_c = (m * n * 4) as i32;
    let rpc_a = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes_a) };
    let rpc_b = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes_b) };
    let rpc_c = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes_c) };
    anyhow::ensure!(
        !rpc_a.is_null() && !rpc_b.is_null() && !rpc_c.is_null(),
        "rpcmem_alloc failed"
    );
    let fd_a = unsafe { rpcmem_to_fd(rpc_a) };
    let fd_b = unsafe { rpcmem_to_fd(rpc_b) };
    let fd_c = unsafe { rpcmem_to_fd(rpc_c) };

    // Copy host_a / host_b into rpcmem (host_c read back at end if needed)
    unsafe {
        std::ptr::copy_nonoverlapping(
            host_a.as_ptr() as *const u8,
            rpc_a as *mut u8,
            bytes_a as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_b.as_ptr() as *const u8,
            rpc_b as *mut u8,
            bytes_b as usize,
        );
    }

    let mk_descriptor_dmabuf =
        |fd: i32, host_data: *mut c_void, dims: &[u32]| -> Qnn_MemDescriptor_t {
            Qnn_MemDescriptor_t {
                memShape: Qnn_MemShape_t {
                    numDim: dims.len() as u32,
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
    let descs = [
        mk_descriptor_dmabuf(fd_a, rpc_a, &dims_a),
        mk_descriptor_dmabuf(fd_b, rpc_b, &dims_b),
        mk_descriptor_dmabuf(fd_c, rpc_c, &dims_c),
    ];
    let mut mh = [ptr::null_mut::<c_void>(); 3];
    let err = unsafe {
        (v_gpu.memRegister.unwrap())(gpu_ctx, descs.as_ptr(), 3, mh.as_mut_ptr())
    };
    anyhow::ensure!(err == 0, "GPU memRegister err=0x{:x}", err);
    println!("QNN-GPU memRegister(A,B,C) DMA_BUF: OK\n");

    // Switch GPU tensors to MEMHANDLE
    gpu_inputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    gpu_inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[0];
    gpu_inputs[1].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    gpu_inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[1];
    gpu_outputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    gpu_outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[2];
    let _ = host_c_gpu; // unused: GPU output goes to rpc_c
    let _ = fd_a; let _ = fd_b; let _ = fd_c;

    // ─────────────────────────────────────────────────────────
    // exec helpers (raw pointers as usize for thread safety)
    // ─────────────────────────────────────────────────────────
    let v_htp_ptr = &v_htp as *const _ as usize;
    let v_gpu_ptr = &v_gpu as *const _ as usize;
    let htp_graph_ptr = htp_graph as usize;
    let gpu_graph_ptr = gpu_graph as usize;
    let htp_in_ptr = htp_inputs.as_ptr() as usize;
    let htp_out_ptr = htp_outputs.as_mut_ptr() as usize;
    let gpu_in_ptr = gpu_inputs.as_ptr() as usize;
    let gpu_out_ptr = gpu_outputs.as_mut_ptr() as usize;

    let exec_htp = move || -> anyhow::Result<f64> {
        let v = unsafe { &*(v_htp_ptr as *const QnnInterface_ImplementationV2_25_t) };
        let t0 = Instant::now();
        let err = unsafe {
            (v.graphExecute.unwrap())(
                htp_graph_ptr as Qnn_GraphHandle_t,
                htp_in_ptr as *const Qnn_Tensor_t,
                2,
                htp_out_ptr as *mut Qnn_Tensor_t,
                1,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        anyhow::ensure!(err == 0, "HTP graphExecute err=0x{:x}", err);
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };
    let exec_gpu = move || -> anyhow::Result<f64> {
        let v = unsafe { &*(v_gpu_ptr as *const QnnInterface_ImplementationV2_25_t) };
        let t0 = Instant::now();
        let err = unsafe {
            (v.graphExecute.unwrap())(
                gpu_graph_ptr as Qnn_GraphHandle_t,
                gpu_in_ptr as *const Qnn_Tensor_t,
                2,
                gpu_out_ptr as *mut Qnn_Tensor_t,
                1,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        anyhow::ensure!(err == 0, "GPU graphExecute err=0x{:x}", err);
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    };

    // Warmup
    println!("Warmup (5 iters each)...");
    for _ in 0..5 {
        let _ = exec_htp()?;
        let _ = exec_gpu()?;
    }
    println!();

    // ─────────────────────────────────────────────────────────
    // Configs
    // ─────────────────────────────────────────────────────────
    enum Cfg {
        HtpOnly,
        GpuOnly,
        Concurrent,
        Sequential,
    }
    let configs: &[(&str, Cfg)] = &[
        ("C1: HTP only", Cfg::HtpOnly),
        ("C2: QNN-GPU only", Cfg::GpuOnly),
        ("C3: HTP + QNN-GPU concurrent", Cfg::Concurrent),
        ("C4: HTP → QNN-GPU sequential (sanity)", Cfg::Sequential),
    ];

    let mut summary: Vec<(String, f64, f64, f64)> = Vec::new();

    for (label, cfg) in configs {
        let mut samples = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let ms = match cfg {
                Cfg::HtpOnly => exec_htp()?,
                Cfg::GpuOnly => exec_gpu()?,
                Cfg::Concurrent => {
                    let v_gpu_t = v_gpu_ptr;
                    let gpu_g = gpu_graph_ptr;
                    let gpu_i = gpu_in_ptr;
                    let gpu_o = gpu_out_ptr;
                    let counter = Arc::new(AtomicUsize::new(0));
                    let counter2 = counter.clone();

                    let gpu_handle = thread::spawn(move || -> anyhow::Result<f64> {
                        let v = unsafe {
                            &*(v_gpu_t as *const QnnInterface_ImplementationV2_25_t)
                        };
                        counter2.fetch_add(1, Ordering::SeqCst);
                        while counter2.load(Ordering::SeqCst) < 2 {
                            std::hint::spin_loop();
                        }
                        let t0 = Instant::now();
                        let err = unsafe {
                            (v.graphExecute.unwrap())(
                                gpu_g as Qnn_GraphHandle_t,
                                gpu_i as *const Qnn_Tensor_t,
                                2,
                                gpu_o as *mut Qnn_Tensor_t,
                                1,
                                ptr::null_mut(),
                                ptr::null_mut(),
                            )
                        };
                        anyhow::ensure!(err == 0, "GPU exec err=0x{:x}", err);
                        Ok(t0.elapsed().as_secs_f64() * 1000.0)
                    });
                    while counter.load(Ordering::SeqCst) < 1 {
                        std::hint::spin_loop();
                    }
                    let t0 = Instant::now();
                    counter.fetch_add(1, Ordering::SeqCst);
                    let _htp_ms = exec_htp()?;
                    let _gpu_ms = gpu_handle.join().unwrap()?;
                    t0.elapsed().as_secs_f64() * 1000.0
                }
                Cfg::Sequential => {
                    let t0 = Instant::now();
                    let _ = exec_htp()?;
                    let _ = exec_gpu()?;
                    t0.elapsed().as_secs_f64() * 1000.0
                }
            };
            samples.push(ms);
        }
        let n_s = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n_s;
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_s;
        let stddev = var.sqrt();
        let cv = stddev / mean;
        println!(
            "{:<42} mean={:.3} median={:.3} σ={:.3} σ/mean={:.3}",
            label, mean, median, stddev, cv
        );
        summary.push((label.to_string(), mean, median, cv));
    }

    let c1 = summary[0].1;
    let c2 = summary[1].1;
    let c3 = summary[2].1;
    let c4 = summary[3].1;
    let max_c1c2 = c1.max(c2);
    let serial_sum = c1 + c2;

    println!("\n=== summary ===");
    for (label, mean, _, _) in &summary {
        println!("{:<42} {:>8.3}ms", label, mean);
    }
    println!();
    println!("max(C1, C2)        = {:.3} ms", max_c1c2);
    println!("C1 + C2 (serial)   = {:.3} ms", serial_sum);
    println!("C3 (concurrent)    = {:.3} ms", c3);
    println!("C4 (sequential)    = {:.3} ms (sanity ≈ serial_sum)", c4);
    println!();
    println!(
        "C3 / max(C1,C2) = {:.3}x  (≤1.30 부분 parallel, ≤1.00 perfect)",
        c3 / max_c1c2
    );
    println!(
        "C3 / (C1+C2)    = {:.3}x  (≤0.70 H/W parallel, 1.00 serialize)",
        c3 / serial_sum
    );

    let parallel_ok = c3 / serial_sum <= 0.70;
    let partial_ok = c3 / max_c1c2 <= 1.30;
    println!(
        "\nVerdict: {}",
        if parallel_ok {
            "✓ H/W parallel 확정 (HTP + QNN-GPU 동시 실행)"
        } else if partial_ok {
            "△ 부분 parallel"
        } else {
            "✗ Serialize 또는 contention 심각"
        }
    );

    unsafe {
        let _ = (v_htp.contextFree.unwrap())(htp_ctx, ptr::null_mut());
        let _ = (v_gpu.contextFree.unwrap())(gpu_ctx, ptr::null_mut());
        let _ = (v_htp.backendFree.unwrap())(htp_be);
        let _ = (v_gpu.backendFree.unwrap())(gpu_be);
    }
    Ok(())
}
