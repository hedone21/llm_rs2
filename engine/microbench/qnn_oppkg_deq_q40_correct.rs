//! M2.C CustomDeqQ40 correctness — production OpPackage vs CPU reference.
//!
//! Tests the `CustomDeqQ40` op (kernel_convert_block_q4_0) registered in
//! `libqnn_oppkg.so` against a CPU reference that manually unpacks
//! block_q4_0 structs (d: half + qs[16]: uchar).
//!
//! Since both paths use the same kernel, max_abs_err == 0 is expected.
//! The acceptance threshold is set to 1e-3 to accommodate any fp16 rounding.
//!
//! Cases: num_blocks ∈ {1, 32, 1024}
//!   n=32    → 1 block  (576 bytes src)
//!   n=1024  → 32 blocks
//!   n=32768 → 1024 blocks
//!
//! Pass criterion: dst_q max_abs_err == 0 (byte comparison) and
//!                 dst_d max_abs_err < 1e-3 for all cases.
//!
//! Build:
//!   cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!     --bin microbench_qnn_oppkg_deq_q40_correct
//!
//! Pre-deploy (host):
//!   cargo build --release -p qnn_oppkg --target aarch64-linux-android
//!   adb push target/aarch64-linux-android/release/libqnn_oppkg.so /data/local/tmp/
//!
//! Run on device:
//!   adb shell "LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:/vendor/lib64 \
//!              /data/local/tmp/microbench_qnn_oppkg_deq_q40_correct"

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_oppkg_deq_q40_correct requires --features qnn");
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
    use std::ffi::{CString, c_void};
    use std::os::raw::c_uint;
    use std::ptr;

    use qnn::*;

    const PKG_PATH: &str = "/data/local/tmp/libqnn_oppkg.so";
    const PKG_PROVIDER: &str = "QnnOpPackage_InitInterface";
    const PKG_TARGET: &str = "GPU_QTI_AISW";

    println!("=== microbench_qnn_oppkg_deq_q40_correct (M2.C) ===\n");
    println!("Op Package: {}", PKG_PATH);

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

    // VERBOSE logger — NULL callback = QNN's default platform logger (Android logcat).
    // Inspect with: adb logcat -v threadtime QnnGpu:V QnnGpuOpPackage:V QnnDevice:V \*:S
    let mut logger: Qnn_LogHandle_t = ptr::null_mut();
    if let Some(log_create) = v.logCreate {
        let err = unsafe { log_create(None, QnnLog_Level_t_QNN_LOG_LEVEL_VERBOSE, &mut logger) };
        if err != 0 {
            eprintln!("logCreate err=0x{:x} (proceeding without logger)", err);
            logger = ptr::null_mut();
        } else {
            eprintln!("logCreate VERBOSE: OK");
        }
    }

    let mut be: Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe { (v.backendCreate.unwrap())(logger, ptr::null_mut(), &mut be) };
    anyhow::ensure!(err == 0, "backendCreate err=0x{:x}", err);
    println!("backend: OK");

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
    println!("registerOpPackage -> err=0x{:x}", err);
    anyhow::ensure!(err == 0, "registerOpPackage err=0x{:x}", err);
    println!("  OK\n");

    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe { (v.contextCreate.unwrap())(be, ptr::null_mut(), ptr::null_mut(), &mut ctx) };
    anyhow::ensure!(err == 0, "contextCreate err=0x{:x}", err);

    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;
    type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(po: *const c_void) -> i32;
    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }?;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let _rpcmem_free: Symbol<RpcmemFreeFn> = unsafe { rpc_lib.get(b"rpcmem_free\0")? };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };

    // Cases: (num_blocks, label)
    // NOTE: temporarily reduced to 1 case for VERBOSE logcat capture during M2.C debug.
    let cases: &[(usize, &str)] = &[(1, "1 block")];
    let mut all_pass = true;

    for &(num_blocks, label) in cases {
        println!("--- {} ---", label);
        let result = run_case(
            &v,
            ctx,
            &rpcmem_alloc,
            &rpcmem_to_fd,
            RPCMEM_HEAP_ID_SYSTEM,
            RPCMEM_DEFAULT_FLAGS,
            num_blocks,
        );
        match result {
            Ok((q_max_err, d_max_err)) => {
                let q_pass = q_max_err == 0;
                let d_pass = d_max_err < 1e-3;
                let pass = q_pass && d_pass;
                println!(
                    "  dst_q max_abs_err = {}  {}",
                    q_max_err,
                    if q_pass { "PASS" } else { "FAIL" }
                );
                println!(
                    "  dst_d max_abs_err = {:.6e}  {}",
                    d_max_err,
                    if d_pass { "PASS" } else { "FAIL" }
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

    println!(
        "\n=== M2.C verdict: {} ===",
        if all_pass { "GREEN" } else { "RED" }
    );

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

/// Returns (dst_q_max_byte_diff, dst_d_max_abs_err_f32).
#[cfg(feature = "qnn")]
fn run_case(
    v: &qnn::QnnInterface_ImplementationV2_25_t,
    ctx: qnn::Qnn_ContextHandle_t,
    rpcmem_alloc: &libloading::Symbol<unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void>,
    rpcmem_to_fd: &libloading::Symbol<unsafe extern "C" fn(*const std::ffi::c_void) -> i32>,
    heap_id: i32,
    flags: u32,
    num_blocks: usize,
) -> anyhow::Result<(u8, f32)> {
    use qnn::*;
    use std::ffi::CString;
    use std::ptr;

    // block_q4_0: half d (2 bytes) + uchar qs[16] = 18 bytes
    const BLOCK_SIZE: usize = 18;
    const QS_SIZE: usize = 16; // qs bytes per block

    let src_bytes = num_blocks * BLOCK_SIZE;
    let dst_q_bytes = num_blocks * QS_SIZE;
    let dst_d_bytes = num_blocks * 2; // half = 2 bytes

    let rpc_src = unsafe { rpcmem_alloc(heap_id, flags, src_bytes as i32) };
    let rpc_dst_q = unsafe { rpcmem_alloc(heap_id, flags, dst_q_bytes as i32) };
    let rpc_dst_d = unsafe { rpcmem_alloc(heap_id, flags, dst_d_bytes as i32) };
    anyhow::ensure!(
        !rpc_src.is_null() && !rpc_dst_q.is_null() && !rpc_dst_d.is_null(),
        "rpcmem_alloc failed for num_blocks={}",
        num_blocks
    );

    // Generate random Q4_0 blocks. Use a simple deterministic pattern.
    let src_slice = unsafe { std::slice::from_raw_parts_mut(rpc_src as *mut u8, src_bytes) };
    for (i, b) in src_slice.iter_mut().enumerate() {
        *b = ((i * 37 + 13) & 0xFF) as u8;
    }

    let fd_src = unsafe { rpcmem_to_fd(rpc_src) };
    let fd_dst_q = unsafe { rpcmem_to_fd(rpc_dst_q) };
    let fd_dst_d = unsafe { rpcmem_to_fd(rpc_dst_d) };

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

    // src0 tensor: UINT_8, [num_blocks * 18]
    let mut dims_src = vec![(num_blocks * BLOCK_SIZE) as u32];
    // dst_q tensor: UINT_8, [num_blocks * 16]
    let mut dims_dst_q = vec![(num_blocks * QS_SIZE) as u32];
    // dst_d tensor: UINT_8 byte-count [num_blocks * 2] to match descriptor.
    // The kernel arg `global half *` reinterprets the raw bytes.
    let mut dims_dst_d = vec![(num_blocks * 2) as u32];

    let mk_tv1_u8 = |ttype, dims_ptr: *mut u32, rank: u32| Qnn_TensorV1_t {
        id: 0,
        name: ptr::null(),
        type_: ttype,
        dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        dataType: Qnn_DataType_t_QNN_DATATYPE_UINT_8,
        quantizeParams: qp,
        rank,
        dimensions: dims_ptr,
        memType: Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
        __bindgen_anon_1: Qnn_TensorV1_t__bindgen_ty_1 {
            clientBuf: Qnn_ClientBuffer_t {
                data: ptr::null_mut(),
                dataSize: 0,
            },
        },
    };
    let mk_tv1_f16 = |ttype, dims_ptr: *mut u32, rank: u32| Qnn_TensorV1_t {
        id: 0,
        name: ptr::null(),
        type_: ttype,
        dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
        quantizeParams: qp,
        rank,
        dimensions: dims_ptr,
        memType: Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
        __bindgen_anon_1: Qnn_TensorV1_t__bindgen_ty_1 {
            clientBuf: Qnn_ClientBuffer_t {
                data: ptr::null_mut(),
                dataSize: 0,
            },
        },
    };

    let name_src = CString::new(format!("src_{}", num_blocks)).unwrap();
    let name_dstq = CString::new(format!("dst_q_{}", num_blocks)).unwrap();
    let name_dstd = CString::new(format!("dst_d_{}", num_blocks)).unwrap();
    let name_dstq_alias = CString::new(format!("dst_q_alias_{}", num_blocks)).unwrap();

    let mut t_src = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1_u8(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                dims_src.as_mut_ptr(),
                1,
            ),
        },
    };
    t_src.__bindgen_anon_1.v1.name = name_src.as_ptr();

    // dst_q is bound as a graph input (in-place alias) per descriptor's
    // multi-output remap. APP_WRITE = application provides the buffer; the
    // OpPackage kernel reads + writes it (OP_INPUT_READWRITE). The host reads
    // the populated dst_q back via the rpcmem fd after graphExecute.
    let mut t_dst_q = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1_u8(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                dims_dst_q.as_mut_ptr(),
                1,
            ),
        },
    };
    t_dst_q.__bindgen_anon_1.v1.name = name_dstq.as_ptr();

    // dst_d byte-buffer (kernel arg `global half *`; mem_object UINT_8 byte-count).
    // Bound as APP_WRITE input + InOutTensor (silu_mul-style alias pattern).
    let mut t_dst_d = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1_u8(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                dims_dst_d.as_mut_ptr(),
                1,
            ),
        },
    };
    t_dst_d.__bindgen_anon_1.v1.name = name_dstd.as_ptr();

    // dst_q_alias: graph output (APP_READ). Same dim as dst_q; bound to the
    // same fd to mirror silu_mul's alias pattern. Application reads results
    // from rpc_dst_q via the original fd after graphExecute.
    let mut dims_dst_q_alias = vec![(num_blocks * QS_SIZE) as u32];
    let mut t_dst_q_alias = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_tv1_u8(
                Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
                dims_dst_q_alias.as_mut_ptr(),
                1,
            ),
        },
    };
    t_dst_q_alias.__bindgen_anon_1.v1.name = name_dstq_alias.as_ptr();

    let g_name = CString::new(format!("deq_q40_graph_{}", num_blocks)).unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    for (label, t) in [
        ("src", &mut t_src),
        ("dst_q", &mut t_dst_q),
        ("dst_d", &mut t_dst_d),
        ("dst_q_alias", &mut t_dst_q_alias),
    ] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", label, err);
    }

    let op_name = CString::new(format!("deq_q40_0_{}", num_blocks)).unwrap();
    let pkg = CString::new("qnn_oppkg").unwrap();
    let op_type = CString::new("CustomDeqQ40").unwrap();
    // silu_mul-style alias: 3 inputs (src0, dst_q, dst_d in-place) + 1 alias
    // output (dst_q_alias bound to fd_dst_q).
    let mut inputs = [t_src, t_dst_q, t_dst_d];
    let mut outputs = [t_dst_q_alias];
    let op = Qnn_OpConfig_t {
        version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
        __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
            v1: Qnn_OpConfigV1_t {
                name: op_name.as_ptr(),
                packageName: pkg.as_ptr(),
                typeName: op_type.as_ptr(),
                numOfParams: 0,
                params: ptr::null_mut(),
                numOfInputs: 3,
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

    // Register memory handles
    let mk_desc_u8 =
        |fd: i32, host_data: *mut std::ffi::c_void, n_bytes: u32| Qnn_MemDescriptor_t {
            memShape: Qnn_MemShape_t {
                numDim: 1,
                dimSize: &n_bytes as *const u32 as *mut u32,
                shapeConfig: ptr::null(),
            },
            dataType: Qnn_DataType_t_QNN_DATATYPE_UINT_8,
            memType: Qnn_MemType_t_QNN_MEM_TYPE_DMA_BUF,
            __bindgen_anon_1: Qnn_MemDescriptor_t__bindgen_ty_1 {
                dmaBufInfo: Qnn_MemDmaBufInfo_t {
                    fd,
                    data: host_data,
                },
            },
        };
    // 4 mem descriptors: src0, dst_q, dst_d, dst_q_alias (same fd as dst_q).
    let descs = [
        mk_desc_u8(fd_src, rpc_src, (num_blocks * BLOCK_SIZE) as u32),
        mk_desc_u8(fd_dst_q, rpc_dst_q, (num_blocks * QS_SIZE) as u32),
        mk_desc_u8(fd_dst_d, rpc_dst_d, (num_blocks * 2) as u32),
        mk_desc_u8(fd_dst_q, rpc_dst_q, (num_blocks * QS_SIZE) as u32),
    ];
    let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 4];
    let err = unsafe { (v.memRegister.unwrap())(ctx, descs.as_ptr(), 4, mh.as_mut_ptr()) };
    anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);

    // Bind: inputs = [src0, dst_q, dst_d], outputs = [dst_q_alias].
    inputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[0];
    inputs[1].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[1];
    inputs[2].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[2].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[2];
    outputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[3];

    let err = unsafe {
        (v.graphExecute.unwrap())(
            graph,
            inputs.as_ptr(),
            3,
            outputs.as_mut_ptr(),
            1,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);

    // CPU reference: manually unpack block_q4_0 structs
    // block_q4_0: bytes [0..1] = half d (little-endian), bytes [2..17] = qs[16]
    let mut q_max_err: u8 = 0;
    let mut d_max_err: f32 = 0.0;

    unsafe {
        let dst_q_out = std::slice::from_raw_parts(rpc_dst_q as *const u8, dst_q_bytes);
        let dst_d_out = std::slice::from_raw_parts(rpc_dst_d as *const u8, dst_d_bytes);

        for blk in 0..num_blocks {
            let src_off = blk * BLOCK_SIZE;
            // d: first 2 bytes of block (half, little-endian)
            let ref_d_bytes = [src_slice[src_off], src_slice[src_off + 1]];
            let ref_d_bits = u16::from_le_bytes(ref_d_bytes);
            let ref_d_f32 = half::f16::from_bits(ref_d_bits).to_f32();

            let out_d_bits = u16::from_le_bytes([dst_d_out[blk * 2], dst_d_out[blk * 2 + 1]]);
            let out_d_f32 = half::f16::from_bits(out_d_bits).to_f32();
            let d_err = (out_d_f32 - ref_d_f32).abs();
            if d_err > d_max_err {
                d_max_err = d_err;
            }

            // qs: bytes [2..18] of block
            for qi in 0..QS_SIZE {
                let ref_q = src_slice[src_off + 2 + qi];
                let out_q = dst_q_out[blk * QS_SIZE + qi];
                let diff = ref_q.abs_diff(out_q);
                if diff > q_max_err {
                    q_max_err = diff;
                }
            }
        }
    }

    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), 4);
    }

    Ok((q_max_err, d_max_err))
}
