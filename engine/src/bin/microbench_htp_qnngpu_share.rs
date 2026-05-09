//! microbench_htp_qnngpu_share — Phase 32b R-Y option 3
//!
//! 목적: 같은 ION fd를 HTP backend + QNN-GPU backend 양쪽에 QnnMem_register하여
//! cross-backend zero-copy 가능한지 검증. HeteroInfer가 사용한 path.
//!
//! Stage:
//!   1. rpcmem alloc (a, b, c) — SYSTEM_HEAP, ION fd
//!   2. HTP backend create + register fds
//!   3. QNN-GPU backend create + register same fds
//!   4. HTP graph: ElementWiseAdd a + b → c (MEMHANDLE)
//!   5. QNN-GPU graph: ElementWiseAdd c + b → d (input from HTP output)
//!   6. host 측 결과 검증 — HTP가 쓴 데이터가 QNN-GPU에 visible
//!
//! Build: cargo build --release --features qnn --target aarch64-linux-android \
//!        --bin microbench_htp_qnngpu_share
//! Run:   `LD_LIBRARY_PATH=/data/local/tmp/qnn:/vendor/lib64 \
//!         ADSP_LIBRARY_PATH=/data/local/tmp/qnn \
//!         adb shell /data/local/tmp/microbench_htp_qnngpu_share`

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_htp_qnngpu_share requires --features qnn");
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

    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;
    type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(po: *const c_void) -> i32;

    let n: usize = 1024;
    let bytes = (n * 4) as i32;

    println!("=== microbench_htp_qnngpu_share (R-Y option 3) ===\n");

    // ── rpcmem ──
    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }
        .or_else(|_| unsafe { Library::new("libcdsprpc.so") })?;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let rpcmem_free: Symbol<RpcmemFreeFn> = unsafe { rpc_lib.get(b"rpcmem_free\0")? };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };

    // Allocate 4 buffers: a, b (HTP inputs), c (HTP→QNN-GPU shared), d (QNN-GPU output)
    let rpc_a = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    let rpc_b = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    let rpc_c = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    let rpc_d = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    anyhow::ensure!(
        !rpc_a.is_null() && !rpc_b.is_null() && !rpc_c.is_null() && !rpc_d.is_null(),
        "rpcmem_alloc failed"
    );
    let fd_a = unsafe { rpcmem_to_fd(rpc_a) };
    let fd_b = unsafe { rpcmem_to_fd(rpc_b) };
    let fd_c = unsafe { rpcmem_to_fd(rpc_c) };
    let fd_d = unsafe { rpcmem_to_fd(rpc_d) };
    println!(
        "rpcmem buffers: a fd={}, b fd={}, c fd={} (SHARED), d fd={}",
        fd_a, fd_b, fd_c, fd_d
    );

    // ── HTP backend ──
    let htp_lib = unsafe { Library::new("/data/local/tmp/qnn/libQnnHtp.so") }?;
    type GetProvidersFn = unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let htp_gp: Symbol<GetProvidersFn> = unsafe { htp_lib.get(b"QnnInterface_getProviders\0")? };
    let mut htp_provs: *mut *const QnnInterface_t = ptr::null_mut();
    let mut htp_n: c_uint = 0;
    let err = unsafe { htp_gp(&mut htp_provs, &mut htp_n) };
    anyhow::ensure!(err == 0 && htp_n > 0, "HTP getProviders err=0x{:x}", err);
    let v_htp = unsafe { (**htp_provs).__bindgen_anon_1.v2_25 };
    let htp_provider_name = unsafe {
        if (**htp_provs).providerName.is_null() {
            "(null)".to_string()
        } else {
            std::ffi::CStr::from_ptr((**htp_provs).providerName)
                .to_string_lossy()
                .into_owned()
        }
    };
    println!("HTP backend: provider={}", htp_provider_name);

    // ── QNN-GPU backend ──
    let gpu_lib = unsafe { Library::new("/data/local/tmp/qnn/libQnnGpu.so") }?;
    let gpu_gp: Symbol<GetProvidersFn> = unsafe { gpu_lib.get(b"QnnInterface_getProviders\0")? };
    let mut gpu_provs: *mut *const QnnInterface_t = ptr::null_mut();
    let mut gpu_n: c_uint = 0;
    let err = unsafe { gpu_gp(&mut gpu_provs, &mut gpu_n) };
    anyhow::ensure!(err == 0 && gpu_n > 0, "GPU getProviders err=0x{:x}", err);
    let v_gpu = unsafe { (**gpu_provs).__bindgen_anon_1.v2_25 };
    let gpu_provider_name = unsafe {
        if (**gpu_provs).providerName.is_null() {
            "(null)".to_string()
        } else {
            std::ffi::CStr::from_ptr((**gpu_provs).providerName)
                .to_string_lossy()
                .into_owned()
        }
    };
    println!("QNN-GPU backend: provider={}", gpu_provider_name);

    // Create HTP backend + context
    let mut htp_be: Qnn_BackendHandle_t = ptr::null_mut();
    let err =
        unsafe { (v_htp.backendCreate.unwrap())(ptr::null_mut(), ptr::null_mut(), &mut htp_be) };
    anyhow::ensure!(err == 0, "HTP backendCreate err=0x{:x}", err);
    let mut htp_ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe {
        (v_htp.contextCreate.unwrap())(htp_be, ptr::null_mut(), ptr::null_mut(), &mut htp_ctx)
    };
    anyhow::ensure!(err == 0, "HTP contextCreate err=0x{:x}", err);

    // Create QNN-GPU backend + context
    let mut gpu_be: Qnn_BackendHandle_t = ptr::null_mut();
    let err =
        unsafe { (v_gpu.backendCreate.unwrap())(ptr::null_mut(), ptr::null_mut(), &mut gpu_be) };
    anyhow::ensure!(err == 0, "GPU backendCreate err=0x{:x}", err);
    let mut gpu_ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe {
        (v_gpu.contextCreate.unwrap())(gpu_be, ptr::null_mut(), ptr::null_mut(), &mut gpu_ctx)
    };
    anyhow::ensure!(err == 0, "GPU contextCreate err=0x{:x}", err);
    println!("Both HTP + QNN-GPU contexts created OK");

    // ── Register fds with both backends ──
    let dims = vec![n as u32];
    let mk_descriptor_ion = |fd: i32| -> Qnn_MemDescriptor_t {
        Qnn_MemDescriptor_t {
            memShape: Qnn_MemShape_t {
                numDim: 1,
                dimSize: dims.as_ptr() as *mut u32,
                shapeConfig: ptr::null(),
            },
            dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            memType: Qnn_MemType_t_QNN_MEM_TYPE_ION,
            __bindgen_anon_1: Qnn_MemDescriptor_t__bindgen_ty_1 {
                ionInfo: Qnn_MemIonInfo_t { fd },
            },
        }
    };
    let mk_descriptor_dmabuf = |fd: i32, host_data: *mut c_void| -> Qnn_MemDescriptor_t {
        Qnn_MemDescriptor_t {
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
        }
    };

    // HTP registers a, b, c (ION 또는 DMA_BUF 시도)
    let htp_descs = [
        mk_descriptor_ion(fd_a),
        mk_descriptor_ion(fd_b),
        mk_descriptor_ion(fd_c),
    ];
    let mut htp_mh = [ptr::null_mut::<c_void>(); 3];
    let err = unsafe {
        (v_htp.memRegister.unwrap())(htp_ctx, htp_descs.as_ptr(), 3, htp_mh.as_mut_ptr())
    };
    anyhow::ensure!(err == 0, "HTP memRegister err=0x{:x}", err);
    println!("HTP memRegister(a,b,c) ION: OK");

    // QNN-GPU: GPU backend는 ION 미지원, DMA_BUF 시도
    let gpu_descs = [
        mk_descriptor_dmabuf(fd_c, rpc_c),
        mk_descriptor_dmabuf(fd_b, rpc_b),
        mk_descriptor_dmabuf(fd_d, rpc_d),
    ];
    let mut gpu_mh = [ptr::null_mut::<c_void>(); 3];
    let err = unsafe {
        (v_gpu.memRegister.unwrap())(gpu_ctx, gpu_descs.as_ptr(), 3, gpu_mh.as_mut_ptr())
    };
    if err != 0 {
        eprintln!("⚠ QNN-GPU memRegister err=0x{:x}", err);
    } else {
        println!("QNN-GPU memRegister(c,b,d): OK — same fd_c registered with both backends");
    }
    let gpu_register_ok = err == 0;

    // ── Build HTP graph: c = a + b ──
    fn build_eltwise_add(
        v: &QnnInterface_ImplementationV2_25_t,
        ctx: Qnn_ContextHandle_t,
        graph_name: &str,
        dims: &[u32],
    ) -> anyhow::Result<(
        Qnn_GraphHandle_t,
        Qnn_Tensor_t,
        Qnn_Tensor_t,
        Qnn_Tensor_t,
        CString,
        CString,
        CString,
        CString,
        CString,
        CString,
    )> {
        let g_name = CString::new(graph_name).unwrap();
        let mut g: Qnn_GraphHandle_t = ptr::null_mut();
        let err =
            unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut g) };
        anyhow::ensure!(err == 0, "graphCreate({}) err=0x{:x}", graph_name, err);

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
        let n_a = CString::new("a").unwrap();
        let n_b = CString::new("b").unwrap();
        let n_c = CString::new("c").unwrap();
        let mut t_a = Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_v1(&n_a, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE),
            },
        };
        let mut t_b = Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_v1(&n_b, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE),
            },
        };
        let mut t_c = Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_v1(&n_c, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ),
            },
        };
        for (l, t) in [("a", &mut t_a), ("b", &mut t_b), ("c", &mut t_c)] {
            let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(g, t) };
            anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", l, err);
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
        let err = unsafe { (v.graphAddNode.unwrap())(g, op) };
        anyhow::ensure!(err == 0, "graphAddNode err=0x{:x}", err);
        let err = unsafe { (v.graphFinalize.unwrap())(g, ptr::null_mut(), ptr::null_mut()) };
        anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);
        let _ = inputs;
        let _ = outputs;
        Ok((g, t_a, t_b, t_c, n_a, n_b, n_c, op_name, pkg, op_type))
    }

    // HTP graph: c = a + b
    let (htp_g, mut htp_t_a, mut htp_t_b, mut htp_t_c, _, _, _, _, _, _) =
        build_eltwise_add(&v_htp, htp_ctx, "htp_add", &dims)?;
    println!("HTP graph (a + b → c) finalize: OK");

    // QNN-GPU graph: d = c + b (reads HTP-written c)
    let (gpu_g, mut gpu_t_c, mut gpu_t_b, mut gpu_t_d, _, _, _, _, _, _) = if gpu_register_ok {
        let r = build_eltwise_add(&v_gpu, gpu_ctx, "gpu_add", &dims)?;
        println!("QNN-GPU graph (c + b → d) finalize: OK");
        r
    } else {
        anyhow::bail!("Cannot build GPU graph: memRegister failed earlier");
    };

    // Switch HTP tensors to MEMHANDLE
    unsafe {
        htp_t_a.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        htp_t_a.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = htp_mh[0];
        htp_t_b.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        htp_t_b.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = htp_mh[1];
        htp_t_c.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        htp_t_c.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = htp_mh[2];
    }
    // Switch GPU tensors to MEMHANDLE — gpu_descs order: c=mh[0], b=mh[1], d=mh[2]
    unsafe {
        gpu_t_c.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        gpu_t_c.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = gpu_mh[0];
        gpu_t_b.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        gpu_t_b.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = gpu_mh[1];
        gpu_t_d.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        gpu_t_d.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = gpu_mh[2];
    }

    // Fill input data via host pointers
    let host_a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let host_b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 + 5.0).collect();
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

    // ── Step 1: HTP execute (a + b → c) ──
    let htp_inputs = [htp_t_a, htp_t_b];
    let mut htp_outputs = [htp_t_c];
    let err = unsafe {
        (v_htp.graphExecute.unwrap())(
            htp_g,
            htp_inputs.as_ptr(),
            2,
            htp_outputs.as_mut_ptr(),
            1,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    anyhow::ensure!(err == 0, "HTP graphExecute err=0x{:x}", err);
    println!("HTP execute (a + b → c): OK");

    // Sanity: read rpc_c via host pointer (HTP write visible to host?)
    unsafe {
        let c_slice = std::slice::from_raw_parts(rpc_c as *const f32, 4);
        println!(
            "  rpc_c [0..4] via host pointer: {:?} (expected {:?})",
            c_slice,
            (0..4).map(|i| host_a[i] + host_b[i]).collect::<Vec<_>>()
        );
    }

    // ── Step 2: QNN-GPU execute (c + b → d) — reads HTP-written c ──
    let gpu_inputs = [gpu_t_c, gpu_t_b];
    let mut gpu_outputs = [gpu_t_d];
    let err = unsafe {
        (v_gpu.graphExecute.unwrap())(
            gpu_g,
            gpu_inputs.as_ptr(),
            2,
            gpu_outputs.as_mut_ptr(),
            1,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    anyhow::ensure!(err == 0, "QNN-GPU graphExecute err=0x{:x}", err);
    println!("QNN-GPU execute (c + b → d): OK");

    // ── Verify ──
    let mut mismatch = 0usize;
    let mut first_bad = None;
    unsafe {
        let d_slice = std::slice::from_raw_parts(rpc_d as *const f32, n);
        for i in 0..n {
            let exp_c = host_a[i] + host_b[i]; // HTP output
            let exp_d = exp_c + host_b[i]; // QNN-GPU output: c + b
            if (d_slice[i] - exp_d).abs() > 1e-3 * exp_d.abs().max(1.0) {
                mismatch += 1;
                if first_bad.is_none() {
                    first_bad = Some((i, d_slice[i], exp_d, exp_c));
                }
            }
        }
    }
    if mismatch == 0 {
        println!("\n=== R-Y option 3: ✓ PASS — HTP↔QNN-GPU zero-copy via shared ION fd 작동 ===");
        println!(
            "  {}/{} elements match. HeteroInfer cross-backend pattern reproduced.",
            n, n
        );
    } else {
        println!("\n=== R-Y option 3: ✗ FAIL ===");
        println!("  {} mismatch / {}", mismatch, n);
        if let Some((i, got, exp_d, exp_c)) = first_bad {
            println!(
                "  first bad: d[{}]={} expected={} (HTP c was {})",
                i, got, exp_d, exp_c
            );
        }
    }

    // Cleanup
    unsafe {
        let _ = (v_htp.memDeRegister.unwrap())(htp_mh.as_ptr(), 3);
        let _ = (v_gpu.memDeRegister.unwrap())(gpu_mh.as_ptr(), 3);
        let _ = (v_htp.contextFree.unwrap())(htp_ctx, ptr::null_mut());
        let _ = (v_gpu.contextFree.unwrap())(gpu_ctx, ptr::null_mut());
        let _ = (v_htp.backendFree.unwrap())(htp_be);
        let _ = (v_gpu.backendFree.unwrap())(gpu_be);
        rpcmem_free(rpc_a);
        rpcmem_free(rpc_b);
        rpcmem_free(rpc_c);
        rpcmem_free(rpc_d);
    }
    if mismatch == 0 {
        Ok(())
    } else {
        std::process::exit(1)
    }
}
