//! microbench_htp_opencl_interop — Phase 32b-? / R-Y: HTP↔OpenCL zero-copy interop
//!
//! 목적: rpcmem (ION-backed dmabuf) → cl_khr_external_memory_dma_buf로 cl_mem 생성
//! → GPU kernel이 host-written data를 zero-copy로 read 검증.
//!
//! 두 단계 검증:
//!   Stage A (sanity):  CPU host→rpcmem write → GPU read (data 일치)
//!   Stage B (HeteroInfer): HTP write to rpcmem → GPU read (zero-copy via dmabuf)
//!
//! Pass-gate (R-Y):
//!   - cl_khr_external_memory_dma_buf 통한 cl_mem 생성 OK
//!   - GPU read 결과 1024/1024 정확
//!   - sync cost 측정 (host write → GPU read 사이)
//!
//! Build: cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!        --bin microbench_htp_opencl_interop
//! Run:   `LD_LIBRARY_PATH=/data/local/tmp/qnn:/vendor/lib64 \
//!         ADSP_LIBRARY_PATH=/data/local/tmp/qnn \
//!         adb shell /data/local/tmp/microbench_htp_opencl_interop`

#[cfg(not(all(feature = "qnn", feature = "opencl")))]
fn main() {
    eprintln!("microbench_htp_opencl_interop requires --features qnn,opencl");
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
    use ocl::core::ClContextPtr;
    use ocl::{Context, Device, Platform, Program, Queue};
    use std::ffi::{CString, c_void};
    use std::os::raw::c_uint;
    use std::ptr;
    use std::time::Instant;

    use qnn::*;

    // CL 3.0 + cl_khr_external_memory_dma_buf
    const CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR: i64 = 0x2067;
    const CL_DEVICE_HANDLE_LIST_KHR: i64 = 0x2051;
    const CL_DEVICE_HANDLE_LIST_END_KHR: i64 = 0x2052;
    const CL_MEM_READ_ONLY: u64 = 1 << 2;
    const CL_MEM_READ_WRITE: u64 = 1 << 0;

    // rpcmem
    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;

    type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(po: *const c_void) -> i32;

    type CreateBufferWithPropsFn = unsafe extern "C" fn(
        context: ocl::ffi::cl_context,
        properties: *const i64,
        flags: u64,
        size: usize,
        host_ptr: *mut c_void,
        errcode_ret: *mut i32,
    ) -> ocl::ffi::cl_mem;
    type AcquireExtMemFn = unsafe extern "C" fn(
        command_queue: ocl::ffi::cl_command_queue,
        num_mem_objects: u32,
        mem_objects: *const ocl::ffi::cl_mem,
        num_events_in_wait_list: u32,
        event_wait_list: *const ocl::ffi::cl_event,
        event: *mut ocl::ffi::cl_event,
    ) -> i32;

    let n_elements: usize = 1024;
    let bytes = (n_elements * 4) as i32;

    println!("=== microbench_htp_opencl_interop (R-Y / Phase 32b-interop) ===\n");
    println!("Buffer: {} f32 elements ({} bytes)", n_elements, bytes);

    // ── Load rpcmem + libOpenCL ──
    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }
        .or_else(|_| unsafe { Library::new("libcdsprpc.so") })?;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let rpcmem_free: Symbol<RpcmemFreeFn> = unsafe { rpc_lib.get(b"rpcmem_free\0")? };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };

    let cl_lib = unsafe { Library::new("libOpenCL.so") }?;
    let create_buf_props: Symbol<CreateBufferWithPropsFn> =
        unsafe { cl_lib.get(b"clCreateBufferWithProperties\0")? };
    let acquire_ext: Symbol<AcquireExtMemFn> =
        unsafe { cl_lib.get(b"clEnqueueAcquireExternalMemObjectsKHR\0")? };
    let release_ext: Symbol<AcquireExtMemFn> =
        unsafe { cl_lib.get(b"clEnqueueReleaseExternalMemObjectsKHR\0")? };
    println!("dlopen libcdsprpc.so + libOpenCL.so: OK");
    println!("dlsym clCreateBufferWithProperties + Acquire/Release ExternalMemObjectsKHR: OK");

    // ── OpenCL setup ──
    let platform = Platform::default();
    let device = Device::first(platform)?;
    let cl_ctx = Context::builder().platform(platform).devices(device).build()?;
    let queue = Queue::new(&cl_ctx, device, None)?;
    let ctx_raw: ocl::ffi::cl_context = ClContextPtr::as_ptr(&&cl_ctx);
    println!("OpenCL platform/device/context: OK ({})", device.name()?);

    // ── Stage A: CPU host→rpcmem write → GPU read ──
    println!("\n[Stage A] CPU host write → rpcmem → GPU kernel read");

    let rpc_in = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    let rpc_out = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    anyhow::ensure!(!rpc_in.is_null() && !rpc_out.is_null(), "rpcmem_alloc failed");
    let fd_in = unsafe { rpcmem_to_fd(rpc_in) };
    let fd_out = unsafe { rpcmem_to_fd(rpc_out) };
    println!(
        "  rpcmem: in={:p} fd={}, out={:p} fd={}",
        rpc_in, fd_in, rpc_out, fd_out
    );

    // Fill input via CPU
    let host_data: Vec<f32> = (0..n_elements).map(|i| (i as f32) * 0.5 + 1.0).collect();
    unsafe {
        std::ptr::copy_nonoverlapping(
            host_data.as_ptr() as *const u8,
            rpc_in as *mut u8,
            bytes as usize,
        );
    }

    // Create cl_mem from dma-buf fd via cl_khr_external_memory_dma_buf
    let props_in = [CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR, fd_in as i64, 0];
    let mut err: i32 = 0;
    let cl_in = unsafe {
        create_buf_props(
            ctx_raw,
            props_in.as_ptr(),
            CL_MEM_READ_ONLY,
            bytes as usize,
            ptr::null_mut(),
            &mut err,
        )
    };
    if err != 0 || cl_in.is_null() {
        anyhow::bail!("clCreateBufferWithProperties (in, dmabuf) err={}", err);
    }
    println!("  cl_mem (in) from dmabuf fd: OK ({:p})", cl_in);

    let props_out = [CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR, fd_out as i64, 0];
    let cl_out = unsafe {
        create_buf_props(
            ctx_raw,
            props_out.as_ptr(),
            CL_MEM_READ_WRITE,
            bytes as usize,
            ptr::null_mut(),
            &mut err,
        )
    };
    if err != 0 || cl_out.is_null() {
        anyhow::bail!("clCreateBufferWithProperties (out, dmabuf) err={}", err);
    }
    println!("  cl_mem (out) from dmabuf fd: OK ({:p})", cl_out);

    // Wrap as ocl Mem (ocl::core::Mem is *mut cl_mem wrapper)
    let cl_in_mem = unsafe { ocl::core::Mem::from_raw_copied_ptr(cl_in) };
    let cl_out_mem = unsafe { ocl::core::Mem::from_raw_copied_ptr(cl_out) };

    // Build kernel: out[i] = in[i] * 2 + 7
    let src = r#"
        __kernel void scale(__global const float* in_buf, __global float* out_buf) {
            int i = get_global_id(0);
            out_buf[i] = in_buf[i] * 2.0f + 7.0f;
        }
    "#;
    let prog = Program::builder().devices(device).src(src).build(&cl_ctx)?;
    let kernel = ocl::core::create_kernel(&prog, "scale")?;

    let q_raw = unsafe { *(&queue as *const Queue as *const ocl::ffi::cl_command_queue) };
    let t_kernel = Instant::now();
    unsafe {
        // Acquire: transition dma-buf ownership host→GPU
        let mems_ac = [cl_in, cl_out];
        let r = acquire_ext(q_raw, 2, mems_ac.as_ptr(), 0, ptr::null(), ptr::null_mut());
        anyhow::ensure!(r == 0, "Acquire (Stage A) err={}", r);

        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(&cl_in_mem))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(&cl_out_mem))?;
        ocl::core::enqueue_kernel(
            &queue,
            &kernel,
            1,
            None,
            &[n_elements, 1, 1],
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;

        // Release: transition back GPU→host
        let r = release_ext(q_raw, 2, mems_ac.as_ptr(), 0, ptr::null(), ptr::null_mut());
        anyhow::ensure!(r == 0, "Release (Stage A) err={}", r);
    }
    ocl::core::finish(&queue)?;
    let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1000.0;
    println!("  GPU kernel exec (with acquire/release): {:.3} ms", kernel_ms);

    // Read result via host pointer (zero-copy claim test)
    let mut mismatch = 0usize;
    let mut first_bad = None;
    unsafe {
        let out_slice = std::slice::from_raw_parts(rpc_out as *const f32, n_elements);
        for i in 0..n_elements {
            let exp = host_data[i] * 2.0 + 7.0;
            if (out_slice[i] - exp).abs() > 1e-4 * exp.abs().max(1.0) {
                mismatch += 1;
                if first_bad.is_none() {
                    first_bad = Some((i, out_slice[i], exp));
                }
            }
        }
    }
    if mismatch == 0 {
        println!("  Stage A correctness: ✓ {}/{} match (host_write→GPU_read zero-copy)", n_elements, n_elements);
    } else {
        println!("  Stage A correctness: ✗ {} mismatch", mismatch);
        if let Some((i, got, exp)) = first_bad {
            println!("    first bad: out[{}]={} (expected {})", i, got, exp);
        }
    }

    // ── Sync cost: host write → GPU read latency ──
    println!("\n[Sync cost] host pointer write → GPU sees update");
    let mut sync_samples = Vec::with_capacity(20);
    for it in 0..20 {
        // Fresh write via host pointer
        let new_val = it as f32 * 0.1;
        unsafe {
            let p = rpc_in as *mut f32;
            for i in 0..n_elements {
                *p.add(i) = new_val + (i as f32);
            }
        }
        let t0 = Instant::now();
        unsafe {
            let mems_ac = [cl_in, cl_out];
            let _ = acquire_ext(q_raw, 2, mems_ac.as_ptr(), 0, ptr::null(), ptr::null_mut());
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(&cl_in_mem))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(&cl_out_mem))?;
            ocl::core::enqueue_kernel(
                &queue,
                &kernel,
                1,
                None,
                &[n_elements, 1, 1],
                None,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            let _ = release_ext(q_raw, 2, mems_ac.as_ptr(), 0, ptr::null(), ptr::null_mut());
        }
        ocl::core::finish(&queue)?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        sync_samples.push(ms);

        // Verify (only first 8 elements for speed)
        unsafe {
            let out_slice = std::slice::from_raw_parts(rpc_out as *const f32, 8);
            for i in 0..8 {
                let exp = (new_val + i as f32) * 2.0 + 7.0;
                if (out_slice[i] - exp).abs() > 1e-4 * exp.abs().max(1.0) {
                    eprintln!("  Sync iter {} sample bad: out[{}]={} expected={}", it, i, out_slice[i], exp);
                }
            }
        }
    }
    let n = sync_samples.len() as f64;
    let mean = sync_samples.iter().sum::<f64>() / n;
    let mut sorted = sync_samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "  20-iter mean GPU kernel time (incl any cache sync): {:.3} ms (median {:.3})",
        mean, sorted[10]
    );

    // ── Stage B: HTP write to rpcmem → GPU read ──
    println!("\n[Stage B] HTP write (ElementWiseAdd) → GPU read same dmabuf cl_mem");

    // HTP setup
    let htp_lib = unsafe { Library::new("/data/local/tmp/qnn/libQnnHtp.so") }
        .or_else(|_| unsafe { Library::new("libQnnHtp.so") })?;
    type GetProvidersFn =
        unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let get_providers: Symbol<GetProvidersFn> =
        unsafe { htp_lib.get(b"QnnInterface_getProviders\0")? };
    let mut providers: *mut *const QnnInterface_t = ptr::null_mut();
    let mut num: c_uint = 0;
    let err = unsafe { get_providers(&mut providers, &mut num) };
    anyhow::ensure!(err == 0 && num > 0, "HTP getProviders err=0x{:x}", err);
    let v = unsafe { (**providers).__bindgen_anon_1.v2_25 };

    let mut backend: Qnn_BackendHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.backendCreate.unwrap())(ptr::null_mut(), ptr::null_mut(), &mut backend) };
    anyhow::ensure!(err == 0, "backendCreate err=0x{:x}", err);
    let mut htp_ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe {
        (v.contextCreate.unwrap())(backend, ptr::null_mut(), ptr::null_mut(), &mut htp_ctx)
    };
    anyhow::ensure!(err == 0, "contextCreate err=0x{:x}", err);

    // Need 3 rpcmem buffers for ElementWiseAdd: a, b, c
    let rpc_a = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    let rpc_b = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    let rpc_c = unsafe { rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes) };
    anyhow::ensure!(
        !rpc_a.is_null() && !rpc_b.is_null() && !rpc_c.is_null(),
        "stage B rpcmem alloc failed"
    );
    let fd_a = unsafe { rpcmem_to_fd(rpc_a) };
    let fd_b = unsafe { rpcmem_to_fd(rpc_b) };
    let fd_c = unsafe { rpcmem_to_fd(rpc_c) };

    // Register fds with QNN
    let dims = vec![n_elements as u32];
    let mk_descriptor = |fd: i32| -> Qnn_MemDescriptor_t {
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
    let descs = [mk_descriptor(fd_a), mk_descriptor(fd_b), mk_descriptor(fd_c)];
    let mut mh = [ptr::null_mut::<c_void>(); 3];
    let err = unsafe {
        (v.memRegister.unwrap())(htp_ctx, descs.as_ptr(), 3, mh.as_mut_ptr())
    };
    anyhow::ensure!(err == 0, "QnnMem_register err=0x{:x}", err);

    // Build HTP graph (RAW + null at finalize, MEMHANDLE at execute)
    let g_name = CString::new("interop").unwrap();
    let mut htp_graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err = unsafe {
        (v.graphCreate.unwrap())(htp_ctx, g_name.as_ptr(), ptr::null_mut(), &mut htp_graph)
    };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    let mk_v1_raw = |name: &CString, ttype: Qnn_TensorType_t| -> Qnn_TensorV1_t {
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
    let n_a = CString::new("a").unwrap();
    let n_b = CString::new("b").unwrap();
    let n_c = CString::new("c").unwrap();
    let mut t_a = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&n_a, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE),
        },
    };
    let mut t_b = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&n_b, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE),
        },
    };
    let mut t_c = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1_raw(&n_c, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ),
        },
    };
    for (l, t) in [("a", &mut t_a), ("b", &mut t_b), ("c", &mut t_c)] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(htp_graph, t) };
        anyhow::ensure!(err == 0, "tensorCreateGraphTensor({}) err=0x{:x}", l, err);
    }
    let op_n = CString::new("add").unwrap();
    let pkg = CString::new("qti.aisw").unwrap();
    let op_t = CString::new("ElementWiseAdd").unwrap();
    let mut htp_inputs = [t_a, t_b];
    let mut htp_outputs = [t_c];
    let op = Qnn_OpConfig_t {
        version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
        __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
            v1: Qnn_OpConfigV1_t {
                name: op_n.as_ptr(),
                packageName: pkg.as_ptr(),
                typeName: op_t.as_ptr(),
                numOfParams: 0,
                params: ptr::null_mut(),
                numOfInputs: 2,
                inputTensors: htp_inputs.as_mut_ptr(),
                numOfOutputs: 1,
                outputTensors: htp_outputs.as_mut_ptr(),
            },
        },
    };
    let err = unsafe { (v.graphAddNode.unwrap())(htp_graph, op) };
    anyhow::ensure!(err == 0, "graphAddNode err=0x{:x}", err);
    let err = unsafe { (v.graphFinalize.unwrap())(htp_graph, ptr::null_mut(), ptr::null_mut()) };
    anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);

    // Switch to MEMHANDLE for execute
    unsafe {
        htp_inputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        htp_inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[0];
        htp_inputs[1].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        htp_inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[1];
        htp_outputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        htp_outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[2];
    }

    // Fill HTP inputs via host pointer
    let h_a: Vec<f32> = (0..n_elements).map(|i| (i as f32) * 0.1).collect();
    let h_b: Vec<f32> = (0..n_elements).map(|i| (i as f32) * 0.01 + 5.0).collect();
    unsafe {
        std::ptr::copy_nonoverlapping(h_a.as_ptr() as *const u8, rpc_a as *mut u8, bytes as usize);
        std::ptr::copy_nonoverlapping(h_b.as_ptr() as *const u8, rpc_b as *mut u8, bytes as usize);
    }

    // HTP execute (writes to rpc_c via MEMHANDLE)
    let err = unsafe {
        (v.graphExecute.unwrap())(
            htp_graph,
            htp_inputs.as_ptr(),
            2,
            htp_outputs.as_mut_ptr(),
            1,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    anyhow::ensure!(err == 0, "HTP graphExecute err=0x{:x}", err);
    println!("  HTP ElementWiseAdd → rpc_c: OK");

    // Now create cl_mem from rpc_c fd, GPU kernel reads it
    let props_c = [CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR, fd_c as i64, 0];
    let mut cl_err: i32 = 0;
    let cl_c = unsafe {
        create_buf_props(
            ctx_raw,
            props_c.as_ptr(),
            CL_MEM_READ_ONLY,
            bytes as usize,
            ptr::null_mut(),
            &mut cl_err,
        )
    };
    anyhow::ensure!(!cl_c.is_null() && cl_err == 0, "cl_mem from rpc_c fd err={}", cl_err);
    let cl_c_mem = unsafe { ocl::core::Mem::from_raw_copied_ptr(cl_c) };

    // Use scale kernel to read rpc_c (in cl_in's place), output to cl_out
    unsafe {
        let mems_ac = [cl_c, cl_out];
        let r = acquire_ext(q_raw, 2, mems_ac.as_ptr(), 0, ptr::null(), ptr::null_mut());
        anyhow::ensure!(r == 0, "Acquire (Stage B) err={}", r);
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(&cl_c_mem))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(&cl_out_mem))?;
        ocl::core::enqueue_kernel(
            &queue,
            &kernel,
            1,
            None,
            &[n_elements, 1, 1],
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        let r = release_ext(q_raw, 2, mems_ac.as_ptr(), 0, ptr::null(), ptr::null_mut());
        anyhow::ensure!(r == 0, "Release (Stage B) err={}", r);
    }
    ocl::core::finish(&queue)?;

    // Verify: GPU saw HTP-written data
    let mut htp_mismatch = 0usize;
    let mut htp_first = None;
    unsafe {
        let out_slice = std::slice::from_raw_parts(rpc_out as *const f32, n_elements);
        for i in 0..n_elements {
            let htp_val = h_a[i] + h_b[i]; // what HTP wrote to rpc_c
            let exp = htp_val * 2.0 + 7.0; // GPU kernel applied to it
            if (out_slice[i] - exp).abs() > 1e-3 * exp.abs().max(1.0) {
                htp_mismatch += 1;
                if htp_first.is_none() {
                    htp_first = Some((i, out_slice[i], exp, htp_val));
                }
            }
        }
    }
    if htp_mismatch == 0 {
        println!(
            "  Stage B correctness: ✓ {}/{} match (HTP write → GPU read via dmabuf zero-copy)",
            n_elements, n_elements
        );
    } else {
        println!("  Stage B correctness: ✗ {} mismatch", htp_mismatch);
        if let Some((i, got, exp, htp)) = htp_first {
            println!("    first bad: out[{}]={} expected={} (HTP_val was {})", i, got, exp, htp);
        }
    }

    // ── R-Y verdict ──
    println!("\n=== R-Y summary ===");
    println!(
        "  cl_khr_external_memory_dma_buf cl_mem 생성: ✓"
    );
    println!(
        "  Stage A (host→rpcmem→GPU): {}",
        if mismatch == 0 { "✓ PASS" } else { "✗ FAIL" }
    );
    println!(
        "  Stage B (HTP→rpcmem→GPU): {}",
        if htp_mismatch == 0 { "✓ PASS" } else { "✗ FAIL" }
    );
    println!("  Sync cost (GPU kernel mean): {:.3} ms", mean);

    if mismatch == 0 && htp_mismatch == 0 {
        println!(
            "\n=> R-Y PASS: HTP↔OpenCL zero-copy interop 검증 OK. \n   진정 async weight loading 가능 (HTP write + GPU read 동시 가능)."
        );
    } else if mismatch == 0 && htp_mismatch != 0 {
        println!(
            "\n=> R-Y PARTIAL: host→GPU OK, HTP→GPU 미정. cache sync 검토 필요."
        );
    } else {
        println!("\n=> R-Y FAIL: dmabuf zero-copy 자체 작동 안 함.");
    }

    // Cleanup
    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), 3);
        let _ = (v.contextFree.unwrap())(htp_ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(backend);
        rpcmem_free(rpc_in);
        rpcmem_free(rpc_out);
        rpcmem_free(rpc_a);
        rpcmem_free(rpc_b);
        rpcmem_free(rpc_c);
    }

    if mismatch == 0 && htp_mismatch == 0 { Ok(()) } else { std::process::exit(1) }
}
