//! microbench_qnngpu_clmem_share — Phase R Wave 2: QNN_GPU_MEM_OPENCL custom path
//!
//! 목적: production OpenCL cl_mem을 QNN-GPU에 register하여 sync-free 메모리 공유 검증.
//! Path: Qnn_MemDescriptor_t { memType=CUSTOM, customInfo=QnnGpu_MemInfoCustom_t {
//!         memType=OPENCL, buffer=host_ptr } }
//! 의의: forward는 production OpenCL 그대로 + QNN-GPU는 같은 cl_mem read/write
//!      → R-B1 RED를 우회 가능 (forward 성능 무손실 + QNN-GPU의 추가 capability)
//!
//! Test sequence:
//!   1. cl::Buffer with CL_MEM_USE_HOST_PTR (host_a, host_b, host_c shared)
//!   2. host에서 host_a, host_b 채움 (USE_HOST_PTR이라 cl_mem 동일 view)
//!   3. QNN-GPU에 host_a/b/c를 QNN_GPU_MEM_OPENCL custom으로 register
//!   4. QNN-GPU MAT_MUL execute (input host_a, host_b → output host_c)
//!   5. host_c verify (CPU reference 비교)
//!   6. 추가: OpenCL kernel로 host_c overwrite → QNN 다시 execute → sync free 검증
//!
//! Build: cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!        --bin microbench_qnngpu_clmem_share

#[cfg(not(all(feature = "qnn", feature = "opencl")))]
fn main() {
    eprintln!("microbench_qnngpu_clmem_share requires --features qnn,opencl");
    std::process::exit(2);
}

#[cfg(all(feature = "qnn", feature = "opencl"))]
#[allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code
)]
mod qnn {
    include!(concat!(env!("OUT_DIR"), "/qnn_bindings.rs"));
}

#[cfg(all(feature = "qnn", feature = "opencl"))]
fn main() -> anyhow::Result<()> {
    use libloading::{Library, Symbol};
    use std::ffi::{CString, c_void};
    use std::os::raw::c_uint;
    use std::ptr;

    use qnn::*;

    // ─────────────────────────────────────────────────────────
    // Tiny PoC dim: 1×128×128 (verifiable by hand)
    // ─────────────────────────────────────────────────────────
    let m: usize = 1;
    let k: usize = 128;
    let n: usize = 128;
    let bytes_a = m * k * 4;
    let bytes_b = k * n * 4;
    let bytes_c = m * n * 4;

    println!("=== microbench_qnngpu_clmem_share (Phase R / QNN_GPU_MEM_OPENCL) ===\n");
    println!("Dim: M={}, K={}, N={} (FP32)", m, k, n);
    println!("Path: cl::Buffer USE_HOST_PTR → QNN-GPU customInfo (OPENCL)\n");

    // ─────────────────────────────────────────────────────────
    // OpenCL setup with USE_HOST_PTR for shared host memory
    // ─────────────────────────────────────────────────────────
    use ocl::core::{ArgVal, MEM_USE_HOST_PTR};
    use ocl::{Context, Device, Platform, Program, Queue};

    let platform = Platform::default();
    let device = Device::first(platform)?;
    let cl_ctx = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let cl_q = Queue::new(&cl_ctx, device, None)?;

    // Shared host buffers (page-aligned not strictly required but cleaner)
    let mut host_a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01) - 0.5).collect();
    let mut host_b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.001) + 0.1).collect();
    let mut host_c: Vec<f32> = vec![0.0; m * n];

    // cl_mem with USE_HOST_PTR (zero-copy on UMA, references same physical memory)
    let buf_a = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            MEM_USE_HOST_PTR,
            bytes_a / 4,
            Some(&host_a),
        )?
    };
    let buf_b = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            MEM_USE_HOST_PTR,
            bytes_b / 4,
            Some(&host_b),
        )?
    };
    let buf_c = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            MEM_USE_HOST_PTR,
            bytes_c / 4,
            Some(&host_c),
        )?
    };
    println!("OpenCL cl::Buffer USE_HOST_PTR: A,B,C alloc OK");
    println!(
        "  host_a ptr: {:p}, host_b ptr: {:p}, host_c ptr: {:p}",
        host_a.as_ptr(),
        host_b.as_ptr(),
        host_c.as_ptr()
    );

    // ─────────────────────────────────────────────────────────
    // QNN-GPU backend
    // ─────────────────────────────────────────────────────────
    let gpu_lib = unsafe { Library::new("/data/local/tmp/qnn/libQnnGpu.so") }?;
    type GetProvidersFn = unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let gp: Symbol<GetProvidersFn> = unsafe { gpu_lib.get(b"QnnInterface_getProviders\0")? };
    let mut provs: *mut *const QnnInterface_t = ptr::null_mut();
    let mut np: c_uint = 0;
    let err = unsafe { gp(&mut provs, &mut np) };
    anyhow::ensure!(err == 0 && np > 0, "GPU getProviders err=0x{:x}", err);
    let v = unsafe { (**provs).__bindgen_anon_1.v2_25 };

    let mut be: Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe { (v.backendCreate.unwrap())(ptr::null_mut(), ptr::null_mut(), &mut be) };
    anyhow::ensure!(err == 0, "GPU backendCreate err=0x{:x}", err);
    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe { (v.contextCreate.unwrap())(be, ptr::null_mut(), ptr::null_mut(), &mut ctx) };
    anyhow::ensure!(err == 0, "GPU contextCreate err=0x{:x}", err);
    println!("QNN-GPU backend + context: OK");

    // ─────────────────────────────────────────────────────────
    // Register cl_mem (via host_ptr) as QNN_MEM_TYPE_CUSTOM + QNN_GPU_MEM_OPENCL
    // ─────────────────────────────────────────────────────────
    // QnnGpu_MemInfoCustom_t (not in bindgen, define manually from QnnGpuMem.h:33)
    #[repr(C)]
    struct QnnGpuMemInfoCustom {
        mem_type: u32, // 0 = QNN_GPU_MEM_OPENCL
        buffer: *mut c_void,
    }

    let dims_a: Vec<u32> = vec![m as u32, k as u32];
    let dims_b: Vec<u32> = vec![k as u32, n as u32];
    let dims_c: Vec<u32> = vec![m as u32, n as u32];

    // customInfo objects must outlive memRegister call.
    // Per QnnGpuMem.h docs the buffer should be a cl_mem handle (opaque). Pass
    // ocl::core::Mem::as_ptr() which returns the underlying cl_mem.
    let cust_a = QnnGpuMemInfoCustom {
        mem_type: 0,
        buffer: buf_a.as_ptr() as *mut c_void,
    };
    let cust_b = QnnGpuMemInfoCustom {
        mem_type: 0,
        buffer: buf_b.as_ptr() as *mut c_void,
    };
    let cust_c = QnnGpuMemInfoCustom {
        mem_type: 0,
        buffer: buf_c.as_ptr() as *mut c_void,
    };
    println!(
        "  cl_mem A: {:p}, cl_mem B: {:p}, cl_mem C: {:p}",
        buf_a.as_ptr(),
        buf_b.as_ptr(),
        buf_c.as_ptr()
    );

    let mk_mem_desc = |dims: &[u32], cust: &QnnGpuMemInfoCustom| -> Qnn_MemDescriptor_t {
        Qnn_MemDescriptor_t {
            memShape: Qnn_MemShape_t {
                numDim: dims.len() as u32,
                dimSize: dims.as_ptr() as *mut u32,
                shapeConfig: ptr::null(),
            },
            dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            memType: Qnn_MemType_t_QNN_MEM_TYPE_CUSTOM,
            __bindgen_anon_1: Qnn_MemDescriptor_t__bindgen_ty_1 {
                customInfo: cust as *const _ as *mut c_void,
            },
        }
    };
    let descs = [
        mk_mem_desc(&dims_a, &cust_a),
        mk_mem_desc(&dims_b, &cust_b),
        mk_mem_desc(&dims_c, &cust_c),
    ];
    let mut mh = [ptr::null_mut::<c_void>(); 3];
    let err = unsafe { (v.memRegister.unwrap())(ctx, descs.as_ptr(), 3, mh.as_mut_ptr()) };
    println!("memRegister(CUSTOM, host_ptr) -> err = 0x{:x}", err);
    anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);
    println!("✓ memRegister succeeded — QNN-GPU accepts host_ptr as cl_mem-backing\n");

    // ─────────────────────────────────────────────────────────
    // Build QNN-GPU MatMul graph
    // ─────────────────────────────────────────────────────────
    let g_name = CString::new("clmem_matmul").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    let mk_v1 = |name: &CString, ttype: Qnn_TensorType_t, dims: &[u32]| -> Qnn_TensorV1_t {
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
            rank: dims.len() as u32,
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
    let n_a = CString::new("A").unwrap();
    let n_b = CString::new("B").unwrap();
    let n_c = CString::new("C").unwrap();
    let mut t_a = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1(&n_a, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims_a),
        },
    };
    let mut t_b = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1(&n_b, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, &dims_b),
        },
    };
    let mut t_c = Qnn_Tensor_t {
        version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
            v1: mk_v1(&n_c, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, &dims_c),
        },
    };
    for (l, t) in [("A", &mut t_a), ("B", &mut t_b), ("C", &mut t_c)] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", l, err);
    }
    let op_name = CString::new("mm0").unwrap();
    let pkg = CString::new("qti.aisw").unwrap();
    let op_type = CString::new("MatMul").unwrap();
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
    anyhow::ensure!(err == 0, "graphAddNode err=0x{:x}", err);
    let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
    anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);
    println!("QNN-GPU graph (MatMul {}x{}x{}) finalize: OK", m, k, n);

    // Switch to MEMHANDLE
    inputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[0];
    inputs[1].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[1];
    outputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[2];

    // ─────────────────────────────────────────────────────────
    // Test 1: explicit OpenCL writeBuffer → QNN execute → readBuffer
    // (USE_HOST_PTR may stage to device-private; force sync via explicit ops)
    // ─────────────────────────────────────────────────────────
    println!("\n--- Test 1: explicit clEnqueueWriteBuffer → QNN exec → clEnqueueReadBuffer ---");
    unsafe {
        ocl::core::enqueue_write_buffer(
            &cl_q,
            &buf_a,
            true,
            0,
            &host_a,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            &cl_q,
            &buf_b,
            true,
            0,
            &host_b,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(&cl_q)?;

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
    println!("graphExecute: OK");

    // Read back from cl_mem to host
    unsafe {
        ocl::core::enqueue_read_buffer(
            &cl_q,
            &buf_c,
            true,
            0,
            &mut host_c,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(&cl_q)?;

    // CPU reference (compare host_c to A @ B)
    let mut ref_c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += host_a[i * k + p] * host_b[p * n + j];
            }
            ref_c[i * n + j] = s;
        }
    }

    let mut max_abs = 0.0f32;
    let mut mismatch = 0usize;
    for i in 0..m * n {
        let d = (host_c[i] - ref_c[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        if d > 1e-2 {
            mismatch += 1;
        }
    }
    println!(
        "host_c vs ref_c: max_abs_err = {:.6}, mismatch (>1e-2) = {} / {}",
        max_abs,
        mismatch,
        m * n
    );
    let test1_pass = mismatch == 0 && max_abs < 1e-2;
    println!("Test 1: {}", if test1_pass { "✓ PASS" } else { "✗ FAIL" });

    // ─────────────────────────────────────────────────────────
    // Test 2: OpenCL kernel write to cl_mem A → QNN execute → host read
    // (sync free 검증: clFinish 없이도 QNN이 같은 view 보는가?)
    // ─────────────────────────────────────────────────────────
    println!("\n--- Test 2: OpenCL kernel write → QNN execute (no explicit sync) ---");
    let fill_src = r#"
        __kernel void fill_seq(__global float* buf, int len) {
            int i = get_global_id(0);
            if (i < len) buf[i] = (float)i * 0.001f + 0.5f;
        }
    "#;
    let cl_program = Program::builder()
        .devices(device)
        .src(fill_src)
        .build(&cl_ctx)?;
    let cl_kernel = ocl::core::create_kernel(&cl_program, "fill_seq")?;
    let len_mk = (m * k) as i32;
    ocl::core::set_kernel_arg(&cl_kernel, 0, ArgVal::mem(&buf_a))?;
    ocl::core::set_kernel_arg(&cl_kernel, 1, ArgVal::scalar(&len_mk))?;
    unsafe {
        ocl::core::enqueue_kernel(
            &cl_q,
            &cl_kernel,
            1,
            None,
            &[m * k, 1, 1],
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    // INTENTIONALLY no clFinish — test sync-free property.
    // (QNN execute may need to see the OpenCL writes despite no explicit fence.)

    // QNN execute (input host_a was filled by OpenCL kernel)
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

    // Verify: ref_c should be (cl-filled host_a) @ host_b
    let mut new_a = vec![0.0f32; m * k];
    for i in 0..m * k {
        new_a[i] = (i as f32) * 0.001 + 0.5;
    }
    let mut ref_c2 = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += new_a[i * k + p] * host_b[p * n + j];
            }
            ref_c2[i * n + j] = s;
        }
    }
    let mut max_abs2 = 0.0f32;
    let mut mismatch2 = 0usize;
    for i in 0..m * n {
        let d = (host_c[i] - ref_c2[i]).abs();
        if d > max_abs2 {
            max_abs2 = d;
        }
        if d > 1e-2 {
            mismatch2 += 1;
        }
    }
    println!(
        "host_c vs ref_c2: max_abs_err = {:.6}, mismatch (>1e-2) = {} / {}",
        max_abs2,
        mismatch2,
        m * n
    );
    let test2_pass = mismatch2 == 0 && max_abs2 < 1e-2;
    println!(
        "Test 2 (sync-free): {}",
        if test2_pass {
            "✓ PASS — OpenCL write visible to QNN-GPU without explicit sync"
        } else {
            "✗ FAIL — sync required (host_c shows stale data)"
        }
    );

    // ─────────────────────────────────────────────────────────
    // Test 3: same as Test 2 but with clFinish (control)
    // ─────────────────────────────────────────────────────────
    println!("\n--- Test 3: OpenCL kernel write + clFinish → QNN execute (control) ---");
    // overwrite with different pattern
    let fill_src2 = r#"
        __kernel void fill_seq2(__global float* buf, int len) {
            int i = get_global_id(0);
            if (i < len) buf[i] = (float)i * (-0.002f) + 0.25f;
        }
    "#;
    let prog2 = Program::builder()
        .devices(device)
        .src(fill_src2)
        .build(&cl_ctx)?;
    let kern2 = ocl::core::create_kernel(&prog2, "fill_seq2")?;
    ocl::core::set_kernel_arg(&kern2, 0, ArgVal::mem(&buf_a))?;
    ocl::core::set_kernel_arg(&kern2, 1, ArgVal::scalar(&len_mk))?;
    unsafe {
        ocl::core::enqueue_kernel(
            &cl_q,
            &kern2,
            1,
            None,
            &[m * k, 1, 1],
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(&cl_q)?; // explicit sync

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

    let mut new_a3 = vec![0.0f32; m * k];
    for i in 0..m * k {
        new_a3[i] = (i as f32) * (-0.002) + 0.25;
    }
    let mut ref_c3 = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += new_a3[i * k + p] * host_b[p * n + j];
            }
            ref_c3[i * n + j] = s;
        }
    }
    let mut max_abs3 = 0.0f32;
    let mut mismatch3 = 0usize;
    for i in 0..m * n {
        let d = (host_c[i] - ref_c3[i]).abs();
        if d > max_abs3 {
            max_abs3 = d;
        }
        if d > 1e-2 {
            mismatch3 += 1;
        }
    }
    println!(
        "host_c vs ref_c3: max_abs_err = {:.6}, mismatch (>1e-2) = {} / {}",
        max_abs3,
        mismatch3,
        m * n
    );
    let test3_pass = mismatch3 == 0 && max_abs3 < 1e-2;
    println!(
        "Test 3 (with clFinish): {}",
        if test3_pass { "✓ PASS" } else { "✗ FAIL" }
    );

    // ─────────────────────────────────────────────────────────
    // Summary
    // ─────────────────────────────────────────────────────────
    println!("\n=== Summary ===");
    println!(
        "Test 1 (host fill → QNN read):           {}",
        if test1_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "Test 2 (OpenCL write → QNN, no sync):    {}",
        if test2_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "Test 3 (OpenCL write → QNN, clFinish):   {}",
        if test3_pass { "PASS" } else { "FAIL" }
    );
    println!();
    if test1_pass && test3_pass {
        println!("✓ QNN_GPU_MEM_OPENCL custom path 작동 (cl_mem-backed host_ptr 공유 가능)");
        if test2_pass {
            println!("✓ Sync-free 메모리 공유 확정 — clFinish 불필요");
        } else {
            println!("△ Sync 필요 — clFinish/fence 필수, 그래도 zero-copy");
        }
    } else if test1_pass {
        println!("△ host fill만 작동 — OpenCL kernel write는 안 보임");
    } else {
        println!("✗ memRegister는 성공했지만 graphExecute 결과 부정확 — host_ptr 공유 안 됨");
    }

    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), 3);
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(be);
    }
    let _ = buf_a;
    let _ = buf_b;
    let _ = buf_c;
    let _ = (&mut host_a, &mut host_b, &mut host_c); // explicit drop after QNN cleanup
    Ok(())
}
