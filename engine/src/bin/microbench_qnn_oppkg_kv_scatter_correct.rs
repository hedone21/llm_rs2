//! M2.E + M3.4 D-D.2 CustomKvScatter correctness — production OpPackage vs raw OpenCL.
//!
//! Maps to `kernel_kv_scatter_f32_to_f16_oop` in `engine/kernels/simple_ops.cl`.
//! The OOP variant reads `write_pos` from `pos_buf[0]` instead of taking it as
//! a SCALAR op param. Reason: QNN bakes SCALAR params at graph finalize, which
//! made multi-token decode produce garbage (every token wrote into pos=0).
//!
//! Kernel:
//!   k_dst[h * capacity * head_dim + write_pos * head_dim + d] = (half)k_src[h * head_dim + d]
//!   v_dst[h * capacity * head_dim + write_pos * head_dim + d] = (half)v_src[h * head_dim + d]
//!   where write_pos = pos_buf[0]
//!
//! **Multi-output design** (M2.H, 2026-05-09):
//!   - inputs[0] = k_src, inputs[1] = v_src, inputs[2] = pos_buf (M3.4 D-D.1)
//!   - outputs[0] = k_dst, outputs[1] = v_dst (both claimed outputs)
//!   - 7 kernel args: k_src, v_src, pos_buf, k_dst, v_dst, head_dim, capacity
//!
//! Cases (kv_heads=2, head_dim=128, capacity=2048) × write_pos ∈ {0, 10, 100, 1000}.
//! pos=0 reproduces M2.E GREEN baseline. pos != 0 cases are the new D-D evidence
//! that proves the runtime pos buffer correctly drives the destination row.
//!
//! Pass criterion: max_abs_err < 1e-2 (F32→F16 cast tolerance).
//!
//! Build (Android cross):
//!   cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!     --bin microbench_qnn_oppkg_kv_scatter_correct
//!
//! Pre-deploy (CRITICAL — run_device.py does NOT auto-rebuild qnn_oppkg cdylib):
//!   cargo build --release -p qnn_oppkg --target aarch64-linux-android
//!   adb push target/aarch64-linux-android/release/libqnn_oppkg.so /data/local/tmp/
//!
//! Run on device:
//!   adb shell "LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:/vendor/lib64 \
//!              /data/local/tmp/microbench_qnn_oppkg_kv_scatter_correct"

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_oppkg_kv_scatter_correct requires --features qnn");
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

    println!("=== microbench_qnn_oppkg_kv_scatter_correct (M2.E + M3.4 D-D.2) ===\n");
    println!("Op Package: {}", PKG_PATH);
    println!("Interface symbol: {}", PKG_PROVIDER);

    let backend_lib_path = std::env::var("QNN_BACKEND_LIB")
        .unwrap_or_else(|_| "/data/local/tmp/qnn/libQnnGpu.so".to_string());
    println!("Backend lib: {}\n", backend_lib_path);

    // ── Path A: raw OpenCL reference ──────────────────────────────────────────
    use ocl::{Context, Device, Platform, Program, Queue};

    let platform = Platform::default();
    let device = Device::first(platform)?;
    let cl_ctx = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let cl_q = Queue::new(&cl_ctx, device, None)?;
    let kernel_src = include_str!("../../kernels/simple_ops.cl");
    let cl_program = Program::builder()
        .devices(device)
        .src(kernel_src)
        .cmplr_opt("-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math")
        .build(&cl_ctx)?;
    // Reference: same OOP kernel as the OpPackage path so both read pos via
    // pos_buf. This proves the QNN-side runtime pos buffer matches a raw
    // OpenCL execution that uses the same buffer-based pos.
    let cl_kernel = ocl::core::create_kernel(&cl_program, "kernel_kv_scatter_f32_to_f16_oop")?;

    // ── Path B: QNN-GPU backend + register OpPackage ──────────────────────────
    let gpu_lib = unsafe { Library::new(&backend_lib_path) }?;
    type GetProvidersFn = unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let gp: Symbol<GetProvidersFn> = unsafe { gpu_lib.get(b"QnnInterface_getProviders\0")? };
    let mut provs: *mut *const QnnInterface_t = ptr::null_mut();
    let mut np: c_uint = 0;
    let err = unsafe { gp(&mut provs, &mut np) };
    anyhow::ensure!(err == 0 && np > 0, "GPU getProviders err=0x{:x}", err);
    let v = unsafe { (**provs).__bindgen_anon_1.v2_25 };

    let mut logger: Qnn_LogHandle_t = ptr::null_mut();
    if let Some(log_create) = v.logCreate {
        let err = unsafe { log_create(None, QnnLog_Level_t_QNN_LOG_LEVEL_ERROR, &mut logger) };
        if err != 0 {
            eprintln!("logCreate err=0x{:x} (proceeding without logger)", err);
            logger = ptr::null_mut();
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

    // rpcmem for DMA_BUF tensors
    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;
    type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(po: *const c_void) -> i32;
    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }?;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let _rpcmem_free: Symbol<RpcmemFreeFn> = unsafe { rpc_lib.get(b"rpcmem_free\0")? };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };

    // Single shape, multiple write_pos values to exercise multi-pos path.
    let kv_heads: usize = 2;
    let head_dim: usize = 128;
    let capacity: usize = 2048;
    let write_positions: &[i32] = &[0, 10, 100, 1000];
    let mut all_pass = true;

    for &write_pos in write_positions {
        println!(
            "--- (kv_heads={}, head_dim={}, capacity={}, write_pos={}) ---",
            kv_heads, head_dim, capacity, write_pos
        );

        let result = run_case(
            &v,
            ctx,
            &cl_q,
            &cl_ctx,
            &cl_kernel,
            &rpcmem_alloc,
            &rpcmem_to_fd,
            RPCMEM_HEAP_ID_SYSTEM,
            RPCMEM_DEFAULT_FLAGS,
            kv_heads,
            head_dim,
            capacity,
            write_pos,
        );

        match result {
            Ok((max_k, max_v)) => {
                let pass_k = max_k < 1e-2;
                let pass_v = max_v < 1e-2;
                println!(
                    "  k_dst max_abs_err = {:.6e}  {}",
                    max_k,
                    if pass_k { "PASS" } else { "FAIL" }
                );
                println!(
                    "  v_dst max_abs_err = {:.6e}  {}",
                    max_v,
                    if pass_v { "PASS" } else { "FAIL" }
                );
                if !(pass_k && pass_v) {
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
        "\n=== M3.4 D-D.2 verdict: {} ===",
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

/// Convert f32 to f16 bits (same helper as matmul_correct).
#[cfg(feature = "qnn")]
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 {
        return sign << 15;
    }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 {
        return sign << 15;
    }
    if new_exp >= 31 {
        return (sign << 15) | (0x1f << 10);
    }
    let new_mant = (mant >> 13) as u16;
    (sign << 15) | ((new_exp as u16) << 10) | new_mant
}

/// Convert f16 bits to f32.
#[cfg(feature = "qnn")]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign: u32 = ((bits >> 15) & 0x1) as u32;
    let exp: u32 = ((bits >> 10) & 0x1f) as u32;
    let mant: u32 = (bits & 0x3ff) as u32;
    if exp == 0 {
        // subnormal / zero
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // subnormal: normalize
        let mut e: u32 = 0;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let f32_exp = 127 - 14 - e;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | ((m & 0x3ff) << 13));
    }
    if exp == 31 {
        // inf or NaN
        return f32::from_bits((sign << 31) | 0x7f80_0000 | (mant << 13));
    }
    let f32_exp = exp - 15 + 127;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
}

#[cfg(feature = "qnn")]
#[allow(clippy::too_many_arguments)]
fn run_case(
    v: &qnn::QnnInterface_ImplementationV2_25_t,
    ctx: qnn::Qnn_ContextHandle_t,
    cl_q: &ocl::Queue,
    cl_ctx: &ocl::Context,
    cl_kernel: &ocl::core::Kernel,
    rpcmem_alloc: &libloading::Symbol<unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void>,
    rpcmem_to_fd: &libloading::Symbol<unsafe extern "C" fn(*const std::ffi::c_void) -> i32>,
    heap_id: i32,
    flags: u32,
    kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    write_pos: i32,
) -> anyhow::Result<(f32, f32)> {
    use ocl::core::ArgVal;
    use qnn::*;
    use std::ffi::CString;
    use std::ptr;

    let src_total = kv_heads * head_dim;
    let dst_total = kv_heads * capacity * head_dim;

    // ── Generate deterministic F32 inputs ─────────────────────────────────────
    let mut host_k_src = vec![0.0f32; src_total];
    let mut host_v_src = vec![0.0f32; src_total];
    for i in 0..src_total {
        host_k_src[i] = (((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5) * 4.0;
        host_v_src[i] = (((i as f32) * 0.0241 + 0.13).rem_euclid(1.0) - 0.5) * 4.0;
    }

    // Pre-initialize dst buffers as zero (F16 = 0x0000).
    let host_k_dst_init = vec![0u16; dst_total];
    let host_v_dst_init = vec![0u16; dst_total];

    // ── Path A: raw OpenCL reference using kernel_kv_scatter_f32_to_f16_oop ───
    // Args (matches new kernel signature exactly):
    //   (k_src, v_src, pos_buf, k_dst, v_dst, head_dim, capacity)
    let buf_k_src = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_ONLY,
            src_total,
            None,
        )?
    };
    let buf_v_src = unsafe {
        ocl::core::create_buffer::<_, f32>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_ONLY,
            src_total,
            None,
        )?
    };
    let buf_pos = unsafe {
        ocl::core::create_buffer::<_, i32>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, 1, None)?
    };
    // dst buffers as u16 (half); ocl treats them as raw bytes.
    let buf_k_dst = unsafe {
        ocl::core::create_buffer::<_, u16>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_WRITE,
            dst_total,
            None,
        )?
    };
    let buf_v_dst = unsafe {
        ocl::core::create_buffer::<_, u16>(
            cl_ctx.as_core(),
            ocl::core::MEM_READ_WRITE,
            dst_total,
            None,
        )?
    };

    let host_pos = [write_pos];
    unsafe {
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_k_src,
            true,
            0,
            &host_k_src,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_v_src,
            true,
            0,
            &host_v_src,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_pos,
            true,
            0,
            &host_pos,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_k_dst,
            true,
            0,
            &host_k_dst_init,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_write_buffer(
            cl_q,
            &buf_v_dst,
            true,
            0,
            &host_v_dst_init,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    let head_dim_i: i32 = head_dim as i32;
    let capacity_i: i32 = capacity as i32;
    ocl::core::set_kernel_arg(cl_kernel, 0, ArgVal::mem(&buf_k_src))?;
    ocl::core::set_kernel_arg(cl_kernel, 1, ArgVal::mem(&buf_v_src))?;
    ocl::core::set_kernel_arg(cl_kernel, 2, ArgVal::mem(&buf_pos))?;
    ocl::core::set_kernel_arg(cl_kernel, 3, ArgVal::mem(&buf_k_dst))?;
    ocl::core::set_kernel_arg(cl_kernel, 4, ArgVal::mem(&buf_v_dst))?;
    ocl::core::set_kernel_arg(cl_kernel, 5, ArgVal::scalar(&head_dim_i))?;
    ocl::core::set_kernel_arg(cl_kernel, 6, ArgVal::scalar(&capacity_i))?;

    let global = [src_total, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            cl_kernel,
            1,
            None,
            &global,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    let mut ref_k_dst = vec![0u16; dst_total];
    let mut ref_v_dst = vec![0u16; dst_total];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_k_dst,
            true,
            0,
            &mut ref_k_dst,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_v_dst,
            true,
            0,
            &mut ref_v_dst,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // ── Path B: QNN graph with CustomKvScatter ────────────────────────────────
    let src_bytes = (src_total * 4) as i32; // F32
    let dst_bytes = (dst_total * 2) as i32; // F16 (2 bytes per element)
    let pos_bytes: i32 = 4; // INT_32 [1]

    let rpc_k_src = unsafe { rpcmem_alloc(heap_id, flags, src_bytes) };
    let rpc_v_src = unsafe { rpcmem_alloc(heap_id, flags, src_bytes) };
    let rpc_pos = unsafe { rpcmem_alloc(heap_id, flags, pos_bytes) };
    let rpc_k_dst = unsafe { rpcmem_alloc(heap_id, flags, dst_bytes) };
    let rpc_v_dst = unsafe { rpcmem_alloc(heap_id, flags, dst_bytes) };
    anyhow::ensure!(
        !rpc_k_src.is_null()
            && !rpc_v_src.is_null()
            && !rpc_pos.is_null()
            && !rpc_k_dst.is_null()
            && !rpc_v_dst.is_null(),
        "rpcmem_alloc failed"
    );

    unsafe {
        std::ptr::copy_nonoverlapping(
            host_k_src.as_ptr() as *const u8,
            rpc_k_src as *mut u8,
            src_bytes as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_v_src.as_ptr() as *const u8,
            rpc_v_src as *mut u8,
            src_bytes as usize,
        );
        // Write the runtime pos value into pos_buf[0].
        *(rpc_pos as *mut i32) = write_pos;
        // Zero-init k_dst and v_dst
        std::ptr::write_bytes(rpc_k_dst as *mut u8, 0, dst_bytes as usize);
        std::ptr::write_bytes(rpc_v_dst as *mut u8, 0, dst_bytes as usize);
    }

    let fd_k_src = unsafe { rpcmem_to_fd(rpc_k_src) };
    let fd_v_src = unsafe { rpcmem_to_fd(rpc_v_src) };
    let fd_pos = unsafe { rpcmem_to_fd(rpc_pos) };
    let fd_k_dst = unsafe { rpcmem_to_fd(rpc_k_dst) };
    let fd_v_dst = unsafe { rpcmem_to_fd(rpc_v_dst) };

    let mut dims_src = vec![src_total as u32];
    let mut dims_pos = vec![1u32];
    let mut dims_dst = vec![dst_total as u32];
    let mut dims_dst_v = vec![dst_total as u32];

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

    macro_rules! mk_tensor {
        ($ttype:expr, $dtype:expr, $dims:expr) => {
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: Qnn_TensorV1_t {
                        id: 0,
                        name: ptr::null(),
                        type_: $ttype,
                        dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                        dataType: $dtype,
                        quantizeParams: qp,
                        rank: 1,
                        dimensions: $dims,
                        memType: Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
                        __bindgen_anon_1: Qnn_TensorV1_t__bindgen_ty_1 {
                            clientBuf: Qnn_ClientBuffer_t {
                                data: ptr::null_mut(),
                                dataSize: 0,
                            },
                        },
                    },
                },
            }
        };
    }

    let name_k_src = CString::new(format!("kvs_k_src_{}", write_pos)).unwrap();
    let name_v_src = CString::new(format!("kvs_v_src_{}", write_pos)).unwrap();
    let name_pos = CString::new(format!("kvs_pos_{}", write_pos)).unwrap();
    let name_k_dst = CString::new(format!("kvs_k_dst_{}", write_pos)).unwrap();
    let name_v_dst = CString::new(format!("kvs_v_dst_{}", write_pos)).unwrap();

    let mut t_k_src = mk_tensor!(
        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
        dims_src.as_mut_ptr()
    );
    t_k_src.__bindgen_anon_1.v1.name = name_k_src.as_ptr();

    let mut t_v_src = mk_tensor!(
        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
        dims_src.as_mut_ptr()
    );
    t_v_src.__bindgen_anon_1.v1.name = name_v_src.as_ptr();

    let mut t_pos = mk_tensor!(
        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        Qnn_DataType_t_QNN_DATATYPE_INT_32,
        dims_pos.as_mut_ptr()
    );
    t_pos.__bindgen_anon_1.v1.name = name_pos.as_ptr();

    // M2.H multi-output: both k_dst and v_dst are claimed outputs (APP_READ).
    let mut t_k_dst = mk_tensor!(
        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
        Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
        dims_dst.as_mut_ptr()
    );
    t_k_dst.__bindgen_anon_1.v1.name = name_k_dst.as_ptr();

    let mut t_v_dst = mk_tensor!(
        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
        Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
        dims_dst_v.as_mut_ptr()
    );
    t_v_dst.__bindgen_anon_1.v1.name = name_v_dst.as_ptr();

    let g_name = CString::new(format!("kvs_graph_{}", write_pos)).unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();
    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), ptr::null_mut(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);

    for (label, t) in [
        ("k_src", &mut t_k_src),
        ("v_src", &mut t_v_src),
        ("pos", &mut t_pos),
        ("k_dst", &mut t_k_dst),
        ("v_dst", &mut t_v_dst),
    ] {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", label, err);
    }

    // Op params: head_dim, capacity (write_pos is now in pos_buf, M3.4 D-D.1).
    let p_name_hd = CString::new("head_dim").unwrap();
    let p_name_cap = CString::new("capacity").unwrap();

    let mut params = [
        Qnn_Param_t {
            paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            name: p_name_hd.as_ptr(),
            __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                scalarParam: Qnn_Scalar_t {
                    dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                    __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                        int32Value: head_dim as i32,
                    },
                },
            },
        },
        Qnn_Param_t {
            paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            name: p_name_cap.as_ptr(),
            __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                scalarParam: Qnn_Scalar_t {
                    dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                    __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                        int32Value: capacity as i32,
                    },
                },
            },
        },
    ];

    let op_name = CString::new(format!("kvs_op0_{}", write_pos)).unwrap();
    let pkg = CString::new("qnn_oppkg").unwrap();
    let op_type = CString::new("CustomKvScatter").unwrap();
    let mut inputs = [t_k_src, t_v_src, t_pos];
    let mut outputs = [t_k_dst, t_v_dst];
    let op = Qnn_OpConfig_t {
        version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
        __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
            v1: Qnn_OpConfigV1_t {
                name: op_name.as_ptr(),
                packageName: pkg.as_ptr(),
                typeName: op_type.as_ptr(),
                numOfParams: 2,
                params: params.as_mut_ptr(),
                numOfInputs: 3,
                inputTensors: inputs.as_mut_ptr(),
                numOfOutputs: 2,
                outputTensors: outputs.as_mut_ptr(),
            },
        },
    };
    let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
    anyhow::ensure!(err == 0, "graphAddNode err=0x{:x}", err);

    let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
    anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);

    let mk_desc = |fd: i32,
                   host_data: *mut std::ffi::c_void,
                   dtype: Qnn_DataType_t,
                   dims: &[u32]|
     -> Qnn_MemDescriptor_t {
        Qnn_MemDescriptor_t {
            memShape: Qnn_MemShape_t {
                numDim: dims.len() as u32,
                dimSize: dims.as_ptr() as *mut u32,
                shapeConfig: ptr::null(),
            },
            dataType: dtype,
            memType: Qnn_MemType_t_QNN_MEM_TYPE_DMA_BUF,
            __bindgen_anon_1: Qnn_MemDescriptor_t__bindgen_ty_1 {
                dmaBufInfo: Qnn_MemDmaBufInfo_t {
                    fd,
                    data: host_data,
                },
            },
        }
    };

    let dims_src_u = vec![src_total as u32];
    let dims_pos_u = vec![1u32];
    let dims_dst_u = vec![dst_total as u32];
    let descs = [
        mk_desc(
            fd_k_src,
            rpc_k_src,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            &dims_src_u,
        ),
        mk_desc(
            fd_v_src,
            rpc_v_src,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            &dims_src_u,
        ),
        mk_desc(
            fd_pos,
            rpc_pos,
            Qnn_DataType_t_QNN_DATATYPE_INT_32,
            &dims_pos_u,
        ),
        mk_desc(
            fd_k_dst,
            rpc_k_dst,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
            &dims_dst_u,
        ),
        mk_desc(
            fd_v_dst,
            rpc_v_dst,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
            &dims_dst_u,
        ),
    ];
    let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 5];
    let err = unsafe { (v.memRegister.unwrap())(ctx, descs.as_ptr(), 5, mh.as_mut_ptr()) };
    anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);

    inputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[0];
    inputs[1].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[1];
    inputs[2].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    inputs[2].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[2];
    outputs[0].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    outputs[0].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[3];
    outputs[1].__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
    outputs[1].__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = mh[4];

    let err = unsafe {
        (v.graphExecute.unwrap())(
            graph,
            inputs.as_ptr(),
            3,
            outputs.as_mut_ptr(),
            2,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);

    // ── Compare QNN output vs reference ──────────────────────────────────────
    // The written region is write_pos row for each head:
    //   dst_idx = h * capacity * head_dim + write_pos * head_dim + d
    let mut max_abs_k = 0.0f32;
    let mut max_abs_v = 0.0f32;
    unsafe {
        let test_k_dst = std::slice::from_raw_parts(rpc_k_dst as *const u16, dst_total);
        let test_v_dst = std::slice::from_raw_parts(rpc_v_dst as *const u16, dst_total);
        for h in 0..kv_heads {
            for d in 0..head_dim {
                let dst_idx = h * capacity * head_dim + (write_pos as usize) * head_dim + d;
                let ref_k = f16_bits_to_f32(ref_k_dst[dst_idx]);
                let ref_v = f16_bits_to_f32(ref_v_dst[dst_idx]);
                let tst_k = f16_bits_to_f32(test_k_dst[dst_idx]);
                let tst_v = f16_bits_to_f32(test_v_dst[dst_idx]);
                let dk = (ref_k - tst_k).abs();
                let dv = (ref_v - tst_v).abs();
                if dk > max_abs_k {
                    max_abs_k = dk;
                }
                if dv > max_abs_v {
                    max_abs_v = dv;
                }
            }
        }
    }

    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), 5);
    }

    Ok((max_abs_k, max_abs_v))
}
