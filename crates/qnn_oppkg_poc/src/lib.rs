//! QNN GPU Op Package PoC — wraps a single ElementWiseAdd OpenCL kernel as a
//! QNN op so that it executes inside the QNN-GPU runtime's cl_context.
//!
//! Goals (Phase R Wave 2 follow-up):
//!   GA: registerOpPackage loads this .so successfully
//!   GB: graphAddNode with our op_type passes validateOpConfig
//!   GC: graphExecute runs our OpenCL kernel and writes the expected output
//!   GD: a co-located prebuilt op (e.g. another ElementWiseAdd) shares the
//!       same cl_mem with our kernel without explicit sync
//!
//! Interface version: V1.4 (9 functions). Symbol name exposed for
//! `interfaceProvider`: `QnnOpPackage_InitInterface`.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals, dead_code)]

#[allow(clippy::all)]
mod qnn {
    include!(concat!(env!("OUT_DIR"), "/qnn_bindings.rs"));
}

use qnn::*;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::sync::OnceLock;

// bindgen forward-declared _QnnOpPackage_Node_t / _QnnOpPackage_OpImpl_t as opaque
// (because the QnnInterface.h forward declaration is processed first). We redeclare
// the GPU-specialized layouts here and cast as needed.
#[repr(C)]
pub struct GpuNode {
    pub optimization: u32,
    pub configs: *mut *const Qnn_OpConfig_t,
    pub storageTypes: *mut *const QnnGpu_TensorStorageType_t,
    pub kernelVariant: i32,
}
#[repr(C)]
pub struct GpuOperation {
    pub outputClaims: *mut *mut QnnGpu_OutputClaim_t,
    pub memoryObjects: *mut *mut QnnGpu_MemoryObject_t,
    pub kernels: *mut *mut QnnGpu_Kernel_t,
}

// ─────────────────────────────────────────────────────────
// Static op metadata. Lives for the lifetime of the .so.
// ─────────────────────────────────────────────────────────
static OP_TYPE_ADD: &str = "CustomAdd";
static OP_TYPE_MATMUL: &str = "CustomMatMul";
static PACKAGE_NAME: &str = "qnn_oppkg_poc";

// All allocations are leaked to 'static so QNN can hold raw pointers indefinitely.
struct StaticInfo {
    info: &'static QnnOpPackage_Info_t,
    add_kernel_source: &'static CString,
    add_kernel_name: &'static CString,
    add_build_options: &'static CString,
    matmul_kernel_source: &'static CString,
    matmul_kernel_name: &'static CString,
    matmul_build_options: &'static CString,
    _hash: &'static CString,
}

unsafe impl Sync for StaticInfo {}
unsafe impl Send for StaticInfo {}

static STATIC_INFO: OnceLock<StaticInfo> = OnceLock::new();

fn ensure_static() -> &'static StaticInfo {
    STATIC_INFO.get_or_init(|| {
        let package_name: &'static CString = Box::leak(Box::new(CString::new(PACKAGE_NAME).unwrap()));
        let op_name_add: &'static CString = Box::leak(Box::new(CString::new(OP_TYPE_ADD).unwrap()));
        let op_name_matmul: &'static CString =
            Box::leak(Box::new(CString::new(OP_TYPE_MATMUL).unwrap()));
        let operation_names: &'static [*const c_char] =
            Box::leak(Box::new([op_name_add.as_ptr(), op_name_matmul.as_ptr()]));
        // Match QNN_GPU_API_VERSION_INIT from QnnGpuCommon.h:32-46
        //   coreApiVersion 2.25.0, backendApiVersion 3.7.0
        let api_version: &'static Qnn_ApiVersion_t = Box::leak(Box::new(Qnn_ApiVersion_t {
            coreApiVersion: Qnn_Version_t { major: 2, minor: 25, patch: 0 },
            backendApiVersion: Qnn_Version_t { major: 3, minor: 7, patch: 0 },
        }));
        let opset_version: &'static Qnn_Version_t = Box::leak(Box::new(Qnn_Version_t {
            major: 1,
            minor: 0,
            patch: 0,
        }));
        // Use exact SDK build id so the backend doesn't reject us for mismatch.
        let build_id: &'static CString = Box::leak(Box::new(
            CString::new("v2.33.0.250327124043_117917").unwrap(),
        ));
        let hash: &'static CString = Box::leak(Box::new(CString::new("custom_add_v1").unwrap()));
        // GPU specialization
        #[repr(C)]
        struct GpuPackageInfo {
            kernel_repo_hash: *const c_char,
        }
        let gpu_pkg_info: &'static GpuPackageInfo = Box::leak(Box::new(GpuPackageInfo {
            kernel_repo_hash: hash.as_ptr(),
        }));
        let info: &'static QnnOpPackage_Info_t = Box::leak(Box::new(QnnOpPackage_Info_t {
            packageName: package_name.as_ptr(),
            operationNames: operation_names.as_ptr() as *mut _,
            operationInfo: ptr::null(),
            numOperations: 2,
            optimizations: ptr::null(),
            numOptimizations: 0,
            sdkBuildId: build_id.as_ptr(),
            sdkApiVersion: api_version,
            packageInfo: gpu_pkg_info as *const _ as *const QnnOpPackage_PackageInfo_t,
            opsetVersion: opset_version,
            reserved: [0; QNN_OP_PACKAGE_RESERVED_INFO_SIZE as usize],
        }));
        let add_kernel_source: &'static CString = Box::leak(Box::new(
            CString::new(
                r#"
__kernel void custom_add(__global const float* a, __global const float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
"#,
            )
            .unwrap(),
        ));
        let add_kernel_name: &'static CString =
            Box::leak(Box::new(CString::new("custom_add").unwrap()));
        let add_build_options: &'static CString =
            Box::leak(Box::new(CString::new("-cl-fast-relaxed-math").unwrap()));

        // Production GEMV kernel — F16 weight × F32 input → F32 output.
        let matmul_kernel_source: &'static CString = Box::leak(Box::new(
            CString::new(include_str!("../../../engine/kernels/mul_mv_f16_f32.cl")).unwrap(),
        ));
        let matmul_kernel_name: &'static CString =
            Box::leak(Box::new(CString::new("kernel_mul_mat_f16_f32").unwrap()));
        let matmul_build_options: &'static CString = Box::leak(Box::new(
            CString::new("-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math").unwrap(),
        ));

        StaticInfo {
            info,
            add_kernel_source,
            add_kernel_name,
            add_build_options,
            matmul_kernel_source,
            matmul_kernel_name,
            matmul_build_options,
            _hash: hash,
        }
    })
}

// ─────────────────────────────────────────────────────────
// V1.4 interface functions
// ─────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn pkg_init(
    _infrastructure: QnnOpPackage_GlobalInfrastructure_t,
) -> Qnn_ErrorHandle_t {
    eprintln!("[oppkg] pkg_init called, infra={:?}", _infrastructure);
    ensure_static();
    eprintln!("[oppkg] pkg_init returning SUCCESS");
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

// V2.0 create: same as init plus handle output
#[unsafe(no_mangle)]
pub extern "C" fn pkg_create(
    _infrastructure: QnnOpPackage_GlobalInfrastructure_t,
    _callback: QnnLog_Callback_t,
    _max_level: QnnLog_Level_t,
    op_package: *mut Qnn_OpPackageHandle_t,
) -> Qnn_ErrorHandle_t {
    eprintln!(
        "[oppkg] pkg_create called, infra={:?} handle_out={:?}",
        _infrastructure, op_package
    );
    if op_package.is_null() {
        return QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT as Qnn_ErrorHandle_t;
    }
    ensure_static();
    // Use static address as our opaque handle
    unsafe { *op_package = ensure_static() as *const _ as Qnn_OpPackageHandle_t };
    eprintln!("[oppkg] pkg_create returning SUCCESS");
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_validate_op_config_h(
    _handle: Qnn_OpPackageHandle_t,
    _op_config: Qnn_OpConfig_t,
) -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_create_op_impl_h(
    _handle: Qnn_OpPackageHandle_t,
    graph_infra: QnnOpPackage_GraphInfrastructure_t,
    node: QnnOpPackage_Node_t,
    op_impl: *mut QnnOpPackage_OpImpl_t,
) -> Qnn_ErrorHandle_t {
    pkg_create_op_impl(graph_infra, node, op_impl)
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_free_op_impl_h(
    _handle: Qnn_OpPackageHandle_t,
    op_impl: QnnOpPackage_OpImpl_t,
) -> Qnn_ErrorHandle_t {
    pkg_free_op_impl(op_impl)
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_log_set_level_h(
    _handle: Qnn_OpPackageHandle_t,
    _max_level: QnnLog_Level_t,
) -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_free(_handle: Qnn_OpPackageHandle_t) -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_terminate() -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_get_info(info: *mut *const QnnOpPackage_Info_t) -> Qnn_ErrorHandle_t {
    eprintln!("[oppkg] pkg_get_info called, info={:?}", info);
    if info.is_null() {
        return QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_INFO as Qnn_ErrorHandle_t;
    }
    let s = ensure_static();
    unsafe { *info = s.info };
    eprintln!(
        "[oppkg] pkg_get_info returning SUCCESS, info_ptr={:p}, opNames={:p}",
        s.info, s.info.operationNames
    );
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_validate_op_config(_op_config: Qnn_OpConfig_t) -> Qnn_ErrorHandle_t {
    // Accept any op of our type. Backend will only call us for op_type=CustomAdd.
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

// Per-node state kept alive between createOpImpl and freeOpImpl
struct OpImplState {
    out_claim: Box<QnnGpu_OutputClaim_t>,
    out_claim_arr: Vec<*mut QnnGpu_OutputClaim_t>,
    mem_objs: Vec<Box<QnnGpu_MemoryObject_t>>,
    mem_obj_arr: Vec<*mut QnnGpu_MemoryObject_t>,
    args: Vec<Box<QnnGpu_KernelArg_t>>,
    args_arr: Vec<*mut QnnGpu_KernelArg_t>,
    kernel: Box<QnnGpu_Kernel_t>,
    kernel_arr: Vec<*mut QnnGpu_Kernel_t>,
    operation: Box<GpuOperation>,
    out_dims: Vec<u32>,
    out_offsets: Vec<u32>,
}

// Per-node state extension: matmul scalar args
struct OpImplStateExt {
    inner: OpImplState,
    // matmul scalar args storage
    scalar_u64: Vec<Box<u64>>,
    scalar_i32: Vec<Box<i32>>,
    in_dims: Vec<Vec<u32>>,
    in_offs: Vec<Vec<u32>>,
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_create_op_impl(
    _graph_infra: QnnOpPackage_GraphInfrastructure_t,
    node: QnnOpPackage_Node_t,
    op_impl: *mut QnnOpPackage_OpImpl_t,
) -> Qnn_ErrorHandle_t {
    if node.is_null() || op_impl.is_null() {
        return QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT as Qnn_ErrorHandle_t;
    }
    let s = ensure_static();

    let gpu_node = node as *const GpuNode;
    let configs = unsafe { (*gpu_node).configs };
    if configs.is_null() {
        return QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT as Qnn_ErrorHandle_t;
    }
    let cfg0 = unsafe { *configs };
    if cfg0.is_null() {
        return QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT as Qnn_ErrorHandle_t;
    }
    let v1 = unsafe { (*cfg0).__bindgen_anon_1.v1 };
    if v1.numOfOutputs == 0 || v1.numOfInputs < 2 {
        return QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE as Qnn_ErrorHandle_t;
    }
    let op_type_str = unsafe {
        std::ffi::CStr::from_ptr(v1.typeName).to_str().unwrap_or("")
    };

    if op_type_str == OP_TYPE_MATMUL {
        return create_matmul_impl(s, v1, op_impl);
    }

    // Default: CustomAdd
    create_add_impl(s, v1, op_impl)
}

fn create_add_impl(
    s: &'static StaticInfo,
    v1: Qnn_OpConfigV1_t,
    op_impl: *mut QnnOpPackage_OpImpl_t,
) -> Qnn_ErrorHandle_t {
    let out_t = unsafe { *v1.outputTensors };
    let out_v1 = unsafe { out_t.__bindgen_anon_1.v1 };
    let out_rank = out_v1.rank;
    let out_dims_ptr = out_v1.dimensions;
    let mut total_elems: u32 = 1;
    for i in 0..out_rank {
        let d = unsafe { *out_dims_ptr.add(i as usize) };
        total_elems = total_elems.saturating_mul(d);
    }

    let mut state = Box::new(OpImplState {
        out_claim: Box::new(QnnGpu_OutputClaim_t {
            opConfigIndex: 0,
            outputIndex: 0,
            memoryObject: ptr::null(),
        }),
        out_claim_arr: Vec::new(),
        mem_objs: Vec::new(),
        mem_obj_arr: Vec::new(),
        args: Vec::new(),
        args_arr: Vec::new(),
        kernel: Box::new(unsafe { std::mem::zeroed() }),
        kernel_arr: Vec::new(),
        operation: Box::new(unsafe { std::mem::zeroed() }),
        out_dims: vec![total_elems],
        out_offsets: vec![0u32],
    });

    let dims_ptr = state.out_dims.as_mut_ptr();
    let offs_ptr = state.out_offsets.as_mut_ptr();
    for _ in 0..3 {
        state.mem_objs.push(Box::new(QnnGpu_MemoryObject_t {
            type_: QnnGpu_MemoryObjectType_t_QNN_GPU_MEM_OBJ_TYPE_BUFFER,
            dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            dimensions: dims_ptr,
            offsets: offs_ptr,
            numDimensions: 1,
            layout: QnnGpu_MemoryLayout_t_QNN_GPU_MEM_LAYOUT_UNDEFINED,
            blockEncodingInfo: QnnGpu_BlockEncodingInfo_t {
                bqBlockSize: ptr::null_mut(),
                bqEncodingTensorId: 0,
            },
        }));
    }
    state.out_claim.memoryObject = &*state.mem_objs[2] as *const _;
    state.out_claim_arr = vec![&mut *state.out_claim as *mut _, ptr::null_mut()];
    state.mem_obj_arr = state.mem_objs.iter_mut().map(|m| &mut **m as *mut _).collect();
    state.mem_obj_arr.push(ptr::null_mut());

    for (kind, ti) in [
        (QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 0),
        (QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 1),
        (QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE, 0),
    ] {
        state.args.push(Box::new(QnnGpu_KernelArg_t {
            type_: kind,
            __bindgen_anon_1: QnnGpu_KernelArg_t__bindgen_ty_1 {
                tensor: QnnGpu_TensorKernelArg_t {
                    opConfigIndex: 0,
                    tensorIndex: ti,
                    element: 0,
                },
            },
        }));
    }
    state.args_arr = state.args.iter_mut().map(|a| &mut **a as *mut _).collect();
    state.args_arr.push(ptr::null_mut());

    let src_bytes = s.add_kernel_source.as_bytes_with_nul();
    *state.kernel = QnnGpu_Kernel_t {
        kernelSource: src_bytes.as_ptr() as *const c_void,
        sourceLength: (src_bytes.len() - 1),
        sourceType: QnnGpu_KernelSourceType_t_QNN_GPU_KERNEL_SOURCE_TYPE_TEXT,
        buildOptions: s.add_build_options.as_ptr(),
        globalWorkDim: 1,
        globalWorkSizes: [total_elems as usize, 1, 1],
        localWorkDim: 0,
        localWorkSizes: [0, 0, 0],
        args: state.args_arr.as_mut_ptr(),
        name: s.add_kernel_name.as_ptr(),
        isDynamic: 0,
        tuningConfigs: ptr::null_mut(),
        reserved: ptr::null_mut(),
    };
    state.kernel_arr = vec![&mut *state.kernel as *mut _, ptr::null_mut()];
    *state.operation = GpuOperation {
        outputClaims: state.out_claim_arr.as_mut_ptr(),
        memoryObjects: state.mem_obj_arr.as_mut_ptr(),
        kernels: state.kernel_arr.as_mut_ptr(),
    };
    let op_ptr = &*state.operation as *const _ as *mut _QnnOpPackage_OpImpl_t;
    let _ = Box::into_raw(state);
    unsafe { *op_impl = op_ptr };
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

fn create_matmul_impl(
    s: &'static StaticInfo,
    v1: Qnn_OpConfigV1_t,
    op_impl: *mut QnnOpPackage_OpImpl_t,
) -> Qnn_ErrorHandle_t {
    // Inputs: weight [N, K] (F16), x [M, K] (F32)
    // Output: y [M, N] (F32)
    let in_w = unsafe { (*v1.inputTensors.add(0)).__bindgen_anon_1.v1 };
    let in_x = unsafe { (*v1.inputTensors.add(1)).__bindgen_anon_1.v1 };
    let out_y = unsafe { (*v1.outputTensors.add(0)).__bindgen_anon_1.v1 };
    let w_n = unsafe { *in_w.dimensions };
    let w_k = unsafe { *in_w.dimensions.add(1) };
    let x_m = unsafe { *in_x.dimensions };
    let _x_k = unsafe { *in_x.dimensions.add(1) };
    let _y_m = unsafe { *out_y.dimensions };
    let y_n = unsafe { *out_y.dimensions.add(1) };
    let k = w_k;
    let n = w_n.max(y_n);
    let m = x_m;

    let mut ext = Box::new(OpImplStateExt {
        inner: OpImplState {
            out_claim: Box::new(QnnGpu_OutputClaim_t {
                opConfigIndex: 0,
                outputIndex: 0,
                memoryObject: ptr::null(),
            }),
            out_claim_arr: Vec::new(),
            mem_objs: Vec::new(),
            mem_obj_arr: Vec::new(),
            args: Vec::new(),
            args_arr: Vec::new(),
            kernel: Box::new(unsafe { std::mem::zeroed() }),
            kernel_arr: Vec::new(),
            operation: Box::new(unsafe { std::mem::zeroed() }),
            out_dims: vec![(m * n) as u32],
            out_offsets: vec![0u32],
        },
        scalar_u64: Vec::new(),
        scalar_i32: Vec::new(),
        in_dims: vec![vec![(n * k) as u32], vec![(m * k) as u32], vec![(m * n) as u32]],
        in_offs: vec![vec![0u32], vec![0u32], vec![0u32]],
    });

    // 3 memory objects: weight (F16), x (F32), y (F32)
    let dims0_ptr = ext.in_dims[0].as_mut_ptr();
    let offs0_ptr = ext.in_offs[0].as_mut_ptr();
    let dims1_ptr = ext.in_dims[1].as_mut_ptr();
    let offs1_ptr = ext.in_offs[1].as_mut_ptr();
    let dims2_ptr = ext.in_dims[2].as_mut_ptr();
    let offs2_ptr = ext.in_offs[2].as_mut_ptr();

    let mo_w = Box::new(QnnGpu_MemoryObject_t {
        type_: QnnGpu_MemoryObjectType_t_QNN_GPU_MEM_OBJ_TYPE_BUFFER,
        dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
        dimensions: dims0_ptr,
        offsets: offs0_ptr,
        numDimensions: 1,
        layout: QnnGpu_MemoryLayout_t_QNN_GPU_MEM_LAYOUT_UNDEFINED,
        blockEncodingInfo: QnnGpu_BlockEncodingInfo_t {
            bqBlockSize: ptr::null_mut(),
            bqEncodingTensorId: 0,
        },
    });
    let mo_x = Box::new(QnnGpu_MemoryObject_t {
        type_: QnnGpu_MemoryObjectType_t_QNN_GPU_MEM_OBJ_TYPE_BUFFER,
        dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
        dimensions: dims1_ptr,
        offsets: offs1_ptr,
        numDimensions: 1,
        layout: QnnGpu_MemoryLayout_t_QNN_GPU_MEM_LAYOUT_UNDEFINED,
        blockEncodingInfo: QnnGpu_BlockEncodingInfo_t {
            bqBlockSize: ptr::null_mut(),
            bqEncodingTensorId: 0,
        },
    });
    let mo_y = Box::new(QnnGpu_MemoryObject_t {
        type_: QnnGpu_MemoryObjectType_t_QNN_GPU_MEM_OBJ_TYPE_BUFFER,
        dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
        dimensions: dims2_ptr,
        offsets: offs2_ptr,
        numDimensions: 1,
        layout: QnnGpu_MemoryLayout_t_QNN_GPU_MEM_LAYOUT_UNDEFINED,
        blockEncodingInfo: QnnGpu_BlockEncodingInfo_t {
            bqBlockSize: ptr::null_mut(),
            bqEncodingTensorId: 0,
        },
    });
    ext.inner.mem_objs.push(mo_w);
    ext.inner.mem_objs.push(mo_x);
    ext.inner.mem_objs.push(mo_y);

    ext.inner.out_claim.memoryObject = &*ext.inner.mem_objs[2] as *const _;
    ext.inner.out_claim_arr = vec![&mut *ext.inner.out_claim as *mut _, ptr::null_mut()];
    ext.inner.mem_obj_arr = ext
        .inner
        .mem_objs
        .iter_mut()
        .map(|m| &mut **m as *mut _)
        .collect();
    ext.inner.mem_obj_arr.push(ptr::null_mut());

    // Kernel args:
    //   0: src0 (weight)             tensor input 0
    //   1: offset0 = 0                ulong
    //   2: src1 (x)                   tensor input 1
    //   3: offset1 = 0                ulong
    //   4: dst (y)                    tensor output 0
    //   5: offsetd = 0                ulong
    //   6: ne00 = K                   int
    //   7: ne01 = N                   int
    //   8: ne02 = 1                   int
    //   9: ne10 = K                   int
    //  10: ne12 = 1                   int
    //  11: ne0  = N                   int
    //  12: ne1  = M                   int
    //  13: r2   = 1                   int
    //  14: r3   = 1                   int
    let mk_tensor_arg =
        |kind: QnnGpu_KernelArgType_t, ti: u32| -> QnnGpu_KernelArg_t {
            QnnGpu_KernelArg_t {
                type_: kind,
                __bindgen_anon_1: QnnGpu_KernelArg_t__bindgen_ty_1 {
                    tensor: QnnGpu_TensorKernelArg_t {
                        opConfigIndex: 0,
                        tensorIndex: ti,
                        element: 0,
                    },
                },
            }
        };
    let mk_ulong = |v: u64| -> QnnGpu_KernelArg_t {
        QnnGpu_KernelArg_t {
            type_: QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_DATA,
            __bindgen_anon_1: QnnGpu_KernelArg_t__bindgen_ty_1 {
                data: QnnGpu_DataKernelArg_t {
                    type_: QnnGpu_DataKernelArgType_t_QNN_GPU_KERNEL_ARG_CL_TYPE_ULONG,
                    __bindgen_anon_1: QnnGpu_DataKernelArg_t__bindgen_ty_1 { qnnULong: v },
                },
            },
        }
    };
    let mk_int = |v: i32| -> QnnGpu_KernelArg_t {
        QnnGpu_KernelArg_t {
            type_: QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_DATA,
            __bindgen_anon_1: QnnGpu_KernelArg_t__bindgen_ty_1 {
                data: QnnGpu_DataKernelArg_t {
                    type_: QnnGpu_DataKernelArgType_t_QNN_GPU_KERNEL_ARG_CL_TYPE_INT,
                    __bindgen_anon_1: QnnGpu_DataKernelArg_t__bindgen_ty_1 { qnnInt: v },
                },
            },
        }
    };

    let args_seq: Vec<QnnGpu_KernelArg_t> = vec![
        mk_tensor_arg(QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 0),
        mk_ulong(0),
        mk_tensor_arg(QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 1),
        mk_ulong(0),
        mk_tensor_arg(QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE, 0),
        mk_ulong(0),
        mk_int(k as i32),       // ne00
        mk_int(n as i32),       // ne01
        mk_int(1),              // ne02
        mk_int(k as i32),       // ne10
        mk_int(1),              // ne12
        mk_int(n as i32),       // ne0
        mk_int(m as i32),       // ne1
        mk_int(1),              // r2
        mk_int(1),              // r3
    ];
    for a in args_seq {
        ext.inner.args.push(Box::new(a));
    }
    ext.inner.args_arr = ext.inner.args.iter_mut().map(|a| &mut **a as *mut _).collect();
    ext.inner.args_arr.push(ptr::null_mut());

    let src_bytes = s.matmul_kernel_source.as_bytes_with_nul();
    let n_dst: u32 = 2;
    let global_x = ((n + n_dst - 1) / n_dst * 64) as usize;
    let global_y = (m * 4) as usize;
    *ext.inner.kernel = QnnGpu_Kernel_t {
        kernelSource: src_bytes.as_ptr() as *const c_void,
        sourceLength: (src_bytes.len() - 1),
        sourceType: QnnGpu_KernelSourceType_t_QNN_GPU_KERNEL_SOURCE_TYPE_TEXT,
        buildOptions: s.matmul_build_options.as_ptr(),
        globalWorkDim: 3,
        globalWorkSizes: [global_x, global_y, 1],
        localWorkDim: 3,
        localWorkSizes: [64, 4, 1],
        args: ext.inner.args_arr.as_mut_ptr(),
        name: s.matmul_kernel_name.as_ptr(),
        isDynamic: 0,
        tuningConfigs: ptr::null_mut(),
        reserved: ptr::null_mut(),
    };
    ext.inner.kernel_arr = vec![&mut *ext.inner.kernel as *mut _, ptr::null_mut()];
    *ext.inner.operation = GpuOperation {
        outputClaims: ext.inner.out_claim_arr.as_mut_ptr(),
        memoryObjects: ext.inner.mem_obj_arr.as_mut_ptr(),
        kernels: ext.inner.kernel_arr.as_mut_ptr(),
    };
    let op_ptr = &*ext.inner.operation as *const _ as *mut _QnnOpPackage_OpImpl_t;
    let _ = Box::into_raw(ext);
    unsafe { *op_impl = op_ptr };
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_free_op_impl(_op_impl: QnnOpPackage_OpImpl_t) -> Qnn_ErrorHandle_t {
    // PoC: state was leaked in createOpImpl. Real impl would reconstruct Box and drop.
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_log_initialize(
    _callback: QnnLog_Callback_t,
    _max_level: QnnLog_Level_t,
) -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_log_set_level(_max_level: QnnLog_Level_t) -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_log_terminate() -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

// ─────────────────────────────────────────────────────────
// Interface provider — symbol given to registerOpPackage(...interfaceProvider...)
// ─────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn QnnOpPackage_InitInterface(
    interface: *mut QnnOpPackage_Interface_t,
) -> Qnn_ErrorHandle_t {
    eprintln!("[oppkg] InitInterface called, interface={:?}", interface);
    if interface.is_null() {
        return QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT as Qnn_ErrorHandle_t;
    }
    let use_v2 = std::env::var("OPPKG_V2").is_ok();
    unsafe {
        if use_v2 {
            (*interface).interfaceVersion = Qnn_Version_t {
                major: 2,
                minor: 0,
                patch: 0,
            };
            (*interface).__bindgen_anon_1.v2_0 = QnnOpPackage_ImplementationV2_0_t {
                create: Some(pkg_create),
                getInfo: Some(pkg_get_info),
                validateOpConfig: Some(pkg_validate_op_config_h),
                createOpImpl: Some(pkg_create_op_impl_h),
                freeOpImpl: Some(pkg_free_op_impl_h),
                logSetLevel: Some(pkg_log_set_level_h),
                free: Some(pkg_free),
            };
            eprintln!("[oppkg] InitInterface returning SUCCESS (V2.0)");
        } else {
            (*interface).interfaceVersion = Qnn_Version_t {
                major: 1,
                minor: 4,
                patch: 0,
            };
            (*interface).__bindgen_anon_1.v1_4 = QnnOpPackage_ImplementationV1_4_t {
                init: Some(pkg_init),
                terminate: Some(pkg_terminate),
                getInfo: Some(pkg_get_info),
                validateOpConfig: Some(pkg_validate_op_config),
                createOpImpl: Some(pkg_create_op_impl),
                freeOpImpl: Some(pkg_free_op_impl),
                logInitialize: Some(pkg_log_initialize),
                logSetLevel: Some(pkg_log_set_level),
                logTerminate: Some(pkg_log_terminate),
            };
            eprintln!("[oppkg] InitInterface returning SUCCESS (V1.4)");
        }
    }
    QNN_SUCCESS as Qnn_ErrorHandle_t
}
