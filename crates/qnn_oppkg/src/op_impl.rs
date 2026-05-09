//! Op implementation builder.
//!
//! `build_op_state` consumes an `OpImplLayout` (produced by the descriptor's
//! `build_layout` callback) and constructs a leaked `OpImplState` whose
//! `GpuOperation` raw pointer is what QNN expects in `Qnn_OpPackage_OpImpl_t`.
//!
//! Memory model: every backing allocation lives inside a `Box` owned by the
//! state. The state itself is `Box::into_raw`'d on success — M1.2 keeps
//! `pkg_free_op_impl` as a no-op; M1.8 will reconstruct the box and drop it.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::args::{ArgSpec, OpError, OpImplLayout};
use crate::qnn::{
    _QnnOpPackage_OpImpl_t, QNN_SUCCESS, Qnn_ErrorHandle_t, QnnGpu_BlockEncodingInfo_t,
    QnnGpu_DataKernelArg_t, QnnGpu_DataKernelArg_t__bindgen_ty_1,
    QnnGpu_DataKernelArgType_t_QNN_GPU_KERNEL_ARG_CL_TYPE_FLOAT,
    QnnGpu_DataKernelArgType_t_QNN_GPU_KERNEL_ARG_CL_TYPE_INT,
    QnnGpu_DataKernelArgType_t_QNN_GPU_KERNEL_ARG_CL_TYPE_UINT,
    QnnGpu_DataKernelArgType_t_QNN_GPU_KERNEL_ARG_CL_TYPE_ULONG, QnnGpu_Kernel_t,
    QnnGpu_KernelArg_t, QnnGpu_KernelArg_t__bindgen_ty_1,
    QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_DATA,
    QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_LOCAL,
    QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ,
    QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READWRITE,
    QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE,
    QnnGpu_KernelSourceType_t_QNN_GPU_KERNEL_SOURCE_TYPE_TEXT, QnnGpu_LocalKernelArg_t,
    QnnGpu_MemoryLayout_t_QNN_GPU_MEM_LAYOUT_UNDEFINED, QnnGpu_MemoryObject_t,
    QnnGpu_MemoryObjectType_t_QNN_GPU_MEM_OBJ_TYPE_BUFFER, QnnGpu_OutputClaim_t,
    QnnGpu_TensorKernelArg_t, QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT,
    QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE, QnnOpPackage_Node_t,
    QnnOpPackage_OpImpl_t,
};
use crate::registry::find_descriptor;
use crate::static_info::ensure_static;
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;
use std::sync::{LazyLock, Mutex};

// bindgen leaves `_QnnOpPackage_Node_t` / `_QnnOpPackage_OpImpl_t` as opaque forward
// declarations because QnnInterface.h is processed first. Re-declare the GPU
// specialisations here. Mirrors PoC layout.
#[repr(C)]
pub struct GpuNode {
    pub optimization: u32,
    pub configs: *mut *const crate::qnn::Qnn_OpConfig_t,
    pub storageTypes: *mut *const crate::qnn::QnnGpu_TensorStorageType_t,
    pub kernelVariant: i32,
}
#[repr(C)]
pub struct GpuOperation {
    pub outputClaims: *mut *mut QnnGpu_OutputClaim_t,
    pub memoryObjects: *mut *mut QnnGpu_MemoryObject_t,
    pub kernels: *mut *mut QnnGpu_Kernel_t,
}

/// Reverse-mapping table for `pkg_free_op_impl` (M1.8, INV-155).
///
/// `build_op_state` returns a raw `*mut _QnnOpPackage_OpImpl_t` to QNN — that
/// pointer aliases the interior of the `Box<OpImplState>::operation` field.
/// We also store the owning `Box<OpImplState>` here keyed by the same raw
/// pointer (cast to `usize`). When QNN later calls `pkg_free_op_impl` we look
/// the box up by pointer and drop it.
///
/// Concurrency: QNN may call `pkg_create_op_impl` / `pkg_free_op_impl` from
/// multiple threads. Mutex<HashMap> serialises both. Each call is O(1) +
/// graphFinalize/graphFree only invokes them once per op, so contention is
/// negligible. `Mutex::new` and `HashMap::new` are both `const fn`, so a
/// `static` is sufficient on stable Rust.
static STATE_MAP: LazyLock<Mutex<HashMap<usize, Box<OpImplState>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Free the state owned by `op_impl` (looked up in `STATE_MAP`). Returns
/// `true` if a state was found and dropped, `false` if not present (already
/// freed / unknown pointer / null). Exposed for unit tests.
pub(crate) fn free_op_impl_state(op_impl: *mut _QnnOpPackage_OpImpl_t) -> bool {
    if op_impl.is_null() {
        return false;
    }
    let key = op_impl as usize;
    let state = {
        let mut map = STATE_MAP.lock().unwrap();
        map.remove(&key)
    };
    state.is_some()
}

/// Number of live entries in `STATE_MAP`. Test helper for INV-155.
#[doc(hidden)]
pub fn state_map_len() -> usize {
    STATE_MAP.lock().unwrap().len()
}

/// All backing storage for one op invocation. Owned by a leaked Box; held alive
/// for as long as QNN may dereference the resulting `GpuOperation`.
///
/// `Vec<Box<…>>` is intentional: we hand QNN raw pointers to the inner elements
/// and `Vec` would invalidate them on growth. The Box gives each element a
/// stable address.
#[allow(clippy::vec_box)]
pub(crate) struct OpImplState {
    // SAFETY (Send/Sync impls below): every raw pointer in this struct aliases
    // an interior `Box<…>` allocation owned by the same `OpImplState`. The Box
    // itself is owned exclusively by `STATE_MAP` (M1.8) and only accessed
    // under its `Mutex`; the QNN runtime treats the returned `GpuOperation*`
    // as opaque between `createOpImpl` and `freeOpImpl`. No interior
    // mutability is exposed across threads outside the mutex.
    //
    // We need both `Send` (so `STATE_MAP` can hold the Box across threads)
    // and `Sync` (so the `static LazyLock<Mutex<…>>` itself is `Sync`).
    pub out_claim: Box<QnnGpu_OutputClaim_t>,
    pub out_claim_arr: Vec<*mut QnnGpu_OutputClaim_t>,
    pub mem_objs: Vec<Box<QnnGpu_MemoryObject_t>>,
    pub mem_obj_arr: Vec<*mut QnnGpu_MemoryObject_t>,
    pub args: Vec<Box<QnnGpu_KernelArg_t>>,
    pub args_arr: Vec<*mut QnnGpu_KernelArg_t>,
    pub kernel: Box<QnnGpu_Kernel_t>,
    pub kernel_arr: Vec<*mut QnnGpu_Kernel_t>,
    pub operation: Box<GpuOperation>,
    // Per mem object, owned dimension/offset arrays. Indexed by mem object position.
    pub dim_storage: Vec<Vec<u32>>,
    pub off_storage: Vec<Vec<u32>>,
    // Owned C strings for kernel name / source / build options.
    pub kernel_name: CString,
    pub kernel_source: CString,
    pub build_options: CString,
}

// SAFETY: see comment on `OpImplState`. The struct is self-referential — all
// raw pointers alias interior Box-owned allocations of the same instance —
// and STATE_MAP is the only owner, gated by a Mutex.
unsafe impl Send for OpImplState {}
unsafe impl Sync for OpImplState {}

/// Build an `OpImplState` from `(layout, kernel_*)`, register it in
/// `STATE_MAP`, and return the raw `GpuOperation` pointer cast to
/// `*mut _QnnOpPackage_OpImpl_t` (what QNN stores in the out-param of
/// `createOpImpl`).
///
/// Ownership: the `Box<OpImplState>` is owned by `STATE_MAP`. The returned
/// pointer aliases the interior `Box<GpuOperation>` and stays valid until
/// `pkg_free_op_impl` (or `free_op_impl_state`) removes the entry. M1.8
/// (INV-155) replaces the M1.2 leak pattern with this lookup-on-free design.
pub(crate) fn build_op_state(
    layout: OpImplLayout,
    kernel_name: &str,
    kernel_source: &str,
    build_options: &str,
) -> Result<*mut _QnnOpPackage_OpImpl_t, OpError> {
    if layout.mem_objects.is_empty() {
        return Err(OpError::ValidationFailure);
    }

    let kernel_name = CString::new(kernel_name).map_err(|_| OpError::InvalidArgument)?;
    let kernel_source = CString::new(kernel_source).map_err(|_| OpError::InvalidArgument)?;
    let build_options = CString::new(build_options).map_err(|_| OpError::InvalidArgument)?;

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
        dim_storage: Vec::new(),
        off_storage: Vec::new(),
        kernel_name,
        kernel_source,
        build_options,
    });

    // Persist dim/offset arrays per mem object, then build QnnGpu_MemoryObject_t.
    for spec in layout.mem_objects.iter() {
        state.dim_storage.push(spec.flat_dims.clone());
        state.off_storage.push(spec.flat_offsets.clone());
    }
    let num_dims_per_obj: Vec<u32> = layout
        .mem_objects
        .iter()
        .map(|s| s.flat_dims.len() as u32)
        .collect();
    let data_types: Vec<_> = layout.mem_objects.iter().map(|s| s.data_type).collect();

    for (i, _spec) in layout.mem_objects.iter().enumerate() {
        let dims_ptr = state.dim_storage[i].as_mut_ptr();
        let offs_ptr = state.off_storage[i].as_mut_ptr();
        let mo = QnnGpu_MemoryObject_t {
            type_: QnnGpu_MemoryObjectType_t_QNN_GPU_MEM_OBJ_TYPE_BUFFER,
            dataType: data_types[i],
            dimensions: dims_ptr,
            offsets: offs_ptr,
            numDimensions: num_dims_per_obj[i],
            layout: QnnGpu_MemoryLayout_t_QNN_GPU_MEM_LAYOUT_UNDEFINED,
            blockEncodingInfo: QnnGpu_BlockEncodingInfo_t {
                bqBlockSize: ptr::null_mut(),
                bqEncodingTensorId: 0,
            },
        };
        state.mem_objs.push(Box::new(mo));
    }

    // Output is the last mem object. Claim it.
    let out_idx = state.mem_objs.len() - 1;
    state.out_claim.memoryObject = &*state.mem_objs[out_idx] as *const _;
    state.out_claim_arr = vec![&mut *state.out_claim as *mut _, ptr::null_mut()];
    state.mem_obj_arr = state
        .mem_objs
        .iter_mut()
        .map(|m| &mut **m as *mut _)
        .collect();
    state.mem_obj_arr.push(ptr::null_mut());

    // Convert ArgSpec sequence to QnnGpu_KernelArg_t sequence.
    for spec in layout.args.iter() {
        state.args.push(Box::new(arg_spec_to_kernel_arg(spec)));
    }
    state.args_arr = state.args.iter_mut().map(|a| &mut **a as *mut _).collect();
    state.args_arr.push(ptr::null_mut());

    // Fill the kernel struct. `sourceLength` excludes the null terminator.
    let src_bytes = state.kernel_source.as_bytes_with_nul();
    *state.kernel = QnnGpu_Kernel_t {
        kernelSource: src_bytes.as_ptr() as *const c_void,
        sourceLength: src_bytes.len() - 1,
        sourceType: QnnGpu_KernelSourceType_t_QNN_GPU_KERNEL_SOURCE_TYPE_TEXT,
        buildOptions: state.build_options.as_ptr(),
        globalWorkDim: layout.global_work_dim as usize,
        globalWorkSizes: layout.global_work,
        localWorkDim: layout.local_work_dim as usize,
        localWorkSizes: layout.local_work,
        args: state.args_arr.as_mut_ptr(),
        name: state.kernel_name.as_ptr(),
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
    // M1.8 (INV-155): hand ownership to STATE_MAP keyed by `op_ptr`. Drops on
    // `pkg_free_op_impl` instead of leaking via `Box::into_raw`.
    let key = op_ptr as usize;
    STATE_MAP.lock().unwrap().insert(key, state);
    Ok(op_ptr)
}

/// Pure conversion from `ArgSpec` to the C union `QnnGpu_KernelArg_t`. Exposed
/// for unit tests.
pub(crate) fn arg_spec_to_kernel_arg(spec: &ArgSpec) -> QnnGpu_KernelArg_t {
    match *spec {
        ArgSpec::InputTensor(ti) => QnnGpu_KernelArg_t {
            type_: QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ,
            __bindgen_anon_1: QnnGpu_KernelArg_t__bindgen_ty_1 {
                tensor: QnnGpu_TensorKernelArg_t {
                    opConfigIndex: 0,
                    tensorIndex: ti,
                    element: 0,
                },
            },
        },
        ArgSpec::OutputTensor(ti) => QnnGpu_KernelArg_t {
            type_: QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE,
            __bindgen_anon_1: QnnGpu_KernelArg_t__bindgen_ty_1 {
                tensor: QnnGpu_TensorKernelArg_t {
                    opConfigIndex: 0,
                    tensorIndex: ti,
                    element: 0,
                },
            },
        },
        ArgSpec::InOutTensor(ti) => QnnGpu_KernelArg_t {
            type_: QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READWRITE,
            __bindgen_anon_1: QnnGpu_KernelArg_t__bindgen_ty_1 {
                tensor: QnnGpu_TensorKernelArg_t {
                    opConfigIndex: 0,
                    tensorIndex: ti,
                    element: 0,
                },
            },
        },
        ArgSpec::LocalMem(bytes) => QnnGpu_KernelArg_t {
            type_: QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_LOCAL,
            __bindgen_anon_1: QnnGpu_KernelArg_t__bindgen_ty_1 {
                local: QnnGpu_LocalKernelArg_t { size: bytes as u32 },
            },
        },
        ArgSpec::Int(v) => QnnGpu_KernelArg_t {
            type_: QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_DATA,
            __bindgen_anon_1: QnnGpu_KernelArg_t__bindgen_ty_1 {
                data: QnnGpu_DataKernelArg_t {
                    type_: QnnGpu_DataKernelArgType_t_QNN_GPU_KERNEL_ARG_CL_TYPE_INT,
                    __bindgen_anon_1: QnnGpu_DataKernelArg_t__bindgen_ty_1 { qnnInt: v },
                },
            },
        },
        ArgSpec::UInt(v) => QnnGpu_KernelArg_t {
            type_: QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_DATA,
            __bindgen_anon_1: QnnGpu_KernelArg_t__bindgen_ty_1 {
                data: QnnGpu_DataKernelArg_t {
                    type_: QnnGpu_DataKernelArgType_t_QNN_GPU_KERNEL_ARG_CL_TYPE_UINT,
                    __bindgen_anon_1: QnnGpu_DataKernelArg_t__bindgen_ty_1 { qnnUInt: v },
                },
            },
        },
        ArgSpec::ULong(v) => QnnGpu_KernelArg_t {
            type_: QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_DATA,
            __bindgen_anon_1: QnnGpu_KernelArg_t__bindgen_ty_1 {
                data: QnnGpu_DataKernelArg_t {
                    type_: QnnGpu_DataKernelArgType_t_QNN_GPU_KERNEL_ARG_CL_TYPE_ULONG,
                    __bindgen_anon_1: QnnGpu_DataKernelArg_t__bindgen_ty_1 { qnnULong: v },
                },
            },
        },
        ArgSpec::Float(v) => QnnGpu_KernelArg_t {
            type_: QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_DATA,
            __bindgen_anon_1: QnnGpu_KernelArg_t__bindgen_ty_1 {
                data: QnnGpu_DataKernelArg_t {
                    type_: QnnGpu_DataKernelArgType_t_QNN_GPU_KERNEL_ARG_CL_TYPE_FLOAT,
                    __bindgen_anon_1: QnnGpu_DataKernelArg_t__bindgen_ty_1 { qnnFloat: v },
                },
            },
        },
    }
}

/// `createOpImpl` body. Open-Closed dispatch: looks the descriptor up in OPS by
/// op_type and delegates to the descriptor's `build_layout` (ENG-QNN-024).
pub extern "C" fn pkg_create_op_impl(
    _graph_infra: crate::qnn::QnnOpPackage_GraphInfrastructure_t,
    node: QnnOpPackage_Node_t,
    op_impl: *mut QnnOpPackage_OpImpl_t,
) -> Qnn_ErrorHandle_t {
    if node.is_null() || op_impl.is_null() {
        return QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT as Qnn_ErrorHandle_t;
    }
    ensure_static();

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

    let op_type_str = unsafe { std::ffi::CStr::from_ptr(v1.typeName).to_str().unwrap_or("") };
    let descriptor = match find_descriptor(op_type_str) {
        Some(d) => d,
        None => {
            return QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE
                as Qnn_ErrorHandle_t;
        }
    };

    let layout = match (descriptor.build_layout)(&v1) {
        Ok(l) => l,
        Err(OpError::InvalidArgument) => {
            return QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT as Qnn_ErrorHandle_t;
        }
        Err(OpError::ValidationFailure) => {
            return QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE
                as Qnn_ErrorHandle_t;
        }
    };

    match build_op_state(
        layout,
        descriptor.kernel_name,
        descriptor.kernel_source,
        descriptor.build_options,
    ) {
        Ok(op_ptr) => {
            unsafe { *op_impl = op_ptr };
            QNN_SUCCESS as Qnn_ErrorHandle_t
        }
        Err(OpError::InvalidArgument) => {
            QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT as Qnn_ErrorHandle_t
        }
        Err(OpError::ValidationFailure) => {
            QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE as Qnn_ErrorHandle_t
        }
    }
}

/// `freeOpImpl`. M1.8 (INV-155): looks `op_impl` up in `STATE_MAP`, removes
/// the entry, and drops the owning `Box<OpImplState>`.
///
/// Returns `QNN_SUCCESS` on success or when `op_impl` is null / unknown
/// (idempotent — QNN runtime spec is silent on double-free, and PoC behaviour
/// was no-op).
pub extern "C" fn pkg_free_op_impl(op_impl: QnnOpPackage_OpImpl_t) -> Qnn_ErrorHandle_t {
    let _ = free_op_impl_state(op_impl);
    QNN_SUCCESS as Qnn_ErrorHandle_t
}
