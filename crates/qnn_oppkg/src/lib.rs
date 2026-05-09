//! QNN GPU Op Package — production crate.
//!
//! Crate layout:
//!   - `args`        — backend-neutral op metadata / kernel arg types.
//!   - `static_info` — leaked static `QnnOpPackage_Info_t`.
//!   - `registry`    — `OPS` slice + `find_descriptor` (Open-Closed dispatcher).
//!   - `op_impl`     — `build_op_state`, `pkg_{create,free}_op_impl`.
//!   - `interface`   — V1.4 / V2.0 fn-pointer table + `QnnOpPackage_InitInterface`.

#![allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code
)]

#[allow(clippy::all)]
mod qnn {
    include!(concat!(env!("OUT_DIR"), "/qnn_bindings.rs"));
}

mod args;
mod interface;
mod op_impl;
mod ops;
mod registry;
mod static_info;

// Public surface: the single FFI entry, plus enough internals for host tests.
pub use interface::QnnOpPackage_InitInterface;

// Test-only / introspection re-exports. `cdylib` callers never use these but
// the integration tests in `tests/` need them.
pub use args::{ArgSpec, MemoryObjectSpec, OpDescriptor, OpError, OpImplLayout};
pub use registry::{OPS, find_descriptor};
pub use static_info::ensure_static;

// Conversion helper exposed for unit tests (verifies ArgSpec → KernelArg union).
#[doc(hidden)]
pub mod __test_support {
    use crate::args::ArgSpec;
    use crate::op_impl::{arg_spec_to_kernel_arg, build_op_state};

    pub use crate::op_impl::state_map_len;

    /// Wraps `pkg_free_op_impl` for host tests (null-safe smoke).
    pub fn call_pkg_free_op_impl(op_impl: *mut std::ffi::c_void) -> u64 {
        crate::op_impl::pkg_free_op_impl(op_impl as crate::qnn::QnnOpPackage_OpImpl_t)
    }

    pub fn arg_to_kernel_arg(spec: &ArgSpec) -> crate::qnn::QnnGpu_KernelArg_t {
        arg_spec_to_kernel_arg(spec)
    }

    /// Discriminator codes used by host tests to verify the C union without
    /// re-exporting bindgen-generated constants.
    pub mod kernel_arg_type {
        pub const OP_INPUT_READ: u32 =
            crate::qnn::QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ;
        pub const OP_INPUT_READWRITE: u32 =
            crate::qnn::QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READWRITE;
        pub const OP_OUTPUT_WRITE: u32 =
            crate::qnn::QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE;
        pub const DATA: u32 = crate::qnn::QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_DATA;
        pub const LOCAL: u32 = crate::qnn::QnnGpu_KernelArgType_t_QNN_GPU_KERNEL_ARG_TYPE_LOCAL;
    }

    /// Data type constants for `MemoryObjectSpec::data_type` comparisons in tests.
    pub mod data_type {
        pub const FLOAT_16: u32 = crate::qnn::Qnn_DataType_t_QNN_DATATYPE_FLOAT_16;
        pub const FLOAT_32: u32 = crate::qnn::Qnn_DataType_t_QNN_DATATYPE_FLOAT_32;
    }

    pub mod data_kernel_arg_type {
        pub const INT: u32 = crate::qnn::QnnGpu_DataKernelArgType_t_QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
        pub const UINT: u32 =
            crate::qnn::QnnGpu_DataKernelArgType_t_QNN_GPU_KERNEL_ARG_CL_TYPE_UINT;
        pub const ULONG: u32 =
            crate::qnn::QnnGpu_DataKernelArgType_t_QNN_GPU_KERNEL_ARG_CL_TYPE_ULONG;
        pub const FLOAT: u32 =
            crate::qnn::QnnGpu_DataKernelArgType_t_QNN_GPU_KERNEL_ARG_CL_TYPE_FLOAT;
    }

    /// Wraps `build_op_state` for host tests. Returns whether the leaked state
    /// pointer is non-null. The state itself is intentionally leaked.
    pub fn build_state_returns_non_null(
        layout: crate::args::OpImplLayout,
        kernel_name: &str,
        kernel_source: &str,
        build_options: &str,
    ) -> bool {
        match build_op_state(layout, kernel_name, kernel_source, build_options) {
            Ok(ptr) => !ptr.is_null(),
            Err(_) => false,
        }
    }

    /// Build an `OpImplState` for `CustomAdd` and return the raw pointer that
    /// QNN would receive from `createOpImpl`. The returned pointer is owned by
    /// the crate's `STATE_MAP` (M1.8); callers must drop it via
    /// `call_pkg_free_op_impl` to avoid leaking. Test-only helper for INV-155.
    pub fn raw_build_op_state_for_add(
        layout: crate::args::OpImplLayout,
    ) -> Result<*mut std::ffi::c_void, crate::args::OpError> {
        match build_op_state(layout, "kernel_add_assign_opt", "/* test */", "") {
            Ok(ptr) => Ok(ptr as *mut std::ffi::c_void),
            Err(e) => Err(e),
        }
    }

    /// Re-export of `pkg_get_info` for FFI tests.
    pub use crate::interface::pkg_get_info;

    pub use crate::qnn::QnnOpPackage_Info_t;

    /// Build an `OpImplLayout` for `CustomAdd` given a flat element count.
    /// Internally constructs a mock `Qnn_OpConfigV1_t` so callers do not need
    /// to depend on bindgen-generated types.
    ///
    /// `n_elems` must be a non-zero multiple of 4.
    pub fn build_add_layout_n(
        n_elems: u32,
    ) -> Result<crate::args::OpImplLayout, crate::args::OpError> {
        use crate::qnn::{
            QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, Qnn_ClientBuffer_t,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
            Qnn_OpConfigV1_t, Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
            Qnn_QuantizeParams_t, Qnn_QuantizeParams_t__bindgen_ty_1, Qnn_ScaleOffset_t,
            Qnn_Tensor_t, Qnn_Tensor_t__bindgen_ty_1, Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
            Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
            Qnn_TensorV1_t, Qnn_TensorV1_t__bindgen_ty_1, Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        };
        use std::ptr;

        let mut dims = vec![n_elems];
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
        let mut inputs = [
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, dims_ptr),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, dims_ptr),
                },
            },
        ];
        let mut outputs = [Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_tv1(Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, dims_ptr),
            },
        }];
        let v1 = Qnn_OpConfigV1_t {
            name: ptr::null(),
            packageName: ptr::null(),
            typeName: ptr::null(),
            numOfParams: 0,
            params: ptr::null_mut(),
            numOfInputs: 2,
            inputTensors: inputs.as_mut_ptr(),
            numOfOutputs: 1,
            outputTensors: outputs.as_mut_ptr(),
        };
        crate::ops::add::build_layout(&v1)
    }

    /// Build an `OpImplLayout` for `CustomMatMulF16F32` given (M, N, K).
    /// Mocks a `Qnn_OpConfigV1_t` with weight[N,K] (FLOAT_16), x[M,K] (FLOAT_32),
    /// y[M,N] (FLOAT_32).
    pub fn build_matmul_layout_for(
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<crate::args::OpImplLayout, crate::args::OpError> {
        use crate::qnn::{
            QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, Qnn_ClientBuffer_t, Qnn_DataType_t,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            Qnn_Definition_t_QNN_DEFINITION_UNDEFINED, Qnn_OpConfigV1_t,
            Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED, Qnn_QuantizeParams_t,
            Qnn_QuantizeParams_t__bindgen_ty_1, Qnn_ScaleOffset_t, Qnn_Tensor_t,
            Qnn_Tensor_t__bindgen_ty_1, Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
            Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
            Qnn_TensorV1_t, Qnn_TensorV1_t__bindgen_ty_1, Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        };
        use std::ptr;

        let mut dims_w = vec![n, k];
        let mut dims_x = vec![m, k];
        let mut dims_y = vec![m, n];
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
        let mk_tv1 = |ttype, dtype: Qnn_DataType_t, dims_ptr: *mut u32| Qnn_TensorV1_t {
            id: 0,
            name: ptr::null(),
            type_: ttype,
            dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            dataType: dtype,
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
        let mut inputs = [
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
                        dims_w.as_mut_ptr(),
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                        dims_x.as_mut_ptr(),
                    ),
                },
            },
        ];
        let mut outputs = [Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_tv1(
                    Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
                    Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                    dims_y.as_mut_ptr(),
                ),
            },
        }];
        let v1 = Qnn_OpConfigV1_t {
            name: ptr::null(),
            packageName: ptr::null(),
            typeName: ptr::null(),
            numOfParams: 0,
            params: ptr::null_mut(),
            numOfInputs: 2,
            inputTensors: inputs.as_mut_ptr(),
            numOfOutputs: 1,
            outputTensors: outputs.as_mut_ptr(),
        };
        crate::ops::matmul_f16_f32::build_layout(&v1)
    }

    /// Build an `OpImplLayout` for `CustomRmsNorm` given (rows, dim).
    /// Mocks a `Qnn_OpConfigV1_t` with x[rows, dim], weight[dim], out[rows, dim],
    /// all FLOAT_32. `rows == 1` exercises the rank-1 fallback path.
    pub fn build_rmsnorm_layout_for(
        rows: u32,
        dim: u32,
    ) -> Result<crate::args::OpImplLayout, crate::args::OpError> {
        use crate::qnn::{
            QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, Qnn_ClientBuffer_t,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
            Qnn_OpConfigV1_t, Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
            Qnn_QuantizeParams_t, Qnn_QuantizeParams_t__bindgen_ty_1, Qnn_ScaleOffset_t,
            Qnn_Tensor_t, Qnn_Tensor_t__bindgen_ty_1, Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
            Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
            Qnn_TensorV1_t, Qnn_TensorV1_t__bindgen_ty_1, Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        };
        use std::ptr;

        let mut dims_x = vec![rows, dim];
        let mut dims_w = vec![1u32, dim];
        let mut dims_y = vec![rows, dim];
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
        let mk_tv1 = |ttype, rank: u32, dims_ptr: *mut u32| Qnn_TensorV1_t {
            id: 0,
            name: ptr::null(),
            type_: ttype,
            dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
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
        let mut inputs = [
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        2,
                        dims_x.as_mut_ptr(),
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        2,
                        dims_w.as_mut_ptr(),
                    ),
                },
            },
        ];
        let mut outputs = [Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_tv1(
                    Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
                    2,
                    dims_y.as_mut_ptr(),
                ),
            },
        }];
        let v1 = Qnn_OpConfigV1_t {
            name: ptr::null(),
            packageName: ptr::null(),
            typeName: ptr::null(),
            numOfParams: 0,
            params: ptr::null_mut(),
            numOfInputs: 2,
            inputTensors: inputs.as_mut_ptr(),
            numOfOutputs: 1,
            outputTensors: outputs.as_mut_ptr(),
        };
        crate::ops::rms_norm::build_layout(&v1)
    }

    /// Build an `OpImplLayout` for `CustomSoftmax` given (rows, dim).
    /// Mocks a `Qnn_OpConfigV1_t` with x[rows, dim] and out[rows, dim], both
    /// FLOAT_32. `rows == 1` exercises the rank-1 fallback path.
    pub fn build_softmax_layout_for(
        rows: u32,
        dim: u32,
    ) -> Result<crate::args::OpImplLayout, crate::args::OpError> {
        use crate::qnn::{
            QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, Qnn_ClientBuffer_t,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
            Qnn_OpConfigV1_t, Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
            Qnn_QuantizeParams_t, Qnn_QuantizeParams_t__bindgen_ty_1, Qnn_ScaleOffset_t,
            Qnn_Tensor_t, Qnn_Tensor_t__bindgen_ty_1, Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
            Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
            Qnn_TensorV1_t, Qnn_TensorV1_t__bindgen_ty_1, Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        };
        use std::ptr;

        let mut dims_x = vec![rows, dim];
        let mut dims_y = vec![rows, dim];
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
        let mut inputs = [Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_tv1(
                    Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                    dims_x.as_mut_ptr(),
                ),
            },
        }];
        let mut outputs = [Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_tv1(
                    Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
                    dims_y.as_mut_ptr(),
                ),
            },
        }];
        let v1 = Qnn_OpConfigV1_t {
            name: ptr::null(),
            packageName: ptr::null(),
            typeName: ptr::null(),
            numOfParams: 0,
            params: ptr::null_mut(),
            numOfInputs: 1,
            inputTensors: inputs.as_mut_ptr(),
            numOfOutputs: 1,
            outputTensors: outputs.as_mut_ptr(),
        };
        crate::ops::softmax::build_layout(&v1)
    }

    /// Build an `OpImplLayout` for `CustomSiluMul` given (rows, dim).
    /// Mocks a `Qnn_OpConfigV1_t` with x[rows, dim], y[rows, dim], and
    /// x_inout[rows, dim] (graph-level alias of x). All FLOAT_32. The kernel
    /// is float4-vectorised; `rows * dim` must be a multiple of 4.
    pub fn build_silumul_layout_for(
        rows: u32,
        dim: u32,
    ) -> Result<crate::args::OpImplLayout, crate::args::OpError> {
        use crate::qnn::{
            QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, Qnn_ClientBuffer_t,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
            Qnn_OpConfigV1_t, Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
            Qnn_QuantizeParams_t, Qnn_QuantizeParams_t__bindgen_ty_1, Qnn_ScaleOffset_t,
            Qnn_Tensor_t, Qnn_Tensor_t__bindgen_ty_1, Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
            Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
            Qnn_TensorV1_t, Qnn_TensorV1_t__bindgen_ty_1, Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        };
        use std::ptr;

        let mut dims_x = vec![rows, dim];
        let mut dims_y = vec![rows, dim];
        let mut dims_xout = vec![rows, dim];
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
        let mut inputs = [
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        dims_x.as_mut_ptr(),
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        dims_y.as_mut_ptr(),
                    ),
                },
            },
        ];
        let mut outputs = [Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_tv1(
                    Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
                    dims_xout.as_mut_ptr(),
                ),
            },
        }];
        let v1 = Qnn_OpConfigV1_t {
            name: ptr::null(),
            packageName: ptr::null(),
            typeName: ptr::null(),
            numOfParams: 0,
            params: ptr::null_mut(),
            numOfInputs: 2,
            inputTensors: inputs.as_mut_ptr(),
            numOfOutputs: 1,
            outputTensors: outputs.as_mut_ptr(),
        };
        crate::ops::silu_mul::build_layout(&v1)
    }
}
