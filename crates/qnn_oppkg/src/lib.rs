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
pub mod graph;
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
        pub const UINT_8: u32 = crate::qnn::Qnn_DataType_t_QNN_DATATYPE_UINT_8;
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

    /// Build an `OpImplLayout` for `CustomRope` given (seq_len, num_heads,
    /// head_dim). Mocks a `Qnn_OpConfigV1_t` with rank-3 input
    /// `x[seq_len, num_heads, head_dim]` and an alias output tensor of the
    /// same shape (M1.7 in-place pattern). Both FLOAT_32. `numOfParams = 0`
    /// so `start_pos` defaults to 0 and `theta` defaults to 10000.0.
    pub fn build_rope_layout_for(
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
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

        let mut dims_x = vec![seq_len, num_heads, head_dim];
        let mut dims_xout = vec![seq_len, num_heads, head_dim];
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
            rank: 3,
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
            numOfInputs: 1,
            inputTensors: inputs.as_mut_ptr(),
            numOfOutputs: 1,
            outputTensors: outputs.as_mut_ptr(),
        };
        crate::ops::rope::build_layout(&v1)
    }

    /// Build an `OpImplLayout` for `CustomDeqQ40` given a block count.
    ///
    /// Mocks a `Qnn_OpConfigV1_t` with:
    ///   - 1 input: src0 UINT_8, flat dim = [num_blocks * 18]
    ///   - 2 outputs: dst_q UINT_8 [num_blocks * 16], dst_d FLOAT_16 [num_blocks]
    pub fn build_deq_q40_layout_for(
        num_blocks: u32,
    ) -> Result<crate::args::OpImplLayout, crate::args::OpError> {
        use crate::qnn::Qnn_DataType_t;
        use crate::qnn::{
            QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, Qnn_ClientBuffer_t,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, Qnn_DataType_t_QNN_DATATYPE_UINT_8,
            Qnn_Definition_t_QNN_DEFINITION_UNDEFINED, Qnn_OpConfigV1_t,
            Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED, Qnn_QuantizeParams_t,
            Qnn_QuantizeParams_t__bindgen_ty_1, Qnn_ScaleOffset_t, Qnn_Tensor_t,
            Qnn_Tensor_t__bindgen_ty_1, Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
            Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
            Qnn_TensorV1_t, Qnn_TensorV1_t__bindgen_ty_1, Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        };
        use std::ptr;

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
        let mk_tv1 = |ttype, dtype: Qnn_DataType_t, dims_ptr: *mut u32, rank: u32| Qnn_TensorV1_t {
            id: 0,
            name: ptr::null(),
            type_: ttype,
            dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            dataType: dtype,
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

        // src0: UINT_8, [num_blocks * 18]
        let mut dims_src0 = vec![num_blocks * 18];
        // dst_q: UINT_8, [num_blocks * 16]
        let mut dims_dst_q = vec![num_blocks * 16];
        // dst_d: FLOAT_16, [num_blocks]
        let mut dims_dst_d = vec![num_blocks];

        let mut inputs = [Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_tv1(
                    Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                    Qnn_DataType_t_QNN_DATATYPE_UINT_8,
                    dims_src0.as_mut_ptr(),
                    1,
                ),
            },
        }];
        let mut outputs = [
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
                        Qnn_DataType_t_QNN_DATATYPE_UINT_8,
                        dims_dst_q.as_mut_ptr(),
                        1,
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
                        dims_dst_d.as_mut_ptr(),
                        1,
                    ),
                },
            },
        ];
        let v1 = Qnn_OpConfigV1_t {
            name: ptr::null(),
            packageName: ptr::null(),
            typeName: ptr::null(),
            numOfParams: 0,
            params: ptr::null_mut(),
            numOfInputs: 1,
            inputTensors: inputs.as_mut_ptr(),
            numOfOutputs: 2,
            outputTensors: outputs.as_mut_ptr(),
        };
        crate::ops::deq_q40::build_layout(&v1)
    }

    /// Build an `OpImplLayout` for `CustomKvScatter` given
    /// `(kv_heads, head_dim, capacity, write_pos)`.
    ///
    /// Mocks a `Qnn_OpConfigV1_t` with (multi-output, M2.H):
    ///   - inputs[0]: k_src FLOAT_32 [kv_heads * head_dim]
    ///   - inputs[1]: v_src FLOAT_32 [kv_heads * head_dim]
    ///   - outputs[0]: k_dst FLOAT_16 [kv_heads * capacity * head_dim]
    ///   - outputs[1]: v_dst FLOAT_16 [kv_heads * capacity * head_dim]
    ///
    /// Params: head_dim, capacity, write_pos (all INT_32).
    pub fn build_kv_scatter_layout_for(
        kv_heads: u32,
        head_dim: i32,
        capacity: i32,
        write_pos: i32,
    ) -> Result<crate::args::OpImplLayout, crate::args::OpError> {
        use crate::qnn::{
            QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, Qnn_ClientBuffer_t, Qnn_DataType_t,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            Qnn_DataType_t_QNN_DATATYPE_INT_32, Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
            Qnn_OpConfigV1_t, Qnn_Param_t, Qnn_Param_t__bindgen_ty_1,
            Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED, Qnn_QuantizeParams_t,
            Qnn_QuantizeParams_t__bindgen_ty_1, Qnn_Scalar_t, Qnn_Scalar_t__bindgen_ty_1,
            Qnn_ScaleOffset_t, Qnn_Tensor_t, Qnn_Tensor_t__bindgen_ty_1,
            Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
            Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, Qnn_TensorV1_t,
            Qnn_TensorV1_t__bindgen_ty_1, Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        };
        use std::ffi::CString;
        use std::ptr;

        let src_total = kv_heads * (head_dim as u32);
        let dst_total = kv_heads * (capacity as u32) * (head_dim as u32);

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
        let mk_tv1 = |ttype, dtype: Qnn_DataType_t, dims_ptr: *mut u32, rank: u32| Qnn_TensorV1_t {
            id: 0,
            name: ptr::null(),
            type_: ttype,
            dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            dataType: dtype,
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

        let mut dims_src = vec![src_total];
        let mut dims_dst = vec![dst_total];
        let mut dims_dst2 = vec![dst_total];

        let mut inputs = [
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                        dims_src.as_mut_ptr(),
                        1,
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                        dims_src.as_mut_ptr(),
                        1,
                    ),
                },
            },
        ];
        let mut outputs = [
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
                        dims_dst.as_mut_ptr(),
                        1,
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
                        dims_dst2.as_mut_ptr(),
                        1,
                    ),
                },
            },
        ];

        // Build params for head_dim, capacity, write_pos.
        let name_head_dim = CString::new("head_dim").unwrap();
        let name_capacity = CString::new("capacity").unwrap();
        let name_write_pos = CString::new("write_pos").unwrap();
        let mut params = [
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: name_head_dim.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: head_dim,
                        },
                    },
                },
            },
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: name_capacity.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: capacity,
                        },
                    },
                },
            },
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: name_write_pos.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: write_pos,
                        },
                    },
                },
            },
        ];

        let v1 = Qnn_OpConfigV1_t {
            name: ptr::null(),
            packageName: ptr::null(),
            typeName: ptr::null(),
            numOfParams: 3,
            params: params.as_mut_ptr(),
            numOfInputs: 2,
            inputTensors: inputs.as_mut_ptr(),
            numOfOutputs: 2,
            outputTensors: outputs.as_mut_ptr(),
        };
        crate::ops::kv_scatter::build_layout(&v1)
    }

    /// Build an `OpImplLayout` for `CustomMatMulQ40F32` given (M, N, K).
    /// Mocks a `Qnn_OpConfigV1_t` with src0_q UINT_8 [num_blocks*16],
    /// src0_d FLOAT_16 [num_blocks], src1 FLOAT_32 [M, K] and
    /// dst FLOAT_32 [M, N], where `num_blocks = N * K / 32`.
    pub fn build_matmul_q40_layout_for(
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<crate::args::OpImplLayout, crate::args::OpError> {
        use crate::qnn::{
            QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, Qnn_ClientBuffer_t, Qnn_DataType_t,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            Qnn_DataType_t_QNN_DATATYPE_UINT_8, Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
            Qnn_OpConfigV1_t, Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
            Qnn_QuantizeParams_t, Qnn_QuantizeParams_t__bindgen_ty_1, Qnn_ScaleOffset_t,
            Qnn_Tensor_t, Qnn_Tensor_t__bindgen_ty_1, Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
            Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
            Qnn_TensorV1_t, Qnn_TensorV1_t__bindgen_ty_1, Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        };
        use std::ptr;

        if k == 0 || n == 0 || m == 0 || !k.is_multiple_of(32) {
            return Err(crate::args::OpError::InvalidArgument);
        }
        let num_blocks = n
            .checked_mul(k)
            .map(|nk| nk / 32)
            .ok_or(crate::args::OpError::InvalidArgument)?;

        let mut dims_q = vec![num_blocks * 16];
        let mut dims_d = vec![num_blocks];
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
        let mk_tv1 = |ttype, dtype: Qnn_DataType_t, rank: u32, dims_ptr: *mut u32| Qnn_TensorV1_t {
            id: 0,
            name: ptr::null(),
            type_: ttype,
            dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            dataType: dtype,
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
                        Qnn_DataType_t_QNN_DATATYPE_UINT_8,
                        1,
                        dims_q.as_mut_ptr(),
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
                        1,
                        dims_d.as_mut_ptr(),
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                        2,
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
            numOfInputs: 3,
            inputTensors: inputs.as_mut_ptr(),
            numOfOutputs: 1,
            outputTensors: outputs.as_mut_ptr(),
        };
        crate::ops::matmul_q40_f32::build_layout(&v1)
    }

    /// Build an `OpImplLayout` for `CustomMatMulQ40F32` with rank-flexible
    /// input/output tensors. Used by M2.H 3rd-attempt unit tests that exercise
    /// the Qwen layer reshape view (matmul output flows directly into rank-3
    /// RoPE / FlashAttn inputs).
    ///
    /// `in_dims` and `out_dims` may be rank 2 or rank 3. Implementation derives
    /// `M = in_dims[0]`, `K = product(in_dims[1..])`, `N = product(out_dims[1..])`
    /// — matching production `build_layout`.
    pub fn build_matmul_q40_layout_with_dims(
        in_dims: &[u32],
        out_dims: &[u32],
    ) -> Result<crate::args::OpImplLayout, crate::args::OpError> {
        use crate::qnn::{
            QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, Qnn_ClientBuffer_t, Qnn_DataType_t,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            Qnn_DataType_t_QNN_DATATYPE_UINT_8, Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
            Qnn_OpConfigV1_t, Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
            Qnn_QuantizeParams_t, Qnn_QuantizeParams_t__bindgen_ty_1, Qnn_ScaleOffset_t,
            Qnn_Tensor_t, Qnn_Tensor_t__bindgen_ty_1, Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
            Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
            Qnn_TensorV1_t, Qnn_TensorV1_t__bindgen_ty_1, Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        };
        use std::ptr;

        if in_dims.len() < 2 || out_dims.len() < 2 {
            return Err(crate::args::OpError::InvalidArgument);
        }
        let m = in_dims[0];
        let k: u32 = in_dims[1..]
            .iter()
            .copied()
            .try_fold(1u32, |a, d| a.checked_mul(d))
            .ok_or(crate::args::OpError::InvalidArgument)?;
        let n: u32 = out_dims[1..]
            .iter()
            .copied()
            .try_fold(1u32, |a, d| a.checked_mul(d))
            .ok_or(crate::args::OpError::InvalidArgument)?;

        if m == 0 || n == 0 || k == 0 || !k.is_multiple_of(32) || in_dims[0] != out_dims[0] {
            return Err(crate::args::OpError::InvalidArgument);
        }
        let num_blocks = n
            .checked_mul(k)
            .map(|nk| nk / 32)
            .ok_or(crate::args::OpError::InvalidArgument)?;

        let mut dims_q = vec![num_blocks * 16];
        let mut dims_d = vec![num_blocks];
        let mut dims_x: Vec<u32> = in_dims.to_vec();
        let mut dims_y: Vec<u32> = out_dims.to_vec();
        let in_rank = dims_x.len() as u32;
        let out_rank = dims_y.len() as u32;

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
        let mk_tv1 = |ttype, dtype: Qnn_DataType_t, rank: u32, dims_ptr: *mut u32| Qnn_TensorV1_t {
            id: 0,
            name: ptr::null(),
            type_: ttype,
            dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            dataType: dtype,
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
                        Qnn_DataType_t_QNN_DATATYPE_UINT_8,
                        1,
                        dims_q.as_mut_ptr(),
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
                        1,
                        dims_d.as_mut_ptr(),
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                        in_rank,
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
                    out_rank,
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
            numOfInputs: 3,
            inputTensors: inputs.as_mut_ptr(),
            numOfOutputs: 1,
            outputTensors: outputs.as_mut_ptr(),
        };
        crate::ops::matmul_q40_f32::build_layout(&v1)
    }

    /// Build an `OpImplLayout` for `CustomRope` with rank-2 input.
    /// `x` shape: `[seq_len, num_heads * head_dim]`, with `num_heads` and
    /// `head_dim` provided as INT_32 op params (M2.H 3rd-attempt path).
    pub fn build_rope_layout_rank2_for(
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<crate::args::OpImplLayout, crate::args::OpError> {
        use crate::qnn::{
            QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, Qnn_ClientBuffer_t,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, Qnn_DataType_t_QNN_DATATYPE_INT_32,
            Qnn_Definition_t_QNN_DEFINITION_UNDEFINED, Qnn_OpConfigV1_t, Qnn_Param_t,
            Qnn_Param_t__bindgen_ty_1, Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED, Qnn_QuantizeParams_t,
            Qnn_QuantizeParams_t__bindgen_ty_1, Qnn_Scalar_t, Qnn_Scalar_t__bindgen_ty_1,
            Qnn_ScaleOffset_t, Qnn_Tensor_t, Qnn_Tensor_t__bindgen_ty_1,
            Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
            Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, Qnn_TensorV1_t,
            Qnn_TensorV1_t__bindgen_ty_1, Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        };
        use std::ffi::CString;
        use std::ptr;

        let flat = num_heads
            .checked_mul(head_dim)
            .ok_or(crate::args::OpError::InvalidArgument)?;
        let mut dims_x = vec![seq_len, flat];
        let mut dims_xout = vec![seq_len, flat];

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
                    dims_xout.as_mut_ptr(),
                ),
            },
        }];

        // Op params: num_heads + head_dim (INT_32 scalars). Names are leaked
        // by leaking the CString, mirroring `build_kv_scatter_layout_for`.
        let nm_nh = CString::new("num_heads").unwrap().into_raw();
        let nm_hd = CString::new("head_dim").unwrap().into_raw();
        let mut params = [
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: nm_nh,
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: num_heads as i32,
                        },
                    },
                },
            },
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: nm_hd,
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: head_dim as i32,
                        },
                    },
                },
            },
        ];
        let v1 = Qnn_OpConfigV1_t {
            name: ptr::null(),
            packageName: ptr::null(),
            typeName: ptr::null(),
            numOfParams: 2,
            params: params.as_mut_ptr(),
            numOfInputs: 1,
            inputTensors: inputs.as_mut_ptr(),
            numOfOutputs: 1,
            outputTensors: outputs.as_mut_ptr(),
        };
        let result = crate::ops::rope::build_layout(&v1);

        // Free the leaked CStrings now that build_layout has consumed them.
        // SAFETY: we leaked these via `into_raw`; recover ownership and drop.
        unsafe {
            drop(CString::from_raw(nm_nh));
            drop(CString::from_raw(nm_hd));
        }
        result
    }

    /// Build an `OpImplLayout` for `CustomFlashAttn` given
    /// `(n_head, n_head_kv, head_dim, kv_capacity, n_kv)`.
    ///
    /// Mocks a `Qnn_OpConfigV1_t` with:
    ///   - inputs[0]: Q     FLOAT_32 [1, n_head, head_dim]
    ///   - inputs[1]: K     FLOAT_16 [1, n_head_kv, kv_capacity, head_dim]
    ///   - inputs[2]: V     FLOAT_16 [1, n_head_kv, kv_capacity, head_dim]
    ///   - inputs[3]: mask  FLOAT_16 [1] (dummy)
    ///   - inputs[4]: sinks FLOAT_32 [1] (dummy)
    ///   - inputs[5]: S     FLOAT_32 [1] (dummy)
    ///   - outputs[0]: O    FLOAT_32 [1, n_head, head_dim]
    ///
    /// Params: n_kv, n_head, n_head_kv, kv_capacity, head_dim (all INT_32).
    /// scale defaults to 1/sqrt(head_dim).
    pub fn build_flash_attn_layout_for(
        n_head: i32,
        n_head_kv: i32,
        head_dim: i32,
        kv_capacity: i32,
        n_kv: i32,
    ) -> Result<crate::args::OpImplLayout, crate::args::OpError> {
        use crate::qnn::{
            QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, Qnn_ClientBuffer_t, Qnn_DataType_t,
            Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            Qnn_DataType_t_QNN_DATATYPE_INT_32, Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
            Qnn_OpConfigV1_t, Qnn_Param_t, Qnn_Param_t__bindgen_ty_1,
            Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED, Qnn_QuantizeParams_t,
            Qnn_QuantizeParams_t__bindgen_ty_1, Qnn_Scalar_t, Qnn_Scalar_t__bindgen_ty_1,
            Qnn_ScaleOffset_t, Qnn_Tensor_t, Qnn_Tensor_t__bindgen_ty_1,
            Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW, Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
            Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE, Qnn_TensorV1_t,
            Qnn_TensorV1_t__bindgen_ty_1, Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
        };
        use std::ffi::CString;
        use std::ptr;

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
        let mk_tv1 = |ttype, dtype: Qnn_DataType_t, rank: u32, dims_ptr: *mut u32| Qnn_TensorV1_t {
            id: 0,
            name: ptr::null(),
            type_: ttype,
            dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            dataType: dtype,
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

        let mut dims_q = vec![1u32, n_head as u32, head_dim as u32];
        let mut dims_kv = vec![1u32, n_head_kv as u32, kv_capacity as u32, head_dim as u32];
        let mut dims_kv2 = dims_kv.clone();
        let mut dims_dummy_f16 = vec![1u32];
        let mut dims_dummy_f32 = vec![1u32];
        let mut dims_dummy_f32_2 = vec![1u32];
        let mut dims_o = vec![1u32, n_head as u32, head_dim as u32];

        let mut inputs = [
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                        3,
                        dims_q.as_mut_ptr(),
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
                        4,
                        dims_kv.as_mut_ptr(),
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
                        4,
                        dims_kv2.as_mut_ptr(),
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
                        1,
                        dims_dummy_f16.as_mut_ptr(),
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                        1,
                        dims_dummy_f32.as_mut_ptr(),
                    ),
                },
            },
            Qnn_Tensor_t {
                version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(
                        Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
                        Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                        1,
                        dims_dummy_f32_2.as_mut_ptr(),
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
                    3,
                    dims_o.as_mut_ptr(),
                ),
            },
        }];

        let n_kv_name = CString::new("n_kv").unwrap();
        let n_head_name = CString::new("n_head").unwrap();
        let n_head_kv_name = CString::new("n_head_kv").unwrap();
        let kv_capacity_name = CString::new("kv_capacity").unwrap();
        let head_dim_name = CString::new("head_dim").unwrap();

        let mut params = [
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: n_kv_name.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 { int32Value: n_kv },
                    },
                },
            },
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: n_head_name.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 { int32Value: n_head },
                    },
                },
            },
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: n_head_kv_name.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: n_head_kv,
                        },
                    },
                },
            },
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: kv_capacity_name.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: kv_capacity,
                        },
                    },
                },
            },
            Qnn_Param_t {
                paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: head_dim_name.as_ptr(),
                __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: Qnn_Scalar_t {
                        dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                        __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                            int32Value: head_dim,
                        },
                    },
                },
            },
        ];

        let v1 = Qnn_OpConfigV1_t {
            name: ptr::null(),
            packageName: ptr::null(),
            typeName: ptr::null(),
            numOfParams: 5,
            params: params.as_mut_ptr(),
            numOfInputs: 6,
            inputTensors: inputs.as_mut_ptr(),
            numOfOutputs: 1,
            outputTensors: outputs.as_mut_ptr(),
        };
        crate::ops::flash_attn::build_layout(&v1)
    }
}
