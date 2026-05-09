//! `CustomSoftmax` op descriptor — out-of-place row-wise softmax.
//!
//! Maps to `kernel_softmax_simple` in `engine/kernels/simple_ops.cl`. This is
//! the one-work-item-per-row variant (no subgroup reduce, no `__local` scratch).
//! M1.6 selects the simple kernel for the same reason as M1.5 (CustomRmsNorm):
//! the QNN GPU OpPackage's `QNN_GPU_KERNEL_ARG_TYPE_LOCAL` path is not yet
//! validated, so the faster `kernel_softmax_opt` (sub_group_reduce_max) is out
//! of scope. M1.6 is correctness-only; performance optimisation is M2.
//!
//! Kernel signature:
//!   kernel void kernel_softmax_simple(
//!       global float * x,        // input rows  [rows * dim]
//!       global float * output,   // output rows [rows * dim]
//!       int dim                  // row width
//!   );
//!
//! Tensor convention:
//!   inputs[0]  = x    (rank 2 [rows, dim] or rank 1 [dim] for rows=1)  FLOAT_32
//!   outputs[0] = out  (same shape as x)                                FLOAT_32
//!
//! Layout:
//!   mem_objects[0] = x    (InputTensor 0,  FLOAT_32, 1D [rows * dim])
//!   mem_objects[1] = out  (OutputTensor 0, FLOAT_32, 1D [rows * dim])
//!
//! Args (3 total, matching kernel_softmax_simple declaration order):
//!   InputTensor(0)  → x
//!   OutputTensor(0) → output
//!   Int(dim)        → dim
//!
//! Workgroup:
//!   global = [rows, 1, 1]
//!   local  = [0, 0, 0] (driver default)

use crate::args::{ArgSpec, MemoryObjectSpec, OpDescriptor, OpError, OpImplLayout};
use crate::qnn::{Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, Qnn_OpConfigV1_t};

pub static DESCRIPTOR: OpDescriptor = OpDescriptor {
    op_type: "CustomSoftmax",
    kernel_name: "kernel_softmax_simple",
    kernel_source: include_str!("../../../../engine/kernels/simple_ops.cl"),
    build_options: "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math",
    build_layout,
};

pub(crate) fn build_layout(v1: &Qnn_OpConfigV1_t) -> Result<OpImplLayout, OpError> {
    if v1.numOfInputs != 1 || v1.numOfOutputs != 1 {
        return Err(OpError::ValidationFailure);
    }

    let in_x = unsafe { (*v1.inputTensors.add(0)).__bindgen_anon_1.v1 };
    let out_y = unsafe { (*v1.outputTensors.add(0)).__bindgen_anon_1.v1 };

    if in_x.dimensions.is_null() || in_x.rank == 0 {
        return Err(OpError::InvalidArgument);
    }
    if out_y.dimensions.is_null() || out_y.rank == 0 {
        return Err(OpError::InvalidArgument);
    }

    // x.dimensions = [rows, dim] (rank 2) or [dim] (rank 1, rows = 1).
    let (rows, dim) = if in_x.rank >= 2 {
        let rows = unsafe { *in_x.dimensions };
        let dim = unsafe { *in_x.dimensions.add(1) };
        (rows, dim)
    } else {
        let dim = unsafe { *in_x.dimensions };
        (1u32, dim)
    };

    if rows == 0 || dim == 0 {
        return Err(OpError::InvalidArgument);
    }

    let total = rows.checked_mul(dim).ok_or(OpError::InvalidArgument)?;

    let mem_objects = vec![
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![total],
            flat_offsets: vec![0u32],
        },
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![total],
            flat_offsets: vec![0u32],
        },
    ];

    let args = vec![
        ArgSpec::InputTensor(0),  // x
        ArgSpec::OutputTensor(0), // output
        ArgSpec::Int(dim as i32), // dim
    ];

    Ok(OpImplLayout {
        mem_objects,
        args,
        global_work_dim: 1,
        local_work_dim: 0,
        global_work: [rows as usize, 1, 1],
        local_work: [0, 0, 0],
    })
}
