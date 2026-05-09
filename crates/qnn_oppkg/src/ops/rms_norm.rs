//! `CustomRmsNorm` op descriptor — out-of-place RMSNorm (Llama-style).
//!
//! Maps to `kernel_rms_norm_simple` in `engine/kernels/simple_ops.cl`. This is
//! the one-work-item-per-row variant (no subgroup reduce, no `__local` scratch).
//! M1.5 selects the simple kernel because the QNN GPU OpPackage abstraction's
//! `QNN_GPU_KERNEL_ARG_TYPE_LOCAL` path is not yet validated end-to-end on this
//! runtime — `kernel_rms_norm_oop` triggers `graphFinalize err=0x1786`
//! (`QNN_GRAPH_ERROR_FINALIZE_FAILED`) on the device, while the simple kernel
//! avoids the LOCAL arg entirely. Performance is suboptimal (one thread per
//! row), but M1.5 scope is correctness-only.
//!
//! Kernel signature:
//!   kernel void kernel_rms_norm_simple(
//!       global float * x,        // input rows  [rows * dim]
//!       global float * weight,   // per-dim gain [dim]
//!       global float * output,   // output rows [rows * dim]
//!       int dim,                 // row width
//!       float eps
//!   );
//!
//! `eps` is fixed (`1e-5`, Llama standard). The `add_unit` (Gemma3) flag is not
//! exposed by this kernel — extending to Gemma needs the OOP variant once
//! LocalMem is validated.
//!
//! Tensor convention:
//!   inputs[0] = x       (rank 2 [rows, dim] or rank 1 [dim] for rows=1)  FLOAT_32
//!   inputs[1] = weight  (rank 1 [dim])                                   FLOAT_32
//!   outputs[0] = out    (same shape as x)                                FLOAT_32
//!
//! Layout:
//!   mem_objects[0] = x       (InputTensor 0, FLOAT_32, 1D [rows * dim])
//!   mem_objects[1] = weight  (InputTensor 1, FLOAT_32, 1D [dim])
//!   mem_objects[2] = out     (output,        FLOAT_32, 1D [rows * dim])
//!
//! Args (5 total, matching kernel_rms_norm_simple declaration order):
//!   InputTensor(0)  → x
//!   InputTensor(1)  → weight
//!   OutputTensor(0) → output
//!   Int(dim)        → dim
//!   Float(EPS)      → eps
//!
//! Workgroup:
//!   global = [rows, 1, 1]
//!   local  = [0, 0, 0] (driver default)

use crate::args::{ArgSpec, MemoryObjectSpec, OpDescriptor, OpError, OpImplLayout};
use crate::qnn::{Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, Qnn_OpConfigV1_t};

const EPS: f32 = 1e-5;

pub static DESCRIPTOR: OpDescriptor = OpDescriptor {
    op_type: "CustomRmsNorm",
    kernel_name: "kernel_rms_norm_simple",
    kernel_source: include_str!("../../../../engine/kernels/simple_ops.cl"),
    build_options: "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math",
    build_layout,
};

pub(crate) fn build_layout(v1: &Qnn_OpConfigV1_t) -> Result<OpImplLayout, OpError> {
    if v1.numOfInputs < 2 || v1.numOfOutputs < 1 {
        return Err(OpError::ValidationFailure);
    }

    let in_x = unsafe { (*v1.inputTensors.add(0)).__bindgen_anon_1.v1 };
    let in_w = unsafe { (*v1.inputTensors.add(1)).__bindgen_anon_1.v1 };
    let out_y = unsafe { (*v1.outputTensors.add(0)).__bindgen_anon_1.v1 };

    if in_x.dimensions.is_null() || in_x.rank == 0 {
        return Err(OpError::InvalidArgument);
    }
    if in_w.dimensions.is_null() || in_w.rank == 0 {
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

    // weight may be rank 1 [dim] or rank 2 [1, dim] (broadcast). Accept either.
    let _ = in_w; // dim already pulled from x.

    let total = rows.checked_mul(dim).ok_or(OpError::InvalidArgument)?;

    let mem_objects = vec![
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![total],
            flat_offsets: vec![0u32],
        },
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![dim],
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
        ArgSpec::InputTensor(1),  // weight
        ArgSpec::OutputTensor(0), // output
        ArgSpec::Int(dim as i32), // dim
        ArgSpec::Float(EPS),      // eps
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
