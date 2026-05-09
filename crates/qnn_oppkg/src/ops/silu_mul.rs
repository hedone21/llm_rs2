//! `CustomSiluMul` op descriptor — in-place SwiGLU activation.
//!
//! Maps to `kernel_silu_mul_simple` in `engine/kernels/simple_ops.cl`. SwiGLU
//! computes `x[i] = silu(x[i]) * y[i]` per float4 element. The production
//! kernel is **in-place**: it writes the result back into `x`. There is no
//! out-of-place variant of silu+mul in the engine kernels (only standalone
//! `kernel_silu` / `kernel_silu_4` exist), so M1.7 maps the in-place pattern
//! directly via `QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READWRITE` (`ArgSpec::InOutTensor`).
//!
//! Kernel signature:
//!   kernel void kernel_silu_mul_simple(
//!       global float4 * x,   // in-place: read + write
//!       global float4 * y,   // read-only
//!       int size4            // float4 element count = total_elements / 4
//!   );
//!
//! Tensor convention:
//!   inputs[0]  = x        (rank 2 [rows, dim] or rank 1 [dim] for rows=1)  FLOAT_32
//!   inputs[1]  = y        (same shape as x)                                FLOAT_32
//!   outputs[0] = x_inout  (alias of inputs[0]; required because
//!                          `build_op_state` claims the last mem_object as
//!                          the op output. The graph-level alias keeps the
//!                          OpPackage abstraction intact while the kernel
//!                          actually mutates inputs[0] via OP_INPUT_READWRITE.)
//!                                                                          FLOAT_32
//!
//! Layout:
//!   mem_objects[0] = x        (FLOAT_32, 1D [total])
//!   mem_objects[1] = y        (FLOAT_32, 1D [total])
//!   mem_objects[2] = x_alias  (FLOAT_32, 1D [total]; output-claim slot)
//!
//! Args (3 total, matching kernel_silu_mul_simple declaration order):
//!   InOutTensor(0)  → x  (read+write, OP_INPUT_READWRITE)
//!   InputTensor(1)  → y  (read-only)
//!   Int(size4)      → size4 = total / 4
//!
//! Workgroup:
//!   global = [size4, 1, 1]
//!   local  = [0, 0, 0] (driver default)
//!
//! Constraint: `total = rows * dim` must be a multiple of 4. The kernel is
//! float4-vectorised; non-multiple-of-4 inputs are rejected at validation.

use crate::args::{ArgSpec, MemoryObjectSpec, OpDescriptor, OpError, OpImplLayout};
use crate::qnn::{Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, Qnn_OpConfigV1_t};

pub static DESCRIPTOR: OpDescriptor = OpDescriptor {
    op_type: "CustomSiluMul",
    kernel_name: "kernel_silu_mul_simple",
    kernel_source: include_str!("../../../../engine/kernels/simple_ops.cl"),
    build_options: "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math",
    build_layout,
};

pub(crate) fn build_layout(v1: &Qnn_OpConfigV1_t) -> Result<OpImplLayout, OpError> {
    if v1.numOfInputs < 2 || v1.numOfOutputs < 1 {
        return Err(OpError::ValidationFailure);
    }

    let in_x = unsafe { (*v1.inputTensors.add(0)).__bindgen_anon_1.v1 };
    let in_y = unsafe { (*v1.inputTensors.add(1)).__bindgen_anon_1.v1 };
    let out_x = unsafe { (*v1.outputTensors.add(0)).__bindgen_anon_1.v1 };

    if in_x.dimensions.is_null() || in_x.rank == 0 {
        return Err(OpError::InvalidArgument);
    }
    if in_y.dimensions.is_null() || in_y.rank == 0 {
        return Err(OpError::InvalidArgument);
    }
    if out_x.dimensions.is_null() || out_x.rank == 0 {
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

    // kernel_silu_mul_simple is float4-vectorised; require total divisible by 4.
    if total % 4 != 0 {
        return Err(OpError::ValidationFailure);
    }
    let size4 = total / 4;

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
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![total],
            flat_offsets: vec![0u32],
        },
    ];

    let args = vec![
        ArgSpec::InOutTensor(0),    // x (read+write, OP_INPUT_READWRITE)
        ArgSpec::InputTensor(1),    // y (read-only)
        ArgSpec::Int(size4 as i32), // size4 = total / 4
    ];

    Ok(OpImplLayout {
        mem_objects,
        args,
        global_work_dim: 1,
        local_work_dim: 0,
        global_work: [size4 as usize, 1, 1],
        local_work: [0, 0, 0],
    })
}
