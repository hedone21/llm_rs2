//! `CustomBiasAdd` op descriptor — out-of-place broadcast row bias add.
//!
//! D-D.6 Phase B: Qwen2.5-style attention QKV bias. graph 14-node chain은
//! 기존에 Q/K/V matmul 직후 bias add op이 없어 fast path forward가
//! production fallback과 발산 (root cause 격리: forward_gen vs microbench
//! stage 비교에서 cos 0.43 → 0.72로 회복 확인).
//!
//! Maps to `kernel_add_row_bias_oop` in `engine/kernels/simple_ops.cl`.
//!
//! Kernel signature:
//!   kernel void kernel_add_row_bias_oop(
//!       global const float * x,
//!       global const float * bias,
//!       global float * y,
//!       int dim,
//!       int total_elements
//!   )
//!   y[gid] = x[gid] + bias[gid % dim] for gid < total_elements
//!
//! Tensor convention:
//!   inputs[0] = x      (rank 2 [rows, dim] or rank 1 [dim])  FLOAT_32
//!   inputs[1] = bias   (rank 1 [dim])                        FLOAT_32
//!   outputs[0] = y     (same shape as x)                     FLOAT_32
//!
//! Layout:
//!   mem_objects[0] = x     (InputTensor 0)
//!   mem_objects[1] = bias  (InputTensor 1)
//!   mem_objects[2] = y     (OutputTensor 0)
//!
//! Args (5 total, matching kernel declaration order):
//!   InputTensor(0)  → x
//!   InputTensor(1)  → bias
//!   OutputTensor(0) → y
//!   Int(dim)        → dim
//!   Int(total)      → total_elements
//!
//! Workgroup: global = [total_elements, 1, 1], local = driver default (0).

use crate::args::{ArgSpec, MemoryObjectSpec, OpDescriptor, OpError, OpImplLayout};
use crate::qnn::{Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, Qnn_OpConfigV1_t};

pub static DESCRIPTOR: OpDescriptor = OpDescriptor {
    op_type: "CustomBiasAdd",
    kernel_name: "kernel_add_row_bias_oop",
    kernel_source: include_str!("../../../../engine/kernels/simple_ops.cl"),
    build_options: "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math",
    build_layout,
};

pub(crate) fn build_layout(v1: &Qnn_OpConfigV1_t) -> Result<OpImplLayout, OpError> {
    if v1.numOfInputs < 2 || v1.numOfOutputs < 1 {
        return Err(OpError::ValidationFailure);
    }

    let in_x = unsafe { (*v1.inputTensors.add(0)).__bindgen_anon_1.v1 };
    let in_b = unsafe { (*v1.inputTensors.add(1)).__bindgen_anon_1.v1 };
    let out_y = unsafe { (*v1.outputTensors.add(0)).__bindgen_anon_1.v1 };

    if in_x.dimensions.is_null() || in_x.rank == 0 {
        return Err(OpError::InvalidArgument);
    }
    if in_b.dimensions.is_null() || in_b.rank == 0 {
        return Err(OpError::InvalidArgument);
    }
    if out_y.dimensions.is_null() || out_y.rank == 0 {
        return Err(OpError::InvalidArgument);
    }

    // x.dimensions = [rows, dim] (rank 2) or [dim] (rank 1).
    // bias.dimensions = [dim] (rank 1).
    let total_elems = (0..in_x.rank).fold(1u32, |acc, i| {
        acc.saturating_mul(unsafe { *in_x.dimensions.add(i as usize) })
    });
    let dim = unsafe { *in_b.dimensions };

    if total_elems == 0 || dim == 0 {
        return Err(OpError::InvalidArgument);
    }
    if !total_elems.is_multiple_of(dim) {
        return Err(OpError::InvalidArgument);
    }

    let mem_objects = vec![
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![total_elems],
            flat_offsets: vec![0u32],
        },
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![dim],
            flat_offsets: vec![0u32],
        },
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![total_elems],
            flat_offsets: vec![0u32],
        },
    ];

    let args = vec![
        ArgSpec::InputTensor(0),
        ArgSpec::InputTensor(1),
        ArgSpec::OutputTensor(0),
        ArgSpec::Int(dim as i32),
        ArgSpec::Int(total_elems as i32),
    ];

    Ok(OpImplLayout {
        mem_objects,
        args,
        output_claims: None,
        global_work_dim: 1,
        local_work_dim: 0,
        global_work: [total_elems as usize, 1, 1],
        local_work: [0, 0, 0],
    })
}
