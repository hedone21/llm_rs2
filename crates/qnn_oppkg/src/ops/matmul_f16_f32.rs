//! `CustomMatMulF16F32` op descriptor — F16 weight × F32 input → F32 output.
//!
//! Maps to `kernel_mul_mat_f16_f32` in `engine/kernels/mul_mv_f16_f32.cl`.
//!
//! Tensor convention (mirrors PoC `create_matmul_impl`):
//!   inputs[0] = weight  [N, K]  FLOAT_16
//!   inputs[1] = x       [M, K]  FLOAT_32
//!   outputs[0] = y      [M, N]  FLOAT_32
//!
//! Layout:
//!   mem_objects[0] = weight  (InputTensor 0, FLOAT_16, 1D [N*K])
//!   mem_objects[1] = x       (InputTensor 1, FLOAT_32, 1D [M*K])
//!   mem_objects[2] = y       (output,        FLOAT_32, 1D [M*N])
//!
//! Args (15 total, matching kernel declaration order):
//!   InputTensor(0)  → src0 (weight)
//!   ULong(0)        → offset0
//!   InputTensor(1)  → src1 (x)
//!   ULong(0)        → offset1
//!   OutputTensor(0) → dst (y)
//!   ULong(0)        → offsetd
//!   Int(K)          → ne00
//!   Int(N)          → ne01
//!   Int(1)          → ne02
//!   Int(K)          → ne10
//!   Int(1)          → ne12
//!   Int(N)          → ne0
//!   Int(M)          → ne1
//!   Int(1)          → r2
//!   Int(1)          → r3
//!
//! Workgroup (N_DST = 2, mirrors kernel `#define N_DST 2`):
//!   global = [ ((N + N_DST - 1) / N_DST) * 64, M * 4, 1 ]
//!   local  = [ 64, 4, 1 ]

use crate::args::{ArgSpec, MemoryObjectSpec, OpDescriptor, OpError, OpImplLayout};
use crate::qnn::{
    Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, Qnn_OpConfigV1_t,
};

const N_DST: u32 = 2;

pub static DESCRIPTOR: OpDescriptor = OpDescriptor {
    op_type: "CustomMatMulF16F32",
    kernel_name: "kernel_mul_mat_f16_f32",
    kernel_source: include_str!("../../../../engine/kernels/mul_mv_f16_f32.cl"),
    build_options: "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math",
    build_layout,
};

pub(crate) fn build_layout(v1: &Qnn_OpConfigV1_t) -> Result<OpImplLayout, OpError> {
    if v1.numOfInputs < 2 || v1.numOfOutputs < 1 {
        return Err(OpError::ValidationFailure);
    }

    // Read tensor dims (PoC create_matmul_impl 422~430).
    let in_w = unsafe { (*v1.inputTensors.add(0)).__bindgen_anon_1.v1 };
    let in_x = unsafe { (*v1.inputTensors.add(1)).__bindgen_anon_1.v1 };
    let out_y = unsafe { (*v1.outputTensors.add(0)).__bindgen_anon_1.v1 };

    if in_w.dimensions.is_null() || in_w.rank < 2 {
        return Err(OpError::InvalidArgument);
    }
    if in_x.dimensions.is_null() || in_x.rank < 2 {
        return Err(OpError::InvalidArgument);
    }
    if out_y.dimensions.is_null() || out_y.rank < 2 {
        return Err(OpError::InvalidArgument);
    }

    let w_n = unsafe { *in_w.dimensions };
    let w_k = unsafe { *in_w.dimensions.add(1) };
    let x_m = unsafe { *in_x.dimensions };
    let y_n = unsafe { *out_y.dimensions.add(1) };

    let k = w_k;
    let n = w_n.max(y_n);
    let m = x_m;

    if k == 0 || n == 0 || m == 0 {
        return Err(OpError::InvalidArgument);
    }

    // Three mem_objects: weight (F16), x (F32), y (F32).
    let mem_objects = vec![
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
            flat_dims: vec![n * k],
            flat_offsets: vec![0u32],
        },
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![m * k],
            flat_offsets: vec![0u32],
        },
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![m * n],
            flat_offsets: vec![0u32],
        },
    ];

    // Args: 15 entries in kernel_mul_mat_f16_f32 declaration order.
    let args = vec![
        ArgSpec::InputTensor(0),
        ArgSpec::ULong(0),
        ArgSpec::InputTensor(1),
        ArgSpec::ULong(0),
        ArgSpec::OutputTensor(0),
        ArgSpec::ULong(0),
        ArgSpec::Int(k as i32), // ne00
        ArgSpec::Int(n as i32), // ne01
        ArgSpec::Int(1),        // ne02
        ArgSpec::Int(k as i32), // ne10
        ArgSpec::Int(1),        // ne12
        ArgSpec::Int(n as i32), // ne0
        ArgSpec::Int(m as i32), // ne1
        ArgSpec::Int(1),        // r2
        ArgSpec::Int(1),        // r3
    ];

    let global_x = (n.div_ceil(N_DST) * 64) as usize;
    let global_y = (m * 4) as usize;

    Ok(OpImplLayout {
        mem_objects,
        args,
        global_work_dim: 3,
        local_work_dim: 3,
        global_work: [global_x, global_y, 1],
        local_work: [64, 4, 1],
    })
}
