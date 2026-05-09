//! `CustomMatMulQ40F32` op descriptor — Q4_0 SOA weight × F32 input → F32 output.
//!
//! Maps to `kernel_mul_mat_q4_0_f32_8x_flat` in
//! `engine/kernels/mul_mv_q4_0_f32_8x_flat.cl` (llama.cpp 8x_flat variant).
//! Single-output kernel — multi-output abstraction limit (see `deq_q40.rs`)
//! does not apply.
//!
//! Tensor convention (matches `microbench_ops::dispatch_llama_q4` host call):
//!   inputs[0]  = src0_q [num_blocks * 16]  UINT_8   (4-bit packed quants)
//!   inputs[1]  = src0_d [num_blocks]       FLOAT_16 (per-block scale)
//!   inputs[2]  = src1   [M, K]             FLOAT_32 (activation x)
//!   outputs[0] = dst    [M, N]             FLOAT_32 (y)
//!
//! `num_blocks = (N * K) / 32` (QK4_0 = 32 elements per block).
//!
//! Layout:
//!   mem_objects[0] = src0_q (InputTensor 0,  UINT_8,   1D [num_blocks*16])
//!   mem_objects[1] = src0_d (InputTensor 1,  FLOAT_16, 1D [num_blocks])
//!   mem_objects[2] = src1   (InputTensor 2,  FLOAT_32, 1D [M*K])
//!   mem_objects[3] = dst    (OutputTensor 0, FLOAT_32, 1D [M*N])
//!
//! Args (15 total, kernel declaration order):
//!   InputTensor(0)  → src0_q
//!   InputTensor(1)  → src0_d
//!   InputTensor(2)  → src1
//!   ULong(0)        → offset1
//!   OutputTensor(0) → dst
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
//! Workgroup (matches `dispatch_llama_q4`, gws/lws ports llama.cpp 8x_flat
//! dispatch as exercised in `engine/src/bin/microbench_ops.rs`):
//!   global = [ ((N + 7) / 8) * 64, 1, 1 ]
//!   local  = [ 64, 1, 1 ]
//!
//! QK4_0 = 32 elements per block. `K * N` must be a multiple of 32 (Q4_0 grain).

use crate::args::{ArgSpec, MemoryObjectSpec, OpDescriptor, OpError, OpImplLayout};
use crate::qnn::{
    Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
    Qnn_DataType_t_QNN_DATATYPE_UINT_8, Qnn_OpConfigV1_t,
};

const QK4_0: u32 = 32;
const QS_PER_BLOCK: u32 = 16;

pub static DESCRIPTOR: OpDescriptor = OpDescriptor {
    op_type: "CustomMatMulQ40F32",
    kernel_name: "kernel_mul_mat_q4_0_f32_8x_flat",
    kernel_source: include_str!("../../../../engine/kernels/mul_mv_q4_0_f32_8x_flat.cl"),
    build_options: "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math",
    build_layout,
};

pub(crate) fn build_layout(v1: &Qnn_OpConfigV1_t) -> Result<OpImplLayout, OpError> {
    if v1.numOfInputs < 3 || v1.numOfOutputs < 1 {
        return Err(OpError::ValidationFailure);
    }

    // src1 [M, K], dst [M, N] drive the dim derivation. src0_q / src0_d are
    // sized as [num_blocks * 16] / [num_blocks] respectively where
    // num_blocks = N * K / 32.
    //
    // Rank-flexible (M2.H 3rd attempt): both input x and output y accept
    //   rank 2 [M, K] / [M, N]                     (canonical)
    //   rank 3 [M, anyDim, head_dim] / [M, n_heads, head_dim]
    //                                              (Qwen layer reshape view —
    //                                               last dim semantics differ
    //                                               but flat element counts
    //                                               match: K = product(in[1..]),
    //                                               N = product(out[1..]))
    // M (batch / sequence) is taken as in.dimensions[0]. Caller is responsible
    // for ensuring `out.dimensions[0] == in.dimensions[0]`.
    let in_x = unsafe { (*v1.inputTensors.add(2)).__bindgen_anon_1.v1 };
    let out_y = unsafe { (*v1.outputTensors.add(0)).__bindgen_anon_1.v1 };

    if in_x.dimensions.is_null() || in_x.rank < 2 {
        return Err(OpError::InvalidArgument);
    }
    if out_y.dimensions.is_null() || out_y.rank < 2 {
        return Err(OpError::InvalidArgument);
    }

    let m = unsafe { *in_x.dimensions };
    let y_m = unsafe { *out_y.dimensions };
    // K = product of all input dims after batch (rank 2: K = dims[1];
    //                                            rank 3: K = dims[1] * dims[2]).
    let k: u32 = (1..in_x.rank).try_fold(1u32, |acc, i| {
        let d = unsafe { *in_x.dimensions.add(i as usize) };
        acc.checked_mul(d).ok_or(OpError::InvalidArgument)
    })?;
    // N = product of all output dims after batch.
    let n: u32 = (1..out_y.rank).try_fold(1u32, |acc, i| {
        let d = unsafe { *out_y.dimensions.add(i as usize) };
        acc.checked_mul(d).ok_or(OpError::InvalidArgument)
    })?;

    if m == 0 || n == 0 || k == 0 || y_m != m {
        return Err(OpError::InvalidArgument);
    }

    // K must be a multiple of QK4_0 (32), and N * K must be a multiple of 32
    // for block alignment. Caller is responsible for ensuring this when
    // generating SOA buffers.
    if !k.is_multiple_of(QK4_0) {
        return Err(OpError::InvalidArgument);
    }

    let num_blocks = match n.checked_mul(k).map(|nk| nk / QK4_0) {
        Some(nb) if nb > 0 => nb,
        _ => return Err(OpError::InvalidArgument),
    };

    // Four mem_objects: src0_q, src0_d, src1, dst.
    let mem_objects = vec![
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_UINT_8,
            flat_dims: vec![num_blocks * QS_PER_BLOCK],
            flat_offsets: vec![0u32],
        },
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
            flat_dims: vec![num_blocks],
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

    // 15 args, kernel declaration order (see kernel signature in module doc).
    let args = vec![
        ArgSpec::InputTensor(0),  // src0_q
        ArgSpec::InputTensor(1),  // src0_d
        ArgSpec::InputTensor(2),  // src1
        ArgSpec::ULong(0),        // offset1
        ArgSpec::OutputTensor(0), // dst
        ArgSpec::ULong(0),        // offsetd
        ArgSpec::Int(k as i32),   // ne00
        ArgSpec::Int(n as i32),   // ne01
        ArgSpec::Int(1),          // ne02
        ArgSpec::Int(k as i32),   // ne10
        ArgSpec::Int(1),          // ne12
        ArgSpec::Int(n as i32),   // ne0
        ArgSpec::Int(m as i32),   // ne1
        ArgSpec::Int(1),          // r2
        ArgSpec::Int(1),          // r3
    ];

    let global_x = (n.div_ceil(8) * 64) as usize;

    Ok(OpImplLayout {
        mem_objects,
        args,
        output_claims: None,
        global_work_dim: 3,
        local_work_dim: 3,
        global_work: [global_x, 1, 1],
        local_work: [64, 1, 1],
    })
}
