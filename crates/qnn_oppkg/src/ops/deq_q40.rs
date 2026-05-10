//! `CustomDeqQ40` op descriptor — Q4_0 block AOS→SOA conversion.
//!
//! **M2.C status (2026-05-09): RED — multi-output OpPackage limitation.**
//!
//! Maps to `kernel_convert_block_q4_0` in `engine/kernels/cvt.cl`.
//!
//! Kernel signature (3 buffer args, 1 input + 2 outputs):
//!   kernel void kernel_convert_block_q4_0(
//!       global struct block_q4_0 * src0,
//!       global uchar  * dst_q,
//!       global half   * dst_d
//!   )
//!
//! ## Investigation summary (M2.C device debug, S25 Adreno 830)
//!
//! 1. Naive mapping (`OutputTensor(0)=dst_q`, `OutputTensor(1)=dst_d`) reaches
//!    `graphFinalize` with `GPU_ERROR_IMMUTABLE - Tensor dst_q is immutable`.
//!    Surfaces as `graphExecute err=0x1770 (INVALID_ARGUMENT)` because the
//!    OpPackage abstraction (`op_impl::build_op_state`) claims a **single**
//!    `QnnGpu_OutputClaim_t` from the last `mem_object`. The non-claimed first
//!    output (dst_q) is treated as immutable.
//!
//! 2. Single-output remap A (dst_q as `InOutTensor` input, dst_d as graph
//!    output): IMMUTABLE error gone, but
//!    `GPU_ERROR_OPENCL(10014) - OpenCL error enqueueing kernel` at finalize.
//!    No further driver detail in logcat. Forcing `local_work=[1,0,0]`
//!    (instead of driver-default 0) did not change the result.
//!
//! 3. silu_mul-style alias (3 inputs incl. dst_q + dst_d as `InOutTensor`,
//!    1 alias output sharing fd with dst_q): still
//!    `GPU_ERROR_OPENCL(10014)` plus `Tensor has unusual lifetime: [1,0]`
//!    warnings. Same ultimate failure.
//!
//! ## Conclusion
//!
//! Multi-output kernels (>1 actual output buffer the kernel writes to)
//! cannot be expressed cleanly with the current OpPackage abstraction —
//! `out_claim` is single-valued and the runtime appears to reject in-place
//! writes to multiple input tensors that are not a true alias of the claimed
//! output. This descriptor remains as the naive 2-output declaration so the
//! limitation is reproducible; M2.C is marked YELLOW with abstraction-level
//! follow-up required (multi-`out_claim` support in `op_impl::build_op_state`).
//!
//! ## M2.D impact (single-output ops)
//!
//! `MatMulQ40F32` is a single-output op (Q4_0 weight + F32 activation in →
//! F32 result out). The multi-output blocker does not apply. The `Q4_0`
//! buffer-handling pattern (struct-pointer kernel arg, fp16 scale embedded
//! per block) is shared and was demonstrated to work in the single-output
//! remap (case 2 above) up to OpenCL enqueue, leaving the actual kernel
//! launch failure as the M2.D risk to verify next.
//!
//! Layout (declared as multi-output to surface the `IMMUTABLE` diagnostic):
//!   mem_objects[0] = src0  (InputTensor 0)  — UINT_8, [num_blocks * 18]
//!   mem_objects[1] = dst_q (OutputTensor 0) — UINT_8, [num_blocks * 16]
//!   mem_objects[2] = dst_d (OutputTensor 1) — FLOAT_16, [num_blocks]
//!
//! Args (3 total, matching kernel declaration order):
//!   InputTensor(0)  → src0
//!   OutputTensor(0) → dst_q
//!   OutputTensor(1) → dst_d
//!
//! global_work: [num_blocks, 1, 1], local 0 (driver default).
//!
//! block_q4_0 layout: half d (2 bytes) + uchar qs[16] = 18 bytes total.

use crate::args::{ArgSpec, MemoryObjectSpec, OpDescriptor, OpError, OpImplLayout};
use crate::qnn::{
    Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, Qnn_DataType_t_QNN_DATATYPE_UINT_8, Qnn_OpConfigV1_t,
};

/// block_q4_0 size in bytes: 2 (half d) + 16 (uchar qs[16]) = 18.
const BLOCK_SIZE: u32 = 18;
/// qs portion size per block in bytes.
const QS_PER_BLOCK: u32 = 16;

pub static DESCRIPTOR: OpDescriptor = OpDescriptor {
    op_type: "CustomDeqQ40",
    kernel_name: "kernel_convert_block_q4_0",
    kernel_source: include_str!("../../../../engine/kernels/cvt.cl"),
    build_options: "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math",
    build_layout,
};

pub(crate) fn build_layout(v1: &Qnn_OpConfigV1_t) -> Result<OpImplLayout, OpError> {
    if v1.numOfInputs < 1 || v1.numOfOutputs < 2 {
        return Err(OpError::InvalidArgument);
    }

    // Derive num_blocks from src0 dimensions.
    // src0 is flat UINT_8: total_bytes = num_blocks * BLOCK_SIZE.
    let src0_t = unsafe { *v1.inputTensors };
    let src0_v1 = unsafe { src0_t.__bindgen_anon_1.v1 };
    let rank = src0_v1.rank;
    let dims_ptr = src0_v1.dimensions;

    if dims_ptr.is_null() || rank == 0 {
        return Err(OpError::InvalidArgument);
    }

    let mut total_bytes: u32 = 1;
    for i in 0..rank {
        let d = unsafe { *dims_ptr.add(i as usize) };
        total_bytes = total_bytes.saturating_mul(d);
    }

    if total_bytes == 0 || !total_bytes.is_multiple_of(BLOCK_SIZE) {
        return Err(OpError::InvalidArgument);
    }

    let num_blocks = total_bytes / BLOCK_SIZE;

    // mem_objects: src0, dst_q, dst_d
    let mem_objects = vec![
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_UINT_8,
            flat_dims: vec![num_blocks * BLOCK_SIZE],
            flat_offsets: vec![0],
        },
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_UINT_8,
            flat_dims: vec![num_blocks * QS_PER_BLOCK],
            flat_offsets: vec![0],
        },
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
            flat_dims: vec![num_blocks],
            flat_offsets: vec![0],
        },
    ];

    // Args match kernel declaration order:
    //   global struct block_q4_0 * src0 → InputTensor(0)
    //   global uchar * dst_q            → OutputTensor(0)
    //   global half  * dst_d            → OutputTensor(1)
    let args = vec![
        ArgSpec::InputTensor(0),
        ArgSpec::OutputTensor(0),
        ArgSpec::OutputTensor(1),
    ];

    Ok(OpImplLayout {
        mem_objects,
        args,
        output_claims: None,
        global_work_dim: 1,
        local_work_dim: 0,
        global_work: [num_blocks as usize, 1, 1],
        local_work: [0, 0, 0],
    })
}
