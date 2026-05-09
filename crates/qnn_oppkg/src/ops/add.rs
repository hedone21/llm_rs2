//! `CustomAdd` op descriptor — out-of-place float4 add (dst = src0 + src1).
//!
//! Maps to `kernel_add_row` in `engine/kernels/add.cl`.
//!
//! Kernel signature:
//!   kernel void kernel_add_row(
//!       global float4* src0, ulong offset0,
//!       global float4* src1, ulong offset1,
//!       global float4* dst,  ulong offsetd,
//!       int ne
//!   )
//!
//! When `ne == n4` (float4 unit count), idx1 = gid % ne = gid for all gid < ne,
//! so the kernel computes dst[gid] = src0[gid] + src1[gid] — elementwise add.
//!
//! Layout:
//!   mem_objects[0] = src0  (InputTensor 0)
//!   mem_objects[1] = src1  (InputTensor 1)
//!   mem_objects[2] = dst   (output — build_op_state claims the last mem_obj)
//!
//! Args (7 total, matching kernel declaration order):
//!   InputTensor(0)  → src0
//!   ULong(0)        → offset0
//!   InputTensor(1)  → src1
//!   ULong(0)        → offset1
//!   OutputTensor(0) → dst
//!   ULong(0)        → offsetd
//!   Int(n4)         → ne (float4 unit count)
//!
//! global_work: [n4, 1, 1], local 0 (driver default).

use crate::args::{ArgSpec, MemoryObjectSpec, OpDescriptor, OpError, OpImplLayout};
use crate::qnn::{Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, Qnn_OpConfigV1_t};

pub static DESCRIPTOR: OpDescriptor = OpDescriptor {
    op_type: "CustomAdd",
    kernel_name: "kernel_add_row",
    kernel_source: include_str!("../../../../engine/kernels/add.cl"),
    build_options: "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math",
    build_layout,
};

pub(crate) fn build_layout(v1: &Qnn_OpConfigV1_t) -> Result<OpImplLayout, OpError> {
    if v1.numOfInputs < 2 || v1.numOfOutputs < 1 {
        return Err(OpError::InvalidArgument);
    }

    // Compute total element count from the output tensor dimensions.
    let out_t = unsafe { *v1.outputTensors };
    let out_v1 = unsafe { out_t.__bindgen_anon_1.v1 };
    let out_rank = out_v1.rank;
    let out_dims_ptr = out_v1.dimensions;

    if out_dims_ptr.is_null() || out_rank == 0 {
        return Err(OpError::InvalidArgument);
    }

    let mut total_elems: u32 = 1;
    for i in 0..out_rank {
        let d = unsafe { *out_dims_ptr.add(i as usize) };
        total_elems = total_elems.saturating_mul(d);
    }

    if total_elems == 0 || !total_elems.is_multiple_of(4) {
        return Err(OpError::InvalidArgument);
    }

    // n4: float4 unit count. When ne == n4, idx1 = gid % n4 == gid → elementwise add.
    let n4 = (total_elems / 4) as i32;

    // Three mem_objects: src0, src1, dst.
    // build_op_state treats the last entry as the output claim.
    let flat_dims = vec![total_elems];
    let flat_offsets = vec![0u32];
    let mk = || MemoryObjectSpec {
        data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
        flat_dims: flat_dims.clone(),
        flat_offsets: flat_offsets.clone(),
    };
    let mem_objects = vec![mk(), mk(), mk()];

    // Args match kernel_add_row declaration order:
    //   global float4* src0  → InputTensor(0)
    //   ulong offset0        → ULong(0)
    //   global float4* src1  → InputTensor(1)
    //   ulong offset1        → ULong(0)
    //   global float4* dst   → OutputTensor(0)
    //   ulong offsetd        → ULong(0)
    //   int ne               → Int(n4)
    let args = vec![
        ArgSpec::InputTensor(0),
        ArgSpec::ULong(0),
        ArgSpec::InputTensor(1),
        ArgSpec::ULong(0),
        ArgSpec::OutputTensor(0),
        ArgSpec::ULong(0),
        ArgSpec::Int(n4),
    ];

    Ok(OpImplLayout {
        mem_objects,
        args,
        global_work_dim: 1,
        local_work_dim: 0,
        global_work: [n4 as usize, 1, 1],
        local_work: [0, 0, 0],
    })
}
