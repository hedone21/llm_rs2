//! `CustomRope` op descriptor — out-of-place neox-style RoPE rotation.
//!
//! Maps to `kernel_rope_simple_oop` in `engine/kernels/simple_ops.cl`. The
//! kernel rotates the (i, i + head_dim/2) pair of every (seq, head) — neox
//! layout — and writes the result to a separate `x_out` buffer.
//!
//! ## OOP variant rationale (M2.H, 2026-05-09)
//!
//! Earlier revisions used `kernel_rope_simple` (in-place) with the
//! `OutputTensorAliased(0)` pattern: the graph-level output edge aliased
//! `inputs[0]` so chain composition could deliver the mutated buffer
//! downstream. Standalone microbench (M2.B) was GREEN with that layout, but
//! the SDK rejected it inside chained graphs: `microbench_qnn_qwen_layer`
//! (B1') confirmed that an OpPackage chain composed entirely of OOP ops works
//! whereas chains where the **last** op is in-place fail. Switching RoPE to a
//! true OOP kernel is the structural fix.
//!
//! Kernel signature:
//!   kernel void kernel_rope_simple_oop(
//!       global const float * x_in,
//!       global float * x_out,
//!       int head_dim,
//!       int num_heads,
//!       int seq_len,
//!       int start_pos,
//!       float theta              // 10000.0 (Llama) or 1000000.0 (Qwen2.5)
//!   );
//!
//! Tensor convention (rank-flexible):
//!   inputs[0]  = x_in   rank 3 [seq_len, num_heads, head_dim]    FLOAT_32
//!                or
//!                rank 2 [seq_len, num_heads * head_dim]           FLOAT_32
//!                  (requires `num_heads` and `head_dim` op params)
//!   outputs[0] = x_out  same shape, distinct buffer               FLOAT_32
//!
//! Op params (all optional):
//!   params[i].name = "start_pos", scalar INT_32  (default 0)
//!   params[i].name = "theta",     scalar FLOAT_32 (default 10000.0)
//!   params[i].name = "num_heads", scalar INT_32  (rank-2 input only)
//!   params[i].name = "head_dim",  scalar INT_32  (rank-2 input only)
//!
//! Layout:
//!   mem_objects[0] = x_in   (FLOAT_32, 1D [total = seq_len * num_heads * head_dim])
//!   mem_objects[1] = x_out  (FLOAT_32, 1D [total]; claimed output)
//!
//! Args (7 total, matching kernel_rope_simple_oop declaration order):
//!   InputTensor(0)        → x_in   (read-only)
//!   OutputTensor(0)       → x_out  (write-only)
//!   Int(head_dim)
//!   Int(num_heads)
//!   Int(seq_len)
//!   Int(start_pos)
//!   Float(theta)
//!
//! Workgroup:
//!   global_work = [seq_len * num_heads * (head_dim / 2), 1, 1]
//!   local_work  = [0, 0, 0] (driver default)
//!
//! Each thread handles one pair (i0, i1) and writes both halves to x_out, so
//! every element of x_out is covered exactly once. No memcpy of un-rotated
//! elements is required.
//!
//! Constraints: `head_dim` must be even (kernel rotates pairs); rank must be
//! 2 or 3.

use crate::args::{ArgSpec, MemoryObjectSpec, OpDescriptor, OpError, OpImplLayout};
use crate::qnn::{
    Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, Qnn_OpConfigV1_t, Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
};

pub static DESCRIPTOR: OpDescriptor = OpDescriptor {
    op_type: "CustomRope",
    kernel_name: "kernel_rope_simple_oop",
    kernel_source: include_str!("../../../../engine/kernels/simple_ops.cl"),
    build_options: "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math",
    build_layout,
};

/// Read a scalar `INT_32` op_config param by name. Returns `None` if absent.
fn read_param_int(v1: &Qnn_OpConfigV1_t, name: &str) -> Option<i32> {
    if v1.params.is_null() || v1.numOfParams == 0 {
        return None;
    }
    for i in 0..v1.numOfParams {
        let p = unsafe { *v1.params.add(i as usize) };
        if p.paramType != Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR || p.name.is_null() {
            continue;
        }
        let pname = match unsafe { std::ffi::CStr::from_ptr(p.name).to_str() } {
            Ok(s) => s,
            Err(_) => continue,
        };
        if pname == name {
            // SAFETY: scalar param's union active variant is determined by
            // dataType. We accept INT_32 only; other widths are ignored.
            return Some(unsafe { p.__bindgen_anon_1.scalarParam.__bindgen_anon_1.int32Value });
        }
    }
    None
}

/// Read a scalar `FLOAT_32` op_config param by name. Returns `None` if absent.
fn read_param_float(v1: &Qnn_OpConfigV1_t, name: &str) -> Option<f32> {
    if v1.params.is_null() || v1.numOfParams == 0 {
        return None;
    }
    for i in 0..v1.numOfParams {
        let p = unsafe { *v1.params.add(i as usize) };
        if p.paramType != Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR || p.name.is_null() {
            continue;
        }
        let pname = match unsafe { std::ffi::CStr::from_ptr(p.name).to_str() } {
            Ok(s) => s,
            Err(_) => continue,
        };
        if pname == name {
            return Some(unsafe { p.__bindgen_anon_1.scalarParam.__bindgen_anon_1.floatValue });
        }
    }
    None
}

pub(crate) fn build_layout(v1: &Qnn_OpConfigV1_t) -> Result<OpImplLayout, OpError> {
    if v1.numOfInputs < 1 || v1.numOfOutputs < 1 {
        return Err(OpError::ValidationFailure);
    }

    let in_x = unsafe { (*v1.inputTensors.add(0)).__bindgen_anon_1.v1 };
    let out_x = unsafe { (*v1.outputTensors.add(0)).__bindgen_anon_1.v1 };

    if in_x.dimensions.is_null() || (in_x.rank != 2 && in_x.rank != 3) {
        return Err(OpError::InvalidArgument);
    }
    if out_x.dimensions.is_null() || out_x.rank == 0 {
        return Err(OpError::InvalidArgument);
    }

    // Rank-flexible:
    //   rank 3 [seq_len, num_heads, head_dim]
    //   rank 2 [seq_len, num_heads * head_dim]   (params provide split)
    let (seq_len, num_heads, head_dim) = if in_x.rank == 3 {
        let s = unsafe { *in_x.dimensions };
        let nh = unsafe { *in_x.dimensions.add(1) };
        let hd = unsafe { *in_x.dimensions.add(2) };
        (s, nh, hd)
    } else {
        // rank 2 — derive head_dim/num_heads from op params.
        let s = unsafe { *in_x.dimensions };
        let flat = unsafe { *in_x.dimensions.add(1) };
        let nh = read_param_int(v1, "num_heads")
            .filter(|v| *v > 0)
            .ok_or(OpError::ValidationFailure)? as u32;
        let hd = read_param_int(v1, "head_dim")
            .filter(|v| *v > 0)
            .ok_or(OpError::ValidationFailure)? as u32;
        if nh.checked_mul(hd) != Some(flat) {
            return Err(OpError::InvalidArgument);
        }
        (s, nh, hd)
    };

    if seq_len == 0 || num_heads == 0 || head_dim == 0 {
        return Err(OpError::InvalidArgument);
    }
    if head_dim % 2 != 0 {
        return Err(OpError::ValidationFailure);
    }

    let total = seq_len
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or(OpError::InvalidArgument)?;

    // Optional params; defaults match Llama 3.2 (start_pos=0, theta=10000.0).
    let start_pos = read_param_int(v1, "start_pos").unwrap_or(0);
    let theta = read_param_float(v1, "theta").unwrap_or(10000.0);

    let half_dim = head_dim / 2;
    let pairs = seq_len
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(half_dim))
        .ok_or(OpError::InvalidArgument)?;

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
        ArgSpec::InputTensor(0),       // x_in (read-only)
        ArgSpec::OutputTensor(0),      // x_out (write-only)
        ArgSpec::Int(head_dim as i32), // head_dim
        ArgSpec::Int(num_heads as i32),
        ArgSpec::Int(seq_len as i32),
        ArgSpec::Int(start_pos),
        ArgSpec::Float(theta),
    ];

    Ok(OpImplLayout {
        mem_objects,
        args,
        output_claims: None,
        global_work_dim: 1,
        local_work_dim: 0,
        global_work: [pairs as usize, 1, 1],
        local_work: [0, 0, 0],
    })
}
