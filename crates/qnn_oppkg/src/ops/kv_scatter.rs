//! `CustomKvScatter` op descriptor — F32→F16 cast + HeadMajor scatter.
//!
//! Maps to `kernel_kv_scatter_f32_to_f16_oop` in `engine/kernels/simple_ops.cl`.
//!
//! ## Multi-output design (M2.H, 2026-05-09)
//!
//! Earlier revisions (Tier 1) routed `k_dst` through `InOutTensor(2)` and only
//! claimed `v_dst` as the single output, working around the M2.C/`deq_q40.rs`
//! single-output abstraction. The M2.H investigation showed in-place patterns
//! break SDK chain composition (B1' bisect: OOP-only chain GREEN, in-place
//! tail RED). The `op_impl::build_op_state` abstraction now supports explicit
//! multi-`OutputClaim` via `OpImplLayout::output_claims`, so KvScatter can
//! declare both `k_dst` and `v_dst` as proper output edges.
//!
//! ## pos-as-buffer rationale (M3.4 D-D.1, 2026-05-10)
//!
//! Earlier the descriptor read `write_pos` as a SCALAR op param (INT_32). QNN
//! bakes SCALAR params into the graph at `graphFinalize` time, which made
//! multi-token decode impossible — every token would scatter into the same
//! cache slot. The fix: introduce a 3rd input tensor `pos_buf : INT_32 [1]`
//! that the kernel reads as `pos_buf[0]`. The graph caller writes the current
//! decode position into `pos_buf` before each `graphExecute`. `head_dim` and
//! `capacity` remain SCALAR (build-time-constant per layer/model).
//!
//! Note: this descriptor uses the OOP variant kernel
//! (`kernel_kv_scatter_f32_to_f16_oop`). The original
//! `kernel_kv_scatter_f32_to_f16` is preserved for the OpenCL backend in
//! `engine/src/backend/opencl/{mod,plan}.rs`, which sets `write_pos` as a
//! per-call kernel arg directly (no graph-finalize barrier).
//!
//! Kernel signature:
//!   kernel void kernel_kv_scatter_f32_to_f16_oop(
//!       global const float * k_src,   // arg 0: InputTensor(0)
//!       global const float * v_src,   // arg 1: InputTensor(1)
//!       global const int   * pos_buf, // arg 2: InputTensor(2) — pos_buf[0] = write_pos
//!       global half * k_dst,          // arg 3: OutputTensor(0)
//!       global half * v_dst,          // arg 4: OutputTensor(1)
//!       int head_dim,                 // arg 5
//!       int capacity                  // arg 6
//!   );
//!
//! Tensor convention:
//!   inputs[0]  = k_src    F32 [kv_heads * head_dim]                  FLOAT_32
//!   inputs[1]  = v_src    F32 [kv_heads * head_dim]                  FLOAT_32
//!   inputs[2]  = pos_buf  I32 [1]                                    INT_32
//!   outputs[0] = k_dst    F16 [kv_heads * capacity * head_dim]       FLOAT_16
//!   outputs[1] = v_dst    F16 [kv_heads * capacity * head_dim]       FLOAT_16
//!
//! Op params (numOfParams ≤ 2):
//!   params[i].name = "head_dim"  scalar INT_32
//!   params[i].name = "capacity"  scalar INT_32
//!
//! Layout:
//!   mem_objects[0] = k_src   FLOAT_32 [kv_heads * head_dim]
//!   mem_objects[1] = v_src   FLOAT_32 [kv_heads * head_dim]
//!   mem_objects[2] = pos_buf INT_32   [1]
//!   mem_objects[3] = k_dst   FLOAT_16 [kv_heads * capacity * head_dim]  ← output[0]
//!   mem_objects[4] = v_dst   FLOAT_16 [kv_heads * capacity * head_dim]  ← output[1]
//!
//! Args (7 total, matching kernel_kv_scatter_f32_to_f16_oop declaration order):
//!   InputTensor(0)  → k_src
//!   InputTensor(1)  → v_src
//!   InputTensor(2)  → pos_buf
//!   OutputTensor(0) → k_dst
//!   OutputTensor(1) → v_dst
//!   Int(head_dim)
//!   Int(capacity)
//!
//! Output claims: both k_dst (mem 3) and v_dst (mem 4) are explicit
//! `QnnGpu_OutputClaim_t` entries via `OpImplLayout::output_claims`.
//!
//! Workgroup:
//!   global = [kv_heads * head_dim, 1, 1]
//!   local  = [0, 0, 0] (driver default)

use crate::args::{
    ArgSpec, MemoryObjectSpec, OpDescriptor, OpError, OpImplLayout, OutputClaimSpec,
};
use crate::qnn::{
    Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
    Qnn_DataType_t_QNN_DATATYPE_INT_32, Qnn_OpConfigV1_t, Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
};

pub static DESCRIPTOR: OpDescriptor = OpDescriptor {
    op_type: "CustomKvScatter",
    kernel_name: "kernel_kv_scatter_f32_to_f16_oop",
    kernel_source: include_str!("../../../../engine/kernels/simple_ops.cl"),
    build_options: "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math",
    build_layout,
};

/// Read a scalar `INT_32` op param by name.
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
            return Some(unsafe { p.__bindgen_anon_1.scalarParam.__bindgen_anon_1.int32Value });
        }
    }
    None
}

pub(crate) fn build_layout(v1: &Qnn_OpConfigV1_t) -> Result<OpImplLayout, OpError> {
    // Require: 3 inputs (k_src, v_src, pos_buf), 2 outputs (k_dst, v_dst).
    if v1.numOfInputs < 3 || v1.numOfOutputs < 2 {
        return Err(OpError::ValidationFailure);
    }

    // k_src: rank 1 [kv_heads * head_dim] or rank 2 [kv_heads, head_dim]
    let k_src_t = unsafe { (*v1.inputTensors.add(0)).__bindgen_anon_1.v1 };
    if k_src_t.dimensions.is_null() || k_src_t.rank == 0 {
        return Err(OpError::InvalidArgument);
    }

    // pos_buf: rank ≥ 1, flat element count = 1.
    let pos_t = unsafe { (*v1.inputTensors.add(2)).__bindgen_anon_1.v1 };
    if pos_t.dimensions.is_null() || pos_t.rank == 0 {
        return Err(OpError::InvalidArgument);
    }
    let mut pos_total: u32 = 1;
    for i in 0..pos_t.rank {
        let d = unsafe { *pos_t.dimensions.add(i as usize) };
        pos_total = pos_total.checked_mul(d).ok_or(OpError::InvalidArgument)?;
    }
    if pos_total != 1 {
        return Err(OpError::InvalidArgument);
    }

    // Derive src_total = kv_heads * head_dim from k_src flat element count.
    let mut src_total: u32 = 1;
    for i in 0..k_src_t.rank {
        let d = unsafe { *k_src_t.dimensions.add(i as usize) };
        src_total = src_total.checked_mul(d).ok_or(OpError::InvalidArgument)?;
    }
    if src_total == 0 {
        return Err(OpError::InvalidArgument);
    }

    // capacity, head_dim from op params (write_pos now lives in pos_buf).
    let capacity = read_param_int(v1, "capacity").ok_or(OpError::ValidationFailure)?;
    let head_dim = read_param_int(v1, "head_dim").ok_or(OpError::ValidationFailure)?;

    if std::env::var("QNN_OPPKG_DEBUG").as_deref() == Ok("1") {
        let dim_dump = |idx: u32, is_out: bool| -> String {
            let t = if is_out {
                unsafe { (*v1.outputTensors.add(idx as usize)).__bindgen_anon_1.v1 }
            } else {
                unsafe { (*v1.inputTensors.add(idx as usize)).__bindgen_anon_1.v1 }
            };
            if t.dimensions.is_null() || t.rank == 0 {
                return "<null>".to_string();
            }
            let mut s = String::from("[");
            for i in 0..t.rank {
                if i > 0 {
                    s.push(',');
                }
                let d = unsafe { *t.dimensions.add(i as usize) };
                s.push_str(&d.to_string());
            }
            s.push(']');
            s
        };
        eprintln!(
            "[oppkg-kvs] params: head_dim={} capacity={} src_total={} (write_pos = pos_buf[0])",
            head_dim, capacity, src_total
        );
        eprintln!(
            "[oppkg-kvs] in dims: k_src={} v_src={} pos_buf={} | out: k_dst={} v_dst={}",
            dim_dump(0, false),
            dim_dump(1, false),
            dim_dump(2, false),
            dim_dump(0, true),
            dim_dump(1, true),
        );
    }

    if capacity <= 0 || head_dim <= 0 {
        return Err(OpError::InvalidArgument);
    }
    if src_total == 0 || (src_total as i32) % head_dim != 0 {
        return Err(OpError::InvalidArgument);
    }
    let kv_heads = src_total / (head_dim as u32);
    let dst_total = kv_heads
        .checked_mul(capacity as u32)
        .and_then(|v| v.checked_mul(head_dim as u32))
        .ok_or(OpError::InvalidArgument)?;

    let mem_objects = vec![
        // k_src F32
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![src_total],
            flat_offsets: vec![0u32],
        },
        // v_src F32
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![src_total],
            flat_offsets: vec![0u32],
        },
        // pos_buf INT_32 [1]
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_INT_32,
            flat_dims: vec![1u32],
            flat_offsets: vec![0u32],
        },
        // k_dst F16 (claimed output 0)
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
            flat_dims: vec![dst_total],
            flat_offsets: vec![0u32],
        },
        // v_dst F16 (claimed output 1)
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
            flat_dims: vec![dst_total],
            flat_offsets: vec![0u32],
        },
    ];

    // Args match kernel_kv_scatter_f32_to_f16_oop declaration order (7 total).
    let args = vec![
        ArgSpec::InputTensor(0),  // k_src
        ArgSpec::InputTensor(1),  // v_src
        ArgSpec::InputTensor(2),  // pos_buf
        ArgSpec::OutputTensor(0), // k_dst
        ArgSpec::OutputTensor(1), // v_dst
        ArgSpec::Int(head_dim),
        ArgSpec::Int(capacity),
    ];

    // Multi-output explicit claims: outputs[0] ↔ mem_objects[3] (k_dst),
    // outputs[1] ↔ mem_objects[4] (v_dst). See `OpImplLayout::output_claims`.
    let output_claims = vec![
        OutputClaimSpec {
            output_index: 0,
            mem_object_index: 3,
        },
        OutputClaimSpec {
            output_index: 1,
            mem_object_index: 4,
        },
    ];

    Ok(OpImplLayout {
        mem_objects,
        args,
        output_claims: Some(output_claims),
        global_work_dim: 1,
        local_work_dim: 0,
        global_work: [src_total as usize, 1, 1],
        local_work: [0, 0, 0],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layout(
        kv_heads: u32,
        head_dim: i32,
        capacity: i32,
        write_pos: i32,
    ) -> Result<OpImplLayout, OpError> {
        crate::__test_support::build_kv_scatter_layout_for(kv_heads, head_dim, capacity, write_pos)
    }

    #[test]
    fn kv_scatter_descriptor_op_type() {
        assert_eq!(DESCRIPTOR.op_type, "CustomKvScatter");
    }

    #[test]
    fn kv_scatter_kernel_source_contains_target() {
        assert!(
            DESCRIPTOR
                .kernel_source
                .contains("kernel_kv_scatter_f32_to_f16_oop"),
            "kernel source must contain 'kernel_kv_scatter_f32_to_f16_oop'"
        );
    }

    #[test]
    fn kv_scatter_build_layout_for_qwen() {
        // kv_heads=2, head_dim=128, capacity=2048, write_pos=100 (write_pos
        // is now ignored at build_layout — supplied via pos_buf at execute).
        let layout = make_layout(2, 128, 2048, 100).expect("build_layout must succeed");
        assert_eq!(layout.args.len(), 7, "must have 7 kernel args");
        assert_eq!(layout.mem_objects.len(), 5, "must have 5 mem_objects");
        // global_work = kv_heads * head_dim = 2 * 128 = 256
        assert_eq!(layout.global_work[0], 256);
        assert_eq!(layout.global_work[1], 1);
        assert_eq!(layout.global_work[2], 1);
        // multi-output: 2 explicit claims pointing at mem 3 (k_dst) and 4 (v_dst).
        let claims = layout
            .output_claims
            .as_ref()
            .expect("output_claims must be Some for multi-output");
        assert_eq!(claims.len(), 2);
        assert_eq!(claims[0].output_index, 0);
        assert_eq!(claims[0].mem_object_index, 3);
        assert_eq!(claims[1].output_index, 1);
        assert_eq!(claims[1].mem_object_index, 4);
    }
}
