//! `CustomRmsNorm` op descriptor — out-of-place RMSNorm (Llama-style).
//!
//! D-D.6: maps to `kernel_rms_norm_oop_subgroup` (single-subgroup, OOP variant).
//! production OpenCL primary가 사용하는 `kernel_rms_norm_opt`의 numerical-
//! equivalent reduction order로 byte-equal 결과 보장.
//!
//! ## Background
//! M1.5는 single-thread `kernel_rms_norm_simple`을 선택했지만, production
//! fallback path (`backend.rms_norm()` → `kernel_rms_norm_opt`)와 floating
//! point 비결합성으로 numerical 차이 발생. fast path가 활성화되면 layer 0의
//! RMS 결과가 fallback path와 미세하게 다르고 28-layer × multi-token decode
//! 누적으로 다른 token sequence 생성.
//!
//! `kernel_rms_norm_opt`는 LOCAL arg (SLM scratch)를 사용하는데 SDK가
//! OpPackage path에서 LOCAL을 validation 못해 `graphFinalize err=0x1786`
//! (`QNN_GRAPH_ERROR_FINALIZE_FAILED`) trigger. D-D.6 fix는 workgroup_size를
//! single subgroup (64 threads, REQD_SUBGROUP_SIZE_64)으로 강제하여 cross-
//! subgroup reduction을 제거 → SLM 불필요 → LOCAL arg 우회.
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
//! Workgroup (D-D.6, single-subgroup):
//!   global = [rows * 64, 1, 1]
//!   local  = [64, 1, 1]

use crate::args::{ArgSpec, MemoryObjectSpec, OpDescriptor, OpError, OpImplLayout};
use crate::qnn::{Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, Qnn_OpConfigV1_t};

const EPS: f32 = 1e-5;

pub static DESCRIPTOR: OpDescriptor = OpDescriptor {
    op_type: "CustomRmsNorm",
    // D-D.6: subgroup-reduce variant for byte-equal numerical match with
    // production `kernel_rms_norm_opt` (parallel reduction).
    kernel_name: "kernel_rms_norm_oop_subgroup",
    kernel_source: include_str!("../../../../engine/kernels/simple_ops.cl"),
    // D-D.6: production OpenCLBackend의 build flag와 byte-equal 통일
    // (`-cl-unsafe-math-optimizations -cl-finite-math-only` 추가). compiler가
    // 같은 reduction order / MAD fusion을 적용하도록 보장.
    build_options: "-cl-std=CL2.0 -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math",
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

    // D-D.6: single-subgroup workgroup. global = [rows * 64], local = [64].
    // Adreno에서 REQD_SUBGROUP_SIZE_64 attribute (qcom_reqd_sub_group_size("half"))로
    // sg_size=64 강제. workgroup이 1 subgroup이라 cross-subgroup reduction 불필요.
    const SG_SIZE: usize = 64;

    Ok(OpImplLayout {
        mem_objects,
        args,
        output_claims: None,
        global_work_dim: 1,
        local_work_dim: 1,
        global_work: [rows as usize * SG_SIZE, 1, 1],
        local_work: [SG_SIZE, 1, 1],
    })
}
