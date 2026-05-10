//! Op descriptor registry.
//!
//! Ops are appended here from M1.3 onward (Add, MatMul, RMSNorm, RoPE, Softmax).
//! M1.2 leaves the slice empty — the registry / dispatcher infra still works
//! end-to-end; QNN simply sees zero ops in `pkg_get_info`.

use crate::args::OpDescriptor;

pub static OPS: &[OpDescriptor] = &[
    crate::ops::add::DESCRIPTOR,
    crate::ops::bias_add::DESCRIPTOR,
    crate::ops::matmul_f16_f32::DESCRIPTOR,
    crate::ops::rms_norm::DESCRIPTOR,
    crate::ops::softmax::DESCRIPTOR,
    crate::ops::silu_mul::DESCRIPTOR,
    crate::ops::rope::DESCRIPTOR,
    crate::ops::deq_q40::DESCRIPTOR,
    crate::ops::matmul_q40_f32::DESCRIPTOR,
    crate::ops::kv_scatter::DESCRIPTOR,
    crate::ops::flash_attn::DESCRIPTOR,
];

/// Open-Closed dispatcher (ENG-QNN-024). No if-else chain on op_type.
pub fn find_descriptor(op_type: &str) -> Option<&'static OpDescriptor> {
    OPS.iter().find(|d| d.op_type == op_type)
}
