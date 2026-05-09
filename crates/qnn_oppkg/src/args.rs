//! Op argument / layout abstraction.
//!
//! Each registered op declares an `OpDescriptor` whose `build_layout` callback
//! converts a `Qnn_OpConfigV1_t` (per-node configuration the QNN runtime hands us)
//! into a backend-neutral `OpImplLayout`. `op_impl::build_op_state` then turns
//! that layout into the concrete `QnnGpu_*` structs the runtime expects.
//!
//! Keeping the abstraction backend-neutral lets unit tests exercise descriptor
//! metadata and arg encoding on the host without an OpenCL device.

use crate::qnn::{Qnn_DataType_t, Qnn_OpConfigV1_t};

/// Top-level descriptor for a registered op. Stored as `&'static` in
/// `registry::OPS`.
#[derive(Copy, Clone)]
pub struct OpDescriptor {
    pub op_type: &'static str,
    pub kernel_name: &'static str,
    pub kernel_source: &'static str,
    pub build_options: &'static str,
    pub build_layout: fn(&Qnn_OpConfigV1_t) -> Result<OpImplLayout, OpError>,
}

/// Layout produced by a descriptor's `build_layout` for one node invocation.
pub struct OpImplLayout {
    /// Memory objects in tensor order. Last entry is the op's output (claimed).
    pub mem_objects: Vec<MemoryObjectSpec>,
    /// Kernel arguments in declaration order.
    pub args: Vec<ArgSpec>,
    /// Number of dimensions used in `global_work` / `local_work`.
    pub global_work_dim: u32,
    pub local_work_dim: u32,
    pub global_work: [usize; 3],
    pub local_work: [usize; 3],
}

/// Description of a single QNN GPU memory object (tensor view).
pub struct MemoryObjectSpec {
    pub data_type: Qnn_DataType_t,
    pub flat_dims: Vec<u32>,
    pub flat_offsets: Vec<u32>,
}

/// One kernel argument. `InputTensor` / `OutputTensor` carry the runtime tensor
/// index that QNN passes through `Qnn_OpConfigV1_t::{inputTensors, outputTensors}`.
/// `InOutTensor` is the in-place variant — kernel reads and writes the same
/// input tensor (maps to `QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READWRITE`). Used by
/// `CustomSiluMul` (M1.7) where the production kernel mutates `x` in-place.
pub enum ArgSpec {
    InputTensor(u32),
    OutputTensor(u32),
    InOutTensor(u32),
    LocalMem(usize),
    Int(i32),
    UInt(u32),
    ULong(u64),
    Float(f32),
}

/// Errors a descriptor's `build_layout` may surface back to QNN.
#[derive(Debug)]
pub enum OpError {
    InvalidArgument,
    ValidationFailure,
}
