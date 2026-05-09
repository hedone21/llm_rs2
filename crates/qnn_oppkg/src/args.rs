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
    /// Memory objects in tensor order. Default behaviour: the **last** entry is
    /// the op's claimed output. Override with `output_mem_indices` (multi-output)
    /// or `OutputTensorAliased(input_idx)` (alias an input as the output edge).
    pub mem_objects: Vec<MemoryObjectSpec>,
    /// Kernel arguments in declaration order.
    pub args: Vec<ArgSpec>,
    /// Optional explicit output claim list (M2.H multi-output extension).
    ///
    /// Each entry is a `(output_index, mem_object_index)` pair: `output_index`
    /// is the position in the graph node's `outputTensors` array (0..N-1),
    /// `mem_object_index` is the row in `mem_objects` whose buffer backs that
    /// output. When `Some`, `build_op_state` builds one `QnnGpu_OutputClaim_t`
    /// per entry. When `None`, fallback to the legacy single-output rule
    /// (claim the last mem_object, or the alias target of `OutputTensorAliased`).
    pub output_claims: Option<Vec<OutputClaimSpec>>,
    /// Number of dimensions used in `global_work` / `local_work`.
    pub global_work_dim: u32,
    pub local_work_dim: u32,
    pub global_work: [usize; 3],
    pub local_work: [usize; 3],
}

/// One entry of `OpImplLayout::output_claims`. See its docstring for semantics.
#[derive(Copy, Clone)]
pub struct OutputClaimSpec {
    pub output_index: u32,
    pub mem_object_index: u32,
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
///
/// `OutputTensorAliased(input_idx)` is the chain-safe in-place variant (M2.G,
/// INV-164): from the graph topology view this op produces an output edge —
/// QNN registers `outputs[0]` so downstream nodes can consume it — but at the
/// kernel-arg level the buffer aliases `inputs[input_idx]`. `build_op_state`
/// points `out_claim.memoryObject` at `mem_objects[input_idx]` so the SDK
/// learns the alias. The kernel-arg type stays `OP_INPUT_READWRITE` and the
/// `tensorIndex` is `input_idx`. Used by `CustomSiluMul` to make the in-place
/// silu+mul chain-composable: when the SDK accepts the alias, downstream ops
/// (e.g. another MatMul fed by SiluMul's output) read the mutated buffer
/// directly. Falls back to YELLOW (Tier 2 split) if the SDK rejects the alias.
pub enum ArgSpec {
    InputTensor(u32),
    OutputTensor(u32),
    InOutTensor(u32),
    OutputTensorAliased(u32),
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
