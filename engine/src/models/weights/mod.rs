//! Swappable weight slot infrastructure (Phase 1 / Phase 2 / Phase 3).
//!
//! Introduces `LayerSlot`, `LayerWeights`, `SecondaryMmap`, `SwapExecutor`,
//! and `QuantNoiseTable` so Phase 3 can compute per-layer quantization noise
//! factors and expose them to `WeightSwapDecider` (Stage B).
//!
//! `async_swap` provides the `AsyncSwapDispatcher` worker thread that commits
//! layer weights off the critical path (LISWAP-2 prototype).
//!
//! Spec: ENG-DAT-090/092/094/095, ENG-ALG-210/211/216, INV-123/124/125/127.

pub mod async_swap;
pub mod decider;
pub mod dynamic_k;
pub mod incremental_plan;
pub mod intra_forward_swap;
#[cfg(feature = "cuda-embedded")]
pub mod layer_object_pool;
pub mod noise_table;
pub mod phase_aware_swap;
pub mod probing_k;
pub mod release_worker;
pub mod rpcmem_secondary;
pub mod secondary_mmap;
pub mod slot;
pub mod swap_executor;

pub use async_swap::{AsyncSwapDispatcher, SwapCommitJob, SwapJob};
pub use decider::{SwapAlgorithm, SwapDecision, WeightSwapDecider, compute_qcf_weight_swap};
pub use dynamic_k::DynamicKController;
pub use incremental_plan::IncrementalSwapPlan;
pub use intra_forward_swap::{
    IntraForwardSwapHook, IntraForwardSwapPlan, LayerBoundaryHook, NoOpHook,
};
pub use noise_table::QuantNoiseTable;
pub use phase_aware_swap::{PhaseAwareSwapDispatcher, WeightChunk};
pub use probing_k::{GrowthMode, ProbingKController};
pub use release_worker::PrimaryReleaseWorker;
pub use secondary_mmap::{
    LayerTensorSlice, LoadError, SecondaryDtypeChoice, SecondaryLayoutChoice, SecondaryMmap,
    SecondaryTensorInfo, build_auf_secondary_from_view, open_secondary, open_secondary_auf,
    open_secondary_with_backend, open_secondary_with_dtype, open_secondary_with_options,
};
pub use slot::{LayerSlot, LayerWeights};
pub use swap_executor::{
    StageBreakdown, SwapError, SwapExecutor, SwapReport, SwappedLayer, dtype_tag_to_dtype,
};
