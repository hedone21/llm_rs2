//! Swappable weight slot infrastructure (Phase 1 / Phase 2 / Phase 3).
//!
//! Introduces `LayerSlot`, `LayerWeights`, `SecondaryMmap`, `SwapExecutor`,
//! and `QuantNoiseTable` so Phase 3 can compute per-layer quantization noise
//! factors and expose them to `WeightSwapDecider` (Stage B).
//!
//! Spec: ENG-DAT-090/092/094/095, ENG-ALG-210/211/216, INV-123/124/125/127.

pub mod decider;
pub mod noise_table;
pub mod secondary_mmap;
pub mod slot;
pub mod swap_executor;

pub use decider::{SwapDecision, WeightSwapDecider, compute_qcf_swap};
pub use noise_table::QuantNoiseTable;
pub use secondary_mmap::{
    LayerTensorSlice, LoadError, SecondaryMmap, SecondaryTensorInfo, open_secondary,
};
pub use slot::{LayerSlot, LayerWeights};
pub use swap_executor::{SwapError, SwapExecutor, SwapReport, SwappedLayer, dtype_tag_to_dtype};
