//! Swappable weight slot infrastructure (Phase 1).
//!
//! Introduces `LayerSlot`, `LayerWeights`, `SecondaryMmap`, and the
//! `TransformerWeights` container so Phase 2 (`SwapExecutor`) can swap
//! decoder layer weights atomically at runtime. Phase 1 itself only
//! exercises the static initial-install path.
//!
//! Spec: ENG-DAT-090/092/093/094, ENG-ALG-210, INV-123/124/125.

pub mod secondary_mmap;
pub mod slot;
pub mod swap_executor;
pub mod transformer_weights;

pub use secondary_mmap::{
    LayerTensorSlice, LoadError, SecondaryMmap, SecondaryTensorInfo, open_secondary,
};
pub use slot::{LayerSlot, LayerWeights};
pub use swap_executor::{SwapError, SwapExecutor, SwapReport, SwappedLayer};
pub use transformer_weights::TransformerWeights;
