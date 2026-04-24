//! Swappable weight slot infrastructure (Phase 1 / Phase 2).
//!
//! Introduces `LayerSlot`, `LayerWeights`, `SecondaryMmap`, and
//! `SwapExecutor` so Phase 2 can swap decoder layer weights atomically at
//! runtime. The `TransformerWeights` container (ENG-DAT-093) was removed in
//! Stage 2 cleanup — its fields are now flat on `TransformerModel` directly.
//!
//! Spec: ENG-DAT-090/092/094, ENG-ALG-210/211, INV-123/124/125.

pub mod secondary_mmap;
pub mod slot;
pub mod swap_executor;

pub use secondary_mmap::{
    LayerTensorSlice, LoadError, SecondaryMmap, SecondaryTensorInfo, open_secondary,
};
pub use slot::{LayerSlot, LayerWeights};
pub use swap_executor::{SwapError, SwapExecutor, SwapReport, SwappedLayer};
