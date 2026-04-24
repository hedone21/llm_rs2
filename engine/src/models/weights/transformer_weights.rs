//! `TransformerWeights` — root container for swap-aware model weights.
//!
//! Holds the `Vec<LayerSlot>` for all decoder layers plus the cross-layer
//! tensors (embedding, final_norm, lm_head) that are never swapped
//! (ENG-DAT-C11). Also retains the shared `Arc<SecondaryMmap>` so INV-125 is
//! structurally enforced: the mmap outlives every `LayerSlot` that references
//! it.
//!
//! Spec: ENG-DAT-093 (TransformerWeights), INV-125 (secondary mmap lifetime).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::tensor::Tensor;

use super::secondary_mmap::SecondaryMmap;
use super::slot::LayerSlot;

/// Root container owning all model weights behind swap-aware slots.
pub struct TransformerWeights {
    /// One `LayerSlot` per decoder block. Length = num decoder layers.
    pub layers: Vec<LayerSlot>,
    /// Token embedding (never swapped, ENG-DAT-C11).
    pub embedding: Arc<Tensor>,
    /// Final RMSNorm weight (never swapped).
    pub final_norm: Arc<Tensor>,
    /// LM head projection. `None` when `tie_word_embeddings == true` and
    /// forward reuses `embedding` for the output projection.
    pub lm_head: Option<Arc<Tensor>>,
    /// Secondary mmap handle, shared with every `LayerSlot::secondary_mmap_handle`.
    /// Kept here as the "last keeper" so INV-125 (lifetime) holds even if all
    /// slot handles are individually dropped during refactors.
    pub secondary_mmap: Option<Arc<SecondaryMmap>>,
    /// Global swap generation counter. Bumped once per `SwapExecutor` batch
    /// (Phase 2). Phase 1 leaves it at 0 — declared only.
    pub ratio_generation: Arc<AtomicU64>,
}

impl TransformerWeights {
    /// Construct a fully-assembled weight container.
    pub fn new(
        layers: Vec<LayerSlot>,
        embedding: Arc<Tensor>,
        final_norm: Arc<Tensor>,
        lm_head: Option<Arc<Tensor>>,
        secondary_mmap: Option<Arc<SecondaryMmap>>,
    ) -> Self {
        Self {
            layers,
            embedding,
            final_norm,
            lm_head,
            secondary_mmap,
            ratio_generation: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Number of decoder layers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Current value of the global swap-generation counter.
    #[inline]
    pub fn ratio_generation(&self) -> u64 {
        self.ratio_generation.load(Ordering::Acquire)
    }
}
