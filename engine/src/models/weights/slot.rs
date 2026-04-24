//! `LayerSlot` — atomic weight container for one decoder layer.
//!
//! Encapsulates one decoder block's weights behind an `ArcSwap` for lock-free
//! snapshot reads. Phase 1 only exercises the initial-install path; runtime
//! swap (`SwapExecutor::swap_layer`) lands in Phase 2.
//!
//! Spec: ENG-DAT-092 (`LayerSlot`), INV-123 (atomic snapshot), INV-124
//! (current_dtype <-> weights dtype consistency).

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use arc_swap::ArcSwap;

use crate::core::buffer::DType;
use crate::layers::transformer_layer::TransformerLayer;

use super::secondary_mmap::SecondaryMmap;

/// Phase 1 alias: one layer's weight bundle is the existing `TransformerLayer`
/// struct. Phase 2 may evolve this to a dedicated weight-only struct; for now
/// we reuse the proven layout to minimise the forward-path diff.
pub type LayerWeights = TransformerLayer;

/// Encode a `DType` into the `AtomicU8` discriminant stored by `LayerSlot`.
#[inline]
fn dtype_to_u8(dt: DType) -> u8 {
    match dt {
        DType::Q4_0 => 0,
        DType::Q4_1 => 1,
        DType::Q8_0 => 2,
        DType::F16 => 3,
        DType::BF16 => 4,
        DType::F32 => 5,
        DType::U8 => 6,
    }
}

#[inline]
fn u8_to_dtype(v: u8) -> DType {
    match v {
        0 => DType::Q4_0,
        1 => DType::Q4_1,
        2 => DType::Q8_0,
        3 => DType::F16,
        4 => DType::BF16,
        5 => DType::F32,
        6 => DType::U8,
        _ => panic!("LayerSlot: corrupt current_dtype discriminant {v}"),
    }
}

/// One decoder layer's swappable slot.
///
/// Readers (forward pass): `slot.load_weights()` → `Arc<LayerWeights>`
/// snapshot, usable for the remainder of the layer without further atomics.
///
/// Writers (`SwapExecutor`, Phase 2): `slot.swap_weights(new_arc, new_dtype)`
/// atomically installs a new snapshot and bumps `generation`.
pub struct LayerSlot {
    /// Current weight dtype. Updated in the same logical step as `weights` on
    /// swap (INV-124). Encoded as `AtomicU8` because `DType` is not atomic.
    current_dtype: AtomicU8,
    /// Wait-free snapshot of the layer weights.
    weights: ArcSwap<LayerWeights>,
    /// Optional secondary mmap handle shared with `TransformerWeights`. When
    /// `None`, this slot cannot be swapped. INV-125 requires that this handle
    /// remain alive for the slot's entire lifetime.
    secondary_mmap_handle: Option<Arc<SecondaryMmap>>,
    /// Per-slot generation counter. Incremented on each successful swap. Used
    /// by downstream plan invalidation analogous to INV-120.
    generation: AtomicU64,
}

impl LayerSlot {
    /// Build a new slot from an initial weight snapshot.
    pub fn new(
        weights: LayerWeights,
        dtype: DType,
        secondary_mmap_handle: Option<Arc<SecondaryMmap>>,
    ) -> Self {
        Self {
            current_dtype: AtomicU8::new(dtype_to_u8(dtype)),
            weights: ArcSwap::from(Arc::new(weights)),
            secondary_mmap_handle,
            generation: AtomicU64::new(0),
        }
    }

    /// Acquire a full Arc snapshot of the current weights. O(1) atomic load
    /// plus one Arc clone; the returned handle is safe to use for the rest of
    /// the current layer's forward pass even if a swap lands concurrently.
    #[inline]
    pub fn load_weights(&self) -> Arc<LayerWeights> {
        self.weights.load_full()
    }

    /// Current dtype (INV-124).
    #[inline]
    pub fn current_dtype(&self) -> DType {
        u8_to_dtype(self.current_dtype.load(Ordering::Acquire))
    }

    /// Current generation counter value.
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Reference to the secondary mmap handle (None if swap disabled for this
    /// slot).
    #[inline]
    pub fn secondary_mmap_handle(&self) -> Option<&Arc<SecondaryMmap>> {
        self.secondary_mmap_handle.as_ref()
    }

    /// Phase 2 seam — atomic weight swap.
    ///
    /// Replaces the weight snapshot, updates `current_dtype`, and bumps
    /// `generation`. INV-124 requires the dtype update and snapshot install to
    /// occur within one logical step; we use Release ordering so any reader
    /// observing the new generation sees both the new Arc and the new dtype.
    ///
    /// Phase 1 does not call this path; it is provided for downstream
    /// component wiring and unit tests.
    pub fn swap_weights(&self, new_weights: Arc<LayerWeights>, new_dtype: DType) {
        self.current_dtype
            .store(dtype_to_u8(new_dtype), Ordering::Release);
        self.weights.store(new_weights);
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// Install fresh weights while preserving the current generation counter.
    /// Used for in-place rewrap flows (e.g. `rewrap_weights_for_dual_access`)
    /// that need the same dtype but a new backing buffer.
    pub fn store_weights_same_dtype(&self, new_weights: Arc<LayerWeights>) {
        self.weights.store(new_weights);
    }

    /// Perform an in-place closure update of the weight snapshot under RCU
    /// semantics. Used during partition (re)configuration: the closure
    /// receives a reference to the current `Arc<LayerWeights>`, clones the
    /// inner struct, mutates it, and the ArcSwap installs the new Arc
    /// atomically.
    ///
    /// On contention the closure may be invoked more than once, so it must be
    /// idempotent / side-effect free.
    pub fn rcu_weights<F>(&self, mut f: F)
    where
        F: FnMut(&LayerWeights) -> LayerWeights,
    {
        self.weights.rcu(|current| Arc::new(f(current.as_ref())));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_roundtrip_covers_all_variants() {
        for dt in [
            DType::Q4_0,
            DType::Q4_1,
            DType::Q8_0,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::U8,
        ] {
            assert_eq!(u8_to_dtype(dtype_to_u8(dt)), dt);
        }
    }
}
