//! Secondary storage abstractions for weight swap.
//!
//! Defines [`SecondaryStore`]: common byte-access surface shared by L3 weight
//! secondary stores (`SecondaryMmap` enum, `RpcmemLayerRegion`). The existing
//! trait [`crate::memory::secondary::SecondaryMmapBytes`] covers the
//! `cuMemHostRegister` byte span path (Step 3-E, V-09); `SecondaryStore` is
//! the broader companion (impl + call site merge TBD with benchmark evidence).

/// Common byte-access surface for L3 secondary weight stores.
///
/// Stage 1: declared but unused. Stage 2 will add `impl SecondaryStore for
/// SecondaryMmap` and `impl SecondaryStore for RpcmemLayerRegion` plus
/// migrate the few call sites that currently match on the enum variant.
pub trait SecondaryStore: Send + Sync {
    /// Raw byte view of the secondary store.
    ///
    /// Implementors that do not back a contiguous byte span (e.g. per-layer
    /// rpcmem region collection) should return an empty slice and rely on
    /// secondary-store-specific APIs for per-region access.
    fn as_bytes(&self) -> &[u8];

    /// Total byte length covered by [`SecondaryStore::as_bytes`].
    fn len(&self) -> usize;

    /// Returns `true` when [`SecondaryStore::as_bytes`] is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
