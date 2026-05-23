//! Secondary storage abstractions for weight swap + LISWAP-6 rpcmem alias.
//!
//! B-5b Phase 2 Stage 1 (infrastructure-only). Introduces two trait sketches
//! that Stage 2 will use to remove the remaining L1↔L2 downcasts in
//! `qnn_oppkg::with_opencl_secondary` and the rpcmem alias path:
//!
//! - [`SecondaryStore`]: common byte-access surface shared by L3 weight
//!   secondary stores (`SecondaryMmap` enum, `RpcmemLayerRegion`). The
//!   existing trait [`crate::memory::secondary::SecondaryMmapBytes`] already
//!   covers the `cuMemHostRegister` byte span path (Step 3-E, V-09); the
//!   intent of `SecondaryStore` is broader (Stage 2 will decide whether to
//!   collapse the two — for Stage 1 we keep a thin parallel trait so the
//!   Stage 2 senior-implementer can pick the merge direction with
//!   benchmark evidence).
//!
//! - [`OpenClSecondary`]: capability returned by
//!   `Backend::as_opencl_secondary()` so callers (currently the QNN OpPackage
//!   `with_opencl_secondary` closure) can reach an OpenCL secondary without
//!   `as_any().downcast_ref::<OpenCLBackend>()`. The trait body is
//!   intentionally empty in Stage 1 — Stage 2 will populate it with the
//!   minimal closure surface needed by `qnn_oppkg/mod.rs:134` and
//!   `qnn_oppkg/mod.rs:140` (currently a `FnOnce(&OpenCLBackend) -> R`
//!   pattern; an `enqueue_write_via` / `queue_for_swap` method pair is the
//!   likely shape).

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

/// OpenCL-backed secondary capability. Implemented by `OpenCLBackend` so a
/// `Backend` trait object can hand out a thin OpenCL handle without the
/// `as_any().downcast_ref::<OpenCLBackend>()` chain used today by the QNN
/// OpPackage backend.
///
/// Stage 1: trait body intentionally empty (placeholder). Stage 2 will add
/// the closure / method surface required by `qnn_oppkg::with_opencl_secondary`
/// (current pattern: `f: impl FnOnce(&OpenCLBackend) -> R`). Until then the
/// `Backend::as_opencl_secondary` default returns `None` and the existing
/// downcast path remains in effect.
#[cfg(feature = "opencl")]
pub trait OpenClSecondary: Send + Sync {
    // Stage 2 will populate. See module docs.
}
