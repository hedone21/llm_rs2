//! Generic L2 buffer views.
//!
//! - [`slice`]: `SliceBuffer` — typed offset view over any parent `Arc<dyn Buffer>`.
//!
//! Memory-resource-specific allocators and view adapters live in
//! `crate::memory::<resource>/`. See ARCHITECTURE.md §13.8-D.
pub mod slice;
