//! Layer-level QNN graph metadata helpers (M2.H).
//!
//! `crates/qnn_oppkg/src/graph/` holds helpers used by host-side QNN graph
//! builders to describe full Transformer-style layers as a single
//! `Qnn_GraphHandle_t`. The graph build itself remains driver-side
//! (`graphCreate` / `graphAddNode` / `graphFinalize`) and lives in microbench
//! binaries to keep this crate dependency-free (INV-151).
//!
//! Public surface intentionally stays minimal: a `LayerConfig` describing
//! Qwen 2.5-1.5B layer dimensions, plus declarative metadata
//! (`LAYER_NODE_COUNT`, `LAYER_INTERMEDIATE_COUNT`) so host tests can verify
//! topology assumptions without depending on the QNN bindings.

pub mod layer;
pub use layer::{LAYER_INTERMEDIATE_COUNT, LAYER_NODE_COUNT, LayerConfig};
