//! [`crate::session::traits::Forward`] concrete implementations (Phase 4-3+).
//!
//! Each submodule provides one `Forward` variant that the `DecodeLoopBuilder`
//! can be wired with. The first is [`model_forward::ModelForward`] — the
//! standard `KVCache`-backed path that wraps
//! [`crate::models::transformer::TransformerModel::forward_into`].

pub mod model_forward;

pub use model_forward::{ModelForward, alloc_standard_kv_caches};
