//! [`crate::session::traits::Forward`] concrete implementations (Phase 4-3+).
//!
//! Each submodule provides one `Forward` variant that the `DecodeLoopBuilder`
//! can be wired with. The first is [`model_forward::ModelForward`] — the
//! standard `KVCache`-backed path that wraps
//! [`crate::models::transformer::TransformerModel::forward_into`].
//!
//! Phase 4-5-a adds [`kivi_forward::KiviForward`] and
//! [`offload_forward::OffloadForward`] for KIVI-quantized and token-streaming
//! offload KV cache paths respectively.

pub mod kivi_forward;
pub mod model_forward;
pub mod offload_forward;

pub use kivi_forward::{KiviForward, alloc_kivi_kv_caches};
pub use model_forward::{ModelForward, alloc_standard_kv_caches};
pub use offload_forward::{OffloadForward, alloc_offload_kv_caches};
