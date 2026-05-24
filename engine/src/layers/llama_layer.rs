//! Backward-compatibility shim — re-exports from `transformer_layer`.
//!
//! All logic has moved to `crate::layers::transformer_layer`.
//! This module preserves the old type names as aliases.

pub use super::transformer_layer::{ForwardGenArgs, LayerForwardArgs, QkvBias, TransformerLayer};

/// Backward-compatible alias for [`TransformerLayer`].
pub type LlamaLayer = TransformerLayer;

/// Backward-compatible alias for [`LayerForwardArgs`].
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O type alias default (KVCacheOps generic 기본형)
pub type LlamaLayerForwardArgs<'a, C = crate::pressure::kv_cache::KVCache> =
    LayerForwardArgs<'a, C>;

/// Backward-compatible alias for [`ForwardGenArgs`].
// LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O type alias default
pub type LlamaForwardGenArgs<'a, C = crate::pressure::kv_cache::KVCache> = ForwardGenArgs<'a, C>;
