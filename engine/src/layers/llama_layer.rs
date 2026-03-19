//! Backward-compatibility shim — re-exports from `transformer_layer`.
//!
//! All logic has moved to `crate::layers::transformer_layer`.
//! This module preserves the old type names as aliases.

pub use super::transformer_layer::{
    ForwardGenArgs, LayerForwardArgs, OpProfiler, QkvBias, TransformerLayer,
};

/// Backward-compatible alias for [`TransformerLayer`].
pub type LlamaLayer = TransformerLayer;

/// Backward-compatible alias for [`LayerForwardArgs`].
pub type LlamaLayerForwardArgs<'a, C = crate::core::kv_cache::KVCache> = LayerForwardArgs<'a, C>;

/// Backward-compatible alias for [`ForwardGenArgs`].
pub type LlamaForwardGenArgs<'a, C = crate::core::kv_cache::KVCache> = ForwardGenArgs<'a, C>;
