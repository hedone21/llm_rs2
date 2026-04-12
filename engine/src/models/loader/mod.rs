//! Model loader abstraction layer.
//!
//! Defines the `TensorSource` trait for format-agnostic tensor loading,
//! and the `load_model()` function that assembles a `TransformerModel` from
//! any `TensorSource` implementation.
//!
//! Current implementations:
//! - `SafetensorsSource` — HuggingFace safetensors format

pub mod convert;
pub mod safetensors;

use anyhow::Result;
use std::sync::Arc;

use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::memory::Memory;
use crate::core::tensor::Tensor;
use crate::layers::transformer_layer::{QkvBias, TransformerLayer};
use crate::models::config::ModelConfig;
use crate::models::transformer::TransformerModel;

/// Internal standard tensor identifier (format-agnostic).
#[derive(Debug, Clone, Copy)]
pub enum LayerWeightKind {
    Wq,
    Wk,
    Wv,
    Wo,
    WGate,
    WUp,
    WDown,
    AttentionNorm,
    FfnNorm,
    // Gemma3 extra norms
    PreFfnNorm,
    PostFfnNorm,
    QNorm,
    KNorm,
}

/// Bias tensor kinds (Qwen2).
#[derive(Debug, Clone, Copy)]
pub enum LayerBiasKind {
    Bq,
    Bk,
    Bv,
}

/// Format-agnostic tensor identifier.
#[derive(Debug, Clone)]
pub enum TensorId {
    Embed,
    FinalNorm,
    LmHead,
    LayerWeight { layer: usize, kind: LayerWeightKind },
    LayerBias { layer: usize, kind: LayerBiasKind },
}

/// Format-agnostic tensor source trait.
///
/// Implementations translate `TensorId` into format-specific lookups
/// (e.g., safetensors weight names, GGUF tensor indices).
pub trait TensorSource {
    /// Model configuration.
    fn config(&self) -> &ModelConfig;

    /// Load a tensor onto the given backend.
    fn load_tensor(
        &self,
        id: &TensorId,
        is_weight: bool,
        backend: &Arc<dyn Backend>,
        memory: &dyn Memory,
    ) -> Result<Tensor>;

    /// Load a tensor onto CPU only (for embed_tokens, etc.).
    fn load_tensor_cpu(
        &self,
        id: &TensorId,
        is_weight: bool,
        memory: &dyn Memory,
    ) -> Result<Tensor>;

    /// Check if a tensor exists in the source.
    fn has_tensor(&self, id: &TensorId) -> bool;

    /// The weight dtype for this source.
    fn weight_dtype(&self) -> DType;

    /// CPU backend reference (for gather_embed fallback paths).
    fn cpu_backend(&self) -> Arc<dyn Backend>;
}

/// Assemble a `TransformerModel` from a `TensorSource`.
///
/// This function contains the layer-building loop and lm_head/embed logic
/// previously in `TransformerModel::load_with_dtype()`.
pub fn load_model(
    source: &dyn TensorSource,
    backend: Arc<dyn Backend>,
    memory: &dyn Memory,
) -> Result<TransformerModel> {
    let config = source.config();
    let weight_dtype = source.weight_dtype();
    let is_cpu = backend.name().contains("CPU");
    let num_layers = config.num_hidden_layers;

    // 1. Load layers
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let load_weight = |kind: LayerWeightKind| -> Result<Tensor> {
            source.load_tensor(
                &TensorId::LayerWeight { layer: i, kind },
                true,
                &backend,
                memory,
            )
        };
        let load_norm = |kind: LayerWeightKind| -> Result<Tensor> {
            source.load_tensor(
                &TensorId::LayerWeight { layer: i, kind },
                false,
                &backend,
                memory,
            )
        };

        let qkv_bias = if config.has_qkv_bias {
            let has_bq = source.has_tensor(&TensorId::LayerBias {
                layer: i,
                kind: LayerBiasKind::Bq,
            });
            if has_bq {
                Some(QkvBias {
                    bq: source.load_tensor(
                        &TensorId::LayerBias {
                            layer: i,
                            kind: LayerBiasKind::Bq,
                        },
                        false,
                        &backend,
                        memory,
                    )?,
                    bk: source.load_tensor(
                        &TensorId::LayerBias {
                            layer: i,
                            kind: LayerBiasKind::Bk,
                        },
                        false,
                        &backend,
                        memory,
                    )?,
                    bv: source.load_tensor(
                        &TensorId::LayerBias {
                            layer: i,
                            kind: LayerBiasKind::Bv,
                        },
                        false,
                        &backend,
                        memory,
                    )?,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Optional Gemma3 tensors
        let q_norm = if source.has_tensor(&TensorId::LayerWeight {
            layer: i,
            kind: LayerWeightKind::QNorm,
        }) {
            Some(load_norm(LayerWeightKind::QNorm)?)
        } else {
            None
        };
        let k_norm = if source.has_tensor(&TensorId::LayerWeight {
            layer: i,
            kind: LayerWeightKind::KNorm,
        }) {
            Some(load_norm(LayerWeightKind::KNorm)?)
        } else {
            None
        };
        let pre_ffn_norm = if source.has_tensor(&TensorId::LayerWeight {
            layer: i,
            kind: LayerWeightKind::PreFfnNorm,
        }) {
            Some(load_norm(LayerWeightKind::PreFfnNorm)?)
        } else {
            None
        };
        let post_ffn_norm = if source.has_tensor(&TensorId::LayerWeight {
            layer: i,
            kind: LayerWeightKind::PostFfnNorm,
        }) {
            Some(load_norm(LayerWeightKind::PostFfnNorm)?)
        } else {
            None
        };

        let layer = TransformerLayer {
            wq: load_weight(LayerWeightKind::Wq)?,
            wk: load_weight(LayerWeightKind::Wk)?,
            wv: load_weight(LayerWeightKind::Wv)?,
            wo: load_weight(LayerWeightKind::Wo)?,
            w_gate: load_weight(LayerWeightKind::WGate)?,
            w_up: load_weight(LayerWeightKind::WUp)?,
            w_down: load_weight(LayerWeightKind::WDown)?,
            attention_norm: load_norm(LayerWeightKind::AttentionNorm)?,
            ffn_norm: load_norm(LayerWeightKind::FfnNorm)?,
            qkv_bias,
            q_norm,
            k_norm,
            pre_ffn_norm,
            post_ffn_norm,
            partition_ctx: None,
        };
        layers.push(layer);
    }

    // 2. Embed tokens (always CPU)
    let embed_tokens =
        source.load_tensor_cpu(&TensorId::Embed, weight_dtype == DType::F16, memory)?;

    // 3. Final norm
    let norm = source.load_tensor(&TensorId::FinalNorm, false, &backend, memory)?;

    // 4. lm_head
    let has_lm_head = source.has_tensor(&TensorId::LmHead);
    let (lm_head, lm_head_on_cpu) = if has_lm_head {
        (
            source.load_tensor(&TensorId::LmHead, true, &backend, memory)?,
            false,
        )
    } else {
        // Tied weights: build lm_head from embed_tokens
        eprintln!(
            "lm_head not found, deriving from embed_tokens ({:?}) for lm_head...",
            weight_dtype
        );
        if is_cpu {
            (embed_tokens.clone(), false)
        } else {
            let embed_size = embed_tokens.size();
            let max_alloc = backend.max_single_alloc();
            if embed_size > max_alloc {
                eprintln!(
                    "lm_head too large for GPU ({:.0} MB > {:.0} MB limit), keeping on CPU",
                    embed_size as f64 / (1024.0 * 1024.0),
                    max_alloc as f64 / (1024.0 * 1024.0),
                );
                (embed_tokens.clone(), true)
            } else {
                (backend.copy_from(&embed_tokens)?, false)
            }
        }
    };

    // 5. CPU backend reference (only when main backend is GPU)
    let stored_cpu_backend = if is_cpu {
        None
    } else {
        Some(source.cpu_backend())
    };

    // 6. GPU-side embed_tokens
    let gpu_embed_tokens = if is_cpu {
        None
    } else if !has_lm_head && !lm_head_on_cpu {
        // Tied weights with GPU lm_head: reuse (zero extra memory)
        Some(lm_head.clone())
    } else {
        match backend.copy_from(&embed_tokens) {
            Ok(gpu_t) => Some(gpu_t),
            Err(e) => {
                eprintln!(
                    "Warning: failed to upload embed_tokens to GPU ({e}), \
                     gather will use CPU path"
                );
                None
            }
        }
    };

    // Clone config for the model (TensorSource owns the original)
    let model_config = ModelConfig {
        arch: config.arch,
        hidden_size: config.hidden_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        intermediate_size: config.intermediate_size,
        vocab_size: config.vocab_size,
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        has_qkv_bias: config.has_qkv_bias,
        tie_word_embeddings: config.tie_word_embeddings,
        eos_token_id: config.eos_token_id,
        rope_local_theta: config.rope_local_theta,
        sliding_window: config.sliding_window,
        sliding_window_pattern: config.sliding_window_pattern,
        query_pre_attn_scalar: config.query_pre_attn_scalar,
        embed_scale: config.embed_scale,
    };

    Ok(TransformerModel {
        config: model_config,
        layers,
        embed_tokens,
        norm,
        lm_head,
        lm_head_on_cpu,
        gpu_embed_tokens,
        cpu_backend: stored_cpu_backend,
        preload_pool: std::sync::Mutex::new(None),
    })
}
