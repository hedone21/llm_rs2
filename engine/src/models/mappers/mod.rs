mod gemma3;
mod llama;
mod qwen2;

use super::config::ModelArch;

/// Per-layer weight tensor names in the safetensors file.
pub struct LayerWeightNames {
    pub wq: String,
    pub wk: String,
    pub wv: String,
    pub wo: String,
    pub w_gate: String,
    pub w_up: String,
    pub w_down: String,
    pub attention_norm: String,
    /// Llama/Qwen2: pre-FFN norm (post_attention_layernorm)
    /// Gemma3: post-attention norm (post_attention_layernorm) — role differs
    pub ffn_norm: String,
    /// Gemma3: pre_feedforward_layernorm. None for Llama/Qwen2.
    pub pre_ffn_norm: Option<String>,
    /// Gemma3: post_feedforward_layernorm. None for Llama/Qwen2.
    pub post_ffn_norm: Option<String>,
    /// Gemma3: QK-Norm weight for Q. None for Llama/Qwen2.
    pub q_norm: Option<String>,
    /// Gemma3: QK-Norm weight for K. None for Llama/Qwen2.
    pub k_norm: Option<String>,
}

/// Per-layer QKV bias tensor names (Qwen2 only).
pub struct LayerBiasNames {
    pub bq: String,
    pub bk: String,
    pub bv: String,
}

/// Maps layer index to safetensors weight names.
/// Different architectures may have different naming conventions or fused weights.
pub trait WeightMapper: Send + Sync {
    /// Core weight tensor names for a given layer.
    fn weight_names(&self, layer_idx: usize) -> LayerWeightNames;

    /// Optional QKV bias tensor names. Returns None if the architecture has no bias.
    fn bias_names(&self, _layer_idx: usize) -> Option<LayerBiasNames> {
        None
    }

    /// Embedding tensor name.
    fn embed_name(&self) -> &str {
        "model.embed_tokens.weight"
    }

    /// Final RMSNorm tensor name.
    fn norm_name(&self) -> &str {
        "model.norm.weight"
    }

    /// LM head tensor name.
    fn lm_head_name(&self) -> &str {
        "lm_head.weight"
    }
}

/// Factory: create the appropriate WeightMapper for a given architecture.
pub fn create_mapper(arch: ModelArch) -> Box<dyn WeightMapper> {
    match arch {
        ModelArch::Llama => Box::new(llama::LlamaMapper),
        ModelArch::Qwen2 => Box::new(qwen2::Qwen2Mapper),
        ModelArch::Gemma3 => Box::new(gemma3::Gemma3Mapper),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_mapper_names() {
        let m = create_mapper(ModelArch::Llama);
        let names = m.weight_names(0);
        assert_eq!(names.wq, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(names.w_gate, "model.layers.0.mlp.gate_proj.weight");
        assert!(names.pre_ffn_norm.is_none());
        assert!(names.post_ffn_norm.is_none());
        assert!(names.q_norm.is_none());
        assert!(names.k_norm.is_none());
        assert!(m.bias_names(0).is_none());
    }

    #[test]
    fn test_qwen2_mapper_names() {
        let m = create_mapper(ModelArch::Qwen2);
        let names = m.weight_names(5);
        assert_eq!(names.wq, "model.layers.5.self_attn.q_proj.weight");
        assert!(names.pre_ffn_norm.is_none());
        assert!(names.post_ffn_norm.is_none());
        assert!(names.q_norm.is_none());
        assert!(names.k_norm.is_none());
        let bias = m.bias_names(5).expect("Qwen2 should have bias");
        assert_eq!(bias.bq, "model.layers.5.self_attn.q_proj.bias");
        assert_eq!(bias.bk, "model.layers.5.self_attn.k_proj.bias");
        assert_eq!(bias.bv, "model.layers.5.self_attn.v_proj.bias");
    }

    #[test]
    fn test_gemma3_mapper_names() {
        let m = create_mapper(ModelArch::Gemma3);
        let names = m.weight_names(0);
        assert_eq!(names.wq, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(names.wk, "model.layers.0.self_attn.k_proj.weight");
        assert_eq!(names.wv, "model.layers.0.self_attn.v_proj.weight");
        assert_eq!(names.wo, "model.layers.0.self_attn.o_proj.weight");
        assert_eq!(names.w_gate, "model.layers.0.mlp.gate_proj.weight");
        assert_eq!(names.w_up, "model.layers.0.mlp.up_proj.weight");
        assert_eq!(names.w_down, "model.layers.0.mlp.down_proj.weight");
        assert_eq!(
            names.attention_norm,
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            names.ffn_norm,
            "model.layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(
            names.pre_ffn_norm.as_deref(),
            Some("model.layers.0.pre_feedforward_layernorm.weight")
        );
        assert_eq!(
            names.post_ffn_norm.as_deref(),
            Some("model.layers.0.post_feedforward_layernorm.weight")
        );
        assert_eq!(
            names.q_norm.as_deref(),
            Some("model.layers.0.self_attn.q_norm.weight")
        );
        assert_eq!(
            names.k_norm.as_deref(),
            Some("model.layers.0.self_attn.k_norm.weight")
        );
        // Gemma3 has no QKV bias
        assert!(m.bias_names(0).is_none());

        // Layer index 12 check
        let names12 = m.weight_names(12);
        assert_eq!(names12.wq, "model.layers.12.self_attn.q_proj.weight");
        assert_eq!(
            names12.pre_ffn_norm.as_deref(),
            Some("model.layers.12.pre_feedforward_layernorm.weight")
        );
        assert_eq!(
            names12.q_norm.as_deref(),
            Some("model.layers.12.self_attn.q_norm.weight")
        );
    }
}
