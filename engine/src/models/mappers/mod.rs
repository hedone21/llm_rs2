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
    pub ffn_norm: String,
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
        // Gemma3 uses Llama-style weight naming (Phase 1 stub — full mapper in Phase 2)
        ModelArch::Gemma3 => Box::new(llama::LlamaMapper),
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
        assert!(m.bias_names(0).is_none());
    }

    #[test]
    fn test_qwen2_mapper_names() {
        let m = create_mapper(ModelArch::Qwen2);
        let names = m.weight_names(5);
        assert_eq!(names.wq, "model.layers.5.self_attn.q_proj.weight");
        let bias = m.bias_names(5).expect("Qwen2 should have bias");
        assert_eq!(bias.bq, "model.layers.5.self_attn.q_proj.bias");
        assert_eq!(bias.bk, "model.layers.5.self_attn.k_proj.bias");
        assert_eq!(bias.bv, "model.layers.5.self_attn.v_proj.bias");
    }
}
