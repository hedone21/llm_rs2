use super::{LayerWeightNames, WeightMapper};

pub struct Gemma3Mapper;

impl WeightMapper for Gemma3Mapper {
    fn weight_names(&self, i: usize) -> LayerWeightNames {
        LayerWeightNames {
            wq: format!("model.layers.{i}.self_attn.q_proj.weight"),
            wk: format!("model.layers.{i}.self_attn.k_proj.weight"),
            wv: format!("model.layers.{i}.self_attn.v_proj.weight"),
            wo: format!("model.layers.{i}.self_attn.o_proj.weight"),
            w_gate: format!("model.layers.{i}.mlp.gate_proj.weight"),
            w_up: format!("model.layers.{i}.mlp.up_proj.weight"),
            w_down: format!("model.layers.{i}.mlp.down_proj.weight"),
            attention_norm: format!("model.layers.{i}.input_layernorm.weight"),
            // Gemma3: this field holds post-attention norm (reuses the same HF name)
            ffn_norm: format!("model.layers.{i}.post_attention_layernorm.weight"),
            pre_ffn_norm: Some(format!("model.layers.{i}.pre_feedforward_layernorm.weight")),
            post_ffn_norm: Some(format!(
                "model.layers.{i}.post_feedforward_layernorm.weight"
            )),
            q_norm: Some(format!("model.layers.{i}.self_attn.q_norm.weight")),
            k_norm: Some(format!("model.layers.{i}.self_attn.k_norm.weight")),
        }
    }
    // bias_names: Gemma3 has no QKV bias — uses default (returns None)
}
