use super::{LayerBiasNames, LayerWeightNames, WeightMapper};

pub struct Qwen2Mapper;

impl WeightMapper for Qwen2Mapper {
    fn weight_names(&self, i: usize) -> LayerWeightNames {
        // Qwen2 uses the same weight names as Llama
        LayerWeightNames {
            wq: format!("model.layers.{i}.self_attn.q_proj.weight"),
            wk: format!("model.layers.{i}.self_attn.k_proj.weight"),
            wv: format!("model.layers.{i}.self_attn.v_proj.weight"),
            wo: format!("model.layers.{i}.self_attn.o_proj.weight"),
            w_gate: format!("model.layers.{i}.mlp.gate_proj.weight"),
            w_up: format!("model.layers.{i}.mlp.up_proj.weight"),
            w_down: format!("model.layers.{i}.mlp.down_proj.weight"),
            attention_norm: format!("model.layers.{i}.input_layernorm.weight"),
            ffn_norm: format!("model.layers.{i}.post_attention_layernorm.weight"),
            pre_ffn_norm: None,
            post_ffn_norm: None,
            q_norm: None,
            k_norm: None,
        }
    }

    fn bias_names(&self, i: usize) -> Option<LayerBiasNames> {
        Some(LayerBiasNames {
            bq: format!("model.layers.{i}.self_attn.q_proj.bias"),
            bk: format!("model.layers.{i}.self_attn.k_proj.bias"),
            bv: format!("model.layers.{i}.self_attn.v_proj.bias"),
        })
    }
}
