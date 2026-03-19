use super::{LayerWeightNames, WeightMapper};

pub struct LlamaMapper;

impl WeightMapper for LlamaMapper {
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
            ffn_norm: format!("model.layers.{i}.post_attention_layernorm.weight"),
        }
    }
}
