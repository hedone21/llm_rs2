use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::fs::File;
use std::path::Path;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::sync::Arc;

use crate::core::backend::Backend;
use crate::core::tensor::Tensor;
use crate::core::shape::Shape;
use crate::core::buffer::DType;
use crate::core::memory::Memory;
use crate::core::kv_cache::KVCache;
use crate::layers::llama_layer::LlamaLayer;
use crate::layers::workspace::LayerWorkspace;

#[derive(Deserialize, Debug, Clone)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub head_dim: usize,
}

pub struct LlamaModel {
    pub config: LlamaConfig,
    pub embed_tokens: Tensor,
    pub layers: Vec<LlamaLayer>,
    pub norm: Tensor,
    pub lm_head: Tensor,
}

impl LlamaModel {
    pub fn load(path: &str, backend: Arc<dyn Backend>, memory: &dyn Memory) -> Result<Self> {
        let config_path = format!("{}/config.json", path);
        let config_file = File::open(config_path)?;
        let config: LlamaConfig = serde_json::from_reader(config_file)?;

        let mut layers = Vec::new();
        let mut embed_tokens = None;
        let mut norm = None;
        let mut lm_head = None;

        // Load tensors
        let model_path = format!("{}/model.safetensors", path);
        let file = File::open(model_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let safetensors = SafeTensors::deserialize(&mmap)?;

        // ... (loading logic abbreviated for brevity, assuming existing structure)
        // I will keep the actual loading logic from the previous view_file if I were to rewrite the whole file,
        // but since it's large, I'll use replace_file_content for the forward methods.
        
        // Wait, I should probably use replace_file_content to not mess up the loading logic which I haven't seen in full recently.
        unimplemented!("Use replace_file_content instead")
    }
}
