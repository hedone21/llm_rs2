use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::fs::File;
use std::path::Path;

/// Discriminator for model architecture dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Llama,
    Qwen2,
}

/// Unified model configuration parsed from HuggingFace config.json.
/// All architecture-specific derivations are resolved at construction time.
#[derive(Debug)]
pub struct ModelConfig {
    pub arch: ModelArch,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub has_qkv_bias: bool,
    pub tie_word_embeddings: bool,
}

/// Raw HuggingFace config.json — supports both Llama and Qwen2 via Option fields.
#[derive(Deserialize)]
struct RawHfConfig {
    architectures: Option<Vec<String>>,
    model_type: Option<String>,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: Option<usize>,
    intermediate_size: usize,
    vocab_size: usize,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    tie_word_embeddings: Option<bool>,
}

impl ModelConfig {
    /// Parse config.json and auto-detect architecture.
    pub fn from_json(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let file = File::open(&config_path)
            .map_err(|e| anyhow!("Cannot open {}: {}", config_path.display(), e))?;
        let raw: RawHfConfig = serde_json::from_reader(file)?;

        let arch = Self::detect_arch(&raw)?;

        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads);

        let has_qkv_bias = match arch {
            ModelArch::Qwen2 => true,
            ModelArch::Llama => false,
        };

        Ok(Self {
            arch,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim,
            intermediate_size: raw.intermediate_size,
            vocab_size: raw.vocab_size,
            rms_norm_eps: raw.rms_norm_eps.unwrap_or(1e-5),
            rope_theta: raw.rope_theta.unwrap_or(10000.0),
            has_qkv_bias,
            tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
        })
    }

    fn detect_arch(raw: &RawHfConfig) -> Result<ModelArch> {
        // Try architectures field first
        if let Some(archs) = &raw.architectures {
            for a in archs {
                match a.as_str() {
                    "LlamaForCausalLM" => return Ok(ModelArch::Llama),
                    "Qwen2ForCausalLM" => return Ok(ModelArch::Qwen2),
                    _ => {}
                }
            }
        }
        // Fallback to model_type
        if let Some(mt) = &raw.model_type {
            match mt.as_str() {
                "llama" => return Ok(ModelArch::Llama),
                "qwen2" => return Ok(ModelArch::Qwen2),
                _ => {}
            }
        }
        Err(anyhow!(
            "Unsupported model architecture: {:?} / {:?}",
            raw.architectures,
            raw.model_type
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_llama_config() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models/llama3.2-1b");
        if !dir.exists() {
            eprintln!("Skipping: model dir not found at {}", dir.display());
            return;
        }
        let config = ModelConfig::from_json(&dir).unwrap();
        assert_eq!(config.arch, ModelArch::Llama);
        assert!(!config.has_qkv_bias);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 16);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
    }

    #[test]
    fn test_parse_qwen2_config() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models/qwen2.5-1.5b");
        if !dir.exists() {
            eprintln!("Skipping: model dir not found at {}", dir.display());
            return;
        }
        let config = ModelConfig::from_json(&dir).unwrap();
        assert_eq!(config.arch, ModelArch::Qwen2);
        assert!(config.has_qkv_bias);
        // head_dim derived: 1536 / 12 = 128
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.hidden_size, 1536);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.num_key_value_heads, 2);
        assert!(config.tie_word_embeddings);
    }
}
