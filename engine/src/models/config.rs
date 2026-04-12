use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::fs::File;
use std::path::Path;

/// Discriminator for model architecture dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Llama,
    Qwen2,
    Gemma3,
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
    pub eos_token_id: u32,

    // Gemma 3 specific fields (None for Llama/Qwen2)
    pub rope_local_theta: Option<f64>,
    pub sliding_window: Option<usize>,
    pub sliding_window_pattern: Option<usize>,
    pub query_pre_attn_scalar: Option<usize>,
    pub embed_scale: Option<f32>,
}

/// Raw HuggingFace config.json — supports Llama, Qwen2, and Gemma3 via Option fields.
#[derive(Deserialize)]
#[allow(dead_code)]
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
    eos_token_id: Option<u32>,
    // Gemma 3 specific
    rope_local_base_freq: Option<f64>,
    sliding_window: Option<usize>,
    sliding_window_pattern: Option<usize>,
    query_pre_attn_scalar: Option<usize>,
    hidden_activation: Option<String>,
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
            ModelArch::Llama | ModelArch::Gemma3 => false,
        };

        // Gemma 3 specific fields
        let (
            rope_local_theta,
            sliding_window,
            sliding_window_pattern,
            query_pre_attn_scalar,
            embed_scale,
        ) = match arch {
            ModelArch::Gemma3 => (
                Some(raw.rope_local_base_freq.unwrap_or(10000.0)),
                raw.sliding_window,
                raw.sliding_window_pattern,
                raw.query_pre_attn_scalar,
                Some((raw.hidden_size as f32).sqrt()),
            ),
            _ => (None, None, None, None, None),
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
            eos_token_id: raw.eos_token_id.unwrap_or(u32::MAX),
            rope_local_theta,
            sliding_window,
            sliding_window_pattern,
            query_pre_attn_scalar,
            embed_scale,
        })
    }

    /// Construct a ModelConfig from GGUF metadata.
    ///
    /// GGUF metadata keys follow the pattern `{arch}.{param_name}` where
    /// `{arch}` comes from `general.architecture`.
    pub fn from_gguf_metadata(gguf: &crate::models::loader::gguf::GgufFile) -> Result<Self> {
        let arch_str = gguf
            .get_str("general.architecture")
            .ok_or_else(|| anyhow!("GGUF: missing 'general.architecture' metadata"))?;

        let arch = match arch_str {
            "llama" => ModelArch::Llama,
            "qwen2" => ModelArch::Qwen2,
            "gemma" | "gemma2" | "gemma3" => ModelArch::Gemma3,
            _ => anyhow::bail!("GGUF: unsupported architecture '{}'", arch_str),
        };

        let prefix = arch_str;

        let hidden_size = gguf
            .get_u32(&format!("{prefix}.embedding_length"))
            .ok_or_else(|| anyhow!("GGUF: missing {prefix}.embedding_length"))?
            as usize;
        let num_hidden_layers =
            gguf.get_u32(&format!("{prefix}.block_count"))
                .ok_or_else(|| anyhow!("GGUF: missing {prefix}.block_count"))? as usize;
        let num_attention_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count"))
            .ok_or_else(|| anyhow!("GGUF: missing {prefix}.attention.head_count"))?
            as usize;
        let num_key_value_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(num_attention_heads as u32) as usize;
        let intermediate_size = gguf
            .get_u32(&format!("{prefix}.feed_forward_length"))
            .ok_or_else(|| anyhow!("GGUF: missing {prefix}.feed_forward_length"))?
            as usize;
        let vocab_size = gguf
            .get_u32(&format!("{prefix}.vocab_size"))
            .unwrap_or(32000) as usize;
        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5) as f64;
        let rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(10000.0) as f64;
        // head_dim: use explicit GGUF metadata if available (Gemma3 has head_dim != hidden/heads)
        let head_dim = gguf
            .get_u32(&format!("{prefix}.attention.head_dim"))
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_attention_heads);

        let has_qkv_bias = matches!(arch, ModelArch::Qwen2);
        let tie_word_embeddings = gguf.find_tensor("output.weight").is_none();

        // Gemma3 specific fields
        let (
            rope_local_theta,
            sliding_window,
            sliding_window_pattern,
            query_pre_attn_scalar,
            embed_scale,
        ) = match arch {
            ModelArch::Gemma3 => {
                let local_theta = gguf
                    .get_f32(&format!("{prefix}.rope.local.freq_base"))
                    .unwrap_or(10000.0) as f64;
                let sw = gguf
                    .get_u32(&format!("{prefix}.attention.sliding_window"))
                    .map(|v| v as usize);
                let sw_pattern = gguf
                    .get_u32(&format!("{prefix}.attention.sliding_window_pattern"))
                    .map(|v| v as usize);
                let qpas = gguf
                    .get_u32(&format!("{prefix}.attention.query_pre_attn_scalar"))
                    .map(|v| v as usize);
                let es = Some((hidden_size as f32).sqrt());
                (Some(local_theta), sw, sw_pattern, qpas, es)
            }
            _ => (None, None, None, None, None),
        };

        // eos_token_id: GGUF may store it as an array or a single value
        let eos_token_id = gguf
            .get_u32("tokenizer.ggml.eos_token_id")
            .unwrap_or(u32::MAX);

        Ok(Self {
            arch,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            has_qkv_bias,
            tie_word_embeddings,
            eos_token_id,
            rope_local_theta,
            sliding_window,
            sliding_window_pattern,
            query_pre_attn_scalar,
            embed_scale,
        })
    }

    fn detect_arch(raw: &RawHfConfig) -> Result<ModelArch> {
        // Try architectures field first
        if let Some(archs) = &raw.architectures {
            for a in archs {
                match a.as_str() {
                    "LlamaForCausalLM" => return Ok(ModelArch::Llama),
                    "Qwen2ForCausalLM" => return Ok(ModelArch::Qwen2),
                    "Gemma3ForCausalLM" => return Ok(ModelArch::Gemma3),
                    _ => {}
                }
            }
        }
        // Fallback to model_type
        if let Some(mt) = &raw.model_type {
            match mt.as_str() {
                "llama" => return Ok(ModelArch::Llama),
                "qwen2" => return Ok(ModelArch::Qwen2),
                "gemma3_text" | "gemma3" => return Ok(ModelArch::Gemma3),
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
    use std::io::Write;

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
        assert_eq!(config.eos_token_id, 128001);
    }

    #[test]
    fn test_parse_gemma3_config() {
        let json = r#"{
            "architectures": ["Gemma3ForCausalLM"],
            "model_type": "gemma3_text",
            "hidden_size": 1152,
            "num_hidden_layers": 26,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 6912,
            "vocab_size": 262144,
            "rms_norm_eps": 0.000001,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "sliding_window": 512,
            "sliding_window_pattern": 6,
            "query_pre_attn_scalar": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "tie_word_embeddings": true,
            "eos_token_id": 1
        }"#;

        // Write to a temp file in /tmp
        let tmp_dir = std::path::PathBuf::from("/tmp/llm_rs2_test_gemma3_config");
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let config_path = tmp_dir.join("config.json");
        let mut f = std::fs::File::create(&config_path).unwrap();
        f.write_all(json.as_bytes()).unwrap();

        let config = ModelConfig::from_json(&tmp_dir).unwrap();
        assert_eq!(config.arch, ModelArch::Gemma3);
        assert!(!config.has_qkv_bias);
        assert_eq!(config.hidden_size, 1152);
        assert_eq!(config.num_hidden_layers, 26);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_key_value_heads, 1);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.intermediate_size, 6912);
        assert_eq!(config.vocab_size, 262144);
        assert!((config.rms_norm_eps - 1e-6).abs() < 1e-10);
        assert!((config.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.eos_token_id, 1);

        // Gemma3 specific fields
        let local_theta = config
            .rope_local_theta
            .expect("rope_local_theta should be set");
        assert!((local_theta - 10000.0).abs() < 1.0);
        assert_eq!(config.sliding_window, Some(512));
        assert_eq!(config.sliding_window_pattern, Some(6));
        assert_eq!(config.query_pre_attn_scalar, Some(256));

        let embed_scale = config.embed_scale.expect("embed_scale should be set");
        let expected_scale = (1152_f32).sqrt();
        assert!(
            (embed_scale - expected_scale).abs() < 1e-3,
            "embed_scale={} expected={}",
            embed_scale,
            expected_scale
        );

        // Llama/Qwen2 fields should be None for Gemma3 — verify non-Gemma fields still work
        // (embed_scale is only Some for Gemma3)
        let _ = std::fs::remove_dir_all(&tmp_dir);
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
        assert_eq!(config.eos_token_id, 151643);
    }
}
