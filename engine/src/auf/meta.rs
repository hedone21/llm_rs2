/// AUF META section — 모델 architecture JSON payload (ENG-DAT-096.4).
///
/// META section은 JSON-in-binary 형식이다. 파싱 결과는 `AufMeta` 구조체.
use crate::auf::error::{AufError, AufResult};
use serde::{Deserialize, Serialize};

/// META section JSON 구조.
///
/// 직렬화 형식: UTF-8 JSON (`serde_json`). size 필드로 길이 명시.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AufMeta {
    /// 모델 아키텍처 (예: "llama", "qwen2").
    pub architecture: String,
    /// decoder layer 수.
    pub n_layers: u32,
    /// Q head 수.
    pub n_heads_q: u32,
    /// KV head 수.
    pub n_kv_heads: u32,
    /// head당 차원.
    pub head_dim: u32,
    /// hidden dim.
    pub hidden_dim: u32,
    /// FFN intermediate dim.
    pub ffn_dim: u32,
    /// vocab 크기.
    pub vocab_size: u32,
    /// 최대 시퀀스 길이.
    pub max_seq_len: u32,
    /// RoPE theta.
    pub rope_theta: f64,
    /// RoPE rotary_dim (rotary 적용 차원 수). 0이면 head_dim 전체.
    pub rotary_dim: u32,
    /// RoPE scaling factor (1.0 = 없음).
    pub rope_scaling: f64,
    /// RMSNorm epsilon.
    pub rms_norm_epsilon: f64,
}

impl AufMeta {
    /// JSON 바이트열에서 파싱한다.
    pub fn from_json_bytes(bytes: &[u8]) -> AufResult<Self> {
        serde_json::from_slice(bytes).map_err(|e| AufError::Other(format!("META JSON parse: {e}")))
    }

    /// JSON 바이트열로 직렬화한다.
    pub fn to_json_bytes(&self) -> AufResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| AufError::Other(format!("META JSON serialize: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_meta() -> AufMeta {
        AufMeta {
            architecture: "llama".to_owned(),
            n_layers: 16,
            n_heads_q: 32,
            n_kv_heads: 8,
            head_dim: 64,
            hidden_dim: 2048,
            ffn_dim: 8192,
            vocab_size: 128256,
            max_seq_len: 2048,
            rope_theta: 500000.0,
            rotary_dim: 64,
            rope_scaling: 1.0,
            rms_norm_epsilon: 1e-5,
        }
    }

    #[test]
    fn round_trip_json() {
        let meta = example_meta();
        let bytes = meta.to_json_bytes().unwrap();
        let meta2 = AufMeta::from_json_bytes(&bytes).unwrap();
        assert_eq!(meta2.architecture, "llama");
        assert_eq!(meta2.n_layers, 16);
        assert_eq!(meta2.n_heads_q, 32);
        assert_eq!(meta2.n_kv_heads, 8);
        assert_eq!(meta2.head_dim, 64);
        assert_eq!(meta2.vocab_size, 128256);
        assert!((meta2.rope_theta - 500000.0).abs() < 1.0);
    }

    #[test]
    fn invalid_json_returns_err() {
        let bad = b"not json";
        let err = AufMeta::from_json_bytes(bad).unwrap_err();
        assert!(matches!(err, AufError::Other(_)));
    }
}
