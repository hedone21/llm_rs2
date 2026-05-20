//! `AufSource` — AUF (Argus Unified Format) primary loader implementing
//! [`TensorSource`].
//!
//! ## 책임
//! - AUF 파일을 mmap 기반 [`AufView`]로 열고 `BackendTag` + `TensorDType` 선택을
//!   생성 시점에 고정 (LSP 안정성).
//! - `META` JSON에서 [`ModelConfig`] 구축 (`from_auf_meta`).
//! - `TensorId` ↔ `TensorEntry` 매핑 (`tensor_id_to_auf`).
//!
//! ## 본 sprint 범위
//! - **C1 (현 commit)**: 골격 + ModelConfig 구축 + has_tensor/cpu_backend/weight_dtype/config 등
//!   non-load 메서드 정상화. `load_tensor` / `load_tensor_cpu`는 `todo!()` placeholder.
//! - **C2 (다음 commit)**: secondary_mmap의 AUF 로직을 `loader/auf/secondary.rs`로 이동하면서
//!   primary용 zero-copy buffer (`AufViewBuffer` 예정) 추가 + load_tensor 본격 구현.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, anyhow};

use crate::auf::{
    AufView, BackendTag, LAYER_IDX_CROSS, TensorDType, TensorKind, open as auf_open,
};
use crate::backend::cpu::CpuBackend;
use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::memory::Memory;
use crate::core::tensor::Tensor;
use crate::models::config::{ModelArch, ModelConfig};
use crate::models::loader::{LayerWeightKind, TensorId, TensorSource};

use super::{AufDtypeChoice, AufVariantChoice};

/// AUF primary tensor source.
///
/// Lifetimes:
/// - `view`: `Arc<AufView>` 보유. mmap이 `AufSource` 폐기 시까지 유지된다.
/// - `variant_tag`/`primary_dtype`: 생성 시 결정, 이후 불변 (LSP).
pub struct AufSource {
    view: Arc<AufView>,
    variant_tag: BackendTag,
    /// 명시된 dtype 또는 `META.default_dtype` 기반 첫 후보.
    /// `lookup_tensor`의 `requested_dtype` 인자로 사용된다.
    primary_dtype: Option<TensorDType>,
    config: ModelConfig,
    cpu_backend: Arc<CpuBackend>,
    /// AUF source path (diagnostics).
    source_path: std::path::PathBuf,
}

impl AufSource {
    /// AUF 파일을 열어 primary `TensorSource`를 생성한다.
    ///
    /// Precedence:
    /// - `variant_choice` `Auto` → build feature 기반 default (`AufVariantChoice::default_tag`)
    /// - `dtype_choice` `Auto` → `META.default_dtype` 사용 (lookup 시점 결정, `primary_dtype = None`)
    /// - `eos_override` → `TOKENIZER.eos_id`가 N/A일 때 fallback
    pub fn open(
        path: &Path,
        variant_choice: AufVariantChoice,
        dtype_choice: AufDtypeChoice,
        eos_override: Option<u32>,
    ) -> Result<Self> {
        let variant_tag = variant_choice.to_backend_tag();
        let view = auf_open(path, variant_tag)
            .map_err(|e| anyhow!("AUF open failed for '{}': {}", path.display(), e))?;
        let view = Arc::new(view);

        let primary_dtype = dtype_choice.to_tensor_dtype();
        let config = from_auf_meta(&view, eos_override)?;

        eprintln!(
            "[AUF] Loaded: '{}' arch={:?} variant={:?} dtype={:?} layers={} vocab={}",
            path.display(),
            config.arch,
            variant_tag,
            primary_dtype,
            config.num_hidden_layers,
            config.vocab_size,
        );

        Ok(Self {
            view,
            variant_tag,
            primary_dtype,
            config,
            cpu_backend: Arc::new(CpuBackend::new()),
            source_path: path.to_path_buf(),
        })
    }

    /// `Arc<AufView>` 공유 핸들 (W-AUF-2 self-secondary에서 사용 예정).
    pub fn view_arc(&self) -> Arc<AufView> {
        Arc::clone(&self.view)
    }

    /// 본 source가 self-secondary swap 후보를 가지는지 (W-AUF-2에서 사용).
    /// multi-dtype capability bit ON이면 true.
    pub fn has_swap_candidate(&self) -> bool {
        self.view.header.has_multi_dtype()
    }

    /// Primary backend variant tag (W-AUF-2 self-secondary에서 자기 제외용).
    pub fn primary_variant_tag(&self) -> BackendTag {
        self.variant_tag
    }

    /// 진단용 source path 반환.
    pub fn source_path(&self) -> &Path {
        &self.source_path
    }
}

/// `TensorId` → AUF `(layer_idx, kind)` 매핑.
///
/// `LayerBias`는 본 sprint에서 미지원 — AUF v0.1 `TensorKind`에 bias variant가 없다.
/// `has_tensor`에서 false를 반환하여 GGUF/Safetensors와의 동작 격차를 막는다.
fn tensor_id_to_auf(id: &TensorId) -> Option<(u32, TensorKind)> {
    match id {
        TensorId::Embed => Some((LAYER_IDX_CROSS, TensorKind::Embedding)),
        TensorId::FinalNorm => Some((LAYER_IDX_CROSS, TensorKind::FinalNorm)),
        TensorId::LmHead => Some((LAYER_IDX_CROSS, TensorKind::LmHead)),
        TensorId::LayerWeight { layer, kind } => {
            let k = layer_weight_kind_to_tensor_kind(*kind)?;
            Some((*layer as u32, k))
        }
        TensorId::LayerBias { .. } => None,
    }
}

fn layer_weight_kind_to_tensor_kind(kind: LayerWeightKind) -> Option<TensorKind> {
    Some(match kind {
        LayerWeightKind::Wq => TensorKind::AttnQ,
        LayerWeightKind::Wk => TensorKind::AttnK,
        LayerWeightKind::Wv => TensorKind::AttnV,
        LayerWeightKind::Wo => TensorKind::AttnO,
        LayerWeightKind::WGate => TensorKind::FfnGate,
        LayerWeightKind::WUp => TensorKind::FfnUp,
        LayerWeightKind::WDown => TensorKind::FfnDown,
        LayerWeightKind::AttentionNorm => TensorKind::AttnNorm,
        LayerWeightKind::FfnNorm => TensorKind::FfnNorm,
        // Gemma3 extras — AUF v0.1 TensorKind에 미정의. 본 sprint는 Llama/Qwen2 한정.
        LayerWeightKind::PreFfnNorm
        | LayerWeightKind::PostFfnNorm
        | LayerWeightKind::QNorm
        | LayerWeightKind::KNorm => return None,
    })
}

/// AUF `META` JSON + `TENSOR_INDEX` + `TOKENIZER`에서 `ModelConfig`를 구축한다.
fn from_auf_meta(view: &AufView, eos_override: Option<u32>) -> Result<ModelConfig> {
    let m = &view.meta;
    let arch = parse_arch_str(&m.architecture)?;
    let has_qkv_bias = matches!(arch, ModelArch::Qwen2);
    let tie_word_embeddings = view.tensor_index.find_lm_head_entry().is_none();
    let eos_token_id = resolve_eos(view, eos_override);

    Ok(ModelConfig {
        arch,
        hidden_size: m.hidden_dim as usize,
        num_hidden_layers: m.n_layers as usize,
        num_attention_heads: m.n_heads_q as usize,
        num_key_value_heads: m.n_kv_heads as usize,
        head_dim: m.head_dim as usize,
        intermediate_size: m.ffn_dim as usize,
        vocab_size: m.vocab_size as usize,
        rms_norm_eps: m.rms_norm_epsilon,
        rope_theta: m.rope_theta,
        has_qkv_bias,
        tie_word_embeddings,
        eos_token_id,
        weight_prefix: String::new(),
        // Gemma3 fields — 본 sprint 미지원 (F4).
        rope_local_theta: None,
        sliding_window: None,
        sliding_window_pattern: None,
        query_pre_attn_scalar: None,
        embed_scale: None,
    })
}

fn parse_arch_str(s: &str) -> Result<ModelArch> {
    match s.to_ascii_lowercase().as_str() {
        "llama" => Ok(ModelArch::Llama),
        "qwen2" | "qwen2.5" => Ok(ModelArch::Qwen2),
        "gemma3" => Ok(ModelArch::Gemma3),
        other => Err(anyhow!(
            "AUF: unsupported architecture '{other}' (expected llama / qwen2 / gemma3)"
        )),
    }
}

/// EOS token id 결정 우선순위:
/// 1. AUF TOKENIZER section의 `eos_id` (>= 0).
/// 2. CLI override (`--eos-token-id`).
/// 3. `u32::MAX` fallback + stderr warning (EOS 정지 조건 비활성).
fn resolve_eos(view: &AufView, cli_override: Option<u32>) -> u32 {
    if view.tokenizer.eos_id >= 0 {
        return view.tokenizer.eos_id as u32;
    }
    cli_override.unwrap_or_else(|| {
        eprintln!(
            "[Warning] AUF TOKENIZER에 eos_id가 없습니다. --eos-token-id로 명시하지 않으면 EOS 정지 조건이 동작하지 않습니다."
        );
        u32::MAX
    })
}

// ── TensorSource impl ────────────────────────────────────────────────

impl TensorSource for AufSource {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weight_dtype(&self) -> DType {
        // primary_dtype이 명시되었으면 그것을 사용, 아니면 META.default_dtype 또는 F16 default.
        let tdt = self.primary_dtype.or_else(|| {
            self.view
                .meta
                .default_dtype
                .as_deref()
                .and_then(parse_meta_dtype_str)
        });
        match tdt {
            Some(TensorDType::F32) => DType::F32,
            Some(TensorDType::F16) => DType::F16,
            Some(TensorDType::Q4_0) => DType::Q4_0,
            Some(TensorDType::Q8_0) => DType::Q8_0,
            // BF16/Q4_1/U8은 engine DType에 직접 매핑이 없으므로 F16 fallback.
            _ => DType::F16,
        }
    }

    fn cpu_backend(&self) -> Arc<dyn Backend> {
        self.cpu_backend.clone() as Arc<dyn Backend>
    }

    fn has_tensor(&self, id: &TensorId) -> bool {
        let Some((layer_idx, kind)) = tensor_id_to_auf(id) else {
            return false;
        };
        let candidates = self.view.tensor_index.entries_for(layer_idx, kind.as_u32());
        !candidates.is_empty()
    }

    fn load_tensor(
        &self,
        _id: &TensorId,
        _is_weight: bool,
        _backend: &Arc<dyn Backend>,
        _memory: &dyn Memory,
    ) -> Result<Tensor> {
        // C2에서 secondary_mmap의 AUF 로직과 함께 zero-copy buffer 구현.
        todo!("AufSource::load_tensor — implement in W-AUF-1 C2 (with AufViewBuffer)")
    }

    fn load_tensor_cpu(
        &self,
        _id: &TensorId,
        _is_weight: bool,
        _memory: &dyn Memory,
    ) -> Result<Tensor> {
        todo!("AufSource::load_tensor_cpu — implement in W-AUF-1 C2")
    }
}

/// AUF META `default_dtype` 문자열 → `TensorDType` (graceful Option).
fn parse_meta_dtype_str(s: &str) -> Option<TensorDType> {
    match s {
        "F32" => Some(TensorDType::F32),
        "F16" => Some(TensorDType::F16),
        "BF16" => Some(TensorDType::BF16),
        "Q4_0" => Some(TensorDType::Q4_0),
        "Q4_1" => Some(TensorDType::Q4_1),
        "Q8_0" => Some(TensorDType::Q8_0),
        "U8" => Some(TensorDType::U8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_id_layer_weight_round_trip() {
        let id = TensorId::LayerWeight {
            layer: 7,
            kind: LayerWeightKind::Wq,
        };
        let (layer_idx, kind) = tensor_id_to_auf(&id).unwrap();
        assert_eq!(layer_idx, 7);
        assert_eq!(kind, TensorKind::AttnQ);
    }

    #[test]
    fn tensor_id_cross_layer() {
        assert_eq!(
            tensor_id_to_auf(&TensorId::Embed).unwrap(),
            (LAYER_IDX_CROSS, TensorKind::Embedding)
        );
        assert_eq!(
            tensor_id_to_auf(&TensorId::FinalNorm).unwrap(),
            (LAYER_IDX_CROSS, TensorKind::FinalNorm)
        );
        assert_eq!(
            tensor_id_to_auf(&TensorId::LmHead).unwrap(),
            (LAYER_IDX_CROSS, TensorKind::LmHead)
        );
    }

    #[test]
    fn tensor_id_bias_unsupported() {
        let id = TensorId::LayerBias {
            layer: 0,
            kind: crate::models::loader::LayerBiasKind::Bq,
        };
        assert!(tensor_id_to_auf(&id).is_none());
    }

    #[test]
    fn parse_arch_handles_known_values() {
        assert_eq!(parse_arch_str("llama").unwrap(), ModelArch::Llama);
        assert_eq!(parse_arch_str("qwen2").unwrap(), ModelArch::Qwen2);
        assert_eq!(parse_arch_str("Qwen2.5").unwrap(), ModelArch::Qwen2);
        assert!(parse_arch_str("unknown_arch").is_err());
    }
}
