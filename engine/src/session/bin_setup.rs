//! 추론 bin 공용 셋업 (argus_cli / argus_bench 공유).
//!
//! `SessionInitCtx::build` → tokenizer resolve/load → prompt encode →
//! KV cache 할당 → resilience adapter 까지 묶어 [`StandardHappyCtx`] 를 만든다.
//! 각 bin 의 `main` 은 reject 가드 + dispatch (run_standard_happy_path /
//! run_experiment_path) 만 담당한다.

use std::sync::Arc;

use anyhow::bail;
use tokenizers::Tokenizer;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::pressure::kv_cache::{KVCache, KVLayout};
use crate::session::cli::Args;
use crate::session::init::SessionInitCtx;
use crate::session::resilience_adapter::ResilienceAdapter;
use crate::session::resilience_init::build_command_executor;
use crate::session::standard_happy::StandardHappyCtx;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// `Args` 를 받아 추론에 필요한 전 컨텍스트를 조립한다.
///
/// `args.enable_resilience` 가 true 면 `build_command_executor` 로 transport 를
/// 연결하고 [`ResilienceAdapter`] 를 만든다 (transport 실패는 Err 전파).
/// false 면 `resilience = None` (NoOp default).
pub fn build_inference_ctx(args: Args) -> anyhow::Result<StandardHappyCtx> {
    let ctx = SessionInitCtx::build(&args)?;
    let backend = ctx.backend;
    let memory = ctx.memory;
    let hardware = ctx.hardware;
    let sampling_config = ctx.sampling_config;
    let model = ctx.model;
    let model_path = ctx.model_path;
    let is_gguf = ctx.is_gguf;

    let tokenizer_path = resolve_tokenizer_path(&args, &model_path, is_gguf);
    eprintln!("[Tokenizer] {}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Cannot load tokenizer from {}: {}", tokenizer_path, e))?;
    check_vocab_compatibility(&tokenizer, &model, &tokenizer_path)?;

    let prompt = if let Some(path) = &args.prompt_file {
        std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read prompt file {}: {}", path, e))?
    } else {
        args.prompt.clone()
    };
    eprintln!("Prompt: {}", prompt);
    let encoding = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let tokens: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("Token Length: {}", tokens.len());

    let max_seq_len = args.max_seq_len;
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;
    let num_layers = model.config.num_hidden_layers;
    let vocab_size = model.config.vocab_size;
    eprintln!(
        "Model config: layers={}, kv_heads={}, head_dim={}, max_seq_len={}",
        num_layers, kv_heads, head_dim, max_seq_len
    );

    let kv_type = match args.kv_type.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "q4" => DType::Q4_0,
        other => bail!("Unsupported KV type: {other}. Use f32, f16, or q4."),
    };

    let initial_kv_capacity = if args.initial_kv_capacity() > 0 {
        args.initial_kv_capacity().min(max_seq_len)
    } else {
        tokens
            .len()
            .saturating_add(args.num_tokens)
            .next_power_of_two()
            .max(128)
            .min(max_seq_len)
    };

    let kv_caches = alloc_standard_kv_caches(
        &backend,
        memory.clone(),
        num_layers,
        initial_kv_capacity,
        max_seq_len,
        kv_heads,
        head_dim,
        kv_type,
    )?;

    // ResilienceAdapter 생성. `--no-resilience` (effective enable_resilience=false)
    // 시 None. transport 연결 실패는 Err 전파 — graceful fail, panic 없음.
    let resilience: Option<ResilienceAdapter> = if args.enable_resilience {
        build_command_executor(&args, &model)?.map(ResilienceAdapter::new)
    } else {
        None
    };

    Ok(StandardHappyCtx {
        args,
        backend,
        memory,
        hardware,
        model,
        tokenizer,
        kv_caches,
        tokens,
        initial_kv_capacity,
        max_seq_len,
        kv_type,
        sampling_config,
        vocab_size,
        resilience,
    })
}

/// `--tokenizer-path` 우선, GGUF 면 sibling stem 검색, safetensors 면 dir 안의
/// `tokenizer.json`. legacy `generate` 와 동일한 resolution 순서.
pub fn resolve_tokenizer_path(args: &Args, model_path: &str, is_gguf: bool) -> String {
    if let Some(p) = args.tokenizer_path.as_ref() {
        return p.to_string_lossy().into_owned();
    }
    if is_gguf {
        let path = std::path::Path::new(model_path);
        let parent = path.parent().unwrap_or(std::path::Path::new("."));
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        const QUANT_SUFFIXES: &[&str] = &["-f16", "-f32", "-q4_0", "-q4_1", "-q8_0", "-q4_k"];
        let stem_lower = stem.to_ascii_lowercase();
        let stem_stripped: Option<String> = QUANT_SUFFIXES.iter().find_map(|suf| {
            stem_lower
                .strip_suffix(suf)
                .map(|s| stem[..s.len()].to_string())
        });
        let mut candidates: Vec<std::path::PathBuf> = Vec::with_capacity(3);
        candidates.push(parent.join(format!("{stem}.tokenizer.json")));
        if let Some(ref s) = stem_stripped {
            candidates.push(parent.join(format!("{s}.tokenizer.json")));
        }
        candidates.push(parent.join("tokenizer.json"));
        candidates
            .iter()
            .find(|p| p.exists())
            .cloned()
            .unwrap_or_else(|| parent.join("tokenizer.json"))
            .to_string_lossy()
            .into_owned()
    } else {
        format!("{}/tokenizer.json", model_path)
    }
}

/// tokenizer vocab 과 model vocab 불일치 검사 (legacy generate 와 동일 정책).
pub fn check_vocab_compatibility(
    tokenizer: &Tokenizer,
    model: &TransformerModel,
    tokenizer_path: &str,
) -> anyhow::Result<()> {
    let tok_vocab = tokenizer.get_vocab_size(true);
    let model_vocab = model.config.vocab_size;
    let oob_tolerance: usize = 8;
    if tok_vocab > model_vocab + oob_tolerance {
        bail!(
            "Tokenizer vocab ({}) exceeds model vocab ({}) by more than {} — OOB embedding lookup risk. \
             Path: {}. Pass --tokenizer-path with the matching tokenizer.json.",
            tok_vocab,
            model_vocab,
            oob_tolerance,
            tokenizer_path
        );
    } else if tok_vocab > model_vocab {
        eprintln!(
            "[Tokenizer] WARNING: tokenizer vocab ({}) > model vocab ({}) by {} (likely multimodal special tokens).",
            tok_vocab,
            model_vocab,
            tok_vocab - model_vocab
        );
    }
    let pad_tolerance: usize = (model_vocab / 20).max(256);
    if model_vocab > tok_vocab + pad_tolerance {
        bail!(
            "Model vocab ({}) exceeds tokenizer vocab ({}) by more than {} — likely wrong tokenizer for model. \
             Path: {}. Pass --tokenizer-path with the matching tokenizer.json.",
            model_vocab,
            tok_vocab,
            pad_tolerance,
            tokenizer_path
        );
    }
    Ok(())
}

/// HeadMajor layout + dynamic grow-on-demand KV cache 를 num_layers 만큼 할당.
#[allow(clippy::too_many_arguments)]
pub fn alloc_standard_kv_caches(
    backend: &Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    num_layers: usize,
    initial_kv_capacity: usize,
    max_seq_len: usize,
    kv_heads: usize,
    head_dim: usize,
    kv_type: DType,
) -> anyhow::Result<Vec<KVCache>> {
    let n_values = initial_kv_capacity * kv_heads * head_dim;
    let kv_buf_size = match kv_type {
        DType::Q4_0 => {
            use crate::quant::{BlockQ4_0, QK4_0};
            (n_values / QK4_0) * std::mem::size_of::<BlockQ4_0>()
        }
        _ => n_values * kv_type.size(),
    };
    eprintln!(
        "KV cache type: {:?}, layout: HeadMajor (initial capacity: {} tokens, {}B per layer, max: {})",
        kv_type, initial_kv_capacity, kv_buf_size, max_seq_len
    );
    let mut kv_caches = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let k_buf = memory.alloc_kv(kv_buf_size, kv_type)?;
        let v_buf = memory.alloc_kv(kv_buf_size, kv_type)?;
        let shape = Shape::new(vec![1, kv_heads, initial_kv_capacity, head_dim]);
        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend.clone());
        kv_caches.push(
            KVCache::new_dynamic(
                k,
                v,
                initial_kv_capacity,
                max_seq_len,
                kv_heads,
                head_dim,
                memory.clone(),
            )
            .with_layout(KVLayout::HeadMajor),
        );
    }
    Ok(kv_caches)
}
