//! argus-cli — single-prompt inference.
//!
//! ARGUS CLI 패밀리의 단일 추론 엔트리. legacy `generate` 의 standard happy
//! path + production resilience 를 지원한다. chat / experiment / ppl / eval /
//! dump / prompt-batch / weight swap / KIVI / offload / profile /
//! tensor-partition 는 아직 미구현이며 명시적으로 reject 한다.
//!
//! ## v1 흡수 진행 (sub-sprint 단위)
//!
//! - **v1-1 (current)**: resilience default-on (`--no-resilience` opt-out).
//! - v1-2: `--prompt-batch` (session::batch).
//! - v1-3: weight swap 8종 (session::decode_fallback::swap_dispatch).
//! - v1-4: `--profile` / `--profile-events`.
//! - v1-5: KIVI / Offload `--kv-mode`.
//! - v1-6: `--tensor-partition > 0`.

use anyhow::bail;
use clap::Parser;
use llm_rs2::backend::Backend;
use llm_rs2::buffer::DType;
use llm_rs2::memory::Memory;
use llm_rs2::models::transformer::TransformerModel;
use llm_rs2::pressure::kv_cache::{KVCache, KVLayout};
use llm_rs2::session::cli::{Args, KvMode};
use llm_rs2::session::init::SessionInitCtx;
use llm_rs2::session::is_standard_happy_path;
use llm_rs2::session::resilience_adapter::ResilienceAdapter;
use llm_rs2::session::resilience_init::build_command_executor;
use llm_rs2::session::standard_happy::{StandardHappyCtx, run_standard_happy_path};
use llm_rs2::shape::Shape;
use llm_rs2::tensor::Tensor;
use std::sync::Arc;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut args = Args::parse();

    // v1-1: resilience default-on. `--no-resilience` 가 명시되면 effective=false,
    // 그 외에는 effective=true (legacy `--enable-resilience` flag 도 그대로 효과).
    // SessionInitCtx / prefill / batch 등 호출지는 모두 `args.enable_resilience`
    // 만 참조하므로 진입 직후 1회만 갱신하면 일관된다.
    args.enable_resilience = !args.no_resilience;

    reject_unsupported_modes_v0(&args)?;

    let ctx = SessionInitCtx::build(&args)?;
    let backend = ctx.backend;
    let memory = ctx.memory;
    let cpu_backend_arc = ctx.cpu_backend_arc;
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

    if !is_standard_happy_path(&args) {
        bail!(
            "argus-cli v0: this combination of args is not yet supported. \
             happy path requires: qcf_dump=none, skip_ratio=0, d2o_layer_alloc=off, \
             profile=off, profile_events=off, eviction_policy=none, tensor_partition=0, \
             swap_intra_forward=off, swap_layer_immediate=off, swap_phase_aware=off."
        );
    }
    if args.num_tokens < 1 {
        bail!("argus-cli v0: --num-tokens must be >= 1");
    }

    // P4: ResilienceAdapter 생성. `--no-resilience` 시 None (NoOp default).
    // transport 연결 실패는 Err로 전파 — graceful fail, panic 없음.
    let resilience: Option<ResilienceAdapter> = if args.enable_resilience {
        build_command_executor(&args, &model)?.map(ResilienceAdapter::new)
    } else {
        None
    };

    run_standard_happy_path(StandardHappyCtx {
        args,
        backend,
        memory,
        cpu_backend_arc,
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

/// v0 에서 미구현인 mode 진입 flag 를 검사하여 즉시 reject 한다.
/// 모든 거부 메시지는 향후 갈 곳 (argus-chat / argus-bench / argus-eval) 을 명시.
fn reject_unsupported_modes_v0(args: &Args) -> anyhow::Result<()> {
    if args.chat {
        bail!("argus-cli v0: --chat moved to argus-chat (planned)");
    }
    if args.chat_socket.is_some() || args.chat_tcp.is_some() {
        bail!("argus-cli v0: --chat-socket / --chat-tcp moved to argus-chat (planned)");
    }
    if args.experiment_schedule.is_some() {
        bail!("argus-cli v0: --experiment-schedule moved to argus-eval experiment (planned)");
    }
    if args.experiment_output.is_some() {
        bail!("argus-cli v0: --experiment-output moved to argus-eval experiment (planned)");
    }
    if args.ppl.is_some() {
        bail!("argus-cli v0: --ppl moved to argus-eval ppl (planned)");
    }
    if args.eval_ll || args.eval_batch.is_some() || args.eval_continuation.is_some() {
        bail!(
            "argus-cli v0: --eval-ll / --eval-batch / --eval-continuation moved to argus-eval ll (planned)"
        );
    }
    if args.dump_importance {
        bail!("argus-cli v0: --dump-importance moved to argus-eval dump importance (planned)");
    }
    if args.qcf_dump.is_some() {
        bail!("argus-cli v0: --qcf-dump moved to argus-eval dump qcf (planned)");
    }
    if args.prompt_batch.is_some() {
        bail!("argus-cli v0: --prompt-batch not yet supported (planned for v1)");
    }
    if !matches!(args.effective_kv_mode(), KvMode::Standard) {
        bail!("argus-cli v0: only --kv-mode standard supported (KIVI/Offload planned for v1)");
    }
    if args.secondary_gguf.is_some()
        || args.force_swap_ratio.is_some()
        || args.swap_incremental_per_tick > 0
        || args.swap_intra_forward
        || args.swap_layer_immediate
        || args.swap_phase_aware
    {
        bail!("argus-cli v0: weight swap options not yet supported (planned for v1)");
    }
    if args.profile || args.profile_events {
        bail!("argus-cli v0: --profile / --profile-events not yet supported (planned for v1)");
    }
    if args.tensor_partition > 0.0 {
        bail!("argus-cli v0: --tensor-partition not yet supported (planned for v1)");
    }
    Ok(())
}

/// `--tokenizer-path` 우선, GGUF 면 sibling stem 검색, safetensors 면 dir 안의
/// `tokenizer.json`. legacy `generate` 와 동일한 resolution 순서.
fn resolve_tokenizer_path(args: &Args, model_path: &str, is_gguf: bool) -> String {
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
fn check_vocab_compatibility(
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
fn alloc_standard_kv_caches(
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
            use llm_rs2::quant::{BlockQ4_0, QK4_0};
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
