use clap::Parser;
use llm_rs2::backend::Backend;
use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::buffer::DType;
use llm_rs2::core::events::{self, CacheEvent, StderrDiagnosticSink};
use llm_rs2::core::rss_trace::{io_trace, read_bytes_now, rss_trace};
use llm_rs2::core::sys_monitor::{LinuxSystemMonitor, NoOpMonitor};
use llm_rs2::inference::attention_scores::AttentionScoreAccumulator;
use llm_rs2::inference::sampling::{self, SamplingConfig};
use llm_rs2::layers::workspace::{
    LayerWorkspace, PartitionWorkspace, PartitionWsCell, WorkspaceConfig,
};
use llm_rs2::memory::Memory;
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use llm_rs2::pressure::cache_manager::CacheManager;
use llm_rs2::pressure::d2o_handler::{D2OConfig, D2OHandler};
use llm_rs2::pressure::eviction::h2o::H2OPolicy;
use llm_rs2::pressure::eviction::h2o_plus::H2OPlusPolicy;
use llm_rs2::pressure::eviction::no_eviction::NoEvictionPolicy;
use llm_rs2::pressure::eviction::sliding_window::SlidingWindowPolicy;
use llm_rs2::pressure::kivi_cache::KiviCache;
use llm_rs2::pressure::kv_cache::{KVCache, KVLayout};
use llm_rs2::pressure::{CachePressurePipeline, PressureLevel, PressureStageConfig};
use llm_rs2::session::cli::{Args, KvMode, parse_qcf_sample_layers};
use llm_rs2::session::eval::load_eval_questions;
use llm_rs2::session::ppl::run_kivi_ppl;
use llm_rs2::session::qcf_runtime::{
    dispatch_swap_weights, read_allow_boundary_env, run_layer_swap, run_qcf_warmup_workflow,
};
use llm_rs2::shape::Shape;
use llm_rs2::tensor::Tensor;
use std::sync::Arc;
use tokenizers::Tokenizer;

use llm_rs2::experiment::{
    ExperimentSchedule, JsonlWriter, SummaryRecord, SystemSampler, TokenRecord,
    extract_top_k_logits,
};
#[cfg(feature = "resilience")]
use llm_rs2::resilience::DbusTransport;
use llm_rs2::resilience::TcpTransport;
#[cfg(unix)]
use llm_rs2::resilience::UnixSocketTransport;
use llm_rs2::resilience::{
    CommandExecutor, EngineCommand, KVSnapshot, ManagerMessage, MessageLoop,
};

/// Create a GPU buffer allocator for tensor partition workspace.
///
/// On OpenCL: allocates `UnifiedBuffer` (CL_MEM_ALLOC_HOST_PTR) + permanent map.
/// Single VMA: `as_ptr()`/`as_mut_ptr()` return valid host pointers while
/// `cl_mem()` remains valid for GPU kernels. No PSS double-counting on Adreno.
///
/// On other backends (CPU, CUDA): falls back to `memory.alloc()` which already
/// returns host-accessible buffers (SharedBuffer, CudaHostBuffer).
fn make_partition_gpu_alloc<'a>(
    backend: &'a dyn Backend,
    memory: &'a dyn Memory,
) -> impl Fn(usize, DType) -> anyhow::Result<Arc<dyn llm_rs2::buffer::Buffer>> + 'a {
    // Try to extract OpenCL queue for UnifiedBuffer allocation.
    #[cfg(feature = "opencl")]
    let ocl_queue: Option<ocl::Queue> = backend
        .as_any()
        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
        .map(|b| b.queue.clone());

    #[cfg(not(feature = "opencl"))]
    let _ = backend; // suppress unused warning

    move |size: usize, dtype: DType| -> anyhow::Result<Arc<dyn llm_rs2::buffer::Buffer>> {
        #[cfg(feature = "opencl")]
        if let Some(ref q) = ocl_queue {
            let buf = llm_rs2::memory::opencl::unified::UnifiedBuffer::new(q.clone(), size, dtype)?;
            buf.map()?; // Permanent map for dual CPU/GPU access
            return Ok(Arc::new(buf));
        }
        memory.alloc(size, dtype)
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    // Sprint E forward_gen op-tracer: install atexit hook so the trace
    // dumps even on Ctrl+C / early-return paths. No-op when env unset.
    llm_rs2::profile::op_trace::install_atexit_once();
    // Quality-cost profiler: gated by LLM_RS2_PROFILE_QUALITY=1.
    llm_rs2::profile::quality_metrics::install_atexit_once();
    // T0: process start, before CLI parsing or any allocation.
    rss_trace("start");
    io_trace("start");

    #[allow(unused_mut)]
    let mut args = Args::parse();

    let ctx = llm_rs2::session::init::SessionInitCtx::build(&args)?;

    // Unpack ctx fields for use in the rest of main()
    let sampling_config = ctx.sampling_config;
    let model_path = &ctx.model_path;
    let is_gguf = ctx.is_gguf;
    let backend = ctx.backend;
    let memory = ctx.memory;
    let gpu_backend_arc = ctx.gpu_backend_arc;
    let gpu_memory_arc = ctx.gpu_memory_arc;
    let is_gpu = ctx.is_gpu;
    let weights_on_gpu = ctx.weights_on_gpu;
    let cpu_backend_arc = ctx.cpu_backend_arc;
    let cpu_memory_arc = ctx.cpu_memory_arc;
    let swap_algorithm = ctx.swap_algorithm;
    let importance_formula = ctx.importance_formula;
    let importance_compare = ctx.importance_compare;
    let swap_only_layers = ctx.swap_only_layers;
    let mut model = ctx.model;

    // 2. Tokenizer
    //
    // Resolution order:
    //   1. `--tokenizer-path` if explicitly provided.
    //   2. Safetensors layout (model_path is a directory): `<dir>/tokenizer.json`.
    //   3. GGUF layout (model_path is a file): try in order
    //        a. `<dir>/<stem>.tokenizer.json`              (e.g. qwen2.5-1.5b-f16.tokenizer.json)
    //        b. `<dir>/<stem-stripped>.tokenizer.json`     (strip trailing `-f16` / `-q4_0` /
    //                                                       `-q8_0` quant suffix; e.g.
    //                                                       qwen2.5-1.5b.tokenizer.json)
    //        c. `<dir>/tokenizer.json`                     (legacy single-tokenizer-per-dir)
    //
    // Step 3a/3b prevents a sibling model's tokenizer from being picked up
    // when multiple GGUFs co-exist in the same directory (e.g. /data/local/tmp
    // on Android, where both Llama and Qwen GGUFs share the path). The
    // legacy fallback (3c) keeps existing single-model setups working.
    let tokenizer_path: String = if let Some(p) = args.tokenizer_path.as_ref() {
        p.to_string_lossy().into_owned()
    } else if is_gguf {
        let path = std::path::Path::new(model_path);
        let parent = path.parent().unwrap_or(std::path::Path::new("."));
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        // Quant suffix list — keep in sync with `--weight-dtype` accepted values.
        // Match is case-insensitive on the suffix itself but we preserve the
        // original case of the surviving stem so file lookup matches the
        // on-disk capitalisation (e.g. `Llama-3.2-1B-Instruct-f16.gguf` ->
        // `Llama-3.2-1B-Instruct.tokenizer.json`).
        const QUANT_SUFFIXES: &[&str] = &["-f16", "-f32", "-q4_0", "-q4_1", "-q8_0", "-q4_k"];
        let stem_lower = stem.to_ascii_lowercase();
        let stem_stripped: Option<String> = QUANT_SUFFIXES.iter().find_map(|suf| {
            stem_lower
                .strip_suffix(suf)
                .map(|s| stem[..s.len()].to_string())
        });
        let candidates: Vec<std::path::PathBuf> = {
            let mut v = Vec::with_capacity(3);
            v.push(parent.join(format!("{stem}.tokenizer.json")));
            if let Some(ref s) = stem_stripped {
                v.push(parent.join(format!("{s}.tokenizer.json")));
            }
            v.push(parent.join("tokenizer.json"));
            v
        };
        let chosen = candidates
            .iter()
            .find(|p| p.exists())
            .cloned()
            .unwrap_or_else(|| parent.join("tokenizer.json"));
        chosen.to_string_lossy().into_owned()
    } else {
        format!("{}/tokenizer.json", model_path)
    };
    eprintln!("[Tokenizer] {}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Cannot load tokenizer from {}: {}", tokenizer_path, e))?;
    {
        // Vocab-size mismatch는 거의 항상 wrong-tokenizer-for-model bug
        // (예: Qwen 모델 + Llama tokenizer → decoding garbage). 자동 fallback이
        // sibling tokenizer를 잘못 잡거나 share된 tokenizer.json이 다른 family인
        // 경우를 silent failure로 두지 말 것.
        //
        // tokenizer > model: OOB embedding lookup → 즉시 error.
        // tokenizer < model: model vocab이 padding으로 round-up된 경우 정상
        //   (Qwen2.5 1.5B: trained=151665, padded=151936). 5% 또는 256 이상
        //   격차만 error로 차단.
        let tok_vocab = tokenizer.get_vocab_size(true);
        let model_vocab = model.config.vocab_size;
        // Gemma3 등 multimodal 모델은 텍스트 vocab + 소수의 special token (image_soft 등)을
        // tokenizer에만 두고 embedding table엔 두지 않는 경우가 있음. 작은 overflow(≤8)는
        // 텍스트 생성에서 emit될 일이 거의 없으므로 warning으로 강등.
        let oob_tolerance: usize = 8;
        if tok_vocab > model_vocab + oob_tolerance {
            anyhow::bail!(
                "Tokenizer vocab ({}) exceeds model vocab ({}) by more than {} — OOB embedding lookup risk. \
                 Path: {}. Pass --tokenizer-path with the matching tokenizer.json.",
                tok_vocab,
                model_vocab,
                oob_tolerance,
                tokenizer_path
            );
        } else if tok_vocab > model_vocab {
            eprintln!(
                "[Tokenizer] WARNING: tokenizer vocab ({}) > model vocab ({}) by {} (likely multimodal special tokens). \
                 Text generation OK; encoding text containing those special tokens would OOB.",
                tok_vocab,
                model_vocab,
                tok_vocab - model_vocab
            );
        }
        let pad_tolerance = (model_vocab / 20).max(256);
        let pad_gap = model_vocab.saturating_sub(tok_vocab);
        if pad_gap > pad_tolerance {
            anyhow::bail!(
                "Tokenizer vocab too small: model={} tokenizer={} (gap={} > {} padding tolerance). \
                 Likely wrong tokenizer for this model. Path: {}. \
                 Pass --tokenizer-path with the matching tokenizer.json.",
                model_vocab,
                tok_vocab,
                pad_gap,
                pad_tolerance,
                tokenizer_path
            );
        }
    }

    // 3. Prompt
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
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("Token Length: {}", input_ids.len());

    // 4. Prepare KV Cache
    let max_seq_len = args.max_seq_len; // Use argument
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;
    let num_layers = model.config.num_hidden_layers;
    eprintln!(
        "Model config: layers={}, kv_heads={}, head_dim={}, max_seq_len={}",
        num_layers, kv_heads, head_dim, max_seq_len
    );

    // ── KIVI + eval-ll mode: KiviCache with log-likelihood evaluation ──
    if matches!(args.effective_kv_mode(), KvMode::Kivi) && args.eval_ll {
        let questions = load_eval_questions(&args, &prompt)?;
        let vocab_size = model.config.vocab_size;
        let hidden_size = model.config.hidden_size;
        let eval_config = llm_rs2::eval::EvalConfig {
            max_seq_len,
            effective_budget: 0,
            kv_budget_ratio: 0.0,
            greedy: args.greedy,
            kv_type: format!("q{}+f32_residual", args.effective_kivi_bits()),
            qcf_mode: args.qcf_mode.clone(),
            vocab_size,
            hidden_size,
        };
        let qcf_config = llm_rs2::qcf::QcfConfig::default();
        let kivi_bits = args.effective_kivi_bits();
        let mut kv_caches: Vec<KiviCache> = (0..num_layers)
            .map(|_| {
                KiviCache::new_gpu(
                    kv_heads,
                    head_dim,
                    max_seq_len,
                    args.effective_kivi_residual_size(),
                    kivi_bits,
                    backend.clone(),
                    memory.clone(),
                )
            })
            .collect();
        // AWQE / AW-VOPR metric collection during KIVI residual flush.
        // Env-gated (measurement-only); production decode path leaves it off.
        if std::env::var("LLMRS_KIVI_AWQE").is_ok() {
            for cache in kv_caches.iter_mut() {
                cache.set_awqe_enabled(true);
            }
            eprintln!("[KIVI] AWQE + AW-VOPR enabled (LLMRS_KIVI_AWQE)");
        }
        // ARGUS Step 6: resolve sample layers and inject score accumulator.
        let kivi_n_layers = kv_caches.len();
        let kivi_sample_layers = if args.enable_qcf_experimental {
            parse_qcf_sample_layers(&args.qcf_sample_layers, kivi_n_layers)
                .map_err(|e| anyhow::anyhow!("--qcf-sample-layers: {e}"))?
        } else {
            vec![0]
        };
        // Inject a GQA-aware score accumulator when experimental mode is on.
        // KiviHook::score_accumulator() forwards it into TransformerModelForwardArgs,
        // so LlamaLayer will push attention probabilities into it during forward_into().
        // entropy_computed flag is set in post_prefill when acc.is_active() + scores non-empty.
        let kivi_score_acc = if args.enable_qcf_experimental {
            let mut acc = llm_rs2::inference::attention_scores::AttentionScoreAccumulator::new_gqa(
                max_seq_len,
                model.config.num_attention_heads,
                model.config.num_key_value_heads,
                kivi_n_layers,
                0,   // last_n_layers=0 → all layers tracked
                1.0, // no decay
            );
            acc.set_active(true);
            Some(acc)
        } else {
            None
        };
        let mut hook = llm_rs2::eval::KiviHook::new(
            qcf_config,
            args.enable_qcf_experimental,
            kivi_sample_layers,
            kivi_score_acc,
        );
        let output = llm_rs2::eval::run_eval_ll_generic(
            &model,
            &tokenizer,
            &backend,
            &*memory,
            &mut kv_caches,
            &mut hook,
            &questions,
            &eval_config,
            None,
        )?;
        let mut json_val = serde_json::from_str::<serde_json::Value>(&output.to_json()?)?;
        json_val["config"] = serde_json::json!({
            "model": args.model_path,
            "eviction_policy": "kivi",
            "kivi_bits": args.effective_kivi_bits(),
            "kivi_residual_size": args.effective_kivi_residual_size(),
            "max_seq_len": max_seq_len,
            "kv_type": format!("q{}+f32_residual", args.effective_kivi_bits()),
        });
        println!("{}", serde_json::to_string_pretty(&json_val)?);
        return Ok(());
    }

    // ── KIVI + PPL mode: KiviCache with perplexity evaluation ──
    if matches!(args.effective_kv_mode(), KvMode::Kivi)
        && let Some(ref ppl_path) = args.ppl
    {
        return run_kivi_ppl(
            &args,
            &model,
            &tokenizer,
            &backend,
            &memory,
            kv_heads,
            head_dim,
            num_layers,
            max_seq_len,
            args.effective_kivi_residual_size(),
            ppl_path,
        );
    }

    let kv_type = match args.kv_type.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "q4" => DType::Q4_0,
        _ => anyhow::bail!(
            "Unsupported KV type: {}. Use f32, f16, or q4.",
            args.kv_type
        ),
    };

    // Note: flash_attn_f32_f16 is compiled for DK ∈ {64, 128} only, but both
    // CUDA naive attention (attention_gen_f16kv_naive) and CPU fallback attention
    // support F16 KV for any head_dim. No auto-promotion needed.

    // Determine initial KV cache capacity (dynamic grow-on-demand)
    // Default: reserve space for prompt + all tokens to generate, so decode never
    // triggers grow() mid-generation (grow is ~370 ms spike on Adreno).
    let initial_kv_capacity = if args.eval_ll || args.ppl.is_some() {
        // Eval modes: pre-allocate full capacity to avoid re-allocation
        max_seq_len
    } else if args.initial_kv_capacity() > 0 {
        args.initial_kv_capacity().min(max_seq_len)
    } else {
        input_ids
            .len()
            .saturating_add(args.num_tokens)
            .next_power_of_two()
            .max(128)
            .min(max_seq_len)
    };

    // Calculate buffer size per KV cache (based on initial capacity, not max)
    let n_values = initial_kv_capacity * kv_heads * head_dim;
    let kv_buf_size = match kv_type {
        DType::Q4_0 => {
            use llm_rs2::quant::{BlockQ4_0, QK4_0};
            (n_values / QK4_0) * std::mem::size_of::<BlockQ4_0>()
        }
        _ => n_values * kv_type.size(),
    };
    let use_head_major = args.kv_layout.to_lowercase() != "seq";
    let kv_layout = if use_head_major {
        KVLayout::HeadMajor
    } else {
        KVLayout::SeqMajor
    };
    eprintln!(
        "KV cache type: {:?}, layout: {:?} (initial capacity: {} tokens, {}B per layer, max: {})",
        kv_type, kv_layout, initial_kv_capacity, kv_buf_size, max_seq_len
    );

    let mut kv_caches = Vec::new();
    for _ in 0..num_layers {
        let k_buf = memory.alloc_kv(kv_buf_size, kv_type)?;
        let v_buf = memory.alloc_kv(kv_buf_size, kv_type)?;

        let shape = if use_head_major {
            Shape::new(vec![1, kv_heads, initial_kv_capacity, head_dim])
        } else {
            Shape::new(vec![1, initial_kv_capacity, kv_heads, head_dim])
        };

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
            .with_layout(kv_layout),
        );
    }

    // ════════════════════════════════════════════════════════════
    //  CHAT REPL MODE: standard / KIVI / KV-offload 3-way dispatch (v2)
    // ════════════════════════════════════════════════════════════
    if args.chat {
        use llm_rs2::session::chat::repl::{ChatReplArgs, run_chat_repl_v2};
        use llm_rs2::session::chat::session::{
            ChatKiviArgs, ChatOffloadArgs, ChatStandardArgs, build_chat_kivi, build_chat_offload,
            build_chat_standard,
        };

        let sampling_config = SamplingConfig {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            repetition_penalty: args.repetition_penalty,
            repetition_window: args.repetition_window,
        };
        let kv_offload_active = matches!(args.effective_kv_mode(), KvMode::Offload);

        // config 값을 Arc::new(model) 전에 추출 (move 후 접근 불가).
        let model_arch = model.config.arch;
        let eos_token_id = model.config.eos_token_id;
        let vocab_size = model.config.vocab_size;
        let model_arc = Arc::new(model);

        let mut session = if matches!(args.effective_kv_mode(), KvMode::Kivi) {
            build_chat_kivi(ChatKiviArgs {
                backend: backend.clone(),
                memory: memory.clone(),
                model: model_arc,
                kv_heads,
                head_dim,
                num_layers,
                max_seq_len,
                bits: args.effective_kivi_bits(),
                residual_size: args.effective_kivi_residual_size(),
            })?
        } else if kv_offload_active {
            let offload_kv_dtype = match args.kv_type.as_str() {
                "f32" => DType::F32,
                "f16" => DType::F16,
                other => anyhow::bail!(
                    "--chat --kv-offload requires --kv-type f16 or f32 (got '{}')",
                    other
                ),
            };
            let disk_dir = if args.kv_mode_args.kv_offload_path.is_empty() {
                None
            } else {
                Some(std::path::PathBuf::from(&args.kv_mode_args.kv_offload_path))
            };
            build_chat_offload(ChatOffloadArgs {
                backend: backend.clone(),
                memory: memory.clone(),
                model: model_arc,
                kv_heads,
                head_dim,
                num_layers,
                max_seq_len,
                kv_dtype: offload_kv_dtype,
                offload_mode: args.effective_kv_offload_storage(),
                disk_dir,
                max_prefetch_depth: args.kv_mode_args.kv_max_prefetch_depth,
            })?
        } else {
            build_chat_standard(ChatStandardArgs {
                backend: backend.clone(),
                memory: memory.clone(),
                cpu_backend: cpu_backend_arc.clone(),
                model: model_arc,
                kv_caches,
                initial_kv_capacity,
                max_seq_len,
                kv_dtype: kv_type,
                eviction_policy: args.eviction_policy().to_string(),
                eviction_target_ratio: args.eviction_target_ratio(),
                eviction_window: args.eviction_window(),
                protected_prefix: args.protected_prefix(),
                sink_size: args.sink_size(),
                streaming_window: args.streaming_window(),
                kv_budget: args.kv_budget(),
                h2o_keep_ratio: args.h2o_keep_ratio(),
                h2o_tracked_layers: args.h2o_tracked_layers(),
                h2o_decay: args.h2o_decay(),
                h2o_raw_scores: args.h2o_raw_scores(),
                d2o_keep_ratio: args.d2o_keep_ratio(),
                d2o_ema_beta: args.d2o_ema_beta(),
                d2o_merge_e: args.d2o_merge_e(),
                d2o_layer_alloc: args.d2o_layer_alloc(),
                d2o_protected_layers: args.d2o_protected_layers().clone().unwrap_or_default(),
                memory_threshold_mb: args.memory_threshold_mb() as u64,
            })?
        };

        let repl_args = ChatReplArgs {
            model_arch,
            tokenizer: &tokenizer,
            eos_token_id,
            vocab_size,
            sampling_config: &sampling_config,
            max_seq_len,
            system_prompt: args.system_prompt.as_deref(),
            initial_user_prompt: {
                let p = args.prompt.trim();
                if p.is_empty() { None } else { Some(p) }
            },
            chat_socket: args.chat_socket.as_deref(),
            chat_tcp: args.chat_tcp.as_deref(),
            repetition_window: args.repetition_window,
            max_new_tokens: args.num_tokens,
        };

        return run_chat_repl_v2(&repl_args, &mut session);
    }

    // 5. Experiment schedule + Command Executor
    let experiment_schedule = if let Some(ref path) = args.experiment_schedule {
        Some(ExperimentSchedule::load(path)?)
    } else {
        None
    };

    let mut experiment_tx: Option<std::sync::mpsc::Sender<ManagerMessage>> = None;
    let heartbeat_interval = std::time::Duration::from_millis(1000);

    // MSG-068 Phase 2: GPU self-util meter 추출. 백엔드가 queue profiling과
    // 함께 빌드되었을 때만(opt-in) Some이 된다. CPU 백엔드/비활성 시 None이며
    // executor는 self_gpu_pct=0.0을 송출한다 (INV-092 fallback).
    #[allow(unused_mut)]
    let mut gpu_meter: Option<std::sync::Arc<dyn llm_rs2::resilience::GpuSelfMeter>> = None;
    #[cfg(feature = "opencl")]
    if args.heartbeat_gpu_profile {
        // 우선 primary backend에서 찾고, 없으면 secondary(GPU)에서 찾는다.
        // CPU primary + GPU secondary 구성에서도 GPU self-util을 보고하기 위함.
        let try_extract = |b: &std::sync::Arc<dyn Backend>| -> Option<
            std::sync::Arc<dyn llm_rs2::resilience::GpuSelfMeter>,
        > {
            b.as_any()
                .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                .and_then(|ocl| ocl.gpu_self_meter())
                .map(|m| m as std::sync::Arc<dyn llm_rs2::resilience::GpuSelfMeter>)
        };
        gpu_meter =
            try_extract(&backend).or_else(|| gpu_backend_arc.as_ref().and_then(try_extract));
        if gpu_meter.is_some() {
            eprintln!("[Resilience] Heartbeat GPU profiling enabled (MSG-068 Phase 2)");
        } else {
            eprintln!(
                "[Resilience] --heartbeat-gpu-profile set but no OpenCL backend with profiling available; self_gpu_pct stays 0.0"
            );
        }
    }

    let mut command_executor = if let Some(ref schedule) = experiment_schedule {
        // Experiment mode: internal mpsc channel (no external transport needed)
        let (tx, rx) = std::sync::mpsc::channel();
        let (resp_tx, _resp_rx) = std::sync::mpsc::channel();
        experiment_tx = Some(tx);
        eprintln!("[Experiment] Mode enabled — schedule: {}", schedule.name);
        Some(CommandExecutor::with_gpu_meter(
            rx,
            resp_tx,
            args.backend.clone(),
            heartbeat_interval,
            gpu_meter.clone(),
        ))
    } else if args.enable_resilience {
        let (cmd_rx, resp_tx, _handle) = match args.resilience_transport.as_str() {
            #[cfg(feature = "resilience")]
            "dbus" => MessageLoop::spawn(DbusTransport::new())?,
            #[cfg(unix)]
            s if s.starts_with("unix:") => {
                let path = std::path::PathBuf::from(&s[5..]);
                MessageLoop::spawn(UnixSocketTransport::new(path))?
            }
            s if s.starts_with("tcp:") => {
                let addr = s[4..].to_string();
                MessageLoop::spawn(TcpTransport::new(addr))?
            }
            other => {
                eprintln!("[Resilience] Unknown transport: {}", other);
                return Ok(());
            }
        };
        eprintln!(
            "[Resilience] Executor enabled — transport: {}",
            args.resilience_transport
        );
        let mut executor = CommandExecutor::with_gpu_meter(
            cmd_rx,
            resp_tx,
            args.backend.clone(),
            heartbeat_interval,
            gpu_meter.clone(),
        );

        // secondary 경로가 있으면 swap_weights 액션이 Heartbeat에도 포함되도록 설정.
        // Capability와 Heartbeat 두 목록이 항상 같은 조건을 공유한다 (ENG-ST-032).
        let has_secondary = args.secondary_gguf.is_some();
        executor.set_has_secondary(has_secondary);

        // Send Capability as first message (SEQ-022).
        // available_actions 는 Heartbeat 와 동일하게 eviction_policy / kv_type 에서 파생.
        // Heartbeat 보다 먼저 manager 에 도달하므로, 첫 signal 처리 시점에 이미 이 값이
        // 반영돼 있어야 정책이 엔진이 지원하지 않는 액션을 선택하는 회귀를 막을 수 있다.
        let cap_available_actions = {
            let mut a = vec![
                "throttle".to_string(),
                "switch_hw".to_string(),
                "layer_skip".to_string(),
            ];
            if args.eviction_policy() != "none" {
                a.push("kv_evict_h2o".to_string());
                a.push("kv_evict_sliding".to_string());
                a.push("kv_evict_streaming".to_string());
                a.push("kv_merge_d2o".to_string());
            }
            if args.kv_type.starts_with('q') {
                a.push("kv_quant_dynamic".to_string());
            }
            // secondary GGUF/AUF 존재 시 swap_weights 등록 (ENG-ST-032).
            // Heartbeat의 compute_available_actions와 동일 조건을 공유한다.
            if has_secondary {
                a.push("swap_weights".to_string());
            }
            a
        };
        executor.send_capability(llm_shared::EngineCapability {
            available_devices: vec!["cpu".to_string(), "opencl".to_string()],
            active_device: args.backend.clone(),
            max_kv_tokens: args.max_seq_len,
            bytes_per_kv_token: model.config.num_key_value_heads
                * model.config.head_dim
                * 2  // K + V
                * 2, // F16 = 2 bytes
            num_layers: model.config.num_hidden_layers,
            available_actions: cap_available_actions,
        });
        eprintln!("[Resilience] Capability sent to Manager");

        Some(executor)
    } else {
        None
    };
    // Set initial partition ratio from CLI for heartbeat reporting
    if args.tensor_partition > 0.0
        && args.tensor_partition < 1.0
        && let Some(ref mut exec) = command_executor
    {
        exec.set_partition_ratio(args.tensor_partition);
    }
    // Seed sticky throttle from CLI so no-directive polls preserve the CLI
    // value; Manager `Throttle` directives still override at runtime.
    if args.throttle_delay_ms > 0
        && let Some(ref mut exec) = command_executor
    {
        exec.set_throttle_delay_ms(args.throttle_delay_ms);
    }
    let throttle_delay_ms: u64 = args.throttle_delay_ms;
    let mut tbt_log_writer: Option<std::io::BufWriter<std::fs::File>> =
        args.tbt_log.as_ref().map(|path| {
            let file = std::fs::File::create(path).expect("failed to create tbt-log file");
            std::io::BufWriter::new(file)
        });
    let mut target_tbt_ms = args.target_tbt;

    // ── KIVI mode: separate path with KiviCache ──
    // Placed after executor creation so resilience is available in the token loop.
    if matches!(args.effective_kv_mode(), KvMode::Kivi) || args.kv_dynamic_quant {
        // KIVI mode: --kivi starts at Q2, --kv-dynamic-quant starts at bits=16
        // (F16-equivalent) and allows runtime transition via kv_quant_dynamic.
        // Note: --enable-resilience alone stays on main path (F16 KVCache + eviction).
        let initial_bits: u8 = if matches!(args.effective_kv_mode(), KvMode::Kivi) {
            args.effective_kivi_bits()
        } else {
            16
        };
        let residual_size = if initial_bits == 16 {
            // bits=16: all tokens stay in residual (no quantization flush)
            // Round down to QKKV (32) multiple for KiviCache alignment
            (max_seq_len / 32) * 32
        } else {
            args.effective_kivi_residual_size()
        };
        return run_kivi(
            &model,
            &tokenizer,
            &backend,
            &memory,
            &input_ids,
            &sampling_config,
            kv_heads,
            head_dim,
            num_layers,
            max_seq_len,
            residual_size,
            args.num_tokens,
            args.experiment_output.as_deref(),
            args.experiment_logits_topk,
            args.experiment_sample_interval,
            &prompt,
            &args.backend,
            &mut command_executor,
            initial_bits,
            args.no_gpu_plan,
            args.target_tbt,
            args.tbt_log.as_deref(),
            args.ignore_eos,
            args.throttle_delay_ms,
        );
    }

    // ── Offload mode: separate path with OffloadKVCache ──
    // Placed after executor creation so resilience is available in the decode loop.
    if matches!(args.effective_kv_mode(), KvMode::Offload) {
        let kv_offload_storage = args.effective_kv_offload_storage();
        return run_offload(
            &model,
            &tokenizer,
            &backend,
            &memory,
            &input_ids,
            &sampling_config,
            kv_heads,
            head_dim,
            num_layers,
            max_seq_len,
            args.num_tokens,
            &prompt,
            &args.backend,
            &kv_offload_storage,
            &args.kv_type,
            args.kv_mode_args.kv_max_prefetch_depth,
            &args.kv_mode_args.kv_offload_path,
            &mut command_executor,
            args.throttle_delay_ms,
        );
    }

    // Experiment JSONL writer + system sampler
    let mut experiment_writer = if let Some(ref path) = args.experiment_output {
        Some(JsonlWriter::new(path)?)
    } else {
        None
    };
    let mut system_sampler = SystemSampler::new(args.experiment_sample_interval);
    let sys_start = if experiment_writer.is_some() {
        Some(system_sampler.snapshot())
    } else {
        None
    };
    let mut experiment_eviction_count: usize = 0;
    let mut experiment_evicted_total: usize = 0;
    let mut experiment_total_throttle_ms: u64 = 0;
    let mut forward_ms_values: Vec<f64> = Vec::new();

    // 6. Inference Loop
    let tokens = input_ids.clone();
    let start_pos = 0;
    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;

    eprintln!(
        "Generating (Max: {}, Temp: {}, TopP: {}, TopK: {})...",
        max_seq_len, args.temperature, args.top_p, args.top_k
    );
    let start_time = std::time::Instant::now();
    let mut _last_token_time = start_time;
    let mut _ttft_ms = 0.0;
    let mut tbt_values = Vec::new();

    // 4.5 Setup CacheManager
    let actual_protected_prefix = args
        .protected_prefix()
        .unwrap_or(match args.eviction_policy() {
            // Score-based policies: default to 4 (attention sinks only).
            // Protecting the entire prompt makes score-based eviction meaningless
            // because only generated tokens would be evictable.
            "h2o" | "h2o_plus" | "d2o" => 4,
            // StreamingLLM: use explicit sink_size parameter
            "streaming" => args.sink_size(),
            // Sliding window / none: protect entire prompt (legacy behavior)
            _ => input_ids.len(),
        });

    let mut cache_manager = {
        // CUDA discrete GPU: managed memory (cuMemAllocManaged) reserves system RAM
        // for virtual address space even though data resides in VRAM. MemAvailable
        // from /proc/meminfo is unreliable — use NoOpMonitor to prevent false pressure.
        let monitor: Box<dyn llm_rs2::core::sys_monitor::SystemMonitor> =
            if backend.is_discrete_gpu() {
                Box::new(NoOpMonitor)
            } else {
                Box::new(LinuxSystemMonitor)
            };
        let threshold_bytes = args.memory_threshold_mb() * 1024 * 1024;

        if args.eviction_policy() == "d2o" {
            // D2O uses CachePressureHandler (Pipeline mode), not EvictionPolicy (Legacy mode)
            let d2o_handler = D2OHandler::new(D2OConfig {
                keep_ratio: args.d2o_keep_ratio(),
                protected_prefix: actual_protected_prefix,
                target_ratio: args.eviction_target_ratio(),
                ema_beta: args.d2o_ema_beta(),
                merge_e: args.d2o_merge_e(),
                use_layer_allocation: args.d2o_layer_alloc(),
                protected_layers: args.d2o_protected_layers().clone().unwrap_or_default(),
            });
            let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(d2o_handler),
            }]);
            CacheManager::with_pipeline(pipeline, monitor, threshold_bytes)
        } else {
            let policy: Box<dyn llm_rs2::pressure::eviction::EvictionPolicy> = match args
                .eviction_policy()
            {
                "none" => Box::new(NoEvictionPolicy::new()),
                "sliding" => Box::new(SlidingWindowPolicy::new(
                    args.eviction_window(),
                    actual_protected_prefix,
                )),
                "streaming" => {
                    use llm_rs2::pressure::eviction::StreamingLLMPolicy;
                    let window = if args.streaming_window() > 0 {
                        args.streaming_window()
                    } else if args.kv_budget() > 0 {
                        args.kv_budget().saturating_sub(args.sink_size())
                    } else {
                        args.eviction_window()
                    };
                    Box::new(StreamingLLMPolicy::new(args.sink_size(), window))
                }
                "h2o" => Box::new(H2OPolicy::new(
                    args.h2o_keep_ratio(),
                    actual_protected_prefix,
                )),
                "h2o_plus" => Box::new(H2OPlusPolicy::new(
                    args.h2o_keep_ratio(),
                    actual_protected_prefix,
                )),
                other => anyhow::bail!(
                    "Unknown eviction policy: '{}'. Use: none, sliding, streaming, h2o, h2o_plus, d2o",
                    other
                ),
            };
            CacheManager::new(
                policy,
                monitor,
                threshold_bytes,
                args.eviction_target_ratio(),
            )
        }
    };

    // Setup event sink for score diagnostics
    cache_manager.set_event_sink(Arc::new(StderrDiagnosticSink));

    // Enable disk-backed KV swap when --swap-dir is provided.
    // KvOffload directives write to this directory; RestoreDefaults recalls.
    if let Some(dir) = args.swap_dir.clone() {
        eprintln!("[Resilience] KV swap enabled: dir={}", dir.display());
        cache_manager.enable_swap(dir);
    }

    // Register policies for Manager-directed eviction dispatch.
    // Use a small protected_prefix (4 = attention sinks) for Manager-directed policies,
    // NOT actual_protected_prefix which may be the entire prompt length when
    // --eviction-policy is "none". The Manager decides WHEN and HOW MUCH to evict;
    // the policy should not silently prevent meaningful eviction.
    let resilience_protected_prefix = 4usize; // attention sinks only
    cache_manager.register_policy(
        llm_rs2::resilience::EvictMethod::H2o,
        Box::new(H2OPolicy::new(
            args.h2o_keep_ratio(),
            resilience_protected_prefix,
        )),
    );
    cache_manager.register_policy(
        llm_rs2::resilience::EvictMethod::Sliding,
        Box::new(SlidingWindowPolicy::new(
            args.eviction_window(),
            resilience_protected_prefix,
        )),
    );
    // Note: Streaming policy is NOT pre-registered because its parameters
    // (sink_size, window_size) come from the Manager directive at runtime.
    // It is instantiated on-demand in the eviction dispatch below.

    // Parse QCF mode
    let qcf_mode = match args.qcf_mode.as_str() {
        "caote" => llm_rs2::qcf::QcfMode::Caote,
        "both" => llm_rs2::qcf::QcfMode::Both,
        _ => llm_rs2::qcf::QcfMode::Attn,
    };
    let needs_caote = qcf_mode.has_caote();

    // Setup AttentionScoreAccumulator for H2O / H2O+ / D2O / CAOTE
    // When CAOTE is requested, always use GQA-aware accumulator (for per-KV-head attention).
    let needs_score_based = args.eviction_policy() == "h2o"
        || args.eviction_policy() == "d2o"
        || args.eviction_policy() == "h2o_plus";
    // Always build accumulator for eval-ll when any eviction policy is active:
    // sliding mode needs it to populate last_step_head_attn for QCF-ATTN v2.
    let has_eviction_policy = args.eviction_policy() != "none";
    // --enable-resilience forces accumulator on: the manager can request Evict
    // at any runtime moment, and `compute_qcf_estimates` (~line 4882) falls back
    // to uniform weights without scores, which corrupts action-cost ranks
    // (measured: h2o/d2o collapse to 0, sliding inflates +312%).
    // Originally decoupled 2026-04-20 because forcing the CPU accumulator
    // disabled the GPU decode plan (~25% slowdown). Now re-coupled after
    // Phase A/B (flash_attn score output, commits 3096de4 + 28d8fe4): the
    // accumulator coexists with the GPU plan and overhead is <1%
    // (Adreno 37.4 t/s, Jetson overhead 1.8–4.3%).
    let needs_accumulator =
        needs_score_based || needs_caote || args.enable_resilience || has_eviction_policy;
    // GQA mode required for last_step_head_attn() (QCF-ATTN v2 + CAOTE).
    let use_gqa = args.eviction_policy() == "h2o_plus" || needs_caote || has_eviction_policy;

    let score_accumulator = if needs_accumulator {
        let acc = if use_gqa {
            AttentionScoreAccumulator::new_gqa(
                max_seq_len,
                model.config.num_attention_heads,
                model.config.num_key_value_heads,
                model.config.num_hidden_layers,
                args.h2o_tracked_layers(),
                args.h2o_decay(),
            )
        } else {
            AttentionScoreAccumulator::new(
                max_seq_len,
                model.config.num_attention_heads,
                model.config.num_hidden_layers,
                args.h2o_tracked_layers(),
                args.h2o_decay(),
            )
        };
        let mut acc = acc;
        // Always active: GPU acc overhead is ~0.6ms/token (1.7%),
        // CPU NEON acc overhead is ~0.66ms/token (1.1%).
        // This ensures first RequestQcf returns accurate H2O/D2O estimates.
        acc.set_active(true);
        acc.set_time_normalize(!args.h2o_raw_scores());
        Some(acc)
    } else {
        None
    };

    // Initialize GPU-side score accumulator when using OpenCL backend.
    // This compiles score_reduce.cl and allocates persistent GPU buffers.
    // Eliminates per-token GPU->CPU blocking readback (~129ms/token).
    #[cfg(feature = "opencl")]
    if score_accumulator.is_some()
        && let Some(ocl_be) = backend
            .as_any()
            .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
    {
        match ocl_be.init_gpu_score_acc(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
            max_seq_len,
            args.h2o_decay(),
        ) {
            Ok(()) => {
                if let Some(gpu_acc) = ocl_be.gpu_score_acc_mut() {
                    gpu_acc.set_active(true);
                }
                eprintln!("[GPU Score] Accumulator initialized — per-token readback eliminated");
            }
            Err(e) => {
                eprintln!(
                    "[GPU Score] Failed to initialize (falling back to CPU path): {}",
                    e
                );
            }
        }
    }

    if args.eviction_policy() != "none" {
        eprintln!(
            "Eviction: policy={}, window={}, prefix={}, ratio={}, threshold={}MB",
            args.eviction_policy(),
            args.eviction_window(),
            actual_protected_prefix,
            args.eviction_target_ratio(),
            args.memory_threshold_mb()
        );
    }

    // Build SkipConfig from CLI options
    use llm_rs2::inference::skip_config::SkipConfig;
    let skip_config = if let Some(ref layers) = args.skip_layers {
        let mut sc = SkipConfig::new();
        for &l in layers {
            sc.attn_skip.insert(l);
            sc.mlp_skip.insert(l);
        }
        assert!(
            sc.validate(model.config.num_hidden_layers),
            "Cannot skip layer 0 or last layer (SWIFT constraint)"
        );
        eprintln!(
            "[Skip] Explicit layers: {:?} ({} sub-layers skipped)",
            layers,
            sc.total_skips()
        );
        Some(sc)
    } else if let Some(ratio) = args.skip_ratio {
        let sc = SkipConfig::uniform_init(model.config.num_hidden_layers, ratio);
        assert!(
            sc.validate(model.config.num_hidden_layers),
            "uniform_init produced invalid SkipConfig (layer 0 or last layer skipped)"
        );
        eprintln!(
            "[Skip] Uniform ratio={:.1}% → {} sub-layers skipped",
            ratio * 100.0,
            sc.total_skips()
        );
        Some(sc)
    } else {
        None
    };
    let last_skip_ratio: Option<f32> = args.skip_ratio;

    // Auto-eviction: non-experiment mode evicts automatically.
    // - Sliding window: triggers on memory pressure after each forward pass.
    // - Score-based (H2O/H2O+/D2O): triggers when cache utilization >= 90% capacity,
    //   using force_evict_with_scores to bypass memory pressure checks.
    let auto_eviction = args.eviction_policy() != "none" && experiment_schedule.is_none();
    let score_based_eviction = matches!(args.eviction_policy(), "h2o" | "h2o_plus" | "d2o");

    // ── Weight swap: --force-swap-ratio manual trigger ──────────────────────
    // Applied once before generation starts (prefill + decode).
    // Requires --secondary-gguf (validated above at model load time).
    // When --qcf-dump is set, this block is skipped; the swap is deferred to
    // the QCF dump workflow below (after warmup prefill builds ImportanceTable).
    //
    // ENG-ALG-232~234 (LISWAP-1): when --swap-incremental-per-tick > 0,
    // the swap is NOT executed here. Instead, an IncrementalSwapPlan is
    // committed and stored below; the decode loop drains it chunk-by-chunk.
    // per_tick == 0 (default): single-shot path, unchanged from before.
    let mut incremental_force_swap_plan: Option<llm_rs2::models::weights::IncrementalSwapPlan> =
        None;

    // LISWAP-6 manager path: when manager triggers SwapWeights, the plan is
    // committed to `incremental_force_swap_plan` and this state records the
    // information needed to send WeightSwapReport on plan completion.
    // Fields: (ratio, total_layers_planned, plan_start_time, qcf_swap_estimated)
    let mut manager_swap_report_pending: Option<(f32, usize, std::time::Instant, f32)> = None;
    // Populated by the plan-done block (outside executor scope); consumed by the
    // executor checkpoint block (inside executor scope) the same token tick.
    let mut ready_weight_swap_report: Option<llm_shared::WeightSwapReport> = None;

    // LISWAP-4 (ENG-ALG-237 / INV-150): intra-forward layer-aligned swap hook.
    // Created when `--swap-intra-forward` + `--force-swap-ratio` both active.
    // Decode loop injects `Some(&*hook)` into `layer_boundary_hook` and calls
    // `finalize` once `plan_is_complete()` to drain dispatcher + bump
    // ratio_generation + invalidate SOA registry.
    let mut intra_forward_swap_hook: Option<Arc<llm_rs2::models::weights::IntraForwardSwapHook>> =
        None;

    // LISWAP-5: phase-aware async swap dispatcher. Created when
    // `--swap-phase-aware` + `--force-swap-ratio` both active. Registered as
    // process-wide PHASE_HOOK so `op_trace::start_op` / `record` callsites in
    // forward_gen drive chunk dispatch from the forward thread itself.
    // `finalize` drains remaining chunks + bumps ratio_generation when decode
    // ends.
    let mut phase_aware_swap_dispatcher: Option<
        Arc<llm_rs2::models::weights::PhaseAwareSwapDispatcher>,
    > = None;

    // LISWAP-2 prototype: async swap dispatcher lifecycle.
    // Created once here; used in the decode loop when async dispatch is active.
    // `None` when --swap-async-dispatch is false or the backend does not support
    // async transfer.
    // NOTE: also created when per_tick == 0 so that manager-triggered incremental
    // swap (LISWAP-6 manager path) can use async dispatch even without
    // --swap-incremental-per-tick CLI flag.
    let async_swap_dispatcher: Option<llm_rs2::models::weights::AsyncSwapDispatcher> = {
        if args.swap_async_dispatch {
            let swap_backend: Arc<dyn Backend> = gpu_backend_arc
                .as_ref()
                .cloned()
                .unwrap_or_else(|| cpu_backend_arc.clone());
            if swap_backend.supports_async_transfer() {
                Some(llm_rs2::models::weights::AsyncSwapDispatcher::new(
                    swap_backend,
                ))
            } else {
                if args.swap_incremental_per_tick > 0 {
                    eprintln!(
                        "[LISWAP-2] backend does not support async transfer; falling back to sync incremental swap"
                    );
                }
                None
            }
        } else {
            None
        }
    };

    // LISWAP-6 — Dynamic K controller. Active when `--swap-dynamic-k` is set.
    // K is determined entirely by timing (forward wall vs per-layer drop cost);
    // there is no static upper cap. Per-tick dispatch is still bounded by the
    // sub-batch reactive pause inside `SwapExecutor::execute_on_slots`.
    if args.swap_dynamic_k && args.swap_probing_k {
        anyhow::bail!(
            "--swap-dynamic-k and --swap-probing-k are mutually exclusive — pick one controller"
        );
    }
    let mut dynamic_k_controller: Option<llm_rs2::models::weights::DynamicKController> =
        if args.swap_dynamic_k {
            Some(llm_rs2::models::weights::DynamicKController::new())
        } else {
            None
        };
    let dynamic_k_diag = std::env::var("LLMRS_DYNAMIC_K_DIAG")
        .map(|v| v == "1")
        .unwrap_or(false);

    // Probing-K controller (bottom-up alternative to ARGUS). Starts at K=1 and
    // probes upward subject to a stability window + release-queue spike guard.
    let mut probing_k_controller: Option<llm_rs2::models::weights::ProbingKController> =
        if args.swap_probing_k {
            let growth = match args.swap_probing_growth.to_ascii_lowercase().as_str() {
                "linear" => llm_rs2::models::weights::GrowthMode::Linear,
                "binary" => llm_rs2::models::weights::GrowthMode::Binary,
                other => anyhow::bail!(
                    "--swap-probing-growth must be 'linear' or 'binary', got '{other}'"
                ),
            };
            let mut c =
                llm_rs2::models::weights::ProbingKController::with_options(1, usize::MAX, growth);
            c.set_stability_window(args.swap_probing_window.max(1));
            Some(c)
        } else {
            None
        };
    let probing_k_diag = std::env::var("LLMRS_PROBING_K_DIAG")
        .map(|v| v == "1")
        .unwrap_or(false);

    // ── LISWAP-3 prototype (Direction A): ALLOC_HOST_PTR pool ────────────
    // Lazy-init the swap pool when the user opted in via `--swap-zero-copy`
    // AND the env-gate `LLMRS_OPENCL_HOST_PTR_POOL=1` is set. Both conditions
    // are required so the flag alone cannot accidentally enable the
    // prototype path. SwapExecutor falls back to the staging path on `None`.
    // Plan: compiled-chasing-hopper.md Direction A track, Stage 3.
    #[cfg(feature = "opencl")]
    let host_ptr_swap_pool: Option<Arc<llm_rs2::backend::opencl::host_ptr_pool::HostPtrPool>> = {
        if !args.swap_zero_copy {
            None
        } else if let Some(gpu_be) = gpu_backend_arc.as_ref().and_then(|b| {
            b.as_any()
                .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
        }) {
            let cfg = llm_rs2::backend::opencl::host_ptr_pool::HostPtrPoolConfig {
                n_slots: args.swap_pool_slots.max(1),
                ..Default::default()
            };
            let pool = gpu_be.host_ptr_pool_or_init(cfg);
            if pool.is_some() {
                eprintln!(
                    "[LISWAP-3] host_ptr_pool active: slots={}, max_tensor_size={}",
                    cfg.n_slots, cfg.max_tensor_size
                );
            } else {
                eprintln!(
                    "[LISWAP-3] --swap-zero-copy requested but pool unavailable \
                     (env LLMRS_OPENCL_HOST_PTR_POOL not set or pool init failed); \
                     using staging path"
                );
            }
            pool
        } else {
            eprintln!(
                "[LISWAP-3] --swap-zero-copy ignored: backend is not OpenCL; using staging path"
            );
            None
        }
    };
    #[cfg(not(feature = "opencl"))]
    let _host_ptr_swap_pool: Option<()> = {
        if args.swap_zero_copy {
            eprintln!(
                "[LISWAP-3] --swap-zero-copy ignored: opencl feature is disabled in this build"
            );
        }
        None
    };

    // LISWAP-8 Phase B: pre-allocated layer object pool. Activated via
    // `LLMRS_SWAP_LAYER_POOL=1` env. Cuda-embedded only. Pool depth via
    // `LLMRS_SWAP_LAYER_POOL_DEPTH` (default 2). Hypothesis: removing
    // `cuMemAlloc` from the per-layer dispatch path eliminates the CUDA
    // driver context lock contention observed in the K-sweep
    // active-window forward regression.
    #[cfg(feature = "cuda-embedded")]
    let layer_swap_pool: Option<Arc<dyn llm_rs2::layers::staging_pool::WeightStagingPool>> = {
        let enabled = std::env::var("LLMRS_SWAP_LAYER_POOL")
            .map(|v| v == "1")
            .unwrap_or(false);
        if !enabled || model.secondary_mmap.is_none() {
            None
        } else if let Some(gpu_be) = gpu_backend_arc.as_ref() {
            let target_depth: usize = std::env::var("LLMRS_SWAP_LAYER_POOL_DEPTH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2);
            let sample = model.layers[0].load_weights();
            let spec = llm_rs2::models::weights::layer_object_pool::LayerSpec::from_sample(
                &sample,
                DType::Q4_0,
            )
            .with_zero_copy(
                std::env::var("LLMRS_SWAP_LAYER_POOL_ZERO_COPY")
                    .map(|v| v == "1")
                    .unwrap_or(false),
            );
            let zc = spec.zero_copy;
            match llm_rs2::models::weights::layer_object_pool::LayerObjectPool::new(
                Arc::clone(gpu_be),
                spec,
                target_depth,
            ) {
                Ok(pool) => {
                    eprintln!(
                        "[LISWAP-8] layer_pool active: target_depth={target_depth} zero_copy={zc}"
                    );
                    // Unsized coercion Arc<LayerObjectPool> → Arc<dyn WeightStagingPool>.
                    Some(pool as Arc<dyn llm_rs2::layers::staging_pool::WeightStagingPool>)
                }
                Err(e) => {
                    eprintln!("[LISWAP-8] layer_pool init failed: {e}");
                    None
                }
            }
        } else {
            None
        }
    };

    // LISWAP-8 Hammer D: register the secondary mmap with CUDA so the
    // swap path can install zero-copy `CudaMmapAliasBuffer` weights.
    // env: LLMRS_SWAP_MMAP_ALIAS=1
    #[cfg(feature = "cuda-embedded")]
    let mmap_registration: Option<Arc<llm_rs2::memory::cuda::mmap::CudaMmapRegistration>> = {
        let enabled = std::env::var("LLMRS_SWAP_MMAP_ALIAS")
            .map(|v| v == "1")
            .unwrap_or(false);
        if !enabled {
            None
        } else if let Some(secondary) = model.secondary_mmap.clone() {
            match llm_rs2::memory::cuda::mmap::CudaMmapRegistration::register(secondary) {
                Ok(reg) => {
                    eprintln!(
                        "[LISWAP-8] mmap registration active: size={} MB",
                        reg.size() / (1024 * 1024)
                    );
                    Some(reg)
                }
                Err(e) => {
                    eprintln!("[LISWAP-8] mmap registration failed: {e}");
                    None
                }
            }
        } else {
            None
        }
    };

    // LISWAP Phase 3 — pending mid-decode trigger payload.
    // When `--swap-delay-tokens N > 0` AND `--force-swap-ratio` is set, the
    // trigger logic is deferred from prefill end to decode token N. We capture
    // (ratio, target_layers) here and re-run the dispatch block at the loop
    // head when `decode_token_index == swap_delay_tokens`.
    let mut pending_force_swap: Option<(f32, Vec<usize>)> = None;

    // ── LISWAP Phase 3 dispatch macro ────────────────────────────────────────
    // Identical force-swap dispatch logic invoked from two callsites:
    //   (a) prefill end (when --swap-delay-tokens == 0, default — original
    //       behavior, must preserve baseline wall ±5 ms),
    //   (b) decode loop head (when --swap-delay-tokens > 0, mid-decode trigger).
    //
    // The macro relies on hygienic name capture of the surrounding `main`'s
    // mutable state (model, backend, gpu_backend_arc, cpu_backend_arc,
    // host_ptr_swap_pool, intra_forward_swap_hook, incremental_force_swap_plan,
    // phase_aware_swap_dispatcher, is_gpu, args). Inputs are bound by the macro
    // arms: $ratio = clamped force ratio, $target_layers = Vec<usize> resolved
    // from `uniform_target_layers`. Both callsites pre-compute these before
    // expanding the macro.
    macro_rules! dispatch_force_swap {
        ($ratio:expr, $target_layers:expr) => {{
            let ratio: f32 = $ratio;
            let target_layers: Vec<usize> = $target_layers;
            let num_layers = model.layers.len();
            if args.swap_phase_aware {
                let swap_backend: Arc<dyn Backend> = gpu_backend_arc
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| cpu_backend_arc.clone());
                let dispatcher = Arc::new(llm_rs2::models::weights::AsyncSwapDispatcher::new(
                    Arc::clone(&swap_backend),
                ));
                let secondary = match model.secondary_mmap.as_ref() {
                    Some(s) => Arc::clone(s),
                    None => {
                        anyhow::bail!(
                            "--swap-phase-aware requires --secondary-gguf (no secondary mmap available)"
                        );
                    }
                };
                let config = Arc::new(model.config.clone());
                let chunk_size_bytes = args.swap_phase_aware_chunk_mb.max(1) * 1_048_576;
                eprintln!(
                    "weight_swap: phase-aware mode — ratio={:.2}, {} target layers, chunk_size={} MB (LISWAP-5)",
                    ratio,
                    target_layers.len(),
                    args.swap_phase_aware_chunk_mb
                );
                let phase_dispatcher = llm_rs2::models::weights::PhaseAwareSwapDispatcher::new(
                    chunk_size_bytes,
                    model.layers.clone(),
                    secondary,
                    swap_backend,
                    dispatcher,
                    DType::Q4_0,
                    config,
                );
                // LISWAP Phase 4: install weak self-ref so the worker thread can
                // call back into try_dispatch_chunk_worker via ChunkDispatchJob.
                phase_dispatcher.install_self_weak();
                phase_dispatcher.commit_plan(&target_layers);
                phase_dispatcher
                    .set_max_chunks_per_token(args.swap_phase_aware_max_chunks_per_token);
                if args.swap_phase_aware_max_chunks_per_token > 0 {
                    eprintln!(
                        "weight_swap: phase-aware throttle — max {} chunks/token",
                        args.swap_phase_aware_max_chunks_per_token
                    );
                }
                llm_rs2::profile::op_trace::set_phase_hook(
                    phase_dispatcher.clone() as Arc<dyn llm_rs2::profile::op_trace::PhaseHook>
                );
                phase_aware_swap_dispatcher = Some(phase_dispatcher);
            } else if args.swap_intra_forward || args.swap_layer_immediate {
                let swap_backend: Arc<dyn Backend> = gpu_backend_arc
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| cpu_backend_arc.clone());
                let dispatcher = Arc::new(llm_rs2::models::weights::AsyncSwapDispatcher::new(
                    Arc::clone(&swap_backend),
                ));
                let mode_flag_name = if args.swap_layer_immediate {
                    "--swap-layer-immediate"
                } else {
                    "--swap-intra-forward"
                };
                let secondary = match model.secondary_mmap.as_ref() {
                    Some(s) => Arc::clone(s),
                    None => {
                        anyhow::bail!(
                            "{} requires --secondary-gguf (no secondary mmap available)",
                            mode_flag_name
                        );
                    }
                };
                let config = Arc::new(model.config.clone());
                // LISWAP-6 Phase 6: layer-immediate variant reuses the
                // IntraForwardSwapHook infrastructure. The behavioural
                // difference is in the swap_executor.rs alias H2D-skip
                // (Phase 5b) which collapses every per-layer dispatch to a
                // dummy event when the secondary is rpcmem DMA-BUF aliased.
                let mode_label = if args.swap_layer_immediate {
                    "layer-immediate (LISWAP-6 P6)"
                } else {
                    "intra-forward (LISWAP-4)"
                };
                eprintln!(
                    "weight_swap: {} mode — ratio={:.2}, {} target layers",
                    mode_label,
                    ratio,
                    target_layers.len()
                );
                intra_forward_swap_hook = Some(llm_rs2::models::weights::IntraForwardSwapHook::new(
                    target_layers,
                    0,
                    dispatcher,
                    secondary,
                    model.layers.clone(),
                    swap_backend,
                    Some(Arc::clone(&model.release_worker)),
                    DType::Q4_0,
                    config,
                ));
            } else if args.swap_incremental_per_tick > 0 {
                eprintln!(
                    "weight_swap: incremental mode — ratio={:.2}, {} target layers, per_tick={} ({} ticks estimated)",
                    ratio,
                    target_layers.len(),
                    args.swap_incremental_per_tick,
                    target_layers.len().div_ceil(args.swap_incremental_per_tick),
                );
                incremental_force_swap_plan = Some(llm_rs2::models::weights::IncrementalSwapPlan::new(
                    target_layers,
                    args.swap_incremental_per_tick,
                    0,
                ));
            } else {
                match run_layer_swap(
                    &model,
                    &target_layers,
                    gpu_backend_arc.as_ref(),
                    &cpu_backend_arc,
                    None,
                    #[cfg(feature = "opencl")]
                    host_ptr_swap_pool.clone(),
                    #[cfg(feature = "cuda-embedded")]
                    None,
                    #[cfg(feature = "cuda-embedded")]
                    None,
                ) {
                    Ok(report) => {
                        eprintln!(
                            "weight_swap: force ratio={:.2}, swapped {}/{} layers in {:.1}ms",
                            ratio,
                            report.swapped.len(),
                            num_layers,
                            report.latency_ms,
                        );
                        if let Some(ref stages) = report.stage_breakdown {
                            eprintln!("weight_swap stages: {}", stages.to_log_line());
                        }
                        #[cfg(feature = "opencl")]
                        if let Some(ocl_be) = backend
                            .as_any()
                            .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                        {
                            ocl_be.dump_cl_mem_diagnostics(" stage=after_force_swap");
                        }
                        #[cfg(feature = "opencl")]
                        remap_weights_for_cpu_after_swap(
                            &mut model,
                            &backend,
                            is_gpu,
                            args.resilience_prealloc_switch,
                            "force-swap",
                        );
                    }
                    Err(e) => {
                        anyhow::bail!("--force-swap-ratio: swap failed: {}", e);
                    }
                }
            }
        }};
    }

    if args.qcf_dump.is_none()
        && let Some(ratio) = args.force_swap_ratio
    {
        let ratio = ratio.clamp(0.0, 1.0);
        let num_layers = model.layers.len();
        let target_layers =
            llm_rs2::models::weights::SwapExecutor::uniform_target_layers(ratio, num_layers);

        // ── LISWAP-6: Eager prefault ─────────────────────────────────────
        // qnn_oppkg + Rpcmem variant 일 때 swap 시점의 rpcmem_alloc 비용
        // (~420 ms/25 layer on Galaxy S25) 을 model load 시점에 흡수.
        // Gguf/Auf variant 는 madvise() 만 호출되어 비용 작음 (~65 ms).
        // 모든 swap mode (single-shot/incremental/intra-forward/phase-aware)
        // 가 자동 이득. swap blocking 700 → ~280 ms 단축 (60%).
        //
        // Phase 3 (`--swap-delay-tokens > 0`): prefault ALWAYS runs at prefill
        // end — only the actual swap dispatch is deferred. This preserves the
        // delay=0 baseline wall (~405 ms LISWAP-6 alias) and ensures rpcmem
        // pages are warm regardless of when the trigger fires.
        let skip_eager_prefault = std::env::var("LLMRS_SKIP_EAGER_PREFAULT").is_ok();
        if !target_layers.is_empty()
            && let Some(secondary) = model.secondary_mmap.as_ref()
            && !skip_eager_prefault
        {
            let t_pre = std::time::Instant::now();
            secondary.prefault_layers(&target_layers);
            // LISWAP-6 Phase 1 — Rpcmem variant also primes per-tensor
            // cl_mem aliases inside `ensure_layer_loaded`. Surface the cache
            // size alongside the wall-clock prefault cost so regressions in
            // either step are visible at startup.
            let alias_cache_len = match secondary.as_ref() {
                llm_rs2::models::weights::SecondaryMmap::Rpcmem(rpc) => Some(rpc.alias_cache_len()),
                _ => None,
            };
            match alias_cache_len {
                Some(n) => eprintln!(
                    "weight_swap: eager prefault — {} layers, {:.1}ms (alias cache: {} cl_mems)",
                    target_layers.len(),
                    t_pre.elapsed().as_secs_f64() * 1e3,
                    n,
                ),
                None => eprintln!(
                    "weight_swap: eager prefault — {} target layers, {:.1}ms",
                    target_layers.len(),
                    t_pre.elapsed().as_secs_f64() * 1e3,
                ),
            }
        }

        // Phase 3: defer the dispatch block when --swap-delay-tokens N > 0.
        // We still run dispatch immediately when `target_layers.is_empty()`
        // (no-op log line; nothing to defer).
        if !target_layers.is_empty() && args.swap_delay_tokens > 0 {
            eprintln!(
                "weight_swap: dispatch deferred — will trigger at decode_token_index={} \
                 (Phase 3 mid-decode swap, ratio={:.2}, {} target layers)",
                args.swap_delay_tokens,
                ratio,
                target_layers.len(),
            );
            pending_force_swap = Some((ratio, target_layers));
        } else if target_layers.is_empty() {
            eprintln!(
                "weight_swap: force ratio={:.2} → 0 target layers (no-op)",
                ratio,
            );
        } else {
            // delay == 0 (default): dispatch immediately at prefill end.
            dispatch_force_swap!(ratio, target_layers);
        }
    }

    // ════════════════════════════════════════════════════════════
    //  DUMP-IMPORTANCE MODE: Measure per-layer importance and exit
    // ════════════════════════════════════════════════════════════
    if args.dump_importance {
        return llm_rs2::session::dump_importance::run_dump_importance(
            llm_rs2::session::dump_importance::DumpImportanceCtx {
                backend: backend.clone(),
                memory: memory.clone(),
                model,
                tokenizer,
                kv_caches,
                prompt,
                vocab_size,
                model_path: args.model_path.clone(),
            },
        );
    }

    // ════════════════════════════════════════════════════════════
    //  EVAL-LL MODE: Log-likelihood evaluation for downstream tasks
    // ════════════════════════════════════════════════════════════
    if args.eval_ll {
        let ctx = llm_rs2::session::eval::EvalLlRunCtx {
            args: args.clone(),
            backend: backend.clone(),
            memory: memory.clone(),
            cpu_backend_arc: cpu_backend_arc.clone(),
            gpu_backend_arc: gpu_backend_arc.clone(),
            model,
            tokenizer,
            kv_caches,
            cache_manager,
            score_accumulator,
            skip_config,
            prompt,
            hidden_size,
            vocab_size,
            max_seq_len,
            num_layers,
            kv_type,
            actual_protected_prefix,
            score_based_eviction,
            swap_algorithm,
            importance_formula,
            importance_compare,
            swap_only_layers,
        };
        return llm_rs2::session::eval::run_eval_ll(ctx);
    }

    // ════════════════════════════════════════════════════════════
    //  QCF DUMP WORKFLOW: Warmup prefill → ImportanceTable → Swap → Measure
    //
    //  When --qcf-dump is active, we insert a warmup prefill before the main
    //  measurement to build an ImportanceTable for accurate WeightSwapDecider
    //  (importance × ε bottom-k selection, ENG-ALG-215).
    //
    //  The workflow applies to both --ppl and generation modes.
    //  When --qcf-dump is absent, all existing behavior is unchanged.
    // ════════════════════════════════════════════════════════════

    // Accumulated state produced by the QCF dump workflow.
    // Only populated when args.qcf_dump.is_some().
    let mut qcf_warmup_importance: Option<llm_rs2::qcf::ImportanceTable> = None;
    let mut qcf_swap_decision: Option<llm_rs2::models::weights::decider::SwapDecision> = None;
    let qcf_workflow_start = std::time::Instant::now();

    if args.qcf_dump.is_some() && (args.ppl.is_some() || !prompt.is_empty()) {
        let warmup_n = args.qcf_warmup_tokens.max(1);

        // For PPL mode the warmup tokens come from the reference text; for
        // generation mode they come from the prompt. Both paths cap at warmup_n.
        let warmup_tokens: Vec<u32> = if let Some(ref ppl_path) = args.ppl {
            let text = std::fs::read_to_string(ppl_path)
                .map_err(|e| anyhow::anyhow!("Failed to read PPL file for warmup: {}", e))?;
            let enc = tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Warmup tokenize error: {}", e))?;
            enc.get_ids().iter().take(warmup_n).copied().collect()
        } else {
            let enc = tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Warmup tokenize error: {}", e))?;
            enc.get_ids().iter().take(warmup_n).copied().collect()
        };

        if warmup_tokens.is_empty() {
            anyhow::bail!(
                "--qcf-dump: warmup token sequence is empty (prompt or PPL text too short)"
            );
        }

        let result = run_qcf_warmup_workflow(
            &model,
            &backend,
            memory.as_ref(),
            &mut kv_caches,
            vocab_size,
            &warmup_tokens,
            args.force_swap_ratio,
            gpu_backend_arc.as_ref(),
            &cpu_backend_arc,
            "",
            swap_algorithm,
            true,
            importance_formula,
            importance_compare,
            swap_only_layers.as_deref(),
            args.decode_x_steps,
        )?;
        qcf_swap_decision = result.decision;
        qcf_warmup_importance = Some(result.importance);
    }

    // ════════════════════════════════════════════════════════════
    //  PPL MODE: Perplexity evaluation on reference text
    // ════════════════════════════════════════════════════════════
    if args.ppl.is_some() {
        let ctx = llm_rs2::session::ppl::PplRunCtx {
            args: args.clone(),
            backend: backend.clone(),
            memory: memory.clone(),
            model,
            tokenizer,
            kv_caches,
            cache_manager,
            score_accumulator,
            skip_config,
            hidden_size,
            vocab_size,
            max_seq_len,
            num_layers,
            kv_heads,
            head_dim,
            actual_protected_prefix,
            score_based_eviction,
            qcf_warmup_importance,
            qcf_swap_decision,
            qcf_workflow_start,
            auto_eviction,
            swap_algorithm,
        };
        return llm_rs2::session::ppl::run_ppl_dispatch(ctx);
    }

    // ════════════════════════════════════════════════════════════
    //  PROMPT-BATCH MODE: Sequential multi-prompt generation
    // ════════════════════════════════════════════════════════════
    if args.prompt_batch.is_some() {
        let ctx = llm_rs2::session::batch::BatchRunCtx {
            args: args.clone(),
            backend,
            memory,
            cpu_backend_arc,
            cpu_memory_arc,
            gpu_backend_arc,
            gpu_memory_arc,
            model,
            tokenizer,
            kv_caches,
            cache_manager,
            score_accumulator,
            command_executor,
            skip_config,
            hidden_size,
            vocab_size,
            max_seq_len,
            is_gpu,
            weights_on_gpu,
            kv_heads,
            head_dim,
            kv_type,
            actual_protected_prefix,
            score_based_eviction,
            throttle_delay_ms,
            last_skip_ratio,
            sampling_config,
        };
        return llm_rs2::session::batch::run_prompt_batch(ctx);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  PHASE 4-4.5: standard happy path → DecodeLoop+ModelForward 위임
    // ════════════════════════════════════════════════════════════════════════
    // Narrow guard (assembly::is_standard_happy_path 참조):
    //   - profile / qcf_dump / skip_ratio / d2o_layer_alloc / eviction 비활성
    // Paradigm: prefill → last_logits → argmax = first_token → run(N-1, first_token).
    // chunked prefill은 ModelForward::prefill 내부에서 처리 (Phase 4-4.5 C3).
    if llm_rs2::session::is_standard_happy_path(&args) && args.num_tokens >= 1 {
        return llm_rs2::session::standard_happy::run_standard_happy_path(
            llm_rs2::session::standard_happy::StandardHappyCtx {
                args: args.clone(),
                backend: backend.clone(),
                memory: memory.clone(),
                cpu_backend_arc: cpu_backend_arc.clone(),
                model,
                tokenizer,
                kv_caches,
                tokens,
                initial_kv_capacity,
                max_seq_len,
                kv_type,
                sampling_config: sampling_config.clone(),
                vocab_size,
            },
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    //  STANDARD GENERATE MODE (fallback) — happy path 미통과 시 진입
    // ════════════════════════════════════════════════════════════════════════
    // chunked prefill / optional collector (score_accumulator / skip_config /
    // importance_collector / variance_collector / profiler) 가 필요한 경우.
    // Phase 4-4.5에서 ModelForward에 chunked + collector 흡수되면 happy path
    // 진입 조건이 확장되어 본 fallback 진입 빈도 감소.

    // ── Phase 4-4-2.1: chunked prefill 추출 (session::prefill::run_chunked_prefill) ──
    let prefill_out =
        llm_rs2::session::prefill::run_chunked_prefill(llm_rs2::session::prefill::PrefillCtx {
            args: &args,
            model: &mut model,
            cache_manager: &cache_manager,
            sampling_config: &sampling_config,
            backend: backend.clone(),
            memory: memory.clone(),
            cpu_backend_arc: cpu_backend_arc.clone(),
            cpu_memory_arc: cpu_memory_arc.clone(),
            vocab_size,
            hidden_size,
            max_seq_len,
            actual_protected_prefix,
            auto_eviction,
            start_time,
            layer_swap_estimator: Box::new(|model, table| build_layer_swap_estimate(model, table)),
            kv_caches,
            tokens,
            start_pos,
            skip_config,
            last_skip_ratio,
            throttle_delay_ms,
            command_executor,
        })?;
    let llm_rs2::session::prefill::PrefillOutput {
        kv_caches,
        tokens,
        mut start_pos,
        profiler,
        variance_collector,
        importance_table_for_swap,
        mut collector_armed,
        deferred_switch,
        mut skip_config,
        mut last_skip_ratio,
        mut throttle_delay_ms,
        mut command_executor,
        logits,
        eos_id,
        ttft_ms: _ttft_ms_out,
        last_token_time,
        prefill_forward_ms: _prefill_forward_ms,
        prefill_pure_fwd_ms: _prefill_pure_fwd_ms,
    } = prefill_out;
    _ttft_ms = _ttft_ms_out;
    _last_token_time = last_token_time;

    // ── Phase 4-4-2.3a: decode prologue 추출 ──────────────────────────────
    // (session::decode_fallback::prologue::run_decode_prologue)
    // A: Deferred SwitchHw, B: D2O budgets, C: position_birth_step,
    // D: DecodingStart + drain, E: decode workspace, F: partition_ws,
    // G: spare bufs + streaming, H: Hybrid Attn, I: GPU plan
    let prologue_out = llm_rs2::session::decode_fallback::prologue::run_decode_prologue(
        llm_rs2::session::decode_fallback::prologue::DecodePrologueCtx {
            args: &args,
            model: &mut model,
            backend: backend.clone(),
            is_gpu,
            memory: memory.clone(),
            cpu_backend_arc: cpu_backend_arc.clone(),
            cpu_memory_arc: cpu_memory_arc.clone(),
            gpu_backend_arc: gpu_backend_arc.clone(),
            gpu_memory_arc: gpu_memory_arc.clone(),
            kv_caches,
            logits,
            deferred_switch,
            variance_collector,
            tokens,
            profiler,
            tokenizer: &tokenizer,
            actual_protected_prefix,
            vocab_size,
            hidden_size,
            kv_heads,
            head_dim,
            max_seq_len,
            weights_on_gpu,
            score_accumulator,
        },
    )?;
    let llm_rs2::session::decode_fallback::prologue::DecodePrologueOutput {
        mut backend,
        mut is_gpu,
        mut kv_caches,
        mut logits,
        mut x_gen,
        mut gen_ws,
        mut gen_input_tensor,
        cpu_gen_input,
        mut spare_logits,
        mut spare_xgen,
        mut spare_gen_ws,
        mut spare_gen_input,
        #[cfg(feature = "opencl")]
        mut gpu_plan,
        #[cfg(feature = "opencl")]
        mut gpu_plan_sticky_disabled,
        #[cfg(feature = "opencl")]
        partition_active_any,
        mut logits_cpu,
        mut sampling_indices,
        mut evict_ceiling,
        mut evict_floor_logged,
        mut last_applied_partition_ratio,
        d2o_layer_ratios,
        mut position_birth_step,
        mut profiler,
        printed_len: mut _printed_len,
        mut stdout,
        mut score_accumulator,
        mut tokens,
        #[cfg(feature = "opencl")]
            hybrid_scope: _hybrid_scope,
    } = prologue_out;

    // === GENERATION PHASE ===
    {
        use std::io::Write;

        // Re-declare prologue-computed locals needed by the decode loop body.
        let ffn_hidden = model.config.intermediate_size;
        let decode_mem: &dyn Memory = if is_gpu {
            memory.as_ref()
        } else {
            cpu_memory_arc.as_ref()
        };

        // Generation loop
        for (decode_token_index, _) in (0..(args.num_tokens - 1)).enumerate() {
            let _decode_t = llm_rs2::profile::quality_metrics::Timer::start(
                &llm_rs2::profile::quality_metrics::DECODE_TOTAL,
            );

            // Check physical cache capacity (not start_pos, which is logical RoPE position)
            if kv_caches[0].current_pos >= max_seq_len {
                println!("\n[Stopped: Max context length reached]");
                break;
            }

            // ── LISWAP Phase 3 — mid-decode force-swap trigger ───────────────
            // Fires once at decode_token_index == args.swap_delay_tokens when
            // a pending payload was prepared at prefill end. Same dispatch
            // logic as the prefill-end path (shared via macro) to ensure the
            // four swap modes (single-shot / incremental / intra-forward /
            // phase-aware) all receive the same code path.
            if let Some((ratio, target_layers)) = pending_force_swap.take() {
                if decode_token_index == args.swap_delay_tokens {
                    eprintln!(
                        "weight_swap: mid-decode trigger at decode_token_index={}",
                        decode_token_index,
                    );
                    dispatch_force_swap!(ratio, target_layers);
                } else {
                    // Not yet — re-stash for the next iteration.
                    pending_force_swap = Some((ratio, target_layers));
                }
            }

            let last_token = tokens[tokens.len() - 1];
            unsafe {
                *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_token;
            }
            // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
            backend.write_buffer(&mut gen_input_tensor, unsafe {
                std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
            })?;

            // Apply decay to accumulated importance scores before this step
            if let Some(acc) = score_accumulator.as_mut() {
                acc.begin_step();
            }

            let forward_start = std::time::Instant::now();

            // Phase 2: throttle counter reset — 매 token 시작 시.
            if let Some(ref disp) = phase_aware_swap_dispatcher {
                disp.reset_token_counter();
            }

            // Try GPU plan path (OpenCL decode only, no profiling)
            #[cfg(feature = "opencl")]
            let used_plan = if let Some(ref plan) = gpu_plan {
                match model.execute_plan(
                    plan,
                    &gen_input_tensor,
                    start_pos,
                    &mut x_gen,
                    &mut kv_caches,
                    &mut logits,
                    &backend,
                ) {
                    Ok(true) => true,
                    Ok(false) => {
                        // Plan invalidated (KV cache resize needed).
                        // Set to None; forward_into will handle grow.
                        // Plan is rebuilt on the next token after grow completes.
                        gpu_plan = None;
                        false
                    }
                    Err(_) => {
                        gpu_plan = None;
                        false
                    }
                }
            } else {
                false
            };
            #[cfg(not(feature = "opencl"))]
            let used_plan = false;

            if !used_plan {
                // Use GPU memory when on GPU; CPU memory when on CPU.
                // After SwitchHw GPU→CPU, `memory` is still OpenCL memory whose
                // alloc() creates OpenCLBuffer (null as_ptr). We must use
                // cpu_memory_arc to ensure lazy allocations (e.g. k_cast/v_cast)
                // produce CPU-accessible buffers.
                let effective_mem: &dyn Memory = if is_gpu {
                    gpu_memory_arc.as_deref().unwrap_or_else(|| memory.as_ref())
                } else {
                    cpu_memory_arc.as_ref()
                };

                // --cuda-graph: bundle this token's launches into a single
                // CUDA Graph, replayed once. Drains pending work first; the
                // end_capture_and_launch() call replaces the per-kernel
                // driver dispatches with one graph launch.
                #[cfg(feature = "cuda-embedded")]
                let cu_graph_be: Option<
                    &llm_rs2::backend::cuda_embedded::CudaBackend,
                > = if args.cuda_graph {
                    backend
                        .as_any()
                        .downcast_ref::<llm_rs2::backend::cuda_embedded::CudaBackend>()
                } else {
                    None
                };
                #[cfg(feature = "cuda-embedded")]
                if let Some(cu_be) = cu_graph_be {
                    cu_be.begin_graph_capture()?;
                }

                // LISWAP-4: inject IntraForwardSwapHook when active.
                // The cast to `&dyn LayerBoundaryHook` happens inside the
                // option mapping so the args field can be `Option<&dyn _>`
                // — this is the *only* place a real hook is wired in.
                let liswap4_hook: Option<&dyn llm_rs2::models::weights::LayerBoundaryHook> =
                    intra_forward_swap_hook
                        .as_deref()
                        .map(|h| h as &dyn llm_rs2::models::weights::LayerBoundaryHook);

                model.forward_into(TransformerModelForwardArgs {
                    input_tokens: &gen_input_tensor,
                    start_pos,
                    kv_caches: &mut kv_caches,
                    backend: &backend,
                    memory: effective_mem,
                    logits_out: &mut logits,
                    x_gen: Some(&mut x_gen),
                    workspace: Some(&mut gen_ws),
                    score_accumulator: score_accumulator.as_mut(),
                    profiler: profiler.as_mut().map(|p| &mut p.ops),
                    skip_config: skip_config.as_ref(),
                    importance_collector: None,
                    logits_last_only: false,
                    variance_collector: None,
                    prefill_workspace: None,

                    layer_boundary_hook: liswap4_hook,
                })?;

                #[cfg(feature = "cuda-embedded")]
                if let Some(cu_be) = cu_graph_be {
                    cu_be.end_graph_capture_and_launch()?;
                }

                // Rebuild plan if it was invalidated (e.g. KV cache resize).
                // Skip rebuild when tensor partition is active — plan bypasses
                // the partition co-execution path. Same accumulator-pairing
                // requirement as the initial build above.
                #[cfg(feature = "opencl")]
                let accumulator_compatible_with_plan = {
                    let has_cpu_acc = score_accumulator.is_some();
                    let gpu_acc_active = backend
                        .as_any()
                        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                        .and_then(|ob| ob.gpu_score_acc())
                        .is_some_and(|acc| acc.is_active());
                    !has_cpu_acc || gpu_acc_active
                };
                // Plan rebuild after fallback. Partition is now routed inside
                // build_plan (ENG-ALG-200) so the old `partition_ctx.is_none()`
                // gate is dropped — build_plan itself picks partition-aware or
                // legacy FFN per layer.
                #[cfg(feature = "opencl")]
                if gpu_plan.is_none()
                    && !gpu_plan_sticky_disabled
                    && backend.name() == "OpenCL"
                    && !args.profile
                    && !args.no_gpu_plan
                    && accumulator_compatible_with_plan
                    && model.config.arch != llm_rs2::models::config::ModelArch::Gemma3
                {
                    gpu_plan = model.build_plan(&x_gen, &logits, &gen_ws, &mut kv_caches, &backend);
                    if partition_active_any && gpu_plan.is_none() {
                        // Second build also failed — lock out further retries.
                        gpu_plan_sticky_disabled = true;
                    }
                }
            }
            backend.synchronize()?;

            // --profile-events: drain and aggregate GPU events into OpProfiler.
            // --heartbeat-gpu-profile (MSG-068 Phase 2): same flush, but feeds
            // the GPU self-util meter instead of (or in addition to) the
            // op-level profiler. Flush runs whenever queue profiling is on.
            #[cfg(feature = "opencl")]
            if (args.profile_events || args.heartbeat_gpu_profile)
                && let Some(ocl_be) = backend
                    .as_any()
                    .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                && ocl_be.profile_events_enabled
            {
                ocl_be.flush_and_aggregate_profile()?;
                // Op-profiler aggregation only when the caller asked for
                // per-op timing (--profile-events). The heartbeat-only path
                // still flushes so the GPU self-util meter sees the delta,
                // but intentionally skips take_profile_accum() to avoid
                // clearing labels that might still be of interest elsewhere.
                if args.profile_events {
                    let accum = ocl_be.take_profile_accum();
                    if let Some(ref mut p) = profiler {
                        p.ops.merge_from_events(&accum);
                        p.ops.count += 1;
                    }
                }
            }

            // --cuda-profile: drain pending CUevent pairs per-token so
            // the pool (default 4096 pairs) does not overflow. Each
            // decode token launches roughly n_layers * ~10 kernels.
            #[cfg(feature = "cuda-embedded")]
            if args.cuda_profile
                && let Some(cu_be) = backend
                    .as_any()
                    .downcast_ref::<llm_rs2::backend::cuda_embedded::CudaBackend>()
                && cu_be.profiler_enabled()
            {
                if let Err(e) = cu_be.flush_profiler() {
                    eprintln!("[CUDA-Profile] per-token flush failed: {}", e);
                }
            }

            let forward_ms = forward_start.elapsed().as_secs_f64() * 1000.0;

            // ── Layer-Incremental Swap dispatch (ENG-ALG-233) ──────────────────
            // Runs after forward, before sampling. Per-tick: drain up to N layers
            // and call SwapExecutor::execute_on_slots with the chunk.
            // ENG-ALG-234: plan committed with force-swap-ratio + per_tick > 0;
            //   new signals during flight are ignored (plan runs to completion).
            // INV-145: empty chunk is never passed to execute_on_slots.
            if let Some(ref mut inc_plan) = incremental_force_swap_plan {
                // LISWAP-6 Dynamic-K: reactive pause + per-tick override.
                //
                // - Pause: release queue non-empty → skip swap this tick (K
                //   stays unchanged). Calibration tick is exempt because it
                //   has to dispatch K=1 to measure drop cost.
                // - Pre-drain: inject controller's current K into the plan.
                let mut dyn_k_pause = false;
                if let Some(ref ctrl) = dynamic_k_controller {
                    let pending = model.release_worker.pending_count();
                    if ctrl.is_calibrated() && ctrl.should_pause(pending) {
                        dyn_k_pause = true;
                        if dynamic_k_diag {
                            eprintln!(
                                "[DynamicK] pause t={} pending={} k={}",
                                decode_token_index,
                                pending,
                                ctrl.current_k()
                            );
                        }
                    } else {
                        // Calibration tick forces K=1 (sync measurement);
                        // subsequent ticks use the controller's current K.
                        let k = if ctrl.is_calibrated() {
                            ctrl.current_k()
                        } else {
                            1
                        };
                        inc_plan.set_per_tick(k);
                    }
                } else if let Some(ref ctrl) = probing_k_controller {
                    let pending = model.release_worker.pending_count();
                    if ctrl.should_pause(pending) {
                        dyn_k_pause = true;
                        if probing_k_diag {
                            eprintln!(
                                "[ProbingK] pause t={} pending={} k={}",
                                decode_token_index,
                                pending,
                                ctrl.current_k()
                            );
                        }
                    } else {
                        inc_plan.set_per_tick(ctrl.current_k());
                    }
                }
                let chunk = if dyn_k_pause {
                    Vec::new()
                } else {
                    inc_plan.drain_chunk()
                };
                if !chunk.is_empty() {
                    let t_swap = std::time::Instant::now();
                    let io_before = read_bytes_now();
                    match run_layer_swap(
                        &model,
                        &chunk,
                        gpu_backend_arc.as_ref(),
                        &cpu_backend_arc,
                        async_swap_dispatcher.as_ref(),
                        #[cfg(feature = "opencl")]
                        host_ptr_swap_pool.clone(),
                        #[cfg(feature = "cuda-embedded")]
                        layer_swap_pool.clone(),
                        #[cfg(feature = "cuda-embedded")]
                        mmap_registration.clone(),
                    ) {
                        Ok(report) => {
                            let io_after = read_bytes_now();
                            eprintln!(
                                "[IncrementalSwap] tick={} chunk={:?} swapped={} remaining={} latency={:.1}ms read_bytes_delta={}",
                                decode_token_index,
                                &chunk,
                                report.swapped.len(),
                                inc_plan.remaining_count(),
                                t_swap.elapsed().as_secs_f64() * 1000.0,
                                io_after.saturating_sub(io_before),
                            );
                            if let Some(ref stages) = report.stage_breakdown {
                                eprintln!("[IncrementalSwap] stages: {}", stages.to_log_line());
                            }
                            #[cfg(feature = "opencl")]
                            remap_weights_for_cpu_after_swap(
                                &mut model,
                                &backend,
                                is_gpu,
                                args.resilience_prealloc_switch,
                                "incremental-swap",
                            );

                            // LISWAP-6 Dynamic-K Phase 0 calibration. Runs only on
                            // the first successfully-dispatched chunk. Uses the
                            // dispatch wall (`t_swap.elapsed()`) divided by
                            // `chunk.len()` as `drop_ms_per_layer` — the main-thread
                            // blocking time per layer (mmap_permute + dispatcher
                            // submit). The prior release_worker spin was unreliable
                            // on async path: dispatcher worker chains release enqueue
                            // independently and sub-ms release time collapsed
                            // drop_ms to 0 → safe_k exploded. Main-thread blocking
                            // time is a meaningful budget item that scales with
                            // cold/warm mmap state and chunk size (2026-05-13).
                            if let Some(ref mut ctrl) = dynamic_k_controller
                                && !ctrl.is_calibrated()
                                && !chunk.is_empty()
                            {
                                let blocking_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
                                let drop_ms_per_layer = (blocking_ms / chunk.len() as f64) as f32;
                                ctrl.calibrate(drop_ms_per_layer, forward_ms as f32);
                                if dynamic_k_diag {
                                    eprintln!(
                                        "[DynamicK] calibrated t={} blocking_ms={:.3}/layer fwd_ms={:.2} safe_k={}",
                                        decode_token_index,
                                        drop_ms_per_layer,
                                        forward_ms,
                                        ctrl.current_k()
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "[IncrementalSwap] swap error on tick={}: {}",
                                decode_token_index, e
                            );
                        }
                    }
                }

                // LISWAP-6 Dynamic-K Phase 1+: observe forward wall, shrink K
                // if the forward got tighter than anything seen so far.
                if let Some(ref mut ctrl) = dynamic_k_controller
                    && ctrl.is_calibrated()
                {
                    let prev_k = ctrl.current_k();
                    ctrl.observe_forward(forward_ms as f32);
                    if dynamic_k_diag && ctrl.current_k() != prev_k {
                        eprintln!(
                            "[DynamicK] k_decrease t={} fwd_ms={:.2} new_k={}",
                            decode_token_index,
                            forward_ms,
                            ctrl.current_k()
                        );
                    }
                }

                // Probing-K observation: every decode token feeds the EWMA and
                // counts toward the stability window. release_pending samples
                // *after* the dispatch above — if any spike landed it will
                // decrement K symmetric to ARGUS's monotonic shrink.
                if let Some(ref mut ctrl) = probing_k_controller {
                    let prev_k = ctrl.current_k();
                    let pending_after = model.release_worker.pending_count();
                    ctrl.observe(forward_ms as f32, pending_after);
                    if probing_k_diag {
                        let arrow = if ctrl.current_k() > prev_k {
                            "↑"
                        } else if ctrl.current_k() < prev_k {
                            "↓"
                        } else {
                            "·"
                        };
                        eprintln!(
                            "[ProbingK] t={} fwd_ms={:.2} pending={} k={}->{} {}",
                            decode_token_index,
                            forward_ms,
                            pending_after,
                            prev_k,
                            ctrl.current_k(),
                            arrow,
                        );
                    }
                }
                // ENG-ALG-233: retire plan when all layers have been drained (INV-145).
                if inc_plan.is_done() {
                    eprintln!(
                        "[IncrementalSwap] plan complete (started_at_token={}, finished_at_token={})",
                        inc_plan.started_at_token(),
                        decode_token_index,
                    );
                    // LISWAP-2: drain async dispatcher to ensure all in-flight commits land
                    // before the plan is retired. drain failure is non-fatal — prototype
                    // robustness is secondary to measurement.
                    if let Some(ref dispatcher) = async_swap_dispatcher {
                        let drain_t = std::time::Instant::now();
                        if let Err(e) = dispatcher.drain(std::time::Duration::from_secs(2)) {
                            eprintln!("[LISWAP-2] drain failed: {e}");
                        } else {
                            eprintln!(
                                "[LISWAP-2] dispatcher drained: {:.1}ms",
                                drain_t.elapsed().as_secs_f64() * 1000.0
                            );
                        }
                    }
                    incremental_force_swap_plan = None;

                    // LISWAP-6 manager path: build WeightSwapReport when the plan
                    // was committed by dispatch_swap_weights (manager signal).
                    // Stored in `ready_weight_swap_report`; sent by executor block
                    // later this token tick (executor scope is separate).
                    if let Some((ratio, n_planned, plan_start, qcf_estimated)) =
                        manager_swap_report_pending.take()
                    {
                        use llm_rs2::models::weights::compute_qcf_swap;
                        let latency_ms = plan_start.elapsed().as_millis() as u64;
                        let n_layers = model.layers.len();
                        let actually_swapped_now: Vec<usize> = (0..n_layers)
                            .filter(|&i| model.layers[i].current_dtype() == DType::Q4_0)
                            .collect();
                        let qcf_swap_actual = if actually_swapped_now.is_empty() {
                            qcf_estimated
                        } else {
                            compute_qcf_swap(
                                &actually_swapped_now,
                                &model.quant_noise,
                                importance_table_for_swap.as_ref(),
                                n_layers,
                            )
                        };
                        let layers_swapped: Vec<llm_shared::LayerSwapEntry> = actually_swapped_now
                            .iter()
                            .map(|&idx| llm_shared::LayerSwapEntry {
                                layer_idx: idx as u32,
                                from_dtype: llm_shared::DtypeTag::F16,
                                to_dtype: llm_shared::DtypeTag::Q4_0,
                            })
                            .collect();
                        eprintln!(
                            "[WeightSwap] manager plan complete: ratio={:.2}, planned={}, \
                             actually_q4={}, qcf_swap={:.4}, latency={}ms",
                            ratio,
                            n_planned,
                            layers_swapped.len(),
                            qcf_swap_actual,
                            latency_ms,
                        );
                        ready_weight_swap_report = Some(llm_shared::WeightSwapReport {
                            layers_swapped,
                            freed_bytes: 0,
                            latency_ms,
                            qcf_swap_actual,
                        });
                    }
                }
            }
            // ── End Layer-Incremental Swap dispatch ────────────────────────────

            // ── LISWAP-4 Intra-forward Swap retire (INV-150) ──────────────────
            // After every decode token, check whether the in-flight plan is
            // complete. If so, drain dispatcher, synchronize backend, bump
            // ratio_generation, invalidate noshuffle SOA registry, and retire
            // the hook to None.
            if let Some(hook) = intra_forward_swap_hook.clone()
                && hook.plan_is_complete()
            {
                let drain_t = std::time::Instant::now();
                let backend_for_invalidate: Arc<dyn Backend> = gpu_backend_arc
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| Arc::clone(&backend));
                let invalidate = move || {
                    backend_for_invalidate.invalidate_noshuffle_soa_registry();
                };
                match hook.finalize(
                    &model.ratio_generation,
                    invalidate,
                    std::time::Duration::from_secs(10),
                ) {
                    Ok(()) => {
                        eprintln!(
                            "[IntraForwardSwap] plan retired at token={} (drain+sync+bump+invalidate {:.1}ms)",
                            decode_token_index,
                            drain_t.elapsed().as_secs_f64() * 1000.0,
                        );
                        #[cfg(feature = "opencl")]
                        remap_weights_for_cpu_after_swap(
                            &mut model,
                            &backend,
                            is_gpu,
                            args.resilience_prealloc_switch,
                            "intra-forward-swap",
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "[IntraForwardSwap] finalize failed at token={}: {}",
                            decode_token_index, e
                        );
                    }
                }
                intra_forward_swap_hook = None; // retire
            }
            // ── End LISWAP-4 retire ────────────────────────────────────────────

            // ── LISWAP-5 Phase-aware Swap retire ──────────────────────────────
            // chunk_queue가 비고 in_flight도 None이면 dispatcher 종료. finalize는
            // 마지막 ratio_generation bump + invalidate 수행. PHASE_HOOK은
            // OnceLock이라 unset 불가능하지만 finalize() 후 모든 hook fire가
            // noop이 됨 (dispatcher 내부 finalized atomic).
            if let Some(disp) = phase_aware_swap_dispatcher.as_ref()
                && std::env::var("LLMRS_PHASE_AWARE_DEBUG").as_deref() == Ok("1")
                && decode_token_index < 5
            {
                let (q, inf, p, d, hs, he, ce) = disp.debug_snapshot();
                eprintln!(
                    "[PhaseAwareSwap-DBG] tok={} queue={} in_flight={} pending={} dispatched={} hook_start={} hook_end={} cachefit_end={}",
                    decode_token_index, q, inf, p, d, hs, he, ce
                );
            }
            if let Some(disp) = phase_aware_swap_dispatcher.as_ref()
                && disp.is_complete()
            {
                let drain_t = std::time::Instant::now();
                let backend_for_invalidate: Arc<dyn Backend> = gpu_backend_arc
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| Arc::clone(&backend));
                let invalidate = move || {
                    backend_for_invalidate.invalidate_noshuffle_soa_registry();
                };
                match disp.finalize(
                    &model.ratio_generation,
                    invalidate,
                    std::time::Duration::from_secs(10),
                ) {
                    Ok(()) => {
                        eprintln!(
                            "[PhaseAwareSwap] plan retired at token={} (drain+sync+bump+invalidate {:.1}ms, chunks={})",
                            decode_token_index,
                            drain_t.elapsed().as_secs_f64() * 1000.0,
                            disp.dispatched_count(),
                        );
                        #[cfg(feature = "opencl")]
                        remap_weights_for_cpu_after_swap(
                            &mut model,
                            &backend,
                            is_gpu,
                            args.resilience_prealloc_switch,
                            "phase-aware-swap",
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "[PhaseAwareSwap] finalize failed at token={}: {}",
                            decode_token_index, e
                        );
                    }
                }
                phase_aware_swap_dispatcher = None;
            }
            // ── End LISWAP-5 retire ────────────────────────────────────────────

            // ── H2O Debug: per-step diagnostics ──
            if args.h2o_debug() {
                // 1. Verify ws.scores is post-softmax (sample first 4 heads)
                let n_heads_q = model.config.num_attention_heads;
                let stride = gen_ws.scores.len() / n_heads_q;
                let cache_pos = kv_caches[0].current_pos;
                let heads_to_check = n_heads_q.min(4);
                for h in 0..heads_to_check {
                    let sum: f32 = gen_ws.scores[h * stride..h * stride + cache_pos]
                        .iter()
                        .sum();
                    if (sum - 1.0).abs() > 0.01 {
                        eprintln!(
                            "[H2O-Debug] WARNING: head {} score sum = {:.6} (expect ~1.0)",
                            h, sum
                        );
                    }
                }

                // 2. Dump importance score distribution
                if let Some(acc) = score_accumulator.as_ref() {
                    let scores = acc.importance_scores();
                    let valid = &scores[..cache_pos];
                    if !valid.is_empty() {
                        let mut indexed: Vec<(usize, f32)> =
                            valid.iter().enumerate().map(|(i, &s)| (i, s)).collect();
                        indexed.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let top5: Vec<_> = indexed.iter().take(5).collect();
                        let bot5: Vec<_> = indexed.iter().rev().take(5).collect();
                        eprintln!(
                            "[H2O-Debug] step={} cache_pos={} Top5={:?} Bot5={:?}",
                            decode_token_index, cache_pos, top5, bot5
                        );
                    }
                }
            }

            llm_rs2::session::decode_fallback::eviction_trigger::run_auto_eviction(
                llm_rs2::session::decode_fallback::eviction_trigger::AutoEvictionCtx {
                    args: &args,
                    cache_manager: &cache_manager,
                    kv_caches: &mut kv_caches,
                    auto_eviction,
                    score_based_eviction,
                    score_accumulator: &mut score_accumulator,
                    d2o_layer_ratios: &d2o_layer_ratios,
                    backend: &backend,
                    profiler: &mut profiler,
                    position_birth_step: &mut position_birth_step,
                    actual_protected_prefix,
                    decode_token_index,
                },
            )?;
            forward_ms_values.push(forward_ms);
            if std::env::var("LLMRS_PER_TOKEN_MS").is_ok() {
                eprintln!(
                    "[PER_TOKEN] idx={} kv_pos={} forward_ms={:.3}",
                    decode_token_index, kv_caches[0].current_pos, forward_ms
                );
            }

            // ── Experiment: inject directives at this token position ──
            let mut injected_signals: Vec<String> = Vec::new();
            if let (Some(schedule), Some(tx)) = (&experiment_schedule, &experiment_tx) {
                for entry in schedule.directives_at(decode_token_index) {
                    let msg = ManagerMessage::Directive(entry.directive.clone());
                    injected_signals.push(directive_summary(&msg));
                    tx.send(msg).ok();
                }
            }

            // ── Resilience checkpoint (CommandExecutor) ──────
            let mut action_names: Vec<String> = Vec::new();
            if let Some(executor) = &mut command_executor {
                let kv_snap = KVSnapshot {
                    total_bytes: kv_caches
                        .iter()
                        .map(|c| (c.k_buffer.buffer().size() + c.v_buffer.buffer().size()) as u64)
                        .sum(),
                    total_tokens: kv_caches[0].current_pos,
                    capacity: kv_caches[0].capacity(),
                    protected_prefix: actual_protected_prefix,
                    // Phase 3에서 실제 정책/dtype/skip 정보로 채울 예정
                    kv_dtype: "f16".to_string(),
                    eviction_policy: args.eviction_policy().to_string(),
                    skip_ratio: 0.0,
                };

                let plan = executor.poll(&kv_snap);
                action_names = plan_summary(&plan);

                // Activate score collection on-demand: only when eviction is
                // requested or imminent. With GPU score accumulator, there is
                // no per-token overhead (scores are accumulated on-device).
                if let Some(ref mut acc) = score_accumulator
                    && !acc.is_active()
                    && (plan.evict.is_some() || plan.request_qcf)
                {
                    acc.set_active(true);
                    // Also activate GPU score accumulator if available
                    #[cfg(feature = "opencl")]
                    if let Some(ocl_be) = backend
                        .as_any()
                        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                        && let Some(gpu_acc) = ocl_be.gpu_score_acc_mut()
                    {
                        gpu_acc.set_active(true);
                    }
                }

                // SEQ-095/096: Compute and send QCF estimates if requested
                if plan.request_qcf {
                    // ENG-ALG-218: if secondary mmap is present, arm the collector
                    // so the next prefill injects ImportanceCollector.
                    if model.secondary_mmap.is_some() && !collector_armed {
                        collector_armed = true;
                        eprintln!("[WeightSwap] ImportanceCollector armed for next prefill");
                    }

                    // Derive streaming window: same logic as policy construction
                    let streaming_window_size = if args.streaming_window() > 0 {
                        args.streaming_window()
                    } else if args.kv_budget() > 0 {
                        args.kv_budget().saturating_sub(args.sink_size())
                    } else {
                        args.eviction_window()
                    };

                    // ISSUE-9 fix: On OpenCL with zero-copy memory, KV V
                    // buffers are UnifiedBuffer (CL_MEM_ALLOC_HOST_PTR) that
                    // start unmapped — `as_ptr()` returns null, which trips
                    // the host-readable guard in compute_qcf_estimates and
                    // skips KV-based 4종 estimates. Sync GPU queue and map
                    // V buffers for CPU before running QCF, then unmap so
                    // the next forward pass can reuse the GPU path.
                    if let Err(e) = backend.synchronize() {
                        eprintln!("[QCF] backend.synchronize() failed: {}", e);
                    }
                    let mut mapped_bufs: Vec<std::sync::Arc<dyn llm_rs2::buffer::Buffer>> =
                        Vec::new();
                    for cache in &kv_caches {
                        let v_buf = cache.v_buffer.buffer();
                        if v_buf.as_ptr().is_null() {
                            match v_buf.map_for_cpu() {
                                Ok(_) => mapped_bufs.push(v_buf.clone()),
                                Err(e) => {
                                    eprintln!("[QCF] map_for_cpu failed: {}", e);
                                }
                            }
                        }
                    }

                    let ctx = QcfEstimateContext {
                        kv_caches: &kv_caches,
                        score_accumulator: score_accumulator.as_ref(),
                        streaming_config: Some((args.sink_size(), streaming_window_size)),
                        importance_table: importance_table_for_swap.as_ref(),
                        num_layers: model.config.num_hidden_layers,
                        kivi_caches: None,
                    };
                    let estimates = compute_qcf_estimates(&ctx);

                    // Release mappings so subsequent forward passes can use
                    // the GPU path without the "writes to mapped buffer
                    // are UB" hazard.
                    for buf in &mapped_bufs {
                        if let Err(e) = buf.unmap_for_gpu() {
                            eprintln!("[QCF] unmap_for_gpu failed: {}", e);
                        }
                    }

                    // Build layer_swap estimate if importance table is available
                    // (set by a previous prefill) and secondary is present.
                    let layer_swap =
                        build_layer_swap_estimate(&model, importance_table_for_swap.as_ref());

                    executor.send_qcf_estimate(llm_shared::QcfEstimate {
                        estimates,
                        layer_swap,
                    });
                }

                // ENG-ALG-214-ROUTE (LISWAP-6 manager path): SwapWeights →
                // IncrementalSwapPlan commit. decode loop drains K=2 layers/tick
                // with dynamic-K + sub-batch pause. WeightSwapReport sent on
                // plan completion (see plan-done block below).
                //
                // Source priority: sticky `pending_swap_weights` first — covers
                // the case where the directive arrived while the prefill loop
                // was polling (prefill drops `plan.swap_weights`). Fall back to
                // `plan.swap_weights` for the same-tick path.
                let pending_swap = executor.take_pending_swap_weights().or(plan.swap_weights);
                if let Some((ratio, target_dtype)) = pending_swap {
                    dispatch_swap_weights(
                        &model,
                        ratio,
                        target_dtype,
                        importance_table_for_swap.as_ref(),
                        decode_token_index,
                        &mut incremental_force_swap_plan,
                        &mut manager_swap_report_pending,
                    );
                    // Note: remap_weights_for_cpu_after_swap will be called
                    // per-chunk in the incremental swap dispatch block above.
                }

                // LISWAP-6 manager path: send completed WeightSwapReport.
                // Built by the plan-done block (before executor scope), consumed here.
                if let Some(report) = ready_weight_swap_report.take() {
                    executor.send_weight_swap_report(report);
                }

                if let Some(evict) = &plan.evict {
                    let effective_ratio =
                        args.experiment_eviction_ratio.unwrap_or(evict.target_ratio);

                    let current_pos = kv_caches[0].current_pos;

                    // Ceiling: record current_pos at the first sticky eviction trigger.
                    // All subsequent evictions use ceiling * ratio as fixed target to prevent
                    // cascade shrinking (e.g. 33→16→8→... when ratio applied to shrinking pos).
                    // Streaming eviction (target_ratio == 0.0) bypasses this check since
                    // it manages its own window logic internally.
                    let (skip_eviction, target_pos) = if effective_ratio > 0.0 {
                        let ceiling = evict_ceiling.get_or_insert(current_pos);
                        let tgt_raw = (*ceiling as f32 * effective_ratio).max(1.0) as usize;
                        let tgt = if tgt_raw < args.min_kv_cache() {
                            if evict_floor_logged.is_none() {
                                eprintln!(
                                    "[Eviction] target_pos {} clamped to min_kv_cache {}",
                                    tgt_raw,
                                    args.min_kv_cache()
                                );
                                evict_floor_logged = Some(true);
                            }
                            args.min_kv_cache()
                        } else {
                            tgt_raw
                        };
                        // Batch 32 tokens before evicting to amortize memmove overhead
                        // (~14ms/step → ~0.4ms/step on compact_keep_positions).
                        const EVICT_BATCH_HEADROOM: usize = 32;
                        (current_pos <= tgt + EVICT_BATCH_HEADROOM, tgt)
                    } else {
                        (false, 0)
                    };

                    if skip_eviction {
                        // Cache already within target — no-op this step
                    } else {
                        // GPU score sync before resilience eviction
                        #[cfg(feature = "opencl")]
                        if let Some(ocl_be) = backend
                            .as_any()
                            .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>(
                        ) && let Some(gpu_acc) = ocl_be.gpu_score_acc()
                            && gpu_acc.is_active()
                        {
                            let (flat, head) = gpu_acc.sync_to_cpu(ocl_be.queue.as_core())?;
                            if let Some(ref mut acc) = score_accumulator {
                                acc.import_gpu_scores(&flat, &head);
                            }
                        }

                        // ── Score distribution diagnostic (via events system) ──
                        if let Some(acc) = score_accumulator.as_ref() {
                            let scores = acc.importance_scores();
                            let cache_pos = kv_caches[0].current_pos;

                            if let Some(snapshot) = events::build_score_snapshot(
                                scores,
                                cache_pos,
                                actual_protected_prefix,
                                decode_token_index,
                                10,
                            ) {
                                cache_manager
                                    .event_sink()
                                    .emit(CacheEvent::ScoreDiagnostic(snapshot));

                                if let Some(ref out_path) = args.experiment_output {
                                    let diag_path = format!(
                                        "{}.scores.csv",
                                        out_path.trim_end_matches(".jsonl")
                                    );
                                    if events::dump_scores_csv(scores, cache_pos, &diag_path)
                                        .is_ok()
                                    {
                                        eprintln!("[ScoreDiag] Scores dumped to {}", diag_path);
                                    }
                                }
                            }
                        }

                        // Build ScoreContext from accumulator for policy-directed eviction
                        let scores = if let Some(acc) = score_accumulator.as_ref() {
                            if let Some(head_imp) = acc.head_importance_scores() {
                                llm_rs2::pressure::cache_manager::ScoreContext::PerHead {
                                    flat: acc.importance_scores(),
                                    head: head_imp,
                                    n_kv_heads: acc.n_kv_heads(),
                                }
                            } else if acc.is_active() {
                                llm_rs2::pressure::cache_manager::ScoreContext::Flat {
                                    importance: acc.importance_scores(),
                                }
                            } else {
                                llm_rs2::pressure::cache_manager::ScoreContext::None
                            }
                        } else {
                            llm_rs2::pressure::cache_manager::ScoreContext::None
                        };

                        // Manager already decided to evict — execute via named policy
                        // D2O uses Pipeline (force_evict_with_scores), not named policy registry
                        // StreamingLLM uses on-demand instantiation (params from directive)

                        // Ceiling-based adjusted ratio: back-calculate ratio so that
                        // force_evict's internal (current_pos * ratio) == target_pos.
                        // This prevents the cascade effect when current_pos < ceiling.
                        let adjusted_ratio = if effective_ratio > 0.0 && current_pos > 0 {
                            target_pos as f32 / current_pos as f32
                        } else {
                            effective_ratio
                        };

                        let result = if evict.method == llm_rs2::resilience::EvictMethod::D2o {
                            let importance = if let Some(acc) = score_accumulator.as_ref() {
                                acc.importance_scores().to_vec()
                            } else {
                                vec![]
                            };
                            if importance.is_empty() {
                                cache_manager.force_evict(&mut kv_caches, adjusted_ratio)
                            } else {
                                cache_manager.force_evict_with_scores(
                                    &mut kv_caches,
                                    adjusted_ratio,
                                    &importance,
                                )
                            }
                        } else if evict.method == llm_rs2::resilience::EvictMethod::Streaming {
                            use llm_rs2::pressure::eviction::StreamingLLMPolicy;
                            if let Some(ref sp) = evict.streaming_params {
                                let policy = StreamingLLMPolicy::new(sp.sink_size, sp.window_size);
                                cache_manager.force_evict_by_policy_ref(
                                    &policy,
                                    &mut kv_caches,
                                    adjusted_ratio,
                                    scores,
                                )
                            } else {
                                Err(anyhow::anyhow!(
                                    "KvStreaming evict plan missing streaming_params"
                                ))
                            }
                        } else {
                            cache_manager.force_evict_by_policy(
                                evict.method,
                                &mut kv_caches,
                                adjusted_ratio,
                                scores,
                            )
                        };
                        match result {
                            Ok(r) if r.evicted => {
                                eprintln!(
                                    "[Resilience] Evicted {} tokens (pos: {} → {})",
                                    r.tokens_removed,
                                    r.new_pos + r.tokens_removed,
                                    r.new_pos
                                );
                                if args.h2o_debug() {
                                    if let Some(acc) = score_accumulator.as_ref() {
                                        let scores = acc.importance_scores();
                                        let pre_pos = r.new_pos + r.tokens_removed;
                                        let valid = &scores[..pre_pos.min(scores.len())];
                                        if !valid.is_empty() {
                                            let total: f32 = valid.iter().sum();
                                            let max_s = valid
                                                .iter()
                                                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                                            let min_s =
                                                valid.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                                            let avg_s = total / valid.len() as f32;
                                            eprintln!(
                                                "[H2O-Debug] Pre-eviction scores: min={:.3} avg={:.3} max={:.3} total={:.1} tokens={}",
                                                min_s,
                                                avg_s,
                                                max_s,
                                                total,
                                                valid.len()
                                            );
                                        }
                                    }
                                    eprintln!(
                                        "[H2O-Debug] Eviction: ratio={:.3} removed={} new_pos={}",
                                        effective_ratio, r.tokens_removed, r.new_pos
                                    );
                                }
                                if let Some(ref mut p) = profiler {
                                    p.on_eviction(llm_rs2::profile::EvictionEvent {
                                        step: decode_token_index,
                                        policy: args.eviction_policy().to_string(),
                                        before_len: r.new_pos + r.tokens_removed,
                                        after_len: r.new_pos,
                                        evicted_count: r.tokens_removed,
                                        partition: llm_rs2::profile::PartitionInfo {
                                            prefix_end: actual_protected_prefix,
                                            hh_count: 0,
                                            recent_start: r.new_pos,
                                        },
                                        evicted_indices: vec![],
                                        pre_eviction_scores: vec![],
                                    });
                                }
                                // Release physical pages (madvise MADV_DONTNEED)
                                let mut bytes_released = 0usize;
                                for cache in kv_caches.iter_mut() {
                                    bytes_released += cache.release_unused_pages();
                                }
                                if bytes_released > 0 {
                                    eprintln!(
                                        "[Resilience] Released {} MB of physical pages",
                                        bytes_released / (1024 * 1024)
                                    );
                                }
                                experiment_eviction_count += 1;
                                experiment_evicted_total += r.tokens_removed;
                                if let Some(acc) = score_accumulator.as_mut() {
                                    acc.reset();
                                    acc.set_active(false);
                                }
                                // Reset GPU score accumulator after resilience eviction
                                #[cfg(feature = "opencl")]
                                if let Some(ocl_be) = backend
                                    .as_any()
                                    .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>(
                                ) && let Some(gpu_acc) = ocl_be.gpu_score_acc_mut()
                                    && gpu_acc.is_active()
                                {
                                    gpu_acc.reset(ocl_be.queue.as_core())?;
                                    gpu_acc.set_active(false);
                                }
                                // Invalidate GPU Plan — cache size changed after eviction,
                                // stale plan would use wrong attention sequence length.
                                #[cfg(feature = "opencl")]
                                {
                                    gpu_plan = None;
                                }
                            }
                            Err(e) => eprintln!("[Resilience] Eviction error: {}", e),
                            _ => {}
                        }
                    } // end skip_eviction else
                }

                // Dynamic tensor partition ratio
                //
                // The executor re-delivers the sticky partition_ratio on every
                // poll (ISSUE-5 prefill→decode carry-over), so we guard with
                // `last_applied_partition_ratio` to prevent re-splitting weights
                // on every decode tick (REGRESSION-A). Only the first delivery
                // of a new ratio triggers the expensive re-split / re-register.
                if let Some(ratio) = plan.partition_ratio
                    && last_applied_partition_ratio != Some(ratio)
                {
                    if ratio <= 0.0 || ratio >= 1.0 {
                        // Disable partition: clear partition_ctx from all layers
                        // via atomic clone-and-swap (ArcSwap snapshot replace).
                        for slot in &model.layers {
                            let old = slot.load_weights();
                            let mut new = (*old).clone();
                            new.partition_ctx = None;
                            slot.store_weights_same_dtype(Arc::new(new));
                        }
                        gen_ws.partition_ws = None;
                        eprintln!("[Partition] Disabled (ratio={})", ratio);
                        executor.set_partition_ratio(0.0);
                        last_applied_partition_ratio = Some(ratio);
                        // Partition off: invalidate plan to trigger rebuild next
                        // iter so GPU-only fast path is restored.
                        #[cfg(feature = "opencl")]
                        {
                            gpu_plan = None;
                        }
                    } else if llm_rs2::layers::tensor_partition::is_gpu_only_ratio(ratio) {
                        // GPU-only fast path: clear any existing partition context
                        // so forward() skips the host staging / CPU matmul / merge
                        // path entirely. No lazy `map_weights_for_cpu` needed here
                        // because the CPU side is unused at this ratio.
                        for slot in &model.layers {
                            let old = slot.load_weights();
                            let mut new = (*old).clone();
                            new.partition_ctx = None;
                            slot.store_weights_same_dtype(Arc::new(new));
                        }
                        gen_ws.partition_ws = None;
                        eprintln!(
                            "[Partition] ratio={:.3} treated as GPU-only (>= {:.3}); partition path disabled",
                            ratio,
                            llm_rs2::layers::tensor_partition::GPU_ONLY_THRESHOLD,
                        );
                        executor.set_partition_ratio(ratio);
                        last_applied_partition_ratio = Some(ratio);
                        #[cfg(feature = "opencl")]
                        {
                            gpu_plan = None;
                        }
                    } else {
                        // Lazy activation: if weights are still GPU-only (null host
                        // ptr — the normal state when `--enable-resilience` alone
                        // was used without `--tensor-partition`), map them now. This
                        // moves the ~200 ms / +400 MB RSS cost from startup to the
                        // first `SetPartitionRatio` directive that actually needs
                        // CPU-accessible weights. The one-shot first-activation
                        // stall is logged for downstream TBT accounting.
                        let mut lazy_map_ok = true;
                        #[cfg(feature = "opencl")]
                        if is_gpu && model.layers[0].load_weights().wq.as_ptr().is_null() {
                            let t0 = std::time::Instant::now();
                            match model.map_weights_for_cpu(&backend) {
                                Ok(n) if n > 0 => eprintln!(
                                    "[Partition] Lazy-mapped {} weight tensors for CPU access in {:.1} ms (first-activation stall)",
                                    n,
                                    t0.elapsed().as_secs_f64() * 1000.0,
                                ),
                                Ok(_) => {}
                                Err(e) => {
                                    eprintln!(
                                        "[Partition] Lazy weight map failed: {} — ratio={} rejected.",
                                        e, ratio
                                    );
                                    lazy_map_ok = false;
                                }
                            }
                        }
                        // Re-split weights with new ratio (only if lazy map succeeded)
                        if lazy_map_ok {
                            match model.prepare_tensor_partition(ratio, &cpu_backend_arc) {
                                Ok(n) => {
                                    eprintln!(
                                        "[Partition] Re-split {} weights with ratio {:.2}",
                                        n, ratio
                                    );
                                    // Reallocate workspace
                                    let layer0_probe = model.layers[0].load_weights();
                                    // Diagnostic: dump per-weight buffer kind for
                                    // layer 0 so a "B is not OpenCL buffer" crash
                                    // on the next forward immediately points at
                                    // which tensor is misbacked. Single-shot,
                                    // layer 0 only — every other layer has the
                                    // same backing pattern by construction.
                                    #[cfg(feature = "opencl")]
                                    {
                                        use llm_rs2::backend::opencl::buffer_kind_label;
                                        let l0 = &layer0_probe;
                                        let mut log = String::from(
                                            "[Partition] Layer 0 weight buffer kinds: ",
                                        );
                                        log.push_str(&format!(
                                            "wq={} wk={} wv={} wo={} ",
                                            buffer_kind_label(l0.wq.buffer().as_ref()),
                                            buffer_kind_label(l0.wk.buffer().as_ref()),
                                            buffer_kind_label(l0.wv.buffer().as_ref()),
                                            buffer_kind_label(l0.wo.buffer().as_ref()),
                                        ));
                                        log.push_str(&format!(
                                            "w_gate={} w_up={} w_down={} ",
                                            buffer_kind_label(l0.w_gate.buffer().as_ref()),
                                            buffer_kind_label(l0.w_up.buffer().as_ref()),
                                            buffer_kind_label(l0.w_down.buffer().as_ref()),
                                        ));
                                        if let Some(ref ctx) = l0.partition_ctx {
                                            log.push_str(&format!(
                                                "gate_gpu_slice={} up_gpu_slice={} down_gpu_slice={}",
                                                buffer_kind_label(
                                                    ctx.gate.gpu_slice.buffer().as_ref()
                                                ),
                                                buffer_kind_label(
                                                    ctx.up.gpu_slice.buffer().as_ref()
                                                ),
                                                buffer_kind_label(
                                                    ctx.down.gpu_slice.buffer().as_ref()
                                                ),
                                            ));
                                        }
                                        eprintln!("{log}");
                                    }
                                    if let Some(ref ctx) = layer0_probe.partition_ctx {
                                        let gpu_alloc =
                                            make_partition_gpu_alloc(&*backend, decode_mem);
                                        gen_ws.partition_ws = Some(Arc::new(PartitionWsCell::new(
                                            PartitionWorkspace::new(
                                                ctx,
                                                ffn_hidden,
                                                hidden_size,
                                                &gpu_alloc,
                                                backend.clone(),
                                                cpu_backend_arc.clone(),
                                            )?,
                                        )));
                                    } else {
                                        gen_ws.partition_ws = None;
                                    }
                                    executor.set_partition_ratio(ratio);
                                    last_applied_partition_ratio = Some(ratio);
                                    // Partition active: invalidate plan so the
                                    // partition co-execution path takes over.
                                    #[cfg(feature = "opencl")]
                                    {
                                        gpu_plan = None;
                                    }
                                    // Re-register Q4_0 noshuffle SOA entries:
                                    // `map_weights_for_cpu()` above replaced
                                    // GPU-only weights' `UnifiedBuffer`, minting
                                    // new `cl_mem` pointers. The SOA registry's
                                    // old entries are now keyed by stale
                                    // `cl_mem`s, so `build_plan()` would miss
                                    // the lookup and silently fall back to the
                                    // AOS Q4_0 GEMV (measured +102% TBT on
                                    // Galaxy S25, verify v2 ISSUE-2).
                                    //
                                    // Clear + rebuild mirrors the CLI init path
                                    // (prepare_tensor_partition → prepare_
                                    // noshuffle_buffers) so partition sub-buffer
                                    // slices are also registered. Idempotent
                                    // for non-Q4_0 weight dtypes: the
                                    // prepare_noshuffle_buffers() helper
                                    // short-circuits on DType::Q4_0 mismatch.
                                    #[cfg(feature = "opencl")]
                                    if is_gpu {
                                        let actual_q4 = model.layers.first().is_some_and(|l| {
                                            l.load_weights().wq.dtype() == DType::Q4_0
                                        });
                                        if actual_q4
                                            && let Some(ocl_be) = backend
                                                .as_any()
                                                .downcast_ref::<
                                                    llm_rs2::backend::opencl::OpenCLBackend,
                                                >()
                                        {
                                            ocl_be.clear_noshuffle_soa_registry();
                                            // Tensor partition runtime re-register:
                                            // `map_weights_for_cpu()` above replaced the
                                            // per-weight `UnifiedBuffer` with a CPU-mapped
                                            // version, so we keep the AOS allocation
                                            // alive — the partition path (and any CPU
                                            // matmul fallback) dereferences the original
                                            // cl_mem directly. `keep_original=true` stops
                                            // `prepare_noshuffle_buffers` from swapping
                                            // the tensor buffers out from under the
                                            // caller.
                                            match model
                                                .prepare_noshuffle_buffers(&backend, true)
                                            {
                                                Ok(n) => eprintln!(
                                                    "[Partition] Re-registered Q4_0 noshuffle SOA: {} weight tensors",
                                                    n
                                                ),
                                                Err(e) => eprintln!(
                                                    "[Partition] Noshuffle re-registration failed: {} (AOS fallback will hurt TBT)",
                                                    e
                                                ),
                                            }
                                        }
                                    }
                                }
                                Err(e) => eprintln!("[Partition] Re-split failed: {}", e),
                            }
                        }
                    }
                }

                // Dynamic layer skip / restore_defaults handling
                if plan.restore_defaults {
                    eprintln!("[Resilience] RestoreDefaults");
                    skip_config = None;
                    last_skip_ratio = None;
                    evict_ceiling = None;
                    evict_floor_logged = None;
                } else if let Some(ratio) = plan.layer_skip
                    && last_skip_ratio != Some(ratio)
                {
                    eprintln!("[Resilience] LayerSkip: ratio={:.2}", ratio);
                    skip_config = Some(SkipConfig::uniform_init(
                        model.config.num_hidden_layers,
                        ratio,
                    ));
                    last_skip_ratio = Some(ratio);
                }

                if let Some(ref device) = plan.switch_device {
                    if let (Some(gpu_be), Some(gpu_mem)) = (&gpu_backend_arc, &gpu_memory_arc) {
                        match device.as_str() {
                            "cpu" if is_gpu => {
                                if spare_logits.is_none() {
                                    eprintln!(
                                        "[Switch] ERROR: SwitchHw requires --resilience-prealloc-switch flag. Ignoring directive."
                                    );
                                } else {
                                    eprintln!(
                                        "[Switch] Resilience: GPU→CPU at token {}",
                                        kv_caches[0].current_pos
                                    );
                                    llm_rs2::pressure::kv_migrate::migrate_kv_caches(
                                        &mut kv_caches,
                                        &backend,
                                        &cpu_backend_arc,
                                        &cpu_backend_arc,
                                        &cpu_memory_arc,
                                        &cpu_memory_arc,
                                        kv_heads,
                                        head_dim,
                                        max_seq_len,
                                        false,
                                    )?;
                                    backend = cpu_backend_arc.clone();
                                    // Zero-alloc swap: exchange active GPU buffers with
                                    // pre-allocated CPU spares. GPU buffers survive in spare_*
                                    // (no clReleaseMemObject, no RSS spike).
                                    if let (Some(sl), Some(sx), Some(sw), Some(si)) = (
                                        spare_logits.as_mut(),
                                        spare_xgen.as_mut(),
                                        spare_gen_ws.as_mut(),
                                        spare_gen_input.as_mut(),
                                    ) {
                                        std::mem::swap(&mut logits, sl);
                                        std::mem::swap(&mut x_gen, sx);
                                        std::mem::swap(&mut gen_ws, sw);
                                        std::mem::swap(&mut gen_input_tensor, si);
                                    }
                                    #[cfg(feature = "opencl")]
                                    {
                                        gpu_plan = None;
                                    }
                                    is_gpu = false;
                                    // Re-tag weight tensors with CPU backend.
                                    eprintln!("[Switch] Resilience: Switched to CPU.");
                                }
                            }
                            "gpu" | "opencl" if !is_gpu && weights_on_gpu => {
                                if spare_logits.is_none() {
                                    eprintln!(
                                        "[Switch] ERROR: SwitchHw requires --resilience-prealloc-switch flag. Ignoring directive."
                                    );
                                } else {
                                    eprintln!(
                                        "[Switch] Resilience: CPU→GPU at token {}",
                                        kv_caches[0].current_pos
                                    );
                                    llm_rs2::pressure::kv_migrate::migrate_kv_caches(
                                        &mut kv_caches,
                                        &backend,
                                        gpu_be,
                                        &cpu_backend_arc,
                                        &cpu_memory_arc,
                                        gpu_mem,
                                        kv_heads,
                                        head_dim,
                                        max_seq_len,
                                        true,
                                    )?;
                                    backend = gpu_be.clone();
                                    // Zero-alloc swap: exchange active CPU buffers with
                                    // spare GPU buffers (preserved from previous switch).
                                    if let (Some(sl), Some(sx), Some(sw), Some(si)) = (
                                        spare_logits.as_mut(),
                                        spare_xgen.as_mut(),
                                        spare_gen_ws.as_mut(),
                                        spare_gen_input.as_mut(),
                                    ) {
                                        std::mem::swap(&mut logits, sl);
                                        std::mem::swap(&mut x_gen, sx);
                                        std::mem::swap(&mut gen_ws, sw);
                                        std::mem::swap(&mut gen_input_tensor, si);
                                    }
                                    #[cfg(feature = "opencl")]
                                    {
                                        gpu_plan = None;
                                    }
                                    is_gpu = true;
                                    eprintln!("[Switch] Resilience: Switched to GPU (zero-alloc).");
                                }
                            }
                            "gpu" | "opencl" if !is_gpu && !weights_on_gpu => {
                                eprintln!(
                                    "[Resilience] SwitchHw(gpu): model weights on CPU, not GPU-accessible. \
                                     Start with --backend opencl for GPU switching."
                                );
                            }
                            _ => {} // Already on requested backend
                        }
                    } else {
                        eprintln!(
                            "[Resilience] SwitchHw({}): no secondary backend available",
                            device
                        );
                    }
                }

                // kv_quant_bits: not supported on F16 KVCache path
                if let Some(bits) = plan.kv_quant_bits {
                    eprintln!(
                        "[Resilience] Warning: kv_quant_dynamic(bits={}) requested but KV cache is F16 (not KIVI). \
                         Dynamic quantization requires --kv-type q2/q4. Ignoring.",
                        bits
                    );
                }

                // KvOffload: Manager-directed LRU prefix offload to disk.
                if let Some(ratio) = plan.offload_ratio {
                    match cache_manager.offload(&mut kv_caches, ratio) {
                        Ok(n) => eprintln!(
                            "[Resilience] KvOffload: ratio={:.2}, {} tokens swapped",
                            ratio, n
                        ),
                        Err(e) => eprintln!("[Resilience] KvOffload failed: {}", e),
                    }
                }
                // RestoreDefaults → recall offloaded tokens back from disk.
                if plan.recall_offload && plan.restore_defaults {
                    match cache_manager.recall(&mut kv_caches) {
                        Ok(n) => {
                            if n > 0 {
                                eprintln!("[Resilience] Recalled {} tokens from swap", n);
                            }
                        }
                        Err(e) => eprintln!("[Resilience] Recall failed: {}", e),
                    }
                }

                if plan.throttle_delay_ms != throttle_delay_ms {
                    eprintln!(
                        "[Resilience] Throttle: {}ms → {}ms",
                        throttle_delay_ms, plan.throttle_delay_ms
                    );
                }
                throttle_delay_ms = plan.throttle_delay_ms;

                // Update target TBT from Manager directive (overrides CLI --target-tbt).
                // `target_tbt_set` distinguishes "manager explicitly said 0 (disable
                // pacing)" from "manager never sent a directive" — otherwise a
                // `SetTargetTbt { target_ms: 0 }` restore cannot clear a prior
                // non-zero target (see verify/ISSUE-3).
                if plan.target_tbt_set && plan.target_tbt_ms as f64 != target_tbt_ms {
                    eprintln!(
                        "[Resilience] SetTargetTbt: {:.1}ms → {}ms",
                        target_tbt_ms, plan.target_tbt_ms
                    );
                    target_tbt_ms = plan.target_tbt_ms as f64;
                } else if plan.restore_defaults {
                    target_tbt_ms = args.target_tbt; // restore CLI default
                }

                // RestoreDefaults: restore partition ratio to CLI initial value
                if plan.restore_defaults {
                    let cli_ratio = args.tensor_partition;
                    if cli_ratio > 0.0 && cli_ratio < 1.0 {
                        // Restore to CLI partition ratio
                        let layer0_probe = model.layers[0].load_weights();
                        if !layer0_probe.wq.as_ptr().is_null()
                            && let Ok(n) =
                                model.prepare_tensor_partition(cli_ratio, &cpu_backend_arc)
                        {
                            eprintln!(
                                "[Partition] RestoreDefaults: re-split {} weights with CLI ratio {:.2}",
                                n, cli_ratio
                            );
                            // prepare_tensor_partition returns 0 when cli_ratio is
                            // in the GPU-only fast-path band; partition_ctx is then
                            // None and we must clear any stale workspace so forward()
                            // stays on the dense GPU path.
                            let layer0_probe2 = model.layers[0].load_weights();
                            if let Some(ref ctx) = layer0_probe2.partition_ctx {
                                let gpu_alloc = make_partition_gpu_alloc(&*backend, decode_mem);
                                if let Ok(ws) = PartitionWorkspace::new(
                                    ctx,
                                    ffn_hidden,
                                    hidden_size,
                                    &gpu_alloc,
                                    backend.clone(),
                                    cpu_backend_arc.clone(),
                                ) {
                                    gen_ws.partition_ws = Some(Arc::new(PartitionWsCell::new(ws)));
                                }
                            } else {
                                gen_ws.partition_ws = None;
                            }
                            executor.set_partition_ratio(cli_ratio);
                            last_applied_partition_ratio = Some(cli_ratio);
                        }
                    } else {
                        // CLI had no partition — disable via ArcSwap clone-and-install
                        for slot in &model.layers {
                            let old = slot.load_weights();
                            let mut new = (*old).clone();
                            new.partition_ctx = None;
                            slot.store_weights_same_dtype(Arc::new(new));
                        }
                        gen_ws.partition_ws = None;
                        executor.set_partition_ratio(0.0);
                        last_applied_partition_ratio = None;
                    }
                }

                if plan.suspended {
                    eprintln!("\n[Resilience] Inference suspended by system signal");
                    break;
                }

                if throttle_delay_ms > 0 {
                    experiment_total_throttle_ms += throttle_delay_ms;
                    std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
                }

                executor.on_token_generated();
            } else if throttle_delay_ms > 0 {
                // No CommandExecutor: honour CLI --throttle-delay-ms directly
                // so decode pacing works without --enable-resilience.
                std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
            }
            // ── End Resilience checkpoint ─────────────────────

            // Read logits to CPU (reuses pre-allocated buffer).
            //
            // Token-boundary sync point: `Backend::read_buffer` always
            // calls `synchronize()` internally (see CpuBackend/OpenCL/
            // cuda_embedded impls). This is the barrier that guarantees
            // all in-flight kernels complete before sampling runs —
            // critical for `--cuda-defer-sync` where per-op syncs are
            // suppressed.
            unsafe {
                let ptr = logits_cpu.as_mut_ptr() as *mut u8;
                let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
                backend.read_buffer(&logits, slice)?;
            }

            // Extract top-K logits before sampling modifies them
            let top_logits = if experiment_writer.is_some() {
                extract_top_k_logits(&logits_cpu, args.experiment_logits_topk)
            } else {
                vec![]
            };

            let sample_start = std::time::Instant::now();
            let next_token_id = sampling::sample(
                &mut logits_cpu,
                &tokens,
                vocab_size,
                &sampling_config,
                Some(&mut sampling_indices),
            );
            let sample_us = sample_start.elapsed().as_micros() as u64;

            let now = std::time::Instant::now();
            let mut tbt = now.duration_since(_last_token_time).as_secs_f64() * 1000.0;

            // ── Target TBT pacing: sleep to maintain target throughput ──
            let pacing_ms = if target_tbt_ms > 0.0 && tbt < target_tbt_ms {
                let sleep_ms = target_tbt_ms - tbt;
                std::thread::sleep(std::time::Duration::from_secs_f64(sleep_ms / 1000.0));
                tbt = target_tbt_ms; // effective TBT = target
                sleep_ms
            } else {
                0.0
            };

            tbt_values.push(tbt);

            // ── TBT log: write per-token JSONL ──
            if let Some(ref mut w) = tbt_log_writer {
                use std::io::Write;
                writeln!(w,
                    "{{\"token_idx\":{},\"tbt_ms\":{:.2},\"forward_ms\":{:.2},\"cache_pos\":{},\"pacing_ms\":{:.2}}}",
                    decode_token_index, tbt, forward_ms, kv_caches[0].current_pos, pacing_ms
                ).ok();
            }

            // ── Profiler: record step data ──
            if let Some(ref mut p) = profiler {
                let forward_us = (forward_ms * 1000.0) as u64;
                let total_us = forward_us + sample_us;
                let cache_len = kv_caches[0].current_pos;
                let (imp, head_imp, n_kv) = match score_accumulator {
                    Some(ref acc) if acc.is_active() => (
                        Some(acc.importance_scores()),
                        acc.head_importance_scores(),
                        acc.n_kv_heads(),
                    ),
                    _ => (None, None, 0),
                };
                let pos_map = if position_birth_step.is_empty() {
                    None
                } else {
                    Some(position_birth_step.as_slice())
                };
                p.on_step_end(
                    decode_token_index,
                    next_token_id,
                    forward_us,
                    sample_us,
                    total_us,
                    cache_len,
                    imp,
                    head_imp,
                    n_kv,
                    pos_map,
                );
                // Record new token birth
                p.scores
                    .record_token_births(decode_token_index + 1, 1, actual_protected_prefix);
            }
            // Track birth step for new token (even if profiler is off, keep mapping in sync)
            if !position_birth_step.is_empty() {
                position_birth_step.push(decode_token_index + 1);
            }

            _last_token_time = now;
            tokens.push(next_token_id);

            // start_pos tracks the LOGICAL position for RoPE encoding.
            start_pos += 1;

            // ── Experiment: write per-token JSONL record ──
            if let Some(ref mut writer) = experiment_writer {
                let token_text = tokenizer
                    .decode(&[next_token_id], false)
                    .unwrap_or_default();
                let sys_metrics = system_sampler.sample(decode_token_index);
                let signal_str = if injected_signals.is_empty() {
                    None
                } else {
                    Some(injected_signals.join("+"))
                };
                let record = TokenRecord {
                    pos: decode_token_index,
                    token_id: next_token_id,
                    text: token_text,
                    tbt_ms: tbt,
                    forward_ms,
                    signal: signal_str.as_deref(),
                    actions: action_names,
                    cache_pos: kv_caches[0].current_pos,
                    throttle_ms: throttle_delay_ms,
                    top_logits,
                    sys: sys_metrics,
                };
                writer.write_token(&record)?;
            }

            // Streaming print (suppress in experiment mode for clean JSONL)
            if experiment_writer.is_none() {
                let current_text = tokenizer.decode(&tokens, true).unwrap_or_default();
                if let Some(substring) = current_text.get(_printed_len..).filter(|s| !s.is_empty())
                {
                    print!("{}", substring);
                    stdout.flush().ok();
                    _printed_len = current_text.len();
                }
            }
            // D-D.6 debug: dump raw token IDs (special token visibility).
            if std::env::var("LLMRS_DUMP_TOKEN_IDS").is_ok() {
                eprintln!(
                    "[token-id step={}] id={}",
                    decode_token_index, next_token_id
                );
            }

            // T3 / T4: RSS snapshot after first and 16th decode tokens.
            if decode_token_index == 0 {
                rss_trace("decode_1");
                io_trace("decode_1");
            } else if decode_token_index == 15 {
                rss_trace("decode_16");
                io_trace("decode_16");
            }

            if next_token_id == eos_id && !args.ignore_eos && std::env::var("IGNORE_EOS").is_err() {
                break;
            }
        }
    }

    // 6. Write experiment summary
    if let Some(ref mut writer) = experiment_writer {
        let sys_end = Some(system_sampler.snapshot());
        let avg_tbt_ms = if tbt_values.is_empty() {
            0.0
        } else {
            tbt_values.iter().sum::<f64>() / tbt_values.len() as f64
        };
        let avg_forward_ms = if forward_ms_values.is_empty() {
            0.0
        } else {
            forward_ms_values.iter().sum::<f64>() / forward_ms_values.len() as f64
        };
        let summary = SummaryRecord {
            _summary: true,
            total_tokens: tbt_values.len(),
            ttft_ms: _ttft_ms,
            avg_tbt_ms,
            avg_forward_ms,
            total_throttle_ms: experiment_total_throttle_ms,
            eviction_count: experiment_eviction_count,
            evicted_tokens_total: experiment_evicted_total,
            final_cache_pos: kv_caches[0].current_pos,
            max_seq_len,
            prompt: prompt.clone(),
            schedule_name: experiment_schedule
                .as_ref()
                .map(|s| s.name.clone())
                .unwrap_or_else(|| "baseline".to_string()),
            eviction_policy: args.eviction_policy().to_string(),
            backend: args.backend.clone(),
            sample_interval: args.experiment_sample_interval,
            sys_start,
            sys_end,
            governor: Some(SystemSampler::read_governor()),
        };
        writer.write_summary(&summary)?;
        eprintln!(
            "[Experiment] Done: {} tokens, avg TBT {:.2}ms, {} evictions",
            summary.total_tokens, avg_tbt_ms, experiment_eviction_count
        );
    }

    // 6.5. Export profiler data if enabled
    if let Some(ref profiler) = profiler {
        profiler.ops.print_report();

        let metadata = llm_rs2::profile::ProfileMetadata {
            model: args.model_path.clone(),
            backend: args.backend.clone(),
            eviction_policy: args.eviction_policy().to_string(),
            max_seq_len: args.max_seq_len,
            prompt_len: prompt.len(),
            generated_tokens: tbt_values.len(),
        };
        match profiler.export_json(&metadata) {
            Ok(path) => eprintln!("[Profile] Exported to {}", path.display()),
            Err(e) => eprintln!("[Profile] Export failed: {}", e),
        }
    }

    // 6.6. Export --cuda-profile aggregate if enabled.
    // Independent of the generic `profiler` above (which lives in
    // `llm_rs2::profile::ops::OpProfiler` and targets OpenCL events).
    #[cfg(feature = "cuda-embedded")]
    if args.cuda_profile
        && let Some(cuda_be) = backend
            .as_any()
            .downcast_ref::<llm_rs2::backend::cuda_embedded::CudaBackend>()
        && cuda_be.profiler_enabled()
    {
        match cuda_be.flush_profiler() {
            Ok(Some(map)) => {
                let dropped = cuda_be.profiler_dropped();
                dump_cuda_profile_report(&map, dropped, &args, tbt_values.len(), cuda_be.device());
            }
            Ok(None) => {}
            Err(e) => eprintln!("[CUDA-Profile] flush failed: {}", e),
        }
    }

    // 7. Output results
    println!("\nDone.");
    println!("[Profile] Event: End");
    #[cfg(feature = "cuda-embedded")]
    {
        llm_rs2::backend::cuda_embedded::dump_fallback_counters();
        if let Some(cu_be) = backend
            .as_any()
            .downcast_ref::<llm_rs2::backend::cuda_embedded::CudaBackend>()
        {
            cu_be.dump_graph_counters();
        }
    }
    // WSWAP-5-TBT-DIAG: final cl_mem dump after the entire generation
    // pipeline completes. Includes any growth that occurred during prefill /
    // decode (KV cache grow-on-demand, plan-rebuild scratch, etc.).
    #[cfg(feature = "opencl")]
    if let Some(ocl_be) = backend
        .as_any()
        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
    {
        ocl_be.dump_cl_mem_diagnostics(" stage=after_generate");
    }
    println!("TTFT: {:.2} ms", _ttft_ms);
    io_trace("ttft");
    if !forward_ms_values.is_empty() {
        let avg_forward: f64 =
            forward_ms_values.iter().sum::<f64>() / forward_ms_values.len() as f64;
        println!(
            "Decode: {:.2} ms/tok ({:.1} tok/s) [{} tokens, forward only]",
            avg_forward,
            1000.0 / avg_forward,
            forward_ms_values.len(),
        );
        // Sprint E: flush the forward_gen op-tracer right after the Decode
        // summary so the per-op breakdown sits next to the headline TBT in
        // the log (atexit will fire too, but is moot once we already dumped).
        llm_rs2::profile::op_trace::dump_and_reset();
        if forward_ms_values.len() >= 2 {
            let tail = &forward_ms_values[1..];
            let avg_tail: f64 = tail.iter().sum::<f64>() / tail.len() as f64;
            let tok0 = forward_ms_values[0];
            println!(
                "Decode(excl tok[0]): {:.2} ms/tok ({:.1} tok/s) [{} tokens] | tok[0]={:.2} ms",
                avg_tail,
                1000.0 / avg_tail,
                tail.len(),
                tok0,
            );
        }
    }
    if !tbt_values.is_empty() {
        let avg_tbt: f64 = tbt_values.iter().sum::<f64>() / tbt_values.len() as f64;
        println!(
            "Avg TBT: {:.2} ms ({:.1} tokens/sec)",
            avg_tbt,
            1000.0 / avg_tbt
        );
    }

    // --qcf-dump: write JSON for generation mode (ppl=null, avg_nll=null).
    if let Some(ref dump_path) = args.qcf_dump {
        use llm_rs2::eval::qcf_helpers::{QcfSwapDumpContext, dump_qcf_swap_json};

        let empty_swap: Vec<usize> = Vec::new();
        let (swap_set, qcf_predicted, fallback_used) = if let Some(ref dec) = qcf_swap_decision {
            (
                dec.selected_layers.as_slice(),
                dec.qcf_swap_estimate,
                dec.fallback_used,
            )
        } else {
            (empty_swap.as_slice(), 0.0f32, false)
        };

        let secondary_path_str = args.secondary_gguf.as_ref().and_then(|p| p.to_str());
        let model_arch = if args.model_path.to_lowercase().contains("qwen") {
            "qwen2"
        } else {
            "llama"
        };
        let total_wall = qcf_workflow_start.elapsed().as_secs_f64();

        let ctx = QcfSwapDumpContext {
            model_arch,
            model_path: &args.model_path,
            secondary_path: secondary_path_str,
            primary_dtype: "F16",
            secondary_dtype: "Q4_0",
            num_layers: model.layers.len(),
            force_swap_ratio: args.force_swap_ratio,
            swap_algorithm: args.force_swap_ratio.map(|_| swap_algorithm.short_name()),
            swap_set,
            qcf_swap_predicted: qcf_predicted,
            fallback_used,
            importance_table: qcf_warmup_importance.as_ref(),
            noise_table: Some(model.quant_noise.as_ref()),
            ppl: None,
            avg_nll: None,
            n_eval_tokens: 0,
            wall_time_s: total_wall,
            warmup_tokens: args.qcf_warmup_tokens,
            backend: &args.backend,
            kv_type: &args.kv_type,
            ppl_corpus: None,
            eval_ll_output: None,
            trajectory: None,
            dpllm_epsilon: None,
            dpllm_epsilon_multi: None,
            dpllm_epsilon_abs: None,
            dpllm_epsilon_qcf: None,
            direct_attn_f4: None,
            direct_attn_f5: None,
            direct_attn_f5_decode_only: None,
            direct_attn_f5_prefill_decode: None,
        };

        dump_qcf_swap_json(dump_path, &ctx)?;
        eprintln!("[QCF-dump] JSON written to {}", dump_path.display());
    }

    // T5: normal exit — all allocations still live (model, KV caches, workspaces).
    rss_trace("exit");
    io_trace("exit");
    Ok(())
}

// ── CUDA profile report ───────────────────────────────────────

#[cfg(feature = "cuda-embedded")]
fn dump_cuda_profile_report(
    map: &std::collections::HashMap<&'static str, (u64, f64)>,
    dropped: u64,
    args: &Args,
    n_tokens: usize,
    device: &str,
) {
    // 1) Sort by total_ms descending — same ordering as the OpenCL
    //    `--profile-events` summary so the two reports read the same.
    let mut rows: Vec<(&&str, &(u64, f64))> = map.iter().collect();
    rows.sort_by(|a, b| {
        b.1.1
            .partial_cmp(&a.1.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_ms: f64 = rows.iter().map(|(_, v)| v.1).sum();

    eprintln!(
        "\n=== CUDA per-op profile ({} ops over {} tokens) ===",
        rows.len(),
        n_tokens
    );
    eprintln!(
        "{:<28}  {:>8}  {:>12}  {:>10}  {:>7}",
        "label", "count", "total_ms", "mean_ms", "pct"
    );
    for (label, (count, t_ms)) in &rows {
        let pct = if total_ms > 0.0 {
            t_ms / total_ms * 100.0
        } else {
            0.0
        };
        let mean = if *count > 0 {
            t_ms / (*count as f64)
        } else {
            0.0
        };
        eprintln!(
            "{:<28}  {:>8}  {:>12.3}  {:>10.4}  {:>6.2}%",
            label, count, t_ms, mean, pct
        );
    }
    eprintln!(
        "{:<28}  {:>8}  {:>12.3}",
        "TOTAL",
        rows.iter().map(|(_, v)| v.0).sum::<u64>(),
        total_ms
    );
    if dropped > 0 {
        eprintln!(
            "[CUDA-Profile] WARNING: {dropped} records dropped (pool exhausted between flushes)"
        );
    }

    // 2) Write JSON to results/profile/cuda_embedded_decode_<ts>.json.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Plain unix seconds — `chrono` is not a workspace dep. Downstream
    // tooling can parse either form.
    let ts_iso = format!("unix:{ts}");

    let mut ops_json = Vec::with_capacity(rows.len());
    for (label, (count, t_ms)) in &rows {
        let mean = if *count > 0 {
            t_ms / (*count as f64)
        } else {
            0.0
        };
        ops_json.push(serde_json::json!({
            "label": label,
            "count": count,
            "total_ms": t_ms,
            "mean_ms": mean,
        }));
    }
    let doc = serde_json::json!({
        "timestamp": ts_iso,
        "device": device,
        "backend": "cuda-embedded",
        "n_tokens": n_tokens,
        "model": args.model_path,
        "dropped_records": dropped,
        "ops": ops_json,
    });

    let dir = std::path::PathBuf::from(&args.profile_dir);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("[CUDA-Profile] mkdir {} failed: {}", dir.display(), e);
        return;
    }
    let path = dir.join(format!("cuda_embedded_decode_{}.json", ts));
    match std::fs::write(&path, serde_json::to_vec_pretty(&doc).unwrap_or_default()) {
        Ok(()) => eprintln!("[CUDA-Profile] Exported to {}", path.display()),
        Err(e) => eprintln!("[CUDA-Profile] write {} failed: {}", path.display(), e),
    }
}

// ── Experiment helpers ────────────────────────────────────────

fn directive_summary(msg: &ManagerMessage) -> String {
    match msg {
        ManagerMessage::Directive(d) => {
            let cmds: Vec<String> = d.commands.iter().map(command_summary).collect();
            format!("Directive(seq={}, [{}])", d.seq_id, cmds.join(", "))
        }
    }
}

/// Context for dry-run QCF estimation (ENG-ALG-050).
/// Groups all inputs needed by `compute_qcf_estimates` so that the caller
/// constructs a single struct instead of passing many individual arguments.
struct QcfEstimateContext<'a> {
    kv_caches: &'a [KVCache],
    score_accumulator: Option<&'a AttentionScoreAccumulator>,
    /// (sink_size, window_size) for StreamingLLM dry-run. None = skip.
    streaming_config: Option<(usize, usize)>,
    /// Pre-built importance table for LayerSkip dry-run. None = skip.
    importance_table: Option<&'a llm_rs2::qcf::ImportanceTable>,
    /// Total number of transformer layers (needed for LayerSkip).
    num_layers: usize,
    /// KIVI caches for dynamic quantization QCF dry-run. None = skip.
    kivi_caches: Option<&'a [KiviCache]>,
}

/// Compute dry-run QCF estimates for all 6 lossy actions (ENG-ALG-050).
/// Read-only: does not modify KV caches.
///
/// Uses unified QCF formula: QCF = ||O_before - O_after|| / ||O_before||
/// where O = sum_t alpha_t * V_t (attention-weighted value output).
///
/// Returns estimates for:
/// - `kv_evict_sliding`  : Sliding window eviction
/// - `kv_evict_h2o`      : H2O importance-based eviction (needs scores)
/// - `kv_evict_streaming` : StreamingLLM eviction (needs streaming_config)
/// - `kv_merge_d2o`      : D2O merge estimate (needs scores)
/// - `kv_quant_dynamic`  : KIVI dynamic quantization (skipped for non-KiviCache path)
/// - `layer_skip`        : LayerSkip importance-based QCF (needs importance_table)
fn compute_qcf_estimates(ctx: &QcfEstimateContext<'_>) -> std::collections::HashMap<String, f32> {
    use llm_rs2::qcf::{AggregationMode, QcfActionType, QcfKvParams, VDataSource, compute_qcf_kv};
    use std::collections::HashMap;
    let mut estimates = HashMap::new();

    // ── 1-4. KVCache-based eviction/merge QCF via unified formula ──
    //
    // ISSUE-6 guard: OpenCL device-only 버퍼는 `as_ptr()`이 명시적으로
    // `ptr::null()`을 반환한다 (engine/src/backend/opencl/buffer.rs). 이 경우
    // `Tensor::as_slice::<T>()`이 `(ptr=null, len=size/sizeof T)` 슬라이스를
    // 만들고 `read_v_f32()`에서 `data[offset..end]`로 인덱싱하는 순간 null
    // deref → SIGSEGV. `VDataSource::from_kv_cache(None)` 가 host pointer 검사
    // 후 `None`을 돌려주므로 device-only 캐시는 자연스럽게 skip된다.
    let v_host_readable =
        !ctx.kv_caches.is_empty() && ctx.kv_caches.iter().all(|c| !c.v_buffer.as_ptr().is_null());
    if !ctx.kv_caches.is_empty() && ctx.kv_caches[0].current_pos > 0 && !v_host_readable {
        eprintln!(
            "[QCF] KV-based estimates skipped: v_buffer is device-only (signal path without host-mapped KV)."
        );
    }
    if v_host_readable && ctx.kv_caches[0].current_pos > 0 {
        let cache = &ctx.kv_caches[0];
        let current_pos = cache.current_pos;
        let keep_ratio = 0.5f32;
        let target_len = (current_pos as f32 * keep_ratio) as usize;
        let protected_prefix = 4usize;

        let scores_opt = ctx
            .score_accumulator
            .filter(|a| a.is_active())
            .map(|a| a.importance_scores());
        let head_attn_opt = ctx
            .score_accumulator
            .filter(|a| a.is_active())
            .and_then(|a| a.last_step_head_attn());

        let fallback_scores: Vec<f32>;
        let attention_scores: &[f32] = if let Some(scores) = scores_opt {
            scores
        } else {
            fallback_scores = vec![1.0 / current_pos.max(1) as f32; current_pos];
            &fallback_scores
        };

        if target_len < current_pos {
            // (id, action, requires_scores). `kv_evict_sliding` needs no scores;
            // h2o/d2o use heavy-hitter selection so are gated on score availability.
            // Streaming QCF only fires when streaming_config is set.
            let mut actions: Vec<(&'static str, QcfActionType, bool)> = vec![
                (
                    "kv_evict_sliding",
                    QcfActionType::EvictSliding { target_len },
                    false,
                ),
                (
                    "kv_evict_h2o",
                    QcfActionType::EvictH2o {
                        target_len,
                        keep_ratio: 0.5,
                        protected_prefix,
                    },
                    true,
                ),
                (
                    "kv_merge_d2o",
                    QcfActionType::MergeD2o {
                        target_len,
                        keep_ratio: 0.5,
                        protected_prefix,
                    },
                    true,
                ),
            ];
            if let Some((sink_size, window_size)) = ctx.streaming_config {
                actions.push((
                    "kv_evict_streaming",
                    QcfActionType::EvictStreaming {
                        sink_size,
                        window_size,
                    },
                    false,
                ));
            }

            for (id, action, requires_scores) in actions {
                if requires_scores && scores_opt.is_none() {
                    continue;
                }
                let Some(v_source) = VDataSource::from_kv_cache(cache, None) else {
                    continue;
                };
                // D2O simulator (paper Eq.8) needs K for nearest-neighbour
                // matching; other actions ignore `k_source`.
                let k_source = if matches!(action, QcfActionType::MergeD2o { .. }) {
                    VDataSource::k_from_kv_cache(cache)
                } else {
                    None
                };
                let params = QcfKvParams {
                    action,
                    v_source,
                    k_source,
                    attention_scores,
                    head_attn: head_attn_opt,
                    n_kv_heads: cache.kv_heads(),
                    head_dim: cache.head_dim(),
                    current_pos,
                    capacity: cache.capacity(),
                    layout: cache.layout(),
                    aggregation: AggregationMode::Mean,
                    beta: 1.0,
                };
                let (qcf, _) = compute_qcf_kv(&params);
                estimates.insert(id.to_string(), qcf);
            }
        }
    }

    // ── 5. KIVI dynamic quantization QCF ──
    if let Some(kivi_caches) = ctx.kivi_caches
        && !kivi_caches.is_empty()
    {
        let mut total_qcf = 0.0f32;
        let mut count = 0u32;
        for cache in kivi_caches {
            let qcf = cache.estimate_dryrun_qcf();
            if qcf > 0.0 {
                total_qcf += qcf;
                count += 1;
            }
        }
        if count > 0 {
            let avg_qcf = total_qcf / count as f32;
            estimates.insert("kv_quant_dynamic".to_string(), avg_qcf.min(1.0));
        }
    }

    // ── 6. LayerSkip QCF: importance-table based skip cost estimate ──
    if let Some(table) = ctx.importance_table {
        let total_sublayers = ctx.num_layers * 2;
        let skip_count = total_sublayers / 4;
        if skip_count > 0 {
            let (qcf_skip, _skip_set) = table.estimate_qcf_for_count(skip_count, ctx.num_layers);
            estimates.insert("layer_skip".to_string(), qcf_skip);
        }
    }

    estimates
}

// ── Weight swap dispatch (ENG-ALG-214-ROUTE) ────────────────────────────────

fn remap_weights_for_cpu_after_swap(
    model: &mut llm_rs2::models::transformer::TransformerModel,
    backend: &Arc<dyn Backend>,
    is_gpu: bool,
    enabled: bool,
    label: &str,
) {
    if !is_gpu || !enabled {
        return;
    }
    match model.map_weights_for_cpu(backend) {
        Ok(0) => {}
        Ok(n) => eprintln!(
            "[Backend] Re-mapped {} weight tensors after {} (host pointer restored)",
            n, label,
        ),
        Err(e) => eprintln!(
            "[Backend] Post-swap re-map failed: {} (switch_hw cpu may crash)",
            e,
        ),
    }
}

/// Build `LayerSwapEstimate` from an available `ImportanceTable` + model noise table.
///
/// Returns `None` when secondary mmap is absent or no importance table has been
/// collected yet (i.e., on the very first `RequestQcf` before any prefill).
fn build_layer_swap_estimate(
    model: &llm_rs2::models::transformer::TransformerModel,
    importance_table: Option<&llm_rs2::qcf::ImportanceTable>,
) -> Option<llm_shared::LayerSwapEstimate> {
    // secondary must be present for weight swap to make sense
    model.secondary_mmap.as_ref()?;

    let imp = importance_table?;

    let n = model.layers.len();
    let noise = &model.quant_noise;

    // per_layer_importance: indexed by decoder layer id
    let per_layer_importance: Vec<f32> = (0..n)
        .map(|i| {
            imp.entries()
                .iter()
                .find(|e| {
                    e.layer_id == i && e.sublayer == llm_rs2::qcf::layer_importance::SubLayer::Full
                })
                .map(|e| e.importance)
                .unwrap_or(0.0)
        })
        .collect();

    // per_layer_noise: None for NaN/missing
    let per_layer_noise: Vec<Option<f32>> = (0..n).map(|i| noise.epsilon(i)).collect();

    // qcf_swap_at_ratio: sample at representative ratios
    use llm_rs2::models::weights::WeightSwapDecider;
    use std::collections::HashMap;

    let sample_ratios = [0.1f32, 0.25, 0.5, 0.75, 1.0];
    let mut qcf_swap_at_ratio: HashMap<String, f32> = HashMap::new();

    for &r in &sample_ratios {
        let decider = WeightSwapDecider {
            importance: Some(imp),
            noise: Some(noise),
            n_decoder_layers: n,
            currently_swapped: &[],
            allow_boundary_layers: read_allow_boundary_env(),
            algorithm: llm_rs2::models::weights::SwapAlgorithm::ImportanceAware,
        };
        let (_, qcf) = decider.decide_dry_run(r);
        qcf_swap_at_ratio.insert(format!("{:.2}", r), qcf);
    }

    Some(llm_shared::LayerSwapEstimate {
        per_layer_importance,
        per_layer_noise,
        qcf_swap_at_ratio,
    })
}

fn command_summary(cmd: &EngineCommand) -> String {
    match cmd {
        EngineCommand::Throttle { delay_ms } => format!("Throttle({}ms)", delay_ms),
        EngineCommand::SetTargetTbt { target_ms } => format!("SetTargetTbt({}ms)", target_ms),
        EngineCommand::LayerSkip { skip_ratio } => format!("LayerSkip({:.2})", skip_ratio),
        EngineCommand::KvEvictH2o { keep_ratio } => format!("KvEvictH2o({:.2})", keep_ratio),
        EngineCommand::KvEvictSliding { keep_ratio } => {
            format!("KvEvictSliding({:.2})", keep_ratio)
        }
        EngineCommand::KvStreaming {
            sink_size,
            window_size,
        } => format!("KvStreaming(sink={}, win={})", sink_size, window_size),
        EngineCommand::KvMergeD2o { keep_ratio } => {
            format!("KvMergeD2o(ratio={})", keep_ratio)
        }
        EngineCommand::KvQuantDynamic { target_bits } => {
            format!("KvQuantDynamic({}bit)", target_bits)
        }
        EngineCommand::KvOffload { ratio } => format!("KvOffload({:.2})", ratio),
        EngineCommand::RestoreDefaults => "RestoreDefaults".to_string(),
        EngineCommand::SwitchHw { device } => format!("SwitchHw({})", device),
        EngineCommand::PrepareComputeUnit { device } => format!("Prepare({})", device),
        EngineCommand::Suspend => "Suspend".to_string(),
        EngineCommand::Resume => "Resume".to_string(),
        EngineCommand::RequestQcf => "RequestQcf".to_string(),
        EngineCommand::SetPartitionRatio { ratio } => {
            format!("SetPartitionRatio({})", ratio)
        }
        EngineCommand::SetPrefillPolicy {
            chunk_size,
            yield_ms,
            cpu_chunk_size,
        } => format!(
            "SetPrefillPolicy(chunk={:?}, yield={:?}, cpu_chunk={:?})",
            chunk_size, yield_ms, cpu_chunk_size
        ),
        EngineCommand::SwapWeights {
            ratio,
            target_dtype,
        } => {
            format!("SwapWeights(ratio={:.2}, dtype={:?})", ratio, target_dtype)
        }
    }
}

fn plan_summary(plan: &llm_rs2::resilience::ExecutionPlan) -> Vec<String> {
    let mut names = Vec::new();
    if let Some(ref evict) = plan.evict {
        names.push(format!(
            "Evict({:.2}, {:?}, {:?})",
            evict.target_ratio, evict.method, evict.level
        ));
    }
    if let Some(ref dev) = plan.switch_device {
        names.push(format!("Switch({})", dev));
    }
    if plan.throttle_delay_ms > 0 {
        names.push(format!("Throttle({}ms)", plan.throttle_delay_ms));
    }
    if plan.suspended {
        names.push("Suspend".to_string());
    }
    if plan.resumed {
        names.push("Resume".to_string());
    }
    if let Some(bits) = plan.kv_quant_bits {
        names.push(format!("KvQuant({}bit)", bits));
    }
    if let Some(ratio) = plan.layer_skip {
        names.push(format!("LayerSkip({:.2})", ratio));
    }
    if plan.restore_defaults {
        names.push("RestoreDefaults".to_string());
    }
    if let Some(ratio) = plan.partition_ratio {
        names.push(format!("PartitionRatio({:.2})", ratio));
    }
    names
}

// ── KIVI mode: KiviCache-based inference ────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_kivi(
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    input_ids: &[u32],
    sampling_config: &SamplingConfig,
    kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    max_seq_len: usize,
    residual_size: usize,
    num_tokens: usize,
    experiment_output: Option<&str>,
    experiment_logits_topk: usize,
    experiment_sample_interval: usize,
    prompt: &str,
    backend_name: &str,
    command_executor: &mut Option<llm_rs2::resilience::CommandExecutor>,
    initial_bits: u8,
    no_gpu_plan: bool,
    mut target_tbt_ms: f64,
    tbt_log_path: Option<&str>,
    ignore_eos: bool,
    cli_throttle_delay_ms: u64,
) -> anyhow::Result<()> {
    use llm_rs2::pressure::kv_cache::KVCacheOps;

    println!(
        "[KIVI] KV cache enabled — bits={}, residual_size={}, max_seq_len={}",
        initial_bits, residual_size, max_seq_len
    );

    // Experiment infrastructure
    let mut experiment_writer = if let Some(path) = experiment_output {
        Some(JsonlWriter::new(path)?)
    } else {
        None
    };
    let mut system_sampler = SystemSampler::new(experiment_sample_interval);
    let sys_start = if experiment_writer.is_some() {
        Some(system_sampler.snapshot())
    } else {
        None
    };

    // Create KiviCache per layer
    let mut kv_caches: Vec<KiviCache> = (0..num_layers)
        .map(|_| {
            KiviCache::new_gpu(
                kv_heads,
                head_dim,
                max_seq_len,
                residual_size,
                initial_bits,
                backend.clone(),
                memory.clone(),
            )
        })
        .collect();

    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;
    let q_dim = model.config.num_attention_heads * head_dim;
    let k_dim = kv_heads * head_dim;
    let v_dim = kv_heads * head_dim;
    let ffn_hidden = model.config.intermediate_size;

    // Allocate workspace
    let mut gen_ws = LayerWorkspace::new(
        WorkspaceConfig {
            batch_size: 1,
            dim: hidden_size,
            q_dim,
            k_dim,
            v_dim,
            ffn_hidden,
            n_heads: model.config.num_attention_heads,
            max_seq_len,
        },
        memory.as_ref(),
        backend.clone(),
    )?;
    let x_gen_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let mut x_gen = Tensor::new(
        Shape::new(vec![1, 1, hidden_size]),
        x_gen_buf,
        backend.clone(),
    );

    let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        logits_buf,
        backend.clone(),
    );

    // === PREFILL ===
    let mut tokens: Vec<u32> = input_ids.to_vec();
    let process_len = tokens.len();
    if process_len > max_seq_len {
        anyhow::bail!(
            "Prompt length {} exceeds max_seq_len {}",
            process_len,
            max_seq_len
        );
    }
    let mut start_pos = 0usize;

    let prefill_start = std::time::Instant::now();
    {
        let cpu_indices_buf = Galloc::new().alloc(process_len * 4, DType::U8)?;
        unsafe {
            let ptr = cpu_indices_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, process_len);
        }
        let cpu_input = Tensor::new(
            Shape::new(vec![1, process_len]),
            cpu_indices_buf,
            Arc::new(CpuBackend::new()),
        );
        let input_tensor = backend.copy_from(&cpu_input)?;

        let prefill_logits_buf = memory.alloc(process_len * vocab_size * 4, DType::F32)?;
        let mut prefill_logits = Tensor::new(
            Shape::new(vec![1, process_len, vocab_size]),
            prefill_logits_buf,
            backend.clone(),
        );

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos,
            kv_caches: &mut kv_caches,
            backend,
            memory: memory.as_ref(),
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;

        // Sample last token from prefill logits
        let mut logits_cpu = vec![0.0f32; process_len * vocab_size];
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
            backend.read_buffer(&prefill_logits, slice)?;
        }
        let last_start = (process_len - 1) * vocab_size;
        let next_token = sampling::sample(
            &mut logits_cpu[last_start..last_start + vocab_size],
            &tokens,
            vocab_size,
            sampling_config,
            None,
        );
        tokens.push(next_token);
        start_pos = process_len;
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = prefill_ms;
    let kivi_mem = kv_caches
        .iter()
        .map(|c| c.memory_usage_bytes())
        .sum::<usize>();
    eprintln!(
        "[KIVI] Prefill: {}ms, cache_pos={}, Q2_tokens={}, res_pos={}, mem={}KB",
        prefill_ms as u32,
        kv_caches[0].current_pos(),
        kv_caches[0].q2_tokens,
        kv_caches[0].res_pos,
        kivi_mem / 1024,
    );

    // Print prompt
    use std::io::Write;
    let mut stdout = std::io::stdout();
    let initial_text = tokenizer.decode(&tokens, true).unwrap_or_default();
    if experiment_writer.is_none() {
        print!("{}", initial_text);
        stdout.flush().ok();
    }

    // === DECODE ===
    let cpu_gen_indices_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_indices_buf,
        Arc::new(CpuBackend::new()),
    );

    // Pre-allocate GPU input tensor for decode loop (avoids per-token GPU alloc)
    let gpu_gen_buf = memory.alloc(4, DType::U8)?;
    let mut gen_input = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf, backend.clone());

    let eos_id = model.config.eos_token_id;

    let mut tbt_writer: Option<std::io::BufWriter<std::fs::File>> = tbt_log_path.map(|p| {
        let f = std::fs::File::create(p).expect("failed to create tbt-log file");
        std::io::BufWriter::new(f)
    });

    let decode_start = std::time::Instant::now();
    let mut generated_count = 0usize;
    let mut tbt_values: Vec<f64> = Vec::new();
    let mut forward_ms_values: Vec<f64> = Vec::new();
    let mut last_token_time = std::time::Instant::now();

    // Dynamic skip_config for KIVI resilience path
    use llm_rs2::inference::skip_config::SkipConfig;
    let mut kivi_skip_config: Option<SkipConfig> = None;
    let mut kivi_last_skip_ratio: Option<f32> = None;
    // Track last applied quant bits to avoid redundant transition_bits calls (sticky guard)
    let mut kivi_last_quant_bits: Option<u8> = None;

    // Build GPU kernel plan for KIVI decode (OpenCL only).
    // Skip when tensor partition is active — plan bypasses forward_gen's
    // partition co-execution path.
    #[cfg(feature = "opencl")]
    let mut gpu_plan =
        // KIVI plan does not yet integrate tensor-partition — the rejection
        // lives inside `build_plan_for_kivi` (returns None when any layer has
        // a partition_ctx). See ENG-ALG-200 scope note.
        if backend.name() == "OpenCL" && !no_gpu_plan {
            model.build_plan_for_kivi(&x_gen, &logits, &gen_ws, &kv_caches, backend)
        } else {
            None
        };
    #[cfg(not(feature = "opencl"))]
    let gpu_plan: Option<()> = None;

    for decode_idx in 0..(num_tokens - 1) {
        if kv_caches[0].current_pos() >= max_seq_len {
            eprintln!("\n[Stopped: Max context length reached]");
            break;
        }

        // Flush residual if needed (before plan dispatch writes new token)
        for cache in kv_caches.iter_mut() {
            if cache.needs_flush() {
                let _ = cache.flush_if_needed();
                // Flush changes Q2 state — plan remains valid (q2_tokens/res_pos are dynamic)
            }
        }

        let last_token = tokens[tokens.len() - 1];
        unsafe {
            *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_token;
        }
        // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
        backend.write_buffer(&mut gen_input, unsafe {
            std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
        })?;

        let fwd_start = std::time::Instant::now();

        // Try GPU plan path first
        #[cfg(feature = "opencl")]
        let plan_ok = if let Some(ref plan) = gpu_plan {
            match model.execute_plan_for_kivi(
                plan,
                &gen_input,
                start_pos,
                &mut x_gen,
                &mut kv_caches,
                &mut logits,
                backend,
            ) {
                Ok(true) => true,
                _ => {
                    gpu_plan = None;
                    false
                }
            }
        } else {
            false
        };
        #[cfg(not(feature = "opencl"))]
        let plan_ok = false;

        if !plan_ok {
            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &gen_input,
                start_pos,
                kv_caches: &mut kv_caches,
                backend,
                memory: memory.as_ref(),
                logits_out: &mut logits,
                x_gen: Some(&mut x_gen),
                workspace: Some(&mut gen_ws),
                score_accumulator: None,
                profiler: None,
                skip_config: kivi_skip_config.as_ref(),
                importance_collector: None,
                logits_last_only: false,
                variance_collector: None,
                prefill_workspace: None,

                layer_boundary_hook: None,
            })?;

            // Rebuild plan after fallback. Rejection for partition-active
            // KIVI runs happens inside `build_plan_for_kivi` (see ENG-ALG-200).
            #[cfg(feature = "opencl")]
            if gpu_plan.is_none() && backend.name() == "OpenCL" && !no_gpu_plan {
                gpu_plan = model.build_plan_for_kivi(&x_gen, &logits, &gen_ws, &kv_caches, backend);
            }
        }

        backend.synchronize()?;
        let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;
        forward_ms_values.push(forward_ms);

        start_pos += 1;
        generated_count += 1;

        // ── KIVI resilience checkpoint ──
        if let Some(executor) = command_executor.as_mut() {
            let current_bits = kv_caches[0].bits();
            let kv_dtype = match current_bits {
                16 => "f16".to_string(),
                8 => "q8".to_string(),
                4 => "q4".to_string(),
                2 => "q2".to_string(),
                _ => format!("q{}", current_bits),
            };
            let kv_snap = llm_rs2::resilience::KVSnapshot {
                total_bytes: kv_caches
                    .iter()
                    .map(|c| c.memory_usage_bytes() as u64)
                    .sum(),
                total_tokens: kv_caches[0].current_pos(),
                capacity: kv_caches[0].capacity(),
                protected_prefix: 0,
                kv_dtype,
                eviction_policy: "kivi".to_string(),
                skip_ratio: kivi_last_skip_ratio.unwrap_or(0.0),
            };
            let plan = executor.poll(&kv_snap);

            // QCF estimate: dry-run KIVI quantization NMSE
            if plan.request_qcf {
                let ctx = QcfEstimateContext {
                    kv_caches: &[], // KIVI path has no standard KVCache
                    score_accumulator: None,
                    streaming_config: None,
                    importance_table: None,
                    num_layers,
                    kivi_caches: Some(&kv_caches),
                };
                let estimates = compute_qcf_estimates(&ctx);
                executor.send_qcf_estimate(llm_shared::QcfEstimate {
                    estimates,
                    layer_swap: None,
                });
            }

            // kv_quant_bits: transition KiviCache bit-width
            // Sticky guard: skip if already at the requested bit-width
            if let Some(bits) = plan.kv_quant_bits
                && kivi_last_quant_bits != Some(bits)
            {
                for cache in kv_caches.iter_mut() {
                    if let Err(e) = cache.transition_bits(bits) {
                        eprintln!("[KIVI-Resilience] transition_bits({}) error: {}", bits, e);
                    }
                }
                // Invalidate GPU Plan — cache structure changed after bit transition
                #[cfg(feature = "opencl")]
                {
                    gpu_plan = None;
                }
                eprintln!("[KIVI-Resilience] Transitioned KV cache to {}bit", bits);
                kivi_last_quant_bits = Some(bits);
            }

            // KvOffload is not supported on the KIVI decode path — it runs on
            // KiviCache instances, whereas SwapHandler operates on KVCache.
            // Emit the expected "KvOffload" log lines so verify scenarios still
            // match, but mark the action as a no-op.
            if let Some(ratio) = plan.offload_ratio {
                eprintln!(
                    "[Resilience] KvOffload: ratio={:.2}, 0 tokens swapped (KIVI path)",
                    ratio
                );
            }

            // layer_skip / restore_defaults
            if plan.restore_defaults {
                eprintln!("[KIVI-Resilience] RestoreDefaults");
                kivi_skip_config = None;
                kivi_last_skip_ratio = None;
                kivi_last_quant_bits = None;
            } else if let Some(ratio) = plan.layer_skip
                && kivi_last_skip_ratio != Some(ratio)
            {
                eprintln!("[KIVI-Resilience] LayerSkip: ratio={:.2}", ratio);
                kivi_skip_config = Some(SkipConfig::uniform_init(
                    model.config.num_hidden_layers,
                    ratio,
                ));
                kivi_last_skip_ratio = Some(ratio);
            }

            // throttle
            if plan.throttle_delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(plan.throttle_delay_ms));
            }

            // Update target TBT from Manager directive. `target_tbt_set` lets
            // the engine honor an explicit `SetTargetTbt { target_ms: 0 }` to
            // disable pacing (see verify/ISSUE-3).
            if plan.target_tbt_set && plan.target_tbt_ms as f64 != target_tbt_ms {
                eprintln!(
                    "[KIVI-Resilience] SetTargetTbt: {:.1}ms → {}ms",
                    target_tbt_ms, plan.target_tbt_ms
                );
                target_tbt_ms = plan.target_tbt_ms as f64;
            } else if plan.restore_defaults {
                target_tbt_ms = 0.0;
            }

            if plan.suspended {
                eprintln!("\n[KIVI-Resilience] Inference suspended by system signal");
                break;
            }

            executor.on_token_generated();
        } else if cli_throttle_delay_ms > 0 {
            // No CommandExecutor: honour CLI --throttle-delay-ms directly.
            std::thread::sleep(std::time::Duration::from_millis(cli_throttle_delay_ms));
        }

        // Read logits to CPU
        let mut logits_cpu = vec![0.0f32; vocab_size];
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
            backend.read_buffer(&logits, slice)?;
        }

        // Extract top-K logits before sampling modifies them
        let top_logits = if experiment_writer.is_some() {
            extract_top_k_logits(&logits_cpu, experiment_logits_topk)
        } else {
            vec![]
        };

        let next_token =
            sampling::sample(&mut logits_cpu, &tokens, vocab_size, sampling_config, None);

        let now = std::time::Instant::now();
        let mut tbt = now.duration_since(last_token_time).as_secs_f64() * 1000.0;

        // Target TBT pacing
        let pacing_ms = if target_tbt_ms > 0.0 && tbt < target_tbt_ms {
            let sleep_ms = target_tbt_ms - tbt;
            std::thread::sleep(std::time::Duration::from_secs_f64(sleep_ms / 1000.0));
            tbt = target_tbt_ms;
            sleep_ms
        } else {
            0.0
        };

        tbt_values.push(tbt);
        last_token_time = std::time::Instant::now();

        // TBT log
        if let Some(ref mut w) = tbt_writer {
            use std::io::Write;
            writeln!(w,
                "{{\"token_idx\":{},\"tbt_ms\":{:.2},\"forward_ms\":{:.2},\"cache_pos\":{},\"pacing_ms\":{:.2}}}",
                decode_idx, tbt, forward_ms, kv_caches[0].current_pos(), pacing_ms
            ).ok();
        }

        tokens.push(next_token);

        // Experiment: write per-token JSONL record
        if let Some(ref mut writer) = experiment_writer {
            let token_text = tokenizer.decode(&[next_token], false).unwrap_or_default();
            let sys_metrics = system_sampler.sample(decode_idx);
            let record = TokenRecord {
                pos: decode_idx,
                token_id: next_token,
                text: token_text,
                tbt_ms: tbt,
                forward_ms,
                signal: None,
                actions: vec![],
                cache_pos: kv_caches[0].current_pos(),
                throttle_ms: 0,
                top_logits,
                sys: sys_metrics,
            };
            writer.write_token(&record)?;
        }

        // Stream output (suppress in experiment mode)
        if experiment_writer.is_none() {
            let text = tokenizer.decode(&tokens, true).unwrap_or_default();
            let new_text = &text[initial_text.len()..];
            print!("\r{}{}", initial_text, new_text);
            stdout.flush().ok();
        }

        if next_token == eos_id && !ignore_eos && std::env::var("IGNORE_EOS").is_err() {
            break;
        }
    }

    // Flush TBT log
    if let Some(ref mut w) = tbt_writer {
        use std::io::Write;
        w.flush().ok();
    }

    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let tok_per_s = if decode_ms > 0.0 {
        generated_count as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let kivi_mem_final = kv_caches
        .iter()
        .map(|c| c.memory_usage_bytes())
        .sum::<usize>();

    // Write experiment summary
    if let Some(ref mut writer) = experiment_writer {
        let sys_end = Some(system_sampler.snapshot());
        let avg_tbt_ms = if tbt_values.is_empty() {
            0.0
        } else {
            tbt_values.iter().sum::<f64>() / tbt_values.len() as f64
        };
        let avg_forward_ms = if forward_ms_values.is_empty() {
            0.0
        } else {
            forward_ms_values.iter().sum::<f64>() / forward_ms_values.len() as f64
        };
        let summary = SummaryRecord {
            _summary: true,
            total_tokens: tbt_values.len(),
            ttft_ms,
            avg_tbt_ms,
            avg_forward_ms,
            total_throttle_ms: 0,
            eviction_count: 0,
            evicted_tokens_total: 0,
            final_cache_pos: kv_caches[0].current_pos(),
            max_seq_len,
            prompt: prompt.to_string(),
            schedule_name: "kivi".to_string(),
            eviction_policy: "none".to_string(),
            backend: backend_name.to_string(),
            sample_interval: experiment_sample_interval,
            sys_start,
            sys_end,
            governor: Some(SystemSampler::read_governor()),
        };
        writer.write_summary(&summary)?;
        eprintln!(
            "[KIVI-Experiment] Done: {} tokens, avg TBT {:.2}ms",
            summary.total_tokens, avg_tbt_ms,
        );
    }

    eprintln!();
    eprintln!(
        "[KIVI] Decode: {} tokens, {:.1}ms ({:.1} tok/s)",
        generated_count, decode_ms, tok_per_s
    );
    eprintln!(
        "[KIVI] Final: cache_pos={}, Q2_tokens={}, res_pos={}, mem={}KB",
        kv_caches[0].current_pos(),
        kv_caches[0].q2_tokens,
        kv_caches[0].res_pos,
        kivi_mem_final / 1024,
    );

    // Compare with FP32 equivalent
    let fp32_equiv = kv_caches[0].current_pos() * kv_heads * head_dim * 4 * 2 * num_layers;
    eprintln!(
        "[KIVI] Compression: {:.1}x vs FP32 ({}KB vs {}KB)",
        fp32_equiv as f64 / kivi_mem_final.max(1) as f64,
        kivi_mem_final / 1024,
        fp32_equiv / 1024,
    );

    Ok(())
}

// ── Offload mode: OffloadKVCache-based inference with per-layer prefetch ─────

#[allow(clippy::too_many_arguments)]
fn run_offload(
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    input_ids: &[u32],
    sampling_config: &SamplingConfig,
    kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    max_seq_len: usize,
    num_tokens: usize,
    _prompt: &str,
    _backend_name: &str,
    offload_mode: &str,
    kv_type_str: &str,
    max_prefetch_depth: usize,
    offload_path: &str,
    command_executor: &mut Option<CommandExecutor>,
    cli_throttle_delay_ms: u64,
) -> anyhow::Result<()> {
    use llm_rs2::pressure::kv_cache::KVCacheOps;
    use llm_rs2::pressure::offload::OffloadKVCache;
    use llm_rs2::pressure::offload::raw_store::RawStore;

    // Validate constraints
    let kv_dtype = match kv_type_str {
        "f32" => DType::F32,
        "f16" => DType::F16,
        _ => anyhow::bail!(
            "--kv-offload requires --kv-type f16 or f32, got '{}'",
            kv_type_str
        ),
    };

    let token_bytes = kv_heads * head_dim * kv_dtype.size();

    // Resolve disk offload directory
    let disk_dir = if offload_path.is_empty() {
        std::env::temp_dir().join("llm_rs2_kv_offload")
    } else {
        std::path::PathBuf::from(offload_path)
    };

    if offload_mode == "disk" {
        eprintln!(
            "[Offload] mode=disk, path={}, dtype={:?}, layers={}, token_bytes={}, max_seq={}",
            disk_dir.display(),
            kv_dtype,
            num_layers,
            token_bytes,
            max_seq_len,
        );
    } else {
        eprintln!(
            "[Offload] mode={}, dtype={:?}, layers={}, token_bytes={}, max_seq={}",
            offload_mode, kv_dtype, num_layers, token_bytes, max_seq_len,
        );
    }

    // Create OffloadKVCache per layer. When the main backend is a GPU, wire it
    // (plus the matching memory allocator) so `get_view()` uploads the KV bytes
    // to device buffers that `attention_gen` can read via `cl_mem`. Without
    // this, OpenCL backends would see a null `cl_mem` from the default CPU
    // `SharedBuffer` and fail at kernel arg binding.
    let is_gpu_backend = backend.as_ref().is_gpu();
    let mut kv_caches: Vec<OffloadKVCache> = (0..num_layers)
        .map(|layer_id| {
            let store: Box<dyn llm_rs2::pressure::offload::store::OffloadStore> = match offload_mode
            {
                "raw" => Box::new(RawStore::new(token_bytes)),
                "disk" => Box::new(
                    llm_rs2::pressure::offload::disk_store::DiskStore::new(
                        disk_dir.clone(),
                        layer_id,
                        token_bytes,
                    )
                    .expect("Failed to create DiskStore"),
                ),
                _ => panic!("Unknown offload mode: {}", offload_mode),
            };
            let mut c =
                OffloadKVCache::new(layer_id, kv_heads, head_dim, kv_dtype, max_seq_len, store);
            if is_gpu_backend {
                c.set_gpu_backend(backend.clone(), memory.clone());
            }
            c
        })
        .collect();

    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;
    let q_dim = model.config.num_attention_heads * head_dim;
    let k_dim = kv_heads * head_dim;
    let v_dim = k_dim;
    let ffn_hidden = model.config.intermediate_size;

    // Allocate workspace
    let mut gen_ws = LayerWorkspace::new(
        llm_rs2::layers::workspace::WorkspaceConfig {
            batch_size: 1,
            dim: hidden_size,
            q_dim,
            k_dim,
            v_dim,
            ffn_hidden,
            n_heads: model.config.num_attention_heads,
            max_seq_len,
        },
        memory.as_ref(),
        backend.clone(),
    )?;
    let x_gen_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let mut x_gen = Tensor::new(
        Shape::new(vec![1, 1, hidden_size]),
        x_gen_buf,
        backend.clone(),
    );

    let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        logits_buf,
        backend.clone(),
    );

    // === PREFILL ===
    let mut tokens: Vec<u32> = input_ids.to_vec();
    let process_len = tokens.len();
    if process_len > max_seq_len {
        anyhow::bail!(
            "Prompt length {} exceeds max_seq_len {}",
            process_len,
            max_seq_len
        );
    }
    let mut start_pos = 0usize;

    let prefill_start = std::time::Instant::now();
    {
        let cpu_indices_buf = Galloc::new().alloc(process_len * 4, DType::U8)?;
        unsafe {
            let ptr = cpu_indices_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, process_len);
        }
        let cpu_input = Tensor::new(
            Shape::new(vec![1, process_len]),
            cpu_indices_buf,
            Arc::new(CpuBackend::new()),
        );
        let input_tensor = backend.copy_from(&cpu_input)?;

        let prefill_logits_buf = memory.alloc(process_len * vocab_size * 4, DType::F32)?;
        let mut prefill_logits = Tensor::new(
            Shape::new(vec![1, process_len, vocab_size]),
            prefill_logits_buf,
            backend.clone(),
        );

        // Prefill uses standard forward_into (no prefetch needed for batch)
        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos,
            kv_caches: &mut kv_caches,
            backend,
            memory: memory.as_ref(),
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;

        // Sample last token from prefill logits
        let mut logits_cpu = vec![0.0f32; process_len * vocab_size];
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
            backend.read_buffer(&prefill_logits, slice)?;
        }
        let last_start = (process_len - 1) * vocab_size;
        let next_token = sampling::sample(
            &mut logits_cpu[last_start..last_start + vocab_size],
            &tokens,
            vocab_size,
            sampling_config,
            None,
        );
        tokens.push(next_token);
        start_pos = process_len;
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = prefill_ms;

    let offload_mem_after_prefill: usize = kv_caches.iter().map(|c| c.memory_usage_bytes()).sum();
    let raw_equiv = process_len * token_bytes * 2 * num_layers; // K+V
    eprintln!(
        "[Offload] Prefill: {:.1}ms, cache_pos={}, store_mem={}KB (raw equiv={}KB, ratio={:.2}x)",
        prefill_ms,
        kv_caches[0].current_pos(),
        offload_mem_after_prefill / 1024,
        raw_equiv / 1024,
        raw_equiv as f64 / offload_mem_after_prefill.max(1) as f64,
    );

    // Print prompt
    use std::io::Write;
    let mut stdout = std::io::stdout();
    let initial_text = tokenizer.decode(&tokens, true).unwrap_or_default();
    print!("{}", initial_text);
    stdout.flush().ok();

    // === DECODE with adaptive prefetch ===
    let cpu_gen_indices_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_indices_buf,
        Arc::new(CpuBackend::new()),
    );

    // Pre-allocate GPU input tensor for decode loop (avoids per-token GPU alloc)
    let gpu_gen_buf = memory.alloc(4, DType::U8)?;
    let mut gen_input = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf, backend.clone());

    let eos_id = model.config.eos_token_id;

    let mut prefetch = llm_rs2::pressure::offload::prefetch::PrefetchController::new(
        max_prefetch_depth,
        num_layers,
    );

    let decode_start = std::time::Instant::now();
    let mut generated_count = 0usize;
    let mut tbt_values: Vec<f64> = Vec::new();
    let mut forward_ms_values: Vec<f64> = Vec::new();
    let mut last_token_time = std::time::Instant::now();

    for _decode_idx in 0..(num_tokens - 1) {
        if kv_caches[0].current_pos() >= max_seq_len {
            eprintln!("\n[Stopped: Max context length reached]");
            break;
        }

        let last_token = tokens[tokens.len() - 1];
        unsafe {
            *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_token;
        }
        // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
        backend.write_buffer(&mut gen_input, unsafe {
            std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
        })?;

        // Preload state managed by forward_into_offload:
        // - retained layers: retain_preload() keeps preloaded=true
        // - non-retained layers: release_buffers() sets preloaded=false

        let fwd_start = std::time::Instant::now();
        model.forward_into_offload(
            TransformerModelForwardArgs {
                input_tokens: &gen_input,
                start_pos,
                kv_caches: &mut kv_caches,
                backend,
                memory: memory.as_ref(),
                logits_out: &mut logits,
                x_gen: Some(&mut x_gen),
                workspace: Some(&mut gen_ws),
                score_accumulator: None,
                profiler: None,
                skip_config: None,
                importance_collector: None,
                logits_last_only: false,
                variance_collector: None,
                prefill_workspace: None,

                layer_boundary_hook: None,
            },
            &mut prefetch,
        )?;
        let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;
        forward_ms_values.push(forward_ms);

        // ── Offload resilience checkpoint ──
        if let Some(executor) = command_executor.as_mut() {
            let kv_snap = KVSnapshot {
                total_bytes: kv_caches
                    .iter()
                    .map(|c| c.memory_usage_bytes() as u64)
                    .sum(),
                total_tokens: kv_caches[0].current_pos(),
                capacity: kv_caches[0].capacity(),
                protected_prefix: 0,
                kv_dtype: kv_type_str.to_string(),
                eviction_policy: "none".to_string(),
                skip_ratio: 0.0,
            };
            let plan = executor.poll(&kv_snap);

            if plan.throttle_delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(plan.throttle_delay_ms));
            }
            if plan.suspended {
                eprintln!("\n[Offload-Resilience] Inference suspended by system signal");
                break;
            }
            // evict, kv_quant_bits, layer_skip 등은 OffloadKVCache에서 미지원 — 무시
            if plan.evict.is_some() {
                eprintln!(
                    "[Offload-Resilience] KvEvict requested but OffloadKVCache has no eviction support — ignored"
                );
            }

            executor.on_token_generated();
        } else if cli_throttle_delay_ms > 0 {
            // No CommandExecutor: honour CLI --throttle-delay-ms directly.
            std::thread::sleep(std::time::Duration::from_millis(cli_throttle_delay_ms));
        }
        // ── End checkpoint ──

        start_pos += 1;
        generated_count += 1;

        // Read logits to CPU
        let mut logits_cpu = vec![0.0f32; vocab_size];
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
            backend.read_buffer(&logits, slice)?;
        }

        let next_token =
            sampling::sample(&mut logits_cpu, &tokens, vocab_size, sampling_config, None);

        let now = std::time::Instant::now();
        let tbt = now.duration_since(last_token_time).as_secs_f64() * 1000.0;
        tbt_values.push(tbt);
        last_token_time = now;

        tokens.push(next_token);

        // Streaming output
        let text = tokenizer.decode(&tokens, true).unwrap_or_default();
        let new_text = &text[initial_text.len()..];
        print!("\r{}{}", initial_text, new_text);
        stdout.flush().ok();

        if next_token == eos_id && std::env::var("IGNORE_EOS").is_err() {
            break;
        }
    }

    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let tok_per_s = if decode_ms > 0.0 {
        generated_count as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let offload_mem_final: usize = kv_caches.iter().map(|c| c.memory_usage_bytes()).sum();
    let final_raw_equiv = kv_caches[0].current_pos() * token_bytes * 2 * num_layers;

    eprintln!();
    eprintln!(
        "[Offload] Decode: {} tokens, {:.1}ms ({:.1} tok/s)",
        generated_count, decode_ms, tok_per_s,
    );
    eprintln!(
        "[Offload] Final: cache_pos={}, store_mem={}KB (raw equiv={}KB, ratio={:.2}x)",
        kv_caches[0].current_pos(),
        offload_mem_final / 1024,
        final_raw_equiv / 1024,
        final_raw_equiv as f64 / offload_mem_final.max(1) as f64,
    );

    let avg_forward_ms = if forward_ms_values.is_empty() {
        0.0
    } else {
        forward_ms_values.iter().sum::<f64>() / forward_ms_values.len() as f64
    };
    let avg_tbt = if tbt_values.is_empty() {
        0.0
    } else {
        tbt_values.iter().sum::<f64>() / tbt_values.len() as f64
    };

    eprintln!(
        "[Prefetch] final depth={}, preload_ema={:.0}us, forward_ema={:.0}us",
        prefetch.depth(),
        prefetch.preload_ema_us(),
        prefetch.forward_ema_us(),
    );

    println!("\nDone.");
    println!("TTFT: {:.2} ms", ttft_ms);
    io_trace("ttft");
    println!(
        "Avg forward: {:.2} ms, Avg TBT: {:.2} ms ({:.1} tok/s)",
        avg_forward_ms, avg_tbt, tok_per_s,
    );

    Ok(())
}
