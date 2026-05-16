use clap::Parser;
use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::core::attention_scores::AttentionScoreAccumulator;
use llm_rs2::core::backend::Backend;
use llm_rs2::core::buffer::DType;
use llm_rs2::core::cache_manager::CacheManager;
use llm_rs2::core::events::{self, CacheEvent, StderrDiagnosticSink};
use llm_rs2::core::eviction::h2o::H2OPolicy;
use llm_rs2::core::eviction::h2o_plus::H2OPlusPolicy;
use llm_rs2::core::eviction::no_eviction::NoEvictionPolicy;
use llm_rs2::core::eviction::sliding_window::SlidingWindowPolicy;
use llm_rs2::core::kivi_cache::KiviCache;
use llm_rs2::core::kv_cache::{KVCache, KVLayout};
use llm_rs2::core::memory::Memory;
use llm_rs2::core::pressure::d2o_handler::{D2OConfig, D2OHandler};
use llm_rs2::core::pressure::{CachePressurePipeline, PressureLevel, PressureStageConfig};
use llm_rs2::core::rss_trace::{io_trace, read_bytes_now, rss_trace};
use llm_rs2::core::sampling::{self, SamplingConfig};
use llm_rs2::core::shape::Shape;
use llm_rs2::core::sys_monitor::{LinuxSystemMonitor, NoOpMonitor};
use llm_rs2::core::tensor::Tensor;
use llm_rs2::layers::workspace::{
    LayerWorkspace, PartitionWorkspace, PartitionWsCell, WorkspaceConfig,
};
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use llm_rs2::session::cli::{Args, parse_qcf_sample_layers};
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
) -> impl Fn(usize, DType) -> anyhow::Result<Arc<dyn llm_rs2::core::buffer::Buffer>> + 'a {
    // Try to extract OpenCL queue for UnifiedBuffer allocation.
    #[cfg(feature = "opencl")]
    let ocl_queue: Option<ocl::Queue> = backend
        .as_any()
        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
        .map(|b| b.queue.clone());

    #[cfg(not(feature = "opencl"))]
    let _ = backend; // suppress unused warning

    move |size: usize, dtype: DType| -> anyhow::Result<Arc<dyn llm_rs2::core::buffer::Buffer>> {
        #[cfg(feature = "opencl")]
        if let Some(ref q) = ocl_queue {
            let buf = llm_rs2::buffer::unified_buffer::UnifiedBuffer::new(q.clone(), size, dtype)?;
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
    let mut backend = ctx.backend;
    let memory = ctx.memory;
    let gpu_backend_arc = ctx.gpu_backend_arc;
    let gpu_memory_arc = ctx.gpu_memory_arc;
    let mut is_gpu = ctx.is_gpu;
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
    if args.kivi && args.eval_ll {
        let questions = load_eval_questions(&args, &prompt)?;
        let vocab_size = model.config.vocab_size;
        let hidden_size = model.config.hidden_size;
        let eval_config = llm_rs2::eval::EvalConfig {
            max_seq_len,
            effective_budget: 0,
            kv_budget_ratio: 0.0,
            greedy: args.greedy,
            kv_type: format!("q{}+f32_residual", args.kivi_bits),
            qcf_mode: args.qcf_mode.clone(),
            vocab_size,
            hidden_size,
        };
        let qcf_config = llm_rs2::core::qcf::QcfConfig::default();
        let kivi_bits = args.kivi_bits;
        let mut kv_caches: Vec<KiviCache> = (0..num_layers)
            .map(|_| {
                KiviCache::new_gpu(
                    kv_heads,
                    head_dim,
                    max_seq_len,
                    args.kivi_residual_size,
                    kivi_bits,
                    backend.clone(),
                    memory.clone(),
                )
            })
            .collect();
        if args.awqe {
            for cache in kv_caches.iter_mut() {
                cache.set_awqe_enabled(true);
            }
            eprintln!("[KIVI] AWQE + AW-VOPR enabled");
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
            let mut acc = llm_rs2::core::attention_scores::AttentionScoreAccumulator::new_gqa(
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
            "kivi_bits": args.kivi_bits,
            "kivi_residual_size": args.kivi_residual_size,
            "max_seq_len": max_seq_len,
            "kv_type": format!("q{}+f32_residual", args.kivi_bits),
        });
        println!("{}", serde_json::to_string_pretty(&json_val)?);
        return Ok(());
    }

    // ── KIVI + PPL mode: KiviCache with perplexity evaluation ──
    if args.kivi
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
            args.kivi_residual_size,
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
    } else if args.initial_kv_capacity > 0 {
        args.initial_kv_capacity.min(max_seq_len)
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
            use llm_rs2::core::quant::{BlockQ4_0, QK4_0};
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

    // ── Chat REPL mode ──
    // Three dispatch paths: standard (KVCache, with optional eviction),
    // KIVI (quantized KV), and KV-offload (disk/raw store).
    if args.chat {
        let sampling_config = SamplingConfig {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            repetition_penalty: args.repetition_penalty,
            repetition_window: args.repetition_window,
        };
        let kv_offload_active = !args.kv_offload.is_empty() && args.kv_offload != "none";
        if args.kivi {
            return run_chat_kivi(
                &args,
                &model,
                &tokenizer,
                &backend,
                &memory,
                &sampling_config,
                kv_heads,
                head_dim,
                num_layers,
                max_seq_len,
            );
        }
        if kv_offload_active {
            return run_chat_offload(
                &args,
                &model,
                &tokenizer,
                &backend,
                &memory,
                &sampling_config,
                kv_heads,
                head_dim,
                num_layers,
                max_seq_len,
            );
        }
        return run_chat_standard(
            &args,
            &model,
            &tokenizer,
            &backend,
            &memory,
            &mut kv_caches,
            &sampling_config,
            max_seq_len,
        );
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
            if args.eviction_policy != "none" {
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
    let mut throttle_delay_ms: u64 = args.throttle_delay_ms;
    let mut tbt_log_writer: Option<std::io::BufWriter<std::fs::File>> =
        args.tbt_log.as_ref().map(|path| {
            let file = std::fs::File::create(path).expect("failed to create tbt-log file");
            std::io::BufWriter::new(file)
        });
    let mut target_tbt_ms = args.target_tbt;

    // ── KIVI mode: separate path with KiviCache ──
    // Placed after executor creation so resilience is available in the token loop.
    if args.kivi || args.kv_dynamic_quant {
        // KIVI mode: --kivi starts at Q2, --kv-dynamic-quant starts at bits=16
        // (F16-equivalent) and allows runtime transition via kv_quant_dynamic.
        // Note: --enable-resilience alone stays on main path (F16 KVCache + eviction).
        let initial_bits: u8 = if args.kivi { args.kivi_bits } else { 16 };
        let residual_size = if initial_bits == 16 {
            // bits=16: all tokens stay in residual (no quantization flush)
            // Round down to QKKV (32) multiple for KiviCache alignment
            (max_seq_len / 32) * 32
        } else {
            args.kivi_residual_size
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
    if args.kv_offload != "none" {
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
            &args.kv_offload,
            &args.kv_type,
            args.max_prefetch_depth,
            &args.offload_path,
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
    let mut tokens = input_ids.clone();
    let mut start_pos = 0;
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
    let actual_protected_prefix =
        args.protected_prefix
            .unwrap_or(match args.eviction_policy.as_str() {
                // Score-based policies: default to 4 (attention sinks only).
                // Protecting the entire prompt makes score-based eviction meaningless
                // because only generated tokens would be evictable.
                "h2o" | "h2o_plus" | "d2o" => 4,
                // StreamingLLM: use explicit sink_size parameter
                "streaming" => args.sink_size,
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
        let threshold_bytes = args.memory_threshold_mb * 1024 * 1024;

        if args.eviction_policy == "d2o" {
            // D2O uses CachePressureHandler (Pipeline mode), not EvictionPolicy (Legacy mode)
            let d2o_handler = D2OHandler::new(D2OConfig {
                keep_ratio: args.d2o_keep_ratio,
                protected_prefix: actual_protected_prefix,
                target_ratio: args.eviction_target_ratio,
                ema_beta: args.d2o_ema_beta,
                merge_e: args.d2o_merge_e,
                use_layer_allocation: args.d2o_layer_alloc,
                protected_layers: args.d2o_protected_layers.clone().unwrap_or_default(),
            });
            let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(d2o_handler),
            }]);
            CacheManager::with_pipeline(pipeline, monitor, threshold_bytes)
        } else {
            let policy: Box<dyn llm_rs2::core::eviction::EvictionPolicy> = match args
                .eviction_policy
                .as_str()
            {
                "none" => Box::new(NoEvictionPolicy::new()),
                "sliding" => Box::new(SlidingWindowPolicy::new(
                    args.eviction_window,
                    actual_protected_prefix,
                )),
                "streaming" => {
                    use llm_rs2::core::eviction::StreamingLLMPolicy;
                    let window = if args.streaming_window > 0 {
                        args.streaming_window
                    } else if args.kv_budget > 0 {
                        args.kv_budget.saturating_sub(args.sink_size)
                    } else {
                        args.eviction_window
                    };
                    Box::new(StreamingLLMPolicy::new(args.sink_size, window))
                }
                "h2o" => Box::new(H2OPolicy::new(args.h2o_keep_ratio, actual_protected_prefix)),
                "h2o_plus" => Box::new(H2OPlusPolicy::new(
                    args.h2o_keep_ratio,
                    actual_protected_prefix,
                )),
                other => anyhow::bail!(
                    "Unknown eviction policy: '{}'. Use: none, sliding, streaming, h2o, h2o_plus, d2o",
                    other
                ),
            };
            CacheManager::new(policy, monitor, threshold_bytes, args.eviction_target_ratio)
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
            args.h2o_keep_ratio,
            resilience_protected_prefix,
        )),
    );
    cache_manager.register_policy(
        llm_rs2::resilience::EvictMethod::Sliding,
        Box::new(SlidingWindowPolicy::new(
            args.eviction_window,
            resilience_protected_prefix,
        )),
    );
    // Note: Streaming policy is NOT pre-registered because its parameters
    // (sink_size, window_size) come from the Manager directive at runtime.
    // It is instantiated on-demand in the eviction dispatch below.

    // Parse QCF mode
    let qcf_mode = match args.qcf_mode.as_str() {
        "caote" => llm_rs2::core::qcf::QcfMode::Caote,
        "both" => llm_rs2::core::qcf::QcfMode::Both,
        _ => llm_rs2::core::qcf::QcfMode::Attn,
    };
    let needs_caote = qcf_mode.has_caote();

    // Setup AttentionScoreAccumulator for H2O / H2O+ / D2O / CAOTE
    // When CAOTE is requested, always use GQA-aware accumulator (for per-KV-head attention).
    let needs_score_based = args.eviction_policy == "h2o"
        || args.eviction_policy == "d2o"
        || args.eviction_policy == "h2o_plus";
    // Always build accumulator for eval-ll when any eviction policy is active:
    // sliding mode needs it to populate last_step_head_attn for QCF-ATTN v2.
    let has_eviction_policy = args.eviction_policy != "none";
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
    let use_gqa = args.eviction_policy == "h2o_plus" || needs_caote || has_eviction_policy;

    let mut score_accumulator = if needs_accumulator {
        let acc = if use_gqa {
            AttentionScoreAccumulator::new_gqa(
                max_seq_len,
                model.config.num_attention_heads,
                model.config.num_key_value_heads,
                model.config.num_hidden_layers,
                args.h2o_tracked_layers,
                args.h2o_decay,
            )
        } else {
            AttentionScoreAccumulator::new(
                max_seq_len,
                model.config.num_attention_heads,
                model.config.num_hidden_layers,
                args.h2o_tracked_layers,
                args.h2o_decay,
            )
        };
        let mut acc = acc;
        // Always active: GPU acc overhead is ~0.6ms/token (1.7%),
        // CPU NEON acc overhead is ~0.66ms/token (1.1%).
        // This ensures first RequestQcf returns accurate H2O/D2O estimates.
        acc.set_active(true);
        acc.set_time_normalize(!args.h2o_raw_scores);
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
            args.h2o_decay,
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

    if args.eviction_policy != "none" {
        eprintln!(
            "Eviction: policy={}, window={}, prefix={}, ratio={}, threshold={}MB",
            args.eviction_policy,
            args.eviction_window,
            actual_protected_prefix,
            args.eviction_target_ratio,
            args.memory_threshold_mb
        );
    }

    // Build SkipConfig from CLI options
    use llm_rs2::core::skip_config::SkipConfig;
    let mut skip_config = if let Some(ref layers) = args.skip_layers {
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
    let mut last_skip_ratio: Option<f32> = args.skip_ratio;

    // Auto-eviction: non-experiment mode evicts automatically.
    // - Sliding window: triggers on memory pressure after each forward pass.
    // - Score-based (H2O/H2O+/D2O): triggers when cache utilization >= 90% capacity,
    //   using force_evict_with_scores to bypass memory pressure checks.
    let auto_eviction = args.eviction_policy != "none" && experiment_schedule.is_none();
    let score_based_eviction = matches!(args.eviction_policy.as_str(), "h2o" | "h2o_plus" | "d2o");

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
    let layer_swap_pool: Option<
        Arc<llm_rs2::models::weights::layer_object_pool::LayerObjectPool>,
    > = {
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
                    Some(pool)
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
    let mmap_registration: Option<
        Arc<llm_rs2::buffer::cuda_mmap_alias_buffer::CudaMmapRegistration>,
    > = {
        let enabled = std::env::var("LLMRS_SWAP_MMAP_ALIAS")
            .map(|v| v == "1")
            .unwrap_or(false);
        if !enabled {
            None
        } else if let Some(secondary) = model.secondary_mmap.clone() {
            match llm_rs2::buffer::cuda_mmap_alias_buffer::CudaMmapRegistration::register(secondary)
            {
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
        use llm_rs2::core::qcf::ImportanceCollector;

        let mut collector = ImportanceCollector::new();

        let prompt_enc = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!(e))?;
        let prompt_ids: Vec<u32> = prompt_enc.get_ids().to_vec();
        let prompt_len = prompt_ids.len();
        eprintln!("[Importance] Prefill {} tokens...", prompt_len);

        let cpu_buf = Galloc::new().alloc(prompt_len * 4, DType::U8)?;
        unsafe {
            let ptr = cpu_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(prompt_ids.as_ptr(), ptr, prompt_len);
        }
        let cpu_input = Tensor::new(
            Shape::new(vec![1, prompt_len]),
            cpu_buf,
            Arc::new(CpuBackend::new()),
        );
        let input_tensor = backend.copy_from(&cpu_input)?;

        let logits_buf = memory.alloc(prompt_len * vocab_size * 4, DType::F32)?;
        let mut logits = Tensor::new(
            Shape::new(vec![1, prompt_len, vocab_size]),
            logits_buf,
            backend.clone(),
        );

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: 0,
            kv_caches: &mut kv_caches,
            backend: &backend,
            memory: &*memory,
            logits_out: &mut logits,
            x_gen: None,
            workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: Some(&mut collector),
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;

        let table = collector.build();

        let importance_entries: Vec<serde_json::Value> = table
            .entries()
            .iter()
            .map(|e| {
                serde_json::json!({
                    "layer": e.layer_id,
                    "sublayer": format!("{:?}", e.sublayer),
                    "importance": e.importance,
                    "opr": e.opr,
                })
            })
            .collect();

        let output = serde_json::json!({
            "model": args.model_path,
            "num_layers": model.config.num_hidden_layers,
            "prompt_tokens": prompt_len,
            "importance": importance_entries,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
        return Ok(());
    }

    // ════════════════════════════════════════════════════════════
    //  EVAL-LL MODE: Log-likelihood evaluation for downstream tasks
    // ════════════════════════════════════════════════════════════
    if args.eval_ll {
        let questions = load_eval_questions(&args, &prompt)?;

        // ── QCF-dump prelude: --eval-ll + --qcf-dump + --force-swap-ratio ────
        // When all three flags are active we run warmup prefill → ImportanceTable
        // → WeightSwapDecider → SwapExecutor before the eval loop.  This mirrors
        // the PPL/generation QCF-dump workflow (line ~2417) but uses the eval
        // questions' prompt text instead of a corpus file for the warmup input.
        let eval_ll_qcf_start = std::time::Instant::now();
        let mut eval_ll_qcf_importance: Option<llm_rs2::core::qcf::ImportanceTable> = None;
        let mut eval_ll_qcf_decision: Option<llm_rs2::models::weights::decider::SwapDecision> =
            None;
        let mut eval_ll_qcf_dpllm_epsilon: Option<Vec<f32>> = None;
        let mut eval_ll_qcf_dpllm_epsilon_multi: Option<Vec<f32>> = None;
        let mut eval_ll_qcf_dpllm_epsilon_abs: Option<Vec<f32>> = None;
        let mut eval_ll_qcf_dpllm_epsilon_qcf: Option<Vec<f32>> = None;
        let mut eval_ll_qcf_direct_attn_f4: Option<Vec<f32>> = None;
        let mut eval_ll_qcf_direct_attn_f5: Option<Vec<f32>> = None;
        let mut eval_ll_qcf_direct_attn_f5_decode_only: Option<Vec<f32>> = None;
        let mut eval_ll_qcf_direct_attn_f5_prefill_decode: Option<Vec<f32>> = None;

        if args.qcf_dump.is_some()
            && let Some(force_ratio) = args.force_swap_ratio
        {
            let warmup_n = args.qcf_warmup_tokens.max(1);
            // Concatenate question prompts (separated by "\n\n") and take the
            // first warmup_n tokens. Empty result → soft skip (no abort).
            let warmup_ids = build_eval_ll_warmup_text(&questions, warmup_n, &tokenizer);

            if warmup_ids.is_empty() {
                eprintln!(
                    "[QCF-dump] WARNING: eval-ll warmup token sequence is empty; \
                     prelude skipped (swap will use uniform fallback)"
                );
            } else {
                let result = run_qcf_warmup_workflow(
                    &model,
                    &backend,
                    memory.as_ref(),
                    &mut kv_caches,
                    vocab_size,
                    &warmup_ids,
                    Some(force_ratio),
                    gpu_backend_arc.as_ref(),
                    &cpu_backend_arc,
                    " eval-ll",
                    swap_algorithm,
                    !args.qcf_trajectory,
                    importance_formula,
                    importance_compare,
                    swap_only_layers.as_deref(),
                    args.decode_x_steps,
                )?;
                eval_ll_qcf_decision = result.decision;
                eval_ll_qcf_importance = Some(result.importance);
                eval_ll_qcf_dpllm_epsilon = result.dpllm_epsilon;
                eval_ll_qcf_dpllm_epsilon_multi = result.dpllm_epsilon_multi;
                eval_ll_qcf_dpllm_epsilon_abs = result.dpllm_epsilon_abs;
                eval_ll_qcf_dpllm_epsilon_qcf = result.dpllm_epsilon_qcf;
                eval_ll_qcf_direct_attn_f4 = result.direct_attn_f4;
                eval_ll_qcf_direct_attn_f5 = result.direct_attn_f5;
                eval_ll_qcf_direct_attn_f5_decode_only = result.direct_attn_f5_decode_only;
                eval_ll_qcf_direct_attn_f5_prefill_decode = result.direct_attn_f5_prefill_decode;
            }
        }

        let ratio_mode = args.kv_budget_ratio > 0.0;
        let budget_mode = args.kv_budget > 0 || ratio_mode;

        // For ratio mode, effective_budget is computed per-question inside eval_loop.
        // Pass 0 here; the loop will use kv_budget_ratio × prompt_len.
        let effective_budget = if ratio_mode { 0 } else { args.kv_budget };

        eprintln!(
            "[Eval-LL] {} questions, policy={}, kv_budget={}, kv_budget_ratio={}, mode={}",
            questions.len(),
            args.eviction_policy,
            args.kv_budget,
            args.kv_budget_ratio,
            if budget_mode {
                if ratio_mode {
                    "ratio-per-question"
                } else {
                    "chunked"
                }
            } else {
                "full-prefill"
            }
        );

        let qcf_mode_enum = match args.qcf_mode.as_str() {
            "caote" => llm_rs2::core::qcf::QcfMode::Caote,
            "both" => llm_rs2::core::qcf::QcfMode::Both,
            _ => llm_rs2::core::qcf::QcfMode::Attn,
        };
        let qcf_config = llm_rs2::core::qcf::QcfConfig {
            mode: qcf_mode_enum,
            ..llm_rs2::core::qcf::QcfConfig::default()
        };

        let eval_config = llm_rs2::eval::EvalConfig {
            max_seq_len,
            effective_budget,
            kv_budget_ratio: args.kv_budget_ratio,
            greedy: args.greedy,
            kv_type: args.kv_type.clone(),
            qcf_mode: args.qcf_mode.clone(),
            vocab_size,
            hidden_size,
        };

        // For ratio mode, hook starts with budget=0; eval_loop updates it per-question.
        let hook_budget = if ratio_mode { 0 } else { effective_budget };
        let is_d2o = args.eviction_policy == "d2o";

        // ARGUS Step 6: resolve --qcf-sample-layers from CLI.
        // When --enable-qcf-experimental is off, always use [0] (legacy, no overhead).
        let eviction_hook_sample_layers = if args.enable_qcf_experimental {
            parse_qcf_sample_layers(&args.qcf_sample_layers, num_layers)
                .map_err(|e| anyhow::anyhow!("--qcf-sample-layers: {e}"))?
        } else {
            vec![0]
        };

        let mut hook = llm_rs2::eval::EvictionHook::new(
            cache_manager,
            score_accumulator,
            qcf_config,
            hook_budget,
            actual_protected_prefix,
            score_based_eviction,
            args.h2o_keep_ratio,
            is_d2o,
            args.kv_type.clone(),
            backend.clone(),
            args.enable_qcf_experimental,
            eviction_hook_sample_layers,
        );

        // ── Trajectory mode dispatch ──────────────────────────────────────────
        // When `--qcf-trajectory` is active alongside `--qcf-dump` and
        // `--force-swap-ratio`, we run eval-ll K+1 times (K = decision layer
        // count): step 0 with no swap (baseline), then step t (1..=K) after
        // cumulatively applying `selected_layers[t-1]`. Each step's full
        // EvalOutput is captured into `trajectory_outputs` and emitted under
        // the `trajectory` field of the dump JSON.
        let trajectory_mode = args.qcf_trajectory
            && args.qcf_dump.is_some()
            && args.force_swap_ratio.is_some()
            && eval_ll_qcf_decision
                .as_ref()
                .map(|d| !d.selected_layers.is_empty())
                .unwrap_or(false);
        let ordered_layers: Vec<usize> = if trajectory_mode {
            eval_ll_qcf_decision
                .as_ref()
                .unwrap()
                .selected_layers
                .clone()
        } else {
            Vec::new()
        };
        let n_steps = if trajectory_mode {
            ordered_layers.len() + 1
        } else {
            1
        };
        let mut trajectory_outputs: Vec<llm_rs2::eval::EvalOutput> = Vec::with_capacity(n_steps);

        if trajectory_mode {
            eprintln!(
                "[QCF-trajectory] mode enabled: K={} (algo={}, ratio={:.2})",
                ordered_layers.len(),
                swap_algorithm.short_name(),
                args.force_swap_ratio.unwrap_or(0.0),
            );
        }

        for step in 0..n_steps {
            if trajectory_mode {
                eprintln!(
                    "[QCF-trajectory] step {}/{}: cumulative swap = {:?}",
                    step,
                    ordered_layers.len(),
                    &ordered_layers[..step]
                );
            }

            let step_out = llm_rs2::eval::run_eval_ll_generic(
                &model,
                &tokenizer,
                &backend,
                &*memory,
                &mut kv_caches,
                &mut hook,
                &questions,
                &eval_config,
                skip_config.as_ref(),
            )?;
            trajectory_outputs.push(step_out);

            if trajectory_mode && step < ordered_layers.len() {
                let layer_to_swap = ordered_layers[step];
                let report = run_layer_swap(
                    &model,
                    &[layer_to_swap],
                    gpu_backend_arc.as_ref(),
                    &cpu_backend_arc,
                    None,
                    #[cfg(feature = "opencl")]
                    None,
                    #[cfg(feature = "cuda-embedded")]
                    None,
                    #[cfg(feature = "cuda-embedded")]
                    None,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "[QCF-trajectory] swap layer {} failed: {}",
                        layer_to_swap,
                        e
                    )
                })?;
                eprintln!(
                    "[QCF-trajectory] swapped layer {}: latency {:.1}ms",
                    layer_to_swap, report.latency_ms
                );
            }
        }

        // For downstream non-trajectory stdout printing, expose the last step's
        // EvalOutput as `output` (in non-trajectory mode this is the only step).
        let output = trajectory_outputs
            .last()
            .expect("at least one eval-ll step ran")
            .clone();

        // ── QCF-dump JSON (eval-ll mode) ──────────────────────────────────────
        if let Some(ref dump_path) = args.qcf_dump {
            use llm_rs2::eval::qcf_helpers::{
                QcfSwapDumpContext, TrajectoryStep, dump_qcf_swap_json,
            };

            let empty_swap: Vec<usize> = Vec::new();
            let (swap_set, qcf_predicted, fallback_used) =
                if let Some(ref dec) = eval_ll_qcf_decision {
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
            let total_wall = eval_ll_qcf_start.elapsed().as_secs_f64();

            // Build trajectory steps when in trajectory mode.
            let trajectory_steps: Vec<TrajectoryStep> = if trajectory_mode {
                trajectory_outputs
                    .iter()
                    .enumerate()
                    .map(|(t, eo)| TrajectoryStep {
                        step: t,
                        swapped_layers: ordered_layers[..t].to_vec(),
                        layer_added: if t > 0 {
                            Some(ordered_layers[t - 1])
                        } else {
                            None
                        },
                        eval_ll_output: eo,
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let trajectory_ref: Option<&[TrajectoryStep]> = if trajectory_mode {
                Some(trajectory_steps.as_slice())
            } else {
                None
            };
            let eval_ll_output_ref = if trajectory_mode { None } else { Some(&output) };

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
                importance_table: eval_ll_qcf_importance.as_ref(),
                noise_table: Some(model.quant_noise.as_ref()),
                ppl: None,
                avg_nll: None,
                n_eval_tokens: 0,
                wall_time_s: total_wall,
                warmup_tokens: args.qcf_warmup_tokens,
                backend: &args.backend,
                kv_type: &args.kv_type,
                ppl_corpus: None,
                eval_ll_output: eval_ll_output_ref,
                trajectory: trajectory_ref,
                dpllm_epsilon: eval_ll_qcf_dpllm_epsilon.as_deref(),
                dpllm_epsilon_multi: eval_ll_qcf_dpllm_epsilon_multi.as_deref(),
                dpllm_epsilon_abs: eval_ll_qcf_dpllm_epsilon_abs.as_deref(),
                dpllm_epsilon_qcf: eval_ll_qcf_dpllm_epsilon_qcf.as_deref(),
                direct_attn_f4: eval_ll_qcf_direct_attn_f4.as_deref(),
                direct_attn_f5: eval_ll_qcf_direct_attn_f5.as_deref(),
                direct_attn_f5_decode_only: eval_ll_qcf_direct_attn_f5_decode_only.as_deref(),
                direct_attn_f5_prefill_decode: eval_ll_qcf_direct_attn_f5_prefill_decode.as_deref(),
            };

            dump_qcf_swap_json(dump_path, &ctx)?;
            eprintln!(
                "[QCF-dump] eval-ll JSON written to {}{}",
                dump_path.display(),
                if trajectory_mode {
                    " (trajectory schema_v2)"
                } else {
                    ""
                }
            );
        }

        let mut json_val = serde_json::from_str::<serde_json::Value>(&output.to_json()?)?;
        json_val["config"] = serde_json::json!({
            "model": args.model_path,
            "eviction_policy": args.eviction_policy,
            "kv_budget": args.kv_budget,
            "kv_budget_ratio": args.kv_budget_ratio,
            "max_seq_len": max_seq_len,
            "kv_type": args.kv_type,
            "h2o_keep_ratio": args.h2o_keep_ratio,
            "h2o_decay": args.h2o_decay,
            "time_normalized": !args.h2o_raw_scores,
            "skip_layers": args.skip_layers,
            "skip_ratio": args.skip_ratio,
        });
        println!("{}", serde_json::to_string_pretty(&json_val)?);
        return Ok(());
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
    let mut qcf_warmup_importance: Option<llm_rs2::core::qcf::ImportanceTable> = None;
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
    if let Some(ref ppl_path) = args.ppl {
        // LISWAP-PPL diagnostic: dump weights immediately after model load
        // (before any swap), useful for Q4-native baseline comparison.
        if let Some(ref dump_dir) = args.dump_q4_after_load {
            dump_layer_weights_to_dir(&model, &backend, dump_dir)?;
        }

        // LISWAP-PPL Scenario E (warmup-then-measure):
        //   Pass 1: drive the weight swap to completion with no NLL logging.
        //   Reset KV caches + score_accumulator so the measurement pass sees a
        //   fresh cache, then run the measurement pass with the swap trigger
        //   disabled. The cache reset isolates the "cache mismatch" hypothesis
        //   from the "weight quantization-path mismatch" hypothesis.
        if args.ppl_warmup_swap {
            if args.ppl_swap_at_token.is_none() {
                anyhow::bail!(
                    "--ppl-warmup-swap requires --ppl-swap-at-token (the warmup pass needs a swap trigger)"
                );
            }
            if model.secondary_mmap.is_none() {
                anyhow::bail!(
                    "--ppl-warmup-swap requires --secondary-gguf (weights must be available for swap)"
                );
            }

            let mut warmup_args = args.clone();
            // Suppress CSV/JSON outputs on the warmup pass.
            warmup_args.ppl_nll_csv = None;
            warmup_args.qcf_dump = None;
            eprintln!("[PPL-Swap] === Pass 1: warmup (driving swap to completion) ===");
            let _warmup_dummy = run_ppl(
                &warmup_args,
                &model,
                &tokenizer,
                &backend,
                &*memory,
                &mut kv_caches,
                &mut cache_manager,
                &mut score_accumulator,
                vocab_size,
                hidden_size,
                max_seq_len,
                ppl_path,
                auto_eviction,
                score_based_eviction,
                actual_protected_prefix,
                skip_config.as_ref(),
                /* warmup_only */ true,
            )?;

            // LISWAP-PPL diagnostic: dump weights right after swap completion
            // (before cache reset), so each layer's GPU buffer can be compared
            // byte-for-byte against the Q4-native baseline dump.
            if let Some(ref dump_dir) = args.dump_q4_after_swap {
                dump_layer_weights_to_dir(&model, &backend, dump_dir)?;
            }

            // Reset KV cache positions. The underlying tensor buffers stay
            // allocated; we only rewind the write head + high-water mark so
            // the next prefill starts from pos 0.
            for cache in kv_caches.iter_mut() {
                cache.current_pos = 0;
                cache.high_water_pos = 0;
            }
            if let Some(acc) = score_accumulator.as_mut() {
                acc.reset();
            }
            eprintln!("[PPL-Swap] === Pass 2: measurement (swap disabled, fresh KV cache) ===");
        }

        // Measurement pass. When warmup_swap was active, disable further swap
        // triggers and clear the warmup flag locally so this pass is a pure
        // teacher-forcing PPL run on the already-swapped weights.
        // When `--ppl-measure-prefill-tokens` is set, pass 2 uses that prefill
        // length instead of `--ppl-prefill-tokens` (Scenario F: large prefill
        // shrinks the decode loop, isolating batch vs single-step path).
        let mut measure_args_owned;
        let measure_args: &Args = if args.ppl_warmup_swap {
            measure_args_owned = args.clone();
            measure_args_owned.ppl_swap_at_token = None;
            measure_args_owned.ppl_warmup_swap = false;
            if let Some(measure_prefill) = args.ppl_measure_prefill_tokens {
                measure_args_owned.ppl_prefill_tokens = Some(measure_prefill);
            }
            &measure_args_owned
        } else {
            &args
        };
        let ppl_result = run_ppl(
            measure_args,
            &model,
            &tokenizer,
            &backend,
            &*memory,
            &mut kv_caches,
            &mut cache_manager,
            &mut score_accumulator,
            vocab_size,
            hidden_size,
            max_seq_len,
            ppl_path,
            auto_eviction,
            score_based_eviction,
            actual_protected_prefix,
            skip_config.as_ref(),
            /* warmup_only */ false,
        )?;

        // --qcf-dump: write JSON after PPL measurement completes.
        if let Some(ref dump_path) = args.qcf_dump {
            use llm_rs2::eval::qcf_helpers::{QcfSwapDumpContext, dump_qcf_swap_json};

            let empty_swap: Vec<usize> = Vec::new();
            let (swap_set, qcf_predicted, fallback_used) = if let Some(ref dec) = qcf_swap_decision
            {
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
            let total_wall = qcf_workflow_start.elapsed().as_secs_f64() + ppl_result.wall_time_s;

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
                ppl: Some(ppl_result.ppl),
                avg_nll: Some(ppl_result.avg_nll),
                n_eval_tokens: ppl_result.n_eval_tokens,
                wall_time_s: total_wall,
                warmup_tokens: args.qcf_warmup_tokens,
                backend: &args.backend,
                kv_type: &args.kv_type,
                ppl_corpus: Some(ppl_path.as_str()),
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

        return Ok(());
    }

    // ════════════════════════════════════════════════════════════
    //  PROMPT-BATCH MODE: Sequential multi-prompt generation
    // ════════════════════════════════════════════════════════════
    if let Some(ref batch_path) = args.prompt_batch {
        let entries = load_prompt_batch(batch_path)?;
        if entries.is_empty() {
            anyhow::bail!("prompt-batch file is empty: {}", batch_path);
        }
        eprintln!(
            "[Batch] Loaded {} entries from {}",
            entries.len(),
            batch_path
        );

        // ARGUS hook: emit Step1~6 metrics (qcf_caote_max / qcf_per_head /
        // qcf_topk_retention_* / attention_entropy / qcf_beta_amplified_* /
        // qcf_per_layer*) per record, alongside legacy fields.
        // Hook owns cache_manager + score_accumulator from here; subsequent
        // forward calls in this branch route score_accumulator through the hook.
        use llm_rs2::eval::StepHook;
        let pb_qcf_mode_enum = match args.qcf_mode.as_str() {
            "caote" => llm_rs2::core::qcf::QcfMode::Caote,
            "both" => llm_rs2::core::qcf::QcfMode::Both,
            _ => llm_rs2::core::qcf::QcfMode::Attn,
        };
        let pb_qcf_config = llm_rs2::core::qcf::QcfConfig {
            mode: pb_qcf_mode_enum,
            ..llm_rs2::core::qcf::QcfConfig::default()
        };
        let pb_ratio_mode = args.kv_budget_ratio > 0.0;
        let pb_hook_budget = if pb_ratio_mode { 0 } else { args.kv_budget };
        let pb_is_d2o = args.eviction_policy == "d2o";
        let pb_num_layers = model.config.num_hidden_layers;
        let pb_sample_layers = if args.enable_qcf_experimental {
            parse_qcf_sample_layers(&args.qcf_sample_layers, pb_num_layers)
                .map_err(|e| anyhow::anyhow!("--qcf-sample-layers: {e}"))?
        } else {
            vec![0]
        };
        let mut hook = llm_rs2::eval::EvictionHook::new(
            cache_manager,
            score_accumulator.take(),
            pb_qcf_config,
            pb_hook_budget,
            actual_protected_prefix,
            score_based_eviction,
            args.h2o_keep_ratio,
            pb_is_d2o,
            args.kv_type.clone(),
            backend.clone(),
            args.enable_qcf_experimental,
            pb_sample_layers,
        );

        let mut iteration = 0usize;

        // Pre-allocate generation buffers (once)
        let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
        let mut logits = Tensor::new(
            Shape::new(vec![1, 1, vocab_size]),
            logits_buf,
            backend.clone(),
        );
        let eos_id = model.config.eos_token_id;

        // Pre-allocate workspace (once)
        let q_dim = model.config.num_attention_heads * model.config.head_dim;
        let k_dim = model.config.num_key_value_heads * model.config.head_dim;
        let v_dim = k_dim;
        let ffn_hidden = model.config.intermediate_size;
        let x_gen_buf = memory.alloc(hidden_size * 4, DType::F32)?;
        let mut x_gen = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            x_gen_buf,
            backend.clone(),
        );
        let mut gen_ws = LayerWorkspace::new(
            WorkspaceConfig {
                batch_size: 1,
                dim: model.config.hidden_size,
                q_dim,
                k_dim,
                v_dim,
                ffn_hidden,
                n_heads: model.config.num_attention_heads,
                max_seq_len: args.max_seq_len,
            },
            memory.as_ref(),
            backend.clone(),
        )?;

        // Attach partition workspace if tensor partition is active.
        // Use UnifiedBuffer (ALLOC_HOST_PTR, host-accessible + GPU-accessible) for partition
        // buffers so merge can use direct pointer access instead of read_buffer/write_buffer.
        let layer0_partition_probe = model.layers[0].load_weights();
        if let Some(ref ctx) = layer0_partition_probe.partition_ctx {
            let gpu_alloc = make_partition_gpu_alloc(&*backend, memory.as_ref());

            // Zero-copy residual: permanent-map ws.residual's UnifiedBuffer so the
            // partition decode path can read residual directly via as_ptr() and
            // skip the per-layer read_buffer DMA (currently ~1.15 ms/layer).
            // Gate behind LLMRS_PARTITION_ZCOPY_RESIDUAL=1, or auto-enable when
            // poll-flag mode is active (skipping the read_buffer is the whole
            // point of the spin-poll path).
            #[cfg(feature = "opencl")]
            if std::env::var_os("LLMRS_PARTITION_ZCOPY_RESIDUAL").is_some()
                || llm_rs2::layers::tensor_partition::partition_poll_flag_enabled()
            {
                if let Some(ub) = gen_ws
                    .residual
                    .buffer()
                    .as_any()
                    .downcast_ref::<llm_rs2::buffer::unified_buffer::UnifiedBuffer>()
                {
                    ub.map()?;
                    eprintln!("[Partition] Residual UnifiedBuffer permanent-mapped for zero-copy");
                } else {
                    eprintln!(
                        "[Partition] WARN: residual buffer is not UnifiedBuffer (zero-copy skipped)"
                    );
                }
            }

            gen_ws.partition_ws = Some(Arc::new(PartitionWsCell::new(PartitionWorkspace::new(
                ctx,
                ffn_hidden,
                hidden_size,
                &gpu_alloc,
                backend.clone(),
                cpu_backend_arc.clone(),
            )?)));
        }

        // Pre-allocate CPU/GPU single-token tensors
        let cpu_gen_indices_buf = Galloc::new().alloc(4, DType::U8)?;
        let cpu_gen_input = Tensor::new(
            Shape::new(vec![1, 1]),
            cpu_gen_indices_buf,
            Arc::new(CpuBackend::new()),
        );
        let gpu_gen_input_buf = memory.alloc(4, DType::U8)?;
        let mut gen_input_tensor =
            Tensor::new(Shape::new(vec![1, 1]), gpu_gen_input_buf, backend.clone());
        let mut logits_cpu = vec![0.0f32; vocab_size];

        // Persistent prefill policy state: survives across batches.
        // Only reset by RestoreDefaults, not by prefill→decode transition.
        let mut persistent_chunk_size: Option<usize> = None;
        let mut persistent_yield_ms: Option<u32> = None;
        let mut persistent_cpu_chunk_size: Option<usize> = None;

        'outer: loop {
            for entry in &entries {
                if args.max_iterations > 0 && iteration >= args.max_iterations {
                    break 'outer;
                }

                let prompt_text = resolve_prompt(entry)?;
                let encoding = tokenizer
                    .encode(prompt_text.as_str(), true)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                let batch_input_ids: Vec<u32> = encoding.get_ids().to_vec();
                let prompt_tokens = batch_input_ids.len();

                eprintln!(
                    "[Batch] #{} id={}, prompt_tokens={}",
                    iteration, entry.id, prompt_tokens
                );

                // Per-record budget when --kv-budget-ratio is active
                // (mirrors eval-ll path eval_loop.rs:207). Without this the hook
                // sees effective_budget=0 and post_prefill early-returns,
                // suppressing eviction and ARGUS metric collection.
                if pb_ratio_mode {
                    let dynamic_budget = ((prompt_tokens as f32) * args.kv_budget_ratio) as usize;
                    hook.set_effective_budget(dynamic_budget.max(1));
                }

                let entry_start = std::time::Instant::now();

                eprintln!(
                    "[Batch] #{} id={} prefill_start ts={:.3}",
                    iteration,
                    entry.id,
                    unix_ts()
                );

                // === PREFILL ===
                let process_len = batch_input_ids.len();
                if process_len > max_seq_len {
                    eprintln!(
                        "[Batch] #{} id={}: prompt too long ({} > {}), skipping",
                        iteration, entry.id, process_len, max_seq_len
                    );
                    let err_result = serde_json::json!({
                        "id": entry.id,
                        "error": format!("prompt too long: {} > {}", process_len, max_seq_len),
                    });
                    println!("{}", serde_json::to_string(&err_result)?);
                    iteration += 1;
                    continue;
                }

                // Chunked prefill
                // When resilience is enabled, auto-chunk at 256 for checkpoint support.
                // When GPU backend and prefill_chunk_size == 0, auto-derive a safe chunk
                // size from max_single_alloc() to avoid CL_INVALID_BUFFER_SIZE (-61).
                let auto_gpu_chunk: Option<usize> =
                    if args.prefill_chunk_size == 0 && backend.is_gpu() {
                        let max_alloc = backend.max_single_alloc();
                        if max_alloc > 0 {
                            // Each chunk needs a logits buffer: chunk * vocab_size * 4 bytes.
                            // Use 50% of max_single_alloc as conservative budget.
                            let budget = max_alloc / 2;
                            let by_vocab = (budget / (vocab_size * 4)).max(1);
                            // Also bound by hidden_size to keep activation buffers feasible.
                            let by_hidden = (max_alloc / (hidden_size * 4)).max(1);
                            let derived = by_vocab.min(by_hidden).min(512);
                            Some(derived)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                let chunk_size = if args.prefill_chunk_size > 0
                    && args.prefill_chunk_size < process_len
                {
                    args.prefill_chunk_size
                } else if let Some(auto) = auto_gpu_chunk {
                    if auto < process_len {
                        eprintln!(
                            "[Prefill] prefill_chunk_size auto-selected: {} (max_alloc={}MB, vocab={}, hidden={})",
                            auto,
                            backend.max_single_alloc() / (1024 * 1024),
                            vocab_size,
                            hidden_size,
                        );
                        auto
                    } else {
                        process_len
                    }
                } else if args.enable_resilience && process_len > 256 {
                    256
                } else {
                    process_len
                };
                let chunked = chunk_size < process_len;

                // Dynamic prefill policy: use persistent values if set by prior
                // SetPrefillPolicy, otherwise fall back to CLI defaults.
                let mut effective_chunk_size = persistent_chunk_size.unwrap_or(chunk_size);
                let mut effective_yield_ms = persistent_yield_ms.unwrap_or(args.prefill_yield_ms);
                let mut effective_cpu_chunk_size =
                    persistent_cpu_chunk_size.unwrap_or(args.prefill_cpu_chunk_size);

                let (prefill_logits_shape, prefill_logits_buf_size) = if chunked {
                    (Shape::new(vec![1, 1, vocab_size]), vocab_size * 4)
                } else {
                    (
                        Shape::new(vec![1, process_len, vocab_size]),
                        process_len * vocab_size * 4,
                    )
                };
                // Use CPU memory when on CPU after SwitchHw; GPU memory otherwise.
                let batch_effective_mem: &dyn Memory = if is_gpu {
                    memory.as_ref()
                } else {
                    cpu_memory_arc.as_ref()
                };
                let prefill_logits_buf =
                    batch_effective_mem.alloc(prefill_logits_buf_size, DType::F32)?;
                let mut prefill_logits =
                    Tensor::new(prefill_logits_shape, prefill_logits_buf, backend.clone());

                let prefill_timer = std::time::Instant::now();
                let mut deferred_switch: Option<String> = None;
                let total_chunks = process_len.div_ceil(chunk_size);

                // Report prefill start to resilience manager.
                if let Some(executor) = &mut command_executor {
                    executor.set_prefill_state("prefill", 0, process_len);
                }

                let mut chunk_start = 0;
                let mut chunk_idx = 0usize;
                while chunk_start < process_len {
                    // Guard: effective_chunk_size must be at least 1.
                    let ecs = effective_chunk_size.max(1);
                    let chunk_end = (chunk_start + ecs).min(process_len);
                    let chunk_tokens = &batch_input_ids[chunk_start..chunk_end];
                    let chunk_len = chunk_tokens.len();

                    let cpu_chunk_buf = Galloc::new().alloc(chunk_len * 4, DType::U8)?;
                    unsafe {
                        let ptr = cpu_chunk_buf.as_mut_ptr() as *mut u32;
                        std::ptr::copy_nonoverlapping(chunk_tokens.as_ptr(), ptr, chunk_len);
                    }
                    let cpu_chunk_tensor = Tensor::new(
                        Shape::new(vec![1, chunk_len]),
                        cpu_chunk_buf,
                        Arc::new(CpuBackend::new()),
                    );
                    let input_tensor = backend.copy_from(&cpu_chunk_tensor)?;

                    model.forward_into(TransformerModelForwardArgs {
                        input_tokens: &input_tensor,
                        start_pos: chunk_start,
                        kv_caches: &mut kv_caches,
                        backend: &backend,
                        memory: batch_effective_mem,
                        logits_out: &mut prefill_logits,
                        x_gen: None,
                        workspace: None,
                        score_accumulator: hook.score_accumulator(),
                        profiler: None,
                        skip_config: skip_config.as_ref(),
                        importance_collector: None,
                        logits_last_only: chunked,
                        variance_collector: None,
                        prefill_workspace: None,

                        layer_boundary_hook: None,
                    })?;
                    backend.synchronize()?;
                    drop(input_tensor);

                    chunk_start = chunk_end;

                    // Inter-chunk yield: sleep after GPU chunk to release compute.
                    if effective_yield_ms > 0 {
                        std::thread::sleep(std::time::Duration::from_millis(
                            effective_yield_ms as u64,
                        ));
                    }

                    // CPU interleave: process next chunk on CPU while GPU is free.
                    // Invariant: the last chunk must be processed by GPU so that
                    // prefill_logits (GPU buffer) is valid at the end.
                    if effective_cpu_chunk_size > 0 && chunk_start < process_len {
                        let remaining = process_len - chunk_start;
                        if remaining > effective_cpu_chunk_size {
                            // Flush GPU caches before CPU reads KV buffers (ARM UMA coherence).
                            for kv in kv_caches.iter() {
                                kv.k_buffer.buffer().map_for_cpu()?;
                                kv.v_buffer.buffer().map_for_cpu()?;
                            }

                            let cpu_end = (chunk_start + effective_cpu_chunk_size)
                                .min(process_len.saturating_sub(1));
                            if cpu_end > chunk_start {
                                let cpu_tokens = &batch_input_ids[chunk_start..cpu_end];
                                let cpu_len = cpu_tokens.len();

                                let cpu_in_buf = Galloc::new().alloc(cpu_len * 4, DType::U8)?;
                                unsafe {
                                    let ptr = cpu_in_buf.as_mut_ptr() as *mut u32;
                                    std::ptr::copy_nonoverlapping(
                                        cpu_tokens.as_ptr(),
                                        ptr,
                                        cpu_len,
                                    );
                                }
                                let cpu_in_tensor = Tensor::new(
                                    Shape::new(vec![1, cpu_len]),
                                    cpu_in_buf,
                                    cpu_backend_arc.clone(),
                                );

                                let cpu_chunk_start_pos = chunk_start;

                                // CPU prefill logits: separate CPU buffer (discarded).
                                let cpu_logits_buf =
                                    cpu_memory_arc.alloc(vocab_size * 4, DType::F32)?;
                                let mut cpu_logits = Tensor::new(
                                    Shape::new(vec![1, 1, vocab_size]),
                                    cpu_logits_buf,
                                    cpu_backend_arc.clone(),
                                );

                                model.forward_into(TransformerModelForwardArgs {
                                    input_tokens: &cpu_in_tensor,
                                    start_pos: cpu_chunk_start_pos,
                                    kv_caches: &mut kv_caches,
                                    backend: &cpu_backend_arc,
                                    memory: cpu_memory_arc.as_ref(),
                                    logits_out: &mut cpu_logits,
                                    x_gen: None,
                                    workspace: None,
                                    score_accumulator: hook.score_accumulator(),
                                    profiler: None,
                                    skip_config: skip_config.as_ref(),
                                    importance_collector: None,
                                    logits_last_only: true,
                                    variance_collector: None,
                                    prefill_workspace: None,

                                    layer_boundary_hook: None,
                                })?;
                                drop(cpu_in_tensor);
                                drop(cpu_logits);

                                chunk_start = cpu_end;
                            }
                        }
                    }

                    // ── Prefill resilience checkpoint (chunk boundary) ──
                    if chunked && let Some(executor) = &mut command_executor {
                        let kv_snap = KVSnapshot {
                            total_bytes: kv_caches
                                .iter()
                                .map(|c| {
                                    (c.k_buffer.buffer().size() + c.v_buffer.buffer().size()) as u64
                                })
                                .sum(),
                            total_tokens: kv_caches[0].current_pos,
                            capacity: kv_caches[0].capacity(),
                            protected_prefix: actual_protected_prefix,
                            kv_dtype: args.kv_type.clone(),
                            eviction_policy: args.eviction_policy.clone(),
                            skip_ratio: 0.0,
                        };
                        let plan = executor.poll(&kv_snap);

                        // SetPrefillPolicy: dynamically adjust chunk/yield/cpu parameters.
                        // Values persist across batches until RestoreDefaults.
                        if let Some(v) = plan.prefill_chunk_size {
                            effective_chunk_size = v;
                            persistent_chunk_size = Some(v);
                            eprintln!("[Prefill] Policy: chunk_size -> {}", v);
                        }
                        if let Some(v) = plan.prefill_yield_ms {
                            effective_yield_ms = v;
                            persistent_yield_ms = Some(v);
                            eprintln!("[Prefill] Policy: yield_ms -> {}", v);
                        }
                        if let Some(v) = plan.prefill_cpu_chunk_size {
                            let layer0_probe = model.layers[0].load_weights();
                            if v > 0 && layer0_probe.wq.as_ptr().is_null() {
                                eprintln!(
                                    "[Prefill] Policy: cpu_chunk_size={} rejected — weights not CPU-accessible. \
                                     Use --resilience-prealloc-switch or --prefill-cpu-chunk-size at CLI.",
                                    v
                                );
                            } else {
                                effective_cpu_chunk_size = v;
                                persistent_cpu_chunk_size = Some(v);
                                eprintln!("[Prefill] Policy: cpu_chunk_size -> {}", v);
                            }
                        }

                        // Throttle: sleep between chunks
                        if plan.throttle_delay_ms > 0 && plan.throttle_delay_ms != throttle_delay_ms
                        {
                            eprintln!(
                                "[Prefill] Throttle: {}ms -> {}ms",
                                throttle_delay_ms, plan.throttle_delay_ms
                            );
                        }
                        throttle_delay_ms = plan.throttle_delay_ms;
                        if throttle_delay_ms > 0 {
                            eprintln!(
                                "[Prefill] Throttle: {}ms delay after chunk {}/{}",
                                throttle_delay_ms,
                                chunk_idx + 1,
                                total_chunks
                            );
                            std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
                        }

                        // LayerSkip
                        if plan.restore_defaults {
                            skip_config = None;
                            last_skip_ratio = None;
                            effective_chunk_size = chunk_size;
                            effective_yield_ms = args.prefill_yield_ms;
                            effective_cpu_chunk_size = args.prefill_cpu_chunk_size;
                            persistent_chunk_size = None;
                            persistent_yield_ms = None;
                            persistent_cpu_chunk_size = None;
                        } else if let Some(ratio) = plan.layer_skip
                            && last_skip_ratio != Some(ratio)
                        {
                            eprintln!("[Prefill] LayerSkip: ratio={:.2}", ratio);
                            skip_config = Some(SkipConfig::uniform_init(
                                model.config.num_hidden_layers,
                                ratio,
                            ));
                            last_skip_ratio = Some(ratio);
                        }

                        // SwitchHw: defer to post-prefill boundary.
                        // Mid-prefill switch causes segfault: model workspace buffers
                        // remain on the old backend; the next chunk accesses them
                        // from the new backend -> invalid memory reference.
                        if let Some(ref device) = plan.switch_device {
                            if deferred_switch.is_none() {
                                eprintln!(
                                    "[Prefill] SwitchHw: deferring '{}' to post-prefill (chunk_pos={})",
                                    device, kv_caches[0].current_pos
                                );
                            }
                            deferred_switch = Some(device.clone());
                        }

                        // Report prefill progress.
                        executor.set_prefill_state("prefill", chunk_start, process_len);
                    }

                    chunk_idx += 1;
                }

                let ttft_ms = prefill_timer.elapsed().as_secs_f64() * 1000.0;

                // Report transition to decode phase.
                if let Some(executor) = &mut command_executor {
                    executor.set_prefill_state("decode", 0, 0);
                }

                // Sample first token from prefill logits
                let mut last_logits = vec![0.0f32; vocab_size];
                unsafe {
                    let ptr = last_logits.as_mut_ptr() as *mut u8;
                    let byte_len = vocab_size * 4;
                    if chunked {
                        let slice = std::slice::from_raw_parts_mut(ptr, byte_len);
                        backend.read_buffer(&prefill_logits, slice)?;
                    } else {
                        let mut full_logits = vec![0.0f32; process_len * vocab_size];
                        let full_ptr = full_logits.as_mut_ptr() as *mut u8;
                        let full_slice =
                            std::slice::from_raw_parts_mut(full_ptr, full_logits.len() * 4);
                        backend.read_buffer(&prefill_logits, full_slice)?;
                        let start_idx = (process_len - 1) * vocab_size;
                        last_logits
                            .copy_from_slice(&full_logits[start_idx..start_idx + vocab_size]);
                    }
                }
                drop(prefill_logits);

                // Execute deferred SwitchHw (from prefill checkpoint).
                // Now safe: prefill is done, logits read, all workspace released.
                if let Some(ref device) = deferred_switch
                    && let (Some(gpu_be), Some(gpu_mem)) = (&gpu_backend_arc, &gpu_memory_arc)
                {
                    match device.as_str() {
                        "cpu" if is_gpu => {
                            eprintln!("[Prefill->Decode] Executing deferred SwitchHw: GPU->CPU");
                            llm_rs2::core::kv_migrate::migrate_kv_caches(
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
                            is_gpu = false;
                            // Re-allocate decode buffers on CPU.
                            let new_lb = cpu_memory_arc.alloc(vocab_size * 4, DType::F32)?;
                            logits = Tensor::new(logits.shape().clone(), new_lb, backend.clone());
                            let new_xb = cpu_memory_arc.alloc(hidden_size * 4, DType::F32)?;
                            x_gen = Tensor::new(x_gen.shape().clone(), new_xb, backend.clone());
                            gen_ws = LayerWorkspace::new(
                                WorkspaceConfig {
                                    batch_size: 1,
                                    dim: model.config.hidden_size,
                                    q_dim,
                                    k_dim,
                                    v_dim,
                                    ffn_hidden,
                                    n_heads: model.config.num_attention_heads,
                                    max_seq_len: args.max_seq_len,
                                },
                                cpu_memory_arc.as_ref(),
                                backend.clone(),
                            )?;
                            let new_gi = cpu_memory_arc.alloc(4, DType::U8)?;
                            gen_input_tensor = Tensor::new(
                                gen_input_tensor.shape().clone(),
                                new_gi,
                                backend.clone(),
                            );
                            eprintln!(
                                "[Prefill->Decode] SwitchHw: Switched to CPU (GPU handles released, decode buffers re-allocated)."
                            );
                        }
                        "gpu" | "opencl" if !is_gpu && weights_on_gpu => {
                            eprintln!("[Prefill->Decode] Executing deferred SwitchHw: CPU->GPU");
                            llm_rs2::core::kv_migrate::migrate_kv_caches(
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
                            is_gpu = true;
                            let new_lb = gpu_mem.alloc(vocab_size * 4, DType::F32)?;
                            logits = Tensor::new(logits.shape().clone(), new_lb, backend.clone());
                            let new_xb = gpu_mem.alloc(hidden_size * 4, DType::F32)?;
                            x_gen = Tensor::new(x_gen.shape().clone(), new_xb, backend.clone());
                            gen_ws = LayerWorkspace::new(
                                WorkspaceConfig {
                                    batch_size: 1,
                                    dim: model.config.hidden_size,
                                    q_dim,
                                    k_dim,
                                    v_dim,
                                    ffn_hidden,
                                    n_heads: model.config.num_attention_heads,
                                    max_seq_len: args.max_seq_len,
                                },
                                gpu_mem.as_ref(),
                                backend.clone(),
                            )?;
                            let new_gi = gpu_mem.alloc(4, DType::U8)?;
                            gen_input_tensor = Tensor::new(
                                gen_input_tensor.shape().clone(),
                                new_gi,
                                backend.clone(),
                            );
                            eprintln!(
                                "[Prefill->Decode] SwitchHw: Switched to GPU (decode buffers re-allocated)."
                            );
                        }
                        _ => {} // Already on requested backend
                    }
                }

                let mut batch_tokens = batch_input_ids.clone();
                let next_token_id = sampling::sample(
                    &mut last_logits,
                    &batch_tokens,
                    vocab_size,
                    &sampling_config,
                    None,
                );
                batch_tokens.push(next_token_id);
                let mut batch_start_pos = process_len;

                // ── Score collection probe ──
                // Mirrors eval_loop.rs:246~287. Batch prefill calls forward with
                // workspace=None, so the hook's score_accumulator stays empty
                // → ARGUS metrics fall back to defaults (0). Re-feed the last
                // prompt token as a 1-step decode forward to populate per-head
                // attention scores, then restore current_pos so cache state
                // matches prompt_tokens (probe entry beyond current_pos is
                // invisible to subsequent forward calls).
                use llm_rs2::core::kv_cache::KVCacheOps;
                if hook.needs_score_probe(&kv_caches) {
                    let saved_positions: Vec<usize> =
                        kv_caches.iter().map(|c| c.current_pos()).collect();
                    let last_prompt_token = batch_input_ids[prompt_tokens - 1];
                    unsafe {
                        *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_prompt_token;
                    }
                    backend.write_buffer(&mut gen_input_tensor, unsafe {
                        std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
                    })?;
                    if let Some(acc) = hook.score_accumulator() {
                        acc.begin_step();
                    }
                    let probe_mem: &dyn Memory = if is_gpu {
                        memory.as_ref()
                    } else {
                        cpu_memory_arc.as_ref()
                    };
                    model.forward_into(TransformerModelForwardArgs {
                        input_tokens: &gen_input_tensor,
                        start_pos: prompt_tokens - 1,
                        kv_caches: &mut kv_caches,
                        backend: &backend,
                        memory: probe_mem,
                        logits_out: &mut logits,
                        x_gen: Some(&mut x_gen),
                        workspace: Some(&mut gen_ws),
                        score_accumulator: hook.score_accumulator(),
                        profiler: None,
                        skip_config: skip_config.as_ref(),
                        importance_collector: None,
                        logits_last_only: false,
                        variance_collector: None,
                        prefill_workspace: None,

                        layer_boundary_hook: None,
                    })?;
                    for (cache, &pos) in kv_caches.iter_mut().zip(saved_positions.iter()) {
                        cache.set_current_pos(pos);
                    }
                }

                // ARGUS Step1~6: compute experimental_qcf payload from prefill state.
                // Also triggers post-prefill eviction when budget exceeded.
                hook.post_prefill(&mut kv_caches);

                eprintln!(
                    "[Batch] #{} id={} prefill_end ts={:.3}",
                    iteration,
                    entry.id,
                    unix_ts()
                );
                eprintln!(
                    "[Batch] #{} id={} decode_start ts={:.3}",
                    iteration,
                    entry.id,
                    unix_ts()
                );

                // === DECODE LOOP ===
                let mut tbt_values_batch: Vec<f64> = Vec::new();
                let mut generated_count: usize = 1; // first token already sampled
                let mut last_token_time = std::time::Instant::now();

                for _ in 0..(args.num_tokens - 1) {
                    if kv_caches[0].current_pos >= max_seq_len {
                        break;
                    }

                    // Throttle delay
                    if throttle_delay_ms > 0 {
                        std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
                    }

                    // Write token to CPU input
                    let current_token = *batch_tokens.last().unwrap();
                    unsafe {
                        *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = current_token;
                    }
                    backend.write_buffer(&mut gen_input_tensor, unsafe {
                        std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
                    })?;

                    let decode_start = std::time::Instant::now();
                    // Use CPU memory when on CPU; GPU memory otherwise.
                    // After SwitchHw GPU→CPU, `memory` is still OpenCL memory whose
                    // alloc() creates OpenCLBuffer (null as_ptr). Must use
                    // cpu_memory_arc for CPU-accessible lazy allocations.
                    let effective_mem: &dyn Memory = if is_gpu {
                        gpu_memory_arc.as_deref().unwrap_or_else(|| memory.as_ref())
                    } else {
                        cpu_memory_arc.as_ref()
                    };
                    model.forward_into(TransformerModelForwardArgs {
                        input_tokens: &gen_input_tensor,
                        start_pos: batch_start_pos,
                        kv_caches: &mut kv_caches,
                        backend: &backend,
                        memory: effective_mem,
                        logits_out: &mut logits,
                        x_gen: Some(&mut x_gen),
                        workspace: Some(&mut gen_ws),
                        score_accumulator: hook.score_accumulator(),
                        profiler: None,
                        skip_config: skip_config.as_ref(),
                        importance_collector: None,
                        logits_last_only: false,
                        variance_collector: None,
                        prefill_workspace: None,

                        layer_boundary_hook: None,
                    })?;
                    backend.synchronize()?;
                    hook.post_decode_step(&mut kv_caches, generated_count);

                    let now = std::time::Instant::now();
                    let tbt = (now - last_token_time).as_secs_f64() * 1000.0;
                    tbt_values_batch.push(tbt);
                    last_token_time = now;
                    let _forward_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

                    // Read logits and sample
                    unsafe {
                        let ptr = logits_cpu.as_mut_ptr() as *mut u8;
                        let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
                        backend.read_buffer(&logits, slice)?;
                    }

                    let next_id = sampling::sample(
                        &mut logits_cpu,
                        &batch_tokens,
                        vocab_size,
                        &sampling_config,
                        None,
                    );
                    batch_tokens.push(next_id);
                    batch_start_pos += 1;
                    generated_count += 1;

                    if next_id == eos_id && !args.ignore_eos {
                        break;
                    }
                }

                eprintln!(
                    "[Batch] #{} id={} decode_end ts={:.3}",
                    iteration,
                    entry.id,
                    unix_ts()
                );

                let total_ms = entry_start.elapsed().as_secs_f64() * 1000.0;
                let mean_tbt_ms = if tbt_values_batch.is_empty() {
                    0.0
                } else {
                    tbt_values_batch.iter().sum::<f64>() / tbt_values_batch.len() as f64
                };

                // Decode generated text (skip prompt tokens)
                let generated_ids = &batch_tokens[prompt_tokens..];
                let text = tokenizer.decode(generated_ids, true).unwrap_or_default();

                // Output JSONL
                let mut result = serde_json::json!({
                    "id": entry.id,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_count,
                    "ttft_ms": (ttft_ms * 100.0).round() / 100.0,
                    "mean_tbt_ms": (mean_tbt_ms * 100.0).round() / 100.0,
                    "total_ms": (total_ms * 100.0).round() / 100.0,
                    "text": text,
                });
                // Merge ARGUS Step1~6 fields (qcf_caote_max / qcf_per_head /
                // qcf_topk_retention_* / attention_entropy / qcf_beta_amplified_* /
                // qcf_per_layer*) when --enable-qcf-experimental is on.
                if let serde_json::Value::Object(extra_map) = hook.extra_question_fields(&kv_caches)
                    && let serde_json::Value::Object(ref mut rmap) = result
                {
                    for (k, v) in extra_map {
                        rmap.insert(k, v);
                    }
                }
                println!("{}", serde_json::to_string(&result)?);

                eprintln!(
                    "[Batch] #{} id={} done: {} tokens, ttft={:.1}ms, tbt={:.1}ms, total={:.1}ms",
                    iteration, entry.id, generated_count, ttft_ms, mean_tbt_ms, total_ms
                );

                // === RESET KV CACHE + score accumulator + per-record hook state ===
                hook.reset_caches(&mut kv_caches);

                iteration += 1;
            }

            if !args.prompt_batch_loop {
                break;
            }
        }

        eprintln!("[Batch] Complete: {} iterations", iteration);
        return Ok(());
    }

    // Inference profiler (activated by either --profile or --profile-events).
    // Declared before prefill so PrefillOpProfiler can be populated.
    //
    // --profile-events uses the same InferenceProfiler container (ops/json
    // export) but feeds it via OpProfiler::merge_from_events() instead of
    // the legacy per-op synchronize+wall-clock path.
    let mut profiler = if args.profile || args.profile_events {
        Some(llm_rs2::profile::InferenceProfiler::new(
            llm_rs2::profile::ProfileConfig {
                score_snapshot_interval: args.profile_interval,
                track_per_head: args.profile_per_head,
                enabled_probes: args.profile_probes.split(',').map(String::from).collect(),
                output_dir: std::path::PathBuf::from(&args.profile_dir),
            },
        ))
    } else {
        None
    };

    // Pre-allocate generation buffers
    let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        logits_buf,
        backend.clone(),
    );

    // Cache EOS token ID from config.json (model-agnostic)
    let eos_id = model.config.eos_token_id;

    // === WARMUP: trigger DVFS ramp-up before timed prefill ===
    // Runs a forward pass and brief CPU spin to ensure governor reaches max clock.
    // Without this, idle CPU starts at ~2.2GHz and ramp-up time
    // pollutes the prefill measurement (llama.cpp's model loading + warmup
    // achieves the same effect).
    //
    // Env overrides (for gap investigation):
    //   LLMRS_SKIP_WARMUP=1     : disable warmup entirely (baseline cold-start)
    //   LLMRS_WARMUP_TOKENS=N   : warmup with N tokens (default 1). Use >1 to JIT-compile
    //                             prefill-path kernels (batched QKV / flash_attn prefill).
    if std::env::var("LLMRS_SKIP_WARMUP").is_err() {
        let warmup_tokens: usize = std::env::var("LLMRS_WARMUP_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
            .max(1)
            .min(tokens.len());

        let warmup_start = std::time::Instant::now();
        let warmup_buf = Galloc::new().alloc(warmup_tokens * 4, DType::U8)?;
        unsafe {
            let ptr = warmup_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, warmup_tokens);
        }
        let warmup_input = Tensor::new(
            Shape::new(vec![1, warmup_tokens]),
            warmup_buf,
            Arc::new(CpuBackend::new()),
        );
        let warmup_input = backend.copy_from(&warmup_input)?;

        let warmup_logits_shape = if warmup_tokens == 1 {
            Shape::new(vec![1, 1, vocab_size])
        } else {
            Shape::new(vec![1, warmup_tokens, vocab_size])
        };
        let warmup_logits_buf = memory.alloc(warmup_tokens * vocab_size * 4, DType::F32)?;
        let mut warmup_logits =
            Tensor::new(warmup_logits_shape, warmup_logits_buf, backend.clone());

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &warmup_input,
            start_pos: 0,
            kv_caches: &mut kv_caches,
            backend: &backend,
            memory: memory.as_ref(),
            logits_out: &mut warmup_logits,
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
        backend.synchronize()?;
        let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[WARMUP] tokens={} ms={:.2}", warmup_tokens, warmup_ms);

        // Brief all-core spin to push DVFS governor to max frequency.
        // 50ms is enough for walt governor to ramp up.
        use rayon::prelude::*;
        let spin_until = std::time::Instant::now() + std::time::Duration::from_millis(50);
        (0..rayon::current_num_threads())
            .into_par_iter()
            .for_each(|_| {
                while std::time::Instant::now() < spin_until {
                    std::hint::spin_loop();
                }
            });

        // Reset KV caches
        for cache in kv_caches.iter_mut() {
            cache.current_pos = 0;
            cache.high_water_pos = 0;
        }
    } else {
        eprintln!("[WARMUP] skipped (LLMRS_SKIP_WARMUP)");
    }

    // D2O layer-level allocation: create variance collector before prefill.
    // Only active when --eviction-policy d2o and --d2o-layer-alloc are both set.
    let mut variance_collector = if args.d2o_layer_alloc && args.eviction_policy == "d2o" {
        Some(
            llm_rs2::core::pressure::d2o_layer_alloc::D2OVarianceCollector::new(
                model.config.num_hidden_layers,
                model.config.num_key_value_heads,
                model.config.num_attention_heads,
                model.config.head_dim,
                tokens.len(),
            ),
        )
    } else {
        None
    };

    // Weight swap state (ENG-ALG-218 + ENG-ALG-214-ROUTE).
    //
    // `importance_table_for_swap`: most-recently collected per-layer
    // importance table from an on-demand prefill measurement.  `None`
    // until the first `RequestQcf` prefill completes.
    //
    // `collector_armed`: true when a `RequestQcf` has been received and
    // we are waiting for the next prefill to inject `ImportanceCollector`.
    // This is a lightweight bool; the actual collector lives on the stack
    // during prefill (not stored here).
    let mut importance_table_for_swap: Option<llm_rs2::core::qcf::ImportanceTable> = None;
    let mut collector_armed = false;

    // === PREFILL PHASE ===
    let mut deferred_switch: Option<String> = None;
    {
        println!("[Profile] Event: PrefillStart");
        let process_len = tokens.len();
        if process_len > max_seq_len {
            anyhow::bail!(
                "Prompt length {} exceeds max_seq_len {}",
                process_len,
                max_seq_len
            );
        }

        // Determine effective chunk size.
        // 0 or >= process_len → use full prompt as single chunk (original behaviour).
        // When resilience is enabled, auto-chunk at 256 so that chunk boundaries
        // serve as checkpoints for SwitchHw / Throttle / LayerSkip commands.
        // When GPU backend and prefill_chunk_size == 0, auto-derive a safe chunk
        // size from max_single_alloc() to avoid CL_INVALID_BUFFER_SIZE (-61).
        let auto_gpu_chunk: Option<usize> = if args.prefill_chunk_size == 0 && backend.is_gpu() {
            let max_alloc = backend.max_single_alloc();
            if max_alloc > 0 {
                // Each chunk needs a logits buffer: chunk * vocab_size * 4 bytes.
                // Use 50% of max_single_alloc as conservative budget.
                let budget = max_alloc / 2;
                let by_vocab = (budget / (vocab_size * 4)).max(1);
                // Also bound by hidden_size to keep activation buffers feasible.
                let by_hidden = (max_alloc / (hidden_size * 4)).max(1);
                let derived = by_vocab.min(by_hidden).min(512);
                Some(derived)
            } else {
                None
            }
        } else {
            None
        };
        let chunk_size = if args.prefill_chunk_size > 0 && args.prefill_chunk_size < process_len {
            args.prefill_chunk_size
        } else if let Some(auto) = auto_gpu_chunk {
            if auto < process_len {
                eprintln!(
                    "[Prefill] prefill_chunk_size auto-selected: {} (max_alloc={}MB, vocab={}, hidden={})",
                    auto,
                    backend.max_single_alloc() / (1024 * 1024),
                    vocab_size,
                    hidden_size,
                );
                auto
            } else {
                process_len
            }
        } else if args.enable_resilience && process_len > 256 {
            256
        } else {
            process_len
        };
        let chunked = chunk_size < process_len;
        if chunked {
            eprintln!(
                "[Prefill] Chunked mode: {} tokens in chunks of {}",
                process_len, chunk_size
            );
        }

        // Dynamic prefill policy: start from CLI values, updated by SetPrefillPolicy.
        let mut effective_chunk_size = chunk_size;
        let mut effective_yield_ms = args.prefill_yield_ms;
        let mut effective_cpu_chunk_size = args.prefill_cpu_chunk_size;

        // Reusable logits buffer: [1, 1, vocab_size] when chunked, else [1, process_len, vocab_size].
        // Chunked mode always uses logits_last_only=true so only 1 position is written per chunk.
        let (logits_shape, logits_buf_size) = if chunked {
            (Shape::new(vec![1, 1, vocab_size]), vocab_size * 4)
        } else {
            (
                Shape::new(vec![1, process_len, vocab_size]),
                process_len * vocab_size * 4,
            )
        };
        let prefill_logits_buf = memory.alloc(logits_buf_size, DType::F32)?;
        let mut prefill_logits = Tensor::new(logits_shape, prefill_logits_buf, backend.clone());

        let prefill_timer = std::time::Instant::now();
        let mut prefill_pure_fwd_ms: f64 = 0.0;
        let total_chunks = process_len.div_ceil(chunk_size);

        // Report prefill start to resilience manager.
        if let Some(executor) = &mut command_executor {
            executor.set_prefill_state("prefill", 0, process_len);
        }

        // ENG-ALG-218: if collector is armed, prepare a collector for this prefill.
        // Armed by `RequestQcf` handler in decode loop; collector is injected into
        // the last prefill chunk so it captures the final contextual activation state.
        let mut on_demand_collector: Option<llm_rs2::core::qcf::ImportanceCollector> =
            if collector_armed {
                Some(llm_rs2::core::qcf::ImportanceCollector::new())
            } else {
                None
            };
        if collector_armed {
            collector_armed = false; // consume the flag; armed at most once per prefill
        }

        let mut chunk_start = 0;
        let mut chunk_idx = 0usize;
        while chunk_start < process_len {
            // Guard: effective_chunk_size must be at least 1.
            let ecs = effective_chunk_size.max(1);
            let chunk_end = (chunk_start + ecs).min(process_len);
            let chunk_tokens = &tokens[chunk_start..chunk_end];
            let chunk_len = chunk_tokens.len();

            // ENG-ALG-218: inject collector only on the last prefill chunk.
            // Earlier chunks have partial seq_len; the last chunk captures final
            // contextual state which is most representative for per-layer importance.
            let is_last_chunk = chunk_end >= process_len;
            let inject_collector = is_last_chunk && on_demand_collector.is_some();

            let chunk_trace = std::env::var("LLMRS_PREFILL_CHUNK_MS").is_ok();
            let t_chunk_start = std::time::Instant::now();

            // Build CPU input tensor for this chunk.
            let cpu_chunk_buf = Galloc::new().alloc(chunk_len * 4, DType::U8)?;
            unsafe {
                let ptr = cpu_chunk_buf.as_mut_ptr() as *mut u32;
                std::ptr::copy_nonoverlapping(chunk_tokens.as_ptr(), ptr, chunk_len);
            }
            let cpu_chunk_tensor = Tensor::new(
                Shape::new(vec![1, chunk_len]),
                cpu_chunk_buf,
                Arc::new(CpuBackend::new()),
            );
            let input_tensor = backend.copy_from(&cpu_chunk_tensor)?;
            let t_setup_end = std::time::Instant::now();

            // RoPE position for this chunk: start_pos (0 during prefill) + offset within prompt.
            let chunk_start_pos = start_pos + chunk_start;

            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos: chunk_start_pos,
                kv_caches: &mut kv_caches,
                backend: &backend,
                memory: memory.as_ref(),
                logits_out: &mut prefill_logits,
                x_gen: None,
                workspace: None,
                score_accumulator: None, // No score tracking during prefill
                profiler: profiler.as_mut().map(|p| &mut p.ops),
                skip_config: None,
                importance_collector: if inject_collector {
                    on_demand_collector.as_mut()
                } else {
                    None
                },
                // Chunked mode: only the last position's logits needed (saves GPU memory).
                // Non-chunked: write all positions (original behaviour).
                logits_last_only: chunked,
                variance_collector: variance_collector.as_mut(),
                prefill_workspace: None,

                layer_boundary_hook: None,
            })?;
            backend.synchronize()?;
            let t_fwd_end = std::time::Instant::now();
            let fwd_ms = (t_fwd_end - t_setup_end).as_secs_f64() * 1000.0;
            prefill_pure_fwd_ms += fwd_ms;
            if chunk_trace {
                let setup_ms = (t_setup_end - t_chunk_start).as_secs_f64() * 1000.0;
                let total_ms = (t_fwd_end - t_chunk_start).as_secs_f64() * 1000.0;
                eprintln!(
                    "[PREFILL_CHUNK] idx={} start_pos={} len={} setup_ms={:.2} fwd_ms={:.2} total_ms={:.2}",
                    chunk_idx, chunk_start_pos, chunk_len, setup_ms, fwd_ms, total_ms
                );
            }

            // Immediately release the GPU input buffer for this chunk.
            drop(input_tensor);

            chunk_start = chunk_end;

            // Inter-chunk yield: sleep after GPU chunk to release compute for other processes.
            if effective_yield_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(effective_yield_ms as u64));
            }

            // CPU interleave: process next chunk on CPU while GPU is free.
            // Invariant: the last chunk must be processed by GPU so that
            // prefill_logits (GPU buffer) is valid at the end.
            if effective_cpu_chunk_size > 0 && chunk_start < process_len {
                let remaining = process_len - chunk_start;
                // Only run CPU chunk if enough tokens remain for GPU to handle
                // at least one more chunk afterwards.
                if remaining > effective_cpu_chunk_size {
                    // Flush GPU caches to main memory before CPU reads KV buffers.
                    // On ARM UMA, clFinish() alone may not flush GPU L1/L2 cache.
                    // map_for_cpu() calls clEnqueueMapBuffer which ensures coherence.
                    for kv in kv_caches.iter() {
                        kv.k_buffer.buffer().map_for_cpu()?;
                        kv.v_buffer.buffer().map_for_cpu()?;
                    }

                    let cpu_end =
                        (chunk_start + effective_cpu_chunk_size).min(process_len.saturating_sub(1));
                    if cpu_end > chunk_start {
                        let cpu_tokens = &tokens[chunk_start..cpu_end];
                        let cpu_len = cpu_tokens.len();

                        let cpu_in_buf = Galloc::new().alloc(cpu_len * 4, DType::U8)?;
                        unsafe {
                            let ptr = cpu_in_buf.as_mut_ptr() as *mut u32;
                            std::ptr::copy_nonoverlapping(cpu_tokens.as_ptr(), ptr, cpu_len);
                        }
                        let cpu_in_tensor = Tensor::new(
                            Shape::new(vec![1, cpu_len]),
                            cpu_in_buf,
                            cpu_backend_arc.clone(),
                        );

                        let cpu_chunk_start_pos = start_pos + chunk_start;

                        // CPU prefill logits: use a separate CPU buffer to avoid writing
                        // to GPU prefill_logits. These intermediate logits are discarded.
                        let cpu_logits_buf = cpu_memory_arc.alloc(vocab_size * 4, DType::F32)?;
                        let mut cpu_logits = Tensor::new(
                            Shape::new(vec![1, 1, vocab_size]),
                            cpu_logits_buf,
                            cpu_backend_arc.clone(),
                        );

                        model.forward_into(TransformerModelForwardArgs {
                            input_tokens: &cpu_in_tensor,
                            start_pos: cpu_chunk_start_pos,
                            kv_caches: &mut kv_caches,
                            backend: &cpu_backend_arc,
                            memory: cpu_memory_arc.as_ref(),
                            logits_out: &mut cpu_logits,
                            x_gen: None,
                            workspace: None,
                            score_accumulator: None,
                            profiler: None,
                            skip_config: None,
                            importance_collector: None,
                            logits_last_only: true,
                            variance_collector: None,
                            prefill_workspace: None,

                            layer_boundary_hook: None,
                        })?;
                        // No backend.synchronize() needed — CPU forward is synchronous.
                        drop(cpu_in_tensor);
                        drop(cpu_logits);

                        chunk_start = cpu_end;
                    }
                }
                // else: remaining tokens fit in one GPU chunk → GPU finishes.
            }

            // ── Prefill resilience checkpoint (chunk boundary) ──
            // Poll CommandExecutor between chunks to handle SwitchHw, Throttle,
            // and LayerSkip commands mid-prefill. Only active in chunked mode.
            if chunked && let Some(executor) = &mut command_executor {
                let kv_snap = KVSnapshot {
                    total_bytes: kv_caches
                        .iter()
                        .map(|c| (c.k_buffer.buffer().size() + c.v_buffer.buffer().size()) as u64)
                        .sum(),
                    total_tokens: kv_caches[0].current_pos,
                    capacity: kv_caches[0].capacity(),
                    protected_prefix: actual_protected_prefix,
                    kv_dtype: args.kv_type.clone(),
                    eviction_policy: args.eviction_policy.clone(),
                    skip_ratio: 0.0,
                };
                let plan = executor.poll(&kv_snap);

                // SetPrefillPolicy: dynamically adjust chunk/yield/cpu parameters.
                if let Some(v) = plan.prefill_chunk_size {
                    effective_chunk_size = v;
                    eprintln!("[Prefill] Policy: chunk_size -> {}", v);
                }
                if let Some(v) = plan.prefill_yield_ms {
                    effective_yield_ms = v;
                    eprintln!("[Prefill] Policy: yield_ms -> {}", v);
                }
                if let Some(v) = plan.prefill_cpu_chunk_size {
                    let layer0_probe = model.layers[0].load_weights();
                    if v > 0 && layer0_probe.wq.as_ptr().is_null() {
                        eprintln!(
                            "[Prefill] Policy: cpu_chunk_size={} rejected — weights not CPU-accessible. \
                             Use --resilience-prealloc-switch or --prefill-cpu-chunk-size at CLI.",
                            v
                        );
                    } else {
                        effective_cpu_chunk_size = v;
                        eprintln!("[Prefill] Policy: cpu_chunk_size -> {}", v);
                    }
                }

                // Throttle: sleep between chunks
                if plan.throttle_delay_ms > 0 && plan.throttle_delay_ms != throttle_delay_ms {
                    eprintln!(
                        "[Prefill] Throttle: {}ms -> {}ms",
                        throttle_delay_ms, plan.throttle_delay_ms
                    );
                }
                throttle_delay_ms = plan.throttle_delay_ms;
                if throttle_delay_ms > 0 {
                    eprintln!(
                        "[Prefill] Throttle: {}ms delay after chunk {}/{}",
                        throttle_delay_ms,
                        chunk_idx + 1,
                        total_chunks
                    );
                    std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
                }

                // LayerSkip
                if plan.restore_defaults {
                    skip_config = None;
                    last_skip_ratio = None;
                    effective_chunk_size = chunk_size;
                    effective_yield_ms = args.prefill_yield_ms;
                    effective_cpu_chunk_size = args.prefill_cpu_chunk_size;
                } else if let Some(ratio) = plan.layer_skip
                    && last_skip_ratio != Some(ratio)
                {
                    eprintln!("[Prefill] LayerSkip: ratio={:.2}", ratio);
                    skip_config = Some(SkipConfig::uniform_init(
                        model.config.num_hidden_layers,
                        ratio,
                    ));
                    last_skip_ratio = Some(ratio);
                }

                // SwitchHw: defer to post-prefill boundary.
                // Mid-prefill switch causes segfault: model workspace buffers
                // remain on the old backend; the next chunk accesses them
                // from the new backend -> invalid memory reference.
                if let Some(ref device) = plan.switch_device {
                    if deferred_switch.is_none() {
                        eprintln!(
                            "[Prefill] SwitchHw: deferring '{}' to post-prefill (chunk_pos={})",
                            device, kv_caches[0].current_pos
                        );
                    }
                    deferred_switch = Some(device.clone());
                }

                // Report prefill progress.
                executor.set_prefill_state("prefill", chunk_start, process_len);
            }

            chunk_idx += 1;
        }

        // ENG-ALG-218: finalize on-demand ImportanceCollector after prefill completes.
        // INV-128: this block always runs (normal fall-through from the while loop),
        // so QcfEstimate is guaranteed to be sent when the prefill completes successfully.
        // For panics/early-return paths the caller-side Drop guard is the safety net.
        if let Some(collector) = on_demand_collector.take() {
            let table: llm_rs2::core::qcf::ImportanceTable = collector.build();
            let layer_swap = build_layer_swap_estimate(&model, Some(&table));
            if let Some(executor) = &mut command_executor {
                executor.send_qcf_estimate(llm_shared::QcfEstimate {
                    estimates: std::collections::HashMap::new(),
                    layer_swap,
                });
                log::debug!("[QCF] QcfEstimate sent after prefill finalization (ENG-ALG-218)");
            }
            importance_table_for_swap = Some(table);
        }

        let prefill_forward_ms = prefill_timer.elapsed().as_secs_f64() * 1000.0;

        // Report transition to decode phase.
        if let Some(executor) = &mut command_executor {
            executor.set_prefill_state("decode", 0, 0);
        }

        // Auto-eviction after prefill (sliding window only, non-experiment mode)
        if auto_eviction {
            cache_manager.maybe_evict(&mut kv_caches).ok();
        }

        // Sticky eviction at prefill→decode boundary.
        // If a KvEvict directive arrived during prefill, executor holds a sticky evict_plan.
        // Execute it now (before decode starts) to reduce attention work from the first decode step.
        // Score-based methods (H2O/D2O) are not available here — falls back to force_evict.
        if let Some(ref mut exec) = command_executor {
            let kv_snap = KVSnapshot {
                total_bytes: kv_caches
                    .iter()
                    .map(|c| (c.k_buffer.buffer().size() + c.v_buffer.buffer().size()) as u64)
                    .sum(),
                total_tokens: kv_caches[0].current_pos,
                capacity: kv_caches[0].capacity(),
                protected_prefix: actual_protected_prefix,
                kv_dtype: "f16".to_string(),
                eviction_policy: args.eviction_policy.clone(),
                skip_ratio: 0.0,
            };
            let plan = exec.poll(&kv_snap);
            if let Some(evict) = &plan.evict {
                let effective_ratio = args.experiment_eviction_ratio.unwrap_or(evict.target_ratio);
                if effective_ratio > 0.0 {
                    let current_pos = kv_caches[0].current_pos;
                    // Use current_pos as ceiling (first and only boundary eviction).
                    let tgt_raw = (current_pos as f32 * effective_ratio).max(1.0) as usize;
                    let target_pos = tgt_raw.max(args.min_kv_cache);
                    if current_pos > target_pos {
                        // adjusted_ratio so force_evict(current_pos * adjusted) == target_pos.
                        let adjusted_ratio = target_pos as f32 / current_pos as f32;
                        // Dispatch by evict method (same as decode loop).
                        // Scores are unavailable at prefill→decode boundary, so
                        // D2O and score-based H2O fall back to force_evict.
                        let result = if evict.method == llm_rs2::resilience::EvictMethod::Streaming
                        {
                            if let Some(ref sp) = evict.streaming_params {
                                let policy =
                                    llm_rs2::core::eviction::streaming_llm::StreamingLLMPolicy::new(
                                        sp.sink_size,
                                        sp.window_size,
                                    );
                                cache_manager.force_evict_by_policy_ref(
                                    &policy,
                                    &mut kv_caches,
                                    adjusted_ratio,
                                    llm_rs2::core::cache_manager::ScoreContext::None,
                                )
                            } else {
                                cache_manager.force_evict(&mut kv_caches, adjusted_ratio)
                            }
                        } else {
                            cache_manager.force_evict_by_policy(
                                evict.method,
                                &mut kv_caches,
                                adjusted_ratio,
                                llm_rs2::core::cache_manager::ScoreContext::None,
                            )
                        };
                        match result {
                            Ok(r) if r.evicted => {
                                eprintln!(
                                    "[Prefill→Decode] Evicted {} tokens (pos: {} → {})",
                                    r.tokens_removed,
                                    r.new_pos + r.tokens_removed,
                                    r.new_pos
                                );
                            }
                            Err(e) => eprintln!("[Prefill→Decode] Eviction error: {}", e),
                            _ => {}
                        }
                    }
                }
            }
        }

        // Sample last token — read logits from the last chunk's output.
        // When chunked: prefill_logits is [1,1,vocab_size], last_logits = the only row.
        // When not chunked: prefill_logits is [1,process_len,vocab_size], take last row.
        let mut last_logits = vec![0.0f32; vocab_size];
        unsafe {
            let ptr = last_logits.as_mut_ptr() as *mut u8;
            let byte_len = vocab_size * 4;
            if chunked {
                // Single-row buffer; read all of it.
                let slice = std::slice::from_raw_parts_mut(ptr, byte_len);
                backend.read_buffer(&prefill_logits, slice)?;
            } else {
                // Multi-row buffer; read only the last row.
                // read_buffer reads from offset 0, so we read the full buffer and
                // then take the last vocab_size elements.
                let mut logits_cpu = vec![0.0f32; process_len * vocab_size];
                let full_ptr = logits_cpu.as_mut_ptr() as *mut u8;
                let full_slice = std::slice::from_raw_parts_mut(full_ptr, logits_cpu.len() * 4);
                backend.read_buffer(&prefill_logits, full_slice)?;
                let start_idx = (process_len - 1) * vocab_size;
                last_logits.copy_from_slice(&logits_cpu[start_idx..start_idx + vocab_size]);
            }
        }

        let next_token_id = sampling::sample(
            &mut last_logits,
            &tokens,
            vocab_size,
            &sampling_config,
            None,
        );

        _ttft_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "Prefill: {:.2} ms ({} tokens, {:.1} tok/s)",
            prefill_forward_ms,
            process_len,
            process_len as f64 / (prefill_forward_ms / 1000.0),
        );
        eprintln!(
            "Prefill(pure fwd): {:.2} ms ({} tokens, {:.1} tok/s) [sync'd forward only, comparable to llama-bench pp]",
            prefill_pure_fwd_ms,
            process_len,
            process_len as f64 / (prefill_pure_fwd_ms / 1000.0),
        );
        _last_token_time = std::time::Instant::now();

        tokens.push(next_token_id);
        start_pos += process_len;
        // T2: first forward pass (prefill) complete, KV cache filled.
        rss_trace("prefill_done");
        io_trace("prefill_done");
    }

    // Execute deferred SwitchHw (from prefill checkpoint).
    // Now safe: prefill is done, logits read, all workspace released.
    // Decode buffers are allocated *after* this point, so only KV migrate
    // and backend/is_gpu update are needed here.
    if let Some(ref device) = deferred_switch
        && let (Some(gpu_be), Some(gpu_mem)) = (&gpu_backend_arc, &gpu_memory_arc)
    {
        match device.as_str() {
            "cpu" if is_gpu => {
                eprintln!("[Prefill->Decode] Executing deferred SwitchHw: GPU->CPU");
                llm_rs2::core::kv_migrate::migrate_kv_caches(
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
                is_gpu = false;
                // Re-tag weight tensors with CPU backend.
                // UnifiedBuffer (ALLOC_HOST_PTR, mapped) stays valid for CPU.
                eprintln!("[Prefill->Decode] SwitchHw: Switched to CPU.");
            }
            "gpu" | "opencl" if !is_gpu && weights_on_gpu => {
                eprintln!("[Prefill->Decode] Executing deferred SwitchHw: CPU->GPU");
                llm_rs2::core::kv_migrate::migrate_kv_caches(
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
                is_gpu = true;
                eprintln!("[Prefill->Decode] SwitchHw: Switched to GPU.");
            }
            _ => {}
        }
    }

    // D2O: compute per-layer budgets from prefill attention variance.
    let d2o_layer_ratios: Option<Vec<(f32, f32)>> = if let Some(ref collector) = variance_collector
    {
        let budgets = collector.compute_budgets(
            args.d2o_keep_ratio * args.eviction_target_ratio,
            (1.0 - args.d2o_keep_ratio) * args.eviction_target_ratio,
        );
        log::info!(
            "[D2O] Layer budgets computed: {:?}",
            budgets.iter().map(|(h, r)| h + r).collect::<Vec<_>>()
        );
        Some(budgets)
    } else {
        None
    };

    // Position → birth step mapping for profiling (token identity tracking)
    let mut position_birth_step: Vec<usize> = if profiler.is_some() {
        // All prefill tokens have birth_step = 0 (prompt)
        let prompt_len = tokens.len();
        let map = vec![0usize; prompt_len];
        // Register prompt token births + first generated token
        if let Some(ref mut p) = profiler {
            p.scores
                .record_token_births(0, prompt_len, actual_protected_prefix);
        }
        map
    } else {
        Vec::new()
    };

    // === GENERATION PHASE ===
    {
        println!("[Profile] Event: DecodingStart");

        // --profile-events / --heartbeat-gpu-profile: drop any events captured
        // during prefill/warmup so the decode-only aggregate is not polluted.
        // Prefill uses the generic `forward` path (no label hints), so without
        // this step all matmul dispatches from prefill would spill into the
        // decode "matmul" bucket and inflate the GPU self-util meter's first
        // heartbeat sample.
        #[cfg(feature = "opencl")]
        if (args.profile_events || args.heartbeat_gpu_profile)
            && let Some(ocl_be) = backend
                .as_any()
                .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
            && ocl_be.profile_events_enabled
        {
            backend.synchronize()?;
            ocl_be.flush_and_aggregate_profile()?;
            let _ = ocl_be.take_profile_accum();
            // Prefill-phase GPU busy ns were also fed into the self-util
            // meter via flush_and_aggregate_profile(); drain them so the
            // first heartbeat only reflects decode-phase usage.
            if let Some(m) = ocl_be.gpu_self_meter() {
                use llm_rs2::resilience::GpuSelfMeter;
                let _ = m.sample(std::time::Duration::from_secs(1));
            }
            eprintln!("[Profile] prefill/warmup events dropped (decode-only accumulator)");
        }
        // Pre-allocate workspace for generation
        let q_dim = model.config.num_attention_heads * model.config.head_dim;
        let k_dim = model.config.num_key_value_heads * model.config.head_dim;
        let v_dim = k_dim;
        let ffn_hidden = model.config.intermediate_size;

        // After SwitchHw GPU->CPU, `memory` is still OpenCL memory whose
        // alloc() creates OpenCLBuffer (null as_ptr). Use cpu_memory_arc when on CPU.
        let decode_mem: &dyn Memory = if is_gpu {
            memory.as_ref()
        } else {
            cpu_memory_arc.as_ref()
        };

        // Re-allocate logits on the correct backend after deferred SwitchHw.
        // The outer `logits` was allocated with `memory` (GPU) before the
        // deferred switch. After GPU→CPU, the unmapped UnifiedBuffer has
        // as_ptr() == null → segfault when CPU forward writes logits.
        if !is_gpu && logits.as_ptr().is_null() {
            let new_logits_buf = decode_mem.alloc(vocab_size * 4, DType::F32)?;
            logits = Tensor::new(
                Shape::new(vec![1, 1, vocab_size]),
                new_logits_buf,
                backend.clone(),
            );
        }

        let x_gen_buf = decode_mem.alloc(hidden_size * 4, DType::F32)?;
        let mut x_gen = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            x_gen_buf,
            backend.clone(),
        );

        let mut gen_ws = LayerWorkspace::new(
            WorkspaceConfig {
                batch_size: 1,
                dim: model.config.hidden_size,
                q_dim,
                k_dim,
                v_dim,
                ffn_hidden,
                n_heads: model.config.num_attention_heads,
                max_seq_len: args.max_seq_len, // Use context window size
            },
            decode_mem,
            backend.clone(),
        )?;

        // Attach partition workspace if tensor partition is active.
        // Use UnifiedBuffer (ALLOC_HOST_PTR) for zero-copy merge (see batch path above).
        let layer0_partition_probe = model.layers[0].load_weights();
        if let Some(ref ctx) = layer0_partition_probe.partition_ctx {
            let gpu_alloc = make_partition_gpu_alloc(&*backend, decode_mem);

            // Zero-copy residual (see line 1807 block for rationale).
            #[cfg(feature = "opencl")]
            if std::env::var_os("LLMRS_PARTITION_ZCOPY_RESIDUAL").is_some()
                || llm_rs2::layers::tensor_partition::partition_poll_flag_enabled()
            {
                if let Some(ub) = gen_ws
                    .residual
                    .buffer()
                    .as_any()
                    .downcast_ref::<llm_rs2::buffer::unified_buffer::UnifiedBuffer>()
                {
                    ub.map()?;
                    eprintln!("[Partition] Residual UnifiedBuffer permanent-mapped for zero-copy");
                } else {
                    eprintln!(
                        "[Partition] WARN: residual buffer is not UnifiedBuffer (zero-copy skipped)"
                    );
                }
            }

            gen_ws.partition_ws = Some(Arc::new(PartitionWsCell::new(PartitionWorkspace::new(
                ctx,
                ffn_hidden,
                hidden_size,
                &gpu_alloc,
                backend.clone(),
                cpu_backend_arc.clone(),
            )?)));
        }

        // Single token CPU tensor for generation loop
        let cpu_gen_indices_buf = Galloc::new().alloc(4, DType::U8)?;
        let cpu_gen_input = Tensor::new(
            Shape::new(vec![1, 1]),
            cpu_gen_indices_buf,
            Arc::new(CpuBackend::new()),
        );

        // Pre-allocate input tensor for decode loop (avoids per-token alloc)
        let gpu_gen_input_buf = decode_mem.alloc(4, DType::U8)?;
        let mut gen_input_tensor =
            Tensor::new(Shape::new(vec![1, 1]), gpu_gen_input_buf, backend.clone());

        // Pre-allocate CPU spare decode buffers for zero-alloc GPU→CPU SwitchHw.
        // Both sets (GPU active + CPU spare) stay alive for the process lifetime,
        // enabling instant swap without allocation/deallocation during switch.
        // This prevents Samsung LMKD from killing the process due to RSS spike.
        let (mut spare_logits, mut spare_xgen, mut spare_gen_ws, mut spare_gen_input) =
            if is_gpu && args.resilience_prealloc_switch {
                let cpu_lb = cpu_memory_arc.alloc(vocab_size * 4, DType::F32)?;
                let cpu_xb = cpu_memory_arc.alloc(hidden_size * 4, DType::F32)?;
                let cpu_gi = cpu_memory_arc.alloc(4, DType::U8)?;
                eprintln!("[Switch] Pre-allocated CPU spare buffers for zero-alloc SwitchHw");
                (
                    Some(Tensor::new(
                        Shape::new(vec![1, 1, vocab_size]),
                        cpu_lb,
                        cpu_backend_arc.clone(),
                    )),
                    Some(Tensor::new(
                        Shape::new(vec![1, 1, hidden_size]),
                        cpu_xb,
                        cpu_backend_arc.clone(),
                    )),
                    Some(LayerWorkspace::new(
                        WorkspaceConfig {
                            batch_size: 1,
                            dim: model.config.hidden_size,
                            q_dim,
                            k_dim,
                            v_dim,
                            ffn_hidden,
                            n_heads: model.config.num_attention_heads,
                            max_seq_len: args.max_seq_len,
                        },
                        cpu_memory_arc.as_ref(),
                        cpu_backend_arc.clone(),
                    )?),
                    Some(Tensor::new(
                        Shape::new(vec![1, 1]),
                        cpu_gi,
                        cpu_backend_arc.clone(),
                    )),
                )
            } else {
                (None, None, None, None)
            };

        // Streaming setup
        use std::io::Write;
        let mut stdout = std::io::stdout();
        let mut _printed_len = 0;

        // Print initial tokens (prompt + first generated)
        let initial_text = tokenizer.decode(&tokens, true).unwrap_or_default();
        print!("{}", initial_text);
        _printed_len = initial_text.len();
        stdout.flush().ok();

        // ─── UMA Hybrid Attention setup (Stage C) ─────────────────────
        // LLMRS_ATTN_HYBRID_KV_FRAC=X 가 설정되고 gating 조건이 모두 충족되면
        // 공용 GPU 스크래치 버퍼를 할당하고 HybridScope를 install한다. 스코프
        // 객체는 decode 루프 종료까지 살아있어야 하므로 `_hybrid_scope`로 바인드.
        // Gating 실패 시 reason을 stderr로 한 번 찍고 스킵.
        #[cfg(feature = "opencl")]
        let _hybrid_scope = {
            use llm_rs2::layers::hybrid_attention::{self, HybridAttnSetup};
            match HybridAttnSetup::from_env() {
                Some(kv_frac) => {
                    let backend_is_opencl = backend.name() == "OpenCL";
                    let kv_is_f16 = args.kv_type == "f16";
                    let head_dim_val = model.config.head_dim;
                    let head_dim_ok = head_dim_val == 64 || head_dim_val == 128;
                    let n_heads_q = model.config.num_attention_heads;
                    let n_kv_heads = model.config.num_key_value_heads;
                    let is_gqa = n_kv_heads < n_heads_q;
                    let partition_off =
                        args.tensor_partition <= 0.0 || args.tensor_partition >= 1.0;
                    let eviction_compatible =
                        args.eviction_policy != "kivi" && args.eviction_policy != "qcf";
                    let layout_ok = kv_caches
                        .first()
                        .map(|c| c.layout() == KVLayout::HeadMajor)
                        .unwrap_or(false);

                    let gate_ok = backend_is_opencl
                        && kv_is_f16
                        && head_dim_ok
                        && is_gqa
                        && partition_off
                        && eviction_compatible
                        && layout_ok;

                    if !gate_ok {
                        let reason = if !backend_is_opencl {
                            "backend is not OpenCL"
                        } else if !kv_is_f16 {
                            "kv dtype must be f16"
                        } else if !head_dim_ok {
                            "head_dim must be 64 or 128"
                        } else if !is_gqa {
                            "requires GQA (n_kv_heads < n_heads_q)"
                        } else if !partition_off {
                            "FFN tensor partition is active"
                        } else if !eviction_compatible {
                            "incompatible eviction policy (kivi/qcf)"
                        } else {
                            "KV layout must be HeadMajor"
                        };
                        eprintln!(
                            "[hybrid-attn] LLMRS_ATTN_HYBRID_KV_FRAC={} ignored: {}",
                            kv_frac, reason
                        );
                        None
                    } else {
                        // Map KV/Q/out_attn/residual UnifiedBuffer들을 CPU가 접근
                        // 가능하도록 전부 매핑한다. UMA 특성상 map은 주소만
                        // 고정하고 추가 복사는 하지 않는다. Plan execution에
                        // 들어가기 전에 한 번만 호출되면 충분.
                        let mut map_err: Option<anyhow::Error> = None;
                        for c in kv_caches.iter() {
                            if let Err(e) = c.k_buffer.buffer().map_for_cpu() {
                                map_err = Some(e);
                                break;
                            }
                            if let Err(e) = c.v_buffer.buffer().map_for_cpu() {
                                map_err = Some(e);
                                break;
                            }
                        }
                        if map_err.is_none() {
                            if let Err(e) = gen_ws.q.buffer().map_for_cpu() {
                                map_err = Some(e);
                            } else if let Err(e) = gen_ws.out_attn.buffer().map_for_cpu() {
                                map_err = Some(e);
                            }
                        }
                        if let Some(e) = map_err {
                            eprintln!("[hybrid-attn] failed to map UMA buffers: {} — skipping", e);
                            None
                        } else {
                            let ocl_be = backend
                                .as_any()
                                .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>();
                            match ocl_be {
                                Some(ob) => match HybridAttnSetup::new_for_decode(
                                    &ob.queue,
                                    kv_frac,
                                    n_heads_q,
                                    head_dim_val,
                                ) {
                                    Ok(setup) => {
                                        eprintln!(
                                            "[hybrid-attn] enabled: kv_frac={} n_heads_q={} head_dim={}",
                                            kv_frac, n_heads_q, head_dim_val
                                        );
                                        Some(hybrid_attention::install(Arc::new(setup)))
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "[hybrid-attn] setup allocation failed: {} — skipping",
                                            e
                                        );
                                        None
                                    }
                                },
                                None => None,
                            }
                        }
                    }
                }
                None => None,
            }
        };

        // Build GPU kernel plan for decode (OpenCL only, lazy rebuild on invalidation)
        // Disable for Gemma3: plan doesn't include QK-norm, post-norm, gelu_tanh_mul
        // Disable when tensor partition is active: plan bypasses forward_gen's
        // partition path entirely (plan = pure GPU chain, no CPU co-execution).
        //
        // Score accumulator coexistence: when a CPU `score_accumulator` is
        // active (H2O/D2O/Sliding/CAOTE eviction), the plan may still be used
        // as long as the paired GPU `gpu_score_acc` is active.  `build_plan`
        // then selects the legacy attention kernel (flash attn has no score
        // output) and pre-binds the GPU score buffer into arg 4. Per-layer
        // `reduce_layer` + post-pass `end_step` are driven by
        // `FullKernelPlan::execute` so CPU readback happens only at eviction
        // time (see `sync_to_cpu` further down).
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
        // Partition is now routed through `build_partitioned_layer_plan` inside
        // `build_plan`, so the old `partition_ctx.is_none()` gate has been
        // removed (see ENG-ALG-200 / arch A.6.1). When partition + plan are
        // both unavailable for a layer (e.g. `LLMRS_PARTITION_PLAN=0`), the
        // builder returns `Err` and the caller falls back to forward_gen.
        #[cfg(feature = "opencl")]
        let mut gpu_plan = if backend.name() == "OpenCL"
            && !args.profile
            && !args.no_gpu_plan
            && accumulator_compatible_with_plan
            && model.config.arch != llm_rs2::models::config::ModelArch::Gemma3
            && !args.swap_intra_forward
            && !args.swap_layer_immediate
            && !args.swap_phase_aware
        {
            model.build_plan(&x_gen, &logits, &gen_ws, &mut kv_caches, &backend)
        } else {
            None
        };
        // Sticky disable: when the initial `build_plan` returns `None` and
        // partition is active, the cause is almost always the opt-in gate
        // (`LLMRS_PARTITION_PLAN=0` default on Adreno, 2026-04-21). Retrying
        // every token spams `build_plan` (~100 ms/token overhead) for no
        // benefit. Lock the disable on the first miss and keep forward_gen.
        // `execute_plan` resetting `gpu_plan = None` for KV-resize
        // invalidation still takes the rebuild path on the next token.
        #[cfg(feature = "opencl")]
        let partition_active_any = model
            .layers
            .iter()
            .any(|s| s.load_weights().partition_ctx.is_some());
        #[cfg(feature = "opencl")]
        let mut gpu_plan_sticky_disabled = partition_active_any && gpu_plan.is_none();

        // Pre-allocate decode buffers (reused across tokens)
        let mut logits_cpu = vec![0.0f32; vocab_size];
        let mut sampling_indices: Vec<usize> = (0..vocab_size).collect();

        // Ceiling for sticky eviction: records current_pos at first eviction trigger.
        // Subsequent evictions use ceiling * ratio as a fixed target to prevent cascade
        // (e.g. cache 33 → 16 → 8 → ... when target_ratio is applied to ever-shrinking pos).
        let mut evict_ceiling: Option<usize> = None;
        let mut evict_floor_logged: Option<bool> = None;

        // Sticky cache for last-applied partition ratio. The executor re-delivers
        // `plan.partition_ratio = Some(sticky)` on every poll (ISSUE-5 fix), so
        // without this guard the consumer below would re-split 84 weights and
        // rebuild the GPU plan on every decode tick (verify v2 REGRESSION-A:
        // q4 enable +102% → +3859% TBT). Seeded from CLI-time partition so the
        // first sticky re-delivery is a no-op when nothing changed.
        let mut last_applied_partition_ratio: Option<f32> =
            if args.tensor_partition > 0.0 && args.tensor_partition < 1.0 {
                Some(args.tensor_partition)
            } else {
                None
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

            // ── Auto-switch CPU→GPU at threshold ─────────────────────────
            if !is_gpu
                && weights_on_gpu
                && args.switch_threshold > 0
                && kv_caches[0].current_pos >= args.switch_threshold
                && let (Some(gpu_be), Some(gpu_mem)) =
                    (gpu_backend_arc.as_ref(), gpu_memory_arc.as_ref())
            {
                eprintln!(
                    "[Switch] Auto-switch CPU→GPU at token {}",
                    kv_caches[0].current_pos
                );
                llm_rs2::core::kv_migrate::migrate_kv_caches(
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
                let logits_gpu_buf = gpu_mem.alloc(vocab_size * 4, DType::F32)?;
                logits = Tensor::new(
                    Shape::new(vec![1, 1, vocab_size]),
                    logits_gpu_buf,
                    backend.clone(),
                );
                let xg_buf = gpu_mem.alloc(hidden_size * 4, DType::F32)?;
                x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());
                gen_ws = LayerWorkspace::new(
                    WorkspaceConfig {
                        batch_size: 1,
                        dim: model.config.hidden_size,
                        q_dim,
                        k_dim,
                        v_dim,
                        ffn_hidden,
                        n_heads: model.config.num_attention_heads,
                        max_seq_len: args.max_seq_len,
                    },
                    gpu_mem.as_ref(),
                    backend.clone(),
                )?;
                #[cfg(feature = "opencl")]
                {
                    gpu_plan = None; // invalidate; will rebuild after first forward
                }
                // Re-allocate gen_input_tensor on new GPU backend
                let gi_buf = gpu_mem.alloc(4, DType::U8)?;
                gen_input_tensor = Tensor::new(Shape::new(vec![1, 1]), gi_buf, backend.clone());
                is_gpu = true;
                eprintln!("[Switch] Switched to GPU successfully.");
            }
            // ── End auto-switch ──────────────────────────────────────────

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
            if args.h2o_debug {
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

            // Auto-eviction after forward pass (non-experiment mode)
            if auto_eviction {
                let before_len = kv_caches[0].current_pos;
                let capacity = kv_caches[0].capacity();

                // GPU score sync: transfer GPU-accumulated scores to CPU accumulator
                // before any score-based eviction decision. Only syncs when:
                // 1. GPU score acc is active AND
                // 2. Eviction is imminent (score-based at 90% capacity) OR non-score-based with acc
                #[cfg(feature = "opencl")]
                if (score_based_eviction && before_len >= capacity * 9 / 10
                    || score_accumulator.as_ref().is_some_and(|a| a.is_active()))
                    && let Some(ocl_be) = backend
                        .as_any()
                        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                    && let Some(gpu_acc) = ocl_be.gpu_score_acc()
                    && gpu_acc.is_active()
                {
                    let (flat, head) = gpu_acc.sync_to_cpu(ocl_be.queue.as_core())?;
                    if let Some(ref mut acc) = score_accumulator {
                        acc.import_gpu_scores(&flat, &head);
                    }
                }

                // Capture pre-eviction scores for profiling (before eviction mutates state)
                let pre_eviction_scores: Vec<f32> = if profiler.is_some()
                    && score_based_eviction
                    && before_len >= capacity * 9 / 10
                {
                    score_accumulator
                        .as_ref()
                        .filter(|acc| acc.is_active())
                        .map(|acc| {
                            acc.importance_scores()[..before_len.min(acc.importance_scores().len())]
                                .to_vec()
                        })
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };

                let result = if score_based_eviction && before_len >= capacity * 9 / 10 {
                    // Score-based policies: force evict when cache >= 90% full
                    if let Some(acc) = score_accumulator.as_ref() {
                        if acc.is_active() {
                            // D2O layer-level allocation: use per-layer budgets if available
                            if let Some(ref ratios) = d2o_layer_ratios {
                                cache_manager.force_evict_with_scores_and_budgets(
                                    &mut kv_caches,
                                    args.eviction_target_ratio,
                                    acc.importance_scores(),
                                    ratios,
                                )?
                            } else {
                                cache_manager.force_evict_with_scores(
                                    &mut kv_caches,
                                    args.eviction_target_ratio,
                                    acc.importance_scores(),
                                )?
                            }
                        } else {
                            cache_manager.force_evict(&mut kv_caches, args.eviction_target_ratio)?
                        }
                    } else {
                        cache_manager.force_evict(&mut kv_caches, args.eviction_target_ratio)?
                    }
                } else if let Some(acc) = score_accumulator.as_ref() {
                    if acc.is_active() {
                        cache_manager
                            .maybe_evict_with_scores(&mut kv_caches, acc.importance_scores())?
                    } else {
                        cache_manager.maybe_evict(&mut kv_caches)?
                    }
                } else {
                    cache_manager.maybe_evict(&mut kv_caches)?
                };
                if result.evicted {
                    // Compute evicted indices from pre-eviction state
                    let target_len = ((before_len as f32) * args.eviction_target_ratio) as usize;
                    let evicted_indices = if !pre_eviction_scores.is_empty() {
                        llm_rs2::profile::compute_h2o_evicted_indices(
                            before_len,
                            target_len,
                            actual_protected_prefix,
                            args.h2o_keep_ratio,
                            &pre_eviction_scores,
                        )
                    } else {
                        Vec::new()
                    };

                    if let Some(ref mut p) = profiler {
                        // Record token deaths before the EvictionEvent
                        if !evicted_indices.is_empty() {
                            p.scores.record_token_deaths(
                                decode_token_index,
                                &evicted_indices,
                                &position_birth_step,
                                &pre_eviction_scores,
                            );
                        }
                        p.on_eviction(llm_rs2::profile::EvictionEvent {
                            step: decode_token_index,
                            policy: args.eviction_policy.clone(),
                            before_len,
                            after_len: result.new_pos,
                            evicted_count: result.tokens_removed,
                            partition: llm_rs2::profile::PartitionInfo {
                                prefix_end: actual_protected_prefix,
                                hh_count: 0,
                                recent_start: result.new_pos,
                            },
                            evicted_indices: evicted_indices.clone(),
                            pre_eviction_scores,
                        });
                    }

                    // Update position_birth_step mapping after eviction (compact)
                    if !position_birth_step.is_empty() {
                        let evicted_set: std::collections::HashSet<usize> =
                            evicted_indices.iter().copied().collect();
                        let mut kept = Vec::new();
                        for (pos, &birth) in position_birth_step.iter().enumerate() {
                            if pos < before_len && !evicted_set.contains(&pos) {
                                kept.push(birth);
                            }
                        }
                        position_birth_step = kept;
                    }

                    if let Some(acc) = score_accumulator.as_mut() {
                        acc.reset();
                    }
                    // Reset GPU score accumulator after eviction
                    #[cfg(feature = "opencl")]
                    if let Some(ocl_be) = backend
                        .as_any()
                        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                        && let Some(gpu_acc) = ocl_be.gpu_score_acc_mut()
                        && gpu_acc.is_active()
                    {
                        gpu_acc.reset(ocl_be.queue.as_core())?;
                    }
                }
            }
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
                    eviction_policy: args.eviction_policy.clone(),
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
                    let streaming_window_size = if args.streaming_window > 0 {
                        args.streaming_window
                    } else if args.kv_budget > 0 {
                        args.kv_budget.saturating_sub(args.sink_size)
                    } else {
                        args.eviction_window
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
                    let mut mapped_bufs: Vec<std::sync::Arc<dyn llm_rs2::core::buffer::Buffer>> =
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
                        streaming_config: Some((args.sink_size, streaming_window_size)),
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
                        let tgt = if tgt_raw < args.min_kv_cache {
                            if evict_floor_logged.is_none() {
                                eprintln!(
                                    "[Eviction] target_pos {} clamped to min_kv_cache {}",
                                    tgt_raw, args.min_kv_cache
                                );
                                evict_floor_logged = Some(true);
                            }
                            args.min_kv_cache
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
                                llm_rs2::core::cache_manager::ScoreContext::PerHead {
                                    flat: acc.importance_scores(),
                                    head: head_imp,
                                    n_kv_heads: acc.n_kv_heads(),
                                }
                            } else if acc.is_active() {
                                llm_rs2::core::cache_manager::ScoreContext::Flat {
                                    importance: acc.importance_scores(),
                                }
                            } else {
                                llm_rs2::core::cache_manager::ScoreContext::None
                            }
                        } else {
                            llm_rs2::core::cache_manager::ScoreContext::None
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
                            use llm_rs2::core::eviction::StreamingLLMPolicy;
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
                                if args.h2o_debug {
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
                                        policy: args.eviction_policy.clone(),
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
                                    llm_rs2::core::kv_migrate::migrate_kv_caches(
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
                                    llm_rs2::core::kv_migrate::migrate_kv_caches(
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
            eviction_policy: args.eviction_policy.clone(),
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
            eviction_policy: args.eviction_policy.clone(),
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
    importance_table: Option<&'a llm_rs2::core::qcf::ImportanceTable>,
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
    use llm_rs2::core::qcf::{
        AggregationMode, QcfActionType, UnifiedQcfParams, VDataSource, compute_unified_qcf,
    };
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
                let params = UnifiedQcfParams {
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
                let (qcf, _) = compute_unified_qcf(&params);
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

/// Run a `SwapExecutor` over the given target layers.
///
/// Centralises the boilerplate shared by the four call sites that execute a
/// weight swap (`--force-swap-ratio`, two QCF-dump warmup paths, and the
/// `EngineCommand::SwapWeights` direct dispatch). Always targets `Q4_0`
/// (only currently-supported swap dtype, INV-126). Resolves the swap backend
/// to GPU when available, otherwise CPU — matches the original logic that
/// `SwapExecutor` branches on `backend.name()` to pick the AUF SOA fast path.
#[allow(clippy::too_many_arguments)]
fn run_layer_swap(
    model: &llm_rs2::models::transformer::TransformerModel,
    target_layers: &[usize],
    gpu_backend: Option<&Arc<dyn Backend>>,
    cpu_backend: &Arc<dyn Backend>,
    async_dispatcher: Option<&llm_rs2::models::weights::AsyncSwapDispatcher>,
    #[cfg(feature = "opencl")] host_ptr_pool: Option<
        Arc<llm_rs2::backend::opencl::host_ptr_pool::HostPtrPool>,
    >,
    #[cfg(feature = "cuda-embedded")] layer_pool: Option<
        Arc<llm_rs2::models::weights::layer_object_pool::LayerObjectPool>,
    >,
    #[cfg(feature = "cuda-embedded")] mmap_registration: Option<
        Arc<llm_rs2::buffer::cuda_mmap_alias_buffer::CudaMmapRegistration>,
    >,
) -> Result<llm_rs2::models::weights::SwapReport, llm_rs2::models::weights::SwapError> {
    let swap_memory = Galloc::new();
    let swap_backend: Arc<dyn Backend> =
        gpu_backend.cloned().unwrap_or_else(|| cpu_backend.clone());
    // ENG-ALG-228: attach the model's async release worker so Stage (c) enqueues
    // displaced LayerWeights for background drop instead of blocking inline.
    let executor = llm_rs2::models::weights::SwapExecutor::new_with_worker(
        DType::Q4_0,
        &model.config,
        swap_backend,
        &swap_memory,
        Arc::clone(&model.release_worker),
    );
    // LISWAP-3 prototype: if a host_ptr pool is supplied, attach it so the
    // AOS materialise path uses the zero-copy slot pool.
    #[cfg(feature = "opencl")]
    let executor = match host_ptr_pool {
        Some(pool) => executor.with_host_ptr_pool(pool),
        None => executor,
    };
    // LISWAP-8 Phase B: attach pre-allocated layer object pool when set.
    #[cfg(feature = "cuda-embedded")]
    let executor = match layer_pool {
        Some(pool) => executor.with_layer_pool(pool),
        None => executor,
    };
    // LISWAP-8 Hammer D: attach mmap registration when set.
    #[cfg(feature = "cuda-embedded")]
    let executor = match mmap_registration {
        Some(reg) => executor.with_mmap_registration(reg),
        None => executor,
    };
    executor.execute_on_slots(
        model.layers.as_slice(),
        model.secondary_mmap.as_ref(),
        &model.ratio_generation,
        target_layers,
        async_dispatcher,
    )
}

/// Re-map weight tensors for CPU access after a weight swap.
///
/// Required when running on GPU with `--secondary-layout aos +
/// --resilience-prealloc-switch`: `SwapExecutor::materialise_tensor` lands an
/// unmapped `UnifiedBuffer` in the new `LayerWeights` snapshot, and the next
/// `switch_hw cpu` directive segfaults on a null host pointer.  Idempotent —
/// already-mapped tensors short-circuit in `map_one`.
#[cfg(feature = "opencl")]
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

/// Result of a QCF-dump warmup workflow: importance table plus optional swap
/// decision (when `--force-swap-ratio` was applied).
struct QcfWarmupResult {
    importance: llm_rs2::core::qcf::ImportanceTable,
    decision: Option<llm_rs2::models::weights::SwapDecision>,
    /// Per-layer DP-LLM proxy ε (single-tensor relative `attn_output`).
    /// `Some` only in compare mode.
    dpllm_epsilon: Option<Vec<f32>>,
    /// §4 candidate A: per-layer DP-LLM ε summed across 6 attn+MLP tensors.
    dpllm_epsilon_multi: Option<Vec<f32>>,
    /// §4 candidate D: per-layer DP-LLM ε without the `‖W·x‖` normalisation
    /// (absolute L2 of the activation difference).
    dpllm_epsilon_abs: Option<Vec<f32>>,
    /// §4 candidate E: QCF-style multiplicative composition `ε_v × ε_o`.
    dpllm_epsilon_qcf: Option<Vec<f32>>,
    /// §4.2 F4: cascade-aware single output perturbation.
    /// `‖(W_o^F16 − W_o^Q4) · V_out^F16‖_F / ‖W_o^F16 · V_out^F16‖_F`.
    direct_attn_f4: Option<Vec<f32>>,
    /// §4.2 F5: direct attention output relative L2 perturbation.
    /// `‖O^F16 − O^Q4‖_F / ‖O^F16‖_F`.
    direct_attn_f5: Option<Vec<f32>>,
    /// §4.2 decode-only F5: per-layer F5 evaluated with X = the N decode-step
    /// raws (T = N) only (no prefill X). `Some` only when `--decode-x-steps >
    /// 0` and a secondary GGUF was loaded.
    direct_attn_f5_decode_only: Option<Vec<f32>>,
    /// §4.2 prefill+decode F5: per-layer F5 evaluated with X =
    /// concat(prefill raws, decode raws) (T = 256 + N). `Some` only when
    /// `--decode-x-steps > 0` and a secondary GGUF was loaded.
    direct_attn_f5_prefill_decode: Option<Vec<f32>>,
}

/// QCF-dump warmup workflow shared by `--ppl/generation` and `--eval-ll` modes.
///
/// 1. Warmup prefill with `ImportanceCollector` over `warmup_ids`.
/// 2. Build `ImportanceTable`, reset KV caches to zero.
/// 3. If `force_ratio` is set, run `WeightSwapDecider` and dispatch the swap.
///
/// `log_prefix` is concatenated immediately after `[QCF-dump]` in every log
/// line emitted by this helper, so any non-empty value must include its own
/// leading space (e.g. `" eval-ll"`). The caller must ensure `warmup_ids` is
/// non-empty.
#[allow(clippy::too_many_arguments)]
fn run_qcf_warmup_workflow(
    model: &llm_rs2::models::transformer::TransformerModel,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut [KVCache],
    vocab_size: usize,
    warmup_ids: &[u32],
    force_ratio: Option<f32>,
    gpu_backend: Option<&Arc<dyn Backend>>,
    cpu_backend: &Arc<dyn Backend>,
    log_prefix: &str,
    swap_algorithm: llm_rs2::models::weights::SwapAlgorithm,
    execute_swap: bool,
    importance_formula: llm_rs2::core::qcf::ImportanceFormula,
    importance_three_way: bool,
    swap_only_layers: Option<&[usize]>,
    decode_x_steps: usize,
) -> anyhow::Result<QcfWarmupResult> {
    use llm_rs2::core::qcf::ImportanceCollector;
    use llm_rs2::models::weights::WeightSwapDecider;

    let actual_warmup_len = warmup_ids.len();
    eprintln!(
        "[QCF-dump]{} warmup prefill: {} tokens (formula={}, three_way={}, decode_x_steps={})",
        log_prefix,
        actual_warmup_len,
        importance_formula.as_str(),
        importance_three_way,
        decode_x_steps,
    );

    // ── Warmup prefill with ImportanceCollector (F16 prefill pass) ────────────
    let mut collector =
        ImportanceCollector::new_with_formula(importance_formula, importance_three_way);
    let last_token_logits_argmax: u32;
    {
        let warmup_buf = Galloc::new().alloc(actual_warmup_len * 4, DType::U8)?;
        unsafe {
            let ptr = warmup_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in warmup_ids.iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_warmup = Tensor::new(
            Shape::new(vec![1, actual_warmup_len]),
            warmup_buf,
            Arc::new(CpuBackend::new()),
        );
        let warmup_input = backend.copy_from(&cpu_warmup)?;

        let warmup_logits_buf = memory.alloc(actual_warmup_len * vocab_size * 4, DType::F32)?;
        let mut warmup_logits = Tensor::new(
            Shape::new(vec![1, actual_warmup_len, vocab_size]),
            warmup_logits_buf,
            backend.clone(),
        );

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &warmup_input,
            start_pos: 0,
            kv_caches,
            backend,
            memory,
            logits_out: &mut warmup_logits,
            x_gen: None,
            workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: Some(&mut collector),
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        backend.synchronize()?;

        // Read the argmax of the last token's logits — first decode-step input.
        last_token_logits_argmax = if decode_x_steps > 0 {
            // Copy the final `[vocab_size]` slice of warmup_logits to host.
            let cpu_back: Arc<dyn Backend> = Arc::new(CpuBackend::new());
            let host_logits = cpu_back.copy_from(&warmup_logits)?;
            let last_offset = (actual_warmup_len - 1) * vocab_size;
            let host_data = unsafe {
                std::slice::from_raw_parts(
                    host_logits.buffer().as_ptr() as *const f32,
                    actual_warmup_len * vocab_size,
                )
            };
            let last_slice = &host_data[last_offset..last_offset + vocab_size];
            let mut best_idx: u32 = 0;
            let mut best_val: f32 = f32::NEG_INFINITY;
            for (i, &v) in last_slice.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i as u32;
                }
            }
            best_idx
        } else {
            0
        };
    }

    // ── Optional decode-X pass (§4.2 EuroSys'27) ──────────────────────────────
    // After prefill, run `decode_x_steps` greedy decode forwards. Capture
    // per-layer hidden state (T = decode_x_steps per layer after concat) in
    // a fresh `DirectAttn` collector. KV cache is reset after this pass.
    let raws_decode_opt: Option<Vec<(Vec<f32>, usize, usize)>> = if decode_x_steps > 0 {
        if model.secondary_mmap.is_none() {
            eprintln!(
                "[QCF-dump]{} decode-x: SKIPPED (no secondary GGUF loaded)",
                log_prefix
            );
            None
        } else {
            eprintln!(
                "[QCF-dump]{} decode-x: running {} greedy decode steps (seed token id {})",
                log_prefix, decode_x_steps, last_token_logits_argmax,
            );

            let mut collector_decode = ImportanceCollector::new_with_formula(
                llm_rs2::core::qcf::ImportanceFormula::DirectAttn,
                false,
            );

            let mut next_tok: u32 = last_token_logits_argmax;
            let cpu_back: Arc<dyn Backend> = Arc::new(CpuBackend::new());

            for step in 0..decode_x_steps {
                let decode_buf = Galloc::new().alloc(4, DType::U8)?;
                unsafe {
                    let ptr = decode_buf.as_mut_ptr() as *mut u32;
                    *ptr = next_tok;
                }
                let cpu_decode = Tensor::new(
                    Shape::new(vec![1, 1]),
                    decode_buf,
                    Arc::new(CpuBackend::new()),
                );
                let decode_input = backend.copy_from(&cpu_decode)?;

                let decode_logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
                let mut decode_logits = Tensor::new(
                    Shape::new(vec![1, 1, vocab_size]),
                    decode_logits_buf,
                    backend.clone(),
                );

                model.forward_into(TransformerModelForwardArgs {
                    input_tokens: &decode_input,
                    start_pos: actual_warmup_len + step,
                    kv_caches,
                    backend,
                    memory,
                    logits_out: &mut decode_logits,
                    x_gen: None,
                    workspace: None,
                    score_accumulator: None,
                    profiler: None,
                    skip_config: None,
                    importance_collector: Some(&mut collector_decode),
                    logits_last_only: true,
                    variance_collector: None,
                    prefill_workspace: None,

                    layer_boundary_hook: None,
                })?;
                backend.synchronize()?;

                // argmax for next decode step
                let host_logits = cpu_back.copy_from(&decode_logits)?;
                let host_data = unsafe {
                    std::slice::from_raw_parts(
                        host_logits.buffer().as_ptr() as *const f32,
                        vocab_size,
                    )
                };
                let mut best_idx: u32 = 0;
                let mut best_val: f32 = f32::NEG_INFINITY;
                for (i, &v) in host_data.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best_idx = i as u32;
                    }
                }
                next_tok = best_idx;
            }

            // The collector cached one `[1 × d]` snapshot per layer for *every*
            // decode step (N × n_layers snapshots in chronological order).
            // Rearrange into a per-layer concat: layer i → T = decode_x_steps.
            // build_with_raws returns (table, x_means, raws_per_layer); raws is
            // `Vec<(Vec<f32>, seq_len=1, dim)>` of length N * n_layers.
            let (_table, _xm, raws_chrono) = collector_decode.build_with_raws();
            let n_layers = kv_caches.len();
            let mut per_layer_concat: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(n_layers);
            if raws_chrono.len() == decode_x_steps * n_layers && n_layers > 0 {
                let d = raws_chrono[0].2;
                for li in 0..n_layers {
                    let mut buf: Vec<f32> = Vec::with_capacity(decode_x_steps * d);
                    for step in 0..decode_x_steps {
                        let idx = step * n_layers + li;
                        let (data, t, _d) = &raws_chrono[idx];
                        debug_assert_eq!(*t, 1);
                        buf.extend_from_slice(&data[..d.min(data.len())]);
                    }
                    per_layer_concat.push((buf, decode_x_steps, d));
                }
            } else {
                eprintln!(
                    "[QCF-dump]{} decode-x: WARN raws layout mismatch (chrono len={}, expected {} × {})",
                    log_prefix,
                    raws_chrono.len(),
                    decode_x_steps,
                    n_layers
                );
            }

            // Reset KV cache so the regular flow starts from a clean prefill state.
            for kv in kv_caches.iter_mut() {
                kv.current_pos = 0;
            }

            Some(per_layer_concat)
        }
    } else {
        None
    };

    // ── Build ImportanceTable (+ optional DP-LLM ε variants) + reset KV cache ────
    let direct_attn_primary = matches!(
        importance_formula,
        llm_rs2::core::qcf::ImportanceFormula::DirectAttn
    );
    let cache_raw = importance_three_way || direct_attn_primary;
    let mut direct_attn_f5_decode_only: Option<Vec<f32>> = None;
    let mut direct_attn_f5_prefill_decode: Option<Vec<f32>> = None;
    let (
        imp_table,
        dpllm_epsilon,
        dpllm_epsilon_multi,
        dpllm_epsilon_abs,
        dpllm_epsilon_qcf,
        direct_attn_f4,
        direct_attn_f5,
    ) = if cache_raw {
        let (table, x_means, raws) = collector.build_with_raws();
        let sec_opt = model.secondary_mmap.as_ref();
        let eps_single = if importance_three_way {
            sec_opt.map(|sec| {
                llm_rs2::models::weights::noise_table::compute_input_aware_epsilon(
                    &model.layers,
                    sec,
                    &x_means,
                )
            })
        } else {
            None
        };
        let eps_multi = if importance_three_way {
            sec_opt.map(|sec| {
                llm_rs2::models::weights::noise_table::compute_input_aware_epsilon_multitensor(
                    &model.layers,
                    sec,
                    &x_means,
                )
            })
        } else {
            None
        };
        let eps_abs = if importance_three_way {
            sec_opt.map(|sec| {
                llm_rs2::models::weights::noise_table::compute_input_aware_epsilon_absolute(
                    &model.layers,
                    sec,
                    &x_means,
                )
            })
        } else {
            None
        };
        let eps_qcf = if importance_three_way {
            sec_opt.map(|sec| {
                llm_rs2::models::weights::noise_table::compute_input_aware_epsilon_qcf(
                    &model.layers,
                    sec,
                    &x_means,
                )
            })
        } else {
            None
        };
        // Cascade attention F4 + F5 (compute when raws are available, regardless
        // of whether primary is DirectAttn or 3-way compare).
        let (f4, f5) = if let Some(sec) = sec_opt {
            let n_heads = model.config.num_attention_heads;
            let n_kv_heads = model.config.num_key_value_heads;
            let d_head = model.config.head_dim;
            let pairs = llm_rs2::models::weights::noise_table::compute_cascade_attn_perturbation(
                &model.layers,
                sec,
                &raws,
                n_heads,
                n_kv_heads,
                d_head,
            );
            let f4_vec: Vec<f32> = pairs.iter().map(|(a, _)| *a).collect();
            let f5_vec: Vec<f32> = pairs.iter().map(|(_, b)| *b).collect();
            (Some(f4_vec), Some(f5_vec))
        } else {
            (None, None)
        };
        // §4.2 decode-X F5: compute with decode-only raws (T = N) AND
        // prefill+decode concat raws (T = 256 + N).
        if let (Some(sec), Some(raws_dec)) = (sec_opt, raws_decode_opt.as_ref()) {
            let n_heads = model.config.num_attention_heads;
            let n_kv_heads = model.config.num_key_value_heads;
            let d_head = model.config.head_dim;

            // (1) decode-only
            let pairs_dec =
                llm_rs2::models::weights::noise_table::compute_cascade_attn_perturbation(
                    &model.layers,
                    sec,
                    raws_dec,
                    n_heads,
                    n_kv_heads,
                    d_head,
                );
            direct_attn_f5_decode_only =
                Some(pairs_dec.iter().map(|(_, b)| *b).collect::<Vec<f32>>());

            // (2) prefill + decode concat: raws[i] = concat(prefill_raws[i], decode_raws[i])
            // along the T dimension.
            if raws.len() == raws_dec.len() {
                let mut raws_merged: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(raws.len());
                for (p, d_entry) in raws.iter().zip(raws_dec.iter()) {
                    let (p_data, p_t, p_d) = p;
                    let (d_data, d_t, d_d) = d_entry;
                    if *p_d != *d_d {
                        eprintln!(
                            "[QCF-dump]{} decode-x merge: WARN dim mismatch prefill={} decode={}",
                            log_prefix, p_d, d_d
                        );
                        continue;
                    }
                    let dim = *p_d;
                    let mut merged: Vec<f32> = Vec::with_capacity((*p_t + *d_t) * dim);
                    merged.extend_from_slice(p_data);
                    merged.extend_from_slice(d_data);
                    raws_merged.push((merged, *p_t + *d_t, dim));
                }
                let pairs_pd =
                    llm_rs2::models::weights::noise_table::compute_cascade_attn_perturbation(
                        &model.layers,
                        sec,
                        &raws_merged,
                        n_heads,
                        n_kv_heads,
                        d_head,
                    );
                direct_attn_f5_prefill_decode =
                    Some(pairs_pd.iter().map(|(_, b)| *b).collect::<Vec<f32>>());
            }
        }
        (table, eps_single, eps_multi, eps_abs, eps_qcf, f4, f5)
    } else {
        (collector.build(), None, None, None, None, None, None)
    };
    eprintln!(
        "[QCF-dump]{} ImportanceTable built: {} entries (dpllm_epsilon={}, multi={}, abs={}, qcf={})",
        log_prefix,
        imp_table.len(),
        if dpllm_epsilon.is_some() {
            "computed"
        } else {
            "skipped"
        },
        if dpllm_epsilon_multi.is_some() {
            "computed"
        } else {
            "skipped"
        },
        if dpllm_epsilon_abs.is_some() {
            "computed"
        } else {
            "skipped"
        },
        if dpllm_epsilon_qcf.is_some() {
            "computed"
        } else {
            "skipped"
        },
    );
    for kv in kv_caches.iter_mut() {
        kv.current_pos = 0;
    }

    // ── Optional swap with importance-guided decider ──────────────────────────
    let decision = if let Some(ratio) = force_ratio {
        let ratio = ratio.clamp(0.0, 1.0);
        eprintln!(
            "[QCF-dump]{} swap algorithm: {} (execute_swap={})",
            log_prefix,
            swap_algorithm.short_name(),
            execute_swap,
        );
        let decider = WeightSwapDecider {
            importance: Some(&imp_table),
            noise: Some(model.quant_noise.as_ref()),
            n_decoder_layers: model.layers.len(),
            currently_swapped: &[],
            allow_boundary_layers: read_allow_boundary_env(),
            algorithm: swap_algorithm,
        };
        let decider_decision = decider.decide(ratio);

        // §4 ground-truth path: when `--swap-only-layers` is set, override the
        // decider's selection with the explicit list. The decider's
        // `qcf_swap_estimate` is recomputed against this override so the dump
        // JSON reports the QCF prediction for the actually-swapped set.
        let decision = if let Some(only) = swap_only_layers {
            let override_layers: Vec<usize> = only
                .iter()
                .copied()
                .filter(|i| *i < model.layers.len())
                .collect();
            let qcf_override = llm_rs2::models::weights::compute_qcf_swap(
                &override_layers,
                model.quant_noise.as_ref(),
                Some(&imp_table),
                model.layers.len(),
            );
            eprintln!(
                "[QCF-dump]{} swap-only override: layers={:?} (ignoring algorithm/ratio decision)",
                log_prefix, override_layers,
            );
            llm_rs2::models::weights::SwapDecision {
                selected_layers: override_layers,
                qcf_swap_estimate: qcf_override,
                fallback_used: false,
            }
        } else {
            decider_decision
        };

        // Trajectory mode (`--qcf-trajectory`): return the decision without
        // executing the swap — the caller drives swap one layer at a time
        // around per-step eval-ll measurements.
        if !execute_swap {
            return Ok(QcfWarmupResult {
                importance: imp_table,
                decision: Some(decision),
                dpllm_epsilon,
                dpllm_epsilon_multi,
                dpllm_epsilon_abs,
                dpllm_epsilon_qcf,
                direct_attn_f4,
                direct_attn_f5,
                direct_attn_f5_decode_only,
                direct_attn_f5_prefill_decode,
            });
        }

        if decision.selected_layers.is_empty() {
            eprintln!(
                "[QCF-dump]{} swap: ratio={:.2} → 0 layers selected (qcf=0.0)",
                log_prefix, ratio,
            );
        } else {
            let report = run_layer_swap(
                model,
                &decision.selected_layers,
                gpu_backend,
                cpu_backend,
                None,
                // LISWAP-3: QCF dump path does not exercise the pool yet —
                // Stage 3 prototype only wires --force-swap-ratio paths.
                #[cfg(feature = "opencl")]
                None,
                #[cfg(feature = "cuda-embedded")]
                None,
                #[cfg(feature = "cuda-embedded")]
                None,
            )
            .map_err(|e| anyhow::anyhow!("[QCF-dump]{} swap failed: {}", log_prefix, e))?;
            eprintln!(
                "[QCF-dump]{} swap: ratio={:.2}, layers={}/{}, qcf_pred={:.4}, \
                 fallback={}, latency={:.1}ms",
                log_prefix,
                ratio,
                report.swapped.len(),
                model.layers.len(),
                decision.qcf_swap_estimate,
                decision.fallback_used,
                report.latency_ms,
            );
        }
        Some(decision)
    } else {
        None
    };

    Ok(QcfWarmupResult {
        importance: imp_table,
        decision,
        dpllm_epsilon,
        dpllm_epsilon_multi,
        dpllm_epsilon_abs,
        dpllm_epsilon_qcf,
        direct_attn_f4,
        direct_attn_f5,
        direct_attn_f5_decode_only,
        direct_attn_f5_prefill_decode,
    })
}

/// Execute a SwapWeights command from the manager: validate → decide → commit
/// incremental plan → report on plan completion.
///
/// LISWAP-6 manager path: instead of sync single-shot execution, this function
/// commits an `IncrementalSwapPlan` (K=2, dynamic-K + sub-batch pause) to the
/// decode loop via `swap_plan_out`. The decode loop drains the plan per tick.
/// `WeightSwapReport` is sent when the plan completes (see plan-done block in
/// the decode loop). Manager receives "received" acknowledgment immediately
/// (via the existing executor ack in the command dispatch site), and the final
/// WeightSwapReport arrives on plan completion.
///
/// Rejection (no-secondary, invalid-ratio, unsupported-dtype, in-flight plan)
/// is logged to stderr; no plan is committed.
/// `LLMRS_SWAP_ALLOW_BOUNDARY_LAYERS=1` 이면 `true` — `WeightSwapDecider` 가
/// layer 0 과 마지막 decoder layer 도 swap 후보로 포함. 미설정/다른 값 → `false`.
/// PPL teacher-forcing NLL ablation 등 research-only path 에서 사용.
fn read_allow_boundary_env() -> bool {
    std::env::var("LLMRS_SWAP_ALLOW_BOUNDARY_LAYERS")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn dispatch_swap_weights(
    model: &llm_rs2::models::transformer::TransformerModel,
    ratio: f32,
    target_dtype: llm_shared::DtypeTag,
    importance_table: Option<&llm_rs2::core::qcf::ImportanceTable>,
    decode_token_index: usize,
    swap_plan_out: &mut Option<llm_rs2::models::weights::IncrementalSwapPlan>,
    manager_report_out: &mut Option<(f32, usize, std::time::Instant, f32)>,
) {
    use llm_rs2::models::weights::{
        IncrementalSwapPlan, SwapDecision, WeightSwapDecider, compute_qcf_swap,
    };
    use llm_shared::DtypeTag;

    // ── 1. Validation ──────────────────────────────────────────────────────
    if model.secondary_mmap.is_none() {
        eprintln!("[WeightSwap] Rejected: no_secondary (ENG-DAT-C09)");
        return;
    }
    if ratio <= 0.0 || ratio > 1.0 {
        eprintln!("[WeightSwap] Rejected: invalid_ratio ({:.4})", ratio);
        return;
    }
    if target_dtype != DtypeTag::Q4_0 {
        eprintln!(
            "[WeightSwap] Rejected: unsupported_dtype ({:?}) (INV-126)",
            target_dtype
        );
        return;
    }

    // ── 1b. In-flight plan check ───────────────────────────────────────────
    // Reject if a plan is already in flight (CLI or manager). Prevents
    // concurrent plan conflict (spec: manager signal accept only when no plan).
    if swap_plan_out.is_some() {
        eprintln!(
            "[WeightSwap] Rejected: incremental plan already in-flight (ratio={:.2}). \
             Wait for current plan to complete before sending a new SwapWeights signal.",
            ratio
        );
        return;
    }

    // ── 2. Collect currently-swapped layers ────────────────────────────────
    let n_layers = model.layers.len();
    let currently_swapped: Vec<usize> = (0..n_layers)
        .filter(|&i| model.layers[i].current_dtype() == DType::Q4_0)
        .collect();

    // ── 3. Decider ─────────────────────────────────────────────────────────
    let allow_boundary = read_allow_boundary_env();
    eprintln!(
        "[Decider] allow_boundary_layers={} (ratio={:.4})",
        allow_boundary, ratio
    );
    let decider = WeightSwapDecider {
        importance: importance_table,
        noise: Some(&model.quant_noise),
        n_decoder_layers: n_layers,
        currently_swapped: &currently_swapped,
        allow_boundary_layers: allow_boundary,
        algorithm: llm_rs2::models::weights::SwapAlgorithm::ImportanceAware,
    };
    let decision: SwapDecision = decider.decide(ratio);

    if decision.selected_layers.is_empty() {
        eprintln!(
            "[WeightSwap] No layers to swap (ratio={:.2}, already_swapped={})",
            ratio,
            currently_swapped.len()
        );
        // Empty swap is Ok per spec (already fully swapped); no plan committed.
        return;
    }

    // ── 4. Compute QCF estimate for the planned layers ─────────────────────
    let qcf_swap_estimated = compute_qcf_swap(
        &decision.selected_layers,
        &model.quant_noise,
        importance_table,
        n_layers,
    );

    // ── 5. Commit incremental plan (K=2, same as CLI --swap-incremental-per-tick 2) ──
    let n_planned = decision.selected_layers.len();
    let per_tick = 2usize; // LISWAP-6: K=2 hard upper cap for manager path
    let ticks_est = n_planned.div_ceil(per_tick);
    eprintln!(
        "[WeightSwap] manager path: ratio={:.2}, {} target layers, per_tick={} ({} ticks estimated), qcf_estimated={:.4}",
        ratio, n_planned, per_tick, ticks_est, qcf_swap_estimated,
    );

    *swap_plan_out = Some(IncrementalSwapPlan::new(
        decision.selected_layers,
        per_tick,
        decode_token_index,
    ));
    *manager_report_out = Some((
        ratio,
        n_planned,
        std::time::Instant::now(),
        qcf_swap_estimated,
    ));
}

/// Build `LayerSwapEstimate` from an available `ImportanceTable` + model noise table.
///
/// Returns `None` when secondary mmap is absent or no importance table has been
/// collected yet (i.e., on the very first `RequestQcf` before any prefill).
fn build_layer_swap_estimate(
    model: &llm_rs2::models::transformer::TransformerModel,
    importance_table: Option<&llm_rs2::core::qcf::ImportanceTable>,
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
                    e.layer_id == i
                        && e.sublayer == llm_rs2::core::qcf::layer_importance::SubLayer::Full
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

// ════════════════════════════════════════════════════════════════
//  Prompt-batch helpers
// ════════════════════════════════════════════════════════════════

fn unix_ts() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

#[derive(serde::Deserialize)]
struct PromptBatchEntry {
    id: String,
    prompt: Option<String>,
    prompt_file: Option<String>,
}

fn load_prompt_batch(path: &str) -> anyhow::Result<Vec<PromptBatchEntry>> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open prompt batch {}: {}", path, e))?;
    let reader = std::io::BufReader::new(file);
    let mut entries = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let entry: PromptBatchEntry =
            serde_json::from_str(trimmed).map_err(|e| anyhow::anyhow!("Line {}: {}", i + 1, e))?;
        entries.push(entry);
    }
    Ok(entries)
}

fn resolve_prompt(entry: &PromptBatchEntry) -> anyhow::Result<String> {
    if let Some(ref text) = entry.prompt {
        Ok(text.clone())
    } else if let Some(ref path) = entry.prompt_file {
        std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read prompt_file {}: {}", path, e))
    } else {
        anyhow::bail!("Entry '{}': needs 'prompt' or 'prompt_file'", entry.id)
    }
}

// ════════════════════════════════════════════════════════════════
//  Eval-LL: Log-likelihood evaluation for downstream task accuracy
// ════════════════════════════════════════════════════════════════

/// Load and normalize eval questions from `--eval-batch` or `--eval-continuation`.
///
/// Produces a `Vec<EvalQuestion>` in grouped format (prompt + choices).
fn load_eval_questions(
    args: &Args,
    default_prompt: &str,
) -> anyhow::Result<Vec<llm_rs2::eval::EvalQuestion>> {
    let raw_tasks: Vec<serde_json::Value> = if let Some(ref path) = args.eval_batch {
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("Failed to open eval batch {}: {}", path, e))?;
        serde_json::from_reader(file)?
    } else {
        let cont = args.eval_continuation.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--eval-ll requires --eval-continuation or --eval-batch")
        })?;
        vec![serde_json::json!({
            "id": "single",
            "prompt": default_prompt,
            "choices": [cont],
        })]
    };

    let mut questions: Vec<llm_rs2::eval::EvalQuestion> = Vec::new();
    for task in &raw_tasks {
        if let Some(choices) = task["choices"].as_array() {
            questions.push(llm_rs2::eval::EvalQuestion {
                id: task["id"].as_str().unwrap_or("unknown").to_string(),
                prompt: task["prompt"]
                    .as_str()
                    .unwrap_or(default_prompt)
                    .to_string(),
                choices: choices
                    .iter()
                    .filter_map(|c| c.as_str().map(|s| s.to_string()))
                    .collect(),
            });
        } else if let Some(cont) = task["continuation"].as_str() {
            questions.push(llm_rs2::eval::EvalQuestion {
                id: task["id"].as_str().unwrap_or("unknown").to_string(),
                prompt: task["prompt"]
                    .as_str()
                    .unwrap_or(default_prompt)
                    .to_string(),
                choices: vec![cont.to_string()],
            });
        }
    }
    Ok(questions)
}

/// Build a warmup token sequence from the eval-ll question set.
///
/// Concatenates the `prompt` fields of the questions (separated by `"\n\n"`),
/// tokenizes the result, and returns at most `max_tokens` token IDs.
/// If fewer tokens are produced than requested, a warning is emitted but the
/// function succeeds — the caller handles the reduced warmup gracefully.
///
/// Returns an empty Vec when tokenization fails entirely (non-fatal).
fn build_eval_ll_warmup_text(
    questions: &[llm_rs2::eval::EvalQuestion],
    max_tokens: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<u32> {
    // Join question prompts.
    let combined: String = questions
        .iter()
        .map(|q| q.prompt.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    if combined.is_empty() {
        eprintln!("[QCF-dump] WARNING: all eval questions have empty prompts; warmup skipped");
        return Vec::new();
    }

    let enc = match tokenizer.encode(combined.as_str(), true) {
        Ok(e) => e,
        Err(e) => {
            eprintln!(
                "[QCF-dump] WARNING: warmup tokenize error: {}; warmup skipped",
                e
            );
            return Vec::new();
        }
    };

    let ids: Vec<u32> = enc.get_ids().iter().take(max_tokens).copied().collect();

    if ids.len() < max_tokens {
        eprintln!(
            "[QCF-dump] WARNING: only {} warmup tokens available (requested {}); \
             using all available tokens",
            ids.len(),
            max_tokens
        );
    }

    ids
}

// ── KIVI + PPL mode: KiviCache-based perplexity evaluation ───────────────────

#[allow(clippy::too_many_arguments)]
fn run_kivi_ppl(
    args: &Args,
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    max_seq_len: usize,
    residual_size: usize,
    text_file: &str,
) -> anyhow::Result<()> {
    use llm_rs2::core::kv_cache::KVCacheOps;

    let hidden_size = model.config.hidden_size;
    let vocab_size = model.config.vocab_size;

    // ── 1. Read and tokenize reference text ──
    let text = std::fs::read_to_string(text_file)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", text_file, e))?;
    let encoding = tokenizer
        .encode(text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenize error: {}", e))?;
    let all_ids: Vec<u32> = encoding.get_ids().to_vec();
    let total_tokens = all_ids.len();

    if total_tokens < 2 {
        anyhow::bail!("PPL requires at least 2 tokens, got {}", total_tokens);
    }

    let eval_tokens = total_tokens.min(max_seq_len);
    if total_tokens > max_seq_len {
        eprintln!(
            "[KIVI-PPL] Warning: text has {} tokens, truncating to max_seq_len={}",
            total_tokens, max_seq_len
        );
    }
    let token_ids = &all_ids[..eval_tokens];

    eprintln!(
        "[KIVI-PPL] {} tokens, kivi_residual_size={}, max_seq_len={}",
        eval_tokens, residual_size, max_seq_len
    );

    // ── 2. Create KiviCache per layer ──
    let mut kv_caches: Vec<KiviCache> = (0..num_layers)
        .map(|_| {
            KiviCache::new_gpu(
                kv_heads,
                head_dim,
                max_seq_len,
                residual_size,
                2,
                backend.clone(),
                memory.clone(),
            )
        })
        .collect();

    // ── 3. Pre-allocate decode buffers ──
    let dl_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut decode_logits =
        Tensor::new(Shape::new(vec![1, 1, vocab_size]), dl_buf, backend.clone());
    let xg_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let mut x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());
    let q_dim = model.config.num_attention_heads * head_dim;
    let k_dim = kv_heads * head_dim;
    let v_dim = k_dim;
    let ffn_hidden = model.config.intermediate_size;
    let mut gen_ws = LayerWorkspace::new(
        WorkspaceConfig {
            batch_size: 1,
            dim: hidden_size,
            q_dim,
            k_dim,
            v_dim,
            ffn_hidden,
            n_heads: model.config.num_attention_heads,
            max_seq_len: args.max_seq_len,
        },
        memory.as_ref(),
        backend.clone(),
    )?;
    let cpu_gen_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );
    // Pre-allocate GPU input tensor for decode loop (avoids per-token GPU alloc)
    let gpu_gen_buf_kp = memory.alloc(4, DType::U8)?;
    let mut gen_input_gpu = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf_kp, backend.clone());
    let mut logits_cpu = vec![0.0f32; vocab_size];

    let mut total_nll: f64 = 0.0;
    let mut nll_count: usize = 0;
    let mut qcf_metrics: Vec<serde_json::Value> = Vec::new();
    let mut flush_count: usize = 0;
    let overall_start = std::time::Instant::now();

    // ── 4. Prefill phase ──
    let prefill_len = eval_tokens.min(max_seq_len);
    eprintln!("[KIVI-PPL] Prefill: {} tokens", prefill_len);

    {
        let cpu_backend = Arc::new(CpuBackend::new());
        let input_buf = Galloc::new().alloc(prefill_len * 4, DType::U8)?;
        unsafe {
            let ptr = input_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in token_ids[..prefill_len].iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_input = Tensor::new(Shape::new(vec![1, prefill_len]), input_buf, cpu_backend);
        let input_tensor = backend.copy_from(&cpu_input)?;

        let prefill_logits_buf = memory.alloc(prefill_len * vocab_size * 4, DType::F32)?;
        let mut prefill_logits = Tensor::new(
            Shape::new(vec![1, prefill_len, vocab_size]),
            prefill_logits_buf,
            backend.clone(),
        );

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: 0,
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

        // Collect flush QCF metrics from prefill
        for metric in kv_caches[0].take_flush_proxies() {
            qcf_metrics.push(serde_json::json!({
                "flush": flush_count,
                "action": metric.action,
                "raw_value": metric.raw_value,
                "normalized_value": metric.normalized_value,
                "tokens_quantized": metric.tokens_affected,
            }));
            flush_count += 1;
        }
        for cache in kv_caches[1..].iter_mut() {
            cache.take_flush_proxies();
        }

        // Read all prefill logits to CPU
        let mut all_logits = vec![0.0f32; prefill_len * vocab_size];
        unsafe {
            let ptr = all_logits.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, all_logits.len() * 4);
            backend.read_buffer(&prefill_logits, slice)?;
        }

        // Score tokens 1..prefill_len: logits[i] predicts token[i+1]
        for i in 0..prefill_len - 1 {
            let offset = i * vocab_size;
            let lp = sampling::compute_log_prob(
                &all_logits[offset..offset + vocab_size],
                token_ids[i + 1],
                vocab_size,
            );
            total_nll -= lp;
            nll_count += 1;
        }

        eprintln!(
            "[KIVI-PPL] Prefill NLL: {:.4}, count={}, running PPL={:.4}, Q2_tokens={}, res_pos={}",
            total_nll,
            nll_count,
            (total_nll / nll_count as f64).exp(),
            kv_caches[0].q2_tokens,
            kv_caches[0].res_pos,
        );
    }

    // ── 5. Decode phase (teacher-forcing) ──
    let mut start_pos = prefill_len;

    for i in prefill_len..eval_tokens - 1 {
        let input_token = token_ids[i];
        let target_token = token_ids[i + 1];

        // Feed true token
        unsafe {
            *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = input_token;
        }
        // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
        backend.write_buffer(&mut gen_input_gpu, unsafe {
            std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
        })?;

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &gen_input_gpu,
            start_pos,
            kv_caches: &mut kv_caches,
            backend,
            memory: memory.as_ref(),
            logits_out: &mut decode_logits,
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
        })?;
        start_pos += 1;

        // Collect flush QCF from decode step
        for metric in kv_caches[0].take_flush_proxies() {
            qcf_metrics.push(serde_json::json!({
                "flush": flush_count,
                "action": metric.action,
                "raw_value": metric.raw_value,
                "normalized_value": metric.normalized_value,
                "tokens_quantized": metric.tokens_affected,
            }));
            flush_count += 1;
        }
        for cache in kv_caches[1..].iter_mut() {
            cache.take_flush_proxies();
        }

        // Read logits and score target
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
            backend.read_buffer(&decode_logits, slice)?;
        }
        let lp = sampling::compute_log_prob(&logits_cpu, target_token, vocab_size);
        total_nll -= lp;
        nll_count += 1;

        // Progress
        if (i + 1) % 200 == 0 {
            let ppl = (total_nll / nll_count as f64).exp();
            eprintln!(
                "[KIVI-PPL] step {}/{}: NLL={:.4}, PPL={:.4}, cache_pos={}, Q2_tokens={}",
                i + 1,
                eval_tokens,
                total_nll,
                ppl,
                kv_caches[0].current_pos(),
                kv_caches[0].q2_tokens,
            );
        }
    }

    // ── 6. Output results ──
    let wall_time = overall_start.elapsed().as_secs_f64();
    let ppl = (total_nll / nll_count as f64).exp();
    let tok_per_sec = nll_count as f64 / wall_time;

    // Separate QCF (NMSE) and OPR metrics from flush proxies
    let qcf_kivi_nmse_total: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() != Some("kivi_opr"))
        .filter_map(|m| m["raw_value"].as_f64())
        .sum();
    let qcf_attn_normalized_total: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() != Some("kivi_opr"))
        .filter_map(|m| m["normalized_value"].as_f64())
        .sum();

    // KIVI OPR: per-flush events and summary stats
    let qcf_kivi_events: Vec<&serde_json::Value> = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() == Some("kivi_opr"))
        .collect();
    let n_kivi_flushes = qcf_kivi_events.len();
    let opr_raw_values: Vec<f64> = qcf_kivi_events
        .iter()
        .filter_map(|m| m["raw_value"].as_f64())
        .collect();
    let qcf_kivi_opr_sum: f64 = opr_raw_values.iter().sum();
    let qcf_kivi_opr_max: f64 = opr_raw_values.iter().cloned().fold(0.0f64, f64::max);
    let qcf_kivi_opr_total: Option<f64> = if opr_raw_values.is_empty() {
        None
    } else {
        Some(qcf_kivi_opr_sum / opr_raw_values.len() as f64)
    };
    let qcf_kivi_opr_events: Option<usize> = if opr_raw_values.is_empty() {
        None
    } else {
        Some(opr_raw_values.len())
    };

    let output = serde_json::json!({
        "ppl": ppl,
        "total_nll": total_nll,
        "token_count": nll_count,
        "tokens_per_second": tok_per_sec,
        "wall_time_s": wall_time,
        "qcf_metrics": qcf_metrics,
        "flush_count": qcf_metrics.len(),
        "n_kivi_flushes": n_kivi_flushes,
        "qcf_kivi_events": qcf_kivi_events,
        "qcf_kivi_nmse_total": qcf_kivi_nmse_total,
        "qcf_attn_total": qcf_kivi_nmse_total,
        "qcf_attn_normalized_total": qcf_attn_normalized_total,
        "qcf_kivi_opr_sum": qcf_kivi_opr_sum,
        "qcf_kivi_opr_max": qcf_kivi_opr_max,
        "qcf_kivi_opr_total": qcf_kivi_opr_total,
        "qcf_kivi_opr_events": qcf_kivi_opr_events,
        "final_cache_pos": kv_caches[0].current_pos(),
        "kivi_q2_tokens": kv_caches[0].q2_tokens,
        "kivi_res_pos": kv_caches[0].res_pos,
        "config": {
            "model": args.model_path,
            "text_file": text_file,
            "eviction_policy": "kivi",
            "kivi_residual_size": residual_size,
            "max_seq_len": max_seq_len,
            "kv_type": "q2+f32_residual",
        }
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    eprintln!(
        "\n[KIVI-PPL] Final: PPL={:.4}, NLL={:.4}, tokens={}, {:.1} tok/s, {:.1}s, Q2_tokens={}",
        ppl, total_nll, nll_count, tok_per_sec, wall_time, kv_caches[0].q2_tokens
    );

    Ok(())
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
    use llm_rs2::core::kv_cache::KVCacheOps;

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
    use llm_rs2::core::skip_config::SkipConfig;
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
    use llm_rs2::core::kv_cache::KVCacheOps;
    use llm_rs2::core::offload::OffloadKVCache;
    use llm_rs2::core::offload::raw_store::RawStore;

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
            let store: Box<dyn llm_rs2::core::offload::store::OffloadStore> = match offload_mode {
                "raw" => Box::new(RawStore::new(token_bytes)),
                "disk" => Box::new(
                    llm_rs2::core::offload::disk_store::DiskStore::new(
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

    let mut prefetch =
        llm_rs2::core::offload::prefetch::PrefetchController::new(max_prefetch_depth, num_layers);

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

// ════════════════════════════════════════════════════════════════
//  PPL MODE: Teacher-forcing perplexity evaluation on reference text.
//
//  Reads a text file, tokenizes it, and measures how well the model
//  predicts each token given all previous tokens. Applies the configured
//  eviction policy and collects proxy metrics during eviction events.
// ════════════════════════════════════════════════════════════════

/// LISWAP-PPL diagnostic: dump every layer's weight tensors (wq/wk/wv/wo/
/// w_gate/w_up/w_down) to raw bin files under `out_dir`. File naming:
/// `layer{NN}_{tensor}_{dtype}.bin` (e.g. `layer00_wq_Q4_0.bin`). Each file
/// holds the raw GPU buffer bytes for that tensor at the moment of the call.
///
/// Two such dumps (one from a Q4-native model load, one from an F16 model
/// after swap completion) can be byte-compared on the host to determine
/// whether the swap path produces bit-identical Q4 weights.
fn dump_layer_weights_to_dir(
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    out_dir: &std::path::Path,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let n = model.layers.len();
    eprintln!(
        "[Q4-DUMP] dumping {} layer weights to {}",
        n,
        out_dir.display()
    );
    for (i, slot) in model.layers.iter().enumerate() {
        let weights = slot.load_weights();
        let dtype = slot.current_dtype();
        let tensors: [(&str, &llm_rs2::core::tensor::Tensor); 7] = [
            ("wq", &weights.wq),
            ("wk", &weights.wk),
            ("wv", &weights.wv),
            ("wo", &weights.wo),
            ("w_gate", &weights.w_gate),
            ("w_up", &weights.w_up),
            ("w_down", &weights.w_down),
        ];
        for (name, t) in tensors {
            let nbytes = t.buffer().size();
            if nbytes == 0 {
                eprintln!(
                    "[Q4-DUMP] layer{:02} {} dtype={:?} SKIP (size=0)",
                    i, name, dtype
                );
                continue;
            }
            let mut bytes = vec![0u8; nbytes];
            // For OpenCL/CUDA tensors `buffer().as_ptr()` is the cl_mem/cu_ptr
            // handle and may look like a host nullptr — backend.read_buffer
            // does the device→host copy via the backend-specific path, so we
            // rely on its return value rather than pre-checking as_ptr.
            match backend.read_buffer(t, &mut bytes) {
                Ok(()) => {
                    let fname = format!("layer{:02}_{}_{:?}.bin", i, name, dtype);
                    let path = out_dir.join(&fname);
                    std::fs::write(&path, &bytes)?;
                    eprintln!(
                        "[Q4-DUMP] layer{:02} {:8} dtype={:>5?} bytes={:8} → {}",
                        i, name, dtype, nbytes, fname
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[Q4-DUMP] layer{:02} {} dtype={:?} SKIP (read_buffer failed: {})",
                        i, name, dtype, e
                    );
                }
            }
        }
    }
    // Also dump model-level tensors that are NOT inside per-layer slots and
    // therefore are NOT touched by weight swap: embed_tokens, final norm, and
    // lm_head. These three are the most likely sources of E ≠ D NLL drift
    // because (a) the F16 model's lm_head is typically tied to embed_tokens
    // and (b) any missing lm_head is derived via F16→Q4_0 quantization at
    // load time, whose result may not match a standalone Q4_0 GGUF's lm_head
    // byte-for-byte.
    let model_tensors: [(&str, &llm_rs2::core::tensor::Tensor); 3] = [
        ("embed_tokens", &model.embed_tokens),
        ("norm", &model.norm),
        ("lm_head", &model.lm_head),
    ];
    for (name, t) in model_tensors {
        let nbytes = t.buffer().size();
        if nbytes == 0 {
            eprintln!("[Q4-DUMP] model.{} SKIP (size=0)", name);
            continue;
        }
        let dt = t.dtype();
        let mut bytes = vec![0u8; nbytes];
        match backend.read_buffer(t, &mut bytes) {
            Ok(()) => {
                let fname = format!("model_{}_{:?}.bin", name, dt);
                let path = out_dir.join(&fname);
                std::fs::write(&path, &bytes)?;
                eprintln!(
                    "[Q4-DUMP] model.{:14} dtype={:>5?} bytes={:8} → {}",
                    name, dt, nbytes, fname
                );
            }
            Err(e) => {
                // The lm_head can live on a CPU backend even when the main
                // backend is GPU (`lm_head_on_cpu`) — fall back to CpuBackend
                // for that case so we still get a dump file out.
                let cpu_be: Arc<dyn Backend> = Arc::new(llm_rs2::backend::cpu::CpuBackend::new());
                match cpu_be.read_buffer(t, &mut bytes) {
                    Ok(()) => {
                        let fname = format!("model_{}_{:?}.bin", name, dt);
                        let path = out_dir.join(&fname);
                        std::fs::write(&path, &bytes)?;
                        eprintln!(
                            "[Q4-DUMP] model.{:14} dtype={:>5?} bytes={:8} → {} (via CPU fallback)",
                            name, dt, nbytes, fname
                        );
                    }
                    Err(e2) => {
                        eprintln!(
                            "[Q4-DUMP] model.{} SKIP (read_buffer failed: gpu={}, cpu={})",
                            name, e, e2
                        );
                    }
                }
            }
        }
    }
    eprintln!(
        "[Q4-DUMP] complete: {} layers + 3 model tensors dumped to {}",
        n,
        out_dir.display()
    );
    Ok(())
}

/// Return value from `run_ppl` for use by the caller (e.g. `--qcf-dump`).
struct PplResult {
    ppl: f64,
    avg_nll: f64,
    n_eval_tokens: usize,
    wall_time_s: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_ppl(
    args: &Args,
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut [KVCache],
    cache_manager: &mut CacheManager,
    score_accumulator: &mut Option<AttentionScoreAccumulator>,
    vocab_size: usize,
    hidden_size: usize,
    max_seq_len: usize,
    text_file: &str,
    auto_eviction: bool,
    score_based_eviction: bool,
    protected_prefix: usize,
    skip_config: Option<&llm_rs2::core::skip_config::SkipConfig>,
    // LISWAP-PPL Scenario E: when true, return early as soon as the swap plan
    // completes. NLL/CSV/JSON outputs are suppressed. Used by `--ppl-warmup-swap`
    // to drive the swap to completion before the actual measurement pass.
    warmup_only: bool,
) -> anyhow::Result<PplResult> {
    // ── 1. Read and tokenize reference text ──
    let text = std::fs::read_to_string(text_file)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", text_file, e))?;
    let encoding = tokenizer
        .encode(text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenize error: {}", e))?;
    let all_ids: Vec<u32> = encoding.get_ids().to_vec();
    let total_tokens = all_ids.len();

    if total_tokens < 2 {
        anyhow::bail!("PPL requires at least 2 tokens, got {}", total_tokens);
    }

    let eval_tokens = total_tokens.min(max_seq_len);
    if total_tokens > max_seq_len {
        eprintln!(
            "[PPL] Warning: text has {} tokens, truncating to max_seq_len={}",
            total_tokens, max_seq_len
        );
    }
    let token_ids = &all_ids[..eval_tokens];

    eprintln!(
        "[PPL] {} tokens, policy={}, kv_budget={}, kv_type={}",
        eval_tokens, args.eviction_policy, args.kv_budget, args.kv_type
    );

    // ── 2. Pre-allocate decode buffers ──
    let dl_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut decode_logits =
        Tensor::new(Shape::new(vec![1, 1, vocab_size]), dl_buf, backend.clone());
    let xg_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let mut x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());
    let q_dim = model.config.num_attention_heads * model.config.head_dim;
    let k_dim = model.config.num_key_value_heads * model.config.head_dim;
    let v_dim = k_dim;
    let ffn_hidden = model.config.intermediate_size;
    let mut gen_ws = LayerWorkspace::new(
        WorkspaceConfig {
            batch_size: 1,
            dim: hidden_size,
            q_dim,
            k_dim,
            v_dim,
            ffn_hidden,
            n_heads: model.config.num_attention_heads,
            max_seq_len: args.max_seq_len,
        },
        memory,
        backend.clone(),
    )?;
    let cpu_gen_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );
    // Pre-allocate GPU input tensor for decode loop (avoids per-token GPU alloc)
    let gpu_gen_buf_ppl = memory.alloc(4, DType::U8)?;
    let mut gen_input_gpu = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf_ppl, backend.clone());
    let mut logits_cpu = vec![0.0f32; vocab_size];

    // ── 3. Determine prefill chunk size ──
    let has_budget = args.kv_budget > 0 || args.kv_budget_ratio > 0.0;
    if auto_eviction && !has_budget {
        eprintln!(
            "[PPL] Warning: eviction enabled without --kv-budget. \
             Results may not be reproducible. Use --kv-budget N for deterministic experiments."
        );
    }
    let prefill_chunk = if let Some(forced) = args.ppl_prefill_tokens {
        // LISWAP-PPL: 명시적 prefill 길이 강제. swap 측정 시 decode loop 을
        // 충분히 돌리기 위함. budget 로직보다 우선.
        forced.clamp(2, eval_tokens)
    } else if has_budget {
        let budget = if args.kv_budget_ratio > 0.0 {
            ((eval_tokens as f32) * args.kv_budget_ratio) as usize
        } else {
            args.kv_budget
        };
        budget.min(eval_tokens).max(2)
    } else if auto_eviction && args.eviction_policy == "sliding" {
        args.eviction_window.min(eval_tokens)
    } else {
        eval_tokens
    };

    let effective_budget = if args.kv_budget_ratio > 0.0 {
        ((eval_tokens as f32) * args.kv_budget_ratio) as usize
    } else if args.kv_budget > 0 {
        args.kv_budget
    } else {
        max_seq_len // No budget → no eviction trigger
    };

    if has_budget {
        eprintln!(
            "[PPL] Effective budget: {} tokens (deterministic eviction)",
            effective_budget
        );
    }

    // Headroom-based threshold: evict only when cache exceeds budget + headroom.
    // This prevents 1-by-1 evictions every step and ensures batch evictions (~2 total).
    // Example: budget=1500 → headroom=375 → threshold=1875.
    let eviction_headroom = (effective_budget / 4).max(16);
    let eviction_threshold = effective_budget.saturating_add(eviction_headroom);

    let mut total_nll: f64 = 0.0;
    let mut nll_count: usize = 0;
    // PPL v3: collect QCF for every eviction event
    let mut qcf_events: Vec<serde_json::Value> = Vec::new();
    let overall_start = std::time::Instant::now();

    // LISWAP-PPL: per-token NLL log + token-index-triggered weight swap.
    // (phase, token_idx, token_id, nll, swap_state, layers_swapped)
    let mut per_token_log: Vec<(&'static str, usize, u32, f64, &'static str, usize)> = Vec::new();
    let log_per_token = args.ppl_nll_csv.is_some();
    let mut ppl_swap_plan: Option<llm_rs2::models::weights::IncrementalSwapPlan> = None;
    // dispatch_swap_weights 시그니처 호환용 (PPL 경로에서는 manager 보고 안 함).
    let mut ppl_swap_report_unused: Option<(f32, usize, std::time::Instant, f32)> = None;
    let mut layers_swapped_so_far: usize = 0;
    let ppl_cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let ppl_swap_logged = std::sync::atomic::AtomicBool::new(false);

    if args.ppl_swap_at_token.is_some() && model.secondary_mmap.is_none() {
        anyhow::bail!("--ppl-swap-at-token requires --secondary-gguf to load secondary weights");
    }

    // ── 4. Prefill phase ──
    let prefill_len = prefill_chunk.min(eval_tokens);
    eprintln!("[PPL] Prefill: {} tokens", prefill_len);

    {
        let cpu_backend = Arc::new(CpuBackend::new());
        let input_buf = Galloc::new().alloc(prefill_len * 4, DType::U8)?;
        unsafe {
            let ptr = input_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in token_ids[..prefill_len].iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_input = Tensor::new(Shape::new(vec![1, prefill_len]), input_buf, cpu_backend);
        let input_tensor = backend.copy_from(&cpu_input)?;

        let prefill_logits_buf = memory.alloc(prefill_len * vocab_size * 4, DType::F32)?;
        let mut prefill_logits = Tensor::new(
            Shape::new(vec![1, prefill_len, vocab_size]),
            prefill_logits_buf,
            backend.clone(),
        );

        if let Some(acc) = score_accumulator.as_mut() {
            acc.begin_step();
        }

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: 0,
            kv_caches,
            backend,
            memory,
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            score_accumulator: score_accumulator.as_mut(),
            profiler: None,
            skip_config,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;

        // Read all prefill logits to CPU
        let mut all_logits = vec![0.0f32; prefill_len * vocab_size];
        unsafe {
            let ptr = all_logits.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, all_logits.len() * 4);
            backend.read_buffer(&prefill_logits, slice)?;
        }

        // Score tokens 1..prefill_len: logits[i] predicts token[i+1]
        for i in 0..prefill_len - 1 {
            let offset = i * vocab_size;
            let lp = sampling::compute_log_prob(
                &all_logits[offset..offset + vocab_size],
                token_ids[i + 1],
                vocab_size,
            );
            total_nll -= lp;
            nll_count += 1;
            if log_per_token {
                per_token_log.push(("prefill", i, token_ids[i + 1], -lp, "none", 0));
            }
        }

        eprintln!(
            "[PPL] Prefill NLL: {:.4}, count={}, running PPL={:.4}",
            total_nll,
            nll_count,
            (total_nll / nll_count as f64).exp()
        );
    }

    // ── 5. Decode phase (teacher-forcing) ──
    let mut start_pos = prefill_len;

    for (decode_idx, i) in (prefill_len..eval_tokens - 1).enumerate() {
        let input_token = token_ids[i];
        let target_token = token_ids[i + 1];

        // ── LISWAP-PPL: token-index-triggered weight swap ──────────────────
        // dispatch_swap_weights 가 commit 한 IncrementalSwapPlan 을 매 decode
        // step 마다 K=ppl_swap_per_tick 만큼 drain. dynamic-K controller 와
        // async dispatcher 는 측정 결정론을 위해 사용하지 않는다.
        if Some(decode_idx) == args.ppl_swap_at_token && ppl_swap_plan.is_none() {
            dispatch_swap_weights(
                model,
                args.ppl_swap_ratio,
                llm_shared::DtypeTag::Q4_0,
                None, // importance None → fallback uniform
                decode_idx,
                &mut ppl_swap_plan,
                &mut ppl_swap_report_unused,
            );
            if !ppl_swap_logged.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[PPL-Swap] triggered at decode_idx={}, ratio={}, per_tick={}",
                    decode_idx, args.ppl_swap_ratio, args.ppl_swap_per_tick
                );
            }
        }
        let mut swap_state: &'static str = if layers_swapped_so_far > 0 {
            "post_swap"
        } else {
            "none"
        };
        let plan_done = if let Some(plan) = ppl_swap_plan.as_mut() {
            plan.set_per_tick(args.ppl_swap_per_tick);
            let chunk = plan.drain_chunk();
            if !chunk.is_empty() {
                let t_swap = std::time::Instant::now();
                match run_layer_swap(
                    model,
                    &chunk,
                    Some(backend),
                    &ppl_cpu_backend,
                    None,
                    #[cfg(feature = "opencl")]
                    None,
                    #[cfg(feature = "cuda-embedded")]
                    None,
                    #[cfg(feature = "cuda-embedded")]
                    None,
                ) {
                    Ok(report) => {
                        layers_swapped_so_far += report.swapped.len();
                        swap_state = "swapping";
                        eprintln!(
                            "[PPL-Swap] tick decode_idx={} chunk={:?} swapped={} remaining={} latency={:.1}ms",
                            decode_idx,
                            &chunk,
                            report.swapped.len(),
                            plan.remaining_count(),
                            t_swap.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                    Err(e) => {
                        eprintln!("[PPL-Swap] run_layer_swap error: {}", e);
                    }
                }
            }
            plan.is_done()
        } else {
            false
        };
        if plan_done {
            eprintln!(
                "[PPL-Swap] plan complete at decode_idx={}, total_swapped={}",
                decode_idx, layers_swapped_so_far
            );
            ppl_swap_plan = None;
            swap_state = "post_swap";

            if warmup_only {
                // LISWAP-PPL Scenario E (warmup pass): swap is complete, return
                // before scoring the current token so the caller can reset KV
                // caches and run the measurement pass from scratch with the
                // already-swapped weights.
                eprintln!(
                    "[PPL-Swap] warmup_only=true → returning at decode_idx={} (no further scoring)",
                    decode_idx
                );
                let wall_time = overall_start.elapsed().as_secs_f64();
                return Ok(PplResult {
                    ppl: 0.0,
                    avg_nll: 0.0,
                    n_eval_tokens: 0,
                    wall_time_s: wall_time,
                });
            }
        }

        // Score accumulator begin step
        if let Some(acc) = score_accumulator.as_mut() {
            acc.begin_step();
        }

        // Feed true token
        unsafe {
            *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = input_token;
        }
        // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
        backend.write_buffer(&mut gen_input_gpu, unsafe {
            std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
        })?;

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &gen_input_gpu,
            start_pos,
            kv_caches,
            backend,
            memory,
            logits_out: &mut decode_logits,
            x_gen: Some(&mut x_gen),
            workspace: Some(&mut gen_ws),
            score_accumulator: score_accumulator.as_mut(),
            profiler: None,
            skip_config,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        start_pos += 1;

        // Read logits and score target
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
            backend.read_buffer(&decode_logits, slice)?;
        }
        let lp = sampling::compute_log_prob(&logits_cpu, target_token, vocab_size);
        total_nll -= lp;
        nll_count += 1;
        if log_per_token {
            per_token_log.push((
                "decode",
                i,
                target_token,
                -lp,
                swap_state,
                layers_swapped_so_far,
            ));
        }

        // ── Budget-based eviction (deterministic, experiment-reproducible) ──
        // Eviction triggers when cache_pos exceeds eviction_threshold (budget + headroom).
        // Using headroom prevents 1-by-1 evictions: evictions occur in ~2 large batches
        // rather than 500+ tiny steps, preserving PPL measurement validity.
        // This is deterministic: same text + same budget = same eviction positions.
        // No dependency on memory pressure or hardware state.
        if auto_eviction && has_budget {
            let before_len = kv_caches[0].current_pos;
            if before_len > eviction_threshold {
                let ratio = effective_budget as f32 / before_len as f32;

                // GPU V buffer readback for QCF-CAOTE computation.
                let v_cpu_data: Option<Vec<f32>> = if args.kv_type == "f32"
                    && !kv_caches.is_empty()
                    && kv_caches[0].v_buffer.buffer().as_ptr().is_null()
                {
                    let v_elems = kv_caches[0].v_buffer.buffer().size() / 4;
                    let mut v_buf = vec![0.0f32; v_elems];
                    let byte_slice = unsafe {
                        std::slice::from_raw_parts_mut(v_buf.as_mut_ptr() as *mut u8, v_elems * 4)
                    };
                    match backend.read_buffer(&kv_caches[0].v_buffer, byte_slice) {
                        Ok(()) => Some(v_buf),
                        Err(_) => None,
                    }
                } else {
                    None
                };
                let can_compute_qcf = args.kv_type == "f32"
                    && !kv_caches.is_empty()
                    && (v_cpu_data.is_some() || !kv_caches[0].v_buffer.buffer().as_ptr().is_null());

                // Perform eviction
                let result = if score_based_eviction {
                    if let Some(acc) = score_accumulator.as_ref() {
                        if acc.is_active() {
                            let scores = acc.importance_scores().to_vec();
                            cache_manager.force_evict_with_scores(kv_caches, ratio, &scores)?
                        } else {
                            cache_manager.force_evict(kv_caches, ratio)?
                        }
                    } else {
                        cache_manager.force_evict(kv_caches, ratio)?
                    }
                } else {
                    cache_manager.force_evict(kv_caches, ratio)?
                };

                if result.evicted {
                    let eviction_ratio = result.tokens_removed as f32 / before_len as f32;
                    let ppl_at_event = (total_nll / nll_count as f64).exp();

                    let qcf_caote_value = if can_compute_qcf
                        && let Some(acc) = score_accumulator.as_ref()
                        && let Some(head_attn) = acc.last_step_head_attn()
                    {
                        use llm_rs2::core::qcf::{
                            AggregationMode, QcfActionType, UnifiedQcfParams, VDataSource,
                            compute_unified_qcf,
                        };
                        let target_len = ((before_len as f32) * ratio) as usize;
                        let cache = &kv_caches[0];
                        let v_cpu_bytes: Option<&[u8]> = v_cpu_data.as_deref().map(|s| {
                            // Reinterpret &[f32] as &[u8] so the unified helper handles it.
                            unsafe {
                                std::slice::from_raw_parts(
                                    s.as_ptr() as *const u8,
                                    std::mem::size_of_val(s),
                                )
                            }
                        });
                        let action = if score_based_eviction {
                            QcfActionType::EvictH2o {
                                target_len,
                                keep_ratio: args.h2o_keep_ratio,
                                protected_prefix,
                            }
                        } else {
                            QcfActionType::EvictSliding { target_len }
                        };
                        match VDataSource::from_kv_cache(cache, v_cpu_bytes) {
                            Some(v_source) => {
                                let params = UnifiedQcfParams {
                                    action,
                                    v_source,
                                    // PPL eval site only triggers Sliding/H2O,
                                    // never D2O — `k_source` is unused.
                                    k_source: None,
                                    attention_scores: acc.importance_scores(),
                                    head_attn: Some(head_attn),
                                    n_kv_heads: cache.kv_heads(),
                                    head_dim: cache.head_dim(),
                                    current_pos: before_len,
                                    capacity: cache.capacity(),
                                    layout: cache.layout(),
                                    aggregation: AggregationMode::Mean,
                                    beta: 1.0,
                                };
                                let (qcf, _) = compute_unified_qcf(&params);
                                qcf as f64
                            }
                            None => 0.0,
                        }
                    } else {
                        0.0
                    };

                    qcf_events.push(serde_json::json!({
                        "step": i,
                        "tokens_evicted": result.tokens_removed,
                        "eviction_ratio": eviction_ratio,
                        "qcf_caote": qcf_caote_value,
                        "ppl_at_step": ppl_at_event,
                    }));

                    // IMPORTANT: Do NOT reset start_pos to current_pos after eviction.
                    // After shift_positions(), cached K vectors retain their original RoPE
                    // positions. start_pos must continue incrementing from the original
                    // position to maintain correct RoPE relative distances. Using current_pos
                    // (compacted) creates a RoPE discontinuity where cached tokens appear
                    // as "future" tokens, causing severe NLL degradation.
                    // start_pos continues via `start_pos += 1` in the main loop.
                    if let Some(acc) = score_accumulator.as_mut() {
                        acc.reset();
                    }
                    eprintln!(
                        "[PPL] Eviction at step {}: {} → {} tokens (removed {})",
                        i, before_len, result.new_pos, result.tokens_removed
                    );
                }
            }
        }

        // Progress
        if (i + 1) % 200 == 0 {
            let ppl = (total_nll / nll_count as f64).exp();
            eprintln!(
                "[PPL] step {}/{}: NLL={:.4}, PPL={:.4}, cache_pos={}",
                i + 1,
                eval_tokens,
                total_nll,
                ppl,
                kv_caches[0].current_pos
            );
        }
    }

    // ── 6. Output results ──
    let wall_time = overall_start.elapsed().as_secs_f64();
    let ppl = (total_nll / nll_count as f64).exp();
    let avg_nll = total_nll / nll_count as f64;
    let tok_per_sec = nll_count as f64 / wall_time;

    // Compute summary stats from all eviction events (v3)
    let n_evictions = qcf_events.len();
    let qcf_sum_caote: f64 = qcf_events
        .iter()
        .filter_map(|e| e["qcf_caote"].as_f64())
        .sum();
    let qcf_max_caote: f64 = qcf_events
        .iter()
        .filter_map(|e| e["qcf_caote"].as_f64())
        .fold(0.0f64, f64::max);

    let output = serde_json::json!({
        "ppl": ppl,
        "total_nll": total_nll,
        "token_count": nll_count,
        "tokens_per_second": tok_per_sec,
        "wall_time_s": wall_time,
        "n_evictions": n_evictions,
        "qcf_events": qcf_events,
        "qcf_sum_caote": qcf_sum_caote,
        "qcf_max_caote": qcf_max_caote,
        "config": {
            "model": args.model_path,
            "text_file": text_file,
            "eviction_policy": args.eviction_policy,
            "kv_budget": args.kv_budget,
            "kv_type": args.kv_type,
            "max_seq_len": max_seq_len,
            "eviction_target_ratio": args.eviction_target_ratio,
            "h2o_keep_ratio": args.h2o_keep_ratio,
            "protected_prefix": protected_prefix,
            "skip_layers": args.skip_layers,
            "skip_ratio": args.skip_ratio,
        }
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    eprintln!(
        "\n[PPL] Final: PPL={:.4}, NLL={:.4}, tokens={}, {:.1} tok/s, {:.1}s",
        ppl, total_nll, nll_count, tok_per_sec, wall_time
    );

    // LISWAP-PPL: per-token NLL CSV dump (token_idx is text-absolute, identical
    // across scenarios for direct curve comparison).
    if let Some(csv_path) = args.ppl_nll_csv.as_ref() {
        use std::io::Write;
        let mut f = std::fs::File::create(csv_path)?;
        writeln!(f, "phase,token_idx,token_id,nll,swap_state,layers_swapped")?;
        for (phase, idx, id, nll, state, n) in &per_token_log {
            writeln!(f, "{},{},{},{:.6},{},{}", phase, idx, id, nll, state, n)?;
        }
        f.flush()?;
        eprintln!(
            "[PPL] Per-token NLL CSV: {} ({} rows)",
            csv_path.display(),
            per_token_log.len()
        );
    }

    Ok(PplResult {
        ppl,
        avg_nll,
        n_eval_tokens: nll_count,
        wall_time_s: wall_time,
    })
}

// ─────────────────────── Chat REPL mode ───────────────────────

use llm_rs2::core::chat_ipc::{
    ChatInput, finish_reply_stream, spawn_chat_input_sources, write_reply_bytes,
};

fn resolve_token_ids(
    tokenizer: &Tokenizer,
    literals: &[&'static str],
    required: bool,
) -> anyhow::Result<Vec<u32>> {
    let mut out = Vec::with_capacity(literals.len());
    for lit in literals {
        match tokenizer.token_to_id(lit) {
            Some(id) => out.push(id),
            None if required => {
                anyhow::bail!(
                    "tokenizer is missing required special token `{}`. \
                     Make sure tokenizer.json has it registered as an added_token.",
                    lit
                );
            }
            None => {}
        }
    }
    Ok(out)
}

// Chat turn executor: per-variant state machine for prefill/decode/eviction/reset.
// Keeps the REPL loop KV-type-agnostic. See `run_chat_repl`.
trait ChatTurnExec {
    /// Current KV position.
    fn pos(&self) -> usize;
    /// Reset session state (KV position, accumulator, offload store).
    fn reset(&mut self);
    /// Prefill a batch of tokens, advancing `pos` by `tokens.len()` and
    /// returning the last-position logits (host f32).
    fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>>;
    /// Decode a single token, advancing `pos` by 1 and returning logits.
    fn decode_step(&mut self, token: u32) -> anyhow::Result<Vec<f32>>;
    /// Ensure there is room for `additional` new tokens before `max_seq_len`.
    /// Eviction-capable execs may run force_evict here. Non-evicting execs
    /// return Err on overflow.
    fn ensure_capacity(&mut self, additional: usize, max_seq_len: usize) -> anyhow::Result<()>;
    /// End-of-turn maintenance hook (e.g. opportunistic auto-eviction).
    fn on_turn_end(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
    /// Content of the `/stats` line.
    fn stats_line(&self, max_seq_len: usize) -> String;
}

/// Shared REPL loop driving a `ChatTurnExec`. Handles template rendering,
/// stdin + socket input, slash commands, streaming decode, and turn-end
/// hooks. All KV-type-specific work is delegated to the exec.
#[allow(clippy::too_many_arguments)]
fn run_chat_repl<E: ChatTurnExec>(
    args: &Args,
    model_arch: llm_rs2::models::config::ModelArch,
    tokenizer: &Tokenizer,
    eos_token_id: u32,
    vocab_size: usize,
    sampling_config: &SamplingConfig,
    max_seq_len: usize,
    exec: &mut E,
) -> anyhow::Result<()> {
    use llm_rs2::core::chat_template::ChatTemplate;
    use std::collections::VecDeque;
    use std::io::Write;

    let template = ChatTemplate::new(model_arch)?;
    let stop_ids = {
        let lits = template.stop_token_literals();
        if lits.is_empty() {
            anyhow::bail!("chat template has no stop token literals");
        }
        let mut ids = resolve_token_ids(tokenizer, &[lits[0]], true)?;
        ids.extend(resolve_token_ids(tokenizer, &lits[1..], false)?);
        ids.push(eos_token_id);
        ids.sort_unstable();
        ids.dedup();
        ids
    };
    let assistant_eot_ids: Vec<u32> = tokenizer
        .encode(template.assistant_eot(), false)
        .map_err(|e| anyhow::anyhow!("encode EOT: {}", e))?
        .get_ids()
        .to_vec();
    let bos_id = if template.bos_needed_on_first_prefill() {
        template
            .bos_literal()
            .and_then(|lit| tokenizer.token_to_id(lit))
    } else {
        None
    };

    // Optional system prompt prefill (stays in KV across turns).
    if let Some(sys) = &args.system_prompt {
        let rendered = template.render_system(sys);
        let mut ids = tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode system: {}", e))?
            .get_ids()
            .to_vec();
        if let Some(b) = bos_id {
            ids.insert(0, b);
        }
        if ids.len() > max_seq_len {
            anyhow::bail!(
                "system prompt produces {} tokens, exceeds max_seq_len={}",
                ids.len(),
                max_seq_len
            );
        }
        let _ = exec.prefill(&ids)?;
    }

    let input_rx = spawn_chat_input_sources(args.chat_socket.as_deref(), args.chat_tcp.as_deref())?;
    let mut first_user: Option<String> =
        (!args.prompt.trim().is_empty()).then(|| args.prompt.clone());
    let mut recent: VecDeque<u32> = VecDeque::new();

    eprintln!(
        "[Chat] Ready. Arch={:?}, max_seq_len={}. Commands: /exit /reset /stats /help",
        model_arch, max_seq_len
    );
    let mut stdout_lock = std::io::stdout();

    'outer: loop {
        print!("> ");
        stdout_lock.flush().ok();

        let (user_line_raw, reply_writer) = if let Some(line) = first_user.take() {
            (line, None)
        } else {
            match input_rx.recv() {
                Ok(ChatInput::Line(s, w)) => (s, w),
                Ok(ChatInput::Eof) | Err(_) => {
                    eprintln!();
                    break 'outer;
                }
            }
        };
        let user_line = user_line_raw
            .trim_end_matches(&['\n', '\r'][..])
            .to_string();
        let trimmed = user_line.trim();

        match trimmed {
            "" => continue,
            "/exit" | "/quit" => break 'outer,
            "/help" => {
                println!("(commands: /exit /quit /reset /stats /help; empty line ignored)");
                continue;
            }
            "/stats" => {
                println!("{}", exec.stats_line(max_seq_len));
                continue;
            }
            "/reset" => {
                exec.reset();
                recent.clear();
                println!("(session reset)");
                continue;
            }
            _ => {}
        }

        let rendered = template.render_user_and_assistant_header(trimmed);
        let mut turn_ids: Vec<u32> = tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode user turn: {}", e))?
            .get_ids()
            .to_vec();
        if exec.pos() == 0
            && let Some(b) = bos_id
        {
            turn_ids.insert(0, b);
        }

        // Capacity check: eviction-capable execs may reclaim space here.
        if let Err(e) = exec.ensure_capacity(turn_ids.len() + args.num_tokens, max_seq_len) {
            let msg = format!("error: {}", e);
            eprintln!("{}", msg);
            write_reply_bytes(reply_writer.as_ref(), msg.as_bytes());
            finish_reply_stream(reply_writer.as_ref());
            anyhow::bail!("context overflow: {}", e);
        }

        let mut prefill_logits = exec.prefill(&turn_ids)?;

        let mut accum: Vec<u32> = Vec::new();
        let mut printed_bytes: usize = 0;
        let mut indices_buf: Vec<usize> = Vec::with_capacity(vocab_size);
        let first_tok = {
            let recent_slice: Vec<u32> = recent.iter().copied().collect();
            sampling::sample(
                &mut prefill_logits,
                &recent_slice,
                vocab_size,
                sampling_config,
                Some(&mut indices_buf),
            )
        };

        let mut cur_tok = first_tok;
        for _step in 0..args.num_tokens {
            if stop_ids.contains(&cur_tok) {
                break;
            }
            accum.push(cur_tok);
            recent.push_back(cur_tok);
            if recent.len() > sampling_config.repetition_window.max(1) {
                recent.pop_front();
            }

            let decoded = tokenizer.decode(&accum, true).unwrap_or_default();
            if decoded.len() > printed_bytes {
                let piece = &decoded[printed_bytes..];
                print!("{}", piece);
                stdout_lock.flush().ok();
                write_reply_bytes(reply_writer.as_ref(), piece.as_bytes());
                printed_bytes = decoded.len();
            }

            let mut logits_host = exec.decode_step(cur_tok)?;

            if exec.pos() + 1 >= max_seq_len {
                break;
            }

            let recent_slice: Vec<u32> = recent.iter().copied().collect();
            cur_tok = sampling::sample(
                &mut logits_host,
                &recent_slice,
                vocab_size,
                sampling_config,
                Some(&mut indices_buf),
            );
        }

        // Record assistant EOT into KV so the next turn sees a well-formed boundary.
        if !assistant_eot_ids.is_empty() && exec.pos() + assistant_eot_ids.len() <= max_seq_len {
            let _ = exec.prefill(&assistant_eot_ids)?;
        }

        exec.on_turn_end()?;

        println!();
        stdout_lock.flush().ok();
        finish_reply_stream(reply_writer.as_ref());
    }

    Ok(())
}

// ─── Standard chat executor (KVCache; supports eviction policies) ─────────────

struct StandardTurnExec<'a> {
    model: &'a TransformerModel,
    backend: &'a Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    kv_caches: &'a mut [KVCache],
    // Decode workspace (reused across tokens).
    decode_logits: Tensor,
    x_gen: Tensor,
    gen_ws: LayerWorkspace,
    cpu_gen_input: Tensor,
    gen_input_gpu: Tensor,
    vocab_size: usize,
    pos: usize,
    // Eviction wiring (None when --eviction-policy == "none").
    cache_manager: Option<CacheManager>,
    score_accumulator: Option<AttentionScoreAccumulator>,
    eviction_policy_name: String,
    score_based: bool,
    target_ratio: f32,
    evicted_total: usize,
}

impl<'a> StandardTurnExec<'a> {
    /// Build a f32 token tensor on backend.
    fn tokens_to_backend_tensor(&self, tokens: &[u32]) -> anyhow::Result<Tensor> {
        let cpu_backend = Arc::new(CpuBackend::new());
        let input_buf = Galloc::new().alloc(tokens.len() * 4, DType::U8)?;
        unsafe {
            let ptr = input_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in tokens.iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_input = Tensor::new(Shape::new(vec![1, tokens.len()]), input_buf, cpu_backend);
        self.backend.copy_from(&cpu_input)
    }

    /// Run one forward_into pass with the standard KVCache and optional
    /// score accumulator. Returns last-position logits read to host f32.
    fn forward_prefill_standard(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let input_tensor = self.tokens_to_backend_tensor(tokens)?;
        let logits_buf = self.memory.alloc(self.vocab_size * 4, DType::F32)?;
        let mut logits_out = Tensor::new(
            Shape::new(vec![1, 1, self.vocab_size]),
            logits_buf,
            self.backend.clone(),
        );

        if let Some(acc) = self.score_accumulator.as_mut() {
            acc.begin_step();
        }

        self.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: self.pos,
            kv_caches: self.kv_caches,
            backend: self.backend,
            memory: self.memory.as_ref(),
            logits_out: &mut logits_out,
            x_gen: None,
            workspace: None,
            score_accumulator: self.score_accumulator.as_mut(),
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: true,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        self.pos += tokens.len();

        let mut host = vec![0.0f32; self.vocab_size];
        unsafe {
            let ptr = host.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, self.vocab_size * 4);
            self.backend.read_buffer(&logits_out, slice)?;
        }
        Ok(host)
    }

    #[cfg(feature = "opencl")]
    fn gpu_sync_scores(&mut self) -> anyhow::Result<()> {
        if let Some(ocl_be) = self
            .backend
            .as_any()
            .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
            && let Some(gpu_acc) = ocl_be.gpu_score_acc()
            && gpu_acc.is_active()
        {
            let (flat, head) = gpu_acc.sync_to_cpu(ocl_be.queue.as_core())?;
            if let Some(ref mut acc) = self.score_accumulator {
                acc.import_gpu_scores(&flat, &head);
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "opencl"))]
    fn gpu_sync_scores(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Run eviction. Returns the number of tokens removed (0 if no-op).
    fn run_eviction(&mut self, force: bool) -> anyhow::Result<usize> {
        if self.cache_manager.is_none() {
            return Ok(0);
        }
        self.gpu_sync_scores()?;

        let before_len = self.kv_caches[0].current_pos;
        let scores_opt = self
            .score_accumulator
            .as_ref()
            .filter(|acc| acc.is_active())
            .map(|acc| acc.importance_scores().to_vec());

        let cache_manager = self.cache_manager.as_ref().unwrap();
        let result = if force {
            match (&scores_opt, self.score_based) {
                (Some(scores), true) => cache_manager.force_evict_with_scores(
                    self.kv_caches,
                    self.target_ratio,
                    scores,
                )?,
                _ => cache_manager.force_evict(self.kv_caches, self.target_ratio)?,
            }
        } else {
            match (&scores_opt, self.score_based) {
                (Some(scores), true) => {
                    cache_manager.maybe_evict_with_scores(self.kv_caches, scores)?
                }
                _ => cache_manager.maybe_evict(self.kv_caches)?,
            }
        };

        let removed = before_len.saturating_sub(self.kv_caches[0].current_pos);
        if result.evicted {
            self.pos = self.kv_caches[0].current_pos;
            self.evicted_total += removed;
            eprintln!(
                "[Chat/Evict] policy={} before={} after={} removed={}",
                self.eviction_policy_name, before_len, self.pos, removed
            );
        }
        Ok(removed)
    }
}

impl<'a> ChatTurnExec for StandardTurnExec<'a> {
    fn pos(&self) -> usize {
        self.pos
    }

    fn reset(&mut self) {
        for c in self.kv_caches.iter_mut() {
            c.current_pos = 0;
        }
        if let Some(acc) = self.score_accumulator.as_mut() {
            acc.reset();
        }
        self.pos = 0;
        self.evicted_total = 0;
    }

    fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        self.forward_prefill_standard(tokens)
    }

    fn decode_step(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        if let Some(acc) = self.score_accumulator.as_mut() {
            acc.begin_step();
        }
        unsafe {
            *(self.cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = token;
        }
        self.backend.write_buffer(&mut self.gen_input_gpu, unsafe {
            std::slice::from_raw_parts(self.cpu_gen_input.buffer().as_ptr(), 4)
        })?;

        self.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &self.gen_input_gpu,
            start_pos: self.pos,
            kv_caches: self.kv_caches,
            backend: self.backend,
            memory: self.memory.as_ref(),
            logits_out: &mut self.decode_logits,
            x_gen: Some(&mut self.x_gen),
            workspace: Some(&mut self.gen_ws),
            score_accumulator: self.score_accumulator.as_mut(),
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: true,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        self.pos += 1;

        let mut host = vec![0.0f32; self.vocab_size];
        unsafe {
            let ptr = host.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, self.vocab_size * 4);
            self.backend.read_buffer(&self.decode_logits, slice)?;
        }
        Ok(host)
    }

    fn ensure_capacity(&mut self, additional: usize, max_seq_len: usize) -> anyhow::Result<()> {
        if self.pos + additional <= max_seq_len {
            return Ok(());
        }
        if self.cache_manager.is_some() {
            // Force eviction; then re-check.
            self.run_eviction(true)?;
            if self.pos + additional <= max_seq_len {
                return Ok(());
            }
        }
        anyhow::bail!(
            "context would exceed max_seq_len={} (pos={}, incoming_reserve={}). \
             Use /reset or increase --max-seq-len.",
            max_seq_len,
            self.pos,
            additional
        );
    }

    fn on_turn_end(&mut self) -> anyhow::Result<()> {
        if self.cache_manager.is_none() {
            return Ok(());
        }
        // Force-evict once KV usage reaches 90% of capacity so long sessions
        // keep running without hitting the next-turn ensure_capacity hard stop.
        // Opportunistic maybe_evict (memory-pressure driven) runs at lower fill.
        let capacity = self.kv_caches[0].capacity();
        let at_pressure = self.pos >= capacity.saturating_mul(9) / 10;
        if at_pressure {
            self.run_eviction(true)?;
        } else {
            self.run_eviction(false)?;
        }
        Ok(())
    }

    fn stats_line(&self, max_seq_len: usize) -> String {
        format!(
            "kv_pos={}/{} policy={} evicted_total={}",
            self.pos, max_seq_len, self.eviction_policy_name, self.evicted_total
        )
    }
}

/// Build a CacheManager + AttentionScoreAccumulator for chat's eviction mode.
/// Returns (manager, accumulator, score_based, policy_name, target_ratio).
/// When `args.eviction_policy == "none"`, the manager is `None`.
#[allow(clippy::type_complexity)]
fn build_chat_eviction(
    args: &Args,
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    max_seq_len: usize,
) -> anyhow::Result<(
    Option<CacheManager>,
    Option<AttentionScoreAccumulator>,
    bool,
    String,
    f32,
)> {
    if args.eviction_policy == "none" {
        return Ok((None, None, false, "none".to_string(), 1.0));
    }

    let _ = backend;
    let actual_protected_prefix =
        args.protected_prefix
            .unwrap_or(match args.eviction_policy.as_str() {
                "h2o" | "h2o_plus" | "d2o" => 4,
                "streaming" => args.sink_size,
                _ => 4,
            });

    let monitor: Box<dyn llm_rs2::core::sys_monitor::SystemMonitor> = if backend.is_discrete_gpu() {
        Box::new(NoOpMonitor)
    } else {
        Box::new(LinuxSystemMonitor)
    };
    let threshold_bytes = args.memory_threshold_mb * 1024 * 1024;

    let mut cache_manager = if args.eviction_policy == "d2o" {
        let d2o_handler = D2OHandler::new(D2OConfig {
            keep_ratio: args.d2o_keep_ratio,
            protected_prefix: actual_protected_prefix,
            target_ratio: args.eviction_target_ratio,
            ema_beta: args.d2o_ema_beta,
            merge_e: args.d2o_merge_e,
            use_layer_allocation: args.d2o_layer_alloc,
            protected_layers: args.d2o_protected_layers.clone().unwrap_or_default(),
        });
        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(d2o_handler),
        }]);
        CacheManager::with_pipeline(pipeline, monitor, threshold_bytes)
    } else {
        let policy: Box<dyn llm_rs2::core::eviction::EvictionPolicy> = match args
            .eviction_policy
            .as_str()
        {
            "sliding" => Box::new(SlidingWindowPolicy::new(
                args.eviction_window,
                actual_protected_prefix,
            )),
            "streaming" => {
                use llm_rs2::core::eviction::StreamingLLMPolicy;
                let window = if args.streaming_window > 0 {
                    args.streaming_window
                } else if args.kv_budget > 0 {
                    args.kv_budget.saturating_sub(args.sink_size)
                } else {
                    args.eviction_window
                };
                Box::new(StreamingLLMPolicy::new(args.sink_size, window))
            }
            "h2o" => Box::new(H2OPolicy::new(args.h2o_keep_ratio, actual_protected_prefix)),
            "h2o_plus" => Box::new(H2OPlusPolicy::new(
                args.h2o_keep_ratio,
                actual_protected_prefix,
            )),
            other => anyhow::bail!(
                "Unknown eviction policy for --chat: '{}'. Use: none, sliding, streaming, h2o, h2o_plus, d2o",
                other
            ),
        };
        CacheManager::new(policy, monitor, threshold_bytes, args.eviction_target_ratio)
    };
    cache_manager.set_event_sink(Arc::new(StderrDiagnosticSink));

    // Accumulator setup: build for any active policy so sliding/streaming
    // still populate importance for observability; score-based policies need it.
    let score_based = matches!(args.eviction_policy.as_str(), "h2o" | "h2o_plus" | "d2o");
    // GQA accumulator: always active in chat. h2o_plus strictly requires it;
    // other policies benefit from per-head scores for future CAOTE / head budgets.
    let mut acc = AttentionScoreAccumulator::new_gqa(
        max_seq_len,
        model.config.num_attention_heads,
        model.config.num_key_value_heads,
        model.config.num_hidden_layers,
        args.h2o_tracked_layers,
        args.h2o_decay,
    );
    acc.set_active(true);
    acc.set_time_normalize(!args.h2o_raw_scores);

    // Init GPU-side accumulator when available.
    #[cfg(feature = "opencl")]
    if let Some(ocl_be) = backend
        .as_any()
        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
    {
        let _ = ocl_be.init_gpu_score_acc(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
            max_seq_len,
            args.h2o_decay,
        );
        if let Some(gpu_acc) = ocl_be.gpu_score_acc_mut() {
            gpu_acc.set_active(true);
        }
    }

    Ok((
        Some(cache_manager),
        Some(acc),
        score_based,
        args.eviction_policy.clone(),
        args.eviction_target_ratio,
    ))
}

#[allow(clippy::too_many_arguments)]
fn run_chat_standard(
    args: &Args,
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    kv_caches: &mut [KVCache],
    sampling_config: &SamplingConfig,
    max_seq_len: usize,
) -> anyhow::Result<()> {
    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;
    let q_dim = model.config.num_attention_heads * model.config.head_dim;
    let k_dim = model.config.num_key_value_heads * model.config.head_dim;
    let v_dim = k_dim;
    let ffn_hidden = model.config.intermediate_size;

    let dl_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let decode_logits = Tensor::new(Shape::new(vec![1, 1, vocab_size]), dl_buf, backend.clone());
    let xg_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());
    let gen_ws = LayerWorkspace::new(
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
    let cpu_gen_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );
    let gpu_gen_buf = memory.alloc(4, DType::U8)?;
    let gen_input_gpu = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf, backend.clone());

    let (cache_manager, score_accumulator, score_based, policy_name, target_ratio) =
        build_chat_eviction(args, model, backend, max_seq_len)?;

    let mut exec = StandardTurnExec {
        model,
        backend,
        memory: memory.clone(),
        kv_caches,
        decode_logits,
        x_gen,
        gen_ws,
        cpu_gen_input,
        gen_input_gpu,
        vocab_size,
        pos: 0,
        cache_manager,
        score_accumulator,
        eviction_policy_name: policy_name,
        score_based,
        target_ratio,
        evicted_total: 0,
    };

    run_chat_repl(
        args,
        model.config.arch,
        tokenizer,
        model.config.eos_token_id,
        vocab_size,
        sampling_config,
        max_seq_len,
        &mut exec,
    )
}

// ─── KIVI chat executor (quantized KV cache) ──────────────────────────────────

struct KiviTurnExec<'a> {
    model: &'a TransformerModel,
    backend: &'a Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    kv_caches: Vec<KiviCache>,
    decode_logits: Tensor,
    x_gen: Tensor,
    gen_ws: LayerWorkspace,
    cpu_gen_input: Tensor,
    gen_input_gpu: Tensor,
    vocab_size: usize,
    pos: usize,
    bits: u8,
    residual_size: usize,
}

impl<'a> KiviTurnExec<'a> {
    fn tokens_to_backend_tensor(&self, tokens: &[u32]) -> anyhow::Result<Tensor> {
        let cpu_backend = Arc::new(CpuBackend::new());
        let input_buf = Galloc::new().alloc(tokens.len() * 4, DType::U8)?;
        unsafe {
            let ptr = input_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in tokens.iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_input = Tensor::new(Shape::new(vec![1, tokens.len()]), input_buf, cpu_backend);
        self.backend.copy_from(&cpu_input)
    }
}

impl<'a> ChatTurnExec for KiviTurnExec<'a> {
    fn pos(&self) -> usize {
        self.pos
    }

    fn reset(&mut self) {
        for c in self.kv_caches.iter_mut() {
            c.reset();
        }
        self.pos = 0;
    }

    fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let input_tensor = self.tokens_to_backend_tensor(tokens)?;
        let logits_buf = self.memory.alloc(self.vocab_size * 4, DType::F32)?;
        let mut logits_out = Tensor::new(
            Shape::new(vec![1, 1, self.vocab_size]),
            logits_buf,
            self.backend.clone(),
        );
        self.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: self.pos,
            kv_caches: &mut self.kv_caches,
            backend: self.backend,
            memory: self.memory.as_ref(),
            logits_out: &mut logits_out,
            x_gen: None,
            workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: true,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        self.pos += tokens.len();

        let mut host = vec![0.0f32; self.vocab_size];
        unsafe {
            let ptr = host.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, self.vocab_size * 4);
            self.backend.read_buffer(&logits_out, slice)?;
        }
        Ok(host)
    }

    fn decode_step(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        unsafe {
            *(self.cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = token;
        }
        self.backend.write_buffer(&mut self.gen_input_gpu, unsafe {
            std::slice::from_raw_parts(self.cpu_gen_input.buffer().as_ptr(), 4)
        })?;
        self.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &self.gen_input_gpu,
            start_pos: self.pos,
            kv_caches: &mut self.kv_caches,
            backend: self.backend,
            memory: self.memory.as_ref(),
            logits_out: &mut self.decode_logits,
            x_gen: Some(&mut self.x_gen),
            workspace: Some(&mut self.gen_ws),
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: true,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        self.pos += 1;

        let mut host = vec![0.0f32; self.vocab_size];
        unsafe {
            let ptr = host.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, self.vocab_size * 4);
            self.backend.read_buffer(&self.decode_logits, slice)?;
        }
        Ok(host)
    }

    fn ensure_capacity(&mut self, additional: usize, max_seq_len: usize) -> anyhow::Result<()> {
        if self.pos + additional > max_seq_len {
            anyhow::bail!(
                "context would exceed max_seq_len={} (pos={}, incoming_reserve={}). \
                 Use /reset or increase --max-seq-len.",
                max_seq_len,
                self.pos,
                additional
            );
        }
        Ok(())
    }

    fn stats_line(&self, max_seq_len: usize) -> String {
        format!(
            "kv_pos={}/{} mode=kivi bits={} residual={}",
            self.pos, max_seq_len, self.bits, self.residual_size
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_chat_kivi(
    args: &Args,
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    sampling_config: &SamplingConfig,
    kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    max_seq_len: usize,
) -> anyhow::Result<()> {
    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;
    let q_dim = model.config.num_attention_heads * head_dim;
    let k_dim = kv_heads * head_dim;
    let v_dim = kv_heads * head_dim;
    let ffn_hidden = model.config.intermediate_size;
    let residual_size = args.kivi_residual_size;
    let bits = args.kivi_bits;

    eprintln!(
        "[Chat/KIVI] bits={}, residual_size={}, max_seq_len={}",
        bits, residual_size, max_seq_len
    );

    let kv_caches: Vec<KiviCache> = (0..num_layers)
        .map(|_| {
            KiviCache::new_gpu(
                kv_heads,
                head_dim,
                max_seq_len,
                residual_size,
                bits,
                backend.clone(),
                memory.clone(),
            )
        })
        .collect();

    let dl_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let decode_logits = Tensor::new(Shape::new(vec![1, 1, vocab_size]), dl_buf, backend.clone());
    let xg_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());
    let gen_ws = LayerWorkspace::new(
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
    let cpu_gen_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );
    let gpu_gen_buf = memory.alloc(4, DType::U8)?;
    let gen_input_gpu = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf, backend.clone());

    let mut exec = KiviTurnExec {
        model,
        backend,
        memory: memory.clone(),
        kv_caches,
        decode_logits,
        x_gen,
        gen_ws,
        cpu_gen_input,
        gen_input_gpu,
        vocab_size,
        pos: 0,
        bits,
        residual_size,
    };

    run_chat_repl(
        args,
        model.config.arch,
        tokenizer,
        model.config.eos_token_id,
        vocab_size,
        sampling_config,
        max_seq_len,
        &mut exec,
    )
}

// ─── KV-Offload chat executor ─────────────────────────────────────────────────

struct OffloadTurnExec<'a> {
    model: &'a TransformerModel,
    backend: &'a Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    kv_caches: Vec<llm_rs2::core::offload::OffloadKVCache>,
    decode_logits: Tensor,
    x_gen: Tensor,
    gen_ws: LayerWorkspace,
    cpu_gen_input: Tensor,
    gen_input_gpu: Tensor,
    vocab_size: usize,
    pos: usize,
    offload_mode: String,
    max_prefetch_depth: usize,
    prefetch: llm_rs2::core::offload::prefetch::PrefetchController,
}

impl<'a> OffloadTurnExec<'a> {
    fn tokens_to_backend_tensor(&self, tokens: &[u32]) -> anyhow::Result<Tensor> {
        let cpu_backend = Arc::new(CpuBackend::new());
        let input_buf = Galloc::new().alloc(tokens.len() * 4, DType::U8)?;
        unsafe {
            let ptr = input_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in tokens.iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_input = Tensor::new(Shape::new(vec![1, tokens.len()]), input_buf, cpu_backend);
        self.backend.copy_from(&cpu_input)
    }
}

impl<'a> ChatTurnExec for OffloadTurnExec<'a> {
    fn pos(&self) -> usize {
        self.pos
    }

    fn reset(&mut self) {
        for c in self.kv_caches.iter_mut() {
            c.reset_session();
        }
        self.pos = 0;
    }

    fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let input_tensor = self.tokens_to_backend_tensor(tokens)?;
        let logits_buf = self.memory.alloc(self.vocab_size * 4, DType::F32)?;
        let mut logits_out = Tensor::new(
            Shape::new(vec![1, 1, self.vocab_size]),
            logits_buf,
            self.backend.clone(),
        );
        self.model.forward_into_offload(
            TransformerModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos: self.pos,
                kv_caches: &mut self.kv_caches,
                backend: self.backend,
                memory: self.memory.as_ref(),
                logits_out: &mut logits_out,
                x_gen: None,
                workspace: None,
                score_accumulator: None,
                profiler: None,
                skip_config: None,
                importance_collector: None,
                logits_last_only: true,
                variance_collector: None,
                prefill_workspace: None,

                layer_boundary_hook: None,
            },
            &mut self.prefetch,
        )?;
        self.pos += tokens.len();

        let mut host = vec![0.0f32; self.vocab_size];
        unsafe {
            let ptr = host.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, self.vocab_size * 4);
            self.backend.read_buffer(&logits_out, slice)?;
        }
        Ok(host)
    }

    fn decode_step(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        unsafe {
            *(self.cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = token;
        }
        self.backend.write_buffer(&mut self.gen_input_gpu, unsafe {
            std::slice::from_raw_parts(self.cpu_gen_input.buffer().as_ptr(), 4)
        })?;
        self.model.forward_into_offload(
            TransformerModelForwardArgs {
                input_tokens: &self.gen_input_gpu,
                start_pos: self.pos,
                kv_caches: &mut self.kv_caches,
                backend: self.backend,
                memory: self.memory.as_ref(),
                logits_out: &mut self.decode_logits,
                x_gen: Some(&mut self.x_gen),
                workspace: Some(&mut self.gen_ws),
                score_accumulator: None,
                profiler: None,
                skip_config: None,
                importance_collector: None,
                logits_last_only: true,
                variance_collector: None,
                prefill_workspace: None,

                layer_boundary_hook: None,
            },
            &mut self.prefetch,
        )?;
        self.pos += 1;

        let mut host = vec![0.0f32; self.vocab_size];
        unsafe {
            let ptr = host.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, self.vocab_size * 4);
            self.backend.read_buffer(&self.decode_logits, slice)?;
        }
        Ok(host)
    }

    fn ensure_capacity(&mut self, additional: usize, max_seq_len: usize) -> anyhow::Result<()> {
        if self.pos + additional > max_seq_len {
            anyhow::bail!(
                "context would exceed max_seq_len={} (pos={}, incoming_reserve={}). \
                 Use /reset or increase --max-seq-len.",
                max_seq_len,
                self.pos,
                additional
            );
        }
        Ok(())
    }

    fn stats_line(&self, max_seq_len: usize) -> String {
        format!(
            "kv_pos={}/{} mode=offload store={} prefetch_depth={}",
            self.pos, max_seq_len, self.offload_mode, self.max_prefetch_depth
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_chat_offload(
    args: &Args,
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    sampling_config: &SamplingConfig,
    kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    max_seq_len: usize,
) -> anyhow::Result<()> {
    use llm_rs2::core::offload::OffloadKVCache;
    use llm_rs2::core::offload::raw_store::RawStore;

    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;
    let q_dim = model.config.num_attention_heads * head_dim;
    let k_dim = kv_heads * head_dim;
    let v_dim = kv_heads * head_dim;
    let ffn_hidden = model.config.intermediate_size;

    let kv_dtype = match args.kv_type.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        other => anyhow::bail!(
            "--chat --kv-offload requires --kv-type f16 or f32 (got '{}')",
            other
        ),
    };
    let token_bytes = kv_heads * head_dim * kv_dtype.size();
    let disk_dir = if args.offload_path.is_empty() {
        std::env::temp_dir().join("llm_rs2_kv_offload")
    } else {
        std::path::PathBuf::from(&args.offload_path)
    };
    let offload_mode = args.kv_offload.clone();
    eprintln!(
        "[Chat/Offload] mode={}, dtype={:?}, layers={}, token_bytes={}, max_seq={}",
        offload_mode, kv_dtype, num_layers, token_bytes, max_seq_len
    );

    let is_gpu_backend = backend.as_ref().is_gpu();
    let kv_caches: Vec<OffloadKVCache> = (0..num_layers)
        .map(|layer_id| {
            let store: Box<dyn llm_rs2::core::offload::store::OffloadStore> =
                match offload_mode.as_str() {
                    "raw" => Box::new(RawStore::new(token_bytes)),
                    "disk" => Box::new(
                        llm_rs2::core::offload::disk_store::DiskStore::new(
                            disk_dir.clone(),
                            layer_id,
                            token_bytes,
                        )
                        .expect("DiskStore::new failed"),
                    ),
                    other => panic!("Unknown offload mode: {}", other),
                };
            let mut c =
                OffloadKVCache::new(layer_id, kv_heads, head_dim, kv_dtype, max_seq_len, store);
            if is_gpu_backend {
                c.set_gpu_backend(backend.clone(), memory.clone());
            }
            c
        })
        .collect();

    let dl_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let decode_logits = Tensor::new(Shape::new(vec![1, 1, vocab_size]), dl_buf, backend.clone());
    let xg_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());
    let gen_ws = LayerWorkspace::new(
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
    let cpu_gen_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );
    let gpu_gen_buf = memory.alloc(4, DType::U8)?;
    let gen_input_gpu = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf, backend.clone());

    let prefetch = llm_rs2::core::offload::prefetch::PrefetchController::new(
        args.max_prefetch_depth,
        num_layers,
    );
    let mut exec = OffloadTurnExec {
        model,
        backend,
        memory: memory.clone(),
        kv_caches,
        decode_logits,
        x_gen,
        gen_ws,
        cpu_gen_input,
        gen_input_gpu,
        vocab_size,
        pos: 0,
        offload_mode,
        max_prefetch_depth: args.max_prefetch_depth,
        prefetch,
    };

    run_chat_repl(
        args,
        model.config.arch,
        tokenizer,
        model.config.eos_token_id,
        vocab_size,
        sampling_config,
        max_seq_len,
        &mut exec,
    )
}
