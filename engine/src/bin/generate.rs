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
use llm_rs2::core::kv_cache::{KVCache, KVLayout};
use llm_rs2::core::memory::Memory;
use llm_rs2::core::pressure::d2o_handler::{D2OConfig, D2OHandler};
use llm_rs2::core::pressure::{CachePressurePipeline, PressureLevel, PressureStageConfig};
use llm_rs2::core::sampling::{self, SamplingConfig};
use llm_rs2::core::shape::Shape;
use llm_rs2::core::sys_monitor::LinuxSystemMonitor;
use llm_rs2::core::tensor::Tensor;
use llm_rs2::layers::workspace::{LayerWorkspace, WorkspaceConfig};
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::models::llama::llama_model::{LlamaModel, LlamaModelForwardArgs};
use std::sync::Arc;
use tokenizers::Tokenizer;

use llm_rs2::experiment::{
    ExperimentSchedule, JsonlWriter, SummaryRecord, SystemSampler, TokenRecord,
    extract_top_k_logits,
};
#[cfg(feature = "resilience")]
use llm_rs2::resilience::DbusTransport;
#[cfg(unix)]
use llm_rs2::resilience::UnixSocketTransport;
use llm_rs2::resilience::{
    InferenceContext, ResilienceAction, ResilienceManager, SignalListener, SystemSignal,
    execute_action,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "models/llama3.2-1b")]
    model_path: String,

    /// Path to a file containing the prompt. Overrides --prompt if set.
    #[arg(long)]
    prompt_file: Option<String>,

    #[arg(short, long, default_value = "Hello, world! I am a")]
    prompt: String,

    #[arg(short, long, default_value_t = 20)]
    num_tokens: usize,

    #[arg(short, long, default_value = "cpu")]
    backend: String,

    /// Use zero-copy shared memory (slower but enables CPU-GPU sharing)
    #[arg(long, default_value_t = false)]
    zero_copy: bool,

    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,

    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    #[arg(long, default_value_t = 40)]
    top_k: usize,

    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,

    #[arg(long, default_value_t = 64)]
    repetition_window: usize,

    /// Use GPU kernel for attention computation (OpenCL only)
    #[arg(long, default_value_t = false)]
    gpu_attn: bool,

    /// KV cache data type (f32, f16, or q4)
    #[arg(long, default_value = "q4")]
    kv_type: String,

    /// Eviction policy for KV cache management (none, sliding, h2o, h2o_plus, d2o)
    #[arg(long, default_value = "none")]
    eviction_policy: String,

    /// Window size for sliding window eviction (tokens)
    #[arg(long, default_value_t = 1024)]
    eviction_window: usize,

    /// Deprecated: recent window is now derived from budget split. Kept for CLI compatibility.
    #[arg(long, default_value_t = 128, hide = true)]
    h2o_recent_window: usize,

    /// Fraction of tokens to keep as heavy hitters (0.0 to 1.0)
    #[arg(long, default_value_t = 0.5)]
    h2o_keep_ratio: f32,

    /// Number of final transformer layers to track for H2O importance scores (0 = all layers)
    #[arg(long, default_value_t = 0)]
    h2o_tracked_layers: usize,

    /// Exponential decay factor for H2O importance scores per step (0.0 = no decay)
    #[arg(long, default_value_t = 0.0)]
    h2o_decay: f32,

    /// D2O heavy-hitter keep ratio (0.0–1.0, paper default 0.75 = 3:1 ratio)
    #[arg(long, default_value_t = 0.75)]
    d2o_keep_ratio: f32,

    /// D2O EMA beta for similarity threshold (0.0–1.0, paper default 0.7)
    #[arg(long, default_value_t = 0.7)]
    d2o_beta: f32,

    /// D2O merge stability constant (paper default 1.0)
    #[arg(long, default_value_t = 1.0)]
    d2o_merge_e: f32,

    /// Number of prefix tokens to protect from eviction.
    /// Defaults to 4 for score-based policies (h2o, h2o_plus, d2o) and prompt length for sliding.
    #[arg(long)]
    protected_prefix: Option<usize>,

    /// Initial KV cache capacity in tokens (0 = auto: prompt length rounded up to power of 2, min 128)
    #[arg(long, default_value_t = 0)]
    initial_kv_capacity: usize,

    /// Memory threshold in MB below which eviction triggers
    #[arg(long, default_value_t = 256)]
    memory_threshold_mb: usize,

    /// Target ratio of cache to keep when evicting (0.1 to 0.99)
    #[arg(long, default_value_t = 0.75)]
    eviction_target_ratio: f32,

    /// Enable resilience manager for adaptive inference
    #[arg(long, default_value_t = false)]
    enable_resilience: bool,

    /// Resilience signal transport: "dbus" or "unix:<path>"
    #[arg(long, default_value = "dbus")]
    resilience_transport: String,

    // ── Experiment mode ──────────────────────────────
    /// Experiment schedule JSON file (enables experiment mode)
    #[arg(long)]
    experiment_schedule: Option<String>,

    /// Experiment output JSONL file path
    #[arg(long)]
    experiment_output: Option<String>,

    /// Number of top-K logits to record per token in experiment mode
    #[arg(long, default_value_t = 10)]
    experiment_logits_topk: usize,

    /// System metric sampling interval (N tokens, 0=disabled)
    #[arg(long, default_value_t = 1)]
    experiment_sample_interval: usize,

    /// Force greedy sampling (temperature=0) for reproducibility
    #[arg(long, default_value_t = false)]
    greedy: bool,

    /// KV cache memory layout: "head" (head-major) or "seq" (seq-major)
    #[arg(long, default_value = "head")]
    kv_layout: String,

    /// Override eviction target_ratio from resilience signals (experiment mode).
    /// When set, all Evict actions will use this ratio instead of the strategy default.
    #[arg(long)]
    experiment_eviction_ratio: Option<f32>,

    /// Enable verbose H2O debug output (per-step scores, softmax validation, eviction details)
    #[arg(long, default_value_t = false)]
    h2o_debug: bool,

    /// Enable time-normalized scoring: importance[t] / steps_in_cache[t].
    /// Removes the cumulative time-in-cache bias that makes H2O select the oldest tokens.
    #[arg(long, default_value_t = false)]
    h2o_time_normalize: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // Configure Rayon to use 8 threads
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();

    #[allow(unused_mut)]
    let mut args = Args::parse();

    // --greedy overrides temperature to 0
    if args.greedy {
        args.temperature = 0.0;
    }

    let sampling_config = SamplingConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        repetition_window: args.repetition_window,
    };

    let model_path = &args.model_path;

    // 1. Setup
    println!("[Profile] Event: ModelLoadStart");
    println!("Loading model from {}", model_path);
    let backend: Arc<dyn Backend> = match args.backend.as_str() {
        "cpu" => Arc::new(CpuBackend::new()),
        "opencl" => Arc::new(llm_rs2::backend::opencl::OpenCLBackend::new()?),
        _ => anyhow::bail!("Unknown backend: {}", args.backend),
    };
    let memory: Arc<dyn Memory> = if args.backend == "opencl" {
        let ocl_backend = backend
            .as_any()
            .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
            .ok_or(anyhow::anyhow!("Failed into cast to OpenCLBackend"))?;
        Arc::new(llm_rs2::backend::opencl::memory::OpenCLMemory::new(
            ocl_backend.context.clone(),
            ocl_backend.queue.clone(),
            args.zero_copy,
        ))
    } else {
        Arc::new(Galloc::new())
    };
    let model = LlamaModel::load(model_path, backend.clone(), &*memory)?;

    // 2. Tokenizer
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
        .map_err(|e| anyhow::anyhow!(e))?;

    // 3. Prompt
    let prompt = if let Some(path) = &args.prompt_file {
        std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read prompt file {}: {}", path, e))?
    } else {
        args.prompt.clone()
    };
    println!("Prompt: {}", prompt);

    let encoding = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("Token Length: {}", input_ids.len());

    // 4. Prepare KV Cache
    let max_seq_len = args.max_seq_len; // Use argument
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;
    let num_layers = model.config.num_hidden_layers;
    println!(
        "Model config: layers={}, kv_heads={}, head_dim={}, max_seq_len={}",
        num_layers, kv_heads, head_dim, max_seq_len
    );

    let kv_type = match args.kv_type.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "q4" => DType::Q4_0,
        _ => anyhow::bail!(
            "Unsupported KV type: {}. Use f32, f16, or q4.",
            args.kv_type
        ),
    };

    // Determine initial KV cache capacity (dynamic grow-on-demand)
    let initial_kv_capacity = if args.initial_kv_capacity > 0 {
        args.initial_kv_capacity.min(max_seq_len)
    } else {
        input_ids
            .len()
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
    println!(
        "KV cache type: {:?}, layout: {:?} (initial capacity: {} tokens, {}B per layer, max: {})",
        kv_type, kv_layout, initial_kv_capacity, kv_buf_size, max_seq_len
    );

    let mut kv_caches = Vec::new();
    for _ in 0..num_layers {
        let k_buf = memory.alloc(kv_buf_size, kv_type)?;
        let v_buf = memory.alloc(kv_buf_size, kv_type)?;

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

    // 5. Experiment schedule + Resilience Manager
    let experiment_schedule = if let Some(ref path) = args.experiment_schedule {
        Some(ExperimentSchedule::load(path)?)
    } else {
        None
    };

    let mut experiment_tx: Option<std::sync::mpsc::Sender<SystemSignal>> = None;
    let mut resilience_manager = if let Some(ref schedule) = experiment_schedule {
        // Experiment mode: internal mpsc channel (no external transport needed)
        let (tx, rx) = std::sync::mpsc::channel();
        experiment_tx = Some(tx);
        eprintln!("[Experiment] Mode enabled — schedule: {}", schedule.name);
        Some(ResilienceManager::new(rx))
    } else if args.enable_resilience {
        let (tx, rx) = std::sync::mpsc::channel();
        let _listener_handle = match args.resilience_transport.as_str() {
            #[cfg(feature = "resilience")]
            "dbus" => SignalListener::new(DbusTransport::new(), tx).spawn(),
            #[cfg(unix)]
            s if s.starts_with("unix:") => {
                let path = std::path::PathBuf::from(&s[5..]);
                SignalListener::new(UnixSocketTransport::new(path), tx).spawn()
            }
            other => {
                eprintln!("[Resilience] Unknown transport: {}", other);
                return Ok(());
            }
        };
        eprintln!(
            "[Resilience] Manager enabled — transport: {}",
            args.resilience_transport
        );
        Some(ResilienceManager::new(rx))
    } else {
        None
    };
    let mut throttle_delay_ms: u64 = 0;

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

    println!(
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
                // Sliding window / none: protect entire prompt (legacy behavior)
                _ => input_ids.len(),
            });

    let mut cache_manager = {
        let monitor = Box::new(LinuxSystemMonitor);
        let threshold_bytes = args.memory_threshold_mb * 1024 * 1024;

        if args.eviction_policy == "d2o" {
            // D2O uses CachePressureHandler (Pipeline mode), not EvictionPolicy (Legacy mode)
            let d2o_handler = D2OHandler::new(D2OConfig {
                keep_ratio: args.d2o_keep_ratio,
                protected_prefix: actual_protected_prefix,
                target_ratio: args.eviction_target_ratio,
                beta: args.d2o_beta,
                merge_e: args.d2o_merge_e,
            });
            let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(d2o_handler),
            }]);
            CacheManager::with_pipeline(pipeline, monitor, threshold_bytes)
        } else {
            let policy: Box<dyn llm_rs2::core::eviction::EvictionPolicy> =
                match args.eviction_policy.as_str() {
                    "none" => Box::new(NoEvictionPolicy::new()),
                    "sliding" => Box::new(SlidingWindowPolicy::new(
                        args.eviction_window,
                        actual_protected_prefix,
                    )),
                    "h2o" => Box::new(H2OPolicy::new(
                        args.h2o_recent_window,
                        args.h2o_keep_ratio,
                        actual_protected_prefix,
                    )),
                    "h2o_plus" => Box::new(H2OPlusPolicy::new(
                        args.h2o_recent_window,
                        args.h2o_keep_ratio,
                        actual_protected_prefix,
                    )),
                    other => anyhow::bail!(
                        "Unknown eviction policy: '{}'. Use: none, sliding, h2o, h2o_plus, d2o",
                        other
                    ),
                };
            CacheManager::new(policy, monitor, threshold_bytes, args.eviction_target_ratio)
        }
    };

    // Setup event sink for score diagnostics
    cache_manager.set_event_sink(Arc::new(StderrDiagnosticSink));

    // Setup AttentionScoreAccumulator for H2O / H2O+ / D2O
    let mut score_accumulator = if args.eviction_policy == "h2o" || args.eviction_policy == "d2o" {
        let mut acc = AttentionScoreAccumulator::new(
            max_seq_len,
            model.config.num_attention_heads,
            model.config.num_hidden_layers,
            args.h2o_tracked_layers,
            args.h2o_decay,
        );
        acc.set_active(true);
        acc.set_time_normalize(args.h2o_time_normalize);
        Some(acc)
    } else if args.eviction_policy == "h2o_plus" {
        let mut acc = AttentionScoreAccumulator::new_gqa(
            max_seq_len,
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
            model.config.num_hidden_layers,
            args.h2o_tracked_layers,
            args.h2o_decay,
        );
        acc.set_active(true);
        acc.set_time_normalize(args.h2o_time_normalize);
        Some(acc)
    } else {
        None
    };

    if args.eviction_policy != "none" {
        println!(
            "Eviction: policy={}, window={}, prefix={}, ratio={}, threshold={}MB",
            args.eviction_policy,
            args.eviction_window,
            actual_protected_prefix,
            args.eviction_target_ratio,
            args.memory_threshold_mb
        );
    }

    // Auto-eviction: sliding window in non-experiment mode checks memory pressure
    // after each forward pass. H2O/D2O are always signal-driven.
    let auto_eviction = args.eviction_policy == "sliding" && experiment_schedule.is_none();

    // Pre-allocate generation buffers
    let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        logits_buf,
        backend.clone(),
    );

    // Cache EOS token ID
    let eos_id = tokenizer
        .get_vocab(true)
        .get("</s>")
        .copied()
        .unwrap_or(u32::MAX);

    // === PREFILL PHASE ===
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

        // Create CPU tensor for input
        let cpu_indices_buf = Galloc::new().alloc(process_len * 4, DType::U8)?;
        unsafe {
            let ptr = cpu_indices_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, process_len);
        }
        let cpu_input_tensor = Tensor::new(
            Shape::new(vec![1, process_len]),
            cpu_indices_buf,
            Arc::new(CpuBackend::new()),
        );
        let input_tensor = backend.copy_from(&cpu_input_tensor)?;

        let prefill_logits_buf = memory.alloc(process_len * vocab_size * 4, DType::F32)?;
        let mut prefill_logits = Tensor::new(
            Shape::new(vec![1, process_len, vocab_size]),
            prefill_logits_buf,
            backend.clone(),
        );

        model.forward_into(LlamaModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos,
            kv_caches: &mut kv_caches,
            backend: &backend,
            memory: memory.as_ref(),
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            use_gpu_attn: args.gpu_attn,
            score_accumulator: None, // No score tracking during prefill
        })?;
        // Auto-eviction after prefill (sliding window only, non-experiment mode)
        if auto_eviction {
            cache_manager.maybe_evict(&mut kv_caches).ok();
        }

        // Sample last token
        // Read logits to CPU
        let mut logits_cpu = vec![0.0f32; process_len * vocab_size];
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
            backend.read_buffer(&prefill_logits, slice)?;
        }

        // Extract last token logits
        let start_idx = (process_len - 1) * vocab_size;
        let mut last_logits = logits_cpu[start_idx..start_idx + vocab_size].to_vec();

        let next_token_id =
            sampling::sample(&mut last_logits, &tokens, vocab_size, &sampling_config);

        _ttft_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        _last_token_time = std::time::Instant::now();

        tokens.push(next_token_id);
        start_pos += process_len;
    }

    // === GENERATION PHASE ===
    {
        println!("[Profile] Event: DecodingStart");
        // Pre-allocate workspace for generation
        let q_dim = hidden_size;
        let k_dim = hidden_size / 4;
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
                max_seq_len: args.max_seq_len, // Use context window size
            },
            memory.as_ref(),
            backend.clone(),
        )?;

        // Single token CPU tensor for generation loop
        let cpu_gen_indices_buf = Galloc::new().alloc(4, DType::U8)?;
        let cpu_gen_input = Tensor::new(
            Shape::new(vec![1, 1]),
            cpu_gen_indices_buf,
            Arc::new(CpuBackend::new()),
        );

        // Streaming setup
        use std::io::Write;
        let mut stdout = std::io::stdout();
        let mut _printed_len = 0;

        // Print initial tokens (prompt + first generated)
        let initial_text = tokenizer.decode(&tokens, true).unwrap_or_default();
        print!("{}", initial_text);
        _printed_len = initial_text.len();
        stdout.flush().ok();

        // Generation loop
        for (decode_token_index, _) in (0..(args.num_tokens - 1)).enumerate() {
            // Check physical cache capacity (not start_pos, which is logical RoPE position)
            if kv_caches[0].current_pos >= max_seq_len {
                println!("\n[Stopped: Max context length reached]");
                break;
            }

            let last_token = tokens[tokens.len() - 1];
            unsafe {
                *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_token;
            }
            let gen_input_tensor = backend.copy_from(&cpu_gen_input)?;

            // Apply decay to accumulated importance scores before this step
            if let Some(ref mut acc) = score_accumulator {
                acc.begin_step();
            }

            let forward_start = std::time::Instant::now();
            model.forward_into(LlamaModelForwardArgs {
                input_tokens: &gen_input_tensor,
                start_pos,
                kv_caches: &mut kv_caches,
                backend: &backend,
                memory: memory.as_ref(),
                logits_out: &mut logits,
                x_gen: Some(&mut x_gen),
                workspace: Some(&mut gen_ws),
                use_gpu_attn: args.gpu_attn,
                score_accumulator: score_accumulator.as_mut(),
            })?;
            let forward_ms = forward_start.elapsed().as_secs_f64() * 1000.0;

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
                if let Some(ref acc) = score_accumulator {
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

            // Auto-eviction after forward pass (sliding window, non-experiment mode)
            if auto_eviction {
                let result = if let Some(ref acc) = score_accumulator {
                    if acc.is_active() {
                        cache_manager
                            .maybe_evict_with_scores(&mut kv_caches, acc.importance_scores())?
                    } else {
                        cache_manager.maybe_evict(&mut kv_caches)?
                    }
                } else {
                    cache_manager.maybe_evict(&mut kv_caches)?
                };
                if result.evicted
                    && let Some(ref mut acc) = score_accumulator
                {
                    acc.reset();
                }
            }
            forward_ms_values.push(forward_ms);

            // ── Experiment: inject signals at this token position ──
            let mut injected_signals: Vec<String> = Vec::new();
            if let (Some(schedule), Some(tx)) = (&experiment_schedule, &experiment_tx) {
                for entry in schedule.signals_at(decode_token_index) {
                    tx.send(entry.signal.clone()).ok();
                    injected_signals.push(signal_summary(&entry.signal));
                }
            }

            // ── Resilience checkpoint ─────────────────────────
            let mut action_names: Vec<String> = Vec::new();
            if let Some(rm) = &mut resilience_manager {
                let mut suspended = false;
                let mut reject_new = false;
                let mut num_tokens = args.num_tokens;
                let mut ctx = InferenceContext {
                    max_tokens: &mut num_tokens,
                    throttle_delay_ms: &mut throttle_delay_ms,
                    suspended: &mut suspended,
                    reject_new: &mut reject_new,
                };

                for action in rm.poll() {
                    action_names.push(action_summary(&action));

                    if let ResilienceAction::Evict { target_ratio } = &action {
                        let effective_ratio =
                            args.experiment_eviction_ratio.unwrap_or(*target_ratio);

                        // ── Score distribution diagnostic (via events system) ──
                        if let Some(ref acc) = score_accumulator {
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

                                // Dump full scores to CSV if experiment output is configured
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

                        let result = if let Some(ref acc) = score_accumulator {
                            if let Some(head_imp) = acc.head_importance_scores() {
                                cache_manager.force_evict_with_head_scores(
                                    &mut kv_caches,
                                    effective_ratio,
                                    acc.importance_scores(),
                                    head_imp,
                                    acc.n_kv_heads(),
                                )
                            } else {
                                cache_manager.force_evict_with_scores(
                                    &mut kv_caches,
                                    effective_ratio,
                                    acc.importance_scores(),
                                )
                            }
                        } else {
                            cache_manager.force_evict(&mut kv_caches, effective_ratio)
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
                                    if let Some(ref acc) = score_accumulator {
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
                                experiment_eviction_count += 1;
                                experiment_evicted_total += r.tokens_removed;
                                if let Some(ref mut acc) = score_accumulator {
                                    acc.reset();
                                }
                            }
                            Err(e) => eprintln!("[Resilience] Eviction error: {}", e),
                            _ => {}
                        }
                    } else {
                        execute_action(&action, &mut ctx);
                    }
                }

                args.num_tokens = num_tokens;

                if suspended {
                    eprintln!("\n[Resilience] Inference suspended by system signal");
                    break;
                }

                if throttle_delay_ms > 0 {
                    experiment_total_throttle_ms += throttle_delay_ms;
                    std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
                }
            }
            // ── End Resilience checkpoint ─────────────────────

            // Read logits to CPU
            let mut logits_cpu = vec![0.0f32; vocab_size];
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

            let next_token_id =
                sampling::sample(&mut logits_cpu, &tokens, vocab_size, &sampling_config);

            let now = std::time::Instant::now();
            let tbt = now.duration_since(_last_token_time).as_secs_f64() * 1000.0;
            tbt_values.push(tbt);

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

            if next_token_id == eos_id {
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

    // 7. Output results
    println!("\nDone.");
    println!("[Profile] Event: End");
    println!("TTFT: {:.2} ms", _ttft_ms);
    if !tbt_values.is_empty() {
        let avg_tbt: f64 = tbt_values.iter().sum::<f64>() / tbt_values.len() as f64;
        println!(
            "Avg TBT: {:.2} ms ({:.1} tokens/sec)",
            avg_tbt,
            1000.0 / avg_tbt
        );
    }

    Ok(())
}

// ── Experiment helpers ────────────────────────────────────────

fn signal_summary(signal: &SystemSignal) -> String {
    match signal {
        SystemSignal::ThermalAlert { level, .. } => format!("Thermal({:?})", level),
        SystemSignal::MemoryPressure { level, .. } => format!("Memory({:?})", level),
        SystemSignal::ComputeGuidance { level, .. } => format!("Compute({:?})", level),
        SystemSignal::EnergyConstraint { level, .. } => format!("Energy({:?})", level),
    }
}

fn action_summary(action: &ResilienceAction) -> String {
    match action {
        ResilienceAction::Evict { target_ratio } => format!("Evict({})", target_ratio),
        ResilienceAction::SwitchBackend { to } => format!("SwitchBackend({:?})", to),
        ResilienceAction::LimitTokens { max_tokens } => format!("LimitTokens({})", max_tokens),
        ResilienceAction::Throttle { delay_ms } => format!("Throttle({}ms)", delay_ms),
        ResilienceAction::Suspend => "Suspend".to_string(),
        ResilienceAction::RejectNew => "RejectNew".to_string(),
        ResilienceAction::RestoreDefaults => "RestoreDefaults".to_string(),
    }
}
