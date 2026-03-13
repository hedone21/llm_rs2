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

    /// Enable profiling (per-op timing, latency, score snapshots).
    #[arg(long, default_value_t = false)]
    profile: bool,

    /// Output directory for profiling data.
    #[arg(long, default_value = "results/profile")]
    profile_dir: String,

    /// Score snapshot interval (1 = every step, 10 = every 10th step).
    #[arg(long, default_value_t = 1)]
    profile_interval: usize,

    /// Comma-separated list of probes: ops,latency,scores,entropy,cache.
    #[arg(long, default_value = "ops,latency,scores")]
    profile_probes: String,

    /// Enable per-KV-head score tracking (for H2O+ analysis).
    #[arg(long, default_value_t = false)]
    profile_per_head: bool,

    /// KV cache data type (f32, f16, or q4)
    #[arg(long, default_value = "f16")]
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

    /// Disable time-normalized scoring (use raw cumulative SUM).
    /// By default, H2O/H2O+ use time-normalized scores to remove cumulative bias.
    #[arg(long, default_value_t = false)]
    h2o_raw_scores: bool,

    // ── Eval-LL mode (log-likelihood evaluation) ──
    /// Enable log-likelihood evaluation mode (downstream task accuracy)
    #[arg(long, default_value_t = false)]
    eval_ll: bool,

    /// Continuation text to evaluate log-likelihood (single task mode)
    #[arg(long)]
    eval_continuation: Option<String>,

    /// Path to evaluation batch JSON file: [{"id","prompt","continuation"}, ...]
    #[arg(long)]
    eval_batch: Option<String>,

    /// Maximum KV cache budget in tokens. Evicts when cache_pos exceeds this.
    /// 0 = no budget limit (default).
    #[arg(long, default_value_t = 0)]
    kv_budget: usize,

    /// KV cache budget as a ratio of prompt length (0.0–1.0).
    /// When set (> 0), overrides --kv-budget per question: budget = prompt_len * ratio.
    /// Matches H2O paper evaluation methodology.
    #[arg(long, default_value_t = 0.0)]
    kv_budget_ratio: f32,

    /// Enable KIVI-style Q2 KV cache compression (ICML 2024).
    /// Mutually exclusive with eviction policies; uses FP32 residual buffer
    /// that batch-quantizes to 2-bit when full.
    #[arg(long, default_value_t = false)]
    kivi: bool,

    /// KIVI residual buffer size in tokens (must be multiple of 32).
    /// Default: 32. Larger values improve quality but use more memory.
    #[arg(long, default_value_t = 32)]
    kivi_residual_size: usize,

    /// Number of threads for parallel computation.
    /// Default: auto-detect CPU core count.
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// KV cache offload mode: none, disk, or zram.
    /// Offloads KV cache to disk files or compressed memory (lossless).
    /// Requires --kv-layout seq and --kv-type f16 or f32.
    #[arg(long, default_value = "none")]
    kv_offload: String,

    /// Directory for disk-based offload storage.
    #[arg(long, default_value = "/tmp/llm_rs2_offload")]
    offload_dir: String,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    #[allow(unused_mut)]
    let mut args = Args::parse();

    // Configure Rayon thread pool: 0 = auto-detect CPU cores
    let num_threads = if args.threads > 0 {
        args.threads
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();
    eprintln!("[Config] Using {} threads", num_threads);

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

    // ── KIVI + eval-ll mode: KiviCache with log-likelihood evaluation ──
    if args.kivi && args.eval_ll {
        return run_kivi_eval_ll(
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
        );
    }

    // ── KIVI mode: separate path with KiviCache ──
    if args.kivi {
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
            args.kivi_residual_size,
            args.num_tokens,
            args.gpu_attn,
            args.experiment_output.as_deref(),
            args.experiment_logits_topk,
            args.experiment_sample_interval,
            &prompt,
            &args.backend,
        );
    }

    // ── Offload mode: separate path with OffloadKVCache ──
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
            args.gpu_attn,
            &prompt,
            &args.backend,
            &args.kv_offload,
            &args.offload_dir,
            &args.kv_type,
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

    // Determine initial KV cache capacity (dynamic grow-on-demand)
    let initial_kv_capacity = if args.eval_ll {
        // Eval-LL batch mode: pre-allocate full capacity to avoid re-allocation
        max_seq_len
    } else if args.initial_kv_capacity > 0 {
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
        acc.set_time_normalize(!args.h2o_raw_scores);
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
        acc.set_time_normalize(!args.h2o_raw_scores);
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

    // Auto-eviction: non-experiment mode evicts automatically.
    // - Sliding window: triggers on memory pressure after each forward pass.
    // - Score-based (H2O/H2O+/D2O): triggers when cache utilization >= 90% capacity,
    //   using force_evict_with_scores to bypass memory pressure checks.
    let auto_eviction = args.eviction_policy != "none" && experiment_schedule.is_none();
    let score_based_eviction = matches!(args.eviction_policy.as_str(), "h2o" | "h2o_plus" | "d2o");

    // ════════════════════════════════════════════════════════════
    //  EVAL-LL MODE: Log-likelihood evaluation for downstream tasks
    // ════════════════════════════════════════════════════════════
    if args.eval_ll {
        return run_eval_ll(
            &args,
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
            &prompt,
        );
    }

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
            profiler: None,
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

    // Inference profiler (only when --profile is set)
    let mut profiler = if args.profile {
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
        // Pre-allocate workspace for generation
        let q_dim = hidden_size;
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
                profiler: profiler.as_mut().map(|p| &mut p.ops),
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

            // Auto-eviction after forward pass (non-experiment mode)
            if auto_eviction {
                let before_len = kv_caches[0].current_pos;
                let capacity = kv_caches[0].capacity();

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
                    if let Some(ref acc) = score_accumulator {
                        if acc.is_active() {
                            cache_manager.force_evict_with_scores(
                                &mut kv_caches,
                                args.eviction_target_ratio,
                                acc.importance_scores(),
                            )?
                        } else {
                            cache_manager.force_evict(&mut kv_caches, args.eviction_target_ratio)?
                        }
                    } else {
                        cache_manager.force_evict(&mut kv_caches, args.eviction_target_ratio)?
                    }
                } else if let Some(ref acc) = score_accumulator {
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

                    if let Some(ref mut acc) = score_accumulator {
                        acc.reset();
                    }
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

            let sample_start = std::time::Instant::now();
            let next_token_id =
                sampling::sample(&mut logits_cpu, &tokens, vocab_size, &sampling_config);
            let sample_us = sample_start.elapsed().as_micros() as u64;

            let now = std::time::Instant::now();
            let tbt = now.duration_since(_last_token_time).as_secs_f64() * 1000.0;
            tbt_values.push(tbt);

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

// ════════════════════════════════════════════════════════════════
//  Eval-LL: Log-likelihood evaluation for downstream task accuracy
// ════════════════════════════════════════════════════════════════

/// KV cache snapshot for save/restore between multi-token choice scoring.
struct KVSnapshot {
    data: Vec<Vec<u8>>, // [layer] = k_bytes ++ v_bytes
    positions: Vec<usize>,
}

fn snapshot_kv(kv_caches: &[KVCache]) -> KVSnapshot {
    let mut data = Vec::with_capacity(kv_caches.len());
    let mut positions = Vec::with_capacity(kv_caches.len());
    for cache in kv_caches {
        let k_size = cache.k_buffer.buffer().size();
        let v_size = cache.v_buffer.buffer().size();
        let mut buf = vec![0u8; k_size + v_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                cache.k_buffer.buffer().as_ptr(),
                buf.as_mut_ptr(),
                k_size,
            );
            std::ptr::copy_nonoverlapping(
                cache.v_buffer.buffer().as_ptr(),
                buf.as_mut_ptr().add(k_size),
                v_size,
            );
        }
        data.push(buf);
        positions.push(cache.current_pos);
    }
    KVSnapshot { data, positions }
}

fn restore_kv(kv_caches: &mut [KVCache], snapshot: &KVSnapshot) {
    for (i, cache) in kv_caches.iter_mut().enumerate() {
        let k_size = cache.k_buffer.buffer().size();
        unsafe {
            std::ptr::copy_nonoverlapping(
                snapshot.data[i].as_ptr(),
                cache.k_buffer.buffer().as_mut_ptr(),
                k_size,
            );
            std::ptr::copy_nonoverlapping(
                snapshot.data[i].as_ptr().add(k_size),
                cache.v_buffer.buffer().as_mut_ptr(),
                snapshot.data[i].len() - k_size,
            );
        }
        cache.current_pos = snapshot.positions[i];
    }
}

/// Grouped eval-LL: each task has a prompt + list of choices.
///
/// Prompt processing:
/// - No budget (kv_budget=0 or prompt fits): full prefill
/// - Budget mode (prompt > kv_budget): prefill first `budget` tokens, then
///   decode remaining tokens one-by-one with eviction
///
/// Choice scoring (multi-token):
/// - Score each choice's full token sequence using KV cache snapshot/restore
/// - NLL = -sum(log_prob(token_i | prompt, tokens[:i])) / n_tokens
///
/// Batch format: [{"id": "q1", "prompt": "...", "choices": [" carbon dioxide", ...]}]
#[allow(clippy::too_many_arguments)]
fn run_eval_ll(
    args: &Args,
    model: &LlamaModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut [KVCache],
    cache_manager: &mut CacheManager,
    score_accumulator: &mut Option<AttentionScoreAccumulator>,
    vocab_size: usize,
    _hidden_size: usize,
    max_seq_len: usize,
    default_prompt: &str,
) -> anyhow::Result<()> {
    let hidden_size = model.config.hidden_size;

    // Load evaluation tasks
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

    // Normalize to grouped format
    struct EvalQuestion {
        id: String,
        prompt: String,
        choices: Vec<String>,
    }
    let mut questions: Vec<EvalQuestion> = Vec::new();
    for task in &raw_tasks {
        if let Some(choices) = task["choices"].as_array() {
            questions.push(EvalQuestion {
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
            questions.push(EvalQuestion {
                id: task["id"].as_str().unwrap_or("unknown").to_string(),
                prompt: task["prompt"]
                    .as_str()
                    .unwrap_or(default_prompt)
                    .to_string(),
                choices: vec![cont.to_string()],
            });
        }
    }

    let ratio_mode = args.kv_budget_ratio > 0.0;
    let budget_mode = args.kv_budget > 0 || ratio_mode;
    eprintln!(
        "[Eval-LL] {} questions, policy={}, kv_budget={}, kv_budget_ratio={}, mode={}",
        questions.len(),
        args.eviction_policy,
        args.kv_budget,
        args.kv_budget_ratio,
        if budget_mode {
            "chunked"
        } else {
            "full-prefill"
        }
    );

    // Pre-allocate decode buffers (needed for multi-token continuation scoring)
    let dl_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut decode_logits =
        Tensor::new(Shape::new(vec![1, 1, vocab_size]), dl_buf, backend.clone());
    let xg_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let mut x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());
    let q_dim = hidden_size;
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

    let mut results: Vec<serde_json::Value> = Vec::new();
    let overall_start = std::time::Instant::now();

    for (q_idx, question) in questions.iter().enumerate() {
        let q_start = std::time::Instant::now();

        // Reset KV caches
        for cache in kv_caches.iter_mut() {
            cache.current_pos = 0;
        }
        if let Some(acc) = score_accumulator.as_mut() {
            acc.reset();
        }

        // Tokenize prompt
        let prompt_enc = tokenizer
            .encode(question.prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!(e))?;
        let prompt_ids: Vec<u32> = prompt_enc.get_ids().to_vec();
        let prompt_len = prompt_ids.len();

        if prompt_len > max_seq_len {
            eprintln!(
                "[Eval-LL] {}: prompt too long ({} > {}), skipping",
                question.id, prompt_len, max_seq_len
            );
            continue;
        }

        let mut eviction_count: usize = 0;
        let mut evicted_total: usize = 0;
        let start_pos_after_prompt: usize;

        // Compute effective budget: ratio-based (per question) or absolute
        let effective_budget = if ratio_mode {
            let b = (prompt_len as f32 * args.kv_budget_ratio).round() as usize;
            b.max(4) // minimum 4 tokens (protected prefix)
        } else {
            args.kv_budget
        };

        if ratio_mode {
            eprintln!(
                "[Eval-LL] {}: prompt_len={}, budget={} (ratio={:.0}%)",
                question.id,
                prompt_len,
                effective_budget,
                args.kv_budget_ratio * 100.0
            );
        }

        // ── PROMPT PROCESSING ──
        let prompt_logits_cpu: Vec<f32> = if budget_mode && prompt_len > effective_budget {
            // ═══ CHUNKED PREFILL + DECODE (budget-constrained) ═══
            let first_chunk_len = effective_budget;
            let cpu_buf = Galloc::new().alloc(first_chunk_len * 4, DType::U8)?;
            unsafe {
                let ptr = cpu_buf.as_mut_ptr() as *mut u32;
                std::ptr::copy_nonoverlapping(prompt_ids.as_ptr(), ptr, first_chunk_len);
            }
            let cpu_input = Tensor::new(
                Shape::new(vec![1, first_chunk_len]),
                cpu_buf,
                Arc::new(CpuBackend::new()),
            );
            let input_tensor = backend.copy_from(&cpu_input)?;

            let prefill_buf = memory.alloc(first_chunk_len * vocab_size * 4, DType::F32)?;
            let mut prefill_logits = Tensor::new(
                Shape::new(vec![1, first_chunk_len, vocab_size]),
                prefill_buf,
                backend.clone(),
            );

            model.forward_into(LlamaModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos: 0,
                kv_caches,
                backend,
                memory,
                logits_out: &mut prefill_logits,
                x_gen: None,
                workspace: None,
                use_gpu_attn: args.gpu_attn,
                score_accumulator: None,
                profiler: None,
            })?;

            let mut start_pos = first_chunk_len;

            // Evict to make room
            if kv_caches[0].current_pos > effective_budget {
                let ratio = effective_budget as f32 / kv_caches[0].current_pos as f32;
                let r = cache_manager.force_evict(kv_caches, ratio)?;
                if r.evicted {
                    eviction_count += 1;
                    evicted_total += r.tokens_removed;
                }
            }

            // Decode remaining prompt tokens one-by-one
            for &token_id in &prompt_ids[first_chunk_len..] {
                unsafe {
                    *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = token_id;
                }
                let gen_input = backend.copy_from(&cpu_gen_input)?;

                if let Some(acc) = score_accumulator.as_mut() {
                    acc.begin_step();
                }

                model.forward_into(LlamaModelForwardArgs {
                    input_tokens: &gen_input,
                    start_pos,
                    kv_caches,
                    backend,
                    memory,
                    logits_out: &mut decode_logits,
                    x_gen: Some(&mut x_gen),
                    workspace: Some(&mut gen_ws),
                    use_gpu_attn: args.gpu_attn,
                    score_accumulator: score_accumulator.as_mut(),
                    profiler: None,
                })?;
                start_pos += 1;

                if kv_caches[0].current_pos > effective_budget {
                    let ratio = effective_budget as f32 / kv_caches[0].current_pos as f32;
                    let evict_result = if let Some(acc) = score_accumulator.as_ref() {
                        cache_manager.force_evict_with_scores(
                            kv_caches,
                            ratio,
                            acc.importance_scores(),
                        )?
                    } else {
                        cache_manager.force_evict(kv_caches, ratio)?
                    };
                    if evict_result.evicted {
                        eviction_count += 1;
                        evicted_total += evict_result.tokens_removed;
                    }
                }
            }

            start_pos_after_prompt = start_pos;

            // Read logits from last decode step
            let mut logits_cpu = vec![0.0f32; vocab_size];
            unsafe {
                let ptr = logits_cpu.as_mut_ptr() as *mut u8;
                let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
                backend.read_buffer(&decode_logits, slice)?;
            }
            logits_cpu
        } else {
            // ═══ FULL PREFILL ═══
            let cpu_indices_buf = Galloc::new().alloc(prompt_len * 4, DType::U8)?;
            unsafe {
                let ptr = cpu_indices_buf.as_mut_ptr() as *mut u32;
                std::ptr::copy_nonoverlapping(prompt_ids.as_ptr(), ptr, prompt_len);
            }
            let cpu_input = Tensor::new(
                Shape::new(vec![1, prompt_len]),
                cpu_indices_buf,
                Arc::new(CpuBackend::new()),
            );
            let input_tensor = backend.copy_from(&cpu_input)?;

            let prefill_logits_buf = memory.alloc(prompt_len * vocab_size * 4, DType::F32)?;
            let mut prefill_logits = Tensor::new(
                Shape::new(vec![1, prompt_len, vocab_size]),
                prefill_logits_buf,
                backend.clone(),
            );

            model.forward_into(LlamaModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos: 0,
                kv_caches,
                backend,
                memory,
                logits_out: &mut prefill_logits,
                x_gen: None,
                workspace: None,
                use_gpu_attn: args.gpu_attn,
                score_accumulator: None,
                profiler: None,
            })?;

            start_pos_after_prompt = prompt_len;

            // Read last-position logits
            let mut all_logits = vec![0.0f32; prompt_len * vocab_size];
            unsafe {
                let ptr = all_logits.as_mut_ptr() as *mut u8;
                let slice = std::slice::from_raw_parts_mut(ptr, all_logits.len() * 4);
                backend.read_buffer(&prefill_logits, slice)?;
            }
            let off = (prompt_len - 1) * vocab_size;
            all_logits[off..off + vocab_size].to_vec()
        };

        // ── SNAPSHOT KV cache after prompt (for multi-token choice scoring) ──
        let kv_snap = snapshot_kv(kv_caches);

        // ── SCORE EACH CHOICE ──
        let mut choice_nlls: Vec<f64> = Vec::new();
        let mut choice_byte_lens: Vec<usize> = Vec::new();
        let mut choice_token_lens: Vec<usize> = Vec::new();
        for choice_text in &question.choices {
            // Tokenize full text to extract continuation tokens
            let full_text = format!("{}{}", question.prompt, choice_text);
            let full_enc = tokenizer
                .encode(full_text.as_str(), true)
                .map_err(|e| anyhow::anyhow!(e))?;
            let full_ids: Vec<u32> = full_enc.get_ids().to_vec();
            let cont_ids: Vec<u32> = full_ids[prompt_ids.len()..].to_vec();

            if cont_ids.is_empty() {
                choice_nlls.push(f64::INFINITY);
                continue;
            }

            // First token scored from prompt logits
            let mut total_nll =
                -sampling::compute_log_prob(&prompt_logits_cpu, cont_ids[0], vocab_size);

            // Multi-token: decode remaining tokens, accumulating NLL
            if cont_ids.len() > 1 {
                // Restore KV cache to post-prompt state
                restore_kv(kv_caches, &kv_snap);
                let mut sp = start_pos_after_prompt;

                for token_pair in cont_ids.windows(2) {
                    let input_token = token_pair[0];
                    let target_token = token_pair[1];

                    unsafe {
                        *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = input_token;
                    }
                    let gen_input = backend.copy_from(&cpu_gen_input)?;

                    model.forward_into(LlamaModelForwardArgs {
                        input_tokens: &gen_input,
                        start_pos: sp,
                        kv_caches,
                        backend,
                        memory,
                        logits_out: &mut decode_logits,
                        x_gen: Some(&mut x_gen),
                        workspace: Some(&mut gen_ws),
                        use_gpu_attn: args.gpu_attn,
                        score_accumulator: None,
                        profiler: None,
                    })?;
                    sp += 1;

                    // Read logits and score target token
                    let mut step_logits = vec![0.0f32; vocab_size];
                    unsafe {
                        let ptr = step_logits.as_mut_ptr() as *mut u8;
                        let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
                        backend.read_buffer(&decode_logits, slice)?;
                    }
                    total_nll -= sampling::compute_log_prob(&step_logits, target_token, vocab_size);
                }
            }

            // Store raw total NLL (Python handles normalization strategy)
            choice_nlls.push(total_nll);
            choice_byte_lens.push(choice_text.len());
            choice_token_lens.push(cont_ids.len());
        }

        // Restore KV cache for consistency (not strictly needed as we reset next iter)
        restore_kv(kv_caches, &kv_snap);

        // Find predicted using byte-length-normalized NLL (acc_norm, lm-eval style)
        let predicted_norm: usize = choice_nlls
            .iter()
            .zip(choice_byte_lens.iter())
            .enumerate()
            .min_by(|(_, (a, al)), (_, (b, bl))| {
                let a_norm = *a / **al as f64;
                let b_norm = *b / **bl as f64;
                a_norm
                    .partial_cmp(&b_norm)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Also compute raw prediction (acc, no normalization)
        let predicted_raw: usize = choice_nlls
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let elapsed_q = q_start.elapsed().as_secs_f64();
        eprintln!(
            "[Eval-LL] {}/{} {} — norm={} raw={} nlls=[{}] evict={} {:.1}s",
            q_idx + 1,
            questions.len(),
            question.id,
            predicted_norm,
            predicted_raw,
            choice_nlls
                .iter()
                .map(|v| format!("{:.3}", v))
                .collect::<Vec<_>>()
                .join(","),
            eviction_count,
            elapsed_q,
        );

        results.push(serde_json::json!({
            "id": question.id,
            "choice_nlls": choice_nlls,
            "choice_byte_lens": choice_byte_lens,
            "choice_token_lens": choice_token_lens,
            "predicted": predicted_norm,
            "predicted_raw": predicted_raw,
            "n_choices": question.choices.len(),
            "n_prompt_tokens": prompt_len,
            "effective_budget": effective_budget,
            "eviction_count": eviction_count,
            "evicted_tokens": evicted_total,
            "final_cache_pos": kv_caches[0].current_pos,
        }));
    }

    let elapsed = overall_start.elapsed().as_secs_f64();
    let output = serde_json::json!({
        "results": results,
        "config": {
            "model": args.model_path,
            "eviction_policy": args.eviction_policy,
            "kv_budget": args.kv_budget,
            "kv_budget_ratio": args.kv_budget_ratio,
            "max_seq_len": max_seq_len,
            "kv_type": args.kv_type,
            "h2o_keep_ratio": args.h2o_keep_ratio,
            "h2o_decay": args.h2o_decay,
            "time_normalized": !args.h2o_raw_scores,
        },
        "wall_time_s": elapsed,
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

// ── KIVI + Eval-LL mode: log-likelihood evaluation with KiviCache ───────────

#[allow(clippy::too_many_arguments)]
fn run_kivi_eval_ll(
    args: &Args,
    model: &LlamaModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    max_seq_len: usize,
    residual_size: usize,
) -> anyhow::Result<()> {
    use llm_rs2::core::kv_cache::KVCacheOps;

    let hidden_size = model.config.hidden_size;
    let vocab_size = model.config.vocab_size;

    eprintln!(
        "[KIVI-Eval] Q2 KV cache, residual_size={}, max_seq_len={}",
        residual_size, max_seq_len
    );

    // Load evaluation tasks (same logic as run_eval_ll)
    let raw_tasks: Vec<serde_json::Value> = if let Some(ref path) = args.eval_batch {
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("Failed to open eval batch {}: {}", path, e))?;
        serde_json::from_reader(file)?
    } else {
        let cont = args.eval_continuation.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--eval-ll requires --eval-continuation or --eval-batch")
        })?;
        let prompt_text = args.prompt.as_str();
        vec![serde_json::json!({
            "id": "single",
            "prompt": prompt_text,
            "choices": [cont],
        })]
    };

    struct EvalQuestion {
        id: String,
        prompt: String,
        choices: Vec<String>,
    }
    let mut questions: Vec<EvalQuestion> = Vec::new();
    for task in &raw_tasks {
        if let Some(choices) = task["choices"].as_array() {
            questions.push(EvalQuestion {
                id: task["id"].as_str().unwrap_or("unknown").to_string(),
                prompt: task["prompt"].as_str().unwrap_or("").to_string(),
                choices: choices
                    .iter()
                    .filter_map(|c| c.as_str().map(|s| s.to_string()))
                    .collect(),
            });
        }
    }

    eprintln!(
        "[KIVI-Eval] {} questions, kivi_res={}, policy=none (KiVi internal compression)",
        questions.len(),
        residual_size,
    );

    // Pre-allocate decode buffers
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

    let mut results: Vec<serde_json::Value> = Vec::new();
    let overall_start = std::time::Instant::now();

    // Create KiviCache template
    let mut kv_caches: Vec<KiviCache> = (0..num_layers)
        .map(|_| KiviCache::new(kv_heads, head_dim, max_seq_len, residual_size))
        .collect();

    for (q_idx, question) in questions.iter().enumerate() {
        let q_start = std::time::Instant::now();

        // Reset KiviCaches
        for cache in kv_caches.iter_mut() {
            cache.reset();
        }

        // Tokenize prompt
        let prompt_enc = tokenizer
            .encode(question.prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!(e))?;
        let prompt_ids: Vec<u32> = prompt_enc.get_ids().to_vec();
        let prompt_len = prompt_ids.len();

        if prompt_len > max_seq_len {
            eprintln!(
                "[KIVI-Eval] {}: prompt too long ({} > {}), skipping",
                question.id, prompt_len, max_seq_len
            );
            continue;
        }

        // ── FULL PREFILL ──
        let cpu_indices_buf = Galloc::new().alloc(prompt_len * 4, DType::U8)?;
        unsafe {
            let ptr = cpu_indices_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(prompt_ids.as_ptr(), ptr, prompt_len);
        }
        let cpu_input = Tensor::new(
            Shape::new(vec![1, prompt_len]),
            cpu_indices_buf,
            Arc::new(CpuBackend::new()),
        );
        let input_tensor = backend.copy_from(&cpu_input)?;

        let prefill_logits_buf = memory.alloc(prompt_len * vocab_size * 4, DType::F32)?;
        let mut prefill_logits = Tensor::new(
            Shape::new(vec![1, prompt_len, vocab_size]),
            prefill_logits_buf,
            backend.clone(),
        );

        model.forward_into(LlamaModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: 0,
            kv_caches: &mut kv_caches,
            backend,
            memory: memory.as_ref(),
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            use_gpu_attn: args.gpu_attn,
            score_accumulator: None,
            profiler: None,
        })?;

        let start_pos_after_prompt = prompt_len;

        // Read last-position logits
        let mut all_logits = vec![0.0f32; prompt_len * vocab_size];
        unsafe {
            let ptr = all_logits.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, all_logits.len() * 4);
            backend.read_buffer(&prefill_logits, slice)?;
        }
        let off = (prompt_len - 1) * vocab_size;
        let prompt_logits_cpu = all_logits[off..off + vocab_size].to_vec();

        // ── SNAPSHOT KV cache (clone) ──
        let kv_snap = kv_caches.clone();

        // ── SCORE EACH CHOICE ──
        let mut choice_nlls: Vec<f64> = Vec::new();
        let mut choice_byte_lens: Vec<usize> = Vec::new();
        let mut choice_token_lens: Vec<usize> = Vec::new();
        for choice_text in &question.choices {
            let full_text = format!("{}{}", question.prompt, choice_text);
            let full_enc = tokenizer
                .encode(full_text.as_str(), true)
                .map_err(|e| anyhow::anyhow!(e))?;
            let full_ids: Vec<u32> = full_enc.get_ids().to_vec();
            let cont_ids: Vec<u32> = full_ids[prompt_ids.len()..].to_vec();

            if cont_ids.is_empty() {
                choice_nlls.push(f64::INFINITY);
                choice_byte_lens.push(choice_text.len());
                choice_token_lens.push(0);
                continue;
            }

            // First token scored from prompt logits
            let mut total_nll =
                -sampling::compute_log_prob(&prompt_logits_cpu, cont_ids[0], vocab_size);

            // Multi-token: decode remaining tokens, accumulating NLL
            if cont_ids.len() > 1 {
                // Restore KV cache to post-prompt state
                kv_caches.clone_from_slice(&kv_snap);
                let mut sp = start_pos_after_prompt;

                for token_pair in cont_ids.windows(2) {
                    let input_token = token_pair[0];
                    let target_token = token_pair[1];

                    unsafe {
                        *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = input_token;
                    }
                    let gen_input = backend.copy_from(&cpu_gen_input)?;

                    model.forward_into(LlamaModelForwardArgs {
                        input_tokens: &gen_input,
                        start_pos: sp,
                        kv_caches: &mut kv_caches,
                        backend,
                        memory: memory.as_ref(),
                        logits_out: &mut decode_logits,
                        x_gen: Some(&mut x_gen),
                        workspace: Some(&mut gen_ws),
                        use_gpu_attn: args.gpu_attn,
                        score_accumulator: None,
                        profiler: None,
                    })?;
                    sp += 1;

                    let mut step_logits = vec![0.0f32; vocab_size];
                    unsafe {
                        let ptr = step_logits.as_mut_ptr() as *mut u8;
                        let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
                        backend.read_buffer(&decode_logits, slice)?;
                    }
                    total_nll -= sampling::compute_log_prob(&step_logits, target_token, vocab_size);
                }
            }

            choice_nlls.push(total_nll);
            choice_byte_lens.push(choice_text.len());
            choice_token_lens.push(cont_ids.len());
        }

        // Restore for next iteration consistency
        kv_caches.clone_from_slice(&kv_snap);

        // Find predicted (byte-length-normalized NLL, acc_norm)
        let predicted_norm: usize = choice_nlls
            .iter()
            .zip(choice_byte_lens.iter())
            .enumerate()
            .min_by(|(_, (a, al)), (_, (b, bl))| {
                let a_norm = *a / **al as f64;
                let b_norm = *b / **bl as f64;
                a_norm
                    .partial_cmp(&b_norm)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let predicted_raw: usize = choice_nlls
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let elapsed_q = q_start.elapsed().as_secs_f64();
        eprintln!(
            "[KIVI-Eval] {}/{} {} — norm={} raw={} nlls=[{}] Q2_tokens={} {:.1}s",
            q_idx + 1,
            questions.len(),
            question.id,
            predicted_norm,
            predicted_raw,
            choice_nlls
                .iter()
                .map(|v| format!("{:.3}", v))
                .collect::<Vec<_>>()
                .join(","),
            kv_caches[0].q2_tokens,
            elapsed_q,
        );

        results.push(serde_json::json!({
            "id": question.id,
            "choice_nlls": choice_nlls,
            "choice_byte_lens": choice_byte_lens,
            "choice_token_lens": choice_token_lens,
            "predicted": predicted_norm,
            "predicted_raw": predicted_raw,
            "n_choices": question.choices.len(),
            "n_prompt_tokens": prompt_len,
            "effective_budget": 0,
            "eviction_count": 0,
            "evicted_tokens": 0,
            "final_cache_pos": kv_caches[0].current_pos(),
            "kivi_q2_tokens": kv_caches[0].q2_tokens,
            "kivi_res_pos": kv_caches[0].res_pos,
        }));
    }

    let elapsed = overall_start.elapsed().as_secs_f64();
    let output = serde_json::json!({
        "results": results,
        "config": {
            "model": args.model_path,
            "eviction_policy": "kivi",
            "kivi_residual_size": residual_size,
            "max_seq_len": max_seq_len,
            "kv_type": "q2+f32_residual",
        },
        "wall_time_s": elapsed,
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

// ── KIVI mode: KiviCache-based inference ────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_kivi(
    model: &LlamaModel,
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
    gpu_attn: bool,
    experiment_output: Option<&str>,
    experiment_logits_topk: usize,
    experiment_sample_interval: usize,
    prompt: &str,
    backend_name: &str,
) -> anyhow::Result<()> {
    use llm_rs2::core::kv_cache::KVCacheOps;

    println!(
        "[KIVI] Q2 KV cache enabled — residual_size={}, max_seq_len={}",
        residual_size, max_seq_len
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
        .map(|_| KiviCache::new(kv_heads, head_dim, max_seq_len, residual_size))
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

        model.forward_into(LlamaModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos,
            kv_caches: &mut kv_caches,
            backend,
            memory: memory.as_ref(),
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            use_gpu_attn: gpu_attn,
            score_accumulator: None,
            profiler: None,
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

    let eos_id = tokenizer
        .get_vocab(true)
        .get("</s>")
        .copied()
        .unwrap_or(u32::MAX);

    let decode_start = std::time::Instant::now();
    let mut generated_count = 0usize;
    let mut tbt_values: Vec<f64> = Vec::new();
    let mut forward_ms_values: Vec<f64> = Vec::new();
    let mut last_token_time = std::time::Instant::now();

    for decode_idx in 0..(num_tokens - 1) {
        if kv_caches[0].current_pos() >= max_seq_len {
            eprintln!("\n[Stopped: Max context length reached]");
            break;
        }

        let last_token = tokens[tokens.len() - 1];
        unsafe {
            *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_token;
        }
        let gen_input = backend.copy_from(&cpu_gen_input)?;

        let fwd_start = std::time::Instant::now();
        model.forward_into(LlamaModelForwardArgs {
            input_tokens: &gen_input,
            start_pos,
            kv_caches: &mut kv_caches,
            backend,
            memory: memory.as_ref(),
            logits_out: &mut logits,
            x_gen: Some(&mut x_gen),
            workspace: Some(&mut gen_ws),
            use_gpu_attn: gpu_attn,
            score_accumulator: None,
            profiler: None,
        })?;
        let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;
        forward_ms_values.push(forward_ms);

        start_pos += 1;
        generated_count += 1;

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

        let next_token = sampling::sample(&mut logits_cpu, &tokens, vocab_size, sampling_config);

        let now = std::time::Instant::now();
        let tbt = now.duration_since(last_token_time).as_secs_f64() * 1000.0;
        tbt_values.push(tbt);
        last_token_time = now;

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

        if next_token == eos_id {
            break;
        }
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
    model: &LlamaModel,
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
    gpu_attn: bool,
    _prompt: &str,
    _backend_name: &str,
    offload_mode: &str,
    offload_dir: &str,
    kv_type_str: &str,
) -> anyhow::Result<()> {
    use llm_rs2::core::kv_cache::KVCacheOps;
    use llm_rs2::core::offload::OffloadKVCache;
    use llm_rs2::core::offload::disk_store::DiskStore;
    use llm_rs2::core::offload::zram_store::ZramStore;

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

    eprintln!(
        "[Offload] mode={}, dtype={:?}, layers={}, token_bytes={}, max_seq={}",
        offload_mode, kv_dtype, num_layers, token_bytes, max_seq_len,
    );

    // Create OffloadKVCache per layer
    let mut kv_caches: Vec<OffloadKVCache> = (0..num_layers)
        .map(|layer_id| {
            let store: Box<dyn llm_rs2::core::offload::store::OffloadStore> = match offload_mode {
                "disk" => {
                    let dir = std::path::Path::new(offload_dir);
                    Box::new(
                        DiskStore::new(dir, layer_id, token_bytes)
                            .expect("Failed to create DiskStore"),
                    )
                }
                "zram" => {
                    let residual_cap = 64;
                    Box::new(ZramStore::new(token_bytes, kv_dtype.size(), residual_cap))
                }
                _ => panic!("Unknown offload mode: {}", offload_mode),
            };
            OffloadKVCache::new(layer_id, kv_heads, head_dim, kv_dtype, max_seq_len, store)
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
        model.forward_into(LlamaModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos,
            kv_caches: &mut kv_caches,
            backend,
            memory: memory.as_ref(),
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            use_gpu_attn: gpu_attn,
            score_accumulator: None,
            profiler: None,
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

    // === DECODE with per-layer prefetch ===
    let cpu_gen_indices_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_indices_buf,
        Arc::new(CpuBackend::new()),
    );

    let eos_id = tokenizer
        .get_vocab(true)
        .get("</s>")
        .copied()
        .unwrap_or(u32::MAX);

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
        let gen_input = backend.copy_from(&cpu_gen_input)?;

        // R-P5: Reset preload flags at token boundary
        for cache in kv_caches.iter_mut() {
            cache.reset_preload();
        }

        let fwd_start = std::time::Instant::now();
        model.forward_into_offload(LlamaModelForwardArgs {
            input_tokens: &gen_input,
            start_pos,
            kv_caches: &mut kv_caches,
            backend,
            memory: memory.as_ref(),
            logits_out: &mut logits,
            x_gen: Some(&mut x_gen),
            workspace: Some(&mut gen_ws),
            use_gpu_attn: gpu_attn,
            score_accumulator: None,
            profiler: None,
        })?;
        let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;
        forward_ms_values.push(forward_ms);

        start_pos += 1;
        generated_count += 1;

        // Read logits to CPU
        let mut logits_cpu = vec![0.0f32; vocab_size];
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
            backend.read_buffer(&logits, slice)?;
        }

        let next_token = sampling::sample(&mut logits_cpu, &tokens, vocab_size, sampling_config);

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

        if next_token == eos_id {
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

    println!("\nDone.");
    println!("TTFT: {:.2} ms", ttft_ms);
    println!(
        "Avg forward: {:.2} ms, Avg TBT: {:.2} ms ({:.1} tok/s)",
        avg_forward_ms, avg_tbt, tok_per_s,
    );

    Ok(())
}
