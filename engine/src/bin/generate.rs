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
use llm_rs2::models::transformer::{TransformerModel, TransformerModelForwardArgs};
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

    /// Backend to use: "cpu" or "opencl" (GPU secondary auto-initialized when available)
    #[arg(short, long, default_value = "cpu")]
    backend: String,

    /// Auto-switch CPU→GPU at this token count (0=disabled). Requires GPU availability.
    #[arg(long, default_value_t = 0)]
    switch_threshold: usize,

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

    /// Disable GPU kernel plan for decode (fallback to forward_into every token)
    #[arg(long, default_value_t = false)]
    no_gpu_plan: bool,

    /// Disable PrefillWorkspace (fallback to per-layer alloc during prefill)
    #[arg(long, default_value_t = false)]
    no_prefill_ws: bool,

    /// Chunked prefill: split long prompts into chunks to limit peak memory.
    /// 0 = disabled (default, process entire prompt as one batch).
    #[arg(long, default_value_t = 0)]
    prefill_chunk_size: usize,

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

    /// Model weight data type (f16 or q4). f16 = no quantization, q4 = Q4_0 quantization at load time.
    #[arg(long, default_value = "f16")]
    weight_dtype: String,

    /// KV cache data type (f32, f16, or q4)
    #[arg(long, default_value = "f16")]
    kv_type: String,

    /// Eviction policy for KV cache management (none, sliding, streaming, h2o, h2o_plus, d2o)
    #[arg(long, default_value = "none")]
    eviction_policy: String,

    /// Window size for sliding window / streaming eviction (tokens).
    /// Default: 1024 for sliding, 2000 for streaming.
    #[arg(long, default_value_t = 1024)]
    eviction_window: usize,

    /// Number of attention sink tokens to preserve (StreamingLLM).
    /// Only used with --eviction-policy streaming.
    #[arg(long, default_value_t = 4)]
    sink_size: usize,

    /// StreamingLLM recent window size. 0 = auto (kv_budget - sink_size).
    /// Only used with --eviction-policy streaming.
    #[arg(long, default_value_t = 0)]
    streaming_window: usize,

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

    /// D2O EMA old-threshold weight α (official default 0.5)
    #[arg(long, default_value_t = 0.5)]
    d2o_ema_alpha: f32,

    /// D2O EMA new-mean weight β (official default 0.5)
    #[arg(long, default_value_t = 0.5)]
    d2o_ema_beta: f32,

    /// Enable D2O layer-level dynamic allocation (uses per-layer attention variance from prefill)
    #[arg(long, default_value_t = false)]
    d2o_layer_alloc: bool,

    /// Protected layers for D2O layer allocation (comma-separated layer indices, e.g. 0,1,2)
    #[arg(long, value_delimiter = ',')]
    d2o_protected_layers: Option<Vec<usize>>,

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

    /// Ignore EOS token and continue generating (for long-running experiments)
    #[arg(long, default_value_t = false)]
    ignore_eos: bool,

    /// Target TBT in milliseconds for pacing (0=disabled).
    /// After each decode step, sleeps to maintain the target TBT.
    /// Used for fair resource comparison across different actions at the same QoS.
    #[arg(long, default_value_t = 0.0)]
    target_tbt: f64,

    /// Path to write per-token TBT JSONL log.
    /// Each line: {"token_idx":N,"tbt_ms":X,"forward_ms":Y,"cache_pos":Z,"pacing_ms":W}
    #[arg(long)]
    tbt_log: Option<String>,

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

    /// QCF variant to compute: "attn" (default), "caote", or "both".
    #[arg(long, default_value = "attn")]
    qcf_mode: String,

    /// Enable AWQE + AW-VOPR metrics for KIVI.
    #[arg(long, default_value_t = false)]
    awqe: bool,

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

    /// Enable dynamic KV cache quantization for resilience.
    /// Starts with bits=16 (F16-equivalent KiviCache) and allows runtime
    /// transition to Q2/Q4/Q8 via kv_quant_dynamic resilience command.
    #[arg(long, default_value_t = false)]
    kv_dynamic_quant: bool,

    /// KIVI quantization bit-width (2, 4, or 8). Default: 2.
    #[arg(long, default_value_t = 2)]
    kivi_bits: u8,

    /// KIVI residual buffer size in tokens (must be multiple of 32).
    /// Default: 32. Larger values improve quality but use more memory.
    #[arg(long, default_value_t = 32)]
    kivi_residual_size: usize,

    /// Number of threads for parallel computation.
    /// Default: auto-detect CPU core count.
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// KV cache offload mode: none, raw (in-memory), or disk (file-based).
    /// Requires --kv-layout seq and --kv-type f16 or f32.
    #[arg(long, default_value = "none")]
    kv_offload: String,

    /// Directory for disk offload files (used with --kv-offload disk).
    /// Defaults to system temp dir if not specified.
    #[arg(long, default_value = "")]
    offload_path: String,

    /// Maximum adaptive prefetch depth for offload KV cache pipeline.
    /// Higher values use more memory but can hide preload latency.
    #[arg(long, default_value_t = 4)]
    max_prefetch_depth: usize,

    /// Use Rayon par_chunks_mut instead of SpinPool for F16 matmul (A/B benchmarking).
    #[arg(long, default_value_t = false)]
    use_rayon: bool,

    /// Path to reference text file for perplexity evaluation (teacher-forcing).
    /// Measures PPL and collects proxy metrics during eviction.
    #[arg(long)]
    ppl: Option<String>,

    /// Comma-separated layer indices to skip (both attn+mlp).
    /// Example: --skip-layers 1,3,5,7
    #[arg(long, value_delimiter = ',')]
    skip_layers: Option<Vec<usize>>,

    /// Skip ratio (0.0-1.0). Uses SkipConfig::uniform_init() to select layers.
    #[arg(long)]
    skip_ratio: Option<f32>,

    /// Dump per-layer importance table and exit (no inference).
    /// Runs prefill with ImportanceCollector on the given prompt.
    #[arg(long, default_value_t = false)]
    dump_importance: bool,
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

    // Wire Rayon vs SpinPool toggle
    #[cfg(target_arch = "aarch64")]
    if args.use_rayon {
        llm_rs2::backend::cpu::neon::USE_RAYON.store(true, std::sync::atomic::Ordering::Relaxed);
        eprintln!("[Config] F16 matmul: Rayon (par_chunks_mut)");
    }

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
    eprintln!("[Profile] Event: ModelLoadStart");
    eprintln!("Loading model from {}", model_path);

    // Backend initialization: primary backend + secondary for SwitchHw resilience.
    // GPU secondary is auto-initialized when available (soft failure OK).
    #[allow(clippy::type_complexity)]
    let (mut backend, memory, gpu_backend_arc, gpu_memory_arc, mut is_gpu): (
        Arc<dyn Backend>,
        Arc<dyn Memory>,
        Option<Arc<dyn Backend>>,
        Option<Arc<dyn Memory>>,
        bool,
    ) = match args.backend.as_str() {
        "cpu" => {
            let cpu = Arc::new(CpuBackend::new()) as Arc<dyn Backend>;
            let cpu_mem: Arc<dyn Memory> = Arc::new(Galloc::new());
            // Try to init GPU as secondary for SwitchHw resilience
            #[cfg(feature = "opencl")]
            let (gpu_be, gpu_mem_arc) = match llm_rs2::backend::opencl::OpenCLBackend::new() {
                Ok(gpu_concrete) => {
                    let gpu_concrete = Arc::new(gpu_concrete);
                    let gm: Arc<dyn Memory> =
                        Arc::new(llm_rs2::backend::opencl::memory::OpenCLMemory::new(
                            gpu_concrete.context.clone(),
                            gpu_concrete.queue.clone(),
                            args.zero_copy,
                        ));
                    let g = gpu_concrete as Arc<dyn Backend>;
                    eprintln!("[Backend] CPU primary, GPU secondary available (SwitchHw ready)");
                    (Some(g), Some(gm))
                }
                Err(e) => {
                    eprintln!("[Backend] CPU only (GPU init failed: {})", e);
                    (None, None)
                }
            };
            #[cfg(not(feature = "opencl"))]
            let (gpu_be, gpu_mem_arc): (
                Option<Arc<dyn Backend>>,
                Option<Arc<dyn Memory>>,
            ) = (None, None);
            (cpu, cpu_mem, gpu_be, gpu_mem_arc, false)
        }
        #[cfg(feature = "opencl")]
        "opencl" | "gpu" => {
            let gpu_concrete = Arc::new(llm_rs2::backend::opencl::OpenCLBackend::new()?);
            let gpu_mem: Arc<dyn Memory> =
                Arc::new(llm_rs2::backend::opencl::memory::OpenCLMemory::new(
                    gpu_concrete.context.clone(),
                    gpu_concrete.queue.clone(),
                    args.zero_copy,
                ));
            let gpu: Arc<dyn Backend> = gpu_concrete;
            // GPU is primary; keep a ref as secondary for SwitchHw round-trip
            (gpu.clone(), gpu_mem.clone(), Some(gpu), Some(gpu_mem), true)
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            let gpu_concrete = Arc::new(llm_rs2::backend::cuda::CudaBackend::new()?);
            let gpu_mem: Arc<dyn Memory> =
                Arc::new(llm_rs2::backend::cuda::memory::CudaMemory::new());
            let gpu: Arc<dyn Backend> = gpu_concrete;
            (gpu.clone(), gpu_mem.clone(), Some(gpu), Some(gpu_mem), true)
        }
        _ => anyhow::bail!(
            "Unknown backend: {}. Use cpu, opencl, or cuda.",
            args.backend
        ),
    };
    // cpu_backend_arc: always available for migration and SwitchHw fallback.
    let cpu_backend_arc: Arc<dyn Backend> = if args.backend == "cpu" {
        backend.clone()
    } else {
        Arc::new(CpuBackend::new())
    };
    let cpu_memory_arc: Arc<dyn Memory> = if args.backend == "cpu" {
        memory.clone()
    } else {
        Arc::new(Galloc::new())
    };
    let w_dtype = match args.weight_dtype.as_str() {
        "f16" => DType::F16,
        "q4" | "q4_0" => DType::Q4_0,
        _ => anyhow::bail!(
            "Unknown weight-dtype: {}. Use f16 or q4.",
            args.weight_dtype
        ),
    };
    eprintln!("[Config] Weight dtype: {:?}", w_dtype);
    let mut model =
        TransformerModel::load_with_dtype(model_path, backend.clone(), &*memory, w_dtype)?;

    // When CPU primary + GPU secondary: migrate weights to GPU zero-copy memory
    // (MadviseableGPUBuffer = CL_MEM_USE_HOST_PTR: host Vec always valid + cl_mem for GPU).
    // This enables CPU→GPU SwitchHw without weight re-upload.
    #[cfg(feature = "opencl")]
    if !is_gpu && let (Some(gpu_be), Some(gpu_mem)) = (&gpu_backend_arc, &gpu_memory_arc) {
        match model.migrate_weights_to_gpu(gpu_mem.as_ref(), gpu_be) {
            Ok(n) => eprintln!("[Backend] Migrated {} weight tensors to GPU zero-copy", n),
            Err(e) => eprintln!("[Backend] Weight migration skipped: {}", e),
        }
    }

    // CUDA: migrate weights to pinned host memory for cuBLAS access.
    // Unlike OpenCL (CL_MEM_USE_HOST_PTR zero-copy wrap), CUDA requires a memcpy into
    // cuMemHostAlloc'd buffers to get device pointers for cuBLAS.
    #[cfg(feature = "cuda")]
    if args.backend == "cuda" {
        match model.migrate_weights_to_cuda(&backend) {
            Ok(n) => eprintln!(
                "[Backend] Migrated {} weight tensors to CUDA pinned memory",
                n
            ),
            Err(e) => eprintln!("[Backend] CUDA weight migration failed: {}", e),
        }
    }

    // Check if model weights are on GPU (cl_mem accessible) — needed for CPU→GPU switch
    #[cfg(feature = "opencl")]
    let weights_on_gpu =
        llm_rs2::backend::opencl::get_cl_mem(model.layers[0].wq.buffer().as_ref()).is_ok();
    #[cfg(not(feature = "opencl"))]
    let weights_on_gpu = false;

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
            use_gpu_attn: args.gpu_attn,
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
        let mut hook = llm_rs2::eval::KiviHook::new(qcf_config);
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

    let mut kv_type = match args.kv_type.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "q4" => DType::Q4_0,
        _ => anyhow::bail!(
            "Unsupported KV type: {}. Use f32, f16, or q4.",
            args.kv_type
        ),
    };

    // On discrete GPUs without flash attention for this head_dim, F16 KV + GPU attention
    // produces incorrect results. Auto-promote to F32 KV for correctness.
    // Flash attention is compiled with DK=64; models with head_dim != 64 can't use it.
    if backend.is_gpu() && kv_type == DType::F16 && head_dim != 64 && backend.is_discrete_gpu() {
        eprintln!(
            "[Config] Auto-promoting KV cache F16 → F32 (discrete GPU, head_dim={} != flash_attn DK=64)",
            head_dim
        );
        kv_type = DType::F32;
    }

    // Determine initial KV cache capacity (dynamic grow-on-demand)
    let initial_kv_capacity = if args.eval_ll || args.ppl.is_some() {
        // Eval modes: pre-allocate full capacity to avoid re-allocation
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

    // 5. Experiment schedule + Command Executor
    let experiment_schedule = if let Some(ref path) = args.experiment_schedule {
        Some(ExperimentSchedule::load(path)?)
    } else {
        None
    };

    let mut experiment_tx: Option<std::sync::mpsc::Sender<ManagerMessage>> = None;
    let heartbeat_interval = std::time::Duration::from_millis(1000);
    let mut command_executor = if let Some(ref schedule) = experiment_schedule {
        // Experiment mode: internal mpsc channel (no external transport needed)
        let (tx, rx) = std::sync::mpsc::channel();
        let (resp_tx, _resp_rx) = std::sync::mpsc::channel();
        experiment_tx = Some(tx);
        eprintln!("[Experiment] Mode enabled — schedule: {}", schedule.name);
        Some(CommandExecutor::new(
            rx,
            resp_tx,
            args.backend.clone(),
            heartbeat_interval,
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
        let executor =
            CommandExecutor::new(cmd_rx, resp_tx, args.backend.clone(), heartbeat_interval);

        // Send Capability as first message (SEQ-022).
        executor.send_capability(llm_shared::EngineCapability {
            available_devices: vec!["cpu".to_string(), "opencl".to_string()],
            active_device: args.backend.clone(),
            max_kv_tokens: args.max_seq_len,
            bytes_per_kv_token: model.config.num_key_value_heads
                * model.config.head_dim
                * 2  // K + V
                * 2, // F16 = 2 bytes
            num_layers: model.config.num_hidden_layers,
        });
        eprintln!("[Resilience] Capability sent to Manager");

        Some(executor)
    } else {
        None
    };
    let mut throttle_delay_ms: u64 = 0;
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
            args.gpu_attn,
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
            args.gpu_attn,
            &prompt,
            &args.backend,
            &args.kv_offload,
            &args.kv_type,
            args.max_prefetch_depth,
            &args.offload_path,
            &mut command_executor,
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
        let monitor = Box::new(LinuxSystemMonitor);
        let threshold_bytes = args.memory_threshold_mb * 1024 * 1024;

        if args.eviction_policy == "d2o" {
            // D2O uses CachePressureHandler (Pipeline mode), not EvictionPolicy (Legacy mode)
            let d2o_handler = D2OHandler::new(D2OConfig {
                keep_ratio: args.d2o_keep_ratio,
                protected_prefix: actual_protected_prefix,
                target_ratio: args.eviction_target_ratio,
                ema_alpha: args.d2o_ema_alpha,
                ema_beta: args.d2o_ema_beta,
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
                    "Unknown eviction policy: '{}'. Use: none, sliding, streaming, h2o, h2o_plus, d2o",
                    other
                ),
            };
            CacheManager::new(policy, monitor, threshold_bytes, args.eviction_target_ratio)
        }
    };

    // Setup event sink for score diagnostics
    cache_manager.set_event_sink(Arc::new(StderrDiagnosticSink));

    // Register policies for Manager-directed eviction dispatch.
    // Use a small protected_prefix (4 = attention sinks) for Manager-directed policies,
    // NOT actual_protected_prefix which may be the entire prompt length when
    // --eviction-policy is "none". The Manager decides WHEN and HOW MUCH to evict;
    // the policy should not silently prevent meaningful eviction.
    let resilience_protected_prefix = 4usize; // attention sinks only
    cache_manager.register_policy(
        llm_rs2::resilience::EvictMethod::H2o,
        Box::new(H2OPolicy::new(
            args.h2o_recent_window,
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
            use_gpu_attn: args.gpu_attn,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: Some(&mut collector),
            logits_last_only: false,
            variance_collector: None,
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
            use_gpu_attn: args.gpu_attn,
            qcf_mode: args.qcf_mode.clone(),
            vocab_size,
            hidden_size,
        };

        // For ratio mode, hook starts with budget=0; eval_loop updates it per-question.
        let hook_budget = if ratio_mode { 0 } else { effective_budget };
        let is_d2o = args.eviction_policy == "d2o";
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
            skip_config.as_ref(),
        )?;

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
    //  PPL MODE: Perplexity evaluation on reference text
    // ════════════════════════════════════════════════════════════
    if let Some(ref ppl_path) = args.ppl {
        return run_ppl(
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
            ppl_path,
            auto_eviction,
            score_based_eviction,
            actual_protected_prefix,
            skip_config.as_ref(),
        );
    }

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
    {
        let warmup_buf = Galloc::new().alloc(4, DType::U8)?;
        unsafe {
            *(warmup_buf.as_mut_ptr() as *mut u32) = tokens[0];
        }
        let warmup_input = Tensor::new(
            Shape::new(vec![1, 1]),
            warmup_buf,
            Arc::new(CpuBackend::new()),
        );
        let warmup_input = backend.copy_from(&warmup_input)?;

        let warmup_logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
        let mut warmup_logits = Tensor::new(
            Shape::new(vec![1, 1, vocab_size]),
            warmup_logits_buf,
            backend.clone(),
        );

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &warmup_input,
            start_pos: 0,
            kv_caches: &mut kv_caches,
            backend: &backend,
            memory: memory.as_ref(),
            logits_out: &mut warmup_logits,
            x_gen: None,
            workspace: None,
            use_gpu_attn: args.gpu_attn,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
        })?;
        backend.synchronize()?;

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

        // Determine effective chunk size.
        // 0 or >= process_len → use full prompt as single chunk (original behaviour).
        let chunk_size = if args.prefill_chunk_size > 0 && args.prefill_chunk_size < process_len {
            args.prefill_chunk_size
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

        let mut chunk_start = 0;
        while chunk_start < process_len {
            let chunk_end = (chunk_start + chunk_size).min(process_len);
            let chunk_tokens = &tokens[chunk_start..chunk_end];
            let chunk_len = chunk_tokens.len();

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
                use_gpu_attn: args.gpu_attn,
                score_accumulator: None, // No score tracking during prefill
                profiler: None,
                skip_config: None,
                importance_collector: None,
                // Chunked mode: only the last position's logits needed (saves GPU memory).
                // Non-chunked: write all positions (original behaviour).
                logits_last_only: chunked,
                variance_collector: variance_collector.as_mut(),
            })?;
            backend.synchronize()?;

            // Immediately release the GPU input buffer for this chunk.
            drop(input_tensor);

            chunk_start = chunk_end;
        }

        let prefill_forward_ms = prefill_timer.elapsed().as_secs_f64() * 1000.0;

        // Auto-eviction after prefill (sliding window only, non-experiment mode)
        if auto_eviction {
            cache_manager.maybe_evict(&mut kv_caches).ok();
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
        _last_token_time = std::time::Instant::now();

        tokens.push(next_token_id);
        start_pos += process_len;
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

        // Pre-allocate GPU input tensor for decode loop (avoids per-token GPU alloc)
        let gpu_gen_input_buf = memory.alloc(4, DType::U8)?;
        let mut gen_input_tensor =
            Tensor::new(Shape::new(vec![1, 1]), gpu_gen_input_buf, backend.clone());

        // Streaming setup
        use std::io::Write;
        let mut stdout = std::io::stdout();
        let mut _printed_len = 0;

        // Print initial tokens (prompt + first generated)
        let initial_text = tokenizer.decode(&tokens, true).unwrap_or_default();
        print!("{}", initial_text);
        _printed_len = initial_text.len();
        stdout.flush().ok();

        // Build GPU kernel plan for decode (OpenCL only, lazy rebuild on invalidation)
        // Disable for Gemma3: plan doesn't include QK-norm, post-norm, gelu_tanh_mul
        // Disable when score_accumulator is active: plan doesn't collect attention scores
        #[cfg(feature = "opencl")]
        let mut gpu_plan = if backend.name() == "OpenCL"
            && !args.profile
            && !args.no_gpu_plan
            && score_accumulator.is_none()
            && model.config.arch != llm_rs2::models::config::ModelArch::Gemma3
        {
            model.build_plan(&x_gen, &logits, &gen_ws, &mut kv_caches, &backend)
        } else {
            None
        };

        // Pre-allocate decode buffers (reused across tokens)
        let mut logits_cpu = vec![0.0f32; vocab_size];
        let mut sampling_indices: Vec<usize> = (0..vocab_size).collect();

        // Generation loop
        for (decode_token_index, _) in (0..(args.num_tokens - 1)).enumerate() {
            // Check physical cache capacity (not start_pos, which is logical RoPE position)
            if kv_caches[0].current_pos >= max_seq_len {
                println!("\n[Stopped: Max context length reached]");
                break;
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
                // Use GPU memory when on GPU; otherwise use the primary memory.
                let effective_mem: &dyn Memory = if is_gpu {
                    gpu_memory_arc.as_deref().unwrap_or_else(|| memory.as_ref())
                } else {
                    memory.as_ref()
                };
                model.forward_into(TransformerModelForwardArgs {
                    input_tokens: &gen_input_tensor,
                    start_pos,
                    kv_caches: &mut kv_caches,
                    backend: &backend,
                    memory: effective_mem,
                    logits_out: &mut logits,
                    x_gen: Some(&mut x_gen),
                    workspace: Some(&mut gen_ws),
                    use_gpu_attn: args.gpu_attn,
                    score_accumulator: score_accumulator.as_mut(),
                    profiler: profiler.as_mut().map(|p| &mut p.ops),
                    skip_config: skip_config.as_ref(),
                    importance_collector: None,
                    logits_last_only: false,
                    variance_collector: None,
                })?;

                // Rebuild plan if it was invalidated (e.g. KV cache resize)
                #[cfg(feature = "opencl")]
                if gpu_plan.is_none()
                    && backend.name() == "OpenCL"
                    && !args.profile
                    && !args.no_gpu_plan
                    && score_accumulator.is_none()
                    && model.config.arch != llm_rs2::models::config::ModelArch::Gemma3
                {
                    gpu_plan = model.build_plan(&x_gen, &logits, &gen_ws, &mut kv_caches, &backend);
                }
            }
            backend.synchronize()?;
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
                    // Derive streaming window: same logic as policy construction
                    let streaming_window_size = if args.streaming_window > 0 {
                        args.streaming_window
                    } else if args.kv_budget > 0 {
                        args.kv_budget.saturating_sub(args.sink_size)
                    } else {
                        args.eviction_window
                    };
                    let ctx = QcfEstimateContext {
                        kv_caches: &kv_caches,
                        score_accumulator: score_accumulator.as_ref(),
                        streaming_config: Some((args.sink_size, streaming_window_size)),
                        importance_table: None, // Not collected in standard decode path
                        num_layers: model.config.num_hidden_layers,
                        kivi_caches: None,
                    };
                    let estimates = compute_qcf_estimates(&ctx);
                    executor.send_qcf_estimate(llm_shared::QcfEstimate { estimates });
                }

                if let Some(evict) = &plan.evict {
                    let effective_ratio =
                        args.experiment_eviction_ratio.unwrap_or(evict.target_ratio);

                    // GPU score sync before resilience eviction
                    #[cfg(feature = "opencl")]
                    if let Some(ocl_be) = backend
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
                                let diag_path =
                                    format!("{}.scores.csv", out_path.trim_end_matches(".jsonl"));
                                if events::dump_scores_csv(scores, cache_pos, &diag_path).is_ok() {
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
                    let result = if evict.method == llm_rs2::resilience::EvictMethod::D2o {
                        let importance = if let Some(acc) = score_accumulator.as_ref() {
                            acc.importance_scores().to_vec()
                        } else {
                            vec![]
                        };
                        if importance.is_empty() {
                            cache_manager.force_evict(&mut kv_caches, effective_ratio)
                        } else {
                            cache_manager.force_evict_with_scores(
                                &mut kv_caches,
                                effective_ratio,
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
                                effective_ratio,
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
                            effective_ratio,
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
                                        let max_s =
                                            valid.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
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
                }

                // Dynamic layer skip / restore_defaults handling
                if plan.restore_defaults {
                    eprintln!("[Resilience] RestoreDefaults");
                    skip_config = None;
                    last_skip_ratio = None;
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
                                let lb = cpu_memory_arc.alloc(vocab_size * 4, DType::F32)?;
                                logits = Tensor::new(
                                    Shape::new(vec![1, 1, vocab_size]),
                                    lb,
                                    backend.clone(),
                                );
                                let xb = cpu_memory_arc.alloc(hidden_size * 4, DType::F32)?;
                                x_gen = Tensor::new(
                                    Shape::new(vec![1, 1, hidden_size]),
                                    xb,
                                    backend.clone(),
                                );
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
                                // Re-allocate gen_input_tensor on CPU backend
                                let gi_buf = cpu_memory_arc.alloc(4, DType::U8)?;
                                gen_input_tensor =
                                    Tensor::new(Shape::new(vec![1, 1]), gi_buf, backend.clone());
                                #[cfg(feature = "opencl")]
                                {
                                    gpu_plan = None;
                                }
                                is_gpu = false;
                                eprintln!("[Switch] Resilience: Switched to CPU.");
                            }
                            "gpu" | "opencl" if !is_gpu && weights_on_gpu => {
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
                                let lb = gpu_mem.alloc(vocab_size * 4, DType::F32)?;
                                logits = Tensor::new(
                                    Shape::new(vec![1, 1, vocab_size]),
                                    lb,
                                    backend.clone(),
                                );
                                let xb = gpu_mem.alloc(hidden_size * 4, DType::F32)?;
                                x_gen = Tensor::new(
                                    Shape::new(vec![1, 1, hidden_size]),
                                    xb,
                                    backend.clone(),
                                );
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
                                    gpu_plan = None;
                                }
                                // Re-allocate gen_input_tensor on new GPU backend
                                let gi_buf = gpu_mem.alloc(4, DType::U8)?;
                                gen_input_tensor =
                                    Tensor::new(Shape::new(vec![1, 1]), gi_buf, backend.clone());
                                is_gpu = true;
                                eprintln!("[Switch] Resilience: Switched to GPU.");
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

                if plan.throttle_delay_ms != throttle_delay_ms {
                    eprintln!(
                        "[Resilience] Throttle: {}ms → {}ms",
                        throttle_delay_ms, plan.throttle_delay_ms
                    );
                }
                throttle_delay_ms = plan.throttle_delay_ms;

                // Update target TBT from Manager directive (overrides CLI --target-tbt)
                if plan.target_tbt_ms > 0 && plan.target_tbt_ms as f64 != target_tbt_ms {
                    eprintln!(
                        "[Resilience] SetTargetTbt: {:.1}ms → {}ms",
                        target_tbt_ms, plan.target_tbt_ms
                    );
                    target_tbt_ms = plan.target_tbt_ms as f64;
                } else if plan.target_tbt_ms > 0 {
                    target_tbt_ms = plan.target_tbt_ms as f64;
                } else if plan.restore_defaults {
                    target_tbt_ms = args.target_tbt; // restore CLI default
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
            }
            // ── End Resilience checkpoint ─────────────────────

            // Read logits to CPU (reuses pre-allocated buffer)
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

    // 7. Output results
    println!("\nDone.");
    println!("[Profile] Event: End");
    println!("TTFT: {:.2} ms", _ttft_ms);
    if !forward_ms_values.is_empty() {
        let avg_forward: f64 =
            forward_ms_values.iter().sum::<f64>() / forward_ms_values.len() as f64;
        println!(
            "Decode: {:.2} ms/tok ({:.1} tok/s) [{} tokens, forward only]",
            avg_forward,
            1000.0 / avg_forward,
            forward_ms_values.len(),
        );
    }
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
    if !ctx.kv_caches.is_empty() && ctx.kv_caches[0].current_pos > 0 {
        let cache = &ctx.kv_caches[0];
        let current_pos = cache.current_pos;
        let capacity = cache.capacity();
        let layout = cache.layout();
        let n_kv_heads = cache.kv_heads();
        let head_dim = cache.head_dim();

        let keep_ratio = 0.5f32;
        let target_len = (current_pos as f32 * keep_ratio) as usize;
        let protected_prefix = 4usize;

        // Get attention scores and V data for unified QCF
        let scores_opt = ctx
            .score_accumulator
            .filter(|a| a.is_active())
            .map(|a| a.importance_scores());

        let head_attn_opt = ctx
            .score_accumulator
            .filter(|a| a.is_active())
            .and_then(|a| a.last_step_head_attn());

        // Access V buffer as F32 (standard KVCache is always F32 or F16)
        let v_dtype = cache.v_buffer.dtype();
        let v_source = match v_dtype {
            DType::F16 => VDataSource::F16(cache.v_buffer.as_slice::<u16>()),
            _ => VDataSource::F32(cache.v_buffer.as_slice::<f32>()),
        };

        // Fallback flat scores if no accumulator
        let fallback_scores: Vec<f32>;
        let attention_scores: &[f32] = if let Some(scores) = scores_opt {
            scores
        } else {
            // Uniform fallback
            fallback_scores = vec![1.0 / current_pos.max(1) as f32; current_pos];
            &fallback_scores
        };

        let aggregation = AggregationMode::Mean;

        if target_len < current_pos {
            // ── 1. Sliding window QCF ──
            {
                let params = UnifiedQcfParams {
                    action: QcfActionType::EvictSliding { target_len },
                    v_source: VDataSource::F32(match &v_source {
                        VDataSource::F32(d) => d,
                        VDataSource::F16(_) => &[],
                    }),
                    attention_scores,
                    head_attn: head_attn_opt,
                    n_kv_heads,
                    head_dim,
                    current_pos,
                    capacity,
                    layout,
                    aggregation: aggregation.clone(),
                };
                // Re-wrap v_source properly
                let params = UnifiedQcfParams {
                    v_source: match v_dtype {
                        DType::F16 => VDataSource::F16(cache.v_buffer.as_slice::<u16>()),
                        _ => VDataSource::F32(cache.v_buffer.as_slice::<f32>()),
                    },
                    ..params
                };
                let (qcf, _) = compute_unified_qcf(&params);
                estimates.insert("kv_evict_sliding".to_string(), qcf);
            }

            // ── 2. H2O eviction QCF (needs scores) ──
            if scores_opt.is_some() {
                let params = UnifiedQcfParams {
                    action: QcfActionType::EvictH2o {
                        target_len,
                        keep_ratio: 0.5,
                        protected_prefix,
                    },
                    v_source: match v_dtype {
                        DType::F16 => VDataSource::F16(cache.v_buffer.as_slice::<u16>()),
                        _ => VDataSource::F32(cache.v_buffer.as_slice::<f32>()),
                    },
                    attention_scores,
                    head_attn: head_attn_opt,
                    n_kv_heads,
                    head_dim,
                    current_pos,
                    capacity,
                    layout,
                    aggregation: aggregation.clone(),
                };
                let (qcf, _) = compute_unified_qcf(&params);
                estimates.insert("kv_evict_h2o".to_string(), qcf);
            }

            // ── 3. Streaming QCF ──
            if let Some((sink_size, window_size)) = ctx.streaming_config {
                let params = UnifiedQcfParams {
                    action: QcfActionType::EvictStreaming {
                        sink_size,
                        window_size,
                    },
                    v_source: match v_dtype {
                        DType::F16 => VDataSource::F16(cache.v_buffer.as_slice::<u16>()),
                        _ => VDataSource::F32(cache.v_buffer.as_slice::<f32>()),
                    },
                    attention_scores,
                    head_attn: head_attn_opt,
                    n_kv_heads,
                    head_dim,
                    current_pos,
                    capacity,
                    layout,
                    aggregation: aggregation.clone(),
                };
                let (qcf, _) = compute_unified_qcf(&params);
                estimates.insert("kv_evict_streaming".to_string(), qcf);
            }

            // ── 4. D2O merge QCF (needs scores) ──
            if scores_opt.is_some() {
                let params = UnifiedQcfParams {
                    action: QcfActionType::MergeD2o {
                        target_len,
                        keep_ratio: 0.5,
                        protected_prefix,
                    },
                    v_source: match v_dtype {
                        DType::F16 => VDataSource::F16(cache.v_buffer.as_slice::<u16>()),
                        _ => VDataSource::F32(cache.v_buffer.as_slice::<f32>()),
                    },
                    attention_scores,
                    head_attn: head_attn_opt,
                    n_kv_heads,
                    head_dim,
                    current_pos,
                    capacity,
                    layout,
                    aggregation: aggregation.clone(),
                };
                let (qcf, _) = compute_unified_qcf(&params);
                estimates.insert("kv_merge_d2o".to_string(), qcf);
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
        EngineCommand::RestoreDefaults => "RestoreDefaults".to_string(),
        EngineCommand::SwitchHw { device } => format!("SwitchHw({})", device),
        EngineCommand::PrepareComputeUnit { device } => format!("Prepare({})", device),
        EngineCommand::Suspend => "Suspend".to_string(),
        EngineCommand::Resume => "Resume".to_string(),
        EngineCommand::RequestQcf => "RequestQcf".to_string(),
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
    names
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
            use_gpu_attn: args.gpu_attn,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
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
            use_gpu_attn: args.gpu_attn,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
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
    gpu_attn: bool,
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
            use_gpu_attn: gpu_attn,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
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

    // Build GPU kernel plan for KIVI decode (OpenCL only)
    #[cfg(feature = "opencl")]
    let mut gpu_plan = if backend.name() == "OpenCL" && !no_gpu_plan {
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
                use_gpu_attn: gpu_attn,
                score_accumulator: None,
                profiler: None,
                skip_config: kivi_skip_config.as_ref(),
                importance_collector: None,
                logits_last_only: false,
                variance_collector: None,
            })?;

            // Rebuild plan after fallback
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
                executor.send_qcf_estimate(llm_shared::QcfEstimate { estimates });
            }

            // kv_quant_bits: transition KiviCache bit-width
            if let Some(bits) = plan.kv_quant_bits {
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
            }

            // layer_skip / restore_defaults
            if plan.restore_defaults {
                eprintln!("[KIVI-Resilience] RestoreDefaults");
                kivi_skip_config = None;
                kivi_last_skip_ratio = None;
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

            // Update target TBT from Manager directive
            if plan.target_tbt_ms > 0 && plan.target_tbt_ms as f64 != target_tbt_ms {
                eprintln!(
                    "[KIVI-Resilience] SetTargetTbt: {:.1}ms → {}ms",
                    target_tbt_ms, plan.target_tbt_ms
                );
                target_tbt_ms = plan.target_tbt_ms as f64;
            } else if plan.target_tbt_ms > 0 {
                target_tbt_ms = plan.target_tbt_ms as f64;
            } else if plan.restore_defaults {
                target_tbt_ms = 0.0;
            }

            if plan.suspended {
                eprintln!("\n[KIVI-Resilience] Inference suspended by system signal");
                break;
            }

            executor.on_token_generated();
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
    gpu_attn: bool,
    _prompt: &str,
    _backend_name: &str,
    offload_mode: &str,
    kv_type_str: &str,
    max_prefetch_depth: usize,
    offload_path: &str,
    command_executor: &mut Option<CommandExecutor>,
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

    // Create OffloadKVCache per layer
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
        model.forward_into(TransformerModelForwardArgs {
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
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
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
                use_gpu_attn: gpu_attn,
                score_accumulator: None,
                profiler: None,
                skip_config: None,
                importance_collector: None,
                logits_last_only: false,
                variance_collector: None,
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
) -> anyhow::Result<()> {
    use llm_rs2::core::qcf::QcfConfig;

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
    let prefill_chunk = if has_budget {
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
    let qcf_config = QcfConfig {
        mode: match args.qcf_mode.as_str() {
            "caote" => llm_rs2::core::qcf::QcfMode::Caote,
            "both" => llm_rs2::core::qcf::QcfMode::Both,
            _ => llm_rs2::core::qcf::QcfMode::Attn,
        },
        ..QcfConfig::default()
    };
    let overall_start = std::time::Instant::now();

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
            use_gpu_attn: args.gpu_attn,
            score_accumulator: score_accumulator.as_mut(),
            profiler: None,
            skip_config,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
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

    for i in prefill_len..eval_tokens - 1 {
        let input_token = token_ids[i];
        let target_token = token_ids[i + 1];

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
            use_gpu_attn: args.gpu_attn,
            score_accumulator: score_accumulator.as_mut(),
            profiler: None,
            skip_config,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
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

                    let (qcf_attn_raw, qcf_attn_norm, qcf_caote_value) =
                        if let Some(acc) = score_accumulator.as_ref() {
                            if let Some(head_attn) = acc.last_step_head_attn() {
                                let positions = if score_based_eviction {
                                    let scores = acc.importance_scores();
                                    let target_len = ((before_len as f32) * ratio) as usize;
                                    let evicted = llm_rs2::core::qcf::identify_evicted_h2o(
                                        scores,
                                        protected_prefix,
                                        args.h2o_keep_ratio,
                                        before_len,
                                        target_len,
                                    );
                                    evicted.iter().map(|(pos, _)| *pos).collect::<Vec<_>>()
                                } else {
                                    llm_rs2::core::qcf::identify_evicted_sliding(
                                        protected_prefix,
                                        result.tokens_removed,
                                        before_len,
                                    )
                                };

                                // QCF-ATTN v2 (closed-form)
                                let n_kv_heads = kv_caches[0].kv_heads().max(1);
                                let max_seq_len = head_attn.len() / n_kv_heads;
                                let attn_metric = llm_rs2::core::qcf::compute_qcf_attn_v2(
                                    head_attn,
                                    &positions,
                                    n_kv_heads,
                                    max_seq_len,
                                    eviction_ratio,
                                );

                                // QCF-CAOTE
                                let caote = if can_compute_qcf && !positions.is_empty() {
                                    let metric = llm_rs2::core::qcf::compute_eviction_qcf_caote(
                                        &positions,
                                        head_attn,
                                        &kv_caches[0],
                                        &qcf_config,
                                        v_cpu_data.as_deref(),
                                    );
                                    metric.raw_value as f64
                                } else {
                                    0.0
                                };

                                (
                                    attn_metric.raw_value as f64,
                                    attn_metric.normalized_value as f64,
                                    caote,
                                )
                            } else {
                                (0.0, 0.0, 0.0)
                            }
                        } else {
                            (0.0, 0.0, 0.0)
                        };

                    qcf_events.push(serde_json::json!({
                        "step": i,
                        "tokens_evicted": result.tokens_removed,
                        "eviction_ratio": eviction_ratio,
                        "qcf_attn_raw": qcf_attn_raw,
                        "qcf_attn_norm": qcf_attn_norm,
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
    let tok_per_sec = nll_count as f64 / wall_time;

    // Compute summary stats from all eviction events (v3)
    let n_evictions = qcf_events.len();
    let qcf_sum_attn_norm: f64 = qcf_events
        .iter()
        .filter_map(|e| e["qcf_attn_norm"].as_f64())
        .sum();
    let qcf_sum_caote: f64 = qcf_events
        .iter()
        .filter_map(|e| e["qcf_caote"].as_f64())
        .sum();
    let qcf_max_attn_norm: f64 = qcf_events
        .iter()
        .filter_map(|e| e["qcf_attn_norm"].as_f64())
        .fold(0.0f64, f64::max);
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
        "qcf_sum_attn_norm": qcf_sum_attn_norm,
        "qcf_sum_caote": qcf_sum_caote,
        "qcf_max_attn_norm": qcf_max_attn_norm,
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

    Ok(())
}
