use clap::Parser;
use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::core::attention_scores::AttentionScoreAccumulator;
use llm_rs2::core::backend::Backend;
use llm_rs2::core::buffer::DType;
use llm_rs2::core::cache_manager::CacheManager;
use llm_rs2::core::eviction::h2o::H2OPolicy;
use llm_rs2::core::eviction::no_eviction::NoEvictionPolicy;
use llm_rs2::core::eviction::sliding_window::SlidingWindowPolicy;
use llm_rs2::core::kv_cache::KVCache;
use llm_rs2::core::memory::Memory;
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

    /// Eviction policy for KV cache management (none, sliding, h2o)
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

    /// Number of prefix tokens to protect from eviction (defaults to the entire prompt length)
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
}

fn sample(logits: &mut [f32], tokens: &[u32], vocab_size: usize, args: &Args) -> u32 {
    // 1. Repetition Penalty
    let start_idx = tokens.len().saturating_sub(args.repetition_window);
    for &token_id in &tokens[start_idx..] {
        let token_id = token_id as usize;
        if token_id < vocab_size {
            let logit = &mut logits[token_id];
            if *logit < 0.0 {
                *logit *= args.repetition_penalty;
            } else {
                *logit /= args.repetition_penalty;
            }
        }
    }

    // 2. Temperature
    let temp = args.temperature;
    if temp == 0.0 {
        // Greedy
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx as u32)
            .unwrap();
    }

    for l in logits.iter_mut() {
        *l /= temp;
    }

    // Softmax
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut exp_sum = 0.0;
    for l in logits.iter_mut() {
        *l = (*l - max_logit).exp();
        exp_sum += *l;
    }
    for l in logits.iter_mut() {
        *l /= exp_sum;
    }

    // 3. Top-K
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].total_cmp(&logits[a])); // Descending

    let top_k = args.top_k.min(vocab_size);
    let mut valid_indices = indices;
    if top_k > 0 {
        valid_indices.truncate(top_k);
    }

    // 4. Top-P
    let mut cumulative_prob = 0.0;
    let mut cutoff_index = valid_indices.len();

    for (i, &idx) in valid_indices.iter().enumerate() {
        cumulative_prob += logits[idx];
        if cumulative_prob > args.top_p {
            cutoff_index = i + 1;
            break;
        }
    }
    valid_indices.truncate(cutoff_index);

    // 5. Sample
    let mut rng = rand::rng();
    let r: f32 = rand::Rng::random(&mut rng); // [0, 1)

    // Normalize probabilities of valid indices
    let mut prob_sum = 0.0;
    for &idx in &valid_indices {
        prob_sum += logits[idx];
    }

    let mut thread_r = r * prob_sum;
    for &idx in &valid_indices {
        thread_r -= logits[idx];
        if thread_r <= 0.0 {
            return idx as u32;
        }
    }

    valid_indices.first().copied().unwrap_or(0) as u32
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
    println!(
        "KV cache type: {:?} (initial capacity: {} tokens, {}B per layer, max: {})",
        kv_type, initial_kv_capacity, kv_buf_size, max_seq_len
    );

    let mut kv_caches = Vec::new();
    for _ in 0..num_layers {
        let k_buf = memory.alloc(kv_buf_size, kv_type)?;
        let v_buf = memory.alloc(kv_buf_size, kv_type)?;

        let k = Tensor::new(
            Shape::new(vec![1, initial_kv_capacity, kv_heads, head_dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, initial_kv_capacity, kv_heads, head_dim]),
            v_buf,
            backend.clone(),
        );

        kv_caches.push(KVCache::new_dynamic(
            k,
            v,
            initial_kv_capacity,
            max_seq_len,
            kv_heads,
            head_dim,
            memory.clone(),
        ));
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
    let actual_protected_prefix = args.protected_prefix.unwrap_or(input_ids.len());

    let cache_manager = {
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
                other => anyhow::bail!(
                    "Unknown eviction policy: '{}'. Use: none, sliding, h2o",
                    other
                ),
            };
        let monitor = Box::new(LinuxSystemMonitor);
        let threshold_bytes = args.memory_threshold_mb * 1024 * 1024;
        CacheManager::new(policy, monitor, threshold_bytes, args.eviction_target_ratio)
    };

    // Setup AttentionScoreAccumulator for H2O
    let mut score_accumulator = if args.eviction_policy == "h2o" {
        let mut acc = AttentionScoreAccumulator::new(
            max_seq_len,
            model.config.num_attention_heads,
            model.config.num_hidden_layers,
            args.h2o_tracked_layers,
            args.h2o_decay,
        );
        acc.set_active(true);
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

    // Determine whether to pass cache_manager to forward_into() for auto-eviction.
    // H2O is signal-driven: eviction is triggered exclusively by resilience signals,
    // not by automatic cache/memory checks. Score accumulation still happens every
    // token via score_accumulator (passed separately).
    let cm_ref = match args.eviction_policy.as_str() {
        "sliding" => Some(&cache_manager),
        _ => None, // "none" and "h2o" — no auto-eviction in forward path
    };

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

        model
            .forward_into(LlamaModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos,
                kv_caches: &mut kv_caches,
                backend: &backend,
                memory: memory.as_ref(),
                logits_out: &mut prefill_logits,
                x_gen: None,
                workspace: None,
                use_gpu_attn: args.gpu_attn,
                cache_manager: cm_ref,
                score_accumulator: None, // No score tracking during prefill
            })?
            .ok_or(())
            .ok(); // Eviction during prefill is unlikely, ignore result

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

        let next_token_id = sample(&mut last_logits, &tokens, vocab_size, &args);

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
            let _eviction_result = model.forward_into(LlamaModelForwardArgs {
                input_tokens: &gen_input_tensor,
                start_pos,
                kv_caches: &mut kv_caches,
                backend: &backend,
                memory: memory.as_ref(),
                logits_out: &mut logits,
                x_gen: Some(&mut x_gen),
                workspace: Some(&mut gen_ws),
                use_gpu_attn: args.gpu_attn,
                cache_manager: cm_ref,
                score_accumulator: score_accumulator.as_mut(),
            })?;
            let forward_ms = forward_start.elapsed().as_secs_f64() * 1000.0;
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
                        let result = if let Some(ref acc) = score_accumulator {
                            cache_manager.force_evict_with_scores(
                                &mut kv_caches,
                                *target_ratio,
                                acc.importance_scores(),
                            )
                        } else {
                            cache_manager.force_evict(&mut kv_caches, *target_ratio)
                        };
                        match result {
                            Ok(r) if r.evicted => {
                                eprintln!(
                                    "[Resilience] Evicted {} tokens (pos: {} → {})",
                                    r.tokens_removed,
                                    r.new_pos + r.tokens_removed,
                                    r.new_pos
                                );
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

            let next_token_id = sample(&mut logits_cpu, &tokens, vocab_size, &args);

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
