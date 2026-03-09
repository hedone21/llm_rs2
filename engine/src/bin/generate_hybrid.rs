use clap::Parser;
use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::backend::opencl::OpenCLBackend;
use llm_rs2::core::attention_scores::AttentionScoreAccumulator;
use llm_rs2::core::backend::Backend;
use llm_rs2::core::buffer::DType;
use llm_rs2::core::cache_manager::CacheManager;
use llm_rs2::core::eviction::h2o::H2OPolicy;
use llm_rs2::core::eviction::no_eviction::NoEvictionPolicy;
use llm_rs2::core::eviction::sliding_window::SlidingWindowPolicy;
use llm_rs2::core::kv_cache::{KVCache, KVLayout};
use llm_rs2::core::memory::Memory;
use llm_rs2::core::shape::Shape;
use llm_rs2::core::sys_monitor::LinuxSystemMonitor;
use llm_rs2::core::tensor::Tensor;
use llm_rs2::layers::workspace::{LayerWorkspace, WorkspaceConfig};
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::models::llama::llama_model::{LlamaModel, LlamaModelForwardArgs};
use llm_rs2::resilience::signal::RecommendedBackend;
use llm_rs2::resilience::{
    InferenceContext, ResilienceAction, ResilienceManager, SignalListener, execute_action,
};
use std::sync::Arc;
use tokenizers::Tokenizer;

#[cfg(feature = "resilience")]
use llm_rs2::resilience::DbusTransport;
#[cfg(unix)]
use llm_rs2::resilience::UnixSocketTransport;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "models/llama3.2-1b")]
    model_path: String,

    #[arg(short, long, default_value = "Hello, world! I am a")]
    prompt: String,

    #[arg(short, long, default_value_t = 20)]
    num_tokens: usize,

    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,

    /// Threshold to switch from CPU to GPU (tokens)
    #[arg(long, default_value_t = 512)]
    switch_threshold: usize,

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

    /// Number of prefix tokens to protect from eviction (defaults to prompt length)
    #[arg(long)]
    protected_prefix: Option<usize>,

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
}

fn sample(logits: &mut [f32], tokens: &[u32], vocab_size: usize, args: &Args) -> u32 {
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
    let temp = args.temperature;
    if temp == 0.0 {
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
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut exp_sum = 0.0;
    for l in logits.iter_mut() {
        *l = (*l - max_logit).exp();
        exp_sum += *l;
    }
    for l in logits.iter_mut() {
        *l /= exp_sum;
    }

    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top_k = args.top_k.min(vocab_size);
    let mut valid_indices = indices;
    if top_k > 0 {
        valid_indices.truncate(top_k);
    }
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
    let mut rng = rand::rng();
    let r: f32 = rand::Rng::random(&mut rng);
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

/// Migrate KV caches between backends (CPU↔GPU).
///
/// Reads KV data from `src_backend`, creates intermediate CPU tensors,
/// then optionally copies to GPU via `dst_backend.copy_from()`.
#[allow(clippy::too_many_arguments)]
fn migrate_kv_caches(
    kv_caches: &mut [KVCache],
    src_backend: &Arc<dyn Backend>,
    dst_backend: &Arc<dyn Backend>,
    cpu_backend: &Arc<dyn Backend>,
    cpu_memory: &Arc<dyn Memory>,
    dst_memory: &Arc<dyn Memory>,
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    copy_to_dst: bool,
) -> anyhow::Result<()> {
    for kv in kv_caches.iter_mut() {
        let current_capacity = kv.capacity();
        let saved_pos = kv.current_pos;
        let k_size = current_capacity * kv_heads * head_dim * 4;

        let mut k_data = vec![0u8; k_size];
        let mut v_data = vec![0u8; k_size];
        src_backend.read_buffer(&kv.k_buffer, &mut k_data)?;
        src_backend.read_buffer(&kv.v_buffer, &mut v_data)?;

        let k_cpu_buf = cpu_memory.alloc(k_size, DType::F32)?;
        unsafe {
            std::ptr::copy_nonoverlapping(k_data.as_ptr(), k_cpu_buf.as_mut_ptr(), k_size);
        }
        let k_cpu_tensor = Tensor::new(kv.k_buffer.shape().clone(), k_cpu_buf, cpu_backend.clone());

        let v_cpu_buf = cpu_memory.alloc(k_size, DType::F32)?;
        unsafe {
            std::ptr::copy_nonoverlapping(v_data.as_ptr(), v_cpu_buf.as_mut_ptr(), k_size);
        }
        let v_cpu_tensor = Tensor::new(kv.v_buffer.shape().clone(), v_cpu_buf, cpu_backend.clone());

        let (k_final, v_final) = if copy_to_dst {
            (
                dst_backend.copy_from(&k_cpu_tensor)?,
                dst_backend.copy_from(&v_cpu_tensor)?,
            )
        } else {
            (k_cpu_tensor, v_cpu_tensor)
        };

        let saved_layout = kv.layout();
        let mut new_kv = KVCache::new_dynamic(
            k_final,
            v_final,
            current_capacity,
            max_seq_len,
            kv_heads,
            head_dim,
            dst_memory.clone(),
        )
        .with_layout(saved_layout);
        new_kv.current_pos = saved_pos;
        *kv = new_kv;
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();

    let mut args = Args::parse();
    let model_path = &args.model_path;

    println!("[Hybrid] Starting hybrid generation...");
    println!(
        "[Hybrid] Model: {}, Threshold: {}",
        model_path, args.switch_threshold
    );

    // 1. Initialize Backends
    let cpu_backend_concrete = Arc::new(CpuBackend::new());
    let cpu_backend: Arc<dyn Backend> = cpu_backend_concrete.clone();

    let gpu_backend_concrete = Arc::new(OpenCLBackend::new()?);
    let gpu_backend: Arc<dyn Backend> = gpu_backend_concrete.clone();

    // Start with CPU
    let mut current_backend = cpu_backend.clone();
    let cpu_memory: Arc<dyn Memory> = Arc::new(Galloc::new());
    // Create GPU memory context but don't use it yet
    let gpu_memory: Arc<dyn Memory> =
        Arc::new(llm_rs2::backend::opencl::memory::OpenCLMemory::new(
            gpu_backend_concrete.context.clone(),
            gpu_backend_concrete.queue.clone(),
            false, // No zero-copy for now
        ));

    // Load Model on CPU initially
    println!("[Hybrid] Loading model on CPU...");
    let model = LlamaModel::load(model_path, current_backend.clone(), &*cpu_memory)?;

    // Tokenizer
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
        .map_err(|e| anyhow::anyhow!(e))?;
    let encoding = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let tokens: Vec<u32> = encoding.get_ids().to_vec();

    // Prepare KV Cache (CPU)
    // We need to keep track of the tensor shapes to re-allocate them on GPU later
    let max_seq_len = args.max_seq_len;
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;
    let num_layers = model.config.num_hidden_layers;

    let initial_kv_capacity = tokens.len().next_power_of_two().max(128).min(max_seq_len);

    let mut kv_caches = Vec::new();
    for _ in 0..num_layers {
        let kv_buf_size = initial_kv_capacity * kv_heads * head_dim * 4;
        let k_buf = cpu_memory.alloc(kv_buf_size, DType::F32)?;
        let v_buf = cpu_memory.alloc(kv_buf_size, DType::F32)?;
        let k = Tensor::new(
            Shape::new(vec![1, kv_heads, initial_kv_capacity, head_dim]),
            k_buf,
            cpu_backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, kv_heads, initial_kv_capacity, head_dim]),
            v_buf,
            cpu_backend.clone(),
        );
        kv_caches.push(
            KVCache::new_dynamic(
                k,
                v,
                initial_kv_capacity,
                max_seq_len,
                kv_heads,
                head_dim,
                cpu_memory.clone(),
            )
            .with_layout(KVLayout::HeadMajor),
        );
    }

    // Setup CacheManager
    let actual_protected_prefix = args.protected_prefix.unwrap_or(tokens.len());

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
            "[Hybrid] Eviction: policy={}, window={}, prefix={}, ratio={}, threshold={}MB",
            args.eviction_policy,
            args.eviction_window,
            actual_protected_prefix,
            args.eviction_target_ratio,
            args.memory_threshold_mb
        );
    }

    // H2O is signal-driven: eviction is triggered exclusively by resilience signals,
    // not by automatic cache/memory checks. Score accumulation still happens via
    // score_accumulator (passed separately to forward_into).
    let cm_ref = match args.eviction_policy.as_str() {
        "sliding" => Some(&cache_manager),
        _ => None, // "none" and "h2o" — no auto-eviction in forward path
    };

    // Resilience Manager (optional)
    let mut resilience_manager = if args.enable_resilience {
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
                eprintln!("[Hybrid/Resilience] Unknown transport: {}", other);
                return Ok(());
            }
        };
        eprintln!(
            "[Hybrid/Resilience] Manager enabled — transport: {}",
            args.resilience_transport
        );
        Some(ResilienceManager::new(rx))
    } else {
        None
    };
    let mut throttle_delay_ms: u64 = 0;

    let mut tokens_generated = Vec::new();
    tokens_generated.extend_from_slice(&tokens);
    let mut start_pos = 0;

    // Logits buffer (re-allocated if backend changes)
    let vocab_size = model.config.vocab_size;
    let logits_buf = cpu_memory.alloc(vocab_size * 4, DType::F32)?;
    let mut logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        logits_buf,
        current_backend.clone(),
    );

    let eos_id = tokenizer
        .get_vocab(true)
        .get("</s>")
        .copied()
        .unwrap_or(u32::MAX);

    // === PREFILL ===
    {
        println!("[Hybrid] Prefilling on CPU...");
        let process_len = tokens.len();
        let cpu_indices_buf = cpu_memory.alloc(process_len * 4, DType::U8)?;
        unsafe {
            let ptr = cpu_indices_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, process_len);
        }
        let input_tensor = Tensor::new(
            Shape::new(vec![1, process_len]),
            cpu_indices_buf,
            cpu_backend.clone(),
        );

        // Prefill logits
        let prefill_logits_buf = cpu_memory.alloc(process_len * vocab_size * 4, DType::F32)?;
        let mut prefill_logits = Tensor::new(
            Shape::new(vec![1, process_len, vocab_size]),
            prefill_logits_buf,
            cpu_backend.clone(),
        );

        model
            .forward_into(LlamaModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos,
                kv_caches: &mut kv_caches,
                backend: &current_backend,
                memory: cpu_memory.as_ref(),
                logits_out: &mut prefill_logits,
                x_gen: None,
                workspace: None,
                use_gpu_attn: false,
                cache_manager: cm_ref,
                score_accumulator: None,
            })?
            .ok_or(())
            .ok();

        // Sample last token
        let mut logits_cpu = vec![0.0f32; process_len * vocab_size];
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
            current_backend.read_buffer(&prefill_logits, slice)?;
        }
        let start_idx = (process_len - 1) * vocab_size;
        let mut last_logits = logits_cpu[start_idx..start_idx + vocab_size].to_vec();
        let next_token_id = sample(&mut last_logits, &tokens_generated, vocab_size, &args);

        tokens_generated.push(next_token_id);
        start_pos += process_len;
    }

    // === GENERATION ===
    println!("[Hybrid] decoding...");
    let hidden_size = model.config.hidden_size;
    let mut _x_gen = None; // Optimization tensors
    let mut _gen_ws = None;

    // Helper to setup workspace for current backend
    let setup_workspace = |backend: &Arc<dyn Backend>,
                           memory: &dyn Memory|
     -> anyhow::Result<(Tensor, LayerWorkspace)> {
        let x_gen_buf = memory.alloc(hidden_size * 4, DType::F32)?;
        let x_gen_t = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            x_gen_buf,
            backend.clone(),
        );
        let ws = LayerWorkspace::new(
            WorkspaceConfig {
                batch_size: 1,
                dim: hidden_size,
                q_dim: hidden_size,
                k_dim: hidden_size / 4,
                v_dim: hidden_size / 4,
                ffn_hidden: model.config.intermediate_size,
                n_heads: model.config.num_attention_heads,
                max_seq_len,
            },
            memory,
            backend.clone(),
        )?;
        Ok((x_gen_t, ws))
    };

    // Setup initial CPU workspace
    let (cpu_x_gen, cpu_ws) = setup_workspace(&cpu_backend, cpu_memory.as_ref())?;
    _x_gen = Some(cpu_x_gen);
    _gen_ws = Some(cpu_ws);

    // Initial print
    print!(
        "{}",
        tokenizer
            .decode(&tokens_generated, true)
            .unwrap_or_default()
    );
    use std::io::Write;
    std::io::stdout().flush().ok();

    let mut is_gpu = false;

    for _ in 0..(args.num_tokens - 1) {
        // Check physical cache capacity
        if kv_caches[0].current_pos >= max_seq_len {
            println!("\n[Hybrid] Stopped: Max context length reached");
            break;
        }

        // CHECK THRESHOLD AND SWITCH
        if !is_gpu && start_pos >= args.switch_threshold {
            println!("\n\n[Hybrid] Switching to GPU at token {}...", start_pos);

            // 1. Migrate KV Cache CPU→GPU
            migrate_kv_caches(
                &mut kv_caches,
                &cpu_backend,
                &gpu_backend,
                &cpu_backend,
                &cpu_memory,
                &gpu_memory,
                kv_heads,
                head_dim,
                max_seq_len,
                true,
            )?;

            // 2. Switch Backend
            current_backend = gpu_backend.clone();

            // 3. Re-allocate Logits & Workspace on GPU
            let logits_gpu_buf = gpu_memory.alloc(vocab_size * 4, DType::F32)?;
            logits = Tensor::new(
                Shape::new(vec![1, 1, vocab_size]),
                logits_gpu_buf,
                gpu_backend.clone(),
            );

            let (gpu_x_gen, gpu_ws) = setup_workspace(&gpu_backend, gpu_memory.as_ref())?;
            _x_gen = Some(gpu_x_gen);
            _gen_ws = Some(gpu_ws);

            is_gpu = true;
            println!("[Hybrid] Switched to GPU successfully.");
        }

        let last_token = tokens_generated.last().copied().unwrap();

        // Input tensor
        let input_tensor = if is_gpu {
            // Must copy from CPU to GPU
            let cpu_indices_buf = cpu_memory.alloc(4, DType::U8)?;
            unsafe {
                *(cpu_indices_buf.as_mut_ptr() as *mut u32) = last_token;
            }
            let cpu_t = Tensor::new(Shape::new(vec![1, 1]), cpu_indices_buf, cpu_backend.clone());
            gpu_backend.copy_from(&cpu_t)?
        } else {
            let cpu_indices_buf = cpu_memory.alloc(4, DType::U8)?;
            unsafe {
                *(cpu_indices_buf.as_mut_ptr() as *mut u32) = last_token;
            }
            Tensor::new(Shape::new(vec![1, 1]), cpu_indices_buf, cpu_backend.clone())
        };

        // Apply decay to accumulated importance scores before this step
        if let Some(ref mut acc) = score_accumulator {
            acc.begin_step();
        }

        model
            .forward_into(LlamaModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos,
                kv_caches: &mut kv_caches,
                backend: &current_backend,
                memory: if is_gpu {
                    gpu_memory.as_ref()
                } else {
                    cpu_memory.as_ref()
                },
                logits_out: &mut logits,
                x_gen: _x_gen.as_mut(),
                workspace: _gen_ws.as_mut(),
                use_gpu_attn: is_gpu,
                cache_manager: cm_ref,
                score_accumulator: score_accumulator.as_mut(),
            })?
            .ok_or(())
            .ok();

        // ── Resilience checkpoint ─────────────────────────
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
                match &action {
                    ResilienceAction::Evict { target_ratio } => {
                        // Use policy-aware eviction via CacheManager.
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
                                    "[Hybrid/Resilience] Evicted {} tokens (pos: {} → {})",
                                    r.tokens_removed,
                                    r.new_pos + r.tokens_removed,
                                    r.new_pos
                                );
                                if let Some(ref mut acc) = score_accumulator {
                                    acc.reset();
                                }
                            }
                            Err(e) => {
                                eprintln!("[Hybrid/Resilience] Eviction error: {}", e)
                            }
                            _ => {}
                        }
                    }
                    ResilienceAction::SwitchBackend { to } => match to {
                        RecommendedBackend::Cpu if is_gpu => {
                            eprintln!(
                                "[Hybrid/Resilience] Switching GPU → CPU at token {}",
                                start_pos
                            );
                            migrate_kv_caches(
                                &mut kv_caches,
                                &gpu_backend,
                                &cpu_backend,
                                &cpu_backend,
                                &cpu_memory,
                                &cpu_memory,
                                kv_heads,
                                head_dim,
                                max_seq_len,
                                false,
                            )?;

                            // 2. Switch Backend
                            current_backend = cpu_backend.clone();

                            // 3. Re-allocate Logits & Workspace on CPU
                            let logits_cpu_buf = cpu_memory.alloc(vocab_size * 4, DType::F32)?;
                            logits = Tensor::new(
                                Shape::new(vec![1, 1, vocab_size]),
                                logits_cpu_buf,
                                cpu_backend.clone(),
                            );
                            let (new_x_gen, new_ws) =
                                setup_workspace(&cpu_backend, cpu_memory.as_ref())?;
                            _x_gen = Some(new_x_gen);
                            _gen_ws = Some(new_ws);

                            is_gpu = false;
                            eprintln!("[Hybrid/Resilience] Switched to CPU successfully.");
                        }
                        RecommendedBackend::Gpu if !is_gpu => {
                            eprintln!(
                                "[Hybrid/Resilience] Switching CPU → GPU at token {}",
                                start_pos
                            );
                            migrate_kv_caches(
                                &mut kv_caches,
                                &cpu_backend,
                                &gpu_backend,
                                &cpu_backend,
                                &cpu_memory,
                                &gpu_memory,
                                kv_heads,
                                head_dim,
                                max_seq_len,
                                true,
                            )?;

                            current_backend = gpu_backend.clone();

                            let logits_gpu_buf = gpu_memory.alloc(vocab_size * 4, DType::F32)?;
                            logits = Tensor::new(
                                Shape::new(vec![1, 1, vocab_size]),
                                logits_gpu_buf,
                                gpu_backend.clone(),
                            );
                            let (new_x_gen, new_ws) =
                                setup_workspace(&gpu_backend, gpu_memory.as_ref())?;
                            _x_gen = Some(new_x_gen);
                            _gen_ws = Some(new_ws);

                            is_gpu = true;
                            eprintln!("[Hybrid/Resilience] Switched to GPU successfully.");
                        }
                        _ => {} // Already on requested backend
                    },
                    _ => execute_action(&action, &mut ctx),
                }
            }

            args.num_tokens = num_tokens;

            if suspended {
                eprintln!("\n[Hybrid/Resilience] Inference suspended by system signal");
                break;
            }

            if throttle_delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
            }
        }
        // ── End Resilience checkpoint ─────────────────────

        // Sample
        let mut logits_cpu = vec![0.0f32; vocab_size];
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
            current_backend.read_buffer(&logits, slice)?;
        }
        let next_token_id = sample(&mut logits_cpu, &tokens_generated, vocab_size, &args);

        tokens_generated.push(next_token_id);
        start_pos += 1;

        let txt = tokenizer
            .decode(&[next_token_id], false)
            .unwrap_or_default();
        print!("{}", txt);
        std::io::stdout().flush().ok();

        if next_token_id == eos_id {
            break;
        }
    }

    println!("\n[Hybrid] Done.");
    Ok(())
}
