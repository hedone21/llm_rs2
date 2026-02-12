use clap::Parser;
use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::core::backend::Backend;
use llm_rs2::core::buffer::DType;
use llm_rs2::core::kv_cache::KVCache;
use llm_rs2::core::memory::Memory;
use llm_rs2::core::shape::Shape;
use llm_rs2::core::tensor::Tensor;
use llm_rs2::layers::workspace::{LayerWorkspace, WorkspaceConfig};
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::models::llama::llama_model::{LlamaModel, LlamaModelForwardArgs};
use std::sync::Arc;
use tokenizers::Tokenizer;

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

    let args = Args::parse();
    let model_path = &args.model_path;

    // 1. Setup
    println!("[Profile] Event: ModelLoadStart");
    println!("Loading model from {}", model_path);
    let backend: Arc<dyn Backend> = match args.backend.as_str() {
        "cpu" => Arc::new(CpuBackend::new()),
        "opencl" => Arc::new(llm_rs2::backend::opencl::OpenCLBackend::new()?),
        _ => anyhow::bail!("Unknown backend: {}", args.backend),
    };
    let memory: Box<dyn Memory> = if args.backend == "opencl" {
        let ocl_backend = backend
            .as_any()
            .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
            .ok_or(anyhow::anyhow!("Failed into cast to OpenCLBackend"))?;
        Box::new(llm_rs2::backend::opencl::memory::OpenCLMemory::new(
            ocl_backend.context.clone(),
            ocl_backend.queue.clone(),
            args.zero_copy,
        ))
    } else {
        Box::new(Galloc::new())
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
        _ => anyhow::bail!("Unsupported KV type: {}. Use f32, f16, or q4.", args.kv_type),
    };
    // Calculate buffer size per KV cache
    let n_values = max_seq_len * kv_heads * head_dim;
    let kv_buf_size = match kv_type {
        DType::Q4_0 => {
            use llm_rs2::core::quant::{BlockQ4_0, QK4_0};
            (n_values / QK4_0) * std::mem::size_of::<BlockQ4_0>()
        },
        _ => n_values * kv_type.size(),
    };
    println!("KV cache type: {:?} ({}B total per layer)", kv_type, kv_buf_size);

    let mut kv_caches = Vec::new();
    for _ in 0..num_layers {
        let k_buf = memory.alloc(kv_buf_size, kv_type)?;
        let v_buf = memory.alloc(kv_buf_size, kv_type)?;

        let k = Tensor::new(
            Shape::new(vec![1, max_seq_len, kv_heads, head_dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq_len, kv_heads, head_dim]),
            v_buf,
            backend.clone(),
        );

        kv_caches.push(KVCache::new(k, v, max_seq_len));
    }

    // 5. Inference Loop
    let mut tokens = input_ids.clone();
    let mut start_pos = 0;
    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;

    println!(
        "Generating (Max: {}, Temp: {}, TopP: {}, TopK: {})...",
        max_seq_len, args.temperature, args.top_p, args.top_k
    );
    let start_time = std::time::Instant::now();
    let mut last_token_time = start_time;
    let mut ttft_ms = 0.0;
    let mut tbt_values = Vec::new();

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
        })?;

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

        ttft_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        last_token_time = std::time::Instant::now();

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
        let mut printed_len = 0;

        // Print initial tokens (prompt + first generated)
        let initial_text = tokenizer.decode(&tokens, true).unwrap_or_default();
        print!("{}", initial_text);
        printed_len = initial_text.len();
        stdout.flush().ok();

        // Generation loop
        for _ in 0..(args.num_tokens - 1) {
            if start_pos >= max_seq_len {
                println!("\n[Stopped: Max context length reached]");
                break;
            }

            let last_token = tokens[tokens.len() - 1];
            unsafe {
                *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_token;
            }
            let gen_input_tensor = backend.copy_from(&cpu_gen_input)?;

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
            })?;

            // Sample
            // Read logits
            let mut logits_cpu = vec![0.0f32; vocab_size];
            unsafe {
                let ptr = logits_cpu.as_mut_ptr() as *mut u8;
                let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
                backend.read_buffer(&logits, slice)?;
            }

            let next_token_id = sample(&mut logits_cpu, &tokens, vocab_size, &args);

            let now = std::time::Instant::now();
            let tbt = now.duration_since(last_token_time).as_secs_f64() * 1000.0;
            tbt_values.push(tbt);

            last_token_time = now;
            tokens.push(next_token_id);
            start_pos += 1;

            // Streaming print
            let current_text = tokenizer.decode(&tokens, true).unwrap_or_default();
            if current_text.len() > printed_len {
                // Check if we are at a valid char boundary.
                // If not (e.g. we are in the middle of a multi-byte char sequence from previous partial decode?),
                // we might need to be careful.
                // However, tokenizer.decode should return valid strings.
                // The issue is likely that `printed_len` (bytes) might not align with `current_text` if decoding changed slightly?
                // Or `printed_len` was set from a previous string.

                // Safe slicing:
                if let Some(substring) = current_text.get(printed_len..) {
                    print!("{}", substring);
                    stdout.flush().ok();
                    printed_len = current_text.len();
                } else {
                    // Verify if printed_len is valid.
                    // Often tokenizers re-decode slightly differently or we accumulate.
                    // A safer way is: just print what's new from this round's decode, but we need to track bytes.
                    // Let's just catch the case where we can't slice.
                }
            }

            if next_token_id == eos_id {
                break;
            }
        }
    }

    // 6. Output results
    println!("\nDone.");
    println!("[Profile] Event: End");
    // let full_text = tokenizer.decode(&tokens[input_ids.len()..], true).map_err(|e| anyhow::anyhow!(e))?;
    // println!("Generated: {}", full_text);
    println!("TTFT: {:.2} ms", ttft_ms);
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
