use clap::Parser;
use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::backend::opencl::OpenCLBackend;
use llm_rs2::core::backend::Backend;
use llm_rs2::core::buffer::{Buffer, DType};
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

fn main() -> anyhow::Result<()> {
    env_logger::init();
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();

    let args = Args::parse();
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
    let cpu_memory = Box::new(Galloc::new());
    // Create GPU memory context but don't use it yet
    let gpu_memory = Box::new(llm_rs2::backend::opencl::memory::OpenCLMemory::new(
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

    let mut kv_caches = Vec::new();
    for _ in 0..num_layers {
        let k_buf = cpu_memory.alloc(max_seq_len * kv_heads * head_dim * 4, DType::F32)?;
        let v_buf = cpu_memory.alloc(max_seq_len * kv_heads * head_dim * 4, DType::F32)?;
        let k = Tensor::new(
            Shape::new(vec![1, max_seq_len, kv_heads, head_dim]),
            k_buf,
            cpu_backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq_len, kv_heads, head_dim]),
            v_buf,
            cpu_backend.clone(),
        );
        kv_caches.push(KVCache::new(k, v, max_seq_len));
    }

    let mut tokens_generated = Vec::new();
    tokens_generated.extend_from_slice(&tokens);
    let mut start_pos = 0;

    // Logits buffer (re-allocated if backend changes)
    let vocab_size = model.config.vocab_size;
    let mut logits_buf = cpu_memory.alloc(vocab_size * 4, DType::F32)?;
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

        model.forward_into(LlamaModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos,
            kv_caches: &mut kv_caches,
            backend: &current_backend,
            memory: cpu_memory.as_ref(),
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            use_gpu_attn: false,
            cache_manager: None,
        })?.ok_or(()).ok(); // No eviction configured

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
    let mut x_gen = None; // Optimization tensors
    let mut gen_ws = None;

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
    x_gen = Some(cpu_x_gen);
    gen_ws = Some(cpu_ws);

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
        // CHECK THRESHOLD AND SWITCH
        if !is_gpu && start_pos >= args.switch_threshold {
            println!("\n\n[Hybrid] Switching to GPU at token {}...", start_pos);

            // 1. Migrate KV Cache
            for (i, kv) in kv_caches.iter_mut().enumerate() {
                // Read from CPU
                let k_size = max_seq_len * kv_heads * head_dim * 4;
                let mut k_data = vec![0u8; k_size];
                let mut v_data = vec![0u8; k_size];

                unsafe {
                    cpu_backend.read_buffer(&kv.k_buffer, &mut k_data)?;
                    cpu_backend.read_buffer(&kv.v_buffer, &mut v_data)?;
                }

                // Create CPU Tensor from data
                let k_cpu_buf = cpu_memory.alloc(k_size, DType::F32)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        k_data.as_ptr(),
                        k_cpu_buf.as_mut_ptr() as *mut u8,
                        k_size,
                    );
                }
                let k_cpu_tensor =
                    Tensor::new(kv.k_buffer.shape().clone(), k_cpu_buf, cpu_backend.clone());

                let v_cpu_buf = cpu_memory.alloc(k_size, DType::F32)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        v_data.as_ptr(),
                        v_cpu_buf.as_mut_ptr() as *mut u8,
                        k_size,
                    );
                }
                let v_cpu_tensor =
                    Tensor::new(kv.v_buffer.shape().clone(), v_cpu_buf, cpu_backend.clone());

                // Copy to GPU and Create KVCache
                *kv = KVCache::new(
                    gpu_backend.copy_from(&k_cpu_tensor)?,
                    gpu_backend.copy_from(&v_cpu_tensor)?,
                    max_seq_len,
                );
            }

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
            x_gen = Some(gpu_x_gen);
            gen_ws = Some(gpu_ws);

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

        model.forward_into(LlamaModelForwardArgs {
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
            x_gen: x_gen.as_mut(),
            workspace: gen_ws.as_mut(),
            use_gpu_attn: is_gpu,
            cache_manager: None,
        })?.ok_or(()).ok(); // No eviction configured

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
