use llm_rs2::core::backend::Backend;
use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::core::memory::Memory;
use llm_rs2::models::llama::llama_model::LlamaModel;
use llm_rs2::core::tensor::Tensor;
use llm_rs2::core::shape::Shape;
use llm_rs2::core::buffer::DType;
use llm_rs2::core::kv_cache::KVCache;
use llm_rs2::layers::workspace::LayerWorkspace;
use std::sync::Arc;
use tokenizers::Tokenizer;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "models/llama3.2-1b")]
    model_path: String,
    
    #[arg(short, long, default_value = "Hello, world! I am a")]
    prompt: String,
    
    #[arg(short, long, default_value_t = 20)]
    num_tokens: usize,

    #[arg(short, long, default_value = "cpu")]
    backend: String,

    /// Use zero-copy shared memory (slower but enables CPU-GPU sharing)
    #[arg(long, default_value_t = false)]
    zero_copy: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    
    // Configure Rayon to use 8 threads
    rayon::ThreadPoolBuilder::new().num_threads(8).build_global().unwrap();
    
    let args = Args::parse();
    let model_path = &args.model_path;
    
    // 1. Setup
    println!("Loading model from {}", model_path);
    let backend: Arc<dyn Backend> = match args.backend.as_str() {
        "cpu" => Arc::new(CpuBackend::new()),
        "opencl" => Arc::new(llm_rs2::backend::opencl::OpenCLBackend::new()?),
        _ => anyhow::bail!("Unknown backend: {}", args.backend),
    };
    let memory: Box<dyn Memory> = if args.backend == "opencl" {
        let ocl_backend = backend.as_any().downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
            .ok_or(anyhow::anyhow!("Failed into cast to OpenCLBackend"))?;
        Box::new(llm_rs2::backend::opencl::memory::OpenCLMemory::new(
            ocl_backend.context.clone(), 
            ocl_backend.queue.clone(),
            args.zero_copy
        ))
    } else {
        Box::new(Galloc::new())
    };
    let model = LlamaModel::load(model_path, backend.clone(), &*memory)?;
    
    // 2. Tokenizer
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
        .map_err(|e| anyhow::anyhow!(e))?;
    
    // 3. Prompt
    let prompt = &args.prompt;
    println!("Prompt: {}", prompt);
    
    let encoding = tokenizer.encode(prompt.as_str(), true).map_err(|e| anyhow::anyhow!(e))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    
    // 4. Prepare KV Cache
    let max_seq_len = 128;
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;
    let num_layers = model.config.num_hidden_layers;
    
    let mut kv_caches = Vec::new();
    for _ in 0..num_layers {
        let k_buf = memory.alloc(max_seq_len * kv_heads * head_dim * 4, DType::F32)?;
        let v_buf = memory.alloc(max_seq_len * kv_heads * head_dim * 4, DType::F32)?;
        
        let k = Tensor::new(Shape::new(vec![1, max_seq_len, kv_heads, head_dim]), k_buf, backend.clone());
        let v = Tensor::new(Shape::new(vec![1, max_seq_len, kv_heads, head_dim]), v_buf, backend.clone());
        
        kv_caches.push(KVCache::new(k, v, max_seq_len));
    }

    // 5. Inference Loop
    let mut tokens = input_ids.clone();
    let mut start_pos = 0;
    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;
    
    println!("Generating...");
    let start_time = std::time::Instant::now();
    let mut last_token_time = start_time; // Keep for now as baseline for prefill
    let mut ttft_ms = 0.0;
    let mut tbt_values = Vec::new();
    
    // Pre-allocate generation buffers
    let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut logits = Tensor::new(Shape::new(vec![1, 1, vocab_size]), logits_buf, backend.clone());
    // gen_indices_buf removed (will use CPU tensor)
    
    // Cache EOS token ID
    let eos_id = tokenizer.get_vocab(true).get("</s>").copied().unwrap_or(u32::MAX);
    
    // === PREFILL PHASE ===
    {
        let process_len = tokens.len();
        // Create CPU tensor for input
        let cpu_indices_buf = Galloc::new().alloc(process_len * 4, DType::U8)?;
        unsafe {
            let ptr = cpu_indices_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, process_len);
        }
        let cpu_input_tensor = Tensor::new(Shape::new(vec![1, process_len]), cpu_indices_buf, Arc::new(CpuBackend::new()));
        let input_tensor = backend.copy_from(&cpu_input_tensor)?;
        
        let prefill_logits_buf = memory.alloc(process_len * vocab_size * 4, DType::F32)?;
        let mut prefill_logits = Tensor::new(
            Shape::new(vec![1, process_len, vocab_size]),
            prefill_logits_buf,
            backend.clone()
        );
        
        model.forward_into(&input_tensor, start_pos, &mut kv_caches, &backend, memory.as_ref(), &mut prefill_logits, None, None)?;
        
        // Argmax on last token
        // Read logits to CPU
        let mut logits_cpu = vec![0.0f32; process_len * vocab_size];
        unsafe {
             let ptr = logits_cpu.as_mut_ptr() as *mut u8;
             let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
             backend.read_buffer(&prefill_logits, slice)?;
        }
        let logits_data = &logits_cpu;
        
        let last_logits = &logits_data[(process_len - 1) * vocab_size..process_len * vocab_size];
        
        let next_token_id = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx as u32)
            .unwrap();
        
        ttft_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        last_token_time = std::time::Instant::now();
        
        tokens.push(next_token_id);
        start_pos += process_len;
    }
    
    // === GENERATION PHASE ===
    {
        // Pre-allocate workspace for generation
        let q_dim = hidden_size;
        let k_dim = hidden_size / 4;
        let v_dim = k_dim;
        let ffn_hidden = model.config.intermediate_size;
        
        let x_gen_buf = memory.alloc(hidden_size * 4, DType::F32)?;
        let mut x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), x_gen_buf, backend.clone());
        
        let mut gen_ws = LayerWorkspace::new(
            1, hidden_size, q_dim, k_dim, v_dim, ffn_hidden, max_seq_len, memory.as_ref(), backend.clone()
        )?;
        
        // Single token CPU tensor for generation loop
        let cpu_gen_indices_buf = Galloc::new().alloc(4, DType::U8)?;
        let cpu_gen_input = Tensor::new(Shape::new(vec![1, 1]), cpu_gen_indices_buf, Arc::new(CpuBackend::new()));

        // Streaming setup
        use std::io::Write;
        let mut stdout = std::io::stdout();
        let mut printed_len = 0;

        // Generation loop
        for _ in 0..(args.num_tokens - 1) {
            let last_token = tokens[tokens.len() - 1];
            unsafe {
                *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_token;
            }
            let gen_input_tensor = backend.copy_from(&cpu_gen_input)?;
            
            model.forward_into(&gen_input_tensor, start_pos, &mut kv_caches, &backend, memory.as_ref(), &mut logits, Some(&mut x_gen), Some(&mut gen_ws))?;
            
            // Argmax
            // Read logits
            let mut logits_cpu = vec![0.0f32; vocab_size];
            unsafe {
                 let ptr = logits_cpu.as_mut_ptr() as *mut u8;
                 let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
                 backend.read_buffer(&logits, slice)?;
            }
            let logits_data = &logits_cpu;
            let next_token_id = logits_data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx as u32)
                .unwrap();
            
            let now = std::time::Instant::now();
            let tbt = now.duration_since(last_token_time).as_secs_f64() * 1000.0;
            tbt_values.push(tbt);
            
            last_token_time = now;
            tokens.push(next_token_id);
            start_pos += 1;
            
            // Streaming print
            let current_text = tokenizer.decode(&tokens[input_ids.len()..], true).unwrap_or_default();
            if current_text.len() > printed_len {
                print!("{}", &current_text[printed_len..]);
                stdout.flush().ok();
                printed_len = current_text.len();
            }

            if next_token_id == eos_id {
                break;
            }
        }
    }
    
    // 6. Output results
    println!("\nDone.");
    let full_text = tokenizer.decode(&tokens[input_ids.len()..], true).map_err(|e| anyhow::anyhow!(e))?;
    println!("Generated: {}", full_text);
    println!("TTFT: {:.2} ms", ttft_ms);
    if !tbt_values.is_empty() {
        let avg_tbt: f64 = tbt_values.iter().sum::<f64>() / tbt_values.len() as f64;
        println!("Avg TBT: {:.2} ms ({:.1} tokens/sec)", avg_tbt, 1000.0 / avg_tbt);
    }

    Ok(())
}
