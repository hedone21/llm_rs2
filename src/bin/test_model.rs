use llm_rs2::core::backend::Backend;
use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::core::memory::Memory;
use llm_rs2::models::llama::llama_model::LlamaModel;
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    println!("=== LLM RS2 Model Loading Test ===");

    let backend = Arc::new(CpuBackend::new());
    let memory = Galloc::new();

    let model_path = "/data/local/tmp/llm_rs2/models/llama3.2-1b";
    println!("Loading model from: {}", model_path);

    let start = Instant::now();
    let model = LlamaModel::load(model_path, backend.clone(), &memory)?;
    let duration = start.elapsed();

    println!("Model loaded successfully in {:?}", duration);
    println!("Config: {:?}", model.config);
    println!("Number of layers: {}", model.layers.len());
    println!("Vocab size: {}", model.config.vocab_size);
    println!("Embed Tokens Shape: {:?}", model.embed_tokens.shape().dims());

    Ok(())
}
