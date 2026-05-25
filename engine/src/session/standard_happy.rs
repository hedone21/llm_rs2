//! Phase 4-4: standard happy path (`session::assembly::is_standard_happy_path` 진입)
//! 분기 추출.
//!
//! `bin/generate.rs::main()` L1764~1844 분기를 외과적으로 이동.
//! DecodeLoop + ModelForward 위임 경로.

use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::inference::sampling::{self, SamplingConfig};
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::pressure::kv_cache::KVCache;
use crate::session::assembly::build_standard_loop;
use crate::session::cli::Args;
use crate::session::resilience_adapter::ResilienceAdapter;

pub struct StandardHappyCtx {
    pub args: Args,
    pub backend: Arc<dyn Backend>,
    pub memory: Arc<dyn Memory>,
    pub cpu_backend_arc: Arc<dyn Backend>,
    pub model: TransformerModel,
    pub tokenizer: Tokenizer,
    pub kv_caches: Vec<KVCache>,
    pub tokens: Vec<u32>,
    pub initial_kv_capacity: usize,
    pub max_seq_len: usize,
    pub kv_type: DType,
    pub sampling_config: SamplingConfig,
    pub vocab_size: usize,
    /// P4: ResilienceAdapter 주입 (None 이면 NoOp default).
    pub resilience: Option<ResilienceAdapter>,
}

pub fn run_standard_happy_path(ctx: StandardHappyCtx) -> anyhow::Result<()> {
    let StandardHappyCtx {
        args,
        backend,
        memory,
        cpu_backend_arc,
        model,
        tokenizer,
        kv_caches,
        tokens,
        initial_kv_capacity,
        max_seq_len,
        kv_type,
        sampling_config,
        vocab_size,
        resilience,
    } = ctx;

    eprintln!(
        "[Phase4-4.5] standard happy path → DecodeLoop+ModelForward (tokens={}, budget={})",
        tokens.len(),
        args.num_tokens
    );

    // Drop dead production-fallback KV caches so build_standard_loop's
    // allocation does not coexist with a never-used pool for the lifetime
    // of `main()`.
    drop(kv_caches);

    let mut decode_loop = build_standard_loop(
        backend.clone(),
        memory.clone(),
        cpu_backend_arc.clone(),
        model,
        initial_kv_capacity,
        max_seq_len,
        kv_type,
        sampling_config.clone(),
        !args.no_gpu_plan,
        resilience,
    )?;

    let t_prefill = std::time::Instant::now();
    let mut last_logits = decode_loop.prefill(&tokens)?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    // Phase 4-4.7: first_token을 raw argmax가 아니라 production fallback과
    // 동일한 `sampling::sample(&mut logits, &tokens, ...)` 호출로 산출.
    // `tokens` 전체가 rep history로 들어가 prompt suffix에 rep penalty가
    // 적용된다.
    let first_token = sampling::sample(
        &mut last_logits,
        &tokens,
        vocab_size,
        &sampling_config,
        None,
    );

    let t_decode = std::time::Instant::now();
    let result = decode_loop.run(args.num_tokens - 1, first_token)?;
    let decode_total_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

    let mut final_tokens: Vec<u32> = tokens.clone();
    final_tokens.push(first_token);
    final_tokens.extend_from_slice(&result.tokens_generated);
    let decoded = tokenizer
        .decode(&final_tokens, true)
        .unwrap_or_else(|_| String::from("[decode error]"));
    println!("{}", decoded);

    let decode_tokens = result.tokens_generated.len();
    let total_gen = 1 + decode_tokens;
    let decode_per_tok = if decode_tokens > 0 {
        decode_total_ms / decode_tokens as f64
    } else {
        0.0
    };
    let avg_tbt = (prefill_ms + decode_total_ms) / total_gen as f64;
    println!("TTFT: {:.2} ms", prefill_ms);
    if decode_tokens > 0 {
        println!(
            "Decode: {:.2} ms/tok ({:.1} tok/s) [{} tokens]",
            decode_per_tok,
            1000.0 / decode_per_tok.max(0.001),
            decode_tokens,
        );
    }
    println!(
        "Avg TBT: {:.2} ms ({:.1} tokens/sec)",
        avg_tbt,
        1000.0 / avg_tbt.max(0.001),
    );
    eprintln!(
        "[Phase4-4.5] generated={} (first={} + run={}) stopped_by={:?} final_pos={}",
        total_gen, first_token, decode_tokens, result.stopped_by, result.final_pos
    );
    Ok(())
}
