//! Phase 4-3 C3: vtable overhead microbench for `DecodeLoop + ModelForward`.
//!
//! Compares two decode paths driven against the same backend, model, and
//! prompt:
//!   `path_a`: `DecodeLoopBuilder::new().with_forward(ModelForward::new(...))
//!               .with_sampler(GreedySampler).build()` then `prefill` +
//!               `run(budget - 1)`.
//!   `path_b`: hand-rolled `model.forward_into()` loop with external greedy
//!               argmax — the production decode pattern from `generate.rs`.
//!
//! Outputs a single JSON document on stderr with TBT, tok0, delta_pct and
//! `bit_identical_first_N`. PASS gate (Phase 4-3 §G8):
//!   `delta_pct <= 5.0` AND `bit_identical_first_N == true`.
//!
//! WARMUP runs once per path before measurement so the lazy
//! `prefill_workspace` allocation, kernel JIT, and cache filling do not
//! contaminate `avg_tbt` (Phase 4-3 §P4 "Hybrid"). All TBT values are
//! tok0-inclusive — see `feedback_tbt_metric_tok0_inclusive.md`.
//!
//! ## Decode-paradigm note
//! `DecodeLoop::prefill` currently returns `()` — the first sampled token
//! must come from `step(pos=prompt_len, prev_token=prompt[last])`. To keep
//! the comparison fair, the direct path here uses the *same paradigm*
//! (prompt last is forwarded again on the first step) rather than the
//! production `generate.rs` shape (which samples the first token from
//! the prefill last-only logits). The probe is measuring "wrapper vs.
//! direct *under the DecodeLoop paradigm*", not "production decode loop
//! shape". Once `DecodeLoop::prefill` is taught to return `Vec<f32>` and
//! `run(budget, first_token)` is reshaped (Phase 4-3.5 follow-up), both
//! paradigms can be unified.
//!
//! `tok0_ms` therefore covers `prefill + first_step + first_sample` for
//! both paths so the warm-up cost is symmetric.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use serde_json::json;

use llm_rs2::backend::Backend;
use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::buffer::DType;
use llm_rs2::format::KVCacheFormat;
use llm_rs2::kv::standard_format::StandardFormat;
use llm_rs2::layers::workspace::{LayerWorkspace, WorkspaceConfig};
use llm_rs2::memory::Memory;
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use llm_rs2::session::forward::{ModelForward, alloc_standard_kv_caches};
use llm_rs2::session::{DecodeLoopBuilder, Forward, GreedySampler, StepCtx};
use llm_rs2::shape::Shape;
use llm_rs2::tensor::Tensor;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Phase 4-3 vtable microbench: DecodeLoop+ModelForward vs direct forward_into"
)]
struct Args {
    #[arg(long)]
    model_path: String,

    #[arg(long)]
    tokenizer_path: String,

    /// Backend: cpu | opencl | cuda. Feature-gated.
    #[arg(long, default_value = "cpu")]
    backend: String,

    #[arg(long, default_value = "The capital of France is")]
    prompt: String,

    /// Generated tokens per run.
    #[arg(long = "gen", default_value_t = 32)]
    gen_tokens: usize,

    /// Measurement runs (median used).
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// KV / workspace cap.
    #[arg(long, default_value_t = 512)]
    max_seq_len: usize,

    /// KV dtype: f16 | f32 | q4.
    #[arg(long, default_value = "f16")]
    kv_dtype: String,

    /// PASS threshold for `delta_pct` (percent).
    #[arg(long, default_value_t = 5.0)]
    pass_pct: f64,
}

fn parse_kv_dtype(s: &str) -> Result<DType> {
    match s.to_lowercase().as_str() {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "q4" | "q4_0" => Ok(DType::Q4_0),
        other => anyhow::bail!("--kv-dtype: unsupported value '{other}'"),
    }
}

fn build_backend(name: &str) -> Result<(Arc<dyn Backend>, Arc<dyn Memory>)> {
    match name.to_lowercase().as_str() {
        "cpu" => Ok((
            Arc::new(CpuBackend::new()) as Arc<dyn Backend>,
            Arc::new(Galloc::new()) as Arc<dyn Memory>,
        )),
        #[cfg(feature = "opencl")]
        "opencl" | "gpu" => {
            let gpu = Arc::new(llm_rs2::backend::opencl::OpenCLBackend::new()?);
            let mem: Arc<dyn Memory> =
                Arc::new(llm_rs2::backend::opencl::memory::OpenCLMemory::new(
                    gpu.context.clone(),
                    gpu.queue.clone(),
                    false,
                ));
            Ok((gpu as Arc<dyn Backend>, mem))
        }
        #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
        "cuda" => {
            let gpu = Arc::new(llm_rs2::backend::cuda::CudaBackend::new()?);
            let mem: Arc<dyn Memory> = if gpu.is_discrete_gpu() {
                Arc::new(llm_rs2::backend::cuda::memory::CudaMemory::managed())
            } else {
                Arc::new(llm_rs2::backend::cuda::memory::CudaMemory::new())
            };
            Ok((gpu as Arc<dyn Backend>, mem))
        }
        other => anyhow::bail!(
            "--backend: '{other}' not available in this build (compile with the right feature)"
        ),
    }
}

fn greedy_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn read_logits(backend: &Arc<dyn Backend>, logits: &Tensor, vocab: usize) -> Result<Vec<f32>> {
    backend.synchronize()?;
    let mut out = vec![0.0f32; vocab];
    unsafe {
        let bytes = std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, vocab * 4);
        backend.read_buffer(logits, bytes)?;
    }
    Ok(out)
}

fn upload_prompt(
    backend: &Arc<dyn Backend>,
    cpu_backend: &Arc<dyn Backend>,
    tokens: &[u32],
) -> Result<Tensor> {
    let cpu_buf = Galloc::new().alloc(tokens.len() * 4, DType::U8)?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            tokens.as_ptr(),
            cpu_buf.as_mut_ptr() as *mut u32,
            tokens.len(),
        );
    }
    let cpu_tensor = Tensor::new(
        Shape::new(vec![1, tokens.len()]),
        cpu_buf,
        cpu_backend.clone(),
    );
    backend.copy_from(&cpu_tensor)
}

/// Per-run measurement: returns (tok0_ms, per_step_ms vec of length budget).
/// The DecodeLoop path samples its first token from `step` (not from prefill),
/// so its `result.tokens_generated` already contains `budget` tokens.
struct RunSample {
    tokens: Vec<u32>,
    tok0_ms: f64,
    rest_avg_ms: f64,
    total_decode_ms: f64,
}

fn impl_run_decode_loop(
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    model: &Arc<TransformerModel>,
    prompt: &[u32],
    budget: usize,
    max_seq_len: usize,
    kv_dtype: DType,
) -> Result<RunSample> {
    let cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let initial_capacity = max_seq_len.next_power_of_two().min(max_seq_len);
    let kv = alloc_standard_kv_caches(
        model,
        backend.clone(),
        memory.clone(),
        initial_capacity,
        max_seq_len,
        kv_dtype,
    )?;
    let forward = ModelForward::new(
        backend.clone(),
        memory.clone(),
        cpu_backend,
        model.clone(),
        kv,
        max_seq_len,
        // Phase 4-4.7: microbench는 vtable overhead만 측정. plan path는 별도
        // device G7' 게이트로 검증되므로 여기서는 비활성화 → forward_into fallback만.
        false,
    )?;

    let mut decode_loop = DecodeLoopBuilder::new()
        .with_forward(forward)
        .with_sampler(GreedySampler)
        .build();

    let t_total = Instant::now();
    let t0 = Instant::now();
    // Phase 4-4.5 paradigm: prefill returns last logits, caller samples the
    // first generated token, then run(budget-1, first_token) samples the rest.
    let last_logits = decode_loop.prefill(prompt)?;
    let first_token = greedy_argmax(&last_logits);
    let tok0_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut tokens = Vec::with_capacity(budget);
    tokens.push(first_token);

    if budget > 1 {
        let rest = decode_loop.run(budget - 1, first_token)?;
        tokens.extend(rest.tokens_generated.iter().copied());
    }

    let total_decode_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    let rest_avg_ms = if budget > 1 {
        (total_decode_ms - tok0_ms) / (budget - 1) as f64
    } else {
        0.0
    };

    Ok(RunSample {
        tokens,
        tok0_ms,
        rest_avg_ms,
        total_decode_ms,
    })
}

/// Direct path mirrors the Phase 4-4.5 unified paradigm:
///   1. prefill (`logits_last_only=true`), read back last logits, argmax.
///   2. `first_token` = argmax of prefill last logits (matches
///      `DecodeLoop::prefill -> Result<Vec<f32>>` + caller-side sampling).
///   3. first decode step uses `prev = first_token`; subsequent steps use the
///      previously sampled token.
///
/// `delta_pct` measures `DecodeLoop` wrapper overhead against this direct
/// reference. Both paths now produce the same token stream as production
/// `generate.rs` since Phase 4-4.5 unified the paradigm.
fn impl_run_direct(
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    model: &Arc<TransformerModel>,
    prompt: &[u32],
    budget: usize,
    max_seq_len: usize,
    kv_dtype: DType,
) -> Result<RunSample> {
    let cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let vocab = model.config.vocab_size;
    let hidden = model.config.hidden_size;
    let head_dim = model.config.head_dim;
    let kv_dim = model.config.num_key_value_heads * head_dim;
    let cfg = WorkspaceConfig {
        batch_size: 1,
        dim: hidden,
        q_dim: model.config.num_attention_heads * head_dim,
        k_dim: kv_dim,
        v_dim: kv_dim,
        ffn_hidden: model.config.intermediate_size,
        n_heads: model.config.num_attention_heads,
        max_seq_len,
    };
    let initial_capacity = max_seq_len.next_power_of_two().min(max_seq_len);
    let kv = alloc_standard_kv_caches(
        model,
        backend.clone(),
        memory.clone(),
        initial_capacity,
        max_seq_len,
        kv_dtype,
    )?;
    // Phase α-K Step 5-F: production(`ModelForward`)과 동일하게 KVCache 를 StandardFormat
    // 으로 wrap 한 뒤 `forward_into`(trait object) 경로로 측정한다. OLD `forward_into<C>`
    // 폐기에 맞춘 미러.
    let dyn_fmts: Vec<Arc<dyn KVCacheFormat>> = kv
        .into_iter()
        .enumerate()
        .map(|(i, c)| Arc::new(StandardFormat::new(i, c)) as Arc<dyn KVCacheFormat>)
        .collect();

    let prefill_input = upload_prompt(backend, &cpu_backend, prompt)?;
    let prefill_logits_buf = memory.alloc(vocab * 4, DType::F32)?;
    let mut prefill_logits = Tensor::new(
        Shape::new(vec![1, 1, vocab]),
        prefill_logits_buf,
        backend.clone(),
    );

    // Pre-alloc decode-side scratch up-front so the timed region covers only
    // the work that DecodeLoop also pays per token (`ModelForward::new` does
    // the same).
    let mut ws = LayerWorkspace::new(cfg, memory.as_ref(), backend.clone())?;
    let x_gen_buf = memory.alloc(hidden * 4, DType::F32)?;
    let mut x_gen = Tensor::new(Shape::new(vec![1, 1, hidden]), x_gen_buf, backend.clone());
    let decode_input_buf = memory.alloc(4, DType::U8)?;
    let mut decode_input = Tensor::new(Shape::new(vec![1, 1]), decode_input_buf, backend.clone());
    let decode_logits_buf = memory.alloc(vocab * 4, DType::F32)?;
    let mut decode_logits = Tensor::new(
        Shape::new(vec![1, 1, vocab]),
        decode_logits_buf,
        backend.clone(),
    );

    let t_total = Instant::now();
    let t0 = Instant::now();

    // Prefill — paradigm 통일: read last logits and argmax for first_token.
    model.forward_into(TransformerModelForwardArgs {
        input_tokens: &prefill_input,
        start_pos: 0,
        fmts: &dyn_fmts,
        backend,
        memory: memory.as_ref(),
        logits_out: &mut prefill_logits,
        x_gen: None,
        workspace: None,
        score_accumulator: None,
        skip_config: None,
        importance_collector: None,
        logits_last_only: true,
        cache_self_need_scores: false,
    })?;
    let prefill_logits_host = read_logits(backend, &prefill_logits, vocab)?;
    let first_token = greedy_argmax(&prefill_logits_host);
    let tok0_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut prev = first_token;
    let mut pos = prompt.len();
    let mut tokens = vec![first_token];

    while tokens.len() < budget {
        backend.write_buffer(&mut decode_input, &prev.to_ne_bytes())?;
        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &decode_input,
            start_pos: pos,
            fmts: &dyn_fmts,
            backend,
            memory: memory.as_ref(),
            logits_out: &mut decode_logits,
            x_gen: Some(&mut x_gen),
            workspace: Some(&mut ws),
            score_accumulator: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            cache_self_need_scores: false,
        })?;
        let logits = read_logits(backend, &decode_logits, vocab)?;
        let next = greedy_argmax(&logits);
        tokens.push(next);
        prev = next;
        pos += 1;
    }

    let total_decode_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    let rest_avg_ms = if budget > 1 {
        (total_decode_ms - tok0_ms) / (budget - 1) as f64
    } else {
        0.0
    };

    Ok(RunSample {
        tokens,
        tok0_ms,
        rest_avg_ms,
        total_decode_ms,
    })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

fn measure<F>(label: &str, runs: usize, mut once: F) -> Result<MeasureSummary>
where
    F: FnMut() -> Result<RunSample>,
{
    eprintln!("[probe] {label}: WARMUP 1 run");
    let warmup = once()?;
    eprintln!(
        "[probe] {label}: warmup tok0={:.2} ms rest_avg={:.2} ms total={:.2} ms tokens={}",
        warmup.tok0_ms,
        warmup.rest_avg_ms,
        warmup.total_decode_ms,
        warmup.tokens.len()
    );

    let mut tok0 = Vec::with_capacity(runs);
    let mut avg_tbt = Vec::with_capacity(runs);
    let mut total = Vec::with_capacity(runs);
    let mut sample_tokens: Vec<u32> = Vec::new();
    for i in 0..runs {
        let s = once()?;
        eprintln!(
            "[probe] {label}: run {}/{} tok0={:.2} rest_avg={:.2} total={:.2}",
            i + 1,
            runs,
            s.tok0_ms,
            s.rest_avg_ms,
            s.total_decode_ms
        );
        tok0.push(s.tok0_ms);
        total.push(s.total_decode_ms);
        let avg = s.total_decode_ms / s.tokens.len().max(1) as f64;
        avg_tbt.push(avg);
        if i == 0 {
            sample_tokens = s.tokens;
        }
    }
    Ok(MeasureSummary {
        tokens: sample_tokens,
        tok0_median: median(&mut tok0),
        avg_tbt_median: median(&mut avg_tbt),
        total_median: median(&mut total),
    })
}

struct MeasureSummary {
    tokens: Vec<u32>,
    tok0_median: f64,
    avg_tbt_median: f64,
    total_median: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let kv_dtype = parse_kv_dtype(&args.kv_dtype)?;

    eprintln!(
        "[probe] backend={} model={} prompt={:?} gen={} runs={}",
        args.backend, args.model_path, args.prompt, args.gen_tokens, args.runs
    );

    let (backend, memory) = build_backend(&args.backend)?;
    let tokenizer = Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer::from_file: {e}"))?;
    let model = Arc::new(TransformerModel::load_gguf(
        &args.model_path,
        backend.clone(),
        memory.as_ref(),
    )?);
    let prompt_tokens: Vec<u32> = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    eprintln!("[probe] prompt_tokens={}", prompt_tokens.len());

    let decode_loop_summary = measure("decode_loop", args.runs, || {
        impl_run_decode_loop(
            &backend,
            &memory,
            &model,
            &prompt_tokens,
            args.gen_tokens,
            args.max_seq_len,
            kv_dtype,
        )
    })?;
    let direct_summary = measure("direct", args.runs, || {
        impl_run_direct(
            &backend,
            &memory,
            &model,
            &prompt_tokens,
            args.gen_tokens,
            args.max_seq_len,
            kv_dtype,
        )
    })?;

    let delta_pct = (decode_loop_summary.avg_tbt_median - direct_summary.avg_tbt_median).abs()
        / direct_summary.avg_tbt_median.max(f64::EPSILON)
        * 100.0;
    let bit_identical = decode_loop_summary.tokens == direct_summary.tokens;
    let verdict = if delta_pct <= args.pass_pct && bit_identical {
        "PASS"
    } else {
        "FAIL"
    };

    let report = json!({
        "model": args.model_path,
        "tokenizer": args.tokenizer_path,
        "backend": args.backend,
        "kv_dtype": args.kv_dtype,
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens.len(),
        "gen": args.gen_tokens,
        "runs": args.runs,
        "max_seq_len": args.max_seq_len,
        "pass_pct_threshold": args.pass_pct,
        "avg_tbt_ms": {
            "decode_loop": decode_loop_summary.avg_tbt_median,
            "direct": direct_summary.avg_tbt_median,
        },
        "tok0_ms": {
            "decode_loop": decode_loop_summary.tok0_median,
            "direct": direct_summary.tok0_median,
        },
        "total_ms": {
            "decode_loop": decode_loop_summary.total_median,
            "direct": direct_summary.total_median,
        },
        "tokens": {
            "decode_loop": decode_loop_summary.tokens,
            "direct": direct_summary.tokens,
        },
        "delta_pct": delta_pct,
        "bit_identical_first_n": bit_identical,
        "verdict": verdict,
    });

    eprintln!("{}", serde_json::to_string_pretty(&report)?);
    println!("{verdict}");

    if verdict == "FAIL" {
        anyhow::bail!(
            "Phase 4-3 vtable gate FAIL: delta_pct={delta_pct:.2}% bit_identical={bit_identical}"
        );
    }
    Ok(())
}

// Workaround: silence unused warnings on the imports the bench retains for
// future signal/stop integration even though it does not exercise them yet.
#[allow(dead_code)]
fn _keep_unused_imports() {
    let _ = std::any::type_name::<AtomicBool>();
    let _ = std::any::type_name::<StepCtx>();
    let _: fn(&mut dyn Forward) = |_| {};
}
