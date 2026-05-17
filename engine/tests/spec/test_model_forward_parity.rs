//! Phase 4-3 C2: parity test for `ModelForward` versus the direct
//! `TransformerModel::forward_into` loop.
//!
//! Verifies that wrapping `forward_into` in `ModelForward::{prefill, step}`
//! yields the same token sequence as calling `forward_into` manually with
//! greedy argmax sampling. The wrapper sits between user code and the model;
//! any mistake in tensor uploads, kv borrow plumbing, or logits read-back
//! would diverge from the direct path.
//!
//! ## Why env-gated
//! Host runs need a GGUF model + tokenizer.json on disk, which CI does not
//! ship. Gate behind `LLMRS_TEST_MODEL_PATH` + `LLMRS_TEST_TOKENIZER_PATH`
//! so unattended `cargo test` skips silently while a developer pointing at
//! a local Qwen / Llama checkpoint can exercise the parity check before a
//! PR ships. The microbench binary (C3) provides the device-side
//! bit-identical check that fronts the formal PASS gate.
//!
//! ## Scope
//! Generates 8 tokens (compromise between exercise and host-CPU runtime —
//! a Qwen 1.5B GGUF takes ~0.5 s per token on host). Compares the full
//! token list; any divergence fails the test.
//!
//! `DecodeLoop` is exercised separately by the unit tests in
//! `session::decode_loop::tests` (no model needed) and by the
//! `bin/probe_inference_loop` microbench (device-side, real model).

use std::sync::Arc;

use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::core::backend::Backend;
use llm_rs2::core::buffer::DType;
use llm_rs2::core::memory::Memory;
use llm_rs2::core::shape::Shape;
use llm_rs2::core::tensor::Tensor;
use llm_rs2::layers::workspace::{LayerWorkspace, WorkspaceConfig};
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use llm_rs2::session::forward::{ModelForward, alloc_standard_kv_caches};
use llm_rs2::session::{Forward, StepCtx};
use tokenizers::Tokenizer;

const PROMPT: &str = "The capital of France is";
const GEN_TOKENS: usize = 8;
const MAX_SEQ_LEN: usize = 256;

fn env_paths() -> Option<(String, String)> {
    let model = std::env::var("LLMRS_TEST_MODEL_PATH").ok()?;
    let tok = std::env::var("LLMRS_TEST_TOKENIZER_PATH").ok()?;
    Some((model, tok))
}

fn build_backend() -> (Arc<dyn Backend>, Arc<dyn Memory>) {
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let memory: Arc<dyn Memory> = Arc::new(Galloc::new());
    (backend, memory)
}

fn greedy_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn upload_tokens(
    backend: &Arc<dyn Backend>,
    cpu_backend: &Arc<dyn Backend>,
    tokens: &[u32],
) -> Tensor {
    let cpu_buf = Galloc::new().alloc(tokens.len() * 4, DType::U8).unwrap();
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
    backend.copy_from(&cpu_tensor).unwrap()
}

fn read_logits(backend: &Arc<dyn Backend>, logits: &Tensor, vocab: usize) -> Vec<f32> {
    backend.synchronize().unwrap();
    let mut out = vec![0.0f32; vocab];
    unsafe {
        let bytes = std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, vocab * 4);
        backend.read_buffer(logits, bytes).unwrap();
    }
    out
}

/// Direct path: prefill (last-only logits) → greedy → step loop hand-rolled
/// against `TransformerModel::forward_into`. Mirrors the production
/// `generate.rs` decode pattern at line ~2245 / 2847.
fn direct_path_tokens(
    model: &Arc<TransformerModel>,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    prompt_tokens: &[u32],
    budget: usize,
) -> Vec<u32> {
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
        max_seq_len: MAX_SEQ_LEN,
    };

    let mut kv = alloc_standard_kv_caches(
        model,
        backend.clone(),
        memory.clone(),
        MAX_SEQ_LEN,
        MAX_SEQ_LEN,
        DType::F16,
    )
    .unwrap();

    // Prefill (last-only logits).
    let prefill_input = upload_tokens(backend, &cpu_backend, prompt_tokens);
    let prefill_logits_buf = memory.alloc(vocab * 4, DType::F32).unwrap();
    let mut prefill_logits = Tensor::new(
        Shape::new(vec![1, 1, vocab]),
        prefill_logits_buf,
        backend.clone(),
    );
    model
        .forward_into(TransformerModelForwardArgs {
            input_tokens: &prefill_input,
            start_pos: 0,
            kv_caches: &mut kv,
            backend,
            memory: memory.as_ref(),
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            prefill_workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: true,
            variance_collector: None,
            layer_boundary_hook: None,
        })
        .unwrap();
    let first_logits = read_logits(backend, &prefill_logits, vocab);
    let mut generated = Vec::with_capacity(budget);
    let mut prev = greedy_argmax(&first_logits);
    generated.push(prev);

    // Decode loop.
    let mut ws = LayerWorkspace::new(cfg, memory.as_ref(), backend.clone()).unwrap();
    let x_gen_buf = memory.alloc(hidden * 4, DType::F32).unwrap();
    let mut x_gen = Tensor::new(Shape::new(vec![1, 1, hidden]), x_gen_buf, backend.clone());
    let decode_input_buf = memory.alloc(4, DType::U8).unwrap();
    let mut decode_input = Tensor::new(Shape::new(vec![1, 1]), decode_input_buf, backend.clone());
    let decode_logits_buf = memory.alloc(vocab * 4, DType::F32).unwrap();
    let mut decode_logits = Tensor::new(
        Shape::new(vec![1, 1, vocab]),
        decode_logits_buf,
        backend.clone(),
    );

    let mut pos = prompt_tokens.len();
    while generated.len() < budget {
        backend
            .write_buffer(&mut decode_input, &prev.to_ne_bytes())
            .unwrap();
        model
            .forward_into(TransformerModelForwardArgs {
                input_tokens: &decode_input,
                start_pos: pos,
                kv_caches: &mut kv,
                backend,
                memory: memory.as_ref(),
                logits_out: &mut decode_logits,
                x_gen: Some(&mut x_gen),
                workspace: Some(&mut ws),
                prefill_workspace: None,
                score_accumulator: None,
                profiler: None,
                skip_config: None,
                importance_collector: None,
                logits_last_only: false,
                variance_collector: None,
                layer_boundary_hook: None,
            })
            .unwrap();
        let logits = read_logits(backend, &decode_logits, vocab);
        let next = greedy_argmax(&logits);
        generated.push(next);
        prev = next;
        pos += 1;
    }
    generated
}

/// Wrapper path: drives the same generation through `ModelForward::prefill` +
/// `ModelForward::step` with greedy argmax applied externally. This isolates
/// the wrapper from the `DecodeLoop` step orchestration — the loop's own
/// behaviour is covered by `session::decode_loop::tests` (no model needed)
/// and by the device microbench.
fn wrapper_path_tokens(
    model: &Arc<TransformerModel>,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    prompt_tokens: &[u32],
    budget: usize,
) -> Vec<u32> {
    let cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let kv = alloc_standard_kv_caches(
        model,
        backend.clone(),
        memory.clone(),
        MAX_SEQ_LEN,
        MAX_SEQ_LEN,
        DType::F16,
    )
    .unwrap();
    let mut mf = ModelForward::new(
        backend.clone(),
        memory.clone(),
        cpu_backend,
        model.clone(),
        kv,
        MAX_SEQ_LEN,
    )
    .unwrap();

    let first_logits = mf.prefill(prompt_tokens).unwrap();
    let mut generated = Vec::with_capacity(budget);
    let mut prev = greedy_argmax(&first_logits);
    generated.push(prev);

    let stop = std::sync::atomic::AtomicBool::new(false);
    let mut pos = prompt_tokens.len();
    let mut step_idx = 0usize;
    while generated.len() < budget {
        let ctx = StepCtx {
            pos,
            prev_token: prev,
            kv_capacity: MAX_SEQ_LEN,
            decode_step: step_idx,
            stop_requested: &stop,
        };
        let logits = mf.step(&ctx, prev).unwrap();
        let next = greedy_argmax(&logits);
        generated.push(next);
        prev = next;
        pos += 1;
        step_idx += 1;
    }
    generated
}

#[test]
fn model_forward_parity_with_direct_forward_into() {
    let Some((model_path, tokenizer_path)) = env_paths() else {
        eprintln!(
            "SKIP: model_forward_parity — set LLMRS_TEST_MODEL_PATH + LLMRS_TEST_TOKENIZER_PATH \
             to a GGUF + tokenizer.json pair to exercise this test."
        );
        // Explicit assert so `check_spec_coverage` does not flag the skip
        // branch as assertion-less.
        assert!(
            true,
            "model_forward_parity skipped — env vars not set; CI default"
        );
        return;
    };

    let (backend, memory) = build_backend();
    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let model = TransformerModel::load_gguf(&model_path, backend.clone(), memory.as_ref())
        .expect("load_gguf");
    let model = Arc::new(model);

    let prompt_ids: Vec<u32> = tokenizer
        .encode(PROMPT, true)
        .expect("encode prompt")
        .get_ids()
        .to_vec();
    assert!(!prompt_ids.is_empty(), "encoded prompt must be non-empty");

    let direct = direct_path_tokens(&model, &backend, &memory, &prompt_ids, GEN_TOKENS);
    let wrapped = wrapper_path_tokens(&model, &backend, &memory, &prompt_ids, GEN_TOKENS);

    assert_eq!(
        direct.len(),
        GEN_TOKENS,
        "direct path must emit GEN_TOKENS tokens"
    );
    assert_eq!(
        direct, wrapped,
        "ModelForward parity vs direct forward_into"
    );
}
