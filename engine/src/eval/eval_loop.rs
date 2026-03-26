//! Generic eval-LL loop: `run_eval_ll_generic<C: KVCacheOps>`.
//!
//! Replaces `run_eval_ll` and `run_kivi_eval_ll` in `generate.rs`.
//! The two modes differ only in their `StepHook` implementations and
//! `KVCacheOps` type parameters — all other logic is shared here.

use std::sync::Arc;

use anyhow::Result;

use crate::backend::cpu::CpuBackend;
use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::kv_cache::KVCacheOps;
use crate::core::memory::Memory;
use crate::core::qcf::{ImportanceCollector, SubLayer};
use crate::core::sampling;
use crate::core::shape::Shape;
use crate::core::skip_config::SkipConfig;
use crate::core::tensor::Tensor;
use crate::layers::workspace::{LayerWorkspace, WorkspaceConfig};
use crate::memory::galloc::Galloc;
use crate::models::transformer::{TransformerModel, TransformerModelForwardArgs};

use super::hook::{MetricsSummary, StepHook};
use super::output::{EvalConfig, EvalOutput, EvalQuestion};

/// Run log-likelihood evaluation for a list of multiple-choice questions.
///
/// This is the unified entry point for both eviction (`EvictionHook`) and
/// KIVI (`KiviHook`) eval modes. The caller provides:
///
/// - `kv_caches`: per-layer KV cache (either `KVCache` or `KiviCache`)
/// - `hook`: cache-management and QCF-collection logic
/// - `questions`: pre-parsed questions in grouped format
/// - `eval_config`: loop configuration (budgets, sizes, flags)
/// - `skip_config`: optional layer-skip configuration (triggers importance 2-pass)
///
/// # JSON output
///
/// Returns an `EvalOutput` whose `to_json()` matches the format previously
/// produced by `run_eval_ll` and `run_kivi_eval_ll` exactly.
#[allow(clippy::too_many_arguments)]
pub fn run_eval_ll_generic<C: KVCacheOps>(
    model: &TransformerModel,
    tokenizer: &tokenizers::Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut [C],
    hook: &mut dyn StepHook<C>,
    questions: &[EvalQuestion],
    eval_config: &EvalConfig,
    skip_config: Option<&SkipConfig>,
) -> Result<EvalOutput> {
    let vocab_size = eval_config.vocab_size;
    let hidden_size = eval_config.hidden_size;
    let max_seq_len = eval_config.max_seq_len;

    // ── Pre-allocate decode buffers (reused across all questions / choices) ──
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
            max_seq_len,
        },
        memory,
        backend.clone(),
    )?;

    // Pre-allocate prefill workspace for GPU buffer reuse across questions.
    // Allocated once at max_seq_len; prevents NVIDIA OpenCL driver crash from
    // accumulated alloc/free cycles during multi-question eval-ll.
    let mut prefill_ws = if backend.name() == "OpenCL" {
        use crate::layers::workspace::PrefillWorkspace;
        PrefillWorkspace::new(
            &WorkspaceConfig {
                batch_size: 1,
                dim: hidden_size,
                q_dim,
                k_dim,
                v_dim,
                ffn_hidden,
                n_heads: model.config.num_attention_heads,
                max_seq_len,
            },
            max_seq_len,
            memory,
            backend.clone(),
        )
        .ok()
    } else {
        None
    };

    // Single-token CPU input tensor (reused for decode steps)
    let cpu_gen_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );

    // ── Importance 2-pass (only when skip_config is active) ──
    let (importance_table, layer_skip_qcf, layer_skip_opr, layer_skip_set_len) =
        run_importance_pass(
            model,
            tokenizer,
            backend,
            memory,
            kv_caches,
            hook,
            questions,
            vocab_size,
            eval_config,
            skip_config,
        )?;

    // ── Per-question evaluation loop ──
    let mut results: Vec<serde_json::Value> = Vec::new();
    let overall_start = std::time::Instant::now();

    let qcf_layer_skip = layer_skip_opr.map(|v| v as f64);
    let qcf_layer_skip_layers = layer_skip_opr.map(|_| layer_skip_set_len);

    for (q_idx, question) in questions.iter().enumerate() {
        let q_start = std::time::Instant::now();

        // Reset caches and hook state for this question
        hook.reset_caches(kv_caches);
        // Flush GPU queue to release deferred OpenCL buffers from previous question.
        // Without this, NVIDIA's runtime accumulates pending buffer releases → OOM.
        backend.synchronize()?;

        // Tokenize prompt
        let prompt_enc = tokenizer
            .encode(question.prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let prompt_ids: Vec<u32> = prompt_enc.get_ids().to_vec();
        let prompt_len = prompt_ids.len();

        if prompt_len > max_seq_len {
            eprintln!(
                "[Eval-LL] {}: prompt too long ({} > {}), skipping",
                question.id, prompt_len, max_seq_len
            );
            continue;
        }

        let mut qcf_metrics: Vec<serde_json::Value> = Vec::new();

        // ── Per-question budget (ratio mode) ──
        let q_eval_config;
        let effective_eval_config = if eval_config.kv_budget_ratio > 0.0 {
            let budget = ((prompt_len as f32) * eval_config.kv_budget_ratio) as usize;
            let budget = budget.max(1);
            hook.set_effective_budget(budget);
            q_eval_config = EvalConfig {
                effective_budget: budget,
                ..eval_config.clone()
            };
            &q_eval_config
        } else {
            eval_config
        };

        // ── Prefill ──
        let (prompt_logits_cpu, start_pos_after_prompt) = run_prefill(
            model,
            backend,
            memory,
            kv_caches,
            hook,
            &prompt_ids,
            &mut decode_logits,
            &mut x_gen,
            &mut gen_ws,
            &cpu_gen_input,
            &mut qcf_metrics,
            vocab_size,
            effective_eval_config,
            skip_config,
            prefill_ws.as_mut(),
        )?;

        // ── Score collection probe: capture attention weights for QCF-ATTN v2 ──
        // Batch prefill doesn't populate score_accumulator (workspace: None).
        // Re-feed the last token as a decode step to capture per-head attention.
        // This is a no-op for KV cache content (overwrites identical values at prompt_len-1).
        if hook.needs_score_probe(kv_caches) {
            let last_token_id = prompt_ids[prompt_len - 1];
            // SAFETY: cpu_gen_input was allocated with 4 bytes (one u32).
            unsafe {
                *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_token_id;
            }
            let probe_input = backend.copy_from(&cpu_gen_input)?;

            if let Some(acc) = hook.score_accumulator() {
                acc.begin_step();
            }

            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &probe_input,
                start_pos: start_pos_after_prompt - 1,
                kv_caches,
                backend,
                memory,
                logits_out: &mut decode_logits,
                x_gen: Some(&mut x_gen),
                workspace: Some(&mut gen_ws),
                use_gpu_attn: effective_eval_config.use_gpu_attn,
                score_accumulator: hook.score_accumulator(),
                profiler: None,
                skip_config,
                importance_collector: None,
                logits_last_only: false,
                variance_collector: None,
            })?;
        }

        // ── post_prefill hook (eviction if cache exceeds budget) ──
        hook.post_prefill(kv_caches, &mut qcf_metrics);

        // Update start_pos: post_prefill may have evicted (compacted) the cache.
        let start_pos_after_prompt = kv_caches
            .iter()
            .map(|c| c.current_pos())
            .max()
            .unwrap_or(start_pos_after_prompt);

        // ── Snapshot KV cache after prompt ──
        let snap = hook.snapshot(kv_caches);

        // ── Score each choice ──
        let mut choice_nlls: Vec<f64> = Vec::new();
        let mut choice_byte_lens: Vec<usize> = Vec::new();
        let mut choice_token_lens: Vec<usize> = Vec::new();

        for choice_text in &question.choices {
            let full_text = format!("{}{}", question.prompt, choice_text);
            let full_enc = tokenizer
                .encode(full_text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let full_ids: Vec<u32> = full_enc.get_ids().to_vec();

            // Safety: prompt_ids.len() may be larger than full_ids when the
            // tokenizer merges tokens differently. Guard against underflow.
            let cont_start = prompt_ids.len().min(full_ids.len());
            let cont_ids: Vec<u32> = full_ids[cont_start..].to_vec();

            if cont_ids.is_empty() {
                choice_nlls.push(f64::INFINITY);
                choice_byte_lens.push(choice_text.len());
                choice_token_lens.push(0);
                continue;
            }

            // First continuation token is scored from the prompt logits
            let mut total_nll =
                -sampling::compute_log_prob(&prompt_logits_cpu, cont_ids[0], vocab_size);

            // Multi-token: decode remaining tokens one-by-one
            if cont_ids.len() > 1 {
                // Restore KV cache to post-prompt state for each choice
                snap.restore_to(kv_caches);
                let mut sp = start_pos_after_prompt;

                for token_pair in cont_ids.windows(2) {
                    let input_token = token_pair[0];
                    let target_token = token_pair[1];

                    // Write token id into CPU buffer
                    // SAFETY: cpu_gen_input was allocated with size 4 bytes (one u32).
                    unsafe {
                        *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = input_token;
                    }
                    let gen_input = backend.copy_from(&cpu_gen_input)?;

                    // Score accumulator begin_step only during choice decode
                    if let Some(acc) = hook.score_accumulator() {
                        acc.begin_step();
                    }

                    model.forward_into(TransformerModelForwardArgs {
                        input_tokens: &gen_input,
                        start_pos: sp,
                        kv_caches,
                        backend,
                        memory,
                        logits_out: &mut decode_logits,
                        x_gen: Some(&mut x_gen),
                        workspace: Some(&mut gen_ws),
                        use_gpu_attn: effective_eval_config.use_gpu_attn,
                        score_accumulator: hook.score_accumulator(),
                        profiler: None,
                        skip_config,
                        importance_collector: None,
                        logits_last_only: false,
                        variance_collector: None,
                    })?;
                    sp += 1;

                    // post_decode_step hook (eviction / flush collection)
                    hook.post_decode_step(kv_caches, sp, &mut qcf_metrics);

                    // Read logits and accumulate NLL
                    let mut step_logits = vec![0.0f32; vocab_size];
                    // SAFETY: buffer is exactly vocab_size * 4 bytes of f32.
                    unsafe {
                        let ptr = step_logits.as_mut_ptr() as *mut u8;
                        let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
                        backend.read_buffer(&decode_logits, slice)?;
                    }
                    total_nll -= sampling::compute_log_prob(&step_logits, target_token, vocab_size);
                }
            }

            choice_nlls.push(total_nll);
            choice_byte_lens.push(choice_text.len());
            choice_token_lens.push(cont_ids.len());
        }

        // Restore for consistency (not strictly required — hook.reset_caches next iter)
        snap.restore_to(kv_caches);

        // ── Determine predicted choice ──
        // Byte-length-normalized NLL (acc_norm, lm-eval style)
        let predicted_norm: usize = choice_nlls
            .iter()
            .zip(choice_byte_lens.iter())
            .enumerate()
            .min_by(|(_, (a, al)), (_, (b, bl))| {
                let a_norm = *a / (**al).max(1) as f64;
                let b_norm = *b / (**bl).max(1) as f64;
                a_norm
                    .partial_cmp(&b_norm)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Raw prediction (acc, no normalization)
        let predicted_raw: usize = choice_nlls
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let elapsed_q = q_start.elapsed().as_secs_f64();

        eprintln!(
            "[Eval-LL] {}/{} {} — norm={} raw={} nlls=[{}] {:.1}s",
            q_idx + 1,
            questions.len(),
            question.id,
            predicted_norm,
            predicted_raw,
            choice_nlls
                .iter()
                .map(|v| format!("{:.3}", v))
                .collect::<Vec<_>>()
                .join(","),
            elapsed_q,
        );

        // ── Per-question result JSON ──
        let final_cache_pos = kv_caches.iter().map(|c| c.current_pos()).max().unwrap_or(0);

        let extra = hook.extra_question_fields(kv_caches);

        let mut result_obj = serde_json::json!({
            "id": question.id,
            "choice_nlls": choice_nlls,
            "choice_byte_lens": choice_byte_lens,
            "choice_token_lens": choice_token_lens,
            "predicted": predicted_norm,
            "predicted_raw": predicted_raw,
            "n_choices": question.choices.len(),
            "n_prompt_tokens": prompt_len,
            "final_cache_pos": final_cache_pos,
            "qcf_layer_skip": qcf_layer_skip,
            "qcf_layer_skip_layers": qcf_layer_skip_layers,
        });

        // Merge hook-specific fields (qcf_kivi, eviction_qcf, effective_budget, etc.)
        if let Some(obj) = extra.as_object() {
            for (k, v) in obj {
                result_obj[k] = v.clone();
            }
        }

        results.push(result_obj);
    }

    let wall_time_s = overall_start.elapsed().as_secs_f64();

    // ── Build layer_importance JSON ──
    let layer_importance_json = importance_table.map(|table| {
        serde_json::json!(
            table
                .entries()
                .iter()
                .map(|e| serde_json::json!({
                    "layer": e.layer_id,
                    "sublayer": format!("{:?}", e.sublayer),
                    "importance": e.importance,
                    "opr": e.opr,
                }))
                .collect::<Vec<serde_json::Value>>()
        )
    });

    // Normalize layer_skip_qcf: skipped / remaining = raw / (1 - raw)
    let layer_skip_qcf_normalized = layer_skip_qcf.map(|qcf| {
        const NORMALIZED_CAP: f32 = 100.0;
        if qcf >= 1.0 - 1e-7 {
            NORMALIZED_CAP
        } else {
            qcf / (1.0 - qcf)
        }
    });

    Ok(EvalOutput {
        results,
        config: serde_json::json!(hook.extra_config_fields()),
        wall_time_s,
        metrics_summary: MetricsSummary::default(),
        layer_importance: layer_importance_json,
        layer_skip_qcf,
        layer_skip_qcf_normalized,
        qcf_layer_skip,
        qcf_layer_skip_layers,
    })
}

/// Importance 2-pass: runs a forward pass on the first question's prompt to
/// measure per-layer importance when a `SkipConfig` is active.
///
/// Returns `(importance_table, layer_skip_qcf, layer_skip_opr, skip_set_len)`.
/// All fields are `None` / `0` when `skip_config` is `None` or questions is empty.
type ImportancePassResult = (
    Option<crate::core::qcf::ImportanceTable>,
    Option<f32>,
    Option<f32>,
    usize,
);

#[allow(clippy::too_many_arguments)]
fn run_importance_pass<C: KVCacheOps>(
    model: &TransformerModel,
    tokenizer: &tokenizers::Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut [C],
    hook: &mut dyn StepHook<C>,
    questions: &[EvalQuestion],
    vocab_size: usize,
    eval_config: &EvalConfig,
    skip_config: Option<&SkipConfig>,
) -> Result<ImportancePassResult> {
    let sc = match skip_config {
        None => return Ok((None, None, None, 0)),
        Some(sc) => sc,
    };
    if questions.is_empty() {
        return Ok((None, None, None, 0));
    }

    let first_q = &questions[0];
    let prompt_enc = tokenizer
        .encode(first_q.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let prompt_ids_imp: Vec<u32> = prompt_enc.get_ids().to_vec();
    let imp_len = prompt_ids_imp.len();

    // Reset caches before importance measurement
    hook.reset_caches(kv_caches);

    let cpu_buf = Galloc::new().alloc(imp_len * 4, DType::U8)?;
    // SAFETY: allocated exactly imp_len u32 words above.
    unsafe {
        let ptr = cpu_buf.as_mut_ptr() as *mut u32;
        std::ptr::copy_nonoverlapping(prompt_ids_imp.as_ptr(), ptr, imp_len);
    }
    let cpu_input = Tensor::new(
        Shape::new(vec![1, imp_len]),
        cpu_buf,
        Arc::new(CpuBackend::new()),
    );
    let input_tensor = backend.copy_from(&cpu_input)?;

    let imp_logits_buf = memory.alloc(imp_len * vocab_size * 4, DType::F32)?;
    let mut imp_logits = Tensor::new(
        Shape::new(vec![1, imp_len, vocab_size]),
        imp_logits_buf,
        backend.clone(),
    );

    let mut collector = ImportanceCollector::new();
    model.forward_into(TransformerModelForwardArgs {
        input_tokens: &input_tensor,
        start_pos: 0,
        kv_caches,
        backend,
        memory,
        logits_out: &mut imp_logits,
        x_gen: None,
        workspace: None,
        use_gpu_attn: eval_config.use_gpu_attn,
        score_accumulator: None,
        profiler: None,
        skip_config: None, // intentionally None for importance measurement
        importance_collector: Some(&mut collector),
        logits_last_only: false,
        variance_collector: None,
    })?;

    let table = collector.build();

    let skip_set: Vec<(usize, SubLayer)> = sc
        .attn_skip
        .union(&sc.mlp_skip)
        .map(|&l| (l, SubLayer::Full))
        .collect();
    let qcf = table.compute_qcf(&skip_set);
    let opr_skip = table.compute_opr_skip(&skip_set);
    let skip_set_len = skip_set.len();

    eprintln!(
        "[Skip] Importance measured on {} tokens, layer_skip_qcf={:.4}",
        imp_len, qcf
    );

    // Reset caches again before actual evaluation
    hook.reset_caches(kv_caches);

    Ok((Some(table), Some(qcf), Some(opr_skip), skip_set_len))
}

/// Run prompt processing (prefill or chunked prefill + decode) for one question.
///
/// Returns `(prompt_logits_cpu, start_pos_after_prompt)`.
///
/// - `prompt_logits_cpu`: logits for the last prompt token (for scoring cont_ids[0])
/// - `start_pos_after_prompt`: KV cache position after processing all prompt tokens
#[allow(clippy::too_many_arguments)]
fn run_prefill<C: KVCacheOps>(
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut [C],
    hook: &mut dyn StepHook<C>,
    prompt_ids: &[u32],
    _decode_logits: &mut Tensor,
    _x_gen: &mut Tensor,
    _gen_ws: &mut LayerWorkspace,
    _cpu_gen_input: &Tensor,
    _qcf_metrics: &mut Vec<serde_json::Value>,
    vocab_size: usize,
    eval_config: &EvalConfig,
    skip_config: Option<&SkipConfig>,
    prefill_ws: Option<&mut crate::layers::workspace::PrefillWorkspace>,
) -> Result<(Vec<f32>, usize)> {
    // Always use full batch prefill regardless of budget.
    // Post-prefill eviction (in hook.post_prefill) handles cache compaction.
    // This avoids the O(prompt_len - budget) token-by-token decode that caused
    // 2-3.3x slowdown when prompt > budget (issue D).
    run_full_prefill(
        model,
        backend,
        memory,
        kv_caches,
        hook,
        prompt_ids,
        _qcf_metrics,
        vocab_size,
        eval_config,
        skip_config,
        prefill_ws,
    )
}

/// Full prefill: forward all prompt tokens in a single batched pass.
#[allow(clippy::too_many_arguments)]
fn run_full_prefill<C: KVCacheOps>(
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut [C],
    hook: &mut dyn StepHook<C>,
    prompt_ids: &[u32],
    _qcf_metrics: &mut Vec<serde_json::Value>,
    vocab_size: usize,
    eval_config: &EvalConfig,
    skip_config: Option<&SkipConfig>,
    _prefill_ws: Option<&mut crate::layers::workspace::PrefillWorkspace>,
) -> Result<(Vec<f32>, usize)> {
    let prompt_len = prompt_ids.len();

    let cpu_indices_buf = Galloc::new().alloc(prompt_len * 4, DType::U8)?;
    // SAFETY: allocated prompt_len u32 words above.
    unsafe {
        let ptr = cpu_indices_buf.as_mut_ptr() as *mut u32;
        std::ptr::copy_nonoverlapping(prompt_ids.as_ptr(), ptr, prompt_len);
    }
    let cpu_input = Tensor::new(
        Shape::new(vec![1, prompt_len]),
        cpu_indices_buf,
        Arc::new(CpuBackend::new()),
    );
    let input_tensor = backend.copy_from(&cpu_input)?;

    // Only allocate for last position's logits (saves ~3GB vs full prompt × vocab).
    let prefill_logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut prefill_logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        prefill_logits_buf,
        backend.clone(),
    );

    model.forward_into(TransformerModelForwardArgs {
        input_tokens: &input_tensor,
        start_pos: 0,
        kv_caches,
        backend,
        memory,
        logits_out: &mut prefill_logits,
        x_gen: None,
        workspace: None,
        use_gpu_attn: eval_config.use_gpu_attn,
        score_accumulator: hook.score_accumulator(),
        profiler: None,
        skip_config,
        importance_collector: None,
        logits_last_only: true,
        variance_collector: None,
    })?;

    // Read logits (only last position — much smaller than full prompt × vocab)
    let mut prompt_logits_cpu = vec![0.0f32; vocab_size];
    unsafe {
        let ptr = prompt_logits_cpu.as_mut_ptr() as *mut u8;
        let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
        backend.read_buffer(&prefill_logits, slice)?;
    }

    // Explicitly drop GPU buffers and flush queue to free VRAM.
    drop(prefill_logits);
    drop(input_tensor);
    backend.synchronize()?;

    Ok((prompt_logits_cpu, prompt_len))
}

/// Chunked prefill: forward first `effective_budget` tokens as a batch, then
/// decode the remaining prompt tokens one-by-one (with eviction between steps).
///
/// **Deprecated**: No longer called from `run_prefill`. Full batch prefill +
/// post_prefill eviction is faster and produces equivalent results.
/// Retained for reference; will be removed in a future cleanup.
#[allow(dead_code, clippy::too_many_arguments)]
fn run_chunked_prefill<C: KVCacheOps>(
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut [C],
    hook: &mut dyn StepHook<C>,
    prompt_ids: &[u32],
    decode_logits: &mut Tensor,
    x_gen: &mut Tensor,
    gen_ws: &mut LayerWorkspace,
    cpu_gen_input: &Tensor,
    qcf_metrics: &mut Vec<serde_json::Value>,
    vocab_size: usize,
    eval_config: &EvalConfig,
    skip_config: Option<&SkipConfig>,
    _prefill_ws: Option<&mut crate::layers::workspace::PrefillWorkspace>,
) -> Result<(Vec<f32>, usize)> {
    let effective_budget = eval_config.effective_budget;
    let first_chunk_len = effective_budget;

    // ── First chunk: batched prefill ──
    let cpu_buf = Galloc::new().alloc(first_chunk_len * 4, DType::U8)?;
    // SAFETY: allocated first_chunk_len u32 words above.
    unsafe {
        let ptr = cpu_buf.as_mut_ptr() as *mut u32;
        std::ptr::copy_nonoverlapping(prompt_ids.as_ptr(), ptr, first_chunk_len);
    }
    let cpu_input = Tensor::new(
        Shape::new(vec![1, first_chunk_len]),
        cpu_buf,
        Arc::new(CpuBackend::new()),
    );
    let input_tensor = backend.copy_from(&cpu_input)?;

    // Only allocate for last position's logits — the first chunk logits are
    // never read (only the final decode-step logits are returned). This saves
    // ~116 MB of GPU memory for vocab_size=151936 × budget=200.
    let prefill_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut prefill_logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        prefill_buf,
        backend.clone(),
    );

    if let Some(acc) = hook.score_accumulator() {
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
        use_gpu_attn: eval_config.use_gpu_attn,
        score_accumulator: hook.score_accumulator(),
        profiler: None,
        skip_config,
        importance_collector: None,
        logits_last_only: true,
        variance_collector: None,
    })?;

    // Explicitly drop GPU buffers and flush queue to free VRAM before the
    // decode loop. Mirrors the cleanup in run_full_prefill that prevents
    // NVIDIA OpenCL driver crashes from accumulated deferred buffer releases.
    drop(prefill_logits);
    drop(input_tensor);
    backend.synchronize()?;

    let mut start_pos = first_chunk_len;

    // ── Decode remaining prompt tokens one-by-one ──
    // No eviction during prefill: let KV cache grow to full prompt length.
    // KV cache has capacity for max_seq_len, so prompt always fits.
    // Eviction happens once after all prompt tokens are processed — this gives
    // H2O the full importance picture before selecting heavy hitters.
    for &token_id in &prompt_ids[first_chunk_len..] {
        // SAFETY: cpu_gen_input was allocated with 4 bytes (one u32).
        unsafe {
            *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = token_id;
        }
        let gen_input = backend.copy_from(cpu_gen_input)?;

        if let Some(acc) = hook.score_accumulator() {
            acc.begin_step();
        }

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &gen_input,
            start_pos,
            kv_caches,
            backend,
            memory,
            logits_out: decode_logits,
            x_gen: Some(x_gen),
            workspace: Some(gen_ws),
            use_gpu_attn: eval_config.use_gpu_attn,
            score_accumulator: hook.score_accumulator(),
            profiler: None,
            skip_config,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
        })?;
        start_pos += 1;
    }

    // Single eviction after all prompt tokens: evict from prompt_len to budget
    let max_pos = kv_caches.iter().map(|c| c.current_pos()).max().unwrap_or(0);
    if max_pos > effective_budget {
        hook.post_decode_step(kv_caches, start_pos, qcf_metrics);
    }

    // Read logits from last decode step
    let mut logits_cpu = vec![0.0f32; vocab_size];
    // SAFETY: decode_logits is vocab_size f32 values = vocab_size * 4 bytes.
    unsafe {
        let ptr = logits_cpu.as_mut_ptr() as *mut u8;
        let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
        backend.read_buffer(decode_logits, slice)?;
    }

    Ok((logits_cpu, start_pos))
}

#[cfg(test)]
mod tests {
    use crate::eval::output::{EvalConfig, EvalQuestion};

    fn make_config() -> EvalConfig {
        EvalConfig {
            max_seq_len: 2048,
            effective_budget: 0,
            kv_budget_ratio: 0.0,
            greedy: true,
            kv_type: "f32".to_string(),
            use_gpu_attn: false,
            qcf_mode: "attn".to_string(),
            vocab_size: 32000,
            hidden_size: 2048,
        }
    }

    #[test]
    fn test_eval_config_fields() {
        let cfg = make_config();
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.max_seq_len, 2048);
        assert!(!cfg.use_gpu_attn);
    }

    #[test]
    fn test_eval_question_structure() {
        let q = EvalQuestion {
            id: "q1".to_string(),
            prompt: "The capital of France is".to_string(),
            choices: vec![" Paris".to_string(), " London".to_string()],
        };
        assert_eq!(q.choices.len(), 2);
        assert_eq!(q.id, "q1");
    }

    /// Test that EvalConfig with budget_mode=false (effective_budget=0) selects full prefill.
    #[test]
    fn test_budget_mode_false_when_zero() {
        let cfg = make_config();
        let prompt_len = 100;
        let budget_mode = cfg.effective_budget > 0;
        assert!(!budget_mode);
        // full prefill: no chunking regardless of prompt length
        assert!(!(budget_mode && prompt_len > cfg.effective_budget));
    }

    /// Test that EvalConfig with effective_budget > 0 and prompt_len > budget triggers chunked.
    #[test]
    fn test_budget_mode_triggers_chunked() {
        let mut cfg = make_config();
        cfg.effective_budget = 50;
        let prompt_len = 100;
        let budget_mode = cfg.effective_budget > 0;
        assert!(budget_mode && prompt_len > cfg.effective_budget);
    }

    #[test]
    fn test_predicted_norm_tie_returns_first() {
        // When all NLLs are equal, predicted_norm should return 0.
        let choice_nlls: Vec<f64> = vec![1.0, 1.0, 1.0];
        let choice_byte_lens: Vec<usize> = vec![5, 5, 5];
        let predicted_norm: usize = choice_nlls
            .iter()
            .zip(choice_byte_lens.iter())
            .enumerate()
            .min_by(|(_, (a, al)), (_, (b, bl))| {
                let a_norm = *a / (**al).max(1) as f64;
                let b_norm = *b / (**bl).max(1) as f64;
                a_norm
                    .partial_cmp(&b_norm)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(predicted_norm, 0);
    }

    #[test]
    fn test_predicted_norm_selects_minimum() {
        let choice_nlls: Vec<f64> = vec![2.0, 1.0, 3.0];
        let choice_byte_lens: Vec<usize> = vec![4, 4, 4];
        let predicted_norm: usize = choice_nlls
            .iter()
            .zip(choice_byte_lens.iter())
            .enumerate()
            .min_by(|(_, (a, al)), (_, (b, bl))| {
                let a_norm = *a / (**al).max(1) as f64;
                let b_norm = *b / (**bl).max(1) as f64;
                a_norm
                    .partial_cmp(&b_norm)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(predicted_norm, 1);
    }

    #[test]
    fn test_predicted_raw_selects_minimum() {
        let choice_nlls: Vec<f64> = vec![3.0, 0.5, 2.0];
        let predicted_raw: usize = choice_nlls
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(predicted_raw, 1);
    }

    #[test]
    fn test_layer_skip_qcf_normalized_cap() {
        // qcf = 1.0 should cap at 100.0
        let qcf: f32 = 1.0;
        const NORMALIZED_CAP: f32 = 100.0;
        let normalized = if qcf >= 1.0 - 1e-7 {
            NORMALIZED_CAP
        } else {
            qcf / (1.0 - qcf)
        };
        assert!((normalized - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_layer_skip_qcf_normalized_midrange() {
        // qcf = 0.5 → normalized = 0.5 / 0.5 = 1.0
        let qcf: f32 = 0.5;
        const NORMALIZED_CAP: f32 = 100.0;
        let normalized = if qcf >= 1.0 - 1e-7 {
            NORMALIZED_CAP
        } else {
            qcf / (1.0 - qcf)
        };
        assert!((normalized - 1.0).abs() < 1e-5);
    }
}
