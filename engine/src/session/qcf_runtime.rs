//! Phase 4-B-1: `session::qcf_runtime` — bin/generate.rs에서 분산된
//! shared helper fn (`run_qcf_warmup_workflow`, `run_layer_swap` +
//! `read_allow_boundary_env`, `QcfWarmupResult` struct)을 lib level로
//! 이동하여 session/eval/, session/ppl/, session/batch/, standard
//! generate 분기에서 공통 호출 가능하게 한다.
//!
//! lift-and-shift: 본문 변경 없음. `crate::X` → `crate::X` 만 적용.

use std::sync::Arc;

use anyhow::Result;

use crate::backend::cpu::CpuBackend;
use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::kv_cache::KVCache;
use crate::core::memory::Memory;
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use crate::memory::galloc::Galloc;
use crate::models::transformer::TransformerModelForwardArgs;

pub struct QcfWarmupResult {
    pub importance: crate::core::qcf::ImportanceTable,
    pub decision: Option<crate::models::weights::SwapDecision>,
    /// Per-layer DP-LLM proxy ε (single-tensor relative `attn_output`).
    /// `Some` only in compare mode.
    pub dpllm_epsilon: Option<Vec<f32>>,
    /// §4 candidate A: per-layer DP-LLM ε summed across 6 attn+MLP tensors.
    pub dpllm_epsilon_multi: Option<Vec<f32>>,
    /// §4 candidate D: per-layer DP-LLM ε without the `‖W·x‖` normalisation
    /// (absolute L2 of the activation difference).
    pub dpllm_epsilon_abs: Option<Vec<f32>>,
    /// §4 candidate E: QCF-style multiplicative composition `ε_v × ε_o`.
    pub dpllm_epsilon_qcf: Option<Vec<f32>>,
    /// §4.2 F4: cascade-aware single output perturbation.
    /// `‖(W_o^F16 − W_o^Q4) · V_out^F16‖_F / ‖W_o^F16 · V_out^F16‖_F`.
    pub direct_attn_f4: Option<Vec<f32>>,
    /// §4.2 F5: direct attention output relative L2 perturbation.
    /// `‖O^F16 − O^Q4‖_F / ‖O^F16‖_F`.
    pub direct_attn_f5: Option<Vec<f32>>,
    /// §4.2 decode-only F5: per-layer F5 evaluated with X = the N decode-step
    /// raws (T = N) only (no prefill X). `Some` only when `--decode-x-steps >
    /// 0` and a secondary GGUF was loaded.
    pub direct_attn_f5_decode_only: Option<Vec<f32>>,
    /// §4.2 prefill+decode F5: per-layer F5 evaluated with X =
    /// concat(prefill raws, decode raws) (T = 256 + N). `Some` only when
    /// `--decode-x-steps > 0` and a secondary GGUF was loaded.
    pub direct_attn_f5_prefill_decode: Option<Vec<f32>>,
}

pub fn run_layer_swap(
    model: &crate::models::transformer::TransformerModel,
    target_layers: &[usize],
    gpu_backend: Option<&Arc<dyn Backend>>,
    cpu_backend: &Arc<dyn Backend>,
    async_dispatcher: Option<&crate::models::weights::AsyncSwapDispatcher>,
    #[cfg(feature = "opencl")] host_ptr_pool: Option<
        Arc<crate::backend::opencl::host_ptr_pool::HostPtrPool>,
    >,
    #[cfg(feature = "cuda-embedded")] layer_pool: Option<
        Arc<crate::models::weights::layer_object_pool::LayerObjectPool>,
    >,
    #[cfg(feature = "cuda-embedded")] mmap_registration: Option<
        Arc<crate::buffer::cuda_mmap_alias_buffer::CudaMmapRegistration>,
    >,
) -> Result<crate::models::weights::SwapReport, crate::models::weights::SwapError> {
    let swap_memory = Galloc::new();
    let swap_backend: Arc<dyn Backend> =
        gpu_backend.cloned().unwrap_or_else(|| cpu_backend.clone());
    // ENG-ALG-228: attach the model's async release worker so Stage (c) enqueues
    // displaced LayerWeights for background drop instead of blocking inline.
    let executor = crate::models::weights::SwapExecutor::new_with_worker(
        DType::Q4_0,
        &model.config,
        swap_backend,
        &swap_memory,
        Arc::clone(&model.release_worker),
    );
    // LISWAP-3 prototype: if a host_ptr pool is supplied, attach it so the
    // AOS materialise path uses the zero-copy slot pool.
    #[cfg(feature = "opencl")]
    let executor = match host_ptr_pool {
        Some(pool) => executor.with_host_ptr_pool(pool),
        None => executor,
    };
    // LISWAP-8 Phase B: attach pre-allocated layer object pool when set.
    #[cfg(feature = "cuda-embedded")]
    let executor = match layer_pool {
        Some(pool) => executor.with_layer_pool(pool),
        None => executor,
    };
    // LISWAP-8 Hammer D: attach mmap registration when set.
    #[cfg(feature = "cuda-embedded")]
    let executor = match mmap_registration {
        Some(reg) => executor.with_mmap_registration(reg),
        None => executor,
    };
    executor.execute_on_slots(
        model.layers.as_slice(),
        model.secondary_mmap.as_ref(),
        &model.ratio_generation,
        target_layers,
        async_dispatcher,
    )
}

/// Re-map weight tensors for CPU access after a weight swap.
///
/// Required when running on GPU with `--secondary-layout aos +
/// --resilience-prealloc-switch`: `SwapExecutor::materialise_tensor` lands an
/// unmapped `UnifiedBuffer` in the new `LayerWeights` snapshot, and the next
/// `switch_hw cpu` directive segfaults on a null host pointer.  Idempotent —
/// already-mapped tensors short-circuit in `map_one`.
#[cfg(feature = "opencl")]

pub fn read_allow_boundary_env() -> bool {
    std::env::var("LLMRS_SWAP_ALLOW_BOUNDARY_LAYERS")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false)
}

pub fn run_qcf_warmup_workflow(
    model: &crate::models::transformer::TransformerModel,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut [KVCache],
    vocab_size: usize,
    warmup_ids: &[u32],
    force_ratio: Option<f32>,
    gpu_backend: Option<&Arc<dyn Backend>>,
    cpu_backend: &Arc<dyn Backend>,
    log_prefix: &str,
    swap_algorithm: crate::models::weights::SwapAlgorithm,
    execute_swap: bool,
    importance_formula: crate::core::qcf::ImportanceFormula,
    importance_three_way: bool,
    swap_only_layers: Option<&[usize]>,
    decode_x_steps: usize,
) -> anyhow::Result<QcfWarmupResult> {
    use crate::core::qcf::ImportanceCollector;
    use crate::models::weights::WeightSwapDecider;

    let actual_warmup_len = warmup_ids.len();
    eprintln!(
        "[QCF-dump]{} warmup prefill: {} tokens (formula={}, three_way={}, decode_x_steps={})",
        log_prefix,
        actual_warmup_len,
        importance_formula.as_str(),
        importance_three_way,
        decode_x_steps,
    );

    // ── Warmup prefill with ImportanceCollector (F16 prefill pass) ────────────
    let mut collector =
        ImportanceCollector::new_with_formula(importance_formula, importance_three_way);
    let last_token_logits_argmax: u32;
    {
        let warmup_buf = Galloc::new().alloc(actual_warmup_len * 4, DType::U8)?;
        unsafe {
            let ptr = warmup_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in warmup_ids.iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_warmup = Tensor::new(
            Shape::new(vec![1, actual_warmup_len]),
            warmup_buf,
            Arc::new(CpuBackend::new()),
        );
        let warmup_input = backend.copy_from(&cpu_warmup)?;

        let warmup_logits_buf = memory.alloc(actual_warmup_len * vocab_size * 4, DType::F32)?;
        let mut warmup_logits = Tensor::new(
            Shape::new(vec![1, actual_warmup_len, vocab_size]),
            warmup_logits_buf,
            backend.clone(),
        );

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &warmup_input,
            start_pos: 0,
            kv_caches,
            backend,
            memory,
            logits_out: &mut warmup_logits,
            x_gen: None,
            workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: Some(&mut collector),
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        backend.synchronize()?;

        // Read the argmax of the last token's logits — first decode-step input.
        last_token_logits_argmax = if decode_x_steps > 0 {
            // Copy the final `[vocab_size]` slice of warmup_logits to host.
            let cpu_back: Arc<dyn Backend> = Arc::new(CpuBackend::new());
            let host_logits = cpu_back.copy_from(&warmup_logits)?;
            let last_offset = (actual_warmup_len - 1) * vocab_size;
            let host_data = unsafe {
                std::slice::from_raw_parts(
                    host_logits.buffer().as_ptr() as *const f32,
                    actual_warmup_len * vocab_size,
                )
            };
            let last_slice = &host_data[last_offset..last_offset + vocab_size];
            let mut best_idx: u32 = 0;
            let mut best_val: f32 = f32::NEG_INFINITY;
            for (i, &v) in last_slice.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i as u32;
                }
            }
            best_idx
        } else {
            0
        };
    }

    // ── Optional decode-X pass (§4.2 EuroSys'27) ──────────────────────────────
    // After prefill, run `decode_x_steps` greedy decode forwards. Capture
    // per-layer hidden state (T = decode_x_steps per layer after concat) in
    // a fresh `DirectAttn` collector. KV cache is reset after this pass.
    let raws_decode_opt: Option<Vec<(Vec<f32>, usize, usize)>> = if decode_x_steps > 0 {
        if model.secondary_mmap.is_none() {
            eprintln!(
                "[QCF-dump]{} decode-x: SKIPPED (no secondary GGUF loaded)",
                log_prefix
            );
            None
        } else {
            eprintln!(
                "[QCF-dump]{} decode-x: running {} greedy decode steps (seed token id {})",
                log_prefix, decode_x_steps, last_token_logits_argmax,
            );

            let mut collector_decode = ImportanceCollector::new_with_formula(
                crate::core::qcf::ImportanceFormula::DirectAttn,
                false,
            );

            let mut next_tok: u32 = last_token_logits_argmax;
            let cpu_back: Arc<dyn Backend> = Arc::new(CpuBackend::new());

            for step in 0..decode_x_steps {
                let decode_buf = Galloc::new().alloc(4, DType::U8)?;
                unsafe {
                    let ptr = decode_buf.as_mut_ptr() as *mut u32;
                    *ptr = next_tok;
                }
                let cpu_decode = Tensor::new(
                    Shape::new(vec![1, 1]),
                    decode_buf,
                    Arc::new(CpuBackend::new()),
                );
                let decode_input = backend.copy_from(&cpu_decode)?;

                let decode_logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
                let mut decode_logits = Tensor::new(
                    Shape::new(vec![1, 1, vocab_size]),
                    decode_logits_buf,
                    backend.clone(),
                );

                model.forward_into(TransformerModelForwardArgs {
                    input_tokens: &decode_input,
                    start_pos: actual_warmup_len + step,
                    kv_caches,
                    backend,
                    memory,
                    logits_out: &mut decode_logits,
                    x_gen: None,
                    workspace: None,
                    score_accumulator: None,
                    profiler: None,
                    skip_config: None,
                    importance_collector: Some(&mut collector_decode),
                    logits_last_only: true,
                    variance_collector: None,
                    prefill_workspace: None,

                    layer_boundary_hook: None,
                })?;
                backend.synchronize()?;

                // argmax for next decode step
                let host_logits = cpu_back.copy_from(&decode_logits)?;
                let host_data = unsafe {
                    std::slice::from_raw_parts(
                        host_logits.buffer().as_ptr() as *const f32,
                        vocab_size,
                    )
                };
                let mut best_idx: u32 = 0;
                let mut best_val: f32 = f32::NEG_INFINITY;
                for (i, &v) in host_data.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best_idx = i as u32;
                    }
                }
                next_tok = best_idx;
            }

            // The collector cached one `[1 × d]` snapshot per layer for *every*
            // decode step (N × n_layers snapshots in chronological order).
            // Rearrange into a per-layer concat: layer i → T = decode_x_steps.
            // build_with_raws returns (table, x_means, raws_per_layer); raws is
            // `Vec<(Vec<f32>, seq_len=1, dim)>` of length N * n_layers.
            let (_table, _xm, raws_chrono) = collector_decode.build_with_raws();
            let n_layers = kv_caches.len();
            let mut per_layer_concat: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(n_layers);
            if raws_chrono.len() == decode_x_steps * n_layers && n_layers > 0 {
                let d = raws_chrono[0].2;
                for li in 0..n_layers {
                    let mut buf: Vec<f32> = Vec::with_capacity(decode_x_steps * d);
                    for step in 0..decode_x_steps {
                        let idx = step * n_layers + li;
                        let (data, t, _d) = &raws_chrono[idx];
                        debug_assert_eq!(*t, 1);
                        buf.extend_from_slice(&data[..d.min(data.len())]);
                    }
                    per_layer_concat.push((buf, decode_x_steps, d));
                }
            } else {
                eprintln!(
                    "[QCF-dump]{} decode-x: WARN raws layout mismatch (chrono len={}, expected {} × {})",
                    log_prefix,
                    raws_chrono.len(),
                    decode_x_steps,
                    n_layers
                );
            }

            // Reset KV cache so the regular flow starts from a clean prefill state.
            for kv in kv_caches.iter_mut() {
                kv.current_pos = 0;
            }

            Some(per_layer_concat)
        }
    } else {
        None
    };

    // ── Build ImportanceTable (+ optional DP-LLM ε variants) + reset KV cache ────
    let direct_attn_primary = matches!(
        importance_formula,
        crate::core::qcf::ImportanceFormula::DirectAttn
    );
    let cache_raw = importance_three_way || direct_attn_primary;
    let mut direct_attn_f5_decode_only: Option<Vec<f32>> = None;
    let mut direct_attn_f5_prefill_decode: Option<Vec<f32>> = None;
    let (
        imp_table,
        dpllm_epsilon,
        dpllm_epsilon_multi,
        dpllm_epsilon_abs,
        dpllm_epsilon_qcf,
        direct_attn_f4,
        direct_attn_f5,
    ) = if cache_raw {
        let (table, x_means, raws) = collector.build_with_raws();
        let sec_opt = model.secondary_mmap.as_ref();
        let eps_single = if importance_three_way {
            sec_opt.map(|sec| {
                crate::models::weights::noise_table::compute_input_aware_epsilon(
                    &model.layers,
                    sec,
                    &x_means,
                )
            })
        } else {
            None
        };
        let eps_multi = if importance_three_way {
            sec_opt.map(|sec| {
                crate::models::weights::noise_table::compute_input_aware_epsilon_multitensor(
                    &model.layers,
                    sec,
                    &x_means,
                )
            })
        } else {
            None
        };
        let eps_abs = if importance_three_way {
            sec_opt.map(|sec| {
                crate::models::weights::noise_table::compute_input_aware_epsilon_absolute(
                    &model.layers,
                    sec,
                    &x_means,
                )
            })
        } else {
            None
        };
        let eps_qcf = if importance_three_way {
            sec_opt.map(|sec| {
                crate::models::weights::noise_table::compute_input_aware_epsilon_qcf(
                    &model.layers,
                    sec,
                    &x_means,
                )
            })
        } else {
            None
        };
        // Cascade attention F4 + F5 (compute when raws are available, regardless
        // of whether primary is DirectAttn or 3-way compare).
        let (f4, f5) = if let Some(sec) = sec_opt {
            let n_heads = model.config.num_attention_heads;
            let n_kv_heads = model.config.num_key_value_heads;
            let d_head = model.config.head_dim;
            let pairs = crate::models::weights::noise_table::compute_cascade_attn_perturbation(
                &model.layers,
                sec,
                &raws,
                n_heads,
                n_kv_heads,
                d_head,
            );
            let f4_vec: Vec<f32> = pairs.iter().map(|(a, _)| *a).collect();
            let f5_vec: Vec<f32> = pairs.iter().map(|(_, b)| *b).collect();
            (Some(f4_vec), Some(f5_vec))
        } else {
            (None, None)
        };
        // §4.2 decode-X F5: compute with decode-only raws (T = N) AND
        // prefill+decode concat raws (T = 256 + N).
        if let (Some(sec), Some(raws_dec)) = (sec_opt, raws_decode_opt.as_ref()) {
            let n_heads = model.config.num_attention_heads;
            let n_kv_heads = model.config.num_key_value_heads;
            let d_head = model.config.head_dim;

            // (1) decode-only
            let pairs_dec = crate::models::weights::noise_table::compute_cascade_attn_perturbation(
                &model.layers,
                sec,
                raws_dec,
                n_heads,
                n_kv_heads,
                d_head,
            );
            direct_attn_f5_decode_only =
                Some(pairs_dec.iter().map(|(_, b)| *b).collect::<Vec<f32>>());

            // (2) prefill + decode concat: raws[i] = concat(prefill_raws[i], decode_raws[i])
            // along the T dimension.
            if raws.len() == raws_dec.len() {
                let mut raws_merged: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(raws.len());
                for (p, d_entry) in raws.iter().zip(raws_dec.iter()) {
                    let (p_data, p_t, p_d) = p;
                    let (d_data, d_t, d_d) = d_entry;
                    if *p_d != *d_d {
                        eprintln!(
                            "[QCF-dump]{} decode-x merge: WARN dim mismatch prefill={} decode={}",
                            log_prefix, p_d, d_d
                        );
                        continue;
                    }
                    let dim = *p_d;
                    let mut merged: Vec<f32> = Vec::with_capacity((*p_t + *d_t) * dim);
                    merged.extend_from_slice(p_data);
                    merged.extend_from_slice(d_data);
                    raws_merged.push((merged, *p_t + *d_t, dim));
                }
                let pairs_pd =
                    crate::models::weights::noise_table::compute_cascade_attn_perturbation(
                        &model.layers,
                        sec,
                        &raws_merged,
                        n_heads,
                        n_kv_heads,
                        d_head,
                    );
                direct_attn_f5_prefill_decode =
                    Some(pairs_pd.iter().map(|(_, b)| *b).collect::<Vec<f32>>());
            }
        }
        (table, eps_single, eps_multi, eps_abs, eps_qcf, f4, f5)
    } else {
        (collector.build(), None, None, None, None, None, None)
    };
    eprintln!(
        "[QCF-dump]{} ImportanceTable built: {} entries (dpllm_epsilon={}, multi={}, abs={}, qcf={})",
        log_prefix,
        imp_table.len(),
        if dpllm_epsilon.is_some() {
            "computed"
        } else {
            "skipped"
        },
        if dpllm_epsilon_multi.is_some() {
            "computed"
        } else {
            "skipped"
        },
        if dpllm_epsilon_abs.is_some() {
            "computed"
        } else {
            "skipped"
        },
        if dpllm_epsilon_qcf.is_some() {
            "computed"
        } else {
            "skipped"
        },
    );
    for kv in kv_caches.iter_mut() {
        kv.current_pos = 0;
    }

    // ── Optional swap with importance-guided decider ──────────────────────────
    let decision = if let Some(ratio) = force_ratio {
        let ratio = ratio.clamp(0.0, 1.0);
        eprintln!(
            "[QCF-dump]{} swap algorithm: {} (execute_swap={})",
            log_prefix,
            swap_algorithm.short_name(),
            execute_swap,
        );
        let decider = WeightSwapDecider {
            importance: Some(&imp_table),
            noise: Some(model.quant_noise.as_ref()),
            n_decoder_layers: model.layers.len(),
            currently_swapped: &[],
            allow_boundary_layers: read_allow_boundary_env(),
            algorithm: swap_algorithm,
        };
        let decider_decision = decider.decide(ratio);

        // §4 ground-truth path: when `--swap-only-layers` is set, override the
        // decider's selection with the explicit list. The decider's
        // `qcf_swap_estimate` is recomputed against this override so the dump
        // JSON reports the QCF prediction for the actually-swapped set.
        let decision = if let Some(only) = swap_only_layers {
            let override_layers: Vec<usize> = only
                .iter()
                .copied()
                .filter(|i| *i < model.layers.len())
                .collect();
            let qcf_override = crate::models::weights::compute_qcf_weight_swap(
                &override_layers,
                model.quant_noise.as_ref(),
                Some(&imp_table),
                model.layers.len(),
            );
            eprintln!(
                "[QCF-dump]{} swap-only override: layers={:?} (ignoring algorithm/ratio decision)",
                log_prefix, override_layers,
            );
            crate::models::weights::SwapDecision {
                selected_layers: override_layers,
                qcf_swap_estimate: qcf_override,
                fallback_used: false,
            }
        } else {
            decider_decision
        };

        // Trajectory mode (`--qcf-trajectory`): return the decision without
        // executing the swap — the caller drives swap one layer at a time
        // around per-step eval-ll measurements.
        if !execute_swap {
            return Ok(QcfWarmupResult {
                importance: imp_table,
                decision: Some(decision),
                dpllm_epsilon,
                dpllm_epsilon_multi,
                dpllm_epsilon_abs,
                dpllm_epsilon_qcf,
                direct_attn_f4,
                direct_attn_f5,
                direct_attn_f5_decode_only,
                direct_attn_f5_prefill_decode,
            });
        }

        if decision.selected_layers.is_empty() {
            eprintln!(
                "[QCF-dump]{} swap: ratio={:.2} → 0 layers selected (qcf=0.0)",
                log_prefix, ratio,
            );
        } else {
            let report = run_layer_swap(
                model,
                &decision.selected_layers,
                gpu_backend,
                cpu_backend,
                None,
                // LISWAP-3: QCF dump path does not exercise the pool yet —
                // Stage 3 prototype only wires --force-swap-ratio paths.
                #[cfg(feature = "opencl")]
                None,
                #[cfg(feature = "cuda-embedded")]
                None,
                #[cfg(feature = "cuda-embedded")]
                None,
            )
            .map_err(|e| anyhow::anyhow!("[QCF-dump]{} swap failed: {}", log_prefix, e))?;
            eprintln!(
                "[QCF-dump]{} swap: ratio={:.2}, layers={}/{}, qcf_pred={:.4}, \
                 fallback={}, latency={:.1}ms",
                log_prefix,
                ratio,
                report.swapped.len(),
                model.layers.len(),
                decision.qcf_swap_estimate,
                decision.fallback_used,
                report.latency_ms,
            );
        }
        Some(decision)
    } else {
        None
    };

    Ok(QcfWarmupResult {
        importance: imp_table,
        decision,
        dpllm_epsilon,
        dpllm_epsilon_multi,
        dpllm_epsilon_abs,
        dpllm_epsilon_qcf,
        direct_attn_f4,
        direct_attn_f5,
        direct_attn_f5_decode_only,
        direct_attn_f5_prefill_decode,
    })
}

// ─── Phase 4-C-1: PPL/standard generate 분기와 공유하는 swap dispatch + ─────
//  weight dump helpers. 본문 변경 없음.

pub fn dispatch_swap_weights(
    model: &crate::models::transformer::TransformerModel,
    ratio: f32,
    target_dtype: llm_shared::DtypeTag,
    importance_table: Option<&crate::core::qcf::ImportanceTable>,
    decode_token_index: usize,
    swap_plan_out: &mut Option<crate::models::weights::IncrementalSwapPlan>,
    manager_report_out: &mut Option<(f32, usize, std::time::Instant, f32)>,
) {
    use crate::models::weights::{
        IncrementalSwapPlan, SwapDecision, WeightSwapDecider, compute_qcf_weight_swap,
    };
    use llm_shared::DtypeTag;

    // ── 1. Validation ──────────────────────────────────────────────────────
    if model.secondary_mmap.is_none() {
        eprintln!("[WeightSwap] Rejected: no_secondary (ENG-DAT-C09)");
        return;
    }
    if ratio <= 0.0 || ratio > 1.0 {
        eprintln!("[WeightSwap] Rejected: invalid_ratio ({:.4})", ratio);
        return;
    }
    if target_dtype != DtypeTag::Q4_0 {
        eprintln!(
            "[WeightSwap] Rejected: unsupported_dtype ({:?}) (INV-126)",
            target_dtype
        );
        return;
    }

    // ── 1b. In-flight plan check ───────────────────────────────────────────
    // Reject if a plan is already in flight (CLI or manager). Prevents
    // concurrent plan conflict (spec: manager signal accept only when no plan).
    if swap_plan_out.is_some() {
        eprintln!(
            "[WeightSwap] Rejected: incremental plan already in-flight (ratio={:.2}). \
             Wait for current plan to complete before sending a new SwapWeights signal.",
            ratio
        );
        return;
    }

    // ── 2. Collect currently-swapped layers ────────────────────────────────
    let n_layers = model.layers.len();
    let currently_swapped: Vec<usize> = (0..n_layers)
        .filter(|&i| model.layers[i].current_dtype() == DType::Q4_0)
        .collect();

    // ── 3. Decider ─────────────────────────────────────────────────────────
    let allow_boundary = read_allow_boundary_env();
    eprintln!(
        "[Decider] allow_boundary_layers={} (ratio={:.4})",
        allow_boundary, ratio
    );
    let decider = WeightSwapDecider {
        importance: importance_table,
        noise: Some(&model.quant_noise),
        n_decoder_layers: n_layers,
        currently_swapped: &currently_swapped,
        allow_boundary_layers: allow_boundary,
        algorithm: crate::models::weights::SwapAlgorithm::ImportanceAware,
    };
    let decision: SwapDecision = decider.decide(ratio);

    if decision.selected_layers.is_empty() {
        eprintln!(
            "[WeightSwap] No layers to swap (ratio={:.2}, already_swapped={})",
            ratio,
            currently_swapped.len()
        );
        // Empty swap is Ok per spec (already fully swapped); no plan committed.
        return;
    }

    // ── 4. Compute QCF estimate for the planned layers ─────────────────────
    let qcf_swap_estimated = compute_qcf_weight_swap(
        &decision.selected_layers,
        &model.quant_noise,
        importance_table,
        n_layers,
    );

    // ── 5. Commit incremental plan (K=2, same as CLI --swap-incremental-per-tick 2) ──
    let n_planned = decision.selected_layers.len();
    let per_tick = 2usize; // LISWAP-6: K=2 hard upper cap for manager path
    let ticks_est = n_planned.div_ceil(per_tick);
    eprintln!(
        "[WeightSwap] manager path: ratio={:.2}, {} target layers, per_tick={} ({} ticks estimated), qcf_estimated={:.4}",
        ratio, n_planned, per_tick, ticks_est, qcf_swap_estimated,
    );

    *swap_plan_out = Some(IncrementalSwapPlan::new(
        decision.selected_layers,
        per_tick,
        decode_token_index,
    ));
    *manager_report_out = Some((
        ratio,
        n_planned,
        std::time::Instant::now(),
        qcf_swap_estimated,
    ));
}

/// LISWAP-PPL diagnostic: dump every layer's weight tensors (wq/wk/wv/wo/
/// w_gate/w_up/w_down) to raw bin files under `out_dir`. File naming:
/// `layer{NN}_{tensor}_{dtype}.bin` (e.g. `layer00_wq_Q4_0.bin`). Each file
/// holds the raw GPU buffer bytes for that tensor at the moment of the call.
///
/// Two such dumps (one from a Q4-native model load, one from an F16 model
/// after swap completion) can be byte-compared on the host to determine
/// whether the swap path produces bit-identical Q4 weights.
pub fn dump_layer_weights_to_dir(
    model: &crate::models::transformer::TransformerModel,
    backend: &Arc<dyn Backend>,
    out_dir: &std::path::Path,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let n = model.layers.len();
    eprintln!(
        "[Q4-DUMP] dumping {} layer weights to {}",
        n,
        out_dir.display()
    );
    for (i, slot) in model.layers.iter().enumerate() {
        let weights = slot.load_weights();
        let dtype = slot.current_dtype();
        let tensors: [(&str, &crate::core::tensor::Tensor); 7] = [
            ("wq", &weights.wq),
            ("wk", &weights.wk),
            ("wv", &weights.wv),
            ("wo", &weights.wo),
            ("w_gate", &weights.w_gate),
            ("w_up", &weights.w_up),
            ("w_down", &weights.w_down),
        ];
        for (name, t) in tensors {
            let nbytes = t.buffer().size();
            if nbytes == 0 {
                eprintln!(
                    "[Q4-DUMP] layer{:02} {} dtype={:?} SKIP (size=0)",
                    i, name, dtype
                );
                continue;
            }
            let mut bytes = vec![0u8; nbytes];
            // For OpenCL/CUDA tensors `buffer().as_ptr()` is the cl_mem/cu_ptr
            // handle and may look like a host nullptr — backend.read_buffer
            // does the device→host copy via the backend-specific path, so we
            // rely on its return value rather than pre-checking as_ptr.
            match backend.read_buffer(t, &mut bytes) {
                Ok(()) => {
                    let fname = format!("layer{:02}_{}_{:?}.bin", i, name, dtype);
                    let path = out_dir.join(&fname);
                    std::fs::write(&path, &bytes)?;
                    eprintln!(
                        "[Q4-DUMP] layer{:02} {:8} dtype={:>5?} bytes={:8} → {}",
                        i, name, dtype, nbytes, fname
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[Q4-DUMP] layer{:02} {} dtype={:?} SKIP (read_buffer failed: {})",
                        i, name, dtype, e
                    );
                }
            }
        }
    }
    // Also dump model-level tensors that are NOT inside per-layer slots and
    // therefore are NOT touched by weight swap: embed_tokens, final norm, and
    // lm_head. These three are the most likely sources of E ≠ D NLL drift
    // because (a) the F16 model's lm_head is typically tied to embed_tokens
    // and (b) any missing lm_head is derived via F16→Q4_0 quantization at
    // load time, whose result may not match a standalone Q4_0 GGUF's lm_head
    // byte-for-byte.
    let model_tensors: [(&str, &crate::core::tensor::Tensor); 3] = [
        ("embed_tokens", &model.embed_tokens),
        ("norm", &model.norm),
        ("lm_head", &model.lm_head),
    ];
    for (name, t) in model_tensors {
        let nbytes = t.buffer().size();
        if nbytes == 0 {
            eprintln!("[Q4-DUMP] model.{} SKIP (size=0)", name);
            continue;
        }
        let dt = t.dtype();
        let mut bytes = vec![0u8; nbytes];
        match backend.read_buffer(t, &mut bytes) {
            Ok(()) => {
                let fname = format!("model_{}_{:?}.bin", name, dt);
                let path = out_dir.join(&fname);
                std::fs::write(&path, &bytes)?;
                eprintln!(
                    "[Q4-DUMP] model.{:14} dtype={:>5?} bytes={:8} → {}",
                    name, dt, nbytes, fname
                );
            }
            Err(e) => {
                // The lm_head can live on a CPU backend even when the main
                // backend is GPU (`lm_head_on_cpu`) — fall back to CpuBackend
                // for that case so we still get a dump file out.
                let cpu_be: Arc<dyn Backend> = Arc::new(crate::backend::cpu::CpuBackend::new());
                match cpu_be.read_buffer(t, &mut bytes) {
                    Ok(()) => {
                        let fname = format!("model_{}_{:?}.bin", name, dt);
                        let path = out_dir.join(&fname);
                        std::fs::write(&path, &bytes)?;
                        eprintln!(
                            "[Q4-DUMP] model.{:14} dtype={:>5?} bytes={:8} → {} (via CPU fallback)",
                            name, dt, nbytes, fname
                        );
                    }
                    Err(e2) => {
                        eprintln!(
                            "[Q4-DUMP] model.{} SKIP (read_buffer failed: gpu={}, cpu={})",
                            name, e, e2
                        );
                    }
                }
            }
        }
    }
    eprintln!(
        "[Q4-DUMP] complete: {} layers + 3 model tensors dumped to {}",
        n,
        out_dir.display()
    );
    Ok(())
}
