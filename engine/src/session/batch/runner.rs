//! Phase 4-A: `run_prompt_batch` — `bin/generate.rs::main()`의 batch 분기
//! (l.2235~3094) 본문을 외과적으로 이동.
//!
//! lift-and-shift 원칙: 본문은 원본을 그대로 재현하며, BatchRunCtx의
//! field를 destructure 후 local var로 풀어 본문이 outer-scope를 그대로
//! 참조하던 패턴을 보존한다. 추상화는 도입하지 않는다.

use std::sync::Arc;

use anyhow::Result;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::hardware::DeviceTarget;
use crate::inference::sampling;
use crate::inference::skip_config::SkipConfig;
use crate::layers::workspace::{
    LayerWorkspace, PartitionWorkspace, PartitionWsCell, WorkspaceConfig,
};
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
use crate::models::transformer::TransformerModelForwardFmtArgs;
use crate::pressure::kv_cache::KVCache;
use crate::resilience::KVSnapshot;
use crate::session::batch::args::BatchRunCtx;
use crate::session::batch::helpers::{
    load_prompt_batch, make_partition_gpu_alloc, resolve_prompt, unix_ts,
};
use crate::session::cli::parse_qcf_sample_layers;
use crate::session::eval::EvalCacheKind;
use crate::shape::Shape;
use crate::tensor::Tensor;

pub fn run_prompt_batch(ctx: BatchRunCtx) -> Result<()> {
    let BatchRunCtx {
        args,
        mut backend,
        memory,
        hardware,
        model,
        tokenizer,
        mut kv_caches,
        cache_manager,
        mut score_accumulator,
        mut command_executor,
        mut skip_config,
        hidden_size,
        vocab_size,
        max_seq_len,
        mut is_gpu,
        actual_protected_prefix,
        score_based_eviction,
        mut throttle_delay_ms,
        weights_on_gpu,
        kv_heads,
        head_dim,
        kv_type: _kv_type,
        mut last_skip_ratio,
        sampling_config,
    } = ctx;

    // Phase α-W-2: hardware resolver 에서 4 secondary Arc 를 재바인딩.
    // 로컬이 정확히 같은 Arc 를 보유하므로 본문 사용처는 무변경.
    let cpu_backend_arc = hardware
        .resolve(DeviceTarget::Cpu)
        .expect("Cpu always resolves")
        .0
        .clone();
    let cpu_memory_arc = hardware
        .resolve(DeviceTarget::Cpu)
        .expect("Cpu always resolves")
        .1
        .clone();
    let gpu_backend_arc: Option<Arc<dyn Backend>> =
        hardware.resolve(DeviceTarget::Gpu).map(|(b, _)| b.clone());
    let gpu_memory_arc: Option<Arc<dyn Memory>> =
        hardware.resolve(DeviceTarget::Gpu).map(|(_, m)| m.clone());

    let batch_path = args
        .prompt_batch
        .as_deref()
        .expect("run_prompt_batch only called when args.prompt_batch is Some");

    let entries = load_prompt_batch(batch_path)?;
    if entries.is_empty() {
        anyhow::bail!("prompt-batch file is empty: {}", batch_path);
    }
    eprintln!(
        "[Batch] Loaded {} entries from {}",
        entries.len(),
        batch_path
    );

    // ARGUS hook: emit Step1~6 metrics (qcf_caote_max / qcf_per_head /
    // qcf_topk_retention_* / attention_entropy / qcf_beta_amplified_* /
    // qcf_per_layer*) per record, alongside legacy fields.
    // Hook owns cache_manager + score_accumulator from here; subsequent
    // forward calls in this branch route score_accumulator through the hook.
    use crate::session::eval::StepHook;
    let pb_qcf_mode_enum = match args.qcf_mode.as_str() {
        "caote" => crate::qcf_types::QcfMode::Caote,
        "both" => crate::qcf_types::QcfMode::Both,
        _ => crate::qcf_types::QcfMode::Attn,
    };
    let pb_qcf_config = crate::qcf_types::QcfConfig {
        mode: pb_qcf_mode_enum,
        ..crate::qcf_types::QcfConfig::default()
    };
    let pb_ratio_mode = args.kv_budget_ratio() > 0.0;
    let pb_hook_budget = if pb_ratio_mode { 0 } else { args.kv_budget() };
    let pb_is_d2o = args.eviction_policy() == "d2o";
    let pb_num_layers = model.config.num_hidden_layers;
    let pb_sample_layers = if args.enable_qcf_experimental {
        parse_qcf_sample_layers(&args.qcf_sample_layers, pb_num_layers)
            .map_err(|e| anyhow::anyhow!("--qcf-sample-layers: {e}"))?
    } else {
        vec![0]
    };
    let mut hook = crate::session::eval::EvictionHook::new(
        cache_manager,
        score_accumulator.take(),
        pb_qcf_config,
        pb_hook_budget,
        actual_protected_prefix,
        score_based_eviction,
        args.h2o_keep_ratio(),
        pb_is_d2o,
        args.kv_type.clone(),
        backend.clone(),
        args.enable_qcf_experimental,
        pb_sample_layers,
    );

    let mut iteration = 0usize;

    // Pre-allocate generation buffers (once)
    let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        logits_buf,
        backend.clone(),
    );
    let eos_id = model.config.eos_token_id;

    // Pre-allocate workspace (once)
    let q_dim = model.config.num_attention_heads * model.config.head_dim;
    let k_dim = model.config.num_key_value_heads * model.config.head_dim;
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
            max_seq_len: args.max_seq_len,
        },
        memory.as_ref(),
        backend.clone(),
    )?;

    // Attach partition workspace if tensor partition is active.
    // Use UnifiedBuffer (ALLOC_HOST_PTR, host-accessible + GPU-accessible) for partition
    // buffers so merge can use direct pointer access instead of read_buffer/write_buffer.
    let layer0_partition_probe = model.layers[0].load_weights();
    if let Some(ref ctx) = layer0_partition_probe.partition_ctx {
        let gpu_alloc = make_partition_gpu_alloc(&*backend, memory.as_ref());

        // Zero-copy residual: permanent-map ws.residual's UnifiedBuffer so the
        // partition decode path can read residual directly via as_ptr() and
        // skip the per-layer read_buffer DMA (currently ~1.15 ms/layer).
        // Gate behind LLMRS_PARTITION_ZCOPY_RESIDUAL=1, or auto-enable when
        // poll-flag mode is active (skipping the read_buffer is the whole
        // point of the spin-poll path).
        #[cfg(feature = "opencl")]
        if std::env::var_os("LLMRS_PARTITION_ZCOPY_RESIDUAL").is_some()
            || crate::layers::tensor_partition::partition_poll_flag_enabled()
        {
            if let Some(ub) = gen_ws
                .residual
                .buffer()
                .as_any()
                .downcast_ref::<crate::memory::opencl::unified::UnifiedBuffer>()
            {
                ub.map()?;
                eprintln!("[Partition] Residual UnifiedBuffer permanent-mapped for zero-copy");
            } else {
                eprintln!(
                    "[Partition] WARN: residual buffer is not UnifiedBuffer (zero-copy skipped)"
                );
            }
        }

        gen_ws.partition_ws = Some(Arc::new(PartitionWsCell::new(PartitionWorkspace::new(
            ctx.gate.split_row,
            ctx.up.split_row,
            ffn_hidden,
            hidden_size,
            &gpu_alloc,
            backend.clone(),
            cpu_backend_arc.clone(),
        )?)));
    }

    // Pre-allocate CPU/GPU single-token tensors
    let cpu_gen_indices_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_indices_buf,
        cpu_backend_arc.clone(),
    );
    let gpu_gen_input_buf = memory.alloc(4, DType::U8)?;
    let mut gen_input_tensor =
        Tensor::new(Shape::new(vec![1, 1]), gpu_gen_input_buf, backend.clone());
    let mut logits_cpu = vec![0.0f32; vocab_size];

    // Persistent prefill policy state: survives across batches.
    // Only reset by RestoreDefaults, not by prefill→decode transition.
    let mut persistent_chunk_size: Option<usize> = None;
    let mut persistent_yield_ms: Option<u32> = None;
    let mut persistent_cpu_chunk_size: Option<usize> = None;

    'outer: loop {
        for entry in &entries {
            if args.max_iterations > 0 && iteration >= args.max_iterations {
                break 'outer;
            }

            let prompt_text = resolve_prompt(entry)?;
            let encoding = tokenizer
                .encode(prompt_text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let batch_input_ids: Vec<u32> = encoding.get_ids().to_vec();
            let prompt_tokens = batch_input_ids.len();

            eprintln!(
                "[Batch] #{} id={}, prompt_tokens={}",
                iteration, entry.id, prompt_tokens
            );

            // Per-record budget when --kv-budget-ratio is active
            // (mirrors eval-ll path eval_loop.rs:207). Without this the hook
            // sees effective_budget=0 and post_prefill early-returns,
            // suppressing eviction and ARGUS metric collection.
            if pb_ratio_mode {
                let dynamic_budget = ((prompt_tokens as f32) * args.kv_budget_ratio()) as usize;
                hook.set_effective_budget(dynamic_budget.max(1));
            }

            let entry_start = std::time::Instant::now();

            eprintln!(
                "[Batch] #{} id={} prefill_start ts={:.3}",
                iteration,
                entry.id,
                unix_ts()
            );

            // === PREFILL ===
            let process_len = batch_input_ids.len();
            if process_len > max_seq_len {
                eprintln!(
                    "[Batch] #{} id={}: prompt too long ({} > {}), skipping",
                    iteration, entry.id, process_len, max_seq_len
                );
                let err_result = serde_json::json!({
                    "id": entry.id,
                    "error": format!("prompt too long: {} > {}", process_len, max_seq_len),
                });
                println!("{}", serde_json::to_string(&err_result)?);
                iteration += 1;
                continue;
            }

            // Chunked prefill
            // When resilience is enabled, auto-chunk at 256 for checkpoint support.
            // When GPU backend and prefill_chunk_size == 0, auto-derive a safe chunk
            // size from max_single_alloc() to avoid CL_INVALID_BUFFER_SIZE (-61).
            let auto_gpu_chunk: Option<usize> = if args.prefill_chunk_size == 0 && backend.is_gpu()
            {
                let max_alloc = backend.max_single_alloc();
                if max_alloc > 0 {
                    // Each chunk needs a logits buffer: chunk * vocab_size * 4 bytes.
                    // Use 50% of max_single_alloc as conservative budget.
                    let budget = max_alloc / 2;
                    let by_vocab = (budget / (vocab_size * 4)).max(1);
                    // Also bound by hidden_size to keep activation buffers feasible.
                    let by_hidden = (max_alloc / (hidden_size * 4)).max(1);
                    let derived = by_vocab.min(by_hidden).min(512);
                    Some(derived)
                } else {
                    None
                }
            } else {
                None
            };
            let chunk_size = if args.prefill_chunk_size > 0 && args.prefill_chunk_size < process_len
            {
                args.prefill_chunk_size
            } else if let Some(auto) = auto_gpu_chunk {
                if auto < process_len {
                    eprintln!(
                        "[Prefill] prefill_chunk_size auto-selected: {} (max_alloc={}MB, vocab={}, hidden={})",
                        auto,
                        backend.max_single_alloc() / (1024 * 1024),
                        vocab_size,
                        hidden_size,
                    );
                    auto
                } else {
                    process_len
                }
            } else if args.enable_resilience && process_len > 256 {
                256
            } else {
                process_len
            };
            let chunked = chunk_size < process_len;

            // Dynamic prefill policy: use persistent values if set by prior
            // SetPrefillPolicy, otherwise fall back to CLI defaults.
            let mut effective_chunk_size = persistent_chunk_size.unwrap_or(chunk_size);
            let mut effective_yield_ms = persistent_yield_ms.unwrap_or(args.prefill_yield_ms);
            let mut effective_cpu_chunk_size =
                persistent_cpu_chunk_size.unwrap_or(args.prefill_cpu_chunk_size);

            let (prefill_logits_shape, prefill_logits_buf_size) = if chunked {
                (Shape::new(vec![1, 1, vocab_size]), vocab_size * 4)
            } else {
                (
                    Shape::new(vec![1, process_len, vocab_size]),
                    process_len * vocab_size * 4,
                )
            };
            // Use CPU memory when on CPU after SwitchHw; GPU memory otherwise.
            let batch_effective_mem: &dyn Memory = if is_gpu {
                memory.as_ref()
            } else {
                cpu_memory_arc.as_ref()
            };
            let prefill_logits_buf =
                batch_effective_mem.alloc(prefill_logits_buf_size, DType::F32)?;
            let mut prefill_logits =
                Tensor::new(prefill_logits_shape, prefill_logits_buf, backend.clone());

            let prefill_timer = std::time::Instant::now();
            let mut deferred_switch: Option<String> = None;
            let total_chunks = process_len.div_ceil(chunk_size);

            // Report prefill start to resilience manager.
            if let Some(executor) = &mut command_executor {
                executor.set_prefill_state("prefill", 0, process_len);
            }

            let mut chunk_start = 0;
            let mut chunk_idx = 0usize;
            while chunk_start < process_len {
                // Guard: effective_chunk_size must be at least 1.
                let ecs = effective_chunk_size.max(1);
                let chunk_end = (chunk_start + ecs).min(process_len);
                let chunk_tokens = &batch_input_ids[chunk_start..chunk_end];
                let chunk_len = chunk_tokens.len();

                let cpu_chunk_buf = Galloc::new().alloc(chunk_len * 4, DType::U8)?;
                unsafe {
                    let ptr = cpu_chunk_buf.as_mut_ptr() as *mut u32;
                    std::ptr::copy_nonoverlapping(chunk_tokens.as_ptr(), ptr, chunk_len);
                }
                let cpu_chunk_tensor = Tensor::new(
                    Shape::new(vec![1, chunk_len]),
                    cpu_chunk_buf,
                    cpu_backend_arc.clone(),
                );
                let input_tensor = backend.copy_from(&cpu_chunk_tensor)?;

                // Phase α-K ①-d: forward_into → fmt round-trip (GPU prefill chunk).
                KVCache::forward_fmt_roundtrip(&mut kv_caches, |fmts| {
                    model.forward_into_fmt(TransformerModelForwardFmtArgs {
                        input_tokens: &input_tensor,
                        start_pos: chunk_start,
                        fmts,
                        backend: &backend,
                        memory: batch_effective_mem,
                        logits_out: &mut prefill_logits,
                        x_gen: None,
                        workspace: None,
                        logits_last_only: chunked,
                        score_accumulator: hook.score_accumulator(),
                        skip_config: skip_config.as_ref(),
                        importance_collector: None,
                        cache_self_need_scores: false,
                    })
                })?;
                backend.synchronize()?;
                drop(input_tensor);

                chunk_start = chunk_end;

                // Inter-chunk yield: sleep after GPU chunk to release compute.
                if effective_yield_ms > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(effective_yield_ms as u64));
                }

                // CPU interleave: process next chunk on CPU while GPU is free.
                // Invariant: the last chunk must be processed by GPU so that
                // prefill_logits (GPU buffer) is valid at the end.
                if effective_cpu_chunk_size > 0 && chunk_start < process_len {
                    let remaining = process_len - chunk_start;
                    if remaining > effective_cpu_chunk_size {
                        // Flush GPU caches before CPU reads KV buffers (ARM UMA coherence).
                        for kv in kv_caches.iter() {
                            kv.k_buffer.buffer().map_for_cpu()?;
                            kv.v_buffer.buffer().map_for_cpu()?;
                        }

                        let cpu_end = (chunk_start + effective_cpu_chunk_size)
                            .min(process_len.saturating_sub(1));
                        if cpu_end > chunk_start {
                            let cpu_tokens = &batch_input_ids[chunk_start..cpu_end];
                            let cpu_len = cpu_tokens.len();

                            let cpu_in_buf = Galloc::new().alloc(cpu_len * 4, DType::U8)?;
                            unsafe {
                                let ptr = cpu_in_buf.as_mut_ptr() as *mut u32;
                                std::ptr::copy_nonoverlapping(cpu_tokens.as_ptr(), ptr, cpu_len);
                            }
                            let cpu_in_tensor = Tensor::new(
                                Shape::new(vec![1, cpu_len]),
                                cpu_in_buf,
                                cpu_backend_arc.clone(),
                            );

                            let cpu_chunk_start_pos = chunk_start;

                            // CPU prefill logits: separate CPU buffer (discarded).
                            let cpu_logits_buf =
                                cpu_memory_arc.alloc(vocab_size * 4, DType::F32)?;
                            let mut cpu_logits = Tensor::new(
                                Shape::new(vec![1, 1, vocab_size]),
                                cpu_logits_buf,
                                cpu_backend_arc.clone(),
                            );

                            // Phase α-K ①-d: forward_into → fmt round-trip (CPU interleave prefill).
                            KVCache::forward_fmt_roundtrip(&mut kv_caches, |fmts| {
                                model.forward_into_fmt(TransformerModelForwardFmtArgs {
                                    input_tokens: &cpu_in_tensor,
                                    start_pos: cpu_chunk_start_pos,
                                    fmts,
                                    backend: &cpu_backend_arc,
                                    memory: cpu_memory_arc.as_ref(),
                                    logits_out: &mut cpu_logits,
                                    x_gen: None,
                                    workspace: None,
                                    logits_last_only: true,
                                    score_accumulator: hook.score_accumulator(),
                                    skip_config: skip_config.as_ref(),
                                    importance_collector: None,
                                    cache_self_need_scores: false,
                                })
                            })?;
                            drop(cpu_in_tensor);
                            drop(cpu_logits);

                            chunk_start = cpu_end;
                        }
                    }
                }

                // ── Prefill resilience checkpoint (chunk boundary) ──
                if chunked && let Some(executor) = &mut command_executor {
                    let kv_snap = KVSnapshot {
                        total_bytes: kv_caches
                            .iter()
                            .map(|c| {
                                (c.k_buffer.buffer().size() + c.v_buffer.buffer().size()) as u64
                            })
                            .sum(),
                        total_tokens: kv_caches[0].current_pos,
                        capacity: kv_caches[0].capacity(),
                        protected_prefix: actual_protected_prefix,
                        kv_dtype: args.kv_type.clone(),
                        eviction_policy: args.eviction_policy().to_string(),
                        skip_ratio: 0.0,
                    };
                    let plan = executor.poll(&kv_snap);

                    // SetPrefillPolicy: dynamically adjust chunk/yield/cpu parameters.
                    // Values persist across batches until RestoreDefaults.
                    if let Some(v) = plan.prefill_chunk_size {
                        effective_chunk_size = v;
                        persistent_chunk_size = Some(v);
                        eprintln!("[Prefill] Policy: chunk_size -> {}", v);
                    }
                    if let Some(v) = plan.prefill_yield_ms {
                        effective_yield_ms = v;
                        persistent_yield_ms = Some(v);
                        eprintln!("[Prefill] Policy: yield_ms -> {}", v);
                    }
                    if let Some(v) = plan.prefill_cpu_chunk_size {
                        let layer0_probe = model.layers[0].load_weights();
                        if v > 0 && layer0_probe.wq.as_ptr().is_null() {
                            eprintln!(
                                "[Prefill] Policy: cpu_chunk_size={} rejected — weights not CPU-accessible. \
                                 Use --resilience-prealloc-switch or --prefill-cpu-chunk-size at CLI.",
                                v
                            );
                        } else {
                            effective_cpu_chunk_size = v;
                            persistent_cpu_chunk_size = Some(v);
                            eprintln!("[Prefill] Policy: cpu_chunk_size -> {}", v);
                        }
                    }

                    // Throttle: sleep between chunks
                    if plan.throttle_delay_ms > 0 && plan.throttle_delay_ms != throttle_delay_ms {
                        eprintln!(
                            "[Prefill] Throttle: {}ms -> {}ms",
                            throttle_delay_ms, plan.throttle_delay_ms
                        );
                    }
                    throttle_delay_ms = plan.throttle_delay_ms;
                    if throttle_delay_ms > 0 {
                        eprintln!(
                            "[Prefill] Throttle: {}ms delay after chunk {}/{}",
                            throttle_delay_ms,
                            chunk_idx + 1,
                            total_chunks
                        );
                        std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
                    }

                    // LayerSkip
                    if plan.restore_defaults {
                        skip_config = None;
                        last_skip_ratio = None;
                        effective_chunk_size = chunk_size;
                        effective_yield_ms = args.prefill_yield_ms;
                        effective_cpu_chunk_size = args.prefill_cpu_chunk_size;
                        persistent_chunk_size = None;
                        persistent_yield_ms = None;
                        persistent_cpu_chunk_size = None;
                    } else if let Some(ratio) = plan.layer_skip
                        && last_skip_ratio != Some(ratio)
                    {
                        eprintln!("[Prefill] LayerSkip: ratio={:.2}", ratio);
                        skip_config = Some(SkipConfig::uniform_init(
                            model.config.num_hidden_layers,
                            ratio,
                        ));
                        last_skip_ratio = Some(ratio);
                    }

                    // SwitchHw: defer to post-prefill boundary.
                    // Mid-prefill switch causes segfault: model workspace buffers
                    // remain on the old backend; the next chunk accesses them
                    // from the new backend -> invalid memory reference.
                    if let Some(ref device) = plan.switch_device {
                        if deferred_switch.is_none() {
                            eprintln!(
                                "[Prefill] SwitchHw: deferring '{}' to post-prefill (chunk_pos={})",
                                device, kv_caches[0].current_pos
                            );
                        }
                        deferred_switch = Some(device.clone());
                    }

                    // Report prefill progress.
                    executor.set_prefill_state("prefill", chunk_start, process_len);
                }

                chunk_idx += 1;
            }

            let ttft_ms = prefill_timer.elapsed().as_secs_f64() * 1000.0;

            // Report transition to decode phase.
            if let Some(executor) = &mut command_executor {
                executor.set_prefill_state("decode", 0, 0);
            }

            // Sample first token from prefill logits
            let mut last_logits = vec![0.0f32; vocab_size];
            unsafe {
                let ptr = last_logits.as_mut_ptr() as *mut u8;
                let byte_len = vocab_size * 4;
                if chunked {
                    let slice = std::slice::from_raw_parts_mut(ptr, byte_len);
                    backend.read_buffer(&prefill_logits, slice)?;
                } else {
                    let mut full_logits = vec![0.0f32; process_len * vocab_size];
                    let full_ptr = full_logits.as_mut_ptr() as *mut u8;
                    let full_slice =
                        std::slice::from_raw_parts_mut(full_ptr, full_logits.len() * 4);
                    backend.read_buffer(&prefill_logits, full_slice)?;
                    let start_idx = (process_len - 1) * vocab_size;
                    last_logits.copy_from_slice(&full_logits[start_idx..start_idx + vocab_size]);
                }
            }
            drop(prefill_logits);

            // Execute deferred SwitchHw (from prefill checkpoint).
            // Now safe: prefill is done, logits read, all workspace released.
            if let Some(ref device) = deferred_switch
                && let (Some(gpu_be), Some(gpu_mem)) = (&gpu_backend_arc, &gpu_memory_arc)
            {
                match device.as_str() {
                    "cpu" if is_gpu => {
                        eprintln!("[Prefill->Decode] Executing deferred SwitchHw: GPU->CPU");
                        crate::pressure::kv_migrate::migrate_kv_caches(
                            &mut kv_caches,
                            &backend,
                            &cpu_backend_arc,
                            &cpu_backend_arc,
                            &cpu_memory_arc,
                            &cpu_memory_arc,
                            kv_heads,
                            head_dim,
                            max_seq_len,
                            false,
                        )?;
                        backend = cpu_backend_arc.clone();
                        is_gpu = false;
                        // Re-allocate decode buffers on CPU.
                        let new_lb = cpu_memory_arc.alloc(vocab_size * 4, DType::F32)?;
                        logits = Tensor::new(logits.shape().clone(), new_lb, backend.clone());
                        let new_xb = cpu_memory_arc.alloc(hidden_size * 4, DType::F32)?;
                        x_gen = Tensor::new(x_gen.shape().clone(), new_xb, backend.clone());
                        gen_ws = LayerWorkspace::new(
                            WorkspaceConfig {
                                batch_size: 1,
                                dim: model.config.hidden_size,
                                q_dim,
                                k_dim,
                                v_dim,
                                ffn_hidden,
                                n_heads: model.config.num_attention_heads,
                                max_seq_len: args.max_seq_len,
                            },
                            cpu_memory_arc.as_ref(),
                            backend.clone(),
                        )?;
                        let new_gi = cpu_memory_arc.alloc(4, DType::U8)?;
                        gen_input_tensor =
                            Tensor::new(gen_input_tensor.shape().clone(), new_gi, backend.clone());
                        eprintln!(
                            "[Prefill->Decode] SwitchHw: Switched to CPU (GPU handles released, decode buffers re-allocated)."
                        );
                    }
                    "gpu" | "opencl" if !is_gpu && weights_on_gpu => {
                        eprintln!("[Prefill->Decode] Executing deferred SwitchHw: CPU->GPU");
                        crate::pressure::kv_migrate::migrate_kv_caches(
                            &mut kv_caches,
                            &backend,
                            gpu_be,
                            &cpu_backend_arc,
                            &cpu_memory_arc,
                            gpu_mem,
                            kv_heads,
                            head_dim,
                            max_seq_len,
                            true,
                        )?;
                        backend = gpu_be.clone();
                        is_gpu = true;
                        let new_lb = gpu_mem.alloc(vocab_size * 4, DType::F32)?;
                        logits = Tensor::new(logits.shape().clone(), new_lb, backend.clone());
                        let new_xb = gpu_mem.alloc(hidden_size * 4, DType::F32)?;
                        x_gen = Tensor::new(x_gen.shape().clone(), new_xb, backend.clone());
                        gen_ws = LayerWorkspace::new(
                            WorkspaceConfig {
                                batch_size: 1,
                                dim: model.config.hidden_size,
                                q_dim,
                                k_dim,
                                v_dim,
                                ffn_hidden,
                                n_heads: model.config.num_attention_heads,
                                max_seq_len: args.max_seq_len,
                            },
                            gpu_mem.as_ref(),
                            backend.clone(),
                        )?;
                        let new_gi = gpu_mem.alloc(4, DType::U8)?;
                        gen_input_tensor =
                            Tensor::new(gen_input_tensor.shape().clone(), new_gi, backend.clone());
                        eprintln!(
                            "[Prefill->Decode] SwitchHw: Switched to GPU (decode buffers re-allocated)."
                        );
                    }
                    _ => {} // Already on requested backend
                }
            }

            let mut batch_tokens = batch_input_ids.clone();
            let next_token_id = sampling::sample(
                &mut last_logits,
                &batch_tokens,
                vocab_size,
                &sampling_config,
                None,
            );
            batch_tokens.push(next_token_id);
            let mut batch_start_pos = process_len;

            // ── Score collection probe ──
            // Mirrors eval_loop.rs:246~287. Batch prefill calls forward with
            // workspace=None, so the hook's score_accumulator stays empty
            // → ARGUS metrics fall back to defaults (0). Re-feed the last
            // prompt token as a 1-step decode forward to populate per-head
            // attention scores, then restore current_pos so cache state
            // matches prompt_tokens (probe entry beyond current_pos is
            // invisible to subsequent forward calls).
            // Phase α-K BC 5-E: KVCache inherent `current_pos` 직접 호출 (KVCacheOps import 제거).
            if hook.needs_score_probe(&kv_caches) {
                let saved_positions: Vec<usize> =
                    kv_caches.iter().map(|c| c.current_pos()).collect();
                let last_prompt_token = batch_input_ids[prompt_tokens - 1];
                unsafe {
                    *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_prompt_token;
                }
                backend.write_buffer(&mut gen_input_tensor, unsafe {
                    std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
                })?;
                if let Some(acc) = hook.score_accumulator() {
                    acc.begin_step();
                }
                let probe_mem: &dyn Memory = if is_gpu {
                    memory.as_ref()
                } else {
                    cpu_memory_arc.as_ref()
                };
                // Phase α-K ①-d: forward_into → fmt round-trip (decode probe; workspace=Some →
                // forward_gen_fmt, 발산 A 무관). score-feed 활성(H2O 누적).
                KVCache::forward_fmt_roundtrip(&mut kv_caches, |fmts| {
                    model.forward_into_fmt(TransformerModelForwardFmtArgs {
                        input_tokens: &gen_input_tensor,
                        start_pos: prompt_tokens - 1,
                        fmts,
                        backend: &backend,
                        memory: probe_mem,
                        logits_out: &mut logits,
                        x_gen: Some(&mut x_gen),
                        workspace: Some(&mut gen_ws),
                        logits_last_only: false,
                        score_accumulator: hook.score_accumulator(),
                        skip_config: skip_config.as_ref(),
                        importance_collector: None,
                        cache_self_need_scores: false,
                    })
                })?;
                for (cache, &pos) in kv_caches.iter_mut().zip(saved_positions.iter()) {
                    cache.set_current_pos(pos);
                }
            }

            // ARGUS Step1~6: compute experimental_qcf payload from prefill state.
            // Also triggers post-prefill eviction when budget exceeded.
            hook.post_prefill(&mut kv_caches);

            eprintln!(
                "[Batch] #{} id={} prefill_end ts={:.3}",
                iteration,
                entry.id,
                unix_ts()
            );
            eprintln!(
                "[Batch] #{} id={} decode_start ts={:.3}",
                iteration,
                entry.id,
                unix_ts()
            );

            // === DECODE LOOP ===
            let mut tbt_values_batch: Vec<f64> = Vec::new();
            let mut generated_count: usize = 1; // first token already sampled
            let mut last_token_time = std::time::Instant::now();

            for _ in 0..(args.num_tokens - 1) {
                if kv_caches[0].current_pos >= max_seq_len {
                    break;
                }

                // Throttle delay
                if throttle_delay_ms > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
                }

                // Write token to CPU input
                let current_token = *batch_tokens.last().unwrap();
                unsafe {
                    *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = current_token;
                }
                backend.write_buffer(&mut gen_input_tensor, unsafe {
                    std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
                })?;

                let decode_start = std::time::Instant::now();
                // Use CPU memory when on CPU; GPU memory otherwise.
                // After SwitchHw GPU→CPU, `memory` is still OpenCL memory whose
                // alloc() creates OpenCLBuffer (null as_ptr). Must use
                // cpu_memory_arc for CPU-accessible lazy allocations.
                let effective_mem: &dyn Memory = if is_gpu {
                    gpu_memory_arc.as_deref().unwrap_or_else(|| memory.as_ref())
                } else {
                    cpu_memory_arc.as_ref()
                };
                // Phase α-K ①-d: forward_into → fmt round-trip (decode loop; workspace=Some →
                // forward_gen_fmt, 발산 A 무관). score-feed 활성(H2O 누적).
                KVCache::forward_fmt_roundtrip(&mut kv_caches, |fmts| {
                    model.forward_into_fmt(TransformerModelForwardFmtArgs {
                        input_tokens: &gen_input_tensor,
                        start_pos: batch_start_pos,
                        fmts,
                        backend: &backend,
                        memory: effective_mem,
                        logits_out: &mut logits,
                        x_gen: Some(&mut x_gen),
                        workspace: Some(&mut gen_ws),
                        logits_last_only: false,
                        score_accumulator: hook.score_accumulator(),
                        skip_config: skip_config.as_ref(),
                        importance_collector: None,
                        cache_self_need_scores: false,
                    })
                })?;
                backend.synchronize()?;
                hook.post_decode_step(&mut kv_caches, generated_count);

                let now = std::time::Instant::now();
                let tbt = (now - last_token_time).as_secs_f64() * 1000.0;
                tbt_values_batch.push(tbt);
                last_token_time = now;
                let _forward_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

                // Read logits and sample
                unsafe {
                    let ptr = logits_cpu.as_mut_ptr() as *mut u8;
                    let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
                    backend.read_buffer(&logits, slice)?;
                }

                let next_id = sampling::sample(
                    &mut logits_cpu,
                    &batch_tokens,
                    vocab_size,
                    &sampling_config,
                    None,
                );
                batch_tokens.push(next_id);
                batch_start_pos += 1;
                generated_count += 1;

                if next_id == eos_id && !args.ignore_eos {
                    break;
                }
            }

            eprintln!(
                "[Batch] #{} id={} decode_end ts={:.3}",
                iteration,
                entry.id,
                unix_ts()
            );

            let total_ms = entry_start.elapsed().as_secs_f64() * 1000.0;
            let mean_tbt_ms = if tbt_values_batch.is_empty() {
                0.0
            } else {
                tbt_values_batch.iter().sum::<f64>() / tbt_values_batch.len() as f64
            };

            // Decode generated text (skip prompt tokens)
            let generated_ids = &batch_tokens[prompt_tokens..];
            let text = tokenizer.decode(generated_ids, true).unwrap_or_default();

            // Output JSONL
            let mut result = serde_json::json!({
                "id": entry.id,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_count,
                "ttft_ms": (ttft_ms * 100.0).round() / 100.0,
                "mean_tbt_ms": (mean_tbt_ms * 100.0).round() / 100.0,
                "total_ms": (total_ms * 100.0).round() / 100.0,
                "text": text,
            });
            // Merge ARGUS Step1~6 fields (qcf_caote_max / qcf_per_head /
            // qcf_topk_retention_* / attention_entropy / qcf_beta_amplified_* /
            // qcf_per_layer*) when --enable-qcf-experimental is on.
            if let serde_json::Value::Object(extra_map) = hook.extra_question_fields(&kv_caches)
                && let serde_json::Value::Object(ref mut rmap) = result
            {
                for (k, v) in extra_map {
                    rmap.insert(k, v);
                }
            }
            println!("{}", serde_json::to_string(&result)?);

            eprintln!(
                "[Batch] #{} id={} done: {} tokens, ttft={:.1}ms, tbt={:.1}ms, total={:.1}ms",
                iteration, entry.id, generated_count, ttft_ms, mean_tbt_ms, total_ms
            );

            // === RESET KV CACHE + score accumulator + per-record hook state ===
            hook.reset_caches(&mut kv_caches);

            iteration += 1;
        }

        if !args.prompt_batch_loop {
            break;
        }
    }

    eprintln!("[Batch] Complete: {} iterations", iteration);
    Ok(())
}
