//! Phase 4-4-2.1: chunked prefill block extracted from `bin/generate.rs` L1798~2375.
//!
//! 목적: G3 (LOC 감소) only — main() 가독성 + 후속 sub-sprint 진입 비용 절감.
//! Trait 추상화 / collector 흡수는 본 sprint scope 외 (handoff §2.5 참조).
//!
//! 본 모듈은 fallback prefill 경로의 self-contained 추출이며, 동작 변경 0.
//! 1:1 cut/paste from generate.rs HEAD `7f693160`.
//!
//! G3-only 정책상 ctx 35+ 필드는 의도된 God Ctx — sub-sprint 4-4-2.2~4까지
//! 동일 패턴 유지. trait 분해는 4 sub-sprint 종료 후 별도 결정 gate.

use std::sync::Arc;

use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::buffer::DType;
use crate::core::cache_manager::CacheManager;
use crate::core::kv_cache::KVCache;
use crate::core::rss_trace::{io_trace, rss_trace};
use crate::inference::sampling::{self, SamplingConfig};
use crate::inference::skip_config::SkipConfig;
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
use crate::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use crate::resilience::{CommandExecutor, KVSnapshot};
use crate::session::cli::Args;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Closure for building LayerSwapEstimate. Binary owns the impl (uses
/// `WeightSwapDecider` + `read_allow_boundary_env`). Closure-based DI keeps
/// the binary as the single source of truth for swap estimation policy.
pub type LayerSwapEstimator<'a> = Box<
    dyn FnMut(
            &TransformerModel,
            Option<&crate::qcf::ImportanceTable>,
        ) -> Option<llm_shared::LayerSwapEstimate>
        + 'a,
>;

pub struct PrefillCtx<'a> {
    // ── Borrowed refs ─────────────────────────────────────────────
    pub args: &'a Args,
    pub model: &'a mut TransformerModel,
    pub cache_manager: &'a CacheManager,
    pub sampling_config: &'a SamplingConfig,

    // ── Arc clones (Backend/Memory) ───────────────────────────────
    pub backend: Arc<dyn Backend>,
    pub memory: Arc<dyn Memory>,
    pub cpu_backend_arc: Arc<dyn Backend>,
    pub cpu_memory_arc: Arc<dyn Memory>,

    // ── Constants ─────────────────────────────────────────────────
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub max_seq_len: usize,
    pub actual_protected_prefix: usize,
    pub auto_eviction: bool,
    pub start_time: std::time::Instant,

    // ── Closure DI ────────────────────────────────────────────────
    pub layer_swap_estimator: LayerSwapEstimator<'a>,

    // ── Owned mut state (threaded in/out) ─────────────────────────
    pub kv_caches: Vec<KVCache>,
    pub tokens: Vec<u32>,
    pub start_pos: usize,
    pub skip_config: Option<SkipConfig>,
    pub last_skip_ratio: Option<f32>,
    pub throttle_delay_ms: u64,
    pub command_executor: Option<CommandExecutor>,
}

pub struct PrefillOutput {
    // ── Owned state ───────────────────────────────────────────────
    pub kv_caches: Vec<KVCache>,
    pub tokens: Vec<u32>,
    pub start_pos: usize,
    pub profiler: Option<crate::profile::InferenceProfiler>,
    pub variance_collector: Option<crate::core::pressure::d2o_layer_alloc::D2OVarianceCollector>,
    pub importance_table_for_swap: Option<crate::qcf::ImportanceTable>,
    pub collector_armed: bool,
    pub deferred_switch: Option<String>,
    pub skip_config: Option<SkipConfig>,
    pub last_skip_ratio: Option<f32>,
    pub throttle_delay_ms: u64,
    pub command_executor: Option<CommandExecutor>,

    // ── Tensors for decode (created in prefill) ───────────────────
    pub logits: Tensor,
    pub eos_id: u32,

    // ── Timing ────────────────────────────────────────────────────
    pub ttft_ms: f64,
    pub last_token_time: std::time::Instant,
    pub prefill_forward_ms: f64,
    pub prefill_pure_fwd_ms: f64,
}

pub fn run_chunked_prefill(ctx: PrefillCtx<'_>) -> anyhow::Result<PrefillOutput> {
    let PrefillCtx {
        args,
        model,
        cache_manager,
        sampling_config,
        backend,
        memory,
        cpu_backend_arc,
        cpu_memory_arc,
        vocab_size,
        hidden_size,
        max_seq_len,
        actual_protected_prefix,
        auto_eviction,
        start_time,
        mut layer_swap_estimator,
        mut kv_caches,
        mut tokens,
        mut start_pos,
        mut skip_config,
        mut last_skip_ratio,
        mut throttle_delay_ms,
        mut command_executor,
    } = ctx;

    // Inference profiler (activated by either --profile or --profile-events).
    // Declared before prefill so PrefillOpProfiler can be populated.
    //
    // --profile-events uses the same InferenceProfiler container (ops/json
    // export) but feeds it via OpProfiler::merge_from_events() instead of
    // the legacy per-op synchronize+wall-clock path.
    let mut profiler = if args.profile || args.profile_events {
        Some(crate::profile::InferenceProfiler::new(
            crate::profile::ProfileConfig {
                score_snapshot_interval: args.profile_interval,
                track_per_head: args.profile_per_head,
                enabled_probes: args.profile_probes.split(',').map(String::from).collect(),
                output_dir: std::path::PathBuf::from(&args.profile_dir),
            },
        ))
    } else {
        None
    };

    // Pre-allocate generation buffers
    let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        logits_buf,
        backend.clone(),
    );

    // Cache EOS token ID from config.json (model-agnostic)
    let eos_id = model.config.eos_token_id;

    // === WARMUP: trigger DVFS ramp-up before timed prefill ===
    // Forward 1회 + 50ms all-core spin → governor max clock. KV cache 위치 리셋까지 포함.
    // Env overrides: LLMRS_SKIP_WARMUP, LLMRS_WARMUP_TOKENS (session::warmup 참조).
    crate::session::warmup::run_warmup(
        model,
        &backend,
        &memory,
        &mut kv_caches,
        &tokens,
        vocab_size,
    )?;

    // D2O layer-level allocation: create variance collector before prefill.
    // Only active when --eviction-policy d2o and --d2o-layer-alloc are both set.
    let mut variance_collector = if args.d2o_layer_alloc() && args.eviction_policy() == "d2o" {
        Some(
            crate::core::pressure::d2o_layer_alloc::D2OVarianceCollector::new(
                model.config.num_hidden_layers,
                model.config.num_key_value_heads,
                model.config.num_attention_heads,
                model.config.head_dim,
                tokens.len(),
            ),
        )
    } else {
        None
    };

    // Weight swap state (ENG-ALG-218 + ENG-ALG-214-ROUTE).
    //
    // `importance_table_for_swap`: most-recently collected per-layer
    // importance table from an on-demand prefill measurement.  `None`
    // until the first `RequestQcf` prefill completes.
    //
    // `collector_armed`: true when a `RequestQcf` has been received and
    // we are waiting for the next prefill to inject `ImportanceCollector`.
    // This is a lightweight bool; the actual collector lives on the stack
    // during prefill (not stored here).
    let mut importance_table_for_swap: Option<crate::qcf::ImportanceTable> = None;
    let mut collector_armed = false;

    // === PREFILL PHASE ===
    let mut deferred_switch: Option<String> = None;
    let prefill_forward_ms;
    let prefill_pure_fwd_ms;
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

        // Determine effective chunk size.
        // 0 or >= process_len → use full prompt as single chunk (original behaviour).
        // When resilience is enabled, auto-chunk at 256 so that chunk boundaries
        // serve as checkpoints for SwitchHw / Throttle / LayerSkip commands.
        // When GPU backend and prefill_chunk_size == 0, auto-derive a safe chunk
        // size from max_single_alloc() to avoid CL_INVALID_BUFFER_SIZE (-61).
        let auto_gpu_chunk: Option<usize> = if args.prefill_chunk_size == 0 && backend.is_gpu() {
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
        let chunk_size = if args.prefill_chunk_size > 0 && args.prefill_chunk_size < process_len {
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
        if chunked {
            eprintln!(
                "[Prefill] Chunked mode: {} tokens in chunks of {}",
                process_len, chunk_size
            );
        }

        // Dynamic prefill policy: start from CLI values, updated by SetPrefillPolicy.
        let mut effective_chunk_size = chunk_size;
        let mut effective_yield_ms = args.prefill_yield_ms;
        let mut effective_cpu_chunk_size = args.prefill_cpu_chunk_size;

        // Reusable logits buffer: [1, 1, vocab_size] when chunked, else [1, process_len, vocab_size].
        // Chunked mode always uses logits_last_only=true so only 1 position is written per chunk.
        let (logits_shape, logits_buf_size) = if chunked {
            (Shape::new(vec![1, 1, vocab_size]), vocab_size * 4)
        } else {
            (
                Shape::new(vec![1, process_len, vocab_size]),
                process_len * vocab_size * 4,
            )
        };
        let prefill_logits_buf = memory.alloc(logits_buf_size, DType::F32)?;
        let mut prefill_logits = Tensor::new(logits_shape, prefill_logits_buf, backend.clone());

        let prefill_timer = std::time::Instant::now();
        let mut prefill_pure_fwd_ms_acc: f64 = 0.0;
        let total_chunks = process_len.div_ceil(chunk_size);

        // Report prefill start to resilience manager.
        if let Some(executor) = &mut command_executor {
            executor.set_prefill_state("prefill", 0, process_len);
        }

        // ENG-ALG-218: if collector is armed, prepare a collector for this prefill.
        // Armed by `RequestQcf` handler in decode loop; collector is injected into
        // the last prefill chunk so it captures the final contextual activation state.
        let mut on_demand_collector: Option<crate::qcf::ImportanceCollector> = if collector_armed {
            Some(crate::qcf::ImportanceCollector::new())
        } else {
            None
        };
        if collector_armed {
            collector_armed = false; // consume the flag; armed at most once per prefill
        }

        let mut chunk_start = 0;
        let mut chunk_idx = 0usize;
        while chunk_start < process_len {
            // Guard: effective_chunk_size must be at least 1.
            let ecs = effective_chunk_size.max(1);
            let chunk_end = (chunk_start + ecs).min(process_len);
            let chunk_tokens = &tokens[chunk_start..chunk_end];
            let chunk_len = chunk_tokens.len();

            // ENG-ALG-218: inject collector only on the last prefill chunk.
            // Earlier chunks have partial seq_len; the last chunk captures final
            // contextual state which is most representative for per-layer importance.
            let is_last_chunk = chunk_end >= process_len;
            let inject_collector = is_last_chunk && on_demand_collector.is_some();

            let chunk_trace = std::env::var("LLMRS_PREFILL_CHUNK_MS").is_ok();
            let t_chunk_start = std::time::Instant::now();

            // Build CPU input tensor for this chunk.
            let cpu_chunk_buf = Galloc::new().alloc(chunk_len * 4, DType::U8)?;
            unsafe {
                let ptr = cpu_chunk_buf.as_mut_ptr() as *mut u32;
                std::ptr::copy_nonoverlapping(chunk_tokens.as_ptr(), ptr, chunk_len);
            }
            let cpu_chunk_tensor = Tensor::new(
                Shape::new(vec![1, chunk_len]),
                cpu_chunk_buf,
                Arc::new(CpuBackend::new()),
            );
            let input_tensor = backend.copy_from(&cpu_chunk_tensor)?;
            let t_setup_end = std::time::Instant::now();

            // RoPE position for this chunk: start_pos (0 during prefill) + offset within prompt.
            let chunk_start_pos = start_pos + chunk_start;

            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos: chunk_start_pos,
                kv_caches: &mut kv_caches,
                backend: &backend,
                memory: memory.as_ref(),
                logits_out: &mut prefill_logits,
                x_gen: None,
                workspace: None,
                score_accumulator: None, // No score tracking during prefill
                profiler: profiler.as_mut().map(|p| &mut p.ops),
                skip_config: None,
                importance_collector: if inject_collector {
                    on_demand_collector.as_mut()
                } else {
                    None
                },
                // Chunked mode: only the last position's logits needed (saves GPU memory).
                // Non-chunked: write all positions (original behaviour).
                logits_last_only: chunked,
                variance_collector: variance_collector.as_mut(),
                prefill_workspace: None,

                layer_boundary_hook: None,
            })?;
            backend.synchronize()?;
            let t_fwd_end = std::time::Instant::now();
            let fwd_ms = (t_fwd_end - t_setup_end).as_secs_f64() * 1000.0;
            prefill_pure_fwd_ms_acc += fwd_ms;
            if chunk_trace {
                let setup_ms = (t_setup_end - t_chunk_start).as_secs_f64() * 1000.0;
                let total_ms = (t_fwd_end - t_chunk_start).as_secs_f64() * 1000.0;
                eprintln!(
                    "[PREFILL_CHUNK] idx={} start_pos={} len={} setup_ms={:.2} fwd_ms={:.2} total_ms={:.2}",
                    chunk_idx, chunk_start_pos, chunk_len, setup_ms, fwd_ms, total_ms
                );
            }

            // Immediately release the GPU input buffer for this chunk.
            drop(input_tensor);

            chunk_start = chunk_end;

            // Inter-chunk yield: sleep after GPU chunk to release compute for other processes.
            if effective_yield_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(effective_yield_ms as u64));
            }

            // CPU interleave: process next chunk on CPU while GPU is free.
            // Invariant: the last chunk must be processed by GPU so that
            // prefill_logits (GPU buffer) is valid at the end.
            if effective_cpu_chunk_size > 0 && chunk_start < process_len {
                let remaining = process_len - chunk_start;
                // Only run CPU chunk if enough tokens remain for GPU to handle
                // at least one more chunk afterwards.
                if remaining > effective_cpu_chunk_size {
                    // Flush GPU caches to main memory before CPU reads KV buffers.
                    // On ARM UMA, clFinish() alone may not flush GPU L1/L2 cache.
                    // map_for_cpu() calls clEnqueueMapBuffer which ensures coherence.
                    for kv in kv_caches.iter() {
                        kv.k_buffer.buffer().map_for_cpu()?;
                        kv.v_buffer.buffer().map_for_cpu()?;
                    }

                    let cpu_end =
                        (chunk_start + effective_cpu_chunk_size).min(process_len.saturating_sub(1));
                    if cpu_end > chunk_start {
                        let cpu_tokens = &tokens[chunk_start..cpu_end];
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

                        let cpu_chunk_start_pos = start_pos + chunk_start;

                        // CPU prefill logits: use a separate CPU buffer to avoid writing
                        // to GPU prefill_logits. These intermediate logits are discarded.
                        let cpu_logits_buf = cpu_memory_arc.alloc(vocab_size * 4, DType::F32)?;
                        let mut cpu_logits = Tensor::new(
                            Shape::new(vec![1, 1, vocab_size]),
                            cpu_logits_buf,
                            cpu_backend_arc.clone(),
                        );

                        model.forward_into(TransformerModelForwardArgs {
                            input_tokens: &cpu_in_tensor,
                            start_pos: cpu_chunk_start_pos,
                            kv_caches: &mut kv_caches,
                            backend: &cpu_backend_arc,
                            memory: cpu_memory_arc.as_ref(),
                            logits_out: &mut cpu_logits,
                            x_gen: None,
                            workspace: None,
                            score_accumulator: None,
                            profiler: None,
                            skip_config: None,
                            importance_collector: None,
                            logits_last_only: true,
                            variance_collector: None,
                            prefill_workspace: None,

                            layer_boundary_hook: None,
                        })?;
                        // No backend.synchronize() needed — CPU forward is synchronous.
                        drop(cpu_in_tensor);
                        drop(cpu_logits);

                        chunk_start = cpu_end;
                    }
                }
                // else: remaining tokens fit in one GPU chunk → GPU finishes.
            }

            // ── Prefill resilience checkpoint (chunk boundary) ──
            // Poll CommandExecutor between chunks to handle SwitchHw, Throttle,
            // and LayerSkip commands mid-prefill. Only active in chunked mode.
            if chunked && let Some(executor) = &mut command_executor {
                let kv_snap = KVSnapshot {
                    total_bytes: kv_caches
                        .iter()
                        .map(|c| (c.k_buffer.buffer().size() + c.v_buffer.buffer().size()) as u64)
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
                if let Some(v) = plan.prefill_chunk_size {
                    effective_chunk_size = v;
                    eprintln!("[Prefill] Policy: chunk_size -> {}", v);
                }
                if let Some(v) = plan.prefill_yield_ms {
                    effective_yield_ms = v;
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

        // ENG-ALG-218: finalize on-demand ImportanceCollector after prefill completes.
        // INV-128: this block always runs (normal fall-through from the while loop),
        // so QcfEstimate is guaranteed to be sent when the prefill completes successfully.
        // For panics/early-return paths the caller-side Drop guard is the safety net.
        if let Some(collector) = on_demand_collector.take() {
            let table: crate::qcf::ImportanceTable = collector.build();
            let layer_swap = (layer_swap_estimator)(model, Some(&table));
            if let Some(executor) = &mut command_executor {
                executor.send_qcf_estimate(llm_shared::QcfEstimate {
                    estimates: std::collections::HashMap::new(),
                    layer_swap,
                });
                log::debug!("[QCF] QcfEstimate sent after prefill finalization (ENG-ALG-218)");
            }
            importance_table_for_swap = Some(table);
        }

        prefill_forward_ms = prefill_timer.elapsed().as_secs_f64() * 1000.0;
        prefill_pure_fwd_ms = prefill_pure_fwd_ms_acc;

        // Report transition to decode phase.
        if let Some(executor) = &mut command_executor {
            executor.set_prefill_state("decode", 0, 0);
        }

        // Auto-eviction after prefill (sliding window only, non-experiment mode)
        if auto_eviction {
            cache_manager.maybe_evict(&mut kv_caches).ok();
        }

        // Sticky eviction at prefill→decode boundary.
        // If a KvEvict directive arrived during prefill, executor holds a sticky evict_plan.
        // Execute it now (before decode starts) to reduce attention work from the first decode step.
        // Score-based methods (H2O/D2O) are not available here — falls back to force_evict.
        if let Some(ref mut exec) = command_executor {
            let kv_snap = KVSnapshot {
                total_bytes: kv_caches
                    .iter()
                    .map(|c| (c.k_buffer.buffer().size() + c.v_buffer.buffer().size()) as u64)
                    .sum(),
                total_tokens: kv_caches[0].current_pos,
                capacity: kv_caches[0].capacity(),
                protected_prefix: actual_protected_prefix,
                kv_dtype: "f16".to_string(),
                eviction_policy: args.eviction_policy().to_string(),
                skip_ratio: 0.0,
            };
            let plan = exec.poll(&kv_snap);
            if let Some(evict) = &plan.evict {
                let effective_ratio = args.experiment_eviction_ratio.unwrap_or(evict.target_ratio);
                if effective_ratio > 0.0 {
                    let current_pos = kv_caches[0].current_pos;
                    // Use current_pos as ceiling (first and only boundary eviction).
                    let tgt_raw = (current_pos as f32 * effective_ratio).max(1.0) as usize;
                    let target_pos = tgt_raw.max(args.min_kv_cache());
                    if current_pos > target_pos {
                        // adjusted_ratio so force_evict(current_pos * adjusted) == target_pos.
                        let adjusted_ratio = target_pos as f32 / current_pos as f32;
                        // Dispatch by evict method (same as decode loop).
                        // Scores are unavailable at prefill→decode boundary, so
                        // D2O and score-based H2O fall back to force_evict.
                        let result = if evict.method == crate::resilience::EvictMethod::Streaming {
                            if let Some(ref sp) = evict.streaming_params {
                                let policy =
                                    crate::core::eviction::streaming_llm::StreamingLLMPolicy::new(
                                        sp.sink_size,
                                        sp.window_size,
                                    );
                                cache_manager.force_evict_by_policy_ref(
                                    &policy,
                                    &mut kv_caches,
                                    adjusted_ratio,
                                    crate::core::cache_manager::ScoreContext::None,
                                )
                            } else {
                                cache_manager.force_evict(&mut kv_caches, adjusted_ratio)
                            }
                        } else {
                            cache_manager.force_evict_by_policy(
                                evict.method,
                                &mut kv_caches,
                                adjusted_ratio,
                                crate::core::cache_manager::ScoreContext::None,
                            )
                        };
                        match result {
                            Ok(r) if r.evicted => {
                                eprintln!(
                                    "[Prefill→Decode] Evicted {} tokens (pos: {} → {})",
                                    r.tokens_removed,
                                    r.new_pos + r.tokens_removed,
                                    r.new_pos
                                );
                            }
                            Err(e) => eprintln!("[Prefill→Decode] Eviction error: {}", e),
                            _ => {}
                        }
                    }
                }
            }
        }

        // Sample last token — read logits from the last chunk's output.
        // When chunked: prefill_logits is [1,1,vocab_size], last_logits = the only row.
        // When not chunked: prefill_logits is [1,process_len,vocab_size], take last row.
        let mut last_logits = vec![0.0f32; vocab_size];
        unsafe {
            let ptr = last_logits.as_mut_ptr() as *mut u8;
            let byte_len = vocab_size * 4;
            if chunked {
                // Single-row buffer; read all of it.
                let slice = std::slice::from_raw_parts_mut(ptr, byte_len);
                backend.read_buffer(&prefill_logits, slice)?;
            } else {
                // Multi-row buffer; read only the last row.
                // read_buffer reads from offset 0, so we read the full buffer and
                // then take the last vocab_size elements.
                let mut logits_cpu = vec![0.0f32; process_len * vocab_size];
                let full_ptr = logits_cpu.as_mut_ptr() as *mut u8;
                let full_slice = std::slice::from_raw_parts_mut(full_ptr, logits_cpu.len() * 4);
                backend.read_buffer(&prefill_logits, full_slice)?;
                let start_idx = (process_len - 1) * vocab_size;
                last_logits.copy_from_slice(&logits_cpu[start_idx..start_idx + vocab_size]);
            }
        }

        let next_token_id =
            sampling::sample(&mut last_logits, &tokens, vocab_size, sampling_config, None);

        let ttft_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "Prefill: {:.2} ms ({} tokens, {:.1} tok/s)",
            prefill_forward_ms,
            process_len,
            process_len as f64 / (prefill_forward_ms / 1000.0),
        );
        eprintln!(
            "Prefill(pure fwd): {:.2} ms ({} tokens, {:.1} tok/s) [sync'd forward only, comparable to llama-bench pp]",
            prefill_pure_fwd_ms,
            process_len,
            process_len as f64 / (prefill_pure_fwd_ms / 1000.0),
        );
        let last_token_time = std::time::Instant::now();

        tokens.push(next_token_id);
        start_pos += process_len;
        // T2: first forward pass (prefill) complete, KV cache filled.
        rss_trace("prefill_done");
        io_trace("prefill_done");

        Ok(PrefillOutput {
            kv_caches,
            tokens,
            start_pos,
            profiler,
            variance_collector,
            importance_table_for_swap,
            collector_armed,
            deferred_switch,
            skip_config,
            last_skip_ratio,
            throttle_delay_ms,
            command_executor,
            logits,
            eos_id,
            ttft_ms,
            last_token_time,
            prefill_forward_ms,
            prefill_pure_fwd_ms,
        })
    }
}
