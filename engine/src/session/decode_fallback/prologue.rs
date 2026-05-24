//! Phase 4-4-2.3a: decode prologue 추출 — `bin/generate.rs` L1841~L2295 (~458 LOC).
//!
//! 목적: G3 (LOC 감소) only — main() 가독성 + 후속 sub-sprint 진입 비용 절감.
//! Trait 추상화는 본 sprint scope 외.
//!
//! 본 모듈은 fallback decode 경로의 pre-loop setup을 담당한다:
//! A: Deferred SwitchHw, B: D2O budgets, C: position_birth_step,
//! D: DecodingStart event + profile-events drain, E: decode workspace,
//! F: partition_ws 부착, G: spare bufs + streaming setup,
//! H: UMA Hybrid Attention setup, I: GPU kernel plan.
//!
//! G3-only 정책상 ctx 25+ 필드는 의도된 God Ctx.

use std::sync::Arc;

use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::buffer::DType;
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::layers::workspace::{
    LayerWorkspace, PartitionWorkspace, PartitionWsCell, WorkspaceConfig,
};
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
use crate::models::transformer::TransformerModel;
use crate::pressure::d2o_layer_alloc::D2OVarianceCollector;
use crate::pressure::kv_cache::KVCache;
use crate::session::cli::Args;
use crate::shape::Shape;
use crate::tensor::Tensor;
use tokenizers::Tokenizer;

pub struct DecodePrologueCtx<'a> {
    pub args: &'a Args,
    pub model: &'a mut TransformerModel,
    pub backend: Arc<dyn Backend>,
    pub is_gpu: bool,
    pub memory: Arc<dyn Memory>,
    pub cpu_backend_arc: Arc<dyn Backend>,
    pub cpu_memory_arc: Arc<dyn Memory>,
    pub gpu_backend_arc: Option<Arc<dyn Backend>>,
    pub gpu_memory_arc: Option<Arc<dyn Memory>>,
    pub kv_caches: Vec<KVCache>,
    pub logits: Tensor,
    pub deferred_switch: Option<String>,
    pub variance_collector: Option<D2OVarianceCollector>,
    pub tokens: Vec<u32>,
    pub profiler: Option<crate::observability::profile::InferenceProfiler>,
    pub tokenizer: &'a Tokenizer,
    pub actual_protected_prefix: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub weights_on_gpu: bool,
    pub score_accumulator: Option<AttentionScoreAccumulator>,
}

pub struct DecodePrologueOutput {
    // mut transferred to loop — 필드 순서는 drop order 역순
    // (gen_ws → spare_* → hybrid_scope)
    pub backend: Arc<dyn Backend>,
    pub is_gpu: bool,
    pub kv_caches: Vec<KVCache>,
    pub logits: Tensor,
    pub x_gen: Tensor,
    pub gen_ws: LayerWorkspace,
    pub gen_input_tensor: Tensor,
    pub cpu_gen_input: Tensor,
    pub spare_logits: Option<Tensor>,
    pub spare_xgen: Option<Tensor>,
    pub spare_gen_ws: Option<LayerWorkspace>,
    pub spare_gen_input: Option<Tensor>,
    #[cfg(feature = "opencl")]
    pub gpu_plan: Option<crate::backend::opencl::plan::FullKernelPlan>,
    #[cfg(feature = "opencl")]
    pub gpu_plan_sticky_disabled: bool,
    #[cfg(feature = "opencl")]
    pub partition_active_any: bool,
    pub logits_cpu: Vec<f32>,
    pub sampling_indices: Vec<usize>,
    pub evict_ceiling: Option<usize>,
    pub evict_floor_logged: Option<bool>,
    pub last_applied_partition_ratio: Option<f32>,
    pub d2o_layer_ratios: Option<Vec<(f32, f32)>>,
    pub position_birth_step: Vec<usize>,
    pub profiler: Option<crate::observability::profile::InferenceProfiler>,
    pub printed_len: usize,
    pub stdout: std::io::Stdout,
    pub score_accumulator: Option<AttentionScoreAccumulator>,
    pub tokens: Vec<u32>,
    // RAII guard (must outlive decode loop)
    #[cfg(feature = "opencl")]
    pub hybrid_scope: Option<crate::hybrid_attention::HybridScope>,
}

/// make_partition_gpu_alloc: GPU buffer allocator for tensor partition workspace.
///
/// On OpenCL: allocates `UnifiedBuffer` (CL_MEM_ALLOC_HOST_PTR) + permanent map.
/// On other backends: falls back to `memory.alloc()`.
fn make_partition_gpu_alloc_local<'a>(
    backend: &'a dyn Backend,
    memory: &'a dyn Memory,
) -> impl Fn(usize, DType) -> anyhow::Result<Arc<dyn crate::buffer::Buffer>> + 'a {
    #[cfg(feature = "opencl")]
    let ocl_queue: Option<ocl::Queue> = backend
        .as_any()
        .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
        .map(|b| b.queue.clone());

    #[cfg(not(feature = "opencl"))]
    let _ = backend;

    move |size: usize, dtype: DType| -> anyhow::Result<Arc<dyn crate::buffer::Buffer>> {
        #[cfg(feature = "opencl")]
        if let Some(ref q) = ocl_queue {
            let buf = crate::memory::opencl::unified::UnifiedBuffer::new(q.clone(), size, dtype)?;
            buf.map()?;
            return Ok(Arc::new(buf));
        }
        memory.alloc(size, dtype)
    }
}

pub fn run_decode_prologue(ctx: DecodePrologueCtx<'_>) -> anyhow::Result<DecodePrologueOutput> {
    let DecodePrologueCtx {
        args,
        model,
        mut backend,
        mut is_gpu,
        memory,
        cpu_backend_arc,
        cpu_memory_arc,
        gpu_backend_arc,
        gpu_memory_arc,
        mut kv_caches,
        mut logits,
        deferred_switch,
        variance_collector,
        tokens,
        mut profiler,
        tokenizer,
        actual_protected_prefix,
        vocab_size,
        hidden_size,
        kv_heads,
        head_dim,
        max_seq_len,
        weights_on_gpu,
        score_accumulator,
    } = ctx;

    // ── A: Deferred SwitchHw (GPU↔CPU KV migrate + backend/is_gpu 재할당) ──
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
                // Re-tag weight tensors with CPU backend.
                // UnifiedBuffer (ALLOC_HOST_PTR, mapped) stays valid for CPU.
                eprintln!("[Prefill->Decode] SwitchHw: Switched to CPU.");
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
                eprintln!("[Prefill->Decode] SwitchHw: Switched to GPU.");
            }
            _ => {}
        }
    }

    // ── B: D2O per-layer budget 계산 ──
    let d2o_layer_ratios: Option<Vec<(f32, f32)>> = if let Some(ref collector) = variance_collector
    {
        let budgets = collector.compute_budgets(
            args.d2o_keep_ratio() * args.eviction_target_ratio(),
            (1.0 - args.d2o_keep_ratio()) * args.eviction_target_ratio(),
        );
        log::info!(
            "[D2O] Layer budgets computed: {:?}",
            budgets.iter().map(|(h, r)| h + r).collect::<Vec<_>>()
        );
        Some(budgets)
    } else {
        None
    };

    // ── C: position_birth_step 초기화 (profiler tracking) ──
    let position_birth_step: Vec<usize> = if profiler.is_some() {
        // All prefill tokens have birth_step = 0 (prompt)
        let prompt_len = tokens.len();
        let map = vec![0usize; prompt_len];
        // Register prompt token births + first generated token
        if let Some(ref mut p) = profiler {
            p.scores
                .record_token_births(0, prompt_len, actual_protected_prefix);
        }
        map
    } else {
        Vec::new()
    };

    // ── D: DecodingStart event + profile-events drain ──
    // === GENERATION PHASE ===
    {
        println!("[Profile] Event: DecodingStart");

        // --profile-events / --heartbeat-gpu-profile: drop any events captured
        // during prefill/warmup so the decode-only aggregate is not polluted.
        // Prefill uses the generic `forward` path (no label hints), so without
        // this step all matmul dispatches from prefill would spill into the
        // decode "matmul" bucket and inflate the GPU self-util meter's first
        // heartbeat sample.
        #[cfg(feature = "opencl")]
        if (args.profile_events || args.heartbeat_gpu_profile)
            && let Some(ocl_be) = backend
                .as_any()
                .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            && ocl_be.profile_events_enabled
        {
            backend.synchronize()?;
            ocl_be.flush_and_aggregate_profile()?;
            let _ = ocl_be.take_profile_accum();
            // Prefill-phase GPU busy ns were also fed into the self-util
            // meter via flush_and_aggregate_profile(); drain them so the
            // first heartbeat only reflects decode-phase usage.
            if let Some(m) = ocl_be.gpu_self_meter() {
                use crate::resilience::GpuSelfMeter;
                let _ = m.sample(std::time::Duration::from_secs(1));
            }
            eprintln!("[Profile] prefill/warmup events dropped (decode-only accumulator)");
        }

        // ── E: Decode workspace 할당 ──
        // Pre-allocate workspace for generation
        let q_dim = model.config.num_attention_heads * model.config.head_dim;
        let k_dim = model.config.num_key_value_heads * model.config.head_dim;
        let v_dim = k_dim;
        let ffn_hidden = model.config.intermediate_size;

        // After SwitchHw GPU->CPU, `memory` is still OpenCL memory whose
        // alloc() creates OpenCLBuffer (null as_ptr). Use cpu_memory_arc when on CPU.
        let decode_mem: &dyn Memory = if is_gpu {
            memory.as_ref()
        } else {
            cpu_memory_arc.as_ref()
        };

        // Re-allocate logits on the correct backend after deferred SwitchHw.
        // The outer `logits` was allocated with `memory` (GPU) before the
        // deferred switch. After GPU→CPU, the unmapped UnifiedBuffer has
        // as_ptr() == null → segfault when CPU forward writes logits.
        if !is_gpu && logits.as_ptr().is_null() {
            let new_logits_buf = decode_mem.alloc(vocab_size * 4, DType::F32)?;
            logits = Tensor::new(
                Shape::new(vec![1, 1, vocab_size]),
                new_logits_buf,
                backend.clone(),
            );
        }

        let x_gen_buf = decode_mem.alloc(hidden_size * 4, DType::F32)?;
        let x_gen = Tensor::new(
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
                max_seq_len: args.max_seq_len, // Use context window size
            },
            decode_mem,
            backend.clone(),
        )?;

        // ── F: Partition workspace 부착 (zero-copy residual mapping) ──
        // Attach partition workspace if tensor partition is active.
        // Use UnifiedBuffer (ALLOC_HOST_PTR) for zero-copy merge (see batch path above).
        let layer0_partition_probe = model.layers[0].load_weights();
        if let Some(ref partition_ctx) = layer0_partition_probe.partition_ctx {
            let gpu_alloc = make_partition_gpu_alloc_local(&*backend, decode_mem);

            // Zero-copy residual (see line 1807 block for rationale).
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
                partition_ctx.gate.split_row,
                partition_ctx.up.split_row,
                ffn_hidden,
                hidden_size,
                &gpu_alloc,
                backend.clone(),
                cpu_backend_arc.clone(),
            )?)));
        }

        // ── G: 단일 토큰 입력 텐서 + spare 버퍼 + streaming setup ──

        // Single token CPU tensor for generation loop
        let cpu_gen_indices_buf = Galloc::new().alloc(4, DType::U8)?;
        let cpu_gen_input = Tensor::new(
            Shape::new(vec![1, 1]),
            cpu_gen_indices_buf,
            Arc::new(CpuBackend::new()),
        );

        // Pre-allocate input tensor for decode loop (avoids per-token alloc)
        let gpu_gen_input_buf = decode_mem.alloc(4, DType::U8)?;
        let gen_input_tensor =
            Tensor::new(Shape::new(vec![1, 1]), gpu_gen_input_buf, backend.clone());

        // Pre-allocate CPU spare decode buffers for zero-alloc GPU→CPU SwitchHw.
        // Both sets (GPU active + CPU spare) stay alive for the process lifetime,
        // enabling instant swap without allocation/deallocation during switch.
        // This prevents Samsung LMKD from killing the process due to RSS spike.
        let (spare_logits, spare_xgen, spare_gen_ws, spare_gen_input) =
            if is_gpu && args.resilience_prealloc_switch {
                let cpu_lb = cpu_memory_arc.alloc(vocab_size * 4, DType::F32)?;
                let cpu_xb = cpu_memory_arc.alloc(hidden_size * 4, DType::F32)?;
                let cpu_gi = cpu_memory_arc.alloc(4, DType::U8)?;
                eprintln!("[Switch] Pre-allocated CPU spare buffers for zero-alloc SwitchHw");
                (
                    Some(Tensor::new(
                        Shape::new(vec![1, 1, vocab_size]),
                        cpu_lb,
                        cpu_backend_arc.clone(),
                    )),
                    Some(Tensor::new(
                        Shape::new(vec![1, 1, hidden_size]),
                        cpu_xb,
                        cpu_backend_arc.clone(),
                    )),
                    Some(LayerWorkspace::new(
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
                        cpu_backend_arc.clone(),
                    )?),
                    Some(Tensor::new(
                        Shape::new(vec![1, 1]),
                        cpu_gi,
                        cpu_backend_arc.clone(),
                    )),
                )
            } else {
                (None, None, None, None)
            };

        // Streaming setup
        use std::io::Write;
        let mut stdout = std::io::stdout();
        let mut _printed_len = 0;

        // Print initial tokens (prompt + first generated)
        let initial_text = tokenizer.decode(&tokens, true).unwrap_or_default();
        print!("{}", initial_text);
        _printed_len = initial_text.len();
        stdout.flush().ok();

        // ── H: UMA Hybrid Attention setup (Stage C) ──
        // LLMRS_ATTN_HYBRID_KV_FRAC=X 가 설정되고 gating 조건이 모두 충족되면
        // 공용 GPU 스크래치 버퍼를 할당하고 HybridScope를 install한다. 스코프
        // 객체는 decode 루프 종료까지 살아있어야 하므로 `_hybrid_scope`로 바인드.
        // Gating 실패 시 reason을 stderr로 한 번 찍고 스킵.
        #[cfg(feature = "opencl")]
        let _hybrid_scope = {
            use crate::hybrid_attention::{self, HybridAttnSetup};
            use crate::kv_cache_ops::KVLayout;
            match HybridAttnSetup::from_env() {
                Some(kv_frac) => {
                    let backend_is_opencl = backend.name() == "OpenCL";
                    let kv_is_f16 = args.kv_type == "f16";
                    let head_dim_val = model.config.head_dim;
                    let head_dim_ok = head_dim_val == 64 || head_dim_val == 128;
                    let n_heads_q = model.config.num_attention_heads;
                    let n_kv_heads = model.config.num_key_value_heads;
                    let is_gqa = n_kv_heads < n_heads_q;
                    let partition_off =
                        args.tensor_partition <= 0.0 || args.tensor_partition >= 1.0;
                    let eviction_compatible =
                        args.eviction_policy() != "kivi" && args.eviction_policy() != "qcf";
                    let layout_ok = kv_caches
                        .first()
                        .map(|c| c.layout() == KVLayout::HeadMajor)
                        .unwrap_or(false);

                    let gate_ok = backend_is_opencl
                        && kv_is_f16
                        && head_dim_ok
                        && is_gqa
                        && partition_off
                        && eviction_compatible
                        && layout_ok;

                    if !gate_ok {
                        let reason = if !backend_is_opencl {
                            "backend is not OpenCL"
                        } else if !kv_is_f16 {
                            "kv dtype must be f16"
                        } else if !head_dim_ok {
                            "head_dim must be 64 or 128"
                        } else if !is_gqa {
                            "requires GQA (n_kv_heads < n_heads_q)"
                        } else if !partition_off {
                            "FFN tensor partition is active"
                        } else if !eviction_compatible {
                            "incompatible eviction policy (kivi/qcf)"
                        } else {
                            "KV layout must be HeadMajor"
                        };
                        eprintln!(
                            "[hybrid-attn] LLMRS_ATTN_HYBRID_KV_FRAC={} ignored: {}",
                            kv_frac, reason
                        );
                        None
                    } else {
                        // Map KV/Q/out_attn/residual UnifiedBuffer들을 CPU가 접근
                        // 가능하도록 전부 매핑한다. UMA 특성상 map은 주소만
                        // 고정하고 추가 복사는 하지 않는다. Plan execution에
                        // 들어가기 전에 한 번만 호출되면 충분.
                        let mut map_err: Option<anyhow::Error> = None;
                        for c in kv_caches.iter() {
                            if let Err(e) = c.k_buffer.buffer().map_for_cpu() {
                                map_err = Some(e);
                                break;
                            }
                            if let Err(e) = c.v_buffer.buffer().map_for_cpu() {
                                map_err = Some(e);
                                break;
                            }
                        }
                        if map_err.is_none() {
                            if let Err(e) = gen_ws.q.buffer().map_for_cpu() {
                                map_err = Some(e);
                            } else if let Err(e) = gen_ws.out_attn.buffer().map_for_cpu() {
                                map_err = Some(e);
                            }
                        }
                        if let Some(e) = map_err {
                            eprintln!("[hybrid-attn] failed to map UMA buffers: {} — skipping", e);
                            None
                        } else {
                            let ocl_be = backend
                                .as_any()
                                .downcast_ref::<crate::backend::opencl::OpenCLBackend>();
                            match ocl_be {
                                Some(ob) => match HybridAttnSetup::new_for_decode(
                                    &ob.queue,
                                    kv_frac,
                                    n_heads_q,
                                    head_dim_val,
                                ) {
                                    Ok(setup) => {
                                        eprintln!(
                                            "[hybrid-attn] enabled: kv_frac={} n_heads_q={} head_dim={}",
                                            kv_frac, n_heads_q, head_dim_val
                                        );
                                        Some(hybrid_attention::install(Arc::new(setup)))
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "[hybrid-attn] setup allocation failed: {} — skipping",
                                            e
                                        );
                                        None
                                    }
                                },
                                None => None,
                            }
                        }
                    }
                }
                None => None,
            }
        };

        // ── I: GPU kernel plan 초기 빌드 + sticky disable + decode buffer pre-alloc + evict ceiling state ──

        // Build GPU kernel plan for decode (OpenCL only, lazy rebuild on invalidation)
        // Disable for Gemma3: plan doesn't include QK-norm, post-norm, gelu_tanh_mul
        // Disable when tensor partition is active: plan bypasses forward_gen's
        // partition path entirely (plan = pure GPU chain, no CPU co-execution).
        //
        // Score accumulator coexistence: when a CPU `score_accumulator` is
        // active (H2O/D2O/Sliding/CAOTE eviction), the plan may still be used
        // as long as the paired GPU `gpu_score_acc` is active.  `build_plan`
        // then selects the legacy attention kernel (flash attn has no score
        // output) and pre-binds the GPU score buffer into arg 4. Per-layer
        // `reduce_layer` + post-pass `end_step` are driven by
        // `FullKernelPlan::execute` so CPU readback happens only at eviction
        // time (see `sync_to_cpu` further down).
        #[cfg(feature = "opencl")]
        let accumulator_compatible_with_plan = {
            let has_cpu_acc = score_accumulator.is_some();
            let gpu_acc_active = backend
                .as_any()
                .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
                .and_then(|ob| ob.gpu_score_acc())
                .is_some_and(|acc| acc.is_active());
            !has_cpu_acc || gpu_acc_active
        };
        // Partition is now routed through `build_partitioned_layer_plan` inside
        // `build_plan`, so the old `partition_ctx.is_none()` gate has been
        // removed (see ENG-ALG-200 / arch A.6.1). When partition + plan are
        // both unavailable for a layer (e.g. `LLMRS_PARTITION_PLAN=0`), the
        // builder returns `Err` and the caller falls back to forward_gen.
        #[cfg(feature = "opencl")]
        let gpu_plan = if backend.name() == "OpenCL"
            && !args.profile
            && !args.no_gpu_plan
            && accumulator_compatible_with_plan
            && model.config.arch != crate::models::config::ModelArch::Gemma3
            && !args.swap_intra_forward
            && !args.swap_layer_immediate
            && !args.swap_phase_aware
        {
            model.build_plan(&x_gen, &logits, &gen_ws, &mut kv_caches, &backend)
        } else {
            None
        };
        // Sticky disable: when the initial `build_plan` returns `None` and
        // partition is active, the cause is almost always the opt-in gate
        // (`LLMRS_PARTITION_PLAN=0` default on Adreno, 2026-04-21). Retrying
        // every token spams `build_plan` (~100 ms/token overhead) for no
        // benefit. Lock the disable on the first miss and keep forward_gen.
        // `execute_plan` resetting `gpu_plan = None` for KV-resize
        // invalidation still takes the rebuild path on the next token.
        #[cfg(feature = "opencl")]
        let partition_active_any = model
            .layers
            .iter()
            .any(|s| s.load_weights().partition_ctx.is_some());
        #[cfg(feature = "opencl")]
        let gpu_plan_sticky_disabled = partition_active_any && gpu_plan.is_none();

        // Pre-allocate decode buffers (reused across tokens)
        let logits_cpu = vec![0.0f32; vocab_size];
        let sampling_indices: Vec<usize> = (0..vocab_size).collect();

        // Ceiling for sticky eviction: records current_pos at first eviction trigger.
        // Subsequent evictions use ceiling * ratio as a fixed target to prevent cascade
        // (e.g. cache 33 → 16 → 8 → ... when target_ratio is applied to ever-shrinking pos).
        let evict_ceiling: Option<usize> = None;
        let evict_floor_logged: Option<bool> = None;

        // Sticky cache for last-applied partition ratio. The executor re-delivers
        // `plan.partition_ratio = Some(sticky)` on every poll (ISSUE-5 fix), so
        // without this guard the consumer below would re-split 84 weights and
        // rebuild the GPU plan on every decode tick (verify v2 REGRESSION-A:
        // q4 enable +102% → +3859% TBT). Seeded from CLI-time partition so the
        // first sticky re-delivery is a no-op when nothing changed.
        let last_applied_partition_ratio: Option<f32> =
            if args.tensor_partition > 0.0 && args.tensor_partition < 1.0 {
                Some(args.tensor_partition)
            } else {
                None
            };

        Ok(DecodePrologueOutput {
            backend,
            is_gpu,
            kv_caches,
            logits,
            tokens,
            x_gen,
            gen_ws,
            gen_input_tensor,
            cpu_gen_input,
            spare_logits,
            spare_xgen,
            spare_gen_ws,
            spare_gen_input,
            #[cfg(feature = "opencl")]
            gpu_plan,
            #[cfg(feature = "opencl")]
            gpu_plan_sticky_disabled,
            #[cfg(feature = "opencl")]
            partition_active_any,
            logits_cpu,
            sampling_indices,
            evict_ceiling,
            evict_floor_logged,
            last_applied_partition_ratio,
            d2o_layer_ratios,
            position_birth_step,
            profiler,
            printed_len: _printed_len,
            stdout,
            score_accumulator,
            #[cfg(feature = "opencl")]
            hybrid_scope: _hybrid_scope,
        })
    }
}
