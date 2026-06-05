//! argus-bench AB-1: resilience eviction 지원 [`DecodeLoop`] 조립자.
//!
//! [`build_standard_loop`](super::build_standard_loop) 와 동일한 ModelForward·
//! sampler·resilience 골격에, CLI `eviction <policy>` 로 구성한 [`CacheManager`]
//! 를 [`DecodeLoopBuilder::with_cache_manager`] 로 주입한다. decode 루프가
//! `plan.evict` (resilience KvEvict directive) 를 받으면 `forward.try_evict` 로
//! mid-decode prune 한다.
//!
//! happy path(AB-0 시나리오, `eviction=none`) 는 `cache_manager=None` 으로
//! 흘러 [`build_standard_loop`] 와 동등하게 동작한다.

use std::sync::Arc;

use anyhow::Result;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::inference::sampling::SamplingConfig;
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::pressure::cache_manager::CacheManager;
use crate::pressure::d2o_handler::{D2OConfig, D2OHandler};
use crate::pressure::eviction::h2o::H2OPolicy;
use crate::pressure::eviction::h2o_plus::H2OPlusPolicy;
use crate::pressure::eviction::no_eviction::NoEvictionPolicy;
use crate::pressure::eviction::sliding_window::SlidingWindowPolicy;
use crate::pressure::eviction::{EvictionPolicy, StreamingLLMPolicy};
use crate::pressure::{CachePressurePipeline, PressureLevel, PressureStageConfig};
use crate::resilience::sys_monitor::{LinuxSystemMonitor, NoOpMonitor};
use crate::session::cli::Args;
use crate::session::forward::{ModelForward, alloc_standard_kv_caches};
use crate::session::resilience_adapter::ResilienceAdapter;
use crate::session::{DecodeLoop, DecodeLoopBuilder, GreedySampler, RepetitionPenaltySampler};

/// CLI `eviction <policy>` + `--swap-dir` 로 resilience-driven force eviction /
/// KvOffload 용 [`CacheManager`] 를 구성한다. `eviction=none` 이고 `--swap-dir`
/// 도 없으면 `None`.
///
/// - AB-1 eviction: score-free `force_evict`(verify eviction 시나리오는
///   `functional_only`) — [`AttentionScoreAccumulator`](crate::inference::attention_scores::AttentionScoreAccumulator)
///   미장착, H2O 는 score 부재 시 recency degrade(chat 과 동일).
/// - AB-3 KvOffload: `--swap-dir` 지정 시 `enable_swap` 으로 SwapHandler 등록.
///   eviction=none + swap-dir 만 있는 경우 [`NoEvictionPolicy`] CacheManager 위에
///   swap 만 활성(offload/recall directive 가 cm.offload/recall 호출).
///
/// d2o 는 layer-alloc/variance 머신을 요구하므로 AB-1 범위(non-layer-alloc)만 지원.
pub fn build_resilience_cache_manager(
    args: &Args,
    backend: &Arc<dyn Backend>,
) -> Result<Option<CacheManager>> {
    let policy_name = args.eviction_policy();
    let swap_dir = args.swap_dir.clone();
    if policy_name == "none" && swap_dir.is_none() {
        return Ok(None);
    }

    let actual_protected_prefix = args.protected_prefix().unwrap_or(match policy_name {
        "h2o" | "h2o_plus" | "d2o" => 4,
        "streaming" => args.sink_size(),
        _ => 4,
    });

    let monitor: Box<dyn crate::resilience::sys_monitor::SystemMonitor> =
        if backend.is_discrete_gpu() {
            Box::new(NoOpMonitor)
        } else {
            Box::new(LinuxSystemMonitor)
        };
    let threshold_bytes = args.memory_threshold_mb() * 1024 * 1024;
    let target_ratio = args.eviction_target_ratio();

    let mut cm = if policy_name == "d2o" {
        let d2o_handler = D2OHandler::new(D2OConfig {
            keep_ratio: target_ratio,
            protected_prefix: actual_protected_prefix,
            target_ratio,
            ema_beta: 0.7,
            merge_e: 0.1,
            use_layer_allocation: false,
            protected_layers: Vec::new(),
        });
        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(d2o_handler),
        }]);
        CacheManager::with_pipeline(pipeline, monitor, threshold_bytes)
    } else {
        let policy: Box<dyn EvictionPolicy> = match policy_name {
            // eviction=none + swap-dir 전용(AB-3): eviction 은 안 하고 offload 만.
            "none" => Box::new(NoEvictionPolicy::new()),
            "sliding" => Box::new(SlidingWindowPolicy::new(
                args.eviction_window(),
                actual_protected_prefix,
            )),
            "streaming" => {
                let window = if args.streaming_window() > 0 {
                    args.streaming_window()
                } else if args.kv_budget() > 0 {
                    args.kv_budget().saturating_sub(args.sink_size())
                } else {
                    args.eviction_window()
                };
                Box::new(StreamingLLMPolicy::new(args.sink_size(), window))
            }
            "h2o" => Box::new(H2OPolicy::new(
                args.h2o_keep_ratio(),
                actual_protected_prefix,
            )),
            "h2o_plus" => Box::new(H2OPlusPolicy::new(
                args.h2o_keep_ratio(),
                actual_protected_prefix,
            )),
            other => anyhow::bail!(
                "argus-bench: unknown eviction policy '{}'. Use: none, sliding, streaming, h2o, h2o_plus.",
                other
            ),
        };
        CacheManager::new(policy, monitor, threshold_bytes, target_ratio)
    };

    if let Some(dir) = swap_dir {
        // legacy generate.rs:935 미러 — KvOffload directive 가 cm.offload 호출 시
        // SwapHandler 가 활성화되어 있어야 한다.
        eprintln!("[Resilience] KV swap enabled: dir={}", dir.display());
        cm.enable_swap(dir);
    }
    Ok(Some(cm))
}

/// [`build_standard_loop`](super::build_standard_loop) 와 동일 골격 + resilience
/// eviction `CacheManager` 주입. `cache_manager=None` 이면 happy-path 와 동등.
#[allow(clippy::too_many_arguments)]
pub fn build_bench_loop(
    backend: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    cpu_backend: Arc<dyn Backend>,
    model: TransformerModel,
    initial_kv_capacity: usize,
    max_seq_len: usize,
    kv_dtype: DType,
    sampling_config: SamplingConfig,
    plan_enabled: bool,
    resilience: Option<ResilienceAdapter>,
    cache_manager: Option<CacheManager>,
) -> Result<DecodeLoop> {
    let vocab_size = model.config.vocab_size;
    let kv = alloc_standard_kv_caches(
        &model,
        backend.clone(),
        memory.clone(),
        initial_kv_capacity,
        max_seq_len,
        kv_dtype,
    )?;
    let mf = ModelForward::new(
        backend,
        memory,
        cpu_backend,
        Arc::new(model),
        kv,
        max_seq_len,
        plan_enabled,
    )?;

    let use_stateful =
        sampling_config.repetition_penalty != 1.0 || sampling_config.temperature != 0.0;
    let builder = DecodeLoopBuilder::new().with_forward(mf);
    let builder = if use_stateful {
        builder.with_sampler(RepetitionPenaltySampler::new(sampling_config, vocab_size))
    } else {
        builder.with_sampler(GreedySampler)
    };
    let builder = match resilience {
        Some(adapter) => builder.with_resilience(adapter),
        None => builder,
    };
    let builder = match cache_manager {
        Some(cm) => builder.with_cache_manager(cm),
        None => builder,
    };
    Ok(builder.build())
}
