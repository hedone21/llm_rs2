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

use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::backend::Backend;
use crate::inference::sampling::SamplingConfig;
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::pressure::cache_manager::CacheManager;
use crate::pressure::d2o_handler::{D2OConfig, D2OHandler};
use crate::pressure::eviction::EvictionPolicy;
use crate::pressure::eviction::h2o_plus::H2OPlusPolicy;
use crate::pressure::eviction::no_eviction::NoEvictionPolicy;
use crate::pressure::eviction::stage_registry::StageBackedPolicy;
use crate::pressure::kv_cache::KVCache;
use crate::pressure::{CachePressurePipeline, PressureLevel, PressureStageConfig};
use crate::resilience::sys_monitor::{LinuxSystemMonitor, NoOpMonitor};
use crate::session::cli::Args;
use crate::session::command_dispatcher::CommandDispatcher;
use crate::session::forward::ModelForward;
use crate::session::pipeline_registry::PipelineRegistry;
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

    // linkme fat-LTO 생존 self-test (ADR-0003 §4): 빌트인 stage 미등록 시 fail-fast.
    crate::pressure::eviction::stage_registry::ensure_builtin_stages_registered()?;

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
            // h2o_plus(per-head, plan_keep→None) → 레거시 직생성 잔류(단계 ⑤, ADR-0004).
            "h2o_plus" => Box::new(H2OPlusPolicy::new(
                args.h2o_keep_ratio(),
                actual_protected_prefix,
            )),
            // sliding/streaming/h2o → KVCacheStage 레지스트리(OCP). chat 경로(session.rs)와 동일
            // factory. 레지스트리 miss = unknown(기존 bail 메시지 보존). World B(compact_parity 게이트).
            name => {
                let streaming_window = if args.streaming_window() > 0 {
                    args.streaming_window()
                } else if args.kv_budget() > 0 {
                    args.kv_budget().saturating_sub(args.sink_size())
                } else {
                    args.eviction_window()
                };
                let params = technique_api::StageParams {
                    eviction_window: args.eviction_window(),
                    protected_prefix: actual_protected_prefix,
                    keep_ratio: args.h2o_keep_ratio(),
                    sink_size: args.sink_size(),
                    streaming_window,
                };
                // 정적(linkme) + 동적(--load-plugin dlopen) 통합 조회(ADR-0009 D3). miss = unknown.
                // d2o 는 위 분기에서 처리되나 유효 정책이라 안내에 포함(session.rs 와 일관).
                let stage = crate::pressure::eviction::stage_registry::make_stage(name, &params)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "argus-bench: unknown eviction policy '{}'. Use: none, sliding, streaming, h2o, h2o_plus, d2o{} (or --load-plugin <.so>).",
                            name,
                            if cfg!(feature = "caote") { ", caote" } else { "" }
                        )
                    })?;
                Box::new(StageBackedPolicy::new(stage))
            }
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
///
/// `kv_caches`: `bin_setup`이 `--kv-format`/`--kv-type` dispatch로 이미 할당한
/// KV cache (typed 또는 ADR-0008 opaque). builder는 재할당하지 않고 소비한다.
#[allow(clippy::too_many_arguments)]
pub fn build_bench_loop(
    backend: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    cpu_backend: Arc<dyn Backend>,
    model: TransformerModel,
    kv_caches: Vec<KVCache>,
    max_seq_len: usize,
    sampling_config: SamplingConfig,
    plan_enabled: bool,
    resilience: Option<ResilienceAdapter>,
    cache_manager: Option<CacheManager>,
) -> Result<DecodeLoop> {
    let vocab_size = model.config.vocab_size;
    // ADR-0008: decode loop가 실제로 쥐는 KV 저장 형태를 진입 시점에 보고
    // (build_standard_loop 와 동일 — alloc-시점 로그는 drop돼도 찍혀 증거 못 됨).
    let kv_is_opaque = kv_caches.first().is_some_and(|c| c.is_opaque());
    eprintln!(
        "[DecodeLoop] kv storage = {} (layers={}, cap={})",
        if kv_is_opaque {
            "OPAQUE (descriptor-driven)"
        } else {
            "typed"
        },
        kv_caches.len(),
        max_seq_len,
    );
    let mf = ModelForward::new(
        backend,
        memory,
        cpu_backend,
        Arc::new(model),
        kv_caches,
        max_seq_len,
        plan_enabled,
    )?;

    // β-3: pos-환류용 layer-0 fmt handle (§5.2.1 (가)). coercion: Arc<StandardFormat> →
    // Arc<dyn KVCacheFormat>.
    let kv_pos_handle: Option<Arc<dyn crate::format::KVCacheFormat>> = mf
        .fmt_caches()
        .first()
        .map(|f| f.clone() as Arc<dyn crate::format::KVCacheFormat>);

    // β-4: EvictionStage 가 prune 할 전체 layer handle (W1 — enumerate 순서 == layer idx).
    let kv_handles: Vec<Arc<crate::pressure::standard_format::StandardFormat>> =
        mf.fmt_caches().to_vec();

    // β-4 (매핑 문서 4부): resilience adapter 에 held-handle 주입 → heartbeat snapshot 의
    // kv_cache_tokens/capacity 를 layer-0 handle 에서 query (poll 인자 제거 대체).
    let resilience = match (resilience, kv_pos_handle.clone()) {
        (Some(mut adapter), Some(h)) => {
            adapter.set_kv_handle(h);
            Some(adapter)
        }
        (other, _) => other,
    };

    // β-4: dispatcher 와 driver 가 공유하는 registry. dispatcher.submit(OneShot EvictionStage) →
    // driver 의 PreEviction dispatch(β-3 배선)가 소비.
    let registry = Arc::new(PipelineRegistry::new());

    // β-4: EngineCommand 분배자. **resilience-on 이면 CM 유무와 무관하게 구성** — control
    // 디렉티브(Throttle/SetTargetTbt/Suspend 등)는 CM 없이 소비 가능하고, v1 도 eviction=none +
    // resilience-on 에서 control 을 적용했다(미구성 시 디렉티브 무소비 드롭 = v1 회귀, β-4 device
    // smoke 실증 2026-06-10). evict 디렉티브는 CM=None 이면 dispatcher 내부에서 inert —
    // v1 (a.5) 의 `cache_manager=None` 스킵과 등가. 둘 다 None 이면 미구성(happy-path 거동-0).
    let dispatcher = if resilience.is_some() || cache_manager.is_some() {
        Some(CommandDispatcher::new(
            Arc::clone(&registry),
            kv_handles,
            cache_manager.map(|cm| Arc::new(Mutex::new(cm))),
        ))
    } else {
        None
    };

    let use_stateful =
        sampling_config.repetition_penalty != 1.0 || sampling_config.temperature != 0.0;
    let builder = DecodeLoopBuilder::new()
        .with_forward(mf)
        .with_pipeline(Arc::clone(&registry));
    let builder = match kv_pos_handle {
        Some(h) => builder.with_kv_pos_handle(h),
        None => builder,
    };
    let builder = if use_stateful {
        builder.with_sampler(RepetitionPenaltySampler::new(sampling_config, vocab_size))
    } else {
        builder.with_sampler(GreedySampler)
    };
    let builder = match resilience {
        Some(adapter) => builder.with_resilience(adapter),
        None => builder,
    };
    let builder = match dispatcher {
        Some(d) => builder.with_command_dispatcher(d),
        None => builder,
    };
    Ok(builder.build())
}
