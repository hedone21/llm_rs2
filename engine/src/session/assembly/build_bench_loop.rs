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
use crate::kv::cache_manager::CacheManager;
use crate::kv::d2o_handler::{D2OConfig, D2OHandler};
use crate::kv::eviction::EvictionPolicy;
use crate::kv::eviction::h2o_plus::H2OPlusPolicy;
use crate::kv::eviction::no_eviction::NoEvictionPolicy;
use crate::kv::eviction::stage_registry::StageBackedPolicy;
use crate::kv::kv_cache::KVCache;
use crate::kv::{CachePressurePipeline, PressureLevel, PressureStageConfig};
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
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
    crate::kv::eviction::stage_registry::ensure_builtin_stages_registered()?;

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
                let stage = crate::kv::eviction::stage_registry::make_stage(name, &params)
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

/// β-5: argus-bench 용 memory-only [`LocalPressureSource`] 를 구성한다.
///
/// [`build_resilience_cache_manager`] 의 monitor/threshold 구성과 동일 의미(discrete GPU 면
/// `NoOpMonitor` = 압력 없음, 그 외 `LinuxSystemMonitor`). eviction/swap 도 resilience 도 없는
/// happy-path(`eviction=none` + swap-dir 없음 + resilience 없음)에서는 호출처가 `None` 으로 흘려
/// **무주입**한다 (per-token syscall 차단, G4). 본 함수는 source 객체만 만들고, 주입 여부는
/// 호출처(`experiment_run.rs`)가 결정한다.
pub fn build_local_pressure_source(
    args: &Args,
    backend: &Arc<dyn Backend>,
) -> Arc<dyn crate::pipeline::PressureSource> {
    let monitor: Arc<dyn crate::resilience::sys_monitor::SystemMonitor> =
        if backend.is_discrete_gpu() {
            Arc::new(NoOpMonitor)
        } else {
            Arc::new(LinuxSystemMonitor)
        };
    let threshold_bytes = args.memory_threshold_mb() * 1024 * 1024;
    Arc::new(crate::session::local_pressure::LocalPressureSource::new(
        monitor,
        threshold_bytes,
    ))
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
    // β-5: graded 압력 source (memory-only). None → 무주입(happy-path per-token syscall 0).
    pressure_source: Option<Arc<dyn crate::pipeline::PressureSource>>,
    // β-5: pressure-driven Persistent EvictionStage 의 force_evict target ratio
    // (CLI `--eviction-target-ratio` — CM 내부 값과 동일 출처를 호출자가 보장).
    pressure_evict_ratio: f32,
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
    let kv_handles: Vec<Arc<crate::kv::standard_format::StandardFormat>> = mf.fmt_caches().to_vec();

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
    // β-5: CM 을 Arc<Mutex> 로 한 번 들어 dispatcher(OneShot 구성)와 Persistent stage 가 공유.
    let shared_cm = cache_manager.map(|cm| Arc::new(Mutex::new(cm)));

    let dispatcher = if resilience.is_some() || shared_cm.is_some() {
        Some(CommandDispatcher::new(
            Arc::clone(&registry),
            kv_handles.clone(),
            shared_cm.clone(),
        ))
    } else {
        None
    };

    // β-5: pressure-driven Persistent EvictionStage — CM + graded source 가 둘 다 있을 때만
    // 상주 등록. band ≥ Warning 상향 에지에서 에피소드당 1회 force_evict (stage 내부
    // edge-trigger). source 부재(None) 면 StepInfo.pressure 가 항상 0(Normal) → 등록해도
    // 영구 무발화이므로 미등록 (의도 명시). ratio = CLI `--eviction-target-ratio`
    // (method-drop 시맨틱과 동일하게 정책은 CM 의 CLI 구성).
    if let (Some(cm), Some(_)) = (&shared_cm, &pressure_source) {
        let persistent = crate::stages::kv::eviction::EvictionStage::persistent(
            kv_handles,
            Arc::clone(cm),
            pressure_evict_ratio,
            llm_shared::Level::Warning,
        );
        registry.submit(Arc::new(persistent));
    }

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
    let builder = match pressure_source {
        Some(s) => builder.with_pressure_source(s),
        None => builder,
    };
    Ok(builder.build())
}
