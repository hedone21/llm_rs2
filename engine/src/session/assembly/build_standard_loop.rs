//! Phase 4-4-a: standard happy path용 [`DecodeLoop`] 조립자.
//!
//! [`build_standard_loop`]는 unpack된 args (backend / memory / cpu_backend /
//! model) 를 받아 표준 `KVCache` 기반 [`ModelForward`] + greedy sampler
//! [`DecodeLoop`]을 반환한다. `model`은 owned consume — `Arc::new(model)`로
//! 1회 변환하여 [`ModelForward`]에 위임 (Q2-B 결정의 변형 α: ctx struct
//! 의존 없이 unpack-args 시그니처).
//!
//! ## 왜 ctx-consume이 아닌 unpack-args인가
//!
//! `bin/generate.rs` line 91~109에서 `SessionInitCtx::build(&args)?` 직후
//! 즉시 ctx unpack이 발생한다. 따라서 표준 path 진입 시점 (line 3032)에
//! ctx struct는 더 이상 존재하지 않으며, unpack된 `backend` / `memory` /
//! `model` 변수만 사용 가능하다. ctx struct를 보존하려면 모든 분기
//! (chat/eval-ll/ppl/batch)에 ctx-borrow 패턴을 적용해야 하므로 4-4 범위
//! 초과. → 헬퍼 시그니처를 unpack-args 형태로 한정하여 영향 범위를
//! standard path 한정으로 유지.
//!
//! Happy path 진입 조건은 [`is_standard_happy_path`] 참조. chunked prefill /
//! optional collector 의존 케이스는 Phase 4-4.5에서 흡수 예정.

use std::sync::Arc;

use anyhow::Result;

use crate::backend::Backend;
use crate::inference::sampling::SamplingConfig;
use crate::kv::kv_cache::KVCache;
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::session::cli::Args;
use crate::session::command_dispatcher::CommandDispatcher;
use crate::session::forward::ModelForward;
use crate::session::pipeline_registry::PipelineRegistry;
use crate::session::resilience_adapter::ResilienceAdapter;
use crate::session::{DecodeLoop, DecodeLoopBuilder, GreedySampler, RepetitionPenaltySampler};

/// Phase 4-4-a: standard generate happy path 진입 가드.
///
/// 다음을 모두 만족할 때만 `true`. 미통과 args는 generate.rs 기존 path 사용.
///
/// - `args.qcf_dump.is_none()`               — `--qcf-dump` 비활성 (importance_collector 미장착)
/// - `args.skip_ratio.unwrap_or(0.0) == 0.0` — `--skip-ratio=0` (skip_config 미장착)
/// - `!args.d2o_layer_alloc()`                 — `--d2o-layer-alloc` 비활성 (variance_collector 미장착)
/// - `!args.profile && !args.profile_events` — profile 비활성 (profiler 미장착)
/// - `args.eviction_policy() == "none"`        — eviction 비활성 (score_accumulator 미장착)
/// - `args.tensor_partition == 0.0`          — Phase 4-4.7: tensor_partition 활성 시
///   plan path가 build_plan에서 None을 반환 → sticky_disabled lock-out → 매 step
///   forward_into fallback이라 성능 저하. happy path에서는 partition 차단.
/// - `!args.swap_intra_forward && !args.swap_layer_immediate && !args.swap_phase_aware`
///   — Phase 4-4.7: weight swap intra-forward / phase-aware는 plan path가 미지원
///   (production generate.rs l.4192-4199 가드와 동치).
///
/// Phase 4-4.7에서 `repetition_penalty == 1.0` 가드가 제거되었다. 대신
/// [`build_standard_loop`]가 `sampling_config`에 따라
/// [`GreedySampler`] 또는 [`RepetitionPenaltySampler`] 중 적절한 sampler를
/// 자동 선택하여 production `sampling::sample` 호출과 paradigm equivalent
/// 결과를 보장한다.
///
/// 호출자는 추가로 `prompt_len <= MAX_NON_CHUNKED_PREFILL_LEN`도 검증해야 한다
/// (chunked prefill 미지원). 그 가드는 generate.rs 호출 site에서 처리.
pub fn is_standard_happy_path(args: &Args) -> bool {
    args.qcf_dump.is_none()
        && args.skip_ratio.unwrap_or(0.0) == 0.0
        && !args.d2o_layer_alloc()
        && !args.profile
        && !args.profile_events
        && args.eviction_policy() == "none"
        && args.tensor_partition == 0.0
        && !args.swap_intra_forward
        && !args.swap_layer_immediate
        && !args.swap_phase_aware
}

/// Phase 4-4-a: unpack-args 형태로 standard `DecodeLoop` 조립.
///
/// **model consume 패턴 (Q2-B α변형)**: `model: TransformerModel` owned 인자를
/// `Arc::new(model)`로 1회 변환하여 [`ModelForward`]에 위임. 호출자
/// (generate.rs main)는 본 헬퍼 호출 후 `model` 변수를 다시 사용할 수 없다.
/// chat/eval-ll/ppl/batch는 early-return 구조이므로 표준 path 진입 시점에
/// 다른 분기로 흐를 가능성 없음 (자연스러운 모순 없음).
///
/// - `kv_caches`: `bin_setup`이 `--kv-format`/`--kv-type` dispatch로 이미 할당한
///   KV cache (typed 또는 ADR-0008 opaque). builder는 재할당하지 않고 소비한다 —
///   과거에는 builder가 `alloc_standard_kv_caches`로 typed를 재할당하여 `ctx`의
///   opaque 선택을 덮어썼다(`--kv-format`이 decode 경로에 도달 못 함).
/// - `max_seq_len`: lazy `PrefillWorkspace` cap
/// - `resilience`: P3.3 — `Some(adapter)` 이면 3 slot에 주입, `None` 이면 NoOp default.
#[allow(clippy::too_many_arguments)]
pub fn build_standard_loop(
    backend: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    cpu_backend: Arc<dyn Backend>,
    model: TransformerModel,
    kv_caches: Vec<KVCache>,
    max_seq_len: usize,
    sampling_config: SamplingConfig,
    plan_enabled: bool,
    resilience: Option<ResilienceAdapter>,
) -> Result<DecodeLoop> {
    let vocab_size = model.config.vocab_size;
    // ADR-0008: decode loop가 실제로 쥐는 KV 저장 형태를 진입 시점에 보고한다.
    // bin_setup의 alloc-시점 "KV format" 로그는 caches가 drop돼도 찍히므로 증거가
    // 못 된다(과거 false-positive e2e의 원인). ModelForward가 소비하기 직전의
    // 이 identity가 진짜 decode 경로 증거다.
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

    // β-4: resilience-on 이면 dispatcher 를 구성한다 — control 디렉티브(Throttle/SetTargetTbt/
    // Suspend 등)는 CM 없이 소비 가능하고, v1 은 argus_cli resilience-on 에서 이를 적용했다
    // (dispatcher 부재 시 디렉티브 무소비 드롭 = v1 회귀, β-4 device smoke 실증 2026-06-10).
    // evict 디렉티브는 CM=None 이라 dispatcher 내부 inert — v1 (a.5) cache_manager=None 스킵 등가.
    // heartbeat kv snapshot 은 held-handle query(매핑 문서 4부 채택안 (가)) — layer-0 handle 주입.
    // `--no-resilience`(None) 경로는 아래 분기 전체 미진입 = 기존과 byte-identical 조립.
    let (resilience, dispatcher_parts) = match resilience {
        Some(mut adapter) => {
            let kv_pos_handle: Option<Arc<dyn crate::format::KVCacheFormat>> = mf
                .fmt_caches()
                .first()
                .map(|f| f.clone() as Arc<dyn crate::format::KVCacheFormat>);
            if let Some(h) = kv_pos_handle.clone() {
                adapter.set_kv_handle(h);
            }
            let registry = Arc::new(PipelineRegistry::new());
            // happy/chat 경로는 partition/swap 미구성 (빈 slots + None hardware/model/swap_runtime).
            let dispatcher = CommandDispatcher::new(
                Arc::clone(&registry),
                mf.fmt_caches().to_vec(),
                None,
                Vec::new(),
                None,
                None,
                None,
                None,
            );
            (Some(adapter), Some((registry, dispatcher, kv_pos_handle)))
        }
        None => (None, None),
    };

    // Phase 4-4.7: sampler 자동 선택. production `sampling::sample`은
    // temperature=0 + repetition_penalty=1.0이면 raw argmax와 동치이므로 두 조건이
    // 모두 만족될 때만 GreedySampler 사용. 그 외는 RepetitionPenaltySampler가
    // 내부 VecDeque ring buffer + scratch logits로 production 호출을 충실히 모사.
    let use_stateful =
        sampling_config.repetition_penalty != 1.0 || sampling_config.temperature != 0.0;
    let builder = DecodeLoopBuilder::new().with_forward(mf);
    let builder = if use_stateful {
        builder.with_sampler(RepetitionPenaltySampler::new(sampling_config, vocab_size))
    } else {
        builder.with_sampler(GreedySampler)
    };
    // P3.3: resilience adapter 주입 (Some → 3 slot 주입, None → NoOp default 유지)
    let builder = match resilience {
        Some(adapter) => builder.with_resilience(adapter),
        None => builder,
    };
    // β-4: dispatcher + 공유 registry + pos-환류 handle 배선 (resilience-on 한정).
    let builder = match dispatcher_parts {
        Some((registry, dispatcher, kv_pos_handle)) => {
            let b = builder
                .with_pipeline(registry)
                .with_command_dispatcher(dispatcher);
            match kv_pos_handle {
                Some(h) => b.with_kv_pos_handle(h),
                None => b,
            }
        }
        None => builder,
    };
    Ok(builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// `Args::parse_from(["llm_rs2"])` — clap의 default_value_t 적용된 baseline.
    /// `Default` derive 미보유 → parse_from으로 default 인스턴스 생성.
    fn default_args() -> Args {
        Args::parse_from(["llm_rs2"])
    }

    #[test]
    fn happy_path_with_no_repetition_penalty() {
        let mut args = default_args();
        args.repetition_penalty = 1.0;
        assert!(
            is_standard_happy_path(&args),
            "기본 args + --repetition-penalty 1.0 → happy path 진입"
        );
    }

    /// Phase 4-4.7: rep_penalty 가드가 제거됨. 기본 CLI (default 1.1) 도
    /// happy path 진입 가능 — `build_standard_loop`가 `RepetitionPenaltySampler`
    /// 를 자동 선택하여 production `sampling::sample` 호출과 동치 결과를 낸다.
    #[test]
    fn accepts_default_repetition_penalty() {
        let args = default_args();
        assert!(
            is_standard_happy_path(&args),
            "Phase 4-4.7: default repetition_penalty=1.1 happy path 진입 허용"
        );
    }

    #[test]
    fn rejects_skip_ratio() {
        let mut args = default_args();
        args.skip_ratio = Some(0.1);
        assert!(!is_standard_happy_path(&args));
    }

    #[test]
    fn rejects_profile() {
        let mut args = default_args();
        args.profile = true;
        assert!(!is_standard_happy_path(&args));
    }

    #[test]
    fn rejects_qcf_dump() {
        let mut args = default_args();
        args.qcf_dump = Some(std::path::PathBuf::from("/tmp/qcf.json"));
        assert!(!is_standard_happy_path(&args));
    }

    #[test]
    fn rejects_d2o_layer_alloc() {
        let mut args = default_args();
        args.eviction = Some(crate::session::cli::TopLevelCmd::Eviction {
            policy: crate::session::cli::EvictionCmd::D2o(crate::session::cli::D2oArgs {
                keep_ratio: 0.75,
                ema_beta: 0.7,
                merge_e: 0.1,
                layer_alloc: true,
                protected_layers: None,
            }),
        });
        assert!(!is_standard_happy_path(&args));
    }

    #[test]
    fn rejects_non_none_eviction() {
        let mut args = default_args();
        args.eviction = Some(crate::session::cli::TopLevelCmd::Eviction {
            policy: crate::session::cli::EvictionCmd::Sliding(crate::session::cli::SlidingArgs {
                window: 1024,
            }),
        });
        assert!(!is_standard_happy_path(&args));
    }

    /// `skip_ratio = Some(0.0)`는 비활성과 동등 (CLI 명시 했지만 0.0 = no skip)
    #[test]
    fn accepts_skip_ratio_zero() {
        let mut args = default_args();
        args.skip_ratio = Some(0.0);
        assert!(is_standard_happy_path(&args));
    }
}
