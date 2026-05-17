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

use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::session::cli::Args;
use crate::session::forward::{ModelForward, alloc_standard_kv_caches};
use crate::session::{DecodeLoop, DecodeLoopBuilder, GreedySampler};

/// Phase 4-4-a: standard generate happy path 진입 가드.
///
/// 다음을 모두 만족할 때만 `true`. 미통과 args는 generate.rs 기존 path 사용.
///
/// - `args.qcf_dump.is_none()`               — `--qcf-dump` 비활성 (importance_collector 미장착)
/// - `args.skip_ratio.unwrap_or(0.0) == 0.0` — `--skip-ratio=0` (skip_config 미장착)
/// - `!args.d2o_layer_alloc`                 — `--d2o-layer-alloc` 비활성 (variance_collector 미장착)
/// - `!args.profile && !args.profile_events` — profile 비활성 (profiler 미장착)
/// - `args.eviction_policy == "none"`        — eviction 비활성 (score_accumulator 미장착)
///
/// 호출자는 추가로 `prompt_len <= MAX_NON_CHUNKED_PREFILL_LEN`도 검증해야 한다
/// (chunked prefill 미지원). 그 가드는 generate.rs 호출 site에서 처리.
pub fn is_standard_happy_path(args: &Args) -> bool {
    args.qcf_dump.is_none()
        && args.skip_ratio.unwrap_or(0.0) == 0.0
        && !args.d2o_layer_alloc
        && !args.profile
        && !args.profile_events
        && args.eviction_policy == "none"
}

/// Phase 4-4-a: unpack-args 형태로 standard `DecodeLoop` 조립.
///
/// **model consume 패턴 (Q2-B α변형)**: `model: TransformerModel` owned 인자를
/// `Arc::new(model)`로 1회 변환하여 [`ModelForward`]에 위임. 호출자
/// (generate.rs main)는 본 헬퍼 호출 후 `model` 변수를 다시 사용할 수 없다.
/// chat/eval-ll/ppl/batch는 early-return 구조이므로 표준 path 진입 시점에
/// 다른 분기로 흐를 가능성 없음 (자연스러운 모순 없음).
///
/// - `max_seq_len`: KV cache 용량 + lazy `PrefillWorkspace` cap
/// - `kv_dtype`: KV cache element type (F32/F16/Q4_0). 호출자가 args에서 결정
pub fn build_standard_loop(
    backend: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    cpu_backend: Arc<dyn Backend>,
    model: TransformerModel,
    initial_kv_capacity: usize,
    max_seq_len: usize,
    kv_dtype: DType,
) -> Result<DecodeLoop> {
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
    )?;
    Ok(DecodeLoopBuilder::new()
        .with_forward(mf)
        .with_sampler(GreedySampler)
        .build())
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
    fn happy_path_default_args() {
        let args = default_args();
        assert!(
            is_standard_happy_path(&args),
            "default Args (no flags)는 happy path 진입해야 함"
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
        args.d2o_layer_alloc = true;
        assert!(!is_standard_happy_path(&args));
    }

    #[test]
    fn rejects_non_none_eviction() {
        let mut args = default_args();
        args.eviction_policy = "sliding".to_string();
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
