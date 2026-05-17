//! Phase 4-4-a: standard happy path용 [`DecodeLoop`] 조립자.
//!
//! [`build_standard_loop`]는 [`SessionInitCtx`]를 consume하여 표준 `KVCache`
//! 기반 [`ModelForward`] + greedy sampler [`DecodeLoop`]을 반환한다. ctx를
//! consume하므로 chat/eval-ll/ppl/batch 등 다른 분기와 동시 사용 불가
//! (early-return 구조라 자연스럽게 모순 없음).
//!
//! Happy path 진입 조건은 [`is_standard_happy_path`] 참조. chunked prefill /
//! optional collector 의존 케이스는 Phase 4-4.5에서 흡수 예정.

use std::sync::Arc;

use anyhow::Result;

use crate::core::buffer::DType;
use crate::session::cli::Args;
use crate::session::forward::{ModelForward, alloc_standard_kv_caches};
use crate::session::init::SessionInitCtx;
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

/// Phase 4-4-a: `SessionInitCtx`를 consume하여 standard `DecodeLoop` 조립.
///
/// **ctx-consume 패턴 (Q2-B)**: `ctx.model`을 `Arc::new(ctx.model)`로 옮기므로
/// ctx를 다시 사용 불가. early-return 구조 (chat/eval-ll/ppl/batch는 본 헬퍼
/// 호출 전 모두 분기됨)에서 충돌 없음.
///
/// - `max_seq_len`: KV cache 용량 + lazy `PrefillWorkspace` cap
/// - `kv_dtype`: KV cache element type (F32/F16/Q4_0). 호출자가 args에서 결정
pub fn build_standard_loop(
    ctx: SessionInitCtx,
    max_seq_len: usize,
    kv_dtype: DType,
) -> Result<DecodeLoop> {
    let kv = alloc_standard_kv_caches(
        &ctx.model,
        ctx.backend.clone(),
        ctx.memory.clone(),
        max_seq_len,
        max_seq_len,
        kv_dtype,
    )?;
    let mf = ModelForward::new(
        ctx.backend.clone(),
        ctx.memory.clone(),
        ctx.cpu_backend_arc.clone(),
        Arc::new(ctx.model),
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
