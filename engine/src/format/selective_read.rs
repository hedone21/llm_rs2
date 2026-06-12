//! `SelectiveRead` capability — `KVCacheFormat` 의 선택적 읽기 opt-in 표면(ADR-0011 D4).
//!
//! 설계 SSOT: `docs/adr/0011-kv-read-plan-surface.md` D4 / Amendment A1.
//!
//! **capability opt-in**: `KVCacheFormat` base trait(6-method)에 추가하지 않는다 — ISP 보존.
//! 미지원 format 은 full read 폴백(정확). `SelectiveRead` 를 구현한 format 만 선택적 읽기를 제공.
//!
//! 첫 구현체: `StandardFormat` (`engine/src/kv/standard_format.rs`).
//! 미구현: KIVI/opaque → 엔진 폴백 = full read(ADR-0011 D4).

use anyhow::Result;
use technique_api::ReadGranularity;

use crate::backend::Backend;
use crate::format::AttnDims;
use crate::tensor::Tensor;

/// KV 캐시의 **선택적 읽기** capability(ADR-0011 D4).
///
/// `attention_into_selected` 는 `select` 된 KV 위치/페이지만 읽어 attention 을 수행한다.
/// **정확성 계약 아님** — `select` 가 전체면 `attention_into` 와 bit-identical(Tier 1 게이트),
/// 부분 select 는 *근사 답*(softmax 분모가 부분집합으로 정규화됨 — Quest 의 의도된 근사).
///
/// `KVCacheFormat` base trait 무변경(ADR-0011 D4 ISP): 이 capability 를 구현 안 한 format 은
/// 엔진이 plan 을 무시하고 `attention_into`(full read) 를 호출한다.
pub trait SelectiveRead {
    /// `select` 된 KV 위치만 읽는 attention.
    ///
    /// - `select`: ascending pos 목록(Token 단위) 또는 page index 목록(Page 단위).
    /// - `granularity`: `ReadGranularity::Token` 이면 `select` 를 pos 그대로 사용,
    ///   `Page { page_size }` 이면 각 page index 를 `[page*page_size .. (page+1)*page_size)` pos 범위로 전개.
    /// - `scores`: `Some` 이면 선택된 토큰에 대한 post-softmax score 를 기록.
    /// - 기타 계약은 `KVCacheFormat::attention_into` 와 동일.
    #[allow(clippy::too_many_arguments)]
    fn attention_into_selected(
        &self,
        q: &Tensor,
        backend: &dyn Backend,
        out: &mut Tensor,
        dims: AttnDims,
        select: &[usize],
        granularity: ReadGranularity,
        scores: Option<&mut [f32]>,
    ) -> Result<()>;
}
