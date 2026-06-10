//! [`Forward`] trait + concrete implementations (Phase 4-3+).
//!
//! Each submodule provides one `Forward` variant that the `DecodeLoopBuilder`
//! can be wired with. The first is [`model_forward::ModelForward`] — the
//! standard `KVCache`-backed path that wraps
//! [`crate::models::transformer::TransformerModel::forward_into`].
//!
//! Phase 4-5-a adds [`kivi_forward::KiviForward`] and
//! [`offload_forward::OffloadForward`] for KIVI-quantized and token-streaming
//! offload KV cache paths respectively.
//!
//! **Phase β-7**: the [`Forward`] trait itself moved here from the deleted
//! `session::traits` (slim internal seam, G2-(ii)). [`StepCtx`] lives in
//! [`crate::inference::sampling`] (shared with [`TokenSampler`]).

pub mod kivi_forward;
pub mod model_forward;
pub mod offload_forward;

pub use kivi_forward::{KiviForward, alloc_kivi_kv_caches};
pub use model_forward::{ModelForward, alloc_standard_kv_caches};
pub use offload_forward::{OffloadForward, alloc_offload_kv_caches};

use crate::inference::sampling::StepCtx;

/// Required forward pass. Provides KV-bearing model semantics.
///
/// `finalize` and `on_kv_prune` are default no-ops so a minimal Forward
/// implementation needs only `prefill` + `step`.
pub trait Forward {
    /// Run prefill over `tokens` starting at KV position `start_pos`.
    /// Returns logits for the last token.
    ///
    /// `start_pos`는 chat multi-turn에서 turn 사이 KV 누적 보존을 위해 필수.
    /// non-chat 모드에선 caller가 0 전달 (단일 prefill).
    fn prefill(&mut self, tokens: &[u32], start_pos: usize) -> anyhow::Result<Vec<f32>>;

    /// Decode 1 step. After return, `pos` is conceptually advanced by 1.
    fn step(&mut self, ctx: &StepCtx, token: u32) -> anyhow::Result<Vec<f32>>;

    /// Called once after [`crate::session::DecodeLoop::run`] exits.
    fn finalize(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Notified after an eviction stage pruned KV state.
    fn on_kv_prune(&mut self, _new_pos: usize) {}

    /// Phase 4-5-d: chat `/reset` 처리용. KV cache를 초기 상태로 reset한다.
    ///
    /// Default no-op — generate 모드는 호출하지 않는다. chat 모드의 각 Forward
    /// 구현체가 override하여 KV-type별 reset 로직을 수행한다.
    fn reset_kv(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Phase 4-5-e: chat `ensure_capacity` / `on_turn_end` 에서 eviction 실행.
    ///
    /// ModelForward만 override한다. KiviForward / OffloadForward는 default no-op
    /// 유지 (eviction 미지원).
    ///
    /// - `cache_manager`: eviction 정책을 보유한 CacheManager.
    /// - `scores`: score-based policy (H2O/D2O)용 importance 점수. `None`이면
    ///   score-free force/maybe evict.
    /// - `force`: true이면 `force_evict*`, false이면 `maybe_evict*`.
    /// - `target_ratio`: `force=true` 시 eviction 목표 비율.
    ///
    /// Returns (removed_count, new_pos). removed_count == 0이면 eviction 미발생.
    fn try_evict(
        &mut self,
        cache_manager: &crate::pressure::cache_manager::CacheManager,
        scores: Option<&[f32]>,
        force: bool,
        target_ratio: f32,
    ) -> anyhow::Result<(usize, usize)> {
        let _ = (cache_manager, scores, force, target_ratio);
        Ok((0, 0))
    }

    /// argus-bench AB-3: resilience KvOffload — LRU prefix 를 disk 로 offload.
    /// `cache_manager` 는 `--swap-dir` 로 enable_swap 된 SwapHandler 보유.
    /// `&mut` 필요(offload 가 handler ratio 변경). ModelForward 만 override(UER).
    ///
    /// Returns (offloaded_count, new_pos). offloaded_count == 0 이면 미발생
    /// (swap 미활성/대상 0).
    fn try_offload(
        &mut self,
        cache_manager: &mut crate::pressure::cache_manager::CacheManager,
        ratio: f32,
    ) -> anyhow::Result<(usize, usize)> {
        let _ = (cache_manager, ratio);
        Ok((0, 0))
    }

    /// argus-bench AB-3: resilience RestoreDefaults — offload 된 prefix 를 disk 에서
    /// recall. ModelForward 만 override.
    ///
    /// Returns (recalled_count, new_pos). recalled_count == 0 이면 미발생.
    fn try_recall(
        &mut self,
        cache_manager: &mut crate::pressure::cache_manager::CacheManager,
    ) -> anyhow::Result<(usize, usize)> {
        let _ = cache_manager;
        Ok((0, 0))
    }
}
