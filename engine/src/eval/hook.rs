//! StepHook trait: abstracts per-step cache management (eviction vs KIVI flush).
//!
//! The generic eval loop calls these hooks without knowing the cache management
//! policy. Each implementation encapsulates its own eviction/flush logic and
//! QCF metric collection.

use crate::core::attention_scores::AttentionScoreAccumulator;
use crate::core::kv_cache::KVCacheOps;

/// Result of a post-decode-step hook invocation.
#[derive(Debug, Default)]
pub struct PostStepResult {
    /// Whether any eviction/flush occurred this step.
    pub evicted: bool,
    /// Number of tokens removed (eviction) or quantized (KIVI flush).
    pub tokens_affected: usize,
    /// New start_pos after eviction (if evicted, caller should update).
    pub new_start_pos: Option<usize>,
}

/// Aggregated QCF/OPR metrics summary for JSON output.
#[derive(Debug, Clone, Default)]
pub struct MetricsSummary {
    pub qcf_attn_total: f64,
    pub qcf_caote_total: f64,
    pub qcf_normalized_total: f64,
    pub qcf_kivi_opr: Option<f64>,
    pub qcf_kivi_opr_events: usize,
}

/// Snapshot of KV cache state for choice-level restore.
pub trait CacheSnapshot<C: KVCacheOps>: Send {
    /// Restore caches to the snapshotted state.
    fn restore_to(&self, caches: &mut [C]);
}

/// Per-step cache management hook for the generic eval loop.
///
/// Implementations:
/// - `EvictionHook` (KVCache): budget-based eviction + CAOTE/attn QCF
/// - `KiviHook` (KiviCache): flush proxy collection (NMSE + OPR)
pub trait StepHook<C: KVCacheOps> {
    /// Called after each decode step. Performs eviction/flush if needed
    /// and collects QCF metrics.
    fn post_decode_step(
        &mut self,
        caches: &mut [C],
        step: usize,
        qcf_metrics: &mut Vec<serde_json::Value>,
    ) -> PostStepResult;

    /// Called after prefill completes. Handles chunked-prefill eviction
    /// residuals or flush proxy collection.
    fn post_prefill(&mut self, caches: &mut [C], qcf_metrics: &mut Vec<serde_json::Value>);

    /// Reset caches for a new question evaluation.
    fn reset_caches(&mut self, caches: &mut [C]);

    /// Create a snapshot of the current cache state (after prefill).
    fn snapshot(&self, caches: &[C]) -> Box<dyn CacheSnapshot<C>>;

    /// Provide mutable access to the score accumulator (if any).
    /// EvictionHook returns Some; KiviHook returns None.
    fn score_accumulator(&mut self) -> Option<&mut AttentionScoreAccumulator>;

    /// Update the effective budget (used by ratio-mode per-question budget).
    /// Default is no-op (e.g., KiviHook ignores budget).
    fn set_effective_budget(&mut self, _budget: usize) {}

    /// Cache-specific per-question JSON fields (e.g., kivi_q2_tokens).
    fn extra_question_fields(&self, caches: &[C]) -> serde_json::Value;

    /// Cache-specific top-level config JSON fields.
    fn extra_config_fields(&self) -> serde_json::Value;

    /// Aggregate collected QCF metrics into a summary.
    fn aggregate_metrics(&self, qcf_metrics: &[serde_json::Value]) -> MetricsSummary;
}
