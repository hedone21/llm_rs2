//! Merge handler stub — combines similar KV tokens to reduce cache size.
//!
//! Not yet implemented. Returns `ActionResult::NoOp`.

use super::{ActionResult, CachePressureHandler, HandlerContext};
use anyhow::Result;

/// Merges similar adjacent KV tokens via weighted averaging.
///
/// Future implementation will:
/// - Compute cosine similarity between adjacent token K/V vectors
/// - Merge pairs above a similarity threshold into weighted averages
/// - Compact the cache via `shift_positions()`
/// - Track merge count per position via `TokenMeta`
/// - Protect prefix tokens from merging
pub struct MergeHandler;

impl MergeHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MergeHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CachePressureHandler for MergeHandler {
    fn handle(&self, _ctx: &mut HandlerContext) -> Result<ActionResult> {
        log::debug!("[MergeHandler] Not yet implemented");
        Ok(ActionResult::NoOp)
    }

    fn name(&self) -> &str {
        "merge"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::pressure::PressureLevel;

    #[test]
    fn test_merge_returns_noop() {
        let handler = MergeHandler::new();
        let mut caches = vec![];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
            target_ratio: None,
            proxy_sink: None,
        };
        let result = handler.handle(&mut ctx).unwrap();
        assert!(!result.is_action());
    }

    #[test]
    fn test_merge_name() {
        assert_eq!(MergeHandler::new().name(), "merge");
    }
}
