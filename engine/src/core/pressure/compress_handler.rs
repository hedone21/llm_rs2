//! Compress handler stub — applies KV compression (e.g., SnapKV).
//!
//! Not yet implemented. Returns `ActionResult::NoOp`.

use super::{ActionResult, CachePressureHandler, HandlerContext};
use anyhow::Result;

/// Compresses KV cache data (e.g., SnapKV prefill-time compression).
///
/// Future implementation will:
/// - At prefill end, use final-token attention scores to identify important KV
/// - Per KV-head, select top-k tokens and discard the rest
/// - This is a one-shot operation (not iterative like eviction)
/// - GQA-correct: average Q-head scores per KV group
/// - Requires head-major layout (already migrated)
pub struct CompressHandler;

impl CompressHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CompressHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CachePressureHandler for CompressHandler {
    fn handle(&self, _ctx: &mut HandlerContext) -> Result<ActionResult> {
        log::debug!("[CompressHandler] Not yet implemented");
        Ok(ActionResult::NoOp)
    }

    fn name(&self) -> &str {
        "compress"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::pressure::PressureLevel;

    #[test]
    fn test_compress_returns_noop() {
        let handler = CompressHandler::new();
        let mut caches = vec![];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Warning,
            mem_available: 0,
            target_ratio: None,
        };
        let result = handler.handle(&mut ctx).unwrap();
        assert!(!result.is_action());
    }

    #[test]
    fn test_compress_name() {
        assert_eq!(CompressHandler::new().name(), "compress");
    }
}
