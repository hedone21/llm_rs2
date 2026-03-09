//! Sparse attention handler stub — applies sparse attention masks.
//!
//! Not yet implemented. Returns `ActionResult::NoOp`.

use super::{ActionResult, CachePressureHandler, HandlerContext};
use anyhow::Result;

/// Applies sparse attention patterns to reduce computation.
///
/// Future implementation will:
/// - Generate sparse attention masks (local window + strided global)
/// - Support adaptive sparsity using per-head importance scores
/// - Does NOT modify KV data — modifies attention behavior
/// - Integrate with `prepare_for_attention()` to pass masks to kernels
pub struct SparseHandler;

impl SparseHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SparseHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CachePressureHandler for SparseHandler {
    fn handle(&self, _ctx: &mut HandlerContext) -> Result<ActionResult> {
        log::debug!("[SparseHandler] Not yet implemented");
        Ok(ActionResult::NoOp)
    }

    fn name(&self) -> &str {
        "sparse"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::pressure::PressureLevel;

    #[test]
    fn test_sparse_returns_noop() {
        let handler = SparseHandler::new();
        let mut caches = vec![];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
        };
        let result = handler.handle(&mut ctx).unwrap();
        assert!(!result.is_action());
    }

    #[test]
    fn test_sparse_name() {
        assert_eq!(SparseHandler::new().name(), "sparse");
    }
}
