//! Quantize handler stub — reduces KV cache precision to save memory.
//!
//! Not yet implemented. Returns `ActionResult::NoOp`.

use super::{ActionResult, CachePressureHandler, HandlerContext};
use anyhow::Result;

/// Reduces KV cache precision (e.g., F16 → Q8_0 → Q4_0).
///
/// Future implementation will:
/// - Track per-cache effective dtype
/// - Step down precision when pressure rises (F16 → Q8 → Q4)
/// - Optionally use importance scores for adaptive quantization
///   (high-importance tokens keep F16, low-importance → Q4)
/// - Support layer-wise quantization (early layers → lower precision)
pub struct QuantizeHandler;

impl QuantizeHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for QuantizeHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CachePressureHandler for QuantizeHandler {
    fn handle(&self, _ctx: &mut HandlerContext) -> Result<ActionResult> {
        log::debug!("[QuantizeHandler] Not yet implemented");
        Ok(ActionResult::NoOp)
    }

    fn name(&self) -> &str {
        "quantize"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::pressure::PressureLevel;

    #[test]
    fn test_quantize_returns_noop() {
        let handler = QuantizeHandler::new();
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
    fn test_quantize_name() {
        assert_eq!(QuantizeHandler::new().name(), "quantize");
    }
}
