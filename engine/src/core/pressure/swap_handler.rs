//! Swap handler stub — moves cold KV data to secondary storage (zram/tmpfs).
//!
//! Not yet implemented. Returns `ActionResult::NoOp`.

use super::{ActionResult, CachePressureHandler, HandlerContext};
use anyhow::Result;

/// Swaps KV cache data to secondary storage (zram, tmpfs, disk).
///
/// Future implementation will:
/// - Track per-layer swap state (Resident / SwappedOut)
/// - Swap-out cold layers when pressure rises
/// - Swap-in before attention via `prepare_for_attention()`
/// - Support async prefetch (swap-in layer N+1 while layer N computes)
pub struct SwapHandler;

impl SwapHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SwapHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CachePressureHandler for SwapHandler {
    fn handle(&self, _ctx: &mut HandlerContext) -> Result<ActionResult> {
        log::debug!("[SwapHandler] Not yet implemented");
        Ok(ActionResult::NoOp)
    }

    fn name(&self) -> &str {
        "swap"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::pressure::PressureLevel;

    #[test]
    fn test_swap_returns_noop() {
        let handler = SwapHandler::new();
        let mut caches = vec![];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
        };
        let result = handler.handle(&mut ctx).unwrap();
        assert!(!result.is_action());
    }

    #[test]
    fn test_swap_name() {
        assert_eq!(SwapHandler::new().name(), "swap");
    }
}
