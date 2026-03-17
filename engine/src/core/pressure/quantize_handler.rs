//! Quantize handler — reduces KV cache precision to save memory.
//!
//! Works with `KiviCache` to dynamically transition quantization bit-width
//! based on pressure level. Maps pressure levels to target bits:
//! - Normal: no action (keep current bits)
//! - Warning: 8-bit
//! - Critical: 4-bit
//! - Emergency: 2-bit

use super::{ActionResult, CachePressureHandler, HandlerContext, PressureLevel};
use anyhow::Result;

/// Reduces KV cache precision by transitioning KIVI cache bit-width.
///
/// This handler operates on the standard `HandlerContext` (which holds `KVCache`),
/// but returns `NoOp` since KIVI caches are managed separately.
///
/// For KIVI-specific quantization, use `QuantizeHandler::target_bits_for_pressure()`
/// directly from the generate loop.
pub struct QuantizeHandler;

impl QuantizeHandler {
    pub fn new() -> Self {
        Self
    }

    /// Map pressure level to target KIVI quantization bits.
    ///
    /// Returns `None` if no transition is needed (Normal pressure).
    pub fn target_bits_for_pressure(level: PressureLevel) -> Option<u8> {
        match level {
            PressureLevel::Normal => None,
            PressureLevel::Warning => Some(8),
            PressureLevel::Critical => Some(4),
            PressureLevel::Emergency => Some(2),
        }
    }
}

impl Default for QuantizeHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CachePressureHandler for QuantizeHandler {
    fn handle(&self, ctx: &mut HandlerContext) -> Result<ActionResult> {
        // Standard KVCache does not support dynamic quantization transition.
        // KIVI caches are handled separately via target_bits_for_pressure().
        if ctx.pressure_level == PressureLevel::Normal {
            return Ok(ActionResult::NoOp);
        }
        log::debug!(
            "[QuantizeHandler] pressure={:?}, target_bits={:?} (KIVI caches handled externally)",
            ctx.pressure_level,
            Self::target_bits_for_pressure(ctx.pressure_level)
        );
        Ok(ActionResult::NoOp)
    }

    fn name(&self) -> &str {
        "quantize"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::kv_cache::KVCache;
    use crate::core::pressure::PressureLevel;

    #[test]
    fn test_quantize_returns_noop() {
        let handler = QuantizeHandler::new();
        let mut caches: Vec<KVCache> = vec![];
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
        // Returns NoOp because standard KVCache doesn't support bit transition
        assert!(!result.is_action());
    }

    #[test]
    fn test_quantize_name() {
        assert_eq!(QuantizeHandler::new().name(), "quantize");
    }

    #[test]
    fn test_target_bits_normal() {
        assert_eq!(QuantizeHandler::target_bits_for_pressure(PressureLevel::Normal), None);
    }

    #[test]
    fn test_target_bits_warning() {
        assert_eq!(QuantizeHandler::target_bits_for_pressure(PressureLevel::Warning), Some(8));
    }

    #[test]
    fn test_target_bits_critical() {
        assert_eq!(QuantizeHandler::target_bits_for_pressure(PressureLevel::Critical), Some(4));
    }

    #[test]
    fn test_target_bits_emergency() {
        assert_eq!(QuantizeHandler::target_bits_for_pressure(PressureLevel::Emergency), Some(2));
    }
}
