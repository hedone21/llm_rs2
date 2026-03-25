//! Swap handler — offloads cold KV cache data to disk on memory pressure.
//!
//! Uses LRU (oldest tokens first) offload strategy.
//! Removes offloaded tokens from the cache (lossy without recall).

use super::{ActionResult, CachePressureHandler, HandlerContext, PressureLevel};
use anyhow::Result;

/// Offload ratio and storage path for disk swap.
pub struct SwapHandler {
    /// Fraction of tokens to offload (0.0–1.0). Default: 0.5.
    pub offload_ratio: f32,
}

impl SwapHandler {
    pub fn new(offload_ratio: f32) -> Self {
        Self { offload_ratio }
    }
}

impl Default for SwapHandler {
    fn default() -> Self {
        Self { offload_ratio: 0.5 }
    }
}

impl CachePressureHandler for SwapHandler {
    fn handle(&self, ctx: &mut HandlerContext) -> Result<ActionResult> {
        // Only activate on Warning+ pressure
        if ctx.pressure_level < PressureLevel::Warning {
            return Ok(ActionResult::NoOp);
        }

        let mut total_swapped = 0;

        for cache in ctx.caches.iter_mut() {
            let total = cache.current_pos;
            if total == 0 {
                continue;
            }

            let offload_count = ((total as f32 * self.offload_ratio) as usize).max(1);
            if offload_count >= total {
                continue; // Don't offload everything
            }

            // Remove oldest tokens via prune_prefix (LRU strategy)
            // The actual disk write is handled externally — this handler
            // just performs the cache eviction. The data could be saved
            // by the caller before calling this handler.
            cache.prune_prefix(offload_count)?;
            total_swapped += offload_count;
        }

        if total_swapped > 0 {
            Ok(ActionResult::Swapped {
                tokens_swapped: total_swapped,
            })
        } else {
            Ok(ActionResult::NoOp)
        }
    }

    fn name(&self) -> &str {
        "swap"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::shared_buffer::SharedBuffer;
    use crate::core::buffer::{Buffer, DType};
    use crate::core::kv_cache::KVCache;
    use crate::core::pressure::PressureLevel;
    use crate::core::shape::Shape;
    use crate::core::tensor::Tensor;
    use std::sync::Arc;

    fn make_cache(num_tokens: usize) -> KVCache {
        let max_seq = 100;
        let heads = 1;
        let dim = 4;
        let backend = Arc::new(CpuBackend::new());
        let buf_size = max_seq * heads * dim * 4;

        let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));

        let k = Tensor::new(
            Shape::new(vec![1, max_seq, heads, dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(Shape::new(vec![1, max_seq, heads, dim]), v_buf, backend);
        let mut cache = KVCache::new(k, v, max_seq);
        cache.current_pos = num_tokens;
        cache
    }

    #[test]
    fn test_swap_normal_noop() {
        let handler = SwapHandler::default();
        let mut caches = vec![make_cache(50)];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Normal,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };
        let result = handler.handle(&mut ctx).unwrap();
        assert!(!result.is_action());
        assert_eq!(ctx.caches[0].current_pos, 50);
    }

    #[test]
    fn test_swap_warning_offloads() {
        let handler = SwapHandler::new(0.5);
        let mut caches = vec![make_cache(50)];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Warning,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };
        let result = handler.handle(&mut ctx).unwrap();
        assert!(result.is_action());
        assert_eq!(ctx.caches[0].current_pos, 25); // 50 - 25 = 25
    }

    #[test]
    fn test_swap_emergency_offloads() {
        let handler = SwapHandler::new(0.75);
        let mut caches = vec![make_cache(40)];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };
        let result = handler.handle(&mut ctx).unwrap();
        assert!(result.is_action());
        assert_eq!(ctx.caches[0].current_pos, 10); // 40 - 30 = 10
    }

    #[test]
    fn test_swap_empty_cache() {
        let handler = SwapHandler::default();
        let mut caches = vec![make_cache(0)];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Warning,
            mem_available: 0,
            target_ratio: None,
            qcf_sink: None,
            layer_ratios: None,
        };
        let result = handler.handle(&mut ctx).unwrap();
        assert!(!result.is_action());
    }

    #[test]
    fn test_swap_name() {
        assert_eq!(SwapHandler::default().name(), "swap");
    }
}
