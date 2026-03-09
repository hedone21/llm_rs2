//! Bridge between existing `EvictionPolicy` and the new `CachePressureHandler` trait.
//!
//! Wraps any `Box<dyn EvictionPolicy>` so that H2O and SlidingWindow policies
//! work seamlessly inside a `CachePressurePipeline`.

use super::{ActionResult, CachePressureHandler, HandlerContext};
use crate::core::eviction::EvictionPolicy;
use anyhow::Result;

/// Adapts an `EvictionPolicy` to the `CachePressureHandler` interface.
///
/// The wrapped policy's `evict()` / `evict_with_scores()` is called with
/// `target_len = current_pos * target_ratio`.
pub struct EvictionHandler {
    policy: Box<dyn EvictionPolicy>,
    target_ratio: f32,
}

impl EvictionHandler {
    /// Create a new eviction handler wrapping the given policy.
    ///
    /// `target_ratio` controls how aggressively to evict:
    /// - 0.8 = keep 80% of tokens (mild)
    /// - 0.5 = keep 50% of tokens (aggressive)
    pub fn new(policy: Box<dyn EvictionPolicy>, target_ratio: f32) -> Self {
        Self {
            policy,
            target_ratio: target_ratio.clamp(0.1, 0.99),
        }
    }
}

impl CachePressureHandler for EvictionHandler {
    fn handle(&self, ctx: &mut HandlerContext) -> Result<ActionResult> {
        if ctx.caches.is_empty() {
            return Ok(ActionResult::NoOp);
        }

        let current_pos = ctx.caches[0].current_pos;
        let target_len = ((current_pos as f32) * self.target_ratio) as usize;
        let target_len = target_len.max(1);

        if current_pos <= target_len {
            return Ok(ActionResult::NoOp);
        }

        log::debug!(
            "[EvictionHandler] policy='{}': {} → {} tokens",
            self.policy.name(),
            current_pos,
            target_len,
        );

        for cache in ctx.caches.iter_mut() {
            if let (Some(flat), Some(head_imp)) = (ctx.importance, ctx.head_importance) {
                if ctx.n_kv_heads > 0 {
                    self.policy.evict_with_head_scores(
                        cache,
                        target_len,
                        flat,
                        head_imp,
                        ctx.n_kv_heads,
                    )?;
                } else {
                    self.policy.evict_with_scores(cache, target_len, flat)?;
                }
            } else if let Some(importance) = ctx.importance {
                self.policy
                    .evict_with_scores(cache, target_len, importance)?;
            } else {
                self.policy.evict(cache, target_len)?;
            }
        }

        let new_pos = ctx.caches[0].current_pos;
        Ok(ActionResult::Evicted {
            tokens_removed: current_pos - new_pos,
            new_pos,
        })
    }

    fn name(&self) -> &str {
        self.policy.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::shared_buffer::SharedBuffer;
    use crate::core::buffer::DType;
    use crate::core::eviction::h2o::H2OPolicy;
    use crate::core::eviction::no_eviction::NoEvictionPolicy;
    use crate::core::eviction::sliding_window::SlidingWindowPolicy;
    use crate::core::kv_cache::KVCache;
    use crate::core::pressure::PressureLevel;
    use crate::core::shape::Shape;
    use crate::core::tensor::Tensor;
    use std::sync::Arc;

    fn make_caches(n_layers: usize, pos: usize) -> Vec<KVCache> {
        let max_seq = 100;
        let backend = Arc::new(CpuBackend::new());
        (0..n_layers)
            .map(|_| {
                let buf_size = max_seq * 1 * 4 * 4;
                let k = Tensor::new(
                    Shape::new(vec![1, max_seq, 1, 4]),
                    Arc::new(SharedBuffer::new(buf_size, DType::F32)),
                    backend.clone(),
                );
                let v = Tensor::new(
                    Shape::new(vec![1, max_seq, 1, 4]),
                    Arc::new(SharedBuffer::new(buf_size, DType::F32)),
                    backend.clone(),
                );
                let mut cache = KVCache::new(k, v, max_seq);
                cache.current_pos = pos;
                cache
            })
            .collect()
    }

    #[test]
    fn test_wraps_sliding_window() {
        let handler = EvictionHandler::new(
            Box::new(SlidingWindowPolicy::new(10, 0)), // window=10, prefix=4(clamped)
            0.5,
        );

        let mut caches = make_caches(4, 40);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
        };

        let result = handler.handle(&mut ctx).unwrap();
        match result {
            ActionResult::Evicted {
                tokens_removed,
                new_pos,
            } => {
                assert!(tokens_removed > 0);
                assert!(new_pos < 40);
                // SlidingWindow with window=10, prefix=4 → max_keep=14
                assert!(new_pos <= 14);
            }
            _ => panic!("Expected Evicted, got {:?}", result),
        }

        // All layers should have the same position
        let pos = ctx.caches[0].current_pos;
        for cache in ctx.caches.iter() {
            assert_eq!(cache.current_pos, pos);
        }
    }

    #[test]
    fn test_wraps_h2o_with_scores() {
        let handler = EvictionHandler::new(
            Box::new(H2OPolicy::new(5, 0.5, 0)), // prefix=4(clamped), keep_ratio=0.5
            0.5,
        );

        let mut caches = make_caches(4, 40);
        let mut importance = vec![0.0f32; 100];
        importance[10] = 10.0;
        importance[20] = 9.0;
        importance[30] = 8.0;
        for i in 4..40 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
        };

        let result = handler.handle(&mut ctx).unwrap();
        match result {
            ActionResult::Evicted {
                tokens_removed,
                new_pos,
            } => {
                assert!(tokens_removed > 0);
                assert!(new_pos < 40);
                // H2O: target = 40 * 0.5 = 20
                assert_eq!(new_pos, 20);
            }
            _ => panic!("Expected Evicted, got {:?}", result),
        }
    }

    #[test]
    fn test_noop_when_below_target() {
        // target_len = 1 * 0.99 = 0 → clamped to 1.
        // current_pos=1 <= target_len=1 → NoOp.
        let handler = EvictionHandler::new(Box::new(SlidingWindowPolicy::new(50, 0)), 0.99);

        let mut caches = make_caches(2, 1);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Warning,
            mem_available: 0,
        };

        let result = handler.handle(&mut ctx).unwrap();
        assert!(matches!(result, ActionResult::NoOp));
        assert_eq!(ctx.caches[0].current_pos, 1); // unchanged
    }

    #[test]
    fn test_noop_on_empty_caches() {
        let handler = EvictionHandler::new(Box::new(SlidingWindowPolicy::new(10, 0)), 0.5);

        let mut caches: Vec<KVCache> = vec![];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
        };

        let result = handler.handle(&mut ctx).unwrap();
        assert!(matches!(result, ActionResult::NoOp));
    }

    #[test]
    fn test_name_delegates_to_policy() {
        let handler = EvictionHandler::new(Box::new(SlidingWindowPolicy::new(10, 0)), 0.5);
        assert_eq!(handler.name(), "sliding_window");

        let handler = EvictionHandler::new(Box::new(H2OPolicy::new(5, 0.5, 0)), 0.5);
        assert_eq!(handler.name(), "h2o");

        let handler = EvictionHandler::new(Box::new(NoEvictionPolicy::new()), 0.5);
        assert_eq!(handler.name(), "none");
    }

    #[test]
    fn test_target_ratio_clamping() {
        // ratio=0.0 should clamp to 0.1
        let handler = EvictionHandler::new(Box::new(SlidingWindowPolicy::new(10, 0)), 0.0);

        let mut caches = make_caches(1, 50);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
        };

        let result = handler.handle(&mut ctx).unwrap();
        match result {
            ActionResult::Evicted { new_pos, .. } => {
                assert!(new_pos > 0, "Should keep at least 1 token");
            }
            _ => panic!("Expected Evicted"),
        }
    }

    #[test]
    fn test_h2o_fallback_without_scores() {
        // H2O without importance scores → fallback to sliding-window-like behavior
        let handler = EvictionHandler::new(Box::new(H2OPolicy::new(5, 0.5, 0)), 0.5);

        let mut caches = make_caches(2, 30);
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None, // no scores
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: PressureLevel::Critical,
            mem_available: 0,
        };

        let result = handler.handle(&mut ctx).unwrap();
        match result {
            ActionResult::Evicted {
                tokens_removed,
                new_pos,
            } => {
                assert!(tokens_removed > 0);
                assert!(new_pos < 30);
            }
            _ => panic!("Expected Evicted"),
        }
    }
}
