//! SnapKV-based KV cache compression handler.
//!
//! Applies one-shot prefill-time compression: uses attention score voting
//! from an observation window to select important prefix tokens per head.
//! Discards the rest, then concatenates selected prefix + observation window.
//!
//! Reference: SnapKV (arXiv 2024) — "LLM Knows What You are Looking for Before Generation"

use super::{ActionResult, CachePressureHandler, HandlerContext};
use crate::core::math_utils::{avg_pool_1d, topk_indices_per_head};
use anyhow::Result;

/// SnapKV-based one-shot KV cache compression.
///
/// Applied once after prefill to reduce KV cache size.
/// Uses per-head importance voting with pooled smoothing and top-k selection.
pub struct SnapKVHandler {
    /// Observation window size (last N tokens used for voting). Default: 32.
    pub window_size: usize,
    /// Maximum capacity after compression. Default: 1024.
    pub max_capacity: usize,
    /// 1D average pooling kernel size for vote smoothing. Default: 5.
    pub kernel_size: usize,
}

impl Default for SnapKVHandler {
    fn default() -> Self {
        Self {
            window_size: 32,
            max_capacity: 1024,
            kernel_size: 5,
        }
    }
}

impl SnapKVHandler {
    pub fn new(window_size: usize, max_capacity: usize, kernel_size: usize) -> Self {
        Self {
            window_size,
            max_capacity,
            kernel_size,
        }
    }
}

impl CachePressureHandler for SnapKVHandler {
    fn handle(&self, ctx: &mut HandlerContext) -> Result<ActionResult> {
        let n_kv_heads = ctx.n_kv_heads;
        if n_kv_heads == 0 {
            return Ok(ActionResult::NoOp);
        }

        let mut total_removed = 0;

        for cache in ctx.caches.iter_mut() {
            let total_len = cache.current_pos;
            if total_len <= self.max_capacity {
                continue;
            }

            let window = self.window_size.min(total_len);
            let prefix_len = total_len.saturating_sub(window);
            if prefix_len == 0 {
                continue;
            }

            let keep_count = self.max_capacity.saturating_sub(window);
            if keep_count == 0 || keep_count >= prefix_len {
                continue;
            }

            // Build per-head votes from importance scores
            let keep_indices = if let Some(head_imp) = ctx.head_importance {
                // Per-head voting using head_importance[kv_h * max_seq + pos]
                let max_seq = head_imp.len() / n_kv_heads;
                let mut flat_votes = vec![0.0f32; n_kv_heads * prefix_len];

                for h in 0..n_kv_heads {
                    let dst = &mut flat_votes[h * prefix_len..(h + 1) * prefix_len];
                    for pos in 0..prefix_len {
                        dst[pos] = head_imp[h * max_seq + pos];
                    }
                    avg_pool_1d(dst, self.kernel_size);
                }

                topk_indices_per_head(&flat_votes, n_kv_heads, prefix_len, keep_count)
            } else if let Some(importance) = ctx.importance {
                // Fallback: global importance → all heads select same tokens
                let mut votes = importance[..prefix_len].to_vec();
                avg_pool_1d(&mut votes, self.kernel_size);

                let flat: Vec<f32> = (0..n_kv_heads).flat_map(|_| votes.iter().copied()).collect();
                topk_indices_per_head(&flat, n_kv_heads, prefix_len, keep_count)
            } else {
                // No scores available — cannot compress
                continue;
            };

            let before = cache.current_pos;
            cache.compress_per_head(&keep_indices, prefix_len)?;
            total_removed += before.saturating_sub(cache.current_pos);
        }

        if total_removed > 0 {
            Ok(ActionResult::Compressed {
                tokens_removed: total_removed,
            })
        } else {
            Ok(ActionResult::NoOp)
        }
    }

    fn name(&self) -> &str {
        "snapkv"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::shared_buffer::SharedBuffer;
    use crate::core::buffer::{Buffer, DType};
    use crate::core::kv_cache::{KVCache, KVLayout};
    use crate::core::pressure::PressureLevel;
    use crate::core::shape::Shape;
    use crate::core::tensor::Tensor;
    use std::sync::Arc;

    fn make_headmajor_cache(num_tokens: usize, kv_heads: usize, head_dim: usize) -> KVCache {
        let max_seq = 256;
        let backend = Arc::new(CpuBackend::new());
        let buf_size = max_seq * kv_heads * head_dim * 4;

        let k_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(buf_size, DType::F32));

        // Fill with recognizable data in HeadMajor layout:
        // offset = head * max_seq * head_dim + pos * head_dim + d
        unsafe {
            let k_ptr = k_buf.as_mut_ptr() as *mut f32;
            let v_ptr = v_buf.as_mut_ptr() as *mut f32;
            for h in 0..kv_heads {
                for pos in 0..num_tokens {
                    for d in 0..head_dim {
                        let idx = h * max_seq * head_dim + pos * head_dim + d;
                        *k_ptr.add(idx) = (pos * 10 + h) as f32 + d as f32 * 0.01;
                        *v_ptr.add(idx) = (pos * 10 + h) as f32 + d as f32 * 0.01 + 1000.0;
                    }
                }
            }
        }

        // HeadMajor shape: [1, kv_heads, max_seq, head_dim]
        let k = Tensor::new(
            Shape::new(vec![1, kv_heads, max_seq, head_dim]),
            k_buf,
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, kv_heads, max_seq, head_dim]),
            v_buf,
            backend,
        );
        let mut cache = KVCache::new(k, v, max_seq).with_layout(KVLayout::HeadMajor);
        cache.current_pos = num_tokens;
        cache
    }

    #[test]
    fn test_snapkv_compress_size() {
        let kv_heads = 2;
        let head_dim = 4;
        let num_tokens = 100;
        let mut cache = make_headmajor_cache(num_tokens, kv_heads, head_dim);

        // Importance: linearly increasing → prefers later positions
        let max_seq = 256;
        let mut importance = vec![0.0f32; max_seq];
        for i in 0..num_tokens {
            importance[i] = i as f32;
        }

        let mut head_importance = vec![0.0f32; kv_heads * max_seq];
        for h in 0..kv_heads {
            for i in 0..num_tokens {
                head_importance[h * max_seq + i] = i as f32 + h as f32 * 0.1;
            }
        }

        let handler = SnapKVHandler::new(32, 50, 5);
        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: Some(&head_importance),
            n_kv_heads: kv_heads,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        assert!(result.is_action());

        // After compression: max_capacity = 50
        assert_eq!(ctx.caches[0].current_pos, 50);
    }

    #[test]
    fn test_snapkv_no_compress_when_small() {
        let kv_heads = 2;
        let head_dim = 4;
        let mut cache = make_headmajor_cache(30, kv_heads, head_dim);

        let importance = vec![1.0f32; 256];
        let handler = SnapKVHandler::new(32, 50, 5);
        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: kv_heads,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        assert!(!result.is_action()); // 30 <= 50, no compression
        assert_eq!(ctx.caches[0].current_pos, 30);
    }

    #[test]
    fn test_snapkv_no_scores() {
        let kv_heads = 2;
        let head_dim = 4;
        let mut cache = make_headmajor_cache(100, kv_heads, head_dim);

        let handler = SnapKVHandler::new(32, 50, 5);
        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: None,
            head_importance: None,
            n_kv_heads: kv_heads,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: None,
        };

        let result = handler.handle(&mut ctx).unwrap();
        assert!(!result.is_action()); // No scores, can't compress
    }

    #[test]
    fn test_snapkv_name() {
        let handler = SnapKVHandler::default();
        assert_eq!(handler.name(), "snapkv");
    }

    #[test]
    fn test_snapkv_window_preserved() {
        // After compression, the observation window tokens should be the last ones
        let kv_heads = 1;
        let head_dim = 4;
        let num_tokens = 80;
        let window = 16;
        let max_cap = 32;
        let mut cache = make_headmajor_cache(num_tokens, kv_heads, head_dim);

        let max_seq = 256;
        let mut importance = vec![0.0f32; max_seq];
        for i in 0..num_tokens {
            importance[i] = i as f32;
        }

        let handler = SnapKVHandler::new(window, max_cap, 5);
        let mut caches = vec![cache];
        let mut ctx = HandlerContext {
            caches: &mut caches,
            importance: Some(&importance),
            head_importance: None,
            n_kv_heads: kv_heads,
            pressure_level: PressureLevel::Emergency,
            mem_available: 0,
            target_ratio: None,
        };

        handler.handle(&mut ctx).unwrap();
        assert_eq!(ctx.caches[0].current_pos, max_cap);
    }
}
