use anyhow::Result;

use crate::core::eviction::EvictionPolicy;
use crate::core::kv_cache::KVCache;
use crate::core::sys_monitor::SystemMonitor;

/// Result of an eviction attempt.
#[derive(Debug, Clone)]
pub struct EvictionResult {
    /// Whether eviction was actually performed.
    pub evicted: bool,
    /// Number of tokens removed per cache.
    pub tokens_removed: usize,
    /// New position after eviction.
    pub new_pos: usize,
}

/// Orchestrates KV cache eviction based on memory pressure and policy decisions.
///
/// CacheManager follows the Dependency Inversion principle:
/// - Depends on `dyn EvictionPolicy` (abstraction), not concrete policies
/// - Depends on `dyn SystemMonitor` (abstraction), not OS-specific implementations
pub struct CacheManager {
    policy: Box<dyn EvictionPolicy>,
    monitor: Box<dyn SystemMonitor>,
    /// Eviction triggers when available memory drops below this threshold (bytes).
    threshold_bytes: usize,
    /// After eviction, target cache size = current_pos * target_ratio.
    target_ratio: f32,
}

impl CacheManager {
    pub fn new(
        policy: Box<dyn EvictionPolicy>,
        monitor: Box<dyn SystemMonitor>,
        threshold_bytes: usize,
        target_ratio: f32,
    ) -> Self {
        Self {
            policy,
            monitor,
            threshold_bytes,
            target_ratio: target_ratio.clamp(0.1, 0.99),
        }
    }

    /// Check memory pressure and evict from all caches if needed.
    ///
    /// Called after each generation step in the inference loop.
    pub fn maybe_evict(&self, caches: &mut [KVCache]) -> Result<EvictionResult> {
        if caches.is_empty() {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: 0,
            });
        }

        // Query system memory
        let mem_available = match self.monitor.mem_stats() {
            Ok(stats) => stats.available,
            Err(e) => {
                log::warn!("Failed to read memory stats: {}, skipping eviction", e);
                return Ok(EvictionResult {
                    evicted: false,
                    tokens_removed: 0,
                    new_pos: caches[0].current_pos,
                });
            }
        };

        // Check if any cache needs eviction
        let representative_cache = &caches[0];
        if !self
            .policy
            .should_evict(representative_cache, mem_available)
        {
            // Also check memory threshold
            if mem_available >= self.threshold_bytes {
                return Ok(EvictionResult {
                    evicted: false,
                    tokens_removed: 0,
                    new_pos: representative_cache.current_pos,
                });
            }
        }

        // Calculate target length
        let current_pos = representative_cache.current_pos;
        let target_len = ((current_pos as f32) * self.target_ratio) as usize;
        let target_len = target_len.max(1); // Keep at least 1 token

        if current_pos <= target_len {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: current_pos,
            });
        }

        log::info!(
            "[CacheManager] Memory pressure detected (available: {} MB, threshold: {} MB). ",
            mem_available / (1024 * 1024),
            self.threshold_bytes / (1024 * 1024),
        );
        log::info!(
            "[CacheManager] Evicting with policy '{}': {} → {} tokens",
            self.policy.name(),
            current_pos,
            target_len
        );

        // Evict from all caches (one per transformer layer)
        let tokens_removed = current_pos - target_len;
        for cache in caches.iter_mut() {
            self.policy.evict(cache, target_len)?;
        }

        let new_pos = caches[0].current_pos;

        Ok(EvictionResult {
            evicted: true,
            tokens_removed,
            new_pos,
        })
    }

    /// Check memory pressure and evict using importance scores.
    ///
    /// Same logic as `maybe_evict()`, but passes importance scores to the policy
    /// via `evict_with_scores()`. Used when `AttentionScoreAccumulator` is active.
    pub fn maybe_evict_with_scores(
        &self,
        caches: &mut [KVCache],
        importance: &[f32],
    ) -> Result<EvictionResult> {
        if caches.is_empty() {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: 0,
            });
        }

        let mem_available = match self.monitor.mem_stats() {
            Ok(stats) => stats.available,
            Err(e) => {
                log::warn!("Failed to read memory stats: {}, skipping eviction", e);
                return Ok(EvictionResult {
                    evicted: false,
                    tokens_removed: 0,
                    new_pos: caches[0].current_pos,
                });
            }
        };

        let representative_cache = &caches[0];
        if !self
            .policy
            .should_evict(representative_cache, mem_available)
            && mem_available >= self.threshold_bytes
        {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: representative_cache.current_pos,
            });
        }

        let current_pos = representative_cache.current_pos;
        let target_len = ((current_pos as f32) * self.target_ratio) as usize;
        let target_len = target_len.max(1);

        if current_pos <= target_len {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: current_pos,
            });
        }

        log::info!(
            "[CacheManager] Evicting with policy '{}' (score-aware): {} → {} tokens",
            self.policy.name(),
            current_pos,
            target_len
        );

        let tokens_removed = current_pos - target_len;
        for cache in caches.iter_mut() {
            self.policy
                .evict_with_scores(cache, target_len, importance)?;
        }

        let new_pos = caches[0].current_pos;

        Ok(EvictionResult {
            evicted: true,
            tokens_removed,
            new_pos,
        })
    }

    /// Returns the name of the active policy.
    pub fn policy_name(&self) -> &str {
        self.policy.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::shared_buffer::SharedBuffer;
    use crate::core::buffer::DType;
    use crate::core::eviction::no_eviction::NoEvictionPolicy;
    use crate::core::eviction::sliding_window::SlidingWindowPolicy;
    use crate::core::shape::Shape;
    use crate::core::sys_monitor::MemoryStats;
    use crate::core::tensor::Tensor;
    use std::sync::Arc;

    /// Mock SystemMonitor for testing
    struct MockMonitor {
        available: usize,
    }

    impl SystemMonitor for MockMonitor {
        fn mem_stats(&self) -> Result<MemoryStats> {
            Ok(MemoryStats {
                total: 4 * 1024 * 1024 * 1024,
                available: self.available,
                free: self.available / 2,
            })
        }
    }

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
    fn test_no_eviction_with_plenty_memory() {
        let cm = CacheManager::new(
            Box::new(NoEvictionPolicy::new()),
            Box::new(MockMonitor {
                available: 1024 * 1024 * 1024,
            }), // 1GB
            256 * 1024 * 1024, // 256MB threshold
            0.75,
        );
        let mut caches = make_caches(4, 50);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(!result.evicted);
        assert_eq!(caches[0].current_pos, 50);
    }

    #[test]
    fn test_sliding_window_with_memory_pressure() {
        let cm = CacheManager::new(
            Box::new(SlidingWindowPolicy::new(30, 0)),
            Box::new(MockMonitor {
                available: 100 * 1024 * 1024,
            }), // 100MB (below threshold)
            256 * 1024 * 1024, // 256MB threshold
            0.75,
        );
        let mut caches = make_caches(4, 50);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(result.evicted);
        // target = 50 * 0.75 = 37, but window + prefix = 30
        // since should_evict triggered via memory threshold
        for cache in &caches {
            assert!(cache.current_pos < 50);
        }
    }

    #[test]
    fn test_eviction_across_all_layers() {
        let cm = CacheManager::new(
            Box::new(SlidingWindowPolicy::new(20, 0)),
            Box::new(MockMonitor {
                available: 10 * 1024 * 1024,
            }), // Very low
            256 * 1024 * 1024,
            0.5,
        );
        let mut caches = make_caches(16, 40);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(result.evicted);
        // All 16 layers should have the same position
        let pos = caches[0].current_pos;
        for cache in &caches {
            assert_eq!(cache.current_pos, pos);
        }
    }

    #[test]
    fn test_empty_caches() {
        let cm = CacheManager::new(
            Box::new(NoEvictionPolicy::new()),
            Box::new(MockMonitor { available: 0 }),
            256 * 1024 * 1024,
            0.75,
        );
        let mut caches: Vec<KVCache> = Vec::new();
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(!result.evicted);
    }

    #[test]
    fn test_policy_name() {
        let cm = CacheManager::new(
            Box::new(SlidingWindowPolicy::new(10, 0)),
            Box::new(MockMonitor { available: 0 }),
            0,
            0.75,
        );
        assert_eq!(cm.policy_name(), "sliding_window");
    }

    /// Mock monitor that always returns an error
    struct ErrorMonitor;
    impl SystemMonitor for ErrorMonitor {
        fn mem_stats(&self) -> Result<MemoryStats> {
            Err(anyhow::anyhow!("simulated monitor failure"))
        }
    }

    #[test]
    fn test_monitor_error_skips_eviction() {
        let cm = CacheManager::new(
            Box::new(SlidingWindowPolicy::new(10, 0)),
            Box::new(ErrorMonitor),
            256 * 1024 * 1024,
            0.75,
        );
        let mut caches = make_caches(4, 50);
        let result = cm.maybe_evict(&mut caches).unwrap();
        // Should not evict when monitor fails
        assert!(!result.evicted);
        assert_eq!(result.new_pos, 50);
    }

    #[test]
    fn test_maybe_evict_with_scores_triggers() {
        use crate::core::eviction::snap_kv::SnapKVPolicy;

        let cm = CacheManager::new(
            Box::new(SnapKVPolicy::new(5, 0.3, 0)), // prefix=4, keep_ratio=0.3
            Box::new(MockMonitor {
                available: 10 * 1024 * 1024,
            }),
            256 * 1024 * 1024,
            0.5,
        );
        let mut caches = make_caches(4, 40);

        let mut importance = vec![0.0f32; 100];
        // Give some tokens high importance
        importance[10] = 10.0;
        importance[20] = 9.0;
        importance[30] = 8.0;

        let result = cm
            .maybe_evict_with_scores(&mut caches, &importance)
            .unwrap();
        assert!(result.evicted);
        // All layers should have the same position
        let pos = caches[0].current_pos;
        for cache in &caches {
            assert_eq!(cache.current_pos, pos);
        }
    }

    #[test]
    fn test_maybe_evict_with_scores_no_eviction_needed() {
        let cm = CacheManager::new(
            Box::new(NoEvictionPolicy::new()),
            Box::new(MockMonitor {
                available: 1024 * 1024 * 1024,
            }),
            256 * 1024 * 1024,
            0.75,
        );
        let mut caches = make_caches(4, 50);
        let importance = vec![1.0f32; 100];

        let result = cm
            .maybe_evict_with_scores(&mut caches, &importance)
            .unwrap();
        assert!(!result.evicted);
        assert_eq!(caches[0].current_pos, 50);
    }

    #[test]
    fn test_target_ratio_clamping() {
        // target_ratio below 0.1 should be clamped to 0.1
        let cm = CacheManager::new(
            Box::new(SlidingWindowPolicy::new(10, 0)),
            Box::new(MockMonitor { available: 10 }),
            256 * 1024 * 1024,
            0.01, // should clamp to 0.1
        );
        let mut caches = make_caches(1, 50);
        let result = cm.maybe_evict(&mut caches).unwrap();
        // target = 50 * 0.1 = 5, but sliding_window max_keep = 14
        // So keep = clamp(5, min_keep, 14)
        assert!(result.evicted);
        // The result should have new_pos > 0 (at least 1)
        assert!(caches[0].current_pos > 0);

        // target_ratio above 0.99 should be clamped to 0.99
        let cm2 = CacheManager::new(
            Box::new(SlidingWindowPolicy::new(10, 0)),
            Box::new(MockMonitor { available: 10 }),
            256 * 1024 * 1024,
            5.0, // should clamp to 0.99
        );
        let mut caches2 = make_caches(1, 50);
        let result2 = cm2.maybe_evict(&mut caches2).unwrap();
        // target = 50 * 0.99 = 49, but max_keep=14, so keep=14
        // Since eviction still happens (50 > 14 because threshold triggers)
        assert!(result2.evicted);
    }
}
