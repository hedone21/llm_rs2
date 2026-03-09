use anyhow::Result;

use crate::core::eviction::EvictionPolicy;
use crate::core::kv_cache::KVCache;
use crate::core::pressure::{ActionResult, CachePressurePipeline, HandlerContext, PressureLevel};
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

/// Orchestrates KV cache management based on memory pressure and policy decisions.
///
/// CacheManager supports two operating modes:
/// - **Legacy mode** (`new()`): single `EvictionPolicy`, backward-compatible
/// - **Pipeline mode** (`with_pipeline()`): multi-handler `CachePressurePipeline`,
///   dispatching different handlers based on `PressureLevel`
///
/// All public API methods work in both modes transparently.
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
    /// Optional pressure pipeline for multi-handler cache management.
    pipeline: Option<CachePressurePipeline>,
}

impl CacheManager {
    /// Create a CacheManager in legacy mode (single eviction policy).
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
            pipeline: None,
        }
    }

    /// Create a CacheManager in pipeline mode (multi-handler pressure pipeline).
    ///
    /// The pipeline dispatches different handlers based on `PressureLevel`,
    /// which is determined from available memory relative to `threshold_bytes`.
    pub fn with_pipeline(
        pipeline: CachePressurePipeline,
        monitor: Box<dyn SystemMonitor>,
        threshold_bytes: usize,
    ) -> Self {
        use crate::core::eviction::no_eviction::NoEvictionPolicy;
        Self {
            policy: Box::new(NoEvictionPolicy::new()),
            monitor,
            threshold_bytes,
            target_ratio: 0.75,
            pipeline: Some(pipeline),
        }
    }

    /// Determine pressure level from available memory.
    ///
    /// - `>= threshold`: Normal
    /// - `>= threshold / 2`: Warning
    /// - `>= threshold / 4`: Critical
    /// - `< threshold / 4`: Emergency
    fn determine_pressure_level(&self, mem_available: usize) -> PressureLevel {
        if mem_available >= self.threshold_bytes {
            PressureLevel::Normal
        } else if mem_available >= self.threshold_bytes / 2 {
            PressureLevel::Warning
        } else if mem_available >= self.threshold_bytes / 4 {
            PressureLevel::Critical
        } else {
            PressureLevel::Emergency
        }
    }

    /// Convert pipeline `ActionResult`s into a legacy `EvictionResult`.
    fn pipeline_results_to_eviction_result(
        results: &[ActionResult],
        caches: &[KVCache],
    ) -> EvictionResult {
        let mut total_removed = 0usize;
        let mut any_action = false;
        let mut last_new_pos = if caches.is_empty() {
            0
        } else {
            caches[0].current_pos
        };

        for r in results {
            match r {
                ActionResult::Evicted {
                    tokens_removed,
                    new_pos,
                } => {
                    total_removed += tokens_removed;
                    last_new_pos = *new_pos;
                    any_action = true;
                }
                ActionResult::NoOp => {}
                _ => {
                    // Non-eviction actions (quantize, swap, etc.) count as action
                    any_action = true;
                }
            }
        }

        EvictionResult {
            evicted: any_action,
            tokens_removed: total_removed,
            new_pos: last_new_pos,
        }
    }

    /// Check memory pressure and evict from all caches if needed.
    ///
    /// Called after each generation step in the inference loop.
    /// Routes to pipeline or legacy path depending on configuration.
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

        // Pipeline path
        if let Some(ref pipeline) = self.pipeline {
            let pressure = self.determine_pressure_level(mem_available);
            if pressure == PressureLevel::Normal {
                return Ok(EvictionResult {
                    evicted: false,
                    tokens_removed: 0,
                    new_pos: caches[0].current_pos,
                });
            }

            log::info!(
                "[CacheManager] Pipeline pressure={:?}, executing '{}'",
                pressure,
                pipeline.name(),
            );

            let mut ctx = HandlerContext {
                caches,
                importance: None,
                pressure_level: pressure,
                mem_available,
            };
            let results = pipeline.execute(&mut ctx)?;
            return Ok(Self::pipeline_results_to_eviction_result(
                &results, ctx.caches,
            ));
        }

        // Legacy path
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
    /// Routes to pipeline or legacy path depending on configuration.
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

        // Pipeline path
        if let Some(ref pipeline) = self.pipeline {
            let pressure = self.determine_pressure_level(mem_available);
            if pressure == PressureLevel::Normal {
                return Ok(EvictionResult {
                    evicted: false,
                    tokens_removed: 0,
                    new_pos: caches[0].current_pos,
                });
            }

            log::info!(
                "[CacheManager] Pipeline pressure={:?} (score-aware), executing '{}'",
                pressure,
                pipeline.name(),
            );

            let mut ctx = HandlerContext {
                caches,
                importance: Some(importance),
                pressure_level: pressure,
                mem_available,
            };
            let results = pipeline.execute(&mut ctx)?;
            return Ok(Self::pipeline_results_to_eviction_result(
                &results, ctx.caches,
            ));
        }

        // Legacy path
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

    /// Force eviction without scores, bypassing should_evict() and memory checks.
    ///
    /// Used when eviction is triggered externally (e.g., by resilience signals).
    /// In pipeline mode, runs at `Emergency` pressure level.
    pub fn force_evict(&self, caches: &mut [KVCache], target_ratio: f32) -> Result<EvictionResult> {
        if caches.is_empty() {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: 0,
            });
        }

        // Pipeline path — force at Emergency level
        if let Some(ref pipeline) = self.pipeline {
            log::info!(
                "[CacheManager] Signal-driven pipeline execution (Emergency): '{}'",
                pipeline.name(),
            );

            let mut ctx = HandlerContext {
                caches,
                importance: None,
                pressure_level: PressureLevel::Emergency,
                mem_available: 0,
            };
            let results = pipeline.execute(&mut ctx)?;
            return Ok(Self::pipeline_results_to_eviction_result(
                &results, ctx.caches,
            ));
        }

        // Legacy path
        let current_pos = caches[0].current_pos;
        let target_len = ((current_pos as f32) * target_ratio.clamp(0.1, 0.99)) as usize;
        let target_len = target_len.max(1);

        if current_pos <= target_len {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: current_pos,
            });
        }

        log::info!(
            "[CacheManager] Signal-driven eviction with policy '{}': {} → {} tokens",
            self.policy.name(),
            current_pos,
            target_len
        );

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

    /// Force eviction with importance scores, bypassing should_evict() and memory checks.
    ///
    /// Used when eviction is triggered externally (e.g., by resilience signals)
    /// for score-aware policies like H2O.
    /// In pipeline mode, runs at `Emergency` pressure level with scores.
    pub fn force_evict_with_scores(
        &self,
        caches: &mut [KVCache],
        target_ratio: f32,
        importance: &[f32],
    ) -> Result<EvictionResult> {
        if caches.is_empty() {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: 0,
            });
        }

        // Pipeline path — force at Emergency level with scores
        if let Some(ref pipeline) = self.pipeline {
            log::info!(
                "[CacheManager] Signal-driven pipeline execution (Emergency, score-aware): '{}'",
                pipeline.name(),
            );

            let mut ctx = HandlerContext {
                caches,
                importance: Some(importance),
                pressure_level: PressureLevel::Emergency,
                mem_available: 0,
            };
            let results = pipeline.execute(&mut ctx)?;
            return Ok(Self::pipeline_results_to_eviction_result(
                &results, ctx.caches,
            ));
        }

        // Legacy path
        let current_pos = caches[0].current_pos;
        let target_len = ((current_pos as f32) * target_ratio.clamp(0.1, 0.99)) as usize;
        let target_len = target_len.max(1);

        if current_pos <= target_len {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: current_pos,
            });
        }

        log::info!(
            "[CacheManager] Signal-driven eviction with policy '{}' (score-aware): {} → {} tokens",
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

    /// Returns the name of the active policy or pipeline.
    pub fn policy_name(&self) -> String {
        if let Some(ref pipeline) = self.pipeline {
            pipeline.name()
        } else {
            self.policy.name().to_string()
        }
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
        use crate::core::eviction::h2o::H2OPolicy;

        let cm = CacheManager::new(
            Box::new(H2OPolicy::new(5, 0.3, 0)), // prefix=4, keep_ratio=0.3
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

    // ── force_evict tests (signal-driven) ──

    #[test]
    fn test_force_evict_bypasses_should_evict() {
        // H2O's should_evict() always returns false, but force_evict must still work
        use crate::core::eviction::h2o::H2OPolicy;

        let cm = CacheManager::new(
            Box::new(H2OPolicy::new(5, 0.3, 0)),
            Box::new(MockMonitor {
                available: 1024 * 1024 * 1024, // plenty of memory
            }),
            256 * 1024 * 1024,
            0.75,
        );
        let mut caches = make_caches(4, 40);

        // maybe_evict should NOT trigger (should_evict=false, memory OK)
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(!result.evicted);
        assert_eq!(caches[0].current_pos, 40);

        // force_evict MUST trigger regardless
        let result = cm.force_evict(&mut caches, 0.5).unwrap();
        assert!(result.evicted);
        assert!(caches[0].current_pos < 40);
    }

    #[test]
    fn test_force_evict_with_scores_bypasses_checks() {
        use crate::core::eviction::h2o::H2OPolicy;

        let cm = CacheManager::new(
            Box::new(H2OPolicy::new(5, 0.3, 0)),
            Box::new(MockMonitor {
                available: 1024 * 1024 * 1024,
            }),
            256 * 1024 * 1024,
            0.75,
        );
        let mut caches = make_caches(4, 40);

        let mut importance = vec![0.0f32; 100];
        importance[10] = 10.0;
        importance[20] = 9.0;
        importance[30] = 8.0;

        let result = cm
            .force_evict_with_scores(&mut caches, 0.5, &importance)
            .unwrap();
        assert!(result.evicted);
        let pos = caches[0].current_pos;
        for cache in &caches {
            assert_eq!(cache.current_pos, pos);
        }
    }

    #[test]
    fn test_force_evict_empty_caches() {
        let cm = CacheManager::new(
            Box::new(NoEvictionPolicy::new()),
            Box::new(MockMonitor { available: 0 }),
            256 * 1024 * 1024,
            0.75,
        );
        let mut caches: Vec<KVCache> = Vec::new();
        let result = cm.force_evict(&mut caches, 0.5).unwrap();
        assert!(!result.evicted);
    }

    #[test]
    fn test_force_evict_ratio_clamping() {
        use crate::core::eviction::h2o::H2OPolicy;

        let cm = CacheManager::new(
            Box::new(H2OPolicy::new(0, 0.5, 0)),
            Box::new(MockMonitor { available: 0 }),
            0,
            0.75,
        );
        let mut caches = make_caches(1, 50);

        // target_ratio=0.0 should clamp to 0.1
        let result = cm.force_evict(&mut caches, 0.0).unwrap();
        assert!(result.evicted);
        assert!(caches[0].current_pos > 0);
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

    // ── Pipeline-backed CacheManager tests ──

    #[test]
    fn test_pipeline_manager_evicts_at_pressure() {
        use crate::core::pressure::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(EvictionHandler::new(
                Box::new(SlidingWindowPolicy::new(10, 0)),
                0.5,
            )),
        }]);

        let cm = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor {
                available: 100 * 1024 * 1024, // 100MB
            }),
            256 * 1024 * 1024, // 256MB threshold → Warning level
                               // (100MB >= 128MB=threshold/2 → Warning)
        );

        let mut caches = make_caches(4, 40);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(result.evicted);
        for cache in &caches {
            assert!(cache.current_pos < 40);
        }
    }

    #[test]
    fn test_pipeline_manager_no_action_at_normal() {
        use crate::core::pressure::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(EvictionHandler::new(
                Box::new(SlidingWindowPolicy::new(10, 0)),
                0.5,
            )),
        }]);

        let cm = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor {
                available: 512 * 1024 * 1024, // 512MB — above 256MB threshold → Normal
            }),
            256 * 1024 * 1024,
        );

        let mut caches = make_caches(4, 40);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(!result.evicted);
        assert_eq!(caches[0].current_pos, 40);
    }

    #[test]
    fn test_pipeline_manager_force_evict() {
        use crate::core::pressure::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Emergency,
            handler: Box::new(EvictionHandler::new(
                Box::new(SlidingWindowPolicy::new(10, 0)),
                0.5,
            )),
        }]);

        let cm = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor {
                available: 1024 * 1024 * 1024, // plenty of memory
            }),
            256 * 1024 * 1024,
        );

        // maybe_evict should NOT trigger (Normal pressure)
        let mut caches = make_caches(4, 40);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(!result.evicted);

        // force_evict MUST trigger (Emergency level)
        let result = cm.force_evict(&mut caches, 0.5).unwrap();
        assert!(result.evicted);
        assert!(caches[0].current_pos < 40);
    }

    #[test]
    fn test_pipeline_manager_force_evict_with_scores() {
        use crate::core::eviction::h2o::H2OPolicy;
        use crate::core::pressure::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Emergency,
            handler: Box::new(EvictionHandler::new(
                Box::new(H2OPolicy::new(5, 0.5, 0)),
                0.5,
            )),
        }]);

        let cm = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor {
                available: 1024 * 1024 * 1024,
            }),
            256 * 1024 * 1024,
        );

        let mut caches = make_caches(4, 40);
        let mut importance = vec![0.0f32; 100];
        importance[10] = 10.0;
        importance[20] = 9.0;

        let result = cm
            .force_evict_with_scores(&mut caches, 0.5, &importance)
            .unwrap();
        assert!(result.evicted);
        assert!(caches[0].current_pos < 40);
    }

    #[test]
    fn test_pipeline_manager_with_scores() {
        use crate::core::eviction::h2o::H2OPolicy;
        use crate::core::pressure::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(EvictionHandler::new(
                Box::new(H2OPolicy::new(5, 0.5, 0)),
                0.5,
            )),
        }]);

        let cm = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor {
                available: 100 * 1024 * 1024, // Warning level
            }),
            256 * 1024 * 1024,
        );

        let mut caches = make_caches(4, 40);
        let mut importance = vec![0.0f32; 100];
        importance[10] = 10.0;
        importance[20] = 9.0;
        for i in 4..40 {
            if importance[i] == 0.0 {
                importance[i] = 0.01;
            }
        }

        let result = cm
            .maybe_evict_with_scores(&mut caches, &importance)
            .unwrap();
        assert!(result.evicted);
        let pos = caches[0].current_pos;
        for cache in &caches {
            assert_eq!(cache.current_pos, pos);
        }
    }

    #[test]
    fn test_pipeline_manager_policy_name() {
        use crate::core::pressure::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        let pipeline = CachePressurePipeline::new(vec![
            PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(EvictionHandler::new(
                    Box::new(SlidingWindowPolicy::new(10, 0)),
                    0.8,
                )),
            },
            PressureStageConfig {
                min_level: PressureLevel::Critical,
                handler: Box::new(EvictionHandler::new(
                    Box::new(SlidingWindowPolicy::new(10, 0)),
                    0.5,
                )),
            },
        ]);

        let cm = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor { available: 0 }),
            256 * 1024 * 1024,
        );

        let name = cm.policy_name();
        assert!(name.contains("sliding_window"));
        assert!(name.contains("Warning"));
        assert!(name.contains("Critical"));
    }

    #[test]
    fn test_pipeline_manager_multi_level_graduated_response() {
        use crate::core::pressure::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        // Two eviction stages: mild at Warning, aggressive at Critical
        let pipeline = CachePressurePipeline::new(vec![
            PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(EvictionHandler::new(
                    Box::new(SlidingWindowPolicy::new(30, 0)),
                    0.9, // keep 90%
                )),
            },
            PressureStageConfig {
                min_level: PressureLevel::Critical,
                handler: Box::new(EvictionHandler::new(
                    Box::new(SlidingWindowPolicy::new(10, 0)),
                    0.5, // keep 50%
                )),
            },
        ]);

        // At Warning level: only the first stage should run
        let cm_warning = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor {
                available: 200 * 1024 * 1024, // 200MB, threshold=400MB → Warning
            }),
            400 * 1024 * 1024,
        );

        let mut caches = make_caches(4, 40);
        let result = cm_warning.maybe_evict(&mut caches).unwrap();
        assert!(result.evicted);
        let pos_after_warning = caches[0].current_pos;

        // At Critical level: both stages should run (more aggressive)
        let pipeline2 = CachePressurePipeline::new(vec![
            PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(EvictionHandler::new(
                    Box::new(SlidingWindowPolicy::new(30, 0)),
                    0.9,
                )),
            },
            PressureStageConfig {
                min_level: PressureLevel::Critical,
                handler: Box::new(EvictionHandler::new(
                    Box::new(SlidingWindowPolicy::new(10, 0)),
                    0.5,
                )),
            },
        ]);

        let cm_critical = CacheManager::with_pipeline(
            pipeline2,
            Box::new(MockMonitor {
                available: 50 * 1024 * 1024, // 50MB, threshold=400MB → Critical
            }),
            400 * 1024 * 1024,
        );

        let mut caches2 = make_caches(4, 40);
        let result2 = cm_critical.maybe_evict(&mut caches2).unwrap();
        assert!(result2.evicted);
        let pos_after_critical = caches2[0].current_pos;

        // Critical should be more aggressive than Warning
        assert!(
            pos_after_critical <= pos_after_warning,
            "Critical ({}) should evict at least as much as Warning ({})",
            pos_after_critical,
            pos_after_warning,
        );
    }

    #[test]
    fn test_pipeline_manager_empty_pipeline() {
        use crate::core::pressure::CachePressurePipeline;

        let pipeline = CachePressurePipeline::new(vec![]);
        let cm = CacheManager::with_pipeline(
            pipeline,
            Box::new(MockMonitor { available: 0 }),
            256 * 1024 * 1024,
        );

        let mut caches = make_caches(4, 40);
        // Emergency level but empty pipeline → no action
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(!result.evicted);
    }

    #[test]
    fn test_pipeline_manager_monitor_error_skips() {
        use crate::core::pressure::{
            CachePressurePipeline, EvictionHandler, PressureLevel, PressureStageConfig,
        };

        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(EvictionHandler::new(
                Box::new(SlidingWindowPolicy::new(10, 0)),
                0.5,
            )),
        }]);

        let cm = CacheManager::with_pipeline(pipeline, Box::new(ErrorMonitor), 256 * 1024 * 1024);

        let mut caches = make_caches(4, 40);
        let result = cm.maybe_evict(&mut caches).unwrap();
        assert!(!result.evicted);
        assert_eq!(result.new_pos, 40);
    }
}
