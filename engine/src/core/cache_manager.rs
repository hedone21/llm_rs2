use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;

use crate::core::events::{CacheEvent, EventSink, NoOpSink};
use crate::core::eviction::EvictionPolicy;
use crate::core::kv_cache::{KVCache, max_cache_pos};
use crate::core::pressure::{
    ActionResult, CachePressurePipeline, EvictionHandler, HandlerContext, PressureLevel,
    PressureStageConfig,
};
use crate::core::sys_monitor::SystemMonitor;
use crate::resilience::EvictMethod;

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

/// Score context variants for the unified dispatch path.
pub enum ScoreContext<'a> {
    /// No importance scores available.
    None,
    /// Flat per-token importance scores.
    Flat { importance: &'a [f32] },
    /// Per-KV-head importance scores (GQA-aware).
    PerHead {
        flat: &'a [f32],
        head: &'a [f32],
        n_kv_heads: usize,
    },
}

/// Orchestrates KV cache management based on memory pressure and policy decisions.
///
/// Internally, CacheManager always operates through a `CachePressurePipeline`.
/// When created with `new()` (legacy API), the `EvictionPolicy` is wrapped in
/// an `EvictionHandler` adapter automatically. This eliminates routing duplication
/// while preserving full backward compatibility.
///
/// CacheManager follows the Dependency Inversion principle:
/// - Depends on `dyn EvictionPolicy` / `CachePressureHandler` (abstractions)
/// - Depends on `dyn SystemMonitor` (abstraction), not OS-specific implementations
pub struct CacheManager {
    pipeline: CachePressurePipeline,
    monitor: Box<dyn SystemMonitor>,
    /// Eviction triggers when available memory drops below this threshold (bytes).
    threshold_bytes: usize,
    /// Event sink for observability. Defaults to `NoOpSink` (zero overhead).
    event_sink: Arc<dyn EventSink>,
    /// Named eviction policies for Manager-directed dispatch (resilience).
    policies: HashMap<EvictMethod, Box<dyn EvictionPolicy>>,
}

impl CacheManager {
    /// Create a CacheManager in legacy mode (single eviction policy).
    ///
    /// The policy is wrapped in an `EvictionHandler` and placed in a single-stage
    /// pipeline at `Warning` level, preserving the original behavior.
    pub fn new(
        policy: Box<dyn EvictionPolicy>,
        monitor: Box<dyn SystemMonitor>,
        threshold_bytes: usize,
        target_ratio: f32,
    ) -> Self {
        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(EvictionHandler::new(policy, target_ratio)),
        }]);
        Self {
            pipeline,
            monitor,
            threshold_bytes,
            event_sink: Arc::new(NoOpSink),
            policies: HashMap::new(),
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
        Self {
            pipeline,
            monitor,
            threshold_bytes,
            event_sink: Arc::new(NoOpSink),
            policies: HashMap::new(),
        }
    }

    /// Set the event sink for observability.
    pub fn set_event_sink(&mut self, sink: Arc<dyn EventSink>) {
        self.event_sink = sink;
    }

    /// Get a reference to the event sink.
    pub fn event_sink(&self) -> &Arc<dyn EventSink> {
        &self.event_sink
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
        let mut last_new_pos = max_cache_pos(caches);

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

    /// Unified dispatch: query memory, determine pressure, build context, execute pipeline.
    ///
    /// When `force` is true, bypasses memory checks and runs at `Emergency` level.
    fn execute_dispatch(
        &self,
        caches: &mut [KVCache],
        scores: ScoreContext,
        force: bool,
        force_target_ratio: Option<f32>,
    ) -> Result<EvictionResult> {
        if caches.is_empty() {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: 0,
            });
        }

        let (pressure, mem_available) = if force {
            (PressureLevel::Emergency, 0)
        } else {
            let mem_available = match self.monitor.mem_stats() {
                Ok(stats) => stats.available,
                Err(e) => {
                    log::warn!("Failed to read memory stats: {}, skipping eviction", e);
                    return Ok(EvictionResult {
                        evicted: false,
                        tokens_removed: 0,
                        new_pos: max_cache_pos(caches),
                    });
                }
            };
            let pressure = self.determine_pressure_level(mem_available);
            if pressure == PressureLevel::Normal {
                return Ok(EvictionResult {
                    evicted: false,
                    tokens_removed: 0,
                    new_pos: max_cache_pos(caches),
                });
            }
            (pressure, mem_available)
        };

        self.event_sink.emit(CacheEvent::PressureDetected {
            level: pressure,
            mem_available,
            forced: force,
        });

        log::info!(
            "[CacheManager] pressure={:?}{}, executing '{}'",
            pressure,
            if force { " (forced)" } else { "" },
            self.pipeline.name(),
        );

        let (importance, head_importance, n_kv_heads) = match scores {
            ScoreContext::None => (None, None, 0),
            ScoreContext::Flat { importance } => (Some(importance), None, 0),
            ScoreContext::PerHead {
                flat,
                head,
                n_kv_heads,
            } => (Some(flat), Some(head), n_kv_heads),
        };

        let mut ctx = HandlerContext {
            caches,
            importance,
            head_importance,
            n_kv_heads,
            pressure_level: pressure,
            mem_available,
            target_ratio: force_target_ratio,
            qcf_sink: None,
            layer_ratios: None,
        };
        let results = self.pipeline.execute(&mut ctx)?;
        let eviction_result = Self::pipeline_results_to_eviction_result(&results, ctx.caches);

        if eviction_result.evicted {
            // Release physical pages for unused KV buffer regions (madvise MADV_DONTNEED)
            let mut bytes_released = 0usize;
            for cache in ctx.caches.iter_mut() {
                bytes_released += cache.release_unused_pages();
            }
            self.event_sink.emit(CacheEvent::EvictionCompleted {
                policy: self.pipeline.name(),
                tokens_removed: eviction_result.tokens_removed,
                new_pos: eviction_result.new_pos,
            });
            if bytes_released > 0 {
                log::info!(
                    "[CacheManager] released {} MB of physical pages after eviction",
                    bytes_released / (1024 * 1024),
                );
            }
        }

        Ok(eviction_result)
    }

    // ── Public API (all signatures preserved) ───────────────────────

    /// Check memory pressure and evict from all caches if needed.
    ///
    /// Called after each generation step in the inference loop.
    pub fn maybe_evict(&self, caches: &mut [KVCache]) -> Result<EvictionResult> {
        self.execute_dispatch(caches, ScoreContext::None, false, None)
    }

    /// Check memory pressure and evict using importance scores.
    ///
    /// Same logic as `maybe_evict()`, but passes importance scores to the handler.
    /// Used when `AttentionScoreAccumulator` is active.
    pub fn maybe_evict_with_scores(
        &self,
        caches: &mut [KVCache],
        importance: &[f32],
    ) -> Result<EvictionResult> {
        self.execute_dispatch(caches, ScoreContext::Flat { importance }, false, None)
    }

    /// Check memory pressure and evict using per-KV-head importance scores.
    ///
    /// GQA-aware version of `maybe_evict_with_scores()`.
    pub fn maybe_evict_with_head_scores(
        &self,
        caches: &mut [KVCache],
        flat_importance: &[f32],
        head_importance: &[f32],
        n_kv_heads: usize,
    ) -> Result<EvictionResult> {
        self.execute_dispatch(
            caches,
            ScoreContext::PerHead {
                flat: flat_importance,
                head: head_importance,
                n_kv_heads,
            },
            false,
            None,
        )
    }

    /// Force eviction without scores, bypassing should_evict() and memory checks.
    ///
    /// Used when eviction is triggered externally (e.g., by resilience signals).
    /// Runs at `Emergency` pressure level.
    pub fn force_evict(&self, caches: &mut [KVCache], target_ratio: f32) -> Result<EvictionResult> {
        self.execute_dispatch(caches, ScoreContext::None, true, Some(target_ratio))
    }

    /// Force eviction with importance scores, bypassing should_evict() and memory checks.
    ///
    /// Used when eviction is triggered externally for score-aware policies like H2O.
    /// Runs at `Emergency` pressure level with scores.
    pub fn force_evict_with_scores(
        &self,
        caches: &mut [KVCache],
        target_ratio: f32,
        importance: &[f32],
    ) -> Result<EvictionResult> {
        self.execute_dispatch(
            caches,
            ScoreContext::Flat { importance },
            true,
            Some(target_ratio),
        )
    }

    /// Force eviction with importance scores and per-layer budget ratios.
    ///
    /// Used when D2O layer-level allocation is active. Passes `layer_ratios`
    /// into `HandlerContext` so the D2OHandler can apply per-layer targets.
    /// Runs at `Emergency` pressure level with scores.
    pub fn force_evict_with_scores_and_budgets(
        &self,
        caches: &mut [KVCache],
        target_ratio: f32,
        importance: &[f32],
        layer_ratios: &[(f32, f32)],
    ) -> Result<EvictionResult> {
        if caches.is_empty() {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: 0,
            });
        }

        let pressure = PressureLevel::Emergency;
        let mem_available = 0;

        self.event_sink.emit(CacheEvent::PressureDetected {
            level: pressure,
            mem_available,
            forced: true,
        });

        log::info!(
            "[CacheManager] pressure={:?} (forced+layer_ratios), executing '{}'",
            pressure,
            self.pipeline.name(),
        );

        let mut ctx = HandlerContext {
            caches,
            importance: Some(importance),
            head_importance: None,
            n_kv_heads: 0,
            pressure_level: pressure,
            mem_available,
            target_ratio: Some(target_ratio),
            qcf_sink: None,
            layer_ratios: Some(layer_ratios),
        };
        let results = self.pipeline.execute(&mut ctx)?;
        let eviction_result = Self::pipeline_results_to_eviction_result(&results, ctx.caches);

        if eviction_result.evicted {
            for cache in ctx.caches.iter_mut() {
                cache.release_unused_pages();
            }
            self.event_sink.emit(CacheEvent::EvictionCompleted {
                policy: self.pipeline.name(),
                tokens_removed: eviction_result.tokens_removed,
                new_pos: eviction_result.new_pos,
            });
        }

        Ok(eviction_result)
    }

    /// Force eviction with per-KV-head importance scores.
    ///
    /// Used when H2O+ (GQA-aware) policy needs per-head eviction.
    /// Runs at `Emergency` pressure level with head scores.
    pub fn force_evict_with_head_scores(
        &self,
        caches: &mut [KVCache],
        target_ratio: f32,
        flat_importance: &[f32],
        head_importance: &[f32],
        n_kv_heads: usize,
    ) -> Result<EvictionResult> {
        self.execute_dispatch(
            caches,
            ScoreContext::PerHead {
                flat: flat_importance,
                head: head_importance,
                n_kv_heads,
            },
            true,
            Some(target_ratio),
        )
    }

    /// Returns the name of the active policy or pipeline.
    pub fn policy_name(&self) -> String {
        self.pipeline.name()
    }

    // ── Named policy registry (Manager-directed dispatch) ──────────

    /// Register an eviction policy for Manager-directed dispatch.
    pub fn register_policy(&mut self, method: EvictMethod, policy: Box<dyn EvictionPolicy>) {
        self.policies.insert(method, policy);
    }

    /// Force eviction using a specific named policy (for resilience directives).
    ///
    /// Bypasses the default pipeline and directly invokes the registered policy.
    /// Events are emitted through the same event_sink as auto-eviction.
    pub fn force_evict_by_policy(
        &self,
        method: EvictMethod,
        caches: &mut [KVCache],
        target_ratio: f32,
        scores: ScoreContext,
    ) -> Result<EvictionResult> {
        let policy = self
            .policies
            .get(&method)
            .ok_or_else(|| anyhow::anyhow!("no registered policy for {:?}", method))?;

        let result = Self::run_policy_eviction(policy.as_ref(), caches, target_ratio, scores)?;

        if result.evicted {
            for cache in caches.iter_mut() {
                cache.release_unused_pages();
            }
            self.event_sink.emit(CacheEvent::EvictionCompleted {
                policy: policy.name().to_string(),
                tokens_removed: result.tokens_removed,
                new_pos: result.new_pos,
            });
        }

        Ok(result)
    }

    /// Force eviction using a caller-provided policy reference.
    ///
    /// Like `force_evict_by_policy()` but takes a `&dyn EvictionPolicy` directly
    /// instead of looking up a registered policy by method. Useful for policies
    /// whose parameters are determined at runtime (e.g. StreamingLLM).
    pub fn force_evict_by_policy_ref(
        &self,
        policy: &dyn EvictionPolicy,
        caches: &mut [KVCache],
        target_ratio: f32,
        scores: ScoreContext,
    ) -> Result<EvictionResult> {
        let result = Self::run_policy_eviction(policy, caches, target_ratio, scores)?;

        if result.evicted {
            for cache in caches.iter_mut() {
                cache.release_unused_pages();
            }
            self.event_sink.emit(CacheEvent::EvictionCompleted {
                policy: policy.name().to_string(),
                tokens_removed: result.tokens_removed,
                new_pos: result.new_pos,
            });
        }

        Ok(result)
    }

    /// Shared eviction logic: compute target_len, dispatch to policy methods.
    /// Used by both `force_evict_by_policy()` and can be reused by EvictionHandler.
    fn run_policy_eviction(
        policy: &dyn EvictionPolicy,
        caches: &mut [KVCache],
        target_ratio: f32,
        scores: ScoreContext,
    ) -> Result<EvictionResult> {
        if caches.is_empty() {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: 0,
            });
        }

        let current_pos = max_cache_pos(caches);
        // target_ratio=0.0 means "let the policy decide" (e.g. StreamingLLM uses
        // its own sink_size + window_size). Pass target_len=0 so the policy's
        // default keep_size is used instead of forcing target_len=1.
        let target_len = if target_ratio <= 0.0 {
            0
        } else {
            ((current_pos as f32) * target_ratio).max(1.0) as usize
        };
        if target_len > 0 && current_pos <= target_len {
            return Ok(EvictionResult {
                evicted: false,
                tokens_removed: 0,
                new_pos: current_pos,
            });
        }

        log::debug!(
            "[CacheManager] policy='{}': {} → {} tokens",
            policy.name(),
            current_pos,
            target_len,
        );

        let (importance, head_importance, n_kv_heads) = match &scores {
            ScoreContext::None => (None, None, 0),
            ScoreContext::Flat { importance } => (Some(*importance), None, 0),
            ScoreContext::PerHead {
                flat,
                head,
                n_kv_heads,
            } => (Some(*flat), Some(*head), *n_kv_heads),
        };

        for cache in caches.iter_mut() {
            if let (Some(flat), Some(head_imp)) = (importance, head_importance) {
                if n_kv_heads > 0 {
                    policy.evict_with_head_scores(cache, target_len, flat, head_imp, n_kv_heads)?;
                } else {
                    policy.evict_with_scores(cache, target_len, flat)?;
                }
            } else if let Some(imp) = importance {
                policy.evict_with_scores(cache, target_len, imp)?;
            } else {
                policy.evict(cache, target_len)?;
            }
        }

        let new_pos = max_cache_pos(caches);
        let tokens_removed = current_pos - new_pos;
        let expected_removed = current_pos - target_len;

        // Warn when eviction achieved significantly less than requested.
        // This catches silent clamping by policies (e.g. protected_prefix > target_len).
        if expected_removed > 0 && tokens_removed < expected_removed / 2 {
            log::warn!(
                "[CacheManager] policy='{}': eviction undershot — removed {} tokens but target was {} ({}% of request). \
                 Check protected_prefix or policy constraints.",
                policy.name(),
                tokens_removed,
                expected_removed,
                tokens_removed * 100 / expected_removed,
            );
        }

        Ok(EvictionResult {
            evicted: true,
            tokens_removed,
            new_pos,
        })
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
        // Legacy mode wraps policy in EvictionHandler at Warning level
        assert!(cm.policy_name().contains("sliding_window"));
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

    // ── Resilience eviction integration tests ──
    // These test the END-TO-END eviction result (cache_pos after eviction),
    // not just plan structure. They catch bugs where eviction appears to succeed
    // but the cache size doesn't actually decrease.

    #[test]
    fn test_resilience_sliding_eviction_reduces_cache_pos() {
        // Simulate Manager-directed sliding eviction with small protected_prefix.
        // Keep ratio = 0.5: should reduce 80 tokens to ~40.
        let policy = SlidingWindowPolicy::new(50, 4); // protected_prefix=4, NOT prompt length
        let mut cm = CacheManager::new(
            Box::new(NoEvictionPolicy::new()),
            Box::new(MockMonitor { available: usize::MAX }),
            0,
            1.0,
        );
        cm.register_policy(
            crate::resilience::EvictMethod::Sliding,
            Box::new(policy),
        );

        let mut caches = make_caches(4, 80);
        let result = cm
            .force_evict_by_policy(
                crate::resilience::EvictMethod::Sliding,
                &mut caches,
                0.5,
                ScoreContext::None,
            )
            .unwrap();

        assert!(result.evicted);
        assert_eq!(result.new_pos, 40, "cache should be halved to 40 tokens");
        assert_eq!(result.tokens_removed, 40);
    }

    #[test]
    fn test_resilience_sliding_large_protected_prefix_limits_eviction() {
        // If protected_prefix is too large (e.g. entire prompt), eviction is limited.
        // This documents the behavior the undershoot warning catches.
        let policy = SlidingWindowPolicy::new(50, 70); // protected_prefix=70 out of 80 tokens
        let mut cm = CacheManager::new(
            Box::new(NoEvictionPolicy::new()),
            Box::new(MockMonitor { available: usize::MAX }),
            0,
            1.0,
        );
        cm.register_policy(
            crate::resilience::EvictMethod::Sliding,
            Box::new(policy),
        );

        let mut caches = make_caches(4, 80);
        let result = cm
            .force_evict_by_policy(
                crate::resilience::EvictMethod::Sliding,
                &mut caches,
                0.5,
                ScoreContext::None,
            )
            .unwrap();

        // target_len=40, but min_keep=(70+16).min(120)=86 > 80 → clamp to 80
        // current_pos(80) <= keep(80) → no meaningful eviction
        assert!(
            result.tokens_removed < 5,
            "large protected_prefix should severely limit eviction (removed {})",
            result.tokens_removed
        );
    }

    #[test]
    fn test_resilience_streaming_keeps_sink_plus_window() {
        use crate::core::eviction::StreamingLLMPolicy;

        // Streaming with sink=4, window=20 should keep exactly 24 tokens.
        let policy = StreamingLLMPolicy::new(4, 20);
        let cm = CacheManager::new(
            Box::new(NoEvictionPolicy::new()),
            Box::new(MockMonitor { available: usize::MAX }),
            0,
            1.0,
        );

        let mut caches = make_caches(4, 80);
        // target_ratio=0.0 means "let the policy decide"
        let result = cm
            .force_evict_by_policy_ref(&policy, &mut caches, 0.0, ScoreContext::None)
            .unwrap();

        assert!(result.evicted);
        assert_eq!(
            result.new_pos, 24,
            "streaming should keep sink(4) + window(20) = 24 tokens, got {}",
            result.new_pos
        );
    }

    #[test]
    fn test_resilience_h2o_eviction_respects_keep_ratio() {
        use crate::core::eviction::h2o::H2OPolicy;

        // H2O with protected_prefix=4, keep_ratio=0.5 on 80 tokens → ~40 tokens
        let policy = H2OPolicy::new(20, 0.5, 4);
        let mut cm = CacheManager::new(
            Box::new(NoEvictionPolicy::new()),
            Box::new(MockMonitor { available: usize::MAX }),
            0,
            1.0,
        );
        cm.register_policy(crate::resilience::EvictMethod::H2o, Box::new(policy));

        let mut caches = make_caches(4, 80);
        let result = cm
            .force_evict_by_policy(
                crate::resilience::EvictMethod::H2o,
                &mut caches,
                0.5,
                ScoreContext::None,
            )
            .unwrap();

        assert!(result.evicted);
        assert_eq!(
            result.new_pos, 40,
            "H2O should reduce to 40 tokens (ratio 0.5), got {}",
            result.new_pos
        );
    }
}
