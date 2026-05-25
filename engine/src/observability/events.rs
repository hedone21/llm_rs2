//! Structured event system for cache management observability.
//!
//! Provides a `CacheEvent` enum capturing key decisions (pressure detection,
//! eviction triggers, score snapshots) and an `EventSink` trait for consumers.
//! Default `NoOpSink` is zero-cost when observability is not needed.

// LAYER-EXEMPT: cross_cutting_trait_usage — §13.8-N §F enum-as-data identifier (V-12)
use crate::pressure::{ActionResult, PressureLevel};

// ── Event types ──────────────────────────────────────────────

/// Score distribution statistics.
#[derive(Debug, Clone)]
pub struct ScoreStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    /// Coefficient of variation (std / mean), 0.0 if mean is zero.
    pub cv: f32,
}

/// Score snapshot at eviction time.
#[derive(Debug, Clone)]
pub struct ScoreSnapshot {
    pub cache_pos: usize,
    pub protected_prefix: usize,
    pub decode_steps: usize,
    pub stats: ScoreStats,
    /// Top-K (position, score) pairs sorted by score descending.
    pub top_k: Vec<(usize, f32)>,
    /// Bottom-K (position, score) pairs sorted by score ascending.
    pub bottom_k: Vec<(usize, f32)>,
    /// Average score for prefix tokens.
    pub prefix_avg: f32,
    /// Average score for non-prefix tokens.
    pub rest_avg: f32,
    /// Fraction of non-prefix tokens above mean+1σ.
    pub above_1sigma_frac: f32,
    /// Fraction of non-prefix tokens above mean+2σ.
    pub above_2sigma_frac: f32,
}

/// Dispatcher identity for `WeightSwapEvent`.
///
/// CLI-flag-determined static choice — `--swap-incremental-per-tick K`,
/// `--swap-intra-forward`, `--swap-phase-aware` are mutually exclusive.
/// `Subsystem` is used for swap_executor's BgFetch/AsyncSwap helpers whose
/// caller dispatcher is indirect (S-1+α: `SwapExecutor` defaults to
/// `IntraForward` and PhaseAware/Subsystem callers override via
/// `.with_kind(...)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WeightSwapKind {
    Incremental,
    #[default]
    IntraForward,
    PhaseAware,
    Subsystem,
}

impl WeightSwapKind {
    /// Stable string label used in stderr emit output. Matches the legacy
    /// S-1 `&'static str` values so downstream grep patterns continue to work.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Incremental => "Incremental",
            Self::IntraForward => "IntraForward",
            Self::PhaseAware => "PhaseAware",
            Self::Subsystem => "Subsystem",
        }
    }
}

impl std::fmt::Display for WeightSwapKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Structured events emitted during weight swap operations.
///
/// 5 variants. Each carries `kind: WeightSwapKind` identifying which dispatcher
/// emitted the event (see `WeightSwapKind` docstring).
#[derive(Debug, Clone)]
pub enum WeightSwapEvent {
    /// Dispatcher committed a swap plan — layer/chunk partition decided.
    PlanCommitted {
        kind: WeightSwapKind,
        algorithm: &'static str,
        ratio: f32,
        k_chunk: usize,
        n_layers: usize,
    },
    /// One chunk drained (transfer + commit completed).
    ChunkDrained {
        kind: WeightSwapKind,
        chunk_idx: usize,
        layers_done: usize,
        latency_ms: f32,
        /// Optional stage breakdown string (e.g. "mmap:12.3/permute:8.1/...").
        stages: Option<String>,
    },
    /// Plan finished — incremental_plan=None or J4/J5 finalize ok.
    PlanRetired {
        kind: WeightSwapKind,
        qcf_actual: Option<f32>,
        token: usize,
        elapsed_ms: f32,
        /// Optional ratio / layer counts (J3 manager-plan path provides these,
        /// J4/J5 do not).
        ratio: Option<f32>,
        n_planned: Option<usize>,
        actually_q4: Option<usize>,
    },
    /// Swap failed mid-flight (build / dispatch / drain / finalize error).
    SwapFailed {
        kind: WeightSwapKind,
        reason: String,
        layer: Option<usize>,
        token: Option<usize>,
    },
    /// Batch-end diagnostic snapshot (env-gated by `LLMRS_SWAP_DRAIN_DIAG=1`).
    ///
    /// Reports worst observed queue depth during the batch. The user's hard
    /// constraint ("memory peak ≤ 1 layer") is satisfied iff both
    /// `max_release_pending` and `max_dispatcher_pending` ≤ 1.
    BatchSummary {
        kind: WeightSwapKind,
        /// "sync" or "async" — swap_executor execution path.
        mode: &'static str,
        target_layers: usize,
        max_release_pending: usize,
        max_dispatcher_pending: usize,
    },
}

/// Structured events emitted during cache management.
#[derive(Debug, Clone)]
pub enum CacheEvent {
    /// Memory pressure detected, about to execute pipeline.
    PressureDetected {
        level: PressureLevel,
        mem_available: usize,
        forced: bool,
    },
    /// An eviction (or other cache action) completed.
    EvictionCompleted {
        policy: String,
        tokens_removed: usize,
        new_pos: usize,
    },
    /// A pipeline stage produced a result.
    PipelineStageExecuted {
        handler: String,
        result: ActionResult,
    },
    /// Score distribution snapshot taken at eviction time.
    ScoreDiagnostic(ScoreSnapshot),
    /// Proxy metric computed during a lossy cache action.
    ProxyComputed(crate::qcf_types::QcfMetric),
    /// Weight swap lifecycle event (S-1).
    WeightSwap(WeightSwapEvent),
}

// ── Sink trait ────────────────────────────────────────────────

/// Consumer of `CacheEvent`s. Implement this to observe cache decisions.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: CacheEvent);
}

/// No-op sink — zero overhead when observability is disabled.
pub struct NoOpSink;

impl EventSink for NoOpSink {
    #[inline(always)]
    fn emit(&self, _event: CacheEvent) {}
}

/// S-1+α: factory returning a `NoOpSink` as `Arc<dyn EventSink>`.
///
/// Allows L3 (e.g. `models/weights/swap_executor.rs`) to initialise an event
/// sink without naming the concrete `NoOpSink` type — preserving the
/// L3-inference → cross-cutting concrete-import boundary (INV-LAYER-003).
pub fn noop_sink() -> std::sync::Arc<dyn EventSink> {
    std::sync::Arc::new(NoOpSink)
}

// ── Built-in sinks ───────────────────────────────────────────

/// Sink that logs events to stderr using the `[ScoreDiag]` prefix format.
/// Drop-in replacement for the inline diagnostic code in generate.rs.
pub struct StderrDiagnosticSink;

impl EventSink for StderrDiagnosticSink {
    fn emit(&self, event: CacheEvent) {
        match event {
            CacheEvent::ScoreDiagnostic(snap) => {
                eprintln!(
                    "[ScoreDiag] cache_pos={}, prefix={}, decode_steps={}",
                    snap.cache_pos, snap.protected_prefix, snap.decode_steps
                );
                eprintln!(
                    "[ScoreDiag] Score stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}, cv={:.4}",
                    snap.stats.min, snap.stats.max, snap.stats.mean, snap.stats.std, snap.stats.cv
                );

                let top_str: Vec<String> = snap
                    .top_k
                    .iter()
                    .map(|(pos, score)| format!("{}:{:.3}", pos, score))
                    .collect();
                let bot_str: Vec<String> = snap
                    .bottom_k
                    .iter()
                    .map(|(pos, score)| format!("{}:{:.3}", pos, score))
                    .collect();
                eprintln!("[ScoreDiag] Top-10: [{}]", top_str.join(", "));
                eprintln!("[ScoreDiag] Bot-10: [{}]", bot_str.join(", "));
                eprintln!(
                    "[ScoreDiag] Prefix avg={:.4}, Rest avg={:.4}, ratio={:.2}x",
                    snap.prefix_avg,
                    snap.rest_avg,
                    if snap.rest_avg > 0.0 {
                        snap.prefix_avg / snap.rest_avg
                    } else {
                        0.0
                    }
                );
                let n_non_prefix = snap.cache_pos.saturating_sub(snap.protected_prefix);
                if n_non_prefix > 0 {
                    let above_1s = (snap.above_1sigma_frac * n_non_prefix as f32) as usize;
                    let above_2s = (snap.above_2sigma_frac * n_non_prefix as f32) as usize;
                    eprintln!(
                        "[ScoreDiag] Non-prefix: n={}, >mean+1σ={}({:.1}%), >mean+2σ={}({:.1}%)",
                        n_non_prefix,
                        above_1s,
                        snap.above_1sigma_frac * 100.0,
                        above_2s,
                        snap.above_2sigma_frac * 100.0,
                    );
                }
            }
            CacheEvent::PressureDetected {
                level,
                mem_available,
                forced,
            } => {
                if forced {
                    // Budget-driven forced eviction (eval-ll, resilience signal).
                    // Don't display pressure level — it's always Emergency by design
                    // to ensure all pipeline handlers run, not because of actual
                    // memory pressure.
                    eprintln!("[CacheEvent] Budget eviction (forced)");
                } else {
                    eprintln!(
                        "[CacheEvent] Pressure {:?}, mem_available={} MB",
                        level,
                        mem_available / (1024 * 1024),
                    );
                }
            }
            CacheEvent::EvictionCompleted {
                ref policy,
                tokens_removed,
                new_pos,
            } => {
                eprintln!(
                    "[CacheEvent] Eviction completed: policy='{}', removed={}, new_pos={}",
                    policy, tokens_removed, new_pos
                );
            }
            CacheEvent::PipelineStageExecuted {
                ref handler,
                ref result,
            } => {
                eprintln!("[CacheEvent] Stage '{}' → {:?}", handler, result);
            }
            CacheEvent::ProxyComputed(ref metric) => {
                eprintln!(
                    "[ProxyDeg] action='{}', proxy={:.6}, tokens_affected={}{}",
                    metric.action,
                    metric.raw_value,
                    metric.tokens_affected,
                    if let Some(ref ph) = metric.per_head {
                        format!(
                            ", per_head=[{}]",
                            ph.iter()
                                .map(|v| format!("{:.4}", v))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    } else {
                        String::new()
                    }
                );
            }
            CacheEvent::WeightSwap(WeightSwapEvent::PlanCommitted {
                kind,
                algorithm,
                ratio,
                k_chunk,
                n_layers,
            }) => {
                eprintln!(
                    "[WeightSwap] PlanCommitted: kind={}, algo={}, ratio={:.2}, k={}, n_layers={}",
                    kind, algorithm, ratio, k_chunk, n_layers
                );
            }
            CacheEvent::WeightSwap(WeightSwapEvent::ChunkDrained {
                kind,
                chunk_idx,
                layers_done,
                latency_ms,
                ref stages,
            }) => {
                if let Some(s) = stages {
                    eprintln!(
                        "[WeightSwap] ChunkDrained: kind={}, idx={}, layers={}, latency={:.1}ms, stages={}",
                        kind, chunk_idx, layers_done, latency_ms, s
                    );
                } else {
                    eprintln!(
                        "[WeightSwap] ChunkDrained: kind={}, idx={}, layers={}, latency={:.1}ms",
                        kind, chunk_idx, layers_done, latency_ms
                    );
                }
            }
            CacheEvent::WeightSwap(WeightSwapEvent::PlanRetired {
                kind,
                qcf_actual,
                token,
                elapsed_ms,
                ratio,
                n_planned,
                actually_q4,
            }) => {
                let qcf_str = qcf_actual
                    .map(|v| format!("{:.4}", v))
                    .unwrap_or_else(|| "n/a".to_string());
                let ratio_str = ratio
                    .map(|v| format!(", ratio={:.2}", v))
                    .unwrap_or_default();
                let planned_str = n_planned
                    .map(|v| format!(", planned={}", v))
                    .unwrap_or_default();
                let actual_str = actually_q4
                    .map(|v| format!(", actually_q4={}", v))
                    .unwrap_or_default();
                eprintln!(
                    "[WeightSwap] PlanRetired: kind={}, qcf={}, token={}, elapsed={:.1}ms{}{}{}",
                    kind, qcf_str, token, elapsed_ms, ratio_str, planned_str, actual_str
                );
            }
            CacheEvent::WeightSwap(WeightSwapEvent::SwapFailed {
                kind,
                ref reason,
                layer,
                token,
            }) => {
                let layer_str = layer.map(|v| format!(", layer={}", v)).unwrap_or_default();
                let token_str = token.map(|v| format!(", token={}", v)).unwrap_or_default();
                eprintln!(
                    "[WeightSwap] SwapFailed: kind={}, reason=\"{}\"{}{}",
                    kind, reason, layer_str, token_str
                );
            }
            CacheEvent::WeightSwap(WeightSwapEvent::BatchSummary {
                kind,
                mode,
                target_layers,
                max_release_pending,
                max_dispatcher_pending,
            }) => {
                eprintln!(
                    "[WeightSwap] BatchSummary: kind={}, mode={}, target_layers={}, max_release_pending={}, max_dispatcher_pending={}",
                    kind, mode, target_layers, max_release_pending, max_dispatcher_pending
                );
            }
        }
    }
}

/// Collecting sink for testing — stores all events in a thread-safe Vec.
#[cfg(test)]
pub struct CollectingSink {
    events: std::sync::Mutex<Vec<CacheEvent>>,
}

#[cfg(test)]
impl Default for CollectingSink {
    fn default() -> Self {
        Self {
            events: std::sync::Mutex::new(Vec::new()),
        }
    }
}

#[cfg(test)]
impl CollectingSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self) -> Vec<CacheEvent> {
        self.events.lock().unwrap().clone()
    }

    /// Take all collected events, leaving the internal buffer empty.
    /// Avoids the Vec clone cost of `events()` when the caller will not need
    /// the sink to retain state (e.g. one-shot assertions).
    pub fn drain_events(&self) -> Vec<CacheEvent> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }
}

#[cfg(test)]
impl EventSink for CollectingSink {
    fn emit(&self, event: CacheEvent) {
        self.events.lock().unwrap().push(event);
    }
}

// ── Score snapshot builder ───────────────────────────────────

/// Build a `ScoreSnapshot` from raw importance scores.
///
/// Extracts statistics, top-K/bottom-K positions, and distribution metrics.
/// This is the library equivalent of the inline diagnostic code that was
/// previously in generate.rs (lines 782-911).
pub fn build_score_snapshot(
    scores: &[f32],
    cache_pos: usize,
    protected_prefix: usize,
    decode_steps: usize,
    k: usize,
) -> Option<ScoreSnapshot> {
    if cache_pos == 0 || scores.len() < cache_pos {
        return None;
    }

    let active = &scores[..cache_pos];

    // Overall stats
    let min = active.iter().copied().fold(f32::INFINITY, f32::min);
    let max = active.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = active.iter().sum::<f32>() / active.len() as f32;
    let var = active.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / active.len() as f32;
    let std = var.sqrt();
    let cv = if mean > 0.0 { std / mean } else { 0.0 };

    // Top-K / Bottom-K
    let mut indexed: Vec<(usize, f32)> = active.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_k: Vec<(usize, f32)> = indexed.iter().take(k).copied().collect();
    let bottom_k: Vec<(usize, f32)> = indexed.iter().rev().take(k).copied().collect();

    // Prefix vs rest averages
    let prefix_end = protected_prefix.min(cache_pos);
    let prefix_avg = if prefix_end > 0 {
        scores[..prefix_end].iter().sum::<f32>() / prefix_end as f32
    } else {
        0.0
    };
    let rest_avg = if cache_pos > protected_prefix {
        scores[protected_prefix..cache_pos].iter().sum::<f32>()
            / (cache_pos - protected_prefix) as f32
    } else {
        0.0
    };

    // σ-distribution for non-prefix tokens
    let (above_1sigma_frac, above_2sigma_frac) = if cache_pos > protected_prefix {
        let non_prefix = &scores[protected_prefix..cache_pos];
        let np_mean = non_prefix.iter().sum::<f32>() / non_prefix.len() as f32;
        let np_var = non_prefix
            .iter()
            .map(|s| (s - np_mean).powi(2))
            .sum::<f32>()
            / non_prefix.len() as f32;
        let np_std = np_var.sqrt();
        let above_1 = non_prefix.iter().filter(|&&s| s > np_mean + np_std).count();
        let above_2 = non_prefix
            .iter()
            .filter(|&&s| s > np_mean + 2.0 * np_std)
            .count();
        (
            above_1 as f32 / non_prefix.len() as f32,
            above_2 as f32 / non_prefix.len() as f32,
        )
    } else {
        (0.0, 0.0)
    };

    Some(ScoreSnapshot {
        cache_pos,
        protected_prefix,
        decode_steps,
        stats: ScoreStats {
            min,
            max,
            mean,
            std,
            cv,
        },
        top_k,
        bottom_k,
        prefix_avg,
        rest_avg,
        above_1sigma_frac,
        above_2sigma_frac,
    })
}

// ── CSV dump utility ─────────────────────────────────────────

/// Dump scores to a CSV file (position,score format).
pub fn dump_scores_csv(scores: &[f32], cache_pos: usize, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "position,score")?;
    for (i, &s) in scores[..cache_pos.min(scores.len())].iter().enumerate() {
        writeln!(f, "{},{:.6}", i, s)?;
    }
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_sink_is_zero_cost() {
        let sink = NoOpSink;
        // Should compile and do nothing
        sink.emit(CacheEvent::PressureDetected {
            level: PressureLevel::Warning,
            mem_available: 1024,
            forced: false,
        });
    }

    #[test]
    fn test_collecting_sink_captures_events() {
        let sink = CollectingSink::new();
        sink.emit(CacheEvent::PressureDetected {
            level: PressureLevel::Critical,
            mem_available: 512,
            forced: true,
        });
        sink.emit(CacheEvent::EvictionCompleted {
            policy: "h2o".to_string(),
            tokens_removed: 10,
            new_pos: 30,
        });

        let events = sink.events();
        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0],
            CacheEvent::PressureDetected {
                level: PressureLevel::Critical,
                ..
            }
        ));
        assert!(matches!(
            events[1],
            CacheEvent::EvictionCompleted {
                tokens_removed: 10,
                ..
            }
        ));
    }

    #[test]
    fn test_collecting_sink_drain_consumes_buffer() {
        let sink = CollectingSink::new();
        sink.emit(CacheEvent::EvictionCompleted {
            policy: "h2o".to_string(),
            tokens_removed: 5,
            new_pos: 10,
        });

        let drained = sink.drain_events();
        assert_eq!(drained.len(), 1);

        // After drain, the sink is empty.
        assert!(sink.events().is_empty());
        assert!(sink.drain_events().is_empty());
    }

    #[test]
    fn test_build_score_snapshot_basic() {
        let scores = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let snap = build_score_snapshot(&scores, 5, 1, 4, 3).unwrap();

        assert_eq!(snap.cache_pos, 5);
        assert_eq!(snap.protected_prefix, 1);
        assert_eq!(snap.decode_steps, 4);
        assert_eq!(snap.stats.min, 1.0);
        assert_eq!(snap.stats.max, 5.0);
        assert_eq!(snap.stats.mean, 3.0);
        assert_eq!(snap.top_k.len(), 3);
        assert_eq!(snap.top_k[0], (1, 5.0)); // position 1 has score 5.0
        assert_eq!(snap.bottom_k.len(), 3);
        assert_eq!(snap.bottom_k[0], (0, 1.0)); // position 0 has score 1.0
    }

    #[test]
    fn test_build_score_snapshot_empty_returns_none() {
        assert!(build_score_snapshot(&[], 0, 0, 0, 10).is_none());
        assert!(build_score_snapshot(&[1.0], 0, 0, 0, 10).is_none());
    }

    #[test]
    fn test_build_score_snapshot_all_prefix() {
        let scores = vec![1.0, 2.0, 3.0];
        let snap = build_score_snapshot(&scores, 3, 3, 0, 5).unwrap();
        // All tokens are prefix, rest_avg should be 0
        assert_eq!(snap.rest_avg, 0.0);
        assert_eq!(snap.above_1sigma_frac, 0.0);
    }

    #[test]
    fn test_noop_sink_accepts_weight_swap() {
        let sink = NoOpSink;
        sink.emit(CacheEvent::WeightSwap(WeightSwapEvent::PlanCommitted {
            kind: WeightSwapKind::IntraForward,
            algorithm: "ImportanceAware",
            ratio: 0.5,
            k_chunk: 8,
            n_layers: 14,
        }));
    }

    #[test]
    fn test_collecting_sink_captures_weight_swap_kinds() {
        let sink = CollectingSink::new();
        sink.emit(CacheEvent::WeightSwap(WeightSwapEvent::PlanCommitted {
            kind: WeightSwapKind::Incremental,
            algorithm: "Sequential",
            ratio: 0.3,
            k_chunk: 4,
            n_layers: 10,
        }));
        sink.emit(CacheEvent::WeightSwap(WeightSwapEvent::ChunkDrained {
            kind: WeightSwapKind::IntraForward,
            chunk_idx: 0,
            layers_done: 8,
            latency_ms: 124.5,
            stages: None,
        }));
        sink.emit(CacheEvent::WeightSwap(WeightSwapEvent::PlanRetired {
            kind: WeightSwapKind::PhaseAware,
            qcf_actual: Some(0.083),
            token: 42,
            elapsed_ms: 890.0,
            ratio: None,
            n_planned: None,
            actually_q4: None,
        }));
        sink.emit(CacheEvent::WeightSwap(WeightSwapEvent::SwapFailed {
            kind: WeightSwapKind::Subsystem,
            reason: "mmap fault".to_string(),
            layer: Some(3),
            token: None,
        }));

        let events = sink.events();
        assert_eq!(events.len(), 4);
        // Verify kind is preserved for each dispatcher source.
        let kinds: Vec<WeightSwapKind> = events
            .iter()
            .filter_map(|e| match e {
                CacheEvent::WeightSwap(WeightSwapEvent::PlanCommitted { kind, .. })
                | CacheEvent::WeightSwap(WeightSwapEvent::ChunkDrained { kind, .. })
                | CacheEvent::WeightSwap(WeightSwapEvent::PlanRetired { kind, .. })
                | CacheEvent::WeightSwap(WeightSwapEvent::SwapFailed { kind, .. })
                | CacheEvent::WeightSwap(WeightSwapEvent::BatchSummary { kind, .. }) => Some(*kind),
                _ => None,
            })
            .collect();
        assert_eq!(
            kinds,
            vec![
                WeightSwapKind::Incremental,
                WeightSwapKind::IntraForward,
                WeightSwapKind::PhaseAware,
                WeightSwapKind::Subsystem,
            ]
        );
    }

    #[test]
    fn test_weight_swap_event_clone_and_debug() {
        // Sanity: derived Debug + Clone work for downstream observers / test fixtures.
        let evt = WeightSwapEvent::ChunkDrained {
            kind: WeightSwapKind::Incremental,
            chunk_idx: 3,
            layers_done: 14,
            latency_ms: 98.3,
            stages: Some("mmap:12.3/permute:8.1".to_string()),
        };
        let cloned = evt.clone();
        assert!(format!("{:?}", cloned).contains("ChunkDrained"));
        assert!(format!("{:?}", cloned).contains("mmap:12.3/permute:8.1"));
    }

    #[test]
    fn test_weight_swap_kind_default_is_intra_forward() {
        // Q1 결정: SwapExecutor 등이 builder default로 IntraForward를 사용한다.
        assert_eq!(WeightSwapKind::default(), WeightSwapKind::IntraForward);
        // Display label은 S-1 legacy &'static str 값과 동일 (grep 호환).
        assert_eq!(WeightSwapKind::Incremental.as_str(), "Incremental");
        assert_eq!(WeightSwapKind::IntraForward.as_str(), "IntraForward");
        assert_eq!(WeightSwapKind::PhaseAware.as_str(), "PhaseAware");
        assert_eq!(WeightSwapKind::Subsystem.as_str(), "Subsystem");
    }

    #[test]
    fn test_batch_summary_emit_and_collect() {
        // BatchSummary는 swap_executor:1285의 [SwapPeak] diag summary를 EventSink로
        // 라우팅한다. 운영 제약 "메모리 peak ≤ 1 layer" 검증선.
        let sink = CollectingSink::new();
        sink.emit(CacheEvent::WeightSwap(WeightSwapEvent::BatchSummary {
            kind: WeightSwapKind::Subsystem,
            mode: "async",
            target_layers: 14,
            max_release_pending: 1,
            max_dispatcher_pending: 1,
        }));
        let events = sink.events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            CacheEvent::WeightSwap(WeightSwapEvent::BatchSummary {
                kind,
                mode,
                target_layers,
                max_release_pending,
                max_dispatcher_pending,
            }) => {
                assert_eq!(*kind, WeightSwapKind::Subsystem);
                assert_eq!(*mode, "async");
                assert_eq!(*target_layers, 14);
                assert_eq!(*max_release_pending, 1);
                assert_eq!(*max_dispatcher_pending, 1);
            }
            other => panic!("expected BatchSummary, got {other:?}"),
        }
    }

    #[test]
    fn test_build_score_snapshot_sigma_distribution() {
        // 100 tokens: 90 with score=1.0, 10 with score=100.0
        let mut scores = vec![1.0f32; 100];
        scores[..10].fill(100.0);
        let snap = build_score_snapshot(&scores, 100, 0, 50, 5).unwrap();
        // The 10 high-score tokens should be well above mean+1σ
        assert!(snap.above_1sigma_frac > 0.0);
        assert!(snap.above_1sigma_frac < 0.2); // ~10% of tokens
    }
}
