//! Adaptive multi-layer prefetch controller.
//!
//! Dynamically adjusts prefetch depth based on observed preload vs forward timing.
//! When preload takes longer than forward (pipeline stall), depth increases.
//! When there is consistent slack, depth decreases (saves memory).

use std::time::Duration;

/// Adaptive prefetch depth controller for offload KV cache pipelines.
///
/// Tracks EMA of preload and forward times, adjusting depth at token boundaries.
/// - Increase: immediate on stall (preload > forward)
/// - Decrease: after `decrease_patience` consecutive slack observations (anti-oscillation)
pub struct PrefetchController {
    /// Current prefetch depth (≥ 1).
    depth: usize,
    /// Maximum allowed depth (memory cap).
    max_depth: usize,
    /// EMA of preload duration (μs).
    preload_ema_us: f64,
    /// EMA of forward duration (μs).
    forward_ema_us: f64,
    /// EMA smoothing coefficient.
    alpha: f64,
    /// Number of samples recorded so far.
    samples: usize,
    /// Warmup period: no adjustments until this many samples.
    warmup_samples: usize,
    /// Consecutive observations of sufficient slack.
    slack_streak: usize,
    /// Number of consecutive slack observations required before decreasing depth.
    decrease_patience: usize,
}

/// Default starting prefetch depth used by [`PrefetchController::new`].
///
/// Chosen to match typical on-device transformer layer counts (Llama 3.2 1B
/// = 16 layers), so the pipeline starts fully primed and the adaptive loop
/// only has to shrink if memory pressure or slack is observed.
pub const DEFAULT_INITIAL_DEPTH: usize = 16;

impl PrefetchController {
    /// Create a new controller with [`DEFAULT_INITIAL_DEPTH`] as the starting
    /// depth (clamped to `max_depth`).
    ///
    /// - `max_depth`: upper bound on prefetch depth (memory limit)
    /// - `num_layers`: total transformer layers (used as warmup period)
    pub fn new(max_depth: usize, num_layers: usize) -> Self {
        Self::with_initial_depth(max_depth, DEFAULT_INITIAL_DEPTH, num_layers)
    }

    /// Create a new controller with an explicit starting depth.
    ///
    /// `initial_depth` is clamped to the range `[1, max_depth]`. Useful for
    /// tests that want deterministic starting conditions or for callers that
    /// want to start conservatively regardless of the global default.
    pub fn with_initial_depth(max_depth: usize, initial_depth: usize, num_layers: usize) -> Self {
        let max_depth = max_depth.max(1);
        Self {
            depth: initial_depth.clamp(1, max_depth),
            max_depth,
            preload_ema_us: 0.0,
            forward_ema_us: 0.0,
            alpha: 0.3,
            samples: 0,
            warmup_samples: num_layers,
            slack_streak: 0,
            decrease_patience: 3,
        }
    }

    /// Current prefetch depth.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Maximum allowed prefetch depth.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// EMA of preload time in microseconds.
    pub fn preload_ema_us(&self) -> f64 {
        self.preload_ema_us
    }

    /// EMA of forward time in microseconds.
    pub fn forward_ema_us(&self) -> f64 {
        self.forward_ema_us
    }

    /// Record a (preload_duration, forward_duration) observation.
    pub fn record(&mut self, preload_dur: Duration, forward_dur: Duration) {
        let preload_us = preload_dur.as_secs_f64() * 1_000_000.0;
        let forward_us = forward_dur.as_secs_f64() * 1_000_000.0;

        if self.samples == 0 {
            self.preload_ema_us = preload_us;
            self.forward_ema_us = forward_us;
        } else {
            self.preload_ema_us =
                self.alpha * preload_us + (1.0 - self.alpha) * self.preload_ema_us;
            self.forward_ema_us =
                self.alpha * forward_us + (1.0 - self.alpha) * self.forward_ema_us;
        }
        self.samples += 1;
    }

    /// Record only a preload duration observation.
    pub fn record_preload(&mut self, preload_dur: Duration) {
        let preload_us = preload_dur.as_secs_f64() * 1_000_000.0;
        if self.samples == 0 {
            self.preload_ema_us = preload_us;
        } else {
            self.preload_ema_us =
                self.alpha * preload_us + (1.0 - self.alpha) * self.preload_ema_us;
        }
    }

    /// Record only a forward duration observation.
    pub fn record_forward(&mut self, forward_dur: Duration) {
        let forward_us = forward_dur.as_secs_f64() * 1_000_000.0;
        if self.samples == 0 {
            self.forward_ema_us = forward_us;
        } else {
            self.forward_ema_us =
                self.alpha * forward_us + (1.0 - self.alpha) * self.forward_ema_us;
        }
        self.samples += 1; // Count forward calls as samples (one per layer)
    }

    /// Adjust depth based on accumulated timing data.
    /// Call once per token boundary (after all layers complete).
    pub fn adjust(&mut self) {
        if self.samples < self.warmup_samples {
            return; // Not enough data yet
        }
        if self.forward_ema_us < 1.0 {
            return; // Avoid division by near-zero
        }

        let slack_ratio = 1.0 - (self.preload_ema_us / self.forward_ema_us);

        if slack_ratio < 0.0 {
            // Stall: preload takes longer than forward → increase immediately
            self.depth = (self.depth + 1).min(self.max_depth);
            self.slack_streak = 0;
        } else if slack_ratio > 0.3 {
            // Ample slack → count toward decrease
            self.slack_streak += 1;
            if self.slack_streak >= self.decrease_patience && self.depth > 1 {
                self.depth -= 1;
                self.slack_streak = 0;
            }
        } else {
            // Balanced — hold steady
            self.slack_streak = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dur_us(us: u64) -> Duration {
        Duration::from_micros(us)
    }

    #[test]
    fn test_warmup_no_adjust() {
        let num_layers = 4;
        let mut ctrl = PrefetchController::with_initial_depth(4, 1, num_layers);

        // Record fewer samples than warmup → depth stays at 1
        for _ in 0..3 {
            ctrl.record(dur_us(5000), dur_us(2000)); // stall condition
        }
        ctrl.adjust();
        assert_eq!(ctrl.depth(), 1, "depth must not change during warmup");
    }

    #[test]
    fn test_increase_on_stall() {
        let num_layers = 2;
        let mut ctrl = PrefetchController::with_initial_depth(4, 1, num_layers);
        assert_eq!(ctrl.depth(), 1);

        // Record enough samples to exit warmup (preload > forward = stall)
        for _ in 0..3 {
            ctrl.record(dur_us(5000), dur_us(2000));
        }
        ctrl.adjust();
        assert_eq!(ctrl.depth(), 2, "depth should increase on stall");

        // Another stall → increase again
        for _ in 0..2 {
            ctrl.record(dur_us(5000), dur_us(2000));
        }
        ctrl.adjust();
        assert_eq!(ctrl.depth(), 3);
    }

    #[test]
    fn test_decrease_with_patience() {
        let num_layers = 2;
        let mut ctrl = PrefetchController::with_initial_depth(4, 1, num_layers);

        // Drive depth up first
        for _ in 0..3 {
            ctrl.record(dur_us(5000), dur_us(2000));
        }
        ctrl.adjust();
        assert_eq!(ctrl.depth(), 2);

        // First: stabilize EMA into slack territory (need enough samples to
        // overcome the stall EMA from warmup phase)
        for _ in 0..20 {
            ctrl.record(dur_us(500), dur_us(3000)); // slack_ratio ≈ 0.83
        }
        // Reset streak by calling adjust during a stall-ema → may increase depth
        // Let's just settle the EMA first, then count patience from scratch
        ctrl.adjust(); // first slack adjust → streak=1

        let depth_after_settle = ctrl.depth();

        // Now count patience: need 2 more consecutive slack adjusts
        for _ in 0..num_layers {
            ctrl.record(dur_us(500), dur_us(3000));
        }
        ctrl.adjust(); // streak=2
        assert_eq!(
            ctrl.depth(),
            depth_after_settle,
            "should not decrease before patience met"
        );

        for _ in 0..num_layers {
            ctrl.record(dur_us(500), dur_us(3000));
        }
        ctrl.adjust(); // streak=3 → decrease
        assert_eq!(
            ctrl.depth(),
            depth_after_settle - 1,
            "depth should decrease after patience met"
        );
    }

    #[test]
    fn test_max_depth_cap() {
        let num_layers = 2;
        let mut ctrl = PrefetchController::with_initial_depth(3, 1, num_layers);

        // Stall many times
        for _ in 0..20 {
            for _ in 0..num_layers {
                ctrl.record(dur_us(5000), dur_us(2000));
            }
            ctrl.adjust();
        }
        assert_eq!(ctrl.depth(), 3, "depth must not exceed max_depth");
    }

    #[test]
    fn test_no_oscillation() {
        let num_layers = 2;
        let mut ctrl = PrefetchController::with_initial_depth(4, 1, num_layers);

        // Warmup with stall → depth=2
        for _ in 0..3 {
            ctrl.record(dur_us(5000), dur_us(2000));
        }
        ctrl.adjust();
        assert_eq!(ctrl.depth(), 2);

        // Alternate: one slack round, one stall round → should not oscillate
        for _ in 0..5 {
            // Slack round
            for _ in 0..num_layers {
                ctrl.record(dur_us(500), dur_us(3000));
            }
            ctrl.adjust();

            // Stall round
            for _ in 0..num_layers {
                ctrl.record(dur_us(5000), dur_us(2000));
            }
            ctrl.adjust();
        }
        // Due to patience=3 on decrease, alternating prevents decrease,
        // but stalls keep pushing up → depth should be >= 2
        assert!(
            ctrl.depth() >= 2,
            "alternating should not cause depth to drop below 2"
        );
    }

    #[test]
    fn test_default_initial_depth_is_clamped() {
        // `new` starts at DEFAULT_INITIAL_DEPTH but is clamped down to
        // `max_depth` when the latter is smaller.
        let small = PrefetchController::new(4, 2);
        assert_eq!(small.depth(), 4, "small max_depth clamps initial depth");

        let large = PrefetchController::new(128, 2);
        assert_eq!(
            large.depth(),
            DEFAULT_INITIAL_DEPTH,
            "initial depth equals DEFAULT_INITIAL_DEPTH when max_depth allows it"
        );
    }

    #[test]
    fn test_with_initial_depth_clamps_to_range() {
        // `initial_depth=0` is promoted to 1, and an oversized initial_depth
        // is clamped to max_depth.
        let floor = PrefetchController::with_initial_depth(8, 0, 2);
        assert_eq!(floor.depth(), 1);

        let ceil = PrefetchController::with_initial_depth(8, 999, 2);
        assert_eq!(ceil.depth(), 8);
    }
}
