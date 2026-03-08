/// Accumulates per-token attention importance scores across layers.
///
/// During decode, each layer's post-softmax attention weights are aggregated
/// into a per-token importance score. H2O uses these scores to decide
/// which tokens to keep vs evict.
pub struct AttentionScoreAccumulator {
    /// Per-token cumulative importance scores, indexed by cache position.
    /// Updated once per step via `end_step()`.
    importance: Vec<f32>,
    /// Per-token step-local importance buffer.
    /// Within a single decode step, each layer's score is aggregated here
    /// using MAX (per-layer independence), then flushed to `importance` in `end_step()`.
    step_importance: Vec<f32>,
    /// Number of query heads in the model.
    #[allow(dead_code)]
    n_heads: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Which layers to track. Empty means track all layers.
    tracked_layers: Vec<usize>,
    /// Total number of layers in the model.
    #[allow(dead_code)]
    total_layers: usize,
    /// Exponential decay factor (0.0 = no decay, 1.0 = full decay).
    decay: f32,
    /// Whether accumulation is active.
    active: bool,
}

impl AttentionScoreAccumulator {
    pub fn new(
        max_seq_len: usize,
        n_heads: usize,
        total_layers: usize,
        last_n_layers: usize,
        decay: f32,
    ) -> Self {
        let tracked_layers = if last_n_layers == 0 || last_n_layers >= total_layers {
            Vec::new()
        } else {
            ((total_layers - last_n_layers)..total_layers).collect()
        };

        Self {
            importance: vec![0.0; max_seq_len],
            step_importance: vec![0.0; max_seq_len],
            n_heads,
            max_seq_len,
            tracked_layers,
            total_layers,
            decay: decay.clamp(0.0, 1.0),
            active: false,
        }
    }

    /// Returns whether this layer should be tracked.
    #[inline]
    pub fn should_track_layer(&self, layer_idx: usize) -> bool {
        self.active && (self.tracked_layers.is_empty() || self.tracked_layers.contains(&layer_idx))
    }

    /// Called once per decode step before layer iteration.
    /// Applies decay to cumulative importance and clears step-local buffer.
    pub fn begin_step(&mut self) {
        if !self.active {
            return;
        }
        if self.decay > 0.0 {
            let factor = 1.0 - self.decay;
            for v in self.importance.iter_mut() {
                *v *= factor;
            }
        }
        self.step_importance.fill(0.0);
    }

    /// Accumulate post-softmax attention scores from one layer.
    ///
    /// Uses per-layer MAX aggregation: for each token, the step-local importance
    /// is the maximum across all layers (not sum). This ensures tokens critical
    /// to ANY layer are preserved, matching the H2O paper's per-layer independence.
    ///
    /// Within a single layer, scores across query heads are summed first to get
    /// a single per-token score for that layer.
    ///
    /// `scores`: flat buffer `[n_heads_q * stride]`, where stride >= cache_seq_len.
    /// `stride`: distance between head score arrays.
    /// `cache_seq_len`: number of valid token positions in cache.
    /// `n_heads_q`: number of query heads.
    pub fn accumulate_layer(
        &mut self,
        scores: &[f32],
        stride: usize,
        cache_seq_len: usize,
        n_heads_q: usize,
    ) {
        let len = cache_seq_len.min(self.max_seq_len);

        // Sum across query heads to get a single per-token score for this layer
        // Then take MAX with existing step_importance (per-layer MAX)
        for t in 0..len {
            let mut layer_score = 0.0f32;
            for h in 0..n_heads_q {
                layer_score += scores[h * stride + t];
            }
            self.step_importance[t] = self.step_importance[t].max(layer_score);
        }
    }

    /// Called once per decode step after all layers have been processed.
    /// Flushes step-local importance (per-layer MAX) into cumulative importance.
    pub fn end_step(&mut self) {
        if !self.active {
            return;
        }
        for (cum, &step) in self.importance.iter_mut().zip(self.step_importance.iter()) {
            *cum += step;
        }
    }

    /// Get the importance scores slice.
    pub fn importance_scores(&self) -> &[f32] {
        &self.importance
    }

    /// Reset all accumulated scores (e.g., after eviction).
    pub fn reset(&mut self) {
        self.importance.fill(0.0);
        self.step_importance.fill(0.0);
    }

    /// Activate accumulation.
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulate_single_layer() {
        let mut acc = AttentionScoreAccumulator::new(8, 2, 4, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        // 2 heads, stride=8, cache_seq_len=4
        // head0: [0.1, 0.2, 0.3, 0.4, 0,0,0,0]
        // head1: [0.4, 0.3, 0.2, 0.1, 0,0,0,0]
        let scores = vec![
            0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0,
        ];
        acc.accumulate_layer(&scores, 8, 4, 2);
        acc.end_step();

        // Per-token: sum across heads → layer_score, then MAX with step_importance
        // token 0: 0.1+0.4=0.5, token 1: 0.2+0.3=0.5, token 2: 0.3+0.2=0.5, token 3: 0.4+0.1=0.5
        let imp = acc.importance_scores();
        assert!((imp[0] - 0.5).abs() < 1e-6);
        assert!((imp[1] - 0.5).abs() < 1e-6);
        assert!((imp[2] - 0.5).abs() < 1e-6);
        assert!((imp[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_accumulate_multi_layer_uses_max() {
        // With per-layer MAX: step_importance = MAX(layer1_score, layer2_score)
        let mut acc = AttentionScoreAccumulator::new(4, 1, 2, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        let scores1 = vec![0.1, 0.2, 0.3, 0.4]; // layer 0 scores
        let scores2 = vec![0.4, 0.1, 0.1, 0.4]; // layer 1 scores

        acc.accumulate_layer(&scores1, 4, 4, 1);
        acc.accumulate_layer(&scores2, 4, 4, 1);
        acc.end_step();

        // Per-layer MAX: max(0.1,0.4)=0.4, max(0.2,0.1)=0.2, max(0.3,0.1)=0.3, max(0.4,0.4)=0.4
        let imp = acc.importance_scores();
        assert!((imp[0] - 0.4).abs() < 1e-6);
        assert!((imp[1] - 0.2).abs() < 1e-6);
        assert!((imp[2] - 0.3).abs() < 1e-6);
        assert!((imp[3] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_cumulative_across_steps() {
        // importance accumulates across multiple decode steps
        let mut acc = AttentionScoreAccumulator::new(4, 1, 1, 0, 0.0);
        acc.set_active(true);

        // Step 1
        acc.begin_step();
        acc.accumulate_layer(&[1.0, 2.0, 3.0, 4.0], 4, 4, 1);
        acc.end_step();

        // Step 2
        acc.begin_step();
        acc.accumulate_layer(&[4.0, 3.0, 2.0, 1.0], 4, 4, 1);
        acc.end_step();

        // importance = step1 + step2 = [5.0, 5.0, 5.0, 5.0]
        let imp = acc.importance_scores();
        assert!((imp[0] - 5.0).abs() < 1e-6);
        assert!((imp[1] - 5.0).abs() < 1e-6);
        assert!((imp[2] - 5.0).abs() < 1e-6);
        assert!((imp[3] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_decay() {
        let mut acc = AttentionScoreAccumulator::new(4, 1, 1, 0, 0.5);
        acc.set_active(true);

        // Step 1: accumulate
        acc.begin_step();
        acc.accumulate_layer(&[1.0, 2.0, 3.0, 4.0], 4, 4, 1);
        acc.end_step();
        // importance = [1, 2, 3, 4]

        // Step 2: begin_step applies decay, then end_step adds new scores
        acc.begin_step(); // decay 0.5 → importance = [0.5, 1.0, 1.5, 2.0]
        // No accumulation in this step, just check decay effect
        let imp = acc.importance_scores();
        assert!((imp[0] - 0.5).abs() < 1e-6);
        assert!((imp[1] - 1.0).abs() < 1e-6);
        assert!((imp[2] - 1.5).abs() < 1e-6);
        assert!((imp[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_should_track_layer() {
        // Track last 2 of 4 layers => layers 2, 3
        let mut acc = AttentionScoreAccumulator::new(4, 1, 4, 2, 0.0);
        acc.set_active(true);

        assert!(!acc.should_track_layer(0));
        assert!(!acc.should_track_layer(1));
        assert!(acc.should_track_layer(2));
        assert!(acc.should_track_layer(3));
    }

    #[test]
    fn test_reset() {
        let mut acc = AttentionScoreAccumulator::new(4, 1, 1, 0, 0.0);
        acc.set_active(true);

        acc.begin_step();
        acc.accumulate_layer(&[1.0, 2.0, 3.0, 4.0], 4, 4, 1);
        acc.end_step();
        acc.reset();

        assert!(acc.importance_scores().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_inactive_no_accumulation() {
        let mut acc = AttentionScoreAccumulator::new(4, 1, 2, 0, 0.0);
        // active defaults to false

        assert!(!acc.should_track_layer(0));
        assert!(!acc.is_active());

        // begin_step and end_step should be no-ops when inactive
        acc.begin_step();
        acc.end_step();
    }

    #[test]
    fn test_end_step_without_begin_step() {
        // end_step should work even if begin_step wasn't called (step_importance starts zeroed)
        let mut acc = AttentionScoreAccumulator::new(4, 1, 1, 0, 0.0);
        acc.set_active(true);
        acc.end_step();
        assert!(acc.importance_scores().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_per_layer_max_preserves_critical_tokens() {
        // Token critical to only one layer should still be preserved
        let mut acc = AttentionScoreAccumulator::new(4, 1, 2, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        // Layer 0: token 0 is critical (score=9.0), token 1 low (0.1)
        acc.accumulate_layer(&[9.0, 0.1, 0.1, 0.1], 4, 4, 1);
        // Layer 1: token 1 is critical (score=9.0), token 0 low (0.1)
        acc.accumulate_layer(&[0.1, 9.0, 0.1, 0.1], 4, 4, 1);
        acc.end_step();

        // MAX: both token 0 and token 1 should have high importance (9.0)
        let imp = acc.importance_scores();
        assert!((imp[0] - 9.0).abs() < 1e-6);
        assert!((imp[1] - 9.0).abs() < 1e-6);
        assert!((imp[2] - 0.1).abs() < 1e-6);
    }
}
