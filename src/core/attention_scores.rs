/// Accumulates per-token attention importance scores across layers.
///
/// During decode, each layer's post-softmax attention weights are aggregated
/// into a per-token importance score. SnapKV uses these scores to decide
/// which tokens to keep vs evict.
pub struct AttentionScoreAccumulator {
    /// Per-token importance scores, indexed by cache position.
    importance: Vec<f32>,
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

    /// Called once per decode step before layer iteration to apply decay.
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
    }

    /// Accumulate post-softmax attention scores from one layer.
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
        for h in 0..n_heads_q {
            let offset = h * stride;
            for t in 0..len {
                self.importance[t] += scores[offset + t];
            }
        }
    }

    /// Get the importance scores slice.
    pub fn importance_scores(&self) -> &[f32] {
        &self.importance
    }

    /// Reset all accumulated scores (e.g., after eviction).
    pub fn reset(&mut self) {
        self.importance.fill(0.0);
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

        // 2 heads, stride=8, cache_seq_len=4
        // head0: [0.1, 0.2, 0.3, 0.4, 0,0,0,0]
        // head1: [0.4, 0.3, 0.2, 0.1, 0,0,0,0]
        let scores = vec![
            0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0,
        ];
        acc.accumulate_layer(&scores, 8, 4, 2);

        let imp = acc.importance_scores();
        assert!((imp[0] - 0.5).abs() < 1e-6); // 0.1 + 0.4
        assert!((imp[1] - 0.5).abs() < 1e-6); // 0.2 + 0.3
        assert!((imp[2] - 0.5).abs() < 1e-6); // 0.3 + 0.2
        assert!((imp[3] - 0.5).abs() < 1e-6); // 0.4 + 0.1
    }

    #[test]
    fn test_accumulate_multi_layer() {
        let mut acc = AttentionScoreAccumulator::new(4, 1, 2, 0, 0.0);
        acc.set_active(true);

        let scores1 = vec![0.1, 0.2, 0.3, 0.4];
        let scores2 = vec![0.4, 0.1, 0.1, 0.4];

        acc.accumulate_layer(&scores1, 4, 4, 1);
        acc.accumulate_layer(&scores2, 4, 4, 1);

        let imp = acc.importance_scores();
        assert!((imp[0] - 0.5).abs() < 1e-6); // 0.1 + 0.4
        assert!((imp[1] - 0.3).abs() < 1e-6); // 0.2 + 0.1
        assert!((imp[2] - 0.4).abs() < 1e-6); // 0.3 + 0.1
        assert!((imp[3] - 0.8).abs() < 1e-6); // 0.4 + 0.4
    }

    #[test]
    fn test_decay() {
        let mut acc = AttentionScoreAccumulator::new(4, 1, 2, 0, 0.5);
        acc.set_active(true);

        let scores = vec![1.0, 2.0, 3.0, 4.0];
        acc.accumulate_layer(&scores, 4, 4, 1);

        // importance = [1, 2, 3, 4]
        acc.begin_step(); // decay 0.5 => factor 0.5
        // importance = [0.5, 1.0, 1.5, 2.0]

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
        let mut acc = AttentionScoreAccumulator::new(4, 1, 2, 0, 0.0);
        acc.set_active(true);

        let scores = vec![1.0, 2.0, 3.0, 4.0];
        acc.accumulate_layer(&scores, 4, 4, 1);
        acc.reset();

        assert!(acc.importance_scores().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_inactive_no_accumulation() {
        let mut acc = AttentionScoreAccumulator::new(4, 1, 2, 0, 0.0);
        // active defaults to false

        assert!(!acc.should_track_layer(0));
        assert!(!acc.is_active());

        // begin_step should be a no-op
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        acc.accumulate_layer(&scores, 4, 4, 1);
        // Note: accumulate_layer itself doesn't check active (caller should check via should_track_layer)
        // The guard is at the call site, not inside accumulate_layer
    }
}
