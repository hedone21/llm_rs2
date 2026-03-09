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
    // ── GQA-aware fields ──
    /// Number of KV heads for GQA grouping. 0 = GQA mode disabled.
    n_kv_heads: usize,
    /// Per-KV-head cumulative importance: `[n_kv_heads * max_seq_len]`, row-major.
    head_importance: Vec<f32>,
    /// Per-KV-head step-local buffer (same layout).
    head_step_importance: Vec<f32>,
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
            n_kv_heads: 0,
            head_importance: Vec::new(),
            head_step_importance: Vec::new(),
        }
    }

    /// Create a GQA-aware accumulator that tracks per-KV-head importance.
    ///
    /// In addition to flat per-token importance (backward compatible),
    /// maintains a 2D `[n_kv_heads, max_seq_len]` importance matrix where
    /// Q-head scores are averaged within each GQA group.
    pub fn new_gqa(
        max_seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        total_layers: usize,
        last_n_layers: usize,
        decay: f32,
    ) -> Self {
        let tracked_layers = if last_n_layers == 0 || last_n_layers >= total_layers {
            Vec::new()
        } else {
            ((total_layers - last_n_layers)..total_layers).collect()
        };

        let head_buf_size = n_kv_heads * max_seq_len;
        Self {
            importance: vec![0.0; max_seq_len],
            step_importance: vec![0.0; max_seq_len],
            n_heads,
            max_seq_len,
            tracked_layers,
            total_layers,
            decay: decay.clamp(0.0, 1.0),
            active: false,
            n_kv_heads,
            head_importance: vec![0.0; head_buf_size],
            head_step_importance: vec![0.0; head_buf_size],
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
            for v in self.head_importance.iter_mut() {
                *v *= factor;
            }
        }
        self.step_importance.fill(0.0);
        self.head_step_importance.fill(0.0);
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

    /// GQA-aware accumulation: in addition to flat per-token scores,
    /// computes per-KV-head importance by averaging Q-head scores within
    /// each GQA group.
    ///
    /// `scores`: flat buffer `[n_heads_q * stride]`.
    /// `n_heads_q`: total query heads (must be divisible by `n_kv_heads`).
    pub fn accumulate_layer_gqa(
        &mut self,
        scores: &[f32],
        stride: usize,
        cache_seq_len: usize,
        n_heads_q: usize,
        n_kv_heads: usize,
    ) {
        let len = cache_seq_len.min(self.max_seq_len);
        let n_rep = n_heads_q / n_kv_heads;
        let inv_rep = 1.0 / n_rep as f32;

        for t in 0..len {
            // Flat accumulation (backward compatible with H2O)
            let mut layer_score = 0.0f32;
            for h in 0..n_heads_q {
                layer_score += scores[h * stride + t];
            }
            self.step_importance[t] = self.step_importance[t].max(layer_score);

            // Per-KV-head: average Q-heads within each GQA group
            for kv_h in 0..n_kv_heads {
                let mut group_score = 0.0f32;
                for r in 0..n_rep {
                    group_score += scores[(kv_h * n_rep + r) * stride + t];
                }
                group_score *= inv_rep;
                let idx = kv_h * self.max_seq_len + t;
                self.head_step_importance[idx] = self.head_step_importance[idx].max(group_score);
            }
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
        for (cum, &step) in self
            .head_importance
            .iter_mut()
            .zip(self.head_step_importance.iter())
        {
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
        self.head_importance.fill(0.0);
        self.head_step_importance.fill(0.0);
    }

    /// Activate accumulation.
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Returns per-KV-head importance scores if GQA mode is active.
    /// Layout: `[n_kv_heads * max_seq_len]`, row-major.
    pub fn head_importance_scores(&self) -> Option<&[f32]> {
        if self.n_kv_heads > 0 {
            Some(&self.head_importance)
        } else {
            None
        }
    }

    /// Number of KV heads (0 = GQA mode disabled).
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
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

    // ── GQA mode tests ──

    #[test]
    fn test_accumulate_gqa_groups_q_heads() {
        // 4 Q-heads, 2 KV-heads → n_rep=2 (heads 0,1 → KV0; heads 2,3 → KV1)
        let mut acc = AttentionScoreAccumulator::new_gqa(4, 4, 2, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        // stride=4, cache_seq_len=4
        // Q-head 0: [1.0, 0.0, 0.0, 0.0]  ← KV group 0
        // Q-head 1: [0.0, 1.0, 0.0, 0.0]  ← KV group 0
        // Q-head 2: [0.0, 0.0, 1.0, 0.0]  ← KV group 1
        // Q-head 3: [0.0, 0.0, 0.0, 1.0]  ← KV group 1
        let scores = vec![
            1.0, 0.0, 0.0, 0.0, // Q-head 0
            0.0, 1.0, 0.0, 0.0, // Q-head 1
            0.0, 0.0, 1.0, 0.0, // Q-head 2
            0.0, 0.0, 0.0, 1.0, // Q-head 3
        ];
        acc.accumulate_layer_gqa(&scores, 4, 4, 4, 2);
        acc.end_step();

        // KV head 0: avg(Q0, Q1) per token
        //   token 0: (1.0+0.0)/2 = 0.5
        //   token 1: (0.0+1.0)/2 = 0.5
        //   token 2: 0.0, token 3: 0.0
        let head_imp = acc.head_importance_scores().unwrap();
        assert!((head_imp[0] - 0.5).abs() < 1e-6); // KV0, tok0
        assert!((head_imp[1] - 0.5).abs() < 1e-6); // KV0, tok1
        assert!((head_imp[2] - 0.0).abs() < 1e-6); // KV0, tok2
        assert!((head_imp[3] - 0.0).abs() < 1e-6); // KV0, tok3

        // KV head 1: avg(Q2, Q3) per token
        //   token 2: (1.0+0.0)/2 = 0.5
        //   token 3: (0.0+1.0)/2 = 0.5
        assert!((head_imp[4] - 0.0).abs() < 1e-6); // KV1, tok0
        assert!((head_imp[5] - 0.0).abs() < 1e-6); // KV1, tok1
        assert!((head_imp[6] - 0.5).abs() < 1e-6); // KV1, tok2
        assert!((head_imp[7] - 0.5).abs() < 1e-6); // KV1, tok3
    }

    #[test]
    fn test_gqa_also_updates_flat() {
        // GQA accumulation should also update flat importance (backward compat)
        let mut acc = AttentionScoreAccumulator::new_gqa(4, 4, 2, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        let scores = vec![
            1.0, 0.0, 0.0, 0.0, // Q0
            0.0, 1.0, 0.0, 0.0, // Q1
            0.0, 0.0, 1.0, 0.0, // Q2
            0.0, 0.0, 0.0, 1.0, // Q3
        ];
        acc.accumulate_layer_gqa(&scores, 4, 4, 4, 2);
        acc.end_step();

        // Flat importance = sum across all Q-heads per token
        let imp = acc.importance_scores();
        assert!((imp[0] - 1.0).abs() < 1e-6);
        assert!((imp[1] - 1.0).abs() < 1e-6);
        assert!((imp[2] - 1.0).abs() < 1e-6);
        assert!((imp[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gqa_decay() {
        let mut acc = AttentionScoreAccumulator::new_gqa(4, 2, 2, 1, 0, 0.5);
        acc.set_active(true);

        // Step 1
        acc.begin_step();
        // 2 Q-heads, 2 KV-heads (n_rep=1)
        let scores = vec![
            2.0, 4.0, 0.0, 0.0, // Q0 → KV0
            0.0, 0.0, 6.0, 8.0, // Q1 → KV1
        ];
        acc.accumulate_layer_gqa(&scores, 4, 4, 2, 2);
        acc.end_step();

        // head_importance: KV0=[2,4,0,0], KV1=[0,0,6,8]

        // Step 2: decay 0.5
        acc.begin_step();
        let head_imp = acc.head_importance_scores().unwrap();
        assert!((head_imp[0] - 1.0).abs() < 1e-6); // 2*0.5
        assert!((head_imp[1] - 2.0).abs() < 1e-6); // 4*0.5
        assert!((head_imp[4] - 0.0).abs() < 1e-6); // 0*0.5
        assert!((head_imp[6] - 3.0).abs() < 1e-6); // 6*0.5
        assert!((head_imp[7] - 4.0).abs() < 1e-6); // 8*0.5
    }

    #[test]
    fn test_gqa_reset() {
        let mut acc = AttentionScoreAccumulator::new_gqa(4, 2, 2, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();
        let scores = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        acc.accumulate_layer_gqa(&scores, 4, 4, 2, 2);
        acc.end_step();
        acc.reset();

        assert!(acc.importance_scores().iter().all(|&v| v == 0.0));
        assert!(
            acc.head_importance_scores()
                .unwrap()
                .iter()
                .all(|&v| v == 0.0)
        );
    }

    #[test]
    fn test_head_importance_accessor() {
        // Non-GQA: returns None
        let acc = AttentionScoreAccumulator::new(4, 1, 1, 0, 0.0);
        assert!(acc.head_importance_scores().is_none());
        assert_eq!(acc.n_kv_heads(), 0);

        // GQA: returns Some
        let acc = AttentionScoreAccumulator::new_gqa(4, 4, 2, 1, 0, 0.0);
        assert!(acc.head_importance_scores().is_some());
        assert_eq!(acc.head_importance_scores().unwrap().len(), 2 * 4);
        assert_eq!(acc.n_kv_heads(), 2);
    }

    #[test]
    fn test_gqa_multi_layer_max() {
        // Per-layer MAX should apply to per-head importance too
        let mut acc = AttentionScoreAccumulator::new_gqa(4, 2, 2, 2, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();

        // Layer 0: KV0 high on tok0, KV1 low
        let scores_l0 = vec![
            9.0, 0.1, 0.1, 0.1, // Q0 → KV0
            0.1, 0.1, 0.1, 0.1, // Q1 → KV1
        ];
        acc.accumulate_layer_gqa(&scores_l0, 4, 4, 2, 2);

        // Layer 1: KV0 low, KV1 high on tok2
        let scores_l1 = vec![
            0.1, 0.1, 0.1, 0.1, // Q0 → KV0
            0.1, 0.1, 9.0, 0.1, // Q1 → KV1
        ];
        acc.accumulate_layer_gqa(&scores_l1, 4, 4, 2, 2);
        acc.end_step();

        let head_imp = acc.head_importance_scores().unwrap();
        // KV0: max(layer0, layer1) per token
        //   tok0: max(9.0, 0.1) = 9.0
        assert!((head_imp[0] - 9.0).abs() < 1e-6);
        //   tok1: max(0.1, 0.1) = 0.1
        assert!((head_imp[1] - 0.1).abs() < 1e-6);

        // KV1: max(layer0, layer1) per token
        //   tok2: max(0.1, 9.0) = 9.0
        assert!((head_imp[4 + 2] - 9.0).abs() < 1e-6);
    }
}
