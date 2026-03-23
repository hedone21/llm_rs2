//! Layer Skip QCF: cosine-similarity-based layer importance.
//!
//! Measures each layer's contribution to hidden state transformation.
//! `importance(layer_i) = 1 - cosine_similarity(input_i, output_i)`
//!
//! QCF = Σ importance(skipped) / Σ importance(all)
//!
//! Built during prefill (1 pass), reused for all decode steps.

/// Sub-layer type for fine-grained importance tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubLayer {
    /// Full layer (attention + MLP combined).
    Full,
    /// Attention sub-layer only.
    Attention,
    /// MLP/FFN sub-layer only.
    Mlp,
}

/// A single importance entry for one (sub-)layer.
#[derive(Debug, Clone)]
pub struct ImportanceEntry {
    pub layer_id: usize,
    pub sublayer: SubLayer,
    pub importance: f32,
    /// Output-to-input Perturbation Ratio for layer skip:
    /// `||output - input|| / ||input||`.
    pub opr: f32,
}

/// Pre-computed importance table for all layers.
///
/// Built once during prefill via `ImportanceCollector`, then reused
/// to compute QCF for any skip pattern.
#[derive(Debug, Clone)]
pub struct ImportanceTable {
    entries: Vec<ImportanceEntry>,
    total_importance: f32,
}

impl ImportanceTable {
    /// Build table from collected entries.
    pub fn from_entries(entries: Vec<ImportanceEntry>) -> Self {
        let total_importance = entries.iter().map(|e| e.importance).sum();
        Self {
            entries,
            total_importance,
        }
    }

    /// Compute total OPR for skipped layers (sum of per-layer OPR).
    ///
    /// `skip_set`: list of `(layer_id, SubLayer)` pairs being skipped.
    /// Returns the sum of `opr` values for matched entries.
    pub fn compute_opr_skip(&self, skip_set: &[(usize, SubLayer)]) -> f32 {
        self.entries
            .iter()
            .filter(|e| skip_set.contains(&(e.layer_id, e.sublayer)))
            .map(|e| e.opr)
            .sum()
    }

    /// Compute QCF for a given skip set.
    ///
    /// `skip_set`: list of `(layer_id, SubLayer)` pairs being skipped.
    /// Returns `Σ importance(skipped) / Σ importance(all)`, in [0, 1].
    pub fn compute_qcf(&self, skip_set: &[(usize, SubLayer)]) -> f32 {
        if self.total_importance < 1e-8 || skip_set.is_empty() {
            return 0.0;
        }

        let skipped_importance: f32 = self
            .entries
            .iter()
            .filter(|e| skip_set.contains(&(e.layer_id, e.sublayer)))
            .map(|e| e.importance)
            .sum();

        (skipped_importance / self.total_importance).clamp(0.0, 1.0)
    }

    /// Estimate QCF for a target skip count by selecting the least
    /// important layers first. Layer 0 and the last layer are protected.
    ///
    /// Returns `(qcf_value, selected_skip_set)`.
    pub fn estimate_qcf_for_count(
        &self,
        skip_count: usize,
        num_layers: usize,
    ) -> (f32, Vec<(usize, SubLayer)>) {
        if skip_count == 0 || self.entries.is_empty() {
            return (0.0, Vec::new());
        }

        let last_layer = if num_layers > 0 { num_layers - 1 } else { 0 };

        // Sort eligible entries by importance (ascending = cheapest to skip first)
        let mut eligible: Vec<&ImportanceEntry> = self
            .entries
            .iter()
            .filter(|e| e.layer_id != 0 && e.layer_id != last_layer)
            .collect();
        eligible.sort_by(|a, b| a.importance.partial_cmp(&b.importance).unwrap());

        let take = skip_count.min(eligible.len());
        let skip_set: Vec<(usize, SubLayer)> = eligible[..take]
            .iter()
            .map(|e| (e.layer_id, e.sublayer))
            .collect();

        let qcf = self.compute_qcf(&skip_set);
        (qcf, skip_set)
    }

    /// Number of entries in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total importance across all entries.
    pub fn total_importance(&self) -> f32 {
        self.total_importance
    }

    /// Get all entries.
    pub fn entries(&self) -> &[ImportanceEntry] {
        &self.entries
    }
}

/// Collects per-layer hidden state snapshots during prefill to build
/// an `ImportanceTable`.
///
/// Usage in the layer loop:
/// ```ignore
/// let mut collector = ImportanceCollector::new();
/// for (i, layer) in layers.iter().enumerate() {
///     collector.snapshot_before(&x);
///     layer.forward(...)?;
///     collector.record_after(&x, i);
/// }
/// let table = collector.build();
/// ```
pub struct ImportanceCollector {
    entries: Vec<ImportanceEntry>,
    /// Snapshot of hidden state before the current layer.
    before_snapshot: Vec<f32>,
}

impl ImportanceCollector {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            before_snapshot: Vec::new(),
        }
    }

    /// Snapshot the hidden state before a layer processes it.
    ///
    /// For prefill, `x` is `[batch, seq_len, dim]`. We take the mean
    /// across the sequence dimension to get a single `[dim]` vector.
    pub fn snapshot_before(&mut self, x_data: &[f32], seq_len: usize, dim: usize) {
        self.before_snapshot.clear();
        self.before_snapshot.resize(dim, 0.0);

        if seq_len == 0 {
            return;
        }

        // Mean-pool across sequence positions (batch=1 assumed)
        for pos in 0..seq_len {
            let offset = pos * dim;
            for d in 0..dim {
                if offset + d < x_data.len() {
                    self.before_snapshot[d] += x_data[offset + d];
                }
            }
        }
        let scale = 1.0 / seq_len as f32;
        for v in &mut self.before_snapshot {
            *v *= scale;
        }
    }

    /// Record importance after a layer has processed the hidden state.
    ///
    /// Computes `importance = 1 - cosine_similarity(before, after)`.
    pub fn record_after(
        &mut self,
        x_data: &[f32],
        seq_len: usize,
        dim: usize,
        layer_id: usize,
        sublayer: SubLayer,
    ) {
        // Mean-pool after state
        let mut after = vec![0.0f32; dim];
        if seq_len > 0 {
            for pos in 0..seq_len {
                let offset = pos * dim;
                for d in 0..dim {
                    if offset + d < x_data.len() {
                        after[d] += x_data[offset + d];
                    }
                }
            }
            let scale = 1.0 / seq_len as f32;
            for v in &mut after {
                *v *= scale;
            }
        }

        let cos_sim = cosine_similarity(&self.before_snapshot, &after);
        let importance = (1.0 - cos_sim).max(0.0);
        let opr = residual_norm_ratio(&self.before_snapshot, &after);

        self.entries.push(ImportanceEntry {
            layer_id,
            sublayer,
            importance,
            opr,
        });
    }

    /// Build the final `ImportanceTable`.
    pub fn build(self) -> ImportanceTable {
        ImportanceTable::from_entries(self.entries)
    }
}

impl Default for ImportanceCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// OPR for layer skip: `||output - input|| / ||input||`.
///
/// Returns 0.0 if `input` has zero magnitude.
fn residual_norm_ratio(before: &[f32], after: &[f32]) -> f32 {
    let len = before.len().min(after.len());
    if len == 0 {
        return 0.0;
    }
    let mut diff_sq = 0.0f32;
    let mut before_sq = 0.0f32;
    for i in 0..len {
        let d = after[i] - before[i];
        diff_sq += d * d;
        before_sq += before[i] * before[i];
    }
    let denom = before_sq.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        diff_sq.sqrt() / denom
    }
}

/// Cosine similarity between two float slices.
///
/// Returns 0.0 if either vector has zero magnitude.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let cos = cosine_similarity(&a, &a);
        assert!((cos - 1.0).abs() < 1e-6, "identical vectors: cos={cos}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let cos = cosine_similarity(&a, &b);
        assert!(cos.abs() < 1e-6, "orthogonal: cos={cos}");
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let cos = cosine_similarity(&a, &b);
        assert!((cos - (-1.0)).abs() < 1e-6, "opposite: cos={cos}");
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        let cos = cosine_similarity(&a, &b);
        assert_eq!(cos, 0.0);
    }

    #[test]
    fn test_importance_table_empty_skip() {
        let entries = vec![
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Full,
                importance: 0.5,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Full,
                importance: 0.3,
                opr: 0.0,
            },
        ];
        let table = ImportanceTable::from_entries(entries);
        assert_eq!(table.compute_qcf(&[]), 0.0);
    }

    #[test]
    fn test_importance_table_full_skip() {
        let entries = vec![
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Full,
                importance: 0.5,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Full,
                importance: 0.3,
                opr: 0.0,
            },
        ];
        let table = ImportanceTable::from_entries(entries);
        let qcf = table.compute_qcf(&[(0, SubLayer::Full), (1, SubLayer::Full)]);
        assert!((qcf - 1.0).abs() < 1e-6, "full skip: qcf={qcf}");
    }

    #[test]
    fn test_importance_table_partial_skip() {
        // Guide §3-3 simplified example
        let entries = vec![
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Full,
                importance: 0.42,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Full,
                importance: 0.08,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 2,
                sublayer: SubLayer::Full,
                importance: 0.31,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 3,
                sublayer: SubLayer::Full,
                importance: 0.05,
                opr: 0.0,
            },
        ];
        let table = ImportanceTable::from_entries(entries);
        // total = 0.86
        // skip layers 1 and 3 (low importance): 0.08 + 0.05 = 0.13
        let qcf = table.compute_qcf(&[(1, SubLayer::Full), (3, SubLayer::Full)]);
        let expected = 0.13 / 0.86;
        assert!(
            (qcf - expected).abs() < 0.01,
            "expected ~{expected:.3}, got {qcf:.3}"
        );
    }

    #[test]
    fn test_estimate_qcf_protects_first_last() {
        let entries = vec![
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Full,
                importance: 0.9,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Full,
                importance: 0.1,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 2,
                sublayer: SubLayer::Full,
                importance: 0.2,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 3,
                sublayer: SubLayer::Full,
                importance: 0.8,
                opr: 0.0,
            },
        ];
        let table = ImportanceTable::from_entries(entries);

        // Skip 2 layers from 4-layer model: layer 0 and 3 protected
        let (qcf, skip_set) = table.estimate_qcf_for_count(2, 4);
        // Only layers 1 and 2 eligible → both skipped
        assert_eq!(skip_set.len(), 2);
        assert!(skip_set.contains(&(1, SubLayer::Full)));
        assert!(skip_set.contains(&(2, SubLayer::Full)));
        // QCF = (0.1 + 0.2) / 2.0 = 0.15
        assert!((qcf - 0.15).abs() < 0.01, "expected ~0.15, got {qcf}");
    }

    #[test]
    fn test_estimate_qcf_selects_least_important() {
        let entries = vec![
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Full,
                importance: 0.9,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Full,
                importance: 0.5,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 2,
                sublayer: SubLayer::Full,
                importance: 0.1,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 3,
                sublayer: SubLayer::Full,
                importance: 0.3,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 4,
                sublayer: SubLayer::Full,
                importance: 0.7,
                opr: 0.0,
            },
        ];
        let table = ImportanceTable::from_entries(entries);

        // Skip 1 from 5-layer model: should pick layer 2 (importance=0.1)
        let (_, skip_set) = table.estimate_qcf_for_count(1, 5);
        assert_eq!(skip_set.len(), 1);
        assert_eq!(skip_set[0], (2, SubLayer::Full));
    }

    #[test]
    fn test_collector_basic() {
        let dim = 4;
        let seq_len = 2;

        // Before: [[1,0,0,0], [1,0,0,0]] → mean = [1,0,0,0]
        let before_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        // After: [[0,1,0,0], [0,1,0,0]] → mean = [0,1,0,0]
        let after_data = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        let mut collector = ImportanceCollector::new();
        collector.snapshot_before(&before_data, seq_len, dim);
        collector.record_after(&after_data, seq_len, dim, 0, SubLayer::Full);

        let table = collector.build();
        assert_eq!(table.len(), 1);
        // cos_sim([1,0,0,0], [0,1,0,0]) = 0 → importance = 1.0
        let entry = &table.entries()[0];
        assert!(
            (entry.importance - 1.0).abs() < 0.01,
            "imp={}",
            entry.importance
        );
    }

    #[test]
    fn test_collector_identity_layer() {
        let dim = 4;
        let seq_len = 1;
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let mut collector = ImportanceCollector::new();
        collector.snapshot_before(&data, seq_len, dim);
        // Same data → identity layer → importance ≈ 0
        collector.record_after(&data, seq_len, dim, 5, SubLayer::Full);

        let table = collector.build();
        let entry = &table.entries()[0];
        assert!(
            entry.importance < 0.01,
            "identity layer imp={}",
            entry.importance
        );
    }

    #[test]
    fn test_sublayer_separation() {
        let entries = vec![
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Attention,
                importance: 0.4,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Mlp,
                importance: 0.1,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Attention,
                importance: 0.3,
                opr: 0.0,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Mlp,
                importance: 0.2,
                opr: 0.0,
            },
        ];
        let table = ImportanceTable::from_entries(entries);
        // total = 1.0
        // Skip only MLP of layer 0 and attention of layer 1
        let qcf = table.compute_qcf(&[(0, SubLayer::Mlp), (1, SubLayer::Attention)]);
        // (0.1 + 0.3) / 1.0 = 0.4
        assert!((qcf - 0.4).abs() < 1e-6, "sublayer qcf={qcf}");
    }

    // --- OPR tests ---

    #[test]
    fn test_residual_norm_ratio_identity() {
        // output == input → residual = 0 → OPR = 0
        let v = vec![1.0, 2.0, 3.0];
        let opr = residual_norm_ratio(&v, &v);
        assert!(opr.abs() < 1e-6, "identity opr={opr}");
    }

    #[test]
    fn test_residual_norm_ratio_orthogonal() {
        // before=[1,0], after=[0,1]: diff=[−1,1], ||diff||=√2, ||before||=1 → OPR=√2
        let before = vec![1.0, 0.0];
        let after = vec![0.0, 1.0];
        let opr = residual_norm_ratio(&before, &after);
        let expected = 2.0_f32.sqrt();
        assert!(
            (opr - expected).abs() < 1e-5,
            "orthogonal opr={opr}, expected {expected}"
        );
    }

    #[test]
    fn test_residual_norm_ratio_zero_input() {
        // zero input → OPR = 0 (no division by zero)
        let before = vec![0.0, 0.0, 0.0];
        let after = vec![1.0, 2.0, 3.0];
        let opr = residual_norm_ratio(&before, &after);
        assert_eq!(opr, 0.0, "zero input opr={opr}");
    }

    #[test]
    fn test_compute_opr_skip() {
        let entries = vec![
            ImportanceEntry {
                layer_id: 0,
                sublayer: SubLayer::Full,
                importance: 0.5,
                opr: 0.2,
            },
            ImportanceEntry {
                layer_id: 1,
                sublayer: SubLayer::Full,
                importance: 0.3,
                opr: 0.4,
            },
            ImportanceEntry {
                layer_id: 2,
                sublayer: SubLayer::Full,
                importance: 0.2,
                opr: 0.1,
            },
        ];
        let table = ImportanceTable::from_entries(entries);

        // Skip layers 0 and 2 → OPR sum = 0.2 + 0.1 = 0.3
        let opr_sum = table.compute_opr_skip(&[(0, SubLayer::Full), (2, SubLayer::Full)]);
        assert!((opr_sum - 0.3).abs() < 1e-6, "opr_sum={opr_sum}");

        // Empty skip set → 0.0
        assert_eq!(table.compute_opr_skip(&[]), 0.0);
    }

    #[test]
    fn test_collector_records_opr() {
        let dim = 4;
        let seq_len = 1;

        // before=[1,0,0,0], after=[0,1,0,0]
        // residual=[-1,1,0,0], ||residual||=√2, ||before||=1 → OPR=√2
        let before_data = vec![1.0, 0.0, 0.0, 0.0];
        let after_data = vec![0.0, 1.0, 0.0, 0.0];

        let mut collector = ImportanceCollector::new();
        collector.snapshot_before(&before_data, seq_len, dim);
        collector.record_after(&after_data, seq_len, dim, 0, SubLayer::Full);

        let table = collector.build();
        let entry = &table.entries()[0];
        let expected_opr = 2.0_f32.sqrt();
        assert!(
            (entry.opr - expected_opr).abs() < 1e-5,
            "collector opr={}, expected {expected_opr}",
            entry.opr
        );
        // importance should still be 1.0 (orthogonal vectors)
        assert!(
            (entry.importance - 1.0).abs() < 0.01,
            "importance should be 1.0, got {}",
            entry.importance
        );
    }
}
