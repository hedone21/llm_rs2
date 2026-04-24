//! `WeightSwapDecider` — importance × ε bottom-k layer selection (ENG-ALG-215).
//!
//! Converts a Manager-supplied `ratio_max` into a concrete set of decoder layer
//! indices to swap from F16 to Q4_0.  Selection key is `importance_i × ε_i`
//! ascending: layers with the smallest product are swapped first (they are the
//! cheapest in quality terms).
//!
//! Layer 0 and the last decoder layer are always protected (ENG-ALG-215).
//! Layers already swapped (in `currently_swapped`) are excluded to keep the
//! operation idempotent.  Layers whose ε is NaN (INV-127) are excluded.
//!
//! When either `ImportanceTable` or `QuantNoiseTable` is absent / uncomputed,
//! the decider falls back to uniform index-spaced selection (ENG-ALG-213
//! fallback, absorbed by ENG-ALG-215).
//!
//! Spec: ENG-ALG-215, ENG-ALG-217, INV-127.

use std::collections::HashSet;

use crate::core::qcf::layer_importance::{ImportanceTable, SubLayer};
use crate::models::weights::QuantNoiseTable;

// ── Public types ──────────────────────────────────────────────────────────────

/// Result of `WeightSwapDecider::decide()`.
#[derive(Debug, Clone)]
pub struct SwapDecision {
    /// Decoder layer indices selected for this swap batch.
    pub selected_layers: Vec<usize>,
    /// QCF_swap estimate for the selected set (ENG-ALG-217).
    pub qcf_swap_estimate: f32,
    /// `true` when a uniform fallback was used (importance or noise absent).
    pub fallback_used: bool,
}

/// Stateless decider that converts `ratio_max` to a layer index set.
///
/// Both `importance` and `noise` are optional: pass `None` when the table has
/// not been built yet.  When either is absent (or effectively empty /
/// uncomputed), the decider switches to the uniform fallback path.
pub struct WeightSwapDecider<'a> {
    /// Importance table from the last prefill (None = fallback).
    pub importance: Option<&'a ImportanceTable>,
    /// Quantization noise table from engine init (None = fallback).
    pub noise: Option<&'a QuantNoiseTable>,
    /// Total number of decoder layers in the model.
    pub n_decoder_layers: usize,
    /// Layers that are already at the target dtype — excluded from re-selection.
    pub currently_swapped: &'a [usize],
}

impl<'a> WeightSwapDecider<'a> {
    /// Decide which layers to swap for the given `ratio_max` (ENG-ALG-215).
    ///
    /// Returns a `SwapDecision` containing the layer indices, the computed
    /// QCF_swap estimate, and whether the uniform fallback was used.
    pub fn decide(&self, ratio_max: f32) -> SwapDecision {
        let ratio = ratio_max.clamp(0.0, 1.0);

        if ratio == 0.0 || self.n_decoder_layers == 0 {
            return SwapDecision {
                selected_layers: Vec::new(),
                qcf_swap_estimate: 0.0,
                fallback_used: false,
            };
        }

        let n = self.n_decoder_layers;
        let target_count = (ratio * n as f32).floor() as usize;
        let already_swapped_set: HashSet<usize> = self.currently_swapped.iter().copied().collect();
        let needed = target_count.saturating_sub(already_swapped_set.len());

        if needed == 0 {
            return SwapDecision {
                selected_layers: Vec::new(),
                qcf_swap_estimate: 0.0,
                fallback_used: false,
            };
        }

        // Protected layers: always exclude layer 0 and last decoder layer.
        let mut protected = HashSet::new();
        protected.insert(0usize);
        if n > 1 {
            protected.insert(n - 1);
        }

        // Build candidate list: exclude protected, already swapped, NaN-ε layers.
        let candidates: Vec<usize> = (0..n)
            .filter(|i| !protected.contains(i))
            .filter(|i| !already_swapped_set.contains(i))
            .filter(|i| {
                // INV-127: exclude layers with NaN ε when the table is present.
                match self.noise {
                    Some(t) => t.epsilon(*i).is_some(),
                    None => true, // absent → treat as ε=1.0, include all
                }
            })
            .collect();

        // Check whether we have valid importance+noise for the scored path.
        let use_fallback = self.importance.map(|t| t.is_empty()).unwrap_or(true)
            || self.noise.map(|t| !t.is_computed()).unwrap_or(true);

        let selected = if use_fallback {
            // Uniform fallback: evenly spaced across candidates (ENG-ALG-213 absorbed).
            uniform_select_by_index(needed, &candidates)
        } else {
            let imp = self.importance.expect("importance checked non-empty");
            let noise = self.noise.expect("noise checked is_computed");

            // Key = importance(i, SubLayer::Full) × ε(i).
            // When importance has no entry for a layer, treat as 0.0.
            let mut scored: Vec<(usize, f32)> = candidates
                .iter()
                .map(|&i| {
                    let imp_val = imp
                        .entries()
                        .iter()
                        .find(|e| e.layer_id == i && e.sublayer == SubLayer::Full)
                        .map(|e| e.importance)
                        .unwrap_or(0.0);
                    let eps_val = noise.epsilon(i).unwrap_or(1.0);
                    (i, imp_val * eps_val)
                })
                .collect();

            // Ascending sort: smallest key → swap first.
            // Tie-breaking: layer index ascending (ENG-ALG-215).
            scored.sort_by(|(ia, ka), (ib, kb)| {
                ka.partial_cmp(kb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(ia.cmp(ib))
            });

            scored.truncate(needed);
            scored.into_iter().map(|(i, _)| i).collect()
        };

        let qcf = compute_qcf_swap_internal(&selected, n, self.importance, self.noise);

        SwapDecision {
            selected_layers: selected,
            qcf_swap_estimate: qcf,
            fallback_used: use_fallback,
        }
    }

    /// Dry-run: compute QCF_swap at the given ratio without executing a swap.
    ///
    /// Returns `(selected_layers, qcf_swap)`.  Used for `LayerSwapEstimate`
    /// curve sampling in `QcfEstimate` (ENG-ALG-218).
    pub fn decide_dry_run(&self, ratio_max: f32) -> (Vec<usize>, f32) {
        let decision = self.decide(ratio_max);
        (decision.selected_layers, decision.qcf_swap_estimate)
    }
}

// ── QCF_swap computation (ENG-ALG-217) ───────────────────────────────────────

/// Compute QCF_swap for a given set of swapped layers (ENG-ALG-217).
///
/// ```text
/// QCF_swap(S) = Σ_{i ∈ S} importance_i × ε_i
///               ───────────────────────────────
///               Σ_{j ∈ all_valid} importance_j × ε_j
/// ```
///
/// - Layers with NaN ε are excluded from both numerator and denominator.
/// - Missing importance entries (table absent) default to `1.0`.
/// - Returns `0.0` when `swap_set` is empty or denominator ≈ 0.
pub fn compute_qcf_swap(
    swap_set: &[usize],
    noise: &QuantNoiseTable,
    importance: Option<&ImportanceTable>,
    n_decoder_layers: usize,
) -> f32 {
    compute_qcf_swap_internal(swap_set, n_decoder_layers, importance, Some(noise))
}

/// Internal implementation used by both the public function and the decider.
fn compute_qcf_swap_internal(
    swap_set: &[usize],
    n_decoder_layers: usize,
    importance: Option<&ImportanceTable>,
    noise: Option<&QuantNoiseTable>,
) -> f32 {
    if swap_set.is_empty() {
        return 0.0;
    }

    let imp_for = |i: usize| -> f32 {
        importance
            .and_then(|t| {
                t.entries()
                    .iter()
                    .find(|e| e.layer_id == i && e.sublayer == SubLayer::Full)
                    .map(|e| e.importance)
            })
            .unwrap_or(1.0)
    };

    let eps_for = |i: usize| -> Option<f32> {
        match noise {
            Some(t) => t.epsilon(i), // None if NaN or out-of-range
            None => Some(1.0),       // absent → treat as 1.0
        }
    };

    let swap_set_hash: HashSet<usize> = swap_set.iter().copied().collect();

    let numerator: f32 = (0..n_decoder_layers)
        .filter(|i| swap_set_hash.contains(i))
        .filter_map(|i| eps_for(i).map(|e| imp_for(i) * e))
        .sum();

    let denominator: f32 = (0..n_decoder_layers)
        .filter_map(|i| eps_for(i).map(|e| imp_for(i) * e))
        .sum();

    if denominator < 1e-8 {
        return 0.0;
    }

    (numerator / denominator).clamp(0.0, 1.0)
}

// ── Uniform fallback helper ───────────────────────────────────────────────────

/// Uniformly spaced index selection from a candidate slice (ENG-ALG-213 fallback).
///
/// `needed` layers are chosen at evenly-spaced positions within `candidates`.
/// Deterministic — no random seed.
fn uniform_select_by_index(needed: usize, candidates: &[usize]) -> Vec<usize> {
    if needed == 0 || candidates.is_empty() {
        return Vec::new();
    }
    if needed >= candidates.len() {
        return candidates.to_vec();
    }
    let stride = candidates.len() as f32 / needed as f32;
    let mut out = Vec::with_capacity(needed);
    for k in 0..needed {
        let idx = (k as f32 * stride).floor() as usize;
        out.push(candidates[idx.min(candidates.len() - 1)]);
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::qcf::layer_importance::{ImportanceEntry, SubLayer};

    // ── Helper builders ──

    fn make_importance(entries: Vec<(usize, f32)>) -> ImportanceTable {
        let entries = entries
            .into_iter()
            .map(|(id, imp)| ImportanceEntry {
                layer_id: id,
                sublayer: SubLayer::Full,
                importance: imp,
                opr: 0.0,
            })
            .collect();
        ImportanceTable::from_entries(entries)
    }

    fn make_noise(vals: Vec<f32>) -> QuantNoiseTable {
        // Uses the #[cfg(test)] constructor from noise_table.rs with computed_at_init=true.
        QuantNoiseTable::new_test(vals)
    }

    // ── Normal-path test (spec example) ──────────────────────────────────────

    /// 4-layer fixture from the spec:
    /// importance = [0.1, 0.5, 0.3, 0.7], ε = [0.2, 0.1, 0.3, 0.05]
    /// key = importance × ε = [0.02, 0.05, 0.09, 0.035]
    /// Layers 0 and 3 are protected; candidates = [1, 2].
    /// ratio=0.5 → k = floor(0.5 × 4) = 2 → need 2 layers from [1, 2].
    /// Sort ascending by key: layer 1 (0.05) < layer 2 (0.09) → selected = [1, 2].
    /// qcf_swap = (0.05 + 0.09) / (0.02 + 0.05 + 0.09 + 0.035) = 0.14 / 0.195
    #[test]
    fn decide_normal_path_spec_example() {
        let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);

        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[],
        };

        let decision = decider.decide(0.5);

        assert!(!decision.fallback_used, "should use scored path");
        assert_eq!(decision.selected_layers.len(), 2);
        // Both candidates (layers 1 and 2) should be selected
        assert!(decision.selected_layers.contains(&1));
        assert!(decision.selected_layers.contains(&2));

        // qcf_swap = 0.14 / 0.195
        let expected_qcf = 0.14f32 / 0.195f32;
        assert!(
            (decision.qcf_swap_estimate - expected_qcf).abs() < 1e-4,
            "qcf={:.6}, expected={:.6}",
            decision.qcf_swap_estimate,
            expected_qcf
        );
    }

    /// NaN ε layers must be excluded from candidates (INV-127).
    #[test]
    fn nan_epsilon_excluded_inv_127() {
        let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
        let noise_vals = vec![0.2, f32::NAN, 0.3, 0.05];
        let noise = QuantNoiseTable::new_test(noise_vals.clone());

        // Layer 1 has NaN ε → must be excluded even though layer 0 and 3 are the protected ones
        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[],
        };

        // ratio=0.5 → k=2, candidates are [1, 2] normally. With layer 1 NaN excluded → [2] only.
        let decision = decider.decide(0.5);
        assert!(
            !decision.selected_layers.contains(&1),
            "layer 1 with NaN epsilon must be excluded"
        );
        // Layer 2 should be selected (only valid candidate)
        assert!(decision.selected_layers.contains(&2));

        // Noise needs to be mutable for this test; keep for later
        let _ = noise_vals;
    }

    /// When ImportanceTable is empty, uniform fallback is used.
    #[test]
    fn fallback_when_importance_empty() {
        let empty_importance = ImportanceTable::from_entries(vec![]);
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);

        let decider = WeightSwapDecider {
            importance: Some(&empty_importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[],
        };

        let decision = decider.decide(0.5);
        assert!(
            decision.fallback_used,
            "should use fallback when importance empty"
        );
        // Uniform fallback still selects some layers
        assert!(!decision.selected_layers.is_empty());
    }

    /// ratio_max = 0.0 → empty decision, qcf_swap = 0.0.
    #[test]
    fn ratio_zero_returns_empty() {
        let importance = make_importance(vec![(0, 0.5), (1, 0.3), (2, 0.4), (3, 0.6)]);
        let noise = make_noise(vec![0.1, 0.2, 0.3, 0.1]);

        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[],
        };

        let decision = decider.decide(0.0);
        assert!(decision.selected_layers.is_empty());
        assert_eq!(decision.qcf_swap_estimate, 0.0);
    }

    /// `currently_swapped` layers must not be re-selected.
    #[test]
    fn currently_swapped_excluded() {
        let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);

        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[2], // layer 2 already swapped
        };

        let decision = decider.decide(0.5);
        assert!(
            !decision.selected_layers.contains(&2),
            "already-swapped layer 2 must not be re-selected"
        );
    }

    /// Layer 0 and last layer must never be selected regardless of importance.
    #[test]
    fn protected_layers_never_selected() {
        // Make layer 0 and 3 look very cheap (low key) to verify they are still excluded
        let importance = make_importance(vec![(0, 0.001), (1, 0.9), (2, 0.9), (3, 0.001)]);
        let noise = make_noise(vec![0.001, 0.9, 0.9, 0.001]);

        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[],
        };

        let decision = decider.decide(0.9);
        assert!(
            !decision.selected_layers.contains(&0),
            "layer 0 is protected"
        );
        assert!(
            !decision.selected_layers.contains(&3),
            "last decoder layer is protected"
        );
    }

    // ── compute_qcf_swap tests (ENG-ALG-217) ─────────────────────────────────

    /// Empty swap set → QCF_swap = 0.0.
    #[test]
    fn qcf_swap_empty_set_is_zero() {
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);
        let result = compute_qcf_swap(&[], &noise, None, 4);
        assert_eq!(result, 0.0);
    }

    /// All-layers swap set (excluding NaN) → QCF_swap ≈ 1.0.
    #[test]
    fn qcf_swap_full_set_approx_one() {
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);
        // All layers in the "valid" set (no NaN)
        let result = compute_qcf_swap(&[0, 1, 2, 3], &noise, None, 4);
        assert!(
            (result - 1.0).abs() < 1e-6,
            "full set should give QCF_swap ≈ 1.0, got {result}"
        );
    }

    /// Monotonic property: adding a layer to the set must not decrease QCF_swap.
    #[test]
    fn qcf_swap_monotonic() {
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);
        let q1 = compute_qcf_swap(&[1], &noise, None, 4);
        let q2 = compute_qcf_swap(&[1, 2], &noise, None, 4);
        let q3 = compute_qcf_swap(&[0, 1, 2], &noise, None, 4);
        assert!(q1 <= q2, "monotonic: q({{1}})={q1} <= q({{1,2}})={q2}");
        assert!(q2 <= q3, "monotonic: q({{1,2}})={q2} <= q({{0,1,2}})={q3}");
    }

    /// NaN ε layer contributes 0 to both numerator and denominator.
    #[test]
    fn qcf_swap_nan_layer_excluded_from_both() {
        let noise = QuantNoiseTable::new_test(vec![0.2, f32::NAN, 0.3, 0.05]);
        // Layer 1 has NaN ε — should contribute 0 to numerator and denominator.
        // Including it in swap_set should not change result vs. excluding it.
        let without_nan = compute_qcf_swap(&[0, 2, 3], &noise, None, 4);
        let with_nan = compute_qcf_swap(&[0, 1, 2, 3], &noise, None, 4);
        assert!(
            (without_nan - with_nan).abs() < 1e-6,
            "NaN layer should not affect QCF_swap: without={without_nan}, with={with_nan}"
        );
    }

    // ── dry-run test ──────────────────────────────────────────────────────────

    #[test]
    fn decide_dry_run_matches_decide() {
        let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
        let noise = make_noise(vec![0.2, 0.1, 0.3, 0.05]);
        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: 4,
            currently_swapped: &[],
        };
        let (layers_dr, qcf_dr) = decider.decide_dry_run(0.5);
        let decision = decider.decide(0.5);
        assert_eq!(layers_dr, decision.selected_layers);
        assert!((qcf_dr - decision.qcf_swap_estimate).abs() < 1e-8);
    }
}
