//! Phase 3 E2E smoke test — `dispatch_swap_weights` end-to-end path.
//!
//! Validates the full dispatch pipeline from manager command to report payload:
//!   `EngineCommand::SwapWeights` → validation → `WeightSwapDecider::decide()`
//!   → `WeightSwapHandler::execute_swap()` → `WeightSwapReport`
//!
//! This test operates at the unit level (no real GGUF file) with synthetic
//! fixtures for `ImportanceTable`, `QuantNoiseTable`, and `WeightSwapHandler`.
//! Manager ↔ Engine IPC integration is deferred to Phase 4.
//!
//! Spec: ENG-ALG-214-ROUTE, ENG-ALG-215, ENG-ALG-217, ENG-ALG-218,
//!       MSG-042, MSG-089, INV-126, INV-127, INV-128.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::core::backend::Backend;
use llm_rs2::core::buffer::DType;
use llm_rs2::core::memory::Memory;
use llm_rs2::core::pressure::ActionResult;
use llm_rs2::core::pressure::weight_swap_handler::{WeightSwapHandler, WeightSwapModelRef};
use llm_rs2::core::qcf::layer_importance::{ImportanceEntry, ImportanceTable, SubLayer};
use llm_rs2::core::shape::Shape;
use llm_rs2::core::tensor::Tensor;
use llm_rs2::layers::transformer_layer::TransformerLayer;
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::models::config::{ModelArch, ModelConfig};
use llm_rs2::models::weights::{LayerSlot, QuantNoiseTable, WeightSwapDecider, compute_qcf_swap};
use llm_shared::DtypeTag;

// ── Fixtures ──────────────────────────────────────────────────────────────────

fn cpu_be() -> Arc<dyn Backend> {
    Arc::new(CpuBackend::new())
}

fn dummy_tensor(be: &Arc<dyn Backend>, numel: usize) -> Tensor {
    let mem = Galloc::new();
    let buf = mem.alloc(numel * 4, DType::F32).unwrap();
    Tensor::new(Shape::new(vec![numel]), buf, be.clone())
}

fn dummy_layer(be: &Arc<dyn Backend>) -> TransformerLayer {
    TransformerLayer {
        wq: dummy_tensor(be, 16),
        wk: dummy_tensor(be, 16),
        wv: dummy_tensor(be, 16),
        wo: dummy_tensor(be, 16),
        w_gate: dummy_tensor(be, 16),
        w_up: dummy_tensor(be, 16),
        w_down: dummy_tensor(be, 16),
        attention_norm: dummy_tensor(be, 4),
        ffn_norm: dummy_tensor(be, 4),
        qkv_bias: None,
        q_norm: None,
        k_norm: None,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        partition_ctx: None,
    }
}

fn minimal_config(n_layers: usize) -> ModelConfig {
    ModelConfig {
        arch: ModelArch::Llama,
        hidden_size: 64,
        num_hidden_layers: n_layers,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        head_dim: 16,
        intermediate_size: 128,
        vocab_size: 256,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        has_qkv_bias: false,
        tie_word_embeddings: false,
        eos_token_id: 1,
        rope_local_theta: None,
        sliding_window: None,
        sliding_window_pattern: None,
        query_pre_attn_scalar: None,
        embed_scale: None,
        weight_prefix: String::new(),
    }
}

/// Build a 4-layer `WeightSwapModelRef` with no secondary mmap.
fn make_model_ref(n_layers: usize) -> Arc<WeightSwapModelRef> {
    let be = cpu_be();
    let layers: Vec<LayerSlot> = (0..n_layers)
        .map(|_| LayerSlot::new(dummy_layer(&be), DType::F16, None))
        .collect();
    Arc::new(WeightSwapModelRef {
        layers: Arc::new(layers),
        secondary_mmap: None,
        ratio_generation: Arc::new(AtomicU64::new(0)),
        config: minimal_config(n_layers),
        backend: be,
    })
}

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

// ── E2E: DtypeTag::Q4_0 validation gate ──────────────────────────────────────

/// E2E gate: Q4_0 must pass DtypeTag validation (INV-126 positive path).
/// Without secondary mmap, dispatch proceeds to handler which returns NoOp.
/// This confirms the validation layer doesn't erroneously reject Q4_0.
#[test]
fn e2e_q4_0_dtype_passes_validation_gate() {
    use llm_rs2::models::weights::swap_executor::dtype_tag_to_dtype;
    let result = dtype_tag_to_dtype(DtypeTag::Q4_0);
    assert!(
        result.is_ok(),
        "Q4_0 must pass the DtypeTag gate in dispatch_swap_weights"
    );
}

/// E2E gate: non-Q4_0 dtypes must fail at the validation gate.
#[test]
fn e2e_reserved_dtype_blocked_at_gate() {
    use llm_rs2::models::weights::swap_executor::dtype_tag_to_dtype;
    for tag in [DtypeTag::F16, DtypeTag::F32, DtypeTag::Q8_0] {
        assert!(
            dtype_tag_to_dtype(tag).is_err(),
            "dtype {tag:?} must be blocked at validation gate"
        );
    }
}

// ── E2E: WeightSwapDecider → selected_layers → handler dispatch ──────────────

/// Full pipeline: importance + noise → decider → execute_swap.
///
/// Fixture: 4-layer model, ratio=0.5.
/// importance = [0.1, 0.5, 0.3, 0.7], ε = [0.2, 0.1, 0.3, 0.05]
/// Protected: layers 0, 3.
/// Candidates: [1, 2].
/// Selected (ascending importance × ε): layer 1 (0.05), layer 2 (0.09).
///
/// Without secondary mmap, handler returns NoOp — confirms pipeline reaches
/// handler without earlier assertion failure.
#[test]
fn e2e_decider_to_handler_no_secondary() {
    let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
    let noise = QuantNoiseTable::from_values(vec![0.2, 0.1, 0.3, 0.05]);

    let decider = WeightSwapDecider {
        importance: Some(&importance),
        noise: Some(&noise),
        n_decoder_layers: 4,
        currently_swapped: &[],
    };

    let decision = decider.decide(0.5);
    assert!(!decision.fallback_used, "should use scored path");
    assert_eq!(decision.selected_layers.len(), 2);

    // Pass to handler (no secondary → NoOp)
    let model_ref = make_model_ref(4);
    let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
    let handler = WeightSwapHandler::new(model_ref, mem, DType::Q4_0);
    let result = handler.execute_swap(&decision.selected_layers).unwrap();
    assert!(
        matches!(result, ActionResult::NoOp),
        "without secondary mmap, handler must return NoOp"
    );
}

// ── E2E: SwapDecision fields contract (MSG-089 WeightSwapReport precursor) ──

/// `SwapDecision` fields satisfy the MSG-089 WeightSwapReport preconditions.
///
/// - `ratio_applied` ≤ requested ratio
/// - `qcf_swap_estimate` in [0, 1]
/// - selected layers do not include protected ones (0 or last)
#[test]
fn e2e_swap_decision_fields_satisfy_msg_089_preconditions() {
    let n_layers = 4usize;
    let requested_ratio = 0.5f32;

    let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
    let noise = QuantNoiseTable::from_values(vec![0.2, 0.1, 0.3, 0.05]);

    let decider = WeightSwapDecider {
        importance: Some(&importance),
        noise: Some(&noise),
        n_decoder_layers: n_layers,
        currently_swapped: &[],
    };

    let decision = decider.decide(requested_ratio);

    // ratio_applied = selected / n_layers
    let ratio_applied = decision.selected_layers.len() as f32 / n_layers as f32;
    assert!(
        ratio_applied <= requested_ratio + 1e-6,
        "ratio_applied {ratio_applied:.4} must be ≤ requested {requested_ratio:.4}"
    );

    // qcf_swap_estimate in [0, 1]
    assert!(
        decision.qcf_swap_estimate >= 0.0 && decision.qcf_swap_estimate <= 1.0,
        "qcf_swap_estimate {:.6} must be in [0, 1]",
        decision.qcf_swap_estimate
    );

    // Protected layers excluded
    assert!(
        !decision.selected_layers.contains(&0),
        "layer 0 must be protected"
    );
    assert!(
        !decision.selected_layers.contains(&(n_layers - 1)),
        "last layer must be protected"
    );

    // selected layers have smaller importance × ε than excluded non-protected
    // (layer 1 key=0.05 and layer 2 key=0.09 selected; there are no others)
    assert!(decision.selected_layers.contains(&1) || decision.selected_layers.contains(&2));
}

// ── E2E: QcfEstimate.layer_swap precursor — dry-run curve ────────────────────

/// `LayerSwapEstimate.qcf_swap_at_ratio` curve must be monotonically
/// non-decreasing across sampled ratios (ENG-ALG-218).
#[test]
fn e2e_qcf_swap_at_ratio_curve_is_monotone() {
    let n_layers = 4usize;
    let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
    let noise = QuantNoiseTable::from_values(vec![0.2, 0.1, 0.3, 0.05]);

    let sample_ratios = [0.1f32, 0.25, 0.5, 0.75, 1.0];
    let mut prev_qcf = -1.0f32;
    for &r in &sample_ratios {
        let decider = WeightSwapDecider {
            importance: Some(&importance),
            noise: Some(&noise),
            n_decoder_layers: n_layers,
            currently_swapped: &[],
        };
        let (_, qcf) = decider.decide_dry_run(r);
        assert!(
            qcf.is_finite(),
            "qcf at ratio {r:.2} must be finite, got {qcf}"
        );
        assert!(
            qcf >= prev_qcf - 1e-6,
            "curve must be non-decreasing: qcf({r:.2})={qcf:.6} < prev={prev_qcf:.6}"
        );
        prev_qcf = qcf;
    }
}

/// Dry-run at ratio=0 must produce empty layers and qcf_swap=0.
#[test]
fn e2e_dry_run_zero_ratio_is_empty() {
    let noise = QuantNoiseTable::from_values(vec![0.2, 0.1, 0.3, 0.05]);
    let decider = WeightSwapDecider {
        importance: None,
        noise: Some(&noise),
        n_decoder_layers: 4,
        currently_swapped: &[],
    };
    let (layers, qcf) = decider.decide_dry_run(0.0);
    assert!(layers.is_empty(), "ratio=0 must give no layers");
    assert_eq!(qcf, 0.0, "ratio=0 must give qcf=0.0");
}

// ── E2E: QCF_swap actual matches compute_qcf_swap ───────────────────────────

/// `compute_qcf_swap` for the actually swapped layers must equal the
/// `qcf_swap_estimate` from the decider (same inputs, same algorithm).
#[test]
fn e2e_qcf_swap_actual_matches_estimate() {
    let n_layers = 4usize;
    let importance = make_importance(vec![(0, 0.1), (1, 0.5), (2, 0.3), (3, 0.7)]);
    let noise = QuantNoiseTable::from_values(vec![0.2, 0.1, 0.3, 0.05]);

    let decider = WeightSwapDecider {
        importance: Some(&importance),
        noise: Some(&noise),
        n_decoder_layers: n_layers,
        currently_swapped: &[],
    };

    let decision = decider.decide(0.5);
    // Recompute qcf_swap using the public compute_qcf_swap function
    let qcf_actual = compute_qcf_swap(
        &decision.selected_layers,
        &noise,
        Some(&importance),
        n_layers,
    );

    assert!(
        (qcf_actual - decision.qcf_swap_estimate).abs() < 1e-5,
        "compute_qcf_swap actual={qcf_actual:.6} must match decider estimate={:.6}",
        decision.qcf_swap_estimate
    );
}

// ── E2E: importance × ε ordering — selected layers have lower cost ────────────

/// Layers selected by the decider must have lower importance × ε than
/// excluded non-protected layers (ENG-ALG-215 ascending sort contract).
#[test]
fn e2e_selected_layers_have_lower_cost_than_excluded() {
    let n_layers = 6usize;
    // Layer costs (importance × ε):
    //   0: 0.1 * 0.5 = 0.05  [protected]
    //   1: 0.3 * 0.4 = 0.12  → candidate
    //   2: 0.5 * 0.1 = 0.05  → candidate (cheapest non-protected)
    //   3: 0.7 * 0.3 = 0.21  → candidate
    //   4: 0.2 * 0.2 = 0.04  → candidate (cheapest overall)
    //   5: 0.8 * 0.6 = 0.48  [protected]
    let importance = make_importance(vec![
        (0, 0.1),
        (1, 0.3),
        (2, 0.5),
        (3, 0.7),
        (4, 0.2),
        (5, 0.8),
    ]);
    let noise = QuantNoiseTable::from_values(vec![0.5, 0.4, 0.1, 0.3, 0.2, 0.6]);

    // ratio=0.5 → k=floor(0.5*6)=3, protected=[0,5], candidates=[1,2,3,4]
    // costs:  1→0.12, 2→0.05, 3→0.21, 4→0.04
    // ascending: 4(0.04) < 2(0.05) < 1(0.12) < 3(0.21)
    // select 3: [4, 2, 1]
    let decider = WeightSwapDecider {
        importance: Some(&importance),
        noise: Some(&noise),
        n_decoder_layers: n_layers,
        currently_swapped: &[],
    };

    let decision = decider.decide(0.5);
    assert_eq!(
        decision.selected_layers.len(),
        3,
        "should select 3 layers for ratio=0.5, 6 total"
    );

    // All selected must be cheaper than excluded non-protected layers
    let selected_max_cost = decision
        .selected_layers
        .iter()
        .filter_map(|&i| {
            let imp = [0.1f32, 0.3, 0.5, 0.7, 0.2, 0.8][i];
            let eps = [0.5f32, 0.4, 0.1, 0.3, 0.2, 0.6][i];
            if imp.is_finite() && eps.is_finite() {
                Some(imp * eps)
            } else {
                None
            }
        })
        .fold(f32::NEG_INFINITY, f32::max);

    // The excluded non-protected layer should be layer 3 (cost 0.21)
    let excluded_non_protected = [1usize, 2, 3, 4]
        .iter()
        .filter(|&&i| !decision.selected_layers.contains(&i))
        .copied()
        .collect::<Vec<_>>();

    for &excl in &excluded_non_protected {
        let imp = [0.1f32, 0.3, 0.5, 0.7, 0.2, 0.8][excl];
        let eps = [0.5f32, 0.4, 0.1, 0.3, 0.2, 0.6][excl];
        let excl_cost = imp * eps;
        assert!(
            selected_max_cost <= excl_cost + 1e-6,
            "selected max_cost={selected_max_cost:.4} must be ≤ excluded cost={excl_cost:.4} (layer {excl})"
        );
    }
}

// ── E2E: SwapDecision with per_layer count matches n_decoder_layers ───────────

/// `SwapDecision` candidate counts must be consistent with `n_decoder_layers`.
#[test]
fn e2e_per_layer_count_consistent() {
    let n_layers = 8usize;
    let noise = QuantNoiseTable::from_values(vec![0.1; n_layers]);
    let importance = make_importance((0..n_layers).map(|i| (i, 0.5f32)).collect());

    let decider = WeightSwapDecider {
        importance: Some(&importance),
        noise: Some(&noise),
        n_decoder_layers: n_layers,
        currently_swapped: &[],
    };

    let decision = decider.decide(0.5);

    // All selected indices must be in [0, n_layers)
    for &idx in &decision.selected_layers {
        assert!(
            idx < n_layers,
            "selected layer index {idx} must be < n_decoder_layers={n_layers}"
        );
    }

    // Count must not exceed k = floor(ratio × n)
    let max_count = (0.5f32 * n_layers as f32).floor() as usize;
    assert!(
        decision.selected_layers.len() <= max_count,
        "selected count {} must be ≤ k={max_count}",
        decision.selected_layers.len()
    );
}
