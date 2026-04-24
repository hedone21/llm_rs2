//! ENG-ALG-211 Stage 2 — `WeightSwapHandler` pipeline integration.
//!
//! Tests that `WeightSwapHandler` implements `CachePressureHandler` correctly:
//! - No-op at Normal / no-secondary-mmap conditions (ENG-DAT-C09).
//! - Pressure-level → ratio mapping (Stage 2 placeholder table).
//! - Handler name is "weight_swap".
//! - `ActionResult::WeightSwapped` is an action (not NoOp).
//! - `SwapExecutor::uniform_target_layers` contract (ENG-ALG-212/213).
//! - Pipeline ordering: WeightSwapHandler can be composed with other handlers.
//!
//! Full execute() path (layer slot + secondary mmap fixture) would require
//! real GGUF fixture files → deferred to device integration tests.
//!
//! Spec: ENG-ALG-211, ENG-ALG-212, INV-121/123, WSWAP-2-HANDLER.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::buffer::shared_buffer::SharedBuffer;
use llm_rs2::core::backend::Backend;
use llm_rs2::core::buffer::DType;
use llm_rs2::core::kv_cache::KVCache;
use llm_rs2::core::memory::Memory;
use llm_rs2::core::pressure::weight_swap_handler::{WeightSwapHandler, WeightSwapModelRef};
use llm_rs2::core::pressure::{
    ActionResult, CachePressureHandler, CachePressurePipeline, HandlerContext, PressureLevel,
    PressureStageConfig,
};
use llm_rs2::core::shape::Shape;
use llm_rs2::core::tensor::Tensor;
use llm_rs2::layers::transformer_layer::TransformerLayer;
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::models::config::{ModelArch, ModelConfig};
use llm_rs2::models::weights::LayerSlot;

// ── Helpers ───────────────────────────────────────────────────────────────────

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

fn minimal_model_config() -> ModelConfig {
    ModelConfig {
        arch: ModelArch::Llama,
        hidden_size: 64,
        num_hidden_layers: 4,
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

/// Build a `WeightSwapModelRef` with no secondary mmap (swap disabled).
fn make_model_ref_no_secondary(n_layers: usize) -> Arc<WeightSwapModelRef> {
    let be = cpu_be();
    let layers: Vec<LayerSlot> = (0..n_layers)
        .map(|_| LayerSlot::new(dummy_layer(&be), DType::F16, None))
        .collect();
    Arc::new(WeightSwapModelRef {
        layers: Arc::new(layers),
        secondary_mmap: None,
        ratio_generation: Arc::new(AtomicU64::new(0)),
        config: minimal_model_config(),
        backend: be,
    })
}

fn make_kv_caches(n: usize) -> Vec<KVCache> {
    let be = cpu_be();
    (0..n)
        .map(|_| {
            let buf_size = 32 * 4 * 4; // 32 positions, 1 head, dim=4, f32
            let k = Tensor::new(
                Shape::new(vec![1, 32, 1, 4]),
                Arc::new(SharedBuffer::new(buf_size, DType::F32)),
                be.clone(),
            );
            let v = Tensor::new(
                Shape::new(vec![1, 32, 1, 4]),
                Arc::new(SharedBuffer::new(buf_size, DType::F32)),
                be.clone(),
            );
            KVCache::new(k, v, 32)
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Handler name must be "weight_swap".
#[test]
fn handler_name() {
    let model_ref = make_model_ref_no_secondary(4);
    let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
    let handler = WeightSwapHandler::new(model_ref, mem, DType::Q4_0);
    assert_eq!(handler.name(), "weight_swap");
}

/// Normal pressure → NoOp (ratio = 0.0).
#[test]
fn normal_pressure_is_noop() {
    let model_ref = make_model_ref_no_secondary(4);
    let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
    let handler = WeightSwapHandler::new(model_ref, mem, DType::Q4_0);

    let mut caches = make_kv_caches(2);
    let mut ctx = HandlerContext {
        caches: &mut caches,
        importance: None,
        head_importance: None,
        n_kv_heads: 0,
        pressure_level: PressureLevel::Normal,
        mem_available: 0,
        target_ratio: None,
        qcf_sink: None,
        layer_ratios: None,
    };
    let result = handler.handle(&mut ctx).unwrap();
    assert!(
        matches!(result, ActionResult::NoOp),
        "Expected NoOp at Normal pressure, got {result:?}"
    );
}

/// No secondary mmap → NoOp regardless of pressure level.
#[test]
fn no_secondary_mmap_is_noop_at_critical() {
    let model_ref = make_model_ref_no_secondary(4);
    let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
    let handler = WeightSwapHandler::new(model_ref, mem, DType::Q4_0);

    let mut caches = make_kv_caches(1);
    let mut ctx = HandlerContext {
        caches: &mut caches,
        importance: None,
        head_importance: None,
        n_kv_heads: 0,
        pressure_level: PressureLevel::Critical,
        mem_available: 0,
        target_ratio: None,
        qcf_sink: None,
        layer_ratios: None,
    };
    let result = handler.handle(&mut ctx).unwrap();
    // ENG-DAT-C09: secondary absent → the entire swap is a no-op.
    assert!(
        matches!(result, ActionResult::NoOp),
        "Expected NoOp (no secondary), got {result:?}"
    );
}

/// Pressure-level → ratio mapping matches the Stage 2 placeholder table.
#[test]
fn pressure_ratio_mapping() {
    assert_eq!(
        WeightSwapHandler::ratio_for_level(PressureLevel::Normal),
        0.0
    );
    assert_eq!(
        WeightSwapHandler::ratio_for_level(PressureLevel::Warning),
        0.25
    );
    assert_eq!(
        WeightSwapHandler::ratio_for_level(PressureLevel::Critical),
        0.50
    );
    assert_eq!(
        WeightSwapHandler::ratio_for_level(PressureLevel::Emergency),
        0.75
    );
}

/// `WeightSwapped` `ActionResult` variant is recognized as an action.
#[test]
fn weight_swapped_is_action() {
    let result = ActionResult::WeightSwapped {
        layers_changed: 2,
        freed_bytes: 1024,
        duration_ms: 15.0,
    };
    assert!(
        result.is_action(),
        "WeightSwapped must be is_action() == true"
    );
}

/// Handler can be composed in a `CachePressurePipeline` alongside other handlers.
#[test]
fn handler_in_pipeline_composition() {
    let model_ref = make_model_ref_no_secondary(4);
    let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
    let handler = WeightSwapHandler::new(model_ref, mem, DType::Q4_0);

    // Build a pipeline with WeightSwapHandler at Emergency level (last resort).
    let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
        min_level: PressureLevel::Emergency,
        handler: Box::new(handler),
    }]);

    assert_eq!(pipeline.len(), 1);
    assert!(pipeline.name().contains("weight_swap"));

    // At Critical, the Emergency stage must not fire.
    let mut caches = make_kv_caches(1);
    let mut ctx = HandlerContext {
        caches: &mut caches,
        importance: None,
        head_importance: None,
        n_kv_heads: 0,
        pressure_level: PressureLevel::Critical,
        mem_available: 0,
        target_ratio: None,
        qcf_sink: None,
        layer_ratios: None,
    };
    let results = pipeline.execute(&mut ctx).unwrap();
    assert!(
        results.is_empty(),
        "WeightSwapHandler at Emergency must not fire at Critical pressure"
    );
}

/// `WeightSwapModelRef` with zero layers → NoOp (no target layers).
#[test]
fn zero_layer_model_is_noop() {
    let be = cpu_be();
    let model_ref = Arc::new(WeightSwapModelRef {
        layers: Arc::new(Vec::new()),
        secondary_mmap: None,
        ratio_generation: Arc::new(AtomicU64::new(0)),
        config: minimal_model_config(),
        backend: be,
    });
    let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
    let handler = WeightSwapHandler::new(model_ref, mem, DType::Q4_0);

    let mut caches = make_kv_caches(0);
    let mut ctx = HandlerContext {
        caches: &mut caches,
        importance: None,
        head_importance: None,
        n_kv_heads: 0,
        pressure_level: PressureLevel::Emergency,
        mem_available: 0,
        target_ratio: None,
        qcf_sink: None,
        layer_ratios: None,
    };
    let result = handler.handle(&mut ctx).unwrap();
    assert!(
        matches!(result, ActionResult::NoOp),
        "Zero-layer model must return NoOp"
    );
}

/// `uniform_target_layers` contract (ENG-ALG-212): idempotent, deduplicated,
/// bounded by ratio.
#[test]
fn uniform_target_layers_contract() {
    use llm_rs2::models::weights::SwapExecutor;

    assert!(SwapExecutor::uniform_target_layers(0.0, 16).is_empty());
    assert_eq!(SwapExecutor::uniform_target_layers(1.0, 16).len(), 16);
    let half = SwapExecutor::uniform_target_layers(0.5, 16);
    assert_eq!(half.len(), 8);
    // No duplicates.
    let mut sorted = half.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        half.len(),
        "uniform_target_layers must be deduped"
    );
    // All indices in range.
    assert!(half.iter().all(|&i| i < 16));
}
