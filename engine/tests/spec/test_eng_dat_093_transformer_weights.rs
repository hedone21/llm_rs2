//! ENG-DAT-093 — `TransformerWeights` container shape.
//!
//! Phase 1 scope: verify the root container holds `Vec<LayerSlot>`, optional
//! `Arc<SecondaryMmap>`, `Arc<AtomicU64>` for `ratio_generation`, and
//! correctly exposes embedding / final_norm / lm_head as non-swappable
//! `Arc<Tensor>` handles (ENG-DAT-C11).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::core::backend::Backend;
use llm_rs2::core::buffer::DType;
use llm_rs2::core::memory::Memory;
use llm_rs2::core::shape::Shape;
use llm_rs2::core::tensor::Tensor;
use llm_rs2::layers::transformer_layer::TransformerLayer;
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::models::weights::{LayerSlot, TransformerWeights};

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

#[test]
fn transformer_weights_basic_shape() {
    let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let slots: Vec<LayerSlot> = (0..4)
        .map(|_| LayerSlot::new(dummy_layer(&be), DType::F16, None))
        .collect();
    let embedding = Arc::new(dummy_tensor(&be, 128));
    let final_norm = Arc::new(dummy_tensor(&be, 16));
    let lm_head = Some(Arc::new(dummy_tensor(&be, 128)));

    let tw = TransformerWeights::new(slots, embedding.clone(), final_norm.clone(), lm_head, None);

    assert_eq!(tw.num_layers(), 4);
    assert!(tw.secondary_mmap.is_none());
    // ratio_generation is declared but not bumped in Phase 1.
    assert_eq!(tw.ratio_generation(), 0);
    assert_eq!(tw.ratio_generation.load(Ordering::Acquire), 0);
    // Cross-layer tensors come back as the same Arcs we installed.
    assert!(Arc::ptr_eq(&tw.embedding, &embedding));
    assert!(Arc::ptr_eq(&tw.final_norm, &final_norm));
}

#[test]
fn transformer_weights_supports_tied_lm_head() {
    // When `tie_word_embeddings == true` (Qwen variants) the model keeps
    // `lm_head` as None and forward code reuses `embedding` for the output
    // projection. The data container must accept that shape.
    let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let slots = vec![LayerSlot::new(dummy_layer(&be), DType::F16, None)];
    let embedding = Arc::new(dummy_tensor(&be, 64));
    let final_norm = Arc::new(dummy_tensor(&be, 8));

    let tw = TransformerWeights::new(slots, embedding, final_norm, None, None);
    assert!(tw.lm_head.is_none());
    assert_eq!(tw.num_layers(), 1);
}
