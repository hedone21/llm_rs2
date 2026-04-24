//! INV-124 — `LayerSlot::current_dtype` consistent with the loaded snapshot.
//!
//! Phase 1 enforces the static case: construction and `store_weights_same_dtype`
//! preserve dtype coherence; `swap_weights` installs the new dtype and the
//! new Arc within the same logical step. Phase 2 will add the concurrent
//! writer/reader coverage once `SwapExecutor` exists.

use std::sync::Arc;

use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::core::backend::Backend;
use llm_rs2::core::buffer::DType;
use llm_rs2::core::memory::Memory;
use llm_rs2::core::shape::Shape;
use llm_rs2::core::tensor::Tensor;
use llm_rs2::layers::transformer_layer::TransformerLayer;
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::models::weights::LayerSlot;

fn dummy_tensor(be: &Arc<dyn Backend>, dtype: DType) -> Tensor {
    let mem = Galloc::new();
    let numel = 16usize;
    let byte_size = match dtype {
        DType::F32 => numel * 4,
        DType::F16 | DType::BF16 => numel * 2,
        DType::Q4_0 => numel * 18 / 32, // block encoding
        DType::U8 => numel,
        _ => numel * 4,
    };
    let buf = mem.alloc(byte_size.max(4), dtype).unwrap();
    Tensor::new(Shape::new(vec![numel]), buf, be.clone())
}

fn dummy_layer(be: &Arc<dyn Backend>, dtype: DType) -> TransformerLayer {
    TransformerLayer {
        wq: dummy_tensor(be, dtype),
        wk: dummy_tensor(be, dtype),
        wv: dummy_tensor(be, dtype),
        wo: dummy_tensor(be, dtype),
        w_gate: dummy_tensor(be, dtype),
        w_up: dummy_tensor(be, dtype),
        w_down: dummy_tensor(be, dtype),
        attention_norm: dummy_tensor(be, DType::F32),
        ffn_norm: dummy_tensor(be, DType::F32),
        qkv_bias: None,
        q_norm: None,
        k_norm: None,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        partition_ctx: None,
    }
}

#[test]
fn initial_slot_dtype_matches_weights() {
    let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let layer = dummy_layer(&be, DType::F16);
    let slot = LayerSlot::new(layer, DType::F16, None);

    let loaded = slot.load_weights();
    assert_eq!(slot.current_dtype(), DType::F16);
    // Every weight tensor installed as F16 must report F16.
    assert_eq!(loaded.wq.dtype(), DType::F16);
    assert_eq!(loaded.wk.dtype(), DType::F16);
    assert_eq!(loaded.w_gate.dtype(), DType::F16);
}

#[test]
fn swap_installs_new_dtype_atomically() {
    let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let slot = LayerSlot::new(dummy_layer(&be, DType::F16), DType::F16, None);

    let new_layer = dummy_layer(&be, DType::Q4_0);
    let new_arc = Arc::new(new_layer);
    slot.swap_weights(new_arc.clone(), DType::Q4_0);

    // After swap, both the dtype atom and the weight tensor must reflect Q4_0.
    let loaded = slot.load_weights();
    assert_eq!(slot.current_dtype(), DType::Q4_0);
    assert_eq!(loaded.wq.dtype(), DType::Q4_0);
    assert_eq!(loaded.w_down.dtype(), DType::Q4_0);
}

#[test]
fn same_dtype_rewrap_preserves_consistency() {
    let be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let slot = LayerSlot::new(dummy_layer(&be, DType::F16), DType::F16, None);

    let rewrapped = Arc::new(dummy_layer(&be, DType::F16));
    slot.store_weights_same_dtype(rewrapped.clone());

    assert_eq!(slot.current_dtype(), DType::F16);
    assert_eq!(slot.load_weights().wq.dtype(), DType::F16);
}
