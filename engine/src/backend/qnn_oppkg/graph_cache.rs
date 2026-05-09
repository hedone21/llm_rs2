//! Per-layer graph cache (M3.1 stub).
//!
//! Spec: `spec/30-engine.md` 부록 C.2 (ENG-QNN-203/INV-167), 부록 C.5
//! (ENG-QNN-209), `arch/30-engine.md` §18.4.
//!
//! M3.1: 빈 `Vec<Arc<LayerGraph>>` slot만 마련 (eager prebuild 결정 D1은
//! M3.2에서 본격 발동, 본 단계는 model load 흐름 진입조차 하지 않는다).

use crate::backend::qnn_oppkg::layer_graph::LayerGraph;
use std::sync::Arc;

/// 28 layer × `LayerGraph` cache (Qwen2.5-1.5B).
///
/// M3.2에서 model load 시점에 `prebuild()` 1회 호출로 28× `graphFinalize`
/// 직렬 실행 후 process lifetime 동안 재사용된다 (INV-167).
pub struct GraphCache {
    layers: Vec<Arc<LayerGraph>>,
}

impl GraphCache {
    /// Empty cache (M3.1 단계). M3.2에서 `prebuild(n_layers, ...)` 추가.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Cache에 등록된 layer graph 개수. eager prebuild 후 `n_layers`와 동일
    /// 해야 한다 (ENG-QNN-203).
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Cache가 비어있는지.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Layer graph 1개에 대한 read-only 핸들. M3.3 dispatch path에서 사용.
    pub fn get(&self, layer_idx: usize) -> Option<&Arc<LayerGraph>> {
        self.layers.get(layer_idx)
    }
}

impl Default for GraphCache {
    fn default() -> Self {
        Self::new()
    }
}
