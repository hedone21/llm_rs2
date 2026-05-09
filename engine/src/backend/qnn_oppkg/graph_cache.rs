//! Per-layer graph cache (M3.2).
//!
//! Spec: `spec/30-engine.md` 부록 C.2 (ENG-QNN-203/INV-167), 부록 C.5
//! (ENG-QNN-209), `arch/30-engine.md` §18.4.
//!
//! ## D1 결정 — Eager prebuild
//! Model load 시점에 N(=28)× `graphFinalize`를 직렬 실행한다 (~33s 1회성
//! spike 수용). Decode 동안 추가 finalize는 0회 (INV-167).
//!
//! ## INV-167 — graph cache lifetime = process lifetime
//! Cache invalidation은 weight swap path에서만 발동 (M4.1 영역). 본 단계의
//! `prebuild`는 모든 layer를 한 번에 채우고 이후 `get(layer_idx)`만 read-only
//! 호출.

use crate::backend::qnn_oppkg::layer_graph::{FINALIZE_BUDGET_MS, LayerConfig, LayerGraph};
use crate::backend::qnn_oppkg::runtime::QnnOppkgRuntime;
use crate::models::weights::LayerSlot;
use anyhow::{Result, ensure};
use std::sync::Arc;

/// Layer graph cache — model load 시점에 1회 채워지고 process lifetime 동안
/// 재사용된다 (INV-167).
pub struct GraphCache {
    layers: Vec<Arc<LayerGraph>>,
    finalize_total_ms: u32,
}

impl GraphCache {
    /// Empty cache. `prebuild()`로 채워야 사용 가능 (INV-167 default false 상태).
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            finalize_total_ms: 0,
        }
    }

    /// Eager prebuild — model load 시점 1회 호출. `slots`의 각 layer에 대해
    /// `LayerGraph::build`를 직렬 실행하고, 각 layer가 INV-167 (≤200 ms)을
    /// 만족하는지 검증한다.
    ///
    /// QNN context는 단일 핸들이므로 build는 직렬화 강제 (parallel build는
    /// QNN context lock 충돌 위험 — Plan §1.4 R9). 28 layer × ≤200 ms ≈ ~33 s
    /// wall-clock 추가.
    ///
    /// host build에서는 `LayerGraph::build`가 즉시 Err로 fail하여 prebuild
    /// 자체가 실패한다 (caller가 명확하게 catch + bail).
    pub fn prebuild(
        &mut self,
        runtime: &QnnOppkgRuntime,
        slots: &[Arc<LayerSlot>],
        cfg: &LayerConfig,
    ) -> Result<()> {
        ensure!(
            self.layers.is_empty(),
            "GraphCache::prebuild called twice (INV-167 violation)"
        );

        for (idx, slot) in slots.iter().enumerate() {
            let weights = slot.load_weights();
            let lg = LayerGraph::build(runtime, idx, weights.as_ref(), cfg)?;

            // INV-167 / ENG-QNN-209: graphFinalize ≤ 200 ms/layer.
            ensure!(
                lg.finalize_ms <= FINALIZE_BUDGET_MS,
                "ENG-QNN-209/INV-167: layer {idx} graphFinalize {} ms > {} ms (budget)",
                lg.finalize_ms,
                FINALIZE_BUDGET_MS
            );

            self.finalize_total_ms = self.finalize_total_ms.saturating_add(lg.finalize_ms);
            self.layers.push(Arc::new(lg));
        }

        Ok(())
    }

    /// Cache에 등록된 layer graph 개수. Eager prebuild 후 `slots.len()`과 동일
    /// 해야 한다 (ENG-QNN-203).
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Cache가 비어있는지.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// 28× graphFinalize 누적 시간 (ms). ENG-QNN-209 — 사용자 가시화 용도.
    pub fn finalize_total_ms(&self) -> u32 {
        self.finalize_total_ms
    }

    /// Layer graph 1개에 대한 read-only 핸들. M3.3 dispatch path에서 사용.
    pub fn get(&self, layer_idx: usize) -> Option<Arc<LayerGraph>> {
        self.layers.get(layer_idx).cloned()
    }
}

impl Default for GraphCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cache_initial_state() {
        let c = GraphCache::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.finalize_total_ms(), 0);
        assert!(c.get(0).is_none());
    }

    #[test]
    fn default_is_empty() {
        let c = GraphCache::default();
        assert!(c.is_empty());
    }
}
