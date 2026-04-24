//! `WeightSwapHandler` — weight-precision downgrade via `SwapExecutor`.
//!
//! Implements `CachePressureHandler` for the decoder-layer weight swap path
//! (F16 → Q4_0). Unlike the KV `SwapHandler` which offloads token data to disk,
//! this handler replaces layer weight snapshots with lower-precision copies
//! loaded from the secondary GGUF mmap.
//!
//! # Handler ordering in the pipeline
//!
//! Recommended pipeline order (least aggressive → most aggressive):
//!   1. `EvictionHandler`   — per-token KV eviction (immediate, reversible)
//!   2. `SwapHandler`       — KV disk offload (slower, potentially reversible)
//!   3. `WeightSwapHandler` — weight dtype downgrade (last resort, **one-way**)
//!
//! Weight swap is placed last because it is irreversible within a session —
//! Q4_0→F16 restoration requires an engine restart.
//!
//! # Pressure-level → ratio mapping (Phase 2 placeholder)
//!
//! Phase 3 will receive `SwapWeights { ratio }` from the manager signal;
//! until then this handler uses a fixed table:
//! - Normal    → ratio 0.0 (no-op)
//! - Warning   → ratio 0.25
//! - Critical  → ratio 0.50
//! - Emergency → ratio 0.75
//!
//! Spec: ENG-ALG-211, INV-121/123, WSWAP-2-HANDLER.

use std::sync::Arc;

use anyhow::Result;

use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::memory::Memory;
use crate::models::config::ModelConfig;
use crate::models::weights::swap_executor::SwapExecutor;
use crate::models::weights::{LayerSlot, SecondaryMmap};

use super::{ActionResult, CachePressureHandler, HandlerContext, PressureLevel};

/// Shared model references needed by `WeightSwapHandler` without requiring
/// `Arc<TransformerModel>` (the model is owned directly in `generate.rs`).
///
/// Constructed once at handler registration time and kept alive for the
/// duration of the `CachePressurePipeline`.
pub struct WeightSwapModelRef {
    /// Decoder layer slots shared via Arc. The forward path also holds
    /// these slots (via `TransformerModel::layers`), so the Arc keeps the
    /// backing data alive even after a swap until all in-flight forwards
    /// complete (INV-121/INV-123).
    pub layers: Arc<Vec<LayerSlot>>,
    /// Secondary GGUF handle. `None` disables the swap path (ENG-DAT-C09).
    pub secondary_mmap: Option<Arc<SecondaryMmap>>,
    /// Global swap generation counter shared with `TransformerModel`.
    /// Bumped by `SwapExecutor::execute_on_slots` after each batch.
    pub ratio_generation: Arc<std::sync::atomic::AtomicU64>,
    /// Model configuration (needed by `SwapExecutor` for Q/K permutation).
    pub config: ModelConfig,
    /// Backend on which fresh weight tensors should land.
    pub backend: Arc<dyn Backend>,
}

/// Adapter that connects `CachePressurePipeline` to `SwapExecutor`.
///
/// On each `handle()` call the handler:
/// 1. Maps `ctx.pressure_level` to a swap ratio.
/// 2. Computes uniform target layers via `SwapExecutor::uniform_target_layers`.
/// 3. Calls `SwapExecutor::execute_on_slots` if the ratio is > 0.
/// 4. Returns `ActionResult::WeightSwapped` or `ActionResult::NoOp`.
pub struct WeightSwapHandler {
    model_ref: Arc<WeightSwapModelRef>,
    memory: Arc<dyn Memory>,
    /// Target dtype to swap *to* (default `Q4_0`).
    target_dtype: DType,
}

impl WeightSwapHandler {
    /// Construct a new handler.
    ///
    /// `model_ref` holds the shared model state; `memory` is used by
    /// `SwapExecutor` for buffer allocation. `target_dtype` is the precision
    /// we are downgrading to (typically `Q4_0`).
    pub fn new(
        model_ref: Arc<WeightSwapModelRef>,
        memory: Arc<dyn Memory>,
        target_dtype: DType,
    ) -> Self {
        Self {
            model_ref,
            memory,
            target_dtype,
        }
    }

    /// Map a pressure level to a target swap ratio.
    ///
    /// Phase 2 placeholder — Phase 3 replaces this with the `SwapWeights`
    /// payload from the manager signal.
    pub fn ratio_for_level(level: PressureLevel) -> f32 {
        match level {
            PressureLevel::Normal => 0.0,
            PressureLevel::Warning => 0.25,
            PressureLevel::Critical => 0.50,
            PressureLevel::Emergency => 0.75,
        }
    }
}

impl CachePressureHandler for WeightSwapHandler {
    fn handle(&self, ctx: &mut HandlerContext) -> Result<ActionResult> {
        let ratio = Self::ratio_for_level(ctx.pressure_level);
        if ratio == 0.0 {
            return Ok(ActionResult::NoOp);
        }

        // Secondary handle absent → swap path disabled (ENG-DAT-C09).
        if self.model_ref.secondary_mmap.is_none() {
            log::debug!("[WeightSwapHandler] no secondary mmap, skipping");
            return Ok(ActionResult::NoOp);
        }

        let num_layers = self.model_ref.layers.len();
        let target_layers = SwapExecutor::uniform_target_layers(ratio, num_layers);
        if target_layers.is_empty() {
            return Ok(ActionResult::NoOp);
        }

        let executor = SwapExecutor::new(
            self.target_dtype,
            &self.model_ref.config,
            self.model_ref.backend.clone(),
            self.memory.as_ref(),
        );

        match executor.execute_on_slots(
            self.model_ref.layers.as_slice(),
            self.model_ref.secondary_mmap.as_ref(),
            &self.model_ref.ratio_generation,
            &target_layers,
        ) {
            Ok(report) => {
                let layers_changed = report.swapped.len();
                if layers_changed == 0 {
                    return Ok(ActionResult::NoOp);
                }
                log::info!(
                    "[WeightSwapHandler] swapped {}/{} layers in {:.1}ms (ratio={:.2})",
                    layers_changed,
                    num_layers,
                    report.latency_ms,
                    ratio,
                );
                Ok(ActionResult::WeightSwapped {
                    layers_changed,
                    // Phase 3: wire actual byte accounting from SwapReport.
                    freed_bytes: 0,
                    duration_ms: report.latency_ms,
                })
            }
            Err(e) => Err(anyhow::anyhow!("[WeightSwapHandler] execute failed: {}", e)),
        }
    }

    fn name(&self) -> &str {
        "weight_swap"
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::pressure::PressureLevel;

    #[test]
    fn ratio_mapping_smoke() {
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

    #[test]
    fn noop_at_normal_pressure() {
        // WeightSwapHandler with no secondary_mmap should return NoOp at
        // every pressure level (ENG-DAT-C09: no secondary → no swap).
        // We cannot instantiate a full handler without a real backend/memory,
        // but we can verify the ratio_for_level helper returns 0.0 for Normal.
        assert_eq!(
            WeightSwapHandler::ratio_for_level(PressureLevel::Normal),
            0.0
        );
    }
}
