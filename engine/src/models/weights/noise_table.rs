//! `QuantNoiseTable` — per-layer quantization noise factor table (ENG-DAT-095).
//!
//! Holds the per-decoder-layer ε_i values computed once at engine init from the
//! primary vs. secondary weight comparison (ENG-ALG-216).  After construction
//! the table is **read-only**; share it with `Arc<QuantNoiseTable>`.
//!
//! `WeightSwapDecider` (Stage B) is the primary consumer.
//!
//! Spec: ENG-DAT-095, ENG-ALG-216, INV-127.

use std::sync::Arc;
use std::time::Instant;

use crate::core::quant::{BlockQ4_0, QK4_0};
use crate::models::weights::secondary_mmap::SecondaryMmap;

// ── QuantNoiseTable ───────────────────────────────────────────────────────────

/// Per-decoder-layer quantization noise factor table (ENG-DAT-095).
///
/// Each entry `per_layer[i] = ε_i` represents the mean relative Frobenius
/// squared error between the primary (F16/F32) and the secondary (Q4_0)
/// weights for decoder layer `i`.  A larger ε_i means the secondary weights
/// diverge more from the primary — that layer is a more "expensive" swap
/// candidate.
///
/// `f32::NAN` entries signal a computation failure for that layer; `epsilon(i)`
/// returns `None` for them, and `WeightSwapDecider` must exclude such layers
/// from the candidate set (INV-127).
pub struct QuantNoiseTable {
    /// Per-decoder-layer ε.  Index = layer_id (decoder layers only).
    per_layer: Vec<f32>,
    /// `true` when built via `new_from_frobenius` (eager path).
    /// `false` for the fallback path (`uniform_ones`).
    computed_at_init: bool,
}

impl QuantNoiseTable {
    /// Construct an **empty** table — used when no secondary is present.
    ///
    /// `len() == 0`, `is_computed() == false`.  The dispatch layer should
    /// reject `SwapWeights` before the decider is invoked in this state.
    pub fn empty() -> Self {
        Self {
            per_layer: Vec::new(),
            computed_at_init: false,
        }
    }

    /// Uniform-ones fallback (ENG-ALG-216 "전체 실패" path).
    ///
    /// All `ε_i = 1.0`, `computed_at_init = false`.  The decider falls back to
    /// importance-only ordering (importance × 1.0 = importance) for all layers.
    pub fn uniform_ones(num_layers: usize) -> Self {
        Self {
            per_layer: vec![1.0f32; num_layers],
            computed_at_init: false,
        }
    }

    /// Compute the table via Frobenius relative error (ENG-ALG-216).
    ///
    /// Iterates all decoder layers present in `secondary`, computes the
    /// mean relative Frobenius squared error across the 7 weight tensors
    /// {Q, K, V, O, gate, up, down} and stores the result in `per_layer[i]`.
    ///
    /// Individual tensor failures (shape mismatch, dequant error) set
    /// `ε_t = 1.0` for that tensor but do not abort the layer.  Layers where
    /// **all** tensors fail get `ε_i = f32::NAN` (INV-127).
    ///
    /// If the entire computation fails (e.g. `secondary` has no layers at
    /// all), the caller should fall back to `QuantNoiseTable::uniform_ones`.
    pub fn new_from_frobenius(
        primary_slots: &[crate::models::weights::LayerSlot],
        secondary: &Arc<SecondaryMmap>,
    ) -> Self {
        let n = primary_slots.len();
        if n == 0 {
            return Self::empty();
        }

        let start = Instant::now();
        let mut per_layer = vec![f32::NAN; n];

        // Tensor sub-names for decoder layers (norm tensors excluded —
        // they are F32 and not quantized in secondary, see ENG-ALG-216).
        const TENSOR_SUBNAMES: &[&str] = &[
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ];

        for (layer_idx, slot) in primary_slots.iter().enumerate() {
            let weights = slot.load_weights();
            let primary_tensors = [
                &weights.wq,
                &weights.wk,
                &weights.wv,
                &weights.wo,
                &weights.w_gate,
                &weights.w_up,
                &weights.w_down,
            ];

            let mut layer_sum = 0.0f64;
            let mut valid_tensors = 0usize;

            for (&subname, primary) in TENSOR_SUBNAMES.iter().zip(primary_tensors.iter()) {
                let epsilon_t =
                    compute_tensor_epsilon(layer_idx, subname, primary, secondary.as_ref());
                match epsilon_t {
                    Some(e) => {
                        layer_sum += e as f64;
                        valid_tensors += 1;
                    }
                    None => {
                        // Fallback: treat as ε_t = 1.0 for the layer sum.
                        // (ENG-ALG-216: "dequantize 실패 시 ε_t = 1.0, warn log")
                        layer_sum += 1.0;
                        valid_tensors += 1;
                    }
                }
            }

            if valid_tensors == 0 {
                // All tensors failed — mark layer as NaN (INV-127).
                per_layer[layer_idx] = f32::NAN;
            } else {
                per_layer[layer_idx] = (layer_sum / valid_tensors as f64) as f32;
            }

            // Progress log (SHOULD per ENG-ALG-216).
            if layer_idx % 4 == 0 || layer_idx + 1 == n {
                let avg: f32 = per_layer[..=layer_idx]
                    .iter()
                    .filter(|v| v.is_finite())
                    .copied()
                    .sum::<f32>()
                    / (layer_idx + 1) as f32;
                log::info!("ε calc: layer {}/{} done, avg={:.6}", layer_idx + 1, n, avg);
            }
        }

        let elapsed_ms = start.elapsed().as_millis();
        let any_fallback = per_layer.iter().any(|v| !v.is_finite());
        log::info!("ε calc done in {}ms, fallback={}", elapsed_ms, any_fallback);

        Self {
            per_layer,
            computed_at_init: true,
        }
    }

    /// Return `Some(ε_i)` if the entry is a valid finite value, else `None`.
    ///
    /// `None` means the layer's ε was not computed (NaN) and the decider must
    /// exclude it from the candidate set (INV-127).
    pub fn epsilon(&self, layer_id: usize) -> Option<f32> {
        let v = *self.per_layer.get(layer_id)?;
        if v.is_finite() { Some(v) } else { None }
    }

    /// Number of decoder layers in the table.
    pub fn len(&self) -> usize {
        self.per_layer.len()
    }

    /// `true` when the table was built via `new_from_frobenius` (eager path).
    pub fn is_computed(&self) -> bool {
        self.computed_at_init
    }

    /// Raw slice of ε values (including possible NaN entries).
    pub fn as_slice(&self) -> &[f32] {
        &self.per_layer
    }

    /// `true` when the table has no entries.
    pub fn is_empty(&self) -> bool {
        self.per_layer.is_empty()
    }

    /// Test-only constructor: build a table with arbitrary ε values and
    /// `computed_at_init = true` so unit tests can exercise the scored path
    /// without a real secondary mmap.
    #[cfg(test)]
    pub fn new_test(values: Vec<f32>) -> Self {
        Self {
            per_layer: values,
            computed_at_init: true,
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Compute ε_t for a single tensor in a single layer.
///
/// Returns `Some(ε_t)` on success, `None` when the secondary tensor is
/// absent, has a shape mismatch, or the dequantization produced invalid data.
fn compute_tensor_epsilon(
    layer_idx: usize,
    subname: &str,
    primary: &crate::core::tensor::Tensor,
    secondary: &SecondaryMmap,
) -> Option<f32> {
    let info = secondary.layer_tensor(layer_idx, subname)?;

    // Verify that the secondary dtype is Q4_0 before dequantizing.
    use crate::core::buffer::DType;
    if info.dtype != DType::Q4_0 {
        log::warn!(
            "ε calc: layer {} '{}' secondary dtype is {:?}, not Q4_0 — skipping",
            layer_idx,
            subname,
            info.dtype
        );
        return None;
    }

    // Number of f32 elements in the primary tensor.
    let numel = primary.shape().numel();
    if numel == 0 {
        return Some(0.0);
    }

    // Number of Q4_0 blocks expected.
    let n_blocks = numel / QK4_0;
    if !numel.is_multiple_of(QK4_0) || n_blocks == 0 {
        log::warn!(
            "ε calc: layer {} '{}' numel {} is not divisible by QK4_0={}",
            layer_idx,
            subname,
            numel,
            QK4_0
        );
        return None;
    }

    let expected_bytes = n_blocks * std::mem::size_of::<BlockQ4_0>();
    let raw = secondary.tensor_bytes(info);
    if raw.len() != expected_bytes {
        log::warn!(
            "ε calc: layer {} '{}' expected {} bytes, got {}",
            layer_idx,
            subname,
            expected_bytes,
            raw.len()
        );
        return None;
    }

    // Dequantize secondary blocks to f32.
    // SAFETY: raw is well-aligned for BlockQ4_0 (repr(C), 18-byte packed read
    // of an f16 + 16 u8; mmap is at least 2-byte aligned by GGUF layout).
    let blocks: &[BlockQ4_0] =
        // SAFETY: BlockQ4_0 has repr(C), size=18 B.  `raw` is a contiguous
        // byte slice from the mmap whose length is exactly `n_blocks * 18`.
        // We assert alignment is fine: GGUF tensors are guaranteed to be
        // 32-byte aligned by the spec; BlockQ4_0 requires only 2-byte
        // alignment (the f16 field).
        unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const BlockQ4_0, n_blocks) };

    let mut secondary_f32 = vec![0.0f32; numel];
    for (bi, block) in blocks.iter().enumerate() {
        let start = bi * QK4_0;
        // SAFETY: We allocated `secondary_f32` with exactly `n_blocks * QK4_0`
        // elements, so `[start..start+QK4_0]` is always in-bounds.
        let out_slice: &mut [f32; QK4_0] = unsafe {
            &mut *(secondary_f32[start..start + QK4_0].as_mut_ptr() as *mut [f32; QK4_0])
        };
        block.dequantize(out_slice);
    }

    // Read primary weights as f32.
    // The primary tensor may be F16 or F32.  Use the buffer's `as_f32_slice`
    // abstraction if available; otherwise fall through to the f16→f32 path.
    let primary_f32: Vec<f32> = primary_tensor_to_f32(primary)?;
    if primary_f32.len() != numel {
        log::warn!(
            "ε calc: layer {} '{}' primary f32 len {} != numel {}",
            layer_idx,
            subname,
            primary_f32.len(),
            numel
        );
        return None;
    }

    // Frobenius relative squared error:
    //   num   = || W_primary - W_secondary_fp ||_F²
    //   denom = || W_primary ||_F²
    //   ε_t   = num / max(denom, EPS)
    const EPS: f64 = 1e-12;
    let (mut num, mut denom) = (0.0f64, 0.0f64);
    for (p, s) in primary_f32.iter().zip(secondary_f32.iter()) {
        let diff = (*p as f64) - (*s as f64);
        num += diff * diff;
        denom += (*p as f64) * (*p as f64);
    }
    let epsilon_t = (num / denom.max(EPS)) as f32;
    Some(epsilon_t)
}

/// Convert a primary weight tensor to a `Vec<f32>`.
///
/// Supports F32 and F16 dtype buffers.  Returns `None` for Q4_0 primaries
/// (should not occur for the ε calculation path) or on access failure.
fn primary_tensor_to_f32(tensor: &crate::core::tensor::Tensor) -> Option<Vec<f32>> {
    use crate::core::buffer::DType;
    use half::f16;

    let numel = tensor.shape().numel();
    match tensor.dtype() {
        DType::F32 => {
            let ptr = tensor.as_ptr();
            if ptr.is_null() {
                return None;
            }
            // SAFETY: ptr points to at least `numel * 4` bytes of F32 data
            // for the lifetime of `tensor`.
            let slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, numel) };
            Some(slice.to_vec())
        }
        DType::F16 => {
            let ptr = tensor.as_ptr();
            if ptr.is_null() {
                return None;
            }
            // SAFETY: ptr points to at least `numel * 2` bytes of F16 data.
            let slice = unsafe { std::slice::from_raw_parts(ptr as *const f16, numel) };
            Some(slice.iter().map(|v| v.to_f32()).collect())
        }
        _ => {
            log::warn!(
                "primary_tensor_to_f32: unsupported dtype {:?}",
                tensor.dtype()
            );
            None
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_table() {
        let t = QuantNoiseTable::empty();
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
        assert!(!t.is_computed());
        assert!(t.epsilon(0).is_none());
        assert_eq!(t.as_slice().len(), 0);
    }

    #[test]
    fn test_uniform_ones() {
        let t = QuantNoiseTable::uniform_ones(16);
        assert_eq!(t.len(), 16);
        assert!(!t.is_computed());
        for i in 0..16 {
            assert_eq!(t.epsilon(i), Some(1.0));
        }
        assert!(t.epsilon(16).is_none());
    }

    #[test]
    fn test_nan_layer_returns_none() {
        let mut table = QuantNoiseTable::uniform_ones(4);
        // Manually inject a NaN to simulate a failed layer.
        table.per_layer[2] = f32::NAN;
        assert_eq!(table.epsilon(0), Some(1.0));
        assert_eq!(table.epsilon(1), Some(1.0));
        // INV-127: NaN → None
        assert!(table.epsilon(2).is_none());
        assert_eq!(table.epsilon(3), Some(1.0));
    }

    #[test]
    fn test_out_of_range_returns_none() {
        let t = QuantNoiseTable::uniform_ones(4);
        assert!(t.epsilon(4).is_none());
        assert!(t.epsilon(100).is_none());
    }

    #[test]
    fn test_is_computed_flag() {
        let empty = QuantNoiseTable::empty();
        assert!(!empty.is_computed());
        let ones = QuantNoiseTable::uniform_ones(4);
        assert!(!ones.is_computed());
    }

    #[test]
    fn test_as_slice_contents() {
        let t = QuantNoiseTable::uniform_ones(3);
        let s = t.as_slice();
        assert_eq!(s.len(), 3);
        assert!(s.iter().all(|&v| (v - 1.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_finite_epsilon_value() {
        // Inject a known finite value and verify epsilon() returns it.
        let mut t = QuantNoiseTable::uniform_ones(4);
        t.per_layer[1] = 0.042;
        assert!((t.epsilon(1).unwrap() - 0.042).abs() < 1e-6);
    }
}
