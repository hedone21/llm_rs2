//! `SwapExecutor` — Phase 2 runtime layer swap (ENG-ALG-211).
//!
//! The executor materialises a fresh `LayerWeights` snapshot from the
//! secondary mmap for each target decoder layer and installs it into the
//! corresponding `LayerSlot` via `ArcSwap::store`. A single
//! `TransformerModel::ratio_generation` bump at the end of the batch
//! triggers plan invalidation (INV-120) exactly once (ENG-ALG-211 step (e)).
//!
//! **Scope (Stage 1)**: uniform, index-based target selection only — QCF
//! importance wiring lands in Phase 3 with `SwapDecider`.
//!
//! **Safety**: the executor is intended to run single-threaded relative to
//! other writers. Readers (forward pass) acquire `Arc<LayerWeights>`
//! snapshots on token entry (ENG-ALG-214-SNAP) and are therefore unaffected
//! by concurrent swaps — INV-121/INV-123.
//!
//! Spec: ENG-ALG-211, ENG-DAT-092/093/094, INV-120/121/123/124/125.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use anyhow::Result;
use llm_shared::DtypeTag;

use crate::buffer::shared_buffer::SharedBuffer;
use crate::core::backend::Backend;
use crate::core::buffer::{Buffer, DType};
use crate::core::memory::Memory;
use crate::core::tensor::Tensor;
use crate::layers::transformer_layer::TransformerLayer;
use crate::models::config::ModelConfig;
use crate::models::loader::gguf::{qk_permute_shape, unpermute_qk_rows};
use crate::models::transformer::TransformerModel;
use crate::models::weights::{LayerSlot, LayerWeights, SecondaryMmap};

/// Errors surfaced by `SwapExecutor::execute`.
#[derive(Debug)]
pub enum SwapError {
    /// A tensor expected by the secondary mmap index is missing.
    SecondaryTensorMissing { layer: usize, subname: String },
    /// Backend refused to allocate a host-side weight tensor.
    BufferAllocationFailed { layer: usize, source: anyhow::Error },
    /// Shape inferred from secondary dims does not match the primary layer.
    ShapeMismatch {
        layer: usize,
        subname: String,
        primary: Vec<usize>,
        secondary: Vec<u64>,
    },
    /// Wire-protocol `DtypeTag` that has no engine-internal `DType` mapping
    /// (reserved variants — INV-126). The variant name is included for
    /// diagnostics.
    UnsupportedDtype(DtypeTag),
}

impl std::fmt::Display for SwapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SwapError::SecondaryTensorMissing { layer, subname } => write!(
                f,
                "secondary mmap: layer {layer} missing tensor '{subname}'"
            ),
            SwapError::BufferAllocationFailed { layer, source } => {
                write!(f, "swap layer {layer}: buffer allocation failed: {source}")
            }
            SwapError::ShapeMismatch {
                layer,
                subname,
                primary,
                secondary,
            } => write!(
                f,
                "swap layer {layer} tensor '{subname}': \
                 shape mismatch (primary={primary:?}, secondary={secondary:?})"
            ),
            SwapError::UnsupportedDtype(tag) => {
                write!(
                    f,
                    "unsupported target dtype: {tag:?} (INV-126 reserved variant)"
                )
            }
        }
    }
}

impl std::error::Error for SwapError {}

/// Map a wire-protocol [`DtypeTag`] to the engine-internal [`DType`] used by
/// `SwapExecutor`. Returns `Err(SwapError::UnsupportedDtype)` for reserved
/// variants that are not yet executable (INV-126).
///
/// Only `DtypeTag::Q4_0` is currently mapped; all other variants are reserved
/// for forward-compat and will be rejected until engine support lands.
pub fn dtype_tag_to_dtype(tag: DtypeTag) -> Result<DType, SwapError> {
    match tag {
        DtypeTag::Q4_0 => Ok(DType::Q4_0),
        other => Err(SwapError::UnsupportedDtype(other)),
    }
}

/// Outcome of a single layer swap inside one batch.
#[derive(Debug, Clone)]
pub struct SwappedLayer {
    pub layer_idx: usize,
    pub from_dtype: DType,
    pub to_dtype: DType,
}

/// Aggregated result of one `execute` batch (ENG-ALG-211 "SwapResult").
#[derive(Debug, Clone, Default)]
pub struct SwapReport {
    /// Layers whose snapshot was atomically replaced.
    pub swapped: Vec<SwappedLayer>,
    /// Layers we intended to swap but skipped (already at target dtype or
    /// outside the range / missing secondary handle).
    pub skipped: Vec<usize>,
    /// Wall clock for the full batch.
    pub latency_ms: f64,
    /// Value of `TransformerModel::ratio_generation` after the batch. `None`
    /// when no actual swap happened (no bump — ENG-ALG-211 step (e) contract).
    pub ratio_generation_after: Option<u64>,
}

/// Stateless executor. One instance per swap request is fine.
pub struct SwapExecutor<'a> {
    /// Target dtype for all layers in this batch. Fixed per batch because
    /// ratio-based swap conceptually picks one secondary dtype per batch.
    pub target_dtype: DType,
    /// Model config — drives Q/K permutation gating (only Llama rearranges
    /// at load time, mirroring `gguf.rs`).
    pub config: &'a ModelConfig,
    /// Where we land the fresh weight tensors (CPU for host tests, the
    /// primary backend in production). Must match the existing layer's
    /// backend so downstream kernels remain compatible.
    pub backend: Arc<dyn Backend>,
    /// Memory allocator paired with `backend`. Used for the permutation
    /// fallback path when we have to materialise an owned buffer.
    pub memory: &'a dyn Memory,
}

impl<'a> SwapExecutor<'a> {
    /// Construct an executor. `target_dtype` is the dtype we are swapping
    /// *to* (e.g. `Q4_0` for F16→Q4_0). Backend+memory must be consistent
    /// with the slots we will touch.
    pub fn new(
        target_dtype: DType,
        config: &'a ModelConfig,
        backend: Arc<dyn Backend>,
        memory: &'a dyn Memory,
    ) -> Self {
        Self {
            target_dtype,
            config,
            backend,
            memory,
        }
    }

    /// Select uniform index-spaced target layers (Stage 1 fallback, ENG-ALG-212).
    ///
    /// Deterministic: `ceil(ratio * num_layers)` layers evenly spaced across
    /// `[0, num_layers)`. Layers already at `target_dtype` are skipped by
    /// `execute` itself (idempotent).
    pub fn uniform_target_layers(ratio: f32, num_layers: usize) -> Vec<usize> {
        if num_layers == 0 {
            return Vec::new();
        }
        let clamped = ratio.clamp(0.0, 1.0);
        let count = ((clamped * num_layers as f32).round() as usize).min(num_layers);
        if count == 0 {
            return Vec::new();
        }
        if count == num_layers {
            return (0..num_layers).collect();
        }
        // Evenly spaced indices. For count==k and n layers, we use
        // floor((i + 0.5) * n / k) which stays stable under tie breaks.
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let idx = ((i as f32 + 0.5) * num_layers as f32 / count as f32) as usize;
            out.push(idx.min(num_layers - 1));
        }
        out.dedup();
        out
    }

    /// Execute a swap batch. `target_layers` is the caller-chosen set of
    /// decoder indices to convert. Out-of-range / already-swapped layers
    /// are silently skipped (ENG-DAT-C08, ENG-ALG-211).
    ///
    /// Returns `SwapReport` describing what landed. On success,
    /// `TransformerModel::ratio_generation` is bumped **exactly once** when
    /// at least one layer was actually swapped (ENG-ALG-211 step (e)).
    pub fn execute(
        &self,
        model: &TransformerModel,
        target_layers: &[usize],
    ) -> Result<SwapReport, SwapError> {
        self.execute_on_slots(
            model.layers.as_slice(),
            model.secondary_mmap.as_ref(),
            &model.ratio_generation,
            target_layers,
        )
    }

    /// Low-level execute that accepts raw slot/mmap/generation references.
    ///
    /// Called by `WeightSwapHandler` which holds these fields directly rather
    /// than through a `&TransformerModel`. The semantics are identical to
    /// `execute()`: ENG-ALG-211 batch swap with a single `ratio_generation`
    /// bump at the end.
    pub fn execute_on_slots(
        &self,
        layers: &[LayerSlot],
        secondary_mmap: Option<&Arc<crate::models::weights::SecondaryMmap>>,
        ratio_generation: &Arc<std::sync::atomic::AtomicU64>,
        target_layers: &[usize],
    ) -> Result<SwapReport, SwapError> {
        let start = Instant::now();
        let mut report = SwapReport::default();

        // ENG-DAT-C09: secondary handle absent → entire operation is a no-op.
        let Some(secondary) = secondary_mmap else {
            report.latency_ms = start.elapsed().as_secs_f64() * 1e3;
            return Ok(report);
        };

        for &layer_idx in target_layers {
            // ENG-DAT-C08: out-of-range silently skipped.
            let Some(slot) = layers.get(layer_idx) else {
                report.skipped.push(layer_idx);
                continue;
            };
            // Idempotent: already at target dtype.
            if slot.current_dtype() == self.target_dtype {
                report.skipped.push(layer_idx);
                continue;
            }
            // Secondary handle may have been stripped (no swap path). Treat
            // as skip rather than error — matches ENG-DAT-C09 spirit when
            // the per-slot handle was never installed.
            if slot.secondary_mmap_handle().is_none() {
                report.skipped.push(layer_idx);
                continue;
            }

            let from_dtype = slot.current_dtype();
            let new_layer = self.build_layer_from_mmap(secondary, slot, layer_idx)?;
            let new_arc: Arc<LayerWeights> = Arc::new(new_layer);

            // Snapshot the old Arc so we can (a) inspect buffer ranges for
            // madvise and (b) hold the strong_count check before dropping
            // our local reference below.
            let old = slot.load_weights();

            // (c) Atomic install — INV-123. swap_weights bumps the per-slot
            // debug generation as part of its contract.
            slot.swap_weights(new_arc, self.target_dtype);

            // (d) Best-effort page reclaim on the just-replaced primary
            // pages. Only safe when we are the last holder; otherwise an
            // in-flight forward still owns the old Arc and madvise would
            // race. Stage 1 conservative policy: skip silently when
            // strong_count > 1 and leave the OS to reclaim on final drop.
            Self::madvise_if_exclusive(&old);
            drop(old);

            report.swapped.push(SwappedLayer {
                layer_idx,
                from_dtype,
                to_dtype: self.target_dtype,
            });
        }

        // (e) Single batch-level bump of the global ratio_generation counter.
        // Empty swaps do NOT bump (ENG-ALG-211: "if !swapped.is_empty()").
        if !report.swapped.is_empty() {
            // ENG-ALG-221 / INV-130: invalidate the Adreno Q4_0 noshuffle SOA
            // registry before the generation bump so that the subsequent
            // `FullKernelPlan` rebuild (triggered by the bump via ENG-ALG-219)
            // re-registers SOA descriptors against the new `cl_mem` keys.
            // No-op on CPU / CUDA backends (trait default).
            //
            // Ordering rationale: clear must precede the bump so any concurrent
            // observer that takes the new generation cannot race against a
            // still-populated stale registry. The per-token `Arc<LayerWeights>`
            // snapshot (INV-121/123) already shields in-flight forwards that
            // captured the old generation.
            self.backend.invalidate_noshuffle_soa_registry();

            // ENG-ALG-222 / INV-131 — Phase 3.7a: re-convert Q4_0 weights to
            // SOA layout and register the entries against the new cl_mem
            // addresses **before** the generation bump. Without this safety
            // net the noshuffle GEMV kernel falls back to the AOS path on
            // Adreno (verified silently incorrect — Phase 3.6 measurements
            // showed only the first decoded token surviving). NoOp on
            // CPU / CUDA backends and on host OpenCL builds whose
            // `cvt_noshuffle` program failed to compile.
            //
            // Ordering rationale: this MUST happen after `invalidate_*` (so
            // we re-populate a clean registry) and before the generation bump
            // (so the next forward seeing the new generation finds a fully
            // populated registry, eliminating any window where lookup misses).
            for swapped in &report.swapped {
                let Some(slot) = layers.get(swapped.layer_idx) else {
                    continue;
                };
                let new_layer = slot.load_weights();
                for tensor in [
                    &new_layer.wq,
                    &new_layer.wk,
                    &new_layer.wv,
                    &new_layer.wo,
                    &new_layer.w_gate,
                    &new_layer.w_up,
                    &new_layer.w_down,
                ] {
                    if let Err(e) = self.backend.ensure_noshuffle_soa_registered(tensor) {
                        // Conversion failure leaves the registry empty for
                        // this tensor → AOS fallback path. Surface as the
                        // batch error so the caller can decide (treating
                        // partial registration as success would risk silent
                        // accuracy loss).
                        return Err(SwapError::BufferAllocationFailed {
                            layer: swapped.layer_idx,
                            source: e,
                        });
                    }
                }
            }

            let new_gen = ratio_generation.fetch_add(1, Ordering::SeqCst) + 1;
            report.ratio_generation_after = Some(new_gen);
        }

        report.latency_ms = start.elapsed().as_secs_f64() * 1e3;
        Ok(report)
    }

    /// Build a fresh `TransformerLayer` from the secondary mmap for the
    /// given `layer_idx`, mirroring the primary loader's dtype handling and
    /// Q/K permutation gating.
    ///
    /// Non-swap fields (partition context) are preserved from the current
    /// snapshot so downstream tensor-partition state is retained.
    fn build_layer_from_mmap(
        &self,
        secondary: &Arc<SecondaryMmap>,
        slot: &LayerSlot,
        layer_idx: usize,
    ) -> Result<TransformerLayer, SwapError> {
        let old = slot.load_weights();

        let wq = self.materialise_tensor(secondary, layer_idx, "attn_q.weight", &old.wq, true)?;
        let wk = self.materialise_tensor(secondary, layer_idx, "attn_k.weight", &old.wk, true)?;
        let wv = self.materialise_tensor(secondary, layer_idx, "attn_v.weight", &old.wv, true)?;
        let wo =
            self.materialise_tensor(secondary, layer_idx, "attn_output.weight", &old.wo, true)?;
        let w_gate =
            self.materialise_tensor(secondary, layer_idx, "ffn_gate.weight", &old.w_gate, true)?;
        let w_up =
            self.materialise_tensor(secondary, layer_idx, "ffn_up.weight", &old.w_up, true)?;
        let w_down =
            self.materialise_tensor(secondary, layer_idx, "ffn_down.weight", &old.w_down, true)?;

        // Norms are typically F32. We re-read them from secondary to avoid
        // relying on primary-lifetime buffers in the new snapshot. If the
        // secondary omits a norm (unlikely for standard GGUFs), fall back
        // to cloning the primary tensor — shape/dtype match is enforced by
        // `open_secondary` metadata validation.
        let attention_norm = self.clone_or_materialise_norm(
            secondary,
            layer_idx,
            "attn_norm.weight",
            &old.attention_norm,
        )?;
        let ffn_norm =
            self.clone_or_materialise_norm(secondary, layer_idx, "ffn_norm.weight", &old.ffn_norm)?;

        // Optional Gemma3 / Qwen norms — preserve existing tensors since
        // they are not part of the swap target set for this stage.
        //
        // DF-35-3 (ENG-ALG-219): tensor_partition × weight swap are mutually
        // exclusive. A weight swap installs new cl_mem handles for the weight
        // buffers; any pre-existing PartitionContext still points at the OLD
        // cl_mem slices and is therefore invalid after the swap. Force
        // partition_ctx to None so the plan-path correctly falls back to the
        // GPU-only FFN rather than dispatching a stale partition step.
        // The caller (generate.rs) will reinstate partition via
        // `prepare_tensor_partition` if `--tensor-partition` is active.
        Ok(TransformerLayer {
            wq,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
            attention_norm,
            ffn_norm,
            qkv_bias: old.qkv_bias.clone(),
            q_norm: old.q_norm.clone(),
            k_norm: old.k_norm.clone(),
            pre_ffn_norm: old.pre_ffn_norm.clone(),
            post_ffn_norm: old.post_ffn_norm.clone(),
            partition_ctx: None, // DF-35-3: cleared on swap (see above)
        })
    }

    /// Materialise one weight tensor from the secondary mmap, applying Q/K
    /// row permutation when `subname` identifies a Llama Q/K weight.
    fn materialise_tensor(
        &self,
        secondary: &Arc<SecondaryMmap>,
        layer_idx: usize,
        subname: &str,
        primary: &Tensor,
        is_weight: bool,
    ) -> Result<Tensor, SwapError> {
        let info = secondary.layer_tensor(layer_idx, subname).ok_or_else(|| {
            SwapError::SecondaryTensorMissing {
                layer: layer_idx,
                subname: subname.to_string(),
            }
        })?;

        let shape = primary.shape().clone();
        let expected_dims: Vec<usize> = primary.shape().dims().to_vec();
        // Secondary dims are stored in reverse order (GGUF innermost-first).
        let sec_rev: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
        if sec_rev != expected_dims {
            return Err(SwapError::ShapeMismatch {
                layer: layer_idx,
                subname: subname.to_string(),
                primary: expected_dims,
                secondary: info.dims.clone(),
            });
        }

        // Raw bytes from the secondary mmap.
        let data = secondary.tensor_bytes(info);

        // Build a canonical GGUF tensor name for the permutation gate, so we
        // reuse the same predicate as the primary loader without copying
        // regex logic.
        let canonical_name = format!("blk.{layer_idx}.{subname}");
        let permuted_bytes: Option<Vec<u8>> =
            if let Some((n_head, head_dim)) = qk_permute_shape(&canonical_name, self.config) {
                let total_rows = shape.dims()[0];
                debug_assert_eq!(total_rows, n_head * head_dim);
                if !data.len().is_multiple_of(total_rows) {
                    return Err(SwapError::ShapeMismatch {
                        layer: layer_idx,
                        subname: subname.to_string(),
                        primary: expected_dims,
                        secondary: info.dims.clone(),
                    });
                }
                let row_size_bytes = data.len() / total_rows;
                Some(unpermute_qk_rows(data, n_head, head_dim, row_size_bytes))
            } else {
                None
            };

        // Always build an owned SharedBuffer on CPU first, then route
        // through the existing copy_weight_from / copy_from paths to land on
        // the final backend. This matches the loader's behaviour for
        // permuted weights and avoids plumbing mmap buffer lifetimes into
        // the slot (the secondary mmap Arc is the lifetime keeper, but
        // keeping per-tensor views alive would complicate madvise).
        let owned_bytes: Vec<u8> = permuted_bytes.unwrap_or_else(|| data.to_vec());
        let shared_buf: Arc<dyn Buffer> = Arc::new(SharedBuffer::from_vec(owned_bytes, info.dtype));

        let cpu_backend: Arc<dyn Backend> =
            Arc::new(crate::backend::cpu::CpuBackend::new()) as Arc<dyn Backend>;
        let cpu_tensor = Tensor::new(shape.clone(), shared_buf, cpu_backend);

        let is_cpu = self.backend.name().contains("CPU");
        if is_cpu {
            Ok(cpu_tensor)
        } else if is_weight {
            self.backend.copy_weight_from(&cpu_tensor).map_err(|e| {
                SwapError::BufferAllocationFailed {
                    layer: layer_idx,
                    source: e,
                }
            })
        } else {
            self.backend
                .copy_from(&cpu_tensor)
                .map_err(|e| SwapError::BufferAllocationFailed {
                    layer: layer_idx,
                    source: e,
                })
        }
    }

    /// Materialise a norm tensor if present in the secondary file; otherwise
    /// fall back to cloning the primary tensor. Norms are rarely stored
    /// differently across GGUFs so the fallback is the common path.
    fn clone_or_materialise_norm(
        &self,
        secondary: &Arc<SecondaryMmap>,
        layer_idx: usize,
        subname: &str,
        primary: &Tensor,
    ) -> Result<Tensor, SwapError> {
        if secondary.layer_tensor(layer_idx, subname).is_some() {
            self.materialise_tensor(secondary, layer_idx, subname, primary, false)
        } else {
            Ok(primary.clone())
        }
    }

    /// Advise the kernel that the primary pages backing `old` can be
    /// reclaimed. Safe only when we hold the last reference — otherwise an
    /// in-flight forward still uses the tensor. Best-effort, never panics.
    fn madvise_if_exclusive(old: &Arc<LayerWeights>) {
        // strong_count is racy relative to readers that have not yet
        // cloned the Arc, but any such reader must have observed the new
        // snapshot (via `slot.load_weights()` after our `swap_weights`) to
        // be newly entering the forward path. Readers that saw the old
        // snapshot before the swap will keep strong_count > 1 here.
        if Arc::strong_count(old) > 1 {
            return;
        }
        #[cfg(target_os = "linux")]
        {
            Self::madvise_tensor(&old.wq);
            Self::madvise_tensor(&old.wk);
            Self::madvise_tensor(&old.wv);
            Self::madvise_tensor(&old.wo);
            Self::madvise_tensor(&old.w_gate);
            Self::madvise_tensor(&old.w_up);
            Self::madvise_tensor(&old.w_down);
        }
    }

    #[cfg(target_os = "linux")]
    fn madvise_tensor(t: &Tensor) {
        // Only MmapBuffer-backed or host-managed buffers are amenable to
        // MADV_DONTNEED. Arbitrary device-only cl_mem allocations return a
        // null pointer and are skipped. See ENG-ALG-C05/C06.
        let ptr = t.buffer().as_ptr();
        if ptr.is_null() {
            return;
        }
        let len = t.size();
        if len == 0 {
            return;
        }
        // Align to a page boundary to satisfy the madvise contract; trim
        // head/tail partial pages rather than round out and stomp a sibling
        // tensor.
        let page = 4096usize;
        let addr = ptr as usize;
        let aligned = addr.div_ceil(page) * page;
        let tail = (addr + len) & !(page - 1);
        if tail <= aligned {
            return;
        }
        let adv_len = tail - aligned;
        // SAFETY: address range [aligned, aligned+adv_len) is a subset of
        // [addr, addr+len) which is live for the lifetime of `t`. MADV_DONTNEED
        // is a hint and failure is non-fatal.
        unsafe {
            let _ = libc::madvise(aligned as *mut libc::c_void, adv_len, libc::MADV_DONTNEED);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_target_layers_matches_spec() {
        // Spec ENG-ALG-213 uniform fallback shape: ceil(ratio * n) layers,
        // evenly spaced. These are not-yet-plan-integration shape tests to
        // guard against refactors that silently change the selection.
        assert!(SwapExecutor::uniform_target_layers(0.0, 16).is_empty());
        assert_eq!(
            SwapExecutor::uniform_target_layers(1.0, 16).len(),
            16,
            "ratio=1.0 selects all layers"
        );
        assert_eq!(SwapExecutor::uniform_target_layers(0.25, 16).len(), 4);
        assert_eq!(SwapExecutor::uniform_target_layers(0.5, 16).len(), 8);
        // Degenerate size.
        assert!(SwapExecutor::uniform_target_layers(0.5, 0).is_empty());
    }

    // INV-126: dtype_tag_to_dtype must accept Q4_0 and reject all reserved variants.

    #[test]
    fn dtype_tag_q4_0_maps_to_dtype_q4_0() {
        let result = dtype_tag_to_dtype(DtypeTag::Q4_0);
        assert!(
            matches!(result, Ok(DType::Q4_0)),
            "Q4_0 should be executable"
        );
    }

    #[test]
    fn dtype_tag_f16_is_unsupported() {
        let result = dtype_tag_to_dtype(DtypeTag::F16);
        assert!(
            matches!(result, Err(SwapError::UnsupportedDtype(DtypeTag::F16))),
            "F16 is reserved, must return UnsupportedDtype"
        );
    }

    #[test]
    fn dtype_tag_f32_is_unsupported() {
        let result = dtype_tag_to_dtype(DtypeTag::F32);
        assert!(
            matches!(result, Err(SwapError::UnsupportedDtype(DtypeTag::F32))),
            "F32 is reserved, must return UnsupportedDtype"
        );
    }

    #[test]
    fn dtype_tag_q8_0_is_unsupported() {
        let result = dtype_tag_to_dtype(DtypeTag::Q8_0);
        assert!(
            matches!(result, Err(SwapError::UnsupportedDtype(DtypeTag::Q8_0))),
            "Q8_0 is reserved, must return UnsupportedDtype"
        );
    }

    #[test]
    fn unsupported_dtype_error_display_contains_inv_126() {
        let err = dtype_tag_to_dtype(DtypeTag::F16).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("INV-126"),
            "Error message should reference INV-126, got: {msg}"
        );
    }
}
