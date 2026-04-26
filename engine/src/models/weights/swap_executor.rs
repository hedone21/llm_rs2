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

/// Per-stage accumulated timings for one `execute` batch.
///
/// Collected via `Instant::now()` pairs at stage boundaries — µs-level
/// overhead, negligible relative to the ms-scale operations measured.
///
/// Stage mapping (WSWAP-4-LATENCY-β + WSWAP-5-COLD-UNIFORM):
/// - `prefault_ms` — (a-pre) WSWAP-5 batch-once `madvise(MADV_WILLNEED)` +
///   page-touch warmup over the secondary mmap weights region. Charged once
///   per `execute_on_slots` call (not per layer) so cold/warm runs are
///   directly comparable.
/// - `mmap_permute_ms` — (a) secondary slice + Q/K permutation + GPU upload
///   (`build_layer_from_mmap`, per-layer accumulated)
/// - `arc_swap_ms` — (b) `LayerSlot::swap_weights` atomic Arc store
/// - `madvise_ms` — (c) `madvise(MADV_DONTNEED)` on replaced pages
/// - `soa_reconvert_ms` — (d) `ensure_noshuffle_soa_registered` loop
///   (Phase 3.7a, hypothesis: dominant bottleneck)
/// - `gen_bump_ms` — (e) `invalidate_noshuffle_soa_registry` +
///   `ratio_generation.fetch_add`
#[derive(Debug, Clone, Default)]
pub struct StageBreakdown {
    /// (a-pre) WSWAP-5 prefault: `madvise(MADV_WILLNEED)` + page-touch warmup.
    /// Single batch-level cost, NOT per layer. Stays at 0.0 on non-Linux
    /// targets / on no-op secondaries (CPU host tests).
    pub prefault_ms: f64,
    /// (a) secondary mmap slice + Q/K permutation + GPU copy_weight_from
    pub mmap_permute_ms: f64,
    /// (b) `LayerSlot::swap_weights` Arc store (per-layer accumulated)
    pub arc_swap_ms: f64,
    /// (c) `madvise(MADV_DONTNEED)` (per-layer accumulated)
    pub madvise_ms: f64,
    /// (d) `ensure_noshuffle_soa_registered` SOA re-conversion loop
    pub soa_reconvert_ms: f64,
    /// (e) `invalidate_noshuffle_soa_registry` + `ratio_generation` bump
    pub gen_bump_ms: f64,
}

impl StageBreakdown {
    /// Format as a compact single-line string for stderr diagnostics.
    ///
    /// Example:
    /// `prefault=12.3ms mmap_permute=123.4ms arc_swap=0.1ms madvise=0.2ms soa_reconvert=45.6ms gen_bump=0.1ms`
    pub fn to_log_line(&self) -> String {
        format!(
            "prefault={:.1}ms mmap_permute={:.1}ms arc_swap={:.1}ms madvise={:.1}ms \
             soa_reconvert={:.1}ms gen_bump={:.1}ms",
            self.prefault_ms,
            self.mmap_permute_ms,
            self.arc_swap_ms,
            self.madvise_ms,
            self.soa_reconvert_ms,
            self.gen_bump_ms,
        )
    }
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
    /// Per-stage timing breakdown (WSWAP-4-LATENCY-β).
    /// Always populated on a successful batch with at least one swapped layer;
    /// `None` when the entire batch was a no-op (no secondary mmap).
    pub stage_breakdown: Option<StageBreakdown>,
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

        // WSWAP-4-LATENCY-β: per-stage accumulated timings.
        let mut stages = StageBreakdown::default();

        // ── Stage (a-pre): WSWAP-5-COLD-UNIFORM prefault ─────────────────────
        // Issue `madvise(MADV_WILLNEED)` + explicit page-touch warmup over the
        // secondary mmap weights region exactly once per batch — before the
        // per-layer copy loop. The cost (a few ms for ~700 MB on Adreno UFS)
        // is paid up-front so subsequent `mmap_permute` reads always hit the
        // page cache. Without this, the 5th measurement showed a per-run
        // bimodal split (cold ~53 ms / warm ~36 ms per layer) driven entirely
        // by mmap demand-paging on cold runs.
        //
        // Charged as a single batch-level stage (NOT per layer) so cold and
        // warm runs are directly comparable. No-op on non-Linux targets.
        let t_pre0 = Instant::now();
        secondary.prefault();
        stages.prefault_ms = t_pre0.elapsed().as_secs_f64() * 1e3;

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

            // ── Stage (a): secondary mmap slice + Q/K permutation + GPU upload ──
            let t_a0 = Instant::now();
            let new_layer = self.build_layer_from_mmap(secondary, slot, layer_idx)?;
            stages.mmap_permute_ms += t_a0.elapsed().as_secs_f64() * 1e3;

            let new_arc: Arc<LayerWeights> = Arc::new(new_layer);

            // ── Stage (b): Atomic install — INV-123. `swap_weights` now uses
            // `ArcSwap::swap()` internally and returns the previous Arc after
            // `wait_for_readers` has flushed any in-flight ArcSwap hazard
            // pointers. The returned Arc is the only outstanding reference
            // held by this dispatch path — readers from prior forward passes
            // have already dropped their snapshots (the swap dispatcher runs
            // strictly between forwards). ──────────────────────────────────────
            let t_b0 = Instant::now();
            let old = slot.swap_weights(new_arc, self.target_dtype);
            stages.arc_swap_ms += t_b0.elapsed().as_secs_f64() * 1e3;

            // ── Stage (c): Primary cl_mem release — WSWAP-5-PRIMARY-DROP.
            //
            // Take ownership of the inner `LayerWeights` from the returned
            // Arc and drop the primary weight tensors explicitly. Each
            // tensor's `Arc<dyn Buffer>` is then the unique owner of its
            // backing `OpenCLBuffer`, so the destructor releases the
            // underlying `cl_mem` immediately. This reclaims the F16
            // primary 2.4 GB / 145 cl_mem footprint that previously stayed
            // alive on Galaxy S25 ratio=1.0 mixed (`phase_5_tbt_diag.md`).
            //
            // Falls back to madvise-only when `Arc::try_unwrap` fails.
            // The latter only happens if some other holder kept the Arc —
            // none expected in production (forwards run sequentially against
            // the dispatcher), but the fallback preserves correctness if a
            // future caller ever holds a snapshot across a swap.
            //
            // ENG-ALG-211 step (c) refined; `Self::madvise_if_exclusive`
            // remains the fallback path so MADV_DONTNEED still fires on
            // mmap-backed primaries when we can't acquire unique ownership.
            let t_c0 = Instant::now();
            match Arc::try_unwrap(old) {
                Ok(layer) => {
                    Self::release_primary_weights(&self.backend, layer);
                }
                Err(arc) => {
                    // Non-exclusive ownership — best-effort reclaim only.
                    // Records strong_count for diagnostics on the AOS / GGUF
                    // path; the AUF SOA bypass path doesn't actually need
                    // madvise (its primary cl_mem was the F16 GGUF copy
                    // routed through `copy_weight_from`, not an mmap'd page).
                    Self::madvise_if_exclusive(&arc);
                }
            }
            stages.madvise_ms += t_c0.elapsed().as_secs_f64() * 1e3;

            report.swapped.push(SwappedLayer {
                layer_idx,
                from_dtype,
                to_dtype: self.target_dtype,
            });
        }

        // (e) Single batch-level bump of the global ratio_generation counter.
        // Empty swaps do NOT bump (ENG-ALG-211: "if !swapped.is_empty()").
        if !report.swapped.is_empty() {
            // ── Stage (e-pre): invalidate registry ───────────────────────────
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
            let t_e0 = Instant::now();
            self.backend.invalidate_noshuffle_soa_registry();
            stages.gen_bump_ms += t_e0.elapsed().as_secs_f64() * 1e3;

            // ── Stage (d): SOA re-conversion ─────────────────────────────────
            // ENG-ALG-222 / INV-131 — Phase 3.7a: re-convert Q4_0 weights to
            // SOA layout and register the entries against the new cl_mem
            // addresses **before** the generation bump. Without this safety
            // net the noshuffle GEMV kernel falls back to the AOS path on
            // Adreno (verified silently incorrect — Phase 3.6 measurements
            // showed only the first decoded token surviving). NoOp on
            // CPU / CUDA backends and on host OpenCL builds whose
            // `cvt_noshuffle` program failed to compile.
            //
            // AUF SOA bypass (Phase 3.7b): when the secondary source is an
            // AUF file with WEIGHTS_ADRENO_SOA, the bytes are already in SOA
            // layout. `ensure_noshuffle_soa_registered` is still called but
            // the backend registers directly without re-conversion. In this
            // case `soa_reconvert_ms` records only the registration overhead
            // (≈ 0 ms). `StageBreakdown::soa_reconvert_ms ≈ 0` signals the
            // fast-path to the caller.
            //
            // Ordering rationale: this MUST happen after `invalidate_*` (so
            // we re-populate a clean registry) and before the generation bump
            // (so the next forward seeing the new generation finds a fully
            // populated registry, eliminating any window where lookup misses).
            let t_d0 = Instant::now();
            let skip_soa_reconvert = secondary_mmap
                .map(|s| s.is_pre_converted_soa())
                .unwrap_or(false);
            if skip_soa_reconvert {
                // AUF WEIGHTS_ADRENO_SOA path: weights are already SOA-backed
                // at materialise time (`materialise_auf_soa_weight` →
                // `Backend::alloc_pre_converted_soa_tensor`). The
                // per-batch `invalidate_noshuffle_soa_registry()` above
                // wiped every entry — including the just-built ones — so we
                // re-register them here against the same `d_buf` keys via
                // `restore_pre_converted_soa_registration`. Tensors whose
                // buffer is *not* a `NoshuffleWeightBuffer` (e.g. AOS
                // fallback inside `materialise_auf_soa_weight`) re-enter
                // the GGUF SOA conversion path through the trait default.
                //
                // `soa_reconvert_ms` charge here is the registration cost
                // only (~µs / tensor). No host-side conversion runs and no
                // new cl_mem is allocated — this is the Phase 5 Sprint C
                // (WSWAP-5-AUF-PLACEHOLDER-DROP) shape of the bypass path.
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
                        if let Err(e) = self.backend.restore_pre_converted_soa_registration(tensor)
                        {
                            return Err(SwapError::BufferAllocationFailed {
                                layer: swapped.layer_idx,
                                source: e,
                            });
                        }
                    }
                }
            } else {
                // GGUF path: standard SOA re-conversion.
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
            }
            stages.soa_reconvert_ms += t_d0.elapsed().as_secs_f64() * 1e3;

            // ── Stage (e-post): ratio_generation bump ─────────────────────────
            let t_e1 = Instant::now();
            let new_gen = ratio_generation.fetch_add(1, Ordering::SeqCst) + 1;
            stages.gen_bump_ms += t_e1.elapsed().as_secs_f64() * 1e3;
            report.ratio_generation_after = Some(new_gen);

            report.stage_breakdown = Some(stages);
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
    ///
    /// **AUF SOA fast path** (`secondary.is_pre_converted_soa()`):
    /// Q4_0 weights are routed through `materialise_auf_soa_weight` which
    /// delegates to `Backend::alloc_pre_converted_soa_tensor`. That call
    /// allocates the SOA `q_buf`/`d_buf`/optional `q_img` cl_mem, uploads
    /// the AUF payload, registers a noshuffle SOA entry keyed on the
    /// `d_buf` address, and returns a tensor whose backing buffer is a
    /// `NoshuffleWeightBuffer`. **No AOS placeholder cl_mem is allocated**
    /// (Phase 5 Sprint C — WSWAP-5-AUF-PLACEHOLDER-DROP). The Stage (d)
    /// `restore_pre_converted_soa_registration` re-registers the entry
    /// after the per-batch `invalidate_noshuffle_soa_registry()` clear so
    /// the ENG-ALG-221 ordering (clear → register → bump) is preserved.
    /// Non-Q4_0 layer tensors (none for the current swap target set) and
    /// non-OpenCL backends fall back to the AOS path.
    fn build_layer_from_mmap(
        &self,
        secondary: &Arc<SecondaryMmap>,
        slot: &LayerSlot,
        layer_idx: usize,
    ) -> Result<TransformerLayer, SwapError> {
        let old = slot.load_weights();

        let wq = self.materialise_weight(secondary, layer_idx, "attn_q.weight", &old.wq)?;
        let wk = self.materialise_weight(secondary, layer_idx, "attn_k.weight", &old.wk)?;
        let wv = self.materialise_weight(secondary, layer_idx, "attn_v.weight", &old.wv)?;
        let wo = self.materialise_weight(secondary, layer_idx, "attn_output.weight", &old.wo)?;
        let w_gate =
            self.materialise_weight(secondary, layer_idx, "ffn_gate.weight", &old.w_gate)?;
        let w_up = self.materialise_weight(secondary, layer_idx, "ffn_up.weight", &old.w_up)?;
        let w_down =
            self.materialise_weight(secondary, layer_idx, "ffn_down.weight", &old.w_down)?;

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

    /// Dispatch a swap-target *weight* tensor through the right materialise
    /// path based on the secondary source format.
    ///
    /// - GGUF / non-pre-converted secondaries → standard `materialise_tensor`
    ///   (AOS bytes, runtime Q/K unpermute, runtime SOA reconversion).
    /// - AUF `WEIGHTS_ADRENO_SOA` (Phase 3.7b SOA bypass) → `materialise_auf_soa_weight`
    ///   which uploads pre-converted SOA bytes into a key-holder `cl_mem` of
    ///   the AOS shape. Q/K permutation is **already applied** at AUF build
    ///   time (`auf_tool::extract_weight_blobs`), so the runtime path is a
    ///   straight zero-copy upload. Non-Q4_0 entries (none for the current
    ///   swap-target set) fall back to the AOS path.
    fn materialise_weight(
        &self,
        secondary: &Arc<SecondaryMmap>,
        layer_idx: usize,
        subname: &str,
        primary: &Tensor,
    ) -> Result<Tensor, SwapError> {
        // AUF SOA fast path is OpenCL-only and Q4_0-only. The dtype gate
        // checks the *target* dtype (post-swap), not the primary — when
        // swapping F16→Q4_0 the primary is F16 but the AUF SOA bypass
        // applies (it converts the secondary's pre-baked SOA payload into a
        // Q4_0 NoshuffleWeightBuffer). Falling back through
        // `materialise_tensor` for any mismatch keeps the host-test surface
        // (CPU backend, non-Q4_0 norms, etc.) on the original code path.
        let secondary_info = secondary.layer_tensor(layer_idx, subname);
        let secondary_dtype_is_q4_0 = secondary_info.is_some_and(|i| i.dtype == DType::Q4_0);
        if secondary.is_pre_converted_soa()
            && !self.backend.name().contains("CPU")
            && self.target_dtype == DType::Q4_0
            && secondary_dtype_is_q4_0
            && let Some(t) =
                self.materialise_auf_soa_weight(secondary, layer_idx, subname, primary)?
        {
            return Ok(t);
        }
        self.materialise_tensor(secondary, layer_idx, subname, primary, true)
    }

    /// Build a SOA-backed GPU tensor for an AUF `WEIGHTS_ADRENO_SOA` Q4_0
    /// weight without running the AOS materialisation pipeline and without
    /// allocating a placeholder AOS cl_mem.
    ///
    /// The returned `Tensor` reports the logical AOS shape and `Q4_0` dtype
    /// but its backing buffer is a `NoshuffleWeightBuffer` containing the
    /// `q_buf` / `d_buf` (and optional `q_img`) cl_mem handles. The AUF
    /// SOA payload (`q_bytes` || `d_bytes`) is uploaded directly into those
    /// fresh GPU buffers via blocking `enqueue_write_buffer`; no AOS
    /// placeholder is created (Phase 5 Sprint C
    /// WSWAP-5-AUF-PLACEHOLDER-DROP — saves 112 cl_mem / ~547 MiB on
    /// Llama 3.2 1B ratio=1.0 mixed).
    ///
    /// `Backend::alloc_pre_converted_soa_tensor` also registers a noshuffle
    /// SOA entry against the `d_buf` cl_mem address so the per-layer
    /// `matmul_q4_0` lookup hits in the steady state. The Stage (d)
    /// `restore_pre_converted_soa_registration` re-installs the entry after
    /// the per-batch `invalidate_noshuffle_soa_registry()` call so the
    /// invariant ordering (clear → register → bump) is preserved.
    ///
    /// Returns `Ok(None)` when:
    /// - The tensor is missing from the secondary index (caller falls back
    ///   to the GGUF AOS path which surfaces a proper error).
    /// - `split_pre_converted_soa` rejects the payload (non-Q4_0 dtype or
    ///   size not divisible by 18).
    /// - The size guards inferred from the primary tensor shape do not match
    ///   the AUF payload (defensive — any mismatch triggers AOS fallback).
    /// - The backend's `alloc_pre_converted_soa_tensor` returns `None`
    ///   (CPU / non-OpenCL / driver-side cvt program missing) — caller
    ///   falls back to `materialise_tensor`.
    fn materialise_auf_soa_weight(
        &self,
        secondary: &Arc<SecondaryMmap>,
        layer_idx: usize,
        subname: &str,
        primary: &Tensor,
    ) -> Result<Option<Tensor>, SwapError> {
        let info = secondary.layer_tensor(layer_idx, subname).ok_or_else(|| {
            SwapError::SecondaryTensorMissing {
                layer: layer_idx,
                subname: subname.to_string(),
            }
        })?;
        let expected_dims: Vec<usize> = primary.shape().dims().to_vec();
        let sec_rev: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
        if sec_rev != expected_dims {
            return Err(SwapError::ShapeMismatch {
                layer: layer_idx,
                subname: subname.to_string(),
                primary: expected_dims,
                secondary: info.dims.clone(),
            });
        }
        // Split AUF payload into q_bytes / d_bytes. None → fall back to AOS path.
        let Some((q_bytes, d_bytes)) = secondary.split_pre_converted_soa(info) else {
            return Ok(None);
        };
        // Total elem count must be QK4_0-aligned; the SOA upload contract
        // requires exactly `num_blocks * 16` (q) + `num_blocks * 2` (d).
        let shape = primary.shape().clone();
        let dims = shape.dims();
        if dims.len() != 2 {
            return Ok(None);
        }
        let total_elems: usize = dims.iter().product();
        if !total_elems.is_multiple_of(32) {
            return Ok(None);
        }
        // Logical: dims[0] = ne01 (rows), dims[1] = ne00 (cols).
        let (ne01, ne00) = (dims[0], dims[1]);
        let num_blocks = ne01 * ne00 / 32;
        if q_bytes.len() != num_blocks * 16 || d_bytes.len() != num_blocks * 2 {
            return Ok(None);
        }

        // Delegate to the backend. This allocates two cl_mem (q_buf, d_buf)
        // + optional image1d view, uploads the SOA bytes verbatim, registers
        // a noshuffle SOA entry keyed on d_buf, and returns a tensor whose
        // backing is a `NoshuffleWeightBuffer` owning the same handles. No
        // placeholder AOS cl_mem is allocated.
        let opt = self
            .backend
            .alloc_pre_converted_soa_tensor(shape, q_bytes, d_bytes, ne00, ne01)
            .map_err(|e| SwapError::BufferAllocationFailed {
                layer: layer_idx,
                source: e,
            })?;
        Ok(opt)
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

    /// WSWAP-5-PRIMARY-DROP: explicitly release the primary weight cl_mem
    /// backing the `LayerWeights` we just displaced.
    ///
    /// `old_layer` is moved by value (consumed via `Arc::try_unwrap`), so
    /// dropping it releases every owned `Arc<dyn Buffer>` whose refcount
    /// reaches zero. For OpenCL backends this triggers the
    /// `OpenCLBuffer::drop` destructor which calls `clReleaseMemObject` and
    /// returns the GPU memory to the driver. For CPU backends this is a
    /// regular heap free.
    ///
    /// Behaviour summary:
    /// - On Adreno UMA + AUF SOA bypass at ratio=1.0: each swapped layer
    ///   releases 7 F16 weight cl_mem (~17 MiB / layer for Llama 3.2 1B).
    ///   Total: ~16 layers × 7 × 17 MiB ≈ 2.0 GiB returned.
    /// - On CPU backend / host tests: heap free; behaviourally identical.
    /// - Norms (`attention_norm`, `ffn_norm`) are intentionally also
    ///   dropped — the new layer always materialises fresh norms either
    ///   from the secondary mmap or by cloning, so the old ones are
    ///   redundant. Optional Gemma3 norms (`q_norm`, `k_norm`,
    ///   `pre_ffn_norm`, `post_ffn_norm`) and `qkv_bias` are also released;
    ///   the new layer rebuilds them from the secondary or clones them.
    ///
    /// Diagnostic: emits a single `weight_swap_released` bucket count via
    /// `record_cl_mem_release` so a `LLMRS_CL_MEM_DIAG=1` dump can confirm
    /// the release path executed (replaces the implicit "destructor was
    /// not instrumented" gap noted in `phase_5_tbt_diag.md`).
    fn release_primary_weights(backend: &Arc<dyn Backend>, old_layer: LayerWeights) {
        // Tally per-tensor sizes before drop so the diagnostic tracks the
        // real reclaimed footprint regardless of which buffer concrete type
        // backed each tensor (OpenCLBuffer / UnifiedBuffer / SharedBuffer /
        // MmapBuffer / NoshuffleWeightBuffer).
        let mut released_bytes: usize = 0;
        let mut released_count: usize = 0;
        let mut tally = |t: &Tensor| {
            released_bytes += t.size();
            released_count += 1;
        };
        tally(&old_layer.wq);
        tally(&old_layer.wk);
        tally(&old_layer.wv);
        tally(&old_layer.wo);
        tally(&old_layer.w_gate);
        tally(&old_layer.w_up);
        tally(&old_layer.w_down);
        // Norms + biases are smaller but still counted so the dump reflects
        // the full primary layer cost. Most paths will leave these on F32
        // primaries (they aren't quantised in the secondary), so tracking
        // is informational rather than load-bearing.
        tally(&old_layer.attention_norm);
        tally(&old_layer.ffn_norm);
        if let Some(bias) = &old_layer.qkv_bias {
            tally(&bias.bq);
            tally(&bias.bk);
            tally(&bias.bv);
        }
        if let Some(t) = &old_layer.q_norm {
            tally(t);
        }
        if let Some(t) = &old_layer.k_norm {
            tally(t);
        }
        if let Some(t) = &old_layer.pre_ffn_norm {
            tally(t);
        }
        if let Some(t) = &old_layer.post_ffn_norm {
            tally(t);
        }
        // Tensor partition state on the OLD layer points at obsolete cl_mem
        // (the new layer uses `partition_ctx: None` per `build_layer_from_mmap`),
        // so dropping `old_layer` here also reclaims any partition slice
        // buffers — accounted under the same bucket for simplicity.

        // Drop fires destructors on every Arc<dyn Buffer> whose refcount
        // reaches zero. For OpenCL backends this is `clReleaseMemObject`.
        drop(old_layer);

        // Diagnostic charge — only OpenCLBackend exposes the diag hook;
        // CPU/CUDA backends quietly skip via the `as_any` downcast.
        record_swap_release(backend, released_count, released_bytes);
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

/// WSWAP-5-PRIMARY-DROP: charge the OpenCL diagnostic counter for primary
/// weight cl_mem released via `release_primary_weights`. NoOp on CPU /
/// CUDA backends and on builds without `LLMRS_CL_MEM_DIAG=1`.
///
/// Records a single bucket entry `weight_swap_released` regardless of which
/// concrete buffer type backed each tensor; the per-tensor classification
/// already lives in the alloc-side counters (`weight_f16_copy`,
/// `weight_q4_aos_copy`, etc.). The release-side tally lets a diag run
/// produce a closed-loop "alloc - release" picture without per-destructor
/// instrumentation (Sprint B `phase_5_tbt_diag.md` over-reported alive
/// bytes because the dump only counted allocs).
fn record_swap_release(backend: &Arc<dyn Backend>, count: usize, bytes: usize) {
    if count == 0 {
        return;
    }
    #[cfg(feature = "opencl")]
    {
        if let Some(ocl_be) = backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
        {
            // The diag bucket is keyed on a `&'static str`; we record one
            // bucket entry per swap layer with the cumulative byte tally.
            // `record_cl_mem_release` is a no-op when the env var is unset,
            // so we don't pay a HashMap update per swap on production runs.
            for _ in 0..count {
                ocl_be.record_cl_mem_release("weight_swap_released", bytes / count);
            }
            return;
        }
    }
    // CPU / CUDA / non-opencl builds: nothing to record.
    let _ = (backend, bytes);
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

    #[test]
    fn stage_breakdown_log_line_includes_prefault_stage() {
        // WSWAP-5-COLD-UNIFORM: the stage breakdown log line must surface the
        // prefault stage so device measurements can plot it next to the other
        // stages. Refactors that drop the prefix will trip this guard.
        let stages = StageBreakdown {
            prefault_ms: 12.34,
            mmap_permute_ms: 100.0,
            arc_swap_ms: 0.5,
            madvise_ms: 0.1,
            soa_reconvert_ms: 50.0,
            gen_bump_ms: 0.2,
        };
        let line = stages.to_log_line();
        assert!(
            line.starts_with("prefault=12.3ms"),
            "log line must lead with prefault stage, got: {line}"
        );
        assert!(
            line.contains("mmap_permute=100.0ms"),
            "log line must contain mmap_permute stage, got: {line}"
        );
    }

    #[test]
    fn stage_breakdown_prefault_default_zero() {
        // Default-constructed StageBreakdown (e.g. CPU host tests, no-op
        // secondaries) must report prefault=0.0 so downstream parsers do not
        // see a missing field.
        let stages = StageBreakdown::default();
        assert_eq!(stages.prefault_ms, 0.0);
    }

    // ── WSWAP-5-PRIMARY-DROP: explicit primary cl_mem release ─────────────
    //
    // The Sprint C-2 contract is: after `LayerSlot::swap_weights` returns
    // the previous Arc, the swap path can `try_unwrap` it deterministically
    // so the tensor destructors fire and the primary `cl_mem` is reclaimed.
    // The host-side proxy for this behaviour is the buffer Arc strong count:
    // each old weight tensor's `Arc<dyn Buffer>` must reach refcount 1
    // (the LayerWeights itself is the sole holder) when the swap path takes
    // ownership. Combined with `Arc::try_unwrap` on the LayerWeights, this
    // implies every backing buffer is uniquely owned by the dispatcher and
    // its destructor will run on `drop(layer)`.

    use crate::buffer::shared_buffer::SharedBuffer;
    use crate::core::backend::Backend;
    use crate::core::shape::Shape;
    use crate::layers::transformer_layer::TransformerLayer;
    use crate::models::weights::LayerSlot;
    use std::sync::Arc;

    fn build_test_layer(be: &Arc<dyn Backend>) -> TransformerLayer {
        // Build a minimal `TransformerLayer` whose weight tensors each carry
        // a fresh `Arc<dyn Buffer>` so we can probe refcounts after a swap.
        let make = |sz: usize| -> Tensor {
            let buf: Arc<dyn crate::core::buffer::Buffer> =
                Arc::new(SharedBuffer::new(sz, DType::F32));
            Tensor::new(Shape::new(vec![sz / 4]), buf, be.clone())
        };
        TransformerLayer {
            wq: make(32),
            wk: make(32),
            wv: make(32),
            wo: make(32),
            w_gate: make(32),
            w_up: make(32),
            w_down: make(32),
            attention_norm: make(8),
            ffn_norm: make(8),
            qkv_bias: None,
            q_norm: None,
            k_norm: None,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            partition_ctx: None,
        }
    }

    #[test]
    fn slot_swap_weights_returns_previous_arc_uniquely_owned() {
        // ENG-ALG-211 step (b/c) refined contract: the returned Arc has
        // refcount==1 (no extant snapshots, swap dispatcher runs strictly
        // between forwards). `Arc::try_unwrap` therefore succeeds on the
        // production path and the inner `LayerWeights` can be dropped to
        // release every buffer Arc. This mirrors the WSWAP-5-PRIMARY-DROP
        // primary cl_mem release semantics on the device.
        let be: Arc<dyn Backend> = Arc::new(crate::backend::cpu::CpuBackend::new());
        let initial = Arc::new(build_test_layer(&be));
        let slot = LayerSlot::new(
            (*initial).clone(),
            DType::F32,
            None, // No secondary mmap; not exercised by this test.
        );
        // Capture the wq buffer Arc from the initial install.
        let initial_wq_buf = Arc::clone(slot.load_weights().wq.buffer());
        let initial_wq_count_before = Arc::strong_count(&initial_wq_buf);
        drop(initial); // release builder snapshot — slot still holds one ref

        // Pre-swap: slot holds one Arc<LayerWeights>; load_weights briefly
        // borrows another, but we drop it immediately.
        // After swap: returned Arc must be the only outstanding reference.
        let new_layer = build_test_layer(&be);
        let returned = slot.swap_weights(Arc::new(new_layer), DType::F32);
        assert_eq!(
            Arc::strong_count(&returned),
            1,
            "WSWAP-5-PRIMARY-DROP contract: swap_weights must return the \
             previous Arc with strong_count==1 so the dispatcher can \
             try_unwrap and release primary cl_mem deterministically",
        );

        // try_unwrap must succeed.
        let inner = match Arc::try_unwrap(returned) {
            Ok(layer) => layer,
            Err(_) => panic!("Arc::try_unwrap must succeed on a uniquely-owned snapshot"),
        };

        // After try_unwrap and before drop, the wq buffer Arc has 2 refs
        // (inner.wq.buffer + the local capture above).
        let post_unwrap = Arc::strong_count(&initial_wq_buf);
        assert_eq!(
            post_unwrap, 2,
            "buffer Arc held by both inner LayerWeights and the local capture"
        );

        // Drop inner — every buffer Arc decrements to 1 (only the local
        // capture remains for `wq`). Other tensors had no external capture
        // so their buffer Arcs go to 0 and the destructors run.
        drop(inner);
        assert_eq!(
            Arc::strong_count(&initial_wq_buf),
            1,
            "after drop(inner): only the local probe holds the wq buffer; \
             a real OpenCL buffer would now be reclaimed",
        );
        // initial_wq_count_before is informational; assert it sanity-checks.
        assert!(initial_wq_count_before >= 1);
    }

    #[test]
    fn release_primary_weights_runs_destructors() {
        // Direct test of the SwapExecutor::release_primary_weights helper:
        // dropping the LayerWeights must drop every contained Tensor and
        // therefore decrement every buffer Arc. We verify by capturing weak
        // references to the buffers and ensuring they all upgrade to None
        // after the helper consumes the layer.
        use std::sync::Weak;

        let be: Arc<dyn Backend> = Arc::new(crate::backend::cpu::CpuBackend::new());
        let layer = build_test_layer(&be);
        // Capture weak refs to the 7 weight buffers.
        let weaks: Vec<Weak<dyn crate::core::buffer::Buffer>> = [
            &layer.wq,
            &layer.wk,
            &layer.wv,
            &layer.wo,
            &layer.w_gate,
            &layer.w_up,
            &layer.w_down,
        ]
        .iter()
        .map(|t| Arc::downgrade(t.buffer()))
        .collect();

        SwapExecutor::release_primary_weights(&be, layer);

        for (i, w) in weaks.iter().enumerate() {
            assert!(
                w.upgrade().is_none(),
                "WSWAP-5-PRIMARY-DROP: buffer {i} must be fully released \
                 after release_primary_weights — Weak::upgrade returning \
                 Some implies a lingering Arc reference",
            );
        }
    }
}
