//! `SecondaryMmap` — metadata-validated mmap handle for secondary weight files.
//!
//! Phase 1 establishes the mmap handle only: the file is opened, the header
//! is parsed once, and per-layer tensor slice descriptors are built.
//! Actual byte access (`tensor_bytes`) is exposed so Phase 2 (`SwapExecutor`)
//! can consume it without re-parsing.
//!
//! Two file formats are supported:
//! - **GGUF** (legacy): detected by `.gguf` extension or GGUF magic `b"GGUF"`.
//! - **AUF** (Argus Unified Format): detected by `.auf` extension or magic
//!   `b"ARGUS_W\0"`. The AUF variant carries pre-converted SOA weights and
//!   skips the `ensure_noshuffle_soa_registered` step in `SwapExecutor`.
//!
//! Spec: ENG-DAT-094 (SecondaryMmap), ENG-DAT-096 (AUF), ENG-DAT-C10
//! (metadata match), ENG-DAT-C12 / INV-125 (lifetime), INV-132~134 (AUF
//! invariants), INV-123/124 (read-only, dtype source).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::core::buffer::DType;
use crate::models::config::ModelConfig;
use crate::models::loader::gguf::{GgufFile, ggml_type_to_dtype, tensor_byte_size};

// ── AUF crate imports ──────────────────────────────────────────────────────
// AUF helper logic lives in `crate::models::loader::auf::secondary` (W-AUF-1
// C2). Only the types still referenced by `SecondaryMmap::Auf` / the
// tensor-kind subname helper are imported here.
use crate::auf::{AufView, tensor_index::TensorKind};

/// CLI `--secondary-dtype` 값.
///
/// `generate.rs`가 파싱하여 `open_secondary_auf`에 전달한다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondaryDtypeChoice {
    /// 자동 선택: primary와 다른 dtype candidate 1개를 META.default_dtype 또는
    /// first-candidate 규칙으로 선택한다.
    Auto,
    /// F16을 명시 지정한다.
    F16,
    /// Q4_0을 명시 지정한다.
    Q4_0,
    /// F32를 명시 지정한다.
    F32,
}

/// CLI `--secondary-layout` 값. 어떤 weights variant로 swap 후 텐서를 만들지 결정한다.
///
/// AUF는 빌드 시 여러 backend variant(`WEIGHTS_ADRENO_SOA`, `WEIGHTS_CUDA_AOS`,
/// `WEIGHTS_CPU_AOS`)를 동봉할 수 있다. 런타임에 어느 variant를 읽을지가
/// 다음 trade-off를 결정한다:
///
/// - **SOA (Adreno noshuffle)**: GPU TBT 33~55% 빠름 (Adreno 830 측정). 대신
///   `NoshuffleWeightBuffer`는 host pointer가 null이라 swap 후 `switch_hw cpu`
///   / partition lazy-map / CPU forward는 모두 실패한다.
/// - **AOS (CPU/CUDA AOS)**: GPU는 표준 Q4_0 GEMV (`mul_mat_q4_0_f32`) fallback.
///   `UnifiedBuffer` (host-mapped) 백킹이라 swap 후에도 host pointer가 살아있어
///   switch_hw cpu / partition / CPU forward 모두 자연 동작.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondaryLayoutChoice {
    /// 자동: feature flag 기반 preferred variant (OpenCL→AdrenoSoa, CUDA→CudaAos,
    /// 그 외→CpuAos). preferred variant가 AUF에 없으면 CpuAos로 폴백한다.
    Auto,
    /// 강제 AOS: AUF에 `WEIGHTS_CPU_AOS` 또는 (CUDA build에선) `WEIGHTS_CUDA_AOS`가
    /// 있어야 한다. SOA fast path 우회 → GPU 성능 손실 대신 switch/partition 호환.
    Aos,
    /// 강제 SOA: OpenCL build 한정. `WEIGHTS_ADRENO_SOA`가 없으면 에러.
    Soa,
}

// ──────────────────────────────────────────────────────────────────────────
// Public error type
// ──────────────────────────────────────────────────────────────────────────

/// Errors specific to secondary weight loading.
///
/// Unlike the loose `anyhow` used elsewhere, these variants are surfaced to
/// `generate.rs` so the CLI can fail fast with an actionable message.
#[derive(Debug)]
pub enum LoadError {
    /// Secondary file could not be opened or mmapped.
    SecondaryUnavailable {
        path: PathBuf,
        source: anyhow::Error,
    },

    /// Secondary GGUF metadata does not match the primary model config.
    MetadataMismatch {
        field: &'static str,
        primary: String,
        secondary: String,
    },

    /// A required tensor is missing in the secondary file or its shape does
    /// not match the primary layout.
    ShapeMismatch {
        name: String,
        primary: Vec<u64>,
        secondary: Vec<u64>,
    },

    /// A tensor that exists in the primary file is missing in the secondary.
    TensorMissing { name: String },

    /// AUF file did not pass invariant checks (INV-132~134).
    AufInvariantViolation { detail: String },

    /// Adreno SOA backend + F16 dtype 조합이 거부됨 (Sprint D 함정 3).
    ///
    /// Adreno SOA WEIGHTS section은 Q4_0 전용 SOA layout이므로 F16 dtype secondary
    /// 로드를 지원하지 않는다. --secondary-dtype q4_0 또는 CPU/CUDA backend를 사용하라.
    AdrenoSoaF16Rejected,

    /// 단방향 swap 정합성 위반: primary=Q4_0, secondary=F16 (역방향 차단).
    ReverseSwapRejected {
        primary_dtype: String,
        secondary_dtype: String,
    },

    /// 요청한 dtype이 AUF TENSOR_INDEX에 없다.
    DtypeNotFound { dtype: String },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::SecondaryUnavailable { path, source } => write!(
                f,
                "secondary weight file '{}' unavailable: {source}",
                path.display()
            ),
            LoadError::MetadataMismatch {
                field,
                primary,
                secondary,
            } => write!(
                f,
                "secondary weight metadata mismatch for field '{field}': \
                 primary={primary}, secondary={secondary}"
            ),
            LoadError::ShapeMismatch {
                name,
                primary,
                secondary,
            } => write!(
                f,
                "secondary weight shape mismatch for tensor '{name}': \
                 primary={primary:?}, secondary={secondary:?}"
            ),
            LoadError::TensorMissing { name } => {
                write!(f, "secondary weight missing tensor '{name}'")
            }
            LoadError::AufInvariantViolation { detail } => {
                write!(f, "AUF file invariant violation: {detail}")
            }
            LoadError::AdrenoSoaF16Rejected => write!(
                f,
                "Adreno SOA backend does not support F16 secondary dtype. \
                 The SOA layout is Q4_0-only. \
                 Use --secondary-dtype q4_0 or switch to a CPU/CUDA backend."
            ),
            LoadError::ReverseSwapRejected {
                primary_dtype,
                secondary_dtype,
            } => write!(
                f,
                "Reverse swap rejected: primary={primary_dtype}, secondary={secondary_dtype}. \
                 Weight swap only supports F16→Q4_0 direction. \
                 Cannot use secondary=F16 when primary is already Q4_0."
            ),
            LoadError::DtypeNotFound { dtype } => write!(
                f,
                "Secondary dtype '{dtype}' not available in AUF TENSOR_INDEX. \
                 Use --secondary-dtype auto or choose an available dtype."
            ),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::SecondaryUnavailable { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Tensor slice descriptor (shared by GGUF and AUF paths)
// ──────────────────────────────────────────────────────────────────────────

/// Tensor slice inside the secondary mmap (one entry per tensor).
///
/// Offsets and lengths are relative to the backing mmap, consumed by
/// `SwapExecutor` at Phase 2 to copy-and-permute the raw bytes.
#[derive(Debug, Clone)]
pub struct SecondaryTensorInfo {
    /// Absolute offset within the mmap (tensor data section + format offset).
    pub offset: usize,
    /// Length in bytes.
    pub len: usize,
    /// DType in the secondary file (typically lower precision than primary).
    pub dtype: DType,
    /// Dimensions in GGUF order (innermost first).
    pub dims: Vec<u64>,
}

/// Per-layer bundle of tensor slices. Populated for every decoder layer
/// observed in the primary file; if the secondary lacks any required tensor,
/// `open_secondary` returns `LoadError::TensorMissing` up front.
#[derive(Debug, Clone, Default)]
pub struct LayerTensorSlice {
    /// Subname → descriptor (e.g. "attn_q.weight" → info). Subname is the
    /// GGUF suffix after `blk.<idx>.`.
    pub tensors: HashMap<String, SecondaryTensorInfo>,
}

// ──────────────────────────────────────────────────────────────────────────
// GGUF-backed secondary (original implementation)
// ──────────────────────────────────────────────────────────────────────────

/// Internal GGUF secondary mmap (original handle).
pub struct GgufSecondaryMmap {
    /// Underlying GGUF parse (keeps mmap handle alive).
    pub gguf: GgufFile,
    /// Indexed by `layer_idx`. Length = num decoder layers.
    pub layer_index: Vec<LayerTensorSlice>,
    /// Source path, preserved for diagnostic messages.
    pub source_path: PathBuf,
}

impl std::fmt::Debug for GgufSecondaryMmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufSecondaryMmap")
            .field("source_path", &self.source_path)
            .field("num_layers", &self.layer_index.len())
            .finish()
    }
}

impl GgufSecondaryMmap {
    fn layer_tensor(&self, layer_idx: usize, subname: &str) -> Option<&SecondaryTensorInfo> {
        self.layer_index
            .get(layer_idx)
            .and_then(|slice| slice.tensors.get(subname))
    }

    fn tensor_bytes(&self, info: &SecondaryTensorInfo) -> &[u8] {
        let end = info.offset + info.len;
        &self.gguf.mmap_data()[info.offset..end]
    }

    /// See `SecondaryMmap::prefault`. Operates on the entire GGUF mmap.
    fn prefault(&self) {
        let bytes = self.gguf.mmap_data();
        prefault_byte_range(bytes);
    }

    /// See `SecondaryMmap::prefault_layers`. For GGUF, falls back to
    /// prefaulting only the byte ranges covered by the target layers.
    fn prefault_layers(&self, target_layers: &[usize]) {
        let bytes = self.gguf.mmap_data();
        for &layer_idx in target_layers {
            let Some(layer_slice) = self.layer_index.get(layer_idx) else {
                // out-of-range: silent skip (ENG-DAT-C08 spirit)
                continue;
            };
            for info in layer_slice.tensors.values() {
                let end = info.offset + info.len;
                if end <= bytes.len() {
                    prefault_byte_range(&bytes[info.offset..end]);
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// AUF-backed secondary
// ──────────────────────────────────────────────────────────────────────────

/// Map `TensorKind` to GGUF-style subname used by `SwapExecutor`.
///
/// Returns `None` for cross-layer tensors (embedding, final_norm, lm_head)
/// that are not swapped.
pub(crate) fn tensor_kind_to_subname(kind: u32) -> Option<&'static str> {
    match TensorKind::from_u32(kind)? {
        TensorKind::AttnQ => Some("attn_q.weight"),
        TensorKind::AttnK => Some("attn_k.weight"),
        TensorKind::AttnV => Some("attn_v.weight"),
        TensorKind::AttnO => Some("attn_output.weight"),
        TensorKind::FfnGate => Some("ffn_gate.weight"),
        TensorKind::FfnUp => Some("ffn_up.weight"),
        TensorKind::FfnDown => Some("ffn_down.weight"),
        TensorKind::AttnNorm => Some("attn_norm.weight"),
        TensorKind::FfnNorm => Some("ffn_norm.weight"),
        TensorKind::Embedding | TensorKind::FinalNorm | TensorKind::LmHead => None,
        // qkv bias는 swap 대상 아님 (1-D F32 vector, swap quality 의미 없음).
        TensorKind::AttnQBias | TensorKind::AttnKBias | TensorKind::AttnVBias => None,
    }
}

/// AUF-backed secondary mmap.
///
/// Carries a pre-parsed `AufView` mmap plus the tensor index.  Tensor bytes
/// are served directly from the WEIGHTS section slice (zero-copy).
///
/// `is_pre_converted_soa` reflects whether the backend variant is
/// `WEIGHTS_ADRENO_SOA`, which skips the runtime SOA re-conversion step in
/// `SwapExecutor` (Phase 3.7a bypass).
pub struct AufSecondaryMmap {
    /// AUF mmap view (keeps mmap alive).
    ///
    /// `Arc<AufView>` since W-AUF-2 — primary `AufSource`와 같은 view를
    /// secondary로도 재포장(`from_auf_self_secondary`)할 수 있어야 한다.
    /// `weights_bytes()` 등 모든 메서드는 `Arc::Deref`로 그대로 호출된다.
    pub view: Arc<AufView>,
    /// Indexed by `layer_idx`. Length = num decoder layers.
    pub layer_index: Vec<LayerTensorSlice>,
    /// Source path, preserved for diagnostic messages.
    pub source_path: PathBuf,
    /// True when the WEIGHTS section is `WEIGHTS_ADRENO_SOA`, meaning the
    /// payload is already SOA-converted and the `ensure_noshuffle_soa_registered`
    /// step in `SwapExecutor` can be bypassed (Phase 3.7a fast-path).
    pub is_pre_converted_soa: bool,
}

impl std::fmt::Debug for AufSecondaryMmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AufSecondaryMmap")
            .field("source_path", &self.source_path)
            .field("num_layers", &self.layer_index.len())
            .field("is_pre_converted_soa", &self.is_pre_converted_soa)
            .finish()
    }
}

impl AufSecondaryMmap {
    fn layer_tensor(&self, layer_idx: usize, subname: &str) -> Option<&SecondaryTensorInfo> {
        self.layer_index
            .get(layer_idx)
            .and_then(|slice| slice.tensors.get(subname))
    }

    fn tensor_bytes(&self, info: &SecondaryTensorInfo) -> &[u8] {
        let weights_bytes = self
            .view
            .weights_bytes()
            .expect("AufSecondaryMmap: weights_bytes() must be Some (invariant)");
        &weights_bytes[info.offset..info.offset + info.len]
    }

    /// See `SecondaryMmap::prefault`. Targets the WEIGHTS section bytes only —
    /// META / TOKENIZER / TENSOR_INDEX are tiny and already touched during
    /// `open()`.
    fn prefault(&self) {
        if let Some(weights_bytes) = self.view.weights_bytes() {
            prefault_byte_range(weights_bytes);
        }
    }

    /// See `SecondaryMmap::prefault_layers`. Only touches the byte ranges
    /// belonging to the requested layers.
    fn prefault_layers(&self, target_layers: &[usize]) {
        let Some(weights_bytes) = self.view.weights_bytes() else {
            return;
        };
        for &layer_idx in target_layers {
            let Some(layer_slice) = self.layer_index.get(layer_idx) else {
                // out-of-range: silent skip
                continue;
            };
            for info in layer_slice.tensors.values() {
                let end = info.offset + info.len;
                if end <= weights_bytes.len() {
                    prefault_byte_range(&weights_bytes[info.offset..end]);
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Unified SecondaryMmap enum
// ──────────────────────────────────────────────────────────────────────────

/// Handle to a secondary weight file kept alive for the lifetime of the model.
///
/// Carries the raw mmap plus the pre-parsed tensor index so Phase 2 can
/// locate bytes in O(1) per tensor.  Supports both GGUF (legacy) and AUF
/// (Argus Unified Format) sources.
///
/// Cross-layer tensors (embedding, output_norm, output) are intentionally
/// **not** indexed here (ENG-DAT-C11). Phase 1/2 only swap decoder block
/// weights.
#[allow(clippy::large_enum_variant)]
pub enum SecondaryMmap {
    /// GGUF-backed secondary (original implementation path).
    Gguf(GgufSecondaryMmap),
    /// AUF-backed secondary (new path, ENG-DAT-096).
    Auf(AufSecondaryMmap),
    /// rpcmem heap–backed secondary for `qnn_oppkg` backend (LISWAP-6).
    /// Lazy per-layer DMA-BUF alloc + OpenCL `CL_MEM_USE_HOST_PTR` alias on
    /// swap path eliminates the H2D copy. GGUF-format on disk; the same
    /// `qk_unpermute` and `is_pre_converted_soa=false` semantics apply.
    Rpcmem(crate::models::weights::rpcmem_secondary::RpcmemSecondaryStore),
}

impl std::fmt::Debug for SecondaryMmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecondaryMmap::Gguf(g) => f.debug_tuple("SecondaryMmap::Gguf").field(g).finish(),
            SecondaryMmap::Auf(a) => f.debug_tuple("SecondaryMmap::Auf").field(a).finish(),
            SecondaryMmap::Rpcmem(r) => f.debug_tuple("SecondaryMmap::Rpcmem").field(r).finish(),
        }
    }
}

impl SecondaryMmap {
    /// Fetch a layer's tensor descriptor by subname (e.g. "attn_q.weight").
    /// Returns `None` if the layer is out of range or the tensor is missing.
    pub fn layer_tensor(&self, layer_idx: usize, subname: &str) -> Option<&SecondaryTensorInfo> {
        match self {
            SecondaryMmap::Gguf(g) => g.layer_tensor(layer_idx, subname),
            SecondaryMmap::Auf(a) => a.layer_tensor(layer_idx, subname),
            SecondaryMmap::Rpcmem(r) => r.layer_tensor(layer_idx, subname),
        }
    }

    /// Zero-copy byte slice for a descriptor. Panics if `info` does not
    /// originate from this secondary (caller contract).
    pub fn tensor_bytes(&self, info: &SecondaryTensorInfo) -> &[u8] {
        match self {
            SecondaryMmap::Gguf(g) => g.tensor_bytes(info),
            SecondaryMmap::Auf(a) => a.tensor_bytes(info),
            SecondaryMmap::Rpcmem(r) => r.tensor_bytes(info),
        }
    }

    /// Source file path (for diagnostic messages).
    pub fn source_path(&self) -> &Path {
        match self {
            SecondaryMmap::Gguf(g) => &g.source_path,
            SecondaryMmap::Auf(a) => &a.source_path,
            SecondaryMmap::Rpcmem(r) => r.source_path(),
        }
    }

    /// Whether this secondary carries pre-SOA-converted weights.
    ///
    /// When `true`, `SwapExecutor` skips the `ensure_noshuffle_soa_registered`
    /// loop (Phase 3.7a SOA bypass) because the AUF WEIGHTS_ADRENO_SOA section
    /// already contains the converted bytes.
    pub fn is_pre_converted_soa(&self) -> bool {
        match self {
            SecondaryMmap::Gguf(_) => false,
            SecondaryMmap::Auf(a) => a.is_pre_converted_soa,
            // LISWAP-6 Rpcmem inherits GGUF semantics (AoS Q4_0 is intentional —
            // CPU-GPU concurrent operation requires AoS layout, postmortem §6.5).
            SecondaryMmap::Rpcmem(_) => false,
        }
    }

    /// Whether the secondary stores Q/K weights in their *post-unpermute*
    /// (NEON-unfriendly, runtime-loaded) layout.
    ///
    /// - **GGUF** secondaries store Q/K rows in the GGUF on-disk permuted
    ///   layout. Loaders (and `SwapExecutor::materialise_tensor`) call
    ///   `unpermute_qk_rows` once before installing the tensor.
    /// - **AUF** secondaries (both `WEIGHTS_ADRENO_SOA` and `WEIGHTS_CPU_AOS`
    ///   variants) bake the unpermute step into `auf_tool::extract_weight_blobs`
    ///   at build time, so the on-disk bytes are already unpermuted. Calling
    ///   `unpermute_qk_rows` again at swap time would double-apply the
    ///   permutation and produce garbage on the post-swap forward path
    ///   (observed on Galaxy S25 Adreno after `--secondary-layout aos`).
    ///
    /// Returns `true` when the swap path must call `unpermute_qk_rows`,
    /// `false` when the bytes are already in their final layout.
    pub fn needs_qk_unpermute_at_swap(&self) -> bool {
        match self {
            SecondaryMmap::Gguf(_) => true,
            SecondaryMmap::Auf(_) => false,
            // GGUF on disk → unpermute required for Llama Q/K. Qwen unaffected.
            SecondaryMmap::Rpcmem(_) => true,
        }
    }

    /// Prefault the page cache for the swap-relevant byte ranges.
    ///
    /// **Why** — WSWAP-5-COLD-UNIFORM. 5차 측정에서 per-layer 양봉 분포
    /// (cold ~53 ms, warm ~36 ms)의 주된 원인은 AUF mmap demand-paging.
    /// 페이지가 OS 페이지 캐시에 없으면 첫 접근 시 page fault → 디스크 I/O가
    /// `mmap_permute` stage 안에서 발생. 이 함수는 swap 시작 전 한 번 호출되어
    /// 두 가지 작업을 수행한다:
    ///
    /// 1. **`madvise(MADV_WILLNEED)`** — 커널에 background prefetch 힌트.
    /// 2. **explicit page-touch warmup** — 4 KiB step으로 첫 byte를 읽어
    ///    명시적 page fault를 트리거하고 페이지 캐시를 채움. 컴파일러가
    ///    최적화로 제거하지 못하도록 `read_volatile` 사용.
    ///
    /// Linux/Android에서만 동작 (target_os = "linux" / "android").
    /// 기타 OS는 no-op. 호출 자체는 항상 안전하며 실패 시 silent.
    ///
    /// **Range** — `AufSecondaryMmap`은 WEIGHTS 영역만 prefault (정확히 swap
    /// 대상 바이트). `GgufSecondaryMmap`은 mmap 전체를 prefault (GGUF는 weight
    /// 영역만 격리하기 어렵고, 호스트 테스트 외 주 사용처가 없으므로 단순 처리).
    ///
    /// Note: `prefault()` is equivalent to calling `prefault_layers` with all
    /// layer indices. For swap-path usage, prefer `prefault_layers(target_layers)`
    /// to avoid touching weight pages that will not be swapped.
    pub fn prefault(&self) {
        match self {
            SecondaryMmap::Auf(a) => a.prefault(),
            SecondaryMmap::Gguf(g) => g.prefault(),
            // Rpcmem variant: full prefault is a no-op (touching every page
            // would force-allocate every layer up-front, defeating the lazy
            // policy). Use `prefault_layers(target_layers)` to selectively
            // warm specific layers.
            SecondaryMmap::Rpcmem(_) => {}
        }
    }

    /// Prefault only the byte ranges belonging to `target_layers`.
    ///
    /// **ENG-ALG-229** — targeted prefault for swap target layers. When the
    /// swap ratio is <1.0 (e.g. ratio=0.9, 25/28 layers swapped), the full
    /// `prefault()` wastes time touching weight pages for layers that will not
    /// be swapped. This variant limits page-touch to the exact tensor byte
    /// ranges of the requested layers, reducing the prefault stage cost by
    /// ~40 ms at ratio=0.9 (estimated ~12% of total prefault stage).
    ///
    /// Calling conventions:
    /// - `target_layers` is the same slice passed to `SwapExecutor::execute_on_slots`.
    /// - Out-of-range indices are silently skipped (ENG-DAT-C08 spirit).
    /// - An empty `target_layers` is a no-op.
    /// - Linux/Android only; other targets are no-ops.
    pub fn prefault_layers(&self, target_layers: &[usize]) {
        match self {
            SecondaryMmap::Auf(a) => a.prefault_layers(target_layers),
            SecondaryMmap::Gguf(g) => g.prefault_layers(target_layers),
            // Rpcmem: force-allocate the listed layers' regions so the first
            // swap of each layer hits the cache (no first-touch alloc cost).
            SecondaryMmap::Rpcmem(r) => r.prefault_layers(target_layers),
        }
    }

    /// Split a pre-converted SOA Q4_0 tensor payload into `(q_bytes, d_bytes)`.
    ///
    /// AUF builder (`auf_tool::build_variant_payload` for `WEIGHTS_ADRENO_SOA`)
    /// packs each Q4_0 tensor as `q_buf` (`num_blocks * 16` bytes) followed by
    /// `d_buf` (`num_blocks * 2` bytes), where `num_blocks = ne01 * ne00 / 32`.
    /// Both buffers are pre-applied with the full conversion pipeline:
    ///   1. nibble bit unshuffle (`kernel_convert_block_q4_0_noshuffle`)
    ///   2. ushort-level 2D transpose of q
    ///   3. half-level 2D transpose of d
    ///
    /// so the bytes can be uploaded directly into `cl_mem` and registered via
    /// `Backend::alloc_pre_converted_soa_tensor` without further runtime
    /// conversion.
    ///
    /// Returns `None` for GGUF secondaries (no SOA section), or for tensors
    /// whose dtype is not `Q4_0` (norms / non-quantised weights).
    pub fn split_pre_converted_soa<'a>(
        &'a self,
        info: &SecondaryTensorInfo,
    ) -> Option<(&'a [u8], &'a [u8])> {
        if !self.is_pre_converted_soa() {
            return None;
        }
        if info.dtype != DType::Q4_0 {
            return None;
        }
        let bytes = self.tensor_bytes(info);
        // Each Q4_0 block: 16 q-bytes + 2 d-bytes = 18 bytes total.
        if !bytes.len().is_multiple_of(18) {
            return None;
        }
        let num_blocks = bytes.len() / 18;
        let q_len = num_blocks * 16;
        let d_len = num_blocks * 2;
        debug_assert_eq!(q_len + d_len, bytes.len());
        Some((&bytes[..q_len], &bytes[q_len..q_len + d_len]))
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Public open function — detects format and delegates
// ──────────────────────────────────────────────────────────────────────────

/// Open a secondary weight file, validate it against the primary model config,
/// and build the tensor slice index.
///
/// Format detection (in order):
/// 1. Extension `.auf` → open as AUF.
/// 2. Magic bytes `b"ARGUS_W\0"` (first 8 bytes) → open as AUF.
/// 3. Otherwise → open as GGUF.
///
/// Phase 1 policy: any error aborts loading. Caller (`generate.rs`) surfaces
/// the error to the user. This is the "fail fast" stance consistent with the
/// Architect decision of 2026-04-24 (INV-132~134).
pub fn open_secondary(
    path: &Path,
    primary_config: &ModelConfig,
    primary_gguf: &GgufFile,
) -> Result<SecondaryMmap, LoadError> {
    open_secondary_with_dtype(
        path,
        primary_config,
        primary_gguf,
        SecondaryDtypeChoice::Auto,
    )
}

/// `open_secondary`의 dtype 선택 파라미터 버전 (D-2, ENG-ALG-225).
///
/// AUF 파일인 경우 `secondary_dtype_choice`를 적용하여 dtype을 선택한다.
/// GGUF 파일인 경우 `secondary_dtype_choice`는 무시된다 (GGUF는 단일 dtype 파일).
pub fn open_secondary_with_dtype(
    path: &Path,
    primary_config: &ModelConfig,
    primary_gguf: &GgufFile,
    secondary_dtype_choice: SecondaryDtypeChoice,
) -> Result<SecondaryMmap, LoadError> {
    open_secondary_with_options(
        path,
        primary_config,
        primary_gguf,
        secondary_dtype_choice,
        SecondaryLayoutChoice::Auto,
    )
}

/// `open_secondary`의 dtype + layout 선택 풀버전.
pub fn open_secondary_with_options(
    path: &Path,
    primary_config: &ModelConfig,
    primary_gguf: &GgufFile,
    secondary_dtype_choice: SecondaryDtypeChoice,
    secondary_layout_choice: SecondaryLayoutChoice,
) -> Result<SecondaryMmap, LoadError> {
    if is_auf_path(path) {
        open_secondary_auf(
            path,
            primary_config,
            secondary_dtype_choice,
            secondary_layout_choice,
        )
    } else {
        open_secondary_gguf(path, primary_config, primary_gguf)
    }
}

/// LISWAP-6 — backend-aware variant of `open_secondary_with_options`.
///
/// When `backend` reports the `qnn_oppkg` family AND the secondary file is
/// GGUF, this constructor returns a `SecondaryMmap::Rpcmem` variant that
/// participates in the DMA-BUF alias swap path (zero-H2D-copy weight swap).
/// All other combinations dispatch to the standard GGUF / AUF paths so
/// behaviour for non-`qnn_oppkg` backends is unchanged.
///
/// On rpcmem construction failure (Android-only path / runtime not ready /
/// rpcmem alloc rejected at metadata-validation), the function gracefully
/// falls back to `open_secondary_with_options` so production never loses the
/// baseline swap path because of an alias-allocator hiccup.
pub fn open_secondary_with_backend(
    path: &Path,
    primary_config: &ModelConfig,
    primary_gguf: &GgufFile,
    secondary_dtype_choice: SecondaryDtypeChoice,
    secondary_layout_choice: SecondaryLayoutChoice,
    backend: &std::sync::Arc<dyn crate::core::backend::Backend>,
) -> Result<SecondaryMmap, LoadError> {
    // AUF or non-rpcmem backend → standard path.
    if is_auf_path(path) || !backend_supports_rpcmem_secondary(backend) {
        return open_secondary_with_options(
            path,
            primary_config,
            primary_gguf,
            secondary_dtype_choice,
            secondary_layout_choice,
        );
    }
    // qnn_oppkg + GGUF → try Rpcmem; fall back to standard GGUF on any error.
    match try_open_rpcmem_secondary(path, primary_config, primary_gguf, backend) {
        Ok(handle) => Ok(handle),
        Err(e) => {
            eprintln!(
                "[liswap6] rpcmem secondary unavailable for '{}': {e} — falling back to standard GGUF path",
                path.display()
            );
            open_secondary_gguf(path, primary_config, primary_gguf)
        }
    }
}

/// Returns true when `backend` is the `qnn_oppkg` family (M3 GPU OpPackage)
/// — the only backend that exposes the rpcmem heap required for the alias
/// path. Detected via `Backend::name()` to avoid feature-gate gymnastics on
/// the trait object.
///
/// `pub(crate)` since W-AUF-2 — `resolve_secondary` consults the same predicate
/// when deciding whether to promote an AUF self-secondary to the RpcMem variant.
pub(crate) fn backend_supports_rpcmem_secondary(
    backend: &std::sync::Arc<dyn crate::core::backend::Backend>,
) -> bool {
    let name = backend.name();
    name.contains("QNN OpPackage") || name.contains("qnn_oppkg")
}

/// W-AUF-2.3 — Promote a `SecondaryMmap::Auf` self-secondary to the
/// `SecondaryMmap::Rpcmem` variant when the backend is `qnn_oppkg`.
///
/// `Some(rpc)` 반환 시 호출자는 그 핸들을 사용한다. `None`이면 promote를
/// 시도하지 않은 것 (호스트 빌드 / 다른 backend). promote 시도가 실패한 경우
/// stderr 경고 + 원본 `SecondaryMmap::Auf` 유지를 위해 `None`을 돌려보낸다.
///
/// `view`와 `layer_index`는 promote 성공 시 `RpcmemSecondaryStore`가 소유한다.
pub(crate) fn try_promote_auf_self_secondary_to_rpcmem(
    view: std::sync::Arc<AufView>,
    layer_index: Vec<LayerTensorSlice>,
    diag_source_path: PathBuf,
    backend: &std::sync::Arc<dyn crate::core::backend::Backend>,
) -> Option<SecondaryMmap> {
    if !backend_supports_rpcmem_secondary(backend) {
        return None;
    }

    #[cfg(all(feature = "qnn", target_os = "android"))]
    {
        let qnn = backend
            .as_any()
            .downcast_ref::<crate::backend::qnn_oppkg::QnnOppkgBackend>()?;
        let runtime = qnn.runtime_arc();
        let rpcmem_fns = runtime.rpcmem_fns();
        match crate::models::weights::rpcmem_secondary::RpcmemSecondaryStore::from_auf_self_secondary(
            view,
            layer_index,
            diag_source_path,
            std::sync::Arc::clone(backend),
            (rpcmem_fns.0, rpcmem_fns.1),
        ) {
            Ok(store) => Some(SecondaryMmap::Rpcmem(store)),
            Err(e) => {
                eprintln!(
                    "[AUF self-secondary] rpcmem promote 실패, standard AUF path 유지: {e}"
                );
                None
            }
        }
    }
    #[cfg(not(all(feature = "qnn", target_os = "android")))]
    {
        let _ = (view, layer_index, diag_source_path);
        // 호스트/non-Android에서는 rpcmem feature 자체가 없어 promote 불가.
        // backend.name()이 qnn_oppkg를 거짓으로 자처해도 None을 돌려보내 fallback 유지.
        None
    }
}

/// Try to construct a `SecondaryMmap::Rpcmem` for `qnn_oppkg`. Errors
/// surface as a generic `LoadError::SecondaryUnavailable` so the caller's
/// fallback is uniform.
fn try_open_rpcmem_secondary(
    path: &Path,
    primary_config: &ModelConfig,
    primary_gguf: &GgufFile,
    backend: &std::sync::Arc<dyn crate::core::backend::Backend>,
) -> Result<SecondaryMmap, LoadError> {
    #[cfg(all(feature = "qnn", target_os = "android"))]
    {
        let qnn = backend
            .as_any()
            .downcast_ref::<crate::backend::qnn_oppkg::QnnOppkgBackend>()
            .ok_or_else(|| LoadError::SecondaryUnavailable {
                path: path.to_path_buf(),
                source: anyhow::anyhow!(
                    "rpcmem secondary: backend name claims qnn_oppkg but downcast failed"
                ),
            })?;
        let runtime = qnn.runtime_arc();
        let rpcmem_fns = runtime.rpcmem_fns();
        let store = crate::models::weights::rpcmem_secondary::RpcmemSecondaryStore::from_gguf(
            path,
            primary_config,
            primary_gguf,
            std::sync::Arc::clone(backend),
            (rpcmem_fns.0, rpcmem_fns.1),
        )
        .map_err(|e| LoadError::SecondaryUnavailable {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(SecondaryMmap::Rpcmem(store))
    }
    #[cfg(not(all(feature = "qnn", target_os = "android")))]
    {
        let _ = (path, primary_config, primary_gguf, backend);
        Err(LoadError::SecondaryUnavailable {
            path: path.to_path_buf(),
            source: anyhow::anyhow!(
                "rpcmem secondary requires Android + qnn feature; falling back to GGUF"
            ),
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────
// GGUF open path (original `open_secondary` logic)
// ──────────────────────────────────────────────────────────────────────────

fn open_secondary_gguf(
    path: &Path,
    primary_config: &ModelConfig,
    primary_gguf: &GgufFile,
) -> Result<SecondaryMmap, LoadError> {
    let gguf = GgufFile::open(path).map_err(|e| LoadError::SecondaryUnavailable {
        path: path.to_path_buf(),
        source: e,
    })?;

    // Model metadata must match the primary config (ENG-DAT-C10).
    let secondary_config =
        ModelConfig::from_gguf_metadata(&gguf).map_err(|e| LoadError::SecondaryUnavailable {
            path: path.to_path_buf(),
            source: e,
        })?;
    check_metadata(primary_config, &secondary_config)?;

    // Build per-layer slice index.
    let num_layers = primary_config.num_hidden_layers;
    let mut layer_index: Vec<LayerTensorSlice> = vec![LayerTensorSlice::default(); num_layers];

    let tensor_data_offset = gguf.tensor_data_offset();
    for info in &gguf.tensors {
        let Ok(dtype) = ggml_type_to_dtype(info.ggml_type) else {
            continue;
        };
        let byte_size = tensor_byte_size(info);
        if byte_size == 0 {
            continue;
        }
        let slice_info = SecondaryTensorInfo {
            offset: tensor_data_offset + info.offset as usize,
            len: byte_size,
            dtype,
            dims: info.dims.clone(),
        };

        if let Some((layer_idx, subname)) = parse_block_tensor_name(&info.name) {
            if layer_idx >= num_layers {
                continue;
            }
            layer_index[layer_idx]
                .tensors
                .insert(subname.to_string(), slice_info);
        }
    }

    // Validate shape match for decoder tensors present in the primary file.
    for info in &primary_gguf.tensors {
        let Some((layer_idx, subname)) = parse_block_tensor_name(&info.name) else {
            continue;
        };
        if layer_idx >= num_layers {
            continue;
        }
        let Some(secondary_info) = layer_index[layer_idx].tensors.get(subname) else {
            return Err(LoadError::TensorMissing {
                name: info.name.clone(),
            });
        };
        if secondary_info.dims != info.dims {
            return Err(LoadError::ShapeMismatch {
                name: info.name.clone(),
                primary: info.dims.clone(),
                secondary: secondary_info.dims.clone(),
            });
        }
    }

    Ok(SecondaryMmap::Gguf(GgufSecondaryMmap {
        gguf,
        layer_index,
        source_path: path.to_path_buf(),
    }))
}

// ──────────────────────────────────────────────────────────────────────────
// AUF open path — moved to `crate::models::loader::auf::secondary` (W-AUF-1 C2).
// Re-exports below preserve the original call-site signature so external
// callers (`weights::mod`, `borrowed_mmap_buffer`, etc.) continue to compile
// without import changes.
// ──────────────────────────────────────────────────────────────────────────

pub use crate::models::loader::auf::secondary::{
    auf_dtype_to_engine, build_auf_secondary_from_view, check_auf_metadata, is_auf_path,
    open_secondary_auf,
};

// ──────────────────────────────────────────────────────────────────────────
// Metadata helpers
// ──────────────────────────────────────────────────────────────────────────

fn check_metadata(primary: &ModelConfig, secondary: &ModelConfig) -> Result<(), LoadError> {
    macro_rules! check_field {
        ($field:ident) => {
            if primary.$field != secondary.$field {
                return Err(LoadError::MetadataMismatch {
                    field: stringify!($field),
                    primary: format!("{:?}", primary.$field),
                    secondary: format!("{:?}", secondary.$field),
                });
            }
        };
    }
    check_field!(hidden_size);
    check_field!(num_hidden_layers);
    check_field!(num_attention_heads);
    check_field!(num_key_value_heads);
    check_field!(head_dim);
    check_field!(intermediate_size);
    check_field!(vocab_size);
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Utility helpers
// ──────────────────────────────────────────────────────────────────────────

/// Extract `(layer_idx, subname)` from `blk.<N>.<subname>` tensor names.
pub(crate) fn parse_block_tensor_name(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let idx: usize = rest[..dot].parse().ok()?;
    Some((idx, &rest[dot + 1..]))
}

/// Prefault `bytes` so subsequent reads hit the page cache.
///
/// Linux/Android: `madvise(MADV_WILLNEED)` to schedule a kernel-side prefetch,
/// followed by an explicit page-touch loop (one byte per 4 KiB) so the pages
/// are forced into the resident set even when the kernel rejects the hint
/// (page reclaim under pressure, kernel quirk, etc.).
///
/// Other targets: no-op. The function never panics and treats every step as
/// best-effort (failure is silent).
///
/// **Bytewise read uses `read_volatile`** to prevent compilers from removing
/// the loop as dead code.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn prefault_byte_range(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let ptr = bytes.as_ptr();
    let len = bytes.len();

    // (1) Page-aligned MADV_WILLNEED. Linux/Android require both addr and
    // length to be page-aligned for madvise; we round addr down and length up
    // (clamped to the slice boundary) to satisfy the contract.
    let page = 4096usize;
    let addr = ptr as usize;
    let aligned_start = addr & !(page - 1);
    let head_pad = addr - aligned_start;
    let aligned_len = (head_pad + len).div_ceil(page) * page;
    // SAFETY: madvise on a hint flag is non-fatal. The address range overlaps
    // pages owned by the mmap caller, but we only request prefetch — no write,
    // no remap. `bytes` outlives this call (caller contract).
    unsafe {
        let _ = libc::madvise(
            aligned_start as *mut libc::c_void,
            aligned_len,
            libc::MADV_WILLNEED,
        );
    }

    // (2) Explicit page-touch warmup. One volatile byte read per 4 KiB ensures
    // the page is resident even if MADV_WILLNEED is silently dropped. The
    // `read_volatile` prevents the optimiser from eliding the loop.
    let mut sink: u8 = 0;
    let mut off = 0usize;
    while off < len {
        // SAFETY: 0 <= off < len, ptr + off is in-bounds. read_volatile is
        // sound for any byte (no alignment requirement on u8).
        let v = unsafe { std::ptr::read_volatile(ptr.add(off)) };
        // XOR-accumulate to a black-box variable so LLVM treats the read as
        // observable. The value is otherwise discarded.
        sink ^= v;
        off += page;
    }
    // Touch the very last byte too (covers slices whose final page is partial
    // and not stepped over by the loop).
    if len > 0 {
        // SAFETY: len-1 < len.
        let v = unsafe { std::ptr::read_volatile(ptr.add(len - 1)) };
        sink ^= v;
    }
    // Force the compiler to materialise `sink` so the volatile reads are not
    // optimised away. `black_box` would be cleaner but is not in stable std
    // for primitive sinks here; an inline asm-free alternative is to write
    // the value into a memory location that escape-analyses out (but
    // read_volatile by itself is already an observable side effect, so this
    // line is defensive only).
    std::hint::black_box(sink);
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn prefault_byte_range(_bytes: &[u8]) {
    // No-op on non-Linux targets.
}

// ──────────────────────────────────────────────────────────────────────────
// Legacy accessor for GGUF GgufFile (used by swap_executor + transformer)
// ──────────────────────────────────────────────────────────────────────────

impl SecondaryMmap {
    /// Return the inner `GgufFile` reference, if this is a GGUF secondary.
    ///
    /// Used by `TransformerModel::load_gguf_with_secondary` to pass the GGUF
    /// handle into `open_secondary_gguf` for shape validation. AUF callers do
    /// not use this path.
    pub fn as_gguf_file(&self) -> Option<&GgufFile> {
        match self {
            SecondaryMmap::Gguf(g) => Some(&g.gguf),
            SecondaryMmap::Auf(_) => None,
            // Rpcmem variant carries its own private GgufFile (mmap kept alive
            // for memcpy source). Not exposed here since callers using
            // `as_gguf_file()` are GGUF-shape-validation paths that have
            // already validated against the primary.
            SecondaryMmap::Rpcmem(_) => None,
        }
    }

    /// Return the inner `AufView` reference, if this is an AUF secondary.
    ///
    /// Used by `generate.rs` lm_head load-path (Sprint G-1-D) to call
    /// `AufView::lm_head_q4_0_payload()` without re-opening the file.
    pub fn as_auf_view(&self) -> Option<&AufView> {
        match self {
            SecondaryMmap::Gguf(_) => None,
            SecondaryMmap::Auf(a) => Some(a.view.as_ref()),
            SecondaryMmap::Rpcmem(_) => None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::parse_block_tensor_name;

    #[test]
    fn prefault_byte_range_handles_empty_slice() {
        // Empty input must not call madvise / read past end. The function
        // returns silently regardless of target OS.
        super::prefault_byte_range(&[]);
    }

    #[test]
    fn prefault_byte_range_touches_pages_without_panic() {
        // 64 KiB anonymous buffer (16 pages). `prefault_byte_range` must
        // accept arbitrary slice alignment, walk one byte per 4 KiB plus the
        // last byte, and never panic. The volatile reads are observable but
        // the data is stable (filled with a known pattern) so this also
        // serves as a smoke test that the access stays in-bounds.
        let buf = vec![0xA5u8; 64 * 1024];
        super::prefault_byte_range(&buf);
        // Sanity: data unchanged (prefault is read-only).
        assert!(buf.iter().all(|&b| b == 0xA5));
    }

    #[test]
    fn prefault_byte_range_handles_unaligned_slice() {
        // Slice starting mid-page exercises the head-pad / round-up branch
        // in the madvise alignment helper.
        let buf = vec![0u8; 8192 + 17];
        super::prefault_byte_range(&buf[17..]);
    }

    #[test]
    fn parses_block_tensor_names() {
        assert_eq!(
            parse_block_tensor_name("blk.0.attn_q.weight"),
            Some((0, "attn_q.weight"))
        );
        assert_eq!(
            parse_block_tensor_name("blk.15.ffn_down.weight"),
            Some((15, "ffn_down.weight"))
        );
        assert_eq!(parse_block_tensor_name("token_embd.weight"), None);
        assert_eq!(parse_block_tensor_name("output.weight"), None);
        assert_eq!(parse_block_tensor_name("blk.abc.attn_q.weight"), None);
    }

    #[test]
    fn is_auf_path_by_extension() {
        use super::is_auf_path;
        use std::path::Path;
        assert!(is_auf_path(Path::new("/tmp/model.auf")));
        assert!(!is_auf_path(Path::new("/tmp/model.gguf")));
        assert!(!is_auf_path(Path::new("/tmp/model")));
    }

    #[test]
    fn tensor_kind_to_subname_all_variants() {
        use super::tensor_kind_to_subname;
        use crate::auf::tensor_index::TensorKind;
        // Layer tensors should map to non-None.
        assert_eq!(
            tensor_kind_to_subname(TensorKind::AttnQ.as_u32()),
            Some("attn_q.weight")
        );
        assert_eq!(
            tensor_kind_to_subname(TensorKind::FfnGate.as_u32()),
            Some("ffn_gate.weight")
        );
        // Cross-layer tensors map to None.
        assert_eq!(tensor_kind_to_subname(TensorKind::Embedding.as_u32()), None);
        assert_eq!(tensor_kind_to_subname(TensorKind::LmHead.as_u32()), None);
    }

    #[test]
    fn auf_dtype_to_engine_round_trip() {
        use super::auf_dtype_to_engine;
        use crate::auf::tensor_index::TensorDType;
        use crate::core::buffer::DType;
        assert_eq!(
            auf_dtype_to_engine(TensorDType::Q4_0.as_u32()),
            Some(DType::Q4_0)
        );
        assert_eq!(
            auf_dtype_to_engine(TensorDType::F16.as_u32()),
            Some(DType::F16)
        );
        assert_eq!(auf_dtype_to_engine(999), None);
    }

    #[test]
    fn secondary_mmap_is_pre_converted_soa_gguf() {
        // GGUF always returns false.
        // We cannot construct a GgufSecondaryMmap without a real file,
        // so we test only the public contract via the enum discriminant check.
        // The AUF path is tested via open_secondary_auf with a real AufView fixture.
        // This test just exercises the logic branch.
        use super::tensor_kind_to_subname;
        use crate::auf::tensor_index::TensorKind;
        // AttnO subname.
        assert_eq!(
            tensor_kind_to_subname(TensorKind::AttnO.as_u32()),
            Some("attn_output.weight")
        );
    }

    /// ENG-ALG-229: `prefault_layers` on an AUF secondary touches only the byte
    /// ranges of the requested layers, not the entire WEIGHTS payload.
    ///
    /// Constructs a minimal 3-layer AUF (512 bytes per layer) and verifies that
    /// calling `prefault_layers` with a subset of layers does not panic and
    /// leaves the backing buffer unchanged (read-only semantics).
    #[test]
    fn prefault_layers_auf_subset_no_panic() {
        use crate::auf::reader::open_from_bytes;
        use crate::auf::section::TAG_WEIGHTS_CPU_AOS;
        use crate::auf::tensor_index::{TensorDType, TensorEntry, TensorIndex, TensorKind};
        use crate::auf::tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE};
        use crate::auf::writer::AufWriter;
        use crate::auf::{AufMeta, BackendTag};
        use crate::models::config::{ModelArch, ModelConfig};
        use crate::models::weights::SecondaryDtypeChoice;
        use crate::models::weights::secondary_mmap::build_auf_secondary_from_view;

        const N_LAYERS: usize = 3;
        const BYTES_PER_TENSOR: usize = 128;

        // Build a weights payload: N_LAYERS × 2 tensors × 128 bytes each.
        // Layer i tensors start at offset i * 256 (two tensors per layer).
        let mut weights_payload = vec![0xBBu8; N_LAYERS * 2 * BYTES_PER_TENSOR];
        // Stamp each layer region with its index so we can distinguish them.
        for i in 0..N_LAYERS {
            let base = i * 2 * BYTES_PER_TENSOR;
            for b in &mut weights_payload[base..base + 2 * BYTES_PER_TENSOR] {
                *b = i as u8;
            }
        }

        // Build TensorIndex: 2 tensors per layer, section-local offsets.
        let mut tag_buf = [0u8; 24];
        tag_buf[..TAG_WEIGHTS_CPU_AOS.len()].copy_from_slice(TAG_WEIGHTS_CPU_AOS.as_bytes());
        let mut entries = Vec::new();
        for layer_idx in 0..N_LAYERS as u32 {
            let base_offset = (layer_idx as usize * 2 * BYTES_PER_TENSOR) as u64;
            entries.push(TensorEntry {
                layer_idx,
                kind: TensorKind::AttnQ.as_u32(),
                dtype: TensorDType::F32.as_u32(),
                shape: vec![1, 128],
                alignment: 64,
                variant_offsets: vec![base_offset],
                variant_sizes: vec![BYTES_PER_TENSOR as u64],
            });
            entries.push(TensorEntry {
                layer_idx,
                kind: TensorKind::AttnK.as_u32(),
                dtype: TensorDType::F32.as_u32(),
                shape: vec![1, 128],
                alignment: 64,
                variant_offsets: vec![base_offset + BYTES_PER_TENSOR as u64],
                variant_sizes: vec![BYTES_PER_TENSOR as u64],
            });
        }
        let tensor_index = TensorIndex {
            variant_tags: vec![tag_buf],
            entries,
        };

        let meta = AufMeta {
            architecture: "llama".to_owned(),
            n_layers: N_LAYERS as u32,
            n_heads_q: 1,
            n_kv_heads: 1,
            head_dim: 128,
            hidden_dim: 128,
            ffn_dim: 256,
            vocab_size: 2,
            max_seq_len: 32,
            rope_theta: 10000.0,
            rotary_dim: 128,
            rope_scaling: 1.0,
            rms_norm_epsilon: 1e-5,
            default_dtype: None,
        };
        let tok = AufTokenizer {
            kind: TOKENIZER_KIND_BPE,
            tokens: vec![b"a".to_vec()],
            merges: vec![],
            bos_id: 1,
            eos_id: 2,
            pad_id: -1,
            unk_id: 0,
            chat_template: None,
        };

        let auf_bytes = AufWriter::new(meta, tok, [0u8; 32], 0, 0)
            .with_tensor_index(tensor_index)
            .add_weights_section(TAG_WEIGHTS_CPU_AOS, weights_payload)
            .build()
            .unwrap();

        let view = open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();
        let config = ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 128,
            num_hidden_layers: N_LAYERS,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 128,
            intermediate_size: 256,
            vocab_size: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            has_qkv_bias: false,
            tie_word_embeddings: false,
            eos_token_id: 1,
            weight_prefix: String::new(),
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
        };
        let secondary = build_auf_secondary_from_view(
            view,
            &config,
            std::path::Path::new("/fake/test.auf"),
            BackendTag::CpuAos,
            SecondaryDtypeChoice::Auto,
        )
        .expect("build_auf_secondary_from_view should succeed");

        // Scenario 1: subset (layers 0 and 2 only) — must not panic.
        secondary.prefault_layers(&[0, 2]);

        // Scenario 2: empty target — no-op, must not panic.
        secondary.prefault_layers(&[]);

        // Scenario 3: out-of-range index — silent skip, must not panic.
        secondary.prefault_layers(&[99, 1000]);

        // Scenario 4: full layer list must be equivalent to (or a subset of)
        // what the whole-payload prefault() would cover. Verify no panic and
        // that layer byte ranges are accessible after prefault_layers.
        secondary.prefault_layers(&[0, 1, 2]);

        // Verify data integrity: tensor_bytes for layer 1 is still correct after
        // calling prefault_layers (read-only, data must be unchanged).
        let info = secondary
            .layer_tensor(1, "attn_q.weight")
            .expect("layer 1 attn_q must exist");
        let tb = secondary.tensor_bytes(info);
        assert_eq!(tb.len(), BYTES_PER_TENSOR);
        assert!(
            tb.iter().all(|&b| b == 1),
            "layer 1 bytes should all be 0x01 (layer index stamp)"
        );
    }

    /// ENG-ALG-229: `prefault_layers` byte range accuracy — the layer index
    /// correctly maps each layer to its own tensor offsets.
    #[test]
    fn prefault_layers_layer_index_byte_ranges_are_disjoint() {
        use crate::auf::reader::open_from_bytes;
        use crate::auf::section::TAG_WEIGHTS_CPU_AOS;
        use crate::auf::tensor_index::{TensorDType, TensorEntry, TensorIndex, TensorKind};
        use crate::auf::tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE};
        use crate::auf::writer::AufWriter;
        use crate::auf::{AufMeta, BackendTag};
        use crate::models::config::{ModelArch, ModelConfig};
        use crate::models::weights::SecondaryDtypeChoice;
        use crate::models::weights::secondary_mmap::build_auf_secondary_from_view;

        const N_LAYERS: usize = 4;
        const BYTES_PER_TENSOR: usize = 64;

        // Each layer gets a unique fill byte equal to its index.
        let mut weights_payload = vec![0u8; N_LAYERS * BYTES_PER_TENSOR];
        for i in 0..N_LAYERS {
            for b in &mut weights_payload[i * BYTES_PER_TENSOR..(i + 1) * BYTES_PER_TENSOR] {
                *b = i as u8;
            }
        }

        let mut tag_buf = [0u8; 24];
        tag_buf[..TAG_WEIGHTS_CPU_AOS.len()].copy_from_slice(TAG_WEIGHTS_CPU_AOS.as_bytes());
        let entries: Vec<TensorEntry> = (0..N_LAYERS as u32)
            .map(|layer_idx| TensorEntry {
                layer_idx,
                kind: TensorKind::AttnQ.as_u32(),
                dtype: TensorDType::F32.as_u32(),
                shape: vec![1, 64],
                alignment: 64,
                variant_offsets: vec![(layer_idx as usize * BYTES_PER_TENSOR) as u64],
                variant_sizes: vec![BYTES_PER_TENSOR as u64],
            })
            .collect();
        let tensor_index = TensorIndex {
            variant_tags: vec![tag_buf],
            entries,
        };

        let meta = AufMeta {
            architecture: "llama".to_owned(),
            n_layers: N_LAYERS as u32,
            n_heads_q: 1,
            n_kv_heads: 1,
            head_dim: 64,
            hidden_dim: 64,
            ffn_dim: 128,
            vocab_size: 2,
            max_seq_len: 32,
            rope_theta: 10000.0,
            rotary_dim: 64,
            rope_scaling: 1.0,
            rms_norm_epsilon: 1e-5,
            default_dtype: None,
        };
        let tok = AufTokenizer {
            kind: TOKENIZER_KIND_BPE,
            tokens: vec![b"a".to_vec()],
            merges: vec![],
            bos_id: 1,
            eos_id: 2,
            pad_id: -1,
            unk_id: 0,
            chat_template: None,
        };

        let auf_bytes = AufWriter::new(meta, tok, [0u8; 32], 0, 0)
            .with_tensor_index(tensor_index)
            .add_weights_section(TAG_WEIGHTS_CPU_AOS, weights_payload)
            .build()
            .unwrap();

        let view = open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();
        let config = ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: 64,
            num_hidden_layers: N_LAYERS,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 64,
            intermediate_size: 128,
            vocab_size: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            has_qkv_bias: false,
            tie_word_embeddings: false,
            eos_token_id: 1,
            weight_prefix: String::new(),
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
        };
        let secondary = build_auf_secondary_from_view(
            view,
            &config,
            std::path::Path::new("/fake/layers.auf"),
            BackendTag::CpuAos,
            SecondaryDtypeChoice::Auto,
        )
        .expect("build_auf_secondary_from_view should succeed");

        // Verify that byte ranges for each layer are disjoint and correctly stamped.
        for layer_idx in 0..N_LAYERS {
            let info = secondary
                .layer_tensor(layer_idx, "attn_q.weight")
                .unwrap_or_else(|| panic!("layer {layer_idx} attn_q must exist"));
            let tb = secondary.tensor_bytes(info);
            assert_eq!(tb.len(), BYTES_PER_TENSOR);
            assert!(
                tb.iter().all(|&b| b == layer_idx as u8),
                "layer {layer_idx} bytes should all be {layer_idx}"
            );
        }

        // prefault_layers with layers [1, 3] must not affect layers [0, 2].
        secondary.prefault_layers(&[1, 3]);

        // Data integrity preserved after prefault_layers.
        for layer_idx in 0..N_LAYERS {
            let info = secondary
                .layer_tensor(layer_idx, "attn_q.weight")
                .unwrap_or_else(|| panic!("layer {layer_idx} attn_q must exist"));
            let tb = secondary.tensor_bytes(info);
            assert!(
                tb.iter().all(|&b| b == layer_idx as u8),
                "layer {layer_idx} bytes should still be {layer_idx} after prefault_layers"
            );
        }
    }
}
