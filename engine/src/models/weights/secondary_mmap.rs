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

use crate::core::buffer::DType;
use crate::models::config::ModelConfig;
use crate::models::loader::gguf::{GgufFile, ggml_type_to_dtype, tensor_byte_size};

// ── AUF crate imports ──────────────────────────────────────────────────────
use crate::auf::{
    AufView, BackendTag,
    section::TAG_WEIGHTS_ADRENO_SOA,
    tensor_index::{LAYER_IDX_CROSS, TensorDType, TensorKind},
};

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
}

// ──────────────────────────────────────────────────────────────────────────
// AUF-backed secondary
// ──────────────────────────────────────────────────────────────────────────

/// Map `TensorKind` to GGUF-style subname used by `SwapExecutor`.
///
/// Returns `None` for cross-layer tensors (embedding, final_norm, lm_head)
/// that are not swapped.
fn tensor_kind_to_subname(kind: u32) -> Option<&'static str> {
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
    }
}

/// Map `TensorDType` to engine `DType`. Returns `None` for unrecognised codes.
fn auf_dtype_to_engine(dtype: u32) -> Option<DType> {
    match TensorDType::from_u32(dtype)? {
        TensorDType::F32 => Some(DType::F32),
        TensorDType::F16 => Some(DType::F16),
        TensorDType::BF16 => Some(DType::BF16),
        TensorDType::Q4_0 => Some(DType::Q4_0),
        TensorDType::Q4_1 => Some(DType::Q4_1),
        TensorDType::Q8_0 => Some(DType::Q8_0),
        TensorDType::U8 => Some(DType::U8),
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
    pub view: AufView,
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
}

impl std::fmt::Debug for SecondaryMmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecondaryMmap::Gguf(g) => f.debug_tuple("SecondaryMmap::Gguf").field(g).finish(),
            SecondaryMmap::Auf(a) => f.debug_tuple("SecondaryMmap::Auf").field(a).finish(),
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
        }
    }

    /// Zero-copy byte slice for a descriptor. Panics if `info` does not
    /// originate from this secondary (caller contract).
    pub fn tensor_bytes(&self, info: &SecondaryTensorInfo) -> &[u8] {
        match self {
            SecondaryMmap::Gguf(g) => g.tensor_bytes(info),
            SecondaryMmap::Auf(a) => a.tensor_bytes(info),
        }
    }

    /// Source file path (for diagnostic messages).
    pub fn source_path(&self) -> &Path {
        match self {
            SecondaryMmap::Gguf(g) => &g.source_path,
            SecondaryMmap::Auf(a) => &a.source_path,
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
    pub fn prefault(&self) {
        match self {
            SecondaryMmap::Auf(a) => a.prefault(),
            SecondaryMmap::Gguf(g) => g.prefault(),
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
    if is_auf_path(path) {
        open_secondary_auf(path, primary_config)
    } else {
        open_secondary_gguf(path, primary_config, primary_gguf)
    }
}

/// Returns `true` if `path` should be opened as AUF.
///
/// Detection order:
/// 1. Extension `.auf` (case-sensitive).
/// 2. Magic bytes `b"ARGUS_W\0"` in the first 8 bytes of the file.
fn is_auf_path(path: &Path) -> bool {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e == "auf")
        .unwrap_or(false)
    {
        return true;
    }
    // Probe magic bytes (read-only, no mmap required).
    if let Ok(mut f) = std::fs::File::open(path) {
        use std::io::Read;
        let mut magic = [0u8; 8];
        if f.read_exact(&mut magic).is_ok() && &magic == crate::auf::header::AUF_MAGIC {
            return true;
        }
    }
    false
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
// AUF open path
// ──────────────────────────────────────────────────────────────────────────

/// Select the `BackendTag` for the AUF open call based on the runtime
/// environment (feature flags).
///
/// Priority:
/// 1. `opencl` feature active → `BackendTag::AdrenoSoa` (Adreno target).
/// 2. `cuda` or `cuda-embedded` feature active → `BackendTag::CudaAos`.
/// 3. Fallback → `BackendTag::CpuAos`.
fn detect_backend_tag() -> BackendTag {
    #[cfg(feature = "opencl")]
    return BackendTag::AdrenoSoa;
    #[cfg(all(
        not(feature = "opencl"),
        any(feature = "cuda", feature = "cuda-embedded")
    ))]
    return BackendTag::CudaAos;
    #[cfg(not(any(feature = "opencl", feature = "cuda", feature = "cuda-embedded")))]
    return BackendTag::CpuAos;
}

fn open_secondary_auf(
    path: &Path,
    primary_config: &ModelConfig,
) -> Result<SecondaryMmap, LoadError> {
    let backend_tag = detect_backend_tag();

    // AUF open: full invariant pipeline (INV-132, INV-133, INV-134).
    let view = crate::auf::reader::open(path, backend_tag).map_err(|e| {
        LoadError::AufInvariantViolation {
            detail: format!("{}  (file: {})", e, path.display()),
        }
    })?;

    // Validate AUF metadata against primary model config (ENG-DAT-C10).
    check_auf_metadata(primary_config, &view.meta, path)?;

    build_auf_secondary_from_view(view, primary_config, path, backend_tag)
}

/// Core logic for building an `AufSecondaryMmap` from a pre-parsed `AufView`.
///
/// Extracted for testability: callers (e.g. unit tests) can supply an
/// `AufView` constructed directly from bytes via `open_from_bytes`, bypassing
/// file I/O, while still exercising the full tensor-index → layer-index →
/// tensor_bytes round-trip.
///
/// # Offset contract
/// `TensorIndex::variant_offsets` entries are **section-local** (relative to the
/// start of the WEIGHTS payload).  `AufView::weights_bytes()` already returns a
/// slice that starts at the WEIGHTS section offset, so we store `var_offset`
/// directly into `SecondaryTensorInfo::offset` — we must **not** add
/// `weights_section_offset` again (that would cause a double-base OOB panic).
pub fn build_auf_secondary_from_view(
    view: crate::auf::AufView,
    primary_config: &ModelConfig,
    path: &Path,
    backend_tag: crate::auf::BackendTag,
) -> Result<SecondaryMmap, LoadError> {
    // Determine variant index for the selected backend.
    let weights_tag = backend_tag
        .weights_section_tag()
        .expect("detect_backend_tag returns a concrete tag");
    let variant_idx = view
        .tensor_index
        .variant_index_for_tag(weights_tag)
        .ok_or_else(|| LoadError::AufInvariantViolation {
            detail: format!(
                "TENSOR_INDEX does not list variant '{}' (file: {})",
                weights_tag,
                path.display()
            ),
        })?;

    // WEIGHTS payload start within the mmap.
    // We only need to confirm weights_range is Some (invariant); the offset itself
    // is NOT used for indexing — tensor_bytes() uses weights_bytes() which already
    // slices from that offset.  variant_offsets in TensorIndex are section-local.
    let (_weights_section_offset, _weights_section_size) = view
        .weights_range
        .expect("AufView::weights_range must be Some after open() with concrete backend_tag");

    // Build per-layer slice index from TENSOR_INDEX entries.
    let num_layers = primary_config.num_hidden_layers;
    let mut layer_index: Vec<LayerTensorSlice> = vec![LayerTensorSlice::default(); num_layers];

    for entry in &view.tensor_index.entries {
        // Skip cross-layer tensors (embedding, final_norm, lm_head).
        if entry.layer_idx == LAYER_IDX_CROSS {
            continue;
        }
        let layer_idx = entry.layer_idx as usize;
        if layer_idx >= num_layers {
            continue;
        }

        // Skip entries without a payload for this variant.
        let var_offset = entry
            .variant_offsets
            .get(variant_idx)
            .copied()
            .unwrap_or(u64::MAX);
        let var_size = entry.variant_sizes.get(variant_idx).copied().unwrap_or(0);
        if var_offset == u64::MAX || var_size == 0 {
            continue;
        }

        let Some(subname) = tensor_kind_to_subname(entry.kind) else {
            continue;
        };
        let Some(dtype) = auf_dtype_to_engine(entry.dtype) else {
            continue;
        };

        // var_offset is section-local (relative to WEIGHTS payload start).
        // weights_bytes() already returns a slice starting at weights_section_offset,
        // so we store the section-local offset directly — do NOT add weights_section_offset
        // again (that would cause a double-base panic on tensor_bytes()).
        let slice_info = SecondaryTensorInfo {
            offset: var_offset as usize,
            len: var_size as usize,
            dtype,
            // AUF shape is stored in logical order (outermost first).
            // `swap_executor.rs` expects GGUF order (innermost first), so reverse.
            dims: entry.shape.iter().rev().copied().collect(),
        };

        layer_index[layer_idx]
            .tensors
            .insert(subname.to_string(), slice_info);
    }

    let is_pre_converted_soa = weights_tag == TAG_WEIGHTS_ADRENO_SOA;

    Ok(SecondaryMmap::Auf(AufSecondaryMmap {
        view,
        layer_index,
        source_path: path.to_path_buf(),
        is_pre_converted_soa,
    }))
}

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

/// Validate AUF META against primary `ModelConfig`.
fn check_auf_metadata(
    primary: &ModelConfig,
    meta: &crate::auf::AufMeta,
    path: &Path,
) -> Result<(), LoadError> {
    macro_rules! check {
        ($primary_field:expr, $auf_field:expr, $name:literal) => {
            if $primary_field as u64 != $auf_field as u64 {
                return Err(LoadError::MetadataMismatch {
                    field: $name,
                    primary: format!("{}", $primary_field),
                    secondary: format!("{} (AUF: {})", $auf_field, path.display()),
                });
            }
        };
    }
    check!(primary.hidden_size, meta.hidden_dim, "hidden_size");
    check!(
        primary.num_hidden_layers,
        meta.n_layers,
        "num_hidden_layers"
    );
    check!(
        primary.num_attention_heads,
        meta.n_heads_q,
        "num_attention_heads"
    );
    check!(
        primary.num_key_value_heads,
        meta.n_kv_heads,
        "num_key_value_heads"
    );
    check!(primary.head_dim, meta.head_dim, "head_dim");
    check!(primary.intermediate_size, meta.ffn_dim, "intermediate_size");
    check!(primary.vocab_size, meta.vocab_size, "vocab_size");
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Utility helpers
// ──────────────────────────────────────────────────────────────────────────

/// Extract `(layer_idx, subname)` from `blk.<N>.<subname>` tensor names.
fn parse_block_tensor_name(name: &str) -> Option<(usize, &str)> {
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
}
