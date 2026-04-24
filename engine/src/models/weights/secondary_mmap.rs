//! `SecondaryMmap` — metadata-validated mmap handle for the secondary GGUF.
//!
//! Phase 1 establishes the mmap handle only: the file is opened, the GGUF
//! header is parsed once, and per-layer tensor slice descriptors are built.
//! Actual byte access (`tensor_bytes`) is exposed so Phase 2 (`SwapExecutor`)
//! can consume it without re-parsing.
//!
//! Spec: ENG-DAT-094 (SecondaryMmap), ENG-DAT-C10 (metadata match),
//! ENG-DAT-C12 / INV-125 (lifetime), INV-123/124 (read-only, dtype source).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::core::buffer::DType;
use crate::models::config::ModelConfig;
use crate::models::loader::gguf::{GgufFile, ggml_type_to_dtype, tensor_byte_size};

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

/// Tensor slice inside the secondary mmap (one entry per GGUF tensor).
///
/// Offsets and lengths are relative to the backing mmap, consumed by
/// `SwapExecutor` at Phase 2 to copy-and-permute the raw bytes.
#[derive(Debug, Clone)]
pub struct SecondaryTensorInfo {
    /// Absolute offset within the mmap (tensor data section + GGUF offset).
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

/// Handle to a secondary GGUF file kept alive for the lifetime of the model.
/// Carries the raw mmap plus the pre-parsed tensor index so Phase 2 can
/// locate bytes in O(1) per tensor.
///
/// Cross-layer tensors (embedding, output_norm, output) are intentionally
/// **not** indexed here (ENG-DAT-C11). Phase 1/2 only swap decoder block
/// weights; the `cross_layer_offsets` stub was removed in Stage 2 cleanup.
pub struct SecondaryMmap {
    /// Underlying GGUF parse (keeps mmap handle alive).
    pub gguf: GgufFile,
    /// Indexed by `layer_idx`. Length = num decoder layers.
    pub layer_index: Vec<LayerTensorSlice>,
    /// Source path, preserved for diagnostic messages.
    pub source_path: PathBuf,
}

impl std::fmt::Debug for SecondaryMmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecondaryMmap")
            .field("source_path", &self.source_path)
            .field("num_layers", &self.layer_index.len())
            .finish()
    }
}

impl SecondaryMmap {
    /// Fetch a layer's tensor descriptor by subname (e.g. "attn_q.weight").
    /// Returns `None` if the layer is out of range or the tensor is missing.
    pub fn layer_tensor(&self, layer_idx: usize, subname: &str) -> Option<&SecondaryTensorInfo> {
        self.layer_index
            .get(layer_idx)
            .and_then(|slice| slice.tensors.get(subname))
    }

    /// Zero-copy byte slice for a descriptor. Panics if `info` does not
    /// originate from this mmap (caller contract).
    pub fn tensor_bytes(&self, info: &SecondaryTensorInfo) -> &[u8] {
        let end = info.offset + info.len;
        &self.gguf.mmap_data()[info.offset..end]
    }
}

/// Open a secondary GGUF file, validate it against the primary model config,
/// and build the tensor slice index.
///
/// Phase 1 policy: any error aborts loading. Caller (`generate.rs`) surfaces
/// the error to the user. This is the "fail fast" stance consistent with the
/// Architect decision of 2026-04-24.
pub fn open_secondary(
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

    // Build per-layer slice index. We enumerate primary tensors to ensure the
    // secondary covers everything we may later swap; the reverse direction
    // (secondary extras) is tolerated. Cross-layer tensors (embedding, norms,
    // lm_head) are intentionally skipped — they are never swapped (ENG-DAT-C11).
    let num_layers = primary_config.num_hidden_layers;
    let mut layer_index: Vec<LayerTensorSlice> = vec![LayerTensorSlice::default(); num_layers];

    let tensor_data_offset = gguf.tensor_data_offset();
    for info in &gguf.tensors {
        let Ok(dtype) = ggml_type_to_dtype(info.ggml_type) else {
            // Unsupported dtype in secondary — record as mismatch so caller
            // can surface a clear error if the primary does need this tensor.
            // We skip the entry; if the primary requires it, the tensor lookup
            // later will yield `TensorMissing`.
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
        // Cross-layer tensors are silently skipped (not indexed, ENG-DAT-C11).
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

    Ok(SecondaryMmap {
        gguf,
        layer_index,
        source_path: path.to_path_buf(),
    })
}

/// Extract `(layer_idx, subname)` from `blk.<N>.<subname>` tensor names.
fn parse_block_tensor_name(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let idx: usize = rest[..dot].parse().ok()?;
    Some((idx, &rest[dot + 1..]))
}

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

// ---------------------------------------------------------------------------
// `GgufFile::mmap_data`/`tensor_data_offset` exposure
// ---------------------------------------------------------------------------
//
// Implemented via `impl`-extensions declared in `gguf.rs` (pub(crate) accessors).
// The helpers are kept on the GGUF side to preserve encapsulation of mmap
// ownership, and used here only by this file to build layer slice info.

#[cfg(test)]
mod tests {
    use super::parse_block_tensor_name;

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
}
