//! `BorrowedMmapBuffer` — zero-copy read-only borrow into a secondary mmap region.
//!
//! Used by `SwapExecutor::materialise_tensor` on the AOS / AUF path
//! (`needs_qk_unpermute_at_swap() == false`) to eliminate the redundant
//! `data.to_vec()` heap copy that was previously performed for every tensor.
//!
//! **Safety contract (INV-143)**: The buffer stores an `Arc<SecondaryMmap>` clone
//! so the secondary mmap region stays alive for the entire lifetime of this
//! buffer. The raw pointer is valid as long as the `Arc` is live, which is
//! guaranteed because the `Arc` is dropped only when `BorrowedMmapBuffer` is
//! dropped.
//!
//! The buffer is read-only; `as_mut_ptr()` returns `null_mut()`.  Callers
//! (backend `copy_weight_from`/`copy_from`) must only read through `as_ptr()`.
//!
//! Spec: ENG-ALG-227, INV-143.

use crate::core::buffer::{Buffer, DType};
use crate::models::weights::SecondaryMmap;
use anyhow::Result;
#[cfg(feature = "opencl")]
use ocl::core::Mem;
use std::any::Any;
use std::sync::Arc;

/// Zero-copy read-only buffer that borrows a byte slice from the secondary
/// mmap region.
///
/// Holds an `Arc<SecondaryMmap>` clone (INV-143) so the mmap region is never
/// unmapped while this buffer is live.  The pointer is derived from the raw
/// byte slice returned by `SecondaryMmap::tensor_bytes` and remains valid for
/// the entire lifetime of the Arc.
pub struct BorrowedMmapBuffer {
    /// Pointer into the secondary mmap region.
    /// Valid for as long as `_secondary_arc` is alive (INV-143).
    ptr: *const u8,
    /// Length in bytes of the borrowed slice.
    len: usize,
    /// Data type of the tensor stored in this slice.
    dtype: DType,
    /// Keeps the secondary mmap `Arc` alive — INV-143.
    _secondary_arc: Arc<SecondaryMmap>,
}

// SAFETY: The underlying mmap region is read-only and physically pinned in the
// OS page cache for the lifetime of the Arc.  Raw pointers derived from mmap
// data are safe to send across threads as long as no thread writes through
// them (we prohibit mutation via a null as_mut_ptr).
unsafe impl Send for BorrowedMmapBuffer {}
unsafe impl Sync for BorrowedMmapBuffer {}

impl BorrowedMmapBuffer {
    /// Create a new `BorrowedMmapBuffer` referencing `data` within `secondary`.
    ///
    /// # Arguments
    /// * `secondary` — Arc to the secondary mmap; will be cloned and stored.
    /// * `data` — byte slice within `secondary`'s mmap region (must originate
    ///   from `secondary.tensor_bytes()`).
    /// * `dtype` — data type of the tensor bytes.
    ///
    /// # Safety contract
    /// `data` must be a sub-slice of memory owned by `secondary`.  The caller
    /// is responsible for ensuring this (the only call site is
    /// `materialise_tensor`, which obtains `data` directly from
    /// `secondary.tensor_bytes(info)`).
    pub fn new(secondary: &Arc<SecondaryMmap>, data: &[u8], dtype: DType) -> Self {
        Self {
            ptr: data.as_ptr(),
            len: data.len(),
            dtype,
            _secondary_arc: secondary.clone(),
        }
    }
}

impl Buffer for BorrowedMmapBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn size(&self) -> usize {
        self.len
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Read-only buffer — mutation is not supported.
    ///
    /// Returns `null_mut()` to make it explicit that writes through this
    /// pointer are undefined behaviour.  Backend `copy_weight_from`/`copy_from`
    /// implementations must only read via `as_ptr()`.
    fn as_mut_ptr(&self) -> *mut u8 {
        std::ptr::null_mut()
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&Mem> {
        None
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        None
    }

    fn sync_device(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auf::{
        AufMeta, BackendTag,
        section::TAG_WEIGHTS_CPU_AOS,
        tensor_index::TensorIndex,
        tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE},
        writer::AufWriter,
    };
    use crate::models::config::{ModelArch, ModelConfig};
    use crate::models::weights::secondary_mmap::{
        SecondaryDtypeChoice, build_auf_secondary_from_view,
    };

    // ── AUF fixture with TensorIndex ─────────────────────────────────────────

    fn auf_meta() -> AufMeta {
        AufMeta {
            architecture: "llama".to_owned(),
            n_layers: 1,
            n_heads_q: 2,
            n_kv_heads: 1,
            head_dim: 8,
            hidden_dim: 16,
            ffn_dim: 32,
            vocab_size: 4,
            max_seq_len: 64,
            rope_theta: 10000.0,
            rotary_dim: 8,
            rope_scaling: 1.0,
            rms_norm_epsilon: 1e-5,
            default_dtype: None,
        }
    }

    fn auf_tokenizer() -> AufTokenizer {
        AufTokenizer {
            kind: TOKENIZER_KIND_BPE,
            tokens: vec![b"a".to_vec(), b"b".to_vec()],
            merges: vec![],
            bos_id: 1,
            eos_id: 2,
            pad_id: -1,
            unk_id: 0,
            chat_template: None,
        }
    }

    fn model_config() -> ModelConfig {
        let m = auf_meta();
        ModelConfig {
            arch: ModelArch::Llama,
            hidden_size: m.hidden_dim as usize,
            num_hidden_layers: m.n_layers as usize,
            num_attention_heads: m.n_heads_q as usize,
            num_key_value_heads: m.n_kv_heads as usize,
            head_dim: m.head_dim as usize,
            intermediate_size: m.ffn_dim as usize,
            vocab_size: m.vocab_size as usize,
            rms_norm_eps: m.rms_norm_epsilon,
            rope_theta: m.rope_theta,
            has_qkv_bias: false,
            tie_word_embeddings: false,
            eos_token_id: 2,
            weight_prefix: String::new(),
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
        }
    }

    /// Build a 24-byte tag buffer from a string (NUL-padded to 24 bytes).
    fn tag_buf(s: &str) -> [u8; 24] {
        let mut buf = [0u8; 24];
        let b = s.as_bytes();
        buf[..b.len().min(24)].copy_from_slice(&b[..b.len().min(24)]);
        buf
    }

    /// Build a minimal AUF payload with `WEIGHTS_CPU_AOS` + empty TensorIndex.
    fn build_minimal_auf() -> Vec<u8> {
        // TensorIndex with the CPU_AOS variant tag but no tensor entries.
        let tensor_index = TensorIndex {
            variant_tags: vec![tag_buf(TAG_WEIGHTS_CPU_AOS)],
            entries: vec![],
        };
        AufWriter::new(auf_meta(), auf_tokenizer(), [0u8; 32], 0, 0)
            .with_tensor_index(tensor_index)
            .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![])
            .build()
            .unwrap()
    }

    /// Build a stub `Arc<SecondaryMmap>` for lifetime tests.
    fn make_secondary() -> Arc<SecondaryMmap> {
        let auf_bytes = build_minimal_auf();
        let view = crate::auf::reader::open_from_bytes(auf_bytes, BackendTag::CpuAos).unwrap();
        let config = model_config();
        let secondary = build_auf_secondary_from_view(
            view,
            &config,
            std::path::Path::new("stub.auf"),
            BackendTag::CpuAos,
            SecondaryDtypeChoice::Auto,
        )
        .unwrap();
        Arc::new(secondary)
    }

    // ── Lifecycle tests ──────────────────────────────────────────────────────

    #[test]
    fn test_basic_lifecycle_getters() {
        let data = [0xAAu8, 0xBBu8, 0xCCu8, 0xDDu8];
        let secondary = make_secondary();

        let buf = BorrowedMmapBuffer::new(&secondary, &data, DType::F16);

        assert_eq!(buf.size(), 4);
        assert_eq!(buf.dtype(), DType::F16);
        assert_eq!(buf.as_ptr(), data.as_ptr());
        assert!(buf.as_mut_ptr().is_null());
        assert!(buf.cl_mem().is_none());
        assert!(buf.sync_device().is_ok());
    }

    #[test]
    fn test_arc_refcount_incremented_while_alive() {
        let data = [0u8; 8];
        let secondary = make_secondary();

        let count_before = Arc::strong_count(&secondary);
        assert_eq!(count_before, 1);

        let buf = BorrowedMmapBuffer::new(&secondary, &data, DType::F32);
        let count_during = Arc::strong_count(&secondary);
        assert!(
            count_during >= 2,
            "strong_count must be >= 2 while buffer is alive"
        );

        drop(buf);
        let count_after = Arc::strong_count(&secondary);
        assert_eq!(
            count_after, count_before,
            "strong_count must return to baseline after drop"
        );
    }

    #[test]
    fn test_as_any_downcast() {
        let data = [0u8; 8];
        let secondary = make_secondary();

        let buf: Arc<dyn Buffer> = Arc::new(BorrowedMmapBuffer::new(&secondary, &data, DType::F32));

        assert!(buf.as_any().downcast_ref::<BorrowedMmapBuffer>().is_some());
        assert!(
            buf.as_any()
                .downcast_ref::<crate::buffer::shared_buffer::SharedBuffer>()
                .is_none()
        );
    }

    #[test]
    fn test_default_trait_method_impls() {
        let data = [0u8; 4];
        let secondary = make_secondary();
        let buf = BorrowedMmapBuffer::new(&secondary, &data, DType::Q4_0);

        assert!(
            buf.is_host_managed(),
            "is_host_managed() should default to true"
        );
        assert!(
            !buf.is_gpu_buffer(),
            "is_gpu_buffer() should default to false"
        );
        assert!(buf.map_for_cpu().is_ok());
        assert!(buf.unmap_for_gpu().is_ok());
        assert!(buf.is_mapped());
    }
}
