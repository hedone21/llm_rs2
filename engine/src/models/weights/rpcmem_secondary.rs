//! `RpcmemSecondaryStore` — LISWAP-6 DMA-BUF alias backing for secondary GGUF
//! weights when the primary backend is `qnn_oppkg`.
//!
//! ## 동기
//!
//! LISWAP-5 v1 fair comparison (postmortem §6, §7.6) 결과 swap copy 비용 (~480 ms
//! blocking) 이 단일-shot, per-tick=1, phase-aware 모든 모드에서 핵심 cost 임이
//! 확인되었다. `qnn_oppkg` backend의 KV cache가 이미 rpcmem heap + OpenCL
//! `CL_MEM_USE_HOST_PTR` alias로 zero-copy 가속을 검증했으므로 (`hybrid_memory.rs::android_alloc_kv`),
//! 본 모듈은 동일 패턴을 secondary weight 로 확장하여 H2D copy 자체를 제거한다.
//!
//! ## 설계 — Lazy per-layer alloc
//!
//! Model load 시점에 rpcmem alloc을 하지 않는다. 첫 swap target layer 가 접근될
//! 때 (`ensure_layer_loaded`) 그 layer 의 모든 tensor 를 단일 rpcmem region 으로
//! alloc + memcpy from mmap. 한 번 alloc 된 layer region 은 model lifetime 까지
//! 유지되어 두 번째 swap 부터는 cache hit 으로 alias 만 새로 만든다.
//!
//! 메모리 피크 분석:
//! - 전체 swap 진행 시: 25 layer × ~50 MB = ~1.25 GB peak (Galaxy S25 12 GB RAM 에서 OK)
//! - swap 일부만 진행: 해당 layer 만 alloc → 피크 감소
//! - swap 미발생: rpcmem 0 byte
//!
//! ## Lifetime / Drop
//!
//! - `RpcmemLayerRegion::Drop` 이 `rpcmem_free(host_ptr)` 호출 → 해당 region 의
//!   모든 외부 cl_mem alias 가 먼저 drop 된 이후에야 안전.
//! - alias buffer 는 `Arc<SecondaryMmap>` 보유 → secondary store 가 살아있는 동안
//!   region map 도 유지 → region drop 은 store drop 시점까지 지연.
//!
//! Spec: ENG-DAT-094 (SecondaryMmap), INV-143 (alias lifetime), ENG-QNN-204
//! (rpcmem heap reuse).

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::core::buffer::DType;
use crate::models::config::ModelConfig;
use crate::models::loader::gguf::GgufFile;
#[cfg(target_os = "android")]
use crate::models::loader::gguf::{ggml_type_to_dtype, tensor_byte_size};
#[cfg(target_os = "android")]
use crate::models::weights::secondary_mmap::parse_block_tensor_name;
use crate::models::weights::secondary_mmap::{LayerTensorSlice, SecondaryTensorInfo};

/// rpcmem alloc/free fn-pointer pair (Android-only). On non-Android targets,
/// the store rejects construction so the field is gated.
#[cfg(target_os = "android")]
type RpcmemAllocFn = unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void;
#[cfg(target_os = "android")]
type RpcmemFreeFn = unsafe extern "C" fn(*mut std::ffi::c_void);

/// Per-layer rpcmem region. Owns one `rpcmem_alloc` allocation that holds
/// every swap-relevant tensor for the layer, with `tensor_map` describing
/// each tensor's offset/length within the region.
pub struct RpcmemLayerRegion {
    /// Base host pointer returned by `rpcmem_alloc` (mmap'd DMA-BUF region).
    host_ptr: *mut u8,
    /// Total bytes allocated. Read by Drop diagnostics + size sanity tests.
    /// Allowed to be `dead_code` on non-Android because the Drop arm is gated.
    #[allow(dead_code)]
    size: usize,
    /// Subname (e.g. "attn_q.weight") → (offset_in_region, byte_len, dtype).
    tensor_map: HashMap<String, (usize, usize, DType)>,
    /// rpcmem_free fn-pointer cached so Drop doesn't need a runtime handle.
    /// Android-only — host targets never construct this struct.
    #[cfg(target_os = "android")]
    rpcmem_free: RpcmemFreeFn,
}

// SAFETY: rpcmem_alloc returns a host-accessible mapping. The underlying
// memory is single-owner (this struct) and protected by `Mutex` at the
// `RpcmemSecondaryStore` level for HashMap insertions. fn-pointer is `Copy`
// and stateless. Drop runs at most once.
unsafe impl Send for RpcmemLayerRegion {}
unsafe impl Sync for RpcmemLayerRegion {}

impl RpcmemLayerRegion {
    /// Resolve a tensor's host pointer + length within this region.
    ///
    /// Returns `(host_ptr_base, offset, len, dtype)`. The caller adds
    /// `offset` to `host_ptr_base` to obtain the tensor's start address.
    pub fn tensor_info(&self, subname: &str) -> Option<(*mut u8, usize, usize, DType)> {
        let (offset, len, dtype) = self.tensor_map.get(subname).copied()?;
        Some((self.host_ptr, offset, len, dtype))
    }
}

/// Result of `RpcmemSecondaryStore::host_ptr_for` — bundles the lifetime
/// guard (`region` Arc) with the alias coordinates so the caller can hand
/// both to `Backend::alloc_alias_weight_buffer` without hitting the
/// "very complex type" clippy lint.
pub struct HostPtrAlias {
    pub region: Arc<RpcmemLayerRegion>,
    pub host_ptr: *mut u8,
    pub offset: usize,
    pub len: usize,
    pub dtype: DType,
}

impl Drop for RpcmemLayerRegion {
    fn drop(&mut self) {
        #[cfg(target_os = "android")]
        if !self.host_ptr.is_null() {
            // SAFETY: host_ptr produced by rpcmem_alloc; Drop runs once;
            // alias lifetime invariant guarantees no live cl_mem references.
            unsafe { (self.rpcmem_free)(self.host_ptr.cast()) };
        }
    }
}

/// Lazy per-layer rpcmem-backed secondary weight store.
///
/// Owns the secondary GGUF mmap (used as the memcpy source for first-touch)
/// plus a `Mutex<HashMap>` of allocated per-layer regions. The store itself
/// does not allocate any rpcmem at construction time.
pub struct RpcmemSecondaryStore {
    /// Secondary GGUF mmap — kept alive for layout parsing and lazy memcpy.
    backing_mmap: Arc<GgufFile>,
    /// Source path for diagnostics.
    source_path: PathBuf,
    /// Pre-parsed per-layer tensor index (subname → SecondaryTensorInfo).
    /// Used by `layer_tensor()` callers and by `ensure_layer_loaded` to know
    /// which tensors to memcpy into the rpcmem region.
    layer_index: Vec<LayerTensorSlice>,
    /// Per-layer rpcmem region cache. `Mutex` synchronises concurrent
    /// `ensure_layer_loaded` calls; lookups are O(1) once allocated.
    layer_regions: Mutex<HashMap<usize, Arc<RpcmemLayerRegion>>>,
    /// rpcmem alloc/free fn-pointers from the QNN runtime.
    /// Android-only — host targets cannot construct this store.
    #[cfg(target_os = "android")]
    rpcmem_alloc: RpcmemAllocFn,
    #[cfg(target_os = "android")]
    rpcmem_free: RpcmemFreeFn,
}

impl std::fmt::Debug for RpcmemSecondaryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RpcmemSecondaryStore")
            .field("source_path", &self.source_path)
            .field("num_layers", &self.layer_index.len())
            .field(
                "regions_loaded",
                &self.layer_regions.lock().map(|m| m.len()).unwrap_or(0),
            )
            .finish()
    }
}

impl RpcmemSecondaryStore {
    /// Open a GGUF secondary file and prepare the lazy store.
    ///
    /// `qnn_runtime_rpcmem_fns` is the `(alloc, free)` pair extracted from
    /// `QnnOppkgRuntime::rpcmem_fns()` (Android-only). On non-Android targets
    /// this constructor returns an error — callers should fall back to the
    /// regular `SecondaryMmap::Gguf` variant.
    pub fn from_gguf(
        path: &std::path::Path,
        primary_config: &ModelConfig,
        primary_gguf: &GgufFile,
        #[cfg(target_os = "android")] qnn_runtime_rpcmem_fns: (RpcmemAllocFn, RpcmemFreeFn),
    ) -> Result<Self> {
        #[cfg(not(target_os = "android"))]
        {
            let _ = (path, primary_config, primary_gguf);
            Err(anyhow!(
                "RpcmemSecondaryStore is Android-only (rpcmem heap unavailable on host targets)"
            ))
        }

        #[cfg(target_os = "android")]
        {
            let gguf = GgufFile::open(path)
                .map_err(|e| anyhow!("rpcmem_secondary: GGUF open failed: {e}"))?;

            // Validate metadata vs primary (same checks as open_secondary_gguf).
            let secondary_config = ModelConfig::from_gguf_metadata(&gguf)
                .map_err(|e| anyhow!("rpcmem_secondary: config parse failed: {e}"))?;
            check_metadata_match(primary_config, &secondary_config)?;

            // Build per-layer slice index (mirror of open_secondary_gguf logic).
            let num_layers = primary_config.num_hidden_layers;
            let mut layer_index: Vec<LayerTensorSlice> =
                vec![LayerTensorSlice::default(); num_layers];

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

            // Validate decoder tensor presence + shapes against the primary.
            for info in &primary_gguf.tensors {
                let Some((layer_idx, subname)) = parse_block_tensor_name(&info.name) else {
                    continue;
                };
                if layer_idx >= num_layers {
                    continue;
                }
                let Some(secondary_info) = layer_index[layer_idx].tensors.get(subname) else {
                    return Err(anyhow!(
                        "rpcmem_secondary: missing tensor '{}' in secondary file",
                        info.name
                    ));
                };
                if secondary_info.dims != info.dims {
                    return Err(anyhow!(
                        "rpcmem_secondary: shape mismatch for '{}': primary={:?}, secondary={:?}",
                        info.name,
                        info.dims,
                        secondary_info.dims
                    ));
                }
            }

            let (rpcmem_alloc, rpcmem_free) = qnn_runtime_rpcmem_fns;

            Ok(Self {
                backing_mmap: Arc::new(gguf),
                source_path: path.to_path_buf(),
                layer_index,
                layer_regions: Mutex::new(HashMap::new()),
                rpcmem_alloc,
                rpcmem_free,
            })
        }
    }

    /// Source path (diagnostic).
    pub fn source_path(&self) -> &std::path::Path {
        &self.source_path
    }

    /// Look up a tensor descriptor by `(layer_idx, subname)`. Mirrors
    /// `GgufSecondaryMmap::layer_tensor`.
    pub fn layer_tensor(&self, layer_idx: usize, subname: &str) -> Option<&SecondaryTensorInfo> {
        self.layer_index
            .get(layer_idx)
            .and_then(|slice| slice.tensors.get(subname))
    }

    /// Zero-copy byte slice from the underlying mmap. Used by fallback paths
    /// (qk-unpermute, GGUF source-of-truth) when the alias path is not viable.
    pub fn tensor_bytes(&self, info: &SecondaryTensorInfo) -> &[u8] {
        let end = info.offset + info.len;
        &self.backing_mmap.mmap_data()[info.offset..end]
    }

    /// Ensure the rpcmem region for `layer_idx` is allocated and populated.
    ///
    /// First call: allocates a single `rpcmem_alloc` of size = sum(layer
    /// tensors), then memcpys each tensor from the mmap into the region.
    /// Subsequent calls: O(1) HashMap hit.
    ///
    /// Returns an `Arc` so callers can share the region across multiple
    /// alias buffers (one per tensor) without lifetime gymnastics.
    pub fn ensure_layer_loaded(&self, layer_idx: usize) -> Result<Arc<RpcmemLayerRegion>> {
        // Fast path: cache hit (no rpcmem alloc).
        {
            let regions = self
                .layer_regions
                .lock()
                .map_err(|_| anyhow!("rpcmem_secondary: layer_regions mutex poisoned"))?;
            if let Some(region) = regions.get(&layer_idx) {
                return Ok(region.clone());
            }
        }

        // Slow path: build the region. Drop the lock during memcpy / alloc
        // so concurrent `ensure_layer_loaded` for *different* layers can
        // proceed. We re-acquire after the build to insert.
        let region = self.build_layer_region(layer_idx)?;

        let mut regions = self
            .layer_regions
            .lock()
            .map_err(|_| anyhow!("rpcmem_secondary: layer_regions mutex poisoned"))?;
        // Race: another thread may have built the same layer concurrently.
        // Use entry() to keep the existing one and discard ours.
        Ok(regions.entry(layer_idx).or_insert(region).clone())
    }

    /// Resolve a tensor's host_ptr + offset + len + dtype, allocating the
    /// region lazily if needed. Used by the alias buffer construction path.
    ///
    /// Returns `Ok(None)` when the tensor is missing from the secondary
    /// (caller should surface a proper error or fall back).
    pub fn host_ptr_for(&self, layer_idx: usize, subname: &str) -> Result<Option<HostPtrAlias>> {
        let region = self.ensure_layer_loaded(layer_idx)?;
        let Some((base, offset, len, dtype)) = region.tensor_info(subname) else {
            return Ok(None);
        };
        Ok(Some(HostPtrAlias {
            region,
            host_ptr: base,
            offset,
            len,
            dtype,
        }))
    }

    /// Force-allocate and populate regions for the given target layers.
    /// Used by `prefault_layers` to amortize first-touch cost.
    pub fn prefault_layers(&self, target_layers: &[usize]) {
        for &layer_idx in target_layers {
            let _ = self.ensure_layer_loaded(layer_idx);
        }
    }

    // ── internal ─────────────────────────────────────────────────────────────

    #[cfg(target_os = "android")]
    fn build_layer_region(&self, layer_idx: usize) -> Result<Arc<RpcmemLayerRegion>> {
        const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
        const RPCMEM_DEFAULT_FLAGS: u32 = 1;

        let layer_slice = self.layer_index.get(layer_idx).ok_or_else(|| {
            anyhow!(
                "rpcmem_secondary: layer_idx {layer_idx} out of range (max {})",
                self.layer_index.len()
            )
        })?;

        // Stable iteration order so the offset map is deterministic across
        // builds (HashMap iteration order is not stable but we sort here for
        // diagnostic clarity; functional correctness does not require it).
        let mut entries: Vec<(&String, &SecondaryTensorInfo)> =
            layer_slice.tensors.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        // Compute total size + per-tensor offset map.
        let mut total: usize = 0;
        let mut tensor_map: HashMap<String, (usize, usize, DType)> =
            HashMap::with_capacity(entries.len());
        for (subname, info) in &entries {
            // Align each tensor to 64-byte boundary so OpenCL `CL_MEM_USE_HOST_PTR`
            // alias respects the host alignment requirement (typically 1-byte for
            // unified memory but 64 is the safe upper bound on Adreno).
            let aligned_offset = total.next_multiple_of(64);
            tensor_map.insert((*subname).clone(), (aligned_offset, info.len, info.dtype));
            total = aligned_offset + info.len;
        }

        if total == 0 {
            return Err(anyhow!(
                "rpcmem_secondary: layer {layer_idx} has zero total tensor bytes"
            ));
        }

        // 1. rpcmem_alloc — single backing region for the whole layer.
        // SAFETY: rpcmem_alloc fn-pointer obtained from libcdsprpc.so via
        // QnnOppkgRuntime::rpcmem_fns(); validity is enforced at runtime init.
        let host_ptr = unsafe {
            (self.rpcmem_alloc)(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, total as i32)
        };
        if host_ptr.is_null() {
            return Err(anyhow!(
                "rpcmem_secondary: rpcmem_alloc(size={total}) returned NULL for layer {layer_idx}"
            ));
        }
        let host_ptr = host_ptr as *mut u8;

        // 2. memcpy each tensor from the mmap into the region. Best-effort:
        // if any copy fails (would only happen on a corrupt mmap range), free
        // the region and bubble up.
        let mmap_bytes = self.backing_mmap.mmap_data();
        for (subname, info) in &entries {
            let Some((dst_offset, len, _dt)) = tensor_map.get(*subname).copied() else {
                continue; // unreachable — we just inserted it
            };
            let src_start = info.offset;
            let src_end = info.offset + info.len;
            if src_end > mmap_bytes.len() || len != info.len {
                // Free + abort.
                #[cfg(target_os = "android")]
                unsafe {
                    (self.rpcmem_free)(host_ptr.cast());
                }
                return Err(anyhow!(
                    "rpcmem_secondary: tensor '{}' offset/len exceeds mmap (layer {layer_idx})",
                    subname
                ));
            }
            // SAFETY: `host_ptr` is `total` bytes; `dst_offset + len <= total`
            // by construction; `mmap_bytes[src_start..src_end]` is a valid
            // read-only mmap slice; src/dst do not overlap (different regions).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mmap_bytes[src_start..src_end].as_ptr(),
                    host_ptr.add(dst_offset),
                    len,
                );
            }
        }

        Ok(Arc::new(RpcmemLayerRegion {
            host_ptr,
            size: total,
            tensor_map,
            rpcmem_free: self.rpcmem_free,
        }))
    }

    #[cfg(not(target_os = "android"))]
    fn build_layer_region(&self, _layer_idx: usize) -> Result<Arc<RpcmemLayerRegion>> {
        Err(anyhow!(
            "rpcmem_secondary: build_layer_region called on non-Android target"
        ))
    }
}

/// Subset of `check_metadata` from `secondary_mmap.rs` reused locally.
/// We don't import the private `check_metadata` to avoid cross-module visibility
/// changes; instead we mirror the few invariants strictly required for swap.
///
/// Only invoked from the Android-gated `from_gguf` path; allowed to appear
/// dead on host targets where the constructor returns early.
#[allow(dead_code)]
fn check_metadata_match(primary: &ModelConfig, secondary: &ModelConfig) -> Result<()> {
    macro_rules! check {
        ($field:ident) => {
            if primary.$field != secondary.$field {
                return Err(anyhow!(
                    "rpcmem_secondary: metadata mismatch on '{}': primary={:?}, secondary={:?}",
                    stringify!($field),
                    primary.$field,
                    secondary.$field
                ));
            }
        };
    }
    check!(num_hidden_layers);
    check!(num_attention_heads);
    check!(num_key_value_heads);
    check!(hidden_size);
    check!(intermediate_size);
    check!(vocab_size);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // `RpcmemLayerRegion::size` is currently only consulted via Debug /
    // diagnostics. This compile-only probe keeps dead_code lint quiet across
    // feature combinations and documents the field's intended public surface.
    #[test]
    fn region_size_field_used() {
        let _check_field = |r: &RpcmemLayerRegion| -> usize { r.size };
    }
}
