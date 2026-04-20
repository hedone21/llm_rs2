//! Qualcomm Adreno OpenCL extension support.
//!
//! This module provides runtime detection of Qualcomm-specific OpenCL
//! extensions and defines the raw constants required to build
//! `cl_mem_ext_host_ptr` structures for allocations that bypass the
//! default CPU cache-maintenance path on Adreno UMA.
//!
//! Milestone 1 of the Doppeladler adoption plan (see
//! `.agent/research/2026-04-20_doppeladler_adoption_summary.md`). Only
//! the infrastructure is added here; production forward paths are not
//! touched until Milestone 2.
//!
//! # References
//!
//! - `cl_qcom_ext_host_ptr`
//!   <https://registry.khronos.org/OpenCL/extensions/qcom/cl_qcom_ext_host_ptr.txt>
//! - `cl_qcom_ext_host_ptr_iocoherent`
//!   <https://registry.khronos.org/OpenCL/extensions/qcom/cl_qcom_ext_host_ptr_iocoherent.txt>
//! - `cl_qcom_ion_host_ptr` (struct layout reference)
//!   <https://registry.khronos.org/OpenCL/extensions/qcom/cl_qcom_ion_host_ptr.txt>

#![allow(dead_code)]

use ocl::Device;

// ── Qualcomm extension constants ────────────────────────────────────
//
// Values are copied verbatim from the public Khronos Qualcomm extension
// specs. See URLs above. Any change must be double-checked against the
// spec — an incorrect bit pattern may be silently accepted by some
// Adreno drivers and crash on others.

/// `clCreateBuffer` flag that signals the `host_ptr` argument is a
/// pointer to a `cl_mem_ext_host_ptr` structure.
///
/// Spec: `cl_qcom_ext_host_ptr`, "New Tokens", `(1 << 29)`.
///
/// Must be combined with `CL_MEM_USE_HOST_PTR` per the spec
/// ("valid only when used together with CL_MEM_USE_HOST_PTR").
pub const CL_MEM_EXT_HOST_PTR_QCOM: u64 = 1 << 29;

// Accepted values for `cl_mem_ext_host_ptr::host_cache_policy`.
// Spec: `cl_qcom_ext_host_ptr` (UNCACHED/WRITEBACK/WRITETHROUGH/WRITE_COMBINING)
// + `cl_qcom_ext_host_ptr_iocoherent` (IOCOHERENT).

/// `CL_MEM_HOST_UNCACHED_QCOM = 0x40A4`
pub const CL_MEM_HOST_UNCACHED_QCOM: u32 = 0x40A4;
/// `CL_MEM_HOST_WRITEBACK_QCOM = 0x40A5`
pub const CL_MEM_HOST_WRITEBACK_QCOM: u32 = 0x40A5;
/// `CL_MEM_HOST_WRITETHROUGH_QCOM = 0x40A6`
pub const CL_MEM_HOST_WRITETHROUGH_QCOM: u32 = 0x40A6;
/// `CL_MEM_HOST_WRITE_COMBINING_QCOM = 0x40A7`
pub const CL_MEM_HOST_WRITE_COMBINING_QCOM: u32 = 0x40A7;
/// `CL_MEM_HOST_IOCOHERENT_QCOM = 0x40A9`
///
/// Spec note: "can only be specified when the memory was originally
/// allocated as cached. Use of this value with an uncached allocation
/// will lead to undefined results."
pub const CL_MEM_HOST_IOCOHERENT_QCOM: u32 = 0x40A9;

// Device info tokens (for padding / page size queries).

/// `CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM = 0x40A0`
pub const CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM: u32 = 0x40A0;
/// `CL_DEVICE_PAGE_SIZE_QCOM = 0x40A1`
pub const CL_DEVICE_PAGE_SIZE_QCOM: u32 = 0x40A1;

/// Layout of `cl_mem_ext_host_ptr` as declared in the Qualcomm spec.
///
/// ```c
/// typedef struct _cl_mem_ext_host_ptr {
///     cl_uint allocation_type;
///     cl_uint host_cache_policy;
/// } cl_mem_ext_host_ptr;
/// ```
///
/// Layered extensions (e.g. `cl_qcom_ion_host_ptr`) extend this by
/// embedding the base struct as the first field of a larger struct.
/// For the plain iocoherent path we pass `allocation_type = 0`: the
/// iocoherent spec itself does not define a new allocation-type token,
/// but community references (Qualcomm SDK notes) indicate `0` is
/// accepted on Adreno for the base-struct-only invocation. This is the
/// empirically-validated convention for the non-ION iocoherent path.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ClMemExtHostPtrQcom {
    /// Allocation type (`0` for base-struct-only, plain iocoherent).
    pub allocation_type: u32,
    /// One of `CL_MEM_HOST_*_QCOM`.
    pub host_cache_policy: u32,
}

/// Host cache policy requested for a buffer allocation.
///
/// `Default` matches the existing `UnifiedBuffer::new()` behaviour
/// (`CL_MEM_ALLOC_HOST_PTR` without any Qualcomm extension). The other
/// variants map 1:1 to `cl_mem_ext_host_ptr::host_cache_policy` values
/// from the `cl_qcom_ext_host_ptr` / `cl_qcom_ext_host_ptr_iocoherent`
/// Khronos specifications.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QcomCachePolicy {
    /// Standard `CL_MEM_ALLOC_HOST_PTR` path (no Qualcomm extension).
    ///
    /// Used when the platform does not expose `cl_qcom_ext_host_ptr`
    /// or when the caller opts out of the extension path. This is the
    /// current production default.
    Default,
    /// `CL_MEM_HOST_IOCOHERENT_QCOM` — GPU maps the host allocation as
    /// io-coherent, eliminating explicit CPU cache flush/invalidate
    /// calls issued by the driver on buffer boundaries.
    IoCoherent,
    /// `CL_MEM_HOST_WRITEBACK_QCOM` — explicit writeback cache policy
    /// (close to the default `CL_MEM_ALLOC_HOST_PTR` behaviour; useful
    /// as a control condition when A/B-testing iocoherent).
    WriteBack,
    /// `CL_MEM_HOST_WRITETHROUGH_QCOM` — write-through cache.
    WriteThrough,
    /// `CL_MEM_HOST_WRITE_COMBINING_QCOM` — write-combining memory,
    /// optimised for streaming CPU writes, poor for CPU reads.
    WriteCombining,
    /// `CL_MEM_HOST_UNCACHED_QCOM` — CPU cache bypassed entirely.
    Uncached,
}

impl QcomCachePolicy {
    /// Returns the `cl_mem_ext_host_ptr::host_cache_policy` token for
    /// this policy, or `None` for `Default` (which does not use the
    /// extension path).
    pub fn host_cache_policy_token(&self) -> Option<u32> {
        match self {
            QcomCachePolicy::Default => None,
            QcomCachePolicy::IoCoherent => Some(CL_MEM_HOST_IOCOHERENT_QCOM),
            QcomCachePolicy::WriteBack => Some(CL_MEM_HOST_WRITEBACK_QCOM),
            QcomCachePolicy::WriteThrough => Some(CL_MEM_HOST_WRITETHROUGH_QCOM),
            QcomCachePolicy::WriteCombining => Some(CL_MEM_HOST_WRITE_COMBINING_QCOM),
            QcomCachePolicy::Uncached => Some(CL_MEM_HOST_UNCACHED_QCOM),
        }
    }
}

/// Result of probing a device for Qualcomm OpenCL extensions.
///
/// All fields default to `false` on non-Adreno devices (NVIDIA, AMD,
/// Intel, etc.); callers should treat `false` as "fall back to the
/// standard `CL_MEM_ALLOC_HOST_PTR` path".
#[derive(Copy, Clone, Debug, Default)]
pub struct QcomCapabilities {
    /// `cl_qcom_ext_host_ptr` — foundation extension. Required for any
    /// use of `CL_MEM_EXT_HOST_PTR_QCOM` or `cl_mem_ext_host_ptr`.
    pub ext_host_ptr: bool,
    /// `cl_qcom_ext_host_ptr_iocoherent` — io-coherent host allocation
    /// path. Layered on top of `ext_host_ptr`.
    pub iocoherent: bool,
    /// `cl_qcom_ion_host_ptr` — ION fd import path. Not used by
    /// Milestone 1, but probed for completeness.
    pub ion_host_ptr: bool,
    /// Device `CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM` value (bytes
    /// of padding to append to every `CL_MEM_EXT_HOST_PTR_QCOM`
    /// allocation). `0` on non-Adreno or when the query fails.
    pub ext_mem_padding_bytes: usize,
    /// Device `CL_DEVICE_PAGE_SIZE_QCOM` value. Host allocations used
    /// with `CL_MEM_EXT_HOST_PTR_QCOM` must be aligned to this. `0` on
    /// non-Adreno or when the query fails; callers should default to
    /// a conservative 4096 B alignment in that case.
    pub page_size_bytes: usize,
}

impl QcomCapabilities {
    /// True if the device supports the full iocoherent stack
    /// (both `ext_host_ptr` and `ext_host_ptr_iocoherent`). The
    /// layered `ext_host_ptr_iocoherent` extension formally requires
    /// `ext_host_ptr`, so we conservatively check both.
    pub fn supports_iocoherent(&self) -> bool {
        self.ext_host_ptr && self.iocoherent
    }

    /// True if any Qualcomm host-ptr extension is available.
    pub fn any_host_ptr_ext(&self) -> bool {
        self.ext_host_ptr || self.iocoherent || self.ion_host_ptr
    }
}

/// Parse an `CL_DEVICE_EXTENSIONS` string (space-separated tokens)
/// for the Qualcomm extensions we care about. Kept as a pure fn to
/// enable unit testing without a live OpenCL device.
fn parse_extensions(extensions: &str) -> QcomCapabilities {
    let mut caps = QcomCapabilities::default();
    for tok in extensions.split_ascii_whitespace() {
        match tok {
            "cl_qcom_ext_host_ptr" => caps.ext_host_ptr = true,
            "cl_qcom_ext_host_ptr_iocoherent" => caps.iocoherent = true,
            "cl_qcom_ion_host_ptr" => caps.ion_host_ptr = true,
            _ => {}
        }
    }
    caps
}

/// Probe the given OpenCL device for Qualcomm extensions.
///
/// On non-Adreno platforms all boolean fields are `false` and byte
/// fields are `0`; the caller should treat the result as "use the
/// standard `CL_MEM_ALLOC_HOST_PTR` path".
pub fn probe(device: &Device) -> QcomCapabilities {
    let extensions_str = device
        .info(ocl::core::DeviceInfo::Extensions)
        .map(|v| v.to_string())
        .unwrap_or_default();
    let mut caps = parse_extensions(&extensions_str);

    if caps.ext_host_ptr {
        // Query the two integer device infos defined by
        // cl_qcom_ext_host_ptr. Failure is non-fatal: we simply keep
        // the default `0` values and let the caller apply a
        // conservative fallback (4 KiB page).
        caps.ext_mem_padding_bytes =
            query_device_uint(device, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM) as usize;
        caps.page_size_bytes = query_device_uint(device, CL_DEVICE_PAGE_SIZE_QCOM) as usize;
    }

    caps
}

/// Query a `cl_uint`-typed device info using the raw C FFI, for token
/// values that are not exposed by `ocl::core::DeviceInfo`.
fn query_device_uint(device: &Device, param_name: u32) -> u32 {
    use ocl::core::ClDeviceIdPtr;
    use ocl::ffi::{cl_device_info, cl_uint, clGetDeviceInfo};
    let mut value: cl_uint = 0;
    let mut size_ret: usize = 0;
    // SAFETY: clGetDeviceInfo follows the standard OpenCL ABI. We pass
    // a valid device id (the one owned by `device`), a properly-sized
    // output buffer (`sizeof(cl_uint)`), and accept any returned error
    // code by silently returning 0. The Qualcomm spec defines these
    // tokens as `cl_uint`, so the buffer size matches.
    let err = unsafe {
        clGetDeviceInfo(
            device.as_ptr(),
            param_name as cl_device_info,
            std::mem::size_of::<cl_uint>(),
            &mut value as *mut cl_uint as *mut _,
            &mut size_ret as *mut _,
        )
    };
    if err == 0 { value as u32 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_adreno_style_extensions() {
        let s = "cl_khr_byte_addressable_store cl_qcom_ext_host_ptr \
                 cl_qcom_ext_host_ptr_iocoherent cl_qcom_ion_host_ptr \
                 cl_img_mem_wrapper";
        let caps = parse_extensions(s);
        assert!(caps.ext_host_ptr);
        assert!(caps.iocoherent);
        assert!(caps.ion_host_ptr);
        assert!(caps.supports_iocoherent());
    }

    #[test]
    fn parse_nvidia_style_extensions() {
        // NVIDIA and similar: no Qualcomm extensions.
        let s = "cl_khr_fp64 cl_khr_icd cl_nv_device_attribute_query";
        let caps = parse_extensions(s);
        assert!(!caps.ext_host_ptr);
        assert!(!caps.iocoherent);
        assert!(!caps.ion_host_ptr);
        assert!(!caps.supports_iocoherent());
        assert!(!caps.any_host_ptr_ext());
    }

    #[test]
    fn parse_empty_extensions() {
        let caps = parse_extensions("");
        assert!(!caps.any_host_ptr_ext());
    }

    #[test]
    fn parse_ext_host_ptr_only_no_iocoherent() {
        // A device can expose the base extension without the
        // iocoherent layered extension — `supports_iocoherent()`
        // must return false here.
        let s = "cl_qcom_ext_host_ptr cl_khr_fp16";
        let caps = parse_extensions(s);
        assert!(caps.ext_host_ptr);
        assert!(!caps.iocoherent);
        assert!(!caps.supports_iocoherent());
    }

    #[test]
    fn host_cache_policy_token_values() {
        // Verify the enum maps to the exact tokens from the Qualcomm spec.
        assert_eq!(
            QcomCachePolicy::IoCoherent.host_cache_policy_token(),
            Some(0x40A9)
        );
        assert_eq!(
            QcomCachePolicy::WriteBack.host_cache_policy_token(),
            Some(0x40A5)
        );
        assert_eq!(
            QcomCachePolicy::WriteThrough.host_cache_policy_token(),
            Some(0x40A6)
        );
        assert_eq!(
            QcomCachePolicy::WriteCombining.host_cache_policy_token(),
            Some(0x40A7)
        );
        assert_eq!(
            QcomCachePolicy::Uncached.host_cache_policy_token(),
            Some(0x40A4)
        );
        assert_eq!(QcomCachePolicy::Default.host_cache_policy_token(), None);
    }

    #[test]
    fn cl_mem_ext_host_ptr_layout_is_two_u32s() {
        // Spec: struct of { cl_uint allocation_type; cl_uint host_cache_policy; }
        // is exactly 8 bytes on every sane ABI. Guard against accidental
        // padding/alignment changes — the Adreno driver reads raw bytes.
        assert_eq!(std::mem::size_of::<ClMemExtHostPtrQcom>(), 8);
        assert_eq!(std::mem::align_of::<ClMemExtHostPtrQcom>(), 4);
    }

    #[test]
    fn ext_host_ptr_flag_bit_position() {
        // Spec: CL_MEM_EXT_HOST_PTR_QCOM = (1 << 29)
        assert_eq!(CL_MEM_EXT_HOST_PTR_QCOM, 0x20000000);
    }
}
