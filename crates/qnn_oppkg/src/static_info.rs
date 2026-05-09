//! Static op-package metadata.
//!
//! All allocations are intentionally leaked (`Box::leak`) so QNN may hold the
//! raw pointers indefinitely. The leak is one-shot and bounded by the static
//! `OnceLock`, not per-call.

use crate::qnn::{
    QNN_OP_PACKAGE_RESERVED_INFO_SIZE, Qnn_ApiVersion_t, Qnn_Version_t, QnnOpPackage_Info_t,
    QnnOpPackage_PackageInfo_t,
};
use crate::registry::OPS;
use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;
use std::sync::OnceLock;

/// Production package name. Distinct from the PoC crate (`qnn_oppkg_poc`) so
/// both libraries can be loaded into the same process during validation.
const PACKAGE_NAME: &str = "qnn_oppkg";

/// SDK build id — must match the QNN backend exactly to avoid rejection.
const SDK_BUILD_ID: &str = "v2.33.0.250327124043_117917";

pub struct StaticInfo {
    pub info: &'static QnnOpPackage_Info_t,
}

unsafe impl Sync for StaticInfo {}
unsafe impl Send for StaticInfo {}

static STATIC_INFO: OnceLock<StaticInfo> = OnceLock::new();

/// Initialise the leaked static metadata on first call. Subsequent calls return
/// the same reference.
pub fn ensure_static() -> &'static StaticInfo {
    STATIC_INFO.get_or_init(|| {
        let package_name: &'static CString =
            Box::leak(Box::new(CString::new(PACKAGE_NAME).unwrap()));

        // operationNames mirrors OPS in declaration order. Empty slice when
        // OPS is empty (M1.2). INV-152 enforces OPS.len() == numOperations.
        let op_name_cstrings: Vec<&'static CString> = OPS
            .iter()
            .map(|d| -> &'static CString { Box::leak(Box::new(CString::new(d.op_type).unwrap())) })
            .collect();
        let op_name_ptrs: Vec<*const c_char> =
            op_name_cstrings.iter().map(|c| c.as_ptr()).collect();
        let operation_names: &'static [*const c_char] = Box::leak(op_name_ptrs.into_boxed_slice());

        // INV-154: backend / core API versions. Hardcoded — must NOT be exposed
        // via env or CLI. Mirrors QNN_GPU_API_VERSION_INIT in QnnGpuCommon.h.
        let api_version: &'static Qnn_ApiVersion_t = Box::leak(Box::new(Qnn_ApiVersion_t {
            coreApiVersion: Qnn_Version_t {
                major: 2,
                minor: 25,
                patch: 0,
            },
            backendApiVersion: Qnn_Version_t {
                major: 3,
                minor: 7,
                patch: 0,
            },
        }));
        let opset_version: &'static Qnn_Version_t = Box::leak(Box::new(Qnn_Version_t {
            major: 1,
            minor: 0,
            patch: 0,
        }));
        let build_id: &'static CString = Box::leak(Box::new(CString::new(SDK_BUILD_ID).unwrap()));
        let hash: &'static CString = Box::leak(Box::new(CString::new("qnn_oppkg_v1").unwrap()));

        // GPU-specialised package info. Mirrors PoC layout.
        #[repr(C)]
        struct GpuPackageInfo {
            kernel_repo_hash: *const c_char,
        }
        let gpu_pkg_info: &'static GpuPackageInfo = Box::leak(Box::new(GpuPackageInfo {
            kernel_repo_hash: hash.as_ptr(),
        }));

        // INV-152: numOperations is statically derived from OPS.len().
        let num_ops = OPS.len() as u32;
        let info: &'static QnnOpPackage_Info_t = Box::leak(Box::new(QnnOpPackage_Info_t {
            packageName: package_name.as_ptr(),
            operationNames: operation_names.as_ptr() as *mut _,
            operationInfo: ptr::null(),
            numOperations: num_ops,
            optimizations: ptr::null(),
            numOptimizations: 0,
            sdkBuildId: build_id.as_ptr(),
            sdkApiVersion: api_version,
            packageInfo: gpu_pkg_info as *const _ as *const QnnOpPackage_PackageInfo_t,
            opsetVersion: opset_version,
            reserved: [0; QNN_OP_PACKAGE_RESERVED_INFO_SIZE as usize],
        }));

        StaticInfo { info }
    })
}
