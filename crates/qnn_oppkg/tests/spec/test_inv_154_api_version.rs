//! Spec test for INV-154 (backendApiVersion = 3.7.0, coreApiVersion = 2.25.0).

use qnn_oppkg::__test_support::{QnnOpPackage_Info_t, pkg_get_info};
use qnn_oppkg::ensure_static;

#[test]
fn api_versions_are_locked() {
    let info = ensure_static();
    let api = unsafe { &*info.info.sdkApiVersion };
    assert_eq!(api.backendApiVersion.major, 3, "INV-154: backend major");
    assert_eq!(api.backendApiVersion.minor, 7, "INV-154: backend minor");
    assert_eq!(api.backendApiVersion.patch, 0, "INV-154: backend patch");
    assert_eq!(api.coreApiVersion.major, 2, "INV-154: core major");
    assert_eq!(api.coreApiVersion.minor, 25, "INV-154: core minor");
    assert_eq!(api.coreApiVersion.patch, 0, "INV-154: core patch");
}

#[test]
fn pkg_get_info_returns_locked_versions() {
    // Exercise the FFI path that QNN actually calls.
    let mut info_ptr: *const QnnOpPackage_Info_t = std::ptr::null();
    let rc = pkg_get_info(&mut info_ptr as *mut _);
    assert_eq!(rc, 0, "pkg_get_info should return QNN_SUCCESS");
    assert!(!info_ptr.is_null());
    let info = unsafe { &*info_ptr };
    let api = unsafe { &*info.sdkApiVersion };
    assert_eq!(api.backendApiVersion.major, 3);
    assert_eq!(api.backendApiVersion.minor, 7);
    assert_eq!(api.backendApiVersion.patch, 0);
}
