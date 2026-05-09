//! V1.4 / V2.0 op-package interface entry. Mirrors PoC behaviour: chooses V2.0
//! when env `OPPKG_V2` is present, otherwise V1.4.
//!
//! These extern "C" fns dereference raw pointers as part of the QNN ABI; the
//! signatures are fixed by `QnnOpPackage_*` typedefs and cannot be marked
//! `unsafe fn`. Each deref site sits inside an `unsafe { … }` block.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::op_impl::{pkg_create_op_impl, pkg_free_op_impl};
use crate::qnn::{
    QNN_SUCCESS, Qnn_ErrorHandle_t, Qnn_OpConfig_t, Qnn_OpPackageHandle_t, Qnn_Version_t,
    QnnLog_Callback_t, QnnLog_Level_t, QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT,
    QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_INFO, QnnOpPackage_GlobalInfrastructure_t,
    QnnOpPackage_GraphInfrastructure_t, QnnOpPackage_ImplementationV1_4_t,
    QnnOpPackage_ImplementationV2_0_t, QnnOpPackage_Info_t, QnnOpPackage_Interface_t,
    QnnOpPackage_Node_t, QnnOpPackage_OpImpl_t,
};
use crate::static_info::ensure_static;

#[unsafe(no_mangle)]
pub extern "C" fn pkg_init(
    _infrastructure: QnnOpPackage_GlobalInfrastructure_t,
) -> Qnn_ErrorHandle_t {
    ensure_static();
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_create(
    _infrastructure: QnnOpPackage_GlobalInfrastructure_t,
    _callback: QnnLog_Callback_t,
    _max_level: QnnLog_Level_t,
    op_package: *mut Qnn_OpPackageHandle_t,
) -> Qnn_ErrorHandle_t {
    if op_package.is_null() {
        return QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT as Qnn_ErrorHandle_t;
    }
    let s = ensure_static();
    unsafe { *op_package = s as *const _ as Qnn_OpPackageHandle_t };
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_validate_op_config(_op_config: Qnn_OpConfig_t) -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_validate_op_config_h(
    _handle: Qnn_OpPackageHandle_t,
    _op_config: Qnn_OpConfig_t,
) -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_create_op_impl_h(
    _handle: Qnn_OpPackageHandle_t,
    graph_infra: QnnOpPackage_GraphInfrastructure_t,
    node: QnnOpPackage_Node_t,
    op_impl: *mut QnnOpPackage_OpImpl_t,
) -> Qnn_ErrorHandle_t {
    pkg_create_op_impl(graph_infra, node, op_impl)
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_free_op_impl_h(
    _handle: Qnn_OpPackageHandle_t,
    op_impl: QnnOpPackage_OpImpl_t,
) -> Qnn_ErrorHandle_t {
    pkg_free_op_impl(op_impl)
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_log_set_level_h(
    _handle: Qnn_OpPackageHandle_t,
    _max_level: QnnLog_Level_t,
) -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_free(_handle: Qnn_OpPackageHandle_t) -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_terminate() -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_get_info(info: *mut *const QnnOpPackage_Info_t) -> Qnn_ErrorHandle_t {
    if info.is_null() {
        return QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_INFO as Qnn_ErrorHandle_t;
    }
    let s = ensure_static();
    unsafe { *info = s.info };
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_log_initialize(
    _callback: QnnLog_Callback_t,
    _max_level: QnnLog_Level_t,
) -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_log_set_level(_max_level: QnnLog_Level_t) -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

#[unsafe(no_mangle)]
pub extern "C" fn pkg_log_terminate() -> Qnn_ErrorHandle_t {
    QNN_SUCCESS as Qnn_ErrorHandle_t
}

/// Diagnostics: number of live `OpImplState` entries in `STATE_MAP`. Exposed
/// for the M1.8 leak microbench to distinguish leaks owned by this cdylib
/// (entries grow without bound) from external leaks (entries stay flat).
///
/// Returns `usize` cast to `u64` for FFI safety.
#[unsafe(no_mangle)]
pub extern "C" fn qnn_oppkg_state_map_len() -> u64 {
    crate::op_impl::state_map_len() as u64
}

/// Entry point exposed to QNN runtime via `registerOpPackage(...interfaceProvider...)`.
///
/// Selects V1.4 by default; V2.0 when `OPPKG_V2` is set in env.
#[unsafe(no_mangle)]
pub extern "C" fn QnnOpPackage_InitInterface(
    interface: *mut QnnOpPackage_Interface_t,
) -> Qnn_ErrorHandle_t {
    if interface.is_null() {
        return QnnOpPackage_Error_t_QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT as Qnn_ErrorHandle_t;
    }
    let use_v2 = std::env::var("OPPKG_V2").is_ok();
    unsafe {
        if use_v2 {
            (*interface).interfaceVersion = Qnn_Version_t {
                major: 2,
                minor: 0,
                patch: 0,
            };
            (*interface).__bindgen_anon_1.v2_0 = QnnOpPackage_ImplementationV2_0_t {
                create: Some(pkg_create),
                getInfo: Some(pkg_get_info),
                validateOpConfig: Some(pkg_validate_op_config_h),
                createOpImpl: Some(pkg_create_op_impl_h),
                freeOpImpl: Some(pkg_free_op_impl_h),
                logSetLevel: Some(pkg_log_set_level_h),
                free: Some(pkg_free),
            };
        } else {
            (*interface).interfaceVersion = Qnn_Version_t {
                major: 1,
                minor: 4,
                patch: 0,
            };
            (*interface).__bindgen_anon_1.v1_4 = QnnOpPackage_ImplementationV1_4_t {
                init: Some(pkg_init),
                terminate: Some(pkg_terminate),
                getInfo: Some(pkg_get_info),
                validateOpConfig: Some(pkg_validate_op_config),
                createOpImpl: Some(pkg_create_op_impl),
                freeOpImpl: Some(pkg_free_op_impl),
                logInitialize: Some(pkg_log_initialize),
                logSetLevel: Some(pkg_log_set_level),
                logTerminate: Some(pkg_log_terminate),
            };
        }
    }
    QNN_SUCCESS as Qnn_ErrorHandle_t
}
