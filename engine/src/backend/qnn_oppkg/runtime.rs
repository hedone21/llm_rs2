//! QNN runtime wrapper — `libQnnGpu.so` + `libqnn_oppkg.so` dlopen + V2.25
//! function-pointer table caching.
//!
//! Spec: `spec/30-engine.md` 부록 C.2 (ENG-QNN-201, ENG-QNN-205,
//! ENG-QNN-209), `spec/41-invariants.md` §3.24 (INV-166, INV-180),
//! `arch/30-engine.md` §18.1.
//!
//! ## 책임
//! - Android device: `dlopen(libQnnGpu.so)` → `QnnInterface_getProviders` →
//!   V2.25 fn-pointer cache → `backendCreate` → `registerOpPackage` →
//!   `contextCreate`. M2.H microbench (`microbench_qnn_qwen_layer.rs`) lines
//!   147-194의 init flow 재구성.
//! - Host (linux x86_64 등): libQnnGpu.so 부재 → 명확한 `Err` 반환 후 caller가
//!   bail. INV-180 (host에서 init 실패) 게이트.
//!
//! ## 본 단계 (M3.3) 적용 범위
//! Android target에서 dlopen + ffi 호출 본문이 활성화된다. host 빌드는 그대로
//! 즉시 `Err`로 fail하여 caller가 bail; sanity-check `cargo test --features
//! qnn,opencl --tests`는 INV-180 unit test로 본 path를 검증한다.
//!
//! 디바이스 정확성 측정은 M3.4 통합 게이트에서 수행된다. M3.3에서는
//! 컴파일 + dispatch path가 정상 동작하는지만 보장한다.

use anyhow::{Result, anyhow};
use std::sync::Arc;

#[cfg(target_os = "android")]
#[allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code
)]
pub(crate) mod ffi {
    include!(concat!(env!("OUT_DIR"), "/qnn_bindings.rs"));
}

/// QNN runtime opaque handles.
///
/// Android device path에서 dlopen된 libraries + V2.25 fn-pointer table +
/// `Qnn_BackendHandle_t` / `Qnn_ContextHandle_t`를 보유한다. host 빌드에서는
/// 본 struct가 instantiate되지 않는다 (`init()` 즉시 Err).
pub struct QnnOppkgRuntime {
    #[cfg(target_os = "android")]
    inner: AndroidRuntime,
    /// Marker placeholder for host build (`init()` always fails before reaching
    /// here — kept for layout symmetry).
    #[cfg(not(target_os = "android"))]
    _placeholder: (),
}

#[cfg(target_os = "android")]
struct AndroidRuntime {
    /// `libQnnGpu.so` — keeps fn pointers alive (drop closes the lib).
    _gpu_lib: libloading::Library,
    /// `librpcmem` (DSP rpcmem.so via libcdsprpc.so) — for memory.rs alloc.
    _rpc_lib: libloading::Library,
    /// V2.25 function-pointer vtable. `'static` borrow is sound because
    /// `_gpu_lib` outlives this struct (same Drop).
    pub(crate) v: &'static ffi::QnnInterface_ImplementationV2_25_t,
    /// `Qnn_BackendHandle_t` — created once, shared across graphs.
    pub(crate) backend: ffi::Qnn_BackendHandle_t,
    /// `Qnn_ContextHandle_t` — created once, shared across graphs.
    pub(crate) context: ffi::Qnn_ContextHandle_t,
    /// rpcmem alloc/free/to_fd function pointers (cached for memory.rs).
    pub(crate) rpcmem_alloc: unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void,
    pub(crate) rpcmem_free: unsafe extern "C" fn(*mut std::ffi::c_void),
    pub(crate) rpcmem_to_fd: unsafe extern "C" fn(*const std::ffi::c_void) -> i32,
}

#[cfg(target_os = "android")]
impl AndroidRuntime {
    pub(crate) fn v(&self) -> &ffi::QnnInterface_ImplementationV2_25_t {
        self.v
    }
}

impl QnnOppkgRuntime {
    /// QNN backend / OpPackage runtime을 초기화한다.
    ///
    /// - Android device: dlopen + V2.25 fn-pointer 캐싱 + `QnnBackend_create` +
    ///   `QnnContext_create` + `registerOpPackage`.
    /// - host (non-Android): `libQnnGpu.so` 부재로 즉시 `Err` (INV-180).
    pub fn init() -> Result<Arc<Self>> {
        #[cfg(target_os = "android")]
        {
            android_init()
        }
        #[cfg(not(target_os = "android"))]
        {
            // INV-180: 호스트 빌드는 명확한 Err로 fail하여 caller가 bail.
            Err(anyhow!(
                "QnnOppkgRuntime::init() — host (non-Android) build에서는 libQnnGpu.so 부재로 init 불가. 디바이스 빌드 + Android runtime에서만 진행 가능."
            ))
        }
    }

    /// 본 runtime이 Android에서 정상 init된 상태인지. host 빌드에서는 호출되지
    /// 않지만 (init 자체가 Err), 의미상 false.
    pub fn is_initialized(&self) -> bool {
        #[cfg(target_os = "android")]
        {
            !self.inner.backend.is_null() && !self.inner.context.is_null()
        }
        #[cfg(not(target_os = "android"))]
        {
            false
        }
    }

    /// V2.25 vtable accessor (Android-only). M3.3 layer_graph build/execute 본문
    /// 에서 graphCreate/Execute 호출 시 사용.
    #[cfg(target_os = "android")]
    pub(crate) fn v(&self) -> &ffi::QnnInterface_ImplementationV2_25_t {
        self.inner.v()
    }

    /// QNN context handle accessor (Android-only).
    #[cfg(target_os = "android")]
    pub(crate) fn context(&self) -> ffi::Qnn_ContextHandle_t {
        self.inner.context
    }

    /// rpcmem fn pointers (Android-only). memory.rs alloc 진입 시 사용.
    #[cfg(target_os = "android")]
    pub(crate) fn rpcmem_fns(
        &self,
    ) -> (
        unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void,
        unsafe extern "C" fn(*mut std::ffi::c_void),
        unsafe extern "C" fn(*const std::ffi::c_void) -> i32,
    ) {
        (
            self.inner.rpcmem_alloc,
            self.inner.rpcmem_free,
            self.inner.rpcmem_to_fd,
        )
    }
}

#[cfg(target_os = "android")]
fn android_init() -> Result<Arc<QnnOppkgRuntime>> {
    use libloading::Symbol;
    use std::ffi::CString;
    use std::os::raw::c_uint;
    use std::ptr;

    // `libQnnGpu.so` path: env override or S25 default.
    let backend_lib_path = std::env::var("QNN_BACKEND_LIB")
        .unwrap_or_else(|_| "/data/local/tmp/qnn/libQnnGpu.so".to_string());
    // OpPackage so + provider + target — these are fixed by the production
    // OpPackage crate (`crates/qnn_oppkg`).
    let pkg_path_str = std::env::var("QNN_OPPKG_LIB")
        .unwrap_or_else(|_| "/data/local/tmp/libqnn_oppkg.so".to_string());
    const PKG_PROVIDER: &str = "QnnOpPackage_InitInterface";
    const PKG_TARGET: &str = "GPU_QTI_AISW";
    let rpc_lib_path = std::env::var("QNN_RPCMEM_LIB")
        .unwrap_or_else(|_| "/vendor/lib64/libcdsprpc.so".to_string());

    // SAFETY: dlopen is always unsafe (executes init code from .so). caller
    // accepts the risk via this entry point.
    let gpu_lib = unsafe {
        libloading::Library::new(&backend_lib_path)
            .map_err(|e| anyhow!("dlopen {} failed: {}", backend_lib_path, e))?
    };
    type GetProvidersFn =
        unsafe extern "C" fn(*mut *mut *const ffi::QnnInterface_t, *mut c_uint) -> u64;
    let provs_fn: Symbol<GetProvidersFn> = unsafe {
        gpu_lib
            .get(b"QnnInterface_getProviders\0")
            .map_err(|e| anyhow!("get QnnInterface_getProviders failed: {}", e))?
    };
    let mut provs: *mut *const ffi::QnnInterface_t = ptr::null_mut();
    let mut np: c_uint = 0;
    let err = unsafe { provs_fn(&mut provs, &mut np) };
    if err != 0 || np == 0 || provs.is_null() {
        return Err(anyhow!(
            "QnnInterface_getProviders err=0x{:x} np={}",
            err,
            np
        ));
    }
    // V2.25 vtable. The pointer lifetime is bound to gpu_lib (same .so), so we
    // re-borrow as `'static`. AndroidRuntime owns gpu_lib so the lifetime is
    // sound (Drop order: v reference is invalidated only when gpu_lib drops).
    let v: &'static ffi::QnnInterface_ImplementationV2_25_t = unsafe {
        let v_ptr =
            &(**provs).__bindgen_anon_1.v2_25 as *const ffi::QnnInterface_ImplementationV2_25_t;
        &*v_ptr
    };

    // D-D.6 디버깅: VERBOSE logger — `LLMRS_QNN_OPPKG_VERBOSE_LOG=1` 시 활성화.
    // adb logcat -v threadtime QnnGpu:V QnnGpuOpPackage:V QnnGraph:V *:S 로 캡처.
    let mut logger: ffi::Qnn_LogHandle_t = ptr::null_mut();
    if std::env::var("LLMRS_QNN_OPPKG_VERBOSE_LOG").as_deref() == Ok("1")
        && let Some(log_create) = v.logCreate
    {
        let err = unsafe {
            log_create(
                None,
                ffi::QnnLog_Level_t_QNN_LOG_LEVEL_VERBOSE,
                &mut logger,
            )
        };
        if err != 0 {
            eprintln!(
                "[qnn_oppkg] logCreate VERBOSE err=0x{:x} (proceeding without logger)",
                err
            );
            logger = ptr::null_mut();
        } else {
            eprintln!("[qnn_oppkg] logCreate VERBOSE: OK");
        }
    }

    let backend_create = v
        .backendCreate
        .ok_or_else(|| anyhow!("backendCreate fn-pointer is NULL in V2.25 vtable"))?;
    let mut backend: ffi::Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe { backend_create(logger, ptr::null_mut(), &mut backend) };
    if err != 0 {
        return Err(anyhow!("QnnBackend_create err=0x{:x}", err));
    }

    let reg_fn = v
        .backendRegisterOpPackage
        .ok_or_else(|| anyhow!("backendRegisterOpPackage fn-pointer is NULL"))?;
    let pkg_path_c =
        CString::new(pkg_path_str.as_str()).map_err(|e| anyhow!("CString PKG_PATH: {}", e))?;
    let pkg_provider_c = CString::new(PKG_PROVIDER).unwrap();
    let pkg_target_c = CString::new(PKG_TARGET).unwrap();
    let err = unsafe {
        reg_fn(
            backend,
            pkg_path_c.as_ptr(),
            pkg_provider_c.as_ptr(),
            pkg_target_c.as_ptr(),
        )
    };
    if err != 0 {
        return Err(anyhow!(
            "backendRegisterOpPackage({}) err=0x{:x}",
            pkg_path_str,
            err
        ));
    }

    let ctx_create = v
        .contextCreate
        .ok_or_else(|| anyhow!("contextCreate fn-pointer is NULL"))?;
    let mut context: ffi::Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe { ctx_create(backend, ptr::null_mut(), ptr::null_mut(), &mut context) };
    if err != 0 {
        return Err(anyhow!("QnnContext_create err=0x{:x}", err));
    }

    // rpcmem load + cache fn pointers.
    let rpc_lib = unsafe {
        libloading::Library::new(&rpc_lib_path)
            .map_err(|e| anyhow!("dlopen {} failed: {}", rpc_lib_path, e))?
    };
    type RpcmemAllocFn = unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(*mut std::ffi::c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(*const std::ffi::c_void) -> i32;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe {
        rpc_lib
            .get(b"rpcmem_alloc\0")
            .map_err(|e| anyhow!("get rpcmem_alloc failed: {}", e))?
    };
    let rpcmem_free: Symbol<RpcmemFreeFn> = unsafe {
        rpc_lib
            .get(b"rpcmem_free\0")
            .map_err(|e| anyhow!("get rpcmem_free failed: {}", e))?
    };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe {
        rpc_lib
            .get(b"rpcmem_to_fd\0")
            .map_err(|e| anyhow!("get rpcmem_to_fd failed: {}", e))?
    };
    // Capture as raw fn pointers — the Symbol borrow is dropped, but the
    // function pointer remains valid as long as `rpc_lib` stays loaded
    // (AndroidRuntime owns it).
    let rpcmem_alloc_raw: unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void =
        unsafe { *rpcmem_alloc.into_raw() };
    let rpcmem_free_raw: unsafe extern "C" fn(*mut std::ffi::c_void) =
        unsafe { *rpcmem_free.into_raw() };
    let rpcmem_to_fd_raw: unsafe extern "C" fn(*const std::ffi::c_void) -> i32 =
        unsafe { *rpcmem_to_fd.into_raw() };

    eprintln!(
        "[qnn_oppkg] runtime initialized: backend_lib={} oppkg_lib={} rpc_lib={}",
        backend_lib_path, pkg_path_str, rpc_lib_path
    );

    Ok(Arc::new(QnnOppkgRuntime {
        inner: AndroidRuntime {
            _gpu_lib: gpu_lib,
            _rpc_lib: rpc_lib,
            v,
            backend,
            context,
            rpcmem_alloc: rpcmem_alloc_raw,
            rpcmem_free: rpcmem_free_raw,
            rpcmem_to_fd: rpcmem_to_fd_raw,
        },
    }))
}

#[cfg(target_os = "android")]
impl Drop for QnnOppkgRuntime {
    fn drop(&mut self) {
        // Best-effort teardown — failures here are logged but do not panic
        // (Drop must not unwind).
        unsafe {
            if !self.inner.context.is_null()
                && let Some(ctx_free) = self.inner.v.contextFree
            {
                let err = ctx_free(self.inner.context, std::ptr::null_mut());
                if err != 0 {
                    eprintln!("[qnn_oppkg] contextFree err=0x{:x} (ignored at Drop)", err);
                }
            }
            if !self.inner.backend.is_null()
                && let Some(be_free) = self.inner.v.backendFree
            {
                let err = be_free(self.inner.backend);
                if err != 0 {
                    eprintln!("[qnn_oppkg] backendFree err=0x{:x} (ignored at Drop)", err);
                }
            }
        }
    }
}

// SAFETY: Runtime은 V2.25 fn-pointer + 핸들을 보유한다. ffi 호출은 layer_graph
// 본문에서 단일 Mutex (graph_cache) 또는 QNN context-level locking으로 직렬화된다.
// host 빌드는 placeholder만 보유.
unsafe impl Send for QnnOppkgRuntime {}
unsafe impl Sync for QnnOppkgRuntime {}

#[cfg(test)]
mod tests {
    use super::*;

    /// INV-180: host (non-Android) build에서는 init이 명확한 Err로 실패해야
    /// 한다. cargo test --features qnn,opencl --tests에서 본 unit test가
    /// 게이트.
    #[cfg(not(target_os = "android"))]
    #[test]
    fn host_init_returns_err() {
        let r = QnnOppkgRuntime::init();
        assert!(
            r.is_err(),
            "host build에서는 libQnnGpu.so 부재로 init 실패해야 한다 (INV-180)"
        );
    }
}
