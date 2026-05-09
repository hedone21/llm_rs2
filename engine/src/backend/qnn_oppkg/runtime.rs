//! QNN runtime wrapper — `libQnnGpu.so` + `libqnn_oppkg.so` dlopen + V2.0
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
//! ## 본 단계 (M3.2) 적용 범위
//! 본 단계의 pass-gate는 호스트 빌드 + dispatch path 컴파일 검증이므로, 실제
//! dlopen + ffi 호출 본체는 `#[cfg(target_os = "android")]`로 격리한다. host
//! 빌드에서는 `init()`이 즉시 `Err`로 실패하여 caller가 bail; sanity-check
//! `cargo test --features qnn,opencl --tests`는 INV-180 unit test로 본 path를
//! 검증한다.

use anyhow::{Result, anyhow};
use std::sync::Arc;

/// QNN runtime opaque handles.
///
/// Android device path에서 dlopen된 libraries + V2.25 fn-pointer table +
/// `Qnn_BackendHandle_t` / `Qnn_ContextHandle_t`를 보유한다. host 빌드에서는
/// 본 struct가 instantiate되지 않는다 (`init()` 즉시 Err).
pub struct QnnOppkgRuntime {
    /// Marker placeholder. M3.3 device forward 진입 시 실제 handle 필드
    /// (`libloading::Library`, ffi V2.25 vtable, backend/context handle)로
    /// 교체된다. 본 단계는 host 빌드 + dispatch 컴파일 검증 범위 한정.
    _placeholder: (),
}

impl QnnOppkgRuntime {
    /// QNN backend / OpPackage runtime을 초기화한다.
    ///
    /// - Android device: dlopen + V2.0 fn-pointer 캐싱 + `QnnBackend_create` +
    ///   `QnnContext_create` + `registerOpPackage` (M3.3 진입 시 본격).
    /// - host (non-Android): `libQnnGpu.so` 부재로 즉시 `Err` (INV-180).
    pub fn init() -> Result<Arc<Self>> {
        #[cfg(target_os = "android")]
        {
            // M3.3 device forward 본문 진입 시 활성화: M2.H microbench의 init
            // flow (libloading::Library::new + V2.25 vtable + backendCreate +
            // registerOpPackage + contextCreate)를 본 path에 이식한다. 본 단계
            // (M3.2)에서는 device 측에서도 init이 호출되지 않는다 (forward
            // unimplemented marker로 차단).
            Err(anyhow!(
                "QnnOppkgRuntime::init() (android) — M3.3에서 dlopen + V2.25 fn-pointer 본격 도입 (본 단계는 컴파일 게이트만)"
            ))
        }
        #[cfg(not(target_os = "android"))]
        {
            // INV-180: 호스트 빌드는 명확한 Err로 fail하여 caller가 bail.
            Err(anyhow!(
                "QnnOppkgRuntime::init() — host (non-Android) build에서는 libQnnGpu.so 부재로 init 불가. 디바이스 빌드 + Android runtime에서만 진행 가능."
            ))
        }
    }
}

// SAFETY: Runtime handle은 Arc 외부에서 동시 접근될 수 있으나, 실제 ffi 호출은
// M3.3에서 `Mutex`로 직렬화되거나 QNN context-level locking에 의존한다. 현재
// placeholder는 `()`만 보유하여 자동 도출되지만, M3.3 진입 시점에 다시 검증해야
// 한다.
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
