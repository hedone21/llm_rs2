//! QNN runtime wrapper — `libQnnGpu.so` + `libqnn_oppkg.so` dlopen + V2.0
//! function-pointer table caching. M3.1 산출물.
//!
//! Spec: `spec/30-engine.md` 부록 C.2 (ENG-QNN-201, ENG-QNN-205,
//! ENG-QNN-209), `spec/41-invariants.md` §3.24 (INV-166, INV-180),
//! `arch/30-engine.md` §18.1.
//!
//! M3.1 단계: `init()` skeleton만 마련. 실제 V2.0 fn-pointer 캐싱과
//! `QnnBackend_create` / `QnnContext_create` / `registerOpPackage` 호출은
//! M3.2에서 M2 microbench (`microbench_qnn_qwen_layer.rs` /
//! `microbench_qnn_oppkg_flash_attn_correct.rs`)의 init flow를 발췌해
//! 채운다.
//!
//! 호스트 (non-Android, libQnnGpu 미설치)에서는 `init()`이 명확한 `Err`로
//! 실패한다 — 디바이스 빌드에서만 의미 있는 path. 본 단계의 sanity-check
//! 빌드 게이트는 `cargo build --features qnn,opencl`이 PASS면 충분하다
//! (실제 dlopen은 호출되지 않음).

use anyhow::{Result, anyhow};
use std::sync::Arc;

/// QNN runtime opaque handles. M3.2에서 `libloading::Library` + V2.0 fn-pointer
/// table + `Qnn_BackendHandle_t` / `Qnn_ContextHandle_t`로 채워진다.
///
/// 본 단계에서는 빈 placeholder이며, `Send + Sync`만 보장한다 (실제 dlopen 핸들은
/// `Send`이지만 동시 호출은 외부 `Arc<Mutex<...>>`로 직렬화하는 PoC 패턴 그대로).
pub struct QnnOppkgRuntime {
    /// Marker: M3.2에서 `Vec<libloading::Library>` 등으로 교체.
    _placeholder: (),
}

impl QnnOppkgRuntime {
    /// QNN backend / OpPackage runtime을 초기화한다.
    ///
    /// 호스트 (non-Android)에서는 `libQnnGpu.so`가 부재하므로 `Err`. 디바이스
    /// 빌드 + Android runtime에서만 정상 진행 가능하다.
    ///
    /// M3.1: 컴파일/링크만 PASS하도록 명확한 Err 반환.
    /// M3.2: M2 microbench의 init flow를 이식.
    pub fn init() -> Result<Arc<Self>> {
        // M3.2에서 dlopen + V2.0 fn-pointer 캐싱 + QnnBackend_create +
        // QnnContext_create + registerOpPackage 본격 구현.
        //
        // 본 단계에서는 호스트/디바이스 공통으로 명확한 placeholder Err 반환:
        //   - 디바이스에서도 forward는 `unimplemented!`이므로 model load만
        //     PASS되면 OK (caller가 Err를 catch하지 않고 bail).
        //   - 호스트에서는 sanity-check가 build 단계에서 끝나므로 init은 호출되지 않음.
        Err(anyhow!(
            "QnnOppkgRuntime::init() — M3.2에서 구현 (dlopen libQnnGpu.so + libqnn_oppkg.so + V2.0 fn-pointer table + registerOpPackage)"
        ))
    }
}

// SAFETY: M3.2에서 실제 dlopen handle 도입 시점에 다시 검증. 현재 placeholder는
// `()`만 보유하므로 자동 도출되지만, 명시적으로 marker를 남겨 M3.2에서 제거를
// 강제한다.
unsafe impl Send for QnnOppkgRuntime {}
unsafe impl Sync for QnnOppkgRuntime {}
