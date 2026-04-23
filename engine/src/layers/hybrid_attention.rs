//! UMA Hybrid CPU-GPU Attention — shared setup & thread-local scope (Stage A).
//!
//! This module introduces a minimal, host-only skeleton for the KV-split
//! flash-decoding integration. The full per-layer buffer/pointer wiring
//! (partial `(m, l, o)` GPU/CPU scratch, synchronisation barriers, etc.) is
//! deferred to Stage C; Stage A only establishes:
//!   * `HybridAttnSetup` — a shared configuration carrier (Arc-shared).
//!   * A thread-local install/clear discipline, guarded by an RAII `HybridScope`
//!     so callers cannot accidentally leak a setup past the decode boundary.
//!   * Environment-variable driven bootstrapping via `HybridAttnSetup::from_env`.
//!
//! The public API is deliberately small: `install` returns a `HybridScope`
//! that restores the previous setup on drop, and `current` returns the
//! currently-installed setup (if any). No direct `clear()` is exposed because
//! the Drop guard handles cleanup; this prevents mismatched install/clear
//! pairs from leaving stale setups across token boundaries.

use std::cell::RefCell;
use std::sync::Arc;

/// Minimum meaningful KV split fraction. Below this, the GPU tail is too
/// small to amortise the dispatch + merge cost.
const HYBRID_KV_FRAC_MIN: f32 = 0.05;
/// Maximum meaningful KV split fraction. Above this, the CPU head is too
/// small to benefit from the split.
const HYBRID_KV_FRAC_MAX: f32 = 0.9;
/// Environment variable for KV-split fraction override.
pub const ENV_KV_FRAC: &str = "LLMRS_ATTN_HYBRID_KV_FRAC";

/// Shared configuration for UMA hybrid CPU-GPU attention.
///
/// Stage A carries only the KV-split fraction. Stage C will extend this
/// struct with per-layer GPU/CPU partial-state buffer pointers and related
/// dispatch metadata.
#[derive(Debug, Clone)]
pub struct HybridAttnSetup {
    /// Fraction of the KV range routed to the GPU partial. The CPU handles
    /// the complementary `[kv_split, kv_len)` range. Clamped into
    /// `[HYBRID_KV_FRAC_MIN, HYBRID_KV_FRAC_MAX]` at construction time.
    pub kv_frac: f32,
}

impl HybridAttnSetup {
    /// Construct a setup with the given KV-split fraction.
    ///
    /// Returns `None` if `kv_frac` is outside the meaningful range
    /// `[HYBRID_KV_FRAC_MIN, HYBRID_KV_FRAC_MAX]` or is not finite.
    pub fn new(kv_frac: f32) -> Option<Self> {
        if !kv_frac.is_finite() {
            return None;
        }
        if !(HYBRID_KV_FRAC_MIN..=HYBRID_KV_FRAC_MAX).contains(&kv_frac) {
            return None;
        }
        Some(Self { kv_frac })
    }

    /// Parse `LLMRS_ATTN_HYBRID_KV_FRAC` from the process environment.
    ///
    /// Returns `Some(setup)` only if the variable is set AND parses to a
    /// value in the meaningful range. Unset / invalid / out-of-range values
    /// all map to `None` (hybrid path disabled). This keeps the caller-side
    /// check trivial: `if let Some(s) = HybridAttnSetup::from_env() { ... }`.
    pub fn from_env() -> Option<Self> {
        let raw = std::env::var(ENV_KV_FRAC).ok()?;
        let val: f32 = raw.trim().parse().ok()?;
        Self::new(val)
    }
}

thread_local! {
    /// Thread-local installed setup. Written only by `install`/`HybridScope::drop`.
    static CURRENT: RefCell<Option<Arc<HybridAttnSetup>>> = const { RefCell::new(None) };
}

/// RAII guard returned by [`install`]. On drop, restores whatever setup was
/// installed before the matching `install` call (typically `None`). This
/// pattern guarantees that nested installs (should they ever occur) unwind
/// cleanly and that the setup never survives past the scope where it was
/// intended to be active.
#[must_use = "HybridScope must be held alive for the duration of the decode; \
              dropping it immediately clears the setup"]
pub struct HybridScope {
    /// Setup that was installed when `install` was called. Re-installed on
    /// drop. `None` means "there was nothing installed before".
    previous: Option<Arc<HybridAttnSetup>>,
}

impl Drop for HybridScope {
    fn drop(&mut self) {
        // 이전 설정으로 원상 복구. 현재 셀 값을 previous로 덮어쓴다.
        let prev = self.previous.take();
        CURRENT.with(|cell| {
            *cell.borrow_mut() = prev;
        });
    }
}

/// Install `setup` as the active hybrid attention configuration for the
/// current thread, returning a scope guard that restores the previous
/// configuration on drop.
pub fn install(setup: Arc<HybridAttnSetup>) -> HybridScope {
    let previous = CURRENT.with(|cell| cell.borrow_mut().replace(setup));
    HybridScope { previous }
}

/// Return the currently-installed hybrid attention setup for this thread,
/// or `None` if no setup is active.
pub fn current() -> Option<Arc<HybridAttnSetup>> {
    CURRENT.with(|cell| cell.borrow().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_valid_range() {
        assert!(HybridAttnSetup::new(0.05).is_some());
        assert!(HybridAttnSetup::new(0.5).is_some());
        assert!(HybridAttnSetup::new(0.9).is_some());
    }

    #[test]
    fn new_rejects_out_of_range() {
        assert!(HybridAttnSetup::new(0.0).is_none());
        assert!(HybridAttnSetup::new(0.04).is_none());
        assert!(HybridAttnSetup::new(0.91).is_none());
        assert!(HybridAttnSetup::new(1.0).is_none());
        assert!(HybridAttnSetup::new(-0.1).is_none());
        assert!(HybridAttnSetup::new(f32::NAN).is_none());
        assert!(HybridAttnSetup::new(f32::INFINITY).is_none());
    }

    #[test]
    fn install_and_current_roundtrip() {
        // 처음엔 아무 것도 설치되지 않아야 한다.
        assert!(current().is_none());

        let setup = Arc::new(HybridAttnSetup::new(0.5).unwrap());
        let scope = install(Arc::clone(&setup));

        let got = current().expect("setup should be installed");
        assert!((got.kv_frac - 0.5).abs() < 1e-6);
        // Arc 동일 인스턴스인지 확인.
        assert!(Arc::ptr_eq(&got, &setup));

        drop(scope);
        // 스코프 종료 후에는 원상복구되어야 한다.
        assert!(current().is_none());
    }

    #[test]
    fn nested_install_restores_previous_on_drop() {
        let outer = Arc::new(HybridAttnSetup::new(0.3).unwrap());
        let inner = Arc::new(HybridAttnSetup::new(0.7).unwrap());

        let outer_scope = install(Arc::clone(&outer));
        assert!((current().unwrap().kv_frac - 0.3).abs() < 1e-6);

        {
            let _inner_scope = install(Arc::clone(&inner));
            assert!((current().unwrap().kv_frac - 0.7).abs() < 1e-6);
        }
        // 내부 스코프가 drop되면 바깥 설정으로 복귀.
        assert!((current().unwrap().kv_frac - 0.3).abs() < 1e-6);

        drop(outer_scope);
        assert!(current().is_none());
    }

    #[test]
    fn from_env_rejects_out_of_range() {
        // 테스트 환경 오염 방지를 위해 set/unset을 한 스레드 안에서 수행.
        // Note: 다른 테스트와의 경쟁을 피하기 위해 이 테스트는 값 검사 후 변수를 즉시 해제한다.
        // SAFETY: set_var / remove_var는 멀티스레드 환경에서 unsafe (Rust 1.74+).
        //         단일 테스트 스레드 내에서만 접근하도록 직렬화는 테스트 런너의 책임.
        unsafe {
            std::env::set_var(ENV_KV_FRAC, "1.5");
        }
        assert!(HybridAttnSetup::from_env().is_none());

        unsafe {
            std::env::set_var(ENV_KV_FRAC, "not-a-number");
        }
        assert!(HybridAttnSetup::from_env().is_none());

        unsafe {
            std::env::set_var(ENV_KV_FRAC, "0.5");
        }
        let s = HybridAttnSetup::from_env().expect("0.5 is in range");
        assert!((s.kv_frac - 0.5).abs() < 1e-6);

        unsafe {
            std::env::remove_var(ENV_KV_FRAC);
        }
        assert!(HybridAttnSetup::from_env().is_none());
    }
}
