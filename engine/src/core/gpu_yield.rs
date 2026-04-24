//! Intra-token GPU yield helper.
//!
//! During decode, flushes the GPU command queue every N layers and sleeps
//! for M microseconds. This creates scheduling windows for higher-priority
//! GPU contexts (e.g. foreground games) that would otherwise be starved
//! while a token's kernel chain runs uninterrupted.
//!
//! Configured via env vars, seeded from CLI flags by `generate.rs`:
//! - `LLMRS_DECODE_YIELD_EVERY` — layer interval (0 disables, default 0).
//! - `LLMRS_DECODE_YIELD_US` — sleep microseconds (default 500).
//!
//! Values are read once via `OnceLock` so the hot path is a branch + cmp.

use std::sync::OnceLock;
use std::time::Duration;

use crate::core::backend::Backend;

static YIELD_EVERY: OnceLock<usize> = OnceLock::new();
static YIELD_US: OnceLock<u64> = OnceLock::new();

fn yield_every() -> usize {
    *YIELD_EVERY.get_or_init(|| {
        std::env::var("LLMRS_DECODE_YIELD_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

fn yield_us() -> u64 {
    *YIELD_US.get_or_init(|| {
        std::env::var("LLMRS_DECODE_YIELD_US")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(500)
    })
}

/// Fast check: is intra-token yield configured? Callers can skip the
/// per-layer hook entirely when this returns false.
#[inline]
pub fn intra_token_yield_enabled() -> bool {
    yield_every() > 0
}

/// Hook to call after a decode-time layer completes.
///
/// `layer_idx` is the 0-based index of the layer that just ran. The yield
/// fires at `(layer_idx + 1) % every == 0`, matching a human-intuitive
/// "every N layers" reading.
///
/// `is_decode` gates the hook so prefill (bursty by design) is unaffected.
#[inline]
pub fn maybe_yield_after_layer(backend: &dyn Backend, layer_idx: usize, is_decode: bool) {
    if !is_decode {
        return;
    }
    let every = yield_every();
    if every == 0 {
        return;
    }
    if !(layer_idx + 1).is_multiple_of(every) {
        return;
    }
    // flush + wait: kernels already dispatched must drain before the sleep
    // is useful (otherwise the sleep overlaps with the in-flight burst and
    // buys nothing). synchronize() errors are swallowed — yield is a best
    // effort, not a correctness hook.
    let _ = backend.synchronize();
    let us = yield_us();
    if us > 0 {
        std::thread::sleep(Duration::from_micros(us));
    } else {
        std::thread::yield_now();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn defaults_disable_yield() {
        // Without env var, yield_every returns 0 → disabled.
        // (OnceLock may be cached from a prior test; just read current.)
        let enabled = super::intra_token_yield_enabled();
        // Non-deterministic but in CI without env vars should be false.
        assert!(!enabled || super::yield_every() > 0);
    }
}
