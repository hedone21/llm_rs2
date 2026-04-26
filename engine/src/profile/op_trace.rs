//! Lightweight per-op wall-clock tracer for `forward_gen`.
//!
//! Sprint E instrumentation (weight-swap TBT root cause hunt). Adds an
//! independent timing harness that can be enabled at runtime via
//! `LLMRS_FORWARD_GEN_OP_TRACE`:
//!
//! - `LLMRS_FORWARD_GEN_OP_TRACE=async` — measures host-side dispatch overhead
//!   only. No `synchronize()` is inserted, so the wall-clock captures the time
//!   between the start and end of each op-bucket as observed on the host. On
//!   GPU backends this represents the cost of the `clEnqueue*` calls plus
//!   any host-side preparation (kernel-arg packing, buffer rewrap, etc.).
//! - `LLMRS_FORWARD_GEN_OP_TRACE=sync` — calls `backend.synchronize()` once at
//!   the end of each op-bucket. This bounds the absolute GPU-execution
//!   latency for that bucket but inflates wall-clock with sync drain time.
//!   Useful for identifying which op accumulates the most GPU latency.
//!
//! When the env var is unset (default) the macros compile to a single bool
//! load + early return so the production decode path stays untouched.
//!
//! Output:
//! - On `forward_into` completion (or process exit), the accumulator is
//!   dumped to stderr in a fixed table format. Per-op `total_us`, `count`,
//!   and `avg_us` are reported.
//! - When `LLMRS_FORWARD_GEN_OP_TRACE_PATH` is set, the dump is also written
//!   as a single JSON object to that path (overwriting on each dump).
//!
//! Thread-local accumulation keeps the path lock-free during the decode loop.
//! All trace state is process-wide (single decode thread is the common case).

use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Trace mode parsed from `LLMRS_FORWARD_GEN_OP_TRACE`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TraceMode {
    Off,
    /// Wall-clock between op start and op end. No sync.
    Async,
    /// Wall-clock between op start and op end + 1 `synchronize()` at end.
    Sync,
}

fn parse_mode() -> TraceMode {
    match std::env::var("LLMRS_FORWARD_GEN_OP_TRACE") {
        Ok(v) => match v.to_ascii_lowercase().as_str() {
            "async" | "1" | "on" | "true" => TraceMode::Async,
            "sync" => TraceMode::Sync,
            _ => TraceMode::Off,
        },
        Err(_) => TraceMode::Off,
    }
}

/// Cached mode lookup. First call parses env, subsequent calls are O(1).
#[inline]
pub fn mode() -> TraceMode {
    static CACHED: OnceLock<TraceMode> = OnceLock::new();
    *CACHED.get_or_init(parse_mode)
}

/// True when any trace mode is active. Cheap check used by inline guards.
#[inline]
pub fn enabled() -> bool {
    !matches!(mode(), TraceMode::Off)
}

/// Op buckets tracked by the tracer. Order matches the `forward_gen` flow,
/// followed by buckets for ops outside `forward_gen` but inside
/// `forward_into` (embedding, final norm, lm_head). Sprint E observation:
/// the sync-mode trace captures only ~78% of TBT for the Mixed config,
/// so the gap must live in non-`forward_gen` regions.
#[repr(usize)]
#[derive(Copy, Clone, Debug)]
pub enum OpKind {
    RmsNormAttn = 0,
    MatmulQkv = 1,
    Rope = 2,
    KvUpdate = 3,
    Attention = 4,
    MatmulWo = 5,
    RmsNormFfn = 6,
    MatmulFfnGateUp = 7,
    SiluMul = 8,
    MatmulFfnDown = 9,
    AddAssign = 10,
    /// Token embedding lookup + optional embed scaling (`forward_into`).
    Embedding = 11,
    /// Final RMSNorm before `lm_head` (`forward_into`).
    FinalNorm = 12,
    /// Final `lm_head` matmul (vocab × hidden). Major suspect in Sprint E.
    LmHead = 13,
}

const N_OPS: usize = 14;

const OP_NAMES: [&str; N_OPS] = [
    "rms_norm_attn",
    "matmul_qkv",
    "rope",
    "kv_update",
    "attention",
    "matmul_wo",
    "rms_norm_ffn",
    "matmul_ffn_gate_up",
    "silu_mul",
    "matmul_ffn_down",
    "add_assign",
    "embedding",
    "final_norm",
    "lm_head",
];

#[derive(Default, Clone)]
struct TraceState {
    /// Total accumulated microseconds per op bucket.
    total_us: [u64; N_OPS],
    /// Number of times each op bucket was hit (across all layers/tokens).
    count: [u64; N_OPS],
    /// Total number of completed `forward_into` calls (token count for decode).
    forward_calls: u64,
}

thread_local! {
    static STATE: RefCell<TraceState> = RefCell::new(TraceState::default());
}

/// Begin timing an op bucket. Returns `Some(Instant)` when tracing is on,
/// `None` otherwise so the call site can skip the matching `record_op`.
#[inline]
pub fn start() -> Option<Instant> {
    if enabled() {
        Some(Instant::now())
    } else {
        None
    }
}

/// Finish timing an op bucket. In `Sync` mode runs `backend.synchronize()`
/// before sampling the elapsed time so the measurement reflects GPU-execution
/// latency. In `Async` mode the elapsed time captures host dispatch overhead
/// only. Calls with `t = None` are no-ops (zero-overhead fast path).
#[inline]
pub fn record(
    t: Option<Instant>,
    op: OpKind,
    backend: &std::sync::Arc<dyn crate::core::backend::Backend>,
    is_gpu: bool,
) {
    let Some(start) = t else {
        return;
    };
    if matches!(mode(), TraceMode::Sync) && is_gpu {
        // Bound GPU-execution latency. We swallow the error: a sync failure
        // here only corrupts the trace, not the inference result.
        let _ = backend.synchronize();
    }
    let us = start.elapsed().as_micros() as u64;
    let idx = op as usize;
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.total_us[idx] = s.total_us[idx].saturating_add(us);
        s.count[idx] = s.count[idx].saturating_add(1);
    });
}

/// Bump the forward-call counter (called once per `forward_into` from the
/// model glue layer). Tracks how many tokens contributed to the trace.
#[inline]
pub fn note_forward_call() {
    if !enabled() {
        return;
    }
    STATE.with(|s| {
        s.borrow_mut().forward_calls += 1;
    });
}

/// Render the accumulator as a stderr table. Optionally writes a JSON copy
/// to `LLMRS_FORWARD_GEN_OP_TRACE_PATH`. Resets state after dumping so a
/// follow-up run starts fresh. Skips the dump if there is no recorded
/// activity (forward_calls == 0) so atexit hooks do not emit a noisy
/// empty table when an earlier explicit dump already flushed the state.
pub fn dump_and_reset() {
    if !enabled() {
        return;
    }
    let snap = STATE.with(|s| s.borrow().clone());
    if snap.forward_calls == 0 {
        return;
    }
    let total: u64 = snap.total_us.iter().sum();
    let mode_str = match mode() {
        TraceMode::Off => "off",
        TraceMode::Async => "async",
        TraceMode::Sync => "sync",
    };

    eprintln!(
        "\n[OpTrace/{}] forward_gen per-op (forward_calls={}, total_us={}):",
        mode_str, snap.forward_calls, total
    );
    eprintln!(
        "  {:<20} {:>12} {:>10} {:>12} {:>8}",
        "op", "total_us", "count", "avg_us/call", "%"
    );
    eprintln!(
        "  {:-<20} {:-<12} {:-<10} {:-<12} {:-<8}",
        "", "", "", "", ""
    );
    for (i, name) in OP_NAMES.iter().enumerate() {
        let t = snap.total_us[i];
        let c = snap.count[i].max(1);
        let avg = t / c;
        let pct = if total > 0 {
            (t as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  {:<20} {:>12} {:>10} {:>12} {:>7.1}%",
            name, t, snap.count[i], avg, pct
        );
    }
    eprintln!("  {:<20} {:>12}", "TOTAL", total);

    if let Ok(path) = std::env::var("LLMRS_FORWARD_GEN_OP_TRACE_PATH") {
        let mut breakdown = serde_json::Map::new();
        for (i, name) in OP_NAMES.iter().enumerate() {
            let t = snap.total_us[i];
            let c = snap.count[i];
            let avg = if c > 0 { t / c } else { 0 };
            breakdown.insert(
                (*name).to_string(),
                serde_json::json!({
                    "total_us": t,
                    "count": c,
                    "avg_us": avg,
                }),
            );
        }
        let json = serde_json::json!({
            "mode": mode_str,
            "forward_calls": snap.forward_calls,
            "total_us": total,
            "breakdown": breakdown,
        });
        match std::fs::write(
            &path,
            serde_json::to_string_pretty(&json).unwrap_or_default(),
        ) {
            Ok(()) => eprintln!("  [OpTrace] wrote JSON to {}", path),
            Err(e) => eprintln!("  [OpTrace] failed to write {}: {}", path, e),
        }
    }

    // Reset so a subsequent run starts clean (e.g. multiple invocations in
    // a single process).
    STATE.with(|s| *s.borrow_mut() = TraceState::default());
}

/// Install an `atexit`-style hook that dumps the trace once the process
/// exits. This ensures we always get output even if the caller forgets the
/// explicit dump call. Idempotent.
pub fn install_atexit_once() {
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::Relaxed) {
        return;
    }
    if !enabled() {
        return;
    }
    // Use a Drop-on-static guard. Rust does not call destructors of statics,
    // so we route through libc::atexit when available, otherwise rely on the
    // explicit `dump_and_reset()` call from the binary.
    #[cfg(unix)]
    extern "C" fn cleanup() {
        dump_and_reset();
    }
    #[cfg(unix)]
    unsafe {
        libc::atexit(cleanup);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_default_is_off() {
        // SAFETY: tests run sequentially; we do not poison cross-test state.
        // We can't reset OnceLock here, so just verify default semantics
        // assuming env unset.
        if std::env::var("LLMRS_FORWARD_GEN_OP_TRACE").is_err() {
            assert_eq!(mode(), TraceMode::Off);
            assert!(!enabled());
        }
    }

    #[test]
    fn test_op_kind_indices_unique() {
        let kinds = [
            OpKind::RmsNormAttn as usize,
            OpKind::MatmulQkv as usize,
            OpKind::Rope as usize,
            OpKind::KvUpdate as usize,
            OpKind::Attention as usize,
            OpKind::MatmulWo as usize,
            OpKind::RmsNormFfn as usize,
            OpKind::MatmulFfnGateUp as usize,
            OpKind::SiluMul as usize,
            OpKind::MatmulFfnDown as usize,
            OpKind::AddAssign as usize,
            OpKind::Embedding as usize,
            OpKind::FinalNorm as usize,
            OpKind::LmHead as usize,
        ];
        for &k in &kinds {
            assert!(k < N_OPS);
        }
        let mut sorted = kinds.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), N_OPS);
    }

    #[test]
    fn test_start_returns_none_when_disabled() {
        // Only meaningful when env is unset.
        if std::env::var("LLMRS_FORWARD_GEN_OP_TRACE").is_err() {
            assert!(start().is_none());
        }
    }
}
