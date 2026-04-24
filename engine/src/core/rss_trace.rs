//! Lightweight RSS (Resident Set Size) tracing utility.
//!
//! Activated only when `LLMRS_RSS_TRACE=1` is set. Reads `/proc/self/status`
//! and prints a single structured line to stderr.
//!
//! Output format:
//!   `[RSS] tag=<tag> VmRSS=<kb> RssFile=<kb> RssAnon=<kb> RssShmem=<kb> VmData=<kb> VmSize=<kb>`
//!
//! On non-Linux targets this is a no-op.

use std::sync::OnceLock;

/// Returns `true` if `LLMRS_RSS_TRACE` is set in the environment.
/// The result is cached after the first call.
fn rss_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LLMRS_RSS_TRACE").is_ok())
}

/// Print a one-line RSS snapshot to stderr, tagged with `tag`.
///
/// When `LLMRS_RSS_TRACE` is not set, this function is a zero-cost no-op:
/// the `rss_trace_enabled()` check is cache-hit after the first call and the
/// function body is entirely skipped.
///
/// On non-Linux targets the function is always a no-op regardless of the
/// environment variable.
pub fn rss_trace(tag: &str) {
    if !rss_trace_enabled() {
        return;
    }
    rss_trace_impl(tag);
}

/// Linux implementation: parse `/proc/self/status` and emit one line.
#[cfg(target_os = "linux")]
fn rss_trace_impl(tag: &str) {
    match read_proc_status() {
        Ok(fields) => {
            eprintln!(
                "[RSS] tag={} VmRSS={} RssFile={} RssAnon={} RssShmem={} VmData={} VmSize={}",
                tag,
                fields.vm_rss,
                fields.rss_file,
                fields.rss_anon,
                fields.rss_shmem,
                fields.vm_data,
                fields.vm_size,
            );
        }
        Err(e) => {
            eprintln!("[RSS] tag={} error={}", tag, e);
        }
    }
}

/// Non-Linux stub: always no-op.
#[cfg(not(target_os = "linux"))]
fn rss_trace_impl(_tag: &str) {}

// ---------------------------------------------------------------------------
// /proc/self/status parser (Linux only)
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
struct ProcStatusFields {
    vm_rss: u64,
    rss_file: u64,
    rss_anon: u64,
    rss_shmem: u64,
    vm_data: u64,
    vm_size: u64,
}

#[cfg(target_os = "linux")]
fn read_proc_status() -> Result<ProcStatusFields, String> {
    let content = std::fs::read_to_string("/proc/self/status").map_err(|e| format!("read: {e}"))?;

    let mut vm_rss = 0u64;
    let mut rss_file = 0u64;
    let mut rss_anon = 0u64;
    let mut rss_shmem = 0u64;
    let mut vm_data = 0u64;
    let mut vm_size = 0u64;

    for line in content.lines() {
        // Lines look like:  "VmRSS:\t  12345 kB"
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let val: u64 = rest
            .split_whitespace()
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        match key.trim() {
            "VmRSS" => vm_rss = val,
            "RssFile" => rss_file = val,
            "RssAnon" => rss_anon = val,
            "RssShmem" => rss_shmem = val,
            "VmData" => vm_data = val,
            "VmSize" => vm_size = val,
            _ => {}
        }
    }

    Ok(ProcStatusFields {
        vm_rss,
        rss_file,
        rss_anon,
        rss_shmem,
        vm_data,
        vm_size,
    })
}
