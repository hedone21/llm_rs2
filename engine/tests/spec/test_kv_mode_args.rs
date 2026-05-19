//! INV-LAYER spec test for KvMode subcommand (S-subcmd C4)

use clap::Parser;
use llm_rs2::session::cli::{Args, KvMode};

fn parse(argv: &[&str]) -> Args {
    let mut full = vec!["generate"];
    full.extend_from_slice(argv);
    Args::try_parse_from(full).expect("parse failed")
}

#[test]
fn default_is_standard() {
    let args = parse(&["--model-path", "/tmp/x.gguf", "--prompt", "hi"]);
    assert_eq!(args.kv_mode_args.kv_mode, KvMode::Standard);
    assert_eq!(args.effective_kv_mode(), KvMode::Standard);
}

#[test]
fn explicit_kivi_parses() {
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-mode",
        "kivi",
        "--kv-kivi-bits",
        "4",
        "--kv-kivi-residual-len",
        "64",
    ]);
    assert_eq!(args.effective_kv_mode(), KvMode::Kivi);
    assert_eq!(args.kv_mode_args.kv_kivi_bits, 4);
    assert_eq!(args.kv_mode_args.kv_kivi_residual_len, 64);
}

#[test]
fn explicit_offload_parses() {
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-mode",
        "offload",
        "--kv-offload-storage",
        "tmpfs",
        "--kv-max-prefetch-depth",
        "4",
    ]);
    assert_eq!(args.effective_kv_mode(), KvMode::Offload);
    assert_eq!(args.kv_mode_args.kv_offload_storage, "tmpfs");
    assert_eq!(args.kv_mode_args.kv_max_prefetch_depth, 4);
}

#[test]
fn legacy_kivi_flag_falls_back_to_kivi_mode() {
    // 기존 --kivi bool flag가 true이면 effective_kv_mode() = Kivi
    let args = parse(&["--model-path", "/tmp/x.gguf", "--prompt", "hi", "--kivi"]);
    assert_eq!(args.kv_mode_args.kv_mode, KvMode::Standard);
    assert_eq!(args.effective_kv_mode(), KvMode::Kivi);
}

#[test]
fn legacy_kv_offload_string_falls_back_to_offload_mode() {
    // 기존 --kv-offload <mode> 가 "none" 이 아니면 effective_kv_mode() = Offload
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-offload",
        "raw",
    ]);
    assert_eq!(args.effective_kv_mode(), KvMode::Offload);
}

#[test]
fn new_flag_wins_over_legacy() {
    // --kv-mode offload 가 명시되면 legacy --kivi bool과 무관하게 Offload
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-mode",
        "offload",
        "--kivi",
    ]);
    assert_eq!(args.effective_kv_mode(), KvMode::Offload);
}

// ── shim sub-args (C4-4b) ────────────────────────────────────────────────────

#[test]
fn effective_kivi_bits_uses_new_when_kv_mode_kivi() {
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-mode",
        "kivi",
        "--kv-kivi-bits",
        "4",
    ]);
    assert_eq!(args.effective_kivi_bits(), 4);
}

#[test]
fn effective_kivi_bits_uses_legacy_when_kivi_flag() {
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kivi",
        "--kivi-bits",
        "3",
    ]);
    assert_eq!(args.effective_kivi_bits(), 3);
}

#[test]
fn effective_kivi_residual_size_uses_new_when_kv_mode_kivi() {
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-mode",
        "kivi",
        "--kv-kivi-residual-len",
        "64",
    ]);
    assert_eq!(args.effective_kivi_residual_size(), 64);
}

#[test]
fn effective_kivi_residual_size_uses_legacy_when_kivi_flag() {
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kivi",
        "--kivi-residual-size",
        "96",
    ]);
    assert_eq!(args.effective_kivi_residual_size(), 96);
}

#[test]
fn effective_kv_offload_storage_uses_new_when_kv_mode_offload() {
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-mode",
        "offload",
        "--kv-offload-storage",
        "tmpfs",
    ]);
    assert_eq!(args.effective_kv_offload_storage(), "tmpfs");
}

#[test]
fn effective_kv_offload_storage_uses_legacy_when_kv_offload_string() {
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-offload",
        "mmap",
    ]);
    assert_eq!(args.effective_kv_offload_storage(), "mmap");
}

// ── deprecation_warnings (C5/C6 옵션 B) ──────────────────────────────────────

#[test]
fn deprecation_warnings_empty_when_no_legacy_flag() {
    let args = parse(&["--model-path", "/tmp/x.gguf", "--prompt", "hi"]);
    assert!(args.deprecation_warnings().is_empty());
}

#[test]
fn deprecation_warnings_emitted_when_kivi_legacy_flag_set() {
    let args = parse(&["--model-path", "/tmp/x.gguf", "--prompt", "hi", "--kivi"]);
    let warnings = args.deprecation_warnings();
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("--kivi is deprecated"));
    assert!(warnings[0].contains("--kv-mode kivi"));
}

#[test]
fn deprecation_warnings_emitted_when_kv_offload_legacy_used() {
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-offload",
        "mmap",
    ]);
    let warnings = args.deprecation_warnings();
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("--kv-offload is deprecated"));
    assert!(warnings[0].contains("--kv-offload-storage mmap"));
}

#[test]
fn deprecation_warnings_silent_when_new_kv_mode_offload_used() {
    // --kv-mode offload만 명시. legacy `--kv-offload`는 기본값 "none" → 경고 없음.
    let args = parse(&[
        "--model-path",
        "/tmp/x.gguf",
        "--prompt",
        "hi",
        "--kv-mode",
        "offload",
    ]);
    assert!(args.deprecation_warnings().is_empty());
}
