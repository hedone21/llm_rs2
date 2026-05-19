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
