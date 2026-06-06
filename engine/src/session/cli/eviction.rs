//! Eviction policy CLI subcommand (S-subcmd C1, 2026-05-19).
//!
//! Replaces the polic-specific flag flat namespace (`--eviction-policy h2o
//! --h2o-keep-ratio 0.5 --h2o-decay 0.0 ...`) with a clap subcommand
//! enum. Each variant exposes only its own parameters, so policy
//! additions (SnapKV, KVSwap, ...) no longer balloon the global CLI
//! surface.
//!
//! Variant common parameters (kv_budget / protected_prefix /
//! memory_threshold_mb / eviction_target_ratio / initial_kv_capacity /
//! min_kv_cache / kv_budget_ratio) live in [`EvictionCommonArgs`] and
//! are `#[clap(flatten)]`'d at the binary's top-level `Args`.
//!
//! Wired in by C2 (cli/mod.rs Args integration).

use clap::{Args, Subcommand};

/// Top-level wrapper exposing `eviction <policy>` as a single clap
/// subcommand group. clap derive registers each [`EvictionCmd`] variant
/// directly as a subcommand, so without this wrapper the CLI would be
/// `generate ... h2o --keep-ratio 0.5`. The wrapper produces the
/// `generate ... eviction h2o --keep-ratio 0.5` form documented in
/// `docs/USAGE.md` and `docs/35_experiment_runner_guide.md`.
#[derive(Subcommand, Debug, Clone)]
pub enum TopLevelCmd {
    /// KV cache eviction policy (variant chosen via nested subcommand).
    Eviction {
        #[command(subcommand)]
        policy: EvictionCmd,
    },
}

/// Eviction policy selection (nested under [`TopLevelCmd::Eviction`]).
///
/// CLI usage:
/// ```text
/// generate -m model.gguf eviction sliding --window 1024
/// generate -m model.gguf eviction h2o --keep-ratio 0.5 --tracked-layers 0
/// generate -m model.gguf eviction d2o --keep-ratio 0.75 --ema-beta 0.7
/// ```
///
/// Omitting the subcommand is equivalent to [`EvictionCmd::None`] —
/// no eviction; the KV cache grows up to `--max-seq-len`.
#[derive(Subcommand, Debug, Clone)]
pub enum EvictionCmd {
    /// No eviction (default). KV cache grows up to --max-seq-len.
    None,

    /// Sliding window — retain the most recent N tokens.
    Sliding(SlidingArgs),

    /// StreamingLLM — keep `sink` initial tokens plus a recent window.
    Streaming(StreamingArgs),

    /// H2O — heavy hitter selection (Round 14 default for Llama 3.2 1B).
    H2o(H2oArgs),

    /// H2O+ — per-head GQA-aware H2O variant.
    H2oPlus(H2oArgs),

    /// D2O — Dynamic Discriminative Operations (arXiv 2406.13035).
    D2o(D2oArgs),

    /// CAOTE — value-aware criticality `a_i·‖v_i − o_h‖` (ADR-0004 §8).
    ///
    /// feature `caote` 플러그인 install 시에만 노출된다(미설치 = subcommand 부재).
    /// 튜닝 파라미터 없음 — V 는 ctx.tensor(Value)로, 가중치 a_i 는 importance 로 자동.
    #[cfg(feature = "caote")]
    Caote,
}

impl EvictionCmd {
    /// Canonical policy name matching the legacy `--eviction-policy` string
    /// (compatibility with existing manager IPC, lua policy DSL, and JSON
    /// dumps). Used during the migration window so downstream code can keep
    /// matching on the policy name.
    pub fn policy_name(&self) -> &'static str {
        match self {
            EvictionCmd::None => "none",
            EvictionCmd::Sliding(_) => "sliding",
            EvictionCmd::Streaming(_) => "streaming",
            EvictionCmd::H2o(_) => "h2o",
            EvictionCmd::H2oPlus(_) => "h2o_plus",
            EvictionCmd::D2o(_) => "d2o",
            #[cfg(feature = "caote")]
            EvictionCmd::Caote => "caote",
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct SlidingArgs {
    /// Tokens to retain in the sliding window.
    #[arg(long, default_value_t = 1024)]
    pub window: usize,
}

#[derive(Args, Debug, Clone)]
pub struct StreamingArgs {
    /// Attention sink tokens to preserve at the start.
    #[arg(long, default_value_t = 4)]
    pub sink: usize,
    /// Recent window size. 0 = auto (`kv_budget - sink`).
    #[arg(long, default_value_t = 0)]
    pub recent_window: usize,
}

#[derive(Args, Debug, Clone)]
pub struct H2oArgs {
    /// Fraction of tokens kept as heavy hitters (0.0–1.0).
    /// 0.0 → equivalent to Sliding; Round 14 confirmed this as the
    /// optimal H2O regime for Llama 3.2 1B.
    #[arg(long, default_value_t = 0.5)]
    pub keep_ratio: f32,

    /// Number of final transformer layers to track for importance scores.
    /// 0 = all layers.
    #[arg(long, default_value_t = 0)]
    pub tracked_layers: usize,

    /// EMA decay factor for cumulative importance scores per step
    /// (0.0 = no decay).
    #[arg(long, default_value_t = 0.0)]
    pub decay: f32,

    /// Disable time-normalized scoring (use raw cumulative SUM).
    /// By default H2O / H2O+ time-normalize to remove cumulative bias.
    #[arg(long, default_value_t = false)]
    pub raw_scores: bool,
}

#[derive(Args, Debug, Clone)]
pub struct D2oArgs {
    /// Heavy-hitter keep ratio (paper default 0.75 = 3:1).
    #[arg(long, default_value_t = 0.75)]
    pub keep_ratio: f32,

    /// EMA smoothing factor β for threshold update
    /// (paper Eq.10, default 0.7).
    #[arg(long, default_value_t = 0.7)]
    pub ema_beta: f32,

    /// Eq.11 normalisation constant `e` controlling retained token's
    /// self-weight (paper default 0.1).
    #[arg(long, default_value_t = 0.1)]
    pub merge_e: f32,

    /// Enable D2O layer-level dynamic allocation
    /// (per-layer attention variance from prefill).
    #[arg(long, default_value_t = false)]
    pub layer_alloc: bool,

    /// Protected layer indices for D2O layer allocation (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub protected_layers: Option<Vec<usize>>,
}

/// Variant-independent eviction parameters.
///
/// Flattened into the binary's top-level `Args` because every policy
/// (and the manager IPC path) reads these regardless of which variant
/// is active.
#[derive(Args, Debug, Clone)]
pub struct EvictionCommonArgs {
    /// Maximum KV cache budget in tokens. Evicts when cache_pos exceeds
    /// this. 0 = no budget limit (default).
    #[arg(long, default_value_t = 0)]
    pub kv_budget: usize,

    /// KV cache budget as ratio of prompt length (0.0–1.0).
    /// When > 0, overrides --kv-budget per question. Matches H2O paper
    /// evaluation methodology.
    #[arg(long, default_value_t = 0.0)]
    pub kv_budget_ratio: f32,

    /// Number of prefix tokens protected from eviction.
    /// Defaults to 4 for score-based policies (h2o, h2o_plus, d2o)
    /// and prompt length for sliding.
    #[arg(long)]
    pub protected_prefix: Option<usize>,

    /// Memory threshold in MB below which eviction triggers.
    #[arg(long, default_value_t = 256)]
    pub memory_threshold_mb: usize,

    /// Target ratio of cache to keep when evicting (0.1–0.99).
    #[arg(long, default_value_t = 0.75)]
    pub eviction_target_ratio: f32,

    /// Initial KV cache capacity in tokens.
    /// 0 = auto (prompt length rounded up to power of 2, min 128).
    #[arg(long, default_value_t = 0)]
    pub initial_kv_capacity: usize,

    /// Minimum KV cache size in tokens. Eviction will not reduce cache
    /// below this.
    #[arg(long, default_value_t = 256)]
    pub min_kv_cache: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Wrapper struct so we can invoke clap's parser on `EvictionCmd`.
    #[derive(Parser, Debug)]
    struct Wrap {
        #[command(subcommand)]
        ev: Option<EvictionCmd>,
    }

    fn parse(args: &[&str]) -> Wrap {
        let mut full = vec!["test"];
        full.extend_from_slice(args);
        Wrap::try_parse_from(full).expect("parse")
    }

    #[test]
    fn parses_no_subcommand_as_none() {
        let w = parse(&[]);
        assert!(w.ev.is_none(), "absence of subcommand ≡ EvictionCmd::None");
    }

    #[test]
    fn parses_explicit_none() {
        let w = parse(&["none"]);
        assert!(matches!(w.ev, Some(EvictionCmd::None)));
        assert_eq!(w.ev.as_ref().unwrap().policy_name(), "none");
    }

    #[test]
    fn parses_sliding_with_window() {
        let w = parse(&["sliding", "--window", "2048"]);
        match w.ev {
            Some(EvictionCmd::Sliding(s)) => assert_eq!(s.window, 2048),
            _ => panic!("expected Sliding"),
        }
    }

    #[test]
    fn parses_streaming_defaults() {
        let w = parse(&["streaming"]);
        match w.ev {
            Some(EvictionCmd::Streaming(s)) => {
                assert_eq!(s.sink, 4);
                assert_eq!(s.recent_window, 0);
            }
            _ => panic!("expected Streaming"),
        }
    }

    #[test]
    fn parses_h2o_full() {
        let w = parse(&[
            "h2o",
            "--keep-ratio",
            "0.3",
            "--tracked-layers",
            "8",
            "--decay",
            "0.1",
            "--raw-scores",
        ]);
        match w.ev {
            Some(EvictionCmd::H2o(h)) => {
                assert!((h.keep_ratio - 0.3).abs() < 1e-6);
                assert_eq!(h.tracked_layers, 8);
                assert!((h.decay - 0.1).abs() < 1e-6);
                assert!(h.raw_scores);
            }
            _ => panic!("expected H2o"),
        }
    }

    #[test]
    fn parses_h2o_plus_uses_same_args() {
        let w = parse(&["h2o-plus", "--keep-ratio", "0.4"]);
        assert_eq!(w.ev.as_ref().unwrap().policy_name(), "h2o_plus");
        match w.ev {
            Some(EvictionCmd::H2oPlus(h)) => {
                assert!((h.keep_ratio - 0.4).abs() < 1e-6);
            }
            _ => panic!("expected H2oPlus"),
        }
    }

    #[test]
    fn parses_d2o_with_protected_layers() {
        let w = parse(&[
            "d2o",
            "--keep-ratio",
            "0.8",
            "--ema-beta",
            "0.6",
            "--merge-e",
            "0.15",
            "--layer-alloc",
            "--protected-layers",
            "0,1,2",
        ]);
        match w.ev {
            Some(EvictionCmd::D2o(d)) => {
                assert!((d.keep_ratio - 0.8).abs() < 1e-6);
                assert!((d.ema_beta - 0.6).abs() < 1e-6);
                assert!((d.merge_e - 0.15).abs() < 1e-6);
                assert!(d.layer_alloc);
                assert_eq!(d.protected_layers, Some(vec![0, 1, 2]));
            }
            _ => panic!("expected D2o"),
        }
    }

    #[test]
    fn rejects_unknown_eviction_arg() {
        // H2o has no --window flag (Sliding does) — clap must reject.
        let r = Wrap::try_parse_from(["test", "h2o", "--window", "256"]);
        assert!(r.is_err(), "h2o subcommand must reject Sliding's --window");
    }

    /// ADR-0004 §8: feature `caote` install 시 `eviction caote` 가 parse 되고 policy_name 이
    /// "caote" — session/build_bench 의 `find_stage(name)` seam 으로 흘러 플러그인을 선택한다.
    #[cfg(feature = "caote")]
    #[test]
    fn parses_caote_unit_subcommand() {
        let w = parse(&["caote"]);
        assert!(matches!(w.ev, Some(EvictionCmd::Caote)));
        assert_eq!(w.ev.as_ref().unwrap().policy_name(), "caote");
    }

    /// feature OFF(plugin 미설치)에서는 caote subcommand 가 존재하지 않아 clap 이 거부한다.
    #[cfg(not(feature = "caote"))]
    #[test]
    fn rejects_caote_when_plugin_absent() {
        let r = Wrap::try_parse_from(["test", "caote"]);
        assert!(
            r.is_err(),
            "feature OFF 시 caote subcommand 미존재 → clap reject"
        );
    }

    #[test]
    fn common_args_parse_independently() {
        // Separate parser exercises EvictionCommonArgs alone.
        #[derive(Parser, Debug)]
        struct C {
            #[clap(flatten)]
            common: EvictionCommonArgs,
        }
        let c = C::try_parse_from([
            "test",
            "--kv-budget",
            "1024",
            "--protected-prefix",
            "4",
            "--memory-threshold-mb",
            "512",
        ])
        .unwrap();
        assert_eq!(c.common.kv_budget, 1024);
        assert_eq!(c.common.protected_prefix, Some(4));
        assert_eq!(c.common.memory_threshold_mb, 512);
        // Other defaults preserved.
        assert_eq!(c.common.min_kv_cache, 256);
        assert!((c.common.eviction_target_ratio - 0.75).abs() < 1e-6);
    }
}
