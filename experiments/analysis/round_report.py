#!/usr/bin/env python3
"""Generate round summary table from experiment JSONL results.

Usage:
    python experiments/analysis/round_report.py --round 2
    python experiments/analysis/round_report.py --round all
"""

import argparse
import json
import os
import sys

# Allow running as script from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quality_metrics import (
    load_jsonl,
    compute_fdt,
    compute_emr,
    compute_rouge_l,
    compute_bleu4,
)


# Round classification by filename prefix
ROUND_PREFIXES = {
    1: ["B-"],
    2: ["T-", "C-", "E-", "M-", "R-"],
    3: ["P-", "RP-"],
    4: ["H-"],
    5: ["X-"],
}

# Baseline mapping: total_tokens -> baseline filename
BASELINE_MAP = {
    32: "B-128.jsonl",    # short experiments use 128-token baseline
    127: "B-128.jsonl",
    128: "B-128.jsonl",
    511: "B-512.jsonl",
    512: "B-512.jsonl",
    1023: "B-1024.jsonl",
    1024: "B-1024.jsonl",
    2047: "B-2048.jsonl",
    2048: "B-2048.jsonl",
    1535: "B-2048.jsonl",
    1536: "B-2048.jsonl",
}


def _get_baseline_file(total_tokens):
    """Determine the appropriate baseline file for a given token count.

    Uses nearest standard baseline: 128, 512, 1024, 2048.

    Args:
        total_tokens: Number of generated tokens.

    Returns:
        Baseline filename string, or None if no suitable baseline.
    """
    if total_tokens in BASELINE_MAP:
        return BASELINE_MAP[total_tokens]

    # Find nearest standard baseline
    standards = [128, 512, 1024, 2048]
    # Map to closest standard token count
    # For experiments, the total_tokens in summary is often N-1 (e.g. 511 for 512-token run)
    for std in standards:
        if abs(total_tokens - std) <= 1 or abs(total_tokens - (std - 1)) <= 1:
            return f"B-{std}.jsonl"

    # Fallback: find nearest
    nearest = min(standards, key=lambda s: abs(total_tokens - s))
    return f"B-{nearest}.jsonl"


def _classify_round(filename):
    """Determine which round a result file belongs to.

    Checks longer prefixes first to avoid ambiguity (e.g., RP- vs R-).

    Args:
        filename: Just the filename (not full path).

    Returns:
        Integer round number, or 0 if unclassified.
    """
    basename = filename.replace(".jsonl", "")

    # Build a flat list of (prefix, round_num), sorted by prefix length descending
    # so longer prefixes match first (e.g., RP- before R-)
    all_prefixes = []
    for round_num, prefixes in ROUND_PREFIXES.items():
        for prefix in prefixes:
            all_prefixes.append((prefix, round_num))
    all_prefixes.sort(key=lambda x: len(x[0]), reverse=True)

    for prefix, round_num in all_prefixes:
        if basename.startswith(prefix):
            return round_num

    return 0


def _extract_signal_short(exp_tokens, exp_summary):
    """Extract short signal description from experiment data.

    Args:
        exp_tokens: List of experiment token records.
        exp_summary: Experiment summary dict.

    Returns:
        Short string like 'Mem.Crit@256' or 'none'.
    """
    signals = []
    for tok in exp_tokens:
        sig = tok.get("signal")
        if sig:
            pos = tok["pos"]
            # Shorten signal names
            short = sig
            short = short.replace("Memory(", "Mem.")
            short = short.replace("Thermal(", "Therm.")
            short = short.replace("Compute(", "Comp.")
            short = short.replace("Energy(", "Enrg.")
            short = short.replace(")", "")
            short = short.replace("Critical", "Crit")
            short = short.replace("Warning", "Warn")
            short = short.replace("Normal", "Norm")
            short = short.replace("Emergency", "Emrg")
            signals.append(f"{short}@{pos}")
    return ", ".join(signals) if signals else "none"


def _pct_change_str(base_val, exp_val):
    """Format percentage change as string."""
    if base_val == 0:
        return "N/A"
    change = (exp_val - base_val) / base_val * 100
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.1f}%"


def _process_experiment(exp_path, base_path, results_dir):
    """Process a single experiment against its baseline.

    Args:
        exp_path: Full path to experiment JSONL.
        base_path: Filename of baseline JSONL.
        results_dir: Directory containing result files.

    Returns:
        Dict with computed metrics, or None on error.
    """
    full_base_path = os.path.join(results_dir, base_path)

    if not os.path.exists(full_base_path):
        return None

    try:
        base_tokens, base_summary = load_jsonl(full_base_path)
        exp_tokens, exp_summary = load_jsonl(exp_path)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  Warning: error loading {exp_path}: {e}", file=sys.stderr)
        return None

    exp_id = os.path.basename(exp_path).replace(".jsonl", "")
    signal_short = _extract_signal_short(exp_tokens, exp_summary)
    eviction = exp_summary.get("eviction_policy", "none")
    total_tokens = exp_summary.get("total_tokens", 0)

    # Speed
    base_tbt = base_summary.get("avg_tbt_ms", 0)
    exp_tbt = exp_summary.get("avg_tbt_ms", 0)
    tbt_pct = _pct_change_str(base_tbt, exp_tbt)

    # Quality
    fdt = compute_fdt(base_tokens, exp_tokens)
    emr = compute_emr(base_tokens, exp_tokens)

    base_text = "".join(t["text"] for t in base_tokens)
    exp_text = "".join(t["text"] for t in exp_tokens)
    rouge_l = compute_rouge_l(base_text, exp_text)

    return {
        "id": exp_id,
        "signal": signal_short,
        "evict": eviction,
        "tokens": total_tokens,
        "tbt_pct": tbt_pct,
        "emr": emr,
        "fdt": fdt,
        "rouge_l": rouge_l["f1"],
    }


def _print_round_table(round_num, results):
    """Print formatted table for a round.

    Args:
        round_num: Round number.
        results: List of result dicts from _process_experiment.
    """
    if not results:
        print(f"\nRound {round_num}: No results found.\n")
        return

    # Header
    header = (
        f"{'ID':<20} {'Signal':<25} {'Evict':<8} {'Tokens':>6} "
        f"{'TBT+%':>8} {'EMR':>6} {'FDT':>5} {'ROUGE-L':>8}"
    )
    sep = "=" * len(header)

    print(f"\nRound {round_num} Summary")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        fdt_str = str(r["fdt"]) if r["fdt"] < r["tokens"] else str(r["tokens"])
        line = (
            f"{r['id']:<20} {r['signal']:<25} {r['evict']:<8} {r['tokens']:>6} "
            f"{r['tbt_pct']:>8} {r['emr']:>6.3f} {fdt_str:>5} {r['rouge_l']:>8.3f}"
        )
        print(line)

    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Generate round summary table from experiment results."
    )
    parser.add_argument(
        "--round",
        required=True,
        help="Round number (1-5) or 'all'",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing JSONL results (default: experiments/results/)",
    )
    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        # Try relative to cwd, then relative to script
        candidates = [
            os.path.join(os.getcwd(), "experiments", "results"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"),
        ]
        results_dir = None
        for c in candidates:
            if os.path.isdir(c):
                results_dir = os.path.abspath(c)
                break
        if results_dir is None:
            print("Error: cannot find results directory.", file=sys.stderr)
            sys.exit(1)

    if not os.path.isdir(results_dir):
        print(f"Error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine which rounds to process
    if args.round.lower() == "all":
        rounds_to_process = sorted(ROUND_PREFIXES.keys())
    else:
        try:
            round_num = int(args.round)
            if round_num not in ROUND_PREFIXES:
                print(
                    f"Error: invalid round {round_num}. Valid: 1-5 or 'all'",
                    file=sys.stderr,
                )
                sys.exit(1)
            rounds_to_process = [round_num]
        except ValueError:
            print(f"Error: invalid round '{args.round}'. Use 1-5 or 'all'", file=sys.stderr)
            sys.exit(1)

    # Discover all JSONL files
    all_files = sorted(f for f in os.listdir(results_dir) if f.endswith(".jsonl"))

    for round_num in rounds_to_process:
        # Filter files for this round
        round_files = [f for f in all_files if _classify_round(f) == round_num]

        if round_num == 1:
            # Round 1 is baselines — just print summary info, no comparison
            print(f"\nRound 1: Baselines")
            sep = "=" * 80
            print(sep)
            print(f"{'ID':<20} {'Tokens':>6} {'Avg TBT':>10} {'Avg FWD':>10} {'Evict Policy':<12}")
            print(sep)
            for f in round_files:
                try:
                    _, summary = load_jsonl(os.path.join(results_dir, f))
                    fid = f.replace(".jsonl", "")
                    tokens = summary.get("total_tokens", 0)
                    tbt = summary.get("avg_tbt_ms", 0)
                    fwd = summary.get("avg_forward_ms", 0)
                    evict = summary.get("eviction_policy", "none")
                    print(f"{fid:<20} {tokens:>6} {tbt:>9.1f}ms {fwd:>9.1f}ms {evict:<12}")
                except Exception as e:
                    print(f"{f:<20} ERROR: {e}", file=sys.stderr)
            print(sep)
            continue

        # For rounds 2+, compare against baselines
        results = []
        for f in round_files:
            exp_path = os.path.join(results_dir, f)
            try:
                _, exp_summary = load_jsonl(exp_path)
            except Exception as e:
                print(f"  Warning: skipping {f}: {e}", file=sys.stderr)
                continue

            total_tokens = exp_summary.get("total_tokens", 0)
            base_file = _get_baseline_file(total_tokens)

            if not os.path.exists(os.path.join(results_dir, base_file)):
                print(
                    f"  Warning: baseline {base_file} not found for {f} "
                    f"(tokens={total_tokens}), skipping.",
                    file=sys.stderr,
                )
                continue

            result = _process_experiment(exp_path, base_file, results_dir)
            if result:
                results.append(result)

        _print_round_table(round_num, results)


if __name__ == "__main__":
    main()
