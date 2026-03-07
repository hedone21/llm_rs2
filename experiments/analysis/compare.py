#!/usr/bin/env python3
"""CLI tool comparing baseline vs experiment JSONL results.

Usage:
    python experiments/analysis/compare.py \\
      --baseline experiments/results/B-512.jsonl \\
      --experiment experiments/results/M-C-256-h2o.jsonl \\
      [--output experiments/reports/M-C-256-h2o_report.md]
"""

import argparse
import os
import sys

# Allow running as script from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quality_metrics import (
    load_jsonl,
    compute_fdt,
    compute_emr,
    compute_suffix_emr,
    compute_rouge_l,
    compute_bleu4,
    compute_topk_overlap,
)


def _extract_signal_info(summary):
    """Extract human-readable signal description from summary fields.

    Args:
        summary: Summary dict from JSONL.

    Returns:
        String describing the signal and eviction policy.
    """
    schedule = summary.get("schedule_name", "unknown")
    eviction = summary.get("eviction_policy", "none")
    return schedule, eviction


def _pct_change(base_val, exp_val):
    """Compute percentage change from base to experiment.

    Returns:
        String like '+4.4%' or '-2.1%'.
    """
    if base_val == 0:
        if exp_val == 0:
            return "+0.0%"
        return "+inf%"
    change = (exp_val - base_val) / base_val * 100
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.1f}%"


def _format_report(
    base_summary,
    exp_summary,
    base_tokens,
    exp_tokens,
    fdt,
    emr,
    suffix_emr,
    rouge_l,
    bleu4,
    topk_overlaps,
):
    """Format the comparison report as a string.

    Returns:
        Multi-line report string.
    """
    schedule_name, eviction_policy = _extract_signal_info(exp_summary)
    base_schedule, _ = _extract_signal_info(base_summary)

    # Find signal info from token records
    signal_info_parts = []
    for tok in exp_tokens:
        sig = tok.get("signal")
        if sig:
            signal_info_parts.append(f"{sig} at token {tok['pos']}")
    signal_info = "; ".join(signal_info_parts) if signal_info_parts else "none"

    # Speed metrics
    base_avg_tbt = base_summary.get("avg_tbt_ms", 0)
    exp_avg_tbt = exp_summary.get("avg_tbt_ms", 0)
    base_avg_fwd = base_summary.get("avg_forward_ms", 0)
    exp_avg_fwd = exp_summary.get("avg_forward_ms", 0)
    exp_throttle = exp_summary.get("total_throttle_ms", 0)

    base_total_tokens = base_summary.get("total_tokens", 1)
    exp_total_tokens = exp_summary.get("total_tokens", 1)
    base_throughput = 1000.0 / base_avg_tbt if base_avg_tbt > 0 else 0
    exp_throughput = 1000.0 / exp_avg_tbt if exp_avg_tbt > 0 else 0

    # Quality metrics
    min_len = min(len(base_tokens), len(exp_tokens))
    emr_matches = sum(
        1
        for i in range(min_len)
        if base_tokens[i]["token_id"] == exp_tokens[i]["token_id"]
    )
    suffix_len = min_len - fdt if fdt < min_len else 0
    suffix_matches = (
        sum(
            1
            for i in range(fdt, min_len)
            if base_tokens[i]["token_id"] == exp_tokens[i]["token_id"]
        )
        if fdt < min_len
        else 0
    )

    # Top-K breakdown
    avg_topk = sum(topk_overlaps) / len(topk_overlaps) if topk_overlaps else 0.0
    pre_fdt_topk = (
        sum(topk_overlaps[:fdt]) / fdt if fdt > 0 and fdt <= len(topk_overlaps) else 0.0
    )
    post_fdt_topk = (
        sum(topk_overlaps[fdt:]) / (len(topk_overlaps) - fdt)
        if fdt < len(topk_overlaps)
        else 0.0
    )

    # Resource metrics
    eviction_count = exp_summary.get("eviction_count", 0)
    evicted_total = exp_summary.get("evicted_tokens_total", 0)
    final_cache = exp_summary.get("final_cache_pos", 0)
    max_seq_len = exp_summary.get("max_seq_len", 2048)
    cache_util = final_cache / max_seq_len if max_seq_len > 0 else 0

    base_rss_start = base_summary.get("sys_start", {}).get("rss_mb", 0)
    base_rss_end = base_summary.get("sys_end", {}).get("rss_mb", 0)
    exp_rss_start = exp_summary.get("sys_start", {}).get("rss_mb", 0)
    exp_rss_end = exp_summary.get("sys_end", {}).get("rss_mb", 0)

    lines = []
    sep = "=" * 69
    lines.append(sep)
    lines.append(f"  {schedule_name} vs {base_schedule}")
    lines.append(f"  {signal_info}  |  {eviction_policy}")
    lines.append(sep)
    lines.append("")
    lines.append("  -- Speed -------------------------------------------------------")
    lines.append(
        f"  Avg TBT:           {base_avg_tbt:.1f}ms -> {exp_avg_tbt:.1f}ms   "
        f"({_pct_change(base_avg_tbt, exp_avg_tbt)})"
    )
    lines.append(
        f"  Avg Forward:       {base_avg_fwd:.1f}ms -> {exp_avg_fwd:.1f}ms   "
        f"({_pct_change(base_avg_fwd, exp_avg_fwd)})"
    )
    lines.append(f"  Throttle:          {exp_throttle}ms total")
    lines.append(
        f"  Throughput:        {base_throughput:.1f} -> {exp_throughput:.1f} t/s   "
        f"({_pct_change(base_throughput, exp_throughput)})"
    )
    lines.append("")
    lines.append("  -- Quality ------------------------------------------------------")
    lines.append(
        f"  First Divergent Token:   {fdt} / {min_len}"
    )
    lines.append(
        f"  Exact Match Rate:        {emr:.3f}  ({emr_matches}/{min_len})"
    )
    lines.append(
        f"  Suffix EMR (post-FDT):   {suffix_emr:.3f}  ({suffix_matches}/{suffix_len})"
    )
    lines.append(f"  ROUGE-L F1:              {rouge_l['f1']:.3f}")
    lines.append(f"  BLEU-4:                  {bleu4:.3f}")
    lines.append(f"  Top-K Overlap (avg):     {avg_topk:.3f}")
    lines.append(f"  Top-K Overlap (pre-FDT): {pre_fdt_topk:.3f}")
    lines.append(f"  Top-K Overlap (post-FDT):{post_fdt_topk:.3f}")
    lines.append("")
    lines.append("  -- Resources ----------------------------------------------------")
    evict_desc = (
        f"{eviction_count} ({evicted_total} tokens removed)"
        if eviction_count > 0
        else "0"
    )
    lines.append(f"  Evictions:          {evict_desc}")
    lines.append(
        f"  Cache Utilization:  {cache_util:.3f}  ({final_cache}/{max_seq_len})"
    )
    lines.append(
        f"  RSS Start:          {base_rss_start:.1f}MB -> {exp_rss_start:.1f}MB"
    )
    lines.append(
        f"  RSS End:            {base_rss_end:.1f}MB -> {exp_rss_end:.1f}MB"
    )
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs experiment JSONL results."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline JSONL file",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Path to experiment JSONL file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write markdown report",
    )
    args = parser.parse_args()

    # Validate files exist
    if not os.path.exists(args.baseline):
        print(f"Error: baseline file not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.experiment):
        print(f"Error: experiment file not found: {args.experiment}", file=sys.stderr)
        sys.exit(1)

    # Load data
    base_tokens, base_summary = load_jsonl(args.baseline)
    exp_tokens, exp_summary = load_jsonl(args.experiment)

    # Compute quality metrics
    fdt = compute_fdt(base_tokens, exp_tokens)
    emr = compute_emr(base_tokens, exp_tokens)
    suffix_emr = compute_suffix_emr(base_tokens, exp_tokens, fdt)

    # Build full text from token records
    base_text = "".join(t["text"] for t in base_tokens)
    exp_text = "".join(t["text"] for t in exp_tokens)

    rouge_l = compute_rouge_l(base_text, exp_text)
    bleu4 = compute_bleu4(base_text, exp_text)
    topk_overlaps = compute_topk_overlap(base_tokens, exp_tokens)

    # Format report
    report = _format_report(
        base_summary,
        exp_summary,
        base_tokens,
        exp_tokens,
        fdt,
        emr,
        suffix_emr,
        rouge_l,
        bleu4,
        topk_overlaps,
    )

    # Output
    print(report)

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            f.write("```\n")
            f.write(report)
            f.write("\n```\n")
        print(f"\nReport written to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
