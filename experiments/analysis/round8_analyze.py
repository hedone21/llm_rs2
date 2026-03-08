#!/usr/bin/env python3
"""Round 8 analysis: Long-sequence H2O vs Sliding comparison."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality_metrics import load_jsonl, compute_fdt, compute_emr, compute_rouge_l, compute_bleu4

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Experiment matrix: (seq_len, inject_pos, baseline, h2o, sliding)
EXPERIMENTS = [
    (2048, 1024, "B-2048", "L-2k-h2o", "L-2k-sl"),
    (4096, 2048, "B-4096", "L-4k-h2o", "L-4k-sl"),
    (8192, 4096, "B-8192", "L-8k-h2o", "L-8k-sl"),
    (16384, 8192, "B-16384", "L-16k-h2o", "L-16k-sl"),
    (32768, 16384, "B-32768", "L-32k-h2o", "L-32k-sl"),
]


def analyze_pair(baseline_path, experiment_path):
    """Compute quality metrics for a baseline-experiment pair."""
    if not os.path.exists(baseline_path) or not os.path.exists(experiment_path):
        return None

    base_records, base_summary = load_jsonl(baseline_path)
    exp_records, exp_summary = load_jsonl(experiment_path)

    base_text = " ".join(r.get("text", "") for r in base_records)
    exp_text = " ".join(r.get("text", "") for r in exp_records)

    fdt = compute_fdt(base_records, exp_records)
    emr = compute_emr(base_records, exp_records)
    rouge_l_result = compute_rouge_l(base_text, exp_text)
    rouge_l = rouge_l_result["f1"] if isinstance(rouge_l_result, dict) else rouge_l_result
    bleu4 = compute_bleu4(base_text, exp_text)

    # Eviction stats
    eviction_count = exp_summary.get("eviction_count", 0)
    evicted_total = exp_summary.get("evicted_tokens_total", 0)
    final_cache = exp_summary.get("final_cache_pos", 0)
    avg_tbt = exp_summary.get("avg_tbt_ms", 0)
    total_tokens = exp_summary.get("total_tokens", 0)

    return {
        "total_tokens": total_tokens,
        "fdt": fdt,
        "emr": emr,
        "rouge_l": rouge_l,
        "bleu4": bleu4,
        "eviction_count": eviction_count,
        "evicted_total": evicted_total,
        "final_cache": final_cache,
        "avg_tbt": avg_tbt,
    }


def main():
    print("=" * 80)
    print("  Round 8: Long-Sequence H2O vs Sliding — Quality Comparison")
    print("=" * 80)
    print()

    # Header
    print(f"{'SeqLen':>7} {'Inject':>7} │ {'Policy':<8} {'EMR':>6} {'FDT':>6} "
          f"{'ROUGE-L':>7} {'BLEU-4':>7} {'Evicted':>7} {'CacheEnd':>8} {'AvgTBT':>7}")
    print("─" * 87)

    for seq_len, inject_pos, base_name, h2o_name, sl_name in EXPERIMENTS:
        base_path = os.path.join(RESULTS_DIR, f"{base_name}.jsonl")

        for policy_name, exp_name in [("H2O", h2o_name), ("Sliding", sl_name)]:
            exp_path = os.path.join(RESULTS_DIR, f"{exp_name}.jsonl")
            result = analyze_pair(base_path, exp_path)

            if result is None:
                print(f"{seq_len:>7} {inject_pos:>7} │ {policy_name:<8} {'(pending)':>6}")
                continue

            print(f"{seq_len:>7} {inject_pos:>7} │ {policy_name:<8} "
                  f"{result['emr']:>6.3f} {result['fdt']:>6} "
                  f"{result['rouge_l']:>7.3f} {result['bleu4']:>7.3f} "
                  f"{result['evicted_total']:>7} {result['final_cache']:>8} "
                  f"{result['avg_tbt']:>6.1f}ms")

        print()

    # Summary comparison
    print()
    print("=" * 80)
    print("  H2O vs Sliding EMR Comparison")
    print("=" * 80)
    print()
    print(f"{'SeqLen':>7} │ {'H2O EMR':>8} {'SL EMR':>8} {'Δ(H2O-SL)':>10} {'Winner':>8}")
    print("─" * 50)

    for seq_len, inject_pos, base_name, h2o_name, sl_name in EXPERIMENTS:
        base_path = os.path.join(RESULTS_DIR, f"{base_name}.jsonl")
        h2o_path = os.path.join(RESULTS_DIR, f"{h2o_name}.jsonl")
        sl_path = os.path.join(RESULTS_DIR, f"{sl_name}.jsonl")

        h2o_result = analyze_pair(base_path, h2o_path)
        sl_result = analyze_pair(base_path, sl_path)

        if h2o_result is None or sl_result is None:
            print(f"{seq_len:>7} │ {'pending':>8}")
            continue

        delta = h2o_result["emr"] - sl_result["emr"]
        winner = "H2O" if delta > 0.005 else ("Sliding" if delta < -0.005 else "Tie")
        print(f"{seq_len:>7} │ {h2o_result['emr']:>8.3f} {sl_result['emr']:>8.3f} "
              f"{delta:>+10.3f} {winner:>8}")


if __name__ == "__main__":
    main()
