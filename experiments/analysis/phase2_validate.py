#!/usr/bin/env python3
"""Phase 2: Prompt diversity validation — confirm the artifact hypothesis.

If the non-linear EMR pattern changes with different prompts,
it confirms that the pattern is caused by repetitive text + phase alignment,
not by fundamental eviction quality issues.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality_metrics import load_jsonl, compute_fdt, compute_emr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load(name):
    path = os.path.join(RESULTS_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        return None, None
    return load_jsonl(path)


def detect_repetition(tokens, min_period=50, max_period=500, check_len=200):
    """Detect repetition cycle in token sequence."""
    n = len(tokens)
    best = None
    best_rate = 0

    for period in range(min_period, min(max_period, n // 2)):
        matches = 0
        compared = 0
        for i in range(min(check_len, n - period)):
            if tokens[i] == tokens[i + period]:
                matches += 1
            compared += 1
        if compared > 0:
            rate = matches / compared
            if rate > best_rate:
                best = period
                best_rate = rate

    return best, best_rate


def analyze_repetition_check():
    """Check if each prompt generates repetitive text."""
    print("=" * 80)
    print("  Repetition Check (512 tokens, 3 prompts)")
    print("=" * 80)
    print()

    prompts = [
        ("V-P1-512", "AI history"),
        ("V-P2-512", "Computer networks"),
        ("V-P3-512", "Village story"),
    ]

    for name, desc in prompts:
        result = load(name)
        if result[0] is None:
            print(f"  {name} ({desc}): FILE NOT FOUND")
            continue

        recs, summary = result
        tokens = [r["token_id"] for r in recs]
        period, rate = detect_repetition(tokens)

        # Show first 200 chars of text
        text = "".join(r.get("text", "") for r in recs[:100])
        if len(text) > 300:
            text = text[:300] + "..."

        print(f"  {name} ({desc}):")
        if period and rate > 0.7:
            print(f"    Repetition: YES — period={period} tokens, match={rate:.1%}")
        else:
            print(f"    Repetition: {'WEAK' if rate > 0.3 else 'NO'} — best period={period}, match={rate:.1%}")
        print(f"    Text: {text[:200]}...")
        print()


def analyze_emr_comparison():
    """Compare EMR across prompts at same sequence lengths."""
    print("=" * 80)
    print("  EMR Comparison Across Prompts")
    print("=" * 80)
    print()

    # Original prompt (from Round 8)
    original = [
        ("P1", "2K", "B-2048", "L-2k-sl", 1024),
        ("P1", "4K", "B-4096", "L-4k-sl", 2048),
    ]

    # New prompts
    new_p2 = [
        ("P2", "2K", "V-P2-B2k", "V-P2-2k", 1024),
        ("P2", "4K", "V-P2-B4k", "V-P2-4k", 2048),
    ]

    new_p3 = [
        ("P3", "2K", "V-P3-B2k", "V-P3-2k", 1024),
        ("P3", "4K", "V-P3-B4k", "V-P3-4k", 2048),
    ]

    all_experiments = original + new_p2 + new_p3

    print(f"{'Prompt':>6} {'SeqLen':>6} │ {'EMR':>6} {'FDT':>6} {'FDT%':>6} {'FDT gap':>8} │ {'RepCycle':>8} {'RepRate':>8}")
    print("─" * 70)

    for prompt_id, seq_label, base_name, exp_name, inject_pos in all_experiments:
        base_result = load(base_name)
        exp_result = load(exp_name)

        if base_result[0] is None or exp_result[0] is None:
            print(f"{prompt_id:>6} {seq_label:>6} │ {'N/A':>6}")
            continue

        base_recs, _ = base_result
        exp_recs, _ = exp_result

        fdt = compute_fdt(base_recs, exp_recs)
        emr = compute_emr(base_recs, exp_recs)
        min_len = min(len(base_recs), len(exp_recs))
        fdt_pct = fdt / min_len * 100 if min_len > 0 else 0
        fdt_gap = fdt - inject_pos

        # Detect repetition in baseline
        base_tokens = [r["token_id"] for r in base_recs]
        period, rate = detect_repetition(base_tokens)

        rep_str = f"{period}" if period and rate > 0.5 else "none"
        rate_str = f"{rate:.0%}" if period else "—"

        print(
            f"{prompt_id:>6} {seq_label:>6} │ {emr:>6.3f} {fdt:>6} {fdt_pct:>5.1f}% {fdt_gap:>8} │ "
            f"{rep_str:>8} {rate_str:>8}"
        )

    print()
    print("  FDT gap = tokens between eviction and first divergence")
    print("  RepCycle = repetition period (tokens)")
    print("  RepRate = match rate of repetition")


def analyze_content_diversity():
    """Show text around FDT for new prompts."""
    print()
    print("=" * 80)
    print("  Content Around FDT (New Prompts)")
    print("=" * 80)
    print()

    experiments = [
        ("P2-2K", "V-P2-B2k", "V-P2-2k"),
        ("P2-4K", "V-P2-B4k", "V-P2-4k"),
        ("P3-2K", "V-P3-B2k", "V-P3-2k"),
        ("P3-4K", "V-P3-B4k", "V-P3-4k"),
    ]

    for label, base_name, exp_name in experiments:
        base_result = load(base_name)
        exp_result = load(exp_name)

        if base_result[0] is None or exp_result[0] is None:
            print(f"  {label}: FILE NOT FOUND")
            continue

        base_recs, _ = base_result
        exp_recs, _ = exp_result

        fdt = compute_fdt(base_recs, exp_recs)
        emr = compute_emr(base_recs, exp_recs)

        print(f"── {label} (FDT={fdt}, EMR={emr:.3f}) ──")

        # Show text around FDT
        start = max(0, fdt - 10)
        end = min(len(base_recs), fdt + 10)

        for i in range(start, min(end, len(exp_recs))):
            marker = ">>>" if i == fdt else "   "
            base_text = base_recs[i].get("text", "")
            exp_text = exp_recs[i].get("text", "") if i < len(exp_recs) else "N/A"
            match = "✓" if base_recs[i]["token_id"] == exp_recs[i]["token_id"] else "✗"
            print(f"  {marker} {i:>6}: {match} base={repr(base_text):>15} exp={repr(exp_text):>15}")

        # Show broader text context
        ctx_start = max(0, fdt - 30)
        ctx_end = min(len(base_recs), fdt + 10)
        text = "".join(base_recs[i].get("text", "") for i in range(ctx_start, ctx_end))
        print(f"  Context: ...{text[-200:]}...")
        print()


def main():
    print("╔" + "═" * 78 + "╗")
    print("║  Phase 2: Prompt Diversity Validation" + " " * 40 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    analyze_repetition_check()
    analyze_emr_comparison()
    analyze_content_diversity()


if __name__ == "__main__":
    main()
