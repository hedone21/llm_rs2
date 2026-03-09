#!/usr/bin/env python3
"""Round 13 Analysis: H2O+ (Per-Head Eviction) Accuracy Comparison.

Compares 4 eviction policies: none (baseline), sliding, h2o, h2o_plus.

Phases:
  1) 4-Way Direct Comparison — EMR/FDT/ROUGE-L table
  2) H2O+ Keep Ratio Sweep — EMR vs keep_ratio curve
  3) H2O vs H2O+ Head-to-Head — matched keep_ratio comparison

Usage:
    python experiments/analysis/round13_analyze.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from quality_metrics import (
    load_jsonl,
    compute_emr,
    compute_fdt,
    compute_suffix_emr,
    compute_rouge_l,
    compute_bleu4,
    compute_topk_overlap,
)

R13_DIR = Path(__file__).parent.parent / "results" / "round13"
R12_DIR = Path(__file__).parent.parent / "results" / "round12"
R11_DIR = Path(__file__).parent.parent / "results" / "round11"
REPORT_DIR = Path(__file__).parent.parent / "reports"

PROMPTS = ["PPL01", "PPL03"]
DOMAINS = {"PPL01": "Literary", "PPL03": "Technical"}


def analyze_ppl_pair(base_path, exp_path):
    """Analyze a PPL baseline vs experiment pair."""
    base_tokens, base_summary = load_jsonl(base_path)
    exp_tokens, exp_summary = load_jsonl(exp_path)

    base_text = "".join(t.get("text", "") for t in base_tokens)
    exp_text = "".join(t.get("text", "") for t in exp_tokens)

    fdt = compute_fdt(base_tokens, exp_tokens)
    emr = compute_emr(base_tokens, exp_tokens)
    suffix_emr = compute_suffix_emr(base_tokens, exp_tokens, fdt)
    rouge = compute_rouge_l(base_text, exp_text)
    bleu = compute_bleu4(base_text, exp_text)
    topk = compute_topk_overlap(base_tokens, exp_tokens)
    avg_topk = sum(topk) / len(topk) if topk else 0.0

    evicted = exp_summary.get("evicted_tokens_total", 0)

    return {
        "emr": emr,
        "fdt": fdt,
        "suffix_emr": suffix_emr,
        "rouge_l": rouge["f1"],
        "bleu4": bleu,
        "topk": avg_topk,
        "evicted": evicted,
        "n_tokens": len(base_tokens),
    }


# ============================================================
#  Phase 1: 4-Way Direct Comparison
# ============================================================
def analyze_phase1():
    """Compare none, sliding, h2o, h2o_plus at keep_ratio=0.5."""
    print("\n" + "=" * 72)
    print("  Phase 1: 4-Way Direct Comparison")
    print("=" * 72)

    # Use fresh baselines from Round 13 (same binary version)
    policies = {
        "Baseline": {"type": "self", "desc": "No eviction"},
        "Sliding": {"dir": R13_DIR, "pattern": "SL-{ppl}.jsonl", "desc": "Recent-only"},
        "H2O": {"dir": R13_DIR, "pattern": "H2O-50-{ppl}.jsonl", "desc": "kr=0.5 (shared HH)"},
        "H2O+": {"dir": R13_DIR, "pattern": "H2OP-50-{ppl}.jsonl", "desc": "kr=0.5 (per-head HH)"},
    }

    all_results = {}  # {(policy_name, ppl_id): metrics}

    for ppl_id in PROMPTS:
        # Use Round 13 fresh baseline (same binary)
        base_file = R13_DIR / f"BASE-{ppl_id}.jsonl"
        if not base_file.exists():
            print(f"  SKIP {ppl_id} (no baseline in round13)")
            continue

        for policy_name, cfg in policies.items():
            if cfg.get("type") == "self":
                # Baseline vs itself
                all_results[(policy_name, ppl_id)] = {
                    "emr": 1.0, "fdt": 2048, "suffix_emr": 1.0,
                    "rouge_l": 1.0, "bleu4": 1.0, "topk": 1.0,
                    "evicted": 0, "n_tokens": 2048,
                }
                continue

            exp_file = cfg["dir"] / cfg["pattern"].format(ppl=ppl_id)
            if not exp_file.exists():
                print(f"  WARN: Missing {exp_file}")
                continue

            metrics = analyze_ppl_pair(base_file, exp_file)
            all_results[(policy_name, ppl_id)] = metrics

    # Print table
    print(f"\n{'Policy':<12} {'Domain':<12} {'EMR':>6} {'FDT':>6} {'ROUGE-L':>8} {'BLEU-4':>8} {'TopK':>6} {'Evicted':>8}")
    print("-" * 72)

    for policy_name in policies:
        for ppl_id in PROMPTS:
            key = (policy_name, ppl_id)
            if key not in all_results:
                continue
            m = all_results[key]
            print(f"{policy_name:<12} {DOMAINS[ppl_id]:<12} {m['emr']:6.3f} {m['fdt']:6d} {m['rouge_l']:8.3f} {m['bleu4']:8.3f} {m['topk']:6.3f} {m['evicted']:8d}")

    # Summary comparison
    print("\n── H2O+ vs H2O Improvement ──")
    for ppl_id in PROMPTS:
        h2o_key = ("H2O", ppl_id)
        h2op_key = ("H2O+", ppl_id)
        if h2o_key in all_results and h2op_key in all_results:
            h2o_emr = all_results[h2o_key]["emr"]
            h2op_emr = all_results[h2op_key]["emr"]
            delta = h2op_emr - h2o_emr
            print(f"  {DOMAINS[ppl_id]}: H2O EMR={h2o_emr:.3f} → H2O+ EMR={h2op_emr:.3f} (Δ={delta:+.3f})")

    return all_results


# ============================================================
#  Phase 2: H2O+ Keep Ratio Sweep
# ============================================================
def analyze_phase2():
    """Analyze H2O+ with varying keep_ratio."""
    print("\n" + "=" * 72)
    print("  Phase 2: H2O+ Keep Ratio Sweep")
    print("=" * 72)

    keep_ratios = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    kr_labels = ["00", "10", "20", "30", "50", "70", "90"]

    results = {}  # {(ppl_id, kr_label): metrics}

    # Sliding reference from Round 13 (same binary)
    sliding_ref = {}
    for ppl_id in PROMPTS:
        base_file = R13_DIR / f"BASE-{ppl_id}.jsonl"
        sl_file = R13_DIR / f"SL-{ppl_id}.jsonl"
        if base_file.exists() and sl_file.exists():
            sliding_ref[ppl_id] = analyze_ppl_pair(base_file, sl_file)

    for ppl_id in PROMPTS:
        base_file = R13_DIR / f"BASE-{ppl_id}.jsonl"
        if not base_file.exists():
            continue

        for kr, label in zip(keep_ratios, kr_labels):
            exp_file = R13_DIR / f"H2OP-{label}-{ppl_id}.jsonl"
            if not exp_file.exists():
                print(f"  WARN: Missing {exp_file}")
                continue
            metrics = analyze_ppl_pair(base_file, exp_file)
            results[(ppl_id, label)] = metrics

    # Print table
    print(f"\n{'KR':<6} {'HH':>4} {'Recent':>7} ", end="")
    for ppl_id in PROMPTS:
        print(f"{'EMR(' + DOMAINS[ppl_id][:3] + ')':>10} ", end="")
    print()
    print("-" * 50)

    for kr, label in zip(keep_ratios, kr_labels):
        available = 172  # approximate: 212 - 40 prefix
        hh = int(available * kr)
        recent = available - hh
        print(f"{kr:<6.1f} {hh:>4} {recent:>7} ", end="")
        for ppl_id in PROMPTS:
            key = (ppl_id, label)
            if key in results:
                print(f"{results[key]['emr']:>10.3f} ", end="")
            else:
                print(f"{'N/A':>10} ", end="")
        print()

    # Sliding reference line
    print(f"{'SL':<6} {'0':>4} {'172':>7} ", end="")
    for ppl_id in PROMPTS:
        if ppl_id in sliding_ref:
            print(f"{sliding_ref[ppl_id]['emr']:>10.3f} ", end="")
        else:
            print(f"{'N/A':>10} ", end="")
    print("  ← Sliding reference")

    return results, sliding_ref


# ============================================================
#  Phase 3: H2O vs H2O+ Head-to-Head
# ============================================================
def analyze_phase3():
    """Compare H2O and H2O+ at matched keep_ratios."""
    print("\n" + "=" * 72)
    print("  Phase 3: H2O vs H2O+ Head-to-Head (matched keep_ratio)")
    print("=" * 72)

    keep_ratios = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    kr_labels = ["00", "10", "20", "30", "50", "70", "90"]

    for ppl_id in PROMPTS:
        base_file = R13_DIR / f"BASE-{ppl_id}.jsonl"
        if not base_file.exists():
            continue

        print(f"\n── {DOMAINS[ppl_id]} ({ppl_id}) ──")
        print(f"{'KR':<6} {'H2O EMR':>9} {'H2O+ EMR':>10} {'Δ':>8} {'Winner':>8}")
        print("-" * 50)

        for kr, label in zip(keep_ratios, kr_labels):
            # H2O from Round 13 (same binary) — available for 00, 10, 30, 50
            h2o_file = R13_DIR / f"H2O-{label}-{ppl_id}.jsonl"

            h2op_file = R13_DIR / f"H2OP-{label}-{ppl_id}.jsonl"

            h2o_emr = None
            h2op_emr = None

            if h2o_file.exists():
                h2o_metrics = analyze_ppl_pair(base_file, h2o_file)
                h2o_emr = h2o_metrics["emr"]

            if h2op_file.exists():
                h2op_metrics = analyze_ppl_pair(base_file, h2op_file)
                h2op_emr = h2op_metrics["emr"]

            if h2o_emr is not None and h2op_emr is not None:
                delta = h2op_emr - h2o_emr
                winner = "H2O+" if delta > 0.01 else ("H2O" if delta < -0.01 else "TIE")
                print(f"{kr:<6.1f} {h2o_emr:>9.3f} {h2op_emr:>10.3f} {delta:>+8.3f} {winner:>8}")
            else:
                h2o_str = f"{h2o_emr:.3f}" if h2o_emr is not None else "N/A"
                h2op_str = f"{h2op_emr:.3f}" if h2op_emr is not None else "N/A"
                print(f"{kr:<6.1f} {h2o_str:>9} {h2op_str:>10} {'N/A':>8} {'N/A':>8}")


# ============================================================
#  Verdict
# ============================================================
def verdict(phase1_results, phase2_results, sliding_ref):
    """Determine which scenario (A/B/C) matches the data."""
    print("\n" + "=" * 72)
    print("  Verdict")
    print("=" * 72)

    # Check H2O+ kr=0.5 vs H2O kr=0.5
    h2o_emrs = []
    h2op_emrs = []
    sl_emrs = []

    for ppl_id in PROMPTS:
        h2o_key = ("H2O", ppl_id)
        h2op_key = ("H2O+", ppl_id)
        if h2o_key in phase1_results:
            h2o_emrs.append(phase1_results[h2o_key]["emr"])
        if h2op_key in phase1_results:
            h2op_emrs.append(phase1_results[h2op_key]["emr"])
        if ppl_id in sliding_ref:
            sl_emrs.append(sliding_ref[ppl_id]["emr"])

    if h2o_emrs and h2op_emrs and sl_emrs:
        avg_h2o = sum(h2o_emrs) / len(h2o_emrs)
        avg_h2op = sum(h2op_emrs) / len(h2op_emrs)
        avg_sl = sum(sl_emrs) / len(sl_emrs)
        delta_h2op_h2o = avg_h2op - avg_h2o
        delta_h2op_sl = avg_h2op - avg_sl

        print(f"\n  Average EMR (kr=0.5):")
        print(f"    Sliding:  {avg_sl:.3f}")
        print(f"    H2O:      {avg_h2o:.3f}")
        print(f"    H2O+:     {avg_h2op:.3f}")
        print(f"    H2O+ - H2O:     {delta_h2op_h2o:+.3f}")
        print(f"    H2O+ - Sliding:  {delta_h2op_sl:+.3f}")

        if delta_h2op_h2o >= 0.10 and abs(delta_h2op_sl) < 0.05:
            print("\n  ★ Scenario A: Per-head HH restores value!")
            print("    Per-head selection makes HH meaningful.")
        elif delta_h2op_h2o < 0.05:
            print("\n  ★ Scenario B: Per-head still fails")
            print("    HH concept itself is flawed for autoregressive generation.")
        else:
            print("\n  ★ Scenario C: Partial improvement")
            print("    Per-head helps, but not enough to match Sliding.")

    # Check best keep_ratio for H2O+
    best_kr = None
    best_emr = -1.0
    for ppl_id in PROMPTS:
        for label in ["00", "10", "20", "30", "50", "70", "90"]:
            key = (ppl_id, label)
            if key in phase2_results:
                emr = phase2_results[key]["emr"]
                if emr > best_emr:
                    best_emr = emr
                    best_kr = label

    if best_kr is not None:
        print(f"\n  Best H2O+ keep_ratio: kr={int(best_kr)/100:.1f} (EMR={best_emr:.3f})")


# ============================================================
#  Report Generation
# ============================================================
def generate_report(phase1_results, phase2_results, sliding_ref):
    """Generate markdown report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "round13_report.md"

    lines = ["# Round 13: H2O+ Per-Head Eviction — Results\n"]

    # Phase 1 table
    lines.append("## Phase 1: 4-Way Direct Comparison\n")
    lines.append("| Policy | Domain | EMR | FDT | ROUGE-L | BLEU-4 |")
    lines.append("|--------|--------|-----|-----|---------|--------|")

    policy_order = ["Baseline", "Sliding", "H2O", "H2O+"]
    for policy in policy_order:
        for ppl_id in PROMPTS:
            key = (policy, ppl_id)
            if key in phase1_results:
                m = phase1_results[key]
                lines.append(
                    f"| {policy} | {DOMAINS[ppl_id]} | "
                    f"{m['emr']:.3f} | {m['fdt']} | "
                    f"{m['rouge_l']:.3f} | {m['bleu4']:.3f} |"
                )

    # Phase 2 table
    lines.append("\n\n## Phase 2: H2O+ Keep Ratio Sweep\n")
    lines.append("| keep_ratio | HH | Recent | Literary EMR | Technical EMR |")
    lines.append("|-----------|-----|--------|------------|------------|")

    keep_ratios = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    kr_labels = ["00", "10", "20", "30", "50", "70", "90"]

    for kr, label in zip(keep_ratios, kr_labels):
        available = 172
        hh = int(available * kr)
        recent = available - hh
        lit_emr = phase2_results.get(("PPL01", label), {}).get("emr")
        tech_emr = phase2_results.get(("PPL03", label), {}).get("emr")
        lit_str = f"{lit_emr:.3f}" if lit_emr is not None else "N/A"
        tech_str = f"{tech_emr:.3f}" if tech_emr is not None else "N/A"
        lines.append(f"| {kr} | {hh} | {recent} | {lit_str} | {tech_str} |")

    # Sliding reference
    sl_lit = sliding_ref.get("PPL01", {}).get("emr")
    sl_tech = sliding_ref.get("PPL03", {}).get("emr")
    sl_lit_str = f"**{sl_lit:.3f}**" if sl_lit is not None else "N/A"
    sl_tech_str = f"**{sl_tech:.3f}**" if sl_tech is not None else "N/A"
    lines.append(f"| Sliding | 0 | 172 | {sl_lit_str} | {sl_tech_str} |")

    # Phase 3 placeholder
    lines.append("\n\n## Phase 3: H2O vs H2O+ Head-to-Head\n")
    lines.append("See console output for detailed comparison.\n")

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)
    print(f"\n  Report saved to: {report_path}")


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 72)
    print("  Round 13: H2O+ (Per-Head Eviction) Analysis")
    print("=" * 72)

    phase1_results = analyze_phase1()
    phase2_results, sliding_ref = analyze_phase2()
    analyze_phase3()
    verdict(phase1_results, phase2_results, sliding_ref)
    generate_report(phase1_results, phase2_results, sliding_ref)


if __name__ == "__main__":
    main()
