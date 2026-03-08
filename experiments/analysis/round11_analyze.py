#!/usr/bin/env python3
"""Round 11 Analysis: Aggressive Eviction Deep Dive.

Sub-experiments:
  A) Long Context PPL (2048tok, 80% eviction) — 5 domains
  B) NIAH Redesign — passkey length scaling + simple facts
  C) PPL03 Position Sensitivity — injection position vs domain

Usage:
    python experiments/analysis/round11_analyze.py
"""

import json
import sys
from pathlib import Path

# Add parent for quality_metrics import
sys.path.insert(0, str(Path(__file__).parent))
from quality_metrics import (
    load_jsonl,
    compute_emr,
    compute_fdt,
    compute_suffix_emr,
    compute_rouge_l,
    compute_bleu4,
    compute_topk_overlap,
    compute_logit_entropy,
    compute_baseline_token_rank,
    evaluate_niah_retrieval,
    compute_f1_score,
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "round11"
REPORT_DIR = Path(__file__).parent.parent / "reports"

# NIAH expected answers
NIAH_EXPECTED = {
    "P3": "729",
    "PASS": "58291",
    "P10": "3847291056",
    "P20": "38472910564813927605",
    "NUM": "42",
    "DATE2": "July 20, 1969",
}


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

    base_entropy = compute_logit_entropy(base_tokens)
    exp_entropy = compute_logit_entropy(exp_tokens)
    avg_base_e = sum(base_entropy) / len(base_entropy) if base_entropy else 1.0
    avg_exp_e = sum(exp_entropy) / len(exp_entropy) if exp_entropy else 0.0
    entropy_ratio = avg_exp_e / avg_base_e if avg_base_e > 0 else 0.0

    ranks = compute_baseline_token_rank(base_tokens, exp_tokens)
    avg_rank = sum(ranks) / len(ranks) if ranks else 0.0

    evicted = exp_summary.get("evicted_tokens_total", 0)

    return {
        "emr": emr,
        "fdt": fdt,
        "suffix_emr": suffix_emr,
        "rouge_l": rouge["f1"],
        "bleu4": bleu,
        "topk": avg_topk,
        "entropy_ratio": entropy_ratio,
        "avg_rank": avg_rank,
        "evicted": evicted,
        "n_tokens": len(base_tokens),
    }


def analyze_niah(base_path, exp_path, expected_answer):
    """Analyze NIAH retrieval."""
    base_tokens, _ = load_jsonl(base_path)
    exp_tokens, exp_summary = load_jsonl(exp_path)

    base_text = "".join(t.get("text", "") for t in base_tokens)
    exp_text = "".join(t.get("text", "") for t in exp_tokens)

    base_eval = evaluate_niah_retrieval(base_text, expected_answer)
    exp_eval = evaluate_niah_retrieval(exp_text, expected_answer)

    evicted = exp_summary.get("evicted_tokens_total", 0)

    return {
        "base_success": base_eval["success"],
        "base_score": base_eval["retrieval_score"],
        "exp_success": exp_eval["success"],
        "exp_score": exp_eval["retrieval_score"],
        "evicted": evicted,
        "base_text": base_text[:200],
        "exp_text": exp_text[:200],
    }


# ============================================================
#  Sub-experiment A: Long Context PPL
# ============================================================
def analyze_11a():
    """Analyze long context PPL results."""
    print("\n" + "=" * 70)
    print("  Sub-experiment A: Long Context PPL (2048tok, 80% eviction)")
    print("=" * 70)

    ppl_ids = ["PPL01", "PPL02", "PPL03", "PPL04", "PPL05"]
    domains = {
        "PPL01": "Literary",
        "PPL02": "Encyclopedic",
        "PPL03": "Technical",
        "PPL04": "Conversational",
        "PPL05": "News",
    }

    results = {}

    print(f"\n{'Prompt':<22} {'Policy':<8} {'EMR':>6} {'FDT':>6} {'ROUGE':>6} "
          f"{'BLEU':>6} {'TopK':>6} {'EntR':>6} {'Evicted':>8}")
    print("-" * 80)

    for ppl_id in ppl_ids:
        base_file = RESULTS_DIR / f"{ppl_id}-2048-base.jsonl"
        if not base_file.exists():
            print(f"  SKIP {ppl_id} (no baseline)")
            continue

        for policy in ["sl", "h2o"]:
            exp_file = RESULTS_DIR / f"{ppl_id}-2048-{policy}.jsonl"
            if not exp_file.exists():
                print(f"  SKIP {ppl_id}-{policy}")
                continue

            r = analyze_ppl_pair(base_file, exp_file)
            key = f"{ppl_id}-{policy}"
            results[key] = r

            print(f"  {domains[ppl_id]:<20} {policy:<8} {r['emr']:>6.3f} {r['fdt']:>6d} "
                  f"{r['rouge_l']:>6.3f} {r['bleu4']:>6.3f} {r['topk']:>6.3f} "
                  f"{r['entropy_ratio']:>6.1f} {r['evicted']:>8d}")

    if results:
        # Summary
        sl_emrs = [r["emr"] for k, r in results.items() if "-sl" in k]
        h2o_emrs = [r["emr"] for k, r in results.items() if "-h2o" in k]
        sl_topks = [r["topk"] for k, r in results.items() if "-sl" in k]
        h2o_topks = [r["topk"] for k, r in results.items() if "-h2o" in k]

        print("-" * 80)
        if sl_emrs:
            print(f"  {'Sliding avg':<20} {'sl':<8} {sum(sl_emrs)/len(sl_emrs):>6.3f} "
                  f"{'':>6} {'':>6} {'':>6} {sum(sl_topks)/len(sl_topks):>6.3f}")
        if h2o_emrs:
            print(f"  {'H2O avg':<20} {'h2o':<8} {sum(h2o_emrs)/len(h2o_emrs):>6.3f} "
                  f"{'':>6} {'':>6} {'':>6} {sum(h2o_topks)/len(h2o_topks):>6.3f}")

    return results


# ============================================================
#  Sub-experiment B: NIAH Redesign
# ============================================================
def analyze_11b():
    """Analyze NIAH with passkey length scaling and simple facts."""
    print("\n" + "=" * 70)
    print("  Sub-experiment B: NIAH Redesign (80% eviction, 16 blocks)")
    print("=" * 70)

    needle_keys = ["P3", "PASS", "P10", "P20", "NUM", "DATE2"]
    needle_labels = {
        "P3": "Pass-3dig",
        "PASS": "Pass-5dig",
        "P10": "Pass-10dig",
        "P20": "Pass-20dig",
        "NUM": "Number(42)",
        "DATE2": "Date(1969)",
    }
    depths = [10, 50, 90]
    blocks = 16
    results = {}

    # Table: per-needle success rates
    print(f"\n{'Needle':<14} {'Depth':>5} {'Base':>6} {'Sl':>6} {'H2O':>6} "
          f"{'Sl_score':>8} {'H2O_score':>9} {'Evicted':>8}")
    print("-" * 72)

    for nk in needle_keys:
        expected = NIAH_EXPECTED[nk]
        for depth_pct in depths:
            niah_id = f"NIAH-{nk}-D{depth_pct}-B{blocks}"
            base_file = RESULTS_DIR / f"{niah_id}-base.jsonl"

            if not base_file.exists():
                continue

            for policy in ["sl", "h2o"]:
                exp_file = RESULTS_DIR / f"{niah_id}-{policy}.jsonl"
                if not exp_file.exists():
                    continue
                r = analyze_niah(base_file, exp_file, expected)
                results[f"{niah_id}-{policy}"] = r

            # Print row
            sl_key = f"{niah_id}-sl"
            h2o_key = f"{niah_id}-h2o"
            sl_r = results.get(sl_key, {})
            h2o_r = results.get(h2o_key, {})

            base_ok = "✓" if sl_r.get("base_success", False) else "✗"
            sl_ok = "✓" if sl_r.get("exp_success", False) else "✗"
            h2o_ok = "✓" if h2o_r.get("exp_success", False) else "✗"
            sl_score = sl_r.get("exp_score", 0.0)
            h2o_score = h2o_r.get("exp_score", 0.0)
            evicted = sl_r.get("evicted", 0)

            print(f"  {needle_labels[nk]:<12} D{depth_pct:>3}% {base_ok:>6} {sl_ok:>6} {h2o_ok:>6} "
                  f"{sl_score:>8.3f} {h2o_score:>9.3f} {evicted:>8}")

    # Summary table: success rate per needle type
    print("\n── NIAH Summary (success rate by needle type) ──")
    print(f"{'Needle':<14} {'Base':>8} {'Sliding':>8} {'H2O':>8}")
    print("-" * 42)

    for nk in needle_keys:
        base_total = base_ok_count = sl_ok_count = h2o_ok_count = 0
        for depth_pct in depths:
            niah_id = f"NIAH-{nk}-D{depth_pct}-B{blocks}"
            sl_key = f"{niah_id}-sl"
            h2o_key = f"{niah_id}-h2o"
            if sl_key in results:
                base_total += 1
                if results[sl_key]["base_success"]:
                    base_ok_count += 1
                if results[sl_key]["exp_success"]:
                    sl_ok_count += 1
            if h2o_key in results:
                if results[h2o_key]["exp_success"]:
                    h2o_ok_count += 1

        if base_total > 0:
            print(f"  {needle_labels[nk]:<12} {base_ok_count}/{base_total:>4} "
                  f"{sl_ok_count}/{base_total:>4}    {h2o_ok_count}/{base_total:>4}")

    return results


# ============================================================
#  Sub-experiment C: PPL03 Position Sensitivity
# ============================================================
def analyze_11c():
    """Analyze eviction injection position sensitivity."""
    print("\n" + "=" * 70)
    print("  Sub-experiment C: PPL03 Position Sensitivity (2048tok, 80% eviction)")
    print("=" * 70)

    ppl_ids = ["PPL01", "PPL03"]
    domains = {"PPL01": "Literary", "PPL03": "Technical"}
    positions = [25, 50, 75]  # P50 from sub-A, P25/P75 from sub-C
    results = {}

    print(f"\n{'Domain':<14} {'Policy':<8} {'P25':>8} {'P50':>8} {'P75':>8}  (EMR)")
    print("-" * 52)

    for ppl_id in ppl_ids:
        base_file = RESULTS_DIR / f"{ppl_id}-2048-base.jsonl"
        if not base_file.exists():
            continue

        for policy in ["sl", "h2o"]:
            row = {}
            for pos in positions:
                if pos == 50:
                    # P50 data from sub-experiment A
                    exp_file = RESULTS_DIR / f"{ppl_id}-2048-{policy}.jsonl"
                else:
                    exp_file = RESULTS_DIR / f"{ppl_id}-2048-{policy}-P{pos}.jsonl"

                if not exp_file.exists():
                    continue

                r = analyze_ppl_pair(base_file, exp_file)
                key = f"{ppl_id}-{policy}-P{pos}"
                results[key] = r
                row[pos] = r["emr"]

            p25 = f"{row.get(25, 0):>8.3f}" if 25 in row else "    N/A "
            p50 = f"{row.get(50, 0):>8.3f}" if 50 in row else "    N/A "
            p75 = f"{row.get(75, 0):>8.3f}" if 75 in row else "    N/A "
            print(f"  {domains[ppl_id]:<12} {policy:<8} {p25} {p50} {p75}")

    # Detailed comparison for PPL03 vs PPL01
    print(f"\n{'Domain':<14} {'Policy':<8} {'Pos':>5} {'EMR':>6} {'FDT':>6} {'ROUGE':>6} "
          f"{'TopK':>6} {'Evicted':>8}")
    print("-" * 66)

    for ppl_id in ppl_ids:
        for policy in ["sl", "h2o"]:
            for pos in positions:
                key = f"{ppl_id}-{policy}-P{pos}"
                if key not in results:
                    continue
                r = results[key]
                print(f"  {domains[ppl_id]:<12} {policy:<8} P{pos:>3}% {r['emr']:>6.3f} "
                      f"{r['fdt']:>6d} {r['rouge_l']:>6.3f} {r['topk']:>6.3f} {r['evicted']:>8d}")

    return results


# ============================================================
#  Report generation
# ============================================================
def generate_report(a_results, b_results, c_results):
    """Generate markdown report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "round11_report.md"

    lines = [
        "# Round 11: Aggressive Eviction Deep Dive Report",
        "",
        "## 실험 개요",
        "",
        "80% eviction (keep_ratio=0.20) 조건에서 두 eviction 정책(Sliding, H2O)의",
        "차이를 심화 분석하는 실험.",
        "",
        "| Sub | 실험 | 주요 변수 | 실험 수 |",
        "|-----|------|----------|--------|",
        "| A | Long Context PPL | 2048tok, 80% eviction, 5 domains | 15 |",
        "| B | NIAH Redesign | passkey 3/5/10/20자리, simple facts | 54 |",
        "| C | Position Sensitivity | PPL03 vs PPL01, inject@25/50/75% | ~12 |",
        "",
        "---",
        "",
    ]

    # Sub-A results
    lines.append("## Sub-A: Long Context PPL (2048tok, 80% eviction)")
    lines.append("")
    domains = {
        "PPL01": "Literary", "PPL02": "Encyclopedic",
        "PPL03": "Technical", "PPL04": "Conversational", "PPL05": "News",
    }

    lines.append("| Prompt | Sliding EMR | H2O EMR | Sliding TopK | H2O TopK | Evicted |")
    lines.append("|--------|------------|---------|-------------|---------|---------|")

    for ppl_id in ["PPL01", "PPL02", "PPL03", "PPL04", "PPL05"]:
        sl_key = f"{ppl_id}-sl"
        h2o_key = f"{ppl_id}-h2o"
        sl = a_results.get(sl_key, {})
        h2o = a_results.get(h2o_key, {})
        sl_emr = f"{sl.get('emr', 0):.3f}" if sl else "N/A"
        h2o_emr = f"{h2o.get('emr', 0):.3f}" if h2o else "N/A"
        sl_topk = f"{sl.get('topk', 0):.3f}" if sl else "N/A"
        h2o_topk = f"{h2o.get('topk', 0):.3f}" if h2o else "N/A"
        evicted = sl.get("evicted", h2o.get("evicted", "N/A"))
        lines.append(
            f"| {domains[ppl_id]} | {sl_emr} | {h2o_emr} | {sl_topk} | {h2o_topk} | {evicted} |"
        )

    lines.extend(["", "---", ""])

    # Sub-B results
    lines.append("## Sub-B: NIAH Redesign")
    lines.append("")

    needle_labels = {
        "P3": "Pass-3dig", "PASS": "Pass-5dig",
        "P10": "Pass-10dig", "P20": "Pass-20dig",
        "NUM": "Number(42)", "DATE2": "Date(1969)",
    }

    lines.append("| Needle | Depth | Baseline | Sliding | H2O |")
    lines.append("|--------|-------|----------|---------|-----|")

    for nk in ["P3", "PASS", "P10", "P20", "NUM", "DATE2"]:
        for depth_pct in [10, 50, 90]:
            niah_id = f"NIAH-{nk}-D{depth_pct}-B16"
            sl_key = f"{niah_id}-sl"
            h2o_key = f"{niah_id}-h2o"
            sl = b_results.get(sl_key, {})
            h2o = b_results.get(h2o_key, {})

            base_ok = "Pass" if sl.get("base_success") else "Fail"
            sl_ok = "Pass" if sl.get("exp_success") else "Fail"
            h2o_ok = "Pass" if h2o.get("exp_success") else "Fail"

            if sl or h2o:
                lines.append(
                    f"| {needle_labels[nk]} | D{depth_pct}% | {base_ok} | {sl_ok} | {h2o_ok} |"
                )

    lines.extend(["", "---", ""])

    # Sub-C results
    lines.append("## Sub-C: Position Sensitivity")
    lines.append("")
    lines.append("| Domain | Policy | P25% EMR | P50% EMR | P75% EMR |")
    lines.append("|--------|--------|----------|----------|----------|")

    for ppl_id in ["PPL01", "PPL03"]:
        for policy in ["sl", "h2o"]:
            cells = []
            for pos in [25, 50, 75]:
                key = f"{ppl_id}-{policy}-P{pos}"
                r = c_results.get(key, {})
                cells.append(f"{r.get('emr', 0):.3f}" if r else "N/A")
            lines.append(
                f"| {domains[ppl_id]} | {policy} | {cells[0]} | {cells[1]} | {cells[2]} |"
            )

    lines.extend(["", "---", ""])

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nReport: {report_path}")


# ============================================================
#  Main
# ============================================================
def main():
    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    a = analyze_11a()
    b = analyze_11b()
    c = analyze_11c()
    generate_report(a, b, c)

    print("\nDone.")


if __name__ == "__main__":
    main()
