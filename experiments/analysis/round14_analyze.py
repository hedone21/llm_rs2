#!/usr/bin/env python3
"""Round 14 Analysis: H2O/H2O+ with Real Attention Scores.

Re-validates Round 12/13 conclusions after fixing Q4_0 score bug.
Score computation now properly dequantizes Q4_0 K cache on CPU.

Phases:
  1) Baselines (reused from Round 13 — score-independent)
  2) H2O Keep Ratio Sweep (with real scores) — new
  3) H2O+ Keep Ratio Sweep (with real scores) — new
  4) H2O vs H2O+ Head-to-Head — matched comparison

Usage:
    python experiments/analysis/round14_analyze.py
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

R14_DIR = Path(__file__).parent.parent / "results" / "round14"
REPORT_DIR = Path(__file__).parent.parent / "reports"

PROMPTS = ["PPL01", "PPL03"]
DOMAINS = {"PPL01": "Literary", "PPL03": "Technical"}

KEEP_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
KR_LABELS = ["00", "10", "20", "30", "50", "70", "90"]


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
#  Phase 1: Baselines
# ============================================================
def analyze_baselines():
    """Load baseline and sliding results."""
    print("\n" + "=" * 72)
    print("  Phase 1: Baselines (reused from Round 13)")
    print("=" * 72)

    sliding_ref = {}
    for ppl_id in PROMPTS:
        base_file = R14_DIR / f"BASE-{ppl_id}.jsonl"
        sl_file = R14_DIR / f"SL-{ppl_id}.jsonl"
        if base_file.exists() and sl_file.exists():
            sliding_ref[ppl_id] = analyze_ppl_pair(base_file, sl_file)
            print(f"  Sliding {DOMAINS[ppl_id]}: EMR={sliding_ref[ppl_id]['emr']:.3f}")
        else:
            print(f"  WARN: Missing baseline or sliding for {ppl_id}")

    return sliding_ref


# ============================================================
#  Phase 2: H2O Keep Ratio Sweep
# ============================================================
def analyze_h2o_sweep():
    """H2O with real attention scores."""
    print("\n" + "=" * 72)
    print("  Phase 2: H2O Keep Ratio Sweep (with REAL scores)")
    print("=" * 72)

    results = {}
    for ppl_id in PROMPTS:
        base_file = R14_DIR / f"BASE-{ppl_id}.jsonl"
        if not base_file.exists():
            continue
        for kr, label in zip(KEEP_RATIOS, KR_LABELS):
            exp_file = R14_DIR / f"H2O-{label}-{ppl_id}.jsonl"
            if not exp_file.exists():
                continue
            metrics = analyze_ppl_pair(base_file, exp_file)
            results[(ppl_id, label)] = metrics

    print(f"\n{'KR':<6} {'HH':>4} {'Recent':>7} ", end="")
    for ppl_id in PROMPTS:
        print(f"{'EMR(' + DOMAINS[ppl_id][:3] + ')':>10} ", end="")
    print()
    print("-" * 50)

    for kr, label in zip(KEEP_RATIOS, KR_LABELS):
        available = 172
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

    return results


# ============================================================
#  Phase 3: H2O+ Keep Ratio Sweep
# ============================================================
def analyze_h2op_sweep():
    """H2O+ with real attention scores."""
    print("\n" + "=" * 72)
    print("  Phase 3: H2O+ Keep Ratio Sweep (with REAL scores)")
    print("=" * 72)

    results = {}
    for ppl_id in PROMPTS:
        base_file = R14_DIR / f"BASE-{ppl_id}.jsonl"
        if not base_file.exists():
            continue
        for kr, label in zip(KEEP_RATIOS, KR_LABELS):
            exp_file = R14_DIR / f"H2OP-{label}-{ppl_id}.jsonl"
            if not exp_file.exists():
                continue
            metrics = analyze_ppl_pair(base_file, exp_file)
            results[(ppl_id, label)] = metrics

    print(f"\n{'KR':<6} {'HH':>4} {'Recent':>7} ", end="")
    for ppl_id in PROMPTS:
        print(f"{'EMR(' + DOMAINS[ppl_id][:3] + ')':>10} ", end="")
    print()
    print("-" * 50)

    for kr, label in zip(KEEP_RATIOS, KR_LABELS):
        available = 172
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

    return results


# ============================================================
#  Phase 4: H2O vs H2O+ Head-to-Head
# ============================================================
def analyze_head_to_head(h2o_results, h2op_results, sliding_ref):
    """Compare H2O and H2O+ at matched keep_ratios."""
    print("\n" + "=" * 72)
    print("  Phase 4: H2O vs H2O+ Head-to-Head")
    print("=" * 72)

    for ppl_id in PROMPTS:
        sl_emr = sliding_ref.get(ppl_id, {}).get("emr")
        sl_str = f"(Sliding: {sl_emr:.3f})" if sl_emr is not None else ""

        print(f"\n── {DOMAINS[ppl_id]} ({ppl_id}) {sl_str} ──")
        print(f"{'KR':<6} {'H2O EMR':>9} {'H2O+ EMR':>10} {'Δ':>8} {'Winner':>8}")
        print("-" * 50)

        for kr, label in zip(KEEP_RATIOS, KR_LABELS):
            h2o_key = (ppl_id, label)
            h2op_key = (ppl_id, label)

            h2o_emr = h2o_results.get(h2o_key, {}).get("emr")
            h2op_emr = h2op_results.get(h2op_key, {}).get("emr")

            if h2o_emr is not None and h2op_emr is not None:
                delta = h2op_emr - h2o_emr
                winner = "H2O+" if delta > 0.01 else ("H2O" if delta < -0.01 else "TIE")
                print(
                    f"{kr:<6.1f} {h2o_emr:>9.3f} {h2op_emr:>10.3f} "
                    f"{delta:>+8.3f} {winner:>8}"
                )
            else:
                h2o_str = f"{h2o_emr:.3f}" if h2o_emr is not None else "N/A"
                h2op_str = f"{h2op_emr:.3f}" if h2op_emr is not None else "N/A"
                print(f"{kr:<6.1f} {h2o_str:>9} {h2op_str:>10} {'N/A':>8} {'N/A':>8}")


# ============================================================
#  Verdict
# ============================================================
def verdict(h2o_results, h2op_results, sliding_ref):
    """Determine which scenario (A/B/C) matches the data."""
    print("\n" + "=" * 72)
    print("  Verdict")
    print("=" * 72)

    # Compare at kr=0.5
    h2o_emrs = []
    h2op_emrs = []
    sl_emrs = []

    for ppl_id in PROMPTS:
        h2o_key = (ppl_id, "50")
        h2op_key = (ppl_id, "50")
        if h2o_key in h2o_results:
            h2o_emrs.append(h2o_results[h2o_key]["emr"])
        if h2op_key in h2op_results:
            h2op_emrs.append(h2op_results[h2op_key]["emr"])
        if ppl_id in sliding_ref:
            sl_emrs.append(sliding_ref[ppl_id]["emr"])

    if h2o_emrs and h2op_emrs and sl_emrs:
        avg_h2o = sum(h2o_emrs) / len(h2o_emrs)
        avg_h2op = sum(h2op_emrs) / len(h2op_emrs)
        avg_sl = sum(sl_emrs) / len(sl_emrs)
        delta_h2op_h2o = avg_h2op - avg_h2o
        delta_h2o_sl = avg_h2o - avg_sl

        print(f"\n  Average EMR (kr=0.5):")
        print(f"    Sliding:  {avg_sl:.3f}")
        print(f"    H2O:      {avg_h2o:.3f}  (vs Sliding: {delta_h2o_sl:+.3f})")
        print(f"    H2O+:     {avg_h2op:.3f}  (vs H2O: {delta_h2op_h2o:+.3f})")

        # Compare Round 12/13 (buggy) vs Round 14 (fixed)
        print(f"\n  Key Question: Did fixing scores change H2O behavior?")
        print(f"    Round 12/13 H2O kr=0.5 EMR: ~0.51-0.54 (zero scores)")
        print(f"    Round 14   H2O kr=0.5 EMR:  {avg_h2o:.3f} (real scores)")

        if avg_h2o > 0.6:
            print(f"\n  ★ Score fix improved H2O! Real scores restore HH value.")
        else:
            print(f"\n  ★ Score fix did NOT improve H2O. HH truly worthless for 1B.")

        if delta_h2op_h2o >= 0.10:
            print(f"  ★ Scenario A: Per-head HH restores additional value!")
        elif delta_h2op_h2o >= 0.03:
            print(f"  ★ Scenario C: Per-head partially improves H2O")
        else:
            print(f"  ★ Scenario B: Per-head adds no value")

    # Best keep_ratio for each policy
    for policy_name, results in [("H2O", h2o_results), ("H2O+", h2op_results)]:
        best_kr = None
        best_emr = -1.0
        for ppl_id in PROMPTS:
            for label in KR_LABELS:
                key = (ppl_id, label)
                if key in results:
                    emr = results[key]["emr"]
                    if emr > best_emr:
                        best_emr = emr
                        best_kr = label
        if best_kr is not None:
            print(f"\n  Best {policy_name} keep_ratio: kr={int(best_kr)/100:.1f} (best EMR={best_emr:.3f})")


# ============================================================
#  Report
# ============================================================
def generate_report(h2o_results, h2op_results, sliding_ref):
    """Generate markdown report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "round14_report.md"

    lines = [
        "# Round 14: H2O/H2O+ with Real Attention Scores\n",
        "## 실험 개요\n",
        "Q4_0 스코어 버그 수정 후 H2O/H2O+ 재검증.",
        "28개 신규 실험 (14 H2O + 14 H2O+), Round 13 baseline 재사용.\n",
        "**핵심 변경**: `compute_attention_scores()` 추가 — Q4_0 K 캐시를 CPU에서 역양자화,",
        "Q·K^T + softmax 계산 후 ws.scores에 기록.\n",
        "**조건**: 2048 토큰, inject@1024, 80% eviction (ratio=0.20), decay=0.0\n",
        "---\n",
    ]

    # Sliding reference
    lines.append("## Baselines (Round 13 재사용)\n")
    lines.append("| Policy | Domain | EMR |")
    lines.append("|--------|--------|-----|")
    for ppl_id in PROMPTS:
        if ppl_id in sliding_ref:
            lines.append(f"| Sliding | {DOMAINS[ppl_id]} | {sliding_ref[ppl_id]['emr']:.3f} |")

    # H2O sweep
    lines.append("\n---\n")
    lines.append("## Phase 2: H2O Keep Ratio Sweep (Real Scores)\n")
    lines.append("| keep_ratio | HH | Recent | Literary EMR | Technical EMR |")
    lines.append("|-----------|-----|--------|------------|------------|")

    for kr, label in zip(KEEP_RATIOS, KR_LABELS):
        available = 172
        hh = int(available * kr)
        recent = available - hh
        lit = h2o_results.get(("PPL01", label), {}).get("emr")
        tech = h2o_results.get(("PPL03", label), {}).get("emr")
        lit_s = f"{lit:.3f}" if lit is not None else "N/A"
        tech_s = f"{tech:.3f}" if tech is not None else "N/A"
        lines.append(f"| {kr} | {hh} | {recent} | {lit_s} | {tech_s} |")

    # H2O+ sweep
    lines.append("\n---\n")
    lines.append("## Phase 3: H2O+ Keep Ratio Sweep (Real Scores)\n")
    lines.append("| keep_ratio | HH | Recent | Literary EMR | Technical EMR |")
    lines.append("|-----------|-----|--------|------------|------------|")

    for kr, label in zip(KEEP_RATIOS, KR_LABELS):
        available = 172
        hh = int(available * kr)
        recent = available - hh
        lit = h2op_results.get(("PPL01", label), {}).get("emr")
        tech = h2op_results.get(("PPL03", label), {}).get("emr")
        lit_s = f"{lit:.3f}" if lit is not None else "N/A"
        tech_s = f"{tech:.3f}" if tech is not None else "N/A"
        lines.append(f"| {kr} | {hh} | {recent} | {lit_s} | {tech_s} |")

    # Head-to-Head
    lines.append("\n---\n")
    lines.append("## Phase 4: H2O vs H2O+ Head-to-Head\n")
    lines.append("| keep_ratio | H2O EMR (Lit) | H2O+ EMR (Lit) | Δ | H2O EMR (Tech) | H2O+ EMR (Tech) | Δ |")
    lines.append("|-----------|:---:|:---:|:---:|:---:|:---:|:---:|")

    for kr, label in zip(KEEP_RATIOS, KR_LABELS):
        h2o_lit = h2o_results.get(("PPL01", label), {}).get("emr")
        h2op_lit = h2op_results.get(("PPL01", label), {}).get("emr")
        h2o_tech = h2o_results.get(("PPL03", label), {}).get("emr")
        h2op_tech = h2op_results.get(("PPL03", label), {}).get("emr")

        def fmt(v):
            return f"{v:.3f}" if v is not None else "N/A"

        def fmt_delta(a, b):
            return f"{b - a:+.3f}" if a is not None and b is not None else "N/A"

        lines.append(
            f"| {kr} | {fmt(h2o_lit)} | {fmt(h2op_lit)} | {fmt_delta(h2o_lit, h2op_lit)} "
            f"| {fmt(h2o_tech)} | {fmt(h2op_tech)} | {fmt_delta(h2o_tech, h2op_tech)} |"
        )

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)
    print(f"\n  Report saved to: {report_path}")


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 72)
    print("  Round 14: H2O/H2O+ with Real Attention Scores")
    print("  (Q4_0 score bug fixed)")
    print("=" * 72)

    sliding_ref = analyze_baselines()
    h2o_results = analyze_h2o_sweep()
    h2op_results = analyze_h2op_sweep()
    analyze_head_to_head(h2o_results, h2op_results, sliding_ref)
    verdict(h2o_results, h2op_results, sliding_ref)
    generate_report(h2o_results, h2op_results, sliding_ref)


if __name__ == "__main__":
    main()
