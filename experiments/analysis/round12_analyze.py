#!/usr/bin/env python3
"""Round 12 Analysis: H2O Root Cause Analysis.

Hypotheses:
  H1: Budget split (keep_ratio) is the primary cause of H2O underperformance
  H3: Decay parameter neutralizes heavy hitter selection
  H5: Heavy hitters are fundamentally valueless for autoregressive generation

Phases:
  1) Keep Ratio Sweep — EMR vs keep_ratio curve
  3) Decay Sweep — EMR vs decay curve
  4) Sliding Reduction — SL-172 vs SL-086 vs H2O-50

Usage:
    python experiments/analysis/round12_analyze.py
"""

import json
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
#  Phase 1: Keep Ratio Sweep (H1)
# ============================================================
def analyze_phase1():
    """Analyze keep_ratio sweep results."""
    print("\n" + "=" * 72)
    print("  Phase 1: Keep Ratio Sweep (H1 — Budget Split)")
    print("=" * 72)

    keep_ratios = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    kr_labels = ["00", "10", "20", "30", "50", "70", "90", "100"]

    results = {}  # {(ppl_id, kr_label): metrics_dict}

    # Load Sliding reference from Round 11
    sliding_ref = {}
    for ppl_id in PROMPTS:
        base_file = R11_DIR / f"{ppl_id}-2048-base.jsonl"
        sl_file = R11_DIR / f"{ppl_id}-2048-sl.jsonl"
        if base_file.exists() and sl_file.exists():
            sliding_ref[ppl_id] = analyze_ppl_pair(base_file, sl_file)

    # Load KR results
    for ppl_id in PROMPTS:
        base_file = R11_DIR / f"{ppl_id}-2048-base.jsonl"
        if not base_file.exists():
            print(f"  SKIP {ppl_id} (no baseline in round11)")
            continue

        for kr, label in zip(keep_ratios, kr_labels):
            if label == "50":
                # KR-50 = Round 11 H2O default
                exp_file = R11_DIR / f"{ppl_id}-2048-h2o.jsonl"
            else:
                exp_file = R12_DIR / f"KR-{label}-{ppl_id}.jsonl"

            if not exp_file.exists():
                print(f"  SKIP KR-{label}-{ppl_id} (not found)")
                continue

            r = analyze_ppl_pair(base_file, exp_file)
            results[(ppl_id, label)] = r

    if not results:
        print("  No Phase 1 results found.")
        return {}

    # Print table
    print(f"\n{'keep_ratio':>10}", end="")
    for ppl_id in PROMPTS:
        print(f"  {DOMAINS[ppl_id]+' EMR':>14}  {DOMAINS[ppl_id]+' TopK':>14}", end="")
    print()
    print("-" * (10 + len(PROMPTS) * 32))

    for kr, label in zip(keep_ratios, kr_labels):
        print(f"  {kr:>8.1f}", end="")
        for ppl_id in PROMPTS:
            r = results.get((ppl_id, label))
            if r:
                print(f"  {r['emr']:>14.3f}  {r['topk']:>14.3f}", end="")
            else:
                print(f"  {'N/A':>14}  {'N/A':>14}", end="")
        print()

    # Sliding reference line
    print(f"  {'Sliding':>8}", end="")
    for ppl_id in PROMPTS:
        sl = sliding_ref.get(ppl_id)
        if sl:
            print(f"  {sl['emr']:>14.3f}  {sl['topk']:>14.3f}", end="")
        else:
            print(f"  {'N/A':>14}  {'N/A':>14}", end="")
    print("  ← reference")

    # H1 Judgment
    print("\n── H1 Judgment ──")
    for ppl_id in PROMPTS:
        kr00 = results.get((ppl_id, "00"))
        sl = sliding_ref.get(ppl_id)
        kr50 = results.get((ppl_id, "50"))

        if kr00 and sl:
            diff = abs(kr00["emr"] - sl["emr"])
            print(f"  {DOMAINS[ppl_id]}: |KR-00 EMR - Sliding EMR| = "
                  f"|{kr00['emr']:.3f} - {sl['emr']:.3f}| = {diff:.3f}"
                  f"  {'→ H1 CONFIRMED' if diff < 0.02 else '→ H1 NOT confirmed'}")

        if kr00 and kr50:
            print(f"  {DOMAINS[ppl_id]}: KR-00 EMR={kr00['emr']:.3f}, "
                  f"KR-50 EMR={kr50['emr']:.3f}, "
                  f"Δ={kr00['emr'] - kr50['emr']:+.3f}"
                  f"  {'→ recent > HH' if kr00['emr'] > kr50['emr'] else ''}")

    # Find optimal keep_ratio
    print("\n── Optimal keep_ratio ──")
    for ppl_id in PROMPTS:
        best_kr, best_emr = None, -1.0
        for kr, label in zip(keep_ratios, kr_labels):
            r = results.get((ppl_id, label))
            if r and r["emr"] > best_emr:
                best_emr = r["emr"]
                best_kr = kr
        if best_kr is not None:
            print(f"  {DOMAINS[ppl_id]}: best keep_ratio={best_kr:.1f}, EMR={best_emr:.3f}")

    # Check monotonicity: does EMR decrease as keep_ratio increases?
    print("\n── Monotonicity ──")
    for ppl_id in PROMPTS:
        emrs = []
        for kr, label in zip(keep_ratios, kr_labels):
            r = results.get((ppl_id, label))
            if r:
                emrs.append((kr, r["emr"]))
        if len(emrs) >= 3:
            decreasing = all(emrs[i][1] >= emrs[i + 1][1] - 0.02
                             for i in range(len(emrs) - 1))
            print(f"  {DOMAINS[ppl_id]}: EMR monotonically decreasing with keep_ratio? "
                  f"{'YES' if decreasing else 'NO'}")
            print(f"    {' → '.join(f'{kr:.1f}:{emr:.3f}' for kr, emr in emrs)}")

    return {"results": results, "sliding_ref": sliding_ref}


# ============================================================
#  Phase 3: Decay Sweep (H3)
# ============================================================
def analyze_phase3():
    """Analyze decay sweep results."""
    print("\n" + "=" * 72)
    print("  Phase 3: Decay Parameter Sweep (H3)")
    print("=" * 72)

    decays = [0.00, 0.01, 0.05, 0.10, 0.50, 0.90]
    d_labels = ["000", "001", "005", "010", "050", "090"]

    results = {}

    for ppl_id in PROMPTS:
        base_file = R11_DIR / f"{ppl_id}-2048-base.jsonl"
        if not base_file.exists():
            continue

        for decay, label in zip(decays, d_labels):
            if label == "010":
                # D-010 = Round 11 H2O default
                exp_file = R11_DIR / f"{ppl_id}-2048-h2o.jsonl"
            else:
                exp_file = R12_DIR / f"D-{label}-{ppl_id}.jsonl"

            if not exp_file.exists():
                print(f"  SKIP D-{label}-{ppl_id} (not found)")
                continue

            r = analyze_ppl_pair(base_file, exp_file)
            results[(ppl_id, label)] = r

    if not results:
        print("  No Phase 3 results found.")
        return {}

    # Print table
    print(f"\n{'decay':>8} {'memory':>8}", end="")
    for ppl_id in PROMPTS:
        print(f"  {DOMAINS[ppl_id]+' EMR':>14}  {DOMAINS[ppl_id]+' TopK':>14}", end="")
    print()
    print("-" * (18 + len(PROMPTS) * 32))

    for decay, label in zip(decays, d_labels):
        # Effective memory range (steps where score > 5% of original)
        if decay == 0.0:
            mem = "∞"
        elif decay >= 1.0:
            mem = "0"
        else:
            import math
            mem = f"~{int(math.log(0.05) / math.log(1 - decay))}"

        print(f"  {decay:>6.2f} {mem:>8}", end="")
        for ppl_id in PROMPTS:
            r = results.get((ppl_id, label))
            if r:
                print(f"  {r['emr']:>14.3f}  {r['topk']:>14.3f}", end="")
            else:
                print(f"  {'N/A':>14}  {'N/A':>14}", end="")
        print()

    # H3 Judgment
    print("\n── H3 Judgment ──")
    for ppl_id in PROMPTS:
        emrs = []
        for decay, label in zip(decays, d_labels):
            r = results.get((ppl_id, label))
            if r:
                emrs.append((decay, r["emr"]))

        if len(emrs) >= 2:
            emr_values = [e for _, e in emrs]
            emr_range = max(emr_values) - min(emr_values)
            best_decay = max(emrs, key=lambda x: x[1])
            worst_decay = min(emrs, key=lambda x: x[1])

            print(f"  {DOMAINS[ppl_id]}: EMR range = {emr_range:.3f}"
                  f"  {'→ H3 REJECTED (decay irrelevant)' if emr_range < 0.05 else '→ H3 plausible'}")
            print(f"    Best:  decay={best_decay[0]:.2f}, EMR={best_decay[1]:.3f}")
            print(f"    Worst: decay={worst_decay[0]:.2f}, EMR={worst_decay[1]:.3f}")

    return {"results": results}


# ============================================================
#  Phase 4: Sliding Window Reduction (H5)
# ============================================================
def analyze_phase4():
    """Analyze reduced sliding window vs H2O comparison."""
    print("\n" + "=" * 72)
    print("  Phase 4: Sliding Window Reduction (H5 — HH Value)")
    print("=" * 72)

    results = {}

    print(f"\n{'Config':<12} {'Recent':>7} {'HH':>4} {'Total':>6}", end="")
    for ppl_id in PROMPTS:
        print(f"  {DOMAINS[ppl_id]+' EMR':>14}  {DOMAINS[ppl_id]+' TopK':>14}", end="")
    print()
    print("-" * (32 + len(PROMPTS) * 32))

    configs = [
        ("SL-172", "sliding", 172, 0, 212),
        ("SL-086", "sliding", 86, 0, 126),
        ("H2O-50", "h2o", 86, 86, 212),
    ]

    for config_name, policy, recent, hh, total in configs:
        print(f"  {config_name:<10} {recent:>7} {hh:>4} {total:>6}", end="")

        for ppl_id in PROMPTS:
            base_file = R11_DIR / f"{ppl_id}-2048-base.jsonl"

            if config_name == "SL-172":
                exp_file = R11_DIR / f"{ppl_id}-2048-sl.jsonl"
            elif config_name == "H2O-50":
                exp_file = R11_DIR / f"{ppl_id}-2048-h2o.jsonl"
            else:  # SL-086
                exp_file = R12_DIR / f"SL-086-{ppl_id}.jsonl"

            if base_file.exists() and exp_file.exists():
                r = analyze_ppl_pair(base_file, exp_file)
                results[(ppl_id, config_name)] = r
                print(f"  {r['emr']:>14.3f}  {r['topk']:>14.3f}", end="")
            else:
                print(f"  {'N/A':>14}  {'N/A':>14}", end="")
        print()

    # H5 Judgment
    print("\n── H5 Judgment ──")
    print("  Comparing SL-086 (86 recent, 0 HH) vs H2O-50 (86 recent, 86 HH)")
    print("  → If EMR similar: HH adds no value (H5 confirmed)")
    print("  → If H2O-50 > SL-086: HH has partial value")
    print()

    for ppl_id in PROMPTS:
        sl086 = results.get((ppl_id, "SL-086"))
        h2o50 = results.get((ppl_id, "H2O-50"))
        sl172 = results.get((ppl_id, "SL-172"))

        if sl086 and h2o50:
            diff = h2o50["emr"] - sl086["emr"]
            print(f"  {DOMAINS[ppl_id]}: H2O-50 EMR={h2o50['emr']:.3f}, "
                  f"SL-086 EMR={sl086['emr']:.3f}, "
                  f"Δ={diff:+.3f}"
                  f"  {'→ HH valueless' if abs(diff) < 0.02 else '→ HH has value' if diff > 0.05 else '→ HH marginal'}")

        if sl086 and sl172:
            diff = sl172["emr"] - sl086["emr"]
            print(f"  {DOMAINS[ppl_id]}: SL-172 EMR={sl172['emr']:.3f}, "
                  f"SL-086 EMR={sl086['emr']:.3f}, "
                  f"Δ={diff:+.3f}"
                  f"  (recent window size effect)")

    return {"results": results}


# ============================================================
#  Hypothesis Verdict
# ============================================================
def judge_hypotheses(p1, p3, p4):
    """Synthesize all phase results into hypothesis verdicts."""
    print("\n" + "=" * 72)
    print("  HYPOTHESIS VERDICTS")
    print("=" * 72)

    # H1: Budget Split
    print("\n  H1: Recent Window 축소가 주원인 (Budget Split)")
    h1_confirmed = True
    for ppl_id in PROMPTS:
        kr00 = p1.get("results", {}).get((ppl_id, "00"))
        sl = p1.get("sliding_ref", {}).get(ppl_id)
        if kr00 and sl:
            diff = abs(kr00["emr"] - sl["emr"])
            if diff >= 0.02:
                h1_confirmed = False
    print(f"    Verdict: {'CONFIRMED' if h1_confirmed else 'NOT CONFIRMED'}")

    # H3: Decay
    print("\n  H3: Decay 파라미터가 Heavy Hitter 선택을 무력화")
    h3_plausible = False
    for ppl_id in PROMPTS:
        emrs = []
        for label in ["000", "001", "005", "010", "050", "090"]:
            r = p3.get("results", {}).get((ppl_id, label))
            if r:
                emrs.append(r["emr"])
        if emrs and (max(emrs) - min(emrs)) >= 0.05:
            h3_plausible = True
    print(f"    Verdict: {'PLAUSIBLE' if h3_plausible else 'REJECTED (decay irrelevant)'}")

    # H5: HH Valueless
    print("\n  H5: Heavy Hitter가 생성 품질에 본질적으로 무관")
    h5_confirmed = True
    for ppl_id in PROMPTS:
        sl086 = p4.get("results", {}).get((ppl_id, "SL-086"))
        h2o50 = p4.get("results", {}).get((ppl_id, "H2O-50"))
        if sl086 and h2o50:
            diff = h2o50["emr"] - sl086["emr"]
            if diff > 0.02:
                h5_confirmed = False
    print(f"    Verdict: {'CONFIRMED' if h5_confirmed else 'NOT CONFIRMED (HH has some value)'}")

    # Overall conclusion
    print("\n  ── Overall Conclusion ──")
    if h1_confirmed and h5_confirmed:
        print("    시나리오 A: Budget split이 주원인.")
        print("    H2O의 50:50 분할이 recent window를 절반으로 축소하여 품질 저하.")
        print("    Heavy hitter는 autoregressive 생성에서 무가치.")
    elif h1_confirmed and not h5_confirmed:
        print("    시나리오 C: Budget split이 주원인이나, HH에 부분적 가치 존재.")
        print("    keep_ratio를 0.1~0.2로 낮추면 H2O 개선 가능.")
    elif not h1_confirmed:
        print("    Budget split만으로 설명 불가. Phase 2 (Random baseline) 필요.")
    print()

    return {
        "h1": "CONFIRMED" if h1_confirmed else "NOT_CONFIRMED",
        "h3": "PLAUSIBLE" if h3_plausible else "REJECTED",
        "h5": "CONFIRMED" if h5_confirmed else "NOT_CONFIRMED",
    }


# ============================================================
#  Report Generation
# ============================================================
def generate_report(p1, p3, p4, verdicts):
    """Generate markdown report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "round12_report.md"

    lines = [
        "# Round 12: H2O 성능 저하 원인 분석 Report",
        "",
        "## 실험 개요",
        "",
        "H2O가 Sliding보다 열등한 원인을 4가지 가설로 분석.",
        "26개 신규 실험 + Round 11 baseline 재사용.",
        "",
        "| Phase | 가설 | 실험 수 | 핵심 변수 |",
        "|-------|------|---------|----------|",
        "| 1 | H1: Budget Split | 14 | keep_ratio 0.0~1.0 |",
        "| 3 | H3: Decay | 10 | decay 0.0~0.9 |",
        "| 4 | H5: HH Value | 2 | Sliding(86) vs H2O(86+86) |",
        "",
        "---",
        "",
    ]

    # Phase 1 table
    lines.append("## Phase 1: Keep Ratio Sweep (H1)")
    lines.append("")
    lines.append("| keep_ratio | HH | Recent |")
    for ppl_id in PROMPTS:
        lines[-1] += f" {DOMAINS[ppl_id]} EMR |"
    lines.append("|-----------|-----|--------|")
    for ppl_id in PROMPTS:
        lines[-1] += "------------|"

    keep_ratios = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    kr_labels = ["00", "10", "20", "30", "50", "70", "90", "100"]

    for kr, label in zip(keep_ratios, kr_labels):
        available = 172
        hh = int(available * kr)
        recent = available - hh
        row = f"| {kr:.1f} | {hh} | {recent} |"
        for ppl_id in PROMPTS:
            r = p1.get("results", {}).get((ppl_id, label))
            if r:
                emr_str = f"**{r['emr']:.3f}**" if label == "00" else f"{r['emr']:.3f}"
                row += f" {emr_str} |"
            else:
                row += " N/A |"
        lines.append(row)

    # Sliding reference
    row = "| Sliding | 0 | 172 |"
    for ppl_id in PROMPTS:
        sl = p1.get("sliding_ref", {}).get(ppl_id)
        if sl:
            row += f" **{sl['emr']:.3f}** |"
        else:
            row += " N/A |"
    lines.append(row)

    lines.extend(["", ""])

    # Phase 3 table
    lines.append("## Phase 3: Decay Sweep (H3)")
    lines.append("")
    lines.append("| decay | 유효 범위 |")
    for ppl_id in PROMPTS:
        lines[-1] += f" {DOMAINS[ppl_id]} EMR |"
    lines.append("|-------|----------|")
    for ppl_id in PROMPTS:
        lines[-1] += "------------|"

    decays = [0.00, 0.01, 0.05, 0.10, 0.50, 0.90]
    d_labels = ["000", "001", "005", "010", "050", "090"]

    import math
    for decay, label in zip(decays, d_labels):
        if decay == 0.0:
            mem = "∞"
        else:
            mem = f"~{int(math.log(0.05) / math.log(1 - decay))} steps"
        row = f"| {decay:.2f} | {mem} |"
        for ppl_id in PROMPTS:
            r = p3.get("results", {}).get((ppl_id, label))
            if r:
                row += f" {r['emr']:.3f} |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.extend(["", ""])

    # Phase 4 table
    lines.append("## Phase 4: Sliding Reduction (H5)")
    lines.append("")
    lines.append("| Config | Recent | HH | Total |")
    for ppl_id in PROMPTS:
        lines[-1] += f" {DOMAINS[ppl_id]} EMR |"
    lines.append("|--------|--------|-----|-------|")
    for ppl_id in PROMPTS:
        lines[-1] += "------------|"

    configs = [
        ("SL-172", 172, 0, 212),
        ("SL-086", 86, 0, 126),
        ("H2O-50", 86, 86, 212),
    ]
    for name, recent, hh, total in configs:
        row = f"| {name} | {recent} | {hh} | {total} |"
        for ppl_id in PROMPTS:
            r = p4.get("results", {}).get((ppl_id, name))
            if r:
                row += f" {r['emr']:.3f} |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.extend(["", ""])

    # Verdicts
    lines.append("## 가설 판정 결과")
    lines.append("")
    lines.append("| 가설 | 판정 | 근거 |")
    lines.append("|------|------|------|")
    lines.append(f"| H1: Budget Split | {verdicts['h1']} | KR-00 vs Sliding EMR 비교 |")
    lines.append(f"| H3: Decay | {verdicts['h3']} | Decay별 EMR 범위 |")
    lines.append(f"| H5: HH Value | {verdicts['h5']} | SL-086 vs H2O-50 비교 |")

    lines.extend(["", ""])

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nReport: {report_path}")


# ============================================================
#  Main
# ============================================================
def main():
    if not R12_DIR.exists() and not R11_DIR.exists():
        print(f"ERROR: Neither {R12_DIR} nor {R11_DIR} found.")
        sys.exit(1)

    p1 = analyze_phase1()
    p3 = analyze_phase3()
    p4 = analyze_phase4()
    verdicts = judge_hypotheses(p1, p3, p4)
    generate_report(p1, p3, p4, verdicts)

    print("Done.")


if __name__ == "__main__":
    main()
