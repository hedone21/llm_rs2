#!/usr/bin/env python3
"""Phase 5: Unified analysis — correlation, α calibration, visualization.

Reads results from all phases and produces:
1. Spearman rank correlation (QCF vs each metric)
2. α calibration (linear regression QCF → PPL increase)
3. Cross-benchmark correlation table
4. Scatter plots and summary charts

Usage:
    python analyze.py
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"


def load_summary(phase: str) -> dict:
    """Load _summary.json for a phase."""
    path = RESULTS_DIR / phase / "_summary.json"
    if not path.exists():
        print(f"[warn] {path} not found, skipping {phase}", file=sys.stderr)
        return {}
    with open(path) as f:
        return json.load(f)


def spearman_rho(x: list, y: list) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 3:
        return float("nan")

    def rank(arr):
        sorted_idx = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        for r, i in enumerate(sorted_idx):
            ranks[i] = r + 1.0
        return ranks

    rx = rank(x)
    ry = rank(y)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1))


def linear_regression(x: list, y: list) -> tuple:
    """Simple linear regression y = a*x + b. Returns (slope, intercept, r_squared)."""
    n = len(x)
    if n < 2:
        return (0.0, 0.0, 0.0)
    mx = sum(x) / n
    my = sum(y) / n
    ss_xx = sum((xi - mx) ** 2 for xi in x)
    ss_yy = sum((yi - my) ** 2 for yi in y)
    ss_xy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))

    if ss_xx < 1e-12:
        return (0.0, my, 0.0)
    slope = ss_xy / ss_xx
    intercept = my - slope * mx
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 1e-12 else 0.0
    return (slope, intercept, r_squared)


def analyze_ppl(summary: dict) -> dict:
    """Analyze PPL phase results."""
    results = summary.get("results", [])
    if not results:
        return {}

    # Filter to non-baseline results with valid data
    data_points = [r for r in results if r.get("delta_ppl", 0) > 0 and r["qcf_avg"] > 0]
    if len(data_points) < 3:
        return {"error": "insufficient data points"}

    qcf_values = [r["qcf_avg"] for r in data_points]
    delta_ppl = [r["delta_ppl"] for r in data_points]

    rho = spearman_rho(qcf_values, delta_ppl)
    slope, intercept, r2 = linear_regression(qcf_values, delta_ppl)

    # Per-policy analysis
    policy_results = {}
    for policy in set(r.get("policy", "") for r in data_points):
        subset = [r for r in data_points if r.get("policy") == policy]
        if len(subset) >= 3:
            qcf_p = [r["qcf_avg"] for r in subset]
            dppl_p = [r["delta_ppl"] for r in subset]
            policy_results[policy] = {
                "rho": spearman_rho(qcf_p, dppl_p),
                "alpha": linear_regression(qcf_p, dppl_p)[0],
                "r2": linear_regression(qcf_p, dppl_p)[2],
                "n": len(subset),
            }

    return {
        "rho": rho,
        "alpha": slope,
        "intercept": intercept,
        "r2": r2,
        "n": len(data_points),
        "by_policy": policy_results,
    }


def analyze_niah(summary: dict) -> dict:
    """Analyze NIAH phase results."""
    by_budget = summary.get("results_by_budget", [])
    if len(by_budget) < 3:
        return {"error": "insufficient data"}

    # Filter to entries with eviction (non-baseline)
    data = [r for r in by_budget if r.get("qcf_avg", 0) > 0 or r["budget"] == 2048]
    if len(data) < 3:
        return {"error": "insufficient data"}

    qcf_values = [r.get("qcf_avg", 0) for r in data]
    failure_rate = [1.0 - r["accuracy"] for r in data]

    rho = spearman_rho(qcf_values, failure_rate)
    return {
        "rho": rho,
        "n": len(data),
        "data": [{"budget": r["budget"], "accuracy": r["accuracy"],
                   "qcf": r.get("qcf_avg", 0)} for r in data],
    }


def analyze_qa(summary: dict) -> dict:
    """Analyze QA phase results."""
    by_budget = summary.get("results_by_budget", [])
    if len(by_budget) < 3:
        return {"error": "insufficient data"}

    data = [r for r in by_budget]
    qcf_values = [r.get("qcf_avg", 0) for r in data]
    f1_drop = [1.0 - r["avg_f1"] for r in data]

    rho = spearman_rho(qcf_values, f1_drop)
    return {
        "rho": rho,
        "n": len(data),
        "data": [{"budget": r["budget"], "f1": r["avg_f1"],
                   "qcf": r.get("qcf_avg", 0)} for r in data],
    }


def analyze_mmlu(summary: dict) -> dict:
    """Analyze MMLU phase results."""
    by_budget = summary.get("results_by_budget", [])
    if len(by_budget) < 3:
        return {"error": "insufficient data"}

    data = [r for r in by_budget]
    qcf_values = [r.get("qcf_avg", 0) for r in data]
    error_rate = [1.0 - r["accuracy"] for r in data]

    rho = spearman_rho(qcf_values, error_rate)
    return {
        "rho": rho,
        "n": len(data),
        "data": [{"budget": r["budget"], "accuracy": r["accuracy"],
                   "qcf": r.get("qcf_avg", 0)} for r in data],
    }


def try_plot(ppl_analysis, niah_analysis, qa_analysis, mmlu_analysis):
    """Generate plots if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available, skipping plots", file=sys.stderr)
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("QCF Validation: Correlation with Quality Metrics", fontsize=14)

    # 1. PPL
    ax = axes[0][0]
    if "by_policy" in ppl_analysis:
        for policy, pa in ppl_analysis["by_policy"].items():
            # Load full data from summary
            ppl_summary = load_summary("ppl")
            points = [r for r in ppl_summary.get("results", [])
                      if r.get("policy") == policy and r.get("delta_ppl", 0) > 0]
            if points:
                x = [p["qcf_avg"] for p in points]
                y = [p["delta_ppl"] for p in points]
                ax.scatter(x, y, label=f"{policy} (ρ={pa['rho']:.2f})", s=60)
    ax.set_xlabel("QCF (avg)")
    ax.set_ylabel("ΔPPL")
    ax.set_title(f"PPL (ρ={ppl_analysis.get('rho', 'N/A')})")
    ax.legend()

    # 2. NIAH
    ax = axes[0][1]
    niah_data = niah_analysis.get("data", [])
    if niah_data:
        x = [d["qcf"] for d in niah_data]
        y = [d["accuracy"] for d in niah_data]
        ax.scatter(x, y, s=80, c="red", zorder=5)
        ax.plot(x, y, "r--", alpha=0.5)
    ax.set_xlabel("QCF (avg)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"NIAH (ρ={niah_analysis.get('rho', 'N/A')})")
    ax.set_ylim(-0.05, 1.05)

    # 3. QA
    ax = axes[1][0]
    qa_data = qa_analysis.get("data", [])
    if qa_data:
        x = [d["qcf"] for d in qa_data]
        y = [d["f1"] for d in qa_data]
        ax.scatter(x, y, s=80, c="green", zorder=5)
        ax.plot(x, y, "g--", alpha=0.5)
    ax.set_xlabel("QCF (avg)")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"QA (ρ={qa_analysis.get('rho', 'N/A')})")

    # 4. MMLU
    ax = axes[1][1]
    mmlu_data = mmlu_analysis.get("data", [])
    if mmlu_data:
        x = [d["qcf"] for d in mmlu_data]
        y = [d["accuracy"] for d in mmlu_data]
        ax.scatter(x, y, s=80, c="purple", zorder=5)
        ax.plot(x, y, "m--", alpha=0.5)
        ax.axhline(y=0.25, color="gray", linestyle=":", label="random (25%)")
    ax.set_xlabel("QCF (avg)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"MMLU (ρ={mmlu_analysis.get('rho', 'N/A')})")
    ax.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "qcf_validation_all.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved: {out}", file=sys.stderr)


def main():
    print("=" * 60, file=sys.stderr)
    print("QCF Validation — Unified Analysis", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Load phase summaries
    ppl_summary = load_summary("ppl")
    niah_summary = load_summary("niah")
    qa_summary = load_summary("qa")
    mmlu_summary = load_summary("mmlu")

    # Analyze each phase
    ppl_analysis = analyze_ppl(ppl_summary) if ppl_summary else {}
    niah_analysis = analyze_niah(niah_summary) if niah_summary else {}
    qa_analysis = analyze_qa(qa_summary) if qa_summary else {}
    mmlu_analysis = analyze_mmlu(mmlu_summary) if mmlu_summary else {}

    # Print correlation table
    print("\n┌──────────────────────────────────────────────────────┐", file=sys.stderr)
    print("│        QCF Correlation with Quality Metrics          │", file=sys.stderr)
    print("├────────────┬────────────┬──────────┬────────────────┤", file=sys.stderr)
    print("│ Benchmark  │ ρ(QCF,deg) │ Target   │ Status         │", file=sys.stderr)
    print("├────────────┼────────────┼──────────┼────────────────┤", file=sys.stderr)

    rows = [
        ("PPL", ppl_analysis.get("rho"), 0.85),
        ("NIAH", niah_analysis.get("rho"), 0.70),
        ("QA", qa_analysis.get("rho"), 0.60),
        ("MMLU", mmlu_analysis.get("rho"), 0.60),
    ]
    for name, rho, target in rows:
        if rho is None or rho != rho:  # NaN check
            status = "—"
            rho_str = "N/A"
        else:
            rho_str = f"{rho:.3f}"
            status = "PASS" if rho >= target else "FAIL"
        print(f"│ {name:<10} │ {rho_str:>10} │ ≥{target:.2f}    │ {status:<14} │",
              file=sys.stderr)

    print("└────────────┴────────────┴──────────┴────────────────┘", file=sys.stderr)

    # α calibration (PPL)
    if "alpha" in ppl_analysis:
        print(f"\n[α Calibration] PPL:", file=sys.stderr)
        print(f"  D = {ppl_analysis['alpha']:.4f} × Q + {ppl_analysis['intercept']:.4f}",
              file=sys.stderr)
        print(f"  R² = {ppl_analysis['r2']:.4f}", file=sys.stderr)
        if "by_policy" in ppl_analysis:
            for policy, pa in ppl_analysis["by_policy"].items():
                print(f"  [{policy}] α={pa['alpha']:.4f}, R²={pa['r2']:.4f}, "
                      f"ρ={pa['rho']:.3f}", file=sys.stderr)

    # Generate plots
    try_plot(ppl_analysis, niah_analysis, qa_analysis, mmlu_analysis)

    # Save analysis report
    report = {
        "ppl": ppl_analysis,
        "niah": niah_analysis,
        "qa": qa_analysis,
        "mmlu": mmlu_analysis,
        "correlation_table": [
            {"benchmark": name, "rho": rho, "target": target,
             "pass": rho is not None and rho == rho and rho >= target}
            for name, rho, target in rows
        ],
    }
    save_result(".", "analysis_report", report)
    print(f"\n[analysis] Report saved to {RESULTS_DIR / 'analysis_report.json'}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
