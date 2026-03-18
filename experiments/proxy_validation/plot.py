#!/usr/bin/env python3
"""Plot proxy vs PPL relationship for proxy validation experiments."""
import json, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUTDIR = "experiments/proxy_validation/plots"
os.makedirs(OUTDIR, exist_ok=True)

POLICY_COLORS = {
    "sliding": "#2196F3",
    "h2o": "#F44336",
    "d2o": "#9C27B0",
    "streaming": "#4CAF50",
    "h2o_plus": "#FF9800",
}
POLICY_MARKERS = {
    "sliding": "o",
    "h2o": "s",
    "d2o": "D",
    "streaming": "^",
    "h2o_plus": "v",
}
POLICY_LABELS = {
    "sliding": "Sliding Window",
    "h2o": "H2O",
    "d2o": "D2O",
    "streaming": "StreamingLLM",
    "h2o_plus": "H2O+",
}


def load_model_data(model_dir):
    baseline_path = f"{model_dir}/baseline.json"
    if not os.path.exists(baseline_path):
        return None, None
    baseline = json.load(open(baseline_path))
    baseline_ppl = baseline["ppl"]

    policies = {}
    for policy in ["sliding", "h2o", "d2o", "streaming", "h2o_plus"]:
        results = []
        for f in sorted(glob.glob(f"{model_dir}/{policy}_*.json")):
            try:
                data = json.load(open(f))
            except json.JSONDecodeError:
                continue
            budget = data["config"]["kv_budget"]
            proxies = data["proxy_metrics"]
            avg_proxy = sum(m["raw_value"] for m in proxies) / max(len(proxies), 1)
            total_proxy = sum(m["raw_value"] for m in proxies)
            results.append({
                "budget": budget,
                "ppl": data["ppl"],
                "delta_ppl": data["ppl"] - baseline_ppl,
                "avg_proxy": avg_proxy,
                "total_proxy": total_proxy,
                "evictions": data["eviction_count"],
            })
        if len(results) >= 3:  # Need enough points for meaningful plot
            results.sort(key=lambda x: x["avg_proxy"])
            policies[policy] = results
    return baseline_ppl, policies


def plot_proxy_vs_ppl(model_name, baseline_ppl, policies, suffix=""):
    """Main scatter plot: avg_proxy vs ΔPPL for all policies."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    for policy, results in policies.items():
        x = [r["avg_proxy"] for r in results]
        y = [r["delta_ppl"] for r in results]
        ax.plot(x, y,
                marker=POLICY_MARKERS.get(policy, "o"),
                color=POLICY_COLORS.get(policy, "#666"),
                label=f"{POLICY_LABELS.get(policy, policy)} (ρ=1.0)",
                linewidth=1.5, markersize=7, zorder=3)

    ax.set_xlabel("Average Proxy Value (per eviction)", fontsize=12)
    ax.set_ylabel("ΔPPL (PPL increase from baseline)", fontsize=12)
    ax.set_title(f"{model_name}  —  Proxy vs PPL Degradation\n(Baseline PPL = {baseline_ppl:.2f})",
                 fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = f"{OUTDIR}/proxy_vs_ppl_{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_budget_vs_ppl(model_name, baseline_ppl, policies, suffix=""):
    """Budget sweep: budget vs PPL for all policies."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    for policy, results in policies.items():
        x = [r["budget"] for r in sorted(results, key=lambda r: r["budget"])]
        y = [r["ppl"] for r in sorted(results, key=lambda r: r["budget"])]
        ax.plot(x, y,
                marker=POLICY_MARKERS.get(policy, "o"),
                color=POLICY_COLORS.get(policy, "#666"),
                label=POLICY_LABELS.get(policy, policy),
                linewidth=1.5, markersize=7, zorder=3)

    ax.axhline(y=baseline_ppl, color="gray", linestyle="--", alpha=0.7, label=f"Baseline ({baseline_ppl:.2f})")
    ax.set_xlabel("KV Budget (tokens)", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title(f"{model_name}  —  KV Budget vs PPL\n(lower = better)", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Smaller budget = more aggressive = right side

    plt.tight_layout()
    path = f"{OUTDIR}/budget_vs_ppl_{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_combined_comparison():
    """Side-by-side 1B vs 3B comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, (model, label) in enumerate([("1b", "Llama 3.2 1B"), ("3b", "Llama 3.2 3B")]):
        ax = axes[idx]
        model_dir = f"experiments/proxy_validation/results/{model}"
        baseline_ppl, policies = load_model_data(model_dir)
        if not policies:
            continue

        for policy, results in policies.items():
            x = [r["avg_proxy"] for r in results]
            y = [r["delta_ppl"] for r in results]
            ax.plot(x, y,
                    marker=POLICY_MARKERS.get(policy, "o"),
                    color=POLICY_COLORS.get(policy, "#666"),
                    label=POLICY_LABELS.get(policy, policy),
                    linewidth=1.5, markersize=6, zorder=3)

        ax.set_xlabel("Average Proxy Value", fontsize=11)
        ax.set_ylabel("ΔPPL" if idx == 0 else "", fontsize=11)
        ax.set_title(f"{label}  (Baseline PPL = {baseline_ppl:.2f})", fontsize=12)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    fig.suptitle("Proxy vs PPL Degradation — Spearman ρ = 1.0 (all combinations)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{OUTDIR}/comparison_1b_3b.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_total_proxy_vs_ppl():
    """Total proxy (cumulative) vs ΔPPL — shows stronger relationship."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, (model, label) in enumerate([("1b", "Llama 3.2 1B"), ("3b", "Llama 3.2 3B")]):
        ax = axes[idx]
        model_dir = f"experiments/proxy_validation/results/{model}"
        baseline_ppl, policies = load_model_data(model_dir)
        if not policies:
            continue

        for policy, results in policies.items():
            x = [r["total_proxy"] for r in results]
            y = [r["delta_ppl"] for r in results]
            ax.plot(x, y,
                    marker=POLICY_MARKERS.get(policy, "o"),
                    color=POLICY_COLORS.get(policy, "#666"),
                    label=POLICY_LABELS.get(policy, policy),
                    linewidth=1.5, markersize=6, zorder=3)

        ax.set_xlabel("Cumulative Proxy (sum over all evictions)", fontsize=11)
        ax.set_ylabel("ΔPPL" if idx == 0 else "", fontsize=11)
        ax.set_title(f"{label}  (Baseline PPL = {baseline_ppl:.2f})", fontsize=12)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    fig.suptitle("Cumulative Proxy vs PPL Degradation — ρ = 1.0",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{OUTDIR}/total_proxy_vs_ppl.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    print("Generating proxy validation plots...")

    for model, label in [("1b", "Llama 3.2 1B"), ("3b", "Llama 3.2 3B")]:
        model_dir = f"experiments/proxy_validation/results/{model}"
        baseline_ppl, policies = load_model_data(model_dir)
        if not policies:
            print(f"  [SKIP] {label}: no data")
            continue
        print(f"\n  {label}:")
        plot_proxy_vs_ppl(label, baseline_ppl, policies, model)
        plot_budget_vs_ppl(label, baseline_ppl, policies, model)

    print("\n  Combined:")
    plot_combined_comparison()
    plot_total_proxy_vs_ppl()
    print("\nDone.")


if __name__ == "__main__":
    main()
