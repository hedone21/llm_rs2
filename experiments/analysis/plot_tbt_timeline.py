#!/usr/bin/env python3
"""TBT timeline chart comparing baseline and experiments.

Usage:
    python experiments/analysis/plot_tbt_timeline.py \\
      --baseline experiments/results/B-512.jsonl \\
      --experiments experiments/results/M-C-256-h2o.jsonl experiments/results/M-C-256-sl.jsonl \\
      --output experiments/reports/plots/tbt_comparison.png
"""

import argparse
import os
import sys

# Allow running as script from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quality_metrics import load_jsonl

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    print(
        "Error: matplotlib is required for plotting. "
        "Install with: pip install matplotlib",
        file=sys.stderr,
    )
    sys.exit(1)


def _compute_stats(values):
    """Compute mean and std of a list of values.

    Args:
        values: List of floats.

    Returns:
        Tuple (mean, std).
    """
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance**0.5
    return mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Plot TBT timeline comparing baseline and experiments."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline JSONL file",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Paths to experiment JSONL files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the plot image (e.g., .png)",
    )
    args = parser.parse_args()

    # Validate files
    if not os.path.exists(args.baseline):
        print(f"Error: baseline file not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)

    for exp_path in args.experiments:
        if not os.path.exists(exp_path):
            print(f"Error: experiment file not found: {exp_path}", file=sys.stderr)
            sys.exit(1)

    # Load baseline
    base_tokens, base_summary = load_jsonl(args.baseline)
    base_positions = [t["pos"] for t in base_tokens]
    base_tbt = [t["tbt_ms"] for t in base_tokens]
    base_mean, base_std = _compute_stats(base_tbt)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Baseline as gray band (mean +/- std)
    ax.fill_between(
        base_positions,
        [base_mean - base_std] * len(base_positions),
        [base_mean + base_std] * len(base_positions),
        alpha=0.3,
        color="gray",
        label=f"Baseline (mean={base_mean:.1f}ms)",
    )

    # Color cycle for experiments
    colors = plt.cm.tab10.colors

    for idx, exp_path in enumerate(args.experiments):
        exp_tokens, exp_summary = load_jsonl(exp_path)
        exp_positions = [t["pos"] for t in exp_tokens]
        exp_tbt = [t["tbt_ms"] for t in exp_tokens]

        schedule_name = exp_summary.get("schedule_name", os.path.basename(exp_path))
        eviction_policy = exp_summary.get("eviction_policy", "")
        label = f"{schedule_name}"
        if eviction_policy and eviction_policy != "none":
            label += f" ({eviction_policy})"

        color = colors[idx % len(colors)]

        # Plot TBT line
        ax.plot(
            exp_positions,
            exp_tbt,
            linewidth=0.8,
            alpha=0.8,
            color=color,
            label=label,
        )

        # Mark signal injection points (vertical dashed lines)
        signal_positions = []
        for t in exp_tokens:
            sig = t.get("signal")
            if sig:
                signal_positions.append(t["pos"])
                ax.axvline(
                    x=t["pos"],
                    color=color,
                    linestyle="--",
                    alpha=0.6,
                    linewidth=1.0,
                )

        # Mark eviction events (triangle markers)
        for t in exp_tokens:
            actions = t.get("actions", [])
            has_evict = any("Evict" in str(a) for a in actions)
            if has_evict:
                ax.plot(
                    t["pos"],
                    t["tbt_ms"],
                    marker="^",
                    markersize=10,
                    color=color,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    zorder=5,
                )

    ax.set_xlabel("Token Position", fontsize=12)
    ax.set_ylabel("TBT (ms)", fontsize=12)
    ax.set_title("Time Between Tokens (TBT) Timeline", fontsize=14)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()
