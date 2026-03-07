#!/usr/bin/env python3
"""RSS timeline chart for experiment results.

Usage:
    python experiments/analysis/plot_rss_timeline.py \\
      --experiments experiments/results/M-C-256-h2o.jsonl experiments/results/R-C-512-h2o.jsonl \\
      --output experiments/reports/plots/rss_comparison.png
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


def main():
    parser = argparse.ArgumentParser(
        description="Plot RSS timeline for experiment results."
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
    for exp_path in args.experiments:
        if not os.path.exists(exp_path):
            print(f"Error: experiment file not found: {exp_path}", file=sys.stderr)
            sys.exit(1)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Color cycle
    colors = plt.cm.tab10.colors

    for idx, exp_path in enumerate(args.experiments):
        exp_tokens, exp_summary = load_jsonl(exp_path)

        # Extract RSS values (skip tokens where sys is null)
        positions = []
        rss_values = []
        for t in exp_tokens:
            sys_data = t.get("sys")
            if sys_data is not None:
                rss = sys_data.get("rss_mb")
                if rss is not None:
                    positions.append(t["pos"])
                    rss_values.append(rss)

        if not positions:
            print(f"Warning: no RSS data in {exp_path}", file=sys.stderr)
            continue

        schedule_name = exp_summary.get("schedule_name", os.path.basename(exp_path))
        eviction_policy = exp_summary.get("eviction_policy", "")
        label = f"{schedule_name}"
        if eviction_policy and eviction_policy != "none":
            label += f" ({eviction_policy})"

        color = colors[idx % len(colors)]

        # Plot RSS line
        ax.plot(
            positions,
            rss_values,
            linewidth=1.2,
            alpha=0.8,
            color=color,
            label=label,
        )

        # Mark signal injection points (vertical dashed lines)
        for t in exp_tokens:
            sig = t.get("signal")
            if sig:
                ax.axvline(
                    x=t["pos"],
                    color=color,
                    linestyle="--",
                    alpha=0.6,
                    linewidth=1.0,
                )

    ax.set_xlabel("Token Position", fontsize=12)
    ax.set_ylabel("RSS (MB)", fontsize=12)
    ax.set_title("Process RSS Memory Timeline", fontsize=14)
    ax.legend(loc="best", fontsize=9)
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
