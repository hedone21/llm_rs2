#!/usr/bin/env python3
"""H2O Cache Analysis — 1B vs 3B Model Comparison Plots.

Generates comprehensive visualizations comparing H2O attention score distributions,
quality metrics, and performance across Llama 3.2 1B and 3B models.

Usage:
    python experiments/analysis/plot_h2o_cache_comparison.py
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "experiments" / "results" / "accuracy_bench"
ROUND14_DIR = ROOT / "experiments" / "results" / "round14"
OUT_DIR = ROOT / "results" / "plots" / "h2o_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colors ───────────────────────────────────────────────────────────────────
COLORS = {
    "BASE": "#2196F3",
    "H2O": "#F44336",
    "H2O+": "#FF9800",
    "SLIDE": "#4CAF50",
}
MODEL_COLORS = {"1B": "#1976D2", "3B": "#E64A19"}


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_scores(filepath):
    """Load position,score CSV into numpy arrays."""
    positions, scores = [], []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions.append(int(row["position"]))
            scores.append(float(row["score"]))
    return np.array(positions), np.array(scores)


def load_jsonl(filepath):
    """Load JSONL → (token_records, summary)."""
    tokens, summary = [], None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            if rec.get("_summary"):
                summary = rec
            else:
                tokens.append(rec)
    return tokens, summary


def compute_emr(base_tokens, exp_tokens):
    n = min(len(base_tokens), len(exp_tokens))
    if n == 0:
        return 1.0
    return sum(1 for i in range(n) if base_tokens[i]["token_id"] == exp_tokens[i]["token_id"]) / n


def compute_fdt(base_tokens, exp_tokens):
    n = min(len(base_tokens), len(exp_tokens))
    for i in range(n):
        if base_tokens[i]["token_id"] != exp_tokens[i]["token_id"]:
            return i
    return n


def compute_topk_overlap(base_tokens, exp_tokens, k=10):
    n = min(len(base_tokens), len(exp_tokens))
    overlaps = []
    for i in range(n):
        b_ids = set(e[0] for e in base_tokens[i].get("top_logits", [])[:k])
        e_ids = set(e[0] for e in exp_tokens[i].get("top_logits", [])[:k])
        if not b_ids and not e_ids:
            overlaps.append(1.0)
        elif not b_ids or not e_ids:
            overlaps.append(0.0)
        else:
            overlaps.append(len(b_ids & e_ids) / k)
    return overlaps


# ── Plot 1: Score Distribution Histogram (1B vs 3B) ─────────────────────────
def plot_score_histogram():
    """Attention score distribution histogram — 1B vs 3B side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("H2O Attention Score Distribution — 1B vs 3B", fontsize=16, fontweight="bold")

    for col, prompt in enumerate(["PPL01", "PPL03"]):
        for row, model in enumerate(["1B", "3B"]):
            ax = axes[row, col]
            score_file = BENCH_DIR / f"{model}-H2O-{prompt}.scores.csv"
            if not score_file.exists():
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            pos, scores = load_scores(score_file)

            # Separate BOS token from rest
            bos_score = scores[0]
            rest_scores = scores[1:]

            # Histogram (excluding BOS)
            ax.hist(rest_scores, bins=50, color=MODEL_COLORS[model], alpha=0.7,
                    edgecolor="white", linewidth=0.5, label=f"Tokens 1-{len(scores)-1}")

            # Mark BOS
            ax.axvline(bos_score, color="red", linestyle="--", linewidth=2,
                       label=f"BOS = {bos_score:.1f}")

            ax.set_title(f"{model} — {prompt}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Attention Score (accumulated)")
            ax.set_ylabel("Token Count")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)

            # Annotate stats
            stats_text = (f"mean={rest_scores.mean():.3f}\n"
                          f"std={rest_scores.std():.3f}\n"
                          f"max={rest_scores.max():.3f}\n"
                          f"BOS/mean={bos_score/rest_scores.mean():.0f}x")
            ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    out = OUT_DIR / "01_score_distribution_histogram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1] {out}")


# ── Plot 2: Score Distribution Log Scale ────────────────────────────────────
def plot_score_log_scale():
    """Score distribution with log-y scale to reveal BOS dominance."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("H2O Score Distribution (Log Scale) — BOS Token Dominance", fontsize=15, fontweight="bold")

    for col, prompt in enumerate(["PPL01", "PPL03"]):
        ax = axes[col]
        for model in ["1B", "3B"]:
            score_file = BENCH_DIR / f"{model}-H2O-{prompt}.scores.csv"
            if not score_file.exists():
                continue
            pos, scores = load_scores(score_file)
            ax.semilogy(pos, scores, "o-", markersize=2, linewidth=0.8,
                        color=MODEL_COLORS[model], alpha=0.7, label=f"{model}")

        ax.set_title(f"{prompt}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Attention Score (log scale)")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "02_score_log_scale.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2] {out}")


# ── Plot 3: Score CDF (Cumulative Distribution) ─────────────────────────────
def plot_score_cdf():
    """Cumulative distribution of scores — shows concentration."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Attention Score CDF — Score Concentration Comparison", fontsize=15, fontweight="bold")

    for col, prompt in enumerate(["PPL01", "PPL03"]):
        ax = axes[col]
        for model in ["1B", "3B"]:
            score_file = BENCH_DIR / f"{model}-H2O-{prompt}.scores.csv"
            if not score_file.exists():
                continue
            _, scores = load_scores(score_file)
            sorted_scores = np.sort(scores)
            cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
            ax.plot(sorted_scores, cdf, linewidth=2, color=MODEL_COLORS[model],
                    label=f"{model} (n={len(scores)})")

        ax.set_title(f"{prompt} — Score CDF", fontsize=13, fontweight="bold")
        ax.set_xlabel("Attention Score")
        ax.set_ylabel("Cumulative Fraction")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    plt.tight_layout()
    out = OUT_DIR / "03_score_cdf.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3] {out}")


# ── Plot 4: Quality Metrics Bar Chart (EMR, FDT, Top-K) ─────────────────────
def plot_quality_metrics():
    """EMR, FDT, and mean Top-K Overlap for all policies × both models."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Quality Metrics — H2O vs H2O+ vs Sliding Window (1B & 3B)",
                 fontsize=15, fontweight="bold")

    policies = ["H2O", "H2OP", "SLIDE"]
    policy_labels = ["H2O", "H2O+", "Sliding"]

    for row, prompt in enumerate(["PPL01", "PPL03"]):
        # Load baselines
        baselines = {}
        for model in ["1B", "3B"]:
            base_file = BENCH_DIR / f"{model}-BASE-{prompt}.jsonl"
            if base_file.exists():
                baselines[model], _ = load_jsonl(base_file)

        # Compute metrics for each policy
        emr_data = {m: [] for m in ["1B", "3B"]}
        fdt_data = {m: [] for m in ["1B", "3B"]}
        topk_data = {m: [] for m in ["1B", "3B"]}

        for policy in policies:
            for model in ["1B", "3B"]:
                exp_file = BENCH_DIR / f"{model}-{policy}-{prompt}.jsonl"
                if not exp_file.exists() or model not in baselines:
                    emr_data[model].append(0)
                    fdt_data[model].append(0)
                    topk_data[model].append(0)
                    continue
                exp_tokens, _ = load_jsonl(exp_file)
                base = baselines[model]

                emr_data[model].append(compute_emr(base, exp_tokens) * 100)
                fdt_data[model].append(compute_fdt(base, exp_tokens))
                overlaps = compute_topk_overlap(base, exp_tokens)
                topk_data[model].append(np.mean(overlaps) * 100 if overlaps else 0)

        x = np.arange(len(policy_labels))
        width = 0.35

        # EMR
        ax = axes[row, 0]
        ax.bar(x - width/2, emr_data["1B"], width, label="1B", color=MODEL_COLORS["1B"], alpha=0.85)
        ax.bar(x + width/2, emr_data["3B"], width, label="3B", color=MODEL_COLORS["3B"], alpha=0.85)
        for i, (v1, v3) in enumerate(zip(emr_data["1B"], emr_data["3B"])):
            ax.text(i - width/2, v1 + 0.5, f"{v1:.1f}%", ha="center", fontsize=8, fontweight="bold")
            ax.text(i + width/2, v3 + 0.5, f"{v3:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax.set_ylabel("EMR (%)")
        ax.set_title(f"{prompt} — Exact Match Rate", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(policy_labels)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # FDT
        ax = axes[row, 1]
        ax.bar(x - width/2, fdt_data["1B"], width, label="1B", color=MODEL_COLORS["1B"], alpha=0.85)
        ax.bar(x + width/2, fdt_data["3B"], width, label="3B", color=MODEL_COLORS["3B"], alpha=0.85)
        for i, (v1, v3) in enumerate(zip(fdt_data["1B"], fdt_data["3B"])):
            ax.text(i - width/2, v1 + 0.5, f"{v1}", ha="center", fontsize=8, fontweight="bold")
            ax.text(i + width/2, v3 + 0.5, f"{v3}", ha="center", fontsize=8, fontweight="bold")
        ax.set_ylabel("FDT (token position)")
        ax.set_title(f"{prompt} — First Divergent Token", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(policy_labels)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Top-K Overlap
        ax = axes[row, 2]
        ax.bar(x - width/2, topk_data["1B"], width, label="1B", color=MODEL_COLORS["1B"], alpha=0.85)
        ax.bar(x + width/2, topk_data["3B"], width, label="3B", color=MODEL_COLORS["3B"], alpha=0.85)
        for i, (v1, v3) in enumerate(zip(topk_data["1B"], topk_data["3B"])):
            ax.text(i - width/2, v1 + 0.3, f"{v1:.1f}%", ha="center", fontsize=8, fontweight="bold")
            ax.text(i + width/2, v3 + 0.3, f"{v3:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax.set_ylabel("Top-10 Overlap (%)")
        ax.set_title(f"{prompt} — Mean Top-K Overlap", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(policy_labels)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "04_quality_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4] {out}")


# ── Plot 5: Top-K Overlap Timeline ──────────────────────────────────────────
def plot_topk_timeline():
    """Per-token Top-K overlap over decode positions."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle("Top-10 Overlap Timeline — Token-by-Token Degradation",
                 fontsize=15, fontweight="bold")

    for col, prompt in enumerate(["PPL01", "PPL03"]):
        for row, model in enumerate(["1B", "3B"]):
            ax = axes[row, col]
            base_file = BENCH_DIR / f"{model}-BASE-{prompt}.jsonl"
            if not base_file.exists():
                continue
            base_tokens, _ = load_jsonl(base_file)

            for policy, label, color in [("H2O", "H2O", COLORS["H2O"]),
                                          ("H2OP", "H2O+", COLORS["H2O+"]),
                                          ("SLIDE", "Sliding", COLORS["SLIDE"])]:
                exp_file = BENCH_DIR / f"{model}-{policy}-{prompt}.jsonl"
                if not exp_file.exists():
                    continue
                exp_tokens, _ = load_jsonl(exp_file)
                overlaps = compute_topk_overlap(base_tokens, exp_tokens)

                # Smoothed with rolling window
                window = 10
                if len(overlaps) >= window:
                    smoothed = np.convolve(overlaps, np.ones(window)/window, mode="valid")
                    ax.plot(range(window-1, len(overlaps)), smoothed, linewidth=1.5,
                            color=color, label=f"{label} (avg={np.mean(overlaps):.2f})", alpha=0.85)
                else:
                    ax.plot(overlaps, linewidth=1, color=color, label=label)

            ax.set_title(f"{model} — {prompt}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Decode Position")
            ax.set_ylabel("Top-10 Overlap")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "05_topk_timeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5] {out}")


# ── Plot 6: Performance (TBT) Comparison ────────────────────────────────────
def plot_performance():
    """Average TBT comparison across policies and models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Performance Impact — Average Time-Between-Tokens (ms)",
                 fontsize=15, fontweight="bold")

    policies = ["BASE", "H2O", "H2OP", "SLIDE"]
    policy_labels = ["Baseline", "H2O", "H2O+", "Sliding"]
    colors = [COLORS["BASE"], COLORS["H2O"], COLORS["H2O+"], COLORS["SLIDE"]]

    for col, prompt in enumerate(["PPL01", "PPL03"]):
        ax = axes[col]

        tbt_1b, tbt_3b = [], []
        for policy in policies:
            for model, tbt_list in [("1B", tbt_1b), ("3B", tbt_3b)]:
                fpath = BENCH_DIR / f"{model}-{policy}-{prompt}.jsonl"
                if fpath.exists():
                    _, summary = load_jsonl(fpath)
                    tbt_list.append(summary.get("avg_tbt_ms", 0))
                else:
                    tbt_list.append(0)

        x = np.arange(len(policy_labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, tbt_1b, width, label="1B", color=MODEL_COLORS["1B"], alpha=0.85)
        bars3 = ax.bar(x + width/2, tbt_3b, width, label="3B", color=MODEL_COLORS["3B"], alpha=0.85)

        for i, (v1, v3) in enumerate(zip(tbt_1b, tbt_3b)):
            ax.text(i - width/2, v1 + 1, f"{v1:.1f}", ha="center", fontsize=8, fontweight="bold")
            ax.text(i + width/2, v3 + 1, f"{v3:.1f}", ha="center", fontsize=8, fontweight="bold")

        ax.set_title(f"{prompt}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Avg TBT (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels(policy_labels)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "06_performance_tbt.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [6] {out}")


# ── Plot 7: Score Heatmap (Position × Score Rank) ───────────────────────────
def plot_score_rank_comparison():
    """Score rank visualization — which tokens would H2O select as Heavy Hitters."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("H2O Score Rank — Token Importance Profiles (1B vs 3B)",
                 fontsize=15, fontweight="bold")

    for col, prompt in enumerate(["PPL01", "PPL03"]):
        for row, model in enumerate(["1B", "3B"]):
            ax = axes[row, col]
            score_file = BENCH_DIR / f"{model}-H2O-{prompt}.scores.csv"
            if not score_file.exists():
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            pos, scores = load_scores(score_file)

            # Rank tokens by score (descending)
            sorted_indices = np.argsort(scores)[::-1]
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(len(scores))

            # Color by rank
            scatter = ax.scatter(pos, scores, c=ranks, cmap="RdYlGn_r",
                                 s=15, alpha=0.7, edgecolors="none")
            plt.colorbar(scatter, ax=ax, label="Rank (0=highest)")

            # Highlight top-10 HH tokens
            top10_idx = sorted_indices[:10]
            ax.scatter(pos[top10_idx], scores[top10_idx], s=60, facecolors="none",
                       edgecolors="red", linewidths=2, label="Top-10 HH")

            ax.set_title(f"{model} — {prompt}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Token Position")
            ax.set_ylabel("Attention Score")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = OUT_DIR / "07_score_rank.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [7] {out}")


# ── Plot 8: 1B vs 3B Score Scatter ──────────────────────────────────────────
def plot_1b_vs_3b_scatter():
    """Scatter plot comparing per-position scores between 1B and 3B."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Score Correlation — 1B vs 3B at Same Token Positions",
                 fontsize=15, fontweight="bold")

    for col, prompt in enumerate(["PPL01", "PPL03"]):
        ax = axes[col]

        f1b = BENCH_DIR / f"1B-H2O-{prompt}.scores.csv"
        f3b = BENCH_DIR / f"3B-H2O-{prompt}.scores.csv"
        if not f1b.exists() or not f3b.exists():
            continue

        _, scores_1b = load_scores(f1b)
        _, scores_3b = load_scores(f3b)
        n = min(len(scores_1b), len(scores_3b))
        s1, s3 = scores_1b[:n], scores_3b[:n]

        # Exclude BOS for cleaner view
        s1_no_bos, s3_no_bos = s1[1:], s3[1:]

        ax.scatter(s1_no_bos, s3_no_bos, s=12, alpha=0.5, c="purple", edgecolors="none")

        # BOS point
        ax.scatter([s1[0]], [s3[0]], s=100, c="red", marker="*", zorder=5, label=f"BOS ({s1[0]:.1f}, {s3[0]:.1f})")

        # Correlation
        if len(s1_no_bos) > 2:
            corr = np.corrcoef(s1_no_bos, s3_no_bos)[0, 1]
            ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
                    fontsize=11, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        ax.set_title(f"{prompt}", fontsize=13, fontweight="bold")
        ax.set_xlabel("1B Score")
        ax.set_ylabel("3B Score")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "08_1b_vs_3b_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [8] {out}")


# ── Plot 9: H2O+ vs H2O Score Comparison ────────────────────────────────────
def plot_h2o_vs_h2oplus():
    """Compare H2O vs H2O+ (per-head) score distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("H2O vs H2O+ Score Distribution Comparison",
                 fontsize=15, fontweight="bold")

    for col, prompt in enumerate(["PPL01", "PPL03"]):
        for row, model in enumerate(["1B", "3B"]):
            ax = axes[row, col]

            h2o_file = BENCH_DIR / f"{model}-H2O-{prompt}.scores.csv"
            h2op_file = BENCH_DIR / f"{model}-H2OP-{prompt}.scores.csv"

            if not h2o_file.exists() or not h2op_file.exists():
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            _, h2o_scores = load_scores(h2o_file)
            _, h2op_scores = load_scores(h2op_file)

            # Exclude BOS for cleaner histogram
            h2o_rest = h2o_scores[1:]
            h2op_rest = h2op_scores[1:]

            bins = np.linspace(0, max(h2o_rest.max(), h2op_rest.max()), 40)
            ax.hist(h2o_rest, bins=bins, alpha=0.6, color=COLORS["H2O"],
                    edgecolor="white", linewidth=0.5, label=f"H2O (BOS={h2o_scores[0]:.1f})")
            ax.hist(h2op_rest, bins=bins, alpha=0.6, color=COLORS["H2O+"],
                    edgecolor="white", linewidth=0.5, label=f"H2O+ (BOS={h2op_scores[0]:.1f})")

            ax.set_title(f"{model} — {prompt}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Attention Score")
            ax.set_ylabel("Token Count")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "09_h2o_vs_h2oplus.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [9] {out}")


# ── Plot 10: Summary Dashboard ──────────────────────────────────────────────
def plot_summary_dashboard():
    """Single-page dashboard with key findings."""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("H2O Cache Eviction — 1B vs 3B Comprehensive Summary",
                 fontsize=18, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.35)

    prompt = "PPL01"

    # ── (0,0-1) Score distribution 1B ──
    ax = fig.add_subplot(gs[0, 0:2])
    f1b = BENCH_DIR / f"1B-H2O-{prompt}.scores.csv"
    if f1b.exists():
        pos, scores = load_scores(f1b)
        ax.semilogy(pos, scores, "o-", markersize=2, linewidth=0.8,
                    color=MODEL_COLORS["1B"], alpha=0.8)
        ax.set_title("1B — Score Distribution (log)", fontweight="bold", fontsize=11)
        ax.set_xlabel("Position")
        ax.set_ylabel("Score (log)")
        ax.grid(True, alpha=0.3)

        # Annotate BOS
        ax.annotate(f"BOS={scores[0]:.1f}", xy=(0, scores[0]),
                    xytext=(20, scores[0]), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="red"),
                    color="red", fontweight="bold")

    # ── (0,2-3) Score distribution 3B ──
    ax = fig.add_subplot(gs[0, 2:4])
    f3b = BENCH_DIR / f"3B-H2O-{prompt}.scores.csv"
    if f3b.exists():
        pos, scores = load_scores(f3b)
        ax.semilogy(pos, scores, "o-", markersize=2, linewidth=0.8,
                    color=MODEL_COLORS["3B"], alpha=0.8)
        ax.set_title("3B — Score Distribution (log)", fontweight="bold", fontsize=11)
        ax.set_xlabel("Position")
        ax.set_ylabel("Score (log)")
        ax.grid(True, alpha=0.3)

        ax.annotate(f"BOS={scores[0]:.1f}", xy=(0, scores[0]),
                    xytext=(20, scores[0]), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="red"),
                    color="red", fontweight="bold")

    # ── (1,0-1) EMR comparison ──
    ax = fig.add_subplot(gs[1, 0:2])
    policies = ["H2O", "H2OP", "SLIDE"]
    policy_labels = ["H2O", "H2O+", "Sliding"]
    emr_1b, emr_3b = [], []

    for p in policies:
        for model, emr_list in [("1B", emr_1b), ("3B", emr_3b)]:
            base_f = BENCH_DIR / f"{model}-BASE-{prompt}.jsonl"
            exp_f = BENCH_DIR / f"{model}-{p}-{prompt}.jsonl"
            if base_f.exists() and exp_f.exists():
                bt, _ = load_jsonl(base_f)
                et, _ = load_jsonl(exp_f)
                emr_list.append(compute_emr(bt, et) * 100)
            else:
                emr_list.append(0)

    x = np.arange(len(policy_labels))
    width = 0.3
    ax.bar(x - width/2, emr_1b, width, label="1B", color=MODEL_COLORS["1B"], alpha=0.85)
    ax.bar(x + width/2, emr_3b, width, label="3B", color=MODEL_COLORS["3B"], alpha=0.85)
    for i in range(len(policy_labels)):
        ax.text(i - width/2, emr_1b[i] + 0.3, f"{emr_1b[i]:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax.text(i + width/2, emr_3b[i] + 0.3, f"{emr_3b[i]:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("EMR (%)")
    ax.set_title(f"EMR — {prompt}", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(policy_labels)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── (1,2-3) TBT comparison ──
    ax = fig.add_subplot(gs[1, 2:4])
    all_policies = ["BASE", "H2O", "H2OP", "SLIDE"]
    all_labels = ["Base", "H2O", "H2O+", "Slide"]
    tbt_1b, tbt_3b = [], []

    for p in all_policies:
        for model, tbt_list in [("1B", tbt_1b), ("3B", tbt_3b)]:
            fpath = BENCH_DIR / f"{model}-{p}-{prompt}.jsonl"
            if fpath.exists():
                _, s = load_jsonl(fpath)
                tbt_list.append(s.get("avg_tbt_ms", 0))
            else:
                tbt_list.append(0)

    x = np.arange(len(all_labels))
    ax.bar(x - width/2, tbt_1b, width, label="1B", color=MODEL_COLORS["1B"], alpha=0.85)
    ax.bar(x + width/2, tbt_3b, width, label="3B", color=MODEL_COLORS["3B"], alpha=0.85)
    for i in range(len(all_labels)):
        ax.text(i - width/2, tbt_1b[i] + 0.5, f"{tbt_1b[i]:.0f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(i + width/2, tbt_3b[i] + 0.5, f"{tbt_3b[i]:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("Avg TBT (ms)")
    ax.set_title(f"Performance — {prompt}", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── (2,0-1) Score stats table ──
    ax = fig.add_subplot(gs[2, 0:2])
    ax.axis("off")
    table_data = []
    for model in ["1B", "3B"]:
        for p in ["PPL01", "PPL03"]:
            sf = BENCH_DIR / f"{model}-H2O-{p}.scores.csv"
            if sf.exists():
                _, scores = load_scores(sf)
                rest = scores[1:]
                table_data.append([
                    f"{model}", p,
                    f"{scores[0]:.1f}", f"{rest.mean():.4f}",
                    f"{rest.std():.4f}", f"{rest.max():.3f}",
                    f"{scores[0]/rest.mean():.0f}x"
                ])
    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=["Model", "Prompt", "BOS", "Mean", "Std", "Max(non-BOS)", "BOS/Mean"],
            loc="center", cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#E0E0E0")
                cell.set_text_props(fontweight="bold")
    ax.set_title("Score Statistics Summary", fontweight="bold", fontsize=11, pad=15)

    # ── (2,2-3) Key findings ──
    ax = fig.add_subplot(gs[2, 2:4])
    ax.axis("off")

    findings = [
        "Key Findings:",
        "",
        "1. BOS dominance: 1B BOS/mean ratio >> 3B",
        "   1B: BOS token overwhelmingly high score",
        "",
        "2. Score distribution: 3B is more uniform",
        "   3B: more uniform => HH selection viable",
        "",
        "3. Sliding Window wins for 1B at all metrics",
        "   H2O < Sliding for 1B model",
        "",
        "4. H2O overhead: ~18% TBT increase",
        "   Due to attention score computation",
        "",
        "5. 3B may benefit more from H2O",
        "   More uniform scores => HH selection useful",
    ]
    ax.text(0.05, 0.95, "\n".join(findings), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4", alpha=0.9))

    out = OUT_DIR / "10_summary_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" [10] {out}")


# ── Plot 11: Round 14 Keep-Ratio Sweep (1B only) ────────────────────────────
def plot_round14_keepratio():
    """H2O keep-ratio sweep (Round 14) — EMR across kr=0.0~0.9."""
    if not ROUND14_DIR.exists():
        print("  [11] Skipped: no Round 14 data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Round 14 — H2O Keep-Ratio Sweep (1B Model)",
                 fontsize=15, fontweight="bold")

    for col, prompt in enumerate(["PPL01", "PPL03"]):
        ax = axes[col]
        base_file = ROUND14_DIR / f"BASE-{prompt}.jsonl"
        if not base_file.exists():
            continue
        base_tokens, _ = load_jsonl(base_file)

        kr_values = ["00", "10", "20", "30", "50", "70", "90"]
        kr_labels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

        for prefix, label, color, marker in [
            ("H2O", "H2O", COLORS["H2O"], "o"),
            ("H2OP", "H2O+", COLORS["H2O+"], "s"),
        ]:
            emrs = []
            valid_kr = []
            for kr, kr_label in zip(kr_values, kr_labels):
                f = ROUND14_DIR / f"{prefix}-{kr}-{prompt}.jsonl"
                if f.exists():
                    et, _ = load_jsonl(f)
                    emrs.append(compute_emr(base_tokens, et) * 100)
                    valid_kr.append(kr_label)

            if valid_kr:
                ax.plot(valid_kr, emrs, f"{marker}-", color=color, linewidth=2,
                        markersize=8, label=label)
                for x, y in zip(valid_kr, emrs):
                    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=8)

        # Sliding baseline
        sl_file = ROUND14_DIR / f"SL-{prompt}.jsonl"
        if sl_file.exists():
            et, _ = load_jsonl(sl_file)
            sl_emr = compute_emr(base_tokens, et) * 100
            ax.axhline(sl_emr, color=COLORS["SLIDE"], linestyle="--", linewidth=2,
                       label=f"Sliding ({sl_emr:.1f}%)")

        ax.set_title(f"{prompt}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Keep Ratio (kr)")
        ax.set_ylabel("EMR (%)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 0.95)

    plt.tight_layout()
    out = OUT_DIR / "11_round14_keepratio.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" [11] {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Output directory: {OUT_DIR}\n")

    print("Generating H2O cache analysis plots...")
    plot_score_histogram()
    plot_score_log_scale()
    plot_score_cdf()
    plot_quality_metrics()
    plot_topk_timeline()
    plot_performance()
    plot_score_rank_comparison()
    plot_1b_vs_3b_scatter()
    plot_h2o_vs_h2oplus()
    plot_summary_dashboard()
    plot_round14_keepratio()

    print(f"\nDone! {len(list(OUT_DIR.glob('*.png')))} plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
