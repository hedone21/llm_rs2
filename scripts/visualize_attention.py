#!/usr/bin/env python3
"""Visualize attention score profiles from llm.rs inference profiler.

Reads the JSON output from `generate --profile` and produces:
  1. Attention score heatmap (importance over time)
  2. Partition evolution chart (prefix / HH / recent)
  3. Eviction detail (before/after score comparison)
  4. Per-head heatmaps (optional, with --per-head)

Usage:
  python scripts/visualize_attention.py results/profile/profile_*.json
  python scripts/visualize_attention.py results/profile/profile_*.json --per-head
  python scripts/visualize_attention.py results/profile/profile_*.json --output plots/
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def load_profile(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def plot_attention_heatmap(data: dict, output_dir: str, filename: str = "attention_heatmap.png"):
    """X=decode step, Y=cache position, color=importance score."""
    scores = data.get("scores", {})
    snapshots = scores.get("snapshots", [])
    if not snapshots:
        print("  [skip] No score snapshots found.")
        return

    # Build 2D matrix: [max_cache_len x num_steps]
    steps = [s["step"] for s in snapshots]
    max_cache_len = max(s["cache_len"] for s in snapshots)

    matrix = np.full((max_cache_len, len(snapshots)), np.nan)
    for col, snap in enumerate(snapshots):
        imp = snap["importance"]
        matrix[:len(imp), col] = imp

    # Replace 0 and NaN with small value for log scale
    plot_matrix = np.where(np.isnan(matrix), 0, matrix)
    vmin = np.min(plot_matrix[plot_matrix > 0]) if np.any(plot_matrix > 0) else 1e-6
    plot_matrix = np.where(plot_matrix <= 0, vmin * 0.1, plot_matrix)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(
        plot_matrix,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        norm=LogNorm(vmin=vmin * 0.1, vmax=np.max(plot_matrix)),
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Importance Score (log)")

    # Mark eviction events
    evictions = scores.get("evictions", [])
    for ev in evictions:
        ev_step = ev["step"]
        # Find column index for this step
        col_idx = None
        for i, s in enumerate(steps):
            if s >= ev_step:
                col_idx = i
                break
        if col_idx is not None:
            ax.axvline(x=col_idx, color="red", linestyle="--", alpha=0.7, linewidth=1)
            ax.text(
                col_idx, max_cache_len * 0.95,
                f"evict\n-{ev['evicted_count']}",
                color="red", fontsize=7, ha="center", va="top",
            )

    # X-axis: step numbers (subsample if too many)
    n_ticks = min(20, len(steps))
    tick_indices = np.linspace(0, len(steps) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([steps[i] for i in tick_indices])

    ax.set_xlabel("Decode Step")
    ax.set_ylabel("Cache Position")
    ax.set_title("H2O Attention Score Heatmap")

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_partition_evolution(data: dict, output_dir: str):
    """Stacked area chart showing prefix/HH/recent partition sizes over time."""
    evictions = data.get("scores", {}).get("evictions", [])
    if not evictions:
        print("  [skip] No eviction events for partition chart.")
        return

    steps = [e["step"] for e in evictions]
    prefix = [e["partition"]["prefix_end"] for e in evictions]
    after_lens = [e["after_len"] for e in evictions]
    before_lens = [e["before_len"] for e in evictions]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(steps, before_lens, "o--", color="gray", label="Before eviction", alpha=0.5)
    ax.plot(steps, after_lens, "o-", color="blue", label="After eviction")
    ax.fill_between(steps, 0, prefix, alpha=0.3, color="green", label="Protected Prefix")

    for i, ev in enumerate(evictions):
        ax.annotate(
            f"-{ev['evicted_count']}",
            (steps[i], after_lens[i]),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=7,
            ha="center",
        )

    ax.set_xlabel("Decode Step")
    ax.set_ylabel("Cache Size (tokens)")
    ax.set_title("Cache Size & Eviction Events")
    ax.legend()

    path = os.path.join(output_dir, "partition_evolution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_eviction_detail(data: dict, output_dir: str):
    """Show score distributions before eviction events."""
    evictions = data.get("scores", {}).get("evictions", [])
    snapshots = data.get("scores", {}).get("snapshots", [])
    if not evictions or not snapshots:
        print("  [skip] No eviction/snapshot data for detail chart.")
        return

    # Find snapshots closest to each eviction
    snap_by_step = {s["step"]: s for s in snapshots}

    n_evictions = min(len(evictions), 4)  # max 4 subplots
    if n_evictions == 0:
        return

    fig, axes = plt.subplots(1, n_evictions, figsize=(5 * n_evictions, 4), squeeze=False)

    for idx, ev in enumerate(evictions[:n_evictions]):
        ax = axes[0][idx]
        ev_step = ev["step"]

        # Find nearest snapshot before eviction
        nearest = None
        for s in snapshots:
            if s["step"] <= ev_step:
                nearest = s

        if nearest is None:
            ax.set_title(f"Step {ev_step}\n(no snapshot)")
            continue

        scores = nearest["importance"]
        positions = np.arange(len(scores))

        prefix_end = ev["partition"]["prefix_end"]

        # Color by partition
        colors = []
        for i in range(len(scores)):
            if i < prefix_end:
                colors.append("green")
            else:
                colors.append("steelblue")

        ax.bar(positions, scores, color=colors, width=1.0, edgecolor="none")
        ax.axvline(x=prefix_end, color="green", linestyle=":", alpha=0.7, label="Prefix")
        ax.set_title(f"Step {ev_step}: -{ev['evicted_count']} tokens")
        ax.set_xlabel("Cache Position")
        ax.set_ylabel("Importance")
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Score Distribution at Eviction Events")
    path = os.path.join(output_dir, "eviction_detail.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_latency(data: dict, output_dir: str):
    """Per-token forward latency over time, colored by cache length."""
    latency = data.get("latency", {})
    records = latency.get("records", [])
    if not records:
        print("  [skip] No latency records.")
        return

    steps = [r["step"] for r in records]
    forward_us = [r["forward_us"] for r in records]
    cache_lens = [r["cache_len"] for r in records]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color1 = "steelblue"
    ax1.plot(steps, [f / 1000.0 for f in forward_us], color=color1, alpha=0.7, linewidth=0.8)
    ax1.set_xlabel("Decode Step")
    ax1.set_ylabel("Forward Latency (ms)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "coral"
    ax2.plot(steps, cache_lens, color=color2, alpha=0.5, linewidth=0.8)
    ax2.set_ylabel("Cache Length", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Forward Latency vs Cache Length")

    path = os.path.join(output_dir, "latency_vs_cache.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_head_heatmaps(data: dict, output_dir: str):
    """Per-KV-head attention score heatmaps."""
    snapshots = data.get("scores", {}).get("snapshots", [])
    if not snapshots:
        print("  [skip] No snapshots for per-head heatmaps.")
        return

    # Check if head_importance data exists
    has_head = any("head_importance" in s and s.get("head_importance") for s in snapshots)
    if not has_head:
        print("  [skip] No per-head importance data (run with --profile-per-head).")
        return

    n_kv_heads = snapshots[0].get("n_kv_heads", 0)
    if n_kv_heads == 0:
        print("  [skip] n_kv_heads = 0.")
        return

    max_cache_len = max(s["cache_len"] for s in snapshots)
    n_steps = len(snapshots)

    for h in range(n_kv_heads):
        matrix = np.full((max_cache_len, n_steps), np.nan)
        for col, snap in enumerate(snapshots):
            hi = snap.get("head_importance", [])
            cache_len = snap["cache_len"]
            if hi:
                start = h * cache_len
                end = start + cache_len
                if end <= len(hi):
                    matrix[:cache_len, col] = hi[start:end]

        plot_matrix = np.where(np.isnan(matrix), 0, matrix)
        vmin = np.min(plot_matrix[plot_matrix > 0]) if np.any(plot_matrix > 0) else 1e-6
        plot_matrix = np.where(plot_matrix <= 0, vmin * 0.1, plot_matrix)

        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(
            plot_matrix,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            norm=LogNorm(vmin=vmin * 0.1, vmax=np.max(plot_matrix)),
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="Importance")

        steps = [s["step"] for s in snapshots]
        n_ticks = min(20, len(steps))
        tick_indices = np.linspace(0, len(steps) - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([steps[i] for i in tick_indices])

        ax.set_xlabel("Decode Step")
        ax.set_ylabel("Cache Position")
        ax.set_title(f"KV Head {h} — Attention Score Heatmap")

        path = os.path.join(output_dir, f"head_heatmap_{h}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize H2O attention score profiles from llm.rs profiler."
    )
    parser.add_argument("file", help="Path to the profile JSON file")
    parser.add_argument("--output", default=None, help="Output directory (default: same as input)")
    parser.add_argument("--per-head", action="store_true", help="Generate per-KV-head heatmaps")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: {args.file} not found.")
        sys.exit(1)

    data = load_profile(args.file)

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.dirname(args.file) or "."
    os.makedirs(output_dir, exist_ok=True)

    meta = data.get("metadata", {})
    print(f"Profile: {meta.get('model', '?')} | {meta.get('backend', '?')} | "
          f"eviction={meta.get('eviction_policy', '?')} | "
          f"tokens={meta.get('generated_tokens', '?')}")

    print("\nGenerating plots...")
    plot_attention_heatmap(data, output_dir)
    plot_partition_evolution(data, output_dir)
    plot_eviction_detail(data, output_dir)
    plot_latency(data, output_dir)

    if args.per_head:
        plot_head_heatmaps(data, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
