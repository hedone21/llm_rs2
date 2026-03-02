#!/usr/bin/env python3
"""
Visualize KV cache memory usage under memory pressure eviction.
Reads: results/data/eviction_memory_test.json
Output: results/plots/eviction_memory.png
"""
import json
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "results", "data", "eviction_memory_test.json")
    out_path = os.path.join(project_root, "results", "plots", "eviction_memory.png")

    with open(data_path) as f:
        result = json.load(f)

    data = result["data"]
    config = result["config"]

    # Extract arrays
    steps = list(range(len(data)))
    memory_mb = [d["memory_mb"] for d in data]
    tokens = [d["tokens"] for d in data]
    events = [d["event"] for d in data]
    levels = [d["level"] for d in data]

    # Color map for levels
    level_colors = {
        "normal": "#4CAF50",
        "warning": "#FF9800",
        "critical": "#F44336",
        "emergency": "#9C27B0",
    }

    # Find eviction events
    eviction_indices = [i for i, e in enumerate(events) if "pre_eviction" in e]
    post_eviction_indices = [i for i, e in enumerate(events) if "post_eviction" in e]

    # ── Figure with 2 subplots ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 2]})
    fig.suptitle("KV Cache Memory Under Pressure-Driven Eviction",
                 fontsize=16, fontweight='bold', y=0.98)

    # ── Top: Memory (MB) ──
    # Background color bands by level
    prev_level = levels[0]
    band_start = 0
    for i in range(1, len(levels)):
        if levels[i] != prev_level or i == len(levels) - 1:
            end = i if levels[i] != prev_level else i + 1
            color = level_colors.get(prev_level, "#E0E0E0")
            ax1.axvspan(band_start, end - 1, alpha=0.12, color=color, linewidth=0)
            band_start = i
            prev_level = levels[i]

    # Main memory line
    ax1.plot(steps, memory_mb, color="#1565C0", linewidth=2, zorder=3, label="KV Cache Memory")

    # Eviction arrows (pre → post)
    for pre_i in eviction_indices:
        post_i = pre_i + 1
        if post_i < len(data):
            # Vertical drop line
            ax1.annotate("",
                xy=(post_i, memory_mb[post_i]),
                xytext=(pre_i, memory_mb[pre_i]),
                arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=2.5),
                zorder=5)

            # Label with reduction
            reduction_mb = memory_mb[pre_i] - memory_mb[post_i]
            reduction_pct = reduction_mb / memory_mb[pre_i] * 100
            level_name = events[pre_i].replace("pre_eviction_", "").upper()
            ax1.annotate(
                f"{level_name}\n−{reduction_mb:.1f} MB\n({reduction_pct:.0f}%)",
                xy=(post_i, memory_mb[post_i]),
                xytext=(post_i + 1.5, memory_mb[post_i] + 1.0),
                fontsize=9, fontweight='bold', color="#D32F2F",
                arrowprops=dict(arrowstyle="-", color="#D32F2F", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D32F2F", alpha=0.9),
                zorder=6)

    # Mark eviction points
    for i in post_eviction_indices:
        ax1.scatter([i], [memory_mb[i]], color="#D32F2F", s=80, zorder=5, marker='v')

    ax1.set_ylabel("KV Cache Memory (MB)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Legend for levels
    patches = [mpatches.Patch(color=c, alpha=0.3, label=l.capitalize())
               for l, c in level_colors.items()]
    patches.append(mpatches.Patch(color="#D32F2F", alpha=0.8, label="Eviction"))
    ax1.legend(handles=patches, loc="upper left", fontsize=9, framealpha=0.9)

    # ── Bottom: Token count ──
    for i in range(1, len(levels)):
        if levels[i] != prev_level or i == len(levels) - 1:
            pass
    prev_level = levels[0]
    band_start = 0
    for i in range(1, len(levels)):
        if levels[i] != prev_level or i == len(levels) - 1:
            end = i if levels[i] != prev_level else i + 1
            color = level_colors.get(prev_level, "#E0E0E0")
            ax2.axvspan(band_start, end - 1, alpha=0.12, color=color, linewidth=0)
            band_start = i
            prev_level = levels[i]

    ax2.plot(steps, tokens, color="#2E7D32", linewidth=2, zorder=3, label="Cached Tokens")
    ax2.fill_between(steps, tokens, alpha=0.15, color="#2E7D32")

    for i in post_eviction_indices:
        ax2.scatter([i], [tokens[i]], color="#D32F2F", s=80, zorder=5, marker='v')

    ax2.set_ylabel("Cached Tokens", fontsize=12)
    ax2.set_xlabel("Measurement Step", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper left", fontsize=9)

    # ── Config annotation ──
    config_text = (
        f"Config: {config['num_layers']} layers × "
        f"{config['num_kv_heads']} KV heads × {config['head_dim']} dim\n"
        f"Per-token: {config['per_token_bytes'] / 1024:.1f} KB "
        f"({config['per_token_bytes']} B)  |  "
        f"Max seq: {config['max_seq_len']}"
    )
    fig.text(0.5, 0.01, config_text, ha='center', fontsize=9, color='gray',
             style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {out_path}")

    # Also print a summary table
    print("\n=== Eviction Summary Table ===")
    print(f"{'Phase':<20} {'Tokens Before':>14} {'Tokens After':>13} {'Memory Before':>14} {'Memory After':>13} {'Reduction':>10}")
    print("-" * 90)

    for pre_i in eviction_indices:
        post_i = pre_i + 1
        if post_i < len(data):
            level_name = events[pre_i].replace("pre_eviction_", "").capitalize()
            reduction = memory_mb[pre_i] - memory_mb[post_i]
            pct = reduction / memory_mb[pre_i] * 100
            print(f"{level_name:<20} {tokens[pre_i]:>14} {tokens[post_i]:>13} "
                  f"{memory_mb[pre_i]:>13.2f}M {memory_mb[post_i]:>12.2f}M "
                  f"{reduction:>7.2f}M ({pct:.0f}%)")


if __name__ == "__main__":
    main()
