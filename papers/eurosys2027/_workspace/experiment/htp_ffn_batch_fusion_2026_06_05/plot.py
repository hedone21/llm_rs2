#!/usr/bin/env python3
"""HTP FFN gate/up batch fusion — decode TBT before/after, q4 + f16 (ARGUS style).

results.json → 2-panel (q4 | f16) HTP decode TBT: batch OFF(per-op drain) vs
ON(enqueue×2→drain×1). per-layer FastRPC dispatch floor 를 gate/up 에서 2회→1회로
줄인 결과. CPU 참조선으로 'NPU < CPU, fusion = floor 제거지 추월 아님' 을 명시.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/go/.claude/skills/argus-figure-style/scripts")
import matplotlib.pyplot as plt  # noqa: E402
from argus_style import (  # noqa: E402
    apply_argus_style, ARGUS, FONT, FIGSIZE, save_figure, panel_title,
)

apply_argus_style()
HERE = Path(__file__).parent
d = json.loads((HERE / "results.json").read_text())


def panel(ax, m, letter, title):
    off_med, on_med = m["off"]["median"], m["on"]["median"]
    off_r, on_r = m["off"]["runs"], m["on"]["runs"]
    meds = [off_med, on_med]
    errs = list(map(list, zip(
        [off_med - min(off_r), max(off_r) - off_med],
        [on_med - min(on_r), max(on_r) - on_med],
    )))
    labels = ["OFF\n(per-op)", "ON\n(batch)"]
    bars = ax.bar(labels, meds, width=0.55,
                  color=[ARGUS.BAR_ORANGE, ARGUS.BAR_TEAL],
                  edgecolor=ARGUS.INK, linewidth=0.7, zorder=3)
    ax.errorbar(range(2), meds, yerr=errs, fmt="none",
                ecolor=ARGUS.INK, elinewidth=0.9, capsize=3, zorder=4)
    for b, v in zip(bars, meds):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}",
                ha="center", va="bottom", fontsize=FONT.NOTE)
    cpu = m["cpu_decode"]
    ax.axhline(cpu, ls=":", lw=1.0, color=ARGUS.LINE_RED, zorder=2)
    ax.text(1.45, cpu + 1, f"CPU {cpu:.1f}", ha="right", va="bottom",
            fontsize=FONT.MICRO, color=ARGUS.LINE_RED)
    delta, pct = m["delta_ms"], m["delta_pct"]
    ax.annotate(f"−{delta:.1f}\n({pct:.1f}%)",
                xy=(1, on_med), xytext=(0.5, on_med * 0.62),
                ha="center", fontsize=FONT.NOTE, color=ARGUS.LINE_GREEN,
                arrowprops=dict(arrowstyle="->", color=ARGUS.LINE_GREEN, lw=0.9))
    ax.set_ylim(0, max(meds) * 1.25)
    ax.grid(axis="y", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    panel_title(ax, letter, title)


fig, (axl, axr) = plt.subplots(1, 2, figsize=FIGSIZE.DOUBLE_COL)
panel(axl, d["q4"], "a", "Q4_0 weight")
panel(axr, d["f16"], "b", "F16 weight")
axl.set_ylabel("HTP decode TBT (ms/tok)")

fig.text(0.5, 0.005,
         "Galaxy S25, Qwen2.5-1.5B, 6T, 64-tok decode (n=3 median, non-overlapping).  "
         "FFN gate/up batching removes one FastRPC dispatch floor per layer (28 layers); "
         "token-id 16/16 preserved.  Still > CPU — floor recovery, not a CPU speedup.",
         ha="center", fontsize=FONT.MICRO, color=ARGUS.MUTED, wrap=True)
fig.subplots_adjust(bottom=0.26, wspace=0.28)

save_figure(fig, "htp_ffn_batch_fusion", source_dir=HERE)
print("saved:", HERE / "htp_ffn_batch_fusion.png")
