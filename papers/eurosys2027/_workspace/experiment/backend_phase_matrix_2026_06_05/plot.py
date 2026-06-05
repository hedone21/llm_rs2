#!/usr/bin/env python3
"""Backend × format × phase 성능 figure (ARGUS style).

results.json (driver.py 산출) → 2-panel grouped bar:
  (a) Prefill throughput (tok/s, 2-point slope)
  (b) Decode  throughput (tok/s, steady state)
x = CPU / GPU / NPU, 그룹 = {Q4_0, F16}.

전 셀이 명시 backend 에서 실제 dispatch (A 실험으로 F16 weight 도 rpcmem +
HTP NPU dispatch 배선 — token-id 16/16 CPU 일치 검증). F16-on-NPU 는
~3.3GB rpcmem 을 DSP 로 흘리는 memory-bound 라 전 매트릭스 최저속.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/go/.claude/skills/argus-figure-style/scripts")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from argus_style import (  # noqa: E402
    apply_argus_style, ARGUS, FONT, FIGSIZE, panel_title, save_figure,
)

apply_argus_style()

HERE = Path(__file__).parent
data = json.loads((HERE / "results.json").read_text())

BACKENDS = ["CPU", "GPU", "NPU"]
FORMATS = [("q4", "Q4_0", ARGUS.BAR_TEAL), ("f16", "F16", ARGUS.BAR_YELLOW)]


def val(backend, fmt, key):
    cell = data.get(f"{backend}/{fmt}")
    if not cell:
        return None
    return cell.get("derived", {}).get(key)


def draw_panel(ax, key, ylabel, letter, title):
    x = np.arange(len(BACKENDS))
    w = 0.38
    for i, (fkey, flabel, color) in enumerate(FORMATS):
        vals = [val(b, fkey, key) for b in BACKENDS]
        offs = x + (i - 0.5) * w
        heights = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(offs, heights, w, label=flabel, color=color,
                      edgecolor=ARGUS.INK, linewidth=0.6, zorder=3)
        for rect, v, b in zip(bars, vals, BACKENDS):
            if v is None:
                # OOM / 측정 실패 → 회색 빈 표식
                ax.text(rect.get_x() + rect.get_width() / 2, 0.5,
                        "n/a", ha="center", va="bottom",
                        fontsize=FONT.MICRO, color=ARGUS.MUTED, rotation=90)
                continue
            ax.text(rect.get_x() + rect.get_width() / 2, v,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=FONT.NOTE)
    ax.set_xticks(x)
    ax.set_xticklabels(BACKENDS)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    # headroom for labels
    ymax = max([v for b in BACKENDS for f, _, _ in FORMATS
                for v in [val(b, f, key)] if v is not None] + [1])
    ax.set_ylim(0, ymax * 1.18)
    panel_title(ax, letter, title)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE.DOUBLE_COL)
draw_panel(ax1, "prefill_tps", "Prefill throughput (tok/s)", "a", "Prefill")
draw_panel(ax2, "decode_tps", "Decode throughput (tok/s)", "b", "Decode")

# 단일 범례 (상단 중앙)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2,
           fontsize=FONT.TICK, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.text(0.5, -0.02,
         "Qwen2.5-1.5B, Galaxy S25 (6 threads).  All cells dispatch on the named "
         "backend; F16 weights are mirrored to rpcmem for NPU dispatch.  "
         "F16-on-NPU is memory-bound (~3.3 GB rpcmem) and hence slowest.",
         ha="center", fontsize=FONT.MICRO, color=ARGUS.MUTED)

save_figure(fig, "backend_phase_matrix", source_dir=HERE)
print("figure saved:", HERE / "backend_phase_matrix.pdf")
