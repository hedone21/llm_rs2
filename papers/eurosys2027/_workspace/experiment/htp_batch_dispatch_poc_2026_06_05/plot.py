#!/usr/bin/env python3
"""HTP async batch dispatch PoC — floor recovery (ARGUS style).

mm_qkv GEMV 를 N회 dispatch: sync(N×write→read) vs batch(N×write→drain).
sync 는 op당 ~146us floor 로 선형, batch 는 floor amortize 로 sublinear.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/go/.claude/skills/argus-figure-style/scripts")
import matplotlib.pyplot as plt  # noqa: E402
from argus_style import (  # noqa: E402
    apply_argus_style, ARGUS, FONT, FIGSIZE, save_figure,
)

apply_argus_style()
HERE = Path(__file__).parent

N = [1, 7, 14, 28]
A = [146.09, 1032.08, 2033.91, 4010.10]   # sync 1:1 (us)
B = [145.68, 449.69, 833.75, 1424.69]      # async batch (us)
ratio = [b / a for a, b in zip(A, B)]

fig, ax = plt.subplots(figsize=FIGSIZE.SINGLE_COL)
ax.plot(N, A, "o-", color=ARGUS.LINE_RED, lw=1.4, ms=4,
        label="sync 1:1 (N×write→read)", zorder=3)
ax.plot(N, B, "s-", color=ARGUS.LINE_GREEN, lw=1.4, ms=4,
        label="async batch (N×write→drain)", zorder=3)

# ratio 주석 (batch 점 아래)
for n, a, b, r in zip(N, A, B, ratio):
    if n == 1:
        continue
    ax.annotate(f"{r:.2f}×", xy=(n, b), xytext=(n, b - 350),
                ha="center", fontsize=FONT.NOTE, color=ARGUS.LINE_GREEN)

ax.set_xlabel("batch size N (ops dispatched)")
ax.set_ylabel("total wall-clock (us)")
ax.set_xticks(N)
ax.grid(True, linewidth=0.4, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=FONT.NOTE, frameon=False)
ax.set_ylim(0, 4400)

fig.text(0.5, 0.005,
         "Galaxy S25, mm_qkv Q4_0 GEMV.  Batching recovers the ~100 us per-op "
         "floor (per-op 146→47 us; err 2.2e-2 preserved).",
         ha="center", fontsize=FONT.MICRO, color=ARGUS.MUTED)
fig.subplots_adjust(bottom=0.30)

save_figure(fig, "htp_batch_dispatch_poc", source_dir=HERE)
print("saved:", HERE / "htp_batch_dispatch_poc.png")
