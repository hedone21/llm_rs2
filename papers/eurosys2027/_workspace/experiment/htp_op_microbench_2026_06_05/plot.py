#!/usr/bin/env python3
"""Per-op HTP NPU vs CPU latency — the ~100us dispatch floor (ARGUS style).

results.json → log-scale grouped bar (CPU vs HTP) + dispatch-floor line.
op-by-op offload 가 경량 op 에서 왜 실패하는지(floor >> compute) 시각화.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/go/.claude/skills/argus-figure-style/scripts")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from argus_style import (  # noqa: E402
    apply_argus_style, ARGUS, FONT, FIGSIZE, save_figure,
)

apply_argus_style()
HERE = Path(__file__).parent
data = json.loads((HERE / "results.json").read_text())
ops = data["ops"]
floor = data["dispatch_floor_us"]

# CPU 시간 오름차순으로 정렬 (flash_attn 은 CPU 없음 → 맨 끝, HTP-only).
order = ["get_rows", "mul", "silu", "add", "rms_norm", "rope", "softmax", "flash_attn"]
labels = ["get_rows", "mul", "silu", "add", "rms_norm", "rope", "softmax", "flash_attn"]
cpu = [ops[o]["cpu_us"] for o in order]
htp = [ops[o]["htp_us"] for o in order]

x = np.arange(len(order))
w = 0.38

fig, ax = plt.subplots(figsize=FIGSIZE.DOUBLE_COL)

# CPU bars (없는 셀=flash_attn 은 skip)
for i, v in enumerate(cpu):
    if v is None:
        continue
    ax.bar(x[i] - w / 2, v, w, color=ARGUS.BAR_TEAL, edgecolor=ARGUS.INK,
           linewidth=0.6, zorder=3, label="CPU (NEON)" if i == 0 else None)
    ax.text(x[i] - w / 2, v * 1.12, f"{v:.2f}" if v < 1 else f"{v:.0f}",
            ha="center", va="bottom", fontsize=FONT.MICRO)
# HTP bars
for i, v in enumerate(htp):
    ax.bar(x[i] + w / 2, v, w, color=ARGUS.BAR_ORANGE, edgecolor=ARGUS.INK,
           linewidth=0.6, zorder=3, label="HTP NPU" if i == 0 else None)
    ax.text(x[i] + w / 2, v * 1.12, f"{v:.0f}", ha="center", va="bottom",
            fontsize=FONT.MICRO)

# dispatch floor line
ax.axhline(floor, ls="--", lw=1.0, color=ARGUS.LINE_RED, zorder=2)
ax.text(-0.4, floor * 1.35, f"HTP dispatch floor ~{floor} us",
        ha="left", va="bottom", fontsize=FONT.NOTE, color=ARGUS.LINE_RED)

ax.set_yscale("log")
ax.set_ylim(0.02, 2000)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Latency per op (us, log)")
ax.grid(axis="y", which="both", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=FONT.TICK, frameon=False)

# softmax = 유일하게 CPU(teal) > HTP(orange) 인 op (NPU 가 빠름)
si = order.index("softmax")
ax.annotate("only op where\nNPU < CPU", xy=(si, cpu[si] * 1.05),
            xytext=(si - 1.7, 700), fontsize=FONT.MICRO, color=ARGUS.LINE_GREEN,
            ha="center",
            arrowprops=dict(arrowstyle="->", color=ARGUS.LINE_GREEN, lw=0.8))

fig.text(0.5, -0.13,
         "Qwen2.5-1.5B decode shapes, Galaxy S25.  HTP latency is floored at "
         "~100 us (FastRPC round-trip + DMA-BUF coherency) regardless of compute;\n"
         "only ops whose compute exceeds the floor (softmax, flash_attn, matmul) "
         "benefit from NPU offload.",
         ha="center", fontsize=FONT.MICRO, color=ARGUS.MUTED)

fig.subplots_adjust(bottom=0.26)
save_figure(fig, "htp_op_microbench", source_dir=HERE)
print("saved:", HERE / "htp_op_microbench.png")
