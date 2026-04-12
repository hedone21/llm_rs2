#!/usr/bin/env python3
"""Score distribution time-series analysis and visualization."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent
CSV_RAW = OUT_DIR / "score_distribution.csv"
CSV_NORM = OUT_DIR / "score_distribution_normalized.csv"

# ── Load ──────────────────────────────────────────────────────────────
df_raw = pd.read_csv(CSV_RAW)
df_norm = pd.read_csv(CSV_NORM)

# ── 1. Summary statistics by token_type ───────────────────────────────
print("=" * 60)
print("Score Distribution Summary")
print("=" * 60)
for tt in df_raw["token_type"].unique():
    sub = df_raw[df_raw["token_type"] == tt]
    print(f"\n[{tt}]  count={len(sub)}")
    print(f"  score  mean={sub['score'].mean():.4f}  std={sub['score'].std():.4f}"
          f"  min={sub['score'].min():.4f}  max={sub['score'].max():.4f}")
    if tt in df_norm["token_type"].unique():
        nsub = df_norm[df_norm["token_type"] == tt]
        print(f"  norm   mean={nsub['norm_score'].mean():.6f}  std={nsub['norm_score'].std():.6f}"
              f"  min={nsub['norm_score'].min():.6f}  max={nsub['norm_score'].max():.6f}")

# Prompt vs generated boundary
prompt_mask = df_raw["token_type"].isin(["prompt", "structural", "bos"])
gen_mask = df_raw["token_type"] == "generated"
prompt_end = df_raw[prompt_mask]["position"].max()
gen_start = df_raw[gen_mask]["position"].min()
print(f"\nPrompt region: 0 – {prompt_end}  ({prompt_mask.sum()} tokens)")
print(f"Generated region: {gen_start} – {df_raw['position'].max()}  ({gen_mask.sum()} tokens)")

# Generated score decay rate
gen = df_raw[gen_mask].copy()
gen_first, gen_last = gen.iloc[0]["score"], gen.iloc[-1]["score"]
print(f"\nGenerated score decay: {gen_first:.2f} → {gen_last:.2f}"
      f"  (Δ={gen_first - gen_last:.2f}, {(gen_first - gen_last)/len(gen)*100:.2f}%/token)")

# ── Color palette ─────────────────────────────────────────────────────
COLORS = {
    "bos": "#e74c3c",
    "prompt": "#3498db",
    "structural": "#e67e22",
    "generated": "#2ecc71",
}

# ── 2. Full score time-series (raw) ──────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 2, 2]})

# Panel A: full raw score
ax = axes[0]
for tt in ["prompt", "structural", "generated", "bos"]:
    sub = df_raw[df_raw["token_type"] == tt]
    ax.scatter(sub["position"], sub["score"], s=4, alpha=0.7,
               color=COLORS[tt], label=tt, zorder=3 if tt == "structural" else 2)
ax.axvline(x=prompt_end + 0.5, color="gray", ls="--", lw=1, alpha=0.6, label="prompt/gen boundary")
ax.set_ylabel("Raw Score")
ax.set_title("Score Distribution over Token Position (Raw)")
ax.legend(loc="upper right", markerscale=3, fontsize=9)
ax.set_xlim(-5, len(df_raw) + 5)

# Panel B: prompt region zoom
ax = axes[1]
p_df = df_raw[prompt_mask]
for tt in ["prompt", "structural", "bos"]:
    sub = p_df[p_df["token_type"] == tt]
    ax.scatter(sub["position"], sub["score"], s=10, alpha=0.8,
               color=COLORS[tt], label=tt, zorder=3 if tt != "prompt" else 2)
ax.set_ylabel("Raw Score")
ax.set_title(f"Prompt Region Zoom (pos 0–{prompt_end})")
ax.legend(loc="upper right", markerscale=2, fontsize=9)
# Exclude BOS from y-axis range for better visibility
prompt_only = p_df[p_df["token_type"].isin(["prompt", "structural"])]
if len(prompt_only) > 0:
    ymin = prompt_only["score"].min() - 1
    ymax = prompt_only["score"].max() + 1
    ax.set_ylim(ymin, ymax)
ax.set_xlim(-2, prompt_end + 5)

# Panel C: generated region — score decay
ax = axes[2]
ax.plot(gen["position"], gen["score"], color=COLORS["generated"], lw=1.2, alpha=0.9)
ax.fill_between(gen["position"], gen["score"], alpha=0.15, color=COLORS["generated"])
# Fit linear trend
z = np.polyfit(gen["position"], gen["score"], 1)
trend = np.poly1d(z)
ax.plot(gen["position"], trend(gen["position"]), "r--", lw=1, alpha=0.6,
        label=f"linear fit (slope={z[0]:.4f})")
ax.set_xlabel("Token Position")
ax.set_ylabel("Raw Score")
ax.set_title("Generated Region — Score Decay")
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(OUT_DIR / "score_timeseries.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT_DIR / 'score_timeseries.png'}")

# ── 3. Normalized score time-series ──────────────────────────────────
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))

# Panel A: full normalized score (excl. BOS for y-scale)
ax = axes2[0]
for tt in ["prompt", "structural", "generated"]:
    sub = df_norm[df_norm["token_type"] == tt]
    ax.scatter(sub["position"], sub["norm_score"], s=4, alpha=0.7,
               color=COLORS[tt], label=tt, zorder=3 if tt == "structural" else 2)
ax.axvline(x=prompt_end + 0.5, color="gray", ls="--", lw=1, alpha=0.6)
ax.set_ylabel("Normalized Score")
ax.set_title("Normalized Score Distribution (BOS excluded from view)")
ax.legend(loc="upper right", markerscale=3, fontsize=9)

# Panel B: generated region normalized
ax = axes2[1]
gen_n = df_norm[df_norm["token_type"] == "generated"]
ax.plot(gen_n["position"], gen_n["norm_score"], color=COLORS["generated"], lw=1.2)
ax.fill_between(gen_n["position"], gen_n["norm_score"], alpha=0.15, color=COLORS["generated"])
ax.set_xlabel("Token Position")
ax.set_ylabel("Normalized Score")
ax.set_title("Generated Region — Normalized Score Decay")

plt.tight_layout()
fig2.savefig(OUT_DIR / "score_timeseries_normalized.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'score_timeseries_normalized.png'}")

# ── 4. Score derivative (rate of change) ─────────────────────────────
fig3, ax3 = plt.subplots(figsize=(14, 4))
gen_diff = gen["score"].diff()
ax3.plot(gen["position"].iloc[1:], gen_diff.iloc[1:], color="#9b59b6", lw=0.8, alpha=0.8)
ax3.axhline(y=0, color="gray", ls="-", lw=0.5)
ax3.set_xlabel("Token Position")
ax3.set_ylabel("Δ Score (per token)")
ax3.set_title("Generated Region — Score Change Rate (1st Derivative)")
plt.tight_layout()
fig3.savefig(OUT_DIR / "score_derivative.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'score_derivative.png'}")

print("\nDone.")
