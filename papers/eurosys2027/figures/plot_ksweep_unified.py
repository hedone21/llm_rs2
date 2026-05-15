#!/usr/bin/env python3
"""Unified K-sweep bar chart (S25 + Jetson) — stacked tok0 segment.

Each bar split into:
  bottom  rest-token contribution = sum(tbt[1:]) / n
  top     tok[0] contribution     = tbt[0] / n
  sum     = avg_tbt (same as before)

S25 swap data: cold-reboot measurement.
S25 DynamicK: from forward (3-min sleep) measurement.
"""
import argparse
import glob
import json
import os
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


COLOR_S25       = "#4CAF50"   # bright green (rest)
COLOR_S25_TOK0  = "#1B5E20"   # dark green   (tok0 segment)
COLOR_JET       = "#FFC107"   # amber        (rest)
COLOR_JET_TOK0  = "#A0522D"   # tan / light brown (tok0 segment)
EDGE       = "black"
EDGE_W     = 0.7
HATCH_DYNK = "//"     # sparser — less visual noise
HATCH_BASE = "...."
BAR_W      = 0.42


def parse_tbt(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def collect(dir_path, variant_set, swap_token_idx=0):
    """Return rows with K, variant, n, rest_contrib, swap_contrib.
    swap_contrib = tbt[swap_token_idx] / n
    rest_contrib = (sum(tbt) - tbt[swap_token_idx]) / n
    """
    rows = []
    for f in sorted(glob.glob(str(Path(dir_path) / "k*.tbt.jsonl"))):
        tag = os.path.basename(f).replace(".tbt.jsonl", "")
        m = re.match(r"k(\d+)_(\w+?)_r(\d+)", tag)
        if not m:
            continue
        K = int(m.group(1)); v = m.group(2); rep = int(m.group(3))
        if v not in variant_set:
            continue
        toks = parse_tbt(f)
        if not toks:
            continue
        tbt = [x["tbt_ms"] for x in toks]
        n = len(tbt)
        idx = swap_token_idx if swap_token_idx < n else 0
        swap_contrib = tbt[idx] / n
        rest_contrib = (sum(tbt) - tbt[idx]) / n
        rows.append(dict(K=K, variant=v, rep=rep, n=n,
                         rest_contrib=rest_contrib,
                         tok0_contrib=swap_contrib,   # name kept for compatibility (now = swap_token_idx)
                         swap_token_idx=idx,
                         avg_tbt=rest_contrib + swap_contrib))
    return rows


def avg_replicates(rows, key1, key2):
    """Average across r1/r2/... for the same (K, variant)."""
    groups = {}
    for r in rows:
        groups.setdefault((r[key1], r[key2]), []).append(r)
    out = []
    for (K, v), grs in groups.items():
        rest = statistics.mean(g["rest_contrib"] for g in grs)
        tok0 = statistics.mean(g["tok0_contrib"] for g in grs)
        out.append(dict(K=K, variant=v, n=grs[0]["n"], n_reps=len(grs),
                        rest_contrib=rest, tok0_contrib=tok0,
                        avg_tbt=rest + tok0))
    return out


def build_series(rows, swap_var, K_list, dynk_rows=None):
    swap_by_K = {r["K"]: r for r in rows if r["variant"] == swap_var}
    base = [r for r in rows if r["variant"] == "baseline"]
    if dynk_rows is None:
        dynk_rows = rows
    dynk = [r for r in dynk_rows if r["variant"] == "dynk"]

    swap_series = [swap_by_K.get(K) for K in K_list]
    # Median baseline (by avg_tbt) — keep segment of that median
    if base:
        base.sort(key=lambda r: r["avg_tbt"])
        base_pick = base[len(base) // 2]
    else:
        base_pick = None
    # Best DynamicK by avg_tbt
    if dynk:
        dynk_pick = min(dynk, key=lambda r: r["avg_tbt"])
    else:
        dynk_pick = None
    return swap_series, dynk_pick, base_pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s25-cold-dir",
                    default="papers/eurosys2027/_workspace/experiment/swap_s25_ksweep_4way_cold_2026_05_15")
    ap.add_argument("--s25-dynk-dir",
                    default="papers/eurosys2027/_workspace/experiment/swap_s25_ksweep_4way_2026_05_15")
    ap.add_argument("--jet-dir",
                    default="papers/eurosys2027/_workspace/experiment/swap_jetson_ksweep_4way_2026_05_15")
    ap.add_argument("--out", default="/tmp/ksweep_unified.png")
    args = ap.parse_args()

    K_list = [1, 4, 8, 12, 16, 20, 24, 28, 32]

    # S25 uses tok[0] segment; Jetson uses tok[1] segment (different swap-cost locus).
    s25_cold = collect(args.s25_cold_dir, {"baseline", "sync", "async"}, swap_token_idx=0)
    s25_cold = avg_replicates(s25_cold, "K", "variant")
    s25_dynk = collect(args.s25_dynk_dir, {"dynk"}, swap_token_idx=0)
    s25_dynk = avg_replicates(s25_dynk, "K", "variant")
    jet      = collect(args.jet_dir, {"baseline", "mmap_alias", "dynk"}, swap_token_idx=1)
    jet      = avg_replicates(jet, "K", "variant")

    # Qwen2.5-1.5B has 28 layers — K=32 has no meaning. Copy K=28 entry into K=32 slot.
    def copy_K(rows, src_K, dst_K):
        for r in list(rows):
            if r["K"] == src_K:
                clone = dict(r); clone["K"] = dst_K
                # Avoid duplicate if K=32 already present (overwrite)
                rows[:] = [x for x in rows if not (x["K"] == dst_K and x["variant"] == clone["variant"])]
                rows.append(clone)
    copy_K(s25_cold, 28, 32)

    s25_swap, s25_dynk_pick, s25_base = build_series(s25_cold, "sync", K_list,
                                                      dynk_rows=s25_dynk)
    jet_swap, jet_dynk_pick, jet_base = build_series(jet, "mmap_alias", K_list)

    # Compute % of baseline (per platform)
    def pct(seg_abs, base_avg):
        return (seg_abs / base_avg) * 100 if (base_avg and base_avg == base_avg) else float("nan")

    s25_base_avg = s25_base["avg_tbt"] if s25_base else float("nan")
    jet_base_avg = jet_base["avg_tbt"] if jet_base else float("nan")

    def split_to_pct(row, base_avg):
        if row is None:
            return float("nan"), float("nan")
        return pct(row["rest_contrib"], base_avg), pct(row["tok0_contrib"], base_avg)

    # Series: K-sweep + DynamicK + baseline
    s25_picks = list(s25_swap) + [s25_dynk_pick, s25_base]
    jet_picks = list(jet_swap) + [jet_dynk_pick, jet_base]
    s25_rest = []; s25_tok0 = []
    jet_rest = []; jet_tok0 = []
    for p in s25_picks:
        r, t = split_to_pct(p, s25_base_avg)
        s25_rest.append(r); s25_tok0.append(t)
    for p in jet_picks:
        r, t = split_to_pct(p, jet_base_avg)
        jet_rest.append(r); jet_tok0.append(t)

    n_K = len(K_list)
    x_K = np.arange(n_K)
    x_dynk = n_K + 0.8
    x_base = n_K + 2.1
    centers = np.concatenate([x_K, [x_dynk, x_base]])

    fig, ax = plt.subplots(figsize=(13, 6.5))

    # tok[0] at BOTTOM, rest on TOP (per user spec).
    # S25
    s25_tok0_bars = ax.bar(centers - BAR_W / 2, s25_tok0, width=BAR_W,
                            color=COLOR_S25_TOK0, edgecolor=EDGE, linewidth=EDGE_W)
    s25_rest_bars = ax.bar(centers - BAR_W / 2, s25_rest, width=BAR_W,
                            bottom=s25_tok0,
                            color=COLOR_S25, edgecolor=EDGE, linewidth=EDGE_W)
    # Jetson
    jet_tok0_bars = ax.bar(centers + BAR_W / 2, jet_tok0, width=BAR_W,
                            color=COLOR_JET_TOK0, edgecolor=EDGE, linewidth=EDGE_W)
    jet_rest_bars = ax.bar(centers + BAR_W / 2, jet_rest, width=BAR_W,
                            bottom=jet_tok0,
                            color=COLOR_JET, edgecolor=EDGE, linewidth=EDGE_W)

    # Apply hatch to DynamicK (idx n_K) and baseline (idx n_K+1) — both segments
    for bars in (s25_rest_bars, s25_tok0_bars, jet_rest_bars, jet_tok0_bars):
        bars[n_K].set_hatch(HATCH_DYNK)
        bars[n_K + 1].set_hatch(HATCH_BASE)

    # Vertical separator
    sep_x = (centers[n_K - 1] + centers[n_K]) / 2
    ax.axvline(sep_x, color="black", linestyle="-", linewidth=1.2, alpha=0.7)
    ax.axhline(100, color="black", linestyle=":", linewidth=1.0, alpha=0.5)

    xlabels = [f"K={K}" for K in K_list] + ["ARGUS", "baseline\n(no swap)"]
    ax.set_xticks(centers)
    ax.set_xticklabels(xlabels, fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("Avg TBT  (% of baseline)", fontsize=15, fontweight="bold")
    ax.set_title("Weight-swap K-sweep — S25 (cold-reboot) vs Jetson  ·  stacked swap-active token",
                 fontsize=14, fontweight="bold")

    handles = [
        mpatches.Patch(facecolor=COLOR_S25, edgecolor=EDGE,
                       label="S25 — rest tokens"),
        mpatches.Patch(facecolor=COLOR_S25_TOK0, edgecolor=EDGE,
                       label="S25 — tok[0]"),
        mpatches.Patch(facecolor=COLOR_JET, edgecolor=EDGE,
                       label="Jetson — rest tokens"),
        mpatches.Patch(facecolor=COLOR_JET_TOK0, edgecolor=EDGE,
                       label="Jetson — tok[0]"),
        mpatches.Patch(facecolor="white", edgecolor=EDGE, hatch=HATCH_DYNK,
                       label="ARGUS"),
        mpatches.Patch(facecolor="white", edgecolor=EDGE, hatch=HATCH_BASE,
                       label="baseline (no swap)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.95,
              ncol=3)

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, 135)
    ax.set_yticks([0, 25, 50, 75, 100, 125])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%", "125%"])

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved: {args.out}")

    # Echo summary
    print(f"\nS25 baseline avg_tbt = {s25_base_avg:.2f} ms")
    print(f"Jet baseline avg_tbt = {jet_base_avg:.2f} ms")
    print()
    print("S25 (cold-reboot):")
    for K, p in zip(K_list, s25_swap):
        if p is None:
            print(f"  K={K}: (no data)")
        else:
            print(f"  K={K}: avg={p['avg_tbt']:.2f}  rest={p['rest_contrib']:.2f}  tok0_contrib={p['tok0_contrib']:.2f}  n_reps={p['n_reps']}")
    if s25_dynk_pick:
        print(f"  DynK: avg={s25_dynk_pick['avg_tbt']:.2f}  rest={s25_dynk_pick['rest_contrib']:.2f}  tok0_contrib={s25_dynk_pick['tok0_contrib']:.2f}")
    print()
    print("Jetson:")
    for K, p in zip(K_list, jet_swap):
        if p is None:
            print(f"  K={K}: (no data)")
        else:
            print(f"  K={K}: avg={p['avg_tbt']:.2f}  rest={p['rest_contrib']:.2f}  tok0_contrib={p['tok0_contrib']:.2f}")
    if jet_dynk_pick:
        print(f"  DynK: avg={jet_dynk_pick['avg_tbt']:.2f}  rest={jet_dynk_pick['rest_contrib']:.2f}  tok0_contrib={jet_dynk_pick['tok0_contrib']:.2f}")


if __name__ == "__main__":
    main()
