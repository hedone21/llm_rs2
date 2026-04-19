#!/usr/bin/env python3
"""Tensor Partition benchmark result aggregator and reporter.

Reads JSONL files produced by experiments/run_tensor_partition.sh and outputs:
  1. Markdown table to stdout (prefill × ratio, avg_tbt_ms mean±Δ, tok/s, ttft_ms,
     Δ vs ratio=1.0 %, thermal peak mC)
  2. PNG plot: ratio → avg_tbt_ms, one line per prefill length
     experiments/reports/tensor_partition/tbt_vs_ratio.png
  3. REPORT.md with table, plot embed, and per-run thermal start records
     experiments/reports/tensor_partition/REPORT.md

Input file pattern:
  experiments/results/tensor_partition/r{R}_p{P}_run{N}.jsonl

Usage:
  python experiments/analysis/tensor_partition_report.py
  python experiments/analysis/tensor_partition_report.py \\
      --results-dir experiments/results/tensor_partition \\
      --output-dir  experiments/reports/tensor_partition
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[warn] matplotlib not available — skipping PNG output", file=sys.stderr)


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    """Load all JSON lines from a file, skipping blank/invalid lines."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def extract_summary(records: list[dict]) -> Optional[dict]:
    """Return the last record that contains 'avg_tbt_ms' (summary record)."""
    for rec in reversed(records):
        if "avg_tbt_ms" in rec:
            return rec
    return None


def extract_thermal_peak(records: list[dict]) -> Optional[float]:
    """Return max sys.thermal_mc across all per-token records."""
    peak = None
    for rec in records:
        sys_data = rec.get("sys", {})
        thermal = sys_data.get("thermal_mc")
        if thermal is not None:
            if peak is None or thermal > peak:
                peak = thermal
    return peak


def extract_thermal_start(records: list[dict]) -> Optional[float]:
    """Return thermal_mc from the first record that has sys.thermal_mc."""
    for rec in records:
        sys_data = rec.get("sys", {})
        thermal = sys_data.get("thermal_mc")
        if thermal is not None:
            return thermal
    return None


# ── File discovery ────────────────────────────────────────────────────────────

def discover_files(results_dir: str) -> dict:
    """Discover JSONL files matching r{R}_p{P}_run{N}.jsonl.

    Returns:
        dict[(ratio_str, prefill_int)] -> list[dict with path, rep, records, summary]
    """
    pattern = os.path.join(results_dir, "r*_p*_run*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[error] No JSONL files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    name_re = re.compile(r"r([\d.]+)_p(\d+)_run(\d+)\.jsonl$")
    data = defaultdict(list)

    for path in files:
        m = name_re.search(os.path.basename(path))
        if not m:
            continue
        ratio_str, prefill_str, rep_str = m.group(1), m.group(2), m.group(3)
        prefill = int(prefill_str)
        records = load_jsonl(path)
        summary = extract_summary(records)
        if summary is None:
            print(f"[warn] No summary record in {path} — skipping", file=sys.stderr)
            continue
        data[(ratio_str, prefill)].append({
            "path": path,
            "rep": int(rep_str),
            "records": records,
            "summary": summary,
        })

    return data


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(data: dict) -> dict:
    """Aggregate reps per (ratio, prefill) cell.

    Returns:
        dict[(ratio_str, prefill)] -> {
            avg_tbt_ms_mean, avg_tbt_ms_delta, ttft_ms_mean,
            thermal_peak_max, thermal_start_first, runs: list
        }
    """
    result = {}
    for key, runs in data.items():
        tbts = []
        ttfts = []
        thermal_peaks = []
        thermal_starts = []
        for run in sorted(runs, key=lambda r: r["rep"]):
            s = run["summary"]
            tbts.append(s.get("avg_tbt_ms", 0.0))
            ttfts.append(s.get("ttft_ms", 0.0))
            thermal_peaks.append(extract_thermal_peak(run["records"]))
            thermal_starts.append(extract_thermal_start(run["records"]))

        mean_tbt = sum(tbts) / len(tbts) if tbts else 0.0
        delta_tbt = (max(tbts) - min(tbts)) / 2.0 if len(tbts) > 1 else 0.0
        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        peaks_valid = [p for p in thermal_peaks if p is not None]
        starts_valid = [s for s in thermal_starts if s is not None]
        result[key] = {
            "avg_tbt_ms_mean": mean_tbt,
            "avg_tbt_ms_delta": delta_tbt,
            "ttft_ms_mean": mean_ttft,
            "thermal_peak_max": max(peaks_valid) if peaks_valid else None,
            "thermal_start_values": starts_valid,
            "runs": sorted(runs, key=lambda r: r["rep"]),
        }
    return result


# ── Formatting helpers ────────────────────────────────────────────────────────

def pct_change(base_val: float, exp_val: float) -> str:
    if base_val == 0:
        return "N/A"
    change = (exp_val - base_val) / base_val * 100
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.1f}%"


def fmt_thermal(mc: Optional[float]) -> str:
    if mc is None:
        return "N/A"
    return f"{mc/1000:.1f}°C"


# ── Markdown table ────────────────────────────────────────────────────────────

def build_markdown_table(agg: dict, prefills: list[int], ratios: list[str]) -> str:
    header = (
        "| prefill | ratio | avg_tbt_ms (mean±Δ) | tok/s | ttft_ms | "
        "Δ vs ratio=1.0 | thermal_peak |\n"
        "|---------|-------|---------------------|-------|---------|"
        "---------------|-------------|\n"
    )
    rows = []
    for p in prefills:
        # baseline: ratio=1.0
        baseline_key = ("1.0", p)
        baseline = agg.get(baseline_key)
        baseline_tbt = baseline["avg_tbt_ms_mean"] if baseline else 0.0

        for r in ratios:
            key = (r, p)
            if key not in agg:
                rows.append(f"| {p} | {r} | — | — | — | — | — |")
                continue
            cell = agg[key]
            tbt = cell["avg_tbt_ms_mean"]
            delta = cell["avg_tbt_ms_delta"]
            toks = 1000.0 / tbt if tbt > 0 else 0.0
            ttft = cell["ttft_ms_mean"]
            delta_vs_base = pct_change(baseline_tbt, tbt) if baseline else "N/A"
            thermal = fmt_thermal(cell["thermal_peak_max"])
            rows.append(
                f"| {p} | {r} | {tbt:.1f}±{delta:.1f} | {toks:.2f} | "
                f"{ttft:.0f} | {delta_vs_base} | {thermal} |"
            )
    return header + "\n".join(rows) + "\n"


# ── Plot ──────────────────────────────────────────────────────────────────────

def build_plot(agg: dict, prefills: list[int], ratios: list[str], output_path: str):
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(8, 5))

    ratio_floats = [float(r) for r in ratios]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, p in enumerate(prefills):
        tbts = []
        x_vals = []
        for r in ratios:
            key = (r, p)
            if key in agg:
                tbts.append(agg[key]["avg_tbt_ms_mean"])
                x_vals.append(float(r))
        if tbts:
            ax.plot(x_vals, tbts, marker="o", label=f"prefill={p}",
                    color=colors[i % len(colors)])

    ax.set_xlabel("tensor-partition ratio")
    ax.set_ylabel("avg_tbt_ms (ms/token)")
    ax.set_title("Tensor Partition: TBT vs Ratio\n(Galaxy S25, Qwen 2.5-1.5B Q4_0, OpenCL)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # 1.0 (GPU-only) on left, 0.75 (more CPU) on right

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {output_path}")


# ── Per-run thermal log ────────────────────────────────────────────────────────

def build_thermal_log(agg: dict, prefills: list[int], ratios: list[str]) -> str:
    lines = ["### Per-run Thermal Start (°C at first sys record)\n"]
    lines.append("| run | ratio | prefill | thermal_start |")
    lines.append("|-----|-------|---------|---------------|")
    for r in ratios:
        for p in prefills:
            key = (r, p)
            if key not in agg:
                continue
            for run in agg[key]["runs"]:
                thermal_start = extract_thermal_start(run["records"])
                t_str = fmt_thermal(thermal_start)
                lines.append(f"| run{run['rep']} | {r} | {p} | {t_str} |")
    return "\n".join(lines) + "\n"


# ── REPORT.md ─────────────────────────────────────────────────────────────────

def build_report(
    table: str,
    thermal_log: str,
    plot_rel_path: str,
    prefills: list[int],
    ratios: list[str],
) -> str:
    return f"""# Tensor Partition Benchmark Report

**Device**: Galaxy S25
**Model**: Qwen 2.5-1.5B Q4_0
**Backend**: OpenCL
**Ratios tested**: {', '.join(ratios)}
**Prefill lengths**: {', '.join(str(p) for p in prefills)}
**Decode tokens**: 128

## Results

{table}

## TBT vs Ratio Plot

![TBT vs Ratio]({plot_rel_path})

{thermal_log}

---
*Generated by experiments/analysis/tensor_partition_report.py*
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tensor partition benchmark report generator")
    parser.add_argument(
        "--results-dir",
        default="experiments/results/tensor_partition",
        help="Directory containing r{R}_p{P}_run{N}.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/reports/tensor_partition",
        help="Directory for REPORT.md and tbt_vs_ratio.png",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Discover and load
    data = discover_files(results_dir)
    agg = aggregate(data)

    # Determine axes from discovered data
    all_ratios_raw = sorted({k[0] for k in agg.keys()}, key=lambda x: -float(x))
    all_prefills = sorted({k[1] for k in agg.keys()})

    # Ensure standard order: 1.0 first for baseline
    standard_ratios = ["1.0", "0.875", "0.75"]
    ratios = [r for r in standard_ratios if r in all_ratios_raw]
    # Append any extra ratios found in results
    for r in all_ratios_raw:
        if r not in ratios:
            ratios.append(r)

    prefills = all_prefills if all_prefills else [128, 1024, 4096]

    print(f"[info] Found {len(data)} cells across ratios={ratios}, prefills={prefills}")

    # Build outputs
    table = build_markdown_table(agg, prefills, ratios)
    thermal_log = build_thermal_log(agg, prefills, ratios)

    plot_path = os.path.join(output_dir, "tbt_vs_ratio.png")
    build_plot(agg, prefills, ratios, plot_path)

    report_path = os.path.join(output_dir, "REPORT.md")
    # Relative path from report to plot
    plot_rel = os.path.basename(plot_path)
    report = build_report(table, thermal_log, plot_rel, prefills, ratios)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[report] Saved: {report_path}")

    # Print markdown table to stdout
    print("\n## Tensor Partition Results\n")
    print(table)
    print(thermal_log)


if __name__ == "__main__":
    main()
