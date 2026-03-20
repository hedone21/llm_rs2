#!/usr/bin/env python3
"""Compare memory usage between llm.rs and llama.cpp on Android device.

Usage:
    python scripts/compare_memory.py [--tokens N] [--dtype q4|f16|both] [--poll-interval 0.2]
"""

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime


PROMPT = "The quick brown fox jumps over the lazy dog and then"


def run_adb(cmd, timeout=30):
    result = subprocess.run(
        ["adb", "shell", cmd],
        capture_output=True, text=True, timeout=timeout
    )
    return result.stdout.strip()


def build_configs(num_tokens, dtype, backend="cpu"):
    """Build run configurations for both frameworks."""
    if dtype == "q4":
        llm_rs_dtype_flag = "--weight-dtype q4"
        gguf_model = "/data/local/tmp/models/llama3.2-1b-q4_0.gguf"
    else:
        llm_rs_dtype_flag = "--weight-dtype f16"
        gguf_model = "/data/local/tmp/models/llama3.2-1b-f16.gguf"

    # Backend selection
    if backend == "opencl":
        llm_rs_backend = "-b opencl --zero-copy"
        llamacpp_gpu = "-ngl 99"
        backend_label = "OpenCL+ZeroCopy"
    else:
        llm_rs_backend = "-b cpu"
        llamacpp_gpu = ""
        backend_label = "CPU"

    return [
        {
            "key": f"llm_rs_{dtype}",
            "name": f"llm.rs ({dtype.upper()}, {backend_label})",
            "cmd": (
                f'LD_LIBRARY_PATH=/data/local/tmp '
                f'/data/local/tmp/generate '
                f'--model-path /data/local/tmp/models/llama3.2-1b '
                f'--prompt "{PROMPT}" '
                f'--num-tokens {num_tokens} '
                f'{llm_rs_backend} {llm_rs_dtype_flag} '
                f'--temperature 0'
            ),
            "process_name": "generate",
        },
        {
            "key": f"llamacpp_{dtype}",
            "name": f"llama.cpp ({dtype.upper()}, {backend_label})",
            "cmd": (
                f'LD_LIBRARY_PATH=/data/local/tmp '
                f'/data/local/tmp/llama-cli-orig '
                f'-m {gguf_model} '
                f'-p "{PROMPT}" '
                f'-n {num_tokens} --temp 0 {llamacpp_gpu}'
            ),
            "process_name": "llama-cli-orig",
        },
    ]


def profile_binary(config, poll_interval=0.2):
    """Run binary on device and poll memory usage via /proc/pid/status VmRSS."""
    name = config["name"]
    cmd = config["cmd"]
    pname = config["process_name"]

    print(f"\n{'='*60}")
    print(f"  Profiling: {name}")
    print(f"{'='*60}")

    # On-device script:
    # 1. cd to /data/local/tmp (llama.cpp needs this)
    # 2. Start inference in background
    # 3. Find PID via pidof (reliable across compound commands)
    # 4. Poll VmRSS from /proc/pid/status (in KB, no page-size ambiguity)
    monitor_script = f"""
cd /data/local/tmp
{cmd} > /data/local/tmp/_mem_output.txt 2>&1 &
sleep 0.3
PID=$(pidof {pname} | awk '{{print $1}}')
if [ -z "$PID" ]; then
  sleep 0.5
  PID=$(pidof {pname} | awk '{{print $1}}')
fi
echo "PID=$PID"
if [ -z "$PID" ]; then
  echo "ERR: process not found"
  wait
  echo "---OUTPUT---"
  cat /data/local/tmp/_mem_output.txt
  exit 1
fi
while [ -d /proc/$PID ]; do
  TS=$(cat /proc/uptime | cut -d" " -f1)
  RSS=$(grep VmRSS /proc/$PID/status 2>/dev/null | awk '{{print $2}}')
  VSS=$(grep VmSize /proc/$PID/status 2>/dev/null | awk '{{print $2}}')
  if [ -n "$RSS" ]; then
    echo "MEM $TS $RSS $VSS"
  fi
  sleep {poll_interval}
done
wait
echo "EXIT=$?"
echo "---OUTPUT---"
cat /data/local/tmp/_mem_output.txt
"""

    proc = subprocess.Popen(
        ["adb", "shell", monitor_script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        stdout, stderr = proc.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("  TIMEOUT!")
        return None

    # Parse results
    samples = []
    pid = None
    output_lines = []
    exit_code = None
    in_output = False

    for line in stdout.splitlines():
        if in_output:
            output_lines.append(line)
        elif line.startswith("---OUTPUT---"):
            in_output = True
        elif line.startswith("PID="):
            try:
                pid = int(line.split("=")[1])
            except ValueError:
                pass
        elif line.startswith("MEM "):
            parts = line.split()
            # MEM <uptime_s> <VmRSS_kB> <VmSize_kB>
            if len(parts) >= 4:
                try:
                    ts = float(parts[1])
                    rss_kb = int(parts[2])
                    vss_kb = int(parts[3])
                    samples.append({
                        "time_s": ts,
                        "rss_mb": round(rss_kb / 1024, 2),
                        "vss_mb": round(vss_kb / 1024, 2),
                    })
                except (ValueError, IndexError):
                    pass
        elif line.startswith("EXIT="):
            try:
                exit_code = int(line.split("=")[1])
            except ValueError:
                pass
        elif line.startswith("ERR:"):
            print(f"  {line}")

    # Normalize timestamps to start from 0
    if samples:
        t0 = samples[0]["time_s"]
        for s in samples:
            s["time_s"] = round(s["time_s"] - t0, 3)

    peak_rss = max(s["rss_mb"] for s in samples) if samples else 0
    peak_vss = max(s["vss_mb"] for s in samples) if samples else 0
    avg_rss = sum(s["rss_mb"] for s in samples) / len(samples) if samples else 0
    duration = samples[-1]["time_s"] if samples else 0

    print(f"  PID: {pid}")
    print(f"  Samples: {len(samples)}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Peak RSS: {peak_rss:.1f} MB")
    print(f"  Avg RSS:  {avg_rss:.1f} MB")
    print(f"  Peak VSS: {peak_vss:.1f} MB")
    print(f"  Exit: {exit_code}")

    # Show inference output (timing lines)
    print(f"  --- Output ---")
    for line in output_lines[-15:]:
        if line.strip():
            print(f"  {line}")

    return {
        "name": name,
        "key": config["key"],
        "samples": samples,
        "peak_rss_mb": round(peak_rss, 2),
        "avg_rss_mb": round(avg_rss, 2),
        "peak_vss_mb": round(peak_vss, 2),
        "duration_s": round(duration, 2),
        "output": "\n".join(output_lines),
    }


def plot_comparison(all_results, output_path):
    """Generate comparison plot for one or two dtype configs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"llm.rs": "#2196F3", "llama.cpp": "#FF5722"}

    def get_color(name):
        for key, color in colors.items():
            if key in name:
                return color
        return "#888888"

    n_dtypes = len(all_results)
    fig, axes = plt.subplots(n_dtypes, 2, figsize=(14, 5 * n_dtypes))
    if n_dtypes == 1:
        axes = [axes]

    for row, (dtype_label, results) in enumerate(all_results.items()):
        # Left: RSS over time
        ax1 = axes[row][0]
        for r in results:
            times = [s["time_s"] for s in r["samples"]]
            rss = [s["rss_mb"] for s in r["samples"]]
            c = get_color(r["name"])
            ax1.plot(times, rss, label=r["name"], color=c, linewidth=2)
            ax1.axhline(y=r["peak_rss_mb"], color=c, linestyle="--", alpha=0.4)

        ax1.set_ylabel("RSS (MB)", fontsize=11)
        ax1.set_xlabel("Time (s)", fontsize=11)
        ax1.set_title(f"RSS Over Time — {dtype_label}", fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Right: Peak bar chart
        ax2 = axes[row][1]
        names = [r["name"] for r in results]
        peaks = [r["peak_rss_mb"] for r in results]
        avgs = [r["avg_rss_mb"] for r in results]
        bar_colors = [get_color(n) for n in names]

        x = range(len(names))
        width = 0.35
        bars1 = ax2.bar([i - width/2 for i in x], peaks, width, label="Peak RSS",
                         color=bar_colors, alpha=0.9)
        bars2 = ax2.bar([i + width/2 for i in x], avgs, width, label="Avg RSS",
                         color=bar_colors, alpha=0.5)

        for bar, val in zip(bars1, peaks):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        for bar, val in zip(bars2, avgs):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)

        ax2.set_ylabel("Memory (MB)", fontsize=11)
        ax2.set_title(f"Peak vs Avg RSS — {dtype_label}", fontsize=13)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(names, fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Memory Comparison: llm.rs vs llama.cpp (Llama 3.2 1B, CPU, Android)",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")


def run_dtype(dtype, num_tokens, poll_interval, backend="cpu"):
    """Run comparison for a single dtype."""
    print(f"\n{'#'*60}")
    print(f"  Testing dtype: {dtype.upper()}, backend: {backend}")
    print(f"{'#'*60}")

    configs = build_configs(num_tokens, dtype, backend)
    results = []
    for config in configs:
        result = profile_binary(config, poll_interval)
        if result:
            results.append(result)
        time.sleep(3)  # cool down
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare memory: llm.rs vs llama.cpp")
    parser.add_argument("--tokens", type=int, default=256, help="Number of tokens to generate")
    parser.add_argument("--dtype", choices=["q4", "f16", "both"], default="both",
                        help="Weight quantization")
    parser.add_argument("--poll-interval", type=float, default=0.15, help="Memory poll interval (s)")
    parser.add_argument("--backend", choices=["cpu", "opencl"], default="cpu",
                        help="Backend (cpu or opencl with zero-copy)")
    parser.add_argument("--output-dir", default="results/memory", help="Output directory")
    args = parser.parse_args()

    print(f"Config: {args.tokens} tokens, {args.dtype.upper()} weights, {args.backend} backend, poll every {args.poll_interval}s")

    dtypes = ["q4", "f16"] if args.dtype == "both" else [args.dtype]
    all_results = {}

    for dtype in dtypes:
        results = run_dtype(dtype, args.tokens, args.poll_interval, args.backend)
        if results:
            all_results[dtype.upper()] = results

    if not all_results:
        print("No results collected!")
        sys.exit(1)

    # Save raw data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"memory_compare_{args.dtype}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "config": {"tokens": args.tokens, "dtype": args.dtype},
            "timestamp": timestamp,
            "results": {k: v for k, v in all_results.items()},
        }, f, indent=2)
    print(f"\nData saved: {json_path}")

    # Plot
    plot_path = output_dir / f"memory_compare_{args.dtype}_{timestamp}.png"
    try:
        plot_comparison(all_results, str(plot_path))
    except ImportError:
        print("matplotlib not available, skipping plot")

    # Summary
    print(f"\n{'='*60}")
    print(f"  MEMORY COMPARISON SUMMARY ({args.tokens} tokens)")
    print(f"{'='*60}")
    for dtype_label, results in all_results.items():
        print(f"\n  [{dtype_label}]")
        print(f"  {'Framework':<25} {'Peak RSS':>10} {'Avg RSS':>10} {'Duration':>10}")
        print(f"  {'-'*55}")
        for r in results:
            print(f"  {r['name']:<25} {r['peak_rss_mb']:>8.1f}MB {r['avg_rss_mb']:>8.1f}MB {r['duration_s']:>8.1f}s")
        if len(results) == 2:
            ratio = results[0]["peak_rss_mb"] / results[1]["peak_rss_mb"] if results[1]["peak_rss_mb"] > 0 else 0
            print(f"  --> llm.rs / llama.cpp = {ratio:.2f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
