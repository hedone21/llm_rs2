#!/usr/bin/env python3
"""Parse comprehensive benchmark results and generate comparison report."""
import csv
import re
import sys
from pathlib import Path

OUTDIR = Path("/tmp/bench_results")

CONFIGS = [
    ("llmrs_cpu", "llm.rs CPU"),
    ("llmrs_gpu", "llm.rs GPU"),
    ("llamacpp_cpu", "llama.cpp CPU"),
    ("llamacpp_gpu", "llama.cpp GPU"),
]


def parse_llmrs_output(path):
    """Parse llm.rs generate output for prefill/decode metrics."""
    text = path.read_text()
    result = {}
    # Prefill: 490.17 ms (83 tokens, 169.3 tok/s)
    m = re.search(r"Prefill:\s+([\d.]+)\s+ms\s+\((\d+)\s+tokens,\s+([\d.]+)\s+tok/s\)", text)
    if m:
        result["prefill_ms"] = float(m.group(1))
        result["prefill_tokens"] = int(m.group(2))
        result["prefill_toks"] = float(m.group(3))
    # Decode: 44.06 ms/tok (22.7 tok/s) [127 tokens, forward only]
    m = re.search(r"Decode:\s+([\d.]+)\s+ms/tok\s+\(([\d.]+)\s+tok/s\)\s+\[(\d+)\s+tokens", text)
    if m:
        result["decode_ms"] = float(m.group(1))
        result["decode_toks"] = float(m.group(2))
        result["decode_tokens"] = int(m.group(3))
    return result


def parse_llamacpp_output(path):
    """Parse llama.cpp output for prefill/decode metrics."""
    text = path.read_text()
    result = {}
    # prompt eval time =     469.46 ms /    83 tokens (    5.66 ms per token,   176.80 tokens per second)
    m = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
        text,
    )
    if m:
        result["prefill_ms"] = float(m.group(1))
        result["prefill_tokens"] = int(m.group(2))
        result["prefill_toks"] = float(m.group(4))
    #        eval time =    5248.49 ms /   127 runs   (   41.33 ms per token,    24.20 tokens per second)
    m = re.search(
        r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
        text,
    )
    if m:
        result["decode_ms"] = float(m.group(3))
        result["decode_toks"] = float(m.group(4))
        result["decode_tokens"] = int(m.group(2))
    return result


def parse_memory(path):
    """Parse memory CSV: time_s,rss_kb,vmpeak_kb"""
    if not path.exists():
        return {}
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rss = int(row["rss_kb"])
                if rss > 0:
                    rows.append(rss)
            except (ValueError, KeyError):
                continue
    if not rows:
        return {}
    peak_mb = max(rows) / 1024
    avg_mb = sum(rows) / len(rows) / 1024
    min_mb = min(rows) / 1024
    samples = len(rows)
    # Compute stddev
    mean = sum(rows) / len(rows)
    variance = sum((x - mean) ** 2 for x in rows) / len(rows)
    stddev_mb = (variance ** 0.5) / 1024
    return {
        "peak_mb": peak_mb,
        "avg_mb": avg_mb,
        "min_mb": min_mb,
        "stddev_mb": stddev_mb,
        "samples": samples,
        "raw_kb": rows,
    }


def main():
    print("=" * 78)
    print(" Comprehensive Benchmark Report: llm.rs vs llama.cpp (Llama 3.2 1B F16)")
    print("=" * 78)
    print()

    results = {}
    for tag, label in CONFIGS:
        out_path = OUTDIR / f"{tag}_out.txt"
        mem_path = OUTDIR / f"{tag}_mem.csv"

        if not out_path.exists():
            print(f"  [{label}] output not found, skipping")
            continue

        if tag.startswith("llmrs"):
            perf = parse_llmrs_output(out_path)
        else:
            perf = parse_llamacpp_output(out_path)

        mem = parse_memory(mem_path)
        results[tag] = {"label": label, "perf": perf, "mem": mem}

    if not results:
        print("No results found!")
        return

    # === Speed Comparison ===
    print("### 1. Prefill Speed (83 tokens)")
    print(f"{'Config':<20} {'ms':>8} {'tok/s':>8} {'ms/tok':>8}")
    print("-" * 48)
    for tag, label in CONFIGS:
        if tag not in results or "prefill_ms" not in results[tag]["perf"]:
            print(f"{label:<20} {'N/A':>8}")
            continue
        p = results[tag]["perf"]
        ms_tok = p["prefill_ms"] / p["prefill_tokens"] if p["prefill_tokens"] > 0 else 0
        print(f"{label:<20} {p['prefill_ms']:>8.1f} {p['prefill_toks']:>8.1f} {ms_tok:>8.2f}")
    print()

    print("### 2. Decode Speed")
    print(f"{'Config':<20} {'ms/tok':>8} {'tok/s':>8} {'tokens':>8}")
    print("-" * 48)
    for tag, label in CONFIGS:
        if tag not in results or "decode_ms" not in results[tag]["perf"]:
            print(f"{label:<20} {'N/A':>8}")
            continue
        p = results[tag]["perf"]
        print(f"{label:<20} {p['decode_ms']:>8.2f} {p['decode_toks']:>8.1f} {p['decode_tokens']:>8}")
    print()

    # === Memory Comparison ===
    print("### 3. Memory Usage (RSS)")
    print(f"{'Config':<20} {'Peak MB':>8} {'Avg MB':>8} {'StdDev':>8} {'Min MB':>8} {'Samples':>8}")
    print("-" * 60)
    for tag, label in CONFIGS:
        if tag not in results or not results[tag]["mem"]:
            print(f"{label:<20} {'N/A':>8}")
            continue
        m = results[tag]["mem"]
        print(
            f"{label:<20} {m['peak_mb']:>8.1f} {m['avg_mb']:>8.1f} "
            f"{m['stddev_mb']:>8.1f} {m['min_mb']:>8.1f} {m['samples']:>8}"
        )
    print()

    # === Memory Timeline (text-based sparkline) ===
    print("### 4. Memory Timeline (RSS, normalized)")
    for tag, label in CONFIGS:
        if tag not in results or not results[tag]["mem"]:
            continue
        raw = results[tag]["mem"]["raw_kb"]
        if len(raw) < 3:
            continue
        # Downsample to 50 points
        n = min(50, len(raw))
        step = max(1, len(raw) // n)
        sampled = [raw[i * step] for i in range(n)]
        # Normalize to 0-8 range for display
        lo, hi = min(sampled), max(sampled)
        bars = " ▁▂▃▄▅▆▇█"
        if hi == lo:
            line = bars[4] * n
        else:
            line = "".join(bars[min(8, int((v - lo) / (hi - lo) * 8))] for v in sampled)
        peak_mb = hi / 1024
        print(f"  {label:<18} [{line}] peak={peak_mb:.0f}MB")
    print()

    # === Summary ===
    print("### 5. Summary")
    cpu_tags = [("llmrs_cpu", "llamacpp_cpu")]
    gpu_tags = [("llmrs_gpu", "llamacpp_gpu")]

    for ours, theirs in cpu_tags + gpu_tags:
        if ours not in results or theirs not in results:
            continue
        op = results[ours]["perf"]
        tp = results[theirs]["perf"]
        label = "CPU" if "cpu" in ours else "GPU"
        print(f"  {label}:")
        if "prefill_toks" in op and "prefill_toks" in tp:
            ratio = op["prefill_toks"] / tp["prefill_toks"] if tp["prefill_toks"] > 0 else 0
            print(f"    Prefill: llm.rs {op['prefill_toks']:.1f} vs llama.cpp {tp['prefill_toks']:.1f} tok/s (ratio {ratio:.2f}x)")
        if "decode_toks" in op and "decode_toks" in tp:
            ratio = op["decode_toks"] / tp["decode_toks"] if tp["decode_toks"] > 0 else 0
            print(f"    Decode:  llm.rs {op['decode_toks']:.1f} vs llama.cpp {tp['decode_toks']:.1f} tok/s (ratio {ratio:.2f}x)")
        om = results[ours]["mem"]
        tm = results[theirs]["mem"]
        if om and tm:
            ratio = om["peak_mb"] / tm["peak_mb"] if tm["peak_mb"] > 0 else 0
            print(f"    Memory:  llm.rs {om['peak_mb']:.0f} vs llama.cpp {tm['peak_mb']:.0f} MB peak (ratio {ratio:.2f}x)")


if __name__ == "__main__":
    main()
