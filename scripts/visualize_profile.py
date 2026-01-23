import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
import os

def parse_time(t_str):
    try:
        # ISO format: 2026-01-23T14:31:27.555023
        return datetime.fromisoformat(t_str)
    except ValueError:
        return t_str

def main():
    parser = argparse.ArgumentParser(description="Visualize Android Benchmark Profile (JSON).")
    parser.add_argument("file", help="Path to the profile JSON file")
    parser.add_argument("--output", help="Save plot to file instead of showing")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        sys.exit(1)

    with open(args.file, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Invalid JSON format.")
            sys.exit(1)

    # 1. Version Check & Schema Validation Strategy
    version = data.get("version", 0) # Default to 0 if missing
    if version > 1:
        print(f"Warning: File version ({version}) is newer than supported (1). Some features might be missing.")
    
    # Robust Parsing: Use .get() with defaults for missing fields
    meta = data.get("metadata", {})
    results = data.get("benchmark_results", {})
    timeseries = data.get("timeseries", [])

    if not timeseries:
        print("Error: No timeseries data found.")
        sys.exit(1)

    # Prepare Data
    timestamps = [parse_time(x.get("timestamp")) for x in timeseries]
    temps = [x.get("temp_c", 0) for x in timeseries]
    gpu_freqs_mhz = [x.get("gpu_freq_hz", 0) / 1e6 for x in timeseries]
    
    # Handle variable number of CPU cores
    # cpu_freqs_khz is list of ints
    num_cores = 0
    if timeseries and "cpu_freqs_khz" in timeseries[0]:
        num_cores = len(timeseries[0]["cpu_freqs_khz"])
    
    cpu_freqs = [[] for _ in range(num_cores)]
    for x in timeseries:
        freqs = x.get("cpu_freqs_khz", [])
        for i in range(num_cores):
            val = freqs[i] if i < len(freqs) else 0
            cpu_freqs[i].append(val / 1e6) # GHz

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Title with Metadata
    title_str = (f"Benchmark Profile: {meta.get('model', 'Unknown')} ({meta.get('backend', 'Unknown')})\n"
                 f"Tokens: {meta.get('num_tokens', '?')} | TTFT: {results.get('ttft_ms', 'N/A')}ms | TBT: {results.get('tbt_ms', 'N/A')}ms")
    fig.suptitle(title_str, fontsize=14)

    # 1. Temperature
    ax1.plot(timestamps, temps, 'r-', label="Battery Temp")
    ax1.set_ylabel("Temperature (°C)")
    ax1.grid(True)
    ax1.legend(loc="upper left")
    
    # 2. CPU Freqs
    # Often 8 cores. Plotting them all can be messy. Group by cluster potentially?
    # For now, plot all but separate colors.
    for i in range(num_cores):
        # Heuristic: Core 7 is usually prime, 4-6 big, 0-3 little
        linestyle = '-' if i >= num_cores - 1 else ('--' if i >= num_cores - 4 else ':')
        ax2.plot(timestamps, cpu_freqs[i], label=f"CPU{i}", linestyle=linestyle)
    
    ax2.set_ylabel("CPU Freq (GHz)")
    ax2.grid(True)
    ax2.legend(loc="upper left", ncol=4, fontsize='small')

    # 3. GPU Freq
    ax3.plot(timestamps, gpu_freqs_mhz, 'g-', label="GPU Freq")
    ax3.set_ylabel("GPU Freq (MHz)")
    ax3.set_xlabel("Time")
    ax3.grid(True)
    ax3.legend(loc="upper left")

    # Format Date Axis
    # Use auto formatter
    fig.autofmt_xdate()

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output)
        print(f"Plot saved to {args.output}")
    else:
        # Check if we have a display
        if os.environ.get('DISPLAY'):
            plt.show()
        else:
            print("No DISPLAY detected. Saving to profile_plot.png instead.")
            plt.savefig("profile_plot.png")

if __name__ == "__main__":
    main()
