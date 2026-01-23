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

    events = data.get("events", [])
    parsed_events = {}
    for e in events:
        parsed_events[e["name"]] = parse_time(e["timestamp"])
        
    cpu_models = meta.get("cpu_models", [])

    # Prepare Data
    timestamps = [parse_time(x.get("timestamp")) for x in timeseries]
    temps = [x.get("temp_c", 0) for x in timeseries]
    gpu_freqs_mhz = [x.get("gpu_freq_hz", 0) / 1e6 for x in timeseries]
    mem_used = [x.get("mem_used_mb", 0) for x in timeseries]
    gpu_load = [x.get("gpu_load_percent", 0) for x in timeseries]
    cpu_load = [x.get("cpu_load_percent", 0) for x in timeseries]
    
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
    fig, axes = plt.subplots(6, 1, figsize=(10, 18), sharex=True)
    (ax1, ax2, ax3, ax4, ax5, ax6) = axes
    
    # Title with Metadata
    title_str = (f"Benchmark Profile: {meta.get('model', 'Unknown')} ({meta.get('backend', 'Unknown')})\n"
                 f"Tokens: {meta.get('num_tokens', '?')} | TTFT: {results.get('ttft_ms', 'N/A')}ms | TBT: {results.get('tbt_ms', 'N/A')}ms")
    fig.suptitle(title_str, fontsize=14)

    # 1. Temperature
    ax1.plot(timestamps, temps, 'r-', label="Battery Temp")
    ax1.set_ylabel("Temp (°C)")
    ax1.grid(True)
    ax1.legend(loc="upper left")
    
    # 2. CPU Freqs
    # 2. CPU Freqs
    for i in range(num_cores):
        # Heuristic: Core 7 is usually prime, 4-6 big, 0-3 little
        core_label = f"CPU{i}"
        if i < len(cpu_models):
            core_label += f" ({cpu_models[i]})"
            
        linestyle = '-' if "Gold" in core_label else ('--' if "Big" in core_label else ':')
        # Fallback if no models detected
        if not cpu_models:
             linestyle = '-' if i >= num_cores - 1 else ('--' if i >= num_cores - 4 else ':')
             
        ax2.plot(timestamps, cpu_freqs[i], label=core_label, linestyle=linestyle)
    
    ax2.set_ylabel("CPU Freq (GHz)")
    ax2.grid(True)
    ax2.legend(loc="upper left", ncol=4, fontsize='small')

    # 3. GPU Freq
    ax3.plot(timestamps, gpu_freqs_mhz, 'g-', label="GPU Freq")
    ax3.set_ylabel("GPU Freq (MHz)")
    ax3.grid(True)
    ax3.legend(loc="upper left")
    
    # 4. Memory Usage
    ax4.plot(timestamps, mem_used, 'b-', label="Mem Used")
    ax4.set_ylabel("Memory (MB)")
    ax4.grid(True)
    ax4.legend(loc="upper left")
    
    # 5. CPU Load
    ax5.plot(timestamps, cpu_load, 'm-', label="CPU Load")
    ax5.set_ylabel("CPU Load (%)")
    ax5.set_ylim(0, 105)
    ax5.grid(True)
    ax5.legend(loc="upper left")
    
    # 6. GPU Load
    ax6.plot(timestamps, gpu_load, 'c-', label="GPU Load")
    ax6.set_ylabel("GPU Load (%)")
    ax6.set_xlabel("Time")
    ax6.set_ylim(0, 105)
    ax6.grid(True)
    ax6.legend(loc="upper left")

    # Format Date Axis
    # Use auto formatter
    fig.autofmt_xdate()

    # Add Event Markers to all plots
    axes_list = [ax1, ax2, ax3, ax4, ax5, ax6]
    
    # Define regions
    # ModelLoadStart -> PrefillStart (Load)
    # PrefillStart -> DecodingStart (Prefill)
    # DecodingStart -> End (Decode)
    
    regions = [
        ("ModelLoadStart", "PrefillStart", "Load", "gray", 0.1),
        ("PrefillStart", "DecodingStart", "Prefill", "orange", 0.1),
        ("DecodingStart", "End", "Decode", "green", 0.1),
    ]
    
    for start_evt, end_evt, label, color, alpha in regions:
        if start_evt in parsed_events and end_evt in parsed_events:
            start_t = parsed_events[start_evt]
            end_t = parsed_events[end_evt]
            # Convert to matplotlib date format if needed, but plot takes datetime objects directly usually
            for ax in axes_list:
                ax.axvspan(start_t, end_t, color=color, alpha=alpha, label=label if ax == ax1 else "")
                
    # Re-legend ax1 to include regions
    handles, labels = ax1.get_legend_handles_labels()
    # Deduplicate
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper left")

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
