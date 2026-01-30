import os
import json
import glob
from datetime import datetime

BENCHMARK_DIR = "benchmarks"
DATA_DIR = os.path.join(BENCHMARK_DIR, "data")
PLOTS_DIR = os.path.join(BENCHMARK_DIR, "plots")
README_PATH = os.path.join(BENCHMARK_DIR, "README.md")

def parse_iso_time(t_str):
    try:
        return datetime.fromisoformat(t_str)
    except:
        return datetime.now()

def analyze_json(filepath):
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except:
            return None

    meta = data.get("metadata", {})
    results = data.get("benchmark_results", {})
    baseline = data.get("baseline", {})
    timeseries = data.get("timeseries", [])

    # Metrics
    ttft = results.get("ttft_ms", "N/A")
    tbt = results.get("tbt_ms", "N/A")
    tps = results.get("tokens_per_sec", "N/A")
    
    # Thermal Analysis
    start_temp = baseline.get("avg_start_temp_c", 0.0)
    end_temp = start_temp
    max_temp = start_temp
    throttling_suspected = False
    
    if timeseries:
        temps = [x.get("temp_c", 0) for x in timeseries]
        if temps:
            end_temp = temps[-1]
            max_temp = max(temps)
        
        # Check frequency drops if CPU
        # Simple heuristic: if frequency dropped by > 30% from max observed while temp > 30C
        # (This is just a placeholder logic, better done with specific rules)
        pass

    temp_str = f"{start_temp:.1f} -> {end_temp:.1f} (Max {max_temp:.1f})"
    
    # Formatting Date
    date_str = meta.get("date", "")[:16].replace("T", " ") # YYYY-MM-DD HH:MM
    
    # Filename link
    filename = os.path.basename(filepath)
    plot_filename = filename.replace('.json', '.png')
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    has_plot = os.path.exists(plot_path)
    
    return {
        "date": date_str,
        "filename": filename,
        "backend": meta.get("backend", "?"),
        "model": meta.get("model", "?"),
        "input": meta.get("prefill_type", "?"),
        "n_tokens": meta.get("num_tokens", "?"),
        "foreground_app": meta.get("foreground_app", "-"),
        "ttft": f"{ttft:.1f}" if isinstance(ttft, (int, float)) else ttft,
        "tbt": f"{tbt:.2f}" if isinstance(tbt, (int, float)) else tbt,
        "tps": f"{tps:.1f}" if isinstance(tps, (int, float)) else tps,
        "temp": temp_str,
        "mem": f"{baseline.get('avg_memory_used_mb', 0):.0f}",
        "plot_link": f"[Graph](plots/{plot_filename})" if has_plot else "-"
    }

def generate_markdown(records):
    # Sort by date descending
    records.sort(key=lambda x: x["date"], reverse=True)
    
    md = "# 📊 LLM Research Benchmark Log\n\n"
    md += f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    md += "## Executive Summary\n"
    md += "- **Total Benchmarks**: {}\n".format(len(records))
    md += "- **Recent Run**: {}\n\n".format(records[0]["date"] if records else "N/A")
    
    md += "## Detailed Results\n\n"
    md += "| Date | Model | Backend | Input | Tokens | FG App | TTFT (ms) | TBT (ms) | T/s | Temp (°C) | Mem (MB) | Data | Plot |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for r in records:
        link = f"[JSON](data/{r['filename']})"
        fg_app = r.get('foreground_app', '-')
        row = f"| {r['date']} | {r['model']} | {r['backend']} | {r['input']} | {r['n_tokens']} | {fg_app} | **{r['ttft']}** | {r['tbt']} | {r['tps']} | {r['temp']} | {r['mem']} | {link} | {r.get('plot_link', '-')} |\n"
        md += row
        
    md += "\n\n## Graphical Analysis\n"
    md += "Plots available in `benchmarks/plots/`. (Manually added below)\n"
    # Auto-add existing plots?
    # plots = glob.glob(os.path.join(PLOTS_DIR, "*.png"))
    # for p in plots:
    #     rel = os.path.relpath(p, BENCHMARK_DIR)
    #     md += f"![{rel}]({rel})\n"

    return md

def main():
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    records = []
    
    print(f"Found {len(json_files)} profiles.")
    for f in json_files:
        rec = analyze_json(f)
        if rec:
            records.append(rec)
            
    md_content = generate_markdown(records)
    
    with open(README_PATH, "w") as f:
        f.write(md_content)
        
    print(f"Generated {README_PATH}")

if __name__ == "__main__":
    main()
