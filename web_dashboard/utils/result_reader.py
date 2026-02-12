import os
import json
import glob
from typing import List, Dict, Optional

from datetime import datetime

RESULTS_DIR = "results/data"
PLOTS_DIR = "results/plots"

def calculate_metrics(data: Dict) -> Dict:
    """
    Calculates TTFT, TBT, TPS from events if missing in benchmark_results.
    """
    results = data.get("benchmark_results", {})
    events = data.get("events", [])
    meta = data.get("metadata", {})
    
    # helper to find timestamp
    def get_ts(name):
        for e in events:
            if e["name"] == name:
                return datetime.fromisoformat(e["timestamp"])
        return None

    ttft = results.get("ttft_ms", "N/A")
    tbt = results.get("tbt_ms", "N/A")
    tps = results.get("tokens_per_sec", "N/A")
    
    # Try to calculate if missing
    if (ttft == "N/A" or tbt == "N/A") and events:
        prefill_start = get_ts("PrefillStart")
        decoding_start = get_ts("DecodingStart")
        end = get_ts("End")
        
        num_tokens = int(meta.get("num_tokens", 0))
        
        if prefill_start and decoding_start:
            ttft_val = (decoding_start - prefill_start).total_seconds() * 1000
            if ttft == "N/A":
                ttft = round(ttft_val, 2)
                
        if decoding_start and end and num_tokens > 1:
            duration = (end - decoding_start).total_seconds()
            # Decoding includes the first token? Usually Prefill produces first token, 
            # so Decoding generates (N-1) tokens.
            # But let's assume num_tokens is total generated.
            # If prefill is prompt only, then generation is num_tokens.
            # Let's stick to simple logic: Duration / (N-1) if N>1
            
            # Using typically accepted definition:
            # TTFT: Prompt processing time (until first token ready)
            # TBT: Time between subsequent tokens
            
            gen_tokens = num_tokens - 1 if num_tokens > 1 else 1 # Avoid div by zero
            
            tbt_val = (duration * 1000) / gen_tokens
            if tbt == "N/A":
                tbt = round(tbt_val, 2)
                
            tps_val = gen_tokens / duration if duration > 0 else 0
            if tps == "N/A":
                tps = round(tps_val, 2)
                
    return {
        "ttft_ms": ttft,
        "tbt_ms": tbt,
        "tokens_per_sec": tps
    }

def get_all_runs() -> List[Dict]:
    """
    Scans results/data/*.json and returns a summary list.
    """
    json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    runs = []
    
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            meta = data.get("metadata", {})
            # results = data.get("benchmark_results", {}) # Removed, use calculated
            calc_res = calculate_metrics(data)
            
            # Basic summary info
            runs.append({
                "filename": os.path.basename(filepath),
                "date": meta.get("date", ""),
                "model": meta.get("model", "?"),
                "backend": meta.get("backend", "?"),
                "input_type": meta.get("prefill_type", "?"),
                "num_tokens": meta.get("num_tokens", "?"),
                "ttft_ms": calc_res["ttft_ms"],
                "tbt_ms": calc_res["tbt_ms"],
                "tokens_per_sec": calc_res["tokens_per_sec"],
            })
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            
    # Sort by date descending
    runs.sort(key=lambda x: x["date"], reverse=True)
    return runs

def get_run_detail(filename: str) -> Optional[Dict]:
    """
    Returns the full JSON content for a specific run.
    """
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        return None
        
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Enrich with calculated metrics if missing
        calc_res = calculate_metrics(data)
        if "benchmark_results" not in data:
            data["benchmark_results"] = {}
        
        # Merge only if missing
        for k, v in calc_res.items():
            if data["benchmark_results"].get(k, "N/A") == "N/A":
                data["benchmark_results"][k] = v
            
        # Check for associated plots
        plot_filename = filename.replace(".json", ".png")
        if os.path.exists(os.path.join(PLOTS_DIR, plot_filename)):
            data["plot_url"] = f"/plots/{plot_filename}"
            
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
