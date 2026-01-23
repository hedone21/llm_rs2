import argparse
import subprocess
import time
import threading
import json
import re
import signal
import sys
import os
from datetime import datetime

def run_adb_command(command, check=True):
    """Runs an adb command safely."""
    result = subprocess.run(
        f"adb {command}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout.strip()

class DeviceMonitor:
    def __init__(self, interval):
        self.interval = interval
        self.stop_event = threading.Event()
        self.data_buffer = []
        self.gpu_path = self.find_gpu_clk_path()
        self.gpu_load_path = self.find_gpu_load_path()
        
    def find_gpu_clk_path(self):
        paths = [
            "/sys/class/kgsl/kgsl-3d0/gpuclk",
            "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq",
            "/sys/devices/platform/1c00000.mali/devfreq/devfreq0/cur_freq",
            "/sys/kernel/gpu/gpu_clock",
        ]
        for p in paths:
            ret = run_adb_command(f'shell "ls {p} 2>/dev/null"', check=False)
            if ret and p in ret:
                return p
        return None

    def find_gpu_load_path(self):
         # Helper to find load/utilization
        paths = [
            "/sys/class/kgsl/kgsl-3d0/gpubusy", # Qualcomm
            "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",
            "/sys/devices/platform/1c00000.mali/gpu_utilization",
        ]
        for p in paths:
            ret = run_adb_command(f'shell "ls {p} 2>/dev/null"', check=False)
            if ret and p in ret:
                return p
        return None

    def get_cpu_freqs(self):
        try:
            cmd = 'shell "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null"'
            output = run_adb_command(cmd)
            freqs = [int(f) for f in output.split()]
            return freqs
        except:
            return []

    def get_gpu_freq(self):
        if not self.gpu_path: return 0
        try:
            output = run_adb_command(f'shell "cat {self.gpu_path}"')
            return int(output) if output.isdigit() else 0
        except: return 0
    
    def get_gpu_load(self):
        if not self.gpu_load_path: return 0.0
        try:
            output = run_adb_command(f'shell "cat {self.gpu_load_path}"')
            # Qualcomm gpubusy: "123 456" -> busy_cycles total_cycles
            parts = output.split()
            if len(parts) >= 2:
                 busy = int(parts[0])
                 total = int(parts[1])
                 return (busy / total) * 100.0 if total > 0 else 0.0
            elif len(parts) == 1 and parts[0].replace('%','').isdigit():
                 return float(parts[0].replace('%',''))
        except: pass
        return 0.0

    def get_temperature(self):
        try:
            output = run_adb_command('shell "dumpsys battery | grep temperature"')
            match = re.search(r'temperature:\s*(\d+)', output)
            if match:
                return float(match.group(1)) / 10.0
        except: pass
        return 0.0
        
    def get_memory_info(self):
        try:
            # simple free mem
            output = run_adb_command('shell "cat /proc/meminfo"')
            mem_total = 0
            mem_free = 0
            mem_avail = 0
            for line in output.splitlines():
                if "MemTotal" in line: mem_total = int(line.split()[1])
                if "MemFree" in line: mem_free = int(line.split()[1])
                if "MemAvailable" in line: mem_avail = int(line.split()[1])
            
            used = mem_total - (mem_avail if mem_avail > 0 else mem_free)
            return used / 1024.0 # MB
        except: return 0.0

    def capture_snapshot(self):
        timestamp = datetime.now().isoformat()
        temp = self.get_temperature()
        cpu_freqs = self.get_cpu_freqs()
        gpu_freq = self.get_gpu_freq()
        return {
            "timestamp": timestamp,
            "temp_c": temp,
            "gpu_freq_hz": gpu_freq,
            "cpu_freqs_khz": cpu_freqs
        }

    def capture_baseline(self, duration=5.0):
        print(f"[Monitor] Collecting baseline stats for {duration}s...")
        samples = []
        end_time = time.time() + duration
        while time.time() < end_time:
            samples.append({
                "mem_used_mb": self.get_memory_info(),
                "gpu_load": self.get_gpu_load(),
                "temp": self.get_temperature()
            })
            time.sleep(1.0)
            
        # Average
        if not samples: return {}
        
        avg_mem = sum(s["mem_used_mb"] for s in samples) / len(samples)
        avg_gpu = sum(s["gpu_load"] for s in samples) / len(samples)
        avg_temp = sum(s["temp"] for s in samples) / len(samples)
        
        return {
            "avg_memory_used_mb": round(avg_mem, 2),
            "avg_gpu_load_percent": round(avg_gpu, 2),
            "avg_start_temp_c": round(avg_temp, 2)
        }

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.monitor_loop)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if hasattr(self, 'thread'):
            self.thread.join()

    def monitor_loop(self):
        while not self.stop_event.is_set():
            self.data_buffer.append(self.capture_snapshot())
            self.stop_event.wait(self.interval)

def extract_metadata(cmd_str):
    meta = {}
    
    # Extract model
    m_match = re.search(r'--model-path\s+([^\s]+)', cmd_str)
    if m_match: meta['model'] = os.path.basename(m_match.group(1))
    
    # Extract tokens
    n_match = re.search(r'(-n|--num-tokens)\s+(\d+)', cmd_str)
    if n_match: meta['num_tokens'] = int(n_match.group(2))
    
    # Extract backend
    b_match = re.search(r'-b\s+([^\s]+)', cmd_str)
    if b_match: meta['backend'] = b_match.group(1)
    
    # Extract prompt (heuristic)
    p_match = re.search(r'--prompt-file\s+([^\s]+)', cmd_str)
    if p_match: 
        meta['prefill_type'] = os.path.basename(p_match.group(1)).replace('.txt', '')
    else:
        meta['prefill_type'] = "custom_prompt"
        
    return meta

def parse_results(stdout):
    results = {}
    # TTFT: 156.48 ms
    ttft = re.search(r'TTFT:\s*([\d\.]+)\s*ms', stdout)
    if ttft: results['ttft_ms'] = float(ttft.group(1))
    
    # Avg TBT: 42.94 ms (23.3 tokens/sec)
    tbt = re.search(r'Avg TBT:\s*([\d\.]+)\s*ms', stdout)
    if tbt: results['tbt_ms'] = float(tbt.group(1))
    
    tps = re.search(r'\(([\d\.]+)\s*tokens/sec\)', stdout)
    if tps: results['tokens_per_sec'] = float(tps.group(1))
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run ADB benchmark with profiling and JSON output.")
    parser.add_argument("--cmd", required=True, help="Command to run via ADB shell")
    parser.add_argument("--interval", type=float, default=1.0, help="Profiling interval in seconds")
    parser.add_argument("--output-dir", default=".", help="Directory to save JSON output")
    parser.add_argument("--name", help="Custom name for output file (optional)")
    
    args = parser.parse_args()

    # 1. Setup Monitor & Metadata
    monitor = DeviceMonitor(args.interval)
    metadata = extract_metadata(args.cmd)
    metadata['date'] = datetime.now().isoformat()
    metadata['command'] = args.cmd
    
    # 2. Capture Baseline
    baseline_stats = monitor.capture_baseline(duration=5.0)
    
    # 3. Running Benchmark
    monitor.start()
    print(f"[Runner] Executing: {args.cmd}")
    
    try:
        start_t = time.time()
        full_cmd = f'adb shell "{args.cmd}"'
        proc = subprocess.run(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = time.time() - start_t
        
        print(f"[Runner] Finished in {duration:.2f}s (Exit: {proc.returncode})")
        if proc.returncode != 0:
            print(f"[Error] Stderr: {proc.stderr}")
            
    except KeyboardInterrupt:
        print("\n[Runner] Interrupted.")
        proc = None
    finally:
        monitor.stop()

    # 4. Parse Results & Construct JSON
    benchmark_results = {}
    if proc and proc.returncode == 0:
        benchmark_results = parse_results(proc.stdout)
        print(f"[Results] TTFT: {benchmark_results.get('ttft_ms', 'N/A')} ms")
    
    final_data = {
        "version": 1,
        "metadata": metadata,
        "baseline": baseline_stats,
        "benchmark_results": benchmark_results,
        "timeseries": monitor.data_buffer
    }
    
    # 5. Save Output
    if args.name:
        filename = args.name
        if not filename.endswith('.json'): filename += ".json"
    else:
        # Auto-name: profile_{backend}_{prefill}_{tokens}_{date}.json
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        backend = metadata.get('backend', 'unknown')
        prefill = metadata.get('prefill_type', 'custom')
        tokens = metadata.get('num_tokens', '0')
        filename = f"profile_{backend}_{prefill}_{tokens}_{date_str}.json"
    
    out_path = os.path.join(args.output_dir, filename)
    with open(out_path, 'w') as f:
        json.dump(final_data, f, indent=2)
        
    print(f"[Output] Saved profile to {out_path}")

if __name__ == "__main__":
    main()
