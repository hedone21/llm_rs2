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
        self.last_cpu_stats = None
        self.target_pid = None
        self.last_proc_cpu_stats = None # (utime+stime, timestamp)
        self.num_cores = len(self.get_cpu_freqs())
        print(f"[Monitor] Detected {self.num_cores} CPU cores for normalization.")
        
    def find_target_pid(self, process_name="generate"):
        try:
            # Simple pidof
            output = run_adb_command(f'shell "pidof {process_name}"', check=False)
            if output and output.strip().isdigit():
                self.target_pid = int(output.strip().split()[0]) # Take first if multiple
                print(f"[Monitor] Found target process '{process_name}': {self.target_pid}")
                return self.target_pid
        except: pass
        return None
        
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

    def get_cpu_usage(self):
        try:
            output = run_adb_command('shell "cat /proc/stat"')
            # cpu  2239 0 2125 102938 ...
            first_line = output.splitlines()[0]
            if not first_line.startswith('cpu '): return 0.0
            
            parts = [int(p) for p in first_line.split()[1:]]
            # user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice
            
            idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
            non_idle = parts[0] + parts[1] + parts[2] + (parts[5] if len(parts) > 5 else 0) + (parts[6] if len(parts) > 6 else 0) + (parts[7] if len(parts) > 7 else 0)
            total = idle + non_idle
            
            usage = 0.0
            if self.last_cpu_stats:
                prev_idle, prev_total = self.last_cpu_stats
                total_d = total - prev_total
                idle_d = idle - prev_idle
                if total_d > 0:
                    usage = (total_d - idle_d) / total_d * 100.0
            
            self.last_cpu_stats = (idle, total)
            return usage
        except:
             return 0.0

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
    def get_process_mem_usage(self):
        if not self.target_pid: return 0.0
        try:
            # VmRSS in /proc/[pid]/status or statm
            # statm: size resident shared text lib data dt
            # resident * 4KB (usually)
            output = run_adb_command(f'shell "cat /proc/{self.target_pid}/statm"', check=False)
            if output:
                parts = output.split()
                if len(parts) >= 2:
                    rss_pages = int(parts[1])
                    return (rss_pages * 4) / 1024.0 # MB (Assuming 4KB pages)
        except: pass
        return 0.0

    def get_process_cpu_usage(self):
        if not self.target_pid: return 0.0
        try:
            # /proc/[pid]/stat
            # 14 (utime), 15 (stime)
            output = run_adb_command(f'shell "cat /proc/{self.target_pid}/stat"', check=False)
            if not output: return 0.0
            
            parts = output.split()
            if len(parts) < 16: return 0.0
            
            # User + System time in clock ticks
            utime = int(parts[13])
            stime = int(parts[14])
            total_time = utime + stime
            
            now = time.time()
            usage = 0.0
            
            if self.last_proc_cpu_stats:
                prev_total, prev_time = self.last_proc_cpu_stats
                
                delta_ticks = total_time - prev_total
                delta_time = now - prev_time
                
                # We need SC_CLK_TCK. Usually 100 on Android/Linux arm64.
                # usage % = (delta_ticks / clk_tck) / delta_time * 100
                # Using 100 as heuristic for now. accurately finding it requires `getconf CLK_TCK`
                clk_tck = 100.0 
                
                if delta_time > 0:
                    usage = (delta_ticks / clk_tck) / delta_time * 100.0
                    # Normalize by num_cores to match system-wide % (0-100 total)
                    if self.num_cores > 0:
                        usage /= self.num_cores
            
            self.last_proc_cpu_stats = (total_time, now)
            return usage
        except: return 0.0
    def capture_snapshot(self):
        timestamp = datetime.now().isoformat()
        temp = self.get_temperature()
        cpu_freqs = self.get_cpu_freqs()
        gpu_freq = self.get_gpu_freq()
        mem_used = self.get_memory_info()
        gpu_load = self.get_gpu_load()
        cpu_load = self.get_cpu_usage()
        
        # Process Specific
        # Try to find PID if not found yet (it might start late)
        if not self.target_pid: self.find_target_pid()
        
        proc_mem = self.get_process_mem_usage()
        proc_cpu = self.get_process_cpu_usage()
        
        return {
            "timestamp": timestamp,
            "temp_c": temp,
            "gpu_freq_hz": gpu_freq,
            "cpu_freqs_khz": cpu_freqs,
            "mem_used_mb": mem_used,
            "gpu_load_percent": gpu_load,
            "cpu_load_percent": cpu_load,
            "process_mem_mb": proc_mem,
            "process_cpu_percent": proc_cpu
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

    def get_cpu_core_info(self):
        try:
            # Get max freqs for all cores
            output = run_adb_command('shell "cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq"')
            freqs = [int(f) for f in output.split()]
            if not freqs: return []
            
            # Simple clustering
            unique_freqs = sorted(list(set(freqs)))
            
            # Label based on rank
            # 1 cluster: All same
            # 2 clusters: Little, Big
            # 3 clusters: Little, Big, Gold
            
            mapping = {}
            if len(unique_freqs) == 1:
                mapping[unique_freqs[0]] = "Core"
            elif len(unique_freqs) == 2:
                mapping[unique_freqs[0]] = "Little"
                mapping[unique_freqs[1]] = "Big"
            elif len(unique_freqs) >= 3:
                mapping[unique_freqs[0]] = "Little"
                # If more than 3, map intermediates to Big, max to Gold
                for f in unique_freqs[1:-1]:
                    mapping[f] = "Big"
                mapping[unique_freqs[-1]] = "Gold"
            
            return [mapping.get(f, "Unknown") for f in freqs]
        except:
            return []

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

    # Extract eviction policy
    ev_match = re.search(r'--eviction-policy\s+([^\s]+)', cmd_str)
    meta['eviction_policy'] = ev_match.group(1) if ev_match else 'none'

    # Extract eviction window
    ew_match = re.search(r'--eviction-window\s+(\d+)', cmd_str)
    if ew_match: meta['eviction_window'] = int(ew_match.group(1))

    # Extract protected prefix
    pp_match = re.search(r'--protected-prefix\s+(\d+)', cmd_str)
    if pp_match: meta['protected_prefix'] = int(pp_match.group(1))

    # Extract memory threshold
    mt_match = re.search(r'--memory-threshold-mb\s+(\d+)', cmd_str)
    if mt_match: meta['memory_threshold_mb'] = int(mt_match.group(1))

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

    # Eviction: policy=sliding, window=1024, prefix=64, threshold=256MB
    ev_match = re.search(r'Eviction:\s*policy=([^,]+),\s*window=(\d+),\s*prefix=(\d+),\s*threshold=(\d+)MB', stdout)
    if ev_match:
        results['eviction_policy'] = ev_match.group(1)
        results['eviction_window'] = int(ev_match.group(2))
        results['eviction_prefix'] = int(ev_match.group(3))
        results['eviction_threshold_mb'] = int(ev_match.group(4))
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run ADB benchmark with profiling and JSON output.")
    parser.add_argument("--cmd", required=True, help="Command to run via ADB shell")
    parser.add_argument("--interval", type=float, default=1.0, help="Profiling interval in seconds")
    parser.add_argument("--output-dir", default=".", help="Directory to save JSON output")
    parser.add_argument("--name", help="Custom name for output file (optional)")
    parser.add_argument("--foreground-app", help="Foreground app context (e.g., 'YouTube', 'Chrome', 'idle')")
    
    args = parser.parse_args()

    # 1. Setup Monitor & Metadata
    monitor = DeviceMonitor(args.interval)
    metadata = extract_metadata(args.cmd)
    metadata['date'] = datetime.now().isoformat()
    metadata['command'] = args.cmd
    if args.foreground_app:
        metadata['foreground_app'] = args.foreground_app
    
    # 2. Capture Baseline
    baseline_stats = monitor.capture_baseline(duration=5.0)
    
    # 3. Running Benchmark & Real-time Parsing
    monitor.start()
    print(f"[Runner] Executing: {args.cmd}")
    
    events = []
    full_stdout = []
    return_code = 0
    start_t = time.time()
    
    try:
        full_cmd = f'adb shell "{args.cmd}"'
        proc = subprocess.Popen(
            full_cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=1, # Line buffered
            universal_newlines=True
        )
        
        # Read stdout line by line
        for line in proc.stdout:
            print(line, end='') # Echo output
            full_stdout.append(line)
            
            # Check for events
            # [Profile] Event: ModelLoadStart
            if "[Profile] Event:" in line:
                parts = line.strip().split(":", 1)
                if len(parts) == 2:
                    event_name = parts[1].strip()
                    events.append({
                        "name": event_name,
                        "timestamp": datetime.now().isoformat()
                    })

        proc.wait()
        return_code = proc.returncode
        duration = time.time() - start_t
        
        print(f"[Runner] Finished in {duration:.2f}s (Exit: {return_code})")
        
        if return_code != 0:
             # Stderr needs to be read separately if we care, 
             # but Popen consumes stdout. stderr might block? 
             # For simple usage, we just assume stderr isn't huge or let Popen handle buffering.
             stderr_out = proc.stderr.read()
             if stderr_out: print(f"[Error] Stderr: {stderr_out}")
            
    except KeyboardInterrupt:
        print("\n[Runner] Interrupted.")
        if 'proc' in locals(): proc.terminate()
        return_code = -1
    finally:
        monitor.stop()

    # 4. Parse Results & Construct JSON
    benchmark_results = {}
    if return_code == 0:
        stdout_str = "".join(full_stdout)
        benchmark_results = parse_results(stdout_str)
        print(f"[Results] TTFT: {benchmark_results.get('ttft_ms', 'N/A')} ms")
    
    cpu_models = monitor.get_cpu_core_info()
    metadata['cpu_models'] = cpu_models
    
    final_data = {
        "version": 1,
        "metadata": metadata,
        "baseline": baseline_stats,
        "benchmark_results": benchmark_results,
        "events": events,
        "timeseries": monitor.data_buffer
    }
    
    # 5. Save Output
    if args.name:
        filename = args.name
        if not filename.endswith('.json'): filename += ".json"
    else:
        # Auto-name: profile_{backend}_{prefill}_{tokens}_{eviction}_{date}.json
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        backend = metadata.get('backend', 'unknown')
        prefill = metadata.get('prefill_type', 'custom')
        tokens = metadata.get('num_tokens', '0')
        eviction = metadata.get('eviction_policy', 'none')
        filename = f"profile_{backend}_{prefill}_{tokens}_{eviction}_{date_str}.json"
    
    out_path = os.path.join(args.output_dir, filename)
    with open(out_path, 'w') as f:
        json.dump(final_data, f, indent=2)
        
    print(f"[Output] Saved profile to {out_path}")

if __name__ == "__main__":
    main()
