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

def run_local_command(command, check=True):
    """Runs a local shell command safely."""
    result = subprocess.run(
        command,
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
        self.target_pid = None
        self.last_cpu_stats = None
        self.last_proc_cpu_stats = None
        self.num_cores = os.cpu_count() or 1
        self.temp_path = self.find_thermal_path()
        print(f"[Monitor] Detected {self.num_cores} CPU cores for normalization.")
        
    def find_target_pid(self, process_name="generate"):
        try:
            output = run_local_command(f'pidof {process_name}', check=False)
            if output and output.strip().isdigit():
                self.target_pid = int(output.strip().split()[0])
                return self.target_pid
        except: pass
        return None
        
    def get_cpu_freqs(self):
        try:
            output = run_local_command('cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null')
            freqs = [int(f) for f in output.split()]
            return freqs
        except:
            return []

    def get_cpu_usage(self):
        try:
            output = run_local_command('cat /proc/stat')
            first_line = output.splitlines()[0]
            if not first_line.startswith('cpu '): return 0.0
            
            parts = [int(p) for p in first_line.split()[1:]]
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
        return 0
    
    def get_gpu_load(self):
        return 0.0

    def find_thermal_path(self):
        """Scans /sys/class/thermal/ for the most relevant thermal zone."""
        # Priorities for PC/Embedded Linux:
        # 1. x86_pkg_temp / coretemp (Intel/AMD Package)
        # 2. cpu-thermal (Raspberry Pi / BCM)
        # 3. k10temp (AMD)
        # 4. Fallback to thermal_zone0
        
        target_types = ['x86_pkg_temp', 'coretemp', 'cpu-thermal', 'k10temp']
        base_dir = "/sys/class/thermal"
        fallback = f"{base_dir}/thermal_zone0/temp"
        
        try:
            if not os.path.exists(base_dir):
                return None
            
            for tz in sorted(os.listdir(base_dir)):
                if not tz.startswith("thermal_zone"): continue
                
                type_path = os.path.join(base_dir, tz, "type")
                if os.path.exists(type_path):
                    with open(type_path, 'r') as f:
                        tz_type = f.read().strip()
                        
                        if any(t in tz_type for t in target_types):
                            print(f"[Monitor] Matched thermal zone '{tz_type}' at {tz}")
                            return os.path.join(base_dir, tz, "temp")
                            
            if os.path.exists(fallback):
                print(f"[Monitor] Using fallback thermal zone at thermal_zone0")
                return fallback
        except: pass
        return None

    def get_temperature(self):
        if not self.temp_path: return 0.0
        try:
            with open(self.temp_path, 'r') as f:
                val = f.read().strip()
                if val.isdigit():
                    return float(val) / 1000.0
        except: pass
        return 0.0
        
    def get_memory_info(self):
        try:
            output = run_local_command('cat /proc/meminfo')
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
            output = run_local_command(f'cat /proc/{self.target_pid}/statm', check=False)
            if output:
                parts = output.split()
                if len(parts) >= 2:
                    rss_pages = int(parts[1])
                    return (rss_pages * 4) / 1024.0
        except: pass
        return 0.0

    def get_process_cpu_usage(self):
        if not self.target_pid: return 0.0
        try:
            output = run_local_command(f'cat /proc/{self.target_pid}/stat', check=False)
            if not output: return 0.0
            
            parts = output.split()
            if len(parts) < 16: return 0.0
            
            utime = int(parts[13])
            stime = int(parts[14])
            total_time = utime + stime
            now = time.time()
            usage = 0.0
            
            if self.last_proc_cpu_stats:
                prev_total, prev_time = self.last_proc_cpu_stats
                delta_ticks = total_time - prev_total
                delta_time = now - prev_time
                clk_tck = 100.0 # Standard linux
                if delta_time > 0:
                    usage = (delta_ticks / clk_tck) / delta_time * 100.0
                    if self.num_cores > 0: usage /= self.num_cores
            
            self.last_proc_cpu_stats = (total_time, now)
            return usage
        except: return 0.0

    def capture_snapshot(self):
        timestamp = datetime.now().isoformat()
        if not self.target_pid: self.find_target_pid()
        
        return {
            "timestamp": timestamp,
            "temp_c": self.get_temperature(),
            "gpu_freq_hz": self.get_gpu_freq(),
            "cpu_freqs_khz": self.get_cpu_freqs(),
            "mem_used_mb": self.get_memory_info(),
            "gpu_load_percent": self.get_gpu_load(),
            "cpu_load_percent": self.get_cpu_usage(),
            "process_mem_mb": self.get_process_mem_usage(),
            "process_cpu_percent": self.get_process_cpu_usage()
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
        return ["PC_Core"] * self.num_cores

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
    m_match = re.search(r'--model-path\s+([^\s]+)', cmd_str)
    if m_match: meta['model'] = os.path.basename(m_match.group(1))
    
    n_match = re.search(r'(-n|--num-tokens)\s+(\d+)', cmd_str)
    if n_match: meta['num_tokens'] = int(n_match.group(2))
    
    b_match = re.search(r'-b\s+([^\s]+)', cmd_str)
    if b_match: meta['backend'] = b_match.group(1)
    
    p_match = re.search(r'--prompt-file\s+([^\s]+)', cmd_str)
    if p_match: meta['prefill_type'] = os.path.basename(p_match.group(1)).replace('.txt', '')
    else: meta['prefill_type'] = "custom_prompt"
    
    ev_match = re.search(r'--eviction-policy\s+([^\s]+)', cmd_str)
    meta['eviction_policy'] = ev_match.group(1) if ev_match else 'none'
    return meta

def parse_results(stdout):
    results = {}
    ttft = re.search(r'TTFT:\s*([\d\.]+)\s*ms', stdout)
    if ttft: results['ttft_ms'] = float(ttft.group(1))
    
    tbt = re.search(r'Avg TBT:\s*([\d\.]+)\s*ms', stdout)
    if tbt: results['tbt_ms'] = float(tbt.group(1))
    
    tps = re.search(r'\(([\d\.]+)\s*tokens/sec\)', stdout)
    if tps: results['tokens_per_sec'] = float(tps.group(1))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--output-name", help="Specific filename")
    parser.add_argument("--foreground-app")
    args = parser.parse_args()

    monitor = DeviceMonitor(args.interval)
    metadata = extract_metadata(args.cmd)
    metadata['date'] = datetime.now().isoformat()
    metadata['command'] = args.cmd
    if args.foreground_app: metadata['foreground_app'] = args.foreground_app
    
    baseline_stats = monitor.capture_baseline(duration=2.0)
    monitor.start()
    
    events = []
    full_stdout = []
    return_code = 0
    start_t = time.time()
    
    try:
        proc = subprocess.Popen(
            args.cmd, 
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True
        )
        for line in proc.stdout:
            print(line, end='')
            full_stdout.append(line)
            if "[Profile] Event:" in line:
                parts = line.strip().split(":", 1)
                if len(parts) == 2:
                    events.append({"name": parts[1].strip(), "timestamp": datetime.now().isoformat()})
        proc.wait()
        return_code = proc.returncode
        duration = time.time() - start_t
        if return_code != 0:
             stderr_out = proc.stderr.read()
             if stderr_out: print(f"[Error] Stderr: {stderr_out}")
    except KeyboardInterrupt:
        if 'proc' in locals(): proc.terminate()
        return_code = -1
    finally:
        monitor.stop()

    benchmark_results = {}
    if return_code == 0:
        stdout_str = "".join(full_stdout)
        benchmark_results = parse_results(stdout_str)
    
    metadata['cpu_models'] = monitor.get_cpu_core_info()
    final_data = {
        "version": 1, "metadata": metadata, "baseline": baseline_stats,
        "benchmark_results": benchmark_results, "events": events, "timeseries": monitor.data_buffer
    }
    
    if args.output_name:
        filename = args.output_name if args.output_name.endswith('.json') else args.output_name + ".json"
    else:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        backend = metadata.get('backend', 'unknown')
        prefill = metadata.get('prefill_type', 'custom')
        tokens = metadata.get('num_tokens', '0')
        eviction = metadata.get('eviction_policy', 'none')
        filename = f"profile_local_{backend}_{prefill}_{tokens}_{eviction}_{date_str}.json"
    
    out_path = os.path.join(args.output_dir, filename)
    with open(out_path, 'w') as f:
        json.dump(final_data, f, indent=2)
    print(f"[Output] Saved profile to {out_path}")

if __name__ == "__main__":
    main()
