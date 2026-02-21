import re
import statistics
from collections import defaultdict

def parse_readme(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Find the table
    table_match = re.search(r'\| Date \|.*?\n\| :--- \|.*?\n((?:\|.*\|\n?)+)', content)
    if not table_match:
        print("No table found")
        return []

    table_content = table_match.group(1)
    rows = []
    for line in table_content.strip().split('\n'):
        cols = [c.strip() for c in line.split('|')[1:-1]]
        if len(cols) < 13: continue
        
        # Parse columns
        date, model, backend, device, eviction, input_type, tokens, fg_app, ttft_str, tbt_str, ts_str, temp_str, mem_str = cols[:13]
        
        # Clean numeric values
        ttft = parse_float(ttft_str)
        tbt = parse_float(tbt_str)
        
        # Parse Max Temp
        max_temp = None
        temp_match = re.search(r'Max ([\d\.]+)', temp_str)
        if temp_match:
            max_temp = float(temp_match.group(1))
            
        rows.append({
            'backend': backend,
            'device': device,
            'input': input_type,
            'tokens': tokens,
            'ttft': ttft,
            'tbt': tbt,
            'max_temp': max_temp
        })
    return rows

def parse_float(s):
    # Remove markdown bold/etc and handling N/A
    clean = re.sub(r'[^\d\.]', '', s)
    if not clean: return None
    return float(clean)

def analyze(rows):
    # Group by (Device, Input, Tokens)
    groups = defaultdict(lambda: {'cpu': [], 'opencl': []})
    
    for row in rows:
        if row['backend'] not in ['cpu', 'opencl']: continue
        key = (row['device'], row['input'], row['tokens'])
        groups[key][row['backend']].append(row)

    print("# Benchmark Analysis ReportBy Device, Input & Tokens\n")
    
    print("## 1. Prefill (TTFT) Comparison\n")
    print("| Device | Input | Tokens | CPU Avg (ms) | OpenCL Avg (ms) | Speedup (CL vs CPU) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for (dev, inp, tok), runs in sorted(groups.items()):
        cpu_ttfts = [r['ttft'] for r in runs['cpu'] if r['ttft'] is not None]
        cl_ttfts = [r['ttft'] for r in runs['opencl'] if r['ttft'] is not None]
        
        if not cpu_ttfts and not cl_ttfts: continue

        cpu_avg = statistics.mean(cpu_ttfts) if cpu_ttfts else None
        cl_avg = statistics.mean(cl_ttfts) if cl_ttfts else None
        
        cpu_str = f"{cpu_avg:.1f}" if cpu_avg else "-"
        cl_str = f"{cl_avg:.1f}" if cl_avg else "-"
        
        speedup = "-"
        if cpu_avg and cl_avg:
            # Lower TTFT is better. Speedup = CPU / CL
            ratio = cpu_avg / cl_avg
            speedup = f"{ratio:.2f}x"
            
        print(f"| {dev} | {inp} | {tok} | {cpu_str} | {cl_str} | {speedup} |")

    print("\n## 2. Decode (TBT) Comparison\n")
    print("| Device | Input | Tokens | CPU Avg (ms) | OpenCL Avg (ms) | Speedup (CL vs CPU) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for (dev, inp, tok), runs in sorted(groups.items()):
        cpu_tbts = [r['tbt'] for r in runs['cpu'] if r['tbt'] is not None]
        cl_tbts = [r['tbt'] for r in runs['opencl'] if r['tbt'] is not None]
        
        if not cpu_tbts and not cl_tbts: continue
        
        cpu_avg = statistics.mean(cpu_tbts) if cpu_tbts else None
        cl_avg = statistics.mean(cl_tbts) if cl_tbts else None
        
        cpu_str = f"{cpu_avg:.2f}" if cpu_avg else "-"
        cl_str = f"{cl_avg:.2f}" if cl_avg else "-"
        
        speedup = "-"
        if cpu_avg and cl_avg:
             # Lower TBT is better. Speedup = CPU / CL
            ratio = cpu_avg / cl_avg
            speedup = f"{ratio:.2f}x"
            
        print(f"| {dev} | {inp} | {tok} | {cpu_str} | {cl_str} | {speedup} |")

    print("\n## 3. Thermal Analysis (Max Temp)\n")
    print("| Device | Input | Tokens | CPU Max Avg (°C) | OpenCL Max Avg (°C) | Diff (CL - CPU) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for (dev, inp, tok), runs in sorted(groups.items()):
        cpu_temps = [r['max_temp'] for r in runs['cpu'] if r['max_temp'] is not None]
        cl_temps = [r['max_temp'] for r in runs['opencl'] if r['max_temp'] is not None]
        
        if not cpu_temps and not cl_temps: continue

        # Filter out invalid temps (e.g. 0.0)
        cpu_temps = [t for t in cpu_temps if t > 0]
        cl_temps = [t for t in cl_temps if t > 0]
        
        cpu_avg = statistics.mean(cpu_temps) if cpu_temps else None
        cl_avg = statistics.mean(cl_temps) if cl_temps else None
        
        cpu_str = f"{cpu_avg:.1f}" if cpu_avg else "-"
        cl_str = f"{cl_avg:.1f}" if cl_avg else "-"
        
        diff = "-"
        if cpu_avg and cl_avg:
            d = cl_avg - cpu_avg
            diff = f"{d:+.1f}"
            
        print(f"| {dev} | {inp} | {tok} | {cpu_str} | {cl_str} | {diff} |")
    
    analyze_thermal_impact(rows)

def analyze_thermal_impact(rows):
    print("\n# 온도에 따른 성능 변화 분석 (Thermal Impact Analysis)\n")
    print("## 온도 vs 디코딩 속도 (TBT) 상관관계 분석\n")
    print("같은 설정(Device, Backend, Input, Tokens)에서 온도가 성능에 미치는 영향을 분석합니다.")
    print("**(참고: 상관계수(Correlation) 양수(+) = 온도가 높을수록 TBT 증가/성능 저하)**\n")
    
    print("| 설정 (Device/Backend/Input/Tokens) | 데이터 수 | 온도 범위 (°C) | 상관계수 (Temp vs TBT) | 분석 |")
    print("| :--- | :--- | :--- | :--- | :--- |")

    # Group by (Device, Backend, Input, Tokens)
    groups = defaultdict(list)
    for row in rows:
        if row['tbt'] is not None and row['max_temp'] is not None and row['max_temp'] > 0:
            key = (row['device'], row['backend'], row['input'], row['tokens'])
            groups[key].append(row)
            
    for key, group in groups.items():
        if len(group) < 3: continue # Need at least 3 points for meaningful trend
        
        device, backend, input_type, tokens = key
        
        temps = [r['max_temp'] for r in group]
        tbts = [r['tbt'] for r in group]
        
        # Calculate correlation
        if len(set(temps)) > 1 and len(set(tbts)) > 1:
            try:
                mean_t = statistics.mean(temps)
                mean_p = statistics.mean(tbts)
                num = sum((t - mean_t) * (p - mean_p) for t, p in zip(temps, tbts))
                den = (sum((t - mean_t)**2 for t in temps) * sum((p - mean_p)**2 for p in tbts)) ** 0.5
                corr = num / den if den != 0 else 0
            except:
                corr = 0
        else:
            corr = 0
            
        temp_range = f"{min(temps):.1f}~{max(temps):.1f}"
        
        analysis = "-"
        if corr > 0.5:
            analysis = "🚨 성능 저하 뚜렷함 (Throttling)"
        elif corr > 0.3:
            analysis = "⚠️ 약한 성능 저하"
        elif corr < -0.3:
            analysis = "❓ 역상관 (온도가 높은데 빠름?)"
        else:
            analysis = "변화 미미함"

        label = f"{device} / {backend} / {input_type} / {tokens}"
        print(f"| {label} | {len(group)} | {temp_range} | {corr:.2f} | {analysis} |")

    # Detailed view for specific cases
    print("\n## 상세 데이터 (Top Variances)\n")
    print("| 설정 | 온도 (°C) | TBT (ms) |")
    print("| :--- | :--- | :--- |")
    
    # Sort groups by variance in TBT to find interesting ones
    sorted_groups = sorted(groups.items(), key=lambda x: statistics.stdev([r['tbt'] for r in x[1]]) if len(x[1])>1 else 0, reverse=True)
    
    for i, (key, group) in enumerate(sorted_groups[:5]):
        device, backend, input_type, tokens = key
        sorted_runs = sorted(group, key=lambda x: x['max_temp'])
        for run in sorted_runs:
            print(f"| {device}/{backend}/{input_type}/{tokens} | {run['max_temp']} | {run['tbt']} |")
        print("| - | - | - |") # Separator

if __name__ == "__main__":
    data = parse_readme("results/README.md")
    analyze(data)
